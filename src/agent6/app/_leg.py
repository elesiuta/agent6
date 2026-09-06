# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The one leg body a fresh run and a resumed leg share.

From the provider session to the end block: prompt revision, providers, the
gate step (per lifecycle: a fresh leg infers, a resumed one reuses the
snapshot's), steer state, the session network and MCP servers, the tool set,
the Workflow, its teardown, the auto-merge, and the end report. `app/run.py`
and `app/resume.py` keep only what differs before it (id, manifest, dirty
tree, snapshot, guards) and after it (the stash, the locks) and hand the leg
its `LegInputs`. One body, so a knob wired into one lifecycle cannot be
missing from the other.
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from agent6.app._session import (
    build_session_providers,
    build_session_tools,
    session_facts_provider,
    tool_result_cap_chars,
)
from agent6.app._setup import (
    BudgetOverrides,
    SandboxOverrides,
    start_mcp_manager_if_enabled,
    wants_session_network,
)
from agent6.app.finalize import (
    auto_merge_eligible,
    finalize_auto_merge,
    fire_notify_hook,
    print_interrupt_end,
    print_session_end,
    session_exit_code,
    stranded_edits,
)
from agent6.app.frontend import SessionFrontend, approval_scopes
from agent6.app.providers import build_prompt_reviser_provider, close_provider, role_temperature
from agent6.app.reporter import Reporter
from agent6.budget import BudgetTracker
from agent6.config import Config, RoleName
from agent6.events import EventSink, EventWriteError
from agent6.git_ops import chain_ref_for, render_commit_trailer
from agent6.paths import chown_to_real_user
from agent6.providers import TranscriptSink
from agent6.sandbox.jail import SessionNetwork
from agent6.sessions.ipc import (
    COMMAND_SCOPE,
    clear_compact_request,
    clear_session_netns_pid,
    clear_stop_request,
    read_compact_request,
    session_allow_set,
    stop_request_pending,
    write_session_netns_pid,
)
from agent6.sessions.layout import SessionLayout
from agent6.tools.dispatch import ToolDispatcher
from agent6.tools.operator_prompts import OperatorPrompts
from agent6.types import AutoCommitDirective, IsolationLevel, ResumableMode
from agent6.workflows._session_state import SessionEndReason
from agent6.workflows.loop import ResumeError, SessionResult, Workflow


@dataclass(frozen=True, slots=True)
class LegInputs:
    """What a fresh leg and a resumed one hand the leg body differently.
    Everything else the body derives itself."""

    session_id: str
    mode: ResumableMode
    role: RoleName
    isolation: IsolationLevel
    tui_enabled: bool
    interactive: bool
    # A fresh leg drives `wf.run(task)` (or the ask REPL); a resumed leg,
    # `task=None`, drives `wf.resume()`.
    task: str | None
    # The gate step, per lifecycle: a fresh leg infers one from the repo, a
    # resumed one reuses the snapshot's; both drop an unrunnable gate and pin
    # the result. Runs once the budget exists (inference may call a model).
    gate: Callable[[Config, BudgetTracker], Config]
    # The chain the leg's commits advance, and the base the review panel and
    # an unborn chain start from ("" when the repo had no head).
    chain_branch: str | None
    base_sha: str
    untracked_at_start: frozenset[str]
    resume_state_path: Path
    # `/undo` from the composer or the pause menu: forks back before the last
    # message and rewinds the checkout; the leg body records the outcome for
    # its end block.
    undo_forker: Callable[[], tuple[str, str] | None]
    # The leg's one gate to the operator: built by the lifecycle, so a question
    # it asks before the loop (the dirty-tree start question) and the
    # dispatcher's approvals share one journal and one id sequence.
    prompts: OperatorPrompts
    # A one-shot ask records its answer under this question; None when the
    # REPL already saved each turn.
    ask_transcript_task: str | None
    # The `--max-usd` / `--auto-approve` a `/parallel` lane inherits.
    budget_overrides: BudgetOverrides | None = None
    sandbox_overrides: SandboxOverrides | None = None
    standing_goal: str = ""
    pins: tuple[str, ...] = ()
    resuming: bool = False
    # A fork leg's checkout is a linked worktree: the repository git dir agent6
    # recorded for it, the one grant its jail makes beyond the workspace.
    worktree_git_dir: Path | None = None


@dataclass(frozen=True, slots=True)
class LegEnd:
    """How the leg ended: the process exit code, and whether the operator
    detached (the caller then releases its locks and spawns the continuation)."""

    rc: int
    detach_requested: bool = False


def detach_to_background(
    *,
    frontend: SessionFrontend,
    cfg: Config,
    layout: SessionLayout,
    cwd: Path,
    flags: Sequence[str],
    reporter: Reporter,
) -> None:
    """Hand a detached leg to a background `resume` under this invocation's
    *flags*, once the caller has released the run's locks: first ask how
    approvals are answered while nothing watches (`run_commands = "ask"` with
    no session-wide grant), then spawn, then print the reattach line, so
    "continues in the background" is said only of a spawn that happened."""
    if cfg.sandbox.run_commands == "ask" and not session_allow_set(
        layout.session_dir, COMMAND_SCOPE
    ):
        frontend.prompt_detach_away_mode(layout.session_dir, approval_scopes(cfg))
    err = frontend.spawn_detached_resume(cwd, layout.session_id, flags)
    if err:
        reporter.note(err)
        return
    reporter.out(f"\n[agent6] detached: {layout.session_id} continues in the background.")
    reporter.out(f"          reattach:  agent6 attach {layout.session_id}")


def run_leg(  # noqa: PLR0911, PLR0912, PLR0915 - one leg body, one return per ending
    cfg: Config,
    layout: SessionLayout,
    inputs: LegInputs,
    *,
    frontend: SessionFrontend,
    reporter: Reporter,
    events: EventSink,
    transcript_sink: TranscriptSink,
    cwd: Path,
    state_dir: Path,
) -> LegEnd:
    """Drive one leg to its end block. See the module docstring."""
    mode, role = inputs.mode, inputs.role
    label = "resume" if inputs.resuming else "run"
    # The interactive revision prompt reads the terminal; with the TUI owning
    # it the prompt would land invisibly in the console log and contend for
    # stdin. Skip revision for this leg instead.
    effective_revise_prompt = cfg.prompt.revise_prompt
    if effective_revise_prompt == "interactive" and inputs.tui_enabled:
        reporter.note(
            "prompt.revise_prompt='interactive' needs the terminal; the TUI"
            " owns it. Skipping prompt revision for this leg."
        )
        effective_revise_prompt = "off"
    stream_text, console_stream = frontend.stream_modes(inputs.tui_enabled)
    if console_stream:
        frontend.attach_console_view(events)
    session = build_session_providers(
        cfg, role=role, events=events, transcript_sink=transcript_sink, stream_text=stream_text
    )
    budget = session.budget
    prompt_reviser_provider = build_prompt_reviser_provider(
        cfg, transcript_sink=transcript_sink, budget=budget, events=events
    )
    cfg = inputs.gate(cfg, budget)

    # Steering (mid-run Ctrl-C -> the pause menu) needs the terminal; the
    # console view's heartbeat spinner is suspended for the prompt so its
    # line-erase cannot wipe the pause-menu line.
    steer_state = frontend.make_steer_state(
        events,
        layout.session_dir,
        session_facts_provider(
            budget, session.rm_role.model, cfg.sandbox.run_commands, inputs.isolation
        ),
    )

    interrupted = False
    result: SessionResult | None = None
    undo_outcome: list[tuple[str, str]] = []
    dispatcher: ToolDispatcher | None = None
    # Spawned inside the try so the finally below tears it down even if a
    # spawn (MCP) fails.
    mcp_manager = None
    session_net: SessionNetwork | None = None
    try:
        reporter.note(f"{'resume ' if inputs.resuming else ''}session id: {inputs.session_id}")

        # Spawn any configured MCP servers BEFORE the workflow starts so their
        # tools are visible from iteration 1. The manager owns its subprocesses;
        # the finally block below closes it. The run's session network, before
        # its first member: the commands and any server that joins it share it.
        if wants_session_network(cfg, inputs.isolation):
            session_net = SessionNetwork.open()
            # Published so `agent6 exec`/`forward` can join it: a separate
            # process names a namespace only through a live /proc entry.
            write_session_netns_pid(layout.session_dir, session_net.holder_pid)
        mcp_manager = start_mcp_manager_if_enabled(
            cfg, cwd, inputs.isolation, reporter=reporter, events=events, session_net=session_net
        )

        loop_log = frontend.loop_logger(mode)
        tools = build_session_tools(
            cfg,
            cwd=cwd,
            state_dir=state_dir,
            layout=layout,
            isolation=inputs.isolation,
            mode=mode,
            events=events,
            prompts=inputs.prompts,
            loop_log=loop_log,
            mcp_manager=mcp_manager,
            session_net=session_net,
            rm_role=session.rm_role,
            worktree_git_dir=inputs.worktree_git_dir,
        )
        curator = tools.curator
        dispatcher = tools.dispatcher
        cfg = tools.cfg

        def _undo_forker() -> tuple[str, str] | None:
            got = inputs.undo_forker()
            if got is not None:
                undo_outcome.append(got)
            return got

        after_auto_commit: Callable[[int, str], AutoCommitDirective] = (
            frontend.build_repl_hook(cwd, budget, inputs.session_id, mcp_manager)
            if inputs.interactive and mode == "run"
            else (lambda _i, _s: "continue")
        )
        wf = Workflow(
            root=cwd,
            config=cfg,
            standing_goal=inputs.standing_goal,
            interactive=inputs.interactive and mode == "run",
            initial_pins=inputs.pins,
            commit_trailer=render_commit_trailer(
                cfg.git.commit.trailer, models=(session.rm_role.model,)
            ),
            # `git.control = "model"` suspends the whole shadow chain: the
            # model's own commits are the record.
            chain_ref=chain_ref_for(inputs.session_id)
            if mode == "run" and cfg.git.control != "model"
            else None,
            chain_branch=inputs.chain_branch,
            chain_fallback_parent=inputs.base_sha or None,
            untracked_at_start=inputs.untracked_at_start,
            commit_per_step=cfg.git.commit_per_step,
            tool_result_cap_chars=tool_result_cap_chars(cfg),
            max_iterations=cfg.workflow.max_iterations,
            provider=session.provider,
            dispatcher=dispatcher,
            logger=loop_log,
            events=events,
            curator=curator,
            steer_requested=steer_state.requested,
            steer_clear=steer_state.clear,
            steer_prompt=steer_state.prompt,
            steer_reset=steer_state.reset_stage,
            # "Compact now" from a front-end: the same file-bridge pattern as
            # steer, honored at the next pre-call boundary.
            compact_requested=lambda: read_compact_request(layout.session_dir),
            compact_clear=lambda: clear_compact_request(layout.session_dir),
            stop_requested=lambda: stop_request_pending(layout.session_dir),
            stop_clear=lambda: clear_stop_request(layout.session_dir),
            should_abort=steer_state.abort_pending,
            undo_forker=_undo_forker,
            should_interrupt=steer_state.interrupt,
            # `/parallel` steer dispatch: the coordinator's group spawner (None
            # in plan/ask, and inside a lane -- depth 1).
            lane_spawner=frontend.build_coordinator_spawner(
                cfg,
                cwd,
                state_dir,
                mode,
                inputs.session_id,
                inputs.budget_overrides.max_usd if inputs.budget_overrides is not None else None,
                inputs.sandbox_overrides.auto_approve
                if inputs.sandbox_overrides is not None
                else False,
            ),
            budget=budget,
            state_dir=state_dir,
            # Written for every mode: `agent6 resume` reaches an ask too.
            resume_state_path=inputs.resume_state_path,
            mode=mode,
            plan_output_path=(layout.session_dir / "plan.md" if mode == "plan" else None),
            after_auto_commit=after_auto_commit,
            review_trigger=cfg.review.trigger,
            review_period=cfg.review.period,
            review_seats=session.review_seats,
            review_decision=cfg.review.decision,
            review_quorum=cfg.review.quorum,
            review_max_total_rejections=cfg.review.max_total_rejections,
            review_budget_fraction=cfg.review.budget_fraction,
            review_concurrency=cfg.review.concurrency,
            base_sha=inputs.base_sha,
            prompt_reviser_provider=prompt_reviser_provider,
            revise_prompt=effective_revise_prompt,
            temperature=role_temperature(cfg, role),
            prompt_reviser_temperature=role_temperature(cfg, "reviewer"),
            prompt_revision_selector=(
                frontend.select_revised_prompt if effective_revise_prompt == "interactive" else None
            ),
            summariser_provider=session.summariser_provider,
            compact_drop_at_chars=tools.compact_drop_at_chars,
            compact_summarise_at_chars=tools.compact_summarise_at_chars,
            context_summary_max_tokens=cfg.context.summary_max_tokens,
            keep_recent_chars=cfg.context.keep_recent_chars,
            keep_thinking_turns=cfg.context.keep_thinking_turns,
            compact_elision_gists=cfg.context.elision_gists,
        )
        try:
            with frontend.tui_session(layout.session_dir, inputs.tui_enabled):
                if inputs.task is None:
                    result = wf.resume()
                elif mode == "ask" and inputs.interactive:
                    result = frontend.run_ask_repl(wf, budget, layout, inputs.task)
                else:
                    result = wf.run(inputs.task)
                # A background command that ended after the last turn is written
                # down now, not at teardown: a viewer left open reads `/shells`
                # meanwhile.
                dispatcher.settle_background()
        except ResumeError as exc:
            reporter.error(str(exc))
            return LegEnd(1)
        except KeyboardInterrupt:
            interrupted = True
            reporter.err(f"\n[agent6] {label} interrupted")
            # The loop was cut mid-step, so it never emitted session.end; do it
            # here so an attached watcher/TUI stops instead of hanging. Carry
            # the iteration the loop reached so session.end keeps one shape.
            # suppress: the interrupt exit (130 + resume hint) must not be
            # masked by a dead journal.
            reason: SessionEndReason = "interrupted"
            with contextlib.suppress(EventWriteError):
                events.emit(
                    "session.end",
                    reason=reason,
                    iterations=wf.iterations_reached,
                    all_passed=False,
                )
        except Exception:
            # Any other escape (a broken stdout pipe from `| head`, an
            # unexpected fault) also leaves the loop without a session.end, and
            # the caller's finally then clears worker.pid -- the only immediate
            # liveness evidence -- so every surface would read the dead run as
            # "running" until the silence window expires. Record the end, then
            # re-raise.
            with contextlib.suppress(EventWriteError):
                events.emit(
                    "session.end",
                    reason="crashed",
                    iterations=wf.iterations_reached,
                    all_passed=False,
                )
            raise
    finally:
        steer_state.restore()
        session.close()
        if prompt_reviser_provider is not None:
            close_provider(prompt_reviser_provider)
        if dispatcher is not None:
            dispatcher.close()
        if mcp_manager is not None:
            mcp_manager.close()
        if session_net is not None:
            # The last handles on the run's network: closing them is what lets
            # the kernel reclaim it.
            session_net.close()
            clear_session_netns_pid(layout.session_dir)
        if (
            not interrupted
            and result is not None
            and auto_merge_eligible(result)
            and cfg.git.auto_merge
        ):
            finalize_auto_merge(
                cwd, layout=layout, cfg=cfg, reporter=reporter, budget=budget, events=events
            )
        # Never leave root-owned run state in the user's repo (sudo case).
        chown_to_real_user(state_dir)

    if interrupted:
        print_interrupt_end(layout=layout, cwd=cwd, budget=budget, reporter=reporter)
        return LegEnd(130)
    if result is None:
        return LegEnd(1)

    if mode == "ask":
        # The answer IS result.summary (kept whole in ask mode). stdout gets
        # just the answer (clean for piping); cost + saved-path go to stderr.
        # The REPL already printed + saved each turn, so only the one-shot path
        # prints/saves here.
        if inputs.ask_transcript_task is not None:
            reporter.out(result.summary)
            frontend.save_ask_transcript(layout, inputs.ask_transcript_task, result.summary)
            reporter.err(f"\n[agent6] answer saved to {layout.session_dir / 'transcript.md'}")
        reporter.err(budget.format_summary())
        return LegEnd(0 if result.completed else 1)

    if result.reason == "undone" and undo_outcome:
        new_id, undone_text = undo_outcome[-1]
        reporter.out(f"\n[agent6] undone: continue as {new_id} with your message back to edit:")
        reporter.out(f"    agent6 resume {new_id} --steer {undone_text!r}")
        return LegEnd(0)
    if result.reason == "detached":
        # Keep going in the background: the caller releases this run's worker
        # lock, then hands the run to `detach_to_background`.
        return LegEnd(0, detach_requested=True)

    print_session_end(
        result,
        layout=layout,
        cwd=cwd,
        budget=budget,
        console_stream=console_stream,
        reporter=reporter,
    )
    fire_notify_hook(
        cfg.notify,
        session_id=layout.session_id,
        session_dir=layout.session_dir,
        ok=result.completed,
        reason=result.reason,
        verified=result.verified,
        reporter=reporter,
    )
    return LegEnd(session_exit_code(result, stranded=stranded_edits(result, layout, cwd)))
