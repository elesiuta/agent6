# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""`agent6 run` (and its plan/ask modes): adapt argv, build the config and the
presentation seam, and hand the lifecycle to `agent6.app.run.run_task`."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from agent6.app._setup import (
    BudgetOverrides,
    SandboxOverrides,
    check_provider_keys,
    load_session_config,
)
from agent6.app.frontend import FrontendCapabilities, SessionFrontend
from agent6.app.parallel import build_coordinator_spawner
from agent6.app.preflight import (
    require_git_repo,
)
from agent6.app.run import run_task
from agent6.config import (
    Config,
    RoleName,
)
from agent6.events import EventSink
from agent6.models.validate import (
    configured_model_refusal,
    validate_configured_model,
    warning_message,
)
from agent6.paths import data_dir
from agent6.skills import operator_skills
from agent6.types import ResumableMode, session_kind
from agent6.ui.btw import asks_dir, direct_launch, make_btw_runner
from agent6.ui.cli._ask import (
    build_ask_session_digest,
    run_ask_repl,
    save_ask_transcript,
)
from agent6.ui.cli._common import error, refuse, warn
from agent6.ui.cli._console_view import ConsoleView
from agent6.ui.cli._interact import (
    build_approver,
    build_questioner,
    lane_away_mode,
    prompt_detach_away_mode,
)
from agent6.ui.cli._live import (
    loop_logger,
    should_spawn_tui,
    stream_modes,
    tui_session,
)
from agent6.ui.cli._preflight import (
    confirm_replay_after_crash,
    confirm_run_on_run_branch,
    confirm_unconfined_autorun,
)
from agent6.ui.cli._repl import build_repl_hook
from agent6.ui.cli._steer import (
    make_steer_state,
    select_revised_prompt,
)
from agent6.ui.cli._task_refs import (
    expand_task_file_refs,
)
from agent6.ui.cli.parallel import dispatch_parallel, lane_runtime
from agent6.ui.spawn import agent6_exe, spawn_detached_resume
from agent6.ui.steer import SteerState
from agent6.viewmodel import session_policy


def _skills_task_prefix(cfg: Config, names: tuple[str, ...]) -> tuple[str, str]:
    """Resolve `--skill` names to a task-prompt prefix. Returns (prefix, error)."""
    resolved = operator_skills(
        cfg.skills.enabled, cfg.skills.extra_dirs, cfg.skills.state, data_dir() / "skills"
    )
    by_name = {s.name: s for s in (*resolved.enabled, *resolved.always)}
    blocks: list[str] = []
    for n in names:
        skill = by_name.get(n)
        if skill is None:
            if not cfg.skills.enabled:
                return "", (
                    f"--skill: skills are disabled, so {n!r} cannot load"
                    " (agent6 config set skills.enabled true)"
                )
            available = ", ".join(sorted(by_name)) or "(none installed)"
            return "", f"--skill: unknown or disabled skill {n!r}; available: {available}"
        blocks.append(f'<skill name="{skill.name}">\n{skill.text.rstrip()}\n</skill>')
    joined = "\n\n".join(blocks)
    return (
        f"Apply the operator-installed skill(s) below to this task.\n\n{joined}\n\n---\n\n",
        "",
    )


def _remember_steer(cell: list[SteerState | None], state: SteerState) -> SteerState:
    """Publish the leg's SteerState for the approver's late-bound read."""
    cell[0] = state
    return state


def session_frontend(config_path: Path | None = None) -> SessionFrontend:
    """Build the presentation seam `app.run.run_task` / `app.resume.resume_task`
    drive: one per invocation (the console-view cell is run-scoped). The console
    view is created lazily on `attach_console_view`; the builders that need it
    close over its cell, so the lifecycle never holds a UI type. The lifecycle
    owns egress (`app.egress`) itself; only the two exe-spawn primitives it
    can't reach (`ui.spawn`) are injected."""
    # Both late-bound: the lifecycle builds the approver and questioner before
    # the leg attaches the console view or the steer state exists; they read
    # the cells at prompt time (an operator prompt pauses the view's heartbeat
    # and counts as a Ctrl-C boundary, see build_approver).
    console_cell: list[ConsoleView | None] = [None]
    steer_cell: list[SteerState | None] = [None]

    def attach_console_view(events: EventSink) -> None:
        # The sink writes into the run dir, so its path is the handle to the
        # run's policy facts without threading the layout through the protocol.
        view = ConsoleView(sys.stderr, policy=lambda: session_policy(events.path.parent).line())
        console_cell[0] = view
        events.subscribe(view)

    def close_console_view() -> None:
        view = console_cell[0]
        if view is not None:
            view.close()

    return SessionFrontend(
        # The CLI is the surface with a terminal: it can do everything. What it
        # cannot do -- ask a human with no tty and no away-mode -- is the
        # lifecycle's own preflight refusal, not a missing capability.
        # The CLI asks on the terminal, so a pipe for stdin means it cannot.
        capabilities=FrontendCapabilities(can_ask=sys.stdin.isatty()),
        should_spawn_tui=lambda tui, interactive, mode: should_spawn_tui(
            tui=tui, interactive=interactive, mode=mode
        ),
        stream_modes=lambda tui_enabled: stream_modes(tui_enabled=tui_enabled),
        attach_console_view=attach_console_view,
        close_console_view=close_console_view,
        loop_logger=lambda mode: loop_logger(mode, console_cell[0]),
        tui_session=lambda session_dir, enabled: tui_session(session_dir, enabled=enabled),
        build_approver=lambda session_dir: build_approver(session_dir, console_cell, steer_cell),
        build_questioner=lambda session_dir: build_questioner(session_dir, console_cell),
        make_steer_state=lambda events, session_dir, facts: _remember_steer(
            steer_cell,
            make_steer_state(
                events,
                session_dir,
                console_cell[0],
                facts,
                # `/btw` spawns beside the run. `direct_launch` is right here: the
                # CLI process is the one with a terminal, so it is not the confined
                # coordinator a `/parallel` lane has to escape from.
                make_btw_runner(
                    session_dir.name,
                    launch=direct_launch,
                    list_asks=lambda: (
                        [d for d in asks_dir(session_dir).iterdir() if d.is_dir()]
                        if asks_dir(session_dir).is_dir()
                        else []
                    ),
                    events=events,
                ),
                config_path=config_path,
            ),
        ),
        confirm_unconfined_autorun=confirm_unconfined_autorun,
        confirm_run_on_run_branch=confirm_run_on_run_branch,
        confirm_replay_after_crash=confirm_replay_after_crash,
        prompt_detach_away_mode=prompt_detach_away_mode,
        select_revised_prompt=lambda original, revised, questions: select_revised_prompt(
            original, revised, questions, console_cell[0]
        ),
        build_repl_hook=lambda cwd, budget, session_id, mcp_manager: build_repl_hook(
            cwd,
            budget,
            session_id=session_id,
            mcp_manager=mcp_manager,
            console_view=console_cell[0],
        ),
        run_ask_repl=lambda wf, budget, layout, first_question: run_ask_repl(
            wf, budget, layout, first_question=first_question
        ),
        save_ask_transcript=lambda layout, question, answer: save_ask_transcript(
            layout, question=question, answer=answer
        ),
        build_coordinator_spawner=(
            lambda cfg, cwd, state_dir, mode, session_id, max_usd, auto_approve: (
                build_coordinator_spawner(
                    cfg,
                    cwd,
                    state_dir,
                    mode=mode,
                    session_id=session_id,
                    runtime=lane_runtime(),
                    max_usd=max_usd,
                    auto_approve=auto_approve,
                    lane_away=lane_away_mode(),
                )
            )
        ),
        agent6_exe=agent6_exe,
        spawn_detached_resume=lambda cwd, sid, flags: spawn_detached_resume(
            cwd, sid, config_path=config_path, flags=flags
        ),
    )


def _configured_model_ok(cfg: Config, role: RoleName) -> bool:
    """The configured-model wall: validate models.<role>.model against its
    provider's listing so a typo refuses cleanly here -- with a did-you-mean,
    like the `/parallel` path -- instead of dying at the first provider call
    and echoing the raw upstream 400. A miss re-checks the live listing before
    refusing (models.validate); a failed re-check warns and proceeds. False =
    refused (the caller exits 2)."""
    verdict = validate_configured_model(cfg, role)
    if verdict.refused:
        # Name the entry the user actually wrote: a plan whose planner fell
        # back to the worker model must say models.worker.model, not point at
        # a models.planner section absent from their config.
        source = cfg.models.source_role(role)
        refuse(f"{configured_model_refusal(verdict, source)}")
        return False
    if verdict.warned:
        # The cached listing lacks the model and the live re-check failed
        # (offline, provider down): proceed -- the first provider call is the
        # final arbiter -- but say why a bad id would die there.
        warn(f"{warning_message(verdict)}")
    return True


def _compose_task(
    task: str, cfg: Config, *, skills: tuple[str, ...], seed_from: str
) -> tuple[str, str]:
    """The prompt the session actually starts from. Returns (task, error).

    One place assembles it: the skills prefix, then another session's context
    when `--from` seeds this one. `--from` starts a NEW session and leaves the
    source untouched -- keeping a session's mode is `fork`; this picks the mode
    by being the command the operator typed.
    """
    if skills:
        prefix, skills_err = _skills_task_prefix(cfg, skills)
        if skills_err:
            return task, skills_err
        task = prefix + task
    if seed_from:
        digest = build_ask_session_digest(Path.cwd(), seed_from, latest=False)
        if digest is None:
            return task, f"could not seed from {seed_from!r}"
        task = f"{digest}\n\n{task}" if task else digest
    return task, ""


def _cmd_run(  # noqa: PLR0911
    config_path: Path | None,
    task: str,
    *,
    session_id: str = "",
    interactive: bool = False,
    tui: bool = False,
    decompose: bool = False,
    mode: ResumableMode = "run",
    seed_from: str = "",
    skills: tuple[str, ...] = (),
    budget_overrides: BudgetOverrides | None = None,
    sandbox_overrides: SandboxOverrides | None = None,
    preset: str = "",
    parallel_spec: str = "",
    standing_goal: str = "",
    pins: tuple[str, ...] = (),
) -> int:
    """Adapt `agent6 run`/`plan`/`ask` argv: build the effective config, apply
    the flag overrides, resolve skills and @file refs, route `--parallel`,
    then drive the lifecycle (`app.run.run_task`) with the injected seam."""
    # The not-a-git-repo wall first: run/plan need git; ask is read-only and
    # may run outside a repo. A user in a scratch non-git dir must not clear
    # the provider, model, and key walls serially only to discover at the end
    # that they also need git.
    if mode != "ask" and not require_git_repo(Path.cwd()):
        return 2
    effective = load_session_config(
        Path.cwd(),
        config_path,
        mode=mode,
        preset=preset,
        budget_overrides=budget_overrides,
        sandbox_overrides=sandbox_overrides,
    )
    cfg, explicit_leaves = effective.config, effective.explicit_leaves
    if decompose:  # --decompose: plan-first for this run (overrides config)
        cfg = cfg.with_decompose("on")
    task, compose_err = _compose_task(task, cfg, skills=skills, seed_from=seed_from)
    if compose_err:
        error(f"{compose_err}")
        return 2
    role = session_kind(mode).role

    # Resolve @path references in the task string before the
    # workflow ever sees it. Lets the user write "fix the bug in @src/x.py
    # described in @notes.md" and have those files inlined verbatim.
    task = expand_task_file_refs(task, Path.cwd())

    # Provider key + models-cache preflight, shared by the single run and the
    # --parallel fan-out: resolves each referenced provider's key AND refreshes
    # its models cache, which carries the pricing explicit_usd_flag_error reads.
    # Runs before the --parallel route so dispatch_parallel's own --max-usd check
    # sees the same refreshed cache a plain --max-usd run does.
    missing = check_provider_keys(cfg)
    if missing is not None:
        print(missing, file=sys.stderr)
        return 2

    if not _configured_model_ok(cfg, role):
        return 2

    # `--parallel`: fan out isolated lanes instead of a single run. Routed here,
    # after config/skills/require_runnable and the key preflight, but BEFORE the
    # single-run sandbox preflight (no branch cut, no run dir on the origin); the
    # orchestrator clones each lane and runs its own `agent6 run`. run mode only.
    if parallel_spec and mode == "run":
        # Depth 1: a subordinate lane (AGENT6_SUBRUN) must never itself fan out.
        if os.environ.get("AGENT6_SUBRUN"):
            refuse(
                "--parallel is unavailable inside a subordinate run (parallel dispatch is depth 1)."
            )
            return 2
        return dispatch_parallel(
            cfg,
            task,
            parallel_spec,
            cwd=Path.cwd(),
            max_usd=budget_overrides.max_usd if budget_overrides is not None else None,
            auto_approve=sandbox_overrides.auto_approve if sandbox_overrides is not None else False,
            pins=pins,
        )

    return run_task(
        cfg,
        task,
        frontend=session_frontend(config_path),
        session_id=session_id,
        interactive=interactive,
        tui=tui,
        mode=mode,
        budget_overrides=budget_overrides,
        sandbox_overrides=sandbox_overrides,
        preset=preset,
        pins=pins,
        standing_goal=standing_goal,
        explicit_leaves=explicit_leaves,
    )
