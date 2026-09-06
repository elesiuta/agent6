# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The `agent6 run` lifecycle (and its plan/ask modes): preflight, branch cut,
manifest, loop construction, finalize. `ui/cli/run.py` adapts argv, builds the
:class:`SessionFrontend` seam, and calls :func:`run_task`; everything that touches
the terminal is injected through that seam so this module never imports
`agent6.ui` (mirrors `LaneRuntime` in `app.parallel`)."""

from __future__ import annotations

import contextlib
import os
import shutil
import sys
from collections.abc import Sequence
from pathlib import Path

from agent6.app._leg import LegInputs, detach_to_background, run_leg
from agent6.app._session import (
    select_isolation,
    warn_install_inside_workspace,
)
from agent6.app._setup import (
    BudgetOverrides,
    SandboxOverrides,
    override_flags,
    session_config,
)
from agent6.app.finalize import (
    finalize_auto_stash,
    stash_recovery_hint,
)
from agent6.app.frontend import (
    SessionFrontend,
    apply_spawned_away_default,
    approval_scopes,
)
from agent6.app.manifest import (
    pin_gate,
    stamp_parked,
    write_session_manifest,
)
from agent6.app.preflight import (
    DirtyTreeChoice,
    SessionRefused,
    dirty_tree_choice,
    dirty_tree_question,
    dirty_tree_refusal,
    drop_gate_if_unrunnable,
    git_preflight,
    headless_approval_refusal,
    infer_verify_if_unset,
    unmerged_run_holding_the_tree,
)
from agent6.app.reporter import STDIO_REPORTER, Reporter
from agent6.budget import BudgetTracker
from agent6.config import Config
from agent6.config.layer import resolved_state_dir
from agent6.events import EventSink
from agent6.git_ops import (
    GitError,
    auto_stash_message,
    modified_paths,
    run_branch_for,
    stash_tracked_changes,
    untracked_paths,
)
from agent6.paths import mkdir_for_real_user
from agent6.providers import TranscriptSink
from agent6.sessions.id import (
    SessionIdError,
    session_id_bucket,
    unused_session_id,
    validate_explicit_session_id,
)
from agent6.sessions.ipc import (
    away_mode,
    clear_away_mode,
    clear_pending_answers,
    clear_worker_pid,
    submit_steer,
    write_worker_pid,
)
from agent6.sessions.layout import LOGS_NAME, SessionLayout, write_untracked_at_start
from agent6.sessions.lock import (
    SINGLE_WRITER_BUSY,
    acquire_repo_writer,
    acquire_single_writer,
    release_single_writer,
    repo_writer_holder,
)
from agent6.sessions.manifest import ManifestError, read_manifest
from agent6.tools.operator_prompts import OperatorPrompts
from agent6.types import ResumableMode, session_bucket, session_kind
from agent6.workflows._context import agents_md_notices


def discard_husk_dir(session_dir: Path) -> None:
    """Remove a run dir a preflight refused before any real content was written
    (no manifest, no logs). Otherwise a refused start (e.g. dirty worktree)
    leaves an empty husk that `agent6 sessions` lists as '(no logs)' forever. Guarded
    on the manifest/logs check so a real run's dir is never removed."""
    if (session_dir / "manifest.json").exists() or (session_dir / LOGS_NAME).exists():
        return
    with contextlib.suppress(OSError):
        shutil.rmtree(session_dir)


def run_task(  # noqa: PLR0911, PLR0912, PLR0915
    cfg: Config,
    task: str,
    *,
    frontend: SessionFrontend,
    session_id: str = "",
    interactive: bool = False,
    tui: bool = False,
    mode: ResumableMode = "run",
    standing_goal: str = "",
    budget_overrides: BudgetOverrides | None = None,
    sandbox_overrides: SandboxOverrides | None = None,
    preset: str = "",
    initial_steer: str = "",
    pins: Sequence[str] = (),
    preset_stamp: tuple[str, bool] | None = None,
    # Which config leaves the operator actually WROTE, as dotted paths. A
    # default that this host cannot honour degrades with a warning; a value
    # they wrote down refuses, because they asked for something specific.
    explicit_leaves: frozenset[str] = frozenset(),
    reporter: Reporter = STDIO_REPORTER,
) -> int:
    """Single-loop agent: one provider, one LLM driving via tool
    calls over the fixed tool surface, deterministic harness (jail +
    budget + verify timeout + DAG curator for persistence/resume).
    Sole `agent6 run` path; returns the process exit code.

    `initial_steer` queues an operator follow-up for the loop's first
    boundary, seeded AFTER this function's own stale-state clear -- the
    parked-resume delegation passes `resume --steer` through it (a pre-seeded
    bridge file would be wiped by that clear and silently lost).

    The caller (`ui/cli/run.py`) has already built *cfg* (config + overrides),
    resolved the task text, checked the git-repo wall / runnable roles /
    provider keys, and routed `--parallel` away. *budget_overrides* /
    *sandbox_overrides* are passed through for the flags the lifecycle re-reads
    (`--max-usd` enforcement, lane dispatch).

    *preset_stamp* `(name, from_flag)` overrides the manifest's stamped
    preset instead of deriving it from *preset*. A parked resume has no
    `--preset` flag but must record the ORIGINAL submission's stamp so a
    later resume/fork replays the same precedence (fork carries it likewise);
    deriving it from the empty *preset* would drop the stamp, and the flag's
    veto with it, on the next leg.

    When `mode="plan"` the same harness drives a planning
    pass instead of an execution pass: planning system prompt,
    edit-tools filtered out, `finish_planning` instead of
    `finish_session`, no auto-commit. The plan markdown lands at
    `<run-dir>/plan.md` and is consumed by `agent6 run --from-plan`.
    The `planner` model role drives plan mode (falls back to `worker`).
    """
    role = session_kind(mode).role

    # Before anything reads a knob (see session_config): an interactive session
    # (ask / plan) never runs a command unwatched, whether it is starting here
    # or resuming -- unless the operator granted this invocation, which lands
    # after the clamp.
    cfg = session_config(cfg, mode, sandbox_overrides)
    # Refuse an unanswerable run BEFORE anything is created: refusing after
    # the session dir and its manifest exist would leave a never-started run
    # listed forever and poison its id (`--session-id` retries answer "already
    # exists, use resume", and resume finds no snapshot). Everything this needs
    # is known here; the clamp above is the last thing that can change
    # `run_commands`.
    tui_enabled = frontend.should_spawn_tui(tui, interactive, mode)
    refusal = headless_approval_refusal(
        cfg,
        tui_enabled=tui_enabled,
        away=os.environ.get("AGENT6_DETACHED_AWAY", ""),
        can_ask=frontend.capabilities.can_ask,
        clamped=session_kind(mode).clamps_commands,
    )
    if refusal is not None:
        reporter.refuse(refusal)
        return 2
    cwd = Path.cwd()
    try:
        isolation = select_isolation(
            cfg,
            cwd=cwd,
            confirm_unconfined=frontend.confirm_unconfined_autorun,
            reporter=reporter,
            explicit_leaves=explicit_leaves,
        )
    except SessionRefused as refusal:
        return refusal.rc

    try:
        git = git_preflight(
            cwd,
            cfg,
            mode,
            confirm_run_on_run_branch=frontend.confirm_run_on_run_branch,
            reporter=reporter,
        )
    except SessionRefused as refusal:
        return refusal.rc
    base_sha, base_branch = git.base_sha, git.base_branch

    # Layout: standard run-dir scaffolding for transcripts + logs. ask sessions
    # live under the per-repo state dir (asks subdir) to stay separate from real runs.
    if session_id:
        try:
            validate_explicit_session_id(session_id)
        except SessionIdError as exc:
            reporter.error(str(exc))
            return 2
    state_dir = resolved_state_dir(cwd)
    bucket = session_bucket(mode)
    # Same-bucket reuse is the resume/park flow below; another bucket's id is
    # a collision every surface would see as ambiguous.
    if session_id and (held := session_id_bucket(state_dir, session_id)) not in (None, bucket):
        reporter.error(
            f"--session-id {session_id!r} already names a session under {held}/;"
            " ids are unique across every bucket. Pick another id."
        )
        return 2
    effective_session_id = session_id or unused_session_id(state_dir, bucket)
    layout = SessionLayout(
        state_dir=state_dir,
        session_id=effective_session_id,
        subdir=bucket,
    )
    # An explicit --session-id that already has a session is a resume, not a fresh start:
    # reusing the dir would write a new manifest + loop_state beside the old run's
    # graph/checkpoints/transcripts (mixed state). Refuse and point at resume.
    # (ask sessions are transient Q&A, so reusing their dir is fine.) The one
    # reusable dir is a PARKED run (manifest carries parked_task, nothing else
    # ever ran): starting it IS its fresh start, and the manifest rewrite below
    # un-parks it.
    if session_id and mode != "ask" and layout.manifest_path.exists():
        try:
            parked = read_manifest(layout.session_dir).parked_task
        except ManifestError:
            parked = ""
        if not parked:
            reporter.error(
                f"run {session_id!r} already exists. Use `agent6 resume {session_id}` to "
                "continue it, or choose a different --session-id."
            )
            return 2
    # Under sudo the first run on a machine creates the whole state ancestry;
    # hand the created dirs back NOW, not at teardown -- a killed run must not
    # leave a root-owned base that blocks every other repo's non-root runs.
    mkdir_for_real_user(layout.session_dir)
    layout.ensure()
    # One authoritative writer per run dir. Acquire BEFORE touching any shared
    # run state (clearing answers, the worker pid, the curator) so a second
    # process refuses cleanly instead of clobbering the live run.
    worker_lock_fd = acquire_single_writer(layout.session_dir)
    if worker_lock_fd is None:
        reporter.err(SINGLE_WRITER_BUSY.format(rid=effective_session_id))
        return 2
    repo_lock_fd: int | None = None
    stashed = False
    # Apply the stash back at run end (onto a clean tree, else the apply line
    # is printed): [git].auto_stash_pop, or "stash" chosen at the start question.
    stash_pop = cfg.git.auto_stash_pop
    untracked_at_start: frozenset[str] = frozenset()
    run_branch: str | None = None
    detach_requested = False

    def _undo_forker() -> tuple[str, str] | None:
        # Lazy: app.fork imports app.resume, which imports this module.
        from agent6.app.fork import undo_fork  # noqa: PLC0415

        return undo_fork(None, effective_session_id, cwd=cwd, reporter=reporter)

    try:
        # Drop stale approve/ask/steer answers from a prior session (the
        # id counters reset on resume, so an old answer must not be read instead of
        # re-prompting; dead front-end claims are pruned by the liveness probe).
        clear_pending_answers(layout.session_dir)
        if initial_steer.strip():
            submit_steer(layout.session_dir, initial_steer.strip())
        if sys.stdin.isatty():  # a foreground start clears a stale detach away-mode
            clear_away_mode(layout.session_dir)
        else:
            apply_spawned_away_default(layout.session_dir, approval_scopes(cfg))
        # A visible branch named after the run id is 1:1 with the run (find it
        # from any run id, `agent6 sessions diff <id>`, or delete the branch to
        # drop the pointer). The name is the unique run id. Only real `run`
        # mode branches: `plan`/`ask` make no commits. The ref itself is
        # advanced by the first chain commit; nothing is cut or checked out.
        if cfg.git.branch_per_run and mode == "run" and cfg.git.control != "model":
            run_branch = run_branch_for(effective_session_id)

        # The operator's uncommitted changes to tracked files. Untracked files
        # are not in question: they stay out of the run (`untracked_at_start`).
        # A run that would have to ask about them but cannot refuses BEFORE
        # anything is created (see the approval refusal above).
        modified = modified_paths(cwd) if mode == "run" else []
        must_ask = bool(modified) and not cfg.git.auto_stash and cfg.git.require_clean_worktree
        answerable = frontend.capabilities.can_ask or away_mode(layout.session_dir) == "wait"
        unmerged_run = (
            unmerged_run_holding_the_tree(
                cwd, state_dir, except_id=effective_session_id, modified=modified
            )
            if must_ask
            else ""
        )
        if must_ask and not answerable:
            reporter.refuse(dirty_tree_refusal(modified, unmerged_run=unmerged_run))
            discard_husk_dir(layout.session_dir)
            return 2

        transcript_sink = TranscriptSink(layout.transcripts_dir)
        events = EventSink(layout.logs_path)
        # The leg's one gate to the operator: every prompt journals and takes
        # its id here, whichever front-end answers.
        prompts = OperatorPrompts(
            approver=frontend.build_approver(layout.session_dir),
            questioner=frontend.build_questioner(layout.session_dir),
            journal=events.emit,
            session_dir=layout.session_dir,
        )

        warn_install_inside_workspace(cwd, reporter=reporter)
        for line in agents_md_notices(cwd):
            reporter.note(line)

        # Write the run manifest. This is the canonical record of where the
        # run started (base_sha + base_branch), which model+provider drove
        # it, and the user_task it was given. `agent6 sessions diff <run-id>` and
        # any future tooling that wants to reproduce a run reads from here.
        # Written before the gates below, which PARK rather than refuse: a
        # parked run keeps its dir and manifest, and `agent6 resume <id>` starts
        # it fresh (that start rewrites the manifest and un-parks it).
        write_session_manifest(
            layout,
            session_id=effective_session_id,
            user_task=task,
            base_sha=base_sha,
            base_branch=base_branch,
            run_branch=run_branch,
            cfg=cfg,
            mode=mode,
            effective_preset=(preset_stamp[0] if preset_stamp else (preset or cfg.preset)),
            preset_from_flag=(preset_stamp[1] if preset_stamp else bool(preset)),
            isolation=isolation,
        )

        def _park(reason: str, detail: str, *, hint: str = "") -> int:
            # *reason* is the short cause every listing shows beside "parked";
            # *detail* is the sentence the operator reads now.
            stamp_parked(layout.session_dir, task=task, reason=reason)
            reporter.err(
                f"PARKED: {detail}. Your task is saved as run {effective_session_id!r}:\n"
                f"    agent6 resume {effective_session_id}    (starts it)"
                + (f"\n{hint}" if hint else "")
            )
            return 2

        if mode == "run":
            # One live run-mode worker per CHECKOUT, not just per run dir: two
            # runs share one worktree, so each would commit the other's
            # in-flight edits into its own chain. Taken BEFORE any tree mutation
            # (auto-stash, branch cut). plan/ask are read-only and skip it.
            repo_lock_fd = acquire_repo_writer(layout.state_dir, effective_session_id)
            if repo_lock_fd is None:
                holder = repo_writer_holder(layout.state_dir) or "another run"
                return _park(
                    "checkout busy",
                    f"run {holder!r} is already driving this checkout, and a second run-mode"
                    " worker would interleave auto-commits on the one working tree",
                    hint=(
                        f"or hand it to the live run as an isolated lane by steering"
                        f" {holder!r} with:\n    /parallel 1 <the same task>"
                    ),
                )
            # Settle the operator's uncommitted changes BEFORE the run's first
            # commit can sweep them up: config decides when it can, else the
            # operator is asked over the same channel as `ask_user`.
            if modified:
                choice: DirtyTreeChoice
                if cfg.git.auto_stash:
                    choice = "stash"
                elif not cfg.git.require_clean_worktree:
                    choice = "include"
                else:
                    question = dirty_tree_question(modified, unmerged_run=unmerged_run)
                    answers = prompts.ask((question,))
                    choice = dirty_tree_choice(answers[0] if answers else "")
                    stash_pop = stash_pop or choice == "stash"
                if choice == "cancel":
                    settle = (
                        f"`agent6 sessions merge {unmerged_run}` lands them (the unmerged work"
                        f" of run {unmerged_run})"
                        if unmerged_run
                        else "commit or stash them"
                    )
                    return _park(
                        "uncommitted changes",
                        f"the working tree has uncommitted changes to tracked files; {settle},"
                        " then start the run",
                    )
                if choice == "stash":
                    try:
                        stash_tracked_changes(cwd, auto_stash_message(effective_session_id))
                        stashed = True
                    except GitError as exc:
                        return _park(
                            "stash failed", f"stashing the working tree's changes failed: {exc}"
                        )
            # The files that are the operator's: untracked at this moment, after
            # any stash. Every chain commit and dirty check leaves them out.
            untracked_at_start = untracked_paths(cwd)
            write_untracked_at_start(layout.session_dir, untracked_at_start)

        def _gate(cfg: Config, budget: BudgetTracker) -> Config:
            # Verify is optional: if unset, infer one for this run (AGENTS.md
            # -> repo signals -> a cheap LLM call) and inject it in-memory.
            # Never persisted. The drop comes LAST so nothing hands the gate
            # back: a leg that cannot run a command is gateless, whatever
            # inference found.
            configured_gate = bool(cfg.workflow.verify_command)
            cfg = infer_verify_if_unset(
                cfg, cwd, mode=mode, events=events, transcript_sink=transcript_sink, budget=budget
            )
            cfg = drop_gate_if_unrunnable(cfg, session_dir=layout.session_dir, reporter=reporter)
            # After resolution, never before: preflight can DROP the gate (a
            # run that cannot run commands), and an empty gate with an origin
            # of "configured" is a self-contradiction the next leg reads back.
            gate_origin = ""
            if cfg.workflow.verify_command:
                gate_origin = "configured" if configured_gate else "inferred"
            # Pin it: from here the run is judged by THIS gate, whatever the
            # file it was inferred from says later.
            pin_gate(
                layout.session_dir,
                cfg.workflow.verify_command,
                gate_origin,
                events=events,
                reporter=reporter,
            )
            return cfg

        # The pid lands only once every refusal is behind: `sessions show` reads
        # it as a live worker (a long provider call emits no events), and a run
        # that refused never had one.
        write_worker_pid(layout.session_dir, os.getpid())
        end = run_leg(
            cfg,
            layout,
            LegInputs(
                session_id=effective_session_id,
                mode=mode,
                role=role,
                isolation=isolation,
                tui_enabled=tui_enabled,
                interactive=interactive,
                task=task,
                gate=_gate,
                chain_branch=run_branch,
                base_sha=base_sha,
                untracked_at_start=untracked_at_start,
                # Written for every mode: `agent6 resume` reaches an ask too.
                resume_state_path=layout.session_dir / "loop_state.json",
                undo_forker=_undo_forker,
                prompts=prompts,
                # The REPL already printed + saved each turn.
                ask_transcript_task=None if interactive else task,
                budget_overrides=budget_overrides,
                sandbox_overrides=sandbox_overrides,
                standing_goal=standing_goal,
                pins=tuple(pins),
            ),
            frontend=frontend,
            reporter=reporter,
            events=events,
            transcript_sink=transcript_sink,
            cwd=cwd,
            state_dir=state_dir,
        )
        detach_requested = end.detach_requested
        return end.rc
    finally:
        # Single owner of worker.pid, both writer locks, and auto-stash
        # finalization, for EVERY exit path: preflight refusals, Ctrl-C during
        # verify inference, and setup-window crashes included. worker.pid and
        # the stash pop happen UNDER the locks, so the releases come after --
        # nested, because they must survive a teardown raise: the ACP front-end
        # calls run_task in-process, where a leaked flock refuses every later
        # run on the session until the server restarts.
        try:
            frontend.close_console_view()  # stop the heartbeat thread, clear any spinner line
            clear_worker_pid(layout.session_dir)
            if stashed:
                if detach_requested:
                    # The run is NOT over: popping the stash now would feed the
                    # user's pre-run files into the detached continuation's
                    # auto-commits. Leave the stash and say so.
                    # By sha, never by position: this hint has the LONGEST window of
                    # any -- the operator reads it now and runs it after a
                    # background run that may take hours, by which point a
                    # positional pop restores whatever else was stashed meanwhile.
                    hint = stash_recovery_hint(
                        cwd, session_id=effective_session_id, base_branch=base_branch
                    )
                    reporter.note(
                        "pre-run changes remain stashed while the run continues"
                        " in the background; after it ends, restore them with:"
                        f" {hint}"
                        if hint
                        else "pre-run changes remain stashed while the run continues"
                        " in the background, but the stash could not be located; check"
                        " `git stash list`"
                    )
                else:
                    finalize_auto_stash(
                        cwd,
                        base_branch=base_branch,
                        run_branch=run_branch,
                        auto_pop=stash_pop,
                        session_id=layout.session_id,
                        exclude=untracked_at_start,
                        reporter=reporter,
                    )
        finally:
            release_single_writer(repo_lock_fd)
            release_single_writer(worker_lock_fd)
        if detach_requested:
            # The worker lock is released now, so the detached `resume` acquires it.
            detach_to_background(
                frontend=frontend,
                cfg=cfg,
                layout=layout,
                cwd=cwd,
                flags=override_flags(budget_overrides, sandbox_overrides),
                reporter=reporter,
            )
