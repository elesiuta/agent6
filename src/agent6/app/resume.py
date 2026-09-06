# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The `agent6 resume` lifecycle: pick a paused or crashed run back up from its
snapshot. `ui/cli/resume.py` adapts argv and injects the same
:class:`agent6.app.run.SessionFrontend` seam `run_task` uses."""

from __future__ import annotations

import contextlib
import json
import os
import sys
from collections.abc import Callable
from pathlib import Path

from agent6.app._leg import LegInputs, detach_to_background, run_leg
from agent6.app._session import (
    select_isolation,
    warn_install_inside_workspace,
)
from agent6.app._setup import (
    BudgetOverrides,
    SandboxOverrides,
    check_provider_keys,
    load_session_config,
    override_flags,
)
from agent6.app.frontend import (
    SessionFrontend,
    apply_spawned_away_default,
    approval_scopes,
)
from agent6.app.manifest import pin_gate, stamp_preset
from agent6.app.preflight import (
    SessionRefused,
    drop_gate_if_unrunnable,
    headless_approval_refusal,
    require_git_repo,
)
from agent6.app.reporter import STDIO_REPORTER, Reporter
from agent6.app.run import run_task
from agent6.budget import BudgetTracker
from agent6.config import (
    Config,
    ConfigError,
)
from agent6.config.layer import (
    resolved_state_dir,
)
from agent6.directive import steer_problem
from agent6.events import EventSink
from agent6.git_ops import (
    CommitIdentity,
    GitError,
    chain_ref_for,
    chain_tip,
    is_ancestor,
    verify_git_identity,
)
from agent6.providers import (
    TranscriptSink,
)
from agent6.sessions.id import SessionIdError, resolve_session
from agent6.sessions.ipc import (
    clear_away_mode,
    clear_pending_answers,
    clear_worker_pid,
    submit_steer,
    write_worker_pid,
)
from agent6.sessions.layout import (
    bucket_dir,
    read_untracked_at_start,
)
from agent6.sessions.lock import (
    SINGLE_WRITER_BUSY,
    acquire_repo_writer,
    acquire_single_writer,
    release_single_writer,
    repo_writer_holder,
)
from agent6.sessions.manifest import ManifestError, read_manifest
from agent6.types import SESSION_KINDS, session_bucket, session_kind
from agent6.viewmodel import newest_session_dir
from agent6.viewmodel.listing import finished_needs_new_work, needs_new_work_refusal
from agent6.workflows._context import agents_md_notices
from agent6.workflows._session_state import (
    TURN_IN_FLIGHT_NAME,
    clear_turn_marker,
    load_session_snapshot,
    read_turn_marker,
)


def resumable_bucket_dirs(state_dir: Path) -> list[Path]:
    """The bucket dirs holding sessions `agent6 resume` can pick up.

    Derived from the mode records rather than listed again here: a new resumable
    mode that a hand-kept list forgot would be resumable by id and invisible to
    the bare form.
    """
    return [
        bucket_dir(state_dir, session_bucket(kind.name))
        for kind in SESSION_KINDS.values()
        if kind.resumable
    ]


def turn_replay_allowed(
    session_dir: Path,
    next_iteration: int,
    confirm: Callable[[int, tuple[str, ...]], bool],
) -> bool:
    """Whether resume may proceed given the mid-turn-crash marker state.

    No marker (clean stop) or a STALE one (crash after the after-tools
    snapshot advanced but before the delete; iteration < next) proceeds, the
    stale marker cleared silently -- no false prompt. A marker matching the
    turn about to re-run is a genuine mid-turn crash: its tools may have
    partially applied, so the front-end decides (interactive default no;
    headless warns and proceeds). Approval clears the marker so one crash
    asks once."""
    marker_path = session_dir / TURN_IN_FLIGHT_NAME
    marker = read_turn_marker(marker_path)
    if marker is None:
        return True
    iteration, tools = marker
    if iteration < next_iteration:
        clear_turn_marker(marker_path)
        return True
    if not confirm(iteration, tools):
        return False
    clear_turn_marker(marker_path)
    return True


def snapshot_head_mismatch(
    snapshot_path: Path, repo_root: Path, *, chain_ref: str
) -> tuple[str, str] | None:
    """(snapshot head, resume-onto head) when the chain resume would continue
    on DIVERGED from the run's last snapshot, else None.

    The head compared is the one resume will commit on top of: the chain ref's
    current value (`refs/agent6/<id>/head`); an unborn ref resumes from the snapshot
    head itself, so there is nothing to compare.

    Divergence, not mere movement: the run's own per-step commits advance the
    chain forward from the snapshot between snapshot writes (a turn commits,
    then a review/metric call runs before the next snapshot), so a kill in that
    window leaves the tip ahead of the recorded head_sha on the SAME line. That
    must resume cleanly. Only refuse when the tip is not a descendant of the
    snapshot head -- someone rewrote or replaced the chain ref -- i.e. the
    model would resume against a record that changed under it. Working-tree
    (uncommitted) divergence is not checked; only committed history.

    Best-effort: the snapshot records head_sha as "" when git was unreadable at
    write time (skip), a corrupt snapshot file is left for the loud
    resume-snapshot load to report (skip), and a non-repo raises nothing here
    (the caller's require_git_repo already ran).
    """
    snap_head = ""
    with contextlib.suppress(OSError, ValueError):
        loaded = json.loads(snapshot_path.read_text(encoding="utf-8"))
        if isinstance(loaded, dict):
            # Raw single-key peek (must not raise); "head_sha" is
            # SessionSnapshot.head_sha -- keep in sync on a field rename.
            snap_head = str(loaded.get("head_sha") or "")
    if not snap_head:
        return None
    try:
        current_head = chain_tip(repo_root, chain_ref)
    except GitError:
        return None
    if current_head is None or current_head == snap_head:
        return None
    if is_ancestor(repo_root, snap_head, current_head):
        # The tip moved forward from the snapshot on the same line (the run's
        # own commits): not divergence.
        return None
    return (snap_head, current_head)


def leg_gate_origin(*, configured: bool, has_gate: bool, pinned: str) -> str:
    """Where THIS leg's gate came from: config outranks the run's pin, the pin
    stands when the leg reused it (an adopted gate stays adopted), and a leg
    that had to re-infer says so. A gateless leg claims nothing, even when
    config named a gate the leg then dropped."""
    if not has_gate:
        return ""
    if configured:
        return "configured"
    return pinned or "inferred"


def resume_task(  # noqa: PLR0911, PLR0912, PLR0915
    config_path: Path | None,
    session_id: str,
    *,
    frontend: SessionFrontend,
    force: bool,
    tui: bool = False,
    budget_overrides: BudgetOverrides | None = None,
    sandbox_overrides: SandboxOverrides | None = None,
    preset: str = "",
    steer: str = "",
    interactive: bool = False,
    reporter: Reporter = STDIO_REPORTER,
) -> int:
    """Resume a paused/crashed run from its snapshot.

    Mirrors `run_task` setup but uses the existing run id, refuses
    if no `loop_state.json` snapshot exists, and calls `wf.resume()`
    instead of `wf.run(task)`. A safety check refuses when the
    workspace HEAD DIVERGED from the snapshot (a rebase/reset/commit on
    another line); plain forward movement on the same line resumes
    cleanly. `--force` overrides the refusal.

    NOTE: token budget on resume is a FRESH ceiling, not a continuation
    of the prior run's accounting. Each `agent6 resume` invocation
    starts at 0 against the `[budget]` ledgers. This is by design - the budget is a per-
    invocation runaway-cost circuit breaker.

    Runs from the repository (the process cwd: its state dir, config, and
    the cwd a detached continuation spawns in). A fork's leg drives the
    fork's own worktree instead (`manifest.worktree`), handed to every step
    as *cwd*; the process cwd stays the repository.
    """
    repo = Path.cwd()
    state_dir = resolved_state_dir(repo)
    if steer.strip() and (problem := steer_problem(steer)) is not None:
        reporter.error(f"--steer: {problem}")
        return 2
    if not session_id:
        # "resume my last session" -- the common recovery case. Every bucket a
        # resumable mode writes to, so splitting plans/ out of runs/ does not
        # hide a plan from the bare form, and so the no-id path finds what the
        # by-id path below already accepts.
        buckets = resumable_bucket_dirs(state_dir)
        latest = newest_session_dir(buckets)
        if latest is None:
            reporter.err('nothing to resume yet. Start a session with `agent6 run "<task>"`.')
            return 2
        session_id = latest.name
        reporter.note(f"resuming most recent session: {session_id}")
    # Across buckets: an ask is a session like any other, so `agent6 resume`
    # continues one by id instead of only finding what lives under runs/.
    # One resolver, no per-bucket fallback: a runs/-only fallback would make an
    # id that prefixes BOTH a run and an ask silently pick the run.
    try:
        layout = resolve_session(state_dir, session_id)
    except SessionIdError as exc:
        reporter.error(str(exc))
        return 2
    session_id = layout.session_id
    # Read the manifest BEFORE taking the lock or clearing any state: resume
    # reaches every bucket, and clearing first would clobber a machine draft's
    # worker pid and pending answers on the way to discovering resume cannot
    # continue it.
    try:
        manifest = read_manifest(layout.session_dir)
        mode = manifest.session_mode()
    except ManifestError as exc:
        reporter.error(f"cannot resume {session_id}: {exc}")
        return 2
    cwd = manifest.worktree or repo
    if manifest.worktree is not None and manifest.worktree_git_dir is None:
        reporter.error(
            f"cannot resume {session_id}: its manifest names a worktree but not the repository"
            f" git dir it points into, so its jail cannot grant one; `agent6 fork {session_id}`"
            " continues its commits in a new worktree."
        )
        return 2
    if manifest.worktree is not None and not (cwd / ".git").exists():
        reporter.error(
            f"cannot resume {session_id}: its worktree {cwd} is gone (pruned or removed)."
            f" Its commits are on {manifest.run_branch or chain_ref_for(session_id)};"
            f" `agent6 fork {session_id}` continues them in a new worktree."
        )
        return 2
    # One authoritative writer per run dir (see acquire_single_writer). Refuse a
    # second resume of a still-live run before touching any shared state.
    worker_lock_fd = acquire_single_writer(layout.session_dir)
    if worker_lock_fd is None:
        reporter.err(SINGLE_WRITER_BUSY.format(rid=session_id))
        return 2
    # A run the agent ENDED has nothing to continue: the resumed leg spends a
    # call, answers in prose with no tool use, and records a silent_finish --
    # so a run that passed reads as failed afterwards, for a tree nobody
    # touched. New work is what --steer is for. Only this one reason: every
    # other ending (budget_exhausted, provider_error, steer_abort, a red
    # verify) is exactly what resume exists for. Read through the same fold the
    # listing uses, so the refusal and the status can never disagree.
    if not steer.strip() and finished_needs_new_work(layout.session_dir):
        reporter.refuse(needs_new_work_refusal(session_id))
        release_single_writer(worker_lock_fd)
        return 2
    # Drop a prior session's stale answer files (the id counters reset
    # on resume, an old answer must not be read instead of re-prompting).
    clear_pending_answers(layout.session_dir)
    if steer.strip():
        # --steer: queue the operator's follow-up as the first steering
        # instruction. Seeded AFTER the stale-state clear (which drops steer
        # files), so the loop's steer poll injects it at its first boundary.
        submit_steer(layout.session_dir, steer.strip())
    # Record this worker's pid so `agent6 sessions show` can probe liveness even while
    # the worker is blocked in a long provider call (which emits no events).
    write_worker_pid(layout.session_dir, os.getpid())

    detach_requested = False
    cfg: Config | None = None  # bound below; the finally reads it (detach away-mode)
    repo_lock_fd: int | None = None
    try:
        # The original run's manifest drives resume: `mode` (a plan run resumes
        # read-only with the plan tools, never as a write run), `preset` (unless
        # `--preset` picks another), `base_sha` (the review-panel diff base), and
        # `run_branch` (the visible ref the chain keeps advancing). Read FIRST: a
        # PARKED run (manifest carries parked_task, no snapshot exists) is
        # started fresh below instead of hitting the no-snapshot refusal.
        # `mode` is security-relevant: a damaged run dir (unreadable, corrupt, or
        # an unknown mode value) must NOT fall open to the more-privileged "run"
        # (write) mode. read_manifest / session_mode fail loud on any of those --
        # the underlying cause carries in the ManifestError detail -- rather than
        # silently escalating a plan run to a write run.
        role = session_kind(mode).role

        if manifest.parked_task:
            # Parked at submission: nothing ever ran, so "resume" is its fresh
            # start. Hand the verbatim saved task to
            # run_task under the same run id; it re-acquires both locks itself
            # (and re-parks with a fresh message if the checkout is STILL busy),
            # so release ours first. Its manifest rewrite clears parked_task.
            try:
                # replay_preset, not the raw stamped name: a config-selected
                # preset re-resolves from the same files, and handing its name
                # back would make _select_preset rank it as a flag (the same
                # rule as the snapshot-resume path below).
                cfg = load_session_config(
                    repo,
                    config_path,
                    mode=mode,
                    preset=preset or manifest.workflow.replay_preset,
                    budget_overrides=budget_overrides,
                    sandbox_overrides=sandbox_overrides,
                ).config
            except ConfigError as exc:
                reporter.error(str(exc))
                return 2
            why = f" ({manifest.parked_reason})" if manifest.parked_reason else ""
            reporter.note(f"run {session_id!r} was parked at submission{why}; starting it now.")
            saved_task = manifest.parked_task
            clear_worker_pid(layout.session_dir)
            release_single_writer(worker_lock_fd)
            worker_lock_fd = None
            return run_task(
                cfg,
                saved_task,
                frontend=frontend,
                session_id=session_id,
                mode=mode,
                budget_overrides=budget_overrides,
                sandbox_overrides=sandbox_overrides,
                preset=preset,
                # Pin the ORIGINAL stamp ONLY for a FLAG-selected preset whose
                # veto must survive, and only when this resume sets no --preset
                # of its own. A CONFIG-selected preset (from_flag False) re-
                # resolves from the CURRENT config below, so pinning the manifest's
                # old NAME would show a stale preset if the config changed since;
                # pass None and let run_task derive it from the re-resolved cfg,
                # like a fresh run. A resume that DOES set --preset is a fresh
                # flag choice, so run_task's own derivation stamps it.
                preset_stamp=(
                    (manifest.workflow.preset, True)
                    if (not preset and manifest.workflow.preset_from_flag)
                    else None
                ),
                # Hand --steer through: run_task's stale-state clear wipes
                # the bridge files seeded above, so a parked resume's follow-up
                # only survives as an argument.
                initial_steer=steer,
                reporter=reporter,
            )

        # One live run-mode worker per CHECKOUT (see acquire_repo_writer): a
        # resumed run drives the shared working tree exactly like a fresh one.
        # The lock lives with the checkout's state dir: a fork's worktree has
        # its own, so it never contends with the repository's checkout.
        if mode == "run":
            checkout_state_dir = resolved_state_dir(cwd)
            repo_lock_fd = acquire_repo_writer(checkout_state_dir, session_id)
            if repo_lock_fd is None:
                holder = repo_writer_holder(checkout_state_dir) or "another run"
                reporter.refuse(
                    f"run {holder!r} is already driving this checkout; a"
                    " second run-mode worker would interleave auto-commits on the"
                    " one working tree. Wait for it, or stop it first:\n"
                    f"    agent6 sessions stop {holder}"
                )
                return 2

        snapshot_path = layout.session_dir / "loop_state.json"
        if not snapshot_path.is_file():
            reporter.error(f"no resume snapshot at {snapshot_path}; nothing to resume.")
            return 2

        # ask is read-only and may run outside a git repo (agent6 self-help),
        # so a resumed ask skips the commit-oriented git preflight the same way
        # a fresh one does: the repo guard, the divergence guard (nothing it
        # would resume onto is code it wrote), the identity check and the run
        # branch. Otherwise `resume <ask-id>` would refuse with talk of
        # branches an ask never cuts.
        writes_code = mode != "ask"
        # The no-repo guard runs BEFORE any git-touching check (which would
        # otherwise print zeroed-out heads first, then the real error).
        if writes_code and not require_git_repo(cwd):
            return 2

        # Snapshot version guard in preflight: a v1 snapshot cannot be resumed,
        # and the refusal must land here (like `fork`) with the checkout and the
        # session network untouched, not after the preamble already printed.
        # wf.resume() re-validates the same snapshot; a corrupt/old file refuses
        # identically here (exit 1).
        try:
            snapshot = load_session_snapshot(snapshot_path)
        except (ValueError, OSError) as exc:
            reporter.error(str(exc))
            return 1
        if not turn_replay_allowed(
            layout.session_dir, snapshot.next_iteration, frontend.confirm_replay_after_crash
        ):
            reporter.err(
                "Not resuming: inspect the working tree and the run log first"
                " (the marker stays; resume again and answer yes to replay)."
            )
            return 2
        manifest_preset = manifest.workflow.replay_preset
        resume_base_sha = manifest.base_sha
        run_branch = manifest.run_branch or ""

        # Safety check: refuse when the chain resume would continue on DIVERGED
        # from the run's last snapshot (a rewritten or replaced chain ref would
        # leave the model reasoning about a record that changed under it);
        # plain forward movement on the same line -- the run's own per-step
        # commits -- resumes cleanly. The snapshot records head_sha best-effort
        # ("" when git was unreadable at write time); skip the check then, and
        # let the loud snapshot load below handle a corrupt file.
        mismatch = (
            snapshot_head_mismatch(snapshot_path, cwd, chain_ref=chain_ref_for(session_id))
            if writes_code
            else None
        )
        if mismatch is not None:
            snap_head, onto_head = mismatch
            reporter.err(
                "GUARD: the code this run would resume onto diverged from its last snapshot."
            )
            reporter.err(f"  snapshot head: {snap_head}")
            reporter.err(f"  resume onto:   {onto_head}")
            if not force:
                reporter.err("REFUSING to resume. Re-run with --force to override.")
                return 2

        try:
            effective = load_session_config(
                repo,
                config_path,
                mode=mode,
                preset=preset or manifest_preset,
                budget_overrides=budget_overrides,
                sandbox_overrides=sandbox_overrides,
            )
        except ConfigError as exc:
            reporter.error(str(exc))
            return 2
        cfg, explicit_leaves = effective.config, effective.explicit_leaves
        if preset:
            # This leg and every later one run under the operator's new
            # choice; the stamp is what listings show and resume replays.
            stamp_preset(layout.session_dir, preset)

        # Needs the config: "approve everything while away" is a grant per
        # scope, and the scopes in play include one per configured MCP server.
        if sys.stdin.isatty():  # a foreground start clears a stale detach away-mode
            clear_away_mode(layout.session_dir)
        else:
            # A front-end (web/TUI) or a detach spawns resume with no terminal; honor
            # AGENT6_DETACHED_AWAY so ask_user/approvals WAIT for a viewer instead of
            # fabricating an empty answer. Mirrors run_task; a pure headless resume
            # (CI) sets no env, so this is a no-op and keeps the non-hanging default.
            apply_spawned_away_default(layout.session_dir, approval_scopes(cfg))

        try:
            isolation = select_isolation(
                cfg,
                cwd=cwd,
                confirm_unconfined=frontend.confirm_unconfined_autorun,
                reporter=reporter,
                explicit_leaves=explicit_leaves,
                worktree_git_dir=manifest.worktree_git_dir,
            )
        except SessionRefused as refusal:
            return refusal.rc

        missing = check_provider_keys(cfg)
        if missing is not None:
            reporter.err(missing)
            return 2

        identity = CommitIdentity(name=cfg.git.commit.name, email=cfg.git.commit.email)
        # (no-repo guard already ran above, before the resume head guard)
        if writes_code:
            try:
                verify_git_identity(cwd, identity)
            except GitError as exc:
                reporter.error(str(exc))
                return 2

        transcript_sink = TranscriptSink(layout.transcripts_dir)
        events = EventSink(layout.logs_path)

        warn_install_inside_workspace(cwd, reporter=reporter)
        for line in agents_md_notices(cwd):
            reporter.note(line)

        tui_enabled = frontend.should_spawn_tui(tui, False, mode)
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

        def _gate(cfg: Config, _budget: BudgetTracker) -> Config:
            # Resume reuses the verify command the ORIGINAL run resolved
            # (stored in the snapshot), so the tool list, prompt, and commit
            # branch stay consistent with the frozen system prompt -- never
            # re-inferring, which could flip and diverge. Config the operator
            # has pinned since outranks it (announced below, and to the worker,
            # since the prompt still names the old one). `()` means the
            # original run was gateless: stay gateless.
            leg_configured = bool(cfg.workflow.verify_command)
            if not leg_configured and snapshot.verify_command:
                cfg = cfg.with_verify_command(snapshot.verify_command)
                gate = " ".join(snapshot.verify_command)
                reporter.note(f"reusing this run's verify command: {gate}")
            # The same leg-start decision a fresh run makes, LAST so nothing
            # hands the gate back: a leg that cannot run a command cannot run
            # its gate, so it is gateless rather than unwinnable. Frozen here,
            # with the system prompt.
            cfg = drop_gate_if_unrunnable(cfg, session_dir=layout.session_dir, reporter=reporter)
            # Re-pin for this leg: config outranks the pin, the pin outranks a
            # re-inference, and the manifest has to say which one this leg used.
            pinned_origin, pinned_gate = "", ()
            with contextlib.suppress(ManifestError, OSError):
                pinned = read_manifest(layout.session_dir).workflow
                pinned_origin, pinned_gate = pinned.verify_origin, pinned.verify_command
            if tuple(pinned_gate) != cfg.workflow.verify_command:
                # Both directions, including none -> gate: the frozen system
                # prompt names the OLD gate either way, so the operator has to
                # know which command is now judging the run.
                was = " ".join(pinned_gate) or "none"
                now = " ".join(cfg.workflow.verify_command) or "none"
                reporter.note(f"this run's verify gate changed: was {was}, now {now}")
            pin_gate(
                layout.session_dir,
                cfg.workflow.verify_command,
                leg_gate_origin(
                    configured=leg_configured,
                    has_gate=bool(cfg.workflow.verify_command),
                    pinned=pinned_origin,
                ),
                events=events,
                reporter=reporter,
            )
            return cfg

        def _undo_forker() -> tuple[str, str] | None:
            # Lazy: app.fork imports this module (see run.py's twin).
            from agent6.app.fork import undo_fork  # noqa: PLC0415

            return undo_fork(config_path, session_id, cwd=repo, reporter=reporter)

        end = run_leg(
            cfg,
            layout,
            LegInputs(
                session_id=session_id,
                mode=mode,
                role=role,
                isolation=isolation,
                tui_enabled=tui_enabled,
                interactive=interactive,
                task=None,
                gate=_gate,
                chain_branch=run_branch or None,
                base_sha=resume_base_sha,
                untracked_at_start=read_untracked_at_start(layout.session_dir),
                resume_state_path=snapshot_path,
                undo_forker=_undo_forker,
                # The follow-up this leg answered, not the run's original task:
                # a `--steer` question that never appeared made the second
                # answer read as more of the answer to the first.
                ask_transcript_task=steer.strip() or manifest.user_task,
                budget_overrides=budget_overrides,
                sandbox_overrides=sandbox_overrides,
                resuming=True,
                worktree_git_dir=manifest.worktree_git_dir,
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
        # Single owner of worker.pid for every resume exit path, refusals and
        # Ctrl-C during verify inference included.
        frontend.close_console_view()  # stop the heartbeat thread, clear any spinner line
        clear_worker_pid(layout.session_dir)
        release_single_writer(repo_lock_fd)
        release_single_writer(worker_lock_fd)
        if detach_requested and cfg is not None:
            detach_to_background(
                frontend=frontend,
                cfg=cfg,
                layout=layout,
                cwd=repo,
                flags=override_flags(budget_overrides, sandbox_overrides),
                reporter=reporter,
            )
