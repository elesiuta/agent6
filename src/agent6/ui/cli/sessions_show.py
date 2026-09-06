# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""`agent6 sessions show`: one-shot liveness + progress of a session from its
dir (worker.pid, the log scan, the manifest's branch facts), text or --json."""

from __future__ import annotations

import contextlib
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from agent6.config.layer import resolved_state_dir
from agent6.git_ops import branch_exists, chain_ref_for, chain_tip, merge_stamp_holds
from agent6.sessions.id import SessionIdError
from agent6.sessions.ipc import listening_ports, pid_alive, read_worker_pid, worker_is_alive
from agent6.sessions.layout import LOGS_NAME
from agent6.sessions.manifest import ManifestError, SessionManifest, read_manifest
from agent6.ui.cli._common import print_no_session_match, resolve_or_newest_layout
from agent6.viewmodel import (
    LogScan,
    existing_run_branch,
    scan_session_log,
    status_for_session_dir,
)
from agent6.viewmodel.format import (
    format_branch,
    format_compare,
    format_cost,
    format_lineage,
    status_label,
)


def _fmt_dur(seconds: float | None) -> str:
    if seconds is None:
        return "-"
    s = int(seconds)
    if s < 60:
        return f"{s}s"
    if s < 3600:
        return f"{s // 60}m{s % 60:02d}s"
    return f"{s // 3600}h{(s % 3600) // 60:02d}m"


def _print_fork_lineage(manifest: SessionManifest) -> None:
    """Print the fork-lineage line for a run created by `agent6 fork` (no-op
    otherwise)."""
    lineage = format_lineage(
        manifest.parent_session_id, manifest.forked_from_turn, manifest.forked_from_sha
    )
    if lineage:
        print(f"forked from: {lineage}")
    if manifest.worktree is not None:
        gone = "" if manifest.worktree.is_dir() else " (gone)"
        print(f"worktree:   {manifest.worktree}{gone}")


def _print_parallel_compare(manifest: SessionManifest) -> None:
    """Print the fan-out compare outcome for a lane (no-op for a non-lane run):
    where it placed, whether it won, judged or mechanical, and the judge's
    rationale when there is one."""
    formatted = format_compare(manifest.compare)
    if formatted is None:
        return
    headline, rationale = formatted
    print(f"compare:    {headline}")
    if rationale:
        print(f"  judge: {rationale}")


def _status_state(session_dir: Path, scan: LogScan, *, last_age: float | None) -> str:
    """The one-line state `sessions show` prints (and emits as --json "state").

    Leads with the LISTING's label -- `status_label` over
    `status_for_session_dir`, the one decision every surface feeds -- then
    appends this surface's diagnostic detail (what to do, or why the word
    applies). The pre-unification words lied twice: a crashed run led with
    "stopped" (the hub's word for an OPERATOR stop) and a launching run with
    "running" (the hub said "starting")."""
    word, reason = status_for_session_dir(session_dir, scan.status_facts())
    if scan.finished:
        # The raw end reason is the diagnostic; it is not repeated when the
        # label already carries it (an ask's word, a failure's reason).
        label = status_label(word, reason)
        return label if scan.end_reason in (word, reason) else f"{label} ({scan.end_reason})"
    detail = {
        "waiting": "needs answer; attach to respond",
        "stale": "no worker, no session.end: likely crashed or killed",
        "parked": f"{reason}; resume to start" if reason else "resume to start",
        # "no events yet" was claimed unconditionally, over logs that HAD
        # events (a worker that died launching writes preflight events).
        "created": "no events yet" if scan.last_type is None else "never started",
    }.get(word, "")
    if word == "running" and last_age is not None and last_age > 120:
        detail = "long step, likely a provider call"
    return f"{word} ({detail})" if detail else word


def _cmd_status(session_id: str, *, as_json: bool = False) -> int:
    """One-shot liveness + progress summary for a run, then exit (no follower).

    Answers "is this run still alive, and what is it doing?" from the run dir
    alone: the worker.pid (probed with signal 0, so liveness is known even while
    the worker is blocked in a long provider call that emits no events) plus the
    last event, current iteration, and elapsed time from logs.jsonl. For a quick
    or scripted check; `agent6 attach` is the live follower.
    """
    try:
        layout = resolve_or_newest_layout(Path.cwd(), session_id)
    except SessionIdError as exc:
        # An ambiguous prefix names its candidates (as attach and runs stop do);
        # swallowing it printed "no session matches <id>", which is false when
        # several do.
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    target = layout.session_dir if layout is not None else None
    if target is None or not target.is_dir():
        print_no_session_match(session_id, resolved_state_dir(Path.cwd()))
        return 2

    loaded: SessionManifest | None = None
    with contextlib.suppress(ManifestError):
        loaded = read_manifest(target)
    # A missing/corrupt manifest still renders (defaults), but `mode` reads "?"
    # rather than the model default so a manifest-less run isn't shown as "run".
    manifest = loaded or SessionManifest()
    mode_display = loaded.mode if loaded is not None else None

    logs = target / LOGS_NAME
    scan = scan_session_log(logs) if logs.is_file() else LogScan()

    pid = read_worker_pid(target)
    alive = worker_is_alive(target)
    last_age = (time.time() - scan.last_ep) if scan.last_ep is not None else None
    # A live run is still elapsing (a wait on the operator writes no event);
    # a finished or dead one stopped at its last event.
    elapsed = (
        ((time.time() if alive else scan.last_ep) - scan.start_ep)
        if scan.last_ep is not None and scan.start_ep is not None
        else None
    )

    state = _status_state(target, scan, last_age=last_age)

    driver = manifest.models.driver
    model = (driver.model if driver else "") or "?"
    compare_json = manifest.compare.model_dump(mode="json") if manifest.compare else None
    changes = _changes(target.name, manifest, undone=scan.finished and scan.end_reason == "undone")

    if as_json:
        print(
            json.dumps(
                {
                    "session_id": target.name,
                    "mode": mode_display,
                    "model": model,
                    "state": state,
                    "alive": alive,
                    "pid": pid,
                    "iteration": scan.iteration,
                    "last_event": scan.last_type,
                    "last_event_age_s": round(last_age, 1) if last_age is not None else None,
                    "elapsed_s": round(elapsed, 1) if elapsed is not None else None,
                    "reason": scan.end_reason if scan.finished else None,
                    "input_tokens": scan.input_tokens,
                    "output_tokens": scan.output_tokens,
                    "cost_usd": scan.cost_usd,
                    # cost_usd is an under-estimate when some spend was
                    # unpriced; the text render marks it, so the JSON must too.
                    "usd_partial": scan.usd_partial if scan.cost_usd is not None else None,
                    "parent_session_id": manifest.parent_session_id,
                    "forked_from_turn": manifest.forked_from_turn,
                    "forked_from_sha": manifest.forked_from_sha,
                    "worktree": str(manifest.worktree) if manifest.worktree else None,
                    "compare": compare_json,
                    "run_branch": existing_run_branch(manifest, Path.cwd()) or None,
                    "base_branch": manifest.base_branch or None,
                    "merged_into": changes.merged_into or None,
                    "pins": list(scan.pins),
                }
            )
        )
        return 0

    pid_note = ""
    if alive:
        pid_note = f"  (worker pid {pid} alive)"
    elif pid is not None and not scan.finished:
        # Liveness matches the recorded start time, so a pid the OS has since
        # handed to something else reads dead -- correctly. Saying "not running"
        # about a number the operator can look up is still false.
        pid_note = (
            f"  (worker pid {pid} was recycled)"
            if pid_alive(pid)
            else f"  (worker pid {pid} not running)"
        )
    print(f"session:    {target.name}  (mode={mode_display or '?'})")
    _print_fork_lineage(manifest)
    _print_parallel_compare(manifest)
    print(f"model:      {model}")
    print(f"state:      {state}{pid_note}")
    print(f"iteration:  {scan.iteration if scan.iteration is not None else '-'}")
    print(
        f"last event: {scan.last_type or '-'}"
        f"{f'  ({_fmt_dur(last_age)} ago)' if last_age is not None else ''}"
    )
    print(f"elapsed:    {_fmt_dur(elapsed)}")
    if scan.input_tokens is not None or scan.cost_usd is not None:
        # Token counters are per-leg, cost is banked across legs; on a resumed
        # run say so, or $0.03 next to the last leg's 10k tokens reads wrong.
        leg_s = " (latest leg)" if scan.legs > 1 else ""
        cost_s = (
            f"  cost {format_cost(scan.cost_usd, partial=scan.usd_partial)}"
            + (f" (all {scan.legs} legs)" if scan.legs > 1 else "")
            if scan.cost_usd is not None
            else ""
        )
        tokens = f"in={scan.input_tokens or 0} out={scan.output_tokens or 0}"
        print(f"usage:      {tokens}{leg_s}{cost_s}")
    if changes.line:
        print(f"changes:    {changes.line}")
    for i, pin in enumerate(scan.pins):
        print(f"{'pins:' if i == 0 else '':<12}{pin}")
    _print_listening_ports(target)
    _print_task_tree(target)
    return 0


@dataclass(frozen=True, slots=True)
class _Changes:
    line: str  # the text row; "" for a session with no run branch
    merged_into: str  # the base the run branch is merged into, else ""


def _changes(session_id: str, manifest: SessionManifest, *, undone: bool) -> _Changes:
    """Where the run's work lives, checked against git as the end-of-run
    footer checks it: merged into the base (the stamp still describes the
    branch), on the run branch awaiting `sessions merge`, on the hidden chain
    ref alone (the branch deleted, the commits kept), a branch no commit
    ever reached, or taken back by /undo (*undone*: no merge to offer, as
    the listing marks none)."""
    run_branch = manifest.run_branch or ""
    if not run_branch:
        return _Changes("", "")
    if undone:
        return _Changes(f"{run_branch} (taken back by /undo)", "")
    cwd = Path.cwd()
    stamp = manifest.merged
    if stamp is not None and merge_stamp_holds(cwd, run_branch, stamp.tip):
        into = stamp.into or manifest.base_branch
        return _Changes(format_branch(run_branch, manifest.base_branch, into), into)
    merge_hint = f"merge with: agent6 sessions merge {session_id}"
    if not branch_exists(cwd, run_branch):
        chain = chain_ref_for(session_id)
        if chain_tip(cwd, chain) is None:
            return _Changes(f"{run_branch} (no commits)", "")
        return _Changes(f"{chain} ({run_branch} is gone; the commits are kept); {merge_hint}", "")
    return _Changes(f"{format_branch(run_branch, manifest.base_branch, '')}; {merge_hint}", "")


def _print_listening_ports(session_dir: Path) -> None:
    """What the run is serving, and how to reach it.

    A run's commands share a network with no way in from outside, so a dev
    server the agent started is invisible here -- including the port it is on.
    This is where someone asks "what is it doing", so it is where the answer
    belongs, with the command that opens it.
    """
    with contextlib.suppress(Exception):
        ports = listening_ports(session_dir)
        if not ports:
            return
        listed = ", ".join(str(p) for p in ports)
        print(f"serving:    {listed} (inside the run)")
        print(f"            open one: agent6 forward {session_dir.name} {ports[0]}")


def _print_task_tree(session_dir: Path) -> None:
    """Show the run's task DAG when it decomposed into subtasks. Makes the plan
    visible for a headless run (no TUI #plan pane), the decompose case the user
    could not see. A single root (no decomposition) is not worth the block."""
    from agent6.graph.storage import load_graph  # noqa: PLC0415
    from agent6.sessions.layout import layout_of  # noqa: PLC0415
    from agent6.ui.cli._task_tree import task_tree_lines  # noqa: PLC0415

    with contextlib.suppress(Exception):
        layout = layout_of(session_dir)
        nodes = load_graph(layout)
        if len(nodes) <= 1:
            return
        lines = task_tree_lines(nodes, show_commit=True)
        if lines:
            print("\nplan:")
            for line in lines:
                print(f"  {line}")
