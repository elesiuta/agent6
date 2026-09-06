# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The unified `agent6 attach <target>`: follow a run or a machine, live.

Resolves <target> to a run (id or unique prefix) or a machine (by name) and
dispatches to the right viewer. Both default to a plain CLI stream (a run is a
no-deps line tail of logs.jsonl; a machine streams its state overview +
reasoning); `--tui` opens the full-screen dashboard instead. `--json` prints a
one-shot snapshot of the folded state, the same wire form a web client reads. An
empty target watches the most recent run. A target that is both a run prefix and
a machine name resolves as the run.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from agent6.config.layer import resolved_state_dir
from agent6.machine import JournalError, MachineError, load_machine
from agent6.sessions.id import SessionIdError
from agent6.sessions.layout import machines_root
from agent6.ui.cli._common import (
    _runs_dir,
    print_no_session_match,
    resolve_session_layout,
)
from agent6.ui.cli.machine_cmds import _cmd_machine_watch
from agent6.ui.cli.plan_watch import _cmd_watch, _resolve_session_dir
from agent6.viewmodel import (
    machine_snapshot,
    session_snapshot,
)


def _run_intent(repo_root: Path, target: str) -> tuple[bool, str | None]:
    """Resolve *target* against the run-style buckets (sessions/runs, /plans, /asks, /machines).
    Returns (is_run, ambiguity_error): (True, None) it resolves; (False, None) no
    match, so the caller may try a machine; (False, msg) it is an ambiguous
    prefix, a run-intent error the caller should surface rather than fall
    through to machine lookup."""
    try:
        resolve_session_layout(repo_root, target)
    except SessionIdError as exc:
        return (False, str(exc)) if exc.ambiguous else (False, None)
    return (True, None)


def _session_json_snapshot(session_dir: Path, repo: Path) -> int:
    """Print a session's snapshot as one JSON object (the wire form the web
    serves; `viewmodel.session_snapshot`, the merged claim checked against
    *repo* like the web's)."""
    print(json.dumps(session_snapshot(session_dir, repo=repo)))
    return 0


def _machine_json_snapshot(machine_dir: Path) -> int:
    """Print a machine's snapshot as one JSON object (`viewmodel.machine_snapshot`)."""
    try:
        snap = machine_snapshot(machine_dir)
    except JournalError as exc:  # a MachineError too: the corrupt-journal wording first
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    except MachineError as exc:
        source = machine_dir / "machine.asm.toml"
        print(f"FAIL: {source}: {'; '.join(exc.problems)}", file=sys.stderr)
        return 1
    print(json.dumps(snap))
    return 0


def _machine_watch_tui(machine_dir: Path) -> int:
    """Open the full-screen MachineWatchScreen for a machine instance (`--tui`)."""
    source = machine_dir / "machine.asm.toml"
    try:
        spec = load_machine(source)
    except MachineError as exc:
        print(f"FAIL: {source}: {'; '.join(exc.problems)}", file=sys.stderr)
        return 1
    try:
        from agent6.ui.tui.machines import run_machine_watch_tui  # noqa: PLC0415
    except ImportError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        print("HINT: drop --tui for the plain text follow.", file=sys.stderr)
        return 3
    return run_machine_watch_tui(machine_dir, spec)


def _cmd_watch_target(  # noqa: PLR0911
    target: str,
    *,
    tui: bool,
    json_out: bool,
    since: int,
    raw: bool,
    config_path: Path | None = None,
) -> int:
    """Resolve *target* to a run or machine and follow it (or snapshot it)."""
    if since and not raw:
        # --since replays event lines, which only the --raw tail renders.
        print("agent6 attach: --since applies to --raw only.", file=sys.stderr)
        return 2
    cwd = Path.cwd()
    runs_dir = _runs_dir(cwd)
    machines_dir = machines_root(resolved_state_dir(cwd))

    # An ambiguous run prefix is a run-intent error: surface the disambiguation
    # rather than falling through to machine lookup and printing "no match".
    is_run, ambiguous = (True, None) if not target else _run_intent(cwd, target)
    if ambiguous is not None:
        print(f"ERROR: {ambiguous}", file=sys.stderr)
        return 2

    # Empty target, or one that resolves to a run id: watch the run.
    if is_run:
        if not json_out:
            return _cmd_watch(target, tui=tui, since=since, raw=raw, config_path=config_path)
        session_dir = _resolve_session_dir(cwd, target)
        if session_dir is None or not session_dir.is_dir():
            print_no_session_match(target, runs_dir.parent)
            return 2
        return _session_json_snapshot(session_dir, cwd)

    # Else a machine by name.
    machine_dir = machines_dir / target
    if machine_dir.is_dir():
        if json_out:
            return _machine_json_snapshot(machine_dir)
        return _machine_watch_tui(machine_dir) if tui else _cmd_machine_watch(target)

    print(
        f"ERROR: no run or machine matches {target!r} (looked under {runs_dir} and {machines_dir})",
        file=sys.stderr,
    )
    return 2
