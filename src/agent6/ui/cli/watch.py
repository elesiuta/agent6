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

from agent6.machine import JournalError, MachineError, load_machine
from agent6.paths import state_dir
from agent6.sessions.id import SessionIdError
from agent6.ui.cli._common import (
    error,
    resolve_session_layout,
    resolve_target,
)
from agent6.ui.cli.machine_cmds import _cmd_machine_watch, machine_instance_root
from agent6.ui.cli.plan_watch import _cmd_watch
from agent6.viewmodel import (
    machine_snapshot,
    session_snapshot,
)


def _run_intent(repo_root: Path, target: str) -> tuple[bool, str | None]:
    """Resolve *target* against the run-style buckets (sessions/runs, /plans, /asks, /machines).
    Returns (is_run, run_error): (True, None) it resolves; (False, None) no
    match, so the caller may try a machine; (False, msg) an ambiguous prefix
    or a husk, a run-intent error the caller surfaces rather than falling
    through to machine lookup."""
    try:
        resolve_session_layout(repo_root, target)
    except SessionIdError as exc:
        return (False, None) if exc.no_match else (False, str(exc))
    return (True, None)


def _machine_json_snapshot(machine_dir: Path) -> int:
    """Print a machine's snapshot as one JSON object (`viewmodel.machine_snapshot`)."""
    try:
        snap = machine_snapshot(machine_dir)
    except JournalError as exc:  # a MachineError too: the corrupt-journal wording first
        error(f"{exc}")
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
        error(f"{e}")
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

    # An ambiguous run prefix or a husk is a run-intent error: surface it
    # rather than falling through to machine lookup and printing "no match".
    is_run, run_error = (True, None) if not target else _run_intent(cwd, target)
    if run_error is not None:
        error(f"{run_error}")
        return 2

    # Empty target, or one that resolves to a run id: watch the run.
    if is_run:
        if not json_out:
            return _cmd_watch(target, tui=tui, since=since, raw=raw, config_path=config_path)
        layout = resolve_target(target)
        if layout is None:
            return 2
        session_dir = layout.session_dir
        # The wire form the web serves, the merged claim checked against the repo.
        print(json.dumps(session_snapshot(session_dir, repo=cwd)))
        return 0

    # Else a machine by name.
    machine_dir = machine_instance_root(target, cwd)
    if machine_dir is not None and machine_dir.is_dir():
        if json_out:
            return _machine_json_snapshot(machine_dir)
        return _machine_watch_tui(machine_dir) if tui else _cmd_machine_watch(target)

    error(f"no run or machine matches {target!r} (looked under {state_dir(cwd)})")
    return 2
