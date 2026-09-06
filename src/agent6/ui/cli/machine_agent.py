# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Subprocess entry: run ONE machine `agent` state.

Invoked as `python -m agent6.ui.cli.machine_agent <request.json> <result.json>`.
Validates the request (`MachineAgentRequest`, the file-shape owner), runs the
agent loop (`agent6.app.machine_agent.run_one`) injecting the live-view console,
and writes the `AgentExecResult` result. The engine enforces the timeout by
killing this process, which gives true mid-call cancellation.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from agent6.app.machine_agent import MachineAgentRequest, run_one
from agent6.events import EventSink, EventWriteError
from agent6.ui.cli._common import error
from agent6.ui.cli._console_view import ConsoleView


def _attach_console(events: EventSink) -> None:
    """At a TTY (or when AGENT6_FORCE_STREAM=1), render the live conversation to
    stderr so `machine create` and live `agent` states are watchable; consumes
    the same events the per-state sink records."""
    if sys.stderr.isatty() or os.environ.get("AGENT6_FORCE_STREAM") == "1":
        events.subscribe(ConsoleView(sys.stderr))


def main() -> int:
    req = MachineAgentRequest.model_validate_json(Path(sys.argv[1]).read_bytes())
    try:
        out = run_one(req, attach_console=_attach_console)
    except EventWriteError as exc:
        # This subprocess never reaches the CLI dispatch backstop, so without
        # this an unwritable per-state journal dumped a raw traceback at the
        # operator and the engine reduced the state to a bare "error".
        error(f"{exc}")
        return 1
    Path(sys.argv[2]).write_text(out.model_dump_json(), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
