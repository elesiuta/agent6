# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""`agent6 steer`: queue a steering instruction for a live run.

Wraps the one steer channel every front-end uses (`sessions.ipc.submit_steer`),
so scripts and cron jobs can drive a running session. A session that is not
running refuses, naming `resume --steer`.
"""

from __future__ import annotations

import sys
from pathlib import Path

from agent6.sessions.id import SessionIdError
from agent6.sessions.ipc import submit_steer, worker_is_alive
from agent6.ui.cli._common import resolve_session_layout
from agent6.viewmodel.listing import summarize_session_dir


def _cmd_steer(target: str, text: str, *, now: bool = False) -> int:
    try:
        layout = resolve_session_layout(Path.cwd(), target)
    except SessionIdError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    if not worker_is_alive(layout.session_dir):
        print(
            f"REFUSING: session {layout.session_id} is not running; a steer needs a"
            f" live run. Queue one for its next leg instead:"
            f" agent6 resume {layout.session_id} --steer TEXT",
            file=sys.stderr,
        )
        return 2
    submit_steer(layout.session_dir, text, now=now)
    picked = (
        "an in-flight model call is interrupted to take it"
        if now
        else "it lands at the next step boundary (--now interrupts the in-flight call)"
    )
    print(f"steer queued for {layout.session_id}: {picked}.")
    summary = summarize_session_dir(layout.session_dir)
    if summary.status == "waiting" and summary.reason:
        # Parked on an operator prompt: no boundaries arrive and no steer
        # (--now included) can break that wait; only the answer can.
        # `agent6 answer` takes a question; an approval needs a front-end.
        how = (
            f"agent6 answer {layout.session_id}"
            if summary.reason.startswith("question")
            else f"agent6 attach {layout.session_id}"
        )
        print(
            f"note: the run is waiting ({summary.reason}); the steer stays"
            f" queued until that is answered: {how}"
        )
    return 0
