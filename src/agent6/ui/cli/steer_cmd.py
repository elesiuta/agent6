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


def _cmd_steer(target: str, text: str) -> int:
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
    submit_steer(layout.session_dir, text)
    print(f"steer queued for {layout.session_id}; picked up at the next iteration boundary.")
    return 0
