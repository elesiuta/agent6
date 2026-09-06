# SPDX-License-Identifier: Apache-2.0
"""Regression tests for clear_pending_answers (cli/ui bridge bugs #7, #22).

#7: a leftover `steer.request` marker from a prior session must be dropped at
    run/resume START, else the resumed run stalls on a phantom steer prompt.
#22: `frontend.pid` must only be cleared when NO live TUI owns it, so a concurrently
    live `agent6 attach` watcher keeps bridging approval/question modals.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

from agent6.sessions.ipc import (
    clear_pending_answers,
    frontend_is_live,
    register_frontend,
    request_steer,
    steer_request_pending,
    unregister_frontend,
)


def test_clear_pending_drops_leftover_steer_request(tmp_path: Path) -> None:
    session_dir = tmp_path / "run"
    session_dir.mkdir()
    request_steer(session_dir)
    assert steer_request_pending(session_dir)

    clear_pending_answers(session_dir, before=time.time() + 60)

    # The phantom steer marker must be gone so resume doesn't stall.
    assert not steer_request_pending(session_dir)


def test_clear_pending_preserves_live_frontend_claims(tmp_path: Path) -> None:
    session_dir = tmp_path / "run"
    session_dir.mkdir()
    # Our own pid is a live process => a live foreign watcher.
    register_frontend(session_dir, os.getpid())
    assert frontend_is_live(session_dir)

    clear_pending_answers(session_dir, before=time.time() + 60)

    # A live watcher's claim must survive so its modals stay wired up.
    assert frontend_is_live(session_dir)


def test_dead_frontend_claims_are_pruned_by_the_liveness_probe(tmp_path: Path) -> None:
    session_dir = tmp_path / "run"
    session_dir.mkdir()
    dead_pid = _find_dead_pid()
    register_frontend(session_dir, dead_pid)
    # A hard-killed front-end's claim reads not-live and is pruned in passing,
    # so the answer-poll never blocks on it and the dir stays tidy.
    assert not frontend_is_live(session_dir)
    assert not (session_dir / "frontends" / str(dead_pid)).exists()


def test_concurrent_frontends_do_not_deregister_each_other(tmp_path: Path) -> None:
    """The single-slot frontend.pid let one front-end's exit strand another
    (attach claims -> web clobbers -> web releases -> attach deregistered, its
    answers never read). One claim file per front-end kills the class: any
    number watch concurrently and each removes only its own claim."""
    session_dir = tmp_path / "run"
    session_dir.mkdir()
    attach_pid = os.getpid()
    web_pid = _find_dead_pid()  # stands in for a second front-end's pid slot
    register_frontend(session_dir, attach_pid)
    register_frontend(session_dir, web_pid)
    unregister_frontend(session_dir, web_pid)  # the browser closes
    assert frontend_is_live(session_dir)  # the attach watcher keeps bridging
    unregister_frontend(session_dir, attach_pid)
    assert not frontend_is_live(session_dir)
    # Unregistering an absent claim is a no-op.
    unregister_frontend(session_dir, attach_pid)


def _find_dead_pid() -> int:
    for candidate in range(2_000_000, 2_000_100):
        try:
            os.kill(candidate, 0)
        except ProcessLookupError:
            return candidate
        except PermissionError:
            continue
    # Fallback: very unlikely to be reached.
    return 2_000_000


def test_clear_pending_keeps_what_was_written_since_the_previous_leg(tmp_path: Path) -> None:
    """The sweep dropped every marker, including a stop an editor requested
    while the leg was still coming up (the run then ran on, spending budget,
    while the turn reported "cancelled"). With `before`, only files older than
    it are stale."""
    import os

    from agent6.sessions.ipc import request_stop, stop_request_pending, write_answer

    session_dir = tmp_path / "run"
    session_dir.mkdir()
    leg_end = 1_700_000_000.0
    write_answer(session_dir, "approval-1", "yes")
    old = session_dir / "approvals" / "approval-1.answer"
    assert old.is_file()
    os.utime(old, (leg_end - 10, leg_end - 10))
    request_stop(session_dir)
    os.utime(session_dir / "stop.request", (leg_end + 1, leg_end + 1))

    clear_pending_answers(session_dir, before=leg_end)

    assert not old.exists()
    assert stop_request_pending(session_dir)
    clear_pending_answers(session_dir, before=time.time() + 60)
    assert not stop_request_pending(session_dir)
