# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The `/btw` runner every composer shares (CLI menu, TUI, web): spawn the
side question, deliver the answer.

The menu owns the grammar and `app.btw` owns the session; this owns the two
things only a front-end can do -- spawning through whatever escape the run
has from its namespace, and landing the finished answer in the run's journal,
which each surface renders at its next turn boundary.
"""

from __future__ import annotations

import os
import subprocess
import time
from collections.abc import Callable
from pathlib import Path
from threading import Thread

from agent6.app.btw import BtwLaunch, BtwSession, btw_answer, render_btw, start_btw
from agent6.events import EventSink
from agent6.sandbox.jail import keep_out_of_the_sweep
from agent6.sessions.layout import LOGS_NAME, bucket_dir, layout_of
from agent6.ui.spawn import agent6_exe

# How often the watcher looks for the answer. A btw is a short question, and
# the cost of a poll is one status fold off the session dir.
_POLL_S = 1.0
_GIVE_UP_S = 900.0


def direct_launch(cwd: Path, argv: list[str], env_extra: dict[str, str]) -> str:
    """Spawn `agent6 <argv>` detached, for a run with no namespace to escape.

    Fire-and-forget: the session dir appearing is the confirmation, exactly as
    for a `/parallel` lane.
    """
    try:
        proc = subprocess.Popen(
            [agent6_exe(), *argv],
            cwd=str(cwd),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
            env={**os.environ, **env_extra},
        )
    except OSError as exc:
        return f"could not start the btw: {exc}"
    # Our child, its own session: the escapee sweep would SIGKILL it at the
    # next background command's teardown otherwise.
    keep_out_of_the_sweep(proc.pid)
    return ""


def make_btw_runner(
    parent_id: str,
    *,
    launch: BtwLaunch,
    list_asks: Callable[[], list[Path]],
    events: EventSink,
) -> Callable[[str, Path], str]:
    """The `/btw <question>` handler the pause menu calls.

    Returns immediately with a line to print; the answer lands later as a
    `btw.answered` event on the run's journal. A btw never blocks the run --
    that is the whole point of asking beside it rather than steering it.
    """

    def run_btw(question: str, _session_dir: Path) -> str:
        session, err = start_btw(
            question, parent_id, cwd=Path.cwd(), launch=launch, list_asks=list_asks
        )
        if session is None:
            return f"[agent6] {err}"
        events.emit("btw.opened", btw_id=session.id, question=session.question)
        Thread(target=_watch, args=(session, events), daemon=True).start()
        return f"[agent6] btw {session.id} opened; its answer prints here when it lands"

    return run_btw


def _watch(session: BtwSession, events: EventSink) -> None:
    """Poll until the btw answers, then put the block on the run's journal.

    The journal, not the console view: under --tui or the web there is no
    console view, and an answer handed to a missing one was dropped after the
    model had already been paid for. Every surface folds the same log, and a
    parent that exits first leaves the answer on disk to read afterwards.

    Daemon thread: a btw must never hold the run open, and an unanswered one at
    exit is simply an ask the operator can resume.
    """
    deadline = time.monotonic() + _GIVE_UP_S
    while time.monotonic() < deadline:
        answer = btw_answer(session)
        if answer is not None:
            events.emit("btw.answered", btw_id=session.id, block=render_btw(session, answer))
            return
        time.sleep(_POLL_S)


def asks_dir(session_dir: Path) -> Path:
    """The asks bucket beside *session_dir*'s own, for `/btw`'s roster.

    Derived from the running session's dir rather than re-resolving the state
    base: the two must agree even when `XDG_STATE_HOME` relocates it.
    """
    return bucket_dir(layout_of(session_dir).state_dir, "asks")


def open_btw(session_dir: Path, question: str) -> tuple[bool, str]:
    """`/btw <question>` from any composer: open the side ask beside the live
    run at *session_dir*. Returns (opened, the line to show); the answer lands
    later as a `btw.answered` event on the run's journal, which every surface
    folds. The flag carries the outcome, so no caller parses the line.
    """
    if not question.strip():
        return False, "[agent6] ask something: `/btw <question>`"
    runner = make_btw_runner(
        session_dir.name,
        launch=direct_launch,
        list_asks=lambda: (
            [d for d in asks_dir(session_dir).iterdir() if d.is_dir()]
            if asks_dir(session_dir).is_dir()
            else []
        ),
        events=EventSink(session_dir / LOGS_NAME),
    )
    line = runner(question, session_dir)
    return " opened;" in line, line
