# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Stdlib JSONL tail-follower. No third-party deps."""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any


def tail_events(
    path: Path,
    *,
    poll_s: float = 0.25,
    follow: bool = True,
    stop_when_finished: bool = False,
    should_stop: Callable[[], bool] | None = None,
    start_at_end: bool = False,
    on_position: Callable[[int], None] | None = None,
) -> Iterator[dict[str, Any]]:
    """Yield JSON-decoded events from *path* as they are appended.

    *on_position* hears the byte offset an event ends at, before that event
    is yielded, so a caller handling the event can order its own lines
    against the journal read so far.

    - Waits for the file to appear (up to forever if follow=True).
    - Yields each existing line on startup, then tails for new ones.
    - If *stop_when_finished* is true, exits after a `session.end` event.
    - If *should_stop* is given, exits at the next poll boundary once it returns
      True (lets a caller cancel a follow, e.g. on client disconnect).
    - If *follow* is false, yields existing lines and returns.
    - If *start_at_end* is true, existing lines are skipped and tailing starts
      at the file's current end -- for a resumed run's journal, which already
      holds the prior legs a viewer has seen.
    - Skips malformed JSON lines silently (the writer may have a partial
      write in flight; we'll pick it up on the next poll).

    Reads bytes and splits on b"\\n" before decoding: writers flush long lines
    in multiple syscalls, so a poll can hit EOF mid multibyte UTF-8 sequence and
    a text-mode read() would raise UnicodeDecodeError. Only complete lines are
    decoded; the byte tail stays pending until its newline arrives.
    """
    while follow and not path.exists():
        if should_stop is not None and should_stop():
            return
        time.sleep(poll_s)
    if not path.exists():
        return

    pos = journal_size(path) if start_at_end else 0
    pending = b""
    heard = on_position or _ignore_position
    final_drain = False  # should_stop fired: read what is already appended, then stop
    while True:
        if should_stop is not None and not final_drain and should_stop():
            # The writer is gone (a dead worker) or the caller is leaving: what
            # sits in the file is final, so hand it over before returning. A
            # worker that finishes and exits within one poll otherwise had its
            # last events (the finish, session.end) unread, and `attach`
            # stopped one step short of the run's end.
            final_drain = True
        try:
            with path.open("rb") as fh:
                fh.seek(pos)
                chunk = fh.read()
                pos = fh.tell()
        except FileNotFoundError:
            if not follow or final_drain:
                return
            time.sleep(poll_s)
            continue

        if chunk:
            base = pos - len(chunk) - len(pending)  # where the first line below starts
            parsed, pending = _complete_lines(pending + chunk, base)
            # stop_when_finished halts at a session.end only when nothing follows it
            # in this batch: a resume appends events AFTER a session.end (a stopped
            # run's steer_abort, or the resume of a finished one), and stopping
            # at that superseded end would silently drop everything the resumed
            # run does. A live run's real end is the batch's last event.
            for i, (end, evt) in enumerate(parsed):
                heard(end)
                yield evt
                if stop_when_finished and i == len(parsed) - 1 and evt.get("type") == "session.end":
                    return

        if not follow or final_drain:
            evt = _parse_event_line(pending)
            if evt is not None:
                heard(pos)
                yield evt
            return
        time.sleep(poll_s)


def _ignore_position(_end: int) -> None:
    return None


def _complete_lines(buffer: bytes, base: int) -> tuple[list[tuple[int, dict[str, Any]]], bytes]:
    """*buffer*'s complete lines as (end offset, event), malformed ones skipped,
    and the trailing fragment; *base* is the offset the buffer starts at."""
    lines = buffer.split(b"\n")
    parsed: list[tuple[int, dict[str, Any]]] = []
    end = base
    for raw in lines[:-1]:
        end += len(raw) + 1
        if (event := _parse_event_line(raw)) is not None:
            parsed.append((end, event))
    return parsed, lines[-1]


def journal_size(path: Path) -> int:
    """The journal's size in bytes, 0 when it does not exist yet."""
    try:
        return path.stat().st_size
    except OSError:
        return 0


def _parse_event_line(line: bytes) -> dict[str, Any] | None:
    if not line.strip():
        return None
    try:
        evt = json.loads(line.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    return evt if isinstance(evt, dict) else None


class LogTail:
    """Incremental logs.jsonl reader for a UI poll loop. Each `read` returns the
    events appended since the last call (byte-offset based, tolerant of a partial
    line at EOF). One reader follows a run and its same-dir resume; cheaper than
    re-reading the whole file every tick."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._pos = 0
        self._pending = b""

    def read(self) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        try:
            with self._path.open("rb") as fh:
                fh.seek(self._pos)
                chunk = fh.read()
                self._pos = fh.tell()
        except OSError:
            return out
        parsed, self._pending = _complete_lines(self._pending + chunk, 0)
        return [evt for _end, evt in parsed]
