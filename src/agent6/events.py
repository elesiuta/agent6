# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Structured JSONL event sink.

Emits one JSON object per line to `<run-dir>/logs.jsonl` (the run dir under the
per-repo state dir), so a front-end or an external tool follows a run by tailing
one file instead of parsing the freeform `print` log.

Design notes:
- Write-only and append-only. No reads, no rotation, no schema validation,
  consumers should be defensive.
- Each call opens, writes one line, flushes, closes. Durable events fsync too;
  the high-frequency streaming deltas (see `_EPHEMERAL_EVENTS`) only flush, so
  a reasoning model's tens of thousands of deltas don't fsync-throttle the run.
- Durable events fail LOUD (`EventWriteError`): the journal is the read model
  every surface trusts, so a run stops rather than continue unrecordable.
  Streaming deltas stay best-effort; the lossless transcripts keep their copy.
"""

from __future__ import annotations

import contextlib
import json
import os
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, date, datetime, time
from pathlib import Path
from threading import RLock
from typing import Any

# High-frequency streaming deltas: written + flushed (so tailers see them live)
# but NOT fsynced. They are ephemeral UI, reconstructable from the lossless
# transcripts, and a reasoning model can emit tens of thousands per run -- an
# fsync each throttles the SSE reader on a slow disk and stalls the stream.
_EPHEMERAL_EVENTS = frozenset({"role.text_delta", "role.thinking_delta"})


class EventWriteError(Exception):
    """A durable event could not be appended to the run journal.

    The journal is what every viewer, listing, hook, and resume trusts; a run
    whose terminal events cannot land would render live forever, so the
    lifecycle stops loudly instead (the CLI reports it once, at dispatch).
    A cleanup emit that must not mask an in-flight exit wraps itself in
    `contextlib.suppress(EventWriteError)`."""


@dataclass(slots=True)
class EventSink:
    """Append structured JSON events to a JSONL file. Thread-safe.

    Uses a *reentrant* lock so emitting from a SIGINT handler (the Ctrl-C steer
    path emits `session.steer_requested`) cannot deadlock against the main thread
    being mid-`emit`, the handler runs in the same thread and re-acquires.
    """

    path: Path
    _lock: RLock
    _listeners: list[Callable[[dict[str, Any]], None]]

    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = RLock()
        self._listeners = []

    def subscribe(self, listener: Callable[[dict[str, Any]], None]) -> None:
        """Also hand each emitted event to an in-process consumer, as it happens.
        The live CLI renderer uses this; the file stays the source for
        out-of-process viewers (TUI, `watch`, web)."""
        self._listeners.append(listener)

    def emit(self, event_type: str, /, **fields: Any) -> None:
        """Append one event. Durable events (everything outside
        `_EPHEMERAL_EVENTS`) raise :class:`EventWriteError` when the append
        fails, and notify in-process listeners only after the write lands, so
        the live view can never show an event the durable record lost."""
        ephemeral = event_type in _EPHEMERAL_EVENTS
        payload: dict[str, Any] = {
            "ts": datetime.now(UTC).isoformat(timespec="microseconds"),
            "type": event_type,
        }
        payload.update(fields)
        try:
            line = json.dumps(payload, default=_json_default, ensure_ascii=False)
        except (TypeError, ValueError) as exc:
            if ephemeral:
                return  # a garbled delta is droppable UI
            raise EventWriteError(f"cannot serialize event {event_type!r}: {exc}") from exc
        # Encode HERE, lossily: json.dumps(ensure_ascii=False) passes a lone
        # surrogate (a split emoji escape in model-emitted tool args, a
        # surrogateescape-decoded argv) through as a str, and a text-mode write
        # would then raise UnicodeEncodeError. Replacing keeps the event
        # recorded and the file strictly valid UTF-8 for every reader.
        data = (line + "\n").encode("utf-8", "replace")
        try:
            with self._lock:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with self.path.open("ab") as fh:
                    fh.write(data)
                    fh.flush()
                    if not ephemeral:
                        os.fsync(fh.fileno())
        except OSError as exc:
            if not ephemeral:
                raise EventWriteError(f"event journal unwritable at {self.path}: {exc}") from exc
            # A lost delta stays live-rendered below; transcripts keep the
            # lossless copy.
        for listener in self._listeners:
            with contextlib.suppress(Exception):  # a UI consumer must never break the run
                listener(payload)


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, datetime | date | time):
        return value.isoformat()
    return repr(value)
