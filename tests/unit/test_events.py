# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Tests for `agent6.events.EventSink`."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from agent6.events import EventSink, EventWriteError


def _read_lines(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def test_emit_appends_json_lines(tmp_path: Path) -> None:
    sink = EventSink(tmp_path / "logs.jsonl")
    sink.emit("session.start", task="do a thing")
    sink.emit("step.start", index=1, title="hello")
    lines = _read_lines(tmp_path / "logs.jsonl")
    assert len(lines) == 2
    assert lines[0]["type"] == "session.start"
    assert lines[0]["task"] == "do a thing"
    assert "ts" in lines[0]
    assert lines[1]["type"] == "step.start"
    assert lines[1]["index"] == 1


def test_emit_creates_parent_dir(tmp_path: Path) -> None:
    target = tmp_path / "nested" / "deeper" / "logs.jsonl"
    sink = EventSink(target)
    sink.emit("hello")
    assert target.is_file()


def test_emit_reprs_non_serializable_fields(tmp_path: Path) -> None:
    """The sink never DROPS a field: _json_default reprs any unknown object
    (circular refs included), so the event lands whole; without that fallback
    the encoder would raise and the WHOLE event would be discarded."""
    sink = EventSink(tmp_path / "logs.jsonl")

    class Bad:
        pass

    bad = Bad()
    bad.self_ref = bad  # type: ignore[attr-defined]
    sink.emit("ok", x=1, p=tmp_path / "a", weird=bad)
    lines = _read_lines(tmp_path / "logs.jsonl")
    assert len(lines) == 1
    assert lines[0]["x"] == 1
    p_value = lines[0]["p"]
    assert isinstance(p_value, str)
    assert p_value.endswith("/a")
    weird = lines[0]["weird"]
    assert isinstance(weird, str) and "Bad" in weird  # repr'd, not dropped


def test_durable_emit_raises_on_unwritable_journal(tmp_path: Path) -> None:
    """A durable event that cannot land raises: the journal is the read model
    every surface trusts, and a run whose session.end was silently lost rendered
    "running" with live affordances forever. The in-process listener is NOT
    notified on the failure, so the live view can never show an event the
    durable record lost; deltas stay best-effort and still render live (the
    lossless transcripts keep their copy)."""
    # Point at a path under a regular file -> mkdir will fail.
    blocker = tmp_path / "blocker"
    blocker.write_text("", encoding="utf-8")
    sink = EventSink(blocker / "subdir" / "logs.jsonl")
    seen: list[dict[str, object]] = []
    sink.subscribe(seen.append)
    with pytest.raises(EventWriteError, match="unwritable"):
        sink.emit("session.end", reason="finish_session", all_passed=True)
    assert seen == []
    sink.emit("role.text_delta", text="still live")  # ephemeral: must not raise
    assert [e["type"] for e in seen] == ["role.text_delta"]


def test_delta_events_flush_but_do_not_fsync(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Ephemeral streaming deltas skip fsync (a reasoning model emits tens of
    thousands; an fsync each throttles the SSE read). Durable events still fsync.
    They are still written + flushed so tailers see them live."""
    synced: list[int] = []

    def _fake_fsync(fd: int) -> None:
        synced.append(fd)

    monkeypatch.setattr(os, "fsync", _fake_fsync)
    sink = EventSink(tmp_path / "logs.jsonl")

    sink.emit("role.thinking_delta", text="reasoning")
    sink.emit("role.text_delta", text="answer")
    assert synced == []  # no fsync for the deltas

    sink.emit("tool.call", name="read_file")
    assert len(synced) == 1  # a durable event fsyncs

    # all three are on disk regardless (flush, not fsync, makes them readable)
    types = [
        json.loads(line)["type"] for line in (tmp_path / "logs.jsonl").read_text().splitlines()
    ]
    assert types == ["role.thinking_delta", "role.text_delta", "tool.call"]


def test_emit_survives_lone_surrogate(tmp_path: Path) -> None:
    """json.dumps(ensure_ascii=False) passes a lone surrogate through; the old
    text-mode write then raised UnicodeEncodeError (a ValueError the OSError
    guard never caught), crashing the run from inside "telemetry must never
    break the run". The event must be recorded (lossily) and the file must
    stay strictly valid UTF-8 for every reader."""
    import json

    sink = EventSink(tmp_path / "logs.jsonl")
    sink.emit("session.start", user_task="caf\udce9")
    sink.emit("tool.call", args={"summary": "done \ud83d"})
    lines = [
        json.loads(line)
        for line in (tmp_path / "logs.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]
    assert [e["type"] for e in lines] == ["session.start", "tool.call"]
    assert "?" in lines[0]["user_task"]  # the surrogate was replaced, not dropped


def test_a_value_that_merely_answers_isoformat_encodes_as_its_repr(tmp_path: Path) -> None:
    """The encoder's date branch keys on the datetime types, not on a
    `isoformat` attribute: a mock (whose every attribute is another mock)
    recursed without end and hung the journal write."""
    from datetime import UTC, datetime
    from unittest.mock import MagicMock

    from agent6.events import _json_default  # pyright: ignore[reportPrivateUsage]

    assert _json_default(datetime(2026, 1, 2, tzinfo=UTC)) == "2026-01-02T00:00:00+00:00"
    mock = MagicMock()
    assert _json_default(mock) == repr(mock)


def test_the_log_dir_is_created_once_not_per_event(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The sink creates its directory through the state tree's one creator,
    whose handback walks the whole dir under sudo: once when missing, never
    again on every emit."""
    from agent6 import events as events_mod
    from agent6.paths import mkdir_for_real_user

    calls: list[Path] = []

    def counting(path: Path) -> None:
        calls.append(path)
        mkdir_for_real_user(path)

    monkeypatch.setattr(events_mod, "mkdir_for_real_user", counting)
    sink = EventSink(tmp_path / "run" / "logs.jsonl")
    sink.emit("session.start")
    sink.emit("loop.tool.call", name="read_file")
    assert calls == [tmp_path / "run"]
    assert (tmp_path / "run" / "logs.jsonl").read_text(encoding="utf-8").count("\n") == 2
