# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Tests for the plain logs.jsonl formatter."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent6.ui.cli import plan_watch
from agent6.ui.cli.plan_watch import (
    event_epoch,  # pyright: ignore[reportPrivateUsage]
    format_plain_event,  # pyright: ignore[reportPrivateUsage]
)


def test_format_plain_event_renders_known_fields() -> None:
    raw = json.dumps(
        {
            "ts": 1100.0,
            "event": "loop.auto_commit",
            "session_id": "ignored-by-formatter",
            "iteration": 3,
            "sha": "abc123def456",
        }
    )
    out = format_plain_event(raw, session_start_ts=1000.0)
    assert "+  100.0s" in out
    assert "loop.auto_commit" in out
    assert "iteration=3" in out
    assert "sha='abc123def456'" in out
    assert "session_id" not in out  # filtered


def test_format_plain_event_handles_garbage_line() -> None:
    out = format_plain_event("not-json-at-all\n", session_start_ts=0.0)
    assert out == "not-json-at-all"


def test_format_plain_event_no_ts_anchor() -> None:
    raw = json.dumps({"event": "ping"})
    out = format_plain_event(raw, session_start_ts=None)
    assert "ping" in out


def test_format_plain_event_tolerates_a_non_string_event_type() -> None:
    """A corrupt event value falls back to readable text instead of crashing the tail."""
    out = format_plain_event('{"type": 42, "value": "kept"}', session_start_ts=None)
    assert "42" in out
    assert "value='kept'" in out


def test_event_epoch_parses_iso_and_numbers() -> None:
    # EventSink writes ISO-8601 strings; the anchor must parse those.
    assert event_epoch("2026-06-08T05:41:39.762404+00:00") is not None
    assert event_epoch(1100.0) == 1100.0
    assert event_epoch("not-a-timestamp") is None
    assert event_epoch(None) is None
    assert event_epoch(True) is None  # bool is not a usable epoch


def test_raw_watch_exits_when_the_worker_is_gone(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A raw attach must not poll forever after the only possible writer exits."""
    target = tmp_path / "dead-run"
    target.mkdir()
    (target / "logs.jsonl").write_text('{"type":"session.start"}\n', encoding="utf-8")

    def _must_not_sleep(_seconds: float) -> None:
        pytest.fail("raw watch slept after the worker was gone")

    monkeypatch.setattr(plan_watch.time, "sleep", _must_not_sleep)
    assert plan_watch._cmd_watch_plain(target, since=0) == 0  # pyright: ignore[reportPrivateUsage]
    assert "crashed or killed" in capsys.readouterr().err


def test_raw_watch_preserves_a_partial_replay_line(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The --since snapshot must buffer its torn last line until the writer completes it."""
    target = tmp_path / "live-run"
    target.mkdir()
    events = target / "logs.jsonl"
    events.write_bytes(b'{"type":"role.')
    appended = False

    def _append_tail(_seconds: float) -> None:
        nonlocal appended
        if appended:
            pytest.fail("raw watch failed to consume the appended session end")
        appended = True
        with events.open("ab") as fh:
            fh.write(b'call","role":"worker"}\n{"type":"session.end"}\n')

    def _worker_is_alive(_target: Path) -> bool:
        return True

    monkeypatch.setattr(plan_watch, "worker_is_alive", _worker_is_alive)
    monkeypatch.setattr(plan_watch.time, "sleep", _append_tail)
    assert plan_watch._cmd_watch_plain(target, since=5) == 0  # pyright: ignore[reportPrivateUsage]
    out = capsys.readouterr().out
    assert "role.call" in out
    assert "role='worker'" in out


def test_format_plain_event_renders_elapsed_for_iso_ts() -> None:
    # Regression: ts is an ISO string (events.py), not a number; the elapsed
    # column must still render rather than always blanking.
    start = "2026-06-08T05:41:39+00:00"
    later = "2026-06-08T05:42:39+00:00"  # +60s
    anchor = event_epoch(start)
    out = format_plain_event(
        json.dumps({"ts": later, "type": "loop.auto_commit"}), session_start_ts=anchor
    )
    assert "+   60.0s" in out
    assert "loop.auto_commit" in out
