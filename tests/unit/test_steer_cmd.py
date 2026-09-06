# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""`agent6 steer`: the cron-friendly wrapper over the one steer channel."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from agent6.config.layer import resolved_state_dir
from agent6.sessions.ipc import steer_request_pending, take_steer_answer, write_worker_pid
from agent6.ui.cli import main


def _run_session(tmp_path: Path, session_id: str) -> Path:
    d = resolved_state_dir(tmp_path) / "sessions" / "runs" / session_id
    d.mkdir(parents=True)
    (d / "logs.jsonl").write_text("", encoding="utf-8")
    return d


def test_steer_queues_for_a_live_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("AGENT6_STATE_HOME", str(tmp_path / ".state"))
    monkeypatch.chdir(tmp_path)
    d = _run_session(tmp_path, "tiny-run-AAAA11")
    write_worker_pid(d, os.getpid())

    assert main(["steer", "tiny-run", "land your best patch now"]) == 0
    out = capsys.readouterr().out
    assert "steer queued for tiny-run-AAAA11" in out
    assert "next step boundary" in out  # the ruled default: no in-flight abort
    # The one shared channel: request marker + answer, exactly what the
    # composers write and the loop consumes. A plain steer never carries the
    # interrupt urgency; --now writes it into the marker.
    from agent6.sessions.ipc import steer_interrupt_pending

    assert steer_request_pending(d)
    assert not steer_interrupt_pending(d)
    assert take_steer_answer(d) == "land your best patch now"

    assert main(["steer", "tiny-run", "wrap up", "--now"]) == 0
    out = capsys.readouterr().out
    assert "interrupted to take it" in out
    assert steer_interrupt_pending(d)
    assert take_steer_answer(d) == "wrap up"


def test_steer_refuses_a_session_that_is_not_running(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A dead session's steer would silently park; the refusal names the
    queue-for-next-leg remedy that already exists (`resume --steer`)."""
    monkeypatch.setenv("AGENT6_STATE_HOME", str(tmp_path / ".state"))
    monkeypatch.chdir(tmp_path)
    d = _run_session(tmp_path, "tiny-run-BBBB22")
    write_worker_pid(d, 10**9)  # a pid that is never alive

    assert main(["steer", "tiny-run-BBBB22", "hello"]) == 2
    err = capsys.readouterr().err
    assert "not running" in err
    assert "agent6 resume tiny-run-BBBB22 --steer" in err
    assert not steer_request_pending(d)


def test_steer_reports_an_unknown_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("AGENT6_STATE_HOME", str(tmp_path / ".state"))
    monkeypatch.chdir(tmp_path)
    assert main(["steer", "nonesuch", "hello"]) == 2
    assert "ERROR" in capsys.readouterr().err


def test_steer_notes_an_unanswered_prompt_park(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A run parked on an approval has no boundaries and no interrupt can
    break the wait; the verb says so honestly instead of implying delivery."""
    import json

    monkeypatch.setenv("AGENT6_STATE_HOME", str(tmp_path / ".state"))
    monkeypatch.chdir(tmp_path)
    d = _run_session(tmp_path, "tiny-run-CCCC33")
    write_worker_pid(d, os.getpid())
    events = [
        {"type": "session.start", "mode": "run", "user_task": "t"},
        {
            "type": "approval.prompt",
            "id": "approval-1",
            "prompt": "Allow fetch: x",
            "ts": "2026-08-24T00:00:00+00:00",
        },
    ]
    (d / "logs.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events), encoding="utf-8")

    assert main(["steer", "tiny-run-CCCC33", "hello"]) == 0
    out = capsys.readouterr().out
    assert "the run is waiting (approval" in out
    assert "agent6 attach tiny-run-CCCC33" in out


def test_steer_names_the_answer_verb_for_a_question(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """`agent6 answer` takes a question, whichever seat the run waits in: its
    terminal prompt reads the answer file too. Naming it for an approval sent
    the operator straight to a refusal."""
    import json

    from agent6.sessions.ipc import set_away_mode

    monkeypatch.setenv("AGENT6_STATE_HOME", str(tmp_path / ".state"))
    monkeypatch.chdir(tmp_path)
    d = _run_session(tmp_path, "tiny-run-DDDD44")
    write_worker_pid(d, os.getpid())
    events = [
        {"type": "session.start", "mode": "run", "user_task": "t"},
        {
            "type": "question.prompt",
            "id": "question-1",
            "questions": [{"question": "Which port?"}],
            "ts": "2026-08-24T00:00:00+00:00",
        },
    ]
    (d / "logs.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events), encoding="utf-8")

    # No away-mode and no front-end: the run's own terminal prompt reads the
    # answer file, so the verb delivers there too.
    assert main(["steer", "tiny-run-DDDD44", "hello"]) == 0
    out = capsys.readouterr().out
    assert "the run is waiting (question" in out
    assert "agent6 answer tiny-run-DDDD44" in out

    # Detached on "wait": the same verb.
    set_away_mode(d, "wait")
    assert main(["steer", "tiny-run-DDDD44", "hello"]) == 0
    assert "agent6 answer tiny-run-DDDD44" in capsys.readouterr().out
