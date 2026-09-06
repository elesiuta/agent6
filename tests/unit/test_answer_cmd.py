# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""`agent6 answer`: the headless seat at an `ask_user` question.

`agent6 steer` refuses a run blocked on a question and every other way in
needs a terminal, so a script that forwards questions somewhere else had no
way to send the reply back.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from agent6.sessions.ipc import read_question_answers, set_away_mode, write_worker_pid
from agent6.sessions.layout import SessionLayout
from agent6.ui.cli.answer_cmd import _cmd_answer  # pyright: ignore[reportPrivateUsage]


def _run_with_question(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, questions: list[dict[str, object]]
) -> SessionLayout:
    monkeypatch.setenv("AGENT6_STATE_HOME", str(tmp_path / "state"))
    repo = tmp_path / "repo"
    repo.mkdir()
    monkeypatch.chdir(repo)
    from agent6.config.layer import resolved_state_dir

    layout = SessionLayout(state_dir=resolved_state_dir(repo), session_id="curious-fox-AAAA11")
    layout.ensure()
    events: list[dict[str, object]] = [
        {"type": "session.start", "mode": "run", "user_task": "t"},
        {"type": "question.prompt", "id": "question-1", "questions": questions},
    ]
    layout.logs_path.write_text(
        "".join(json.dumps(e) + "\n" for e in events),
        encoding="utf-8",
    )
    write_worker_pid(layout.session_dir, os.getpid())  # this process = a live worker
    # A detached run left on "wait" is the seat this verb exists for.
    set_away_mode(layout.session_dir, "wait")
    return layout


def test_answer_writes_the_file_the_run_is_waiting_on(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    layout = _run_with_question(
        tmp_path,
        monkeypatch,
        questions=[{"question": "Which port?", "options": ["8080", "9090"]}],
    )

    assert _cmd_answer("curious-fox", ("9090",)) == 0

    assert read_question_answers(layout.session_dir, "question-1", timeout_s=1.0) == ("9090",)
    assert "answered curious-fox-AAAA11" in capsys.readouterr().out


def test_answer_with_no_text_prints_the_question_and_its_options(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The operator has to see what they are answering, so a bare call reads
    the prompt instead of guessing at it."""
    _run_with_question(
        tmp_path,
        monkeypatch,
        questions=[{"question": "Which port?", "options": ["8080", "9090"]}],
    )

    assert _cmd_answer("curious-fox", ()) == 0

    out = capsys.readouterr().out
    assert "Which port?" in out
    assert "options: 8080, 9090" in out


def test_a_short_answer_list_is_refused_rather_than_misaligned(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Answers align to the prompt's questions by index: one answer for a
    two-question prompt would answer the wrong one."""
    layout = _run_with_question(
        tmp_path,
        monkeypatch,
        questions=[{"question": "Which port?"}, {"question": "Which host?"}],
    )

    assert _cmd_answer("curious-fox", ("9090",)) == 2

    err = capsys.readouterr().err
    assert "2 question(s); 1 answer(s) given" in err
    assert not (layout.session_dir / "questions" / "question-1.answer").exists()


def test_answer_refuses_a_run_that_is_not_waiting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("AGENT6_STATE_HOME", str(tmp_path / "state"))
    repo = tmp_path / "repo"
    repo.mkdir()
    monkeypatch.chdir(repo)
    from agent6.config.layer import resolved_state_dir

    layout = SessionLayout(state_dir=resolved_state_dir(repo), session_id="curious-fox-AAAA11")
    layout.ensure()
    layout.logs_path.write_text(
        json.dumps({"type": "session.start", "mode": "run", "user_task": "t"}) + "\n",
        encoding="utf-8",
    )
    write_worker_pid(layout.session_dir, os.getpid())
    set_away_mode(layout.session_dir, "wait")

    assert _cmd_answer("curious-fox", ("yes",)) == 2

    assert "not waiting on a question" in capsys.readouterr().err


def test_answer_refuses_a_dead_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Only a live run holds a question open; a dead one's answer file would
    sit unread forever."""
    layout = _run_with_question(tmp_path, monkeypatch, questions=[{"question": "Which port?"}])
    write_worker_pid(layout.session_dir, 999_999_999)

    assert _cmd_answer("curious-fox", ("9090",)) == 2

    assert "not running" in capsys.readouterr().err


def test_answer_refuses_a_run_waiting_at_its_own_terminal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A foreground run blocks on its terminal and never reads the answer file;
    writing one there and printing "answered" left both sides waiting."""
    layout = _run_with_question(tmp_path, monkeypatch, questions=[{"question": "Which port?"}])
    (layout.session_dir / "approvals" / "away.mode").unlink()  # no away-mode, no front-end

    assert _cmd_answer("curious-fox", ("9090",)) == 2

    err = capsys.readouterr().err
    assert "waiting at its own terminal" in err
    assert "agent6 attach curious-fox-AAAA11" in err
    assert not (layout.session_dir / "questions" / "question-1.answer").exists()


def test_a_bare_call_prints_the_question_even_on_a_terminal_bound_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Reading the question needs no delivery channel. Gating the read on one
    hid the question from the operator it was asked of."""
    layout = _run_with_question(
        tmp_path, monkeypatch, questions=[{"question": "Which port?", "options": ["8080"]}]
    )
    (layout.session_dir / "approvals" / "away.mode").unlink()  # terminal-bound

    assert _cmd_answer("curious-fox", ()) == 0

    assert "Which port?" in capsys.readouterr().out


def test_a_live_run_with_no_question_says_so(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The refusal names the state the run is in, not one it is not: a run that
    is not waiting at all was told it was waiting at its terminal."""
    monkeypatch.setenv("AGENT6_STATE_HOME", str(tmp_path / "state"))
    repo = tmp_path / "repo"
    repo.mkdir()
    monkeypatch.chdir(repo)
    from agent6.config.layer import resolved_state_dir

    layout = SessionLayout(state_dir=resolved_state_dir(repo), session_id="curious-fox-AAAA11")
    layout.ensure()
    layout.logs_path.write_text(
        json.dumps({"type": "session.start", "mode": "run", "user_task": "t"}) + "\n",
        encoding="utf-8",
    )
    write_worker_pid(layout.session_dir, os.getpid())  # live, no away-mode, no front-end

    assert _cmd_answer("curious-fox", ("yes",)) == 2

    assert "is not waiting on a question" in capsys.readouterr().err
