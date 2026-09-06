# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""What a fresh install says when there is nothing yet.

This is the first thing a new operator sees, and the difference between "you
have no sessions, here is how to start one" and "ERROR: no runs directory at
/home/.../sessions/runs" is the difference between an empty state and a broken
install.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agent6.ui.cli import main

# Commands that can only answer "you have nothing yet" on a fresh state dir.
_EMPTY_STATE = [
    ["sessions", "list"],
    ["sessions", "show"],
    ["sessions", "diff"],
    ["sessions", "commits"],
    ["sessions", "merge"],
    ["sessions", "stop"],
    ["sessions", "rm"],
    ["attach"],
    ["resume"],
    ["fork"],
    ["plan", "show"],
]


@pytest.mark.parametrize("argv", _EMPTY_STATE, ids=[" ".join(a) for a in _EMPTY_STATE])
def test_it_does_not_read_as_a_broken_install(
    argv: list[str],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(tmp_path)
    main(argv)
    said = capsys.readouterr()
    text = said.out + said.err

    assert "Traceback" not in text, text
    # An internal path is what makes an empty state look like a fault.
    assert "/sessions/runs" not in text, text
    # And it is a session, whatever mode it would have been.
    assert "no runs" not in text, text


@pytest.mark.parametrize("argv", _EMPTY_STATE, ids=[" ".join(a) for a in _EMPTY_STATE])
def test_it_says_how_to_make_one(
    argv: list[str],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Every dead end names the command that gets you out of it."""
    monkeypatch.chdir(tmp_path)
    main(argv)
    said = capsys.readouterr()
    assert "agent6 " in said.out + said.err, said.out + said.err


def test_a_run_verb_over_a_plan_alone_says_no_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """`sessions commits` (and diff, merge) act on a run's branch: with a plan
    recorded and no run, they said "no sessions yet" over a real session."""
    from agent6.config.layer import resolved_state_dir
    from agent6.sessions.layout import bucket_dir

    monkeypatch.chdir(tmp_path)
    session = bucket_dir(resolved_state_dir(tmp_path), "plans") / "brave-oak-AAAAAA"
    session.mkdir(parents=True)
    (session / "logs.jsonl").write_text(
        '{"type": "session.start", "mode": "plan", "user_task": "t"}\n', encoding="utf-8"
    )
    assert main(["sessions", "commits"]) == 2
    assert 'no runs yet. Start one with `agent6 run "<task>"`.' in capsys.readouterr().err
