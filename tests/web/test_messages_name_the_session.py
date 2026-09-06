# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""What the page says back when an action cannot happen.

Every driving action reaches any session by id, so "no run 'x'" and "run is not
live" are wrong two times in three -- and the unknown-sub-route message named
`run/<id>/<sub>`, a path that has not existed since the route became
`/api/session/<id>/<sub>`.
"""

from __future__ import annotations

import functools
import json
from pathlib import Path

import pytest

from agent6.paths import state_dir
from agent6.sessions.layout import bucket_dir
from agent6.ui.web import actions


def _steer(cwd: Path, session_id: str, *, text: str) -> tuple[bool, str]:
    return actions.steer(cwd, session_id, text)


def _ask(state: Path) -> str:
    session = bucket_dir(state, "asks") / "curious-otter-AAAAAA"
    session.mkdir(parents=True)
    # FINISHED: driving a finished session is what hits the refusal path.
    (session / "logs.jsonl").write_text(
        "\n".join(
            json.dumps(e)
            for e in (
                {"type": "session.start", "mode": "ask", "user_task": "t"},
                {"type": "session.end", "reason": "answered", "all_passed": True},
            )
        )
        + "\n",
        encoding="utf-8",
    )
    (session / "manifest.json").write_text(
        json.dumps(
            {
                "version": 3,
                "session_id": "curious-otter-AAAAAA",
                "mode": "ask",
                "user_task": "t",
            }
        ),
        encoding="utf-8",
    )
    return "curious-otter-AAAAAA"


@pytest.mark.parametrize(
    "call",
    [
        actions.stop_after_step,
        actions.compact_run,
        functools.partial(_steer, text="go on"),
    ],
)
def test_a_refusal_does_not_call_an_ask_a_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, call: object
) -> None:

    monkeypatch.chdir(tmp_path)
    session_id = _ask(state_dir(tmp_path))
    ok, message = call(tmp_path, session_id)  # pyright: ignore[reportCallIssue, reportGeneralTypeIssues]
    assert not ok
    assert "run" not in message, message


def test_an_unknown_id_is_not_called_a_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    ok, message = actions.stop_after_step(tmp_path, "nope-nope-NOPE00")
    assert not ok
    assert "run" not in message, message
