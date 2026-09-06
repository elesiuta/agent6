# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""resolve_session_layout finds a session in any bucket under sessions/ (so
anything a listing shows -- an ask, a `machine create` draft -- is inspectable
and watchable by id too)."""

from __future__ import annotations

from pathlib import Path

import pytest

from agent6.paths import state_dir
from agent6.sessions.id import SessionIdError
from agent6.ui.cli._common import resolve_session_layout


@pytest.fixture(autouse=True)
def isolated_state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "st"))


def test_resolves_runs_and_asks_with_correct_subdir(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    state = state_dir(repo)
    (state / "sessions" / "runs" / "run-abc").mkdir(parents=True)
    (state / "sessions" / "runs" / "run-abc" / "logs.jsonl").write_text("{}\n", encoding="utf-8")
    (state / "sessions" / "asks" / "ask-xyz").mkdir(parents=True)
    (state / "sessions" / "asks" / "ask-xyz" / "logs.jsonl").write_text("{}\n", encoding="utf-8")

    session_layout = resolve_session_layout(repo, "run-abc")
    assert session_layout.subdir == "runs" and session_layout.session_id == "run-abc"

    ask_layout = resolve_session_layout(repo, "ask-xyz")
    assert ask_layout.subdir == "asks" and ask_layout.session_id == "ask-xyz"
    # The layout points at the ask's own directory (where its graph now lives).
    assert ask_layout.session_dir == state / "sessions" / "asks" / "ask-xyz"

    # Unique-prefix resolution works too.
    assert resolve_session_layout(repo, "ask-").session_id == "ask-xyz"


def test_resolves_a_machine_create_draft(tmp_path: Path) -> None:
    # `agent6 attach <draft-id>` follows the authoring agent's live log.
    repo = tmp_path / "repo"
    repo.mkdir()
    state = state_dir(repo)
    (state / "sessions" / "machines" / "blue-meadow-X1").mkdir(parents=True)
    (state / "sessions" / "machines" / "blue-meadow-X1" / "logs.jsonl").write_text(
        "{}\n", encoding="utf-8"
    )

    layout = resolve_session_layout(repo, "blue-")
    assert layout.subdir == "machines"
    assert layout.session_dir == state / "sessions" / "machines" / "blue-meadow-X1"


def test_prefix_must_be_unique_across_runs_and_asks(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    state = state_dir(repo)
    (state / "sessions" / "runs" / "same-run").mkdir(parents=True)
    (state / "sessions" / "runs" / "same-run" / "logs.jsonl").write_text("{}\n", encoding="utf-8")
    (state / "sessions" / "asks" / "same-ask").mkdir(parents=True)
    (state / "sessions" / "asks" / "same-ask" / "logs.jsonl").write_text("{}\n", encoding="utf-8")

    with pytest.raises(SessionIdError) as exc:
        resolve_session_layout(repo, "same-")
    assert not exc.value.no_match  # an ambiguous prefix is not "no such session"
    assert "runs/same-run" in str(exc.value)
    assert "asks/same-ask" in str(exc.value)


def test_exact_match_wins_over_cross_bucket_prefix(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    state = state_dir(repo)
    (state / "sessions" / "runs" / "run").mkdir(parents=True)
    (state / "sessions" / "runs" / "run" / "logs.jsonl").write_text("{}\n", encoding="utf-8")
    (state / "sessions" / "asks" / "run-question").mkdir(parents=True)
    (state / "sessions" / "asks" / "run-question" / "logs.jsonl").write_text(
        "{}\n", encoding="utf-8"
    )

    layout = resolve_session_layout(repo, "run")
    assert layout.subdir == "runs"
    assert layout.session_id == "run"


def test_empty_query_is_invalid(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (state_dir(repo) / "sessions" / "runs" / "run-abc").mkdir(parents=True)
    (state_dir(repo) / "sessions" / "runs" / "run-abc" / "logs.jsonl").write_text(
        "{}\n", encoding="utf-8"
    )

    with pytest.raises(SessionIdError, match="empty run id"):
        resolve_session_layout(repo, "")


def test_raises_when_no_match(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (state_dir(repo) / "sessions" / "runs" / "run-abc").mkdir(parents=True)
    (state_dir(repo) / "sessions" / "runs" / "run-abc" / "logs.jsonl").write_text(
        "{}\n", encoding="utf-8"
    )
    with pytest.raises(SessionIdError):
        resolve_session_layout(repo, "nope")
