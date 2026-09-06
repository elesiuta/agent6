# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The shared _common glue helper resolve_or_newest_layout (a run by id, or
the latest across every bucket)."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from agent6.paths import state_dir
from agent6.sessions.id import SessionIdError
from agent6.ui.cli._common import resolve_or_newest_layout


def _session_dir(state: Path, bucket: str, session_id: str, *, log_mtime: float) -> Path:
    """Seed a run dir with a logs.jsonl at a controlled mtime (what
    newest_session_dir sorts by)."""
    d = state / "sessions" / bucket / session_id
    d.mkdir(parents=True)
    log = d / "logs.jsonl"
    log.write_text("{}\n", encoding="utf-8")
    os.utime(log, (log_mtime, log_mtime))
    return d


# --- resolve_or_newest_layout ------------------------------------------------


def test_explicit_id_resolves_across_buckets(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    state = state_dir(repo)
    (state / "sessions" / "asks" / "ask-xyz").mkdir(parents=True)
    (state / "sessions" / "asks" / "ask-xyz" / "logs.jsonl").write_text("{}\n", encoding="utf-8")

    layout = resolve_or_newest_layout(repo, "ask-")
    assert layout is not None
    assert layout.subdir == "asks" and layout.session_id == "ask-xyz"


def test_empty_id_picks_the_newest_across_buckets(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    state = state_dir(repo)
    _session_dir(state, "runs", "old-run", log_mtime=1000.0)
    _session_dir(state, "asks", "new-ask", log_mtime=2000.0)

    layout = resolve_or_newest_layout(repo, "")
    assert layout is not None
    assert layout.session_id == "new-ask" and layout.subdir == "asks"
    assert layout.session_dir == state / "sessions" / "asks" / "new-ask"


def test_empty_id_with_no_runs_returns_none(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    assert resolve_or_newest_layout(repo, "") is None


def test_bad_explicit_id_raises(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (state_dir(repo) / "sessions" / "runs" / "run-abc").mkdir(parents=True)
    (state_dir(repo) / "sessions" / "runs" / "run-abc" / "logs.jsonl").write_text(
        "{}\n", encoding="utf-8"
    )
    with pytest.raises(SessionIdError):
        resolve_or_newest_layout(repo, "nope")
