# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""`agent6 run` with no task: the newest-run/plan fallbacks and refusals."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from agent6.paths import state_dir
from agent6.ui.cli import main
from agent6.ui.cli.plan_watch import (
    _most_recent_plan_session_id,  # pyright: ignore[reportPrivateUsage]
)
from agent6.viewmodel import newest_session_dir


def test_newest_run_dir_none_for_missing_bucket(tmp_path: Path) -> None:
    assert newest_session_dir([tmp_path / "missing"]) is None


def test_newest_run_dir_none_when_empty(tmp_path: Path) -> None:
    runs = state_dir(tmp_path) / "sessions" / "runs"
    runs.mkdir(parents=True)
    assert newest_session_dir([runs]) is None


def test_newest_run_dir_uses_log_activity_not_frontend_dir_touch(tmp_path: Path) -> None:
    runs = state_dir(tmp_path) / "sessions" / "runs"
    runs.mkdir(parents=True)
    older = runs / "alpha-bravo-charlie"
    newer = runs / "delta-echo-foxtrot"
    older.mkdir()
    newer.mkdir()
    (older / "logs.jsonl").write_text('{"type":"session.start"}\n', encoding="utf-8")
    (newer / "logs.jsonl").write_text('{"type":"session.start"}\n', encoding="utf-8")
    os.utime(older / "logs.jsonl", (100, 100))
    os.utime(newer / "logs.jsonl", (1000, 1000))
    (older / "frontend.pid").write_text("12345", encoding="utf-8")
    newest = newest_session_dir([runs])
    assert newest is not None
    assert newest.name == "delta-echo-foxtrot"


def test_most_recent_plan_run_id_uses_log_activity_not_frontend_dir_touch(tmp_path: Path) -> None:
    plans = state_dir(tmp_path) / "sessions" / "plans"
    plans.mkdir(parents=True)
    older = plans / "older-plan"
    newer = plans / "newer-plan"
    older.mkdir()
    newer.mkdir()
    for session_dir in (older, newer):
        (session_dir / "plan.md").write_text("# Plan\n", encoding="utf-8")
        (session_dir / "logs.jsonl").write_text('{"type":"session.start"}\n', encoding="utf-8")
    os.utime(older / "logs.jsonl", (100, 100))
    os.utime(newer / "logs.jsonl", (1000, 1000))
    (older / "frontend.pid").write_text("12345", encoding="utf-8")
    assert _most_recent_plan_session_id(plans) == "newer-plan"


def test_run_without_task_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "agent6.toml").write_text("# placeholder\n", encoding="utf-8")
    rc = main(["run"])
    assert rc == 2
    # With no task AND no prior plan to fall back to, `run` still errors.
    assert "needs a task" in capsys.readouterr().err


def test_run_no_task_points_at_most_recent_plan(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # No task given but a prior plan exists: non-interactively (pytest stdin is
    # not a TTY) refuse, but point the user at the plan + the --from form.
    monkeypatch.chdir(tmp_path)
    session_dir = state_dir(tmp_path) / "sessions" / "plans" / "tidy-otter-AB12CD"
    session_dir.mkdir(parents=True)
    (session_dir / "plan.md").write_text("# Plan: wire up the thing\n", encoding="utf-8")
    rc = main(["run"])
    assert rc == 2
    err = capsys.readouterr().err
    assert "tidy-otter-AB12CD" in err
    assert "--from" in err


def test_run_continue_flag_is_gone(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # `run --continue` was a strict subset of `resume`; the one obvious way
    # remains `agent6 resume`. argparse refuses the dropped flag like any
    # unknown flag (no alias, no special-cased message).
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit) as exc:
        main(["run", "--continue"])
    assert exc.value.code == 2


def test_parallel_refuses_an_explicit_run_id(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # Each lane mints its own run id; the flag was silently dropped (never
    # forwarded to dispatch_parallel), so refuse it like -i/--tui.
    monkeypatch.chdir(tmp_path)
    rc = main(["run", "--parallel", "2", "--session-id", "myid", "task"])
    assert rc == 2
    assert "--session-id" in capsys.readouterr().err
