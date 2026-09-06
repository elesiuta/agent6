# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""`agent6 sessions stop` drops the graceful stop marker for a running run."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from agent6.paths import state_dir
from agent6.sessions.ipc import stop_request_pending, write_worker_pid
from agent6.sessions.layout import SessionLayout
from agent6.ui.cli import main


def _session_dir(repo: Path, session_id: str) -> Path:
    layout = SessionLayout(state_dir=state_dir(repo), session_id=session_id)
    layout.ensure()
    layout.manifest_path.write_text('{"version": 2}', encoding="utf-8")
    (layout.session_dir / "logs.jsonl").write_text("", encoding="utf-8")
    return layout.session_dir


def test_runs_stop_requests_stop_for_a_live_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    rd = _session_dir(tmp_path, "live-run-AAA111")
    write_worker_pid(rd, os.getpid())  # this process is alive -> the run reads as running
    assert not stop_request_pending(rd)

    assert main(["sessions", "stop", "live-run-AAA111"]) == 0
    assert stop_request_pending(rd)  # the marker the worker honors at the next step
    assert "requested stop" in capsys.readouterr().out


def test_runs_stop_on_a_finished_run_with_lingering_pid_is_a_noop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """session.end lands before teardown clears worker.pid; in that window the loop
    has exited, so "requested stop ... it ends after the current step" was a
    promise nobody would keep. The gate is the liveness owner, not the pid."""
    monkeypatch.chdir(tmp_path)
    rd = _session_dir(tmp_path, "done-run-CCC333")
    (rd / "logs.jsonl").write_text(
        '{"type": "session.start", "mode": "run"}\n'
        '{"type": "session.end", "all_passed": true, "reason": "finish_session"}\n',
        encoding="utf-8",
    )
    write_worker_pid(rd, os.getpid())  # teardown not finished yet
    assert main(["sessions", "stop", "done-run-CCC333"]) == 0
    assert not stop_request_pending(rd)
    assert "not running" in capsys.readouterr().err


def test_runs_stop_on_a_dead_run_is_a_noop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    rd = _session_dir(tmp_path, "dead-run-BBB222")
    # A pid that is not alive: no worker.
    write_worker_pid(rd, 2**31 - 1)
    assert main(["sessions", "stop", "dead-run-BBB222"]) == 0
    assert not stop_request_pending(rd)
    assert "not running" in capsys.readouterr().err


def test_runs_stop_on_a_fan_out_says_what_stops(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A fan-out coordinator has no step to finish and no loop to resume: the
    stop asks its lanes to stop and imports what landed, and the message says
    that instead of promising a resume."""
    monkeypatch.chdir(tmp_path)
    rd = _session_dir(tmp_path, "fan-AAAA11")
    (rd / "manifest.json").write_text(
        '{"version": 3, "mode": "run", "fanout": {"lanes": 2, "spec": "2"}}', encoding="utf-8"
    )
    write_worker_pid(rd, os.getpid())
    assert main(["sessions", "stop", "fan-AAAA11"]) == 0
    assert stop_request_pending(rd)
    out = capsys.readouterr().out
    assert "its lanes are asked to stop" in out and "resume" not in out
