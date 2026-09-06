# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""`agent6 ps` lists every live session machine-wide, machine instances
included: their worker.pid sits at the instance root, not under a bucket."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from agent6.sessions.ipc import write_worker_pid
from agent6.ui.cli.ps_cmd import cmd_ps


def test_ps_lists_a_live_machine_instance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    base = tmp_path / "state"
    monkeypatch.setenv("AGENT6_STATE_HOME", str(base))
    monkeypatch.setenv("AGENT6_CONFIG_HOME", str(tmp_path / "cfg"))
    repo = base / "repo-abcdef"
    (repo / "sessions" / "runs").mkdir(parents=True)
    instance = repo / "machines" / "demo"
    instance.mkdir(parents=True)
    write_worker_pid(instance, os.getpid())  # this process: alive by the pid rule
    assert cmd_ps() == 0
    out = capsys.readouterr().out
    row = next(line for line in out.splitlines() if "demo" in line)
    assert "machine" in row and "running" in row and str(os.getpid()) in row
    assert "agent6 machine status <id>" in out


def test_ps_lists_a_linked_lane_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A fan-out lane's run dir is linked under the coordinator repo as well as
    its own state dir: one live session, one row."""
    base = tmp_path / "state"
    monkeypatch.setenv("AGENT6_STATE_HOME", str(base))
    monkeypatch.setenv("AGENT6_CONFIG_HOME", str(tmp_path / "cfg"))
    lane = base / "lane-repo-000000" / "sessions" / "runs" / "fan-l1"
    lane.mkdir(parents=True)
    (lane / "manifest.json").write_text(json.dumps({"mode": "run"}), encoding="utf-8")
    (lane / "logs.jsonl").write_text(
        json.dumps({"type": "session.start", "mode": "run", "user_task": "t"}) + "\n",
        encoding="utf-8",
    )
    write_worker_pid(lane, os.getpid())
    coordinator = base / "coord-repo-111111" / "sessions" / "runs"
    coordinator.mkdir(parents=True)
    (coordinator / "fan-l1").symlink_to(lane, target_is_directory=True)
    assert cmd_ps() == 0
    assert capsys.readouterr().out.count("fan-l1") == 1
