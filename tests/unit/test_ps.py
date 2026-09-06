# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""`agent6 ps` lists every live session machine-wide, machine instances
included: their worker.pid sits at the instance root, not under a bucket."""

from __future__ import annotations

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
