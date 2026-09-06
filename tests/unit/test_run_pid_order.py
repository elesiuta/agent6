# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""A fresh run writes worker.pid only once its preflight passed: `sessions
show` reads the pid as a live worker, so a run refused after the write (a
dirty tree, the checkout lock) read alive for the preflight's duration."""

from __future__ import annotations

import subprocess as sp
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from agent6.app import run as run_mod
from agent6.config import Config


def _repo(root: Path) -> None:
    root.mkdir()
    sp.run(["git", "init", "-q", "-b", "main"], cwd=root, check=True)
    sp.run(["git", "config", "user.email", "t@example.com"], cwd=root, check=True)
    sp.run(["git", "config", "user.name", "t"], cwd=root, check=True)
    (root / "a.py").write_text("x = 1\n", encoding="utf-8")
    sp.run(["git", "add", "a.py"], cwd=root, check=True)
    sp.run(["git", "commit", "-q", "-m", "seed"], cwd=root, check=True)


def test_run_writes_its_worker_pid_only_after_the_preflight_passed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("AGENT6_STATE_HOME", str(tmp_path / "state"))
    repo = tmp_path / "repo"
    _repo(repo)
    monkeypatch.chdir(repo)
    order: list[str] = []

    def _pid(_session_dir: Path, _pid: int) -> None:
        order.append("pid")

    def _isolation(*_a: object, **_k: object) -> str:
        return "none"

    monkeypatch.setattr(run_mod, "write_worker_pid", _pid)
    monkeypatch.setattr(run_mod, "select_isolation", _isolation)

    def _leg(*_a: object, **_k: object) -> object:
        order.append("leg")
        raise RuntimeError("stop here")

    monkeypatch.setattr(run_mod, "run_leg", _leg)
    cfg = Config.model_validate({"sandbox": {"run_commands": "yes"}})
    # A dirty tree refuses before any pid lands.
    (repo / "a.py").write_text("x = 2\n", encoding="utf-8")
    assert run_mod.run_task(cfg, "t", frontend=MagicMock(), mode="run") == 2
    assert order == []
    # A passing preflight writes the pid, then runs the leg.
    sp.run(["git", "checkout", "-q", "--", "a.py"], cwd=repo, check=True)
    with pytest.raises(RuntimeError, match="stop here"):
        run_mod.run_task(cfg, "t", frontend=MagicMock(), mode="run")
    assert order == ["pid", "leg"]
