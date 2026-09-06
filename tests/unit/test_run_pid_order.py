# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""A run writes worker.pid before it can ask the operator anything.

The pid is what every surface gates on: `sessions list` reads a run without
one as "created", and `agent6 answer` refuses it as "not running". A run
parked on its own dirty-tree start question is neither -- it is a live worker
waiting for exactly that answer. Refusals BEFORE the first prompt still write
none, and the teardown clears it on every exit path.
"""

from __future__ import annotations

import subprocess as sp
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from agent6.app import run as run_mod
from agent6.config import Config
from agent6.tools.operator_prompts import QuestionAnswer, QuestionRequest


def _repo(root: Path) -> None:
    root.mkdir()
    sp.run(["git", "init", "-q", "-b", "main"], cwd=root, check=True)
    sp.run(["git", "config", "user.email", "t@example.com"], cwd=root, check=True)
    sp.run(["git", "config", "user.name", "t"], cwd=root, check=True)
    (root / "a.py").write_text("x = 1\n", encoding="utf-8")
    sp.run(["git", "add", "a.py"], cwd=root, check=True)
    sp.run(["git", "commit", "-q", "-m", "seed"], cwd=root, check=True)


def test_run_writes_its_worker_pid_before_it_asks_the_operator(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
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

    def _cancel(_request: QuestionRequest, /) -> QuestionAnswer:
        order.append("ask")
        return QuestionAnswer(("cancel",), "stdin")

    # A front-end whose operator cancels the dirty-tree start question.
    frontend = MagicMock()
    frontend.build_questioner.return_value = _cancel
    # The pid lands BEFORE the question: while a run waits on it, `agent6
    # answer` and the listings must read it as the live worker it is.
    (repo / "a.py").write_text("x = 2\n", encoding="utf-8")
    assert run_mod.run_task(cfg, "t", frontend=frontend, mode="run") == 2
    assert order == ["pid", "ask"]
    # A passing preflight writes the pid, then runs the leg -- in every mode:
    # an ask blocks on questions too, and `agent6 ps` and `steer` gate on the
    # same file. Only run mode took the checkout lock the write sat behind.
    sp.run(["git", "checkout", "-q", "--", "a.py"], cwd=repo, check=True)
    for mode in ("run", "plan", "ask"):
        order.clear()
        with pytest.raises(RuntimeError, match="stop here"):
            run_mod.run_task(cfg, "t", frontend=frontend, mode=mode)
        assert order == ["pid", "leg"], mode


def test_a_cancelled_start_question_leaves_no_pid_behind(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The teardown clears it on every exit path, so a run that asked and was
    then cancelled does not go on reading as live."""
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    repo = tmp_path / "repo"
    _repo(repo)
    monkeypatch.chdir(repo)

    def _isolation(*_a: object, **_k: object) -> str:
        return "none"

    monkeypatch.setattr(run_mod, "select_isolation", _isolation)

    def _cancel(_request: QuestionRequest, /) -> QuestionAnswer:
        return QuestionAnswer(("cancel",), "stdin")

    frontend = MagicMock()
    frontend.build_questioner.return_value = _cancel
    (repo / "a.py").write_text("x = 2\n", encoding="utf-8")

    assert (
        run_mod.run_task(
            Config.model_validate({"sandbox": {"run_commands": "yes"}}),
            "t",
            frontend=frontend,
            mode="run",
        )
        == 2
    )

    from agent6.config.layer import resolved_state_dir
    from agent6.sessions.layout import bucket_dir

    runs = bucket_dir(resolved_state_dir(repo), "runs")
    pids = list(runs.glob("*/worker.pid"))
    assert [p for p in pids if p.read_text(encoding="utf-8").strip()] == []
