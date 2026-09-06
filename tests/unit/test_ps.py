# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""`agent6 ps` lists every live session machine-wide, machine instances
included: their worker.pid sits at the instance root, not under a bucket."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from agent6.config.layer import resolved_state_dir
from agent6.sessions.ipc import write_worker_pid
from agent6.ui.cli import main
from agent6.ui.cli.ps_cmd import cmd_ps

WAITER = """
machine = "waiter_demo"
version = 1
initial = "poll"

[budget]
max_usd = 1.0
max_transitions = 100

[vars.operator]
secs = { type = "int", value = 3600 }

[states.poll]
kind = "wait"
every_secs = "{{ secs }}"
on = { tick = "done", signal = "woken" }

[states.done]
kind = "terminal"
status = "ok"
reason = "ticked"

[states.woken]
kind = "terminal"
status = "ok"
reason = "signalled"
"""


def test_ps_reads_a_live_machine_through_the_shared_status_word(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A live worker in a wait state is "waiting" on every other surface
    (`machine_word_for_dir`); ps printed a hardcoded "running". The repo has
    run only a machine, so its state dir holds no sessions/ at all: ps must
    still list it."""
    monkeypatch.chdir(tmp_path)
    f = tmp_path / "waiter.asm.toml"
    f.write_text(WAITER, encoding="utf-8")
    assert main(["machine", "run", str(f), "--exit-on-wait"]) == 0
    capsys.readouterr()
    assert not (resolved_state_dir(tmp_path) / "sessions").exists()
    instance = resolved_state_dir(tmp_path) / "machines" / "waiter_demo"
    write_worker_pid(instance, os.getpid())  # this process: alive by the pid rule
    assert cmd_ps() == 0
    out = capsys.readouterr().out
    row = next(line for line in out.splitlines() if "waiter_demo" in line)
    assert "machine" in row and "waiting" in row and str(os.getpid()) in row, row
    assert "running" not in row
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


def test_ps_json_carries_the_row_facts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """`ps` was the one listing with no --json, so the cross-repo view could
    not be read by a script."""
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
    assert main(["ps", "--json"]) == 0
    (row,) = json.loads(capsys.readouterr().out)
    assert row["id"] == "fan-l1" and row["mode"] == "run" and row["pid"] == os.getpid()
    assert row["attached"] is False and row["repo_id"] == "lane-repo-000000"
    assert row["directory"] is None  # a state-dir id with no checkout behind it
