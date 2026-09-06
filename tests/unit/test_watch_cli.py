# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""CLI tests for the unified `agent6 attach <target>` (run + machine, --json)."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from agent6.paths import state_dir
from agent6.sessions.ipc import write_worker_pid
from agent6.ui.cli import main

# A branch -> terminal machine: no model/jail, reaches a journaled end at once.
TINY = """
machine = "tiny"
version = 1
initial = "route"

[budget]
max_transitions = 10

[vars.code]
n = { type = "int", default = 0 }

[states.route]
kind = "branch"
when = [
  { if = "n == 0", goto = "done" },
  { else = true, goto = "done" },
]

[states.done]
kind = "terminal"
status = "ok"
reason = "routed"
"""


def _make_run(tmp_path: Path, session_id: str, events: list[dict[str, object]]) -> None:
    runs = state_dir(tmp_path) / "sessions" / "runs" / session_id
    runs.mkdir(parents=True)
    body = "".join(json.dumps(e) + "\n" for e in events)
    (runs / "logs.jsonl").write_text(body, encoding="utf-8")


def test_watch_run_json_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # A target that resolves to a run id (here by exact match) yields the folded
    # SessionState as JSON -- the same wire form a web client reads.
    monkeypatch.chdir(tmp_path)
    _make_run(
        tmp_path,
        "willing-glen-001",
        [
            {"type": "session.start", "user_task": "demo"},
            {"type": "tool.call", "name": "grep", "args": {"q": "x"}},
        ],
    )
    assert main(["attach", "willing-glen-001", "--json"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["user_task"] == "demo"
    assert out["tool_calls"][0]["name"] == "grep"


def test_watch_machine_json_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # A target that is not a run but names a machine instance routes to the
    # machine fold.
    monkeypatch.chdir(tmp_path)
    f = tmp_path / "tiny.asm.toml"
    f.write_text(TINY, encoding="utf-8")
    assert main(["machine", "run", str(f)]) == 0
    capsys.readouterr()  # drop run output
    assert main(["attach", "tiny", "--json"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["machine"] == "tiny"
    assert out["current"] == "done"
    assert out["ended"]["status"] == "ok"


def test_watch_unknown_target_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    assert main(["attach", "nope"]) == 2
    err = capsys.readouterr().err
    assert "no run or machine matches" in err
    # The search covers every session bucket and the machines: the refusal
    # names the state dir it walked, not one directory of five.
    assert str(state_dir(tmp_path)) in err and "sessions/runs" not in err


def test_watch_ambiguous_prefix_surfaces_disambiguation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # An ambiguous run prefix must report the ambiguity, not fall through to a
    # machine lookup and print "no run or machine matches".
    monkeypatch.chdir(tmp_path)
    _make_run(tmp_path, "willing-glen-001", [{"type": "session.start"}])
    _make_run(tmp_path, "willing-glen-002", [{"type": "session.start"}])
    assert main(["attach", "willing-glen"]) == 2
    err = capsys.readouterr().err
    assert "ambiguous" in err
    assert "no run or machine matches" not in err


def test_attach_to_a_crashed_run_ends_readonly_with_a_truthful_line(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A crashed worker never writes session.end: attach used to replay, re-ask the
    dead worker's pending approval, then follow forever behind a "working"
    spinner while `sessions show` called the same run stopped. With a stale
    worker.pid it must render read-only, never prompt, and end with the
    truthful crashed line."""
    import threading

    from agent6.ui.cli import plan_watch as pw

    _make_run(
        tmp_path,
        "dead-run",
        [
            {"type": "session.start", "user_task": "t"},
            {"type": "tool.call", "name": "run_command", "args": {}},
            {"type": "approval.prompt", "id": "approval-1", "prompt": "run?"},
        ],
    )
    monkeypatch.chdir(tmp_path)
    session_dir = state_dir(tmp_path) / "sessions" / "runs" / "dead-run"
    (session_dir / "worker.pid").write_text("999999", encoding="utf-8")

    def _no_prompt(*a: object, **k: object) -> None:
        raise AssertionError("a dead worker's prompt must not be re-asked")

    monkeypatch.setattr(pw._CliFrontEnd, "handle", _no_prompt)  # pyright: ignore[reportPrivateUsage]
    result: list[int] = []
    t = threading.Thread(target=lambda: result.append(main(["attach", "dead-run"])), daemon=True)
    t.start()
    t.join(timeout=5)
    assert not t.is_alive(), "attach failed to terminate on a crashed run"
    assert result == [0]
    err = capsys.readouterr().err
    assert "crashed or killed" in err


def test_attach_names_a_parked_run_instead_of_a_filesystem_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A parked submission (the busy-checkout refusal saved it) has no log yet.
    Every listing calls it "parked · checkout busy"; attach answered "ERROR: no
    logs.jsonl in <path>" and exited 2, so the operator who clicked through from
    a listing got a path instead of the state and the way out."""
    monkeypatch.chdir(tmp_path)
    session_dir = state_dir(tmp_path) / "sessions" / "runs" / "parked-run-77"
    session_dir.mkdir(parents=True)
    (session_dir / "manifest.json").write_text(
        json.dumps(
            {
                "version": 2,
                "session_id": "parked-run-77",
                "mode": "run",
                "user_task": "t",
                "parked_task": "finish the parser",
            }
        ),
        encoding="utf-8",
    )

    assert main(["attach", "parked-run-77"]) == 0
    out = capsys.readouterr().out
    assert "parked" in out
    assert "resume" in out
    assert "logs.jsonl" not in out

    assert main(["attach", "parked-run-77", "--json"]) == 0
    snap = json.loads(capsys.readouterr().out)
    assert snap["status_label"].startswith("parked")
    assert snap["session_id"] == "parked-run-77"


def test_attach_to_a_launching_run_says_starting_not_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A run still in preflight (egress + the ~80s verify inference before the
    first log line) has a LIVE worker but no log yet -> status "starting". It IS
    running, not resumable: telling the operator to `resume` would refuse or fork
    a second worker, so attach says it is starting instead."""
    monkeypatch.chdir(tmp_path)
    session_dir = state_dir(tmp_path) / "sessions" / "runs" / "launching-run-88"
    session_dir.mkdir(parents=True)
    (session_dir / "manifest.json").write_text(
        json.dumps(
            {"version": 2, "session_id": "launching-run-88", "mode": "run", "user_task": "t"}
        ),
        encoding="utf-8",
    )
    write_worker_pid(session_dir, os.getpid())  # a live worker, mid-preflight

    assert main(["attach", "launching-run-88"]) == 0
    out = capsys.readouterr().out
    assert "starting" in out
    assert "resume" not in out  # not resumable; it is already running


def test_attach_to_a_run_whose_pid_file_is_gone_does_not_follow_forever(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The sibling of the crashed-run case, and the commoner one: a worker that
    unwound through its finally CLEARS worker.pid and writes no session.end.

    attach had its own liveness rule -- "no pid is not dead" -- so it followed a
    log nothing would ever append to, while `sessions list` called the same
    session stale. One session, two surfaces, opposite answers.
    """
    import threading

    _make_run(
        tmp_path,
        "vanished-run",
        [
            {"type": "session.start", "user_task": "t"},
            {"type": "tool.call", "name": "run_command", "args": {}},
        ],
    )
    monkeypatch.chdir(tmp_path)
    session_dir = state_dir(tmp_path) / "sessions" / "runs" / "vanished-run"
    assert not (session_dir / "worker.pid").exists()

    result: list[int] = []
    t = threading.Thread(
        target=lambda: result.append(main(["attach", "vanished-run"])), daemon=True
    )
    t.start()
    t.join(timeout=5)
    assert not t.is_alive(), "attach followed a session with no worker"
    assert result == [0]
    assert "crashed or killed" in capsys.readouterr().err


def test_attach_to_a_finished_run_reports_its_outcome_not_a_crash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A run that ended cleanly is not a crashed one.

    Routing attach through the liveness fold collapsed two different states:
    `session_is_live` is False for a FINISHED run as much as for a dead one, so
    attaching to a run that had just printed "passed" told the operator it
    "never ended (crashed or killed)" -- while `sessions list` and `sessions
    show` both said passed. Three surfaces, two stories.
    """
    import threading

    _make_run(
        tmp_path,
        "done-run",
        [
            {"type": "session.start", "user_task": "t"},
            {"type": "session.end", "all_passed": True, "reason": "finish_session"},
        ],
    )
    monkeypatch.chdir(tmp_path)

    result: list[int] = []
    t = threading.Thread(target=lambda: result.append(main(["attach", "done-run"])), daemon=True)
    t.start()
    t.join(timeout=5)
    assert not t.is_alive(), "attach failed to terminate on a finished run"
    assert result == [0]
    err = capsys.readouterr().err
    assert "crashed or killed" not in err, f"a clean finish rendered as a crash: {err!r}"


def test_attach_prints_the_runs_policy_line_like_the_run_did(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The replay of a finished run carries the same header the live console
    printed under the task: model, isolation, command policy, gate."""
    _make_run(
        tmp_path,
        "done-run",
        [
            {"type": "session.start", "user_task": "t"},
            {"type": "session.end", "all_passed": True, "reason": "finish_session"},
        ],
    )
    session_dir = state_dir(tmp_path) / "sessions" / "runs" / "done-run"
    (session_dir / "manifest.json").write_text(
        json.dumps(
            {
                "mode": "run",
                "models": {"driver": {"model": "m-1"}},
                "policy": {"run_commands": "ask", "isolation": "strict"},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    assert main(["attach", "done-run"]) == 0
    assert "  m-1 · strict · commands ask · " in capsys.readouterr().out


def test_watch_json_checks_the_merged_claim_against_the_repo(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """`attach --json` claims merged only while the stamp still describes the
    branch (a run resumed after its merge commits past it), like the web
    snapshot and `sessions show`."""
    import subprocess

    monkeypatch.chdir(tmp_path)
    env = {
        "GIT_AUTHOR_NAME": "t",
        "GIT_AUTHOR_EMAIL": "t@t",
        "GIT_COMMITTER_NAME": "t",
        "GIT_COMMITTER_EMAIL": "t@t",
    }
    git = ["git", "-C", str(tmp_path)]
    subprocess.run([*git, "init", "-q", "-b", "main"], check=True)
    subprocess.run([*git, "commit", "-q", "--allow-empty", "-m", "base"], check=True, env=env)
    subprocess.run([*git, "branch", "agent6/stamped-run"], check=True)
    tip = subprocess.run(
        [*git, "rev-parse", "agent6/stamped-run"], check=True, capture_output=True, text=True
    ).stdout.strip()
    _make_run(tmp_path, "stamped-run", [{"type": "session.start", "user_task": "t"}])
    session_dir = state_dir(tmp_path) / "sessions" / "runs" / "stamped-run"
    (session_dir / "manifest.json").write_text(
        json.dumps(
            {
                "mode": "run",
                "run_branch": "agent6/stamped-run",
                "base_branch": "main",
                "merged": {"into": "main", "sha": tip, "tip": tip},
            }
        ),
        encoding="utf-8",
    )
    assert main(["attach", "stamped-run", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["merged_into"] == "main"
    subprocess.run([*git, "checkout", "-q", "agent6/stamped-run"], check=True)
    subprocess.run([*git, "commit", "-q", "--allow-empty", "-m", "more"], check=True, env=env)
    assert main(["attach", "stamped-run", "--json"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert "merged_into" not in out
    assert out["branch_line"] == "agent6/stamped-run → merges into main"


def test_attach_replay_reads_finished_from_the_fold_not_the_last_line(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The watch read "finished" off the journal's LAST line while every other
    surface reads the listing fold, which only a resumed leg un-finishes. A
    side answer journaled after session.end made the replay follow a finished
    run until its watcher's own pid died."""
    import os
    import threading

    from agent6.sessions.ipc import write_worker_pid

    _make_run(
        tmp_path,
        "done-run",
        [
            {"type": "session.start", "user_task": "t"},
            {"type": "session.end", "reason": "finish_session", "all_passed": True},
            {"type": "btw.answered", "question": "q", "block": "a"},
        ],
    )
    monkeypatch.chdir(tmp_path)
    session_dir = state_dir(tmp_path) / "sessions" / "runs" / "done-run"
    write_worker_pid(session_dir, os.getpid())  # reads as live: the follow path
    result: list[int] = []
    t = threading.Thread(
        target=lambda: result.append(main(["attach", "done-run", "--raw", "--since", "5"])),
        daemon=True,
    )
    t.start()
    t.join(timeout=5)
    assert not t.is_alive(), "attach followed a finished run"
    assert result == [0]
    capsys.readouterr()
