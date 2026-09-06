# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Unit tests for the web write side's argv building and spawn wiring (no HTTP)."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from agent6.paths import state_dir
from agent6.sessions.layout import machines_root
from agent6.ui.cli.parser import (
    _inject_default_verb,  # pyright: ignore[reportPrivateUsage]
    build_parser,
)
from agent6.ui.web import actions

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


# --- body-derived strings ride behind `--`, never parsed as flags -------------


def test_spawn_machine_create_argv_ends_options_before_task(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: list[list[str]] = []

    def _fake_locate(argv: list[str], cwd: Path, **_k: object) -> tuple[Path | None, str]:
        captured.append(list(argv))
        return None, "not started"

    monkeypatch.setattr(actions, "spawn_and_locate", _fake_locate)
    actions.spawn_machine_create(tmp_path, "-dashy task")
    assert captured[-1][1:] == ["machine", "create", "--", "-dashy task"]


def test_merge_and_config_argv_end_options_before_values(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: list[list[str]] = []

    def _fake_capture(argv: list[str], cwd: Path, **_k: object) -> tuple[bool, str]:
        captured.append(list(argv))
        return True, "ok"

    monkeypatch.setattr(actions, "run_cli_capture", _fake_capture)
    actions.merge_run(tmp_path, "-rid", "squash")
    assert captured[-1][1:] == ["sessions", "merge", "--strategy", "squash", "--", "-rid"]
    actions.set_config(tmp_path, "sandbox.protect_git", "-1", repo=True)
    assert captured[-1][1:] == ["config", "set", "--repo", "--", "sandbox.protect_git", "-1"]


@pytest.mark.parametrize(
    "argv",
    [
        ["run", "--preset", "quick", "--", "-dashy task"],
        ["run", "--parallel", "2", "--", "-dashy task"],
        ["run", "--preset", "quick", "--parallel", "gpt-5,opus", "--", "-dashy task"],
        ["plan", "--", "-dashy task"],
        ["ask", "--", "-dashy question"],
        ["machine", "create", "--", "-dashy task"],
        ["sessions", "merge", "--strategy", "squash", "--", "-rid"],
        ["config", "set", "--repo", "--", "sandbox.protect_git", "-1"],
    ],
)
def test_cli_parser_accepts_double_dash_before_positionals(argv: list[str]) -> None:
    # The argv shapes the web actions build must parse: `--` ends options and the
    # dashy value lands in the positional.
    ns = build_parser().parse_args(_inject_default_verb(argv))
    positional = ns.task if hasattr(ns, "task") else getattr(ns, "session_id", None) or ns.value
    assert str(positional).startswith("-")


# --- machine run: refusals surface instead of a false "started" ---------------


def test_spawn_machine_run_propagates_refusal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mf = tmp_path / "tiny.asm.toml"
    mf.write_text(TINY, encoding="utf-8")

    def _refuse(*_a: object, **_k: object) -> str:
        return "agent6 machine exited (1):\nlock held"

    monkeypatch.setattr(actions, "spawn_and_confirm", _refuse)
    ok, msg = actions.spawn_machine_run(tmp_path, str(mf))
    assert ok is False
    assert "lock held" in msg


def test_spawn_machine_run_started_signal_is_child_worker_pid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # started(pid) fires only when the instance worker.pid holds the CHILD's own
    # pid: a live worker.pid from an already-running machine (lock held) must
    # not read as "this spawn started".
    from agent6.sessions.ipc import write_worker_pid

    mf = tmp_path / "tiny.asm.toml"
    mf.write_text(TINY, encoding="utf-8")
    captured_argv: list[list[str]] = []
    started_fns: list[Callable[[int], bool]] = []

    def _fake_confirm(
        argv: list[str],
        cwd: Path,
        *,
        started: Callable[[int], bool],
        timeout_s: float = 25.0,
    ) -> str:
        captured_argv.append(list(argv))
        started_fns.append(started)
        return ""

    monkeypatch.setattr(actions, "spawn_and_confirm", _fake_confirm)
    ok, msg = actions.spawn_machine_run(tmp_path, str(mf))
    assert ok is True and msg == "started"
    assert captured_argv[-1][1:] == ["machine", "run", str(mf)]
    started = started_fns[-1]
    instance = machines_root(state_dir(tmp_path)) / "tiny"
    instance.mkdir(parents=True)
    assert started(4242) is False  # no worker.pid yet
    write_worker_pid(instance, 4242)
    assert started(4242) is True  # the child owns the instance
    assert started(4243) is False  # someone else's pid (a prior runner)


# --- ended machines take no input ---------------------------------------------


def _ended_machine(cwd: Path, name: str) -> Path:
    """An instance whose journal records a MachineEnd, with one state-log dir."""
    inst = state_dir(cwd) / "machines" / name
    (inst / "states" / "0000-route").mkdir(parents=True)
    (inst / "machine.asm.toml").write_text(TINY, encoding="utf-8")
    (inst / "states" / "0000-route" / "logs.jsonl").write_text("", encoding="utf-8")
    (inst / "journal.jsonl").write_text(
        '{"type":"machine.begin","ts":"2026-07-12T00:00:00+00:00","machine":"tiny","version":1}\n'
        '{"type":"machine.end","ts":"2026-07-12T00:00:01+00:00","status":"ok",'
        '"reason":"routed","state":"done","transitions":1}\n',
        encoding="utf-8",
    )
    return inst


def test_machine_poke_refuses_ended_machine(tmp_path: Path) -> None:
    inst = _ended_machine(tmp_path, "tiny")
    ok, msg = actions.machine_poke(tmp_path, "tiny", message="wake up")
    assert not ok
    assert "ended" in msg
    assert not (inst / "signal").exists()  # nothing pretends to be delivered


def test_machine_approve_refuses_ended_machine(tmp_path: Path) -> None:
    inst = _ended_machine(tmp_path, "tiny")
    ok, msg = actions.machine_approve(tmp_path, "tiny", "approval-1", "yes")
    assert not ok
    assert "ended" in msg
    assert not list((inst / "states" / "0000-route").glob("**/*.answer"))


def test_machine_answer_refuses_ended_machine(tmp_path: Path) -> None:
    inst = _ended_machine(tmp_path, "tiny")
    ok, msg = actions.machine_answer(tmp_path, "tiny", "question-1", ["yes"])
    assert not ok
    assert "ended" in msg
    assert not list((inst / "states" / "0000-route").glob("**/*.answer"))


def test_machine_steer_refuses_ended_machine(tmp_path: Path) -> None:
    inst = _ended_machine(tmp_path, "tiny")
    ok, msg = actions.machine_steer(tmp_path, "tiny", "do more")
    assert not ok
    assert "ended" in msg
    assert not list((inst / "states" / "0000-route").glob("*.answer"))


def _parked_machine(cwd: Path, name: str) -> Path:
    """An instance parked on an armed wait (no live worker, no MachineEnd) whose
    newest state dir is a COMPLETED agent state."""
    inst = state_dir(cwd) / "machines" / name
    (inst / "states" / "0001-work").mkdir(parents=True)
    (inst / "machine.asm.toml").write_text(TINY, encoding="utf-8")
    (inst / "states" / "0001-work" / "logs.jsonl").write_text("", encoding="utf-8")
    (inst / "journal.jsonl").write_text(
        '{"type":"machine.begin","ts":"2026-07-12T00:00:00+00:00","machine":"tiny","version":1}\n',
        encoding="utf-8",
    )
    (inst / "wait.json").write_text(
        '{"state":"idle","wake_epoch":9999999999.0}\n', encoding="utf-8"
    )
    return inst


def test_machine_steer_refuses_when_no_state_is_executing(tmp_path: Path) -> None:
    """A parked/stopped machine's newest state dir is a finished agent state
    whose run loop has exited, so nothing polls its steer marker. Reporting
    "steer requested" dropped the operator's course-correction on the floor --
    the run steer refuses "run is not live" for exactly this reason."""
    inst = _parked_machine(tmp_path, "tiny")
    ok, msg = actions.machine_steer(tmp_path, "tiny", "skip the deploy this cycle")
    assert not ok
    assert "not" in msg  # names the reason rather than claiming success
    assert not list((inst / "states" / "0001-work").glob("*.answer"))
    assert not (inst / "states" / "0001-work" / "steer.request").exists()


def test_approve_and_answer_refuse_a_dead_run(tmp_path: Path) -> None:
    """A run killed while blocked on a prompt still renders its Allow/Deny box
    (the client filters on `answered`, not liveness), and these two POSTs wrote
    the answer and reported success with no worker left to consume it -- the
    same shape as the typed steer that used to reach a corpse. Every sibling
    action already refuses "run is not live"."""
    session_dir = state_dir(tmp_path) / "sessions" / "runs" / "dead-run-A1"
    session_dir.mkdir(parents=True)
    (session_dir / "manifest.json").write_text(
        '{"version":2,"session_id":"dead-run-A1","mode":"run","user_task":"t"}', encoding="utf-8"
    )
    (session_dir / "logs.jsonl").write_text(
        '{"type":"session.start","mode":"run","user_task":"t"}\n'
        '{"type":"approval.prompt","id":"approval-1","prompt":"ok?"}\n',
        encoding="utf-8",
    )
    (session_dir / "worker.pid").write_text("999999999", encoding="utf-8")  # never alive
    before = sorted(p.name for p in session_dir.iterdir())

    ok, msg = actions.approve(tmp_path, "dead-run-A1", "approval-1", "yes")
    assert ok is False and "not live" in msg
    ok2, msg2 = actions.answer_question(tmp_path, "dead-run-A1", "q-1", ["yes"])
    assert ok2 is False and "not live" in msg2
    assert sorted(p.name for p in session_dir.iterdir()) == before, "nothing may be written"


def test_approve_and_answer_reach_a_run_waiting_at_its_own_terminal(tmp_path: Path) -> None:
    """A foreground run's terminal prompt reads the answer file while it waits,
    so a live run with no away-mode and no front-end claim takes the web's
    answer. The web once refused it as "waiting at its own terminal"; before
    that it wrote the file and said "answered" to a run that never looked."""
    import os

    from agent6.sessions.ipc import read_answer, read_question_answers

    session_dir = state_dir(tmp_path) / "sessions" / "runs" / "tty-run-A1"
    session_dir.mkdir(parents=True)
    (session_dir / "manifest.json").write_text(
        '{"version":2,"session_id":"tty-run-A1","mode":"run","user_task":"t"}', encoding="utf-8"
    )
    (session_dir / "logs.jsonl").write_text(
        '{"type":"session.start","mode":"run","user_task":"t"}\n'
        '{"type":"approval.prompt","id":"approval-1","prompt":"ok?"}\n'
        '{"type":"question.prompt","id":"question-1","questions":[{"question":"port?"}]}\n',
        encoding="utf-8",
    )
    (session_dir / "worker.pid").write_text(str(os.getpid()), encoding="utf-8")  # live

    assert actions.approve(tmp_path, "tty-run-A1", "approval-1", "yes") == (True, "answered")
    assert actions.answer_question(tmp_path, "tty-run-A1", "question-1", ["9090"]) == (
        True,
        "answered",
    )

    assert read_answer(session_dir, "approval-1", timeout_s=1.0) == "yes"
    assert read_question_answers(session_dir, "question-1", timeout_s=1.0) == ("9090",)


def test_machine_prompt_answers_refuse_a_machine_that_is_not_running(tmp_path: Path) -> None:
    """The newest state dir of a parked or dead machine is a FINISHED agent state
    whose loop has exited, so a marker written there is polled by nobody. steer
    already refused; approve and answer reported {"ok": true, "answered"} and the
    prompt box never cleared, so the operator had every reason to think it landed."""
    from agent6.ui.web import actions

    inst = state_dir(tmp_path) / "machines" / "dead"
    (inst / "states" / "0001-work").mkdir(parents=True)
    (inst / "machine.asm.toml").write_text("machine = 'x'\n", encoding="utf-8")
    (inst / "journal.jsonl").write_text("", encoding="utf-8")
    (inst / "states" / "0001-work" / "logs.jsonl").write_text("", encoding="utf-8")
    (inst / "worker.pid").write_text("999999999", encoding="utf-8")  # dead

    ok, msg = actions.machine_approve(tmp_path, "dead", "approval-1", "yes")
    assert ok is False and "not running" in msg
    ok, msg = actions.machine_answer(tmp_path, "dead", "q-1", ["yes"])
    assert ok is False and "not running" in msg


_UNKNOWN_MACHINE_CALLS: list[tuple[Callable[[Path], tuple[bool, str]], str]] = [
    (lambda cwd: actions.machine_approve(cwd, "ghost", "approval-1", "yes"), "approve"),
    (lambda cwd: actions.machine_answer(cwd, "ghost", "question-1", ["yes"]), "answer"),
    (lambda cwd: actions.machine_steer(cwd, "ghost", "do more"), "steer"),
]


@pytest.mark.parametrize(("call", "label"), _UNKNOWN_MACHINE_CALLS)
def test_an_unknown_machine_is_named_as_unknown_not_as_stopped(
    tmp_path: Path, call: Callable[[Path], tuple[bool, str]], label: str
) -> None:
    """A machine that does not exist must not be described as one that stopped.

    The liveness gate read a missing instance dir as LIVE, so the action sailed
    past it and failed one step later with "no active agent state" -- telling
    the operator to go looking for a state belonging to a machine that was
    never there.
    """
    ok, msg = call(tmp_path)
    assert not ok
    assert "no machine 'ghost'" in msg, f"{label} misdescribed an unknown machine: {msg!r}"
    assert "not running" not in msg, f"{label} implied the machine exists: {msg!r}"
    assert "agent state" not in msg, f"{label} pointed at a state that never existed: {msg!r}"


def test_the_composer_refuses_an_empty_resume_of_a_finished_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A run the agent ENDED has nothing to continue, and the web spawn is
    DETACHED -- the same refusal from `agent6 resume` would land on a process
    nobody reads, so the composer would report "resuming" for a run that never
    started. Refused here instead, naming the composer (the CLI wording quotes
    a --steer line that is not the remedy on this surface), and an instruction
    still goes straight through.
    """
    import json

    session_dir = state_dir(tmp_path) / "sessions" / "runs" / "done-WEB111"
    session_dir.mkdir(parents=True)
    (session_dir / "manifest.json").write_text(
        json.dumps({"version": 2, "session_id": "done-WEB111", "mode": "run", "user_task": "t"}),
        encoding="utf-8",
    )
    (session_dir / "logs.jsonl").write_text(
        json.dumps({"type": "session.end", "reason": "finish_session", "all_passed": True}) + "\n",
        encoding="utf-8",
    )
    spawned: list[str] = []

    def _spawn(
        _cwd: Path,
        _session_id: str,
        *,
        steer: str = "",
        preset: str = "",
        config_path: object = None,
    ) -> str:
        spawned.append(steer)
        return ""

    monkeypatch.setattr(actions, "spawn_detached_resume", _spawn)

    ok, msg = actions.resume_run(tmp_path, "done-WEB111")
    assert ok is False
    assert "already finished" in msg and "type what to do next" in msg
    assert spawned == []

    ok, _ = actions.resume_run(tmp_path, "done-WEB111", "do more")
    assert ok is True
    assert spawned == ["do more"]


def test_machine_stop_refuses_ended_and_marks_a_live_one(tmp_path: Path) -> None:
    """The stop verb never plants a marker an ended or dead instance would
    trip over later; a live worker gets the durable stop marker."""
    from unittest.mock import patch

    from agent6.viewmodel import machine_state as machine_state_mod

    inst = _ended_machine(tmp_path, "tiny")
    ok, msg = actions.machine_stop(tmp_path, "tiny")
    assert not ok and "ended" in msg
    assert not (inst / "stop").exists()

    (inst / "journal.jsonl").write_text(
        '{"type":"machine.begin","ts":"2026-07-12T00:00:00+00:00","machine":"tiny","version":1}\n',
        encoding="utf-8",
    )
    ok, msg = actions.machine_stop(tmp_path, "tiny")
    assert not ok and "not running" in msg
    with patch.object(machine_state_mod, "worker_is_alive", return_value=True):  # the gate's owner
        ok, msg = actions.machine_stop(tmp_path, "tiny")
    assert ok and "stop requested" in msg
    assert (inst / "stop").is_file()


def test_run_plan_spawns_from_plan_and_refuses_non_plans(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """ "Run this plan" spawns `agent6 run --from <id>` detached and hands
    back the new run id; a non-plan session and a plan with no plan.md refuse
    without spawning (the plan itself is untouched either way)."""
    import json

    plan = state_dir(tmp_path) / "sessions" / "plans" / "planny-one-AAAAAA"
    plan.mkdir(parents=True)
    (plan / "manifest.json").write_text(
        json.dumps({"version": 2, "session_id": plan.name, "mode": "plan", "user_task": "t"}),
        encoding="utf-8",
    )
    (plan / "logs.jsonl").write_text("", encoding="utf-8")
    run = state_dir(tmp_path) / "sessions" / "runs" / "runny-one-AAAAAA"
    run.mkdir(parents=True)
    (run / "manifest.json").write_text(
        json.dumps({"version": 2, "session_id": run.name, "mode": "run", "user_task": "t"}),
        encoding="utf-8",
    )
    (run / "logs.jsonl").write_text("", encoding="utf-8")
    seen: dict[str, object] = {}

    def _fake_spawn(argv: list[str], _cwd: Path, **kw: object) -> tuple[Path, str]:
        seen["argv"] = argv
        seen["env"] = kw.get("env")
        child = state_dir(tmp_path) / "sessions" / "runs" / "fresh-run-BBBBBB"
        child.mkdir(parents=True, exist_ok=True)
        return child, ""

    monkeypatch.setattr(actions, "spawn_and_locate", _fake_spawn)
    # No plan.md yet: still planning (or never finished) -> refuse, no spawn.
    payload, err = actions.run_plan(tmp_path, "planny-one-AAAAAA")
    assert payload is None and "no plan.md" in err and not seen

    (plan / "plan.md").write_text("# Plan\n", encoding="utf-8")
    payload, err = actions.run_plan(tmp_path, "planny-one-AAAAAA")
    assert err == "" and payload == {"run_id": "fresh-run-BBBBBB"}
    argv = seen["argv"]
    assert isinstance(argv, list) and argv[-3:] == ["run", "--from", "planny-one-AAAAAA"]
    env = seen["env"]
    assert isinstance(env, dict) and env["AGENT6_DETACHED_AWAY"] == "wait"

    payload, err = actions.run_plan(tmp_path, "runny-one-AAAAAA")
    assert payload is None and "not a plan" in err


def test_spawn_machine_run_takes_the_listed_name_or_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The hub lists a file by path and name; both name it. An unknown value
    is refused with the accepted forms."""
    mf = tmp_path / "tiny.asm.toml"
    mf.write_text(TINY, encoding="utf-8")
    spawned: list[list[str]] = []

    def _record(argv: list[str], *_a: object, **_k: object) -> str:
        spawned.append(argv)
        return ""

    monkeypatch.setattr(actions, "spawn_and_confirm", _record)
    assert actions.spawn_machine_run(tmp_path, "tiny.asm.toml") == (True, "started")
    assert spawned[-1][-1] == str(mf)
    ok, msg = actions.spawn_machine_run(tmp_path, "/elsewhere/tiny.asm.toml")
    assert ok is False and "listed path or name" in msg and "tiny.asm.toml" in msg


def test_prune_carries_the_squash_opt_in_only_when_asked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The hub's prune ran the bare verb, so under the default squash strategy
    it deleted nothing: every merged run's branch is unreachable and kept."""
    captured: list[list[str]] = []

    def _fake_capture(argv: list[str], cwd: Path, **_k: object) -> tuple[bool, str]:
        captured.append(list(argv))
        return True, "ok"

    monkeypatch.setattr(actions, "run_cli_capture", _fake_capture)

    actions.prune_sessions(tmp_path)
    assert captured[-1][1:] == ["sessions", "prune"]
    actions.prune_sessions(tmp_path, delete_squashed=True)
    assert captured[-1][1:] == ["sessions", "prune", "--delete-squashed"]


def test_a_now_steer_writes_the_urgent_marker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`/now <text>` on the web composer is the CLI's `steer --now`: the answer
    is the text alone and the request marker carries the urgency."""
    import os

    from agent6.paths import state_dir
    from agent6.sessions.ipc import STEER_ANSWER_FILE, STEER_REQUEST_FILE, write_worker_pid
    from agent6.ui.web import actions

    monkeypatch.chdir(tmp_path)
    d = state_dir(tmp_path) / "sessions" / "runs" / "live-one-AAAAAA"
    d.mkdir(parents=True)
    (d / "logs.jsonl").write_text('{"type": "session.start", "mode": "run"}\n', encoding="utf-8")
    write_worker_pid(d, os.getpid())
    ok, msg = actions.steer(tmp_path, "live-one-AAAAAA", "/now stop and report")
    assert ok and msg == "steer requested now"
    assert (d / STEER_REQUEST_FILE).read_text(encoding="utf-8") == "now"
    assert (d / STEER_ANSWER_FILE).read_text(encoding="utf-8") == "stop and report"
    ok, msg = actions.steer(tmp_path, "live-one-AAAAAA", "/now")
    assert not ok and "needs the instruction" in msg
