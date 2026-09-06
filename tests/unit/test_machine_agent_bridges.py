# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Tests for the machine `agent` state interactivity bridges.

Answers live in the per-state dir; the liveness gate probes the instance dir
where a front-end registers its `frontends/` claim.
"""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path

import pytest

from agent6.app.machine_agent import (
    _build_machine_bridges,  # pyright: ignore[reportPrivateUsage]
)
from agent6.events import EventSink
from agent6.sessions.ipc import (
    register_frontend,
    write_answer,
    write_question_answers,
    write_steer_answer,
)
from agent6.tools.schema import UserQuestion


def _dirs(tmp_path: Path) -> tuple[Path, Path, EventSink]:
    instance = tmp_path / "inst"
    state = instance / "states" / "0000-review"
    state.mkdir(parents=True)
    return instance, state, EventSink(state / "logs.jsonl")


def test_a_machine_command_grant_does_not_answer_another_scopes_prompt(tmp_path: Path) -> None:
    """The run approver honoured the prompt's scope and the machine one did not,
    so a machine state's "allow every command" auto-passed the gates that
    deliberately have no standing answer -- the same defect, in the copy."""
    from agent6.sessions.ipc import COMMAND_SCOPE, set_session_allow

    instance, state = tmp_path / "inst", tmp_path / "inst" / "1-agent"
    state.mkdir(parents=True)
    set_session_allow(state, COMMAND_SCOPE)
    bridges = _build_machine_bridges(instance, state, EventSink(state / "logs.jsonl"))

    assert bridges.prompts.approve("Allow run_command: ls", scope=COMMAND_SCOPE) is True
    # A headless deny, not the grant.
    assert bridges.prompts.approve("Allow fetch: evil.example /x") is False


def test_stale_answers_cleared_before_state_reexecution(tmp_path: Path) -> None:
    # Crash recovery re-executes the same `<seq>-<state>` dir with fresh prompt-id
    # counters; an answer file left by the aborted attempt must not satisfy this
    # execution's first prompt. Building the bridges drops the stale files.
    instance, state, events = _dirs(tmp_path)
    register_frontend(instance, os.getpid())
    write_answer(state, "approval-1", "yes")  # stale: from the aborted attempt
    write_question_answers(state, "question-1", ["stale"])
    _build_machine_bridges(instance, state, events)
    assert not (state / "approvals" / "approval-1.answer").exists()
    assert not (state / "questions" / "question-1.answer").exists()
    # The instance-dir front-end registration is untouched (it lives one level up).
    assert (instance / "frontends" / str(os.getpid())).exists()


def test_headless_defaults_when_no_frontend(tmp_path: Path) -> None:
    instance, state, events = _dirs(tmp_path)
    b = _build_machine_bridges(instance, state, events)
    # No front-end claim on the instance dir: deny approvals, empty answers, no steer.
    assert b.prompts.approve("run rm -rf?") is False
    assert b.prompts.ask((UserQuestion(question="pick", options=("a", "b")),)).answers == ("",)
    assert b.steer_requested() is False
    assert b.steer_prompt() is None


def test_approval_answer_read_from_per_state_dir(tmp_path: Path) -> None:
    instance, state, events = _dirs(tmp_path)
    register_frontend(instance, os.getpid())  # a live front-end owns the instance
    b = _build_machine_bridges(instance, state, events)  # clears pre-existing answers
    # A real front-end writes the answer AFTER approve() emits the prompt (approve
    # clears any premature pre-write first). A writer thread does exactly that;
    # the answer lands in the PER-STATE dir and read_answer picks it up promptly.
    threading.Thread(
        target=lambda: (time.sleep(0.2), write_answer(state, "approval-1", "yes")),
        daemon=True,
    ).start()
    assert b.prompts.approve("allow?") is True


def test_question_answer_read_from_per_state_dir(tmp_path: Path) -> None:
    instance, state, events = _dirs(tmp_path)
    register_frontend(instance, os.getpid())
    b = _build_machine_bridges(instance, state, events)
    threading.Thread(
        target=lambda: (time.sleep(0.2), write_question_answers(state, "question-1", ["chosen"])),
        daemon=True,
    ).start()
    question = UserQuestion(question="which?", options=("chosen", "other"))
    assert b.prompts.ask((question,)).answers == ("chosen",)


def test_machine_approval_ignores_a_premature_answer(tmp_path: Path) -> None:
    # The security property on the machine surface: an answer pre-written before
    # the prompt is emitted (a premature /api/machine/<name>/approve) is cleared
    # and not consumed -- the headless default (deny) applies instead.
    instance, state, events = _dirs(tmp_path)
    register_frontend(instance, os.getpid())
    b = _build_machine_bridges(instance, state, events)
    write_answer(state, "approval-1", "yes")  # premature: no prompt yet
    # No writer thread: nothing arrives after the prompt, so with the premature
    # answer cleared the approver falls through to the headless deny. Shrink the
    # read timeout so the poll gives up quickly instead of blocking 600s.
    from agent6.app import machine_agent

    orig = machine_agent.read_answer

    def _fast_read(rd: Path, pid: str, **kw: object) -> bool | None:
        return orig(rd, pid, timeout_s=0.3, poll_s=0.05, live_dir=kw.get("live_dir"))  # type: ignore[arg-type]

    machine_agent.read_answer = _fast_read  # type: ignore[assignment]
    try:
        assert b.prompts.approve("run rm -rf?") is False
    finally:
        machine_agent.read_answer = orig


def test_steer_request_and_answer_bridge(tmp_path: Path) -> None:
    instance, state, events = _dirs(tmp_path)
    register_frontend(instance, os.getpid())
    b = _build_machine_bridges(instance, state, events)
    # A front-end drops a steer.request in the per-state dir.
    from agent6.sessions.ipc import request_steer

    request_steer(state)
    assert b.steer_requested() is True
    write_steer_answer(state, "focus on tests")
    assert b.steer_prompt() == "focus on tests"
    b.steer_clear()
    assert b.steer_requested() is False


def test_machine_agent_wires_the_summariser_seat(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The machine agent built its Workflow without a summariser_provider, so
    compaction side-calls fell back to the worker-stamped provider and their
    transcripts carried seat="worker" -- the class of misfold the seat
    stamping exists to prevent. It now wires the same reviewer-role
    summariser the run path uses, sharing ONE TranscriptSink so the per-run
    seq counter cannot collide."""
    from typing import Any

    from agent6.app import machine_agent
    from agent6.machine.engine import AgentRequest
    from agent6.workflows.loop import SessionResult

    gdir = tmp_path / "g"
    (gdir / "agent6").mkdir(parents=True, exist_ok=True)
    (gdir / "agent6" / "config.toml").write_text(
        '[providers.anthropic]\napi_format = "anthropic"\n'
        '[models.worker]\nprovider = "anthropic"\nmodel = "claude-x"\n',
        encoding="utf-8",
    )
    monkeypatch.setenv("XDG_CONFIG_HOME", str(gdir))

    wf_kwargs: dict[str, Any] = {}
    sinks: dict[str, Any] = {}
    summariser = object()

    class _FakeWf:
        def __init__(self, **kw: Any) -> None:
            wf_kwargs.update(kw)

        def run(self, _prompt: str) -> SessionResult:
            return SessionResult(
                reason="finish_session", completed=True, summary="done", iterations=1, tool_calls=0
            )

    def _fake_role(*_a: Any, **k: Any) -> object:
        sinks["worker"] = k.get("transcript_sink")
        return object()

    def _fake_summariser(*_a: Any, **k: Any) -> object:
        sinks["summariser"] = k.get("transcript_sink")
        return summariser

    monkeypatch.setattr(machine_agent, "Workflow", _FakeWf)
    monkeypatch.setattr(machine_agent, "build_role_provider", _fake_role)
    monkeypatch.setattr(machine_agent, "reviewer_seat_provider", _fake_summariser)

    def _fake_dispatcher(**_k: Any) -> object:
        return object()

    monkeypatch.setattr(machine_agent, "ToolDispatcher", _fake_dispatcher)

    req = machine_agent.MachineAgentRequest(
        cwd=tmp_path,
        root=tmp_path,
        overlay={},
        isolation="none",
        transcript_dir=tmp_path / "t",
        request=AgentRequest(model="claude-x", prompt="go", timeout_s=5.0, provider="anthropic"),
    )
    out = machine_agent.run_one(req)
    assert out.reason == "finish_session"
    assert wf_kwargs["summariser_provider"] is summariser
    assert sinks["worker"] is sinks["summariser"]  # one sink, one seq counter


def test_away_wait_parks_a_prompt_for_the_frontend(tmp_path: Path) -> None:
    """A hub-spawned machine (away-mode "wait") parks approvals and questions
    for the front-end instead of inventing the headless answer -- the claim's
    TIMING no longer decides: an answer that arrives after the prompt fired
    (the viewer registering post-spawn) is honoured, exactly like a detached
    run's."""
    from agent6.sessions.ipc import set_away_mode

    instance, state, events = _dirs(tmp_path)
    set_away_mode(instance, "wait")
    b = _build_machine_bridges(instance, state, events)

    def _answer_late() -> None:
        time.sleep(0.4)
        register_frontend(instance, os.getpid())
        write_answer(state, "approval-1", "yes")

    t = threading.Thread(target=_answer_late)
    t.start()
    assert b.prompts.approve("run ls?", scope="command") is True
    t.join()

    def _answer_question_late() -> None:
        time.sleep(0.4)
        write_question_answers(state, "question-1", ("blue",))

    t2 = threading.Thread(target=_answer_question_late)
    t2.start()
    q = UserQuestion(question="colour?", options=("blue", "red"))
    assert b.prompts.ask((q,)).answers == ("blue",)
    t2.join()


def test_away_wait_prompt_stops_with_the_run(tmp_path: Path) -> None:
    """A parked prompt must not outlive the operator's Stop: the steer abort
    breaks the wait and the approval resolves to the safe deny."""
    from agent6.sessions.ipc import request_steer, set_away_mode, write_steer_answer

    instance, state, events = _dirs(tmp_path)
    set_away_mode(instance, "wait")
    b = _build_machine_bridges(instance, state, events)

    def _stop_late() -> None:
        time.sleep(0.4)
        request_steer(instance)
        write_steer_answer(instance, "abort")

    t = threading.Thread(target=_stop_late)
    t.start()
    assert b.prompts.approve("run ls?", scope="command") is False
    t.join()
