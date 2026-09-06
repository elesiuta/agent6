# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The one gate every operator prompt goes through (`tools.operator_prompts`):
it mints the ids, journals the prompt/answer pair, names the call a prompt
gates, and no front-end journals a copy."""

from __future__ import annotations

import json
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pytest

from agent6.config import Config
from agent6.events import EventSink
from agent6.sessions.ipc import COMMAND_SCOPE, set_session_allow, write_answer
from agent6.tools.dispatch import ToolDispatcher
from agent6.tools.errors import ToolDenied
from agent6.tools.operator_prompts import (
    ApprovalAnswer,
    ApprovalRequest,
    OperatorPrompts,
    QuestionAnswer,
    QuestionRequest,
)
from agent6.tools.schema import UserQuestion

_SRC = Path(__file__).resolve().parents[2] / "src" / "agent6"


def _journal(session_dir: Path) -> list[dict[str, Any]]:
    lines = (session_dir / "logs.jsonl").read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines]


def _of(session_dir: Path, event_type: str) -> list[dict[str, Any]]:
    return [e for e in _journal(session_dir) if e["type"] == event_type]


def _deny(_request: ApprovalRequest, /) -> ApprovalAnswer:
    return ApprovalAnswer(False, "stdin")


def _sink(session_dir: Path) -> EventSink:
    session_dir.mkdir(parents=True, exist_ok=True)
    return EventSink(session_dir / "logs.jsonl")


def _prompts(session_dir: Path, events: EventSink, **kw: Any) -> OperatorPrompts:
    return OperatorPrompts(journal=events.emit, session_dir=session_dir, **kw)


def _dispatcher(session_dir: Path, events: EventSink, prompts: OperatorPrompts) -> ToolDispatcher:
    """A run's wiring: the gate and the dispatcher journal to the one sink."""
    cfg = Config.model_validate(
        {"sandbox": {"run_commands": "ask"}, "workflow": {"verify_command": ["true"]}}
    )
    return ToolDispatcher(
        root=session_dir.parent, config=cfg, prompts=prompts, events=events, session_dir=session_dir
    )


def test_an_approval_is_journaled_with_the_call_it_gates(tmp_path: Path) -> None:
    """The dispatcher journals tool.call, then the gate journals the prompt
    stamped with THAT call's id and the answer with the approver's source."""
    session_dir = tmp_path / "run"
    events = _sink(session_dir)
    d = _dispatcher(session_dir, events, _prompts(session_dir, events, approver=_deny))
    with pytest.raises(ToolDenied):
        d.dispatch("run_command", {"argv": ["ls"]})
    (call,) = _of(session_dir, "tool.call")
    (prompt,) = _of(session_dir, "approval.prompt")
    (answer,) = _of(session_dir, "approval.answer")
    assert prompt["call_id"] == call["call_id"]
    assert (prompt["id"], prompt["prompt"], prompt["standing"]) == (
        "approval-1",
        "Allow run_command: ls",
        True,
    )
    assert (answer["id"], answer["approved"], answer["source"]) == ("approval-1", False, "stdin")


def test_a_question_is_journaled_with_the_call_it_gates(tmp_path: Path) -> None:
    session_dir = tmp_path / "run"
    events = _sink(session_dir)
    seen: list[QuestionRequest] = []

    def _pick(request: QuestionRequest, /) -> QuestionAnswer:
        seen.append(request)
        return QuestionAnswer(("b",), "stdin")

    d = _dispatcher(session_dir, events, _prompts(session_dir, events, questioner=_pick))
    out = d.dispatch("ask_user", {"questions": [{"question": "which?", "options": ["a", "b"]}]})
    assert out.to_wire() == {"answers": ["b"]}
    (call,) = _of(session_dir, "tool.call")
    (prompt,) = _of(session_dir, "question.prompt")
    (answer,) = _of(session_dir, "question.answer")
    assert prompt["call_id"] == call["call_id"] == seen[0].call_id
    assert prompt["id"] == "question-1" == seen[0].id
    assert prompt["questions"] == [{"question": "which?", "options": ["a", "b"]}]
    assert (answer["answers"], answer["source"]) == (["b"], "stdin")


def test_concurrent_seats_each_name_their_own_call(tmp_path: Path) -> None:
    """Two seats dispatching at once on one dispatcher, both past their stamp
    before either reads it: each prompt names the call its own thread is
    dispatching (a stamp shared across threads would name the other seat's)."""
    session_dir = tmp_path / "run"
    events = _sink(session_dir)
    named: dict[str, int | None] = {}

    def _record(request: ApprovalRequest, /) -> ApprovalAnswer:
        named[request.prompt] = request.call_id
        return ApprovalAnswer(False, "stdin")

    d = _dispatcher(session_dir, events, _prompts(session_dir, events, approver=_record))
    both_stamped = threading.Barrier(2)
    policy = d.command_policy

    def _rendezvous() -> str:
        both_stamped.wait(timeout=10)
        return policy()

    d.command_policy = _rendezvous  # read after the stamp, before the gate
    with ThreadPoolExecutor(max_workers=2) as pool:
        seats = [pool.submit(d.dispatch, "run_command", {"argv": ["ls", arg]}) for arg in "ab"]
        for seat in seats:
            with pytest.raises(ToolDenied):
                seat.result(timeout=30)
    stamped = {" ".join(e["args"]["argv"]): e["call_id"] for e in _of(session_dir, "tool.call")}
    assert named == {f"Allow run_command: {argv}": cid for argv, cid in stamped.items()}
    assert len(set(named.values())) == 2


def test_a_verify_the_harness_runs_gates_no_call(tmp_path: Path) -> None:
    """`run_verify` outside a dispatch (the harness's own certification) goes
    through the same gate and carries no call; so does a question asked
    before the loop."""
    session_dir = tmp_path / "run"
    events = _sink(session_dir)
    prompts = _prompts(session_dir, events, approver=_deny)
    d = _dispatcher(session_dir, events, prompts)
    with pytest.raises(ToolDenied):
        d.run_verify()
    prompts.ask((UserQuestion(question="stash?", options=("stash", "cancel")),))
    (approval,) = _of(session_dir, "approval.prompt")
    (question,) = _of(session_dir, "question.prompt")
    assert approval["prompt"] == "Allow run_verify_command: true"
    assert approval["call_id"] is None and question["call_id"] is None


def test_a_standing_grant_answers_without_a_prompt(tmp_path: Path) -> None:
    """ "Allow all" for a scope answers that scope's later prompts itself: no
    front-end is asked and no prompt is journaled, only the answer. The id is
    still consumed, so the sequence stays in step on every surface."""
    session_dir = tmp_path / "run"
    events = _sink(session_dir)
    seen: list[str] = []

    def _record(request: ApprovalRequest, /) -> ApprovalAnswer:
        seen.append(request.id)
        return ApprovalAnswer(False, "stdin")

    prompts = _prompts(session_dir, events, approver=_record)
    set_session_allow(session_dir, COMMAND_SCOPE)
    assert prompts.approve("Allow run_command: ls", scope=COMMAND_SCOPE) is True
    assert seen == [] and _of(session_dir, "approval.prompt") == []
    (answer,) = _of(session_dir, "approval.answer")
    assert (answer["id"], answer["approved"], answer["source"]) == ("approval-1", True, "session")
    prompts.approve("Allow fetch: evil.example /x")  # no scope: the grant covers nothing
    assert seen == ["approval-2"]


def test_a_premature_answer_is_cleared_before_the_prompt_is_journaled(tmp_path: Path) -> None:
    """Ids are predictable counters, so an answer written ahead of its prompt
    must be gone before any front-end could read the slot: the gate clears
    it, then journals the prompt, then asks."""
    session_dir = tmp_path / "run"
    events = _sink(session_dir)
    write_answer(session_dir, "approval-1", "yes")  # the premature POST
    slot = session_dir / "approvals" / "approval-1.answer"
    assert slot.exists()
    observed: list[tuple[bool, list[str]]] = []

    def _observe(_request: ApprovalRequest, /) -> ApprovalAnswer:
        observed.append((slot.exists(), [e["type"] for e in _journal(session_dir)]))
        return ApprovalAnswer(False, "stdin")

    prompts = _prompts(session_dir, events, approver=_observe)
    assert prompts.approve("Allow run_command: ls", scope=COMMAND_SCOPE) is False
    assert observed == [(False, ["approval.prompt"])]


def test_answers_align_to_the_questions(tmp_path: Path) -> None:
    """A front-end that answered fewer questions left the rest unanswered
    (the TUI writes no answers for a dismissed modal); more answers than
    questions is a front-end defect and fails loudly."""
    session_dir = tmp_path / "run"
    events = _sink(session_dir)
    questions = (UserQuestion(question="a?"), UserQuestion(question="b?"))

    def _short(_request: QuestionRequest, /) -> QuestionAnswer:
        return QuestionAnswer((), "frontend")

    assert _prompts(session_dir, events, questioner=_short).ask(questions).answers == ("", "")
    (answer,) = _of(session_dir, "question.answer")
    assert answer["answers"] == ["", ""]

    def _long(_request: QuestionRequest, /) -> QuestionAnswer:
        return QuestionAnswer(("x", "y", "z"), "frontend")

    with pytest.raises(ValueError, match="2 questions with 3 answers"):
        _prompts(session_dir, events, questioner=_long).ask(questions)


def test_every_prompt_event_has_exactly_one_emitter() -> None:
    """The prompt/answer pairs are journaled in one place; a front-end that
    journals its own copy drifts (its own counter, its own idea of the gated
    call)."""
    emits = re.compile(r'(?:emit|journal)\(\s*"(approval|question)\.(prompt|answer)"')
    sites: dict[str, set[str]] = {}
    for path in _SRC.rglob("*.py"):
        for match in emits.finditer(path.read_text(encoding="utf-8")):
            event = f"{match.group(1)}.{match.group(2)}"
            sites.setdefault(event, set()).add(str(path.relative_to(_SRC)))
    assert sites == {
        "approval.prompt": {"tools/operator_prompts.py"},
        "approval.answer": {"tools/operator_prompts.py"},
        "question.prompt": {"tools/operator_prompts.py"},
        "question.answer": {"tools/operator_prompts.py"},
    }


def test_an_unseen_question_says_so_in_its_result(tmp_path: Path) -> None:
    """A headless run answered `ask_user` with bare empty strings, and the
    model asked again: the reason (nobody attached) reached the console only.
    The result carries it; a blank a person left does not."""
    from agent6.tools.operator_prompts import UNANSWERED_NOTE, unanswered_note

    session_dir = tmp_path / "s"
    session_dir.mkdir()
    events = _sink(session_dir)

    def _nobody(request: QuestionRequest, /) -> QuestionAnswer:
        return QuestionAnswer(tuple("" for _ in request.questions), "headless-default", unseen=True)

    def _blank(request: QuestionRequest, /) -> QuestionAnswer:
        return QuestionAnswer(tuple("" for _ in request.questions), "stdin")

    questions = (UserQuestion(question="which?", options=("a", "b")),)
    unseen = _prompts(session_dir, events, questioner=_nobody).ask(questions)
    assert unanswered_note(unseen) == UNANSWERED_NOTE
    assert unanswered_note(_prompts(session_dir, events, questioner=_blank).ask(questions)) == ""
    d = _dispatcher(session_dir, events, _prompts(session_dir, events, questioner=_nobody))
    wire = d.dispatch("ask_user", {"questions": [{"question": "which?"}]}).to_wire()
    assert wire == {"answers": [""], "note": UNANSWERED_NOTE}
