# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The one gate every prompt to the operator goes through.

`OperatorPrompts` mints the prompt ids, clears a prompt's answer slot, journals
`approval.prompt` / `question.prompt` before asking and `approval.answer` /
`question.answer` after, and names the tool call a prompt gates. A front-end
supplies only the two callables that ANSWER (`Approver`, `Questioner`) and say
which source answered; it never journals.
"""

from __future__ import annotations

import itertools
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol

from agent6.sessions.ipc import clear_answer, clear_question_answers, session_allow_set
from agent6.tools.schema import UserQuestion

# Who answered, as `approval.answer` / `question.answer` journal it. The CLI:
# "stdin" (its own terminal), "frontend" (a live TUI, web, or attach answering
# by file), "await-frontend" (parked until one attached), "away-deny" (the
# detach choice; approvals only), "away-wait" (the park; questions only),
# "headless-default" (no terminal and no front-end: empty answers). The gate:
# "session" (a standing grant answered). A machine state: "headless" (nobody to
# ask: denied, or empty). An editor over ACP: "acp" ("headless" when it declared
# it cannot be asked).
Source = Literal[
    "stdin",
    "frontend",
    "await-frontend",
    "away-deny",
    "away-wait",
    "headless-default",
    "session",
    "headless",
    "acp",
]

UNANSWERED_NOTE = (
    "no operator is attached to this run, so the questions went unanswered;"
    " decide on your own judgment and go on"
)


@dataclass(frozen=True, slots=True)
class ApprovalRequest:
    """One approval put to the operator, as journaled in `approval.prompt`."""

    # The answer slot's key: a file-bridge front-end answers into
    # `approvals/<id>.answer`.
    id: str
    prompt: str
    # What a standing answer grants: "command" for the command tools,
    # "mcp.<server>" for one server's. None offers no standing answer, so the
    # operator is asked every time (`fetch`). Nothing may read one scope's
    # grant as consent for another.
    scope: str | None
    # The dispatched tool call the prompt gates; None for one gating no call
    # (a verify the harness runs itself).
    call_id: int | None


@dataclass(frozen=True, slots=True)
class QuestionRequest:
    """One `ask_user` (or a pre-run question), as journaled in `question.prompt`."""

    id: str
    questions: tuple[UserQuestion, ...]
    call_id: int | None


@dataclass(frozen=True, slots=True)
class ApprovalAnswer:
    approved: bool
    source: Source


@dataclass(frozen=True, slots=True)
class QuestionAnswer:
    # Aligned to the request's questions by index; a shorter tuple leaves the
    # rest unanswered ("").
    answers: tuple[str, ...]
    source: Source
    # Nobody could see the questions: no terminal and no front-end, or a park
    # that ended empty. A person who left them blank is not unseen.
    unseen: bool = False


class Approver(Protocol):
    """Answers one approval and says who answered."""

    def __call__(self, request: ApprovalRequest, /) -> ApprovalAnswer: ...


class Questioner(Protocol):
    """Answers one question set and says who answered."""

    def __call__(self, request: QuestionRequest, /) -> QuestionAnswer: ...


class Journal(Protocol):
    """Where the gate's events go: a run's `EventSink.emit`."""

    def __call__(self, event_type: str, /, **fields: Any) -> None: ...


def unjournaled(event_type: str, /, **fields: Any) -> None:
    """The default journal: nothing is written (a gate built with no sink)."""


def _default_approver(request: ApprovalRequest, /) -> ApprovalAnswer:  # pragma: no cover
    try:
        ans = input(f"{request.prompt} [y/N] ").strip().lower()
    except EOFError:
        return ApprovalAnswer(False, "stdin")
    return ApprovalAnswer(ans in {"y", "yes"}, "stdin")


def _default_questioner(request: QuestionRequest, /) -> QuestionAnswer:  # pragma: no cover
    """Fallback for `ask_user` when no front-end is wired: numbered stdin
    prompts, one per question. A non-TTY/headless stdin answers "" for each so a
    run never hangs (mirrors ui/cli/_interact.py's default_stdin_questioner)."""
    if not sys.stdin.isatty():
        return QuestionAnswer(tuple("" for _ in request.questions), "headless-default", unseen=True)
    answers: list[str] = []
    for q in request.questions:
        lines = [q.question, *(f"  {i}) {opt}" for i, opt in enumerate(q.options, start=1))]
        try:
            ans = input("\n".join(lines) + "\n> ").strip()
        except EOFError:
            ans = ""
        if ans.isdigit() and 1 <= int(ans) <= len(q.options):
            ans = q.options[int(ans) - 1]
        answers.append(ans)
    return QuestionAnswer(tuple(answers), "stdin")


class OperatorPrompts:
    """Every prompt to the operator for one leg, journaled once.

    `journal` takes every event the gate writes (a run's `EventSink.emit`).
    `session_dir` is where the file bridge lives (`sessions.ipc`: the answer
    slots and the operator's standing choices): the run dir, or a machine
    state's dir. None keeps no bridge (a bare dispatcher, a test).
    """

    def __init__(
        self,
        *,
        approver: Approver | None = None,
        questioner: Questioner | None = None,
        journal: Journal = unjournaled,
        session_dir: Path | None = None,
    ) -> None:
        self._approver: Approver = approver or _default_approver
        self._questioner: Questioner = questioner or _default_questioner
        self._journal = journal
        self._session_dir = session_dir
        self._approvals = itertools.count(1)
        self._questions = itertools.count(1)

    def approve(self, prompt: str, *, scope: str | None = None, call_id: int | None = None) -> bool:
        """Ask, unless a standing grant for *scope* already answers.

        The answer slot is cleared BEFORE the prompt is journaled: ids are
        predictable counters, so an answer written ahead of its prompt (a
        premature approve POST) must never be the one consumed, and only the
        process that journals knows the exact moment. `standing` tells every
        front-end whether to OFFER an "allow all": a button that silently
        answered only this call would lie about itself.
        """
        request = ApprovalRequest(
            id=f"approval-{next(self._approvals)}", prompt=prompt, scope=scope, call_id=call_id
        )
        if scope and self._session_dir is not None and session_allow_set(self._session_dir, scope):
            self._journal("approval.answer", id=request.id, approved=True, source="session")
            return True
        if self._session_dir is not None:
            clear_answer(self._session_dir, request.id)
        self._journal(
            "approval.prompt",
            id=request.id,
            prompt=prompt,
            standing=bool(scope),
            call_id=call_id,
        )
        answer = self._approver(request)
        self._journal(
            "approval.answer", id=request.id, approved=answer.approved, source=answer.source
        )
        return answer.approved

    def ask(
        self, questions: tuple[UserQuestion, ...], *, call_id: int | None = None
    ) -> QuestionAnswer:
        """Put *questions* to the operator; the answers align to them by index
        (a question the front-end left unanswered is ""), with who answered."""
        request = QuestionRequest(
            id=f"question-{next(self._questions)}", questions=questions, call_id=call_id
        )
        if self._session_dir is not None:
            clear_question_answers(self._session_dir, request.id)
        self._journal(
            "question.prompt",
            id=request.id,
            questions=[{"question": q.question, "options": list(q.options)} for q in questions],
            call_id=call_id,
        )
        answer = self._questioner(request)
        if len(answer.answers) > len(questions):
            raise ValueError(
                f"{answer.source} answered {len(questions)} questions"
                f" with {len(answer.answers)} answers"
            )
        answers = answer.answers + ("",) * (len(questions) - len(answer.answers))
        self._journal("question.answer", id=request.id, answers=list(answers), source=answer.source)
        return QuestionAnswer(answers, answer.source, answer.unseen)


def unanswered_note(answer: QuestionAnswer) -> str:
    """What the model is told when nobody saw its questions; "" when someone
    answered, or declined by leaving them blank."""
    return UNANSWERED_NOTE if answer.unseen else ""
