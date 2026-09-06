# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""`agent6 answer`: answer a live run's `ask_user` question from a script.

The sibling of `agent6 steer` for the other operator prompt: `steer` refuses a
run blocked on a question ("stays queued until that is answered"), and every
other way in (`attach`, the TUI, the web composer) needs a person at a screen.
This writes the same answer file those front-ends write, so a listener that
forwards questions somewhere else can send the reply back.
"""

from __future__ import annotations

import sys
from pathlib import Path

from agent6.sessions.id import SessionIdError
from agent6.sessions.ipc import (
    away_mode,
    frontend_is_live,
    worker_is_alive,
    write_question_answers,
)
from agent6.sessions.layout import LOGS_NAME
from agent6.ui.cli._common import resolve_session_layout
from agent6.viewmodel import QuestionPrompt, fold_session, tail_events


def _open_question(session_dir: Path) -> QuestionPrompt | None:
    """The run's unanswered `ask_user` prompt, oldest first; None when none is
    open."""
    state = fold_session(tail_events(session_dir / LOGS_NAME, follow=False))
    return next((q for q in state.pending_questions if not q.answered), None)


def _print_question(session_id: str, prompt: QuestionPrompt) -> None:
    print(f"{session_id} is waiting on {len(prompt.questions)} question(s):")
    for i, q in enumerate(prompt.questions, start=1):
        print(f"  {i}. {q.question}")
        if q.options:
            print(f"     options: {', '.join(q.options)}")
    print(f"\nanswer with: agent6 answer {session_id} {' '.join(['TEXT'] * len(prompt.questions))}")


def _unanswerable(session_dir: Path, session_id: str) -> str:
    """Why this run cannot take a written answer, or "".

    The run reads the answer file only for a live front-end or an away-mode of
    "wait"; a foreground run with a terminal is blocked on that terminal and
    never looks. Writing one there and printing "answered" left the operator
    waiting on a run that was waiting on them.
    """
    if not worker_is_alive(session_dir):
        return f"session {session_id} is not running; only a live run holds a question open."
    if not frontend_is_live(session_dir) and away_mode(session_dir) != "wait":
        return (
            f"{session_id} is waiting at its own terminal, which is where the answer"
            f" has to go: agent6 attach {session_id}"
        )
    return ""


def _cmd_answer(target: str, answers: tuple[str, ...]) -> int:
    """Answer the run's open question, or print it when no answer is given."""
    try:
        layout = resolve_session_layout(Path.cwd(), target)
    except SessionIdError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    if refusal := _unanswerable(layout.session_dir, layout.session_id):
        print(f"REFUSING: {refusal}", file=sys.stderr)
        return 2
    prompt = _open_question(layout.session_dir)
    if prompt is None:
        print(
            f"REFUSING: {layout.session_id} is not waiting on a question"
            f" (an approval is answered by attaching: agent6 attach {layout.session_id}).",
            file=sys.stderr,
        )
        return 2
    if not answers:
        _print_question(layout.session_id, prompt)
        return 0
    if len(answers) != len(prompt.questions):
        # Answers align to the prompt's questions by index, so a short list
        # would answer the wrong one and a long one would be silently cut.
        _print_question(layout.session_id, prompt)
        print(
            f"\nERROR: that prompt has {len(prompt.questions)} question(s);"
            f" {len(answers)} answer(s) given.",
            file=sys.stderr,
        )
        return 2
    write_question_answers(layout.session_dir, prompt.id, answers)
    print(f"answered {layout.session_id}: {', '.join(answers)}")
    return 0
