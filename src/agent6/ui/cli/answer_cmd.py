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
from agent6.sessions.ipc import worker_is_alive, write_question_answers
from agent6.ui.cli._common import error, refuse, resolve_session_layout
from agent6.viewmodel import QuestionPrompt, open_question


def _print_question(session_id: str, prompt: QuestionPrompt) -> None:
    print(f"{session_id} is waiting on {len(prompt.questions)} question(s):")
    for i, q in enumerate(prompt.questions, start=1):
        print(f"  {i}. {q.question}")
        if q.options:
            print(f"     options: {', '.join(q.options)}")
    print(f"\nanswer with: agent6 answer {session_id} {' '.join(['TEXT'] * len(prompt.questions))}")


def _cmd_answer(target: str, answers: tuple[str, ...]) -> int:
    """Answer the run's open question, or print it when no answer is given:
    the session, its liveness, then the question."""
    try:
        layout = resolve_session_layout(Path.cwd(), target)
    except SessionIdError as exc:
        error(f"{exc}")
        return 2
    if not worker_is_alive(layout.session_dir):
        refuse(
            f"session {layout.session_id} is not running; only a live run holds a question open."
        )
        return 2
    prompt = open_question(layout.session_dir)
    if prompt is None:
        refuse(
            f"{layout.session_id} is not waiting on a question (an approval is answered"
            f" by attaching: agent6 attach {layout.session_id})."
        )
        return 2
    if not answers or len(answers) != len(prompt.questions):
        # Answers align to the prompt's questions by index, so a short list
        # would answer the wrong one and a long one would be silently cut.
        _print_question(layout.session_id, prompt)
        if not answers:
            return 0
        print(
            f"\nERROR: that prompt has {len(prompt.questions)} question(s);"
            f" {len(answers)} answer(s) given.",
            file=sys.stderr,
        )
        return 2
    write_question_answers(layout.session_dir, prompt.id, answers)
    print(f"answered {layout.session_id}: {', '.join(answers)}")
    return 0
