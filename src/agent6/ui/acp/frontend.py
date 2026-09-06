# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The `SessionFrontend` an ACP client provides.

Every prompt the lifecycle raises becomes a `session/request_permission` to
the editor; everything a terminal front-end would draw becomes nothing, because
an ACP client renders from `session/update` instead.

A client that declared it cannot be asked is never asked: the answer comes from
the CAUTIOUS default rather than a hang or an invented yes. That is what
`FrontendCapabilities` is for, and why an editor with no way to show a prompt
still gets a working session -- one where the model simply has fewer powers.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from contextlib import nullcontext
from pathlib import Path
from typing import Protocol

from agent6.app.frontend import FrontendCapabilities, SessionFacts, SessionFrontend, SteerHooks
from agent6.budget import BudgetTracker
from agent6.config import Config
from agent6.events import EventSink
from agent6.sessions.ipc import (
    answer_written,
    question_answers_written,
    read_answer,
    read_question_answers,
    record_answer,
)
from agent6.sessions.layout import SessionLayout
from agent6.tools.operator_prompts import (
    ApprovalAnswer,
    ApprovalRequest,
    Approver,
    QuestionAnswer,
    Questioner,
    QuestionRequest,
)
from agent6.types import AutoCommitDirective, IsolationLevel
from agent6.ui.steer import file_bridge_steer
from agent6.workflows.loop import SessionResult, Workflow

# How long a permission request waits for the editor. An operator who has
# walked away must not hold a run forever, and the seam already reads silence
# as the cautious answer: an approval becomes a denial, a question becomes no
# answer at all.
PERMISSION_TIMEOUT_S = 300.0


# What the client is asked, and what an unaskable client is assumed to have
# said. Every one of these is the CAUTIOUS answer: a session that cannot ask
# is a session that does less, never one that does something unwatched.
# (prompt, options, standing, call_id) -> the chosen option, or None for no
# answer. `standing` is None for a QUESTION, whose options the model wrote: an
# answer among several is not a permission, and must never be offered as one
# the editor may remember. `call_id` is the dispatcher's stamp on the tool
# call the prompt gates (the `call_id` its journaled prompt carries), or None
# for a prompt that gates no call (a pre-run question, a verify the harness
# runs itself). The keyword `until`, polled while the editor's answer is
# pending, ends the wait with None: the question was answered by another
# route (the session's answer file, which every other seat writes).
class Asker(Protocol):
    def __call__(
        self,
        prompt: str,
        options: tuple[str, ...],
        standing: bool | None,
        call_id: int | None,
        until: Callable[[], bool] | None = None,
        /,
    ) -> str | None: ...


def acp_frontend(
    *,
    ask: Asker,
    capabilities: FrontendCapabilities,
    agent6_exe: Callable[[], str],
    spawn_detached_resume: Callable[[Path, str, Sequence[str]], str],
) -> SessionFrontend:
    """Wire the lifecycle to one ACP client."""

    def _approve(
        prompt: str,
        /,
        *,
        scope: str | None = None,
        call_id: int | None = None,
        until: Callable[[], bool] | None = None,
    ) -> bool | None:
        """The editor's verdict; None when it gave none (a timeout, or *until*
        held first)."""
        if not capabilities.can_ask:
            return False  # nobody to ask, so the answer is no
        # No scope means an "always allow" the editor remembers must NOT cover
        # this one -- the fetch tool's off-list host, where a GET can carry data
        # out in its path. The option names carry it, because an editor that
        # offers "always" needs something to key that decision on.
        standing = scope is not None
        options = ("allow", "deny") if standing else ("allow once", "deny")
        answer = ask(prompt, options, standing, call_id, until)
        return None if answer is None else answer.startswith("allow")

    def _build_approver(session_dir: Path) -> Approver:
        def approve(request: ApprovalRequest, /) -> ApprovalAnswer:
            # A client that cannot be asked denies as a headless run does, and
            # the journal names that: nobody answered.
            if not capabilities.can_ask:
                return ApprovalAnswer(False, "headless")
            approved = _approve(
                request.prompt,
                scope=request.scope,
                call_id=request.call_id,
                until=lambda: answer_written(session_dir, request.id),
            )
            if approved is None:
                # The answer file: `agent6 answer`, the web, an attached TUI.
                filed = read_answer(session_dir, request.id, timeout_s=0.0)
                if filed is not None:
                    return ApprovalAnswer(
                        record_answer(session_dir, filed, request.scope), "frontend"
                    )
                return ApprovalAnswer(False, "acp")
            return ApprovalAnswer(approved, "acp")

        return approve

    def _build_questioner(session_dir: Path) -> Questioner:
        def ask_questions(request: QuestionRequest, /) -> QuestionAnswer:
            if not capabilities.can_ask:
                return QuestionAnswer(tuple("" for _ in request.questions), "headless", unseen=True)
            # An unanswered question becomes an empty string, which the loop
            # already treats as "the operator said nothing", not as a value.
            # One deadline for the request: a timeout per question made an
            # N-question ask wait N times the documented bound.
            deadline = time.monotonic() + PERMISSION_TIMEOUT_S
            answers: list[str] = []
            for question in request.questions:
                answer = ask(
                    question.question,
                    question.options,
                    None,
                    request.call_id,
                    lambda: (
                        question_answers_written(session_dir, request.id)
                        or time.monotonic() >= deadline
                    ),
                )
                if answer is None:
                    filed = read_question_answers(session_dir, request.id, timeout_s=0.0)
                    if filed is not None:
                        return QuestionAnswer(filed, "frontend")
                answers.append(answer or "")
            return QuestionAnswer(tuple(answers), "acp")

        return ask_questions

    def _confirm_unconfined(isolation: IsolationLevel, cfg: Config) -> bool:
        """Only ask when it is actually true.

        The lifecycle calls this on EVERY run; the "is this dangerous" test
        lives in the answer, not the call. Asking regardless told the editor a
        confined run was unsandboxed -- a false statement about the run, on the
        one approval that must never become reflexive.
        """
        if isolation != "none" or cfg.sandbox.run_commands != "yes":
            return True
        # No scope: docs/security.md documents this as a ONE-TIME gate, and
        # ACP's `allow_always` is exactly the button that would let one click
        # silence it for every later session.
        return bool(_approve("Run commands UNSANDBOXED on this host, with no per-command prompt?"))

    def _steer(
        _events: EventSink, session_dir: Path, _facts: Callable[[], SessionFacts]
    ) -> SteerHooks:
        # The file bridge: a later prompt on this session resumes the run with
        # its text seeded through the steer files (resume --steer), and the
        # loop's pre-call drain reads THESE hooks, so the seeded instruction
        # reaches the resumed model. Mid-run nothing here writes steer files,
        # so no new affordance is offered.
        return file_bridge_steer(session_dir)

    def _no_repl(
        _session_dir: Path, _budget: BudgetTracker, _task: str, _mcp: object
    ) -> Callable[[int, str], AutoCommitDirective]:
        # ACP has its own turn loop; an interactive REPL inside it would be a
        # second one, with two things reading the same stdin. The hook exists
        # and always continues.
        return lambda _iteration, _summary: "continue"

    def _no_ask_repl(
        _wf: Workflow, _budget: BudgetTracker, _layout: SessionLayout, _task: str
    ) -> SessionResult:
        raise RuntimeError("an ACP session drives its own turns; the ask REPL is not used")

    return SessionFrontend(
        capabilities=capabilities,
        should_spawn_tui=lambda _tui, _interactive, _mode: False,
        # Stream the deltas as events (session/update reads them); the editor
        # is the live view, so the ending's headline and summary are the
        # fold's done item, and the console view a terminal would attach is
        # nothing here.
        stream_modes=lambda _tui_enabled: (True, True),
        attach_console_view=lambda _events: None,
        close_console_view=lambda: None,
        loop_logger=lambda _mode: lambda _line: None,
        tui_session=lambda _session_dir, _enabled: nullcontext(),
        build_approver=_build_approver,
        build_questioner=_build_questioner,
        make_steer_state=_steer,
        confirm_unconfined_autorun=_confirm_unconfined,
        confirm_run_on_run_branch=lambda branch: bool(
            _approve(f"Continue this run on {branch!r}, which is already a run branch?")
        ),
        confirm_replay_after_crash=lambda iteration, tools: bool(
            _approve(
                f"The previous run died mid-turn (iteration {iteration};"
                f" {', '.join(tools) or 'unknown tools'}). Its tools may have partially"
                " applied; replaying can repeat a non-idempotent effect. Re-run the turn?"
            )
        ),
        prompt_detach_away_mode=lambda _session_dir, _scopes: None,
        select_revised_prompt=lambda _original, _revised, _notes: None,
        build_repl_hook=_no_repl,
        run_ask_repl=_no_ask_repl,
        save_ask_transcript=lambda _layout, _question, _answer: None,
        build_coordinator_spawner=_no_coordinator,
        agent6_exe=agent6_exe,
        spawn_detached_resume=spawn_detached_resume,
    )


def _no_coordinator(
    _cfg: Config,
    _cwd: Path,
    _state_dir: Path,
    _mode: str,
    _session_id: str,
    _max_usd: float | None,
    _auto_approve: bool,
) -> None:
    """`/parallel` fans out sibling runs, which need somewhere to be watched.
    An ACP client renders ONE session; lanes would run invisibly."""
    return None
