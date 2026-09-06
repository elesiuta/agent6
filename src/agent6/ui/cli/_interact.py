# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Operator interaction for a live run: the `run_command` approver, the
`ask_user` questioner, their /dev/tty fallbacks, and the detach away-mode
(deny / wait / spawn the background resume)."""

from __future__ import annotations

import contextlib
import os
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

from agent6.sessions.ipc import (
    answer_written,
    await_frontend_reply,
    away_mode,
    frontend_is_live,
    question_answers_written,
    read_answer,
    read_question_answers,
    record_answer,
    set_away_mode,
    set_session_allow,
)
from agent6.tools.operator_prompts import (
    ApprovalAnswer,
    ApprovalRequest,
    Approver,
    QuestionAnswer,
    Questioner,
    QuestionRequest,
    Source,
)
from agent6.tools.schema import UserQuestion
from agent6.ui.cli._console_view import ConsoleView
from agent6.ui.cli._steer import (
    tty_message,
    tty_prompt,
)
from agent6.ui.steer import SteerState
from agent6.viewmodel import approval_parts
from agent6.viewmodel.transcript import scrub_terminal_controls


def _pause(cv: ConsoleView | None) -> contextlib.AbstractContextManager[None]:
    """Pause the live console spinner around an interactive /dev/tty prompt so it
    cannot erase the question and the operator's keystrokes. No-op when headless
    (no ConsoleView: a TUI-bridged, detached, or piped run)."""
    return cv.pause() if cv is not None else contextlib.nullcontext()


def _has_controlling_tty() -> bool:
    """True iff a controlling terminal exists (so the stdin approver can actually
    prompt). A foreground run has one; a web/hub-spawned or fully headless run
    does not, and there falls back to waiting for a front-end rather than a
    no-terminal deny."""
    try:
        fd = os.open("/dev/tty", os.O_RDWR | os.O_NOCTTY)
    except OSError:
        return False
    os.close(fd)
    return True


def default_stdin_approver(
    prompt: str, *, standing: bool = True, until: Callable[[], bool] | None = None
) -> str | None:
    """Plain-terminal fallback for tool approval (no live TUI, or its answer
    timed out). Returns "yes", "no", "session" (allow all of this prompt's scope
    for the rest of the run) or "session-deny" (withhold that scope for it);
    None when nothing was typed (no terminal, or `until` held first).

    A plain y/n answers ONE call, either way; only the two session choices
    persist, and they mirror each other. `standing=False` is a gate that has no
    session answer to give (`fetch`), so it does not offer one. Routed via
    /dev/tty so the prompt stays visible when a TUI has redirected the std
    streams to its console log.

    Every dispatch prompt is "Allow <tool>: <payload>"; the payload renders on
    its own indented lines with a blank line before the answer line, so the
    input point stands clear of a long or wrapped command. The console's
    vocabulary marks it: a bold yellow `?` and bold header (the question),
    the command plain (the thing under judgment), the answer line dim --
    the sibling of the `->` call line that follows an allow."""
    suffix = "[y/N/a/d]  (a = allow all, d = deny all, this session): " if standing else "[y/N]: "
    bold, dim, yellow, reset = "\033[1m", "\033[2m", "\033[33m", "\033[0m"
    # The text under judgment carries no sequence at all, styling included:
    # conceal (SGR 8) hides the part of a command it wraps.
    head, payload = approval_parts(scrub_terminal_controls(prompt))
    if payload:
        body = "\n".join(f"    {ln}" for ln in payload.splitlines())
        rendered = (
            f"{bold}{yellow}?{reset} {bold}{head}:{reset}\n\n{body}\n\n  {dim}{suffix}{reset}"
        )
        plain = f"? {head}:\n\n{body}\n\n  {suffix}"
    else:
        rendered = plain = f"{prompt} {suffix}"
    # /dev/tty is a terminal by definition; the stdin fallback prints to
    # stdout, which may be a pipe, so it gets the text without escapes.
    ans = tty_prompt(rendered, plain=plain, until=until)
    if ans is None:
        return None
    ans = ans.strip().lower()
    if standing and ans in {"a", "all", "always", "session"}:
        return "session"
    if standing and ans in {"d", "deny", "never"}:
        return "session-deny"
    return "yes" if ans in {"y", "yes"} else "no"


def prompt_detach_away_mode(session_dir: Path, scopes: tuple[str, ...]) -> None:
    """On detach with run_commands=ask, ask how approvals/questions should be
    handled while nothing is watching, and record it for the background run.

    "Approve all" grants every scope in play (`scopes`): the command tools and
    each configured MCP server. Granting only one would leave the run blocked
    on the first prompt from another, with nobody there to answer.

    The default is WAIT: a deny throws away the run's work (the model's commands
    are refused and it flails, burning tokens for nothing), while wait pauses
    cleanly at the approval and is resumable -- re-attach with `agent6 attach`
    and answer. Non-interactive (no tty) also defaults to wait."""
    if not sys.stdin.isatty():
        set_away_mode(session_dir, "wait")
        return
    print(
        "[agent6] Detaching with run_commands=ask; nothing will be watching to approve.",
        file=sys.stderr,
    )
    ans = tty_prompt(
        "  While away: [w]ait for a reattached front-end / [a]pprove all / [d]eny all? [w]: ",
        fall_back_to_stdin=False,
    )
    choice = (ans or "").strip().lower()
    if choice in {"a", "approve"}:
        for scope in scopes:
            set_session_allow(session_dir, scope)
        covered = "run_command and MCP tool call" if len(scopes) > 1 else "run_command"
        print(f"  -> approving every {covered}.", file=sys.stderr)
    elif choice in {"d", "deny"}:
        set_away_mode(session_dir, "deny")
        print("  -> denying run_commands until you reattach.", file=sys.stderr)
    else:
        set_away_mode(session_dir, "wait")
        print("  -> waiting; reattach (agent6 attach / the TUI) to approve.", file=sys.stderr)


def build_approver(
    session_dir: Path,
    console_cell: Sequence[ConsoleView | None] | None = None,
    steer_cell: Sequence[SteerState | None] | None = None,
) -> Approver:
    """Build the command approver, bridged to a live TUI when present.

    The gate has journaled the prompt (`approval.prompt`) before this is
    asked; if a front-end is live (it wrote a `frontends/` claim) the answer
    comes from its Allow/Deny modal via the file bridge
    (`approvals/<id>.answer`), otherwise -- or if the front-end dies / times
    out -- it falls back to the stdin `[y/N]` prompt. That prompt reads the
    same file while it waits, so `agent6 answer`, the web, or a front-end
    attached after the question was put to the terminal answers it too: one
    bridge, whichever seat the operator is in.

    `console_cell` and `steer_cell` are the CLI leg's late-bound console view
    and SteerState, read at prompt time: the view pauses its heartbeat around
    the terminal prompt, and an operator prompt counts as a Ctrl-C boundary,
    so with a pause armed the prompt says so and the pause menu opens right
    after the answer (its action seeds the steer the next between-steps
    boundary consumes)."""

    def approve(request: ApprovalRequest, /) -> ApprovalAnswer:
        # A live front-end ALWAYS gets asked, in its own UI, regardless of the
        # detach away-mode: away-mode governs only the window when nothing is
        # attached. (A foreground run writes no front-end claim, so it falls through
        # to the stdin prompt below.)
        if frontend_is_live(session_dir):
            answer = read_answer(session_dir, request.id)
            if answer is not None:
                return ApprovalAnswer(record_answer(session_dir, answer, request.scope), "frontend")
        # Nothing attached (or the front-end died mid-prompt): the detached run's
        # chosen away-mode governs. deny/wait are only reached headless.
        away = away_mode(session_dir)
        if away == "deny":
            return ApprovalAnswer(False, "away-deny")
        wait_for_frontend = away == "wait" or not _has_controlling_tty()
        if wait_for_frontend:
            # away="wait", OR an unattended run with no away-mode and no terminal
            # (a web/hub-spawned run whose viewers have all left): block until a
            # front-end attaches and answers, rather than deny. Deny discards the
            # run's work; wait pauses cleanly and is resumable (the default).
            tty_message(
                f"[agent6] waiting: an approval awaits a front-end; answer it with:"
                f" agent6 attach {session_dir.name}\n"
            )
            reply = await_frontend_reply(
                session_dir,
                lambda: read_answer(session_dir, request.id, timeout_s=20.0, dead_grace_s=8.0),
            )
            approved = reply is not None and record_answer(session_dir, reply, request.scope)
            return ApprovalAnswer(approved, "await-frontend")
        # Foreground (a controlling tty, no away-mode): prompt on it directly.
        steer = steer_cell[0] if steer_cell else None
        # The view is attached after the prompts are built: read it now.
        with _pause(console_cell[0] if console_cell else None):
            if steer is not None and steer.armed():
                tty_message("\n[agent6] pause armed: the menu opens after this answer.\n")
            answer_s = default_stdin_approver(
                request.prompt,
                standing=bool(request.scope),
                until=lambda: answer_written(session_dir, request.id),
            )
        source: Source = "stdin"
        if answer_s is None:
            filed = read_answer(session_dir, request.id, timeout_s=0.0)
            if filed is None:
                answer_s = "no"
            else:
                tty_message("[agent6] answered elsewhere.\n")
                answer_s, source = filed, "frontend"
        # A session choice persists (across this run's resumes); session-deny
        # WITHDRAWS the scope's tools from the next turn rather than refusing
        # every later call, so the model stops spending turns on a door that
        # will not open.
        approved = record_answer(session_dir, answer_s, request.scope)
        if steer is not None and steer.armed():
            steer.prompt_now()
        return ApprovalAnswer(approved, source)

    return approve


def build_questioner(
    session_dir: Path, console_cell: Sequence[ConsoleView | None] | None = None
) -> Questioner:
    """Build the `ask_user` questioner, bridged to a live TUI when present.

    The gate has journaled the prompt (`question.prompt`) before this is
    asked; if a TUI is live the answer comes from its question modal via
    `questions/<id>.answer`, otherwise (or if the TUI dies / times out) it
    falls back to a numbered stdin prompt, which reads that file while it
    waits (see `build_approver`). A headless run (no TUI, no TTY) gets an
    empty answer rather than hanging."""

    def ask(request: QuestionRequest, /) -> QuestionAnswer:
        questions = request.questions
        # A live front-end (re-attached CLI watch, TUI, web) always gets asked,
        # whatever the away-mode; away-mode is the no-front-end fallback.
        if frontend_is_live(session_dir):
            answers = read_question_answers(session_dir, request.id)
            if answers is not None:
                return QuestionAnswer(answers, "frontend")
        if away_mode(session_dir) == "wait":
            # Detached 'wait', nothing attached: block until a front-end answers.
            tty_message(
                f"[agent6] waiting: a question awaits a front-end; answer it with:"
                f" agent6 attach {session_dir.name}"
                f" (or, with no terminal: agent6 answer {session_dir.name} TEXT)\n"
            )
            reply = await_frontend_reply(
                session_dir,
                lambda: read_question_answers(
                    session_dir, request.id, timeout_s=20.0, dead_grace_s=8.0
                ),
            )
            if isinstance(reply, tuple):
                return QuestionAnswer(reply, "frontend")
            return QuestionAnswer(tuple("" for _ in questions), "away-wait", unseen=True)
        with _pause(console_cell[0] if console_cell else None):
            stdin_answers = default_stdin_questioner(
                questions, until=lambda: question_answers_written(session_dir, request.id)
            )
        if stdin_answers is None:
            filed = read_question_answers(session_dir, request.id, timeout_s=0.0)
            if filed is not None:
                tty_message("[agent6] answered elsewhere.\n")
                return QuestionAnswer(filed, "frontend")
            # No front-end and no controlling terminal: nobody saw the
            # question. Answer empty so the run never hangs, and say so
            # where a watcher will see it instead of failing silently.
            tty_message(
                "[agent6] no front-end attached and no terminal to answer the"
                " question; returning empty answers\n"
            )
            return QuestionAnswer(tuple("" for _ in questions), "headless-default", unseen=True)
        return QuestionAnswer(stdin_answers, "stdin")

    return ask


def ask_one_stdin(
    q: UserQuestion, prefix: str = "", until: Callable[[], bool] | None = None
) -> str | None:
    """Prompt one question on /dev/tty; a digit picks an option, else free text.
    None means no terminal (headless), or `until` held first."""
    lines = [
        f"{prefix}{q.question}",
        *(f"  {i}) {opt}" for i, opt in enumerate(q.options, start=1)),
    ]
    ans = tty_prompt("\n".join(lines) + "\n> ", fall_back_to_stdin=False, until=until)
    if ans is None:
        return None
    ans = ans.strip()
    if ans.isdigit() and 1 <= int(ans) <= len(q.options):
        return q.options[int(ans) - 1]
    return ans


def default_stdin_questioner(
    questions: tuple[UserQuestion, ...], until: Callable[[], bool] | None = None
) -> tuple[str, ...] | None:
    """Ask each question on /dev/tty (visible under a TUI's stream redirect). For a
    series, print a summary afterwards and let the operator revise any answer (type
    its number) before submitting (blank). Returns None without a controlling
    terminal (headless) so the caller can answer empty -- never hanging or eating
    piped stdin -- and say so, or once `until` holds (the whole prompt was
    answered by another route)."""
    answers: list[str] = []
    multi = len(questions) > 1
    for i, q in enumerate(questions, start=1):
        prefix = f"[{i}/{len(questions)}] " if multi else ""
        ans = ask_one_stdin(q, prefix, until)
        if ans is None:
            return None  # no tty: never block
        answers.append(ans)
    while multi:  # review + revise loop; blank submits
        summary = "\n".join(
            f"  {n}) {q.question} -> {a or '(empty)'}"
            for n, (q, a) in enumerate(zip(questions, answers, strict=True), start=1)
        )
        pick = tty_prompt(
            f"Review:\n{summary}\nEnter to submit, or a number to change that answer: ",
            fall_back_to_stdin=False,
            until=until,
        )
        if pick is None:
            if until is not None and until():
                return None
            break
        if not pick.strip():
            break
        if pick.strip().isdigit() and 1 <= int(pick.strip()) <= len(questions):
            j = int(pick.strip()) - 1
            revised = ask_one_stdin(questions[j], until=until)
            if revised is None:
                return None
            answers[j] = revised
    return tuple(answers)
