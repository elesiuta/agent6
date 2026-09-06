# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Mid-run steering: a SIGINT handler that lets the operator pause the loop
and inject a one-shot instruction (or abort), plus interactive revised-prompt
selection. Independent of the run command; run.py wires it in.
"""

from __future__ import annotations

import contextlib
import io
import os
import shlex
import signal
import subprocess
import sys
import tempfile
import termios
from collections.abc import Callable, Generator
from pathlib import Path
from typing import Any

from agent6.app.frontend import SessionFacts
from agent6.events import EventSink
from agent6.sessions.ipc import (
    clear_steer_answer,
    clear_steer_request,
    frontend_is_live,
    read_steer_answer,
    steer_answer_is_abort,
    steer_answer_written,
    steer_interrupt_pending,
    steer_request_pending,
    take_steer_answer,
    write_steer_answer,
)
from agent6.ui.cli._console_view import ConsoleView
from agent6.ui.cli._menu_input import menu_capable, read_line_until
from agent6.ui.cli._steer_menu import BtwRunner, normalize_steer_choice, pause_menu
from agent6.ui.steer import SteerState, file_bridge_steer
from agent6.viewmodel.format import format_usd


@contextlib.contextmanager
def repl_prompt_sigint() -> Generator[None]:
    """Default Ctrl-C for the duration of an idle REPL prompt (ask> /
    agent6>). No step is in flight there, so the run's escalating steer
    handler would lie ("pausing after this step"), PEP 475 would retry the
    interrupted input() (three presses to leave), and the armed stage would
    open a phantom pause menu on the next question. The escalation applies
    only while a leg runs; at the prompt one Ctrl-C simply raises."""
    prev = signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGINT, signal.default_int_handler)
    try:
        yield
    finally:
        signal.signal(signal.SIGINT, prev)


def select_revised_prompt(
    original: str,
    revised: str,
    questions: tuple[str, ...],
    console_view: ConsoleView | None = None,
) -> str | None:
    """Interactive accept/edit/skip prompt for prompt.revise_prompt.

    `console_view`, when given, has its heartbeat suspended for the whole
    exchange (the spinner's per-tick line-erase otherwise wipes the choice
    prompt and the typed echo while the operator reads the proposal)."""
    pause = console_view.pause if console_view is not None else contextlib.nullcontext
    with pause():
        return _select_revised_prompt(original, revised, questions)


def _select_revised_prompt(
    original: str,
    revised: str,
    questions: tuple[str, ...],
) -> str | None:
    print("\n[agent6] prompt revision proposed:", file=sys.stderr)
    print("\n--- revised ---", file=sys.stderr)
    print(revised, file=sys.stderr)
    if questions:
        print("\n--- clarifying questions ---", file=sys.stderr)
        for question in questions:
            print(f"- {question}", file=sys.stderr)
    print("\n--- original ---", file=sys.stderr)
    print(original, file=sys.stderr)
    while True:
        try:
            choice = (
                input("[agent6] revise_prompt: [a]ccept, [o]riginal, [e]dit, [q]uit? ")
                .strip()
                .lower()
            )
        except (EOFError, KeyboardInterrupt):
            return None
        if choice in {"", "a", "accept", "y", "yes"}:
            return revised
        if choice in {"o", "orig", "original", "s", "skip"}:
            return original
        if choice in {"q", "quit", "abort"}:
            return None
        if choice in {"e", "edit"}:
            # $EDITOR may be a command with flags ("code --wait"); split it,
            # and a missing binary is a choose-again, not a run-killing crash.
            editor = os.environ.get("EDITOR", "vi")
            editor_argv = shlex.split(editor) or ["vi"]
            with tempfile.NamedTemporaryFile(
                "w",
                encoding="utf-8",
                prefix="agent6-revised-task-",
                suffix=".md",
                delete=False,
            ) as tmp:
                tmp_path = Path(tmp.name)
                tmp.write(revised.rstrip() + "\n")
            try:
                try:
                    result = subprocess.run([*editor_argv, str(tmp_path)], check=False)
                except OSError as exc:
                    print(
                        f"[agent6] cannot run $EDITOR ({editor!r}): {exc}; choose again.",
                        file=sys.stderr,
                    )
                    continue
                if result.returncode != 0:
                    print(
                        f"[agent6] editor exited {result.returncode}; choose again.",
                        file=sys.stderr,
                    )
                    continue
                edited = tmp_path.read_text(encoding="utf-8").strip()
            finally:
                with contextlib.suppress(OSError):
                    tmp_path.unlink()
            if edited:
                return edited
            print("[agent6] edited prompt was empty; choose again.", file=sys.stderr)
            continue
        print("[agent6] choose accept, original, edit, or quit.", file=sys.stderr)


def tty_message(text: str) -> None:
    """Print to the controlling terminal directly, bypassing any stdout/stderr
    redirection (the TUI redirects the run's std streams to a log file)."""
    try:
        with open("/dev/tty", "w", encoding="utf-8") as tty:  # noqa: PTH123
            tty.write(text)
            tty.flush()
            return
    except OSError:
        with contextlib.suppress(Exception):
            print(text, file=sys.stderr, flush=True)


def tty_prompt(
    text: str,
    *,
    fall_back_to_stdin: bool = True,
    plain: str | None = None,
    until: Callable[[], bool] | None = None,
) -> str | None:
    """Prompt on the controlling terminal directly (see `tty_message`).
    Falls back to stdin when there is no controlling terminal, unless the
    caller must never consume piped stdin (`fall_back_to_stdin=False`:
    return None); the fallback prints `plain` when given (the text without
    terminal escapes, since stdout may be a pipe). `until` is polled while
    the prompt waits: once it holds, the prompt ends with None and a partly
    typed line is discarded (the answer arrived by another route)."""
    try:
        # The getpass recipe: O_RDWR on the device + an unbuffered FileIO.
        # A plain open("/dev/tty", "r+") NEVER works -- buffered update mode
        # requires a seekable stream and a tty is not -- so every /dev/tty
        # prompt silently used the stdin fallback (or, without the fallback,
        # returned no answer at all).
        fd = os.open("/dev/tty", os.O_RDWR | os.O_NOCTTY)
        # Discard type-ahead before prompting (the sudo/ssh rule): text typed
        # before this prompt existed was aimed at something else -- e.g. a
        # pause-menu command typed during the "pausing after this step" window
        # must not ride into a run_command [y/N/a] approval as its answer.
        with contextlib.suppress(Exception):
            termios.tcflush(fd, termios.TCIFLUSH)
        tty = io.TextIOWrapper(
            io.FileIO(fd, "r+"), encoding="utf-8", errors="replace", write_through=True
        )
    except OSError:
        if not fall_back_to_stdin:
            return None
        try:
            if until is None:
                return input(text if plain is None else plain)
            sys.stdout.write(text if plain is None else plain)
            sys.stdout.flush()
            return read_line_until(sys.stdin, sys.stdin.fileno(), until)
        except (EOFError, KeyboardInterrupt, OSError, ValueError):
            return None
    try:
        with tty:
            tty.write(text)
            line = read_line_until(tty, fd, until)
            if line is None and until is not None:
                # Whatever was typed was aimed at a prompt that is over.
                with contextlib.suppress(Exception):
                    termios.tcflush(fd, termios.TCIFLUSH)
                tty.write("\n")
            return line
    except OSError:
        # The terminal vanished mid-prompt; the text already printed, so do
        # not prompt again on stdin.
        return None


def format_session_facts(facts: SessionFacts) -> str:
    """The one-line status the pause banner and Ctrl-Z print: the few things a
    CLI operator cannot otherwise see (a TUI/web viewer has widgets for them).
    Spend first -- it is the fact that decides whether to interrupt now."""
    return (
        f"{format_usd(facts.spend_usd, partial=facts.spend_partial)}"
        f" · {facts.model} · commands {facts.run_commands} · {facts.isolation}"
    )


def _status_suffix(session_facts: Callable[[], SessionFacts] | None) -> str:
    """The indented status line under the pause banner, or nothing when the
    lifecycle passed no facts (a detached leg has no terminal anyway)."""
    if session_facts is None:
        return ""
    return f"          {format_session_facts(session_facts())}\n"


_JOB_CONTROL_HINT = (
    "[agent6] job control is unavailable here; Ctrl-C then /detach keeps it running.\n"
)


def _install_status_signal(
    state: dict[str, Any], session_facts: Callable[[], SessionFacts] | None
) -> Any:
    """Ctrl-Z: print the run's state, and stand an armed pause back down, so
    checking on a run never costs it a step. Replaces SIGTSTP's default on
    purpose -- a suspended agent freezes its live provider stream, which the
    server then kills mid-response: a real suspend would not pause the run,
    it would corrupt it. The printed hint names the alternative."""
    if not hasattr(signal, "SIGTSTP"):
        return None

    def _handler(_signum: int, _frame: Any) -> None:
        line = _status_suffix(session_facts).strip() or "no live facts for this leg"
        if state["stage"] == 1:
            state["stage"] = 0
            tty_message(
                f"\n[agent6] {line}\n[agent6] pause cancelled; the run continues.\n"
                + _JOB_CONTROL_HINT
            )
        else:
            tty_message(f"\n[agent6] {line}\n" + _JOB_CONTROL_HINT)

    return signal.signal(signal.SIGTSTP, _handler)


def install_steer_sigint(  # noqa: PLR0915 - a closure factory over one shared stage dict
    events: EventSink,
    session_dir: Path,
    console_view: ConsoleView | None = None,
    session_facts: Callable[[], SessionFacts] | None = None,
    btw_runner: BtwRunner | None = None,
    config_path: Path | None = None,
) -> SteerState:
    """Install a SIGINT handler with escalating stages.

    * 1st Ctrl-C: pause at the next safe boundary (between steps; the
      in-flight model call finishes first). Emits `session.steer_requested`;
      the prompt is a TUI modal when the TUI is live, otherwise the
      interactive pause menu; with redirected std streams the menu cannot
      own the line, so a plain prompt goes to the controlling terminal
      (`/dev/tty`) instead.
    * 2nd Ctrl-C: interrupt the in-flight model call and prompt now.
    * 3rd Ctrl-C (or Ctrl-C at the pause prompt itself): KeyboardInterrupt,
      stopping the run (resumable with `agent6 resume`).
    * Ctrl-Z prints the same one-line status WITHOUT arming anything, and
      cancels an armed pause -- so checking on a run costs it nothing. It also
      replaces SIGTSTP's default: a suspended agent freezes its live provider
      stream, which the server then kills mid-response, so a real suspend
      would corrupt the run rather than pause it. The hint it prints names
      /detach as the way to step away.

    `console_view`, when given, has its heartbeat spinner suspended for the
    prompt's duration: the spinner's per-tick line-erase otherwise wipes the
    pause-menu line and its Tab preview.

    Returns callables for the workflow plus a `restore` hook to put the
    previous handler back when the run is done.
    """
    state: dict[str, Any] = {"stage": 0, "prompting": False}

    def _handler(_signum: int, _frame: Any) -> None:
        # A boundary can be a whole model response away (a reasoning model may
        # think for 30-60s), hence the escalation; at the pause prompt itself a
        # Ctrl-C stops the run, as the pause banner promised.
        if state["prompting"] or state["stage"] >= 2:
            raise KeyboardInterrupt
        if state["stage"] == 1:
            state["stage"] = 2
            if not frontend_is_live(session_dir):
                tty_message("\n[agent6] interrupting this step. Ctrl-C again to stop the run.\n")
            return
        state["stage"] = 1
        # Drop a STALE answer file (one without a request marker) so it is not
        # instantly consumed as this new prompt's answer. An answer with a
        # pending request is a live front-end steer the loop has not consumed
        # yet; deleting it would silently discard the operator's instruction.
        if not steer_request_pending(session_dir):
            clear_steer_answer(session_dir)
        events.emit("session.steer_requested", source="sigint")
        # With the TUI up, the steer prompt is a modal, don't scribble on the
        # terminal it owns. Otherwise tell the user a prompt is coming.
        if not frontend_is_live(session_dir):
            tty_message(
                "\n[agent6] pausing after this step: Enter continues, type to steer,"
                " /stop ends it, /detach backgrounds it. Ctrl-C again to interrupt now.\n"
                + _status_suffix(session_facts)
            )

    previous = signal.signal(signal.SIGINT, _handler)
    previous_tstp = _install_status_signal(state, session_facts)

    def requested() -> bool:
        # Either a Ctrl-C (any stage) OR a front-end steer request marker.
        return state["stage"] >= 1 or steer_request_pending(session_dir)

    def interrupt() -> bool:
        # A double Ctrl-C aborts the in-flight call; so does a front-end steer
        # carrying the `now` urgency (`steer --now`). A plain steer waits for
        # the boundary: aborting wastes the streamed tokens and the step's
        # partial work.
        return state["stage"] >= 2 or steer_interrupt_pending(session_dir)

    def clear() -> None:
        state["stage"] = 0
        clear_steer_answer(session_dir)
        clear_steer_request(session_dir)

    def prompt() -> str | None:
        # An answer already on disk (a `resume --steer` seed, the end-of-session
        # follow-up, a front-end's answer that landed first) IS the steer: the
        # terminal menu is for an unanswered request only. Without this the
        # follow-up typed at "next:" opened the pause menu asking for it again.
        seeded = take_steer_answer(session_dir)
        if seeded is not None:
            return seeded
        # TUI live: the user answers a modal; read its file-bridge result.
        if frontend_is_live(session_dir):
            answer = read_steer_answer(session_dir)
            # A dismissed/abandoned modal yields None (read_steer_answer timed out
            # or the TUI died). Clear the request marker on THIS no-answer path so a
            # persisting `steer.request` cannot re-trigger another 600s blocking
            # read at the very next boundary, looping the run. A genuinely-answered
            # steer leaves clearing to the caller's clear() (with the answer already
            # consumed). The SIGINT stage is also cleared so a stale Ctrl-C request
            # doesn't immediately re-arm the same dead prompt.
            if answer is None:
                state["stage"] = 0
                clear_steer_request(session_dir)
            return answer
        return _menu()

    def _menu() -> str | None:
        pause = console_view.pause if console_view is not None else contextlib.nullcontext
        state["prompting"] = True
        try:
            with pause():
                if menu_capable():
                    # The interactive pause menu: line editing, history, and a
                    # fish-style Tab preview of the slash commands.
                    return pause_menu(session_dir, btw_runner=btw_runner, config_path=config_path)
                typed = tty_prompt(
                    "[agent6] paused: [enter] continue · type to steer"
                    " · q stop · exit leave · d detach: ",
                    until=lambda: steer_answer_written(session_dir),
                )
                if typed is None and steer_answer_written(session_dir):
                    # A front-end's steer landed while the prompt waited.
                    tty_message("[agent6] a steer arrived from a front-end; taking it\n")
                    return take_steer_answer(session_dir)
                return normalize_steer_choice(typed)
        finally:
            state["prompting"] = False

    def armed() -> bool:
        return state["stage"] >= 1

    def prompt_now() -> None:
        """The pause menu right after an operator prompt's answer (an operator
        prompt counts as a boundary): the action seeds the steer answer the
        next between-steps boundary consumes without re-prompting; an empty
        action (continue) disarms instead."""
        action = _menu()
        if action is None or not action.strip():
            state["stage"] = 0
            return
        write_steer_answer(session_dir, action)

    def restore() -> None:
        with contextlib.suppress(Exception):
            signal.signal(signal.SIGINT, previous)
        if previous_tstp is not None:
            with contextlib.suppress(Exception):
                signal.signal(signal.SIGTSTP, previous_tstp)

    def reset_stage() -> None:
        state["stage"] = 0

    return SteerState(
        requested=requested,
        clear=clear,
        prompt=prompt,
        restore=restore,
        abort_pending=lambda: steer_answer_is_abort(session_dir),
        interrupt=interrupt,
        reset_stage=reset_stage,
        armed=armed,
        prompt_now=prompt_now,
    )


def make_steer_state(
    events: EventSink,
    session_dir: Path,
    console_view: ConsoleView | None = None,
    session_facts: Callable[[], SessionFacts] | None = None,
    btw_runner: BtwRunner | None = None,
    config_path: Path | None = None,
) -> SteerState:
    """Install the steer SIGINT handler when a controlling terminal exists
    (covers run/plan/ask with or without the TUI); else steer purely over the
    front-end file bridge (detached runs)."""
    try:
        with open("/dev/tty", encoding="utf-8"):  # noqa: PTH123
            pass
    except OSError:
        return file_bridge_steer(session_dir)
    return install_steer_sigint(
        events, session_dir, console_view, session_facts, btw_runner, config_path
    )
