# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The steer seam front-ends hand the run/resume lifecycle, and its file-bridge
implementation for surfaces with no controlling terminal (a detached spawn
driven from the TUI hub or web, an ACP connection). Satisfies
`app.frontend.SteerHooks` structurally; the CLI layers its SIGINT pause-menu
machinery on the same shape."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from agent6.sessions.ipc import (
    clear_steer_answer,
    clear_steer_request,
    read_steer_answer,
    steer_answer_is_abort,
    steer_interrupt_pending,
    steer_request_pending,
)


@dataclass
class SteerState:
    requested: Callable[[], bool]
    clear: Callable[[], None]
    prompt: Callable[[], str | None]
    restore: Callable[[], None]
    # Polled during a streaming call so a Stop interrupts a long turn promptly.
    abort_pending: Callable[[], bool]
    # Polled during a streaming call: True aborts the in-flight model call so
    # the steer prompt runs now instead of at the next between-step boundary.
    interrupt: Callable[[], bool]
    # Called at each leg entry (wf.run/resume): a stage armed in a finished leg
    # must not open a phantom pause (or abort the first call) in the next one.
    # Zeroes only the SIGINT stage; the steer marker files stay, because
    # resume --steer seeds the next leg through them.
    reset_stage: Callable[[], None]
    # A Ctrl-C pause is armed (an operator prompt counts as a boundary: the
    # approval prompt consults this to open the menu right after its answer).
    armed: Callable[[], bool] = field(default=lambda: False)
    # Run the pause menu NOW and seed its action as the steer answer the next
    # boundary consumes without re-prompting; an empty action (continue)
    # disarms instead. No-op off the terminal (the file bridge has no menu).
    prompt_now: Callable[[], None] = field(default=lambda: None)


def file_bridge_steer(session_dir: Path) -> SteerState:
    """Steer for a run with no controlling terminal (detached spawn from the
    TUI hub or the web UI, an ACP connection): no SIGINT handler, requests and
    answers travel only over the front-end file bridge. Without this, a
    hub-spawned run would never poll the `steer.request` marker, every web/TUI
    steer would be silently lost, and a `resume --steer` seed (an ACP
    session's later prompt) would never reach the resumed model."""

    def prompt() -> str | None:
        answer = read_steer_answer(session_dir)
        # No answer (front-end died or abandoned the prompt): clear the
        # request marker so it cannot re-trigger another blocking read at the
        # very next boundary, looping the run.
        if answer is None:
            clear_steer_request(session_dir)
        return answer

    def clear() -> None:
        clear_steer_answer(session_dir)
        clear_steer_request(session_dir)

    return SteerState(
        requested=lambda: steer_request_pending(session_dir),
        clear=clear,
        prompt=prompt,
        restore=lambda: None,
        abort_pending=lambda: steer_answer_is_abort(session_dir),
        interrupt=lambda: steer_interrupt_pending(session_dir),
        reset_stage=lambda: None,  # no SIGINT stage on the file bridge
    )
