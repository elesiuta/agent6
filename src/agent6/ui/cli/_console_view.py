# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Render a run's event stream to a terminal as a live conversation.

The CLI skin over `viewmodel.TranscriptFold`: assistant reasoning and text stream
inline as they arrive, every tool call shows with its result, and nothing prints
a blank block. One `ConsoleView` serves both `agent6 run` (in-process, subscribed
to the EventSink) and `agent6 attach` (out-of-process, fed by the log tailer), so
the two render identically.

Reasoning/text deltas are streamed by this class (the live-typing feel); the
structural steps (tool call+result, commit, verdict) come from `TranscriptFold`,
which is shared with the TUI and web skins.
"""

from __future__ import annotations

import contextlib
import sys
import time
from collections.abc import Callable, Generator
from threading import Event, RLock, Thread
from typing import Any, TextIO

from agent6.ui.cli._task_tree import tree_lines_from_event_nodes
from agent6.viewmodel.events import event_epoch
from agent6.viewmodel.format import spinner_frame
from agent6.viewmodel.listing import task_snippet
from agent6.viewmodel.transcript import (
    DONE,
    THINK,
    TranscriptFold,
    TranscriptItem,
    scrub_terminal_controls,
)
from agent6.viewmodel.transcript_style import StyleName, item_lines

_ANSI = {
    "dim": "\033[2m",
    "reset": "\033[0m",
    "bold": "\033[1m",
    "cyan": "\033[36m",
    "blue": "\033[34m",
    "green": "\033[32m",
    "red": "\033[31m",
    "yellow": "\033[33m",
    "magenta": "\033[35m",
    "italic": "\033[3m",
}

# Semantic style name -> ANSI escape. The TUI has the sibling Rich map; both skins
# render item_lines() (viewmodel.transcript_style), so the structure and which
# element is coloured live in ONE place and can't drift.
_STYLE_ANSI: dict[StyleName, str] = {
    "thinking": _ANSI["dim"],
    "think-marker": _ANSI["blue"],
    "text": "",
    "call": _ANSI["bold"] + _ANSI["cyan"],
    "verify": _ANSI["bold"] + _ANSI["yellow"],
    "arg": _ANSI["dim"],
    "ok": _ANSI["green"],
    "fail": _ANSI["red"],
    "detail": _ANSI["dim"],
    "more": _ANSI["dim"] + _ANSI["italic"],
    "tail": _ANSI["dim"],
    "commit": _ANSI["magenta"],
    "marker": _ANSI["dim"] + _ANSI["italic"],
    "done-ok": _ANSI["bold"] + _ANSI["green"],
    "done-fail": _ANSI["bold"] + _ANSI["yellow"],
    "done-neutral": _ANSI["bold"],
    "body": "",
    "done-detail": _ANSI["dim"],
    "operator": _ANSI["bold"] + _ANSI["green"],
}

_FLUSH_EVERY_S = 0.03  # coalesce streaming-delta flushes; see ConsoleView._raw
_HEARTBEAT_TICK_S = 0.5  # how often the spinner refreshes
_STALL_AFTER_S = 1.5  # show the heartbeat once output has been silent this long
# A mid-block gap gets much longer before the spinner interrupts: drawing it
# closes the open prose block and the next delta reopens a new bullet, visibly
# splitting a streamed word (e.g. a file path) in two. Slow token cadence
# routinely pauses a few seconds; only a real stall is worth that cost.
_MID_BLOCK_STALL_S = 10.0


class ConsoleView:
    """Fold events to styled terminal lines. `feed`/`__call__` take one event;
    thread-safe so it can subscribe to an EventSink that several roles emit to."""

    def __init__(
        self,
        out: TextIO | None = None,
        *,
        color: bool | None = None,
        policy: Callable[[], str] | None = None,
    ) -> None:
        # The run's policy line (viewmodel.session_policy), printed under the
        # task so an operator sees the model, the command setting, the sandbox
        # and the gate without interrupting. Read WHEN the task prints, not when
        # the view is built: the gate is inferred and pinned between the two,
        # and an early read said "no verify gate" over a run that had one. None
        # when the caller has no run dir.
        self._policy = policy
        # Finished /btw answers waiting for a clean break. A btw completes while
        # the run is streaming; printing it then would cut the transcript in
        # half, so it waits for a turn boundary and lands whole.
        self._btw: list[str] = []
        self._out = out if out is not None else sys.stderr
        self._color = self._out.isatty() if color is None else color
        self._fold = TranscriptFold()
        # Reentrant: the SIGINT steer handler emits an event (re-entering feed on
        # the same thread) while a delta write may hold the lock.
        self._lock = RLock()
        self._phase: str | None = None  # None | "thinking" | "text": the open prose block
        self._last_flush = 0.0
        self._plan_count = 0  # tasks shown in the last plan block; reprint when it grows
        # Live heartbeat: a turn can stream text then wedge mid-token (a stalled
        # SSE stream) or pause between turns with nothing on screen. A background
        # thread shows a spinner + "working… Ns" during silence so the run never
        # looks hung; only on a real terminal (no spinner in a pipe or a test).
        self._last_output_at = time.monotonic()
        # The epoch ts of the event currently being fed (None between feeds or
        # for a ts-less event): _bump_idle anchors the idle timer to it, so
        # `agent6 attach` replaying history measures from when the run last
        # spoke -- an arrival anchor made a run wedged 40 minutes read
        # "working… 3s".
        self._event_ep: float | None = None
        self._active = False  # run is between session.start and session.end (a turn or a tool)
        self._status_active = False  # a transient spinner line is on screen now
        self._paused = False  # True while an interactive /dev/tty prompt owns the line
        self._spin = 0
        self._stop = Event()
        self._heartbeat: Thread | None = None
        if self._out.isatty():
            self._heartbeat = Thread(target=self._heartbeat_loop, daemon=True)
            self._heartbeat.start()

    def __call__(self, event: dict[str, Any]) -> None:
        self.feed(event)

    def queue_btw(self, block: str) -> None:
        """Hand a finished /btw answer to the view. Called from the watcher
        thread; printed whole at the next turn boundary, never mid-stream."""
        with self._lock:
            self._btw.append(block)

    def _drain_btw(self) -> None:
        """Print any finished btw answers. Caller holds the lock and has just
        closed the open block, so this lands between turns."""
        for block in self._btw:
            self._line(block)
        self._btw.clear()

    def _bump_idle(self) -> None:
        """Reset the idle timer to the fed event's own age (its ts), or to now
        for a ts-less event."""
        age = 0.0 if self._event_ep is None else max(0.0, time.time() - self._event_ep)
        self._last_output_at = time.monotonic() - age

    def feed(self, event: dict[str, Any]) -> None:  # noqa: PLR0911 - one per event kind
        etype = event.get("type", "")
        with self._lock:
            # Anchor per EVENT, not only per rendered line: a replay can end on
            # an event that renders nothing yet (a tool.call whose result never
            # came -- the wedged case), and events between renders are activity.
            self._event_ep = event_epoch(event.get("ts"))
            self._bump_idle()
            # The heartbeat spins whenever the run is active and output has gone
            # silent -- covering BOTH a thinking provider call AND a long tool /
            # verify command running in the jail (which happens between role.result
            # and the next role.call, so a role-only flag would miss it and the
            # CLI would look frozen through a whole test suite).
            if etype in ("session.start", "role.call", "tool.call"):
                self._active = True
            elif etype in ("session.end", "session.steer_requested"):
                self._active = False
                # A btw that lands after the last turn would otherwise sit in
                # the queue forever: the run ending IS a clean break.
                self._end_block()
                self._drain_btw()
            if etype in ("role.thinking_delta", "role.text_delta"):
                self._stream(str(event.get("text", "")), thinking=etype == "role.thinking_delta")
                return
            if etype in ("role.call", "role.result"):
                self._end_block()  # a provider call boundary closes any open prose
                self._drain_btw()
                return
            if etype == "session.steer_requested":
                # A Ctrl-C pause message is about to print to the same terminal;
                # close any open (dim) block so it doesn't bleed into the message.
                self._end_block()
                return
            if etype == "session.start":
                # The first user-authored line, clipped: a `--from` task
                # carries the whole plan and flattened it into one endless line.
                task = task_snippet(str(event.get("user_task", "")), max_chars=200)
                self._line(self._c("bold", self._c("cyan", DONE) + " " + task) + "\n")
                policy = self._policy() if self._policy is not None else ""
                if policy:
                    self._line(self._c("dim", f"  {policy}") + "\n")
                # The fold still reads the start (mode, first timestamp) for
                # the receipt; its operator item is this headline, not printed twice.
                self._fold.feed(event)
                return
            if etype == "btw.answered":
                # Queued, not printed: it lands whole at the next turn boundary
                # so it can never break up a streaming turn.
                self._btw.append(str(event.get("block", "")))
                return
            if etype == "graph.update":
                self._render_plan(event)
                return
            if etype == "loop.provider.retry":
                # A retry resets the idle clock, so without a line the "working…
                # Ns" counter restarts with nothing said and a run wedged behind
                # provider failures reads as freshly started.
                self._end_block()
                attempt = event.get("attempt")
                self._line(
                    self._c("dim", f"  retrying after a provider error (attempt {attempt}): ")
                    + self._c("dim", str(event.get("error", "")))
                    + "\n"
                )
                return
            for item in self._fold.feed(event):
                self._end_block()
                self._render(item)

    # -- inline prose streaming --------------------------------------------
    def _stream(self, piece: str, *, thinking: bool) -> None:
        # The piece is MODEL text headed for a real terminal: scrub controls
        # (OSC 52 writes the clipboard; the fold's previews are scrubbed, but
        # this live path printed the delta raw). A sequence split across deltas
        # cannot reassemble: any piece containing its opener loses the tail
        # from the ESC on, and the continuation prints as inert text.
        piece = scrub_terminal_controls(piece)
        want = "thinking" if thinking else "text"
        if self._phase != want:
            if not piece.strip():
                return  # never open a block on whitespace: kills empty response blocks
            self._end_block()
            self._phase = want
            self._raw("  " + (self._dim() + THINK + " " if thinking else ""))
            piece = piece.lstrip()
        self._bump_idle()  # a delta is real progress
        # keep wrapped lines under the block's indent; dim (thinking) spans them all
        self._raw(piece.replace("\n", "\n    " if thinking else "\n  "))

    def _end_block(self) -> None:
        if self._phase == "thinking":
            self._raw(self._reset())
        if self._phase is not None:
            self._raw("\n")
            self._flush()  # show the completed prose block now
        self._phase = None

    def _render_plan(self, event: dict[str, Any]) -> None:
        """Print the decomposed task tree when it first appears and each time it
        grows (new subtasks as the model explores), so a headless run's plan is
        visible in the stream, not only in the TUI pane. A single root (no
        decomposition) is not a plan worth a block."""
        nodes = event.get("nodes", {}) or {}
        if not isinstance(nodes, dict) or len(nodes) <= 1 or len(nodes) <= self._plan_count:
            return
        self._plan_count = len(nodes)
        cursor = event.get("cursor")
        lines = tree_lines_from_event_nodes(nodes, cursor if isinstance(cursor, str) else None)
        if not lines:
            return
        self._end_block()
        self._line("\n" + self._c("bold", f"plan ({len(nodes)} tasks)") + "\n")
        for line in lines:
            self._line(self._c("dim", "  " + line) + "\n")

    # -- structural items ---------------------------------------------------
    def _render(self, item: TranscriptItem) -> None:
        """The CLI skin over the shared item_lines(): map each span's semantic style
        to ANSI, behind a two-space left gutter (a blank spec line stays blank).
        An in-flight tool call prints nothing (on a tty the heartbeat shows the
        wait; a pipe sees nothing until the result); the settled item prints
        the call whole."""
        if item.kind == "tool" and item.ok is None:
            return
        for line in item_lines(item, detail="collapsed"):
            rendered = "".join(
                f"{_STYLE_ANSI[style]}{text}{_ANSI['reset']}"
                if self._color and _STYLE_ANSI[style]
                else text
                for text, style in line
            )
            self._line(("  " + rendered if rendered else "") + "\n")

    # -- output helpers -----------------------------------------------------
    def _c(self, name: str, text: str) -> str:
        return f"{_ANSI[name]}{text}{_ANSI['reset']}" if self._color else text

    def _dim(self) -> str:
        return _ANSI["dim"] if self._color else ""

    def _reset(self) -> str:
        return _ANSI["reset"] if self._color else ""

    def _clear_status(self) -> None:
        """Erase the transient spinner line so real output prints cleanly. Caller
        holds the lock (or is the constructor before the thread starts)."""
        if self._status_active:
            self._out.write("\r\x1b[2K")  # carriage return + erase whole line
            self._status_active = False

    def _raw(self, text: str) -> None:
        # Low-level writer, used by streaming deltas AND internal block-closing;
        # it clears the spinner but does NOT bump _last_output_at (that tracks
        # real model output -- set by _stream / _line -- so closing a block from
        # the heartbeat can't reset the idle timer and suppress the spinner).
        # Streaming path: flush at most every _FLUSH_EVERY_S. A per-token flush on
        # a slow terminal (SSH, a busy emulator) backpressures the SSE read in the
        # same thread and can stall the stream; ~30ms is imperceptible and cuts
        # thousands of flushes to a few dozen a second.
        self._clear_status()
        self._out.write(text)
        now = time.monotonic()
        if now - self._last_flush >= _FLUSH_EVERY_S:
            self._out.flush()
            self._last_flush = now

    def _line(self, text: str) -> None:
        # Structural lines (tool call/result, commit, verdict) are discrete model
        # progress: show them at once and reset the idle timer.
        self._clear_status()
        self._bump_idle()
        self._out.write(text)
        self._flush()

    def _heartbeat_loop(self) -> None:
        """Refresh a transient "⠋ working… Ns" line while a turn is in flight and
        output has gone silent (a stalled stream, or a between-turn pause), so the
        run never looks hung. Runs only on a real terminal."""
        while not self._stop.wait(_HEARTBEAT_TICK_S):
            with self._lock:
                if self._paused:
                    continue  # an interactive prompt owns the terminal: draw nothing
                idle = time.monotonic() - self._last_output_at
                stall_after = _MID_BLOCK_STALL_S if self._phase is not None else _STALL_AFTER_S
                if not self._active or idle < stall_after:
                    self._clear_status()  # output flowing or turn done: no spinner
                    if self._status_active is False:
                        self._out.flush()
                    continue
                # Silent mid-turn: close any open prose block so the cursor sits on
                # a clean line, then draw/refresh the spinner in place.
                if self._phase is not None:
                    self._end_block()
                self._spin += 1
                glyph = spinner_frame(self._spin)
                hint = "  (Ctrl-C to steer or stop)" if idle >= 20 else ""
                body = f"{glyph} working… {int(idle)}s{hint}"
                self._out.write("\r\x1b[2K" + (self._c("dim", body) if self._color else body))
                self._out.flush()
                self._status_active = True

    def notice(self, msg: str) -> None:
        """Print a workflow notice (auto-commit, review, tool_error) on the SAME
        stream as the stream/spinner, clearing the spinner first under the lock so
        the notice can't collide with a spinner write on a shared terminal."""
        with self._lock:
            self._clear_status()
            self._out.write(msg if msg.endswith("\n") else msg + "\n")
            self._out.flush()

    @contextlib.contextmanager
    def pause(self) -> Generator[None]:
        """Suspend the heartbeat spinner and clear its line so an interactive
        /dev/tty prompt (ask_user, a run_command approval) can own the terminal,
        then restore. Without this the spinner's per-tick line-erase wipes the
        question and the operator's keystrokes. The lock is released across the
        yield so the blocking prompt cannot stall feed()/notice()."""
        with self._lock:
            self._paused = True
            self._clear_status()
            self._out.flush()
        try:
            yield
        finally:
            with self._lock:
                self._paused = False

    def close(self) -> None:
        """Stop the heartbeat thread and clear any spinner line. Safe to call more
        than once; the daemon thread also dies with the process."""
        self._stop.set()
        if self._heartbeat is not None:
            self._heartbeat.join(timeout=1.0)
        with self._lock:
            self._clear_status()
            self._out.flush()

    def _flush(self) -> None:
        self._out.flush()
        self._last_flush = time.monotonic()
