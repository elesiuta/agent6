# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""One ACP prompt becomes one agent6 run.

The protocol OWNS stdout, so the run's reporter writes to stderr. One status
line on stdout desynchronises the stream irrecoverably, and no editor recovers
from it.

The run id is minted HERE, before the run starts, so `session/cancel` has
something to address. Letting the lifecycle mint its own left the session with
no handle: the cancel reported success while the run continued to completion,
spending budget and making commits. It is minted ONCE per ACP session: later
prompts resume that same run with the new text seeded as its first steering
instruction, so the session is one conversation, not a row of strangers.

`run_task` reads the process cwd, so a run in a session's directory has to
chdir there -- which is process-global. Runs are therefore serialised on the
connection: a second prompt waits rather than running in the wrong repository.
"""

from __future__ import annotations

import os
import sys
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, BinaryIO

from agent6.app._setup import load_session_config
from agent6.app.finalize import EXIT_NO_COMMIT_LANDED, EXIT_VERIFY_FAILED
from agent6.app.frontend import FrontendCapabilities, SessionFrontend
from agent6.app.reporter import Reporter
from agent6.app.resume import resume_task
from agent6.app.run import run_task
from agent6.config.layer import resolved_state_dir
from agent6.sessions.id import unused_session_id
from agent6.types import session_bucket
from agent6.ui.acp.frontend import acp_frontend
from agent6.ui.acp.server import ACPServer
from agent6.ui.acp.session import ACP_MODE, Session, Sessions, StopReason
from agent6.ui.acp.updates import (
    ending,
    message_update,
    printable,
    tool_call_id,
    updates_for,
    wire_call_id,
)
from agent6.ui.spawn import agent6_exe, spawn_detached_resume
from agent6.viewmodel.tail import journal_size, tail_events
from agent6.viewmodel.transcript import TranscriptFold

# How long a permission request waits for the editor. An operator who has
# walked away must not hold a run forever, and the seam already reads silence
# as the cautious answer: an approval becomes a denial, a question becomes no
# answer at all.
PERMISSION_TIMEOUT_S = 300.0
# A safety net on joining the streaming tail, not the normal path: `_stop`
# ends it one read pass after the run returns. This bounds a tail wedged on a
# filesystem that is not answering.
DRAIN_S = 5.0
# How often a queued turn checks for its own cancel while another session's
# turn holds the run lock.
QUEUE_POLL_S = 0.1


def _stderr(message: str) -> None:
    print(message, file=sys.stderr)


@dataclass
class ProseOrder:
    """The lifecycle's lines, queued for the journal tail to emit in order.

    The lifecycle speaks from the run thread while the tail projects the
    journal from its own, a poll behind; sent as they are said, an ending
    line landed before the turn's last tool calls. Each line is stamped with
    the journal's size when said, and the tail emits it once it has read
    past that point (everything, once the tail is done).
    """

    server: ACPServer
    acp_session_id: str
    logs_path: Path
    _lock: threading.Lock = field(default_factory=threading.Lock)
    _pending: list[tuple[int, str]] = field(default_factory=list)

    def say(self, text: str) -> None:
        with self._lock:
            self._pending.append((journal_size(self.logs_path), text))

    def flush(self, consumed: int | None) -> None:
        """Emit every line stamped at or before *consumed* bytes of journal; all of
        them for None."""
        with self._lock:
            due = [t for stamp, t in self._pending if consumed is None or stamp <= consumed]
            self._pending = [
                (s, t) for s, t in self._pending if consumed is not None and s > consumed
            ]
        for text in due:
            # `note`/`warn` already carry the marker `message_update` adds.
            self.server.notify_raw(
                message_update(self.acp_session_id, text.removeprefix("[agent6] "))
            )


def forwarding_reporter(
    server: ACPServer, acp_session_id: str, said: list[str], *, order: ProseOrder | None = None
) -> Reporter:
    """The lifecycle's reporter: stderr (the editor's agent log), and the same
    line to the editor as agent6's own prose, in journal order through *order*
    when a tail is projecting the journal (else as it is said).

    What the lifecycle says is stated nowhere else: no journal event carries a
    refusal's reason (a missing git identity, another writer holding the
    repo, a dirty worktree), the auto-stash notice, or the where-are-my-changes
    footer. The cost receipt goes to stderr only: the fold's done item already
    carries the cost. *said* collects the forwarded lines, so a turn that
    ended with nothing said can be told apart from one that explained itself.
    """

    def _say(message: str) -> None:
        _stderr(message)
        text = message.strip()
        if not text:
            return
        said.append(text)
        if order is not None:
            order.say(text)
        else:
            server.notify_raw(message_update(acp_session_id, text.removeprefix("[agent6] ")))

    return Reporter(out=_say, err=_say, receipt=_stderr)


def option_kind(text: str, standing: bool | None) -> str:
    """ACP's button kinds, from WHO asked -- never from the option text.

    `standing=True` is an approval an editor may REMEMBER. `False` is the
    fetch tool's off-list host, where remembering would silently cover a
    different host. `None` is a `UserQuestion`, whose options the MODEL
    wrote: keying on the text let a model emit an option literally named
    "allow" and have it advertised as `allow_always`, so an editor keying its
    memory on the title would auto-approve later real permission requests.
    """
    if standing is None:
        return "allow_once"
    if text == "deny":
        return "reject_once"
    return "allow_always" if standing else "allow_once"


def stop_reason(code: int) -> StopReason:
    """ACP's vocabulary, from the lifecycle's exit code.

    A deliberate finish is `end_turn` even when the verify gate stayed red
    (exit 4) or the edits stranded uncommitted (exit 5): the agent answered,
    and that state is already on the wire as messages. `refusal` is for a run
    that could not complete -- it broke, was refused, or hit its budget; ACP
    has no finer failure word, and the DETAIL again arrives as messages.
    """
    if code == 130:
        return "cancelled"
    return "end_turn" if code in (0, EXIT_VERIFY_FAILED, EXIT_NO_COMMIT_LANDED) else "refusal"


def _selected(answer: dict[str, Any], options: tuple[str, ...]) -> str | None:
    """The option the editor chose, or None for no usable answer.

    A cancel, a timeout and an id we did not issue are all "no answer" -- and
    the answer has to be one we offered, or an unknown string could become an
    "allow" by prefix.
    """
    outcome = answer.get("outcome")
    if not isinstance(outcome, dict) or outcome.get("outcome") != "selected":
        return None
    chosen = outcome.get("optionId")
    if not isinstance(chosen, str) or not chosen.isdigit():
        return None
    index = int(chosen)
    return options[index] if index < len(options) else None


class Announced:
    """One turn's register: its number, and the tool calls the editor has
    been told about.

    Written by the tail as it announces; a permission request for a call
    waits here until the call it names has been announced, so the editor
    never hears of a call first through its approval. The wait is bounded by
    liveness, never a clock: the tail closes the register when it stops
    reading, and a cancelled turn abandons the wait.
    """

    def __init__(self, turn: int) -> None:
        self.turn = turn
        self._ids: set[str] = set()
        self._closed = False
        self._changed = threading.Condition()

    def __contains__(self, tool_call_id: str) -> bool:
        with self._changed:
            return tool_call_id in self._ids

    def add(self, tool_call_id: str) -> None:
        with self._changed:
            self._ids.add(tool_call_id)
            self._changed.notify_all()

    def close(self) -> None:
        with self._changed:
            self._closed = True
            self._changed.notify_all()

    def wait_for(self, tool_call_id: str, *, abandoned: Callable[[], bool]) -> None:
        with self._changed:
            while tool_call_id not in self._ids and not self._closed and not abandoned():
                self._changed.wait(0.5)  # *abandoned* is polled; add/close wake at once


@dataclass
class RunBridge:
    """Runs prompts for one ACP connection."""

    server: ACPServer
    # The top-level `--config FILE` overlay; every session load threads it.
    config_path: Path | None = None
    # One at a time: the chdir in `_run` is process-global, and a run in the
    # wrong directory commits to the wrong repository.
    _runs: threading.Lock = field(default_factory=threading.Lock)
    # The session whose turn holds `_runs`, named to a turn queued behind it.
    _running: Session | None = None
    _asks: threading.Lock = field(default_factory=threading.Lock)
    _asked: int = 0

    def sessions(self) -> Sessions:
        return Sessions(run=self.run, state_dir_for=resolved_state_dir)

    def ask(
        self,
        session: Session,
        announced: Announced,
        prompt: str,
        options: tuple[str, ...],
        standing: bool | None,
        call_id: int | None,
        until: Callable[[], bool] | None = None,
    ) -> str | None:
        """Put one approval or question to the editor.

        ACP v1 has no method for a free-form question, so a `UserQuestion` goes
        out as a permission request too -- its options ARE the answers. The
        editor renders buttons either way, which is what the seam needs.

        A question with no options has no buttons, so there is nothing for the
        operator to press: asking would stall the whole permission timeout and
        then answer "said nothing" regardless. Not asking is the same answer
        immediately, without holding the run for five minutes.

        `toolCall` is required on a permission request, and is the only text
        the editor has to render: it carries the prompt as the call's title,
        which a ToolCallUpdate exists to update. The announced title is
        `salient_arg` clipped to 60 chars, which would show the operator an
        argv whose first line looks benign and whose rest they never see. A
        prompt gating a tool call (*call_id*, the dispatcher's
        stamp) names THAT call, once the tail has announced it; its lifecycle
        carries on from there (pending, then its outcome). A prompt gating no
        call announces an entity of its own, and closes it: an entity ACP
        models as having a lifecycle needs its end, or an editor keeps one
        pending tool call per approval for the life of the session.
        """
        if not options:
            return None
        if call_id is not None:
            gated = wire_call_id(session.session_id, announced.turn, str(call_id))
            announced.wait_for(gated, abandoned=lambda: session.cancelled)
            tool_call: dict[str, Any] = {
                "toolCallId": gated,
                "title": printable(prompt),
                "status": "pending",
            }
        else:
            with self._asks:
                self._asked += 1
                gated = f"ask-{session.acp_id}-{self._asked}"
            tool_call = {
                "toolCallId": gated,
                "title": printable(prompt),
                "kind": "other",
                "status": "pending",
            }
        answer = self.server.request(
            "session/request_permission",
            {
                "sessionId": session.acp_id,
                "toolCall": tool_call,
                "options": [
                    {
                        # An INDEX, not the option text: the text can be
                        # model-written (a UserQuestion's options are), and an
                        # identifier is not a place for model input. It also
                        # makes "only an option we offered" structural rather
                        # than a string comparison.
                        "optionId": str(index),
                        "name": printable(text),
                        "kind": option_kind(text, standing),
                    }
                    for index, text in enumerate(options)
                ],
            },
            timeout_s=PERMISSION_TIMEOUT_S,
            until=until,
        )
        chosen = _selected(answer, options)
        if call_id is None:
            self.server.notify_raw(
                {
                    "jsonrpc": "2.0",
                    "method": "session/update",
                    "params": {
                        "sessionId": session.acp_id,
                        "update": {
                            "sessionUpdate": "tool_call_update",
                            "toolCallId": gated,
                            "status": "completed" if chosen else "failed",
                        },
                    },
                }
            )
        return chosen

    def _frontend(self, session: Session, announced: Announced) -> SessionFrontend:
        return acp_frontend(
            ask=lambda prompt, options, standing, call_id, until=None: self.ask(
                session, announced, prompt, options, standing, call_id, until
            ),
            # `initialize` has not landed if this is None, and nothing is
            # known about the client; the cautious answer is that it can do
            # nothing.
            capabilities=self.server.client_capabilities or FrontendCapabilities(),
            agent6_exe=agent6_exe,
            spawn_detached_resume=lambda cwd, sid, flags: spawn_detached_resume(
                cwd, sid, config_path=self.config_path, flags=flags
            ),
        )

    def _resumable(self, session: Session) -> bool:
        """A later prompt continues the session's run: the prior turn left a
        resume snapshot. A first prompt, or one whose run died before its
        first save, starts fresh under a new id."""
        if not session.session_id:
            return False
        layout = session.layout(resolved_state_dir(session.cwd))
        return (layout.session_dir / "loop_state.json").is_file()

    def had_journal(self, session: Session) -> bool:
        """Whether this turn got far enough to write a journal of its own."""
        if not session.session_id:
            return False
        return session.layout(resolved_state_dir(session.cwd)).logs_path.exists()

    def run(self, session: Session, text: str) -> StopReason:
        # BEFORE the queue, not after. `_runs` is held for a whole run, so a
        # second session's turn can wait here for many minutes, and deciding
        # the id inside would leave that whole window with no run to address:
        # a cancel writes no marker, the turn runs to completion spending
        # budget and making commits, and the editor is told "cancelled".
        # Through the owner: a fresh id reaches run_task as an EXPLICIT one,
        # which skips the lifecycle's own minting -- a collision would refuse
        # the turn with "use agent6 resume <id>" over an id the editor never
        # chose.
        resuming = self._resumable(session)
        if not resuming:
            session.session_id = unused_session_id(
                resolved_state_dir(session.cwd), session_bucket(ACP_MODE)
            )
        if not self._runs.acquire(blocking=False):
            # Queued behind another session's turn: say so, and keep listening
            # for a cancel while waiting (a turn that has not started is
            # stopped by not starting it).
            holder = self._running
            who = f"session {holder.acp_id}" if holder is not None else "another session"
            self.server.notify_raw(
                message_update(
                    session.acp_id,
                    f"waiting: {who} is running a turn; this prompt starts when it ends",
                )
            )
            while not self._runs.acquire(timeout=QUEUE_POLL_S):
                if session.cancelled:
                    return "cancelled"
        try:
            self._running = session
            if session.cancelled:
                # Cancelled while queued. The marker is for a run in flight;
                # one that has not started is stopped by not starting it.
                return "cancelled"
            try:
                return self._run(session, text, resuming=resuming)
            except Exception as exc:
                # A run that dies before it has a journal has no other way to
                # say so, and the turn still ends with a stop reason. A broken
                # config is the ordinary case: the CLI prints it, and here the
                # editor would have seen a turn end with no words at all.
                # Once a journal exists the fold has already reported the
                # ending, and saying this too contradicts it.
                if not self.had_journal(session):
                    self.server.notify_raw(
                        message_update(session.acp_id, f"the run could not start: {exc}")
                    )
                return "refusal"
        finally:
            self._running = None
            self._runs.release()

    def _run(self, session: Session, text: str, *, resuming: bool) -> StopReason:
        layout = session.layout(resolved_state_dir(session.cwd))
        os.chdir(session.cwd)
        session.turn += 1

        said: list[str] = []
        announced = Announced(turn=session.turn)
        ended, drained = threading.Event(), threading.Event()

        def _stop() -> bool:
            """Stop the tail one read pass after the run returns.

            `tail_events` checks this at the TOP of each poll, so answering
            False once lets the journal's last lines still reach the editor.
            Stopping immediately dropped them; waiting for the run's own
            `session.end` taxed every turn that ends without one (a config error,
            an early refusal) with the full drain timeout.
            """
            if not ended.is_set():
                return False
            if drained.is_set():
                return True
            drained.set()
            return False

        order = ProseOrder(self.server, session.acp_id, layout.logs_path)
        tail = threading.Thread(
            target=self._stream,
            args=(session, layout.logs_path, _stop, resuming, announced, order),
            name=f"acp-tail-{session.acp_id}",
            daemon=True,
        )
        tail.start()
        reporter = forwarding_reporter(self.server, session.acp_id, said, order=order)
        try:
            if resuming:
                # The prompt rides in as the resumed run's first steering
                # instruction; resume accepts a finished run exactly when it
                # carries one. resume_task loads config itself, from the same
                # explicit path.
                code = resume_task(
                    self.config_path,
                    session.session_id,
                    frontend=self._frontend(session, announced),
                    force=False,
                    steer=text,
                    reporter=reporter,
                )
            else:
                effective = load_session_config(session.cwd, self.config_path, mode=ACP_MODE)
                code = run_task(
                    effective.config,
                    text,
                    frontend=self._frontend(session, announced),
                    session_id=session.session_id,
                    explicit_leaves=effective.explicit_leaves,
                    reporter=reporter,
                )
        finally:
            ended.set()
            tail.join(timeout=DRAIN_S)
            order.flush(None)  # a tail that outlived the drain still owes these
        if code != 0 and not said and not self.had_journal(session):
            # A stop before the run had anything to say for itself.
            self.server.notify_raw(message_update(session.acp_id, f"the run stopped (exit {code})"))
        return stop_reason(code)

    def _stream(
        self,
        session: Session,
        logs_path: Path,
        stop: Callable[[], bool],
        resuming: bool,
        announced: Announced,
        order: ProseOrder | None = None,
    ) -> None:
        """Project the run's journal into `session/update` as it is written,
        the lifecycle's own lines taking their place between events.

        A resumed run appends to the journal its prior legs already fill, and
        the editor rendered those turns as they happened -- start at the end,
        or the whole conversation replays as if new. The ending also goes to
        stderr, the editor's agent log: the editor is the live view, so the
        lifecycle prints no ending of its own."""
        fold = TranscriptFold()
        consumed = [0]

        def _at(position: int) -> None:
            consumed[0] = position

        try:
            for event in tail_events(
                logs_path,
                stop_when_finished=True,
                should_stop=stop,
                start_at_end=resuming,
                on_position=_at,
            ):
                for item in fold.feed(event):
                    wire_id = (
                        tool_call_id(item, session.session_id, announced.turn)
                        if item.kind == "tool"
                        else ""
                    )
                    for body in updates_for(
                        item,
                        acp_session_id=session.acp_id,
                        wire_id=wire_id,
                        announced=wire_id in announced,
                    ):
                        self.server.notify_raw(body)
                    if item.kind == "tool":
                        announced.add(wire_id)
                    elif item.kind == "done":
                        _stderr(ending(item))
                if order is not None:
                    order.flush(consumed[0])
        finally:
            announced.close()
            if order is not None:
                order.flush(None)


def serve_acp(
    stdin: BinaryIO | None = None,
    stdout: BinaryIO | None = None,
    *,
    config_path: Path | None = None,
) -> int:
    """Speak ACP on this process's stdio until the editor closes it."""
    server = ACPServer(
        stdin=stdin if stdin is not None else sys.stdin.buffer,
        stdout=stdout if stdout is not None else sys.stdout.buffer,
    )
    server.sessions = RunBridge(server=server, config_path=config_path).sessions()
    server.serve()
    return 0
