# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""JSON-RPC 2.0 over stdio, and the `initialize` handshake.

Framing is line-delimited JSON with a bounded read, the same shape
`ui/mcp_server.py` uses and for the same reason: an unbounded `readline`
buffers a whole line before any size check, so a runaway client could exhaust
memory before the cap could refuse it. The dispatch is NOT shared -- different
protocol, different methods.
"""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, BinaryIO

from agent6 import __version__
from agent6.app.frontend import FrontendCapabilities
from agent6.ui.acp.rpc import (
    INTERNAL_ERROR,
    INVALID_REQUEST,
    METHOD_NOT_FOUND,
    RpcError,
)
from agent6.ui.acp.session import Sessions, prompt_text
from agent6.ui.acp.updates import message_update

# The ACP version this front-end speaks. Negotiation is bilateral: the client
# sends the newest it supports, we answer with this, and the client disconnects
# if it cannot live with the answer.
PROTOCOL_VERSION = 1
# 4 MiB, mirroring the MCP server's cap. A prompt with a large pasted context
# is the legitimate big case; past this the payload is dropped, not buffered.
MAX_LINE_BYTES = 1 << 22
# How long EOF waits for a turn to reach its next boundary. Long enough for a
# verify to finish, short enough that a wedged run does not hold the editor's
# exit forever.
EOF_GRACE_S = 30.0


# A handler returns this instead of a result when the reply comes later, from
# a worker thread. `session/prompt` is the case: answering it inline would
# block the read loop for the whole run, and a blocked loop cannot receive the
# `session/cancel` that ACP requires to work during one.
DEFERRED = object()


@dataclass
class _Pending:
    """One outstanding request to the client."""

    arrived: threading.Event = field(default_factory=threading.Event)
    answer: dict[str, Any] | None = None


def capabilities_from(_client: dict[str, Any]) -> FrontendCapabilities:
    """What the CLIENT said it can do, as the seam every front-end declares.

    `session/request_permission` is required of every ACP client, so a
    connected one can always be asked."""
    return FrontendCapabilities(can_ask=True)


@dataclass
class ACPServer:
    """One ACP connection. Owns the framing; the methods live beside it."""

    stdin: BinaryIO
    stdout: BinaryIO
    client_capabilities: FrontendCapabilities | None = None
    # How a prompt becomes a run. None in a transport-only test.
    sessions: Sessions | None = None
    _handlers: dict[str, Any] = field(default_factory=dict)
    # One writer at a time: the read loop answers requests while worker threads
    # stream session/update, and two interleaved writes are a line no editor
    # can parse.
    _write_lock: threading.Lock = field(default_factory=threading.Lock)
    # Requests WE sent the client, awaiting its answer.
    _pending: dict[object, _Pending] = field(default_factory=dict)
    _pending_lock: threading.Lock = field(default_factory=threading.Lock)
    _next_id: int = 0
    # The client's end of the pipe is closed; further writes have no reader.
    _gone: bool = False

    def __post_init__(self) -> None:
        self._handlers = {
            "initialize": self._initialize,
            "session/new": self._session_new,
            "session/prompt": self._session_prompt,
            "session/cancel": self._session_cancel,
        }

    def serve(self) -> None:
        """Read messages until EOF. Requests are answered; notifications are
        acted on and not answered, per JSON-RPC."""
        while True:
            line = self.stdin.readline(MAX_LINE_BYTES + 1)
            if not line:
                # EOF: the editor closed. Let a live turn stop at a boundary
                # rather than being torn down mid-git holding the locks.
                self.abandon_pending()
                if self.sessions is not None:
                    self.sessions.wait_for_turns(timeout_s=EOF_GRACE_S)
                return
            if len(line) > MAX_LINE_BYTES:
                # Drain the rest of the oversized line in bounded chunks and
                # drop the whole payload: refusing beats buffering it.
                while line and not line.endswith(b"\n"):
                    line = self.stdin.readline(MAX_LINE_BYTES + 1)
                continue
            if not line.strip():
                continue
            self._handle(line)

    def _handle(self, line: bytes) -> None:
        parsed = self._envelope(line)
        if parsed is None:
            return
        req_id, method, params = parsed
        handler = self._handlers.get(method)
        if handler is None:
            if req_id is not None:  # a notification we do not know is ignorable
                self.reply(req_id, error=(METHOD_NOT_FOUND, f"unknown method: {method!r}"))
            return
        try:
            result = handler(params, req_id)
        except RpcError as exc:
            if req_id is not None:
                self.reply(req_id, error=(exc.code, exc.message))
            return
        except Exception as exc:  # a handler bug must not kill the connection
            if req_id is not None:
                self.reply(req_id, error=(INTERNAL_ERROR, f"{type(exc).__name__}"))
            return
        if result is DEFERRED:
            return  # a worker owns this reply now
        if req_id is not None:
            self.reply(req_id, result=result)

    def _envelope(self, line: bytes) -> tuple[object, str, dict[str, Any]] | None:
        """`(id, method, params)`, or None when there is nothing to act on.

        A malformed line has no id to answer against, which is the one case
        with no reply at all -- and dropping it beats ending the session an
        editor is mid-conversation on.
        """
        try:
            message = json.loads(line)
        except json.JSONDecodeError:
            return None
        if not isinstance(message, dict):
            return None
        req_id = message.get("id")
        method = message.get("method")
        raw = message.get("params")
        if not isinstance(method, str):
            # A message with no method and an id we allocated is the CLIENT
            # answering something we asked -- the reply path for
            # session/request_permission. Without this it came back as "no
            # method", and every approval blocked its worker forever.
            if req_id is not None and self._deliver(req_id, message):
                return None
            if req_id is not None:
                self.reply(req_id, error=(INVALID_REQUEST, "no method"))
            return None
        return req_id, method, raw if isinstance(raw, dict) else {}

    def abandon_pending(self) -> None:
        """Answer every outstanding request with nothing, because nobody will.

        The read loop is the only thing that delivers a client's answer, so
        once it is gone a worker waiting on an approval waits the full
        permission timeout -- far longer than the EOF grace, so the process
        always exited and killed the run it was trying to let finish. The
        seam already reads an empty answer as the cautious deny, so the run
        reaches its next boundary and the stop marker takes effect.
        """
        with self._pending_lock:
            waiting = list(self._pending.values())
            self._pending.clear()
        for slot in waiting:
            slot.arrived.set()

    def _deliver(self, req_id: object, message: dict[str, Any]) -> bool:
        """Hand a client response to whoever is waiting for it. True if it was
        ours."""
        if "result" not in message and "error" not in message:
            # A JSON-RPC response carries one or the other. Without this, any
            # malformed frame that happened to carry an outstanding id became
            # that approval's answer -- and an unreadable answer denies.
            return False
        with self._pending_lock:
            slot = self._pending.pop(req_id, None)
        if slot is None:
            return False
        slot.answer = message
        slot.arrived.set()
        return True

    def request(
        self,
        method: str,
        params: dict[str, Any],
        *,
        timeout_s: float,
        until: Callable[[], bool] | None = None,
    ) -> dict[str, Any]:
        """Ask the CLIENT something and wait for its answer.

        Called from a worker thread, never from the read loop -- the loop is
        what delivers the answer, so waiting on it there would deadlock. A
        timeout answers with nothing rather than wedging the turn: an editor
        that never replies must not cost the session. So does *until*
        holding (polled every 0.2 s): the question was answered by another
        route, and the editor's reply, if one comes, answers nothing.
        """
        with self._pending_lock:
            self._next_id += 1
            req_id = f"agent6-{self._next_id}"
            slot = _Pending()
            self._pending[req_id] = slot
        self.notify_raw({"jsonrpc": "2.0", "id": req_id, "method": method, "params": params})
        deadline = time.monotonic() + timeout_s
        while not slot.arrived.wait(timeout=min(0.2, max(0.0, deadline - time.monotonic()))):
            if time.monotonic() >= deadline or (until is not None and until()):
                with self._pending_lock:
                    self._pending.pop(req_id, None)
                return {}
        answer = slot.answer or {}
        result = answer.get("result")
        return result if isinstance(result, dict) else {}

    def _session_new(self, params: dict[str, Any], _req_id: object) -> dict[str, Any]:
        return self._sessions().new(params)

    def _session_prompt(self, params: dict[str, Any], req_id: object) -> object:
        if req_id is None:
            # A turn's whole point is the stopReason it answers with. Sent as a
            # notification there is nobody to answer, and replying with a null
            # id is not valid JSON-RPC.
            raise RpcError(INVALID_REQUEST, "session/prompt is a request, not a notification")
        sessions = self._sessions()
        session = sessions.get(params)
        text = prompt_text(params)
        sessions.start_turn(
            session,
            text,
            finish=lambda reason: self.reply(req_id, result={"stopReason": reason}),
        )
        return DEFERRED

    def _session_cancel(self, params: dict[str, Any], _req_id: object) -> dict[str, Any]:
        # A notification in ACP: no reply, and it must land while the turn it
        # cancels is still running -- which is why the turn is not on this
        # thread. A cancel for a session this server does not have would otherwise vanish
        # with zero bytes written, so the stop button does nothing and says
        # nothing; tell the editor instead.
        sessions = self._sessions()
        try:
            sessions.cancel(sessions.get(params))
        except RpcError as exc:
            self.notify_raw(message_update(str(params.get("sessionId")), f"cancel: {exc.message}"))
        return {}

    def _sessions(self) -> Sessions:
        if self.sessions is None:
            raise RpcError(INTERNAL_ERROR, "this connection has no session runner wired")
        return self.sessions

    def _initialize(self, params: dict[str, Any], _req_id: object) -> dict[str, Any]:
        raw = params.get("clientCapabilities")
        self.client_capabilities = capabilities_from(raw if isinstance(raw, dict) else {})
        return {
            "protocolVersion": PROTOCOL_VERSION,
            "agentCapabilities": {
                # `session/load` is what v2 reorganises, and resume is where
                # agent6 has the most of its own semantics. Absent, not half.
                "loadSession": False,
                # Not advertised: `prompt_text` keeps only text blocks, and a
                # resource block's uri is client-controlled, so passing one
                # through would be path injection. Claiming support and then
                # dropping the attachment silently is worse than saying no.
                "promptCapabilities": {"embeddedContext": False},
            },
            "agentInfo": {"name": "agent6", "version": __version__},
            "authMethods": [],
        }

    def reply(
        self,
        req_id: object,
        *,
        result: dict[str, Any] | None = None,
        error: tuple[int, str] | None = None,
    ) -> None:
        body: dict[str, Any] = {"jsonrpc": "2.0", "id": req_id}
        if error is not None:
            body["error"] = {"code": error[0], "message": error[1]}
        else:
            body["result"] = result if result is not None else {}
        self.notify_raw(body)

    def notify_raw(self, body: dict[str, Any]) -> None:
        """Write one message. Encoded lossily on purpose: a lone surrogate in
        model-emitted text would otherwise raise mid-write and desynchronise
        the stream, which is worse than a replacement character."""
        line = json.dumps(body, ensure_ascii=False, default=str) + "\n"
        gone = False
        with self._write_lock:
            if self._gone:
                return
            try:
                self.stdout.write(line.encode("utf-8", "replace"))
                self.stdout.flush()
            except BrokenPipeError:
                # The editor closed the connection. There is nobody left to
                # tell, and a live run's tail would otherwise raise once per
                # event; the run itself keeps going to its next boundary.
                self._gone = True
                gone = True
        if gone:
            # Nothing this server asked can be answered now either.
            self.abandon_pending()
