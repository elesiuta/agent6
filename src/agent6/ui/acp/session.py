# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""`session/new`, `session/prompt` and `session/cancel`.

A prompt runs on a worker thread, not on the read loop. Answering it inline
would block reading for the whole run, and a blocked loop cannot receive the
`session/cancel` that ACP requires to work DURING one -- so the cancel an
editor sends would arrive only after the thing it meant to stop had finished.
"""

from __future__ import annotations

import sys
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agent6.app.preflight import git_repo_refusal
from agent6.sessions.id import friendly_token
from agent6.sessions.ipc import request_stop
from agent6.sessions.layout import SessionLayout
from agent6.types import session_bucket
from agent6.ui.acp.rpc import INVALID_PARAMS, RpcError

# What ACP is told a turn ended as. `cancelled` is the operator's own act, so
# it is reported as itself rather than as a failure.
StopReason = str
# Every ACP turn is an `agent6 run`; the editor has no way to ask for another
# mode. Named so the session dir's bucket is stated rather than defaulted.
ACP_MODE = "run"


@dataclass(slots=True)
class Session:
    """One ACP session: a working directory, and at most one live turn."""

    # ACP's own id for this conversation, minted by `session/new` and what
    # every notification addresses. Distinct from `session_id`, agent6's.
    acp_id: str
    cwd: Path
    # agent6's run id: empty until the first prompt mints it; later prompts
    # resume that same run, so the ACP session stays one conversation.
    session_id: str = ""
    # The turn in progress, 1-based: one leg of the run, and the component
    # that keeps a tool call's wire id unique across turns (`wire_call_id`).
    turn: int = 0
    thread: threading.Thread | None = None
    cancelled: bool = False
    # Cleared BEFORE the turn answers. `thread.is_alive()` is still true while
    # `finish` runs -- and `finish` IS the reply -- so a conforming editor that
    # writes its next prompt the instant it reads the answer was refused at
    # random.
    turn_live: bool = False

    def is_running(self) -> bool:
        return self.turn_live

    def layout(self, state_dir: Path) -> SessionLayout:
        """This turn's agent6 session dir. Named once: three call sites built it
        with the DEFAULT bucket, which is only right while ACP runs one mode."""
        return SessionLayout(
            state_dir=state_dir, session_id=self.session_id, subdir=session_bucket(ACP_MODE)
        )


@dataclass
class Sessions:
    """The connection's sessions, and how a prompt becomes a run."""

    # (session, prompt text) -> the stop reason. Injected so the transport can
    # be tested without a provider, and so the lifecycle stays in `app`.
    run: Callable[[Session, str], StopReason]
    state_dir_for: Callable[[Path], Path]
    _by_id: dict[str, Session] = field(default_factory=dict)

    def new(self, params: dict[str, Any]) -> dict[str, Any]:
        raw_cwd = params.get("cwd")
        if not isinstance(raw_cwd, str) or not Path(raw_cwd).is_absolute():
            # The spec makes every path absolute; a relative one would resolve
            # against whatever directory the editor happened to launch us in.
            raise RpcError(INVALID_PARAMS, "cwd must be an absolute path")
        cwd = Path(raw_cwd)
        # The same wall `agent6 run` puts in front of a workspace. This
        # directory becomes what the jail mounts WRITABLE, and here it arrives
        # over the wire: without this a client could point a run at any
        # absolute path, and `$HOME` on a machine with dotfiles under git would
        # hand the model the whole home directory.
        refusal = git_repo_refusal(cwd)
        if refusal is not None:
            raise RpcError(INVALID_PARAMS, refusal)
        servers = params.get("mcpServers")
        if isinstance(servers, list) and servers:
            # agent6's MCP servers are operator config, never editor-supplied
            # (the tool surface is fixed; see docs/security.md). Accepting the
            # session and silently not starting them would read as connected.
            raise RpcError(
                INVALID_PARAMS,
                "agent6 does not take MCP servers from the editor: configure them in "
                "agent6's own config ([mcp.servers], `agent6 mcp connect`) and remove "
                "them from this agent's entry.",
            )
        session = Session(acp_id=friendly_token(), cwd=cwd)
        self._by_id[session.acp_id] = session
        return {"sessionId": session.acp_id}

    def get(self, params: dict[str, Any]) -> Session:
        acp_session_id = params.get("sessionId")
        session = self._by_id.get(acp_session_id) if isinstance(acp_session_id, str) else None
        if session is None:
            raise RpcError(INVALID_PARAMS, f"no session {acp_session_id!r}")
        return session

    def start_turn(
        self, session: Session, text: str, *, finish: Callable[[StopReason], None]
    ) -> None:
        """Run the prompt on a worker, and answer when it ends."""
        if session.is_running():
            raise RpcError(INVALID_PARAMS, "that session already has a turn in flight")
        session.cancelled = False

        def _work() -> None:
            try:
                reason = self.run(session, text)
            except Exception:  # a run that dies must still end the turn
                reason = "refusal"
            answer = "cancelled" if session.cancelled else reason
            session.turn_live = False
            finish(answer)

        session.turn_live = True
        session.thread = threading.Thread(target=_work, name=f"acp-{session.acp_id}", daemon=True)
        try:
            session.thread.start()
        except RuntimeError:
            # Set before starting on purpose (the worker clears it), so a
            # thread that never ran left the session refusing every later
            # prompt as busy -- and EOF joining an unstarted thread raised out
            # of the read loop.
            session.turn_live = False
            session.thread = None
            raise

    def wait_for_turns(self, *, timeout_s: float) -> None:
        """Let live turns finish before the process goes.

        Closing the editor is the ordinary way EOF arrives, and a daemon worker
        torn down mid-git holds the repo and worker single-writer locks and the
        run-dir pid. Cancel first so it stops at its next boundary rather than
        running to completion nobody is watching.
        """
        live = [s for s in self._by_id.values() if s.is_running()]
        for session in live:
            self.cancel(session)
        # ONE deadline across every join: per-thread timeouts made N sessions
        # wait N times the documented bound.
        deadline = time.monotonic() + timeout_s
        for session in live:
            if session.thread is not None:
                session.thread.join(timeout=max(0.0, deadline - time.monotonic()))

    def cancel(self, session: Session) -> None:
        """Ask the run to stop at its next boundary.

        A marker, not a kill: the finished step's tool results and auto-commit
        land first, so a cancelled turn leaves the workspace in a state the
        operator can read rather than halfway through one.
        """
        session.cancelled = True
        if session.session_id and not request_stop(
            session.layout(self.state_dir_for(session.cwd)).session_dir
        ):
            # A notification has no reply: stderr is the one channel left.
            print(
                f"[agent6] could not write the stop request for {session.session_id}",
                file=sys.stderr,
            )


def prompt_text(params: dict[str, Any]) -> str:
    """The prompt's text and resource_link blocks, joined.

    ACP sends content blocks; agent6's task is prose. A `resource_link` (the
    baseline attach-a-file shape) is rendered as its uri VERBATIM: the model
    reads it through the ordinary tools, so the workspace boundary still
    decides what the path reaches. Other non-text blocks (an image, an
    embedded resource) are dropped rather than rendered as a placeholder the
    model would try to read; initialize does not advertise them.
    """
    blocks = params.get("prompt")
    if not isinstance(blocks, list):
        raise RpcError(INVALID_PARAMS, "prompt must be a list of content blocks")
    parts: list[str] = []
    for b in blocks:
        if not isinstance(b, dict):
            continue
        if b.get("type") == "text" and b.get("text"):
            parts.append(str(b["text"]))
        elif b.get("type") == "resource_link" and b.get("uri"):
            parts.append(f"Attached: {b['uri']}")
    text = "\n\n".join(parts).strip()
    if not text:
        raise RpcError(INVALID_PARAMS, "the prompt carried no text")
    return text
