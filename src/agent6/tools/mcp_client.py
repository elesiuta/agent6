# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Minimal stdio MCP (Model Context Protocol) client.

agent6 spawns each configured MCP server as a long-lived subprocess and
talks JSON-RPC 2.0 over stdin/stdout. Only the subset the loop needs is
implemented:

* `initialize` (handshake).
* `notifications/initialized` (we send it; we ignore incoming
  notifications).
* `tools/list` (discover tools at startup).
* `tools/call` (dispatch one tool call).

Anything else the server might send (`logging/*`, `prompts/*`,
`resources/*`, server-side `ping`) is silently dropped on the
client side, we do not advertise the corresponding capabilities.

Threat model
============

Each MCP server is spawned as a jailed child by default (its own
`[mcp.servers.<name>.sandbox]` policy; `unconfined = true` opts out) under
a curated env that NEVER carries the provider API keys; `pass_env` adds
named vars, and config refuses a `pass_env` naming a provider key. The
argv comes exclusively from your config (`[mcp.servers.<name>] command =
[...]`); the LLM cannot influence it.

What the LLM *can* influence is the *arguments* to `tools/call` once
a server is connected. The MCP server is responsible for validating
those, agent6 forwards them verbatim. Operators should treat each MCP
server as a tool surface as serious as any agent6 built-in tool.

A misbehaving server (crash, hang, malformed JSON, oversized reply)
must not take the agent down. Each `call_tool` is wrapped in a
timeout and a try/except; the manager surfaces a clean `MCPError` to
the dispatcher, which converts it to a `tool.result ok=false` event.
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import signal
import subprocess
import threading
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import IO, Any

from agent6 import __version__
from agent6.child_env import curated_env
from agent6.sandbox.jail import (
    JailedProcess,
    JailUnavailableError,
    SessionNetwork,
    die_with_parent,
    spawn_in_jail,
)
from agent6.tools.mcp_http import HttpTransport, MCPHttpError, MCPSessionExpired
from agent6.types import JailPolicy

# MCP protocol version we speak. The spec is versioned by date string;
# we negotiate this in `initialize` and accept whatever the server says
# back (we don't validate compatibility beyond "we got a result").
_MCP_PROTOCOL_VERSION = "2024-11-05"

# Anything longer than this on a single line is treated as a protocol
# error and the line is dropped. 8 MiB is generous for a tools/list
# response on a server with a few dozen tools.
_MAX_LINE_BYTES = 8 * 1024 * 1024

# Under the transport cap, a compromised (or buggy) operator-run server can
# still emit multi-MiB tool descriptions and results. Unbounded, a description
# rides in EVERY provider request's tools array and a result floods the
# context: the run breaks every turn instead of degrading. Bound both at this
# trust boundary; the marker says what was cut. The result cap matches
# fetch.MAX_BYTES, the largest single payload any built-in tool returns.
_MAX_INLINE_TEXT_CHARS = 2048  # tool descriptions + error detail
_MAX_RESULT_CHARS = 1 << 20


def _bounded_inline_text(text: str) -> str:
    if len(text) <= _MAX_INLINE_TEXT_CHARS:
        return text
    return text[:_MAX_INLINE_TEXT_CHARS] + " …[agent6: truncated]"


def _bounded_result(result: dict[str, Any]) -> dict[str, Any]:
    """Degrade an oversized tools/call result instead of flooding the run:
    keep the text content up to the cap, drop everything else, and say so."""
    blob = json.dumps(result, ensure_ascii=False, default=str)
    if len(blob) <= _MAX_RESULT_CHARS:
        return result
    content = result.get("content")
    texts = [
        c["text"]
        for c in (content if isinstance(content, list) else ())
        if isinstance(c, dict) and isinstance(c.get("text"), str)
    ]
    note = (
        f"[agent6: this result was {len(blob)} chars serialized; text content kept up"
        f" to {_MAX_RESULT_CHARS} chars, everything else dropped]"
    )
    kept = "\n".join(texts)[:_MAX_RESULT_CHARS]
    return {"content": [{"type": "text", "text": f"{note}\n{kept}".rstrip()}]}


# Prefix every MCP tool name with this + the server name so collisions
# with built-in tools (and across servers) are structurally impossible.
# Sonnet / GPT-4o / Kimi all accept `[A-Za-z0-9_]+` tool names of
# 64-128 chars; double-underscore segmentation keeps the prefix human-
# parseable in transcripts.
MCP_TOOL_PREFIX = "mcp__"


def split_tool_name(qualified_name: str) -> tuple[str, str]:
    """`mcp__<server>__<tool>` -> (server, tool).

    Splits on the FIRST double-underscore after the prefix, so a tool name that
    contains "__" itself survives intact (server names cannot: see
    `mcp_server_name_refusal`). One parser, because the dispatcher needs the
    server to know whose approval rule applies and the manager needs it to route.
    """
    if not qualified_name.startswith(MCP_TOOL_PREFIX):
        raise MCPError(f"not an MCP tool name: {qualified_name!r}")
    suffix = qualified_name[len(MCP_TOOL_PREFIX) :]
    try:
        server_name, tool_name = suffix.split("__", 1)
    except ValueError as exc:
        raise MCPError(f"malformed MCP tool name: {qualified_name!r}") from exc
    return server_name, tool_name


# A server-advertised tool name is spliced into the LLM-visible
# `mcp__<server>__<tool>`; the provider tool-name grammar is
# `[A-Za-z0-9_-]{1,64}`. A name with whitespace/dots/other chars would make
# the qualified name an invalid tool definition (rejected by the API) or shadow
# a built-in, so tools whose names don't match are skipped at registration.
# Matched with fullmatch, not `$`: `$` also matches just before a terminal
# newline, so `foo\n` would pass and splice a newline into the tool name.
_VALID_MCP_TOOL_NAME = re.compile(r"[A-Za-z0-9_-]+")

# The 64-char cross-provider bound from that grammar, applied to the WHOLE
# qualified name (prefix + operator server name + separators + tool name):
# one over-limit entry would invalidate the entire tools array.
_MAX_QUALIFIED_TOOL_NAME_LEN = 64


class MCPError(RuntimeError):
    """Anything the MCP client refuses to do or could not complete."""


class MCPTimeout(MCPError):
    """A request the server did not answer within its timeout."""


class MCPRestarted(MCPError):
    """A request cut short because another caller's timeout replaced the server."""


@dataclass(frozen=True, slots=True)
class MCPToolDescriptor:
    """One tool advertised by one MCP server. `qualified_name` is what
    the LLM sees and what the dispatcher routes on."""

    server_name: str
    tool_name: str
    description: str
    input_schema: dict[str, Any]

    @property
    def qualified_name(self) -> str:
        return f"{MCP_TOOL_PREFIX}{self.server_name}__{self.tool_name}"


@dataclass(frozen=True, slots=True)
class MCPStartFailure:
    """A configured server that is not there. Recorded rather than only logged:
    a log line goes wherever the front-end's stderr goes, which under an editor
    is a pane nobody is watching."""

    name: str
    error: str


# What we keep of a server's stderr. Enough for a traceback or a launcher
# setup failure, bounded because the writer is third-party code: capturing it
# to a file let a hostile server write 1.8 GB in three seconds.
_STDERR_KEEP_BYTES = 8192


def _spawn_server(
    command: tuple[str, ...],
    policy: JailPolicy | None,
    pass_env: tuple[str, ...],
    session_net: SessionNetwork | None = None,
) -> JailedProcess:
    """Start one stdio server: through the jail when it has a policy, as a
    plain subprocess when the operator opted it out.

    The confined path is `spawn_in_jail`, the same launcher and the same
    JailPolicy a jailed command gets: a second confinement stack would drift
    (no seccomp, no private /proc, no hidden-path masking).

    Stderr is a PIPE the caller drains, not /dev/null: everything that can go
    wrong before the handshake -- a command that does not exist, a grant the
    kernel refused, the launcher's own setup -- says so there and nowhere
    else; discarded, every one of those reads as the same "died before
    responding to initialize". Drained rather than collected, and
    capped: an undrained pipe blocks the writer at 64 KB, and a file grows
    until the disk is gone.
    """
    if policy is not None:
        return spawn_in_jail(
            policy,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            session_net=session_net,
        )
    return JailedProcess(
        subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
            # A curated env, not this process's: the full one carries the
            # provider API keys, and an MCP server is third-party code that may
            # log or forward what it was given. A server that needs a token
            # names it in `pass_env`.
            env=curated_env(passthrough=pass_env, desktop=True),
            # Its own session: a terminal Ctrl-C signals the foreground process
            # group, and an MCP server that dies with it breaks its tools for
            # the rest of the run.
            start_new_session=True,
            # Dies with the agent (PDEATHSIG, same tie the jailed branch gets
            # through the launcher): a server that ignores stdin EOF must not
            # outlive a SIGKILLed agent6.
            preexec_fn=die_with_parent(os.getpid(), sig=signal.SIGKILL),  # noqa: PLW1509
        )
    )


def _drain_stderr(pipe: IO[bytes], keep: list[bytes]) -> None:
    """Read a server's stderr forever, keeping only the tail.

    Forever, because a pipe nobody reads stops the writer at 64 KB -- a server
    that logs would wedge itself. Only the tail, because the writer is
    third-party code with no reason to be polite about volume.
    """
    with contextlib.suppress(OSError, ValueError):
        while chunk := pipe.read(4096):
            keep.append(chunk)
            while len(keep) > 2:
                keep.pop(0)


def _stderr_tail(keep: list[bytes], limit: int = 400) -> str:
    """The last of what the server (or the launcher) said, for a failure
    message. Best-effort: a diagnostic must never raise over the failure it
    is describing."""
    text = b"".join(keep)[-_STDERR_KEEP_BYTES:].decode(errors="replace").strip()
    return text[-limit:].strip() if text else ""


def _result_of(response: dict[str, Any], *, name: str, method: str) -> Any:
    """The `result` of a JSON-RPC response, or raise its `error`."""
    if "error" in response:
        err = response["error"]
        msg = err.get("message", "(no message)") if isinstance(err, dict) else str(err)
        raise MCPError(f"server {name!r} {method} returned error: {msg}")
    return response.get("result")


@dataclass(frozen=True, slots=True)
class MCPServerSpec:
    """What starting one MCP server needs. The config's shape, at the boundary:
    a positional tuple grew a field per feature and every caller had to count."""

    name: str
    command: tuple[str, ...]
    startup_timeout_s: float
    call_timeout_s: float
    # Environment variables this server needs BY NAME. Everything else comes
    # from the curated base; naming each one is what keeps a provider key out.
    pass_env: tuple[str, ...] = ()
    # Set instead of `command` for a server the operator runs.
    http: HttpTransport | None = None
    # The sandbox this server runs under, or None for an unconfined one
    # (`[mcp.servers.<n>.sandbox].unconfined`). Built by the caller from the
    # same `jail_policy` a command uses.
    policy: JailPolicy | None = None


@dataclass
class _MCPServer:
    """One running MCP server. Owns its subprocess + an id counter +
    a stdout-reader thread that publishes responses into `_pending`
    keyed by request id."""

    name: str
    command: tuple[str, ...]
    startup_timeout_s: float
    call_timeout_s: float
    pass_env: tuple[str, ...] = ()
    # The sandbox this server runs under; None is the operator's explicit
    # `unconfined = true`.
    policy: JailPolicy | None = None
    # The run's session network, for a server whose policy joins it.
    session_net: SessionNetwork | None = None
    # Set instead of `command` for a server the OPERATOR runs: agent6 connects
    # rather than spawning, so it owns none of that server's environment,
    # lifetime or confinement.
    http: HttpTransport | None = None
    _proc: JailedProcess | None = None
    # The tail of this server's stderr, drained by a thread and read only to
    # explain a failure.
    _errors: list[bytes] = field(default_factory=list)
    _next_id: int = 1
    _id_lock: threading.Lock = field(default_factory=threading.Lock)
    # Serializes stdin writes: concurrent tools/call threads (explore-review
    # seats share one dispatcher) interleave pipe writes larger than PIPE_BUF,
    # corrupting the JSON-RPC framing for every in-flight request.
    _stdin_lock: threading.Lock = field(default_factory=threading.Lock)
    # One slot per in-flight request: _request registers `id -> None` before
    # writing, the reader fills ONLY registered slots, and the requester's
    # finally clears its slot -- so a reply landing after a timeout (or a
    # duplicate/unsolicited response shape) is dropped, and _pending is
    # bounded by the number of concurrently outstanding requests.
    _pending: dict[int, dict[str, Any] | None] = field(default_factory=dict)
    _pending_cv: threading.Condition = field(default_factory=threading.Condition)
    _reader: threading.Thread | None = None
    _reader_stop: threading.Event = field(default_factory=threading.Event)
    _tools: tuple[MCPToolDescriptor, ...] = ()
    # Bumped under `_restart_lock` by the caller whose timed-out call replaces
    # the process (`_restart`): a request in flight under another caller ends
    # as MCPRestarted the moment it changes, and a second timed-out caller
    # finds the restart already done. The lock also holds a new call back
    # until a restart's handshake is complete.
    _generation: int = 0
    _restart_lock: threading.Lock = field(default_factory=threading.Lock)
    # Releases the keeper thread a restart spawned the process from (see
    # `_restart`); set by `close`.
    _keeper_release: threading.Event | None = None

    def _redact_secrets(self, text: str) -> str:
        """Strip `pass_env` credential VALUES from a diagnostic string. A
        third-party server may echo a passed secret to its stderr, and that
        tail rides into `MCPError` and the durable `mcp.server_unavailable`
        event, so it is redacted before it leaves here."""
        for name in self.pass_env:
            value = os.environ.get(name, "")
            if value:
                text = text.replace(value, "<REDACTED>")
        return text

    def start(self) -> None:
        """Spawn the subprocess and pump it through `initialize` +
        `tools/list`. Raises `MCPError` if anything in the handshake
        fails, leaving the subprocess terminated."""
        if self._proc is not None:
            raise MCPError(f"server {self.name!r} already started")
        if self.http is not None:
            self._handshake()
            return
        try:
            self._proc = _spawn_server(
                self.command, self.policy, self.pass_env, session_net=self.session_net
            )
        except (OSError, FileNotFoundError, JailUnavailableError) as exc:
            raise MCPError(f"could not spawn MCP server {self.name!r}: {exc}") from exc
        # Start the reader before issuing the first request so the
        # initialize response can't race the reader thread.
        self._reader = threading.Thread(
            target=self._read_loop,
            name=f"mcp-reader[{self.name}]",
            daemon=True,
        )
        self._reader.start()
        if self._proc.stderr is not None:
            threading.Thread(
                target=_drain_stderr,
                args=(self._proc.stderr, self._errors),
                name=f"mcp-stderr[{self.name}]",
                daemon=True,
            ).start()
        self._handshake()

    def _handshake(self) -> None:
        """`initialize` + `tools/list`, the same either way the bytes move."""
        try:
            init_result = self._request(
                "initialize",
                {
                    "protocolVersion": _MCP_PROTOCOL_VERSION,
                    "capabilities": {},
                    "clientInfo": {"name": "agent6", "version": __version__},
                },
                timeout_s=self.startup_timeout_s,
            )
        except MCPError:
            self.close()
            raise
        if not isinstance(init_result, dict):
            self.close()
            raise MCPError(f"server {self.name!r} returned non-dict initialize result")
        self._notify("notifications/initialized", {})
        try:
            listed = self._request("tools/list", {}, timeout_s=self.startup_timeout_s)
        except MCPError:
            self.close()
            raise
        tools_raw = listed.get("tools") if isinstance(listed, dict) else None
        if not isinstance(tools_raw, list):
            self.close()
            raise MCPError(f"server {self.name!r} tools/list returned no tools array")
        descs: list[MCPToolDescriptor] = []
        seen: set[str] = set()
        for entry in tools_raw:
            if not isinstance(entry, dict):
                continue
            tname = entry.get("name")
            if not isinstance(tname, str) or not tname:
                continue
            if not _VALID_MCP_TOOL_NAME.fullmatch(tname) or tname in seen:
                # Skip tools whose names can't form a valid provider tool name
                # (mcp__<server>__<tool> must be [A-Za-z0-9_-]) and duplicates
                # of an already-registered name (first wins): either would
                # break the whole tools array at call time. Silently skip,
                # consistent with the non-string-name skip just above.
                continue
            desc = entry.get("description")
            schema = entry.get("inputSchema")
            if not isinstance(schema, dict):
                schema = {"type": "object"}
            qualified = MCPToolDescriptor(
                server_name=self.name,
                tool_name=tname,
                description=_bounded_inline_text(str(desc) if desc is not None else ""),
                input_schema=schema,
            )
            if len(qualified.qualified_name) > _MAX_QUALIFIED_TOOL_NAME_LEN:
                # Providers cap tool names at 64 chars; registering this one
                # would 400 every request carrying the tools array.
                continue
            seen.add(tname)
            descs.append(qualified)
        self._tools = tuple(descs)

    @property
    def tools(self) -> tuple[MCPToolDescriptor, ...]:
        return self._tools

    def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        # The name rides in from the LLM: a name outside the negotiated set
        # (filtered at registration, or never advertised by the server) is
        # refused before any request leaves agent6.
        if tool_name not in {d.tool_name for d in self._tools}:
            raise MCPError(f"server {self.name!r} did not advertise tool {tool_name!r}")
        # Once more when another caller's timeout replaced the server under
        # this call; a call the fresh server loses the same way gives up.
        for _ in range(2):
            with self._restart_lock:
                if self._proc is None and self.http is None:
                    raise MCPError(f"server {self.name!r} is not running")
                generation = self._generation
            try:
                return self._call(tool_name, arguments)
            except MCPTimeout as exc:
                raise MCPError(f"{exc}; {self._restart(generation)}") from exc
            except MCPRestarted:
                continue
        raise MCPError(f"server {self.name!r} was restarted under tools/call twice; giving up")

    def _restart(self, generation: int) -> str:
        """Replace the process after a timed-out call (a stdio server still
        busy with the call it never answered cannot take the next one; agent6
        owns the spawn), once per generation, and say what happened for the
        call's error."""
        with self._restart_lock:
            if self._generation != generation:
                return "the server was already restarted by another call"
            self._generation += 1
            self.close()
            self._reader_stop.clear()
            self._errors = []
            # PDEATHSIG ties a child to the THREAD that forked it (the launcher's
            # own tie and `die_with_parent` alike), and this caller may be a pool
            # worker about to end: the spawn runs on a keeper thread that lives
            # as long as this process does.
            release = threading.Event()
            spawned = threading.Event()
            failure: list[MCPError] = []

            def keep() -> None:
                try:
                    self.start()
                except MCPError as exc:
                    failure.append(exc)
                spawned.set()
                release.wait()

            self._keeper_release = release
            threading.Thread(target=keep, name=f"mcp-keeper[{self.name}]", daemon=True).start()
            spawned.wait()
            if failure:
                return f"restarting it failed ({failure[0]})"
            return "the server was restarted"

    def _call(self, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        result = self._request(
            "tools/call",
            {"name": tool_name, "arguments": arguments},
            timeout_s=self.call_timeout_s,
        )
        if not isinstance(result, dict):
            raise MCPError(f"server {self.name!r} tools/call returned non-dict result")
        if result.get("isError") is True:
            # MCP tool-execution failure: the spec returns it as a SUCCESSFUL
            # JSON-RPC result with isError=true (not a JSON-RPC error), so
            # surface it as an error here to match built-in tool semantics.
            content = result.get("content")
            text = ""
            if isinstance(content, list):
                text = " ".join(
                    c.get("text", "")
                    for c in content
                    if isinstance(c, dict) and isinstance(c.get("text"), str)
                ).strip()
            detail = _bounded_inline_text(text) or "(no detail)"
            raise MCPError(f"server {self.name!r} tool {tool_name!r} reported error: {detail}")
        return _bounded_result(result)

    def close(self) -> None:
        """Best-effort shutdown. Idempotent. Never raises.

        An HTTP server is the operator's: agent6 did not start it and must not
        stop it. There is nothing to tear down but the connection, which each
        request already closes.
        """
        self._reader_stop.set()
        if self._keeper_release is not None:
            self._keeper_release.set()
            self._keeper_release = None
        proc = self._proc
        self._proc = None
        if proc is None:
            return
        try:
            # The handle takes the whole process group down and sweeps a
            # `hardened` server's setsid escapees, which signalling the launcher
            # pid alone would miss.
            proc.close()
        finally:
            # Wake any thread blocked on _pending_cv so it can exit
            # cleanly instead of hanging on a server this teardown just killed.
            with self._pending_cv:
                self._pending_cv.notify_all()

    # ----- internals -----

    def _allocate_id(self) -> int:
        with self._id_lock:
            req_id = self._next_id
            self._next_id += 1
            return req_id

    def _reinitialize(self) -> None:
        """Re-run just the `initialize` handshake after a session expiry: the
        transport captures the fresh session id, and the tool list does not
        change, so there is nothing to re-list. HTTP only (a stdio server has
        no session to expire)."""
        init = self._request(
            "initialize",
            {
                "protocolVersion": _MCP_PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": {"name": "agent6", "version": __version__},
            },
            timeout_s=self.startup_timeout_s,
        )
        if not isinstance(init, dict):
            raise MCPError(f"server {self.name!r} returned non-dict re-initialize result")
        self._notify("notifications/initialized", {})

    def _request(
        self,
        method: str,
        params: dict[str, Any],
        *,
        timeout_s: float,
    ) -> Any:
        req_id = self._allocate_id()
        payload = {
            "jsonrpc": "2.0",
            "id": req_id,
            "method": method,
            "params": params,
        }
        if self.http is not None:
            # HTTP pairs request and response itself: no pending slot, no
            # reader thread, no id collision with a server-initiated request.
            try:
                response = self.http.send(payload, timeout_s=timeout_s)
            except MCPSessionExpired:
                # The server dropped this client's session (the transport already cleared
                # the id). Re-initialize per the spec and retry this request
                # once. The re-initialize carries no session id, so its own
                # 404 (if any) is a plain error and cannot loop back here.
                self._reinitialize()
                try:
                    response = self.http.send(payload, timeout_s=timeout_s)
                except MCPHttpError as exc:
                    raise MCPError(str(exc)) from exc
            except MCPHttpError as exc:
                raise MCPError(str(exc)) from exc
            if response is None:
                raise MCPError(f"server {self.name!r} sent no response to {method}")
            # The same two checks the stdio reader applies, for the same
            # reason: a keepalive frame, a server-initiated request
            # (sampling/createMessage, roots/list) or a multiplexing gateway
            # can put SOMEONE ELSE'S message first, and taking it handed the
            # model another request's answer as this call's result.
            if "method" in response:
                raise MCPError(
                    f"server {self.name!r} answered {method} with its own"
                    f" {response['method']!r} request, not a response"
                )
            if response.get("id") != req_id:
                raise MCPError(
                    f"server {self.name!r} answered {method} with a response to"
                    f" id {response.get('id')!r}, not to {req_id}"
                )
            return _result_of(response, name=self.name, method=method)
        generation = self._generation
        with self._pending_cv:
            self._pending[req_id] = None
        try:
            self._write_line(payload)
            deadline = time.monotonic() + timeout_s
            with self._pending_cv:
                while (response := self._pending[req_id]) is None:
                    if self._generation != generation:
                        raise MCPRestarted(f"server {self.name!r} was restarted under {method}")
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise MCPTimeout(
                            f"server {self.name!r} timed out after {timeout_s:.1f}s on {method}"
                        )
                    # If the reader thread died (server crashed mid-call)
                    # the call would otherwise wait the full timeout for nothing.
                    if self._reader is not None and not self._reader.is_alive():
                        # Its own words if it left any: a command that does not
                        # exist, a refused grant, the launcher's setup failure
                        # all read the same from out here otherwise.
                        said = self._redact_secrets(_stderr_tail(self._errors))
                        detail = f": {said}" if said else ""
                        raise MCPError(
                            f"server {self.name!r} died before responding to {method}{detail}"
                        )
                    self._pending_cv.wait(timeout=min(remaining, 0.25))
        finally:
            with self._pending_cv:
                self._pending.pop(req_id, None)
        return _result_of(response, name=self.name, method=method)

    def _notify(self, method: str, params: dict[str, Any]) -> None:
        # JSON-RPC notifications have no id and expect no response.
        payload = {"jsonrpc": "2.0", "method": method, "params": params}
        if self.http is not None:
            with contextlib.suppress(MCPHttpError):
                self.http.send(payload, timeout_s=self.startup_timeout_s)
            return
        self._write_line(payload)

    def _write_line(self, obj: dict[str, Any]) -> None:
        proc = self._proc
        if self.http is not None:
            raise MCPError(f"server {self.name!r} is HTTP; _write_line is the stdio path")
        if proc is None or proc.stdin is None:
            raise MCPError(f"server {self.name!r} is not writable (process gone)")
        line = json.dumps(obj, separators=(",", ":")).encode("utf-8") + b"\n"
        try:
            with self._stdin_lock:
                proc.stdin.write(line)
                proc.stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            raise MCPError(f"server {self.name!r} stdin closed: {exc}") from exc

    def _read_loop(self) -> None:
        proc = self._proc
        if proc is None or proc.stdout is None:
            return
        stream = proc.stdout
        while not self._reader_stop.is_set():
            try:
                # Bound the read: an unbounded readline() would buffer an entire
                # multi-GiB line from a runaway/malicious server into memory
                # BEFORE any size check, OOM'ing the agent. Cap at the limit + 1
                # so the reader can detect (and drain) an oversized line.
                raw = stream.readline(_MAX_LINE_BYTES + 1)
            except (OSError, ValueError):
                break
            if not raw:
                break  # EOF
            if len(raw) > _MAX_LINE_BYTES:
                # Oversized: drain the rest of this line (up to its newline) in
                # bounded chunks, discarding, then drop the whole payload.
                # Refusing to parse is safer than OOM on a runaway server.
                while raw and not raw.endswith(b"\n"):
                    raw = stream.readline(_MAX_LINE_BYTES + 1)
                continue
            try:
                msg = json.loads(raw.decode("utf-8", errors="replace"))
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue
            if not isinstance(msg, dict):
                continue
            req_id = msg.get("id")
            # We only consume responses: messages that carry an id this client sent
            # and have no "method" key. A message with both an int id and a
            # "method" is a server-INITIATED request (e.g. sampling/createMessage,
            # roots/list, elicitation/create); its id is the server's own counter
            # and can collide with one of ours, so it must NOT be stored as a
            # response. Notifications (no id) and server requests are ignored.
            if isinstance(req_id, int) and "method" not in msg:
                with self._pending_cv:
                    if req_id in self._pending:
                        self._pending[req_id] = msg
                        self._pending_cv.notify_all()


@dataclass
class MCPManager:
    """Owns N MCP server subprocesses for one agent6 run; closed by the
    lifecycle that built it.

    The `configs` arg is an iterable of (name, command, startup_timeout_s,
    call_timeout_s) tuples; we keep this constructor decoupled from the
    `Config` types so tests can pass plain tuples without booting
    the whole config validator.
    """

    _servers: dict[str, _MCPServer] = field(default_factory=dict)
    # Configured servers that did not start, in configuration order.
    failures: tuple[MCPStartFailure, ...] = ()

    @classmethod
    def start(
        cls,
        configs: Iterable[MCPServerSpec],
        *,
        logger: Callable[[str], None] | None = None,
        session_net: SessionNetwork | None = None,
    ) -> MCPManager:
        mgr = cls()
        failures: list[MCPStartFailure] = []
        for spec in configs:
            name = spec.name
            if name in mgr._servers:
                raise MCPError(f"duplicate MCP server name {name!r}")
            srv = _MCPServer(
                name=name,
                command=spec.command,
                startup_timeout_s=spec.startup_timeout_s,
                call_timeout_s=spec.call_timeout_s,
                pass_env=spec.pass_env,
                http=spec.http,
                policy=spec.policy,
                session_net=session_net,
            )
            try:
                srv.start()
            except MCPError as exc:
                # One bad server shouldn't take the whole agent down; it is
                # recorded and skipped, and the run simply does not see its
                # tools. The caller turns the record into a journal event, so
                # the absence reaches the conversation and not only a log.
                failures.append(MCPStartFailure(name=name, error=str(exc)))
                if logger is not None:
                    logger(f"[mcp] failed to start {name!r}: {exc}")
                srv.close()
                continue
            mgr._servers[name] = srv
            if logger is not None:
                # The RESOLVED network, not the configured word: `auto` means
                # nothing to a reader wondering why their browser server cannot
                # see the app. Named every time, for every server, so nobody has
                # to know to go looking.
                where = "unconfined" if srv.policy is None else srv.policy.network
                n = len(srv.tools)
                logger(
                    f"[mcp] started {name!r} ({n} tool{'' if n == 1 else 's'}, network: {where})"
                )
        mgr.failures = tuple(failures)
        return mgr

    def descriptors(self) -> tuple[MCPToolDescriptor, ...]:
        out: list[MCPToolDescriptor] = []
        for srv in self._servers.values():
            out.extend(srv.tools)
        return tuple(out)

    def call(self, qualified_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        server_name, tool_name = split_tool_name(qualified_name)
        srv = self._servers.get(server_name)
        if srv is None:
            raise MCPError(f"unknown MCP server: {server_name!r}")
        return srv.call_tool(tool_name, arguments)

    def close(self) -> None:
        for srv in self._servers.values():
            srv.close()
        self._servers.clear()
