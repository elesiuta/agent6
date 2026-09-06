# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Talk to an MCP server the OPERATOR is running, over HTTP.

The stdio transport has agent6 spawn the server, which means agent6 owns its
environment, its lifetime and its confinement. For a server that wants a
browser, a device or a network of its own, that is the wrong owner: the
operator runs it however they like -- their container, their sandbox, their
credentials -- and agent6 only connects.

One request, one response: JSON-RPC over POST. What that buys in simplicity it
does not buy in trust, so this side carries the same defences the `fetch` tool
does -- no compression, a streamed cap, a total deadline -- plus the id check
the stdio reader has always applied. A response is only this call's answer if
it says so.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Any

import httpx2

from agent6.tools.http_body import BodyRefused, read_capped

# The same bound the stdio reader applies, and applied the same way: while the
# body arrives, not after. `response.content` materializes first, so a 400 MiB
# body reached 849 MiB of RSS before the check, and a 1 MiB gzip bomb reached
# 2 GiB -- enough to OOM the process that owns the run and the provider keys.
MAX_BODY_BYTES = 8 << 20


def _clean_session_id(value: str) -> str:
    """A server-assigned session id the transport can safely ECHO back in a header, or "".

    The value comes from an operator-run server but crosses the wire, so it is
    untrusted the same way the token in `_auth` is: a non-ASCII byte makes the
    HTTP layer raise on the next send with the value IN its message (which
    reaches stderr, the launch log and the model's context), and a control
    character rides straight into the outgoing header. The spec restricts a
    session id to visible ASCII (0x21-0x7E); anything else is dropped so we
    simply do not echo it, and the caller treats it as a stateless response.
    Never raises, never quotes the value: a malformed id degrades to no
    session, it does not take the connection down.
    """
    if value and all("\x21" <= ch <= "\x7e" for ch in value):
        return value
    return ""


# The version agent6 negotiates in `initialize`, echoed on every later request
# as the spec requires.
PROTOCOL_VERSION = "2024-11-05"


class MCPHttpError(Exception):
    """The server could not be reached, or answered with something unusable."""


class MCPSessionExpired(MCPHttpError):
    """A stateful server answered a request carrying this transport's session id with 404:
    the spec's signal that it expired the session. The caller re-initializes.
    A subclass of MCPHttpError so a plain `except MCPHttpError` still catches
    it, but the manager can single it out to re-handshake."""


@dataclass(slots=True)
class HttpTransport:
    """A connection to one operator-run MCP server. Not frozen: `session_id`
    is live connection state the server assigns on `initialize` (the rest is
    config)."""

    name: str
    url: str
    # The env var holding the bearer token, named in config. The VALUE is read
    # here and never logged, never written to a transcript, and never part of
    # an error message.
    token_env: str = ""
    # Forward httpx's trust_env (default off): the ambient HTTP(S)_PROXY is
    # ignored, so this server's bearer token never routes to a proxy. See `send`.
    httpx_trust_env: bool = False
    # The streamable-HTTP session id: captured from the `initialize` response
    # (see `send`), echoed on every later request (see `_headers`), and cleared
    # on the 404 that means the server expired it. Stays "" for a stateless
    # server, which never sends one.
    session_id: str = ""

    def _auth(self) -> str:
        """The bearer header value, or "" -- refusing a token that cannot be
        one. A stray CR (a token file with CRLF endings) makes the HTTP layer
        raise with the header VALUE in its message, and that message reaches
        stderr, the launch log and the model's context."""
        token = os.environ.get(self.token_env, "") if self.token_env else ""
        if not token:
            return ""
        if any(ch in token for ch in "\r\n\x00") or not token.isprintable():
            raise MCPHttpError(
                f"the token in ${self.token_env} is not a usable header value"
                " (it contains a newline or a control character)"
            )
        return f"Bearer {token}"

    def _headers(self) -> dict[str, str]:
        headers = {
            "content-type": "application/json",
            # Streamable HTTP: a server may answer with either.
            "accept": "application/json, text/event-stream",
            "mcp-protocol-version": PROTOCOL_VERSION,
            # Compression is declined here and refused in `send` if the server
            # answers with it anyway: the cap counts what ARRIVES, and a
            # decoded stream would expand past it before any check.
            "accept-encoding": "identity",
        }
        if auth := self._auth():
            headers["authorization"] = auth
        if self.session_id:
            headers["mcp-session-id"] = self.session_id
        return headers

    def send(self, payload: dict[str, Any], *, timeout_s: float) -> dict[str, Any] | None:
        """POST one JSON-RPC message; return the response, or None for a
        notification the server acknowledged with no body.

        `trust_env` is off by default: an ambient `HTTP_PROXY` would otherwise
        capture this connection -- loopback included, since httpx has no implicit
        bypass -- sending the bearer token to the proxy while the operator's own
        server received nothing. `[mcp.servers.<name>].httpx_trust_env` opts a
        server in (one reachable only through the environment's proxy).
        """
        try:
            with (
                httpx2.Client(
                    timeout=timeout_s, follow_redirects=False, trust_env=self.httpx_trust_env
                ) as client,
                client.stream(
                    "POST",
                    self.url,
                    headers=self._headers(),
                    content=json.dumps(payload, separators=(",", ":")).encode("utf-8"),
                ) as response,
            ):
                if response.status_code == 404 and self.session_id:
                    # The spec: a 404 to a request bearing a session id means
                    # the server expired that session. Drop it so the transport does not
                    # keep echoing a dead id, and signal a re-initialize.
                    self.session_id = ""
                    raise MCPSessionExpired(f"server {self.name!r} expired its session (HTTP 404)")
                if response.status_code >= 400:
                    raise MCPHttpError(f"server {self.name!r} returned HTTP {response.status_code}")
                # The server assigns the session id on the initialize response;
                # capture it here and every request after echoes it. A stateless
                # server sends none, so this leaves session_id "".
                assigned = _clean_session_id(response.headers.get("mcp-session-id", ""))
                if assigned:
                    self.session_id = assigned
                deadline = time.monotonic() + timeout_s
                try:
                    body = read_capped(
                        response, cap=MAX_BODY_BYTES, deadline=deadline, timeout_s=timeout_s
                    )
                except BodyRefused as exc:
                    raise MCPHttpError(f"server {self.name!r}: {exc}") from exc
        except MCPHttpError:
            raise
        except Exception as exc:
            # Deliberately broad: httpx2.InvalidURL does not derive from
            # HTTPError, so an operator typo in `url` escaped a narrower catch
            # and crashed the run instead of being logged and skipped. The
            # message is the exception's TYPE, never its text, which can quote
            # a rejected header value back into the run's output.
            raise MCPHttpError(f"server {self.name!r} unreachable ({type(exc).__name__})") from None
        if not body.strip():
            return None  # an accepted notification
        message = _parse(body, name=self.name)
        return message


def _parse(raw: bytes, *, name: str) -> dict[str, Any]:
    """The JSON-RPC message in *raw*, whether it arrived bare or as SSE."""
    text = raw.decode("utf-8", errors="replace").lstrip("﻿")
    if text.lstrip().startswith(("event:", "data:", "id:", "retry:", ":")):
        text = _sse_data(text, name=name)
    try:
        message = json.loads(text)
    except (json.JSONDecodeError, ValueError) as exc:
        raise MCPHttpError(f"server {name!r} sent invalid JSON: {exc}") from exc
    if not isinstance(message, dict):
        raise MCPHttpError(f"server {name!r} sent a non-object response")
    return message


def _sse_data(text: str, *, name: str) -> str:
    """The `data` payload of the first SSE event carrying one.

    A real field parser, not a line scan: an event may open with `id:` or
    `retry:` (resumability), may carry `data` across several lines the spec
    says to join with newlines, and its line endings may be CR, LF or CRLF.
    `str.splitlines()` also splits on U+2028/U+2029/U+0085, which are LEGAL
    raw characters inside a JSON string -- so a tool result containing one was
    cut in half, every time, and the model could plant one deliberately.
    """
    data: list[str] = []
    for line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        if line.startswith("data:"):
            data.append(line[len("data:") :].removeprefix(" "))
        elif not line.strip() and data:
            break  # end of the first event that carried data
    if not data:
        raise MCPHttpError(f"server {name!r} sent an SSE response with no data")
    return "\n".join(data)
