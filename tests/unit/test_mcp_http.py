# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Connecting to an MCP server the operator runs, rather than spawning one."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any

import pytest

from agent6.config import Config
from agent6.tools.mcp_client import MCPManager, MCPServerSpec
from agent6.tools.mcp_http import MAX_BODY_BYTES, HttpTransport, MCPHttpError


def _serve(
    reply: Any,
    *,
    sse: bool = False,
    status: int = 200,
    body: bytes | None = None,
    encoding: str = "",
):
    """A one-connection MCP server on loopback. Returns (url, seen_headers)."""
    seen: dict[str, str] = {}

    class _Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            seen.update({k.lower(): v for k, v in self.headers.items()})
            request = json.loads(self.rfile.read(int(self.headers["content-length"])))
            self.send_response(status)
            self.send_header("content-type", "text/event-stream" if sse else "application/json")
            if encoding:
                self.send_header("content-encoding", encoding)
            self.end_headers()
            if body is not None:
                self.wfile.write(body)
                return
            answer = reply(request) if callable(reply) else reply
            raw = json.dumps(answer).encode()
            self.wfile.write(b"event: message\ndata: " + raw + b"\n\n" if sse else raw)

        def log_message(self, format: str, *args: Any) -> None:
            return  # a test server must not print to stderr

    httpd = HTTPServer(("127.0.0.1", 0), _Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return f"http://127.0.0.1:{httpd.server_port}/mcp", seen, httpd


def _mcp_reply(request: dict[str, Any]) -> dict[str, Any]:
    method = request.get("method")
    if method == "initialize":
        result: Any = {"protocolVersion": "2024-11-05", "capabilities": {}}
    elif method == "tools/list":
        result = {"tools": [{"name": "ping", "description": "d", "inputSchema": {}}]}
    else:
        result = {"content": [{"type": "text", "text": "pong"}]}
    return {"jsonrpc": "2.0", "id": request.get("id"), "result": result}


def test_agent6_connects_instead_of_spawning() -> None:
    """A server that wants a browser or a device is the operator's to run, in
    whatever container they chose; agent6 owning its lifetime is the wrong
    owner. The handshake and a call go over one POST each."""
    url, _seen, httpd = _serve(_mcp_reply)
    logs: list[str] = []
    try:
        mgr = MCPManager.start(
            [
                MCPServerSpec(
                    name="remote",
                    command=(),
                    startup_timeout_s=10.0,
                    call_timeout_s=10.0,
                    http=HttpTransport(name="remote", url=url),
                )
            ],
            logger=logs.append,
        )
        try:
            assert [(d.server_name, d.tool_name) for d in mgr.descriptors()] == [("remote", "ping")]
            # Its network is its own: agent6 jails nothing about it.
            assert logs == ["[mcp] started 'remote' (1 tool, network: remote (not jailed))"]
            assert mgr.call("mcp__remote__ping", {}) == {
                "content": [{"type": "text", "text": "pong"}]
            }
        finally:
            mgr.close()
    finally:
        httpd.shutdown()


def test_a_streamed_answer_is_read_like_any_other() -> None:
    """Streamable HTTP lets a server answer one request with an SSE frame.
    Reading only a bare body made every such server look like it sent
    garbage."""
    url, _seen, httpd = _serve(_mcp_reply, sse=True)
    try:
        got = HttpTransport(name="s", url=url).send({"jsonrpc": "2.0", "id": 1}, timeout_s=5.0)
        assert got is not None and got["id"] == 1
    finally:
        httpd.shutdown()


def test_the_token_is_read_from_the_environment_never_the_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A secret written into a config file is a secret in a backup."""
    monkeypatch.setenv("MCP_TEST_TOKEN", "s3cr3t")
    url, seen, httpd = _serve(_mcp_reply)
    try:
        HttpTransport(name="s", url=url, token_env="MCP_TEST_TOKEN").send(
            {"jsonrpc": "2.0", "id": 1}, timeout_s=5.0
        )
        assert seen["authorization"] == "Bearer s3cr3t"
    finally:
        httpd.shutdown()


def test_httpx_trust_env_reaches_the_client(monkeypatch: pytest.MonkeyPatch) -> None:
    """The per-server httpx_trust_env flag reaches the httpx client verbatim:
    off by default so a local server's bearer token never routes to an ambient
    proxy, on when the operator opts a proxied server in."""
    from agent6.tools import mcp_http

    seen: list[Any] = []
    real = mcp_http.httpx2.Client

    def _spy(**kwargs: Any) -> Any:
        seen.append(kwargs.get("trust_env"))
        return real(**kwargs)

    monkeypatch.setattr(mcp_http.httpx2, "Client", _spy)
    url, _s, httpd = _serve(_mcp_reply)
    try:
        HttpTransport(name="s", url=url).send({"jsonrpc": "2.0", "id": 1}, timeout_s=5.0)
        HttpTransport(name="s", url=url, httpx_trust_env=True).send(
            {"jsonrpc": "2.0", "id": 2}, timeout_s=5.0
        )
    finally:
        httpd.shutdown()
    assert seen == [False, True]


def test_httpx_trust_env_is_rejected_on_a_spawned_server() -> None:
    """It only affects the http client dialling a `url` server; a spawned
    (command) server has no client, so the setting is refused rather than
    silently dead."""
    Config.model_validate(
        {"mcp": {"servers": {"r": {"url": "https://h/mcp", "httpx_trust_env": True}}}}
    )
    with pytest.raises(Exception, match="httpx_trust_env is for"):
        Config.model_validate(
            {"mcp": {"servers": {"s": {"command": ["srv"], "httpx_trust_env": True}}}}
        )


def test_an_oversized_body_is_refused_rather_than_buffered() -> None:
    """The same bound the stdio reader applies: a runaway server must not be
    able to buffer an unbounded body into the agent."""
    url, _seen, httpd = _serve(None, body=b"x" * (MAX_BODY_BYTES + 64))
    try:
        with pytest.raises(MCPHttpError, match="more than"):
            HttpTransport(name="s", url=url).send({"jsonrpc": "2.0", "id": 1}, timeout_s=10.0)
    finally:
        httpd.shutdown()


def test_a_compressed_answer_is_refused_not_decoded() -> None:
    """The identity we ask for binds nothing: the server's `Content-Encoding`
    picks httpx's decoder, so a compromised server's small body expanded in
    memory ahead of the byte count. Anything but identity is refused."""
    import gzip

    payload = json.dumps({"jsonrpc": "2.0", "id": 1, "result": "ok"}).encode()
    url, _seen, httpd = _serve(None, body=gzip.compress(payload), encoding="gzip")
    try:
        with pytest.raises(MCPHttpError, match="content-encoding"):
            HttpTransport(name="s", url=url).send({"jsonrpc": "2.0", "id": 1}, timeout_s=5.0)
    finally:
        httpd.shutdown()


def test_an_http_failure_is_a_clean_tool_error() -> None:
    url, _seen, httpd = _serve(_mcp_reply, status=503)
    try:
        with pytest.raises(MCPHttpError, match="HTTP 503"):
            HttpTransport(name="s", url=url).send({"jsonrpc": "2.0", "id": 1}, timeout_s=5.0)
    finally:
        httpd.shutdown()


@pytest.mark.parametrize(
    ("entry", "message"),
    [
        ({}, "exactly one"),
        ({"command": ["x"], "url": "https://h/mcp"}, "exactly one"),
        ({"url": "ftp://h/mcp"}, "http"),
        ({"command": ["x"], "token_env": "T"}, "pass_env"),
    ],
)
def test_a_server_names_one_transport(entry: dict[str, Any], message: str) -> None:
    """Both or neither is a config error, not a guess."""
    with pytest.raises(ValueError, match=message):
        Config.model_validate({"mcp": {"enabled": True, "servers": {"s": entry}}})


def test_a_token_that_cannot_be_a_header_is_refused_before_it_leaks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A token file with CRLF endings keeps the CR. The HTTP layer then raises
    with the header VALUE in its message, and that message reaches stderr, the
    launch log and the model's context."""
    monkeypatch.setenv("MCP_TEST_TOKEN", "sk-live-DEADBEEF\r")
    with pytest.raises(MCPHttpError) as caught:
        HttpTransport(name="s", url="https://h/mcp", token_env="MCP_TEST_TOKEN").send(
            {"jsonrpc": "2.0", "id": 1}, timeout_s=5.0
        )
    assert "DEADBEEF" not in str(caught.value)
    assert "not a usable header value" in str(caught.value)


def test_an_unreachable_server_never_quotes_the_exception_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The message is the exception TYPE. Its text can quote a rejected header
    value straight back at us -- and InvalidURL does not derive from HTTPError,
    so a narrower catch let an operator typo crash the run."""
    monkeypatch.setenv("MCP_TEST_TOKEN", "s3cr3t")
    with pytest.raises(MCPHttpError) as caught:
        HttpTransport(name="s", url="http://[::1/mcp", token_env="MCP_TEST_TOKEN").send(
            {"jsonrpc": "2.0", "id": 1}, timeout_s=5.0
        )
    assert "s3cr3t" not in str(caught.value)
    assert "unreachable" in str(caught.value)


def test_a_body_is_capped_while_it_arrives_not_after() -> None:
    """`response.content` materializes first: a 400 MiB body reached 849 MiB of
    RSS before the check, and a 1 MiB gzip bomb reached 2 GiB -- enough to OOM
    the process that owns the run and the provider keys."""
    import tracemalloc

    url, seen, httpd = _serve(None, body=b"x" * (48 << 20))
    try:
        tracemalloc.start()
        try:
            with pytest.raises(MCPHttpError, match="more than"):
                HttpTransport(name="s", url=url).send({"jsonrpc": "2.0", "id": 1}, timeout_s=30.0)
            peak = tracemalloc.get_traced_memory()[1]
        finally:
            tracemalloc.stop()
    finally:
        httpd.shutdown()
    assert peak < 32 << 20, f"buffered {peak} bytes of a 48 MiB body"
    assert seen["accept-encoding"] == "identity", "a decoded stream expands past the cap"


def test_an_ambient_proxy_does_not_capture_the_token(monkeypatch: pytest.MonkeyPatch) -> None:
    """httpx trusts the environment by default and has no loopback bypass, so
    an exported HTTP_PROXY sent the bearer token to the proxy in cleartext
    while the operator own server received nothing."""
    monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:9")
    monkeypatch.setenv("MCP_TEST_TOKEN", "s3cr3t")
    url, seen, httpd = _serve(_mcp_reply)
    try:
        HttpTransport(name="s", url=url, token_env="MCP_TEST_TOKEN").send(
            {"jsonrpc": "2.0", "id": 1}, timeout_s=5.0
        )
    finally:
        httpd.shutdown()
    assert seen["authorization"] == "Bearer s3cr3t", "the request went to the proxy"


def test_another_requests_answer_is_not_taken_as_this_one() -> None:
    """A keepalive frame, a server-initiated request, or a multiplexing gateway
    can put SOMEONE ELSE message first. The stdio reader has always checked the
    id; taking the first frame handed the model another call answer."""
    from agent6.tools.mcp_client import (
        MCPError,
        _MCPServer,  # pyright: ignore[reportPrivateUsage]
    )

    body = (
        b'data: {"jsonrpc":"2.0","id":4242,"result":"ANOTHER ANSWER"}\n\n'
        b'data: {"jsonrpc":"2.0","id":1,"result":"MINE"}\n\n'
    )
    url, _seen, httpd = _serve(None, sse=True, body=body)
    try:
        srv = _MCPServer(  # pyright: ignore[reportPrivateUsage]
            name="s",
            command=(),
            startup_timeout_s=5.0,
            call_timeout_s=5.0,
            http=HttpTransport(name="s", url=url),
        )
        with pytest.raises(MCPError, match="a response to id 4242"):
            srv._request("tools/call", {}, timeout_s=5.0)  # pyright: ignore[reportPrivateUsage]
    finally:
        httpd.shutdown()


def test_a_server_request_is_not_taken_as_a_response() -> None:
    from agent6.tools.mcp_client import (
        MCPError,
        _MCPServer,  # pyright: ignore[reportPrivateUsage]
    )

    body = b'data: {"jsonrpc":"2.0","id":1,"method":"sampling/createMessage","params":{}}\n\n'
    url, _seen, httpd = _serve(None, sse=True, body=body)
    try:
        srv = _MCPServer(  # pyright: ignore[reportPrivateUsage]
            name="s",
            command=(),
            startup_timeout_s=5.0,
            call_timeout_s=5.0,
            http=HttpTransport(name="s", url=url),
        )
        with pytest.raises(MCPError, match="its own"):
            srv._request("tools/call", {}, timeout_s=5.0)  # pyright: ignore[reportPrivateUsage]
    finally:
        httpd.shutdown()


@pytest.mark.parametrize(
    "body",
    [
        b'id: 7\nevent: message\ndata: {"jsonrpc":"2.0","id":1,"result":"ok"}\n\n',
        b'retry: 100\ndata: {"jsonrpc":"2.0","id":1,"result":"ok"}\n\n',
        b'\xef\xbb\xbfdata: {"jsonrpc":"2.0","id":1,"result":"ok"}\n\n',
        b'data: {"jsonrpc":"2.0",\ndata: "id":1,"result":"ok"}\n\n',
        b'data: {"jsonrpc":"2.0","id":1,"result":"ok"}\r\n\r\n',
        b': a comment\ndata: {"jsonrpc":"2.0","id":1,"result":"ok"}\n\n',
    ],
    ids=["id-field", "retry-field", "bom", "multi-line-data", "crlf", "comment"],
)
def test_every_spec_legal_sse_framing_is_read(body: bytes) -> None:
    """A line scan rejected all of these as invalid JSON."""
    url, _seen, httpd = _serve(None, sse=True, body=body)
    try:
        got = HttpTransport(name="s", url=url).send({"jsonrpc": "2.0", "id": 1}, timeout_s=5.0)
        assert got is not None and got["result"] == "ok"
    finally:
        httpd.shutdown()


def test_a_line_separator_inside_a_json_string_does_not_cut_the_message() -> None:
    """U+2028/U+2029/U+0085 are LEGAL raw characters inside a JSON string, and
    `str.splitlines()` splits on them -- so a tool result containing one was
    cut in half every time, and the model could plant one deliberately."""
    sep = "\u2028"
    payload = json.dumps({"jsonrpc": "2.0", "id": 1, "result": f"a{sep}bc"})
    url, _seen, httpd = _serve(None, sse=True, body=f"data: {payload}\n\n".encode())
    try:
        got = HttpTransport(name="s", url=url).send({"jsonrpc": "2.0", "id": 1}, timeout_s=5.0)
        assert got is not None and got["result"] == f"a{sep}bc"
    finally:
        httpd.shutdown()
