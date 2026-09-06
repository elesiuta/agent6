# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The ACP transport and handshake, driven the way an editor drives it."""

from __future__ import annotations

import io
import json
from typing import Any

from agent6.ui.acp.server import (
    INVALID_REQUEST,
    MAX_LINE_BYTES,
    METHOD_NOT_FOUND,
    PROTOCOL_VERSION,
    ACPServer,
    capabilities_from,
)


def _exchange(*messages: object, raw: bytes = b"") -> list[dict[str, Any]]:
    """Feed messages in, return whatever came back out."""
    payload = raw or b"".join(json.dumps(m).encode() + b"\n" for m in messages)
    out = io.BytesIO()
    ACPServer(stdin=io.BytesIO(payload), stdout=out).serve()
    return [json.loads(line) for line in out.getvalue().splitlines() if line.strip()]


def _init(**client_caps: Any) -> dict[str, Any]:
    return {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {"protocolVersion": 1, "clientCapabilities": client_caps},
    }


def test_the_handshake_answers_with_what_agent6_can_do() -> None:
    (reply,) = _exchange(_init())
    assert reply["id"] == 1
    result = reply["result"]
    assert result["protocolVersion"] == PROTOCOL_VERSION
    assert result["agentInfo"]["name"] == "agent6"


def test_session_load_is_reported_absent_rather_than_half_answered() -> None:
    """It is exactly what ACP v2 reorganises, and resume is where agent6 has
    the most of its own semantics."""
    (reply,) = _exchange(_init())
    assert reply["result"]["agentCapabilities"]["loadSession"] is False


def test_the_clients_capabilities_become_the_frontend_seam() -> None:
    """The whole reason FrontendCapabilities went in first: ACP's handshake IS
    a capability exchange, so it maps rather than needing new plumbing."""
    bare = capabilities_from({})
    assert bare.can_ask is True, "every ACP client must answer session/request_permission"


def test_an_unknown_method_is_an_error_not_a_crash() -> None:
    (reply,) = _exchange({"jsonrpc": "2.0", "id": 7, "method": "session/load", "params": {}})
    assert reply["error"]["code"] == METHOD_NOT_FOUND
    assert "session/load" in reply["error"]["message"]


def test_a_notification_is_acted_on_and_not_answered() -> None:
    """JSON-RPC: no id means no reply. Answering one desynchronises a client
    that is not waiting for anything."""
    assert _exchange({"jsonrpc": "2.0", "method": "initialize", "params": {}}) == []


def test_a_request_with_no_method_is_refused_by_id() -> None:
    (reply,) = _exchange({"jsonrpc": "2.0", "id": 3})
    assert reply["error"]["code"] == INVALID_REQUEST


def test_garbage_does_not_kill_the_connection() -> None:
    """An editor that sends one bad line must not lose the session."""
    replies = _exchange(raw=b"not json\n" + json.dumps(_init()).encode() + b"\n")
    assert len(replies) == 1 and replies[0]["id"] == 1


def test_an_oversized_line_is_dropped_not_buffered() -> None:
    """An unbounded readline buffers the whole line BEFORE any size check, so
    a runaway client could exhaust memory before the cap could refuse it."""
    huge = b'{"jsonrpc":"2.0","id":9,"method":"initialize","params":{"x":"'
    huge += b"A" * (MAX_LINE_BYTES + 64) + b'"}}\n'
    replies = _exchange(raw=huge + json.dumps(_init()).encode() + b"\n")
    assert [r["id"] for r in replies] == [1], "the oversized message was answered"


def test_text_that_cannot_encode_does_not_desynchronise_the_stream() -> None:
    """A lone surrogate in model-emitted text would otherwise raise mid-write,
    leaving a half-written line an editor cannot parse."""
    out = io.BytesIO()
    server = ACPServer(stdin=io.BytesIO(b""), stdout=out)
    server.notify_raw({"jsonrpc": "2.0", "method": "x", "params": {"t": "ok \ud83d tail"}})
    line = out.getvalue()
    assert line.endswith(b"\n")
    assert json.loads(line)["params"]["t"].startswith("ok ")
