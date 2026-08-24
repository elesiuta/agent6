# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Tests for ChatGPTProvider (Responses request build, SSE parse, auth)."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any
from unittest import mock

import pytest

from agent6.budget import BudgetTracker
from agent6.providers import ProviderError
from agent6.providers.chatgpt import (
    ChatGPTProvider,
    responses_input,
    tools_to_responses,
)
from agent6.providers.chatgpt_oauth import ChatGPTCredential
from agent6.providers.types import ToolDefinition
from agent6.secrets import OAuthTokens, save_oauth_tokens


@pytest.fixture
def signed_in(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> ChatGPTCredential:
    """A gcfg-backed credential holding an unexpired sign-in."""
    monkeypatch.setenv("AGENT6_CONFIG_HOME", str(tmp_path / "g"))
    save_oauth_tokens("chatgpt", OAuthTokens("AT0", "RT1", time.time() + 3600, "acct-1"))
    return ChatGPTCredential("chatgpt", issuer="https://auth.example", client_id="app_X")


def _provider(credential: ChatGPTCredential, **kwargs: Any) -> ChatGPTProvider:
    return ChatGPTProvider(
        model="gpt-5-codex",
        credential=credential,
        account_id="acct-1",
        base_url="https://chatgpt.com/backend-api/codex",
        **kwargs,
    )


class _FakeStreamResponse:
    def __init__(self, *, status_code: int, lines: list[str], error_body: str = "") -> None:
        self.status_code = status_code
        self._lines = lines
        self._error_body = error_body
        self.headers: dict[str, str] = {}

    def __enter__(self) -> _FakeStreamResponse:
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def iter_lines(self) -> list[str]:
        return self._lines

    def read(self) -> bytes:
        return self._error_body.encode("utf-8")


def _evt(data: dict[str, Any]) -> list[str]:
    return [f"event: {data.get('type', '')}", f"data: {json.dumps(data)}", ""]


_USAGE = {
    "input_tokens": 42,
    "input_tokens_details": {"cached_tokens": 7},
    "output_tokens": 9,
    "total_tokens": 51,
}


def _serve(lines: list[str]):
    """A stream stub serving *lines* regardless of the request."""

    def stream(method: str, url: str, **kwargs: Any) -> _FakeStreamResponse:
        del method, url, kwargs
        return _FakeStreamResponse(status_code=200, lines=lines)

    return stream


def _happy_stream() -> list[str]:
    out: list[str] = []
    out += _evt({"type": "response.created", "response": {"id": "resp_1"}})
    out += _evt({"type": "response.output_text.delta", "delta": "hel"})
    out += _evt({"type": "response.output_text.delta", "delta": "lo"})
    out += _evt(
        {
            "type": "response.output_item.done",
            "item": {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "hello"}],
            },
        }
    )
    out += _evt(
        {
            "type": "response.completed",
            "response": {"id": "resp_1", "status": "completed", "usage": _USAGE},
        }
    )
    return out


def test_request_body_and_headers_speak_the_codex_dialect(
    signed_in: ChatGPTCredential,
) -> None:
    provider = _provider(signed_in, reasoning_effort="medium")
    captured: dict[str, Any] = {}

    def fake_stream(method: str, url: str, **kwargs: Any) -> _FakeStreamResponse:
        captured["url"] = url
        captured["headers"] = kwargs["headers"]
        captured["body"] = json.loads(kwargs["content"])
        return _FakeStreamResponse(status_code=200, lines=_happy_stream())

    tools = [ToolDefinition(name="read_file", description="d", input_schema={"type": "object"})]
    history: list[dict[str, Any]] = [
        {"role": "user", "content": "task"},
        {
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "private"},
                {"type": "text", "text": "on it"},
                {"type": "tool_use", "id": "call_1", "name": "read_file", "input": {"p": "."}},
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "call_1",
                    "content": [{"type": "text", "text": "ok"}],
                }
            ],
        },
    ]
    with mock.patch("httpx2.stream", side_effect=fake_stream):
        resp = provider.call(system="SYS", messages=history, tools=tools, temperature=0.7)

    assert captured["url"] == "https://chatgpt.com/backend-api/codex/responses"
    body = captured["body"]
    assert body["model"] == "gpt-5-codex" and body["instructions"] == "SYS"
    assert body["store"] is False and body["stream"] is True
    assert body["include"] == ["reasoning.encrypted_content"]
    assert body["prompt_cache_key"] == provider.session_id
    assert len(provider.session_id) <= 64
    assert body["reasoning"] == {"effort": "medium", "summary": "auto"}
    assert "max_output_tokens" not in body and "temperature" not in body
    assert body["tools"] == [
        {
            "type": "function",
            "name": "read_file",
            "description": "d",
            "parameters": {"type": "object"},
            "strict": False,
        }
    ]
    assert body["input"] == [
        {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "task"}]},
        {
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "on it"}],
        },
        {
            "type": "function_call",
            "call_id": "call_1",
            "name": "read_file",
            "arguments": '{"p": "."}',
        },
        {"type": "function_call_output", "call_id": "call_1", "output": "ok"},
    ]
    headers = captured["headers"]
    assert headers["authorization"] == "Bearer AT0"
    assert headers["chatgpt-account-id"] == "acct-1"
    assert headers["originator"] == "agent6"
    assert headers["openai-beta"] == "responses=experimental"
    assert headers["accept"] == "text/event-stream"
    assert headers["session-id"] == provider.session_id
    assert resp.text == "hello"


def test_stream_deltas_feed_callbacks_and_usage_normalises(
    signed_in: ChatGPTCredential,
) -> None:
    provider = _provider(signed_in)
    pieces: list[str] = []
    with mock.patch(
        "httpx2.stream",
        side_effect=_serve(_happy_stream()),
    ):
        resp = provider.call(
            system="s",
            messages=[{"role": "user", "content": "x"}],
            text_delta_callback=pieces.append,
        )
    assert pieces == ["hel", "lo"]
    assert resp.text == "hello" and resp.stop_reason == "end_turn"
    assert (resp.input_tokens, resp.cache_read_tokens, resp.output_tokens) == (35, 7, 9)
    assert resp.cost_usd == 0.0


def test_tool_call_and_reasoning_items_parse(signed_in: ChatGPTCredential) -> None:
    lines: list[str] = []
    lines += _evt(
        {
            "type": "response.output_item.done",
            "item": {
                "type": "reasoning",
                "summary": [{"type": "summary_text", "text": "thought"}],
            },
        }
    )
    lines += _evt(
        {
            "type": "response.output_item.done",
            "item": {
                "type": "function_call",
                "id": "fc_1",
                "call_id": "call_9",
                "name": "run",
                "arguments": '{"cmd": "ls"}',
            },
        }
    )
    lines += _evt(
        {
            "type": "response.completed",
            "response": {"id": "r", "status": "completed", "usage": _USAGE},
        }
    )
    provider = _provider(signed_in)
    with mock.patch(
        "httpx2.stream",
        side_effect=_serve(lines),
    ):
        resp = provider.call(system="s", messages=[{"role": "user", "content": "x"}])
    assert resp.tool_uses == ({"id": "call_9", "name": "run", "input": {"cmd": "ls"}},)
    thinking = resp.raw["content"][0]
    assert thinking["type"] == "thinking" and thinking["thinking"] == "thought"
    assert thinking["chatgpt_reasoning"]["type"] == "reasoning"


def test_incomplete_max_output_tokens_maps_to_max_tokens(
    signed_in: ChatGPTCredential,
) -> None:
    lines = _evt(
        {
            "type": "response.incomplete",
            "response": {
                "id": "r",
                "status": "incomplete",
                "incomplete_details": {"reason": "max_output_tokens"},
                "usage": _USAGE,
                "output": [
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "partial"}],
                    }
                ],
            },
        }
    )
    provider = _provider(signed_in)
    with mock.patch(
        "httpx2.stream",
        side_effect=_serve(lines),
    ):
        resp = provider.call(system="s", messages=[{"role": "user", "content": "x"}])
    assert resp.stop_reason == "max_tokens" and resp.text == "partial"


def test_failed_event_raises_with_usage_limit_status(signed_in: ChatGPTCredential) -> None:
    lines = _evt(
        {
            "type": "response.failed",
            "response": {
                "error": {
                    "code": "usage_limit_reached",
                    "message": "limit hit",
                    "plan_type": "plus",
                }
            },
        }
    )
    provider = _provider(signed_in)
    with (
        mock.patch(
            "httpx2.stream",
            side_effect=_serve(lines),
        ),
        pytest.raises(ProviderError) as exc,
    ):
        provider.call(system="s", messages=[{"role": "user", "content": "x"}])
    assert exc.value.status_code == 429 and "plus plan" in str(exc.value)


def test_cut_stream_is_retryable_not_a_completed_turn(signed_in: ChatGPTCredential) -> None:
    lines = _evt({"type": "response.output_text.delta", "delta": "half"})
    provider = _provider(signed_in)
    with (
        mock.patch(
            "httpx2.stream",
            side_effect=_serve(lines),
        ),
        pytest.raises(ProviderError, match="ended without"),
    ):
        provider.call(system="s", messages=[{"role": "user", "content": "x"}])


def test_401_refreshes_the_credential_once_and_retries(
    signed_in: ChatGPTCredential, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A 401 from the backend invalidates the cached token, refreshes via the
    token endpoint, and re-sends with the fresh bearer."""

    def fake_refresh(url: str, data: dict[str, str], timeout_s: float) -> Any:
        class R:
            status_code = 200
            text = ""

            @staticmethod
            def json() -> dict[str, Any]:
                return {"access_token": "AT1", "refresh_token": "RT2", "expires_in": 3600}

        return R()

    monkeypatch.setattr("agent6.providers.chatgpt_oauth._post_form", fake_refresh)
    seen_auth: list[str] = []

    def fake_stream(method: str, url: str, **kwargs: Any) -> _FakeStreamResponse:
        seen_auth.append(kwargs["headers"]["authorization"])
        if len(seen_auth) == 1:
            return _FakeStreamResponse(status_code=401, lines=[], error_body="expired")
        return _FakeStreamResponse(status_code=200, lines=_happy_stream())

    provider = _provider(signed_in)
    with mock.patch("httpx2.stream", side_effect=fake_stream):
        resp = provider.call(system="s", messages=[{"role": "user", "content": "x"}])
    assert seen_auth == ["Bearer AT0", "Bearer AT1"]
    assert resp.text == "hello"


def test_budgeted_call_requires_usage(signed_in: ChatGPTCredential) -> None:
    lines = _evt({"type": "response.completed", "response": {"id": "r", "status": "completed"}})
    provider = _provider(
        signed_in, budget=BudgetTracker(max_usd=-1, max_tokens_fallback=1_000_000, max_percent=-1)
    )
    with (
        mock.patch(
            "httpx2.stream",
            side_effect=_serve(lines),
        ),
        pytest.raises(ProviderError, match="usage"),
    ):
        provider.call(system="s", messages=[{"role": "user", "content": "x"}])


def test_responses_input_flattens_odd_content() -> None:
    items = responses_input(
        [
            {"role": "system", "content": "mapped to user"},
            {
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": "c", "content": "plain"}],
            },
            {"role": "assistant", "content": ""},
        ]
    )
    assert items == [
        {
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": "mapped to user"}],
        },
        {"type": "function_call_output", "call_id": "c", "output": "plain"},
    ]
    assert tools_to_responses([])[0:0] == []


def test_responses_input_drops_blank_name_calls_and_their_results() -> None:
    """A blank-name tool_use (another provider's malformed call, carried in a
    resumed history) is skipped together with its paired tool_result, so the
    replayed conversation never holds an output with no matching call."""
    items = responses_input(
        [
            {
                "role": "assistant",
                "content": [{"type": "tool_use", "id": "bad", "name": " ", "input": {}}],
            },
            {
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": "bad", "content": "x"}],
            },
        ]
    )
    assert items == []


def test_plan_usage_headers_feed_the_percent_budget(signed_in: ChatGPTCredential) -> None:
    """The x-codex primary-window headers ride each response into the budget:
    plan-metered (no fallback drain, $0 authoritative) with the account
    percent observable in the snapshot."""
    provider = _provider(
        signed_in, budget=BudgetTracker(max_usd=10.0, max_tokens_fallback=100, max_percent=-1)
    )

    def stream(method: str, url: str, **kwargs: Any) -> _FakeStreamResponse:
        resp = _FakeStreamResponse(status_code=200, lines=_happy_stream())
        resp.headers = {
            "x-codex-primary-used-percent": "37",
            "x-codex-primary-window-minutes": "10080",
            "x-codex-primary-reset-at": "2000000000",
        }
        return resp

    with mock.patch("httpx2.stream", side_effect=stream):
        provider.call(system="s", messages=[{"role": "user", "content": "x"}])
    assert provider.budget is not None
    snap = provider.budget.snapshot()
    assert snap.plan_latest is not None
    assert snap.plan_latest.used_percent == 37.0
    assert snap.plan_latest.window_minutes == 10080
    assert snap.unmetered_tokens == 0
    assert "plan usage: 37% of the 7-day window" in provider.budget.format_summary()


def test_completed_stream_without_message_item_keeps_delta_text(
    signed_in: ChatGPTCredential,
) -> None:
    """A backend that streamed text deltas but closed with no final message
    item still yields the watched text, not an empty turn, and the turn's
    other blocks stay in history behind it."""
    lines: list[str] = []
    lines += _evt({"type": "response.output_text.delta", "delta": "half"})
    lines += _evt({"type": "response.output_text.delta", "delta": " answer"})
    lines += _evt(
        {
            "type": "response.output_item.done",
            "item": {"type": "function_call", "call_id": "c1", "name": "run", "arguments": "{}"},
        }
    )
    lines += _evt(
        {
            "type": "response.completed",
            "response": {"id": "r", "status": "completed", "usage": _USAGE},
        }
    )
    provider = _provider(signed_in)
    with mock.patch("httpx2.stream", side_effect=_serve(lines)):
        resp = provider.call(system="s", messages=[{"role": "user", "content": "x"}])
    assert resp.text == "half answer"
    assert [b["type"] for b in resp.raw["content"]] == ["text", "tool_use"]


def test_tool_calling_completed_turn_reports_tool_use(signed_in: ChatGPTCredential) -> None:
    """A completed response whose output holds function_call items says
    stop_reason tool_use (Anthropic-shape semantics; also what arms the
    loop's empty-tool-call contradiction detector for this wire)."""
    lines: list[str] = []
    lines += _evt(
        {
            "type": "response.output_item.done",
            "item": {"type": "function_call", "call_id": "c1", "name": "run", "arguments": "{}"},
        }
    )
    lines += _evt(
        {
            "type": "response.completed",
            "response": {"id": "r", "status": "completed", "usage": _USAGE},
        }
    )
    provider = _provider(signed_in)
    with mock.patch("httpx2.stream", side_effect=_serve(lines)):
        resp = provider.call(system="s", messages=[{"role": "user", "content": "x"}])
    assert resp.stop_reason == "tool_use" and len(resp.tool_uses) == 1


def test_plan_usage_parses_the_credits_family() -> None:
    """The x-codex credits headers ride every response; the parse feeds the
    paid-credit guard (has/unlimited booleans, balance string)."""
    from agent6.providers.chatgpt import _plan_usage_of  # pyright: ignore[reportPrivateUsage]

    plan = _plan_usage_of(
        {
            "x-codex-primary-used-percent": "100",
            "x-codex-primary-window-minutes": "10080",
            "x-codex-primary-reset-at": "2000000000",
            "x-codex-credits-has-credits": "true",
            "x-codex-credits-unlimited": "false",
            "x-codex-credits-balance": " $12.50 ",
        }
    )
    assert plan is not None
    assert plan.has_credits is True
    assert plan.credits_unlimited is False
    assert plan.credits_balance == "$12.50"
    bare = _plan_usage_of({"x-codex-primary-used-percent": "40"})
    assert bare is not None and bare.has_credits is False


def test_reasoning_items_are_captured_and_replayed_in_order() -> None:
    """With store=false the encrypted reasoning item is the model's own
    chain-of-thought state: the parse keeps the raw item opaque, in its
    wire position, inside the thinking block that displays its summary, and
    the next request replays it verbatim immediately before its
    function_call."""
    from agent6.providers.chatgpt import parse_output_items

    reasoning = {
        "type": "reasoning",
        "id": "rs_1",
        "encrypted_content": "OPAQUE",
        "summary": [{"type": "summary_text", "text": "plan"}],
    }
    call = {
        "type": "function_call",
        "call_id": "c1",
        "name": "read_file",
        "arguments": '{"path": "x"}',
    }
    got = parse_output_items([reasoning, call], usage={}, stop_reason="end_turn")
    blocks = got.raw["content"]
    assert [b["type"] for b in blocks] == ["thinking", "tool_use"]
    assert blocks[0]["thinking"] == "plan" and blocks[0]["chatgpt_reasoning"] == reasoning

    items = responses_input(
        [
            {"role": "assistant", "content": blocks},
            {
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": "c1", "content": "hi"}],
            },
        ]
    )
    assert [i["type"] for i in items] == ["reasoning", "function_call", "function_call_output"]
    assert items[0] == reasoning


def test_interleaved_items_persist_and_replay_in_wire_order() -> None:
    """A turn that reasons, comments, reasons again and calls a tool is
    persisted as one block per output item in wire order and replayed in
    that order: the commentary message never hoists ahead of the reasoning
    that produced it. A display-only thinking block (another provider's,
    or one whose item was stripped) replays nothing."""
    from agent6.providers.chatgpt import parse_output_items

    r1 = {"type": "reasoning", "id": "rs_1", "encrypted_content": "A", "summary": []}
    note = {
        "type": "message",
        "role": "assistant",
        "content": [{"type": "output_text", "text": "Looking at x."}],
    }
    r2 = {
        "type": "reasoning",
        "id": "rs_2",
        "encrypted_content": "B",
        "summary": [{"type": "summary_text", "text": "then read"}],
    }
    call = {"type": "function_call", "call_id": "c1", "name": "read_file", "arguments": "{}"}
    got = parse_output_items([r1, note, r2, call], usage={}, stop_reason="end_turn")
    blocks = got.raw["content"]
    assert [b["type"] for b in blocks] == ["thinking", "text", "thinking", "tool_use"]
    assert got.text == "Looking at x." and blocks[0]["thinking"] == ""
    assert blocks[2]["thinking"] == "then read"

    items = responses_input(
        [
            {
                "role": "assistant",
                "content": [{"type": "thinking", "thinking": "foreign"}, *blocks],
            },
            {
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": "c1", "content": "body"}],
            },
        ]
    )
    assert [i.get("id") or i["type"] for i in items] == [
        "rs_1",
        "message",
        "rs_2",
        "function_call",
        "function_call_output",
    ]

    # A final answer keeps its reasoning ahead of the message too.
    final = parse_output_items([r1, note], usage={}, stop_reason="end_turn")
    items = responses_input([{"role": "assistant", "content": final.raw["content"]}])
    assert [i.get("id") or i["type"] for i in items] == ["rs_1", "message"]


def test_orphaned_reasoning_is_dropped_with_its_call() -> None:
    """A reasoning item whose paired call is dropped (blank tool name from a
    cross-provider resume) must not replay alone: an orphan violates the
    paired-item rules and 400s the whole request."""
    item = {"type": "reasoning", "id": "rs_1"}
    blocks = [
        {"type": "thinking", "thinking": "", "chatgpt_reasoning": item},
        {"type": "tool_use", "id": "c1", "name": "", "input": {}},
    ]
    items = responses_input([{"role": "assistant", "content": blocks}])
    assert items == []
