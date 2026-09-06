# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Regression tests for provider bug fixes.

Covers five independently-reported bugs:

* Bug #1 - non-JSON 2xx body in the NON-streaming path is converted to a
  retryable ``ProviderError`` instead of leaking a ``json.JSONDecodeError``
  (both OpenAI and Anthropic).
* Bug #2 - the Anthropic SSE streaming path has an idle watchdog: a wedged
  upstream that only emits ``ping`` heartbeats is closed and surfaced as a
  retryable ``ProviderError`` (mirrors the existing OpenAI watchdog).
* Bug #3 - OpenAI-direct o-series/reasoning models receive
  ``max_completion_tokens`` (not ``max_tokens``) and no explicit
  ``temperature``; other hosts are unchanged.
* Bug #4 - native tool_calls that arrive with no id get a synthesised
  distinct id so tool_use/tool_result pairing stays one-to-one.
* Bug #5 - budgeted calls fail closed when the upstream omits token usage
  accounting instead of recording a zero-token turn.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Iterator
from typing import Any, ClassVar
from unittest import mock

import httpx2
import pytest

from agent6.budget import BudgetTracker
from agent6.providers import _stream as stream_mod
from agent6.providers._openai_parse import parse_response as _parse_response
from agent6.providers.anthropic import AnthropicProvider, ProviderError
from agent6.providers.openai import OpenAIProvider


# --------------------------------------------------------------------------
# Bug #1: non-JSON 2xx body -> retryable ProviderError (non-streaming)
# --------------------------------------------------------------------------
class _FakeJSONResponse:
    def __init__(self, *, status_code: int, text: str) -> None:
        self.status_code = status_code
        self.headers: dict[str, str] = {}
        self.text = text

    def json(self) -> Any:
        return json.loads(self.text)  # raises on non-JSON


def test_openai_non_json_200_is_provider_error() -> None:
    provider = OpenAIProvider(api_key="sk-test", model="gpt-4o-mini")
    resp = _FakeJSONResponse(status_code=200, text="<html>502 Bad Gateway</html>")
    with (
        mock.patch("agent6.providers._transport.http_post", return_value=resp),
        pytest.raises(ProviderError) as ei,
    ):
        provider.call(system="sys", messages=[{"role": "user", "content": "x"}])
    # Retryable: status_code stays unset (None), not a non-retryable 4xx.
    assert ei.value.status_code is None
    assert "non-JSON" in str(ei.value)


def test_anthropic_non_json_200_is_provider_error() -> None:
    provider = AnthropicProvider(api_key="sk-test", model="claude-3-5-sonnet")
    resp = _FakeJSONResponse(status_code=200, text="<html>502 Bad Gateway</html>")
    with (
        mock.patch("agent6.providers._transport.http_post", return_value=resp),
        pytest.raises(ProviderError) as ei,
    ):
        provider.call(system="sys", messages=[{"role": "user", "content": "x"}])
    assert ei.value.status_code is None
    assert "non-JSON" in str(ei.value)


def test_openai_2xx_envelope_permanent_status_is_not_retried() -> None:
    """A 402 (insufficient credits) / 400 / 401 / 404 in a 2xx error envelope
    carries a PERMANENT upstream status in error.code. My first envelope fix
    always left status_code=None (retryable), re-creating the exact
    402-retried-every-turn regression `ProviderCaller` documents. The upstream
    code now becomes the status, so NON_RETRYABLE classifies it permanent, and
    the hint (HTTP 402) survives."""
    from agent6.workflows._provider_call import NON_RETRYABLE_HTTP_STATUSES

    budget = BudgetTracker(max_usd=-1, max_tokens_fallback=1, max_percent=-1)
    provider = OpenAIProvider(api_key="sk-test", model="gpt-4o-mini", budget=budget)
    resp = _FakeJSONResponse(
        status_code=200,
        text=json.dumps({"error": {"code": 402, "message": "Insufficient credits"}}),
    )
    with (
        mock.patch("agent6.providers._transport.http_post", return_value=resp),
        pytest.raises(ProviderError) as ei,
    ):
        provider.call(system="sys", messages=[{"role": "user", "content": "x"}])
    assert ei.value.status_code == 402
    assert ei.value.status_code in NON_RETRYABLE_HTTP_STATUSES  # permanent, not retried
    assert "402" in str(ei.value) and "Insufficient credits" in str(ei.value)


def test_openai_2xx_envelope_transient_status_stays_retryable() -> None:
    # A 429/5xx envelope keeps a retryable classification (not in the set).
    from agent6.workflows._provider_call import NON_RETRYABLE_HTTP_STATUSES

    budget = BudgetTracker(max_usd=-1, max_tokens_fallback=1, max_percent=-1)
    provider = OpenAIProvider(api_key="sk-test", model="gpt-4o-mini", budget=budget)
    resp = _FakeJSONResponse(
        status_code=200, text=json.dumps({"error": {"code": 503, "message": "Overloaded"}})
    )
    with (
        mock.patch("agent6.providers._transport.http_post", return_value=resp),
        pytest.raises(ProviderError) as ei,
    ):
        provider.call(system="sys", messages=[{"role": "user", "content": "x"}])
    # Pin the CARRIED status, not just "not in the set" -- a dropped (None) status
    # also satisfies `not in`, so the weaker assertion passed on the pre-fix code.
    assert ei.value.status_code == 503
    assert ei.value.status_code not in NON_RETRYABLE_HTTP_STATUSES  # retryable


def test_envelope_status_classifies_permanent_string_codes() -> None:
    """String error codes/types (OpenAI `code`, Anthropic `type`) that are
    permanent map to their terminal HTTP status so a budgeted run fails fast;
    transient strings and numerics behave as before. The numeric-only map left
    every string code None (retryable), retrying a quota/auth error every turn."""
    from agent6.providers._transport import envelope_status
    from agent6.workflows._provider_call import NON_RETRYABLE_HTTP_STATUSES

    permanent = [
        ({"code": "insufficient_quota"}, 402),
        ({"code": "invalid_api_key"}, 401),
        ({"code": "model_not_found"}, 404),
        ({"type": "authentication_error"}, 401),
        ({"type": "permission_error"}, 403),
        ({"type": "not_found_error"}, 404),
        ({"type": "invalid_request_error"}, 400),
    ]
    for err, status in permanent:
        assert envelope_status(err) == status, err
        assert status in NON_RETRYABLE_HTTP_STATUSES
    # Transient string codes/types stay retryable (None), never guessed permanent.
    for err in (
        {"code": "rate_limit_exceeded"},
        {"type": "overloaded_error"},
        {"type": "api_error"},
    ):
        assert envelope_status(err) is None, err
    # Numeric codes and the empty/non-dict cases are unchanged.
    assert envelope_status({"code": 402}) == 402
    assert envelope_status({"code": 502}) == 502
    assert envelope_status({"code": 200}) is None  # a 2xx code is not an error status
    assert envelope_status({}) is None and envelope_status("nope") is None


def test_openai_2xx_string_error_and_placeholder_choices_are_envelopes() -> None:
    """A string `error`, or an error object beside a null-content placeholder
    `choices` entry, must be the envelope -- not fall through to the misleading
    metering 422. The guard tested key presence and required error to be a dict."""
    budget = BudgetTracker(max_usd=-1, max_tokens_fallback=1, max_percent=-1)
    for body in (
        {"error": "model not found"},
        {
            "error": {"code": 400, "message": "bad"},
            "choices": [{"message": {"content": None}, "finish_reason": "error"}],
        },
    ):
        provider = OpenAIProvider(api_key="sk-test", model="gpt-4o-mini", budget=budget)
        resp = _FakeJSONResponse(status_code=200, text=json.dumps(body))
        with (
            mock.patch("agent6.providers._transport.http_post", return_value=resp),
            pytest.raises(ProviderError) as ei,
        ):
            provider.call(system="sys", messages=[{"role": "user", "content": "x"}])
        assert "usage" not in str(ei.value), f"misattributed metering 422 for {body}"


def test_openai_error_key_beside_real_content_still_parses() -> None:
    # An `error` key beside a REAL assistant message is incidental, not an
    # envelope: the response must parse normally.
    budget = BudgetTracker(max_usd=-1, max_tokens_fallback=1, max_percent=-1)
    provider = OpenAIProvider(api_key="sk-test", model="gpt-4o-mini", budget=budget)
    resp = _FakeJSONResponse(
        status_code=200,
        text=json.dumps(
            {
                "error": None,
                "choices": [{"message": {"content": "hi"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 5, "completion_tokens": 1},
            }
        ),
    )
    with mock.patch("agent6.providers._transport.http_post", return_value=resp):
        out = provider.call(system="sys", messages=[{"role": "user", "content": "x"}])
    assert out.text == "hi"


def test_openai_2xx_error_envelope_is_the_upstreams_failure() -> None:
    """OpenRouter/LiteLLM deliver an upstream 5xx/429 as HTTP 200 with a
    top-level `error` object. The body has no `usage`, so the metering gate
    blamed agent6's own accounting ("no usage input tokens", 422 = permanent)
    and a transient upstream failure killed the run -- and every compaction
    side-call -- with no retry. The streaming paths already surface the
    envelope; the non-streaming path must match: upstream code/message, and a
    502/429 stays retryable (not in NON_RETRYABLE_HTTP_STATUSES)."""
    from agent6.workflows._provider_call import NON_RETRYABLE_HTTP_STATUSES

    budget = BudgetTracker(max_usd=-1, max_tokens_fallback=1, max_percent=-1)
    provider = OpenAIProvider(api_key="sk-test", model="gpt-4o-mini", budget=budget)
    resp = _FakeJSONResponse(
        status_code=200,
        text=json.dumps({"error": {"code": 502, "message": "Provider returned error"}}),
    )
    with (
        mock.patch("agent6.providers._transport.http_post", return_value=resp),
        pytest.raises(ProviderError) as ei,
    ):
        provider.call(system="sys", messages=[{"role": "user", "content": "x"}])
    assert ei.value.status_code == 502  # carried, not dropped to None
    assert ei.value.status_code not in NON_RETRYABLE_HTTP_STATUSES  # 502 -> retryable
    assert "502" in str(ei.value) and "Provider returned error" in str(ei.value)
    assert "usage" not in str(ei.value)  # names the upstream, not the accounting


def test_anthropic_2xx_error_envelope_is_the_upstreams_failure() -> None:
    budget = BudgetTracker(max_usd=-1, max_tokens_fallback=1, max_percent=-1)
    provider = AnthropicProvider(api_key="sk-test", model="claude-3-5-sonnet", budget=budget)
    resp = _FakeJSONResponse(
        status_code=200,
        text=json.dumps(
            {"type": "error", "error": {"type": "overloaded_error", "message": "Overloaded"}}
        ),
    )
    with (
        mock.patch("agent6.providers._transport.http_post", return_value=resp),
        pytest.raises(ProviderError) as ei,
    ):
        provider.call(system="sys", messages=[{"role": "user", "content": "x"}])
    assert ei.value.status_code is None
    assert "overloaded_error" in str(ei.value) and "Overloaded" in str(ei.value)


def test_openai_budgeted_response_requires_usage_tokens() -> None:
    budget = BudgetTracker(max_usd=-1, max_tokens_fallback=1, max_percent=-1)
    provider = OpenAIProvider(api_key="sk-test", model="gpt-4o-mini", budget=budget)
    resp = _FakeJSONResponse(
        status_code=200,
        text=json.dumps(
            {
                "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
                "usage": {},
            }
        ),
    )
    with (
        mock.patch("agent6.providers._transport.http_post", return_value=resp),
        pytest.raises(ProviderError) as ei,
    ):
        provider.call(system="sys", messages=[{"role": "user", "content": "x"}])
    # Retryable (no status): a usage-less reply is stream/gateway integrity
    # failure; the loop's bounded retry lane owns it, and a fake 422 here
    # made ONE mangled stream kill a budgeted run.
    assert ei.value.status_code is None
    assert "usage accounting" in str(ei.value)
    assert budget.snapshot().per_model == {}


def test_anthropic_budgeted_response_requires_usage_tokens() -> None:
    budget = BudgetTracker(max_usd=-1, max_tokens_fallback=1, max_percent=-1)
    provider = AnthropicProvider(api_key="sk-test", model="claude-3-5-sonnet", budget=budget)
    resp = _FakeJSONResponse(
        status_code=200,
        text=json.dumps(
            {
                "content": [{"type": "text", "text": "ok"}],
                "stop_reason": "end_turn",
                "usage": {},
            }
        ),
    )
    with (
        mock.patch("agent6.providers._transport.http_post", return_value=resp),
        pytest.raises(ProviderError) as ei,
    ):
        provider.call(system="sys", messages=[{"role": "user", "content": "x"}])
    # Retryable (no status): a usage-less reply is stream/gateway integrity
    # failure; the loop's bounded retry lane owns it, and a fake 422 here
    # made ONE mangled stream kill a budgeted run.
    assert ei.value.status_code is None
    assert "usage accounting" in str(ei.value)
    assert budget.snapshot().per_model == {}


def test_openai_budgeted_response_rejects_zero_token_usage() -> None:
    # Presence is not enough: a gateway with usage tracking off returns 0/0, and
    # every turn would record zero so the budget never trips. Fail closed.
    budget = BudgetTracker(max_usd=-1, max_tokens_fallback=1, max_percent=-1)
    provider = OpenAIProvider(api_key="sk-test", model="gpt-4o-mini", budget=budget)
    resp = _FakeJSONResponse(
        status_code=200,
        text=json.dumps(
            {
                "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 0, "completion_tokens": 0},
            }
        ),
    )
    with (
        mock.patch("agent6.providers._transport.http_post", return_value=resp),
        pytest.raises(ProviderError) as ei,
    ):
        provider.call(system="sys", messages=[{"role": "user", "content": "x"}])
    # Retryable (no status): a usage-less reply is stream/gateway integrity
    # failure; the loop's bounded retry lane owns it, and a fake 422 here
    # made ONE mangled stream kill a budgeted run.
    assert ei.value.status_code is None
    assert "usage accounting" in str(ei.value)
    assert budget.snapshot().per_model == {}


def test_anthropic_budgeted_response_rejects_zero_token_usage() -> None:
    budget = BudgetTracker(max_usd=-1, max_tokens_fallback=1, max_percent=-1)
    provider = AnthropicProvider(api_key="sk-test", model="claude-3-5-sonnet", budget=budget)
    resp = _FakeJSONResponse(
        status_code=200,
        text=json.dumps(
            {
                "content": [{"type": "text", "text": "ok"}],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 0, "output_tokens": 0},
            }
        ),
    )
    with (
        mock.patch("agent6.providers._transport.http_post", return_value=resp),
        pytest.raises(ProviderError) as ei,
    ):
        provider.call(system="sys", messages=[{"role": "user", "content": "x"}])
    # Retryable (no status): a usage-less reply is stream/gateway integrity
    # failure; the loop's bounded retry lane owns it, and a fake 422 here
    # made ONE mangled stream kill a budgeted run.
    assert ei.value.status_code is None
    assert "usage accounting" in str(ei.value)
    assert budget.snapshot().per_model == {}


def test_anthropic_budgeted_response_accepts_fully_cached_turn() -> None:
    # A fully-cached turn legitimately reports input_tokens: 0 with a positive
    # cache_read count; the metering check must NOT false-reject it.
    budget = BudgetTracker(max_usd=-1, max_tokens_fallback=-1, max_percent=-1)
    provider = AnthropicProvider(api_key="sk-test", model="claude-3-5-sonnet", budget=budget)
    resp = _FakeJSONResponse(
        status_code=200,
        text=json.dumps(
            {
                "content": [{"type": "text", "text": "ok"}],
                "stop_reason": "end_turn",
                "usage": {
                    "input_tokens": 0,
                    "output_tokens": 5,
                    "cache_read_input_tokens": 120,
                    "cache_creation_input_tokens": 0,
                },
            }
        ),
    )
    with mock.patch("agent6.providers._transport.http_post", return_value=resp):
        provider.call(system="sys", messages=[{"role": "user", "content": "x"}])
    assert budget.snapshot().per_model != {}


# --------------------------------------------------------------------------
# Bug #2: Anthropic SSE idle watchdog
# --------------------------------------------------------------------------
class _PingOnlyStreamResponse:
    """A stream that only ever emits ``ping`` heartbeats.

    ``iter_lines`` blocks (via an event) until the watchdog calls
    ``close()``, at which point it raises ``httpx2.ReadError`` exactly as
    httpx2 would when the underlying socket is closed mid-read.
    """

    def __init__(self) -> None:
        self.status_code = 200
        self.headers: dict[str, str] = {}
        self._closed = threading.Event()

    def __enter__(self) -> _PingOnlyStreamResponse:
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def close(self) -> None:
        self._closed.set()

    def iter_lines(self):  # type: ignore[no-untyped-def]
        # Emit a couple of ping heartbeats, then block until closed.
        yield "event: ping"
        yield 'data: {"type": "ping"}'
        yield ""
        yield "event: ping"
        yield 'data: {"type": "ping"}'
        yield ""
        # Now park as if waiting for real data. The watchdog must fire.
        if not self._closed.wait(timeout=10.0):
            raise AssertionError("watchdog never closed the response")
        raise httpx2.ReadError("connection closed by watchdog")


def test_anthropic_streaming_idle_watchdog_fires(monkeypatch: pytest.MonkeyPatch) -> None:
    # Ping-only, no data event ever = the prefill-wedge case, so the FIRST-data
    # timeout governs. Make it fire fast so the test runs in well under a second.
    monkeypatch.setattr(stream_mod, "STREAM_FIRST_DATA_TIMEOUT_S", 0.05)
    monkeypatch.setattr(stream_mod, "STREAM_WATCHDOG_TICK_S", 0.01)

    provider = AnthropicProvider(api_key="sk-test", model="claude-3-5-sonnet")

    def fake_stream(method: str, url: str, **kwargs: Any) -> _PingOnlyStreamResponse:
        return _PingOnlyStreamResponse()

    with (
        mock.patch("httpx2.stream", side_effect=fake_stream),
        pytest.raises(ProviderError) as ei,
    ):
        provider.call(
            system="sys",
            messages=[{"role": "user", "content": "x"}],
            text_delta_callback=lambda _s: None,
        )
    # Surfaced as a retryable (status_code None) idle error, not a generic
    # ReadError leaking out of the loop.
    assert ei.value.status_code is None
    assert "idle" in str(ei.value).lower()


# --------------------------------------------------------------------------
# Bug #3: OpenAI-direct o-series uses max_completion_tokens, drops temperature
# --------------------------------------------------------------------------
def _capture_body(provider: OpenAIProvider) -> dict[str, Any]:
    captured: dict[str, Any] = {}

    class _Resp:
        status_code = 200
        text = "{}"

        def json(self) -> Any:
            return {
                "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            }

    def fake_post(url: str, **kwargs: Any) -> _Resp:
        captured.update(json.loads(kwargs["content"]))
        return _Resp()

    with mock.patch("agent6.providers._transport.http_post", side_effect=fake_post):
        provider.call(
            system="sys",
            messages=[{"role": "user", "content": "x"}],
            temperature=0.2,
        )
    return captured


def test_openai_direct_oseries_uses_max_completion_tokens() -> None:
    provider = OpenAIProvider(
        api_key="sk-test",
        model="o3-mini",
        base_url="https://api.openai.com/v1",
        deployment="direct",
    )
    body = _capture_body(provider)
    assert "max_tokens" not in body
    assert "max_completion_tokens" in body
    # Temperature is dropped for o-series direct.
    assert "temperature" not in body


def test_openrouter_oseries_still_uses_max_tokens() -> None:
    # Same reasoning model, but routed via OpenRouter: must keep max_tokens
    # (OpenRouter normalises it) and forward temperature.
    provider = OpenAIProvider(
        api_key="sk-test",
        model="o3-mini",
        base_url="https://openrouter.ai/api/v1",
        deployment="direct",
    )
    body = _capture_body(provider)
    assert "max_tokens" in body
    assert "max_completion_tokens" not in body
    assert body.get("temperature") == 0.2


def test_openai_direct_nonreasoning_keeps_max_tokens() -> None:
    provider = OpenAIProvider(
        api_key="sk-test",
        model="gpt-4o-mini",
        base_url="https://api.openai.com/v1",
        deployment="direct",
    )
    body = _capture_body(provider)
    assert "max_tokens" in body
    assert "max_completion_tokens" not in body
    assert body.get("temperature") == 0.2


# --------------------------------------------------------------------------
# Bug #4: synthesise distinct ids for native tool_calls missing an id
# --------------------------------------------------------------------------
def test_parse_response_synthesises_distinct_tool_call_ids() -> None:
    data = {
        "choices": [
            {
                "message": {
                    "content": "",
                    "tool_calls": [
                        {"function": {"name": "list_dir", "arguments": "{}"}},
                        {"function": {"name": "read_file", "arguments": "{}"}},
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1},
    }
    parsed = _parse_response(data)
    ids = [tu["id"] for tu in parsed.tool_uses]
    assert all(i for i in ids), "every tool_use must have a non-empty id"
    assert len(set(ids)) == len(ids), "ids must be distinct"


def test_parse_response_preserves_provided_tool_call_ids() -> None:
    data = {
        "choices": [
            {
                "message": {
                    "content": "",
                    "tool_calls": [
                        {"id": "call_real", "function": {"name": "list_dir", "arguments": "{}"}},
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1},
    }
    parsed = _parse_response(data)
    assert parsed.tool_uses[0]["id"] == "call_real"


# --------------------------------------------------------------------------
# claude-opus-4-8 rejects `temperature` (400) -> drop it and retry, then latch
# --------------------------------------------------------------------------
def test_anthropic_temperature_400_retries_without_temperature_then_latches() -> None:
    provider = AnthropicProvider(api_key="sk-test", model="claude-opus-4-8")
    err400 = _FakeJSONResponse(
        status_code=400,
        text=json.dumps(
            {
                "type": "error",
                "error": {
                    "type": "invalid_request_error",
                    "message": "`temperature` is deprecated for this model.",
                },
            }
        ),
    )
    ok200 = _FakeJSONResponse(
        status_code=200,
        text=json.dumps(
            {
                "id": "msg_1",
                "type": "message",
                "role": "assistant",
                "model": "claude-opus-4-8",
                "content": [{"type": "text", "text": "hello"}],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 10, "output_tokens": 5},
            }
        ),
    )

    bodies: list[dict[str, Any]] = []

    def first_call(*_a: object, **k: object) -> _FakeJSONResponse:
        bodies.append(json.loads(k["content"]))  # type: ignore[arg-type]
        return err400 if len(bodies) == 1 else ok200

    with mock.patch("agent6.providers._transport.http_post", side_effect=first_call):
        resp = provider.call(
            system="sys", messages=[{"role": "user", "content": "x"}], temperature=0.0
        )
    # First request carried temperature; the retry dropped it and succeeded.
    assert "temperature" in bodies[0]
    assert "temperature" not in bodies[1]
    assert resp.text == "hello"

    # The flag latched: a later call omits temperature from the very first request
    # (no wasted 400 + full-context resend every iteration).
    bodies2: list[dict[str, Any]] = []

    def second_call(*_a: object, **k: object) -> _FakeJSONResponse:
        bodies2.append(json.loads(k["content"]))  # type: ignore[arg-type]
        return ok200

    with mock.patch("agent6.providers._transport.http_post", side_effect=second_call):
        provider.call(system="sys", messages=[{"role": "user", "content": "y"}], temperature=0.0)
    assert "temperature" not in bodies2[0]


# --------------------------------------------------------------------------
# Connection errors name the dialled URL + api format (a bare "HTTP error
# calling OpenAI" pointed users at the wrong party for local endpoints).
# --------------------------------------------------------------------------
def test_openai_connection_error_names_url_and_format() -> None:
    provider = OpenAIProvider(api_key="", model="llama3", base_url="http://localhost:11434/v1")
    with (
        mock.patch(
            "agent6.providers._transport.http_post",
            side_effect=httpx2.HTTPError("[Errno 111] Connection refused"),
        ),
        pytest.raises(ProviderError) as ei,
    ):
        provider.call(system="sys", messages=[{"role": "user", "content": "x"}])
    msg = str(ei.value)
    assert "http://localhost:11434/v1/chat/completions" in msg
    assert "openai format" in msg
    assert "Connection refused" in msg


def test_anthropic_connection_error_names_url_and_format() -> None:
    provider = AnthropicProvider(api_key="sk-test", model="claude-3-5-sonnet")
    with (
        mock.patch(
            "agent6.providers._transport.http_post",
            side_effect=httpx2.HTTPError("[Errno 111] Connection refused"),
        ),
        pytest.raises(ProviderError) as ei,
    ):
        provider.call(system="sys", messages=[{"role": "user", "content": "x"}])
    msg = str(ei.value)
    assert "api.anthropic.com" in msg
    assert "anthropic format" in msg


def test_non_object_json_200_is_retryable_provider_error() -> None:
    """Valid JSON that is not an object (an array from a glitching gateway)
    would AttributeError past the ProviderError-only retry; it must convert
    to a retryable ProviderError like the non-JSON case."""
    provider = OpenAIProvider(api_key="sk-test", model="gpt-4o-mini")
    resp = _FakeJSONResponse(status_code=200, text='["not", "an", "object"]')
    with (
        mock.patch("agent6.providers._transport.http_post", return_value=resp),
        pytest.raises(ProviderError) as ei,
    ):
        provider.call(system="sys", messages=[{"role": "user", "content": "x"}])
    assert ei.value.status_code is None
    assert "non-object" in str(ei.value)


def test_openai_malformed_choices_entry_is_provider_error() -> None:
    """choices[0] null/string (a flaky local endpoint) must raise a retryable
    ProviderError, not an AttributeError that kills the run."""
    for body in ('{"choices": [null]}', '{"choices": ["err"]}'):
        with pytest.raises(ProviderError) as ei:
            _parse_response(json.loads(body))
        assert ei.value.status_code is None


def test_anthropic_malformed_content_is_provider_error() -> None:
    """content as a bare string, or a list holding a non-dict element, must
    raise a retryable ProviderError, not iterate characters into an
    AttributeError."""
    from agent6.providers.anthropic import (
        _parse_response as _anthropic_parse,  # pyright: ignore[reportPrivateUsage]
    )

    for body in (
        {"content": "hello", "usage": {"input_tokens": 5, "output_tokens": 3}},
        {"content": [{"type": "text", "text": "hi"}, "oops"], "usage": {}},
    ):
        with pytest.raises(ProviderError) as ei:
            _anthropic_parse(body)
        assert ei.value.status_code is None


def test_metered_gate_coerces_gateway_typed_counts() -> None:
    """A gateway serializing counts as floats/strings ("700", 700.0) is
    meterable -- parse_response coerces them -- but the isinstance(int) gate
    refused it, killing a budgeted run on its first call. Absent/zero/
    non-numeric still fails closed, as a RETRYABLE refusal (see
    test_*_requires_usage_tokens)."""
    from agent6.providers.anthropic import (
        _require_metered_usage as _anthropic_gate,  # pyright: ignore[reportPrivateUsage]
    )
    from agent6.providers.openai import (
        _require_metered_usage as _openai_gate,  # pyright: ignore[reportPrivateUsage]
    )

    _openai_gate({"prompt_tokens": 700.0, "completion_tokens": 50}, source="t")
    _openai_gate({"prompt_tokens": "700"}, source="t")  # missing completion is fine
    _anthropic_gate({"input_tokens": 700.0, "output_tokens": 3}, source="t")
    for bad in ({"prompt_tokens": 0}, {"prompt_tokens": "abc"}, {}):
        with pytest.raises(ProviderError) as ei:
            _openai_gate(bad, source="t")
        assert ei.value.status_code is None


def test_boolean_reported_cost_reads_as_absent() -> None:
    """bool subclasses int: usage.cost == true yielded float(True) == a
    phantom $1.00 recorded per call, which becomes the AUTHORITATIVE reported
    figure and can trip the max_usd hard stop."""
    resp = _parse_response(
        {
            "choices": [{"message": {"role": "assistant", "content": "hi"}}],
            "usage": {"prompt_tokens": 500, "completion_tokens": 100, "cost": True},
        }
    )
    assert resp.cost_usd == 0.0


def test_credential_refresh_roundtrip_is_recorded(tmp_path: Any) -> None:
    """The 401/403 that triggers a token refresh hit the wire; the transcript
    contract is one file per round-trip (the streaming path records it; the
    non-streaming refresh branch silently dropped it)."""
    from pathlib import Path

    from agent6.providers import TranscriptSink
    from agent6.providers.token_command import CommandToken

    sink = TranscriptSink(Path(tmp_path) / "transcripts")
    calls = {"n": 0}

    def _post(url: str, *, headers: Any, content: Any, timeout: Any) -> Any:
        calls["n"] += 1
        if calls["n"] == 1:
            return _FakeJSONResponse(status_code=401, text='{"error": "expired"}')
        return _FakeJSONResponse(
            status_code=200,
            text=json.dumps(
                {
                    "choices": [{"message": {"role": "assistant", "content": "hi"}}],
                    "usage": {"prompt_tokens": 5, "completion_tokens": 2},
                }
            ),
        )

    cred = CommandToken(["echo", "tok"])
    provider = OpenAIProvider(
        api_key="", model="gpt-4o-mini", transcript_sink=sink, credential=cred
    )
    with mock.patch("agent6.providers._transport.http_post", side_effect=_post):
        provider.call(system="sys", messages=[{"role": "user", "content": "x"}])
    statuses = sorted(
        json.loads(p.read_text(encoding="utf-8"))["response"]["status"]
        for p in (Path(tmp_path) / "transcripts").glob("*.json")
    )
    assert statuses == [200, 401]


def test_connect_phase_is_bounded_below_the_read_budget() -> None:
    """A blackholed connect must fail in seconds: the stream watchdog has no
    response to close until the connect returns, so with a single-float
    timeout the 600s read default sat on a dropped SYN for ten minutes
    (caught live: verify inference wedged an ACP run)."""
    from agent6.providers._transport import CONNECT_TIMEOUT_S, granular_timeout

    t = granular_timeout(600.0)
    assert t.connect == CONNECT_TIMEOUT_S
    assert t.read == 600.0
    assert t.write == 600.0
    # A budget tighter than the connect bound wins: the operator asked for it.
    assert granular_timeout(5.0).connect == 5.0


def test_both_http_seams_pass_the_granular_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    import contextlib

    import httpx2

    from agent6.providers import _stream, _transport

    seen: list[object] = []

    @contextlib.contextmanager
    def fake_stream(method: str, url: str, **kw: object):
        seen.append(kw["timeout"])
        raise httpx2.ConnectError("stop")
        yield  # pragma: no cover

    monkeypatch.setattr(_transport.httpx2, "stream", fake_stream)
    monkeypatch.setattr(_stream.httpx2, "stream", fake_stream)
    with contextlib.suppress(httpx2.HTTPError):
        _transport.http_post("https://x", headers={}, content=b"", timeout=600.0)
    with (
        contextlib.suppress(httpx2.HTTPError),
        _stream.http_stream("POST", "https://x", headers={}, content=b"", timeout=600.0),
    ):
        pass  # pragma: no cover -- the stub raises before yielding
    assert [type(t) for t in seen] == [httpx2.Timeout, httpx2.Timeout]
    assert all(t.connect == _transport.CONNECT_TIMEOUT_S for t in seen)  # type: ignore[union-attr]


def test_an_oversized_provider_response_is_refused_not_buffered(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The non-streaming seam buffered whatever arrived; a multi-MiB (or
    hostile, unbounded) body was materialized whole while fetch and MCP bound
    their reads. The body is read under MAX_RESPONSE_BYTES; exceeding it is a
    retryable ProviderError, and a normal body still round-trips."""
    import contextlib

    from agent6.providers import _transport
    from agent6.providers.types import ProviderError

    class _FakeStreamResp:
        status_code = 200
        headers: ClassVar[dict[str, str]] = {}
        request = httpx2.Request("POST", "https://x")

        def __init__(self, chunks: list[bytes]) -> None:
            self._chunks = chunks

        def iter_bytes(self) -> Iterator[bytes]:
            yield from self._chunks

    def _serving(chunks: list[bytes]):
        @contextlib.contextmanager
        def fake_stream(method: str, url: str, **kw: object):
            yield _FakeStreamResp(chunks)

        return fake_stream

    monkeypatch.setattr(_transport, "MAX_RESPONSE_BYTES", 1024)
    monkeypatch.setattr(_transport.httpx2, "stream", _serving([b"x" * 600, b"y" * 600]))
    with pytest.raises(ProviderError, match="exceeded"):
        _transport.http_post("https://x", headers={}, content=b"", timeout=5.0)

    monkeypatch.setattr(_transport.httpx2, "stream", _serving([b'{"ok": true}']))
    resp = _transport.http_post("https://x", headers={}, content=b"", timeout=5.0)
    assert resp.status_code == 200
    assert resp.json() == {"ok": True}


def test_a_gzip_encoded_provider_response_is_not_decoded_twice(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`iter_bytes()` yields the DECODED body, but the rebuilt Response
    carried the wire's `content-encoding: gzip` header, so httpx2 ran the
    decoder again over plaintext and every gzip-encoded provider response
    (Anthropic always, OpenRouter past its size threshold) failed with
    `DecodingError: incorrect header check`. The representation headers are
    dropped on rebuild; the rest (retry-after here) survive."""
    import contextlib

    from agent6.providers import _transport

    class _FakeStreamResp:
        status_code = 200
        headers = httpx2.Headers(
            {"content-encoding": "gzip", "content-length": "57", "retry-after": "7"}
        )
        request = httpx2.Request("POST", "https://x")

        def iter_bytes(self) -> Iterator[bytes]:
            yield b'{"ok": true}'

    @contextlib.contextmanager
    def fake_stream(method: str, url: str, **kw: object):
        yield _FakeStreamResp()

    monkeypatch.setattr(_transport.httpx2, "stream", fake_stream)
    resp = _transport.http_post("https://x", headers={}, content=b"", timeout=5.0)
    assert resp.json() == {"ok": True}
    assert "content-encoding" not in resp.headers
    assert resp.headers.get("retry-after") == "7"
