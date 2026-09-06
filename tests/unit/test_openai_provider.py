# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Tests for `agent6.providers.openai.OpenAIProvider`."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest import mock

import httpx2
import pytest

from agent6.providers import ProviderError
from agent6.providers.openai import OpenAIProvider
from agent6.providers.token_command import CommandToken


def _fake_response(body: dict[str, Any], status: int = 200) -> httpx2.Response:
    return httpx2.Response(
        status_code=status,
        request=httpx2.Request("POST", "https://api.openai.com/v1/chat/completions"),
        content=json.dumps(body).encode("utf-8"),
        headers={"content-type": "application/json"},
    )


def test_call_translates_messages_and_parses_usage() -> None:
    provider = OpenAIProvider(api_key="sk-test", model="gpt-x")
    captured: dict[str, Any] = {}

    def fake_post(*_a: Any, **kw: Any) -> httpx2.Response:
        captured["headers"] = kw["headers"]
        captured["body"] = json.loads(kw["content"])
        return _fake_response(
            {
                "choices": [
                    {
                        "message": {"content": "hello"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 100,
                    "completion_tokens": 25,
                    "prompt_tokens_details": {"cached_tokens": 40},
                },
            }
        )

    with mock.patch("agent6.providers._transport.http_post", side_effect=fake_post):
        resp = provider.call(
            system="you are a reviewer",
            messages=[{"role": "user", "content": "judge this"}],
        )

    assert captured["headers"]["authorization"] == "Bearer sk-test"
    assert captured["body"]["model"] == "gpt-x"
    assert captured["body"]["messages"][0] == {"role": "system", "content": "you are a reviewer"}
    assert captured["body"]["messages"][1] == {"role": "user", "content": "judge this"}
    assert resp.text == "hello"
    assert resp.stop_reason == "stop"
    # cache-token normalisation: OpenAI reports `prompt_tokens` as the
    # TOTAL prompt size including cached portion. We normalise to Anthropic's
    # semantics where `input_tokens` is fresh (non-cached) only, with the
    # cached portion surfaced under `cache_read_tokens`. So a usage block with
    # prompt_tokens=100, cached_tokens=40 yields input_tokens=60 (fresh).
    assert resp.input_tokens == 60
    assert resp.output_tokens == 25
    assert resp.cache_read_tokens == 40


def test_openai_direct_reasoning_uses_top_level_reasoning_effort() -> None:
    """api.openai.com o-series/gpt-5 take a TOP-LEVEL ``reasoning_effort``; the
    nested ``reasoning`` object (OpenRouter's convention) 400s there. A non-direct
    host keeps the nested object. (Found by GLM during dogfood, rewritten here.)"""
    captured: dict[str, Any] = {}

    def fake_post(*_a: Any, **kw: Any) -> httpx2.Response:
        captured["body"] = json.loads(kw["content"])
        return _fake_response(
            {"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}], "usage": {}}
        )

    # OpenAI-direct reasoning model (default base_url = api.openai.com).
    direct = OpenAIProvider(api_key="sk", model="o3-mini", reasoning_effort="medium")
    with mock.patch("agent6.providers._transport.http_post", side_effect=fake_post):
        direct.call(system="s", messages=[{"role": "user", "content": "hi"}])
    assert captured["body"].get("reasoning_effort") == "medium"
    assert "reasoning" not in captured["body"]

    # Non-direct host (OpenRouter): nested reasoning object, no top-level field.
    router = OpenAIProvider(
        api_key="sk",
        model="z-ai/glm-5.2",
        base_url="https://openrouter.ai/api/v1",
        reasoning_effort="high",
    )
    with mock.patch("agent6.providers._transport.http_post", side_effect=fake_post):
        router.call(system="s", messages=[{"role": "user", "content": "hi"}])
    assert captured["body"].get("reasoning") == {"effort": "high"}
    assert "reasoning_effort" not in captured["body"]


def test_openai_direct_gpt5_honors_reasoning_effort() -> None:
    """gpt-5 / bare o1 / o3 match _is_openai_direct_reasoning_model but NOT
    _is_reasoning_model, so a configured reasoning_effort used to be silently
    dropped (the reasoning block was gated on _is_reasoning_model alone). It must
    emit the top-level reasoning_effort like any other openai-direct reasoner."""
    from agent6.providers.openai import _is_reasoning_model  # pyright: ignore[reportPrivateUsage]

    assert _is_reasoning_model("gpt-5") is False  # the exact gap this closes
    captured: dict[str, Any] = {}

    def fake_post(*_a: Any, **kw: Any) -> httpx2.Response:
        captured["body"] = json.loads(kw["content"])
        return _fake_response(
            {"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}], "usage": {}}
        )

    direct = OpenAIProvider(api_key="sk", model="gpt-5", reasoning_effort="high")
    with mock.patch("agent6.providers._transport.http_post", side_effect=fake_post):
        direct.call(system="s", messages=[{"role": "user", "content": "hi"}])
    assert captured["body"].get("reasoning_effort") == "high"
    assert "reasoning" not in captured["body"]  # not the nested OpenRouter object
    assert "max_completion_tokens" in captured["body"]  # the direct rename still applies


def test_call_merges_extra_body() -> None:
    # extra_body (e.g. OpenRouter `provider` routing) is merged into the request
    # body, last, so an operator can pin a caching/fast backend.
    provider = OpenAIProvider(
        api_key="sk-test",
        model="kimi",
        extra_body={"provider": {"sort": "throughput"}},
    )
    captured: dict[str, Any] = {}

    def fake_post(*_a: Any, **kw: Any) -> httpx2.Response:
        captured["body"] = json.loads(kw["content"])
        return _fake_response(
            {"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}], "usage": {}}
        )

    with mock.patch("agent6.providers._transport.http_post", side_effect=fake_post):
        provider.call(system="s", messages=[{"role": "user", "content": "x"}])

    assert captured["body"]["provider"] == {"sort": "throughput"}


def test_extra_body_cannot_replace_the_structural_request_shape() -> None:
    """Tuning keys merge last and win (max_tokens); the structural set the
    loop depends on (tools, tool_choice, response_format, n) never does --
    replacing the tool schema silently changes the model's surface, and a
    response the parser cannot read as choices[0] breaks every call."""
    provider = OpenAIProvider(
        api_key="sk-test",
        model="kimi",
        extra_body={
            "max_tokens": 5,
            "tools": [{"type": "function", "function": {"name": "evil"}}],
            "tool_choice": "none",
            "response_format": {"type": "json_object"},
            "n": 3,
        },
    )
    captured: dict[str, Any] = {}

    def fake_post(*_a: Any, **kw: Any) -> httpx2.Response:
        captured["body"] = json.loads(kw["content"])
        return _fake_response(
            {"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}], "usage": {}}
        )

    with mock.patch("agent6.providers._transport.http_post", side_effect=fake_post):
        provider.call(system="s", messages=[{"role": "user", "content": "x"}])

    body = captured["body"]
    assert body["max_tokens"] == 5
    for key in ("tools", "tool_choice", "response_format", "n"):
        assert key not in body, f"extra_body must not inject {key}"


def test_call_clamps_negative_fresh_input_to_zero() -> None:
    """Defensive: a misbehaving upstream reporting cached > prompt must not
    produce a negative `input_tokens` (which would corrupt the BudgetTracker
    counters)."""
    provider = OpenAIProvider(api_key="sk", model="gpt-x")

    def fake_post(*_a: Any, **_kw: Any) -> httpx2.Response:
        return _fake_response(
            {
                "choices": [{"message": {"content": "x"}, "finish_reason": "stop"}],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 1,
                    "prompt_tokens_details": {"cached_tokens": 999},
                },
            }
        )

    with mock.patch("agent6.providers._transport.http_post", side_effect=fake_post):
        resp = provider.call(system="s", messages=[{"role": "user", "content": "x"}])

    assert resp.input_tokens == 0
    # cache_read is clamped to the prompt total too (single source of truth): an
    # upstream reporting cached(999) > prompt(10) no longer leaves cache_read_tokens
    # inconsistent with the clamped input (which budget.py would mis-bill).
    assert resp.cache_read_tokens == 10


def test_call_flattens_anthropic_block_content() -> None:
    provider = OpenAIProvider(api_key="sk", model="gpt-x")
    captured: dict[str, Any] = {}

    def fake_post(*_a: Any, **kw: Any) -> httpx2.Response:
        captured["body"] = json.loads(kw["content"])
        return _fake_response({"choices": [{"message": {"content": "ok"}}], "usage": {}})

    msg_content = [
        {"type": "text", "text": "hello "},
        {"type": "text", "text": "world"},
    ]
    with mock.patch("agent6.providers._transport.http_post", side_effect=fake_post):
        provider.call(system="s", messages=[{"role": "user", "content": msg_content}])

    assert captured["body"]["messages"][1] == {"role": "user", "content": "hello world"}


def test_call_raises_provider_error_on_http_status() -> None:
    provider = OpenAIProvider(api_key="sk", model="gpt-x")
    with (
        mock.patch(
            "agent6.providers._transport.http_post",
            return_value=_fake_response({"error": "no"}, status=500),
        ),
        pytest.raises(ProviderError, match="OpenAI API error 500"),
    ):
        provider.call(system="s", messages=[{"role": "user", "content": "x"}])


def test_from_env_missing_env_var_yields_no_auth_header(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Empty/unset env var is allowed (Ollama-style local endpoint)."""
    monkeypatch.delenv("MY_OAI_KEY", raising=False)
    provider = OpenAIProvider.from_env(model="gpt-x", env_var="MY_OAI_KEY")
    assert provider.api_key == ""

    captured: dict[str, Any] = {}

    def fake_post(_url: str, *_a: Any, **kw: Any) -> httpx2.Response:
        captured["headers"] = kw["headers"]
        return _fake_response(
            {
                "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            }
        )

    with mock.patch("agent6.providers._transport.http_post", side_effect=fake_post):
        provider.call(system="s", messages=[{"role": "user", "content": "x"}])

    assert "authorization" not in {k.lower() for k in captured["headers"]}


def test_from_env_none_env_var_loads(monkeypatch: pytest.MonkeyPatch) -> None:
    """env_var=None is a valid Ollama/llama.cpp shape."""
    provider = OpenAIProvider.from_env(model="gpt-x", env_var=None)
    assert provider.api_key == ""


def test_base_url_override_and_extra_headers() -> None:
    """OpenRouter-style usage: custom endpoint + required identifying headers."""
    provider = OpenAIProvider(
        api_key="or-test",
        model="meta-llama/llama-3.3-70b-instruct",
        base_url="https://openrouter.ai/api/v1",
        extra_headers=(("HTTP-Referer", "https://example.com/r"), ("X-Title", "agent6")),
    )
    captured: dict[str, Any] = {}

    def fake_post(url: str, *_a: Any, **kw: Any) -> httpx2.Response:
        captured["url"] = url
        captured["headers"] = kw["headers"]
        return _fake_response({"choices": [{"message": {"content": "k"}}], "usage": {}})

    with mock.patch("agent6.providers._transport.http_post", side_effect=fake_post):
        provider.call(system="s", messages=[{"role": "user", "content": "x"}])

    assert captured["url"] == "https://openrouter.ai/api/v1/chat/completions"
    # extra_headers are lowercased into the request:
    assert captured["headers"]["http-referer"] == "https://example.com/r"
    assert captured["headers"]["x-title"] == "agent6"
    # default auth still present
    assert captured["headers"]["authorization"] == "Bearer or-test"


def test_from_env_threads_base_url(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OR_KEY", "k")
    p = OpenAIProvider.from_env(
        model="m",
        env_var="OR_KEY",
        base_url="http://localhost:11434/v1",
        extra_headers={"X-Title": "t"},
    )
    assert p.base_url == "http://localhost:11434/v1"
    assert p.endpoint == "http://localhost:11434/v1/chat/completions"
    assert dict(p.extra_headers) == {"X-Title": "t"}


# --- : reasoning-model handling -----------------------------------


def test_is_reasoning_model_detects_thinking_models() -> None:
    from agent6.providers import openai as oai

    _is_reasoning_model = oai._is_reasoning_model  # pyright: ignore[reportPrivateUsage]

    assert _is_reasoning_model("kimi-k2-thinking")
    assert _is_reasoning_model("deepseek-r1-distill")
    assert _is_reasoning_model("qwq-32b-preview")
    assert _is_reasoning_model("o1-preview")
    assert _is_reasoning_model("o3-mini")
    assert _is_reasoning_model("Reasoning-Pro-2")
    # bare-name reasoning emitters (no "effort" suffix advertised).
    assert _is_reasoning_model("moonshotai/kimi-k2.6")
    assert _is_reasoning_model("moonshotai/kimi-k2.5")
    # The whole Moonshot K family reasons, not one generation: kimi-k3 missed
    # the old "kimi-k2" hint and starved a 3-seat review panel at 4.5k
    # (finish=length, 0 content chars, ~5.8k reasoning chars, all abstained).
    assert _is_reasoning_model("moonshotai/kimi-k3")
    assert _is_reasoning_model("minimax/minimax-m2.7")
    assert _is_reasoning_model("minimax/minimax-m2")
    assert _is_reasoning_model("nvidia/nemotron-3-nano-30b-a3b")
    # GLM-4.x/5.x all stream a separate reasoning channel and starve at the
    # default cap (direct OpenRouter probe: glm-4.6/4.7/5.2 each returned
    # finish_reason="length" with empty content and ~all tokens as reasoning).
    assert _is_reasoning_model("z-ai/glm-4.6")
    assert _is_reasoning_model("z-ai/glm-5.2")
    assert not _is_reasoning_model("gpt-4o")
    assert not _is_reasoning_model("claude-3-5-sonnet")
    assert not _is_reasoning_model("llama-3-70b")


def test_reasoning_floor_covers_kimi_latest_without_the_effort_default() -> None:
    """The max_tokens FLOOR and the effort DEFAULT are split: `kimi-latest`
    (Moonshot's rolling alias) emits reasoning_content and needs the headroom, but
    the `kimi-k` family match misses it -- and adding it to the effort set would
    pin it to an UNMEASURED reasoning_effort="low" for whatever it resolves to. It
    gets the floor only; the floor set is a superset of the effort set."""
    from agent6.providers import openai as oai

    needs_headroom = oai._needs_reasoning_headroom  # pyright: ignore[reportPrivateUsage]
    is_reasoning = oai._is_reasoning_model  # pyright: ignore[reportPrivateUsage]

    assert needs_headroom("moonshotai/kimi-latest")  # floored
    assert not is_reasoning("moonshotai/kimi-latest")  # but NOT effort="low"-defaulted
    # every effort-set model still needs headroom (superset), non-reasoners none.
    assert needs_headroom("moonshotai/kimi-k3") and is_reasoning("moonshotai/kimi-k3")
    assert not needs_headroom("gpt-4o")
    # OpenAI's own o-series / gpt-5 reason and starve too, but are matched
    # narrowly (for the direct-host rename); they still need the floor.
    assert needs_headroom("gpt-5") and needs_headroom("o1") and needs_headroom("o3-mini")
    assert not needs_headroom("gpt-4o-mini")  # non-reasoning openai model stays out


def test_call_bumps_max_tokens_for_reasoning_models() -> None:
    """Kimi-K2-Thinking should get >=32768 max_tokens even if caller asks
    for 16384 - reasoning_content shares the budget with content + tool
    calls and starves them at low caps. Non-reasoning models keep the
    caller-supplied value."""
    from agent6.providers.openai import REASONING_MODEL_MIN_MAX_TOKENS

    provider = OpenAIProvider(api_key="sk", model="kimi-k2-thinking")
    captured: dict[str, Any] = {}

    def fake_post(*_a: Any, **kw: Any) -> httpx2.Response:
        captured["body"] = json.loads(kw["content"])
        return _fake_response({"choices": [{"message": {"content": "ok"}}], "usage": {}})

    with mock.patch("agent6.providers._transport.http_post", side_effect=fake_post):
        provider.call(system="s", messages=[{"role": "user", "content": "hi"}], max_tokens=16384)
    assert captured["body"]["max_tokens"] == REASONING_MODEL_MIN_MAX_TOKENS

    # Caller-supplied value above the floor wins.
    with mock.patch("agent6.providers._transport.http_post", side_effect=fake_post):
        provider.call(system="s", messages=[{"role": "user", "content": "hi"}], max_tokens=65536)
    assert captured["body"]["max_tokens"] == 65536


def test_call_does_not_bump_max_tokens_for_normal_models() -> None:
    provider = OpenAIProvider(api_key="sk", model="gpt-4o")
    captured: dict[str, Any] = {}

    def fake_post(*_a: Any, **kw: Any) -> httpx2.Response:
        captured["body"] = json.loads(kw["content"])
        return _fake_response({"choices": [{"message": {"content": "ok"}}], "usage": {}})

    with mock.patch("agent6.providers._transport.http_post", side_effect=fake_post):
        provider.call(system="s", messages=[{"role": "user", "content": "hi"}], max_tokens=4096)
    assert captured["body"]["max_tokens"] == 4096


def test_reasoning_effort_arg_overrides_default(monkeypatch: Any) -> None:
    """An explicit ``reasoning_effort`` argument takes precedence
    over the AGENT6_REASONING_EFFORT env override and the built-in
    default. : ``"off"`` sends ``reasoning={"enabled": False}`` to
    truly disable the reasoning channel (omitting the block left it ON by
    default on K2.6, so the recovery turn still starved)."""
    monkeypatch.setenv("AGENT6_REASONING_EFFORT", "medium")
    provider = OpenAIProvider(api_key="sk", model="moonshotai/kimi-k2.6")
    captured: dict[str, Any] = {}

    def fake_post(*_a: Any, **kw: Any) -> httpx2.Response:
        captured["body"] = json.loads(kw["content"])
        return _fake_response({"choices": [{"message": {"content": "ok"}}], "usage": {}})

    # No arg -> env override wins.
    with mock.patch("agent6.providers._transport.http_post", side_effect=fake_post):
        provider.call(system="s", messages=[{"role": "user", "content": "hi"}])
    assert captured["body"]["reasoning"] == {"effort": "medium"}

    # Explicit "off" -> reasoning channel explicitly disabled.
    with mock.patch("agent6.providers._transport.http_post", side_effect=fake_post):
        provider.call(
            system="s", messages=[{"role": "user", "content": "hi"}], reasoning_effort="off"
        )
    assert captured["body"]["reasoning"] == {"enabled": False}

    # Explicit "low" -> overrides env "medium".
    with mock.patch("agent6.providers._transport.http_post", side_effect=fake_post):
        provider.call(
            system="s", messages=[{"role": "user", "content": "hi"}], reasoning_effort="low"
        )
    assert captured["body"]["reasoning"] == {"effort": "low"}


def test_call_captures_reasoning_content_in_raw() -> None:
    """Kimi-shaped ``reasoning_content`` is preserved on resp.raw["content"]
    as a Anthropic-style ``{"type": "thinking"}`` block, but does NOT leak
    into resp.text (workflows.loop strips ``<thinking>`` prefixes from the
    auto-commit summary, and we don't want it double-printed)."""
    provider = OpenAIProvider(api_key="sk", model="kimi-k2-thinking")

    def fake_post(*_a: Any, **_kw: Any) -> httpx2.Response:
        return _fake_response(
            {
                "choices": [
                    {
                        "message": {
                            "content": "the answer is 42",
                            "reasoning_content": "step 1: think. step 2: 42.",
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 50},
            }
        )

    with mock.patch("agent6.providers._transport.http_post", side_effect=fake_post):
        resp = provider.call(system="s", messages=[{"role": "user", "content": "q"}])

    assert resp.text == "the answer is 42"
    raw_content = resp.raw["content"]
    assert raw_content[0] == {
        "type": "thinking",
        "thinking": "step 1: think. step 2: 42.",
    }
    assert raw_content[1] == {"type": "text", "text": "the answer is 42"}


def test_call_captures_deepseek_reasoning_field() -> None:
    """DeepSeek-R1 / OpenRouter spell it ``reasoning`` (no _content)."""
    provider = OpenAIProvider(api_key="sk", model="deepseek-r1")

    def fake_post(*_a: Any, **_kw: Any) -> httpx2.Response:
        return _fake_response(
            {
                "choices": [
                    {
                        "message": {"content": "ok", "reasoning": "thinking out loud"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {},
            }
        )

    with mock.patch("agent6.providers._transport.http_post", side_effect=fake_post):
        resp = provider.call(system="s", messages=[{"role": "user", "content": "q"}])

    assert any(
        b.get("type") == "thinking" and b.get("thinking") == "thinking out loud"
        for b in resp.raw["content"]
    )


def _counter_argv(tmp_path: Path) -> list[str]:
    counter = tmp_path / "counter"
    script = (
        f'n=$(cat "{counter}" 2>/dev/null || echo 0); '
        f'n=$((n + 1)); printf %s "$n" > "{counter}"; printf "tok%s" "$n"'
    )
    return ["sh", "-c", script]


def test_credential_overrides_static_key_in_auth_header() -> None:
    # A token_command credential mints the bearer; the static api_key is ignored.
    provider = OpenAIProvider(
        api_key="static-key", model="m", credential=CommandToken(["printf", "minted-tok"])
    )
    captured: dict[str, Any] = {}

    def fake_post(*_a: Any, **kw: Any) -> httpx2.Response:
        captured["auth"] = kw["headers"].get("authorization")
        return _fake_response(
            {"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]}
        )

    with mock.patch("agent6.providers._transport.http_post", side_effect=fake_post):
        resp = provider.call(system="s", messages=[{"role": "user", "content": "hi"}])

    assert captured["auth"] == "Bearer minted-tok"
    assert resp.text == "ok"


def test_401_refreshes_token_command_and_retries(tmp_path: Path) -> None:
    # First attempt 401s; the credential is invalidated and the retry carries a
    # freshly-minted token (tok2), then succeeds.
    provider = OpenAIProvider(
        api_key="", model="m", credential=CommandToken(_counter_argv(tmp_path), ttl_s=1000.0)
    )
    seen: list[str | None] = []
    responses = [
        _fake_response({}, status=401),
        _fake_response(
            {"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]}, status=200
        ),
    ]

    def fake_post(*_a: Any, **kw: Any) -> httpx2.Response:
        seen.append(kw["headers"].get("authorization"))
        return responses[len(seen) - 1]

    with mock.patch("agent6.providers._transport.http_post", side_effect=fake_post):
        resp = provider.call(system="s", messages=[{"role": "user", "content": "hi"}])

    assert seen == ["Bearer tok1", "Bearer tok2"]
    assert resp.text == "ok"


def test_401_without_credential_is_not_retried() -> None:
    # No credential -> single attempt, the 401 surfaces immediately (no loop).
    provider = OpenAIProvider(api_key="static", model="m")
    calls = {"n": 0}

    def fake_post(*_a: Any, **_kw: Any) -> httpx2.Response:
        calls["n"] += 1
        return _fake_response({}, status=401)

    with (
        mock.patch("agent6.providers._transport.http_post", side_effect=fake_post),
        pytest.raises(ProviderError),
    ):
        provider.call(system="s", messages=[{"role": "user", "content": "hi"}])
    assert calls["n"] == 1


def test_an_upstream_error_completion_is_retryable_and_still_metered() -> None:
    """Observed from OpenRouter: a 200 whose choice carries
    `finish_reason: "error"`, a null content and nothing else, after the model
    spent its whole budget in the reasoning channel. Returned as a finished
    turn it spends a went-quiet nudge on an upstream failure and abstains a
    review seat as if the model had answered; the tokens are billed either
    way, so it meters first and then retries."""
    from agent6.budget import BudgetTracker

    failed = {
        "choices": [
            {"finish_reason": "error", "message": {"content": None, "reasoning": "thinking"}}
        ],
        "usage": {"prompt_tokens": 12000, "completion_tokens": 16801},
    }
    budget = BudgetTracker(max_usd=-1, max_tokens_fallback=-1, max_percent=-1)
    provider = OpenAIProvider(api_key="k", model="gpt-x", budget=budget)

    def fake_post(*_a: Any, **_kw: Any) -> httpx2.Response:
        return _fake_response(failed)

    with (
        mock.patch("agent6.providers._transport.http_post", side_effect=fake_post),
        pytest.raises(ProviderError, match="finish_reason='error'"),
    ):
        provider.call(system="s", messages=[{"role": "user", "content": "hi"}])

    snap = budget.snapshot()
    assert (snap.input_total, snap.output_total) == (12000, 16801), "billed tokens went unmetered"

    # A partial answer under the same finish reason is still handed back.
    from agent6.providers._openai_parse import parse_response

    partial = {
        "choices": [{"finish_reason": "error", "message": {"content": "half an answer"}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5},
    }
    assert parse_response(partial).text == "half an answer"
