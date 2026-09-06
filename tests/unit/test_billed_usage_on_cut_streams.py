# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""A stream that dies after the provider reported usage still cost money.

`budget.record` ran only when a stream completed, so every early exit -- a
retryable mid-stream error, the idle watchdog, an operator steer or stop --
spent money `max_usd` never saw. Each retry re-sends the whole input and is
billed again, so a run with any flakiness had no ceiling at all: the operator
set a number for the task and could pass it without being told.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest import mock

import httpx2
import pytest

from agent6.budget import BudgetTracker
from agent6.providers import AnthropicProvider, OpenAIProvider
from tests.unit.test_anthropic_streaming import FakeStreamResponse


class _CutAfter(FakeStreamResponse):
    """Serves lines until `cut_at`, then dies like a dropped connection."""

    cut_at: int = 0

    def iter_lines(self) -> Any:
        for i, line in enumerate(self._lines):
            if i >= self.cut_at:
                raise httpx2.ReadError("connection dropped mid-stream")
            yield line


def _cut(lines: list[str], at: int) -> _CutAfter:
    resp = _CutAfter(status_code=200, lines=lines)
    resp.cut_at = at
    return resp


def _sse(event: str, data: dict[str, Any]) -> list[str]:
    return [f"event: {event}", f"data: {json.dumps(data)}", ""]


def test_anthropic_records_what_a_cut_stream_already_cost(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # The USD assertion needs a table price; the suite isolates the model-price
    # cache, so seed one (the suite never reads the developer's real cache).
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    (tmp_path / "agent6" / "models").mkdir(parents=True, exist_ok=True)
    pricing = {"claude-sonnet-4-5": [3.0, 15.0]}
    (tmp_path / "agent6" / "models" / "anthropic.json").write_text(
        json.dumps({"models": list(pricing), "pricing": pricing}), encoding="utf-8"
    )
    lines = _sse(
        "message_start",
        {
            "message": {
                "usage": {
                    "input_tokens": 50_000,
                    "output_tokens": 0,
                    "cache_read_input_tokens": 0,
                    "cache_creation_input_tokens": 0,
                }
            }
        },
    )
    lines += _sse(
        "content_block_start", {"index": 0, "content_block": {"type": "text", "text": ""}}
    )
    lines += _sse(
        "content_block_delta", {"index": 0, "delta": {"type": "text_delta", "text": "partial"}}
    )

    budget = BudgetTracker(max_usd=10.0, max_tokens_fallback=-1, max_percent=-1)
    provider = AnthropicProvider(api_key="k", model="claude-sonnet-4-5", budget=budget)
    with (
        mock.patch("agent6.providers._stream.http_stream", return_value=_cut(lines, at=9)),
        pytest.raises(Exception),
    ):
        provider.call(
            messages=[{"role": "user", "content": "hi"}],
            system="",
            tools=[],
            # The callback is what selects the STREAMING path.
            text_delta_callback=lambda _t: None,
        )

    snap = budget.snapshot()
    assert snap.input_total == 50_000, "the provider billed this input; the cap must see it"
    spent, _ = budget.estimate_usd()
    assert spent > 0


def test_openai_records_a_cut_stream_and_keeps_the_cached_split() -> None:
    """Through parse_response, so the cached-vs-fresh mapping has one owner."""

    def chunk(obj: dict[str, Any]) -> list[str]:
        return [f"data: {json.dumps(obj)}", ""]

    lines = chunk({"choices": [{"index": 0, "delta": {"role": "assistant", "content": "hi"}}]})
    lines += chunk(
        {
            "usage": {
                "prompt_tokens": 40_000,
                "completion_tokens": 120,
                "prompt_tokens_details": {"cached_tokens": 10_000},
            },
            "choices": [],
        }
    )
    lines += chunk({"choices": [{"index": 0, "delta": {"content": " more"}}]})

    budget = BudgetTracker(max_usd=10.0, max_tokens_fallback=-1, max_percent=-1)
    provider = OpenAIProvider(api_key="k", model="gpt-4o", budget=budget)
    with (
        mock.patch("agent6.providers._stream.http_stream", return_value=_cut(lines, at=6)),
        pytest.raises(Exception),
    ):
        provider.call(
            messages=[{"role": "user", "content": "hi"}],
            system="",
            tools=[],
            # The callback is what selects the STREAMING path.
            text_delta_callback=lambda _t: None,
        )

    snap = budget.snapshot()
    assert snap.input_total == 30_000  # fresh input, cached counted separately
    assert snap.cache_read_total == 10_000
    assert snap.output_total == 120


def test_a_stream_that_reported_nothing_records_nothing() -> None:
    """An unknown amount is not a licence to invent one: a stream cut before
    any usage arrived must leave the ledger untouched."""
    lines = _sse("content_block_start", {"index": 0, "content_block": {"type": "text", "text": ""}})

    budget = BudgetTracker(max_usd=10.0, max_tokens_fallback=-1, max_percent=-1)
    provider = AnthropicProvider(api_key="k", model="claude-sonnet-4-5", budget=budget)
    with (
        mock.patch("agent6.providers._stream.http_stream", return_value=_cut(lines, at=1)),
        pytest.raises(Exception),
    ):
        provider.call(
            messages=[{"role": "user", "content": "hi"}],
            system="",
            tools=[],
            # The callback is what selects the STREAMING path.
            text_delta_callback=lambda _t: None,
        )

    snap = budget.snapshot()
    assert snap.input_total == 0
    assert snap.output_total == 0
    # per_model is where a spurious zero-count record would show: a stream that
    # reported nothing must not seed a model entry at all.
    assert snap.per_model == {}


def _pricing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    (tmp_path / "agent6" / "models").mkdir(parents=True, exist_ok=True)
    pricing = {"claude-sonnet-4-5": [3.0, 15.0], "gpt-4o": [3.0, 15.0]}
    (tmp_path / "agent6" / "models" / "x.json").write_text(
        json.dumps({"models": list(pricing), "pricing": pricing}), encoding="utf-8"
    )


def test_anthropic_records_a_completed_stream_its_meter_guard_refuses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A gateway with usage tracking off completes the message with zero input
    tokens; the guard refused it retryably WITHOUT recording the 800 output
    tokens it had billed, so every retry re-sent the input and `max_usd`
    never moved."""
    from agent6.providers.types import ProviderError

    _pricing(tmp_path, monkeypatch)
    lines = _sse(
        "message_start",
        {
            "message": {
                "usage": {
                    "input_tokens": 0,
                    "cache_read_input_tokens": 0,
                    "cache_creation_input_tokens": 0,
                }
            }
        },
    )
    lines += _sse(
        "content_block_start", {"index": 0, "content_block": {"type": "text", "text": ""}}
    )
    lines += _sse(
        "content_block_delta", {"index": 0, "delta": {"type": "text_delta", "text": "hi"}}
    )
    lines += _sse("content_block_stop", {"index": 0})
    lines += _sse(
        "message_delta", {"delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 800}}
    )
    lines += _sse("message_stop", {})
    budget = BudgetTracker(max_usd=10.0, max_tokens_fallback=-1, max_percent=-1)
    provider = AnthropicProvider(api_key="k", model="claude-sonnet-4-5", budget=budget)
    with (
        mock.patch(
            "agent6.providers._stream.http_stream",
            return_value=FakeStreamResponse(status_code=200, lines=lines),
        ),
        pytest.raises(ProviderError, match="no usage input tokens"),
    ):
        provider.call(
            system="s",
            messages=[{"role": "user", "content": "hi"}],
            text_delta_callback=lambda _s: None,
        )
    assert budget.snapshot().output_total == 800
    assert budget.estimate_usd()[0] == pytest.approx(800 * 15.0 / 1e6)


def test_openai_records_a_completed_stream_its_meter_guard_refuses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The OpenAI twin: a [DONE] stream whose usage trailer says prompt_tokens
    0 was refused with its 900 completion tokens unrecorded."""
    from agent6.providers.types import ProviderError

    _pricing(tmp_path, monkeypatch)
    lines = [
        "data: " + json.dumps({"choices": [{"index": 0, "delta": {"content": "hello"}}]}),
        "",
        "data: " + json.dumps({"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}),
        "",
        "data: "
        + json.dumps({"choices": [], "usage": {"prompt_tokens": 0, "completion_tokens": 900}}),
        "",
        "data: [DONE]",
        "",
    ]
    budget = BudgetTracker(max_usd=10.0, max_tokens_fallback=-1, max_percent=-1)
    provider = OpenAIProvider(api_key="k", model="gpt-4o", budget=budget)
    with (
        mock.patch(
            "agent6.providers._stream.http_stream",
            return_value=FakeStreamResponse(status_code=200, lines=lines),
        ),
        pytest.raises(ProviderError, match="no usage input tokens"),
    ):
        provider.call(
            system="s",
            messages=[{"role": "user", "content": "hi"}],
            text_delta_callback=lambda _s: None,
        )
    assert budget.snapshot().output_total == 900


def test_anthropic_meters_a_completed_stream_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The refusal-path record ran before the guards for every completed
    message, so an accepted completion was booked twice: once there, once by
    meter_completion. `max_usd` tripped at half the spend."""
    _pricing(tmp_path, monkeypatch)
    lines = _sse(
        "message_start",
        {
            "message": {
                "usage": {
                    "input_tokens": 1_000,
                    "cache_read_input_tokens": 0,
                    "cache_creation_input_tokens": 0,
                }
            }
        },
    )
    lines += _sse(
        "content_block_start", {"index": 0, "content_block": {"type": "text", "text": ""}}
    )
    lines += _sse(
        "content_block_delta", {"index": 0, "delta": {"type": "text_delta", "text": "hi"}}
    )
    lines += _sse("content_block_stop", {"index": 0})
    lines += _sse(
        "message_delta", {"delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 20}}
    )
    lines += _sse("message_stop", {})
    budget = BudgetTracker(max_usd=10.0, max_tokens_fallback=-1, max_percent=-1)
    provider = AnthropicProvider(api_key="k", model="claude-sonnet-4-5", budget=budget)
    with mock.patch(
        "agent6.providers._stream.http_stream",
        return_value=FakeStreamResponse(status_code=200, lines=lines),
    ):
        provider.call(
            system="s",
            messages=[{"role": "user", "content": "hi"}],
            text_delta_callback=lambda _s: None,
        )
    snap = budget.snapshot()
    assert (snap.input_total, snap.output_total) == (1_000, 20)


def test_openai_meters_a_completed_stream_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The OpenAI twin of the double booking."""
    _pricing(tmp_path, monkeypatch)
    lines = [
        "data: " + json.dumps({"choices": [{"index": 0, "delta": {"content": "hello"}}]}),
        "",
        "data: " + json.dumps({"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}),
        "",
        "data: "
        + json.dumps({"choices": [], "usage": {"prompt_tokens": 1_000, "completion_tokens": 20}}),
        "",
        "data: [DONE]",
        "",
    ]
    budget = BudgetTracker(max_usd=10.0, max_tokens_fallback=-1, max_percent=-1)
    provider = OpenAIProvider(api_key="k", model="gpt-4o", budget=budget)
    with mock.patch(
        "agent6.providers._stream.http_stream",
        return_value=FakeStreamResponse(status_code=200, lines=lines),
    ):
        provider.call(
            system="s",
            messages=[{"role": "user", "content": "hi"}],
            text_delta_callback=lambda _s: None,
        )
    snap = budget.snapshot()
    assert (snap.input_total, snap.output_total) == (1_000, 20)
