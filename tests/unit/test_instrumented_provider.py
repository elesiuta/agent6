# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The cli ``InstrumentedProvider`` wrapper must forward every
provider.call kwarg to the inner provider. A missing passthrough is
invisible to unit tests that call providers directly but crashes every
real run (regression: ``reasoning_effort`` was added to the providers
and the loop but not the wrapper, so the perf bench died with
``TypeError: ... got an unexpected keyword argument 'reasoning_effort'``).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from agent6.app.providers import InstrumentedProvider
from agent6.budget import BudgetTracker
from agent6.providers import ProviderResponse


def _resp() -> ProviderResponse:
    return ProviderResponse(
        text="ok",
        tool_uses=(),
        stop_reason="end_turn",
        input_tokens=1,
        output_tokens=1,
        cache_read_tokens=0,
        cache_creation_tokens=0,
        raw={"content": [{"type": "text", "text": "ok"}]},
    )


def _wrap(inner: MagicMock) -> InstrumentedProvider:
    return InstrumentedProvider(
        inner=inner,
        role="worker",
        model="moonshotai/kimi-k2.6",
        provider_name="openai",
        events=MagicMock(),
        budget=BudgetTracker(max_usd=-1, max_tokens_fallback=-1, max_percent=-1),
    )


def test_instrumented_provider_forwards_reasoning_effort() -> None:
    inner = MagicMock()
    inner.call.return_value = _resp()
    wrapper = _wrap(inner)

    wrapper.call(
        system="s",
        messages=[{"role": "user", "content": "hi"}],
        reasoning_effort="off",
    )

    kwargs: dict[str, Any] = inner.call.call_args.kwargs
    assert kwargs["reasoning_effort"] == "off"


def test_instrumented_provider_forwards_should_abort() -> None:
    inner = MagicMock()
    inner.call.return_value = _resp()
    wrapper = _wrap(inner)

    def _abort() -> bool:
        return True

    wrapper.call(system="s", messages=[{"role": "user", "content": "hi"}], should_abort=_abort)
    assert inner.call.call_args.kwargs["should_abort"] is _abort


def test_instrumented_provider_defaults_reasoning_effort_to_none() -> None:
    inner = MagicMock()
    inner.call.return_value = _resp()
    wrapper = _wrap(inner)

    wrapper.call(system="s", messages=[{"role": "user", "content": "hi"}])

    assert inner.call.call_args.kwargs["reasoning_effort"] is None


def test_the_journal_records_what_the_assistant_said(tmp_path: Path) -> None:
    """The contract three readers depend on: `read_session`, `/btw`, and the
    transcript fold all reconstruct the conversation from this event.

    The prose used to reach the journal only as `role.text_delta`, emitted only
    when streaming is on, so a headless run recorded none of it -- and each
    reader had a hand-written fixture inventing this field, so all three were
    green against a shape the engine never emitted. Pinned at the EMITTER: a
    fixture can drift, this cannot.
    """
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    from agent6.app.providers import InstrumentedProvider
    from agent6.budget import BudgetTracker
    from agent6.events import EventSink

    events = EventSink(tmp_path / "logs.jsonl")
    inner = MagicMock()
    inner.call.return_value = SimpleNamespace(
        text="the answer",
        tool_uses=(),
        stop_reason="end_turn",
        input_tokens=1,
        output_tokens=1,
        cache_read_tokens=0,
        cache_creation_tokens=0,
        raw={},
    )
    InstrumentedProvider(
        inner=inner,
        role="worker",
        model="m",
        provider_name="p",
        events=events,
        budget=BudgetTracker(max_usd=-1, max_tokens_fallback=-1, max_percent=-1),
    ).call(system="s", messages=[], tools=[], max_tokens=8)

    settled = [
        json.loads(line)
        for line in (tmp_path / "logs.jsonl").read_text(encoding="utf-8").splitlines()
        if json.loads(line)["type"] == "role.result"
    ]
    assert [e["text"] for e in settled] == ["the answer"]


def test_a_failed_call_still_reports_what_it_spent(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A cut stream is billed, and `budget.update` is the only path that spend
    takes to a surface.

    The providers record what a dead stream already cost, but the emission sat
    on the success path only, so those dollars reached the USD ceiling and
    nothing else: not the live cost meters, not `sessions list`, and not the
    machine spend ledger, which rebuilds a state's cost from the last such event
    in its log. The end-of-run summary prints to the terminal and is never
    journalled, so the under-report was permanent.
    """
    from agent6.events import EventSink
    from agent6.providers import ProviderError

    # The USD assertion needs a table price; the suite isolates the model-price
    # cache, so seed one (the suite never reads the developer's real cache).
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    (tmp_path / "agent6" / "models").mkdir(parents=True, exist_ok=True)
    pricing = {"anthropic/claude-haiku-4.5": [1.0, 5.0]}
    (tmp_path / "agent6" / "models" / "anthropic.json").write_text(
        json.dumps({"models": list(pricing), "pricing": pricing}), encoding="utf-8"
    )
    events = EventSink(tmp_path / "logs.jsonl")
    budget = BudgetTracker(max_usd=10.0, max_tokens_fallback=2_000_000, max_percent=-1)

    def _cut_stream(**_: object) -> ProviderResponse:
        budget.record(
            model="anthropic/claude-haiku-4.5",
            input_tokens=50_000,
            output_tokens=120,
            cache_read_tokens=0,
            cache_creation_tokens=0,
        )
        raise ProviderError("stream cut before completion")

    inner = MagicMock()
    inner.call.side_effect = _cut_stream
    wrapper = InstrumentedProvider(
        inner=inner,
        role="worker",
        model="anthropic/claude-haiku-4.5",
        provider_name="anthropic",
        events=events,
        budget=budget,
    )

    with pytest.raises(ProviderError):
        wrapper.call(system="s", messages=[])

    updates = [
        json.loads(line)
        for line in (tmp_path / "logs.jsonl").read_text(encoding="utf-8").splitlines()
        if json.loads(line)["type"] == "budget.update"
    ]
    assert [(e["input_total"], e["output_total"]) for e in updates] == [(50_000, 120)]
    assert updates[0]["usd_total"] > 0.0


def test_a_provider_error_is_stamped_with_the_provider_name() -> None:
    """The loop's credential hint names the failing provider's config key
    (`[providers.openai].api_key_env`) instead of a `<name>` placeholder; the
    wrapper is the one place that knows the name."""
    from agent6.providers import ProviderError

    inner = MagicMock()
    inner.call.side_effect = ProviderError("401 nope", status_code=401)
    with pytest.raises(ProviderError) as info:
        _wrap(inner).call(system="s", messages=[])
    assert info.value.provider == "openai"
