# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Tests for cache-fetched model pricing (agent6.models.pricing + models_cache)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent6.models.cache import _parse_pricing  # pyright: ignore[reportPrivateUsage]
from agent6.models.pricing import lookup_price


def _write_pricing(cache: Path, name: str, pricing: dict[str, list[float]]) -> None:
    (cache / "models").mkdir(parents=True, exist_ok=True)
    (cache / "models" / f"{name}.json").write_text(
        json.dumps({"models": list(pricing), "pricing": pricing}), encoding="utf-8"
    )


def test_lookup_price_reads_cache(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("AGENT6_CACHE_HOME", str(tmp_path))
    _write_pricing(tmp_path, "openrouter", {"a/model": [0.5, 2.5]})
    assert lookup_price("a/model") == (0.5, 2.5)
    assert lookup_price("nobody/else") is None


def test_lookup_price_sees_cache_written_after_first_miss(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # The CLI preflight refreshes the cache AFTER config construction already
    # did a lookup; the memo must not pin the early empty result.
    monkeypatch.setenv("AGENT6_CACHE_HOME", str(tmp_path))
    assert lookup_price("a/model") is None
    _write_pricing(tmp_path, "openrouter", {"a/model": [0.5, 2.5]})
    assert lookup_price("a/model") == (0.5, 2.5)


def test_lookup_price_ignores_malformed_entries(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("AGENT6_CACHE_HOME", str(tmp_path))
    (tmp_path / "models").mkdir(parents=True)
    (tmp_path / "models" / "bad.json").write_text(
        json.dumps(
            {
                "pricing": {
                    "neg/model": [-1.0, 2.0],
                    "str/model": ["x", "y"],
                    "short/model": [1.0],
                    "ok/model": [1.0, 2.0],
                }
            }
        ),
        encoding="utf-8",
    )
    assert lookup_price("ok/model") == (1.0, 2.0)
    for bad in ("neg/model", "str/model", "short/model"):
        assert lookup_price(bad) is None


def test_parse_pricing_openrouter_shape() -> None:
    # OpenRouter reports USD per TOKEN as strings; normalized to per-MTok.
    payload = {
        "data": [
            {
                "id": "moonshotai/kimi-k2.6",
                "pricing": {"prompt": "0.00000068", "completion": "0.00000341"},
            },
            {"id": "no-pricing-model"},
            {"id": "bad-pricing", "pricing": {"prompt": "free", "completion": "0"}},
        ]
    }
    got = _parse_pricing(payload)
    assert got["moonshotai/kimi-k2.6"] == (pytest.approx(0.68), pytest.approx(3.41))
    assert "no-pricing-model" not in got
    assert "bad-pricing" not in got


def test_lookup_price_direct_anthropic_alias(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # A direct-Anthropic id resolves through its OpenRouter listing: date
    # suffix stripped, trailing version dotted, `anthropic/` prefixed.
    monkeypatch.setenv("AGENT6_CACHE_HOME", str(tmp_path))
    _write_pricing(
        tmp_path,
        "openrouter",
        {
            "anthropic/claude-haiku-4.5": [1.0, 5.0],
            "anthropic/claude-opus-4.8": [5.0, 25.0],
            "anthropic/claude-sonnet-5": [2.0, 10.0],
        },
    )
    assert lookup_price("claude-haiku-4-5-20251001") == (1.0, 5.0)
    assert lookup_price("claude-opus-4-8") == (5.0, 25.0)
    assert lookup_price("claude-sonnet-5") == (2.0, 10.0)


def test_lookup_price_alias_never_shadows_exact(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("AGENT6_CACHE_HOME", str(tmp_path))
    _write_pricing(
        tmp_path,
        "openrouter",
        {"claude-opus-4-8": [9.0, 9.0], "anthropic/claude-opus-4.8": [5.0, 25.0]},
    )
    assert lookup_price("claude-opus-4-8") == (9.0, 9.0)


def test_lookup_price_alias_misses_stay_unpriced(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("AGENT6_CACHE_HOME", str(tmp_path))
    _write_pricing(tmp_path, "openrouter", {"anthropic/claude-3.5-sonnet": [3.0, 15.0]})
    # Legacy version-first naming is deliberately not mapped.
    assert lookup_price("claude-3-5-sonnet-20241022") is None
    # Namespaced ids are never rewritten.
    assert lookup_price("someorg/claude-haiku-4-5") is None


def test_a_model_two_providers_list_is_priced_by_its_route(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every provider cache was merged by model id, first file name winning,
    so a cheap gateway's listing priced a call that went through OpenRouter
    at a 41x under-count of the enforced `max_usd`."""
    import json

    from agent6.budget import BudgetTracker
    from agent6.models.pricing import lookup_price

    monkeypatch.setenv("AGENT6_CACHE_HOME", str(tmp_path))
    (tmp_path / "models").mkdir()
    (tmp_path / "models" / "aaa_cheap.json").write_text(
        json.dumps({"models": ["openai/gpt-4o"], "pricing": {"openai/gpt-4o": [0.10, 0.20]}}),
        encoding="utf-8",
    )
    (tmp_path / "models" / "openrouter.json").write_text(
        json.dumps({"models": ["openai/gpt-4o"], "pricing": {"openai/gpt-4o": [2.50, 10.00]}}),
        encoding="utf-8",
    )
    assert lookup_price("openai/gpt-4o", "openrouter") == (2.5, 10.0)
    assert lookup_price("openai/gpt-4o", "aaa_cheap") == (0.1, 0.2)
    budget = BudgetTracker(max_usd=100.0, max_tokens_fallback=-1, max_percent=-1)
    budget.note_route("openai/gpt-4o", "openrouter")
    budget.record(
        model="openai/gpt-4o",
        input_tokens=1_000_000,
        output_tokens=1_000_000,
        cache_read_tokens=0,
        cache_creation_tokens=0,
    )
    assert budget.estimate_usd()[0] == pytest.approx(12.5)
