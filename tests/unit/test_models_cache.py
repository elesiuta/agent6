# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Tests for the live + cached provider model listing (agent6.models.cache)."""

from __future__ import annotations

import json
import time
from pathlib import Path

import httpx2
import pytest

from agent6.config import AnthropicProviderEntry, OpenAIProviderEntry
from agent6.models import cache as models_cache
from agent6.models import registry as models_registry


@pytest.fixture
def cache_home(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    monkeypatch.setenv("AGENT6_CACHE_HOME", str(tmp_path / "cache"))
    return tmp_path / "cache"


def _ok_response(ids: list[str]) -> object:
    def _get(url: str, headers: dict[str, str], timeout: float) -> httpx2.Response:
        return httpx2.Response(
            200, json={"data": [{"id": i} for i in ids]}, request=httpx2.Request("GET", url)
        )

    return _get


def test_fetches_and_caches_openai(cache_home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(httpx2, "get", _ok_response(["gpt-x", "gpt-y"]))
    entry = OpenAIProviderEntry(api_format="openai", base_url="https://api.openai.com/v1")
    out = models_cache.list_models("openai", entry, "sk-test")
    assert out == ["gpt-x", "gpt-y"]
    cached = json.loads((cache_home / "models" / "openai.json").read_text())
    assert cached["models"] == ["gpt-x", "gpt-y"]


def test_fresh_cache_skips_network(cache_home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = cache_home / "models" / "anthropic.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"models": ["claude-cached"]}), encoding="utf-8")

    def _boom(*a: object, **k: object) -> httpx2.Response:
        raise AssertionError("network must not be hit when cache is fresh")

    monkeypatch.setattr(httpx2, "get", _boom)
    entry = AnthropicProviderEntry(api_format="anthropic")
    assert models_cache.list_models("anthropic", entry, "sk-test") == ["claude-cached"]


def test_stale_cache_used_on_network_error(
    cache_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = cache_home / "models" / "openrouter.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"models": ["stale-1"]}), encoding="utf-8")
    # Make the cache look old so a refresh is attempted.
    old = time.time() - 10_000
    import os

    os.utime(path, (old, old))

    def _fail(*a: object, **k: object) -> httpx2.Response:
        raise httpx2.ConnectError("no route")

    monkeypatch.setattr(httpx2, "get", _fail)
    entry = OpenAIProviderEntry(api_format="openai", base_url="https://openrouter.ai/api/v1")
    assert models_cache.list_models("openrouter", entry, None) == ["stale-1"]


def test_no_cache_network_error_returns_empty(
    cache_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _fail(*a: object, **k: object) -> httpx2.Response:
        raise httpx2.ConnectTimeout("slow")

    monkeypatch.setattr(httpx2, "get", _fail)
    entry = OpenAIProviderEntry(api_format="openai")
    assert models_cache.list_models("openai", entry, "sk") == []


def test_never_raises_on_bad_payload(cache_home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def _garbage(*a: object, **k: object) -> httpx2.Response:
        return httpx2.Response(
            200, text="not json", request=httpx2.Request("GET", "http://x/models")
        )

    monkeypatch.setattr(httpx2, "get", _garbage)
    entry = OpenAIProviderEntry(api_format="openai")
    assert models_cache.list_models("openai", entry, "sk") == []


def test_unsafe_provider_name_has_no_cache_path() -> None:
    # A provider name with path separators / traversal must not form a cache
    # path (no writing the cache outside cache_dir/models).
    cache_path = models_cache._cache_path  # pyright: ignore[reportPrivateUsage]
    assert cache_path("../../etc/cron") is None
    assert cache_path("a/b") is None
    assert cache_path("..") is None
    assert cache_path("openrouter") is not None


def test_unsafe_provider_name_still_fetches(
    cache_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # An unsafe name skips the cache but still fetches live (never raises).
    monkeypatch.setattr(httpx2, "get", _ok_response(["m1"]))
    entry = OpenAIProviderEntry(api_format="openai", base_url="https://api.openai.com/v1")
    assert models_cache.list_models("../evil", entry, "sk") == ["m1"]
    assert not (cache_home / "models").exists()  # nothing written outside


# --- context-window reads (file format owned here, policy in registry) -----


def _ok_full(models: list[dict[str, object]]) -> object:
    def _get(url: str, headers: dict[str, str], timeout: float) -> httpx2.Response:
        return httpx2.Response(200, json={"data": models}, request=httpx2.Request("GET", url))

    return _get


def test_caches_context_length_and_reads_it_back(
    cache_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        httpx2,
        "get",
        _ok_full(
            [
                {"id": "vendor/big", "context_length": 200000},
                {"id": "vendor/small", "context_length": 8192},
                {"id": "vendor/nocontext"},  # missing -> absent (unknown beats wrong)
            ]
        ),
    )
    entry = OpenAIProviderEntry(api_format="openai", base_url="https://x/v1")
    models_cache.list_models("vendorx", entry, "k")
    cached = json.loads((cache_home / "models" / "vendorx.json").read_text())
    assert cached["context"] == {"vendor/big": 200000, "vendor/small": 8192}
    # The narrow reader and the registry both see the write (no network).
    assert models_cache.cached_context_window("vendorx", ("vendor/big",)) == 200000
    assert models_registry.context_window("vendorx", "vendor/big") == 200000
    assert models_registry.context_window("vendorx", "vendor/nocontext") is None


def test_fetch_models_live_bypasses_ttl_and_signals_failure(
    cache_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The validation seam: a FRESH cache is ignored (the point is live
    # evidence), success rewrites the cache, and failure returns None -- never
    # a stale fallback -- so the caller can tell evidence from leftovers.
    path = cache_home / "models" / "o.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"models": ["stale-1"]}), encoding="utf-8")
    entry = OpenAIProviderEntry(api_format="openai", base_url="https://x/v1")
    monkeypatch.setattr(httpx2, "get", _ok_response(["fresh-1"]))
    assert models_cache.fetch_models_live("o", entry, None) == ["fresh-1"]
    assert json.loads(path.read_text())["models"] == ["fresh-1"]

    def _fail(*a: object, **k: object) -> httpx2.Response:
        raise httpx2.ConnectError("down")

    monkeypatch.setattr(httpx2, "get", _fail)
    assert models_cache.fetch_models_live("o", entry, None) is None


def test_boolean_context_and_pricing_values_are_rejected(tmp_path: Path) -> None:
    """bool subclasses int: a provider entry with context_length: true cached a
    1-token window (collapsing the compaction thresholds every turn), and
    pricing true would coerce to $1/MTok. Both must read as absent."""
    from agent6.models.cache import (
        _parse_context,  # pyright: ignore[reportPrivateUsage]
        _parse_pricing,  # pyright: ignore[reportPrivateUsage]
    )

    payload = {
        "data": [
            {
                "id": "vendor/x",
                "context_length": True,
                "pricing": {"prompt": True, "completion": "1"},
            },
            {
                "id": "vendor/y",
                "context_length": 200_000,
                "pricing": {"prompt": "3", "completion": "15"},
            },
        ]
    }
    ctx = _parse_context(payload)
    assert "vendor/x" not in ctx
    assert ctx["vendor/y"] == 200_000
    pricing = _parse_pricing(payload)
    assert "vendor/x" not in pricing
    assert "vendor/y" in pricing


def test_pricing_catalog_refresh_prices_a_bare_claude_id(
    cache_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A config with only [providers.anthropic] never fetched the OpenRouter
    catalog, so every claude-* run was honestly-but-needlessly unpriced on a
    cold cache and the $ cap never bound (found live: an unmetered opus run
    inside a container)."""
    from agent6.models import pricing as models_pricing

    def _get(url: str, headers: dict[str, str], timeout: float) -> httpx2.Response:
        assert "openrouter.ai" in url
        assert "Authorization" not in headers, "catalog fetch must be keyless"
        return httpx2.Response(
            200,
            json={
                "data": [
                    {
                        "id": "anthropic/claude-opus-5",
                        "pricing": {"prompt": "0.000005", "completion": "0.000025"},
                    }
                ]
            },
            request=httpx2.Request("GET", url),
        )

    monkeypatch.setattr(httpx2, "get", _get)
    assert models_pricing.lookup_price("claude-opus-5") is None
    models_cache.refresh_pricing_catalog()
    price = models_pricing.lookup_price("claude-opus-5")
    assert price is not None
    assert price.input > 0 and price.output > 0


def test_chatgpt_listing_fetches_with_the_sign_in(
    cache_home: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The chatgpt listing comes from the backend's own /models with the
    stored bearer + account header and a ceiling client_version; hidden
    entries stay out of completion; context windows land in the cache."""
    from agent6.config import ChatGPTProviderEntry
    from agent6.secrets import OAuthTokens, save_oauth_tokens

    monkeypatch.setenv("AGENT6_CONFIG_HOME", str(tmp_path / "g"))
    save_oauth_tokens("chatgpt", OAuthTokens("AT", "RT", time.time() + 3600, "acct-1"))
    seen: dict[str, object] = {}

    def _get(url: str, headers: dict[str, str], timeout: float) -> httpx2.Response:
        seen["url"] = url
        seen["headers"] = headers
        body = {
            "models": [
                {"slug": "gpt-5.6-sol", "context_window": 272000, "visibility": "list"},
                {"slug": "codex-auto-review", "context_window": 272000, "visibility": "hide"},
            ]
        }
        return httpx2.Response(200, json=body, request=httpx2.Request("GET", url))

    monkeypatch.setattr(httpx2, "get", _get)
    entry = ChatGPTProviderEntry(api_format="chatgpt")
    ids = models_cache.list_models("chatgpt", entry, None, ttl_s=0)
    assert ids == ["gpt-5.6-sol"]
    assert str(seen["url"]).endswith("/models?client_version=1.0.0")
    headers = seen["headers"]
    assert isinstance(headers, dict)
    assert headers["authorization"] == "Bearer AT"
    assert headers["chatgpt-account-id"] == "acct-1"
    assert models_cache.cached_context_window("chatgpt", ("gpt-5.6-sol",)) == 272000


def test_chatgpt_listing_without_sign_in_fails_soft(
    cache_home: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from agent6.config import ChatGPTProviderEntry

    monkeypatch.setenv("AGENT6_CONFIG_HOME", str(tmp_path / "empty"))

    def _never(url: str, headers: dict[str, str], timeout: float) -> httpx2.Response:
        pytest.fail("no sign-in: nothing to fetch with")

    monkeypatch.setattr(httpx2, "get", _never)
    entry = ChatGPTProviderEntry(api_format="chatgpt")
    assert models_cache.list_models("chatgpt", entry, None, ttl_s=0) == []
