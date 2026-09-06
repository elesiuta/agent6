# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""`probe_provider_key` — the `agent6 connect` key-validation probe."""

from __future__ import annotations

from typing import Any

import httpx2
import pytest

from agent6.config import AnthropicProviderEntry, OpenAIProviderEntry
from agent6.models.cache import probe_provider_key


class _FakeResp:
    def __init__(self, status_code: int, payload: Any = None) -> None:
        self.status_code = status_code
        self._payload = payload if payload is not None else {"data": []}

    def json(self) -> Any:
        return self._payload


def _patch_get(monkeypatch: pytest.MonkeyPatch, resp_or_exc: Any) -> list[str]:
    """Patch httpx2.get; return a list that records the URL(s) hit."""
    urls: list[str] = []

    def _get(url: str, **_kw: Any) -> _FakeResp:
        urls.append(url)
        if isinstance(resp_or_exc, Exception):
            raise resp_or_exc
        return resp_or_exc

    monkeypatch.setattr("agent6.models.cache.httpx2.get", _get)
    return urls


def test_probe_ok_with_models(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_get(monkeypatch, _FakeResp(200, {"data": [{"id": "a"}, {"id": "b"}]}))
    r = probe_provider_key(AnthropicProviderEntry(api_format="anthropic"), "sk-ant-real")
    assert r.ok and r.status == "ok"
    assert "2 models" in r.detail


def test_probe_ok_empty_list(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_get(monkeypatch, _FakeResp(200, {"data": []}))
    r = probe_provider_key(OpenAIProviderEntry(api_format="openai"), "sk-real")
    assert r.ok and r.status == "ok"
    assert "accepted the key" in r.detail


@pytest.mark.parametrize("code", [401, 403])
def test_probe_auth_failed(monkeypatch: pytest.MonkeyPatch, code: int) -> None:
    _patch_get(monkeypatch, _FakeResp(code))
    r = probe_provider_key(AnthropicProviderEntry(api_format="anthropic"), "sk-ant-bad")
    assert not r.ok and r.status == "auth_failed"
    assert str(code) in r.detail


def test_probe_other_4xx_5xx_is_unreachable(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_get(monkeypatch, _FakeResp(500))
    r = probe_provider_key(OpenAIProviderEntry(api_format="openai"), "sk-real")
    assert not r.ok and r.status == "unreachable"


def test_probe_network_error_is_unreachable(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_get(monkeypatch, httpx2.ConnectError("refused"))
    r = probe_provider_key(OpenAIProviderEntry(api_format="openai"), "sk-real")
    assert not r.ok and r.status == "unreachable"


def test_probe_openrouter_uses_auth_gated_key_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    # OpenRouter's /models is public, so the probe must hit /key (auth-gated).
    urls = _patch_get(monkeypatch, _FakeResp(200, {"data": {"label": "k"}}))
    entry = OpenAIProviderEntry(api_format="openai", base_url="https://openrouter.ai/api/v1")
    r = probe_provider_key(entry, "sk-or-real")
    assert r.ok
    assert urls == ["https://openrouter.ai/api/v1/key"]


def test_probe_non_openrouter_uses_models(monkeypatch: pytest.MonkeyPatch) -> None:
    urls = _patch_get(monkeypatch, _FakeResp(200, {"data": []}))
    entry = OpenAIProviderEntry(api_format="openai", base_url="https://api.openai.com/v1")
    probe_provider_key(entry, "sk-real")
    assert urls == ["https://api.openai.com/v1/models"]


def test_probe_openrouter_match_is_host_not_substring(monkeypatch: pytest.MonkeyPatch) -> None:
    # A proxy whose URL merely CONTAINS "openrouter.ai" (in the path) must use
    # /models, not OpenRouter's /key. The real host (and subdomains) use /key.
    urls = _patch_get(monkeypatch, _FakeResp(200, {"data": []}))
    proxy = OpenAIProviderEntry(
        api_format="openai", base_url="https://myproxy.com/openrouter.ai/v1"
    )
    probe_provider_key(proxy, "sk")
    assert urls == ["https://myproxy.com/openrouter.ai/v1/models"]
    urls.clear()
    sub = OpenAIProviderEntry(api_format="openai", base_url="https://gateway.openrouter.ai/api/v1")
    probe_provider_key(sub, "sk")
    assert urls == ["https://gateway.openrouter.ai/api/v1/key"]


def test_probe_unsupported_deployment_skips_network(monkeypatch: pytest.MonkeyPatch) -> None:
    # Vertex/Azure have no uniform /models; the probe must NOT make a request.
    urls = _patch_get(monkeypatch, _FakeResp(200))
    entry = AnthropicProviderEntry(
        api_format="anthropic",
        deployment="vertex",
        base_url="https://example.com",
        token_command=("true",),
    )
    r = probe_provider_key(entry, "tok")
    assert r.status == "unsupported"
    assert urls == []  # no network call
