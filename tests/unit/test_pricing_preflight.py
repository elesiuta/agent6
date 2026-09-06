# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The preflight fetches the OpenRouter pricing catalog for bare claude-* ids.

Prices for direct-Anthropic ids live only in that catalog (pricing's alias);
a config with just [providers.anthropic] refreshed nothing that carries
prices, so the $ cap ran unpriced on a cold cache."""

from __future__ import annotations

import pytest

from agent6.app import _setup
from agent6.config import Config


def _cfg(model: str, provider_block: dict[str, object]) -> Config:
    return Config.model_validate(
        {
            "providers": provider_block,
            "models": {"worker": {"provider": next(iter(provider_block)), "model": model}},
        }
    )


def test_claude_only_config_refreshes_the_catalog(monkeypatch: pytest.MonkeyPatch) -> None:
    called: list[bool] = []

    def _key(*_a: object, **_k: object) -> str:
        return "sk-test"

    def _models(*_a: object, **_k: object) -> list[str]:
        return []

    monkeypatch.setattr(_setup, "refresh_pricing_catalog", lambda: called.append(True))
    monkeypatch.setattr(_setup, "load_secrets", dict)
    monkeypatch.setattr(_setup, "resolve_api_key", _key)
    monkeypatch.setattr(_setup, "list_models", _models)
    cfg = _cfg(
        "claude-opus-5",
        {"anthropic": {"api_format": "anthropic", "api_key_env": "X_KEY"}},
    )
    assert _setup.check_provider_keys(cfg) is None
    assert called, "bare claude-* with no openrouter provider must refresh the catalog"


def test_openrouter_config_does_not_double_refresh(monkeypatch: pytest.MonkeyPatch) -> None:
    called: list[bool] = []

    def _key(*_a: object, **_k: object) -> str:
        return "sk-test"

    def _models(*_a: object, **_k: object) -> list[str]:
        return []

    monkeypatch.setattr(_setup, "refresh_pricing_catalog", lambda: called.append(True))
    monkeypatch.setattr(_setup, "load_secrets", dict)
    monkeypatch.setattr(_setup, "resolve_api_key", _key)
    monkeypatch.setattr(_setup, "list_models", _models)
    # A BARE claude-* id through openrouter: the one shape where the guard
    # decides (a non-claude model skips the refresh before the guard is read).
    cfg = _cfg(
        "claude-opus-5",
        {
            "openrouter": {
                "api_format": "openai",
                "api_key_env": "X_KEY",
                "base_url": "https://openrouter.ai/api/v1",
            }
        },
    )
    assert _setup.check_provider_keys(cfg) is None
    assert not called, "a configured openrouter provider already refreshes with a key"


def test_a_review_seat_provider_is_key_checked(monkeypatch: pytest.MonkeyPatch) -> None:
    """A seat pinning `persona@prov/model` used to skip key preflight: the run
    started, mutated state, and died only when the seat was constructed."""

    def _no_key(*_a: object, **_k: object) -> str:
        return ""

    monkeypatch.setattr(_setup, "load_secrets", dict)
    monkeypatch.setattr(_setup, "resolve_api_key", _no_key)
    cfg = Config.model_validate(
        {
            "providers": {
                "main": {"api_format": "anthropic", "auth_style": "none"},
                "seatp": {"api_format": "anthropic", "api_key_env": "SEAT_KEY"},
            },
            "models": {
                "worker": {"provider": "main", "model": "m"},
                "reviewer": {"provider": "main", "model": "m"},
            },
            "review": {"seats": ["security@seatp/judge-1"]},
        }
    )
    err = _setup.check_provider_keys(cfg)
    assert err is not None and "seatp" in err


def test_a_seat_naming_an_absent_provider_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_setup, "load_secrets", dict)
    cfg = Config.model_validate(
        {
            "providers": {"main": {"api_format": "anthropic", "auth_style": "none"}},
            "models": {"worker": {"provider": "main", "model": "m"}},
            "review": {"seats": ["security@ghost/judge-1"]},
        }
    )
    err = _setup.check_provider_keys(cfg)
    assert err is not None and "ghost" in err


def test_machine_state_pins_ride_the_same_preflight(monkeypatch: pytest.MonkeyPatch) -> None:
    """The machine call site passes per-state provider pins as extras; an
    absent pinned provider refuses before any state runs."""
    monkeypatch.setattr(_setup, "load_secrets", dict)
    cfg = Config.model_validate(
        {
            "providers": {"main": {"api_format": "anthropic", "auth_style": "none"}},
            "models": {"worker": {"provider": "main", "model": "m"}},
        }
    )
    err = _setup.check_provider_keys(cfg, extra_providers=["pinned-ghost"])
    assert err is not None and "pinned-ghost" in err
    assert _setup.check_provider_keys(cfg) is None


def test_budget_preflight_prices_a_seat_pinned_model() -> None:
    """A seat's pinned model joins the reachable set: with unmetered calls
    refused, an unpriced seat model refuses up front instead of mid-review."""
    from agent6.app.preflight import budget_preflight

    cfg = Config.model_validate(
        {
            "providers": {"main": {"api_format": "anthropic", "auth_style": "none"}},
            "models": {"worker": {"provider": "main", "model": "claude-opus-5"}},
            "review": {"seats": ["security@main/very-unpriced-model"]},
            "budget": {"max_tokens_fallback": 0},
        }
    )
    err = budget_preflight(cfg)
    assert err is not None and "very-unpriced-model" in err


def test_chatgpt_provider_without_sign_in_is_refused_statically(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A routed chatgpt provider with no stored OAuth sign-in fails the
    preflight (naming `agent6 connect chatgpt`), not mid-setup after state
    exists; with tokens stored it passes without any key lookup."""
    monkeypatch.setattr(_setup, "load_secrets", dict)
    cfg = _cfg("gpt-5-codex", {"chatgpt": {"api_format": "chatgpt"}})
    err = _setup.check_provider_keys(cfg)
    assert err is not None and "agent6 connect chatgpt" in err

    def stored(*_a: object, **_k: object) -> object:
        return object()

    monkeypatch.setattr(_setup, "load_oauth_tokens", stored)
    listed: list[str] = []

    def fake_list(name: str, *_a: object, **_k: object) -> list[str]:
        listed.append(name)
        return []

    monkeypatch.setattr(_setup, "list_models", fake_list)
    assert _setup.check_provider_keys(cfg) is None
    assert listed == ["chatgpt"]  # the signed-in path refreshes the listing


def test_plan_metered_routes_skip_the_fallback_note(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A model routed through a ChatGPT plan is percent-metered: the
    unpriced-fallback note must not claim the token ledger bounds it, and
    max_percent = 0 refuses it up front like the sibling zeros."""
    from agent6.app.preflight import budget_preflight

    cfg = _cfg("gpt-5.6-sol", {"chatgpt": {"api_format": "chatgpt"}})
    assert budget_preflight(cfg) is None
    out = capsys.readouterr().err
    assert "fallback tokens" not in out
    assert "draws on a subscription plan" in out

    refused = Config.model_validate(
        {
            "providers": {"chatgpt": {"api_format": "chatgpt"}},
            "models": {"worker": {"provider": "chatgpt", "model": "gpt-5.6-sol"}},
            "budget": {"max_percent": 0},
        }
    )
    err = budget_preflight(refused)
    assert err is not None and "max_percent is 0" in err


def test_claude_code_routes_are_plan_metered(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A model routed through a claude_code provider is percent-metered like a
    chatgpt one: the plan note, the max_percent = 0 refusal, and no OpenRouter
    catalog refresh for its bare claude-* id (an authoritative $0 needs no price)."""
    from agent6.app.preflight import budget_preflight

    cfg = _cfg("claude-haiku-4-5", {"claude": {"api_format": "claude_code"}})
    assert budget_preflight(cfg) is None
    out = capsys.readouterr().err
    assert "fallback tokens" not in out
    assert "'claude-haiku-4-5' draws on a subscription plan" in out

    refused = Config.model_validate(
        {
            "providers": {"claude": {"api_format": "claude_code"}},
            "models": {"worker": {"provider": "claude", "model": "claude-haiku-4-5"}},
            "budget": {"max_percent": 0},
        }
    )
    err = budget_preflight(refused)
    assert err is not None and "max_percent is 0" in err and "claude-haiku-4-5" in err

    called: list[bool] = []
    monkeypatch.setattr(_setup, "refresh_pricing_catalog", lambda: called.append(True))
    monkeypatch.setattr(_setup, "load_secrets", dict)
    assert _setup.check_provider_keys(cfg) is None
    assert called == []


def test_machine_pins_carry_their_provider_into_the_notes(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A per-state (provider, model) pin routes the note correctly: a
    chatgpt-pinned model is plan-metered, never 'bounded by fallback
    tokens' (the pin's provider was dropped and the note lied)."""
    from agent6.app.preflight import budget_preflight

    cfg = _cfg(
        "anthropic-model",
        {"anthropic": {"api_format": "anthropic"}, "chatgpt": {"api_format": "chatgpt"}},
    )
    assert budget_preflight(cfg, extra_routes=[("chatgpt", "gpt-5.6-sol")]) is None
    err = capsys.readouterr().err
    assert "'gpt-5.6-sol' draws on a subscription plan" in err
    assert "gpt-5.6-sol' ha" not in err.replace("draws on", "")  # not in the fallback note
