# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Tests for the model-capability registry (agent6.models.registry)."""

from __future__ import annotations

from pathlib import Path

import pytest

from agent6.app.providers import resolve_decompose
from agent6.config import Config
from agent6.models import registry as models_registry


@pytest.fixture
def cache_home(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    monkeypatch.setenv("AGENT6_CACHE_HOME", str(tmp_path / "cache"))
    return tmp_path / "cache"


def _cfg(model: str, decompose: str = "auto") -> Config:
    return Config.model_validate(
        {
            "providers": {"o": {"api_format": "openai", "base_url": "https://x/v1"}},
            "models": {"worker": {"provider": "o", "model": model}},
            "prompt": {"decompose": decompose},
        }
    )


def test_context_window_bundled_and_normalized(cache_home: Path) -> None:
    # Bundled table: exact + a dated/tagged id normalised to the canonical key.
    assert models_registry.context_window("anthropic", "claude-sonnet-4-6") == 200_000
    # The 5-family's 1M default: a missing row silently fell back to the fixed
    # 256k thresholds, over-compacting flagship models (caught live on a
    # sonnet-5 dogfood run).
    assert models_registry.context_window("anthropic", "claude-sonnet-5") == 1_000_000
    assert models_registry.context_window("anthropic", "claude-fable-5") == 1_000_000
    assert models_registry.context_window("anthropic", "claude-haiku-4-5-20251001") == 200_000
    assert models_registry.context_window("openrouter", "qwen/qwen3-coder:free") == 1_048_576
    assert models_registry.context_window("openrouter", "vendor/totally-unknown") is None


def test_compaction_thresholds_explicit_override_wins(cache_home: Path) -> None:
    assert models_registry.compaction_thresholds(
        "openrouter", "moonshotai/kimi-k2.6", drop_override=111, summarise_override=222
    ) == (111, 222)


def test_compaction_thresholds_adaptive_from_window(cache_home: Path) -> None:
    drop, summarise = models_registry.compaction_thresholds(
        "openrouter", "moonshotai/kimi-k2.6", drop_override=None, summarise_override=None
    )
    assert drop == int(262_144 * 4 * 0.45)
    # Tier-2 is a near-edge valve: the window minus a 16k-token reserve
    # (pi's reserveTokens shape), not a mid-window fraction.
    assert summarise == (262_144 - 16_384) * 4
    assert drop < summarise  # tier-2 escalates above tier-1


def test_a_small_window_clamps_the_verbatim_tail_and_says_so(
    cache_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The config validator refuses an explicit pair whose verbatim tail alone
    re-triggers tier 2. A small window sizes exactly that pair from the
    80,000-char default, and under it the restart kept NO verbatim turns at
    all -- the freshest tool results, paraphrased away, every restart."""
    from agent6.app.providers import resolve_compaction_thresholds
    from agent6.config import Config, RoleModel

    monkeypatch.setattr(models_registry, "context_window", lambda _p, _m: 32_768)
    logs: list[str] = []

    drop, summarise, keep = resolve_compaction_thresholds(
        Config(), RoleModel(provider="p", model="m"), log=logs.append
    )

    assert drop < summarise
    assert keep == summarise // 2, "the tail must leave room for the restart to shrink"
    assert any("keep_recent_chars 80,000 ->" in line for line in logs), logs

    # A window the default already fits inside keeps the operator's number.
    monkeypatch.setattr(models_registry, "context_window", lambda _p, _m: 262_144)
    assert resolve_compaction_thresholds(Config(), RoleModel(provider="p", model="m"))[2] == 80_000


def test_compaction_thresholds_fixed_fallback_when_unknown(cache_home: Path) -> None:
    # No bundled entry, empty cache -> historical 256k/768k (behaviour preserved).
    assert models_registry.compaction_thresholds(
        "openrouter", "vendor/totally-unknown", drop_override=None, summarise_override=None
    ) == (256_000, 768_000)


# --- decompose capability ---------------------------------------------------


def test_decompose_default_measured_families_only() -> None:
    dd = models_registry.decompose_default
    # The measured winner matches with org prefix, :tag, and a snapshot suffix.
    assert dd("mistralai/mistral-small-3.2-24b-instruct")
    assert dd("mistralai/mistral-small-3.2-24b-instruct:free")
    assert dd("mistral-small-3.2-24b-instruct-2506")
    # Models benched at the ceiling (or never benched) stay off.
    assert not dd("claude-haiku-4-5")
    assert not dd("qwen/qwen3-coder-30b-a3b-instruct")
    assert not dd("vendor/unknown-model")


def test_resolve_decompose_pins_auto_from_registry() -> None:
    win = _cfg("mistralai/mistral-small-3.2-24b-instruct")
    assert resolve_decompose(win, win.models.resolve("worker")).prompt.decompose == "on"
    ceiling = _cfg("claude-haiku-4-5")
    assert resolve_decompose(ceiling, ceiling.models.resolve("worker")).prompt.decompose == "off"
    # Unresolvable model pins off (never leaves "auto" for the engine).
    assert resolve_decompose(_cfg("x"), None).prompt.decompose == "off"


def test_resolve_decompose_explicit_setting_passes_through() -> None:
    forced_on = _cfg("claude-haiku-4-5", decompose="on")
    assert resolve_decompose(forced_on, forced_on.models.resolve("worker")) is forced_on
    forced_off = _cfg("mistralai/mistral-small-3.2-24b-instruct", decompose="off")
    assert resolve_decompose(forced_off, forced_off.models.resolve("worker")) is forced_off


def test_with_decompose_pins_value() -> None:
    assert Config().with_decompose("on").prompt.decompose == "on"
    assert Config().with_decompose("off").prompt.decompose == "off"


def test_resolved_adaptive_values_reports_auto_decompose(cache_home: Path) -> None:
    win = models_registry.resolved_adaptive_values(_cfg("mistralai/mistral-small-3.2-24b-instruct"))
    assert win["prompt.decompose"] == "on"
    ceiling = models_registry.resolved_adaptive_values(_cfg("claude-haiku-4-5"))
    assert ceiling["prompt.decompose"] == "off"
    # Explicitly pinned config has nothing to resolve.
    pinned = models_registry.resolved_adaptive_values(_cfg("claude-haiku-4-5", decompose="on"))
    assert "prompt.decompose" not in pinned
