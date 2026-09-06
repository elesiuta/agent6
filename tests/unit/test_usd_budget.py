# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Config-side budget guards: the `budget_preflight` refusals/notice and the
override path.

USD is a single runtime bound (`BudgetTracker.max_usd`), never a load-time
token conversion. Pricing has no static table: it comes from the
provider-fetched models cache (agent6.models.pricing reads
$XDG_CACHE_HOME/models/*.json). Tests inject prices by writing a real cache
file, exercising the same path production uses.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

PRICED_MODEL = "test/priced-model"
CHEAP_MODEL = "test/cheap-model"


@pytest.fixture
def price_cache(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    cache = tmp_path / "cache"
    (cache / "agent6" / "models").mkdir(parents=True, exist_ok=True)
    (cache / "agent6" / "models" / "testprovider.json").write_text(
        json.dumps(
            {
                "models": [PRICED_MODEL, CHEAP_MODEL],
                "pricing": {PRICED_MODEL: [3.0, 15.0], CHEAP_MODEL: [0.27, 1.10]},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("XDG_CACHE_HOME", str(cache))
    return cache


def _cfg(worker: str, budget: dict[str, Any] | None = None, reviewer: str | None = None) -> Any:
    from agent6.config import Config

    models: dict[str, Any] = {"worker": {"provider": "p", "model": worker}}
    if reviewer is not None:
        models["reviewer"] = {"provider": "p", "model": reviewer}
    return Config.model_validate(
        {
            "providers": {"p": {"api_format": "openai", "base_url": "http://localhost:1"}},
            "models": models,
            **({"budget": budget} if budget else {}),
        }
    )


def test_budget_overrides_write_the_fields_they_name(price_cache: Path) -> None:
    """--max-usd / --max-tokens-fallback override exactly their config fields;
    nothing is derived or ratcheted from one into the other."""
    cfg = _cfg(PRICED_MODEL, budget={"max_usd": 5.0, "max_tokens_fallback": 999_999_999})
    out = cfg.with_budget_overrides(max_usd=50.0)
    assert out.budget.max_usd == 50.0
    assert out.budget.max_tokens_fallback == 999_999_999  # untouched
    out2 = cfg.with_budget_overrides(max_tokens_fallback=7)
    assert out2.budget.max_usd == 5.0
    assert out2.budget.max_tokens_fallback == 7


def test_fallback_zero_refuses_an_unpriced_model(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # max_tokens_fallback = 0 is the strict "never run unmetered" promise:
    # refuse up front when a configured role model has no price data.
    from agent6.app.preflight import budget_preflight

    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "empty-cache"))
    err = budget_preflight(_cfg("nobody/unpriced", budget={"max_tokens_fallback": 0}))
    assert err is not None and "max_tokens_fallback" in err and "nobody/unpriced" in err


def test_fallback_zero_ok_when_all_models_priced(price_cache: Path) -> None:
    from agent6.app.preflight import budget_preflight

    assert budget_preflight(_cfg(PRICED_MODEL, budget={"max_tokens_fallback": 0})) is None


def test_usd_zero_refuses_a_priced_model(price_cache: Path) -> None:
    # max_usd = 0 is the run-nothing-metered policy (local-only rig).
    from agent6.app.preflight import budget_preflight

    err = budget_preflight(_cfg(PRICED_MODEL, budget={"max_usd": 0}))
    assert err is not None and "max_usd" in err and PRICED_MODEL in err


def test_unpriced_model_gets_the_fallback_notice(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """An unpriced role model is not an error: its spend is bounded by the
    fallback ledger, and startup says so once, naming the model."""
    from agent6.app.preflight import budget_preflight

    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "empty-cache"))
    assert budget_preflight(_cfg("nobody/unpriced")) is None
    noted = capsys.readouterr().err
    assert "nobody/unpriced" in noted and "fallback" in noted


def test_priced_models_are_silent(price_cache: Path, capsys: pytest.CaptureFixture[str]) -> None:
    from agent6.app.preflight import budget_preflight

    assert budget_preflight(_cfg(PRICED_MODEL, reviewer=CHEAP_MODEL)) is None
    assert capsys.readouterr().err == ""


@pytest.mark.parametrize(
    ("flag", "value"),
    [("--max-usd", -5.0), ("--max-usd", float("nan")), ("--max-tokens-fallback", -2)],
)
def test_a_bad_budget_flag_refuses_naming_the_flag(flag: str, value: float) -> None:
    """A typo'd flag is the operator's, not a defect in agent6.

    The override went straight into `Config.model_validate`, so its
    `ValidationError` escaped to the last-resort handler: "unexpected
    ValidationError", a crash log in /tmp, an invitation to file a bug, and exit
    1 -- while `config set budget.max_usd -5` refuses cleanly at exit 2. The
    message named `budget.max_usd`, a key the operator never typed."""
    from agent6.app._setup import BudgetOverrides  # pyright: ignore[reportPrivateUsage]
    from agent6.config import Config, ConfigError

    kwargs = {"max_usd": value} if flag == "--max-usd" else {"max_tokens_fallback": int(value)}
    with pytest.raises(ConfigError, match=flag):
        BudgetOverrides(**kwargs).apply(Config())  # pyright: ignore[reportArgumentType]
