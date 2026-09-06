# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The two-ledger budget: metered calls bound by max_usd, unmetered calls by
max_tokens_fallback; -1 = unlimited, 0 = refuse that ledger, > 0 = the cap."""

from __future__ import annotations

import json

import pytest

from agent6.budget import BudgetExceeded, BudgetTracker, PlanUsage, PlanWindow


@pytest.fixture(autouse=True)
def price_cache(monkeypatch: pytest.MonkeyPatch, tmp_path_factory: pytest.TempPathFactory) -> None:
    """Priced: claude-sonnet-4-5. Anything else is unpriced (no static table)."""
    cache = tmp_path_factory.mktemp("price-cache")
    (cache / "models").mkdir()
    (cache / "models" / "testprovider.json").write_text(
        json.dumps({"models": [], "pricing": {"claude-sonnet-4-5": [3.0, 15.0]}}),
        encoding="utf-8",
    )
    monkeypatch.setenv("AGENT6_CACHE_HOME", str(cache))


def _rec(bt: BudgetTracker, model: str, tokens_in: int, tokens_out: int, cost: float = 0.0) -> None:
    bt.record(
        model=model,
        input_tokens=tokens_in,
        output_tokens=tokens_out,
        cache_read_tokens=0,
        cache_creation_tokens=0,
        cost_usd=cost,
    )


def test_unmetered_calls_count_only_against_the_fallback() -> None:
    # local-model is unpriced and reports no cost: its tokens land in the
    # fallback ledger; the USD meter stays at $0 and never trips.
    bt = BudgetTracker(max_usd=0.01, max_tokens_fallback=1_000, max_percent=-1)
    _rec(bt, "local-model", 400, 300)
    bt.check()  # 700 < 1000 and $0 < $0.01: both ledgers have room
    _rec(bt, "local-model", 200, 200)
    with pytest.raises(BudgetExceeded, match="fallback"):
        bt.check()  # 1100 >= 1000


def test_metered_calls_count_only_against_max_usd() -> None:
    # A priced model never touches the fallback ledger, however many tokens.
    bt = BudgetTracker(max_usd=1.0, max_tokens_fallback=100, max_percent=-1)
    _rec(bt, "claude-sonnet-4-5", 5_000, 5_000)  # >> fallback cap, but metered
    bt.check()  # fallback ledger untouched; ~$0.09 < $1
    _rec(bt, "claude-sonnet-4-5", 250_000, 20_000)  # ~$1.05 more -> over $1 total
    with pytest.raises(BudgetExceeded, match="USD"):
        bt.check()


def test_reported_cost_makes_an_unpriced_model_metered() -> None:
    # A gateway-reported per-call cost is real billing: the call is metered
    # even with no table price, so the fallback ledger stays empty.
    bt = BudgetTracker(max_usd=1.0, max_tokens_fallback=100, max_percent=-1)
    _rec(bt, "exotic-model", 5_000, 5_000, cost=0.02)
    bt.check()


def test_minus_one_means_unlimited_in_both_ledgers() -> None:
    bt = BudgetTracker(max_usd=-1, max_tokens_fallback=-1, max_percent=-1)
    _rec(bt, "claude-sonnet-4-5", 10_000_000, 1_000_000)  # ~$45 metered
    _rec(bt, "local-model", 50_000_000, 1_000_000)  # 51M unmetered tokens
    bt.check()


def test_zero_fallback_refuses_any_unmetered_call() -> None:
    # max_tokens_fallback = 0: zero unmetered tokens allowed -- the strict
    # "never run an unmeterable model" promise, enforced as a runtime backstop.
    bt = BudgetTracker(max_usd=10.0, max_tokens_fallback=0, max_percent=-1)
    _rec(bt, "local-model", 1, 0)
    with pytest.raises(BudgetExceeded, match="unmetered"):
        bt.check()


def test_zero_usd_refuses_any_metered_call() -> None:
    # max_usd = 0: a run-nothing-metered policy (local-only rig).
    bt = BudgetTracker(max_usd=0.0, max_tokens_fallback=1_000_000, max_percent=-1)
    _rec(bt, "claude-sonnet-4-5", 10, 10)
    with pytest.raises(BudgetExceeded, match="USD"):
        bt.check()


def test_a_plan_call_zeroes_only_its_own_calls_not_the_model_id() -> None:
    """One model id reaches both a subscription provider and a paid API (a
    review seat, a machine pin), and the ledger buckets by id: the plan call's
    authoritative $0 stood for the whole bucket, so the API dollars under that
    id left the receipt AND the ceiling they were supposed to bind."""
    bt = BudgetTracker(max_usd=1.0, max_tokens_fallback=-1, max_percent=-1)
    bt.record(
        model="claude-sonnet-4-5",
        input_tokens=10,
        output_tokens=10,
        cache_read_tokens=0,
        cache_creation_tokens=0,
        plan_usage=PlanUsage(windows=(PlanWindow("primary", 12.0, 300, 0.0),)),
    )
    _rec(bt, "claude-sonnet-4-5", 250_000, 20_000)  # the SAME id, on the paid API

    assert bt.estimate_usd()[0] == pytest.approx(1.05, abs=0.01)
    with pytest.raises(BudgetExceeded, match="USD"):
        bt.check()
    summary = bt.format_summary()
    assert "$1.05" in summary and "(subscription)" not in summary


def test_a_pure_subscription_model_still_costs_an_authoritative_zero() -> None:
    """Plan calls are not billed per token: an unpriced one reads $0, not "$?
    (unknown price)", and never draws on the fallback ledger."""
    bt = BudgetTracker(max_usd=1.0, max_tokens_fallback=100, max_percent=-1)
    for _ in range(3):
        bt.record(
            model="unpriced-plan-model",
            input_tokens=5_000,
            output_tokens=5_000,
            cache_read_tokens=0,
            cache_creation_tokens=0,
            plan_usage=PlanUsage(windows=(PlanWindow("primary", 12.0, 300, 0.0),)),
        )

    bt.check()
    assert bt.estimate_usd() == (0.0, False)
    assert "(subscription)" in bt.format_summary()


def test_fraction_remaining_tracks_the_tighter_ledger() -> None:
    bt = BudgetTracker(max_usd=1.0, max_tokens_fallback=1_000, max_percent=-1)
    _rec(bt, "local-model", 500, 400)  # fallback 90% used; USD 0%
    assert bt.fraction_remaining() == pytest.approx(0.1, abs=0.01)
