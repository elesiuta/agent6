# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Tests for agent6.budget — hard-stop token tracker."""

from __future__ import annotations

import json

import pytest

from agent6.budget import BudgetExceeded, BudgetTracker, PlanUsage, format_plan_usage


@pytest.fixture(autouse=True)
def price_cache(monkeypatch: pytest.MonkeyPatch, tmp_path_factory: pytest.TempPathFactory) -> None:
    """Inject prices via a real models-cache file (there is no static table)."""
    cache = tmp_path_factory.mktemp("price-cache")
    (cache / "models").mkdir()
    (cache / "models" / "testprovider.json").write_text(
        json.dumps(
            {
                "models": [],
                "pricing": {
                    "claude-sonnet-4-5": [3.0, 15.0],
                    "claude-sonnet-4-20250514": [3.0, 15.0],
                    "free-or-unpriced": [0.0, 0.0],  # OpenRouter reports 0/0 for some routes
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("AGENT6_CACHE_HOME", str(cache))


def _t(*, fallback: int = 100) -> BudgetTracker:
    # model "m" is unpriced in the fixture cache, so its tokens land in the
    # fallback ledger; max_usd stays unlimited to keep the tests single-ledger.
    return BudgetTracker(max_usd=-1, max_tokens_fallback=fallback, max_percent=-1)


def test_usd_ceiling_counts_cache_tokens_token_caps_would_miss() -> None:
    # Token caps huge (never fire) + fresh input ~0, but cache_creation alone
    # costs > $1: the USD ceiling must catch the overspend the token caps miss.
    t = BudgetTracker(max_usd=1.0, max_tokens_fallback=-1, max_percent=-1)
    # sonnet-4 input $3/M; cache_creation surcharge 1.25x -> $3.75/M.
    # 300k * 3.75/1e6 = $1.125 > $1.
    t.record(
        model="claude-sonnet-4-20250514",
        input_tokens=10,
        output_tokens=10,
        cache_read_tokens=0,
        cache_creation_tokens=300_000,
    )
    with pytest.raises(BudgetExceeded) as exc:
        t.check()
    assert "USD budget" in str(exc.value)


def test_usd_ceiling_off_when_unlimited() -> None:
    # max_usd = -1 (unlimited): the same heavy-cache call trips nothing.
    t = BudgetTracker(max_usd=-1, max_tokens_fallback=-1, max_percent=-1)
    t.record(
        model="claude-sonnet-4-20250514",
        input_tokens=10,
        output_tokens=10,
        cache_read_tokens=0,
        cache_creation_tokens=300_000,
    )
    t.check()  # no raise


def test_negative_usage_never_reduces_the_ledger() -> None:
    """A gateway's counts are third-party arithmetic: negative fields (a
    malformed or hostile payload) subtracted from the totals and un-exhausted
    a cap. The one sink clamps signs, so spend only ever grows."""
    t = _t()
    t.record(
        model="m", input_tokens=5, output_tokens=3, cache_read_tokens=1, cache_creation_tokens=2
    )
    before = t.snapshot()
    t.record(
        model="m",
        input_tokens=-500,
        output_tokens=-500,
        cache_read_tokens=-500,
        cache_creation_tokens=-500,
        cost_usd=-9.0,
    )
    snap = t.snapshot()
    assert snap.input_total == before.input_total
    assert snap.output_total == before.output_total
    assert snap.cache_read_total == before.cache_read_total
    assert snap.cache_creation_total == before.cache_creation_total
    assert t.estimate_usd()[0] >= 0.0


def test_record_accumulates() -> None:
    t = _t()
    t.record(
        model="m", input_tokens=5, output_tokens=3, cache_read_tokens=1, cache_creation_tokens=2
    )
    t.record(
        model="m", input_tokens=4, output_tokens=2, cache_read_tokens=0, cache_creation_tokens=0
    )
    snap = t.snapshot()
    assert snap.input_total == 9
    assert snap.output_total == 5
    assert snap.cache_read_total == 1
    assert snap.cache_creation_total == 2
    assert snap.exhausted is False
    t.check()  # should not raise


def test_fallback_ceiling_hard_stop() -> None:
    # The unmetered ledger sums input+output; the call that reaches the cap
    # exhausts it (exclusive ceiling, enforced on the next check).
    t = _t(fallback=10)
    t.record(
        model="m", input_tokens=7, output_tokens=3, cache_read_tokens=0, cache_creation_tokens=0
    )
    assert t.is_exhausted()
    with pytest.raises(BudgetExceeded, match="fallback token budget"):
        t.check()


def test_per_model_tracking() -> None:
    t = _t(fallback=1000)
    t.record(
        model="a", input_tokens=10, output_tokens=2, cache_read_tokens=0, cache_creation_tokens=0
    )
    t.record(
        model="b", input_tokens=20, output_tokens=4, cache_read_tokens=0, cache_creation_tokens=0
    )
    t.record(
        model="a", input_tokens=5, output_tokens=1, cache_read_tokens=0, cache_creation_tokens=0
    )
    pm = t.snapshot().per_model
    assert pm["a"].input_tokens == 15
    assert pm["a"].calls == 2
    assert pm["b"].input_tokens == 20
    assert pm["b"].calls == 1


def test_format_summary_renders_known_and_unknown_prices() -> None:
    # claude-sonnet-4-5 IS priced by the fixture ($3/$15 per Mtok): the known
    # half must render a real dollar figure, not fall through to "$?" with the
    # priced path unexercised. 1000 in + 100 out = $0.003 + $0.0015 = $0.0045.
    t = _t(fallback=10000)
    t.record(
        model="claude-sonnet-4-5",
        input_tokens=1000,
        output_tokens=100,
        cache_read_tokens=0,
        cache_creation_tokens=0,
    )
    t.record(
        model="totally-fake-model",
        input_tokens=500,
        output_tokens=50,
        cache_read_tokens=0,
        cache_creation_tokens=0,
    )
    summary = t.format_summary()
    assert "claude-sonnet-4-5" in summary
    assert "$0.0045" in summary  # the PRICED path rendered a real figure
    assert "totally-fake-model" in summary
    assert "$? (unknown price)" in summary
    assert "TOTAL:" in summary


def test_format_summary_marks_exhausted() -> None:
    t = _t(fallback=5)
    t.record(
        model="m", input_tokens=10, output_tokens=0, cache_read_tokens=0, cache_creation_tokens=0
    )
    assert "BUDGET EXCEEDED" in t.format_summary()


def test_the_caps_have_one_home() -> None:
    """A bare BudgetTracker() silently used its own 10.0/2M while the config
    supplied the real caps, so changing the config default left an unconfigured
    tracker on the old number. The caps are required now; BudgetConfig is the
    one place the defaults live, and the docs quote it."""
    import inspect
    from pathlib import Path

    from agent6.budget import BudgetTracker
    from agent6.config import BudgetConfig

    params = inspect.signature(BudgetTracker).parameters
    for name in ("max_usd", "max_tokens_fallback", "max_percent"):
        assert params[name].default is inspect.Parameter.empty, name

    cfg = BudgetConfig()
    docs = Path(__file__).resolve().parents[2] / "docs" / "config.md"
    row_usd, row_tokens = "", ""
    for line in docs.read_text(encoding="utf-8").splitlines():
        if line.startswith("| `max_usd`"):
            row_usd = line
        elif line.startswith("| `max_tokens_fallback`"):
            row_tokens = line
    assert f"`{cfg.max_usd}`" in row_usd, row_usd
    assert f"`{cfg.max_tokens_fallback}`" in row_tokens, row_tokens


def test_percent_meter_sawtooth_and_cap() -> None:
    """Plan-metered calls: the first reading is the baseline, rises are the
    run's consumption, a drop (window reset) counts from zero, and the call
    that reaches max_percent trips check(). Plan calls never drain the
    fallback ledger and report an authoritative $0."""
    from agent6.budget import PlanUsage

    t = BudgetTracker(max_usd=10.0, max_tokens_fallback=100, max_percent=10.0)

    def rec(pct: float) -> None:
        t.record(
            model="gpt-5.6-sol",
            input_tokens=1000,
            output_tokens=50,
            cache_read_tokens=0,
            cache_creation_tokens=0,
            plan_usage=PlanUsage.single(used_percent=pct, window_minutes=10080, resets_at=2e9),
        )

    rec(37.0)  # baseline
    rec(40.0)  # +3
    rec(2.0)  # reset: +2
    rec(5.0)  # +3 -> consumed 8
    t.check()  # under the cap of 10
    snap = t.snapshot()
    assert snap.plan_consumed == pytest.approx(8.0)
    assert snap.plan_latest is not None and snap.plan_latest.used_percent == 5.0
    assert snap.unmetered_tokens == 0  # percent-metered, never fallback
    usd, partial = t.estimate_usd()
    assert usd == 0.0 and partial is False  # authoritative $0, not unpriced
    rec(8.0)  # +3 -> consumed 11 >= 10
    with pytest.raises(BudgetExceeded, match="plan budget exhausted"):
        t.check()
    assert "plan usage: 8% of the 7-day window" in t.format_summary()
    assert "(subscription)" in t.format_summary()


def test_percent_zero_refuses_plan_metered_calls() -> None:
    from agent6.budget import PlanUsage

    t = BudgetTracker(max_usd=-1, max_tokens_fallback=-1, max_percent=0)
    t.record(
        model="gpt-5.6-sol",
        input_tokens=10,
        output_tokens=1,
        cache_read_tokens=0,
        cache_creation_tokens=0,
        plan_usage=PlanUsage.single(used_percent=1.0, window_minutes=300, resets_at=2e9),
    )
    with pytest.raises(BudgetExceeded, match="percent budget is 0"):
        t.check()


def _plan_with_credits(used: float, *, unlimited: bool = False) -> PlanUsage:
    return PlanUsage.single(
        used_percent=used,
        window_minutes=10080,
        resets_at=2e9,
        has_credits=True,
        credits_unlimited=unlimited,
        credits_balance="$12.50",
    )


def _record_plan(t: BudgetTracker, plan: PlanUsage) -> None:
    t.record(
        model="gpt-5.6-sol",
        input_tokens=10,
        output_tokens=1,
        cache_read_tokens=0,
        cache_creation_tokens=0,
        plan_usage=plan,
    )


def test_exhausted_window_with_credits_refuses_by_default() -> None:
    """Past the included window, chatgpt calls draw on PURCHASED credits
    (auto top-up can buy more): real money the $0-authoritative stance would
    hide, so the default refuses and names [budget].allow_paid_credits."""
    t = BudgetTracker(max_usd=-1, max_tokens_fallback=-1, max_percent=-1)
    _record_plan(t, _plan_with_credits(100.0))
    with pytest.raises(BudgetExceeded, match="allow_paid_credits"):
        t.check()


def test_credits_spend_allowed_when_opted_in() -> None:
    t = BudgetTracker(max_usd=-1, max_tokens_fallback=-1, max_percent=-1, allow_paid_credits=True)
    _record_plan(t, _plan_with_credits(100.0))
    t.check()


def test_credits_inside_the_included_window_do_not_refuse() -> None:
    t = BudgetTracker(max_usd=-1, max_tokens_fallback=-1, max_percent=-1)
    _record_plan(t, _plan_with_credits(41.0))
    t.check()


def test_unlimited_credits_do_not_refuse() -> None:
    t = BudgetTracker(max_usd=-1, max_tokens_fallback=-1, max_percent=-1)
    _record_plan(t, _plan_with_credits(100.0, unlimited=True))
    t.check()


def test_fraction_remaining_counts_the_plan_percent_ledger() -> None:
    """fraction_remaining consulted only the USD and fallback-token ledgers,
    so a plan-metered run reported ~1.0 remaining until the max_percent hard
    stop: none of the graceful near-budget behaviour (wind-down nudges,
    review gating, metric decisions) ever engaged. The plan ledger
    contributes used = plan_consumed / max_percent like the others."""
    from agent6.budget import BudgetTracker, PlanUsage

    t = BudgetTracker(max_usd=-1.0, max_tokens_fallback=-1, max_percent=5.0)
    t.record(
        model="gpt-5.6-sol",
        input_tokens=10,
        output_tokens=10,
        cache_read_tokens=0,
        cache_creation_tokens=0,
        plan_usage=PlanUsage.single(used_percent=40.0, window_minutes=300, resets_at=0.0),
    )
    t.record(
        model="gpt-5.6-sol",
        input_tokens=10,
        output_tokens=10,
        cache_read_tokens=0,
        cache_creation_tokens=0,
        plan_usage=PlanUsage.single(used_percent=42.5, window_minutes=300, resets_at=0.0),
    )
    # The run consumed ~2.5 points of its 5-point cap.
    assert 0.4 <= t.fraction_remaining() <= 0.6


def test_preflight_reading_seeds_the_baseline_and_guards_credits() -> None:
    """A preflight reading is the baseline the first response's delta counts
    from (no call's spend is invisible), and the paid-credit guard sees it
    before any call; a secondary window at 100 counts as exhausted."""
    t = BudgetTracker(max_usd=1.0, max_tokens_fallback=100, max_percent=-1)
    t.record_plan_preflight(
        "m", PlanUsage.single(used_percent=40.0, window_minutes=10080, resets_at=0)
    )
    t.record(
        model="m",
        input_tokens=1,
        output_tokens=1,
        cache_read_tokens=0,
        cache_creation_tokens=0,
        plan_usage=PlanUsage.single(used_percent=42.0, window_minutes=10080, resets_at=0),
    )
    assert t.snapshot().plan_consumed == 2.0
    guarded = BudgetTracker(max_usd=1.0, max_tokens_fallback=100, max_percent=-1)
    guarded.record_plan_preflight(
        "m",
        PlanUsage.single(
            used_percent=10.0,
            window_minutes=10080,
            resets_at=0,
            has_credits=True,
            secondary_used_percent=100.0,
        ),
    )
    with pytest.raises(BudgetExceeded, match="purchased"):
        guarded.check()


def test_the_binding_window_meters_the_run_whatever_its_name() -> None:
    """Consumption is tracked per window and the cap binds on the one that
    moved most: a per-model family the run burns counts even while the
    primary window barely moves (the window spark burned)."""
    from agent6.budget import PlanWindow

    def reading(primary: float, spark: float) -> PlanUsage:
        return PlanUsage(
            windows=(
                PlanWindow("primary", primary, 10080, 2e9),
                PlanWindow("gpt-5-6-spark", spark, 300, 2e9),
            )
        )

    t = BudgetTracker(max_usd=1.0, max_tokens_fallback=100, max_percent=5.0)
    _record_plan(t, reading(10.0, 40.0))  # baselines
    _record_plan(t, reading(10.5, 43.0))
    assert t.snapshot().plan_consumed == 3.0  # spark moved 3, primary 0.5
    _record_plan(t, reading(11.0, 46.0))
    with pytest.raises(BudgetExceeded, match="gpt-5-6-spark window"):
        t.check()
    # A reset on one window restarts that window's count from zero only.
    t2 = BudgetTracker(max_usd=1.0, max_tokens_fallback=100, max_percent=50.0)
    _record_plan(t2, reading(10.0, 90.0))
    _record_plan(t2, reading(12.0, 1.0))
    assert t2.snapshot().plan_consumed == 2.0


def test_purchased_credit_spend_meters_against_max_usd() -> None:
    """With allow_paid_credits the balance that left the account during the
    run is dollars spent: it counts into the USD estimate and trips max_usd.
    A balance that is not a number meters nothing."""
    t = BudgetTracker(max_usd=1.0, max_tokens_fallback=100, max_percent=-1, allow_paid_credits=True)

    def reading(balance: str) -> PlanUsage:
        return PlanUsage.single(
            used_percent=100.0,
            window_minutes=10080,
            resets_at=2e9,
            has_credits=True,
            credits_balance=balance,
        )

    _record_plan(t, reading("$12.50"))
    _record_plan(t, reading("$11.90"))
    assert t.estimate_usd()[0] == pytest.approx(0.60)
    t.check()
    _record_plan(t, reading("$11.40"))
    with pytest.raises(BudgetExceeded, match="purchased credits spent"):
        t.check()
    opaque = BudgetTracker(
        max_usd=1.0, max_tokens_fallback=100, max_percent=-1, allow_paid_credits=True
    )
    _record_plan(opaque, reading("lots"))
    _record_plan(opaque, reading("fewer"))
    assert opaque.estimate_usd()[0] == 0.0


def test_credits_balance_units_convert_to_usd() -> None:
    """The backend sells credits in 1,000-credit packs at $40 (25 per
    dollar): a bare balance number is credits and converts; a "$"-prefixed
    balance is already dollars and stays as sent. Treating a raw credit
    count as dollars over-reported the balance 25x."""

    def usd(balance: str) -> float | None:
        return PlanUsage.single(
            0.0, 10080, 0.0, has_credits=True, credits_balance=balance
        ).credits_usd

    assert usd("500") == 20.0
    assert usd("1,000") == 40.0
    assert usd("$12.50") == 12.50
    assert usd("") is None
    assert usd("n/a") is None


def test_plan_usage_line_names_a_window_with_no_reported_length() -> None:
    """The backend reports a `secondary` window with `window_minutes` 0; when
    it binds, the line names the window without inventing a length."""
    from agent6.budget import PlanWindow

    t = BudgetTracker(max_usd=1.0, max_tokens_fallback=100, max_percent=-1)
    _record_plan(
        t,
        PlanUsage(
            windows=(PlanWindow("primary", 1.0, 10080, 2e9), PlanWindow("secondary", 3.0, 0, 2e9))
        ),
    )
    line = format_plan_usage(t.snapshot())
    assert "3% of the window (secondary)" in line
    assert "0-minute" not in line
