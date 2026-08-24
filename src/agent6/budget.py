# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Per-invocation token budget tracker with hard-stop enforcement.

A tracker is created fresh per invocation and never persists, so `resume`
gets a FULL ceiling again: across N resumes real spend can reach N x the cap
(the CLI notes this on resume). Deliberate -- a per-invocation circuit breaker
against runaway spend, not a ledger across a multi-day task.

Budget enforcement is a HARD STOP (not a warning): once a ledger
crosses its cap, the next provider call raises `BudgetExceeded`; the
workflow drains and the process exits with a distinct exit code so
resume tooling can recognise the condition.

Every call is bounded in exactly ONE currency. A call the meter can
price -- provider-reported cost when available, else price x tokens at
the model's fetched rates, cache_read/cache_creation included -- counts
against `max_usd`. A call with neither counts its input+output tokens
against `max_tokens_fallback`. Both caps: -1 unlimited, 0 refuse that
ledger, > 0 the cap (see `[budget]` in config).

This module is import-light (stdlib + agent6.models.pricing, which is itself
stdlib + cache-file reads); both providers wire it in via
constructor.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any

from agent6.models.pricing import lookup_price

# There is NO static price table. Prices come from the provider's own models
# endpoint, fetched + cached by agent6.models.cache and read back through
# agent6.models.pricing.lookup_price. A model without a published price is reported
# as "$? (unknown price)" and the runtime USD ceiling does not bind for it:
# an unknown price is honest, an outdated hardcoded one is wrong.


class BudgetExceeded(Exception):
    """Raised by `BudgetTracker.check()` once a configured limit is exceeded."""


@dataclass(frozen=True, slots=True)
class PlanWindow:
    """One rate-limit window of a subscription plan, as the backend labels
    it (`primary`, `secondary`, a per-model family), in percent of its
    included allowance."""

    name: str
    used_percent: float
    window_minutes: int
    resets_at: float


@dataclass(frozen=True, slots=True)
class PlanUsage:
    """One provider-reported plan-usage reading (subscription providers).

    `windows` holds every window the backend reported, the primary first;
    the BINDING window is the one closest to its cap, and it is what every
    percent-of-plan reading means. Provider specifics (which headers, which
    names) stay in the provider; this shape is the generic N-window meter.
    """

    windows: tuple[PlanWindow, ...]
    # The account's purchased-credit state (the x-codex credits family):
    # after the included window, calls draw on these, which is real money.
    has_credits: bool = False
    credits_unlimited: bool = False
    credits_balance: str = ""
    # The backend's own verdict that a window is exhausted.
    limit_reached: bool = False

    @classmethod
    def single(
        cls,
        used_percent: float,
        window_minutes: int,
        resets_at: float,
        *,
        secondary_used_percent: float | None = None,
        **rest: Any,
    ) -> PlanUsage:
        """A reading with one primary window (and an optional secondary),
        the shape the backend reports for an account with no per-model
        families."""
        windows = [PlanWindow("primary", used_percent, window_minutes, resets_at)]
        if secondary_used_percent is not None:
            windows.append(PlanWindow("secondary", secondary_used_percent, 0, resets_at))
        return cls(windows=tuple(windows), **rest)

    @property
    def binding(self) -> PlanWindow:
        """The window closest to its cap: the one that stops the next call."""
        return max(self.windows, key=lambda w: w.used_percent)

    @property
    def used_percent(self) -> float:
        return self.binding.used_percent

    @property
    def window_minutes(self) -> int:
        return self.binding.window_minutes

    @property
    def resets_at(self) -> float:
        return self.binding.resets_at

    @property
    def window_exhausted(self) -> bool:
        """Whether the next plan-metered call draws past the included
        allowance: any window at 100, or the backend says so."""
        return self.limit_reached or self.used_percent >= 100.0

    @property
    def credits_usd(self) -> float | None:
        """The purchased-credit balance as dollars, None when the backend
        sent none or something that is not a number."""
        try:
            return float(self.credits_balance.lstrip("$").replace(",", ""))
        except ValueError:
            return None


@dataclass(slots=True)
class _ModelTotals:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_creation_tokens: int = 0
    calls: int = 0
    # Sum of provider-reported per-call USD cost, authoritative for the calls
    # that carried `usage.cost` in the response body (today: OpenRouter).
    reported_cost_usd: float = 0.0
    reported_calls: int = 0
    # Token counts for ONLY the calls that reported no cost, banked per call in
    # record(): the price table covers exactly this bucket, so mixed reporting
    # never discards reported dollars and never prices a call twice.
    unreported_input_tokens: int = 0
    unreported_output_tokens: int = 0
    percent_metered: bool = False
    unreported_cache_read_tokens: int = 0
    unreported_cache_creation_tokens: int = 0


@dataclass(frozen=True, slots=True)
class ModelUsage:
    """Immutable per-model usage totals inside a :class:`BudgetSnapshot`."""

    input_tokens: int
    output_tokens: int
    cache_read_tokens: int
    cache_creation_tokens: int
    calls: int
    reported_cost_usd: float
    reported_calls: int
    unreported_input_tokens: int
    unreported_output_tokens: int
    unreported_cache_read_tokens: int
    unreported_cache_creation_tokens: int
    percent_metered: bool = False


@dataclass(frozen=True, slots=True)
class _ModelCost:
    """One model's resolved cost: the USD figure, which sources fed it
    (provider-reported dollars, price-table estimate, or both), and whether it
    is a known under-estimate (some calls priced by neither)."""

    usd: float
    reported: bool
    estimated: bool
    partial: bool = False


def format_usd(usd: float) -> str:
    """A dollar figure as every surface prints it: cents at >= $1, four
    decimals below (a sub-cent cap or spend is never "$0.00")."""
    return f"${usd:.2f}" if usd >= 0.995 else f"${usd:.4f}"


def _model_cost_usd(model: str, t: _ModelTotals | ModelUsage) -> _ModelCost | None:
    """Per-model USD cost: the ONE owner of the pricing arithmetic, shared by
    `_estimate_usd_locked` (the enforced USD ceiling) and `format_summary`
    (the printed figure) so a drifted copy can never misreport spend.

    Provider-reported `usage.cost` is authoritative for the calls that
    carried it; the price table prices ONLY the unreported calls' tokens (the
    `unreported_*` bucket), so the figure is reported + estimated with
    nothing dropped and nothing priced twice. With no table price the reported
    subset still counts, flagged partial (a known lower bound). Returns None
    only when the model has no cached price and reported nothing: the caller
    reports it as unknown.

    Pricing model (Anthropic-accurate):
      fresh input:      price[0]         (already excludes cached portion)
      cache_creation:   price[0] * 1.25  (5-min cache write surcharge)
      cache_read:       price[0] * 0.10  (cache hit discount)
      output:           price[1]
    OpenAI-route models (Kimi etc.) currently report cache_creation_tokens=0
    since the chat-completions usage block has no separate write-surcharge
    field, so the 1.25x branch is a no-op for them.
    """
    if t.percent_metered:
        # Included-plan subscription calls: not billed per token, so the figure is
        # an authoritative $0 -- never "unknown", never table-priced.
        return _ModelCost(0.0, reported=True, estimated=False, partial=t.reported_calls < t.calls)
    reported = t.reported_cost_usd > 0.0
    price = lookup_price(model)
    if price is None:
        if reported:
            # Dropping the reported dollars here would zero real spend out of
            # the estimate and the USD cap; keep them, flagged partial when
            # some calls carried no figure at all.
            return _ModelCost(
                t.reported_cost_usd,
                reported=True,
                estimated=False,
                partial=t.reported_calls < t.calls,
            )
        return None
    in_usd = t.unreported_input_tokens * price[0] / 1e6
    cache_creation_usd = t.unreported_cache_creation_tokens * (price[0] * 1.25) / 1e6
    cache_read_usd = t.unreported_cache_read_tokens * (price[0] * 0.1) / 1e6
    out_usd = t.unreported_output_tokens * price[1] / 1e6
    estimate = in_usd + cache_creation_usd + cache_read_usd + out_usd
    return _ModelCost(t.reported_cost_usd + estimate, reported=reported, estimated=estimate > 0.0)


@dataclass(frozen=True, slots=True)
class BudgetSnapshot:
    """Immutable snapshot of a BudgetTracker's counters at one instant."""

    input_total: int
    output_total: int
    cache_read_total: int
    cache_creation_total: int
    unmetered_tokens: int
    max_usd: float
    max_tokens_fallback: int
    max_percent: float
    plan_latest: PlanUsage | None
    plan_consumed: float
    exhausted: bool
    exhausted_reason: str
    per_model: dict[str, ModelUsage]


@dataclass(slots=True)
class BudgetTracker:
    """Thread-safe spend accumulator: every call is bounded in ONE currency.

    A call the meter can price (provider-reported cost, else a table price for
    its model) counts against `max_usd`; a call carrying a plan-usage reading
    (subscription providers) counts the run's consumed percentage points
    against `max_percent`; a call with neither counts its input+output tokens
    against `max_tokens_fallback`. All caps share one rule: `-1` = unlimited,
    `0` = refuse calls in that ledger, `> 0` = an exclusive ceiling -- the
    call that brings a ledger to or over its cap triggers `BudgetExceeded` on
    the *next* `check()`, so a single call may cross the line but no further
    call is issued.

    `max_percent` meters CONSUMPTION: the rise in the account's reported
    used-percent across this run's observations, accumulated across window
    resets (so a cap above 100 is meaningful for a run spanning windows).
    The reading is account-global, so a concurrent run's spend lands in
    whichever run observes it next -- over-counting, never under.

    The caps are REQUIRED constructor arguments: `[budget]` is where the
    defaults live, and a tracker carrying its own copy could silently meter
    against a different number than the operator set.
    """

    max_usd: float
    max_tokens_fallback: int
    max_percent: float
    # False forgets SAFE (credits refused): unlike the metering caps above,
    # an omitted value can never widen spend, so sites may rely on it.
    allow_paid_credits: bool = False
    _lock: threading.Lock = field(default_factory=threading.Lock)
    _per_model: dict[str, _ModelTotals] = field(default_factory=dict)
    _input_total: int = 0
    _output_total: int = 0
    _cache_read_total: int = 0
    _cache_creation_total: int = 0
    _unmetered_tokens: int = 0
    _exceeded_reason: str = ""
    _plan_latest: PlanUsage | None = None
    # Per window: the last reading, and this run's consumption sawtooth.
    _plan_last_percent: dict[str, float] = field(default_factory=dict)
    _plan_consumed_by_window: dict[str, float] = field(default_factory=dict)
    # Purchased credits observed leaving the account during this run, in
    # dollars (the balance header read as dollars); folds into the USD meter.
    _credits_last_usd: float | None = None
    _credits_spent_usd: float = 0.0

    def record(
        self,
        *,
        model: str,
        input_tokens: int,
        output_tokens: int,
        cache_read_tokens: int,
        cache_creation_tokens: int,
        cost_usd: float = 0.0,
        plan_usage: PlanUsage | None = None,
    ) -> None:
        """Add the usage from a single provider response to the running totals.

        `cost_usd` is the provider-reported USD figure for this single
        call when available (OpenRouter surfaces it as `usage.cost`).
        Pass 0.0 (the default) when no authoritative figure is supplied;
        a table price meters the call instead, and a call with neither
        lands in the fallback token ledger. `plan_usage` marks a
        percent-metered call (subscription providers): it feeds the
        `max_percent` ledger, reports an authoritative $0 for included-plan
        usage, and never drains the fallback ledger; a call that would draw
        on PURCHASED credits refuses unless `allow_paid_credits` is set.
        """
        with self._lock:
            # A gateway is third-party arithmetic: a negative count (malformed
            # or hostile) would SUBTRACT from the ledger and un-exhaust a cap,
            # so the one sink clamps signs. Missing/zero input is the provider
            # layer's fail-closed check; signs are this ledger's.
            input_tokens = max(input_tokens, 0)
            output_tokens = max(output_tokens, 0)
            cache_read_tokens = max(cache_read_tokens, 0)
            cache_creation_tokens = max(cache_creation_tokens, 0)
            cost_usd = max(cost_usd, 0.0)
            totals = self._per_model.setdefault(model, _ModelTotals())
            totals.input_tokens += input_tokens
            totals.output_tokens += output_tokens
            totals.cache_read_tokens += cache_read_tokens
            totals.cache_creation_tokens += cache_creation_tokens
            totals.calls += 1
            if plan_usage is not None:
                # Percent-metered: an authoritative $0 (subscription), so the
                # price table never invents API-rate dollars for these calls.
                totals.reported_calls += 1
                totals.percent_metered = True
                self._note_plan_usage(plan_usage)
            elif cost_usd > 0.0:
                totals.reported_cost_usd += cost_usd
                totals.reported_calls += 1
            else:
                totals.unreported_input_tokens += input_tokens
                totals.unreported_output_tokens += output_tokens
                totals.unreported_cache_read_tokens += cache_read_tokens
                totals.unreported_cache_creation_tokens += cache_creation_tokens
            self._input_total += input_tokens
            self._output_total += output_tokens
            self._cache_read_total += cache_read_tokens
            self._cache_creation_total += cache_creation_tokens
            if plan_usage is not None:
                self._check_plan_ceilings(model, plan_usage)
                return
            metered = cost_usd > 0.0 or lookup_price(model) is not None
            if not metered:
                self._unmetered_tokens += input_tokens + output_tokens
            if metered and self.max_usd == 0.0:
                self._exceeded_reason = (
                    "USD budget is 0: metered calls are refused"
                    f" (model {model!r} is priced; raise [budget].max_usd)"
                )
            elif self.max_usd > 0.0:
                cost, _ = self._estimate_usd_locked()
                if cost >= self.max_usd:
                    self._exceeded_reason = (
                        f"USD budget exhausted: ~{format_usd(cost)} >= {format_usd(self.max_usd)}"
                        " (includes cache_read/cache_creation cost)"
                    )
            if not metered and self.max_tokens_fallback == 0:
                self._exceeded_reason = (
                    f"unmetered call refused: model {model!r} has no reported cost and no"
                    " price data, and [budget].max_tokens_fallback is 0"
                )
            elif (
                self.max_tokens_fallback > 0 and self._unmetered_tokens >= self.max_tokens_fallback
            ):
                self._exceeded_reason = (
                    f"fallback token budget exhausted: {self._unmetered_tokens} unmetered"
                    f" tokens >= {self.max_tokens_fallback}"
                )

    @property
    def _plan_consumed(self) -> float:
        """This run's consumption on its binding window: the most any one
        window moved, since the cap is "no more than N points of the plan"."""
        return max(self._plan_consumed_by_window.values(), default=0.0)

    def _note_plan_usage(self, plan: PlanUsage) -> None:
        """Fold one reading into the per-window consumption sawtooth (lock
        held), and the credit balance into the USD meter.

        A rise since the last reading is this run's consumption (plus any
        concurrent run's -- account-global, over-counting is the safe side);
        a DROP is a window reset, and everything observed after it counts
        from zero. The first reading of a window is its baseline and
        contributes 0. A credit balance that fell since the last reading is
        money this run (or a concurrent one) spent."""
        for w in plan.windows:
            last = self._plan_last_percent.get(w.name)
            if last is not None:
                delta = w.used_percent - last
                self._plan_consumed_by_window[w.name] = self._plan_consumed_by_window.get(
                    w.name, 0.0
                ) + (delta if delta >= 0 else w.used_percent)
            self._plan_last_percent[w.name] = w.used_percent
        balance = plan.credits_usd
        if balance is not None:
            if self._credits_last_usd is not None and balance < self._credits_last_usd:
                self._credits_spent_usd += self._credits_last_usd - balance
            self._credits_last_usd = balance
        self._plan_latest = plan

    def _check_plan_ceilings(self, model: str, plan_usage: PlanUsage) -> None:
        """The plan-metered ceilings, most binding first: the paid-credit
        guard (real money), then the zero refusal, then this run's cap."""
        if (
            plan_usage.has_credits
            and not plan_usage.credits_unlimited
            and not self.allow_paid_credits
            and plan_usage.window_exhausted
        ):
            balance = plan_usage.credits_balance or "unknown"
            self._exceeded_reason = (
                "the plan window is exhausted and the account holds purchased"
                f" credits (balance {balance}): continuing would spend them."
                " Set [budget].allow_paid_credits = true to allow that"
            )
        elif self.max_percent == 0.0:
            self._exceeded_reason = (
                "percent budget is 0: plan-metered calls are refused"
                f" (model {model!r} draws on a subscription plan;"
                " raise [budget].max_percent)"
            )
        elif self.max_percent > 0.0 and self._plan_consumed >= self.max_percent:
            self._exceeded_reason = (
                f"plan budget exhausted: this run consumed ~{self._plan_consumed:.1f}"
                f" percentage points >= max_percent {self.max_percent:g}"
                f" (account at {plan_usage.used_percent:g}% on its"
                f" {plan_usage.binding.name} window)"
            )
        elif self.max_usd > 0.0 and self._credits_spent_usd >= self.max_usd:
            # Purchased credits are dollars: they meter against max_usd like
            # any priced call once allow_paid_credits lets them be spent.
            self._exceeded_reason = (
                f"USD budget exhausted: ~{format_usd(self._credits_spent_usd)} of purchased"
                f" credits spent >= {format_usd(self.max_usd)}"
            )

    def record_plan_preflight(self, model: str, plan: PlanUsage) -> None:
        """A usage reading taken BEFORE the first plan-metered call: the
        baseline every later delta counts from, and the paid-credit guard's
        first look, so a run that would draw on purchased credits refuses at
        its first call instead of after it."""
        with self._lock:
            self._note_plan_usage(plan)
            self._check_plan_ceilings(model, plan)

    def check(self) -> None:
        """Raise `BudgetExceeded` if a prior `record()` crossed a ceiling."""
        with self._lock:
            reason = self._exceeded_reason
        if reason:
            raise BudgetExceeded(reason)

    def is_exhausted(self) -> bool:
        with self._lock:
            return bool(self._exceeded_reason)

    def fraction_remaining(self) -> float:
        """Fraction of the budget still available, in `[0.0, 1.0]`.

        Computed against whichever ceiling is closest to exhaustion, so a run
        that has burned 90% of one ceiling but only 10% of another reports 0.10,
        the conservative, decision-relevant figure. Used by the workflow to
        decide whether a metric plateau is worth quitting on, whether enough
        budget remains to keep pivoting, and when to nudge a graceful wind-down
        (verify + finish_session) before the hard stop.

        Each ledger contributes its own used-fraction (spent/cap for the USD
        meter, unmetered-tokens/cap for the fallback, consumed-points/cap for
        the plan percent); an unlimited (-1) or
        refuse (0) cap contributes nothing -- 0 either never engaged (nothing
        recorded in that ledger) or already tripped `_exceeded_reason`.
        """
        with self._lock:
            if self._exceeded_reason:
                return 0.0
            used = 0.0
            if self.max_usd > 0.0:
                usd_spent, _ = self._estimate_usd_locked()
                used = max(used, usd_spent / self.max_usd)
            if self.max_tokens_fallback > 0:
                used = max(used, self._unmetered_tokens / self.max_tokens_fallback)
            if self.max_percent > 0.0:
                used = max(used, self._plan_consumed / self.max_percent)
        return max(0.0, 1.0 - used)

    def snapshot(self) -> BudgetSnapshot:
        """Immutable snapshot of all counters."""
        with self._lock:
            per_model = {
                model: ModelUsage(
                    input_tokens=t.input_tokens,
                    output_tokens=t.output_tokens,
                    cache_read_tokens=t.cache_read_tokens,
                    cache_creation_tokens=t.cache_creation_tokens,
                    calls=t.calls,
                    reported_cost_usd=t.reported_cost_usd,
                    reported_calls=t.reported_calls,
                    unreported_input_tokens=t.unreported_input_tokens,
                    unreported_output_tokens=t.unreported_output_tokens,
                    unreported_cache_read_tokens=t.unreported_cache_read_tokens,
                    unreported_cache_creation_tokens=t.unreported_cache_creation_tokens,
                    percent_metered=t.percent_metered,
                )
                for model, t in sorted(self._per_model.items())
            }
            return BudgetSnapshot(
                input_total=self._input_total,
                output_total=self._output_total,
                cache_read_total=self._cache_read_total,
                cache_creation_total=self._cache_creation_total,
                unmetered_tokens=self._unmetered_tokens,
                max_usd=self.max_usd,
                max_tokens_fallback=self.max_tokens_fallback,
                max_percent=self.max_percent,
                plan_latest=self._plan_latest,
                plan_consumed=self._plan_consumed,
                exhausted=bool(self._exceeded_reason),
                exhausted_reason=self._exceeded_reason,
                per_model=per_model,
            )

    def estimate_usd(self) -> tuple[float, bool]:
        """Estimate cumulative USD spend across all recorded calls.

        Returns `(usd_total, any_unknown)` where `any_unknown` is True
        when any recorded call could not be priced: a model absent from the
        pricing table, or a priced model with unpriced calls. Either way the
        figure is a lower bound.

        The live TUI cost meter and the in-record USD ceiling read this;
        the end-of-run summary iterates `_model_cost_usd` itself, the same
        arithmetic per model.
        """
        with self._lock:
            return self._estimate_usd_locked()

    def _estimate_usd_locked(self) -> tuple[float, bool]:
        """Cost estimate computed directly from `self._per_model`.

        Assumes `self._lock` is already held (called from both `record` --
        under the lock -- and `estimate_usd`), so it never re-acquires it.
        """
        total_usd = self._credits_spent_usd
        any_unknown = False
        for model, t in self._per_model.items():
            cost = _model_cost_usd(model, t)
            if cost is None:
                any_unknown = True
                continue
            any_unknown = any_unknown or cost.partial
            total_usd += cost.usd
        return total_usd, any_unknown

    def format_summary(self) -> str:
        """Human-facing end-of-run summary with USD estimate where known."""
        snap = self.snapshot()
        lines = ["Token + cost summary:"]
        total_usd = 0.0
        any_unknown = False
        for model, totals in snap.per_model.items():
            cost = _model_cost_usd(model, totals)
            cost_str: str
            if cost is None:
                cost_str = "$? (unknown price)"
                any_unknown = True
            else:
                total_usd += cost.usd
                any_unknown = any_unknown or cost.partial
                if cost.partial:
                    note = " (reported, some calls unpriced)"
                elif cost.reported and cost.estimated:
                    note = " (reported + estimated)"
                elif totals.percent_metered:
                    note = " (subscription)"
                elif cost.reported:
                    note = " (reported)"
                else:
                    note = ""
                cost_str = f"{format_usd(cost.usd)}{note}"
            lines.append(
                f"  {model}: "
                f"in={totals.input_tokens} out={totals.output_tokens} "
                f"cache_r={totals.cache_read_tokens} "
                f"cache_c={totals.cache_creation_tokens} "
                f"calls={totals.calls} {cost_str}"
            )
        usd_cap = "unlimited" if snap.max_usd == -1 else format_usd(snap.max_usd)
        budget_line = (
            f"  TOTAL: in={snap.input_total} out={snap.output_total} "
            f"cost~{format_usd(total_usd)} of {usd_cap}"
        )
        if snap.unmetered_tokens:
            fb_cap = (
                "unlimited" if snap.max_tokens_fallback == -1 else str(snap.max_tokens_fallback)
            )
            budget_line += f" (unmetered: {snap.unmetered_tokens}/{fb_cap} fallback tokens)"
        if any_unknown:
            # The figure is a lower bound; at least one model has no cached
            # provider price (see agent6.models.pricing: no static fallback).
            budget_line += "+ (some models unpriced; figure is a lower bound)"
        lines.append(budget_line)
        if snap.plan_latest is not None:
            lines.append("  " + format_plan_usage(snap))
        if snap.exhausted:
            lines.append(f"  STATUS: BUDGET EXCEEDED ({snap.exhausted_reason})")
        return "\n".join(lines)


def format_plan_usage(snap: BudgetSnapshot) -> str:
    """The one plan-usage line every surface prints for subscription spend.

    Names the account's reported percent, the window, this run's consumed
    points against `max_percent`, and the reset."""
    plan = snap.plan_latest
    assert plan is not None
    minutes = plan.window_minutes
    if minutes >= 1440:
        window = f"{minutes / 1440:g}-day"
    elif minutes >= 60:
        window = f"{minutes / 60:g}-hour"
    else:
        window = f"{minutes}-minute"
    cap = "" if snap.max_percent == -1 else f" of max_percent {snap.max_percent:g}"
    resets_h = max(0.0, (plan.resets_at - time.time()) / 3600)
    which = "" if plan.binding.name == "primary" else f" ({plan.binding.name})"
    return (
        f"plan usage: {plan.used_percent:g}% of the {window} window{which}"
        f" (this run ~{snap.plan_consumed:g} points{cap}; resets in {resets_h:.0f}h)"
        + (
            f"; purchased credits balance {plan.credits_balance or 'present'}"
            if plan.has_credits and not plan.credits_unlimited
            else ""
        )
    )
