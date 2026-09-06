# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Candidate-ranking core shared by `--parallel`'s auto-compare (`app.parallel`)
and the standalone `sessions compare` (`ui/cli/sessions_cmds.py`): rank candidates (judge
via the reviewer model when one is built, else the deterministic mechanical
fallback) and print the ranked table. One implementation so the two callers can
never drift.

The reviewer-provider wiring and the `judging...` in-flight status are injected
by the caller (ui/cli supplies the console spinner + the role-provider builder),
so `app` never imports `ui`. A caller that shows nothing passes
`contextlib.nullcontext` as *judging_status*.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from contextlib import AbstractContextManager
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from agent6.app._setup import budget_tracker
from agent6.app.reporter import STDIO_REPORTER, Reporter
from agent6.budget import BudgetTracker
from agent6.config import Config
from agent6.providers import Provider, ProviderError, TranscriptSink
from agent6.sessions.manifest import ManifestError, read_manifest
from agent6.viewmodel.format import format_usd
from agent6.workflows.judge import CandidateBrief, JudgeError, compare, mechanical_ranking

# The reviewer provider the judge call uses, built by the caller from the
# configured `reviewer` role (ui/cli wires it via `build_role_provider`).
BuildProvider = Callable[[Config, TranscriptSink, BudgetTracker], Provider]
# A no-arg context manager shown around the (~50-60s, otherwise silent) judge
# call. ui/cli supplies the console spinner; `nullcontext` shows nothing.
JudgingStatus = Callable[[], AbstractContextManager[None]]


@dataclass(frozen=True, slots=True)
class RankOutcome:
    """`rank()`'s result: candidates best-first, plus which path produced them.

    `ranked_by` is `"judge"` only when the reviewer model actually produced
    the order, else `"mechanical"` -- the honest signal the compare stamp
    records (`CompareStamp.ranked_by`, which stays a lenient `str` for
    reads of history). `rationale` is empty on the mechanical path.
    `judge_cost_usd` is the judge call's estimated spend, real money even
    when a failed judge fell back to the mechanical ranking; it is 0.0 only
    when no judge call was made. `judge_cost_partial` marks it a lower bound (the
    reviewer model is unpriced and reported no cost), the same flag
    `format_usd` renders as `~`.
    """

    ranking: tuple[str, ...]
    rationale: str
    ranked_by: Literal["judge", "mechanical"]
    judge_cost_usd: float = 0.0
    judge_cost_partial: bool = False


def manifest_task(session_dir: Path, fallback: str) -> str:
    """The run's own recorded `user_task`, else *fallback*."""
    try:
        manifest = read_manifest(session_dir)
    except ManifestError:
        return fallback
    return manifest.user_task or fallback


def rank(
    cfg: Config,
    candidates: list[CandidateBrief],
    *,
    transcript_dir: Path,
    build_provider: BuildProvider,
    judging_status: JudgingStatus,
    max_usd: float | None = None,
    reporter: Reporter = STDIO_REPORTER,
) -> RankOutcome:
    """Rank candidates best-first. Use the configured reviewer model as the
    compare judge when one resolves; fall back to the deterministic mechanical
    ranking when it is unset, there is only one candidate, or the judge call
    fails (see `RankOutcome.ranked_by`).

    *max_usd* caps the judge like one more lane (the fan-out advertises
    "$X/lane x N + judge"); None falls back to the config budget."""
    reviewer = cfg.models.resolve("reviewer")
    if len(candidates) > 1 and reviewer is not None:
        sink = TranscriptSink(transcript_dir)
        budget = budget_tracker(cfg, max_usd=max_usd)
        try:
            provider: Provider = build_provider(cfg, sink, budget)
            with judging_status():
                verdict = compare(provider, reviewer.model, candidates)
            spent, unknown = budget.estimate_usd()
            return RankOutcome(verdict.ranking, verdict.rationale, "judge", spent, unknown)
        except (ProviderError, JudgeError) as exc:
            # A configured reviewer that fails must not degrade to the mechanical
            # table silently: say so, so the report isn't mistaken for a judged one.
            # Failed judge attempts still bill; carry and report what they spent.
            detail = str(exc).splitlines()[0] if str(exc).strip() else exc.__class__.__name__
            spent, unknown = budget.estimate_usd()
            spent_s = (
                f"; judge spend {format_usd(spent, partial=unknown)}"
                if spent > 0 or unknown
                else ""
            )
            reporter.err(f"judge failed ({detail}); ranked mechanically{spent_s}")
            return RankOutcome(mechanical_ranking(candidates), "", "mechanical", spent, unknown)
    return RankOutcome(mechanical_ranking(candidates), "", "mechanical")


def verify_word(verify_ok: bool | None) -> str:
    """A candidate's gate verdict as the report and the journal word it."""
    return "passed" if verify_ok else "failed" if verify_ok is False else "no-verify"


def print_ranked_candidates(
    candidates: list[CandidateBrief],
    outcome: RankOutcome,
    *,
    merged_into: Mapping[str, str] | None = None,
    reporter: Reporter = STDIO_REPORTER,
) -> None:
    """Print the ranked table (best first) + a `sessions merge` line per candidate
    (or where it is already merged, per *merged_into*), a total-spend line
    (candidate costs plus any judge cost), then the judge's rationale if there
    is one. Prints nothing when the ranking is empty."""
    if not outcome.ranking:
        return
    by_id = {c.session_id: c for c in candidates}
    reporter.out("ranked candidates (best first):")
    for rnk, rid in enumerate(outcome.ranking, start=1):
        c = by_id[rid]
        verify = verify_word(c.verify_ok)
        into = (merged_into or {}).get(rid, "")
        landing = f"merged into {into}" if into else f"merge with: agent6 sessions merge {rid}"
        reporter.out(f"  {rnk}. {rid}  {verify:<9} {format_usd(c.cost_usd)}   {landing}")
    if len(candidates) > 1:
        cand_total = sum(c.cost_usd for c in candidates)
        judge = outcome.judge_cost_usd
        partial = outcome.judge_cost_partial
        if judge > 0 or partial:
            reporter.out(
                f"total: candidates {format_usd(cand_total)}"
                f" + judge {format_usd(judge, partial=partial)}"
                f" = {format_usd(cand_total + judge, partial=partial)}"
            )
        else:
            reporter.out(f"total: candidates {format_usd(cand_total)}")
    if outcome.rationale:
        reporter.out(f"\njudge: {outcome.rationale}")
