# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Review-panel seat call + sequential orchestration.

A *seat* is one adversarial reviewer: a single grounded LLM call over the diff
(+ verify result) that returns a structured `ReviewVerdict`. `run_panel` runs
the seats and folds them with the pure `aggregate_verdicts` (in `_panel`).
The grounding that actually prevents false-blocks is enforced in the aggregator,
not here; this module just asks each model for findings in a parseable shape.

Network calls live here (each seat takes an injected `Provider`); the pure
grounding/aggregation stays in `_panel` so it is testable without the network.
"""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any

from agent6.config import ReviewTier
from agent6.prompts.review import EXPLORE_REVIEW_SYSTEM_PROMPT, REVIEW_SYSTEM_PROMPT
from agent6.providers import (
    Provider,
    ProviderError,
    ProviderResponse,
    ToolDefinition,
    output_cap_truncated,
)
from agent6.tools.results import ToolResult
from agent6.workflows._llm_json import extract_json
from agent6.workflows._panel import (
    ALL_CATEGORIES,
    Finding,
    PanelResult,
    ReviewContext,
    ReviewDecision,
    ReviewVerdict,
    aggregate_verdicts,
)


@dataclass(frozen=True, slots=True)
class CritiqueResult:
    """The in-loop panel's verdict the trigger logic consumes: the findings
    text injected for the worker, and whether the panel is satisfied
    (`satisfied=False` only when a blocking decision mode rejects)."""

    text: str
    satisfied: bool


# A read-only dispatch callable for explore-tier seats: (tool_name, input) -> result.
ReviewDispatch = Callable[[str, dict[str, Any]], ToolResult]


@dataclass(frozen=True, slots=True)
class ReviewSeat:
    """One panel seat: a persona stance bound to a provider/model.

    `tier` is "diff" (a single grounded call over the diff) or "explore" (a
    read-only tool-using mini-loop that investigates the broader repo first);
    typed as the config's `ReviewTier` Literal, the vocabulary's one owner."""

    persona: str
    model: str
    provider: Provider
    tier: ReviewTier = "diff"


def parse_seat_spec(spec: str) -> tuple[str, str, str]:
    """Parse a `review_seats` entry into `(persona, provider, model)`.

    `"security@openrouter/moonshotai/kimi-k2"` -> ``("security", "openrouter",
    "moonshotai/kimi-k2")`; `"security"` (no `@`) -> `("security", "", "")``
    (route via the reviewer role); `"@anthropic/claude-opus-4-8"` ->
    `("", "anthropic", "claude-opus-4-8")`. The model may itself contain `/`
    (only the first `/` after `@` splits provider from model)."""
    persona, sep, route = spec.partition("@")
    if not sep:
        return (spec.strip(), "", "")
    provider, _, model = route.partition("/")
    return (persona.strip(), provider.strip(), model.strip())


def _build_user_message(ctx: ReviewContext) -> str:
    parts: list[str] = [f"TASK:\n{ctx.task.strip()[:4000]}"]
    if ctx.agents_md.strip():
        parts.append(f"AGENTS.md:\n{ctx.agents_md.strip()[:8000]}")
    if ctx.verify_ok is None:
        parts.append("VERIFY: none configured for this run.")
    else:
        status = "PASSED" if ctx.verify_ok else "FAILED"
        out = ctx.verify_output.strip()[-2000:]
        parts.append(f"VERIFY: {status}\n{out}" if out else f"VERIFY: {status}")
    if ctx.prior_findings:
        already = "; ".join(f"{f.file_line} {f.category}" for f in ctx.prior_findings[:20])
        parts.append(f"ALREADY RAISED (do not repeat): {already}")
    parts.append(f"DIFF:\n{ctx.diff[:60_000]}")
    return "\n\n".join(parts)


def _coerce_findings(raw: object) -> tuple[Finding, ...]:
    out: list[Finding] = []
    if not isinstance(raw, list):
        return ()
    for item in raw:
        if not isinstance(item, dict):
            continue
        category = str(item.get("category", "other"))
        if category not in ALL_CATEGORIES:
            category = "other"
        severity = str(item.get("severity", "warn"))
        if severity not in ("block", "warn", "nit"):
            severity = "warn"
        out.append(
            Finding(
                category=category,
                severity=severity,  # type: ignore[arg-type]
                file_line=str(item.get("file_line", "")).strip(),
                title=str(item.get("title", "")).strip()[:200],
                detail=str(item.get("detail", "")).strip()[:1000],
            )
        )
    return tuple(out)


def _no_verdict_error(resp: ProviderResponse) -> str:
    """Why a seat produced no verdict JSON, in the reviewer's own terms: the
    output cap ate the answer, the reviewer returned nothing to parse, or what
    it returned would not parse. A generic "unparseable reviewer output" over
    the first two blamed the parser for the provider's own truncation or
    error, hiding the one actionable fact."""
    if output_cap_truncated(resp):
        detail = (
            "before emitting any content (likely all reasoning)"
            if not resp.text.strip()
            else "mid-answer, before the verdict JSON completed"
        )
        return (
            f"output hit the cap {detail}"
            f" (stop_reason={resp.stop_reason}, {resp.output_tokens} output tokens);"
            " raise max_tokens or use a model with more output headroom"
        )
    if not resp.text.strip():
        # No content at all: an upstream error, or a reasoning model that spent
        # its whole budget in the reasoning channel. Blaming the parser for an
        # empty body hid both.
        return (
            f"the reviewer returned no content (stop_reason={resp.stop_reason},"
            f" {resp.output_tokens} output tokens)"
        )
    return "unparseable reviewer output"


def structured_review(
    provider: Provider, ctx: ReviewContext, *, seat: str, model: str, max_tokens: int = 1500
) -> ReviewVerdict:
    """Run one seat. Returns a ReviewVerdict; any failure (provider error, junk
    output) yields an ABSTAINING verdict (`error` set) -- never a false pass."""
    system = REVIEW_SYSTEM_PROMPT.format(persona=ctx.persona or "general correctness")
    try:
        resp = provider.call(
            system=system,
            messages=[{"role": "user", "content": _build_user_message(ctx)}],
            max_tokens=max_tokens,
        )
    except ProviderError as exc:
        return ReviewVerdict(seat=seat, model=model, verdict="pass", error=f"provider: {exc}")
    obj = extract_json(resp.text, prefer=("verdict", "findings"))
    if obj is None:
        return ReviewVerdict(seat=seat, model=model, verdict="pass", error=_no_verdict_error(resp))
    return _verdict_from_obj(obj, seat, model)


def _verdict_from_obj(obj: dict[str, Any], seat: str, model: str) -> ReviewVerdict:
    findings = _coerce_findings(obj.get("findings"))
    verdict = "block" if str(obj.get("verdict", "")).lower() == "block" else "pass"
    return ReviewVerdict(
        seat=seat,
        model=model,
        verdict=verdict,
        findings=findings,
        summary=str(obj.get("summary", "")).strip()[:300],
    )


def explore_review(
    provider: Provider,
    ctx: ReviewContext,
    *,
    seat: str,
    model: str,
    tools: list[ToolDefinition],
    dispatch: ReviewDispatch,
    max_iters: int = 6,
    max_tokens: int = 2000,
    deadline_s: float = 90.0,
) -> ReviewVerdict:
    """A read-only tool-using reviewer: a bounded mini-loop where the seat may
    call read-only tools to investigate the repo, then emits a ReviewVerdict.
    Tools are an explicit read-only allowlist enforced by the caller's dispatch;
    any failure (provider error, deadline, no verdict within max_iters) ABSTAINS."""
    system = EXPLORE_REVIEW_SYSTEM_PROMPT.format(persona=ctx.persona or "general correctness")
    messages: list[dict[str, Any]] = [{"role": "user", "content": _build_user_message(ctx)}]
    start = time.monotonic()
    for i in range(max_iters):
        if time.monotonic() - start > deadline_s:
            return ReviewVerdict(
                seat=seat, model=model, verdict="pass", error="explore: deadline exceeded"
            )
        try:
            resp = provider.call(
                system=system, messages=messages, tools=tools, max_tokens=max_tokens
            )
        except ProviderError as exc:
            return ReviewVerdict(seat=seat, model=model, verdict="pass", error=f"provider: {exc}")
        messages.append({"role": "assistant", "content": resp.raw.get("content") or []})
        if not resp.tool_uses:
            obj = extract_json(resp.text, prefer=("verdict", "findings"))
            if obj is None:
                return ReviewVerdict(
                    seat=seat, model=model, verdict="pass", error=_no_verdict_error(resp)
                )
            return _verdict_from_obj(obj, seat, model)
        # On the last allowed iteration, a verdict emitted ALONGSIDE tool calls
        # still counts (don't waste the investigation by abstaining). With no
        # verdict, skip the dispatches: no model call follows to consume their
        # results, so executing them only spends tool time on an abstention.
        if i == max_iters - 1:
            obj = extract_json(resp.text, prefer=("verdict", "findings"))
            if obj is not None and ("verdict" in obj or "findings" in obj):
                return _verdict_from_obj(obj, seat, model)
            break
        tool_results: list[dict[str, Any]] = []
        for tu in resp.tool_uses:
            name = tu.get("name", "")
            tu_id = tu.get("id", "")
            try:
                out = dispatch(name, tu.get("input", {}) or {})
                content = json.dumps(out.to_wire(), ensure_ascii=False)[:8000]
            except Exception as exc:
                content = f"error: {exc}"[:2000]
            tool_results.append({"type": "tool_result", "tool_use_id": tu_id, "content": content})
        messages.append({"role": "user", "content": tool_results})
    return ReviewVerdict(
        seat=seat, model=model, verdict="pass", error="explore: no verdict within max_iters"
    )


def run_panel(
    seats: list[ReviewSeat],
    ctx: ReviewContext,
    *,
    decision: ReviewDecision,
    quorum: int,
    panel_id: str,
    concurrency: int = 1,
    tools: list[ToolDefinition] | None = None,
    dispatch: ReviewDispatch | None = None,
) -> PanelResult:
    """Run every seat and aggregate. Each seat sees the same context with its own
    persona substituted. With `concurrency > 1` the seat calls run on a thread
    pool (the shared budget tracker + transcript sink are both lock-protected, and
    each seat has its own provider); results stay in seat order, so the merged
    verdict is deterministic regardless of how the calls interleave."""

    def _run(s: ReviewSeat) -> ReviewVerdict:
        seat_ctx = replace(ctx, persona=s.persona)
        if s.tier == "explore" and tools is not None and dispatch is not None:
            return explore_review(
                s.provider, seat_ctx, seat=s.persona, model=s.model, tools=tools, dispatch=dispatch
            )
        return structured_review(s.provider, seat_ctx, seat=s.persona, model=s.model)

    if concurrency > 1 and len(seats) > 1:
        verdicts = _run_seats_concurrently(seats, _run, concurrency)
    else:
        verdicts = [_run(s) for s in seats]
    return aggregate_verdicts(verdicts, ctx, decision=decision, quorum=quorum, panel_id=panel_id)


def _run_seats_concurrently(
    seats: list[ReviewSeat],
    run_seat: Callable[[ReviewSeat], ReviewVerdict],
    concurrency: int,
) -> list[ReviewVerdict]:
    """Run the seat calls on daemon threads; results stay in seat order.

    Deliberately not a ThreadPoolExecutor: its workers are non-daemon and
    joined at interpreter exit, and an in-flight seat call is a non-streaming
    provider POST with no abort hook -- Ctrl-C on `agent6 review` would hang
    until every in-flight AND queued seat finished.
    Daemon threads die with the process, and the timeout-polling wait lets
    KeyboardInterrupt land promptly on the main thread."""
    slots: list[ReviewVerdict | None] = [None] * len(seats)
    errors: list[BaseException] = []
    gate = threading.Semaphore(min(concurrency, len(seats)))
    done = threading.Semaphore(0)

    def work(i: int, seat: ReviewSeat) -> None:
        with gate:
            try:
                slots[i] = run_seat(seat)
            except BaseException as exc:  # surfaced below; a pool would do the same
                errors.append(exc)
            finally:
                done.release()

    threads = [
        threading.Thread(target=work, args=(i, s), name=f"review-seat-{i}", daemon=True)
        for i, s in enumerate(seats)
    ]
    for t in threads:
        t.start()
    for _ in seats:
        while not done.acquire(timeout=0.2):
            pass
    if errors:
        raise errors[0]
    verdicts = [v for v in slots if v is not None]
    if len(verdicts) != len(seats):  # a worker ended with neither verdict nor error
        raise RuntimeError("review seat thread ended without a verdict")
    return verdicts


__all__ = [
    "ReviewDispatch",
    "ReviewSeat",
    "explore_review",
    "parse_seat_spec",
    "run_panel",
    "structured_review",
]
