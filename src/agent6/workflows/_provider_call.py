# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The loop's one path to the provider, and the classification around it.

`ProviderCaller` runs `provider.call` under a bounded retry. The predicates and
constants classify one response or status in isolation (which HTTP statuses are
permanent, how long an upstream Retry-After is honored, the hint for a fatal
error, the empty tool call that earns a blind retry, the reasoning that starved
a turn), so they stay unit-testable without a Workflow.
"""

from __future__ import annotations

import random
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from agent6.providers import (
    Provider,
    ProviderAborted,
    ProviderError,
    ProviderInterrupted,
    ProviderResponse,
    ToolDefinition,
    output_cap_truncated,
)

# HTTP statuses that will never succeed on a blind retry of the same request.
# 400 bad request, 401/403 auth, 402 insufficient credits, 404 bad
# model/endpoint, 422 malformed body. Retrying these only burns wall-time
# 408/409/429 and all 5xx remain retryable and fall through to
# the normal backoff.
NON_RETRYABLE_HTTP_STATUSES = frozenset({400, 401, 402, 403, 404, 422})

# Upper bound on how long we honor an upstream Retry-After hint. A 429/503 often
# carries Retry-After: <seconds>; we wait at least that long (the provider's own
# backoff is usually shorter and just exhausts the retries before the window
# clears), but never longer than this so a buggy/hostile header can't hang a run.
RETRY_AFTER_CEILING_S = 120.0

# Finish/stop reasons that promise a tool call. A response carrying one of these
# but with NO tool_use and NO text is self-contradictory and gets retried (see
# is_empty_tool_call_response).
TOOL_CALL_STOP_REASONS = frozenset({"tool_calls", "tool_use"})


def provider_error_hint(status_code: int | None, provider: str = "") -> str:
    """A short, actionable suffix for a fatal provider error, or "".

    The raw upstream body (e.g. a 401 JSON blob) tells a user nothing about how
    to fix it. Map the common credential/quota statuses to a next step, naming
    the failing *provider*'s config key when known.
    """
    if status_code in (401, 403):
        return (
            " Authentication failed: verify the provider key with `agent6 connect`"
            f" or check [providers.{provider or '<name>'}].api_key_env."
        )
    if status_code == 402:
        return " Insufficient credits/quota at the provider; top up or switch providers."
    return ""


def is_empty_tool_call_response(resp: Any) -> bool:
    """A self-contradictory provider response: the finish/stop reason says the
    model stopped to make a tool call, but no tool_use and no text came back.

    Seen on GLM via OpenRouter after a tier-2 context restart (~50% of turns):
    finish_reason=tool_calls with an empty payload. A blind retry recovers it
    about half the time; without one the loop counts it as went_quiet and the run
    dies at the first compaction. Excludes stop_reason=="length" (deterministic
    reasoning starvation, handled separately with its own nudge)."""
    return (
        str(getattr(resp, "stop_reason", "")) in TOOL_CALL_STOP_REASONS
        and not resp.tool_uses
        and not (resp.text or "").strip()
    )


def reasoning_starvation(resp: ProviderResponse) -> int:
    """The reasoning characters of a turn the output cap cut with billed
    output, 0 otherwise. In the went-quiet handler the count tells a starved
    reasoner (its whole budget went to thinking) from a model that gave up."""
    if not output_cap_truncated(resp) or resp.output_tokens <= 0:
        return 0
    reasoning_chars = 0
    raw_content = (resp.raw or {}).get("content") or []
    if isinstance(raw_content, list):
        for block in raw_content:
            if isinstance(block, dict) and block.get("type") == "thinking":
                reasoning_chars += len(str(block.get("thinking") or ""))
    return reasoning_chars


@dataclass(frozen=True, slots=True)
class ProviderCaller:
    """`provider.call` under a bounded retry: `retry_count + 1` attempts at most.

    Two retry paths share that budget. A transient `ProviderError` (a 529, a
    502, a socket timeout) backs off exponentially with full jitter, waiting
    at least the upstream Retry-After capped at `RETRY_AFTER_CEILING_S`; a
    permanent one (`fatal`, or a status in `NON_RETRYABLE_HTTP_STATUSES`)
    re-raises at once, since the same request cannot succeed. An empty
    tool-call response (`is_empty_tool_call_response`) is re-asked after a
    short fixed delay, and when every attempt is empty the last is returned
    for the loop's went-quiet handler. An abort, an interrupt and
    `BudgetExceeded` are never retried.
    """

    provider: Provider
    retry_count: int
    retry_delay_s: float
    retry_max_delay_s: float
    temperature: float | None
    should_abort: Callable[[], bool]
    should_interrupt: Callable[[], bool]
    log: Callable[[str], None]
    emit: Callable[..., None]

    def call(
        self,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[ToolDefinition],
        max_tokens: int,
    ) -> ProviderResponse:
        attempts = max(1, self.retry_count + 1)
        attempt = 1
        while True:
            try:
                resp = self.provider.call(
                    system=system,
                    messages=messages,
                    tools=tools,
                    max_tokens=max_tokens,
                    temperature=self.temperature,
                    should_abort=self.should_abort,
                    should_interrupt=self.should_interrupt,
                )
            except (ProviderAborted, ProviderInterrupted):
                raise  # an operator stop or steer is handled, never retried
            except ProviderError as exc:
                if exc.fatal or exc.status_code in NON_RETRYABLE_HTTP_STATUSES:
                    self.log(
                        f"LOOP: provider error {exc.status_code or 'fatal'} is permanent;"
                        " not retrying"
                    )
                    self.emit(
                        "loop.provider.fatal", status_code=exc.status_code, error=str(exc)[:200]
                    )
                    raise
                if attempt == attempts:
                    raise
                delay = self._backoff(attempt, exc.retry_after_s)
                self.log(
                    f"LOOP: provider error attempt {attempt}/{attempts}: {exc}"
                    f" - retrying in {delay:.2f}s"
                )
                self.emit("loop.provider.retry", attempt=attempt, error=str(exc)[:200])
            else:
                if attempt == attempts or not is_empty_tool_call_response(resp):
                    return resp
                # Model flakiness, not rate limiting: a short fixed delay.
                delay = min(self.retry_delay_s, 1.0) * random.uniform(0.5, 1.0)  # noqa: S311
                self.log(
                    f"LOOP: empty tool-call response attempt {attempt}/{attempts}"
                    f" (stop_reason={resp.stop_reason!r}, no tool_use/text);"
                    f" retrying in {delay:.2f}s"
                )
                self.emit(
                    "loop.provider.empty_tool_call_retry",
                    attempt=attempt,
                    stop_reason=str(resp.stop_reason),
                )
            time.sleep(delay)
            attempt += 1

    def _backoff(self, attempt: int, retry_after_s: float | None) -> float:
        """Exponential backoff with full jitter (floored at half), capped at
        `retry_max_delay_s`; never shorter than an upstream Retry-After, itself
        capped at `RETRY_AFTER_CEILING_S` so a hostile header cannot hang a run."""
        capped = min(self.retry_delay_s * 2 ** (attempt - 1), self.retry_max_delay_s)
        delay = capped * random.uniform(0.5, 1.0)  # noqa: S311
        if retry_after_s is not None:
            delay = max(delay, min(retry_after_s, RETRY_AFTER_CEILING_S))
        return delay


__all__ = [
    "NON_RETRYABLE_HTTP_STATUSES",
    "RETRY_AFTER_CEILING_S",
    "TOOL_CALL_STOP_REASONS",
    "ProviderCaller",
    "is_empty_tool_call_response",
    "provider_error_hint",
    "reasoning_starvation",
]
