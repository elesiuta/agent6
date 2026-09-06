# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Shared SSE lifecycle for the provider streaming paths.

Both providers speak Server-Sent Events over a single POST and need the same
machinery around their event loops: an idle watchdog that heartbeats cannot
satisfy, operator stop/steer that ends an in-flight turn promptly, and
classification of the teardown into `ProviderAborted` /
`ProviderInterrupted` / a retryable `ProviderError`. Event parsing stays
per-provider (the two wire formats share nothing); this module owns
everything around it.

Why a watchdog at all: httpx2's `timeout` (float or `httpx2.Timeout` with
`read=`) resets on EVERY received byte, and gateways emit heartbeat bytes
while a request is in flight (OpenRouter/Cloudflare send `:` SSE comment
lines every ~15s; Anthropic sends `ping` events). If the upstream model
truly hangs (observed: Kimi K2.6 sessions held in ESTABLISHED state with 0
bytes of payload for 800+ seconds while heartbeats continued), the read
timeout never fires and the orchestrator parks forever with no spend cap to
save it. The fix: the per-provider consume loop marks each MEANINGFUL event
on a :class:`StreamClock` (heartbeats deliberately do not count), and a
watchdog thread closes the response once the gap exceeds the threshold. The
blocking `iter_lines` then raises an `httpx2.HTTPError` that
:meth:`SseCall.run` re-raises as a descriptive error so the loop can
retry-or-quit at its own layer.

Three idle phases, because "no data yet", "data stopped", and "thinking" mean
different things:

- Before the first real output token the gap is prefill / time-to-first-token,
  which legitimately runs long on a big context or a slow model, so be patient
  (`STREAM_FIRST_DATA_TIMEOUT_S`).
- Once real output has started, models emit a data event every few seconds; a
  45s gap then means the stream wedged. Recovering a mid-stream wedge in 45s
  instead of 180s is 4x faster (`STREAM_IDLE_TIMEOUT_S`).
- Inside a display:omitted extended-thinking block (Anthropic adaptive thinking
  on Sonnet 5 / Opus 4.7+ / Fable 5) the stream is ping-only by design while the
  model reasons, so neither budget above applies; wait out a generous thinking
  budget instead (`STREAM_THINKING_IDLE_TIMEOUT_S`). The consume loop brackets
  the block with `enter_thinking()` / `exit_thinking()`.
"""

from __future__ import annotations

import contextlib
import json
import threading
import time
from collections.abc import Callable, Generator
from dataclasses import dataclass
from typing import Any

import httpx2

from agent6.budget import BudgetTracker, PlanUsage
from agent6.providers._transport import granular_timeout
from agent6.providers.types import (
    ProviderAborted,
    ProviderError,
    ProviderInterrupted,
    TranscriptRecorder,
    parse_retry_after,
    scrub_secret_values,
)

STREAM_FIRST_DATA_TIMEOUT_S = 120.0
STREAM_IDLE_TIMEOUT_S = 45.0
# A display:omitted extended-thinking block streams only `ping` heartbeats
# while the model reasons (no content deltas), so the tight mid-stream budget
# above would false-kill a long think. While inside a thinking block the
# watchdog waits this much instead; a genuine wedge is still bounded, just less
# tightly. Tunable if a max-effort model ever reasons past it in one block.
STREAM_THINKING_IDLE_TIMEOUT_S = 300.0
# The watchdog also polls should_abort/should_interrupt each tick, so keep it
# short: this bounds how long a Stop/steer/detach waits to end a long in-flight
# turn. A quarter second reads as immediate without the impatient second Ctrl-C.
STREAM_WATCHDOG_TICK_S = 0.25


@contextlib.contextmanager
def http_stream(
    method: str, url: str, *, headers: dict[str, str], content: bytes, timeout: float
) -> Generator[httpx2.Response]:
    """Streaming POST seam: tests stub this name, never `httpx2` globally.

    `granular_timeout` bounds the connect phase: the idle watchdog has no
    response to close until the connect returns, so a blackholed connect is
    httpx2's to cut, and it must not wait the full read budget to do it."""
    with httpx2.stream(
        method, url, headers=headers, content=content, timeout=granular_timeout(timeout)
    ) as resp:
        yield resp


def bounded_lines(resp: httpx2.Response, *, max_line_bytes: int = 8 * 1024 * 1024):
    """`resp.iter_lines()` with a per-line ceiling on what is handed
    downstream: a line over it raises a retryable ProviderError instead of
    being parsed (the non-streaming path caps its whole body). A line that
    never ends is bounded by the watchdog, since iter_lines materializes it
    first. Both providers' consume loops read through this."""
    for line in resp.iter_lines():
        if len(line) * 4 > max_line_bytes and len(line.encode("utf-8")) > max_line_bytes:
            raise ProviderError(
                f"stream frame exceeded {max_line_bytes} bytes; refusing to buffer it"
            )
        yield line


class StreamClock:
    """Idle bookkeeping the per-provider consume loop feeds.

    `mark_data()` on every meaningful wire event; heartbeats must not be
    marked, they are exactly the bytes that mask a wedged upstream.
    `mark_output()` when the model has produced real content (text /
    reasoning / tool tokens), which ends the generous prefill budget and
    starts the short mid-stream idle budget. `enter_thinking()` /
    `exit_thinking()` bracket a display:omitted thinking block, whose
    ping-only stream needs the patient thinking budget rather than either.
    """

    __slots__ = ("_in_thinking", "_seen_output", "last_data_at")

    def __init__(self) -> None:
        self.last_data_at = time.monotonic()
        self._seen_output = threading.Event()
        self._in_thinking = threading.Event()

    def mark_data(self) -> None:
        self.last_data_at = time.monotonic()

    def mark_output(self) -> None:
        self._seen_output.set()

    def enter_thinking(self) -> None:
        self._in_thinking.set()

    def exit_thinking(self) -> None:
        self._in_thinking.clear()

    def idle_budget(self) -> tuple[float, str]:
        """The active idle timeout and a label for it: a thinking block gets the
        patient budget, before real output it is prefill, after it the tight
        mid-stream budget."""
        if self._in_thinking.is_set():
            return (STREAM_THINKING_IDLE_TIMEOUT_S, "mid-thinking")
        if self._seen_output.is_set():
            return (STREAM_IDLE_TIMEOUT_S, "mid-stream")
        return (STREAM_FIRST_DATA_TIMEOUT_S, "before any data (prefill)")


def safe_poll(fn: Callable[[], bool] | None) -> bool:
    """An operator-state poll (should_abort, should_interrupt): absent or
    raising, it reads False, so a poll never kills the watcher."""
    if fn is None:
        return False
    try:
        return bool(fn())
    except Exception:
        return False


def record_billed_usage(
    budget: BudgetTracker | None,
    model: str,
    *,
    input_tokens: int,
    output_tokens: int,
    cache_read_tokens: int = 0,
    cache_creation_tokens: int = 0,
    cost_usd: float = 0.0,
    plan_usage: PlanUsage | None = None,
) -> None:
    """Record what a call that did NOT complete already cost.

    A stream that dies after the provider reported usage has been billed: the
    input was accepted, and whatever was generated was produced. Counting only
    completed calls left that spend invisible to `max_usd`, so a retry-heavy
    run had no ceiling at all -- every retry re-sends the whole input and is
    billed again. The operator set a number for the task; going past it without
    being told is the failure, and a run can always be resumed.

    Records nothing when the provider reported nothing: an unknown amount is
    not a licence to invent one. A reported plan window is a report: the
    window moved even when the body carried no token counts.
    """
    if budget is None:
        return
    if plan_usage is None and not (
        input_tokens or output_tokens or cache_read_tokens or cache_creation_tokens
    ):
        return
    budget.record(
        model=model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_read_tokens=cache_read_tokens,
        cache_creation_tokens=cache_creation_tokens,
        cost_usd=cost_usd,
        plan_usage=plan_usage,
    )


@dataclass(frozen=True, slots=True)
class SseCall:
    """One provider SSE request: what the shared lifecycle needs around the
    per-provider event loop."""

    api_label: str  # "OpenAI" / "Anthropic"; leads API-error messages
    api_format: str  # "openai" / "anthropic"; names the wire format
    url: str
    headers: dict[str, str]
    body: dict[str, Any]
    timeout_s: float
    transcript_sink: TranscriptRecorder | None
    should_abort: Callable[[], bool] | None
    should_interrupt: Callable[[], bool] | None

    def record(self, *, status: int, response: dict[str, Any] | str) -> None:
        """Write one transcript entry for this request (no-op without a sink)."""
        if self.transcript_sink is not None:
            self.transcript_sink.record(
                url=self.url,
                request_headers=self.headers,
                request_body=self.body,
                response_status=status,
                response_body=response,
            )

    def run(  # noqa: PLR0915
        self, consume: Callable[[httpx2.Response, StreamClock], None]
    ) -> None:
        """Open the stream, run `consume` under the watchdog, classify teardown.

        `consume` iterates `resp.iter_lines()` and parses the provider's
        events, marking the clock as it goes; accumulation happens in the
        caller's closure. A `ProviderError` it raises (mid-stream error
        frame) propagates unchanged; any other shape error a malformed 2xx
        frame provokes is normalized to a retryable ProviderError here, the
        one seam both providers stream through, so it can never bypass the
        loop's retry wrapper as a raw traceback.
        """
        clock = StreamClock()
        aborted = threading.Event()
        interrupted = threading.Event()
        idle_killed = threading.Event()
        watchdog_stop = threading.Event()
        # Mutable holder so the watchdog can reach the response without racing
        # on assignment (the `with` body runs in a different frame from the
        # watchdog closure).
        resp_holder: dict[str, httpx2.Response] = {}

        def _watchdog() -> None:
            while not watchdog_stop.wait(STREAM_WATCHDOG_TICK_S):
                resp = resp_holder.get("resp")
                if resp is None:
                    continue
                # A poll that raised would end this thread and with it the
                # idle-hang detection.
                if safe_poll(self.should_abort):
                    aborted.set()
                    with contextlib.suppress(Exception):
                        resp.close()
                    return
                # A steer request (Ctrl-C / TUI `s`) closes the stream so a long
                # thinking turn reaches the loop's steer boundary at once.
                if safe_poll(self.should_interrupt):
                    interrupted.set()
                    with contextlib.suppress(Exception):
                        resp.close()
                    return
                timeout, _ = clock.idle_budget()
                if time.monotonic() - clock.last_data_at <= timeout:
                    continue
                idle_killed.set()
                with contextlib.suppress(Exception):
                    resp.close()
                return

        watchdog = threading.Thread(
            target=_watchdog, name=f"agent6-{self.api_format}-sse-watchdog", daemon=True
        )
        watchdog.start()

        try:
            with http_stream(
                "POST",
                self.url,
                headers=self.headers,
                content=json.dumps(self.body).encode("utf-8"),
                timeout=self.timeout_s,
            ) as resp:
                resp_holder["resp"] = resp
                if resp.status_code >= 400:
                    error_body = resp.read().decode("utf-8", errors="replace")[:8192]
                    self.record(status=resp.status_code, response=error_body)
                    raise ProviderError(
                        f"{self.api_label} API error {resp.status_code}: "
                        f"{scrub_secret_values(error_body, self.headers)[:500]}",
                        status_code=resp.status_code,
                        retry_after_s=parse_retry_after(resp.headers),
                    )
                try:
                    consume(resp, clock)
                except (AttributeError, KeyError, TypeError, ValueError, IndexError) as exc:
                    raise ProviderError(
                        f"{self.api_label} stream frame did not match the wire shape:"
                        f" {exc!r} (malformed 2xx event; retryable)"
                    ) from exc
        except httpx2.HTTPError as exc:
            if interrupted.is_set():
                # The operator asked to steer; the watchdog closed the stream so
                # the loop reaches its steer boundary without waiting out the turn.
                raise ProviderInterrupted("steer requested mid-stream") from exc
            if aborted.is_set():
                # The operator stopped the run; the watchdog closed the stream.
                raise ProviderAborted("run stopped by operator") from exc
            if idle_killed.is_set():
                # Convert the watchdog-induced HTTPError into a purpose-specific
                # ProviderError so the loop's retry/quit path can log a meaningful
                # reason rather than a generic "ReadError" / "connection closed".
                phase_s, where = clock.idle_budget()
                self.record(
                    status=0,
                    response=(
                        f"SSE idle watchdog: no data event for {phase_s:.0f}s {where} "
                        f"(only heartbeats). Upstream model appears wedged."
                    ),
                )
                raise ProviderError(
                    f"{self.api_label} SSE stream idle for >{phase_s:.0f}s {where} "
                    "(only heartbeats received); upstream model appears wedged."
                ) from exc
            self.record(status=0, response=f"HTTPError: {exc}")
            raise ProviderError(
                f"HTTP error streaming from {self.url} ({self.api_format} format): {exc}"
            ) from exc
        finally:
            watchdog_stop.set()
