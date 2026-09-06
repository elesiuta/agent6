# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Anthropic provider.

HTTP transport and SSE lifecycle are shared with the OpenAI provider
(`providers/_transport.py`, `providers/_stream.py`); both use httpx2
directly (no SDK) for a smaller audit surface and pinned URL. Supports
prompt caching via the `cache_control` block field on system / tool
entries.
"""

from __future__ import annotations

import contextlib
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import httpx2

from agent6.budget import BudgetTracker
from agent6.providers._stream import SseCall, StreamClock, bounded_lines, record_billed_usage
from agent6.providers._transport import ProviderCall, envelope_status, meter_completion
from agent6.providers.types import (
    ProviderError,
    ProviderResponse,
    ToolDefinition,
    TranscriptRecorder,
)
from agent6.providers.wire import AuthStyle, Deployment, auth_header, request_url

if TYPE_CHECKING:
    # Imported only for typing (used in a type hint); no runtime import needed.
    from agent6.providers.token_command import CommandToken

ANTHROPIC_DEFAULT_BASE_URL = "https://api.anthropic.com/v1"
ANTHROPIC_VERSION = "2023-06-01"
# Vertex carries the protocol version in the request body (not a header) under
# a Vertex-specific value; see _anthropic_version.
ANTHROPIC_VERTEX_VERSION = "vertex-2023-10-16"
DEFAULT_MAX_TOKENS = 8192


def _anthropic_version(deployment: str) -> tuple[str, str]:
    """Return `(placement, value)` for the Anthropic protocol version.

    Direct sends it as the `anthropic-version` HEADER; Vertex (and future
    Bedrock) send it as an `anthropic_version` BODY field with a
    deployment-specific value.
    """
    if deployment == "vertex":
        return ("body", ANTHROPIC_VERTEX_VERSION)
    return ("header", ANTHROPIC_VERSION)


# Legacy extended-thinking `budget_tokens` per cross-provider `effort`
# level. Anthropic REMOVED budget_tokens (a 400) on the
# models in _ADAPTIVE_THINKING_MARKERS below in favour of adaptive thinking plus
# output_config.effort, so this map is only for older models. Anthropic requires
# `budget_tokens < max_tokens`; the call site lifts `max_tokens` so the
# model keeps room to answer after thinking.
_THINKING_BUDGET_TOKENS: dict[str, int] = {
    "low": 4096,
    "medium": 8192,
    "high": 16384,
    # The above-high tiers collapse to the top budget on this wire.
    "xhigh": 16384,
    "max": 16384,
}

# Models whose extended thinking must be adaptive: Anthropic removed
# `budget_tokens` (a 400) on Opus 4.7+, Sonnet 5, and Fable 5, and deprecated
# it on the 4.6 generation. All of these accept `thinking: {type: adaptive}`
# and `output_config.effort`.
_ADAPTIVE_THINKING_MARKERS = (
    "fable-5",
    "mythos-5",
    "mythos-preview",
    "opus-4-6",
    "opus-4-7",
    "opus-4-8",
    "opus-5",
    "sonnet-4-6",
    "sonnet-5",
)
# Of those, the models whose thinking display defaults to omitted: ask for a
# summary so a long think streams progress (which also keeps the SSE idle
# watchdog fed) and matches the documented default-override for these.
_SUMMARISE_DISPLAY_MARKERS = (
    "fable-5",
    "mythos-5",
    "mythos-preview",
    "opus-4-7",
    "opus-4-8",
    "opus-5",
    "sonnet-5",
)


def _is_adaptive_thinking(model: str) -> bool:
    return any(m in model for m in _ADAPTIVE_THINKING_MARKERS)


def _summarise_thinking_display(model: str) -> bool:
    return any(m in model for m in _SUMMARISE_DISPLAY_MARKERS)


def _require_metered_usage(usage: object, *, source: str) -> None:
    """Fail closed when a budgeted Anthropic call cannot be metered.

    Presence alone is not enough: a gateway with usage tracking disabled returns
    all-zero counts and every turn records zero, so the budget never trips.
    Require a positive input side, but sum in the cache counters: a fully-cached
    turn legitimately reports `input_tokens: 0` with `cache_read_input_tokens`
    > 0, so a plain `input_tokens > 0` check would false-reject it."""
    if isinstance(usage, Mapping):
        # Coerce numerically (the streaming path already does): a proxy typing
        # counts as floats/strings is meterable. Absent/zero/non-numeric still
        # fails closed below.
        def _count(key: str) -> int:
            try:
                return int(usage.get(key) or 0)
            except (TypeError, ValueError):
                return 0

        total_input = (
            _count("input_tokens")
            + _count("cache_read_input_tokens")
            + _count("cache_creation_input_tokens")
        )
        if total_input > 0:
            return
    # No status code: retryable, same reasoning as the OpenAI guard -- a
    # usage-less reply is stream/gateway integrity failure, not a permanent
    # request defect; the failed attempt returns no response so nothing
    # unmetered enters the conversation.
    raise ProviderError(
        f"{source} reported no usage input tokens (usage.input_tokens missing or 0); "
        "budgeted runs require provider usage accounting"
    )


def strip_cache_control_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return `messages` with every `cache_control` marker removed.

    The workflow places rolling breakpoints in the message list (see
    `agent6.workflows._conversation`); when the operator sets
    `prompt_caching = false` this strips them before the request is built.
    Copy-on-write: unmarked messages pass through untouched, marked blocks are
    shallow-copied so the caller's list (shared with resume snapshots) is
    never mutated.
    """
    out: list[dict[str, Any]] = []
    changed = False
    for msg in messages:
        content = msg.get("content")
        if not isinstance(content, list) or not any(
            isinstance(b, dict) and "cache_control" in b for b in content
        ):
            out.append(msg)
            continue
        new_content = [
            {k: v for k, v in b.items() if k != "cache_control"}
            if isinstance(b, dict) and "cache_control" in b
            else b
            for b in content
        ]
        out.append({**msg, "content": new_content})
        changed = True
    return out if changed else messages


def drop_foreign_blocks(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return `messages` without other wires' opaque blocks.

    agent6's canonical blocks are Anthropic-shaped and pass through verbatim,
    so a block another provider minted for its own replay (the ChatGPT
    `thinking` block carrying a `chatgpt_reasoning` item: unsigned here,
    its payload foreign) would reach this API raw and 400 the request on a
    cross-provider resume. Copy-on-write like the cache_control strip.
    """

    def foreign(block: Any) -> bool:
        return isinstance(block, dict) and "chatgpt_reasoning" in block

    out: list[dict[str, Any]] = []
    changed = False
    for msg in messages:
        content = msg.get("content")
        if not isinstance(content, list) or not any(foreign(b) for b in content):
            out.append(msg)
            continue
        out.append({**msg, "content": [b for b in content if not foreign(b)]})
        changed = True
    return out if changed else messages


def _is_temperature_400(status: int | None, text: str, body: dict[str, Any]) -> bool:
    """True when a 400 says the model rejects `temperature` (e.g. claude-opus-4-8:
    "temperature is deprecated for this model") AND temperature is still in the
    request body -- the signal to drop it and retry once."""
    return status == 400 and "temperature" in body and "temperature" in (text or "").lower()


@dataclass(frozen=True, slots=True)
class AnthropicProvider:
    """Stateless provider; constructed once per run."""

    api_key: str
    model: str
    base_url: str = ANTHROPIC_DEFAULT_BASE_URL
    deployment: Deployment = "direct"
    # Auth header style (config AuthConfig.style). "x_api_key" for direct
    # Anthropic, "bearer" for Vertex (Google OAuth via token_command).
    auth_style: AuthStyle = "x_api_key"
    prompt_caching: bool = True
    timeout_s: float = 120.0
    transcript_sink: TranscriptRecorder | None = None
    budget: BudgetTracker | None = None
    # Reasoning-effort level (config `effort`). When not "off" the
    # call enables Anthropic extended thinking with a budget drawn from
    # `_THINKING_BUDGET_TOKENS` and drops `temperature` (Anthropic
    # rejects temperature overrides while thinking is enabled).
    effort: str | None = None
    extra_headers: tuple[tuple[str, str], ...] = ()
    extra_body: dict[str, Any] = field(default_factory=dict)
    extra_query: dict[str, str] = field(default_factory=dict)
    # Short-lived bearer source (config auth.token_command). When set it mints
    # the auth token per call instead of api_key, and a 401/403 triggers one
    # refresh + retry. Internally mutable (cache), hence held by reference.
    credential: CommandToken | None = None
    # Some newer models (e.g. claude-opus-4-8) reject ANY `temperature` with a
    # 400 "temperature is deprecated for this model". agent6 pins temperature for
    # determinism, so on that 400 the call drops it and retries, latching this flag so the
    # rest of the run omits it (avoids re-sending the full context every call).
    # A 1-element list because the dataclass is frozen but the list is mutable.
    _omit_temperature: list[bool] = field(default_factory=lambda: [False])

    def _adapt_body_for_400(self, status: int | None, text: str, body: dict[str, Any]) -> bool:
        """Drop `temperature` and latch `_omit_temperature` on a
        "temperature is deprecated" 400 (e.g. claude-opus-4-8); the transport
        retries once with the adapted body."""
        if not _is_temperature_400(status, text, body):
            return False
        self._omit_temperature[0] = True
        body.pop("temperature", None)
        return True

    def _build_headers(self, token: str) -> dict[str, str]:
        """Per-attempt request headers. Rebuilt each attempt because a
        `token_command` credential mints a short-lived bearer (Vertex Google
        OAuth); on a 401/403 the transport refreshes it once and retries."""
        headers: dict[str, str] = {"content-type": "application/json"}
        authed = auth_header(self.auth_style, token)
        if authed is not None:
            headers[authed[0]] = authed[1]
        version_placement, version_value = _anthropic_version(self.deployment)
        if version_placement == "header":
            headers["anthropic-version"] = version_value
        if self.prompt_caching:
            headers["anthropic-beta"] = "prompt-caching-2024-07-31"
        for k, v in self.extra_headers:
            headers[k.lower()] = v
        return headers

    def call(  # noqa: PLR0912
        self,
        *,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[ToolDefinition] | None = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float | None = None,
        reasoning_effort: str | None = None,
        text_delta_callback: Callable[[str], None] | None = None,
        thinking_delta_callback: Callable[[str], None] | None = None,
        should_abort: Callable[[], bool] | None = None,
        should_interrupt: Callable[[], bool] | None = None,
    ) -> ProviderResponse:
        # `reasoning_effort` is the OpenAI-reasoning-model knob; Anthropic
        # extended thinking uses a different shape and is configured on the
        # provider itself (`self.effort`), so the cross-provider call
        # argument is ignored here.
        del reasoning_effort
        # Hard-stop: refuse the call up front if the run is already over budget.
        if self.budget is not None:
            self.budget.check()
        streaming = text_delta_callback is not None or thinking_delta_callback is not None
        url, model_in_body = request_url(
            api_format="anthropic",
            deployment=self.deployment,
            base_url=self.base_url,
            model=self.model,
            streaming=streaming,
            extra_query=self.extra_query,
        )
        version_placement, version_value = _anthropic_version(self.deployment)

        # Breakpoint budget (Anthropic max 4 per request): this provider marks
        # the system block and the last tool (2); the workflow's rolling pair
        # in `messages` (agent6.workflows._conversation) accounts for the other 2.
        system_blocks: list[dict[str, Any]] = [{"type": "text", "text": system}]
        if self.prompt_caching:
            system_blocks[0]["cache_control"] = {"type": "ephemeral"}
        else:
            messages = strip_cache_control_messages(messages)

        tool_payload: list[dict[str, Any]] = []
        if tools:
            for i, t in enumerate(tools):
                block: dict[str, Any] = {
                    "name": t.name,
                    "description": t.description,
                    "input_schema": t.input_schema,
                }
                # Cache the last tool entry too, anthropic caches up to that block.
                if self.prompt_caching and i == len(tools) - 1:
                    block["cache_control"] = {"type": "ephemeral"}
                tool_payload.append(block)

        # Extended thinking. Modern models (see _ADAPTIVE_THINKING_MARKERS) took
        # adaptive thinking + output_config.effort and dropped budget_tokens;
        # older models still use budget_tokens. "off" sends neither.
        level = self.effort or "off"
        adaptive_thinking = level != "off" and _is_adaptive_thinking(self.model)
        thinking_budget = None if adaptive_thinking else _THINKING_BUDGET_TOKENS.get(level)
        if adaptive_thinking or thinking_budget is not None:
            # Room to answer after thinking. Adaptive carries no explicit budget,
            # so reserve the same headroom as the deepest fixed budget.
            reserve = thinking_budget or _THINKING_BUDGET_TOKENS["high"]
            max_tokens = max(max_tokens, reserve + DEFAULT_MAX_TOKENS)

        body: dict[str, Any] = {
            "max_tokens": max_tokens,
            "system": system_blocks,
            "messages": drop_foreign_blocks(messages),
        }
        # Direct carries the model in the body; Vertex carries it in the URL
        # path and moves the protocol version into the body.
        if model_in_body:
            body["model"] = self.model
        if version_placement == "body":
            body["anthropic_version"] = version_value
        if adaptive_thinking:
            # Adaptive is the only on-mode on these models; where display
            # defaults to omitted ask for a summary so a long think streams
            # progress, and map the level onto effort. Temperature is dropped
            # for thinking, same as the legacy branch (the transport also
            # one-shot-adapts a temperature 400).
            thinking_cfg: dict[str, Any] = {"type": "adaptive"}
            if _summarise_thinking_display(self.model):
                thinking_cfg["display"] = "summarized"
            body["thinking"] = thinking_cfg
            # xhigh/max are OpenAI-tier spellings; Anthropic tops out at high.
            body["output_config"] = {"effort": "high" if level in ("xhigh", "max") else level}
        elif thinking_budget is not None:
            # Legacy extended thinking; incompatible with temperature overrides.
            body["thinking"] = {"type": "enabled", "budget_tokens": thinking_budget}
        elif temperature is not None and not self._omit_temperature[0]:
            body["temperature"] = temperature
        if tool_payload:
            body["tools"] = tool_payload
        if self.extra_body:
            # The STRUCTURAL keys agent6 owns: replacing the conversation, the
            # tool schema, or how a response arrives silently changes what the
            # loop sent or breaks its parser. Tuning keys (max_tokens,
            # temperature, thinking) merge last and win.
            reserved = {
                "system",
                "messages",
                "model",
                "stream",
                "anthropic_version",
                "tools",
                "tool_choice",
            }
            body.update({k: v for k, v in self.extra_body.items() if k not in reserved})

        # The transport rebuilds headers per attempt (a token_command
        # credential mints a short-lived Vertex bearer; a 401/403 refreshes it
        # once and retries) and reserves one extra attempt for the one-shot
        # "temperature is deprecated" 400 adaptation.
        return ProviderCall(
            api_label="Anthropic",
            api_format="anthropic",
            url=url,
            body=body,
            timeout_s=self.timeout_s,
            api_key=self.api_key,
            credential=self.credential,
            transcript_sink=self.transcript_sink,
            budget=self.budget,
            model=self.model,
            build_headers=self._build_headers,
            adapt_400=self._adapt_body_for_400,
            adapt_attempts=int("temperature" in body),
            require_metered=lambda data: _require_metered_usage(
                data.get("usage"), source="Anthropic response"
            ),
            parse=_parse_response,
            stream=(
                lambda attempt_headers: self._call_streaming(
                    url=url,
                    headers=attempt_headers,
                    body=body,
                    text_delta_callback=text_delta_callback,
                    thinking_delta_callback=thinking_delta_callback,
                    should_abort=should_abort,
                    should_interrupt=should_interrupt,
                )
            )
            if streaming
            else None,
        ).run()

    def _call_streaming(  # noqa: PLR0915
        self,
        *,
        url: str,
        headers: dict[str, str],
        body: dict[str, Any],
        text_delta_callback: Callable[[str], None] | None = None,
        thinking_delta_callback: Callable[[str], None] | None = None,
        should_abort: Callable[[], bool] | None = None,
        should_interrupt: Callable[[], bool] | None = None,
    ) -> ProviderResponse:
        """SSE streaming variant.

        The stream lifecycle (idle watchdog, operator stop/steer, teardown
        classification) is `providers._stream.SseCall`; this method owns the
        Anthropic Messages event shape. It fans text_delta and thinking_delta
        deltas to their callbacks as they arrive, and at message_stop returns
        a ProviderResponse whose .raw is shaped identically to a non-streaming
        response so callers (Workflow, transcript replay) don't need a
        streaming-aware code path.
        """
        body = dict(body)
        # Direct enables streaming with a body flag; Vertex selects it via the
        # `:streamRawPredict` URL suffix (already baked into `url`) and rejects
        # a `stream` body field.
        if self.deployment == "direct":
            body["stream"] = True
        stream_headers = dict(headers)
        stream_headers["accept"] = "text/event-stream"

        # Accumulators for the synthesised non-streaming-shape response.
        content_blocks: list[dict[str, Any]] = []
        # Per-index in-flight builders. Anthropic indexes content blocks
        # 0..N within a single message; one block at a time is "open".
        text_acc: dict[int, list[str]] = {}
        tool_acc: dict[int, dict[str, Any]] = {}
        json_partial: dict[int, list[str]] = {}
        # Extended-thinking builders. `thinking_acc` collects the visible
        # reasoning text and `signature_acc` the cryptographic signature
        # Anthropic requires to be echoed back on the next turn when a tool
        # call follows a thinking block. Dropping either breaks multi-turn
        # tool use under extended thinking, so both must round-trip.
        thinking_acc: dict[int, list[str]] = {}
        signature_acc: dict[int, list[str]] = {}
        stop_reason: str = ""
        # The stream is complete only when a `message_stop` event arrives. A
        # clean EOF before it means the connection was cut mid-message; the
        # accumulated blocks are a truncated turn, not a finished one.
        saw_message_stop = False
        usage_input = 0
        usage_output = 0
        usage_cache_read = 0
        usage_cache_creation = 0
        saw_input_usage = False
        saw_output_usage = False

        call = SseCall(
            api_label="Anthropic",
            api_format="anthropic",
            url=url,
            headers=stream_headers,
            body=body,
            timeout_s=self.timeout_s,
            transcript_sink=self.transcript_sink,
            should_abort=should_abort,
            should_interrupt=should_interrupt,
        )

        def consume(resp: httpx2.Response, clock: StreamClock) -> None:  # noqa: PLR0912, PLR0915
            nonlocal stop_reason, saw_message_stop, usage_input, usage_output
            nonlocal usage_cache_read, usage_cache_creation, saw_input_usage, saw_output_usage
            event_type = ""
            for line in bounded_lines(resp):
                if not line:
                    event_type = ""
                    continue
                if line.startswith("event:"):
                    event_type = line[6:].strip()
                    continue
                if not line.startswith("data:"):
                    continue
                data_str = line[5:].strip()
                if not data_str:
                    continue
                try:
                    evt: dict[str, Any] = json.loads(data_str)
                except json.JSONDecodeError:
                    continue
                et = event_type or str(evt.get("type", ""))
                # Reset the idle clock on every MEANINGFUL event. `ping`
                # heartbeats are deliberately excluded: they are exactly the
                # bytes that would otherwise mask a wedged upstream. mark_output
                # (the switch to the short mid-stream idle timeout) fires only
                # when actual content starts (content_block_* below), NOT on
                # message_start -- that metadata arrives before the model has
                # produced anything, and ending the generous prefill budget
                # there would false-kill a long silent reason.
                if et != "ping":
                    clock.mark_data()
                if et in ("content_block_start", "content_block_delta"):
                    clock.mark_output()
                if et == "message_start":
                    msg = evt.get("message", {})
                    u = msg.get("usage", {}) or {}
                    if "input_tokens" in u and u.get("input_tokens") is not None:
                        saw_input_usage = True
                    usage_input = int(u.get("input_tokens") or usage_input)
                    usage_cache_read = int(u.get("cache_read_input_tokens") or usage_cache_read)
                    usage_cache_creation = int(
                        u.get("cache_creation_input_tokens") or usage_cache_creation
                    )
                elif et == "content_block_start":
                    idx = int(evt.get("index", 0))
                    cb = evt.get("content_block", {}) or {}
                    btype = cb.get("type")
                    if btype == "text":
                        text_acc[idx] = [str(cb.get("text", ""))]
                    elif btype == "thinking":
                        # A thinking block streams only ping heartbeats under
                        # display:omitted; tell the watchdog to wait out the
                        # patient thinking budget until it closes, else the tight
                        # mid-stream budget false-kills a long reason (see
                        # providers/_stream.py idle phases).
                        clock.enter_thinking()
                        thinking_acc[idx] = [str(cb.get("thinking", ""))]
                        signature_acc[idx] = [str(cb.get("signature", ""))]
                    elif btype == "redacted_thinking":
                        # Opaque encrypted block, pass straight through.
                        content_blocks.append(
                            {"type": "redacted_thinking", "data": cb.get("data", "")}
                        )
                    elif btype == "tool_use":
                        tool_acc[idx] = {
                            "type": "tool_use",
                            "id": cb.get("id", ""),
                            "name": cb.get("name", ""),
                            "input": cb.get("input", {}) or {},
                        }
                        json_partial[idx] = []
                elif et == "content_block_delta":
                    idx = int(evt.get("index", 0))
                    d = evt.get("delta", {}) or {}
                    dt = d.get("type")
                    if dt == "text_delta":
                        piece = str(d.get("text", ""))
                        text_acc.setdefault(idx, []).append(piece)
                        if piece and text_delta_callback is not None:
                            # Callback failure must never break the
                            # stream, cosmetic surface.
                            with contextlib.suppress(Exception):
                                text_delta_callback(piece)
                    elif dt == "thinking_delta":
                        piece = str(d.get("thinking", ""))
                        thinking_acc.setdefault(idx, []).append(piece)
                        if piece and thinking_delta_callback is not None:
                            with contextlib.suppress(Exception):
                                thinking_delta_callback(piece)
                    elif dt == "signature_delta":
                        signature_acc.setdefault(idx, []).append(str(d.get("signature", "")))
                    elif dt == "input_json_delta":
                        json_partial.setdefault(idx, []).append(str(d.get("partial_json", "")))
                elif et == "content_block_stop":
                    idx = int(evt.get("index", 0))
                    if idx in text_acc:
                        content_blocks.append(
                            {
                                "type": "text",
                                "text": "".join(text_acc.pop(idx)),
                            }
                        )
                    elif idx in thinking_acc:
                        # Thinking done; real output (or the next block) resumes
                        # normal idle budgeting.
                        clock.exit_thinking()
                        block_out: dict[str, Any] = {
                            "type": "thinking",
                            "thinking": "".join(thinking_acc.pop(idx)),
                        }
                        sig = "".join(signature_acc.pop(idx, []))
                        if sig:
                            block_out["signature"] = sig
                        content_blocks.append(block_out)
                    elif idx in tool_acc:
                        tu = tool_acc.pop(idx)
                        partial = "".join(json_partial.pop(idx, []))
                        if partial:
                            try:
                                tu["input"] = json.loads(partial)
                            except json.JSONDecodeError:
                                # Stream truncated mid-JSON; surface
                                # what arrived rather than dropping
                                # the tool_use entirely.
                                tu["input"] = {"_partial_json": partial}
                        content_blocks.append(tu)
                elif et == "message_delta":
                    d = evt.get("delta", {}) or {}
                    if "stop_reason" in d:
                        stop_reason = str(d.get("stop_reason", "") or "")
                    u = evt.get("usage", {}) or {}
                    if "output_tokens" in u:
                        if u.get("output_tokens") is not None:
                            saw_output_usage = True
                        usage_output = int(u.get("output_tokens") or usage_output)
                elif et == "message_stop":
                    saw_message_stop = True
                    return
                elif et == "error":
                    err = evt.get("error", {}) or {}
                    # Record the frame before raising so the upstream failure
                    # is auditable in the transcript (parity with the OpenAI
                    # provider's mid-stream error handling). Carry the upstream
                    # status like the non-streaming 2xx-envelope path, so a
                    # permanent error delivered mid-stream fails fast instead of
                    # retrying every turn (streaming is the default path).
                    call.record(status=0, response=data_str[:8192])
                    raise ProviderError(
                        f"Anthropic stream error: {err.get('type')}: {err.get('message')}",
                        status_code=envelope_status(err),
                    )

        def _record_billed() -> None:
            record_billed_usage(
                self.budget,
                self.model,
                input_tokens=usage_input,
                output_tokens=usage_output,
                cache_read_tokens=usage_cache_read,
                cache_creation_tokens=usage_cache_creation,
            )

        try:
            call.run(consume)
        except BaseException:
            # Billed already: input usage arrives in message_start, long before
            # a mid-stream error, the idle watchdog, or an operator steer can
            # end the turn. The retry re-sends the whole input and is billed
            # again, so a run with any flakiness had no ceiling.
            _record_billed()
            raise

        # No `message_stop` means the stream was cut mid-message (a clean EOF is
        # not a completion signal). The accumulated blocks are a truncated turn,
        # possibly with text already fanned to the TUI; returning them as a
        # finished response feeds the loop a bogus went_quiet/silent_finish.
        # Raise a retryable ProviderError so _call_with_retry re-issues the
        # request -- but record what the cut turn already cost first: the
        # operator's cap is about money spent, not turns completed.
        if not saw_message_stop:
            _record_billed()
            call.record(status=0, response="stream ended without message_stop (truncated)")
            raise ProviderError(
                f"Anthropic SSE stream from {url} ended prematurely "
                "(no message_stop); upstream appears cut off."
            )

        # Synthesise the non-streaming-shaped response body so
        # downstream consumers (transcript replay, assistant_blocks
        # reconstruction in Workflow) see the same shape they would
        # see from a non-streaming call.
        synthesised: dict[str, Any] = {
            "type": "message",
            "role": "assistant",
            "content": content_blocks,
            "stop_reason": stop_reason,
            "usage": {
                "input_tokens": usage_input,
                "output_tokens": usage_output,
                "cache_read_input_tokens": usage_cache_read,
                "cache_creation_input_tokens": usage_cache_creation,
            },
        }
        call.record(status=200, response=synthesised)
        if self.budget is not None:
            try:
                if not (saw_input_usage and saw_output_usage):
                    raise ProviderError(
                        "Anthropic stream omitted usage.input_tokens/output_tokens; "
                        "budgeted runs require provider usage accounting",
                        status_code=422,
                    )
                _require_metered_usage(synthesised.get("usage"), source="Anthropic stream")
            except ProviderError:
                # Billed for what it generated, like every cut stream above;
                # a completion the guards accept is metered by meter_completion.
                _record_billed()
                raise
        parsed = _parse_response(synthesised)
        meter_completion(self.budget, self.model, parsed, "Anthropic")
        return parsed


def _parse_response(data: dict[str, Any]) -> ProviderResponse:
    content = data.get("content") or []
    if not isinstance(content, list):
        # A misbehaving Anthropic-format proxy returning `content` as a bare
        # string would iterate characters and AttributeError past the loop's
        # ProviderError-only retry. Raise retryably (status_code unset), like
        # the non-JSON and truncation guards.
        raise ProviderError(
            f"Anthropic response `content` was {type(content).__name__}, not a"
            " list (malformed 2xx from upstream gateway)"
        )
    text_parts: list[str] = []
    tool_uses: list[dict[str, Any]] = []
    for block in content:
        if not isinstance(block, dict):
            raise ProviderError(
                "Anthropic response content block was not an object"
                " (malformed 2xx from upstream gateway)"
            )
        block_type = block.get("type")
        if block_type == "text":
            text_parts.append(str(block.get("text", "")))
        elif block_type == "tool_use":
            tool_uses.append(
                {
                    "id": block.get("id", ""),
                    "name": block.get("name", ""),
                    "input": block.get("input", {}),
                }
            )
    usage = data.get("usage") or {}
    # `or 0` throughout: a gateway returning null token fields on a 2xx would
    # make bare int(None) raise TypeError, which escapes the loop's
    # ProviderError-only retry wrapper and kills the run.
    return ProviderResponse(
        text="".join(text_parts),
        tool_uses=tuple(tool_uses),
        stop_reason=str(data.get("stop_reason", "")),
        input_tokens=int(usage.get("input_tokens") or 0),
        output_tokens=int(usage.get("output_tokens") or 0),
        cache_read_tokens=int(usage.get("cache_read_input_tokens") or 0),
        cache_creation_tokens=int(usage.get("cache_creation_input_tokens") or 0),
        raw=data,
    )
