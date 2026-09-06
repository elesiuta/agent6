# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""OpenAI Chat Completions response parsing.

The response-parsing half of the provider: choices[0].message ->
`ProviderResponse` in agent6's canonical Anthropic shape (tool_calls ->
tool_uses, reasoning_content -> a leading thinking block in `raw`,
prompt_tokens normalised to fresh-input semantics). Both the non-streaming
path and the synthesised streaming response in `providers/openai.py` call
it; the text-embedded tool-call fallback lives in
`providers/_openai_recovery.py`.
"""

from __future__ import annotations

import json
from typing import Any

from agent6.providers._openai_recovery import (
    coerce_text_tool_calls,
    lenient_json_object,
)
from agent6.providers.types import ProviderError, ProviderResponse


def parse_response(  # noqa: PLR0912, PLR0915
    data: dict[str, Any],
    *,
    tool_names: frozenset[str] = frozenset(),
    tool_schemas: dict[str, dict[str, Any]] | None = None,
) -> ProviderResponse:
    choices = data.get("choices") or []
    text = ""
    reasoning_text = ""
    stop_reason = ""
    tool_uses: tuple[dict[str, Any], ...] = ()
    if choices:
        first = choices[0]
        if not isinstance(first, dict):
            # A malformed 2xx (choices[0] null/string from a flaky local
            # endpoint) must surface as a retryable ProviderError, not an
            # AttributeError that bypasses the loop's retry wrapper.
            raise ProviderError(
                f"OpenAI choices[0] is {type(first).__name__}, not an object (malformed 2xx body)"
            )
        message = first.get("message") or {}
        text = str(message.get("content") or "")
        # Kimi (`reasoning_content`), DeepSeek-R1 /
        # OpenRouter (`reasoning`), and OpenAI o-series surface
        # reasoning in a sibling field. Capture both spellings.
        raw_reasoning = message.get("reasoning_content") or message.get("reasoning") or ""
        reasoning_text = str(raw_reasoning) if raw_reasoning else ""
        stop_reason = str(first.get("finish_reason") or "")
        raw_calls = message.get("tool_calls") or []
        parsed_calls: list[dict[str, Any]] = []
        for i, call in enumerate(raw_calls):
            if not isinstance(call, dict):
                continue
            func = call.get("function") or {}
            # Small open-weight models (via some OpenRouter backends,
            # Novita backend) sometimes emit a NATIVE tool_call with a blank
            # `function.name`. Dispatching it yields "Unknown tool: " and, worse,
            # echoing the blank-name call back in the next request makes strict
            # backends reject the whole conversation with a 400
            # invalid_request_error, killing the run. Drop the malformed call
            # here so it never enters history; valid calls in the same turn
            # still proceed.
            if not str(func.get("name") or "").strip():
                continue
            args_raw = func.get("arguments", "")
            # OpenAI returns arguments as a JSON string. Convert to dict
            # for the Anthropic-shape input field. Malformed JSON
            # surfaces as an empty dict + the raw string under
            # `_raw_arguments` so debugging is possible.
            #
            # A degenerate tool-arg payload (tens of KB of repeated escape
            # sequences) would be echoed in the tool_error message and re-enter
            # the context, priming the same degeneration next turn. Cap the
            # diagnostic at 500 chars so the
            # repetition doesn't survive the round-trip.
            _RAW_ARGS_CAP = 500
            try:
                parsed_input = json.loads(args_raw) if args_raw else {}
                if not isinstance(parsed_input, dict):
                    parsed_input = {"_value": parsed_input}
            except (json.JSONDecodeError, TypeError):
                # Before giving up, try a lenient re-parse. Weak/open models
                # commonly emit args that strict JSON rejects: a raw newline in a
                # multiline code/regex param, or trailing junk (a leaked
                # `</invoke>` / prose). Recovering here means the tool just runs,
                # instead of a wasted round-trip on a validation error. Only a
                # parse that yields an object is accepted, so a bad guess can't
                # feed the handler garbage; anything still unparseable becomes the
                # `_raw_arguments` sentinel (dispatch turns that into a clear
                # "resend valid JSON" error).
                repaired = lenient_json_object(args_raw)
                if repaired is not None:
                    parsed_input = repaired
                else:
                    raw_str = str(args_raw)
                    if len(raw_str) > _RAW_ARGS_CAP:
                        raw_str = (
                            raw_str[:_RAW_ARGS_CAP]
                            + f"... <truncated; original was {len(str(args_raw))} chars>"
                        )
                    parsed_input = {"_raw_arguments": raw_str}
            parsed_calls.append(
                {
                    # Synthesise a distinct id when the backend omits one
                    # (some open-weight models stream tool_calls with no id).
                    # Two native tool_calls both with id="" would otherwise
                    # collapse to ambiguous/duplicate tool_call_id pairing on
                    # the next request, tripping a strict-backend 400. Mirrors
                    # the call_text_{i} fallback used for recovered calls.
                    "id": str(call.get("id") or f"call_auto_{i}"),
                    "name": str(func.get("name", "")),
                    "input": parsed_input,
                }
            )
        tool_uses = tuple(parsed_calls)
        # Fallback: no NATIVE tool_calls but the model leaked a tool call
        # into its text content (small local models via Ollama/llama.cpp).
        # Guarded by `tool_names` so this only fires for tools actually
        # offered and never for flagship models (which populate
        # tool_calls) or models legitimately answering with JSON.
        if not tool_uses and tool_names:
            recovered, remaining_text = coerce_text_tool_calls(text, tool_names, tool_schemas)
            if recovered:
                tool_uses = tuple(
                    {"id": f"call_text_{i}", "name": r["name"], "input": r["input"]}
                    for i, r in enumerate(recovered)
                )
                text = remaining_text
        # The upstream's own failure signal, seen from OpenRouter as a 200 whose
        # choice carries `finish_reason: "error"`, a null content and nothing
        # else. Handing that back as a finished turn spends a went-quiet nudge
        # on an error, or abstains a review seat as if the model had answered;
        # raise it as retryable, like a stream that ends without [DONE].
        if stop_reason == "error" and not text.strip() and not tool_uses:
            raise ProviderError(
                "OpenAI response carries finish_reason='error' with no content:"
                " the upstream failed this completion"
            )
    usage = data.get("usage") or {}
    # OpenAI's cached_tokens field, when present, lives under
    # usage.prompt_tokens_details.cached_tokens. Treat absent as 0.
    #
    # CRITICAL provider-format asymmetry: Anthropic's `input_tokens`
    # already EXCLUDES cache-read tokens (they're surfaced separately under
    # `cache_read_input_tokens`). OpenAI's `prompt_tokens`, by contrast, is
    # the TOTAL prompt size, cached + fresh. We normalise to Anthropic's
    # semantics here so `ProviderResponse.input_tokens` consistently means
    # "fresh, non-cached input" across providers. Without this, the
    # BudgetTracker would charge cached tokens against the input-token cap
    # at full rate (causing premature budget exhaustion on cache-heavy
    # OpenAI runs) AND the cost formula in budget.py would double-count the
    # cache portion (full input rate plus an additional 10% cache-read
    # surcharge).
    cached = 0
    details = usage.get("prompt_tokens_details") or {}
    if isinstance(details, dict):
        cached = int(details.get("cached_tokens", 0) or 0)
    # `or 0` throughout: a gateway returning `"prompt_tokens": null` on a 2xx
    # would make bare int(None) raise TypeError, which escapes the loop's
    # ProviderError-only retry wrapper and kills the run.
    prompt_total = int(usage.get("prompt_tokens") or 0)
    # Clamp cached to the prompt total as the SINGLE source of truth: a
    # misbehaving upstream that reports cached > prompt would otherwise drive
    # input_tokens negative AND leave cache_read_tokens -- billed at the 10%
    # cache rate in budget.py -- inconsistent with it. One clamp keeps both
    # fields consistent.
    cached = min(cached, prompt_total)
    fresh_input = prompt_total - cached
    # Build a content-blocks raw payload mirroring Anthropic's response
    # shape so callers that inspect resp.raw["content"] (the worker_loop
    # does this to reconstruct the assistant message verbatim) see the
    # same structure regardless of provider.
    raw_content: list[dict[str, Any]] = []
    if reasoning_text:
        # A leading `thinking` block, so every provider yields one shape.
        # Reasoning is NOT promoted into `text`: surfaces that echo
        # `resp.text` (CLI logger, transcripts) would double-print it.
        raw_content.append({"type": "thinking", "thinking": reasoning_text})
    if text:
        raw_content.append({"type": "text", "text": text})
    for tu in tool_uses:
        raw_content.append(
            {
                "type": "tool_use",
                "id": tu["id"],
                "name": tu["name"],
                "input": tu["input"],
            }
        )
    enriched_raw = {**data, "content": raw_content}
    # Prefer provider-reported USD cost when the upstream
    # gateway includes it (OpenRouter does, OpenAI direct does not).
    # Treat negative or non-numeric values as absent.
    reported_cost = 0.0
    raw_cost = usage.get("cost")
    # not-bool: bool subclasses int, and float(True) == 1.0 would record a
    # phantom dollar per call that becomes the AUTHORITATIVE reported figure.
    if isinstance(raw_cost, int | float) and not isinstance(raw_cost, bool) and raw_cost > 0:
        reported_cost = float(raw_cost)
    return ProviderResponse(
        text=text,
        tool_uses=tool_uses,
        stop_reason=stop_reason,
        input_tokens=fresh_input,
        output_tokens=int(usage.get("completion_tokens") or 0),
        cache_read_tokens=cached,
        cache_creation_tokens=0,
        cost_usd=reported_cost,
        raw=enriched_raw,
    )
