# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""ChatGPT-subscription provider (the Codex Responses backend).

Speaks the Responses API at `chatgpt.com/backend-api/codex/responses`,
authorized by the OAuth credential from `agent6 connect chatgpt` plus the
`chatgpt-account-id` header. The backend accepts streaming only, so every
call runs over SSE; the delta callbacks stay optional. `instructions`
carries agent6's own system prompt; with `store=false` the model's
encrypted reasoning items replay verbatim, in wire order, so its chain of
thought survives across tool calls.

Usage draws on the ChatGPT plan's limits, not a metered key, so
`ProviderResponse.cost_usd` stays 0; token counts are still metered for
the budget's token caps. Feedback/rating endpoints are never called: a
rating would opt those turns into provider-side training, so agent6 has no
rating surface at all.
"""

from __future__ import annotations

import contextlib
import json
import re
import time
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from typing import Any

import httpx2

from agent6.budget import BudgetTracker, PlanUsage, PlanWindow
from agent6.providers._openai_recovery import lenient_json_object
from agent6.providers._stream import SseCall, StreamClock, bounded_lines, record_billed_usage
from agent6.providers._transport import ProviderCall
from agent6.providers.chatgpt_oauth import ChatGPTCredential
from agent6.providers.types import (
    ProviderError,
    ProviderResponse,
    ToolDefinition,
    TranscriptRecorder,
)
from agent6.providers.wire import request_url

DEFAULT_MAX_TOKENS = 8192

# Terminal stream events, keyed by their `type`. `completed` and `incomplete`
# both carry the final response object (usage, status, output); `failed`
# carries the error envelope.
_TERMINAL_EVENTS = frozenset({"response.completed", "response.done", "response.incomplete"})

# Backend error codes that mean the plan's usage window is exhausted: carry
# 429 so the loop treats them as retryable-with-backoff, not a provider bug.
_USAGE_LIMIT_CODES = frozenset({"usage_limit_reached", "usage_not_included", "rate_limit_exceeded"})


def responses_input(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """agent6's canonical Anthropic-shape messages -> Responses `input` items.

    Block order is the wire order: text runs flush as one message item;
    `tool_use` becomes a `function_call` item (arguments as a JSON string,
    keyed by `call_id`); `tool_result` becomes `function_call_output` with
    the content flattened to a string. A `thinking` block carrying a
    `chatgpt_reasoning` item replays that raw Responses item in place, only
    WITH its following kept item (orphans violate the paired-item rules);
    any other `thinking` block is display-only and dropped.
    """
    items: list[dict[str, Any]] = []
    # Ids of blank-name tool_use blocks skipped below (a resumed history can
    # carry one another provider emitted); their paired tool_result must be
    # skipped too, or the request carries an output with no matching call and
    # the backend rejects the whole conversation.
    dropped_ids: set[str] = set()
    for msg in messages:
        role = str(msg.get("role", "user"))
        if role not in ("user", "assistant"):
            role = "user"
        content = msg.get("content", "")
        if isinstance(content, list):
            items.extend(_content_items(role, content, dropped_ids))
        elif content:
            items.append(_message_item(role, str(content)))
    return items


def _content_items(role: str, blocks: list[Any], dropped_ids: set[str]) -> list[dict[str, Any]]:
    """One message's content blocks -> input items, text runs batched."""
    items: list[dict[str, Any]] = []
    text_run: list[str] = []
    # A reasoning item is replayed only WITH its following kept item: an
    # orphaned reasoning item (its paired call was dropped) violates the
    # paired-item rules and would 400 the whole request.
    pending_reasoning: list[dict[str, Any]] = []

    def flush() -> None:
        if text_run:
            items.append(_message_item(role, "".join(text_run)))
            text_run.clear()

    for block in blocks:
        if not isinstance(block, dict):
            continue
        btype = block.get("type")
        if btype == "text":
            items.extend(pending_reasoning)
            pending_reasoning.clear()
            text_run.append(str(block.get("text", "")))
        elif btype == "thinking" and role == "assistant":
            item = block.get("chatgpt_reasoning")
            if isinstance(item, dict):
                flush()
                pending_reasoning.append(item)
        elif btype == "tool_use" and role == "assistant":
            if not str(block.get("name") or "").strip():
                dropped_ids.add(str(block.get("id", "")))
                pending_reasoning.clear()
                continue
            flush()
            items.extend(pending_reasoning)
            pending_reasoning.clear()
            items.append(
                {
                    "type": "function_call",
                    "call_id": str(block.get("id", "")),
                    "name": str(block.get("name", "")),
                    "arguments": json.dumps(block.get("input") or {}),
                }
            )
        elif btype == "tool_result":
            if str(block.get("tool_use_id", "")) in dropped_ids:
                continue
            flush()
            items.append(
                {
                    "type": "function_call_output",
                    "call_id": str(block.get("tool_use_id", "")),
                    "output": _tool_result_text(block.get("content", "")),
                }
            )
    flush()
    items.extend(pending_reasoning)
    return items


def _message_item(role: str, text: str) -> dict[str, Any]:
    kind = "output_text" if role == "assistant" else "input_text"
    return {"type": "message", "role": role, "content": [{"type": kind, "text": text}]}


def _tool_result_text(tr_content: Any) -> str:
    """Flatten a tool_result's content to the string the wire wants."""
    if isinstance(tr_content, list):
        parts = [
            str(b.get("text", ""))
            for b in tr_content
            if isinstance(b, dict) and b.get("type") == "text"
        ]
        return "".join(parts) if parts else json.dumps(tr_content)
    return str(tr_content)


def tools_to_responses(tools: list[ToolDefinition]) -> list[dict[str, Any]]:
    """`ToolDefinition`s -> Responses function tools (flat, not nested)."""
    return [
        {
            "type": "function",
            "name": t.name,
            "description": t.description,
            "parameters": t.input_schema,
            "strict": False,
        }
        for t in tools
    ]


def _tool_use_of(item: dict[str, Any], *, n: int) -> dict[str, Any] | None:
    """A `function_call` item -> an Anthropic-shape tool_use, or None for a
    blank name (a malformed call must not enter history; see the OpenAI
    parser's identical drop). Unparseable arguments keep the lenient-repair /
    `_raw_arguments` fallback so dispatch can ask for a valid resend."""
    name = str(item.get("name", "")).strip()
    if not name:
        return None
    args_raw = item.get("arguments", "")
    try:
        parsed = json.loads(args_raw) if args_raw else {}
        if not isinstance(parsed, dict):
            parsed = {"_value": parsed}
    except (json.JSONDecodeError, TypeError):
        parsed = lenient_json_object(str(args_raw)) or {"_raw_arguments": str(args_raw)[:500]}
    return {
        "id": str(item.get("call_id") or item.get("id") or f"call_{n}"),
        "name": name,
        "input": parsed,
    }


def parse_output_items(
    items: list[Any], *, usage: Mapping[str, Any], stop_reason: str
) -> ProviderResponse:
    """Final Responses output items -> `ProviderResponse` (Anthropic shape).

    `raw["content"]` holds one block per output item in wire order: a
    message's text, a reasoning item as a `thinking` block (its summary for
    display, the raw item under `chatgpt_reasoning` for verbatim replay,
    never decrypted), a function_call as `tool_use`. Usage normalisation
    matches the OpenAI parser: the backend's `input_tokens` is the
    cached+fresh total, so cached moves to `cache_read_tokens` and
    `input_tokens` keeps fresh-only semantics.
    """
    text_parts: list[str] = []
    tool_uses: list[dict[str, Any]] = []
    blocks: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        itype = item.get("type")
        if itype == "message":
            text = "".join(
                str(part.get("text", ""))
                for part in item.get("content") or []
                if isinstance(part, dict) and part.get("type") in ("output_text", "text")
            )
            if text:
                text_parts.append(text)
                blocks.append({"type": "text", "text": text})
        elif itype == "reasoning":
            summary = "\n\n".join(
                str(part["text"])
                for part in item.get("summary") or []
                if isinstance(part, dict) and str(part.get("text", ""))
            )
            blocks.append({"type": "thinking", "thinking": summary, "chatgpt_reasoning": item})
        elif itype == "function_call":
            tool_use = _tool_use_of(item, n=len(tool_uses))
            if tool_use is not None:
                tool_uses.append(tool_use)
                blocks.append({"type": "tool_use", **tool_use})
    text = "".join(text_parts)
    if tool_uses and stop_reason == "end_turn":
        # Anthropic-shape semantics: a turn that stopped to call tools says
        # so. This also arms the loop's empty-tool-call detector for this
        # wire (stop says tool_use + nothing came = a retryable contradiction).
        stop_reason = "tool_use"
    details = usage.get("input_tokens_details")
    cached = int(details.get("cached_tokens", 0) or 0) if isinstance(details, Mapping) else 0
    prompt_total = int(usage.get("input_tokens") or 0)
    cached = min(cached, prompt_total)
    return ProviderResponse(
        text=text,
        tool_uses=tuple(tool_uses),
        stop_reason=stop_reason,
        input_tokens=prompt_total - cached,
        output_tokens=int(usage.get("output_tokens") or 0),
        cache_read_tokens=cached,
        cache_creation_tokens=0,
        cost_usd=0.0,
        raw={"content": blocks, "usage": dict(usage), "output": items},
    )


def _no_adapt(status: int | None, text: str, body: dict[str, Any]) -> bool:
    del status, text, body
    return False


def _unreachable_hook(data: dict[str, Any]) -> Any:
    raise ProviderError("chatgpt provider is stream-only")  # pragma: no cover


_WINDOW_HEADER = re.compile(r"^x-codex-(?P<name>.+)-used-percent$")


def _num(value: Any) -> float:
    try:
        return float(value or 0)
    except (TypeError, ValueError):
        return 0.0


def _window_order(name: str) -> tuple[int, str]:
    """Primary first, secondary second, every other family by name."""
    return ({"primary": 0, "secondary": 1}.get(name, 2), name)


def _plan_usage_of(headers: Mapping[str, str]) -> PlanUsage | None:
    """The rate-limit windows off a response's `x-codex-*` headers.

    Every `x-codex-<name>-used-percent` family is one window (`primary`,
    `secondary`, and whatever per-model families the backend adds), with
    its `-window-minutes` and `-reset-at` / `-reset-after-seconds` siblings.
    The percent budget and every plan-usage surface read this one parse;
    None when the backend sent no primary reading (absent or malformed)."""
    lowered = {k.lower(): v for k, v in headers.items()}
    windows: list[PlanWindow] = []
    for key in sorted(lowered):
        m = _WINDOW_HEADER.match(key)
        if m is None:
            continue
        name = m.group("name")
        try:
            used = float(lowered[key])
        except (TypeError, ValueError):
            continue
        resets_at = _num(lowered.get(f"x-codex-{name}-reset-at"))
        if not resets_at:
            resets_at = time.time() + _num(lowered.get(f"x-codex-{name}-reset-after-seconds"))
        windows.append(
            PlanWindow(
                name=name,
                used_percent=used,
                window_minutes=int(_num(lowered.get(f"x-codex-{name}-window-minutes"))),
                resets_at=resets_at,
            )
        )
    if not any(w.name == "primary" for w in windows):
        return None

    def _flag(name: str) -> bool:
        return (lowered.get(name) or "").strip().lower() == "true"

    return PlanUsage(
        windows=tuple(sorted(windows, key=lambda w: _window_order(w.name))),
        has_credits=_flag("x-codex-credits-has-credits"),
        credits_unlimited=_flag("x-codex-credits-unlimited"),
        credits_balance=(lowered.get("x-codex-credits-balance") or "").strip(),
    )


def plan_usage_from_usage_body(body: Mapping[str, Any]) -> PlanUsage | None:
    """The account's plan state off the backend's `/usage` body: every
    `<name>_window` under `rate_limit` is one window, plus its own
    limit-reached verdict and the purchased-credit family. None when the
    body carries no primary window."""
    limits = body.get("rate_limit")
    if not isinstance(limits, Mapping):
        return None
    windows: list[PlanWindow] = []
    for key, raw in limits.items():
        if not (isinstance(key, str) and key.endswith("_window") and isinstance(raw, Mapping)):
            continue
        name = key.removesuffix("_window")
        try:
            used = float(raw.get("used_percent", ""))
        except (TypeError, ValueError):
            continue
        resets_at = _num(raw.get("reset_at"))
        if not resets_at:
            resets_at = time.time() + _num(raw.get("reset_after_seconds"))
        windows.append(
            PlanWindow(
                name=name,
                used_percent=used,
                window_minutes=int(_num(raw.get("limit_window_seconds")) / 60),
                resets_at=resets_at,
            )
        )
    if not any(w.name == "primary" for w in windows):
        return None
    credits = body.get("credits")
    credits = credits if isinstance(credits, Mapping) else {}
    return PlanUsage(
        windows=tuple(sorted(windows, key=lambda w: _window_order(w.name))),
        has_credits=bool(credits.get("has_credits")),
        credits_unlimited=bool(credits.get("unlimited")),
        credits_balance=str(credits.get("balance") or "").strip(),
        limit_reached=bool(limits.get("limit_reached")),
    )


def _stream_error(evt: dict[str, Any]) -> ProviderError:
    """A `response.failed` / `error` frame -> a classified ProviderError."""
    response = evt.get("response")
    err = response.get("error") if isinstance(response, dict) else evt.get("error")
    if not isinstance(err, dict):
        err = {"message": str(err or evt.get("message") or "unknown stream error")}
    code = str(err.get("code") or "")
    status = 429 if code in _USAGE_LIMIT_CODES else None
    detail = str(err.get("message") or code or "unknown error")
    if code in _USAGE_LIMIT_CODES:
        plan = str(err.get("plan_type") or "")
        detail += f" (ChatGPT {plan} plan usage limit)" if plan else " (ChatGPT usage limit)"
    return ProviderError(f"ChatGPT stream error: {code or 'error'}: {detail}", status_code=status)


@dataclass(frozen=True, slots=True)
class ChatGPTProvider:
    """Stateless provider for the ChatGPT-subscription Codex backend."""

    model: str
    credential: ChatGPTCredential
    account_id: str
    base_url: str
    extra_headers: tuple[tuple[str, str], ...] = ()
    extra_body: dict[str, Any] = field(default_factory=dict)
    extra_query: dict[str, str] = field(default_factory=dict)
    timeout_s: float = 600.0
    transcript_sink: TranscriptRecorder | None = None
    budget: BudgetTracker | None = None
    # Default reasoning effort, wired from `[models.<role>].effort`; the
    # model's own default applies when unset, and "off" sends the wire's
    # explicit "none".
    reasoning_effort: str | None = None
    # Stable per-provider id: the backend's `prompt_cache_key` (<= 64 chars)
    # and `session-id` header, so caching keys to this run's conversation.
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    # One usage preflight per provider (a mutable cell on a frozen dataclass).
    _preflighted: list[bool] = field(default_factory=lambda: [False])

    def preflight(self) -> PlanUsage | None:
        """The account's plan state BEFORE any call, off the backend's
        `/usage` (same host as `base_url`): both windows and the credit
        family. Best effort: any failure reads as no reading, never as a
        block. The body is parsed and dropped (it carries the account's
        email), never recorded."""
        try:
            token = self.credential.token()
            resp = httpx2.get(
                f"{self.base_url.rstrip('/')}/usage",
                headers=self._build_headers(token),
                timeout=20.0,
            )
            if resp.status_code != 200:
                return None
            body = resp.json()
        except (httpx2.HTTPError, ValueError, OSError):
            return None
        return plan_usage_from_usage_body(body) if isinstance(body, dict) else None

    def _build_headers(self, token: str) -> dict[str, str]:
        headers = {
            "content-type": "application/json",
            "authorization": f"Bearer {token}",
            "chatgpt-account-id": self.account_id,
            "openai-beta": "responses=experimental",
            "originator": "agent6",
            "session-id": self.session_id,
        }
        for k, v in self.extra_headers:
            headers[k.lower()] = v
        return headers

    def call(
        self,
        *,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[ToolDefinition] | None = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float | None = None,
        extended_thinking: dict[str, Any] | None = None,
        reasoning_effort: str | None = None,
        text_delta_callback: Callable[[str], None] | None = None,
        thinking_delta_callback: Callable[[str], None] | None = None,
        should_abort: Callable[[], bool] | None = None,
        should_interrupt: Callable[[], bool] | None = None,
    ) -> ProviderResponse:
        # The backend sizes output itself (`max_output_tokens` is not part of
        # the Codex dialect), hard-rejects `temperature` (400 "Unsupported
        # parameter"; effort is the only sampling knob), and the
        # Anthropic-shaped `extended_thinking` has no mapping here either.
        del max_tokens, temperature, extended_thinking
        if self.budget is not None:
            if not self._preflighted[0]:
                self._preflighted[0] = True
                plan = self.preflight()
                if plan is not None:
                    self.budget.record_plan_preflight(self.model, plan)
            self.budget.check()
        url, _ = request_url(
            api_format="chatgpt",
            deployment="direct",
            base_url=self.base_url,
            model=self.model,
            streaming=True,
            extra_query=self.extra_query,
        )
        body: dict[str, Any] = {
            "model": self.model,
            "instructions": system,
            "input": responses_input(messages),
            "tool_choice": "auto",
            "parallel_tool_calls": True,
            "store": False,
            "stream": True,
            # Explicit: the encrypted reasoning items we replay for stateless
            # chain-of-thought continuity across tool calls.
            "include": ["reasoning.encrypted_content"],
            "prompt_cache_key": self.session_id,
        }
        if tools:
            body["tools"] = tools_to_responses(tools)
        effort = reasoning_effort if reasoning_effort is not None else self.reasoning_effort
        if effort:
            # The wire has an explicit "none" (verified live); merely omitting
            # the field would leave the model's own default on, so "off" maps
            # to it rather than silently meaning "default".
            wire = "none" if effort == "off" else effort
            body["reasoning"] = {"effort": wire, "summary": "auto"}
        if self.extra_body:
            reserved = {
                "model",
                "instructions",
                "input",
                "tools",
                "tool_choice",
                "store",
                "stream",
                "include",
                "prompt_cache_key",
            }
            body.update({k: v for k, v in self.extra_body.items() if k not in reserved})

        return ProviderCall(
            api_label="ChatGPT",
            api_format="chatgpt",
            url=url,
            body=body,
            timeout_s=self.timeout_s,
            api_key="",
            credential=self.credential,
            transcript_sink=self.transcript_sink,
            budget=self.budget,
            model=self.model,
            build_headers=self._build_headers,
            adapt_400=_no_adapt,
            adapt_attempts=0,
            # Stream-only: ProviderCall never takes its non-streaming branch,
            # so these two hooks are unreachable; metering happens in-stream.
            require_metered=_unreachable_hook,
            parse=_unreachable_hook,
            stream=lambda attempt_headers: self._call_streaming(
                url=url,
                headers=attempt_headers,
                body=body,
                text_delta_callback=text_delta_callback,
                thinking_delta_callback=thinking_delta_callback,
                should_abort=should_abort,
                should_interrupt=should_interrupt,
            ),
        ).run()

    def _call_streaming(  # noqa: PLR0915
        self,
        *,
        url: str,
        headers: dict[str, str],
        body: dict[str, Any],
        text_delta_callback: Callable[[str], None] | None,
        thinking_delta_callback: Callable[[str], None] | None,
        should_abort: Callable[[], bool] | None,
        should_interrupt: Callable[[], bool] | None,
    ) -> ProviderResponse:
        """One SSE round-trip against the Responses backend.

        Event shape: each `data:` frame is a JSON object whose `type` names
        the event. Deltas (`response.output_text.delta`,
        `response.reasoning_*.delta`) feed the callbacks only; the final
        content comes from `response.output_item.done` items, reconciled
        against the terminal `response.completed` object (usage lives
        there). A stream ending without a terminal event is a cut, never a
        completed turn.
        """
        stream_headers = dict(headers)
        stream_headers["accept"] = "text/event-stream"

        items: list[Any] = []
        delta_text: list[str] = []
        usage: dict[str, Any] = {}
        plan_usage: PlanUsage | None = None
        stop_reason = ""
        done = False

        call = SseCall(
            api_label="ChatGPT",
            api_format="chatgpt",
            url=url,
            headers=stream_headers,
            body=body,
            timeout_s=self.timeout_s,
            transcript_sink=self.transcript_sink,
            should_abort=should_abort,
            should_interrupt=should_interrupt,
        )

        def consume(resp: httpx2.Response, clock: StreamClock) -> None:  # noqa: PLR0912
            nonlocal usage, stop_reason, done, plan_usage
            plan_usage = _plan_usage_of(resp.headers)
            for raw_line in bounded_lines(resp):
                line = raw_line.strip()
                if not line or line.startswith(":") or not line.startswith("data:"):
                    continue
                clock.mark_data()
                data_str = line[5:].strip()
                if not data_str or data_str == "[DONE]":
                    continue
                try:
                    evt: dict[str, Any] = json.loads(data_str)
                except json.JSONDecodeError:
                    continue
                kind = str(evt.get("type", ""))
                if kind == "response.output_text.delta":
                    piece = str(evt.get("delta", ""))
                    if piece:
                        clock.mark_output()
                        delta_text.append(piece)
                        if text_delta_callback is not None:
                            with contextlib.suppress(Exception):
                                text_delta_callback(piece)
                elif kind in (
                    "response.reasoning_summary_text.delta",
                    "response.reasoning_text.delta",
                ):
                    piece = str(evt.get("delta", ""))
                    if piece:
                        clock.mark_output()
                        if thinking_delta_callback is not None:
                            with contextlib.suppress(Exception):
                                thinking_delta_callback(piece)
                elif kind == "response.output_item.done":
                    clock.mark_output()
                    items.append(evt.get("item"))
                elif kind in ("response.failed", "error"):
                    call.record(status=0, response=data_str[:8192])
                    raise _stream_error(evt)
                elif kind in _TERMINAL_EVENTS:
                    response = evt.get("response") or {}
                    evt_usage = response.get("usage")
                    if isinstance(evt_usage, dict):
                        usage = evt_usage
                    if not items and isinstance(response.get("output"), list):
                        items.extend(response["output"])
                    if kind == "response.incomplete":
                        reason = str((response.get("incomplete_details") or {}).get("reason") or "")
                        stop_reason = (
                            "max_tokens"
                            if reason == "max_output_tokens"
                            else (reason or "incomplete")
                        )
                    else:
                        stop_reason = "end_turn"
                    done = True
                    return

        def _record_billed() -> None:
            if not usage and plan_usage is None:
                return
            billed = parse_output_items([], usage=usage, stop_reason="")
            record_billed_usage(
                self.budget,
                self.model,
                input_tokens=billed.input_tokens,
                output_tokens=billed.output_tokens,
                cache_read_tokens=billed.cache_read_tokens,
                cache_creation_tokens=0,
                cost_usd=0.0,
                plan_usage=plan_usage,
            )

        try:
            call.run(consume)
        except BaseException:
            _record_billed()
            raise

        if not done:
            _record_billed()
            call.record(status=0, response="stream ended without a terminal response event")
            raise ProviderError(
                f"ChatGPT stream from {url} ended without response.completed;"
                " upstream appears cut off."
            )

        parsed = parse_output_items(items, usage=usage, stop_reason=stop_reason)
        if not parsed.text and delta_text:
            # A backend that streamed text deltas but no final message item:
            # keep what the operator already watched arrive, ahead of the
            # turn's other blocks.
            text = "".join(delta_text)
            parsed = replace(
                parsed,
                text=text,
                raw={
                    **parsed.raw,
                    "content": [{"type": "text", "text": text}, *parsed.raw["content"]],
                },
            )
        call.record(
            status=200,
            response={
                "output": parsed.raw.get("output", []),
                "usage": usage,
                "status": stop_reason,
            },
        )
        if self.budget is not None:
            if int(usage.get("input_tokens") or 0) <= 0:
                # Billed for what it generated, and the plan window moved: on
                # the ledger before the refusal.
                _record_billed()
                raise ProviderError(
                    "ChatGPT stream reported no usage input tokens;"
                    " budgeted runs require provider usage accounting"
                )
            self.budget.record(
                model=self.model,
                input_tokens=parsed.input_tokens,
                output_tokens=parsed.output_tokens,
                cache_read_tokens=parsed.cache_read_tokens,
                cache_creation_tokens=0,
                cost_usd=0.0,
                plan_usage=plan_usage,
            )
        return parsed
