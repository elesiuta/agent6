# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""OpenAI Chat Completions-compatible provider.

Works against any endpoint speaking the OpenAI Chat Completions API: OpenAI,
OpenRouter, Ollama (`/v1`), vLLM, LM Studio, llama.cpp's server, Moonshot,
DeepSeek. HTTP transport and SSE lifecycle are shared with the Anthropic
provider (`_transport.py`, `_stream.py`); both use httpx2 directly (no SDK)
for a smaller audit surface.

agent6's internal lingua franca is Anthropic content-blocks (text + tool_use +
tool_result inline, the most expressive shape); translation both ways lives in
`_openai_messages` / `_openai_parse`, so workflow code sees one shape across
providers. Deliberately NOT translated: `cache_control` markers are stripped
(OpenAI caches server-side), and Anthropic's `extended_thinking` budget_tokens
has no equivalent -- OpenAI reasoning is the `reasoning_effort` knob, wired
from `[models.<role>].effort`.
"""

from __future__ import annotations

import contextlib
import json
import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlsplit

import httpx2

from agent6.budget import BudgetTracker
from agent6.providers._openai_messages import anthropic_to_openai_messages, tools_to_openai
from agent6.providers._openai_parse import parse_response
from agent6.providers._stream import SseCall, StreamClock, bounded_lines, record_billed_usage
from agent6.providers._transport import ProviderCall, envelope_status
from agent6.providers.token_command import CommandToken
from agent6.providers.types import (
    ProviderError,
    ProviderResponse,
    ToolDefinition,
    TranscriptRecorder,
)
from agent6.providers.wire import AuthStyle, Deployment, auth_header, request_url

OPENAI_DEFAULT_BASE_URL = "https://api.openai.com/v1"
DEFAULT_MAX_TOKENS = 8192

# Reasoning models stream a separate `reasoning_content` whose tokens count
# against `max_tokens` SERVER-SIDE, so at the ordinary per-call cap reasoning
# consumes the budget and the assistant `content`/`tool_calls` truncate
# mid-message: the loop sees stop_reason="length", empty text, no tool calls,
# and stalls. A floor gives reasoning room; the budget tracker is unaffected
# (it counts every emitted token via usage.completion_tokens).
REASONING_MODEL_MIN_MAX_TOKENS = 32768
_REASONING_MODEL_HINTS: tuple[str, ...] = (
    "thinking",
    "reasoning",
    "deepseek-r1",
    "qwq",
    "o1-",
    "o3-",
    "o4-",
    # Reasoning-channel emitters whose model name does not advertise it: they
    # return finish_reason="length" with empty content + empty tool_calls unless
    # given output headroom. Match the FAMILY, not one generation; a false
    # positive is harmless (the floor only raises a ceiling).
    "kimi-k",
    "minimax-m2",
    "nemotron",
    "glm",
)


def _require_metered_usage(usage: object, *, source: str) -> None:
    """Fail closed when a budgeted OpenAI-compatible call cannot be metered.

    Presence alone is not enough: a gateway with usage tracking disabled returns
    `prompt_tokens: 0` and every turn records zero, so the budget never trips.
    `prompt_tokens` is total input (cached + fresh) and is never legitimately 0
    for a real call, so require it strictly positive; a run must not proceed on a
    call it cannot meter."""
    if isinstance(usage, Mapping):
        # Coerce numerically, mirroring parse_response: a gateway serializing
        # counts as JSON floats/strings (700.0, "700") is meterable.
        # Absent/zero/non-numeric still fails closed below. completion_tokens
        # presence is not required: the contract gates on the input side only.
        try:
            prompt = int(usage.get("prompt_tokens") or 0)
        except (TypeError, ValueError):
            prompt = 0
        if prompt > 0:
            return
    # No status code: a usage-less reply is a stream/gateway integrity failure
    # (a degenerate stream the gateway cut, a proxy dropping the usage frame),
    # so it rides the loop's bounded retry lane -- the failed attempt returns
    # no response, so nothing unmetered enters the conversation, and repeated
    # failure ends the run. A permanent classification here (a fake 422) would
    # let ONE mangled stream kill a budgeted run with its budget unspent.
    raise ProviderError(
        f"{source} reported no usage input tokens (usage.prompt_tokens missing or 0); "
        "budgeted runs require provider usage accounting"
    )


def _is_reasoning_model(model: str) -> bool:
    """True if `model` looks like a reasoning model that emits
    `reasoning_content` separately from `content`. Gates the effort DEFAULT
    (reasoning_effort="low"), a measured behaviour change -- so it stays the
    measured family set, NOT the broader floor set below."""
    lowered = model.lower()
    return any(hint in lowered for hint in _REASONING_MODEL_HINTS)


# The max_tokens FLOOR matches more broadly than the effort default: raising the
# token ceiling only avoids a truncated reply, it never changes model behaviour,
# so it is safe to catch aliases the measured effort set does not. `kimi-latest`
# (Moonshot's rolling alias) emits reasoning_content and starves at the 16k
# default, but the `kimi-k` family match misses it -- and adding it to
# `_REASONING_MODEL_HINTS` would also pin it to the UNMEASURED effort="low"
# default for whatever it currently resolves to. Floor only.
_REASONING_FLOOR_ONLY_HINTS: tuple[str, ...] = ("kimi-latest",)


def _needs_reasoning_headroom(model: str) -> bool:
    """True if `model` needs the max_tokens floor: any reasoning model, OpenAI's
    own o-series / gpt-5 (which reason and starve just as hard, but are matched
    narrowly only for the direct-host param rename), plus reasoning aliases
    (kimi-latest) not in the effort set. Safe to match broadly -- the floor raises
    a ceiling, it does not change behaviour, so a false positive costs nothing."""
    lowered = model.lower()
    return (
        _is_reasoning_model(model)
        or _is_openai_direct_reasoning_model(model)
        or any(h in lowered for h in _REASONING_FLOOR_ONLY_HINTS)
    )


# OpenAI's OWN reasoning families (o-series + gpt-5). On the api.openai.com
# direct host these reject the legacy `max_tokens` param (400, "Use
# max_completion_tokens") and reject `temperature != 1`. Kept narrower than
# `_is_reasoning_model` on purpose: third-party reasoning models (kimi,
# deepseek, qwq) are never served by api.openai.com, so they must NOT trigger
# the rename even if someone points them at the default base_url.
_OPENAI_DIRECT_REASONING_PREFIXES: tuple[str, ...] = ("o1", "o3", "o4", "gpt-5")


def _is_openai_direct_reasoning_model(model: str) -> bool:
    """True if `model` is one of OpenAI's own o-series/gpt-5 reasoning
    models (only meaningful when the request targets api.openai.com)."""
    lowered = model.lower()
    return any(
        lowered == p or lowered.startswith(p + "-") for p in _OPENAI_DIRECT_REASONING_PREFIXES
    )


_EFFORT_LEVELS = ("off", "low", "medium", "high", "xhigh", "max")


def is_openai_direct_host(base_url: str, deployment: str) -> bool:
    """True when requests go to api.openai.com itself, whose o-series/gpt-5
    models take parameters no other openai-compatible host does."""
    return deployment == "direct" and urlsplit(base_url).hostname == "api.openai.com"


def sent_reasoning_effort(
    model: str, configured: str | None, *, direct_openai: bool = False
) -> str | None:
    """The reasoning effort this role resolves to for *model*, or None when the
    model takes no reasoning knob at all and the request carries none.

    One owner for the rule `config show` prints and `complete` sends.
    Precedence: *configured* (the role's `effort`, or a per-call override) >
    `AGENT6_REASONING_EFFORT` > `low`. `off` is a resolved level, not a wire
    value: `complete` maps it per host, omitting the parameter on
    api.openai.com (whose o-series always reasons) and sending
    `{"enabled": false}` elsewhere.
    """
    if not (
        _is_reasoning_model(model) or (direct_openai and _is_openai_direct_reasoning_model(model))
    ):
        return None
    if configured is not None:
        return configured.strip().lower()
    env_override = os.environ.get("AGENT6_REASONING_EFFORT", "").strip().lower()
    return env_override if env_override in _EFFORT_LEVELS else "low"


@dataclass(frozen=True, slots=True)
class OpenAIProvider:
    """Stateless OpenAI Chat Completions-compatible provider.

    `api_key` may be empty for unauthenticated local endpoints (Ollama,
    llama.cpp's `server`); when empty, no `Authorization` header is sent.
    """

    api_key: str
    model: str
    base_url: str = OPENAI_DEFAULT_BASE_URL
    deployment: Deployment = "direct"
    # Auth header style (config AuthConfig.style): "bearer" (default),
    # "api_key_header" (Azure's `api-key`), or "none" (local endpoints).
    auth_style: AuthStyle = "bearer"
    extra_headers: tuple[tuple[str, str], ...] = ()
    # Provider-specific JSON merged into every request body (e.g. OpenRouter
    # `provider` routing, see config OpenAIProviderEntry.extra_body). Keys here
    # override computed tuning fields, EXCEPT the structural set filtered in
    # `call` (messages/model/stream/stream_options/tools/tool_choice/
    # response_format/n).
    extra_body: dict[str, Any] = field(default_factory=dict)
    # Static URL query params merged onto every request (e.g. Azure's
    # api-version). See config extra_query.
    extra_query: dict[str, str] = field(default_factory=dict)
    timeout_s: float = 120.0
    transcript_sink: TranscriptRecorder | None = None
    budget: BudgetTracker | None = None
    # Default reasoning effort for this provider (config EffortLevel), wired
    # from the role's `[models.<role>].effort`. A per-call `reasoning_effort`
    # argument takes precedence; below this sits the AGENT6_REASONING_EFFORT
    # env override. Only affects OpenAI-compatible reasoning models.
    reasoning_effort: str | None = None
    # Short-lived bearer source (config `token_command`). When set, it mints
    # the `Authorization` token per call instead of `api_key`, and a 401/403
    # triggers one refresh + retry. The object is internally mutable (cache),
    # which is why the otherwise-frozen provider holds only a reference to it.
    credential: CommandToken | None = None
    # Some OpenAI-compatible backends agent6 cannot fingerprint up front (an Azure
    # o-series/gpt-5 deployment has an arbitrary deployment name) reject the
    # legacy `max_tokens` with a 400 saying to use `max_completion_tokens`,
    # and/or reject any explicit `temperature`. On that 400 the call adapts
    # the body and retries once, latching here so the rest of the run builds
    # the right body first time. 1-element lists because the dataclass is
    # frozen but the lists are mutable (same pattern as AnthropicProvider).
    _use_max_completion_tokens: list[bool] = field(default_factory=lambda: [False])
    _omit_temperature: list[bool] = field(default_factory=lambda: [False])

    @property
    def endpoint(self) -> str:
        return self.base_url.rstrip("/") + "/chat/completions"

    def _adapt_body_for_400(self, status: int | None, text: str, body: dict[str, Any]) -> bool:
        """Mutate `body` to satisfy a parameter-rejection 400 and latch the
        provider so later calls build the right body first time. Covers the
        two rejections a reasoning deployment we cannot fingerprint up front
        (an Azure o-series/gpt-5 deployment has an arbitrary name) sends:
        "use max_completion_tokens" and "temperature is not supported".
        Returns True when an adaptation was made (caller retries once)."""
        if status != 400:
            return False
        if "max_tokens" in body and "max_completion_tokens" in (text or ""):
            self._use_max_completion_tokens[0] = True
            body["max_completion_tokens"] = body.pop("max_tokens")
            return True
        if "temperature" in body and "temperature" in (text or "").lower():
            self._omit_temperature[0] = True
            body.pop("temperature", None)
            return True
        return False

    def _build_headers(self, token: str) -> dict[str, str]:
        """Per-attempt request headers. Rebuilt each attempt because a
        `token_command` credential mints a short-lived bearer that takes
        precedence over the static api_key; on a 401/403 the transport
        refreshes it once and retries, so an expired token self-heals."""
        headers: dict[str, str] = {"content-type": "application/json"}
        authed = auth_header(self.auth_style, token)
        if authed is not None:
            headers[authed[0]] = authed[1]
        for k, v in self.extra_headers:
            headers[k.lower()] = v
        return headers

    @classmethod
    def from_env(
        cls,
        *,
        model: str,
        env_var: str | None,
        base_url: str = OPENAI_DEFAULT_BASE_URL,
        extra_headers: dict[str, str] | None = None,
        timeout_s: float = 120.0,
        transcript_sink: TranscriptRecorder | None = None,
        budget: BudgetTracker | None = None,
    ) -> OpenAIProvider:
        # env_var is optional: Ollama and similar local endpoints take no
        # API key. If it's set, an empty value is still allowed (treated as "no key").
        key = "" if env_var is None else os.environ.get(env_var, "").strip()
        return cls(
            api_key=key,
            model=model,
            base_url=base_url,
            extra_headers=tuple(sorted((extra_headers or {}).items())),
            timeout_s=timeout_s,
            transcript_sink=transcript_sink,
            budget=budget,
        )

    def call(  # noqa: PLR0912
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
        # extended_thinking is Anthropic-shaped (`budget_tokens`).
        # OpenAI reasoning models use `reasoning_effort` instead; no
        # 1:1 mapping. Silently no-op so cross-provider workflow code
        # doesn't have to branch.
        del extended_thinking
        if self.budget is not None:
            self.budget.check()

        oai_messages = anthropic_to_openai_messages(system, messages)

        # Lift max_tokens for reasoning models so reasoning_content
        # doesn't starve the actual assistant content + tool_calls. See
        # REASONING_MODEL_MIN_MAX_TOKENS for the rationale.
        effective_max_tokens = max_tokens
        if (
            _needs_reasoning_headroom(self.model)
            and effective_max_tokens < REASONING_MODEL_MIN_MAX_TOKENS
        ):
            effective_max_tokens = REASONING_MODEL_MIN_MAX_TOKENS

        streaming = text_delta_callback is not None or thinking_delta_callback is not None
        url, model_in_body = request_url(
            api_format="openai",
            deployment=self.deployment,
            base_url=self.base_url,
            model=self.model,
            streaming=streaming,
            extra_query=self.extra_query,
        )
        # OpenAI-direct o-series/reasoning models (o1/o3/o4/gpt-5-style)
        # REJECT the legacy `max_tokens` parameter with a hard 400
        # ("Use max_completion_tokens"), and reject `temperature != 1`.
        # They are reached only on the OpenAI-direct host; other
        # openai-compatible hosts (OpenRouter, Azure, vLLM, llama.cpp) still
        # require `max_tokens` and accept arbitrary temperature, so gate the
        # rename on host + model. OpenRouter masked this by normalising
        # `max_tokens` -> `max_completion_tokens` itself.
        is_openai_direct = is_openai_direct_host(self.base_url, self.deployment)
        is_openai_direct_reasoning = is_openai_direct and _is_openai_direct_reasoning_model(
            self.model
        )
        body: dict[str, Any] = {"messages": oai_messages}
        if is_openai_direct_reasoning or self._use_max_completion_tokens[0]:
            body["max_completion_tokens"] = effective_max_tokens
        else:
            body["max_tokens"] = effective_max_tokens
        # Direct/Vertex carry the model in the body; Azure carries the
        # deployment name in the URL path, so omit it from the body there.
        if model_in_body:
            body["model"] = self.model
        # The reasoning knob differs per host, and the wrong one is silently
        # ignored rather than rejected:
        #   OpenRouter-style: nested `reasoning.effort`; top-level
        #     `reasoning_effort` and `reasoning.max_tokens` are no-ops, and
        #     `off` must SEND `{"enabled": False}` (omitting leaves it on).
        #   api.openai.com o-series/gpt-5: top-level `reasoning_effort`; the
        #     nested object 400s as an unknown parameter.
        # `is_openai_direct_reasoning` does not imply `_is_reasoning_model`,
        # so gate on both, else the configured effort is dropped for exactly the
        # models whose only control is the top-level one. Suppression is never
        # automatic (measured: bench/perf/README.md).
        effort = sent_reasoning_effort(
            self.model,
            reasoning_effort if reasoning_effort is not None else self.reasoning_effort,
            direct_openai=is_openai_direct,
        )
        if effort is not None:
            if is_openai_direct_reasoning:
                # api.openai.com Chat Completions o-series/gpt-5 take a TOP-LEVEL
                # `reasoning_effort` (low/medium/high), NOT the nested
                # `reasoning` object OpenRouter invented -- sending the nested
                # object there is an unknown parameter and 400s. Reasoning cannot
                # be disabled on o-series, so "off" omits the param (server
                # default) rather than sending {"enabled": False}.
                if effort != "off":
                    body["reasoning_effort"] = effort
            elif effort == "off":
                body["reasoning"] = {"enabled": False}
            else:
                body["reasoning"] = {"effort": effort}
        # OpenAI-direct o-series/reasoning models reject any explicit
        # `temperature` (only the server default is accepted), so omit it
        # there. Other hosts forward it as-is (until a 400 latches the omit).
        if (
            temperature is not None
            and not is_openai_direct_reasoning
            and not self._omit_temperature[0]
        ):
            body["temperature"] = temperature
        if tools:
            body["tools"] = tools_to_openai(tools)
        # Operator-supplied body extras (e.g. OpenRouter `provider` routing to
        # pin a caching/fast backend). Merged last so it can override computed
        # tuning keys, never the structural keys: replacing
        # `messages`/`model` would silently send a different request, and
        # flipping `stream` would make the non-streaming path get an SSE body
        # that `resp.json()` can't parse. Those are filtered out.
        if self.extra_body:
            # Structural keys only: the conversation, the tool schema, tool
            # choice, and the response shape the parser reads (`n` > 1 and a
            # response_format change both break choices[0]-as-the-answer).
            reserved = {
                "messages",
                "model",
                "stream",
                "stream_options",
                "tools",
                "tool_choice",
                "response_format",
                "n",
            }
            body.update({k: v for k, v in self.extra_body.items() if k not in reserved})
        # Names of the tools actually offered this turn. Used purely as
        # a guard for the text-embedded-tool-call recovery in
        # `parse_response`: we only ever coerce a text blob into a
        # tool_use when its `name` matches a tool we really offered, so
        # well-behaved models (native tool_calls) and models that happen
        # to answer with JSON are never affected.
        tool_names = frozenset(t.name for t in tools) if tools else frozenset()
        # Per-tool input JSON Schemas, keyed by name. Used by the
        # text-embedded-tool-call recovery to coerce a `<parameter>` string
        # value to its declared type (array/object/integer/...) so a leaked
        # Qwen-style XML call rebuilds correctly. Empty when no tools.
        tool_schemas = {t.name: t.input_schema for t in tools} if tools else {}

        # Streaming is chosen by the caller supplying a delta callback. It is
        # also the only reliable path for OpenRouter-style gateways whose
        # `: OPENROUTER PROCESSING` SSE comment heartbeats land in `resp.text`
        # as garbage on the non-streaming path and break `resp.json()`; bench
        # shell scripts force it via AGENT6_FORCE_STREAM=1 (the CLI translates
        # that into a no-op callback).
        return ProviderCall(
            api_label="OpenAI",
            api_format="openai",
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
            adapt_attempts=int("max_tokens" in body) + int("temperature" in body),
            require_metered=lambda data: _require_metered_usage(
                data.get("usage"), source="OpenAI response"
            ),
            parse=lambda data: parse_response(
                data, tool_names=tool_names, tool_schemas=tool_schemas
            ),
            stream=(
                lambda attempt_headers: self._call_streaming(
                    url=url,
                    headers=attempt_headers,
                    body=body,
                    text_delta_callback=text_delta_callback,
                    thinking_delta_callback=thinking_delta_callback,
                    should_abort=should_abort,
                    should_interrupt=should_interrupt,
                    tool_names=tool_names,
                    tool_schemas=tool_schemas,
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
        tool_names: frozenset[str] = frozenset(),
        tool_schemas: dict[str, dict[str, Any]] | None = None,
    ) -> ProviderResponse:
        """SSE streaming variant of the OpenAI Chat Completions call.

        The stream lifecycle (idle watchdog, operator stop/steer, teardown
        classification) is `providers._stream.SseCall`; this method owns
        the Chat Completions event shape:

        * Single `data:` line per frame (no `event:` typing); frames
          are JSON objects with a `choices` array carrying `delta`.
        * Tool calls stream as `choices[0].delta.tool_calls[]` with an
          `index` field; id + name arrive once, `function.arguments`
          arrives across many chunks and must be concatenated per
          index.
        * Reasoning models surface a separate `delta.reasoning_content`
          (Kimi, DeepSeek) or `delta.reasoning` (OpenRouter).
        * Usage only arrives if `stream_options.include_usage` is set
          and lands in a terminal chunk whose `choices` is `[]`.
        * `data: [DONE]` marks end of stream.
        * Gateways like OpenRouter emit SSE comment heartbeats
          (`:OPENROUTER PROCESSING`) for long requests. `iter_lines`
          surfaces them as lines starting with `:`; we skip those.
        """
        body = dict(body)
        body["stream"] = True
        body["stream_options"] = {"include_usage": True}
        stream_headers = dict(headers)
        stream_headers["accept"] = "text/event-stream"

        text_parts: list[str] = []
        reasoning_parts: list[str] = []
        # tool_calls keyed by chunk-level `index` (not the call's
        # external id, which sometimes arrives late).
        tool_calls: dict[int, dict[str, Any]] = {}
        tool_arg_buf: dict[int, list[str]] = {}
        finish_reason = ""
        usage: dict[str, Any] = {}
        # Stream-completion tracking: a legit stream ends with `[DONE]` and/or a
        # non-empty `finish_reason`. A stream that ends with neither was cut off
        # (gateway timed out the upstream and closed the body cleanly, the same
        # failure family OpenRouter delivers as a mid-stream `error` frame); its
        # half-assembled content must NOT be returned as a completed turn.
        done_seen = False

        call = SseCall(
            api_label="OpenAI",
            api_format="openai",
            url=url,
            headers=stream_headers,
            body=body,
            timeout_s=self.timeout_s,
            transcript_sink=self.transcript_sink,
            should_abort=should_abort,
            should_interrupt=should_interrupt,
        )

        def consume(resp: httpx2.Response, clock: StreamClock) -> None:  # noqa: PLR0912, PLR0915
            nonlocal finish_reason, usage, done_seen
            for raw_line in bounded_lines(resp):
                line = raw_line.strip()
                if not line:
                    continue
                # SSE comment heartbeats (OpenRouter, etc). Deliberately NOT
                # marked on the clock -- heartbeats are exactly the bytes that
                # mask an upstream hang from httpx2's read timeout.
                if line.startswith(":"):
                    continue
                if not line.startswith("data:"):
                    continue
                # Real SSE data line. Reset the idle clock; the watchdog is
                # satisfied as long as these keep arriving at all (even
                # `[DONE]` counts as progress). NOTE: mark_output (the switch
                # to the short mid-stream idle timeout) happens later, only on
                # the first real CONTENT token -- an empty role/keepalive delta
                # arrives immediately and must not end the generous prefill
                # budget before the model has actually started producing output.
                clock.mark_data()
                data_str = line[5:].strip()
                if not data_str or data_str == "[DONE]":
                    if data_str == "[DONE]":
                        done_seen = True
                        return
                    continue
                try:
                    evt: dict[str, Any] = json.loads(data_str)
                except json.JSONDecodeError:
                    continue
                # Mid-stream error frame (OpenRouter/OpenAI/LiteLLM deliver an
                # upstream 5xx/429/4xx this way, then end the stream). Surface it
                # instead of silently returning the partial turn, mirroring the
                # Anthropic `error` event. Carry the upstream status like the
                # non-streaming 2xx-envelope path -- streaming is the default, so
                # a permanent code (402/insufficient_quota) delivered mid-stream
                # would otherwise be retried every turn and lose its hint.
                err = evt.get("error")
                if isinstance(err, dict):
                    call.record(status=0, response=data_str[:8192])
                    raise ProviderError(
                        f"OpenAI stream error: {err.get('code')}: {err.get('message')}",
                        status_code=envelope_status(err),
                    )
                evt_usage = evt.get("usage")
                if isinstance(evt_usage, dict):
                    usage = evt_usage
                choices = evt.get("choices") or []
                if not choices:
                    continue
                choice = choices[0]
                if not isinstance(choice, dict):
                    continue
                fr = choice.get("finish_reason")
                if fr:
                    finish_reason = str(fr)
                delta = choice.get("delta") or {}
                if not isinstance(delta, dict):
                    continue
                content = delta.get("content")
                if isinstance(content, str) and content:
                    clock.mark_output()  # real output: the mid-stream idle budget applies
                    # Accumulation is unconditional; the callback is optional
                    # (streaming may be triggered by thinking_delta alone).
                    text_parts.append(content)
                    if text_delta_callback is not None:
                        with contextlib.suppress(Exception):
                            text_delta_callback(content)
                reasoning = delta.get("reasoning_content") or delta.get("reasoning")
                if isinstance(reasoning, str) and reasoning:
                    clock.mark_output()  # streamed reasoning counts as output too
                    reasoning_parts.append(reasoning)
                    if thinking_delta_callback is not None:
                        with contextlib.suppress(Exception):
                            thinking_delta_callback(reasoning)
                raw_tc = delta.get("tool_calls") or []
                if not isinstance(raw_tc, list):
                    continue
                if raw_tc:
                    clock.mark_output()  # tool-call tokens are real output
                for tc in raw_tc:
                    if not isinstance(tc, dict):
                        continue
                    raw_idx = tc.get("index")
                    tc_id = str(tc.get("id") or "")
                    if raw_idx is not None:
                        idx = int(raw_idx)
                    elif tc_id and any(s["id"] == tc_id for s in tool_calls.values()):
                        # Indexless delta continuing a known call: route by id.
                        idx = next(i for i, s in tool_calls.items() if s["id"] == tc_id)
                    elif tc_id and tool_calls:
                        # Indexless chunk carrying a NEW id (a gateway that
                        # sends whole calls in one chunk without index
                        # fields): open a fresh slot instead of collapsing
                        # every call onto slot 0 (which overwrote the first
                        # call and concatenated both argument strings).
                        idx = max(tool_calls) + 1
                    else:
                        idx = max(tool_calls) if tool_calls else 0
                    slot = tool_calls.setdefault(
                        idx,
                        {
                            "id": "",
                            "type": "function",
                            "function": {"name": "", "arguments": ""},
                        },
                    )
                    if tc.get("id"):
                        slot["id"] = str(tc["id"])
                    func = tc.get("function") or {}
                    if isinstance(func, dict):
                        name = func.get("name")
                        if isinstance(name, str) and name:
                            slot["function"]["name"] = name
                        args_piece = func.get("arguments")
                        if isinstance(args_piece, str) and args_piece:
                            tool_arg_buf.setdefault(idx, []).append(args_piece)

        def _record_billed() -> None:
            # Through parse_response so the usage mapping (cached vs fresh
            # input, the reported cost) has ONE owner; the empty message is
            # discarded, only its usage is kept.
            if not usage:
                return
            billed = parse_response(
                {"choices": [], "usage": usage},
                tool_names=tool_names,
                tool_schemas=tool_schemas,
            )
            record_billed_usage(
                self.budget,
                self.model,
                input_tokens=billed.input_tokens,
                output_tokens=billed.output_tokens,
                cache_read_tokens=billed.cache_read_tokens,
                cache_creation_tokens=billed.cache_creation_tokens,
                cost_usd=billed.cost_usd,
            )

        try:
            call.run(consume)
        except BaseException:
            # Billed already: a mid-stream error, the idle watchdog or an
            # operator steer ends the turn after the provider has accepted the
            # input, and the retry re-sends and is billed again.
            _record_billed()
            raise

        # A stream that ended without `[DONE]` and without any `finish_reason`
        # was cut off mid-generation (a clean EOF is not a completion signal).
        # Returning the accumulated partial text / half-built tool call as a
        # finished turn feeds the loop a bogus silent_finish or a truncated
        # tool_use; raise a retryable ProviderError so the call is re-issued.
        if not done_seen and not finish_reason:
            _record_billed()
            call.record(
                status=0,
                response="stream ended without [DONE] or finish_reason (truncated)",
            )
            raise ProviderError(
                f"OpenAI stream from {url} ended prematurely "
                "(no [DONE], no finish_reason); upstream appears cut off."
            )

        # Finalise tool_call arguments.
        final_tool_calls: list[dict[str, Any]] = []
        for idx in sorted(tool_calls):
            slot = tool_calls[idx]
            args = "".join(tool_arg_buf.get(idx, []))
            slot["function"]["arguments"] = args
            final_tool_calls.append(slot)

        # role: a non-streamed response message always carries it, and the
        # recorded body is read back as a transcript, so the synthesised one
        # must too.
        message: dict[str, Any] = {"role": "assistant", "content": "".join(text_parts)}
        if reasoning_parts:
            message["reasoning_content"] = "".join(reasoning_parts)
        if final_tool_calls:
            message["tool_calls"] = final_tool_calls

        synthesised: dict[str, Any] = {
            "choices": [
                {
                    "index": 0,
                    "message": message,
                    "finish_reason": finish_reason,
                }
            ],
            "usage": usage,
        }
        if self.budget is not None and not done_seen and not usage:
            # include_usage is always set, so the usage chunk arrives AFTER
            # finish_reason and before [DONE]: a stream that stopped in that
            # window was cut, not unmetered. Raise the retryable truncation
            # error rather than the permanent no-accounting 422 -- recorded as
            # the cut it was, like the sibling truncation above, so a retried
            # run's transcript does not show a clean 200 for it.
            call.record(
                status=0,
                response="stream cut before its usage trailer (truncated)",
            )
            raise ProviderError(
                f"OpenAI stream from {url} was cut off before its usage trailer"
                " (finish_reason seen, no [DONE], no usage); truncated response."
            )
        call.record(status=200, response=synthesised)
        if self.budget is not None:
            _require_metered_usage(usage, source="OpenAI stream")
        parsed = parse_response(synthesised, tool_names=tool_names, tool_schemas=tool_schemas)
        if self.budget is not None:
            self.budget.record(
                model=self.model,
                input_tokens=parsed.input_tokens,
                output_tokens=parsed.output_tokens,
                cache_read_tokens=parsed.cache_read_tokens,
                cache_creation_tokens=parsed.cache_creation_tokens,
                cost_usd=parsed.cost_usd,
            )
        return parsed
