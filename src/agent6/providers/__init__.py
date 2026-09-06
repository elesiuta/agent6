# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Provider package.

`AnthropicProvider` (Anthropic Messages), `OpenAIProvider` (any OpenAI
Chat Completions-compatible endpoint: OpenAI, OpenRouter, Ollama, vLLM,
llama.cpp), `ChatGPTProvider` (the ChatGPT-subscription Codex backend), and
`ClaudeCodeProvider` (the operator's installed Claude Code binary) all
satisfy the `Provider` Protocol and can serve ANY sub-agent role.
Role-to-provider routing lives in `[models.<role>]` in your config; the
providers themselves are interchangeable from the sub-agents' point of view.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol, runtime_checkable

from agent6.providers._claude_code_wire import CLAUDE_CODE_RESULT_CAP_BYTES, result_text
from agent6.providers.anthropic import AnthropicProvider
from agent6.providers.chatgpt import ChatGPTProvider
from agent6.providers.chatgpt_oauth import ChatGPTCredential
from agent6.providers.claude_code import ClaudeCodeProvider
from agent6.providers.openai import OpenAIProvider
from agent6.providers.token_command import CommandToken
from agent6.providers.types import (
    ProviderAborted,
    ProviderError,
    ProviderInterrupted,
    ProviderResponse,
    RoleTranscriptSink,
    ToolDefinition,
    TranscriptRecorder,
    TranscriptSink,
    output_cap_truncated,
)


@runtime_checkable
class Provider(Protocol):
    """Vendor-agnostic surface used by every sub-agent.

    `AnthropicProvider` and `OpenAIProvider` both satisfy this. The worker
    loop and the review seats pass real `tools` every turn; execution itself
    is Python-side via `ToolDispatcher`.

    `text_delta_callback` / `thinking_delta_callback` are opt-in SSE
    streaming hooks. When either is set, providers MAY stream visible
    text / reasoning deltas to the matching callback as they arrive. When
    both are `None` (default), providers use the non-streaming code path.
    """

    def call(
        self,
        *,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[ToolDefinition] | None = ...,
        max_tokens: int = ...,
        temperature: float | None = ...,
        reasoning_effort: str | None = ...,
        text_delta_callback: Callable[[str], None] | None = ...,
        thinking_delta_callback: Callable[[str], None] | None = ...,
        should_abort: Callable[[], bool] | None = ...,
        should_interrupt: Callable[[], bool] | None = ...,
    ) -> ProviderResponse: ...


def call_for_text(provider: Provider, *, system: str, user: str, max_tokens: int) -> str | None:
    """One guarded text-only call: the stripped reply, or None on ANY failure.

    For best-effort drafting (commit messages) where the caller holds a
    deterministic fallback: the broad except is the point, a drafting hiccup
    must never surface as a run error."""
    try:
        resp = provider.call(
            system=system,
            messages=[{"role": "user", "content": user}],
            tools=None,
            max_tokens=max_tokens,
        )
    except Exception:
        return None
    return (resp.text or "").strip() or None


__all__ = [
    "CLAUDE_CODE_RESULT_CAP_BYTES",
    "AnthropicProvider",
    "ChatGPTCredential",
    "ChatGPTProvider",
    "ClaudeCodeProvider",
    "CommandToken",
    "OpenAIProvider",
    "Provider",
    "ProviderAborted",
    "ProviderError",
    "ProviderInterrupted",
    "ProviderResponse",
    "RoleTranscriptSink",
    "ToolDefinition",
    "TranscriptRecorder",
    "TranscriptSink",
    "call_for_text",
    "output_cap_truncated",
    "result_text",
]
