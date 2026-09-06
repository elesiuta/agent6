# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""OpenAI Chat Completions request-message translation.

Anthropic content-blocks (agent6's internal lingua franca) -> the OpenAI
Chat Completions `messages` / `tools` wire shape. See
`providers/openai.py`'s module docstring for the translation rationale
(Shape B tool-use translation); this module is the request-building half.
"""

from __future__ import annotations

import json
from typing import Any

from agent6.providers.types import ToolDefinition


def tool_result_text(tr_content: Any) -> str:
    """A tool_result's content as the one string the OpenAI-style wires carry:
    the text blocks joined, else the content as JSON, else as text."""
    if isinstance(tr_content, list):
        parts = [
            str(b.get("text", ""))
            for b in tr_content
            if isinstance(b, dict) and b.get("type") == "text"
        ]
        return "".join(parts) if parts else json.dumps(tr_content)
    return str(tr_content)


def anthropic_to_openai_messages(  # noqa: PLR0912
    system: str, anthropic_msgs: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Translate agent6's canonical Anthropic-shape messages into the
    OpenAI Chat Completions `messages` array.

    Three block types in Anthropic content are non-trivial:

    - `text` -> string content on the message (concatenated for
      multi-text-block messages).
    - `tool_use` (assistant) -> moved into `message.tool_calls` as
      OpenAI function-call objects; the assistant's text content
      stays in `message.content`.
    - `tool_result` (user) -> emitted as a SEPARATE message with
      `role="tool"` and `tool_call_id` set; cannot stay in the
      user-message position because OpenAI puts tool replies in their
      own role.
    """
    out: list[dict[str, Any]] = [{"role": "system", "content": system}]
    # Ids of assistant tool_use blocks dropped for a blank name (see
    # `parse_response`). Their paired tool_result must be dropped too, else
    # the request carries a role=tool message with no matching tool_call and
    # strict backends reject it. Defense-in-depth for resumed runs whose
    # snapshot history predates the parse-time filter.
    dropped_tool_use_ids: set[str] = set()
    for msg in anthropic_msgs:
        role = str(msg.get("role", "user"))
        content = msg.get("content", "")
        if isinstance(content, str):
            out.append({"role": role, "content": content})
            continue
        if not isinstance(content, list):
            out.append({"role": role, "content": str(content)})
            continue
        # Walk content blocks. Behaviour depends on the block types
        # present.
        text_chunks: list[str] = []
        tool_calls: list[dict[str, Any]] = []
        tool_results: list[dict[str, Any]] = []
        for block in content:
            if not isinstance(block, dict):
                continue
            btype = block.get("type")
            if btype == "text":
                text_chunks.append(str(block.get("text", "")))
            elif btype == "tool_use" and role == "assistant":
                if not str(block.get("name") or "").strip():
                    # Blank-name tool_use: drop it and remember its id so its
                    # paired tool_result is dropped below.
                    dropped_tool_use_ids.add(str(block.get("id", "")))
                    continue
                tool_calls.append(
                    {
                        "id": str(block.get("id", "")),
                        "type": "function",
                        "function": {
                            "name": str(block.get("name", "")),
                            # OpenAI requires arguments as a JSON string,
                            # not an object.
                            "arguments": json.dumps(block.get("input") or {}),
                        },
                    }
                )
            elif btype == "tool_result":
                if str(block.get("tool_use_id", "")) in dropped_tool_use_ids:
                    # Orphaned result for a dropped blank-name tool_use. Skip it
                    # so the request stays well-formed.
                    continue
                # Tool results become separate role=tool messages.
                # `content` field may be a string or a list of text
                # blocks; OpenAI accepts either string or its own
                # content-blocks shape. Flatten to string for the
                # broadest compatibility (Ollama, Kimi, etc).
                tr_text = tool_result_text(block.get("content", ""))
                tool_results.append(
                    {
                        "role": "tool",
                        "tool_call_id": str(block.get("tool_use_id", "")),
                        "content": tr_text,
                    }
                )
        if role == "assistant":
            assistant_msg: dict[str, Any] = {"role": "assistant"}
            if text_chunks:
                assistant_msg["content"] = "".join(text_chunks)
            elif tool_calls:
                assistant_msg["content"] = None
            else:
                # A thinking-only turn (reasoning starvation) yields neither
                # text nor tool_calls. Chat Completions requires `content`
                # unless `tool_calls` is present; `null` without tool_calls
                # 400s on strict backends (non-retryable), so send "".
                assistant_msg["content"] = ""
            if tool_calls:
                assistant_msg["tool_calls"] = tool_calls
            out.append(assistant_msg)
        else:
            # user (or other) message: tool_results MUST come first
            # because OpenAI requires every `role=tool` message to
            # immediately follow the assistant turn whose `tool_calls`
            # it answers. Emitting text_chunks FIRST would (a) insert a
            # user message between the assistant's tool_calls and the tool
            # replies -- most OpenAI-compatible gateways tolerate that, but
            # it is technically malformed -- and (b) make injected
            # "[loop-guard]" / "[harness]" / "[review]" notices arrive
            # before the tool result they were commenting on, so weak
            # models lost the causal link entirely. Tool results first,
            # then any operator/harness text as a follow-up user turn.
            for tr in tool_results:
                out.append(tr)
            if text_chunks:
                out.append({"role": role, "content": "".join(text_chunks)})
    return out


def tools_to_openai(tools: list[ToolDefinition]) -> list[dict[str, Any]]:
    """Translate `ToolDefinition` tuples into OpenAI function-tool entries."""
    return [
        {
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description,
                "parameters": t.input_schema,
            },
        }
        for t in tools
    ]
