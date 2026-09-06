# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The claude_code provider's wire vocabulary, independent of a live process:
the child's argv and environment, the plan reading off a `rate_limit_event`,
the history rendered for a replay, the stdin line shapes, and the helpers over
Anthropic-shaped wire messages. `claude_code` owns the process."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from agent6.budget import PlanUsage, PlanWindow
from agent6.child_env import curated_env

# Claude Code writes a tool result above this many bytes under
# ~/.claude/projects and hands the model a 2 KB preview of it.
CLAUDE_CODE_PERSIST_BYTES = 50_000
# The loop's result cap for this provider: under the threshold, with room
# for multibyte text.
CLAUDE_CODE_RESULT_CAP_CHARS = 45_000


MCP_SERVER = "agent6"
# Claude Code names an sdk-server tool `mcp__<server>__<tool>` to the model;
# `tools/call` carries the bare name.
TOOL_PREFIX = f"mcp__{MCP_SERVER}__"
_MCP_CONFIG = json.dumps(
    {"mcpServers": {MCP_SERVER: {"type": "sdk", "name": MCP_SERVER}}}, separators=(",", ":")
)

# Every Claude Code capability beyond the model is off, and the child dials
# nothing beyond the API and its login refresh.
CLAUDE_CODE_ENV: dict[str, str] = {
    "CLAUDE_CODE_DISABLE_CLAUDE_MDS": "1",
    "CLAUDE_CODE_DISABLE_AUTO_MEMORY": "1",
    "DISABLE_AUTO_COMPACT": "1",
    "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
    "DISABLE_AUTOUPDATER": "1",
    "DISABLE_TELEMETRY": "1",
    "DISABLE_ERROR_REPORTING": "1",
}
# The one operator variable that reaches the child: it selects which login,
# never a credential.
_PASSTHROUGH = ("CLAUDE_CONFIG_DIR",)
_HARNESS_REPLAY = (
    "[harness] This session continues an earlier one on the same repository. The"
    " transcript so far follows, oldest first; every tool call in it was executed, and"
    " its result follows as this session holds it (older ones elided to a placeholder"
    " or a gist). Continue from the end of it; do not redo finished steps."
)


def claude_argv(
    binary: str, model: str, effort: str | None, system_prompt_file: Path
) -> tuple[str, ...]:
    """The child's argv: operator config and literals only, never model or
    repo text (the system prompt is a file path; prompts ride stdin)."""
    argv = [
        binary,
        "-p",
        "--verbose",
        "--model",
        model,
        "--input-format",
        "stream-json",
        "--output-format",
        "stream-json",
        "--include-partial-messages",
        "--tools",
        "",
        "--allowedTools",
        f"mcp__{MCP_SERVER}",
        "--mcp-config",
        _MCP_CONFIG,
        "--strict-mcp-config",
        "--setting-sources",
        "",
        "--disable-slash-commands",
        "--no-session-persistence",
        "--system-prompt-file",
        str(system_prompt_file),
    ]
    if effort:
        argv += ["--effort", effort]
    return tuple(argv)


def child_env() -> dict[str, str]:
    """The child's environment: the curated base (HOME included, so the binary
    finds its own login), `CLAUDE_CONFIG_DIR` when set, and the fixed toggles.
    No `ANTHROPIC_*` or `CLAUDE*` variable from the operator shell reaches it: a
    key env var would override the subscription login inside the child."""
    return curated_env(passthrough=_PASSTHROUGH, extra=CLAUDE_CODE_ENV, desktop=False)


def bare_tool_name(name: str) -> str:
    """The agent6-side tool name behind Claude Code's `mcp__agent6__` prefix."""
    return name[len(TOOL_PREFIX) :] if name.startswith(TOOL_PREFIX) else name


def _window_minutes(name: str) -> int:
    if name == "five_hour":
        return 300
    return 10_080 if name.startswith("seven_day") else 0


def plan_usage_from_rate_limit(info: Mapping[str, Any]) -> PlanUsage | None:
    """The plan windows off one `rate_limit_event.rate_limit_info`: every
    `unifiedWindows` entry as a PlanWindow (utilization is a fraction), the
    backend's own exhausted verdict, and whether extra usage is enabled. None
    when the event names no window."""
    raw = info.get("unifiedWindows")
    if not isinstance(raw, Mapping):
        return None
    windows: list[PlanWindow] = []
    for name, window in raw.items():
        if not isinstance(window, Mapping):
            continue
        used, resets_at = window.get("utilization"), window.get("resetsAt")
        if not isinstance(used, (int, float)) or not isinstance(resets_at, (int, float)):
            continue
        windows.append(
            PlanWindow(str(name), float(used) * 100.0, _window_minutes(str(name)), float(resets_at))
        )
    if not windows:
        return None
    return PlanUsage(
        windows=tuple(windows),
        has_credits=info.get("overageStatus") == "allowed" or bool(info.get("isUsingOverage")),
        limit_reached=info.get("status") == "rejected",
    )


def message_blocks(message: Mapping[str, Any]) -> list[dict[str, Any]]:
    content = message.get("content")
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    return [b for b in (content or ()) if isinstance(b, dict)]


def message_texts(message: Mapping[str, Any]) -> list[str]:
    return [str(b.get("text", "")) for b in message_blocks(message) if b.get("type") == "text"]


def tool_use_ids(message: Mapping[str, Any]) -> tuple[str, ...]:
    return tuple(
        str(b.get("id", "")) for b in message_blocks(message) if b.get("type") == "tool_use"
    )


def result_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            str(b.get("text", "")) for b in content if isinstance(b, dict) and "text" in b
        )
    return json.dumps(content, ensure_ascii=False)


def tool_results(message: Mapping[str, Any]) -> dict[str, str]:
    """`tool_use_id -> content` for the tool_result blocks, in wire order."""
    return {
        str(b.get("tool_use_id", "")): result_text(b.get("content"))
        for b in message_blocks(message)
        if b.get("type") == "tool_result"
    }


Skeleton = tuple[str, tuple[tuple[str, str], ...]]


def message_skeleton(message: Mapping[str, Any]) -> Skeleton:
    """What identifies a message the process has consumed: its role and, per
    block, the tool_use id, the tool_result id, or the text. A tool_result's
    content and thinking are left out: tier-1 elision and thinking strips
    rewrite those in place without changing what the process was sent."""
    keys = {"text": "text", "tool_use": "id", "tool_result": "tool_use_id"}
    return (
        str(message.get("role", "")),
        tuple(
            (str(kind), str(block.get(keys[kind], "")))
            for block in message_blocks(message)
            if (kind := block.get("type")) in keys
        ),
    )


def history_skeleton(messages: Sequence[Mapping[str, Any]]) -> tuple[Skeleton, ...]:
    return tuple(message_skeleton(m) for m in messages)


def render_history(messages: Sequence[Mapping[str, Any]]) -> str:
    """The mirror as one user message: the first user text verbatim, then, for
    a longer history, a harness paragraph and every later turn as labelled
    text (tool calls with their inputs, results as the mirror holds them;
    thinking dropped)."""
    if not messages:
        return ""
    first = "\n\n".join(message_texts(messages[0]))
    if len(messages) == 1:
        return first
    parts = [first, _HARNESS_REPLAY]
    names: dict[str, str] = {}
    for message in messages[1:]:
        role = str(message.get("role", "user"))
        for block in message_blocks(message):
            kind = block.get("type")
            if kind == "text":
                parts.append(f"### {role}\n{block.get('text', '')}")
            elif kind == "tool_use":
                name = str(block.get("name", ""))
                names[str(block.get("id", ""))] = name
                parts.append(
                    f"[tool_use {name}] {json.dumps(block.get('input'), ensure_ascii=False)}"
                )
            elif kind == "tool_result":
                name = names.get(str(block.get("tool_use_id", "")), "tool")
                parts.append(f"### tool_result {name}\n{result_text(block.get('content'))}")
    return "\n\n".join(parts)


def user_line(text: str) -> dict[str, Any]:
    return {
        "type": "user",
        "message": {"role": "user", "content": [{"type": "text", "text": text}]},
    }


def mcp_answer(request_id: str, rpc_id: Any, result: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "control_response",
        "response": {
            "subtype": "success",
            "request_id": request_id,
            "response": {"mcp_response": {"jsonrpc": "2.0", "id": rpc_id, "result": result}},
        },
    }
