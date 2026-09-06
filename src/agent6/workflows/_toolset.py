# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The tool surface each loop mode exposes to the model.

Builds the ToolDefinition list from the dispatcher's availability plus the
mode's extras (run / plan / ask / machine / agent), and the read-only review
surface shared by the in-loop panel and `agent6 review`.
"""

from __future__ import annotations

from typing import Any, Literal

from agent6.providers import ToolDefinition
from agent6.tools.dispatch import ToolDispatcher, ToolError
from agent6.tools.results import ToolResult
from agent6.tools.schema import RunMetricInput, UseSkillInput, mode_tools, wire_schema
from agent6.types import session_kind
from agent6.workflows._review import ReviewDispatch

# The ONLY tools an explore-tier reviewer may use: read-only navigation, no
# edits/commits/run_command/dag/finish. Enforced both by what we expose AND by
# the dispatch wrapper (defense in depth).
READONLY_REVIEW_TOOLS = frozenset(
    {
        "read_file",
        "list_dir",
        "outline",
        "find_definition",
        "find_references",
        "agent6_docs",
    }
)


def tool_definitions(
    dispatcher: ToolDispatcher,
    *,
    mode: Literal["run", "plan", "ask", "machine", "agent"] = "run",
) -> list[ToolDefinition]:
    """Build the tool list exposed to the loop: the mode's surface
    (`schema.mode_tools`, which the dispatcher also enforces as its
    backstop) filtered by what the dispatcher actually allows (e.g.
    run_command may be disabled)."""
    available = set(dispatcher.available_tool_names())
    surface = mode_tools(mode)
    out: list[ToolDefinition] = []
    for cls in (*surface.base, *surface.extras):
        if cls.TOOL_NAME not in available and cls not in surface.extras:
            # Extras (finish_session/finish_planning, run_metric_command,
            # the task tools, ask_user, use_skill) are always exposed even
            # though they're not in ALL_TOOLS.
            continue
        if dispatcher.tool_is_withheld(cls.TOOL_NAME):
            continue
        if cls.TOOL_NAME == RunMetricInput.TOOL_NAME and not dispatcher.metric_configured():
            # No [workflow.metric]: the tool can only answer "no metric
            # configured", which the model cannot fix. Hidden like use_skill
            # below and run_verify_command in the dispatcher.
            continue
        if cls.TOOL_NAME == UseSkillInput.TOOL_NAME and not dispatcher.skills_available():
            # No installed/enabled skills (or [skills].enabled off): hide the
            # tool rather than offer one that can only error, matching the
            # LSP-gating pattern.
            continue
        out.append(
            ToolDefinition(
                name=cls.TOOL_NAME,
                description=cls.TOOL_DESCRIPTION,
                input_schema=wire_schema(cls),
            )
        )
    # Any MCP tools the dispatcher's manager discovered get
    # appended verbatim -- in run mode ONLY. MCP tools are arbitrary external
    # capabilities agent6 cannot classify as read-only, so the read-only modes
    # (plan/ask/machine/agent) must not offer them at all; the dispatcher
    # refuses mcp__* in those modes as the backstop. Names already carry the
    # `mcp__<server>__` prefix so they can never collide with built-ins.
    if session_kind(mode).edits:
        for desc in dispatcher.mcp_descriptors():
            schema = dict(desc.input_schema)
            schema.setdefault("type", "object")
            out.append(
                ToolDefinition(
                    name=desc.qualified_name,
                    description=desc.description or f"MCP tool {desc.tool_name!r}",
                    input_schema=schema,
                )
            )
    return out


def build_readonly_review_tools(
    dispatcher: ToolDispatcher,
) -> tuple[list[ToolDefinition], ReviewDispatch]:
    """Read-only tool surface for explore-tier review seats: the navigation tools
    *dispatcher* exposes filtered to `READONLY_REVIEW_TOOLS`, plus a dispatch
    wrapper that REFUSES anything outside the allowlist (so a reviewer can never
    edit, commit, run a command, or mutate the task graph). Shared by the in-loop
    panel and the post-hoc `agent6 review` path."""
    tools = [t for t in tool_definitions(dispatcher, mode="run") if t.name in READONLY_REVIEW_TOOLS]

    def dispatch(name: str, tool_input: dict[str, Any]) -> ToolResult:
        if name not in READONLY_REVIEW_TOOLS:
            raise ToolError(f"review reviewer may not call {name!r} (read-only)")
        return dispatcher.dispatch(name, tool_input)

    return tools, dispatch
