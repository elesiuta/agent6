# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Workflow package: built-in deterministic state machines."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from agent6.config import Config
from agent6.memory import decisions_path, decisions_text, memory_dir
from agent6.memory import index_text as memory_index_text
from agent6.providers import ToolDefinition
from agent6.sandbox.detect import IsolationUnavailableError, detect, resolve_isolation
from agent6.skills import ResolvedSkills
from agent6.tools.dispatch import ToolDispatcher
from agent6.types import IsolationLevel
from agent6.workflows._context import load_repo_summary
from agent6.workflows._dag_focus import initial_dag_hint
from agent6.workflows._prompt_blocks import build_system_prompt, initial_instructions
from agent6.workflows._toolset import tool_definitions
from agent6.workflows.review import CodeReviewError, code_review

__all__ = [
    "CodeReviewError",
    "ModelExchange",
    "code_review",
    "model_exchange_for",
    "system_prompt_for",
]


def system_prompt_for(
    config: Config,
    root: Path,
    mode: Literal["run", "plan", "ask", "agent"] = "run",
    *,
    state_dir: Path | None = None,
) -> str:
    """Assemble the exact system prompt agent6 would send for *root* + *config*
    in *mode*. Public entry point for `agent6 prompt show` and tooling. Builds a
    ToolDispatcher so the `<repo-priors>` block is FULLY enriched (repo map +
    AGENTS.md + recent commits + hot symbols + co-change + symbol outline) -- the
    same view the run loop sees, so prompt show matches reality.

    The memory index and installed skills are loaded on
    the loop's own rules (none of the first two in machine/agent modes, skills
    in run mode only): omitting them would print "(none recorded yet)" for
    an operator checking what future runs actually receive. *state_dir* is
    the per-repo state dir those live under, injected by the caller exactly as
    the loop's is."""
    dispatcher = (
        ToolDispatcher(root=root, config=config) if config.prompt.structural_priors else None
    )
    repo = load_repo_summary(root, dispatcher=dispatcher)
    # Machine and agent modes assemble without repo context, so neither half of
    # per-repo recall applies: one gate, not one per block.
    recall = None if mode == "agent" else state_dir
    return build_system_prompt(
        config=config,
        repo=repo,
        mode=mode,
        memory_index=memory_index_text(recall) if recall is not None else "",
        memory_dir_path=str(memory_dir(recall)) if recall is not None else "",
        decisions=decisions_text(recall) if recall is not None else "",
        decisions_path=str(decisions_path(recall)) if recall is not None else "",
        skills=_installed_skills(root, config, mode),
        isolation=_shown_isolation(config),
    )


@dataclass(frozen=True, slots=True)
class ModelExchange:
    """Everything the model receives on a run's first call, for `agent6 prompt
    show`: the system prompt, the tool definitions (name, description, input
    schema; the API's `tools` field), and the first user message's operational
    header (the task text follows it). MCP tools are discovered at run start
    and are not part of this static picture."""

    mode: str
    system: str
    tools: tuple[ToolDefinition, ...]
    first_message: str
    mcp_pending: bool  # [mcp].enabled with servers: their tools join at run start


def model_exchange_for(
    config: Config,
    root: Path,
    mode: Literal["run", "plan", "ask", "agent"] = "run",
    *,
    state_dir: Path | None = None,
) -> ModelExchange:
    """The exact exchange a run here would open with: `system_prompt_for` plus
    the tool list the loop builds (`tool_definitions` over a dispatcher on this
    config, so `run_commands = "no"`, a missing gate, `network = "host"`, and
    the installed skills withhold exactly what they withhold in a run) and the
    first message header."""
    system = system_prompt_for(config, root, mode, state_dir=state_dir)
    dispatcher = ToolDispatcher(
        root=root,
        config=config,
        mode="machine" if mode == "agent" else mode,
        state_dir=state_dir,
    )
    tools = tuple(tool_definitions(dispatcher, mode=mode))
    header = initial_instructions(
        mode, config.sandbox.run_commands, has_gate=bool(config.workflow.verify_command)
    )
    hint = initial_dag_hint("<root task id>", mode, config.prompt.decompose == "on")
    return ModelExchange(
        mode=mode,
        system=system,
        tools=tools,
        first_message=f"TASK:\n<the task text>\n\n{header}{hint}",
        mcp_pending=config.mcp.enabled and any(s.enabled for s in config.mcp.servers.values()),
    )


def _shown_isolation(config: Config) -> IsolationLevel:
    """The level a run here would resolve, for prompt display; an explicit
    setting this host cannot honor shows as "none" rather than refusing a
    read-only preview."""
    try:
        return resolve_isolation(config.sandbox.isolation, detect())
    except IsolationUnavailableError:
        return "none"


def _installed_skills(
    root: Path, config: Config, mode: Literal["run", "plan", "ask", "agent"]
) -> ResolvedSkills | None:
    """The loop's `_load_skills` rules: run mode only, and nothing installed
    renders no block."""
    if mode != "run":
        return None
    resolved = ToolDispatcher(root=root, config=config).resolved_skills()
    return resolved if (resolved.enabled or resolved.always) else None
