# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""`agent6 prompt` subcommands: inspect what the model receives."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from agent6.config.layer import load_effective, resolved_state_dir
from agent6.verify_infer import infer_verify_command, read_agents_md
from agent6.workflows import ModelExchange, model_exchange_for


def _cmd_prompt_show(
    config_path: Path | None,
    *,
    mode: Literal["run", "plan", "ask", "agent"],
    as_json: bool = False,
) -> int:
    """Print everything the model receives on a run's first call here, for
    THIS repo and the effective (layered) config, in the given mode: the system
    prompt (its static blocks and the per-repo `<repo-priors>` block), the tool
    definitions the API's `tools` field carries (name, description, input
    schema, exactly the list this config exposes), and the first user message
    around the task. `--json` prints the same as one object."""
    cwd = Path.cwd()
    eff = load_effective(cwd, config_path)
    cfg = eff.config
    if mode in ("run", "plan") and not cfg.workflow.verify_command and cfg.workflow.verify_infer:
        # A run infers its gate before assembling the prompt, and the gate
        # decides the `<verify-command>` block, the commit rule and whether
        # `run_verify_command` is offered at all, so the audit surface infers
        # it the same way. The LLM tier is a run's own call (it spends), so
        # only the deterministic ones run here.
        inferred = infer_verify_command(cwd, read_agents_md(cwd), llm_call=None)
        if inferred is not None:
            cfg = cfg.with_verify_command(inferred.argv)
    exchange = model_exchange_for(cfg, cwd, mode, state_dir=resolved_state_dir(cwd))
    print(_as_json(exchange) if as_json else _as_text(exchange))
    return 0


def _as_json(x: ModelExchange) -> str:
    return json.dumps(
        {
            "mode": x.mode,
            "system": x.system,
            "tools": [
                {"name": t.name, "description": t.description, "input_schema": t.input_schema}
                for t in x.tools
            ],
            "first_message": x.first_message,
            "mcp_tools_pending": x.mcp_pending,
        },
        indent=2,
    )


def _as_text(x: ModelExchange) -> str:
    out = [
        f"=== system prompt ({x.mode} mode, {len(x.system):,} chars) ===",
        x.system.rstrip(),
        "",
        f"=== tools ({len(x.tools)}; the API's `tools` field: name, description, input schema) ===",
    ]
    for t in x.tools:
        out.append(f"--- {t.name}")
        out.append(t.description)
        out.append("schema: " + json.dumps(t.input_schema, indent=2))
    if x.mcp_pending:
        out.append("--- (plus the tools of the enabled MCP servers, discovered at run start)")
    out += ["", "=== first user message ===", x.first_message]
    return "\n".join(out)
