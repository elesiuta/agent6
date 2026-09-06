# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Run-control signal handlers: finish_session, finish_planning.

Neither acts; the workflow checks for the tool name in the response's
tool_uses and exits the loop after dispatching it."""

from __future__ import annotations

from typing import Any

from agent6.tools.results import FinishPlanningResult, FinishSessionResult
from agent6.tools.schema import FinishPlanningInput, FinishSessionInput


def finish_session(raw: dict[str, Any]) -> FinishSessionResult:
    """Signal the workflow to terminate. Handler echoes the validated summary
    (and any structured `result` payload, used by state-machine agent
    states)."""
    args = FinishSessionInput.model_validate(raw)
    return FinishSessionResult(
        summary_text=args.summary, result=args.result, stale_gate=args.stale_gate
    )


def finish_planning(raw: dict[str, Any]) -> FinishPlanningResult:
    """Signal the planning pass is done. Plan-mode counterpart of finish_session;
    the workflow writes `plan_markdown` to disk and exits after dispatching
    it. Handler echoes the validated summary."""
    args = FinishPlanningInput.model_validate(raw)
    return FinishPlanningResult(
        summary_text=args.summary,
        plan_bytes=len(args.plan_markdown.encode("utf-8")),
    )
