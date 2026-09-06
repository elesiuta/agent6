# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""A machine agent state's output_schema as an in-leg finish contract.

A run-mode state with `output_schema`/`finish_json` could never satisfy its
contract: the leg was never told it, finish_session accepted a payload-less
call, and the engine then failed the leg over correct work. The request now
carries the schema table, the leg's task states the contract, and the loop
refuses a non-conforming finish with the problems so the retry happens in-leg.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

from agent6.app.machine_agent import (
    _finish_validator,  # pyright: ignore[reportPrivateUsage]
    _task_with_contract,  # pyright: ignore[reportPrivateUsage]
)
from agent6.config import Config
from agent6.machine import AgentRequest
from agent6.machine.model import FieldSpec
from agent6.workflows._conversation import Notice
from agent6.workflows.loop import (
    TurnState,
    Workflow,
)

_SCHEMAS = {
    "verdict": {
        "ok": FieldSpec(type="bool"),
        "detail": FieldSpec(type="finding", optional=True),
    },
    "finding": {"label": FieldSpec(type="str", enum=("pass", "fail"))},
}


def _request(schema: str | None) -> AgentRequest:
    return AgentRequest(
        prompt="judge the tree",
        timeout_s=60.0,
        output_schema=schema,
        schemas=_SCHEMAS if schema else {},
    )


def test_the_task_states_the_contract_with_nested_records() -> None:
    task = _task_with_contract(_request("verdict"))
    assert task.startswith("judge the tree")
    assert "matching schema 'verdict'" in task
    assert "verdict = {ok: bool; detail: finding (optional)}" in task
    assert "finding = {label: str one of [pass, fail]}" in task


def test_a_schemaless_request_leaves_the_task_alone() -> None:
    assert _task_with_contract(_request(None)) == "judge the tree"
    assert _finish_validator(_request(None)) is None


def _wf(validator: Any) -> Workflow:
    wf = Workflow(
        root=Path("/tmp"),
        config=Config.model_validate({}),
        provider=MagicMock(),
        dispatcher=MagicMock(),
        logger=lambda _m: None,
        mode="run",
        finish_validator=validator,
    )
    return wf


def _finishing_turn(payload: dict[str, Any] | None) -> TurnState:
    turn = TurnState(iteration=3, resp=MagicMock(), assistant=MagicMock())
    turn.finish_kind = "finish_session"
    turn.finish_signal = "done"
    turn.finish_payload = payload
    return turn


def test_a_nonconforming_finish_is_refused_with_the_problems() -> None:
    validator = _finish_validator(_request("verdict"))
    assert validator is not None
    wf = _wf(validator)
    turn = _finishing_turn(None)
    wf._gate_finish_contract(turn)  # pyright: ignore[reportPrivateUsage]
    assert turn.finish_signal is None and turn.finish_payload is None
    notices = [r.text for r in turn.tool_results if isinstance(r, Notice)]
    assert any("finish_session refused" in n and "verdict" in n for n in notices)

    # The retry with a conforming payload stands.
    turn2 = _finishing_turn({"ok": True})
    wf._gate_finish_contract(turn2)  # pyright: ignore[reportPrivateUsage]
    assert turn2.finish_signal == "done"
    assert turn2.tool_results == []


def test_a_run_without_a_contract_is_never_gated() -> None:
    wf = _wf(None)
    turn = _finishing_turn(None)
    wf._gate_finish_contract(turn)  # pyright: ignore[reportPrivateUsage]
    assert turn.finish_signal == "done"
