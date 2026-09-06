# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""finish_session result coercion: a stringified JSON object still lands as the
structured finish payload (weak models routinely stringify it; a machine
agent state's whole cycle used to fail on shape)."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

from agent6.config import Config
from agent6.workflows._conversation import AssistantTurn
from agent6.workflows.loop import (
    TurnState,
    Workflow,
)


def _wf(**kw: Any) -> Workflow:
    kw.setdefault("state_dir", Path("/tmp/state"))
    return Workflow(
        root=Path("/tmp"),
        config=Config.model_validate({}),
        provider=MagicMock(),
        dispatcher=MagicMock(),
        logger=lambda _m: None,
        **kw,
    )


def _capture(tool_input: dict[str, Any]) -> TurnState:
    wf = _wf()
    turn = TurnState(iteration=1, resp=MagicMock(), assistant=AssistantTurn((), ()))
    wf._capture_finish(turn, "finish_session", tool_input)  # pyright: ignore[reportPrivateUsage]
    return turn


def test_finish_result_object_passes_through() -> None:
    turn = _capture({"summary": "s", "result": {"found": True}})
    assert turn.finish_payload == {"found": True}


def test_finish_result_stringified_object_is_coerced() -> None:
    turn = _capture({"summary": "s", "result": '{"found": true, "file": "a.py"}'})
    assert turn.finish_payload == {"found": True, "file": "a.py"}


def test_finish_result_garbage_string_stays_none() -> None:
    turn = _capture({"summary": "s", "result": "not json"})
    assert turn.finish_payload is None


def test_finish_result_stringified_non_object_stays_none() -> None:
    turn = _capture({"summary": "s", "result": '["a", "b"]'})
    assert turn.finish_payload is None
