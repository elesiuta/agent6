# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Byte pin for the machine-agent subprocess IPC.

`MachineAgentRequest` (app/machine_agent.py) owns the ``request.json`` file
shape and `AgentExecResult` (machine/engine.py) owns ``result.json``; the argv
contract (``python -m agent6.ui.cli.machine_agent <request.json>
<result.json>``) is frozen. The files are transient per-invocation (writer and
reader are always the same install), so the pin is a same-version one: the
fixed objects below must serialize to exactly these bytes, and validating the
bytes must reproduce the objects.
"""

from __future__ import annotations

from pathlib import Path

from agent6.app.machine_agent import MachineAgentRequest
from agent6.git_ops import CommitIdentity
from agent6.machine import AgentExecResult, AgentRequest
from agent6.machine.model import FieldSpec

_REQUEST = MachineAgentRequest(
    cwd=Path("/work/repo"),
    root=Path("/work/repo"),
    overlay={"budget": {"max_tokens_fallback": 9000}},
    isolation="strict",
    transcript_dir=Path("/state/machines/m/i/transcripts"),
    events_log=Path("/state/machines/m/i/states/0002-review/logs.jsonl"),
    protect_paths=(Path("/work/repo/m.asm.toml"),),
    commit_identity=CommitIdentity(name="Machine Bot", email="bot@example.com"),
    request=AgentRequest(
        prompt="review the queue",
        timeout_s=600.0,
        model="claude-x",
        provider="anthropic",
        effort="low",
        temperature=0.2,
        max_usd=1.5,
        max_tokens_fallback=200000,
        mode="run",
        state_name="review",
        step_seq=2,
        output_schema="verdict",
        schemas={
            "verdict": {"ok": FieldSpec(type="bool"), "note": FieldSpec(type="str", optional=True)}
        },
    ),
)

_REQUEST_BYTES = (
    '{"cwd":"/work/repo","root":"/work/repo",'
    '"overlay":{"budget":{"max_tokens_fallback":9000}},"isolation":"strict",'
    '"transcript_dir":"/state/machines/m/i/transcripts",'
    '"events_log":"/state/machines/m/i/states/0002-review/logs.jsonl",'
    '"protect_paths":["/work/repo/m.asm.toml"],'
    '"commit_identity":{"name":"Machine Bot","email":"bot@example.com","trailer":null},'
    '"request":{"prompt":"review the queue","timeout_s":600.0,"model":"claude-x",'
    '"provider":"anthropic","effort":"low","temperature":0.2,"max_usd":1.5,'
    '"max_tokens_fallback":200000,"mode":"run",'
    '"state_name":"review","step_seq":2,"output_schema":"verdict",'
    '"schemas":{"verdict":{"ok":{"type":"bool","optional":false,"enum":null},'
    '"note":{"type":"str","optional":true,"enum":null}}}}}'
)

_RESULT = AgentExecResult(
    reason="finish_session",
    payload={"ok": True, "notes": "queue drained"},
    usd=0.0588752,
    input_tokens=66084,
    output_tokens=838,
)

_RESULT_BYTES = (
    '{"reason":"finish_session","payload":{"ok":true,"notes":"queue drained"},'
    '"usd":0.0588752,"usd_partial":false,"input_tokens":66084,"output_tokens":838}'
)


def test_request_serializes_to_pinned_bytes() -> None:
    assert _REQUEST.model_dump_json() == _REQUEST_BYTES


def test_request_bytes_validate_to_same_object() -> None:
    assert MachineAgentRequest.model_validate_json(_REQUEST_BYTES) == _REQUEST


def test_result_serializes_to_pinned_bytes() -> None:
    assert _RESULT.model_dump_json() == _RESULT_BYTES


def test_result_bytes_validate_to_same_object() -> None:
    assert AgentExecResult.model_validate_json(_RESULT_BYTES) == _RESULT


def test_result_payload_with_a_lone_surrogate_still_serializes() -> None:
    """A model's finish_session arguments reach the payload through json.loads,
    which accepts a lone surrogate, while model_dump_json refuses one. The
    subprocess writes result.json with exactly that call, so an unscrubbed
    payload killed it before the write; the host then read the dead subprocess
    as reason="error" and routed a SUCCESSFUL agent state to its on.failed edge.
    """
    import json

    res = AgentExecResult(reason="finish_session", payload={"note": "bad \ud800 tail"})
    text = res.model_dump_json()
    assert json.loads(text)["payload"]["note"].startswith("bad ")
    assert "\ud800" not in json.loads(text)["payload"]["note"]


def test_defaulted_request_omits_nothing() -> None:
    # Optional envelope fields serialize explicitly (null / []), never key-drop:
    # the reader side needs no .get defaults, which is the point of the model.
    minimal = MachineAgentRequest(
        cwd=Path("/w"),
        root=Path("/w"),
        overlay={},
        isolation="none",
        transcript_dir=Path("/t"),
        request=AgentRequest(prompt="p", timeout_s=1.0),
    )
    assert minimal.model_dump_json() == (
        '{"cwd":"/w","root":"/w","overlay":{},"isolation":"none","transcript_dir":"/t",'
        '"events_log":null,"protect_paths":[],"commit_identity":null,'
        '"request":{"prompt":"p","timeout_s":1.0,"model":null,"provider":null,'
        '"effort":null,"temperature":null,"max_usd":null,'
        '"max_tokens_fallback":null,"mode":"agent","state_name":"","step_seq":0,'
        '"output_schema":null,"schemas":{}}}'
    )
