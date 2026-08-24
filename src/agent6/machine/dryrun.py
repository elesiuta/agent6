# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Author-time dry-run for `agent6 machine test` (§5.1, §4.5).

Pure validation of a loaded :class:`MachineSpec` with **no** real-world I/O,
no jail, no network, no provider calls, no clock. Two passes:

- **Per-state**: synthesize the success fact each non-branch state would emit
  (a tool's `output_schema`-shaped JSON / an agent's `finish_session` payload),
  push it through the real :func:`agent6.machine.engine.reduce`, and confirm the
  capture binds cleanly and the produced label routes to a declared state.
- **Per-branch**: evaluate every `when` clause against an operator-supplied
  blackboard fixture (overlaid on the declared defaults) and report the
  winning `goto`.

Everything reuses the engine/predicate/model code paths the live runner uses,
so a green `machine test` means the plumbing, schemas, captures, and routing
are sound, only the actual tool output / agent judgement / wall-clock differ
at run time.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from agent6.machine._semantics import validate_record_payload
from agent6.machine.engine import EngineError, initial_blackboard, reduce
from agent6.machine.journal import AgentFact, ToolFact
from agent6.machine.model import (
    AgentState,
    BranchState,
    MachineSpec,
    TerminalState,
    ToolState,
    WaitState,
)
from agent6.machine.predicate import PredicateError, evaluate, parse_predicate
from agent6.machine.template import TemplateError

__all__ = [
    "BranchCheck",
    "DryRunReport",
    "StateCheck",
    "dry_run",
    "synthesize_record",
]

_LIST_RE = re.compile(r"^list\[([a-z0-9_]+)\]$")
_SCALAR_EXAMPLES: dict[str, Any] = {"str": "", "int": 0, "float": 0.0, "bool": False}


@dataclass(frozen=True, slots=True)
class StateCheck:
    name: str
    kind: str
    ok: bool
    label: str | None
    goto: str | None
    detail: str


@dataclass(frozen=True, slots=True)
class BranchCheck:
    name: str
    clause_index: int | None
    predicate: str | None
    goto: str | None
    ok: bool
    detail: str


@dataclass(frozen=True, slots=True)
class DryRunReport:
    states: tuple[StateCheck, ...]
    branches: tuple[BranchCheck, ...]

    @property
    def ok(self) -> bool:
        return all(s.ok for s in self.states) and all(b.ok for b in self.branches)


def synthesize_record(spec: MachineSpec, schema_name: str, _seen: tuple[str, ...] = ()) -> Any:
    """A minimal, schema-valid example object for *schema_name*.

    Produces exactly the REQUIRED fields (so it passes the strict payload
    check): scalars get a zero value, lists an empty list, enums their first
    member, nested records recurse. Optional fields are OMITTED -- the weakest
    state the capture gate permits -- so a dry-run reading one unguarded fails
    offline exactly as the live run would, instead of routing on invented
    data. Schema cycles (already rejected by `validate_semantics`) are guarded
    with `_seen`.
    """
    fields = spec.schemas.get(schema_name)
    if fields is None:  # pragma: no cover - validate_semantics guarantees it exists
        return {}
    out: dict[str, Any] = {}
    for fname, field in fields.items():
        if field.optional:
            continue
        out[fname] = _synthesize_field(spec, field, (*_seen, schema_name))
    return out


def _synthesize_field(spec: MachineSpec, field: Any, seen: tuple[str, ...]) -> Any:
    if field.enum:
        return field.enum[0]
    t: str = field.type
    if t in _SCALAR_EXAMPLES:
        return _SCALAR_EXAMPLES[t]
    if t == "json":
        return {}
    if _LIST_RE.match(t):
        return []
    if t in spec.schemas:  # record reference (guard cycles already rejected at load)
        return {} if t in seen else synthesize_record(spec, t, seen)
    return None  # pragma: no cover - unknown types already rejected at load


def _capture_summary(capture: Any) -> str:
    """The variables a state's capture binds (for the report; not a value diff)."""
    if capture is None:
        return "no capture"
    if capture.stdout_json is not None:
        targets = [capture.stdout_json]
    elif capture.finish_json is not None:
        targets = [capture.finish_json]
    elif capture.set is not None:
        targets = sorted(capture.set)
    else:  # pragma: no cover - Capture validator guarantees one mode is set
        targets = []
    return f"captures {', '.join(targets)}" if targets else "no capture"


def _check_tool(
    spec: MachineSpec, name: str, state: ToolState, blackboard: dict[str, Any]
) -> StateCheck:
    if state.output_schema is not None:
        stdout = json.dumps(synthesize_record(spec, state.output_schema))
    else:
        stdout = ""
    fact = ToolFact(exit_code=0, stdout=stdout, timed_out=False)
    reduce(spec, state, fact, blackboard)  # exercises capture rendering; raises on a bad template
    goto = state.on["ok"]
    if goto not in spec.states:
        return StateCheck(name, "tool", False, "ok", goto, f"on.ok -> {goto!r} is not a state")
    return StateCheck(name, "tool", True, "ok", goto, _capture_summary(state.capture))


def _check_agent(
    spec: MachineSpec, name: str, state: AgentState, blackboard: dict[str, Any]
) -> StateCheck:
    payload = synthesize_record(spec, state.output_schema)
    problems = validate_record_payload(
        spec.schemas, state.output_schema, payload, where="finish_session payload"
    )
    if problems:  # pragma: no cover - synthesis is schema-valid by construction
        return StateCheck(name, "agent", False, "ok", None, "; ".join(problems))
    fact = AgentFact(outcome="ok", reason="finish_session", payload=payload)
    reduce(spec, state, fact, blackboard)  # exercises capture rendering; raises on a bad template
    goto = state.on["ok"]
    if goto not in spec.states:
        return StateCheck(name, "agent", False, "ok", goto, f"on.ok -> {goto!r} is not a state")
    return StateCheck(name, "agent", True, "ok", goto, _capture_summary(state.capture))


def _check_state(
    spec: MachineSpec, name: str, state: Any, blackboard: dict[str, Any]
) -> StateCheck:
    try:
        if isinstance(state, ToolState):
            return _check_tool(spec, name, state, blackboard)
        if isinstance(state, AgentState):
            return _check_agent(spec, name, state, blackboard)
        if isinstance(state, WaitState):
            # A wait with no timer parks until a signal poke (no `tick` edge);
            # a timed wait simulates the tick path.
            forever = state.every_secs is None and state.until is None
            label = "signal" if forever else "tick"
            goto = state.on[label]
            ok = goto in spec.states
            detail = f"{label} path" if ok else f"on.{label} -> {goto!r} is not a state"
            return StateCheck(name, "wait", ok, label, goto, detail)
        if isinstance(state, TerminalState):
            return StateCheck(name, "terminal", True, None, None, f"{state.status}: {state.reason}")
    except (EngineError, TemplateError, PredicateError) as exc:
        return StateCheck(name, getattr(state, "kind", "?"), False, None, None, str(exc))
    return StateCheck(name, getattr(state, "kind", "?"), True, None, None, "")  # pragma: no cover


def _check_branch(
    spec: MachineSpec, name: str, state: BranchState, blackboard: dict[str, Any]
) -> BranchCheck:
    try:
        for index, clause in enumerate(state.when):
            if clause.else_ is not None:
                fired, label, goto = True, "else", clause.goto
            else:
                assert clause.if_ is not None
                fired, label, goto = (
                    evaluate(parse_predicate(clause.if_), blackboard),
                    clause.if_,
                    clause.goto,
                )
            if fired:
                ok = goto in spec.states
                detail = "" if ok else f"goto {goto!r} is not a state"
                return BranchCheck(name, index, label, goto, ok, detail)
    except (PredicateError, TemplateError) as exc:
        return BranchCheck(name, None, None, None, False, f"predicate error: {exc}")
    # validate_semantics guarantees a final else, so this is unreachable.
    return BranchCheck(name, None, None, None, False, "no clause matched")  # pragma: no cover


def dry_run(spec: MachineSpec, blackboard_fixture: dict[str, Any] | None = None) -> DryRunReport:
    """Run the per-state and per-branch dry-run passes over *spec*.

    *blackboard_fixture* (e.g. from `--blackboard`) is overlaid on the
    declared variable defaults before each pass, letting an operator steer
    branch predicates and capture templates without any real execution.
    """
    base = initial_blackboard(spec)
    # Record vars default to {} (4.2), but a branch that reads
    # `verdict.field` cannot evaluate against an empty record, so the realistic
    # agent-verdict -> branch machine would always fail here without a fixture.
    # Synthesize the schema-zero record of its REQUIRED fields instead (the
    # weakest state the capture gate permits; an optional field stays absent,
    # `has()` is its guard); the fixture below still overrides it.
    for name, var in (*spec.vars.code.items(), *spec.vars.agent.items()):
        if var.type in spec.schemas and base.get(name) == {}:
            base[name] = synthesize_record(spec, var.type)
    if blackboard_fixture:
        base.update(blackboard_fixture)
    states: list[StateCheck] = []
    branches: list[BranchCheck] = []
    for name, state in spec.states.items():
        if isinstance(state, BranchState):
            branches.append(_check_branch(spec, name, state, dict(base)))
        else:
            states.append(_check_state(spec, name, state, dict(base)))
    return DryRunReport(states=tuple(states), branches=tuple(branches))
