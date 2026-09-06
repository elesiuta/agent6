# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Parse and validate a `.asm.toml` machine file into a `MachineSpec`.

The parse boundary is pydantic v2 (`extra="forbid", frozen=True`),
exactly like `agent6.config`. Structural shape is caught by pydantic;
the cross-cutting rules from the spec (§4.5), global name uniqueness
across owner subtables, the ownership wall, reference/field type-checking,
total branches, reachability, are enforced by :func:`validate_semantics`.

Every violation is a *load-time* error, aggregated into
:class:`MachineError` so `agent6 machine check` can print them all at once.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Annotated, Any, Literal

from pydantic import (
    AfterValidator,
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

__all__ = [
    "AgentState",
    "BranchState",
    "Edge",
    "MachineError",
    "MachineSpec",
    "NotifySpec",
    "StateSpec",
    "TerminalState",
    "ToolState",
    "TypeRef",
    "WaitState",
    "edges",
    "parse_type",
    "reachable_states",
    "type_str",
]

_MODEL_CONFIG = ConfigDict(extra="forbid", frozen=True)

IDENT_RE = re.compile(r"^[a-z][a-z0-9_]*$")
_LIST_RE = re.compile(r"^list\[([a-z0-9_]+)\]$")

_SCALARS = ("str", "int", "float", "bool")
# What `parse_type` resolves before it looks at a machine's declared schemas,
# so a schema of one of these names could never be referenced.
BUILTIN_TYPE_NAMES = frozenset({*_SCALARS, "json"})
RESERVED_NAMES = frozenset({"vars", "operator", "code", "agent", "result"})

AGENT_LABELS = frozenset({"ok", "failed", "budget_exhausted", "timeout"})
TOOL_LABELS = frozenset({"ok", "nonzero", "timeout"})
WAIT_LABELS = frozenset({"tick", "signal"})


class MachineError(Exception):
    """Raised when a machine file does not load and validate cleanly.

    `problems` is the full, ordered list of diagnostics.
    """

    def __init__(self, problems: list[str]) -> None:
        self.problems = problems
        super().__init__("\n".join(problems))


# --------------------------------------------------------------------------
# Type system
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ScalarT:
    name: str  # one of _SCALARS


@dataclass(frozen=True, slots=True)
class ListT:
    elem: str  # one of _SCALARS


@dataclass(frozen=True, slots=True)
class JsonT:
    pass


@dataclass(frozen=True, slots=True)
class RecordT:
    name: str


TypeRef = ScalarT | ListT | JsonT | RecordT


class TypeParseError(Exception):
    pass


def parse_type(text: str, schema_names: frozenset[str]) -> TypeRef:
    if text in _SCALARS:
        return ScalarT(text)
    if text == "json":
        return JsonT()
    list_match = _LIST_RE.match(text)
    if list_match:
        elem = list_match.group(1)
        if elem not in _SCALARS:
            raise TypeParseError(
                f"list element type must be a scalar (str/int/float/bool), got {elem!r}"
            )
        return ListT(elem)
    if text in schema_names:
        return RecordT(text)
    raise TypeParseError(f"unknown type {text!r}")


def type_str(t: TypeRef) -> str:
    if isinstance(t, ScalarT):
        return t.name
    if isinstance(t, ListT):
        return f"list[{t.elem}]"
    if isinstance(t, JsonT):
        return "json"
    return f"record {t.name!r}"


def type_decl(t: TypeRef) -> str:
    """The value an author writes in a type = "..." declaration."""
    return t.name if isinstance(t, RecordT) else type_str(t)


# --------------------------------------------------------------------------
# Pydantic parse models (trust boundary)
# --------------------------------------------------------------------------


def _normalize_field(value: Any) -> Any:
    if isinstance(value, str):
        return {"type": value}
    return value


class FieldSpec(BaseModel):
    model_config = _MODEL_CONFIG

    type: str = Field(min_length=1)
    optional: bool = False
    enum: tuple[str, ...] | None = None


_FieldSpecT = Annotated[FieldSpec, BeforeValidator(_normalize_field)]


def _normalize_notify(value: Any) -> Any:
    if isinstance(value, str):
        return {"message": value}
    return value


class NotifySpec(BaseModel):
    """A state's optional `notify`: a templated message emitted on entry.

    Presentation only (§4.3): entering the state journals a `machine.notify`
    event and fires the operator notify hook; it adds no edge and no control
    flow. Authors write `notify = "msg"` (level defaults to "info") or
    `notify = { message = "msg", level = "warn" }`.
    """

    model_config = _MODEL_CONFIG

    message: str = Field(min_length=1)
    level: Literal["info", "warn", "error"] = "info"


_NotifySpecT = Annotated[NotifySpec, BeforeValidator(_normalize_notify)]


class OperatorVar(BaseModel):
    model_config = _MODEL_CONFIG

    type: str = Field(min_length=1)
    value: Any


class MutableVar(BaseModel):
    model_config = _MODEL_CONFIG

    type: str = Field(min_length=1)
    default: Any


class VarsSection(BaseModel):
    model_config = _MODEL_CONFIG

    operator: dict[str, OperatorVar] = Field(default_factory=dict)
    code: dict[str, MutableVar] = Field(default_factory=dict)
    agent: dict[str, MutableVar] = Field(default_factory=dict)


def _finite_usd(v: float) -> float:
    # inf passes gt=0.0 and then can never bind, silently disabling the cap.
    if not math.isfinite(v):
        raise ValueError("max_usd must be a finite cap")
    return v


_FiniteUsd = Annotated[float, AfterValidator(_finite_usd)]


class BudgetSpec(BaseModel):
    """Whole-machine spend bounds. `max_transitions` always binds.

    `max_usd` (optional) caps the machine's cumulative METERED spend
    (reported cost, else price x tokens); a state whose model has no price
    data is bounded per state by `[budget].max_tokens_fallback` in the
    effective config instead (0 there refuses unmetered models outright).
    """

    model_config = _MODEL_CONFIG

    max_usd: _FiniteUsd | None = Field(default=None, gt=0.0)
    max_transitions: int = Field(gt=0)


class Capture(BaseModel):
    model_config = _MODEL_CONFIG

    stdout_json: str | None = None
    finish_json: str | None = None
    set: dict[str, str] | None = None

    @model_validator(mode="after")
    def _exactly_one(self) -> Capture:
        present = [
            name
            for name, value in (
                ("stdout_json", self.stdout_json),
                ("finish_json", self.finish_json),
                ("set", self.set),
            )
            if value is not None
        ]
        if len(present) != 1:
            raise ValueError(
                "capture must declare exactly one of `stdout_json`, `finish_json`, or `set`"
                f" (found: {present or 'none'})"
            )
        return self


class WhenClause(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)

    if_: str | None = Field(default=None, alias="if")
    else_: bool | None = Field(default=None, alias="else")
    goto: str = Field(min_length=1)

    @model_validator(mode="after")
    def _exactly_one(self) -> WhenClause:
        if (self.if_ is None) == (self.else_ is None):
            raise ValueError("a `when` clause must declare exactly one of `if` or `else`")
        if self.else_ is not None and self.else_ is not True:
            raise ValueError("`else` must be `true` when present")
        return self


class AgentState(BaseModel):
    model_config = _MODEL_CONFIG

    kind: Literal["agent"]
    # Optional templated message emitted on entry (§4.3); presentation only.
    notify: _NotifySpecT | None = None
    # "inherit" (the default) uses the operator's effective worker model, so a
    # machine need not hardcode a model the operator may not have configured,
    # the #1 way an LLM-authored machine passed `machine check` but died at run
    # time. Set an explicit provider/model only to pin a specific one.
    model: str = Field(default="inherit", min_length=1)
    # "agent" (default): a read-only structured-output judge, classify/score/
    # decide and return a finish_session result; cannot edit the repo. Set "run" for
    # an agent state that must do real coding work (edit/verify/commit tools).
    mode: Literal["agent", "run"] = "agent"
    prompt: str = Field(min_length=1)
    output_schema: str = Field(min_length=1)
    capture: Capture
    timeout_secs: int = Field(gt=0)
    on: dict[str, str]
    # Optional per-state overrides for how this agent loop is driven. When
    # unset each falls back through the effective config (machine `[config]`
    # overlay, then repo, then global, then the built-in default). `provider` selects which
    # `[providers.*]` entry backs the call; `effort` and `temperature`
    # tune reasoning/sampling; the budget caps bound this single agent slice.
    # Secrets/connection keys are never expressed here, only the provider
    # *name*, which must already exist in the effective config.
    provider: str | None = None
    effort: Literal["off", "low", "medium", "high", "xhigh", "max"] | None = None
    temperature: float | None = None
    # Per-state overrides of the effective config's [budget] ledgers: metered
    # spend (max_usd) and the unmetered input+output token bound
    # (max_tokens_fallback, -1 unlimited / 0 refuse). Unset inherits.
    max_usd: _FiniteUsd | None = Field(default=None, gt=0.0)
    max_tokens_fallback: int | None = Field(default=None, ge=-1)


class ToolState(BaseModel):
    model_config = _MODEL_CONFIG

    kind: Literal["tool"]
    notify: _NotifySpecT | None = None
    command: tuple[str, ...] = Field(min_length=1)
    output_schema: str | None = None
    capture: Capture | None = None
    timeout_secs: int = Field(gt=0)
    on: dict[str, str]
    # Which network this tool's jailed subprocess joins, in the vocabulary the
    # sandbox and MCP servers use:
    #  - `auto` (default): one of its own, where the isolation level can give
    #    one (`strict`); where it cannot (`hardened` has no namespaces) the
    #    tool shares the host's and a warning says so. Runs anywhere.
    #  - `host`: the machine's network. Granted only if the operator permits
    #    it via `sandbox.network` (`only_explicit_states` or `host`);
    #    otherwise the run is refused naming this state. Enforceable because the
    #    machine engine is a host-netns supervisor: this tool's jail can reach
    #    the network while everything else stays off it.
    #  - `none`: one of its own, REQUIRED -- refuse on `hardened` rather than
    #    run connected, unlike `auto` which tolerates it.
    # There is no `session` here: a machine state's processes die with the
    # state (no background commands, no MCP servers, escapees swept), so a
    # shared network would never have a second member. Add it if machines ever
    # get a run-scoped jail session.
    # The tool only *declares*; whether `host` is granted is the operator's
    # call (`sandbox.network`, read from global/repo config, never a machine
    # overlay).
    network: Literal["auto", "host", "none"] = "auto"
    # Env var names this tool's jailed command receives from the operator's
    # environment: only those the operator lists in `[machine].pass_env`
    # (global/repo config, never a machine overlay); a name the operator has
    # not allowed refuses the run at startup, naming every such state and var.
    # A tool jail otherwise gets the fixed passthrough environment alone.
    pass_env: tuple[str, ...] = ()

    @field_validator("pass_env")
    @classmethod
    def _env_names(cls, names: tuple[str, ...]) -> tuple[str, ...]:
        for name in names:
            if not _ENV_NAME_RE.fullmatch(name):
                raise ValueError(f"pass_env names an invalid environment variable name {name!r}")
        return names


_ENV_NAME_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def _seconds_as_str(value: object) -> object:
    # The field is a string because it may be a template ("{{ config.poll }}"),
    # but a bare TOML integer is the natural spelling; refusing `every_secs =
    # 30` with "Input should be a valid string" tripped machine authors (model
    # and human alike). Floats stay refused: sub-second waits are not a thing
    # here, and silently truncating one would lie.
    if isinstance(value, int) and not isinstance(value, bool):
        return str(value)
    return value


class WaitState(BaseModel):
    model_config = _MODEL_CONFIG

    kind: Literal["wait"]
    notify: _NotifySpecT | None = None
    every_secs: Annotated[str, BeforeValidator(_seconds_as_str)] | None = None
    until: str | None = None
    on: dict[str, str]


class BranchState(BaseModel):
    model_config = _MODEL_CONFIG

    kind: Literal["branch"]
    notify: _NotifySpecT | None = None
    when: tuple[WhenClause, ...] = Field(min_length=1)


class TerminalState(BaseModel):
    model_config = _MODEL_CONFIG

    kind: Literal["terminal"]
    notify: _NotifySpecT | None = None
    status: Literal["ok", "failed"]
    reason: str = Field(min_length=1)


StateSpec = Annotated[
    AgentState | ToolState | WaitState | BranchState | TerminalState,
    Field(discriminator="kind"),
]


# Operator-only security policy an untrusted machine `[config]` overlay may never
# carry. The loader enforces it and `config set --machine-file` refuses the same
# keys, both off this one set. `mcp`, `notify.on_complete`, `machine.notify`, and
# `git.run_repo_hooks` all spawn an operator/repo argv on the host outside the jail.
PROTECTED_OVERLAY_TABLES: tuple[str, ...] = ("providers", "sandbox", "presets", "mcp")
PROTECTED_OVERLAY_LEAVES: dict[str, str] = {
    "machine.notify": "the notify hook runs an operator argv on the host outside the jail",
    "notify.on_complete": "the completion hook runs an operator argv on the host outside the jail",
    "git.run_repo_hooks": (
        "honoring the repo's .git/hooks runs repo-controlled code on the host outside the jail"
    ),
    "git.run_repo_filters": (
        "honoring the repo's content drivers runs repo-controlled code on the host outside the jail"
    ),
    "machine.pass_env": (
        "it decides which of the operator's environment variables reach a tool jail"
    ),
    "prompt.system_prompt_file": (
        "the system prompt is read from the host, outside the jail, and sent to the provider"
    ),
}


class MachineSpec(BaseModel):
    """A validated `.asm.toml` machine definition: budget, typed `schemas`, the
    named `states` graph, and an optional agent6 `[config]` overlay whose
    operator-only security policy is refused (see `PROTECTED_OVERLAY_*`) so an
    untrusted machine file cannot weaken the sandbox."""

    model_config = _MODEL_CONFIG

    machine: str = Field(pattern=r"^[a-z][a-z0-9_-]*$")
    version: Literal[1]
    initial: str = Field(min_length=1)
    budget: BudgetSpec
    vars: VarsSection = Field(default_factory=VarsSection)
    schemas: dict[str, dict[str, _FieldSpecT]] = Field(default_factory=dict)
    states: dict[str, StateSpec]
    # Highest-precedence config layer for the machine run: an ordinary agent6
    # config fragment (most `agent6 config show` knobs), minus the operator-only
    # security policy PROTECTED_OVERLAY_* refuses. Unset keys read through.
    config: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _forbid_protected_overlay_tables(self) -> MachineSpec:
        # An overlay is the highest config layer at run time but may be untrusted
        # (LLM-drafted, shared), so it must not carry operator-only security
        # policy: the jail, connections/secrets, MCP servers, host-argv hooks, or
        # the strategy presets that define them (a `[presets.<selected>]` splices
        # straight into the effective config). Refused off PROTECTED_OVERLAY_*.
        for table in PROTECTED_OVERLAY_TABLES:
            if table in self.config:
                raise ValueError(
                    f"machine `[config]` overlay must not declare `[{table}.*]`:"
                    " connections/secrets, sandbox policy, strategy presets, and MCP"
                    " servers are operator decisions set in the global/repo config,"
                    " never in a .asm.toml file"
                )
        # Individual operator-only leaves (the rest of their table stays a
        # legitimate overlay knob, e.g. [config.git.commit] identity).
        for dotted, why in PROTECTED_OVERLAY_LEAVES.items():
            head, _, leaf = dotted.partition(".")
            sub = self.config.get(head)
            if isinstance(sub, dict) and leaf in sub:
                raise ValueError(
                    f"machine `[config]` overlay must not set `{dotted}`: {why};"
                    " it is an operator decision in the global/repo config, never"
                    " in a .asm.toml file"
                )
        return self


# --------------------------------------------------------------------------
# Graph edges
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Edge:
    src: str
    dst: str
    label: str


def edges(spec: MachineSpec) -> tuple[Edge, ...]:
    """Every directed, labelled edge in the machine graph."""
    out: list[Edge] = []
    for name, state in spec.states.items():
        if isinstance(state, BranchState):
            for clause in state.when:
                label = clause.if_ if clause.if_ is not None else "else"
                out.append(Edge(src=name, dst=clause.goto, label=label))
        elif isinstance(state, (AgentState, ToolState, WaitState)):
            for label, target in state.on.items():
                out.append(Edge(src=name, dst=target, label=label))
    return tuple(out)


def reachable_states(spec: MachineSpec) -> frozenset[str]:
    """States reachable from `initial` following declared edges."""
    adjacency: dict[str, list[str]] = {name: [] for name in spec.states}
    for edge in edges(spec):
        if edge.dst in adjacency:
            adjacency[edge.src].append(edge.dst)
    seen: set[str] = set()
    if spec.initial in spec.states:
        stack = [spec.initial]
        while stack:
            current = stack.pop()
            if current in seen:
                continue
            seen.add(current)
            stack.extend(adjacency[current])
    return frozenset(seen)
