# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The machine engine: a pure reducer loop driven by journaled facts (§5.1).

The engine executes one state at a time. Every impure step is a
:class:`World` method (tools, agent runs, waits, the clock, poke files,
notify); an observed result is passed through `reduce` for validation, then
journaled before the returned blackboard replaces the current one.
`reduce` and `next_state` are pure, so:

* **Crash recovery**, on restart, recorded facts are replayed through the
  same pure reducer to rebuild the blackboard and position, then execution
  continues live from the last completed step.
* **Replay**, the identical reconstruction runs with `live=False` and no
  `World` at all, reproducing the recorded path offline for backtesting.

The `agent` kind runs a normal agent6 loop through an injected
:class:`World.run_agent` and captures the schema-validated `finish_session`
payload into the blackboard; `tool`/`branch`/`wait`/`terminal` are
fully deterministic.
"""

from __future__ import annotations

import contextlib
import json
import shutil
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict, field_validator
from pydantic import Field as PydanticField

from agent6.machine._semantics import validate_record_payload
from agent6.machine.journal import (
    AgentFact,
    AttemptSpend,
    BranchFact,
    Fact,
    JournalError,
    MachineBegin,
    MachineEnd,
    MachineJournal,
    MachineNotify,
    PendingWait,
    Snapshot,
    StepEvent,
    ToolFact,
    WaitFact,
    clear_stop_request,
    scrub_lone_surrogates,
    stop_requested,
)
from agent6.machine.model import (
    AgentState,
    BranchState,
    FieldSpec,
    MachineSpec,
    StateSpec,
    TerminalState,
    ToolState,
    WaitState,
)
from agent6.machine.predicate import PredicateError, evaluate, parse_predicate
from agent6.machine.template import (
    TemplateError,
    parse_template,
    render_command,
    render_string,
    render_value,
)
from agent6.portable import atomic_write
from agent6.sandbox.jail import JailUnavailableError, run_in_jail
from agent6.sessions.layout import LOGS_NAME
from agent6.types import CommandResult, JailPolicy, NetworkMode

__all__ = [
    "AgentExecResult",
    "AgentRequest",
    "EngineError",
    "LiveWorld",
    "MachineResult",
    "ToolExecResult",
    "WaitWake",
    "World",
    "drive",
]


class EngineError(Exception):
    """Raised when a machine cannot be executed (bad data, unsupported kind)."""


class StateRuntimeError(EngineError):
    """Raised when a state reaches invalid data despite load-time checks."""


# Runtime failures from a state's predicate/template/capture. A check-passing
# machine should not hit these (the load-time validators catch type errors), but
# defense in depth: they are converted to a clean failed `MachineResult`, never an
# uncaught traceback, and never journaled as a poison StepEvent that would
# re-crash every later reduce (status/replay/resume).
_STATE_RUNTIME_ERRORS = (StateRuntimeError, PredicateError, TemplateError)


def _now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="microseconds")


# --------------------------------------------------------------------------
# The world boundary, the only impure surface.
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ToolExecResult:
    """The observable result of running one `tool` command."""

    exit_code: int
    stdout: str
    timed_out: bool
    stderr: str = ""


class AgentRequest(BaseModel):
    """What the engine asks the world to run for one `agent` state.

    Crosses the machine-agent subprocess boundary verbatim: it is the
    `request` block of `request.json` (envelope: `MachineAgentRequest` in
    `app/machine_agent.py`), so it is pydantic per the IPC rule and owns that
    wire shape. Bytes pinned by `tests/unit/test_machine_agent_ipc.py`.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    prompt: str
    timeout_s: float
    # Optional per-state overrides mirrored from `AgentState`. `None`
    # means "fall back to the effective config" in the world implementation.
    # `model` is optional too: a `machine run` agent state always sets it
    # (AgentState.model is min_length=1), but `machine create`'s authoring
    # agent has no state and must INHERIT the operator's worker model: an
    # empty-string override there would overwrite the worker model with "" and
    # fail min_length validation.
    model: str | None = None
    provider: str | None = None
    effort: str | None = None
    temperature: float | None = None
    max_usd: float | None = None
    max_tokens_fallback: int | None = None
    # Workflow mode for the nested loop: "agent" (default) for a machine
    # `agent` state, a read-only structured-output judge; "run" for an agent
    # state that opted into coding work; "machine" for the `machine create`
    # authoring agent. machine_agent maps anything else to "run".
    mode: str = "agent"
    # Which state, at which transition, this agent invocation is. The live World
    # uses them to give each agent-state execution its own watchable logs.jsonl
    # (`<instance>/states/<seq>-<name>/`), so a running machine is followable
    # like a run. Empty/0 for the `machine create` authoring agent (no state).
    state_name: str = ""
    step_seq: int = 0
    # The state's finish contract: the leg refuses a non-conforming
    # finish_session `result` in-run (the model retries with the problems),
    # and the engine's own validation stays the authority on the recorded
    # fact. `schemas` is the spec's schema table verbatim, so nested records
    # resolve leg-side with the same validator.
    output_schema: str | None = None
    schemas: dict[str, dict[str, FieldSpec]] = PydanticField(default_factory=dict)


class AgentExecResult(BaseModel):
    """The observable result of one agent loop.

    `reason` is the agent loop's stop reason (e.g. `"finish_session"`,
    `"budget_exhausted"`, `"timeout"`, `"max_iterations"`); `payload` is
    the structured object the agent passed to `finish_session` (`None` if it
    never called it or passed no structured result). `usd` and the token
    counts report the slice this agent loop spent, summed into machine-level
    spend for `machine status` (§6).

    Crosses the machine-agent subprocess boundary verbatim as `result.json`
    (written by `run_one`, validated back by the host runner in
    `app/machine_agent.py`), so it is pydantic per the IPC rule and owns that
    file shape. Bytes pinned by `tests/unit/test_machine_agent_ipc.py`.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    reason: str
    payload: dict[str, Any] | None
    usd: float = 0.0
    # True when `usd` is a known under-estimate (an unpriced model in the
    # loop's per-model breakdown); threaded into the AgentFact so the booked
    # ledger keeps the '~' truth. Defaults False for old result.json files.
    usd_partial: bool = False
    input_tokens: int = 0
    output_tokens: int = 0

    @field_validator("payload")
    @classmethod
    def _scrub_payload(cls, value: dict[str, Any] | None) -> dict[str, Any] | None:
        """The payload is the model's `finish_session` arguments, parsed by
        `json.loads`, which accepts a lone surrogate that `model_dump_json`
        then refuses. The subprocess writes `result.json` with exactly that
        call, so an unscrubbed payload killed it before the write and the host
        read the dead subprocess as an error -- routing a state whose agent
        finished successfully to its `on.failed` edge. Scrub here, on the type
        that owns the file shape, so every construction path is safe."""
        return value if value is None else scrub_lone_surrogates(value)


@dataclass(frozen=True, slots=True)
class WaitWake:
    """How a `wait` woke: a clock `tick`, an operator `signal` poke, or a
    `stop` request interrupting the sleep (the wait stays armed; nothing is
    journaled for it).

    `payload` is the JSON a poke carried (`None` for a bare poke or a tick);
    the engine journals it in the :class:`WaitFact` so a replay re-reads it.
    """

    woke_by: Literal["tick", "signal", "stop"]
    payload: Any = None


class World(Protocol):
    """Everything the engine is allowed to observe from the outside."""

    def run_tool(
        self, argv: tuple[str, ...], timeout_s: float, *, network: NetworkMode = "none"
    ) -> ToolExecResult: ...

    def run_agent(self, request: AgentRequest) -> AgentExecResult: ...

    def now(self) -> float: ...

    # `wake_epoch` is None for a wait with no timer: block until a signal poke.
    def sleep_until(self, wake_epoch: float | None) -> WaitWake: ...

    # Materialize a poke payload where the next tool can read it (a no-op when
    # the world has no persistent data dir; see LiveWorld.materialize_poke).
    def materialize_poke(self, payload: Any) -> None: ...

    # Fire the out-of-band operator notify hook on a state's `notify` message
    # (`kind="notify"`, `level` in info/warn/error) or a terminal
    # `machine.end` (`kind="end"`, `message` the reason, `level` the
    # status). Presentation only; a no-op when no hook is configured.
    def notify(self, kind: str, state: str, message: str, level: str) -> None: ...


# The per-call tool-jail policy, injected by the CLI (see LiveWorld.tool_policy):
# (argv, timeout_s, network) -> the JailPolicy the shared builder produced.
ToolPolicyFactory = Callable[[tuple[str, ...], float, NetworkMode], JailPolicy]


def _state_log_seq(p: Path) -> int:
    """The numeric transition seq from a `<seq>-<state>` per-state log dir name
    (so the sort is by seq, not lexical -- correct past 9999)."""
    prefix = p.name.split("-", 1)[0]
    return int(prefix) if prefix.isdigit() else -1


def _prune_state_logs(root: Path, *, keep: int) -> None:
    """Keep only the most recent *keep-1* per-state log dirs under *root* (leaving
    room for the one about to be written), so a long-running machine's reasoning
    logs stay bounded. `keep=0` prunes nothing, like `snapshot_keep`. The journal
    (the durable audit) keeps the full transition history regardless. Best
    effort: never let cleanup break a run."""
    if keep == 0:
        return
    try:
        dirs = sorted((p for p in root.iterdir() if p.is_dir()), key=_state_log_seq)
    except OSError:
        return
    for stale in dirs[: max(0, len(dirs) - keep + 1)]:
        shutil.rmtree(stale, ignore_errors=True)


@dataclass(frozen=True, slots=True)
class LiveWorld:
    """Production :class:`World`: tools go through the jail, waits really sleep.

    A `wait` blocks in-process until its absolute instant (§4.3) or until an
    operator drops a `signal` file in the machine directory, whichever comes
    first. Because the wake instant is journaled, the persisted-wake driver
    (`drive(exit_on_wait=True)`) replays the identical file with no format
    change.

    `agent` states are delegated to an injected `agent_runner` so the engine
    module need not import the provider / workflow stack; the CLI wires the real
    runner (loading the effective config, building a provider and the loop). When no
    runner is configured, reaching an `agent` state fails loudly.
    """

    cwd: Path
    journal: MachineJournal
    # Per-call `events_log`: each agent-state execution gets its own logs.jsonl
    # (None for the rare runner that wants no log). The World derives the path.
    agent_runner: Callable[[AgentRequest, Path | None], AgentExecResult] | None = None
    poll_interval_s: float = 0.5
    # Builds each tool jail's policy. The CLI wires the ONE shared builder
    # (tools.policy.jail_policy) with the machine deltas baked in -- bundle
    # protect paths, the data-dir RW grant + $AGENT6_MACHINE_DATA_DIR -- so a
    # machine tool is confined exactly like a run command: same operator
    # grants, same protect_git, same hidden paths, same env. Injected so this
    # module needs no config import, like agent_runner.
    tool_policy: ToolPolicyFactory | None = None
    # When set, each agent-state execution writes a watchable event stream to
    # `<state_log_root>/<seq>-<state>/logs.jsonl` (the CLI points it at
    # `<instance>/states`), pruned to the most recent `state_log_keep` so a
    # long-running machine's logs stay bounded. None disables per-state logs.
    state_log_root: Path | None = None
    # From [machine].state_log_keep (the CLI wires it); 0 keeps all.
    state_log_keep: int = 50
    # Out-of-band operator notify hook, fired on a `notify` message and on the
    # terminal `machine.end`. The CLI wires it to the operator's configured argv
    # (`[machine.notify].on_event`), run on the host outside the jail. None means
    # no hook; the in-page/TUI/CLI front-ends still render the journaled events.
    notify_hook: Callable[[str, str, str, str], None] | None = None
    # The machine's persistent, writable scratch dir, surfaced to scripts as
    # $AGENT6_MACHINE_DATA_DIR (the tool_policy factory bakes in the RW grant
    # and the env var). It lives out of the workspace (under the per-repo
    # state dir) and persists across iterations, so it is where a `tool` keeps
    # DURABLE state (a built venv, caches). cwd is writable too, but it is the
    # repo, not durable machine state. Set by the CLI to <instance>/data.
    data_dir: Path | None = None
    # Executes one tool policy. The CLI overrides it for a machine with
    # `mode = "run"` states so tools run in the machine's own tree (a fresh
    # clone at the chain tip); None is the plain jail. Injected so this
    # module needs no git import, like agent_runner.
    jail_runner: Callable[[JailPolicy], CommandResult] | None = None

    def run_tool(
        self, argv: tuple[str, ...], timeout_s: float, *, network: NetworkMode = "none"
    ) -> ToolExecResult:
        # `network` is authoritative here: opt-in was gated by the CLI at
        # startup (sandbox.network), and the factory's builder clamps it to
        # what the isolation level truthfully provides.
        if self.tool_policy is None:
            raise EngineError("LiveWorld has no tool_policy factory wired")
        try:
            policy = self.tool_policy(tuple(argv), float(timeout_s), network)
            result = (self.jail_runner or run_in_jail)(policy)
        except JailUnavailableError as exc:
            raise EngineError(f"jail unavailable: {exc}") from exc
        # run_in_jail's contract is to RETURN a rc=124 result on timeout, never
        # raise TimeoutExpired (the Rust launcher and jail.py both collapse a
        # timeout to rc 124). Deriving timed_out from that is what makes a tool
        # state's on.timeout transition reachable; an `except TimeoutExpired`
        # here would be dead code, every real timeout labelled not-timed-out.
        return ToolExecResult(
            exit_code=result.returncode,
            stdout=result.stdout,
            timed_out=result.returncode == 124,
            stderr=result.stderr,
        )

    def run_agent(self, request: AgentRequest) -> AgentExecResult:
        if self.agent_runner is None:
            raise EngineError("machine reached an `agent` state but no agent runner is configured")
        return self.agent_runner(request, self._state_log(request))

    def _state_log(self, request: AgentRequest) -> Path | None:
        """The per-execution event-log path for this agent state, or None when
        per-state logs are disabled. Prunes to the most recent `state_log_keep`
        first so a long-running machine never accumulates them without bound."""
        if self.state_log_root is None or not request.state_name:
            return None
        _prune_state_logs(self.state_log_root, keep=self.state_log_keep)
        return self.state_log_root / f"{request.step_seq:04d}-{request.state_name}" / LOGS_NAME

    def now(self) -> float:
        return time.time()

    def sleep_until(self, wake_epoch: float | None) -> WaitWake:
        """Block until the wake instant or an operator signal poke, whichever
        first. `wake_epoch=None` is a wait with no timer: park until a poke."""
        while True:
            signaled, payload = self.journal.take_signal()
            if signaled:
                return WaitWake("signal", payload)
            if stop_requested(self.journal.root):
                # Interrupt the sleep; the wait's PendingWait stays armed so a
                # later `machine run` resumes the same instant.
                return WaitWake("stop")
            if wake_epoch is None:
                time.sleep(self.poll_interval_s)
                continue
            remaining = wake_epoch - time.time()
            if remaining <= 0:
                return WaitWake("tick")
            time.sleep(min(remaining, self.poll_interval_s))

    def materialize_poke(self, payload: Any) -> None:
        """Write a signal poke's payload to `$AGENT6_MACHINE_DATA_DIR/poke.json`
        so the next `tool` can read it. A no-op without a data dir.

        Atomic and fsync'd (temp + fsync + rename, like the journal's snapshot /
        pending-wait writers) and called BEFORE the StepEvent is fsync-appended,
        so if the step is durable poke.json is too: crash recovery replays the
        step and finds the identical, non-torn file without re-materializing.
        """
        if self.data_dir is None:
            return
        self.data_dir.mkdir(parents=True, exist_ok=True)
        atomic_write(self.data_dir / "poke.json", json.dumps(payload, sort_keys=True))

    def notify(self, kind: str, state: str, message: str, level: str) -> None:
        if self.notify_hook is not None:
            self.notify_hook(kind, state, message, level)


# --------------------------------------------------------------------------
# Result.
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class MachineResult:
    status: Literal["ok", "failed", "incomplete", "waiting", "stopped"]
    reason: str
    state: str
    transitions: int

    @classmethod
    def from_end(cls, end: MachineEnd) -> MachineResult:
        """The engine outcome for a recorded end. `waiting`/`incomplete` outcomes
        (no journaled end) are built directly; only the end fact projects here."""
        return cls(end.status, end.reason, end.state, end.transitions)


# --------------------------------------------------------------------------
# Pure blackboard helpers.
# --------------------------------------------------------------------------


def initial_blackboard(spec: MachineSpec) -> dict[str, Any]:
    blackboard: dict[str, Any] = {}
    for name, var in spec.vars.operator.items():
        blackboard[name] = var.value
    for name, var in spec.vars.code.items():
        blackboard[name] = var.default
    for name, var in spec.vars.agent.items():
        blackboard[name] = var.default
    return blackboard


def _apply_capture(
    spec: MachineSpec, state: ToolState, stdout: str, blackboard: dict[str, Any]
) -> None:
    capture = state.capture
    if capture is None:
        return
    try:
        result_obj: Any = scrub_lone_surrogates(json.loads(stdout)) if stdout.strip() else None
    except json.JSONDecodeError as exc:
        raise StateRuntimeError(f"tool stdout is not valid JSON for capture: {exc}") from exc
    # Validate the parsed stdout against the tool's declared output_schema
    # before it touches the blackboard -- the capture gate docs/state-machines.md
    # §5.4 promises ("a malformed output halts the machine loudly"). Without it a
    # nonconforming value (a str where the schema says int) silently corrupts the
    # blackboard and misroutes downstream branches. Raised as a StateRuntimeError
    # so _step halts cleanly before journaling a poison fact, exactly like the
    # invalid-JSON case above and the agent finish_session path. output_schema is
    # optional on a tool (unlike an agent state); a schema-less tool declares no
    # shape, so there is nothing to check.
    if state.output_schema is not None:
        problems = validate_record_payload(
            spec.schemas, state.output_schema, result_obj, where="tool stdout"
        )
        if problems:
            raise StateRuntimeError(
                "tool stdout does not match output_schema: " + "; ".join(problems)
            )
    if capture.stdout_json is not None:
        blackboard[capture.stdout_json] = result_obj
        return
    if capture.set is not None:
        scope: dict[str, Any] = {**blackboard, "result": result_obj}
        for target, template_text in capture.set.items():
            template = parse_template(template_text)
            blackboard[target] = render_value(template, scope, where=f"state capture.set.{target}")


def _apply_agent_capture(state: AgentState, payload: Any, blackboard: dict[str, Any]) -> None:
    capture = state.capture
    if capture.finish_json is not None:
        blackboard[capture.finish_json] = payload
        return
    if capture.set is not None:
        scope: dict[str, Any] = {**blackboard, "result": payload}
        for target, template_text in capture.set.items():
            template = parse_template(template_text)
            blackboard[target] = render_value(template, scope, where=f"agent capture.set.{target}")


def reduce(
    spec: MachineSpec, state: StateSpec, fact: Fact, blackboard: dict[str, Any]
) -> dict[str, Any]:
    """Apply a journaled *fact* to the blackboard, returning a new dict."""
    updated = dict(blackboard)
    if (
        isinstance(state, ToolState)
        and isinstance(fact, ToolFact)
        and not fact.timed_out
        and fact.exit_code == 0
    ):
        _apply_capture(spec, state, fact.stdout, updated)
    elif isinstance(state, AgentState) and isinstance(fact, AgentFact) and fact.outcome == "ok":
        _apply_agent_capture(state, fact.payload, updated)
    return updated


# --------------------------------------------------------------------------
# Pure branch routing.
# --------------------------------------------------------------------------


def _route_branch(state: BranchState, blackboard: Mapping[str, object]) -> tuple[int, str, str]:
    for index, clause in enumerate(state.when):
        if clause.else_ is not None:
            return index, "else", clause.goto
        assert clause.if_ is not None
        if evaluate(parse_predicate(clause.if_), blackboard):
            return index, clause.if_, clause.goto
    # validate_semantics guarantees a final `else`, so this is unreachable.
    raise EngineError(f"branch fell through with no matching clause: {state.when!r}")


# --------------------------------------------------------------------------
# Wait timing.
# --------------------------------------------------------------------------


def _is_forever(state: WaitState) -> bool:
    """A `wait` with no timer parks until a signal poke (§4.3), no wake instant."""
    return state.every_secs is None and state.until is None


def _compute_wake(state: WaitState, blackboard: Mapping[str, object], now: float) -> float:
    if state.every_secs is not None:
        rendered = render_string(parse_template(state.every_secs), blackboard, where="every_secs")
        try:
            seconds = int(rendered)
        except ValueError as exc:
            raise StateRuntimeError(
                f"`every_secs` did not render to an integer: {rendered!r}"
            ) from exc
        if seconds < 1:
            raise StateRuntimeError(f"`every_secs` must be >= 1: {seconds}")
        return now + seconds
    if state.until is not None:
        rendered = render_string(parse_template(state.until), blackboard, where="until")
        try:
            moment = datetime.fromisoformat(rendered)
        except ValueError as exc:
            raise StateRuntimeError(f"`until` is not an ISO-8601 instant: {rendered!r}") from exc
        if moment.tzinfo is None:
            moment = moment.replace(tzinfo=UTC)
        return moment.timestamp()
    raise StateRuntimeError("a timerless `wait` has no wake instant to compute")


def _block_on_wait(
    state: WaitState,
    blackboard: Mapping[str, object],
    journal: MachineJournal,
    world: World,
    state_name: str,
) -> tuple[str, str, Fact] | None:
    """Foreground wait: block until the durable wake instant or a poke, or
    None when a stop request interrupted the sleep -- the pending wait stays
    armed, nothing is journaled, and the caller parks.

    The deadline persists BEFORE the sleep -- the same `PendingWait` the
    `--exit-on-wait` path keeps -- so a supervisor death mid-sleep resumes the
    original instant instead of re-running the full interval from a fresh
    `now()`. The driver clears it once the transition it produced is in the
    journal: a stale wait.json would suppress this state's notify on re-entry,
    reuse a stale wake_epoch under a later `--exit-on-wait`, and pin
    machine_is_parked in the web UI.
    """
    pending = journal.read_pending_wait()
    if pending is None or pending.state != state_name:
        wake = None if _is_forever(state) else _compute_wake(state, blackboard, world.now())
        pending = PendingWait(state=state_name, wake_epoch=wake)
        journal.write_pending_wait(pending)
    woke = world.sleep_until(pending.wake_epoch)
    if woke.woke_by == "stop":
        return None
    return (
        woke.woke_by,
        state.on[woke.woke_by],
        WaitFact(wake_epoch=pending.wake_epoch, woke_by=woke.woke_by, payload=woke.payload),
    )


def _fire_persisted_wait(
    state: WaitState,
    blackboard: Mapping[str, object],
    journal: MachineJournal,
    world: World,
    state_name: str,
) -> tuple[str, str, Fact] | None:
    """Arm-or-fire a `wait` without blocking (`--exit-on-wait`, §6).

    On first reaching the wait, the absolute wake instant is computed once and
    persisted so re-invocations compare against the same instant. Returns the
    `(label, goto, fact)` triple when the wait fires (a signal arrived or the
    instant has passed); returns `None` when the wait is not yet ready, leaving
    the record persisted for the caller to yield on. The record is the
    caller's to clear, once the transition it produced is in the journal.
    """
    pending = journal.read_pending_wait()
    if pending is None or pending.state != state_name:
        wake = None if _is_forever(state) else _compute_wake(state, blackboard, world.now())
        pending = PendingWait(state=state_name, wake_epoch=wake)
        journal.write_pending_wait(pending)
    signaled, payload = journal.take_signal()
    if signaled:
        return (
            "signal",
            state.on["signal"],
            WaitFact(wake_epoch=pending.wake_epoch, woke_by="signal", payload=payload),
        )
    if pending.wake_epoch is not None and world.now() >= pending.wake_epoch:
        return "tick", state.on["tick"], WaitFact(wake_epoch=pending.wake_epoch, woke_by="tick")
    return None


# --------------------------------------------------------------------------
# One impure step.
# --------------------------------------------------------------------------


def _agent_outcome(
    spec: MachineSpec, state: AgentState, result: AgentExecResult
) -> Literal["ok", "failed", "budget_exhausted", "timeout"]:
    if result.reason == "budget_exhausted":
        return "budget_exhausted"
    if result.reason == "timeout":
        return "timeout"
    if result.reason == "finish_session" and result.payload is not None:
        problems = validate_record_payload(
            spec.schemas, state.output_schema, result.payload, where="finish_session payload"
        )
        if not problems:
            return "ok"
    return "failed"


def _agent_usd_cap(state_cap: float | None, remaining: float | None) -> float | None:
    """The request's hard spend cap: the smaller of the state's own override
    and the machine budget still unspent. The aggregate gate only guards state
    STARTS, so without this an agent state's cap could exceed what the
    machine had left and the child billed past the machine's max_usd."""
    if state_cap is None:
        return remaining
    if remaining is None:
        return state_cap
    return min(state_cap, remaining)


def _execute(
    spec: MachineSpec,
    state: StateSpec,
    blackboard: Mapping[str, object],
    world: World,
    *,
    seq: int = 0,
    state_name: str = "",
    remaining_usd: float | None = None,
) -> tuple[str, str, Fact]:
    if isinstance(state, ToolState):
        argv = render_command(state.command, blackboard, where="command")
        # Under the explicit-only model a tool reaches the network iff it set
        # network = "host" (the operator-set ceiling + hardened limits are
        # enforced as machine-run startup refusals). "auto"/"none" → isolated.
        result = world.run_tool(
            tuple(argv),
            float(state.timeout_secs),
            # `auto` resolves to a network of its own: a machine state's
            # processes die with the state, so there is never a sibling to
            # share one with (see StateSpec.network).
            network="host" if state.network == "host" else "none",
        )
        if result.timed_out:
            label = "timeout"
        elif result.exit_code != 0:
            label = "nonzero"
        else:
            label = "ok"
        fact: Fact = ToolFact(
            exit_code=result.exit_code,
            stdout=result.stdout,
            timed_out=result.timed_out,
            stderr=result.stderr,
        )
        return label, state.on[label], fact
    if isinstance(state, BranchState):
        index, label, goto = _route_branch(state, blackboard)
        return label, goto, BranchFact(clause_index=index)
    if isinstance(state, AgentState):
        prompt = render_string(parse_template(state.prompt), blackboard, where="agent prompt")
        result = world.run_agent(
            AgentRequest(
                # "inherit" -> no override (None), so the world uses the
                # operator's effective worker model.
                model=None if state.model == "inherit" else state.model,
                prompt=prompt,
                timeout_s=float(state.timeout_secs),
                provider=state.provider,
                effort=state.effort,
                temperature=state.temperature,
                max_usd=_agent_usd_cap(state.max_usd, remaining_usd),
                max_tokens_fallback=state.max_tokens_fallback,
                mode=state.mode,
                # So the live World can give this execution its own watchable log.
                state_name=state_name,
                step_seq=seq,
                output_schema=state.output_schema,
                schemas=spec.schemas,
            )
        )
        outcome = _agent_outcome(spec, state, result)
        payload = result.payload if outcome == "ok" else None
        return (
            outcome,
            state.on[outcome],
            AgentFact(
                outcome=outcome,
                reason=result.reason,
                payload=payload,
                usd=result.usd,
                usd_partial=result.usd_partial,
                input_tokens=result.input_tokens,
                output_tokens=result.output_tokens,
            ),
        )
    raise EngineError(f"cannot execute terminal state directly: {state!r}")


# --------------------------------------------------------------------------
# The driver.
# --------------------------------------------------------------------------


def _emit_notify(
    state: StateSpec,
    blackboard: Mapping[str, object],
    journal: MachineJournal,
    world: World,
    state_name: str,
) -> None:
    """Journal a state's `notify` message on entry and fire the operator hook
    (§4.3). Presentation only: the render may raise, and the caller SWALLOWS that
    (a notify never affects control flow, so it never flips a terminal's real
    ok/failed status). No-op for a state with no `notify`."""
    if state.notify is None:
        return
    message = render_string(parse_template(state.notify.message), blackboard, where="notify")
    journal.append(
        MachineNotify(ts=_now_iso(), state=state_name, message=message, level=state.notify.level)
    )
    world.notify("notify", state_name, message, state.notify.level)


def _emit_end(
    journal: MachineJournal,
    world: World,
    *,
    status: Literal["ok", "failed"],
    reason: str,
    state: str,
    transitions: int,
    unbooked: AgentFact | None = None,
) -> MachineResult:
    """Journal a `machine.end` and fire the operator notify hook for it.

    *unbooked* is an agent slice that ran but whose StepEvent was never written
    (a capture that could not be reduced): its spend rides on the end event so
    the ledger still sees the dollars actually spent.
    """
    end = MachineEnd(
        ts=_now_iso(),
        status=status,
        reason=reason,
        state=state,
        transitions=transitions,
        usd=unbooked.usd if unbooked is not None else 0.0,
        usd_partial=unbooked.usd_partial if unbooked is not None else False,
        input_tokens=unbooked.input_tokens if unbooked is not None else 0,
        output_tokens=unbooked.output_tokens if unbooked is not None else 0,
    )
    journal.append(end)
    world.notify("end", state, reason, status)
    return MachineResult.from_end(end)


def _end_failed(
    journal: MachineJournal,
    world: World,
    state: str,
    transitions: int,
    exc: Exception,
    unbooked: AgentFact | None = None,
) -> MachineResult:
    """Journal a clean failed `MachineEnd` for a runtime state error and return it."""
    return _emit_end(
        journal,
        world,
        status="failed",
        reason=f"state {state!r}: {exc}",
        state=state,
        transitions=transitions,
        unbooked=unbooked,
    )


@dataclass(slots=True)
class _EngineState:
    """Mutable bookkeeping threaded through the engine's two phases.

    `drive` builds one, `_rebuild_from_journal` folds the recorded facts
    into it (crash recovery when live, offline backtest when not), then -- live
    only -- `_run_live_loop` continues from where the journal ends. Carrying
    the four cross-phase values in one object (rather than a six-arg call
    returning a four-tuple) lets each phase be a function taking `state`, per
    the AGENTS.md decompose rule.
    """

    spec: MachineSpec
    journal: MachineJournal
    world: World | None
    exit_on_wait: bool
    # Blackboard + position, folded by replay then advanced by the live loop.
    # `state` is the current state name; `spent_usd` sums agent-fact spend for
    # the cumulative usd_limit check.
    blackboard: dict[str, Any]
    state: str
    transitions: int = 0
    spent_usd: float = 0.0


_STATE_FACT_KINDS: dict[type[StateSpec], type[Fact]] = {
    ToolState: ToolFact,
    AgentState: AgentFact,
    WaitState: WaitFact,
    BranchState: BranchFact,
}


def _declared_gotos(state: StateSpec) -> frozenset[str]:
    """Every destination *state* can legally journal."""
    if isinstance(state, BranchState):
        return frozenset(clause.goto for clause in state.when)
    if isinstance(state, ToolState | AgentState | WaitState):
        return frozenset(state.on.values())
    return frozenset()


def _rebuild_from_journal(eng: _EngineState, events: list[Any]) -> None:
    """Replay recorded StepEvents through the pure reducer to rebuild the
    blackboard and position, advancing *eng* in place. Non-StepEvents
    (begin/notify/end) are skipped.

    Every recorded field the fold consumes is held to the machine: the step's
    state must match the replayed position, seqs must be contiguous, the fact
    kind must fit the state, and the goto must be an edge the state declares.
    A journal that fails any of these (corruption, hand-editing, a torn write)
    surfaces as a clean EngineError -- silently folding it would rebuild a
    position and blackboard the machine never reached."""
    spec = eng.spec
    blackboard = eng.blackboard
    state = eng.state
    transitions = eng.transitions
    spent_usd = eng.spent_usd
    remedy = " Archive the instance directory to start fresh."
    for event in events:
        if isinstance(event, AttemptSpend):
            # A crashed attempt's booked slice: real budget spent with no step
            # to carry it. Counts against max_usd; never moves the position.
            spent_usd += event.usd
            continue
        if not isinstance(event, StepEvent):
            continue
        state_spec = spec.states.get(state)
        if state_spec is None:
            raise EngineError(
                f"journal references state {state!r}, which the loaded machine no"
                " longer declares (the file was edited since this instance started);"
                f" archive the instance directory to start fresh: {eng.journal.root}"
            )
        if event.state != state:
            raise EngineError(
                f"journal diverges at seq {event.seq}: it records state"
                f" {event.state!r}, the replayed position is {state!r}.{remedy}"
            )
        if event.seq != transitions:
            raise EngineError(
                f"journal is not contiguous: expected seq {transitions}, found"
                f" {event.seq} (state {state!r}).{remedy}"
            )
        expected_kind = _STATE_FACT_KINDS.get(type(state_spec))
        if expected_kind is None or not isinstance(event.fact, expected_kind):
            raise EngineError(
                f"journal fact at seq {event.seq} is {type(event.fact).__name__},"
                f" which state {state!r} cannot produce.{remedy}"
            )
        if event.goto not in _declared_gotos(state_spec):
            raise EngineError(
                f"journal goto {event.goto!r} at seq {event.seq} is not an edge"
                f" state {state!r} declares.{remedy}"
            )
        try:
            blackboard = reduce(spec, state_spec, event.fact, blackboard)
        except _STATE_RUNTIME_ERRORS as exc:
            # An older journal (written before captures were validated pre-journal)
            # can hold a fact that no longer reduces. Surface it as a clean error,
            # not a traceback, so status/replay/resume stay inspectable.
            raise EngineError(f"cannot replay journaled step at state {state!r}: {exc}") from exc
        if isinstance(event.fact, AgentFact):
            spent_usd += event.fact.usd
        state = event.goto
        transitions = event.seq + 1
    eng.blackboard = blackboard
    eng.state = state
    eng.transitions = transitions
    eng.spent_usd = spent_usd


def _run_live_loop(eng: _EngineState) -> MachineResult:  # noqa: PLR0911, PLR0912, PLR0915
    """Continue live from where the journal ends: execute one state per
    iteration, journal its fact, and advance, until a terminal state (or a
    budget cap, a runtime state error, or an `--exit-on-wait` park) ends it."""
    spec = eng.spec
    journal = eng.journal
    exit_on_wait = eng.exit_on_wait
    world = eng.world
    if world is None:  # pragma: no cover - defensive
        raise EngineError("live execution requires a World")
    blackboard = eng.blackboard
    state = eng.state
    transitions = eng.transitions
    spent_usd = eng.spent_usd
    while True:
        # The durable stop request (`machine stop`, or any writer of the
        # instance's stop marker) parks at this boundary: the state in flight
        # already journaled its fact, no MachineEnd is written, and the
        # instance resumes exactly like a parked wait.
        if stop_requested(journal.root):
            clear_stop_request(journal.root)
            return MachineResult("stopped", "stop requested by the operator", state, transitions)
        current = spec.states.get(state)
        if current is None:
            raise EngineError(
                f"journal resumes at state {state!r}, which the loaded machine no"
                " longer declares (the file was edited since this instance started);"
                f" archive the instance directory to start fresh: {journal.root}"
            )
        # Emit a state's `notify` on entry (§4.3), before executing it. At-least-
        # once across a crash: a resume re-enters the current state and re-emits.
        # A wait whose PendingWait is already armed is NOT a fresh entry: every
        # --exit-on-wait scheduler tick re-drives into the parked state, and
        # without this guard the notify (and the operator hook: a page, an
        # email) re-fired once per poll for one park.
        already_parked = False
        if isinstance(current, WaitState):
            with contextlib.suppress(JournalError):
                pending = journal.read_pending_wait()
                already_parked = pending is not None and pending.state == state
        if not already_parked:
            try:
                _emit_notify(current, blackboard, journal, world, state)
            except _STATE_RUNTIME_ERRORS as exc:
                # Non-fatal by design (presentation never flips a terminal's
                # real status), but never silent: the failure is journaled and
                # the hook told, so a broken notify template is visible.
                fail = f"notify failed: {exc}"
                with contextlib.suppress(JournalError):
                    journal.append(
                        MachineNotify(ts=_now_iso(), state=state, message=fail, level="error")
                    )
                world.notify("notify", state, fail, "error")
        if isinstance(current, TerminalState):
            result = _emit_end(
                journal,
                world,
                status=current.status,
                reason=current.reason,
                state=state,
                transitions=transitions,
            )
            journal.write_snapshot(Snapshot(seq=transitions, state=state, blackboard=blackboard))
            return result
        if transitions >= spec.budget.max_transitions:
            reason = f"max_transitions ({spec.budget.max_transitions}) exceeded"
            return _emit_end(
                journal, world, status="failed", reason=reason, state=state, transitions=transitions
            )
        usd_limit = spec.budget.max_usd
        if usd_limit is not None and spent_usd >= usd_limit:
            reason = f"max_usd (${usd_limit}) exceeded (spent ~${spent_usd:.4f})"
            return _emit_end(
                journal, world, status="failed", reason=reason, state=state, transitions=transitions
            )
        remaining_usd = None if usd_limit is None else max(usd_limit - spent_usd, 0.0)

        try:
            if exit_on_wait and isinstance(current, WaitState):
                fired = _fire_persisted_wait(current, blackboard, journal, world, state)
                if fired is None:
                    pending = journal.read_pending_wait()
                    if pending is not None and pending.wake_epoch is not None:
                        detail = f"until {pending.wake_at}"
                    else:
                        detail = "until a signal poke"
                    return MachineResult(
                        "waiting", f"waiting in {state!r} {detail}", state, transitions
                    )
                label, goto, fact = fired
            elif isinstance(current, WaitState):
                blocked = _block_on_wait(current, blackboard, journal, world, state)
                if blocked is None:
                    clear_stop_request(journal.root)
                    return MachineResult(
                        "stopped", "stop requested by the operator", state, transitions
                    )
                label, goto, fact = blocked
            else:
                label, goto, fact = _execute(
                    spec,
                    current,
                    blackboard,
                    world,
                    seq=transitions,
                    state_name=state,
                    remaining_usd=remaining_usd,
                )
        except _STATE_RUNTIME_ERRORS as exc:
            # A data-driven state failure (e.g. an absent optional field, a tool
            # command rendering a non-scalar, a dynamic wait interval of zero):
            # halt cleanly with a journaled MachineEnd in BOTH the blocking and
            # the --exit-on-wait paths. Broader EngineError faults still propagate.
            return _end_failed(journal, world, state, transitions, exc)
        # Deliver a signal poke's payload to the next tool (both wait paths).
        if isinstance(fact, WaitFact) and fact.woke_by == "signal":
            world.materialize_poke(fact.payload)
        # Apply the capture BEFORE journaling the StepEvent. If a malformed output
        # (non-JSON / missing field / mistyped) can't be reduced, the machine halts
        # cleanly here instead of writing a poison fact that would re-crash every
        # later reduce (resume/status/replay), bricking the instance. The side
        # effect already ran; halting loudly matches the §4.2 capture contract.
        try:
            next_blackboard = reduce(spec, current, fact, blackboard)
        except _STATE_RUNTIME_ERRORS as exc:
            # The agent already ran and billed; the step is deliberately NOT
            # journaled (a fact whose capture fails would re-crash every later
            # replay), so hand its spend to the end event instead of dropping it.
            return _end_failed(
                journal,
                world,
                state,
                transitions,
                exc,
                unbooked=fact if isinstance(fact, AgentFact) else None,
            )
        journal.append(
            StepEvent(
                ts=_now_iso(),
                seq=transitions,
                state=state,
                label=label,
                goto=goto,
                fact=fact,
            )
        )
        # The wake record and the poke's claim are dropped only now that the
        # transition they produced is durable; a death anywhere earlier
        # re-delivers the poke and re-reads the SAME wake instant, rather than
        # arming a fresh interval from a new now().
        if isinstance(fact, WaitFact):
            journal.clear_pending_wait()
            if fact.woke_by == "signal":
                journal.ack_signal()
        blackboard = next_blackboard
        if isinstance(fact, AgentFact):
            spent_usd += fact.usd
        transitions += 1
        journal.write_snapshot(Snapshot(seq=transitions, state=goto, blackboard=blackboard))
        state = goto


def drive(
    spec: MachineSpec,
    journal: MachineJournal,
    world: World | None,
    *,
    live: bool,
    exit_on_wait: bool = False,
) -> MachineResult:
    """Run or replay *spec* against its *journal*.

    With `live=True` (`machine run`) the engine recovers from any existing
    journal, then continues to a terminal state, appending new facts. With
    `live=False` (`machine replay`) it only reconstructs the recorded path
    and reports where the journal ends; *world* is ignored.

    With `exit_on_wait=True` (`machine run --exit-on-wait`) the engine makes
    all the progress it can, but the first time it reaches a `wait` that is
    not yet ready it persists the absolute wake instant and returns a
    `"waiting"` result instead of blocking, for an external scheduler
    (systemd timer / cron) to re-invoke and resume (§6).
    """
    events = journal.read()
    if events and isinstance(events[-1], MachineEnd):
        return MachineResult.from_end(events[-1])

    # The instance is keyed only by the `machine` id, so a different file (or an
    # incompatible edit) can land on the same journal. Cross-check the recorded
    # identity so a mismatch fails loudly here, not as a KeyError mid-recovery.
    begin = events[0] if events else None
    if isinstance(begin, MachineBegin) and (
        begin.machine != spec.machine or begin.version != spec.version
    ):
        raise EngineError(
            f"this journal was started by machine {begin.machine!r} v{begin.version},"
            f" but the file declares {spec.machine!r} v{spec.version}. A different"
            " machine reused the id, or the file changed since this instance began;"
            f" archive the instance directory to start fresh: {journal.root}"
        )

    if not events and live:
        journal.ensure_dirs()
        journal.begin(machine=spec.machine, version=spec.version)

    eng = _EngineState(
        spec=spec,
        journal=journal,
        world=world,
        exit_on_wait=exit_on_wait,
        blackboard=initial_blackboard(spec),
        state=spec.initial,
    )
    _rebuild_from_journal(eng, events)

    if not live:
        current = spec.states.get(eng.state)
        if isinstance(current, TerminalState):
            return MachineResult(current.status, current.reason, eng.state, eng.transitions)
        return MachineResult(
            "incomplete", "journal ends before a terminal state", eng.state, eng.transitions
        )

    return _run_live_loop(eng)
