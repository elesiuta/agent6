# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Tests for the deterministic engine: run, capture, branch, replay, recovery."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from agent6.app.machine.run import machine_tool_policy_factory
from agent6.config import Config
from agent6.machine._semantics import load_machine
from agent6.machine.engine import (
    AgentExecResult,
    AgentRequest,
    EngineError,
    LiveWorld,
    MachineResult,
    ToolExecResult,
    WaitWake,
    World,
    drive,
)
from agent6.machine.journal import (
    AgentFact,
    BranchFact,
    MachineBegin,
    MachineEnd,
    MachineJournal,
    PendingWait,
    StepEvent,
    ToolFact,
)
from agent6.types import NetworkMode

# A minimal tool/branch/terminal machine: scan -> (branch on items) -> record -> stop.
COUNTER = """
machine = "counter"
version = 1
initial = "scan"

[budget]
max_usd = 1.0
max_transitions = 100

[vars.code]
items = { type = "list[str]", default = [] }

[schemas.scan_result]
items = "list[str]"

[states.scan]
kind = "tool"
command = ["scan"]
output_schema = "scan_result"
capture = { set = { items = "{{ result.items }}" } }
timeout_secs = 5
on = { ok = "check", nonzero = "stop_fail", timeout = "stop_fail" }

[states.check]
kind = "branch"
when = [
  { if = "len(items) == 0", goto = "stop_ok" },
  { else = true, goto = "record" },
]

[states.record]
kind = "tool"
command = ["record", "{{ items }}"]
timeout_secs = 5
on = { ok = "stop_ok", nonzero = "stop_fail", timeout = "stop_fail" }

[states.stop_ok]
kind = "terminal"
status = "ok"
reason = "done"

[states.stop_fail]
kind = "terminal"
status = "failed"
reason = "tool failed"
"""

# A waiting machine.
WAITER = """
machine = "waiter"
version = 1
initial = "poll"

[budget]
max_usd = 1.0
max_transitions = 100

[vars.operator]
secs = { type = "int", value = 1 }

[states.poll]
kind = "wait"
every_secs = "{{ secs }}"
on = { tick = "done", signal = "woken" }

[states.done]
kind = "terminal"
status = "ok"
reason = "ticked"

[states.woken]
kind = "terminal"
status = "ok"
reason = "signalled"
"""

# A mutable wait interval can become invalid at runtime; the engine must fail
# cleanly instead of busy-looping.
WAITER_DYNAMIC_ZERO = """
machine = "waiter_dynamic_zero"
version = 1
initial = "poll"

[budget]
max_usd = 1.0
max_transitions = 100

[vars.code]
secs = { type = "int", default = 0 }

[states.poll]
kind = "wait"
every_secs = "{{ secs }}"
on = { tick = "done", signal = "woken" }

[states.done]
kind = "terminal"
status = "ok"
reason = "ticked"

[states.woken]
kind = "terminal"
status = "ok"
reason = "signalled"
"""

# A waiting machine with a non-zero poll interval (for --exit-on-wait).
WAITER_DELAYED = """
machine = "waiter_delayed"
version = 1
initial = "poll"

[budget]
max_usd = 1.0
max_transitions = 100

[vars.operator]
secs = { type = "int", value = 60 }

[states.poll]
kind = "wait"
every_secs = "{{ secs }}"
on = { tick = "done", signal = "woken" }

[states.done]
kind = "terminal"
status = "ok"
reason = "ticked"

[states.woken]
kind = "terminal"
status = "ok"
reason = "signalled"
"""

# A wait with no timer: park until a signal poke, then a tool consumes the
# poke payload it materialized.
FOREVER = """
machine = "forever"
version = 1
initial = "park"

[budget]
max_usd = 1.0
max_transitions = 100

[vars.code]
last = { type = "json", default = {} }

[states.park]
kind = "wait"
on = { signal = "record" }

[states.record]
kind = "tool"
command = ["record"]
capture = { stdout_json = "last" }
timeout_secs = 5
on = { ok = "stop_ok", nonzero = "stop_fail", timeout = "stop_fail" }

[states.stop_ok]
kind = "terminal"
status = "ok"
reason = "done"

[states.stop_fail]
kind = "terminal"
status = "failed"
reason = "fail"
"""


# A terminal whose `notify` template fails to render at runtime (references an
# optional field that is absent) -- validates at load, raises at render.
NOTIFY_FAIL = """
machine = "notify_fail"
version = 1
initial = "route"

[budget]
max_transitions = 10

[schemas.r]
note = { type = "str", optional = true }

[vars.agent]
out = { type = "r", default = {} }

[states.route]
kind = "branch"
when = [ { else = true, goto = "done" } ]

[states.done]
kind = "terminal"
notify = "bye {{ out.note }}"
status = "ok"
reason = "finished"
"""


# A machine with a `notify` message on a tool state, plus a terminal.
NOTIFIER = """
machine = "notifier"
version = 1
initial = "work"

[budget]
max_usd = 1.0
max_transitions = 100

[vars.operator]
who = { type = "str", value = "ops" }

[states.work]
kind = "tool"
notify = { message = "hi {{ who }}", level = "warn" }
command = ["noop"]
timeout_secs = 5
on = { ok = "done", nonzero = "done", timeout = "done" }

[states.done]
kind = "terminal"
status = "ok"
reason = "finished"
"""


# An unbounded loop, guarded only by max_transitions.
SPINNER = """
machine = "spinner"
version = 1
initial = "spin"

[budget]
max_usd = 1.0
max_transitions = 3

[states.spin]
kind = "tool"
command = ["noop"]
timeout_secs = 5
on = { ok = "spin", nonzero = "spin", timeout = "spin" }
"""


# An agent state: review -> (branch on verdict.approved) -> stop_ok / stop_fail.
REVIEWER = """
machine = "reviewer"
version = 1
initial = "review"

[budget]
max_usd = 1.0
max_transitions = 100

[schemas.verdict]
approved = "bool"
note = "str"

[vars.agent]
verdict = { type = "verdict", default = {} }

[states.review]
kind = "agent"
model = "claude-sonnet-4-5"
prompt = "Review the change."
output_schema = "verdict"
capture = { finish_json = "verdict" }
timeout_secs = 600
on = { ok = "route", failed = "stop_fail", budget_exhausted = "halt", timeout = "expired" }

[states.route]
kind = "branch"
when = [
  { if = "verdict.approved", goto = "stop_ok" },
  { else = true, goto = "stop_fail" },
]

[states.stop_ok]
kind = "terminal"
status = "ok"
reason = "approved"

[states.stop_fail]
kind = "terminal"
status = "failed"
reason = "rejected"

[states.halt]
kind = "terminal"
status = "failed"
reason = "budget"

[states.expired]
kind = "terminal"
status = "failed"
reason = "timeout"
"""


# An agent state capturing a scalar field via `set` into a declared var.
SCORER = """
machine = "scorer"
version = 1
initial = "score"

[budget]
max_usd = 1.0
max_transitions = 100

[schemas.score_result]
points = "int"

[vars.agent]
total = { type = "int", default = 0 }

[states.score]
kind = "agent"
model = "m"
prompt = "Score it."
output_schema = "score_result"
capture = { set = { total = "{{ result.points }}" } }
timeout_secs = 60
on = { ok = "stop_ok", failed = "stop_fail", budget_exhausted = "stop_fail", timeout = "stop_fail" }

[states.stop_ok]
kind = "terminal"
status = "ok"
reason = "done"

[states.stop_fail]
kind = "terminal"
status = "failed"
reason = "fail"
"""


@dataclass
class FakeWorld:
    """A deterministic :class:`World`: programmed tool results and wakes."""

    tool_results: dict[str, ToolExecResult]
    wakes: list[WaitWake] = field(default_factory=list)
    clock: float = 1000.0
    calls: list[tuple[str, ...]] = field(default_factory=list)
    net_calls: list[tuple[tuple[str, ...], NetworkMode]] = field(default_factory=list)
    agent_results: list[AgentExecResult] = field(default_factory=list)
    agent_calls: list[AgentRequest] = field(default_factory=list)
    sleep_deadlines: list[float | None] = field(default_factory=list)
    materialized: list[Any] = field(default_factory=list)
    notifications: list[tuple[str, str, str, str]] = field(default_factory=list)

    def run_tool(
        self, argv: tuple[str, ...], timeout_s: float, *, network: NetworkMode = "none"
    ) -> ToolExecResult:
        self.calls.append(argv)
        self.net_calls.append((argv, network))
        return self.tool_results[argv[0]]

    def run_agent(self, request: AgentRequest) -> AgentExecResult:
        self.agent_calls.append(request)
        return self.agent_results.pop(0)

    def now(self) -> float:
        return self.clock

    def sleep_until(self, wake_epoch: float | None) -> WaitWake:
        self.sleep_deadlines.append(wake_epoch)
        return self.wakes.pop(0) if self.wakes else WaitWake("tick")

    def materialize_poke(self, payload: Any) -> None:
        self.materialized.append(payload)

    def notify(self, kind: str, state: str, message: str, level: str) -> None:
        self.notifications.append((kind, state, message, level))


def _ok(stdout: str = "") -> ToolExecResult:
    return ToolExecResult(exit_code=0, stdout=stdout, timed_out=False)


def _load(tmp_path: Path, text: str) -> tuple[MachineJournal, Path]:
    f = tmp_path / "m.asm.toml"
    f.write_text(text, encoding="utf-8")
    return MachineJournal(tmp_path / "inst"), f


def test_full_run_reaches_ok_terminal(tmp_path: Path) -> None:
    journal, f = _load(tmp_path, COUNTER)
    spec = load_machine(f)
    world: World = FakeWorld({"scan": _ok('{"items": ["a", "b"]}'), "record": _ok()})
    result = drive(spec, journal, world, live=True)
    assert result == MachineResult("ok", "done", "stop_ok", 3)
    snap = journal.latest_snapshot()
    assert snap is not None
    assert snap.blackboard["items"] == ["a", "b"]


def test_branch_routes_to_ok_when_empty(tmp_path: Path) -> None:
    journal, f = _load(tmp_path, COUNTER)
    spec = load_machine(f)
    world = FakeWorld({"scan": _ok('{"items": []}')})
    result = drive(spec, journal, world, live=True)
    # scan -> check -> stop_ok (record never runs)
    assert result == MachineResult("ok", "done", "stop_ok", 2)
    assert world.calls == [("scan",)]


def test_tool_stdout_violating_output_schema_halts(tmp_path: Path) -> None:
    """A tool whose parsed stdout does not match its declared output_schema must
    halt the machine loudly (a clean failed MachineResult), not silently capture
    a mistyped value that corrupts the blackboard and misroutes downstream
    branches. scan_result.items is list[str]; emit a bare string instead."""
    journal, f = _load(tmp_path, COUNTER)
    spec = load_machine(f)
    world = FakeWorld({"scan": _ok('{"items": "notalist"}')})
    result = drive(spec, journal, world, live=True)
    assert result.status == "failed"
    assert "output_schema" in result.reason
    # The capture never ran, so the poison value is not journaled and no
    # snapshot carries a corrupted `items`.
    snap = journal.latest_snapshot()
    assert snap is None or snap.blackboard.get("items") == []


def test_tool_nonzero_routes_to_failure(tmp_path: Path) -> None:
    journal, f = _load(tmp_path, COUNTER)
    spec = load_machine(f)
    world = FakeWorld({"scan": ToolExecResult(exit_code=1, stdout="", timed_out=False)})
    result = drive(spec, journal, world, live=True)
    assert result.status == "failed"
    assert result.state == "stop_fail"


def test_tool_stderr_is_journaled_on_the_fact(tmp_path: Path) -> None:
    # A failing tool's stderr flows ToolExecResult -> ToolFact -> journal, so the
    # failure is debuggable from the journal. Routing still keys off exit_code.
    journal, f = _load(tmp_path, COUNTER)
    spec = load_machine(f)
    world = FakeWorld(
        {"scan": ToolExecResult(exit_code=2, stdout="", timed_out=False, stderr="boom: bad flag\n")}
    )
    assert drive(spec, journal, world, live=True).state == "stop_fail"
    tool_steps = [
        e for e in journal.read() if isinstance(e, StepEvent) and isinstance(e.fact, ToolFact)
    ]
    assert tool_steps
    fact = tool_steps[0].fact
    assert isinstance(fact, ToolFact) and fact.stderr == "boom: bad flag\n"


def test_tool_timeout_routes_to_failure(tmp_path: Path) -> None:
    journal, f = _load(tmp_path, COUNTER)
    spec = load_machine(f)
    world = FakeWorld({"scan": ToolExecResult(exit_code=0, stdout="", timed_out=True)})
    result = drive(spec, journal, world, live=True)
    assert result.status == "failed"
    assert result.state == "stop_fail"


def test_tool_bad_stdout_fails_clean_without_poisoning_journal(tmp_path: Path) -> None:
    # A tool that exits 0 but prints non-JSON stdout cannot be captured. The
    # machine must halt FAILED cleanly, and -- critically -- never journal the
    # poison fact, so a later replay/status re-reduces without re-crashing.
    journal, f = _load(tmp_path, COUNTER)
    spec = load_machine(f)
    world = FakeWorld({"scan": _ok("not json at all")})
    result = drive(spec, journal, world, live=True)
    assert result.status == "failed"
    assert "not valid JSON" in result.reason
    # No StepEvent was written: only MachineBegin + MachineEnd.
    events = journal.read()
    assert not any(isinstance(e, StepEvent) for e in events)
    assert isinstance(events[-1], MachineEnd)
    # Replay over the same journal returns the failure, it does not raise.
    replayed = drive(spec, journal, None, live=False)
    assert replayed.status == "failed"


def test_recovery_rejects_a_goto_the_state_never_declared(tmp_path: Path) -> None:
    # A fabricated or corrupted destination must fail loudly at its own event,
    # not fold a position the machine never reached.
    journal, f = _load(tmp_path, COUNTER)
    spec = load_machine(f)
    journal.ensure_dirs()
    journal.begin(machine="counter", version=1)
    journal.append(
        StepEvent(
            ts="t",
            seq=0,
            state="scan",
            label="ok",
            goto="ghost",  # not an edge scan declares
            fact=ToolFact(exit_code=0, stdout='{"items": []}', timed_out=False),
        )
    )
    with pytest.raises(EngineError, match="not an edge"):
        drive(spec, journal, FakeWorld({}), live=True)


def test_recovery_rejects_a_state_the_edited_file_dropped(tmp_path: Path) -> None:
    # A legitimately recorded journal against a machine file later edited to
    # drop the destination state must fail loudly, not raise a bare KeyError.
    journal, _f = _load(tmp_path, COUNTER)
    journal.ensure_dirs()
    journal.begin(machine="counter", version=1)
    journal.append(
        StepEvent(
            ts="t",
            seq=0,
            state="scan",
            label="ok",
            goto="check",  # a real edge when recorded
            fact=ToolFact(exit_code=0, stdout='{"items": ["a"]}', timed_out=False),
        )
    )
    journal.append(
        StepEvent(
            ts="t",
            seq=1,
            state="check",
            label="[1]",
            goto="record",
            fact=BranchFact(clause_index=1),
        ),
    )
    edited = COUNTER.replace(
        """[states.record]
kind = "tool"
command = ["record", "{{ items }}"]
timeout_secs = 5
on = { ok = "stop_ok", nonzero = "stop_fail", timeout = "stop_fail" }

""",
        "",
    ).replace('{ else = true, goto = "record" }', '{ else = true, goto = "stop_ok" }')
    edited_dir = tmp_path / "edited"
    edited_dir.mkdir()
    spec = load_machine(_load(edited_dir, edited)[1])
    with pytest.raises(EngineError, match=r"no longer declares|not an edge"):
        drive(spec, journal, FakeWorld({}), live=True)


def test_recovery_rejects_machine_id_mismatch(tmp_path: Path) -> None:
    # A different machine reusing the same instance id is caught up front.
    journal, f = _load(tmp_path, COUNTER)
    spec = load_machine(f)
    journal.ensure_dirs()
    journal.begin(machine="someone_else", version=1)
    with pytest.raises(EngineError, match="someone_else"):
        drive(spec, journal, FakeWorld({}), live=True)


def test_record_splices_captured_list(tmp_path: Path) -> None:
    journal, f = _load(tmp_path, COUNTER)
    spec = load_machine(f)
    world = FakeWorld({"scan": _ok('{"items": ["x", "y", "z"]}'), "record": _ok()})
    drive(spec, journal, world, live=True)
    assert world.calls == [("scan",), ("record", "x", "y", "z")]


def test_replay_reproduces_path_without_world(tmp_path: Path) -> None:
    journal, f = _load(tmp_path, COUNTER)
    spec = load_machine(f)
    world = FakeWorld({"scan": _ok('{"items": ["a"]}'), "record": _ok()})
    live = drive(spec, journal, world, live=True)
    replayed = drive(spec, journal, None, live=False)
    assert replayed == live


def test_replay_of_incomplete_journal(tmp_path: Path) -> None:
    journal, f = _load(tmp_path, COUNTER)
    spec = load_machine(f)
    journal.ensure_dirs()
    journal.begin(machine="counter", version=1)
    journal.append(
        StepEvent(
            ts="t",
            seq=0,
            state="scan",
            label="ok",
            goto="check",
            fact=ToolFact(exit_code=0, stdout='{"items": ["a"]}', timed_out=False),
        )
    )
    result = drive(spec, journal, None, live=False)
    assert result.status == "incomplete"
    assert result.state == "check"
    assert result.transitions == 1


def test_crash_recovery_continues_without_redoing_step(tmp_path: Path) -> None:
    journal, f = _load(tmp_path, COUNTER)
    spec = load_machine(f)
    # Simulate a crash right after `scan` completed: journal has begin + the
    # scan step, but no terminal. Recovery must rebuild `items` from the
    # recorded fact and continue from `check` without re-running `scan`.
    journal.ensure_dirs()
    journal.begin(machine="counter", version=1)
    journal.append(
        StepEvent(
            ts="t",
            seq=0,
            state="scan",
            label="ok",
            goto="check",
            fact=ToolFact(exit_code=0, stdout='{"items": ["a", "b"]}', timed_out=False),
        )
    )
    world = FakeWorld({"record": _ok()})  # no "scan" — proving it is not re-run
    result = drive(spec, journal, world, live=True)
    assert result == MachineResult("ok", "done", "stop_ok", 3)
    assert world.calls == [("record", "a", "b")]


def test_resume_finished_machine_is_idempotent(tmp_path: Path) -> None:
    journal, f = _load(tmp_path, COUNTER)
    spec = load_machine(f)
    world = FakeWorld({"scan": _ok('{"items": []}')})
    first = drive(spec, journal, world, live=True)
    # A second run with no world at all returns the recorded terminal result.
    again = drive(spec, journal, None, live=True)
    assert again == first


def test_max_transitions_halts_loop(tmp_path: Path) -> None:
    journal, f = _load(tmp_path, SPINNER)
    spec = load_machine(f)
    world = FakeWorld({"noop": _ok()})
    result = drive(spec, journal, world, live=True)
    assert result.status == "failed"
    assert "max_transitions" in result.reason
    assert result.transitions == 3


def test_wait_tick_path(tmp_path: Path) -> None:
    journal, f = _load(tmp_path, WAITER)
    spec = load_machine(f)
    world = FakeWorld({}, wakes=[WaitWake("tick")])
    result = drive(spec, journal, world, live=True)
    assert result == MachineResult("ok", "ticked", "done", 1)
    events = journal.read()
    step = next(e for e in events if isinstance(e, StepEvent))
    assert step.label == "tick"


def test_wait_zero_dynamic_interval_fails_cleanly(tmp_path: Path) -> None:
    journal, f = _load(tmp_path, WAITER_DYNAMIC_ZERO)
    spec = load_machine(f)
    result = drive(spec, journal, FakeWorld({}), live=True)
    assert result.status == "failed"
    assert "`every_secs` must be >= 1" in result.reason
    assert not any(isinstance(event, StepEvent) for event in journal.read())


def test_run_tool_uses_the_injected_jail_runner(tmp_path: Path) -> None:
    """The tool step executes through LiveWorld.jail_runner, the seam the CLI
    overrides so a run-machine's tools run in the machine's own tree; the
    default stays the plain jail."""
    from agent6.types import CommandResult, JailPolicy

    seen: list[JailPolicy] = []

    def _runner(policy: JailPolicy) -> CommandResult:
        seen.append(policy)
        return CommandResult(argv=policy.argv, returncode=0, stdout="{}", stderr="", duration_s=0.0)

    world = LiveWorld(
        cwd=tmp_path,
        journal=MachineJournal(tmp_path / "i"),
        tool_policy=lambda argv, timeout_s, network: JailPolicy(cwd=tmp_path, argv=argv),
        jail_runner=_runner,
    )
    res = world.run_tool(("echo", "hi"), 5.0)
    assert res.exit_code == 0 and not res.timed_out
    assert [p.argv for p in seen] == [("echo", "hi")]


def test_exit_on_wait_zero_dynamic_interval_fails_cleanly(tmp_path: Path) -> None:
    # The --exit-on-wait path must halt FAILED with a journaled MachineEnd too,
    # not leave the instance "incomplete" with the error escaping as a CLI error.
    journal, f = _load(tmp_path, WAITER_DYNAMIC_ZERO)
    spec = load_machine(f)
    result = drive(spec, journal, FakeWorld({}), live=True, exit_on_wait=True)
    assert result.status == "failed"
    assert "`every_secs` must be >= 1" in result.reason
    assert isinstance(journal.read()[-1], MachineEnd)


def test_wait_signal_path(tmp_path: Path) -> None:
    journal, f = _load(tmp_path, WAITER)
    spec = load_machine(f)
    world = FakeWorld({}, wakes=[WaitWake("signal")])
    result = drive(spec, journal, world, live=True)
    assert result == MachineResult("ok", "signalled", "woken", 1)


def test_a_foreground_wait_persists_its_deadline_before_sleeping(tmp_path: Path) -> None:
    """Only --exit-on-wait persisted the wake instant; a supervisor death
    mid-foreground-sleep lost it, so restart recomputed from a fresh now() and
    the full interval ran again. The foreground path writes the same durable
    PendingWait before blocking, hands the sleep that instant, and resumes a
    pre-existing pending for the state instead of recomputing."""

    class _ClockedWorld:
        def __init__(self, journal: MachineJournal) -> None:
            self._journal = journal
            self.slept: list[float | None] = []
            self.persisted: list[float | None] = []

        def run_tool(self, argv: Any, timeout_s: Any, *, network: Any = "none") -> Any:
            raise AssertionError("no tool states")

        def run_agent(self, request: Any, events_log: Any = None) -> Any:
            raise AssertionError("no agent states")

        def now(self) -> float:
            return 1000.0

        def sleep_until(self, wake_epoch: float | None) -> WaitWake:
            pending = self._journal.read_pending_wait()
            self.persisted.append(None if pending is None else pending.wake_epoch)
            self.slept.append(wake_epoch)
            return WaitWake("tick")

        def materialize_poke(self, payload: Any) -> None:
            pass

        def notify(self, kind: str, state: str, message: str, level: str) -> None:
            pass

    journal, f = _load(tmp_path, WAITER_DELAYED)
    spec = load_machine(f)
    world = _ClockedWorld(journal)
    assert drive(spec, journal, world, live=True).status == "ok"
    # The deadline was durable before the sleep, and the sleep received it.
    assert world.persisted == [1060.0]
    assert world.slept == [1060.0]
    assert journal.read_pending_wait() is None  # consumed wake leaves no stale park

    # Restart mid-sleep: the persisted instant is resumed, never recomputed.
    other = tmp_path / "other"
    journal2 = MachineJournal(other / "inst")
    journal2.write_pending_wait(PendingWait(state="poll", wake_epoch=1013.5))
    world2 = _ClockedWorld(journal2)
    assert drive(spec, journal2, world2, live=True).status == "ok"
    assert world2.slept == [1013.5]


def test_a_signal_wake_acks_its_poke_after_the_step(tmp_path: Path) -> None:
    """The poke's claim file outlives take_signal (a death before the wake's
    StepEvent re-delivers it on restart) and is dropped by the driver once the
    step is durable -- without that ack every poke would re-deliver forever."""
    journal, f = _load(tmp_path, WAITER)
    spec = load_machine(f)
    journal.poke("p")
    assert journal.take_signal() == (True, "p")  # the claim is made, unacked
    world = FakeWorld({}, wakes=[WaitWake("signal", "p")])
    result = drive(spec, journal, world, live=True)
    assert result == MachineResult("ok", "signalled", "woken", 1)
    assert not journal.signal_path.with_suffix(".consuming").exists()
    assert journal.take_signal() == (False, None)


def test_a_wake_record_outlives_the_step_that_consumes_it(tmp_path: Path) -> None:
    """The record goes the way the poke's claim goes: dropped only once the
    transition it produced is in the journal. Cleared before the append, a
    death in that window re-armed the wait from the state and started a
    24-hour sleep again from zero."""

    journal, f = _load(tmp_path, WAITER)
    spec = load_machine(f)
    armed: list[PendingWait | None] = []
    appended = journal.append

    def _die_on_the_step(event: object) -> None:
        if isinstance(event, StepEvent):
            armed.append(journal.read_pending_wait())
            raise KeyboardInterrupt("died between the wake and its StepEvent")
        appended(event)  # pyright: ignore[reportArgumentType]

    journal.append = _die_on_the_step  # type: ignore[method-assign]
    with pytest.raises(KeyboardInterrupt):
        drive(spec, journal, FakeWorld({}, wakes=[WaitWake("tick")]), live=True)
    journal.append = appended  # type: ignore[method-assign]

    # The record the crashed run armed is still there for the restart, so the
    # wait resumes on the SAME instant instead of arming a fresh one.
    assert armed and armed[0] is not None
    assert journal.read_pending_wait() == armed[0]


def test_notify_journals_event_and_fires_hook(tmp_path: Path) -> None:
    from agent6.machine.journal import MachineNotify

    journal, f = _load(tmp_path, NOTIFIER)
    spec = load_machine(f)
    world = FakeWorld({"noop": _ok()})
    result = drive(spec, journal, world, live=True)
    assert result.status == "ok"
    # The `notify` message rendered and journaled on entry to `work`.
    note = next(e for e in journal.read() if isinstance(e, MachineNotify))
    assert note.state == "work"
    assert note.message == "hi ops"
    assert note.level == "warn"
    # The operator hook fired for the notify AND the terminal end.
    assert ("notify", "work", "hi ops", "warn") in world.notifications
    assert ("end", "done", "finished", "ok") in world.notifications


def test_terminal_notify_render_failure_keeps_status_but_is_never_silent(
    tmp_path: Path,
) -> None:
    """`notify` is presentation only: a render failure on a terminal must NOT
    flip the terminal's real ok/failed status. It must not vanish either --
    the swallow left no journal entry and no hook fire, so a broken template
    meant notifications silently stopped. The failure is journaled as an
    error-level machine.notify and the hook is told."""
    from agent6.machine.journal import MachineNotify

    journal, f = _load(tmp_path, NOTIFY_FAIL)
    spec = load_machine(f)
    world = FakeWorld({})
    result = drive(spec, journal, world, live=True)
    assert result.status == "ok"
    assert result.reason == "finished"
    note = next(e for e in journal.read() if isinstance(e, MachineNotify))
    assert note.level == "error" and "notify failed" in note.message
    assert any(
        kind == "notify" and level == "error" and "notify failed" in message
        for kind, _state, message, level in world.notifications
    )
    events_ok = journal.read()
    assert isinstance(events_ok[-1], MachineEnd)
    assert events_ok[-1].status == "ok"


def test_live_world_materializes_poke_atomically(tmp_path: Path) -> None:
    from agent6.machine.engine import LiveWorld

    data_dir = tmp_path / "data"
    world = LiveWorld(cwd=tmp_path, journal=MachineJournal(tmp_path / "i"), data_dir=data_dir)
    world.materialize_poke({"cmd": "go", "n": 2})
    poke = data_dir / "poke.json"
    assert json.loads(poke.read_text(encoding="utf-8")) == {"cmd": "go", "n": 2}
    # No leftover temp file (atomic temp+rename), so a reader never sees a torn file.
    assert not (data_dir / "poke.json.tmp").exists()
    # No data dir -> a silent no-op, never a crash.
    LiveWorld(cwd=tmp_path, journal=MachineJournal(tmp_path / "i")).materialize_poke("x")


def test_data_dir_env_matches_jail_mount(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # The data dir lives OUTSIDE cwd by design; the jail mounts extra_rw_paths
    # at their real locations in every isolation, so `$AGENT6_MACHINE_DATA_DIR`
    # is always the host abspath (strict used to need a /rw prefix rewrite).
    from agent6.machine import engine
    from agent6.machine.engine import LiveWorld
    from agent6.types import CommandResult, IsolationLevel, JailPolicy

    data_dir = tmp_path / "state" / "machines" / "m" / "data"
    captured: dict[str, JailPolicy] = {}

    def fake_run_in_jail(policy: JailPolicy) -> CommandResult:
        captured["policy"] = policy
        return CommandResult(argv=policy.argv, returncode=0, stdout="{}", stderr="", duration_s=0.0)

    monkeypatch.setattr(engine, "run_in_jail", fake_run_in_jail)

    levels: tuple[IsolationLevel, ...] = ("strict", "hardened")
    for isolation in levels:
        world = LiveWorld(
            cwd=tmp_path,
            journal=MachineJournal(tmp_path / "i"),
            data_dir=data_dir,
            tool_policy=machine_tool_policy_factory(
                Config(), tmp_path, isolation, protect_paths=(), data_dir=data_dir
            ),
        )
        world.run_tool(("python3", "x.py"), 5.0)
        policy = captured["policy"]
        assert dict(policy.env)["AGENT6_MACHINE_DATA_DIR"] == str(data_dir)
        assert data_dir in policy.extra_rw_paths


def test_tool_jails_carry_the_operator_hide_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A machine's tool jails are jailed commands like any other, so
    [sandbox].hide_paths reaches them (agent6's own private dirs are unioned
    in by the launcher and need no wiring)."""
    from agent6.machine import engine
    from agent6.machine.engine import LiveWorld
    from agent6.types import CommandResult, JailPolicy

    captured: dict[str, JailPolicy] = {}

    def fake_run_in_jail(policy: JailPolicy) -> CommandResult:
        captured["policy"] = policy
        return CommandResult(argv=policy.argv, returncode=0, stdout="", stderr="", duration_s=0.0)

    monkeypatch.setattr(engine, "run_in_jail", fake_run_in_jail)
    hidden = tmp_path / "cred.txt"
    cfg = Config.model_validate({"sandbox": {"hide_paths": [str(hidden)]}})
    world = LiveWorld(
        cwd=tmp_path,
        journal=MachineJournal(tmp_path / "i"),
        tool_policy=machine_tool_policy_factory(
            cfg, tmp_path, "strict", protect_paths=(), data_dir=None
        ),
    )
    world.run_tool(("python3", "x.py"), 5.0)
    assert hidden in captured["policy"].hide_paths


def test_live_world_run_tool_maps_rc124_to_timed_out(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """run_in_jail's contract is 'a timeout is returncode 124, never a raised
    TimeoutExpired'. LiveWorld.run_tool must derive timed_out from that so a
    tool state's on.timeout transition is reachable; the old `except
    TimeoutExpired` was dead code and every real timeout returned
    timed_out=False (routing to on.nonzero and journaling not-timed-out)."""
    from agent6.machine import engine
    from agent6.machine.engine import LiveWorld
    from agent6.types import CommandResult, JailPolicy

    def fake_run_in_jail(policy: JailPolicy) -> CommandResult:
        # The launcher SIGKILLed the child at the deadline and reported rc=124.
        return CommandResult(argv=policy.argv, returncode=124, stdout="", stderr="", duration_s=0.0)

    monkeypatch.setattr(engine, "run_in_jail", fake_run_in_jail)
    world = LiveWorld(
        cwd=tmp_path,
        journal=MachineJournal(tmp_path / "i"),
        tool_policy=machine_tool_policy_factory(
            Config(), tmp_path, "strict", protect_paths=(), data_dir=None
        ),
    )
    res = world.run_tool(("sleep", "99"), 1.0)
    assert res.timed_out is True
    assert res.exit_code == 124

    # A normal nonzero exit stays not-timed-out.
    def plain_nonzero(policy: JailPolicy) -> CommandResult:
        return CommandResult(argv=policy.argv, returncode=2, stdout="", stderr="", duration_s=0.0)

    monkeypatch.setattr(engine, "run_in_jail", plain_nonzero)
    res2 = world.run_tool(("false",), 1.0)
    assert res2.timed_out is False
    assert res2.exit_code == 2


def test_notify_does_not_change_replay_path(tmp_path: Path) -> None:
    # machine.notify is presentation only: replay ignores it and reproduces.
    journal, f = _load(tmp_path, NOTIFIER)
    spec = load_machine(f)
    live = drive(spec, journal, FakeWorld({"noop": _ok()}), live=True)
    replayed = drive(spec, journal, None, live=False)
    assert replayed == live


def test_wait_forever_blocks_on_signal_only(tmp_path: Path) -> None:
    journal, f = _load(tmp_path, FOREVER)
    spec = load_machine(f)
    world = FakeWorld(
        {"record": _ok('{"got": true}')}, wakes=[WaitWake("signal", {"cmd": "go", "n": 2})]
    )
    result = drive(spec, journal, world, live=True)
    assert result == MachineResult("ok", "done", "stop_ok", 2)
    # A no-timer wait passes None as the deadline (park until a signal).
    assert world.sleep_deadlines == [None]
    # The poke payload was materialized for the next tool and journaled.
    assert world.materialized == [{"cmd": "go", "n": 2}]
    from agent6.machine.journal import WaitFact

    wait_step = next(
        e for e in journal.read() if isinstance(e, StepEvent) and isinstance(e.fact, WaitFact)
    )
    assert isinstance(wait_step.fact, WaitFact)
    assert wait_step.fact.wake_epoch is None
    assert wait_step.fact.payload == {"cmd": "go", "n": 2}


def test_wait_forever_payload_reproduces_on_replay(tmp_path: Path) -> None:
    # The journaled poke payload is a fact: replay rebuilds the identical path.
    journal, f = _load(tmp_path, FOREVER)
    spec = load_machine(f)
    world = FakeWorld({"record": _ok('{"got": true}')}, wakes=[WaitWake("signal", {"cmd": "go"})])
    live = drive(spec, journal, world, live=True)
    replayed = drive(spec, journal, None, live=False)
    assert replayed == live


def test_exit_on_wait_forever_parks_until_signal(tmp_path: Path) -> None:
    journal, f = _load(tmp_path, FOREVER)
    spec = load_machine(f)
    # No timer: --exit-on-wait persists a signal-only pending wait (no instant).
    result = drive(spec, journal, FakeWorld({}), live=True, exit_on_wait=True)
    assert result.status == "waiting"
    assert "signal poke" in result.reason
    pending = journal.read_pending_wait()
    assert pending is not None
    assert pending.wake_epoch is None
    # A poke with a payload fires it on the next scheduler invocation.
    journal.poke({"cmd": "go"})
    result = drive(
        spec,
        journal,
        FakeWorld({"record": _ok('{"got": true}')}),
        live=True,
        exit_on_wait=True,
    )
    assert result == MachineResult("ok", "done", "stop_ok", 2)
    wait_step = next(e for e in journal.read() if isinstance(e, StepEvent))
    from agent6.machine.journal import WaitFact

    assert isinstance(wait_step.fact, WaitFact)
    assert wait_step.fact.payload == {"cmd": "go"}


def test_exit_on_wait_arms_and_yields_waiting(tmp_path: Path) -> None:
    journal, f = _load(tmp_path, WAITER_DELAYED)
    spec = load_machine(f)
    world = FakeWorld({}, clock=1000.0)
    result = drive(spec, journal, world, live=True, exit_on_wait=True)
    assert result.status == "waiting"
    assert result.state == "poll"
    assert result.transitions == 0
    # The wake instant was persisted once; no step was appended yet.
    pending = journal.read_pending_wait()
    assert pending is not None
    assert pending.state == "poll"
    assert pending.wake_epoch == 1060.0
    assert not any(isinstance(e, StepEvent) for e in journal.read())
    # Both surfaces that show an operator when this wakes read one owner, so
    # the run banner cannot print a raw epoch while `machine status` prints a
    # timestamp for the same instant.
    assert pending.wake_at == "1970-01-01T00:17:40+00:00"
    assert pending.wake_at in result.reason


def test_exit_on_wait_fires_tick_when_due(tmp_path: Path) -> None:
    journal, f = _load(tmp_path, WAITER_DELAYED)
    spec = load_machine(f)
    # First invocation arms the wait (wake at 1060).
    drive(spec, journal, FakeWorld({}, clock=1000.0), live=True, exit_on_wait=True)
    # A later scheduler tick re-invokes once the instant has passed.
    result = drive(spec, journal, FakeWorld({}, clock=1060.0), live=True, exit_on_wait=True)
    assert result == MachineResult("ok", "ticked", "done", 1)
    step = next(e for e in journal.read() if isinstance(e, StepEvent))
    assert step.label == "tick"
    # The persisted wait was cleared once it fired.
    assert journal.read_pending_wait() is None


def test_exit_on_wait_fires_signal_before_due(tmp_path: Path) -> None:
    journal, f = _load(tmp_path, WAITER_DELAYED)
    spec = load_machine(f)
    drive(spec, journal, FakeWorld({}, clock=1000.0), live=True, exit_on_wait=True)
    # Operator poke arrives before the wake instant.
    journal.poke()
    result = drive(spec, journal, FakeWorld({}, clock=1005.0), live=True, exit_on_wait=True)
    assert result == MachineResult("ok", "signalled", "woken", 1)
    assert journal.read_pending_wait() is None


def test_blocking_wait_clears_persisted_wait(tmp_path: Path) -> None:
    # An --exit-on-wait invocation arms wait.json; a later BLOCKING run consumes
    # the wake via sleep_until. The stale record must be cleared with it: left
    # behind it suppresses the state's notify (already_parked), feeds a stale
    # wake_epoch to a later --exit-on-wait run, and pins machine_is_parked.
    journal, f = _load(tmp_path, WAITER_DELAYED)
    spec = load_machine(f)
    drive(spec, journal, FakeWorld({}, clock=1000.0), live=True, exit_on_wait=True)
    assert journal.read_pending_wait() is not None
    result = drive(spec, journal, FakeWorld({}, wakes=[WaitWake("tick")]), live=True)
    assert result == MachineResult("ok", "ticked", "done", 1)
    assert journal.read_pending_wait() is None


def test_exit_on_wait_wake_epoch_computed_once(tmp_path: Path) -> None:
    journal, f = _load(tmp_path, WAITER_DELAYED)
    spec = load_machine(f)
    drive(spec, journal, FakeWorld({}, clock=1000.0), live=True, exit_on_wait=True)
    # A second not-ready invocation must NOT re-arm (would be 1030+60 = 1090).
    result = drive(spec, journal, FakeWorld({}, clock=1030.0), live=True, exit_on_wait=True)
    assert result.status == "waiting"
    pending = journal.read_pending_wait()
    assert pending is not None
    assert pending.wake_epoch == 1060.0


def test_journal_begins_once(tmp_path: Path) -> None:
    journal, f = _load(tmp_path, COUNTER)
    spec = load_machine(f)
    world = FakeWorld({"scan": _ok('{"items": []}')})
    drive(spec, journal, world, live=True)
    events = journal.read()
    assert sum(isinstance(e, MachineBegin) for e in events) == 1
    assert sum(isinstance(e, MachineEnd) for e in events) == 1


# --------------------------------------------------------------------------
# agent state (Phase 3)
# --------------------------------------------------------------------------


def _agent(reason: str, payload: dict[str, Any] | None) -> AgentExecResult:
    return AgentExecResult(reason=reason, payload=payload)


def test_agent_ok_captures_payload_and_routes(tmp_path: Path) -> None:
    journal, f = _load(tmp_path, REVIEWER)
    spec = load_machine(f)
    payload = {"approved": True, "note": "lgtm"}
    world = FakeWorld({}, agent_results=[_agent("finish_session", payload)])
    result = drive(spec, journal, world, live=True)
    assert result == MachineResult("ok", "approved", "stop_ok", 2)
    snap = journal.latest_snapshot()
    assert snap is not None
    assert snap.blackboard["verdict"] == payload
    # The rendered prompt was passed through to the runner.
    assert world.agent_calls[0].model == "claude-sonnet-4-5"
    assert world.agent_calls[0].prompt == "Review the change."


def test_agent_per_state_knobs_threaded_to_request(tmp_path: Path) -> None:
    body = REVIEWER.replace(
        'prompt = "Review the change."',
        'prompt = "Review the change."\n'
        'provider = "anthropic"\n'
        'effort = "high"\n'
        "temperature = 0.3\n"
        "max_usd = 2.5\n"
        "max_tokens_fallback = 90000",
    )
    journal, f = _load(tmp_path, body)
    spec = load_machine(f)
    world = FakeWorld({}, agent_results=[_agent("finish_session", {"approved": True})])
    drive(spec, journal, world, live=True)
    req = world.agent_calls[0]
    assert req.provider == "anthropic"
    assert req.effort == "high"
    assert req.temperature == 0.3
    # min(2.5 state cap, 1.0 machine budget): a state can never be handed
    # more than the machine has.
    assert req.max_usd == 1.0
    assert req.max_tokens_fallback == 90000


def test_agent_ok_but_rejected_routes_fail(tmp_path: Path) -> None:
    journal, f = _load(tmp_path, REVIEWER)
    spec = load_machine(f)
    payload = {"approved": False, "note": "needs work"}
    world = FakeWorld({}, agent_results=[_agent("finish_session", payload)])
    result = drive(spec, journal, world, live=True)
    # Valid payload (label ok) captured, then branch routes to stop_fail.
    assert result.status == "failed"
    assert result.state == "stop_fail"
    snap = journal.latest_snapshot()
    assert snap is not None
    assert snap.blackboard["verdict"] == payload


def test_agent_invalid_payload_routes_failed_no_capture(tmp_path: Path) -> None:
    journal, f = _load(tmp_path, REVIEWER)
    spec = load_machine(f)
    # Missing the required `note` field -> schema validation fails -> "failed".
    world = FakeWorld({}, agent_results=[_agent("finish_session", {"approved": True})])
    result = drive(spec, journal, world, live=True)
    assert result.status == "failed"
    assert result.state == "stop_fail"
    snap = journal.latest_snapshot()
    assert snap is not None
    # `verdict` keeps its declared default; nothing was captured.
    assert snap.blackboard["verdict"] == {}


def test_agent_finish_session_without_payload_routes_failed(tmp_path: Path) -> None:
    journal, f = _load(tmp_path, REVIEWER)
    spec = load_machine(f)
    world = FakeWorld({}, agent_results=[_agent("finish_session", None)])
    result = drive(spec, journal, world, live=True)
    assert result.status == "failed"
    assert result.state == "stop_fail"


def test_agent_budget_exhausted_label(tmp_path: Path) -> None:
    journal, f = _load(tmp_path, REVIEWER)
    spec = load_machine(f)
    world = FakeWorld({}, agent_results=[_agent("budget_exhausted", None)])
    result = drive(spec, journal, world, live=True)
    assert result.status == "failed"
    assert result.state == "halt"
    assert result.reason == "budget"


def test_agent_timeout_label(tmp_path: Path) -> None:
    journal, f = _load(tmp_path, REVIEWER)
    spec = load_machine(f)
    world = FakeWorld({}, agent_results=[_agent("timeout", None)])
    result = drive(spec, journal, world, live=True)
    assert result.status == "failed"
    assert result.state == "expired"
    assert result.reason == "timeout"


_SPENDER = """
machine = "spender"
version = 1
initial = "work"

[budget]
max_usd = 0.05
max_transitions = 100

[schemas.r]
ok = "bool"

[vars.agent]
out = { type = "r", default = {} }

[states.work]
kind = "agent"
model = "m"
prompt = "do"
output_schema = "r"
capture = { finish_json = "out" }
timeout_secs = 60
on = { ok = "work", failed = "stop_fail", budget_exhausted = "stop_fail", timeout = "stop_fail" }

[states.stop_fail]
kind = "terminal"
status = "failed"
reason = "fail"
"""


def test_machine_stops_when_cumulative_max_usd_exceeded(tmp_path: Path) -> None:
    # Each agent step costs $0.10 > the $0.05 machine budget and loops on ok.
    # The engine must stop on the budget guard, not run unbounded.
    journal, f = _load(tmp_path, _SPENDER)
    spec = load_machine(f)
    world = FakeWorld(
        {}, agent_results=[AgentExecResult(reason="finish_session", payload={"ok": True}, usd=0.10)]
    )
    result = drive(spec, journal, world, live=True)
    assert result.status == "failed"
    assert "max_usd" in result.reason
    assert len(world.agent_calls) == 1  # one step ran, then the budget guard fired


def test_the_agent_request_cap_is_clamped_to_the_remaining_machine_budget(
    tmp_path: Path,
) -> None:
    """The aggregate guard only stops the NEXT state; the request itself
    carried only the state's own override, so a $0.10 state cap against a
    $0.05 machine budget handed the child more than the machine had. The
    request's cap is min(state cap, remaining machine budget)."""
    body = _SPENDER.replace('prompt = "do"', 'prompt = "do"\nmax_usd = 0.10')
    journal, f = _load(tmp_path, body)
    spec = load_machine(f)
    world = FakeWorld(
        {},
        agent_results=[
            AgentExecResult(reason="finish_session", payload={"ok": True}, usd=0.04),
            AgentExecResult(reason="finish_session", payload={"ok": True}, usd=0.04),
        ],
    )
    result = drive(spec, journal, world, live=True)
    assert result.status == "failed" and "max_usd" in result.reason
    caps = [req.max_usd for req in world.agent_calls]
    assert caps[0] == 0.05  # min(0.10 state cap, 0.05 machine budget)
    assert caps[1] == pytest.approx(0.01)  # the machine's remaining cent, not 0.10


def test_agent_spend_threaded_into_fact(tmp_path: Path) -> None:
    journal, f = _load(tmp_path, REVIEWER)
    spec = load_machine(f)
    result = AgentExecResult(
        reason="finish_session",
        payload={"approved": True, "note": "ok"},
        usd=0.25,
        input_tokens=2000,
        output_tokens=300,
    )
    world = FakeWorld({}, agent_results=[result])
    drive(spec, journal, world, live=True)
    step = next(
        e for e in journal.read() if isinstance(e, StepEvent) and isinstance(e.fact, AgentFact)
    )
    assert isinstance(step.fact, AgentFact)
    assert step.fact.usd == 0.25
    assert step.fact.input_tokens == 2000
    assert step.fact.output_tokens == 300


def test_agent_set_capture_extracts_scalar_field(tmp_path: Path) -> None:
    journal, f = _load(tmp_path, SCORER)
    spec = load_machine(f)
    world = FakeWorld({}, agent_results=[_agent("finish_session", {"points": 7})])
    result = drive(spec, journal, world, live=True)
    assert result == MachineResult("ok", "done", "stop_ok", 1)
    snap = journal.latest_snapshot()
    assert snap is not None
    assert snap.blackboard["total"] == 7


def test_agent_replay_reproduces_path_without_world(tmp_path: Path) -> None:
    journal, f = _load(tmp_path, REVIEWER)
    spec = load_machine(f)
    payload = {"approved": True, "note": "ok"}
    world = FakeWorld({}, agent_results=[_agent("finish_session", payload)])
    live = drive(spec, journal, world, live=True)
    replayed = drive(spec, journal, None, live=False)
    assert replayed == live


def test_agent_crash_recovery_does_not_rerun(tmp_path: Path) -> None:
    journal, f = _load(tmp_path, REVIEWER)
    spec = load_machine(f)
    # Crash right after the agent ran: journal has begin + the agent step, no
    # terminal. Recovery must rebuild `verdict` and continue from `route`
    # without calling the runner again (empty agent_results would IndexError).
    journal.ensure_dirs()
    journal.begin(machine="reviewer", version=1)
    journal.append(
        StepEvent(
            ts="t",
            seq=0,
            state="review",
            label="ok",
            goto="route",
            fact=AgentFact(
                outcome="ok",
                reason="finish_session",
                payload={"approved": True, "note": "lgtm"},
            ),
        )
    )
    world = FakeWorld({})  # no programmed agent results — proving it is not re-run
    result = drive(spec, journal, world, live=True)
    assert result == MachineResult("ok", "approved", "stop_ok", 2)
    assert world.agent_calls == []


def test_agent_without_runner_raises(tmp_path: Path) -> None:
    from agent6.machine.engine import EngineError, LiveWorld

    journal, f = _load(tmp_path, REVIEWER)
    spec = load_machine(f)
    journal.ensure_dirs()
    world = LiveWorld(cwd=tmp_path, journal=journal, agent_runner=None)
    with pytest.raises(EngineError, match="no agent runner"):
        drive(spec, journal, world, live=True)


def test_per_state_agent_log_path_and_prune(tmp_path: Path) -> None:
    """Each agent-state execution gets its own watchable logs.jsonl at
    <state_log_root>/<seq>-<state>/, and the dirs are pruned to the most recent
    state_log_keep so a long-running machine's logs stay bounded."""
    from agent6.machine.engine import LiveWorld

    journal = MachineJournal(tmp_path / "inst")
    states_root = tmp_path / "inst" / "states"
    captured: list[Path | None] = []

    def fake_runner(req: AgentRequest, events_log: Path | None) -> AgentExecResult:
        captured.append(events_log)
        if events_log is not None:  # the real subprocess creates the log; simulate it
            events_log.parent.mkdir(parents=True, exist_ok=True)
            events_log.write_text("{}\n", encoding="utf-8")
        return AgentExecResult(reason="finish_session", payload=None)

    world = LiveWorld(
        cwd=tmp_path,
        journal=journal,
        agent_runner=fake_runner,
        state_log_root=states_root,
        state_log_keep=3,
    )
    for seq in range(5):
        world.run_agent(AgentRequest(prompt="x", timeout_s=1.0, state_name="s", step_seq=seq))

    # Each call got its own seq-named log path.
    assert captured[0] == states_root / "0000-s" / "logs.jsonl"
    assert captured[4] == states_root / "0004-s" / "logs.jsonl"
    # Pruned to keep=3: only the three most recent state dirs survive on disk.
    assert sorted(p.name for p in states_root.iterdir()) == ["0002-s", "0003-s", "0004-s"]


def test_state_log_keep_zero_disables_pruning(tmp_path: Path) -> None:
    """`[machine].state_log_keep = 0` keeps every per-state log dir, like
    `snapshot_keep`; retention is an audited config value, not a hidden
    hardcoded deletion."""
    from agent6.machine.engine import LiveWorld

    journal = MachineJournal(tmp_path / "inst")
    states_root = tmp_path / "inst" / "states"

    def fake_runner(req: AgentRequest, events_log: Path | None) -> AgentExecResult:
        if events_log is not None:
            events_log.parent.mkdir(parents=True, exist_ok=True)
            events_log.write_text("{}\n", encoding="utf-8")
        return AgentExecResult(reason="finish_session", payload=None)

    world = LiveWorld(
        cwd=tmp_path,
        journal=journal,
        agent_runner=fake_runner,
        state_log_root=states_root,
        state_log_keep=0,
    )
    for seq in range(5):
        world.run_agent(AgentRequest(prompt="x", timeout_s=1.0, state_name="s", step_seq=seq))
    assert len(list(states_root.iterdir())) == 5


def test_per_state_log_disabled_without_root(tmp_path: Path) -> None:
    """No state_log_root -> no per-state log (the create authoring agent and any
    runner that doesn't want logs get None)."""
    from agent6.machine.engine import LiveWorld

    seen: list[Path | None] = []

    def fake_runner(req: AgentRequest, events_log: Path | None) -> AgentExecResult:
        seen.append(events_log)
        return AgentExecResult(reason="finish_session", payload=None)

    world = LiveWorld(
        cwd=tmp_path, journal=MachineJournal(tmp_path / "i"), agent_runner=fake_runner
    )
    world.run_agent(AgentRequest(prompt="x", timeout_s=1.0, state_name="s", step_seq=0))
    assert seen == [None]


def test_best_effort_usd_limit_no_longer_validates(tmp_path: Path) -> None:
    # The hard/soft pair collapsed to one max_usd (unmetered spend is bounded
    # by [budget].max_tokens_fallback in the effective config); the old field
    # must fail the grammar loudly, never load as an ignored knob.
    from agent6.machine import MachineError

    body = _SPENDER.replace("max_usd = 0.05", "best_effort_usd_limit = 0.05")
    f = tmp_path / "m.asm.toml"
    f.write_text(body, encoding="utf-8")
    with pytest.raises(MachineError, match="best_effort_usd_limit"):
        load_machine(f)


def test_agent_state_max_tokens_fallback_flows_to_request(tmp_path: Path) -> None:
    body = _SPENDER.replace('kind = "agent"', 'kind = "agent"\nmax_tokens_fallback = 5000', 1)
    journal, f = _load(tmp_path, body)
    spec = load_machine(f)
    world = FakeWorld(
        {}, agent_results=[AgentExecResult(reason="finish_session", payload={"ok": True}, usd=0.10)]
    )
    drive(spec, journal, world, live=True)
    assert world.agent_calls[0].max_tokens_fallback == 5000


NOTIFY_WAIT = """
machine = "notifywait"
version = 1
initial = "park"

[budget]
max_usd = 1.0
max_transitions = 100

[states.park]
kind = "wait"
notify = { message = "machine parked, poke me", level = "info" }
on = { signal = "done" }

[states.done]
kind = "terminal"
status = "ok"
reason = "finished"
"""


def test_parked_wait_notify_fires_once_across_scheduler_ticks(tmp_path: Path) -> None:
    # Re-driving a parked --exit-on-wait machine (a cron/systemd tick) must not
    # re-fire the wait's notify (or the operator hook: a page, an email) once
    # per poll; the notify belongs to state ENTRY, and an armed PendingWait
    # means the state was already entered.
    from agent6.machine.journal import MachineNotify

    journal, f = _load(tmp_path, NOTIFY_WAIT)
    spec = load_machine(f)
    hook_notifies = 0
    for _ in range(3):
        world = FakeWorld({})
        result = drive(spec, journal, world, live=True, exit_on_wait=True)
        assert result.status == "waiting"
        hook_notifies += sum(1 for n in world.notifications if n[0] == "notify")
    assert hook_notifies == 1
    assert sum(1 for e in journal.read() if isinstance(e, MachineNotify)) == 1
    # The firing tick (poke consumed) adds no duplicate either.
    journal.poke(None)
    result = drive(spec, journal, FakeWorld({}), live=True, exit_on_wait=True)
    assert result.status == "ok"
    assert sum(1 for e in journal.read() if isinstance(e, MachineNotify)) == 1


def test_poke_atomic_write_leaves_no_temp_and_keeps_payload(tmp_path: Path) -> None:
    # poke() must publish the signal file atomically (temp + rename): the
    # engine's take_signal polls from another process, and a plain write let
    # it consume an empty/partial file and drop the payload.
    journal, _ = _load(tmp_path, FOREVER)
    journal.poke({"cmd": "deploy", "target": "prod"})
    assert not any(p.name.endswith(".tmp") for p in journal.root.iterdir())
    present, payload = journal.take_signal()
    assert present is True
    assert payload == {"cmd": "deploy", "target": "prod"}


def test_machine_is_parked_reflects_pending_wait(tmp_path: Path) -> None:
    from agent6.viewmodel import machine_is_parked

    journal, f = _load(tmp_path, FOREVER)
    spec = load_machine(f)
    assert machine_is_parked(journal.root) is False
    result = drive(spec, journal, FakeWorld({}), live=True, exit_on_wait=True)
    assert result.status == "waiting"
    assert machine_is_parked(journal.root) is True


def test_live_world_run_tool_uses_the_shared_jail_tool_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A tool-state jail resolves operator tools EXACTLY like run_command's jail
    # and the machine-check probe: the computed PATH string plus the RO+exec
    # mounts that make it true (sandbox._tool_paths.operator_tool_paths). Copying the
    # host PATH named dirs the jail never mounts, so a tool `machine check`
    # proved reachable still died 127 on the machine's first real transition.
    from agent6.machine import engine as engine_mod
    from agent6.machine.engine import LiveWorld
    from agent6.machine.journal import MachineJournal
    from agent6.types import CommandResult, JailPolicy

    captured: dict[str, JailPolicy] = {}

    def fake_run_in_jail(policy: JailPolicy) -> CommandResult:
        captured["policy"] = policy
        return CommandResult(argv=policy.argv, returncode=0, stdout="{}", stderr="", duration_s=0.0)

    def fake_tool_paths() -> tuple[str, tuple[Path, ...]]:
        return "/usr/bin:/bin:/fake/bin", (Path("/fake/real-tools"),)

    monkeypatch.setattr(engine_mod, "run_in_jail", fake_run_in_jail)
    monkeypatch.setattr("agent6.tools.policy.operator_tool_paths", fake_tool_paths)
    monkeypatch.setenv("PATH", "/host/only/path")  # must NOT leak into the jail
    world = LiveWorld(
        cwd=tmp_path,
        journal=MachineJournal(tmp_path),
        tool_policy=machine_tool_policy_factory(
            Config(), tmp_path, "strict", protect_paths=(), data_dir=None
        ),
    )
    world.run_tool(("sometool", "arg"), timeout_s=5.0)

    policy = captured["policy"]
    env = dict(policy.env)
    assert env["PATH"] == "/usr/bin:/bin:/fake/bin"  # the jail-correct PATH
    assert policy.tool_paths == (Path("/fake/real-tools"),)  # with its mounts
    assert "/host/only/path" not in env.values()
    assert env["UV_NO_SYNC"] == "1"  # same offline-jail rule as run_command


def test_a_captured_lone_surrogate_never_reaches_the_blackboard(tmp_path: Path) -> None:
    """The journal writers survive a lone surrogate, but the value stayed raw on
    the blackboard, so the poison just moved one step downstream: the next agent
    state serializes the blackboard into its request payload and
    model_dump_json raises PydanticSerializationError -- a ValueError, caught by
    neither the engine's _STATE_RUNTIME_ERRORS nor run_machine's handler. Same
    traceback with no MachineEnd the sanitizing commit set out to remove."""
    from pydantic import BaseModel

    from agent6.machine.engine import _apply_capture  # pyright: ignore[reportPrivateUsage]
    from agent6.machine.model import ToolState

    class _Payload(BaseModel):  # the agent request's shape, minimally
        task: str

    _journal, f = _load(tmp_path, COUNTER)
    spec = load_machine(f)
    state = spec.states["scan"]
    assert isinstance(state, ToolState)
    blackboard: dict[str, Any] = {}
    # The journal sanitizes on write, so only the IN-MEMORY value matters here.
    _apply_capture(spec, state, '{"items": ["emoji tail \\ud83d"]}', blackboard)

    captured = blackboard["items"][0]
    assert "emoji tail" in captured  # kept, not dropped
    _Payload(task=captured).model_dump_json()  # the sink that raised


def test_replay_rejects_a_diverged_state_seq_or_fact_kind(tmp_path: Path) -> None:
    """The fold trusted every recorded field: a step whose state or seq
    disagreed with the replayed position folded silently, and a fact kind the
    state cannot produce no-op'd through reduce -- a fabricated journal
    rebuilt a position and blackboard the machine never reached."""
    base = [
        dict(ts="t", seq=0, state="scan", label="ok", goto="check"),
    ]
    counter = iter(range(10))

    def journal_with(**overrides: object) -> tuple[Any, Any]:
        case_dir = tmp_path / f"case{next(counter)}"
        case_dir.mkdir()
        journal, f = _load(case_dir, COUNTER)
        journal.ensure_dirs()
        journal.begin(machine="counter", version=1)
        fields: dict[str, Any] = {
            **base[0],
            "fact": ToolFact(exit_code=0, stdout='{"items": []}', timed_out=False),
        }
        fields.update(overrides)
        journal.append(StepEvent(**fields))
        return journal, load_machine(f)

    journal, spec = journal_with(state="record")  # not the replayed position
    with pytest.raises(EngineError, match="diverges"):
        drive(spec, journal, FakeWorld({}), live=True)

    journal, spec = journal_with(seq=41)  # noncontiguous
    with pytest.raises(EngineError, match="not contiguous"):
        drive(spec, journal, FakeWorld({}), live=True)

    journal, spec = journal_with(fact=BranchFact(clause_index=0))  # wrong kind for a tool state
    with pytest.raises(EngineError, match="cannot produce"):
        drive(spec, journal, FakeWorld({}), live=True)


def test_stop_request_parks_at_the_transition_boundary(tmp_path: Path) -> None:
    """A stop marker written mid-state parks the machine at the next boundary:
    the finished state's fact is journaled, no MachineEnd is written, the
    marker is consumed, and a later drive continues to the terminal."""
    from agent6.machine.journal import stop_requested, write_stop_request

    journal, f = _load(tmp_path, COUNTER)
    spec = load_machine(f)
    world = FakeWorld({"scan": _ok('{"items": ["a"]}'), "record": _ok()})
    real_run_tool = world.run_tool

    def stop_during_first_tool(
        argv: tuple[str, ...], timeout_s: float, *, network: NetworkMode = "none"
    ) -> ToolExecResult:
        write_stop_request(journal.root)
        return real_run_tool(argv, timeout_s, network=network)

    world.run_tool = stop_during_first_tool  # type: ignore[method-assign]
    result = drive(spec, journal, world, live=True)
    assert result.status == "stopped"
    assert not stop_requested(journal.root)  # consumed at the park
    events = journal.read()
    assert not isinstance(events[-1], MachineEnd)  # resumable, no end journaled
    done = drive(
        spec, journal, FakeWorld({"scan": _ok('{"items": ["a"]}'), "record": _ok()}), live=True
    )
    assert done.status == "ok"


def test_stop_interrupts_a_foreground_wait_and_keeps_it_armed(tmp_path: Path) -> None:
    """A stop that lands mid-sleep parks without consuming the wait: the
    PendingWait survives, so a later run resumes the same wake instant."""
    journal, f = _load(tmp_path, WAITER)
    spec = load_machine(f)
    world = FakeWorld({}, wakes=[WaitWake("stop")])
    result = drive(spec, journal, world, live=True)
    assert result.status == "stopped"
    pending = journal.read_pending_wait()
    assert pending is not None and pending.state == "poll"


def test_live_world_sleep_wakes_on_a_stop_request(tmp_path: Path) -> None:
    """The stop marker interrupts a real sleep (a machine parked in an
    hour-long or forever wait must not sleep through its own stop)."""
    import time as _time

    from agent6.machine.engine import LiveWorld
    from agent6.machine.journal import write_stop_request

    journal = MachineJournal(tmp_path / "inst")
    journal.ensure_dirs()
    world = LiveWorld(cwd=tmp_path, journal=journal, poll_interval_s=0.01)
    write_stop_request(journal.root)
    assert world.sleep_until(_time.time() + 3600).woke_by == "stop"
