# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""agent6 state machines, declarative, replayable mini-agents.

Load + validate a `.asm.toml` (`_semantics`, shapes in `model`), render it as
a diagram (`graph`), and
execute it deterministically (`engine`) over an append-only `journal` with
crash recovery and offline replay. The `agent` state kind runs a normal agent6
loop through an injected runner; `machine status`/`poke` and the
external-scheduler `--exit-on-wait` mode (persist the next wake instead of
blocking) cover 24/7 use.
"""

from __future__ import annotations

from agent6.machine._semantics import (
    fixture_problems,
    load_machine,
    validate_record_payload,
    validate_semantics,
)
from agent6.machine.authoring import (
    MACHINE_AUTHOR_GUIDE,
    build_authoring_prompt,
)
from agent6.machine.dryrun import BranchCheck, DryRunReport, StateCheck, dry_run
from agent6.machine.engine import (
    AgentExecResult,
    AgentRequest,
    EngineError,
    LiveWorld,
    MachineResult,
    ToolExecResult,
    ToolPolicyFactory,
    WaitWake,
    World,
    drive,
)
from agent6.machine.graph import render_dot, render_mermaid
from agent6.machine.journal import (
    AgentFact,
    AttemptSpend,
    JournalError,
    MachineBegin,
    MachineEnd,
    MachineJournal,
    MachineNotify,
    PendingWait,
    Snapshot,
    StepEvent,
    WaitFact,
    bundle_drift,
    clear_stop_request,
    machine_lock,
    read_source,
    stop_requested,
    write_bundle,
    write_source,
    write_stop_request,
)
from agent6.machine.model import (
    PROTECTED_OVERLAY_LEAVES,
    PROTECTED_OVERLAY_TABLES,
    AgentState,
    FieldSpec,
    MachineError,
    MachineSpec,
    StateSpec,
    ToolState,
)

__all__ = [
    "MACHINE_AUTHOR_GUIDE",
    "PROTECTED_OVERLAY_LEAVES",
    "PROTECTED_OVERLAY_TABLES",
    "AgentExecResult",
    "AgentFact",
    "AgentRequest",
    "AgentState",
    "AttemptSpend",
    "BranchCheck",
    "DryRunReport",
    "EngineError",
    "FieldSpec",
    "JournalError",
    "LiveWorld",
    "MachineBegin",
    "MachineEnd",
    "MachineError",
    "MachineJournal",
    "MachineNotify",
    "MachineResult",
    "MachineSpec",
    "PendingWait",
    "Snapshot",
    "StateCheck",
    "StateSpec",
    "StepEvent",
    "ToolExecResult",
    "ToolPolicyFactory",
    "ToolState",
    "WaitFact",
    "WaitWake",
    "World",
    "build_authoring_prompt",
    "bundle_drift",
    "clear_stop_request",
    "drive",
    "dry_run",
    "fixture_problems",
    "load_machine",
    "machine_lock",
    "read_source",
    "render_dot",
    "render_mermaid",
    "stop_requested",
    "validate_record_payload",
    "validate_semantics",
    "write_bundle",
    "write_source",
    "write_stop_request",
]
