# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Machine command lifecycle: the engine composition behind `agent6 machine
run`/`create` and the preflight/validation helpers the read-only CLI commands
share.

`ui/cli/machine_cmds.py` keeps only argv adaptation + console rendering; the
composition lives here, behind the `MachineFrontend` seam (mirroring
`app.run.SessionFrontend`). The machine ENGINE itself (`agent6.machine`) is
unchanged; this package composes it, resolves the sandbox/egress/budget
preflight, and spawns the per-`agent`-state runner.
"""

from __future__ import annotations

from agent6.app.machine._bundle import (
    MachineFileSummary,
    summarize_machine_file,
    validate_bundle,
)
from agent6.app.machine._frontend import MachineFrontend
from agent6.app.machine._preflight import (
    build_machine_notify_hook,
    machine_network_refusal,
    machine_pass_env_refusal,
    machine_protect_paths,
)
from agent6.app.machine._scriptcheck import (
    OfflineTestOutcome,
    available_tools,
    lint_and_typecheck,
    run_offline_tests,
)
from agent6.app.machine._spend import book_crashed_attempt
from agent6.viewmodel.machine_state import Spend, machine_spend, read_budget_totals

# The composition entry points `create_machine` / `run_machine` are the public
# modules `agent6.app.machine.create` / `.run` (mirroring `agent6.app.run`),
# imported directly by `ui/cli`, NOT re-exported here: both pull in
# `app.machine_agent`, which imports this package's `_spend` submodule, so
# re-exporting them from this `__init__` would loop back through it.
__all__ = [
    "MachineFileSummary",
    "MachineFrontend",
    "OfflineTestOutcome",
    "Spend",
    "available_tools",
    "book_crashed_attempt",
    "build_machine_notify_hook",
    "lint_and_typecheck",
    "machine_network_refusal",
    "machine_pass_env_refusal",
    "machine_protect_paths",
    "machine_spend",
    "read_budget_totals",
    "run_offline_tests",
    "summarize_machine_file",
    "validate_bundle",
]
