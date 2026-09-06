# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The presentation seam `ui/cli` injects into machine run/create."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agent6.app.machine._preflight import NetworkRefusal
from agent6.app.reporter import Reporter
from agent6.config import Config
from agent6.machine import ToolState
from agent6.types import IsolationLevel

# A hard tool-network refusal is resolved interactively: explain it, then offer
# to apply the minimal config fix and continue, simulate the machine offline, or
# stop. Returns the new (cfg, isolation) when a fix applied and re-validated clear,
# else an exit code. Held cli-side because it needs a TTY (and the offline
# `machine test` escape hatch); the lifecycle only calls it.
ResolveNetworkFix = Callable[
    [Path, NetworkRefusal, Config, IsolationLevel, list[ToolState], Path, dict[str, Any]],
    "int | tuple[Config, IsolationLevel]",
]


@dataclass(frozen=True, slots=True)
class MachineFrontend:
    """The presentation callables `app.machine` run/create drive, injected by
    `ui/cli`. Mirrors `app.run.SessionFrontend` but far thinner: machines are
    headless-first, so it is a two-channel `Reporter` for status output plus one
    interactive `resolve_network_fix` callback. `create_machine` uses only
    `reporter` (as `resume_task` never calls SessionFrontend's run-only fields)."""

    reporter: Reporter
    resolve_network_fix: ResolveNetworkFix
