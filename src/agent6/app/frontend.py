# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The presentation seam a front-end injects into the run/resume lifecycle
(`SessionFrontend` + its capability and steer contracts), and the away-mode
policy applied when a launcher spawns a run detached."""

from __future__ import annotations

import os
from collections.abc import Callable, Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from agent6.budget import BudgetTracker
from agent6.config import Config
from agent6.events import EventSink
from agent6.sessions.ipc import (
    AWAY_MODES,
    COMMAND_SCOPE,
    MCP_SCOPE_PREFIX,
    away_mode,
    set_away_mode,
    set_session_allow,
)
from agent6.sessions.layout import SessionLayout
from agent6.tools.mcp_client import MCPManager
from agent6.tools.operator_prompts import Approver, Questioner
from agent6.types import AutoCommitDirective, IsolationLevel
from agent6.workflows.loop import SessionResult, Workflow
from agent6.workflows.subrun import GroupLaneSpawner


@dataclass(frozen=True, slots=True)
class SessionFacts:
    """The live facts the CLI pause banner shows, so an operator deciding
    whether to interrupt can see what this run is doing without the widgets a
    TUI/web viewer has. Built by the lifecycle (which holds the tracker and the
    resolved config) and rendered by the front-end; read inside a signal
    handler, so every field is already in memory -- no file read, no fold."""

    spend_usd: float
    spend_partial: bool  # a model with no price data contributed: a lower bound
    model: str
    run_commands: str
    isolation: str


class SteerHooks(Protocol):
    """What the lifecycle needs of the front-end's steer state (the SIGINT
    pause menu or the file-bridge steer); `ui/cli/_steer.SteerState` satisfies
    it structurally."""

    requested: Callable[[], bool]
    clear: Callable[[], None]
    prompt: Callable[[], str | None]
    restore: Callable[[], None]
    abort_pending: Callable[[], bool]
    interrupt: Callable[[], bool]
    reset_stage: Callable[[], None]


def approval_scopes(cfg: Config) -> tuple[str, ...]:
    """Every scope this run can be asked about: the command tools, plus one per
    live MCP server. "Approve everything while I am away" has to name them --
    a grant is per scope, so a run left with only the command scope granted
    would still block on the first MCP call with nobody there to answer."""
    servers = (
        tuple(f"{MCP_SCOPE_PREFIX}{name}" for name, s in cfg.mcp.servers.items() if s.enabled)
        if cfg.mcp.enabled
        else ()
    )
    return (COMMAND_SCOPE, *servers)


def apply_spawned_away_default(session_dir: Path, scopes: tuple[str, ...]) -> None:
    """Honor AGENT6_DETACHED_AWAY, set by a front-end launcher (web/TUI hub) that
    spawns a run detached and drives it over the bridge. Without it a spawned run
    with no terminal fabricates empty ask_user answers when no viewer is live;
    'wait' makes approvals and questions block for a front-end. A pure headless
    run (no launcher) sets no env, so this is a no-op and it keeps its default.

    A DEFAULT: an away mode already on the run dir is the operator's own detach
    answer, and the resume this spawns carries 'wait' regardless -- overwriting
    silently upgraded a chosen 'deny' to 'wait', so the run blocked on an
    approval nobody was there to give instead of denying and carrying on."""
    away = os.environ.get("AGENT6_DETACHED_AWAY", "")
    if not away or away_mode(session_dir):
        return
    if away == "approve":
        # approve is never stored in away.mode (deny|wait): like the interactive
        # detach prompt, approve-all sets an allow marker per scope in play.
        for scope in scopes:
            set_session_allow(session_dir, scope)
    elif away in AWAY_MODES:
        set_away_mode(session_dir, away)


@dataclass(frozen=True, slots=True)
class FrontendCapabilities:
    """What this surface can actually do, declared once at wiring.

    A surface declares what it can do, so a headless run with no away-mode
    denies rather than fabricating an empty `ask_user` answer.
    """

    # Approvals and ask_user reach a human. False for a headless run with no
    # away-mode -- which is exactly what `headless_approval_refusal` computes.
    can_ask: bool = True


@dataclass(frozen=True, slots=True)
class SessionFrontend:
    """The presentation + process-spawn callables `ui/cli` injects into the
    run/resume lifecycle: the live console view (held cli-side; the lifecycle
    only signals attach/close), the interactive prompts, and the REPLs. The
    lifecycle owns the run-dir bridge (`sessions.ipc`); only the exe-spawn
    primitives it can't reach stay injected.
    One value serves both `run_task` and `resume_task`; resume simply never
    calls the run-only fields."""

    # What this surface can do at all. Read before offering something, rather
    # than discovered by trying it.
    capabilities: FrontendCapabilities
    # live view: the console-view instance lives cli-side; builders that need it
    # (approver/questioner/steer/logger) close over it there.
    should_spawn_tui: Callable[[bool, bool, str], bool]
    stream_modes: Callable[[bool], tuple[bool, bool]]
    attach_console_view: Callable[[EventSink], None]
    close_console_view: Callable[[], None]
    loop_logger: Callable[[str], Callable[[str], None]]
    tui_session: Callable[[Path, bool], AbstractContextManager[None]]
    # operator interaction: the callables that ANSWER a prompt the gate
    # (`tools.operator_prompts`) has journaled, keyed on the run dir's bridge.
    build_approver: Callable[[Path], Approver]
    build_questioner: Callable[[Path], Questioner]
    make_steer_state: Callable[[EventSink, Path, Callable[[], SessionFacts]], SteerHooks]
    confirm_unconfined_autorun: Callable[[IsolationLevel, Config], bool]
    confirm_run_on_run_branch: Callable[[str], bool]
    # Resume found a mid-turn-crash marker matching the turn about to re-run:
    # its tools may have partially applied. (iteration, tool names) -> replay?
    # Interactive fronts prompt (default no); headless warns and proceeds.
    confirm_replay_after_crash: Callable[[int, tuple[str, ...]], bool]
    prompt_detach_away_mode: Callable[[Path, tuple[str, ...]], None]
    select_revised_prompt: Callable[[str, str, tuple[str, ...]], str | None]
    # `run -i` / `ask -i`
    build_repl_hook: Callable[
        [Path, BudgetTracker, str, MCPManager | None],
        Callable[[int, str], AutoCommitDirective],
    ]
    run_ask_repl: Callable[[Workflow, BudgetTracker, SessionLayout, str], SessionResult]
    save_ask_transcript: Callable[[SessionLayout, str, str], None]
    # `/parallel` coordinator dispatch (the cli builds LaneRuntime + spawner).
    build_coordinator_spawner: Callable[
        [Config, Path, Path, str, str, float | None, bool],
        GroupLaneSpawner | None,
    ]
    # process-spawn primitives the front-end owns (`ui.spawn`, mirroring
    # LaneRuntime's injected spawner).
    agent6_exe: Callable[[], str]
    # (cwd, session_id, flags): the flags are this invocation's overrides as
    # CLI options, so the detached leg runs under them (see
    # `_setup.override_flags`).
    spawn_detached_resume: Callable[[Path, str, Sequence[str]], str]
