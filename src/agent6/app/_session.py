# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""One owner for the session assembly `run_task` and `resume_task` share: the
isolation preflight and the provider/dispatcher/tools build. The lifecycles
keep their own workspace steps (branch cut + manifest vs snapshot guards) and
their Workflow wiring."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import agent6
from agent6.app._setup import budget_tracker, detect_env
from agent6.app.confine import (
    check_network_support,
    config_refusal,
    warn_cleartext_credential_endpoints,
    warn_sandbox_gaps,
)
from agent6.app.frontend import SessionFacts
from agent6.app.preflight import (
    SessionRefused,
    budget_preflight,
    warn_if_prompt_override_incomplete,
)
from agent6.app.providers import (
    InstrumentedProvider,
    build_review_seats,
    build_role_provider,
    close_provider,
    resolve_compaction_thresholds,
    resolve_decompose,
    reviewer_seat_provider,
)
from agent6.app.reporter import STDIO_REPORTER, Reporter
from agent6.budget import BudgetTracker
from agent6.config import ClaudeCodeProviderEntry, Config, RoleModel, RoleName
from agent6.events import EventSink
from agent6.graph.curator import GraphCurator
from agent6.providers import Provider, TranscriptSink
from agent6.sandbox.detect import Environment, IsolationUnavailableError, resolve_isolation
from agent6.sandbox.jail import JailUnavailableError, SessionNetwork
from agent6.sessions.layout import SessionLayout
from agent6.tools.dispatch import ToolDispatcher
from agent6.tools.mcp_client import MCPManager
from agent6.tools.operator_prompts import OperatorPrompts
from agent6.types import IsolationLevel, ResumableMode
from agent6.workflows._compaction import CLAUDE_CODE_RESULT_CAP_BYTES, TOOL_RESULT_CAP_BYTES
from agent6.workflows.review import ReviewSeat


def resolve_isolation_or_refuse(
    cfg: Config, env: Environment, *, reporter: Reporter
) -> IsolationLevel:
    """The isolation level *cfg* resolves to on this host, or a REFUSING line
    and :class:`SessionRefused` when an explicit level is unavailable here (an
    `auto` degrades inside `resolve_isolation`)."""
    try:
        return resolve_isolation(cfg.sandbox.isolation, env)
    except IsolationUnavailableError as exc:
        reporter.refuse(str(exc))
        raise SessionRefused(2) from exc


def tool_result_cap_bytes(cfg: Config, role: RoleName) -> int:
    """The size bound, in bytes of UTF-8, for one tool result entering the
    conversation of the provider driving *role*: the loop's default, or the
    tighter bound of a provider that hands the model less (Claude Code
    persists a result above its threshold and serves a preview)."""
    rm = cfg.models.resolve(role)
    entry = cfg.providers.get(rm.provider) if rm is not None else None
    if isinstance(entry, ClaudeCodeProviderEntry):
        return CLAUDE_CODE_RESULT_CAP_BYTES
    return TOOL_RESULT_CAP_BYTES


def select_isolation(
    cfg: Config,
    *,
    cwd: Path,
    confirm_unconfined: Callable[[IsolationLevel, Config], bool],
    reporter: Reporter,
    explicit_leaves: frozenset[str] = frozenset(),
    worktree_git_dir: Path | None = None,
) -> IsolationLevel:
    """The isolation preflight: pick the sandbox isolation for this environment,
    confirm an unconfined autorun, and refuse configs the isolation cannot honor
    (network mode, strict egress, budget) or a workspace no tool could read.
    Raises :class:`SessionRefused`."""
    try:
        env = detect_env()
    except JailUnavailableError as exc:
        # The strict probe could not run the jail binary itself: no isolation
        # can be selected over a binary no command will run.
        reporter.refuse(str(exc))
        raise SessionRefused(2) from exc
    selected = resolve_isolation_or_refuse(cfg, env, reporter=reporter)
    try:
        warn_sandbox_gaps(
            selected, env, cfg, root=cwd, worktree_git_dir=worktree_git_dir, reporter=reporter
        )
    except JailUnavailableError as exc:
        # The hardened exposure scan builds the run's policy, which creates the
        # jail's HOME and refuses one it cannot make.
        reporter.refuse(str(exc))
        raise SessionRefused(2) from exc
    warn_cleartext_credential_endpoints(cfg, reporter=reporter)
    if not confirm_unconfined(selected, cfg):
        reporter.note("aborted.")
        raise SessionRefused(1)
    net_err = check_network_support(cfg, selected)
    if net_err is not None:
        reporter.refuse(net_err)
        raise SessionRefused(2)
    # The shared list (`config_refusal`): a default this host cannot honour
    # degraded with a warning above; a value the operator wrote down refuses.
    cfg_err = config_refusal(
        cfg, selected, cwd, explicit_leaves=explicit_leaves, worktree_git_dir=worktree_git_dir
    )
    if cfg_err is not None:
        reporter.refuse(cfg_err)
        raise SessionRefused(2)
    budget_err = budget_preflight(cfg, reporter=reporter)
    if budget_err is not None:
        reporter.refuse(budget_err)
        raise SessionRefused(2)
    return selected


def install_inside_workspace(cwd: Path) -> Path | None:
    """agent6's own install root when it sits inside *cwd*, else None.

    An in-tree install (pip into the project's venv) is inside the jail's
    writable workspace, so a jailed command can rewrite the running agent.
    """
    root = Path(agent6.__file__).resolve().parent
    return root if root.is_relative_to(cwd.resolve()) else None


def warn_install_inside_workspace(cwd: Path, *, reporter: Reporter) -> None:
    """Warn when agent6 is installed inside the workspace a jailed command can
    write (never refuse it: agent6 developing agent6 is exactly that shape)."""
    if (root := install_inside_workspace(cwd)) is not None:
        reporter.warn(
            f"agent6 is installed inside this workspace ({root});"
            " a jailed command can rewrite the running agent. Install it outside"
            " the project (pipx / uv tool)."
        )


@dataclass(frozen=True, slots=True)
class SessionProviders:
    """The per-run provider battery: the driving role's instrumented provider
    plus the summariser and review seats, all metering into one tracker."""

    budget: BudgetTracker
    rm_role: RoleModel
    provider: Provider
    summariser_provider: Provider | None
    review_seats: list[ReviewSeat]

    def close(self) -> None:
        """Release every provider's held process (a `claude_code` session)."""
        close_provider(self.provider)
        if self.summariser_provider is not None:
            close_provider(self.summariser_provider)
        for seat in self.review_seats:
            close_provider(seat.provider)


def build_session_providers(
    cfg: Config,
    *,
    role: RoleName,
    events: EventSink,
    transcript_sink: TranscriptSink,
    stream_text: bool,
    reporter: Reporter = STDIO_REPORTER,
) -> SessionProviders:
    budget = budget_tracker(cfg)
    inner = build_role_provider(cfg, role, transcript_sink=transcript_sink, budget=budget)
    rm_role = cfg.models.resolve(role)
    assert rm_role is not None  # require_runnable validated this
    warn_if_prompt_override_incomplete(cfg, reporter=reporter)
    provider: Provider = InstrumentedProvider(
        inner=inner,
        role=role,
        model=rm_role.model,
        provider_name=rm_role.provider,
        events=events,
        budget=budget,
        stream_text=stream_text,
    )
    summariser_provider = reviewer_seat_provider(
        cfg, "summariser", transcript_sink=transcript_sink, budget=budget, events=events
    )
    # The panel is THE in-loop review: trigger on with no seats builds the
    # simple one-seat roster on the reviewer model.
    review_seats = (
        build_review_seats(cfg, transcript_sink=transcript_sink, budget=budget, n=1, events=events)
        if cfg.review.trigger != "off"
        else []
    )
    return SessionProviders(
        budget=budget,
        rm_role=rm_role,
        provider=provider,
        summariser_provider=summariser_provider,
        review_seats=review_seats,
    )


@dataclass(frozen=True, slots=True)
class SessionTools:
    """The curator + dispatcher pair and the model-derived loop knobs.
    `cfg` is the decompose-resolved config the Workflow must be built with."""

    curator: GraphCurator
    dispatcher: ToolDispatcher
    compact_drop_at_chars: int
    compact_summarise_at_chars: int
    keep_recent_chars: int
    cfg: Config


def build_session_tools(
    cfg: Config,
    *,
    cwd: Path,
    state_dir: Path,
    layout: SessionLayout,
    isolation: IsolationLevel,
    mode: ResumableMode,
    events: EventSink,
    prompts: OperatorPrompts,
    loop_log: Callable[[str], None],
    mcp_manager: MCPManager | None,
    rm_role: RoleModel,
    session_net: SessionNetwork | None = None,
    worktree_git_dir: Path | None = None,
) -> SessionTools:
    # The DAG curator runs in-process: the run's worker.lock already makes
    # this the sole writer, so no subprocess or socket is needed.
    curator = GraphCurator(layout)
    dispatcher = ToolDispatcher(
        root=cwd,
        config=cfg,
        isolation=isolation,
        prompts=prompts,
        events=events,
        curator=curator,
        run_root_node_id=None,  # Workflow seeds the root + calls set_run_root_node_id
        mcp_manager=mcp_manager,
        worktree_git_dir=worktree_git_dir,
        mode=mode,
        state_dir=state_dir,
        session_dir=layout.session_dir,
        # One jail process for this run's commands.
        use_jail_session=True,
        session_net=session_net,
    )
    compact_drop, compact_summarise, keep_recent = resolve_compaction_thresholds(
        cfg, rm_role, log=loop_log
    )
    cfg = resolve_decompose(cfg, rm_role, log=loop_log)
    return SessionTools(
        curator=curator,
        dispatcher=dispatcher,
        compact_drop_at_chars=compact_drop,
        compact_summarise_at_chars=compact_summarise,
        keep_recent_chars=keep_recent,
        cfg=cfg,
    )


def session_facts_provider(
    budget: BudgetTracker, model: str, run_commands: str, isolation: str
) -> Callable[[], SessionFacts]:
    """The live-facts thunk the front-end's pause banner reads. The fixed
    fields bind now (they never change for the leg); spend reads live."""

    def facts() -> SessionFacts:
        spend, partial = budget.estimate_usd()
        return SessionFacts(
            spend_usd=spend,
            spend_partial=partial,
            model=model,
            run_commands=run_commands,
            isolation=isolation,
        )

    return facts
