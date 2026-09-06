# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The agent loop: one system prompt, one model driving tool calls, and a
deterministic harness around it (jail, budget, verify timeout, DAG curator
for persistence and resume). One driver, no subagent cascade: the
review panel gates checkpoints, it never steers. Green verifies
auto-commit, so the chain records each tree a verify certified.
"""

from __future__ import annotations

import itertools
import json
import os
import random
import shutil
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from agent6.budget import BudgetExceeded, BudgetTracker
from agent6.config import Config
from agent6.directive import DirectiveError, Segment, parse_directive, parse_pin
from agent6.git_ops import (
    CommitIdentity,
    GitError,
    chain_commit,
    chain_dirty,
    chain_dirty_paths,
    chain_tip,
    commit_diff,
    conventional_commit_subject,
    diff_since,
    tree_diff_paths,
    worktree_name_status,
    worktree_tree,
)
from agent6.git_ops import status as git_status
from agent6.graph.curator import GraphCurator
from agent6.graph.models import (
    AddSubtaskIntent,
    NodeStatus,
    RecordCommitIntent,
    SetCursorIntent,
    TaskNodeDraft,
    UpdateStatusIntent,
)
from agent6.memory import decisions_path, decisions_text, memory_dir, record_decision
from agent6.memory import index_text as memory_index_text
from agent6.portable import atomic_write
from agent6.prompts.revision import (
    CONTEXT_SUMMARY_SYSTEM_PROMPT,
    GIST_DISTILL_SYSTEM_PROMPT,
    PINS_NO_RESTATE_CLAUSE,
    PROMPT_REVISION_SYSTEM_PROMPT,
    context_restart_notice,
    pinned_block,
    progress_summary_from_notice,
)
from agent6.providers import (
    Provider,
    ProviderAborted,
    ProviderError,
    ProviderInterrupted,
    ProviderResponse,
    ToolDefinition,
    call_for_text,
    output_cap_truncated,
)
from agent6.sessions.ipc import emit_session_start
from agent6.skills import ResolvedSkills, skill_command, skill_steer_payload
from agent6.task_text import operator_task_text
from agent6.tools.dispatch import (
    OperatorCommandUnexecutable,
    ToolDenied,
    ToolDispatcher,
    ToolError,
)
from agent6.tools.mcp_client import MCP_TOOL_PREFIX
from agent6.tools.results import AnswersResult, ExecResult, MetricResult, ToolResult
from agent6.tools.schema import (
    FinishPlanningInput,
    FinishSessionInput,
)
from agent6.types import AutoCommitDirective, RepoSummary
from agent6.verify_infer import infer_verify_command, read_agents_md
from agent6.workflows._compaction import (
    DROP_BLOCKS_AT_CHARS,
    KEEP_RECENT_CHARS,
    SUMMARISE_AT_CHARS,
    TOOL_RESULT_CHAR_CAP,
    GistRequest,
    cap_tool_result,
    compact_old_tool_results,
    context_chars,
    count_elisions,
    parse_checkoff,
    parse_gist_lines,
    recent_tail_start,
    recently_edited_paths,
    request_prefix_chars,
    strip_checkoff,
    strip_old_thinking,
)
from agent6.workflows._context import agents_md_text, load_repo_summary
from agent6.workflows._conversation import (
    AssistantTurn,
    Conversation,
    Notice,
    ToolResultItem,
    format_transcript_tail,
)
from agent6.workflows._dag_focus import (
    DAG_MUTATING_TOOLS,
    STUCK_NUDGE_MAX,
    STUCK_ON_TASK_AFTER,
    current_task_banner,
    current_task_id,
    initial_dag_hint,
    ready_subtask,
    stuck_on_task_nudge,
)
from agent6.workflows._loop_state import (
    NEXT_TURN,
    LoopState,
    NextTurn,
    TurnState,
    restore_completion_state,
)
from agent6.workflows._metric import (
    METRIC_EARLY_FINISH_PATIENCE,
    METRIC_FINISH_NUDGE,
    METRIC_PLATEAU_PATIENCE,
    METRIC_PLATEAU_STOP_BELOW_BUDGET,
    MetricSample,
    best_metric_sample,
    coerce_metric_score,
    extract_metric_targets,
    format_metric_feedback,
    metric_at_fraction_ceiling,
    metric_goal,
    metric_plateau_nudge,
    metric_plateau_summary,
)
from agent6.workflows._nearest_tests import (
    diff_changed_paths,
    is_bare_pytest,
    nearest_test_paths,
)
from agent6.workflows._nudges import (
    BASELINE_RED_NOTICE,
    MEMORY_FINISH_NUDGE,
    MEMORY_FLIP_NUDGE,
    NO_PROGRESS_ESCALATE_AFTER,
    NO_PROGRESS_ESCALATION,
    NO_PROGRESS_NUDGE,
    NO_PROGRESS_NUDGE_AFTER,
    NO_PROGRESS_STOP_AFTER,
    PLAN_BUDGET_NUDGE,
    PLAN_BUDGET_NUDGE_BELOW,
    PLAN_NUDGE_AFTER_ITERS,
    PLAN_ON_DISK_HEADER,
    QUESTION_NUDGE,
    RUN_BUDGET_NUDGE,
    RUN_BUDGET_NUDGE_BELOW,
    RUN_BUDGET_NUDGE_GATELESS,
    SILENT_NO_WORK_NUDGE,
    SILENT_NO_WORK_PATIENCE,
    STAGNATION_NUDGE,
    STAGNATION_NUDGE_GATELESS,
    TASK_FINISH_PATIENCE,
    TOOL_DENIED_NUDGE,
    TOOL_ERROR_ESCALATE_AFTER,
    TOOL_ERROR_ESCALATION,
    TOOL_ERROR_NUDGE,
    TOOL_ERROR_NUDGE_AFTER,
    TOOL_ERROR_STOP_AFTER,
    VERIFY_BROKEN_NUDGE,
    VERIFY_SETTLED_NUDGE,
    VERIFY_SETTLED_NUDGE_AFTER,
    VERIFY_SETTLED_STOP_AFTER,
    VERIFY_UNADOPTED_NOTICE,
    ends_with_question,
    is_test_path,
    standing_fruitless_nudge,
    standing_resume_nudge,
    test_only_green_notice,
    tool_error_signature,
    unrunnable_signature,
    verify_did_not_run,
    verify_failure_signature,
)
from agent6.workflows._panel import (
    ReviewContext,
    ReviewDecision,
    inconclusive_note,
    panel_is_inconclusive,
    render_findings,
)
from agent6.workflows._parallel_dispatch import (
    LaneJoin,
    join_lane_result,
    segment_lanes,
    segment_stamp,
    summary_text,
)
from agent6.workflows._prompt_blocks import build_system_prompt, initial_instructions
from agent6.workflows._prompt_revision import (
    PromptRevision,
    PromptRevisionError,
    clip_text,
    format_effective_task,
    format_prompt_revision_context,
    parse_prompt_revision,
)
from agent6.workflows._provider_call import (
    NON_RETRYABLE_HTTP_STATUSES,
    RETRY_AFTER_CEILING_S,
    is_empty_tool_call_response,
    provider_error_hint,
)
from agent6.workflows._review import CritiqueResult, ReviewDispatch, ReviewSeat, run_panel
from agent6.workflows._session_state import (
    TURN_IN_FLIGHT_NAME,
    ResumeError,
    SessionEndReason,
    SessionResult,
    SessionSnapshot,
    Verification,
    clear_turn_marker,
    load_session_snapshot,
    write_turn_marker,
)
from agent6.workflows._toolset import (
    build_readonly_review_tools,
    tool_definitions,
)
from agent6.workflows._verify_gate import (
    finish_red_notice,
    gate_withheld_notice,
    harness_verify_due,
    harness_verify_notice,
    scoped_verify_notice,
)
from agent6.workflows.subrun import (
    GroupLaneSpawner,
    SubrunError,
)

# A re-served tool result must exceed this many bytes before the back-to-back
# dedupe elides it; below it the stub would not save enough to matter and the
# small results (finish/dag echoes) should pass through verbatim.
_DEDUPE_MIN_CHARS = 500


if TYPE_CHECKING:
    from agent6.events import EventSink


# Consecutive went-quiet turns after which a metric run drops the worker's
# per-call output cap from metric_task_max_tokens back to per_call_max_tokens
# (see Workflow._worker_max_tokens). 2 spares a one-off starvation its full
# recovery room while breaking a reasoning-binge spiral.
_STARVATION_BACKOFF_AFTER_QUIETS = 2

# Total chars of operator `/pin` instructions a run may hold. Pins are
# re-injected verbatim into every tier-2 restart, so the cap bounds what every
# post-compaction context permanently re-pays. An over-cap pin is delivered as
# an ordinary steer (the instruction still reaches the model once); only its
# survives-compaction durability is refused.
PINS_MAX_CHARS = 4_000

# Paths counted for the operator-stop dirty-tree note; a bigger tree reads "N+".
_DIRTY_NOTE_CAP = 500


def _first_prose_line(text: str, *, fallback: str) -> str:
    """The agent's first prose line (leading `<thinking>` blocks dropped,
    heading/bullet markers stripped), or *fallback* on a pure tool-call turn."""
    cleaned = text
    while cleaned.lstrip().startswith("<thinking>"):
        end = cleaned.find("</thinking>")
        if end == -1:
            cleaned = ""
            break
        cleaned = cleaned[end + len("</thinking>") :]
    for raw_line in cleaned.splitlines():
        line = raw_line.strip().lstrip("#").lstrip("-*").strip()
        if line:
            return line
    return fallback


def _summarise_assistant_text_for_commit(
    text: str, iteration: int, *, fallback: str = "verify passed"
) -> str:
    """`agent6 iter N: <first line>`, the first line truncated to 72 chars
    (git `--oneline` width). Free: `resp.text` is already in hand."""
    subject_body = _first_prose_line(text, fallback=fallback)[:72]
    return f"agent6 iter {iteration}: {subject_body}"


def _plan_is_title_only(plan_md: str) -> bool:
    """True when plan_markdown has no body: only heading lines (`# ...`) and
    blanks, so `--from-plan` would get a stub. Weak models leave it a bare title
    and put the plan in `summary`; the caller salvages that case."""
    return not any(
        line.strip() and not line.lstrip().startswith("#") for line in plan_md.splitlines()
    )


def _last_assistant_prose(conversation: Conversation) -> str:
    """The text of the newest assistant turn ("" when the last turn is not
    the assistant's), for pairing a steer with the question it answers."""
    turns = conversation.turns
    if not turns or not isinstance(turns[-1], AssistantTurn):
        return ""
    return "".join(
        str(b.get("text", ""))
        for b in turns[-1].raw_content
        if isinstance(b, dict) and b.get("type") == "text"
    )


@dataclass
class Workflow:
    """Single-loop agent workflow.

    The agent decides everything via tool calls in one large loop:
    when to read, when to plan (implicitly via subsequent tool calls),
    when to edit, when to verify, when to measure the metric, when to
    pivot, when to stop. The harness keeps the loop bounded
    (max_iterations, budget caps, verify_timeout) and observable
    (events).
    """

    root: Path
    config: Config
    provider: Provider
    dispatcher: ToolDispatcher
    logger: Callable[[str], None] = field(default=print)
    events: EventSink | None = None
    # In-process GraphCurator. When None,
    # DAG-as-tool handlers raise ToolError and the loop runs without DAG
    # persistence (still usable for bench / one-off tasks). When wired,
    # Workflow.run() seeds a root task and the agent can add subtasks
    # and update statuses; survives crashes via <run-dir>/graph.jsonl.
    curator: GraphCurator | None = None
    # Per-invocation token budget tracker (the same instance wired into
    # the provider). When present the loop can read how much budget
    # remains and use it to decide whether a metric plateau is worth
    # quitting on. None in test / MCP paths; the loop degrades to fixed
    # count-based heuristics when it is unset.
    budget: BudgetTracker | None = None
    # Per-repo state dir holding the cross-run memory store
    # (<state_dir>/memory/). When set, the memory index is injected into
    # the system prompt at run start; the CLI wires the same path into the
    # dispatcher so memory-dir edits persist across runs.
    # None (bench / tests / one-off embedders) runs memory-less.
    state_dir: Path | None = None
    # Rendered [git.commit].trailer line (render_commit_trailer), appended once
    # to every commit this loop makes. None = no trailer configured.
    commit_trailer: str | None = None
    # The run's detached commit chain. Per-step commits land on chain_ref
    # (`refs/agent6/<session>/head`, the gc anchor) via a temp index: HEAD, the
    # operator's index, and the checkout are never touched, so operator or
    # model git activity mid-run cannot collide with the run's own record.
    # None (plan/ask, unit-test embedders) = the loop never commits.
    chain_ref: str | None = None
    # Visible `refs/heads/<name>` advanced to the same tip ([git].branch_per_run);
    # None = hidden ref only.
    chain_branch: str | None = None
    # Parent for the chain's first commit when chain_ref does not exist yet:
    # HEAD's sha at run start (None in an unborn repo).
    chain_fallback_parent: str | None = None
    # Files that were untracked when the run started (repo-root-relative). The
    # operator's, so no chain commit records them and no dirty check counts
    # them; a file the model creates is not in the set and is committed.
    untracked_at_start: frozenset[str] = frozenset()
    # [git].commit_per_step: False disables every agent commit. The chain never
    # advances; resume-from-git, sessions diff/merge, and /parallel dispatch
    # from a changed tree degrade, and the work stays only in the worktree.
    commit_per_step: bool = True
    # Cap on assistant turns for THIS leg (config [workflow].max_iterations;
    # -1 unlimited). Each turn = one provider.call. A resumed leg re-arms the
    # allowance: the cap is relative to its start_iteration, so a standing
    # run is bounded per leg, never by the sum of its history.
    max_iterations: int = 200
    # Per-call max_tokens for the LLM response. NOT the bench's total
    # output budget (that's BudgetTracker's job). Sized for ONE turn:
    # enough for reasoning + tool-call args + content on a reasoning
    # model, small enough to fit alongside the input in a 262k-context
    # model like Kimi 2.6. Sonnet (no reasoning) uses ~600 of this;
    # Kimi-k2.6 reasoning needs ~5-15k.
    per_call_max_tokens: int = 16384
    # Per-call output cap for the worker on metric-optimization runs (mode
    # "run" with a configured continuous metric). Those tasks reward large
    # single-turn edits, so a tight cap truncates mid-apply_patch and wastes the
    # turn; ordinary feature/bugfix runs stay on the tighter default, where a
    # giant turn mostly means a confused model. At 32k a heavy reasoner still
    # hit stop_reason="length" on ~30% of turns before emitting a tool call
    # (measured: bench/perf/README.md).
    metric_task_max_tokens: int = 65536
    # Sampling temperature pinned for every provider call (worker and
    # review seats); unset, each provider routes to its own default, and OpenRouter's
    # per-model defaults are high enough to produce degenerate output. Pinning
    # 0.0 makes the tool-use loop reproducible. CLI wires these from
    # `cfg.models.<role>.temperature`.
    temperature: float | None = 0.0
    # Tiered context compaction thresholds (chars).
    compact_drop_at_chars: int = DROP_BLOCKS_AT_CHARS
    compact_summarise_at_chars: int = SUMMARISE_AT_CHARS
    # One tool result's size bound before it enters the conversation; a
    # provider that hands the model less than the default gets a tighter one.
    tool_result_cap_chars: int = TOOL_RESULT_CHAR_CAP
    # Verbatim recent-history tail kept through a tier-2 restart (chars; 0
    # keeps none). Sized to pi's keepRecentTokens default.
    keep_recent_chars: int = KEEP_RECENT_CHARS
    # Thinking blocks are dropped from assistant turns older than this many
    # assistant turns, at tier-1 moments. 0 (default) keeps all thinking,
    # today's and pi's behavior; Claude Code clears old thinking.
    keep_thinking_turns: int = 0
    # Retry the provider call on transient ProviderError before aborting the
    # run. Common cases: Anthropic 529 overload, Anthropic "Server disconnected
    # without sending a response" (httpx2 RemoteProtocolError, no HTTP status),
    # OpenRouter 502, brief socket timeouts. Such a disconnect can flap for a
    # few seconds, and a long, expensive run must not abort on one blip.
    # With exponential backoff (2s/4s/8s/16s, full-jittered, capped
    # at provider_retry_max_delay_s) four retries ride out a multi-second flap;
    # permanent statuses (401/402/403/404/422) and BudgetExceeded still fail
    # fast. Set to 0 to disable retrying.
    provider_retry_count: int = 4
    provider_retry_delay_s: float = 2.0
    provider_retry_max_delay_s: float = 30.0
    # Steering interrupt callbacks, polled between iterations; on request the
    # workflow prompts the operator for an instruction or "abort". When unset
    # (the defaults) the loop runs without operator interaction.
    steer_requested: Callable[[], bool] = field(default=lambda: False)
    # A machine agent state's finish contract: called on each finish_session
    # payload, returning the problems (empty = conforms). Injected by the
    # machine leg builder from the state's output_schema; None (every plain
    # run) leaves finishes ungated. The engine's own validation of the
    # recorded fact stays the authority.
    finish_validator: Callable[[dict[str, Any] | None], list[str]] | None = None
    steer_clear: Callable[[], None] = field(default=lambda: None)
    steer_prompt: Callable[[], str | None] = field(default=lambda: None)
    # Called at each leg entry (run/resume): disarms a SIGINT stage the prior
    # leg never consumed, without touching the steer marker files.
    steer_reset: Callable[[], None] = field(default=lambda: None)
    # Manual compaction request (the TUI's "Compact now"): polled at the same
    # pre-call boundary as the tiered thresholds; a positive forces the tier-2
    # summarise-and-restart. The marker travels the same file bridge as steer.
    compact_requested: Callable[[], str | None] = field(default=lambda: None)
    compact_clear: Callable[[], None] = field(default=lambda: None)
    # Operator "stop after this step": polled at each completed-iteration
    # boundary (post tool results + auto-commit), ending the run cleanly there.
    # The mid-turn immediate stop stays the steer "abort" answer.
    stop_requested: Callable[[], bool] = field(default=lambda: False)
    stop_clear: Callable[[], None] = field(default=lambda: None)
    # Polled DURING a streaming model call (not just between steps): True once the
    # operator has asked to stop, so a long reasoning turn aborts promptly.
    should_abort: Callable[[], bool] = field(default=lambda: False)
    # Polled DURING a streaming call: True once the operator has asked to STEER
    # (Ctrl-C / TUI `s`), so the watchdog ends the turn and the loop reaches its
    # steer boundary (the menu) at once instead of waiting the whole turn out.
    should_interrupt: Callable[[], bool] = field(default=lambda: False)
    # Hook invoked once per successful auto-commit (after the
    # commit lands). Returning "stop" exits the loop cleanly with
    # completed=True, reason="interactive_stop"; "undo" takes the steer's
    # /undo path (fork back before the last message); "continue" (the
    # default) lets the next iteration run. The CLI's `agent6 run -i`
    # installs a TTY prompt here for the REPL; default no-op preserves
    # autonomous behaviour for `agent6 run` and `agent6 resume`.
    after_auto_commit: Callable[[int, str], AutoCommitDirective] = field(
        default=lambda _i, _sha: "continue"
    )
    # `/parallel` steer dispatch: the ui-side group spawner that runs a sibling
    # group of subordinate lanes to completion and imports their branches into
    # this run's repo (workflows.subrun.GroupLaneSpawner). None (the default, and
    # every headless / non-run path) makes a `/parallel` directive answer with
    # steer feedback and continue -- never a crash. Depth 1: the ui side tags lane
    # spawns AGENT6_SUBRUN=1 and run.py leaves this None inside a lane.
    lane_spawner: GroupLaneSpawner | None = None
    # `/undo`: commits the tree as it stands onto this session's ref, forks
    # the session at the state before its last operator message, and puts the
    # checkout back to that state's tree (app.fork.undo_fork, injected:
    # workflows never import app); returns (new_session_id, undone_text), or
    # None with the reason printed.
    undo_forker: Callable[[], tuple[str, str] | None] | None = None
    # In-loop review panel. When `review_trigger != "off"` and `review_seats`
    # is non-empty, the panel runs at the configured trigger (verify-failure /
    # before finish_session / every review_period iters) over the run diff and
    # injects its findings back into the conversation on the next user turn.
    review_trigger: Literal["off", "on_verify_fail", "before_finish", "periodic"] = "off"
    review_period: int = 10
    # Optional one-shot prompt revision before the first worker call.
    # The CLI wires this to the reviewer model when prompt.revise_prompt !=
    # "off". It never receives tools and never iterates.
    prompt_reviser_provider: Provider | None = None
    revise_prompt: Literal["off", "auto", "interactive"] = "off"
    prompt_reviser_temperature: float | None = 0.0
    prompt_revision_max_tokens: int = 2048
    prompt_revision_selector: Callable[[str, str, tuple[str, ...]], str | None] | None = None
    # Tier-2 context compaction (summarise-and-restart). When the
    # cumulative tool_result size crosses `compact_summarise_at_chars`,
    # the loop asks this provider to summarise the elided history into a
    # compact progress block and restarts the message list from (original
    # task + summary). Wired by the CLI to the reviewer role (cheaper than
    # the worker). When None the loop falls back to `provider` so the
    # feature still works without explicit wiring.
    summariser_provider: Provider | None = None
    # Pins seeded before the first turn (a /parallel lane inherits the
    # coordinator's standing instructions via the spawner's --pin channel,
    # out-of-band of user_task). Fresh runs only; resume/fork restore pins
    # from the snapshot instead.
    initial_pins: Sequence[str] = ()
    context_summary_max_tokens: int = 2048
    # Tier-1 gist elision (`context.elision_gists`): large read_file results
    # decay to a distilled-gist placeholder (summariser model, one batched call
    # per drop event) before the bare marker. Off = pre-gist behavior.
    compact_elision_gists: bool = True
    # Cap on consecutive `before_finish` rejections.
    # When the worker repeatedly calls finish_session and the panel keeps
    # rejecting, the loop would otherwise burn budget bouncing.
    # After this many back-to-back rejections, the next finish_session is
    # accepted (with a `[review]` warning still injected so the
    # transcript records the disagreement). 0 disables the cap.
    max_consecutive_review_rejections: int = 2
    # The in-loop review panel: every trigger runs the grounded panel
    # (run_panel over the run diff + verify result). `review_decision` gates
    # only for veto/quorum; "advisory" just injects findings as a [review]
    # message.
    # The panel reviews `git diff base_sha` (the run's cumulative change). The
    # per-run rejection counter auto-disarms the gate after
    # `review_max_total_rejections` blocks so it can never stall the run.
    review_seats: list[ReviewSeat] = field(default_factory=list)
    review_decision: ReviewDecision = "advisory"
    review_quorum: int = 2
    review_max_total_rejections: int = 4
    review_budget_fraction: float = 0.25
    review_concurrency: int = 1
    base_sha: str = ""
    # When set, : Workflow writes a JSON snapshot of (system, messages,
    # tool_calls, next_iteration, root_task_id) before every LLM call. The
    # snapshot is provider-agnostic (it holds the anthropic-shaped message
    # list the loop maintains internally, not the on-the-wire OpenAI-shaped body
    # the openai provider sends) so `agent6 resume` works regardless of which
    # provider the prior run used. Atomic write (tmp + rename) so a crash
    # mid-write leaves the prior snapshot intact.
    resume_state_path: Path | None = None
    # The operator's standing goal (`run --standing`): seeded as a standing
    # task under the root at run start. "" = none.
    standing_goal: str = ""
    # An operator is watching and can steer live (a foreground CLI/TUI run or
    # an interactive resume). A quiet turn then PARKS for a steer instead of
    # ending: interactively, going quiet is the most normal thing an agent
    # does, not a failure.
    interactive: bool = False
    # Plan mode. When `mode="plan"`, the workflow uses the
    # planning system prompt + plan-mode tool list (no apply_edit /
    # apply_patch; finish_planning replaces finish_session), skips auto-
    # commit-on-verify-pass, and on finish_planning writes the
    # `plan_markdown` argument to `plan_output_path` before exiting.
    # `plan_output_path` is required when `mode="plan"`.
    mode: Literal["run", "plan", "ask", "machine", "agent"] = "run"
    plan_output_path: Path | None = None
    # weak-model resilience. Open-weights models sometimes emit an empty turn
    # mid-run (no text, no tool_use, stop_reason="end_turn" or
    # equivalent) and would otherwise terminate the run immediately.
    # When `went_quiet_max_nudges > 0`, the loop instead injects a
    # short [harness] notice into the conversation and re-asks the
    # model, up to this many times PER RUN. Reset on any non-empty
    # turn. Set to 0 to restore the "fail fast on went_quiet"
    # behaviour. Reasoning-starvation bursts count as went_quiet, so the cap is
    # sized to survive a few of them.
    went_quiet_max_nudges: int = 4
    # loop-guard escalation. The guard injects a one-shot
    # notice when the same (tool, args) signature streak hits
    # `repeat_threshold` (default 3). When the worker ignores it and the
    # streak reaches `loop_guard_kill_threshold`, the loop forcibly
    # terminates with reason="loop_guard_killed" rather than letting
    # the worker burn the rest of the budget circling the same call.
    # Set to 0 to disable forced termination (notice-only behaviour).
    loop_guard_kill_threshold: int = 10
    # One factual notice when this much wall clock passes with zero edit and
    # zero verify calls. Notice-only, never kills; 0 disables. Measured basis:
    # recall-spiral runs die by timeout with 3-10 total calls, below every
    # call-count guard's horizon.
    stagnation_notice_after_s: float = 300.0
    # One-shot guard so a persistently unwritable state dir (full disk, quota,
    # read-only mount) warns once instead of every turn. Snapshot persistence is
    # recovery state; a failure disables resume/fork but must not abort the run.
    _snapshot_write_failed: bool = field(default=False, init=False)
    # The loop iteration currently being driven (0 before the loop starts). The
    # app-level KeyboardInterrupt fallbacks in run/resume read it so their
    # emergency session.end carries a truthful iteration count, matching the shape
    # the loop's own session.end emitters use.
    iterations_reached: int = field(default=0, init=False)

    # ---- run / resume entry ----------------------------------------------------

    def run(self, user_task: str) -> SessionResult:
        """Drive the single-loop agent to completion."""
        self.steer_reset()  # a leg starts with no armed Ctrl-C
        if self.mode == "plan" and self.plan_output_path is None:
            raise ValueError("Workflow(mode='plan') requires plan_output_path to be set")
        # The run dir name is the authoritative run id; stamped into session.start so
        # every fold reports it without re-deriving it from the path.
        session_id = self.events.path.parent.name if self.events is not None else ""
        # The event carries the operator's own words (a seed digest or skill
        # block prepended by `run --from`/`--skill` is context, not the task),
        # clipped: every headline reads this field.
        self._emit_start(
            "session.start",
            session_id=session_id,
            user_task=operator_task_text(user_task)[:200],
            mode=self.mode,
        )
        self._log("LOOP: LOAD_CONTEXT")
        repo = self._load_repo_summary()
        system = build_system_prompt(
            config=self.config,
            repo=repo,
            mode=self.mode,
            memory_index=self._load_memory_index(),
            memory_dir_path=str(memory_dir(self.state_dir)) if self.state_dir is not None else "",
            decisions=self._load_decisions(),
            decisions_path=str(decisions_path(self.state_dir)) if self.state_dir else "",
            skills=self._load_skills(),
            isolation=self.dispatcher.isolation,
            commands_allowed=self.dispatcher.command_policy() != "no",
        )

        try:
            effective_task = self._maybe_revise_prompt(user_task, repo)
        except PromptRevisionError as exc:
            self._log(f"LOOP: prompt revision failed: {exc}")
            self._emit(
                "session.end",
                reason="prompt_revision_failed",
                iterations=0,
                all_passed=False,
            )
            return SessionResult(
                completed=False,
                reason="prompt_revision_failed",
                summary=str(exc),
                iterations=0,
                tool_calls=0,
            )

        # Seed the run's root task and wire its id into the
        # dispatcher so add_task with parent_id=None has a parent. Skipped
        # gracefully if no curator is configured (DAG tools then
        # raise ToolError if called).
        root_id = self._seed_root_task(effective_task)
        if root_id is not None:
            self.dispatcher.set_run_root_node_id(root_id)
            self._log(f"LOOP: DAG root task seeded: {root_id}")
            if self.standing_goal.strip() and self.curator is not None:
                node = self.curator.add_subtask(
                    AddSubtaskIntent(
                        parent_id=root_id,
                        draft=TaskNodeDraft(
                            title=self.standing_goal.strip(),
                            standing=True,
                            created_by="steering",
                        ),
                    )
                )
                self._log(f"LOOP: standing goal seeded: {node.id}")
            self._emit_graph_snapshot()  # show the root in the live task view

        self._log(
            f"LOOP: mode={self.mode} system={len(system)} chars, task={len(effective_task)} chars"
        )

        # Initial user turn - the task + a brief operational header.
        # Cache breakpoints are rolled by the conversation each iteration,
        # so the growing history stays cached across turns.
        dag_hint = initial_dag_hint(root_id, self.mode, self.config.prompt.decompose == "on")
        instructions = initial_instructions(
            self.mode,
            self.config.sandbox.run_commands,
            has_gate=bool(self.config.workflow.verify_command),
        )
        initial_user = f"TASK:\n{effective_task}\n\n{instructions}{dag_hint}"
        conversation = Conversation()
        conversation.notice(initial_user)

        return self._drive_loop(
            system=system,
            conversation=conversation,
            tool_calls=0,
            start_iteration=1,
            root_task_id=root_id,
            original_task=effective_task,
        )

    def resume(self) -> SessionResult:
        """Resume a paused/crashed run from its snapshot.

        Reads `self.resume_state_path` (the snapshot written by the
        loop before each LLM call), reattaches the DAG root task id to
        the dispatcher, and re-enters the loop at the saved iteration
        with the saved conversation. The budget tracker is fresh per
        invocation (by design - see `agent6.budget` docstring); the
        DAG state on disk is restored by spawning a curator against the
        same run layout in the CLI.
        """
        self.steer_reset()  # a leg starts with no armed Ctrl-C
        if self.resume_state_path is None:
            raise ResumeError("resume() called but resume_state_path is None")
        try:
            snapshot = load_session_snapshot(self.resume_state_path)
            conversation = Conversation.from_wire(snapshot.messages)
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            raise ResumeError(
                f"failed to load resume snapshot from {self.resume_state_path}: {exc}"
            ) from exc

        # The leg's log opens with this event: stamp session_id + mode like
        # session.start so the log identifies itself (the manifest owns the task).
        self._emit_start(
            "loop.resume.start",
            session_id=self.events.path.parent.name if self.events is not None else "",
            mode=self.mode,
            iteration=snapshot.next_iteration,
            messages=len(snapshot.messages),
        )
        self._log(
            f"LOOP: RESUME from {self.resume_state_path} "
            f"(iter={snapshot.next_iteration}, messages={len(snapshot.messages)}, "
            f"tool_calls={snapshot.tool_calls})"
        )

        if snapshot.root_task_id is not None:
            self.dispatcher.set_run_root_node_id(snapshot.root_task_id)
            self._log(f"LOOP: DAG root task restored: {snapshot.root_task_id}")

        # The system prompt is the run's, frozen: config that gained (or lost) a
        # verify command between legs swaps what judges the work while the
        # instructions still name the old gate. Say so rather than let the
        # worker run a command nothing checks.
        gate = tuple(self.config.workflow.verify_command)
        if gate != snapshot.verify_command:
            was = " ".join(snapshot.verify_command) or "none"
            now = " ".join(gate) or "none"
            conversation.notice(
                f"[harness] This run's verify gate changed between legs: it was `{was}`,"
                f" it is now `{now}`. The instructions above still name the old one."
            )
            self._log(f"LOOP: verify gate swapped on resume: {was} -> {now}")
            self._emit("loop.verify_swapped", was=list(snapshot.verify_command), now=list(gate))

        return self._drive_loop(
            system=snapshot.system,
            conversation=conversation,
            tool_calls=snapshot.tool_calls,
            start_iteration=snapshot.next_iteration,
            root_task_id=snapshot.root_task_id,
            original_task=snapshot.original_task,
            resume_from=snapshot,
        )

    def _seed_carryover(
        self, state: LoopState, conversation: Conversation, resume_from: SessionSnapshot | None
    ) -> None:
        """Seed the leg's carried state and announce it for the read model.

        A resumed/forked leg re-announces its restored pins and elision
        counters: a fork's fresh logs.jsonl has no pin.added or compact
        events to fold (the fold REPLACES on these events, so a plain resume
        never double-counts). Announced even when empty: a pin whose
        pin.added reached the log but whose snapshot never did is still
        folded from the same log, so only a replace with the real (empty)
        list stops the surfaces listing a pin no restart will re-inject.

        A FRESH run seeded with pins (--pin; the /parallel lane channel) uses
        the same state, the same replace-fold event, and the same block a
        restart re-shows, so the wording never depends on the delivery path.
        """
        if resume_from is not None:
            restore_completion_state(state, resume_from)
            self._carry_verify_verdict(state, resume_from)
            self._emit("loop.pin.restored", pins=list(state.pins), count=len(state.pins))
            elided, gists = count_elisions(conversation)
            self._emit("loop.compact.restored", elided=elided, gists=gists)
        elif self.initial_pins:
            # Seed via the pin owner so --pin honors the cap + non-empty check
            # (writing state.pins directly skipped both); a --pin that doesn't
            # fit is refused loudly, not silently dropped or wedged in.
            for pin in self.initial_pins:
                if not self._try_pin(state, pin):
                    self._log(f"  --pin refused (empty or over the {PINS_MAX_CHARS}-char cap)")
                    self._emit("loop.pin.refused", chars=len(pin), limit=PINS_MAX_CHARS)
            self._emit("loop.pin.restored", pins=list(state.pins), count=len(state.pins))
            if state.pins:
                conversation.notice(pinned_block(state.pins))

    def _carry_verify_verdict(self, state: LoopState, snap: SessionSnapshot) -> None:
        """Carry the prior leg's verify observation when it still describes THIS
        tree: HEAD is the snapshot's and the worktree is clean. An operator
        commit or edit between legs invalidates it -- fails closed, like the
        baseline probe, so the leg starts unobserved rather than wrongly green
        or red. `baseline_ok` is about the BASE commit, which resume never
        moves: it carries unconditionally."""
        state.verify.baseline_ok = snap.baseline_ok
        state.standing_tools_mark = snap.standing_tools_mark
        state.standing_fruitless = snap.standing_fruitless
        state.ok_tool_calls = snap.ok_tool_calls
        if snap.last_verify_ok is None or not snap.head_sha:
            return
        try:
            status = git_status(self.root, exclude=self.untracked_at_start)
        except (GitError, OSError):
            return
        if status.is_clean and status.head_sha == snap.head_sha:
            state.verify.last_ok = snap.last_verify_ok
            state.verify.edited_since = snap.edited_since_verify

    # ---- the turn pipeline -----------------------------------------------------

    def _drive_loop(  # noqa: PLR0911, PLR0912
        self,
        *,
        system: str,
        conversation: Conversation,
        tool_calls: int,
        start_iteration: int,
        root_task_id: str | None,
        original_task: str,
        resume_from: SessionSnapshot | None = None,
    ) -> SessionResult:
        """Shared loop body for both fresh `run()` and `resume()`: one
        `TurnState` per tool-use iteration, driven through the turn phases
        in order. Any phase returning a SessionResult ends the run.

        `original_task` is the exact task string (in-loop review calls ground
        on it): run() threads it straight through, resume() reads it verbatim
        from the snapshot -- never re-derived from the message history.

        Before each provider call, writes a snapshot of the workflow's
        in-memory state to `self.resume_state_path` (if set) so a
        crash mid-call can be resumed from the same point.
        """
        state = LoopState(
            original_task=original_task,
            tool_calls=tool_calls,
            # steer-boundary phases parent DAG nodes here, and snapshot with
            # the system prompt (see _dispatch_parallel).
            root_task_id=root_task_id,
            system=system,
        )
        self._seed_carryover(state, conversation, resume_from)
        # This LEG's allowance: start..start-1+max (-1 = unbounded); a resumed
        # leg re-arms rather than inheriting a spent absolute counter.
        for iteration in (
            range(start_iteration, start_iteration + self.max_iterations)
            if self.max_iterations >= 0
            else itertools.count(start_iteration)
        ):
            self.iterations_reached = iteration
            if resume_from is not None and iteration == start_iteration:
                seeded = self._seeded_steer(conversation, iteration, state)
                if seeded is not None:
                    return seeded
            # Rebuilt per turn, not per leg: a gate adopted mid-run, or a
            # policy the operator denies mid-run, changes what the worker has.
            # A frozen list offers a tool that is gone, or keeps offering one
            # that only raises. Built BEFORE the context prep, which measures
            # the request the tools ride in.
            tools = tool_definitions(self.dispatcher, mode=self.mode)
            wire = self._turn_pre_call(
                system=system,
                conversation=conversation,
                state=state,
                iteration=iteration,
                start_iteration=start_iteration,
                root_task_id=root_task_id,
                prefix_chars=request_prefix_chars(system, tools),
            )
            if isinstance(wire, SessionResult):
                return wire
            got = self._turn_provider_call(
                system,
                conversation,
                wire,
                tools,
                state,
                iteration=iteration,
            )
            if isinstance(got, SessionResult):
                return got
            if isinstance(got, NextTurn):
                continue
            # The response's blocks enter the history verbatim, so tool_use
            # IDs (and thinking blocks) round-trip cleanly.
            assistant = conversation.assistant(got.raw.get("content") or [])
            if not assistant.tool_uses:
                result = self._handle_no_tool_use(
                    got, assistant, conversation, state, iteration=iteration
                )
                if result is not None:
                    return result
                # A completed prose turn is snapshotted like a tool turn, so an
                # operator stop at the boundary below resumes from AFTER the
                # prose + nudge instead of re-paying the provider call.
                self._save_resume_snapshot(
                    system=system,
                    messages=conversation.to_wire(),
                    tool_calls=state.tool_calls,
                    next_iteration=iteration + 1,
                    root_task_id=root_task_id,
                    state=state,
                )
                # A prose turn is a completed iteration too: without this
                # boundary a model answering in prose could never be stopped
                # or steered (the stop marker sat pending while the run kept
                # calling the provider).
                outcome = self._operator_boundary(conversation, iteration, state)
                if outcome is not None:
                    return outcome
                continue
            turn = TurnState(iteration=iteration, resp=got, assistant=assistant)
            # BEFORE dispatch: a crash between a tool's side effect and the
            # after-tools snapshot below leaves this marker at the iteration
            # resume would re-run, so resume can ask instead of silently
            # replaying a non-idempotent effect.
            if self.resume_state_path is not None:
                write_turn_marker(
                    self.resume_state_path.parent / TURN_IN_FLIGHT_NAME,
                    iteration,
                    tuple(tu.name for tu in assistant.tool_uses),
                )
            result = self._turn_dispatch_tools(state, turn)
            if result is not None:
                return result
            # One task-DAG snapshot per turn (not per mutation), so several
            # add_task/update_task calls in a turn collapse to a single event.
            if turn.dag_mutated:
                self._emit_graph_snapshot()
            result = self._turn_auto_commit_and_metric(state, turn)
            if result is not None:
                return result
            self._turn_review_triggers(state, turn, conversation)
            self._turn_finish_gates(state, turn, conversation)
            self._turn_notices(state, turn)
            self._turn_metric_plateau(state, turn)
            result = self._turn_verify_settled(state, turn)
            if result is not None:
                return result
            self._turn_no_progress(state, turn)
            conversation.results(turn.tool_results)
            # Snapshot AFTER the executed tools (assistant turn + tool_results
            # are now in the conversation) so a crash before iteration N+1's
            # pre-call snapshot resumes from AFTER the dispatched tools instead
            # of replaying them. The dispatch->snapshot window itself stays
            # open (the side effect and this write are not atomic); the
            # in-flight marker above covers it, so resume detects the one case
            # where replay may repeat a non-idempotent effect and asks. Marker
            # deletion comes AFTER this write: a crash mid-snapshot then leaves
            # a stale marker resume clears silently, never a missed one.
            self._save_resume_snapshot(
                system=system,
                messages=conversation.to_wire(),
                tool_calls=state.tool_calls,
                next_iteration=iteration + 1,
                root_task_id=root_task_id,
                state=state,
            )
            if self.resume_state_path is not None:
                clear_turn_marker(self.resume_state_path.parent / TURN_IN_FLIGHT_NAME)
            result = self._turn_stop_checks(state, turn, conversation)
            if result is not None:
                return result
            outcome = self._operator_boundary(conversation, iteration, state)
            if outcome is not None:
                return outcome

        self._log(f"LOOP: max_iterations={self.max_iterations} reached")
        self._final_checkpoint(self.iterations_reached)
        self._emit(
            "session.end",
            reason="max_iterations",
            iterations=self.iterations_reached,
            all_passed=False,
        )
        return SessionResult(
            completed=False,
            reason="max_iterations",
            summary=f"max_iterations={self.max_iterations} reached without finish_session",
            iterations=self.iterations_reached,
            tool_calls=state.tool_calls,
        )

    def _seeded_steer(
        self, conversation: Conversation, iteration: int, state: LoopState
    ) -> SessionResult | None:
        """Consume the follow-up a `resume --steer` queued before the loop
        started (`resume.py` write_steer_answer).

        Up front, so it enters the conversation ahead of the first provider
        call and drives this turn: a resumed already-finished conversation
        silent-finishes on iteration 1 and returns before the end-of-iteration
        poll ever runs, dropping the follow-up. Only the first resumed
        iteration -- mid-run Ctrl-C steering stays on the completed-iteration
        poll, and a Ctrl-C cannot precede this point."""
        return self._steer_outcome(
            self._maybe_handle_steer(conversation, iteration, state), iteration, state
        )

    def _turn_pre_call(
        self,
        *,
        system: str,
        conversation: Conversation,
        state: LoopState,
        iteration: int,
        start_iteration: int,
        root_task_id: str | None,
        prefix_chars: int = 0,
    ) -> list[dict[str, Any]] | SessionResult:
        """Prepare the context for this turn's provider call: budget heartbeat,
        tiered compaction, the plan re-read, pre-call nudges, rolling cache
        breakpoints, then the pre-call resume snapshot. Returns the serialized
        wire, so the snapshot on disk and the provider call carry the same list
        by construction -- or the parked SessionResult when the plan file
        cannot be read.

        The cache breakpoints advance AFTER compaction + nudges (the tail must
        be final) and BEFORE the snapshot (markers persist across resume).
        After the snapshot write, a crash anywhere up to the next iteration's
        snapshot can be resumed by re-running this same call."""
        self._emit_budget(iteration)
        if self._maybe_compact(conversation, state, prefix_chars=prefix_chars):
            # A tier-2 restart wiped the surfaced focus banner and the plan
            # block; let the passes below put both back into the fresh context.
            state.surfaced_task_id = None
            state.plan_injected = ""
        parked = self._maybe_inject_plan(conversation, state, iteration=iteration)
        if parked is not None:
            return parked
        self._maybe_pre_call_nudges(
            conversation, state, iteration=iteration, start_iteration=start_iteration
        )
        conversation.roll_cache_marks()
        wire = conversation.to_wire()
        self._save_resume_snapshot(
            system=system,
            messages=wire,
            tool_calls=state.tool_calls,
            next_iteration=iteration,
            root_task_id=root_task_id,
            state=state,
            # The one numbered-checkpoint writer: this state is what turn
            # `iteration`'s provider call consumes.
            write_checkpoint=True,
        )
        return wire

    def _turn_provider_call(
        self,
        system: str,
        conversation: Conversation,
        wire: list[dict[str, Any]],
        tools: list[ToolDefinition],
        state: LoopState,
        *,
        iteration: int,
    ) -> SessionResult | NextTurn | ProviderResponse:
        """One worker call with terminal-error classification. Returns the
        provider response on success, a SessionResult to end the run, or
        `NEXT_TURN` when a mid-stream steer discarded the turn (the menu
        chose continue, or injected an instruction, so the turn is re-done).
        `wire` is the pre-call serialization (already snapshotted); the
        conversation is only touched on the steer path."""
        try:
            return self._call_with_retry(
                system,
                wire,
                tools,
                self._worker_max_tokens(state),
            )
        except BudgetExceeded as exc:
            self._log(f"LOOP: budget exhausted at iter {iteration} ({exc})")
            self._final_checkpoint(iteration)
            self._emit(
                "session.end",
                reason="budget_exhausted",
                iterations=iteration,
                all_passed=False,
            )
            return SessionResult(
                completed=False,
                reason="budget_exhausted",
                summary=f"budget exhausted at iter {iteration}: {exc}",
                iterations=iteration,
                tool_calls=state.tool_calls,
            )
        except ProviderAborted:
            self.steer_clear()  # consume the stop; don't leave it on disk to re-read
            self._log(f"LOOP: operator stopped the run mid-turn at iter {iteration}")
            self._emit("session.end", reason="steer_abort", iterations=iteration, all_passed=False)
            return SessionResult(
                completed=False,
                reason="steer_abort",
                summary=f"operator stopped the run at iter {iteration}{self._dirty_tree_note()}",
                iterations=iteration,
                tool_calls=state.tool_calls,
            )
        except ProviderInterrupted:
            # A steer was requested mid-stream; the watchdog ended the (thinking)
            # turn so the loop handles it now rather than waiting it out. The partial turn
            # is discarded; the menu decides continue / steer / stop / detach.
            self._log(f"LOOP: steer requested mid-turn at iter {iteration}")
            outcome = self._steer_outcome(
                self._maybe_handle_steer(conversation, iteration, state), iteration, state
            )
            if outcome is not None:
                return outcome
            return NEXT_TURN  # "continue" or an injected instruction -> re-do the turn
        except ProviderError as exc:
            hint = provider_error_hint(exc.status_code, exc.provider)
            # The full upstream body (which can carry a noisy account user_id)
            # goes in this one diagnostic log line; the end-block summary below
            # stays concise so the raw blob is not echoed to the operator twice.
            self._log(f"LOOP: provider error at iter {iteration}: {exc}{hint}")
            self._final_checkpoint(iteration)
            self._emit(
                "session.end",
                reason="provider_error",
                iterations=iteration,
                all_passed=False,
            )
            status = f" (HTTP {exc.status_code})" if exc.status_code else ""
            # A fatal error's text is agent6's own remedy, not an upstream body.
            detail = f": {exc}" if exc.fatal else ""
            return SessionResult(
                completed=False,
                reason="provider_error",
                summary=f"provider error at iter {iteration}{status}{hint}{detail}",
                iterations=iteration,
                tool_calls=state.tool_calls,
            )

    def _turn_dispatch_tools(self, state: LoopState, turn: TurnState) -> SessionResult | None:
        """Dispatch each tool_use in the turn, appending one tool_result per
        call and noting effects (verify / metric / edits / DAG / finish) on
        `turn`, then the harness's own gate run when one is due. Returns a
        SessionResult only for the unexecutable-operator-command abort; tool
        errors become error tool_results instead."""
        # This iteration produced tool_uses, so the went_quiet
        # nudge budget refills (failures are per-streak, not per-run).
        state.went_quiet_nudges_used = 0
        for tu in turn.assistant.tool_uses:
            name = tu.name
            tool_input = tu.input
            if turn.finish_signal is not None:
                # A finish ends the turn's work: the calls after it are not
                # executed, as the finish tools' descriptions state.
                turn.tool_results.append(
                    ToolResultItem(
                        tool_use_id=tu.id,
                        content=json.dumps(
                            {"error": f"{name} not executed: it follows {turn.finish_kind}"}
                        ),
                        for_call=tu,
                    )
                )
                continue
            state.tool_calls += 1
            # degenerate-loop signature tracking. Stable
            # JSON so dict key order does not break equality. Same
            # (name, args) back-to-back across iterations increments
            # `state.spiral.call_streak`; anything else resets it.
            try:
                sig = f"{name}:{json.dumps(tool_input, sort_keys=True, ensure_ascii=False)}"
            except (TypeError, ValueError):
                sig = f"{name}:<unhashable>"
            state.spiral.note_call(sig)
            self._emit("loop.tool.call", name=name, iteration=turn.iteration)
            served = None
            try:
                result = self.dispatcher.dispatch(name, tool_input)
                content = json.dumps(result.to_wire(), ensure_ascii=False)
                self._note_tool_effects(state, turn, name, result, tool_input)
                # Dedupe a back-to-back identical (name, args) call whose result
                # bytes are unchanged: serve a short stub instead of re-sending
                # the full payload, so a re-read spiral cannot grow the context.
                # The call still dispatched (a CHANGED result serves in full);
                # only the redundant re-serve is elided.
                if state.spiral.stub_repeat(content, min_chars=_DEDUPE_MIN_CHARS):
                    served = json.dumps(
                        {
                            "repeated": (
                                f"Identical to your previous {name} call --"
                                f" result unchanged ({len(content)} bytes elided)."
                                " Do not re-issue the same call; if you need"
                                " different data, change the arguments, otherwise"
                                " act on what you already have."
                            )
                        }
                    )
                state.spiral.note_success(content)
                # Control verbs are not WORK: a revoked finish_session that
                # counted here would reset the standing fruitless streak
                # every round, and standing_patience could never engage.
                if name not in ("finish_session", "finish_planning"):
                    state.ok_tool_calls += 1
                self._note_jail_exec_failure(state, turn, name, tool_input, result)
                # Only a DISPATCHED finish counts: a refused finish tool (mode
                # backstop, schema error) is an error result the model recovers
                # from, not an end to the run.
                self._capture_finish(turn, name, tool_input)
            except ToolError as exc:
                content = self._note_tool_error(state, name, tool_input, exc)
                self._maybe_tool_error_ladder(state, turn)
            except OperatorCommandUnexecutable as exc:
                return self._unexecutable_abort(
                    exc, iteration=turn.iteration, tool_calls=state.tool_calls
                )
            turn.tool_results.append(
                ToolResultItem(
                    tool_use_id=tu.id,
                    content=cap_tool_result(
                        served if served is not None else content,
                        tool_name=name,
                        cap=self.tool_result_cap_chars,
                    ),
                    for_call=tu,
                )
            )
        # The gate run the harness adds to the turn, after the model's calls.
        return self._turn_harness_verify(state, turn)

    # A jailed command that hit its timeout, per sandbox.jail's contract.
    _EXIT_TIMEOUT = 124

    def _judged_the_base_commit(self, state: LoopState, result: ExecResult) -> bool:
        """True when this verify judged the commit the RUN started from.

        "The model has not edited yet" is the wrong test: every reason an
        operator resumes -- a budget stop, an iteration cap, a provider error --
        commits the leg's work first, so leg two opens on a clean tree whose
        HEAD already carries leg one's breakage, and reading that as the base
        would tell the worker its own failures are inherited. `/parallel` does
        the same by merging lane commits into the workspace.

        So: HEAD must still BE the base commit, the tree must be clean, and the
        gate must have actually produced a verdict -- a runner that was absent
        (instant exit) or timed out (124) never judged anything, and recording
        either would excuse every real failure for the rest of the run.

        A run that has already made the gate GREEN is answerable for a later
        red: it demonstrably could pass.

        Fails CLOSED. Every other user of `_worktree_dirty` treats an
        unreadable git as "assume clean"; here that would be a false
        exoneration, so an unreadable git records nothing.
        """
        if (
            state.verify.ever_passed
            or result.exec_failed
            or result.returncode == self._EXIT_TIMEOUT
            or verify_did_not_run(result.stdout, result.stderr, result.duration_s)
            or not self.base_sha
        ):
            return False
        try:
            status = git_status(self.root, exclude=self.untracked_at_start)
        except (GitError, OSError):
            return False
        return status.is_clean and status.head_sha == self.base_sha

    def _note_verify_result(self, state: LoopState, turn: TurnState, result: ExecResult) -> None:
        """Verify bookkeeping: pass/fail flags, the grounding tail, and the
        no-progress streak (consecutive fails sharing one signature)."""
        rc = result.returncode
        verdict = state.verify
        if rc == 0:
            turn.verify_just_passed = True
            if verdict.last_ok is False:
                turn.verify_flipped_green = True
                if paths := self._test_only_paths_since_red(verdict.red_tree):
                    turn.tool_results.append(Notice(test_only_green_notice(paths)))
                    self._log(f"  verify flipped green over test-only edits: {' '.join(paths)}")
                    self._emit(
                        "loop.test_only_green.notice", iteration=turn.iteration, paths=list(paths)
                    )
            # This verify validated the current tree; any earlier
            # edit is now covered.
            turn.edit_since_verify_pass = False
        else:
            turn.verify_just_failed = True
            if verdict.adopted and (
                why := unrunnable_signature(verdict.adopted, rc, result.stdout, result.stderr)
            ):
                # An ADOPTED gate that cannot run here: un-adopt (the run is
                # gateless again, the argv never re-adopted) and say so. A
                # configured gate stays a loud red.
                cmd = " ".join(verdict.adopted)
                verdict.unadoptable.add(verdict.adopted)
                verdict.adopted = ()
                # The gate produced no verdict and no longer exists: the turn
                # is not "verify failed" (an on_verify_fail panel and the
                # checkpoint logic key on it).
                turn.verify_just_failed = False
                self.config = self.config.with_verify_command(())
                self.dispatcher.drop_verify_command()
                self._log(f"LOOP: verify un-adopted ({why}): {cmd}")
                self._emit(
                    "loop.verify_inferred",
                    command=[],
                    source="unadopted",
                    adopted_at=turn.iteration,
                )
                turn.tool_results.append(Notice(VERIFY_UNADOPTED_NOTICE.format(cmd=cmd, why=why)))
                return
            # A verify that exited instantly without running any tests (runner
            # absent) is a broken verify, not a real failure: flag it once so
            # the model does not "fix" working code or finish unchecked.
            if not verdict.broken_warned and verify_did_not_run(
                result.stdout, result.stderr, result.duration_s
            ):
                verdict.broken_warned = True
                turn.tool_results.append(Notice(VERIFY_BROKEN_NUDGE))
                self._emit("loop.verify_broken.nudge", iteration=turn.iteration)
        if verdict.baseline_ok is None and self._judged_the_base_commit(state, result):
            # This verify judged the run's BASE commit, so it IS the
            # baseline: no second gate run is needed to learn the same answer.
            verdict.baseline_ok = rc == 0
            self._emit("loop.baseline", ok=rc == 0, iteration=turn.iteration)
            if rc != 0:
                turn.tool_results.append(Notice(BASELINE_RED_NOTICE))
        tail = f"{result.stdout}\n{result.stderr}"
        verdict.last_tail = tail.strip()[-2000:]
        if rc == 0:
            verdict.note_pass()
            state.no_progress_nudges_used = 0
            return
        verdict.note_fail(verify_failure_signature(result.stdout, result.stderr))
        verdict.red_tree = self._worktree_tree_sha()
        if verdict.fail_streak == 1:
            # A NEW stuck point: the nudge allowance starts over with it.
            state.no_progress_nudges_used = 0

    def _left_the_tree_dirty(self, name: str) -> bool:
        """True when a child-process tool left uncommitted changes behind.

        Only `run_command` and MCP tools are asked: `run_verify_command` and
        `run_metric_command` are the operator's own gates, and the caches they
        drop must not invalidate the pass they just produced. Git itself decides,
        so a read-only probe (`ls`, `grep`) costs its pass nothing --
        and gitignored build artifacts never count as a change.
        """
        if name != "run_command" and not name.startswith(MCP_TOOL_PREFIX):
            return False
        # Chain-relative (see _worktree_dirty): against a fixed HEAD every
        # already-committed step would count as "left dirty" forever.
        return self._worktree_dirty()

    def _note_tool_effects(
        self,
        state: LoopState,
        turn: TurnState,
        name: str,
        result: ToolResult,
        tool_input: Any,
    ) -> None:
        """Record a dispatched tool's side effects on the turn: verify results
        (they feed auto-commit-on-verify-pass and ground the review panel:
        verify-pass presumes correctness, verify-red is the hard signal),
        manual metric samples, tree edits, and DAG mutations."""
        if name == "ask_user" and isinstance(result, AnswersResult):
            questions = tool_input.get("questions") if isinstance(tool_input, dict) else None
            for q, answer in zip(questions or [], result.answers, strict=False):
                text = q.get("question", "") if isinstance(q, dict) else str(q)
                self._record_decision(state, str(text), answer)
        if name == "run_verify_command" and isinstance(result, ExecResult):
            # The model's own gate overran its budget: the same scoped
            # follow-up the harness gate gets, whose verdict is the turn's
            # (the 124 is not noted beside it); under `never` the harness
            # runs nothing and the timeout is the verdict.
            if not (
                self.mode == "run"
                and self.config.workflow.verify_when != "never"
                and result.returncode == self._EXIT_TIMEOUT
                and not state.verify.scoped
                and self._scoped_gate_followup(state, turn) is not None
            ):
                if result.returncode == 0:
                    # The model's call runs the full argv: a green there is a
                    # full pass, so later harness gates run full again.
                    state.verify.scoped = False
                self._note_verify_result(state, turn, result)
        elif name == "run_metric_command" and isinstance(result, MetricResult):
            if turn.verify_just_passed:
                turn.metric_after_verify_pass = True
            turn.metric_feedback = self._record_metric_result(
                state.metric_history,
                result,
                iteration=turn.iteration,
                label=f"manual iter {turn.iteration}",
                sha="",
            )
            if turn.verify_just_passed:
                turn.metric_plateau_finish = self._plateau_finish(state.metric_history)
        if name in ("apply_edit", "apply_patch"):
            # An edit under the memory dir is a memory write, not workspace
            # work: both memory nudges stay quiet for the rest of the run and
            # none of the tree bookkeeping below applies (the gate's tree is
            # untouched). Judged on the MODEL'S input path: the store sits
            # outside the workspace root, so only an absolute path reaches it
            # (EditResult.path is store-relative; matching on it never fires,
            # and the write would fall through to the tree bookkeeping).
            raw_path = str(tool_input.get("path", "")) if isinstance(tool_input, dict) else ""
            if (
                self.state_dir is not None
                and raw_path.startswith("/")
                and Path(raw_path).resolve().is_relative_to(memory_dir(self.state_dir))
            ):
                state.memory_written = True
                return
            turn.edited = True
            state.ever_edited = True
            # Invalidate a same-turn earlier verify pass: the commit
            # gate must not label this edited tree "verify passed".
            turn.edit_since_verify_pass = True
            state.verify.note_edit()
        elif self._left_the_tree_dirty(name):
            # A command (or an MCP tool) can change the tree just as an edit
            # tool can, and a green verify must not survive it: the tree the
            # gate approved is no longer the tree we have. Asked of git rather
            # than assumed from the tool name, so a read-only `ls` or `grep`
            # through run_command keeps the pass it had.
            turn.edit_since_verify_pass = True
            state.verify.note_edit()
        if name in DAG_MUTATING_TOOLS:
            turn.dag_mutated = True  # snapshot once after the turn

    def _capture_finish(self, turn: TurnState, name: str, tool_input: Any) -> None:
        """Capture a finish_session / finish_planning call's summary + payload on
        the turn (the finish gates may still revoke it). finish_planning also
        persists the plan markdown: schema validation already guaranteed the
        field when the dispatcher dispatched it, but the raw tool_input is what
        the model sent us, so stay defensive against a malformed call."""
        if name == FinishSessionInput.TOOL_NAME:
            turn.finish_kind = "finish_session"
            turn.finish_signal = (
                tool_input.get("summary", "(no summary)")
                if isinstance(tool_input, dict)
                else "(no summary)"
            )
            raw_result = tool_input.get("result") if isinstance(tool_input, dict) else None
            if isinstance(raw_result, str):
                # Weak models routinely STRINGIFY the structured result; one
                # tolerant parse here, schema validation downstream stays
                # strict about content.
                try:
                    raw_result = json.loads(raw_result)
                except ValueError:
                    raw_result = None
            turn.finish_payload = raw_result if isinstance(raw_result, dict) else None
            turn.finish_stale_gate = (
                str(tool_input.get("stale_gate", "")).strip()
                if isinstance(tool_input, dict)
                else ""
            )
        elif name == FinishPlanningInput.TOOL_NAME:
            turn.finish_kind = "finish_planning"
            turn.finish_signal = (
                tool_input.get("summary", "(no summary)")
                if isinstance(tool_input, dict)
                else "(no summary)"
            )
            plan_md = ""
            summary = ""
            if isinstance(tool_input, dict):
                plan_md = str(tool_input.get("plan_markdown", ""))
                summary = str(tool_input.get("summary", ""))
            # Salvage a title-only plan_markdown: weak models put the real plan
            # in `summary`, leaving plan.md a stub that --from-plan must
            # re-derive. Fold the summary under the title so the plan
            # carries content. The review gate judged content quality; this only
            # rescues field misuse.
            if _plan_is_title_only(plan_md) and len(summary) > len(plan_md):
                title = next((ln for ln in plan_md.splitlines() if ln.strip()), "# Plan")
                plan_md = f"{title}\n\n{summary}"
                self._log("  plan salvaged: folded summary into a title-only plan_markdown")
            if self.plan_output_path is not None and plan_md:
                try:
                    self.plan_output_path.parent.mkdir(parents=True, exist_ok=True)
                    self.plan_output_path.write_text(plan_md, encoding="utf-8")
                    self._log(f"  plan written: {self.plan_output_path} ({len(plan_md)} chars)")
                    self._emit(
                        "loop.plan_written",
                        path=str(self.plan_output_path),
                        bytes=len(plan_md.encode("utf-8")),
                    )
                except OSError as exc:
                    self._log(f"  plan write failed: {exc}")
                    self._emit(
                        "loop.plan_write.failed",
                        path=str(self.plan_output_path),
                        error=str(exc),
                    )

    def _maybe_adopt_verify(self, state: LoopState, turn: TurnState) -> None:
        """A gateless run that commits has just materialized project files the
        preflight inference never saw (an empty repo infers nothing, then the
        run creates a pyproject two minutes later and finishes ungated). Re-run
        the DETERMINISTIC inference tiers (an AGENTS.md fence, repo signals;
        never the LLM tier) at each gateless commit until one lands, then adopt
        it for the rest of the run: the loop's gates, the dispatcher's
        run_verify_command, and the resume snapshot all read the adopted
        command. The model is told, so the gate flip is never silent; first
        adoption wins (the config gaining a command ends the gateless branch).
        `verify_infer = false` pins gatelessness: no adoption either."""
        if not self.config.workflow.verify_infer:
            return
        inferred = infer_verify_command(self.root, read_agents_md(self.root), llm_call=None)
        if inferred is None or inferred.argv in state.verify.unadoptable:
            return
        if not self.dispatcher.adopt_verify_command(inferred.argv):
            # An inferred runner the jail cannot execute: adopting it would
            # turn the honest settle into an unexecutable-verify abort. Stay
            # gateless; re-inferred (and re-declined) at the next commit.
            self._log(f"LOOP: verify inference declined; {inferred.argv[0]} not on the jail PATH")
            return
        self.config = self.config.with_verify_command(inferred.argv)
        state.verify.adopted = inferred.argv
        cmd = " ".join(inferred.argv)
        self._log(f"LOOP: verify adopted from {inferred.source}: {cmd}")
        self._emit(
            "loop.verify_inferred",
            command=list(inferred.argv),
            source=inferred.source,
            adopted_at=turn.iteration,
        )
        turn.tool_results.append(
            Notice(
                "[harness] The repo now has a recognizable project, so a verify"
                f" command was adopted and gates the rest of this run: `{cmd}`."
                " Run run_verify_command to check your work."
            )
        )

    def _turn_harness_verify(
        self, state: LoopState, turn: TurnState, *, ending: bool = False
    ) -> SessionResult | None:
        """Run the gate the harness owes this turn (`[workflow].verify_when`):
        after an editing turn under `step`, and when the run is ending (a
        finish_session, or `ending`: an end the harness declares) over a tree
        no green run covers under `step` or `finish`. The model's own
        run_verify_command this turn already judged the tree, so nothing runs
        on top of it. `run_commands = "no"` withholds the gate from the
        harness as it does from the model, and a DENIED gate (ask: a human's
        no, or the unattended auto-deny) is withheld for the rest of the run
        the same way. An unexecutable operator command
        ends the run as it does on the tool path."""
        why = harness_verify_due(
            when=self.config.workflow.verify_when,
            gate_present=(
                self.mode == "run"
                and bool(self.config.workflow.verify_command)
                and self.dispatcher.command_policy() != "no"
                and not state.verify.denied
            ),
            # An edit after the turn's own verify un-verifies it: the gate
            # judged a tree that no longer exists.
            verified_this_turn=(
                (turn.verify_just_passed or turn.verify_just_failed)
                and not turn.edit_since_verify_pass
            ),
            changed_this_turn=turn.edit_since_verify_pass,
            finishing=ending
            or (turn.finish_signal is not None and turn.finish_kind == "finish_session"),
            green_and_untouched=state.verify.green_and_untouched,
        )
        if why is None:
            return None
        self._log(f"LOOP: harness verify ({why}) at iter {turn.iteration}")
        self._emit("loop.verify_harness", why=why, iteration=turn.iteration)
        try:
            scope = self._gate_scope_paths() if state.verify.scoped else ()
            result = self.dispatcher.run_verify(extra_argv=scope)
        except ToolDenied as exc:
            state.verify.denied = True
            turn.tool_results.append(Notice(gate_withheld_notice(f"[harness verify] {why}", exc)))
            return None
        except ToolError as exc:
            turn.tool_results.append(Notice(f"[harness verify] {why}: not run: {exc}"))
            return None
        except OperatorCommandUnexecutable as exc:
            return self._unexecutable_abort(
                exc, iteration=turn.iteration, tool_calls=state.tool_calls
            )
        if (
            result.returncode == self._EXIT_TIMEOUT
            and not state.verify.scoped
            and self._scoped_gate_followup(state, turn) is not None
        ):
            return None
        self._note_verify_result(state, turn, result)
        notice = (
            scoped_verify_notice(
                result, timeout_s=self.config.workflow.verify_timeout_s, paths=scope
            )
            if scope
            else harness_verify_notice(result, why)
        )
        turn.tool_results.append(Notice(notice))
        return None

    def _gate_scope_paths(self) -> tuple[str, ...]:
        """The scoped-gate selection: tests nearest the run's cumulative diff.
        Empty unless the gate is a pytest argv naming no paths (the one shape
        that takes appended test files as its selection), or when nothing
        near the change exists to run."""
        if not is_bare_pytest(tuple(self.config.workflow.verify_command)):
            return ()
        return nearest_test_paths(self.root, diff_changed_paths(self._run_diff()))

    def _scoped_gate_followup(self, state: LoopState, turn: TurnState) -> ExecResult | None:
        """The scoped re-run after a full gate overran its budget, wherever
        that gate ran (the harness's own, or the model's run_verify_command):
        the same command over the tests nearest the run's diff, noted and
        noticed like any gate run. Arms `verdict.scoped`, so later harness
        gates skip the doomed full run. None when the gate is not pytest or
        nothing near the change exists to run."""
        scope = self._gate_scope_paths()
        if not scope:
            return None
        state.verify.scoped = True
        self._log(f"LOOP: verify overran; gate scoped to {len(scope)} test files")
        self._emit("loop.verify_scoped", paths=list(scope), iteration=turn.iteration)
        try:
            result = self.dispatcher.run_verify(extra_argv=scope)
        except ToolDenied as exc:
            state.verify.denied = True
            turn.tool_results.append(Notice(gate_withheld_notice("[verify] scoped re-run", exc)))
            return None
        except ToolError as exc:
            turn.tool_results.append(Notice(f"[verify] scoped re-run not run: {exc}"))
            return None
        self._note_verify_result(state, turn, result)
        turn.tool_results.append(
            Notice(
                scoped_verify_notice(
                    result, timeout_s=self.config.workflow.verify_timeout_s, paths=scope
                )
            )
        )
        return result

    def _turn_auto_commit_and_metric(
        self, state: LoopState, turn: TurnState
    ) -> SessionResult | None:
        """Auto-commit the turn's work, then take the automatic metric sample.

        A step the gate judged green commits as a verified step; every other
        editing step (a gateless run, and under `verify_when = "finish"` a step
        the model did not verify itself) commits as an un-gated checkpoint, so
        resume and the audit trail still work.
        `turn.edited` (apply_edit/apply_patch) is the cheap fast-path; the
        worktree-dirty fallback catches run_command-authored edits (else they'd
        never be committed gateless). Plan mode is read-only and never commits.
        Best-effort: commit failures (e.g. nothing to commit) are logged but
        don't abort the run; the catch includes OSError so a transient FS
        hiccup doesn't kill an otherwise-fine run.

        Returns a SessionResult for the REPL hook's "stop" directive or an
        unexecutable operator metric command; None otherwise."""
        gateless = not self.config.workflow.verify_command
        # A step no gate judged commits as a checkpoint: every gateless step,
        # and under `verify_when = "finish"` every step the model did not
        # verify itself (the gate certifies the tree the run ends on).
        unjudged = gateless or (
            self.config.workflow.verify_when == "finish"
            and not (turn.verify_just_passed or turn.verify_just_failed)
        )
        unjudged_changed = unjudged and (turn.edited or self._worktree_dirty())
        verified_commit = turn.verify_just_passed and not turn.edit_since_verify_pass
        if self.mode != "run" or not (verified_commit or unjudged_changed):
            return None
        if not self.commit_per_step:
            # `commit_per_step` governs the COMMIT. The metric is measurement:
            # the prompt promises a [harness metric] block after every verified
            # edit, and with the sampling behind this gate the model was pushed
            # to optimise a number it was never shown.
            return self._sample_metric(state, turn, sha="")
        commit_subject = self._checkpoint_subject(
            turn, fallback="checkpoint" if unjudged_changed else "verify passed"
        )
        sha = ""
        try:
            sha = self._chain_commit(commit_subject)
            if sha:
                # "" is chain_commit's nothing-changed answer (a green verify
                # with no new edits); an event or log line for it would claim
                # a commit that never happened.
                self._log(f"  auto-commit: {sha[:12]}")
                self._emit(
                    "loop.auto_commit", iteration=turn.iteration, sha=sha, subject=commit_subject
                )
            turn.committed = bool(sha)
            if unjudged_changed and sha:
                # Seed the idle-stop net for runs where no green verify
                # fires per step; see the verify-settled bookkeeping.
                state.gateless_ever_committed = True
            if gateless and sha:
                self._maybe_adopt_verify(state, turn)
            if sha:
                # Surface "what the worker just changed" to a live viewer
                # (the TUI diff panel). Capped; best-effort.
                self._emit(
                    "diff.updated",
                    sha=sha,
                    patch=commit_diff(self.root, sha, max_bytes=8000),
                )
        except (GitError, OSError) as exc:
            self._report_auto_commit_failure(exc, commit_subject, iteration=turn.iteration)
        # REPL hook. Default no-op returns "continue".
        if sha:
            directive = self.after_auto_commit(turn.iteration, sha)
            if directive in ("undo", "exit"):
                ended = self._steer_outcome(directive, turn.iteration, state)
                if ended is not None:
                    return ended
            if directive == "stop":
                self._log(f"LOOP: interactive stop at iter {turn.iteration}")
                # An operator stop is deliberate, not verified success: the
                # same truth rule as steer_abort ("stopped", never "passed").
                self._pass_pending_root_tasks()
                self._emit(
                    "session.end",
                    reason="interactive_stop",
                    iterations=turn.iteration,
                    all_passed=False,
                )
                return SessionResult(
                    completed=True,
                    verified=self._verification(state),
                    reason="interactive_stop",
                    summary=(
                        f"stopped interactively after iter {turn.iteration}"
                        f"{self._dirty_tree_note()}"
                    ),
                    iterations=turn.iteration,
                    tool_calls=state.tool_calls,
                )
        return self._sample_metric(state, turn, sha=sha)

    def _sample_metric(
        self, state: LoopState, turn: TurnState, *, sha: str
    ) -> SessionResult | None:
        """Run the configured metric over the step just taken and hand the
        model the reading, unless it ran the metric itself this turn."""
        if turn.metric_after_verify_pass:
            return None
        # The auto path raises OperatorCommandUnexecutable just like a
        # manual run_metric_command would; abort the same way the
        # per-tool handler does (it is a distinct exception, NOT a
        # ToolError, so _auto_metric_feedback does not swallow it).
        try:
            turn.metric_feedback = self._auto_metric_feedback(
                state.metric_history,
                iteration=turn.iteration,
                sha=sha,
            )
        except OperatorCommandUnexecutable as exc:
            return self._unexecutable_abort(
                exc, iteration=turn.iteration, tool_calls=state.tool_calls
            )
        turn.metric_plateau_finish = self._plateau_finish(state.metric_history)
        return None

    def _report_auto_commit_failure(
        self, exc: GitError | OSError, commit_subject: str, *, iteration: int
    ) -> None:
        """Log + emit a non-benign auto-commit failure with a worktree status
        snapshot, so the event payload tells the operator what was in the tree
        at the failure point. "nothing to commit" variants are benign and stay
        silent: the phrase can arrive in either the stdout or the stderr half
        of the detail string (see git_ops._run); "no changes added" covers the
        variant when only paths outside the worktree (or .gitignore'd) changed;
        "working tree clean" covers a verify pass without any file mutation."""
        msg = str(exc).lower()
        benign = (
            "nothing to commit" in msg or "no changes added" in msg or "working tree clean" in msg
        )
        if benign:
            return
        self._log(f"  auto-commit failed: {exc}")
        # Best-effort: if status itself raises (rare; the outside-a-repo case
        # is already gone by this point in the loop), omit the snapshot.
        worktree_status = ""
        try:
            st = git_status(self.root, exclude=self.untracked_at_start)
            worktree_status = (
                f"branch={st.branch}"
                f" head={st.head_sha[:12]}"
                f" clean={st.is_clean}"
                f" modified={st.modified_count}"
                f" untracked={st.untracked_count}"
            )
        except (GitError, OSError):
            pass
        self._emit(
            "loop.auto_commit.failed",
            iteration=iteration,
            error=str(exc)[:2000],
            worktree_status=worktree_status,
            commit_subject=commit_subject[:200],
        )

    def _turn_review_triggers(
        self, state: LoopState, turn: TurnState, conversation: Conversation
    ) -> None:
        """Observe-only review triggers (before_finish, which can revoke a
        finish, lives in the finish gates):

          on_verify_fail - the verify just failed; surface a critique
                           alongside the failure so the worker has a second
                           opinion before its next edit.
          periodic       - every review_period iterations.
        """
        if (
            self.review_trigger == "on_verify_fail"
            and turn.verify_just_failed
            and self._has_reviewer()
        ):
            critique = self._run_review_panel(
                state, trigger="verify_failed", iteration=turn.iteration
            )
            if critique is not None:
                turn.review_text = critique.text
        elif (
            self.review_trigger == "periodic"
            and self._has_reviewer()
            and turn.iteration % max(1, self.review_period) == 0
        ):
            critique = self._run_review_panel(state, trigger="periodic", iteration=turn.iteration)
            if critique is not None:
                turn.review_text = critique.text

    # ---- finish gates ----------------------------------------------------------

    def _turn_finish_gates(
        self, state: LoopState, turn: TurnState, conversation: Conversation
    ) -> None:
        """Gates that can revoke this turn's finish_session, in precedence order:
        the finish contract (a machine state's output_schema), review
        (before_finish), metric early-finish, open subtasks, verify
        green, memory backstop. Each clears `turn.finish_signal` and appends
        its nudge; later gates then see the finish as already revoked and stay
        quiet."""
        self._gate_finish_contract(turn)
        self._gate_before_finish_review(state, turn, conversation)
        self._gate_metric_early_finish(state, turn)
        self._gate_task_finish(state, turn)
        self._gate_verify_finish(state, turn)
        self._gate_memory_finish(state, turn)
        self._gate_standing_finish(state, turn)

    def _gate_before_finish_review(
        self, state: LoopState, turn: TurnState, conversation: Conversation
    ) -> None:
        """Gate the agent's finish_session on panel approval: an unsatisfied
        verdict suppresses the finish (the tool_result still goes back so the
        call isn't half-applied) and the loop carries on with the findings
        visible."""
        del conversation
        if not (
            turn.finish_signal is not None
            and turn.finish_kind == "finish_session"
            and self._end_is_reviewed(state, turn, ending="finish_session")
        ):
            return
        turn.finish_signal = None
        turn.finish_payload = None

    def _end_is_reviewed(self, state: LoopState, turn: TurnState, *, ending: str) -> bool:
        """The before-finish panel over an end (`finish_session`, or the
        settled stop the harness declares): True when the panel rejected it
        and the run carries on with the findings injected. After
        `max_consecutive_review_rejections` back-to-back rejections the end
        goes through (findings still injected) so the worker can't bounce
        indefinitely. False when there is no panel or it approved."""
        if not (self.review_trigger == "before_finish" and self._has_reviewer()):
            return False
        critique = self._run_review_panel(state, trigger="before_finish", iteration=turn.iteration)
        if critique is None:
            return False
        cap = self.max_consecutive_review_rejections
        cap_reached = cap > 0 and state.consecutive_review_rejections >= cap
        if not critique.satisfied and not cap_reached:
            self._log(f"  review rejected {ending} at iter {turn.iteration}")
            self._emit("loop.review.rejected_finish", iteration=turn.iteration, ending=ending)
            state.consecutive_review_rejections += 1
            turn.review_text = (
                (
                    "The review panel rejected your finish_session call. Address the"
                    " issues below before calling finish_session again.\n\n"
                )
                if ending == "finish_session"
                else (
                    "The review panel rejected your silent finish (no tool_use, just"
                    " text). Address the issues below and continue the task.\n\n"
                )
                if ending == "silent_finish"
                else (
                    "The review panel rejected the settled end. Address the issues"
                    " below; the run ends when it settles again or finish_session"
                    " passes.\n\n"
                )
            ) + critique.text
            return True
        if not critique.satisfied:
            self._log(
                f"  review rejected {ending} at iter {turn.iteration} but"
                f" rejection cap ({cap}) reached - letting the end through"
            )
            self._emit(
                "loop.review.rejection_cap_reached",
                iteration=turn.iteration,
                rejections=state.consecutive_review_rejections,
            )
            turn.review_text = (
                "The review panel flagged issues but the rejection cap was"
                " reached; the end stands. Findings:\n\n" + critique.text
            )
        else:
            self._log(f"  review approved {ending}")
        state.consecutive_review_rejections = 0
        return False

    def _gate_metric_early_finish(self, state: LoopState, turn: TurnState) -> None:
        """Metric-run early-finish guard. On optimisation runs the worker often
        calls finish_session with most of its budget unspent, even though the task
        asks it to keep optimising up to the cap. Mirror the plateau policy:
        while the run still has runway above the final budget slice, reject an
        early finish_session a few times and nudge the worker to keep going; only
        honour it once we are in the final budget slice or patience is
        exhausted. Requires a real budget signal - with none (tests / MCP) we
        defer to the worker's own judgement so a finish can never deadlock."""
        if not (
            turn.finish_signal is not None
            and turn.finish_kind == "finish_session"
            and self.mode == "run"
            and metric_goal(self.config.workflow.metric) is not None
            and not self._metric_at_ceiling(state.metric_history)
        ):
            return
        finish_budget_remaining = self._budget_fraction_remaining()
        has_runway = (
            finish_budget_remaining is not None
            and finish_budget_remaining > METRIC_PLATEAU_STOP_BELOW_BUDGET
        )
        if has_runway and state.metric_finish_nudges_used < METRIC_EARLY_FINISH_PATIENCE:
            assert finish_budget_remaining is not None
            state.metric_finish_nudges_used += 1
            turn.finish_signal = None
            turn.finish_payload = None
            turn.tool_results.append(Notice(METRIC_FINISH_NUDGE))
            self._log(
                f"  metric early-finish rejected #{state.metric_finish_nudges_used}"
                f" at iter {turn.iteration} (budget {finish_budget_remaining:.0%} left)"
            )
            self._emit(
                "loop.metric_early_finish.rejected",
                iteration=turn.iteration,
                nudges_used=state.metric_finish_nudges_used,
                budget_remaining=finish_budget_remaining,
            )

    def _gate_task_finish(self, state: LoopState, turn: TurnState) -> None:
        """Task finish-gate: don't let finish_session through while the worker's
        own subtasks are still open (capped; see _task_finish_gate_nudge)."""
        if not (
            turn.finish_signal is not None
            and turn.finish_kind == "finish_session"
            and self.mode == "run"
        ):
            return
        task_nudge = self._task_finish_gate_nudge(state)
        if task_nudge is None:
            return
        turn.finish_signal = None
        turn.finish_payload = None
        turn.tool_results.append(Notice(task_nudge))
        self._log(
            f"  finish_session gated: open subtasks remain (nudge"
            f" #{state.task_finish_nudges_used}) at iter {turn.iteration}"
        )
        self._emit(
            "loop.task_finish.gated",
            iteration=turn.iteration,
            nudges_used=state.task_finish_nudges_used,
        )

    def _gate_standing_finish(self, state: LoopState, turn: TurnState) -> None:
        """While a ready standing task exists, finish_session re-enters it
        instead of ending the run (uncapped -- the goal is deliberate; the
        absorb still refuses on spent budget or a spin, so the finish then
        goes through)."""
        if not (
            turn.finish_signal is not None
            and turn.finish_kind == "finish_session"
            and self.mode == "run"
        ):
            return
        nudge = self._standing_absorb(state, reason="finish_session", iteration=turn.iteration)
        if nudge is None:
            return
        turn.finish_signal = None
        turn.finish_payload = None
        turn.tool_results.append(Notice(nudge))

    def _gate_finish_contract(self, turn: TurnState) -> None:
        """A finish_session whose `result` does not satisfy the machine state's
        output_schema returns to the model with the problems, so the retry
        happens in-leg instead of the leg ending failed over correct work.
        Unbounded on purpose: the budget and max_iterations backstops end a
        model that never conforms, and the engine records that truthfully."""
        if (
            self.finish_validator is None
            or turn.finish_signal is None
            or turn.finish_kind != "finish_session"
        ):
            return
        problems = self.finish_validator(turn.finish_payload)
        if not problems:
            return
        turn.finish_signal = None
        turn.finish_payload = None
        turn.tool_results.append(
            Notice(
                "finish_session refused: "
                + "; ".join(problems)
                + ". Call finish_session again with a `result` that satisfies the schema."
            )
        )
        self._log(
            f"  finish_session returned: result violates the contract at iter {turn.iteration}"
        )
        self._emit("loop.finish_contract.refused", iteration=turn.iteration, problems=problems)

    def _gate_verify_finish(self, state: LoopState, turn: TurnState) -> None:
        """A finish over a tree the gate did not certify returns to the model
        `verify_retries` times, then stands (reported finished, never passed;
        the honest all_passed=False in the stop checks applies either way).
        A gate that was red before the run touched anything is not the
        model's to fix, so it is never returned."""
        wf = self.config.workflow
        if not (
            turn.finish_signal is not None
            and turn.finish_kind == "finish_session"
            and self.mode == "run"
            and wf.verify_when != "never"
            and wf.verify_command
            and self._tree_is_verify_green(state) is False
            and state.verify.baseline_ok is not False
            and state.verify_finish_retries_used < wf.verify_retries
            and self.dispatcher.command_policy() != "no"
            and not state.verify.denied
        ):
            return
        state.verify_finish_retries_used += 1
        turn.finish_signal = None
        turn.finish_payload = None
        turn.tool_results.append(
            Notice(
                finish_red_notice(used=state.verify_finish_retries_used, retries=wf.verify_retries)
            )
        )
        self._log(
            f"  finish_session returned: verify not green (return"
            f" #{state.verify_finish_retries_used} of {wf.verify_retries}) at iter"
            f" {turn.iteration}"
        )
        self._emit(
            "loop.verify_finish.gated",
            iteration=turn.iteration,
            nudges_used=state.verify_finish_retries_used,
        )

    def _gate_memory_finish(self, state: LoopState, turn: TurnState) -> None:
        """Memory write-side backstop: defer the first finish_session ONCE when the
        run recovered from a red verify to green and recorded nothing via
        a memory write - the nudge asks for the root cause or an immediate re-finish
        (see _nudges for the measurement behind it). Explicit finish_session only: a
        went-quiet worker is never bounced here."""
        if not (
            turn.finish_signal is not None
            and turn.finish_kind == "finish_session"
            and self.mode == "run"
            and self.state_dir is not None
            and state.verify.ever_failed
            and state.verify.last_ok is True
            and not state.memory_written
            and not state.memory_finish_nudged
        ):
            return
        state.memory_finish_nudged = True
        turn.finish_signal = None
        turn.finish_payload = None
        turn.tool_results.append(Notice(MEMORY_FINISH_NUDGE))
        self._log(f"  finish_session deferred once: memory backstop at iter {turn.iteration}")
        self._emit("loop.memory_finish.gated", iteration=turn.iteration)

    # ---- turn notices and spiral guards ----------------------------------------

    def _turn_notices(self, state: LoopState, turn: TurnState) -> None:
        """Append the turn's advisory texts to the tool_results block: review
        findings, metric feedback, the memory flip advisory, then the
        degenerate-loop notice.

        The memory flip advisory fires once per run, at the first verify that
        goes green after a red one, while nothing has been recorded via
        a memory write: that is the moment a hard-won root cause is in hand (see
        _nudges for the measurement behind it).

        The loop-guard notice fires when the same (tool, args) signature has
        been called >= 3 times in a row, re-emitted once per "fresh" streak
        (when a new streak crosses the threshold) so spamming the same call
        only triggers once per latch episode. The repeat counter resets on any
        new signature, so a normal re-read after an edit does not trigger."""
        if turn.review_text:
            turn.tool_results.append(Notice(f"[review]\n{turn.review_text}"))
        if turn.metric_feedback:
            turn.tool_results.append(Notice(turn.metric_feedback))
        if (
            turn.verify_flipped_green
            and self.mode == "run"
            and self.state_dir is not None
            and not state.memory_written
            and not state.memory_flip_nudged
        ):
            state.memory_flip_nudged = True
            turn.tool_results.append(Notice(MEMORY_FLIP_NUDGE))
            self._log("  memory: verify flipped green - injecting memory advisory")
            self._emit("loop.memory_flip.nudged", iteration=turn.iteration)
        repeat_threshold = 3
        if (
            state.spiral.call_streak >= repeat_threshold
            and state.spiral.warned_at_iteration < turn.iteration - 1
        ):
            # Strip the args-JSON suffix for the user-facing text.
            latched_name = (state.spiral.last_call_sig or "").split(":", 1)[0] or "<unknown>"
            notice = (
                f"[loop-guard] You have called `{latched_name}` with"
                f" identical arguments {state.spiral.call_streak} times in a row."
                " The tool result has not changed. Re-issuing the same"
                " call again will not yield new information. Change"
                " your approach: try different arguments, a different"
                " tool, commit to an edit, or call `finish_session` if"
                " you have already done what the task requires."
            )
            turn.tool_results.append(Notice(notice))
            self._emit(
                "loop.loop_guard.triggered",
                iteration=turn.iteration,
                tool=latched_name,
                streak=state.spiral.call_streak,
            )
            self._log(
                f"  loop-guard: {latched_name} called"
                f" {state.spiral.call_streak}x in a row - injecting notice"
            )
            state.spiral.warned_at_iteration = turn.iteration
        attemptless = (
            self.stagnation_notice_after_s > 0
            and not state.stagnation_nudged
            and not state.ever_edited
            and state.verify.last_ok is None
            and self.mode == "run"
        )
        elapsed = time.monotonic() - state.started_monotonic
        if attemptless and elapsed >= self.stagnation_notice_after_s:
            # Time blocked on the operator is not the model's.
            elapsed -= self.dispatcher.operator_wait_s
        if attemptless and elapsed >= self.stagnation_notice_after_s:
            state.stagnation_nudged = True
            minutes = max(1, int(elapsed // 60))
            gated = bool(self.config.workflow.verify_command)
            notice = STAGNATION_NUDGE if gated else STAGNATION_NUDGE_GATELESS
            turn.tool_results.append(Notice(notice.format(minutes=minutes)))
            self._emit("loop.stagnation.nudged", iteration=turn.iteration, elapsed_s=int(elapsed))
            self._log(f"  stagnation: {minutes}m with no attempt - injecting notice")

    def _turn_metric_plateau(self, state: LoopState, turn: TurnState) -> None:
        """Metric-plateau handling. When a verified metric merely ties the
        prior best, the plateau detector fires. Rather than quit at the first
        stall (often with most of the budget unspent), nudge the worker to
        pivot to a different approach; only stop once we are in the final
        budget slice and have still failed to beat the best after a few pivot
        nudges. With no budget signal (tests / MCP) the fixed
        `METRIC_PLATEAU_PATIENCE` bounds the nudging. Sets
        `turn.plateau_should_stop`; the stop itself happens in the stop
        checks, after the post-tools snapshot."""
        if turn.metric_plateau_finish is None:
            return
        budget_remaining = self._budget_fraction_remaining()
        in_final_slice = (
            budget_remaining is None or budget_remaining <= METRIC_PLATEAU_STOP_BELOW_BUDGET
        )
        if self._metric_at_ceiling(state.metric_history):
            # A metric at its provable ceiling (e.g. SCORE: 27/27) cannot
            # improve: stop now rather than nudge the worker to "pivot" toward
            # a number that does not exist. This is the dominant cause of weak
            # reasoning models burning their whole budget (and wall-clock)
            # re-deriving a solved task.
            turn.plateau_should_stop = True
            self._emit("loop.metric_ceiling.stop", iteration=turn.iteration)
        elif in_final_slice and state.plateau_nudges_used >= METRIC_PLATEAU_PATIENCE:
            turn.plateau_should_stop = True
        else:
            # Count patience only against final-slice nudges. While the run
            # still has runway (in_final_slice False), keep nudging the
            # worker to explore without consuming the budget, exactly as the
            # early-finish guard only counts rejections while it has runway.
            # Counting runway ties here would exhaust METRIC_PLATEAU_PATIENCE
            # before the final slice, so the run would stop the instant the
            # budget crossed the threshold and the escalating FINAL
            # ("make your one best bet") nudge would never fire.
            if in_final_slice:
                state.plateau_nudges_used += 1
            nudge_text = metric_plateau_nudge(budget_remaining)
            turn.tool_results.append(Notice(nudge_text))
            budget_note = "n/a" if budget_remaining is None else f"{budget_remaining:.0%} left"
            self._log(
                f"  metric_plateau pivot-nudge at iter {turn.iteration} (budget"
                f" {budget_note}; final-slice patience"
                f" {state.plateau_nudges_used}/{METRIC_PLATEAU_PATIENCE})"
            )
            self._emit(
                "loop.metric_plateau.nudge",
                iteration=turn.iteration,
                nudges_used=state.plateau_nudges_used,
                budget_remaining=budget_remaining,
            )

    def _end_gates(self, state: LoopState, turn: TurnState, *, ending: str) -> SessionResult | None:
        """An end declared without finish_session (`settled`: the harness's
        idle stop; `silent_finish`: a prose turn with no tool call) passes the
        gates a finish_session would: the verify certification (`verify_when`)
        and the before-finish review panel. A red gate with returns left, or a
        rejected panel, hands the end back (`turn.end_returned`) with the
        output; the unexecutable-command abort ends the run as it does on the
        tool path."""
        aborted = self._turn_harness_verify(state, turn, ending=True)
        if aborted is not None:
            return aborted
        wf = self.config.workflow
        red_returned = (
            wf.verify_when != "never"
            and bool(wf.verify_command)
            and turn.verify_just_failed
            and state.verify.baseline_ok is not False
            and state.verify_finish_retries_used < wf.verify_retries
        )
        if red_returned:
            state.verify_finish_retries_used += 1
            turn.tool_results.append(
                Notice(
                    finish_red_notice(
                        used=state.verify_finish_retries_used, retries=wf.verify_retries
                    )
                )
            )
            self._emit(
                "loop.verify_finish.gated",
                iteration=turn.iteration,
                nudges_used=state.verify_finish_retries_used,
            )
        turn.end_returned = red_returned or self._end_is_reviewed(state, turn, ending=ending)
        return None

    def _turn_verify_settled(self, state: LoopState, turn: TurnState) -> SessionResult | None:
        """Verify-settled completion bookkeeping (run mode): count no-progress
        iterations after the first green verify; nudge once, then stop (the
        stop happens in the stop checks, via `turn.verify_settled_stop`).

        "Progress" is any forward motion the prompt encourages, so a
        legitimately-working run is never truncated: an apply_edit/apply_patch,
        a new commit, or an uncommitted worktree change (an edit made via
        run_command). A verify RUN itself (re-verifying between reads is active
        work, not idle) is held neutral so it neither resets nor accrues. Only
        the pathology, spinning on read-only commands with a clean,
        already-committed tree, accrues idle.

        Only governs PLAIN runs. A metric/optimisation run is also mode=="run"
        but its completion is owned by the metric early-finish guard +
        plateau/ceiling logic (which deliberately keep going while budget
        remains); measure/analyse/read iterations there legitimately make no
        commit, so the settled detector must defer to them. (Gating the
        bookkeeping here also keeps the worktree-dirty git check off the
        metric hot path.)"""
        non_metric_run = self.mode == "run" and metric_goal(self.config.workflow.metric) is None
        # "Settled" once the run reached a good state: a green verify, or (on a
        # gateless run, where verify never fires) a committed edit.
        settled_seeded = state.verify.ever_passed or state.gateless_ever_committed
        if non_metric_run and settled_seeded:
            made_progress = turn.committed or turn.edited or self._worktree_dirty()
            if made_progress:
                state.verify_settled_idle = 0
                state.verify_settled_nudged = False  # a fresh idle streak may re-nudge
            elif not (turn.verify_just_passed or turn.verify_just_failed):
                state.verify_settled_idle += 1
        turn.verify_settled_stop = (
            non_metric_run
            and turn.finish_signal is None
            and settled_seeded
            and state.verify_settled_idle >= VERIFY_SETTLED_STOP_AFTER
        )
        if turn.verify_settled_stop:
            aborted = self._end_gates(state, turn, ending="settled")
            if aborted is not None:
                return aborted
            if turn.end_returned:
                turn.verify_settled_stop = False
                state.verify_settled_idle = 0
                state.verify_settled_nudged = False
        if (
            non_metric_run
            and turn.finish_signal is None
            and not turn.verify_settled_stop
            and settled_seeded
            and state.verify_settled_idle >= VERIFY_SETTLED_NUDGE_AFTER
            and not state.verify_settled_nudged
        ):
            state.verify_settled_nudged = True
            turn.tool_results.append(Notice(VERIFY_SETTLED_NUDGE))
            self._emit(
                "loop.verify_settled.nudge",
                iteration=turn.iteration,
                idle=state.verify_settled_idle,
            )
        return None

    def _note_tool_error(
        self, state: LoopState, name: str, tool_input: dict[str, Any], exc: ToolError
    ) -> str:
        """Bookkeeping for one failed dispatch: the served error content, the
        denial/binary records the reachability note reads, and the
        same-signature streak the nudge ladder climbs."""
        content = json.dumps({"error": str(exc)})
        self._log(f"  tool_error: {name}: {exc}")
        state.spiral.note_error(
            tool_error_signature(name, str(exc)),
            denial=isinstance(exc, ToolDenied),
            content=content,
        )
        return content

    def _maybe_tool_error_ladder(self, state: LoopState, turn: TurnState) -> None:
        """Nudge/escalate/stop on a streak of identical tool errors (a call
        that keeps failing the same way -- malformed args, bad path). Fires
        inside the dispatch loop, only on a plain run-mode streak; metric runs
        defer to their own machinery, mirroring the verify no-progress guard."""
        non_metric_run = self.mode == "run" and metric_goal(self.config.workflow.metric) is None
        if not non_metric_run:
            return
        streak = state.spiral.error_streak
        if streak >= TOOL_ERROR_STOP_AFTER and state.spiral.error_nudges_used >= 2:
            turn.tool_error_stop = True
            return
        # A denial streak is a POLICY outcome: "your call is malformed" would
        # be false, and a refusal says nothing about jail reachability.
        denial = state.spiral.last_error_was_denial
        nudge = TOOL_DENIED_NUDGE if denial else TOOL_ERROR_NUDGE
        escalation = TOOL_DENIED_NUDGE if denial else TOOL_ERROR_ESCALATION
        if streak >= TOOL_ERROR_ESCALATE_AFTER and state.spiral.error_nudges_used == 1:
            state.spiral.error_nudges_used = 2
            turn.tool_results.append(Notice(escalation))
            self._emit("loop.tool_error.nudge", iteration=turn.iteration, streak=streak, level=2)
        elif streak >= TOOL_ERROR_NUDGE_AFTER and state.spiral.error_nudges_used == 0:
            state.spiral.error_nudges_used = 1
            turn.tool_results.append(Notice(nudge))
            self._emit("loop.tool_error.nudge", iteration=turn.iteration, streak=streak, level=1)

    def _note_jail_exec_failure(
        self,
        state: LoopState,
        turn: TurnState,
        name: str,
        tool_input: dict[str, Any],
        result: ToolResult,
    ) -> None:
        """Sandbox-reachability tracking. The one true "host-present but
        jail-broken" signal is a run_command the jail failed to EXEC
        (`exec_failed`; a nonzero exit is the command's own result) for a
        binary `shutil.which` finds on the host. The second consecutive
        exec failure of the same binary tells the model once and emits the
        event finalize's operator warning reads. Tool errors never feed this:
        a validation error or denial never entered the jail."""
        if name != "run_command" or not isinstance(result, ExecResult):
            return
        argv = tool_input.get("argv") or []
        binary = str(argv[0]) if isinstance(argv, list) and argv else ""
        if not result.exec_failed or not binary:
            state.jail_exec_failed_binary = ""
            state.jail_exec_failed_streak = 0
            return
        if binary == state.jail_exec_failed_binary:
            state.jail_exec_failed_streak += 1
        else:
            state.jail_exec_failed_binary = binary
            state.jail_exec_failed_streak = 1
        if (
            state.jail_exec_failed_streak < 2
            or state.sandbox_reachability_warned
            or shutil.which(binary) is None
        ):
            return
        state.sandbox_reachability_warned = True
        self._emit("loop.sandbox_tool_unreachable", binary=binary)
        self._log(f"LOOP: sandbox tool unreachable: {binary} exists on host, fails in jail")
        turn.tool_results.append(
            Notice(
                f"NOTE: `{binary}` is installed on this machine but the sandbox"
                " cannot execute it: a reachability problem (a per-user or"
                " version-manager install the jail does not mount), not a problem"
                " with your code. Tell the operator to install it into a standard"
                " bin dir (~/.local/bin, /usr/local/bin) or grant its real"
                " directory via sandbox.extra_read_paths; if the tool exists"
                " inside the workspace, call it by that path. Do not keep probing"
                " for it."
            )
        )

    def _turn_no_progress(self, state: LoopState, turn: TurnState) -> None:
        """Inject the spiral-guard nudges: fires only on a PLAIN run-mode
        streak of identical verify failures (see _nudges rationale). Metric
        runs are excluded: repeated verify failures while searching for an
        optimization are expected there, and the metric plateau / early-finish
        / ceiling machinery owns when such a run stops -- firing here would
        truncate the budgeted search and end the run completed=false."""
        non_metric_run = self.mode == "run" and metric_goal(self.config.workflow.metric) is None
        if not non_metric_run or not turn.verify_just_failed:
            return
        streak = state.verify.fail_streak
        if streak >= NO_PROGRESS_STOP_AFTER and state.no_progress_nudges_used >= 2:
            # Both nudges delivered and the identical failure persists: stop in
            # the stop checks rather than burn the rest of the budget.
            turn.no_progress_stop = True
            return
        if streak >= NO_PROGRESS_ESCALATE_AFTER and state.no_progress_nudges_used == 1:
            state.no_progress_nudges_used = 2
            turn.tool_results.append(Notice(NO_PROGRESS_ESCALATION))
            self._emit("loop.no_progress.nudge", iteration=turn.iteration, streak=streak, level=2)
        elif streak >= NO_PROGRESS_NUDGE_AFTER and state.no_progress_nudges_used == 0:
            state.no_progress_nudges_used = 1
            turn.tool_results.append(Notice(NO_PROGRESS_NUDGE))
            self._emit("loop.no_progress.nudge", iteration=turn.iteration, streak=streak, level=1)

    def _standing_task(self) -> tuple[str, str] | None:
        """The ready standing task's (id, title), if this run has one."""
        if self.curator is None:
            return None
        try:
            nodes = self.curator.nodes()
        except Exception:
            return None
        for nid, node in nodes.items():
            if node.standing and ready_subtask(nodes, node):
                return nid, node.title[:120]
        return None

    def _standing_absorb(self, state: LoopState, *, reason: str, iteration: int) -> str | None:
        """The standing-goal conversion for a soft end: the nudge text to
        inject when the run should re-enter the standing task instead of
        ending, else None. None when there is no ready standing task, when
        the budget is spent (the hard bounds always win), or once
        `[workflow].standing_patience` fruitless re-entries (no executed
        tool call since the last one) are used up. At the default (-1) a
        fruitless round never ends the run by itself: the nudge escalates
        to "dig deeper or try a different approach" instead, and the run
        ends on its budget, iteration cap, or an operator stop."""
        st = self._standing_task()
        if st is None:
            return None
        remaining = self._budget_fraction_remaining()
        if remaining is not None and remaining <= 0.0:
            return None
        nid, title = st
        if state.ok_tool_calls == state.standing_tools_mark:
            state.standing_fruitless += 1
            patience = self.config.workflow.standing_patience
            if 0 <= patience < state.standing_fruitless:
                self._log(
                    f"  standing: {state.standing_fruitless} fruitless re-entries >"
                    f" standing_patience {patience}; honouring {reason}"
                )
                return None
            nudge = standing_fruitless_nudge(reason, nid, title, state.standing_fruitless)
        else:
            state.standing_fruitless = 0
            nudge = standing_resume_nudge(reason, nid, title)
        state.standing_tools_mark = state.ok_tool_calls
        self._log(f"  standing re-entry ({reason}) -> {nid} at iter {iteration}")
        self._emit("loop.standing.resumed", reason=reason, task_id=nid, iteration=iteration)
        return nudge

    def _absorb_soft_stop(
        self, state: LoopState, turn: TurnState, conversation: Conversation
    ) -> None:
        """A standing task converts the soft out-of-work endings into
        re-entry: the pending stop flag is cleared and the standing nudge
        joins the conversation. Faults (tool_error), the loop guard, and
        every hard bound still end the run; the absorb itself refuses on
        spent budget or a spin."""
        soft = (
            "verify_settled"
            if turn.verify_settled_stop
            else "no_progress"
            if turn.no_progress_stop
            else "metric_plateau"
            if turn.plateau_should_stop
            else None
        )
        if soft is None or turn.tool_error_stop:
            return
        nudge = self._standing_absorb(state, reason=soft, iteration=turn.iteration)
        if nudge is None:
            return
        turn.verify_settled_stop = False
        turn.no_progress_stop = False
        turn.plateau_should_stop = False
        state.verify_settled_idle = 0
        state.verify.fail_streak = 0
        state.no_progress_nudges_used = 0
        state.plateau_nudges_used = 0
        conversation.notice(nudge)

    # ---- stop checks, silent finish, went-quiet --------------------------------

    def _turn_stop_checks(  # noqa: PLR0911 - a flat precedence ladder of terminal checks
        self, state: LoopState, turn: TurnState, conversation: Conversation
    ) -> SessionResult | None:
        """Terminal checks, run after the turn's tool_results are in
        `messages` and the post-tools snapshot is written, in precedence
        order: verify-settled stop, metric-plateau stop, loop-guard kill, then
        honouring a finish call that survived the gates."""
        self._absorb_soft_stop(state, turn, conversation)
        if turn.tool_error_stop:
            self._log(
                f"LOOP: tool_error stop at iter {turn.iteration}"
                f" (streak {state.spiral.error_streak})"
            )
            self._final_checkpoint(turn.iteration)
            self._emit(
                "session.end",
                reason="tool_error_stuck",
                iterations=turn.iteration,
                all_passed=False,
            )
            return SessionResult(
                completed=False,
                reason="tool_error_stuck",
                summary=(
                    "stopped: the same tool call failed"
                    f" {state.spiral.error_streak} times with the identical error"
                    " despite two harness interventions; resume with a different"
                    " approach"
                ),
                iterations=turn.iteration,
                tool_calls=state.tool_calls,
            )
        if turn.no_progress_stop:
            self._log(
                f"LOOP: no_progress stop at iter {turn.iteration}"
                f" (streak {state.verify.fail_streak})"
            )
            self._final_checkpoint(turn.iteration)
            self._emit(
                "session.end",
                reason="no_progress",
                iterations=turn.iteration,
                all_passed=False,
            )
            return SessionResult(
                completed=False,
                reason="no_progress",
                summary=(
                    "stopped: the same verify failure persisted through"
                    f" {state.verify.fail_streak} consecutive runs despite two"
                    " harness interventions; resume with a new approach or a"
                    " bigger budget"
                ),
                iterations=turn.iteration,
                tool_calls=state.tool_calls,
            )
        if turn.verify_settled_stop:
            self._log(
                f"LOOP: verify_settled at iter {turn.iteration} (idle {state.verify_settled_idle})"
            )
            self._final_checkpoint(turn.iteration)
            # Ground on the TREE, not on verify_ever_passed: a green verify
            # followed by un-reverified edits must not settle as "passed"
            # (finish_session grounds on the same probe, so the two clean ends
            # cannot disagree).
            if state.verify.ever_passed and self._tree_is_verify_green(state) is not False:
                self._emit_run_end_passed(reason="verify_settled", iterations=turn.iteration)
                return SessionResult(
                    completed=True,
                    verified=self._verification(state),
                    reason="verify_settled",
                    summary="verify passed and the worker stopped making changes",
                    iterations=turn.iteration,
                    tool_calls=state.tool_calls,
                )
            # The work is committed and the worker went quiet, but nothing
            # verified the FINAL tree, so this end never claims "passed".
            self._pass_pending_root_tasks()
            self._emit("session.end", reason="settled", iterations=turn.iteration, all_passed=False)
            if state.verify.ever_passed:
                summary = (
                    "the worker settled, but edits after the last green verify were"
                    " never re-verified"
                )
            elif self.config.workflow.verify_command:
                # A command can exist here only via mid-run adoption (an
                # operator-set one is never gateless).
                summary = (
                    "the worker settled after committing work; the adopted verify never passed"
                )
            else:
                summary = (
                    "the worker settled after committing work; no verify command existed to gate it"
                )
            return SessionResult(
                completed=True,
                verified=self._verification(state),
                reason="settled",
                summary=summary,
                iterations=turn.iteration,
                tool_calls=state.tool_calls,
            )
        if turn.plateau_should_stop:
            assert turn.metric_plateau_finish is not None
            self._log(f"LOOP: metric_plateau at iter {turn.iteration}")
            self._final_checkpoint(turn.iteration)
            # Ground on the tree like the sibling clean ends (finish_session,
            # verify_settled): an edit after the plateau's green verify means
            # nothing verified the FINAL tree, so this must not claim passed.
            self._emit_run_end_grounded(
                reason="metric_plateau", iteration=turn.iteration, state=state
            )
            return SessionResult(
                completed=True,
                verified=self._verification(state),
                reason="metric_plateau",
                summary=turn.metric_plateau_finish,
                iterations=turn.iteration,
                tool_calls=state.tool_calls,
            )
        # loop-guard escalation. The notice in _turn_notices is advisory; if
        # the worker keeps issuing the same call past loop_guard_kill_threshold,
        # terminate the run before it burns the rest of the budget circling.
        # Threshold of 0 disables (notice-only behaviour). The kill happens
        # AFTER the tool_results were appended so the transcript on disk
        # reflects exactly what the model produced up to the kill, which is
        # essential when triaging "why did my run die at iter N".
        if (
            self.loop_guard_kill_threshold > 0
            and state.spiral.call_streak >= self.loop_guard_kill_threshold
        ):
            latched_name = (state.spiral.last_call_sig or "").split(":", 1)[0] or "<unknown>"
            self._log(
                f"LOOP: loop_guard_killed at iter {turn.iteration} -"
                f" {latched_name} called {state.spiral.call_streak}x in a row"
                f" (threshold={self.loop_guard_kill_threshold})"
            )
            self._final_checkpoint(turn.iteration)
            self._emit(
                "session.end",
                reason="loop_guard_killed",
                iterations=turn.iteration,
                all_passed=False,
                tool=latched_name,
                streak=state.spiral.call_streak,
            )
            return SessionResult(
                completed=False,
                reason="loop_guard_killed",
                summary=(
                    f"loop-guard killed run: `{latched_name}`"
                    f" called {state.spiral.call_streak}x in a row with"
                    f" identical arguments (threshold"
                    f" {self.loop_guard_kill_threshold})"
                ),
                iterations=turn.iteration,
                tool_calls=state.tool_calls,
            )
        if turn.finish_signal is not None:
            self._log(f"LOOP: {turn.finish_kind} called at iter {turn.iteration}")
            self._final_checkpoint(turn.iteration)
            # Honest finish: finish_planning is always a clean finish, but a
            # finish_session over a red/stale verify is "finished", not "passed"
            # -- all_passed reflects the actual verify state, never just "the
            # model called finish_session".
            reason = self._finish_reason(turn, state)
            self._check_decisions_recorded(state)
            if turn.finish_kind == "finish_session":
                self._emit_run_end_grounded(reason=reason, iteration=turn.iteration, state=state)
            else:
                self._emit_run_end_passed(reason=reason, iterations=turn.iteration)
            return SessionResult(
                completed=True,
                verified=self._verification(state),
                reason=reason,
                summary=turn.finish_signal,
                iterations=turn.iteration,
                tool_calls=state.tool_calls,
                finish_payload=turn.finish_payload,
                stale_gate=turn.finish_stale_gate,
            )
        return None

    def _maybe_inject_plan(
        self, conversation: Conversation, state: LoopState, *, iteration: int
    ) -> SessionResult | None:
        """Put the CURRENT plan.md in front of the planner, every turn.

        plan.md on disk is the plan; the conversation only ever holds a copy, and
        `agent6 plan edit` writes the operator's answers to the file between legs.
        So the file is re-read here rather than resynced at one chosen moment, and
        injected only when it differs from what the planner was last shown -- an
        untouched plan costs nothing. finish_planning stays the only writer.

        An UNREADABLE plan parks the leg (the returned SessionResult): the file
        may carry operator answers the planner's own copy supersedes, and
        continuing without them spends budget on stale direction. A missing
        file is normal (the first finish_planning creates it).
        """
        if self.mode != "plan" or self.plan_output_path is None:
            return None
        try:
            text = self.plan_output_path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return None  # no plan yet; the first finish_planning creates it
        except (OSError, UnicodeDecodeError) as exc:
            session_id = self.events.path.parent.name if self.events is not None else "<session-id>"
            remedy = f"plan.md unreadable: {exc}; fix it and `agent6 resume {session_id}`"
            self._log(f"LOOP: {remedy}")
            self._emit("loop.plan_read.failed", path=str(self.plan_output_path), error=str(exc))
            self._emit(
                "session.end", reason="plan_unreadable", iterations=iteration, all_passed=False
            )
            return SessionResult(
                completed=False,
                reason="plan_unreadable",
                summary=remedy,
                iterations=iteration,
                tool_calls=state.tool_calls,
            )
        if text == state.plan_injected:
            return None
        state.plan_injected = text
        conversation.notice(f"{PLAN_ON_DISK_HEADER}\n\n{text}")
        self._log(f"  plan re-read from disk: {len(text)} chars")
        self._emit(
            "loop.plan_reread",
            path=str(self.plan_output_path),
            bytes=len(text.encode("utf-8")),
        )

    def _maybe_pre_call_nudges(
        self,
        conversation: Conversation,
        state: LoopState,
        *,
        iteration: int,
        start_iteration: int,
    ) -> None:
        """Before the LLM call, surface the current task for one-task focus, and
        inject a one-shot finish directive when a verbose planner or a non-metric
        run is reading forever without landing a plan / verify+finish before the
        budget dies."""
        # Surface-current-task first, so when a low budget ALSO fires a finish
        # directive below, that finish nudge is the most-recent (strongest)
        # message rather than the focus banner.
        self._maybe_surface_current_task(conversation, state)
        # Force a verbose planner to land a plan. Trigger on EITHER a low
        # token budget OR too many planning turns, with prompt caching a
        # planner can take many cheap turns, so an iteration cap is the
        # reliable lever for the "reads forever" failure mode. A rough
        # delivered plan beats an exhaustive one that never gets emitted.
        if self.mode == "plan" and not state.plan_finish_nudged:
            remaining = self._budget_fraction_remaining()
            low_budget = remaining is not None and remaining <= PLAN_BUDGET_NUDGE_BELOW
            too_many_turns = iteration - start_iteration + 1 >= PLAN_NUDGE_AFTER_ITERS
            if low_budget or too_many_turns:
                state.plan_finish_nudged = True
                conversation.notice(PLAN_BUDGET_NUDGE)
                self._log(
                    f"LOOP: plan finish-nudge at iter {iteration}"
                    f" (turns={too_many_turns}, low_budget={low_budget})"
                )
                self._emit(
                    "loop.plan_finish.nudge", iteration=iteration, budget_remaining=remaining
                )

        # Same lever for a non-metric coding run: force a verify + finish
        # before the budget dies (metric runs have their own end-game).
        if (
            self.mode == "run"
            and not state.run_budget_nudged
            and metric_goal(self.config.workflow.metric) is None
        ):
            remaining = self._budget_fraction_remaining()
            if remaining is not None and remaining <= RUN_BUDGET_NUDGE_BELOW:
                state.run_budget_nudged = True
                nudge = (
                    RUN_BUDGET_NUDGE
                    if self.config.workflow.verify_command
                    else RUN_BUDGET_NUDGE_GATELESS
                )
                conversation.notice(nudge)
                self._log(f"LOOP: run budget-nudge at iter {iteration}")
                self._emit("loop.run_budget.nudge", iteration=iteration, budget_remaining=remaining)

    def _maybe_surface_current_task(self, conversation: Conversation, state: LoopState) -> None:
        """Surface-current-task: keep the worker on ONE task at a time.

        Compute the current task (the cursor if it still points at an open
        subtask, else the first dependency-satisfied open subtask), advance the
        cursor to it, and inject a focus banner when the focus first appears,
        changes, or was wiped by a tier-2 restart (`surfaced_task_id` reset to
        None there). Advancing the cursor each turn means that once the worker
        marks the current task passed, the next turn's frontier recompute moves
        focus to the next ready task -- the cursor walks the frontier on its own.

        Also runs the anti-grind counter: when the focus task holds for
        `STUCK_ON_TASK_AFTER` turns with no forward motion, fire one nudge
        offering to split / pass / skip it.

        Run mode only; no curator or no open subtask is a no-op (the finish-gate
        covers the empty-frontier finish). Best-effort throughout: a curator
        hiccup logs and returns rather than breaking the loop.
        """
        if self.mode != "run" or self.curator is None:
            return
        try:
            cursor = self.curator.cursor()
        except Exception as exc:  # a curator read error must not break the loop
            self._log(f"LOOP: surface-current-task skipped: {exc}")
            return
        nodes = self.curator.nodes()
        current_id = current_task_id(nodes, cursor)
        if current_id is None:
            state.turns_on_task = 0  # frontier empty: nothing to grind on
            state.last_focus_id = None
            return  # nothing decomposed yet, or the frontier is empty
        if cursor != current_id:
            # Advance the cursor onto the frontier task (auto-advance: a passed
            # cursor task drops out of the frontier, so this moves forward).
            try:
                self.curator.set_cursor(SetCursorIntent(id=current_id))
            except Exception as exc:  # cursor advance is advisory; never fatal
                self._log(f"LOOP: cursor advance skipped: {exc}")
        # Anti-grind: count consecutive turns on the same focus task. Any forward
        # motion (cursor advance, a task marked done, or a decompose that moves the
        # cursor to a new subtask) changes current_id and resets the count; survives
        # compaction (last_focus_id is not reset there). Re-fire every
        # STUCK_ON_TASK_AFTER turns, capped at STUCK_NUDGE_MAX per task.
        if current_id != state.last_focus_id:
            state.turns_on_task = 0
            state.last_focus_id = current_id
            state.stuck_nudges_fired = 0
        else:
            state.turns_on_task += 1
            if (
                state.turns_on_task % STUCK_ON_TASK_AFTER == 0
                and state.stuck_nudges_fired < STUCK_NUDGE_MAX
            ):
                state.stuck_nudges_fired += 1
                conversation.notice(
                    stuck_on_task_nudge(current_id, nodes[current_id], state.turns_on_task)
                )
                self._log(
                    f"LOOP: stuck-on-task nudge #{state.stuck_nudges_fired} for"
                    f" {current_id} after {state.turns_on_task} turns"
                )
                self._emit(
                    "loop.task.stuck_nudge",
                    task_id=current_id,
                    turns=state.turns_on_task,
                    n=state.stuck_nudges_fired,
                )
        if current_id == state.surfaced_task_id:
            return  # already surfaced; the banner survives tier-1 elision
        node = nodes[current_id]
        if node.status == "pending":
            # Reflect that this task is now being worked, keeping the DAG honest
            # for the TUI and the check-off / finish-gate "open" set. Best-effort.
            try:
                self.curator.update_status(
                    UpdateStatusIntent(id=current_id, new_status="in_progress")
                )
            except Exception as exc:
                self._log(f"LOOP: mark in_progress skipped: {exc}")
        banner = current_task_banner(
            current_id, node, decompose=self.config.prompt.decompose == "on"
        )
        conversation.notice(banner)
        state.surfaced_task_id = current_id
        self._log(f"LOOP: surfaced current task {current_id}")
        self._emit("loop.task.surfaced", task_id=current_id)
        # The harness-driven cursor/status writes bypass the tool-dispatch path
        # that emits graph.update, so refresh the live view here.
        self._emit_graph_snapshot()

    def _handle_no_tool_use(
        self,
        resp: ProviderResponse,
        assistant: AssistantTurn,
        conversation: Conversation,
        state: LoopState,
        *,
        iteration: int,
    ) -> SessionResult | None:
        """Handle a turn with no tool_use. Either a silent finish (the agent
        emitted text; gated like an explicit finish_session) or went-quiet (an
        empty turn; nudged up to a cap). Returns a terminal SessionResult, or None
        to continue the loop after appending a nudge.

        Distinguishing the two matters: "agent talked then stopped" is likely
        an implicit finish (the user gets the text as summary), while "agent
        emitted nothing" is a went-quiet failure (an empty provider response,
        or a confused agent) that bench scoring must NOT treat as success."""
        text = resp.text.strip() if resp.text else ""
        if text:
            # A prose turn is NON-EMPTY: the went_quiet nudge budget refills
            # here exactly as on a tool_use turn (the documented per-streak
            # contract, "reset on any non-empty turn"). Without this, quiet
            # streaks interleaved with bounced prose turns (silent-finish
            # gates, question nudges) drained one shared budget and ended the
            # run as went_quiet although no streak reached the cap -- and the
            # starvation output-cap backoff stayed stuck reduced.
            state.went_quiet_nudges_used = 0
            turn = TurnState(iteration=iteration, resp=resp, assistant=assistant)
            return self._handle_silent_finish(text, conversation, state, turn)
        return self._handle_went_quiet(resp, conversation, state, iteration=iteration)

    def _silent_end_gates(
        self, state: LoopState, turn: TurnState, conversation: Conversation
    ) -> SessionResult | None:
        """The verify certification and the before_finish panel over a silent
        finish, as for an explicit finish_session; a prose turn has no tool
        results, so the gates' notices go to the conversation directly."""
        aborted = self._end_gates(state, turn, ending="silent_finish")
        for item in turn.tool_results:
            if isinstance(item, Notice):
                conversation.notice(item.text)
        if turn.review_text:
            conversation.notice(f"[review]\n{turn.review_text}")
        return aborted

    def _handle_silent_finish(  # noqa: PLR0911 - a gate chain of early bounces
        self, text: str, conversation: Conversation, state: LoopState, turn: TurnState
    ) -> SessionResult | None:
        """A no-tool_use turn WITH text: treat it as an implicit finish and run
        it through the same gates as an explicit finish_session. Returns None (with
        a nudge appended to the conversation) when a gate sends the worker back to
        work; the silent_finish SessionResult once every gate lets it through."""
        iteration = turn.iteration
        # An EARLY prose turn on an untouched tree is a stall, not an
        # implicit finish (observed: kimi answering a SWE-bench problem in
        # prose at iteration 2, ending the run patchless). Bounded to the
        # first iterations: an engaged run that read its fill and answers in
        # prose is a legitimate implicit finish and must not be taxed.
        if (
            self.mode == "run"
            and iteration <= 3
            and not state.ever_edited
            and not state.verify.ever_passed
            and state.silent_no_work_nudges_used < SILENT_NO_WORK_PATIENCE
        ):
            state.silent_no_work_nudges_used += 1
            conversation.notice(SILENT_NO_WORK_NUDGE)
            self._log(
                f"  silent finish rejected: no work yet (nudge"
                f" #{state.silent_no_work_nudges_used}) at iter {iteration}"
            )
            self._emit(
                "loop.silent_no_work.nudge",
                iteration=iteration,
                nudges_used=state.silent_no_work_nudges_used,
            )
            return None
        aborted = self._silent_end_gates(state, turn, conversation)
        if aborted is not None or turn.end_returned:
            return aborted
        # metric-run early-finish guard, mirroring the finish_session path: a
        # silent finish on an optimisation run with budget to spare should be
        # nudged to keep optimising rather than accepted. Without this,
        # dropping tool_use was a way to skip the plateau/early-finish policy
        # entirely.
        if (
            self.mode == "run"
            and metric_goal(self.config.workflow.metric) is not None
            and not self._metric_at_ceiling(state.metric_history)
        ):
            finish_budget_remaining = self._budget_fraction_remaining()
            has_runway = (
                finish_budget_remaining is not None
                and finish_budget_remaining > METRIC_PLATEAU_STOP_BELOW_BUDGET
            )
            if has_runway and state.metric_finish_nudges_used < METRIC_EARLY_FINISH_PATIENCE:
                assert finish_budget_remaining is not None
                state.metric_finish_nudges_used += 1
                conversation.notice(METRIC_FINISH_NUDGE)
                self._log(
                    f"  metric early-finish (silent) rejected"
                    f" #{state.metric_finish_nudges_used} at iter {iteration}"
                    f" (budget {finish_budget_remaining:.0%} left)"
                )
                self._emit(
                    "loop.metric_early_finish.rejected",
                    iteration=iteration,
                    nudges_used=state.metric_finish_nudges_used,
                    budget_remaining=finish_budget_remaining,
                    trigger="silent_finish",
                )
                return None
        # Task finish-gate (silent path): a worker that stops emitting tool
        # calls with its own subtasks still open is steered back to the list
        # rather than silently finished (shares the cap with the finish_session
        # path).
        task_nudge = self._task_finish_gate_nudge(state)
        if task_nudge is not None:
            self._log(
                f"  silent_finish gated: open subtasks remain (nudge"
                f" #{state.task_finish_nudges_used}) at iter {iteration}"
            )
            self._emit(
                "loop.task_finish.gated",
                iteration=iteration,
                nudges_used=state.task_finish_nudges_used,
                trigger="silent_finish",
            )
            conversation.notice(task_nudge)
            return None
        # Question-nudge (run mode, once): the model ended by asking the
        # operator something in prose without calling ask_user, so the run
        # would silently finish with an unanswered question. Nudge once to
        # call ask_user / finish_session; if it asks again, accept the finish
        # (bounded, so a stubborn model cannot loop the run).
        if self.mode == "run" and not state.question_nudged and ends_with_question(text):
            state.question_nudged = True
            self._log(f"  silent_finish nudged: ended on a question at iter {iteration}")
            self._emit("loop.question_nudge", iteration=iteration)
            conversation.notice(QUESTION_NUDGE)
            return None
        # A quiet run does not have to end: a standing goal re-enters, else an
        # interactive run parks for a steer (never in ask mode, where the
        # prose IS the answer).
        cont = self._quiet_continuation(
            conversation, state, iteration=iteration, reason="silent_finish"
        )
        if cont is not None:
            return None if isinstance(cont, NextTurn) else cont
        # In ask mode a prose answer with no tool call is the NORMAL success (the
        # answer IS the text), so end as "answered", not "silent_finish" -- the
        # latter read as a failure diagnostic on a perfectly good answer. run/plan
        # keep silent_finish: there, stopping without finish_session is mildly anomalous.
        reason: SessionEndReason = "answered" if self.mode == "ask" else "silent_finish"
        if self.mode == "ask":
            self._log(f"  ask answered at iter {iteration}")
        else:
            self._log(
                f"LOOP: silent_finish at iter {iteration} - agent emitted text but no tool_use"
            )
        self._final_checkpoint(iteration)
        # Honest finish: run/plan ground exactly like the explicit
        # finish_session path (observed green -> "passed", red or stale ->
        # "failed", ungated -> "finished"). Ask mode's prose answer is the
        # success (it never runs verify), so it always ends passed.
        if reason == "silent_finish":
            self._emit_run_end_grounded(reason=reason, iteration=iteration, state=state)
        else:
            self._emit_run_end_passed(reason=reason, iterations=iteration)
        return SessionResult(
            completed=True,
            verified=self._verification(state),
            reason=reason,
            # In ask mode the final prose IS the answer the caller
            # prints, so keep it whole; run/plan only need a short
            # summary line.
            summary=text if self.mode == "ask" else text[:1000],
            iterations=iteration,
            tool_calls=state.tool_calls,
        )

    def _handle_went_quiet(
        self,
        resp: ProviderResponse,
        conversation: Conversation,
        state: LoopState,
        *,
        iteration: int,
    ) -> SessionResult | None:
        """A fully-empty turn (no text, no tool_use): surface reasoning
        starvation explicitly, then nudge-and-retry up to the per-streak cap
        before ending the run as went_quiet.

        The nudge is cheap (~50 input tokens vs aborting the entire run) and
        almost always gets a weak open-weights model back on track. The empty
        assistant turn is dropped from the conversation first: Anthropic rejects an
        assistant message with empty content, a THINKING-ONLY turn (reasoning
        starvation: blocks but no text/tool_use) translates to one with no
        content and no tool_calls that strict OpenAI-compatible backends reject
        with a non-retryable 400, and either way it is dead context.
        AGENT6_WENT_QUIET_MAX_NUDGES overrides the cap."""
        # reasoning-starvation trip-wire. When a model spends its entire output
        # budget on reasoning_content and emits nothing user-visible, the
        # provider returns stop_reason="length" with empty text + no tool_uses.
        # Otherwise indistinguishable from a model that genuinely gave up, so
        # surface it explicitly rather than leaving raw transcripts as the only
        # diagnosis.
        reasoning_chars = 0
        raw_content = (resp.raw or {}).get("content") or []
        if isinstance(raw_content, list):
            for block in raw_content:
                if isinstance(block, dict) and block.get("type") == "thinking":
                    reasoning_chars += len(str(block.get("thinking") or ""))
        starved = output_cap_truncated(resp) and reasoning_chars > 0 and resp.output_tokens > 0
        if starved:
            self._log(
                f"LOOP: reasoning_starvation at iter {iteration}"
                f" - stop_reason=length, reasoning_chars={reasoning_chars},"
                f" output_tokens={resp.output_tokens}; the model spent"
                f" its entire output budget on reasoning_content."
                f" Add this model to _REASONING_MODEL_HINTS in"
                f" providers/openai.py if it isn't already."
            )
            self._emit(
                "loop.reasoning_starvation",
                iteration=iteration,
                reasoning_chars=reasoning_chars,
                output_tokens=resp.output_tokens,
                stop_reason=resp.stop_reason,
            )
        # An empty turn the provider still billed output tokens for is not a
        # model that chose silence: the tokens went to reasoning that never
        # surfaced, or to a tool call the upstream failed to parse and dropped
        # (seen on OpenRouter-routed qwen at temperature 0, deterministically
        # per prompt). Say so, in the log and on the event, so the transcript
        # file is not the only place the difference shows.
        # "billed" is a dollar word: on a subscription plan those tokens cost
        # $0, so the plan-metered run says "spent" instead of claiming a bill.
        plan_metered = self.budget is not None and self.budget.snapshot().plan_latest is not None
        spent_word = "spent" if plan_metered else "billed"
        billed = (
            f" ({resp.output_tokens} output tokens {spent_word} on it: reasoning that never"
            " surfaced, or a tool call the provider dropped)"
            if resp.output_tokens > 0 and not starved
            else ""
        )
        self._log(
            f"LOOP: went_quiet at iter {iteration} - agent emitted no text and no tool_use{billed}"
        )
        env_max = os.environ.get("AGENT6_WENT_QUIET_MAX_NUDGES", "").strip()
        effective_max_nudges = int(env_max) if env_max.isdigit() else self.went_quiet_max_nudges
        # Drop the dead turn before any exit: a provider rejects an assistant
        # message with empty content, and every path below either calls again
        # (nudge, standing goal, park) or snapshots the conversation for resume.
        conversation.pop_quiet_assistant()
        if state.went_quiet_nudges_used < effective_max_nudges:
            state.went_quiet_nudges_used += 1
            # A starved reasoner gets its own nudge: the generic empty-turn
            # message gives it nothing actionable, so it repeats the same
            # loop next turn.
            if starved:
                nudge_text = (
                    "[harness] Your previous turn spent its entire"
                    f" output budget ({resp.output_tokens} tokens) on"
                    " reasoning_content with no visible content and"
                    " no tool_use. STOP REASONING. On this next turn,"
                    " emit a tool_use IMMEDIATELY — do not think"
                    " further. If you genuinely don't know what to do"
                    " next, call `read_file` on the most relevant"
                    " source file to ground your next decision, or"
                    " call `finish_session` if the task is complete. Any"
                    " response that is not a tool_use will waste the"
                    " entire run."
                )
            else:
                nudge_text = (
                    "[harness] Your previous turn was empty: no text"
                    " content and no tool_use. This is a synthetic"
                    " prompt from the agent6 harness. Either call a"
                    " tool to make progress, or call `finish_session`"
                    " with a summary if the task is complete. Do"
                    " not reply with another empty turn."
                )
            conversation.notice(nudge_text)
            self._emit(
                "loop.went_quiet.nudge",
                iteration=iteration,
                nudges_used=state.went_quiet_nudges_used,
                nudges_max=effective_max_nudges,
                output_tokens=resp.output_tokens,
            )
            return None
        cont = self._quiet_continuation(
            conversation, state, iteration=iteration, reason="went_quiet"
        )
        if cont is not None:
            return None if isinstance(cont, NextTurn) else cont
        self._final_checkpoint(iteration)
        self._emit(
            "session.end",
            reason="went_quiet",
            iterations=iteration,
            all_passed=False,
        )
        return SessionResult(
            completed=False,
            reason="went_quiet",
            summary="(agent emitted no text and no tool_use)",
            iterations=iteration,
            tool_calls=state.tool_calls,
        )

    # ---- snapshots and carryover -----------------------------------------------

    def _seed_root_task(self, user_task: str) -> str | None:
        """Create the run's root task in the DAG when the curator
        is wired. Returns the new node id, or None if no curator.

        The root is the user's task itself. Subsequent agent `add_task`
        calls with `parent_id=None` attach under this root."""
        if self.curator is None:
            return None
        # TaskNodeDraft.title has min_length=1, so take the first NON-EMPTY
        # line ("(run)" when the task is blank).
        first_nonempty = next(
            (line.strip() for line in user_task.splitlines() if line.strip()),
            "",
        )
        title = first_nonempty[:200] if first_nonempty else "(run)"
        try:
            draft = TaskNodeDraft(
                title=title,
                rationale="single-loop run; root task seeded by Workflow",
                acceptance="",
                relevant_paths=(),
                created_by="user",
            )
            node = self.curator.add_subtask(AddSubtaskIntent(parent_id=None, draft=draft))
            return node.id
        except Exception as exc:
            self._log(f"LOOP: failed to seed root task: {exc}")
            return None

    def _worktree_dirty(self) -> bool:
        """True if the worktree holds content not yet recorded on the run's
        chain, e.g. an edit a worker made via run_command that the verify-pass
        auto-commit hasn't captured yet. The verify-settled detector treats
        that as in-progress work. (Plain `git status` would be wrong here: the
        operator's HEAD never moves during a run, so everything the agent has
        long since committed to the chain still reads as "dirty" against it.)
        Best-effort: any git error reports clean, so a hiccup can't wedge the
        detector; no chain (unit-test embedders) falls back to status."""
        try:
            if self.chain_ref is not None:
                return chain_dirty(
                    self.root,
                    self.chain_ref,
                    self.chain_fallback_parent,
                    exclude=self.untracked_at_start,
                )
            return not git_status(self.root, exclude=self.untracked_at_start).is_clean
        except (GitError, OSError):
            return False

    def _worktree_tree_sha(self) -> str:
        """Tree sha of the worktree's content (minus untracked_at_start),
        seeded on the chain tip like a chain commit; "" when git cannot say."""
        try:
            seed = self._chain_tip_sha() or self.chain_fallback_parent
            return worktree_tree(self.root, seed, self.untracked_at_start)
        except (GitError, OSError):
            return ""

    def _test_only_paths_since_red(self, red_tree: str) -> tuple[str, ...]:
        """Paths whose content differs between *red_tree* (the tree at the
        last red verify) and the current tree, when every one is a test file;
        () when either tree is unknown, nothing changed, or a non-test file did.
        Asked of git, so a run_command edit counts like an apply_edit."""
        if not red_tree:
            return ()
        tree = self._worktree_tree_sha()
        if not tree:
            return ()
        try:
            paths = tree_diff_paths(self.root, red_tree, tree)
        except (GitError, OSError):
            return ()
        if paths and all(is_test_path(p) for p in paths):
            return tuple(sorted(paths))
        return ()

    def _dirty_tree_note(self) -> str:
        """Summary suffix naming an uncommitted worktree, or "" when clean.

        An operator stop deliberately skips `_final_checkpoint`: committing
        over someone who is taking over would remove their choice to discard.
        The work is still in the checkout, but nothing reads it there --
        `sessions diff` and `sessions merge` both read git history -- so the state
        is stated rather than left silent."""
        if self.mode != "run" or self.chain_ref is None:
            return ""
        try:
            paths = chain_dirty_paths(
                self.root,
                self.chain_ref,
                self.chain_fallback_parent,
                _DIRTY_NOTE_CAP,
                exclude=self.untracked_at_start,
            )
        except (GitError, OSError):
            return ""
        if not paths:
            return ""
        more = "+" if len(paths) == _DIRTY_NOTE_CAP else ""
        noun = "file" if len(paths) == 1 and not more else "files"
        return f"; worktree left dirty ({len(paths)}{more} {noun} uncommitted, not checkpointed)"

    def _final_checkpoint(self, iteration: int) -> None:
        """Best-effort commit of any dirty worktree on a successful exit so
        run_command-authored edits on a gated run aren't lost from git history.

        On a gated run (verify_command set) the in-loop auto-commit only fires
        on a green verify; an edit made via run_command after a prior green
        verify, never re-verified, is left only in the working tree and is
        silently lost when the run ends (score.sh, resume, and the diff viewer
        all read git history). Capturing it here closes that gap."""
        if self.mode != "run" or not self.commit_per_step or not self._worktree_dirty():
            return
        try:
            subject = f"checkpoint (iter {iteration})"
            sha = self._chain_commit(subject)
            if sha:
                self._log(f"  final checkpoint: {sha[:12]}")
                self._emit("loop.auto_commit", iteration=iteration, sha=sha, subject=subject)
                # Also emit diff.updated so the commit is COUNTED: every fold
                # (web/TUI/CLI) tallies commits and the latest diff from
                # diff.updated alone, never from loop.auto_commit.
                self._emit(
                    "diff.updated",
                    sha=sha,
                    patch=commit_diff(self.root, sha, max_bytes=8000),
                )
        except (GitError, OSError) as exc:
            self._log(f"  final checkpoint commit failed: {exc}")

    def _pass_pending_root_tasks(self) -> None:
        """On successful completion, mark still-pending root task(s) as passed.

        The loop seeds one root task per `run()` (each ask REPL follow-up seeds
        another), but the worker finishes via `finish_session` without ever
        touching it -- so a completed ask/run otherwise reads `tasks 0/1`. Pass
        any root (`parent_id is None`) still pending/in-progress so the DAG --
        and every viewer + resume -- agrees the run completed. Subtasks the
        worker deliberately left unfinished are untouched (kept honest).
        Best-effort: a curator hiccup must never break completion."""
        if self.curator is None:
            return
        changed = False
        for nid, node in self.curator.nodes().items():
            if node.parent_id is None and node.status in ("pending", "in_progress"):
                try:
                    self.curator.update_status(UpdateStatusIntent(id=nid, new_status="passed"))
                    changed = True
                except Exception as exc:  # a curator write error must not break finish
                    self._log(f"LOOP: auto-pass root {nid} failed: {exc}")
                    break  # a curator write failure fails for every remaining node too
        if changed:
            self._emit_graph_snapshot()

    def _verification(self, state: LoopState) -> Verification:
        """The verify verdict for the SessionResult, from the same tri-state
        `session.end.all_passed` is grounded on, so the result and the event can
        never disagree. Not-green splits on the last observation: "failed"
        claims someone SAW a red gate, so a leg where no verify ran (or edits
        landed after the last green) is "unverified" instead -- both exit 4,
        but only one sends the operator chasing a red that never happened.

        Only a run is gated: plan and ask finish clean whatever the tree looks
        like (finish_planning and the ask answer both emit all_passed=True), and
        preflight still INFERS a verify command for a plan that never runs one,
        so grounding on the tree there reported failure against their own
        events."""
        if self.mode != "run":
            return "not_applicable"
        green = self._tree_is_verify_green(state)
        if green is None:
            return "not_applicable"
        if green:
            return "passed"
        return "failed" if state.verify.last_ok is False else "unverified"

    def _emit_run_end_passed(self, *, reason: str, iterations: int) -> None:
        """Emit a successful `session.end`, first auto-passing any still-pending
        root task so the DAG (and every viewer + resume) agrees the run
        completed -- otherwise a finish_session-only ask/run reads `tasks 0/1`."""
        self._pass_pending_root_tasks()
        self._emit("session.end", reason=reason, iterations=iterations, all_passed=True)

    def _tree_is_verify_green(self, state: LoopState) -> bool | None:
        """Is the current tree in a verified-green state? None when no verify
        command is configured (nothing to gate on); else True iff the last verify
        was green AND nothing has been edited since. Grounds both the honest
        finish signal and the opt-in hard finish gate, so 'passed' can never mean
        'finished over a red or stale verify'."""
        if not self.config.workflow.verify_command:
            return None
        return state.verify.green_and_untouched

    def _finish_reason(self, turn: TurnState, state: LoopState) -> SessionEndReason:
        """What this finish is called.

        `gate_stale` needs a gate that is actually RED. Green means it passed,
        truthfully. And `_tree_is_verify_green` returns None for a GATELESS run,
        where there is no gate to be stale -- reading that as "not green" made a
        gateless run declaring one end as passed, exit 0 and auto-merged, while
        printing a `config set` line for a command nothing had run. The proposal
        is recorded either way.

        `gate_red_at_base` outranks a plain finish over red: the gate was
        already failing before this run touched anything, so a red end is not
        this run's failure. Only ever from an observation, never a guess -- a
        run where nothing ever verified a clean tree says nothing.
        """
        if turn.finish_kind == "finish_session" and self._tree_is_verify_green(state) is False:
            if turn.finish_stale_gate:
                return "gate_stale"
            if state.verify.baseline_ok is False:
                return "gate_red_at_base"
        return turn.finish_kind

    def _emit_run_end_grounded(self, *, reason: str, iteration: int, state: LoopState) -> None:
        """Emit a clean end honestly: `all_passed` carries the verify
        tri-state. True only when the FINAL tree is OBSERVED verify-green,
        False when it is not (red or stale), None when nothing gated it (no
        verify command) -- so 'passed' can never mean 'ended over a red or
        stale verify' or 'nothing gated it', and an ungated end reads
        "finished", never "failed". finish_session, metric_plateau, and a
        run/plan silent finish ground the same way; `_verification` gives the
        same state its not_applicable verdict.

        The roots pass either way, like the settled path: the DAG tracks work
        items and the run-level word carries the verify truth, so grounding it
        there too left a red-verify finish reading `tasks 0/1` forever.

        `scoped` carries whether the gate ran scoped to the tests nearest the
        diff (the full command overran verify_timeout_s), so a scoped green
        reads "passed · scoped gate" on every surface, never a bare pass."""
        self._pass_pending_root_tasks()
        self._emit(
            "session.end",
            reason=reason,
            iterations=iteration,
            all_passed=self._tree_is_verify_green(state),
            scoped=state.verify.scoped,
        )

    def _emit_graph_snapshot(self) -> None:
        """Emit the current task DAG so a live viewer (the TUI) can render it.
        The worker's add_task/update_task tree lives in the curator, not the
        event log, so we snapshot it (once per turn, see the call site).

        Project to ONLY the fields the viewer renders, a full node dump carries
        unbounded model-authored text (rationale/acceptance/notes/paths) that
        bloats the fsync'd event log for no benefit. Best-effort: a curator
        hiccup must never break the run."""
        if self.curator is None:
            return
        try:
            cursor = self.curator.cursor()
        except Exception as exc:
            # cursor() reads cursor.json from disk; a hiccup (OSError, a
            # malformed cursor) must never break an otherwise-healthy run.
            self._log(f"LOOP: graph snapshot skipped: {exc}")
            return
        # FROZEN wire surface: project each node to exactly these four fields,
        # children as a JSON list -- the graph.update shape old run dirs, the
        # viewmodel fold, web and TUI all already hold. Pinned by
        # test_graph_update_snapshot_payload_is_wire_stable.
        nodes = {
            nid: {
                "title": n.title,
                "status": n.status,
                "parent_id": n.parent_id,
                "children": list(n.children),
            }
            for nid, n in self.curator.nodes().items()
        }
        self._emit("graph.update", nodes=nodes, cursor=cursor)

    def _load_repo_summary(self) -> RepoSummary:
        """Base summary plus structural priors when `prompt.structural_priors`
        is on; see `load_repo_summary`."""
        # prompt.structural_priors=false -> base summary only (no hot symbols /
        # co-change / symbol outline), a leaner prompt that leans on on-demand tools.
        disp = self.dispatcher if self.config.prompt.structural_priors else None
        return load_repo_summary(self.root, dispatcher=disp)

    def _load_memory_index(self) -> str:
        """The repo memory index for the system prompt.

        "" when no state_dir is wired, and for machine/agent modes (whose
        prompt assembly drops repo context). An unreadable index degrades to
        "" inside the store: memory is context, not correctness.
        """
        if self.state_dir is None or self.mode in ("machine", "agent"):
            return ""
        return memory_index_text(self.state_dir)

    def _load_decisions(self) -> str:
        """The operator's recorded rulings for the prompt ("" without a state
        dir); every mode sees them, a ruling binds a planner as much as a
        worker."""
        return decisions_text(self.state_dir) if self.state_dir is not None else ""

    def _record_decision(self, state: LoopState, question: str, answer: str) -> None:
        """An operator answer becomes a durable ruling the moment it arrives:
        appended to the repo's DECISIONS.md by the harness (never by the
        model), remembered for the finish-time check."""
        if self.state_dir is None or not answer.strip():
            return
        session = self.events.path.parent.name if self.events is not None else ""
        try:
            entry = record_decision(
                self.state_dir, question=question, answer=answer, session=session
            )
        except OSError as exc:
            self._log(f"LOOP: decision not recorded: {exc}")
            self._emit("loop.decision.unrecorded", error=str(exc))
            return
        state.decisions_recorded.append(entry)
        self._log(f"LOOP: decision recorded ({len(answer)} chars)")
        self._emit("loop.decision.recorded", question=question[:200], answer=answer[:200])

    def _check_decisions_recorded(self, state: LoopState) -> None:
        """The finish-time check: every ruling this leg recorded is in the
        file. A miss is reported (log + event), never a block."""
        if self.state_dir is None or not state.decisions_recorded:
            return
        try:
            text = decisions_path(self.state_dir).read_text(encoding="utf-8")
        except OSError:
            text = ""
        missing = [e for e in state.decisions_recorded if e.strip() not in text]
        if missing:
            self._log(f"LOOP: {len(missing)} recorded decision(s) missing from DECISIONS.md")
            self._emit("loop.decision.unrecorded", missing=len(missing))

    def _load_skills(self) -> ResolvedSkills | None:
        """Operator-installed skills for the system prompt, run mode only.

        Reuses the dispatcher's one-shot resolution so the <skills> index and
        what use_skill actually serves can never diverge. None (nothing
        installed, subsystem off, or non-run mode) renders no block.
        """
        if self.mode != "run":
            return None
        resolved = self.dispatcher.resolved_skills()
        for w in resolved.warnings:
            self._log(f"LOOP: skills: WARNING: {w}")
            self._emit("loop.skills.warning", warning=str(w))
        if resolved.enabled or resolved.always:
            self._log(
                f"LOOP: skills: {len(resolved.enabled)} indexed, {len(resolved.always)} always-on"
            )
            return resolved
        return None

    def _save_resume_snapshot(
        self,
        *,
        system: str,
        messages: list[dict[str, Any]],
        tool_calls: int,
        next_iteration: int,
        root_task_id: str | None,
        state: LoopState,
        write_checkpoint: bool = False,
    ) -> None:
        """Write loop state to disk for resume.

        Called before each LLM call and again at the end of each iteration
        (after the executed tool_results are appended) so a crash after a
        non-idempotent tool dispatch resumes from AFTER the executed tools
        rather than replaying them. Every call advances `loop_state.json`
        (the latest pointer resume follows); only the pre-call save passes
        `write_checkpoint` and owns `checkpoints/<next_iteration>.json` -- the
        state that turn's provider call consumes, written once, so
        `fork --at-turn N` has one meaning. Atomic via tmp-file + replace so a
        crash mid-write leaves the prior snapshot intact. No-op if
        `resume_state_path` is None (e.g. unit tests).
        """
        if self.resume_state_path is None:
            return
        goal = metric_goal(self.config.workflow.metric)
        best = best_metric_sample(state.metric_history, goal=goal) if goal is not None else None
        snapshot = SessionSnapshot(
            system=system,
            messages=messages,
            tool_calls=tool_calls,
            next_iteration=next_iteration,
            root_task_id=root_task_id,
            original_task=state.original_task,
            verify_command=self.config.workflow.verify_command,
            review_rejections_total=state.review_rejections_total,
            verify_ever_passed=state.verify.ever_passed,
            gateless_ever_committed=state.gateless_ever_committed,
            parallel_groups_dispatched=state.parallel_groups_dispatched,
            pins=tuple(state.pins),
            metric_best_score=best.score if best is not None else None,
            metric_at_ceiling=self._metric_at_ceiling(state.metric_history),
            last_verify_ok=state.verify.last_ok,
            edited_since_verify=state.verify.edited_since,
            baseline_ok=state.verify.baseline_ok,
            verify_scoped=state.verify.scoped,
            standing_tools_mark=state.standing_tools_mark,
            standing_fruitless=state.standing_fruitless,
            ok_tool_calls=state.ok_tool_calls,
            head_sha=self._checkpoint_head_sha(),
            graph_version=self._checkpoint_graph_version(),
        )
        blob = snapshot.model_dump_json()
        # The snapshot is recovery state, not run output: an unwritable state dir
        # (full disk, quota, read-only mount) disables resume/fork but must not
        # abort an otherwise-healthy run whose edits + commits are already on disk
        # independently. Warn once, then continue.
        try:
            # Write the append-only checkpoint first, then advance loop_state.json
            # as the latest pointer. If the second write fails, default fork still
            # follows loop_state.json, while explicit --at-turn can use the durable
            # checkpoint.
            if write_checkpoint:
                cp_dir = self.resume_state_path.parent / "checkpoints"
                atomic_write(cp_dir / f"{next_iteration:04d}.json", blob)
            atomic_write(self.resume_state_path, blob)
        except OSError as exc:
            if not self._snapshot_write_failed:
                self._snapshot_write_failed = True
                self._log(
                    f"LOOP: WARNING could not persist resume snapshot ({exc}); "
                    "resume/fork are unavailable for this run, continuing anyway"
                )

    def _checkpoint_head_sha(self) -> str:
        """Tip of the run's commit line for the per-turn checkpoint; "" if it
        can't be read. fork cuts its chain here; resume compares it to the
        live chain to warn about divergence. Without a chain (unit-test
        embedders) it is HEAD, and a checkpoint is best-effort recovery
        state -- a missing sha must not crash the snapshot."""
        if self.chain_ref is not None:
            return self._chain_tip_sha()
        try:
            return git_status(self.root).head_sha
        except (GitError, OSError):
            return ""

    def _checkpoint_graph_version(self) -> int:
        """Curator DAG version for the per-turn checkpoint; 0 if no curator."""
        if self.curator is None:
            return 0
        return self.curator.graph_version

    # ---- metric ----------------------------------------------------------------

    def _record_metric_result(
        self,
        history: list[MetricSample],
        result: MetricResult,
        *,
        iteration: int,
        label: str,
        sha: str,
    ) -> str | None:
        metric_cfg = self.config.workflow.metric
        goal = metric_goal(metric_cfg)
        if goal is None:
            return None
        assert metric_cfg is not None  # goal is None otherwise
        score = coerce_metric_score(result.score)
        returncode = result.returncode
        stdout = result.stdout
        stderr = result.stderr
        combined = f"{stdout}\n{stderr}"
        targets = extract_metric_targets(combined, goal=goal)
        at_ceiling = (
            goal == "maximize"
            and score is not None
            # Only count an X/Y ceiling reported on the score-match line, so an
            # incidental "100/100" progress bar elsewhere cannot latch it.
            and metric_at_fraction_ceiling(combined, score, pattern=metric_cfg.pattern)
        )
        sample = MetricSample(
            label=label,
            score=score,
            returncode=returncode,
            sha=sha,
            stdout_tail=stdout[-500:],
            stderr_tail=stderr[-500:],
            targets=targets,
            at_ceiling=at_ceiling,
        )
        history.append(sample)
        self._emit(
            "loop.metric.sample",
            iteration=iteration,
            label=label,
            score=score,
            returncode=returncode,
            sha=sha[:12],
        )
        return format_metric_feedback(history, goal=goal)

    def _auto_metric_feedback(
        self,
        history: list[MetricSample],
        *,
        iteration: int,
        sha: str,
    ) -> str | None:
        metric_cfg = self.config.workflow.metric
        goal = metric_goal(metric_cfg)
        if self.mode != "run" or goal is None:
            return None
        self._log(f"LOOP: auto metric after verify-pass at iter {iteration}")
        self._emit("loop.metric.auto_call", iteration=iteration, sha=sha[:12])
        try:
            result = self.dispatcher.dispatch("run_metric_command", {})
        except ToolError as exc:
            sample = MetricSample(
                label=f"auto iter {iteration}",
                score=None,
                returncode=None,
                sha=sha,
                error=str(exc),
            )
            history.append(sample)
            self._emit(
                "loop.metric.auto_failed",
                iteration=iteration,
                error=str(exc)[:200],
            )
            return format_metric_feedback(history, goal=goal)
        assert isinstance(result, MetricResult)  # run_metric_command's result type
        return self._record_metric_result(
            history,
            result,
            iteration=iteration,
            label=f"auto iter {iteration}",
            sha=sha,
        )

    def _plateau_finish(self, history: list[MetricSample]) -> str | None:
        """The plateau summary for THIS run: the shared rule, plus the two
        conditions only a run knows (run mode, a configured goal)."""
        goal = metric_goal(self.config.workflow.metric)
        if self.mode != "run" or goal is None:
            return None
        return metric_plateau_summary(history, goal=goal)

    def _metric_at_ceiling(self, history: list[MetricSample]) -> bool:
        """True once any verified sample reached the metric's provable
        ceiling (e.g. `SCORE: 27/27`). Such a metric cannot be improved, so
        the loop honours an early `finish_session` and stops nudging instead of
        spending the rest of the budget chasing an unbeatable number."""
        return any(sample.at_ceiling for sample in history)

    def _budget_fraction_remaining(self) -> float | None:
        """Fraction of the token budget still available, or None when no
        BudgetTracker is wired in (tests / MCP path)."""
        if self.budget is None:
            return None
        return self.budget.fraction_remaining()

    def _unexecutable_abort(
        self, exc: OperatorCommandUnexecutable, *, iteration: int, tool_calls: int
    ) -> SessionResult:
        """Graceful abort when an operator verify/metric command cannot run in
        the jail (e.g. its binary is not on the jail PATH). The model cannot fix
        operator config, so stop loudly rather than flail against a gate that
        never executes or silently report success. Shared by the manual per-tool
        path and the auto-metric-after-verify path so the same misconfiguration
        ends the same way regardless of who triggered the command."""
        self._log(f"LOOP: aborting -- {exc}")
        # The worst checkpoint case of all the harness ends: verify can never
        # go green here, so the per-turn auto-commit never fired and ALL of
        # the run's edits may exist only in the worktree.
        self._final_checkpoint(iteration)
        self._emit(
            "session.end",
            reason="verify_command_unexecutable",
            iterations=iteration,
            all_passed=False,
        )
        return SessionResult(
            completed=False,
            reason="verify_command_unexecutable",
            summary=str(exc),
            iterations=iteration,
            tool_calls=tool_calls,
        )

    def _worker_max_tokens(self, state: LoopState) -> int:
        """Per-call output cap for the worker turn.

        Metric-optimization runs (mode "run" with a configured continuous
        metric) lift the ceiling to `metric_task_max_tokens` so a single turn
        can rewrite a hot function wholesale without truncating mid-apply_patch.
        Every other run keeps `per_call_max_tokens`.

        Starvation backoff: once the worker has gone quiet (no text + no
        tool_use -- typically a reasoning model that spent its whole output
        budget on reasoning_content) on >= 2 CONSECUTIVE turns, drop back to
        `per_call_max_tokens` even on a metric run. A spiraling over-reasoner
        (observed: GLM 5.2) otherwise burns a fresh ~65k-token reasoning binge
        every nudged turn until it exhausts `went_quiet_max_nudges` and the run
        dies with zero progress. A tight cap plus the forceful "emit a tool_use
        now" nudge pressures it to ACT; `went_quiet_nudges_used` resets to 0 on
        the first productive turn, so the very next turn gets the full ceiling
        back for the real edit (the recovery edit itself is never truncated).
        The 2-quiet threshold spares the model the high ceiling was raised FOR
        (Kimi K2.x finishes its reasoning within 65k and rarely goes quiet, let
        alone twice in a row), so the backoff targets the spiral, not the model.
        """
        metric_run = self.mode == "run" and metric_goal(self.config.workflow.metric) is not None
        if metric_run and state.went_quiet_nudges_used < _STARVATION_BACKOFF_AFTER_QUIETS:
            return max(self.per_call_max_tokens, self.metric_task_max_tokens)
        return self.per_call_max_tokens

    # ---- context compaction drivers --------------------------------------------

    def _maybe_compact(
        self, conversation: Conversation, state: LoopState, *, prefix_chars: int = 0
    ) -> bool:
        """Tiered compaction. Returns True iff a tier-2 summarise-and-restart
        actually replaced the history (so the caller can re-surface the
        current-task banner the restart wiped); False otherwise.

        Tier 1 (cheap): drop old tool_result blocks once cumulative content
        exceeds `compact_drop_at_chars`.

        Tier 2 (expensive): once the WHOLE post-elision context (text +
        tool_use inputs + surviving tool_results, via `context_chars`)
        crosses `compact_summarise_at_chars`, summarise the elided history
        into a compact progress block and restart the conversation from
        (original task + summary). Fail-safe: if
        summarisation errors or returns nothing, the conversation is left
        untouched (tier-1 elision already ran) and the run continues.

        An operator compact request (`compact_requested`, the TUI's
        "Compact now") forces tier 2 regardless of the size thresholds; the
        marker is consumed here so one request means one compaction.
        """
        forced = self.compact_requested()
        if forced is not None:
            self.compact_clear()
            focus_note = f" (focus: {forced[:80]})" if forced else ""
            self._log(f"LOOP: operator requested a manual compaction{focus_note}")
            self._emit("loop.compact.requested", focus=forced)
        stats = compact_old_tool_results(
            conversation,
            max_total_bytes=self.compact_drop_at_chars,
            keep_recent=2,
            protect_paths=recently_edited_paths(conversation),
            gister=self._distill_gists if self.compact_elision_gists else None,
        )
        if self.keep_thinking_turns > 0 and (
            stats.deduped
            or stats.elided
            or context_chars(conversation) > self.compact_drop_at_chars
        ):
            # Same cache-bundling rule as dedup: only at tier-1 pressure
            # moments, never as a rolling per-iteration rewrite.
            n_turns, n_chars = strip_old_thinking(conversation, keep_turns=self.keep_thinking_turns)
            if n_turns:
                self._log(
                    f"LOOP: compaction dropped thinking from {n_turns} old turns ({n_chars} chars)"
                )
                self._emit("loop.compact.thinking_dropped", turns=n_turns, chars=n_chars)
        if stats.deduped:
            self._log(f"LOOP: compaction deduplicated {stats.deduped} identical tool results")
            self._emit("loop.compact.deduped", n=stats.deduped, calls=list(stats.deduped_calls))
        if stats.elided:
            detail = f", {stats.gisted} kept as distilled gists" if stats.gisted else ""
            self._log(f"LOOP: compaction elided {stats.elided} old tool_result blocks{detail}")
            self._emit("loop.compact.dropped", n=stats.elided, calls=list(stats.elided_calls))
        if stats.demoted:
            self._log(f"LOOP: compaction demoted {stats.demoted} gists to bare placeholders")
        if stats.gisted or stats.demoted:
            self._emit(
                "loop.compact.gists",
                gisted=stats.gisted,
                demoted=stats.demoted,
                paths=list(stats.gist_paths),
                demoted_paths=list(stats.demoted_paths),
            )
        # Measure the WHOLE post-elision request, not just tool_results: tier 1
        # already bounded those, so re-measuring them could never cross the
        # larger tier-2 threshold -- and the window bounds the request, so the
        # system prompt and the tool definitions count too
        # (`request_prefix_chars`).
        total = context_chars(conversation) + prefix_chars
        # Tier 2 needs at least an original-task turn plus enough history
        # to be worth summarising; below that a restart would lose more than
        # it saves. The growth floor (see LoopState.tier2_floor_chars) keeps
        # a restart that lands near the threshold from summarising every
        # other iteration; a forced (operator) compaction bypasses it.
        over = total > self.compact_summarise_at_chars and total >= state.tier2_floor_chars
        if (forced is not None or over) and len(conversation) > 3:
            return self._summarise_and_restart(conversation, state, focus=forced or "")
        if forced is not None:
            # The request was consumed above (one request, one compaction), so a
            # silent return would drop it: the front-end has already told the
            # operator it "applies before the next model call", and the focus
            # text is gone. Say the floor refused it.
            self._log("LOOP: manual compaction skipped: too little history to summarise")
            self._emit("loop.compact.refused", reason="too little history to summarise")
        return False

    def _distill_gists(self, requests: tuple[GistRequest, ...]) -> dict[str, str]:
        """Distill about-to-be-elided file reads into one-line gists with the
        summariser model (same seat as tier-2). Fail-safe: any provider error
        returns {} and every victim gets the bare placeholder, so gisting can
        slow a drop event but never break one."""
        provider = self.summariser_provider or self.provider
        files = "\n\n".join(f"=== FILE {r.path} ===\n{r.content}" for r in requests)
        self._emit("loop.compact.gist.call", files=len(requests))
        try:
            resp = provider.call(
                system=GIST_DISTILL_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": files}],
                tools=[],
                max_tokens=self.context_summary_max_tokens,
                temperature=0.0,
            )
        except (ProviderError, BudgetExceeded) as exc:
            self._log(f"  gist distillation failed: {exc}; eliding without gists")
            self._emit("loop.compact.gist.failed", error=str(exc)[:200])
            return {}
        return parse_gist_lines(resp.text or "", paths=[r.path for r in requests])

    def _summarise_and_restart(
        self, conversation: Conversation, state: LoopState, *, focus: str = ""
    ) -> bool:
        """Replace the history with (original task + a model-written progress
        summary), in place. The loop only calls this at the top of an
        iteration, where the history is balanced (every `tool_use` already
        has its `tool_result`), so the restart can drop the middle without
        orphaning a tool-call pairing. Returns True iff the history was
        actually replaced; False on every fail-safe path (the tier-1-elided
        context is kept and the run continues).
        """
        provider = self.summariser_provider or self.provider
        turns = conversation.turns
        # The verbatim tail survives the restart, so the summary covers only
        # what is actually dropped (pi's keepRecentTokens shape).
        tail_start = recent_tail_start(turns, self.keep_recent_chars)
        if tail_start <= 1:
            # A cap that swallows the whole history would make the restart
            # grow the context instead of shrinking it; keep nothing.
            tail_start = len(turns)
        transcript = format_transcript_tail(
            turns[1:tail_start], max_messages=len(conversation), max_chars=60_000
        )
        # The DAG is agent6's compaction memory: at each restart we ask the
        # summariser to check off finished tasks and surface newly-found ones, so
        # task state stays accurate across compaction without depending on the
        # worker calling update_task (which weak models rarely do).
        open_tasks = self._open_tasks_for_checkoff()
        if open_tasks:
            task_lines = "\n".join(f"- {tid}: {title}" for tid, title in open_tasks)
            checkoff_req = (
                "\n\nThe worker is tracking these OPEN tasks:\n"
                f"{task_lines}\n\n"
                "After the summary, append a fenced block exactly like:\n"
                "```checkoff\n"
                '{"completed_ids": ["<ids the transcript clearly shows finished>"], '
                '"new_tasks": ["<short title of work discovered but not yet tracked>"]}\n'
                "```\n"
                "Mark a task completed ONLY if the transcript clearly shows it done;"
                " leave the rest open. Use [] when none apply."
            )
        else:
            checkoff_req = ""
        focus_req = (
            f"\n\nOperator focus for this summary — weigh these aspects heavily:\n{focus}"
            if focus
            else ""
        )
        pins_req = ""
        if state.pins:
            pin_lines = "\n".join(f"{i}. {p}" for i, p in enumerate(state.pins, start=1))
            pins_req = PINS_NO_RESTATE_CLAUSE + pin_lines
        # The previous restart's summary rides at the HEAD of the post-restart
        # history and the transcript above is tail-clipped, so it was dropped
        # first: the new summary then started at the last restart while the
        # preamble promised the worker everything was captured. Carry it
        # out-of-band, like pins.
        prior_req = ""
        for turn in conversation.turns:
            for item in getattr(turn, "items", ()):
                if isinstance(item, Notice) and (prior := progress_summary_from_notice(item.text)):
                    prior_req = (
                        "\n\nThis conversation was ALREADY compacted; the summary from"
                        " that restart follows, and the transcript below covers only what"
                        " happened SINCE. Carry anything still relevant into the new"
                        f" summary:\n{prior}"
                    )
        user_msg = (
            "Summarise the following agent transcript for a context restart."
            f"\n\nTASK (the goal, verbatim):\n{state.original_task}"
            f"{checkoff_req}{focus_req}{pins_req}{prior_req}"
            f"\n\nTRANSCRIPT (oldest first):\n{transcript}"
        )
        self._log(f"LOOP: tier-2 compaction summarise-and-restart ({len(conversation)} msgs)")
        self._emit("loop.compact.summarise.call", messages=len(conversation))
        try:
            resp = provider.call(
                system=CONTEXT_SUMMARY_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_msg}],
                tools=[],
                max_tokens=self.context_summary_max_tokens,
                temperature=0.0,
            )
        except (ProviderError, BudgetExceeded) as exc:
            # Fail-safe: keep the current (tier-1-elided) context. A real
            # budget exhaustion is re-detected by the next provider call.
            self._log(f"  tier-2 summarise failed: {exc}; keeping current context")
            self._emit("loop.compact.summarise.failed", error=str(exc)[:200])
            return False
        raw = (resp.text or "").strip()
        if not raw:
            self._emit("loop.compact.summarise.failed", error="empty summary")
            return False
        # Apply the check-off to the curator (best-effort) and strip the block
        # from the summary so the restarted worker sees narrative, not bookkeeping.
        if open_tasks:
            self._apply_compaction_checkoff(raw, valid_ids={tid for tid, _ in open_tasks})
        summary = strip_checkoff(raw) if open_tasks else raw
        conversation.restart(
            context_restart_notice(self.mode, pins=state.pins, decisions=self._load_decisions())
            + summary,
            keep=turns[tail_start:],
        )
        state.tier2_floor_chars = int(context_chars(conversation) * 1.25)
        self._emit(
            "loop.compact.summarise.done",
            summary_chars=len(summary),
            summary=summary,
            kept_turns=len(turns) - tail_start,
        )
        return True

    def _open_tasks_for_checkoff(self) -> list[tuple[str, str]]:
        """(id, title) of every pending/in_progress task in the DAG, for the
        tier-2 compaction check-off. Best-effort: no curator or a curator error
        yields an empty list, so compaction degrades to the plain summary."""
        if self.curator is None:
            return []
        out: list[tuple[str, str]] = []
        for nid, node in self.curator.nodes().items():
            # Subtasks only: never offer the auto-root (parent_id is None) for
            # check-off, mirroring the finish-gate and surface rules. The root is
            # the whole-run container, so a mid-run summary must not mark it
            # passed and end the run early.
            if node.parent_id is None or node.standing:
                continue
            if node.status in ("pending", "in_progress"):
                out.append((nid, node.title[:120]))
        return out

    def _apply_compaction_checkoff(self, summary_text: str, *, valid_ids: set[str]) -> None:
        """Parse the summariser's ```checkoff block and apply it to the curator:
        mark completed tasks passed, queue newly-discovered ones as children of
        the first root. Best-effort: a curator hiccup must never break the run."""
        if self.curator is None:
            return
        completed, new_tasks = parse_checkoff(summary_text)
        completed = [cid for cid in completed if cid in valid_ids]  # ignore hallucinated ids
        if not completed and not new_tasks:
            return
        changed = False
        try:
            for cid in completed:
                self.curator.update_status(
                    UpdateStatusIntent(id=cid, new_status="passed", note="compaction check-off")
                )
                changed = True
            if new_tasks:
                root_id = self._first_root_id()
                for title in new_tasks[:8]:  # cap: a runaway summary can't flood the DAG
                    self.curator.add_subtask(
                        AddSubtaskIntent(
                            parent_id=root_id,
                            draft=TaskNodeDraft(title=title, created_by="planner"),
                        )
                    )
                    changed = True
        except Exception as exc:  # a curator write error must not break the run
            self._log(f"LOOP: compaction check-off partial ({exc})")
        if changed:
            self._log(
                f"LOOP: compaction check-off -- passed {len(completed)}, queued {len(new_tasks)}"
            )
            self._emit_graph_snapshot()

    def _first_root_id(self) -> str | None:
        """The first root task id (parent_id is None), or None. Best-effort."""
        if self.curator is None:
            return None
        for nid, node in self.curator.nodes().items():
            if node.parent_id is None:
                return nid
        return None

    def _task_finish_gate_nudge(self, state: LoopState) -> str | None:
        """If the worker created subtasks and any are still open, return a nudge
        message to re-prompt with instead of finishing; else None (finish OK).

        Only SUBTASKS (parent_id is not None) gate -- the auto-root is pending
        until the run ends, so gating on it would deadlock. Capped by
        `TASK_FINISH_PATIENCE`: after that many blocked finishes the finish is
        honoured (a task the worker can't close, and won't mark obsolete/skipped,
        must not bounce the loop forever). Best-effort: no curator -> no gate."""
        if self.curator is None:
            return None
        open_subtasks = [
            (nid, node.title[:120])
            for nid, node in self.curator.nodes().items()
            if node.parent_id is not None
            and node.status in ("pending", "in_progress")
            # A standing task is not unfinished work: it gates the finish via
            # its own re-entry, never via this capped nudge.
            and not node.standing
        ]
        if not open_subtasks:
            return None
        if state.task_finish_nudges_used >= TASK_FINISH_PATIENCE:
            return None  # cap reached: stop bouncing, honour the finish
        state.task_finish_nudges_used += 1
        listing = "\n".join(f"- {tid}: {title}" for tid, title in open_subtasks)
        return (
            "[harness] You still have open tasks; finish the work before stopping. "
            f"{len(open_subtasks)} task(s) are pending/in_progress:\n{listing}\n"
            "Continue with the next one. If a task is genuinely not needed or you"
            " cannot do it, call update_task to mark it skipped or obsolete -- do"
            " not just abandon it. Then finish_session once the list is clear."
        )

    # ---- prompt revision and provider retry ------------------------------------

    def _maybe_revise_prompt(self, user_task: str, repo: RepoSummary) -> str:
        if self.revise_prompt == "off":
            return user_task
        if self.prompt_reviser_provider is None:
            raise PromptRevisionError(
                "prompt.revise_prompt is enabled but no reviser provider is wired"
            )

        context = format_prompt_revision_context(repo)
        user_msg = (
            f"RAW_TASK:\n{user_task}\n\nREPO_CONTEXT:\n{context}\n\nRewrite the raw task now."
        )
        self._log(f"LOOP: prompt revision ({self.revise_prompt})")
        self._emit("loop.prompt_revision.call", mode=self.revise_prompt)
        try:
            resp = self.prompt_reviser_provider.call(
                system=PROMPT_REVISION_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_msg}],
                tools=[],
                max_tokens=self.prompt_revision_max_tokens,
                temperature=self.prompt_reviser_temperature,
            )
        except (ProviderError, BudgetExceeded) as exc:
            self._emit("loop.prompt_revision.failed", error=str(exc)[:200])
            raise PromptRevisionError(str(exc)) from exc

        revision = parse_prompt_revision(resp.text or "")
        if not revision.revised_task:
            self._emit("loop.prompt_revision.failed", error="empty revised task")
            raise PromptRevisionError("reviser returned an empty task")

        self._emit(
            "loop.prompt_revision.result",
            raw_chars=len(user_task),
            revised_chars=len(revision.revised_task),
            questions=len(revision.clarifying_questions),
        )
        self._log(
            "PROMPT REVISION\n"
            "--- original ---\n"
            f"{clip_text(user_task, 4000)}\n"
            "--- revised ---\n"
            f"{clip_text(revision.revised_task, 6000)}"
        )
        if revision.clarifying_questions:
            self._log(
                "PROMPT REVISION QUESTIONS\n"
                + "\n".join(f"- {q}" for q in revision.clarifying_questions)
            )

        if self.revise_prompt == "interactive":
            if self.prompt_revision_selector is None:
                raise PromptRevisionError(
                    "prompt.revise_prompt='interactive' needs an interactive selector"
                )
            selected = self.prompt_revision_selector(
                user_task,
                revision.revised_task,
                revision.clarifying_questions,
            )
            if selected is None or not selected.strip():
                raise PromptRevisionError("operator aborted prompt revision")
            selected_task = selected.strip()
            if selected_task == user_task.strip():
                return user_task
            return format_effective_task(
                user_task,
                PromptRevision(
                    revised_task=selected_task,
                    clarifying_questions=revision.clarifying_questions,
                ),
            )

        return format_effective_task(user_task, revision)

    def _call_with_retry(
        self,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[ToolDefinition],
        max_tokens: int,
    ) -> ProviderResponse:
        """Bounded-retry wrapper around `provider.call`: up to
        `provider_retry_count + 1` attempts. Two retry paths share that budget:

        - Transient `ProviderError` (Anthropic 529, OpenRouter 502, brief socket
          timeout): retried with exponential backoff + full jitter so one flap
          doesn't abort the run. `BudgetExceeded` is never retried (hard stop).
          Permanent client errors (`ProviderError.status_code` in
          `NON_RETRYABLE_HTTP_STATUSES`: 400/401/402/403/404/422, or
          `ProviderError.fatal`) re-raise immediately without consuming a
          retry: a second identical request cannot succeed.
        - A self-contradictory empty tool-call response
          (`is_empty_tool_call_response`: stop_reason promises a tool call but
          none and no text came back -- GLM via OpenRouter, ~50% post-restart):
          retried with a short fixed delay (model flakiness, not rate-limiting),
          excluding `stop_reason=length` starvation. If every attempt is empty
          the last is returned and the loop's went_quiet handler takes over.
        """
        attempts = max(1, self.provider_retry_count + 1)
        last_exc: ProviderError | None = None
        for attempt in range(1, attempts + 1):
            try:
                resp = self.provider.call(
                    system=system,
                    messages=messages,
                    tools=tools,
                    max_tokens=max_tokens,
                    temperature=self.temperature,
                    should_abort=self.should_abort,
                    should_interrupt=self.should_interrupt,
                )
            except (ProviderAborted, ProviderInterrupted):
                raise  # operator stop/steer: handle it, never retry as a fault
            except ProviderError as exc:
                last_exc = exc
                non_retryable = exc.fatal or exc.status_code in NON_RETRYABLE_HTTP_STATUSES
                if attempt < attempts and not non_retryable:
                    base_delay = self.provider_retry_delay_s * (2 ** (attempt - 1))
                    capped_delay = min(base_delay, self.provider_retry_max_delay_s)
                    # jitter (full jitter, lower-bounded at 0.5) decorrelates
                    # concurrent retriers; non-crypto randomness is fine here.
                    delay = capped_delay * random.uniform(0.5, 1.0)  # noqa: S311
                    # Honor an upstream Retry-After (429/503): wait at least the
                    # advertised window (bounded), since our own backoff is
                    # usually shorter and would just burn the retries before the
                    # rate-limit clears.
                    if exc.retry_after_s is not None:
                        delay = max(delay, min(exc.retry_after_s, RETRY_AFTER_CEILING_S))
                    self._log(
                        f"LOOP: provider error attempt {attempt}/{attempts}: "
                        f"{exc} - retrying in {delay:.2f}s"
                    )
                    self._emit(
                        "loop.provider.retry",
                        attempt=attempt,
                        error=str(exc)[:200],
                    )
                    time.sleep(delay)
                    continue
                if non_retryable:
                    self._log(
                        f"LOOP: provider error {exc.status_code or 'fatal'} is permanent;"
                        " not retrying"
                    )
                    self._emit(
                        "loop.provider.fatal",
                        status_code=exc.status_code,
                        error=str(exc)[:200],
                    )
                raise
            # A self-contradictory empty tool-call response (GLM via OpenRouter,
            # ~50% after a context restart): retry the identical request, which
            # recovers it about half the time. Bounded by the same attempt budget;
            # if every attempt comes back empty the loop's went_quiet handler takes
            # over. A short delay (no exponential growth) -- this is model
            # flakiness, not rate-limiting.
            if is_empty_tool_call_response(resp) and attempt < attempts:
                delay = min(self.provider_retry_delay_s, 1.0) * random.uniform(0.5, 1.0)  # noqa: S311
                self._log(
                    f"LOOP: empty tool-call response attempt {attempt}/{attempts}"
                    f" (stop_reason={resp.stop_reason!r}, no tool_use/text);"
                    f" retrying in {delay:.2f}s"
                )
                self._emit(
                    "loop.provider.empty_tool_call_retry",
                    attempt=attempt,
                    stop_reason=str(resp.stop_reason),
                )
                time.sleep(delay)
                continue
            return resp
        # Defensive: loop above either returns or raises; this is unreachable.
        # Kept for type-checker exhaustiveness in case the loop body changes.
        assert last_exc is not None
        raise last_exc

    # ---- review panel ----------------------------------------------------------

    def _has_reviewer(self) -> bool:
        """A second opinion is available: the review panel has seats. Gates
        every in-loop review trigger."""
        return bool(self.review_seats)

    def _run_diff(self) -> str:
        """The run's cumulative change: base commit vs the working tree, so it
        includes committed AND uncommitted edits, with the RUN'S new
        untracked files as additions (untracked-at-start files excluded,
        matching the chain). Empty if no base is known or git fails. Routed through
        git_ops so the repo-controlled fsmonitor/diff.external/hooks keys stay
        neutralized (a raw `git diff` here would run a poisoned `.git/config`
        payload on the host)."""
        if not self.base_sha:
            return ""
        return diff_since(self.root, self.base_sha, exclude=self.untracked_at_start)

    def _read_agents_md(self) -> str:
        # The same text the run prompt injects (repo root's file included on a
        # subdirectory start), so review and worker see one set of conventions.
        return agents_md_text(self.root)

    def _readonly_review_tools(self) -> tuple[list[ToolDefinition], ReviewDispatch]:
        return build_readonly_review_tools(self.dispatcher)

    def _run_review_panel(
        self, state: LoopState, *, trigger: str, iteration: int
    ) -> CritiqueResult | None:
        """Run the grounded review panel over the run diff. Returns a
        `CritiqueResult` (`satisfied=False` only when the panel BLOCKS and
        the gate is still armed). Per-seat + panel events are emitted in seat
        order; the per-run rejection counter decays on a pass and disarms the gate
        once it hits the cap so a gating panel can never stall the run."""
        diff = self._run_diff()
        if not diff.strip():
            # No diff to ground against (nothing changed, or base_sha missing on a
            # pre-field resume). Can't review -> approve, but make the skip visible
            # so a "gate didn't run" is never silent.
            self._emit(
                "loop.review.skipped", iteration=iteration, trigger=trigger, reason="no_diff"
            )
            return None
        # Skip the panel once the run's remaining token budget falls below
        # review_budget_fraction: reviewing is most expensive (esp. explore-tier
        # seats) exactly when budget is scarcest, and a skipped panel is
        # approve-and-proceed (the before_finish gate only blocks on an explicit
        # unsatisfied critique, so returning None here lets finish through). This
        # is the sole read site for review_budget_fraction.
        remaining = self._budget_fraction_remaining()
        if remaining is not None and remaining < self.review_budget_fraction:
            self._emit(
                "loop.review.skipped",
                iteration=iteration,
                trigger=trigger,
                reason="budget_fraction",
                remaining=round(remaining, 3),
            )
            return None
        # on_verify_fail/periodic never gate (advisory text only); only
        # before_finish consumes .satisfied + the rejection counter.
        decision: ReviewDecision = (
            self.review_decision if trigger == "before_finish" else "advisory"
        )
        ctx = ReviewContext(
            task=state.original_task,
            agents_md=self._read_agents_md(),
            diff=diff,
            verify_ok=state.verify.last_ok,
            verify_output=state.verify.last_tail,
        )
        self._emit(
            "loop.review.start", iteration=iteration, trigger=trigger, seats=len(self.review_seats)
        )
        tools: list[ToolDefinition] | None = None
        dispatch: ReviewDispatch | None = None
        if any(s.tier == "explore" for s in self.review_seats):
            tools, dispatch = self._readonly_review_tools()
        try:
            result = run_panel(
                self.review_seats,
                ctx,
                decision=decision,
                quorum=self.review_quorum,
                panel_id=f"{trigger}-{iteration}",
                concurrency=self.review_concurrency,
                tools=tools,
                dispatch=dispatch,
            )
        except BudgetExceeded:
            self._emit("loop.review.skipped", iteration=iteration, reason="budget")
            return None
        for v in result.per_seat:
            self._emit(
                "loop.review.seat",
                iteration=iteration,
                seat=v.seat,
                model=v.model,
                verdict="abstain" if v.error else v.verdict,
                findings=len(v.findings),
            )
        disarmed = state.review_rejections_total >= self.review_max_total_rejections
        effective_blocked = result.blocked and not disarmed
        self._emit(
            "loop.review.panel",
            iteration=iteration,
            trigger=trigger,
            decision=decision,
            blocked=effective_blocked,
            raw_blocked=result.blocked,
            disarmed=disarmed,
            n_block=result.n_block,
            n_abstain=result.n_abstain,
        )
        if trigger == "before_finish":
            if effective_blocked:
                state.review_rejections_total += 1
            else:
                state.review_rejections_total = max(0, state.review_rejections_total - 1)
        # An all-abstain panel reviewed nothing: name that in the critique text
        # (the model reads it) instead of "No blocking findings.", which the CLI
        # verdict was fixed for too. The gate still lets the finish through -- a
        # panel must never deadlock a run -- so `satisfied` is unchanged.
        if panel_is_inconclusive(result):
            text = inconclusive_note(result)
        else:
            text = render_findings(result.merged_findings) or "No blocking findings."
        return CritiqueResult(text=text, satisfied=not effective_blocked)

    # ---- steering and operator boundaries --------------------------------------

    def _operator_boundary(
        self, conversation: Conversation, iteration: int, state: LoopState
    ) -> SessionResult | None:
        """The end-of-iteration operator-control boundary, run after EVERY
        completed iteration (tool turns and prose turns alike): honor a
        pending "stop after this step" marker, then poll the steering flag.
        The safe point is AFTER a complete iteration, so a stop or an
        injected instruction never splits a tool_use / tool_result pair; the
        per-iteration snapshot is the resume point."""
        # Before the menu below can print them: a background command's ending
        # only reaches disk when someone observes it, and `/shells` reads from
        # there.
        self.dispatcher.settle_background()
        if self.stop_requested():
            self.stop_clear()
            self._log(f"LOOP: operator stop at the step boundary (iter {iteration})")
            self._emit("session.end", reason="steer_abort", iterations=iteration, all_passed=False)
            return SessionResult(
                completed=False,
                reason="steer_abort",
                summary=(
                    f"operator stopped the run after step {iteration}{self._dirty_tree_note()}"
                ),
                iterations=iteration,
                tool_calls=state.tool_calls,
            )
        # The operator can press Ctrl-C once to drop a steering instruction
        # into the conversation; a second Ctrl-C within 2s raises
        # KeyboardInterrupt and aborts.
        return self._steer_outcome(
            self._maybe_handle_steer(conversation, iteration, state), iteration, state
        )

    def _quiet_continuation(
        self, conversation: Conversation, state: LoopState, *, iteration: int, reason: str
    ) -> SessionResult | NextTurn | None:
        """The run-mode continuations for a quiet turn, in priority order: a
        standing goal re-enters (autonomy first), else an interactive run
        parks for a steer. Returns NEXT_TURN to continue the loop, a park's
        terminal steer verb, or None when neither applies (the caller ends
        the run as before)."""
        if self.mode != "run":
            return None
        nudge = self._standing_absorb(state, reason=reason, iteration=iteration)
        if nudge is not None:
            conversation.notice(nudge)
            return NEXT_TURN
        if self.interactive:
            parked = self._park_for_steer(conversation, state, iteration=iteration, reason=reason)
            return NEXT_TURN if parked is None else parked
        return None

    def _park_for_steer(
        self, conversation: Conversation, state: LoopState, *, iteration: int, reason: str
    ) -> SessionResult | None:
        """The interactive turn boundary: the model went quiet, so the run
        parks -- the SAME in-memory conversation, its snapshot already on
        disk -- until the operator steers it from any composer or the pause
        menu. Returns None when a steer (or a bare poke) continued the run,
        or the steer verb's terminal result. No timeout: parked-until-steered
        is the point; stop/abort are the exits."""
        self._log(
            f"LOOP: parked at iter {iteration} ({reason}) - waiting for your steer"
            " (any composer or the pause menu; abort ends the run)"
        )
        self._emit("loop.parked", iteration=iteration, reason=reason)
        while True:
            if self.stop_requested():
                self.stop_clear()
                self._emit(
                    "session.end", reason="steer_abort", iterations=iteration, all_passed=False
                )
                return SessionResult(
                    completed=False,
                    reason="steer_abort",
                    summary=f"operator stopped the parked run{self._dirty_tree_note()}",
                    iterations=iteration,
                    tool_calls=state.tool_calls,
                )
            if self.should_abort():
                return self._steer_outcome("abort", iteration, state)
            if self.steer_requested():
                verb = self._maybe_handle_steer(conversation, iteration, state)
                if verb is not None:
                    return self._steer_outcome(verb, iteration, state)
                # Injected (or a bare poke): the run continues where it parked.
                self._emit("loop.parked.resumed", iteration=iteration)
                return None
            time.sleep(0.5)

    def _steer_outcome(
        self, steer_result: str | None, iteration: int, state: LoopState
    ) -> SessionResult | None:
        """Map a _maybe_handle_steer result to a terminal SessionResult, or None to keep
        going (empty steer, or an instruction injected into messages)."""
        if steer_result in ("abort", "exit"):
            # "exit" is /exit at the pause menu: the same stop, but the end
            # reason tells the CLI to skip the follow-up prompt and leave.
            reason: SessionEndReason = "steer_exit" if steer_result == "exit" else "steer_abort"
            self._emit("session.end", reason=reason, iterations=iteration, all_passed=False)
            return SessionResult(
                completed=False,
                reason=reason,
                summary=(
                    f"operator {'exited' if steer_result == 'exit' else 'aborted'}"
                    f" at iter {iteration} via steering prompt{self._dirty_tree_note()}"
                ),
                iterations=iteration,
                tool_calls=state.tool_calls,
            )
        if steer_result == "undo":
            forked = self.undo_forker() if self.undo_forker is not None else None
            if forked is None:
                # The forker printed why (or no forker is wired); keep running.
                self._log("  /undo: nothing to undo; continuing")
                return None
            new_id, undone_text = forked
            self._emit("session.undone", new_session_id=new_id, undone_text=undone_text)
            # An undo is the operator's own end, like an abort: without a
            # session.end the run read "stale" (a dead worker and no end).
            self._emit("session.end", reason="undone", iterations=iteration, all_passed=False)
            return SessionResult(
                completed=False,
                reason="undone",
                summary=f"operator undid the last message at iter {iteration}; forked to {new_id}",
                iterations=iteration,
                tool_calls=state.tool_calls,
            )
        if steer_result == "detach":
            # Not an end: the caller respawns a detached `resume` that appends to this
            # same log, so a persistent viewer follows straight through (no session.end).
            # The per-iteration snapshot is the resume point.
            return SessionResult(
                completed=False,
                reason="detached",
                summary=f"operator detached at iter {iteration}; resuming in the background",
                iterations=iteration,
                tool_calls=state.tool_calls,
            )
        return None

    def _maybe_handle_steer(  # noqa: PLR0911 - one return per steer verb
        self,
        conversation: Conversation,
        iteration: int,
        state: LoopState,
    ) -> str | None:
        """Operator steering between iterations.

        Returns `"abort"` if the operator typed "abort" at the prompt;
        the loop should then return a steer_abort result. Returns `None`
        in all other cases (no request, empty steer, `/parallel` dispatch,
        or instruction injected into the conversation).

        Polls steer_requested() and, on a positive, calls steer_prompt()
        to capture operator text. Empty / None / KeyboardInterrupt aborts;
        boundary is between completed iters so a tool_use / tool_result pair
        is never split. A message starting with the exact `/parallel` token
        is a dispatch directive (see `_dispatch_parallel`), not an injected
        instruction.
        """
        if not self.steer_requested():
            return None
        self._emit("loop.steer.requested", iteration=iteration)
        self._log(f"STEER: operator steering at iter {iteration}")
        try:
            text = self.steer_prompt()
        finally:
            self.steer_clear()
        if text is None or not text.strip():
            self._log("  (empty - continuing)")
            return None
        steer_text = text.strip()
        if steer_text.lower() == "abort":
            self._emit("loop.steer.aborted")
            self._log("  abort - halting the run")
            return "abort"
        if steer_text.lower() == "exit":
            self._emit("loop.steer.exited")
            self._log("  exit - halting the run and leaving the terminal")
            return "exit"
        if steer_text.lower() == "/undo":
            self._emit("loop.steer.undo")
            self._log("  /undo - forking back before the last message")
            return "undo"
        if steer_text.lower() == "detach":
            self._emit("loop.steer.detached")
            self._log("  detach - stopping to resume in the background")
            return "detach"
        if (
            self._steer_directive(conversation, iteration, state, steer_text)
            or self._steer_pin(conversation, state, steer_text)
            or self._steer_skill(conversation, steer_text)
        ):
            return None
        self._log(f"  injecting steering instruction ({len(steer_text)} chars)")
        self._emit("loop.steer.injected", chars=len(steer_text), text=steer_text)
        asked = _last_assistant_prose(conversation)
        if ends_with_question(asked):
            self._record_decision(state, asked.strip().splitlines()[-1], steer_text)
        conversation.notice(
            "OPERATOR STEERING (mid-run instruction; "
            "incorporate this into your next step):\n"
            f"{steer_text}"
        )
        return None

    def _steer_skill(self, conversation: Conversation, steer_text: str) -> bool:
        """Handle a `/<skill> [args]` steer from any composer: the skill's
        full text is injected as the instruction (the same payload on every
        surface). Returns True when handled; False when *steer_text* names no
        enabled skill."""
        if not steer_text.startswith("/"):
            return False
        found = skill_command(steer_text, self.dispatcher.resolved_skills())
        if found is None:
            return False
        skill, args = found
        self._log(f"  skill steer: {skill.name}")
        self._emit("loop.steer.skill", name=skill.name, args=args)
        conversation.notice(
            "OPERATOR STEERING (mid-run instruction; incorporate this into your next step):\n"
            + skill_steer_payload(skill.name, skill.text, args)
        )
        return True

    def _steer_pin(self, conversation: Conversation, state: LoopState, steer_text: str) -> bool:
        """Handle a steer that is a `/pin` directive. A recorded pin is injected
        as a marked instruction AND re-injected verbatim after every tier-2
        restart. Over the total cap, the instruction is still delivered as an
        ordinary steer -- only the durability is refused, loudly. Returns True
        when handled; False when *steer_text* is not a pin directive."""
        try:
            instruction = parse_pin(steer_text)
        except DirectiveError as exc:
            conversation.notice(f"OPERATOR STEERING: nothing pinned: {exc}")
            self._log(f"  /pin refused: {exc}")
            return True
        if instruction is None:
            return False
        if not self._try_pin(state, instruction):
            # parse_pin already rejects an empty directive, so a refusal here is
            # always the cap: deliver the instruction as an ordinary steer.
            self._log(f"  /pin over cap (> {PINS_MAX_CHARS}); delivered as an ordinary steer")
            self._emit("loop.pin.refused", chars=len(instruction), limit=PINS_MAX_CHARS)
            conversation.notice(
                f"OPERATOR STEERING (pin refused: the {PINS_MAX_CHARS}-char pin cap "
                "is full, so this is an ordinary instruction that will NOT survive "
                "context compaction; incorporate it into your next step):\n"
                f"{instruction}"
            )
            return True
        self._log(f"  pinned instruction ({len(instruction)} chars, {len(state.pins)} pins)")
        self._emit(
            "loop.pin.added", text=instruction, chars=len(instruction), count=len(state.pins)
        )
        conversation.notice(
            "OPERATOR STEERING (PINNED — this instruction survives context "
            "compaction; it stays binding for the rest of the run):\n"
            f"{instruction}"
        )
        return True

    def _try_pin(self, state: LoopState, instruction: str) -> bool:
        """Append *instruction* to the run's pins IF it is non-empty and fits
        the PINS_MAX_CHARS cap; return whether it was pinned. THE single owner
        of the pin invariants -- both `/pin` and the pre-run --pin seeding go
        through it, so seeding cannot skip the cap (a huge --pin otherwise rode
        every restart and permanently wedged /pin) or the empty check
        (`--pin ""` seeded a blank pin)."""
        instruction = instruction.strip()
        if not instruction:
            return False
        held = sum(len(p) for p in state.pins)
        if held + len(instruction) > PINS_MAX_CHARS:
            return False
        state.pins.append(instruction)
        return True

    # ---- /parallel steer dispatch (coordinator) --------------------------

    def _steer_directive(
        self,
        conversation: Conversation,
        iteration: int,
        state: LoopState,
        steer_text: str,
    ) -> bool:
        """Handle a steer that is a `/parallel` directive: dispatch a valid one,
        or answer a malformed one (a bare `/parallel`, a spec with no task) and
        continue. Returns True when handled; False when *steer_text* is ordinary
        steering to inject as an instruction."""
        try:
            segments = parse_directive(steer_text)
        except DirectiveError as exc:
            self._inject_parallel_feedback(conversation, f"nothing dispatched: {exc}")
            return True
        if segments is None:
            return False
        self._dispatch_parallel(conversation, iteration, state, segments)
        return True

    # ---- parallel lane dispatch ------------------------------------------------

    def _dispatch_parallel(
        self,
        conversation: Conversation,
        iteration: int,
        state: LoopState,
        segments: list[Segment],
    ) -> None:
        """Dispatch a `/parallel` sibling group at the steer boundary: clone the
        coordinator's committed HEAD into one isolated lane per expanded lane
        (a segment with spec=3 -> three lanes of that task; spec=m1,m2 -> one lane
        per model), run them via the injected group spawner, join each branch back
        in dispatch order, and inject ONE summary so the model continues informed.
        Runs synchronously -- no provider calls happen while the group is in
        flight, so the run's budget is untouched by the wait.

        Never ends the run: an unavailable spawner, a bad spec, a dirty tree it
        cannot auto-commit, a spawner fault, a failed lane, or a join conflict
        each answer the steer with a message and continue."""
        if self.lane_spawner is None:
            self._inject_parallel_feedback(
                conversation,
                "parallel dispatch is not available in this front-end; continuing normally.",
            )
            return
        try:
            # One DAG node per SEGMENT (task); its lanes join under it.
            lanes_cap = self.config.parallel.max_lanes
            per_segment = [segment_lanes(seg, state.pins, limit=lanes_cap) for seg in segments]
        except DirectiveError as exc:
            self._inject_parallel_feedback(
                conversation, f"bad /parallel spec: {exc}; nothing dispatched."
            )
            return
        lanes = [lane for seg_lanes in per_segment for lane in seg_lanes]
        # Lanes cut from the chain tip only: chain-commit a changed tree first,
        # and refuse (rather than dispatch stale work) if it will not come clean.
        if not self._ensure_clean_for_dispatch(iteration):
            self._inject_parallel_feedback(
                conversation,
                "refusing to dispatch: the working tree is not clean and could not be"
                " auto-committed. Commit or discard your changes, then retry /parallel.",
            )
            return

        state.parallel_groups_dispatched += 1
        group = f"p{state.parallel_groups_dispatched}"
        # Persist the bump BEFORE the group blocks. This runs inside the
        # operator boundary, which is after the iteration's snapshot and before
        # the next one, so the counter would otherwise live only in memory for
        # the entire group: a crash there would resume with the stale count
        # and the next /parallel would re-use this group's id, colliding with
        # its lane clones and branches.
        self._save_resume_snapshot(
            system=state.system,
            messages=conversation.to_wire(),
            tool_calls=state.tool_calls,
            next_iteration=iteration + 1,
            root_task_id=state.root_task_id,
            state=state,
        )
        self._log(
            f"PARALLEL: dispatching group {group} "
            f"({len(lanes)} lane(s) across {len(segments)} task(s))"
        )
        # Lane ids do not exist until the spawner names them; the dispatched
        # event carries the truth it has (per-segment tasks + group), and
        # joined/failed name the real per-lane ids from each LaneResult.
        self._emit(
            "loop.parallel.dispatched", group=group, tasks=[seg.task[:200] for seg in segments]
        )
        parent_id = self._parallel_parent_id(state.root_task_id)
        node_ids = [self._add_parallel_node(seg.task, parent_id) for seg in segments]
        if any(n is not None for n in node_ids):
            self._emit_graph_snapshot()

        try:
            # Lanes cut from the run's chain tip, which _ensure_clean_for_dispatch
            # just made current; blocks, no provider calls meanwhile.
            results = self.lane_spawner(lanes, group, at=self._chain_tip_sha() or None)
            if len(results) != len(lanes):
                raise SubrunError(
                    f"group spawner returned {len(results)} result(s) for {len(lanes)} lane(s)"
                )
        except Exception as exc:
            # The spawner is an injected ui-side callback (clones, thread pool,
            # detached spawns); any fault it leaks -- OSError, SubrunError, a
            # result-count mismatch -- must answer the steer, never abort the
            # run. Everything after this point is never-raising by construction
            # (_join_lane_result and _stamp_parallel_node catch their own faults).
            self._log(f"PARALLEL: group {group} dispatch failed: {exc}")
            for nid in node_ids:
                self._stamp_parallel_node(nid, status="failed", note=f"dispatch failed: {exc}")
            self._emit_graph_snapshot()
            self._emit("loop.parallel.failed", group=group, error=str(exc))
            self._inject_parallel_feedback(
                conversation,
                f"group {group} dispatch failed: {exc}. Nothing was joined; continuing normally.",
            )
            return

        # The spawner names the group `<coordinator>-<group>` and stamps that on
        # every lane's manifest, so it is the id `sessions compare` takes. Read
        # it back off a lane (each is `<group id>-l<n>`, the derivation
        # `run_parallel` uses too) rather than printing the local counter, which
        # names no group on disk.
        group = results[0].spec.session_id.rsplit("-l", 1)[0] if results else group

        # Join every lane sequentially in dispatch order (a merge mutates the one
        # workspace, so joins can never run concurrently), then stamp one DAG node
        # per segment from its lanes' joins.
        joined = [
            join_lane_result(
                self.root,
                res,
                ref=self.chain_ref or "",
                fallback_parent=self.chain_fallback_parent,
                identity=self._commit_identity(),
                also_branch=self.chain_branch,
            )
            for res in results
        ]
        cursor = 0
        for nid, seg_lanes in zip(node_ids, per_segment, strict=True):
            width = len(seg_lanes)
            self._stamp_segment_node(nid, joined[cursor : cursor + width])
            cursor += width
        self._emit_graph_snapshot()

        payload = [
            {"session_id": j.session_id, "branch": j.branch, "status": j.status, "sha": j.sha}
            for j in joined
        ]
        self._emit("loop.parallel.joined", group=group, lanes=payload)
        failures = [p for p, j in zip(payload, joined, strict=True) if j.status != "joined"]
        if failures:
            self._emit("loop.parallel.failed", group=group, lanes=failures)
        self._inject_parallel_summary(conversation, group, joined)

    def _ensure_clean_for_dispatch(self, iteration: int) -> bool:
        """True when the chain tip carries the worktree's content, so lanes cut
        from it see current work. Changed content is chain-committed first;
        returns whether it came clean (with commit_per_step off, a changed
        tree cannot be captured and dispatch is refused)."""
        if not self._worktree_dirty():
            return True
        if not self.commit_per_step:
            return False
        try:
            sha = self._chain_commit(f"checkpoint before /parallel dispatch (iter {iteration})")
            if sha:
                self._log(f"  pre-dispatch checkpoint: {sha[:12]}")
                self._emit("loop.auto_commit", iteration=iteration, sha=sha)
        except (GitError, OSError) as exc:
            self._log(f"PARALLEL: pre-dispatch checkpoint failed: {exc}")
        return not self._worktree_dirty()

    def _parallel_parent_id(self, root_task_id: str | None) -> str | None:
        """Parent for a dispatched subtask: the curator cursor when it points at
        an open node, else the run root. Best-effort -- a curator hiccup falls
        back to the root."""
        if self.curator is None:
            return root_task_id
        try:
            cursor = self.curator.cursor()
        except Exception:
            return root_task_id
        return current_task_id(self.curator.nodes(), cursor) or root_task_id

    def _add_parallel_node(self, task: str, parent_id: str | None) -> str | None:
        """Add a steering-created DAG node for one dispatched task; None when no
        curator is wired or the add fails (the dispatch still proceeds)."""
        if self.curator is None:
            return None
        title = next((ln.strip() for ln in task.splitlines() if ln.strip()), "")[:200]
        try:
            node = self.curator.add_subtask(
                AddSubtaskIntent(
                    parent_id=parent_id,
                    draft=TaskNodeDraft(
                        title=title or "(parallel task)",
                        rationale="dispatched via /parallel steering",
                        created_by="steering",
                    ),
                )
            )
            return node.id
        except Exception as exc:
            self._log(f"PARALLEL: DAG node add failed: {exc}")
            return None

    def _stamp_parallel_node(
        self, node_id: str | None, *, status: NodeStatus, note: str, sha: str = ""
    ) -> None:
        """Record a dispatched node's outcome: its join sha (when given) then its
        final status. Best-effort -- a curator hiccup must not break the run."""
        if self.curator is None or node_id is None:
            return
        try:
            if sha:
                self.curator.record_commit(RecordCommitIntent(id=node_id, sha=sha))
            self.curator.update_status(UpdateStatusIntent(id=node_id, new_status=status, note=note))
        except Exception as exc:
            self._log(f"PARALLEL: DAG node stamp failed for {node_id}: {exc}")

    def _stamp_segment_node(self, node_id: str | None, lanes: list[LaneJoin]) -> None:
        """Stamp one segment's DAG node from its lanes' joins (the reduction
        lives in `segment_stamp`)."""
        status, note, sha = segment_stamp(lanes)
        self._stamp_parallel_node(node_id, status=status, note=note, sha=sha)

    def _inject_parallel_feedback(self, conversation: Conversation, msg: str) -> None:
        """Answer a `/parallel` steer with a one-line notice and continue."""
        self._log(f"PARALLEL: {msg}")
        conversation.notice(f"[parallel] {msg}")

    def _inject_parallel_summary(
        self, conversation: Conversation, group: str, joined: list[LaneJoin]
    ) -> None:
        """Inject the lane-outcome summary so the model continues informed."""
        conversation.notice(summary_text(group, joined))

    # ---- chain commits, events, infra ------------------------------------------

    def _log(self, msg: str) -> None:
        self.logger(f"[agent6] {msg}")

    def _emit(self, event_type: str, **fields: Any) -> None:
        if self.events is not None:
            self.events.emit(event_type, **fields)

    def _emit_start(self, event_type: str, **fields: Any) -> None:
        """A start-family event goes through the one emitter that stamps the
        worker pid first (see :func:`agent6.sessions.ipc.emit_session_start`)."""
        if self.events is not None:
            emit_session_start(self.events, self.events.path.parent, event_type, **fields)

    def _commit_identity(self) -> CommitIdentity | None:
        """Author identity plus the provenance trailer for this loop's commits.

        `[git.commit].name`/`.email` are the only identity on a machine whose
        git has none: preflight accepts them, so dropping them here made every
        chain commit fail with "Author identity unknown".
        """
        commit = self.config.git.commit
        if not (commit.name or commit.email or self.commit_trailer):
            return None
        return CommitIdentity(
            name=commit.name or None, email=commit.email or None, trailer=self.commit_trailer
        )

    def _chain_commit(self, subject: str) -> str:
        """One commit of the worktree onto the run's detached chain; "" when
        nothing changed since the tip or no chain is configured."""
        if self.chain_ref is None:
            return ""
        return (
            chain_commit(
                self.root,
                subject,
                ref=self.chain_ref,
                fallback_parent=self.chain_fallback_parent,
                identity=self._commit_identity(),
                also_branch=self.chain_branch,
                exclude=self.untracked_at_start,
            )
            or ""
        )

    def _chain_tip_sha(self) -> str:
        """Tip of the run's commit line; "" when the chain has no commits and
        no fallback (unborn repo) or no chain is configured."""
        if self.chain_ref is None:
            return ""
        try:
            return chain_tip(self.root, self.chain_ref) or self.chain_fallback_parent or ""
        except (GitError, OSError):
            return ""

    def _checkpoint_subject(self, turn: TurnState, *, fallback: str) -> str:
        """The per-step commit message, per `[git.commit.checkpoint].message`."""
        agent6_subject = _summarise_assistant_text_for_commit(
            turn.resp.text or "", turn.iteration, fallback=fallback
        )
        style = self.config.git.commit.checkpoint.message
        if style == "agent6":
            return agent6_subject
        summary = _first_prose_line(turn.resp.text or "", fallback=fallback)
        changes = worktree_name_status(self.root)
        if style == "conventional":
            return conventional_commit_subject(changes, summary=summary)
        msg = self._model_commit_message(changes, hint=summary)
        if msg:
            return msg
        self._log("WARNING: model commit message failed; using the agent6 style")
        return agent6_subject

    def _model_commit_message(self, changes: Sequence[tuple[str, str]], *, hint: str) -> str | None:
        """Model-drafted checkpoint message from git facts only; None on any
        failure (the caller degrades to the agent6 style)."""
        listing = "\n".join(f"{s}\t{p}" for s, p in changes[:200])
        return call_for_text(
            self.provider,
            system=(
                "Write a git commit message for the change set: one"
                " imperative subject line under 72 characters, optionally a"
                " blank line and a short body. Use only the facts given."
                " Output the message text only."
            ),
            user=f"Summary hint: {hint}\nChanged files (status\tpath):\n{listing}",
            max_tokens=400,
        )

    def _emit_budget(self, iteration: int) -> None:
        """Per-iteration usage heartbeat: running token + cost totals. Lets
        `agent6 sessions show` / the TUI show live spend, and leaves a recent event at
        the start of each iteration so a long provider call is still
        distinguishable from a stall."""
        if self.budget is None:
            return
        snap = self.budget.snapshot()
        cost, _ = self.budget.estimate_usd()
        self._emit(
            "loop.budget",
            iteration=iteration,
            input_tokens=snap.input_total,
            output_tokens=snap.output_total,
            cache_read_tokens=snap.cache_read_total,
            cost_usd=round(cost, 6),
        )
