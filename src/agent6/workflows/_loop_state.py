# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The agent loop's mutable state shapes.

`LoopState` is the per-run bookkeeping threaded through every loop phase,
`TurnState` the per-turn slice one dispatching iteration accumulates, and
`NEXT_TURN` the sentinel for a discarded turn. `restore_completion_state`
carries a resume snapshot's completion-relevant fields into fresh state.
The loop's phase methods live in `loop.py`; these are the shapes they take.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Literal

from agent6.providers import ProviderResponse
from agent6.workflows._conversation import AssistantTurn, Notice, ToolResultItem
from agent6.workflows._metric import MetricSample
from agent6.workflows._session_state import SessionSnapshot
from agent6.workflows._spiral_guards import SpiralGuard
from agent6.workflows._verify_verdict import VerifyVerdict


@dataclass(slots=True)
class LoopState:
    """Mutable per-run bookkeeping threaded through the agent loop.

    The loop accumulates cross-iteration state: how often each intervention
    (review rejection, went-quiet / plateau / early-finish nudge, plan/run
    budget nudge) has fired against its cap, the degenerate-repeat-call guard,
    and verify-settled completion tracking. Holding it in one object lets the
    loop's phases be methods that take `state` rather than a long parameter
    list, so adding an intervention is a one-field change, not another local
    threaded by hand.
    """

    original_task: str
    tool_calls: int
    # Dispatches that EXECUTED (no ToolError): the standing-goal spin guard
    # reads this, so a refused call (a malformed edit, a retirement the
    # curator rejects) is not "work since the last re-entry" -- a goal round
    # that produced nothing ends the run instead of re-entering to its budget.
    ok_tool_calls: int = 0
    # Rulings this leg appended to DECISIONS.md, for the finish-time check.
    decisions_recorded: list[str] = field(default_factory=list)
    metric_history: list[MetricSample] = field(default_factory=list)
    # Tier-2 re-fires only after the context grew 25% past the last restart's
    # size: a restart that lands near the threshold (tiny explicit thresholds,
    # or a huge kept tail) must not summarise every other iteration. Leg-local
    # (not snapshotted): a resumed leg rebuilds a small context anyway.
    tier2_floor_chars: int = 0
    # The worktree tree the metric was last sampled on: one reading per state
    # of the tree, whether or not the harness commits between them.
    metric_tree: str = ""
    # Consecutive before_finish review rejections, so a stubborn worker can't
    # burn the budget bouncing off the panel.
    consecutive_review_rejections: int = 0
    # Per-run TOTAL review-panel blocks (persisted across resume). Decays on a
    # pass; once it hits review_max_total_rejections the gate auto-disarms to
    # advisory for the rest of the run (oscillation can't burn the budget).
    review_rejections_total: int = 0
    # The verify verdict: the one object every "is the run green" consumer
    # reads (gates, review grounding, snapshot, notices).
    verify: VerifyVerdict = field(default_factory=VerifyVerdict)
    no_progress_nudges_used: int = 0
    # The dispatch loop's degenerate-spiral bookkeeping (repeat + error
    # streaks and the last-served bytes), transitions owned by the guard.
    spiral: SpiralGuard = field(default_factory=SpiralGuard)
    # Sandbox-reachability signal: argv[0] of a run_command the JAIL failed to
    # exec (exec_failed, not a nonzero exit) and its consecutive-failure count.
    # Only executed commands feed it; validation errors and denials never
    # entered the jail, so they say nothing about reachability.
    jail_exec_failed_binary: str = ""
    jail_exec_failed_streak: int = 0
    sandbox_reachability_warned: bool = False
    # Intervention nudge counters (each capped by a module-level patience
    # const). Leg-local BY DESIGN, like every counter below not named in
    # SessionSnapshot: resume is operator-initiated everywhere, so a resumed
    # leg's refreshed patience is the operator deliberately granting another
    # window. Only the completion-relevant subset persists (see
    # restore_completion_state).
    went_quiet_nudges_used: int = 0
    plateau_nudges_used: int = 0
    metric_finish_nudges_used: int = 0
    task_finish_nudges_used: int = 0
    # Red finish certifications returned to the model so far (`verify_retries`).
    verify_finish_retries_used: int = 0
    ever_edited: bool = False
    # Attemptless-stagnation notice: recall spirals make few calls with long
    # reasoning between them, so the identical-signature repeat guard never
    # sees them; wall clock with zero attempts does. Monotonic is
    # process-relative, so neither field persists to the snapshot: a resumed
    # run gets a fresh window.
    started_monotonic: float = field(default_factory=time.monotonic)
    stagnation_nudged: bool = False
    silent_no_work_nudges_used: int = 0
    plan_finish_nudged: bool = False
    # plan mode: the plan.md text the planner was last shown. Fresh per leg, so a
    # resumed planner is always re-shown the file the operator may have edited.
    plan_injected: str = ""
    # A turn that ends in a prose question with no tool_use is nudged ONCE to
    # call ask_user (or finish_session) instead of narrating; then silent_finish is
    # accepted so a model that ignores the nudge cannot loop the run.
    question_nudged: bool = False
    # verify-settled completion (run mode): once verify has passed -- or, on a
    # gateless run, once an edit has been committed -- count no-progress
    # iterations; nudge then stop a worker that spins after success.
    gateless_ever_committed: bool = False
    verify_settled_idle: int = 0
    verify_settled_nudged: bool = False
    # Standing-goal re-entry bookkeeping: `ok_tool_calls` at the last
    # absorption (-1 = never absorbed), and the consecutive fruitless
    # re-entries since work last landed. `[workflow].standing_patience`
    # decides how many fruitless rounds are absorbed before ends are
    # honoured (-1 = never on its own).
    standing_tools_mark: int = -1
    standing_fruitless: int = 0
    run_budget_nudged: bool = False
    # Cross-run memory write nudges (run mode, memory store wired): one flip
    # advisory when verify first goes green after failing, one deferred
    # finish_session as the backstop. Both suppressed once the worker records
    # anything; a run whose verify never failed is never nudged.
    memory_written: bool = False
    memory_flip_nudged: bool = False
    memory_finish_nudged: bool = False
    # surface-current-task: id of the subtask last injected as the focus banner.
    # Re-surface only on a focus change or after a tier-2 restart (reset to None
    # there) -- the banner survives tier-1 elision, so the worker keeps seeing it
    # between those events without re-appending it every turn.
    surfaced_task_id: str | None = None
    # anti-grind: the focus task being counted, how many consecutive turns it has
    # held (NOT reset by compaction -- only by forward motion), and how many stuck
    # nudges have fired for THIS focus task (reset on focus change; capped).
    last_focus_id: str | None = None
    turns_on_task: int = 0
    stuck_nudges_fired: int = 0
    # DAG root task id (set once by _drive_loop), so a steer-boundary phase can
    # parent a node without threading it through every call site.
    root_task_id: str | None = None
    # The system prompt (set once by _drive_loop), for the same reason: a
    # steer-boundary phase that must snapshot needs it and is not handed it.
    system: str = ""
    # How many `/parallel` sibling groups this run has dispatched. Names each
    # group's lanes (`<run-id>-p<seq>-l<i>`); increments per dispatch.
    parallel_groups_dispatched: int = 0
    # Operator `/pin` instructions, re-injected verbatim after every tier-2
    # restart. Total chars capped at PINS_MAX_CHARS; persisted in the snapshot.
    pins: list[str] = field(default_factory=list)


def restore_completion_state(state: LoopState, snap: SessionSnapshot) -> None:
    """Carry a resume snapshot's completion-relevant bookkeeping into fresh loop
    state, so the review gate-disarm, metric, and verify-settled stop logic don't
    regress to zero after a resume (re-rejecting a correct finish_session, re-counting
    idle). A fresh run() never calls this and keeps LoopState's defaults. Adding a
    persisted completion field is one field on SessionSnapshot plus one line here."""
    state.review_rejections_total = snap.review_rejections_total
    state.verify.ever_passed = snap.verify_ever_passed
    state.verify.scoped = snap.verify_scoped
    state.gateless_ever_committed = snap.gateless_ever_committed
    state.parallel_groups_dispatched = snap.parallel_groups_dispatched
    state.pins = list(snap.pins)
    if snap.metric_at_ceiling or snap.metric_best_score is not None:
        # Seed one synthetic sample so `_metric_at_ceiling` and the plateau guard
        # see the prior best (we persist a compact summary, not the full history,
        # by design). `label` marks it resume-reconstructed. Consequence:
        # `metric_plateau_summary` needs several parsed samples to fire, so a
        # resumed already-plateaued run takes a few measurements to re-arm the
        # plateau-stop (it never stops early; the ceiling-stop is immediate) -- the
        # predictable trade for not carrying the whole sample history across resume.
        state.metric_history.append(
            MetricSample(
                label="resumed",
                score=snap.metric_best_score,
                returncode=0,
                at_ceiling=snap.metric_at_ceiling,
            )
        )


class NextTurn:
    """Sentinel returned by `_turn_provider_call`: the turn was discarded (a
    mid-stream steer chose continue, or injected an instruction) and the loop
    should start the next iteration immediately."""


NEXT_TURN = NextTurn()


@dataclass(slots=True)
class TurnState:
    """Mutable bookkeeping for ONE assistant turn that dispatched tools.

    `_drive_loop` creates one per tool-use iteration and threads it through
    the turn phases; a field earns its place by being written in one phase and
    read in a later one, so each phase is a method taking `(state, turn)`
    rather than a slice of ~15 hand-threaded locals. Cross-iteration state
    stays on `LoopState`.
    """

    iteration: int
    # The provider response driving this turn.
    resp: ProviderResponse
    # The response's turn in the conversation; its parsed tool_uses drive the
    # dispatch (the conversation is the single source of what was called).
    assistant: AssistantTurn
    # A finish_session/finish_planning call captured this turn; the finish gates
    # may revoke it (set back to None) before the stop checks honour it.
    finish_signal: str | None = None
    finish_payload: dict[str, Any] | None = None
    # An end the harness or the model declared without finish_session (a
    # settled stop, a silent finish) that a gate handed back this turn.
    end_returned: bool = False
    # A finish that declared the configured gate stale, with the replacement it
    # proposes. Recorded and surfaced; the gate itself never moves.
    finish_stale_gate: str = ""
    finish_kind: Literal["finish_session", "finish_planning"] = "finish_session"
    # The user-turn items accumulated for this turn: tool results in dispatch
    # order, with advisory notices (review, metric, nudges) appended after
    # (or, for the broken-verify flag, between them).
    tool_results: list[ToolResultItem | Notice] = field(default_factory=list)
    verify_just_passed: bool = False
    verify_just_failed: bool = False
    # Verify went green THIS turn after the run's last verify was red; feeds
    # the one-shot memory flip advisory in _turn_notices.
    verify_flipped_green: bool = False
    # An apply_edit/apply_patch AFTER a passing verify in the same turn changes
    # the tree that verify validated, so the green no longer applies. Tracked
    # separately from verify_just_passed (which the metric path also reads) so
    # only the auto-commit gate is affected.
    edit_since_verify_pass: bool = False
    edited: bool = False
    committed: bool = False
    dag_mutated: bool = False
    metric_after_verify_pass: bool = False
    metric_feedback: str | None = None
    metric_plateau_finish: str | None = None
    review_text: str | None = None
    plateau_should_stop: bool = False
    verify_settled_stop: bool = False
    no_progress_stop: bool = False
    tool_error_stop: bool = False
