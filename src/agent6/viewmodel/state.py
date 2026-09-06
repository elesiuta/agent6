# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Pure event-fold: list[event_dict] -> SessionState.

The wire form `session_state_as_dict` built from a SessionState is the data
contract for any external viewer (`attach --json`, the web page, a future
TS mirror): SessionState's fields plus `status`/`status_label`/`live`/
`operator_blocked`, with `log_tail` as plain strings. Keep its keys stable.

The fold itself does no I/O (dataclasses + an `apply_event` that returns a
new frozen `SessionState`, so "if state is state_prev, nothing changed");
`session_state_as_dict` with a session_dir also reads the dir's status probes and
manifest to fill the dir-backed fields.
"""

from __future__ import annotations

import contextlib
import functools
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Literal

from agent6.models.registry import context_window
from agent6.sessions.ipc import listening_ports
from agent6.sessions.layout import LOGS_NAME
from agent6.sessions.manifest import ManifestError, read_manifest
from agent6.viewmodel import events
from agent6.viewmodel.format import status_label
from agent6.viewmodel.listing import (
    LIVE_STATUS_WORDS,
    StatusFacts,
    status_for_session_dir,
    status_word,
)
from agent6.viewmodel.log_line import format_log_line, render_args
from agent6.viewmodel.policy import session_policy
from agent6.viewmodel.transcript import scrub_terminal_controls

NodeStatus = Literal["pending", "in_progress", "passed", "failed", "skipped", "obsolete"]


@dataclass(frozen=True, slots=True)
class TaskNodeView:
    """One node of the live task DAG, flattened (DFS pre-order) with a depth for
    tree rendering. Mirrors graph.models.TaskNode; fed by the `graph.update`
    snapshot the worker emits whenever it mutates its task breakdown."""

    id: str
    title: str
    status: NodeStatus = "pending"
    depth: int = 0
    is_cursor: bool = False


@dataclass(frozen=True, slots=True)
class ToolCallView:
    name: str
    args_preview: str  # rendered, per-value truncated, for the inline table
    args_full: str = ""  # rendered with a generous per-value cap, for the detail modal
    result_summary: str = ""
    ok: bool | None = None  # None = in-flight
    task_id: str | None = None  # DAG task in focus when the call ran (for filtering)
    call_id: int | None = None  # per-dispatch correlation id; None on id-less logs


@dataclass(frozen=True, slots=True)
class LogLine:
    """One audit-log line plus the DAG task in focus when it was emitted, so a
    viewer can filter the log to a selected task."""

    text: str
    task_id: str | None = None


@dataclass(frozen=True, slots=True)
class DiffView:
    """One auto-commit diff plus the task in focus when it landed."""

    patch: str
    task_id: str | None = None
    sha: str = ""


@dataclass(frozen=True, slots=True)
class VerifyView:
    cmd: tuple[str, ...]
    exit_code: int | None = None  # None = in-flight
    duration_s: float = 0.0
    stdout_tail: str = ""
    stderr_tail: str = ""


@dataclass(frozen=True, slots=True)
class BudgetView:
    # Token counters are the CURRENT leg's (they pair with the per-leg
    # enforcement caps); usd_total is CUMULATIVE across resume legs -- "cost"
    # on any surface means what the run cost, and the hub scanner
    # (listing.scan_session_log) sums legs the same way, so the surfaces agree.
    input_total: int = 0
    output_total: int = 0
    usd_total: float = 0.0
    usd_prior_legs: float = 0.0  # banked spend of completed resume legs
    usd_partial: bool = False  # True if some models had no price (under-estimate)
    usd_cap: float = 0.0  # [budget].max_usd for this leg (-1 unlimited, 0 unknown)
    tokens_unmetered: int = 0  # input+output tokens of calls the meter could not price
    tokens_fallback_cap: int = 0  # [budget].max_tokens_fallback (-1 unlimited, 0 unknown)
    # Subscription plan usage (percent-metered providers), leg-local like the
    # caps: the account's reported percent, this leg's consumed points, and
    # [budget].max_percent. 0s when no percent-metered call has run.
    plan_used_percent: float = 0.0
    plan_consumed: float = 0.0
    plan_cap: float = 0.0
    plan_resets_at: float = 0.0


@dataclass(frozen=True, slots=True)
class RoleCall:
    role: str
    model: str
    in_flight: bool
    # The provider that dialled the model (role.call carries it); pairs with
    # `model` for the registry's context-window lookup.
    provider: str = ""
    # Context size at the LAST COMPLETED call: the full prompt in tokens
    # (fresh input + cache reads + cache writes -- input_tokens is normalised
    # to fresh-only across providers). 0 until a result lands.
    ctx_tokens: int = 0
    # Live SSE text accumulator. Reset on every role.call,
    # appended-to on each role.text_delta, frozen on role.result.
    streamed_text: str = ""
    # Live reasoning accumulator, fed by role.thinking_delta. Same
    # lifecycle as streamed_text; shown in the TUI's "thinking" view so a
    # long reasoning burst reads as progress rather than a hang.
    streamed_thinking: str = ""


@dataclass(frozen=True, slots=True)
class CommitStep:
    """One per-step commit of the run: the iteration it closed, its sha, its
    subject. The dashboards select among these."""

    iteration: int
    sha: str
    subject: str


def approval_parts(prompt: str) -> tuple[str, str]:
    """An approval prompt's two parts: the head (`Allow run_command`, the
    question) and the payload (the command under judgment, possibly several
    lines). Every dispatch prompt is "Allow <tool>: <payload>"; one without a
    payload is all head. The CLI, TUI and web render exactly these two."""
    head, sep, payload = prompt.partition(": ")
    if sep and payload.strip():
        return head, payload
    return prompt, ""


@dataclass(frozen=True, slots=True)
class ApprovalPrompt:
    id: str
    prompt: str
    # False when no "allow all" is on offer for this prompt (see the event).
    standing: bool = True
    answered: bool = False
    approved: bool | None = None
    asked_ep: float | None = None  # for the waiting status's age

    @property
    def head(self) -> str:
        return approval_parts(self.prompt)[0]

    @property
    def payload(self) -> str:
        return approval_parts(self.prompt)[1]


@dataclass(frozen=True, slots=True)
class Question:
    """One question within an `ask_user` prompt. `options` are selectable presets;
    the user may also type a free-text answer."""

    question: str
    options: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class QuestionPrompt:
    """An agent->user `ask_user` prompt: one or more related questions the operator
    answers together (reviewing before submitting). `answers` align to `questions`.
    `from_harness`: agent6 itself asked (a start gate such as the dirty-tree
    question), decided by when it was asked: before the session started or
    after it finished, no model is running to ask anything."""

    id: str
    questions: tuple[Question, ...] = ()
    answered: bool = False
    answers: tuple[str, ...] = ()
    from_harness: bool = False
    asked_ep: float | None = None  # for the waiting status's age


@dataclass(frozen=True, slots=True)
class SessionState:
    session_id: str = ""
    user_task: str = ""
    tasks: tuple[TaskNodeView, ...] = ()  # live task DAG, DFS pre-order
    cursor_task_id: str | None = None
    last_role: RoleCall | None = None
    tool_calls: tuple[ToolCallView, ...] = ()  # most-recent-last, bounded
    last_verify: VerifyView | None = None
    budget: BudgetView = field(default_factory=BudgetView)
    pending_approvals: tuple[ApprovalPrompt, ...] = ()
    pending_questions: tuple[QuestionPrompt, ...] = ()
    log_tail: tuple[LogLine, ...] = ()  # most-recent-last, bounded
    log_count: int = 0  # monotonic total log lines ever (log_tail is windowed)
    recent_diffs: tuple[DiffView, ...] = ()  # auto-commit diffs, bounded, for task filtering
    started: bool = False  # a session.start was folded (a parked/created run has none)
    finished: bool = False
    all_passed: bool | None = None
    verify_scoped: bool = False  # session.end scoped: the judging gate ran scoped
    end_reason: str = ""  # session.end reason: finish_session | steer_abort | provider_error | ...
    undone_to: str = ""  # /undo's fork: the child session id surfaces follow
    undone_text: str = ""  # the message /undo took back (composer refill)
    finish_summary: str = ""  # the finish tool's summary: the agent's closing statement
    latest_diff: str = ""  # patch of the most recent auto-commit (diff.updated)
    # The run's per-step commits, oldest first (the dashboard's step selector).
    steps: tuple[CommitStep, ...] = ()
    # Monotonic count of mid-run steer requests (Ctrl-C). The TUI compares it
    # against its own "seen" count to react exactly once per press.
    steer_requests: int = 0
    # Context-compaction truth for the status surfaces: elision markers and
    # live gists in the CURRENT context (a demoted gist is back to a bare
    # marker; a tier-2 restart wipes every marker, so both reset there).
    compact_elided: int = 0
    compact_gists_live: int = 0
    # Operator /pin instructions recorded so far (loop.pin.added), most-recent-last.
    pins: tuple[str, ...] = ()
    # Epoch of the last folded event's own ts: the idle anchor every
    # "working… Ns" timer measures from, so replayed history reads as its true
    # age, never as fresh activity.
    last_event_ep: float | None = None


def initial_state() -> SessionState:
    return SessionState()


_MAX_TOOL_HISTORY = 50
_MAX_DIFF_HISTORY = 30  # auto-commit diffs retained for per-task filtering
MAX_LOG_TAIL = 400  # public: the inline log RichLog caps to this so it stays a gapless window
# Live streamed reasoning/text is the frontier of an in-flight call; keep only the
# tail so a 25k-char reasoning burst doesn't bloat every SSE frame or re-render.
# The full turn is preserved in the transcript, which the conversation view folds.
_STREAM_TAIL = 6000

# Streaming deltas are ephemeral live-view events -- the reasoning shows in the
# stream/conversation panes as it arrives. They are NOT audit-log events, so the
# log_tail and the full LogScreen skip them; otherwise a reasoning model floods the
# log with thousands of contentless "role.thinking_delta" lines.
STREAM_DELTA_EVENTS = frozenset({"role.thinking_delta", "role.text_delta"})
# Loop-side mirrors of events already rendered (tool.call carries the args,
# budget.update the totals); they doubled every tool call and budget tick in
# the log view without adding a field worth reading.
LOG_NOISE_EVENTS = frozenset({"loop.tool.call", "loop.budget"})


def _answered_only[PromptT: (ApprovalPrompt, QuestionPrompt)](
    prompts: tuple[PromptT, ...],
) -> tuple[PromptT, ...]:
    """Drop unanswered prompts at a leg boundary: they belong to the leg that
    died holding them, and the new leg re-asks with restarted ids (see the
    SessionStart/ResumeStart arms)."""
    return tuple(p for p in prompts if p.answered)


def apply_event(state: SessionState, event: dict[str, Any]) -> SessionState:  # noqa: PLR0911, PLR0912, PLR0915
    """Fold one event into the session state. Pure function.

    The event is parsed once (`events.parse_event`) into a typed family; each arm
    reads typed fields instead of sniffing the dict. The log line and the session_id
    peek still read the raw dict (they render arbitrary events, including the
    RawEvent long tail). An unknown/telemetry type folds to RawEvent -> no state
    change."""
    etype = event.get("type", "")
    if not state.session_id and event.get("session_id"):
        state = replace(state, session_id=str(event["session_id"]))
    # The idle anchor for every "working… Ns" timer: the EVENT's own ts, so a
    # viewer that replays history (attach, the web/TUI catch-up) measures from
    # when the run last spoke, not from when it started watching -- an arrival
    # anchor would read "working… 3s" on a run wedged 40 minutes.
    if (ep := events.event_epoch(event.get("ts"))) is not None:
        state = replace(state, last_event_ep=ep)
    if etype not in STREAM_DELTA_EVENTS and etype not in LOG_NOISE_EVENTS:
        # Deltas are live-stream only; noise mirrors add no readable field.
        # cursor_task_id is the focus task (graph.update lands before a turn's calls).
        entry = LogLine(format_log_line(event), state.cursor_task_id)
        new_log = _push_bounded(state.log_tail, entry, MAX_LOG_TAIL)
        # log_count is monotonic; log_tail is a sliding window. A live viewer must
        # diff on the count (which keeps growing) -- diffing on len(log_tail) freezes
        # the panel once the window saturates at MAX_LOG_TAIL.
        state = replace(state, log_tail=new_log, log_count=state.log_count + 1)

    match events.parse_event(event):
        case events.SessionStart(user_task=task):
            # A session.start begins a leg: by definition it is running. The ask REPL
            # re-enters wf.run() per follow-up on the same log, so a second
            # session.start must clear the prior leg's terminal state. Unlike
            # ResumeStart, do NOT bank usd: the REPL reuses one BudgetTracker,
            # so usd_total is already cumulative across legs.
            return replace(
                state,
                user_task=task,
                started=True,
                finished=False,
                end_reason="",
                pending_approvals=_answered_only(state.pending_approvals),
                pending_questions=_answered_only(state.pending_questions),
            )

        case events.ResumeStart():
            # A resume restarts a finished/stopped run in place (it appends to the
            # same log): it is running again, so clear the terminal state. The new
            # leg's budget counters start fresh, so bank the cumulative spend now
            # (usd_total keeps its value until the leg's first budget.update) and
            # zero the token counters/caps: BudgetView documents them as the
            # CURRENT leg's, and scan_session_log resets for the same reason.
            # Unanswered prompts are the DEAD leg's: the resumed leg re-asks
            # with restarted ids, so a held-over orphan would read "waiting"
            # forever and duplicate when the same id is re-prompted.
            return replace(
                state,
                # `started` = a leg has begun, NOT "a session.start was seen": a
                # fork is driven by resume(), so its fresh log never carries one.
                started=True,
                finished=False,
                end_reason="",
                pending_approvals=_answered_only(state.pending_approvals),
                pending_questions=_answered_only(state.pending_questions),
                budget=replace(
                    state.budget,
                    usd_prior_legs=state.budget.usd_total,
                    input_total=0,
                    output_total=0,
                    usd_cap=0.0,
                    tokens_unmetered=0,
                    tokens_fallback_cap=0,
                ),
            )

        case events.GraphUpdate(nodes=nodes, cursor=cursor):
            return replace(
                state,
                tasks=task_tree_views(nodes, cursor),
                cursor_task_id=cursor,
            )

        case events.AutoCommit(iteration=iteration, sha=sha, subject=subject):
            if not sha:
                return state
            step = CommitStep(iteration=iteration, sha=sha, subject=subject)
            return replace(state, steps=(*state.steps, step))

        case events.DiffUpdated(patch=patch, sha=sha):
            entry = DiffView(patch=patch, task_id=state.cursor_task_id, sha=sha)
            return replace(
                state,
                latest_diff=patch,
                recent_diffs=_push_bounded(state.recent_diffs, entry, _MAX_DIFF_HISTORY),
            )

        case events.RoleCall(role=role, model=model, provider=provider):
            prior = state.last_role
            return replace(
                state,
                last_role=RoleCall(
                    role=role,
                    model=model,
                    in_flight=True,
                    provider=provider,
                    # Keep the last known context size until this call's result
                    # lands, so the readout doesn't blink to nothing per turn.
                    ctx_tokens=prior.ctx_tokens if prior is not None else 0,
                    streamed_text="",
                    streamed_thinking="",
                ),
            )

        case events.RoleTextDelta(text=piece):
            # Append SSE delta to the in-flight RoleCall. Scrub the
            # CONCATENATION: an escape sequence can arrive split across deltas,
            # and per-piece scrubbing would let the reassembled whole through.
            last = state.last_role
            if last is None or not last.in_flight or not piece:
                return state
            joined = scrub_terminal_controls(last.streamed_text + piece)
            return replace(
                state,
                last_role=replace(last, streamed_text=joined[-_STREAM_TAIL:]),
            )

        case events.RoleThinkingDelta(text=piece):
            # Append a reasoning delta to the in-flight RoleCall (scrubbed as a
            # whole, like the text deltas above).
            last = state.last_role
            if last is None or not last.in_flight or not piece:
                return state
            joined = scrub_terminal_controls(last.streamed_thinking + piece)
            return replace(
                state,
                last_role=replace(last, streamed_thinking=joined[-_STREAM_TAIL:]),
            )

        case events.RoleResult(tokens_in=tin, cache_read=cr, cache_creation=cc):
            last = state.last_role
            if last is None:
                return state
            # The full prompt of this call = the context size right now.
            ctx = tin + cr + cc
            return replace(
                state,
                last_role=replace(
                    last, in_flight=False, ctx_tokens=ctx if ctx > 0 else last.ctx_tokens
                ),
            )

        case events.ToolCall(name=name, args=raw_args, call_id=cid):
            tc = ToolCallView(
                name=name,
                args_preview=render_args(raw_args),
                args_full=render_args(raw_args, max_value=4000),
                ok=None,
                task_id=state.cursor_task_id,
                call_id=cid,
            )
            # The finish tools' summary is the agent's closing statement; keep it
            # so an ended run's panes can render the end story, not a dead one.
            finish_summary = state.finish_summary
            if name in ("finish_session", "finish_planning") and isinstance(raw_args, dict):
                finish_summary = str(raw_args.get("summary", "")).strip() or finish_summary
            return replace(
                state,
                tool_calls=_push_bounded(state.tool_calls, tc, _MAX_TOOL_HISTORY),
                finish_summary=finish_summary,
            )

        case events.ToolResult(name=name, ok=ok, summary=summary, call_id=cid):
            if not state.tool_calls:
                return state
            if cid is not None:
                # Pair on the stamped id: concurrent seats interleave events, so
                # the matching call is not necessarily the last entry.
                for i in range(len(state.tool_calls) - 1, -1, -1):
                    if state.tool_calls[i].call_id == cid:
                        updated = replace(state.tool_calls[i], ok=ok, result_summary=summary)
                        return replace(
                            state,
                            tool_calls=(
                                *state.tool_calls[:i],
                                updated,
                                *state.tool_calls[i + 1 :],
                            ),
                        )
                return state
            # Id-less (historical) event: the sequential last-entry pairing.
            last = state.tool_calls[-1]
            if last.name != name:
                return state
            updated_last = replace(last, ok=ok, result_summary=summary)
            return replace(
                state,
                tool_calls=(*state.tool_calls[:-1], updated_last),
            )

        case events.VerifyStart(cmd=cmd):
            return replace(state, last_verify=VerifyView(cmd=cmd))

        case events.VerifyEnd(
            cmd=cmd, exit_code=code, duration_s=dur, stdout_tail=out, stderr_tail=err
        ):
            return replace(
                state,
                last_verify=VerifyView(
                    cmd=cmd, exit_code=code, duration_s=dur, stdout_tail=out, stderr_tail=err
                ),
            )

        case events.BudgetUpdate(
            input_total=it,
            output_total=ot,
            usd_total=usd,
            usd_partial=partial,
            usd_cap=ucap,
            tokens_unmetered=unmet,
            tokens_fallback_cap=fcap,
            plan_used_percent=plan_pct,
            plan_consumed=plan_used,
            plan_cap=plan_cap,
            plan_resets_at=plan_resets,
        ):
            # The event's usd_total is the current LEG's; the view's is
            # cumulative. usd_partial is sticky: unpriced spend in any prior
            # leg keeps the cumulative total an under-estimate.
            return replace(
                state,
                budget=BudgetView(
                    input_total=it,
                    output_total=ot,
                    usd_total=state.budget.usd_prior_legs + usd,
                    usd_prior_legs=state.budget.usd_prior_legs,
                    usd_partial=partial or state.budget.usd_partial,
                    usd_cap=ucap,
                    tokens_unmetered=unmet,
                    tokens_fallback_cap=fcap,
                    plan_used_percent=plan_pct,
                    plan_consumed=plan_used,
                    plan_cap=plan_cap,
                    plan_resets_at=plan_resets,
                ),
            )

        case events.ApprovalPrompt(id=aid, prompt=prompt, standing=standing, asked_ep=asked_ep):
            ap = ApprovalPrompt(id=aid, prompt=prompt, standing=standing, asked_ep=asked_ep)
            return replace(state, pending_approvals=(*state.pending_approvals, ap))

        case events.ApprovalAnswer(id=wanted_id, approved=approved):
            new = tuple(
                replace(a, answered=True, approved=approved) if a.id == wanted_id else a
                for a in state.pending_approvals
            )
            return replace(state, pending_approvals=new)

        case events.QuestionPrompt(id=qid, questions=qs, asked_ep=asked_ep):
            questions = tuple(Question(question=q.question, options=q.options) for q in qs)
            qp = QuestionPrompt(
                id=qid,
                questions=questions,
                from_harness=not state.started or state.finished,
                asked_ep=asked_ep,
            )
            return replace(state, pending_questions=(*state.pending_questions, qp))

        case events.QuestionAnswer(id=wanted, answers=answers):
            new_q = tuple(
                replace(q, answered=True, answers=answers) if q.id == wanted else q
                for q in state.pending_questions
            )
            return replace(state, pending_questions=new_q)

        case events.PinAdded(text=text):
            return replace(state, pins=(*state.pins, text))

        case events.PinsRestored(pins=pins):
            return replace(state, pins=pins)

        case events.CompactRestored(elided=elided, gists=gists):
            return replace(state, compact_elided=elided, compact_gists_live=gists)

        case events.CompactDropped(n=n):
            return replace(state, compact_elided=state.compact_elided + n)

        case events.CompactGists(gisted=gisted, demoted=demoted):
            live = max(0, state.compact_gists_live + gisted - demoted)
            return replace(state, compact_gists_live=live)

        case events.CompactSummarised():
            return replace(state, compact_elided=0, compact_gists_live=0)

        case events.SteerRequested():
            return replace(state, steer_requests=state.steer_requests + 1)

        case events.SessionEnd(all_passed=all_passed, reason=reason, scoped=scoped):
            return replace(
                state,
                finished=True,
                all_passed=all_passed,
                end_reason=reason,
                verify_scoped=scoped,
            )

        case events.SessionUndone(new_session_id=new_id, undone_text=text):
            return replace(state, undone_to=new_id, undone_text=text)

        case events.RawEvent():
            return state


def task_tree_views(nodes: dict[str, Any], cursor: str | None) -> tuple[TaskNodeView, ...]:
    """Flatten the curator's node map into a DFS pre-order list with depths, so
    the TUI can render the DAG as an indented tree. Roots are nodes with no
    parent (or whose parent is missing); children follow their parent's recorded
    order. Cycles/dupes are guarded by a visited set."""
    out: list[TaskNodeView] = []
    seen: set[str] = set()

    def visit(nid: str, depth: int) -> None:
        node = nodes.get(nid)
        # isinstance (not `is None`) so a malformed non-dict value is skipped
        # rather than crashing .get(), consistent with the roots filter below.
        if not isinstance(node, dict) or nid in seen:
            return
        seen.add(nid)
        out.append(
            TaskNodeView(
                id=nid,
                title=str(node.get("title", "")),
                status=node.get("status", "pending"),
                depth=depth,
                is_cursor=(nid == cursor),
            )
        )
        for child in node.get("children", ()) or ():
            visit(str(child), depth + 1)

    roots = [
        nid
        for nid, n in nodes.items()
        if not isinstance(n, dict) or n.get("parent_id") is None or n.get("parent_id") not in nodes
    ]
    for nid in roots:
        visit(nid, 0)
    # Any node not reachable from a root (shouldn't happen) still gets shown.
    for nid in nodes:
        visit(nid, 0)
    return tuple(out)


def _push_bounded[T](existing: tuple[T, ...], item: T, cap: int) -> tuple[T, ...]:
    new = (*existing, item)
    if len(new) > cap:
        return new[-cap:]
    return new


def fold_session(events: Iterable[dict[str, Any]]) -> SessionState:
    """Reduce a session's whole event stream to one SessionState (apply_event from the
    initial state). The snapshot a one-shot viewer or the JSON wire form builds
    on; the TUI folds incrementally and a CLI tail renders line-by-line instead."""
    state = initial_state()
    for event in events:
        state = apply_event(state, event)
    return state


def open_question(session_dir: Path) -> QuestionPrompt | None:
    """The run's unanswered `ask_user` prompt, oldest first; None when none is
    open. Every surface that writes an answer file checks it against this, so
    an answer list of the wrong length is refused instead of consumed and
    thrown away by the asking side."""
    from agent6.viewmodel.tail import tail_events  # noqa: PLC0415 -- cycle at import time

    state = fold_session(tail_events(session_dir / LOGS_NAME, follow=False))
    return next((q for q in state.pending_questions if not q.answered), None)


def fold_until_commit(events: Iterable[dict[str, Any]], sha: str) -> SessionState | None:
    """The state as of one of the run's commits: every event up to and
    including its loop.auto_commit folded, later ones dropped (the details a
    step selector time-travels to). *sha* is the full sha or a prefix of at
    least 7 hex digits (the first commit it matches wins). None when no
    commit has it."""
    if len(sha) < 7:
        return None
    state = initial_state()
    for event in events:
        state = apply_event(state, event)
        if state.steps and state.steps[-1].sha.startswith(sha):
            return state
    return None


def session_status_label(state: SessionState) -> str:
    """The status label for a stream with genuinely NO run dir (the
    `attach --json` wire form). It distinguishes a stop from a finish from an
    error (all three set finished=True; the reason tells them apart) via the
    shared `listing.status_word`, but it folds events alone, so it reads
    every unfinished state as "running". A surface that HAS a run dir must call
    `listing.status_for_session_dir` with `status_facts` instead -- the dir
    knows parked/starting/created/stale/waiting, this cannot."""
    word, reason = status_word(
        finished=state.finished,
        all_passed=state.all_passed,
        end_reason=state.end_reason,
        scoped=state.verify_scoped,
    )
    return status_label(word, reason)


def status_facts(state: SessionState) -> StatusFacts:
    """The fold's answers to the status questions -- the typed twin of
    `LogScan.status_facts()`, for surfaces that hold a `SessionState`. The two
    producers must agree on the same log (pinned by the status matrix test)."""
    pending: list[tuple[str, float | None]] = [
        ("approval", a.asked_ep) for a in state.pending_approvals if not a.answered
    ] + [("question", q.asked_ep) for q in state.pending_questions if not q.answered]
    oldest = min(pending, key=lambda p: p[1] if p[1] is not None else float("inf"), default=None)
    return StatusFacts(
        started=state.started,
        finished=state.finished,
        all_passed=state.all_passed,
        verify_scoped=state.verify_scoped,
        end_reason=state.end_reason,
        operator_blocked=bool(pending),
        blocked_kind=oldest[0] if oldest else "",
        blocked_since_ep=oldest[1] if oldest else None,
    )


@functools.lru_cache(maxsize=64)
def _window(provider: str, model: str) -> int | None:
    # The registry reads the bundled table and the provider's model cache file;
    # every surface asks per heartbeat, so the answer is memoised per model.
    return context_window(provider, model)


def context_fill(state: SessionState) -> int | None:
    """Context-window fill (percent) at the last completed model call: the
    call's full prompt tokens over the model's window (bundled priors, else the
    provider listing cache). None until both sides are known. The one rule
    behind every surface's `ctx N%` readout."""
    role = state.last_role
    if role is None or role.ctx_tokens <= 0 or not role.model:
        return None
    window = _window(role.provider, role.model)
    if not window:
        return None
    return min(100, round(100 * role.ctx_tokens / window))


def session_state_as_dict(state: SessionState, session_dir: Path | None = None) -> dict[str, Any]:
    """The JSON-able wire form of a SessionState, stable field names: what
    `agent6 attach --json` and a web client serialize. Tuples become lists, nested
    view dataclasses become dicts. `status_label` is a computed convenience the
    web/CLI render verbatim so the label logic lives in one place.

    Pass *session_dir* whenever the caller has one: the label is then THE dir-aware
    status (parked/starting/stale/waiting, not the fold's blanket "running"),
    `live` says whether steer/stop/compact would reach anything, `ports` lists
    what the run's network is serving, a plan's `plan_md` is its written
    deliverable, and the dir-backed identity (session_id, the manifest's
    user_task) fills what the fold left empty.
    Without it the payload keeps the fold-only label and `live: None` --
    correct only for a genuinely dir-less stream (the machine reasoning
    snapshot)."""
    d = asdict(state)
    d["context_pct"] = context_fill(state)
    for ap, row in zip(state.pending_approvals, d["pending_approvals"], strict=True):
        row["head"], row["payload"] = ap.head, ap.payload
    if session_dir is not None:
        word, reason = status_for_session_dir(session_dir, status_facts(state))
        d["live"] = word in LIVE_STATUS_WORDS
        # The dir is authoritative for identity: a resumed/forked leg's log can
        # start at loop.resume.start, folding session_id/user_task empty. Fill them
        # HERE so every consumer (web, watch, SSE) carries the same identity.
        # The same fold the CLI banner and the TUI composer read, so a web
        # client cannot show a different answer.
        d["policy"] = session_policy(session_dir).line()
        d["session_id"] = d["session_id"] or session_dir.name
        # The MODE is dir-backed identity too: without it a client cannot say
        # WHAT it is showing and heads every session "Run", right one time in
        # three.
        d["mode"] = d.get("mode") or ""
        with contextlib.suppress(ManifestError):
            manifest = read_manifest(session_dir)
            d["user_task"] = d["user_task"] or manifest.user_task
            d["mode"] = d["mode"] or manifest.mode
        # What the run is serving (a dev server the agent started): the ports
        # its session network listens on, reachable via `agent6 forward`.
        # A live probe, [] once the network is gone.
        d["ports"] = listening_ports(session_dir)
        if d["mode"] == "plan":
            # The planning run's deliverable (`agent6 plan show` prints the
            # same file), per frame: it lands when the plan finishes.
            with contextlib.suppress(OSError):
                d["plan_md"] = (session_dir / "plan.md").read_text(encoding="utf-8")
    else:
        # A genuinely dir-less stream (the machine reasoning snapshot):
        # liveness is unknowable here.
        d["live"] = None
        word, reason = status_word(
            finished=state.finished,
            all_passed=state.all_passed,
            end_reason=state.end_reason,
            scoped=state.verify_scoped,
        )
    # The raw status WORD, not only the human label, so a client can branch on it
    # -- e.g. render the waiting line instead of the "working" heartbeat when the
    # run is blocked on the operator (a "waiting" run is still LIVE).
    d["status"] = word
    d["status_label"] = status_label(word, reason)
    # Whether an operator prompt is unanswered, straight from the fold: a DIR-LESS
    # consumer (the machine watch folds an agent-state log with no session_dir, so it
    # has no dir status) still needs the "blocked, not working" signal to quiet
    # its heartbeat.
    d["operator_blocked"] = status_facts(state).operator_blocked
    # log_tail is LogLine objects now; the wire form stays a flat list of strings
    # (web + `watch --json` consumers render lines verbatim). task_id filtering is
    # a TUI-local concern that reads the SessionState directly.
    d["log_tail"] = [line.text for line in state.log_tail]
    return d
