# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Typed read model for the logs.jsonl event families the SessionState fold consumes.

The write side (`agent6.events.EventSink`) appends free-form `{"type", "ts",
**fields}` dicts and never validates; ~90 distinct types exist. The SessionState fold
(`viewmodel.state.apply_event`) structurally consumes only the families defined
here. `parse_event` turns one raw event dict into exactly one of those frozen
families, or a `RawEvent` passthrough for every other type, the compatibility
surface that keeps old run dirs folding: a type this module does not know becomes
`RawEvent`, which the fold drops, never a crash.

Hand-rolled frozen dataclasses, not pydantic (unlike `machine/journal.py`):
logs.jsonl is append-only history, so the fold keeps the exact coercion
semantics old run dirs were written against (`str()`/`int()`/`bool()` with
per-field defaults, `_as_int`'s swallow-to-zero, the isinstance guards). A
pydantic model would impose its own coercion and validation-failure semantics,
changing how a malformed old line folds; these parsers hold that coercion in
one place per family. `parse_event` never raises on an unknown type (RawEvent)
but keeps the latent raise on a non-coercible known field (e.g. `verify.end`
exit_code).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

# A run SESSION begins: a fresh run() emits session.start; a resumed leg emits only
# loop.resume.start (never a second session.start). Per-process state that restarts
# at a session boundary -- the prompt-id counters (approval-1/question-1 again),
# a screen's live/finished tracking, the receipt's mode -- must key on BOTH;
# keying on session.start alone made every resumed leg invisible to that state
# (swallowed modals, a steer bar that mislabeled a live leg). One definition so
# the folds can't drift.
SESSION_START_EVENTS = frozenset({"session.start", "loop.resume.start"})


def event_epoch(value: object) -> float | None:
    """Parse an event `ts` to epoch seconds, or None if unparseable.

    EventSink writes `ts` as an ISO-8601 string (`datetime.isoformat`),
    so the elapsed-time anchor must parse that, not only bare numbers.
    """
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value).timestamp()
        except ValueError:
            return None
    return None


def tool_result_ok(value: Any) -> bool:
    """The persisted `tool.result.ok` flag, tolerating the historical
    stringified form: "True" is ok, "False" (and everything else) is not. THE
    one coercion both folds use, so a run-state surface and the conversation
    can never disagree on a tool's verdict."""
    return value in (True, "True")


def readable_summary(value: Any) -> str:
    """A tool result's `summary` should be a string; a malformed dict/list value
    renders as neutral JSON, not the single-quoted Python repr `str()` produces
    (which leaked `{'unexpected': ...}` into the web/TUI tool detail + log tail)."""
    if isinstance(value, str):
        return value
    if isinstance(value, (dict, list)):
        try:
            return json.dumps(value, default=str)
        except (TypeError, ValueError):
            return str(value)
    return str(value)


def _as_int(value: object) -> int:
    """An event field as an int; 0 for anything unusable (untrusted log data)."""
    try:
        return int(value)  # type: ignore[arg-type]  # int() rejects bad types itself
    except (TypeError, ValueError):
        return 0


@dataclass(frozen=True, slots=True)
class SessionStart:
    user_task: str


@dataclass(frozen=True, slots=True)
class ResumeStart:
    """loop.resume.start: a finished/stopped run restarts in place."""


@dataclass(frozen=True, slots=True)
class GraphUpdate:
    # The node map is walked defensively by the tree builder (isinstance guards for
    # cycles, dupes, and malformed non-dict values), so it stays raw here.
    nodes: Any
    cursor: str | None


@dataclass(frozen=True, slots=True)
class DiffUpdated:
    patch: str
    sha: str


@dataclass(frozen=True, slots=True)
class AutoCommit:
    """One per-step commit on the run's chain (`loop.auto_commit`)."""

    iteration: int
    sha: str
    subject: str


@dataclass(frozen=True, slots=True)
class RoleCall:
    role: str
    model: str
    provider: str


@dataclass(frozen=True, slots=True)
class RoleResult:
    tokens_in: int
    cache_read: int
    cache_creation: int


@dataclass(frozen=True, slots=True)
class RoleTextDelta:
    text: str


@dataclass(frozen=True, slots=True)
class RoleThinkingDelta:
    text: str


@dataclass(frozen=True, slots=True)
class ToolCall:
    name: str
    # Raw args: rendered per-value and isinstance-checked for the finish summary,
    # so a non-dict value degrades exactly as the fold did inline.
    args: Any
    # Correlation id stamped per dispatch; None on historical id-less logs.
    call_id: int | None = None


@dataclass(frozen=True, slots=True)
class ToolResult:
    name: str
    ok: bool
    summary: str
    call_id: int | None = None


@dataclass(frozen=True, slots=True)
class VerifyStart:
    cmd: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class VerifyEnd:
    cmd: tuple[str, ...]
    exit_code: int
    duration_s: float
    stdout_tail: str
    stderr_tail: str


@dataclass(frozen=True, slots=True)
class BudgetUpdate:
    input_total: int
    output_total: int
    usd_total: float
    usd_partial: bool
    usd_cap: float
    tokens_unmetered: int
    tokens_fallback_cap: int
    # Subscription plan usage (percent-metered providers); 0s when absent,
    # and on logs written before the fields existed.
    plan_used_percent: float = 0.0
    plan_consumed: float = 0.0
    plan_cap: float = 0.0
    plan_resets_at: float = 0.0


@dataclass(frozen=True, slots=True)
class ApprovalPrompt:
    id: str
    prompt: str
    # Whether an "allow all" would actually cover anything beyond this call, so
    # a front-end only offers the button when it means something. A log written
    # before the field existed folds True, the old behaviour.
    standing: bool = True
    # When it was asked (epoch), for the waiting status's age; None on a log
    # whose line carried no parseable ts.
    ts_ep: float | None = None


@dataclass(frozen=True, slots=True)
class ApprovalAnswer:
    id: str
    approved: bool


@dataclass(frozen=True, slots=True)
class EventQuestion:
    question: str
    options: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class QuestionPrompt:
    id: str
    questions: tuple[EventQuestion, ...]
    ts_ep: float | None = None  # asked-at epoch, for the waiting status's age


@dataclass(frozen=True, slots=True)
class QuestionAnswer:
    id: str
    answers: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class PinAdded:
    """loop.pin.added: an operator /pin instruction was recorded."""

    text: str


@dataclass(frozen=True, slots=True)
class PinsRestored:
    """loop.pin.restored: a resume/fork leg restored the snapshot's pins. The
    full list replaces the fold's pins (a plain resume's log already carries
    the pin.added events; a fork's fresh log carries only this)."""

    pins: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class CompactRestored:
    """loop.compact.restored: a resume/fork leg counted the elision markers its
    RESTORED context actually carries. The counts replace the fold's (a fork's
    fresh log has no compact.dropped events to fold, so it reported zero over a
    context full of markers)."""

    elided: int
    gists: int


@dataclass(frozen=True, slots=True)
class CompactDropped:
    """loop.compact.dropped: tier-1 elision, with the elided call identities."""

    n: int
    calls: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class CompactGists:
    """loop.compact.gists: gists created / demoted in a tier-1 pass."""

    gisted: int
    demoted: int


@dataclass(frozen=True, slots=True)
class CompactSummarised:
    """loop.compact.summarise.done: a tier-2 restart replaced the history (and
    with it every elision marker and gist the context held)."""


@dataclass(frozen=True, slots=True)
class SteerRequested:
    """session.steer_requested: an operator Ctrl-C mid-run."""


@dataclass(frozen=True, slots=True)
class SessionEnd:
    # The verify tri-state: True = final tree observed verify-green, False =
    # not green (red/stale/error), None = nothing gated it (no verify command).
    all_passed: bool | None
    reason: str


@dataclass(frozen=True, slots=True)
class SessionUndone:
    """session.undone: /undo forked this run; surfaces follow the child with
    the undone text back in the composer."""

    new_session_id: str
    undone_text: str


@dataclass(frozen=True, slots=True)
class RawEvent:
    """Any event the fold does not structurally consume (the ~65 loop.* telemetry
    types, unknown/future types, a line with no `type`). Carries the raw dict so the
    log-line renderer still reads it; the fold drops it (its old `case _`)."""

    type: str
    raw: dict[str, Any] = field(default_factory=dict)


Event = (
    SessionStart
    | ResumeStart
    | GraphUpdate
    | DiffUpdated
    | AutoCommit
    | RoleCall
    | RoleResult
    | RoleTextDelta
    | RoleThinkingDelta
    | ToolCall
    | ToolResult
    | VerifyStart
    | VerifyEnd
    | BudgetUpdate
    | ApprovalPrompt
    | ApprovalAnswer
    | QuestionPrompt
    | QuestionAnswer
    | PinAdded
    | PinsRestored
    | CompactRestored
    | CompactDropped
    | CompactGists
    | CompactSummarised
    | SteerRequested
    | SessionEnd
    | SessionUndone
    | RawEvent
)


def _call_id(raw: dict[str, Any]) -> int | None:
    cid = raw.get("call_id")
    return cid if isinstance(cid, int) else None


def parse_event(raw: dict[str, Any]) -> Event:
    """One raw logs.jsonl event dict -> one typed family, or RawEvent for the rest.

    A malformed field inside a KNOWN family (a torn numeric in `verify.end` or
    `budget.update`) degrades to RawEvent exactly like an unknown type: the
    fold runs unwrapped inside live tails (web SSE, TUI reader), so it must
    never raise on a line an interrupted writer left behind."""
    try:
        return _parse_known(raw)
    except (ValueError, TypeError, AttributeError, KeyError, IndexError):
        return RawEvent(type=str(raw.get("type", "")), raw=raw)


def _parse_known(raw: dict[str, Any]) -> Event:  # noqa: PLR0911, PLR0912
    """The per-family arms. Each reproduces, field-for-field, the coercion the
    SessionState fold applied inline before this module existed, so the fold output
    is byte-identical for every historical event."""
    match raw.get("type", ""):
        case "session.start":
            return SessionStart(user_task=str(raw.get("user_task", "")))
        case "loop.resume.start":
            return ResumeStart()
        case "graph.update":
            nodes = raw.get("nodes", {}) or {}
            if not isinstance(nodes, dict):
                # Degrade, don't coerce: an empty-dict fold would REPLACE the
                # task tree; RawEvent keeps the last good graph.
                raise ValueError("graph.update nodes must be an object")
            cursor = raw.get("cursor")
            return GraphUpdate(
                nodes=nodes,
                cursor=cursor if isinstance(cursor, str) else None,
            )
        case "diff.updated":
            return DiffUpdated(patch=str(raw.get("patch", "")), sha=str(raw.get("sha", "")))
        case "loop.auto_commit":
            return AutoCommit(
                iteration=_as_int(raw.get("iteration")),
                sha=str(raw.get("sha") or ""),
                subject=str(raw.get("subject") or ""),
            )
        case "role.call":
            return RoleCall(
                role=str(raw.get("role", "")),
                model=str(raw.get("model", "")),
                provider=str(raw.get("provider", "")),
            )
        case "role.result":
            return RoleResult(
                tokens_in=_as_int(raw.get("tokens_in")),
                cache_read=_as_int(raw.get("cache_read")),
                cache_creation=_as_int(raw.get("cache_creation")),
            )
        case "role.text_delta":
            return RoleTextDelta(text=str(raw.get("text", "")))
        case "role.thinking_delta":
            return RoleThinkingDelta(text=str(raw.get("text", "")))
        case "tool.call":
            args = raw.get("args")
            return ToolCall(
                name=str(raw.get("name", "")),
                # Coerce, don't degrade: the call happened even if its args
                # field is garbled, and args is display-only downstream.
                args=args if isinstance(args, dict) else {},
                call_id=_call_id(raw),
            )
        case "tool.result":
            return ToolResult(
                name=str(raw.get("name", "")),
                ok=tool_result_ok(raw.get("ok")),
                summary=readable_summary(raw.get("summary", "")),
                call_id=_call_id(raw),
            )
        case "verify.start":
            return VerifyStart(cmd=tuple(str(x) for x in raw.get("cmd", []) or []))
        case "verify.end":
            return VerifyEnd(
                cmd=tuple(str(x) for x in raw.get("cmd", []) or []),
                exit_code=int(raw.get("exit_code", -1)),
                duration_s=float(raw.get("duration_s", 0.0)),
                stdout_tail=str(raw.get("stdout_tail", "")),
                stderr_tail=str(raw.get("stderr_tail", "")),
            )
        case "budget.update":
            return BudgetUpdate(
                input_total=int(raw.get("input_total", 0)),
                output_total=int(raw.get("output_total", 0)),
                usd_total=float(raw.get("usd_total", 0.0)),
                usd_partial=bool(raw.get("usd_partial", False)),
                # Post-redesign keys; a historical log without them folds 0.
                usd_cap=float(raw.get("usd_cap", 0.0)),
                tokens_unmetered=int(raw.get("tokens_unmetered", 0)),
                tokens_fallback_cap=int(raw.get("tokens_fallback_cap", 0)),
                plan_used_percent=float(raw.get("plan_used_percent", 0.0)),
                plan_consumed=float(raw.get("plan_consumed", 0.0)),
                plan_cap=float(raw.get("plan_cap", 0.0)),
                plan_resets_at=float(raw.get("plan_resets_at", 0.0)),
            )
        case "approval.prompt":
            return ApprovalPrompt(
                id=str(raw.get("id", "")),
                prompt=str(raw.get("prompt", "")),
                standing=bool(raw.get("standing", True)),
                ts_ep=event_epoch(raw.get("ts")),
            )
        case "approval.answer":
            return ApprovalAnswer(
                id=str(raw.get("id", "")), approved=bool(raw.get("approved", False))
            )
        case "question.prompt":
            questions = tuple(
                EventQuestion(
                    question=str(q.get("question", "")),
                    options=tuple(str(o) for o in (q.get("options", ()) or ())),
                )
                for q in (raw.get("questions", ()) or ())
                if isinstance(q, dict)
            )
            return QuestionPrompt(
                id=str(raw.get("id", "")), questions=questions, ts_ep=event_epoch(raw.get("ts"))
            )
        case "question.answer":
            raw_ans = raw.get("answers", ()) or ()
            answers = tuple(str(a) for a in raw_ans) if isinstance(raw_ans, (list, tuple)) else ()
            return QuestionAnswer(id=str(raw.get("id", "")), answers=answers)
        case "loop.pin.added":
            return PinAdded(text=str(raw.get("text", "")))
        case "loop.pin.restored":
            raw_pins = raw.get("pins", ()) or ()
            pins = tuple(str(x) for x in raw_pins) if isinstance(raw_pins, (list, tuple)) else ()
            return PinsRestored(pins=pins)
        case "loop.compact.restored":
            return CompactRestored(
                elided=_as_int(raw.get("elided")), gists=_as_int(raw.get("gists"))
            )
        case "loop.compact.dropped":
            raw_calls = raw.get("calls", ()) or ()
            calls = tuple(str(c) for c in raw_calls) if isinstance(raw_calls, (list, tuple)) else ()
            return CompactDropped(n=_as_int(raw.get("n")), calls=calls)
        case "loop.compact.gists":
            return CompactGists(
                gisted=_as_int(raw.get("gisted")), demoted=_as_int(raw.get("demoted"))
            )
        case "loop.compact.summarise.done":
            return CompactSummarised()
        case "session.steer_requested":
            return SteerRequested()
        case "session.undone":
            return SessionUndone(
                new_session_id=str(raw.get("new_session_id", "") or ""),
                undone_text=str(raw.get("undone_text", "") or ""),
            )
        case "session.end":
            # An explicit null is the ungated tri-state; an absent key (a
            # pre-tri-state log) stays False, reading as it always did.
            raw_ap = raw.get("all_passed", False)
            return SessionEnd(
                all_passed=None if raw_ap is None else bool(raw_ap),
                reason=str(raw.get("reason", "") or ""),
            )
        case other:
            return RawEvent(type=str(other), raw=raw)
