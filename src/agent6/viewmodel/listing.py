# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Shared run-listing helpers, used by every front-end's hub/watch listing.

The last-activity time and the task snippet live only here; the CLI, TUI,
and web hub all read them, so the three listings cannot disagree.
"""

from __future__ import annotations

import contextlib
import json
import re
import time
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

from agent6.sessions.ipc import read_worker_pid, worker_is_alive
from agent6.sessions.layout import HUB_BUCKETS, LOGS_NAME, MANIFEST_NAME, bucket_dir
from agent6.sessions.manifest import CompareStamp, ManifestError, SessionManifest, read_manifest
from agent6.task_text import operator_task_text
from agent6.viewmodel.events import event_epoch
from agent6.viewmodel.format import (
    format_age,
    format_cost_cell,
    listing_status_label,
    status_level,
    winner_id,
)


def session_mtime(session_dir: Path) -> float:
    """Last-activity time of a session: the mtime of its `logs.jsonl` (when the
    run last appended an event), else its manifest (written once, when the
    session was created), else the dir.

    NOT the run-directory mtime: a viewer writes its `frontends/` claim into the
    dir on open, bumping the DIRECTORY mtime, so sorting by it floats a
    merely-viewed run to "most recent". A run with no log yet (parked, or a
    `fork --no-run`) has a manifest
    and nothing else that moves, so that is its time.
    """
    for candidate in (session_dir / LOGS_NAME, session_dir / MANIFEST_NAME, session_dir):
        try:
            return candidate.stat().st_mtime
        except OSError:
            continue
    return 0.0


def session_dirs(state_dir: Path, buckets: Iterable[str] = HUB_BUCKETS) -> list[Path]:
    """Every session dir a listing shows across *buckets* (names under
    `sessions/`, the hub's by default), newest first by last activity; husks
    skipped, like every listing skips them. The one enumeration behind the
    hubs' tables and a spawn's before/after locate set."""
    dirs: list[Path] = []
    for name in buckets:
        bucket = bucket_dir(state_dir, name)
        if bucket.is_dir():
            dirs.extend(p for p in bucket.iterdir() if p.is_dir() and not is_session_husk(p))
    dirs.sort(key=session_mtime, reverse=True)
    return dirs


def newest_session_dir(buckets: Iterable[Path]) -> Path | None:
    """The most recently active session dir (by logs.jsonl mtime, not dir mtime: a
    viewer's front-end claim must not float a run to latest) across the given
    bucket dirs.

    The one run-recency query: callers name the buckets in scope explicitly --
    a lone `runs/` dir for run/plan/resume/fork/ask scope, or every
    `SESSION_BUCKETS` dir for a cross-bucket listing (attach / runs stop). A
    missing bucket dir is skipped; returns None when no bucket holds a run.
    Callers that key off the id take `.name` of the result.

    Husks are skipped, like every listing skips them: a crash-orphaned dir with
    no manifest and no log is newer than the real runs, so returning it pointed
    bare `attach` / `sessions show` / `sessions stop` at a phantom the operator cannot
    see in any listing.
    """
    runs: list[Path] = []
    for bucket in buckets:
        if bucket.is_dir():
            runs.extend(p for p in bucket.iterdir() if p.is_dir() and not is_session_husk(p))
    dirs = sorted(runs, key=session_mtime, reverse=True)
    return dirs[0] if dirs else None


# An ask transcript's headers: the title (`# agent6 ask`, `# agent6 ask
# (interactive)`), then `## Question` / `## Answer` for a one-shot ask or
# `## Q1` / `## A1` (numbered) for an interactive one.
_ASK_QUESTION_HEADER = re.compile(r"^## (Question|Q\d+)$")
_ASK_ANSWER_HEADER = re.compile(r"^## (Answer|A\d+)$")


def first_task_line(lines: Iterable[str]) -> str | None:
    """First user-authored line: the ask headers (one-shot and interactive)
    skipped, the composed context (`operator_task_text`: seed digests, file
    seeds, skill blocks) stripped. Returns None when nothing stands out."""
    for line in operator_task_text("\n".join(lines)).splitlines():
        s = line.strip()
        if s.startswith("# agent6 ask") or _ASK_QUESTION_HEADER.match(s):
            continue
        if _ASK_ANSWER_HEADER.match(s):
            break
        if s and not s.startswith("<"):
            return s
    return None


def task_snippet(text: str, max_chars: int | None = None) -> str:
    """One-line summary of a task or ask transcript for a listing: the first
    user-authored line (block bodies skipped), else the stripped text; clipped
    to *max_chars* with an ellipsis (the bare slices each surface carried
    clipped mid-word and read as the whole task)."""
    snip = first_task_line(text.splitlines()) or text.strip()
    if max_chars is not None and len(snip) > max_chars:
        snip = snip[: max_chars - 1] + "…"
    return snip


def is_session_husk(session_dir: Path) -> bool:
    """True for a session dir that never really started: neither manifest.json nor
    logs.jsonl (a preflight refused it, or a crash orphaned it). Listings skip
    husks -- "(no logs)" forever is noise, not a run -- and id lookups must not
    let one shadow a real run of the same id in another bucket (runs/ vs asks/).

    Exception: a dir with a LIVE worker.pid is a just-launched run in its
    pre-manifest preflight window, not a husk -- keep it listed (it reads
    "starting"). Only a dir with no live worker is a true husk."""
    if (session_dir / "manifest.json").exists() or (session_dir / LOGS_NAME).exists():
        return False
    return not worker_is_alive(session_dir)


def session_compare(session_dir: Path) -> CompareStamp | None:
    """The `compare` stamp a fan-out's auto-compare recorded on an imported
    lane's manifest (rank/of/winner/ranked_by/rationale), or None for a run that
    was never part of a compared fan-out. The event fold doesn't carry it (it is
    post-import manifest state), so every run view reads it from here. Best effort:
    a missing/corrupt manifest reads as None, never an error."""
    try:
        manifest = read_manifest(session_dir)
    except ManifestError:
        return None
    return manifest.compare


def is_winner(session_dir: Path) -> bool:
    """True when a run is the fan-out compare winner (rank 1), for a listing
    marker. False for any run outside a compared fan-out."""
    compare = session_compare(session_dir)
    return compare is not None and compare.winner


@dataclass(frozen=True, slots=True)
class SessionSummary:
    """One listing row: everything a hub or `sessions list` needs, uncolored."""

    session_id: str
    mode: str  # run | plan | ask | ?
    task: str  # raw task text; callers snippet/truncate for their layout
    # created|starting|running|waiting|stale|passed|answered|planned|finished|
    # stopped|undone|failed
    status: str
    reason: str  # detail: the end reason when "failed", "needs answer" when "waiting", else ""
    cost_usd: float
    usd_partial: bool  # sticky: cost_usd is a lower bound (unpriced spend in some leg)
    mtime: float
    # The run branch holds commits its base does not (per the merge stamp and
    # the caller's branch-tips snapshot); False when unknowable (no snapshot),
    # and for an undone run: its /undo child carries the mark for the commits
    # up to the checkpoint, and the later ones were taken back.
    unmerged: bool = False
    # The gate verdict from the gate facts (LogScan.verify_verdict), NOT the
    # status word: the compare table and the judge read it, and the word calls
    # a red-gated finish "finished".
    verify_ok: bool | None = None


def summary_row(
    s: SessionSummary, *, winner: bool = False, task_chars: int | None = None
) -> dict[str, object]:
    """One listing row as JSON: the shape `sessions list --json` prints and
    `/api/hub` serves, so one name per fact reaches every reader.

    `label` and `level` are the rendered status cell and its colour level, so a
    client needs no copy of the status maps. *task_chars* asks for a one-line
    snippet clipped to that width, for a card with a row to fill; without it
    the task rides whole, since a JSON reader has its own layout and a
    multi-line task otherwise arrives as its first line with nothing to say so.
    """
    return {
        "session_id": s.session_id,
        "mode": s.mode,
        "task": s.task if task_chars is None else task_snippet(s.task, max_chars=task_chars),
        "status": s.status,
        "reason": s.reason,
        "label": listing_status_label(s.mode, s.status, s.reason, unmerged=s.unmerged),
        "level": status_level(s.status),
        "mtime": s.mtime,
        "cost_usd": s.cost_usd,
        "usd_partial": s.usd_partial,
        "cost": format_cost_cell(s.cost_usd, partial=s.usd_partial),
        "id_cell": winner_id(s.session_id, winner=winner),
        "unmerged": s.unmerged,
        "verify_ok": s.verify_ok,
        "winner": winner,
    }


def status_word(
    *, finished: bool, all_passed: bool | None, end_reason: str, scoped: bool = False
) -> tuple[str, str]:
    """Map an end state to `(word, reason-detail)`.

    The single place that decides how a run's outcome reads -- shared by
    `session_state_as_dict` (headers) and `summarize_session_dir` (listings) so the
    surfaces can never disagree. "stopped" and "undone" are the operator's
    own acts (a stop, an /undo), not failures; "planned" and "answered" are
    the no-verify clean exits (a plan pass / an ask, where "passed" would
    mislead); "passed" means all verify gates green, "finished" is a
    deliberate finish without all-passed, and anything else is "failed" with
    the reason (provider_error, went_quiet, ...).

    `all_passed` is the wire's verify tri-state: True = the final tree was
    observed verify-green, False = it was not (red, stale, or an error end),
    None = NOTHING gated it (no verify command). None reads "finished"
    whatever the reason: an ungated end never claims "passed" and never
    reads "failed". `scoped` qualifies a pass: the gate that certified the
    tree ran scoped to the tests nearest the run's diff, so it reads
    "passed · scoped gate", never a bare "passed".
    """
    if not finished:
        return "running", ""
    if end_reason in ("steer_abort", "steer_exit", "interrupted", "interactive_stop"):
        return "stopped", ""  # each is the operator's own act, not a failure
    # A clean exit that verified nothing gets its own word, never "passed": a
    # plan pass ends via finish_planning, an ask by answering, /undo takes the
    # last message back (the fork it cut continues), and a gateless run
    # settles with committed work no verify ever gated (deliberate, so
    # "finished"; never green, never "failed").
    no_verify = {
        "finish_planning": ("planned", ""),
        "answered": ("answered", ""),
        "undone": ("undone", ""),
        "settled": ("finished", "unverified"),
        # The gate is red, and a verify against an UNMODIFIED tree proved it
        # was red before this run touched anything. "Your run failed" and "your
        # change broke nothing new" are different facts.
        "gate_red_at_base": ("finished", "gate was already red"),
    }
    if end_reason in no_verify:
        return no_verify[end_reason]
    if all_passed:
        return "passed", "scoped gate" if scoped else ""
    # Only an OBSERVED not-green (False) can word "failed"; the ungated None
    # falls through to "finished" whatever the reason.
    if all_passed is False and end_reason and end_reason != "finish_session":
        return "failed", end_reason
    return "finished", ""


# The two prompt events that mean "alive but blocked on the OPERATOR". One
# definition: the hub listing and `sessions show` both key their "waiting (needs
# answer)" status on it, so the two surfaces can't disagree.
OPERATOR_PROMPT_EVENTS = frozenset({"approval.prompt", "question.prompt"})
OPERATOR_ANSWER_EVENTS = frozenset({"approval.answer", "question.answer"})


# The word a parked submission reads as; the detail beside it is the
# manifest's short cause ("checkout busy", "uncommitted changes"), and every
# surface says the same thing about it: a resume starts it.
PARKED_WORD = "parked"


@dataclass(frozen=True, slots=True)
class StatusFacts:
    """The event-derived inputs to :func:`status_for_session_dir`, producible from
    either event reader -- `LogScan.status_facts()` (the tolerant scanner
    behind listings and `sessions show`) and `state.status_facts` (the typed
    fold behind the live views) -- so every surface feeds the one status
    decision the same answers for the same log."""

    started: bool = False  # a session.start was seen (a parked/created run has none)
    finished: bool = False
    all_passed: bool | None = False  # None = the end was ungated (no verify command)
    verify_scoped: bool = False  # the judging gate ran scoped (qualifies a pass)
    end_reason: str = ""
    operator_blocked: bool = False  # alive but waiting on an unanswered approval/question
    blocked_kind: str = ""  # "approval" | "question" | "" (oldest unanswered prompt)
    blocked_since_ep: float | None = None  # that prompt's asked-at epoch


def status_for_session_dir(session_dir: Path, facts: StatusFacts) -> tuple[str, str]:
    """THE `(word, reason)` for a session that has a dir on disk.

    Every listing and header feeds this the event facts and lets the DIR
    supply what events cannot: a parked submission (manifest) and worker
    liveness (worker.pid). The pure fold's `session_state_as_dict`
    is only for a stream with genuinely no dir (`attach --json`); a surface
    with a run dir that folds events alone reads every non-`session.end` state
    as "running" and disagrees with the hub.

    A started session is live iff its worker is: the pid is written before
    session.start, so no pid file means the worker cleared it on the way out.
    Log silence cannot stand in for this; it inverts the evidence. A `kill -9`
    LEAVES the pid file (silence would read "stale" at once) while an abnormal
    exit through the finally (SIGPIPE from `run ... | head`) clears it
    (silence would read "running" for the whole 600s window).
    """
    if facts.finished:
        return status_word(
            finished=True,
            all_passed=facts.all_passed,
            end_reason=facts.end_reason,
            scoped=facts.verify_scoped,
        )
    if facts.operator_blocked and worker_is_alive(session_dir):
        # Before session.start too: a run asks about the working tree's
        # uncommitted changes before it starts. The detail names WHAT it
        # waits on and for how long ("approval 12m"); a log whose prompt
        # carried no parseable ts keeps the generic wording.
        if facts.blocked_kind and facts.blocked_since_ep is not None:
            age = format_age(time.time() - facts.blocked_since_ep)
            return "waiting", f"{facts.blocked_kind} {age}"
        return "waiting", "needs answer"
    if not facts.started:
        return _unstarted_status(session_dir)
    if not worker_is_alive(session_dir):
        return "stale", ""
    return "running", ""


def _unstarted_status(session_dir: Path) -> tuple[str, str]:
    """Status before any session.start: a live worker is still launching (egress +
    verify inference run before the loop's first turn) -> "starting". No live
    worker is a parked submission (saved with its cause; resume starts it), a
    worker that died launching (its pid file survives the kill; a clean
    refusal clears it), or a never-started dir (`fork --no-run`) -> "created".
    The dead-pid case is kept distinct from "created": a killed preflight
    spent real dollars and must not wear the never-ran word."""
    if worker_is_alive(session_dir):
        return "starting", ""
    with contextlib.suppress(ManifestError):
        manifest = read_manifest(session_dir)
        if manifest.parked_task:
            return PARKED_WORD, manifest.parked_reason
    if read_worker_pid(session_dir) is not None:
        return "stale", "died launching"
    return "created", ""


# Status words for a run that reached terminal WITHOUT its own session.end: the
# worker died (stale) or never started (created/parked/?). The fan-out's
# awaiting gate deliberately accepts them so an await cannot hang; the web live
# view closes their stream, and `sessions compare` screens them out (no verdict
# to compare, spend truncated at the death).
_DIED_WITHOUT_END = frozenset({"stale", "created", "parked", "?"})


def died_without_end(status: str) -> bool:
    """Whether *status* is a session that never reached its own `session.end`."""
    return status in _DIED_WITHOUT_END


# Status words for a session that ended deliberately: its own clean session.end
# (passed/finished/planned/answered) or the operator's stop. Only these are
# results a fan-out may rank, crown, or join; "failed" (an abnormal end:
# provider_error, went_quiet, ...), a died-without-end word, or a live word is
# work with no verdict. A positive set: an unknown new status word is not a
# result until it earns membership.
_RESULT_WORDS = frozenset({"passed", "finished", "stopped", "planned", "answered"})


def produced_result(status: str) -> bool:
    """Whether the session ended deliberately and left mergeable work: THE
    lane-candidacy question -- only such a lane is a fan-out compare candidate
    or joins a coordinator's branch."""
    return status in _RESULT_WORDS


# Status words for a run that can still receive operator input over the file
# bridge. Anything else (parked/created: never started, stale: worker gone, and
# every end word) means a surface must offer resume instead -- a steer or answer
# marker there is read by nobody.
LIVE_STATUS_WORDS = frozenset({"running", "starting", "waiting"})


def session_is_live(session_dir: Path) -> bool:
    """Whether the operator can still act on this session: THE affordance question,
    "will anything read what I write", not `worker_is_alive`'s "is a pid
    running" (a parked run resumes; a dead worker's buttons reach nobody).

    Derived from the status word, so a surface cannot disagree with the label
    it is showing; fed empty facts it degenerates to the pid probe.
    """
    logs = session_dir / LOGS_NAME
    facts = scan_session_log(logs).status_facts() if logs.is_file() else StatusFacts()
    return status_for_session_dir(session_dir, facts)[0] in LIVE_STATUS_WORDS


@dataclass(frozen=True, slots=True)
class LogScan:
    """One tolerant pass over a session's `logs.jsonl`: the shared scan behind the
    hub listing and `sessions show`. One owner, because the resume rules (bank
    cost legs, un-finish) and the torn-line tolerances drifted when each
    consumer scanned for itself.

    Token counters are the CURRENT leg's; `cost_usd` is cumulative across
    resume legs (None = no budget.update ever), matching the typed fold's
    BudgetView so no two surfaces can disagree on what a run cost. `legs`
    lets a renderer say which scope a figure describes when they differ.
    """

    saw_start: bool = (
        False  # session.start OR loop.resume.start seen (a leg began); neither = unstarted
    )
    mode: str = "?"
    task: str = ""
    finished: bool = False  # a later resume un-finishes
    all_passed: bool | None = False  # None = the end was ungated (no verify command)
    verify_scoped: bool = False  # session.end scoped: the judging gate ran scoped
    end_reason: str = ""
    cost_usd: float | None = None
    usd_partial: bool = False  # sticky: unpriced spend in any leg -> under-estimate
    legs: int = 1  # 1 + completed resume legs
    input_tokens: int | None = None
    output_tokens: int | None = None
    iteration: int | None = None  # last event carrying an int iteration
    # session.start's ts (epoch seconds), else the first event's: a fork's log
    # opens with loop.resume.start and never carries a session.start.
    start_ep: float | None = None
    last_ep: float | None = None  # last event with a parseable ts
    last_type: str | None = None  # last event's type
    operator_blocked: bool = False  # a prompt is still unanswered on this leg
    blocked_kind: str = ""  # oldest unanswered prompt's kind ("approval"/"question")
    blocked_since_ep: float | None = None  # its asked-at epoch
    last_verify_rc: int | None = None  # this leg's last verify.end exit code
    pins: tuple[str, ...] = ()  # the operator's pinned instructions in force

    def verify_verdict(self) -> bool | None:
        """The gate verdict from the gate FACTS, for judging candidates: True =
        the run ended all-passed (the gate vouched for the final tree), False =
        this leg's last verify ran and failed, None = nothing observed the
        final tree (gateless, no verify this leg, or a green made stale by
        later edits). Deriving this from the folded status word called a RED
        gate "no verify": finish_session over red folds to "finished"."""
        if self.mode != "run":
            return None
        if self.finished and self.all_passed:
            return True
        if self.last_verify_rc is not None and self.last_verify_rc != 0:
            return False
        return None

    def status_facts(self) -> StatusFacts:
        """This scan's answers to the status questions, for status_for_session_dir.
        The typed fold's `state.status_facts` must agree on the same log
        (pinned by the status matrix test)."""
        return StatusFacts(
            started=self.saw_start,
            finished=self.finished,
            all_passed=self.all_passed,
            verify_scoped=self.verify_scoped,
            end_reason=self.end_reason,
            operator_blocked=self.operator_blocked,
            blocked_kind=self.blocked_kind,
            blocked_since_ep=self.blocked_since_ep,
        )


def _tolerant_usd(raw: object, last_good: float) -> float:
    """*raw* as a float when it is a real number or numeric string; else the
    last good figure. A torn/adversarial usd_total degrades like a torn line,
    never aborts the scan (the typed fold makes the same call in parse_event),
    and falsy junk (`""`, `False`) must KEEP the figure; an `or 0.0`
    fallback silently reset it."""
    if isinstance(raw, (int, float)) and not isinstance(raw, bool):
        return float(raw)
    if isinstance(raw, str):
        with contextlib.suppress(ValueError):
            return float(raw)
    return last_good


def finished_needs_new_work(session_dir: Path) -> bool:
    """Whether resuming this run would have nothing to do.

    True only when the agent ENDED it by calling `finish_session`: the resumed
    leg spends a call, answers in prose with no tool use, records a
    silent_finish, and leaves a run that passed reading as failed for a tree
    nobody touched. Every other ending -- budget_exhausted, provider_error,
    steer_abort, a red verify -- is exactly what resume is for. Read through
    the same fold the listing uses, so a refusal and the status it contradicts
    cannot disagree.
    """
    scan = scan_session_log(session_dir / LOGS_NAME)
    return scan.finished and scan.end_reason == "finish_session"


def needs_new_work_refusal(session_id: str) -> str:
    """The one wording for it, so `agent6 resume` and the web composer refuse a
    finished run in the same words."""
    return (
        f"run {session_id!r} already finished (the agent called finish_session)."
        " Give it new work with:\n"
        f'    agent6 resume {session_id} --steer "<what to do next>"'
    )


def scan_session_log(logs: Path) -> LogScan:  # noqa: PLR0912, PLR0915 (linear fold, like build_parser)
    """Fold `logs.jsonl` into a :class:`LogScan`: session.start (mode/task), the
    last session.end (un-finished again by a later resume), the running per-leg
    budget banked across resumes into a cumulative total, and the liveness
    anchors (timestamps, iteration, last event type) `sessions show` reads.

    errors="replace": a live writer can leave a torn multibyte UTF-8 tail; strict
    decoding would take down the whole listing. The mangled line just fails
    json.loads and is skipped."""
    mode, task = "?", ""
    finished, end_reason = False, ""
    all_passed: bool | None = False
    verify_scoped = False
    saw_start = False
    usd_leg = 0.0  # latest leg's running total
    usd_prior_legs = 0.0  # summed totals of completed (resumed-past) legs
    saw_budget = False
    usd_partial = False
    legs = 1
    input_tokens: int | None = None
    output_tokens: int | None = None
    iteration: int | None = None
    start_ep: float | None = None
    first_ep: float | None = None
    last_ep: float | None = None
    last_type: str | None = None
    # Prompt ids still awaiting their answer. A later event must not clear the
    # bit: Ctrl-C emits session.steer_requested while an approval still waits.
    pending_prompts: dict[str, tuple[str, float | None]] = {}
    last_verify_rc: int | None = None
    pins: list[str] = []
    try:
        with logs.open(encoding="utf-8", errors="replace") as fh:
            for line in fh:
                try:
                    ev = json.loads(line)
                except ValueError:
                    continue
                if not isinstance(ev, dict):
                    continue  # a valid-JSON non-object line (torn/adversarial)
                etype = ev.get("type")
                ep = event_epoch(ev.get("ts"))
                if ep is not None:
                    last_ep = ep
                    if first_ep is None:
                        first_ep = ep
                if isinstance(etype, str):
                    last_type = etype
                if isinstance(ev.get("iteration"), int):
                    iteration = ev["iteration"]
                if etype in OPERATOR_PROMPT_EVENTS:
                    # Coerce like events.py: the answer side discards str(id), so a
                    # non-string id (an int) must be stored as str to match and
                    # clear -- else the run stays "waiting" forever.
                    if (pid := ev.get("id")) is not None:
                        kind = "approval" if etype == "approval.prompt" else "question"
                        pending_prompts[str(pid)] = (kind, ep)
                elif etype in OPERATOR_ANSWER_EVENTS:
                    pending_prompts.pop(str(ev.get("id")), None)
                if etype == "session.start":
                    saw_start = True
                    finished = False  # a leg is starting (ask REPL re-runs in place)
                    mode = str(ev.get("mode", mode))
                    task = str(ev.get("user_task", ""))
                    # A leg boundary invalidates unanswered prompts: the new leg
                    # re-asks with restarted ids, so a held-over entry would keep
                    # the run "waiting" forever (the typed fold's rule too).
                    pending_prompts.clear()
                    last_verify_rc = None
                    if start_ep is None:
                        start_ep = ep
                elif etype == "session.end":
                    finished = True
                    # An explicit null is the ungated tri-state; an ABSENT key
                    # stays False.
                    raw_ap = ev.get("all_passed", False)
                    all_passed = None if raw_ap is None else bool(raw_ap)
                    verify_scoped = bool(ev.get("scoped", False))
                    end_reason = str(ev.get("reason", ""))
                elif etype == "loop.resume.start":
                    if saw_start:
                        # A PRIOR leg exists: bank its budget and count a new
                        # leg. Each resume leg starts a FRESH budget (usd_total
                        # resets to 0), so bank the finished leg's total before
                        # it does -- the displayed cost is then the true
                        # cumulative spend across all legs (per-leg budgets stay
                        # the enforcement mechanism). The typed fold applies the
                        # same rule (state.BudgetView), so the hub row and the
                        # run view can never disagree. Token counters reset too:
                        # they are documented as the current leg's. A fork's log
                        # opens with this event, which begins leg 1.
                        usd_prior_legs += usd_leg
                        usd_leg = 0.0
                        input_tokens = output_tokens = None
                        last_verify_rc = None  # leg-scoped, like the token counters
                        legs += 1
                    saw_start = True  # a leg has begun; a fork's log has only this
                    finished = False  # a resume un-finishes the run
                    pending_prompts.clear()  # see session.start
                elif etype == "verify.end":
                    rc = ev.get("exit_code")
                    if isinstance(rc, int) and not isinstance(rc, bool):
                        last_verify_rc = rc
                elif etype == "loop.pin.added":
                    pins.append(str(ev.get("text", "")))
                elif etype == "loop.pin.restored":
                    # The full list in force at leg start (the typed fold's rule).
                    raw_pins = ev.get("pins")
                    pins = [str(p) for p in raw_pins] if isinstance(raw_pins, list) else []
                elif etype == "budget.update":
                    saw_budget = True
                    usd_leg = _tolerant_usd(ev.get("usd_total"), usd_leg)
                    usd_partial = bool(ev.get("usd_partial")) or usd_partial
                    ti, to = ev.get("input_total"), ev.get("output_total")
                    input_tokens = ti if isinstance(ti, int) else input_tokens
                    output_tokens = to if isinstance(to, int) else output_tokens
    except OSError:
        pass
    return LogScan(
        saw_start=saw_start,
        mode=mode,
        task=task,
        finished=finished,
        all_passed=all_passed,
        verify_scoped=verify_scoped,
        end_reason=end_reason,
        cost_usd=(usd_prior_legs + usd_leg) if saw_budget else None,
        usd_partial=usd_partial,
        legs=legs,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        iteration=iteration,
        start_ep=start_ep if start_ep is not None else first_ep,
        last_ep=last_ep,
        last_type=last_type,
        operator_blocked=bool(pending_prompts),
        blocked_kind=(
            oldest[0]
            if (
                oldest := min(
                    pending_prompts.values(),
                    key=lambda kv: kv[1] if kv[1] is not None else float("inf"),
                    default=None,
                )
            )
            else ""
        ),
        blocked_since_ep=oldest[1] if oldest else None,
        last_verify_rc=last_verify_rc,
        pins=tuple(pins),
    )


def summarize_session_dir(
    session_dir: Path, *, branch_tips: Mapping[str, str] | None = None
) -> SessionSummary:
    """One listing row from `logs.jsonl` + the manifest. Replaced the
    near-duplicate scanners in the TUI hub and the web hub that badged a
    provider_error death as a neutral "done". The manifest owns the task (the
    event clips it to 200 chars); an "ask" run's task is replaced by its
    transcript, which shows what was asked.

    *branch_tips* is the caller's one-call `git_ops.run_branch_tips` snapshot;
    with it the row says whether the run branch still holds unmerged commits
    (the tip is not the base and not the merge stamp's tip). Without it
    `unmerged` stays False: no mark, never a wrong one."""
    logs = session_dir / LOGS_NAME
    scan = scan_session_log(logs) if logs.is_file() else LogScan()
    manifest: SessionManifest | None = None
    with contextlib.suppress(ManifestError):
        manifest = read_manifest(session_dir)
    mode, task = scan.mode, scan.task
    if manifest is not None:
        # The mode falls back to the manifest's for a log with no session.start:
        # a launching run still in preflight (verify inference is a ~80s LLM
        # call BEFORE the loop's first turn), a manifest-only `fork --no-run`,
        # or a forked/resumed leg whose log opens with loop.resume.start (which
        # begins a leg but records no mode).
        task = manifest.user_task or task
        if mode == "?":
            mode = manifest.mode or mode
    if mode == "?" and not task:
        task = "(no logs)"  # a husk: no manifest, and a log naming nothing
    word, reason = status_for_session_dir(session_dir, scan.status_facts())
    if mode == "ask":
        # An ask has no task, so its row shows what was asked: the transcript's
        # own first line, never the whole file. A JSON reader takes this field
        # whole, and a long answer there is a transcript, not a task.
        with contextlib.suppress(OSError):
            transcript = (session_dir / "transcript.md").read_text(
                encoding="utf-8", errors="replace"
            )
            # The question is the first line under the transcript's first `##`
            # heading (`## Question`, or `## Q1` from the REPL form); the first
            # non-heading line would be the ANSWER whenever the question begins
            # with `#`.
            lines = transcript.splitlines()
            heading = next((i for i, ln in enumerate(lines) if ln.startswith("## ")), None)
            body = lines[heading + 1 :] if heading is not None else []
            asked = next((ln.strip() for ln in body if ln.strip()), "")
            task = asked[:200] or transcript.strip()[:200]
    unmerged = False
    if branch_tips is not None and manifest is not None and word != "undone":
        tip = branch_tips.get(manifest.run_branch or "")
        unmerged = (
            tip is not None
            and tip != manifest.base_sha
            and (manifest.merged is None or manifest.merged.tip != tip)
        )
    return SessionSummary(
        session_id=session_dir.name,
        mode=mode,
        task=task,
        status=word,
        reason=reason,
        unmerged=unmerged,
        cost_usd=scan.cost_usd or 0.0,
        usd_partial=scan.usd_partial,
        mtime=session_mtime(session_dir),
        verify_ok=scan.verify_verdict(),
    )
