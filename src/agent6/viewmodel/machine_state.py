# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Pure fold of a machine's journal into a render-ready watch view.

The machine analogue of state.py: where SessionState folds a run's logs.jsonl, this
folds a machine instance's journal (the StepEvent / MachineEnd stream) plus its
spec into a MachineState that the CLI `agent6 attach`, the TUI
MachineWatchScreen, and the web client all render. The agent reasoning
inside an `agent` state is itself a run log, so it folds through SessionState
(state.py); this module models only the machine level: which states exist, where
we are, the path taken, and how it ended.

Position is exposed semantically (is_current / is_visited), not as a marker
glyph, so each front-end picks its own (the CLI uses ".", the TUI "·", a web
client a CSS class) without the model dictating presentation.
"""

from __future__ import annotations

import contextlib
import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

from agent6.machine import MachineError, MachineResult, load_machine
from agent6.machine.journal import (
    AgentFact,
    AttemptSpend,
    JournalError,
    MachineEnd,
    MachineJournal,
    MachineNotify,
    StepEvent,
    ToolFact,
)
from agent6.machine.model import MachineSpec
from agent6.sessions.ipc import worker_is_alive
from agent6.sessions.layout import LOGS_NAME, machines_root
from agent6.viewmodel.format import format_transition, machine_state_mark
from agent6.viewmodel.state import fold_session
from agent6.viewmodel.tail import tail_events

# How many recent machine.notify events a MachineState carries. Front-ends render
# them as ephemeral surfaces, so only the tail matters; the journal keeps them all.
_NOTIFY_KEEP = 20


@dataclass(frozen=True, slots=True)
class MachineStateView:
    """One state in the overview: its name, kind, and where we are relative to it."""

    name: str
    kind: str
    is_current: bool
    is_visited: bool
    mark: str = ""  # the mark before the name, as every surface draws it

    def __post_init__(self) -> None:
        if not self.mark:
            mark = machine_state_mark(is_current=self.is_current, is_visited=self.is_visited)
            object.__setattr__(self, "mark", mark)


@dataclass(frozen=True, slots=True)
class TransitionView:
    """One journaled transition: state --label--> goto, in order.

    `detail` is the failure evidence a debugging operator needs at the
    surface, bounded: a failed tool's exit code and last stderr/stdout line,
    a failed agent state's stop reason. Empty on success -- the happy path
    stays one line."""

    seq: int
    state: str
    label: str
    goto: str
    detail: str = ""
    line: str = ""  # `format_transition` of the fields, as every surface prints it


@dataclass(frozen=True, slots=True)
class NotificationView:
    """One journaled `machine.notify` (a state's `notify` message), in order."""

    ts: str
    state: str
    message: str
    level: str


@dataclass(frozen=True, slots=True)
class MachineState:
    machine: str
    version: int
    initial: str
    current: str  # where the machine is, or is about to run
    states: tuple[MachineStateView, ...]  # spec order, position-flagged
    transitions: tuple[TransitionView, ...]  # the path taken, in order
    ended: MachineResult | None
    notifications: tuple[NotificationView, ...]  # recent machine.notify, oldest first

    @property
    def current_kind(self) -> str | None:
        """The current state's kind ("wait" parks the machine), None at the end."""
        return next((s.kind for s in self.states if s.is_current), None)


def _transition_view(s: StepEvent) -> TransitionView:
    detail = _fact_detail(s)
    line = format_transition(s.seq, s.state, s.label, s.goto, detail)
    return TransitionView(
        seq=s.seq, state=s.state, label=s.label, goto=s.goto, detail=detail, line=line
    )


def _fact_detail(step: StepEvent) -> str:
    """Bounded failure evidence for one transition; "" on success."""
    fact = step.fact
    if isinstance(fact, ToolFact) and (fact.exit_code != 0 or fact.timed_out):
        tail = next(
            (
                ln.strip()
                for ln in reversed((fact.stderr or fact.stdout).splitlines())
                if ln.strip()
            ),
            "",
        )
        head = "timed out" if fact.timed_out else f"exit {fact.exit_code}"
        return f"{head}: {tail[:160]}" if tail else head
    if isinstance(fact, AgentFact) and fact.outcome != "ok":
        return f"{fact.outcome}: {fact.reason}"[:160]
    return ""


def fold_machine(spec: MachineSpec, events: Sequence[object]) -> MachineState:
    """Reduce a machine journal (StepEvent/MachineEnd stream) to a watch view.

    current = the goto of the last transition (where the machine is, or is about
    to run), else the initial state. visited = every state entered or left.
    """
    steps = [e for e in events if isinstance(e, StepEvent)]
    end = next((e for e in reversed(events) if isinstance(e, MachineEnd)), None)
    current = steps[-1].goto if steps else spec.initial
    visited: set[str] = set()
    for s in steps:
        visited.update((s.state, s.goto))
    states = tuple(
        MachineStateView(
            name=name,
            kind=st.kind,
            is_current=(name == current),
            is_visited=(name in visited),
        )
        for name, st in spec.states.items()
    )
    transitions = tuple(_transition_view(s) for s in steps)
    ended = MachineResult.from_end(end) if end is not None else None
    notes = [e for e in events if isinstance(e, MachineNotify)]
    notifications = tuple(
        NotificationView(ts=n.ts, state=n.state, message=n.message, level=n.level)
        for n in notes[-_NOTIFY_KEEP:]
    )
    return MachineState(
        machine=spec.machine,
        version=spec.version,
        initial=spec.initial,
        current=current,
        states=states,
        transitions=transitions,
        ended=ended,
        notifications=notifications,
    )


def machine_status_word(
    ms: MachineState, *, parked: bool, alive: bool, blocked: bool = False
) -> str:
    """THE liveness word front-ends read, so a machine that isn't working never
    renders busy. Terminal reports its ok/failed end; an armed `--exit-on-wait`
    wait (parked), a live worker blocked in a foreground `wait` state, or a live
    worker whose agent state holds an unanswered operator prompt (blocked) is
    "waiting"; a live worker in any other state is "running"; a dead pid that is
    neither parked nor ended is "stopped". The fold is pure, so parked (a
    persisted PendingWait), alive (a live worker.pid) and blocked (the newest
    state log's open prompt) are probed by the caller."""
    if ms.ended is not None:
        return ms.ended.status
    if parked:
        return "waiting"
    if alive:
        if blocked:
            return "waiting"
        return "waiting" if ms.current_kind == "wait" else "running"
    return "stopped"


def machine_operator_blocked(machine_dir: Path) -> str:
    """The state dir (`0001-attempt`) whose agent leg holds an unanswered
    approval or question, else "": the machine waits on the operator there."""
    log = newest_state_log(machine_dir)
    if log is None:
        return ""
    state = fold_session(tail_events(log, follow=False))
    open_prompts = [*state.pending_approvals, *state.pending_questions]
    return log.parent.name if any(not p.answered for p in open_prompts) else ""


def machine_is_parked(machine_dir: Path) -> bool:
    """True when the instance is parked in an armed wait (a PendingWait is
    persisted). Under --exit-on-wait scheduling a parked machine legitimately
    has no live process, so liveness probes must not read "dead pid" as
    "crashed" while this holds. A corrupt wait file counts as parked: better
    to keep streaming than to close on a guess."""
    try:
        return MachineJournal(machine_dir).read_pending_wait() is not None
    except JournalError:
        return True


@dataclass(frozen=True, slots=True)
class MachineSummary:
    """One machine-instance row: what a hub or `machine list` shows, uncolored."""

    name: str  # the instance dir name (the machine's name)
    machine: str  # the spec's declared name; "" when unreadable
    current: str  # where the machine is; "" when unreadable
    status: str  # machine_word_for_dir's word, or "unreadable"
    reason: str  # a failed end's reason, or what a live instance waits on; else ""
    mtime: float


@dataclass(frozen=True, slots=True)
class Spend:
    """A dollar + token spend triple, summable so booked and live spend fold.

    `partial` marks a known under-estimate (an unpriced model contributed
    $0 to the dollar figure); it ORs across folds so one unpriceable slice
    taints the total, and the render adds the shared '~' marker instead of
    showing a lower bound as exact."""

    usd: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    partial: bool = False

    def __add__(self, other: Spend) -> Spend:
        return Spend(
            self.usd + other.usd,
            self.input_tokens + other.input_tokens,
            self.output_tokens + other.output_tokens,
            self.partial or other.partial,
        )


def read_budget_totals(log_path: Path, *, from_offset: int = 0) -> Spend:
    """The latest running budget totals from an agent state's per-state event log,
    or `Spend()` if there is none / the log is unreadable.

    Each turn's `budget.update` event carries cumulative totals FROM THAT
    CALL'S OWN BudgetTracker, so the last one is the running total -- of
    whichever call wrote it. `from_offset` scopes the read to events appended
    after a byte offset: a caller salvaging one call on a SHARED log (machine
    create's draft log spans every attempt) must pass the log size captured
    before its spawn, or a call that died before its first budget.update reads
    the PRIOR call's totals and double-books them. Recovers spend for a
    timed-out/killed subprocess whose `result.json` never landed, and reads
    the LIVE total of an in-flight state whose `StepEvent` is not written yet
    (an agent state's spend would otherwise book as
    $0, so a 24/7 machine burns real money against a $0 ledger and its budget
    guard never trips)."""
    usd, tin, tout = 0.0, 0, 0
    partial = False
    with contextlib.suppress(OSError):
        with log_path.open("rb") as fh:
            if from_offset > 0:
                fh.seek(from_offset)
            body = fh.read().decode("utf-8", errors="replace")
        for line in body.splitlines():
            try:
                e = json.loads(line)
            except ValueError:
                continue
            if e.get("type") == "budget.update":
                usd = float(e.get("usd_total", usd) or 0.0)
                tin = int(e.get("input_total", tin) or 0)
                tout = int(e.get("output_total", tout) or 0)
                # Sticky, like the run surface: once any update flags an
                # under-estimate the whole figure is one.
                partial = partial or bool(e.get("usd_partial", False))
    return Spend(usd, tin, tout, partial)


def state_dir_seq(dir_name: str) -> int | None:
    """The transition seq encoded in a `<seq>-<state>` per-state log dir name."""
    head = dir_name.split("-", 1)[0]
    return int(head) if head.isdigit() else None


def machine_spend(events: Sequence[object], root: Path, *, alive: bool) -> tuple[Spend, str]:
    """Total spend for a machine instance and the in-flight state's name (`""`
    if none): the sum of completed states' booked AgentFacts and crashed
    attempts' booked AttemptSpends, PLUS the live spend of the
    currently-running state.

    A state books its StepEvent only when it completes, so a machine
    mid-agent-state otherwise reads $0/dead while burning money. The running
    state's per-state log dir is numbered with the current transition seq, which
    has no StepEvent yet, so a newest-log seq absent from the booked seqs is
    unambiguously the in-flight state (no double-count); we fold it only when the
    worker is alive so a crashed in-flight log is ignored."""
    total = Spend()
    step_seqs: set[int] = set()
    for event in events:
        if isinstance(event, StepEvent):
            step_seqs.add(event.seq)
            if isinstance(event.fact, AgentFact):
                total += Spend(
                    event.fact.usd,
                    event.fact.input_tokens,
                    event.fact.output_tokens,
                    event.fact.usd_partial,
                )
        elif isinstance(event, AttemptSpend):
            # A crashed attempt's booked slice (see book_crashed_attempt).
            total += Spend(event.usd, event.input_tokens, event.output_tokens, event.usd_partial)
        elif isinstance(event, MachineEnd):
            # A slice that ran but never got a StepEvent (a capture that could
            # not be reduced) rides on the end event. Folded unconditionally: an
            # end with no unbooked slice contributes Spend() anyway, while
            # gating on a truthy `usd` would drop an UNPRICED slice whole (its
            # usd is 0.0 by definition) with its tokens and the sticky
            # lower-bound flag.
            total += Spend(event.usd, event.input_tokens, event.output_tokens, event.usd_partial)
    inflight_state = ""
    newest = newest_state_log(root) if alive else None
    if newest is not None:
        seq = state_dir_seq(newest.parent.name)
        if seq is not None and seq not in step_seqs:
            total += read_budget_totals(newest)
            inflight_state = newest.parent.name.split("-", 1)[-1]
    return total, inflight_state


def machine_mtime(machine_dir: Path) -> float:
    """Last activity: the journal's mtime, else the dir's."""
    for candidate in (machine_dir / "journal.jsonl", machine_dir):
        try:
            return candidate.stat().st_mtime
        except OSError:
            continue
    return 0.0


def machine_instance_dirs(state_dir: Path) -> list[Path]:
    """Every machine instance under the state machines/ dir (holds
    machine.asm.toml + journal), newest first."""
    root = machines_root(state_dir)
    if not root.is_dir():
        return []
    dirs = [d for d in root.iterdir() if d.is_dir() and (d / "machine.asm.toml").is_file()]
    return sorted(dirs, key=machine_mtime, reverse=True)


def summarize_machine_dir(machine_dir: Path) -> MachineSummary:
    """The instance row from its dir: spec + journal folded to the shared status
    word. A corrupt source or journal (JournalError is a MachineError) reads
    "unreadable" rather than vanishing from a listing."""
    mtime = machine_mtime(machine_dir)
    try:
        spec = load_machine(machine_dir / "machine.asm.toml")
        ms = fold_machine(spec, MachineJournal(machine_dir).read())
    except (MachineError, OSError):
        return MachineSummary(machine_dir.name, "", "", "unreadable", "", mtime)
    reason = ms.ended.reason if ms.ended is not None and ms.ended.status == "failed" else ""
    if ms.ended is None and (blocked_in := machine_operator_blocked(machine_dir)):
        reason = f"waiting on an approval in {blocked_in}"
    return MachineSummary(
        name=machine_dir.name,
        machine=ms.machine,
        current=ms.current,
        status=machine_word_for_dir(ms, machine_dir),
        reason=reason,
        mtime=mtime,
    )


def machine_files(cwd: Path) -> list[Path]:
    """The authored `.asm.toml` files a hub offers to run or create from: the
    cwd top level (where `machine create` writes by default) plus a
    conventional `machines/` subdir, sorted by path."""
    found: set[Path] = set(cwd.glob("*.asm.toml"))
    sub = cwd / "machines"
    if sub.is_dir():
        found.update(sub.glob("*.asm.toml"))
    return sorted(found)


MachineVerb = Literal["stop", "poke", "steer", "answer"]


def verb_refusals(
    name: str,
    *,
    ended: MachineResult | None,
    alive: bool,
    waiting: bool,
) -> dict[MachineVerb, str]:
    """Why each verb cannot reach machine *name*, "" where it can. THE decision,
    pure like :func:`machine_status_word`: an unknown machine is named as
    unknown (not as stopped); an ended one consumes no signal; a stopped one has
    no state polling a marker (a poke still wakes it); a live one in a wait
    state reads no steer (a poke wakes it) but takes a stop and an answer.

    The probes are the caller's, so a caller holding the fold does not read the
    journal a second time to reach the same answer.
    """
    if ended is not None:
        done = f"machine {name!r} already ended in {ended.state!r} ({ended.status}: {ended.reason})"
        return {
            "stop": f"{done}; nothing to stop",
            "poke": f"{done}; a poke would never be consumed",
            "steer": f"{done}; there is no state to steer",
            "answer": f"{done}; the prompt is closed",
        }
    if not alive:
        return {
            "stop": (
                f"machine {name!r} is not running; nothing to stop (a parked instance resumes"
                " with `agent6 machine run`)"
            ),
            "steer": (
                f"machine {name!r} is not running, so no agent state would read a steer"
                " (poke it to wake a waiting machine)"
            ),
            "answer": f"machine {name!r} is not running; poke it to wake a waiting machine",
            "poke": "",  # waking a waiting machine is what a poke is for
        }
    steer = (
        f"machine {name!r} is waiting; a wait state reads no steer (poke it to wake it)"
        if waiting
        else ""
    )
    return {"stop": "", "poke": "", "answer": "", "steer": steer}


def machine_verb_refusals(machine_dir: Path, name: str) -> dict[MachineVerb, str]:
    """:func:`verb_refusals` over an instance dir, reading the journal itself.
    A front-end paints every verb at once, so it asks once; the CLI asks for the
    one verb it is about to run through :func:`machine_verb_refusal`."""
    verbs: tuple[MachineVerb, ...] = ("stop", "poke", "steer", "answer")
    if not machine_dir.is_dir():
        return dict.fromkeys(verbs, f"no machine {name!r}")
    try:
        events = MachineJournal(machine_dir).read()
    except JournalError as exc:
        return dict.fromkeys(verbs, f"machine {name!r}: {exc}")
    end = events[-1] if events and isinstance(events[-1], MachineEnd) else None
    return verb_refusals(
        name,
        ended=MachineResult.from_end(end) if end is not None else None,
        alive=worker_is_alive(machine_dir),
        waiting=_in_wait_state(machine_dir, events),
    )


def wait_line(machine_id: str, state: str, wake_at: str) -> str:
    """Where a parked machine is, when it wakes, and how to wake it now.

    `machine status` and the foreground `machine run` say the same sentence:
    the run blocks in the wait with nothing on the terminal otherwise, which
    reads as a hang for the whole interval.
    """
    poke = f"agent6 machine poke {machine_id} [--message TEXT]"
    if wake_at:
        return f"waiting in {state!r}: wakes at {wake_at}; a poke wakes it now: {poke}"
    return f"waiting in {state!r} for a poke: {poke}"


def machine_verb_refusal(machine_dir: Path, name: str, verb: MachineVerb) -> str:
    """Why *verb* cannot reach machine *name* now, or "" when it can
    (:func:`machine_verb_refusals` for the whole set)."""
    return machine_verb_refusals(machine_dir, name)[verb]


def _in_wait_state(machine_dir: Path, events: Sequence[object]) -> bool:
    """A live machine's current state is a wait: an armed `--exit-on-wait`
    wait, or the fold's current state of kind `wait`. An unloadable source
    reads as not waiting; the operation's own error then says what is wrong."""
    if machine_is_parked(machine_dir):
        return True
    try:
        spec = load_machine(machine_dir / "machine.asm.toml")
    except MachineError:
        return False
    ms = fold_machine(spec, events)
    return ms.current_kind == "wait"


def machine_word_for_dir(ms: MachineState, machine_dir: Path) -> str:
    """THE status word for a machine instance with a dir on disk:
    :func:`machine_status_word` fed the two dir probes (armed wait, worker
    pid), so surfaces cannot pair the probes differently."""
    return machine_status_word(
        ms,
        parked=machine_is_parked(machine_dir),
        alive=worker_is_alive(machine_dir),
        blocked=bool(machine_operator_blocked(machine_dir)),
    )


def notification_key(n: NotificationView) -> tuple[str, str, str]:
    """A stable identity for a notification, for dedup across the sliding window
    (front-ends track which they have surfaced by this key, not by a count into
    the capped `notifications` tuple). Mirrors the web client's ts|state|message."""
    return (n.ts, n.state, n.message)


def newest_state_log(root: Path) -> Path | None:
    """The logs.jsonl of the most recent agent-state execution (highest seq), or
    None. That is the state whose reasoning a watcher should follow live."""
    states = root / "states"
    if not states.is_dir():
        return None

    def seq_of(p: Path) -> int:
        head = p.name.split("-", 1)[0]
        return int(head) if head.isdigit() else -1

    for d in sorted((p for p in states.iterdir() if p.is_dir()), key=seq_of, reverse=True):
        log = d / LOGS_NAME
        if log.is_file():
            return log
    return None


def read_complete_lines(path: Path, offset: int) -> tuple[list[str], int]:
    """Complete new lines of *path* past byte *offset*, plus the new offset
    (the start of any partial trailing line, re-read next poll).

    Byte reads: a poll can hit EOF mid multibyte UTF-8 sequence (the writer
    flushes long lines in several syscalls) and a text-mode readline would
    raise UnicodeDecodeError there. Only complete lines are decoded."""
    lines: list[str] = []
    pos = offset
    try:
        with path.open("rb") as fh:
            fh.seek(offset)
            while True:
                pos = fh.tell()
                raw = fh.readline()
                if not raw.endswith(b"\n"):
                    break
                lines.append(raw.decode("utf-8", errors="replace"))
    except OSError:
        pass
    return lines, pos


@dataclass
class MachineWatchCursor:
    """What a live machine watcher has already surfaced.

    One implementation of the three dedup rules every front-end (the CLI watch
    loop, the TUI machine screen) must agree on: transitions by count,
    notifications by identity (`ms.notifications` is a sliding window, so a
    count index would miss every notify past its cap), and the newest state
    log by (path, byte offset) with only complete lines consumed."""

    seen_steps: int = 0
    seen_notifications: set[tuple[str, str, str]] | None = None
    log_path: Path | None = None
    log_offset: int = 0

    def seed_notifications(self, ms: MachineState) -> None:
        """Mark every already-recorded notification as seen, so opening a watch
        does not re-announce history."""
        self.seen_notifications = {notification_key(n) for n in ms.notifications}

    def new_transitions(self, ms: MachineState) -> list[TransitionView]:
        out = list(ms.transitions[self.seen_steps :])
        self.seen_steps = len(ms.transitions)
        return out

    def new_notifications(self, ms: MachineState) -> list[NotificationView]:
        if self.seen_notifications is None:
            self.seen_notifications = set()
        out: list[NotificationView] = []
        for n in ms.notifications:
            key = notification_key(n)
            if key not in self.seen_notifications:
                self.seen_notifications.add(key)
                out.append(n)
        return out

    def advance_log(self, root: Path) -> tuple[Path | None, bool]:
        """Track the newest per-state log under *root*. Returns the current log
        and True when it changed; the caller resets its render state (elapsed
        anchor, pending text) and announces the new agent state."""
        newest = newest_state_log(root)
        if newest != self.log_path:
            self.log_path, self.log_offset = newest, 0
            return newest, True
        return newest, False

    def read_log_lines(self) -> list[str]:
        """Complete new lines of the current state log since the last poll."""
        if self.log_path is None:
            return []
        lines, self.log_offset = read_complete_lines(self.log_path, self.log_offset)
        return lines


def machine_state_as_dict(ms: MachineState, machine_dir: Path | None = None) -> dict[str, Any]:
    """The JSON-able wire form of a MachineState, stable field names: what
    `agent6 attach --json` and a web client serialize.

    Pass *machine_dir* whenever the caller has one: `status` is then THE
    dir-aware word (:func:`machine_word_for_dir`), so a client can tell a
    parked "waiting" instance from a running one. Without it a client's only
    liveness signal is `ended`, and Steer on a parked machine looked live."""
    d = asdict(ms)
    if machine_dir is not None:
        parked = machine_is_parked(machine_dir)
        alive = worker_is_alive(machine_dir)
        d["status"] = machine_status_word(
            ms,
            parked=parked,
            alive=alive,
            blocked=bool(machine_operator_blocked(machine_dir)),
        )
        # Every verb's refusal, so a front-end gates and labels its buttons from
        # the one decision the CLI and the TUI already use instead of deriving
        # its own from the status word (which conflates parked, in-a-wait-state
        # and live-but-blocked). Fed from THIS fold and these probes: asking
        # `machine_verb_refusals` would read the journal and fold it again,
        # doubling the work of every SSE frame.
        d["refusals"] = verb_refusals(
            machine_dir.name,
            ended=ms.ended,
            alive=alive,
            waiting=parked or ms.current_kind == "wait",
        )
    return d
