# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Machine spend accounting: reconstruct a machine's dollar/token totals from
its journal + per-state event logs.

The agent loop writes a `budget.update` event per turn carrying cumulative
usd/token totals; the last one in a state's log is that state's running total.
`read_budget_totals` reads it, used both to salvage a killed/timed-out agent
subprocess's spend (its `result.json` never landed) and to fold an in-flight
state's live spend into `machine status` (its `StepEvent` is not booked yet).
"""

from __future__ import annotations

import contextlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from agent6.machine import AgentFact, AttemptSpend, MachineEnd, MachineJournal, StepEvent
from agent6.viewmodel.machine_state import newest_state_log


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


def book_crashed_attempt(journal: MachineJournal, root: Path) -> None:
    """Journal an `AttemptSpend` for an orphaned in-flight state log, then
    retire the log dir (renamed `crashed-<timestamp>-<original>`, out of
    every seq matcher and unique per crash) so the booked slice can never fold
    twice and the re-run starts a fresh log.

    A supervisor death mid-agent-state leaves real provider spend recorded
    only in the per-state log; without this, resume re-granted the budget and
    status under-reported. Called by the resuming supervisor before the drive
    re-runs the state. No orphan log, an already-booked seq, or an empty
    total books nothing.
    """
    newest = newest_state_log(root)
    if newest is None:
        return
    seq = _state_dir_seq(newest.parent.name)
    if seq is None:
        return
    if any(isinstance(e, StepEvent) and e.seq == seq for e in journal.read()):
        return
    state = newest.parent.name.split("-", 1)[-1]
    # Retire the log dir FIRST, under a name no later attempt can collide with:
    # the seq does not advance across a crashed attempt, so a second crash in
    # the same state reused `crashed-<seq>-<state>` and the rename died
    # ENOTEMPTY -- after appending a second booking for the same attempt. A
    # crash between the rename and the append now loses one booking, which is
    # the behaviour this function was added to improve on, never a duplicate.
    ts = datetime.now(UTC).isoformat(timespec="microseconds")
    retired = newest.parent.with_name(f"crashed-{ts.replace(':', '')}-{newest.parent.name}")
    newest.parent.rename(retired)
    spend = read_budget_totals(retired / newest.name)
    if spend.usd or spend.input_tokens or spend.output_tokens:
        journal.append(
            AttemptSpend(
                ts=ts,
                seq=seq,
                state=state,
                usd=spend.usd,
                usd_partial=spend.partial,
                input_tokens=spend.input_tokens,
                output_tokens=spend.output_tokens,
            )
        )


def _state_dir_seq(dir_name: str) -> int | None:
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
            # gating on a truthy `usd` dropped an UNPRICED slice whole -- its
            # usd is 0.0 by definition, taking 48k tokens and the sticky
            # lower-bound flag with it.
            total += Spend(event.usd, event.input_tokens, event.output_tokens, event.usd_partial)
    inflight_state = ""
    newest = newest_state_log(root) if alive else None
    if newest is not None:
        seq = _state_dir_seq(newest.parent.name)
        if seq is not None and seq not in step_seqs:
            total += read_budget_totals(newest)
            inflight_state = newest.parent.name.split("-", 1)[-1]
    return total, inflight_state
