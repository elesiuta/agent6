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

from datetime import UTC, datetime
from pathlib import Path

from agent6.machine import AttemptSpend, MachineJournal, StepEvent
from agent6.viewmodel.machine_state import (
    newest_state_log,
    read_budget_totals,
    state_dir_seq,
)


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
    seq = state_dir_seq(newest.parent.name)
    if seq is None:
        return
    if any(isinstance(e, StepEvent) and e.seq == seq for e in journal.read()):
        return
    state = newest.parent.name.split("-", 1)[-1]
    # Retire the log dir FIRST, under a name no later attempt can collide with:
    # the seq does not advance across a crashed attempt, so a seq-derived name
    # collides on a second crash in the same state. A crash between the rename
    # and the append loses one booking, never duplicates one.
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
