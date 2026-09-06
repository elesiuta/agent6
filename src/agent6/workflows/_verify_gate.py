# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The harness-run verify gate (`[workflow].verify_when`): when the harness
runs the gate itself, and what the model is told about a run it did not
start.

`finish` certifies the tree a run ends on; `step` also judges every editing
turn; `never` leaves every gate run to the model's own `run_verify_command`
calls. A turn whose own verify call already judged the tree is never judged
twice. These are pure decisions over the turn's facts; the loop owns the
running and the bookkeeping.
"""

from __future__ import annotations

from typing import Literal

from agent6.tools.results import ExecResult

VerifyWhen = Literal["finish", "step", "never"]
HarnessVerifyWhy = Literal["finish", "step"]

_TAIL_CHARS = 2000


def harness_verify_due(
    *,
    when: VerifyWhen,
    gate_present: bool,
    verified_this_turn: bool,
    changed_this_turn: bool,
    finishing: bool,
    green_and_untouched: bool,
) -> HarnessVerifyWhy | None:
    """Why the harness runs the gate after this turn, or None: `step` after a
    turn that changed the tree, `finish` when finish_session arrived over a
    tree no green run covers; never on top of the model's own run."""
    if when == "never" or not gate_present or verified_this_turn:
        return None
    if finishing and not green_and_untouched:
        return "finish"
    if when == "step" and changed_this_turn:
        return "step"
    return None


def harness_verify_notice(result: ExecResult, why: HarnessVerifyWhy) -> str:
    """What the model sees of a gate run it did not start: the verdict and
    the output tail, labelled by what triggered it."""
    verdict = "passed" if result.returncode == 0 else f"exit {result.returncode}"
    tail = f"{result.stdout}\n{result.stderr}".strip()[-_TAIL_CHARS:]
    head = f"[harness verify] {why}: verify_command {verdict} ({result.duration_s:.0f}s)."
    return f"{head}\n{tail}" if tail else head


def scoped_verify_notice(result: ExecResult, *, timeout_s: float, paths: tuple[str, ...]) -> str:
    """What the model sees of a scoped gate run: the full command overran its
    budget (this run or an earlier one; scoping stays armed), so the gate ran
    only the tests nearest the run's change, listed by path. A scoped green
    certifies less than a full pass and the notice says so. One notice for
    the harness gate and the follow-up to the model's own timed-out call."""
    verdict = "passed" if result.returncode == 0 else f"exit {result.returncode}"
    tail = f"{result.stdout}\n{result.stderr}".strip()[-_TAIL_CHARS:]
    head = (
        f"[verify] verify_command overran its {timeout_s:.0f}s budget;"
        f" the gate ran scoped to the tests nearest the run's change ({', '.join(paths)}):"
        f" {verdict} ({result.duration_s:.0f}s). A scoped green is not a full-suite pass."
    )
    return f"{head}\n{tail}" if tail else head


def gate_withheld_notice(head: str, exc: Exception) -> str:
    """A gate run not approved (a human's no, or the unattended auto-deny):
    withheld for the rest of the run, whichever site asked."""
    return (
        f"{head}: not run: {exc}."
        " The gate is withheld for the rest of the run; the run ends unverified."
    )


def finish_red_notice(*, used: int, retries: int) -> str:
    """The finish came back: the gate did not certify the tree."""
    left = retries - used
    ending = (
        "the next red finish ends the run, reported as finished, not passed"
        if left == 0
        else (
            f"{left} more red finish{'es return' if left != 1 else ' returns'}"
            " before the run ends red"
        )
    )
    return (
        f"[harness] finish_session not honoured: the verify gate did not certify the tree"
        f" (return {used} of {retries}); {ending}."
    )
