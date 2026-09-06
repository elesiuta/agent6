# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The run's central truth: the verify gate's last verdict, and whether the
tree moved since.

Every consumer of "is the run green" (the finish gates, the review panel's
grounding, the resume snapshot, the turn notices) reads this one object
instead of poking parallel booleans on the loop state. Transitions live here
so the streak/reset rules cannot drift apart across call sites.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class VerifyVerdict:
    """Mutable verify bookkeeping for one run leg.

    `last_ok` is the last verify's result (None = no verify yet).
    `baseline_ok` answers "was the gate already failing before this run
    touched anything?": recorded when a verify runs against an unmodified
    tree; None means no such verify happened, and "I do not know" is the
    honest answer then. `edited_since` spans iterations: a stale earlier
    pass must not count as currently green for the finish gate.
    `fail_streak` counts consecutive failures sharing one normalized
    signature (the no-progress spiral guard); a green verify or a new
    signature resets it. `denied` records an approval denial of the gate
    (a human's no, or the unattended auto-deny): the gate is withheld for
    the rest of the run, like `run_commands = "no"`, instead of a finish
    bounce nobody can discharge.
    """

    last_ok: bool | None = None
    denied: bool = False
    # The full gate overran verify_timeout_s: harness gates run scoped
    # (command + tests nearest the run's diff) until a full run passes.
    scoped: bool = False
    baseline_ok: bool | None = None
    last_tail: str = ""
    fail_signature: str = ""
    fail_streak: int = 0
    broken_warned: bool = False
    edited_since: bool = False
    ever_passed: bool = False
    ever_failed: bool = False
    # The gate adopted mid-run (`()` when the gate is configured or absent),
    # and every adopted argv that proved unrunnable, never re-adopted.
    adopted: tuple[str, ...] = ()
    unadoptable: set[tuple[str, ...]] = field(default_factory=set)

    def note_pass(self) -> None:
        """A green verify: the tree as of now is verified; streaks reset."""
        self.last_ok = True
        self.ever_passed = True
        self.edited_since = False
        self.fail_signature = ""
        self.fail_streak = 0

    def note_fail(self, signature: str) -> None:
        """A red verify: extend the streak when the failure looks the same,
        restart it when the signature changed (a NEW stuck point)."""
        self.last_ok = False
        self.ever_failed = True
        if signature == self.fail_signature:
            self.fail_streak += 1
        else:
            self.fail_signature = signature
            self.fail_streak = 1

    def note_edit(self) -> None:
        self.edited_since = True

    @property
    def green_and_untouched(self) -> bool:
        """The finish gate's question: verified green, and nothing edited
        since that verify."""
        return self.last_ok is True and not self.edited_since
