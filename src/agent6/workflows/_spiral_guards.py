# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The dispatch loop's degenerate-spiral bookkeeping, in one object.

Two interlocking streaks guard a worker that stops making progress: the
REPEAT streak (the same (tool, args) signature back to back, which powers the
identical-result stub and the repeat warning) and the ERROR streak (the same
tool failing the same way, which climbs the nudge/escalate/stop ladder). They
share `last_served_content` — the bytes most recently served to the model,
success or error — and a successful dispatch must clear the whole error
spiral: that reset-covers-every-field invariant lives in `note_success` so
adding a field cannot silently miss the reset site.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class SpiralGuard:
    """Leg-local (never snapshotted): a resumed leg starts unspiralled."""

    last_call_sig: str | None = None
    call_streak: int = 0
    last_served_content: str | None = None
    # The iteration the repeat warning last fired at, so it re-arms only
    # after a quiet iteration rather than firing every turn of a spiral.
    warned_at_iteration: int = 0
    error_sig: str | None = None
    error_streak: int = 0
    error_nudges_used: int = 0
    last_error_was_denial: bool = False

    def note_call(self, sig: str, *, polling: bool = False) -> None:
        """Same signature back to back extends the repeat streak; anything
        else restarts it.

        A POLL is not a repeat: `read_background` exists to be called again
        with the same id until the job ends, which `run_command`'s own
        description tells the model to do; counted as a spiral it would draw
        three nudges and then end the run for following the instruction."""
        if sig == self.last_call_sig and not polling:
            self.call_streak += 1
        else:
            self.last_call_sig = sig
            self.call_streak = 1

    def stub_repeat(self, content: str, *, min_chars: int) -> bool:
        """Serve a short stub instead of *content*? Only when the call is a
        back-to-back repeat AND the result bytes are unchanged AND big enough
        that the stub actually saves context."""
        return (
            self.call_streak >= 2
            and content == self.last_served_content
            and len(content) > min_chars
        )

    def note_success(self, content: str) -> None:
        """A successful dispatch is progress: remember what was served and
        clear the WHOLE error spiral."""
        self.last_served_content = content
        self.error_sig = None
        self.error_streak = 0
        self.error_nudges_used = 0
        self.last_error_was_denial = False

    def note_error(self, sig: str, *, denial: bool, content: str) -> None:
        """Same error signature extends the streak; a new one restarts it and
        re-arms the nudge allowance (a NEW failure mode may nudge again)."""
        self.last_served_content = content
        self.last_error_was_denial = denial
        if sig == self.error_sig:
            self.error_streak += 1
        else:
            self.error_sig = sig
            self.error_streak = 1
            self.error_nudges_used = 0
