# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The two output channels the app pipelines write through instead of calling
`print` directly.

`out` is stdout (a piped result the operator captures); `err` is stderr (status,
warnings, refusals). `ui/cli` is the composition root that owns the real streams
(`STDIO_REPORTER`); a test or an alternate front-end injects a capturing pair.
Each channel takes one already-formatted line and writes it exactly as the
matching `print` would, so threading the reporter is behaviour-preserving."""

from __future__ import annotations

import sys
from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Reporter:
    out: Callable[[str], None]
    err: Callable[[str], None]
    # The cost receipt (`BudgetTracker.format_summary`): the one end-of-run
    # block a live view also renders on its done line, so a front-end whose
    # live view carries it routes the receipt to its log. None means `out`.
    receipt: Callable[[str], None] | None = None

    def cost(self, msg: str) -> None:
        (self.receipt or self.out)(msg)

    # The four stderr conventions, owned here so every lifecycle words them
    # the same: a refusal (the run does not start, exit 2), an error, a loud
    # warning, and a status note.
    def refuse(self, msg: str) -> None:
        self.err(f"REFUSING: {msg}")

    def error(self, msg: str) -> None:
        self.err(f"ERROR: {msg}")

    def warn(self, msg: str) -> None:
        self.err(f"[agent6] WARNING: {msg}")

    def note(self, msg: str) -> None:
        self.err(f"[agent6] {msg}")


def _print_out(msg: str) -> None:
    print(msg)


def _print_err(msg: str) -> None:
    print(msg, file=sys.stderr)


# The real-stream wiring: identical to `print(msg)` / `print(msg, file=sys.stderr)`.
# The default the app entry points fall back to and `ui/cli` relies on.
STDIO_REPORTER = Reporter(out=_print_out, err=_print_err)
