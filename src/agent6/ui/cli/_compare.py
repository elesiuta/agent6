# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""CLI adapter over the shared ranking core (`agent6.app.compare`).

The rank/report core is headless in `app.compare`; this module supplies the two
presentation pieces it cannot: the console `judging...` spinner shown while the
judge call is in flight, and the reviewer-provider builder wired from the
configured `reviewer` role. `rank` binds those into `app.compare.rank` so
`sessions compare` (`sessions_cmds.py`) and the fan-out's auto-compare share one
implementation.
"""

from __future__ import annotations

import contextlib
import sys
import threading
from collections.abc import Generator
from pathlib import Path

from agent6.app.compare import RankOutcome
from agent6.app.compare import rank as core_rank
from agent6.app.providers import build_role_provider
from agent6.budget import BudgetTracker
from agent6.config import Config
from agent6.providers import Provider, TranscriptSink
from agent6.ui.cli._console_view import _HEARTBEAT_TICK_S
from agent6.ui.cli._terminal_guard import raw_stream
from agent6.viewmodel.format import spinner_frame
from agent6.workflows.judge import CandidateBrief

__all__ = ["rank"]


@contextlib.contextmanager
def _judging_status() -> Generator[None]:
    """Show progress around the (~50-60s, otherwise silent) judge call: a real
    terminal gets the SAME spinner glyphs/cadence as the run stream's
    provider-call heartbeat (`_console_view`'s `_HEARTBEAT_TICK_S` + the shared
    spinner);
    a non-tty (piped, detached orchestrator) gets one plain line so logs stay
    truthful -- no animation frames written to a file."""
    if not sys.stdout.isatty():
        print("judging...")
        yield
        return
    stop = threading.Event()

    def spin() -> None:
        i = 0
        while True:
            raw_stream(sys.stdout).write("\r\x1b[2K")
            sys.stdout.write(f"{spinner_frame(i)} judging...")
            sys.stdout.flush()
            i += 1
            if stop.wait(_HEARTBEAT_TICK_S):
                return

    thread = threading.Thread(target=spin, daemon=True)
    thread.start()
    try:
        yield
    finally:
        stop.set()
        thread.join(timeout=1.0)
        raw_stream(sys.stdout).write("\r\x1b[2K")
        sys.stdout.flush()


def _reviewer_provider(cfg: Config, sink: TranscriptSink, budget: BudgetTracker) -> Provider:
    """Build the configured `reviewer` provider for the judge call."""
    return build_role_provider(cfg, "reviewer", transcript_sink=sink, budget=budget)


def rank(cfg: Config, candidates: list[CandidateBrief], *, transcript_dir: Path) -> RankOutcome:
    """Rank candidates best-first via the shared core, injecting the CLI's
    console judging-status and reviewer-provider builder. The single rank
    implementation `sessions compare` and `--parallel`'s auto-compare both use."""
    return core_rank(
        cfg,
        candidates,
        transcript_dir=transcript_dir,
        build_provider=_reviewer_provider,
        judging_status=_judging_status,
    )
