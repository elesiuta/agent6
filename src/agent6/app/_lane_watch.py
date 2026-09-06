# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Watching started lanes until they land, and saying what they wait on.

The fan-out's await loop and the single-lane await and drain behind
`run_lane_to_completion`, the live symlink a lane gets under the origin's
runs dir while it runs, and the pending-prompt probe the status line uses.
The types a lane is described with (`LaneSpec`, `LaneResult`) are
`workflows/subrun`'s; `app/parallel.py`, the orchestrator this serves, drives
the lanes.
"""

from __future__ import annotations

import contextlib
import json
import threading
import time
from collections.abc import Callable
from pathlib import Path

from agent6.app.reporter import STDIO_REPORTER, Reporter
from agent6.sessions.ipc import request_stop, worker_is_alive
from agent6.sessions.layout import LOGS_NAME, bucket_dir
from agent6.viewmodel import summarize_session_dir
from agent6.viewmodel.format import format_usd
from agent6.workflows.subrun import LaneResult, LaneSpec

# How often the await loop polls lane liveness, and how long Ctrl+C waits for a
# stop-requested lane to finish its in-flight step before giving up on it.
POLL_INTERVAL_S = 2.0


STOP_GRACE_S = 30.0


def lane_terminal(session_dir: Path, status: str, worker_is_alive: Callable[[Path], bool]) -> bool:
    """Terminal gate for an awaited lane: the fold left "running" AND the worker
    pid is cleared/dead. session.end lands in logs.jsonl before the lane's teardown
    clears worker.pid, so status alone races the teardown, and importing inside
    that window would misread a finished lane as still running. A lane that dies
    WITHOUT a session.end cannot hang this gate: the fold flips a dead recorded pid
    to "stale" at once, a pid-less silent lane to "stale" after its bounded
    silence window, and a lane that never wrote logs reads "?" (see
    `summarize_session_dir`)."""
    return status != "running" and not worker_is_alive(session_dir)


def await_lane(
    res: LaneResult,
    *,
    poll_interval_s: float = POLL_INTERVAL_S,
    should_stop: Callable[[], bool] | None = None,
) -> bool:
    """Block until *res*'s lane is terminal (True), awaited on its REAL run
    dir, or until *should_stop* goes true first (False): the coordinator's
    abort channel must be able to interrupt a group await that otherwise
    blocks until every lane ends. Same gate as the fan-out's `await_lanes`,
    for a single lane."""
    while True:
        summary = summarize_session_dir(res.session_dir)
        if lane_terminal(res.session_dir, summary.status, worker_is_alive):
            return True
        if should_stop is not None and should_stop():
            return False
        time.sleep(poll_interval_s)


def drain_lane(
    res: LaneResult, *, poll_interval_s: float, hard_stop: threading.Event | None
) -> bool:
    """Bounded post-stop grace (mirrors the fan-out's stop_and_drain): True when
    the lane lands terminal in time, so its finished work still imports; False
    to leave it running un-imported. A hard stop (process teardown) skips the
    wait."""
    deadline = time.monotonic() + STOP_GRACE_S
    while time.monotonic() < deadline:
        if hard_stop is not None and hard_stop.is_set():
            return False
        summary = summarize_session_dir(res.session_dir)
        if lane_terminal(res.session_dir, summary.status, worker_is_alive):
            return True
        if hard_stop is not None:
            if hard_stop.wait(poll_interval_s):
                return False
        else:
            time.sleep(poll_interval_s)
    return False


def lane_link(origin_state: Path, session_id: str) -> Path:
    return bucket_dir(origin_state, "runs") / session_id


def symlink_lane(origin_state: Path, res: LaneResult) -> None:
    """Symlink a located lane's (clone-side) run dir into the origin's `runs/` so
    `agent6 sessions`/hub shows it live. Replaced by the real imported dir at import."""
    link = lane_link(origin_state, res.spec.session_id)
    link.parent.mkdir(parents=True, exist_ok=True)
    with contextlib.suppress(FileNotFoundError):
        link.unlink()
    with contextlib.suppress(OSError):
        link.symlink_to(res.session_dir)


def await_lanes(
    started: list[LaneResult],
    *,
    already_interrupted: bool = False,
    should_stop: Callable[[], bool] | None = None,
    reporter: Reporter = STDIO_REPORTER,
) -> bool:
    """Poll every started lane's REAL run dir (in the clone's state; the origin
    symlink is a view for the hub, never the source of truth) until it is
    terminal (`lane_terminal`), printing one line per lane on a status/cost
    change. Returns True if interrupted (Ctrl+C): request a clean stop on each
    still-running lane, wait a bounded grace for them to finish their in-flight
    step, then return so the caller imports what landed.

    `already_interrupted=True` (a Ctrl+C the spawn loop caught before the await
    even began) skips the normal poll and goes straight into that same stop-grace
    path, so a mid-spawn interrupt stops the already-started lanes identically.
    *should_stop* (the coordinator's own stop request, read between polls) ends
    the wait the same way."""
    pending = {r.spec.session_id: r for r in started}
    seen: dict[str, tuple[str, str, float]] = {}

    def poll_once() -> None:
        for rid, res in list(pending.items()):
            summary = summarize_session_dir(res.session_dir)
            # A "waiting" lane is blocked on an approval/question no detached
            # lane can answer; point the operator at the hub. pending_prompt
            # supplies only the approval-vs-question wording.
            waiting = pending_prompt(res.session_dir) if summary.status == "waiting" else ""
            key = (summary.status, waiting, round(summary.cost_usd, 4))
            if seen.get(rid) != key:
                seen[rid] = key
                print_lane_status(
                    res.spec, summary.status, summary.cost_usd, waiting=waiting, reporter=reporter
                )
            if lane_terminal(res.session_dir, summary.status, worker_is_alive):
                pending.pop(rid)

    def stop_and_drain() -> None:
        reporter.err("\n[agent6] interrupted; stopping lanes...")
        for res in pending.values():
            request_stop(res.session_dir)
        deadline = time.monotonic() + STOP_GRACE_S
        with contextlib.suppress(KeyboardInterrupt):
            while pending and time.monotonic() < deadline:
                poll_once()
                if pending:
                    time.sleep(POLL_INTERVAL_S)

    if already_interrupted:
        stop_and_drain()
        return True
    try:
        while pending:
            poll_once()
            if pending and should_stop is not None and should_stop():
                stop_and_drain()
                return True
            if pending:
                time.sleep(POLL_INTERVAL_S)
        return False
    except KeyboardInterrupt:
        stop_and_drain()
        return True


# The two prompt/answer event pairs a lane can block on, for `pending_prompt`.
_PROMPT_KIND = {"approval.prompt": "approval", "question.prompt": "a question"}


_ANSWER_EVENTS = frozenset({"approval.answer", "question.answer"})


def pending_prompt(session_dir: Path) -> str:
    """ "approval" / "a question" if the lane is blocked on an unanswered prompt,
    else "". The worker emits `approval.prompt`/`question.prompt` then BLOCKS on
    its `*.answer` (lanes run with AGENT6_DETACHED_AWAY=wait, so a prompt with no
    hub attached waits rather than denies), so the LAST prompt/answer event in
    logs.jsonl decides it -- a cheap trailing scan, no `*.request` marker exists
    for approvals/questions. Deliberately not the heavyweight SessionState fold; the
    fan-out status line needs only this one bit."""
    try:
        lines = (session_dir / LOGS_NAME).read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return ""
    for raw in reversed(lines):
        if "approval." not in raw and "question." not in raw:
            continue  # fast reject before json.loads
        try:
            ev = json.loads(raw)
        except ValueError:
            continue
        etype = ev.get("type") if isinstance(ev, dict) else None
        if etype in _ANSWER_EVENTS:
            return ""
        if etype in _PROMPT_KIND:
            return _PROMPT_KIND[etype]
    return ""


def print_lane_status(
    spec: LaneSpec,
    status: str,
    cost: float,
    *,
    waiting: str = "",
    reporter: Reporter = STDIO_REPORTER,
) -> None:
    model = f" ({spec.model})" if spec.model else ""
    cost_s = f"  {format_usd(cost)}" if cost > 0 else ""
    state = (
        f"waiting on {waiting} (answer via agent6 attach {spec.session_id}, the web or TUI hub)"
        if waiting
        else status
    )
    reporter.note(f"lane {spec.lane} [{spec.session_id}]{model}: {state}{cost_s}")
