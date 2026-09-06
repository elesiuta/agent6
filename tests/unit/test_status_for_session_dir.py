# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""One status decision for any run with a dir.

``status_for_session_dir`` is the single place a run's status word is decided from
its dir; listings (``summarize_session_dir``) and every header feed it facts from
either producer -- the tolerant scanner (``LogScan``) or the typed fold
(``SessionState``). The matrix pins the word per dir state for BOTH producers and
for the listing, so no two surfaces can disagree about any non-``session.end``
state (parked/created/starting/stale/waiting each shipped a real
surface-disagreement bug before this existed).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from agent6.viewmodel.listing import (
    LogScan,
    scan_session_log,
    status_for_session_dir,
    summarize_session_dir,
)
from agent6.viewmodel.state import fold_session, status_facts

LIVE = os.getpid()
DEAD = 999999999


def _mk(
    tmp_path: Path,
    name: str,
    events: list[dict[str, object]] | None,
    *,
    parked: str = "",
    pid: int | None = None,
) -> Path:
    d = tmp_path / name
    d.mkdir()
    manifest: dict[str, object] = {"mode": "run", "session_id": name, "user_task": "t"}
    if parked:
        manifest["parked_task"] = parked
        manifest["parked_reason"] = "checkout busy"
    (d / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    if events is not None:
        (d / "logs.jsonl").write_text(
            "".join(json.dumps(e) + "\n" for e in events), encoding="utf-8"
        )
    if pid is not None:
        (d / "worker.pid").write_text(str(pid), encoding="utf-8")
    return d


_START: dict[str, object] = {"type": "session.start", "mode": "run", "user_task": "t"}
_TOOL: dict[str, object] = {"type": "tool.call", "name": "grep", "args": {}}
_APPROVAL: dict[str, object] = {"type": "approval.prompt", "id": "approval-1", "prompt": "ok?"}
_QUESTION: dict[str, object] = {
    "type": "question.prompt",
    "id": "q-1",
    "questions": [{"question": "x?"}],
}


_ANSWER: dict[str, object] = {"type": "approval.answer", "id": "approval-1", "approved": True}
_RESUME: dict[str, object] = {"type": "loop.resume.start", "iteration": 2}


def _end(reason: str, all_passed: bool = False) -> dict[str, object]:
    return {"type": "session.end", "reason": reason, "all_passed": all_passed}


# (name, events (None = no logs.jsonl), parked_task, pid, expected word, expected reason)
MATRIX: list[tuple[str, list[dict[str, object]] | None, str, int | None, str, str]] = [
    ("created", None, "", None, "created", ""),
    ("parked", None, "fix it", None, "parked", "checkout busy"),
    # A parked run being resumed: a live worker in its pre-session.start preflight
    # window reads "starting" even while the manifest still carries parked_task
    # (resume clears it only once under way).
    ("parked-resuming", None, "fix it", LIVE, "starting", ""),
    ("starting", [], "", LIVE, "starting", ""),
    ("running", [_START, _TOOL], "", LIVE, "running", ""),
    ("waiting-approval", [_START, _APPROVAL], "", LIVE, "waiting", "needs answer"),
    ("waiting-question", [_START, _QUESTION], "", LIVE, "waiting", "needs answer"),
    # The start question about uncommitted changes comes before session.start.
    ("waiting-before-start", [_QUESTION], "", LIVE, "waiting", "needs answer"),
    # ... and a worker that died on it is not "waiting" for anyone.
    ("died-on-start-question", [_QUESTION], "", DEAD, "stale", "died launching"),
    (
        "answered-runs-on",
        [_START, _APPROVAL, _ANSWER],
        "",
        LIVE,
        "running",
        "",
    ),
    # A crash while waiting, then a resume: the new leg re-prompts with fresh
    # ids, so the orphaned prompt must not read "waiting" (or duplicate) forever.
    (
        "resumed-past-orphaned-prompt",
        [_START, _APPROVAL, _RESUME, _TOOL],
        "",
        LIVE,
        "running",
        "",
    ),
    # A prompt is still pending when a LATER event lands: Ctrl-C in the
    # launching terminal emits session.steer_requested from the signal handler
    # while the approval waits. "Blocked" is about the unanswered prompt, not
    # about which event happened to be last.
    (
        "waiting-with-a-later-event",
        [_START, _APPROVAL, {"type": "session.steer_requested", "source": "sigint"}],
        "",
        LIVE,
        "waiting",
        "needs answer",
    ),
    # A FORK is driven by resume(), so its fresh log never carries a session.start:
    # keying "started" on that alone left every forked run "starting" while alive
    # and "created" (the never-started word) once it died.
    ("forked-live", [_RESUME, _TOOL], "", LIVE, "running", ""),
    ("forked-dead", [_RESUME, _TOOL], "", DEAD, "stale", ""),
    ("forked-waiting", [_RESUME, _APPROVAL], "", LIVE, "waiting", "needs answer"),
    ("stale", [_START, _TOOL], "", DEAD, "stale", ""),
    ("stale-beats-waiting", [_START, _APPROVAL], "", DEAD, "stale", ""),
    # A worker killed during PREFLIGHT (its pid file survives a kill; no
    # session.start ever lands) used to read "created" -- the fork --no-run
    # word -- beside the real dollars its verify-inference call spent.
    (
        "killed-in-preflight",
        [{"type": "budget.update", "usd_total": 0.02}],
        "",
        DEAD,
        "stale",
        "died launching",
    ),
    ("killed-in-preflight-no-log", None, "", DEAD, "stale", "died launching"),
    ("passed", [_START, _end("finish_session", True)], "", None, "passed", ""),
    (
        "passed-scoped",
        [_START, {**_end("finish_session", True), "scoped": True}],
        "",
        None,
        "passed",
        "scoped gate",
    ),
    (
        "finished-gate-red",
        [_START, {"type": "verify.end", "cmd": ["pytest"], "exit_code": 1}, _end("finish_session")],
        "",
        None,
        "finished",
        "gate red",
    ),
    ("finished-unverified", [_START, _end("finish_session")], "", None, "finished", "unverified"),
    (
        "finished-gateless",
        [_START, {**_end("finish_session"), "all_passed": None}],
        "",
        None,
        "finished",
        "",
    ),
    ("settled", [_START, _end("settled")], "", None, "finished", "unverified"),
    ("failed", [_START, _end("provider_error")], "", None, "failed", "provider_error"),
    ("stopped", [_START, _end("steer_abort")], "", None, "stopped", ""),
    ("planned", [_START, _end("finish_planning")], "", None, "planned", ""),
    ("answered", [_START, _end("answered")], "", None, "answered", ""),
]


@pytest.mark.parametrize(
    ("name", "events", "parked", "pid", "word", "reason"),
    MATRIX,
    ids=[row[0] for row in MATRIX],  # LIVE is this process's pid: never in an id
)
def test_both_fact_producers_and_the_listing_agree(
    tmp_path: Path,
    name: str,
    events: list[dict[str, object]] | None,
    parked: str,
    pid: int | None,
    word: str,
    reason: str,
) -> None:
    d = _mk(tmp_path, name, events, parked=parked, pid=pid)
    logs = d / "logs.jsonl"
    scan = scan_session_log(logs) if logs.is_file() else LogScan()
    fold = fold_session([] if events is None else events)
    assert status_for_session_dir(d, scan.status_facts()) == (word, reason)
    assert status_for_session_dir(d, status_facts(fold)) == (word, reason)
    summary = summarize_session_dir(d)
    assert (summary.status, summary.reason) == (word, reason)


def test_resume_clears_orphaned_pending_prompts() -> None:
    """A leg boundary invalidates unanswered prompts: the resumed leg re-asks
    with restarted ids, so a held-over pending entry would both mislabel the
    run "waiting" and duplicate when the new leg's same-id prompt arrives."""
    events = [
        _START,
        _APPROVAL,
        _QUESTION,
        {"type": "loop.resume.start", "iteration": 2},
        _APPROVAL,
    ]
    state = fold_session(events)
    assert len(state.pending_approvals) == 1  # the new leg's, not the orphan + a dup
    assert state.pending_questions == ()


def test_waiting_names_the_prompt_kind_and_age_in_both_producers(tmp_path: Path) -> None:
    """ "waiting · needs answer" said neither WHAT the run waits on nor for how
    long. With a prompt ts, both fact producers word it "approval 5m" /
    "question 5m" (oldest unanswered prompt); a ts-less log keeps the generic
    wording."""
    import datetime
    import os

    ts = (datetime.datetime.now(datetime.UTC) - datetime.timedelta(minutes=5)).isoformat()
    prompts: tuple[tuple[str, dict[str, object]], ...] = (
        ("approval", {"type": "approval.prompt", "id": "approval-1", "prompt": "p", "ts": ts}),
        ("question", {"type": "question.prompt", "id": "question-1", "questions": [], "ts": ts}),
    )
    for kind, ev in prompts:
        events: list[dict[str, object]] = [
            {"type": "session.start", "mode": "run", "user_task": "t"},
            ev,
        ]
        d = _mk(tmp_path, f"waiting-{kind}", events, parked="", pid=os.getpid())
        scan = scan_session_log(d / "logs.jsonl")
        fold = fold_session(events)
        expect = ("waiting", f"{kind} 5m")
        assert status_for_session_dir(d, scan.status_facts()) == expect
        assert status_for_session_dir(d, status_facts(fold)) == expect

    no_ts: list[dict[str, object]] = [
        {"type": "session.start", "mode": "run", "user_task": "t"},
        {"type": "approval.prompt", "id": "approval-1", "prompt": "p"},
    ]
    d = _mk(tmp_path, "waiting-no-ts", no_ts, parked="", pid=os.getpid())
    assert status_for_session_dir(d, scan_session_log(d / "logs.jsonl").status_facts()) == (
        "waiting",
        "needs answer",
    )
