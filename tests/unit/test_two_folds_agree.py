# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The two folds over one session log agree on the facts they share.

`viewmodel.listing.scan_session_log` folds `logs.jsonl` for the listings and
`viewmodel.state.fold_session` folds the same log for the viewers, each with
its own rules for the resume-then-end-again shape and the budget banking.
Two readers of one stream drift; this folds every fixture both ways.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from agent6.viewmodel.listing import scan_session_log
from agent6.viewmodel.state import fold_session

_GOLDEN = Path(__file__).parent / "data" / "golden_session_logs.jsonl"

_START: dict[str, Any] = {
    "ts": "2026-07-14T10:00:00+00:00",
    "type": "session.start",
    "session_id": "two-folds-AAAA11",
    "user_task": "fix the clock",
    "mode": "run",
}
_END_PASSED: dict[str, Any] = {
    "ts": "2026-07-14T10:01:00+00:00",
    "type": "session.end",
    "reason": "finish_session",
    "all_passed": True,
}
_SHAPES: dict[str, list[dict[str, Any]]] = {
    "started": [_START],
    "passed": [_START, _END_PASSED],
    "failed": [
        _START,
        {"ts": "2026-07-14T10:00:30+00:00", "type": "budget.update", "usd_total": 0.25},
        {
            "ts": "2026-07-14T10:01:00+00:00",
            "type": "session.end",
            "reason": "provider_error",
            "all_passed": False,
        },
    ],
    "ungated": [_START, {**_END_PASSED, "all_passed": None}],
    "resumed": [
        _START,
        {"ts": "2026-07-14T10:00:30+00:00", "type": "budget.update", "usd_total": 0.25},
        _END_PASSED,
        {"ts": "2026-07-14T10:02:00+00:00", "type": "loop.resume.start", "iteration": 3},
        {
            "ts": "2026-07-14T10:02:30+00:00",
            "type": "budget.update",
            "usd_total": 0.10,
            "usd_partial": True,
        },
        {**_END_PASSED, "ts": "2026-07-14T10:03:00+00:00", "reason": "steer_abort"},
    ],
    "blocked": [
        _START,
        {"ts": "2026-07-14T10:00:30+00:00", "type": "question.prompt", "id": "question-1"},
    ],
    "answered": [
        _START,
        {"ts": "2026-07-14T10:00:30+00:00", "type": "approval.prompt", "id": "approval-1"},
        {
            "ts": "2026-07-14T10:00:40+00:00",
            "type": "approval.answer",
            "id": "approval-1",
            "approved": True,
            "source": "stdin",
        },
    ],
}


def _shared_facts(events: list[dict[str, Any]], tmp_path: Path) -> tuple[dict[str, Any], ...]:
    log = tmp_path / "logs.jsonl"
    log.write_text("".join(json.dumps(e) + "\n" for e in events), encoding="utf-8")
    scan = scan_session_log(log)
    state = fold_session(events)
    listing: dict[str, Any] = {
        "task": scan.task,
        "finished": scan.finished,
        "end_reason": scan.end_reason if scan.finished else "",
        "all_passed": scan.all_passed if scan.finished else None,
        "verify_scoped": scan.verify_scoped,
        "cost_usd": scan.cost_usd,
        "usd_partial": scan.usd_partial,
        "pins": scan.pins,
        "blocked": scan.operator_blocked,
    }
    viewer: dict[str, Any] = {
        "task": state.user_task,
        "finished": state.finished,
        "end_reason": state.end_reason if state.finished else "",
        "all_passed": state.all_passed if state.finished else None,
        "verify_scoped": state.verify_scoped,
        "cost_usd": state.budget.usd_total if scan.cost_usd is not None else None,
        "usd_partial": state.budget.usd_partial,
        "pins": state.pins,
        "blocked": any(
            not p.answered for p in (*state.pending_approvals, *state.pending_questions)
        ),
    }
    return listing, viewer


@pytest.mark.parametrize("shape", sorted(_SHAPES))
def test_the_listing_and_the_viewer_fold_agree(shape: str, tmp_path: Path) -> None:
    listing, viewer = _shared_facts(_SHAPES[shape], tmp_path)
    assert listing == viewer, shape


def test_the_folds_agree_on_the_golden_log(tmp_path: Path) -> None:
    from agent6.viewmodel.tail import tail_events

    listing, viewer = _shared_facts(list(tail_events(_GOLDEN, follow=False)), tmp_path)
    assert listing == viewer
