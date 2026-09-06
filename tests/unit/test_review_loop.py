# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""In-loop review-panel wiring: the panel fires at before_finish, grounds against
the run diff, and gates the finish only under a gating decision (no network)."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

from agent6.tools.results import RawResult
from agent6.workflows._conversation import Conversation
from agent6.workflows._review import ReviewSeat
from agent6.workflows.loop import Workflow
from tests.unit.test_review_gate import (
    _finish_tool_use,  # pyright: ignore[reportPrivateUsage]
    _resp,  # pyright: ignore[reportPrivateUsage]
    _resp_with_tool_use,  # pyright: ignore[reportPrivateUsage]
    _wf,  # pyright: ignore[reportPrivateUsage]
)

# A diff touching foo.py new line 11, so a "foo.py:11" citation grounds.
_DIFF = """\
--- a/foo.py
+++ b/foo.py
@@ -10,1 +10,2 @@ def f():
     x = 1
+    API_KEY = "sk-secret"
"""
_BLOCK = (
    '{"verdict":"block","summary":"leak","findings":[{"category":"security",'
    '"severity":"block","file_line":"foo.py:11","title":"hardcoded key","detail":"x"}]}'
)
_PASS = '{"verdict":"pass","summary":"ok","findings":[]}'
_NONGATING = (  # grounded but a non-block-eligible category -> downgraded, never gates
    '{"verdict":"block","summary":"style","findings":[{"category":"style",'
    '"severity":"block","file_line":"foo.py:11","title":"rename it","detail":"x"}]}'
)


def _seat(provider: Any, persona: str = "security", model: str = "m1") -> ReviewSeat:
    return ReviewSeat(persona=persona, model=model, provider=provider)


def _disp() -> MagicMock:
    d = MagicMock()
    d.dispatch.return_value = RawResult(
        {"ok": True}
    )  # JSON-serializable (finish_session is dispatched)
    return d


def _begin() -> list[dict[str, Any]]:
    return [{"role": "user", "content": [{"type": "text", "text": "TASK:\ngo\n\nBegin."}]}]


def _drive(wf: Workflow, messages: list[dict[str, Any]]) -> Any:
    conversation = Conversation.from_wire(messages)
    with patch.object(Workflow, "_run_diff", return_value=_DIFF):
        result = wf._drive_loop(  # pyright: ignore[reportPrivateUsage]
            system="S",
            conversation=conversation,
            tool_calls=0,
            start_iteration=1,
            root_task_id=None,
            original_task="t",
        )
    # The callers' fixtures still read the final history via *messages*.
    messages[:] = conversation.to_wire()
    return result


def test_has_reviewer_true_with_seats() -> None:
    wf = _wf(review_seats=[_seat(MagicMock())])
    assert wf._has_reviewer() is True  # pyright: ignore[reportPrivateUsage]
    assert _wf(review_seats=[])._has_reviewer() is False  # pyright: ignore[reportPrivateUsage]


def test_panel_blocks_finish_under_veto_then_accepts() -> None:
    """A grounded security block under veto revokes the first finish_session; once the
    seat passes, the second finish_session is accepted."""
    worker = MagicMock()
    worker.call.side_effect = [
        _resp_with_tool_use("f1", _finish_tool_use("a", "done")),
        _resp_with_tool_use("f2", _finish_tool_use("b", "done")),
    ]
    seat_provider = MagicMock()
    seat_provider.call.side_effect = [_resp(_BLOCK), _resp(_PASS)]
    wf = _wf(
        provider=worker,
        dispatcher=_disp(),
        review_seats=[_seat(seat_provider)],
        review_decision="veto",
        review_trigger="before_finish",
        base_sha="b",
    )
    messages = _begin()
    result = _drive(wf, messages)
    assert result.reason == "finish_session" and result.iterations == 2
    assert seat_provider.call.call_count == 2  # panel ran at each finish attempt
    # the first (blocked) finish injected the findings for the worker to address
    injected = "".join(
        b.get("text", "") for m in messages for b in (m.get("content") or []) if isinstance(b, dict)
    )
    assert "rejected your finish_session" in injected and "hardcoded key" in injected


def test_panel_skipped_when_budget_fraction_low() -> None:
    """When remaining budget < review_budget_fraction the panel is SKIPPED
    (approve-and-proceed): reviewing costs most when budget is scarcest. The
    seat that WOULD block is never called, and finish_session is accepted. This is
    the only behavioural test of review_budget_fraction (previously dead config)."""
    worker = MagicMock()
    worker.call.return_value = _resp_with_tool_use("f1", _finish_tool_use("a", "done"))
    seat_provider = MagicMock()
    seat_provider.call.side_effect = [_resp(_BLOCK)]  # would block IF the panel ran
    wf = _wf(
        provider=worker,
        dispatcher=_disp(),
        review_seats=[_seat(seat_provider)],
        review_decision="veto",
        review_trigger="before_finish",
        base_sha="b",
    )  # review_budget_fraction defaults to 0.25
    with patch.object(Workflow, "_budget_fraction_remaining", return_value=0.10):
        result = _drive(wf, _begin())
    assert seat_provider.call.call_count == 0  # panel skipped, not run
    assert result.reason == "finish_session" and result.iterations == 1


def test_panel_advisory_does_not_block_finish() -> None:
    """The SAME grounded block under advisory does not gate -> finish on iter 1."""
    worker = MagicMock()
    worker.call.side_effect = [_resp_with_tool_use("f1", _finish_tool_use("a", "done"))]
    seat_provider = MagicMock()
    seat_provider.call.return_value = _resp(_BLOCK)
    wf = _wf(
        provider=worker,
        dispatcher=_disp(),
        review_seats=[_seat(seat_provider)],
        review_decision="advisory",
        review_trigger="before_finish",
        base_sha="b",
    )
    result = _drive(wf, _begin())
    assert result.reason == "finish_session" and result.iterations == 1
    assert seat_provider.call.call_count == 1  # panel still ran (events), just didn't gate


def test_panel_does_not_block_on_nongating_category_even_under_veto() -> None:
    """A grounded 'style' block is downgraded by the aggregator, so veto can't
    stall on taste -- finish accepted on iter 1."""
    worker = MagicMock()
    worker.call.side_effect = [_resp_with_tool_use("f1", _finish_tool_use("a", "done"))]
    seat_provider = MagicMock()
    seat_provider.call.return_value = _resp(_NONGATING)
    wf = _wf(
        provider=worker,
        dispatcher=_disp(),
        review_seats=[_seat(seat_provider)],
        review_decision="veto",
        review_trigger="before_finish",
        base_sha="b",
    )
    result = _drive(wf, _begin())
    assert result.reason == "finish_session" and result.iterations == 1


def test_disarm_after_max_total_rejections_lets_finish_through() -> None:
    """Once review_rejections_total hits the cap, the gate disarms to advisory so
    a persistently-blocking panel can't stall the run forever."""
    worker = MagicMock()
    # the worker keeps trying to finish; the seat keeps blocking
    worker.call.side_effect = [
        _resp_with_tool_use(f"f{i}", _finish_tool_use(str(i), "done")) for i in range(6)
    ]
    seat_provider = MagicMock()
    seat_provider.call.return_value = _resp(_BLOCK)
    wf = _wf(
        provider=worker,
        dispatcher=_disp(),
        review_seats=[_seat(seat_provider)],
        review_decision="veto",
        review_trigger="before_finish",
        max_consecutive_review_rejections=0,  # isolate the per-run total disarm
        review_max_total_rejections=2,
        base_sha="b",
    )
    result = _drive(wf, _begin())
    # blocks on rejections 1 and 2, disarms on the 3rd attempt -> finish accepted
    assert result.reason == "finish_session" and result.iterations == 3


def test_in_loop_panel_all_abstain_names_the_abstention() -> None:
    """The in-loop panel is the copy that spends the run's budget and feeds the
    critique to the model. An all-abstain panel must name the abstention, not
    'No blocking findings.' (the CLI verdict was fixed for the same reason).
    Uses the shared panel_is_inconclusive/inconclusive_note owner; the gate
    still lets the finish through (a panel never deadlocks a run)."""
    import agent6.workflows.loop as loop_mod
    from agent6.workflows._panel import PanelResult, ReviewVerdict
    from agent6.workflows.loop import LoopState

    abstain = ReviewVerdict(seat="s", model="m", verdict="pass", error="output hit the cap")
    res = PanelResult(
        panel_id="p",
        decision="advisory",
        blocked=False,
        merged_findings=(),
        per_seat=(abstain, abstain, abstain),
        n_block=0,
        n_abstain=3,
    )
    wf = _wf(
        provider=MagicMock(),
        dispatcher=_disp(),
        review_seats=[_seat(MagicMock())],
        review_decision="advisory",
        base_sha="b",
    )
    state = LoopState(original_task="t", tool_calls=0)
    with (
        patch.object(loop_mod.Workflow, "_run_diff", return_value=_DIFF),
        patch.object(loop_mod, "run_panel", return_value=res),
    ):
        out = wf._run_review_panel(  # pyright: ignore[reportPrivateUsage]
            state, trigger="periodic", iteration=1
        )
    assert out is not None
    assert "inconclusive" in out.text and "abstained" in out.text
    assert "No blocking findings" not in out.text
    assert out.satisfied is True  # never deadlocks the run
