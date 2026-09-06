# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The in-loop review gate: how the loop schedules and honours panel verdicts.

The panel itself (grounding, decision modes, dedup) is pinned in
test_review_panel.py; here the LOOP's plumbing is the unit -- a NEEDS-WORK
verdict revokes finish_session and injects the findings, the rejection cap
disarms the gate, trigger "off" never runs a panel, and the periodic /
on_verify_fail triggers fire only on their schedule. The panel is stubbed at
the `_run_review_panel` seam."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

from agent6.providers import ProviderResponse
from agent6.tools.results import ExecResult, RawResult
from agent6.workflows._conversation import Conversation
from agent6.workflows._review import CritiqueResult
from agent6.workflows.loop import Workflow

# The `[git]` surface the loop reads: the checkpoint message and the commit
# identity (`_commit_identity`), empty as a real Config carries it unset.
_GIT_STUB = SimpleNamespace(
    commit=SimpleNamespace(
        checkpoint=SimpleNamespace(message="agent6"), name="", email="", trailer=""
    )
)


def _silent(_msg: str) -> None:
    return None


def _wf(**kw: Any) -> Workflow:
    defaults: dict[str, Any] = {
        "root": Path("/tmp"),
        "config": MagicMock(
            git=_GIT_STUB,
            prompt=MagicMock(system_prompt_file=""),
            workflow=MagicMock(verify_command=(), verify_when="never", verify_retries=2),
        ),
        "provider": MagicMock(),
        "dispatcher": MagicMock(),
        "logger": _silent,
        "provider_retry_delay_s": 0.01,
        "review_seats": [MagicMock()],
    }
    defaults.update(kw)
    return Workflow(**defaults)


def _resp(text: str) -> ProviderResponse:
    return ProviderResponse(
        text=text,
        tool_uses=(),
        stop_reason="end_turn",
        input_tokens=10,
        output_tokens=20,
        cache_read_tokens=0,
        cache_creation_tokens=0,
    )


def _finish_tool_use(tool_id: str, summary: str) -> dict[str, Any]:
    return {
        "type": "tool_use",
        "id": tool_id,
        "name": "finish_session",
        "input": {"summary": summary},
    }


def _resp_with_tool_use(text: str, tool_use: dict[str, Any]) -> ProviderResponse:
    blocks: list[dict[str, Any]] = [{"type": "text", "text": text}] if text else []
    blocks.append({"type": "tool_use", **tool_use})
    return ProviderResponse(
        text=text,
        tool_uses=(tool_use,),
        stop_reason="tool_use",
        input_tokens=10,
        output_tokens=20,
        cache_read_tokens=0,
        cache_creation_tokens=0,
        raw={"content": blocks},
    )


class _PanelScript:
    """A scripted `_run_review_panel` stand-in: pops verdicts in order and
    counts how often the loop consulted the panel."""

    def __init__(self, verdicts: list[CritiqueResult | None]) -> None:
        self.verdicts = list(verdicts)
        self.calls = 0

    def __call__(self, _state: Any, *, trigger: str, iteration: int) -> Any:
        self.calls += 1
        return self.verdicts.pop(0) if self.verdicts else None


_MSGS: list[dict[str, Any]] = [
    {"role": "user", "content": [{"type": "text", "text": "TASK:\nfix it\n\nBegin."}]}
]


def test_before_finish_panel_revokes_finish_and_injects_findings() -> None:
    """A NEEDS-WORK panel verdict on finish_session suppresses the finish
    (the tool_result still returns so the call is not half-applied) and the
    findings ride into the next user turn under [review]."""
    worker = MagicMock()
    worker.call.side_effect = [
        _resp_with_tool_use("attempting to finish", _finish_tool_use("tu1", "wrap up")),
        _resp("ok, looks good"),
        _resp("ok, looks good"),
        _resp("ok, looks good"),
    ]
    dispatcher = MagicMock()
    dispatcher.dispatch.return_value = RawResult({"ok": True})
    panel = _PanelScript(
        [
            CritiqueResult(text="* still TODOs left", satisfied=False),
            CritiqueResult(text="* now fine", satisfied=True),
        ]
    )
    wf = _wf(provider=worker, dispatcher=dispatcher, review_trigger="before_finish")
    with patch.object(Workflow, "_run_review_panel", panel):
        conversation = Conversation.from_wire(_MSGS)
        result = wf._drive_loop(  # pyright: ignore[reportPrivateUsage]
            system="S",
            conversation=conversation,
            tool_calls=0,
            start_iteration=1,
            root_task_id=None,
            original_task="t",
        )
    assert result.iterations == 4
    assert result.reason == "silent_finish"
    assert panel.calls == 2
    iter1_user_msg = conversation.to_wire()[2]
    assert iter1_user_msg["role"] == "user"
    blocks = [b for b in iter1_user_msg["content"] if b.get("type") == "text"]
    assert any("[review]" in b["text"] for b in blocks)
    assert any("TODOs" in b["text"] for b in blocks)


def test_before_finish_panel_satisfied_accepts_finish() -> None:
    worker = MagicMock()
    worker.call.return_value = _resp_with_tool_use(
        "wrapping up", _finish_tool_use("tu1", "all done")
    )
    dispatcher = MagicMock()
    dispatcher.dispatch.return_value = RawResult({"ok": True})
    panel = _PanelScript([CritiqueResult(text="* clean", satisfied=True)])
    wf = _wf(provider=worker, dispatcher=dispatcher, review_trigger="before_finish")
    with patch.object(Workflow, "_run_review_panel", panel):
        result = wf._drive_loop(  # pyright: ignore[reportPrivateUsage]
            system="S",
            conversation=Conversation.from_wire(_MSGS),
            tool_calls=0,
            start_iteration=1,
            root_task_id=None,
            original_task="t",
        )
    assert result.completed is True
    assert result.reason == "finish_session"
    assert result.iterations == 1
    assert panel.calls == 1


def test_before_finish_rejection_cap_lets_finish_through() -> None:
    """After max_consecutive_review_rejections back-to-back rejections the
    finish goes through (findings still injected) so the worker cannot
    bounce forever."""
    worker = MagicMock()
    worker.call.return_value = _resp_with_tool_use("finishing", _finish_tool_use("tu1", "done"))
    dispatcher = MagicMock()
    dispatcher.dispatch.return_value = RawResult({"ok": True})
    panel = _PanelScript([CritiqueResult(text="* nope", satisfied=False)] * 3)
    wf = _wf(
        provider=worker,
        dispatcher=dispatcher,
        review_trigger="before_finish",
        max_consecutive_review_rejections=2,
    )
    with patch.object(Workflow, "_run_review_panel", panel):
        result = wf._drive_loop(  # pyright: ignore[reportPrivateUsage]
            system="S",
            conversation=Conversation.from_wire(_MSGS),
            tool_calls=0,
            start_iteration=1,
            root_task_id=None,
            original_task="t",
        )
    assert result.completed is True
    assert result.reason == "finish_session"
    assert panel.calls == 3  # two rejections, then the cap admits the third


def test_trigger_off_never_runs_a_panel() -> None:
    worker = MagicMock()
    worker.call.return_value = _resp_with_tool_use("done", _finish_tool_use("tu1", "d"))
    dispatcher = MagicMock()
    dispatcher.dispatch.return_value = RawResult({"ok": True})
    panel = _PanelScript([])
    wf = _wf(provider=worker, dispatcher=dispatcher, review_trigger="off")
    with patch.object(Workflow, "_run_review_panel", panel):
        result = wf._drive_loop(  # pyright: ignore[reportPrivateUsage]
            system="S",
            conversation=Conversation.from_wire(_MSGS),
            tool_calls=0,
            start_iteration=1,
            root_task_id=None,
            original_task="t",
        )
    assert result.reason == "finish_session"
    assert panel.calls == 0


def test_silent_finish_panel_revokes_and_continues() -> None:
    """A prose-only finish is also gated: rejected once, the run continues
    with the findings visible; a later pass exits."""
    worker = MagicMock()
    worker.call.side_effect = [
        _resp_with_tool_use(
            "edit",
            {
                "type": "tool_use",
                "id": "t1",
                "name": "apply_edit",
                "input": {"path": "a", "edits": []},
            },
        ),
        _resp("i think we are done"),
        _resp("still done"),
    ]
    dispatcher = MagicMock()
    dispatcher.dispatch.return_value = RawResult({"ok": True})
    panel = _PanelScript(
        [
            CritiqueResult(text="* not yet", satisfied=False),
            CritiqueResult(text="* fine now", satisfied=True),
        ]
    )
    wf = _wf(provider=worker, dispatcher=dispatcher, review_trigger="before_finish")
    with patch.object(Workflow, "_run_review_panel", panel):
        conversation = Conversation.from_wire(_MSGS)
        result = wf._drive_loop(  # pyright: ignore[reportPrivateUsage]
            system="S",
            conversation=conversation,
            tool_calls=0,
            start_iteration=1,
            root_task_id=None,
            original_task="t",
        )
    assert result.reason == "silent_finish"
    assert panel.calls == 2
    wire = conversation.to_wire()
    joined = " ".join(b.get("text", "") for m in wire for b in m["content"] if isinstance(b, dict))
    assert "[review]" in joined and "not yet" in joined


def _verify_pass_tool_use(tu_id: str) -> dict[str, Any]:
    return {
        "type": "tool_use",
        "id": tu_id,
        "name": "run_verify_command",
        "input": {},
    }


def _exec(returncode: int, stderr: str = "") -> ExecResult:
    return ExecResult(
        returncode=returncode, stdout="", stderr=stderr, duration_s=0.0, exec_failed=False
    )


def test_periodic_panel_fires_every_n_iterations() -> None:
    """review_trigger=periodic with review_period=2 runs the panel on iters 2
    and 4 only; the iter-5 finish_session is NOT gated under periodic."""
    worker = MagicMock()
    worker.call.side_effect = [
        _resp_with_tool_use("t1", _verify_pass_tool_use("v1")),
        _resp_with_tool_use("t2", _verify_pass_tool_use("v2")),
        _resp_with_tool_use("t3", _verify_pass_tool_use("v3")),
        _resp_with_tool_use("t4", _verify_pass_tool_use("v4")),
        _resp_with_tool_use("done", _finish_tool_use("f", "summary")),
    ]
    dispatcher = MagicMock()
    dispatcher.dispatch.return_value = _exec(0)
    panel = _PanelScript([CritiqueResult(text="* fine", satisfied=True)] * 2)
    wf = _wf(provider=worker, dispatcher=dispatcher, review_trigger="periodic", review_period=2)
    # Each verify pass commits real progress (the normal success path), so the
    # verify-settled detector stays dormant and all 5 iterations run.
    with (
        patch.object(Workflow, "_run_review_panel", panel),
        patch("agent6.workflows.loop.chain_commit", return_value="sha"),
    ):
        result = wf._drive_loop(  # pyright: ignore[reportPrivateUsage]
            system="S",
            conversation=Conversation.from_wire(_MSGS),
            tool_calls=0,
            start_iteration=1,
            root_task_id=None,
            original_task="t",
        )
    assert result.iterations == 5
    assert result.reason == "finish_session"
    assert panel.calls == 2


def test_periodic_panel_injects_text_into_next_user_msg() -> None:
    """Periodic findings are advisory: injected under [review] even when the
    verdict is satisfied."""
    worker = MagicMock()
    worker.call.side_effect = [
        _resp_with_tool_use("t1", _verify_pass_tool_use("v1")),
        _resp_with_tool_use("done", _finish_tool_use("f", "ok")),
    ]
    dispatcher = MagicMock()
    dispatcher.dispatch.return_value = _exec(0)
    panel = _PanelScript([CritiqueResult(text="* CONSIDER X", satisfied=True)])
    wf = _wf(provider=worker, dispatcher=dispatcher, review_trigger="periodic", review_period=1)
    conversation = Conversation.from_wire(_MSGS)
    with patch.object(Workflow, "_run_review_panel", panel):
        wf._drive_loop(  # pyright: ignore[reportPrivateUsage]
            system="S",
            conversation=conversation,
            tool_calls=0,
            start_iteration=1,
            root_task_id=None,
            original_task="t",
        )
    iter1_user_msg = conversation.to_wire()[2]
    text_blocks = [b for b in iter1_user_msg["content"] if b.get("type") == "text"]
    assert any("[review]" in b["text"] for b in text_blocks)
    assert any("CONSIDER X" in b["text"] for b in text_blocks)


def test_on_verify_fail_panel_fires_only_on_nonzero_exit() -> None:
    """review_trigger=on_verify_fail runs the panel only on iterations where
    run_verify_command exited non-zero; passing verifies never fire it."""
    worker = MagicMock()
    worker.call.side_effect = [
        _resp_with_tool_use("t1", _verify_pass_tool_use("v1")),  # passes
        _resp_with_tool_use("t2", _verify_pass_tool_use("v2")),  # FAILS
        _resp_with_tool_use("t3", _verify_pass_tool_use("v3")),  # passes
        _resp_with_tool_use("done", _finish_tool_use("f", "ok")),
    ]
    dispatcher = MagicMock()
    dispatcher.dispatch.side_effect = [
        _exec(0),
        _exec(1, stderr="test x failed"),
        _exec(0),
        RawResult({"ok": True}),
    ]
    panel = _PanelScript([CritiqueResult(text="* hmm", satisfied=False)])
    wf = _wf(provider=worker, dispatcher=dispatcher, review_trigger="on_verify_fail")
    conversation = Conversation.from_wire(_MSGS)
    with patch.object(Workflow, "_run_review_panel", panel):
        result = wf._drive_loop(  # pyright: ignore[reportPrivateUsage]
            system="S",
            conversation=conversation,
            tool_calls=0,
            start_iteration=1,
            root_task_id=None,
            original_task="t",
        )
    assert result.iterations == 4
    assert result.reason == "finish_session"
    assert panel.calls == 1
    iter2_user_msg = conversation.to_wire()[4]
    text_blocks = [b for b in iter2_user_msg["content"] if b.get("type") == "text"]
    assert any("[review]" in b["text"] for b in text_blocks)


def test_on_verify_fail_panel_skipped_when_no_verify_call() -> None:
    """An iteration with no run_verify_command call has no failure signal, so
    on_verify_fail never consults the panel."""
    edit_tool = {"type": "tool_use", "id": "e1", "name": "list_dir", "input": {"path": "."}}
    worker = MagicMock()
    worker.call.side_effect = [
        _resp_with_tool_use("editing", edit_tool),
        _resp_with_tool_use("done", _finish_tool_use("f", "ok")),
    ]
    dispatcher = MagicMock()
    dispatcher.dispatch.return_value = RawResult({"entries": []})
    panel = _PanelScript([])
    wf = _wf(provider=worker, dispatcher=dispatcher, review_trigger="on_verify_fail")
    with patch.object(Workflow, "_run_review_panel", panel):
        wf._drive_loop(  # pyright: ignore[reportPrivateUsage]
            system="S",
            conversation=Conversation.from_wire(_MSGS),
            tool_calls=0,
            start_iteration=1,
            root_task_id=None,
            original_task="t",
        )
    assert panel.calls == 0


# ---- the settled end passes the same gates a finish_session does ------------


def _settled_state() -> Any:
    from agent6.workflows._nudges import VERIFY_SETTLED_STOP_AFTER
    from agent6.workflows.loop import LoopState

    state = LoopState(original_task="t", tool_calls=0)
    state.gateless_ever_committed = True
    state.verify_settled_idle = VERIFY_SETTLED_STOP_AFTER
    return state


def _idle_turn() -> Any:
    from agent6.workflows.loop import TurnState

    return TurnState(iteration=9, resp=MagicMock(), assistant=MagicMock())


def test_a_settled_end_is_reviewed_like_a_finish() -> None:
    """A gateless run that commits and goes idle ends "settled" without ever
    calling finish_session; the before-finish panel judges that end too: a
    rejection hands the findings to the model and restarts the idle count, an
    approval lets the end stand."""
    wf = _wf(review_trigger="before_finish")
    wf.mode = "run"
    wf.config.workflow.metric = None
    panel = _PanelScript(
        [
            CritiqueResult(text="* a test is missing", satisfied=False),
            CritiqueResult(text="* fine", satisfied=True),
        ]
    )
    with patch.object(Workflow, "_run_review_panel", panel):
        state = _settled_state()
        turn = _idle_turn()
        assert wf._turn_verify_settled(state, turn) is None  # pyright: ignore[reportPrivateUsage]
        assert turn.verify_settled_stop is False
        assert state.verify_settled_idle == 0
        assert turn.review_text is not None and "a test is missing" in turn.review_text
        state = _settled_state()
        turn = _idle_turn()
        wf._turn_verify_settled(state, turn)  # pyright: ignore[reportPrivateUsage]
        assert turn.verify_settled_stop is True
    assert panel.calls == 2


def test_a_settled_end_is_certified_by_the_harness_gate() -> None:
    """Under `verify_when = "finish"` a settled end over an unverified tree
    runs the gate like a finish would: red returns to the model with the
    output (bounded by verify_retries), green lets the end stand."""
    from agent6.config import Config

    cfg = Config.model_validate(
        {"workflow": {"verify_command": ["true"], "verify_when": "finish", "verify_retries": 1}}
    )
    dispatcher = MagicMock()
    dispatcher.command_policy.return_value = "yes"
    dispatcher.run_verify.return_value = ExecResult(
        returncode=1, stdout="1 failed", stderr="", duration_s=1.0, exec_failed=False
    )
    wf = _wf(config=cfg, dispatcher=dispatcher, review_seats=[])
    wf.mode = "run"
    state = _settled_state()
    turn = _idle_turn()
    assert wf._turn_verify_settled(state, turn) is None  # pyright: ignore[reportPrivateUsage]
    dispatcher.run_verify.assert_called_once_with()
    assert turn.verify_settled_stop is False and state.verify_settled_idle == 0
    assert state.verify_finish_retries_used == 1
    notices = [r.text for r in turn.tool_results if hasattr(r, "text")]
    assert any("[harness verify] finish" in n for n in notices)
    assert any("the next red finish ends the run" in n for n in notices)
    # The return is spent: the next settled end stands, red and all.
    state.verify_settled_idle = 6
    turn = _idle_turn()
    wf._turn_verify_settled(state, turn)  # pyright: ignore[reportPrivateUsage]
    assert turn.verify_settled_stop is True
    # A green gate certifies the end at once.
    dispatcher.run_verify.return_value = ExecResult(
        returncode=0, stdout="", stderr="", duration_s=1.0, exec_failed=False
    )
    state = _settled_state()
    turn = _idle_turn()
    wf._turn_verify_settled(state, turn)  # pyright: ignore[reportPrivateUsage]
    assert turn.verify_settled_stop is True and state.verify.green_and_untouched


def test_a_silent_finish_is_certified_and_reviewed_like_a_finish() -> None:
    """A prose turn with no tool call is an end too: under `verify_when =
    "finish"` the harness runs the gate first (red returns the worker to work
    with the output, in the conversation since there are no tool results),
    then the before-finish panel judges it."""
    from agent6.config import Config
    from agent6.workflows.loop import LoopState, TurnState

    cfg = Config.model_validate(
        {"workflow": {"verify_command": ["true"], "verify_when": "finish", "verify_retries": 1}}
    )
    dispatcher = MagicMock()
    dispatcher.command_policy.return_value = "yes"
    dispatcher.run_verify.return_value = ExecResult(
        returncode=1, stdout="2 failed", stderr="", duration_s=1.0, exec_failed=False
    )
    panel = _PanelScript([CritiqueResult(text="* fine", satisfied=True)])
    wf = _wf(config=cfg, dispatcher=dispatcher, review_trigger="before_finish")
    wf.mode = "run"
    state = LoopState(original_task="t", tool_calls=0)
    state.ever_edited = True
    state.verify.note_pass()
    state.verify.note_edit()  # green once, edited since: stale
    conv = Conversation()
    with patch.object(Workflow, "_run_review_panel", panel):
        turn = TurnState(iteration=5, resp=_resp("Done."), assistant=MagicMock())
        assert wf._handle_silent_finish("Done.", conv, state, turn) is None  # pyright: ignore[reportPrivateUsage]
        texts = [b["text"] for m in conv.to_wire() for b in m["content"] if b.get("type") == "text"]
        assert any("[harness verify] finish: verify_command exit 1" in t for t in texts)
        assert any("the next red finish ends the run" in t for t in texts)
        assert panel.calls == 0  # a red gate returns the end before the panel sits
        # The return is spent: the next silent finish stands, the panel sits and approves.
        turn = TurnState(iteration=6, resp=_resp("Done."), assistant=MagicMock())
        ended = wf._handle_silent_finish("Done.", conv, state, turn)  # pyright: ignore[reportPrivateUsage]
    assert ended is not None and ended.reason == "silent_finish"
    assert ended.verified == "failed"
    assert panel.calls == 1
