# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The verify finish gate: finish_session can never report 'passed' over a red or
stale verify, and `[workflow].verify_when` has the harness run the gate itself
at finish (or every step), returning a red finish `verify_retries` times. Both
ground on _tree_is_verify_green."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal
from unittest.mock import MagicMock

import pytest

from agent6.config import Config
from agent6.viewmodel.listing import status_word
from agent6.workflows._verify_verdict import VerifyVerdict
from agent6.workflows.loop import (
    LoopState,
    TurnState,
    Workflow,
)


def _wf(
    *,
    verify: bool,
    mode: Literal["run", "plan", "ask", "agent"] = "run",
    root: Path = Path("/tmp"),
) -> Workflow:
    data: dict[str, Any] = {"workflow": {"verify_command": ["true"]}} if verify else {}
    return Workflow(
        root=root,
        config=Config.model_validate(data),
        provider=MagicMock(),
        dispatcher=MagicMock(),
        logger=lambda _m: None,
        mode=mode,
    )


def _green(wf: Workflow, **verdict_kw: Any) -> bool | None:
    state = LoopState(original_task="t", tool_calls=0, verify=VerifyVerdict(**verdict_kw))
    return wf._tree_is_verify_green(state)  # pyright: ignore[reportPrivateUsage]


def test_no_verify_command_is_not_gated() -> None:
    # Nothing to gate on -> None -> finish is always an honest pass.
    assert _green(_wf(verify=False), last_ok=None) is None
    assert _green(_wf(verify=False), last_ok=False) is None


def test_green_only_when_last_verify_passed_and_tree_unedited() -> None:
    wf = _wf(verify=True)
    assert _green(wf, last_ok=True, edited_since=False) is True
    # Never verified, or last verify failed -> not green.
    assert _green(wf, last_ok=None) is False
    assert _green(wf, last_ok=False) is False
    # A green verify that has since been edited over is stale -> not green.
    assert _green(wf, last_ok=True, edited_since=True) is False


def test_the_harness_gate_defaults_to_finish_with_two_returns() -> None:
    wf = Config().workflow
    assert (wf.verify_when, wf.verify_retries) == ("finish", 2)


def _verified(wf: Workflow, **verdict_kw: Any) -> str:
    state = LoopState(original_task="t", tool_calls=0, verify=VerifyVerdict(**verdict_kw))
    return wf._verification(state)  # pyright: ignore[reportPrivateUsage]


def test_verification_carries_the_same_verdict_the_event_does() -> None:
    """SessionResult.verified is the app layer's copy of session.end.all_passed's
    grounding, so exit code, auto-merge, and the notify hook read the verify
    truth instead of `completed` (true for any deliberate finish).

    "failed" means someone OBSERVED a red gate. Folding "no verify ran this
    leg" into it printed "the gate is red" over a gate that never ran and sent
    the operator to bisect the base commit for a failure that never happened;
    those finishes are "unverified"."""
    assert _verified(_wf(verify=True), last_ok=True, edited_since=False) == "passed"
    assert _verified(_wf(verify=True), last_ok=False) == "failed"
    # Red, then edited without re-verifying: the red observation stands.
    assert _verified(_wf(verify=True), last_ok=False, edited_since=True) == "failed"
    # Green but edited since: no observation covers the final tree.
    assert _verified(_wf(verify=True), last_ok=True, edited_since=True) == "unverified"
    # Never observed this leg: not red, not green.
    assert _verified(_wf(verify=True), last_ok=None) == "unverified"
    # Gateless: nothing ever gated this run, so there is no verdict to claim.
    assert _verified(_wf(verify=False), last_ok=None) == "not_applicable"


def test_a_gateless_end_and_its_verdict_agree() -> None:
    """`_emit_run_end_grounded` turned the gateless None into all_passed=True
    (`is not False`) while `_verification` mapped the same None to
    not_applicable: the run read "passed" on every surface though nothing ever
    gated it, and the docstring claimed the two could never disagree.
    all_passed=True needs an OBSERVED green; the gateless end carries the
    tri-state's None on the wire (words as "finished", never "passed" or
    "failed"), agreeing with the not_applicable verdict."""
    emitted: list[dict[str, Any]] = []

    def _capture(_type: str, **fields: Any) -> None:
        emitted.append(fields)

    cases: tuple[tuple[bool, VerifyVerdict, bool | None, str], ...] = (
        (False, VerifyVerdict(last_ok=None), None, "not_applicable"),
        (True, VerifyVerdict(last_ok=True, edited_since=False), True, "passed"),
        (True, VerifyVerdict(last_ok=False), False, "failed"),
    )
    for verify, verify_verdict, all_passed, verdict in cases:
        wf = _wf(verify=verify)
        wf.events = MagicMock(emit=_capture)
        wf.events.emit = _capture  # type: ignore[method-assign]
        state = LoopState(original_task="t", tool_calls=0, verify=verify_verdict)
        emitted.clear()
        wf._emit_run_end_grounded(  # pyright: ignore[reportPrivateUsage]
            reason="finish_session", iteration=1, state=state
        )
        assert emitted and emitted[-1]["all_passed"] is all_passed
        assert wf._verification(state) == verdict  # pyright: ignore[reportPrivateUsage]
        # The invariant the docstring promises: the event and the result agree.
        assert (emitted[-1]["all_passed"] is True) == (
            wf._verification(state) == "passed"  # pyright: ignore[reportPrivateUsage]
        )


def test_the_end_event_carries_whether_the_certifying_gate_ran_scoped() -> None:
    """A scoped green is a pass with a qualifier: `session.end` carries
    `scoped`, and every surface words it "passed · scoped gate" through the
    one status_word. A run that ended over a red or a stale scoped gate is
    "finished" as before: the qualifier belongs to a pass only."""
    emitted: list[dict[str, Any]] = []

    def _capture(_type: str, **fields: Any) -> None:
        emitted.append(fields)

    wf = _wf(verify=True)
    wf.events = MagicMock(emit=_capture)
    wf.events.emit = _capture  # type: ignore[method-assign]
    for scoped in (True, False):
        state = LoopState(
            original_task="t", tool_calls=0, verify=VerifyVerdict(last_ok=True, scoped=scoped)
        )
        wf._emit_run_end_grounded(  # pyright: ignore[reportPrivateUsage]
            reason="finish_session", iteration=1, state=state
        )
        assert (emitted[-1]["all_passed"], emitted[-1]["scoped"]) == (True, scoped)
    assert status_word(
        finished=True, all_passed=True, end_reason="finish_session", scoped=True
    ) == ("passed", "scoped gate")
    assert status_word(
        finished=True, all_passed=False, end_reason="finish_session", scoped=True
    ) == ("finished", "")


def test_plan_and_ask_are_never_gated_on_verify() -> None:
    """plan/ask end clean whatever the tree looks like -- finish_planning and
    the ask answer both emit session.end all_passed=True -- so they have no verify
    verdict to report. Reporting one made `agent6 plan` exit 4 (preflight
    INFERS a verify command for plan, and plan never runs it, so the tree read
    as red) while its own journal and every listing said passed."""
    for mode in ("plan", "ask"):
        assert _verified(_wf(verify=True, mode=mode), last_ok=None) == "not_applicable"
        assert _verified(_wf(verify=True, mode=mode), last_ok=False) == "not_applicable"


def test_a_command_that_dirties_the_tree_invalidates_the_verify_pass(tmp_path: Path) -> None:
    """A green verify must not survive a run_command that changed the tree:
    edited_since_verify was set only by apply_edit/apply_patch, so a model
    could verify green, then mutate through run_command (or an MCP tool) and
    still finish reporting verified="passed" -- defeating exit 4,
    the finish certification, and the auto-merge gate together. Grounded on
    git, so a read-only command keeps the pass it had."""
    import subprocess as sp

    sp.run(["git", "init", "-q", "-b", "main"], cwd=tmp_path, check=True)
    sp.run(["git", "config", "user.email", "t@example.com"], cwd=tmp_path, check=True)
    sp.run(["git", "config", "user.name", "t"], cwd=tmp_path, check=True)
    (tmp_path / "a.txt").write_text("x\n", encoding="utf-8")
    sp.run(["git", "add", "a.txt"], cwd=tmp_path, check=True)
    sp.run(["git", "commit", "-q", "-m", "seed"], cwd=tmp_path, check=True)

    dirty = _wf(verify=True, root=tmp_path)._left_the_tree_dirty  # pyright: ignore[reportPrivateUsage]

    assert dirty("run_command") is False  # clean tree: a read-only probe costs nothing
    assert dirty("read_file") is False  # never asked of in-process read tools
    (tmp_path / "a.txt").write_text("mutated\n", encoding="utf-8")
    assert dirty("run_command") is True
    assert dirty("mcp__srv__write") is True
    # verify/metric are the operator's own gates; their caches must not
    # invalidate the pass they just produced.
    assert dirty("run_verify_command") is False
    assert dirty("run_metric_command") is False


def _git_seed(tmp_path: Path) -> str:
    import subprocess as sp

    sp.run(["git", "init", "-q", "-b", "main"], cwd=tmp_path, check=True)
    (tmp_path / "a.txt").write_text("x\n", encoding="utf-8")
    sp.run(["git", "add", "a.txt"], cwd=tmp_path, check=True)
    sp.run(["git", "commit", "-q", "-m", "seed"], cwd=tmp_path, check=True)
    out = sp.run(
        ["git", "rev-parse", "HEAD"], cwd=tmp_path, check=True, capture_output=True, text=True
    )
    return out.stdout.strip()


def _snap(**kw: Any) -> Any:
    from agent6.workflows._session_state import SessionSnapshot

    base: dict[str, Any] = {
        "system": "s",
        "messages": [],
        "tool_calls": 0,
        "next_iteration": 1,
        "root_task_id": None,
        "original_task": "t",
        "verify_command": ("true",),
    }
    return SessionSnapshot(**{**base, **kw})


def _resumed_state(wf: Workflow, snap: Any) -> LoopState:
    from agent6.workflows._conversation import Conversation

    state = LoopState(original_task="t", tool_calls=0)
    wf._seed_carryover(state, Conversation.from_wire([]), snap)  # pyright: ignore[reportPrivateUsage]
    return state


def test_a_resumed_leg_carries_the_verify_verdict_over_an_unmoved_tree(tmp_path: Path) -> None:
    """last_verify_ok was leg-scoped, so resuming a green-finished run and
    finishing without edits read "unverified" (previously: exit 4 claiming a
    red gate) over the very tree the gate approved. The verdict carries when
    HEAD is the snapshot's and the worktree is clean; baseline_ok is about the
    base commit, which resume never moves, so it always carries."""
    head = _git_seed(tmp_path)
    wf = _wf(verify=True, root=tmp_path)
    snap = _snap(head_sha=head, last_verify_ok=True, edited_since_verify=False, baseline_ok=False)
    state = _resumed_state(wf, snap)
    assert state.verify.last_ok is True
    assert state.verify.edited_since is False
    assert state.verify.baseline_ok is False
    assert wf._verification(state) == "passed"  # pyright: ignore[reportPrivateUsage]
    # A red observation carries the same way: the resumed leg stays answerable.
    red = _resumed_state(wf, _snap(head_sha=head, last_verify_ok=False))
    assert red.verify.last_ok is False


def test_the_carried_verdict_is_dropped_when_the_tree_moved(tmp_path: Path) -> None:
    """An operator commit or edit between legs means no observation covers
    THIS tree: the leg starts unobserved (fails closed, like the baseline
    probe), never wrongly green or red."""
    import subprocess as sp

    head = _git_seed(tmp_path)
    wf = _wf(verify=True, root=tmp_path)
    green = {"last_verify_ok": True, "edited_since_verify": False, "baseline_ok": True}

    # Worktree dirtied between legs.
    (tmp_path / "a.txt").write_text("edited\n", encoding="utf-8")
    state = _resumed_state(wf, _snap(head_sha=head, **green))
    assert state.verify.last_ok is None
    assert state.verify.baseline_ok is True  # the base commit did not move

    # HEAD moved forward between legs.
    sp.run(["git", "commit", "-qam", "operator work"], cwd=tmp_path, check=True)
    assert _resumed_state(wf, _snap(head_sha=head, **green)).verify.last_ok is None

    # No head recorded at write time: nothing to compare against.
    assert _resumed_state(wf, _snap(head_sha="", **green)).verify.last_ok is None


def test_a_resumed_leg_carries_the_scoped_gate(tmp_path: Path) -> None:
    """The full gate overran once: the resumed leg goes straight to the
    scoped form instead of burning the timeout again. Carried whatever the
    tree did between legs (the fact is about the suite, not the tree), while
    the verdict itself still drops when the tree moved."""
    _git_seed(tmp_path)
    wf = _wf(verify=True, root=tmp_path)
    (tmp_path / "a.txt").write_text("edited\n", encoding="utf-8")
    state = _resumed_state(wf, _snap(verify_scoped=True, last_verify_ok=True))
    assert state.verify.scoped is True
    assert state.verify.last_ok is None


# ---- the harness-run gate (`[workflow].verify_when`) -------------------------


def _exec(rc: int, out: str = "") -> Any:
    from agent6.tools.results import ExecResult

    return ExecResult(returncode=rc, stdout=out, stderr="", duration_s=1.0, exec_failed=False)


def _harness_wf(when: str, retries: int = 2, *, policy: str = "yes") -> tuple[Workflow, MagicMock]:
    """A run-mode loop over a gate, and the mock dispatcher that owns `run_verify`."""
    data: dict[str, Any] = {
        "workflow": {"verify_command": ["true"], "verify_when": when, "verify_retries": retries}
    }
    dispatcher = MagicMock()
    dispatcher.command_policy.return_value = policy
    wf = Workflow(
        root=Path("/tmp"),
        config=Config.model_validate(data),
        provider=MagicMock(),
        dispatcher=dispatcher,
        logger=lambda _m: None,
        mode="run",
    )
    return wf, dispatcher


def _turn(*, finishing: bool = False, edited: bool = False) -> Any:
    from agent6.workflows.loop import TurnState

    turn = TurnState(iteration=3, resp=MagicMock(), assistant=MagicMock())
    if finishing:
        turn.finish_signal = "done"
        turn.finish_kind = "finish_session"
    if edited:
        turn.edited = True
        turn.edit_since_verify_pass = True
    return turn


def _notices(turn: Any) -> list[str]:
    from agent6.workflows._conversation import Notice

    return [r.text for r in turn.tool_results if isinstance(r, Notice)]


def test_finish_mode_runs_the_gate_when_a_finish_arrives_over_an_unverified_tree() -> None:
    """`verify_when = "finish"`: the harness runs the gate on finish_session and the
    model sees the verdict; a green run certifies the tree (the verdict's own
    bookkeeping, so the finish gate sees green and auto-commit sees a pass)."""
    wf, dispatcher = _harness_wf("finish")
    dispatcher.run_verify.return_value = _exec(0, "3 passed")
    state = LoopState(original_task="t", tool_calls=0)
    turn = _turn(finishing=True, edited=True)

    assert wf._turn_harness_verify(state, turn) is None  # pyright: ignore[reportPrivateUsage]

    dispatcher.run_verify.assert_called_once_with(extra_argv=())
    assert state.verify.green_and_untouched and turn.verify_just_passed
    assert _notices(turn) == ["[harness verify] finish: verify_command passed (1s).\n3 passed"]
    wf._gate_verify_finish(state, turn)  # pyright: ignore[reportPrivateUsage]
    assert turn.finish_signal == "done"


def test_a_red_finish_certification_returns_to_the_model_verify_retries_times() -> None:
    """A red gate at finish returns the finish with the output `verify_retries`
    times; the next red finish stands (reported finished, never passed)."""
    wf, dispatcher = _harness_wf("finish", retries=2)
    dispatcher.run_verify.return_value = _exec(1, "1 failed")
    state = LoopState(original_task="t", tool_calls=0)
    seen: list[str | None] = []
    notices: list[str] = []
    for _ in range(3):
        turn = _turn(finishing=True, edited=True)
        wf._turn_harness_verify(state, turn)  # pyright: ignore[reportPrivateUsage]
        wf._gate_verify_finish(state, turn)  # pyright: ignore[reportPrivateUsage]
        seen.append(turn.finish_signal)
        notices.extend(_notices(turn))
    assert seen == [None, None, "done"]
    assert state.verify_finish_retries_used == 2
    assert wf._tree_is_verify_green(state) is False  # pyright: ignore[reportPrivateUsage]
    assert any("(return 1 of 2); 1 more red finish returns" in n for n in notices)
    assert any("(return 2 of 2); the next red finish ends the run" in n for n in notices)


def test_zero_retries_lets_the_first_red_finish_stand() -> None:
    wf, dispatcher = _harness_wf("finish", retries=0)
    dispatcher.run_verify.return_value = _exec(1)
    state = LoopState(original_task="t", tool_calls=0)
    turn = _turn(finishing=True, edited=True)
    wf._turn_harness_verify(state, turn)  # pyright: ignore[reportPrivateUsage]
    wf._gate_verify_finish(state, turn)  # pyright: ignore[reportPrivateUsage]
    assert turn.finish_signal == "done"
    assert wf._verification(state) == "failed"  # pyright: ignore[reportPrivateUsage]


def test_a_tree_the_model_already_certified_is_not_judged_twice() -> None:
    """Green and untouched since: the finish needs no second run. And a turn
    whose own run_verify_command judged the tree is never judged on top."""
    wf, dispatcher = _harness_wf("finish")
    state = LoopState(original_task="t", tool_calls=0)
    state.verify.note_pass()
    turn = _turn(finishing=True)
    wf._turn_harness_verify(state, turn)  # pyright: ignore[reportPrivateUsage]
    dispatcher.run_verify.assert_not_called()

    # And a RED verdict the run already holds for this tree is not re-run
    # (the finish reports the red it knows), where a green-only skip re-judged
    # it and fed the no-progress streak the one red a second time.
    wf2, dispatcher2 = _harness_wf("finish")
    state2 = LoopState(original_task="t", tool_calls=0)
    state2.verify.note_edit()
    state2.verify.note_fail("sig")  # the model's own red verify, tree untouched since
    wf2._turn_harness_verify(state2, _turn(finishing=True))  # pyright: ignore[reportPrivateUsage]
    dispatcher2.run_verify.assert_not_called()


def test_step_mode_judges_every_editing_turn_and_finish_mode_does_not() -> None:
    for when, calls in (("step", 1), ("finish", 0), ("never", 0)):
        wf, dispatcher = _harness_wf(when)
        dispatcher.run_verify.return_value = _exec(0)
        state = LoopState(original_task="t", tool_calls=0)
        wf._turn_harness_verify(state, _turn(edited=True))  # pyright: ignore[reportPrivateUsage]
        assert dispatcher.run_verify.call_count == calls, when


def test_never_mode_leaves_a_finish_over_an_unverified_tree_alone() -> None:
    """`never`: the measured model-driven shape. The harness neither runs the gate
    nor returns the finish; the end is reported finished, not passed."""
    wf, dispatcher = _harness_wf("never")
    state = LoopState(original_task="t", tool_calls=0)
    turn = _turn(finishing=True, edited=True)
    wf._turn_harness_verify(state, turn)  # pyright: ignore[reportPrivateUsage]
    wf._gate_verify_finish(state, turn)  # pyright: ignore[reportPrivateUsage]
    dispatcher.run_verify.assert_not_called()
    assert turn.finish_signal == "done"
    assert wf._verification(state) == "unverified"  # pyright: ignore[reportPrivateUsage]


def test_run_commands_no_withholds_the_gate_from_the_harness_too() -> None:
    wf, dispatcher = _harness_wf("finish", policy="no")
    state = LoopState(original_task="t", tool_calls=0)
    turn = _turn(finishing=True, edited=True)
    wf._turn_harness_verify(state, turn)  # pyright: ignore[reportPrivateUsage]
    wf._gate_verify_finish(state, turn)  # pyright: ignore[reportPrivateUsage]
    dispatcher.run_verify.assert_not_called()
    assert turn.finish_signal == "done"


def test_a_denied_gate_is_withheld_for_the_run_and_the_finish_stands() -> None:
    """Not approved under `ask` (a human's no, or the unattended auto-deny):
    the gate is withheld for the rest of the run like `run_commands = "no"`,
    the model is told so, and the finish stands unverified. Bouncing the
    finish against a denial burned every retry on a wall nobody could open
    (a live machine leg failed with its fix committed and tests green)."""
    from agent6.tools.errors import ToolDenied

    wf, dispatcher = _harness_wf("finish", retries=2)
    dispatcher.run_verify.side_effect = ToolDenied("run_verify_command not approved")
    state = LoopState(original_task="t", tool_calls=0)
    turn = _turn(finishing=True, edited=True)
    wf._turn_harness_verify(state, turn)  # pyright: ignore[reportPrivateUsage]
    assert _notices(turn) == [
        "[harness verify] finish: not run: run_verify_command not approved."
        " The gate is withheld for the rest of the run; the run ends unverified."
    ]
    wf._gate_verify_finish(state, turn)  # pyright: ignore[reportPrivateUsage]
    assert turn.finish_signal == "done"  # no bounce
    assert state.verify_finish_retries_used == 0
    assert wf._verification(state) == "unverified"  # pyright: ignore[reportPrivateUsage]

    # A later end never re-asks: the withheld gate stays withheld.
    turn2 = _turn(finishing=True, edited=True)
    wf._turn_harness_verify(state, turn2)  # pyright: ignore[reportPrivateUsage]
    assert dispatcher.run_verify.call_count == 1
    assert _notices(turn2) == []


def test_the_prompt_states_when_the_harness_runs_the_gate() -> None:
    from agent6.types import RepoSummary
    from agent6.workflows._prompt_blocks import build_system_prompt

    repo = RepoSummary(
        root=Path("/tmp"),
        branch="main",
        head_sha="0" * 40,
        file_count=0,
        top_level=(),
        agents_md="",
        recent_log="",
    )

    def block(when: str, mode: Literal["run", "plan"] = "run") -> str:
        cfg = Config.model_validate(
            {"workflow": {"verify_command": ["true"], "verify_when": when, "verify_retries": 1}}
        )
        return build_system_prompt(config=cfg, repo=repo, mode=mode, skills=None)

    assert "The harness runs it when finish_session is called" in block("finish")
    assert "returns to you 1 time(s)" in block("finish")
    assert "after every turn that edits the tree" in block("step")
    assert "The harness never runs it; only your run_verify_command calls do." in block("never")
    # plan and ask never run the gate, whatever the knob says
    assert "The harness never runs it" in block("finish", mode="plan")
    # the commit fact follows: a finish-certified run commits each editing step
    assert "commits each editing step automatically" in block("finish")
    assert "commits automatically after each passing verify" in block("never")
    assert "a passing run auto-commits the step" not in block("never")


def test_a_verify_followed_by_an_edit_in_one_turn_is_judged_again() -> None:
    """`step`: the model runs the gate green, then edits later in the same
    turn; the turn's final tree is unjudged, so the harness runs the gate.
    Self-review 2026-08-23: the turn-wide boolean skipped it."""
    wf, dispatcher = _harness_wf("step")
    dispatcher.run_verify.return_value = _exec(0)
    state = LoopState(original_task="t", tool_calls=0)
    turn = _turn(edited=True)
    turn.verify_just_passed = True  # the model's own green, then the edit
    turn.edit_since_verify_pass = True
    wf._turn_harness_verify(state, turn)  # pyright: ignore[reportPrivateUsage]
    dispatcher.run_verify.assert_called_once_with(extra_argv=())


def _scoped_wf(
    root: Path, command: list[str], *, when: str = "finish"
) -> tuple[Workflow, MagicMock]:
    """A harness-gated loop whose root holds pkg/mod.py + tests/test_mod.py."""
    for rel in ("pkg/mod.py", "tests/test_mod.py"):
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text("")
    data: dict[str, Any] = {"workflow": {"verify_command": command, "verify_when": when}}
    dispatcher = MagicMock()
    dispatcher.command_policy.return_value = "yes"
    wf = Workflow(
        root=root,
        config=Config.model_validate(data),
        provider=MagicMock(),
        dispatcher=dispatcher,
        logger=lambda _m: None,
        mode="run",
    )
    return wf, dispatcher


def _fake_diff(_self: Workflow) -> str:
    return "diff --git a/pkg/mod.py b/pkg/mod.py\n"


def test_a_timed_out_gate_reruns_scoped_to_the_nearest_tests(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """The full gate overran verify_timeout_s (exit 124): the harness re-runs
    the same pytest command scoped to the tests nearest the run's diff, the
    verdict comes from the scoped run, and the notice names the scope. Later
    gates go straight to the scoped form instead of burning the timeout again.
    Grounds the SWE-rebench broke-P2P class: big-repo legs finished over a
    gate that timed out and certified nothing."""
    monkeypatch.setattr(Workflow, "_run_diff", _fake_diff)
    wf, dispatcher = _scoped_wf(tmp_path, ["python", "-m", "pytest", "-q"])
    emitted: list[tuple[str, dict[str, Any]]] = []

    def _capture(event_type: str, **fields: Any) -> None:
        emitted.append((event_type, fields))

    wf.events = MagicMock(emit=_capture)
    dispatcher.run_verify.side_effect = [_exec(124), _exec(0)]
    state = LoopState(original_task="t", tool_calls=0)
    turn = _turn(finishing=True)
    wf._turn_harness_verify(state, turn)  # pyright: ignore[reportPrivateUsage]
    assert [c.kwargs["extra_argv"] for c in dispatcher.run_verify.call_args_list] == [
        (),
        ("tests/test_mod.py",),
    ]
    assert state.verify.scoped is True
    assert turn.verify_just_passed is True
    notice = turn.tool_results[-1].text
    assert "the gate ran scoped to the tests nearest the run's change (tests/test_mod.py)" in notice
    assert "not a full-suite pass" in notice
    assert ("loop.verify_scoped", {"paths": ["tests/test_mod.py"], "iteration": 3}) in emitted
    # The next gate skips the doomed full run.
    dispatcher.run_verify.reset_mock(side_effect=True)
    dispatcher.run_verify.return_value = _exec(0)
    state.verify.note_edit()
    turn2 = _turn(finishing=True, edited=True)
    wf._turn_harness_verify(state, turn2)  # pyright: ignore[reportPrivateUsage]
    dispatcher.run_verify.assert_called_once_with(extra_argv=("tests/test_mod.py",))


def test_a_timed_out_non_pytest_gate_stays_a_plain_timeout(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """Only pytest takes file-path selection; a make/other gate that times out
    is reported as-is, one run, no scoped rerun."""
    monkeypatch.setattr(Workflow, "_run_diff", _fake_diff)
    wf, dispatcher = _scoped_wf(tmp_path, ["make", "test"])
    dispatcher.run_verify.return_value = _exec(124)
    state = LoopState(original_task="t", tool_calls=0)
    turn = _turn(finishing=True)
    wf._turn_harness_verify(state, turn)  # pyright: ignore[reportPrivateUsage]
    dispatcher.run_verify.assert_called_once_with(extra_argv=())
    assert state.verify.scoped is False
    assert "scoped" not in turn.tool_results[-1].text


@pytest.mark.parametrize(
    "command",
    [
        ["sh", "-c", "uv run ruff check && uv run pytest"],
        ["python", "-m", "pytest", "-q", "tests"],
    ],
    ids=["sh-c-pipeline", "pytest-naming-a-path"],
)
def test_a_gate_that_cannot_take_appended_paths_stays_a_plain_timeout(
    tmp_path: Path, monkeypatch: Any, command: list[str]
) -> None:
    """A `sh -c` script binds appended paths as $0/$1 with the script
    unchanged, and `pytest tests` unions the dir with the files: the re-run
    would be the identical full command, a second timeout, and a false "ran
    scoped" notice. The substring predicate scoped both; neither scopes."""
    monkeypatch.setattr(Workflow, "_run_diff", _fake_diff)
    wf, dispatcher = _scoped_wf(tmp_path, command)
    dispatcher.run_verify.return_value = _exec(124)
    state = LoopState(original_task="t", tool_calls=0)
    turn = _turn(finishing=True)
    wf._turn_harness_verify(state, turn)  # pyright: ignore[reportPrivateUsage]
    dispatcher.run_verify.assert_called_once_with(extra_argv=())
    assert state.verify.scoped is False
    assert _notices(turn) == ["[harness verify] finish: verify_command exit 124 (1s)."]


def test_a_timeout_with_no_nearby_tests_stands(tmp_path: Path, monkeypatch: Any) -> None:
    """Nothing near the change to run: the timeout is the verdict; scoping
    never arms on an empty selection."""

    def no_tests_diff(_self: Workflow) -> str:
        return "diff --git a/docs/page.md b/docs/page.md\n"

    monkeypatch.setattr(Workflow, "_run_diff", no_tests_diff)
    wf, dispatcher = _scoped_wf(tmp_path, ["python", "-m", "pytest", "-q"])
    dispatcher.run_verify.return_value = _exec(124)
    state = LoopState(original_task="t", tool_calls=0)
    turn = _turn(finishing=True)
    wf._turn_harness_verify(state, turn)  # pyright: ignore[reportPrivateUsage]
    dispatcher.run_verify.assert_called_once_with(extra_argv=())
    assert state.verify.scoped is False
    assert turn.verify_just_failed is True


def test_a_models_own_timed_out_gate_gets_the_scoped_followup(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """run_verify_command exit 124 from the model's OWN call gets the scoped
    follow-up too. The harness-gate fallback alone never reached this flow (a
    self-judged turn is not re-judged), so pilot legs timed out at the full
    budget with no scoped re-run ever firing."""
    monkeypatch.setattr(Workflow, "_run_diff", _fake_diff)
    wf, dispatcher = _scoped_wf(tmp_path, ["python", "-m", "pytest", "-q"])
    dispatcher.run_verify.return_value = _exec(0)
    state = LoopState(original_task="t", tool_calls=0)
    turn = _turn()
    wf._note_tool_effects(  # pyright: ignore[reportPrivateUsage]
        state, turn, "run_verify_command", _exec(124), {}
    )
    dispatcher.run_verify.assert_called_once_with(extra_argv=("tests/test_mod.py",))
    assert state.verify.scoped is True
    assert turn.verify_just_passed is True  # the scoped green stands
    assert "not a full-suite pass" in turn.tool_results[-1].text
    # One verdict per turn, as on the harness path: the 124 is not noted as a
    # fail beside the scoped green (an on_verify_fail panel and the memory
    # flip nudge key on those flags).
    assert turn.verify_just_failed is False
    assert turn.verify_flipped_green is False
    assert state.verify.fail_streak == 0


def test_never_mode_leaves_the_models_timed_out_gate_alone(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """`never`: only the model's own run_verify_command calls run the gate, so
    a timeout there gets no harness re-run; the 124 is the turn's verdict."""
    monkeypatch.setattr(Workflow, "_run_diff", _fake_diff)
    wf, dispatcher = _scoped_wf(tmp_path, ["python", "-m", "pytest", "-q"], when="never")
    state = LoopState(original_task="t", tool_calls=0)
    turn = _turn()
    wf._note_tool_effects(  # pyright: ignore[reportPrivateUsage]
        state, turn, "run_verify_command", _exec(124), {}
    )
    dispatcher.run_verify.assert_not_called()
    assert state.verify.scoped is False
    assert turn.verify_just_failed is True


def test_a_full_green_from_the_models_own_gate_unarms_scoping(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """The model's own run_verify_command runs the full argv: a green there is
    a full pass, so scoping (armed by an earlier timeout) ends and the run's
    end reads a plain "passed", never "passed · scoped gate"."""
    monkeypatch.setattr(Workflow, "_run_diff", _fake_diff)
    wf, dispatcher = _scoped_wf(tmp_path, ["python", "-m", "pytest", "-q"])
    dispatcher.run_verify.return_value = _exec(0)
    state = LoopState(original_task="t", tool_calls=0)
    wf._note_tool_effects(  # pyright: ignore[reportPrivateUsage]
        state, _turn(), "run_verify_command", _exec(124), {}
    )
    assert state.verify.scoped is True
    turn = _turn()
    wf._note_tool_effects(  # pyright: ignore[reportPrivateUsage]
        state, turn, "run_verify_command", _exec(0), {}
    )
    assert state.verify.scoped is False
    assert turn.verify_just_passed is True
    dispatcher.run_verify.assert_called_once_with(extra_argv=("tests/test_mod.py",))


def test_a_denied_scoped_rerun_withholds_the_gate_for_the_run(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """One denial means the same on both call sites: the scoped re-run not
    approved withholds the gate for the rest of the run, in the harness
    path's words, and a later finish never asks again."""
    from agent6.tools.errors import ToolDenied

    monkeypatch.setattr(Workflow, "_run_diff", _fake_diff)
    wf, dispatcher = _scoped_wf(tmp_path, ["python", "-m", "pytest", "-q"])
    dispatcher.run_verify.side_effect = [
        _exec(124),
        ToolDenied("run_verify_command not approved"),
    ]
    state = LoopState(original_task="t", tool_calls=0)
    turn = _turn(finishing=True)
    wf._turn_harness_verify(state, turn)  # pyright: ignore[reportPrivateUsage]
    assert state.verify.denied is True
    assert (
        "[verify] scoped re-run: not run: run_verify_command not approved."
        " The gate is withheld for the rest of the run; the run ends unverified."
    ) in _notices(turn)
    turn2 = _turn(finishing=True, edited=True)
    wf._turn_harness_verify(state, turn2)  # pyright: ignore[reportPrivateUsage]
    assert dispatcher.run_verify.call_count == 2
    wf._gate_verify_finish(state, turn2)  # pyright: ignore[reportPrivateUsage]
    assert turn2.finish_signal == "done"


def test_a_silent_finish_over_a_standing_red_is_handed_back() -> None:
    """The model ran the gate itself and it was red; its next turn is prose
    with no tool call. The harness gate is skipped (that red already covers
    the untouched tree), so no verify fails on THIS turn, and the end was
    accepted as if the gate had never been red. The standing verdict decides."""
    from agent6.tools.results import ExecResult

    dispatcher = MagicMock()
    dispatcher.command_policy.return_value = "yes"
    dispatcher.run_verify.return_value = ExecResult(
        returncode=1, stdout="1 failed", stderr="", duration_s=1.0, exec_failed=False
    )
    wf = Workflow(
        root=Path("/tmp"),
        config=Config.model_validate(
            {"workflow": {"verify_command": ["true"], "verify_when": "finish", "verify_retries": 2}}
        ),
        provider=MagicMock(),
        dispatcher=dispatcher,
        logger=lambda _m: None,
        mode="run",
    )
    state = LoopState(original_task="t", tool_calls=0)
    state.verify.note_edit()
    state.verify.note_fail("sig")
    turn = TurnState(iteration=4, resp=MagicMock(), assistant=MagicMock())

    wf._end_gates(state, turn, ending="silent_finish")  # pyright: ignore[reportPrivateUsage]

    assert dispatcher.run_verify.call_count == 0, "the standing red covers the tree"
    assert turn.end_returned is True
    assert state.verify_finish_retries_used == 1


@pytest.mark.parametrize(("policy", "denied"), [("no", False), ("ask", True)])
def test_a_silent_end_is_not_handed_back_over_a_gate_the_model_cannot_run(
    policy: str, denied: bool
) -> None:
    """The standing red decides the hand-back, with the guards finish_session
    already carries: a gate the operator withheld (`run_commands = "no"`) or
    denied is not the model's to fix, and bouncing the end told it to."""
    dispatcher = MagicMock()
    dispatcher.command_policy.return_value = policy
    wf = Workflow(
        root=Path("/tmp"),
        config=Config.model_validate(
            {"workflow": {"verify_command": ["true"], "verify_when": "step", "verify_retries": 2}}
        ),
        provider=MagicMock(),
        dispatcher=dispatcher,
        logger=lambda _m: None,
        mode="run",
    )
    state = LoopState(original_task="t", tool_calls=0)
    state.verify.note_edit()
    state.verify.note_fail("sig")
    state.verify.denied = denied
    state.verify.note_edit()
    turn = TurnState(iteration=7, resp=MagicMock(), assistant=MagicMock())

    wf._end_gates(state, turn, ending="silent_finish")  # pyright: ignore[reportPrivateUsage]

    assert turn.end_returned is False
    assert dispatcher.run_verify.call_count == 0
