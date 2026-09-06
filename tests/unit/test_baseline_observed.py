# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Was the gate already red before this run touched anything?

Observed for free during the run -- a verify against an unmodified tree IS the
answer -- rather than bought with a second full gate run in the teardown.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from agent6.tools.results import ExecResult
from agent6.workflows.loop import (
    LoopState,
    TurnState,
    Workflow,
)

_BASE = "b" * 40


def _wf(*, head: str = _BASE, clean: bool = True) -> Workflow:
    wf = Workflow.__new__(Workflow)
    wf.base_sha = _BASE
    wf.root = Path("/nonexistent")
    object.__setattr__(wf, "_git_status", lambda: SimpleNamespace(is_clean=clean, head_sha=head))
    object.__setattr__(wf, "_emit", _quiet)
    wf.config = SimpleNamespace(  # pyright: ignore[reportAttributeAccessIssue]
        workflow=SimpleNamespace(verify_command=("pytest",))
    )
    return wf


def _quiet(*_a: object, **_k: object) -> None:
    return None


def _patch_git(monkeypatch: pytest.MonkeyPatch, wf: Workflow) -> None:
    def _status(_root: object, **_kw: object) -> object:
        return wf._git_status()  # pyright: ignore[reportAttributeAccessIssue]

    monkeypatch.setattr("agent6.workflows.loop.git_status", _status)


def _state() -> LoopState:
    return LoopState(original_task="t", tool_calls=0)


def _verify(rc: int, *, duration_s: float = 5.0) -> ExecResult:
    return ExecResult(returncode=rc, stdout="", stderr="", duration_s=duration_s, exec_failed=False)


def _turn() -> TurnState:
    return TurnState(iteration=1, resp=MagicMock(), assistant=MagicMock())


@pytest.mark.parametrize("rc", [0, 1])
def test_a_verify_at_the_base_commit_is_the_baseline(
    rc: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    state, turn = _state(), _turn()
    wf = _wf()
    _patch_git(monkeypatch, wf)
    wf._note_verify_result(state, turn, _verify(rc))  # pyright: ignore[reportPrivateUsage]
    assert state.verify.baseline_ok is (rc == 0)


def test_the_worker_is_told_when_it_inherited_a_red_gate(monkeypatch: pytest.MonkeyPatch) -> None:
    """So it stops chasing failures it did not cause, DURING the run -- which
    is worth more than the same fact explained afterwards."""
    state, turn = _state(), _turn()
    wf = _wf()
    _patch_git(monkeypatch, wf)
    wf._note_verify_result(state, turn, _verify(1))  # pyright: ignore[reportPrivateUsage]
    assert any("already failing" in str(n) for n in turn.tool_results)


def test_a_leg_that_moved_past_the_base_claims_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    """THE resume bug: every reason an operator resumes -- a budget stop, an
    iteration cap, a provider error -- commits the leg's work first. Leg two
    then opens on a CLEAN tree whose HEAD already carries leg one's breakage,
    and "has the model edited yet" read that as the base. `/parallel` does the
    same by merging lane commits into the workspace."""
    state, turn = _state(), _turn()
    wf = _wf(head="c" * 40)
    _patch_git(monkeypatch, wf)
    wf._note_verify_result(state, turn, _verify(1))  # pyright: ignore[reportPrivateUsage]
    assert state.verify.baseline_ok is None


def test_a_dirty_tree_claims_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    state, turn = _state(), _turn()
    wf = _wf(clean=False)
    _patch_git(monkeypatch, wf)
    wf._note_verify_result(state, turn, _verify(1))  # pyright: ignore[reportPrivateUsage]
    assert state.verify.baseline_ok is None


def test_an_unreadable_git_claims_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every other caller treats an unreadable git as "assume clean". Here that
    would exonerate the run for its own breakage, so it fails closed."""
    from agent6.git_ops import GitError

    def _boom(_root: object, **_kw: object) -> object:
        raise GitError("index.lock held")

    state, turn = _state(), _turn()
    monkeypatch.setattr("agent6.workflows.loop.git_status", _boom)
    _wf()._note_verify_result(state, turn, _verify(1))  # pyright: ignore[reportPrivateUsage]
    assert state.verify.baseline_ok is None


def test_a_run_that_already_went_green_owns_its_later_red(monkeypatch: pytest.MonkeyPatch) -> None:
    """It demonstrably could pass, so a later red is its own -- even if the
    gate was red at the base."""
    state, turn = _state(), _turn()
    state.verify.ever_passed = True
    wf = _wf()
    _patch_git(monkeypatch, wf)
    wf._note_verify_result(state, turn, _verify(1))  # pyright: ignore[reportPrivateUsage]
    assert state.verify.baseline_ok is None


@pytest.mark.parametrize(
    "result",
    [
        ExecResult(
            returncode=127,
            stdout="",
            stderr="pytest: command not found",
            duration_s=0.01,
            exec_failed=False,
        ),
        ExecResult(returncode=124, stdout="", stderr="", duration_s=600.0, exec_failed=False),
        ExecResult(returncode=1, stdout="", stderr="", duration_s=1.0, exec_failed=True),
    ],
    ids=["runner-absent", "timed-out", "could-not-exec"],
)
def test_a_gate_that_never_produced_a_verdict_is_not_a_red_baseline(
    result: ExecResult, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Recording one would excuse every real failure for the rest of the run."""
    state, turn = _state(), _turn()
    wf = _wf()
    _patch_git(monkeypatch, wf)
    wf._note_verify_result(state, turn, result)  # pyright: ignore[reportPrivateUsage]
    assert state.verify.baseline_ok is None


def test_a_plan_pass_is_not_reported_as_a_red_gate() -> None:
    """Plan mode can run the gate but never edits, so a red one is always
    "already red" -- and `finish_planning` would have been relabelled, turning
    a clean plan into "gate was already red"."""
    wf = Workflow.__new__(Workflow)
    wf.config = SimpleNamespace(  # pyright: ignore[reportAttributeAccessIssue]
        workflow=SimpleNamespace(verify_command=("pytest",))
    )
    state = _state()
    state.verify.baseline_ok = False
    state.verify.last_ok = False
    turn = _turn()
    turn.finish_kind = "finish_planning"
    assert wf._finish_reason(turn, state) == "finish_planning"  # pyright: ignore[reportPrivateUsage]


def test_a_red_tree_still_exits_red_whoever_caused_it() -> None:
    """Attribution belongs in the word, not the exit code: a script reading 0
    would take it as a passing gate, and the tree is not green either way."""
    from agent6.app.finalize import session_exit_code
    from agent6.workflows._session_state import SessionResult

    inherited = SessionResult(
        completed=True,
        reason="gate_red_at_base",
        summary="s",
        iterations=1,
        tool_calls=1,
        verified="failed",
    )
    assert session_exit_code(inherited) == 4


def test_the_listing_and_the_header_agree_on_the_word() -> None:
    from agent6.viewmodel.listing import status_word

    assert status_word(finished=True, all_passed=False, end_reason="gate_red_at_base") == (
        "finished",
        "gate was already red",
    )


def test_green_is_not_demanded_of_a_run_that_inherited_a_red_gate(tmp_path: Path) -> None:
    """The finish certification returns a red finish until the gate goes green.
    Over a gate that was already red, that is demanding the worker repair
    whatever it inherited before it may stop."""
    wf = _wf()
    wf.mode = "run"
    wf.config = SimpleNamespace(  # pyright: ignore[reportAttributeAccessIssue]
        workflow=SimpleNamespace(verify_command=("pytest",), verify_when="finish", verify_retries=2)
    )
    state = LoopState(original_task="t", tool_calls=0)
    state.verify.last_ok = False
    state.verify.baseline_ok = False
    turn = TurnState(iteration=1, resp=MagicMock(), assistant=MagicMock())
    turn.finish_signal = MagicMock()
    turn.finish_kind = "finish_session"

    wf._gate_verify_finish(state, turn)  # pyright: ignore[reportPrivateUsage]

    assert turn.finish_signal is not None, "the finish was bounced over an inherited failure"
