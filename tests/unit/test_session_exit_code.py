# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""`agent6 run`/`resume` process exit codes.

CONFIG.md documents a budget-exhausted run as exit 3 (resumable: raise the cap
and `agent6 resume`) and a finish over a red verify as exit 4; everything else
completed=False is exit 1, a clean or ungated finish is 0.
"""

from __future__ import annotations

from pathlib import Path

from agent6.app.finalize import session_exit_code
from agent6.workflows._session_state import SessionEndReason, Verification
from agent6.workflows.loop import SessionResult


def _result(
    *, completed: bool, reason: SessionEndReason, verified: Verification = "not_applicable"
) -> SessionResult:
    return SessionResult(
        completed=completed,
        reason=reason,
        summary="",
        iterations=1,
        tool_calls=1,
        verified=verified,
    )


def test_exit_code_success_is_zero() -> None:
    assert session_exit_code(_result(completed=True, reason="finish_session")) == 0


def test_exit_code_budget_exhausted_is_three() -> None:
    # The documented "raise the cap and resume" signal.
    assert session_exit_code(_result(completed=False, reason="budget_exhausted")) == 3


def test_exit_code_other_failures_are_one() -> None:
    for reason in ("provider_error", "max_iterations", "went_quiet", "steer_abort"):
        assert session_exit_code(_result(completed=False, reason=reason)) == 1


def test_exit_code_finish_over_a_red_verify_is_four() -> None:
    """`completed` means the agent stopped deliberately, not that the work
    verified: a finish_session over a red or stale gate exited 0 and read as
    success to every script. Its own code, distinct from a broken run (1)."""
    assert (
        session_exit_code(_result(completed=True, reason="finish_session", verified="failed")) == 4
    )
    assert session_exit_code(_result(completed=True, reason="settled", verified="failed")) == 4


def test_exit_code_verified_finish_is_zero() -> None:
    # Green, and gateless (nothing to verify) -- both are exit 0.
    assert (
        session_exit_code(_result(completed=True, reason="finish_session", verified="passed")) == 0
    )
    assert (
        session_exit_code(_result(completed=True, reason="settled", verified="not_applicable")) == 0
    )


def test_exit_code_unverified_finish_is_four() -> None:
    """4 means "the tree is not green": a gated finish nothing observed (no
    verify ran this leg, or edits landed after the last green) exits 4 like a
    red one -- exiting 0 would let a worker pass by never running the gate."""
    assert (
        session_exit_code(_result(completed=True, reason="finish_session", verified="unverified"))
        == 4
    )


def test_auto_merge_needs_a_vouched_for_tree() -> None:
    """auto_merge lands only work the gate vouched for (or that had no gate):
    a red OR unverified finish stays on its branch. The eligibility check is
    one shared predicate, so run and resume cannot drift."""
    from agent6.app.finalize import auto_merge_eligible

    assert auto_merge_eligible(_result(completed=True, reason="finish_session", verified="passed"))
    assert auto_merge_eligible(_result(completed=True, reason="settled", verified="not_applicable"))
    for bad in ("failed", "unverified"):
        assert not auto_merge_eligible(
            _result(completed=True, reason="finish_session", verified=bad)  # pyright: ignore[reportArgumentType]
        )
    assert not auto_merge_eligible(
        _result(completed=False, reason="max_iterations", verified="passed")
    )


def test_exit_code_stranded_edits_are_five() -> None:
    """Completed, gate green (or absent), but the promised branch never
    materialized and the edits sit uncommitted: 0 would tell a script the
    deliverable landed. A red gate outranks 5 (the gate is the primary
    signal); an unstranded finish stays 0."""
    ok = _result(completed=True, reason="finish_session", verified="passed")
    assert session_exit_code(ok, stranded=True) == 5
    assert session_exit_code(ok, stranded=False) == 0
    red = _result(completed=True, reason="finish_session", verified="failed")
    assert session_exit_code(red, stranded=True) == 4
    broke = _result(completed=False, reason="provider_error")
    assert session_exit_code(broke, stranded=True) == 1


def test_stranded_edits_reads_git_reality(tmp_path: Path) -> None:
    """The predicate is true exactly when the manifest promised a branch that
    does not exist AND the tree is dirty; a clean tree (nothing to commit) and
    an existing branch are both False."""
    import subprocess

    from agent6.app.finalize import stranded_edits
    from agent6.sessions.layout import SessionLayout

    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    (repo / "a.txt").write_text("x", encoding="utf-8")
    subprocess.run(["git", "-C", str(repo), "add", "a.txt"], check=True)
    subprocess.run(
        ["git", "-C", str(repo), "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "i"],
        check=True,
    )
    import json

    layout = SessionLayout(state_dir=tmp_path / "state", session_id="r1")
    layout.session_dir.mkdir(parents=True)
    (layout.session_dir / "manifest.json").write_text(
        json.dumps({"user_task": "t", "run_branch": "agent6/r1", "base_branch": "master"}),
        encoding="utf-8",
    )
    result = _result(completed=True, reason="finish_session", verified="passed")
    import os

    old = Path.cwd()
    os.chdir(repo)
    try:
        assert stranded_edits(result, layout, repo) is False  # clean tree
        (repo / "a.txt").write_text("changed", encoding="utf-8")
        assert stranded_edits(result, layout, repo) is True  # dirty + branch missing
        subprocess.run(["git", "-C", str(repo), "branch", "agent6/r1"], check=True)
        assert stranded_edits(result, layout, repo) is False  # branch exists
    finally:
        os.chdir(old)
