# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The end-of-run console headline must agree with `agent6 sessions`.

A finish_session over a red/stale verify emits session.end all_passed=false, so the
listing reads "finished". The console block used to read result.completed
(true for any finish_session) and print "passed" — the exact disagreement
status_word exists to prevent. print_session_end now folds the same session.end.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent6.app import finalize as _finalize
from agent6.app.finalize import print_interrupt_end, print_session_end
from agent6.app.reporter import STDIO_REPORTER, Reporter
from agent6.budget import BudgetTracker
from agent6.git_ops import GitStatus
from agent6.sessions.layout import SessionLayout
from agent6.workflows._session_state import SessionResult


def _layout(tmp_path: Path, session_id: str, events: list[dict[str, object]]) -> SessionLayout:
    rd = tmp_path / "sessions" / "runs" / session_id
    rd.mkdir(parents=True)
    (rd / "logs.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events), encoding="utf-8")
    return SessionLayout(state_dir=tmp_path, session_id=session_id)


def test_finish_session_over_red_verify_is_not_headlined_passed(
    tmp_path: Path, capsys: object
) -> None:
    layout = _layout(
        tmp_path,
        "r1",
        [
            {"type": "session.start", "session_id": "r1", "user_task": "t"},
            {"type": "session.end", "reason": "finish_session", "all_passed": False},
        ],
    )
    result = SessionResult(
        completed=True,
        reason="finish_session",
        summary="all tests pass",
        iterations=3,
        tool_calls=5,
    )
    print_session_end(
        result,
        layout=layout,
        cwd=tmp_path,
        budget=BudgetTracker(max_usd=-1, max_tokens_fallback=-1, max_percent=-1),
        console_stream=False,
        reporter=STDIO_REPORTER,
    )
    out = capsys.readouterr().out  # type: ignore[attr-defined]
    assert "finished" in out
    assert "passed" not in out.split("\n")[1]  # the headline line, not the agent's summary


def test_all_green_finish_is_headlined_passed(tmp_path: Path, capsys: object) -> None:
    layout = _layout(
        tmp_path,
        "r2",
        [
            {"type": "session.start", "session_id": "r2", "user_task": "t"},
            {"type": "session.end", "reason": "finish_session", "all_passed": True},
        ],
    )
    result = SessionResult(
        completed=True, reason="finish_session", summary="done", iterations=2, tool_calls=3
    )
    print_session_end(
        result,
        layout=layout,
        cwd=tmp_path,
        budget=BudgetTracker(max_usd=-1, max_tokens_fallback=-1, max_percent=-1),
        console_stream=False,
        reporter=STDIO_REPORTER,
    )
    out = capsys.readouterr().out  # type: ignore[attr-defined]
    assert "passed" in out


def _end_output(
    tmp_path: Path,
    session_id: str,
    result: SessionResult,
    capsys: pytest.CaptureFixture[str],
    manifest: dict[str, object] | None = None,
) -> str:
    layout = _layout(
        tmp_path,
        session_id,
        [
            {"type": "session.start", "session_id": session_id, "user_task": "t"},
            {"type": "session.end", "reason": result.reason, "all_passed": False},
        ],
    )
    if manifest is not None:
        (layout.session_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    print_session_end(
        result,
        layout=layout,
        cwd=tmp_path,
        budget=BudgetTracker(max_usd=-1, max_tokens_fallback=-1, max_percent=-1),
        console_stream=False,
        reporter=STDIO_REPORTER,
    )
    return capsys.readouterr().out


def test_the_red_gate_errand_is_only_printed_over_a_real_red(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """ "the gate is red, and nothing checked it before this run started" fired
    on verified="failed", which used to include legs where NO verify ran: the
    operator was sent to run the full gate at the base commit over a failure
    nobody observed. An unverified finish now says what is missing instead."""
    manifest: dict[str, object] = {
        "version": 3,
        "session_id": "r-red",
        "mode": "run",
        "user_task": "t",
        "base_sha": "a" * 40,
        "workflow": {"verify_command": ["pytest", "-q"], "verify_origin": "configured"},
    }
    result = SessionResult(
        completed=True,
        reason="finish_session",
        summary="",
        iterations=1,
        tool_calls=1,
        verified="failed",
    )
    out = _end_output(tmp_path, "r-red", result, capsys, manifest)
    assert "the gate is red" in out

    unverified = SessionResult(
        completed=True,
        reason="finish_session",
        summary="",
        iterations=1,
        tool_calls=1,
        verified="unverified",
    )
    out = _end_output(tmp_path, "r-unv", unverified, capsys, manifest)
    assert "the gate is red" not in out
    assert "no verify ran this leg" in out


def test_the_stale_gate_proposal_survives_an_unverified_verdict(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A worker may declare the gate stale because it CANNOT RUN AT ALL -- a
    leg with no verify observation. Keying the proposal print on
    verified="failed" alone would silently drop it there."""
    result = SessionResult(
        completed=True,
        reason="gate_stale",
        summary="",
        iterations=1,
        tool_calls=1,
        stale_gate="pytest -q tests/",
        verified="unverified",
    )
    out = _end_output(tmp_path, "r-stale", result, capsys)
    assert "it proposes: pytest -q tests/" in out


def test_the_stale_gate_remedy_is_a_command_that_installs_that_gate(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """`workflow.verify_command` is argv and takes no shell, so a proposal with
    a pipeline wraps as `sh -c`. Splitting it word by word printed a command
    that installs a gate handing `&& ruff check` to pytest as arguments -- and
    `config set` accepts it silently."""
    result = SessionResult(
        completed=True,
        reason="gate_stale",
        summary="",
        iterations=1,
        tool_calls=1,
        stale_gate="pytest -q tests/ && ruff check",
        verified="failed",
    )

    out = _end_output(tmp_path, "r-stale2", result, capsys)

    assert '\'["sh", "-c", "pytest -q tests/ && ruff check"]\'' in out


def test_end_banner_does_not_claim_merged_from_a_prior_legs_stamp(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A resumed run keeps committing on its branch under the FIRST leg's
    merged stamp (and this leg's auto-merge may have conflicted): the end block
    read the stamp alone, claimed "changes merged into main" over unmerged
    commits, and hid the merge command. The claim now holds only while the
    branch still points at the tip the stamp recorded -- the same comparison
    `sessions prune` trusts."""
    import subprocess as sp

    repo = tmp_path / "repo"
    repo.mkdir()
    sp.run(["git", "init", "-q", "-b", "main"], cwd=repo, check=True)
    (repo / "a.txt").write_text("x\n", encoding="utf-8")
    sp.run(["git", "add", "a.txt"], cwd=repo, check=True)
    sp.run(["git", "commit", "-q", "-m", "seed"], cwd=repo, check=True)
    sp.run(["git", "switch", "-qc", "agent6/r-leg2"], cwd=repo, check=True)
    merged_tip = sp.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, check=True, capture_output=True, text=True
    ).stdout.strip()
    (repo / "a.txt").write_text("leg 2 work\n", encoding="utf-8")
    sp.run(["git", "commit", "-qam", "leg 2"], cwd=repo, check=True)
    monkeypatch.chdir(repo)

    layout = _layout(
        tmp_path,
        "r-leg2",
        [
            {"type": "session.start", "session_id": "r-leg2", "user_task": "t"},
            {"type": "session.end", "reason": "finish_session", "all_passed": True},
        ],
    )
    layout.manifest_path.write_text(
        json.dumps(
            {
                "run_branch": "agent6/r-leg2",
                "base_branch": "main",
                "merged": {
                    "into": "main",
                    "sha": "abc123def456",
                    "ts": "2026-01-01T00:00:00Z",
                    "tip": merged_tip,
                },
            }
        ),
        encoding="utf-8",
    )
    result = SessionResult(
        completed=True, reason="finish_session", summary="done", iterations=1, tool_calls=1
    )
    print_session_end(
        result,
        layout=layout,
        cwd=repo,
        budget=BudgetTracker(max_usd=-1, max_tokens_fallback=-1, max_percent=-1),
        console_stream=False,
        reporter=STDIO_REPORTER,
    )
    out = capsys.readouterr().out
    assert "changes merged into" not in out
    assert "merge with:" in out


def test_end_banner_does_not_offer_merge_for_an_auto_merged_branch(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """auto_merge already merged (and auto_prune may have deleted) the run
    branch, so the footer must say it merged, not tell the operator to run
    `agent6 sessions merge` on a branch that is gone."""
    layout = _layout(
        tmp_path,
        "r-merged",
        [
            {"type": "session.start", "session_id": "r-merged", "user_task": "t"},
            {"type": "session.end", "reason": "finish_session", "all_passed": True},
        ],
    )
    layout.manifest_path.write_text(
        json.dumps(
            {
                "run_branch": "agent6/r-merged",
                "base_branch": "main",
                "merged": {"into": "main", "sha": "abc123def456", "ts": "2026-01-01T00:00:00Z"},
            }
        ),
        encoding="utf-8",
    )
    result = SessionResult(
        completed=True, reason="finish_session", summary="done", iterations=1, tool_calls=1
    )
    print_session_end(
        result,
        layout=layout,
        cwd=tmp_path,
        budget=BudgetTracker(max_usd=-1, max_tokens_fallback=-1, max_percent=-1),
        console_stream=False,
        reporter=STDIO_REPORTER,
    )
    out = capsys.readouterr().out
    assert "changes merged into main" in out
    assert "runs merge" not in out


def test_end_banner_does_not_advertise_a_run_branch_that_never_got_a_commit(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """branch_per_run named agent6/<id>, but no commit ever landed on it -- the
    chain's update-ref failed and the loop swallowed the error. The footer used
    to print "changes are on agent6/<id>" and tell the operator to
    `agent6 sessions merge` a branch that does not exist, while the run reported
    success and the edits sat uncommitted. It must state the truth instead."""
    import subprocess as sp

    repo = tmp_path / "repo"
    repo.mkdir()
    sp.run(["git", "init", "-q", "-b", "main"], cwd=repo, check=True)
    sp.run(["git", "config", "user.email", "t@t"], cwd=repo, check=True)
    sp.run(["git", "config", "user.name", "t"], cwd=repo, check=True)
    (repo / "seed.txt").write_text("seed\n", encoding="utf-8")
    sp.run(["git", "add", "-A"], cwd=repo, check=True)
    sp.run(["git", "commit", "-qm", "seed"], cwd=repo, check=True)
    # The agent's edit is left in the tree; no agent6/miss branch was ever cut.
    (repo / "work.txt").write_text("stranded agent work\n", encoding="utf-8")
    monkeypatch.chdir(repo)

    layout = _layout(
        tmp_path,
        "miss",
        [
            {"type": "session.start", "session_id": "miss", "user_task": "t"},
            {"type": "session.end", "reason": "finish_session", "all_passed": True},
        ],
    )
    layout.manifest_path.write_text(
        json.dumps({"run_branch": "agent6/miss", "base_branch": "main"}), encoding="utf-8"
    )
    result = SessionResult(
        completed=True, reason="finish_session", summary="done", iterations=1, tool_calls=1
    )
    print_session_end(
        result,
        layout=layout,
        cwd=repo,
        budget=BudgetTracker(max_usd=-1, max_tokens_fallback=-1, max_percent=-1),
        console_stream=False,
        reporter=STDIO_REPORTER,
    )
    out = capsys.readouterr().out
    assert "changes are on agent6/miss" not in out
    assert "sessions merge" not in out  # never advertise merge for a missing branch
    assert "no commit on agent6/miss" in out  # the truthful warning
    assert "uncommitted in the working tree" in out


def test_end_banner_warns_when_checkout_is_parked_on_the_run_branch(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    layout = _layout(
        tmp_path,
        "r3",
        [
            {"type": "session.start", "session_id": "r3", "user_task": "t"},
            {"type": "session.end", "reason": "finish_session", "all_passed": True},
        ],
    )
    layout.manifest_path.write_text(
        json.dumps({"run_branch": "agent6/r3", "base_branch": "main"}), encoding="utf-8"
    )

    # The checkout is still on the run branch (branch_per_run never switches back).
    def _on_run_branch(_p: Path) -> GitStatus:
        return GitStatus(
            branch="agent6/r3", head_sha="x", is_clean=True, untracked_count=0, modified_count=0
        )

    monkeypatch.setattr(_finalize, "git_status", _on_run_branch)

    # Being ON the branch means it exists; the footer only names it once its
    # commits actually landed.
    def _branch_present(_p: Path, _n: str) -> bool:
        return True

    monkeypatch.setattr(_finalize, "branch_exists", _branch_present)
    result = SessionResult(
        completed=True, reason="finish_session", summary="done", iterations=1, tool_calls=1
    )
    print_session_end(
        result,
        layout=layout,
        cwd=tmp_path,
        budget=BudgetTracker(max_usd=-1, max_tokens_fallback=-1, max_percent=-1),
        console_stream=False,
        reporter=STDIO_REPORTER,
    )
    out = capsys.readouterr().out
    assert "you are on agent6/r3" in out
    assert "git switch main" in out


def test_interrupt_end_prints_cost_resume_and_branch_hints(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    # A Ctrl-C interrupt used to print only "run interrupted": no spend, no resume
    # hint, and no note the user was left on the run branch.
    layout = _layout(
        tmp_path, "r4", [{"type": "session.start", "session_id": "r4", "user_task": "t"}]
    )
    layout.manifest_path.write_text(
        json.dumps({"run_branch": "agent6/r4", "base_branch": "main"}), encoding="utf-8"
    )

    def _on_run_branch(_p: Path) -> GitStatus:
        return GitStatus(
            branch="agent6/r4", head_sha="x", is_clean=True, untracked_count=0, modified_count=0
        )

    monkeypatch.setattr(_finalize, "git_status", _on_run_branch)
    print_interrupt_end(
        layout=layout,
        cwd=tmp_path,
        budget=BudgetTracker(max_usd=-1, max_tokens_fallback=-1, max_percent=-1),
        reporter=STDIO_REPORTER,
    )
    out = capsys.readouterr().out
    assert "Token + cost summary" in out  # the budget/cost block
    assert "resume with:  agent6 resume r4" in out
    assert "you are on agent6/r4" in out and "git switch main" in out


def test_provider_error_is_headlined_failed(tmp_path: Path, capsys: object) -> None:
    layout = _layout(
        tmp_path,
        "r3",
        [
            {"type": "session.start", "session_id": "r3", "user_task": "t"},
            {"type": "session.end", "reason": "provider_error", "all_passed": False},
        ],
    )
    result = SessionResult(
        completed=False, reason="provider_error", summary="", iterations=1, tool_calls=0
    )
    print_session_end(
        result,
        layout=layout,
        cwd=tmp_path,
        budget=BudgetTracker(max_usd=-1, max_tokens_fallback=-1, max_percent=-1),
        console_stream=False,
        reporter=STDIO_REPORTER,
    )
    out = capsys.readouterr().out  # type: ignore[attr-defined]
    assert "failed" in out and "provider error" in out


def test_end_banner_adds_the_run_total_across_resume_legs(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # The tracker's "TOTAL" line is per-leg (each resume starts a fresh budget);
    # a resumed run's banner must also state the true cumulative spend.
    layout = _layout(
        tmp_path,
        "r7",
        [
            {"type": "session.start", "session_id": "r7", "user_task": "t"},
            {"type": "budget.update", "usd_total": 0.019},
            {"type": "session.end", "reason": "finish_session", "all_passed": True},
            {"type": "loop.resume.start", "iteration": 4},
            {"type": "budget.update", "usd_total": 0.0126},
            {"type": "session.end", "reason": "finish_session", "all_passed": True},
        ],
    )
    result = SessionResult(
        completed=True, reason="finish_session", summary="", iterations=5, tool_calls=2
    )
    print_session_end(
        result,
        layout=layout,
        cwd=tmp_path,
        budget=BudgetTracker(max_usd=-1, max_tokens_fallback=-1, max_percent=-1),
        console_stream=False,
        reporter=STDIO_REPORTER,
    )
    out = capsys.readouterr().out
    assert "RUN TOTAL (all 2 legs): $0.03" in out


def test_end_banner_stays_quiet_on_a_single_leg_run(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    layout = _layout(
        tmp_path,
        "r8",
        [
            {"type": "session.start", "session_id": "r8", "user_task": "t"},
            {"type": "budget.update", "usd_total": 0.01},
            {"type": "session.end", "reason": "finish_session", "all_passed": True},
        ],
    )
    result = SessionResult(
        completed=True, reason="finish_session", summary="", iterations=2, tool_calls=1
    )
    print_session_end(
        result,
        layout=layout,
        cwd=tmp_path,
        budget=BudgetTracker(max_usd=-1, max_tokens_fallback=-1, max_percent=-1),
        console_stream=False,
        reporter=STDIO_REPORTER,
    )
    assert "RUN TOTAL" not in capsys.readouterr().out


def test_finalize_auto_stash_pops_the_run_stash_not_the_latest(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The finalizer restores THE stash the run pushed (found by its run-id
    message), not stash@{0}: a stash pushed during the run otherwise got
    popped as the 'pre-run work' while the real pre-run work stayed hidden."""
    import subprocess

    from agent6.app.finalize import finalize_auto_stash
    from agent6.git_ops import auto_stash_message, stash_tracked_changes

    repo = tmp_path / "repo"
    repo.mkdir()

    def git(*args: str) -> None:
        subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)

    git("init", "-q")
    git("config", "user.email", "t@t")
    git("config", "user.name", "t")
    (repo / "base.txt").write_text("base\n", encoding="utf-8")
    (repo / "pre.txt").write_text("", encoding="utf-8")
    (repo / "mid.txt").write_text("", encoding="utf-8")
    git("add", "-A")
    git("commit", "-qm", "base")
    (repo / "pre.txt").write_text("pre-run work\n", encoding="utf-8")
    stash_tracked_changes(repo, auto_stash_message("r1"))
    (repo / "mid.txt").write_text("mid-run work\n", encoding="utf-8")
    stash_tracked_changes(repo, "operator stash pushed mid-run")
    base = subprocess.run(
        ["git", "-C", str(repo), "branch", "--show-current"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    finalize_auto_stash(
        repo,
        base_branch=base,
        run_branch=None,
        auto_pop=True,
        session_id="r1",
        reporter=STDIO_REPORTER,
    )
    assert "restored your pre-run changes" in capsys.readouterr().err
    assert (repo / "pre.txt").read_text(encoding="utf-8") == "pre-run work\n"
    assert (repo / "mid.txt").read_text(encoding="utf-8") == ""  # the mid-run stash stays a stash


def test_finalize_auto_stash_reports_a_vanished_stash(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A stash the operator already popped mid-run is reported, not silently
    'restored' (and no longer pops whatever happens to sit at stash@{0})."""
    import subprocess

    from agent6.app.finalize import finalize_auto_stash

    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "-C", str(repo), "init", "-q"], check=True)
    finalize_auto_stash(
        repo,
        base_branch="master",
        run_branch=None,
        auto_pop=True,
        session_id="r1",
        reporter=STDIO_REPORTER,
    )
    assert "auto-stash not found" in capsys.readouterr().err


def test_finalize_auto_stash_prints_a_failed_bystander_putback(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """When the restore raises because a raced drop took a concurrent stash and
    putting it back failed, finalization prints the recovery command and
    finishes -- the loss must reach the operator, not crash the finalizer."""
    import subprocess

    from agent6.app import finalize as finalize_mod
    from agent6.app.finalize import finalize_auto_stash
    from agent6.git_ops import GitError, auto_stash_message, stash_tracked_changes

    repo = tmp_path / "repo"
    repo.mkdir()

    def git(*args: str) -> None:
        subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)

    git("init", "-q", "-b", "main")
    git("config", "user.email", "t@t")
    git("config", "user.name", "t")
    (repo / "base.txt").write_text("base\n", encoding="utf-8")
    git("add", "-A")
    git("commit", "-qm", "base")
    (repo / "base.txt").write_text("pre-run work\n", encoding="utf-8")
    stash_tracked_changes(repo, auto_stash_message("r1"))

    def raising_restore(cwd: object, entry: object) -> bool:
        raise GitError(
            "a stash pushed concurrently ('x') was taken by a raced drop and putting"
            " it back failed; restore it with:\n    git stash store -m 'x' abc123"
        )

    monkeypatch.setattr(finalize_mod, "restore_stash", raising_restore)
    finalize_auto_stash(
        repo,
        base_branch="main",
        run_branch=None,
        auto_pop=True,
        session_id="r1",
        reporter=STDIO_REPORTER,
    )
    err = capsys.readouterr().err
    assert "restored your pre-run changes" in err
    assert "git stash store" in err  # the recovery command reaches the operator


def test_stash_recovery_hint_is_identity_stable(tmp_path: Path) -> None:
    """The hint a DETACHED run prints has the longest window of all -- the
    operator comes back hours later -- and it still named a positional
    `git stash pop`, the exact failure `restore_stash` was changed to avoid.
    One owner builds the sha-based line for every caller."""
    import subprocess

    from agent6.app.finalize import stash_recovery_hint
    from agent6.git_ops import auto_stash_message, stash_tracked_changes

    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", "-b", "main"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "t@t"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=repo, check=True)
    (repo / "f.txt").write_text("base\n", encoding="utf-8")
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", "init"], cwd=repo, check=True)
    (repo / "f.txt").write_text("pre-run work\n", encoding="utf-8")
    stash_tracked_changes(repo, auto_stash_message("r9"))
    # A stash pushed later shifts every position; the hint must not care.
    (repo / "f.txt").write_text("someone else\n", encoding="utf-8")
    stash_tracked_changes(repo, "an unrelated stash")

    hint = stash_recovery_hint(repo, session_id="r9", base_branch="main")
    assert hint is not None
    assert "git stash pop" not in hint  # positional restores the wrong stash
    # The chain never moves the checkout: on main, no `git checkout main` prefix.
    assert hint.startswith("git stash apply ")
    subprocess.run(["git", "checkout", "-q", "-b", "elsewhere"], cwd=repo, check=True)
    away = stash_recovery_hint(repo, session_id="r9", base_branch="main")
    assert away is not None and away.startswith("git checkout main && git stash apply ")
    subprocess.run(["git", "checkout", "-q", "main"], cwd=repo, check=True)
    sha = hint.rsplit(" ", 1)[1]
    assert len(sha) == 40
    # The sha names the RUN's stash, not the newest one.
    subject = subprocess.run(
        ["git", "log", "-1", "--format=%s", sha],
        cwd=repo,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    assert "r9" in subject

    # No stash for that run: the caller gets None and says so its own way.
    assert stash_recovery_hint(repo, session_id="nope", base_branch="main") is None


@pytest.mark.parametrize(
    ("mode", "expected"),
    [
        (
            "plan",
            [
                "agent6 plan edit quiet-fox-AAAAAA",
                "agent6 resume quiet-fox-AAAAAA --steer",
                "agent6 run --from quiet-fox-AAAAAA",
            ],
        ),
        ("ask", ["agent6 run --from quiet-fox-AAAAAA"]),
        ("run", []),
    ],
)
def test_a_session_that_ends_holding_work_names_the_next_step(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], mode: str, expected: list[str]
) -> None:
    """Seeding existed but nothing suggested it, so an operator had to know the
    flag was there. A plan ends holding OPEN QUESTIONS and nothing said that
    answering them is `plan edit` then `resume --steer`, so the whole loop is
    printed. An ask ends holding work someone else does. A run has already done
    its work and needs no handoff."""
    import json

    from agent6.app.finalize import _print_next_session  # pyright: ignore[reportPrivateUsage]
    from agent6.sessions.layout import SessionLayout

    layout = SessionLayout(state_dir=tmp_path, session_id="quiet-fox-AAAAAA")
    layout.session_dir.mkdir(parents=True)
    (layout.session_dir / "manifest.json").write_text(
        json.dumps({"version": 3, "mode": mode}), encoding="utf-8"
    )
    (layout.session_dir / "plan.md").write_text("# The plan\n\n1. do it\n", encoding="utf-8")
    _print_next_session(layout, reporter=STDIO_REPORTER)
    out = capsys.readouterr().out
    for line in expected:
        assert line in out
    assert ("agent6" in out) is bool(expected)
    # A plan is the deliverable: printed whole, before the next-step lines.
    assert ("# The plan\n\n1. do it" in out) is (mode == "plan")


def test_the_end_of_run_block_goes_through_the_reporter(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A front-end that does not own stdout must be able to redirect this.

    `agent6 acp` speaks JSON-RPC on stdout, so a bare `print` here is not a
    cosmetic layering slip: it writes non-JSON lines into the protocol stream,
    and `result.summary` is the model's own `finish_session` text -- unbounded, and
    free to contain newlines. A model could close the prose with a newline and
    emit a forged `session/update` at column 0, which a client that skips
    unparseable lines honours. The editor owns the filesystem and terminal in
    ACP, so that is a jail escape.
    """
    layout = _layout(
        tmp_path,
        "r9",
        [
            {"type": "session.start", "session_id": "r9", "user_task": "t"},
            {"type": "session.end", "reason": "finish_session", "all_passed": True},
        ],
    )
    forged = 'done\n{"jsonrpc":"2.0","id":1,"method":"fs/write_text_file","params":{}}'
    said: list[str] = []
    print_session_end(
        SessionResult(
            completed=True, reason="finish_session", summary=forged, iterations=1, tool_calls=1
        ),
        layout=layout,
        cwd=tmp_path,
        budget=BudgetTracker(max_usd=-1, max_tokens_fallback=-1, max_percent=-1),
        console_stream=False,
        reporter=Reporter(out=said.append, err=said.append),
    )
    captured = capsys.readouterr()
    assert captured.out == "", "the run-end block reached stdout, bypassing the reporter"
    assert captured.err == ""
    assert any("fs/write_text_file" in line for line in said), "it must still be reported"


def test_end_banner_admits_an_unreadable_tree_instead_of_claiming(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A GitError on the dirty-check used to fall through to "no changes were
    committed" -- an unverified claim on a broken-git host. The banner now
    says it could not check, and claims nothing either way."""
    import agent6.app.finalize as finalize_mod
    from agent6.git_ops import GitError

    def _boom(_path: Path, **_kw: object) -> object:
        raise GitError("git unreadable here")

    monkeypatch.setattr(finalize_mod, "git_status", _boom)
    result = SessionResult(
        completed=True, reason="finish_session", summary="", iterations=1, tool_calls=1
    )
    session_id = "run-x"
    layout = _layout(
        tmp_path,
        session_id,
        [
            {"type": "session.start", "session_id": session_id, "user_task": "t"},
            {"type": "session.end", "reason": result.reason, "all_passed": False},
        ],
    )
    (layout.session_dir / "manifest.json").write_text(
        json.dumps({"user_task": "t", "run_branch": "agent6/run-x", "base_branch": "master"}),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    print_session_end(
        result,
        layout=layout,
        cwd=tmp_path,
        budget=BudgetTracker(max_usd=-1, max_tokens_fallback=-1, max_percent=-1),
        console_stream=False,
        reporter=STDIO_REPORTER,
    )
    out = capsys.readouterr().out
    assert "could not check the working tree" in out
    assert "no changes were committed" not in out
    assert "WARNING: the run finished with no commit on" not in out


def test_a_failed_run_keeps_its_reason_on_the_console_stream(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The live done line carries the finish summary only for a clean finish,
    and session.end carries no message, so suppressing `result.summary` under
    `console_stream` left a foreground operator with "● provider error" and no
    URL, no errno, no status."""
    layout = _layout(
        tmp_path,
        "r-fail",
        [
            {"type": "session.start", "session_id": "r-fail", "user_task": "t"},
            {"type": "session.end", "reason": "provider_error", "all_passed": None},
        ],
    )
    result = SessionResult(
        completed=False,
        reason="provider_error",
        summary="provider error at iter 1: HTTP error calling http://127.0.0.1:9/v1 (openai)",
        iterations=1,
        tool_calls=0,
    )

    print_session_end(
        result,
        layout=layout,
        cwd=tmp_path,
        budget=BudgetTracker(max_usd=-1, max_tokens_fallback=-1, max_percent=-1),
        console_stream=True,
        reporter=STDIO_REPORTER,
    )

    out = capsys.readouterr().out
    assert "HTTP error calling http://127.0.0.1:9/v1" in out


def test_a_clean_finish_does_not_repeat_the_summary_the_stream_showed(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    layout = _layout(
        tmp_path,
        "r-ok",
        [
            {"type": "session.start", "session_id": "r-ok", "user_task": "t"},
            {"type": "session.end", "reason": "finish_session", "all_passed": True},
        ],
    )
    result = SessionResult(
        completed=True, reason="finish_session", summary="fixed it", iterations=1, tool_calls=1
    )

    print_session_end(
        result,
        layout=layout,
        cwd=tmp_path,
        budget=BudgetTracker(max_usd=-1, max_tokens_fallback=-1, max_percent=-1),
        console_stream=True,
        reporter=STDIO_REPORTER,
    )

    assert "fixed it" not in capsys.readouterr().out


def test_the_sandbox_warning_states_its_remedy_once(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The four remedy bullets were printed per binary: three unreachable tools
    filled eighteen lines, twelve of them the same boilerplate."""
    layout = _layout(
        tmp_path,
        "r-tools",
        [
            {"type": "session.start", "session_id": "r-tools", "user_task": "t"},
            *(
                {"type": "loop.sandbox_tool_unreachable", "binary": b}
                for b in ("cargo", "node", "pyenv")
            ),
            {"type": "session.end", "reason": "finish_session", "all_passed": True},
        ],
    )
    result = SessionResult(
        completed=True, reason="finish_session", summary="", iterations=1, tool_calls=1
    )

    print_session_end(
        result,
        layout=layout,
        cwd=tmp_path,
        budget=BudgetTracker(max_usd=-1, max_tokens_fallback=-1, max_percent=-1),
        console_stream=False,
        reporter=STDIO_REPORTER,
    )
    out = capsys.readouterr().out

    assert out.count("WARNING:") == 1
    assert out.count("- run with --dangerously-disable-sandbox") == 1
    for binary in ("cargo", "node", "pyenv"):
        assert f"`{binary}`" in out


def test_the_run_total_rides_the_receipt_channel(tmp_path: Path) -> None:
    """A front-end with a live view routes the cost receipt to its log; the
    RUN TOTAL line went through `out`, so on a resumed ACP turn it was the one
    receipt line the editor saw, split from the block it belongs to."""
    from agent6.app.reporter import Reporter

    layout = _layout(
        tmp_path,
        "r8",
        [
            {"type": "session.start", "session_id": "r8", "user_task": "t"},
            {"type": "budget.update", "usd_total": 0.019},
            {"type": "session.end", "reason": "finish_session", "all_passed": True},
            {"type": "loop.resume.start", "iteration": 4},
            {"type": "budget.update", "usd_total": 0.0126},
            {"type": "session.end", "reason": "finish_session", "all_passed": True},
        ],
    )
    out: list[str] = []
    receipt: list[str] = []
    print_session_end(
        SessionResult(
            completed=True, reason="finish_session", summary="", iterations=5, tool_calls=2
        ),
        layout=layout,
        cwd=tmp_path,
        budget=BudgetTracker(max_usd=-1, max_tokens_fallback=-1, max_percent=-1),
        console_stream=False,
        reporter=Reporter(out=out.append, err=out.append, receipt=receipt.append),
    )
    assert any("RUN TOTAL (all 2 legs)" in line for line in receipt)
    assert not any("RUN TOTAL" in line for line in out)
