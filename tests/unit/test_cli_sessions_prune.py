# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Tests for `agent6 sessions prune`."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from agent6.config.layer import resolved_state_dir
from agent6.git_ops import chain_ref_for
from agent6.sessions.layout import SessionLayout
from agent6.ui.cli import main


def _git(repo: Path, *args: str, check: bool = True) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args], check=check, capture_output=True, text=True
    ).stdout.strip()


def _branch_exists(repo: Path, name: str) -> bool:
    return bool(_git(repo, "branch", "--list", name))


def _make_branch(repo: Path, session_id: str, fname: str) -> None:
    _git(repo, "checkout", "-q", "-b", f"agent6/{session_id}", "main")
    (repo / fname).write_text("x\n", encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", f"work {session_id}")
    _git(repo, "checkout", "-q", "main")


def _manifest(
    repo: Path, session_id: str, base: str, *, merged: bool, merged_tip: str = ""
) -> None:
    layout = SessionLayout(state_dir=resolved_state_dir(repo), session_id=session_id)
    layout.ensure()
    data: dict[str, object] = {
        "version": 2,
        "session_id": session_id,
        "base_sha": base,
        "base_branch": "main",
        "run_branch": f"agent6/{session_id}",
        "user_task": "t",
    }
    if merged:
        tip = merged_tip or _git(repo, "rev-parse", f"agent6/{session_id}", check=False)
        data["merged"] = {"into": "main", "sha": "0" * 40, "tip": tip}
    layout.manifest_path.write_text(json.dumps(data) + "\n", encoding="utf-8")


def test_runs_prune_classifies_branches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    _git(tmp_path, "init", "-q", "-b", "main")
    _git(tmp_path, "config", "user.email", "t@t")
    _git(tmp_path, "config", "user.name", "t")
    (tmp_path / "README.md").write_text("base\n", encoding="utf-8")
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-q", "-m", "init")
    base = _git(tmp_path, "rev-parse", "HEAD")

    # reachable-merged (--no-ff): git branch -d can delete it
    _make_branch(tmp_path, "reach11", "r.txt")
    _git(tmp_path, "merge", "--no-ff", "-m", "merge reach", "agent6/reach11")
    _manifest(tmp_path, "reach11", base, merged=True)
    # squash-merged: content in main but the branch is unreachable
    _make_branch(tmp_path, "sqush11", "s.txt")
    _git(tmp_path, "merge", "--squash", "agent6/sqush11")
    _git(tmp_path, "commit", "-q", "-m", "squash sqush11")
    _manifest(tmp_path, "sqush11", base, merged=True)
    # genuinely unmerged
    _make_branch(tmp_path, "unmrg11", "u.txt")
    _manifest(tmp_path, "unmrg11", base, merged=False)

    rc = main(["sessions", "prune"])
    cap = capsys.readouterr()
    text = cap.out + cap.err
    assert rc == 0
    assert not _branch_exists(tmp_path, "agent6/reach11")  # safely deleted
    assert _branch_exists(tmp_path, "agent6/sqush11")  # kept (unreachable squash)
    assert _branch_exists(tmp_path, "agent6/unmrg11")  # kept (unmerged)
    assert "deleted agent6/reach11" in text
    assert "squash-merged" in text  # sqush11 classification
    assert "NOT merged" in text  # unmrg11 classification
    assert cap.out.index("kept agent6/sqush11") < cap.out.index("[agent6] deleted 1; kept 2")


def test_runs_commits_and_diff_after_prune_say_where_the_work_went(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # A pruned (deleted) run branch: diff/commits must not leak a raw git fatal.
    # The manifest recorded the squash merge, so report it instead.
    monkeypatch.chdir(tmp_path)
    _git(tmp_path, "init", "-q", "-b", "main")
    _git(tmp_path, "config", "user.email", "t@t")
    _git(tmp_path, "config", "user.name", "t")
    (tmp_path / "README.md").write_text("base\n", encoding="utf-8")
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-q", "-m", "init")
    base = _git(tmp_path, "rev-parse", "HEAD")
    # Manifest says the run branch existed and was squash-merged, but the branch
    # itself is gone (never created here = pruned).
    _manifest(tmp_path, "gone11", base, merged=True)

    assert main(["sessions", "commits", "gone11"]) == 0
    out = capsys.readouterr().out
    # The manifest stamp records where, not with which strategy.
    assert "was pruned" in out and "merged into main" in out

    assert main(["sessions", "diff", "gone11"]) == 0
    assert "was pruned" in capsys.readouterr().out


def test_a_run_that_committed_nothing_is_not_reported_as_deleted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A run that made no commits never cuts its branch (nor its chain ref), so
    "no longer exists (deleted...)" told the operator a branch had been removed
    that was never created. Observed on a run whose task started a server and
    edited nothing."""
    monkeypatch.chdir(tmp_path)
    _git(tmp_path, "init", "-q", "-b", "main")
    _git(tmp_path, "config", "user.email", "t@t")
    _git(tmp_path, "config", "user.name", "t")
    (tmp_path / "README.md").write_text("base\n", encoding="utf-8")
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-q", "-m", "init")
    base = _git(tmp_path, "rev-parse", "HEAD")
    _manifest(tmp_path, "nocommit1", base, merged=False)  # no branch, no chain ref

    assert main(["sessions", "commits", "nocommit1"]) == 0
    out = capsys.readouterr().out
    assert "committed nothing" in out and "never cut" in out
    assert "no longer exists" not in out

    # Nothing to land is not a failure (a zero-commit branch says the same).
    assert main(["sessions", "merge", "nocommit1"]) == 0
    assert "nothing to merge: this run committed nothing" in capsys.readouterr().out


def test_a_deleted_branch_names_the_chain_ref_that_still_holds_the_work(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Deleting the run branch by hand leaves the commits on the chain ref, so
    the message that reports the missing branch says where they are."""
    monkeypatch.chdir(tmp_path)
    _git(tmp_path, "init", "-q", "-b", "main")
    _git(tmp_path, "config", "user.email", "t@t")
    _git(tmp_path, "config", "user.name", "t")
    (tmp_path / "README.md").write_text("base\n", encoding="utf-8")
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-q", "-m", "init")
    base = _git(tmp_path, "rev-parse", "HEAD")
    _make_branch(tmp_path, "gonebr1", "a.txt")
    tip = _git(tmp_path, "rev-parse", "agent6/gonebr1")
    _git(tmp_path, "update-ref", chain_ref_for("gonebr1"), tip)
    _git(tmp_path, "branch", "-D", "agent6/gonebr1")
    _manifest(tmp_path, "gonebr1", base, merged=False)

    assert main(["sessions", "commits", "gonebr1"]) == 0
    out = capsys.readouterr().out
    assert chain_ref_for("gonebr1") in out
    assert "committed nothing" not in out


def test_runs_prune_delete_squashed_removes_only_confirmed_squash_merged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # --delete-squashed force-deletes a manifest-confirmed squash-merged branch
    # (content-safe in the base commit) and prints an undelete hint; an unmerged
    # branch is NEVER force-deleted.
    monkeypatch.chdir(tmp_path)
    _git(tmp_path, "init", "-q", "-b", "main")
    _git(tmp_path, "config", "user.email", "t@t")
    _git(tmp_path, "config", "user.name", "t")
    (tmp_path / "README.md").write_text("base\n", encoding="utf-8")
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-q", "-m", "init")
    base = _git(tmp_path, "rev-parse", "HEAD")

    _make_branch(tmp_path, "sqush22", "s.txt")
    sha = _git(tmp_path, "rev-parse", "agent6/sqush22")
    _git(tmp_path, "merge", "--squash", "agent6/sqush22")
    _git(tmp_path, "commit", "-q", "-m", "squash sqush22")
    _manifest(tmp_path, "sqush22", base, merged=True)
    _make_branch(tmp_path, "unmrg22", "u.txt")
    _manifest(tmp_path, "unmrg22", base, merged=False)

    rc = main(["sessions", "prune", "--delete-squashed"])
    cap = capsys.readouterr()
    text = cap.out + cap.err
    assert rc == 0
    assert not _branch_exists(tmp_path, "agent6/sqush22")  # force-deleted (content safe)
    assert _branch_exists(tmp_path, "agent6/unmrg22")  # unmerged: never force-deleted
    assert "deleted agent6/sqush22 (squash-merged into main)" in text
    assert f"undelete: git branch agent6/sqush22 {sha[:12]}" in text  # recoverable
    assert "(1 squash-merged)" in text


def test_runs_prune_from_non_base_does_not_mislabel_merge_as_squash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    _git(tmp_path, "init", "-q", "-b", "main")
    _git(tmp_path, "config", "user.email", "t@t")
    _git(tmp_path, "config", "user.name", "t")
    (tmp_path / "README.md").write_text("base\n", encoding="utf-8")
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-q", "-m", "init")
    base = _git(tmp_path, "rev-parse", "HEAD")
    # merge-merged into main
    _make_branch(tmp_path, "reach22", "r.txt")
    _git(tmp_path, "merge", "--no-ff", "-m", "merge reach", "agent6/reach22")
    _manifest(tmp_path, "reach22", base, merged=True)
    # switch to a branch cut from the ORIGINAL base, so reach22 is unreachable here
    _git(tmp_path, "checkout", "-q", "-b", "feature", base)

    rc = main(["sessions", "prune"])
    cap = capsys.readouterr()
    text = cap.out + cap.err
    assert rc == 0
    assert _branch_exists(tmp_path, "agent6/reach22")  # not reachable from feature, so kept
    assert "not reachable from 'feature'" in text  # accurate reason
    assert "squash-merged" not in text  # the merge must NOT be mislabeled as squash


def test_runs_prune_no_branches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    _git(tmp_path, "init", "-q", "-b", "main")
    _git(tmp_path, "config", "user.email", "t@t")
    _git(tmp_path, "config", "user.name", "t")
    (tmp_path / "README.md").write_text("base\n", encoding="utf-8")
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-q", "-m", "init")
    rc = main(["sessions", "prune"])
    assert rc == 0
    assert "no agent6/* run branches" in capsys.readouterr().out


def test_runs_prune_delete_squashed_keeps_a_branch_that_advanced_after_the_merge(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The one sanctioned force-delete must prove the CURRENT tip is what was
    merged. A run that is squash-merged and then resumed keeps committing on the
    same branch under a stale merge stamp; force-deleting it destroys commits
    that exist in no other ref (reflog-only recovery)."""
    monkeypatch.chdir(tmp_path)
    _git(tmp_path, "init", "-q", "-b", "main")
    _git(tmp_path, "config", "user.email", "t@t")
    _git(tmp_path, "config", "user.name", "t")
    (tmp_path / "README.md").write_text("base\n", encoding="utf-8")
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-q", "-m", "init")
    base = _git(tmp_path, "rev-parse", "HEAD")

    _make_branch(tmp_path, "resmd33", "s.txt")
    merged_tip = _git(tmp_path, "rev-parse", "agent6/resmd33")
    _git(tmp_path, "merge", "--squash", "agent6/resmd33")
    _git(tmp_path, "commit", "-q", "-m", "squash resmd33")
    _manifest(tmp_path, "resmd33", base, merged=True, merged_tip=merged_tip)
    # The operator resumes the run: a new commit lands on the run branch only.
    _git(tmp_path, "checkout", "-q", "agent6/resmd33")
    (tmp_path / "after.txt").write_text("post-merge work\n", encoding="utf-8")
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-q", "-m", "agent6 iter 2: post-merge follow-up work")
    after = _git(tmp_path, "rev-parse", "HEAD")
    _git(tmp_path, "checkout", "-q", "main")

    rc = main(["sessions", "prune", "--delete-squashed"])
    text = "".join(capsys.readouterr())
    assert rc == 0
    assert _branch_exists(tmp_path, "agent6/resmd33")  # the post-merge commit survives
    assert _git(tmp_path, "rev-parse", "agent6/resmd33") == after
    assert "advanced since the merge" in text


def test_runs_prune_says_why_a_pre_tip_manifest_is_kept(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A run merged before agent6 recorded the merged tip cannot be confirmed,
    so --delete-squashed keeps it. The message must say that and name the manual
    command -- it told the operator to run `sessions prune --delete-squashed`, the
    very command that had just skipped the branch."""
    monkeypatch.chdir(tmp_path)
    _git(tmp_path, "init", "-q", "-b", "main")
    _git(tmp_path, "config", "user.email", "t@t")
    _git(tmp_path, "config", "user.name", "t")
    (tmp_path / "README.md").write_text("base\n", encoding="utf-8")
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-q", "-m", "init")
    base = _git(tmp_path, "rev-parse", "HEAD")

    _make_branch(tmp_path, "pretip1", "s.txt")
    _git(tmp_path, "merge", "--squash", "agent6/pretip1")
    _git(tmp_path, "commit", "-q", "-m", "squash pretip1")
    # A manifest written before MergeStamp.tip existed: merged, but no tip.
    layout = SessionLayout(state_dir=resolved_state_dir(tmp_path), session_id="pretip1")
    layout.ensure()
    layout.manifest_path.write_text(
        json.dumps(
            {
                "version": 2,
                "session_id": "pretip1",
                "base_sha": base,
                "base_branch": "main",
                "run_branch": "agent6/pretip1",
                "user_task": "t",
                "merged": {"into": "main", "sha": "0" * 40},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    assert main(["sessions", "prune", "--delete-squashed"]) == 0
    text = "".join(capsys.readouterr())
    assert _branch_exists(tmp_path, "agent6/pretip1")  # unconfirmed: kept
    assert "no recorded merge tip" in text
    assert "git branch -D agent6/pretip1" in text
    # It must NOT tell the operator to re-run the command that just skipped it.
    assert "--delete-squashed, or:" not in text


def test_plain_prune_never_points_at_a_flag_that_would_skip_the_branch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The advice loop was only closed on the path the operator was already on.
    Plain `sessions prune` still advertised --delete-squashed for a branch that
    command refuses -- and every manifest written before the tip stamp is such a
    branch, so it was the default. Same for a recorded tip whose base branch is
    gone: the confirmation needs both."""
    monkeypatch.chdir(tmp_path)
    _git(tmp_path, "init", "-q", "-b", "main")
    _git(tmp_path, "config", "user.email", "t@t")
    _git(tmp_path, "config", "user.name", "t")
    (tmp_path / "README.md").write_text("base\n", encoding="utf-8")
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-q", "-m", "init")
    base = _git(tmp_path, "rev-parse", "HEAD")

    _make_branch(tmp_path, "pretip2", "s.txt")
    _git(tmp_path, "merge", "--squash", "agent6/pretip2")
    _git(tmp_path, "commit", "-q", "-m", "squash pretip2")
    layout = SessionLayout(state_dir=resolved_state_dir(tmp_path), session_id="pretip2")
    layout.ensure()
    layout.manifest_path.write_text(
        json.dumps(
            {
                "version": 2,
                "session_id": "pretip2",
                "base_sha": base,
                "base_branch": "main",
                "run_branch": "agent6/pretip2",
                "user_task": "t",
                "merged": {"into": "main", "sha": "0" * 40},  # pre-tip manifest
            }
        )
        + "\n",
        encoding="utf-8",
    )

    assert main(["sessions", "prune"]) == 0
    plain = "".join(capsys.readouterr())
    assert "no recorded merge tip" in plain
    assert "git branch -D agent6/pretip2" in plain
    assert "--delete-squashed, or:" not in plain

    # A recorded tip is not enough on its own: --delete-squashed also needs the
    # base branch to confirm against, so a deleted base must not be advertised.
    _manifest(tmp_path, "pretip2", base, merged=True)
    _git(tmp_path, "checkout", "-q", "-b", "elsewhere")
    _git(tmp_path, "branch", "-q", "-m", "main", "renamed")
    assert main(["sessions", "prune", "--delete-squashed"]) == 0
    no_base = "".join(capsys.readouterr())
    assert _branch_exists(tmp_path, "agent6/pretip2")
    assert "--delete-squashed, or:" not in no_base


def test_runs_dir_prints_the_state_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """One bare line so it composes: `ls "$(agent6 sessions dir)"`."""
    repo = tmp_path / "repo"
    repo.mkdir()
    monkeypatch.chdir(repo)
    assert main(["sessions", "dir"]) == 0
    printed = capsys.readouterr().out.strip()
    assert printed == str(resolved_state_dir(repo))
    assert "\n" not in printed


def test_runs_rm_deletes_history_but_refuses_a_live_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """`rm` is the HISTORY verb (prune is the branch verb), and it will not
    delete a run that is still live -- the worker would keep writing into a
    directory the operator believes is gone."""
    import os

    from agent6.sessions.ipc import write_worker_pid

    repo = tmp_path / "repo"
    repo.mkdir()
    monkeypatch.chdir(repo)
    runs = resolved_state_dir(repo) / "sessions" / "runs"
    live, dead = runs / "live-run-AAAA11", runs / "dead-run-BBBB22"
    for d in (live, dead):
        d.mkdir(parents=True)
        (d / "logs.jsonl").write_text(
            '{"type": "session.start", "mode": "run"}\n', encoding="utf-8"
        )
    write_worker_pid(live, os.getpid())  # this test process is genuinely alive
    with (dead / "logs.jsonl").open("a", encoding="utf-8") as fh:
        fh.write('{"type": "session.end", "reason": "finish_session", "all_passed": true}\n')

    assert main(["sessions", "rm", "live-run"]) == 2
    assert "still live" in capsys.readouterr().err
    assert live.is_dir()

    assert main(["sessions", "rm", "dead-run"]) == 0
    assert "removed dead-run-BBBB22" in capsys.readouterr().out
    assert not dead.exists()


def test_runs_rm_names_the_kept_branch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """rm deletes history and the chain ref but never the visible branch (the
    unmerged work's anchor; only prune's kept-list owns branch deletion) - and
    it SAYS so: the old message named only the chain ref, leaving the operator
    to discover an orphan branch later."""
    import subprocess

    repo = tmp_path / "repo"
    repo.mkdir()
    monkeypatch.chdir(repo)
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.email=t@t",
            "-c",
            "user.name=t",
            "commit",
            "-q",
            "--allow-empty",
            "-m",
            "seed",
        ],
        cwd=repo,
        check=True,
    )
    subprocess.run(
        ["git", "update-ref", "refs/agent6/gone-run-CCCC33/head", "HEAD"], cwd=repo, check=True
    )
    subprocess.run(["git", "branch", "agent6/gone-run-CCCC33"], cwd=repo, check=True)
    d = resolved_state_dir(repo) / "sessions" / "runs" / "gone-run-CCCC33"
    d.mkdir(parents=True)
    (d / "logs.jsonl").write_text(
        '{"type": "session.start", "mode": "run"}\n'
        '{"type": "session.end", "reason": "finish_session", "all_passed": true}\n',
        encoding="utf-8",
    )

    assert main(["sessions", "rm", "gone-run-CCCC33"]) == 0
    out = capsys.readouterr().out
    assert "branch agent6/gone-run-CCCC33 kept" in out
    assert "prune" in out
    ref = subprocess.run(
        ["git", "rev-parse", "--verify", "-q", "refs/agent6/gone-run-CCCC33/head"],
        cwd=repo,
        capture_output=True,
        check=False,
    )
    assert ref.returncode != 0  # the chain ref is gone
    br = subprocess.run(
        ["git", "rev-parse", "--verify", "-q", "refs/heads/agent6/gone-run-CCCC33"],
        cwd=repo,
        capture_output=True,
        check=False,
    )
    assert br.returncode == 0  # the branch survives


def test_runs_rm_asks_clears_the_bucket(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Asks accumulate one state dir per directory they are run from, so the
    bucket gets its own sweep; mixing it with a run id is refused."""
    repo = tmp_path / "repo"
    repo.mkdir()
    monkeypatch.chdir(repo)
    asks = resolved_state_dir(repo) / "sessions" / "asks"
    for name in ("ask-one", "ask-two"):
        (asks / name).mkdir(parents=True)
    assert main(["sessions", "rm", "some-id", "--asks"]) == 2
    assert main(["sessions", "rm", "--asks"]) == 0
    assert "removed 2 asks" in capsys.readouterr().out
    assert not asks.exists()


def _chain_ref_exists(repo: Path, session_id: str) -> bool:
    return bool(_git(repo, "for-each-ref", chain_ref_for(session_id), check=False))


def test_prune_drops_chain_refs_of_confirmed_merged_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A merged run's refs/agent6/<id> falls with the same rules as branches:
    reachable-merged deletes outright, squash-merged only with
    --delete-squashed while the ref matches the recorded tip, unmerged and
    manifest-less (machine) refs are kept silently."""
    monkeypatch.chdir(tmp_path)
    _git(tmp_path, "init", "-q", "-b", "main")
    _git(tmp_path, "config", "user.email", "t@t")
    _git(tmp_path, "config", "user.name", "t")
    (tmp_path / "README.md").write_text("base\n", encoding="utf-8")
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-q", "-m", "init")
    base = _git(tmp_path, "rev-parse", "HEAD")

    # merged-reachable: work committed on main itself (a merge landed it).
    _make_branch(tmp_path, "run-RCH111", "a.txt")
    _git(tmp_path, "merge", "-q", "--no-ff", "-m", "land", "agent6/run-RCH111")
    _git(tmp_path, "update-ref", chain_ref_for("run-RCH111"), "agent6/run-RCH111")
    _manifest(tmp_path, "run-RCH111", base, merged=True)
    # squash-merged: content in main via squash; the ref is unreachable.
    _make_branch(tmp_path, "run-SQH111", "b.txt")
    _git(tmp_path, "update-ref", chain_ref_for("run-SQH111"), "agent6/run-SQH111")
    _git(tmp_path, "branch", "-D", "agent6/run-SQH111")
    _manifest(
        tmp_path,
        "run-SQH111",
        base,
        merged=True,
        merged_tip=_git(tmp_path, "rev-parse", chain_ref_for("run-SQH111")),
    )
    # unmerged: the ref is the work's only anchor; must survive both passes.
    _make_branch(tmp_path, "run-UNM111", "c.txt")
    _git(tmp_path, "update-ref", chain_ref_for("run-UNM111"), "agent6/run-UNM111")
    _git(tmp_path, "branch", "-D", "agent6/run-UNM111")
    _manifest(tmp_path, "run-UNM111", base, merged=False)
    # machine chain: no run manifest; never touched, never mentioned.
    _git(tmp_path, "update-ref", chain_ref_for("machine-box1"), base)

    assert main(["sessions", "prune"]) == 0
    out = capsys.readouterr().out
    assert not _chain_ref_exists(tmp_path, "run-RCH111")
    assert _chain_ref_exists(tmp_path, "run-SQH111")  # squash needs the flag
    assert _chain_ref_exists(tmp_path, "run-UNM111")
    assert _chain_ref_exists(tmp_path, "machine-box1")
    assert f"deleted {chain_ref_for('run-RCH111')}" in out
    assert "machine-box1" not in out

    assert main(["sessions", "prune", "--delete-squashed"]) == 0
    assert not _chain_ref_exists(tmp_path, "run-SQH111")
    assert _chain_ref_exists(tmp_path, "run-UNM111")
    assert _chain_ref_exists(tmp_path, "machine-box1")


def test_prune_reaches_chain_refs_with_no_run_branches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Chain refs are prunable whether or not a run BRANCH survives.

    The command returned early on an empty branch list, so with
    `branch_per_run` off -- where there is never one -- a merged run's chain ref
    could not be pruned at all, and neither could the refs an earlier pass kept
    once it had deleted the last branch.
    """
    monkeypatch.chdir(tmp_path)
    _git(tmp_path, "init", "-q", "-b", "main")
    _git(tmp_path, "config", "user.email", "t@t")
    _git(tmp_path, "config", "user.name", "t")
    (tmp_path / "README.md").write_text("base\n", encoding="utf-8")
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-q", "-m", "init")
    base = _git(tmp_path, "rev-parse", "HEAD")
    # A merged run whose branch is already gone: the chain ref is all that is left.
    _make_branch(tmp_path, "run-NOBR11", "a.txt")
    _git(tmp_path, "merge", "-q", "--no-ff", "-m", "land", "agent6/run-NOBR11")
    _git(tmp_path, "update-ref", chain_ref_for("run-NOBR11"), "agent6/run-NOBR11")
    _git(tmp_path, "branch", "-D", "agent6/run-NOBR11")
    _manifest(tmp_path, "run-NOBR11", base, merged=True)
    assert not _git(tmp_path, "branch", "--list", "agent6/*")

    assert main(["sessions", "prune"]) == 0

    assert not _chain_ref_exists(tmp_path, "run-NOBR11")
    assert f"deleted {chain_ref_for('run-NOBR11')}" in capsys.readouterr().out


def test_prune_with_nothing_at_all_says_so(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A repo with neither run branches nor chain refs still answers plainly."""
    monkeypatch.chdir(tmp_path)
    _git(tmp_path, "init", "-q", "-b", "main")
    _git(tmp_path, "config", "user.email", "t@t")
    _git(tmp_path, "config", "user.name", "t")
    (tmp_path / "README.md").write_text("base\n", encoding="utf-8")
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-q", "-m", "init")

    assert main(["sessions", "prune"]) == 0
    assert "nothing to prune" in capsys.readouterr().out


def test_rm_reports_a_deletion_failure_instead_of_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """rmtree ran with ignore_errors=True and the command printed the removal
    line with rc 0 while the directory survived; the chain-ref cleanup then
    ran against a session that still existed. A deletion failure is rc 1
    naming the error, and the session dir (and its ref) stay untouched."""
    repo = tmp_path / "repo"
    repo.mkdir()
    monkeypatch.chdir(repo)
    runs = resolved_state_dir(repo) / "sessions" / "runs"
    target = runs / "stuck-run-CCCC33"
    target.mkdir(parents=True)
    (target / "logs.jsonl").write_text(
        '{"type": "session.start", "mode": "run"}\n'
        '{"type": "session.end", "reason": "finish_session", "all_passed": true}\n',
        encoding="utf-8",
    )
    runs.chmod(0o500)  # the parent refuses the unlink
    try:
        rc = main(["sessions", "rm", "stuck-run"])
    finally:
        runs.chmod(0o700)
    assert rc == 1
    err = capsys.readouterr().err
    assert "could not remove" in err
    assert target.is_dir()  # nothing pretended otherwise


def _fork_with_worktree(repo: Path, session_id: str, *, merged: bool, record: bool = True) -> Path:
    """A fork session as `create_fork` leaves it: a linked worktree of *repo*
    under `[parallel].workdir` and a manifest naming it (`record=False`: the
    worktree of a session whose record `sessions rm` deleted)."""
    from agent6.app.parallel import subordinate_workdir_root
    from agent6.config import Config
    from agent6.git_ops import add_worktree

    base = _git(repo, "rev-parse", "HEAD")
    worktree = subordinate_workdir_root(Config(), repo, session_id)
    add_worktree(repo, worktree, base)
    _git(repo, "branch", f"agent6/{session_id}", base)
    if record:
        layout = SessionLayout(state_dir=resolved_state_dir(repo), session_id=session_id)
        layout.ensure()
        data: dict[str, object] = {
            "version": 3,
            "session_id": session_id,
            "mode": "run",
            "base_sha": base,
            "base_branch": "main",
            "run_branch": f"agent6/{session_id}",
            "user_task": "t",
            "worktree": str(worktree),
        }
        if merged:
            data["merged"] = {"into": "main", "sha": base, "tip": base}
        layout.manifest_path.write_text(json.dumps(data) + "\n", encoding="utf-8")
    return worktree


def test_prune_removes_the_worktree_of_a_merged_fork_and_keeps_an_unmerged_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A fork's worktree is its checkout until its work lands: prune removes it
    (and git's record of it) once the fork's manifest carries the merge stamp,
    and keeps an unmerged fork's, saying so. The lane sweep treated a
    worktree dir as an empty fan-out group and deleted it on every prune."""
    monkeypatch.chdir(tmp_path)
    _git(tmp_path, "init", "-q", "-b", "main")
    (tmp_path / "README.md").write_text("base\n", encoding="utf-8")
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-q", "-m", "init")
    merged = _fork_with_worktree(tmp_path, "fork-merged11", merged=True)
    kept = _fork_with_worktree(tmp_path, "fork-unmrgd11", merged=False)
    (kept / "wip.txt").write_text("in flight\n", encoding="utf-8")

    assert main(["sessions", "prune"]) == 0
    out = capsys.readouterr().out
    assert not merged.exists()
    assert (kept / "wip.txt").read_text(encoding="utf-8") == "in flight\n"
    listed = _git(tmp_path, "worktree", "list", "--porcelain")
    assert str(merged) not in listed
    assert str(kept) in listed
    assert "removed fork-merged11's worktree (merged)" in out
    assert "kept fork-unmrgd11's worktree (unmerged)" in out


def test_prune_leaves_a_worktree_no_manifest_records_alone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Only a worktree a session manifest records is agent6's to remove. A
    linked worktree the operator put under agent6's workdir scope, and the
    worktree of a fork whose record is gone, keep their uncommitted work. The
    sweep deleted any dir with a `.git` file there whose name matched no
    session ("no session record")."""
    from agent6.app.parallel import subordinate_workdir_root
    from agent6.config import Config
    from agent6.git_ops import add_worktree

    monkeypatch.chdir(tmp_path)
    _git(tmp_path, "init", "-q", "-b", "main")
    (tmp_path / "README.md").write_text("base\n", encoding="utf-8")
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-q", "-m", "init")
    foreign = subordinate_workdir_root(Config(), tmp_path, "my-experiment")
    add_worktree(tmp_path, foreign, _git(tmp_path, "rev-parse", "HEAD"))
    (foreign / "draft.txt").write_text("uncommitted\n", encoding="utf-8")
    forgotten = _fork_with_worktree(tmp_path, "fork-forgot11", merged=False, record=False)
    (forgotten / "wip.txt").write_text("in flight\n", encoding="utf-8")

    assert main(["sessions", "prune"]) == 0
    out = capsys.readouterr().out
    assert (foreign / "draft.txt").read_text(encoding="utf-8") == "uncommitted\n"
    assert (forgotten / "wip.txt").read_text(encoding="utf-8") == "in flight\n"
    listed = _git(tmp_path, "worktree", "list", "--porcelain")
    assert str(foreign) in listed and str(forgotten) in listed
    assert "my-experiment" not in out and "fork-forgot11's worktree" not in out


def test_rm_removes_the_worktree_its_manifest_records(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """`sessions rm <fork>` deletes the fork's worktree with its record (the
    one moment the ledger still names it), and git's record of the worktree."""
    monkeypatch.chdir(tmp_path)
    _git(tmp_path, "init", "-q", "-b", "main")
    (tmp_path / "README.md").write_text("base\n", encoding="utf-8")
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-q", "-m", "init")
    worktree = _fork_with_worktree(tmp_path, "fork-gone11", merged=False)

    assert main(["sessions", "rm", "fork-gone11"]) == 0
    out = capsys.readouterr().out
    assert not worktree.exists()
    assert str(worktree) not in _git(tmp_path, "worktree", "list", "--porcelain")
    assert "removed fork-gone11" in out and "worktree" in out


def _record(repo: Path, session_id: str, worktree: Path, *, merged: bool) -> Path:
    """A session manifest naming *worktree* (an `/undo` fork's shares its
    source's)."""
    base = _git(repo, "rev-parse", "HEAD")
    layout = SessionLayout(state_dir=resolved_state_dir(repo), session_id=session_id)
    layout.ensure()
    data: dict[str, object] = {
        "version": 3,
        "session_id": session_id,
        "mode": "run",
        "base_sha": base,
        "base_branch": "main",
        "run_branch": f"agent6/{session_id}",
        "user_task": "t",
        "worktree": str(worktree),
    }
    if merged:
        data["merged"] = {"into": "main", "sha": base, "tip": base}
    layout.manifest_path.write_text(json.dumps(data) + "\n", encoding="utf-8")
    return layout.session_dir


def test_a_worktree_stays_while_any_session_naming_it_still_needs_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Ownership is every manifest naming the path. A merged fork whose /undo
    child (same worktree) is unmerged keeps the worktree, for both prune and
    `rm` of the parent; a merged fork resumed after its merge (its branch
    moved past the stamp) or still live keeps its worktree too. Keyed on the
    merged parent alone, prune deleted the child's checkout."""
    import os

    from agent6.sessions.ipc import write_worker_pid

    monkeypatch.chdir(tmp_path)
    _git(tmp_path, "init", "-q", "-b", "main")
    (tmp_path / "README.md").write_text("base\n", encoding="utf-8")
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-q", "-m", "init")
    shared = _fork_with_worktree(tmp_path, "fork-parent11", merged=True)
    _record(tmp_path, "fork-child011", shared, merged=False)
    (shared / "wip.txt").write_text("child's\n", encoding="utf-8")
    moved = _fork_with_worktree(tmp_path, "fork-moved011", merged=True)
    (moved / "later.txt").write_text("after the merge\n", encoding="utf-8")
    base = _git(tmp_path, "rev-parse", "HEAD")
    after = _git(tmp_path, "commit-tree", f"{base}^{{tree}}", "-p", base, "-m", "a later leg")
    _git(tmp_path, "update-ref", "refs/heads/agent6/fork-moved011", after)
    live = _fork_with_worktree(tmp_path, "fork-live0011", merged=True)
    write_worker_pid(
        SessionLayout(
            state_dir=resolved_state_dir(tmp_path), session_id="fork-live0011"
        ).session_dir,
        os.getpid(),
    )

    assert main(["sessions", "prune"]) == 0
    out = capsys.readouterr().out
    assert (shared / "wip.txt").exists() and (moved / "later.txt").exists() and live.exists()
    assert "kept fork-child011's worktree (unmerged)" in out
    assert "kept fork-parent11's worktree (shared with fork-child011)" in out
    assert "kept fork-moved011's worktree (unmerged)" in out
    assert "kept fork-live0011's worktree (live)" in out

    assert main(["sessions", "rm", "fork-parent11"]) == 0
    out = capsys.readouterr().out
    assert (shared / "wip.txt").exists()
    assert "its worktree stays: fork-child011" in out
