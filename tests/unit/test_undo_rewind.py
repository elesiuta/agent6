# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""`/undo` puts the checkout back: the fork it cuts keeps the undone session's
checkout, so the tree has to be the checkpoint's again for every path the
run changed after it, with the operator's own files left alone and the later
chain commits kept on the undone session's ref."""

from __future__ import annotations

import json
import subprocess as sp
from pathlib import Path

import pytest

from agent6.app.fork import undo_fork
from agent6.app.reporter import Reporter
from agent6.config.layer import resolved_state_dir
from agent6.git_ops import chain_commit, chain_ref_for, chain_tip
from agent6.sessions.layout import SessionLayout, write_untracked_at_start
from agent6.sessions.lock import acquire_repo_writer, release_single_writer
from agent6.workflows._session_state import SessionSnapshot


def _git(repo: Path, *args: str) -> str:
    return sp.run(
        ["git", "-C", str(repo), *args], capture_output=True, text=True, check=True
    ).stdout.strip()


def _checkpoint(
    layout: SessionLayout, turn: int, *, head_sha: str, ops: int, at: int | None = None
) -> None:
    """A checkpoint at *turn* whose conversation holds *ops* operator messages
    (the task, then steers) and records *head_sha* as the workspace head.
    *at* is the file it sits in when that is not *turn* (a fork's seed: file 0
    holding its source's turn)."""
    messages: list[dict[str, object]] = [{"role": "user", "content": "do the thing"}]
    for i in range(1, ops):
        messages.append(
            {
                "role": "user",
                "content": [{"type": "text", "text": f"OPERATOR STEERING (x):\nsteer {i}"}],
            }
        )
    snap = SessionSnapshot(
        system="s",
        messages=messages,
        tool_calls=0,
        next_iteration=turn,
        root_task_id=None,
        original_task="do the thing",
        verify_command=(),
        head_sha=head_sha,
    )
    layout.checkpoint_path(turn if at is None else at).write_text(
        snap.model_dump_json(), encoding="utf-8"
    )


def _run_that_moved_on(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, str, str]:
    """A repo and a run whose turn-1 checkpoint sits at commit c1; after it the
    run's chain committed c2 (a.txt edited, gen.txt created), then edited
    a.txt again and created more.txt without committing, and the operator's
    untracked notes.md was there from the start. Returns (repo, c1, c2)."""
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    (repo / "a.txt").write_text("one\n", encoding="utf-8")
    (repo / "same.txt").write_text("same\n", encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "c1")
    c1 = _git(repo, "rev-parse", "HEAD")
    monkeypatch.chdir(repo)
    (repo / "notes.md").write_text("mine\n", encoding="utf-8")
    state_dir = resolved_state_dir(repo)
    layout = SessionLayout(state_dir=state_dir, session_id="run-AAAA11")
    layout.ensure()
    layout.manifest_path.write_text(
        json.dumps(
            {
                "version": 3,
                "session_id": "run-AAAA11",
                "mode": "run",
                "user_task": "do the thing",
                "base_sha": c1,
                "base_branch": "main",
                "run_branch": "agent6/run-AAAA11",
            }
        ),
        encoding="utf-8",
    )
    write_untracked_at_start(layout.session_dir, {"notes.md"})
    _checkpoint(layout, 1, head_sha=c1, ops=1)
    (repo / "a.txt").write_text("two\n", encoding="utf-8")
    (repo / "gen.txt").write_text("generated\n", encoding="utf-8")
    c2 = chain_commit(
        repo,
        "step",
        ref=chain_ref_for("run-AAAA11"),
        fallback_parent=c1,
        also_branch="agent6/run-AAAA11",
        exclude={"notes.md"},
    )
    assert c2 is not None
    _checkpoint(layout, 2, head_sha=c2, ops=2)
    (repo / "a.txt").write_text("three\n", encoding="utf-8")
    (repo / "more.txt").write_text("in flight\n", encoding="utf-8")
    (layout.session_dir / "loop_state.json").write_text(
        layout.checkpoint_path(2).read_text(encoding="utf-8"), encoding="utf-8"
    )
    return repo, c1, c2


def test_undo_puts_the_checkout_back_and_keeps_the_tree_as_it_stood(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The tree as it stands goes onto the run's ref first (the in-flight
    edit, the file that appeared mid-run), then every tracked path that
    differs from turn 1's tree is put back: the committed edit, the committed
    new file, the in-flight edit and file. The operator's untracked-at-start
    file and a file the run never touched stay; HEAD and the index stay; the
    later commit stays on the run's ref and branch; the fork's chain starts at
    turn 1's sha. Rewinding straight from the worktree deleted the in-flight
    work with no commit holding it."""
    repo, c1, c2 = _run_that_moved_on(tmp_path, monkeypatch)
    said: list[str] = []

    result = undo_fork(None, "run-AAAA11", cwd=repo, reporter=Reporter(said.append, said.append))

    assert result is not None
    child, undone_text = result
    assert undone_text == "steer 1"
    assert (repo / "a.txt").read_text(encoding="utf-8") == "one\n"
    assert not (repo / "gen.txt").exists() and not (repo / "more.txt").exists()
    assert (repo / "same.txt").read_text(encoding="utf-8") == "same\n"
    assert (repo / "notes.md").read_text(encoding="utf-8") == "mine\n"
    assert _git(repo, "rev-parse", "HEAD") == c1  # HEAD and the index untouched
    assert _git(repo, "diff", "--cached", "--name-only") == ""
    tip = chain_tip(repo, chain_ref_for("run-AAAA11"))
    assert tip is not None and tip != c2
    assert _git(repo, "rev-parse", f"{tip}^") == c2  # the pre-undo commit sits on c2
    assert _git(repo, "show", f"{tip}:a.txt") == "three"  # the in-flight edit, kept
    assert _git(repo, "show", f"{tip}:more.txt") == "in flight"  # the mid-run file, kept
    assert _git(repo, "rev-parse", "agent6/run-AAAA11") == tip
    assert chain_tip(repo, chain_ref_for(child)) == c1
    text = "\n".join(said)
    assert "a.txt" in text and "gen.txt" in text and "more.txt" in text
    assert f"commit {tip[:12]} on agent6/run-AAAA11" in text
    assert "HEAD and the index are untouched" in text


def test_undo_leaves_a_staged_copy_in_the_index_and_says_so(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The index is the operator's: a staged copy of a rewound file stays
    staged (git shows `MM`), and the notice says the index is untouched."""
    repo, _c1, _c2 = _run_that_moved_on(tmp_path, monkeypatch)
    _git(repo, "add", "a.txt")
    said: list[str] = []

    assert undo_fork(None, "run-AAAA11", cwd=repo, reporter=Reporter(said.append, said.append))
    assert (repo / "a.txt").read_text(encoding="utf-8") == "one\n"
    assert _git(repo, "diff", "--cached", "--name-only") == "a.txt"
    assert _git(repo, "status", "--short", "--", "a.txt") == "MM a.txt"
    assert any("HEAD and the index are untouched" in line for line in said)


def test_undo_refuses_while_another_live_run_drives_the_checkout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A rewind under a live worker pulls the tree out from under its next
    commit: refused, naming the run, with nothing committed, forked, or
    moved. The same holds for a worker of the undone session itself running in
    another process: only the process holding the checkout is exempt."""
    repo, _c1, c2 = _run_that_moved_on(tmp_path, monkeypatch)
    state_dir = resolved_state_dir(repo)
    for holder in ("other-LIVE11", "run-AAAA11"):
        said: list[str] = []
        fd = acquire_repo_writer(state_dir, holder)
        try:
            result = undo_fork(
                None, "run-AAAA11", cwd=repo, reporter=Reporter(said.append, said.append)
            )
        finally:
            release_single_writer(fd)
        assert result is None
        assert any(holder in line for line in said)
        assert (repo / "a.txt").read_text(encoding="utf-8") == "three\n"
        assert chain_tip(repo, chain_ref_for("run-AAAA11")) == c2
        assert not (state_dir / "lineage.jsonl").exists()


def test_undo_from_the_live_session_itself_is_allowed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The loop's /undo runs inside the worker that holds the checkout (its
    worker.pid is this process): its own lock is no obstacle."""
    import os

    from agent6.sessions.ipc import write_worker_pid

    repo, c1, _c2 = _run_that_moved_on(tmp_path, monkeypatch)
    state_dir = resolved_state_dir(repo)
    write_worker_pid(
        SessionLayout(state_dir=state_dir, session_id="run-AAAA11").session_dir, os.getpid()
    )
    fd = acquire_repo_writer(state_dir, "run-AAAA11")
    try:
        result = undo_fork(None, "run-AAAA11", cwd=repo, reporter=Reporter(print, print))
    finally:
        release_single_writer(fd)
    assert result is not None
    assert (repo / "a.txt").read_text(encoding="utf-8") == "one\n"
    assert _git(repo, "rev-parse", "HEAD") == c1


def test_undo_of_a_fork_whose_worktree_is_gone_refuses_before_creating_anything(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A recorded worktree that no longer exists is named, with the recovery,
    before any child exists: git run in the missing directory raised after
    the child had been created."""
    repo, _c1, _c2 = _run_that_moved_on(tmp_path, monkeypatch)
    state_dir = resolved_state_dir(repo)
    layout = SessionLayout(state_dir=state_dir, session_id="run-AAAA11")
    manifest = json.loads(layout.manifest_path.read_text(encoding="utf-8"))
    manifest["worktree"] = str(tmp_path / "gone-worktree")
    manifest["worktree_git_dir"] = str((repo / ".git").resolve())
    layout.manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    said: list[str] = []
    before = sorted(p.name for p in (state_dir / "sessions" / "runs").iterdir())

    result = undo_fork(None, "run-AAAA11", cwd=repo, reporter=Reporter(said.append, said.append))

    assert result is None
    assert any("gone-worktree" in line and "agent6 fork run-AAAA11" in line for line in said)
    assert any("its commits are on agent6/run-AAAA11" in line for line in said)
    assert sorted(p.name for p in (state_dir / "sessions" / "runs").iterdir()) == before
    assert not (state_dir / "lineage.jsonl").exists()

    # Pruned after a merge (branch gone): the refusal points at the merge that
    # landed the commits, as `sessions commits` does, never at the gone branch.
    manifest["merged"] = {"into": "main", "sha": _c2, "tip": _c2}
    layout.manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    _git(repo, "branch", "-D", "agent6/run-AAAA11")
    said.clear()
    assert (
        undo_fork(None, "run-AAAA11", cwd=repo, reporter=Reporter(said.append, said.append)) is None
    )
    assert any(f"merged into main as {_c2[:12]}" in line for line in said)
    assert not any("agent6/run-AAAA11" in line or "refs/agent6" in line for line in said)


def test_undo_names_the_turn_every_other_surface_names(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A fork's seed checkpoint sits in file 0 but holds its source's turn
    (`next_iteration`), the number the fork notice, `sessions show` and the
    child's manifest print. An /undo resolved at it said "turn 0" while the
    fork it cut read "@turn 3"; the notice and the commit that keeps the tree
    as it stood name the checkpoint's turn."""
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    (repo / "a.txt").write_text("one\n", encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "c1")
    c1 = _git(repo, "rev-parse", "HEAD")
    monkeypatch.chdir(repo)
    state_dir = resolved_state_dir(repo)
    fork = SessionLayout(state_dir=state_dir, session_id="fork-BBBB22")
    fork.ensure()
    fork.manifest_path.write_text(
        json.dumps(
            {
                "version": 3,
                "session_id": "fork-BBBB22",
                "mode": "run",
                "user_task": "do the thing",
                "base_sha": c1,
                "base_branch": "main",
                "run_branch": "agent6/fork-BBBB22",
                "parent_session_id": "src-AAAA11",
                "forked_from_turn": 3,
                "forked_from_sha": c1,
            }
        ),
        encoding="utf-8",
    )
    _checkpoint(fork, 3, head_sha=c1, ops=2, at=0)  # the seed: the source's turn 3
    _checkpoint(fork, 4, head_sha=c1, ops=3)  # the fork's own turn after a steer
    (repo / "a.txt").write_text("two\n", encoding="utf-8")
    said: list[str] = []

    result = undo_fork(None, "fork-BBBB22", cwd=repo, reporter=Reporter(said.append, said.append))

    assert result is not None
    child, _text = result
    text = "\n".join(said)
    assert f"back to turn 3 ({c1[:12]})" in text and "turn 0" not in text
    child_manifest = json.loads(
        SessionLayout(state_dir=state_dir, session_id=child).manifest_path.read_text(
            encoding="utf-8"
        )
    )
    assert child_manifest["forked_from_turn"] == 3
    subject = _git(repo, "log", "-1", "--format=%s", chain_ref_for("fork-BBBB22"))
    assert subject == "agent6 undo: the tree before turn 3 was taken back"


def test_an_undo_resolved_in_an_ancestor_keeps_the_undone_sessions_checkout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A fork resumed without a steer carries only its seed checkpoint, so
    its /undo resolves in the parent it was cut from. The child still works
    where the undone fork worked (its worktree, the checkout that was put
    back), never the parent's or the operator's: recording the parent's
    checkout handed the child's model a writable checkout it was never given."""
    from agent6.app.manifest import write_session_manifest
    from agent6.config import Config
    from agent6.git_ops import add_worktree

    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    (repo / "a.txt").write_text("one\n", encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "c1")
    c1 = _git(repo, "rev-parse", "HEAD")
    (repo / "a.txt").write_text("two\n", encoding="utf-8")
    _git(repo, "commit", "-qam", "c2")
    c2 = _git(repo, "rev-parse", "HEAD")
    monkeypatch.chdir(repo)
    state_dir = resolved_state_dir(repo)
    parent = SessionLayout(state_dir=state_dir, session_id="parent-AAAA11")
    parent.ensure()
    write_session_manifest(
        parent,
        session_id="parent-AAAA11",
        user_task="do the thing",
        base_sha=c1,
        base_branch="main",
        run_branch="agent6/parent-AAAA11",
        cfg=Config(),
        mode="run",
    )
    _checkpoint(parent, 1, head_sha=c1, ops=1)
    _checkpoint(parent, 2, head_sha=c2, ops=2)
    worktree = tmp_path / "fork-wt"
    add_worktree(repo, worktree, c2)
    fork = SessionLayout(state_dir=state_dir, session_id="fork-BBBB22")
    fork.ensure()
    write_session_manifest(
        fork,
        session_id="fork-BBBB22",
        user_task="do the thing",
        base_sha=c1,
        base_branch="main",
        run_branch="agent6/fork-BBBB22",
        cfg=Config(),
        mode="run",
        parent_session_id="parent-AAAA11",
        forked_from_turn=2,
        forked_from_sha=c2,
        worktree=worktree,
        worktree_git_dir=(repo / ".git").resolve(),
    )
    _checkpoint(fork, 0, head_sha=c2, ops=2)  # the seed: the parent's turn-2 conversation

    result = undo_fork(None, "fork-BBBB22", cwd=repo, reporter=Reporter(print, print))

    assert result is not None
    child, undone_text = result
    assert undone_text == "steer 1"
    child_manifest = json.loads(
        SessionLayout(state_dir=state_dir, session_id=child).manifest_path.read_text(
            encoding="utf-8"
        )
    )
    assert child_manifest["parent_session_id"] == "parent-AAAA11"
    assert child_manifest["worktree"] == str(worktree)
    assert (worktree / "a.txt").read_text(encoding="utf-8") == "one\n"  # the fork's, rewound
    assert (repo / "a.txt").read_text(encoding="utf-8") == "two\n"  # the operator's, untouched


def test_an_undo_fork_keeps_the_source_run_untracked_set(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An /undo fork continues in the source's checkout, where the run's own
    files still read untracked: chain commits never touch the index. Observing
    there made the child exclude the very work it continues, so every commit it
    made silently dropped those paths."""
    from agent6.sessions.layout import read_untracked_at_start

    repo, _c1, _c2 = _run_that_moved_on(tmp_path, monkeypatch)
    said: list[str] = []

    result = undo_fork(None, "run-AAAA11", cwd=repo, reporter=Reporter(said.append, said.append))

    assert result is not None
    child, _text = result
    child_dir = SessionLayout(state_dir=resolved_state_dir(repo), session_id=child).session_dir
    assert read_untracked_at_start(child_dir) == frozenset({"notes.md"}), (
        "the child inherits the operator's set, not the run's own output"
    )
