# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Tests for `git.auto_merge` (ui/cli/_finalize.py finalize_auto_merge)."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from agent6.app import finalize as finmod
from agent6.app.reporter import STDIO_REPORTER
from agent6.config.layer import load_effective
from agent6.git_ops import chain_ref_for
from agent6.paths import state_dir
from agent6.sessions.layout import SessionLayout


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args], check=True, capture_output=True, text=True
    ).stdout.strip()


def _setup_run_on_branch(
    tmp_path: Path, session_id: str, *, commits: list[tuple[str, str, str]], run_branch: str | None
) -> str:
    """Init a repo and put *commits* on agent6/<session_id> without moving the
    checkout off main (the end-of-run state: the chain advances refs only).
    Writes the manifest with *run_branch* recorded (None to simulate
    branch_per_run off). Returns base sha."""
    _git(tmp_path, "init", "-q", "-b", "main")
    _git(tmp_path, "config", "user.email", "t@t")
    _git(tmp_path, "config", "user.name", "t")
    (tmp_path / "README.md").write_text("base\n", encoding="utf-8")
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-q", "-m", "init")
    base_sha = _git(tmp_path, "rev-parse", "HEAD")
    branch = f"agent6/{session_id}"
    _git(tmp_path, "checkout", "-q", "-b", branch)
    for name, content, msg in commits:
        (tmp_path / name).write_text(content, encoding="utf-8")
        _git(tmp_path, "add", "-A")
        _git(tmp_path, "commit", "-q", "-m", msg)
    _git(tmp_path, "checkout", "-q", "main")
    # The worktree carries the run's work, exactly as a finished run leaves it.
    for name, content, _msg in commits:
        (tmp_path / name).write_text(content, encoding="utf-8")
    layout = SessionLayout(state_dir=state_dir(tmp_path), session_id=session_id)
    layout.ensure()
    layout.manifest_path.write_text(
        json.dumps(
            {
                "version": 2,
                "session_id": session_id,
                "base_sha": base_sha,
                "base_branch": "main",
                "run_branch": run_branch,
                "user_task": "implement the thing",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return base_sha


def test_auto_merge_squashes_and_lands_on_base(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    base = _setup_run_on_branch(
        tmp_path,
        "run-AM1111",
        commits=[
            ("a.txt", "a\n", "agent6 iter 1: add a"),
            ("b.txt", "b\n", "agent6 iter 2: add b"),
        ],
        run_branch="agent6/run-AM1111",
    )
    cfg = load_effective(tmp_path, None).config
    finmod.finalize_auto_merge(
        tmp_path,
        layout=SessionLayout(state_dir(tmp_path), "run-AM1111"),
        cfg=cfg,
        reporter=STDIO_REPORTER,
    )
    assert _git(tmp_path, "rev-parse", "--abbrev-ref", "HEAD") == "main"  # never switched
    assert _git(tmp_path, "rev-list", "--count", f"{base}..main") == "1"  # one squash commit
    assert _git(tmp_path, "status", "--porcelain") == ""  # index+worktree brought forward
    m = json.loads(
        (state_dir(tmp_path) / "sessions" / "runs" / "run-AM1111" / "manifest.json").read_text()
    )
    assert m["merged"]["into"] == "main"
    assert m["merged"]["sha"]


def test_auto_merge_lands_the_hidden_chain_ref(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """branch_per_run off records no branch; auto_merge merges the run's
    refs/agent6/<id> chain ref into the base instead."""
    monkeypatch.chdir(tmp_path)
    base = _setup_run_on_branch(
        tmp_path,
        "run-AMREF1",
        commits=[("a.txt", "a\n", "agent6 iter 1: add a")],
        run_branch=None,
    )
    _git(tmp_path, "update-ref", chain_ref_for("run-AMREF1"), "agent6/run-AMREF1")
    _git(tmp_path, "checkout", "-q", "--detach")
    _git(tmp_path, "branch", "-D", "agent6/run-AMREF1")  # only the chain ref remains
    _git(tmp_path, "checkout", "-q", "main")
    cfg = load_effective(tmp_path, None).config
    git2 = cfg.git.model_copy(update={"auto_merge": True})
    finmod.finalize_auto_merge(
        tmp_path,
        layout=SessionLayout(state_dir(tmp_path), "run-AMREF1"),
        cfg=cfg.model_copy(update={"git": git2}),
        reporter=STDIO_REPORTER,
    )
    assert _git(tmp_path, "rev-list", "--count", f"{base}..main") == "1"
    m = json.loads(
        (state_dir(tmp_path) / "sessions" / "runs" / "run-AMREF1" / "manifest.json").read_text()
    )
    assert m["merged"]["into"] == "main"


def test_auto_merge_noop_without_run_branch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    base = _setup_run_on_branch(
        tmp_path,
        "run-AM2222",
        commits=[("a.txt", "a\n", "agent6 iter 1: add a")],
        run_branch=None,  # branch_per_run was off
    )
    cfg = load_effective(tmp_path, None).config
    # On main (no run branch); the helper must no-op without crashing.
    _git(tmp_path, "checkout", "-q", "main")
    finmod.finalize_auto_merge(
        tmp_path,
        layout=SessionLayout(state_dir(tmp_path), "run-AM2222"),
        cfg=cfg,
        reporter=STDIO_REPORTER,
    )
    assert _git(tmp_path, "rev-list", "--count", f"{base}..main") == "0"  # nothing merged


def test_auto_merge_conflict_keeps_run_branch_intact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    _setup_run_on_branch(
        tmp_path,
        "run-AM3333",
        commits=[("conflict.txt", "from-run\n", "agent6 iter 1: edit")],
        run_branch="agent6/run-AM3333",
    )
    # Make base diverge so the squash conflicts on the same file.
    (tmp_path / "conflict.txt").write_text("from-base\n", encoding="utf-8")
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-q", "-m", "base edits the same file")
    diverged = _git(tmp_path, "rev-parse", "HEAD")
    cfg = load_effective(tmp_path, None).config
    finmod.finalize_auto_merge(
        tmp_path,
        layout=SessionLayout(state_dir(tmp_path), "run-AM3333"),
        cfg=cfg,
        reporter=STDIO_REPORTER,
    )
    err = capsys.readouterr().err
    assert "conflict" in err.lower()
    assert _git(tmp_path, "status", "--porcelain") == ""  # nothing touched, no partial merge
    assert _git(tmp_path, "rev-parse", "--abbrev-ref", "HEAD") == "main"  # never switched
    # the run branch still has its commit, and the conflicted merge advanced
    # main by nothing (a clean tree alone would also pass with a LANDED merge)
    assert "agent6 iter 1: edit" in _git(tmp_path, "log", "--oneline", "agent6/run-AM3333")
    assert _git(tmp_path, "rev-parse", "main") == diverged


def test_auto_merge_skips_when_base_branch_is_gone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    _setup_run_on_branch(
        tmp_path,
        "run-GONE11",
        commits=[("a.txt", "a\n", "agent6 iter 1: add a")],
        run_branch="agent6/run-GONE11",
    )
    # operator deleted the base branch mid-run (detach first; -D refuses on HEAD)
    _git(tmp_path, "checkout", "-q", "--detach")
    _git(tmp_path, "branch", "-D", "main")
    cfg = load_effective(tmp_path, None).config
    finmod.finalize_auto_merge(
        tmp_path,
        layout=SessionLayout(state_dir(tmp_path), "run-GONE11"),
        cfg=cfg,
        reporter=STDIO_REPORTER,
    )
    assert _git(tmp_path, "branch", "--list", "main") == ""  # base NOT fabricated
    manifest = json.loads(
        (state_dir(tmp_path) / "sessions" / "runs" / "run-GONE11" / "manifest.json").read_text()
    )
    assert manifest.get("merged") is None  # no phantom merge recorded
    assert "failed" in capsys.readouterr().err.lower()


def test_auto_prune_deletes_reachable_merge_branch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    _setup_run_on_branch(
        tmp_path,
        "run-AP1111",
        commits=[("a.txt", "a\n", "agent6 iter 1: add a")],
        run_branch="agent6/run-AP1111",
    )
    cfg = load_effective(tmp_path, None).config
    git2 = cfg.git.model_copy(
        update={"auto_merge": True, "auto_prune": True, "merge_strategy": "merge"}
    )
    cfg2 = cfg.model_copy(update={"git": git2})
    finmod.finalize_auto_merge(
        tmp_path,
        layout=SessionLayout(state_dir(tmp_path), "run-AP1111"),
        cfg=cfg2,
        reporter=STDIO_REPORTER,
    )
    assert _git(tmp_path, "branch", "--list", "agent6/run-AP1111") == ""  # pruned (reachable)


def test_auto_prune_follows_a_recorded_noop_merge(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A recorded merge takes the same post-merge path whatever it added:
    auto_prune ran only after a moved target, so the branch of a run whose
    first merge added nothing (main already contained it) stayed behind."""
    monkeypatch.chdir(tmp_path)
    _setup_run_on_branch(
        tmp_path,
        "run-AP3333",
        commits=[("a.txt", "a\n", "agent6 iter 1: add a")],
        run_branch="agent6/run-AP3333",
    )
    _git(tmp_path, "update-ref", "refs/heads/main", "agent6/run-AP3333")  # main has it already
    cfg = load_effective(tmp_path, None).config
    git2 = cfg.git.model_copy(
        update={"auto_merge": True, "auto_prune": True, "merge_strategy": "squash"}
    )
    finmod.finalize_auto_merge(
        tmp_path,
        layout=SessionLayout(state_dir(tmp_path), "run-AP3333"),
        cfg=cfg.model_copy(update={"git": git2}),
        reporter=STDIO_REPORTER,
    )
    err = capsys.readouterr().err
    assert "recorded as merged" in err and "auto_pruned agent6/run-AP3333" in err
    assert _git(tmp_path, "branch", "--list", "agent6/run-AP3333") == ""


def test_auto_prune_keeps_squash_branch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    _setup_run_on_branch(
        tmp_path,
        "run-AP2222",
        commits=[("a.txt", "a\n", "agent6 iter 1: add a")],
        run_branch="agent6/run-AP2222",
    )
    cfg = load_effective(tmp_path, None).config
    git2 = cfg.git.model_copy(
        update={"auto_merge": True, "auto_prune": True, "merge_strategy": "squash"}
    )
    cfg2 = cfg.model_copy(update={"git": git2})
    finmod.finalize_auto_merge(
        tmp_path,
        layout=SessionLayout(state_dir(tmp_path), "run-AP2222"),
        cfg=cfg2,
        reporter=STDIO_REPORTER,
    )
    assert _git(tmp_path, "branch", "--list", "agent6/run-AP2222")  # kept (squash unreachable)
    assert "git branch -D" in capsys.readouterr().err
