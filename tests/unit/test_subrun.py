# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Tests for agent6.workflows.subrun on temporary git repositories."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from agent6.git_ops import branch_exists, commit_all, create_branch
from agent6.workflows.subrun import SubrunError, clone_workspace, import_run


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)


def _init_repo(repo: Path) -> None:
    repo.mkdir(parents=True, exist_ok=True)
    _git(repo, "init", "-q", "-b", "main")
    _git(repo, "config", "user.email", "t@t")
    _git(repo, "config", "user.name", "t")
    (repo / "README.md").write_text("hi\n", encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "init")


def test_clone_workspace_produces_independent_clone(tmp_path: Path) -> None:
    origin = tmp_path / "origin"
    _init_repo(origin)
    dest = tmp_path / "lane-1"

    clone_workspace(origin, dest)

    assert (dest / "README.md").read_text(encoding="utf-8") == "hi\n"
    (dest / "README.md").write_text("edited in the lane\n", encoding="utf-8")
    assert origin.joinpath("README.md").read_text(encoding="utf-8") == "hi\n"


def test_clone_workspace_missing_origin_raises_subrun_error(tmp_path: Path) -> None:
    with pytest.raises(SubrunError):
        clone_workspace(tmp_path / "does-not-exist", tmp_path / "lane-1")


def test_clone_workspace_raises_subrun_error_on_failure(tmp_path: Path) -> None:
    origin = tmp_path / "origin"
    _init_repo(origin)
    dest = tmp_path / "lane-1"
    dest.mkdir()
    (dest / "existing.txt").write_text("occupied\n", encoding="utf-8")

    with pytest.raises(SubrunError):
        clone_workspace(origin, dest)


def test_import_run_lands_branch_and_moves_run_dir(tmp_path: Path) -> None:
    origin = tmp_path / "origin"
    _init_repo(origin)
    lane_repo = tmp_path / "lane-1"
    clone_workspace(origin, lane_repo)

    branch = "agent6/lane-1"
    create_branch(lane_repo, branch)
    (lane_repo / "feature.txt").write_text("new stuff\n", encoding="utf-8")
    commit_all(lane_repo, "lane change")

    lane_session_dir = tmp_path / "lane-state" / "sessions" / "runs" / "01ABC"
    lane_session_dir.mkdir(parents=True)
    (lane_session_dir / "manifest.json").write_text("{}\n", encoding="utf-8")
    origin_state = tmp_path / "origin-state"

    imported = import_run(origin, lane_repo, branch, lane_session_dir, origin_state)

    assert imported == origin_state / "sessions" / "runs" / "01ABC"
    assert (imported / "manifest.json").read_text(encoding="utf-8") == "{}\n"
    assert not lane_session_dir.exists()
    assert branch_exists(origin, branch)


def test_import_run_lands_a_lane_that_never_committed(tmp_path: Path) -> None:
    """A lane stopped before its first commit has no branch to land. Reported
    as a failed fetch, every lane of a fleet stopped by an expired key read
    "couldn't find remote ref", and the reason stayed in a session dir the
    operator could no longer reach."""
    origin = tmp_path / "origin"
    _init_repo(origin)
    lane_repo = tmp_path / "lane-1"
    clone_workspace(origin, lane_repo)
    lane_session_dir = tmp_path / "lane-state" / "sessions" / "runs" / "01DEAD"
    lane_session_dir.mkdir(parents=True)
    (lane_session_dir / "manifest.json").write_text("{}\n", encoding="utf-8")

    imported = import_run(
        origin, lane_repo, "agent6/lane-1", lane_session_dir, tmp_path / "origin-state"
    )

    assert (imported / "manifest.json").is_file()
    assert not branch_exists(origin, "agent6/lane-1")


def test_import_run_refuses_existing_branch(tmp_path: Path) -> None:
    origin = tmp_path / "origin"
    _init_repo(origin)
    branch = "agent6/lane-1"
    create_branch(origin, branch)
    _git(origin, "checkout", "main")

    lane_repo = tmp_path / "lane-1"
    clone_workspace(origin, lane_repo)
    lane_session_dir = tmp_path / "lane-state" / "sessions" / "runs" / "01ABC"
    lane_session_dir.mkdir(parents=True)
    (lane_session_dir / "manifest.json").write_text("{}\n", encoding="utf-8")
    origin_state = tmp_path / "origin-state"

    with pytest.raises(SubrunError):
        import_run(origin, lane_repo, branch, lane_session_dir, origin_state)
    # Refused before moving anything.
    assert lane_session_dir.exists()
    assert not (origin_state / "sessions" / "runs" / "01ABC").exists()


def test_import_run_refuses_existing_run_dir(tmp_path: Path) -> None:
    origin = tmp_path / "origin"
    _init_repo(origin)
    lane_repo = tmp_path / "lane-1"
    clone_workspace(origin, lane_repo)
    branch = "agent6/lane-1"
    create_branch(lane_repo, branch)
    (lane_repo / "feature.txt").write_text("new stuff\n", encoding="utf-8")
    commit_all(lane_repo, "lane change")

    lane_session_dir = tmp_path / "lane-state" / "sessions" / "runs" / "01ABC"
    lane_session_dir.mkdir(parents=True)
    (lane_session_dir / "manifest.json").write_text("{}\n", encoding="utf-8")
    origin_state = tmp_path / "origin-state"
    (origin_state / "sessions" / "runs" / "01ABC").mkdir(parents=True)  # already imported

    with pytest.raises(SubrunError):
        import_run(origin, lane_repo, branch, lane_session_dir, origin_state)
    # Refused before fetching or moving anything.
    assert not branch_exists(origin, branch)
    assert lane_session_dir.exists()
