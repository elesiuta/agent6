# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""A fork's leg runs in a linked worktree whose `.git` is a pointer into the
repository's git dir. The jail policy builder derives that read-only grant
from the workspace shape, so every consumer of the one policy (each command,
the hardened exposure scan) sees the same grant."""

from __future__ import annotations

import subprocess
from pathlib import Path

from agent6.app.confine import _hardened_grant_regions  # pyright: ignore[reportPrivateUsage]
from agent6.config import Config
from agent6.git_ops import add_worktree
from agent6.tools.policy import jail_policy, linked_worktree_git_dir


def _repo_and_worktree(tmp_path: Path) -> tuple[Path, Path]:
    repo = tmp_path / "repo"
    subprocess.run(["git", "init", "-q", "-b", "main", str(repo)], check=True)
    (repo / "a.txt").write_text("a\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(repo), "add", "-A"], check=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-q", "-m", "init"], check=True)
    worktree = tmp_path / "wt"
    add_worktree(repo, worktree, "HEAD")
    return repo, worktree


def test_a_linked_worktree_policy_grants_the_repository_git_read_only(tmp_path: Path) -> None:
    """The one policy builder adds the repository's `.git` as a read-only
    path for a linked worktree root, resolved from the worktree's own `.git`
    pointer and its `commondir`, and nothing for an ordinary checkout."""
    repo, worktree = _repo_and_worktree(tmp_path)
    git_dir = (repo / ".git").resolve()
    assert linked_worktree_git_dir(worktree) == git_dir
    assert linked_worktree_git_dir(repo) is None
    policy = jail_policy(worktree, Config(), "strict", ("true",))
    assert git_dir in policy.extra_ro_paths and git_dir not in policy.extra_rw_paths
    assert jail_policy(repo, Config(), "strict", ("true",)).extra_ro_paths == ()


def test_the_hardened_exposure_scan_sees_the_repository_git_grant(tmp_path: Path) -> None:
    """The regions the hardened preflight checks `hide_paths` against come
    from the same builder, so the repository `.git` a fork's leg is granted
    is one of them, named for what it is. Added only when the jail was built,
    the grant was invisible to the preflight: a hidden path inside it neither
    refused nor warned while the model could read it. (A repo under the
    shared /tmp is exposed through that region first, so the pin reads the
    region list rather than the refusal text.)"""
    repo, worktree = _repo_and_worktree(tmp_path)
    regions = dict(_hardened_grant_regions(Config(), worktree))
    assert regions[(repo / ".git").resolve()].startswith("the repository's .git")
    assert (repo / ".git").resolve() not in dict(_hardened_grant_regions(Config(), repo))
