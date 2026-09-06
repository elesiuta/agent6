# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""A fork's leg runs in a linked worktree whose `.git` is a pointer into the
repository's git dir. The jail grants that dir from what agent6 recorded
when it added the worktree (the manifest's `worktree_git_dir`), never from
the pointer file: under hardened the pointer sits in the writable workspace,
so a jailed command could rewrite it to name any host directory."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from agent6.app.confine import _hardened_grant_regions  # pyright: ignore[reportPrivateUsage]
from agent6.config import Config
from agent6.git_ops import add_worktree
from agent6.sandbox.jail import JailUnavailableError
from agent6.tools.policy import jail_policy


def _repo_and_worktree(tmp_path: Path) -> tuple[Path, Path]:
    repo = tmp_path / "repo"
    subprocess.run(["git", "init", "-q", "-b", "main", str(repo)], check=True)
    (repo / "a.txt").write_text("a\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(repo), "add", "-A"], check=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-q", "-m", "init"], check=True)
    worktree = tmp_path / "wt"
    add_worktree(repo, worktree, "HEAD")
    return repo, worktree


def _rewrite_pointer(worktree: Path, target: str) -> None:
    """What a jailed command can do under hardened: point the workspace's
    own `.git` file at a directory whose `commondir` names any host dir."""
    evil = worktree / "evil"
    evil.mkdir()
    (evil / "commondir").write_text(f"{target}\n", encoding="utf-8")
    (worktree / ".git").write_text(f"gitdir: {evil}\n", encoding="utf-8")


def test_the_recorded_git_dir_is_granted_read_only(tmp_path: Path) -> None:
    """With the recorded dir matching the worktree's pointer, the policy
    grants that dir read-only, and the hardened exposure scan lists it under
    the same name."""
    repo, worktree = _repo_and_worktree(tmp_path)
    git_dir = (repo / ".git").resolve()
    policy = jail_policy(worktree, Config(), "strict", ("true",), worktree_git_dir=git_dir)
    assert git_dir in policy.extra_ro_paths and git_dir not in policy.extra_rw_paths
    regions = dict(_hardened_grant_regions(Config(), worktree, git_dir))
    assert regions[git_dir].startswith("the repository's .git")


def test_a_rewritten_pointer_refuses_instead_of_granting(tmp_path: Path) -> None:
    """A pointer that no longer resolves to the recorded dir refuses the
    policy, naming both; the rewritten target is granted nothing. Derived
    from the pointer, the policy granted whatever host dir it named."""
    repo, worktree = _repo_and_worktree(tmp_path)
    git_dir = (repo / ".git").resolve()
    _rewrite_pointer(worktree, "/etc")
    with pytest.raises(JailUnavailableError, match="/etc") as refused:
        jail_policy(worktree, Config(), "hardened", ("true",), worktree_git_dir=git_dir)
    assert str(git_dir) in str(refused.value)
    with pytest.raises(JailUnavailableError):
        _hardened_grant_regions(Config(), worktree, git_dir)


def test_a_linked_worktree_agent6_did_not_record_gets_no_grant(tmp_path: Path) -> None:
    """Without a record there is nothing to grant: a foreign linked worktree,
    honest pointer or rewritten, gets no path beyond the workspace."""
    repo, worktree = _repo_and_worktree(tmp_path)
    assert jail_policy(worktree, Config(), "strict", ("true",)).extra_ro_paths == ()
    _rewrite_pointer(worktree, "/etc")
    assert jail_policy(worktree, Config(), "hardened", ("true",)).extra_ro_paths == ()
    assert (repo / ".git").resolve() not in dict(_hardened_grant_regions(Config(), repo))
