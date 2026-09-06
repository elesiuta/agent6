# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""A fork's linked git worktree after the fork: the dirt check before one is
removed, the sessions that still own one, and the sweep `sessions prune` runs
to remove the worktrees of forks whose tips landed.
"""

from __future__ import annotations

import contextlib
from collections.abc import Sequence
from pathlib import Path

from agent6.git_ops import (
    GitError,
    chain_dirty,
    chain_dirty_paths,
    chain_ref_for,
    chain_tip,
    merge_stamp_holds,
    remove_worktree,
)
from agent6.paths import state_dir
from agent6.sessions.ipc import worker_is_alive
from agent6.sessions.lock import (
    checkout_lock_path,
)
from agent6.sessions.manifest import (
    ManifestError,
    SessionManifest,
    read_manifest,
)
from agent6.viewmodel import session_dirs


def remove_fork_worktree(repo: Path, worktree: Path, tips: tuple[str, ...]) -> tuple[bool, str]:
    """Delete a fork's worktree (only a linked worktree of *repo*, see
    `git_ops.remove_worktree`) and the checkout lock its legs took, unless it
    holds work none of *tips* (the commits its sessions landed) has. Returns
    `(removed, note)`: removed is False when *worktree* is not one, could not
    be deleted, or holds such work, and the note then says which; "" on
    success.

    The dirty check is git's own rule for `worktree remove`: prune and rm land
    on a merged fork, and the tree can still carry an uncommitted edit or a
    file that was never added, which `rmtree` would take with no way back."""
    dirt = uncommitted_in_worktree(worktree, tips)
    if dirt:
        return False, dirt
    lock_path = checkout_lock_path(state_dir(worktree), worktree)
    if not remove_worktree(repo, worktree):
        return False, (
            "could not be removed: not a linked worktree of this repository,"
            " or a file in it would not delete"
        )
    lock_path.unlink(missing_ok=True)
    return True, ""


def uncommitted_in_worktree(worktree: Path, tips: tuple[str, ...]) -> str:
    """What *worktree* holds that none of *tips* does, as one phrase for a keep
    line; "" when a tip covers it or it is unreadable (a missing dir is not
    dirt).

    A fork's worktree stays detached at its fork point while its run commits to
    the chain, so `git status` there reports the whole run as dirt: the
    comparison is against the run's own tips (HEAD when it has none, a run
    whose commits the model makes itself)."""
    if not worktree.is_dir():
        return ""  # a worktree prune already removed is not dirt to keep
    tips = tips or ("HEAD",)
    try:
        for tip in tips:
            if not chain_dirty(worktree, tip, None):
                return ""
        held = chain_dirty_paths(worktree, tips[-1], None, 5)
    except (GitError, OSError):
        # git runs WITH cwd=worktree, so a directory that vanished between the
        # check above and here is an OSError, not a GitError: unreadable is not
        # dirt.
        return ""
    if not held:
        return ""
    named = ", ".join(held[:4]) + (", ..." if len(held) > 4 else "")
    return f"holds work no commit has: {named}"


def worktree_owners(state_dir: Path) -> dict[Path, list[tuple[Path, SessionManifest]]]:
    """Every worktree a session manifest names, with the sessions naming it
    (an `/undo` fork shares its source's). The manifests are the only record
    of which directories are agent6's: a path no manifest names is never
    touched, wherever it sits."""
    owners: dict[Path, list[tuple[Path, SessionManifest]]] = {}
    for session_dir in session_dirs(state_dir):
        with contextlib.suppress(ManifestError):
            manifest = read_manifest(session_dir)
            if manifest.worktree is not None:
                owners.setdefault(manifest.worktree, []).append((session_dir, manifest))
    return owners


def _still_needs_worktree(repo: Path, session_dir: Path, manifest: SessionManifest) -> str:
    """Why *session_dir* still needs its worktree ("live", "unmerged"), or ""
    when its work has landed: the merge stamp is the prune's own test of
    "merged" (`merge_stamp_holds`: the branch still points where the merge
    left it)."""
    if worker_is_alive(session_dir):
        return "live"
    merged = manifest.merged is not None and merge_stamp_holds(
        repo, session_dir.name, manifest.run_branch or "", manifest.merged.tip
    )
    return "" if merged else "unmerged"


def _landed_tips(repo: Path, sessions: Sequence[tuple[Path, SessionManifest]]) -> tuple[str, ...]:
    """The commits *sessions* landed their work on: each one's chain tip, else
    the tip its merge stamp recorded (`--delete-squashed` deletes the ref in
    the same sweep, and the commit outlives it)."""
    tips = (
        chain_tip(repo, chain_ref_for(d.name)) or (m.merged.tip if m.merged else "")
        for d, m in sessions
    )
    return tuple(tip for tip in tips if tip)


def sweep_fork_worktrees(repo: Path, state: Path) -> tuple[list[str], list[tuple[str, str]]]:
    """Remove every fork worktree whose sessions have all landed their work
    (merged, none live), and keep the rest. Returns `([removed id], [(kept
    id, why)])`; a session sharing a kept worktree is kept for the session
    that needs it."""
    removed: list[str] = []
    kept: list[tuple[str, str]] = []
    for worktree, sessions in worktree_owners(state).items():
        if not worktree.exists():
            continue
        needs = {d.name: why for d, m in sessions if (why := _still_needs_worktree(repo, d, m))}
        if needs:
            first = next(iter(needs))
            kept.extend((d.name, needs.get(d.name, f"shared with {first}")) for d, _ in sessions)
            continue
        gone, note = remove_fork_worktree(repo, worktree, _landed_tips(repo, sessions))
        if gone:
            removed.extend(d.name for d, _ in sessions)
        elif note:
            # Merged, but the tree carries work no commit has, or would not
            # delete: keeping it is the only safe answer, and the operator
            # has to see why.
            kept.extend((d.name, note) for d, _ in sessions)
    return removed, kept
