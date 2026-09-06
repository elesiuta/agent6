# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""`agent6 sessions merge/prune`: landing a run's chain on its base, and
cleaning up the branches and chain refs already landed."""

from __future__ import annotations

import contextlib
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from agent6.app.fork import sweep_fork_worktrees
from agent6.app.merge import execute_merge, left_behind_line, noop_merge_line
from agent6.app.parallel import adopt_orphan_lane, sweep_fanout_clones
from agent6.commit_message import render_commit_trailer
from agent6.config import Config, ConfigError
from agent6.config.layer import load_effective
from agent6.git_ops import (
    BRANCH_PREFIX,
    CommitIdentity,
    GitError,
    branch_exists,
    branch_tip_sha,
    chain_ref_for,
    chain_tip,
    delete_branch_if_merged,
    delete_ref,
    force_delete_squash_merged_branch,
    is_ancestor,
    is_git_repo,
    list_chain_refs,
    list_run_branches,
    list_run_commits,
    verify_git_identity,
)
from agent6.git_ops import status as git_status
from agent6.paths import state_dir
from agent6.sessions.ipc import worker_is_alive
from agent6.sessions.layout import LOGS_NAME, SessionLayout, session_layout
from agent6.sessions.manifest import (
    NO_MERGE_COMMIT,
    ManifestError,
    MergeStamp,
    SessionManifest,
    read_manifest,
)
from agent6.ui.cli._common import error, refuse, sgr
from agent6.ui.cli.sessions_cmds import (
    _commits_ref,
    _committed_nothing,
    _resolve_session_manifest,
)
from agent6.viewmodel import tail_events, worker_models
from agent6.workflows.subrun import SubrunError


@dataclass(frozen=True, slots=True)
class _MergePlan:
    """A validated, mutation-ready merge: everything `_cmd_merge` needs after every
    guard has passed. `_plan_merge` builds it without touching the repo."""

    layout: SessionLayout
    manifest: SessionManifest
    run_branch: str
    target: str
    base_sha: str
    strategy: str
    identity: CommitIdentity
    cfg: Config


def _plan_merge(  # noqa: PLR0911
    cwd: Path,
    session_id: str,
    into: str | None,
    strategy: str | None,
    *,
    config_path: Path | None,
) -> _MergePlan | int:
    """Resolve and validate everything a merge needs, or return an exit code. Pure:
    every guard fails before `_cmd_merge` mutates the repo."""
    res = _resolve_session_manifest(cwd, session_id)
    if isinstance(res, int):
        return res
    layout, manifest = res
    # A live run's chain keeps growing and its edits sit in the worktree; a
    # merge now would land a prefix of its work and bring the operator's index
    # forward under the worker. The gate is the raw pid, NOT session_is_live:
    # after session.end the worker's finalizer may still be at work. The run's
    # own end-of-run finalize_auto_merge is unaffected (it calls execute_merge
    # directly, not this planner).
    if worker_is_alive(layout.session_dir):
        refuse(
            f"run {session_id!r} is still live; a merge now lands only the"
            " commits so far (its later ones need another merge) and moves your"
            " index while the worker still edits. Stop it first:\n"
            f"    agent6 sessions stop {session_id}"
        )
        return 2
    ref = _commits_ref(cwd, manifest)
    if ref.reason:
        error(f"this session has no branch to merge ({ref.reason}).")
        return 2
    run_branch = ref.head_ref
    target = into or manifest.base_branch
    if not target:
        error("no target branch (manifest has no base_branch); pass --into <branch>.")
        return 2
    if target == run_branch:
        error(f"target {target!r} is the run branch itself; pass --into <other-branch>.")
        return 2
    try:
        cfg = load_effective(cwd, config_path).config
    except ConfigError as exc:
        error(f"{exc}")
        return 2
    # An orphaned fan-out lane (coordinator died before importing it) is
    # adopted here: fetch its branch from the lane clone and replace the
    # live-view symlink, then merge like any run.
    try:
        adopted = adopt_orphan_lane(cwd, cfg, layout, manifest)
    except SubrunError as exc:
        error(f"{exc}")
        return 2
    if adopted is not None:
        print(f"[agent6] {adopted}")
        layout = SessionLayout(state_dir=layout.state_dir, session_id=layout.session_id)
    # chain_tip resolves both shapes head_ref takes: a branch name and the
    # hidden refs/agent6/<id>/head chain ref.
    if chain_tip(cwd, run_branch) is None:
        if _committed_nothing(cwd, manifest.session_id):
            # Not a failure: the command's job is to land work, and there is
            # none (a zero-commit branch reads the same way below).
            print("[agent6] nothing to merge: this run committed nothing.")
            return 0
        error(
            f"run ref {run_branch!r} no longer exists; its commits survive at"
            f" {chain_ref_for(manifest.session_id)}."
        )
        return 2
    if not branch_exists(cwd, target):
        error(f"target branch {target!r} does not exist; pass --into <existing-branch>.")
        return 2
    identity = CommitIdentity(
        name=cfg.git.commit.name,
        email=cfg.git.commit.email,
        trailer=render_commit_trailer(
            cfg.git.commit.trailer,
            models=worker_models(tail_events(layout.session_dir / LOGS_NAME, follow=False))
            or ((manifest.models.driver.model,) if manifest.models.driver else ()),
        ),
    )
    try:
        verify_git_identity(cwd, identity)  # refuse cleanly before mutating anything
    except GitError as exc:
        error(f"{exc}")
        return 2
    return _MergePlan(
        layout=layout,
        manifest=manifest,
        run_branch=run_branch,
        target=target,
        base_sha=manifest.base_sha,
        strategy=strategy or cfg.git.merge_strategy,
        identity=identity,
        cfg=cfg,
    )


def _manual_merge_cmd(cwd: Path, plan: _MergePlan) -> str:
    """The by-hand merge for a plumbing conflict, from the checkout as it is: git
    refuses to merge over modified tracked files (a finished run leaves its work
    in the tree until merged), so a dirty tree is stashed first, and a checkout
    on another branch moves to the target."""
    steps: list[str] = []
    with contextlib.suppress(GitError):
        st = git_status(cwd)
        if st.modified_count:
            steps.append("git stash")
        if st.branch != plan.target:
            steps.append(f"git checkout {plan.target}")
    squash = " --squash" if plan.strategy == "squash" else ""
    steps.append(f"git merge{squash} {plan.run_branch}")
    return " && ".join(steps)


def _cmd_merge(
    *,
    session_id: str,
    strategy: str | None,
    into: str | None,
    message: str | None,
    config_path: Path | None,
) -> int:
    """Land a run's work on a target branch (default: the branch the run was
    cut from) with the chosen strategy (default: git.merge_strategy). Ref
    plumbing only: your checkout, index, and worktree are never the medium, so
    a worktree still carrying the run's work is no obstacle."""
    cwd = Path.cwd()
    plan = _plan_merge(cwd, session_id, into, strategy, config_path=config_path)
    if isinstance(plan, int):
        return plan
    if plan.base_sha and not list_run_commits(cwd, plan.base_sha, plan.run_branch):
        # A success line here would be indistinguishable from a real merge.
        print(f"[agent6] nothing to merge: run branch {plan.run_branch} has no commits.")
        return 0
    outcome = execute_merge(
        cwd,
        layout=plan.layout,
        manifest=plan.manifest,
        run_branch=plan.run_branch,
        target=plan.target,
        base_sha=plan.base_sha,
        strategy=plan.strategy,
        message=message,
        cfg=plan.cfg,
        identity=plan.identity,
        warn=lambda m: print(f"[agent6] {m}", file=sys.stderr),
    )
    if outcome.status == "error":
        error(f"{outcome.error}")
        return 1
    if outcome.status == "conflict":
        print(
            f"CONFLICT: merging {plan.run_branch} into {plan.target} hit conflicts in "
            f"{', '.join(outcome.conflicts)}. Your checkout is untouched (no partial "
            f"merge); resolve it by hand if you want:\n"
            f"    {_manual_merge_cmd(cwd, plan)}",
            file=sys.stderr,
        )
        return 1
    note = (
        f"\n  (merge record could not be written: {outcome.stamp_error};"
        " `sessions prune` will call this branch unmerged)"
        if outcome.stamp_error
        else ""
    )
    if outcome.status == "noop":
        print(f"[agent6] {noop_merge_line(plan.run_branch, plan.target, outcome)}.{note}")
        return 0
    print(
        f"[agent6] merged {plan.run_branch} into {plan.target} "
        f"({plan.strategy}) -> {outcome.merged_sha[:12]}{note}"
    )
    if kept := left_behind_line(plan.target, outcome):
        print(f"[agent6] {kept}")
    return 0


def _session_stamp(layout: SessionLayout | None) -> tuple[MergeStamp | None, str]:
    """A session's recorded merge, and why none could be read: "no session
    record", "unreadable manifest" (kept, never force-deleted), else "" with
    the stamp, None when the manifest records no merge."""
    if layout is None:
        return None, "no session record"
    try:
        return read_manifest(layout.session_dir).merged, ""
    except ManifestError:
        return None, "unreadable manifest"


def _base_gone(into: str) -> str:
    return f"base {into} is gone"


def _squash_unconfirmed(cwd: Path, stamp: MergeStamp) -> str:
    """Why a squash-merge stamp does not prove a force-delete content-safe,
    "" when it does: the merged tip must be recorded, and the base must still
    hold the commit the record names (the merge commit, or, for a merge that
    added nothing, the base tip that already held the content), since a reset
    or rewrite of the base after the merge leaves the branch as the content's
    only holder."""
    if not stamp.tip:
        return "no merge tip was recorded"
    if stamp.sha == NO_MERGE_COMMIT:
        if not stamp.into_tip:
            return "the record names no commit to check"
        if not is_ancestor(cwd, stamp.into_tip, stamp.into):
            return f"{stamp.into} no longer holds its content"
    elif not is_ancestor(cwd, stamp.sha, stamp.into):
        return f"{stamp.into} no longer holds the merge commit"
    return ""


@dataclass(frozen=True, slots=True)
class Landed:
    """How a run's commits stand against its merge stamp, for both prune loops:
    `merged` (its tip is an ancestor of the base), `squashed` (the stamp proves
    the base holds them and the operator asked for the force-delete), else
    `keep` with the reason each loop prints and counts."""

    verdict: Literal["merged", "squashed", "keep"]
    why: str = ""


def landed(cwd: Path, stamp: MergeStamp, tip: str | None, *, delete_squashed: bool) -> Landed:
    """The one classification (see :class:`Landed`) of a run with a recorded
    merge; *tip* is the branch's or chain ref's sha, None when the branch is
    gone."""
    if not branch_exists(cwd, stamp.into):
        return Landed("keep", _base_gone(stamp.into))
    if tip is not None and is_ancestor(cwd, tip, stamp.into):
        return Landed("merged")
    if tip is not None and stamp.tip and stamp.tip != tip:
        # A resumed run committing on after the merge: those commits are in
        # no other ref. "squash-merged" would read as an invitation to a flag
        # that refuses it, so each refusal is its own reason.
        return Landed("keep", "advanced since the merge")
    if why := _squash_unconfirmed(cwd, stamp):
        return Landed("keep", why)
    return Landed("squashed") if delete_squashed else Landed("keep", "squash-merged")


def _keep_branch_line(br: str, stamp: MergeStamp, why: str) -> str:
    if why == "squash-merged":
        return (
            f"[agent6] kept {br} (squash-merged into {stamp.into}, unreachable; "
            f"remove with: sessions prune --delete-squashed, or: git branch -D {br})"
        )
    at = "" if stamp.sha == NO_MERGE_COMMIT else f" at {stamp.sha[:12]}"
    return (
        f"[agent6] kept {br} (squash-merged into {stamp.into}{at}, but {why};"
        f" review, then: git branch -D {br})"
    )


def _prune_branch(cwd: Path, br: str, stamp: MergeStamp, state: Landed, current: str) -> bool:
    """Act on one run branch's classification: force-delete a proven squash
    (with the undelete hint: the commit survives in the reflog until GC), keep
    the rest and say why. Returns whether it was deleted."""
    if state.verdict == "merged":
        # Reachable-merged into its base, so `git branch -d` only refused because
        # HEAD is not the base; deleting it cleanly needs to run from the base.
        print(
            f"[agent6] kept {br} (merged into {stamp.into} but not reachable from "
            f"{current!r}; re-run prune on {stamp.into}, or: git branch -D {br})"
        )
        return False
    if state.verdict == "keep":
        print(_keep_branch_line(br, stamp, state.why))
        return False
    sha = branch_tip_sha(cwd, br)
    if sha is not None and force_delete_squash_merged_branch(cwd, br):
        print(f"[agent6] deleted {br} (squash-merged into {stamp.into})")
        print(sgr(f"          undelete: git branch {br} {sha[:12]}", "2"))
        return True
    print(f"[agent6] kept {br} (squash-merged into {stamp.into}; git refused the delete)")
    return False


def _cmd_prune(*, delete_squashed: bool = False, config_path: Path | None = None) -> int:
    """Delete agent6/* run branches that `git branch -d` can safely remove
    (reachable-merged into HEAD, i.e. merge/ff strategies). Report squash-merged
    ones and unmerged ones (review first). Sweep fan-out clone dirs whose every
    lane branch tip already exists in this repo (content-safe by commit proof;
    a clone holding any commit this repo lacks is kept whole), and the
    worktrees of merged forks (an unmerged fork keeps its worktree; `sessions
    rm` removes a fork's with its record).

    With `--delete-squashed` also force-delete branches and chain refs the
    manifest confirms were squash-merged into an existing base -- their content
    is safe in that base commit, and each deletion prints the exact command to
    undelete it (a branch's commit survives in its reflog until GC; a chain ref
    has none, so its line carries the sha). Unmerged runs are never
    force-deleted."""
    cwd = Path.cwd()
    if not is_git_repo(cwd):
        error("not a git repository")
        return 2
    branches = list_run_branches(cwd)
    try:
        current = git_status(cwd).branch
    except GitError as exc:
        error(f"{exc}")
        return 2
    repo_state = state_dir(cwd)
    deleted = squashed_deleted = merged_kept = unmerged_kept = live_kept = 0
    for br in branches:
        if br == current:
            print(f"[agent6] skipped {br} (checked out)", file=sys.stderr)
            continue
        layout = session_layout(repo_state, br.removeprefix(BRANCH_PREFIX))
        if layout is not None and worker_is_alive(layout.session_dir):
            # The run is still committing to it, whatever git makes of its tip
            # and whether its manifest reads.
            live_kept += 1
            print(f"[agent6] kept {br} (live)")
            continue
        if delete_branch_if_merged(cwd, br):
            deleted += 1
            print(f"[agent6] deleted {br} (merged)")
            continue
        stamp, why = _session_stamp(layout)
        if why:
            unmerged_kept += 1
            print(f"[agent6] kept {br} ({why}; review, then: git branch -D {br})")
            continue
        if stamp is None or not stamp.sha:
            unmerged_kept += 1
            print(f"[agent6] kept {br} (NOT merged; review, then: git branch -D {br})")
            continue
        state = landed(cwd, stamp, branch_tip_sha(cwd, br), delete_squashed=delete_squashed)
        if _prune_branch(cwd, br, stamp, state, current):
            squashed_deleted += 1
        else:
            merged_kept += 1
    # Chain refs are pruned whether or not any run BRANCH survives: with
    # `branch_per_run` off there is never one, and once prune has deleted the
    # last branch the refs it kept for a later pass would be unreachable by
    # this command forever.
    refs_deleted, refs_kept = _prune_chain_refs(cwd, repo_state, delete_squashed=delete_squashed)
    clones_note, swept_any = _sweep_workdirs(cwd, repo_state, config_path)
    if not branches and not (refs_deleted or refs_kept or swept_any):
        print("[agent6] nothing to prune: no agent6/* run branches, no chain refs.")
        return 0
    kept = merged_kept + unmerged_kept + live_kept
    total_deleted = deleted + squashed_deleted
    squashed_note = f" ({squashed_deleted} squash-merged)" if squashed_deleted else ""
    live_note = f", {live_kept} live" if live_kept else ""
    refs_note = ""
    if refs_deleted or refs_kept:
        why = ", ".join(f"{n} {reason}" for reason, n in sorted(refs_kept.items()))
        refs_note = f"; chain refs: deleted {refs_deleted}, kept {sum(refs_kept.values())}" + (
            f" ({why})" if why else ""
        )
    print(
        f"\n[agent6] deleted {total_deleted}{squashed_note}; kept {kept} "
        f"({merged_kept} merged, {unmerged_kept} unmerged{live_note}){refs_note}{clones_note}",
    )
    return 0


def _sweep_workdirs(cwd: Path, state: Path, config_path: Path | None) -> tuple[str, bool]:
    """Sweep the fan-out clones and fork worktrees under this repo's
    `[parallel].workdir` scope, printing each keep and each worktree removal.
    Returns (the summary-line note, whether anything was swept or kept)."""
    try:
        cfg = load_effective(cwd, config_path).config
    except ConfigError as exc:
        print(f"[agent6] workdir sweep skipped (config unreadable: {exc})", file=sys.stderr)
        return "", False
    clones_swept, clones_kept = sweep_fanout_clones(cwd, cfg)
    if clones_kept:
        print(
            f"[agent6] kept {clones_kept} fan-out clone dir(s) holding commits this"
            " repo lacks (merge or archive their lanes first)"
        )
    worktrees_removed, worktrees_kept = sweep_fork_worktrees(cwd, state)
    for fork_id in worktrees_removed:
        print(f"[agent6] removed {fork_id}'s worktree (merged)")
    for fork_id, why in worktrees_kept:
        print(f"[agent6] kept {fork_id}'s worktree ({why})")
    note = (
        f"; fan-out clones: swept {clones_swept}, kept {clones_kept}"
        if clones_swept or clones_kept
        else ""
    )
    return note, bool(clones_swept or clones_kept or worktrees_removed)


def _prune_chain_refs(
    cwd: Path, repo_state: Path, *, delete_squashed: bool
) -> tuple[int, Counter[str]]:
    """Drop `refs/agent6/<id>/head` chain refs whose manifest confirms the run
    merged, under the same safety rules as branches: reachable-from-base
    deletes outright; a squash-merge (content in the base commit, ref
    unreachable) deletes only with --delete-squashed AND only while the ref
    still points at the recorded merged tip. Live runs, unmerged runs, and
    refs with no run manifest (machine chains) are kept, counted by reason
    and never named: an unmerged ref is the run's anchor, not clutter.
    Returns (deleted, kept by reason), every ref counted once."""
    refs_deleted = 0
    kept: Counter[str] = Counter()
    for sid, sha in list_chain_refs(cwd):
        layout = session_layout(repo_state, sid)
        if layout is None:
            # A machine's chain (`machine_chain_ref_for`) has no session record.
            kept["machine" if sid.startswith("machine-") else "no session record"] += 1
            continue
        if worker_is_alive(layout.session_dir):
            kept["live"] += 1
            continue
        stamp, why = _session_stamp(layout)
        if why:
            kept[why] += 1
            continue
        if stamp is None or not stamp.sha:
            kept["unmerged"] += 1
            continue
        state = landed(cwd, stamp, sha, delete_squashed=delete_squashed)
        if state.verdict == "keep":
            kept[state.why] += 1
            continue
        ref = chain_ref_for(sid)
        delete_ref(cwd, ref)
        refs_deleted += 1
        how = "merged" if state.verdict == "merged" else "squash-merged"
        print(f"[agent6] deleted {ref} ({how} into {stamp.into})")
        if state.verdict == "squashed":
            # A chain ref has no reflog: the sha is the only way back.
            print(sgr(f"          undelete: git update-ref {ref} {sha[:12]}", "2"))
    return refs_deleted, kept
