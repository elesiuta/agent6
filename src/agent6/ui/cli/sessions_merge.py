# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""`agent6 sessions merge/prune`: landing a run's chain on its base, and
cleaning up the branches and chain refs already landed."""

from __future__ import annotations

import contextlib
import sys
from dataclasses import dataclass
from pathlib import Path

from agent6.app.fork import sweep_fork_worktrees
from agent6.app.merge import execute_merge
from agent6.app.parallel import adopt_orphan_lane, sweep_fanout_clones
from agent6.config import Config, ConfigError
from agent6.config.layer import load_effective, resolved_state_dir
from agent6.git_ops import (
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
    render_commit_trailer,
    verify_git_identity,
)
from agent6.git_ops import status as git_status
from agent6.sessions.ipc import worker_is_alive
from agent6.sessions.layout import LOGS_NAME, SessionLayout, session_layout
from agent6.sessions.manifest import (
    ManifestError,
    SessionManifest,
    manifest_for_branch,
    read_manifest,
)
from agent6.ui.cli._common import sgr
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
        print(
            f"REFUSING: run {session_id!r} is still live; a merge now lands only the"
            " commits so far (its later ones need another merge) and moves your"
            " index while the worker still edits. Stop it first:\n"
            f"    agent6 sessions stop {session_id}",
            file=sys.stderr,
        )
        return 2
    ref = _commits_ref(cwd, manifest)
    if ref.reason:
        print(f"ERROR: this session has no branch to merge ({ref.reason}).", file=sys.stderr)
        return 2
    run_branch = ref.head_ref
    target = into or manifest.base_branch
    if not target:
        print(
            "ERROR: no target branch (manifest has no base_branch); pass --into <branch>.",
            file=sys.stderr,
        )
        return 2
    if target == run_branch:
        print(
            f"ERROR: target {target!r} is the run branch itself; pass --into <other-branch>.",
            file=sys.stderr,
        )
        return 2
    try:
        cfg = load_effective(cwd, config_path).config
    except ConfigError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    # An orphaned fan-out lane (coordinator died before importing it) is
    # adopted here: fetch its branch from the lane clone and replace the
    # live-view symlink, then merge like any run.
    try:
        adopted = adopt_orphan_lane(cwd, cfg, layout, manifest)
    except SubrunError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
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
        print(
            f"ERROR: run ref {run_branch!r} no longer exists; its commits survive at"
            f" {chain_ref_for(manifest.session_id)}.",
            file=sys.stderr,
        )
        return 2
    if not branch_exists(cwd, target):
        print(
            f"ERROR: target branch {target!r} does not exist; pass --into <existing-branch>.",
            file=sys.stderr,
        )
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
        print(f"ERROR: {exc}", file=sys.stderr)
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
        print(f"ERROR: {outcome.error}", file=sys.stderr)
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
    if outcome.status == "noop":
        print(f"[agent6] nothing left to merge from {plan.run_branch} into {plan.target}.")
        return 0
    note = (
        f"\n  (merge record could not be written: {outcome.stamp_error};"
        " `sessions prune` will call this branch unmerged)"
        if outcome.stamp_error
        else ""
    )
    print(
        f"[agent6] merged {plan.run_branch} into {plan.target} "
        f"({plan.strategy}) -> {outcome.merged_sha[:12]}{note}"
    )
    return 0


def _manifest_merged_into(state_dir: Path, branch: str) -> str:
    """The base branch the run owning *branch* (agent6/<session_id>) was merged into, or
    "" if there is no (readable) manifest or it was never recorded as merged.

    Frozen semantics: `sessions prune --delete-squashed` force-deletes a branch ONLY
    when this returns a base name (a manifest-confirmed merge with a recorded sha).
    An unreadable/corrupt/unmerged manifest returns "" -> the branch is KEPT, never
    force-deleted (fail-safe). Only the nested `merged` stamp counts (superseded
    keys are dropped on read), and any parse failure raises ManifestError -> ""."""
    manifest = manifest_for_branch(state_dir, branch)
    if manifest is None:
        return ""
    return manifest.merged.into if (manifest.merged and manifest.merged.sha) else ""


def _merged_tip(state_dir: Path, branch: str) -> str:
    """The run-branch tip recorded when the merge happened, or "" when none was
    recorded (a pre-`tip` manifest, or no readable manifest)."""
    manifest = manifest_for_branch(state_dir, branch)
    if manifest is None:
        return ""
    return manifest.merged.tip if manifest.merged else ""


def _prune_squash_merged(
    cwd: Path, br: str, merged_into: str, *, state_dir: Path, delete_squashed: bool
) -> bool:
    """Handle one squash-merged run branch: force-delete it when that is
    provably content-safe, else keep it and say exactly why. Returns whether it
    was deleted.

    The force-delete is content-safe only when the branch STILL POINTS where the
    merge left it: a resumed run keeps committing on the same branch under the
    same merge stamp, and those commits are in no other ref."""
    sha = branch_tip_sha(cwd, br)
    recorded_tip = _merged_tip(state_dir, br)
    if sha is not None and recorded_tip and recorded_tip != sha:
        print(
            f"[agent6] kept {br} (squash-merged into {merged_into}, but the branch"
            f" advanced since the merge; review, then: git branch -D {br})"
        )
        return False
    # Confirmable = the manifest recorded the merged tip AND the base it went
    # into still exists. Both are needed to prove the delete is content-safe,
    # so both decide the advice: pointing at --delete-squashed for a branch it
    # will skip is a loop, whether or not the operator already ran it.
    confirmable = bool(recorded_tip) and branch_exists(cwd, merged_into)
    if delete_squashed and confirmable and sha is not None:
        if force_delete_squash_merged_branch(cwd, br):
            print(f"[agent6] deleted {br} (squash-merged into {merged_into})")
            # A faded undelete hint: the commit survives in the reflog until GC.
            print(sgr(f"          undelete: git branch {br} {sha[:12]}", "2"))
            return True
        print(f"[agent6] kept {br} (squash-merged into {merged_into}; git refused the delete)")
        return False
    if not confirmable:
        missing = "no recorded merge tip" if not recorded_tip else f"no {merged_into} branch"
        print(
            f"[agent6] kept {br} (squash-merged into {merged_into}, but there is"
            f" {missing} to confirm it; review, then: git branch -D {br})"
        )
        return False
    print(
        f"[agent6] kept {br} (squash-merged into {merged_into}, unreachable; "
        f"remove with: sessions prune --delete-squashed, or: git branch -D {br})"
    )
    return False


def _cmd_prune(*, delete_squashed: bool = False, config_path: Path | None = None) -> int:
    """Delete agent6/* run branches that `git branch -d` can safely remove
    (reachable-merged into HEAD, i.e. merge/ff strategies). Report squash-merged
    ones and unmerged ones (review first). Sweep fan-out clone dirs whose every
    lane branch tip already exists in this repo (content-safe by commit proof;
    a clone holding any commit this repo lacks is kept whole), and the
    worktrees of merged forks (an unmerged fork keeps its worktree; `sessions
    rm` removes a fork's with its record).

    With `--delete-squashed` also force-delete branches the manifest confirms
    were squash-merged into an existing base -- their content is safe in that
    base commit, and each deletion prints the exact command to undelete it (the
    commit survives in the reflog until GC). Unmerged branches are never
    force-deleted."""
    cwd = Path.cwd()
    if not is_git_repo(cwd):
        print("ERROR: not a git repository", file=sys.stderr)
        return 2
    branches = list_run_branches(cwd)
    try:
        current = git_status(cwd).branch
    except GitError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    state_dir = resolved_state_dir(cwd)
    deleted = squashed_deleted = merged_kept = unmerged_kept = 0
    for br in branches:
        if br == current:
            print(f"[agent6] skipped {br} (checked out)", file=sys.stderr)
            continue
        if delete_branch_if_merged(cwd, br):
            deleted += 1
            print(f"[agent6] deleted {br} (merged)")
            continue
        merged_into = _manifest_merged_into(state_dir, br)
        if not merged_into:
            unmerged_kept += 1
            print(f"[agent6] kept {br} (NOT merged; review, then: git branch -D {br})")
            continue
        reachable = branch_exists(cwd, merged_into) and is_ancestor(cwd, br, merged_into)
        if reachable:
            # Reachable-merged into its base, so `git branch -d` only refused because
            # HEAD is not the base; deleting it cleanly needs to run from the base.
            merged_kept += 1
            print(
                f"[agent6] kept {br} (merged into {merged_into} but not reachable from "
                f"{current!r}; re-run prune on {merged_into}, or: git branch -D {br})"
            )
            continue
        # Squash-merged into its base: content is in the base commit but the
        # branch is unreachable, so `git branch -d` refuses it.
        if _prune_squash_merged(
            cwd, br, merged_into, state_dir=state_dir, delete_squashed=delete_squashed
        ):
            squashed_deleted += 1
        else:
            merged_kept += 1
    # Chain refs are pruned whether or not any run BRANCH survives: with
    # `branch_per_run` off there is never one, and once prune has deleted the
    # last branch the refs it kept for a later pass would be unreachable by
    # this command forever.
    refs_deleted, refs_kept = _prune_chain_refs(cwd, state_dir, delete_squashed=delete_squashed)
    clones_note, swept_any = _sweep_workdirs(cwd, state_dir, config_path)
    if not branches and not (refs_deleted or refs_kept or swept_any):
        print("[agent6] nothing to prune: no agent6/* run branches, no chain refs.")
        return 0
    kept = merged_kept + unmerged_kept
    total_deleted = deleted + squashed_deleted
    squashed_note = f" ({squashed_deleted} squash-merged)" if squashed_deleted else ""
    refs_note = (
        f"; chain refs: deleted {refs_deleted}, kept {refs_kept}"
        if refs_deleted or refs_kept
        else ""
    )
    print(
        f"\n[agent6] deleted {total_deleted}{squashed_note}; kept {kept} "
        f"({merged_kept} merged, {unmerged_kept} unmerged){refs_note}{clones_note}",
    )
    return 0


def _sweep_workdirs(cwd: Path, state_dir: Path, config_path: Path | None) -> tuple[str, bool]:
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
    worktrees_removed, worktrees_kept = sweep_fork_worktrees(cwd, state_dir)
    for fork_id, note in worktrees_removed:
        print(f"[agent6] removed {fork_id}'s worktree (merged)" + (f"; {note}" if note else ""))
    for fork_id, why in worktrees_kept:
        print(f"[agent6] kept {fork_id}'s worktree ({why})")
    note = (
        f"; fan-out clones: swept {clones_swept}, kept {clones_kept}"
        if clones_swept or clones_kept
        else ""
    )
    return note, bool(clones_swept or clones_kept or worktrees_removed)


def _prune_chain_refs(cwd: Path, state_dir: Path, *, delete_squashed: bool) -> tuple[int, int]:
    """Drop `refs/agent6/<id>/head` chain refs whose manifest confirms the run
    merged, under the same safety rules as branches: reachable-from-base
    deletes outright; a squash-merge (content in the base commit, ref
    unreachable) deletes only with --delete-squashed AND only while the ref
    still points at the recorded merged tip. Live runs, unmerged runs, and
    refs with no run manifest (machine chains) are kept; keeps are silent --
    an unmerged ref is the run's anchor, not clutter. Returns (deleted, kept
    merged-but-not-deletable)."""
    refs_deleted = refs_kept = 0
    for sid, sha in list_chain_refs(cwd):
        layout = session_layout(state_dir, sid)
        if layout is None or worker_is_alive(layout.session_dir):
            continue  # no session record (a machine chain), or still running
        try:
            manifest = read_manifest(layout.session_dir)
        except ManifestError:
            continue
        if not (manifest.merged and manifest.merged.sha):
            continue
        into = manifest.merged.into
        ref = chain_ref_for(sid)
        if branch_exists(cwd, into) and is_ancestor(cwd, sha, into):
            delete_ref(cwd, ref)
            refs_deleted += 1
            print(f"[agent6] deleted {ref} (merged into {into})")
        elif delete_squashed and manifest.merged.tip == sha:
            delete_ref(cwd, ref)
            refs_deleted += 1
            print(f"[agent6] deleted {ref} (squash-merged into {into})")
        else:
            refs_kept += 1
    return refs_deleted, refs_kept
