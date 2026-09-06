# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The merge engine shared by `sessions merge` and `git.auto_merge`.

`cli.sessions_merge` validates + resolves a run, then calls `execute_merge`; the run
finalizer (`app.finalize.finalize_auto_merge`) calls it directly with the run
context it already holds. Landing is pure ref plumbing (`git_ops.plumb_merge`):
no checkout, no clean-tree requirement -- the worktree that necessarily carries
the run's own work after every run is never an obstacle. One place to mutate
means both honor the same strategy dispatch and manifest record."""

from __future__ import annotations

import datetime as _dt
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from agent6.app._setup import apply_git_ops_policy
from agent6.app.manifest import write_manifest
from agent6.app.providers import InstrumentedProvider, build_role_provider
from agent6.budget import BudgetTracker
from agent6.commit_message import CommitRow, condense_commit_message, conventional_commit_subject
from agent6.config import Config
from agent6.events import EventSink
from agent6.git_ops import (
    CommitIdentity,
    GitError,
    MergeResult,
    branch_exists,
    branch_tip_sha,
    chain_tip,
    is_ancestor,
    list_run_commits,
    plumb_merge,
    range_name_status,
)
from agent6.providers import TranscriptSink, call_for_text
from agent6.sessions.layout import SessionLayout
from agent6.sessions.manifest import (
    NO_MERGE_COMMIT,
    ManifestError,
    MergeStamp,
    SessionManifest,
    read_manifest,
)


@dataclass(frozen=True, slots=True)
class MergeOutcome:
    """Result of execute_merge. `status` is merged / noop / conflict / error; the
    other fields carry that status's detail.

    `noop` is a branch the target already holds the content of: git stages
    nothing and leaves the target where it was, so `merged_sha` is the
    target's own tip. A run whose merge is a noop is merged all the same
    (its content is on the target), and `recorded` says this call stamped
    the manifest so, with NO_MERGE_COMMIT for the sha: a first merge, or one
    covering commits the earlier record does not (a resumed run's, on the
    target by another route). A noop over the tip already recorded leaves
    the record of the merge that did happen alone."""

    status: Literal["merged", "noop", "conflict", "error"]
    merged_sha: str = ""
    conflicts: tuple[str, ...] = ()
    error: str = ""
    # Why the manifest stamp did not land, "" when it did. The merge happened
    # either way; without this, `prune` calls the branch unmerged.
    stamp_error: str = ""
    recorded: bool = False
    # Paths the checkout kept its own version of (`left_behind_line`).
    left_behind: tuple[str, ...] = ()


def record_merge_in_manifest(
    layout: SessionLayout, *, merged_into: str, merged_sha: str, merged_tip: str = ""
) -> str:
    """Record a successful merge in the run manifest so later tooling can tell a
    merged run branch from an unmerged one. *merged_tip* is the run-branch tip
    that was merged: `sessions prune --delete-squashed` force-deletes only a branch
    still pointing there. Best-effort: a missing/corrupt manifest must not fail a
    merge that already happened.

    Returns "" when the stamp landed, else why it did not. Silence made `prune`
    call a branch agent6 had merged minutes earlier "NOT merged", and left
    `--delete-squashed` unable to clean it up ever."""
    try:
        m = read_manifest(layout.session_dir)
    except ManifestError as exc:
        return str(exc)
    stamped = m.model_copy(
        update={
            "merged": MergeStamp(
                into=merged_into,
                sha=merged_sha,
                tip=merged_tip,
                ts=_dt.datetime.now(tz=_dt.UTC).isoformat(timespec="seconds"),
            )
        }
    )
    # Also ManifestError: a manifest newer than this binary can rewrite is left
    # alone rather than downgraded, and the merge it records already happened.
    try:
        write_manifest(layout.manifest_path, stamped)
    except (OSError, ManifestError) as exc:
        return str(exc)
    return ""


def dispatch_merge(
    cwd: Path,
    strategy: str,
    target: str,
    run_branch: str,
    base_sha: str,
    manifest: SessionManifest,
    message: str | None,
    cfg: Config,
    identity: CommitIdentity,
    *,
    transcript_dir: Path | None = None,
    budget: BudgetTracker | None = None,
    events: EventSink | None = None,
    warn: Callable[[str], None] = lambda _m: None,
    merge_base: str | None = None,
) -> MergeResult:
    """Run the chosen strategy on *target* via plumb_merge. squash builds its
    message per `[git.commit.squash].message` (the trailer is identity's);
    an operator *message* overrides any style."""
    if strategy == "squash" and message is None:
        message = _squash_message(
            cwd,
            cfg,
            manifest,
            base_sha=base_sha,
            run_branch=run_branch,
            transcript_dir=transcript_dir,
            budget=budget,
            events=events,
            warn=warn,
        )
    if message is None and strategy == "merge":
        message = f"Merge {run_branch}"
    return plumb_merge(
        cwd,
        target,
        run_branch,
        strategy=strategy,
        message=message,
        identity=identity,
        merge_base=merge_base,
    )


def landed_base(
    cwd: Path, layout: SessionLayout, manifest: SessionManifest, target: str, tip: str
) -> str | None:
    """The merge base for a run whose chain *target* already holds as a squash,
    from its own stamp (a resumed leg merging again) or an ancestor's (a fork),
    or None where git's own base serves.

    A squash commit is content git cannot relate to the chain it came from, so
    git's own base reads every commit before it as new work against the
    squash that holds the same lines. The base is the merged tip when the chain
    continues past it, else the point a fork left its ancestor's chain (the
    merged tip holds that point's content)."""
    node, fork_point = manifest, ""
    for _ in range(64):  # a lineage deeper than this is not a fork chain
        stamp = node.merged
        if stamp is not None and stamp.into == target and stamp.tip:
            if is_ancestor(cwd, stamp.tip, target):
                break  # a real merge: git relates the two histories itself
            if is_ancestor(cwd, stamp.tip, tip):
                return stamp.tip
            if fork_point and is_ancestor(cwd, fork_point, stamp.tip):
                return fork_point
            break
        fork_point = node.forked_from_sha
        if not node.parent_session_id:
            break
        try:
            node = read_manifest(layout.session_dir.parent / node.parent_session_id)
        except (ManifestError, OSError):
            break
    return None


def _squash_message(
    cwd: Path,
    cfg: Config,
    manifest: SessionManifest,
    *,
    base_sha: str,
    run_branch: str,
    transcript_dir: Path | None,
    budget: BudgetTracker | None,
    events: EventSink | None,
    warn: Callable[[str], None],
) -> str | None:
    """The squash commit's message per `[git.commit.squash].message`; None
    means let git combine (its own SQUASH_MSG)."""
    style = cfg.git.commit.squash.message
    rows = list_run_commits(cwd, base_sha, run_branch)
    if style == "combine":
        # Git's own SQUASH_MSG shape, synthesized (the plumbing merge never
        # runs `merge --squash`, so git never writes one).
        parts = ["Squashed commit of the following:\n"]
        parts += [f"commit {r.sha}\n\n    {r.subject}" for r in rows]
        return "\n".join(parts) if rows else None
    base_msg = condense_commit_message(rows, subject=manifest.user_task or "agent6 run")
    if style == "conventional":
        subject = conventional_commit_subject(
            range_name_status(cwd, base_sha, run_branch), summary=base_msg.splitlines()[0]
        )
        return "\n".join([subject, *base_msg.splitlines()[1:]])
    if style == "model":
        msg = _model_squash_message(
            cwd,
            cfg,
            rows,
            base_sha=base_sha,
            run_branch=run_branch,
            task=manifest.user_task or "agent6 run",
            transcript_dir=transcript_dir,
            budget=budget,
            events=events,
        )
        if msg:
            return msg
        warn("model squash message failed; using the agent6 style")
    return base_msg


def _model_squash_message(
    cwd: Path,
    cfg: Config,
    rows: tuple[CommitRow, ...],
    *,
    base_sha: str,
    run_branch: str,
    task: str,
    transcript_dir: Path | None,
    budget: BudgetTracker | None,
    events: EventSink | None,
) -> str | None:
    """One provider call writing the squash message from git facts only; None
    on any failure (the caller degrades to the agent6 style).

    *budget* is the RUN's tracker when auto_merge runs inside a run: the call
    spends the run's remainder, not a fresh full cap. `sessions merge` is its
    own invocation and passes None (per-invocation ceiling, as everywhere).
    With *events* the call is instrumented, so its spend reaches the log."""
    if transcript_dir is None:
        return None
    try:
        tracker = (
            budget
            if budget is not None
            else BudgetTracker(
                max_usd=cfg.budget.max_usd,
                max_percent=cfg.budget.max_percent,
                allow_paid_credits=cfg.budget.allow_paid_credits,
                max_tokens_fallback=cfg.budget.max_tokens_fallback,
            )
        )
        provider = build_role_provider(
            cfg,
            "worker",
            transcript_sink=TranscriptSink(transcript_dir),
            budget=tracker,
        )
        if events is not None:
            rm = cfg.models.resolve("worker")
            provider = InstrumentedProvider(
                inner=provider,
                role="squash",
                model=rm.model if rm is not None else "",
                provider_name=rm.provider if rm is not None else "",
                events=events,
                budget=tracker,
            )
        steps = "\n".join(f"- {r.subject}" for r in rows[:100])
        files = "\n".join(
            f"{s}\t{p}" for s, p in range_name_status(cwd, base_sha, run_branch)[:200]
        )
        return call_for_text(
            provider,
            system=(
                "Write a git commit message for a squashed branch: one"
                " imperative subject line under 72 characters, a blank line,"
                " then a short body. Use only the facts given. Output the"
                " message text only."
            ),
            user=(
                f"Task: {task}\nPer-step subjects:\n{steps}\nChanged files (status\tpath):\n{files}"
            ),
            max_tokens=500,
        )
    except Exception:
        # Building the drafting provider is best-effort too; the caller
        # falls back to a fixed subject.
        return None


def execute_merge(
    cwd: Path,
    *,
    layout: SessionLayout,
    manifest: SessionManifest,
    run_branch: str,
    target: str,
    base_sha: str,
    strategy: str,
    message: str | None,
    cfg: Config,
    identity: CommitIdentity,
    budget: BudgetTracker | None = None,
    events: EventSink | None = None,
    warn: Callable[[str], None] = lambda _m: None,
) -> MergeOutcome:
    """Land *run_branch* (a branch name or the run's chain ref) on *target*
    with *strategy* and record the merge. Ref plumbing only: the checkout is
    never switched and the worktree is never required clean. The caller
    validates first; this mutates."""
    apply_git_ops_policy(cfg)
    if not branch_exists(cwd, target):
        # The merge target must already exist; never fabricate it. sessions merge
        # pre-checks this for a nicer message; auto_merge relies on this guard
        # if the base was deleted mid-run.
        return MergeOutcome("error", error=f"target branch {target!r} does not exist")
    if (
        strategy == "ff"
        and not is_ancestor(cwd, target, run_branch)
        and not is_ancestor(cwd, run_branch, target)
    ):
        # Pre-check the fast-forward so the refusal names the agent6 remedy
        # (auto_merge with an ff config would otherwise fail raw on a moved
        # base). A run the target already CONTAINS is not refused: that is a
        # clean no-op below.
        return MergeOutcome(
            "error",
            error=(
                f"{target!r} has moved since the run started, so a"
                " fast-forward is impossible; merge with --strategy merge or"
                " squash instead"
            ),
        )
    # Where the target stood before the merge touched it. Every strategy that merges
    # something moves it (squash commits, merge commits, a fast-forward), so an
    # unmoved target means there was nothing to merge.
    target_tip_before = branch_tip_sha(cwd, target) or ""
    try:
        merge_base = landed_base(cwd, layout, manifest, target, chain_tip(cwd, run_branch) or "")
        result = dispatch_merge(
            cwd,
            strategy,
            target,
            run_branch,
            base_sha,
            manifest,
            message,
            cfg,
            identity,
            transcript_dir=layout.session_dir / "transcripts",
            budget=budget,
            events=events,
            warn=warn,
            merge_base=merge_base,
        )
    except GitError as exc:
        return MergeOutcome("error", error=f"merge failed: {exc}")
    if result.conflicted:
        return MergeOutcome("conflict", conflicts=result.conflicts)
    # Nothing merged when the target did not move: it already holds the
    # branch's content.
    noop = bool(result.merged_sha) and result.merged_sha == target_tip_before
    merged_tip = chain_tip(cwd, run_branch) or ""
    if noop and manifest.merged is not None and manifest.merged.tip == merged_tip:
        # The record already covers this tip: stamping the target's own tip
        # over it would credit the run with whatever was committed there since
        # and destroy the record of the merge that did happen.
        return MergeOutcome("noop", merged_sha=result.merged_sha)
    # A merge that added nothing still records the run as merged up to this
    # tip: the target holds its content, so prune and the listings treat it so.
    stamp_error = record_merge_in_manifest(
        layout,
        merged_into=target,
        merged_sha=NO_MERGE_COMMIT if noop else result.merged_sha,
        merged_tip=merged_tip,
    )
    return MergeOutcome(
        "noop" if noop else "merged",
        merged_sha=result.merged_sha,
        stamp_error=stamp_error,
        recorded=noop and not stamp_error,
        left_behind=result.left_behind,
    )


def left_behind_line(target: str, outcome: MergeOutcome) -> str:
    """The one line for merged files the checkout keeps its own version of, on
    every surface; "" when it holds everything the merge landed.

    The merge is ref plumbing: it moves the branch and brings the checkout
    forward only where a file still matches what the branch held, so an edit
    of the operator's (or another run's work sitting in the tree) is never
    overwritten. This line says so, since the tree the operator then tests,
    and commits, holds the older content."""
    if not outcome.left_behind:
        return ""
    named = ", ".join(outcome.left_behind[:4]) + (", ..." if len(outcome.left_behind) > 4 else "")
    return f"your checkout keeps its own {named}; {target} holds what was merged"


def noop_merge_line(run_branch: str, target: str, outcome: MergeOutcome) -> str:
    """The one line for a merge that added nothing, on every surface: what
    this call recorded, or that the record already stood (a stamp error is
    the caller's note)."""
    if outcome.recorded or outcome.stamp_error:
        recorded = f"; recorded as merged into {target}" if outcome.recorded else ""
        return f"nothing to add from {run_branch}: {target} already has its content{recorded}"
    return f"nothing left to merge from {run_branch} into {target}"
