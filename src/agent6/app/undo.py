# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The `/undo` rewind: take back a session's last message by cloning its
state as of the checkpoint before it (:func:`agent6.app.fork.create_fork`)
into a fork that keeps the undone session's checkout, then putting that
checkout back to the checkpoint's tree.

Nothing is lost: the tree as it stands (the session's in-flight edits, every
file that appeared since it started) is committed onto the undone session's
ref first, so the later commits and that one stay there; the session's
untracked-at-start files (the operator's) are left alone, and so are HEAD
and the index.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agent6.app.fork import Checkout, create_fork, resolve_source
from agent6.app.reporter import STDIO_REPORTER, Reporter
from agent6.app.resume import commits_note
from agent6.config import ConfigError
from agent6.config.layer import load_effective
from agent6.git_ops import (
    CommitIdentity,
    GitError,
    chain_commit,
    chain_ref_for,
    chain_tip,
    sync_worktree,
    tree_diff_paths,
    worktree_tree,
)
from agent6.graph.storage import (
    list_checkpoint_turns,
)
from agent6.paths import state_dir
from agent6.sessions.ipc import read_worker_pid
from agent6.sessions.layout import (
    SessionLayout,
    read_untracked_at_start,
)
from agent6.sessions.lock import (
    acquire_repo_writer,
    release_single_writer,
    repo_writer_holder,
)
from agent6.sessions.manifest import (
    ManifestError,
    model_git_refusal,
    read_manifest,
)
from agent6.task_text import operator_task_text
from agent6.workflows._session_state import SessionSnapshot, load_session_snapshot

_STEER_NOTICE = "OPERATOR STEERING"


def _text_of(content: object) -> str:
    """The plain text of an anthropic-shaped message content (str or blocks)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [
            str(block.get("text", ""))
            for block in content  # pyright: ignore[reportUnknownVariableType]
            if isinstance(block, dict) and block.get("type") == "text"
        ]
        return "\n".join(part for part in parts if part)
    return ""


def _operator_messages(messages: list[dict[str, Any]]) -> list[str]:
    """The operator's words in a restored conversation: the opening task, then
    every steer notice with its wrapper line stripped. Tool results and other
    harness notices stay out."""
    out: list[str] = []
    for i, message in enumerate(messages):
        if message.get("role") != "user":
            continue
        text = _text_of(message.get("content"))
        if not text:
            continue
        if i == 0:
            out.append(text)
        elif text.startswith(_STEER_NOTICE):
            body = text.partition("\n")[2].strip()
            out.append(body or text)
    return out


@dataclass(frozen=True, slots=True)
class _Checkpoint:
    """A checkpoint of *session_id*: the file it sits in (`at_turn`, what
    `fork --at-turn` addresses) and the turn its conversation stands before
    (`turn`, the snapshot's `next_iteration`, the number every surface
    prints). The two differ for a fork's seed, file 0 holding its source's
    turn N."""

    session_id: str
    at_turn: int
    turn: int


@dataclass(frozen=True, slots=True)
class UndoTarget:
    """Where `/undo` forks *session*: *source*'s checkpoint at *at_turn* (the
    session itself, or an ancestor up its fork lineage), the turn that
    checkpoint stands before (see :class:`_Checkpoint`), and the message it
    takes back (composer-refill text)."""

    session: SessionLayout
    source_session_id: str
    at_turn: int
    turn: int
    undone_text: str


def _snapshot_at(layout: SessionLayout, at_turn: int) -> SessionSnapshot | None:
    """The checkpoint in file *at_turn*, None if unreadable."""
    try:
        return load_session_snapshot(layout.checkpoint_path(at_turn))
    except (OSError, ValueError):
        return None


def undo_target(  # noqa: PLR0911 - each refusal names its own reason
    state_dir: Path, session_id: str, *, reporter: Reporter = STDIO_REPORTER
) -> UndoTarget | None:
    """Resolve `/undo` for *session_id*: the newest checkpoint -- in this
    session or up its fork lineage -- whose restored conversation ends before
    the session's last operator message. With only the opening task, the
    earliest checkpoint (start over, task back in the composer). None, with
    the reason printed, when nothing qualifies."""
    src = resolve_source(state_dir, session_id, reporter=reporter)
    if src is None:
        return None
    turns = sorted(list_checkpoint_turns(src))
    if not turns:
        reporter.err(f"nothing to undo: {src.session_id} has no checkpoints.")
        return None
    newest = turns[-1]
    try:
        snap = load_session_snapshot(src.checkpoint_path(newest))
    except (OSError, ValueError) as exc:
        reporter.error(f"cannot read checkpoint {newest} of {src.session_id}: {exc}")
        return None
    ops = _operator_messages(snap.messages)
    if len(ops) <= 1:
        # Only the opening task: /undo means start over from the first
        # checkpoint, with the task back in the composer to edit.
        if len(turns) < 2:
            reporter.err(f"nothing to undo: {src.session_id} is at its opening message.")
            return None
        first = _snapshot_at(src, turns[0])
        if first is None:
            reporter.error(f"cannot read checkpoint {turns[0]} of {src.session_id}.")
            return None
        try:
            task = read_manifest(src.session_dir).user_task
        except ManifestError:
            task = operator_task_text(ops[0]) if ops else ""
        return UndoTarget(src, src.session_id, turns[0], first.next_iteration, task)
    target = _newest_checkpoint_below(src, len(ops))
    if target is None:
        reporter.err(f"nothing to undo: no state before the last message of {src.session_id}.")
        return None
    return UndoTarget(src, target.session_id, target.at_turn, target.turn, ops[-1])


def _newest_checkpoint_below(
    layout: SessionLayout, current_ops: int, *, seen: frozenset[str] = frozenset()
) -> _Checkpoint | None:
    """The newest checkpoint of *layout* -- or, following fork lineage, of an
    ancestor -- whose conversation holds fewer operator messages than
    *current_ops*. A fork carries one seed checkpoint, so walking back past it
    means resolving in the parent it was cut from.

    *seen* stops a cyclic lineage: forks always point at an OLDER run, so a
    cycle only exists in a corrupt or hand-edited manifest, and following one
    would recurse until the stack blows. A revisited id ends the walk (no
    resolvable ancestor) instead of crashing."""
    for at_turn in sorted(list_checkpoint_turns(layout), reverse=True):
        snap = _snapshot_at(layout, at_turn)
        if snap is not None and len(_operator_messages(snap.messages)) < current_ops:
            return _Checkpoint(layout.session_id, at_turn, snap.next_iteration)
    try:
        parent = read_manifest(layout.session_dir).parent_session_id
    except ManifestError:
        return None
    if not parent or parent in seen:
        return None
    parent_layout = SessionLayout(
        state_dir=layout.state_dir, session_id=parent, subdir=layout.subdir
    )
    if not parent_layout.session_dir.is_dir():
        return None
    return _newest_checkpoint_below(parent_layout, current_ops, seen=seen | {layout.session_id})


def _rewind_checkout(checkout: Path, *, tip: str, sha: str, exclude: frozenset[str]) -> list[str]:
    """Put *checkout* back to *sha*'s tree for every tracked path whose
    content differs from it (minus *exclude*, the session's untracked-at-start
    files); HEAD and the shared index stay untouched. Returns the paths put
    back. The current content is staged as a tree first (seeded on *tip*, the
    chain commit that holds it), so the two-tree sync moves exactly the paths
    that differ and nothing identical is rewritten."""
    current = worktree_tree(checkout, tip, exclude)
    paths = tree_diff_paths(checkout, sha, current)
    if paths:
        sync_worktree(checkout, current, sha)
    return paths


def undo_fork(  # noqa: PLR0911 - each refusal names its own reason
    config_path: Path | None,
    session_id: str,
    *,
    cwd: Path,
    reporter: Reporter = STDIO_REPORTER,
) -> tuple[str, str] | None:
    """`/undo`: commit the tree as it stands onto *session_id*'s ref, fork the
    session at its undo target (unstarted, in the session's own checkout), and
    put that checkout back to the target's tree. Returns `(child_id,
    undone_text)`: the text goes back in the composer to edit and resend. None
    with the reason already printed.

    *cwd* is the repository (the state dir's anchor); the checkout rewound is
    the session's worktree when it has one, else *cwd*. The checkout's writer
    lock is held across the commit and the rewind, unless this process is the
    session's live worker (the loop's own `/undo`, which holds it already):
    any other live run driving the checkout refuses."""
    state = state_dir(cwd)
    target = undo_target(state, session_id, reporter=reporter)
    if target is None:
        return None
    undone = target.session
    try:
        manifest = read_manifest(undone.session_dir)
    except ManifestError as exc:
        reporter.error(f"cannot read the manifest of {undone.session_id}: {exc}")
        return None
    refusal = model_git_refusal(manifest, "undo")
    if refusal is not None:
        # Before the commit below: a model-controlled run has no agent6 chain,
        # and a chain ref written here is one auto_merge would land.
        reporter.error(refusal)
        return None
    checkout = manifest.worktree or cwd
    if manifest.worktree is not None and manifest.worktree_git_dir is None:
        reporter.error(
            f"cannot undo {undone.session_id}: its manifest names a worktree but not the"
            f" repository git dir it points into; `agent6 fork {undone.session_id}` continues"
            " its commits in a new worktree."
        )
        return None
    if manifest.worktree is not None and not (checkout / ".git").exists():
        reporter.error(
            f"cannot undo {undone.session_id}: its worktree {checkout} is gone (pruned or"
            f" removed); {commits_note(cwd, manifest)}; `agent6 fork {undone.session_id}`"
            " continues it in a new worktree."
        )
        return None
    try:
        cfg = load_effective(cwd, config_path).config
    except ConfigError as exc:
        reporter.error(str(exc))
        return None
    lock_fd: int | None = None
    if read_worker_pid(undone.session_dir) != os.getpid():
        lock_fd = acquire_repo_writer(state, checkout, undone.session_id)
        if lock_fd is None:
            holder = repo_writer_holder(state, checkout) or "another run"
            reporter.refuse(
                f"run {holder!r} is driving this checkout, and /undo would put the tree back"
                " under it. Stop it first:\n"
                f"    agent6 sessions stop {holder}"
            )
            return None
    ref = chain_ref_for(undone.session_id)
    where = manifest.run_branch or ref
    exclude = read_untracked_at_start(undone.session_dir)
    try:
        try:
            kept = chain_commit(
                checkout,
                f"agent6 undo: the tree before turn {target.turn} was taken back",
                ref=ref,
                fallback_parent=manifest.base_sha or None,
                identity=CommitIdentity(name=cfg.git.commit.name, email=cfg.git.commit.email),
                also_branch=manifest.run_branch,
                exclude=exclude,
            )
        except GitError as exc:
            reporter.error(f"the tree as it stands could not be committed onto {where}: {exc}")
            return None
        child, rc = create_fork(
            config_path,
            target.source_session_id,
            at_turn=target.at_turn,
            cwd=cwd,
            worktree=False,
            checkout=(
                Checkout(manifest.worktree, manifest.worktree_git_dir)
                if manifest.worktree is not None and manifest.worktree_git_dir is not None
                else None
            ),
            checkout_untracked=exclude,
            reporter=reporter,
        )
        if rc != 0:
            return None
        child_layout = SessionLayout(state_dir=state, session_id=child, subdir=undone.subdir)
        sha = read_manifest(child_layout.session_dir).forked_from_sha or ""
        turn = f"turn {target.turn} ({sha[:12]})"
        try:
            paths = _rewind_checkout(
                checkout, tip=chain_tip(checkout, ref) or sha, sha=sha, exclude=exclude
            )
        except GitError as exc:
            reporter.error(
                f"the checkout was NOT put back to {turn}: {exc}."
                f" Fork {child} continues from the tree as it is."
            )
            return child, target.undone_text
    finally:
        release_single_writer(lock_fd)
    _report_rewind(reporter, paths, turn=turn, kept=kept, where=where)
    return child, target.undone_text


def _report_rewind(
    reporter: Reporter, paths: list[str], *, turn: str, kept: str | None, where: str
) -> None:
    """The undo notice: the paths put back (HEAD and the index stay), and
    where the tree as it stood and the later commits live."""
    if paths:
        shown = ", ".join(paths[:10]) + (f", +{len(paths) - 10} more" if len(paths) > 10 else "")
        reporter.note(
            f"put {len(paths)} path(s) back to {turn}: {shown} (HEAD and the index are untouched)"
        )
    else:
        reporter.note(f"the checkout already matches {turn}")
    stood = f"the tree as it stood is commit {kept[:12]} on {where}; " if kept else ""
    reporter.note(f"{stood}the later commits stay on {where}")
