# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The `agent6 fork` lifecycle: clone a run's state as of a checkpoint into a
NEW run, and the `/undo` rewind built on the same clone.

A fork copies a source run's state, as of checkpoint turn N, into a fresh run
dir with a new id and the same repo, recording lineage (parent run + the
turn). The source run is never mutated: sessions as trees, done as
clone-to-new-session. `ui/cli/fork.py` adapts argv, calls :func:`create_fork`,
then (unless `--no-run`) continues the new run from turn N over the resume
path.

A fork of a run is the repo at the checkpoint's committed HEAD, in its own
worktree, plus the conversation up to that turn. `create_fork` adds a linked
git worktree detached at that sha (under `[parallel].workdir`, beside the
lane clones) and records it in the manifest: the fork's chain grows there,
`agent6 resume <fork>` runs the leg there, and the source run and the
operator's checkout stay as they are. The worktree shares the repository's
refs, so `sessions diff|commits|merge <fork>` read the fork like any run's,
and `sessions prune` removes the worktree once the fork is merged. A plan or
ask fork edits nothing and reads the operator's checkout, like every plan.
On a gated run (commits fire only on a green verify), an edit made but not
committed at the forked turn is ABSENT from the fork's tree even though the
copied transcript mentions it: the same committed-history-only posture
`resume` documents. The DAG is not copied but REBUILT: the checkpoint's
`graph_version` names an exact past state, and `graph.replay` undoes every
journal-recorded mutation stamped after it, so the fork's tasks, statuses,
and cursor match the turn its conversation came from.

`/undo` (:func:`undo_fork`) clones the state the same way but adds no
worktree: the fork keeps the undone session's checkout, and that checkout is
put back to the checkpoint's tree. Nothing is lost: the tree as it stands
(the session's in-flight edits, every file that appeared since it started)
is committed onto the undone session's ref first, so the later commits and
that one stay there; the session's untracked-at-start files (the operator's)
are left alone, and so are HEAD and the index.
"""

from __future__ import annotations

import contextlib
import datetime as _dt
import json
import os
import shutil
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agent6.app._setup import SandboxOverrides, detect_env, session_config
from agent6.app.manifest import write_session_manifest
from agent6.app.parallel import subordinate_workdir_root
from agent6.app.reporter import STDIO_REPORTER, Reporter
from agent6.app.resume import commits_note, resumable_bucket_dirs
from agent6.config import Config, ConfigError
from agent6.config.layer import load_effective
from agent6.git_ops import (
    CommitIdentity,
    GitError,
    add_worktree,
    chain_commit,
    chain_dirty,
    chain_dirty_paths,
    chain_ref_for,
    chain_tip,
    create_branch_at,
    git_common_dir,
    merge_stamp_holds,
    remove_worktree,
    run_branch_for,
    set_ref,
    sync_worktree,
    tree_diff_paths,
    untracked_paths,
    worktree_tree,
)
from agent6.graph.replay import graph_at_version, journal_prefix
from agent6.graph.storage import (
    append_jsonl,
    flock,
    list_checkpoint_turns,
    load_graph,
    read_cursor,
    write_cursor,
    write_node,
)
from agent6.paths import state_dir
from agent6.portable import atomic_write
from agent6.sandbox.detect import resolve_isolation
from agent6.sessions.id import (
    SessionIdError,
    resolve_session,
    session_id_bucket,
    unused_session_id,
    validate_explicit_session_id,
)
from agent6.sessions.ipc import read_worker_pid, worker_is_alive
from agent6.sessions.layout import (
    SessionLayout,
    read_untracked_at_start,
    write_untracked_at_start,
)
from agent6.sessions.lock import (
    acquire_repo_writer,
    checkout_lock_path,
    release_single_writer,
    repo_writer_holder,
)
from agent6.sessions.manifest import (
    ManifestError,
    SessionManifest,
    model_git_refusal,
    read_manifest,
)
from agent6.task_text import operator_task_text
from agent6.types import ResumableMode, session_bucket
from agent6.viewmodel import newest_session_dir, session_dirs
from agent6.workflows._session_state import SessionSnapshot, load_session_snapshot

# Curator-owned DAG artifacts copied verbatim into the fork; each is a
# top-level entry under the run dir (`graph/` is a directory).
_DAG_ARTIFACTS: tuple[str, ...] = ("graph", "graph.jsonl", "cursor.json")


def _resolve_source(state_dir: Path, query: str, *, reporter: Reporter) -> SessionLayout | None:
    """The session to fork, by id or (empty query) the most recent one.

    The same cross-bucket resolver resume uses: a plan lives in plans/ and an
    ask in asks/, and a runs/-only lookup silently could not see either.
    """
    if not query:
        # The same bucket set bare `resume` uses, so the two agree on what
        # "most recent" means.
        latest = newest_session_dir(resumable_bucket_dirs(state_dir))
        if latest is None:
            reporter.err('nothing to fork yet. Start a session with `agent6 run "<task>"`.')
            return None
        query = latest.name
        reporter.note(f"forking most recent session: {query}")
    try:
        return resolve_session(state_dir, query)
    except SessionIdError as exc:
        reporter.error(str(exc))
        return None


def _copy_dag(src: SessionLayout, dst: SessionLayout, *, graph_version: int) -> None:
    """Write *dst*'s DAG as the source's stood at *graph_version*.

    Reads under the SOURCE curator's per-mutation flock: a live source run's
    curator atomic-renames node files and prunes stale ones mid-read, so an
    unlocked copytree could hit a vanishing file (shutil.Error) or produce a
    torn, mixed-instant DAG. Holding the same lock the curator takes for every
    mutation (graph.storage.flock on <run>/.lock) makes this one consistent
    point-in-time read; a crashed source driver's fcntl lock releases on
    process death, so no stale-lock hang.

    `graph_version <= 0` means the checkpoint predates the stamp or the
    curator was unreadable when it was written: with no version to rebuild at,
    copy the DAG verbatim (what every fork did before replay existed).
    """
    with flock(src.lock_path):
        if graph_version <= 0:
            for name in _DAG_ARTIFACTS:
                src_path = src.session_dir / name
                if not src_path.exists():
                    continue
                dst_path = dst.session_dir / name
                if src_path.is_dir():
                    shutil.copytree(src_path, dst_path, dirs_exist_ok=True, symlinks=True)
                else:
                    shutil.copy2(src_path, dst_path)
            return
        nodes = load_graph(src)
        journal = _read_journal(src)
        replayed = graph_at_version(nodes, journal, graph_version, current_cursor=read_cursor(src))
    dst.ensure()
    for node in replayed.nodes.values():
        write_node(dst, replayed.nodes, node)
    write_cursor(dst, replayed.cursor)
    # The journal prefix, so the fork's curator resumes numbering from the
    # version it actually holds instead of the source's later one.
    kept = journal_prefix(journal, graph_version)
    atomic_write(dst.journal_path, "".join(json.dumps(e, sort_keys=True) + "\n" for e in kept))


def _read_journal(src: SessionLayout) -> list[dict[str, Any]]:
    """The source's graph.jsonl as dicts; a torn or non-object line is skipped
    (the same tolerance `load_graph` gives a malformed node file)."""
    out: list[dict[str, Any]] = []
    try:
        text = src.journal_path.read_text(encoding="utf-8")
    except OSError:
        return out
    for line in text.splitlines():
        if not line.strip():
            continue
        try:
            entry = json.loads(line)
        except ValueError:
            continue
        if isinstance(entry, dict):
            out.append(entry)  # pyright: ignore[reportUnknownArgumentType]
    return out


def _select_checkpoint_path(
    src: SessionLayout, at_turn: int | None, *, reporter: Reporter = STDIO_REPORTER
) -> Path | None:
    """Resolve which snapshot of *src* to fork from, or None on error (printed).

    Default is the rolling `loop_state.json` (the newest state, ahead of the
    last per-turn checkpoint mid-turn); `--at-turn N` selects checkpoint N from
    the per-turn store, refusing turns the store does not hold.
    """
    turns = list_checkpoint_turns(src)
    if at_turn is None:
        rolling = src.session_dir / "loop_state.json"
        if rolling.is_file():
            return rolling
        if turns:
            return src.checkpoint_path(turns[-1])
        reporter.error(
            f"{src.session_id} has no checkpoints and no loop_state.json; nothing to fork."
        )
        return None
    if at_turn in turns:
        return src.checkpoint_path(at_turn)
    avail = ", ".join(str(t) for t in turns) or "none"
    reporter.error(
        f"no checkpoint at turn {at_turn} for {src.session_id}. Available turns: {avail}"
    )
    return None


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
    src = _resolve_source(state_dir, session_id, reporter=reporter)
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


@dataclass(frozen=True, slots=True)
class Checkout:
    """A fork's own checkout: its linked worktree and the repository git dir
    that worktree points into, recorded together (a fork leg's jail grants
    the dir from this record, never from the worktree's `.git` pointer)."""

    worktree: Path
    git_dir: Path


class _ForkRefused(Exception):
    """The fork was refused before anything was created; the reason is
    printed and `rc` is the exit code."""

    def __init__(self, rc: int) -> None:
        super().__init__(rc)
        self.rc = rc


@dataclass(frozen=True, slots=True)
class _ForkPlan:
    """Everything a fork writes, resolved and validated first so a refusal
    creates nothing: the source, the child's layout, the checkpoint to seed
    from, the manifest facts carried forward, and the child's config."""

    src: SessionLayout
    dst: SessionLayout
    checkpoint_path: Path
    graph_version: int
    forked_from_turn: int
    forked_from_sha: str
    base_sha: str
    base_branch: str
    user_task: str
    mode: ResumableMode
    preset: str
    preset_from_flag: bool
    cfg: Config
    # The source's pinned verify command and its origin. A fork inherits it
    # rather than deriving one from the current config: an inferred or adopted
    # gate has no config to derive from.
    gate: tuple[Sequence[str], str]


def _plan_fork(
    config_path: Path | None,
    source_session_id: str,
    *,
    at_turn: int | None = None,
    new_session_id: str = "",
    cwd: Path,
    sandbox_overrides: SandboxOverrides | None = None,
    refuse_continuation: Callable[[Config, str], str | None] | None = None,
    reporter: Reporter = STDIO_REPORTER,
) -> _ForkPlan:
    """Resolve a fork of *source_session_id* at checkpoint *at_turn*; raises
    :class:`_ForkRefused` with the exit code after printing the reason.

    The child's config is built as its continuation builds it (the source's
    preset, the mode clamp, this invocation's *sandbox_overrides*), so the
    manifest stamps the policy the fork runs under. *refuse_continuation*,
    given that config and the mode, returns why the continuation would refuse
    (`headless_approval_refusal` for `agent6 fork` without `--no-run`) or
    None; a reason refuses BEFORE anything is created, the order `run` keeps,
    so no never-started fork stays listed and its id stays free.
    """
    state = state_dir(cwd)
    src = _resolve_source(state, source_session_id, reporter=reporter)
    if src is None:
        raise _ForkRefused(2)

    checkpoint_path = _select_checkpoint_path(src, at_turn, reporter=reporter)
    if checkpoint_path is None:
        raise _ForkRefused(2)

    try:
        checkpoint = load_session_snapshot(checkpoint_path)
    except (OSError, ValueError) as exc:
        reporter.error(f"failed to load checkpoint {checkpoint_path}: {exc}")
        raise _ForkRefused(1) from exc

    # Read the source manifest to carry base_sha / base_branch / mode forward.
    # `mode` is security-relevant: a missing/corrupt source manifest must NOT
    # fall open to the more-privileged "run" (write) mode -- forking a plan run
    # as a write run would hand it the mutating tools. A valid run always wrote
    # a manifest, so a damaged run dir (unreadable, corrupt, or an unknown mode
    # value) fails loud via read_manifest / session_mode rather than silently
    # escalating (same contract as resume).
    try:
        sm = read_manifest(src.session_dir)
        src_mode = sm.session_mode()
    except ManifestError as exc:
        reporter.error(f"cannot read source run manifest {src.manifest_path}: {exc}")
        raise _ForkRefused(2) from exc
    refusal = model_git_refusal(sm, "fork")
    if refusal is not None:
        reporter.error(refusal)
        raise _ForkRefused(2)

    forked_from_sha = checkpoint.head_sha
    if not forked_from_sha:
        reporter.error(
            "the chosen checkpoint records no head_sha, so the fork branch "
            "cannot be cut. (A checkpoint from before per-turn sha capture.)"
        )
        raise _ForkRefused(1)

    try:
        # The source's preset: resume replays it (preset or manifest_preset),
        # so the child manifest's models/workflow stamp must be derived from
        # the SAME preset-resolved config or `sessions show` reports a model the forked
        # run never uses.
        cfg = session_config(
            load_effective(cwd, config_path, preset=sm.workflow.replay_preset).config,
            src_mode,
            sandbox_overrides,
        )
    except ConfigError as exc:
        reporter.error(str(exc))
        raise _ForkRefused(2) from exc
    if refuse_continuation is not None:
        refusal = refuse_continuation(cfg, src_mode)
        if refusal is not None:
            reporter.refuse(refusal)
            raise _ForkRefused(2)

    if new_session_id:
        try:
            validate_explicit_session_id(new_session_id)
        except SessionIdError as exc:
            reporter.error(str(exc))
            raise _ForkRefused(2) from exc
        # Any bucket holding it makes the id ambiguous on every surface; the
        # same-bucket case would also fail the target-dir check later.
        if (held := session_id_bucket(state, new_session_id)) is not None:
            reporter.error(
                f"--session-id {new_session_id!r} already names a session under {held}/;"
                " ids are unique across every bucket. Pick another id."
            )
            raise _ForkRefused(2)
    child_id = new_session_id or unused_session_id(state, session_bucket(src_mode))
    return _ForkPlan(
        src=src,
        # A fork keeps its source's mode, so its dir belongs in that mode's
        # bucket: a forked plan in runs/ would be the one session whose
        # directory disagreed with its own manifest.
        dst=SessionLayout(state_dir=state, session_id=child_id, subdir=session_bucket(src_mode)),
        checkpoint_path=checkpoint_path,
        graph_version=checkpoint.graph_version,
        forked_from_turn=checkpoint.next_iteration,
        forked_from_sha=forked_from_sha,
        base_sha=sm.base_sha,
        base_branch=sm.base_branch,
        user_task=sm.user_task,
        mode=src_mode,
        # The child's preset as the run/resume paths stamp it: a FLAG-selected
        # source replays its flag name (replay_preset); a CONFIG-selected one
        # re-derives from the CURRENT config (cfg.preset), never the source
        # manifest's possibly-stale name.
        preset=sm.workflow.replay_preset or cfg.preset,
        preset_from_flag=sm.workflow.preset_from_flag,
        cfg=cfg,
        gate=(sm.workflow.verify_command, sm.workflow.verify_origin),
    )


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


def create_fork(
    config_path: Path | None,
    source_session_id: str,
    *,
    at_turn: int | None = None,
    new_session_id: str = "",
    cwd: Path,
    sandbox_overrides: SandboxOverrides | None = None,
    refuse_continuation: Callable[[Config, str], str | None] | None = None,
    worktree: bool = True,
    checkout: Checkout | None = None,
    checkout_untracked: frozenset[str] | None = None,
    reporter: Reporter = STDIO_REPORTER,
) -> tuple[str, int]:
    """Create a new run cloned from *source_session_id* at checkpoint *at_turn*.

    Materializes the fork on disk WITHOUT starting it: the cloned checkpoint
    + DAG, the manifest, `agent6/<child>` cut at the checkpoint's committed
    HEAD, the lineage record and, for a run with *worktree* set, a linked
    worktree detached at that sha which the manifest names and `resume` runs
    the leg in. With *worktree* off (`/undo`) the child works in *checkout*
    (a worktree with its recorded git dir, or None for the operator's): the
    checkout the undone session ran in, which its source, an ancestor up the
    lineage, need not share, with *checkout_untracked* the operator's files in
    THAT checkout (the ancestor's set describes a different one). A plan or ask
    fork never edits and reads the operator's checkout, so it gets no worktree
    either way. Returns `(child_id, 0)` on success,
    else `("", rc)` after printing the reason. The caller (`ui/cli/fork.py`)
    then either reports the created id (`--no-run`) or continues it over
    resume. *cwd* is the repository; see :func:`_plan_fork` for
    *refuse_continuation*.
    """
    try:
        plan = _plan_fork(
            config_path,
            source_session_id,
            at_turn=at_turn,
            new_session_id=new_session_id,
            cwd=cwd,
            sandbox_overrides=sandbox_overrides,
            refuse_continuation=refuse_continuation,
            reporter=reporter,
        )
    except _ForkRefused as refused:
        return "", refused.rc
    added = worktree and plan.mode == "run"
    if added and plan.cfg.git.control == "model":
        reporter.error(
            "a fork runs in a linked worktree whose .git is read-only in the jail; under"
            ' [git].control = "model" the model could not commit there. Set control ='
            ' "agent6" for the fork, or take the run back with /undo in its checkout.'
        )
        return "", 2
    if added:
        path = subordinate_workdir_root(plan.cfg, cwd, plan.dst.session_id)
        try:
            # The git dir the worktree points into, taken from the repository
            # agent6 runs in: the record a fork leg's jail grants from.
            checkout = Checkout(path, git_common_dir(cwd))
            add_worktree(cwd, path, plan.forked_from_sha)
        except GitError as exc:
            reporter.error(f"could not add the fork's worktree at {path}: {exc}")
            return "", 1
    rc = _materialize_fork(
        plan,
        cwd=cwd,
        checkout=checkout,
        fresh_checkout=added,
        checkout_untracked=checkout_untracked,
        reporter=reporter,
    )
    if rc != 0:
        if added and checkout is not None:
            remove_fork_worktree(cwd, checkout.worktree, (plan.forked_from_sha,))
        return "", rc
    return plan.dst.session_id, 0


def _materialize_fork(
    plan: _ForkPlan,
    *,
    cwd: Path,
    checkout: Checkout | None,
    fresh_checkout: bool,
    checkout_untracked: frozenset[str] | None = None,
    reporter: Reporter = STDIO_REPORTER,
) -> int:
    """Write the fork's state on disk: clone the checkpoint + DAG, the manifest
    (naming *checkout*, the worktree the fork works in and its git dir; None
    for the operator's), the git refs, and the lineage record. Returns 0 on
    success, else an error code (after printing). The source run is never
    touched."""
    src, dst = plan.src, plan.dst
    if dst.session_dir.exists():
        reporter.error(f"target run dir already exists: {dst.session_dir}")
        return 2
    dst.ensure()

    # Seed the new run's resume pointer + origin checkpoint from the chosen
    # checkpoint, then rebuild the DAG as it stood at that checkpoint.
    blob = plan.checkpoint_path.read_text(encoding="utf-8")
    atomic_write(dst.session_dir / "loop_state.json", blob)
    atomic_write(dst.checkpoint_path(0), blob)
    _copy_dag(src, dst, graph_version=plan.graph_version)
    # The files this fork's commits leave out. A FRESH worktree is a checkout
    # of the sha and nothing else, so its own untracked set is the answer (and
    # is normally empty); the source's set names paths that do not exist there.
    # An /undo fork continues in an existing checkout, so the set is the one
    # for THAT checkout, which the caller knows: chain commits never touch the
    # index, so everything the run created still reads untracked, and observing
    # there would make the fork exclude the very work it is continuing. The
    # source's own set is the fallback, right when the two share a checkout.
    if fresh_checkout and checkout is not None:
        excluded = untracked_paths(checkout.worktree)
    elif checkout_untracked is not None:
        excluded = checkout_untracked
    else:
        excluded = read_untracked_at_start(src.session_dir)
    write_untracked_at_start(dst.session_dir, excluded)

    run_branch = run_branch_for(dst.session_id) if plan.cfg.git.branch_per_run else None
    write_session_manifest(
        dst,
        session_id=dst.session_id,
        user_task=plan.user_task,
        base_sha=plan.base_sha,
        base_branch=plan.base_branch,
        run_branch=run_branch,
        cfg=plan.cfg,
        mode=plan.mode,
        effective_preset=plan.preset,
        preset_from_flag=plan.preset_from_flag,
        parent_session_id=src.session_id,
        forked_from_turn=plan.forked_from_turn,
        forked_from_sha=plan.forked_from_sha,
        gate=plan.gate,
        isolation=resolve_isolation(plan.cfg.sandbox.isolation, detect_env()),
        worktree=checkout.worktree if checkout is not None else None,
        worktree_git_dir=checkout.git_dir if checkout is not None else None,
    )

    # Seed the fork's chain at the historical sha WITHOUT touching the
    # operator's checkout: the hidden ref always, the visible branch per
    # [git].branch_per_run (both additive ref writes, never a checkout).
    try:
        # The branch is the write that can refuse (it exists at another sha, as
        # after `sessions rm`, which keeps branches), so cut it first: nothing
        # of the fork's is on disk yet to unpick.
        if run_branch is not None:
            create_branch_at(cwd, run_branch, plan.forked_from_sha)
        set_ref(cwd, chain_ref_for(dst.session_id), plan.forked_from_sha)
    except GitError as exc:
        reporter.error(f"could not cut fork refs at {plan.forked_from_sha[:12]}: {exc}")
        # A fork exists only with its refs: the run dir written above goes too.
        # Nothing else is unpicked -- a chain ref under this id is a previous
        # run's anchor (`sessions prune` keeps exactly those), not the fork's.
        shutil.rmtree(dst.session_dir, ignore_errors=True)
        return 1

    append_jsonl(
        src.state_dir / "lineage.jsonl",
        {
            "child": dst.session_id,
            "parent": src.session_id,
            "turn": plan.forked_from_turn,
            "sha": plan.forked_from_sha,
            "ts": _dt.datetime.now(tz=_dt.UTC).isoformat(timespec="microseconds"),
        },
    )
    at = f"(branch {run_branch} " if run_branch else f"({chain_ref_for(dst.session_id)} "
    where = f" in {checkout.worktree}" if checkout is not None else ""
    reporter.note(
        f"forked {src.session_id}@turn {plan.forked_from_turn} -> {dst.session_id} "
        f"{at}at {plan.forked_from_sha[:12]}){where}"
    )
    return 0
