# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The `agent6 fork` lifecycle: clone a run (rolled back to a checkpoint) into a
NEW run.

A fork copies a source run's state, as of checkpoint turn N, into a fresh run
dir with a new id and the same repo, recording lineage (parent run + the turn).
The source run is never mutated -- this is Pi-style "sessions as trees" done as
clone-to-new-session, not in-place branching. `ui/cli/fork.py` adapts argv,
calls :func:`create_fork`, then (unless `--no-run`) continues the new run from
turn N over the resume path.

A fork is the repo at the checkpoint's committed HEAD plus the conversation up
to that turn. So on a gated run (commits fire only on a green verify), an edit
made but not committed at the forked turn is ABSENT from the fork's tree even
though the copied transcript mentions it -- the same committed-history-only
posture `resume` documents, and deliberate: the alternative is snapshotting
uncommitted bytes into every checkpoint. That rollback governs TRACKED
content only: an untracked file lives outside every sha, so one created
after the fork point persists in the shared checkout, and a fork that edits
it absorbs the whole file into its first commit. The DAG is not copied but REBUILT: the
checkpoint's `graph_version` names an exact past state, and `graph.replay`
undoes every journal-recorded mutation stamped after it, so the fork's tasks,
statuses, and cursor match the turn its conversation came from.
"""

from __future__ import annotations

import datetime as _dt
import json
import shutil
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agent6.app._setup import SandboxOverrides, detect_env, session_config
from agent6.app.manifest import write_session_manifest
from agent6.app.reporter import STDIO_REPORTER, Reporter
from agent6.app.resume import resumable_bucket_dirs
from agent6.config import Config, ConfigError
from agent6.config.layer import load_effective, resolved_state_dir
from agent6.git_ops import GitError, chain_ref_for, create_branch_at, run_branch_for, set_ref
from agent6.graph.replay import graph_at_version, journal_prefix
from agent6.graph.storage import (
    append_jsonl,
    flock,
    list_checkpoint_turns,
    load_graph,
    read_cursor,
    write_cursor,
    write_dot,
    write_node,
)
from agent6.portable import atomic_write
from agent6.sandbox.detect import resolve_isolation
from agent6.sessions.id import (
    SessionIdError,
    resolve_session,
    session_id_bucket,
    unused_session_id,
    validate_explicit_session_id,
)
from agent6.sessions.layout import (
    SessionLayout,
    read_untracked_at_start,
    write_untracked_at_start,
)
from agent6.sessions.manifest import ManifestError, model_git_refusal, read_manifest
from agent6.types import session_bucket
from agent6.viewmodel import newest_session_dir
from agent6.workflows._session_state import load_session_snapshot

# Curator-owned DAG artifacts copied verbatim into the fork; each is a
# top-level entry under the run dir (`graph/` is a directory).
_DAG_ARTIFACTS: tuple[str, ...] = ("graph", "graph.jsonl", "graph.dot", "cursor.json")


def _lineage_entry(*, child: str, parent: str, turn: int, sha: str, ts: str) -> dict[str, object]:
    """One per-repo lineage event. Pure: the caller passes the timestamp in."""
    return {"child": child, "parent": parent, "turn": turn, "sha": sha, "ts": ts}


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
    write_dot(dst, replayed.nodes)
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
class UndoTarget:
    """Where `/undo` forks a session: *source*'s checkpoint at *at_turn*, with
    the message it takes back (composer-refill text)."""

    source_session_id: str
    at_turn: int
    undone_text: str


def _ops_at(layout: SessionLayout, turn: int) -> int | None:
    """Operator-message count in the checkpoint at *turn*, None if unreadable."""
    try:
        snap = load_session_snapshot(layout.checkpoint_path(turn))
    except (OSError, ValueError):
        return None
    return len(_operator_messages(snap.messages))


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
        try:
            task = read_manifest(src.session_dir).user_task
        except ManifestError:
            task = ops[0] if ops else ""
        return UndoTarget(src.session_id, turns[0], task)
    target = _newest_checkpoint_below(src, len(ops))
    if target is None:
        reporter.err(f"nothing to undo: no state before the last message of {src.session_id}.")
        return None
    return UndoTarget(target[0], target[1], ops[-1])


def _newest_checkpoint_below(
    layout: SessionLayout, current_ops: int, *, seen: frozenset[str] = frozenset()
) -> tuple[str, int] | None:
    """The newest checkpoint of *layout* -- or, following fork lineage, of an
    ancestor -- whose conversation holds fewer operator messages than
    *current_ops*. A fork carries one seed checkpoint, so walking back past it
    means resolving in the parent it was cut from.

    *seen* stops a cyclic lineage: forks always point at an OLDER run, so a
    cycle only exists in a corrupt or hand-edited manifest, and following one
    would recurse until the stack blows. A revisited id ends the walk (no
    resolvable ancestor) instead of crashing."""
    for turn in sorted(list_checkpoint_turns(layout), reverse=True):
        ops = _ops_at(layout, turn)
        if ops is not None and ops < current_ops:
            return layout.session_id, turn
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


def undo_fork(
    config_path: Path | None,
    session_id: str,
    *,
    cwd: Path,
    reporter: Reporter = STDIO_REPORTER,
) -> tuple[str, str] | None:
    """`/undo`: fork *session_id* at its undo target, unstarted. Returns
    `(child_id, undone_text)` -- the text goes back in the composer to edit
    and resend -- or None with the reason already printed."""
    state_dir = resolved_state_dir(cwd)
    target = undo_target(state_dir, session_id, reporter=reporter)
    if target is None:
        return None
    child, rc = create_fork(
        config_path,
        target.source_session_id,
        at_turn=target.at_turn,
        cwd=cwd,
        reporter=reporter,
    )
    if rc != 0:
        return None
    return child, target.undone_text


def create_fork(  # noqa: PLR0911, PLR0912
    config_path: Path | None,
    source_session_id: str,
    *,
    at_turn: int | None = None,
    new_session_id: str = "",
    cwd: Path,
    sandbox_overrides: SandboxOverrides | None = None,
    refuse_continuation: Callable[[Config, str], str | None] | None = None,
    reporter: Reporter = STDIO_REPORTER,
) -> tuple[str, int]:
    """Create a new run cloned from *source_session_id* at checkpoint *at_turn*.

    Materializes the fork on disk (clone the checkpoint + DAG, write the
    manifest, cut `agent6/<child>` at the checkpoint's committed HEAD, record
    lineage) WITHOUT starting it. Returns `(child_id, 0)` on success, else
    `("", rc)` after printing the reason. The caller (`ui/cli/fork.py`) then
    either reports the created id (`--no-run`) or continues it over resume.

    The child's config is built as its continuation builds it (the source's
    preset, the mode clamp, this invocation's *sandbox_overrides*), so the
    manifest stamps the policy the fork runs under. *refuse_continuation*,
    given that config and the mode, returns why the continuation would refuse
    (`headless_approval_refusal` for `agent6 fork` without `--no-run`) or None;
    a reason refuses BEFORE anything is created, the order `run` keeps, so no
    never-started fork stays listed and its id stays free.
    """
    state_dir = resolved_state_dir(cwd)
    src = _resolve_source(state_dir, source_session_id, reporter=reporter)
    if src is None:
        return "", 2

    checkpoint_path = _select_checkpoint_path(src, at_turn, reporter=reporter)
    if checkpoint_path is None:
        return "", 2

    try:
        checkpoint = load_session_snapshot(checkpoint_path)
    except (OSError, ValueError) as exc:
        reporter.error(f"failed to load checkpoint {checkpoint_path}: {exc}")
        return "", 1

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
        return "", 2
    refusal = model_git_refusal(sm, "fork")
    if refusal is not None:
        # /undo forks in-process, so this one guard covers it too: without an
        # agent6 chain there is no checkpoint to fork or rewind to.
        reporter.error(refusal)
        return "", 2
    src_base_sha = sm.base_sha
    src_base_branch = sm.base_branch
    src_user_task = sm.user_task
    src_preset_from_flag = sm.workflow.preset_from_flag

    forked_from_sha = checkpoint.head_sha
    if not forked_from_sha:
        reporter.error(
            "the chosen checkpoint records no head_sha, so the fork branch "
            "cannot be cut. (A checkpoint from before per-turn sha capture.)"
        )
        return "", 1

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
        return "", 2
    if refuse_continuation is not None:
        refusal = refuse_continuation(cfg, src_mode)
        if refusal is not None:
            reporter.refuse(refusal)
            return "", 2

    # Stamp the child's preset like the run/resume paths (`preset or cfg.preset`):
    # a FLAG-selected source replays its flag name (replay_preset), a CONFIG-
    # selected one re-derives from the CURRENT config (cfg.preset) rather than the
    # source manifest's possibly-stale name -- the fork sibling of the parked-resume
    # stamp fix. (bool(replay_preset) == preset_from_flag, so the flag bit stands.)
    forked_preset = sm.workflow.replay_preset or cfg.preset

    if new_session_id:
        try:
            validate_explicit_session_id(new_session_id)
        except SessionIdError as exc:
            reporter.error(str(exc))
            return "", 2
        # Any bucket holding it makes the id ambiguous on every surface; the
        # same-bucket case would also fail the target-dir check later.
        if (held := session_id_bucket(state_dir, new_session_id)) is not None:
            reporter.error(
                f"--session-id {new_session_id!r} already names a session under {held}/;"
                " ids are unique across every bucket. Pick another id."
            )
            return "", 2
    child_id = new_session_id or unused_session_id(state_dir, session_bucket(src_mode))
    rc = _materialize_fork(
        cwd=cwd,
        src=src,
        # A fork keeps its source's mode, so its dir belongs in that mode's
        # bucket: a forked plan in runs/ would be the one session whose
        # directory disagreed with its own manifest.
        dst=SessionLayout(
            state_dir=state_dir, session_id=child_id, subdir=session_bucket(src_mode)
        ),
        checkpoint_path=checkpoint_path,
        graph_version=checkpoint.graph_version,
        forked_from_turn=checkpoint.next_iteration,
        forked_from_sha=forked_from_sha,
        base_sha=src_base_sha,
        base_branch=src_base_branch,
        user_task=src_user_task,
        mode=src_mode,
        preset=forked_preset,
        preset_from_flag=src_preset_from_flag,
        cfg=cfg,
        gate=(sm.workflow.verify_command, sm.workflow.verify_origin),
        reporter=reporter,
    )
    if rc != 0:
        return "", rc
    return child_id, 0


def _materialize_fork(
    *,
    cwd: Path,
    src: SessionLayout,
    dst: SessionLayout,
    checkpoint_path: Path,
    graph_version: int,
    forked_from_turn: int,
    forked_from_sha: str,
    base_sha: str,
    base_branch: str,
    user_task: str,
    mode: str,
    preset: str,
    preset_from_flag: bool,
    cfg: Config,
    gate: tuple[Sequence[str], str],
    reporter: Reporter = STDIO_REPORTER,
) -> int:
    """Write the fork's state on disk: clone the checkpoint + DAG, the manifest,
    the git branch, and the lineage record. Returns 0 on success, else an error
    code (after printing). The source run is never touched.

    *gate* is the source's pinned verify command and its origin. A fork inherits
    it: derived from the current config instead, a source whose gate was
    inferred or adopted forked to a run the manifest called gateless."""
    if dst.session_dir.exists():
        reporter.error(f"target run dir already exists: {dst.session_dir}")
        return 2
    dst.ensure()

    # Seed the new run's resume pointer + origin checkpoint from the chosen
    # checkpoint, then rebuild the DAG as it stood at that checkpoint.
    blob = checkpoint_path.read_text(encoding="utf-8")
    atomic_write(dst.session_dir / "loop_state.json", blob)
    atomic_write(dst.checkpoint_path(0), blob)
    _copy_dag(src, dst, graph_version=graph_version)
    # Same checkout, same operator files: the fork leaves out of its commits
    # what the source did.
    write_untracked_at_start(dst.session_dir, read_untracked_at_start(src.session_dir))

    run_branch = run_branch_for(dst.session_id) if cfg.git.branch_per_run else None
    write_session_manifest(
        dst,
        session_id=dst.session_id,
        user_task=user_task,
        base_sha=base_sha,
        base_branch=base_branch,
        run_branch=run_branch,
        cfg=cfg,
        mode=mode,
        effective_preset=preset,
        preset_from_flag=preset_from_flag,
        parent_session_id=src.session_id,
        forked_from_turn=forked_from_turn,
        forked_from_sha=forked_from_sha,
        gate=gate,
        isolation=resolve_isolation(cfg.sandbox.isolation, detect_env()),
    )

    # Seed the fork's chain at the historical sha WITHOUT touching the
    # operator's checkout: the hidden ref always, the visible branch per
    # [git].branch_per_run (both additive ref writes, never a checkout).
    try:
        set_ref(cwd, chain_ref_for(dst.session_id), forked_from_sha)
        if run_branch is not None:
            create_branch_at(cwd, run_branch, forked_from_sha)
    except GitError as exc:
        reporter.error(f"could not cut fork refs at {forked_from_sha[:12]}: {exc}")
        # The fork dir was just materialized; don't leave an orphan run dir +
        # manifest (and a lineage gap) when the refs couldn't be cut.
        shutil.rmtree(dst.session_dir, ignore_errors=True)
        return 1

    # Append the per-repo lineage event (ts minted here, passed into the pure helper).
    append_jsonl(
        src.state_dir / "lineage.jsonl",
        _lineage_entry(
            child=dst.session_id,
            parent=src.session_id,
            turn=forked_from_turn,
            sha=forked_from_sha,
            ts=_dt.datetime.now(tz=_dt.UTC).isoformat(timespec="microseconds"),
        ),
    )
    at = f"(branch {run_branch} " if run_branch else f"({chain_ref_for(dst.session_id)} "
    reporter.note(
        f"forked {src.session_id}@turn {forked_from_turn} -> {dst.session_id} "
        f"{at}at {forked_from_sha[:12]})"
    )
    return 0
