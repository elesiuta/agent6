# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Build + write the canonical manifest.json a run starts with (run/fork). The
reader and the on-disk shape (:class:`SessionManifest`) live in `sessions.manifest`."""

from __future__ import annotations

import datetime as _dt
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from agent6 import __version__
from agent6.app.reporter import Reporter
from agent6.config import Config
from agent6.events import EventSink
from agent6.portable import atomic_write
from agent6.sessions.layout import SessionLayout
from agent6.sessions.manifest import (
    MANIFEST_VERSION,
    ManifestError,
    ModelBrief,
    ModelsBrief,
    PolicyStamp,
    SessionManifest,
    WorkflowStamp,
    read_manifest,
)
from agent6.task_text import operator_task_text
from agent6.types import session_kind


def _model_brief(rm: Any) -> ModelBrief | None:
    """A `ModelBrief` for a resolved role, or None when unset."""
    if rm is None:
        return None
    return ModelBrief(provider=rm.provider, model=rm.model)


def write_manifest(path: Path, m: SessionManifest) -> None:
    """Serialize *m* to *path* (indent=2 + trailing newline), atomically.

    The one place a SessionManifest reaches disk: the initial write below and the
    stamp rewrites (merge / lineage / compare) all route through here, so the
    format lives in one spot. Durable temp+replace: the TUI hub and `sessions show`
    poll this file on live runs, and resume/fork need it after a crash.

    Refuses to rewrite a NEWER manifest (`ManifestError`): reading one is
    lenient so every run dir keeps rendering, but `extra="ignore"` drops the
    keys this binary doesn't know, so a stamp would silently downgrade the
    record it was only meant to annotate. An OLDER manifest carries nothing to
    lose, so it is upgraded to the shape actually written.
    """
    if m.version > MANIFEST_VERSION:
        raise ManifestError(
            f"refusing to rewrite {path}: it is version {m.version}, newer than this agent6 "
            f"understands (version {MANIFEST_VERSION}). Upgrade agent6 to stamp this run."
        )
    if m.version != MANIFEST_VERSION:
        m = m.model_copy(update={"version": MANIFEST_VERSION})
    atomic_write(path, m.model_dump_json(indent=2) + "\n")


def write_session_manifest(
    layout: SessionLayout,
    *,
    session_id: str,
    user_task: str,
    base_sha: str,
    base_branch: str,
    run_branch: str | None,
    cfg: Config,
    mode: str = "run",
    effective_preset: str = "",
    preset_from_flag: bool = False,
    gate: tuple[Sequence[str], str] | None = None,
    isolation: str = "",
    parent_session_id: str | None = None,
    forked_from_turn: int | None = None,
    forked_from_sha: str | None = None,
    worktree: Path | None = None,
    worktree_git_dir: Path | None = None,
) -> None:
    """Write the canonical manifest.json for a run.

    Format is JSON for the same reason logs.jsonl is JSON: trivially grep-able
    from a shell and easy to consume from any language. The on-disk shape is
    *liquid* until 1.0 - bump `SessionManifest.version` only when the new shape
    genuinely improves a downstream consumer.

    `parent_session_id` / `forked_from_turn` / `forked_from_sha` / `gate` are
    set only for a run created by `agent6 fork`; they record the lineage
    (source run + the turn forked from + the workspace sha at that turn + the
    gate the source was judged by). A non-forked run leaves them null.
    *worktree* is the fork's own checkout and *worktree_git_dir* the repository
    git dir it points into (see `SessionManifest.worktree`).
    """
    lineage = _parallel_lineage()
    # A fork passes the source's pin; a fresh run carries the configured gate
    # as such (a parked run keeps this stamp, no leg having run) until
    # `pin_gate` stamps the pair the leg resolved.
    verify_command, verify_origin = gate or (
        cfg.workflow.verify_command,
        "configured" if cfg.workflow.verify_command else "",
    )
    m = SessionManifest(
        agent6_version=__version__,
        session_id=session_id,
        # run | plan | ask. `fork` and `resume` act on session_mode(), never on
        # this string: a damaged manifest must not silently escalate a
        # read-only session to the privileged write tools.
        mode=mode,
        start_ts=_dt.datetime.now(tz=_dt.UTC).isoformat(timespec="microseconds"),
        # The display twin of the OPERATOR's words (a seed digest or skill
        # block `run --from`/`--skill` prepends is context, not the task),
        # clipped; every listing reads it. SessionSnapshot.original_task
        # carries the verbatim engine copy, and nothing here feeds the engine.
        user_task=operator_task_text(user_task)[:4000],
        base_sha=base_sha,
        base_branch=base_branch,
        run_branch=run_branch,
        git_control=cfg.git.control,
        models=ModelsBrief(
            # The role that actually drives this mode: a plan run recorded the
            # worker here and `sessions show` then named a model that never ran.
            driver=_model_brief(cfg.models.resolve(session_kind(mode).role)),
            reviewer=_model_brief(cfg.models.resolve("reviewer")),
        ),
        workflow=WorkflowStamp(
            review_trigger=cfg.review.trigger,
            revise_prompt=cfg.prompt.revise_prompt,
            # The preset the run actually used (--preset flag or top-level
            # `preset`), with how it was chosen: only a flag-selected one is
            # replayed as an override on resume (see WorkflowStamp.replay_preset).
            preset=effective_preset,
            preset_from_flag=preset_from_flag,
            verify_command=tuple(verify_command),
            verify_origin=verify_origin,
        ),
        policy=PolicyStamp(
            run_commands=cfg.sandbox.run_commands,
            # What the run RESOLVED to, not the knob: `auto` degrades, and a
            # surface printing "auto" told the operator nothing about whether
            # the run was actually confined.
            isolation=isolation or str(cfg.sandbox.isolation),
            network=str(cfg.sandbox.network),
        ),
        parent_session_id=parent_session_id,
        forked_from_turn=forked_from_turn,
        forked_from_sha=forked_from_sha,
        worktree=worktree,
        worktree_git_dir=worktree_git_dir,
        parallel_id=(lineage[0] if lineage else None),
        lane=(lineage[1] if lineage else None),
    )
    write_manifest(layout.manifest_path, m)


def _parallel_lineage() -> tuple[str, int] | None:
    """The fan-out lineage the spawner stamped into this lane's environment
    (`AGENT6_PARALLEL_LINEAGE=<fanout>:<lane>`), or None for an ordinary run.

    Read here, in the manifest's one writer, so the lane is self-describing
    from birth: the grouping survives a coordinator death instead of waiting
    on a post-import stamp only a live coordinator could write."""
    raw = os.environ.get("AGENT6_PARALLEL_LINEAGE", "")
    fanout, sep, lane = raw.rpartition(":")
    if not sep or not fanout or not lane.isdigit():
        return None
    return fanout, int(lane)


def stamp_parked(session_dir: Path, *, task: str, reason: str) -> None:
    """Record that this run was submitted and never started: the verbatim
    task (resume starts it fresh), why it waits, and no run branch (none was
    cut). The fresh start's manifest rewrite replaces all three."""
    m = read_manifest(session_dir)
    write_manifest(
        session_dir / "manifest.json",
        m.model_copy(update={"parked_task": task, "parked_reason": reason, "run_branch": None}),
    )


def stamp_preset(session_dir: Path, name: str) -> None:
    """Record the preset a resumed leg was started under with `--preset`: from
    here the run runs under it, and a later resume without a flag replays it
    (`WorkflowStamp.replay_preset`)."""
    m = read_manifest(session_dir)
    workflow = m.workflow.model_copy(update={"preset": name, "preset_from_flag": True})
    write_manifest(session_dir / "manifest.json", m.model_copy(update={"workflow": workflow}))


def stamp_verify_gate(session_dir: Path, argv: Sequence[str], origin: str) -> None:
    """Pin the gate this run is judged by, and where it came from.

    Written after resolution rather than at run start because inference runs
    later; from here on the pair is the run's, so a mid-run edit to AGENTS.md
    cannot move the gate under it, on this leg or a resumed one.
    """
    m = read_manifest(session_dir)
    workflow = m.workflow.model_copy(
        update={"verify_command": tuple(argv), "verify_origin": origin}
    )
    write_manifest(session_dir / "manifest.json", m.model_copy(update={"workflow": workflow}))


def pin_gate(
    session_dir: Path,
    argv: Sequence[str],
    origin: str,
    *,
    events: EventSink,
    reporter: Reporter,
) -> None:
    """Pin this leg's gate and KEEP it pinned when the loop adopts one mid-run.

    Every lifecycle that starts a leg calls this. Stamping and re-stamping were
    two separate concerns wired only in `run`, so a resumed leg that adopted a
    gate kept a manifest reading gateless. A failure is reported rather than
    raised (a leg is still worth running) and never swallowed: the manifest is
    what every viewer, the baseline and the next leg read the gate from.
    """

    def _stamp(gate: Sequence[str], why: str) -> None:
        try:
            stamp_verify_gate(session_dir, gate, why)
        except (ManifestError, OSError) as exc:
            reporter.note(f"could not record this run's verify gate: {exc}")

    _stamp(argv, origin)

    def _repin_adopted_gate(event: dict[str, Any]) -> None:
        if event.get("type") == "loop.verify_inferred" and event.get("adopted_at") is not None:
            command = tuple(event.get("command", ()))
            _stamp(command, "adopted" if command else "unadopted")

    # EventSink swallows a listener's exceptions so a UI consumer cannot break
    # the run; _stamp reports for itself rather than relying on that.
    events.subscribe(_repin_adopted_gate)
