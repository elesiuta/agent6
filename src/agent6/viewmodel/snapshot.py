# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The one-object wire snapshots: a session's folded state and a machine
instance's, as `agent6 attach --json` prints them and the web serves them.
One fold each, so the two never disagree."""

from __future__ import annotations

import contextlib
from pathlib import Path
from typing import Any

from agent6.git_ops import branch_exists, merge_stamp_holds
from agent6.machine import MachineJournal, load_machine
from agent6.sessions.ipc import worker_is_alive
from agent6.sessions.layout import LOGS_NAME
from agent6.sessions.manifest import ManifestError, SessionManifest, read_manifest
from agent6.viewmodel.format import format_branch, format_lineage
from agent6.viewmodel.listing import session_compare
from agent6.viewmodel.machine_state import (
    fold_machine,
    machine_spend,
    machine_state_as_dict,
)
from agent6.viewmodel.state import fold_session, fold_until_commit, session_state_as_dict
from agent6.viewmodel.tail import tail_events


def existing_run_branch(manifest: SessionManifest, repo: Path | None) -> str:
    """The run's branch while it exists, else "". The manifest names the
    branch at run start and git creates it at the first commit, so a run that
    never committed has none to merge or prune, whatever the manifest says
    (`sessions show --json` reports `run_branch` by this rule). With no *repo*
    to ask, the manifest's word stands, as `merged_into` trusts its stamp."""
    name = manifest.run_branch or ""
    if not name or (repo is not None and not branch_exists(repo, name)):
        return ""
    return name


def manifest_branches(session_dir: Path, *, repo: Path | None = None) -> dict[str, str]:
    """Branch facts from the run's manifest (run_branch / base_branch /
    merged_into, and `branch_line`, their one wording) for the run header.
    The event fold does not carry them, and an operator needs to SEE where a
    run's work lives and where Merge lands (consecutive spawns chain branches
    invisibly otherwise). Empty for a run with no manifest (or branch_per_run
    off). With *repo*, `run_branch` is the branch only while it exists
    (`existing_run_branch`) and `merged_into` is claimed only while the merge
    stamp still describes the branch (a resumed run commits past its stamp;
    the footer and `sessions show` apply the same check)."""
    try:
        manifest = read_manifest(session_dir)
    except ManifestError:
        return {}
    out: dict[str, str] = {}
    run_branch = existing_run_branch(manifest, repo)
    if run_branch:
        out["run_branch"] = run_branch
    if manifest.base_branch:
        out["base_branch"] = manifest.base_branch
    stamp = manifest.merged
    merged_into = ""
    if stamp and stamp.into:
        holds = repo is None or merge_stamp_holds(
            repo, manifest.session_id, manifest.run_branch or "", stamp.tip
        )
        merged_into = stamp.into if holds else ""
    if merged_into:
        out["merged_into"] = merged_into
    # The line names the manifest's branch when merged (the stamp describes
    # it, pruned or not, as `sessions show` prints it) or existing.
    named = (manifest.run_branch or "") if merged_into else run_branch
    line = format_branch(named, manifest.base_branch or "", merged_into)
    if line:
        out["branch_line"] = line
    return out


def manifest_header(session_dir: Path, *, repo: Path | None = None) -> dict[str, Any]:
    """Manifest-derived session-header fields the event fold does not carry:
    the branch facts, the fork lineage (`forked_from`, one wording), and the
    fan-out compare outcome (rank/winner/rationale). Merged into every session
    snapshot (one-shot and streamed) so the header a page paints from cannot
    drift. Empty for a run with no (readable) manifest."""
    header: dict[str, Any] = dict(manifest_branches(session_dir, repo=repo))
    with contextlib.suppress(ManifestError):
        m = read_manifest(session_dir)
        header["git_control"] = m.git_control
        header["base_sha"] = m.base_sha
        lineage = format_lineage(m.parent_session_id, m.forked_from_turn, m.forked_from_sha)
        if lineage:
            header["forked_from"] = lineage
        if m.worktree is not None:
            header["worktree"] = str(m.worktree)
    compare = session_compare(session_dir)
    if compare is not None:
        header["compare"] = compare.model_dump(mode="json")
    return header


class UnknownStepError(ValueError):
    """A step sha that is none of the run's commits."""


def session_snapshot(
    session_dir: Path, *, repo: Path | None = None, step: str = ""
) -> dict[str, Any]:
    """A session's folded state as the wire dict, with the dir-aware status
    (parked / stale / waiting, not the fold's blanket "running"), the
    dir-backed identity fill, and the manifest header. A session with no log
    yet (a parked submission, a `fork --no-run`) folds nothing and lets the
    dir supply the word. *step* (a commit sha of the run) folds only up to
    that commit and stamps `as_of`."""
    events = tail_events(session_dir / LOGS_NAME, follow=False)
    as_of: dict[str, Any] | None = None
    if step:
        at = fold_until_commit(events, step)
        if at is None:
            raise UnknownStepError(f"no commit {step} in this run")
        state = at
        as_of = {"iteration": at.steps[-1].iteration, "sha": at.steps[-1].sha}
    else:
        state = fold_session(events)
    snap = session_state_as_dict(state, session_dir)
    snap["as_of"] = as_of
    snap.update(manifest_header(session_dir, repo=repo))
    return snap


def machine_snapshot(machine_dir: Path) -> dict[str, Any]:
    """A machine instance's folded MachineState as the wire dict. Raises
    MachineError for an unloadable source and JournalError for a corrupt
    journal; the callers word those.

    Carries the instance's `spend`, which only `machine status` used to
    compute: a machine is the one thing that runs unattended against
    `[budget].max_usd`, and the two surfaces an operator watches it from could
    not say what it had spent."""
    spec = load_machine(machine_dir / "machine.asm.toml")
    events = MachineJournal(machine_dir).read()
    ms = fold_machine(spec, events)
    d = machine_state_as_dict(ms, machine_dir)
    spend, in_flight = machine_spend(events, machine_dir, alive=worker_is_alive(machine_dir))
    d["spend"] = {
        "usd": spend.usd,
        "usd_partial": spend.partial,
        "input_tokens": spend.input_tokens,
        "output_tokens": spend.output_tokens,
        "in_flight_state": in_flight,
    }
    return d
