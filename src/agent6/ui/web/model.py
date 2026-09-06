# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Pure JSON payload builders for the web UI.

The web server is a thin renderer: every payload it serves is built here from the
shared read-side (viewmodel folds, config_layer, transcript_render, the machine
spec/journal). Pure functions, no HTTP or threads, so the run/machine snapshots
are exactly `session_state_as_dict` / `machine_state_as_dict` (identical to
`agent6 attach --json`).
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from agent6.app.parallel import subordinate_workdir_root
from agent6.config import ConfigError
from agent6.config.layer import available_preset_names, load_effective, resolved_state_dir
from agent6.git_ops import EMPTY_TREE, commit_diff, diff_range, run_branch_tips
from agent6.models.choices import config_value_choices
from agent6.models.registry import resolved_adaptive_values
from agent6.sessions.ipc import worker_is_alive
from agent6.sessions.layout import (
    HUB_BUCKETS,
    LOGS_NAME,
    bucket_dir,
    is_safe_session_id,
    machines_root,
)
from agent6.sessions.manifest import ManifestError, read_manifest
from agent6.viewmodel import (
    MachineSummary,
    fold_session,
    fold_transcript,
    is_session_husk,
    is_winner,
    machine_files,
    machine_instance_dirs,
    newest_state_log,
    operator_inputs,
    restate,
    session_dirs,
    session_state_as_dict,
    summarize_machine_dir,
    summarize_session_dir,
    summary_row,
    tail_events,
)
from agent6.viewmodel.config_view import render_show
from agent6.viewmodel.format import status_label, status_level
from agent6.viewmodel.transcript_style import item_lines


def session_dir_for(cwd: Path, session_id: str) -> Path | None:
    """Locate a session dir by exact id across the hub buckets (no prefix match: the
    web client always sends the full id from the hub payload). Rejects a session_id
    that is not a single safe path component. Husks are skipped so an orphaned
    dir in runs/ cannot shadow a real ask of the same id. An id in TWO buckets
    (state from before ids were one namespace) is ambiguous, so it resolves to
    None rather than silently showing one of two sessions; the CLI resolver
    names the ambiguity."""
    if not is_safe_session_id(session_id):
        return None
    found: Path | None = None
    for sub in HUB_BUCKETS:
        d = bucket_dir(resolved_state_dir(cwd), sub) / session_id
        if d.is_dir() and not is_session_husk(d):
            if found is not None:
                return None
            found = d
    return found


def machine_dir_for(cwd: Path, name: str) -> Path | None:
    if not is_safe_session_id(name):
        return None
    d = machines_root(resolved_state_dir(cwd)) / name
    return d if d.is_dir() else None


def draft_dir_for(cwd: Path, name: str) -> Path | None:
    """A `machine create` draft dir by name. Its logs.jsonl is a run-style log of
    the authoring agent, so it is watched through the run endpoints."""
    if not is_safe_session_id(name):
        return None
    d = bucket_dir(resolved_state_dir(cwd), "machines") / name
    return d if d.is_dir() else None


def draft_workspace(cwd: Path, name: str, config_path: Path | None) -> Path | None:
    """Where a `machine create` draft's commits live: its drafting workspace, a
    repo of its own beside the other subordinate working trees. None once it
    is gone (a published draft's workspace is removed with it)."""
    try:
        cfg = load_effective(cwd, config_path).config
    except ConfigError:
        return None
    workspace = subordinate_workdir_root(cfg, cwd, name)
    return workspace if (workspace / ".git").exists() else None


def draft_step_diff_payload(
    workspace: Path, sha: str, *, cumulative: bool
) -> tuple[dict[str, Any] | None, str]:
    """The patch one step of a draft introduced, or the whole bundle as of that
    step (from the empty tree: a draft starts from nothing) when *cumulative*.
    (payload, "") or (None, why)."""
    if not re.fullmatch(r"[0-9a-f]{7,40}", sha):
        return None, f"not a commit sha: {sha!r}"
    patch = diff_range(workspace, EMPTY_TREE, sha) if cumulative else commit_diff(workspace, sha)
    if not patch:
        return None, f"no diff for {sha[:12]} (not a commit of this draft)"
    return {"sha": sha, "cumulative": cumulative, "patch": patch}, ""


def draft_dir_paths(cwd: Path) -> list[Path]:
    """Every machine-create draft directory (where `machine create` writes)."""
    d = bucket_dir(resolved_state_dir(cwd), "machines")
    return [p for p in d.iterdir() if p.is_dir()] if d.is_dir() else []


# --- hub listing -------------------------------------------------------------


def _session_summary(session_dir: Path, branch_tips: Mapping[str, str]) -> dict[str, Any]:
    """The hub's one-line run summary: the shared listing row (`summary_row`),
    task clipped for the card. One shape across `/api/hub`, `sessions list
    --json` and the TUI hub, so a provider_error death or an unmerged pass
    reads the same everywhere and the client keeps no copy of the status maps."""
    return summary_row(
        summarize_session_dir(session_dir, branch_tips=branch_tips),
        winner=is_winner(session_dir),  # fan-out compare winner: a ★ on the hub row
        task_chars=100,
    )


def _list_sessions(cwd: Path) -> list[dict[str, Any]]:
    """Every session a hub lists, summarized, newest first (`session_dirs`)."""
    tips = run_branch_tips(cwd)
    return [_session_summary(p, tips) for p in session_dirs(resolved_state_dir(cwd))]


def _machine_row(s: MachineSummary) -> dict[str, Any]:
    """One machine-instance row for the hub, from the shared fold."""
    entry: dict[str, Any] = {
        "name": s.name,
        "mtime": s.mtime,
        "status": s.status,
        "level": status_level(s.status),
    }
    if s.status != "unreadable":
        entry["machine"] = s.machine
        entry["current"] = s.current
    if s.reason:
        # The shared cell, like every other surface: `reason` is also set for a
        # LIVE machine blocked on an operator prompt, and hardcoding "failed"
        # told the operator it had died instead of sending them to answer it.
        entry["label"] = status_label(s.status, s.reason)
    return entry


def _list_machines(cwd: Path) -> list[dict[str, Any]]:
    """Machine instances, newest first (`viewmodel.machine_instance_dirs`), each
    a watchable run of an authored machine, summarized by the shared fold."""
    dirs = machine_instance_dirs(resolved_state_dir(cwd))
    return [_machine_row(summarize_machine_dir(d)) for d in dirs]


def _list_drafts(cwd: Path) -> list[dict[str, Any]]:
    """`machine create` drafts summarized like runs (their logs.jsonl is a
    run-style authoring log), newest first, so the machines page can link to
    the #/draft/<name> view."""
    summaries = [_session_summary(p, {}) for p in draft_dir_paths(cwd)]
    summaries.sort(key=lambda s: s["mtime"], reverse=True)
    return summaries


def list_machine_files(cwd: Path) -> list[dict[str, str]]:
    """The hub's machine-file rows (`viewmodel.machine_files`)."""
    return [{"path": str(p), "name": p.name} for p in machine_files(cwd)]


def hub_payload(cwd: Path, config_path: Path | None = None) -> dict[str, Any]:
    """The hub: every run, machine instance, and machine-create draft, plus the
    authored machine files (to run or create from), summarized for the listing,
    and the presets the new-work composer offers (the same list `--preset`
    resolves against)."""
    return {
        "sessions": _list_sessions(cwd),
        "machines": _list_machines(cwd),
        "machine_files": list_machine_files(cwd),
        "drafts": _list_drafts(cwd),
        "presets": available_preset_names(cwd, config_path),
    }


# --- run snapshot + conversation ----------------------------------------------


def conversation_items(
    events: list[dict[str, Any]], *, worker_dead: bool = False
) -> list[dict[str, Any]]:
    """The events folded into rendered conversation items, one entry per
    `TranscriptItem`: its `kind`, the collapsed `lines` (lists of
    `[text, style]` spans from the shared `item_lines` renderer, the same
    fold the CLI stream and the TUI conversation view draw), and `full` (the
    expanded rendering) only when it differs, so the page can offer per-item
    expansion without re-implementing any clipping client-side. A dead
    worker's calls still open settle as never returned (the fold's rule,
    applied here because only a dir reader can probe the worker)."""
    out: list[dict[str, Any]] = []
    for item in fold_transcript(events, worker_dead=worker_dead):
        collapsed = item_lines(item, detail="collapsed")
        expanded = item_lines(item, detail="expanded")
        entry: dict[str, Any] = {"kind": item.kind, "lines": collapsed}
        if expanded != collapsed:
            entry["full"] = expanded
        out.append(entry)
    return out


def conversation_payload(session_dir: Path) -> dict[str, Any]:
    """A run's conversation, folded from its event log, plus the operator's
    own past inputs (the task, then every steer) for the composer's Ctrl-R
    history search. One read serves both keys."""
    events = list(tail_events(session_dir / LOGS_NAME, follow=False))
    return {
        "session_id": session_dir.name,
        "items": conversation_items(events, worker_dead=not worker_is_alive(session_dir)),
        "operator_inputs": operator_inputs(events),
    }


def restate_payload(session_dir: Path) -> dict[str, Any]:
    """`/restate` for the web composer: the same fold-side renderer the CLI
    pause menu prints, over the session's whole journal."""
    events = list(tail_events(session_dir / LOGS_NAME, follow=False))
    return {"text": restate(events, worker_dead=not worker_is_alive(session_dir))}


def machine_conversation_payload(machine_dir: Path) -> dict[str, Any]:
    """The conversation of the machine's most recent agent-state execution
    (empty when no agent state has produced a log yet), plus the per-state dir
    it came from so a client can tell when the machine advanced."""
    log = newest_state_log(machine_dir)
    if log is None:
        return {"state_dir": "", "items": []}
    events = list(tail_events(log, follow=False))
    # The machine's worker (one pid for every state) is the one to probe.
    items = conversation_items(events, worker_dead=not worker_is_alive(machine_dir))
    return {"state_dir": log.parent.name, "items": items}


# --- machine snapshot (structure + watch + reasoning) -----------------------


def machine_reasoning_snapshot(machine_dir: Path) -> dict[str, Any]:
    """The SessionState of the machine's most recent agent-state execution: the live
    reasoning + tool calls inside the state the machine is running. Empty when no
    agent state has produced a log yet.

    Carries `state_dir` (the per-state dir name, e.g. `0001-work`) so a
    client echoes it back when answering a prompt: prompt ids reset per state
    (`approval-1` in every state), so routing an answer to whichever state is
    newest AT POST TIME would misdeliver it if the machine advanced meanwhile.

    Also carries `last_event_ep`, the epoch of the newest folded event, which
    is what the stream turns into the age the client's "working… Ns" timer
    anchors to. The EPOCH rides in the payload rather than the age because the
    machine stream only sends a frame when the payload changes: an age would
    differ on every poll and send one every time, while the epoch moves only
    when something actually happened.
    """
    log = newest_state_log(machine_dir)
    if log is None:
        return {}
    state = fold_session(tail_events(log, follow=False))
    snap = session_state_as_dict(state)
    snap["state_dir"] = log.parent.name
    if state.last_event_ep is not None:
        snap["last_event_ep"] = state.last_event_ep
    return snap


# --- config ------------------------------------------------------------------


def config_payload(cwd: Path, config_path: Path | None = None) -> dict[str, Any]:
    """The effective config as a per-leaf view (value/effective/default/source/
    modified/adaptive/type/choices), keyed by dotted key. The same structure
    `agent6 config show --json` prints, its adaptive leaves (the compaction
    thresholds, `prompt.decompose = auto`) resolved from the worker model the
    same way; never includes secrets."""
    eff = load_effective(cwd, config_path)
    resolved = resolved_adaptive_values(eff.config)
    return json.loads(render_show(eff, as_json=True, resolved=resolved))


def config_suggestions(cwd: Path, key: str, config_path: Path | None = None) -> list[str]:
    """Value suggestions for one open-text config leaf: `preset` offers the
    preset names, everything else what `models.choices.config_value_choices`
    offers (a role's provider's model ids; the `/parallel` autocomplete's model
    ids under the pseudo-key `parallel.models`). Enum leaves already carry
    their choices in the config payload; any error suggests nothing, since
    suggestions are best-effort, never a failure."""
    if key == "preset":
        return available_preset_names(cwd, config_path)
    try:
        eff = load_effective(cwd, config_path)
    except ConfigError:
        return []
    return config_value_choices(eff, key)


def step_diff_payload(
    repo: Path, session_dir: Path, sha: str, *, cumulative: bool
) -> tuple[dict[str, Any] | None, str]:
    """The patch one step of the run introduced (`sha^..sha`), or the whole
    chain up to it (`base..sha`) when *cumulative*. (payload, "") or (None,
    why): a model-controlled run has no chain to select from."""
    try:
        m = read_manifest(session_dir)
    except ManifestError as exc:
        return None, f"unreadable manifest: {exc}"
    if m.git_control == "model":
        return None, "the model owns git in this run: no step chain to select from"
    if not re.fullmatch(r"[0-9a-f]{7,40}", sha):
        return None, f"not a commit sha: {sha!r}"
    patch = (
        diff_range(repo, m.base_sha, sha) if cumulative and m.base_sha else commit_diff(repo, sha)
    )
    if not patch:
        return None, f"no diff for {sha[:12]} (pruned, or not a commit of this run)"
    return {"sha": sha, "cumulative": cumulative, "patch": patch}, ""
