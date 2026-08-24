# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""`agent6 sessions compare`: the advisory ranked comparison across already-run
candidates, its own module so `sessions list` does not load the judge."""

from __future__ import annotations

import contextlib
import sys
from pathlib import Path

from agent6.config.layer import load_effective, resolved_state_dir
from agent6.git_ops import (
    branch_exists,
    diff_range,
)
from agent6.sessions.layout import (
    SessionLayout,
)
from agent6.sessions.manifest import (
    ManifestError,
    SessionManifest,
    read_manifest,
)
from agent6.ui.cli._common import (
    _runs_dir,
)
from agent6.ui.cli._compare import manifest_task, print_ranked_candidates, rank
from agent6.ui.cli.sessions_cmds import _resolve_session_manifest
from agent6.viewmodel import (
    LIVE_STATUS_WORDS,
    died_without_end,
    summarize_session_dir,
)
from agent6.viewmodel.format import (
    WINNER_GLYPH,
)
from agent6.workflows.judge import CandidateBrief


def _candidate_diff(cwd: Path, base_sha: str, run_branch: str) -> str:
    """The diff a run's branch introduced (base_sha..run_branch), read-only,
    without checking out the branch (unlike `_cmd_diff`, several candidates are
    compared in one call, and only one can be the current checkout). "" if the
    branch is gone -- never blocks the comparison."""
    if not base_sha or not run_branch or not branch_exists(cwd, run_branch):
        return ""
    return diff_range(cwd, base_sha, run_branch)


def _screen_candidates(
    cwd: Path, resolved: list[tuple[SessionLayout, SessionManifest]]
) -> tuple[list[CandidateBrief], list[str]]:
    """Briefs for the comparable runs, plus printed notes naming each excluded
    one. A run without a session.end -- died, or simply not there YET -- has no
    verdict to compare and a truncated (lowest) spend, so ranking floated it to
    first place and offered a merge (for a live run, of a branch still moving).
    The fan-out excludes such lanes; say which run was dropped rather than
    silently shrinking the table."""
    candidates: list[CandidateBrief] = []
    notes: list[str] = []
    for layout, manifest in resolved:
        summary = summarize_session_dir(layout.session_dir)
        if summary.status in LIVE_STATUS_WORDS:
            notes.append(
                f"note: {layout.session_id} is still {summary.status};"
                " excluded (stop it or let it finish first)"
            )
            continue
        if died_without_end(summary.status):
            notes.append(
                f"note: {layout.session_id} never finished ({summary.status});"
                " excluded from the ranking"
            )
            continue
        candidates.append(
            CandidateBrief(
                session_id=layout.session_id,
                task=manifest_task(layout.session_dir, fallback=layout.session_id),
                diff=_candidate_diff(cwd, manifest.base_sha, manifest.run_branch or ""),
                verify_ok=summary.verify_ok,
                cost_usd=summary.cost_usd,
            )
        )
    return candidates, notes


def _fanout_lanes(cwd: Path, parallel_id: str) -> tuple[str, ...]:
    """The lane ids of the fan-out *parallel_id* (each lane's manifest names
    it), in lane order; empty when no run does."""
    lanes: list[tuple[int, str]] = []
    runs = _runs_dir(cwd)
    if runs.is_dir():
        for d in runs.iterdir():
            with contextlib.suppress(ManifestError):
                m = read_manifest(d)
                if m.parallel_id == parallel_id:
                    lanes.append((m.lane or 0, d.name))
    return tuple(name for _, name in sorted(lanes))


def _cmd_compare(*, session_ids: tuple[str, ...], config_path: Path | None) -> int:
    """Advisory ranked comparison across >=2 already-run candidates: the same
    ranked report `--parallel`'s auto-compare prints (judge via the reviewer
    model when configured, else the mechanical verify+cost ranking) -- for
    runs picked by hand, not necessarily from the same fan-out or even the
    same task (each candidate's own manifest `user_task` is its task).
    Read-only: no merges, no writes."""
    cwd = Path.cwd()
    if len(session_ids) == 1:
        # One id: a fan-out's, comparing its lanes (the console prints the
        # fan-out id, `sessions show` calls it ambiguous); anything else is one
        # run, too few.
        session_ids = _fanout_lanes(cwd, session_ids[0]) or session_ids
    if len(session_ids) < 2:
        print(
            "ERROR: sessions compare needs 2 or more run ids, or one --parallel fan-out id"
            f" (its lanes); got {len(session_ids)}.",
            file=sys.stderr,
        )
        return 2
    resolved: list[tuple[SessionLayout, SessionManifest]] = []
    seen: set[str] = set()
    for query in session_ids:
        res = _resolve_session_manifest(cwd, query)
        if isinstance(res, int):
            return res
        layout, manifest = res
        if layout.session_id in seen:
            print(f"ERROR: run {layout.session_id!r} was given more than once.", file=sys.stderr)
            return 2
        seen.add(layout.session_id)
        resolved.append((layout, manifest))
    cfg = load_effective(cwd, config_path).config

    candidates, notes = _screen_candidates(cwd, resolved)
    for note in notes:
        print(note)
    if not candidates:
        print(
            "ERROR: no comparable runs; every run given is still live or never finished.",
            file=sys.stderr,
        )
        return 2

    reviewer = cfg.models.resolve("reviewer")
    # `sessions compare` is advisory and stateless: it ranks + prints but never stamps
    # a manifest (only the fan-out's auto-compare does), so `ranked_by` is unused.
    outcome = rank(cfg, candidates, transcript_dir=resolved_state_dir(cwd) / "compare")
    print(f"[agent6] comparing {len(candidates)} runs:")
    merged = {
        layout.session_id: manifest.merged.into
        for layout, manifest in resolved
        if manifest.merged is not None and manifest.merged.into
    }
    print_ranked_candidates(candidates, outcome, merged_into=merged)
    # A fresh judgment can contradict the fan-out's recorded verdict (the star in
    # listings comes from the auto-compare stamp, which this command never
    # rewrites); when re-judging one fan-out's own lanes, disclose the clash
    # rather than let the two surfaces silently disagree.
    groups = {manifest.parallel_id for _, manifest in resolved}
    if outcome.ranking and len(groups) == 1 and None not in groups:
        stamped = next(
            (
                layout.session_id
                for layout, manifest in resolved
                if manifest.compare is not None and manifest.compare.winner
            ),
            None,
        )
        if stamped is not None and stamped != outcome.ranking[0]:
            print(
                f"\nnote: the recorded fan-out verdict picked {stamped}"
                f" (the {WINNER_GLYPH} in listings); this fresh ranking is advisory"
                " and nothing was re-stamped."
            )
    if reviewer is None:
        print(
            "\n(no reviewer model configured; ranked mechanically: verify-pass first, then"
            " lower cost)"
        )
    return 0
