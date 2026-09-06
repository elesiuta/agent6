# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""`agent6 sessions compare`: the advisory ranked comparison across already-run
candidates, its own module so `sessions list` does not load the judge."""

from __future__ import annotations

import contextlib
from pathlib import Path

from agent6.app.compare import RankOutcome, manifest_task, print_ranked_candidates
from agent6.config.layer import load_effective
from agent6.git_ops import (
    GitError,
    branch_exists,
    diff_range,
)
from agent6.paths import state_dir
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
    error,
    plural,
)
from agent6.ui.cli._compare import rank
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


def _candidate_diff(cwd: Path, manifest: SessionManifest) -> tuple[str, bool]:
    """The diff a run introduced (base_sha..run_branch), read-only, without
    checking out the branch (unlike `_cmd_diff`, several candidates are compared
    in one call, and only one can be the current checkout). A pruned branch
    reads from the recorded merge: its merged tip while the objects exist, else
    the commit it landed as. Returns (diff, from_merge); "" when nothing records
    the change -- never blocks the comparison."""
    base_sha, run_branch = manifest.base_sha, manifest.run_branch or ""
    if not base_sha:
        return "", False
    if run_branch and branch_exists(cwd, run_branch):
        return diff_range(cwd, base_sha, run_branch), False
    merged = manifest.merged
    if merged is None:
        return "", False
    for ref in (merged.tip, merged.sha):
        if ref and set(ref) != {"0"}:
            try:
                return diff_range(cwd, base_sha, ref), True
            except GitError:
                continue
    return "", False


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
        diff, from_merge = _candidate_diff(cwd, manifest)
        if from_merge:
            notes.append(
                f"note: {layout.session_id}'s branch is pruned; its change is read from"
                " the recorded merge"
            )
        candidates.append(
            CandidateBrief(
                session_id=layout.session_id,
                task=manifest_task(layout.session_dir, fallback=layout.session_id),
                diff=diff,
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


def _recorded_outcome(
    resolved: list[tuple[SessionLayout, SessionManifest]], candidates: list[CandidateBrief]
) -> RankOutcome | None:
    """One fan-out's own stamped verdict as a `RankOutcome`, so the recorded
    order prints through the same table a fresh ranking does.

    None when any comparable candidate carries no auto-compare stamp: a
    fan-out whose auto-compare never ran has no verdict to read, and is judged.
    """
    ids = {c.session_id for c in candidates}
    stamps = {
        layout.session_id: manifest.compare
        for layout, manifest in resolved
        if layout.session_id in ids and manifest.compare is not None and manifest.compare.rank
    }
    if len(stamps) != len(ids):
        return None
    first = stamps[min(stamps, key=lambda sid: stamps[sid].rank)]
    return RankOutcome(
        ranking=tuple(sorted(stamps, key=lambda sid: stamps[sid].rank)),
        rationale=first.rationale,
        ranked_by="judge" if first.ranked_by == "judge" else "mechanical",
        judge_cost_usd=first.judge_cost_usd,
        judge_cost_partial=first.judge_cost_partial,
    )


def _cmd_compare(
    *, session_ids: tuple[str, ...], config_path: Path | None, rejudge: bool = False
) -> int:
    """Advisory ranked comparison across >=2 already-run candidates: the same
    ranked report `--parallel`'s auto-compare prints (judge via the reviewer
    model when configured, else the mechanical verify+cost ranking) -- for
    runs picked by hand, not necessarily from the same fan-out or even the
    same task (each candidate's own manifest `user_task` is its task).
    Read-only: no merges, no writes.

    A fan-out id prints the verdict that fan-out recorded, so asking twice
    costs nothing and answers what `sessions show` shows; ids named one by one
    are a comparison to make, and are judged. `rejudge` judges either way,
    which can rank differently than the stamp the listings read."""
    cwd = Path.cwd()
    by_fanout = False
    if len(session_ids) == 1:
        # One id: a fan-out's, comparing its lanes (the console prints the
        # fan-out id, `sessions show` calls it ambiguous); anything else is one
        # run, too few.
        lanes = _fanout_lanes(cwd, session_ids[0])
        by_fanout = bool(lanes)
        session_ids = lanes or session_ids
    if len(session_ids) < 2:
        error(
            "sessions compare needs 2 or more run ids, or one --parallel fan-out id"
            f" (its lanes); got {len(session_ids)}."
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
            error(f"run {layout.session_id!r} was given more than once.")
            return 2
        seen.add(layout.session_id)
        resolved.append((layout, manifest))
    cfg = load_effective(cwd, config_path).config

    candidates, notes = _screen_candidates(cwd, resolved)
    for note in notes:
        print(note)
    if not candidates:
        error("no comparable runs; every run given is still live or never finished.")
        return 2

    merged = {
        layout.session_id: manifest.merged.into
        for layout, manifest in resolved
        if manifest.merged is not None and manifest.merged.into
    }
    recorded = _recorded_outcome(resolved, candidates) if by_fanout and not rejudge else None
    if recorded is not None:
        print(f"[agent6] the recorded verdict for {plural(len(candidates), 'lane')}:")
        print_ranked_candidates(candidates, recorded, merged_into=merged)
        print("\n(recorded when the fan-out ran; `--rejudge` spends a fresh judge call)")
        return 0

    reviewer = cfg.models.resolve("reviewer")
    # `sessions compare` is advisory and stateless: it ranks + prints but never stamps
    # a manifest (only the fan-out's auto-compare does), so `ranked_by` is unused.
    outcome = rank(cfg, candidates, transcript_dir=state_dir(cwd) / "compare")
    print(f"[agent6] comparing {len(candidates)} runs:")
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
