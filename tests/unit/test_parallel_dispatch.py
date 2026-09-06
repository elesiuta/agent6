# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The Workflow-free lane bookkeeping behind `/parallel` dispatch, pinned at
the unit level now that it lives outside the loop (workflows/_parallel_dispatch)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from agent6.directive import DirectiveError, Segment
from agent6.git_ops import GitError
from agent6.workflows import _parallel_dispatch as pd
from agent6.workflows._parallel_dispatch import (
    LaneJoin,
    join_lane_result,
    lane_note,
    segment_lanes,
    segment_stamp,
    summary_text,
)
from agent6.workflows.subrun import LaneResult, LaneSpec


def _res(*, ok: bool, error: str = "", branch: str = "agent6/lane-1") -> LaneResult:
    spec = LaneSpec(lane=1, session_id="lane-1", workdir=Path("/nowhere"), model=None)
    return LaneResult(spec=spec, session_dir=Path("/nowhere"), branch=branch, ok=ok, error=error)


def test_segment_lanes_expands_counts_and_models() -> None:
    three = segment_lanes(Segment(task="t", spec="3"), limit=4)
    assert [lt.model for lt in three] == [None, None, None]
    lanes = segment_lanes(Segment(task="t", spec="m1,m2"), limit=4)
    assert [lt.model for lt in lanes] == ["m1", "m2"]
    assert all(lt.task == "t" for lt in lanes)
    with pytest.raises(DirectiveError):
        segment_lanes(Segment(task="t", spec="0"), limit=4)


def test_join_lane_result_never_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """A failed lane, a conflicted merge, and a GitError each reduce to a
    non-"joined" LaneJoin instead of aborting the run."""
    kw: dict[str, Any] = {
        "ref": "refs/agent6/run",
        "fallback_parent": None,
        "identity": None,
        "also_branch": None,
    }
    assert join_lane_result(Path("/r"), _res(ok=False, error="died"), **kw) == LaneJoin(
        "lane-1", "agent6/lane-1", "failed", "", "died"
    )

    def _conflict(*_a: Any, **_k: Any) -> str | None:
        return None

    monkeypatch.setattr(pd, "chain_merge", _conflict)
    assert join_lane_result(Path("/r"), _res(ok=True), **kw).status == "conflict"

    def _boom(*_a: Any, **_k: Any) -> str | None:
        raise GitError("fetch failed")

    monkeypatch.setattr(pd, "chain_merge", _boom)
    j = join_lane_result(Path("/r"), _res(ok=True), **kw)
    assert (j.status, j.detail) == ("failed", "fetch failed")

    def _clean(*_a: Any, **_k: Any) -> str | None:
        return "a" * 40

    monkeypatch.setattr(pd, "chain_merge", _clean)
    j = join_lane_result(Path("/r"), _res(ok=True), **kw)
    assert (j.status, j.sha) == ("joined", "a" * 40)


def _join(status: Any, session_id: str = "lane-1", sha: str = "") -> LaneJoin:
    return LaneJoin(
        session_id, f"agent6/{session_id}", status, sha, "boom" if status == "failed" else ""
    )


def test_segment_stamp_reduces_lanes() -> None:
    # Single lane keeps the old shape: passed with the join sha, or failed.
    assert segment_stamp([_join("joined", sha="abc123def4567")]) == (
        "passed",
        "lane-1 joined at abc123def456",
        "abc123def4567",
    )
    status, note, sha = segment_stamp([_join("failed")])
    assert (status, sha) == ("failed", "")
    assert "lane-1 failed: boom" in note
    # Multi-lane: any join passes, recording the LAST joined sha; the note
    # names every lane. All-conflict fails (NodeStatus has no "blocked").
    status, note, sha = segment_stamp(
        [_join("joined", "a", sha="1111111111111"), _join("joined", "b", sha="2222222222222")]
    )
    assert (status, sha) == ("passed", "2222222222222")
    assert "a joined at" in note and "b joined at" in note
    status, note, _sha = segment_stamp([_join("conflict"), _join("conflict", "b")])
    assert status == "failed"
    assert "conflicted; merge manually" in note


def test_summary_text_names_every_outcome() -> None:
    text = summary_text(
        "p1",
        [
            _join("joined", "a", sha="abc123def4567"),
            _join("conflict", "b"),
            _join("failed", "c"),
        ],
    )
    assert "group p1 complete (3 lane(s))" in text
    assert "a (agent6/a): joined at abc123def456" in text
    assert "CONFLICT" in text and "git merge agent6/b" in text
    assert "FAILED -- boom; nothing joined." in text
    assert text.endswith("Review what landed and continue.")


def test_lane_note_wordings() -> None:
    assert lane_note(_join("joined", sha="abc123def4567")) == "lane-1 joined at abc123def456"
    assert lane_note(_join("conflict")) == "lane-1 conflicted; merge manually"
    assert lane_note(_join("failed")) == "lane-1 failed: boom"


def test_segment_lanes_carry_the_operator_pins_out_of_band() -> None:
    """`/pin` tells the worker an instruction "stays binding for the rest of the
    run", and a lane's branch is merged back into the coordinator's -- so a lane
    that never saw the pin could violate a standing instruction and have that
    work land anyway. But a pin folded into the TASK became the lane's manifest
    user_task, so every listing and the judge's CandidateBrief led with
    "PINNED operator instructions (verbatim):" instead of the work. Pins ride
    the LaneTask out-of-band; the task text stays the task."""
    lanes = segment_lanes(
        Segment(spec="2", task="refactor the model layer"), ["never touch schema files"], limit=4
    )
    assert len(lanes) == 2
    for lane in lanes:
        assert lane.task == "refactor the model layer"
        assert lane.pins == ("never touch schema files",)

    # No pins: nothing rides along.
    plain = segment_lanes(Segment(spec="", task="do it"), limit=4)
    assert plain[0].task == "do it" and plain[0].pins == ()


def test_a_dirty_origin_fans_out_under_stash_and_include(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """`--parallel` read `require_clean_worktree` alone, so the documented
    stash-without-asking setup refused to fan out and named a key the run
    path ignored. One knob decides for both."""
    from agent6.config import Config
    from agent6.ui.cli import parallel as cli_parallel

    def _dirty(_origin: Path) -> list[str]:
        return ["a.py"]

    def _clear(_cfg: Config) -> None:
        return None

    monkeypatch.setattr(cli_parallel, "modified_paths", _dirty)
    monkeypatch.setattr(cli_parallel, "budget_preflight", _clear)
    monkeypatch.setattr(cli_parallel, "_parallel_approval_refusal", _clear)
    fanned: list[str] = []

    def _fake_run_parallel(task: str, *_a: object, **_kw: object) -> int:
        fanned.append(task)
        return 0

    monkeypatch.setattr(cli_parallel, "run_parallel", _fake_run_parallel)
    ask = Config.model_validate({"git": {"dirty_tree": "ask"}})
    assert cli_parallel.dispatch_parallel(ask, "t", "2", cwd=tmp_path) == 2
    assert "dirty_tree" in capsys.readouterr().err and fanned == []
    for choice in ("stash", "include"):
        cfg = Config.model_validate({"git": {"dirty_tree": choice}})
        assert cli_parallel.dispatch_parallel(cfg, "t", "2", cwd=tmp_path) == 0, choice
    assert fanned == ["t", "t"]
