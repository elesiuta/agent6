# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""`agent6 sessions show` on a fan-out coordinator lists its lanes; on a lane
it names the coordinator."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent6.paths import state_dir
from agent6.sessions.layout import SessionLayout
from agent6.ui.cli import main


def _session(repo: Path, session_id: str, manifest: dict[str, object]) -> None:
    layout = SessionLayout(state_dir=state_dir(repo), session_id=session_id)
    layout.ensure()
    layout.manifest_path.write_text(
        json.dumps({"version": 3, "session_id": session_id, "mode": "run", **manifest}),
        encoding="utf-8",
    )
    layout.logs_path.write_text(
        json.dumps({"type": "session.start", "mode": "run", "user_task": "t"})
        + "\n"
        + json.dumps({"type": "session.end", "all_passed": True, "reason": "finish_session"})
        + "\n",
        encoding="utf-8",
    )


@pytest.fixture
def fan_out(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    repo = tmp_path / "repo"
    repo.mkdir()
    monkeypatch.chdir(repo)
    _session(repo, "fan", {"fanout": {"lanes": 2, "spec": "2"}})
    for lane in (1, 2):
        _session(
            repo,
            f"fan-l{lane}",
            {
                "parallel": {"group": "fan", "lane": lane, "coordinator": "fan"},
                "compare": {"rank": 3 - lane, "of": 2, "winner": lane == 2, "ranked_by": "judge"},
            },
        )
    return repo


def test_show_on_a_coordinator_lists_its_lanes(
    fan_out: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["sessions", "show", "fan"]) == 0
    out = capsys.readouterr().out
    assert "fan-out:    2 lanes (--parallel 2)" in out
    lanes = [ln for ln in out.splitlines() if "fan-l" in ln]
    assert len(lanes) == 2 and "fan-l1" in lanes[0] and "fan-l2" in lanes[1]
    assert lanes[1].split()[:2] == ["rank", "1/2"] and "passed" in lanes[1]
    assert main(["sessions", "show", "fan", "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["fanout"] == {"lanes": 2, "spec": "2"}
    assert [ln["session_id"] for ln in data["lanes"]] == ["fan-l1", "fan-l2"]
    assert data["lanes"][1]["winner"] is True


def test_show_on_a_lane_names_its_coordinator(
    fan_out: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["sessions", "show", "fan-l1"]) == 0
    out = capsys.readouterr().out
    assert "lane of:    fan (lane 1 of group fan)" in out
    assert main(["sessions", "show", "fan-l1", "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["parallel"] == {"group": "fan", "lane": 1, "coordinator": "fan"}
    assert data["lanes"] == [] and data["fanout"] is None


def test_show_marks_a_lane_unmerged_like_the_listing(
    fan_out: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The fan-out view is where the operator picks a lane to merge: a lane
    whose branch holds commits its base lacks reads unmerged there too."""
    import subprocess

    def git(*args: str) -> str:
        return subprocess.run(
            ["git", "-C", str(fan_out), *args], check=True, capture_output=True, text=True
        ).stdout.strip()

    git("init", "-q", "-b", "main")
    (fan_out / "a.txt").write_text("a\n", encoding="utf-8")
    git("add", "a.txt")
    git("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "base")
    base = git("rev-parse", "HEAD")
    git("branch", "agent6/fan-l1", base)
    git("checkout", "-q", "agent6/fan-l1")
    (fan_out / "b.txt").write_text("b\n", encoding="utf-8")
    git("add", "b.txt")
    git("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "lane work")
    git("checkout", "-q", "main")
    layout = SessionLayout(state_dir=state_dir(fan_out), session_id="fan-l1")
    manifest = json.loads(layout.manifest_path.read_text(encoding="utf-8"))
    manifest.update({"base_sha": base, "base_branch": "main", "run_branch": "agent6/fan-l1"})
    layout.manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    assert main(["sessions", "show", "fan"]) == 0
    out = capsys.readouterr().out
    (line,) = [ln for ln in out.splitlines() if "fan-l1" in ln]
    assert "unmerged" in line
    assert main(["sessions", "show", "fan", "--json"]) == 0
    lanes = json.loads(capsys.readouterr().out)["lanes"]
    assert lanes[0]["session_id"] == "fan-l1" and lanes[0]["unmerged"] is True
