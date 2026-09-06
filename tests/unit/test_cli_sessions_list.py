# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""`agent6 sessions` (list): the winner marker on fan-out compare winners."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent6.paths import state_dir
from agent6.ui.cli._common import _runs_dir, styled_status  # pyright: ignore[reportPrivateUsage]
from agent6.ui.cli.sessions_cmds import (
    _cmd_list,  # pyright: ignore[reportPrivateUsage]
    _cmd_sessions_dir,  # pyright: ignore[reportPrivateUsage]
)


def _run(runs: Path, session_id: str, *, winner: bool | None = None) -> None:
    d = runs / session_id
    d.mkdir(parents=True)
    manifest: dict[str, object] = {"mode": "run"}
    if winner is not None:
        rank = 1 if winner else 2
        manifest["compare"] = {"group": "fan", "rank": rank, "of": 2, "winner": winner}
    (d / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (d / "logs.jsonl").write_text(
        json.dumps({"type": "session.start", "mode": "run", "user_task": session_id})
        + "\n"
        + json.dumps({"type": "session.end", "all_passed": True, "reason": "finish_session"})
        + "\n",
        encoding="utf-8",
    )


def test_runs_list_marks_the_fan_out_winner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    repo = tmp_path / "repo"
    repo.mkdir()
    monkeypatch.chdir(repo)
    runs = _runs_dir(repo)
    _run(runs, "fan-l1", winner=False)
    _run(runs, "fan-l2", winner=True)
    _run(runs, "solo")  # a run outside any fan-out: no marker

    assert _cmd_list() == 0
    out = capsys.readouterr().out
    assert "fan-l2 ★" in out  # the winner id carries the ★
    assert "fan-l1 ★" not in out and "solo ★" not in out  # losers / non-lanes do not


def test_runs_list_json_carries_the_row_facts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """`sessions list --json` is the table's rows as data: one object per
    session with the listing facts, the winner as a boolean, no styling."""
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    repo = tmp_path / "repo"
    repo.mkdir()
    monkeypatch.chdir(repo)
    assert _cmd_list(as_json=True) == 0
    assert json.loads(capsys.readouterr().out) == []  # the empty listing is data too
    runs = _runs_dir(repo)
    _run(runs, "fan-l2", winner=True)
    assert _cmd_list(as_json=True) == 0
    rows = json.loads(capsys.readouterr().out)
    assert [r["session_id"] for r in rows] == ["fan-l2"]
    row = rows[0]
    assert row["winner"] is True
    assert (row["mode"], row["status"], row["task"]) == ("run", "passed", "fan-l2")
    # The one listing row shape, shared with `/api/hub` (viewmodel.summary_row).
    assert set(row) == {
        "session_id",
        "mode",
        "status",
        "reason",
        "label",
        "level",
        "unmerged",
        "verify_ok",
        "cost_usd",
        "cost",
        "id_cell",
        "usd_partial",
        "mtime",
        "winner",
        "task",
    }


def test_sessions_dir_names_a_sessions_own_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """`sessions dir <id>` prints that session's directory (an unambiguous
    prefix resolves like everywhere else); an unknown id is an error, not the
    repo root."""
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    repo = tmp_path / "repo"
    repo.mkdir()
    monkeypatch.chdir(repo)
    runs = _runs_dir(repo)
    _run(runs, "solo-ABC123")
    assert _cmd_sessions_dir("solo") == 0
    assert capsys.readouterr().out.strip() == str(runs / "solo-ABC123")
    assert _cmd_sessions_dir("nope") == 2
    assert "ERROR" in capsys.readouterr().err


def test_runs_list_marks_a_partial_cost(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A cost the scanner knows is a lower bound (unpriced model in some leg)
    renders with the '~' marker in the listing, matching `sessions show`."""
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    repo = tmp_path / "repo"
    repo.mkdir()
    monkeypatch.chdir(repo)
    runs = _runs_dir(repo)
    d = runs / "unpriced"
    d.mkdir(parents=True)
    (d / "manifest.json").write_text(json.dumps({"mode": "run"}), encoding="utf-8")
    (d / "logs.jsonl").write_text(
        json.dumps({"type": "session.start", "mode": "run", "user_task": "t"})
        + "\n"
        + json.dumps({"type": "budget.update", "usd_total": 0.0123, "usd_partial": True})
        + "\n"
        + json.dumps({"type": "session.end", "all_passed": True, "reason": "finish_session"})
        + "\n",
        encoding="utf-8",
    )
    assert _cmd_list() == 0
    assert "~$0.01" in capsys.readouterr().out


def test_styled_status_colors_stale_red_and_parked_yellow() -> None:
    """The CLI status colors mirror the TUI/web: a lost worker (stale) is red and
    a parked submission (needs a resume) is yellow, not the old dim/uncolored that
    let a dead or unstarted run read as neutral in `agent6 sessions`."""
    stale, _ = styled_status("stale", "", color=True)
    assert "\x1b[1;31m" in stale  # the error level, like failed: the run header + web pill agree
    parked, _ = styled_status("parked", "resume to start", color=True)
    assert "\x1b[33m" in parked  # yellow: attention, not a neutral done


def test_runs_list_columns_stay_aligned_with_a_machine_draft(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A `machine create` draft lists with mode `machine`, wider than the
    fixed four-column mode cell; every row's cost column must still start
    where the header's does."""
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    repo = tmp_path / "repo"
    repo.mkdir()
    monkeypatch.chdir(repo)
    _run(_runs_dir(repo), "runny-one-AAAAAA")
    draft = _runs_dir(repo).parent / "machines" / "drafty-two-BBBBBB"
    draft.mkdir(parents=True)
    (draft / "manifest.json").write_text(json.dumps({"mode": "machine"}), encoding="utf-8")
    (draft / "logs.jsonl").write_text(
        json.dumps({"type": "session.start", "mode": "machine", "user_task": "draft it"})
        + "\n"
        + json.dumps({"type": "session.end", "all_passed": True, "reason": "finish_session"})
        + "\n",
        encoding="utf-8",
    )
    assert _cmd_list() == 0
    lines = capsys.readouterr().out.splitlines()
    id_col = lines[0].index("  id  ") + 2
    assert lines[1].index("drafty-two-BBBBBB") == id_col
    assert lines[2].index("runny-one-AAAAAA") == id_col


def test_listing_status_label_folds_mode_reason_and_unmerged() -> None:
    """One cell for the three surfaces: the mode when the word does not imply
    it, the reason, the unmerged mark on ended runs only."""
    from agent6.viewmodel.format import listing_status_label

    assert listing_status_label("run", "passed") == "passed"
    assert listing_status_label("run", "passed", unmerged=True) == "passed · unmerged"
    assert listing_status_label("plan", "planned") == "planned"
    assert listing_status_label("plan", "running") == "plan · running"
    assert listing_status_label("ask", "answered") == "answered"
    assert listing_status_label("machine", "finished") == "machine · finished"
    assert (
        listing_status_label("run", "failed", "provider_error", unmerged=True)
        == "failed · provider error · unmerged"
    )
    # A live run's branch is unmerged by definition: no mark.
    assert listing_status_label("run", "running", unmerged=True) == "running"


def test_runs_list_marks_an_unmerged_run_and_drops_the_mark_after_merge(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The listing answers "does anything still need merging": a finished run
    whose branch holds commits reads `passed · unmerged`; merging (stamp tip ==
    branch tip) or a zero-commit branch (tip == base) drops the mark."""
    import subprocess

    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    repo = tmp_path / "repo"
    repo.mkdir()
    monkeypatch.chdir(repo)

    def git(*args: str) -> str:
        return subprocess.run(
            ["git", "-C", str(repo), *args], check=True, capture_output=True, text=True
        ).stdout.strip()

    git("init", "-q", "-b", "main")
    git("config", "user.email", "t@t")
    git("config", "user.name", "t")
    (repo / "a.txt").write_text("a\n", encoding="utf-8")
    git("add", "-A")
    git("commit", "-qm", "init")
    base = git("rev-parse", "HEAD")
    git("checkout", "-qb", "agent6/unmerged-run-AAAAAA")
    (repo / "b.txt").write_text("b\n", encoding="utf-8")
    git("add", "-A")
    git("commit", "-qm", "work")
    tip = git("rev-parse", "HEAD")
    git("checkout", "-q", "main")

    runs = _runs_dir(repo)
    for sid, branch, merged in (
        ("unmerged-run-AAAAAA", "agent6/unmerged-run-AAAAAA", None),
        (
            "merged-run-BBBBBB",
            "agent6/unmerged-run-AAAAAA",
            {"into": "main", "sha": tip, "tip": tip},
        ),
    ):
        d = runs / sid
        d.mkdir(parents=True)
        manifest: dict[str, object] = {
            "version": 2,
            "session_id": sid,
            "mode": "run",
            "user_task": "t",
            "base_sha": base,
            "run_branch": branch,
        }
        if merged:
            manifest["merged"] = merged
        (d / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        (d / "logs.jsonl").write_text(
            json.dumps({"type": "session.start", "mode": "run", "user_task": "t"})
            + "\n"
            + json.dumps({"type": "session.end", "reason": "finish_session", "all_passed": True})
            + "\n",
            encoding="utf-8",
        )
    assert _cmd_list() == 0
    out = capsys.readouterr().out
    unmerged_row = next(line for line in out.splitlines() if "unmerged-run-AAAAAA" in line)
    merged_row = next(line for line in out.splitlines() if "merged-run-BBBBBB" in line)
    assert "passed · unmerged" in unmerged_row
    assert "unmerged" not in merged_row.replace("unmerged-run", "")
    assert "mode" not in out.splitlines()[0]  # the column folded into status


def test_model_controlled_run_refuses_the_git_surfaces() -> None:
    """A git_control = "model" manifest turns sessions diff/merge/commits and
    fork away with one message: the record is the model's own commits."""
    from agent6.sessions.manifest import SessionManifest, model_git_refusal

    agent6_run = SessionManifest(mode="run", session_id="x1")
    assert model_git_refusal(agent6_run, "sessions") is None
    model_run = SessionManifest(mode="run", session_id="x2", git_control="model")
    msg = model_git_refusal(model_run, "sessions diff")
    assert msg is not None and "model" in msg and "x2" in msg


def test_the_json_row_carries_the_whole_task(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The table clips for width; `--json` is the surface a script reads, and a
    one-line snippet there is indistinguishable from a one-line task."""
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    repo = tmp_path / "repo"
    repo.mkdir()
    monkeypatch.chdir(repo)
    from agent6.sessions.layout import SessionLayout

    task = "fix the parser bug\nsecond line of the task\nthird line"
    layout = SessionLayout(state_dir=state_dir(repo), session_id="run-TASK11")
    layout.ensure()
    layout.logs_path.write_text(
        json.dumps({"type": "session.start", "mode": "run", "user_task": task}) + "\n",
        encoding="utf-8",
    )

    from agent6.ui.cli import main

    assert main(["sessions", "list", "--json"]) == 0

    (row,) = json.loads(capsys.readouterr().out)
    assert row["task"] == task
