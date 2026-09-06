# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Anything a listing shows is reachable by every command that takes an id.

Each of these sites rebuilt "id -> layout" or "newest -> layout" with `runs/`
hardcoded, so it saw only one bucket. Splitting plans/ out of runs/ turned that
latent narrowness into a plan nothing could open.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent6.paths import state_dir


def _seed(state: Path, bucket: str, session_id: str, *, mode: str, marker: str = "") -> Path:
    d = state / "sessions" / bucket / session_id
    d.mkdir(parents=True)
    (d / "logs.jsonl").write_text(
        json.dumps({"type": "session.start", "mode": mode, "user_task": "t"}) + "\n",
        encoding="utf-8",
    )
    (d / "manifest.json").write_text(
        json.dumps({"version": 1, "session_id": session_id, "mode": mode, "user_task": "t"}),
        encoding="utf-8",
    )
    if marker:
        (d / marker).mkdir()
    return d


def test_history_graph_without_an_id_finds_a_plan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The no-id form scanned runs/ and then built a runs/ layout, so the split
    made a plan's graph unopenable -- and an id from anywhere else pointed at a
    directory that does not exist."""
    from agent6.ui.cli import main

    monkeypatch.chdir(tmp_path)
    state = state_dir(tmp_path)
    _seed(state, "plans", "brave-oak-AAAAAA", mode="plan", marker="graph")

    main(["sessions", "graph"])
    err = capsys.readouterr().err
    # It RESOLVED the plan (an empty graph is a separate, honest complaint);
    # before, it could not see the bucket at all.
    assert "brave-oak-AAAAAA" in err
    assert "no sessions with a graph" not in err


def test_history_transcript_without_an_id_finds_a_plan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from agent6.ui.cli import main

    monkeypatch.chdir(tmp_path)
    state = state_dir(tmp_path)
    d = _seed(state, "plans", "brave-oak-AAAAAA", mode="plan", marker="transcripts")
    (d / "transcripts" / "0001.json").write_text(
        json.dumps({"seq": 1, "request": {}, "response": {}}), encoding="utf-8"
    )

    assert main(["sessions", "transcript"]) == 0
    assert "brave-oak-AAAAAA" in capsys.readouterr().err


def test_sessions_diff_names_the_real_problem_for_a_plan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A plan HAS no branch to diff, which is a different sentence from "no
    session matches that id". The id resolves; the answer is about branches."""
    from agent6.ui.cli import main

    monkeypatch.chdir(tmp_path)
    state = state_dir(tmp_path)
    _seed(state, "plans", "brave-oak-AAAAAA", mode="plan")
    # A populated runs/ so the resolver takes its real path rather than
    # short-circuiting on a missing bucket with a different error.
    _seed(state, "runs", "quiet-fox-BBBBBB", mode="run")

    main(["sessions", "diff", "brave-oak-AAAAAA"])
    err = capsys.readouterr().err
    assert "no session matches" not in err, err


def test_the_repl_watch_reads_its_own_log(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """`/watch` inside an ask looked under runs/ for an id that lives in asks/,
    so it always reported a missing log."""
    from agent6.ui.cli._repl import repl_show_recent_events  # pyright: ignore[reportPrivateUsage]

    state = state_dir(tmp_path)
    _seed(state, "asks", "brave-oak-AAAAAA", mode="ask")

    repl_show_recent_events(tmp_path, "brave-oak-AAAAAA", n=5)
    out = capsys.readouterr()
    assert "no logs.jsonl" not in out.err + out.out


def test_the_mcp_tools_see_every_bucket(tmp_path: Path) -> None:
    """`list_sessions` is named for what it lists. It read runs/ only, so a plan
    or an ask was invisible to an editor driving agent6 over MCP."""
    import io

    from agent6.config import Config
    from agent6.ui.mcp_server import MCPServer

    state = state_dir(tmp_path)
    _seed(state, "plans", "brave-oak-AAAAAA", mode="plan")
    _seed(state, "asks", "quiet-fox-BBBBBB", mode="ask")

    server = MCPServer(root=tmp_path, config=Config(), stdin=io.BytesIO(), stdout=io.BytesIO())
    listed = {s["session_id"] for s in server._h_list_sessions({})["sessions"]}  # pyright: ignore[reportPrivateUsage]
    assert listed == {"brave-oak-AAAAAA", "quiet-fox-BBBBBB"}
