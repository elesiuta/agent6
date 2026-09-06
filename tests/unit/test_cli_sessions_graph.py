# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Tests for `agent6 sessions graph` DFS tree rendering."""

from __future__ import annotations

import os
from datetime import UTC, datetime
from pathlib import Path

import pytest

from agent6.graph.models import TaskNode
from agent6.graph.storage import write_node
from agent6.paths import state_dir
from agent6.sessions.ipc import register_frontend
from agent6.sessions.layout import SessionLayout
from agent6.ui.cli import main


def _node(
    nid: str,
    *,
    parent: str | None,
    title: str,
    children: tuple[str, ...] = (),
    status: str = "pending",
    commit_sha: str = "",
) -> TaskNode:
    now = datetime(2025, 1, 1, tzinfo=UTC)
    return TaskNode.model_validate(
        {
            "id": nid,
            "parent_id": parent,
            "title": title,
            "rationale": "",
            "acceptance": "",
            "relevant_paths": (),
            "depends_on": (),
            "children": children,
            "status": status,
            "created_at": now,
            "updated_at": now,
            "created_by": "planner",
            "commit_sha": commit_sha,
        }
    )


def _seed_tree(tmp_path: Path, session_id: str) -> None:
    """Build a small tree:
    root
      step1 (passed, commit aaaaaaa...)
        sub1a
        sub1b
      step2 (failed)
    """
    layout = SessionLayout(state_dir=state_dir(tmp_path), session_id=session_id)
    layout.ensure()
    (layout.session_dir / "logs.jsonl").write_text("{}\n", encoding="utf-8")
    root_id = "0" * 25 + "R"
    s1_id = "0" * 25 + "1"
    s2_id = "0" * 25 + "2"
    s1a_id = "0" * 25 + "A"
    s1b_id = "0" * 25 + "B"
    root = _node(root_id, parent=None, title="root task", children=(s1_id, s2_id))
    s1 = _node(
        s1_id,
        parent=root_id,
        title="step 1",
        children=(s1a_id, s1b_id),
        status="passed",
        commit_sha="abcdef1234567890",
    )
    s2 = _node(s2_id, parent=root_id, title="step 2", status="failed")
    s1a = _node(s1a_id, parent=s1_id, title="sub 1a")
    s1b = _node(s1b_id, parent=s1_id, title="sub 1b")
    nodes = {n.id: n for n in (root, s1, s2, s1a, s1b)}
    for n in nodes.values():
        write_node(layout, nodes, n)


def test_history_graph_renders_dfs_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    _seed_tree(tmp_path, "test-run-AAAA11")
    rc = main(["sessions", "graph", "test-run-AAAA11"])
    out = capsys.readouterr().out
    assert rc == 0
    lines = [line for line in out.splitlines() if line and not line.startswith("Session id:")]
    # Strict DFS: root, then step1, then deep-left sub1a, then sub1b, then step2.
    # Status is a glyph, shared with the TUI tree / web task graph / runs show.
    assert lines == [
        "· root task",
        "  ✓ step 1  (abcdef1)",
        "    · sub 1a",
        "    · sub 1b",
        "  ✗ step 2",
    ]


def test_a_half_linked_task_is_still_shown(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """`add_subtask` writes the child before its parent's `children` list, and
    says so: a crash between them leaves a node with a valid parent_id that no
    parent names. The frontier, `list_tasks` and the TUI all show it; the CLI
    walked children only, so the operator's view omitted the very task the run
    was working on."""
    monkeypatch.chdir(tmp_path)
    layout = SessionLayout(state_dir=state_dir(tmp_path), session_id="half-run-AAAA11")
    layout.ensure()
    (layout.session_dir / "logs.jsonl").write_text("{}\n", encoding="utf-8")
    root_id, orphan_id = "0" * 25 + "R", "0" * 25 + "C"
    root = _node(root_id, parent=None, title="root task")  # children never updated
    orphan = _node(orphan_id, parent=root_id, title="step A")
    nodes = {n.id: n for n in (root, orphan)}
    for n in nodes.values():
        write_node(layout, nodes, n)

    assert main(["sessions", "graph", "half-run-AAAA11"]) == 0

    out = capsys.readouterr().out
    assert "step A" in out, out


def test_history_graph_uses_most_recent_when_no_arg(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    _seed_tree(tmp_path, "older-run-AAAA11")
    _seed_tree(tmp_path, "newer-run-BBBB22")
    runs = state_dir(tmp_path) / "sessions" / "runs"
    for name in ("older-run-AAAA11", "newer-run-BBBB22"):
        (runs / name / "logs.jsonl").write_text('{"type":"session.start"}\n', encoding="utf-8")
    os.utime(runs / "older-run-AAAA11" / "logs.jsonl", (100, 100))
    os.utime(runs / "newer-run-BBBB22" / "logs.jsonl", (1000, 1000))
    register_frontend(runs / "older-run-AAAA11", 12345)
    rc = main(["sessions", "graph"])
    captured = capsys.readouterr()
    assert rc == 0
    assert "· root task" in captured.out
    assert "newer-run-BBBB22" in captured.err


def test_history_graph_missing_run_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    rc = main(["sessions", "graph", "nonexistent"])
    err = capsys.readouterr().err
    assert rc == 2
    assert "no runs directory" in err or "no session matches" in err


def test_history_graph_empty_graph_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    layout = SessionLayout(state_dir=state_dir(tmp_path), session_id="empty-run-CCCC33")
    layout.ensure()
    (layout.session_dir / "logs.jsonl").write_text("{}\n", encoding="utf-8")
    rc = main(["sessions", "graph", "empty-run-CCCC33"])
    err = capsys.readouterr().err
    assert rc == 2
    assert "no persisted graph nodes" in err
