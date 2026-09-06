# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Storage + frontmatter round-trip tests."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from agent6.graph.models import TaskNode
from agent6.graph.storage import (
    SessionLayout,
    _dump_frontmatter,  # pyright: ignore[reportPrivateUsage]
    _parse_frontmatter,  # pyright: ignore[reportPrivateUsage]
    load_graph,
    read_cursor,
    write_cursor,
    write_node,
)


def _mk_node(
    nid: str,
    *,
    parent: str | None = None,
    title: str = "t",
    rationale: str = "r",
    relevant_paths: tuple[str, ...] = (),
    children: tuple[str, ...] = (),
) -> TaskNode:
    now = datetime(2025, 1, 1, tzinfo=UTC)
    return TaskNode(
        id=nid,
        parent_id=parent,
        title=title,
        rationale=rationale,
        acceptance="a",
        relevant_paths=relevant_paths,
        depends_on=(),
        children=children,
        status="pending",
        created_at=now,
        updated_at=now,
        created_by="planner",
    )


@pytest.mark.parametrize(
    "evil",
    [
        "carriage\rreturn",
        "line\u2028sep",
        "para\u2029sep",
        "vtab\x0bhere",
        "formfeed\x0chere",
        "nel\x85here",
        "filesep\x1chere",
        "recordsep\x1ehere",
        "crlf\r\nmix",
        "trailing\r",
        'quote"and\\back',
        "newline\nhere",
    ],
)
def test_frontmatter_round_trips_adversarial_scalars(evil: str) -> None:
    """An LLM-chosen task title with line-separator chars must not break the
    frontmatter round-trip (it used to crash _parse_frontmatter on resume --
    a denial-of-resume from the untrusted worker)."""
    node = _mk_node("0" * 25 + "A", title=evil, rationale=evil)
    node = node.model_copy(update={"acceptance": evil})
    rt = _parse_frontmatter(_dump_frontmatter(node))  # must not raise
    assert rt.title == evil
    assert rt.rationale == evil
    assert rt.acceptance == evil


def test_layout_ensure_creates_dirs(tmp_path: Path) -> None:
    layout = SessionLayout(state_dir=tmp_path / ".agent6", session_id="run1")
    layout.ensure()
    assert layout.graph_dir.is_dir()
    assert layout.transcripts_dir.is_dir()


def test_write_and_load_single_node(tmp_path: Path) -> None:
    layout = SessionLayout(state_dir=tmp_path / ".agent6", session_id="run1")
    layout.ensure()
    n = _mk_node("0" * 25 + "A", relevant_paths=("src/a.py", "src/b.py"))
    write_node(layout, {n.id: n}, n)
    loaded = load_graph(layout)
    assert n.id in loaded
    got = loaded[n.id]
    assert got.title == "t"
    assert got.relevant_paths == ("src/a.py", "src/b.py")


def test_frontmatter_quotes_special_chars(tmp_path: Path) -> None:
    layout = SessionLayout(state_dir=tmp_path / ".agent6", session_id="run1")
    layout.ensure()
    n = _mk_node(
        "0" * 25 + "B",
        title='has "quotes" and \\backslash',
        rationale="line1\nline2",
    )
    write_node(layout, {n.id: n}, n)
    loaded = load_graph(layout)
    assert loaded[n.id].title == 'has "quotes" and \\backslash'
    assert loaded[n.id].rationale == "line1\nline2"


def test_load_graph_reconstructs_parent_dir_layout(tmp_path: Path) -> None:
    layout = SessionLayout(state_dir=tmp_path / ".agent6", session_id="run1")
    layout.ensure()
    root = _mk_node("0" * 25 + "C", children=("0" * 25 + "D",))
    child = _mk_node("0" * 25 + "D", parent=root.id)
    nodes = {root.id: root, child.id: child}
    write_node(layout, nodes, root)
    write_node(layout, nodes, child)
    loaded = load_graph(layout)
    assert loaded[root.id].children == (child.id,)
    assert loaded[child.id].parent_id == root.id


def test_checkpoints_dir_and_path(tmp_path: Path) -> None:
    layout = SessionLayout(state_dir=tmp_path, session_id="run1")
    layout.ensure()
    assert layout.checkpoints_dir.is_dir()
    assert layout.checkpoint_path(7) == layout.checkpoints_dir / "0007.json"
    assert layout.checkpoint_path(1234) == layout.checkpoints_dir / "1234.json"


def test_list_checkpoint_turns(tmp_path: Path) -> None:
    from agent6.graph.storage import list_checkpoint_turns

    layout = SessionLayout(state_dir=tmp_path, session_id="run1")
    # No checkpoints dir yet (old run): empty.
    assert list_checkpoint_turns(layout) == []
    layout.ensure()
    for turn in (3, 1, 10, 2):
        layout.checkpoint_path(turn).write_text("{}", encoding="utf-8")
    # A stray non-numeric file is ignored.
    (layout.checkpoints_dir / "notes.json").write_text("{}", encoding="utf-8")
    assert list_checkpoint_turns(layout) == [1, 2, 3, 10]


def test_load_graph_skips_a_path_traversing_node_id(tmp_path: Path, capsys: object) -> None:
    """A node .md whose 26-char id carries path separators ('../zzz...') must
    be SKIPPED like every other corrupt file -- unvalidated, it survived load
    and the next write_node resolved OUTSIDE graph_dir (an atomic write landed
    above the run's graph tree). The Crockford charset validator at the reload
    boundary turns it into the standard skip-with-warning."""
    from agent6.graph.storage import load_graph
    from agent6.sessions.layout import SessionLayout

    layout = SessionLayout(state_dir=tmp_path / "state", session_id="r1")
    layout.ensure()
    bad_id = "../zzzzzzzzzzzzzzzzzzzzzzz"
    assert len(bad_id) == 26
    (layout.graph_dir / "planted.md").write_text(
        f"---\nid: {bad_id}\nparent_id: null\ntitle: evil\n"
        "status: pending\ncreated_at: 2026-01-01T00:00:00+00:00\n"
        "updated_at: 2026-01-01T00:00:00+00:00\ncreated_by: worker\n---\nbody\n",
        encoding="utf-8",
    )
    nodes = load_graph(layout)
    assert bad_id not in nodes  # skipped, not loaded


def test_graph_version_round_trips_and_old_files_default_zero(tmp_path: Path) -> None:
    """The node file carries the mutation stamp; a file written before stamps
    existed loads as 0 (never a parse failure)."""
    layout = SessionLayout(state_dir=tmp_path, session_id="s")
    layout.ensure()
    node = _mk_node("0" * 26).model_copy(update={"graph_version": 7})
    write_node(layout, {node.id: node}, node)
    loaded = load_graph(layout)
    assert loaded[node.id].graph_version == 7
    # Strip the stamp line: the pre-stamp file shape.
    md = next(layout.graph_dir.rglob("*.md"))
    md.write_text(
        "\n".join(
            ln
            for ln in md.read_text(encoding="utf-8").split("\n")
            if not ln.startswith("graph_version:")
        ),
        encoding="utf-8",
    )
    assert load_graph(layout)[node.id].graph_version == 0


def test_a_malformed_cursor_reads_as_none_and_says_so(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A torn or hand-edited cursor.json crashed fork and /undo (a bare
    AttributeError for a list, a JSONDecodeError for a fragment) and left a
    half-built run dir; its sibling readers degrade, and so does this one."""
    layout = SessionLayout(state_dir=tmp_path / ".agent6", session_id="run1")
    layout.ensure()
    for bad in ("[]", '{"node_id"', '{"node_id": 5}', '"n9"', "{}"):
        layout.cursor_path.write_text(bad, encoding="utf-8")
        assert read_cursor(layout) is None
    assert capsys.readouterr().err.count("ignoring malformed") == 5
    write_cursor(layout, "n1")
    assert read_cursor(layout) == "n1"
