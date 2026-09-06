# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Render a persisted task DAG as an indented text tree.

Shared by `sessions graph`, `sessions show`, and the live CLI stream's plan block, so
the decomposed plan reads the same everywhere (and is visible for a headless
run that never opened the TUI #plan pane)."""

from __future__ import annotations

from agent6.graph.models import TaskNode
from agent6.viewmodel import task_tree_views
from agent6.viewmodel.format import TASK_STATUS_GLYPH


def tree_lines_from_event_nodes(nodes: dict[str, object], cursor: str | None = None) -> list[str]:
    """Same tree, from a `graph.update` event's raw node dicts (title / status /
    parent_id / children) rather than TaskNode models. Used by the live CLI
    stream, which folds events, not the persisted graph. The focus task is
    marked with the in-progress glyph regardless of its stored status."""
    return [
        f"{'  ' * v.depth}{TASK_STATUS_GLYPH.get('in_progress' if v.is_cursor else v.status, '·')}"
        f" {v.title}"
        for v in task_tree_views(nodes, cursor)
    ]


def task_tree_lines(nodes: dict[str, TaskNode], *, show_commit: bool = False) -> list[str]:
    """DFS, left-to-right, one line per node: `<indent><glyph> <title>`.

    Roots are ordered by creation; children keep insertion order (the curator
    preserves it). Returns [] for an empty graph."""
    roots = sorted((n for n in nodes.values() if n.parent_id is None), key=lambda n: n.created_at)
    out: list[str] = []
    for root in roots:
        _walk(root, nodes, depth=0, out=out, show_commit=show_commit)
    return out


def _walk(
    node: TaskNode, nodes: dict[str, TaskNode], *, depth: int, out: list[str], show_commit: bool
) -> None:
    glyph = TASK_STATUS_GLYPH.get(node.status, "·")
    commit = f"  ({node.commit_sha[:7]})" if show_commit and node.commit_sha else ""
    out.append(f"{'  ' * depth}{glyph} {node.title}{commit}")
    for child_id in node.children:
        child = nodes.get(child_id)
        if child is None:
            out.append(f"{'  ' * (depth + 1)}? <missing child {child_id}>")
            continue
        _walk(child, nodes, depth=depth + 1, out=out, show_commit=show_commit)
