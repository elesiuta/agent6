# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Render a task DAG as an indented text tree.

Shared by `sessions graph`, `sessions show`, and the live CLI stream's plan
block, so the decomposed plan reads the same everywhere (and is visible for a
headless run that never opened the TUI #plan pane)."""

from __future__ import annotations

from typing import Any

from agent6.viewmodel import task_tree_views
from agent6.viewmodel.format import TASK_STATUS_GLYPH


def task_tree_lines(nodes: dict[str, Any], cursor: str | None = None) -> list[str]:
    """DFS, left-to-right, one line per node: `<indent><glyph> <title>`, plus
    the commit's short sha when the node carries one. Over raw node dicts (a
    `graph.update` event's, or a persisted graph's `model_dump`, in the order
    roots should read), through the read model's own walk, so this renders
    what the frontier and `list_tasks` see: a node its parent's `children`
    list does not name (the crash window `add_subtask` documents) is still
    shown. The focus task wears the in-progress glyph regardless of its
    stored status."""
    out: list[str] = []
    for view in task_tree_views(nodes, cursor):
        node = nodes.get(view.id)
        sha = node.get("commit_sha") if isinstance(node, dict) else None
        commit = f"  ({sha[:7]})" if isinstance(sha, str) and sha else ""
        glyph = TASK_STATUS_GLYPH.get("in_progress" if view.is_cursor else view.status, "·")
        out.append(f"{'  ' * view.depth}{glyph} {view.title}{commit}")
    return out
