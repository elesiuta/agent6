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

    Through the read model's own walk, so this renders what the frontier and
    `list_tasks` see: a node its parent's `children` list does not name (the
    crash window `add_subtask` documents) is still shown, where a walk of its
    own dropped it -- and the operator's DAG view then omitted the very task
    the run was working on. Nodes are ordered by id, which is a ULID, so roots
    read in creation order however the graph came off disk."""
    ordered = {nid: nodes[nid].model_dump() for nid in sorted(nodes)}
    out: list[str] = []
    for view in task_tree_views(ordered, None):
        commit_sha = nodes[view.id].commit_sha
        commit = f"  ({commit_sha[:7]})" if show_commit and commit_sha else ""
        glyph = TASK_STATUS_GLYPH.get(view.status, "·")
        out.append(f"{'  ' * view.depth}{glyph} {view.title}{commit}")
    return out
