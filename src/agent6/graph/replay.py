# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Rebuild a task graph as it stood at an older `graph_version`.

`graph.jsonl` records every curator mutation, stamped with the version it
produced, so a checkpoint's `graph_version` names an exact past state of the
DAG. `agent6 fork --at-turn N` needs that state: the conversation and the
workspace commit come from turn N, and a graph from the run's future would show
the forked session tasks it never created and statuses for work its tree does
not contain.

The journal records OPERATIONS, not node content, and the content the
operations never touch (title, rationale, acceptance, relevant_paths,
created_at, created_by) is immutable after creation. So the rebuild starts from
the current nodes and undoes every mutation stamped after the target version.
Two fields cannot be unwound and stay at their current value, both display-only:
`notes` (appended prose with no per-append record) and `updated_at`. A
mutation whose journal entry was lost to a crash (see the curator's boot
resync) has nothing to undo, so it stays visible in every replayed version.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from agent6.graph.models import TaskNode


@dataclass(frozen=True, slots=True)
class ReplayedGraph:
    """A DAG as of one `graph_version`: the surviving nodes and the cursor."""

    nodes: dict[str, TaskNode]
    cursor: str | None


def _usable(journal: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Journal entries in applied order, skipping any line without a usable op
    and version. The file is append-only and already ordered; a torn or foreign
    line is dropped rather than allowed to abort the rebuild (`load_graph`
    treats a malformed node file the same way)."""
    return [
        e
        for e in journal
        if isinstance(e.get("op"), str) and isinstance(e.get("graph_version"), int)
    ]


def journal_prefix(journal: Iterable[dict[str, Any]], version: int) -> list[dict[str, Any]]:
    """The entries that produced *version*, so a rebuilt graph ships with the
    journal it actually has and its curator keeps numbering from there."""
    return [e for e in _usable(journal) if int(e["graph_version"]) <= version]


def _str_field(entry: dict[str, Any], key: str) -> str | None:
    value = entry.get(key)
    return value if isinstance(value, str) else None


def _children_at(current: tuple[str, ...], kept: dict[str, TaskNode]) -> tuple[str, ...]:
    """The parent's children at the target version: the surviving ones, in the
    order the parent holds them (which is the order the frontier executes)."""
    return tuple(c for c in current if c in kept)


def graph_at_version(
    nodes: dict[str, TaskNode],
    journal: Iterable[dict[str, Any]],
    version: int,
    *,
    current_cursor: str | None = None,
) -> ReplayedGraph:
    """The graph as of *version*, from the CURRENT *nodes* plus the *journal*.

    A node whose `add_subtask` is stamped after *version* did not exist yet
    and is dropped. A node the journal never mentions (a truncated or
    pre-journal graph) is kept as-is: unknown creation time is not evidence of
    a later one, and dropping it would silently empty the fork's DAG.
    """
    past = _usable(journal)
    at = journal_prefix(past, version)

    born: dict[str, int] = {
        nid: int(e["graph_version"])
        for e in past
        if e["op"] == "add_subtask" and (nid := _str_field(e, "id")) is not None
    }
    kept = {nid: n for nid, n in nodes.items() if born.get(nid, version) <= version}

    # Last write wins, exactly as the curator applied them.
    status: dict[str, str] = {}
    for e in at:
        nid = _str_field(e, "id")
        if nid is None:
            continue
        if e["op"] == "obsolete":
            status[nid] = "obsolete"
        elif e["op"] == "update_status" and (new := _str_field(e, "new_status")) is not None:
            status[nid] = new
    commit: dict[str, str] = {
        nid: sha
        for e in at
        if e["op"] == "record_commit"
        and (nid := _str_field(e, "id")) is not None
        and (sha := _str_field(e, "sha")) is not None
    }
    deps_after: dict[str, set[str]] = {}
    for e in past:
        if e["op"] != "add_dependency" or int(e["graph_version"]) <= version:
            continue
        nid, dep = _str_field(e, "id"), _str_field(e, "depends_on")
        if nid is not None and dep is not None:
            deps_after.setdefault(nid, set()).add(dep)
    # The cursor as of *version*: the last one the prefix set. With none in the
    # prefix but some later in the journal, the run held NO cursor here, and
    # the CURRENT one would give a fork at turn 1 the focus of the last turn.
    # *current_cursor* is
    # for a graph whose journal records no cursor at all (a pre-journal graph,
    # where the cursor file is the only record).
    cursors = [e for e in at if e["op"] == "set_cursor"]
    if cursors:
        cursor = _str_field(cursors[-1], "id")
    elif any(e["op"] == "set_cursor" for e in past):
        cursor = None
    else:
        cursor = current_cursor

    rebuilt = {
        nid: node.model_copy(
            update={
                # A node whose birth the journal recorded started out pending
                # with no commit; one it never mentioned keeps what it has.
                "status": status.get(nid, "pending" if nid in born else node.status),
                "commit_sha": commit.get(nid, "" if nid in born else node.commit_sha),
                "depends_on": tuple(d for d in node.depends_on if d not in deps_after.get(nid, ())),
                "children": _children_at(node.children, kept),
                # A stamp past *version* names a mutation this rebuild undid;
                # left in place, the curator reads the rebuilt journal as a
                # lost tail on every open of the fork.
                "graph_version": min(node.graph_version, version),
            }
        )
        for nid, node in kept.items()
    }
    return ReplayedGraph(nodes=rebuilt, cursor=cursor if cursor in rebuilt else None)
