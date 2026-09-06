# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Authoritative in-process graph mutator.

`GraphCurator` is the single source of truth for one run's task graph. It runs
in-process in the agent (`app/_session.py` constructs it for run and resume),
inheriting that process's confinement and writing the run's graph under the
out-of-tree per-repo state dir. Unit tests instantiate it the same way.

Mutations are validated structurally, then applied as:

  1. mutate in-memory graph state
  2. atomically write the affected node `.md` files, each stamped with the
     version this mutation will journal (nodes are the content authority)
  3. append the entry to `graph.jsonl` (the journal, append-only audit log)
     and commit the `graph_version` bump
  4. (if topology changed) atomically regenerate `graph.dot`

The flock around every mutation prevents interleaved file writes from
accidental parallel curator instances (which we explicitly forbid). It does
not merge their in-memory state: each instance caches the graph at
construction, so a second live instance would still lose updates. One curator
per run is the invariant; the lock only bounds the damage if it is broken. The
CLI upholds the invariant with a run-level single-writer flock
(`sessions.lock.acquire_single_writer` on `<session-dir>/worker.lock`, the analogue
of `machine_lock`): a second `agent6 run`/`resume` on the same run dir
refuses rather than constructing a second curator (`fork` copies under the
graph flock and never constructs one).

Fail-safe: a mutation updates `self._nodes` in memory BEFORE writing to disk,
so a write-path fault (ENOSPC, a serialization error, a cycle surfacing from
`write_node`) can leave in-memory state ahead of disk. `_mutating` reloads
from disk (the source of truth) before surfacing such a fault, so a later read
never observes a node that was never persisted.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Generator
from contextlib import contextmanager
from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict

from agent6.graph.models import (
    AddDependencyIntent,
    AddSubtaskIntent,
    NodeActor,
    NodeStatus,
    RecordCommitIntent,
    SetCursorIntent,
    TaskNode,
    UpdateStatusIntent,
)
from agent6.graph.order import has_open_child
from agent6.graph.storage import (
    SessionLayout,
    flock,
    load_graph,
    read_cursor,
    write_cursor,
    write_dot,
    write_journal,
    write_node,
)
from agent6.graph.ulid import new_ulid


class CuratorError(Exception):
    """A curator intent was rejected (validation, not I/O)."""


class _JournalBase(BaseModel):
    """Base of the typed graph.jsonl entries: what each mutation appends to the
    append-only audit log. The node `.md` files are the source of truth; the
    fields read back are `graph_version` (`_compute_graph_version`) and,
    by `graph.replay` for `fork --at-turn`, each entry's mutation fields,
    stamped by `_post_mutation` after the bump (0 only pre-stamp). The
    `ts` timestamp is added by `storage.write_journal`, which also sorts
    keys, so field order here is presentational only.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    graph_version: int = 0


class AddSubtaskJournal(_JournalBase):
    op: Literal["add_subtask"] = "add_subtask"
    id: str  # the ASSIGNED ULID (the intent has none yet)
    parent_id: str | None
    by: NodeActor


class UpdateStatusJournal(_JournalBase):
    op: Literal["update_status"] = "update_status"
    id: str
    new_status: NodeStatus


class AddDependencyJournal(_JournalBase):
    op: Literal["add_dependency"] = "add_dependency"
    id: str
    depends_on: str


class RecordCommitJournal(_JournalBase):
    op: Literal["record_commit"] = "record_commit"
    id: str
    sha: str


class SetCursorJournal(_JournalBase):
    op: Literal["set_cursor"] = "set_cursor"
    id: str | None


JournalEntry = (
    AddSubtaskJournal
    | UpdateStatusJournal
    | AddDependencyJournal
    | RecordCommitJournal
    | SetCursorJournal
)


def _place(children: tuple[str, ...], new_id: str, after: str | None) -> tuple[str, ...]:
    """*children* with *new_id* appended, or inserted just after *after*."""
    if after is None:
        return (*children, new_id)
    at = children.index(after) + 1
    return (*children[:at], new_id, *children[at:])


def _now() -> datetime:
    return datetime.now(tz=UTC)


class GraphCurator:
    """Owns one run's graph, in-memory and on-disk."""

    def __init__(self, layout: SessionLayout) -> None:
        self._layout = layout
        layout.ensure()
        self._nodes: dict[str, TaskNode] = load_graph(layout)
        self._graph_version = self._compute_graph_version()
        # A node stamped newer than the journal's max version is a death
        # between the node write and its journal append: the entry is gone.
        # Resync the counter so the lost number is never REUSED (two ops
        # sharing a version corrupts fork-at-version undo) and say so; the
        # change stays current but is invisible to historical replay.
        node_max = max((n.graph_version for n in self._nodes.values()), default=0)
        if node_max > self._graph_version:
            sys.stderr.write(
                f"agent6: graph journal lost its tail (a node is stamped v{node_max},"
                f" the journal ends at v{self._graph_version}); continuing from"
                f" v{node_max}. The lost operation stays in the current graph but"
                " will not appear in fork --at-turn replays.\n"
            )
            self._graph_version = node_max

    def _compute_graph_version(self) -> int:
        return max(
            (
                int(gv)
                for entry in self._iter_recent_journal()
                if isinstance((gv := entry.get("graph_version", 0)), int)
            ),
            default=len(self._nodes),
        )

    # ---- accessors --------------------------------------------------------

    @property
    def layout(self) -> SessionLayout:
        return self._layout

    @property
    def graph_version(self) -> int:
        return self._graph_version

    def nodes(self) -> dict[str, TaskNode]:
        return dict(self._nodes)

    def get(self, node_id: str) -> TaskNode:
        if node_id not in self._nodes:
            raise CuratorError(f"unknown node: {node_id}")
        return self._nodes[node_id]

    def cursor(self) -> str | None:
        return read_cursor(self._layout)

    # ---- mutations --------------------------------------------------------

    @contextmanager
    def _mutating(self) -> Generator[None]:
        """Flock the run dir for one mutation, with the disk-fault fail-safe.

        A `CuratorError` is a pre-mutation validation reject (nothing was
        applied), so it propagates untouched. Any other fault escapes AFTER the
        in-memory graph was already updated, so reload from disk (the source of
        truth) before re-raising: a later read then never sees a node the write
        path failed to persist. The reload runs under the same flock so a
        concurrent operator read can't observe the skewed state."""
        with flock(self._layout.lock_path):
            try:
                yield
            except CuratorError:
                raise
            except Exception:
                self._nodes = load_graph(self._layout)
                self._graph_version = self._compute_graph_version()
                raise

    def _write(self, node: TaskNode) -> TaskNode:
        """Stamp *node* with the version this mutation will journal, cache it,
        write its file, and return the stamped copy. Every write inside one
        mutation carries the same number `_post_mutation` then records, so a
        journal that lost its tail is detectable at load (the resync in
        __init__)."""
        stamped = node.model_copy(update={"graph_version": self._graph_version + 1})
        self._nodes[stamped.id] = stamped
        write_node(self._layout, self._nodes, stamped)
        return stamped

    def add_subtask(self, intent: AddSubtaskIntent) -> TaskNode:
        with self._mutating():
            parent = self._nodes.get(intent.parent_id) if intent.parent_id else None
            if intent.parent_id is not None and parent is None:
                raise CuratorError(f"add_subtask: unknown parent {intent.parent_id!r}")
            if intent.after is not None and (parent is None or intent.after not in parent.children):
                raise CuratorError(
                    f"add_subtask: after {intent.after!r} is not a child of"
                    f" {intent.parent_id!r}; a position names a sibling"
                )
            for dep in intent.draft.depends_on:
                if dep not in self._nodes:
                    raise CuratorError(f"add_subtask: unknown dep {dep!r}")
            now = _now()
            node = TaskNode(
                id=new_ulid(),
                parent_id=intent.parent_id,
                title=intent.draft.title,
                rationale=intent.draft.rationale,
                acceptance=intent.draft.acceptance,
                relevant_paths=intent.draft.relevant_paths,
                depends_on=intent.draft.depends_on,
                standing=intent.draft.standing,
                children=(),
                status="pending",
                created_at=now,
                updated_at=now,
                created_by=intent.draft.created_by,
            )
            # Write the child node BEFORE the parent->child link so a crash in
            # between can at worst leave an orphan node (parent_id set, not yet
            # listed in parent.children) rather than a dangling reference to a
            # child whose .md never made it to disk.
            node = self._write(node)
            if parent is not None:
                updated_parent = parent.model_copy(
                    update={
                        "children": _place(parent.children, node.id, intent.after),
                        "updated_at": now,
                    }
                )
                self._write(updated_parent)
            self._post_mutation(
                AddSubtaskJournal(
                    id=node.id, parent_id=intent.parent_id, by=intent.draft.created_by
                )
            )
            return node

    def update_status(self, intent: UpdateStatusIntent) -> TaskNode:
        with self._mutating():
            node = self.get(intent.id)
            # An end is final: a passed task may only be retired, and a retired
            # one stays retired, or `passed -> obsolete -> pending` would walk
            # around the first rule and re-open work every dependent was told
            # had passed. Needed again means a new task.
            if node.status == "passed" and intent.new_status != "obsolete":
                raise CuratorError(
                    f"cannot transition passed node {intent.id} to {intent.new_status}"
                )
            if node.status in ("skipped", "obsolete") and intent.new_status != node.status:
                raise CuratorError(
                    f"{intent.id} is retired ({node.status}) and stays retired;"
                    " add_task if the work is needed after all"
                )
            if node.standing and intent.new_status == "passed":
                raise CuratorError(
                    f"a standing task never passes ({intent.id}); mark it skipped or"
                    " obsolete to retire it"
                )
            if (
                intent.new_status == "passed"
                and node.parent_id is not None
                and has_open_child(self._nodes, node)
            ):
                # A parent with open children is a container: the frontier
                # surfaces its children instead, and passing it would satisfy
                # every dependency on it while the work they name goes undone.
                # The root is the whole
                # job, not a unit of work: nothing depends on it, and a run
                # that ends with a standing goal or a subtask left open still
                # completed it.
                raise CuratorError(
                    f"{intent.id} has open children, so it is not finished; mark them"
                    " passed, skipped or obsolete first"
                )
            updated = node.model_copy(
                update={
                    "status": intent.new_status,
                    "updated_at": _now(),
                    "notes": (
                        node.notes if not intent.note else (node.notes + "\n" + intent.note).strip()
                    ),
                }
            )
            updated = self._write(updated)
            self._post_mutation(UpdateStatusJournal(id=updated.id, new_status=intent.new_status))
            return updated

    def add_dependency(self, intent: AddDependencyIntent) -> TaskNode:
        with self._mutating():
            node = self.get(intent.id)
            if intent.depends_on not in self._nodes:
                raise CuratorError(f"unknown dep {intent.depends_on!r}")
            if intent.depends_on in node.depends_on:
                return node
            if self._would_introduce_cycle(intent.id, intent.depends_on):
                raise CuratorError(
                    f"add_dependency {intent.id} -> {intent.depends_on} would introduce cycle"
                )
            updated = node.model_copy(
                update={
                    "depends_on": (*node.depends_on, intent.depends_on),
                    "updated_at": _now(),
                }
            )
            updated = self._write(updated)
            self._post_mutation(AddDependencyJournal(id=updated.id, depends_on=intent.depends_on))
            return updated

    def record_commit(self, intent: RecordCommitIntent) -> TaskNode:
        with self._mutating():
            node = self.get(intent.id)
            updated = node.model_copy(update={"commit_sha": intent.sha, "updated_at": _now()})
            updated = self._write(updated)
            self._post_mutation(RecordCommitJournal(id=updated.id, sha=intent.sha))
            return updated

    def set_cursor(self, intent: SetCursorIntent) -> None:
        with self._mutating():
            if intent.id is not None and intent.id not in self._nodes:
                raise CuratorError(f"set_cursor: unknown node {intent.id!r}")
            write_cursor(self._layout, intent.id)
            self._post_mutation(SetCursorJournal(id=intent.id), regen_dot=False)

    # ---- internals --------------------------------------------------------

    def _post_mutation(self, entry: JournalEntry, *, regen_dot: bool = True) -> None:
        self._graph_version += 1
        stamped = entry.model_copy(update={"graph_version": self._graph_version})
        write_journal(self._layout, stamped.model_dump(mode="json"))
        if regen_dot:
            write_dot(self._layout, self._nodes)

    def _iter_recent_journal(self) -> list[dict[str, object]]:
        path = self._layout.journal_path
        if not path.is_file():
            return []
        entries: list[dict[str, object]] = []
        for raw in path.read_text(encoding="utf-8").splitlines():
            stripped = raw.strip()
            if not stripped:
                continue
            try:
                entries.append(json.loads(stripped))
            except json.JSONDecodeError:
                # A crash mid-append can leave a torn final line. The node .md
                # files are the source of truth (read atomically by load_graph)
                # and graph_version is a self-healing monotonic counter, so skip
                # the corrupt line rather than crashing curator startup -- which
                # would otherwise make the whole run unresumable.
                sys.stderr.write(f"agent6: skipping malformed journal line: {stripped[:80]!r}\n")
        return entries

    def _would_introduce_cycle(self, src: str, new_dep: str) -> bool:
        """True iff adding src→new_dep would create a cycle in the dep DAG."""
        # Walk dep transitively from new_dep; if the walk reaches src, it's a cycle.
        stack = [new_dep]
        seen: set[str] = set()
        while stack:
            cur = stack.pop()
            if cur == src:
                return True
            if cur in seen:
                continue
            seen.add(cur)
            node = self._nodes.get(cur)
            if node is None:
                continue  # dangling depends_on edge (target missing): not a cycle
            stack.extend(node.depends_on)
        return False
