# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The persistent task-graph models: nodes plus the LLM-emitted curator intents
that mutate them, a doubly-linked tree keyed by time-sortable ULID ids.

Every node carries a 26-char Crockford-base32 ULID `id` and a `parent_id`; the
tree is doubly linked (a parent lists each child's id in `children`, each child
names its `parent_id`), a symmetry the curator maintains on every mutation.
`status` ranges over the fixed `NodeStatus` vocabulary (pending, in_progress,
passed, failed, skipped, obsolete).

These cross trust boundaries (LLM-emitted intents, disk reload), so they are
pydantic per project convention. Internal-only value types remain frozen
dataclasses in `agent6.types`.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from agent6.graph.ulid import CROCKFORD

_MODEL_CONFIG = ConfigDict(extra="forbid", frozen=True)

# ---- domain types ---------------------------------------------------------

NodeStatus = Literal[
    "pending",
    "in_progress",
    "passed",
    "failed",
    "skipped",
    "obsolete",
]

NodeActor = Literal[
    "planner",
    "worker",
    "steering",
    "alignment_guard",
    "user",
    "reviewer",
]


class TaskNodeDraft(BaseModel):
    """A new-node payload, id is assigned by the curator on insert."""

    model_config = _MODEL_CONFIG

    title: str = Field(min_length=1)
    rationale: str = ""
    acceptance: str = ""
    relevant_paths: tuple[str, ...] = ()
    depends_on: tuple[str, ...] = ()
    created_by: NodeActor
    # A standing task is the run's fallback: it never passes, and the focus
    # frontier selects it only when no ordinary subtask is ready.
    standing: bool = False


class TaskNode(BaseModel):
    """A persisted task-graph node: a time-sortable 26-char ULID `id`, a
    `parent_id`/`children` pair the curator keeps mutually consistent, and a
    `status` drawn from the fixed `NodeStatus` vocabulary."""

    model_config = _MODEL_CONFIG

    id: str = Field(min_length=26, max_length=26)
    parent_id: str | None
    title: str = Field(min_length=1)
    rationale: str = ""
    acceptance: str = ""
    relevant_paths: tuple[str, ...] = ()
    depends_on: tuple[str, ...] = ()
    children: tuple[str, ...] = ()
    status: NodeStatus = "pending"
    created_at: datetime
    updated_at: datetime
    created_by: NodeActor
    commit_sha: str = ""
    notes: str = ""
    # See TaskNodeDraft.standing: the never-passing fallback node.
    standing: bool = False
    # The graph_version of the mutation that last wrote this node (the same
    # number its journal entry carries). 0 = written before stamps existed.
    # Lets the curator detect a journal that lost its tail: a node stamped
    # newer than the journal's max version is exactly that crash.
    graph_version: int = 0

    @field_validator("id")
    @classmethod
    def _id_is_crockford(cls, v: str) -> str:
        # The id becomes a filesystem path component (node_md_path builds the
        # on-disk path from the ancestor id chain), so the RELOAD trust
        # boundary must reject a crafted 26-char id carrying separators
        # ('../zzz...') that would make the next write_node escape graph_dir.
        # A bad-id file then fails validation -> load_graph skips it with a
        # warning, exactly like every other corrupt node file. new_ulid()
        # always emits valid Crockford, so real ids are unaffected.
        if any(ch not in CROCKFORD for ch in v):
            raise ValueError(f"node id is not Crockford base32: {v!r}")
        return v


# ---- curator intent payloads ---------------------------------------------


class AddSubtaskIntent(BaseModel):
    model_config = _MODEL_CONFIG

    op: Literal["add_subtask"] = "add_subtask"
    parent_id: str | None
    draft: TaskNodeDraft
    # Place the new child directly after this sibling instead of appending.
    # The children list is the order the frontier executes, so this is how
    # work is inserted between two steps rather than re-planned around.
    after: str | None = None


class UpdateStatusIntent(BaseModel):
    model_config = _MODEL_CONFIG

    op: Literal["update_status"] = "update_status"
    id: str
    new_status: NodeStatus
    note: str = ""


class AddDependencyIntent(BaseModel):
    model_config = _MODEL_CONFIG

    op: Literal["add_dependency"] = "add_dependency"
    id: str
    depends_on: str


class RecordCommitIntent(BaseModel):
    model_config = _MODEL_CONFIG

    op: Literal["record_commit"] = "record_commit"
    id: str
    sha: str


class SetCursorIntent(BaseModel):
    model_config = _MODEL_CONFIG

    op: Literal["set_cursor"] = "set_cursor"
    id: str | None
