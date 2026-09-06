# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""DAG-as-tool handlers: add_task, update_task, list_tasks. All raise
ToolError if no curator was wired so standalone test instantiation works
unchanged."""

from __future__ import annotations

from typing import Any

from agent6.graph.curator import GraphCurator
from agent6.graph.models import (
    AddDependencyIntent,
    AddSubtaskIntent,
    TaskNodeDraft,
    UpdateStatusIntent,
)
from agent6.graph.order import tree_order
from agent6.tools.errors import ToolError
from agent6.tools.results import (
    AddTaskResult,
    ListTasksResult,
    UpdateTaskResult,
)
from agent6.tools.schema import (
    DagAddTaskInput,
    DagListTasksInput,
    DagUpdateTaskInput,
)


def add_task(
    curator: GraphCurator | None, run_root_node_id: str | None, raw: dict[str, Any]
) -> AddTaskResult:
    if curator is None:
        raise ToolError("DAG curator not available in this run")
    args = DagAddTaskInput.model_validate(raw)
    parent_id = args.parent_id or run_root_node_id
    draft = TaskNodeDraft(
        title=args.title,
        rationale=args.rationale,
        acceptance=args.acceptance,
        relevant_paths=args.relevant_paths,
        depends_on=args.depends_on,
        standing=args.standing,
        created_by="worker",
    )
    intent = AddSubtaskIntent(parent_id=parent_id, draft=draft, after=args.after)
    node = curator.add_subtask(intent)
    return AddTaskResult(
        id=node.id,
        parent_id=node.parent_id,
        title=node.title,
        status=node.status,
    )


def update_task(curator: GraphCurator | None, raw: dict[str, Any]) -> UpdateTaskResult:
    if curator is None:
        raise ToolError("DAG curator not available in this run")
    args = DagUpdateTaskInput.model_validate(raw)
    node = None
    if args.status is not None:
        if args.status in ("skipped", "obsolete"):
            current = curator.get(args.id)
            if current.standing and current.created_by == "steering":
                # The operator's --standing goal: the operator retires it (a
                # steer, or stopping the run). The model retiring it converts
                # the never-finishing fallback into an ordinary early finish.
                raise ToolError(
                    f"update_task: {args.id} is the operator's standing goal;"
                    " it stays until the operator retires it. Work it when"
                    " nothing else is ready, or finish_session on a hard limit."
                )
        intent = UpdateStatusIntent(
            id=args.id,
            new_status=args.status,  # type: ignore[arg-type]  # pydantic validates the literal
            note=args.note,
        )
        node = curator.update_status(intent)
    # Unknown ids and cycles are rejected by the curator; dispatch()'s generic
    # wrapper surfaces that rejection to the model as a ToolError.
    for dep in args.depends_on:
        node = curator.add_dependency(AddDependencyIntent(id=args.id, depends_on=dep))
    if node is None:
        raise ToolError("pass status and/or depends_on")
    return UpdateTaskResult(
        id=node.id, status=node.status, title=node.title, depends_on=tuple(node.depends_on)
    )


def list_tasks(curator: GraphCurator | None, raw: dict[str, Any]) -> ListTasksResult:
    if curator is None:
        raise ToolError("DAG curator not available in this run")
    args = DagListTasksInput.model_validate(raw)
    # Wire surface: to_wire() JSONs each task to the model; the projected shape
    # (and its list-valued relevant_paths/depends_on) is what tool callers hold.
    out: list[dict[str, Any]] = []
    # Tree order, the order the frontier executes and every renderer shows:
    # iterating the map gave insertion order live and filesystem order after a
    # resume, so the model read back a plan it had not written.
    nodes = curator.nodes()
    for node_id in tree_order(nodes):
        node = nodes[node_id]
        if args.status and node.status != args.status:
            continue
        out.append(
            {
                "id": node_id,
                "parent_id": node.parent_id,
                "title": node.title,
                "status": node.status,
                "acceptance": node.acceptance,
                "relevant_paths": list(node.relevant_paths),
                "depends_on": list(node.depends_on),
            }
        )
    return ListTasksResult(tasks=tuple(out), count=len(out))
