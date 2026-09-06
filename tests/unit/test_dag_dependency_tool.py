# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""DAG dependency edges at the LLM-facing layer: `depends_on` rides `add_task`
and `update_task` (there is no separate dependency tool).

Curator-level semantics (cycle rejection, journal op, focus gating on
depends_on) are covered by test_graph_curator.py and test_workflow.py; these
tests cover the LLM-facing layer added on top.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent6.config import Config, load_config
from agent6.graph.curator import GraphCurator
from agent6.graph.models import AddSubtaskIntent, TaskNodeDraft, UpdateStatusIntent
from agent6.sessions.layout import SessionLayout
from agent6.tools.dispatch import ToolDispatcher, ToolError
from agent6.tools.schema import DagAddTaskInput, DagUpdateTaskInput
from agent6.workflows import loop as loopmod

_VALID_TOML = """
[agent6]
config_version = 1
[providers.anthropic]
api_format = "anthropic"
api_key_env = "ANTHROPIC_API_KEY"
[models.worker]
provider = "anthropic"
model = "x"
[models.reviewer]
provider = "anthropic"
model = "x"
[workflow]
verify_command = ["true"]
"""

_B = "01" + "B" * 24


def _config(tmp_path: Path) -> Config:
    p = tmp_path / "agent6.toml"
    p.write_text(_VALID_TOML, encoding="utf-8")
    return load_config(p)


def _curator(tmp_path: Path) -> GraphCurator:
    return GraphCurator(SessionLayout(state_dir=tmp_path / ".agent6", session_id="run1"))


def test_no_separate_dependency_tool_and_both_carriers_expose_depends_on(
    tmp_path: Path,
) -> None:
    """The folded surface: no mode lists an `add_dependency` tool, and the two
    carriers' schemas expose `depends_on` instead."""
    d = ToolDispatcher(root=tmp_path, config=_config(tmp_path))
    for mode in ("run", "plan", "ask", "machine", "agent"):
        names = {t.name for t in loopmod.tool_definitions(d, mode=mode)}  # pyright: ignore[reportPrivateUsage]
        assert "add_dependency" not in names, mode
    assert "depends_on" in DagAddTaskInput.model_json_schema()["properties"]
    assert "depends_on" in DagUpdateTaskInput.model_json_schema()["properties"]


def test_add_task_carries_depends_on_to_the_node(tmp_path: Path) -> None:
    cur = _curator(tmp_path)
    root = cur.add_subtask(
        AddSubtaskIntent(parent_id=None, draft=TaskNodeDraft(title="root", created_by="planner"))
    )
    d = ToolDispatcher(root=tmp_path, config=_config(tmp_path), curator=cur)
    d.set_run_root_node_id(root.id)
    a = d.dispatch("add_task", {"title": "first"}).to_wire()
    b = d.dispatch("add_task", {"title": "second", "depends_on": [a["id"]]}).to_wire()
    assert list(cur.nodes()[b["id"]].depends_on) == [a["id"]]


def test_update_task_appends_edges_without_a_status(tmp_path: Path) -> None:
    cur = _curator(tmp_path)
    root = cur.add_subtask(
        AddSubtaskIntent(parent_id=None, draft=TaskNodeDraft(title="root", created_by="planner"))
    )
    d = ToolDispatcher(root=tmp_path, config=_config(tmp_path), curator=cur)
    d.set_run_root_node_id(root.id)
    a = d.dispatch("add_task", {"title": "first"}).to_wire()
    b = d.dispatch("add_task", {"title": "second"}).to_wire()
    out = d.dispatch("update_task", {"id": b["id"], "depends_on": [a["id"]]}).to_wire()
    # Status untouched, the edge landed, and the result names it.
    assert out == {
        "id": b["id"],
        "status": "pending",
        "title": "second",
        "depends_on": [a["id"]],
    }
    json.dumps(out)  # the loop JSONs the result for the model; must not raise


def test_update_task_with_neither_status_nor_edges_refuses(tmp_path: Path) -> None:
    cur = _curator(tmp_path)
    root = cur.add_subtask(
        AddSubtaskIntent(parent_id=None, draft=TaskNodeDraft(title="root", created_by="planner"))
    )
    d = ToolDispatcher(root=tmp_path, config=_config(tmp_path), curator=cur)
    d.set_run_root_node_id(root.id)
    a = d.dispatch("add_task", {"title": "first"}).to_wire()
    with pytest.raises(ToolError, match="status and/or depends_on"):
        d.dispatch("update_task", {"id": a["id"]})


def test_update_task_surfaces_cycle_rejection(tmp_path: Path) -> None:
    cur = _curator(tmp_path)
    root = cur.add_subtask(
        AddSubtaskIntent(parent_id=None, draft=TaskNodeDraft(title="root", created_by="planner"))
    )
    d = ToolDispatcher(root=tmp_path, config=_config(tmp_path), curator=cur)
    d.set_run_root_node_id(root.id)
    a = d.dispatch("add_task", {"title": "first"}).to_wire()
    b = d.dispatch("add_task", {"title": "second", "depends_on": [a["id"]]}).to_wire()
    with pytest.raises(ToolError, match="cycle"):
        d.dispatch("update_task", {"id": a["id"], "depends_on": [b["id"]]})


def test_depends_on_ids_validate_at_the_schema(tmp_path: Path) -> None:
    cur = _curator(tmp_path)
    d = ToolDispatcher(root=tmp_path, config=_config(tmp_path), curator=cur)
    with pytest.raises(ToolError):
        d.dispatch("update_task", {"id": _B, "depends_on": ["short"]})
    assert not cur.nodes()  # rejected at the schema, never reached the curator


def test_status_and_edges_apply_together(tmp_path: Path) -> None:
    cur = _curator(tmp_path)
    root = cur.add_subtask(
        AddSubtaskIntent(parent_id=None, draft=TaskNodeDraft(title="root", created_by="planner"))
    )
    d = ToolDispatcher(root=tmp_path, config=_config(tmp_path), curator=cur)
    d.set_run_root_node_id(root.id)
    a = d.dispatch("add_task", {"title": "first"}).to_wire()
    b = d.dispatch("add_task", {"title": "second"}).to_wire()
    out = d.dispatch(
        "update_task", {"id": b["id"], "status": "in_progress", "depends_on": [a["id"]]}
    ).to_wire()
    assert out["status"] == "in_progress" and out["depends_on"] == [a["id"]]


def test_list_tasks_wire_shape_is_stable(tmp_path: Path) -> None:
    """FROZEN wire surface: the list_tasks result dict is JSON'd verbatim to the
    model. Each task projects to exactly {id, parent_id, title, status,
    acceptance, relevant_paths, depends_on} with the sequence fields as JSON
    lists (not tuples), under a top-level {tasks, count}. Interface-independent:
    drives a real curator + real dispatcher, so it pins the returned shape
    regardless of how the curator hands state to the tool internally."""
    cur = _curator(tmp_path)
    root = cur.add_subtask(
        AddSubtaskIntent(parent_id=None, draft=TaskNodeDraft(title="root", created_by="planner"))
    )
    a = cur.add_subtask(
        AddSubtaskIntent(
            parent_id=root.id,
            draft=TaskNodeDraft(
                title="review providers",
                acceptance="no bugs left",
                relevant_paths=("a.py",),
                created_by="worker",
            ),
        )
    )
    b = cur.add_subtask(
        AddSubtaskIntent(
            parent_id=root.id,
            draft=TaskNodeDraft(title="review sandbox", depends_on=(a.id,), created_by="worker"),
        )
    )
    cur.update_status(UpdateStatusIntent(id=a.id, new_status="in_progress"))

    d = ToolDispatcher(root=tmp_path, config=_config(tmp_path), curator=cur)
    out = d.dispatch("list_tasks", {}).to_wire()
    # Exact equality also pins list-vs-tuple: ("a.py",) != ["a.py"]. `standing`
    # rides along because the finish gate excludes those: without it the model
    # read three open tasks while the gate counted one, with no way to tell
    # which.
    assert out == {
        "tasks": [
            {
                "id": root.id,
                "parent_id": None,
                "title": "root",
                "status": "pending",
                "acceptance": "",
                "relevant_paths": [],
                "depends_on": [],
                "standing": False,
            },
            {
                "id": a.id,
                "parent_id": root.id,
                "title": "review providers",
                "status": "in_progress",
                "acceptance": "no bugs left",
                "relevant_paths": ["a.py"],
                "depends_on": [],
                "standing": False,
            },
            {
                "id": b.id,
                "parent_id": root.id,
                "title": "review sandbox",
                "status": "pending",
                "acceptance": "",
                "relevant_paths": [],
                "depends_on": [a.id],
                "standing": False,
            },
        ],
        "count": 3,
    }
    json.dumps(out)  # the loop JSONs the result for the model; must not raise

    # The status filter narrows tasks and count together.
    filtered = d.dispatch("list_tasks", {"status": "in_progress"}).to_wire()
    assert filtered["count"] == 1
    assert [t["id"] for t in filtered["tasks"]] == [a.id]


def test_dag_prompt_blocks_teach_depends_on_not_a_tool() -> None:
    from agent6.prompts.loop import DAG_RULES_DECOMPOSE, DAG_RULES_OPTIONAL

    for block in (DAG_RULES_OPTIONAL, DAG_RULES_DECOMPOSE):
        assert "depends_on" in block
        assert "add_dependency" not in block
