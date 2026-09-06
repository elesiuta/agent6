# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Tests for `GraphCurator` mutations."""

from __future__ import annotations

from pathlib import Path

import pytest

from agent6.graph.curator import CuratorError, GraphCurator
from agent6.graph.models import (
    AddDependencyIntent,
    AddSubtaskIntent,
    RecordCommitIntent,
    SetCursorIntent,
    TaskNodeDraft,
    UpdateStatusIntent,
)
from agent6.sessions.layout import SessionLayout


def _layout(tmp_path: Path) -> SessionLayout:
    return SessionLayout(state_dir=tmp_path / ".agent6", session_id="run1")


def _draft(title: str = "do thing", deps: tuple[str, ...] = ()) -> TaskNodeDraft:
    return TaskNodeDraft(title=title, depends_on=deps, created_by="planner")


def test_curator_startup_tolerates_torn_journal_line(tmp_path: Path) -> None:
    # Build a real graph, then simulate a crash mid-append by tacking a torn
    # (invalid JSON) line onto graph.jsonl. Curator startup must NOT crash --
    # otherwise the run is permanently unresumable.
    layout = _layout(tmp_path)
    c = GraphCurator(layout)
    c.add_subtask(AddSubtaskIntent(parent_id=None, draft=_draft("root")))
    with layout.journal_path.open("a", encoding="utf-8") as fh:
        fh.write('{"op": "add_subtask", "graph_v')  # torn: no newline, invalid JSON
    reopened = GraphCurator(layout)  # must not raise
    assert reopened.graph_version >= 1
    assert len(reopened.nodes()) == 1


def test_add_subtask_with_no_parent_creates_root(tmp_path: Path) -> None:
    c = GraphCurator(_layout(tmp_path))
    n = c.add_subtask(AddSubtaskIntent(parent_id=None, draft=_draft("root")))
    assert n.parent_id is None
    assert n.status == "pending"
    assert c.graph_version >= 1


def test_add_subtask_unknown_parent_raises(tmp_path: Path) -> None:
    c = GraphCurator(_layout(tmp_path))
    with pytest.raises(CuratorError, match="unknown parent"):
        c.add_subtask(AddSubtaskIntent(parent_id="X" * 26, draft=_draft()))


def test_add_subtask_links_child_to_parent(tmp_path: Path) -> None:
    c = GraphCurator(_layout(tmp_path))
    p = c.add_subtask(AddSubtaskIntent(parent_id=None, draft=_draft("parent")))
    ch = c.add_subtask(AddSubtaskIntent(parent_id=p.id, draft=_draft("child")))
    assert c.get(p.id).children == (ch.id,)


def test_update_status_passed_then_obsolete_ok_other_rejected(tmp_path: Path) -> None:
    c = GraphCurator(_layout(tmp_path))
    n = c.add_subtask(AddSubtaskIntent(parent_id=None, draft=_draft()))
    c.update_status(UpdateStatusIntent(id=n.id, new_status="passed"))
    # passed -> obsolete is fine
    c.update_status(UpdateStatusIntent(id=n.id, new_status="obsolete"))
    # but passed -> anything else would be rejected, set it back first
    n2 = c.add_subtask(AddSubtaskIntent(parent_id=None, draft=_draft()))
    c.update_status(UpdateStatusIntent(id=n2.id, new_status="passed"))
    with pytest.raises(CuratorError):
        c.update_status(UpdateStatusIntent(id=n2.id, new_status="failed"))


def test_a_retired_task_stays_retired(tmp_path: Path) -> None:
    """`passed -> obsolete -> pending` walked around the passed-node guard, and
    re-opened work every dependent had been told passed."""
    c = GraphCurator(_layout(tmp_path))
    n = c.add_subtask(AddSubtaskIntent(parent_id=None, draft=_draft()))
    c.update_status(UpdateStatusIntent(id=n.id, new_status="passed"))
    c.update_status(UpdateStatusIntent(id=n.id, new_status="obsolete"))
    with pytest.raises(CuratorError, match="stays retired"):
        c.update_status(UpdateStatusIntent(id=n.id, new_status="pending"))
    # A note on a retired task still lands: the status is unchanged.
    c.update_status(UpdateStatusIntent(id=n.id, new_status="obsolete", note="superseded"))
    assert "superseded" in c.get(n.id).notes

    skipped = c.add_subtask(AddSubtaskIntent(parent_id=None, draft=_draft()))
    c.update_status(UpdateStatusIntent(id=skipped.id, new_status="skipped"))
    with pytest.raises(CuratorError, match="stays retired"):
        c.update_status(UpdateStatusIntent(id=skipped.id, new_status="in_progress"))


def test_add_dependency_detects_cycle(tmp_path: Path) -> None:
    c = GraphCurator(_layout(tmp_path))
    a = c.add_subtask(AddSubtaskIntent(parent_id=None, draft=_draft("a")))
    b = c.add_subtask(AddSubtaskIntent(parent_id=None, draft=_draft("b")))
    c.add_dependency(AddDependencyIntent(id=b.id, depends_on=a.id))
    with pytest.raises(CuratorError, match="cycle"):
        c.add_dependency(AddDependencyIntent(id=a.id, depends_on=b.id))


def test_cycle_check_survives_dangling_depends_on(tmp_path: Path) -> None:
    # A node carrying a depends_on edge to an id absent from the loaded graph (a
    # partially-loaded/corrupt graph) must not crash the transitive cycle walk
    # with a KeyError; the missing target is simply treated as not-a-cycle.
    c = GraphCurator(_layout(tmp_path))
    a = c.add_subtask(AddSubtaskIntent(parent_id=None, draft=_draft("a")))
    b = c.add_subtask(AddSubtaskIntent(parent_id=None, draft=_draft("b")))
    # Inject a dangling depends_on on b (the public add_dependency would reject an
    # unknown target, so corrupt the in-memory node directly to model a bad load).
    c._nodes[b.id] = b.model_copy(update={"depends_on": ("ghost-id",)})  # pyright: ignore[reportPrivateUsage]
    # add_dependency(a -> b) walks b's deps (incl. the ghost); must not raise KeyError.
    updated = c.add_dependency(AddDependencyIntent(id=a.id, depends_on=b.id))
    assert b.id in updated.depends_on


def test_a_container_with_open_children_cannot_pass(tmp_path: Path) -> None:
    """A parent with open children is a container: the frontier surfaces its
    children instead, and every dependency ON it counts as satisfied once it
    passes -- so passing it skipped the tasks it stood for while the work they
    named went undone. The tier-2 check-off offers containers to the
    summariser, which is how a model came to mark one."""
    from agent6.graph.models import UpdateStatusIntent

    c = GraphCurator(_layout(tmp_path))
    root = c.add_subtask(AddSubtaskIntent(parent_id=None, draft=_draft("run root")))
    parent = c.add_subtask(AddSubtaskIntent(parent_id=root.id, draft=_draft("phase")))
    child = c.add_subtask(AddSubtaskIntent(parent_id=parent.id, draft=_draft("step")))

    with pytest.raises(CuratorError, match="open children"):
        c.update_status(UpdateStatusIntent(id=parent.id, new_status="passed"))

    c.update_status(UpdateStatusIntent(id=child.id, new_status="passed"))
    assert c.update_status(UpdateStatusIntent(id=parent.id, new_status="passed")).status == "passed"
    # The root is the whole job, not a unit of work: nothing depends on it, so
    # a run ending with a subtask left open still passes it.
    open_child = c.add_subtask(AddSubtaskIntent(parent_id=root.id, draft=_draft("later")))
    assert c.update_status(UpdateStatusIntent(id=root.id, new_status="passed")).status == "passed"
    assert c.get(open_child.id).status == "pending"


def test_retire_as_obsolete_and_record_commit(tmp_path: Path) -> None:
    c = GraphCurator(_layout(tmp_path))
    n = c.add_subtask(AddSubtaskIntent(parent_id=None, draft=_draft()))
    c.record_commit(RecordCommitIntent(id=n.id, sha="abcd1234"))
    c.update_status(UpdateStatusIntent(id=n.id, new_status="obsolete", note="user-canceled"))
    final = c.get(n.id)
    assert final.commit_sha == "abcd1234"
    assert final.status == "obsolete"
    assert "user-canceled" in final.notes


def test_set_cursor_persists_and_validates(tmp_path: Path) -> None:
    c = GraphCurator(_layout(tmp_path))
    n = c.add_subtask(AddSubtaskIntent(parent_id=None, draft=_draft()))
    c.set_cursor(SetCursorIntent(id=n.id))
    assert c.cursor() == n.id
    with pytest.raises(CuratorError):
        c.set_cursor(SetCursorIntent(id="Z" * 26))
    c.set_cursor(SetCursorIntent(id=None))
    assert c.cursor() is None


def test_curator_reload_preserves_state(tmp_path: Path) -> None:
    layout = _layout(tmp_path)
    c = GraphCurator(layout)
    n = c.add_subtask(AddSubtaskIntent(parent_id=None, draft=_draft("persist me")))
    c.update_status(UpdateStatusIntent(id=n.id, new_status="in_progress"))
    v_before = c.graph_version
    c2 = GraphCurator(layout)
    again = c2.get(n.id)
    assert again.status == "in_progress"
    assert c2.graph_version == v_before


def test_journal_entry_shapes_are_pinned(tmp_path: Path) -> None:
    # The typed JournalEntry union owns the graph.jsonl audit shape; this pins
    # the per-op key sets/values against the pre-typed writer's format (old
    # journal dirs and the typed writer serialize identically).
    import json

    layout = _layout(tmp_path)
    c = GraphCurator(layout)
    root = c.add_subtask(AddSubtaskIntent(parent_id=None, draft=_draft("root")))
    a = c.add_subtask(AddSubtaskIntent(parent_id=root.id, draft=_draft("a")))
    b = c.add_subtask(AddSubtaskIntent(parent_id=root.id, draft=_draft("b")))
    c.update_status(UpdateStatusIntent(id=a.id, new_status="in_progress"))
    c.add_dependency(AddDependencyIntent(id=b.id, depends_on=a.id))
    c.record_commit(RecordCommitIntent(id=a.id, sha="abcd1234"))
    c.update_status(UpdateStatusIntent(id=b.id, new_status="obsolete", note="dropped"))
    c.set_cursor(SetCursorIntent(id=a.id))

    lines = [
        json.loads(raw)
        for raw in layout.journal_path.read_text(encoding="utf-8").splitlines()
        if raw.strip()
    ]
    for entry in lines:
        assert entry.pop("ts")  # storage stamps it; not part of the typed shape
    assert lines == [
        {
            "op": "add_subtask",
            "id": root.id,
            "parent_id": None,
            "by": "planner",
            "graph_version": 1,
        },
        {
            "op": "add_subtask",
            "id": a.id,
            "parent_id": root.id,
            "by": "planner",
            "graph_version": 2,
        },
        {
            "op": "add_subtask",
            "id": b.id,
            "parent_id": root.id,
            "by": "planner",
            "graph_version": 3,
        },
        {"op": "update_status", "id": a.id, "new_status": "in_progress", "graph_version": 4},
        {"op": "add_dependency", "id": b.id, "depends_on": a.id, "graph_version": 5},
        {"op": "record_commit", "id": a.id, "sha": "abcd1234", "graph_version": 6},
        {"op": "update_status", "id": b.id, "new_status": "obsolete", "graph_version": 7},
        {"op": "set_cursor", "id": a.id, "graph_version": 8},
    ]


def test_every_write_in_one_mutation_carries_the_journaled_version(tmp_path: Path) -> None:
    """add_subtask writes the child and relinks the parent; both node files
    carry the same graph_version the mutation's journal entry records, so a
    journal that lost its tail is detectable from the nodes alone."""
    import json as _json

    layout = _layout(tmp_path)
    c = GraphCurator(layout)
    root = c.add_subtask(AddSubtaskIntent(parent_id=None, draft=_draft("root")))
    child = c.add_subtask(AddSubtaskIntent(parent_id=root.id, draft=_draft("child")))
    assert root.graph_version == 1
    assert child.graph_version == 2
    nodes = c.nodes()
    assert nodes[root.id].graph_version == 2  # relinked by the child's mutation
    assert nodes[child.id].graph_version == 2
    entries = [
        _json.loads(line)
        for line in layout.journal_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert [e["graph_version"] for e in entries] == [1, 2]


def test_a_lost_journal_tail_resyncs_the_version_and_says_so(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A death between a node write and its journal append lost the entry;
    the reused version number then made two operations share one version,
    corrupting fork-at-version undo. Boot detects the newer node stamp,
    resyncs the counter past the lost number, and names the residual."""
    layout = _layout(tmp_path)
    c = GraphCurator(layout)
    root = c.add_subtask(AddSubtaskIntent(parent_id=None, draft=_draft("root")))
    c.add_subtask(AddSubtaskIntent(parent_id=root.id, draft=_draft("child")))
    # Simulate the crash: drop the journal's last line (the v2 entry).
    lines = layout.journal_path.read_text(encoding="utf-8").splitlines()
    layout.journal_path.write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")

    reopened = GraphCurator(layout)
    err = capsys.readouterr().err
    assert "lost its tail" in err and "v2" in err
    assert reopened.graph_version == 2
    third = reopened.add_subtask(AddSubtaskIntent(parent_id=root.id, draft=_draft("late")))
    assert third.graph_version == 3  # the lost number is never reused
