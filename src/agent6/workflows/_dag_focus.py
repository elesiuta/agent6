# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""surface-current-task: the DAG focus frontier and its directives.

Pure helpers over the curator's nodes dict. The loop keeps a small/weak
worker on ONE task at a time: each turn it computes the current task -- the
curator cursor while it still points at a focusable subtask (the worker's
explicit choice wins), else the first dependency-satisfied open subtask in creation
order -- advances the cursor to it, and injects a focus banner when the focus
first appears, changes, or was wiped by a tier-2 restart. The banner survives
tier-1 elision, so the worker keeps seeing it between those events without
re-appending every turn. Only SUBTASKS are focus candidates, mirroring the
finish-gate: the always-pending auto-root is the whole job, not a unit of
work to surface.
"""

from __future__ import annotations

from agent6.graph.models import TaskNode
from agent6.graph.order import OPEN_STATUSES, has_open_child, tree_order

# Tool names that mutate the task DAG; after one runs the loop re-snapshots the
# graph (graph.update event) so a live viewer can render the worker's task
# breakdown.
DAG_MUTATING_TOOLS = frozenset({"add_task", "update_task"})

DEPS_SATISFIED_STATUSES = frozenset({"passed", "skipped", "obsolete"})

# Anti-grind: a weak model on a vague/oversized task can stay on one DAG task for
# many turns, reading without ever marking it done, decomposing it, or trying to
# finish -- so neither the finish-gate (fires on a finish attempt) nor went_quiet
# (it is busy) catches it. Every this-many consecutive turns on the SAME task with
# no forward motion (cursor advance / mark-done / decompose, any of which changes
# the focus and resets the count), fire a nudge offering split / pass / skip. It
# re-fires periodically (one nudge is easy to ignore) but caps at
# STUCK_NUDGE_MAX per task so it cannot nag forever; generous so a
# model making normal progress (which changes focus well before this) never sees it.
STUCK_ON_TASK_AFTER = 20
STUCK_NUDGE_MAX = 3


def ready_subtask(nodes: dict[str, TaskNode], node: TaskNode) -> bool:
    """An open SUBTASK whose dependencies are satisfied and whose children are
    all settled (a decomposed parent is not itself a unit of work)."""
    if node.parent_id is None or node.status not in OPEN_STATUSES:
        return False
    for dep in node.depends_on:
        d = nodes.get(dep)
        if d is None or d.status not in DEPS_SATISFIED_STATUSES:
            return False  # a missing or not-yet-done dependency blocks the subtask
    return not has_open_child(nodes, node)


def is_focusable_subtask(nodes: dict[str, TaskNode], node: TaskNode) -> bool:
    """A ready ORDINARY subtask. Standing tasks are excluded here: they are
    the fallback, selected only when nothing ordinary is ready."""
    return not node.standing and ready_subtask(nodes, node)


def first_ready_subtask(nodes: dict[str, TaskNode]) -> str | None:
    """First focusable subtask (open, deps satisfied, no open child), in the
    order the task tree shows: depth-first through each parent's `children`
    list. That list is what every renderer and `list_tasks` display, so a
    reordered or positionally-inserted child executes where it appears.

    Roots (and any node an ancestor does not reach, e.g. a stale parent
    reference) fall back to id order: ids are time-sortable ULIDs, so that is
    creation order even on a resumed run, where the nodes dict arrives in
    filesystem order.

    When nothing ordinary is ready, the first ready STANDING task is the
    fallback: ordinary pending work always outranks it, so a run drains real
    tasks first and returns to the standing goal when the queue empties.
    Returns None when nothing at all is ready."""
    for nid in tree_order(nodes):
        if is_focusable_subtask(nodes, nodes[nid]):
            return nid
    for nid in tree_order(nodes):
        node = nodes[nid]
        if node.standing and ready_subtask(nodes, node):
            return nid
    return None


def current_task_id(nodes: dict[str, TaskNode], cursor: str | None) -> str | None:
    """The subtask to focus on now: the curator cursor when it still points at a
    focusable subtask (a decomposed parent does NOT qualify -- its leaves do, so
    a split moves focus forward), else the first ready subtask. None when no
    subtask is focusable."""
    if cursor is not None:
        node = nodes.get(cursor)
        if node is not None and is_focusable_subtask(nodes, node):
            return cursor
    return first_ready_subtask(nodes)


def current_task_banner(task_id: str, node: TaskNode, *, decompose: bool = False) -> str:
    """The per-turn focus directive naming the current task and its acceptance."""
    title = node.title.strip() or "(untitled)"
    lines = [f"[harness focus] Current task ({task_id}): {title}"]
    acceptance = node.acceptance.strip()
    if acceptance:
        lines.append(f"Acceptance: {acceptance}")
    paths = node.relevant_paths
    if paths:
        lines.append("Relevant paths: " + ", ".join(paths[:8]))
    if node.standing:
        # A standing task never passes; the curator refuses `passed`, and the
        # operator's own goal (created_by "steering") refuses every retirement.
        retire = (
            "only the operator retires it"
            if node.created_by == "steering"
            else "retire it with update_task (skipped or obsolete) once it no longer applies"
        )
        lines.append(
            "This is a standing task: it never passes, so do not mark it passed;"
            f" {retire}. Work a round on it now, add_task each follow-up you find so"
            " it is worked in turn, and call finish_session when a round finds"
            " nothing left to do."
        )
    else:
        lines.append(
            "Work this ONE task to completion before anything else. When its"
            " acceptance is met, mark it passed with update_task -- you will then be"
            " moved to the next task. If you find unrelated work, add_task it"
            " instead of switching to it now."
        )
    # Decompose runs plan recursively: invite a finer plan for a task that turns
    # out large, at the point the model has the most context to plan it.
    if decompose and not node.children:
        lines.append(
            "If this task is itself large or multi-step, add child subtasks under"
            f" it (parent_id={task_id}) breaking it into finer steps, then do those."
        )
    return "\n".join(lines)


def stuck_on_task_nudge(task_id: str, node: TaskNode, turns: int) -> str:
    """The anti-grind directive: the model has spent `turns` turns on one task
    without concluding it; offer the three ways to record progress. Never
    for a standing task: it concludes nothing by design, and two of the three
    moves are refused on it."""
    title = node.title.strip() or "(untitled)"
    return (
        f"[harness] You have spent {turns} turns on the current task"
        f" ({task_id}: {title}) without concluding it. Pick ONE now and record it:\n"
        "- Too big? Split it into smaller ordered subtasks with add_task and work"
        " the first one.\n"
        "- Effectively done? Mark it passed with update_task.\n"
        "- Not needed? Mark it obsolete or skipped with update_task.\n"
        "Keep the task list in step with your progress rather than working on"
        " without updating it."
    )


def initial_dag_hint(root_id: str | None, mode: str, decompose: bool) -> str:
    """The DAG hint appended to the first user message, only for modes whose
    tool surface HAS the DAG tools (run, plan; see tools/schema.py): ask wires
    a curator too, but exposes no `add_task`, so a hint there names a tool the
    model cannot call. The decompose-first directive is RUN-MODE ONLY -- it
    references the run-only `<decompose-first>` system block and tells the
    worker to edit."""
    if root_id is None or mode not in ("run", "plan"):
        return ""
    if mode == "run" and decompose:
        return (
            "\n\nThe DAG-as-tool surface is wired (root task id"
            f" `{root_id}`). START by calling `add_task` several times to"
            " lay out your whole plan as ordered subtasks (see"
            " <decompose-first>), then work the first one. Do not edit"
            " before the plan exists."
        )
    return (
        "\n\nThe DAG-as-tool surface is wired. Root task id is"
        f" `{root_id}`. Use `add_task` to break this into trackable"
        " subtasks (or skip the DAG entirely - it's optional)."
    )
