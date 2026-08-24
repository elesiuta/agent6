# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The dashboard's step selector time-travels the details: the task tree and
the cost line show the state as of the selected commit."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
from textual.widgets import Static, Tree

from agent6.ui.tui.app import Agent6TUI, DashboardScreen


def _mk(d: Path) -> None:
    d.mkdir(parents=True)
    events = [
        {"type": "session.start", "session_id": d.name, "mode": "run", "user_task": "t"},
        {"type": "loop.auto_commit", "iteration": 1, "sha": "a" * 40, "subject": "one"},
        {"type": "loop.auto_commit", "iteration": 2, "sha": "b" * 40, "subject": "two"},
        {"type": "session.end", "reason": "finish_session", "all_passed": True},
    ]
    (d / "logs.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events), encoding="utf-8")


def _no_patch(self: DashboardScreen, sha: str) -> str:
    return "(no diff)"  # the diff pane's git read is not under test


def test_a_selected_step_relabels_the_details_as_of_that_iteration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    d = tmp_path / "s1"
    _mk(d)
    monkeypatch.setattr(DashboardScreen, "_step_patch", _no_patch)

    async def scenario() -> None:
        app = Agent6TUI(d)
        async with app.run_test(size=(140, 40)) as pilot:
            await pilot.pause()
            dash = app._dash  # pyright: ignore[reportPrivateUsage]
            dash._step_sel = "a" * 40  # pyright: ignore[reportPrivateUsage]
            dash.render_state()
            await pilot.pause()
            assert dash.query_one("#plan", Tree).border_title == "tasks · as of iter 1"
            assert "as of iter 1" in str(dash.query_one("#top", Static).render())
            dash._step_sel = ""  # pyright: ignore[reportPrivateUsage]
            dash.render_state()
            await pilot.pause()
            assert dash.query_one("#plan", Tree).border_title == ""

    asyncio.run(scenario())


def test_the_top_line_counts_tasks_as_of_the_selected_step(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With a step selected, `tasks:` follows that step like `cost:` does, and
    the as-of marker sits after both; only `ctx:` is live."""
    d = tmp_path / "s2"
    d.mkdir(parents=True)
    events = [
        {"type": "session.start", "session_id": d.name, "mode": "run", "user_task": "t"},
        {
            "type": "graph.update",
            "cursor": "t1",
            "nodes": {"t1": {"title": "first", "parent_id": None, "status": "pending"}},
        },
        {"type": "loop.auto_commit", "iteration": 1, "sha": "a" * 40, "subject": "one"},
        {
            "type": "graph.update",
            "cursor": "t2",
            "nodes": {
                "t1": {"title": "first", "parent_id": None, "status": "passed"},
                "t2": {"title": "second", "parent_id": None, "status": "pending"},
            },
        },
        {"type": "loop.auto_commit", "iteration": 2, "sha": "b" * 40, "subject": "two"},
        {"type": "session.end", "reason": "finish_session", "all_passed": True},
    ]
    (d / "logs.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events), encoding="utf-8")
    monkeypatch.setattr(DashboardScreen, "_step_patch", _no_patch)

    async def scenario() -> None:
        app = Agent6TUI(d)
        async with app.run_test(size=(140, 40)) as pilot:
            await pilot.pause()
            dash = app._dash  # pyright: ignore[reportPrivateUsage]
            dash.render_state()
            await pilot.pause()
            live = str(dash.query_one("#top", Static).render())
            assert "tasks: 1/2" in live and "as of iter" not in live
            dash._step_sel = "a" * 40  # pyright: ignore[reportPrivateUsage]
            dash.render_state()
            await pilot.pause()
            top = str(dash.query_one("#top", Static).render())
            assert "tasks: 0/1" in top
            assert top.index("cost:") < top.index("as of iter 1")

    asyncio.run(scenario())


def test_the_diff_pane_keeps_saying_the_model_owns_git(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Under `[git].control = "model"` the diff pane's title survives every
    paint, a selected task included."""
    d = tmp_path / "s3"
    _mk(d)
    monkeypatch.setattr(DashboardScreen, "_step_patch", _no_patch)

    def model_owns_git(_self: DashboardScreen) -> str:
        return "model"

    monkeypatch.setattr(DashboardScreen, "_git_control", model_owns_git)

    async def scenario() -> None:
        app = Agent6TUI(d)
        async with app.run_test(size=(140, 40)) as pilot:
            await pilot.pause()
            dash = app._dash  # pyright: ignore[reportPrivateUsage]
            dash.render_state()
            await pilot.pause()
            assert dash.query_one("#diff").border_title == "diff · the model owns git"
            dash._selected_task_id = "t1"  # pyright: ignore[reportPrivateUsage]
            dash.render_state()
            await pilot.pause()
            assert dash.query_one("#diff").border_title == "diff · the model owns git"

    asyncio.run(scenario())
