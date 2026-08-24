# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Ctrl-R searches this session's past messages into the composer, on both
run views, and does nothing loud when there is nothing to search."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from agent6.ui.tui.app import Agent6TUI
from agent6.ui.tui.composer import SteerInput
from agent6.ui.tui.modals import HistorySearchModal


def _run_dir(tmp_path: Path) -> Path:
    run = tmp_path / "run"
    run.mkdir()
    (run / "logs.jsonl").write_text(
        "".join(
            json.dumps(e) + "\n"
            for e in (
                {"type": "session.start", "mode": "run", "user_task": "polish the\nTUI"},
                {"type": "loop.steer.injected", "chars": 14, "text": "focus on tests"},
                {"type": "loop.steer.injected", "chars": 12, "text": "fix the docs"},
            )
        ),
        encoding="utf-8",
    )
    return run


def test_ctrl_r_fills_the_conversation_composer(tmp_path: Path) -> None:
    """Ctrl-R opens the picker (task included, newlines flattened); typing
    narrows, ↓ highlights, Enter puts the pick in the composer for editing --
    nothing is sent."""

    async def scenario() -> None:
        app = Agent6TUI(_run_dir(tmp_path))
        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.press("ctrl+r")
            await pilot.pause()
            assert isinstance(app.screen, HistorySearchModal)
            for c in "poli":
                await pilot.press(c)
            await pilot.press("down")  # highlight the one match
            await pilot.press("enter")
            await pilot.pause()
            field = app._conv.query_one("#conv-input", SteerInput)
            assert field.text == "polish the TUI"

    asyncio.run(scenario())


def test_ctrl_r_works_from_the_dashboard_and_esc_cancels(tmp_path: Path) -> None:
    async def scenario() -> None:
        app = Agent6TUI(_run_dir(tmp_path))
        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.press("ctrl+d")  # over to the dashboard view
            await pilot.pause()
            await pilot.press("ctrl+r")
            await pilot.pause()
            assert isinstance(app.screen, HistorySearchModal)
            await pilot.press("escape")
            await pilot.pause()
            assert not isinstance(app.screen, HistorySearchModal)
            assert app.screen.query_one("#dash-input", SteerInput).text == ""

    asyncio.run(scenario())


def test_ctrl_r_with_no_recorded_messages_opens_nothing(tmp_path: Path) -> None:
    run = tmp_path / "run"
    run.mkdir()
    (run / "logs.jsonl").write_text(
        json.dumps({"type": "tool.call", "name": "read_file", "args": {}}) + "\n",
        encoding="utf-8",
    )

    async def scenario() -> None:
        app = Agent6TUI(run)
        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.press("ctrl+r")
            await pilot.pause()
            assert not isinstance(app.screen, HistorySearchModal)

    asyncio.run(scenario())
