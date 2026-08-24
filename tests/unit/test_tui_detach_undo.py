# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Ctrl-Z detaches the session TUI (the run keeps going); Ctrl-_ is text undo."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from textual.app import App, ComposeResult

from agent6.sessions.layout import LOGS_NAME
from agent6.ui.tui.app import Agent6TUI
from agent6.ui.tui.composer import SteerInput


def _session_dir(tmp_path: Path) -> Path:
    d = tmp_path / "run-x"
    d.mkdir()
    events = [
        {"type": "session.start", "user_task": "t", "ts": 1.0},
        {"type": "session.end", "reason": "finish_session", "all_passed": True, "ts": 2.0},
    ]
    (d / LOGS_NAME).write_text("".join(json.dumps(e) + "\n" for e in events), encoding="utf-8")
    return d


def test_ctrl_z_detaches_and_says_so(tmp_path: Path) -> None:
    """Ctrl-Z exits the app with `detached` set (run_tui prints the reattach
    hint off it), even while the composer TextArea -- whose built-in ctrl+z
    undo the app binding must out-prioritize -- holds focus."""
    app = Agent6TUI(_session_dir(tmp_path))

    async def drive() -> None:
        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.press("ctrl+z")
            await pilot.pause()

    asyncio.run(drive())
    assert app.detached is True


def test_ctrl_underscore_undoes_composer_typing(tmp_path: Path) -> None:
    class _Harness(App[None]):
        def compose(self) -> ComposeResult:
            yield SteerInput(id="conv-input")

    app = _Harness()

    async def drive() -> None:
        async with app.run_test() as pilot:
            await pilot.pause()
            box = app.query_one("#conv-input", SteerInput)
            box.focus()
            await pilot.pause()
            box.insert("hello")
            await pilot.pause()
            await pilot.press("ctrl+underscore")
            await pilot.pause()
            assert box.text == ""

    asyncio.run(drive())
