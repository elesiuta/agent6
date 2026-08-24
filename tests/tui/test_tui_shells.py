# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""`/shells` on the TUI composer: the roster every surface reads off disk, as a
read-only text view (the CLI pause menu prints the same lines)."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from textual.widgets import TextArea

from agent6.ui.tui.app import Agent6TUI
from agent6.ui.tui.modals import TextModal


def _mk(d: Path) -> None:
    d.mkdir(parents=True)
    evs = [
        {"type": "session.start", "session_id": d.name, "mode": "run", "user_task": "t"},
        {"type": "session.end", "reason": "finish_session", "all_passed": True},
    ]
    (d / "logs.jsonl").write_text("".join(json.dumps(e) + "\n" for e in evs), encoding="utf-8")
    shell = d / "shells" / "bg-1"
    shell.mkdir(parents=True)
    (shell / "meta.json").write_text(json.dumps({"command": "sleep 5"}), encoding="utf-8")
    (shell / "result.json").write_text(json.dumps({"returncode": 0}) + "\n", encoding="utf-8")


def test_shells_opens_the_roster_as_a_text_view(tmp_path: Path) -> None:
    d = tmp_path / "s1"
    _mk(d)

    async def scenario() -> None:
        app = Agent6TUI(d)
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            app.submit_instruction("/shells")
            await pilot.pause()
            assert isinstance(app.screen, TextModal)
            shown = app.screen.query_one("#text-view", TextArea).text
            assert "[bg-1] exited 0: sleep 5" in shown

    asyncio.run(scenario())
