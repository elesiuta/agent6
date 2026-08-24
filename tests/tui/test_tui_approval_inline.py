# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""An approval on the conversation screen is an inline item plus a docked key
row, never a modal: the conversation stays scrollable and readable while the
command under judgment sits at its tail, one key answers, and the item
collapses to a dim line once answered."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

from textual.widgets import Static

from agent6.ui.tui.app import Agent6TUI
from agent6.ui.tui.conversation import ApprovalRow
from agent6.ui.tui.modals import ApprovalModal


def _live_run(d: Path) -> None:
    d.mkdir(parents=True, exist_ok=True)
    (d / "approvals").mkdir(exist_ok=True)
    (d / "worker.pid").write_text(str(os.getpid()), encoding="utf-8")
    evs = [
        {"type": "session.start", "session_id": d.name, "mode": "run", "user_task": "add it"},
        {"type": "role.call", "role": "worker", "model": "m"},
    ]
    (d / "logs.jsonl").write_text("".join(json.dumps(e) + "\n" for e in evs), encoding="utf-8")


def _append(d: Path, ev: dict[str, object]) -> None:
    with (d / "logs.jsonl").open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(ev) + "\n")


def test_an_approval_is_an_inline_item_with_a_key_row(tmp_path: Path) -> None:
    run = tmp_path / "live-run-AAAAAA"
    _live_run(run)

    async def scenario() -> None:
        app = Agent6TUI(run)
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            await pilot.pause()
            _append(
                run,
                {
                    "type": "approval.prompt",
                    "id": "ap1",
                    "prompt": "Allow run_command: pytest -q tests/unit/test_x.py",
                    "standing": True,
                },
            )
            app._conv._poll()  # pyright: ignore[reportPrivateUsage]
            await pilot.pause()
            await pilot.pause()
            assert not isinstance(app.screen, ApprovalModal)
            item = app._conv.query_one("#conv-approval", Static)  # pyright: ignore[reportPrivateUsage]
            assert item.display
            text = str(item.render())
            assert "approval needed" in text and "pytest -q tests/unit/test_x.py" in text
            row = app._conv.query_one(ApprovalRow)  # pyright: ignore[reportPrivateUsage]
            assert app.focused is row
            await pilot.press("a")
            await pilot.pause()
            assert (run / "approvals" / "ap1.answer").read_text(encoding="utf-8") == "yes"
            _append(run, {"type": "approval.answer", "id": "ap1", "approved": True})
            app._conv._poll()  # pyright: ignore[reportPrivateUsage]
            await pilot.pause()
            await pilot.pause(0.3)
            app._tick()  # pyright: ignore[reportPrivateUsage]
            app._conv._poll()  # pyright: ignore[reportPrivateUsage]
            await pilot.pause()
            assert not app._conv.query(ApprovalRow)  # pyright: ignore[reportPrivateUsage]
            assert "allowed" in str(item.render())

    asyncio.run(scenario())
