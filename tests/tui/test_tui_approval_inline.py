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
from typing import Any

from textual.widgets import Static

from agent6.ui.tui.app import Agent6TUI
from agent6.ui.tui.composer import ApprovalRow, SteerInput
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
            assert app._conv.query(ApprovalRow)  # pyright: ignore[reportPrivateUsage]
            # The composer keeps focus: an empty one lets the row's keys answer.
            assert app.focused is app._conv.query_one("#conv-input", SteerInput)  # pyright: ignore[reportPrivateUsage]
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


async def _open_approval(app: Agent6TUI, pilot: Any, run: Path) -> None:
    await pilot.pause()
    await pilot.pause()
    prompt = {"type": "approval.prompt", "id": "ap1", "prompt": "Allow run_command: ls"}
    _append(run, {**prompt, "standing": True})
    app._conv._poll()  # pyright: ignore[reportPrivateUsage]
    await pilot.pause()
    await pilot.pause()
    assert app._conv.query(ApprovalRow)  # pyright: ignore[reportPrivateUsage]


def test_a_typed_message_never_answers_the_approval(tmp_path: Path) -> None:
    """The row took focus, so a sentence typed at the composer answered the
    approval on its first `s`. The composer keeps focus and the keys fire only
    while it is empty: the text lands, nothing is granted."""
    run = tmp_path / "live-run-CCCCCC"
    _live_run(run)

    async def scenario() -> None:
        app = Agent6TUI(run)
        async with app.run_test(size=(120, 40)) as pilot:
            await _open_approval(app, pilot, run)
            await pilot.press("slash", "b", "t", "w", "space", "s", "u", "r", "e")
            await pilot.pause()
            assert not (run / "approvals" / "ap1.answer").exists()
            bar = app._conv.query_one("#conv-input", SteerInput)  # pyright: ignore[reportPrivateUsage]
            assert bar.text == "/btw sure"
            assert app._conv.query(ApprovalRow)  # pyright: ignore[reportPrivateUsage]

    asyncio.run(scenario())


def test_a_click_on_a_row_label_answers(tmp_path: Path) -> None:
    run = tmp_path / "live-run-DDDDDD"
    _live_run(run)

    async def scenario() -> None:
        app = Agent6TUI(run)
        async with app.run_test(size=(120, 40)) as pilot:
            await _open_approval(app, pilot, run)
            label = app._conv.query_one(".answer-yes", Static)  # pyright: ignore[reportPrivateUsage]
            await pilot.click(label)
            await pilot.pause()
            assert (run / "approvals" / "ap1.answer").read_text(encoding="utf-8") == "yes"

    asyncio.run(scenario())


def test_a_dead_runs_approval_is_shown_but_not_answerable(tmp_path: Path) -> None:
    """A run killed with its prompt open: the fact stays on the surface, the
    key row (whose answer would reach nothing) is not offered."""
    run = tmp_path / "dead-run-AAAAAA"
    _live_run(run)
    (run / "worker.pid").write_text("4194304", encoding="utf-8")  # past pid_max: gone
    _append(
        run,
        {"type": "approval.prompt", "id": "ap1", "prompt": "Allow run_command: rm -rf build"},
    )

    async def scenario() -> None:
        app = Agent6TUI(run)
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            await pilot.pause()
            app._conv._poll()  # pyright: ignore[reportPrivateUsage]
            await pilot.pause()
            assert not app.session_controllable()
            item = app._conv.query_one("#conv-approval", Static)  # pyright: ignore[reportPrivateUsage]
            assert item.display
            text = str(item.render())
            assert "approval pending when the run ended" in text and "rm -rf build" in text
            assert not app._conv.query(ApprovalRow)  # pyright: ignore[reportPrivateUsage]

    asyncio.run(scenario())


def test_escape_with_a_menu_open_closes_the_menu_not_the_view(tmp_path: Path) -> None:
    run = tmp_path / "live-run-BBBBBB"
    _live_run(run)

    async def scenario() -> None:
        from agent6.ui.tui.menubar import MenuBar

        app = Agent6TUI(run)
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            bar = app._conv.query_one(MenuBar)  # pyright: ignore[reportPrivateUsage]
            bar.open("r")
            await pilot.pause()
            assert bar.opened
            await pilot.press("escape")
            await pilot.pause()
            assert not bar.opened
            assert app.is_running and app.screen is app._conv  # pyright: ignore[reportPrivateUsage]

    asyncio.run(scenario())


def test_a_non_standing_approvals_session_keys_type_the_letter(tmp_path: Path) -> None:
    """An approval nobody may answer for the session offers no `s`/`x`: the
    key is the letter it is, typed into the composer."""
    run = tmp_path / "live-run-EEEEEE"
    _live_run(run)

    async def scenario() -> None:
        app = Agent6TUI(run)
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            await pilot.pause()
            prompt = {"type": "approval.prompt", "id": "ap1", "prompt": "Allow fetch: x.io"}
            _append(run, {**prompt, "standing": False})
            app._conv._poll()  # pyright: ignore[reportPrivateUsage]
            await pilot.pause()
            await pilot.pause()
            assert app._conv.query(ApprovalRow)  # pyright: ignore[reportPrivateUsage]
            await pilot.press("s", "x")
            await pilot.pause()
            assert not (run / "approvals" / "ap1.answer").exists()
            bar = app._conv.query_one("#conv-input", SteerInput)  # pyright: ignore[reportPrivateUsage]
            assert bar.text == "sx"

    asyncio.run(scenario())


def test_a_key_off_the_composer_answers_nothing(tmp_path: Path) -> None:
    """The answer keys are the composer's: with focus elsewhere (the
    scrollback), a key neither answers nor types; the label's click does."""
    run = tmp_path / "live-run-FFFFFF"
    _live_run(run)

    async def scenario() -> None:
        app = Agent6TUI(run)
        async with app.run_test(size=(120, 40)) as pilot:
            await _open_approval(app, pilot, run)
            app._conv.query_one("#conv-scroll").focus()  # pyright: ignore[reportPrivateUsage]
            await pilot.pause()
            await pilot.press("a")
            await pilot.pause()
            assert not (run / "approvals" / "ap1.answer").exists()
            bar = app._conv.query_one("#conv-input", SteerInput)  # pyright: ignore[reportPrivateUsage]
            assert bar.text == ""
            assert app._conv.query(ApprovalRow)  # pyright: ignore[reportPrivateUsage]

    asyncio.run(scenario())
