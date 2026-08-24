# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Headless drive of the scrollable run-log viewer (LogScreen)."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from textual.app import App
from textual.widgets import Static

from agent6.ui.tui.logview import LogScreen
from agent6.viewmodel.log_line import format_log_line


class _Host(App[None]):
    def __init__(self, logs_path: Path) -> None:
        super().__init__()
        self._logs = logs_path

    def on_mount(self) -> None:
        self.push_screen(LogScreen(self._logs, title=lambda: "logs · test"))


def _write_log(path: Path, events: list[dict[str, object]]) -> None:
    path.write_text("\n".join(json.dumps(e) for e in events) + "\n", encoding="utf-8")


def _lines(screen: LogScreen) -> list[str]:
    # The body is a Static (selectable, unlike RichLog); its text is the renderable.
    body = screen.query_one("#logview-body", Static)
    return [ln for ln in str(body.content).splitlines() if ln.strip()]


def test_logscreen_renders_structural_events(tmp_path: Path) -> None:
    logs = tmp_path / "logs.jsonl"
    events: list[dict[str, object]] = [
        {
            "type": "session.start",
            "mode": "run",
            "user_task": "do x",
            "ts": "2026-06-22T01:02:03.4Z",
        },
        {"type": "tool.call", "name": "read_file", "args": {"path": "a.py"}, "ts": "t"},
        {"type": "verify.end", "exit_code": 0, "duration_s": 1.2, "ts": "t"},
        {"type": "session.end", "all_passed": True, "ts": "t"},
    ]
    _write_log(logs, events)

    async def scenario() -> None:
        app = _Host(logs)
        async with app.run_test() as pilot:
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, LogScreen)
            assert len(_lines(screen)) == len(events)  # one line per structural event

    asyncio.run(scenario())


def test_logscreen_skips_streaming_deltas(tmp_path: Path) -> None:
    # A reasoning model emits thousands of role.thinking_delta events; they are
    # live-stream noise, not audit-log lines, so LogScreen must not render them.
    logs = tmp_path / "logs.jsonl"
    _write_log(
        logs,
        [
            {"type": "role.call", "role": "worker", "ts": "t"},
            {"type": "role.thinking_delta", "text": "hmm", "ts": "t"},
            {"type": "role.thinking_delta", "text": " let me", "ts": "t"},
            {"type": "role.text_delta", "text": "ok", "ts": "t"},
            {"type": "role.result", "ts": "t"},
        ],
    )

    async def scenario() -> None:
        app = _Host(logs)
        async with app.run_test() as pilot:
            await pilot.pause()
            lines = _lines(app.screen)  # type: ignore[arg-type]
            assert len(lines) == 2  # role.call + role.result; the 3 deltas are dropped
            assert not any("delta" in ln for ln in lines)

    asyncio.run(scenario())


def test_logscreen_skips_the_loop_mirrors_like_the_dashboard_tail(tmp_path: Path) -> None:
    """loop.tool.call / loop.budget mirror events already rendered (the tool
    call carries the args, budget.update the totals) and format to an empty
    detail; the dashboard's log tail drops them, so the full log view does too."""
    logs = tmp_path / "logs.jsonl"
    _write_log(
        logs,
        [
            {"type": "loop.tool.call", "iteration": 1, "ts": "t"},
            {"type": "tool.call", "name": "read_file", "args": {"path": "a"}, "ts": "t"},
            {"type": "loop.budget", "iteration": 1, "ts": "t"},
            {"type": "budget.update", "input_total": 1, "output_total": 2, "ts": "t"},
        ],
    )

    async def scenario() -> None:
        app = _Host(logs)
        async with app.run_test() as pilot:
            await pilot.pause()
            lines = _lines(app.screen)  # type: ignore[arg-type]
            assert len(lines) == 2
            assert not any("loop." in ln for ln in lines)

    asyncio.run(scenario())


def test_logscreen_reload_picks_up_appended_lines(tmp_path: Path) -> None:
    logs = tmp_path / "logs.jsonl"
    _write_log(logs, [{"type": "session.start", "mode": "run", "user_task": "x", "ts": "t"}])

    async def scenario() -> None:
        app = _Host(logs)
        async with app.run_test() as pilot:
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, LogScreen)
            assert len(_lines(screen)) == 1
            # A live run keeps appending; reload pulls the new lines in.
            with logs.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps({"type": "session.end", "all_passed": True, "ts": "t"}) + "\n")
            screen.action_reload()
            await pilot.pause()
            assert len(_lines(screen)) == 2

    asyncio.run(scenario())


def test_logscreen_empty_log(tmp_path: Path) -> None:
    logs = tmp_path / "logs.jsonl"  # does not exist

    async def scenario() -> None:
        app = _Host(logs)
        async with app.run_test() as pilot:
            await pilot.pause()
            body = app.screen.query_one("#logview-body", Static)
            assert "no events yet" in str(body.content)

    asyncio.run(scenario())


def test_format_log_line_is_public_and_compact() -> None:
    line = format_log_line({"type": "tool.call", "name": "grep", "args": {}, "ts": "t"})
    assert "tool.call" in line and "grep" in line
