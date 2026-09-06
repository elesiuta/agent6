# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Tests for the `agent6 tui` hub helpers + the ask_user question bridge."""

from __future__ import annotations

import asyncio
import json
import os
import time
from pathlib import Path
from typing import Any

import pytest

from agent6.paths import state_dir
from agent6.sessions.ipc import (
    approvals_dir,
    clear_pending_answers,
    questions_dir,
    read_question_answers,
    register_frontend,
    write_answer,
    write_question_answers,
)
from agent6.viewmodel import session_dirs, session_mtime


def _write_run(
    agent6_dir: Path, sub: str, session_id: str, events: list[dict[str, object]]
) -> Path:
    rd = agent6_dir / "sessions" / sub / session_id
    rd.mkdir(parents=True)
    (rd / "logs.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events), encoding="utf-8")
    return rd


async def _wait_for(pilot: Any, cond: Any, what: str, timeout: float = 10.0) -> None:
    deadline = time.monotonic() + timeout
    while not cond():
        assert time.monotonic() < deadline, f"timed out waiting for {what}"
        await pilot.pause(0.05)


def test_list_runs_spans_runs_and_asks(tmp_path: Path) -> None:
    a6 = tmp_path / ".agent6"
    _write_run(a6, "runs", "r1", [{"type": "session.start", "mode": "run"}])
    _write_run(a6, "asks", "a1", [{"type": "session.start", "mode": "ask"}])
    names = {p.name for p in session_dirs(a6)}
    assert names == {"r1", "a1"}


def test_run_mtime_is_log_activity_not_dir_mtime(tmp_path: Path) -> None:
    """A run's listed/sorted time is its logs.jsonl mtime (last run activity), not
    the run-dir mtime. Opening a run writes a front-end claim into the dir, bumping the dir
    mtime; that must NOT move the run's 'when' or its sort position."""
    import os

    a6 = tmp_path / ".agent6"
    rd = _write_run(a6, "runs", "r1", [{"type": "session.start", "mode": "run"}])
    os.utime(rd / "logs.jsonl", (1000, 1000))  # last real activity
    # Simulate opening the dashboard: it writes a front-end claim, bumping the dir
    # mtime well past the log's. Pre-fix this became the displayed/sort time.
    register_frontend(rd, 123)
    os.utime(rd, (5000, 5000))
    assert session_mtime(rd) == 1000.0  # pyright: ignore[reportPrivateUsage]


def test_run_mtime_falls_back_to_dir_before_log_exists(tmp_path: Path) -> None:
    import os

    rd = tmp_path / "sessions" / "runs" / "fresh"
    rd.mkdir(parents=True)
    os.utime(rd, (2000, 2000))
    assert session_mtime(rd) == 2000.0  # pyright: ignore[reportPrivateUsage] - no log yet -> dir mtime


def test_question_bridge_round_trip(tmp_path: Path) -> None:
    # No front-end claim: consumption is claim-free (the answer's existence is
    # the proof); liveness only paces the wait for one that has yet to land.
    write_question_answers(tmp_path, "q1", ["use B"])
    assert read_question_answers(tmp_path, "q1", timeout_s=1.0) == ("use B",)


def test_read_question_answer_returns_none_when_no_tui(tmp_path: Path) -> None:
    # With no front-end claim the read gives up after dead_grace_s, NOT the
    # full timeout: a headless run must not sit out the whole answer window.
    start = time.monotonic()
    assert read_question_answers(tmp_path, "q1", timeout_s=10.0, dead_grace_s=0.05) is None
    assert time.monotonic() - start < 5.0, "the dead-front-end grace never broke the wait"


def test_read_question_answer_consumes_the_file(tmp_path: Path) -> None:
    # The answer file is unlinked after reading, so a later prompt with the same
    # id (counters reset on resume) can't re-read a stale answer.
    write_question_answers(tmp_path, "q1", ["first"])
    assert read_question_answers(tmp_path, "q1", timeout_s=1.0) == ("first",)
    assert not (questions_dir(tmp_path) / "q1.answer").exists()


def test_clear_pending_answers_wipes_stale_state(tmp_path: Path) -> None:
    write_answer(tmp_path, "approval-1", "yes")
    write_question_answers(tmp_path, "question-1", ["stale"])
    register_frontend(tmp_path, 12345)
    clear_pending_answers(tmp_path, before=time.time() + 60)
    assert not (approvals_dir(tmp_path) / "approval-1.answer").exists()
    assert not (questions_dir(tmp_path) / "question-1.answer").exists()


def test_refresh_keeps_runs_list_aligned_with_table_when_a_run_vanishes(tmp_path: Path) -> None:
    """A run dir that disappears between the listing and its stat() must be dropped
    from BOTH the table and self._runs. Otherwise the two desync and every
    cursor_row-indexed action (open/logs/merge) maps to the wrong run for rows
    past the gap."""
    import asyncio
    import shutil

    from textual.widgets import DataTable

    from agent6.ui.tui.home import Agent6HomeApp, HomeScreen

    a6 = tmp_path / ".agent6"
    for rid in ("r1", "r2", "r3"):
        _write_run(a6, "runs", rid, [{"type": "session.start", "mode": "run", "user_task": rid}])

    async def scenario() -> None:
        app = Agent6HomeApp(a6, tmp_path)
        async with app.run_test() as pilot:
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, HomeScreen)
            table = screen.query_one("#sessions", DataTable)
            assert table.row_count == 3
            # Delete the run currently shown in the MIDDLE row, then refresh.
            vanished = screen._runs[1]  # pyright: ignore[reportPrivateUsage]
            shutil.rmtree(vanished)
            screen.action_refresh()
            await pilot.pause()
            runs = screen._runs  # pyright: ignore[reportPrivateUsage]
            assert table.row_count == 2
            assert len(runs) == 2  # pre-fix this stayed 3 (the vanished run kept)
            assert vanished not in runs
            assert all(rd.exists() for rd in runs)  # every selectable row maps to a live run

    asyncio.run(scenario())


def test_home_app_lists_runs_and_opens_the_new_task_view(tmp_path: Path) -> None:
    import asyncio

    from agent6.ui.tui.home import Agent6HomeApp
    from agent6.ui.tui.new_work import NewWorkScreen

    a6 = tmp_path / ".agent6"
    _write_run(a6, "runs", "r1", [{"type": "session.start", "mode": "run", "user_task": "do [x]"}])

    async def scenario() -> None:
        app = Agent6HomeApp(a6, tmp_path)
        async with app.run_test() as pilot:
            from textual.widgets import DataTable

            from agent6.ui.tui.home import HomeScreen

            await pilot.pause()  # let on_mount push the HomeScreen
            assert isinstance(app.screen, HomeScreen)  # hub lives on its own screen
            table = app.screen.query_one("#sessions", DataTable)
            assert table.row_count == 1  # the one run is listed
            # 'n' opens the new-task view (an empty conversation); Esc backs out.
            await pilot.press("n")
            await pilot.pause()
            assert isinstance(app.screen, NewWorkScreen)
            await pilot.press("escape")
            await pilot.pause()
            assert isinstance(app.screen, HomeScreen)

    asyncio.run(scenario())


def test_new_task_view_starts_the_chosen_mode_and_preset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Enter in the draft composer starts `<mode>` under the picked preset (the
    same spawn every hub makes); a located session dir is the hub's return
    value. Ctrl-J is a newline, so a task can span lines."""
    import asyncio

    from textual.widgets import Select

    from agent6.ui.tui import new_work
    from agent6.ui.tui.composer import SteerInput
    from agent6.ui.tui.home import Agent6HomeApp
    from agent6.ui.tui.new_work import NewWorkScreen

    a6 = tmp_path / ".agent6"
    _write_run(a6, "runs", "r1", [{"type": "session.start", "mode": "run", "user_task": "x"}])
    started: list[tuple[str, str, str]] = []

    def _spawn(cwd: Path, mode: str, task: str, *, preset: str = "", config_path: object = None):
        started.append((mode, task, preset))
        return tmp_path / "located", ""

    monkeypatch.setattr(new_work, "spawn_new_work", _spawn)

    async def scenario() -> None:
        app = Agent6HomeApp(a6, tmp_path)
        async with app.run_test() as pilot:
            await pilot.pause()
            app.push_screen(NewWorkScreen(tmp_path, presets=["ultra"]))
            await pilot.pause()
            assert isinstance(app.screen, NewWorkScreen)
            bar = app.screen.query_one("#draft-input", SteerInput)
            assert bar.border_title == "new task"
            app.screen.query_one("#draft-mode", Select).value = "plan"
            app.screen.query_one("#draft-preset", Select).value = "ultra"
            bar.focus()
            await pilot.pause()
            await pilot.press("a", "ctrl+j", "b", "enter")
            deadline = time.monotonic() + 10
            while app.return_value is None and time.monotonic() < deadline:
                await pilot.pause(0.05)
            assert started == [("plan", "a\nb", "ultra")]
            assert app.return_value == tmp_path / "located"

    asyncio.run(scenario())


def test_a_start_whose_screen_was_left_still_opens_the_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The worker read `self.app` from its thread, where the screen it belongs
    to may already be popped: the located session then died in the thread and
    the hub never opened it. The worker reaches the app it was handed on the
    UI thread."""
    import asyncio
    import threading

    from agent6.ui.tui import new_work
    from agent6.ui.tui.composer import SteerInput
    from agent6.ui.tui.home import Agent6HomeApp
    from agent6.ui.tui.new_work import NewWorkScreen

    a6 = tmp_path / ".agent6"
    entered = threading.Event()
    gate = threading.Event()

    def _spawn(cwd: Path, mode: str, task: str, *, preset: str = "", config_path: object = None):
        entered.set()
        if not gate.wait(timeout=5.0):
            return None, "the test never released the spawn"
        return tmp_path / "located", ""

    monkeypatch.setattr(new_work, "spawn_new_work", _spawn)

    async def scenario() -> None:
        app = Agent6HomeApp(a6, tmp_path)
        async with app.run_test() as pilot:
            await pilot.pause()
            app.push_screen(NewWorkScreen(tmp_path))
            await pilot.pause()
            bar = app.screen.query_one("#draft-input", SteerInput)
            bar.focus()
            await pilot.pause()
            await pilot.press("f", "i", "x", "enter")
            await pilot.pause()
            assert entered.wait(2.0), "the spawn never ran"
            app.pop_screen()  # Esc while the start is still locating
            await pilot.pause()
            gate.set()
            deadline = time.monotonic() + 10
            while app.return_value is None and time.monotonic() < deadline:
                await pilot.pause(0.05)
            assert app.return_value == tmp_path / "located"

    asyncio.run(scenario())


def test_a_refusal_after_the_screen_was_left_still_reaches_the_operator(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The refusal branch wrote into the composer's notice widget, which a
    popped screen no longer has: the query raised inside the cancelled worker
    and the refusal reached nobody. Without a composer it toasts."""
    import asyncio
    import threading

    from agent6.ui.tui import new_work
    from agent6.ui.tui.composer import SteerInput
    from agent6.ui.tui.home import Agent6HomeApp
    from agent6.ui.tui.new_work import NewWorkScreen

    a6 = tmp_path / ".agent6"
    entered = threading.Event()
    gate = threading.Event()

    def _spawn(cwd: Path, mode: str, task: str, *, preset: str = "", config_path: object = None):
        entered.set()
        gate.wait(timeout=5.0)
        return None, "REFUSING: the tree is dirty"

    monkeypatch.setattr(new_work, "spawn_new_work", _spawn)

    async def scenario() -> None:
        app = Agent6HomeApp(a6, tmp_path)
        async with app.run_test() as pilot:
            await pilot.pause()
            app.push_screen(NewWorkScreen(tmp_path))
            await pilot.pause()
            app.screen.query_one("#draft-input", SteerInput).focus()
            await pilot.pause()
            await pilot.press("f", "i", "x", "enter")
            await pilot.pause()
            assert entered.wait(2.0), "the spawn never ran"
            app.pop_screen()  # Esc while the start is still locating
            await pilot.pause()
            gate.set()
            deadline = time.monotonic() + 10
            toasts: list[tuple[str, str]] = []
            while not toasts and time.monotonic() < deadline:
                await pilot.pause(0.05)
                toasts = [(str(n.message), n.severity) for n in app._notifications]  # pyright: ignore[reportPrivateUsage]
            assert ("REFUSING: the tree is dirty", "error") in toasts, toasts

    asyncio.run(scenario())


def test_new_task_view_keeps_the_text_on_a_refusal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A start the child refuses renders its reason where the transcript will
    be and hands the typed text back to the composer to fix and resend."""
    import asyncio

    from textual.widgets import Static

    from agent6.ui.tui import new_work
    from agent6.ui.tui.composer import SteerInput
    from agent6.ui.tui.home import Agent6HomeApp
    from agent6.ui.tui.new_work import NewWorkScreen

    a6 = tmp_path / ".agent6"

    def _spawn(cwd: Path, mode: str, task: str, *, preset: str = "", config_path: object = None):
        return None, "REFUSING: 1 tracked file has uncommitted changes:\n- [git] seed.txt"

    monkeypatch.setattr(new_work, "spawn_new_work", _spawn)

    async def scenario() -> None:
        app = Agent6HomeApp(a6, tmp_path)
        async with app.run_test() as pilot:
            await pilot.pause()
            app.push_screen(NewWorkScreen(tmp_path))
            await pilot.pause()
            bar = app.screen.query_one("#draft-input", SteerInput)
            bar.focus()
            await pilot.pause()
            await pilot.press("f", "i", "x", "enter")
            deadline = time.monotonic() + 10
            notice = app.screen.query_one("#draft-notice", Static)
            while "REFUSING" not in str(notice.render()) and time.monotonic() < deadline:
                await pilot.pause(0.05)
            shown = str(notice.render())
            assert "REFUSING" in shown and "[git] seed.txt" in shown  # markup-safe
            assert bar.text == "fix"  # back in the composer, not lost
            assert app.return_value is None and isinstance(app.screen, NewWorkScreen)

    asyncio.run(scenario())


def test_run_merge_cli_builds_argv_and_parses_result(tmp_path: Path, monkeypatch: object) -> None:
    """The hub's merge helper shells out to `agent6 sessions merge <id>` and reports the
    captured output as (ok, message) -- it never touches git_ops itself."""
    import subprocess

    from agent6.ui.tui import home

    captured: list[list[str]] = []

    class _Proc:
        pid = 424242
        returncode = 0
        stdout = "[agent6] merged agent6/r1 into main (squash) -> abcdef123456\n"
        stderr = ""

    def _fake_run(argv: list[str], **_kw: object) -> _Proc:
        captured.append(list(argv))
        return _Proc()

    monkeypatch.setattr(subprocess, "run", _fake_run)  # type: ignore[attr-defined]

    ok, msg = home._run_merge_cli(tmp_path, "r1")  # pyright: ignore[reportPrivateUsage]
    assert captured[-1][1:] == ["sessions", "merge", "r1"]
    assert ok is True
    assert "merged agent6/r1" in msg


def test_merge_action_confirms_then_shells_out(tmp_path: Path, monkeypatch: object) -> None:
    """Pressing `m` opens a confirm modal; confirming runs `agent6 sessions merge` for the
    selected run (stubbed here so no real CLI is spawned)."""
    import asyncio
    import subprocess as sp

    from textual.widgets import DataTable

    from agent6.ui.tui import home
    from agent6.ui.tui.home import Agent6HomeApp
    from agent6.ui.tui.modals import ConfirmModal

    a6 = tmp_path / ".agent6"
    rd = _write_run(a6, "runs", "r1", [{"type": "session.start", "mode": "run", "user_task": "x"}])
    # Merge is offered only for a run whose branch holds commits its base does
    # not: the key is dimmed otherwise, as the CLI refuses those.
    repo = tmp_path
    sp.run(["git", "init", "-q", "-b", "main"], cwd=repo, check=True)
    sp.run(["git", "config", "user.email", "t@example.com"], cwd=repo, check=True)
    sp.run(["git", "config", "user.name", "t"], cwd=repo, check=True)
    (repo / "seed.txt").write_text("seed\n")
    sp.run(["git", "add", "seed.txt"], cwd=repo, check=True)
    sp.run(["git", "commit", "-q", "-m", "base"], cwd=repo, check=True)
    base = sp.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True, check=True
    ).stdout.strip()
    (repo / "work.txt").write_text("work\n")
    sp.run(["git", "add", "work.txt"], cwd=repo, check=True)
    sp.run(["git", "commit", "-q", "-m", "work"], cwd=repo, check=True)
    sp.run(["git", "branch", "agent6/r1"], cwd=repo, check=True)
    sp.run(["git", "reset", "-q", "--hard", base], cwd=repo, check=True)
    (rd / "manifest.json").write_text(
        json.dumps(
            {
                "version": 3,
                "session_id": "r1",
                "mode": "run",
                "user_task": "x",
                "base_sha": base,
                "base_branch": "main",
                "run_branch": "agent6/r1",
            }
        ),
        encoding="utf-8",
    )

    calls: list[str] = []

    def _fake_merge(cwd: Path, session_id: str, config_path: object = None) -> tuple[bool, str]:
        calls.append(session_id)
        return True, "merged"

    monkeypatch.setattr(home, "_run_merge_cli", _fake_merge)  # type: ignore[attr-defined]

    async def scenario() -> None:
        app = Agent6HomeApp(a6, tmp_path)
        async with app.run_test() as pilot:
            await pilot.pause()
            tbl = app.screen.query_one("#sessions", DataTable)
            tbl.focus()
            tbl.move_cursor(row=0)
            await pilot.press("m")
            await pilot.pause()
            assert isinstance(app.screen, ConfirmModal)
            await pilot.press("y")  # confirm
            await pilot.pause()
            assert calls == ["r1"]  # merged the selected run

    asyncio.run(scenario())


def test_home_open_run_returns_its_dir(tmp_path: Path) -> None:
    """Selecting a run on the hub (Enter on the row) opens it: the app exits
    returning that run directory for the dashboard to watch."""
    import asyncio

    from textual.widgets import DataTable

    from agent6.ui.tui.home import Agent6HomeApp

    a6 = tmp_path / ".agent6"
    rd = _write_run(a6, "runs", "r1", [{"type": "session.start", "mode": "run", "user_task": "x"}])

    async def scenario() -> None:
        app = Agent6HomeApp(a6, tmp_path)
        async with app.run_test() as pilot:
            await pilot.pause()
            tbl = app.screen.query_one("#sessions", DataTable)
            tbl.focus()
            tbl.move_cursor(row=0)
            await pilot.press("enter")
            await pilot.pause()
        assert app.return_value == rd  # opened the selected run

    asyncio.run(scenario())


def test_hub_repaints_a_dying_run_without_a_keypress(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The hub was the one TUI screen with no poll: a run that died while
    listed kept its last-computed word (bold-cyan "running") until a keypress
    or a screen change."""
    from textual.widgets import DataTable

    import agent6.ui.tui.home as home_mod
    from agent6.ui.tui.home import Agent6HomeApp

    monkeypatch.setattr(home_mod, "_HUB_POLL_S", 0.2)
    a6 = tmp_path / ".agent6"
    rd = _write_run(a6, "runs", "r1", [{"type": "session.start", "mode": "run"}])
    (rd / "worker.pid").write_text(str(os.getpid()), encoding="utf-8")

    async def scenario() -> None:
        app = Agent6HomeApp(a6, tmp_path)
        async with app.run_test(size=(140, 40)) as pilot:

            def status_cell() -> str:
                table = app.screen.query_one("#sessions", DataTable)
                if table.row_count == 0:
                    return ""
                return str(table.get_row_at(0)[1])

            await _wait_for(pilot, lambda: "running" in status_cell(), "the running row")
            (rd / "worker.pid").write_text("999999999", encoding="utf-8")  # dies
            # No keypress, no screen change: the poll alone must repaint.
            await _wait_for(pilot, lambda: "stale" in status_cell(), "the stale repaint")

    asyncio.run(scenario())


def test_hub_refresh_keeps_the_selected_run_as_rows_reorder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The poll rebuilds the table; the operator's selection must follow the
    run it was on (by id), not snap back to whatever lands in that row index."""
    from textual.widgets import DataTable

    import agent6.ui.tui.home as home_mod
    from agent6.ui.tui.home import Agent6HomeApp, HomeScreen

    monkeypatch.setattr(home_mod, "_HUB_POLL_S", 3600.0)  # manual refresh only
    a6 = tmp_path / ".agent6"
    for name, ts in (("r1", 1000), ("r2", 2000), ("r3", 3000)):
        rd = _write_run(a6, "runs", name, [{"type": "session.start", "mode": "run"}])
        os.utime(rd / "logs.jsonl", (ts, ts))

    async def scenario() -> None:
        app = Agent6HomeApp(a6, tmp_path)
        async with app.run_test(size=(140, 40)) as pilot:

            def table() -> DataTable[Any]:  # newest first: r3, r2, r1
                return app.screen.query_one("#sessions", DataTable)

            await _wait_for(pilot, lambda: table().row_count == 3, "the three rows")
            await pilot.press("down")  # cursor onto r2
            scr = app.screen
            assert isinstance(scr, HomeScreen)
            runs = scr._runs  # pyright: ignore[reportPrivateUsage]
            assert runs[table().cursor_row].name == "r2"
            # r1 gets fresh activity and jumps to the top: order becomes r1, r3, r2.
            os.utime(runs[2] / "logs.jsonl", (9000, 9000))
            await pilot.press("r")
            await pilot.pause()
            runs = scr._runs  # pyright: ignore[reportPrivateUsage]
            assert runs[table().cursor_row].name == "r2"

    asyncio.run(scenario())


def test_cost_cell_marks_partial_and_keeps_zero_clean() -> None:
    """The listing rows (hub and `sessions`) render the same '~' lower-bound
    marker as `sessions show`; an all-unpriced run's ~$0.0000 is information,
    a clean $0 stays blank."""
    from agent6.viewmodel.format import format_cost_cell

    assert format_cost_cell(0.0123, partial=True) == "~$0.01"
    assert format_cost_cell(0.0, partial=True) == "~$0.0000"
    assert format_cost_cell(0.0, partial=False) == ""
    assert format_cost_cell(0.0123, partial=False) == "$0.01"


def test_tui_hub_is_pointed_at_the_state_dir_not_the_sessions_root(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """`agent6 tui` hands `run_home` the STATE dir, the base every bucket lookup
    is relative to.

    Handing it `<state>/sessions` made `bucket_dir` append `sessions/` a second
    time, so the hub listed nothing while the CLI and the web listed every
    session, and the TUI's machine watch read the authoring bucket instead of
    the instance dir. Every other test calls `session_dirs` directly, so
    nothing covered the argument."""
    from agent6.ui.cli import plan_watch
    from agent6.ui.tui import home

    monkeypatch.chdir(tmp_path)
    seen: list[Path] = []

    def _capture(base: Path, _cwd: Path, _cp: object = None) -> None:
        seen.append(base)

    monkeypatch.setattr(home, "run_home", _capture)
    assert plan_watch._cmd_tui() == 0  # pyright: ignore[reportPrivateUsage]
    assert seen == [state_dir(tmp_path)]


def test_merge_is_greyed_out_for_a_live_run(tmp_path: Path) -> None:
    """`sessions merge` always refuses a live run, so the hub greys the key
    instead of confirming a modal to be told no. None greys it; False would
    hide it, and a key missing from the footer reads as a capability the hub
    does not have."""
    import asyncio
    import os

    from agent6.ui.tui.home import Agent6HomeApp

    a6 = tmp_path / ".agent6"
    live = _write_run(
        a6, "runs", "r-live", [{"type": "session.start", "mode": "run", "user_task": "x"}]
    )
    (live / "worker.pid").write_text(str(os.getpid()), encoding="utf-8")

    async def scenario() -> None:
        app = Agent6HomeApp(a6, tmp_path)
        async with app.run_test() as pilot:
            await pilot.pause()
            screen = app.screen
            assert screen.check_action("merge_selected", ()) is None
            assert screen.check_action("refresh", ()) is True

    asyncio.run(scenario())


def test_the_hub_table_names_its_columns_like_the_cli(tmp_path: Path) -> None:
    """The time column had three names across the hubs: `updated` (CLI),
    `when` (TUI) and a locale string (web)."""
    import asyncio

    from textual.widgets import DataTable

    from agent6.ui.tui.home import Agent6HomeApp, HomeScreen

    a6 = tmp_path / ".agent6"
    _write_run(a6, "runs", "r1", [{"type": "session.start", "mode": "run", "user_task": "r1"}])

    async def scenario() -> None:
        app = Agent6HomeApp(a6, tmp_path)
        async with app.run_test() as pilot:
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, HomeScreen)
            table = screen.query_one("#sessions", DataTable)
            labels = [str(column.label) for column in table.columns.values()]
            assert labels == ["updated", "status", "cost", "id", "task"]

    asyncio.run(scenario())


def test_delete_action_confirms_then_shells_out(tmp_path: Path, monkeypatch: object) -> None:
    """Merge had a hub key and delete lived only in the run view's menu, so the
    two verbs over one row were reached from opposite ends of the app.
    Pressing `d` opens a confirm modal; confirming runs `agent6 sessions rm`
    for the selected run (stubbed here so no real CLI is spawned)."""
    import asyncio

    from textual.widgets import DataTable

    from agent6.ui.tui import home
    from agent6.ui.tui.home import Agent6HomeApp
    from agent6.ui.tui.modals import ConfirmModal

    a6 = tmp_path / ".agent6"
    _write_run(a6, "runs", "r1", [{"type": "session.start", "mode": "run", "user_task": "x"}])
    calls: list[str] = []

    def _fake_delete(cwd: Path, session_id: str, config_path: object = None) -> tuple[bool, str]:
        calls.append(session_id)
        return True, "removed"

    monkeypatch.setattr(home, "_run_delete_cli", _fake_delete)  # type: ignore[attr-defined]

    async def scenario() -> None:
        app = Agent6HomeApp(a6, tmp_path)
        async with app.run_test() as pilot:
            await pilot.pause()
            tbl = app.screen.query_one("#sessions", DataTable)
            tbl.focus()
            tbl.move_cursor(row=0)
            await pilot.press("d")
            await pilot.pause()
            assert isinstance(app.screen, ConfirmModal)
            await pilot.press("y")
            await pilot.pause()
            assert calls == ["r1"]

    asyncio.run(scenario())


def test_delete_is_greyed_out_for_a_live_run(tmp_path: Path) -> None:
    import asyncio
    import os

    from agent6.ui.tui.home import Agent6HomeApp

    a6 = tmp_path / ".agent6"
    live = _write_run(
        a6, "runs", "r-live", [{"type": "session.start", "mode": "run", "user_task": "x"}]
    )
    (live / "worker.pid").write_text(str(os.getpid()), encoding="utf-8")

    async def scenario() -> None:
        app = Agent6HomeApp(a6, tmp_path)
        async with app.run_test() as pilot:
            await pilot.pause()
            assert app.screen.check_action("delete_selected", ()) is None

    asyncio.run(scenario())


def test_delete_is_on_the_hubs_file_menu_like_its_sibling_verbs() -> None:
    """`d` deleted the selected run from the footer only; the File menu
    offered New, Open, Merge, Refresh and Quit, so the one destructive verb
    was the one a menu reader could not find."""
    from agent6.ui.tui.home import HomeScreen

    file_menu = HomeScreen.MENUS[0]
    assert file_menu.title == "File"
    assert "delete_selected" in {item.action for item in file_menu.items}


def test_prune_and_clear_asks_are_hub_actions_that_shell_the_cli(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The web hub prunes (with the squash-merged opt-in) and clears saved
    asks; the TUI hub reached neither. Each is a File menu action that
    confirms, then shells the same fixed argv the CLI takes."""
    import asyncio

    from agent6.ui.tui import home
    from agent6.ui.tui.home import Agent6HomeApp, HomeScreen
    from agent6.ui.tui.modals import ConfirmModal

    a6 = tmp_path / ".agent6"
    _write_run(a6, "runs", "r1", [{"type": "session.start", "mode": "run", "user_task": "x"}])
    calls: list[tuple[str, bool]] = []

    def _fake_prune(
        cwd: Path, *, delete_squashed: bool, config_path: object = None
    ) -> tuple[bool, str]:
        calls.append(("prune", delete_squashed))
        return True, "pruned"

    def _fake_clear(cwd: Path, config_path: object = None) -> tuple[bool, str]:
        calls.append(("asks", False))
        return True, "cleared"

    monkeypatch.setattr(home, "_run_prune_cli", _fake_prune)  # type: ignore[attr-defined]
    monkeypatch.setattr(home, "_run_clear_asks_cli", _fake_clear)  # type: ignore[attr-defined]
    actions = {item.action for item in HomeScreen.MENUS[0].items}
    assert {"prune", "prune_squashed", "clear_asks"} <= actions

    async def scenario() -> None:
        app = Agent6HomeApp(a6, tmp_path)
        async with app.run_test() as pilot:
            await pilot.pause()
            for action in ("prune", "prune_squashed", "clear_asks"):
                screen = app.screen
                assert isinstance(screen, HomeScreen)
                getattr(screen, f"action_{action}")()
                await pilot.pause()
                assert isinstance(app.screen, ConfirmModal)
                await pilot.press("y")
                await pilot.pause()
            assert calls == [("prune", False), ("prune", True), ("asks", False)]

    asyncio.run(scenario())


def test_the_hub_folds_a_fan_outs_lanes_and_space_expands_them(tmp_path: Path) -> None:
    """A fan-out is one row carrying its lane count; Space on it lists the
    lanes under it (the row list stays 1:1 with the table), Space again folds
    them."""
    import asyncio

    from textual.widgets import DataTable

    from agent6.ui.tui.home import Agent6HomeApp, HomeScreen

    a6 = tmp_path / ".agent6"
    start: dict[str, object] = {"type": "session.start", "mode": "run", "user_task": "t"}
    fan = _write_run(a6, "runs", "fan", [start])
    (fan / "manifest.json").write_text(
        json.dumps({"mode": "run", "fanout": {"lanes": 2, "spec": "2"}}), encoding="utf-8"
    )
    for lane in (1, 2):
        d = _write_run(a6, "runs", f"fan-l{lane}", [start])
        (d / "manifest.json").write_text(
            json.dumps(
                {"mode": "run", "parallel": {"group": "fan", "lane": lane, "coordinator": "fan"}}
            ),
            encoding="utf-8",
        )

    async def scenario() -> None:
        app = Agent6HomeApp(a6, tmp_path)
        async with app.run_test() as pilot:
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, HomeScreen)
            table = screen.query_one("#sessions", DataTable)
            runs = screen._runs  # pyright: ignore[reportPrivateUsage]
            assert table.row_count == 1 and [rd.name for rd in runs] == ["fan"]
            assert "(2 lanes)" in str(table.get_row_at(0)[3])
            screen.action_toggle_lanes()
            await pilot.pause()
            runs = screen._runs  # pyright: ignore[reportPrivateUsage]
            assert [rd.name for rd in runs] == ["fan", "fan-l1", "fan-l2"]
            assert "(2 lanes)" not in str(table.get_row_at(0)[3])
            assert "└ fan-l1" in str(table.get_row_at(1)[3])
            screen.action_toggle_lanes()
            await pilot.pause()
            assert table.row_count == 1

    asyncio.run(scenario())
