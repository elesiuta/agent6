# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""/undo in the TUI: the fork is the continuation, and the message taken back is
the operator's to edit and resend. The fold's `undone_to` (a live /undo) and
this view's own `undo_fork` on a finished run both route the composer there."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest
from textual.widgets import Static

from agent6.ui.tui import app as app_mod
from agent6.ui.tui.app import Agent6TUI
from agent6.ui.tui.composer import SteerInput


def _undone_run(d: Path) -> None:
    d.mkdir(parents=True, exist_ok=True)
    evs = [
        {"type": "session.start", "session_id": d.name, "mode": "run", "user_task": "add it"},
        {"type": "loop.steer.injected", "chars": 14, "text": "name it better"},
        {
            "type": "session.undone",
            "new_session_id": "fork-child-AAAAAA",
            "undone_text": "name it better",
        },
        {"type": "session.end", "reason": "undone", "all_passed": False},
    ]
    (d / "logs.jsonl").write_text("".join(json.dumps(e) + "\n" for e in evs), encoding="utf-8")


def test_a_live_undo_hands_the_follow_up_to_the_fork(tmp_path: Path, monkeypatch: Any) -> None:
    """The composer holds the undone text, its title names the fork, and Enter
    resumes the fork (this view's run is over; the fork carries on). The
    view used to keep an empty composer pointed at the undone run, whose
    resume would have started a leg on the wrong session."""
    spawned: list[tuple[str, str]] = []

    def _fake_resume(
        _cwd: Path, rid: str, *, steer: str = "", preset: str = "", config_path: object = None
    ) -> str:
        spawned.append((rid, steer))
        return ""

    monkeypatch.setattr(app_mod, "spawn_detached_resume", _fake_resume)
    run = tmp_path / "undone-run-AAAAAA"
    _undone_run(run)

    async def scenario() -> None:
        app = Agent6TUI(run)
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            await pilot.pause()
            bar = app._conv.query_one("#conv-input", SteerInput)  # pyright: ignore[reportPrivateUsage]
            assert app.continue_as == "fork-child-AAAAAA"
            assert bar.text == "name it better"
            assert "continue as fork-child-AAAAAA" in str(bar.border_title)
            app.submit_instruction("name it much better")
            await app.workers.wait_for_complete()
            assert spawned == [("fork-child-AAAAAA", "name it much better")]
            # The dashboard's bar agrees.
            await pilot.press("ctrl+d")
            await pilot.pause()
            app._heartbeat_at = 0.0  # pyright: ignore[reportPrivateUsage]
            app._tick()  # pyright: ignore[reportPrivateUsage]
            await pilot.pause()
            dash_bar = app._dash.query_one("#dash-input", SteerInput)  # pyright: ignore[reportPrivateUsage]
            assert "continue as fork-child-AAAAAA" in str(dash_bar.border_title)
            assert isinstance(app._dash.query_one("#top", Static), Static)  # pyright: ignore[reportPrivateUsage]

    asyncio.run(scenario())


def test_undo_of_a_finished_run_fills_the_composer_and_routes_to_the_child(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """The non-live path forks in-process (no event lands in this run's log);
    the view still hands the follow-up to the child."""
    spawned: list[tuple[str, str]] = []

    def _fake_resume(
        _cwd: Path, rid: str, *, steer: str = "", preset: str = "", config_path: object = None
    ) -> str:
        spawned.append((rid, steer))
        return ""

    def _fake_undo_fork(*_a: object, **_k: object) -> tuple[str, str]:
        return ("fork-child-BBBBBB", "the message taken back")

    monkeypatch.setattr(app_mod, "spawn_detached_resume", _fake_resume)
    monkeypatch.setattr(app_mod, "undo_fork", _fake_undo_fork)
    run = tmp_path / "done-run-AAAAAA"
    run.mkdir()
    evs = [
        {"type": "session.start", "session_id": run.name, "mode": "run", "user_task": "t"},
        {"type": "session.end", "reason": "finish_session", "all_passed": True},
    ]
    (run / "logs.jsonl").write_text("".join(json.dumps(e) + "\n" for e in evs), encoding="utf-8")

    async def scenario() -> None:
        app = Agent6TUI(run)
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            await pilot.pause()
            app.submit_instruction("/undo")
            await pilot.pause()
            bar = app._conv.query_one("#conv-input", SteerInput)  # pyright: ignore[reportPrivateUsage]
            assert bar.text == "the message taken back"
            assert app.continue_as == "fork-child-BBBBBB"
            app.submit_instruction("the message, edited")
            await app.workers.wait_for_complete()
            assert spawned == [("fork-child-BBBBBB", "the message, edited")]

    asyncio.run(scenario())


@pytest.mark.parametrize("continue_as", ["", "fork-child-AAAAAA"])
def test_composer_labels_name_the_fork(continue_as: str) -> None:
    from agent6.ui.tui.composer import composer_labels

    title, _ = composer_labels("resume", continue_as=continue_as)
    assert title == ("continue as fork-child-AAAAAA" if continue_as else "continue this session")


def test_fork_of_a_finished_run_hands_the_composer_to_the_unstarted_fork(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """Run > Fork used to spawn `agent6 fork <id>` continuing at once; with
    no direction, a fork of a finished run re-read a done conversation and
    ended as a silent finish. It now creates the fork unstarted; on a finished
    run the composer routes the next line to the fork (Enter resumes it)."""
    spawned: list[tuple[str, str]] = []

    def _fake_resume(
        _cwd: Path, rid: str, *, steer: str = "", preset: str = "", config_path: object = None
    ) -> str:
        spawned.append((rid, steer))
        return ""

    def _fake_create_fork(*_a: object, **_k: object) -> tuple[str, int]:
        return ("fork-child-CCCCCC", 0)

    monkeypatch.setattr(app_mod, "spawn_detached_resume", _fake_resume)
    monkeypatch.setattr(app_mod, "create_fork", _fake_create_fork)
    run = tmp_path / "done-run-AAAAAA"
    run.mkdir()
    evs = [
        {"type": "session.start", "session_id": run.name, "mode": "run", "user_task": "t"},
        {"type": "session.end", "reason": "finish_session", "all_passed": True},
    ]
    (run / "logs.jsonl").write_text("".join(json.dumps(e) + "\n" for e in evs), encoding="utf-8")

    async def scenario() -> None:
        app = Agent6TUI(run)
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            await pilot.pause()
            app.action_fork()
            await pilot.pause()
            assert app.continue_as == "fork-child-CCCCCC"
            bar = app._conv.query_one("#conv-input", SteerInput)  # pyright: ignore[reportPrivateUsage]
            assert "continue as fork-child-CCCCCC" in str(bar.border_title)
            app.submit_instruction("try the other design")
            await app.workers.wait_for_complete()
            assert spawned == [("fork-child-CCCCCC", "try the other design")]

    asyncio.run(scenario())


def test_fork_of_a_live_run_leaves_the_composer_steering_this_run(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """The live composer steers THIS run; a fork made while live is created
    unstarted and the notice says how it starts."""
    import os

    from agent6.sessions.ipc import write_worker_pid

    def _fake_create_fork(*_a: object, **_k: object) -> tuple[str, int]:
        return ("fork-child-DDDDDD", 0)

    monkeypatch.setattr(app_mod, "create_fork", _fake_create_fork)
    run = tmp_path / "live-run-AAAAAA"
    run.mkdir()
    (run / "logs.jsonl").write_text(
        json.dumps(
            {"type": "session.start", "session_id": run.name, "mode": "run", "user_task": "t"}
        )
        + "\n",
        encoding="utf-8",
    )
    write_worker_pid(run, os.getpid())

    async def scenario() -> None:
        app = Agent6TUI(run)
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            app.action_fork()
            await pilot.pause()
            assert app.continue_as == ""
            assert app.session_controllable()

    asyncio.run(scenario())


def test_resume_of_a_finished_run_refuses_here_and_points_at_the_composer(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """Run > Resume on a run the agent ended spawned a detached `agent6 resume`
    that refused ("already finished; give it new work") on a stderr nobody
    read, while the toast said "resuming…". The refusal lands here; the
    composer below is how it gets new work. A stopped/crashed run still
    resumes."""
    spawned: list[tuple[str, str]] = []

    def _fake_resume(
        _cwd: Path, rid: str, *, steer: str = "", preset: str = "", config_path: object = None
    ) -> str:
        spawned.append((rid, steer))
        return ""

    monkeypatch.setattr(app_mod, "spawn_detached_resume", _fake_resume)
    run = tmp_path / "done-run-AAAAAA"
    run.mkdir()
    evs = [
        {"type": "session.start", "session_id": run.name, "mode": "run", "user_task": "t"},
        {"type": "session.end", "reason": "finish_session", "all_passed": True},
    ]
    (run / "logs.jsonl").write_text("".join(json.dumps(e) + "\n" for e in evs), encoding="utf-8")
    notes: list[str] = []

    async def scenario() -> None:
        app = Agent6TUI(run)
        original = app.notify

        def spy(message: Any, *a: Any, **k: Any) -> None:
            notes.append(str(message))
            original(message, *a, **k)

        monkeypatch.setattr(app, "notify", spy)
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            await pilot.pause()
            app.action_resume()
            await pilot.pause()
            assert spawned == []
            assert any("type what to do next" in n for n in notes)
            # Stopped by the operator: resume is the continuation.
            (run / "logs.jsonl").write_text(
                "".join(
                    json.dumps(e) + "\n"
                    for e in (
                        evs[0],
                        {"type": "session.end", "reason": "steer_abort", "all_passed": False},
                    )
                ),
                encoding="utf-8",
            )
            app.action_resume()
            await app.workers.wait_for_complete()
            assert spawned == [(run.name, "")]

    asyncio.run(scenario())


def test_run_this_plan_spawns_the_run_detached(tmp_path: Path, monkeypatch: Any) -> None:
    """Run > Run this plan on a finished plan spawns `agent6 run --from
    <id>` with the detached env; a non-plan session refuses without spawning."""
    seen: dict[str, Any] = {}

    def _fake_spawn(argv: list[str], _cwd: Path, **kw: Any) -> tuple[Path | None, str]:
        seen["argv"] = argv
        seen["env"] = kw.get("env")
        return tmp_path / "sessions" / "runs" / "fresh-run-CCCCCC", ""

    monkeypatch.setattr(app_mod, "spawn_and_locate", _fake_spawn)
    plan = tmp_path / "sessions" / "plans" / "planny-one-AAAAAA"
    plan.mkdir(parents=True)
    (plan / "manifest.json").write_text(
        json.dumps({"version": 2, "session_id": plan.name, "mode": "plan", "user_task": "t"}),
        encoding="utf-8",
    )
    (plan / "logs.jsonl").write_text("", encoding="utf-8")
    (plan / "plan.md").write_text("# Plan\n", encoding="utf-8")

    async def scenario() -> None:
        app = Agent6TUI(plan)
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            app.action_run_plan()
            await app.workers.wait_for_complete()

    asyncio.run(scenario())
    assert seen["argv"][-3:] == ["run", "--from", "planny-one-AAAAAA"]
    assert seen["env"]["AGENT6_DETACHED_AWAY"] == "wait"

    seen.clear()
    run = tmp_path / "sessions" / "runs" / "runny-one-AAAAAA"
    run.mkdir(parents=True)
    (run / "logs.jsonl").write_text("", encoding="utf-8")

    async def refuse() -> None:
        app = Agent6TUI(run)
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            app.action_run_plan()
            await app.workers.wait_for_complete()

    asyncio.run(refuse())
    assert not seen


def test_a_refused_btw_toasts_as_a_warning(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """`open_btw` carries its outcome, and the composer dropped it: a refused
    side question toasted at the success severity, unlike every other refusal
    beside it."""

    def refused(_session_dir: Path, _question: str) -> tuple[bool, str]:
        return False, "no live run to ask beside"

    def controllable(_app: Agent6TUI) -> bool:
        return True

    monkeypatch.setattr(app_mod, "open_btw", refused)
    monkeypatch.setattr(Agent6TUI, "session_controllable", controllable)
    run = tmp_path / "sessions" / "runs" / "runny-two-BBBBBB"
    run.mkdir(parents=True)
    (run / "logs.jsonl").write_text("", encoding="utf-8")

    async def scenario() -> list[tuple[str, str]]:
        app = Agent6TUI(run)
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            app.submit_instruction("/btw why?")
            await pilot.pause()
            notes = app._notifications  # pyright: ignore[reportPrivateUsage]
            return [(str(n.message), n.severity) for n in notes]

    assert ("no live run to ask beside", "warning") in asyncio.run(scenario())
