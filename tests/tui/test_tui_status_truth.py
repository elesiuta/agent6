# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The TUI run views read run status from THE dir decision (status_for_session_dir).

Before this, the TUI derived status three separate ways (a pure event fold for
the label, a one-way run_ended latch for liveness, the conversation's own
event-tracked _live) and each lied somewhere: a parked run rendered a blank
label over "(waiting for the model…)" with a steer composer nobody would ever
read; a dead worker was labelled "worker exited" where the hub says "stale";
a crash->resume kept "worker exited" painted over the live leg forever; and
the two composer bars disagreed with each other live.
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any

import pytest
from textual.app import ScreenStackError
from textual.widgets import Static

from agent6.ui.tui.app import Agent6TUI
from agent6.ui.tui.composer import ApprovalRow, SteerInput
from agent6.ui.tui.modals import ApprovalModal
from agent6.viewmodel.state import apply_event


def _mk_parked(d: Path) -> None:
    d.mkdir(parents=True, exist_ok=True)
    (d / "manifest.json").write_text(
        json.dumps(
            {
                "version": 2,
                "session_id": d.name,
                "mode": "run",
                "user_task": "fix the flaky test",
                "parked_task": "fix the flaky test",
                "parked_reason": "checkout busy",
            }
        ),
        encoding="utf-8",
    )


def _mk_crashed(d: Path) -> None:
    d.mkdir(parents=True, exist_ok=True)
    evs = [
        {"type": "session.start", "session_id": d.name, "mode": "run", "user_task": "t"},
        {"type": "role.call", "role": "worker", "model": "m", "provider": "p"},
    ]
    (d / "logs.jsonl").write_text("".join(json.dumps(e) + "\n" for e in evs), encoding="utf-8")
    (d / "worker.pid").write_text("999999999", encoding="utf-8")


def _screen_is(app: Agent6TUI, name: str) -> bool:
    """`app.screen` raises while the stack is transiently empty (startup,
    mid-switch); a poll reads that as "not yet", never an error."""
    try:
        current = app.screen
    except ScreenStackError:
        return False
    return current is getattr(app, name)


async def _wait_for(pilot: Any, cond: Any, what: str, timeout: float = 10.0) -> None:
    deadline = time.monotonic() + timeout
    while not cond():
        assert time.monotonic() < deadline, f"timed out waiting for {what}"
        await pilot.pause(0.05)


async def _open_dash(app: Agent6TUI, pilot: Any) -> None:
    await _wait_for(pilot, lambda: _screen_is(app, "_conv"), "the conversation screen")
    await pilot.press("ctrl+d")
    await _wait_for(pilot, lambda: _screen_is(app, "_dash"), "the dashboard screen")
    app._heartbeat_at = 0.0  # age the throttle so the dir-status probe fires now
    app._tick()
    await pilot.pause()


def test_the_dashboard_title_word_is_the_sessions_mode(tmp_path: Path) -> None:
    """The menu-bar title led with a fixed "run" for every session; a plan read
    "run · <task> · planned". The word is the manifest's mode, as the web
    panel heading states it."""
    d = tmp_path / "plan1"
    d.mkdir()
    (d / "manifest.json").write_text(
        json.dumps({"version": 2, "session_id": d.name, "mode": "plan", "user_task": "lay it out"}),
        encoding="utf-8",
    )
    (d / "logs.jsonl").write_text(
        json.dumps(
            {
                "type": "session.start",
                "session_id": d.name,
                "mode": "plan",
                "user_task": "lay it out",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    async def scenario() -> None:
        app = Agent6TUI(d)
        async with app.run_test(size=(140, 40)) as pilot:
            await _open_dash(app, pilot)
            assert app.run_title().startswith("plan · lay it out")

    asyncio.run(scenario())


def test_a_finished_plans_deliverable_is_in_the_stream_pane(tmp_path: Path) -> None:
    """A plan's product is plan.md; the CLI prints it at the end and the web
    shows it in a card, but the dashboard's end story showed only the summary
    line, sending the operator to `plan show`."""
    d = tmp_path / "plan2"
    d.mkdir()
    (d / "manifest.json").write_text(
        json.dumps({"version": 2, "session_id": d.name, "mode": "plan", "user_task": "lay it out"}),
        encoding="utf-8",
    )
    evs = [
        {"type": "session.start", "session_id": d.name, "mode": "plan", "user_task": "lay it out"},
        {"type": "tool.call", "name": "finish_planning", "args": {"summary": "Plan seeded."}},
        {"type": "tool.result", "name": "finish_planning", "ok": True, "summary": "ok"},
        {"type": "session.end", "reason": "finish_planning", "all_passed": True},
    ]
    (d / "logs.jsonl").write_text("".join(json.dumps(e) + "\n" for e in evs), encoding="utf-8")
    (d / "plan.md").write_text(
        "# Plan: lay it out\n\n## Tasks\n1. do the thing\n", encoding="utf-8"
    )

    async def scenario() -> None:
        app = Agent6TUI(d)
        async with app.run_test(size=(140, 40)) as pilot:
            await _open_dash(app, pilot)
            body = str(app._dash.query_one("#stream-body", Static).render())
            assert "planned" in body
            assert "Plan seeded." in body
            assert "1. do the thing" in body

    asyncio.run(scenario())


def test_the_header_names_the_pins_in_force(tmp_path: Path) -> None:
    """The pinned instructions bind for the whole run; the dashboard header
    lists them (the web header's and `sessions show`'s line), so an operator
    watching a run sees what --pin or /pin set without reading the log."""
    d = tmp_path / "pinned1"
    d.mkdir()
    evs = [
        {"type": "session.start", "session_id": d.name, "mode": "run", "user_task": "t"},
        {"type": "loop.pin.restored", "pins": ["never touch tests"], "count": 1},
        {"type": "loop.pin.added", "text": "keep the API stable", "chars": 19, "count": 2},
    ]
    (d / "logs.jsonl").write_text("".join(json.dumps(e) + "\n" for e in evs), encoding="utf-8")

    async def scenario() -> None:
        app = Agent6TUI(d)
        async with app.run_test(size=(140, 40)) as pilot:
            await _open_dash(app, pilot)
            top = str(app._dash.query_one("#top", Static).render())
            assert "pins: never touch tests | keep the API stable" in top

    asyncio.run(scenario())


def test_parked_run_tells_the_truth_on_every_pane(tmp_path: Path, monkeypatch: Any) -> None:
    """A parked run's dashboard leads with the hub's words ("parked · checkout
    busy"), the stream pane says parked (never the "(waiting for the model…)"
    lie -- no model is coming), and the composer routes to resume, exactly like
    a finished run's."""
    from agent6.ui.tui import app as app_mod

    spawned: list[tuple[str, str]] = []

    def _fake_resume(
        _cwd: Path, rid: str, *, steer: str = "", preset: str = "", config_path: object = None
    ) -> str:
        spawned.append((rid, steer))
        return ""

    monkeypatch.setattr(app_mod, "spawn_detached_resume", _fake_resume)
    _mk_parked(tmp_path / "parked1")

    async def scenario() -> None:
        app = Agent6TUI(tmp_path / "parked1")
        async with app.run_test(size=(140, 40)) as pilot:
            await _open_dash(app, pilot)
            assert app.session_controllable() is False  # resume is the one action
            top = str(app._dash.query_one("#top", Static).render())
            assert "parked · checkout busy" in top
            assert "task: fix the flaky test" in top  # manifest fallback, not a blank line
            body = str(app._dash.query_one("#stream-body", Static).render())
            assert "parked" in body
            assert "waiting for the model" not in body
            assert "working…" not in body
            # Both composer bars offer the resume action, not a dead-end steer.
            assert app._dash.query_one("#dash-input", SteerInput).border_title == (
                "continue this session"
            )
            app.submit_instruction("go ahead")
            await app.workers.wait_for_complete()
            assert spawned == [("parked1", "go ahead")]

    asyncio.run(scenario())


def test_a_resume_from_the_composer_carries_the_picked_preset(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """A run that is not live shows the preset picker above its composer (both
    views); the pick rides the detached resume as `--preset`, and "(as
    recorded)" sends none."""
    from textual.widgets import Select

    from agent6.ui.tui import app as app_mod
    from agent6.ui.tui.composer import ResumePreset

    spawned: list[tuple[str, str, str]] = []

    def _fake_resume(
        _cwd: Path, rid: str, *, steer: str = "", preset: str = "", config_path: object = None
    ) -> str:
        spawned.append((rid, steer, preset))
        return ""

    monkeypatch.setattr(app_mod, "spawn_detached_resume", _fake_resume)

    def _presets(_cwd: Path, _cp: object) -> list[str]:
        return ["quick", "ultra"]

    monkeypatch.setattr(app_mod, "available_preset_names", _presets)
    _mk_parked(tmp_path / "parked2")

    async def scenario() -> None:
        app = Agent6TUI(tmp_path / "parked2")
        async with app.run_test(size=(140, 40)) as pilot:
            await _wait_for(pilot, lambda: _screen_is(app, "_conv"), "the conversation screen")
            picker = app._conv.query_one("#conv-preset", ResumePreset)
            await _wait_for(pilot, lambda: picker.display, "the preset picker")
            picker.query_one(Select).value = "quick"
            await pilot.pause()
            assert app.resume_preset == "quick"
            app.submit_instruction("go ahead")
            await app.workers.wait_for_complete()
            assert spawned == [("parked2", "go ahead", "quick")]
            # The dashboard's picker shows the same choice.
            await _open_dash(app, pilot)
            dash_picker = app._dash.query_one("#dash-preset", ResumePreset)
            await _wait_for(pilot, lambda: dash_picker.display, "the dashboard's picker")
            assert dash_picker.query_one(Select).value == "quick"

    asyncio.run(scenario())


def test_dead_worker_leads_with_the_hub_word_stale(tmp_path: Path) -> None:
    """The top-line label for a lost worker is "stale" -- the word the hub row
    shows for the same probe -- with the explanatory sentence kept in the
    stream pane. Two surfaces, one word."""
    _mk_crashed(tmp_path / "crashed1")

    async def scenario() -> None:
        app = Agent6TUI(tmp_path / "crashed1")
        async with app.run_test(size=(140, 40)) as pilot:
            await _open_dash(app, pilot)
            await _wait_for(pilot, lambda: app.worker_lost, "the dead-worker probe")
            app._tick()
            await pilot.pause()
            top = str(app._dash.query_one("#top", Static).render())
            assert "stale" in top
            assert "worker exited" not in top  # the label is the hub's word now
            body = str(app._dash.query_one("#stream-body", Static).render())
            assert "worker exited without finishing" in body  # the detail stays

    asyncio.run(scenario())


def test_crash_then_resume_recovers_liveness(tmp_path: Path) -> None:
    """The dead-worker state is DERIVED, not latched: after the operator
    resumes (new leg appends events, live worker.pid), the dashboard label
    clears, the composers relabel to steer, and submits steer the live leg --
    the one-way run_ended latch kept "worker exited" painted over the live
    resumed leg and silently dropped operator input."""
    d = tmp_path / "revived1"
    _mk_crashed(d)

    async def scenario() -> None:
        app = Agent6TUI(d)
        async with app.run_test(size=(140, 40)) as pilot:
            await _open_dash(app, pilot)
            await _wait_for(pilot, lambda: app.worker_lost, "the dead-worker probe")
            # The operator resumes: a new leg appends to the log and records a
            # live worker pid.
            with (d / "logs.jsonl").open("a", encoding="utf-8") as fh:
                fh.write(json.dumps({"type": "loop.resume.start", "iteration": 2}) + "\n")
                fh.write(json.dumps({"type": "role.call", "role": "worker", "model": "m"}) + "\n")
            (d / "worker.pid").write_text(str(os.getpid()), encoding="utf-8")
            await _wait_for(pilot, lambda: not app.worker_lost, "liveness to recover after resume")
            assert app.session_controllable() is True
            app._heartbeat_at = 0.0
            app._tick()
            await pilot.pause()
            top = str(app._dash.query_one("#top", Static).render())
            assert "stale" not in top and "worker exited" not in top
            # BOTH bars agree on the live mode -- the covered conversation too.
            assert "steer" in (app._dash.query_one("#dash-input", SteerInput).border_title or "")
            assert "steer" in (app._conv.query_one("#conv-input", SteerInput).border_title or "")

    asyncio.run(scenario())


def test_conversation_bar_tells_the_truth_about_a_dead_worker(tmp_path: Path) -> None:
    """The PRIMARY conversation view keys its composer on the host's liveness,
    not its own event tracking: a worker killed without a session.end relabels the
    bar to resume (its old event-only _live stayed True forever, and typed
    steers went to a corpse with a success toast)."""
    d = tmp_path / "convdead1"
    _mk_crashed(d)

    async def scenario() -> None:
        app = Agent6TUI(d)
        async with app.run_test(size=(140, 40)) as pilot:
            await _wait_for(pilot, lambda: _screen_is(app, "_conv"), "the conversation screen")
            app._heartbeat_at = 0.0
            app._tick()
            await _wait_for(pilot, lambda: app.worker_lost, "the dead-worker probe")
            app._heartbeat_at = 0.0
            app._tick()
            await pilot.pause()
            bar = app._conv.query_one("#conv-input", SteerInput)
            assert bar.border_title == "continue this session"

    asyncio.run(scenario())


def test_a_dead_workers_open_call_settles_into_the_scrollback(tmp_path: Path) -> None:
    """A worker killed mid-command leaves the call open with no session.end;
    the host knows the worker is gone (its pid probe), so the conversation
    settles the call as one that never returned instead of dropping it."""
    d = tmp_path / "convdead2"
    _mk_crashed(d)
    with (d / "logs.jsonl").open("a", encoding="utf-8") as fh:
        fh.write(
            json.dumps(
                {"type": "tool.call", "name": "run_command", "args": {"argv": ["sleep", "60"]}}
            )
            + "\n"
        )

    async def scenario() -> None:
        app = Agent6TUI(d)
        async with app.run_test(size=(140, 40)) as pilot:
            await _wait_for(pilot, lambda: _screen_is(app, "_conv"), "the conversation screen")
            app._heartbeat_at = 0.0
            app._tick()
            await _wait_for(pilot, lambda: app.worker_lost, "the dead-worker probe")
            app._conv._poll()  # pyright: ignore[reportPrivateUsage]
            await pilot.pause()
            body = "\n".join(
                str(w.content)
                for w in app._conv.query(".conv-chunk").results(Static)  # pyright: ignore[reportPrivateUsage]
            )
            assert "→ run_command  sleep 60" in body
            assert "no result (the run died)" in body
            assert "running" not in body

    asyncio.run(scenario())


def test_conversation_composer_routes_through_the_host_parser(tmp_path: Path) -> None:
    """A composer line on the PRIMARY conversation view routes through the
    host's submit_instruction, so `/compact <focus>` becomes an out-of-band
    compaction request exactly as on the dashboard -- not a literal steer the
    model is told to obey (the bar's own title advertises /compact)."""
    d = tmp_path / "convcompact1"
    d.mkdir()
    (d / "logs.jsonl").write_text(
        "".join(
            json.dumps(e) + "\n"
            for e in (
                {"type": "session.start", "session_id": d.name, "mode": "run", "user_task": "t"},
                {"type": "role.call", "role": "worker", "model": "m", "provider": "p"},
            )
        ),
        encoding="utf-8",
    )
    (d / "worker.pid").write_text(str(os.getpid()), encoding="utf-8")  # live

    async def scenario() -> None:
        app = Agent6TUI(d)
        async with app.run_test(size=(140, 40)) as pilot:
            await _wait_for(pilot, lambda: _screen_is(app, "_conv"), "the conversation screen")
            app._heartbeat_at = 0.0
            app._tick()
            await pilot.pause()
            bar = app._conv.query_one("#conv-input", SteerInput)
            bar.post_message(SteerInput.Submitted("/compact keep the auth decisions"))
            await pilot.pause()
            await pilot.pause()
            assert (d / "compact.request").read_text(encoding="utf-8") == (
                "keep the auth decisions"
            )
            assert not (d / "steer.answer").exists()
            assert not (d / "steer.request").exists()

    asyncio.run(scenario())


def _mk_blocked(d: Path, *, alive: bool) -> None:
    """A run blocked on an unanswered approval, with a live or dead worker."""
    d.mkdir(parents=True, exist_ok=True)
    evs = [
        {"type": "session.start", "session_id": d.name, "mode": "run", "user_task": "t"},
        {"type": "approval.prompt", "id": "ap1", "prompt": "Allow run_command: pytest"},
    ]
    (d / "logs.jsonl").write_text("".join(json.dumps(e) + "\n" for e in evs), encoding="utf-8")
    (d / "worker.pid").write_text(str(os.getpid()) if alive else "999999999", encoding="utf-8")


def test_dead_run_pops_no_approval_modal(tmp_path: Path) -> None:
    """The fold keeps an unanswered prompt past a worker death (it clears only
    on an answer event or a leg boundary), so the dashboard popped live-looking
    Allow/Deny over a corpse and wrote the answer where nobody polls."""
    d = tmp_path / "ghost1"
    _mk_blocked(d, alive=False)

    async def scenario() -> None:
        app = Agent6TUI(d)
        async with app.run_test(size=(140, 40)) as pilot:
            await _wait_for(pilot, lambda: _screen_is(app, "_conv"), "the conversation screen")
            await _wait_for(pilot, lambda: app.state.pending_approvals, "the prompt to fold")
            app._heartbeat_at = 0.0
            app._tick()
            await pilot.pause()
            assert not isinstance(app.screen, ApprovalModal)
            assert not (d / "approvals" / "ap1.answer").exists()

    asyncio.run(scenario())


def _approval_ready(app: Agent6TUI) -> bool:
    # The conversation screen renders an approval inline: the item plus its key
    # row, mounted, with the composer keeping focus (a modal only on the other
    # screens).
    rows = app._conv.query(ApprovalRow)  # pyright: ignore[reportPrivateUsage]
    bar = app._conv.query_one("#conv-input", SteerInput)  # pyright: ignore[reportPrivateUsage]
    return _screen_is(app, "_conv") and bool(rows) and app.focused is bar


def test_screen_probe_tolerates_an_empty_stack(tmp_path: Path) -> None:
    """`app.screen` raises ScreenStackError on an empty stack (a real window
    during startup and screen switches; CI's Python 3.14.7 scheduling hit it
    inside a poll lambda). The probe reads it as "not yet"."""
    app = Agent6TUI(tmp_path)  # never run: the screen stack is empty
    with pytest.raises(ScreenStackError):
        _ = app.screen
    assert _screen_is(app, "_conv") is False
    # The app's own probe (its tick paths dispatch prompts and retitle
    # through it) answers None instead of raising.
    assert app._screen_or_none() is None  # pyright: ignore[reportPrivateUsage]


def test_live_run_still_gets_the_inline_approval(tmp_path: Path) -> None:
    # The converse: gating on liveness must not cost the live run its approval row.
    d = tmp_path / "blocked1"
    _mk_blocked(d, alive=True)

    async def scenario() -> None:
        app = Agent6TUI(d)
        async with app.run_test(size=(140, 40)) as pilot:
            await _wait_for(pilot, lambda: _approval_ready(app), "the approval row")

    asyncio.run(scenario())


def test_answer_after_death_reports_instead_of_writing(tmp_path: Path) -> None:
    """The worker dies while the approval row is open: the row is withdrawn
    (an answer would reach nothing), the prompt stays visible as a fact, and a
    key press writes no answer file for the next resume to drop."""
    d = tmp_path / "dies-mid-modal"
    _mk_blocked(d, alive=True)

    async def scenario() -> None:
        app = Agent6TUI(d)
        async with app.run_test(size=(140, 40)) as pilot:
            await _wait_for(pilot, lambda: _approval_ready(app), "the approval row")
            (d / "worker.pid").write_text("999999999", encoding="utf-8")
            app._heartbeat_at = 0.0
            app._tick()
            await _wait_for(pilot, lambda: app.worker_lost, "the dead-worker probe")
            app._conv._poll()  # pyright: ignore[reportPrivateUsage]
            await pilot.pause()
            await pilot.press("a")
            await pilot.pause()
            assert not (d / "approvals" / "ap1.answer").exists()
            assert not app._conv.query(ApprovalRow)  # pyright: ignore[reportPrivateUsage]
            item = app._conv.query_one("#conv-approval", Static)  # pyright: ignore[reportPrivateUsage]
            assert "approval pending when the run ended" in str(item.render())

    asyncio.run(scenario())


def test_exit_on_end_holds_over_a_ghost_prompt_and_ctrl_q_leaves(tmp_path: Path) -> None:
    """A dead run's ghost prompt once pinned the auto-spawned dashboard open
    forever with no explanation. The end now HOLDS deliberately instead: the
    header names the state and the leave key, and Ctrl+Q closes."""
    d = tmp_path / "ghost2"
    _mk_blocked(d, alive=False)

    async def scenario() -> None:
        app = Agent6TUI(d, exit_on_end=True)
        async with app.run_test(size=(140, 40)) as pilot:
            await _wait_for(pilot, lambda: app._end_hold, "the end hold")
            assert "ctrl+q to leave" in app.sub_title
            assert app.is_running
            await pilot.press("ctrl+q")
            deadline = time.monotonic() + 5.0
            while app.is_running and time.monotonic() < deadline:
                await asyncio.sleep(0.05)
            assert not app.is_running, "ctrl+q did not leave the held dashboard"

    asyncio.run(scenario())


def test_finished_run_holds_the_dashboard_until_the_user_leaves(tmp_path: Path) -> None:
    """The payoff (green verify, diff, cost) vanished exactly when the user
    was looking at it: exit_on_end tore the TUI down on session.end and dumped
    plain text to the shell. The dashboard now holds, the header says how to
    leave, and the composer still routes a typed follow-up to resume."""
    d = tmp_path / "done1"
    d.mkdir(parents=True)
    evs = [
        {"type": "session.start", "session_id": d.name, "mode": "run", "user_task": "t"},
        {"type": "session.end", "reason": "finish_session", "iterations": 2, "all_passed": True},
    ]
    (d / "logs.jsonl").write_text("".join(json.dumps(e) + "\n" for e in evs), encoding="utf-8")

    async def scenario() -> None:
        app = Agent6TUI(d, exit_on_end=True)
        async with app.run_test(size=(140, 40)) as pilot:
            await _wait_for(pilot, lambda: app._end_hold, "the end hold")
            assert app.is_running
            # The hold leads with the hub's own status word ("passed" here).
            assert "passed" in app.sub_title and "ctrl+q to leave" in app.sub_title
            # A screen stamping its title AFTER the hold began (the mount /
            # tick order is load-dependent) must not wipe the hold: titles are
            # computed at stamp time, not frozen at construction.
            app._conv.on_screen_resume()
            assert "passed" in app.sub_title and "ctrl+q to leave" in app.sub_title
            assert "· t ·" in app.sub_title, "the live task name, not the dir fallback"
            await _open_dash(app, pilot)
            assert app._dash.query_one("#dash-input", SteerInput).border_title == (
                "continue this session"
            )

    asyncio.run(scenario())


def test_a_finished_log_is_read_before_the_first_tick(tmp_path: Path) -> None:
    """A finished run takes its status from the log, not from the fold the
    reader thread has not filled yet: in between it reads as killed (no worker,
    no end), and the exit_on_end hold stamped "stale" over a run that passed."""
    d = tmp_path / "seeded"
    d.mkdir(parents=True)
    evs = [
        {"type": "session.start", "session_id": d.name, "mode": "run", "user_task": "t"},
        {"type": "session.end", "reason": "finish_session", "iterations": 1, "all_passed": True},
    ]
    (d / "logs.jsonl").write_text("".join(json.dumps(e) + "\n" for e in evs), encoding="utf-8")

    app = Agent6TUI(d, exit_on_end=True)
    app._seed_from_disk()
    assert app.dir_status[0] == "passed"
    # Mid-catch-up: the reader has delivered session.start and nothing else, so
    # the status turns "stale" (started, no worker, no end folded yet). The run
    # is not lost, it is unread.
    app.state = apply_event(app.state, evs[0])
    app._refresh_dir_status()
    assert app.dir_status[0] == "stale"
    assert not app.worker_lost, "the fold has not reached the end the log already holds"
    app.state = apply_event(app.state, evs[1])
    app._refresh_dir_status()
    assert app.dir_status[0] == "passed"
    assert not app.worker_lost


def test_dead_pane_hints_point_at_controls_that_exist(tmp_path: Path) -> None:
    """The dead/parked/created hints said "press r to resume", but the r
    binding was removed (no plain-letter shortcuts) and the composer holds
    focus, so pressing r typed the letter into the box. Point at the
    composer's Enter, the action that exists."""
    d = tmp_path / "crashed-hint"
    _mk_crashed(d)

    async def scenario() -> None:
        app = Agent6TUI(d)
        async with app.run_test(size=(140, 40)) as pilot:
            await _open_dash(app, pilot)
            await _wait_for(pilot, lambda: app.worker_lost, "the dead-worker probe")
            app._tick()
            await pilot.pause()
            body = str(app._dash.query_one("#stream-body", Static).render())
            assert "press r" not in body
            assert "Enter resumes" in body

    asyncio.run(scenario())


def test_waiting_run_pane_says_waiting_not_working(tmp_path: Path) -> None:
    """A run blocked on an unanswered prompt read "waiting · needs answer" on
    the top line while the stream pane ticked a live "worker working…"
    spinner beside it -- two lines, two claims. The pane now says what the
    run is doing: waiting on the operator."""
    d = tmp_path / "blocked-pane"
    _mk_blocked(d, alive=True)

    async def scenario() -> None:
        app = Agent6TUI(d)
        async with app.run_test(size=(140, 40)) as pilot:
            # Deny the inline approval (d writes only the bridge file;
            # no answer EVENT lands, so the fold keeps the run "waiting").
            await _wait_for(pilot, lambda: _approval_ready(app), "the approval row")
            await pilot.press("d")
            await _open_dash(app, pilot)
            await _wait_for(pilot, lambda: app.dir_status[0] == "waiting", "the waiting word")
            app._tick()
            await pilot.pause()
            body = str(app._dash.query_one("#stream-body", Static).render())
            assert "waiting for your answer" in body
            assert "working…" not in body

    asyncio.run(scenario())


def test_prompt_and_answer_events_update_the_chip_immediately(tmp_path: Path) -> None:
    """The header chip flips on the prompt/answer event itself, never a
    heartbeat later. Filmed on the dashboard: the log pane already showed
    approval.answer + verify.end while the chip still read "waiting · needs
    answer" -- the synchronous dir-status refresh covered only session
    boundaries, so the chip (and both composer bars) lagged the fold by up to
    ~1s. Asserted with NO awaits between the event and the read, so the
    heartbeat cannot mask the regression."""
    d = tmp_path / "live1"
    d.mkdir(parents=True)
    evs = [
        {"type": "session.start", "session_id": d.name, "mode": "run", "user_task": "t"},
        {"type": "role.call", "role": "worker", "model": "m", "provider": "p"},
    ]
    (d / "logs.jsonl").write_text("".join(json.dumps(e) + "\n" for e in evs), encoding="utf-8")
    (d / "worker.pid").write_text(str(os.getpid()), encoding="utf-8")  # a live worker

    async def scenario() -> None:
        app = Agent6TUI(d)
        async with app.run_test(size=(140, 40)) as pilot:
            await _open_dash(app, pilot)
            assert app.dir_status[1] != "needs answer"
            # The prompt arrives: the chip must say so NOW (no pause between).
            app._handle_event({"type": "approval.prompt", "id": "approval-1", "prompt": "run x?"})
            assert app.dir_status == ("waiting", "needs answer")
            # The answer lands: the chip must clear NOW.
            app._handle_event({"type": "approval.answer", "id": "approval-1", "approved": True})
            assert app.dir_status[1] != "needs answer"

    asyncio.run(scenario())


def test_dashboard_header_says_where_the_changes_are(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The header carries the run's branch line (the web header's and
    `sessions show`'s wording): the run branch, once its first commit created
    it, and the base a merge lands on; reopened after the merge stamp lands,
    the branch merged."""
    repo = tmp_path / "repo"
    repo.mkdir()
    git = ["git", "-C", str(repo)]
    env = {"GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t"}
    env["GIT_COMMITTER_EMAIL"] = "t@t"
    subprocess.run([*git, "init", "-q", "-b", "main"], check=True)
    subprocess.run([*git, "commit", "-q", "--allow-empty", "-m", "base"], check=True, env=env)
    monkeypatch.chdir(repo)  # the dashboard reads the branch facts from the cwd checkout
    d = tmp_path / "branched"
    d.mkdir()
    evs = [
        {"type": "session.start", "session_id": d.name, "mode": "run", "user_task": "t"},
        {"type": "session.end", "all_passed": True, "reason": "finish_session"},
    ]
    (d / "logs.jsonl").write_text("".join(json.dumps(e) + "\n" for e in evs), encoding="utf-8")
    manifest: dict[str, Any] = {
        "mode": "run",
        "run_branch": "agent6/branched",
        "base_branch": "main",
    }
    (d / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    async def header() -> str:
        app = Agent6TUI(d)
        async with app.run_test(size=(140, 40)) as pilot:
            await _open_dash(app, pilot)
            return str(app._dash.query_one("#top", Static).render())

    assert "branch:" not in asyncio.run(header())  # no commit yet: no branch to name
    subprocess.run([*git, "branch", "agent6/branched"], check=True)
    assert "branch: agent6/branched → merges into main" in asyncio.run(header())
    tip = subprocess.run(
        [*git, "rev-parse", "agent6/branched"], check=True, capture_output=True, text=True
    ).stdout.strip()
    manifest["merged"] = {"into": "main", "sha": tip, "tip": tip}
    (d / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    assert "branch: agent6/branched (merged into main)" in asyncio.run(header())  # a reopen


def test_dashboard_header_says_what_the_run_serves(tmp_path: Path) -> None:
    """A dev server the agent started is reachable only through `agent6
    forward`; the header names the port and that command (the web header's
    and `sessions show`'s line)."""
    import os
    import socket

    from agent6.sessions.ipc import write_session_netns_pid

    d = tmp_path / "serving"
    d.mkdir()
    evs = [{"type": "session.start", "session_id": d.name, "mode": "run", "user_task": "t"}]
    (d / "logs.jsonl").write_text("".join(json.dumps(e) + "\n" for e in evs), encoding="utf-8")
    with socket.socket() as srv:
        srv.bind(("127.0.0.1", 0))
        srv.listen(1)
        port = srv.getsockname()[1]
        write_session_netns_pid(d, os.getpid())

        async def header() -> str:
            app = Agent6TUI(d)
            async with app.run_test(size=(140, 40)) as pilot:
                await _open_dash(app, pilot)
                return str(app._dash.query_one("#top", Static).render())

        top = asyncio.run(header())
    # This process stands in for the network holder, so the host's own listeners
    # show too: the test socket is among them, and the forward line names one.
    serving = next(line for line in top.splitlines() if line.startswith("serving: "))
    assert str(port) in serving and "· agent6 forward serving " in serving


def test_a_parked_sessions_empty_view_names_the_reason() -> None:
    """The dashboard row said "parked · uncommitted changes" while the opened
    conversation said only "(no conversation yet)"; the placeholder now
    carries the parked reason and the way forward."""
    from agent6.ui.tui.conversation import empty_conversation_note

    note = empty_conversation_note("parked", "uncommitted changes", ended=False)
    assert "parked" in note and "uncommitted changes" in note and "type below" in note
    assert empty_conversation_note("parked", "", ended=False).startswith("(parked ")
    assert empty_conversation_note("", "", ended=True) == "this run made no conversation"
    assert "appears as the run streams" in empty_conversation_note("", "", ended=False)
