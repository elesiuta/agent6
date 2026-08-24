# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Headless drive of the textual dashboard via textual's run_test Pilot.

textual ships in the base install, so these run in CI. They cover the bits
that previously could only be checked by a human: that streamed reasoning +
markup-hostile model output render without crashing, that the approval modal
is keyboard-answerable (the y/n routing bug), and that the new Ctrl-C steer
modal writes the right bridge file.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from pathlib import Path
from typing import Any

from textual.app import App, ScreenStackError
from textual.widgets import Button, DataTable, Input, RichLog, Static, TextArea, Tree

from agent6.ui.tui.app import Agent6TUI
from agent6.ui.tui.composer import ApprovalRow
from agent6.ui.tui.modals import (
    ApprovalModal,
    QuestionModal,
    ToolCallDetailModal,
)
from agent6.viewmodel.state import Question


def _ev(**fields: Any) -> dict[str, object]:
    return dict(fields)


def _screen_is(app: Agent6TUI, name: str) -> bool:
    """`app.screen` raises while the stack is transiently empty (startup,
    mid-switch); a poll reads that as "not yet", never an error."""
    try:
        current = app.screen
    except ScreenStackError:
        return False
    return current is getattr(app, name)


async def _wait_for(pilot: Any, cond: Any, what: str, timeout: float = 10.0) -> None:
    """Deadline-based condition wait. Iteration-capped pause loops spin through
    in milliseconds under load while the awaited work lags behind, then fall
    through silently and fail at some later assert; a wall-clock deadline with
    a loud timeout fails at the wait that actually missed."""
    deadline = time.monotonic() + timeout
    while not cond():
        assert time.monotonic() < deadline, f"timed out waiting for {what}"
        await pilot.pause(0.05)


async def _show_dashboard(pilot: Any) -> None:
    """The app opens on the conversation view; flip to the dashboard (Ctrl+D)
    so the pane tests drive the dashboard like before. Waits for each screen to
    actually be on top: startup pushes the screens asynchronously, and a Ctrl+D
    fired before the conversation lands would type into the wrong screen."""
    app = pilot.app
    await _wait_for(pilot, lambda: _screen_is(app, "_conv"), "the conversation screen")
    await pilot.press("ctrl+d")
    await _wait_for(pilot, lambda: _screen_is(app, "_dash"), "the dashboard screen")


async def _settle_focus(pilot: Any, widget: Any) -> None:
    """Wait for a deferred focus() (Widget.focus defers via call_later) to land."""
    await _wait_for(pilot, lambda: pilot.app.focused is widget, f"focus on {widget}")


def test_question_modal_digit_in_freetext_is_not_hijacked() -> None:
    """A digit typed into an answer field is plain text: the multi-question modal
    has no digit quick-select. An option button fills its question's field (never
    dismisses); ctrl+s submits the collected answers as a tuple."""
    result: dict[str, tuple[str, ...] | None] = {}

    class _Host(App[None]):
        def on_mount(self) -> None:
            self.push_screen(
                QuestionModal("q1", (Question(question="pick?", options=("alpha", "beta")),)),
                lambda v: result.__setitem__("v", v),
            )

    async def scenario() -> None:
        app = _Host()
        async with app.run_test() as pilot:
            await pilot.pause()
            modal = app.screen
            assert isinstance(modal, QuestionModal)
            modal.query_one("#ans-0", Input).focus()
            await pilot.pause()
            await pilot.press("2")  # a digit is just text now, not an option pick
            await pilot.pause()
            assert isinstance(app.screen, QuestionModal)  # still open (no digit-select)
            assert "v" not in result
            assert modal.query_one("#ans-0", Input).value == "2"  # digit typed as text
            # An option button fills that question's field; it does not dismiss.
            modal.query_one("#opt-0-0", Button).press()
            await pilot.pause()
            assert isinstance(app.screen, QuestionModal)  # still open (fill, not submit)
            assert modal.query_one("#ans-0", Input).value == "alpha"  # filled from the option
            await pilot.press("ctrl+s")  # submit collects the answers
            await pilot.pause()
            assert result.get("v") == ("alpha",)  # tuple of answers, aligned to questions

    asyncio.run(scenario())


def test_modal_arrow_keys_move_focus() -> None:
    """Arrow keys move focus in a modal like Tab (the app.focus_next fix). Tested
    on the button-only approval dialog, where no text field consumes the arrows."""

    class _Host(App[None]):
        def on_mount(self) -> None:
            self.push_screen(ApprovalModal("a", "allow?"), lambda _v: None)

    async def scenario() -> None:
        app = _Host()
        async with app.run_test() as pilot:
            await pilot.pause()
            modal = app.screen
            assert isinstance(modal, ApprovalModal)
            first = modal.focused
            assert isinstance(first, Button)
            await pilot.press("right")  # arrow moves focus to the other button
            await pilot.pause()
            assert isinstance(modal.focused, Button) and modal.focused is not first
            await pilot.press("left")  # and back
            await pilot.pause()
            assert modal.focused is first

    asyncio.run(scenario())


def test_tools_table_maximizes_to_full_height(tmp_path: Path) -> None:
    """Pressing `f` on the focused tool table fills the screen, not its 20%
    resting height -- regression: the explicit `height: 20%` made the maximized
    view stay short until `#tools.-maximized { height: 1fr; }` was added."""
    (tmp_path / "logs.jsonl").write_text("", encoding="utf-8")

    async def scenario() -> None:
        app = Agent6TUI(tmp_path)
        async with app.run_test(size=(100, 40)) as pilot:
            await _show_dashboard(pilot)
            app._handle_event(_ev(type="tool.call", name="grep", args={"pattern": "x"}))
            app._tick()
            await pilot.pause()
            table = app._dash.query_one("#tools", DataTable)
            table.focus()
            await _settle_focus(pilot, table)
            resting_h = table.size.height
            app._dash.action_fullscreen()  # View menu / palette action (no bare letter)
            await pilot.pause()
            assert app.screen.maximized is table
            assert table.has_class("-maximized")
            maxed_h = table.size.height
            assert maxed_h > resting_h * 2  # fills the screen, not the 20% resting band
            assert maxed_h >= 25  # ~full of a 40-row screen (minus menu + footer)

    asyncio.run(scenario())


def test_plan_tree_maximizes_to_full_width(tmp_path: Path) -> None:
    """Pressing `f` on the focused task-graph pane fills the screen WIDTH, not its
    32% resting width -- the width analogue of the tool-table height bug; the
    explicit `width: 32%` made the maximized view stay a narrow column until
    `#plan.-maximized { width: 1fr; }` was added."""
    (tmp_path / "logs.jsonl").write_text("", encoding="utf-8")

    async def scenario() -> None:
        app = Agent6TUI(tmp_path)
        async with app.run_test(size=(100, 40)) as pilot:
            await _show_dashboard(pilot)
            app._handle_event(
                _ev(
                    type="graph.update",
                    cursor="t1",
                    nodes={
                        "t1": {
                            "title": "do the thing",
                            "parent_id": None,
                            "status": "in_progress",
                            "children": [],
                        }
                    },
                )
            )
            app._tick()
            await pilot.pause()
            tree = app._dash.query_one("#plan", Tree)
            tree.focus()
            await _settle_focus(pilot, tree)
            resting_w = tree.size.width
            app._dash.action_fullscreen()  # View menu / palette action (no bare letter)
            await pilot.pause()
            assert app.screen.maximized is tree
            assert tree.has_class("-maximized")
            maxed_w = tree.size.width
            assert maxed_w > resting_w * 2  # fills the screen, not the 32% resting column
            assert maxed_w >= 90  # ~full of a 100-col screen

    asyncio.run(scenario())


def test_tool_row_enter_opens_detail_with_full_args(tmp_path: Path) -> None:
    """Enter on a tool-calls row opens a read-only detail modal carrying the FULL
    arg value, not the column-truncated preview."""
    (tmp_path / "logs.jsonl").write_text("", encoding="utf-8")
    long_val = "abc/" * 100  # 400 chars, well past the 80-char preview + 90-char column

    async def scenario() -> None:
        app = Agent6TUI(tmp_path)
        async with app.run_test() as pilot:
            await _show_dashboard(pilot)
            app._handle_event(_ev(type="tool.call", name="run_command", args={"cmd": long_val}))
            app._handle_event(
                _ev(type="tool.result", name="run_command", ok=True, summary="lots of output")
            )
            app._tick()
            await pilot.pause()
            app._dash.query_one("#tools", DataTable).focus()
            await pilot.pause()
            await pilot.press("enter")  # select the single row
            await pilot.pause()
            assert isinstance(app.screen, ToolCallDetailModal)
            args_text = app.screen.query_one("#tc-args", TextArea).text
            assert long_val in args_text  # full value present, not the "…" preview
            await pilot.press("escape")  # closes past the focused read-only TextArea
            await pilot.pause()
            assert not isinstance(app.screen, ToolCallDetailModal)

    asyncio.run(scenario())


def test_render_and_modals(tmp_path: Path) -> None:
    # The app suppresses live affordances for a session whose worker is gone;
    # these drive a LIVE run, so its pid belongs on disk as it would be.
    (tmp_path / "worker.pid").write_text(str(os.getpid()), encoding="utf-8")
    (tmp_path / "logs.jsonl").write_text("", encoding="utf-8")

    async def scenario() -> None:
        app = Agent6TUI(tmp_path)
        async with app.run_test() as pilot:
            await _show_dashboard(pilot)
            # Render with bracket-laden (markup-hostile) content must not crash —
            # exercises the header, the plan TREE (step titles), the tool TABLE
            # (names/args), the stream pane and the diff pane, all of which carry
            # model output that would otherwise be parsed as Rich markup.
            for ev in (
                _ev(type="session.start", user_task="do [a] thing", mode="run"),
                _ev(
                    type="graph.update",
                    cursor="t1",
                    nodes={
                        "t1": {
                            "title": "fix [the] bug",
                            "parent_id": None,
                            "status": "in_progress",
                            "children": ["t2"],
                        },
                        "t2": {
                            "title": "add [/close] tag",
                            "parent_id": "t1",
                            "status": "pending",
                            "children": [],
                        },
                    },
                ),
                _ev(type="role.call", role="worker", model="kimi-k2.6"),
                _ev(type="role.thinking_delta", role="worker", text="let me [check]"),
                _ev(type="role.text_delta", role="worker", text="answer is [x]"),
                _ev(type="tool.call", name="grep", args={"pattern": "[a-z]+", "path": "x.py"}),
                _ev(type="tool.result", name="grep", ok=True, summary="3 matches in [src]"),
                _ev(type="diff.updated", index=1, patch="--- a\n+++ b\n@@ [x] @@"),
            ):
                app._handle_event(ev)
            app._tick()  # the coalesced repaint happens in the tick, not per event
            await pilot.pause()

            # Approval modal: keyboard 'y' must reach the modal (routing bug).
            app._handle_event(_ev(type="approval.prompt", id="ap1", prompt="run_command(['ls'])"))
            app._tick()
            await pilot.pause()
            assert isinstance(app.screen, ApprovalModal)
            await pilot.press("y")
            await pilot.pause()
            assert (tmp_path / "approvals" / "ap1.answer").read_text(encoding="utf-8") == "yes"

            # Approval modal: keyboard 'n'.
            app._handle_event(_ev(type="approval.prompt", id="ap2", prompt="rm -rf"))
            app._tick()
            await pilot.pause()
            await pilot.press("n")
            await pilot.pause()
            assert (tmp_path / "approvals" / "ap2.answer").read_text(encoding="utf-8") == "no"

            # An external steer request routes to the docked composer bar (no
            # popup): the bar takes focus, typing + Enter answers over the bridge.
            from agent6.ui.tui.composer import SteerInput

            app._handle_event(_ev(type="session.steer_requested", source="sigint"))
            app._tick()
            bar = app._dash.query_one("#dash-input", SteerInput)
            await _settle_focus(pilot, bar)
            assert app.focused is bar
            await pilot.press("f", "i", "x")
            await pilot.press("enter")
            await pilot.pause()
            assert (tmp_path / "steer.answer").read_text(encoding="utf-8") == "fix"

            # Question modal (ask_user): markup-hostile options render; clicking an
            # option fills its answer field, and ctrl+s writes the bridge file (a
            # JSON list of answers aligned to the questions).
            app._handle_event(
                _ev(
                    type="question.prompt",
                    id="q1",
                    questions=[
                        {"question": "which [approach]?", "options": ["use [A]", "use [B]"]}
                    ],
                )
            )
            app._tick()
            await pilot.pause()
            assert isinstance(app.screen, QuestionModal)
            app.screen.query_one("#opt-0-1", Button).press()  # fill ans-0 with the 2nd option
            await pilot.pause()
            await pilot.press("ctrl+s")  # submit
            await pilot.pause()
            assert (tmp_path / "questions" / "q1.answer").read_text(encoding="utf-8") == json.dumps(
                ["use [B]"]
            )

            # Question modal: a typed free-text answer is sent verbatim.
            app._handle_event(
                _ev(
                    type="question.prompt",
                    id="q2",
                    questions=[{"question": "name?", "options": []}],
                )
            )
            app._tick()
            await pilot.pause()
            await pilot.press("z", "z")
            await pilot.press("enter")
            await pilot.pause()
            assert (tmp_path / "questions" / "q2.answer").read_text(encoding="utf-8") == json.dumps(
                ["zz"]
            )

    asyncio.run(scenario())


def test_start_question_before_session_start_is_answerable(tmp_path: Path) -> None:
    """A run asks about the working tree's uncommitted changes BEFORE
    session.start (the same channel as ask_user); a TUI opened on that dir
    reads it as waiting, shows the question, and answers over the bridge."""
    (tmp_path / "worker.pid").write_text(str(os.getpid()), encoding="utf-8")
    (tmp_path / "manifest.json").write_text(
        json.dumps({"mode": "run", "session_id": tmp_path.name, "user_task": "t"}),
        encoding="utf-8",
    )
    (tmp_path / "logs.jsonl").write_text(
        json.dumps(
            {
                "type": "question.prompt",
                "id": "question-1",
                "questions": [
                    {
                        "question": "1 tracked file has uncommitted changes:\n    seed.txt\n"
                        "How should this run treat them?",
                        "options": ["stash", "include", "cancel"],
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    async def scenario() -> None:
        app = Agent6TUI(tmp_path)
        async with app.run_test(size=(120, 40)) as pilot:
            await _wait_for(
                pilot,
                lambda: (
                    isinstance(app.screen, QuestionModal) and bool(app.screen.query("#opt-0-0"))
                ),
                "the question modal",
            )
            app._heartbeat_at = 0.0
            app._tick()
            assert app.dir_status == ("waiting", "needs answer")
            app.screen.query_one("#opt-0-0", Button).press()  # stash
            await pilot.pause()
            await pilot.press("ctrl+s")
            await pilot.pause()
            assert (tmp_path / "questions" / "question-1.answer").read_text(
                encoding="utf-8"
            ) == json.dumps(["stash"])

    asyncio.run(scenario())


def test_back_and_quit_exit_codes(tmp_path: Path) -> None:
    """Esc leaves the run view for the hub (exit 0) from both the conversation and
    the dashboard (their composer bars own plain letters, so there is no q alias);
    Ctrl+Q quits the hub (QUIT_HUB_CODE) from anywhere. Standalone, every one of
    them just closes (0)."""
    from agent6.ui.tui.app import QUIT_HUB_CODE

    async def press(from_hub: bool, *keys: str) -> int | None:
        (tmp_path / "logs.jsonl").write_text("", encoding="utf-8")
        app = Agent6TUI(tmp_path, from_hub=from_hub)
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()
            for key in keys:
                await pilot.press(key)
                await pilot.pause()
        return app.return_value

    assert asyncio.run(press(True, "escape")) == 0  # conversation Esc -> back to the hub
    assert asyncio.run(press(True, "ctrl+d", "escape")) == 0  # dashboard Esc -> the hub
    assert asyncio.run(press(True, "ctrl+q")) == QUIT_HUB_CODE  # only Ctrl+Q quits the hub
    assert asyncio.run(press(False, "escape")) == 0  # standalone: just close
    assert asyncio.run(press(False, "ctrl+q")) == 0  # standalone: just close


def test_dashboard_pane_maximize_and_restore(tmp_path: Path) -> None:
    """f maximizes the focused pane to full screen; Esc and f both restore it. Esc
    while maximized must minimize (not also back out to the hub), and a non-default
    pane like the diff must be focusable for this to work."""

    async def scenario() -> None:
        (tmp_path / "logs.jsonl").write_text("", encoding="utf-8")
        app = Agent6TUI(tmp_path, from_hub=True)
        async with app.run_test(size=(120, 40)) as pilot:
            await _show_dashboard(pilot)
            await pilot.pause()
            diff = app._dash.query_one("#diff")
            diff.focus()
            await _settle_focus(pilot, diff)
            app._dash.action_fullscreen()  # maximize the focused pane
            await pilot.pause()
            assert app.screen.maximized is diff
            await pilot.press("escape")  # Esc restores, does NOT exit to the hub
            await pilot.pause()
            assert app.screen.maximized is None
            assert app.return_value is None  # still running; Esc was consumed by minimize
            diff.focus()
            await _settle_focus(pilot, diff)
            app._dash.action_fullscreen()
            await pilot.pause()
            assert app.screen.maximized is diff
            app._dash.action_fullscreen()  # the action toggles back too
            await pilot.pause()
            assert app.screen.maximized is None

    asyncio.run(scenario())


def test_dashboard_diff_pane_scrolls(tmp_path: Path) -> None:
    """A long diff overflows the diff pane, which is a scroll container, so it can be
    scrolled -- inline and while maximized (regression: it used to be a plain Static
    that just clipped)."""

    async def scenario() -> None:
        (tmp_path / "logs.jsonl").write_text("", encoding="utf-8")
        app = Agent6TUI(tmp_path)
        async with app.run_test(size=(120, 40)) as pilot:
            await _show_dashboard(pilot)
            await pilot.pause()
            long_patch = "--- a\n+++ b\n" + "".join(f"+added line {i}\n" for i in range(200))
            app._handle_event(_ev(type="diff.updated", index=1, patch=long_patch))
            app._tick()
            await pilot.pause()
            diff = app._dash.query_one("#diff")
            assert diff.max_scroll_y > 0  # content overflows the pane -> scrollable
            diff.focus()
            await _settle_focus(pilot, diff)  # focus() defers; land it before maximizing
            app._dash.action_fullscreen()  # maximize, then it must still scroll
            await pilot.pause()
            assert app.screen.maximized is diff
            assert diff.max_scroll_y > 0

    asyncio.run(scenario())


def test_dashboard_inline_log_is_a_bounded_gapless_window(tmp_path: Path) -> None:
    """Coalescing folds many events between paints. The inline log must stay a bounded
    window: feed a pre-burst, then a burst larger than the window in one tick, and the
    RichLog caps at MAX_LOG_TAIL -- the gap-causing pre-burst lines are evicted, so it
    is the gapless recent window, not pre-burst lines + a hole + the tail."""
    from agent6.viewmodel.state import MAX_LOG_TAIL

    async def scenario() -> None:
        (tmp_path / "logs.jsonl").write_text("", encoding="utf-8")
        app = Agent6TUI(tmp_path)
        async with app.run_test(size=(120, 40)) as pilot:
            await _show_dashboard(pilot)
            await pilot.pause()
            for i in range(100):
                app._handle_event(_ev(type="tool.call", name=f"pre{i}"))
            app._tick()
            await pilot.pause()
            for i in range(MAX_LOG_TAIL + 100):
                app._handle_event(_ev(type="tool.call", name=f"burst{i}"))
            app._tick()
            await pilot.pause()
            log = app._dash.query_one("#log", RichLog)
            assert len(log.lines) == MAX_LOG_TAIL  # bounded; pre-burst lines evicted

    asyncio.run(scenario())


def test_conversation_and_dashboard_footers_match(tmp_path: Path) -> None:
    """The two run views share one shortcut scheme: the same footer entries in
    the same order, Ctrl+D leftmost (only its label differs: Dashboard vs
    Conversation), and no plain-letter keys (the composer bars own letters).
    One deliberate extra on the conversation: ^t Detail (the transcript's
    detail cycle -- the dashboard has no transcript)."""
    from textual.widgets._footer import FooterKey

    async def scenario() -> None:
        (tmp_path / "logs.jsonl").write_text("", encoding="utf-8")
        app = Agent6TUI(tmp_path, from_hub=True)
        async with app.run_test(size=(120, 30)) as pilot:
            await pilot.pause()
            conv = [(fk.key_display, fk.description) for fk in app.screen.query(FooterKey)]
            await pilot.press("ctrl+d")
            await pilot.pause()
            dash = [(fk.key_display, fk.description) for fk in app.screen.query(FooterKey)]
            shared = [(k, d) for k, d in conv if d != "Detail"]
            assert [k for k, _ in shared] == [k for k, _ in dash]  # same keys, same order
            assert conv[0][0] == "^d" and conv[0][1] == "Dashboard"  # leftmost toggle
            assert dash[0][1] == "Conversation"
            toggles = ("Dashboard", "Conversation")
            assert [lbl for _, lbl in shared if lbl not in toggles] == [
                lbl for _, lbl in dash if lbl not in toggles
            ]
            # No plain single-letter shortcuts on either view.
            assert all(len(k) > 1 for k, _ in conv + dash)

    asyncio.run(scenario())


def test_dashboard_claims_are_per_process(tmp_path: Path) -> None:
    """Concurrent front-ends hold independent claims: mounting registers OUR
    claim without touching a live peer's, and unmount removes only ours -- the
    single-slot frontend.pid (where one viewer's exit could deregister
    another) is gone."""
    import os
    import subprocess
    import sys

    (tmp_path / "logs.jsonl").write_text("", encoding="utf-8")
    peer = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        from agent6.sessions.ipc import frontend_is_live, register_frontend

        register_frontend(tmp_path, peer.pid)  # a live web viewer's claim

        async def scenario() -> None:
            app = Agent6TUI(tmp_path)
            async with app.run_test() as pilot:
                await pilot.pause()
                assert (tmp_path / "frontends" / str(os.getpid())).exists()  # ours
                assert (tmp_path / "frontends" / str(peer.pid)).exists()  # peer intact
            # Unmount removed only our claim; the peer keeps bridging.
            assert not (tmp_path / "frontends" / str(os.getpid())).exists()
            assert (tmp_path / "frontends" / str(peer.pid)).exists()
            assert frontend_is_live(tmp_path)

        asyncio.run(scenario())
    finally:
        peer.kill()
        peer.wait()


def test_dead_peer_claim_does_not_mask_the_live_dashboard(tmp_path: Path) -> None:
    """A hard-killed viewer's stale claim must not affect liveness: the probe
    reads the dashboard's own live claim (and prunes the dead one)."""
    import os

    (tmp_path / "logs.jsonl").write_text("", encoding="utf-8")
    from agent6.sessions.ipc import frontend_is_live, register_frontend

    register_frontend(tmp_path, 999999999)  # dead

    async def scenario() -> None:
        app = Agent6TUI(tmp_path)
        async with app.run_test() as pilot:
            await pilot.pause()
            assert frontend_is_live(tmp_path)  # our claim carries it
            assert not (tmp_path / "frontends" / "999999999").exists()  # pruned
            assert (tmp_path / "frontends" / str(os.getpid())).exists()

    asyncio.run(scenario())


def test_resume_reopens_the_approval_for_a_reused_prompt_id(tmp_path: Path) -> None:
    """`agent6 resume` appends a new session whose prompt ids restart at
    approval-1; a dashboard held across the resume must pop the new session's
    modal, not swallow it as already seen."""
    # The app suppresses live affordances for a session whose worker is gone;
    # these drive a LIVE run, so its pid belongs on disk as it would be.
    (tmp_path / "worker.pid").write_text(str(os.getpid()), encoding="utf-8")
    (tmp_path / "logs.jsonl").write_text("", encoding="utf-8")

    async def scenario() -> None:
        app = Agent6TUI(tmp_path)
        async with app.run_test() as pilot:
            app._handle_event(_ev(type="session.start", user_task="session one", mode="run"))
            app._handle_event(_ev(type="approval.prompt", id="approval-1", prompt="first?"))
            app._tick()
            app._conv._poll()
            await pilot.pause()
            assert app._conv.query(ApprovalRow)
            await pilot.press("a")
            await pilot.pause()
            app._handle_event(_ev(type="approval.answer", id="approval-1", approved=True))
            # The resume: a real resumed leg emits ONLY loop.resume.start (never
            # a second session.start -- workflows/loop.py run() vs resume()), then
            # the new session's approval-1. Feeding session.start here masked the
            # bug where the seen-set was cleared only on session.start and every
            # resumed leg's modals were swallowed forever.
            app._handle_event(_ev(type="loop.resume.start", iteration=2, messages=4))
            app._handle_event(_ev(type="approval.prompt", id="approval-1", prompt="again?"))
            app._tick()
            app._conv._poll()
            await pilot.pause()
            assert app._conv.query(ApprovalRow)  # re-shown, not swallowed
            await pilot.press("d")
            await pilot.pause()
            answer = (tmp_path / "approvals" / "approval-1.answer").read_text(encoding="utf-8")
            assert answer == "no"

    asyncio.run(scenario())


def test_steer_request_marker_round_trip(tmp_path: Path) -> None:
    from agent6.sessions.ipc import clear_steer_request, request_steer, steer_request_pending

    assert not steer_request_pending(tmp_path)
    request_steer(tmp_path)
    assert steer_request_pending(tmp_path)  # the run's requested() sees this
    clear_steer_request(tmp_path)
    assert not steer_request_pending(tmp_path)


def test_dashboard_bar_is_default_focus_and_steers(tmp_path: Path) -> None:
    """The dashboard opens ready to type -- the composer bar is the default focus
    (like the conversation) -- and Enter drops the steer.request marker + the
    instruction together, for the run to inject at its next boundary. The run
    must be LIVE (a recorded live worker): a dir with no worker routes the
    composer to resume instead, because a steer file there is never read."""
    import os

    from agent6.sessions.ipc import steer_request_pending, write_worker_pid
    from agent6.ui.tui.composer import SteerInput

    async def scenario() -> None:
        (tmp_path / "logs.jsonl").write_text("", encoding="utf-8")
        write_worker_pid(tmp_path, os.getpid())
        app = Agent6TUI(tmp_path)
        async with app.run_test(size=(100, 30)) as pilot:
            await _show_dashboard(pilot)
            bar = app._dash.query_one("#dash-input", SteerInput)
            await _settle_focus(pilot, bar)
            assert app.focused is bar  # default focus: type at once, no popup
            assert not steer_request_pending(tmp_path)
            await pilot.press("g", "o")
            await pilot.press("enter")
            await pilot.pause()
            assert steer_request_pending(tmp_path)  # marker dropped for the run
            assert (tmp_path / "steer.answer").read_text(encoding="utf-8") == "go"

    asyncio.run(scenario())


def test_finished_run_bar_resumes_with_the_instruction(tmp_path: Path, monkeypatch: Any) -> None:
    """Typing into the composer bar of a FINISHED run spawns a detached
    `agent6 resume --steer=<text>`: the follow-up rides the flag (a pre-seeded
    steer file would be wiped by resume's stale-state clear) and is injected at
    the resumed session's first boundary -- the claude-code follow-up flow."""
    from agent6.ui.tui import app as app_mod
    from agent6.ui.tui.composer import SteerInput

    spawned: list[tuple[str, str]] = []

    def _fake_resume(
        _cwd: Path, rid: str, *, steer: str = "", preset: str = "", config_path: object = None
    ) -> str:
        spawned.append((rid, steer))
        return ""

    monkeypatch.setattr(app_mod, "spawn_detached_resume", _fake_resume)
    (tmp_path / "logs.jsonl").write_text(
        "".join(
            json.dumps(e) + "\n"
            for e in (
                _ev(type="session.start", user_task="x", mode="run"),
                _ev(type="session.end", reason="completed", all_passed=True),
            )
        ),
        encoding="utf-8",
    )

    async def scenario() -> None:
        app = Agent6TUI(tmp_path)
        async with app.run_test(size=(120, 40)) as pilot:
            for _ in range(50):
                await pilot.pause()
                if app.state.finished:
                    break
            await pilot.pause()
            bar = app._conv.query_one("#conv-input", SteerInput)
            assert bar.display  # the primary view keeps the bar after session.end
            assert bar.border_title == "continue this session"  # relabelled for resume
            bar.post_message(SteerInput.Submitted("also add tests"))
            await pilot.pause()
            await pilot.pause()
            # The instruction rides --steer on the detached resume.
            assert spawned == [(tmp_path.name, "also add tests")]

    asyncio.run(scenario())


def test_stop_now_aborts_via_bridge(tmp_path: Path) -> None:
    """Run > Stop now on a LIVE run confirms, then writes an abort over the
    file bridge -- the stream watchdog interrupts the in-flight turn."""
    import os

    from agent6.sessions.ipc import steer_request_pending, write_worker_pid
    from agent6.ui.tui.modals import ConfirmModal

    async def scenario() -> None:
        (tmp_path / "logs.jsonl").write_text("", encoding="utf-8")
        write_worker_pid(tmp_path, os.getpid())
        app = Agent6TUI(tmp_path)
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()
            app.action_stop_now()
            await pilot.pause()
            assert isinstance(app.screen, ConfirmModal)  # confirms before stopping
            await pilot.press("y")  # confirm
            await pilot.pause()
            assert (tmp_path / "steer.answer").read_text(encoding="utf-8") == "abort"
            assert steer_request_pending(tmp_path)

    asyncio.run(scenario())


def test_ctrl_z_on_the_run_spawned_view_detaches_the_run_itself(tmp_path: Path) -> None:
    """The view `agent6 run --tui` spawns fronts a run in the terminal's own
    process; leaving the view alone left that run streaming in the foreground
    (the shell never came back). Ctrl-Z there steers a detach, so the
    lifecycle hands the run to a background resume at its next step; a plain
    viewer (`attach --tui`) leaves the run it watches untouched."""
    import os

    from agent6.sessions.ipc import steer_request_pending, write_worker_pid

    async def spawned_view() -> None:
        (tmp_path / "logs.jsonl").write_text("", encoding="utf-8")
        write_worker_pid(tmp_path, os.getpid())
        app = Agent6TUI(tmp_path, exit_on_end=True)
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()
            await pilot.press("ctrl+z")
            await pilot.pause()
        assert app.detached
        assert (tmp_path / "steer.answer").read_text(encoding="utf-8") == "detach"
        assert steer_request_pending(tmp_path)

    async def viewer() -> None:
        (tmp_path / "steer.answer").unlink()
        (tmp_path / "steer.request").unlink()
        app = Agent6TUI(tmp_path)
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()
            await pilot.press("ctrl+z")
            await pilot.pause()
        assert app.detached
        assert not steer_request_pending(tmp_path)

    asyncio.run(spawned_view())
    asyncio.run(viewer())


def test_stop_after_step_drops_the_marker(tmp_path: Path) -> None:
    """Run > Stop after this step on a LIVE run confirms, then drops the
    stop.request marker the loop honors at its next completed-iteration
    boundary."""
    import os

    from agent6.sessions.ipc import stop_request_pending, write_worker_pid
    from agent6.ui.tui.modals import ConfirmModal

    async def scenario() -> None:
        (tmp_path / "logs.jsonl").write_text("", encoding="utf-8")
        write_worker_pid(tmp_path, os.getpid())
        app = Agent6TUI(tmp_path)
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()
            assert not stop_request_pending(tmp_path)
            app.action_stop_step()
            await pilot.pause()
            assert isinstance(app.screen, ConfirmModal)  # confirms before stopping
            await pilot.press("y")  # confirm
            await pilot.pause()
            assert stop_request_pending(tmp_path)  # marker for the boundary stop
            assert not (tmp_path / "steer.answer").exists()  # no mid-turn abort

    asyncio.run(scenario())


def test_context_pct_readout_in_top_line_and_bar(tmp_path: Path, monkeypatch: Any) -> None:
    """With the model's context window known, the dashboard's top line shows
    `ctx: NN%` and the composer bar's subtitle carries the same readout."""
    from agent6.ui.tui.composer import SteerInput

    def _window(_provider: str, _model: str) -> int:
        return 100_000

    monkeypatch.setattr("agent6.viewmodel.state._window", _window)
    (tmp_path / "logs.jsonl").write_text("", encoding="utf-8")

    async def scenario() -> None:
        app = Agent6TUI(tmp_path)
        async with app.run_test(size=(150, 40)) as pilot:
            await _show_dashboard(pilot)
            app._handle_event(_ev(type="session.start", user_task="x", mode="run"))
            app._handle_event(_ev(type="role.call", role="worker", model="m", provider="p"))
            app._handle_event(
                _ev(
                    type="role.result",
                    role="worker",
                    ok=True,
                    tokens_in=1_000,
                    cache_read=40_000,
                    cache_creation=0,
                )
            )
            app._tick()
            await pilot.pause()
            assert app.context_pct() == 41
            top = str(app._dash.query_one("#top", Static).render())
            assert "ctx: 41%" in top
            bar = app._dash.query_one("#dash-input", SteerInput)
            assert "ctx 41%" in (bar.border_subtitle or "")

    asyncio.run(scenario())


def test_working_timer_anchors_to_the_last_events_ts_not_the_attach(tmp_path: Path) -> None:
    """Replayed history bumped the idle anchor too, so attaching to a run
    wedged 40 minutes read "working… 3s" -- the timer meant to tell "thinking"
    from "hung" concealed the hang. The anchor is the last event's own ts."""

    async def scenario() -> None:
        from agent6.sessions.ipc import write_worker_pid

        (tmp_path / "logs.jsonl").write_text("", encoding="utf-8")
        write_worker_pid(tmp_path, os.getpid())  # alive-but-wedged, not stale
        app = Agent6TUI(tmp_path)
        async with app.run_test(size=(150, 40)) as pilot:
            await _show_dashboard(pilot)
            wedged_at = time.time() - 2400
            app._handle_event(
                _ev(type="session.start", user_task="x", mode="run", ts=wedged_at - 5)
            )
            app._handle_event(_ev(type="role.call", role="worker", model="m", ts=wedged_at))
            app._tick()
            await pilot.pause()
            assert app.seconds_since_event() >= 2399
            top = str(app._dash.query_one("#top", Static).render())
            import re

            assert re.search(r" 2[34]\d\ds", top), top  # the visible beat reads the helper

    asyncio.run(scenario())


def test_budget_meter_reads_this_legs_spend_not_the_runs_total(tmp_path: Path) -> None:
    """The cap re-arms each resume leg while usd_total stays cumulative, so
    dividing the cumulative spend by the CURRENT leg's cap showed a resumed run
    at "budget: 100%" having used 20% of it."""

    async def scenario() -> None:
        (tmp_path / "logs.jsonl").write_text("", encoding="utf-8")
        app = Agent6TUI(tmp_path)
        async with app.run_test(size=(150, 40)) as pilot:
            await _show_dashboard(pilot)
            app._handle_event(_ev(type="session.start", user_task="x", mode="run"))
            app._handle_event(_ev(type="role.call", role="worker", model="m"))
            app._handle_event(_ev(type="budget.update", usd_total=2.0, usd_cap=2.5))
            app._handle_event(_ev(type="loop.resume.start"))
            app._handle_event(_ev(type="budget.update", usd_total=0.5, usd_cap=2.5))
            app._tick()
            await pilot.pause()
            top = str(app._dash.query_one("#top", Static).render())
            assert "budget: 20%" in top, top
            assert "$2.50" in top  # the cost figure stays cumulative

    asyncio.run(scenario())


def test_compact_now_drops_the_marker_for_a_live_run(tmp_path: Path) -> None:
    """The Run menu's "Compact context now" drops the compact.request marker for
    the run to honor at its next boundary; a finished run refuses (nothing to
    compact)."""
    import os

    from agent6.sessions.ipc import read_compact_request, write_worker_pid

    async def scenario() -> None:
        (tmp_path / "logs.jsonl").write_text("", encoding="utf-8")
        write_worker_pid(tmp_path, os.getpid())
        app = Agent6TUI(tmp_path)
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()
            assert read_compact_request(tmp_path) is None
            app.action_compact()
            await pilot.pause()
            assert read_compact_request(tmp_path) is not None  # marker dropped for the run
            # A finished run: the action refuses instead of dropping a marker.
            (tmp_path / "compact.request").unlink()
            app._handle_event(_ev(type="session.end", reason="completed", all_passed=True))
            app.action_compact()
            await pilot.pause()
            assert read_compact_request(tmp_path) is None

    asyncio.run(scenario())


def test_payload_literal_does_not_swallow_a_real_steer(tmp_path: Path) -> None:
    # The seed counts steers the same way the fold does: an event whose PAYLOAD
    # contains the quoted literal (a grep of the source for the event name) must
    # not inflate the baseline, or the NEXT real steer is silently swallowed
    # while the run blocks awaiting an instruction.
    from agent6.ui.tui.composer import SteerInput

    (tmp_path / "logs.jsonl").write_text(
        "".join(
            json.dumps(e) + "\n"
            for e in (
                _ev(type="session.start", user_task="x"),
                _ev(
                    type="tool.call",
                    name="run_command",
                    args={"argv": ["rg", "session.steer_requested", "src/"]},
                ),
                _ev(type="role.call", role="worker", model="kimi"),
            )
        ),
        encoding="utf-8",
    )

    async def scenario() -> None:
        app = Agent6TUI(tmp_path)
        async with app.run_test() as pilot:
            await _show_dashboard(pilot)
            app._dash.query_one("#log").focus()
            await pilot.pause()
            for _ in range(50):  # let the reader thread replay the existing log
                await pilot.pause()
            assert app._seen_steer == 0  # pyright: ignore[reportPrivateUsage]
            assert app.state.steer_requests == 0
            app._handle_event(_ev(type="session.steer_requested", source="sigint"))  # pyright: ignore[reportPrivateUsage]
            app._tick()  # pyright: ignore[reportPrivateUsage]
            bar = app._dash.query_one(SteerInput)
            await _settle_focus(pilot, bar)
            assert app.focused is bar

    asyncio.run(scenario())


def test_historical_steer_request_does_not_grab_the_bar_on_open(tmp_path: Path) -> None:
    # A CLI Ctrl-C that DETACHED leaves session.steer_requested in the log. Opening the
    # TUI must not treat that stale (already-handled) request as live -- only one
    # that arrives AFTER the TUI is watching should route to the composer bar.
    from agent6.ui.tui.composer import SteerInput

    (tmp_path / "logs.jsonl").write_text(
        "".join(
            json.dumps(e) + "\n"
            for e in (
                _ev(type="session.start", user_task="fix it"),
                _ev(type="session.steer_requested", source="sigint"),
                _ev(type="role.call", role="worker", model="kimi"),
            )
        ),
        encoding="utf-8",
    )

    async def scenario() -> None:
        app = Agent6TUI(tmp_path)
        async with app.run_test() as pilot:
            await _show_dashboard(pilot)
            app._dash.query_one("#log").focus()  # park focus off the default-focused bar
            await pilot.pause()
            for _ in range(50):  # let the reader thread replay the existing log
                await pilot.pause()
                if app.state.steer_requests >= 1:
                    break
            app._tick()
            await pilot.pause()
            assert app.state.steer_requests == 1  # the historical event WAS folded
            bar = app._dash.query_one("#dash-input", SteerInput)
            assert app.focused is not bar  # but it did NOT grab the composer
            # a NEW steer request (a live Ctrl-C while watching) still routes here
            app._handle_event(_ev(type="session.steer_requested", source="sigint"))
            app._tick()
            await _settle_focus(pilot, bar)
            assert app.focused is bar

    asyncio.run(scenario())


def test_toggle_and_log_viewer_keys(tmp_path: Path) -> None:
    # Ctrl+D flips conversation <-> dashboard even with a composer bar focused
    # (the default focus on both); the log viewer opens from the View menu (the
    # run views have no bare letters) and closes with its own keys.
    from agent6.ui.tui.dashboard import DashboardScreen
    from agent6.ui.tui.logview import LogScreen

    (tmp_path / "logs.jsonl").write_text(
        json.dumps({"type": "session.start", "user_task": "x"}) + "\n", encoding="utf-8"
    )

    async def scenario() -> None:
        app = Agent6TUI(tmp_path)
        async with app.run_test() as pilot:
            await pilot.pause()
            assert app.screen is app._conv  # the conversation is the primary view
            await pilot.press("ctrl+d")  # show the dashboard
            await pilot.pause()
            assert isinstance(app.screen, DashboardScreen)
            from agent6.ui.tui.composer import SteerInput

            assert isinstance(app.focused, SteerInput)  # the bar is the default focus
            depth = len(app.screen_stack)
            app._dash.action_view_logs()  # View menu / palette action (no bare letters)
            await pilot.pause()
            assert isinstance(app.screen, LogScreen)
            await pilot.press("l")  # LogScreen (no input box) keeps its letters
            await pilot.pause()
            assert isinstance(app.screen, DashboardScreen)
            assert len(app.screen_stack) == depth
            await pilot.press("ctrl+d")  # flip back to the conversation (bar focused)
            await pilot.pause()
            assert app.screen is app._conv
            await pilot.press("ctrl+d")  # and again to the dashboard
            await pilot.pause()
            assert isinstance(app.screen, DashboardScreen)

    asyncio.run(scenario())


def test_task_filter_scopes_tools_log_and_diff(tmp_path: Path) -> None:
    # Two tasks, each with a tool call + a commit. Selecting a task filters the
    # tools table / log / diff to just that task's activity; the fold stamps each
    # event with the cursor task in focus when it landed.
    def _nodes(cur_status: dict[str, str]) -> dict[str, object]:
        return {
            tid: {"title": t, "status": cur_status[tid], "parent_id": None, "children": []}
            for tid, t in (("t1", "Task one"), ("t2", "Task two"))
        }

    events = [
        {"type": "session.start", "user_task": "x"},
        {
            "type": "graph.update",
            "nodes": _nodes({"t1": "in_progress", "t2": "pending"}),
            "cursor": "t1",
        },
        {"type": "tool.call", "name": "read_file", "args": {"path": "a.py"}},
        {"type": "diff.updated", "sha": "aaa", "patch": "diff --git a/a.py b/a.py\n+one"},
        {
            "type": "graph.update",
            "nodes": _nodes({"t1": "passed", "t2": "in_progress"}),
            "cursor": "t2",
        },
        {"type": "tool.call", "name": "apply_edit", "args": {"path": "b.py"}},
        {"type": "diff.updated", "sha": "bbb", "patch": "diff --git a/b.py b/b.py\n+two"},
    ]
    (tmp_path / "logs.jsonl").write_text(
        "\n".join(json.dumps(e) for e in events) + "\n", encoding="utf-8"
    )

    async def scenario() -> None:
        app = Agent6TUI(tmp_path)
        async with app.run_test(size=(150, 42)) as pilot:
            for _ in range(60):
                await pilot.pause()
                if len(app.state.recent_diffs) >= 2:  # the last events emitted
                    break
            # Fold stamped each tool call + diff with the task in focus at the time.
            assert [tc.task_id for tc in app.state.tool_calls] == ["t1", "t2"]
            assert [d.task_id for d in app.state.recent_diffs] == ["t1", "t2"]

            dash = app._dash
            dash.render_state()  # the dashboard renders on demand while covered
            await pilot.pause()
            assert len(dash._visible_tools) == 2  # unfiltered: both

            def log_text() -> str:
                return " ".join(strip.text for strip in dash.query_one("#log", RichLog).lines)

            def diff_text() -> str:
                return str(dash.query_one("#diff-body", Static).content)

            dash._selected_task_id = "t1"  # filter to task one (the handler re-renders)
            dash.render_state()
            await pilot.pause()
            assert [tc.name for tc in dash._visible_tools] == ["read_file"]
            assert "read_file" in log_text() and "apply_edit" not in log_text()
            assert "+one" in diff_text() and "+two" not in diff_text()

            dash._selected_task_id = "t2"
            dash.render_state()
            await pilot.pause()
            assert [tc.name for tc in dash._visible_tools] == ["apply_edit"]
            assert "apply_edit" in log_text() and "read_file" not in log_text()
            assert "+two" in diff_text() and "+one" not in diff_text()

    asyncio.run(scenario())


def test_question_modal_multi_collects_all_answers() -> None:
    """A prompt with several questions: each has its own answer field, option
    buttons fill their own field, and Submit (ctrl+s) returns every answer as a
    tuple aligned to the questions."""
    result: dict[str, tuple[str, ...] | None] = {}
    qs = (
        Question(question="Framework?", options=("React", "Vue")),
        Question(question="Component name?", options=()),
    )

    class _Host(App[None]):
        def on_mount(self) -> None:
            self.push_screen(QuestionModal("q1", qs), lambda v: result.__setitem__("v", v))

    async def scenario() -> None:
        app = _Host()
        async with app.run_test() as pilot:
            await pilot.pause()
            modal = app.screen
            assert isinstance(modal, QuestionModal)
            modal.query_one("#opt-0-1", Button).press()  # pick "Vue" for the first question
            await pilot.pause()
            assert modal.query_one("#ans-0", Input).value == "Vue"
            modal.query_one("#ans-1", Input).value = "widget"  # type the second answer
            await pilot.press("ctrl+s")
            await pilot.pause()
            assert result.get("v") == ("Vue", "widget")  # both, aligned to the questions

    asyncio.run(scenario())


def test_conversation_is_the_primary_view(tmp_path: Path) -> None:
    """The app opens on the run's conversation; Ctrl+D toggles the dashboard and
    back with the SAME conversation instance (state persists); Esc on the primary
    view leaves for the hub (exit 0)."""
    from agent6.ui.tui.conversation import ConversationScreen
    from agent6.ui.tui.dashboard import DashboardScreen

    (tmp_path / "logs.jsonl").write_text(
        json.dumps({"type": "session.start", "user_task": "x"}) + "\n", encoding="utf-8"
    )

    async def scenario() -> None:
        app = Agent6TUI(tmp_path, from_hub=True)
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            assert isinstance(app.screen, ConversationScreen)
            first = app.screen
            await pilot.press("ctrl+d")
            await pilot.pause()
            assert isinstance(app.screen, DashboardScreen)
            await pilot.press("ctrl+d")
            await pilot.pause()
            assert app.screen is first  # the same instance: nothing was rebuilt
            await pilot.press("escape")  # Esc on the primary view leaves for the hub
            await pilot.pause()
        assert app.return_value == 0

    asyncio.run(scenario())


def test_pushed_conversation_viewer_still_dismisses(tmp_path: Path) -> None:
    """A ConversationScreen pushed as a read-only viewer (the hub's t) keeps its
    old behavior: Esc dismisses back to the host; Ctrl+D is inert (no dashboard)."""
    from agent6.ui.tui.conversation import ConversationScreen

    (tmp_path / "logs.jsonl").write_text("", encoding="utf-8")

    class _Host(App[None]):
        def on_mount(self) -> None:
            self.push_screen(
                ConversationScreen(tmp_path / "logs.jsonl", title=lambda ctx: f"{ctx} · x")
            )

    async def scenario() -> None:
        app = _Host()
        async with app.run_test() as pilot:
            await pilot.pause()
            assert isinstance(app.screen, ConversationScreen)
            await pilot.press("ctrl+d")  # inert on a viewer
            await pilot.pause()
            assert isinstance(app.screen, ConversationScreen)
            await pilot.press("escape")  # Esc dismisses the viewer
            await pilot.pause()
            assert not isinstance(app.screen, ConversationScreen)

    asyncio.run(scenario())


def test_dashboard_detects_a_dead_worker_and_tells_the_truth(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """A worker killed without a session.end (kill -9 / OOM) must not render as a
    live spinner forever. Every other surface probes worker.pid (the hub says
    "stale", the web refuses steer); the dashboard was the one surface with no
    liveness check: it spun "working…", accepted steer with a success toast
    nobody would read, and REFUSED resume -- the one correct action. The ~1/s
    heartbeat probe flips the dir status to stale, the body says the worker
    exited, and the composer routes to resume."""
    import json

    from agent6.ui.tui import app as app_mod

    events = [
        {"type": "session.start", "session_id": "dead-01", "mode": "run", "user_task": "t"},
        {"type": "role.call", "role": "worker", "model": "m", "provider": "p"},
    ]
    (tmp_path / "logs.jsonl").write_text(
        "".join(json.dumps(e) + "\n" for e in events), encoding="utf-8"
    )
    (tmp_path / "worker.pid").write_text("999999999", encoding="utf-8")  # dead pid

    spawned: list[tuple[str, str]] = []

    def _fake_resume(
        _cwd: Path, rid: str, *, steer: str = "", preset: str = "", config_path: object = None
    ) -> str:
        spawned.append((rid, steer))
        return ""

    monkeypatch.setattr(app_mod, "spawn_detached_resume", _fake_resume)

    async def scenario() -> None:
        app = Agent6TUI(tmp_path)
        async with app.run_test(size=(120, 40)) as pilot:
            await _show_dashboard(pilot)
            app._heartbeat_at = 0.0  # age the throttle so the probe fires now
            app._tick()
            await pilot.pause()
            assert app.worker_lost is True
            assert app.session_controllable() is False
            body = str(app._dash.query_one("#stream-body", Static).render())
            assert "worker exited" in body
            assert "working…" not in body
            # The composer routes to resume (not a steer nobody will read).
            app.submit_instruction("carry on")
            assert spawned == [(tmp_path.name, "carry on")]
            # And action_resume resumes instead of refusing "still going".
            app.action_resume()
            assert len(spawned) == 2

    asyncio.run(scenario())


def test_dashboard_heartbeat_ticks_while_active(tmp_path: Path) -> None:
    """An attached dashboard on a live-but-silent run shows a ticking "working…
    Ns" heartbeat, so a thinking / resuming run reads as alive, not hung. The
    elapsed count advances across ticks even with no new events."""
    import json

    events = [
        {"type": "session.start", "session_id": "live-01", "mode": "run", "user_task": "t"},
        {"type": "role.call", "role": "worker", "model": "m", "provider": "p"},
    ]
    (tmp_path / "logs.jsonl").write_text(
        "".join(json.dumps(e) + "\n" for e in events), encoding="utf-8"
    )
    # "live-but-silent" is the whole subject: a run with no worker.pid is one
    # whose worker exited, and a dead run must NOT show a ticking heartbeat.
    (tmp_path / "worker.pid").write_text(str(os.getpid()), encoding="utf-8")

    import re

    def _seconds(text: str) -> int:
        m = re.search(r"working… (\d+)s", text)
        return int(m.group(1)) if m else 0

    async def scenario() -> str:
        app = Agent6TUI(tmp_path)
        async with app.run_test(size=(120, 40)) as pilot:
            await _show_dashboard(pilot)

            def advanced() -> bool:
                app._tick()  # pyright: ignore[reportPrivateUsage]
                return _seconds(str(app._dash.query_one("#stream-body", Static).render())) >= 1

            # The heartbeat needs real wall time (>=1s since the last event);
            # poll until it shows instead of betting on a fixed budget.
            await _wait_for(pilot, advanced, "the working… heartbeat to advance", timeout=15.0)
            return str(app._dash.query_one("#stream-body", Static).render())

    text = asyncio.run(scenario())
    assert "working…" in text
    assert _seconds(text) >= 1  # the heartbeat advanced


def test_tick_survives_an_empty_screen_stack(tmp_path: Path) -> None:
    """The 0.2s _tick interval races shutdown: teardown pops every screen, and a
    tick landing in that window hit the raising App.screen property, crashing the
    app (ScreenStackError surfaced at run_test exit -- the load-only flake that
    took down a different TUI test each full-suite run). A tick on an empty stack
    must be a no-op, including the steer-request routing path."""

    async def scenario() -> None:
        (tmp_path / "logs.jsonl").write_text("", encoding="utf-8")
        app = Agent6TUI(tmp_path)
        async with app.run_test() as pilot:
            await pilot.pause()
            # Empty the stack only around the synchronous tick (restored after),
            # so run_test teardown still sees the screens it expects to pop.
            stack = app._screen_stack  # pyright: ignore[reportPrivateUsage]
            saved = list(stack)
            stack.clear()
            app._seen_steer = -1  # pyright: ignore[reportPrivateUsage]
            app._tick()  # pyright: ignore[reportPrivateUsage]
            stack.extend(saved)

    asyncio.run(scenario())


def test_dashboard_follows_live_appends_after_attach(tmp_path: Path) -> None:
    """The detach->attach symptom the user hit: after opening on a live run, NEW
    events appended by the background process must appear (not a frozen snapshot).
    Attach on a partial log, append more, and assert the new tool row shows."""
    import json

    logs = tmp_path / "logs.jsonl"

    def append(events: list[dict[str, object]]) -> None:
        with logs.open("a", encoding="utf-8") as fh:
            for e in events:
                fh.write(json.dumps(e) + "\n")

    logs.write_text("", encoding="utf-8")
    append(
        [
            {"type": "session.start", "session_id": "live-02", "mode": "run", "user_task": "t"},
            {"type": "tool.call", "name": "read_file", "args": {"path": "a.py"}},
            {"type": "tool.result", "name": "read_file", "ok": True, "summary": "1 byte"},
        ]
    )

    async def scenario() -> int:
        app = Agent6TUI(tmp_path)
        async with app.run_test(size=(120, 40)) as pilot:
            await _show_dashboard(pilot)

            def rows() -> int:
                app._tick()  # pyright: ignore[reportPrivateUsage]
                return app._dash.query_one("#tools", DataTable).row_count

            # Wait through the reader thread's initial fold, not a fixed budget.
            await _wait_for(pilot, lambda: rows() >= 1, "the attach-time fold", timeout=15.0)
            before = app._dash.query_one("#tools", DataTable).row_count
            # The background process appends a new turn AFTER we attached.
            append(
                [
                    {"type": "tool.call", "name": "apply_edit", "args": {"path": "a.py"}},
                    {"type": "tool.result", "name": "apply_edit", "ok": True, "summary": "applied"},
                ]
            )
            await _wait_for(pilot, lambda: rows() > before, "the appended turn", timeout=15.0)
            return app._dash.query_one("#tools", DataTable).row_count

    assert asyncio.run(scenario()) == 2


def test_composer_compact_directive_routes_to_compact_request(tmp_path: Path) -> None:
    """A composer `/compact <focus>` on a LIVE run becomes a compact request
    carrying the focus (no steer files); `/pin <text>` stays a steer for the
    loop's own parser."""
    import json
    import os

    events = [
        {"type": "session.start", "session_id": "live-01", "mode": "run", "user_task": "t"},
        {"type": "role.call", "role": "worker", "model": "m", "provider": "p"},
    ]
    (tmp_path / "logs.jsonl").write_text(
        "".join(json.dumps(e) + "\n" for e in events), encoding="utf-8"
    )
    (tmp_path / "worker.pid").write_text(str(os.getpid()), encoding="utf-8")  # live

    async def scenario() -> None:
        app = Agent6TUI(tmp_path)
        async with app.run_test(size=(120, 40)) as pilot:
            await _show_dashboard(pilot)
            app.submit_instruction("/compact keep the auth decisions")
            await pilot.pause()
            assert (tmp_path / "compact.request").read_text(encoding="utf-8") == (
                "keep the auth decisions"
            )
            assert not (tmp_path / "steer.answer").exists()
            assert not (tmp_path / "steer.request").exists()
            app.submit_instruction("/pin keep the API stable")
            await pilot.pause()
            assert (tmp_path / "steer.answer").read_text(encoding="utf-8") == (
                "/pin keep the API stable"
            )

    asyncio.run(scenario())


def test_a_prompt_with_no_scope_shows_no_allow_session_button() -> None:
    """The button would grant nothing beyond the call it was clicked on (the
    fetch gate opts out of standing answers), so it is not there -- and neither
    is its "a" key, footer entry included."""

    class _Host(App[None]):
        def on_mount(self) -> None:
            self.push_screen(ApprovalModal("a", "allow fetch?", standing=False), lambda _v: None)

    async def scenario() -> None:
        app = _Host()
        async with app.run_test() as pilot:
            await pilot.pause()
            modal = app.screen
            assert isinstance(modal, ApprovalModal)
            labels = [str(b.label) for b in modal.query(Button)]
            assert labels == ["Allow (y)", "Deny (n)"]
            assert modal.check_action("approve_session", ()) is False
            await pilot.press("a")  # the removed binding must not answer
            await pilot.pause()
            assert app.screen is modal, "'a' dismissed a modal that offers no session answer"

    asyncio.run(scenario())


def test_the_composer_title_shows_its_brackets(tmp_path: Path) -> None:
    """A border title is markup: the live composer's `/compact [focus]` hint
    lost its `[focus]` (rendered as "/compact )") the way `[git]` vanished from
    a toast, so the title is escaped where it is set."""
    import os

    from rich.markup import escape

    from agent6.sessions.ipc import write_worker_pid
    from agent6.ui.tui.composer import SteerInput, composer_labels

    async def scenario() -> None:
        (tmp_path / "logs.jsonl").write_text("", encoding="utf-8")
        write_worker_pid(tmp_path, os.getpid())
        app = Agent6TUI(tmp_path)
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()
            bar = app._conv.query_one("#conv-input", SteerInput)
            title, _keys = composer_labels("steer")
            assert "[focus]" in title
            assert bar.border_title == escape(title)

    asyncio.run(scenario())


def test_the_menu_bar_title_keeps_the_tasks_brackets(tmp_path: Path) -> None:
    """The bar mirrors the app subtitle, which carries the task the user
    typed; a Static given a str parses it as markup, so `[wip]` vanished."""
    from rich.text import Text
    from textual.widgets import Static

    async def scenario() -> None:
        (tmp_path / "logs.jsonl").write_text(
            json.dumps(_ev(type="session.start", user_task="fix [wip] parser", mode="run")) + "\n",
            encoding="utf-8",
        )
        app = Agent6TUI(tmp_path)
        async with app.run_test(size=(120, 30)) as pilot:
            for _ in range(30):
                await pilot.pause()
                if app.state.user_task:
                    break
            await pilot.pause()
            shown = app.screen.query_one(".app-title", Static).render()
            text = shown.plain if isinstance(shown, Text) else str(shown)
            assert "[wip]" in text, text

    asyncio.run(scenario())
