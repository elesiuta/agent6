# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Tests for the `agent6 tui` Machines page."""

from __future__ import annotations

import asyncio
from pathlib import Path

from textual.app import App
from textual.widgets import DataTable, Input

from agent6.machine import MachineSpec
from agent6.ui.tui import machines as machmod
from agent6.ui.tui.machines import (
    CreateMachineModal,
    MachineDetailScreen,
    MachinesScreen,
    MachineWatchScreen,
    machine_detail_text,
)
from agent6.ui.tui.modals import ConfirmModal
from agent6.viewmodel import machine_files

# A no-I/O machine that reaches a terminal immediately (branch -> terminal), so a
# `machine run` produces a finished instance with no model/jail needed.
TINY = """
machine = "tiny"
version = 1
initial = "route"

[budget]
max_transitions = 10

[vars.code]
n = { type = "int", default = 0 }

[states.route]
kind = "branch"
when = [
  { if = "n == 0", goto = "done" },
  { else = true, goto = "done" },
]

[states.done]
kind = "terminal"
status = "ok"
reason = "routed"
"""

WAITER = """
machine = "waiter_demo"
version = 1
initial = "poll"

[budget]
max_usd = 1.0
max_transitions = 100

[vars.operator]
secs = { type = "int", value = 3600 }

[states.poll]
kind = "wait"
every_secs = "{{ secs }}"
on = { tick = "done", signal = "woken" }

[states.done]
kind = "terminal"
status = "ok"
reason = "ticked"

[states.woken]
kind = "terminal"
status = "ok"
reason = "signalled"
"""


def _write(path: Path, body: str = WAITER) -> Path:
    path.write_text(body, encoding="utf-8")
    return path


def test_machine_files_cwd_and_subdir(tmp_path: Path) -> None:
    _write(tmp_path / "a.asm.toml")
    (tmp_path / "machines").mkdir()
    _write(tmp_path / "machines" / "b.asm.toml")
    (tmp_path / "not-a-machine.toml").write_text("x = 1\n", encoding="utf-8")
    names = {p.name for p in machine_files(tmp_path)}
    assert names == {"a.asm.toml", "b.asm.toml"}


def test_machine_detail_text_parses_a_valid_machine(tmp_path: Path) -> None:
    text = machine_detail_text(_write(tmp_path / "m.asm.toml"))
    assert "machine: waiter_demo" in text
    assert "initial: poll" in text
    # Named for what it ran: this view checks semantics, not the script bundle.
    assert "semantics: OK" in text
    assert "graph (mermaid):" in text
    # States read as the user's kind word (agent/tool/wait/terminal), matching the
    # watch screen + web, not the internal class name (AgentState/TerminalState).
    assert "poll  (wait)" in text and "done  (terminal)" in text
    assert "State)" not in text


def test_machine_detail_text_reports_a_bad_file(tmp_path: Path) -> None:
    bad = tmp_path / "bad.asm.toml"
    bad.write_text("this is not = valid [[[\n", encoding="utf-8")
    assert "failed to load bad.asm.toml" in machine_detail_text(bad)


class _Host(App[None]):
    def __init__(self, repo_cwd: Path) -> None:
        super().__init__()
        self._repo = repo_cwd

    def on_mount(self) -> None:
        self.push_screen(MachinesScreen(self._repo / ".agent6", self._repo))


def test_machines_menu_items_all_resolve(tmp_path: Path) -> None:
    async def scenario() -> None:
        app = _Host(tmp_path)
        async with app.run_test() as pilot:
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, MachinesScreen)
            for menu in screen.MENUS:
                for item in menu.items:
                    resolved = getattr(screen, f"action_{item.action}", None) or getattr(
                        app, f"action_{item.action}", None
                    )
                    assert resolved is not None, f"no handler for {item.action}"

    asyncio.run(scenario())


def test_row_actions_are_dimmed_on_an_empty_machines_page(tmp_path: Path) -> None:
    """View, Run and Watch all need a row; with none they were silent no-ops
    the footer still offered."""

    async def scenario() -> None:
        app = _Host(tmp_path)
        async with app.run_test() as pilot:
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, MachinesScreen)
            # None greys the key; False would hide it, and a missing key reads
            # as a capability the page does not have.
            assert [screen.check_action(a, ()) for a in ("view", "run", "watch")] == [
                None,
                None,
                None,
            ]
            assert screen.check_action("create", ()) is True

    asyncio.run(scenario())


def test_watch_screen_carries_the_menu_bar_and_its_items_resolve(
    tmp_path: Path, monkeypatch: object
) -> None:
    """The watch screen has the chrome every other screen has (a menu bar,
    `?` help), and every menu item resolves to an action."""
    from textual.widgets import Footer

    from agent6.machine import load_machine
    from agent6.ui.tui.menubar import MenuBar

    monkeypatch.chdir(tmp_path)  # type: ignore[attr-defined]
    f = tmp_path / "tiny.asm.toml"
    f.write_text(TINY, encoding="utf-8")
    spec = load_machine(f)
    instance = tmp_path / "instance"
    instance.mkdir()

    class _Host(App[None]):
        def on_mount(self) -> None:
            self.push_screen(MachineWatchScreen(instance, spec))

    async def scenario() -> None:
        app = _Host()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, MachineWatchScreen)
            assert screen.query(MenuBar) and screen.query(Footer)
            for menu in screen.MENUS:
                for item in menu.items:
                    assert getattr(screen, f"action_{item.action}", None) or getattr(
                        app, f"action_{item.action}", None
                    ), f"no handler for {item.action}"

    asyncio.run(scenario())


def test_watch_screen_shows_states_transitions_and_end(tmp_path: Path, monkeypatch: object) -> None:
    """The Machines watch screen renders the state overview (current marked `>`,
    visited `.`), the transition in the log, and the ended status -- the in-TUI
    equivalent of `agent6 attach`."""
    from agent6.config.layer import resolved_state_dir
    from agent6.machine import load_machine
    from agent6.ui.cli import main as cli_main

    monkeypatch.chdir(tmp_path)  # type: ignore[attr-defined]
    f = tmp_path / "tiny.asm.toml"
    f.write_text(TINY, encoding="utf-8")
    assert cli_main(["machine", "run", str(f)]) == 0
    instance = resolved_state_dir(tmp_path) / "machines" / "tiny"
    spec = load_machine(f)

    class _Host(App[None]):
        def on_mount(self) -> None:
            self.push_screen(MachineWatchScreen(instance, spec))

    async def scenario() -> None:
        app = _Host()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            for _ in range(3):  # let a poll or two run
                await pilot.pause()
            screen = app.screen
            assert isinstance(screen, MachineWatchScreen)
            table = screen.query_one("#mw-states", DataTable)
            assert table.row_count == len(spec.states)
            assert table.get_cell("done", "mark") == "▸"  # current (terminal) state
            assert table.get_cell("route", "mark") == "·"  # visited
            from textual.widgets import RichLog

            log = screen.query_one("#mw-log", RichLog)
            assert len(log.lines) >= 1  # the route->done transition was logged

    asyncio.run(scenario())


def test_watch_screen_does_not_reannounce_a_stale_end(tmp_path: Path, monkeypatch: object) -> None:
    """Reviewing a machine that finished long ago must not pop a fresh toast +
    desktop notification for the stale end; the end flag seeds from the same
    fold that seeds notification history. A machine ending WHILE watched still
    announces (ended is None at mount)."""
    from agent6.config.layer import resolved_state_dir
    from agent6.machine import load_machine
    from agent6.ui.cli import main as cli_main
    from agent6.ui.tui import machines as machines_mod

    monkeypatch.chdir(tmp_path)  # type: ignore[attr-defined]
    fired: list[tuple[str, str]] = []

    def fake_notify(title: str, body: str) -> None:
        fired.append((title, body))

    monkeypatch.setattr(machines_mod, "desktop_notify", fake_notify)  # type: ignore[attr-defined]
    f = tmp_path / "tiny.asm.toml"
    f.write_text(TINY, encoding="utf-8")
    assert cli_main(["machine", "run", str(f)]) == 0
    instance = resolved_state_dir(tmp_path) / "machines" / "tiny"
    spec = load_machine(f)

    class _Host(App[None]):
        def on_mount(self) -> None:
            self.push_screen(MachineWatchScreen(instance, spec))

    async def scenario() -> None:
        app = _Host()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            for _ in range(4):  # let a few polls run
                await pilot.pause()
            screen = app.screen
            assert isinstance(screen, MachineWatchScreen)
            assert screen._end_notified is True  # pyright: ignore[reportPrivateUsage]
            assert fired == []  # no desktop notification for the day-old end
            notes = [str(n.message) for n in app._notifications]  # pyright: ignore[reportPrivateUsage]
            assert not any(m == "ok" or m.endswith(" ok") for m in notes), notes

    asyncio.run(scenario())


def test_watch_screen_disables_steer_and_message_when_ended(
    tmp_path: Path, monkeypatch: object
) -> None:
    """An ended machine takes no input: the watch screen dims Steer/Message (like
    the web disables both buttons) and their actions are no-ops, never dropping a
    steer marker into the dead per-state dir."""
    from agent6.config.layer import resolved_state_dir
    from agent6.machine import load_machine
    from agent6.ui.cli import main as cli_main

    monkeypatch.chdir(tmp_path)  # type: ignore[attr-defined]
    f = tmp_path / "tiny.asm.toml"
    f.write_text(TINY, encoding="utf-8")
    assert cli_main(["machine", "run", str(f)]) == 0
    instance = resolved_state_dir(tmp_path) / "machines" / "tiny"
    spec = load_machine(f)
    # A per-state dir so _current_state_dir() resolves -- the "dead dir" a steer would hit.
    state = instance / "states" / "0000-route"
    state.mkdir(parents=True)
    (state / "logs.jsonl").write_text("", encoding="utf-8")

    class _WatchHost(App[None]):
        def on_mount(self) -> None:
            self.push_screen(MachineWatchScreen(instance, spec))

    async def scenario() -> None:
        app = _WatchHost()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            for _ in range(3):  # let a poll set _ended
                await pilot.pause()
            screen = app.screen
            assert isinstance(screen, MachineWatchScreen)
            assert screen._ended  # pyright: ignore[reportPrivateUsage]
            assert screen.check_action("steer", ()) is False
            assert screen.check_action("poke", ()) is False
            screen.action_steer()  # no-op when ended
            await pilot.pause()
            assert not (state / "steer.request").exists()  # nothing dropped in the dead dir

    asyncio.run(scenario())


def test_watch_screen_suppresses_phantom_thinking_on_an_ended_machine(
    tmp_path: Path, monkeypatch: object
) -> None:
    """An ended machine's final agent-state log ends on a role.call ("thinking…"),
    which must NOT render as a live thinking line while the header says ended."""
    from textual.widgets import RichLog

    from agent6.config.layer import resolved_state_dir
    from agent6.machine import load_machine
    from agent6.ui.cli import main as cli_main

    monkeypatch.chdir(tmp_path)  # type: ignore[attr-defined]
    f = tmp_path / "tiny.asm.toml"
    f.write_text(TINY, encoding="utf-8")
    assert cli_main(["machine", "run", str(f)]) == 0  # terminates at once -> ended
    instance = resolved_state_dir(tmp_path) / "machines" / "tiny"
    spec = load_machine(f)
    state = instance / "states" / "0000-route"
    state.mkdir(parents=True)
    (state / "logs.jsonl").write_text(
        '{"type": "role.call", "role": "worker", "model": "kimi"}\n', encoding="utf-8"
    )

    class _WatchHost(App[None]):
        def on_mount(self) -> None:
            self.push_screen(MachineWatchScreen(instance, spec))

    async def scenario() -> None:
        app = _WatchHost()
        async with app.run_test(size=(120, 40)) as pilot:
            for _ in range(4):
                await pilot.pause()
            screen = app.screen
            assert isinstance(screen, MachineWatchScreen)
            assert screen._ended  # pyright: ignore[reportPrivateUsage]
            log = screen.query_one("#mw-log", RichLog)
            assert not any("effort" in strip.text for strip in log.lines)

    asyncio.run(scenario())


def test_discrete_log_line_renders_tool_events_only() -> None:
    # The shared journal fold (current/visited/transitions) is tested in
    # tests/unit/test_viewmodel_machine_state.py; this covers the TUI-only
    # presentation helper for the per-state agent log.
    from agent6.ui.tui.machines import _discrete_log_line

    # A tool call renders compactly; a thinking delta is not a discrete line.
    assert _discrete_log_line({"type": "role.effort_delta", "text": "hm"}) is None
    line = _discrete_log_line({"type": "tool.call", "name": "grep", "args": {"q": "x"}})
    assert line is not None and "grep" in line.plain
    # The verdict goes through the shared coercion (tool_result_ok), never
    # bool(): a historical stringified "False" would have painted a green tick.
    bad = _discrete_log_line({"type": "tool.result", "ok": "False", "summary": "boom"})
    assert bad is not None and "✗" in bad.plain
    good = _discrete_log_line({"type": "tool.result", "ok": "True", "summary": "fine"})
    assert good is not None and "✓" in good.plain


def test_create_opens_dashboard_on_the_draft(tmp_path: Path, monkeypatch: object) -> None:
    """Creating a machine spawns `machine create`, locates the draft it produces,
    and hands that dir to the dashboard via app.exit -- so it is watchable live,
    not fire-and-forget."""
    draft = tmp_path / "draft"
    draft.mkdir()

    def _fake_locate(*_a: object, **_k: object) -> tuple[Path, str]:
        return draft, ""

    monkeypatch.setattr(machmod, "spawn_and_locate", _fake_locate)  # type: ignore[attr-defined]

    async def scenario() -> None:
        app = _Host(tmp_path)
        async with app.run_test() as pilot:
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, MachinesScreen)
            screen._on_create("make a greeter")  # pyright: ignore[reportPrivateUsage]
            await pilot.pause()
        assert app.return_value == draft  # handed the draft to the dashboard

    asyncio.run(scenario())


def test_machines_page_lists_and_views(tmp_path: Path) -> None:
    _write(tmp_path / "m.asm.toml")

    async def scenario() -> None:
        app = _Host(tmp_path)
        async with app.run_test() as pilot:
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, MachinesScreen)
            table = screen.query_one("#machines", DataTable)
            assert table.row_count == 1
            table.focus()
            table.move_cursor(row=0)
            await pilot.press("v")  # view -> parsed detail screen
            await pilot.pause()
            assert isinstance(app.screen, MachineDetailScreen)

    asyncio.run(scenario())


def test_machines_page_title_counts_or_names_the_empty_case(tmp_path: Path) -> None:
    """An empty machines table read as still loading (the CLI says "no machines
    yet" and how to draft one); the title carries the count, like the hub's."""

    async def scenario() -> None:
        app = _Host(tmp_path)
        async with app.run_test() as pilot:
            await pilot.pause()
            assert app.sub_title.endswith("· no machines yet (c creates one)")
            _write(tmp_path / "m.asm.toml")
            screen = app.screen
            assert isinstance(screen, MachinesScreen)
            screen.action_refresh()
            await pilot.pause()
            assert app.sub_title.endswith("· 1 machine")

    asyncio.run(scenario())


def test_machines_menu_bar_dispatches_an_item(tmp_path: Path) -> None:
    """Selecting an item from the menu bar (not just the key binding) runs its
    action -- exercises action_menu + on_menu_bar_selected, the dead-menu bug class."""
    from agent6.ui.tui.menubar import MenuBar, _Dropdown

    async def scenario() -> None:
        app = _Host(tmp_path)
        async with app.run_test() as pilot:
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, MachinesScreen)
            screen.query_one(MenuBar).open("m")  # the "Machines" menu
            await pilot.pause()
            dd = next(iter(screen.query(_Dropdown)))
            idx = next(
                i for i in range(dd.option_count) if dd.get_option_at_index(i).id == "create"
            )
            dd.highlighted = idx
            await pilot.press("enter")
            await pilot.pause()
            assert isinstance(app.screen, CreateMachineModal)  # the menu actually fired

    asyncio.run(scenario())


def test_machine_run_confirms_then_spawns(tmp_path: Path, monkeypatch: object) -> None:
    path = _write(tmp_path / "m.asm.toml")
    captured: list[list[str]] = []

    def _fake_spawn(argv: list[str], cwd: Path, **_k: object) -> str:
        captured.append(list(argv))
        return ""

    monkeypatch.setattr(machmod, "spawn_and_confirm", _fake_spawn)  # type: ignore[attr-defined]

    async def scenario() -> None:
        app = _Host(tmp_path)
        async with app.run_test() as pilot:
            await pilot.pause()
            table = app.screen.query_one("#machines", DataTable)
            table.focus()
            table.move_cursor(row=0)
            await pilot.press("r")  # run -> confirm modal
            await pilot.pause()
            assert isinstance(app.screen, ConfirmModal)
            await pilot.press("y")  # confirm
            await pilot.pause()
            assert captured and captured[-1][-3:] == ["machine", "run", str(path)]

    asyncio.run(scenario())


def test_machine_run_refusal_notifies_and_skips_watch(tmp_path: Path, monkeypatch: object) -> None:
    """A `machine run` refusal (lock held, exit 2) must surface as an error
    notification, not open a watch screen on nothing."""
    _write(tmp_path / "m.asm.toml")

    def _fake_spawn(argv: list[str], cwd: Path, **_k: object) -> str:
        return "agent6 machine exited (1) before starting:\nERROR: lock held"

    monkeypatch.setattr(machmod, "spawn_and_confirm", _fake_spawn)  # type: ignore[attr-defined]

    async def scenario() -> None:
        app = _Host(tmp_path)
        async with app.run_test() as pilot:
            await pilot.pause()
            table = app.screen.query_one("#machines", DataTable)
            table.focus()
            table.move_cursor(row=0)
            await pilot.press("r")
            await pilot.pause()
            await pilot.press("y")  # confirm the run
            await pilot.pause()
            assert not isinstance(app.screen, MachineWatchScreen)  # no watch on nothing
            notes = [str(n.message) for n in app._notifications]  # pyright: ignore[reportPrivateUsage]
            assert any("lock held" in n for n in notes)

    asyncio.run(scenario())


def test_watch_screen_survives_corrupt_journal(tmp_path: Path) -> None:
    """A corrupt journal line must not crash the watch screen every poll tick;
    the header shows the corruption and polling continues."""
    from textual.widgets import Static

    from agent6.machine import load_machine

    f = _write(tmp_path / "m.asm.toml", TINY)
    spec = load_machine(f)
    instance = tmp_path / ".agent6" / "machines" / "tiny"
    instance.mkdir(parents=True)
    (instance / "journal.jsonl").write_text('{"type": "step", "bogus": 1}\n', encoding="utf-8")

    class _WatchHost(App[None]):
        def on_mount(self) -> None:
            self.push_screen(MachineWatchScreen(instance, spec))

    async def scenario() -> None:
        app = _WatchHost()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            for _ in range(3):
                await pilot.pause()
            screen = app.screen
            assert isinstance(screen, MachineWatchScreen)  # still alive, not crashed
            head = screen.query_one("#mw-head", Static)
            assert "journal unreadable" in str(head.render())

    asyncio.run(scenario())


def test_watch_screen_tolerates_torn_utf8_state_log(tmp_path: Path) -> None:
    """A per-state agent log whose tail ends mid multibyte UTF-8 sequence (the
    writer flushes long lines in several syscalls) must not crash the poll; the
    complete prefix renders and the torn tail is picked up once completed."""
    import json as _json

    from textual.widgets import RichLog

    from agent6.machine import load_machine

    f = _write(tmp_path / "m.asm.toml", TINY)
    spec = load_machine(f)
    instance = tmp_path / ".agent6" / "machines" / "tiny"
    state = instance / "states" / "0000-route"
    state.mkdir(parents=True)
    (instance / "journal.jsonl").write_text("", encoding="utf-8")
    full = _json.dumps({"type": "tool.call", "name": "café", "args": {}}, ensure_ascii=False)
    raw = full.encode("utf-8")
    cut = raw.rindex(b"\xc3\xa9") + 1  # keep only the first byte of the é sequence
    (state / "logs.jsonl").write_bytes(
        _json.dumps({"type": "tool.call", "name": "grep", "args": {}}).encode() + b"\n" + raw[:cut]
    )

    class _WatchHost(App[None]):
        def on_mount(self) -> None:
            self.push_screen(MachineWatchScreen(instance, spec))

    async def scenario() -> None:
        app = _WatchHost()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            for _ in range(3):
                await pilot.pause()
            screen = app.screen
            assert isinstance(screen, MachineWatchScreen)  # no UnicodeDecodeError crash
            log = screen.query_one("#mw-log", RichLog)
            assert any("grep" in line.text for line in log.lines)
            assert not any("café" in line.text for line in log.lines)  # torn line held back
            # Completing the line delivers it on a later poll.
            with (state / "logs.jsonl").open("ab") as fh:
                fh.write(raw[cut:] + b"\n")
            for _ in range(4):
                await pilot.pause()
            screen._poll()  # pyright: ignore[reportPrivateUsage]
            await pilot.pause()
            assert any("café" in line.text for line in log.lines)

    asyncio.run(scenario())


def test_machine_create_spawns_with_task(tmp_path: Path, monkeypatch: object) -> None:
    """The create modal threads the typed task into `agent6 machine create <task>`
    (then the draft is located + handed to the dashboard)."""
    captured: list[list[str]] = []
    draft = tmp_path / "d"
    draft.mkdir()

    def _fake_locate(argv: list[str], cwd: Path, **_k: object) -> tuple[Path, str]:
        captured.append(list(argv))
        return draft, ""

    monkeypatch.setattr(machmod, "spawn_and_locate", _fake_locate)  # type: ignore[attr-defined]

    async def scenario() -> None:
        app = _Host(tmp_path)
        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.press("c")  # create -> task modal
            await pilot.pause()
            assert isinstance(app.screen, CreateMachineModal)
            app.screen.query_one("#create-input", Input).value = "nightly sweep"
            await pilot.press("enter")  # submit
            await pilot.pause()
            assert captured and captured[-1][-3:] == ["create", "--", "nightly sweep"]

    asyncio.run(scenario())


def test_watch_screen_refuses_a_steer_no_state_would_read(
    tmp_path: Path, monkeypatch: object
) -> None:
    """A parked machine has not ENDED, but its worker is gone and its newest
    state dir is a finished agent state, so nothing will ever poll the marker.
    The web refuses this with a reason; the TUI wrote the marker and reported
    success, silently dropping the operator's course-correction."""
    from agent6.machine import load_machine

    monkeypatch.chdir(tmp_path)  # type: ignore[attr-defined]
    f = tmp_path / "tiny.asm.toml"
    f.write_text(TINY, encoding="utf-8")
    spec = load_machine(f)
    instance = tmp_path / "machines" / "parked"
    instance.mkdir(parents=True)
    # A begun-but-not-ended journal: not `_ended`, and no worker.pid -> dead.
    (instance / "journal.jsonl").write_text(
        '{"kind": "machine.begin", "ts": "t", "machine": "tiny", "version": 1}\n',
        encoding="utf-8",
    )
    state = instance / "states" / "0001-work"
    state.mkdir(parents=True)
    (state / "logs.jsonl").write_text("", encoding="utf-8")

    class _WatchHost(App[None]):
        def on_mount(self) -> None:
            self.push_screen(MachineWatchScreen(instance, spec))

    async def scenario() -> None:
        app = _WatchHost()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            for _ in range(3):
                await pilot.pause()
            screen = app.screen
            assert isinstance(screen, MachineWatchScreen)
            assert not screen._ended  # pyright: ignore[reportPrivateUsage]
            assert screen.check_action("steer", ()) is False
            screen.action_steer()
            await pilot.pause()
            assert not (state / "steer.request").exists()

    asyncio.run(scenario())


def test_watch_header_reads_a_corrupt_wait_as_waiting(tmp_path: Path) -> None:
    """A corrupt pending-wait file counts as parked (the rule machine_is_parked
    documents: never read "dead pid" as "crashed" while a wait may be armed),
    so the watch header says "waiting", not "stopped"."""
    from textual.widgets import Static

    from agent6.machine import load_machine

    f = tmp_path / "tiny.asm.toml"
    f.write_text(TINY, encoding="utf-8")
    spec = load_machine(f)
    instance = tmp_path / "machines" / "tiny"
    instance.mkdir(parents=True)
    (instance / "journal.jsonl").write_text("", encoding="utf-8")  # started, not ended
    (instance / "wait.json").write_text("{ not json", encoding="utf-8")
    (instance / "worker.pid").write_text("999999999", encoding="utf-8")  # dead

    class _WaitHost(App[None]):
        def on_mount(self) -> None:
            self.push_screen(MachineWatchScreen(instance, spec))

    async def scenario() -> None:
        app = _WaitHost()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            for _ in range(3):
                await pilot.pause()
            screen = app.screen
            assert isinstance(screen, MachineWatchScreen)
            head = str(screen.query_one("#mw-head", Static).render())
            assert "waiting" in head
            assert "stopped" not in head

    asyncio.run(scenario())


def test_watch_footer_steer_key_follows_liveness(tmp_path: Path) -> None:
    """check_action("steer") reads _steerable(), but refresh_bindings only
    fired on the _ended edge -- a killed worker (or an --exit-on-wait park)
    kept the footer's Steer key lit for a machine nobody can steer. The poll
    now refreshes bindings when steerability flips."""
    import os

    from agent6.machine import load_machine

    f = tmp_path / "tiny.asm.toml"
    f.write_text(TINY, encoding="utf-8")
    spec = load_machine(f)
    instance = tmp_path / "machines" / "tiny"
    instance.mkdir(parents=True)
    (instance / "journal.jsonl").write_text("", encoding="utf-8")  # started, not ended
    (instance / "worker.pid").write_text(str(os.getpid()), encoding="utf-8")  # live

    class _LiveHost(App[None]):
        def on_mount(self) -> None:
            self.push_screen(MachineWatchScreen(instance, spec))

    async def scenario() -> None:
        app = _LiveHost()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, MachineWatchScreen)
            assert screen.check_action("steer", ()) is True  # live: the key is real
            (instance / "worker.pid").write_text("999999999", encoding="utf-8")  # dies
            for _ in range(4):  # let the 0.5s poll observe the flip
                await pilot.pause(0.3)
            assert screen.check_action("steer", ()) is False
            assert screen._was_steerable is False  # pyright: ignore[reportPrivateUsage]

    asyncio.run(scenario())


def _blocked_machine(tmp_path: Path, *, alive: bool) -> tuple[Path, MachineSpec]:
    """A machine instance whose newest agent state is blocked on an unanswered
    approval, with a live or dead worker."""
    import json
    import os

    from agent6.machine import load_machine

    f = tmp_path / "tiny.asm.toml"
    f.write_text(TINY, encoding="utf-8")
    spec = load_machine(f)
    instance = tmp_path / "machines" / "tiny"
    state = instance / "states" / "0000-route"
    state.mkdir(parents=True)
    instance.joinpath("journal.jsonl").write_text("", encoding="utf-8")  # started, not ended
    instance.joinpath("worker.pid").write_text(
        str(os.getpid()) if alive else "999999999", encoding="utf-8"
    )
    (state / "logs.jsonl").write_text(
        json.dumps({"type": "session.start", "mode": "run", "user_task": "t"})
        + "\n"
        + json.dumps({"type": "approval.prompt", "id": "ap1", "prompt": "Allow rm -rf"})
        + "\n",
        encoding="utf-8",
    )
    return instance, spec


def test_watch_screen_does_not_pop_prompts_on_a_dead_machine(tmp_path: Path) -> None:
    """The fold keeps an unanswered prompt in the newest agent state past a
    worker death, so the watch screen popped live-looking Allow/Deny (a
    destructive-command approval among them) over a machine nobody can answer
    and wrote the answer into a per-state dir whose loop has exited. The
    machine twin of the run-modal gate."""
    from textual.widgets import Static

    from agent6.ui.tui.modals import ApprovalModal

    instance, spec = _blocked_machine(tmp_path, alive=False)

    class _Host(App[None]):
        def on_mount(self) -> None:
            self.push_screen(MachineWatchScreen(instance, spec))

    async def scenario() -> None:
        app = _Host()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            for _ in range(4):  # let several polls run
                await pilot.pause()
            assert not isinstance(app.screen, ApprovalModal), "popped a modal on a dead machine"
            screen = app.screen
            assert isinstance(screen, MachineWatchScreen)
            assert "stopped" in str(screen.query_one("#mw-head", Static).render())

    asyncio.run(scenario())


def test_watch_screen_pops_prompts_on_a_live_machine(tmp_path: Path) -> None:
    # The converse: gating on liveness must not cost a RUNNING machine its modal.
    from agent6.ui.tui.modals import ApprovalModal

    instance, spec = _blocked_machine(tmp_path, alive=True)

    class _Host(App[None]):
        def on_mount(self) -> None:
            self.push_screen(MachineWatchScreen(instance, spec))

    async def scenario() -> None:
        app = _Host()
        async with app.run_test(size=(120, 40)) as pilot:
            deadline = 0
            while not isinstance(app.screen, ApprovalModal) and deadline < 80:
                await pilot.pause(0.05)
                deadline += 1
            assert isinstance(app.screen, ApprovalModal), "a live machine must still pop the modal"

    asyncio.run(scenario())


def test_machines_page_lists_instances_with_their_files(
    tmp_path: Path, monkeypatch: object
) -> None:
    """The page shows the rows `agent6 machine` lists: an instance's status
    and current state joined with its authored file, then the files no
    instance ran (blank status)."""
    from agent6.config.layer import resolved_state_dir
    from agent6.ui.cli import main as cli_main

    monkeypatch.chdir(tmp_path)  # type: ignore[attr-defined]
    f = tmp_path / "tiny.asm.toml"
    f.write_text(TINY, encoding="utf-8")
    (tmp_path / "other.asm.toml").write_text(TINY.replace('"tiny"', '"other"'), encoding="utf-8")
    assert cli_main(["machine", "run", str(f)]) == 0
    state_dir = resolved_state_dir(tmp_path)

    class _Host(App[None]):
        def on_mount(self) -> None:
            self.push_screen(MachinesScreen(state_dir, tmp_path))

    async def scenario() -> None:
        app = _Host()
        async with app.run_test(size=(140, 40)) as pilot:
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, MachinesScreen)
            table = screen.query_one("#machines", DataTable)
            assert table.row_count == 2
            rows = [[str(cell) for cell in table.get_row_at(i)] for i in range(table.row_count)]
            ran = next(r for r in rows if r[0] == "tiny")
            assert ran[1] == "ok" and ran[2] == "done" and ran[5] == "valid"
            never = next(r for r in rows if r[0] == "other")
            assert never[1] == "" and never[2] == "-" and never[3] == "-"

    asyncio.run(scenario())


def test_watch_screen_stop_on_a_parked_machine_says_why(
    tmp_path: Path, monkeypatch: object
) -> None:
    """`x` on a machine with no live worker prints the CLI's refusal. Disabling
    the binding made it a silent no-op that Help still listed."""
    from agent6.config.layer import resolved_state_dir
    from agent6.machine import load_machine
    from agent6.ui.cli import main as cli_main

    monkeypatch.chdir(tmp_path)  # type: ignore[attr-defined]
    f = _write(tmp_path / "waiter.asm.toml")
    assert cli_main(["machine", "run", str(f), "--exit-on-wait"]) == 0  # parks, worker gone
    instance = resolved_state_dir(tmp_path) / "machines" / "waiter_demo"
    spec = load_machine(f)

    class _WatchHost(App[None]):
        def on_mount(self) -> None:
            self.push_screen(MachineWatchScreen(instance, spec))

    async def scenario() -> None:
        app = _WatchHost()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            await pilot.press("x")
            await pilot.pause()
            notes = [str(n.message) for n in app._notifications]  # pyright: ignore[reportPrivateUsage]
            assert any("not running" in n for n in notes), notes
            assert not (instance / "stop").exists()

    asyncio.run(scenario())


def test_machine_run_confirm_backs_out_on_q(tmp_path: Path, monkeypatch: object) -> None:
    """The footer under the confirm reads "Esc/q Back": q backs out like Esc."""
    _write(tmp_path / "m.asm.toml")
    captured: list[list[str]] = []

    def _fake_spawn(argv: list[str], cwd: Path, **_k: object) -> str:
        captured.append(list(argv))
        return ""

    monkeypatch.setattr(machmod, "spawn_and_confirm", _fake_spawn)  # type: ignore[attr-defined]

    async def scenario() -> None:
        app = _Host(tmp_path)
        async with app.run_test() as pilot:
            await pilot.pause()
            table = app.screen.query_one("#machines", DataTable)
            table.focus()
            table.move_cursor(row=0)
            await pilot.press("r")
            await pilot.pause()
            assert isinstance(app.screen, ConfirmModal)
            await pilot.press("q")
            await pilot.pause()
            assert not isinstance(app.screen, ConfirmModal), "q did not back out"
            assert captured == []

    asyncio.run(scenario())


def test_a_live_machine_marks_only_the_turn_in_flight(tmp_path: Path) -> None:
    """The watch screen replays the whole state log on open and marked every
    historical `role.call` as thinking, gated on the MACHINE being live: three
    finished turns and one in flight read as four live markers."""
    import json
    import os
    from typing import Any

    from textual.app import ComposeResult
    from textual.widgets import RichLog

    from agent6.machine import load_machine
    from agent6.machine.journal import MachineJournal

    root = tmp_path / "hunt"
    root.mkdir()
    (root / "machine.asm.toml").write_text(
        """machine = "hunt"
version = 1
initial = "work"

[budget]
max_usd = 1.0
max_transitions = 100

[schemas.verdict]
approved = "bool"

[vars.agent]
verdict = { type = "verdict", default = {} }

[states.work]
kind = "agent"
model = "m1"
prompt = "do the thing"
output_schema = "verdict"
capture = { finish_json = "verdict" }
timeout_secs = 600
on = { ok = "done", failed = "done", budget_exhausted = "done", timeout = "done" }

[states.done]
kind = "terminal"
status = "ok"
reason = "done"
""",
        encoding="utf-8",
    )
    j = MachineJournal(root)
    j.ensure_dirs()
    j.begin(machine="hunt", version=1)
    sd = root / "states" / "0000-work"
    sd.mkdir(parents=True)
    evs: list[dict[str, Any]] = []
    for i in (1, 2, 3):  # three COMPLETED turns
        evs += [
            {"type": "role.call", "role": "worker", "model": "m1"},
            {"type": "role.thinking_delta", "text": f"turn {i} reasoning. "},
            {"type": "role.result", "role": "worker", "tokens_in": 10, "tokens_out": 5},
        ]
    evs.append({"type": "role.call", "role": "worker", "model": "m1"})  # in flight
    (sd / "logs.jsonl").write_text("".join(json.dumps(e) + "\n" for e in evs), encoding="utf-8")
    (root / "worker.pid").write_text(str(os.getpid()), encoding="utf-8")  # live

    class WatchApp(App[int]):
        def compose(self) -> ComposeResult:
            return iter(())

        def on_mount(self) -> None:
            self.push_screen(
                machmod.MachineWatchScreen(root, load_machine(root / "machine.asm.toml"))
            )

    out: list[str] = []

    async def scenario() -> None:
        app = WatchApp()
        async with app.run_test(size=(140, 40)) as pilot:
            for _ in range(14):
                await pilot.pause(0.25)
            log = app.screen.query_one("#mw-log", RichLog)
            out.append(
                "\n".join(
                    "".join(seg.text for seg in line._segments)  # pyright: ignore[reportPrivateUsage]
                    for line in log.lines
                )
            )

    asyncio.run(scenario())
    assert out[0].count("thinking…") == 1, out[0]
