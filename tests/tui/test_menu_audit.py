# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Every menu item, on every screen, must resolve to a real action handler --
so none silently does nothing (the "clicking Quit doesn't quit" class of bug,
which was an unawaited coroutine: the handler existed but the dispatch dropped
its result). Here we assert the handler is reachable; the dispatch-awaits-it
behaviour is covered by the quit-exits tests below.
"""

from __future__ import annotations

import asyncio
import json
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import cast

from rich.text import Text
from textual.app import App

from agent6.ui.tui.app import Agent6TUI
from agent6.ui.tui.config_page import ConfigScreen
from agent6.ui.tui.dashboard import DashboardScreen
from agent6.ui.tui.home import Agent6HomeApp
from agent6.ui.tui.menubar import Menu, _Dropdown


def _resolve(host: object, action: str) -> Callable[..., object] | None:
    # EXACTLY the real dispatcher (MenuBar.on_menu_bar_selected / palette
    # commands): getattr on the screen, then the app. No dotted-namespace
    # parsing -- a dotted "app.foo" item is dead in every real consumer, so the
    # audit must not resolve it either.
    app = getattr(host, "app", host)
    return getattr(host, f"action_{action}", None) or getattr(app, f"action_{action}", None)


def _assert_all_items_resolve(host: object, menus: tuple[Menu, ...]) -> None:
    missing = [
        item.action for menu in menus for item in menu.items if _resolve(host, item.action) is None
    ]
    assert not missing, f"menu items with no action handler: {missing}"


def test_home_menu_items_all_resolve() -> None:
    adir, repo = Path(tempfile.mkdtemp()), Path(tempfile.mkdtemp())

    async def scenario() -> None:
        app = Agent6HomeApp(adir, repo)
        async with app.run_test() as pilot:
            await pilot.pause()
            _assert_all_items_resolve(app.screen, app.screen.MENUS)  # type: ignore[attr-defined]

    asyncio.run(scenario())


def test_config_menu_items_all_resolve(tmp_path: Path) -> None:
    class _Host(App[None]):
        def on_mount(self) -> None:
            self.push_screen(ConfigScreen(tmp_path))

    async def scenario() -> None:
        app = _Host()
        async with app.run_test() as pilot:
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, ConfigScreen)
            _assert_all_items_resolve(screen, screen.MENUS)

    asyncio.run(scenario())


def test_dashboard_menu_items_all_resolve(tmp_path: Path) -> None:
    run = tmp_path / "run"
    run.mkdir()
    (run / "logs.jsonl").write_text(
        json.dumps({"type": "session.start", "mode": "run", "user_task": "x"}) + "\n",
        encoding="utf-8",
    )

    async def scenario() -> None:
        app = Agent6TUI(run)
        async with app.run_test() as pilot:
            await pilot.pause()
            # Both sibling views of the run app: the primary conversation and the
            # dashboard behind it (its menus must resolve even while covered).
            _assert_all_items_resolve(app._conv, app._conv.MENUS)
            assert isinstance(app._dash, DashboardScreen)
            _assert_all_items_resolve(app._dash, app._dash.MENUS)

    asyncio.run(scenario())


def test_quit_from_menu_exits_home() -> None:
    """The whole chain: select the Quit item -> OptionSelected -> MenuBar.Selected
    -> the (async) action_quit is awaited -> the app exits. Regression for the
    unawaited-coroutine bug."""
    adir, repo = Path(tempfile.mkdtemp()), Path(tempfile.mkdtemp())

    async def scenario() -> None:
        from agent6.ui.tui.menubar import MenuBar

        app = Agent6HomeApp(adir, repo)
        async with app.run_test() as pilot:
            await pilot.pause()
            mb = app.screen.query_one(MenuBar)
            mb.open("f")
            await pilot.pause()
            dd = next(iter(app.screen.query(_Dropdown)))
            qi = next(i for i in range(dd.option_count) if dd.get_option_at_index(i).id == "quit")
            dd.highlighted = qi
            await pilot.press("enter")
            await pilot.pause()
            await pilot.pause()
            assert app._running is False  # pyright: ignore[reportPrivateUsage]

    asyncio.run(scenario())


def test_f10_opens_menu_bar(tmp_path: Path) -> None:
    """F10 opens the menu bar (terminal-robust: some terminals eat Alt+f)."""

    class _Host(App[None]):
        def on_mount(self) -> None:
            self.push_screen(ConfigScreen(tmp_path))

    async def scenario() -> None:
        app = _Host()
        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.press("f10")
            await pilot.pause()
            assert len(list(app.screen.query(_Dropdown))) == 1

    asyncio.run(scenario())


def test_q_key_quits_home() -> None:
    """The footer's 'q Quit' must actually quit. A Screen doesn't inherit the
    App's built-in action_quit and the binding doesn't bubble to it, so
    HomeScreen defines its own -- else only Ctrl+Q (an app default) would work."""
    adir, repo = Path(tempfile.mkdtemp()), Path(tempfile.mkdtemp())

    async def scenario() -> None:
        app = Agent6HomeApp(adir, repo)
        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.press("q")
            await pilot.pause()
            assert app._running is False  # pyright: ignore[reportPrivateUsage]

    asyncio.run(scenario())


def test_menu_dropdown_keys_right_align_to_common_edge() -> None:
    """Dropdown shortcut keys share a right edge (labels left), so they line up in
    a column instead of floating a fixed gap after each varying-width label."""
    from agent6.ui.tui.menubar import MenuItem, _menu_options

    items = (
        MenuItem("New run/plan/ask", "a", "n"),
        MenuItem("Open selected", "b", "enter"),
        MenuItem("Theme…", "c", None),  # keyless
        MenuItem("Quit", "d", "q"),
    )
    # No live bindings map -> falls back to each item's own key hint.
    opts = {o.id: cast(Text, o.prompt).plain for o in _menu_options(items, {})}
    keyed = [opts["a"], opts["b"], opts["d"]]
    assert len({len(r) for r in keyed}) == 1  # all padded to one width => shared right edge
    assert opts["a"].endswith(" n") and opts["b"].endswith("Enter") and opts["d"].endswith(" q")
    assert opts["c"] == "Theme…"  # keyless row is just the label


def test_help_screen_closes_after_resize_reflow() -> None:
    """The help page reflows on resize via recompose, which replaces the focused
    #help-scroll. Focus must move to the new instance: left on the detached old
    one, its binding chain no longer reaches the screen and Esc/q/? stop
    closing the page (a keyboard trap; ttyd/vhs always resize right after
    mount, so the tour hit it on every run)."""
    adir, repo = Path(tempfile.mkdtemp()), Path(tempfile.mkdtemp())

    async def scenario() -> None:
        app = Agent6HomeApp(adir, repo)
        async with app.run_test(size=(190, 50)) as pilot:
            await pilot.pause()
            await pilot.press("question_mark")
            await pilot.pause()
            assert type(app.screen).__name__ == "HelpScreen"
            await pilot.resize_terminal(150, 40)  # triggers the reflow recompose
            await pilot.pause()
            await pilot.press("escape")
            await pilot.pause()
            assert type(app.screen).__name__ == "HomeScreen"

    asyncio.run(scenario())
