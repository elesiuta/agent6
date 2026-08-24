# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""A full-screen, scrollable view of one run's logs.jsonl (current or past).

The dashboard's live log pane is a small sliding window that snaps to the
bottom on every new line, so a fast run "plays through" with no way to scroll
back, and finished runs in the hub had no log view at all. `LogScreen` reads
a run's whole logs.jsonl, renders each STRUCTURAL event with the SAME one-line
formatter the dashboard uses (so the two read identically), and lets the
operator scroll -- and select/copy -- freely. It is read-only; reload re-reads
the file (a live run keeps appending).

Two deliberate choices:
- Ephemeral streaming deltas are skipped (see STREAM_DELTA_EVENTS): a reasoning
  model emits thousands of contentless `role.thinking_delta` events, which are
  noise in an audit log (the reasoning itself is in the conversation view). So
  are the loop-side mirrors (LOG_NOISE_EVENTS), as in the dashboard's log tail.
- The body is a `Static` inside a `VerticalScroll`, not a `RichLog`. A `RichLog`
  renders as line Strips, which the framework's text selection can't extract, so
  its text is not copyable; a `Static` renders as `Content` and is selectable.

Chrome matches every other screen: the File/View/Help menu bar (its shortcuts
drawn from the live bindings), the same PgUp/PgDn + Ctrl+Home/End scroll keys as
the conversation view, and Esc/q back.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import ClassVar

from rich.text import Text
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import VerticalScroll
from textual.screen import Screen
from textual.widgets import Footer, Static

from agent6.ui.tui.menubar import (
    Menu,
    MenuBar,
    MenuItem,
    menu_bindings,
)
from agent6.ui.tui.screen_chrome import ScreenChrome
from agent6.viewmodel.log_line import format_log_line
from agent6.viewmodel.state import LOG_NOISE_EVENTS, STREAM_DELTA_EVENTS
from agent6.viewmodel.tail import LogTail


class LogScreen(ScreenChrome, Screen[None]):
    """Scrollable, read-only, selectable log of a single run (live or finished)."""

    CSS = """
    LogScreen { background: $surface; }
    #logview-scroll { height: 1fr; }
    #logview-body { height: auto; padding: 0 1; pointer: text; }  /* selectable: I-beam */
    """

    HELP_TITLE: ClassVar = "agent6 — log"
    MENUS: ClassVar = (
        Menu("File", (MenuItem("Back", "close"),)),
        Menu(
            "View",
            (
                MenuItem("Scroll ↑ a page", "page_up"),
                MenuItem("Scroll ↓ a page", "page_down"),
                MenuItem("Scroll → top", "scroll_top"),
                MenuItem("Scroll → end", "scroll_bottom"),
                MenuItem("Reload", "reload"),
            ),
        ),
        Menu(
            "Help",
            (MenuItem("Keys & actions", "help"), MenuItem("Command palette", "command_palette")),
        ),
    )

    BINDINGS: ClassVar = [
        Binding("escape", "close", "Back", key_display="Esc/q"),
        Binding("q", "close", "Back", show=False),
        # l closes too: the key that opened the view (dashboard `l`, hub `l`)
        # toggles it shut, so open/close is one keystroke from either side.
        Binding("l", "close", "Back", show=False),
        Binding("r", "reload", "Reload"),
        Binding("pageup", "page_up", "Scroll up", show=False),
        Binding("pagedown", "page_down", "Scroll down", show=False),
        Binding("ctrl+home", "scroll_top", "Top", show=False),
        Binding("ctrl+end", "scroll_bottom", "End", show=False),
        *menu_bindings(MENUS),
    ]

    def __init__(self, logs_path: Path, *, title: Callable[[], str]) -> None:
        super().__init__()
        self._logs_path = logs_path
        self._title = title
        self._tail = LogTail(logs_path)
        self._text = Text()

    def compose(self) -> ComposeResult:
        yield MenuBar(self.MENUS)  # top row: menus + "agent6 — <run>", like every screen
        with VerticalScroll(id="logview-scroll"):
            yield Static(id="logview-body")  # renders as Content -> its text is selectable
        yield Footer()

    def on_mount(self) -> None:
        self.app.sub_title = self._title()  # show the run in the menu bar's title
        self._reload()
        # Follow live: a resume appends to the same file, so keep reading.
        self.set_interval(0.5, self._poll)

    def _scroll(self) -> VerticalScroll:
        return self.query_one("#logview-scroll", VerticalScroll)

    def _append(self, events: list[dict[str, object]]) -> bool:
        added = False
        for event in events:
            if event.get("type") in STREAM_DELTA_EVENTS or event.get("type") in LOG_NOISE_EVENTS:
                continue
            self._text.append(format_log_line(event) + "\n")
            added = True
        return added

    def _reload(self) -> None:
        self._tail = LogTail(self._logs_path)
        self._text = Text()
        self._append(self._tail.read())
        shown = self._text if len(self._text) else Text("(no events yet)", style="dim italic")
        self.query_one("#logview-body", Static).update(shown)
        self._scroll().scroll_end(animate=False)
        self._scroll().focus()

    def _poll(self) -> None:
        scroll = self._scroll()
        at_bottom = scroll.is_vertical_scroll_end
        if not self._append(self._tail.read()):
            return
        self.query_one("#logview-body", Static).update(self._text)
        if at_bottom:  # sticky bottom: hold position if the operator scrolled up
            scroll.scroll_end(animate=False)

    def action_reload(self) -> None:
        self._reload()

    def action_page_up(self) -> None:
        self._scroll().scroll_page_up(animate=False)  # instant: animation reads as lag

    def action_page_down(self) -> None:
        self._scroll().scroll_page_down(animate=False)

    def action_scroll_top(self) -> None:
        self._scroll().scroll_home(animate=False)

    def action_scroll_bottom(self) -> None:
        self._scroll().scroll_end(animate=False)

    def action_close(self) -> None:
        self.dismiss()
