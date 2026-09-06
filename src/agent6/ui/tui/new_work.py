# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The conversation view before there is a conversation.

`n` on the hub opens it: the same chrome as a run's conversation (menu bar,
transcript pane, composer bar, footer), with the transcript pane empty and a
mode + preset row above the composer. Enter starts the run / plan / ask
detached and hands the located session to the live view; a start refusal
renders where the transcript will be, selectable, and the typed text stays in
the composer to fix and resend.
"""

from __future__ import annotations

import contextlib
from pathlib import Path
from typing import ClassVar

from rich.text import Text
from textual import on, work
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.css.query import NoMatches
from textual.screen import Screen
from textual.widgets import Footer, Select, Static, TextArea

from agent6.config import ConfigError
from agent6.config.layer import load_effective
from agent6.directive import spec_fragment
from agent6.models.validate import known_models
from agent6.types import OPERATOR_MODES
from agent6.ui.spawn import spawn_new_work
from agent6.ui.tui.composer import SteerInput, SteerSuggest
from agent6.ui.tui.menubar import Menu, MenuBar, MenuItem, menu_bindings
from agent6.ui.tui.screen_chrome import MenuCommands, ScreenChrome

# The preset dropdown's first entry: "" => no --preset, so the run uses the
# top-level `preset` from config (or the plain defaults).
DEFAULT_PRESET_LABEL = "(config default)"

_INTRO = (
    "Describe the task (or the question, for ask). Enter starts it; Ctrl-J adds a line.\n"
    "Tab reaches the mode and preset pickers below.\n"
    "/parallel [N|models] <task> fans out isolated lanes (repeat to queue more)."
)


def available_models(repo_cwd: Path, config_path: Path | None) -> list[str]:
    """Model ids for the `/parallel` autocomplete: the worker's model plus the
    worker provider's cached listing (cache-only, no network), exactly the set
    `run --parallel` validation accepts. Empty on any config error."""
    try:
        cfg = load_effective(repo_cwd, config_path).config
    except ConfigError:
        return []
    return sorted(known_models(cfg))


def model_suggestions(models: list[str], text: str, *, limit: int = 8) -> Text | None:
    """The suggestion line for a `/parallel` spec fragment under the caret:
    matching model ids (prefix matches first), or None when the caret is not
    in a spec token or there is nothing to offer."""
    frag = spec_fragment(text)
    if frag is None or not models:
        return None
    q = frag.lower()
    starts = [m for m in models if m.lower().startswith(q)]
    rest = [m for m in models if q in m.lower() and not m.lower().startswith(q)]
    shown = (starts + rest)[:limit]
    if not shown:
        return Text("no matching model ids", style="dim")
    total = len(starts) + len(rest)
    more = f"  (+{total - len(shown)} more, keep typing)" if total > len(shown) else ""
    return Text("models: ", style="dim") + Text("  ".join(shown)) + Text(more, style="dim")


class NewWorkScreen(ScreenChrome, Screen[None]):
    """Type a task, pick a mode and a preset, Enter starts it (see the module
    docstring). Lives in the hub app: a located session dir is the hub's
    return value, and the run view opens on it."""

    CSS = """
    NewWorkScreen { background: $surface; }
    #draft-main { height: 1fr; }
    #draft-scroll { height: 1fr; }
    #draft-notice { height: auto; padding: 0 1; pointer: text; }
    #draft-options { height: 3; padding: 0 1; }
    .draft-label { width: auto; padding: 1 1 0 0; color: $text-muted; }
    #draft-mode { width: 14; }
    #draft-preset { width: 1fr; max-width: 40; }
    #draft-input { height: auto; max-height: 8; border: round $primary; background: $surface; }
    #draft-input:focus { border: round $accent; }
    """

    MENUS: ClassVar = (
        Menu("File", (MenuItem("Back", "close"), MenuItem("Quit", "quit_hub", "ctrl+q"))),
        Menu(
            "View",
            (MenuItem("Theme…", "choose_theme"), MenuItem("Copy method…", "choose_copy_method")),
        ),
        Menu(
            "Help",
            (MenuItem("Keys & actions", "help"), MenuItem("Command palette", "command_palette")),
        ),
    )
    BINDINGS: ClassVar = [
        Binding("escape", "close", "Back", key_display="Esc", priority=True),
        Binding("ctrl+q", "quit_hub", "Quit", priority=True, show=False),
        Binding("question_mark", "help", "Help", show=False),
        *menu_bindings(MENUS),
    ]
    COMMANDS: ClassVar = Screen.COMMANDS | {MenuCommands}
    HELP_TITLE: ClassVar = "agent6 — new task"
    HELP_HINTS: ClassVar = (
        "Enter starts the task; Ctrl-J or Shift+Enter inserts a newline",
        "Tab moves between the text, the mode and the preset",
    )

    def __init__(
        self,
        repo_cwd: Path,
        config_path: Path | None = None,
        *,
        presets: list[str] | None = None,
        models: list[str] | None = None,
    ) -> None:
        super().__init__()
        self.repo_cwd = repo_cwd
        self.config_path = config_path
        self._presets = presets if presets is not None else []
        self._models = models if models is not None else []
        self._starting = False

    def compose(self) -> ComposeResult:
        yield MenuBar(self.MENUS)
        with Vertical(id="draft-main"), VerticalScroll(id="draft-scroll"):
            yield Static(Text(_INTRO, style="dim italic"), id="draft-notice")
        yield SteerSuggest(id="draft-suggest")
        with Horizontal(id="draft-options"):
            yield Static("mode", classes="draft-label")
            yield Select(
                [(m, m) for m in OPERATOR_MODES], value="run", allow_blank=False, id="draft-mode"
            )
            yield Static("preset", classes="draft-label")
            # value="" is the "(config default)" sentinel: no --preset, so the
            # config's own `preset` (or plain defaults) applies.
            yield Select(
                [(DEFAULT_PRESET_LABEL, ""), *((p, p) for p in self._presets)],
                value="",
                allow_blank=False,
                id="draft-preset",
            )
        yield SteerInput(id="draft-input")
        yield Footer()

    def on_mount(self) -> None:
        self.app.sub_title = "new task"
        bar = self.query_one("#draft-input", SteerInput)
        bar.set_mode(mode="start")
        bar.focus()

    def action_close(self) -> None:
        self.dismiss(None)

    def action_quit_hub(self) -> None:
        self.app.exit(None)

    @on(TextArea.Changed, "#draft-input")
    def _on_task_changed(self, event: TextArea.Changed) -> None:
        self.query_one("#draft-suggest", SteerSuggest).show_text(
            model_suggestions(self._models, event.text_area.text)
        )

    def on_steer_input_submitted(self, message: SteerInput.Submitted) -> None:
        if self._starting:
            return
        mode = str(self.query_one("#draft-mode", Select).value)
        preset = str(self.query_one("#draft-preset", Select).value)
        self._starting = True
        self._notice(Text(f"starting the {mode}…", style="bold cyan"))
        self._start(mode, message.text, preset)

    @work(thread=True, exclusive=True)
    def _start(self, mode: str, task: str, preset: str) -> None:
        """Spawn detached and locate the session, off the UI thread: the locate
        waits for the run's first event, which can take seconds."""
        session_dir, err = spawn_new_work(
            self.repo_cwd, mode, task, preset=preset, config_path=self.config_path
        )
        self.app.call_from_thread(self._started, session_dir, err, task)

    def _started(self, session_dir: Path | None, err: str, task: str) -> None:
        self._starting = False
        if session_dir is not None:
            self.app.exit(session_dir)
            return
        # The refusal is the conversation so far: selectable, and above the
        # text it refused, which is back in the composer to fix and resend.
        self._notice(Text(err or "could not start", style="bold red"))
        with contextlib.suppress(NoMatches):
            bar = self.query_one("#draft-input", SteerInput)
            bar.load_text(task)
            bar.move_cursor(bar.document.end)
            bar.focus()

    def _notice(self, text: Text) -> None:
        self.query_one("#draft-notice", Static).update(text)
