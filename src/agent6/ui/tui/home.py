# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The `agent6 tui` hub: a home screen to browse recent runs and start new work.

CLI-first by design, the hub never reimplements the workflow. "Start a run /
plan / ask" spawns the normal `agent6` CLI as a detached subprocess
(whose non-TTY stdout means it won't try to open its own TUI) and then opens the
read-only dashboard on the run directory it creates. So everything here is a
thin driver over the CLI + the same file/event contract the dashboard reads.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from pathlib import Path
from typing import ClassVar

try:
    from rich.text import Text
    from textual.app import App, ComposeResult, SystemCommand
    from textual.binding import Binding
    from textual.screen import Screen
    from textual.widgets import DataTable, Footer
except ImportError as e:  # pragma: no cover - clear runtime message
    raise ImportError(
        "agent6 TUI requires the 'textual' package (part of the base install)."
        " Reinstall agent6, or `pip install textual`."
    ) from e

# Safe at module top: the textual guard above runs first, so this (which also
# needs textual) is only reached when textual is present.
from agent6.config import ConfigError
from agent6.config.layer import available_preset_names, load_effective
from agent6.git_ops import run_branch_tips
from agent6.sessions.layout import LOGS_NAME
from agent6.ui.spawn import agent6_argv, run_cli_capture
from agent6.ui.tui.config_page import ConfigScreen
from agent6.ui.tui.logview import LogScreen
from agent6.ui.tui.machines import MachinesScreen
from agent6.ui.tui.menubar import Menu, MenuBar, MenuItem, menu_bindings
from agent6.ui.tui.modals import ConfirmModal
from agent6.ui.tui.new_work import NewWorkScreen, available_models
from agent6.ui.tui.screen_chrome import MenuCommands, ScreenChrome
from agent6.ui.tui.theme import (
    PALETTE_CSS,
    MuxPointerShapes,
    PlainNotify,
    setup_theme,
    status_style,
)
from agent6.viewmodel import (
    LIVE_STATUS_WORDS,
    SessionSummary,
    is_winner,
    session_dirs,
    summarize_session_dir,
    task_snippet,
)
from agent6.viewmodel.format import (
    format_cost_cell,
    format_when,
    listing_status_label,
    winner_id,
)

# The hub re-asks on this cadence (matching the web hub's poll rate), so a
# session that ends while you watch stops reading as running.
_HUB_POLL_S = 4.0


def _status_cell(summary: SessionSummary) -> Text:
    label = listing_status_label(
        summary.mode, summary.status, summary.reason, unmerged=summary.unmerged
    )
    return Text(label, style=status_style(summary.status))


class HomeScreen(ScreenChrome, Screen[None]):
    """The hub view: browse recent runs, start new work, open the config editor.
    Its bindings live here (not on the App) so the footer of a pushed screen --
    e.g. the config editor -- shows only that screen's keys, not the hub's."""

    MENUS: ClassVar = (
        Menu(
            "File",
            (
                MenuItem("New task", "new_work", "n"),
                MenuItem("Open selected", "open_selected", "enter"),
                MenuItem("Merge selected run", "merge_selected", "m"),
                MenuItem("Delete selected run…", "delete_selected", "d"),
                MenuItem("Refresh", "refresh", "r"),
                MenuItem("Quit", "quit", "q"),
            ),
        ),
        Menu("Config", (MenuItem("Open config", "open_config", "c"),)),
        Menu("Machines", (MenuItem("Open machines", "open_machines", "M"),)),
        Menu(
            "View",
            (
                # Viewing a selected run's raw event log is filed under View to
                # match the run views' View menus. There is no separate
                # transcript viewer: Enter opens the run on its conversation.
                MenuItem("View logs", "view_logs", "l"),
                MenuItem("Theme…", "choose_theme"),
                MenuItem("Copy method…", "choose_copy_method"),
            ),
        ),
        Menu(
            "Help",
            (
                MenuItem("Keys & actions", "help", "question_mark"),
                MenuItem("Command palette", "command_palette", "ctrl+p"),
            ),
        ),
    )
    BINDINGS: ClassVar = [
        # Footer order: run-list actions, then Config, then meta (Help, Quit, Menu).
        Binding("n", "new_work", "New run/plan/ask"),
        Binding("enter", "open_selected", "Open"),
        Binding("l", "view_logs", "View logs"),
        Binding("m", "merge_selected", "Merge run"),
        Binding("d", "delete_selected", "Delete run"),
        Binding("r", "refresh", "Refresh"),
        Binding("c", "open_config", "Config"),
        Binding("M", "open_machines", "Machines"),
        Binding("question_mark", "help", "Help"),
        Binding("q", "quit", "Quit"),
        *menu_bindings(MENUS),
    ]
    COMMANDS: ClassVar = Screen.COMMANDS | {MenuCommands}
    HELP_HINTS: ClassVar = (
        "Enter opens the selected run",
        "Pickers: ↑↓ highlight · Space selects",
    )

    def __init__(self, agent6_dir: Path, repo_cwd: Path, config_path: Path | None = None) -> None:
        super().__init__()
        self.agent6_dir = agent6_dir
        # The repo to launch new runs in. The state dir is out of the workspace,
        # so it can't be derived from agent6_dir; the caller passes it.
        self.repo_cwd = repo_cwd
        # The hub's `--config F`, stamped into everything it spawns or loads.
        self.config_path = config_path
        self._runs: list[Path] = []
        # The rows the poll built, by run id: `check_action` reads them instead
        # of re-folding a session on every binding refresh.
        self._summaries: dict[str, SessionSummary] = {}

    def compose(self) -> ComposeResult:
        yield MenuBar(self.MENUS)  # the top row: menus + "agent6 — <path>"
        yield DataTable(id="sessions")
        yield Footer()

    def on_mount(self) -> None:
        table = self.query_one("#sessions", DataTable)
        table.cursor_type = "row"
        table.add_columns("updated", "status", "cost", "id", "task")
        self.action_refresh()
        table.focus()
        self.set_interval(_HUB_POLL_S, self._poll)

    def on_screen_resume(self) -> None:
        # Returning from a pushed screen (e.g. config) doesn't re-run on_mount, so
        # refresh -- which also resets the menu-bar sub_title that config changed
        # to "config · …" (otherwise the hub keeps showing "agent6 — config").
        self.action_refresh()

    def _poll(self) -> None:
        # Only while the hub is the top screen: a pushed screen refreshes on
        # resume anyway, and a rebuild under an open modal would shift the rows
        # out from under it.
        if self.app.screen is self:
            self.action_refresh()

    def action_refresh(self) -> None:
        table = self.query_one("#sessions", DataTable)
        # The poll rebuilds the whole table; keep the operator's selection by
        # run id, not row index -- new activity reorders the rows.
        selected = ""
        if self._runs and 0 <= table.cursor_row < len(self._runs):
            selected = self._runs[table.cursor_row].name
        table.clear()
        # Keep self._runs 1:1 with the table rows: a run dir that vanished between
        # the listing and its stat() must be dropped from BOTH, or every
        # cursor_row-indexed selection action (open/logs/merge) maps to the wrong
        # run for cursor positions past the gap.
        survivors: list[Path] = []
        rows: dict[str, SessionSummary] = {}
        tips = run_branch_tips(self.repo_cwd)
        for rd in session_dirs(self.agent6_dir):
            if not rd.is_dir():
                continue  # vanished since the listing snapshot — skip it
            s = summarize_session_dir(rd, branch_tips=tips)
            # Text cells: task is model/user input and may carry markup brackets.
            table.add_row(
                format_when(s.mtime),
                _status_cell(s),
                format_cost_cell(s.cost_usd, partial=s.usd_partial),
                Text(winner_id(s.session_id, winner=is_winner(rd))),
                Text(task_snippet(s.task, max_chars=60)),
            )
            survivors.append(rd)
            rows[rd.name] = s
        self._runs = survivors
        self._summaries = rows
        if selected:
            row = next((i for i, rd in enumerate(survivors) if rd.name == selected), None)
            if row is not None:
                table.move_cursor(row=row)
        # Useful context in the header sub-title rather than a duplicate hint bar.
        # "sessions", not "runs": this hub lists every bucket, so a hub of one
        # run, one plan and one ask announced "3 runs".
        count = len(self._runs)
        # The empty state says what to do next, as the CLI and the web do and as
        # the machines screen next door does: a blank table is not an answer.
        tally = (
            'no sessions yet (n starts one, or: agent6 run "<task>")'
            if not count
            else f"{count} session{'' if count == 1 else 's'}"
        )
        self.app.sub_title = f"{self.repo_cwd} · {tally}"
        # An empty table shouldn't paint a full-height focus cursor over its body.
        table.show_cursor = table.row_count > 0

    def action_open_selected(self) -> None:
        table = self.query_one("#sessions", DataTable)
        if self._runs and 0 <= table.cursor_row < len(self._runs):
            self.app.exit(self._runs[table.cursor_row])

    def action_view_logs(self) -> None:
        """Open a scrollable, read-only log of the selected run (current or
        finished) without leaving the hub -- the run list only shows a one-line
        status, so this is how you read what a past run actually did."""
        table = self.query_one("#sessions", DataTable)
        if not (self._runs and 0 <= table.cursor_row < len(self._runs)):
            return
        session_dir = self._runs[table.cursor_row]
        self.app.push_screen(
            LogScreen(session_dir / LOGS_NAME, title=lambda: f"logs · {session_dir.name}")
        )

    def on_data_table_row_selected(self, _event: DataTable.RowSelected) -> None:
        # Enter / double-click a run row opens it. The DataTable consumes Enter
        # for its own RowSelected, so the screen's `enter` binding never fires --
        # handle the row event itself instead.
        self.action_open_selected()

    def action_quit(self) -> None:
        # On the App, `quit` is a built-in; on a Screen it isn't, and the binding
        # doesn't bubble to it -- so define it here, or the footer's "q Quit"
        # would lie (only Ctrl+Q, an app-level default, would work).
        self.app.exit()

    def action_new_work(self) -> None:
        self.app.push_screen(
            NewWorkScreen(
                self.repo_cwd,
                self.config_path,
                presets=available_preset_names(self.repo_cwd, self.config_path),
                models=available_models(self.repo_cwd, self.config_path),
            )
        )

    def check_action(self, action: str, parameters: tuple[object, ...]) -> bool | None:
        """Grey Merge and Delete out on a LIVE run, which `sessions merge` and
        `sessions rm` always refuse
        (the web disables the same button and says why). None, not False:
        False also HIDES the key, and a key missing from the footer reads as a
        capability this hub does not have.

        The other refusals (no commits, already merged) are the CLI's to make:
        deciding them here needs a git read per selection, and the summary's
        `unmerged` mark is branch-derived, so it reads False for a run whose
        commits live only on its chain ref -- one the CLI merges fine.
        """
        del parameters
        if action in ("merge_selected", "delete_selected"):
            rd = self._selected_dir()
            if rd is None:
                return None
            s = self._summaries.get(rd.name)
            return None if s is not None and s.status in LIVE_STATUS_WORDS else True
        return True

    def on_data_table_row_highlighted(self, _event: DataTable.RowHighlighted) -> None:
        # Moving the cursor changes what the row actions would act on, and
        # Textual re-asks `check_action` only on a bindings refresh: without
        # this the footer describes the row selected when the screen mounted.
        self.refresh_bindings()

    def _selected_dir(self) -> Path | None:
        """The run dir under the cursor, or None on an empty table."""
        table = self.query_one("#sessions", DataTable)
        if not (self._runs and 0 <= table.cursor_row < len(self._runs)):
            return None
        return self._runs[table.cursor_row]

    def action_merge_selected(self) -> None:
        """Merge the selected run's branch into its base, after a confirm. The TUI
        shells out to `agent6 sessions merge` (never git_ops directly); the CLI applies
        git.merge_strategy and refuses a dirty tree / unconfigured identity."""
        table = self.query_one("#sessions", DataTable)
        if not (self._runs and 0 <= table.cursor_row < len(self._runs)):
            return
        session_id = self._runs[table.cursor_row].name
        self.app.push_screen(
            ConfirmModal(
                f"Merge run {session_id}?",
                "Runs `agent6 sessions merge` to land this run's branch on its base using your "
                "git.merge_strategy. Ref plumbing only: the checkout never moves.",
                confirm_label="Merge",
            ),
            self._on_merge_confirm(session_id),
        )

    def action_delete_selected(self) -> None:
        """Delete the selected run's history, after a confirm: the same verb the
        run view's menu offers, from the row where the run is selected. History
        only: the run branch and its commits are git's (`sessions prune` is the
        branch verb)."""
        table = self.query_one("#sessions", DataTable)
        if not (self._runs and 0 <= table.cursor_row < len(self._runs)):
            return
        session_id = self._runs[table.cursor_row].name
        self.app.push_screen(
            ConfirmModal(
                f"Delete run {session_id}'s history?",
                "Runs `agent6 sessions rm`: removes its transcripts, events and manifest from"
                " the state dir. The run branch and its commits are kept.",
                confirm_label="Delete",
            ),
            self._on_delete_confirm(session_id),
        )

    def _on_delete_confirm(self, session_id: str) -> Callable[[bool | None], None]:
        def cb(confirmed: bool | None) -> None:
            if not confirmed:
                return
            ok, msg = _run_delete_cli(self.repo_cwd, session_id, self.config_path)
            self.app.notify(msg, severity="information" if ok else "error", timeout=10.0)
            self.action_refresh()

        return cb

    def _on_merge_confirm(self, session_id: str) -> Callable[[bool | None], None]:
        def cb(confirmed: bool | None) -> None:
            if not confirmed:
                return
            ok, msg = _run_merge_cli(self.repo_cwd, session_id, self.config_path)
            self.app.notify(msg, severity="information" if ok else "error", timeout=10.0)
            self.action_refresh()

        return cb

    def action_open_config(self) -> None:
        # An invalid config (e.g. a stale value or a leftover table from a removed
        # feature) would crash the config screen on load. Pre-check so we can point
        # at `agent6 config fix` instead of taking down the TUI.

        try:
            load_effective(self.repo_cwd, self.config_path)
        except ConfigError as exc:
            self.app.notify(
                "Config is invalid, so it can't be opened. Run `agent6 config fix` in a"
                f" terminal to drop invalid entries, then reopen.\n{exc}",
                severity="error",
                timeout=15.0,
            )
            return
        self.app.push_screen(ConfigScreen(self.repo_cwd, self.config_path))

    def action_open_machines(self) -> None:
        self.app.push_screen(MachinesScreen(self.agent6_dir, self.repo_cwd, self.config_path))


class Agent6HomeApp(PlainNotify, MuxPointerShapes, App[Path | None]):
    """Home hub. `run()` returns the run directory the user chose to open (to be
    watched by the dashboard), or None to quit. A thin shell around
    :class:`HomeScreen` so the hub's key bindings stay screen-scoped."""

    TITLE = "agent6"
    CSS = (
        PALETTE_CSS
        + """
    Screen { layers: base dropdown; background: $surface; }
    /* The flat Screen rule above also matches ModalScreens, which would make
       their backdrops opaque; restore textual's translucent dim (same
       specificity, later rule wins) so the screen shows through behind dialogs. */
    ModalScreen { background: $background 60%; }
    * { scrollbar-size-vertical: 1; scrollbar-size-horizontal: 1; }  /* half the 2-wide default */
    /* A footer that does not fit clips (textual's default); the 1-row widget has no
       room for the scrollbar the universal rule gives it, which replaced every hint. */
    Footer { scrollbar-size-vertical: 0; scrollbar-size-horizontal: 0; }
    /* I-beam over anything you can type into (kitty OSC 22; inert elsewhere). */
    Input, TextArea { pointer: text; }
    #sessions { height: 1fr; border: round $primary; background: $surface; }
    #sessions:focus { border: round $accent; }
    /* Panel-coloured header bar (matches the menu bar + footer + config header),
       and a selection bar only when focused. */
    #sessions > .datatable--header { background: $panel; color: $foreground; text-style: bold; }
    #sessions > .datatable--cursor { background: transparent; color: $foreground; }
    #sessions:focus > .datatable--cursor {
        background: $primary 40%; color: $text; text-style: bold;
    }
    """
    )

    def __init__(self, agent6_dir: Path, repo_cwd: Path, config_path: Path | None = None) -> None:
        super().__init__()
        self.agent6_dir = agent6_dir
        self.repo_cwd = repo_cwd
        self.config_path = config_path

    def on_mount(self) -> None:
        setup_theme(self)  # apply the saved theme before the first paint
        self.push_screen(HomeScreen(self.agent6_dir, self.repo_cwd, self.config_path))

    def get_system_commands(self, screen: Screen[object]) -> Iterable[SystemCommand]:
        # Drop textual's "Keys" panel (our Help page replaces it), "Screenshot"
        # (an unused default whose SVG export is broken in our terminals), and
        # "Theme" (replaced by our live-preview Theme… picker). Every home action,
        # including Open config / Theme… / Keys & actions, is searchable by name
        # via _HomeCommands, so nothing is added here.
        for cmd in super().get_system_commands(screen):
            if cmd.title not in ("Keys", "Screenshot", "Theme"):
                yield cmd


def _run_merge_cli(
    repo_cwd: Path, session_id: str, config_path: Path | None = None
) -> tuple[bool, str]:
    """Run `agent6 sessions merge <session_id>` (capturing output) and return (ok, message).
    The hub shells out to the same CLI a user would, so merging stays a CLI concern
    and the UI never touches git_ops. Synchronous: a merge is a quick git op."""
    return run_cli_capture([*agent6_argv(config_path), "sessions", "merge", session_id], repo_cwd)


def _run_delete_cli(
    repo_cwd: Path, session_id: str, config_path: Path | None = None
) -> tuple[bool, str]:
    """Run `agent6 sessions rm -- <session_id>` and return (ok, message): the
    hub shells out to the CLI, like merge, so deletion stays a CLI concern."""
    ok, msg = run_cli_capture(
        [*agent6_argv(config_path), "sessions", "rm", "--", session_id], repo_cwd
    )
    return ok, msg or ("removed" if ok else "could not remove")


def run_home(agent6_dir: Path, repo_cwd: Path, config_path: Path | None = None) -> Path | None:
    return Agent6HomeApp(agent6_dir, repo_cwd, config_path).run()
