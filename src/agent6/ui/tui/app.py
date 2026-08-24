# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The agent6 run dashboard (`agent6 run` / `agent6 attach` / `agent6 tui`).

`textual` ships in the base install; importing this module fails clearly if it
has been stripped out. The CLI imports it lazily.

Architecture:
- `Agent6TUI` (the App) is the data plane: a background thread tails
  logs.jsonl -> apply_event -> call_from_thread, and the app owns the folded
  SessionState, the approval/question/steer prompt dispatch, run control (steer /
  stop / resume / fork), and the exit codes.
- `DashboardScreen` is the presentation: the panes, their key bindings and
  menus, and the coalesced repaint of the app's SessionState.

The dashboard is READ-ONLY on the log stream and only writes the answer files
the workflow polls: `<session_dir>/approvals/<id>.answer` (approve), `.../questions/
<id>.answer` (ask_user), `<session_dir>/steer.answer` (steer), and the
`<session_dir>/compact.request` marker (Compact now). Any other front-end can
mirror this contract.
"""

from __future__ import annotations

import contextlib
import os
import subprocess
import threading
import time
from collections.abc import Iterable
from pathlib import Path
from typing import ClassVar

try:
    from rich.markup import escape
    from rich.text import Text
    from textual.app import App, ComposeResult, SystemCommand
    from textual.binding import Binding
    from textual.containers import Horizontal, ScrollableContainer, VerticalScroll
    from textual.css.query import NoMatches
    from textual.screen import ModalScreen, Screen
    from textual.scroll_view import ScrollView
    from textual.widget import Widget
    from textual.widgets import (
        Checkbox,
        DataTable,
        Footer,
        RichLog,
        Select,
        Static,
        TextArea,
        Tree,
    )
except ImportError as e:  # pragma: no cover - clear runtime message
    raise ImportError(
        "agent6 TUI requires the 'textual' package (part of the base install)."
        " Reinstall agent6, or `pip install textual`."
    ) from e

from agent6.app.fork import create_fork, undo_fork
from agent6.app.reporter import Reporter
from agent6.config.layer import available_preset_names
from agent6.directive import parse_btw, parse_compact
from agent6.git_ops import commit_diff, diff_range
from agent6.sessions.ipc import (
    listening_ports,
    register_frontend,
    request_compact,
    request_stop,
    submit_steer,
    unregister_frontend,
)
from agent6.sessions.layout import LOGS_NAME, bucket_dir, layout_of
from agent6.sessions.manifest import ManifestError, read_manifest
from agent6.tools.background import shells_text
from agent6.ui.btw import open_btw
from agent6.ui.spawn import (
    DETACHED_RUN_ENV,
    agent6_argv,
    run_cli_capture,
    spawn_and_locate,
    spawn_detached_resume,
)
from agent6.ui.tui import clipboard
from agent6.ui.tui.conversation import (
    RUN_MENU,
    ComposerMode,
    ConversationScreen,
    ResumePreset,
    SteerInput,
    SteerSuggest,
    open_history_search,
)
from agent6.ui.tui.logview import LogScreen
from agent6.ui.tui.menubar import Menu, MenuBar, MenuItem, menu_bindings
from agent6.ui.tui.modals import (
    ConfirmModal,
    TextModal,
    ToolCallDetailModal,
)
from agent6.ui.tui.prompts import PromptDispatcher
from agent6.ui.tui.screen_chrome import MenuCommands, ScreenChrome
from agent6.ui.tui.settings import get_copy_method
from agent6.ui.tui.theme import (
    PALETTE_CSS,
    MuxPointerShapes,
    PlainNotify,
    setup_theme,
    status_style,
)
from agent6.viewmodel import manifest_branches, manifest_header, restate, session_compare
from agent6.viewmodel.events import SESSION_START_EVENTS
from agent6.viewmodel.format import (
    TASK_STATUS_GLYPH,
    format_compare,
    format_cost,
    spinner_frame,
    status_label,
)
from agent6.viewmodel.listing import (
    LIVE_STATUS_WORDS,
    finished_needs_new_work,
    status_for_session_dir,
    task_snippet,
)
from agent6.viewmodel.state import (
    MAX_LOG_TAIL,
    STREAM_DELTA_EVENTS,
    SessionState,
    ToolCallView,
    apply_event,
    context_fill,
    fold_session,
    initial_state,
    status_facts,
)
from agent6.viewmodel.tail import tail_events

_TASK_ICONS = TASK_STATUS_GLYPH

# How many recent tool calls the inline table shows. The RowSelected handler maps
# a visual row back through the same window, so both must use this one value.
_TOOL_TABLE_ROWS = 20

# Answer submitted after the worker died mid-modal: the file bridge has no
# reader, and the next resume re-asks the prompt itself.
_ANSWER_LOST = "the session is not live; the answer reached nothing (a resume re-asks the prompt)"

# Events after which dir_status is recomputed synchronously rather than on the
# ~1s heartbeat: session boundaries plus the operator-blocking prompt/answer
# pairs. The chip, worker_lost, and both composer bars all route off
# dir_status, so serving the previous state for even one heartbeat lies.
_STATUS_NOW_EVENTS = SESSION_START_EVENTS | {
    "session.end",
    "approval.prompt",
    "approval.answer",
    "question.prompt",
    "question.answer",
}

# Dashboard exit code meaning "quit the whole hub" (vs 0 == back to the hub).
QUIT_HUB_CODE = 99


class _ScrollPane(VerticalScroll):
    """A scrollable pane that can be tabbed to and maximized (View menu).
    VerticalScroll is
    focusable but disables maximize by default, so re-enable it; the content is a
    child Static the dashboard updates in place."""

    ALLOW_MAXIMIZE = True


class DashboardScreen(ScreenChrome, Screen[None]):
    """The run dashboard panes: task graph, live stream, tool table, log window,
    diff/verify, and the composer bar. Presentation only -- it renders the app's
    folded SessionState and dispatches run control back through the app (see the
    module docstring)."""

    CSS = """
    /* Top row: the task graph is usually a few nodes, so it stays compact beside
       the model's live output. */
    #top { height: auto; max-height: 7; padding: 0 1; }
    #head { height: 28%; }
    #plan { width: 32%; border: round $primary; }
    #stream { width: 1fr; border: round $primary; padding: 0 1; }
    /* The tool table spans the full width so all four columns stay visible. */
    #tools { height: 20%; border: round $primary; }
    /* Maximized, a pane fills the screen instead of holding its resting
       size -- textual tags the maximized widget with `-maximized`. The tool table
       drops its 20% height; the task graph drops its 32% width (else it stays a
       narrow column when maximized, like the tool table stayed short). */
    #tools.-maximized { height: 1fr; }
    #plan.-maximized { width: 1fr; }
    /* Log and diff share the tallest row; either maximizes full-screen. */
    #body { height: 1fr; }
    #log { width: 1fr; border: round $primary; }
    #diff { width: 1fr; border: round $primary; padding: 0 1; }
    /* The stream/diff bodies fill their scroll pane so long content scrolls;
       they are selectable text, so the pointer shows an I-beam over them. */
    #stream-body, #diff-body { width: 1fr; height: auto; pointer: text; }
    /* The composer bar (the same widget as the conversation's): auto-grows with
       its content, squeezing the 1fr #body row above. */
    #dash-input { height: auto; max-height: 8; border: round $primary; }
    #dash-input:focus { border: round $accent; }
    /* One card background everywhere. Tree/DataTable/RichLog default to $surface
       but the Static-based stream/diff panes are transparent (screen background),
       so set it explicitly to keep every card the same. */
    #plan, #stream, #tools, #log, #diff, #dash-input { background: $surface; }
    /* Uniform resting border (matches the home table + config card); the focused
       panel goes $accent. */
    #plan:focus, #stream:focus, #tools:focus, #log:focus, #diff:focus { border: round $accent; }
    """

    COMMANDS: ClassVar = Screen.COMMANDS | {MenuCommands}
    HELP_HINTS: ClassVar = (
        "Tab focuses a pane · PgUp/PgDn, Home/End scroll it",
        "Enter on a tool row opens its full detail",
        "Pickers: ↑↓ highlight · Space selects",
    )

    MENUS: ClassVar = (
        Menu(
            "File",
            (MenuItem("Back", "to_hub"), MenuItem("Quit", "quit_hub", "ctrl+q")),
        ),
        RUN_MENU,  # shared verbatim with the primary conversation view
        Menu(
            "View",
            (
                MenuItem("Next pane", "focus_next_pane", "tab"),
                MenuItem("Prev pane", "focus_prev_pane", "shift+tab"),
                MenuItem("Maximize pane", "fullscreen"),
                MenuItem("Full log…", "view_logs"),
                MenuItem("Conversation…", "toggle_dashboard"),
                MenuItem("Theme…", "choose_theme"),
                MenuItem("Copy method…", "choose_copy_method"),
            ),
        ),
        Menu(
            "Help",
            (
                MenuItem("Keys & actions", "help"),
                MenuItem("Command palette", "command_palette", "ctrl+p"),
            ),
        ),
    )
    # The composer bar is the default focus, so -- exactly like the conversation
    # view -- there are no plain-letter shortcuts: the same priority-bound set,
    # in the same footer order, on both screens. Run control lives in the Run
    # menu and the palette. `?` opens help when focus is not in the bar.
    BINDINGS: ClassVar = [
        Binding("ctrl+d", "toggle_dashboard", "Conversation", priority=True),
        Binding("ctrl+c", "copy", "Copy", priority=True),
        Binding("ctrl+r", "history_search", "History", priority=True),
        Binding("escape", "to_hub", "Back", key_display="Esc", priority=True),
        Binding("pageup", "page_up", "Scroll up", priority=True, show=False),
        Binding("pagedown", "page_down", "Scroll down", priority=True, show=False),
        Binding("ctrl+home", "scroll_top", "Top", priority=True, show=False),
        Binding("ctrl+end", "scroll_bottom", "End", priority=True, show=False),
        Binding("question_mark", "help", "Help", show=False),
        *menu_bindings(MENUS),
    ]

    def _sync_diff_nav(self, s: SessionState) -> None:
        """The step selector lists the run's commits (newest first) behind
        "latest commit"; hidden while nothing is committed, and under
        `[git].control = "model"` the pane says so (no chain to select from)."""
        nav = self.query_one("#diff-nav", Horizontal)
        if self._git_control() == "model":
            nav.display = False
            self.query_one("#diff").border_title = "diff · the model owns git"
            return
        if not s.steps:
            nav.display = False
            return
        nav.display = True
        if len(s.steps) != self._nav_steps:
            self._nav_steps = len(s.steps)
            options = [("latest commit", "")] + [
                (f"iter {st.iteration} · {st.sha[:7]} · {st.subject[:40]}", st.sha)
                for st in reversed(s.steps)
            ]
            select = self.query_one("#diff-step", Select)
            select.set_options(options)
            select.value = self._step_sel if any(v == self._step_sel for _, v in options) else ""

    def _git_control(self) -> str:
        with contextlib.suppress(ManifestError):
            return read_manifest(self._tui.session_dir).git_control
        return "agent6"

    def _step_patch(self, sha: str) -> str:
        if self._cumulative:
            with contextlib.suppress(ManifestError):
                base = read_manifest(self._tui.session_dir).base_sha
                if base:
                    return diff_range(Path.cwd(), base, sha) or "(no diff)"
        return commit_diff(Path.cwd(), sha) or "(no diff)"

    def on_select_changed(self, event: Select.Changed) -> None:
        if event.select.id == "diff-step":
            self._step_sel = str(event.value or "")
            self.render_state()

    def on_checkbox_changed(self, event: Checkbox.Changed) -> None:
        if event.checkbox.id == "diff-cumulative":
            self._cumulative = bool(event.value)
            self.render_state()

    def __init__(self, *, presets: list[str] | None = None) -> None:
        super().__init__()
        self._presets = presets if presets is not None else []
        # Select a task in the #plan tree to filter tools/log/diff to it; re-select
        # to clear. _log_filter tracks what the RichLog currently shows so a filter
        # change forces one full re-render (it is append-only otherwise).
        self._selected_task_id: str | None = None
        self._log_filter: str | None = None
        self._last_log_count = 0
        self._visible_tools: tuple[ToolCallView, ...] = ()  # the tool rows on screen now
        # What each pane last rendered (strong refs; the fold's replace() keeps
        # untouched fields identical, so `is` says "nothing to redo"). Rebuilding
        # the tree/table/diff on every structural event would be most of a
        # burst's cost.
        self._rendered_tree: tuple[object, object] | None = None
        self._rendered_tools: tuple[object, object] | None = None
        self._rendered_diff: tuple[object, ...] | None = None
        self._step_sel = ""  # a selected step's sha ("" = latest)
        self._cumulative = False
        self._nav_steps = -1  # how many steps the selector lists
        self._compare_line: str | None = None  # cached fan-out compare header (terminal state)
        self._branch_line: str | None = None  # cached branch header (fixed for the leg)
        self._lineage_line: str | None = None  # cached fork lineage (never changes)

    def _compare_top(self) -> str:
        """The fan-out compare outcome for the header's task line (empty for a
        non-lane run). Read from the manifest once it appears (a lane is stamped
        post-import, by which point it is finished) and cached: it never changes."""
        if self._compare_line is not None:
            return self._compare_line
        formatted = format_compare(session_compare(self._tui.session_dir))
        if formatted is None:
            return ""  # not stamped (yet); don't cache -- a live lane may get stamped later
        headline, rationale = formatted
        rat = f" — {rationale[:100]}" if rationale else ""
        self._compare_line = f"\ncompare: {headline}{rat}"
        return self._compare_line

    def _lineage_top(self) -> str:
        """Where a forked run came from, for the header (the web header's and
        `sessions show`'s line); read once, it never changes."""
        if self._lineage_line is None:
            lineage = manifest_header(self._tui.session_dir).get("forked_from", "")
            self._lineage_line = f"\nforked from: {lineage}" if lineage else ""
        return self._lineage_line

    def _branch_top(self) -> str:
        """Where the run's work lives, for the header: the run branch and the
        base a merge lands on, or the branch merged (the web header's line and
        `sessions show`'s `changes:`). Read from the manifest once it names a
        branch and cached: the merge stamp lands after the run ends, when this
        screen no longer repaints (a reopen re-reads)."""
        if self._branch_line is not None:
            return self._branch_line
        line = manifest_branches(self._tui.session_dir, repo=Path.cwd()).get("branch_line", "")
        if not line:
            return ""  # no manifest yet (a launching run); don't cache
        self._branch_line = f"\nbranch: {line}"
        return self._branch_line

    @staticmethod
    def _pins_top(s: SessionState) -> str:
        """The operator's pinned instructions in force, for the header (the web
        header's and `sessions show`'s line)."""
        return f"\npins: {' | '.join(s.pins)}" if s.pins else ""

    def _serving_top(self) -> str:
        """What the run is serving, for the header: the ports its network
        listens on and the `agent6 forward` line that reaches one (the web
        header's and `sessions show`'s line). A live probe: "" once the
        network is gone."""
        ports = listening_ports(self._tui.session_dir)
        if not ports:
            return ""
        listed = ", ".join(str(p) for p in ports)
        return f"\nserving: {listed} · agent6 forward {self._tui.session_dir.name} {ports[0]}"

    @property
    def _tui(self) -> Agent6TUI:
        app = self.app
        assert isinstance(app, Agent6TUI)
        return app

    # --- layout -------------------------------------------------------

    def compose(self) -> ComposeResult:
        yield MenuBar(self.MENUS)  # the top row: menus + "agent6 — <run>"
        yield Static("", id="top")
        with Horizontal(id="head"):
            yield Tree("tasks", id="plan")
            with _ScrollPane(id="stream"):
                yield Static("", id="stream-body")
        # cursor_type="row": the whole row highlights and Enter opens its full
        # detail (the columns truncate long args/summaries; see RowSelected).
        yield DataTable(id="tools", cursor_type="row")
        with Horizontal(id="body"):
            # markup=False: log lines contain raw tool args like `args=[a,b]` which
            # Rich would otherwise try to parse as markup and crash. auto_scroll off:
            # _render does sticky-bottom itself (snap to the newest line only when the
            # operator is already at the bottom).
            # max_lines == the state log window: a burst that outruns the window
            # between coalesced paints evicts the pre-burst lines, so the inline
            # pane stays a gapless recent window (Full log is the history).
            yield RichLog(
                id="log",
                highlight=False,
                markup=False,
                wrap=False,
                auto_scroll=False,
                max_lines=MAX_LOG_TAIL,
            )
            with _ScrollPane(id="diff"):
                with Horizontal(id="diff-nav"):
                    yield Select(
                        [("latest commit", "")], value="", id="diff-step", allow_blank=False
                    )
                    yield Checkbox("cumulative", id="diff-cumulative")
                yield Static("", id="diff-body")
        yield SteerSuggest(id="dash-suggest")  # command hints while typing `/…`
        yield ResumePreset(self._presets, id="dash-preset")  # shown while the composer resumes
        yield SteerInput(id="dash-input")
        yield Footer()

    def on_mount(self) -> None:
        self.query_one("#tools", DataTable).add_columns("tool", "args", "ok", "summary")
        self.render_state()  # initial paint; later paints are coalesced in the app's tick
        # Like the conversation: open ready to type (Tab moves out to the panes).
        self.query_one("#dash-input", SteerInput).focus()

    # --- actions ------------------------------------------------------

    def on_steer_input_submitted(self, message: SteerInput.Submitted) -> None:
        self._tui.submit_instruction(message.text)

    def action_history_search(self) -> None:
        open_history_search(self, self.query_one("#dash-input", SteerInput), self._tui.logs_path)

    def on_text_area_changed(self, event: TextArea.Changed) -> None:
        if event.text_area.id != "dash-input":
            return
        with contextlib.suppress(NoMatches):
            self.query_one("#dash-suggest", SteerSuggest).show_for(
                event.text_area.text,
                mode="steer" if self._tui.session_controllable() else "resume",
            )

    def action_toggle_dashboard(self) -> None:
        self._tui.action_toggle_dashboard()

    def action_to_hub(self) -> None:
        self._tui.action_to_hub()

    def action_quit_hub(self) -> None:
        self._tui.action_quit_hub()

    def action_copy(self) -> None:
        """Copy the mouse selection via the copy_method preference (the same
        Ctrl+C the conversation has; textual's built-in copy would emit a bare
        OSC 52, which multiplexers like tmux swallow)."""
        text = self.get_selected_text()
        if not text or not text.strip():
            self.notify("nothing selected")
            return
        driver = self.app._driver  # pyright: ignore[reportPrivateUsage]

        def emit(seq: str) -> None:
            if driver is not None:
                driver.write(seq)

        try:
            status = clipboard.emit_clipboard(
                text, clipboard.resolve_method(get_copy_method()), emit
            )
        except (OSError, subprocess.CalledProcessError) as exc:
            self.notify(f"copy failed: {exc}", severity="error")
            return
        self.notify(f"copied selection ({status})")

    def _scroll_target(self) -> Widget:
        """The pane the shared scroll keys drive: the focused scrollable if any
        (Tab reaches every pane), else the log -- the dashboard's main scrollback."""
        focused = self.focused
        if isinstance(focused, (ScrollView, ScrollableContainer)):
            return focused
        return self.query_one("#log", RichLog)

    def action_page_up(self) -> None:
        self._scroll_target().scroll_page_up(animate=False)  # instant, like the viewers

    def action_page_down(self) -> None:
        self._scroll_target().scroll_page_down(animate=False)

    def action_scroll_top(self) -> None:
        self._scroll_target().scroll_home(animate=False)

    def action_scroll_bottom(self) -> None:
        self._scroll_target().scroll_end(animate=False)

    def action_focus_next_pane(self) -> None:
        # Local action wrapping the App's framework action so it resolves from a
        # menu item / palette entry (a namespaced `app.focus_next` does not).
        self.app.action_focus_next()

    def action_focus_prev_pane(self) -> None:
        self.app.action_focus_previous()

    def action_fullscreen(self) -> None:
        """Maximize the focused pane; Esc (or the action again) restores it."""
        if self.maximized is not None:
            self.minimize()
        elif self.focused is not None and self.focused.allow_maximize:
            self.maximize(self.focused)

    def action_view_logs(self) -> None:
        """Open the full, scrollable log of THIS run -- the inline #log pane is a
        small sliding window; this is the whole history, scroll-anchored. (l again
        inside the view closes it: LogScreen binds l -> close.)"""
        self.app.push_screen(
            LogScreen(self._tui.logs_path, title=lambda: self._tui.screen_title("logs"))
        )

    def on_screen_resume(self) -> None:
        # The conversation stamps its own sub_title; re-stamp ours when the
        # toggle (or a closing viewer) brings the dashboard back on top, and
        # repaint the light parts (the composer's mode and preset picker can
        # have moved while the conversation was on top).
        self.app.sub_title = self._tui.run_title()
        with contextlib.suppress(NoMatches):
            self.render_heartbeat()

    # --- command palette ---------------------------------------------

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        """Enter on a tool-calls row opens its full args + summary in a modal (the
        columns truncate long values). Map the visual row back through the same
        window the table was built from; ignore an out-of-range index from a race
        with a rebuild."""
        if event.data_table.id != "tools":
            return
        window = self._visible_tools  # exactly the rows on screen (task filter applied)
        if 0 <= event.cursor_row < len(window):
            tc = window[event.cursor_row]
            self.app.push_screen(
                ToolCallDetailModal(tc.name, tc.ok, tc.args_full, tc.result_summary)
            )

    def on_tree_node_selected(self, event: Tree.NodeSelected[str | None]) -> None:
        """Task-filter click handler for the #plan tree (see __init__'s notes)."""
        if event.control.id != "plan":
            return
        tid = event.node.data
        if not isinstance(tid, str):
            return
        self._selected_task_id = None if tid == self._selected_task_id else tid
        self.render_state()  # a selection, not an event: re-render with the new filter now

    # --- rendering ---------------------------------------------------

    def _end_label(self) -> str:
        """The top-line status label, from THE dir decision (status_for_session_dir,
        the same word the hub row shows), in the word's shared colour; empty
        while running (the heartbeat line carries live activity)."""
        word, reason = self._tui.dir_status
        if word == "running":
            return ""
        return f"[b {status_style(word)}]{escape(status_label(word, reason))}[/]"

    def render_heartbeat(self) -> None:
        """The CHEAP once-a-second repaint: the top status line, the composer
        bar's labels, and the live stream pane. The full pane rebuild
        (render_state) runs only when events actually arrive: rebuilding the
        task tree and tool table every heartbeat would be pure idle churn."""
        tui = self._tui
        s = tui.state
        # Relabel every paint: mode flips on finished, and the context readout
        # in the subtitle moves with the run.
        mode: ComposerMode = "steer" if tui.session_controllable() else "resume"
        self.query_one("#dash-input", SteerInput).set_mode(
            mode=mode, ctx_pct=tui.context_pct(), continue_as=tui.continue_as
        )
        self.query_one("#dash-preset", ResumePreset).show(mode == "resume")
        role = s.last_role
        # Live heartbeat: a spinner + seconds since the last event, shown while
        # the run is active. Silent thinking / the resume gap now visibly tick.
        # NOT while "waiting": a run blocked on an operator prompt is controllable
        # (steerable) but not working, so the ticking beat would contradict the
        # same line's "waiting · needs answer" -- the rule the stream body honors.
        active = tui.session_controllable() and tui.dir_status[0] != "waiting"
        beat = ""
        if active and role is not None:
            spinner = spinner_frame(tui.spin)
            beat = f" {spinner} {tui.seconds_since_event()}s"
        role_line = f"{role.role} / {role.model}{beat}" if role else "(idle)"
        done_n = sum(1 for t in s.tasks if t.status in ("passed", "skipped"))
        step = f"tasks: {done_n}/{len(s.tasks)}" if s.tasks else "tasks: —"
        finished = self._end_label()
        cost = f"[b]{format_cost(s.budget.usd_total, partial=s.budget.usd_partial)}[/]"
        # Consumption of the binding ledger: THIS leg's metered spend vs its
        # usd_cap (resume re-arms the cap while usd_total stays cumulative),
        # plus the unmetered-token fraction when that ledger has traffic.
        budget = ""
        if s.budget.usd_cap > 0:
            leg_usd = s.budget.usd_total - s.budget.usd_prior_legs
            budget = f"   budget: {min(leg_usd / s.budget.usd_cap, 1.0):.0%}"
        if s.budget.tokens_unmetered and s.budget.tokens_fallback_cap > 0:
            unmet = min(s.budget.tokens_unmetered / s.budget.tokens_fallback_cap, 1.0)
            budget += f"   unmetered: {unmet:.0%}"
        if s.budget.plan_used_percent > 0:
            budget += f"   plan: {s.budget.plan_used_percent:g}%"
            if s.budget.plan_cap > 0:
                budget += f" (run {s.budget.plan_consumed:g}/{s.budget.plan_cap:g}pt)"
        pct = tui.context_pct()
        ctx = f"   ctx: {pct}%" if pct is not None else ""
        self.query_one("#top", Static).update(
            f"[b]agent6[/]  {step}   role: {escape(role_line)}   cost: {cost}{budget}{ctx}"
            f"   {finished}\n"
            f"task: {escape(task_snippet(s.user_task or tui.fallback_task, max_chars=120))}"
            f"{escape(self._lineage_top())}{escape(self._branch_top())}"
            f"{escape(self._pins_top(s))}{escape(self._serving_top())}"
            f"{escape(self._compare_top())}"
        )

        # Live reasoning / response pane. Built as rich Text so model output is
        # never parsed as markup.
        self.query_one("#stream-body", Static).update(self._stream_story(s, active=active))

    def _stream_story(self, s: SessionState, *, active: bool) -> Text:
        """What the stream pane says: the end story for a finished run, live
        deltas or the working heartbeat while active, and a truthful line for
        every dead state (stale/parked/created) -- never "(waiting for the
        model…)" over a run no model will ever touch."""
        tui = self._tui
        role = s.last_role
        st = Text()
        streaming = (
            role is not None and role.in_flight and (role.streamed_thinking or role.streamed_text)
        )
        if s.finished:
            # The end story, not a stale "idle": how it ended + the closing
            # summary, and a plan's deliverable (the CLI prints plan.md at the
            # end; the web shows it in its plan.md card).
            word, reason = tui.dir_status
            st.append(status_label(word, reason) + "\n", style=f"bold {status_style(word)}")
            if s.finish_summary:
                st.append(s.finish_summary, style="dim")
            if plan := tui.plan_md():
                st.append("\n\n" + plan)
        elif streaming:
            assert role is not None
            if role.streamed_thinking:
                st.append("💭 ", style="bold")
                st.append(role.streamed_thinking[-1200:] + "\n", style="dim")
            if role.streamed_text:
                st.append(role.streamed_text[-1200:])
        elif tui.dir_status[0] == "waiting":
            st.append("waiting for your answer (see the prompt)", style="bold yellow")
        elif active and role is not None:
            # No live deltas: the model is thinking, or a resume is rebuilding
            # context. A ticking heartbeat, never a stale "idle" or blank.
            spinner = spinner_frame(tui.spin)
            secs = tui.seconds_since_event()
            st.append(f"{spinner} {role.role} working… {secs}s", style="dim italic")
        elif tui.dir_status[0] == "stale":
            # The composer below has focus and Enter resumes; there is no
            # plain-letter shortcut to point at (pressing r would just type r).
            st.append(
                "worker exited without finishing (crashed or killed) — type a"
                " follow-up below (Enter resumes)",
                style="bold red",
            )
        elif tui.dir_status[0] == "parked":
            # No model is coming: the run was saved at submission and never
            # started (the cause rides in the status detail). Resume is the
            # one action.
            cause = f" ({tui.dir_status[1]})" if tui.dir_status[1] else ""
            st.append(f"parked at submission{cause}\n", style="bold yellow")
            st.append("type the go-ahead below (Enter resumes)", style="dim")
        elif tui.dir_status[0] == "created":
            st.append("created — the run has not started\n", style="bold")
            st.append("type a follow-up below (Enter resumes)", style="dim")
        else:
            st.append("(waiting for the model…)", style="dim")
        return st

    def render_state(self) -> None:  # noqa: PLR0912, PLR0915
        self.render_heartbeat()
        tui = self._tui
        s = tui.state

        # A task selected in the #plan tree filters tools/log/diff to it. sel=None
        # is the unfiltered live view; the border titles show which task when set.
        sel = self._selected_task_id
        sel_title = next((t.title for t in s.tasks if t.id == sel), "") if sel else ""
        # A border title is markup; the task title is the model's or the user's.
        filt = f" · task: {escape(sel_title[:28])}" if sel else ""

        # Task DAG: the worker's live add_task/update_task breakdown (graph.update
        # snapshots), indented by depth, cursor marked. Rebuilt only when the
        # tasks tuple (or the selection highlight) actually changed.
        if self._rendered_tree is None or not (
            self._rendered_tree[0] is s.tasks and self._rendered_tree[1] == sel
        ):
            self._rendered_tree = (s.tasks, sel)
            tree = self.query_one("#plan", Tree)
            tree.clear()
            for tv in s.tasks:
                icon = _TASK_ICONS.get(tv.status, "·")
                indent = "  " * tv.depth
                marker = "▸ " if tv.is_cursor else ""
                label = Text(f"{indent}{marker}{icon} {tv.title}")
                if tv.id == sel:  # the task the panes are filtered to
                    label.stylize("bold reverse")
                tree.root.add_leaf(label, data=tv.id)
            tree.root.expand()

        table = self.query_one("#tools", DataTable)
        if self._rendered_tools is None or not (
            self._rendered_tools[0] is s.tool_calls and self._rendered_tools[1] == sel
        ):
            self._rendered_tools = (s.tool_calls, sel)
            table.clear()
            tools = [tc for tc in s.tool_calls if sel is None or tc.task_id == sel]
            self._visible_tools = tuple(tools[-_TOOL_TABLE_ROWS:])
            for tc in self._visible_tools:
                ok = "…" if tc.ok is None else ("✓" if tc.ok else "✗")
                table.add_row(
                    Text(tc.name), Text(tc.args_preview[:90]), ok, Text(tc.result_summary[:40])
                )
            table.border_title = f"tools{filt}" if sel else ""

        # Log. Diff on the monotonic log_count, not len(log_tail): log_tail is a
        # sliding window, so a length-based diff freezes once it saturates.
        # Sticky-bottom: only snap to the newest line if the operator was already
        # at the bottom, so scrolling up to read holds position. End (pane focused) / Full
        # log jump back to the live tail. A filter change forces one full
        # re-render (the RichLog is append-only, so it cannot re-window itself
        # incrementally).
        log = self.query_one("#log", RichLog)
        log.border_title = f"log{filt}" if sel else ""
        if sel != self._log_filter:
            log.clear()
            for ln in s.log_tail:
                if sel is None or ln.task_id == sel:
                    log.write(ln.text)
            log.scroll_end(animate=False)
            self._log_filter = sel
            self._last_log_count = s.log_count
        else:
            n_new = min(s.log_count - self._last_log_count, len(s.log_tail))
            if n_new > 0:
                at_bottom = (log.max_scroll_y - log.scroll_offset.y) <= 1
                for ln in s.log_tail[-n_new:]:
                    if sel is None or ln.task_id == sel:
                        log.write(ln.text)
                if at_bottom:
                    log.scroll_end(animate=False)
            self._last_log_count = s.log_count

        # Diff: the latest auto-commit or live verify output -- or, when a task is
        # selected, the commits made while it was in focus. Built as rich Text to
        # avoid markup parsing of diff/verify bodies (which contain brackets).
        # Skipped whenever none of its inputs changed.
        self._sync_diff_nav(s)
        diff_key = (
            sel,
            s.recent_diffs,
            s.last_verify,
            s.latest_diff,
            self._step_sel,
            self._cumulative,
        )
        if self._rendered_diff is not None and all(
            a is b for a, b in zip(self._rendered_diff, diff_key, strict=True)
        ):
            return
        self._rendered_diff = diff_key
        diff_widget = self.query_one("#diff-body", Static)
        self.query_one("#diff").border_title = f"diff{filt}" if sel else ""
        verify = s.last_verify
        dt = Text()
        if self._step_sel:
            step = next((st for st in s.steps if st.sha == self._step_sel), None)
            if step is not None:
                what = "cumulative to" if self._cumulative else "step"
                dt.append(
                    f"{what} iter {step.iteration} · {step.sha[:7]} · {step.subject}\n",
                    style="bold",
                )
                _append_colored_diff(dt, self._step_patch(step.sha)[:4000])
                diff_widget.update(dt)
                return
        if sel is not None:
            task_diffs = [d for d in s.recent_diffs if d.task_id == sel]
            if task_diffs:
                n = len(task_diffs)
                dt.append(f"selected task · {n} commit{'s' if n != 1 else ''}\n", style="bold")
                _append_colored_diff(dt, task_diffs[-1].patch[:2000])
            else:
                dt.append("(no commits during the selected task yet)", style="dim")
            diff_widget.update(dt)
        # A RUNNING or FAILED verify takes precedence so a failure is never
        # hidden behind a stale passing diff. A passed verify yields to the diff.
        elif verify is not None and verify.exit_code is None:
            dt.append("verify running: ", style="bold")
            dt.append(" ".join(verify.cmd)[:200] + "\n")
            dt.append("…", style="dim")
            diff_widget.update(dt)
        elif verify is not None and verify.exit_code != 0:
            dt.append(f"verify exit={verify.exit_code} ", style="bold red")
            dt.append(f"({verify.duration_s:.1f}s)  {' '.join(verify.cmd)[:160]}\n")
            dt.append((verify.stderr_tail or verify.stdout_tail)[:2000] or "(no output)")
            diff_widget.update(dt)
        elif s.latest_diff:
            dt.append("latest commit diff\n", style="bold")
            _append_colored_diff(dt, s.latest_diff[:2000])
            diff_widget.update(dt)
        elif verify is not None:
            dt.append(f"verify passed ({verify.duration_s:.1f}s)", style="bold green")
            diff_widget.update(dt)
        else:
            diff_widget.update(Text("(no diffs yet)", style="dim"))


class Agent6TUI(PlainNotify, MuxPointerShapes, App[int]):
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
    """
    )

    BINDINGS: ClassVar = [
        # App-level so it works from any screen (viewers included); the hub-aware
        # exit code needs our handler, not textual's default quit.
        Binding("ctrl+q", "quit_hub", "Quit", show=False),
        # Ctrl-Z means "step away": the run keeps going and the shell comes
        # back (raw mode keeps the key, and a real SIGTSTP would freeze a live
        # provider stream mid-response). priority so the composer TextArea's
        # built-in ctrl+z undo never shadows it; Ctrl-_ is undo there, and the
        # detach hint says so.
        Binding("ctrl+z", "detach_exit", "Detach", show=True, priority=True),
    ]

    def __init__(
        self,
        session_dir: Path,
        *,
        exit_on_end: bool = False,
        from_hub: bool = False,
        config_path: Path | None = None,
    ) -> None:
        super().__init__()
        self.session_dir = session_dir
        # The invocation's `--config F`; a detach-resume it spawns re-applies F.
        self.config_path = config_path
        # When launched from the hub loop, Esc returns to it and q quits the hub
        # (signalled by the exit code); standalone, both just close the dashboard.
        self.from_hub = from_hub
        self.logs_path = session_dir / LOGS_NAME
        self.state: SessionState = initial_state()
        self._prompts = PromptDispatcher(
            self,
            answerable=self.session_controllable,
            lost=_ANSWER_LOST,
            inline_approvals=lambda: isinstance(self.screen, ConversationScreen),
        )
        self._seen_steer = 0
        self._dirty = False  # a structural event arrived; _tick coalesces the repaint
        self._light_dirty = False  # only stream deltas / heartbeat: light repaint
        self._stop = threading.Event()
        # When True (the auto-spawned co-process of `agent6 run`), the view
        # ends WITH the run: once it finishes, the dashboard holds on the
        # payoff until the user leaves (Ctrl+Q), and only then does the parent
        # command return. `agent6 attach --tui` leaves this False and keeps
        # following.
        self.exit_on_end = exit_on_end
        # The run ended under exit_on_end and the dashboard is holding.
        self._end_hold = False
        # The fork this view created (/undo on a finished run, Run > Fork; the
        # fold has no event for either); continue_as routes the follow-up there.
        self._continue_child = ""
        # Set by action_detach_exit; run_tui reads it to print the reattach hint.
        self.detached = False
        # THE (word, reason) for this run -- status_for_session_dir, the same
        # decision the hub row shows -- refreshed on the ~1/s heartbeat.
        # Derived, never latched: a crash->resume flips it back to running
        # (a one-way latch would keep "worker exited" painted over the live
        # resumed leg and drop operator steers).
        self.dir_status: tuple[str, str] = status_for_session_dir(
            session_dir, status_facts(self.state)
        )
        # Lines the log held at open; worker_lost waits for the fold to reach it.
        self._seed_log_count = 0
        # Header task line for a run with no session.start yet (parked/created):
        # the fold has no user_task, but the manifest knows the work.
        self.fallback_task = ""
        # The session's mode (run / plan / ask): the dashboard's title word, as
        # the web panel heading states it; "run" for a manifest-less dir.
        self.mode = "run"
        with contextlib.suppress(ManifestError):
            manifest = read_manifest(session_dir)
            self.fallback_task = manifest.user_task
            self.mode = manifest.mode or "run"
        # Live heartbeat: a run can be silent for a whole reasoning turn (or the
        # resume context-rebuild gap). Track when the last event landed and
        # repaint ~1/s while active so an elapsed timer + spinner visibly tick --
        # the difference between "thinking" and "hung" the user could not see.
        self.last_event_at = time.monotonic()
        self._heartbeat_at = 0.0
        self.spin = 0
        # The preset a resume from a composer continues under ("" = as
        # recorded); both run views' pickers read and write it.
        self.resume_preset = ""
        presets = available_preset_names(Path.cwd(), config_path)
        self._dash = DashboardScreen(presets=presets)
        self._conv = ConversationScreen(
            self.logs_path,
            title=self.screen_title,
            primary=True,
            presets=presets,
            prompts=self._prompts,
        )

    def _task_lead(self) -> str:
        """What names this run in a title: the TASK (clipped), the pet name
        only when no task is known yet -- the web hub's rows lead the same
        way, and the id stays in the header line and every resume hint."""
        task = self.state.user_task or self.fallback_task
        return task_snippet(task, max_chars=57) or self.session_dir.name

    def screen_title(self, context: str) -> str:
        """Menu-bar subtitle for a run screen: the view's context word, the live
        task name, and -- once the run ended -- the status plus how to leave.
        Screens stamp via a provider calling this at stamp time, never a string
        frozen at construction: the task name lands after the first fold, and
        the end hold must survive whichever stamp runs last."""
        if self._end_hold:
            return f"{context} · {self._task_lead()} · {self.dir_status[0]} — ctrl+q to leave"
        return f"{context} · {self._task_lead()}"

    def run_title(self) -> str:
        return self.screen_title(self.mode)

    def on_mount(self) -> None:
        setup_theme(self)  # apply the saved theme before the first paint
        # Per-process claim file: nothing to defend or re-assert, concurrent
        # web/TUI/attach viewers each hold their own.
        register_frontend(self.session_dir, os.getpid())
        self.sub_title = self.run_title()  # menu-bar title context
        self._seed_from_disk()
        # Pushed (not the app's default screen): only the push path loads a
        # screen's CSS, and the hub pushes its HomeScreen the same way. The
        # conversation opens on top -- the primary view -- with the dashboard
        # beneath it; Ctrl+D toggles between them.
        # Installed, so popping the conversation hides rather than destroys it.
        self.push_screen(self._dash)
        self.install_screen(self._conv, "conversation")
        self.push_screen(self._conv)
        # Auto-spawn close: the exit condition (run over, prompts answered) is
        # polled from a timer in the app's OWN loop and exits there. Exit()
        # scheduled from inside a call_from_thread callback does not take effect,
        # but exiting from a timer callback does. The same timer also drives the
        # approval / question modals and the steer composer focus.
        self.set_interval(0.2, self._tick)
        self._thread = threading.Thread(target=self._reader_loop, daemon=True)
        self._thread.start()

    def _seed_from_disk(self) -> None:
        """Fold the log already on disk, before the reader thread starts.

        It seeds the status this viewer opens on, the line the fold must reach
        for an absent `session.end` to mean anything (`worker_lost`), and the
        steer baseline, so a request already in the log does not prompt.
        """
        with contextlib.suppress(OSError):
            seeded = fold_session(tail_events(self.logs_path, follow=False))
            self._seen_steer = seeded.steer_requests
            self._seed_log_count = seeded.log_count
            self.dir_status = status_for_session_dir(self.session_dir, status_facts(seeded))

    def on_unmount(self) -> None:
        self._stop.set()
        # Drop only our own front-end claim; concurrent viewers keep theirs.
        unregister_frontend(self.session_dir, os.getpid())

    # --- reader thread -----------------------------------------------

    def _reader_loop(self) -> None:
        for event in tail_events(
            self.logs_path,
            follow=True,
            stop_when_finished=self.exit_on_end,
            # Without this, closing the dashboard on a run that never ends
            # (finished + exit_on_end=False, or crashed) leaks this thread in
            # the idle poll forever -- one per run the hub session opens.
            should_stop=self._stop.is_set,
        ):
            if self._stop.is_set():
                return
            self.call_from_thread(self._handle_event, event)

    def seconds_since_event(self) -> int:
        """Idle seconds for the "working… Ns" heartbeat, anchored to the last
        EVENT's ts (the fold carries it), not to when this viewer folded it:
        an arrival anchor would read "working… 3s" on attach to a run wedged
        40 minutes. Falls back to the fold time for
        a log whose events carry no ts."""
        if self.state.last_event_ep is None:
            return int(time.monotonic() - self.last_event_at)
        return max(0, int(time.time() - self.state.last_event_ep))

    def _handle_event(self, event: dict[str, object]) -> None:
        self.state = apply_event(self.state, event)
        self.last_event_at = time.monotonic()  # the ts-less fallback anchor
        if event.get("type") == "session.undone" and self.state.undone_to:
            # /undo forked the run at the state before the operator's last
            # message: that fork is the continuation, and the message taken
            # back is the operator's to edit and resend (the web does the same).
            self._fill_composers(self.state.undone_text)
            self.notify(f"undone: continue as {self.state.undone_to}; your message is back to edit")
        if event.get("type") in SESSION_START_EVENTS:
            # A session boundary (fresh run OR a resumed leg -- a resume emits
            # only loop.resume.start, never a second session.start) restarts the
            # prompt id counters at approval-1/question-1; a stale seen-set
            # would swallow the new session's first prompts and the run would
            # block forever on a modal that never opens.
            self._prompts.reset()
            # The task is known once the boundary folds: retitle the menu bar
            # (the conversation screen stamps its own title while on top).
            if self.screen is self._dash:
                self.sub_title = self.run_title()
        if event.get("type") in _STATUS_NOW_EVENTS:
            # A terminal / leg-boundary / operator-blocking event changes the
            # status NOW: refresh synchronously so the chip, the label, and the
            # composer routing never serve the previous state for up to a
            # heartbeat.
            self._refresh_dir_status()
        # Coalesce: mark dirty and let the 0.2s _tick repaint once. Replaying a
        # finished run floods hundreds of events on open; rendering each one would
        # rebuild the whole dashboard per event (UI thrash). Streaming deltas
        # only move the live stream pane, so they take the LIGHT repaint (a
        # reasoning burst would otherwise force full rebuilds 5x/s).
        if event.get("type") in STREAM_DELTA_EVENTS:
            self._light_dirty = True
        else:
            self._dirty = True

    @property
    def worker_lost(self) -> bool:
        """The recorded worker is gone without a session.end (kill -9 / OOM),
        the hub's "stale". Derived from dir_status, so a resume that brings a
        live worker back clears it. False while the fold trails the log: an end
        it has not reached is not an absent one."""
        return self.dir_status[0] == "stale" and self.state.log_count >= self._seed_log_count

    def _refresh_dir_status(self) -> None:
        """Recompute dir_status (a pid probe + a manifest read pre-start; the
        same cost class as the spinner tick, so it rides the ~1/s heartbeat).
        A change repaints and relabels BOTH composer bars -- the covered
        screen's too, which otherwise kept a stale label until its next
        event-driven paint (the two bars visibly disagreed live)."""
        status = status_for_session_dir(self.session_dir, status_facts(self.state))
        if status != self.dir_status:
            self.dir_status = status
            self._dirty = True
            self._conv.refresh_liveness()

    def _tick(self) -> None:
        # Prompt modals only while the run can consume an answer: the fold
        # keeps an unanswered prompt across session.end and a worker death (it
        # clears only on the answer event or a leg boundary), so an open would
        # otherwise pop live-looking Allow/Deny over a dead run and write the
        # answer into a file nobody polls. Skipped ids are NOT marked seen, so
        # a prompt that outlives a stale probe still pops on the next tick.
        if self.session_controllable():
            self._prompts.dispatch(self.session_dir, self.state)
        # Route an external steer request to the composer bar, once per Ctrl-C
        # (steer_requests is monotonic).
        if self.state.steer_requests > self._seen_steer:
            self._seen_steer = self.state.steer_requests
            self._steer_request_to_bar()
        # Heartbeat: refresh the dir status ~1/s (always -- it is how a death,
        # a parked resume, or a revival is noticed with no event to trigger a
        # paint), and while the run is live advance the spinner so the
        # "working… Ns" timer visibly ticks (thinking, not hung).
        now = time.monotonic()
        if now - self._heartbeat_at >= 1.0:
            self._heartbeat_at = now
            self._refresh_dir_status()
            if self.session_controllable():
                self.spin += 1
                self._light_dirty = True
        # Coalesced repaint: once per tick, and only when the dashboard is the
        # active, mounted screen. A pushed viewer, a modal, or shutdown leaves the
        # dashboard covered or torn down, so querying its widgets raises; defer
        # the paint (dirty stays set) until it is back on top. Structural events
        # rebuild the panes; deltas/heartbeat repaint only the light parts.
        # Read screen_stack, not App.screen: the interval outlives the stack
        # during shutdown, and a tick landing after the last screen pops must be
        # a no-op, not a ScreenStackError crash.
        stack = self.screen_stack
        if not self._stop.is_set() and stack and stack[-1] is self._dash:
            if self._dirty:
                self._dirty = self._light_dirty = False
                with contextlib.suppress(NoMatches):
                    self._dash.render_state()
            elif self._light_dirty:
                self._light_dirty = False
                with contextlib.suppress(NoMatches):
                    self._dash.render_heartbeat()
        # Once the run ended (a clean session.end, or the worker died without
        # one) and no modal is open -- never yank an in-flight answer, but a
        # ghost prompt on a dead run (its answer read by nobody) must not pin
        # the hold off forever -- the dashboard HOLDS on the payoff (verify,
        # diff, cost) instead of tearing down under the user. Ctrl+Q leaves;
        # the composer still routes a typed follow-up to resume.
        if (
            self.exit_on_end
            and not self._end_hold
            and (self.state.finished or self.worker_lost)
            and not (stack and isinstance(stack[-1], ModalScreen))
        ):
            self._end_hold = True
            self.sub_title = self.run_title()
            self.notify(
                f"{self.dir_status[0]} — Ctrl+Q to leave, or type below to continue the session",
                timeout=8.0,
            )
            self._dirty = True

    # --- run control (dispatched from the composer bars, keys, and menus) --

    def submit_instruction(self, text: str) -> None:
        """A composer-bar line. Live: inject it at the run's next safe boundary
        (after the current step, never mid tool-call) -- the run keeps going.
        Finished: resume THIS run with the instruction as the follow-up."""
        if text.strip() == "/undo":
            self._undo_session()
            return
        if text.strip() == "/restate":
            # Local and free: rendered from the journal, nothing reaches the model.
            text = restate(list(tail_events(self.logs_path, follow=False)))
            self.push_screen(TextModal("since your last message", text))
            return
        if text.strip() == "/shells":
            self.push_screen(TextModal("background commands", shells_text(self.session_dir)))
            return
        if self.session_controllable():
            question = parse_btw(text)
            if question is not None:
                self.notify(open_btw(self.session_dir, question))
                return
            focus = parse_compact(text)
            if focus is not None:
                # `/compact [focus]` is an out-of-band request, not steer text;
                # /pin and /parallel stay steers the loop parses itself.
                if request_compact(self.session_dir, focus=focus):
                    self.notify("compaction requested; applies before the next model call")
                else:
                    self.notify("could not write the compaction request", severity="warning")
                return
            submit_steer(self.session_dir, text)
            self.notify("steering this session…")
        else:
            self.resume_with_instruction(text)

    def _undo_session(self) -> None:
        """`/undo`: fork this run at the state before its last operator message,
        unstarted; the hub lists the fork and the undone text is reported. A
        live run is refused: stop it first, then undo."""
        if self.session_controllable():
            # The loop forks at its next boundary and emits session.undone;
            # the fold's undone_to hands the follow-up to this app.
            submit_steer(self.session_dir, "/undo")
            self.notify("undo requested; applies at the next step")
            return
        said: list[str] = []
        result = undo_fork(
            None,
            self.session_dir.name,
            cwd=Path.cwd(),
            reporter=Reporter(out=said.append, err=said.append),
        )
        if result is None:
            self.notify(said[-1].strip() if said else "undo failed", severity="warning")
            return
        child, text = result
        self._continue_child = child
        self._fill_composers(text)
        self.notify(f"undone: continue as {child}; your message is back to edit")

    def plan_md(self) -> str:
        """A planning run's written deliverable (plan.md), "" for a run or a
        plan that has not written one."""
        if self.mode != "plan":
            return ""
        with contextlib.suppress(OSError):
            return (self.session_dir / "plan.md").read_text(encoding="utf-8")
        return ""

    @property
    def continue_as(self) -> str:
        """The session a typed follow-up resumes: the fork an undone run named
        (its continuation, from the fold), the fork this view created (/undo
        on a finished run, Run > Fork), else "" for this run itself."""
        return self.state.undone_to or self._continue_child

    def _fill_composers(self, text: str) -> None:
        """Put *text* in both composer bars (the covered view's too), ready to
        edit and resend."""
        for screen, bar_id in ((self._conv, "#conv-input"), (self._dash, "#dash-input")):
            with contextlib.suppress(NoMatches):
                screen.query_one(bar_id, SteerInput).load_text(text)

    def resume_with_instruction(self, text: str) -> None:
        """Resume this run (or, after /undo, the fork it named) with *text* as
        its first steering instruction (rides `agent6 resume --steer`, which
        seeds the steer files AFTER its stale-state clear; a pre-seed here
        would be wiped by that clear). The new session's steer poll injects
        the text at its first boundary."""
        target = self.continue_as or self.session_dir.name
        err = spawn_detached_resume(
            Path.cwd(),
            target,
            steer=text,
            preset=self.resume_preset,
            config_path=self.config_path,
        )
        under = f" under preset {self.resume_preset}" if self.resume_preset else ""
        self.notify(
            err or f"resuming {target}{under} with your instruction…",
            severity="error" if err else "information",
        )

    def _focus_composer(self) -> None:
        """Focus the visible composer bar; with a viewer or modal on top, the
        caller's notice alone points the operator at the bar."""
        stack = self.screen_stack
        if not stack:  # shutdown race: the tick fired after the last screen popped
            return
        if stack[-1] is self._conv:
            self._conv.focus_bar()
        elif stack[-1] is self._dash:
            with contextlib.suppress(NoMatches):
                self._dash.query_one("#dash-input", SteerInput).focus()

    def _steer_request_to_bar(self) -> None:
        """An external steer request (a CLI Ctrl-C on an attached run, `agent6
        steer`): route it to the visible composer bar instead of a popup --
        focus it and say why."""
        self._focus_composer()
        self.notify("steering requested: type an instruction and press Enter")

    def action_compact(self) -> None:
        """Ask the run to compact its context now: drop the compact.request
        marker (the same file-bridge pattern as steer); the loop honors it at
        its next safe boundary by forcing a summarise-and-restart."""
        if not self.session_controllable():
            self.notify("nothing to compact: the session is not live", severity="warning")
            return
        if request_compact(self.session_dir):
            self.notify("compaction requested; applies at the next safe boundary")
        else:
            self.notify("could not write the compaction request", severity="warning")

    def action_stop_now(self) -> None:
        """Stop the run immediately: confirm, then write the abort answer over
        the file bridge -- the stream watchdog interrupts the in-flight turn and
        the run ends (resumable)."""
        if not self.session_controllable():
            self.notify("nothing to stop: the session is not live", severity="warning")
            return

        def _confirmed(yes: bool | None) -> None:
            if yes:
                submit_steer(self.session_dir, "abort")

        self.push_screen(
            ConfirmModal(
                "Stop this session now?",
                "Interrupts the current step; the run ends at once and can be resumed "
                "later with `agent6 resume`.",
                confirm_label="Stop now",
            ),
            _confirmed,
        )

    def action_stop_step(self) -> None:
        """Stop AFTER the current step completes: drop the stop.request marker
        the loop honors at its next completed-iteration boundary, so the step's
        tool results and auto-commit land before the run ends (resumable)."""
        if not self.session_controllable():
            self.notify("nothing to stop: the session is not live", severity="warning")
            return

        def _confirmed(yes: bool | None) -> None:
            if yes:
                request_stop(self.session_dir)
                self.notify("stopping after this step…")

        self.push_screen(
            ConfirmModal(
                "Stop after this step?",
                "The current step finishes (its tool results and auto-commit land), "
                "then the run stops. Resume later with `agent6 resume`.",
                confirm_label="Stop",
            ),
            _confirmed,
        )

    def action_delete_session(self) -> None:
        """Delete this run's history and return to the hub. History only: the run
        branch and its commits are git's (`sessions prune` is the branch verb)."""
        if self.session_controllable():
            self.notify("stop the session first: it is still live", severity="warning")
            return

        def _confirmed(yes: bool | None) -> None:
            if yes:
                ok, msg = run_cli_capture(
                    [*agent6_argv(self.config_path), "sessions", "rm", "--", self.session_dir.name],
                    Path.cwd(),
                )
                self.notify(
                    msg or ("removed" if ok else "could not remove"),
                    severity="information" if ok else "error",
                )
                if ok:
                    self.action_to_hub()

        self.push_screen(
            ConfirmModal(
                "Delete this session's history?",
                "Removes its transcripts, events and manifest from the state dir. "
                "The run branch and its commits are kept.",
                confirm_label="Delete",
            ),
            _confirmed,
        )

    def action_resume(self) -> None:
        """Resume a finished/stopped run: it continues in the background (appending
        to the same log) and this dashboard follows straight through. A run the
        agent ENDED has nothing to continue: the refusal `agent6 resume` would
        print lands here instead of on a detached child's stderr (the composer
        below gives it new work)."""
        if self.session_controllable():
            self.notify("nothing to resume: the session is still going", severity="warning")
            return
        if finished_needs_new_work(self.session_dir):
            self.notify(
                "this run finished; type what to do next below (Enter resumes it with the"
                " instruction)",
                severity="warning",
            )
            self._focus_composer()
            return
        err = spawn_detached_resume(Path.cwd(), self.session_dir.name, config_path=self.config_path)
        self.notify(
            err or f"resuming {self.session_dir.name} in the background…",
            severity="error" if err else "information",
        )

    def action_run_plan(self) -> None:
        """Execute a finished plan: spawn `agent6 run --from-plan <id>` detached
        (the hub lists and opens it). The plan session is untouched, so the
        composer keeps revising it."""
        if self.mode != "plan":
            self.notify("this session is not a plan", severity="warning")
            return
        if not (self.session_dir / "plan.md").is_file():
            self.notify("no plan.md yet (still planning, or never finished)", severity="warning")
            return
        runs = bucket_dir(layout_of(self.session_dir).state_dir, "runs")
        runs.mkdir(parents=True, exist_ok=True)
        new_dir, err = spawn_and_locate(
            [*agent6_argv(self.config_path), "run", "--from-plan", self.session_dir.name],
            Path.cwd(),
            before={p for p in runs.iterdir() if p.is_dir()},
            list_dirs=lambda: [p for p in runs.iterdir() if p.is_dir()],
            env={**os.environ, **DETACHED_RUN_ENV},
        )
        if new_dir is None:
            self.notify(err or "could not start the run", severity="error")
            return
        self.notify(f"run started: {new_dir.name} (open it from the hub)")

    def action_fork(self) -> None:
        """Fork this run at its latest checkpoint into a NEW run, unstarted. On
        a finished run the composer is handed to the fork: the next typed line
        is its instruction (Enter resumes the fork with it), the way /undo
        hands over its fork. On a live run the composer keeps steering THIS
        run, so the notice says how the fork starts. A fork that simply
        continued had no direction: of a finished run it re-read a done
        conversation and ended as a silent finish."""
        said: list[str] = []
        child, rc = create_fork(
            self.config_path,
            self.session_dir.name,
            cwd=Path.cwd(),
            reporter=Reporter(out=said.append, err=said.append),
        )
        if rc != 0:
            self.notify(said[-1].strip() if said else "fork failed", severity="error")
            return
        if self.session_controllable():
            self.notify(f"forked to {child} (unstarted); start it: agent6 resume {child} --steer …")
            return
        self._continue_child = child
        self.notify(f"forked to {child}; type what it should do below (Enter resumes it)")
        self._conv.refresh_liveness()
        with contextlib.suppress(NoMatches):
            self._dash.render_heartbeat()
        self._focus_composer()

    def context_pct(self) -> int | None:
        """Context-window fill (percent) at the last completed model call (the
        viewmodel's rule, shared with the web header and the pause menu)."""
        return context_fill(self.state)

    def session_controllable(self) -> bool:
        """True while the run can receive operator input over the file bridge:
        the dir status is a live word. Parked/created (never started), stale
        (worker gone), and every end word route the composer to resume -- the
        one action that will actually be read."""
        return self.dir_status[0] in LIVE_STATUS_WORDS

    def action_toggle_dashboard(self) -> None:
        """Flip between the conversation (the primary view) and the dashboard
        (Ctrl+D anywhere, t from the dashboard). The conversation is installed,
        so popping it hides it -- both views keep their state. No-op while a
        modal or a pushed viewer is on top."""
        if self.screen is self._conv:
            self.pop_screen()
        elif self.screen is self._dash:
            self.push_screen(self._conv)

    def action_to_hub(self) -> None:
        self.exit(0)  # back to the hub loop (or just close, standalone)

    def action_quit_hub(self) -> None:
        # In the hub loop, signal "quit the hub" via the exit code; standalone,
        # there's nothing to return to, so a plain close (0) is the same thing.
        self.exit(QUIT_HUB_CODE if self.from_hub else 0)

    def action_detach_exit(self) -> None:
        # A viewer (`attach --tui`, the hub) leaves a run that is detached
        # already. The view `agent6 run --tui` spawned (exit_on_end) fronts a
        # run in the terminal's own process, so it steers that run to detach:
        # at its next step boundary the lifecycle hands it to a background
        # resume and exits, as the CLI pause menu's /detach does, and the
        # shell comes back. run_tui prints the hint once the terminal is
        # restored.
        if self.exit_on_end:
            submit_steer(self.session_dir, "detach")
        self.detached = True
        self.exit(QUIT_HUB_CODE if self.from_hub else 0)

    def get_system_commands(self, screen: Screen[object]) -> Iterable[SystemCommand]:
        # Drop textual's "Keys" panel (our Help page replaces it), "Screenshot" (an
        # unused default whose SVG export is broken in our terminals), "Theme"
        # (replaced by our live-preview Theme… picker), and "Quit" (its plain exit()
        # returns the wrong code here -- our File menu's Back to hub / Quit do). All
        # of these are provided by MENUS via palette_commands, so nothing's added.
        for cmd in super().get_system_commands(screen):
            if cmd.title not in ("Keys", "Screenshot", "Theme", "Quit"):
                yield cmd


def _append_colored_diff(dt: Text, patch: str) -> None:
    """Append a unified diff with +/- line coloring (no markup parsing)."""
    for line in patch.splitlines():
        if line.startswith("+") and not line.startswith("+++"):
            dt.append(line + "\n", style="green")
        elif line.startswith("-") and not line.startswith("---"):
            dt.append(line + "\n", style="red")
        elif line.startswith("@@"):
            dt.append(line + "\n", style="cyan")
        else:
            dt.append(line + "\n")


def run_tui(
    session_dir: Path,
    *,
    exit_on_end: bool = False,
    from_hub: bool = False,
    config_path: Path | None = None,
) -> int:
    app = Agent6TUI(
        session_dir, exit_on_end=exit_on_end, from_hub=from_hub, config_path=config_path
    )
    rc = app.run() or 0
    if app.detached:
        sid = session_dir.name
        if exit_on_end:
            # The lifecycle in the parent prints the reattach line once the run
            # has been handed to the background (after its current step).
            print(f"[agent6] leaving the view: {sid} detaches to the background after this step.")
        else:
            print(f"[agent6] detached: {sid} keeps running.")
            print(f"          reattach:  agent6 attach {sid}")
        print("          (Ctrl+_ undoes typing in the composer)")
    return rc
