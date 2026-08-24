# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The run dashboard screen: the panes, their key bindings and menus, and the
coalesced repaint of the app's folded SessionState. `Agent6TUI` (`app.py`)
owns the data plane and pushes this screen."""

from __future__ import annotations

import contextlib
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, cast

try:
    from rich.markup import escape
    from rich.text import Text
    from textual.app import ComposeResult
    from textual.binding import Binding
    from textual.containers import Horizontal, ScrollableContainer, VerticalScroll
    from textual.css.query import NoMatches
    from textual.screen import Screen
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

from agent6.git_ops import commit_diff, diff_range
from agent6.sessions.ipc import (
    listening_ports,
)
from agent6.sessions.manifest import ManifestError, read_manifest
from agent6.ui.tui import clipboard
from agent6.ui.tui.composer import (
    RUN_MENU,
    ComposerMode,
    ResumePreset,
    SteerInput,
    SteerSuggest,
    open_history_search,
)
from agent6.ui.tui.logview import LogScreen
from agent6.ui.tui.menubar import Menu, MenuBar, MenuItem, menu_bindings
from agent6.ui.tui.modals import (
    ToolCallDetailModal,
)
from agent6.ui.tui.screen_chrome import MenuCommands, ScreenChrome
from agent6.ui.tui.settings import get_copy_method
from agent6.ui.tui.theme import (
    status_style,
)
from agent6.viewmodel import manifest_branches, manifest_header, session_compare
from agent6.viewmodel.format import (
    TASK_STATUS_GLYPH,
    format_compare,
    format_cost,
    spinner_frame,
    status_label,
)
from agent6.viewmodel.listing import (
    task_snippet,
)
from agent6.viewmodel.state import (
    MAX_LOG_TAIL,
    SessionState,
    ToolCallView,
    fold_until_commit,
)
from agent6.viewmodel.tail import tail_events

if TYPE_CHECKING:
    from agent6.ui.tui.app import Agent6TUI

_TASK_ICONS = TASK_STATUS_GLYPH

# How many recent tool calls the inline table shows. The RowSelected handler maps
# a visual row back through the same window, so both must use this one value.
_TOOL_TABLE_ROWS = 20


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

    def _details_state(self, s: SessionState) -> tuple[SessionState, str]:
        """The state the task tree and the cost line show: live, or as of the
        selected step (folded once per selection from the log)."""
        sha = self._step_sel
        if not sha:
            return s, ""
        if self._step_state is None or self._step_state[0] != sha:
            at = fold_until_commit(tail_events(self._tui.logs_path, follow=False), sha)
            if at is None:
                return s, ""
            self._step_state = (sha, at)
        at = self._step_state[1]
        return at, f" · as of iter {at.steps[-1].iteration}"

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
        self._rendered_tree: tuple[object, ...] | None = None
        self._rendered_tools: tuple[object, object] | None = None
        self._rendered_diff: tuple[object, ...] | None = None
        self._step_sel = ""  # a selected step's sha ("" = latest)
        self._cumulative = False
        self._nav_steps = -1  # how many steps the selector lists
        self._step_state: tuple[str, SessionState] | None = None  # the fold as of _step_sel
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
        # Only Agent6TUI pushes this screen.
        return cast("Agent6TUI", self.app)

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
        finished = self._end_label()
        ds, as_of = self._details_state(s)
        # tasks and cost are both as-of the selected step; ctx is live.
        done_n = sum(1 for t in ds.tasks if t.status in ("passed", "skipped"))
        step = f"tasks: {done_n}/{len(ds.tasks)}" if ds.tasks else "tasks: —"
        cost = f"[b]{format_cost(ds.budget.usd_total, partial=ds.budget.usd_partial)}[/]"
        # Consumption of the binding ledger: THIS leg's metered spend vs its
        # usd_cap (resume re-arms the cap while usd_total stays cumulative),
        # plus the unmetered-token fraction when that ledger has traffic.
        budget = ""
        if ds.budget.usd_cap > 0:
            leg_usd = ds.budget.usd_total - ds.budget.usd_prior_legs
            budget = f"   budget: {min(leg_usd / ds.budget.usd_cap, 1.0):.0%}"
        if ds.budget.tokens_unmetered and ds.budget.tokens_fallback_cap > 0:
            unmet = min(ds.budget.tokens_unmetered / ds.budget.tokens_fallback_cap, 1.0)
            budget += f"   unmetered: {unmet:.0%}"
        if ds.budget.plan_used_percent > 0:
            budget += f"   plan: {ds.budget.plan_used_percent:g}%"
            if ds.budget.plan_cap > 0:
                budget += f" (run {ds.budget.plan_consumed:g}/{ds.budget.plan_cap:g}pt)"
        pct = tui.context_pct()
        ctx = f"   ctx: {pct}%" if pct is not None else ""
        self.query_one("#top", Static).update(
            f"[b]agent6[/]  {step}   role: {escape(role_line)}   cost: {cost}{budget}{as_of}{ctx}"
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
        ds, as_of = self._details_state(s)
        if self._rendered_tree is None or not (
            self._rendered_tree[0] is ds.tasks
            and self._rendered_tree[1] == sel
            and self._rendered_tree[2] == as_of
        ):
            self._rendered_tree = (ds.tasks, sel, as_of)
            tree = self.query_one("#plan", Tree)
            tree.clear()
            tree.border_title = f"tasks{as_of}" if as_of else ""
            for tv in ds.tasks:
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
        self.query_one("#diff").border_title = (
            "diff · the model owns git"
            if self._git_control() == "model"
            else (f"diff{filt}" if sel else "")
        )
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
