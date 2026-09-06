# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""A full-screen, scrollable view of a run's LLM conversation (current or past).

The companion to `LogScreen`: it folds the same `logs.jsonl` stream through
the shared `TranscriptFold` into the conversation -- assistant reasoning and
text, every tool call with its result, commits, and the verdict -- with the same
glyphs the CLI stream uses.

Completed turns scroll in the main pane; a docked live pane at the bottom streams
the turn IN PROGRESS -- a reasoning model can think for 30-60s before producing a
tool call, so without it the view looks frozen.

The scrollback is a `Static` in a `VerticalScroll` (not a `RichLog`): a
`RichLog` renders as line Strips, which the framework's text selection cannot
extract, so its text is not copyable; a `Static` renders as `Content` and is
selectable -- matching the live pane, which is already a `Static`.
"""

from __future__ import annotations

import bisect
import contextlib
import os
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import ClassVar

from rich.text import Text
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical, VerticalScroll
from textual.css.query import NoMatches
from textual.geometry import Offset
from textual.screen import Screen
from textual.timer import Timer
from textual.widgets import Footer, Static, TextArea

from agent6.sessions.ipc import submit_steer, write_answer
from agent6.tools.background import shells_text
from agent6.ui.tui import clipboard
from agent6.ui.tui.composer import (
    RUN_MENU,
    ApprovalRow,
    ComposerMode,
    ResumePreset,
    SteerInput,
    SteerSuggest,
    open_history_search,
)
from agent6.ui.tui.logview import LogScreen
from agent6.ui.tui.menubar import (
    Menu,
    MenuBar,
    MenuItem,
    menu_bindings,
)
from agent6.ui.tui.modals import TextModal
from agent6.ui.tui.prompts import PromptDispatcher
from agent6.ui.tui.screen_chrome import MenuCommands, ScreenChrome
from agent6.ui.tui.settings import get_copy_method
from agent6.viewmodel import approval_parts, restate
from agent6.viewmodel.events import SESSION_START_EVENTS
from agent6.viewmodel.format import spinner_frame
from agent6.viewmodel.policy import session_policy
from agent6.viewmodel.state import SessionState
from agent6.viewmodel.tail import LogTail, tail_events
from agent6.viewmodel.transcript import (
    TranscriptFold,
    TranscriptItem,
)
from agent6.viewmodel.transcript_style import DetailLevel, StyleName, item_lines

_LIVE_TAIL = 1600  # chars of the in-progress turn kept in the live pane
# Sealed-chunk size for the transcript body. The body is a SEQUENCE of Static
# chunks, not one widget: appending to a single Static re-wraps the WHOLE
# transcript every poll (185ms at ~1800 lines, and growing -- the live-run input
# lag), while only the small tail chunk ever changes here.
_CHUNK_LINES = 200

# The single detail shortcut cycles through these in order.
_DETAIL_CYCLE: dict[DetailLevel, DetailLevel] = {
    "hidden": "collapsed",
    "collapsed": "expanded",
    "expanded": "hidden",
}


def _tail(text: str, n: int) -> str:
    return text if len(text) <= n else "…" + text[-n:]


# Semantic style name -> Rich style. The CLI has the sibling ANSI map; both skins
# render item_lines(), so the structure and which element is coloured live in ONE
# place (transcript_style) and can't drift.
_STYLE_RICH: dict[StyleName, str] = {
    "thinking": "#6C7086",
    "think-marker": "blue",
    "text": "",
    "call": "bold cyan",
    "verify": "bold yellow",
    "arg": "dim",
    "ok": "green",
    "fail": "red",
    "detail": "dim",
    "more": "dim italic",
    "tail": "dim",
    "commit": "magenta",
    "marker": "dim italic",
    "done-ok": "bold green",
    "done-fail": "bold yellow",
    "body": "",
    "done-detail": "dim",
    "operator": "bold green",
}


def _item_renderables(item: TranscriptItem, *, detail: DetailLevel) -> list[Text]:
    """The TUI skin over the shared item_lines(): one Rich Text per line, mapping
    each span's semantic style, with a blank line after the item for spacing."""
    lines = item_lines(item, detail=detail)
    if not lines:
        return []
    out: list[Text] = []
    for line in lines:
        text = Text()
        for chunk, style in line:
            text.append(chunk, style=_STYLE_RICH[style] or None)
        out.append(text)
    out.append(Text(""))  # one blank line after the item
    return out


# What Enter in the composer does: steer a LIVE session, resume a finished
# one with a follow-up, or start a new session from a draft.


def empty_conversation_note(word: str, detail: str, *, ended: bool) -> str:
    """The empty-conversation placeholder: a parked run says why it is parked
    and how to start it (the dashboard row carried the reason; this view said
    only "no conversation yet"); otherwise past tense only when the session
    positively ended."""
    if word == "parked":
        why = f": {detail}" if detail else ""
        return f"(parked{why} \u2014 type below to start it)"
    if ended:
        return "this run made no conversation"
    return "(no conversation yet; it appears as the run streams)"


class _ChromeStatic(Static):
    """A Static that never joins a text selection, so dragging over the title or
    the live pane doesn't grab their text (or stall the auto-scroll) mid-select.
    Only the transcript body (the `.conv-chunk` Statics) is selectable/copyable."""

    ALLOW_SELECT = False


_JUMP_LABEL = "↓ bottom · Ctrl+End"


class _JumpButton(Static):
    """A small floating jump-to-bottom pill: shown while the transcript is
    scrolled up; click -- or Ctrl+End -- snaps back to the live tail. Floats
    on the dropdown layer (the _Dropdown recipe), so it never displaces the
    layout."""

    ALLOW_SELECT = False
    DEFAULT_CSS = """
    _JumpButton {
        layer: dropdown; overlay: screen; constrain: none inside;
        display: none; width: auto; height: 1; padding: 0 1;
        background: $panel; color: $accent;
    }
    _JumpButton:hover { background: $primary 30%; }
    """

    def on_click(self) -> None:
        handler = getattr(self.screen, "action_scroll_bottom", None)
        if callable(handler):
            handler()


class ConversationScreen(ScreenChrome, Screen[None]):
    """Scrollable, live-following, selectable LLM conversation for a single run."""

    CSS = """
    ConversationScreen { background: $surface; }
    #conv-main { height: 1fr; }
    #conv-scroll { height: 1fr; }
    .conv-chunk { height: auto; padding: 0 1; pointer: text; }  /* selectable: I-beam */
    #conv-live {
        height: auto; max-height: 12; padding: 0 1;
        border-top: solid $border; background: $surface;
    }
    /* The steer bar: shown only while the run is live (see _sync_input). */
    #conv-input {
        display: none; height: auto; max-height: 8;
        border: round $primary; background: $surface;
    }
    #conv-input:focus { border: round $accent; }
    """

    _VIEW_ITEMS: ClassVar = (
        MenuItem("Detail: none / collapsed / expanded", "cycle_detail"),
        MenuItem("Scroll ↑ a page", "page_up"),
        MenuItem("Scroll ↓ a page", "page_down"),
        MenuItem("Scroll → top", "scroll_top"),
        MenuItem("Scroll → end", "scroll_bottom"),
        MenuItem("Reload the log", "reload"),
        MenuItem("Full log…", "view_logs"),
        MenuItem("Copy selection / all", "copy"),
        MenuItem("Copy via terminal", "suspend_copy"),
        MenuItem("Copy via pager", "pager"),
        MenuItem("Save transcript to file", "write_file"),
        MenuItem("Theme…", "choose_theme"),
        MenuItem("Copy method…", "choose_copy_method"),
    )
    _HELP_MENU: ClassVar = Menu(
        "Help",
        (
            MenuItem("Keys & actions", "help"),
            MenuItem("Command palette", "command_palette"),
        ),
    )
    # The class-level MENUS is the pushed-viewer shape (Esc dismisses back to
    # wherever it was opened). The PRIMARY view (the run app's main screen)
    # replaces it per-instance in __init__: File matches the dashboard's
    # (Back = to the hub, Quit), Run appears (the same RUN_MENU the dashboard
    # shows), and View gains the dashboard toggle. menu_bindings unions the
    # openers for both shapes, so Alt+r works only where Run exists.
    MENUS: ClassVar = (
        Menu("File", (MenuItem("Back", "close"),)),
        Menu("View", _VIEW_ITEMS),
        _HELP_MENU,
    )

    # The composer bar owns plain letters + Enter, so every shortcut here is a
    # priority binding (fires before the bar) on a modified key -- the same set,
    # in the same footer order, as the dashboard. Everything else lives in the
    # menu bar (which shows the shortcuts from these bindings) and the palette.
    # `?` opens help too, when focus is not in the bar.
    BINDINGS: ClassVar = [
        # Primary view only (check_action hides it on pushed viewers): flip
        # between the conversation and the dashboard.
        Binding("ctrl+d", "toggle_dashboard", "Dashboard", priority=True),
        Binding("ctrl+c", "copy", "Copy", priority=True),
        # The thinking/tool-detail cycle: none -> collapsed -> expanded.
        Binding("ctrl+t", "cycle_detail", "Detail", priority=True),
        Binding("ctrl+r", "history_search", "History", priority=True),
        Binding("escape", "close", "Back", key_display="Esc", priority=True),
        Binding("pageup", "page_up", "Scroll up", priority=True, show=False),
        Binding("pagedown", "page_down", "Scroll down", priority=True, show=False),
        Binding("ctrl+home", "scroll_top", "Top", priority=True, show=False),
        Binding("ctrl+end", "scroll_bottom", "End", priority=True, show=False),
        Binding("question_mark", "help", "Help", show=False),
        *menu_bindings((*MENUS, Menu("Run", ()))),  # + the Alt+r opener (primary shape)
    ]
    COMMANDS: ClassVar = {MenuCommands}
    HELP_TITLE: ClassVar = "agent6 — conversation"
    HELP_HINTS: ClassVar = (
        "Steer bar: Enter sends the instruction",
        "Ctrl-J or Shift+Enter inserts a newline",
    )

    def __init__(
        self,
        logs_path: Path,
        *,
        title: Callable[[str], str],
        primary: bool = False,
        presets: list[str] | None = None,
        prompts: PromptDispatcher | None = None,
    ) -> None:
        """*primary* marks the run app's main screen (Esc leaves the app, Ctrl+D
        toggles the dashboard); pushed read-only viewers (the hub, the dashboard)
        leave it False, where Esc just dismisses. *presets* are the config
        presets a resume may continue under (the primary view's picker).
        *prompts* is the host's dispatcher, the one record of which prompts a
        surface already took (a modal on another screen, this screen's row)."""
        super().__init__()
        self._logs_path = logs_path
        self._prompts = prompts
        self._title = title
        self._primary = primary
        self._presets = presets if presets is not None else []
        # The instance's menus (MENUS stays the class-level viewer shape, which
        # menu_bindings derives the Alt+letter openers from -- same titles).
        self._menus: tuple[Menu, ...] = self.MENUS
        if primary:
            self._menus = (
                Menu(
                    "File",
                    (
                        MenuItem("Back", "close"),
                        MenuItem("Quit", "quit_hub", "ctrl+q"),
                    ),
                ),
                RUN_MENU,
                Menu("View", (*self._VIEW_ITEMS, MenuItem("Dashboard…", "toggle_dashboard"))),
                self._HELP_MENU,
            )
        self._detail: DetailLevel = "collapsed"  # one shortcut cycles none/collapsed/expanded
        self._tail = LogTail(logs_path)
        self._fold = TranscriptFold()
        self._content = Text()  # the WHOLE transcript (copy + anchor bookkeeping)
        self._item_starts: list[int] = []  # logical start line of each rendered item (anchor)
        self._content_lines = 0  # total logical lines in _content
        # The not-yet-sealed tail of the transcript: the only widget content the
        # live appends re-render (see _CHUNK_LINES).
        self._tail_text = Text()
        self._tail_lines = 0
        self._approval: tuple[str, str, bool] | None = None  # (id, prompt, standing)
        self._approval_done: str | None = None
        self._row: ApprovalRow | None = None
        self._row_id = ""
        self._live_think: list[str] = []
        self._live_text: list[str] = []
        # A session-start event (session.start OR loop.resume.start) seen and no
        # session.end since -> the steer bar shows.
        self._live = False
        self._spin = 0  # live-pane spinner tick, advanced by _poll
        self._timer: Timer | None = None  # the 0.3s poll; paused while covered

    def compose(self) -> ComposeResult:
        yield MenuBar(self._menus)  # top row: menus + "agent6 — <run>", like every screen
        with Vertical(id="conv-main"):
            with VerticalScroll(id="conv-scroll"):
                # The transcript body: sealed chunks are mounted above this
                # active tail as the log grows (each renders as Content ->
                # selectable; only the tail is ever re-rendered).
                yield Static(id="conv-tail", classes="conv-chunk")
            yield _ChromeStatic("", id="conv-live")  # chrome: not part of a selection
        yield Static(id="conv-approval", classes="conv-chunk")  # the open approval, inline
        yield SteerSuggest(id="conv-suggest")  # command hints while typing `/…`
        yield ResumePreset(self._presets, id="conv-preset")  # shown while the composer resumes
        yield SteerInput(id="conv-input")  # steer bar (hidden unless the run is live)
        yield _JumpButton(_JUMP_LABEL, id="conv-jump")  # floats; shown when scrolled up
        yield Footer()  # Footer is ALLOW_SELECT=False in textual already

    def on_mount(self) -> None:
        self.app.sub_title = self._title("conversation")  # run context in the menu bar
        self._reload()
        self._timer = self.set_interval(0.3, self._poll)
        # The jump pill follows the scroll position (mouse wheel included) and
        # content growth (the poll calls _sync_jump too).
        self.watch(self._scroll(), "scroll_y", self._sync_jump, init=False)

    def _sync_jump(self, *_: object) -> None:
        """Show the jump-to-bottom pill while scrolled up, pinned to the scroll
        area's bottom-right corner (an overlay: never part of the layout)."""
        with contextlib.suppress(NoMatches):
            jump = self.query_one("#conv-jump", _JumpButton)
            scroll = self._scroll()
            show = scroll.max_scroll_y > 0 and not self._at_bottom(scroll)
            if jump.display != show:  # a same-value write still costs a relayout
                jump.display = show
            if show:
                region = scroll.region
                width = len(_JUMP_LABEL) + 2  # + the 1-cell padding each side
                jump.absolute_offset = Offset(
                    max(region.x, region.right - width - 2), max(region.y, region.bottom - 2)
                )

    def on_screen_suspend(self) -> None:
        # Hidden behind the dashboard (or another pushed screen): stop polling.
        if self._timer is not None:
            self._timer.pause()

    def on_screen_resume(self) -> None:
        # Back on top (a dashboard toggle, or a viewer above was closed):
        # re-stamp the title the covering screen may have changed, catch up on
        # events that landed while hidden, and resume the poll.
        self.app.sub_title = self._title("conversation")
        if self._timer is not None:
            self._poll()
            self._timer.resume()

    def menus(self) -> tuple[Menu, ...]:
        return self._menus

    def _scroll(self) -> VerticalScroll:
        return self.query_one("#conv-scroll", VerticalScroll)

    def _append(self, item: TranscriptItem) -> bool:
        self._item_starts.append(self._content_lines)  # where this item begins (for the anchor)
        wrote = False
        for line in _item_renderables(item, detail=self._detail):
            self._content.append_text(line)
            self._content.append("\n")
            self._content_lines += 1
            self._tail_text.append_text(line)
            self._tail_text.append("\n")
            self._tail_lines += 1
            wrote = True
        return wrote

    def _tail_widget(self) -> Static:
        return self.query_one("#conv-tail", Static)

    def _flush_tail(self) -> None:
        """Push the tail chunk to its widget, sealing it into an immutable chunk
        above once it is big enough -- so a growing transcript never re-renders
        more than the last ~_CHUNK_LINES lines."""
        tail = self._tail_widget()
        tail.update(self._tail_text)
        if self._tail_lines >= _CHUNK_LINES:
            sealed = Static(self._tail_text, classes="conv-chunk conv-sealed")
            self._scroll().mount(sealed, before=tail)
            self._tail_text = Text()
            self._tail_lines = 0
            tail.update(self._tail_text)

    def _track_live(self, event: dict[str, object]) -> None:
        etype = event.get("type")
        if etype in SESSION_START_EVENTS:
            # session.start OR loop.resume.start: a resumed leg is live too (the
            # dashboard's fold un-finishes on ResumeStart; this screen must
            # agree, or its composer mislabels the live leg "resume" and a
            # submit spawns a second resume that dies on the run lock while
            # the toast claims success).
            self._live = True
        elif etype == "session.end":
            self._live = False
        if etype == "approval.prompt":
            self._approval = (
                str(event.get("id", "")),
                str(event.get("prompt", "")),
                bool(event.get("standing", True)),
            )
            self._approval_done = None
        elif etype == "approval.answer":
            if self._approval is not None and self._approval[0] == str(event.get("id", "")):
                self._note_answered(bool(event.get("approved")))
        if etype in ("role.call", "role.result"):
            self._live_think.clear()
            self._live_text.clear()
            self._approval_done = None
        elif etype == "role.thinking_delta":
            self._live_think.append(str(event.get("text", "")))
        elif etype == "role.text_delta":
            self._live_text.append(str(event.get("text", "")))

    def _open_approval(self) -> tuple[str, str, bool] | None:
        """The approval awaiting an answer: the host's folded state when this
        screen runs under the Agent6TUI host (one fold, whatever fed it), else
        what this screen's own tail saw."""
        state = getattr(self.app, "state", None)
        if isinstance(state, SessionState):
            if self._approval is not None:
                aid = self._approval[0]
                answered = next((a for a in state.pending_approvals if a.id == aid), None)
                if answered is not None and answered.answered:
                    self._note_answered(bool(answered.approved))
            open_ones = [
                ap for ap in state.pending_approvals if not ap.answered and not self._taken(ap.id)
            ]
            if open_ones:
                ap = open_ones[-1]  # the newest: a resumed leg reuses prompt ids
                return ap.id, ap.prompt, ap.standing
        return self._approval

    def _taken(self, aid: str) -> bool:
        if self._prompts is None:
            return False
        return self._prompts.seen(self._logs_path.parent, aid)

    def _note_answered(self, approved: bool) -> None:
        if self._approval is None:
            return
        head, payload = approval_parts(self._approval[1])
        verdict = "allowed" if approved else "denied"
        self._approval_done = f"{verdict} · {(payload or head).splitlines()[0][:60]}"
        self._approval = None

    def _render_approval(self) -> None:
        """The open approval as a conversation-tail item (the command under
        judgment, fixed-width) with the answer row docked above the composer;
        after the answer the item collapses to one dim line until the next
        model turn."""
        item = self.query_one("#conv-approval", Static)
        current = self._open_approval()
        if current is not None and not self._host_live():
            # The run died with the prompt open: the fact stays visible, the
            # key row (whose answer would reach nothing) does not.
            self._approval = current
            head, payload = approval_parts(current[1])
            body = Text(f"? {head}: approval pending when the run ended", style="dim")
            if payload:
                body.append("\n" + "\n".join(f"    {ln}" for ln in payload.splitlines()))
            item.update(body)
            item.display = True
            current = None
        if current is not None:
            self._approval = current
            aid, prompt, standing = current
            head, payload = approval_parts(prompt)
            body = Text()
            body.append("? ", style="bold yellow")
            body.append(f"{head}: approval needed", style="bold")
            if payload:
                body.append("\n" + "\n".join(f"    {ln}" for ln in payload.splitlines()))
            item.update(body)
            item.display = True
            if self._row is None or self._row_id != aid:
                # One row per approval: a resumed leg reuses prompt ids, so a
                # new prompt always gets a fresh row (the old one may still be
                # unmounting).
                if self._row is not None:
                    self._row.remove()
                self._row = ApprovalRow(standing=standing)
                self._row_id = aid
                self.mount(self._row, before=self.query_one("#conv-suggest"))
                self.call_after_refresh(self._row.focus)
            return
        if self._row is not None:
            self._row.remove()
            self._row = None
            self._row_id = ""
        if self._approval is not None:
            return  # the dead run's prompt, rendered above
        if self._approval_done:
            item.update(Text(f"? {self._approval_done}", style="dim"))
            item.display = True
        else:
            item.display = False

    def on_approval_row_answered(self, message: ApprovalRow.Answered) -> None:
        if self._approval is None:
            return
        aid = self._approval[0]
        if not self._host_live():
            self.notify("the run is gone: the answer reached nothing", severity="warning")
            return
        write_answer(self._logs_path.parent, aid, message.answer)
        if self._prompts is not None:
            self._prompts.claim(self._logs_path.parent, aid)
        self._note_answered(message.answer in ("yes", "session"))
        self._render_approval()
        self.notify(f"answered: {message.answer}")
        with contextlib.suppress(NoMatches):
            self.query_one("#conv-input", SteerInput).focus()

    def _render_live(self) -> None:
        self._render_approval()
        live = self.query_one("#conv-live", Static)
        if not self._host_live():
            # The deltas of the turn a killed worker never finished sit in the
            # buffers forever (only role.call/role.result clear them), so this
            # pane would keep saying "thinking…" over a corpse -- on the primary
            # view, which carries no status label to contradict it.
            live.display = False
            return
        if self._host_waiting():
            # Blocked on the operator (an approval or a question is open): the
            # model is not thinking and no tool is running, so the pulse that
            # says so would lie under the very modal asking.
            live.display = True
            live.update(Text("waiting for your answer…", style="bold yellow"))
            return
        think = "".join(self._live_think).strip()
        text = "".join(self._live_text).strip()
        frame = spinner_frame(self._spin)
        if not think and not text:
            if not self._live:
                live.display = False
                return
            # Mid-run with nothing streaming (the model is being called, or
            # tools are executing): a vanished pane reads as frozen for the
            # whole stretch, so keep something moving.
            body = Text()
            body.append(f"{frame} working… ", style="bold cyan")
            live.display = True
            live.update(body)
            return
        body = Text()
        if think:
            # Always show the live "thinking…" indicator (feedback that a turn is
            # working); stream the reasoning itself only when expanded (muted grey).
            body.append(f"{frame} thinking… ", style="bold cyan")
            if self._detail == "expanded":
                body.append(_tail(think, _LIVE_TAIL), style="#6C7086")
        if text:
            if think:
                body.append("\n\n")
            body.append(_tail(text, _LIVE_TAIL))
        live.display = True
        live.update(body)

    def _reload(self) -> None:
        """Re-read the whole log from scratch (mount, reload, detail cycle)."""
        self._tail = LogTail(self._logs_path)
        self._fold = TranscriptFold()
        self._content = Text()
        self._item_starts = []
        self._content_lines = 0
        self._tail_text = Text()
        self._tail_lines = 0
        self.query(".conv-sealed").remove()  # rebuilt below by _flush_tail
        self._live_think.clear()
        self._live_text.clear()
        self._live = False
        wrote = False
        for event in self._tail.read():
            self._track_live(event)
            for item in self._fold.feed(event):
                wrote = self._append(item) or wrote
        if wrote:
            self._flush_tail()
        else:
            # Past tense only when the host POSITIVELY knows the session ended.
            # `_host_live`'s event-derived fallback is False before the first
            # event, so using it here would promise nothing to a run that has
            # simply not started streaming yet -- the same lie, inverted.
            live_fn = getattr(self.app, "session_controllable", None)
            ended = callable(live_fn) and not live_fn()
            word, detail = getattr(self.app, "dir_status", ("", ""))
            empty = empty_conversation_note(word, detail, ended=ended)
            self._tail_widget().update(Text(empty, style="dim italic"))
        self._render_live()
        self._sync_input()
        self._scroll().scroll_end(animate=False)
        self._focus_default()

    def _at_bottom(self, scroll: VerticalScroll) -> bool:
        """Following the log: at the bottom within a small tolerance, so a one-line
        layout nudge (the live pane or steer bar resizing) keeps follow mode, while
        a deliberate scroll up of more than that drops it."""
        return scroll.max_scroll_y - scroll.scroll_y <= 2.0

    def _poll(self) -> None:
        """Append newly-completed turns (sticking to the bottom unless scrolled
        up) and refresh the live in-progress pane."""
        self._spin += 1
        self._render_approval()
        new_events = self._tail.read()
        if not new_events:
            # No data this tick, but a live run's pane must keep moving: the
            # spinner is the only sign of life between events. Same follow
            # discipline as the data path: a pane repaint can resize the
            # viewport and drop bottom-follow on a quiet tick.
            if self._live and self._host_live():
                scroll = self._scroll()
                following = self._at_bottom(scroll)
                self._render_live()
                if following:
                    scroll.scroll_end(animate=False)
            return
        scroll = self._scroll()
        following = self._at_bottom(scroll)  # BEFORE this frame's layout changes
        wrote = False
        for event in new_events:
            self._track_live(event)
            for item in self._fold.feed(event):
                wrote = self._append(item) or wrote
        if wrote:
            self._flush_tail()
        self._render_live()
        self._sync_input()
        # Re-pin AFTER the live pane / steer bar have (re)sized this frame: growing
        # them shrinks the scroll viewport and would otherwise nudge us off the exact
        # bottom, silently dropping follow mode even when nothing new was appended.
        if following:
            scroll.scroll_end(animate=False)
        self._sync_jump()

    def _bar_shown(self) -> bool:
        # The primary view keeps the bar even after the run ends (Enter then
        # RESUMES the run with the typed follow-up); a pushed viewer shows it
        # only while there is a live run to steer.
        return self._live or self._primary

    def _host_waiting(self) -> bool:
        """Whether the run is blocked on the operator, per the Agent6TUI host's
        dir status ("waiting"); False on a pushed viewer with no host."""
        status = getattr(self.app, "dir_status", None)
        return isinstance(status, tuple) and status[0] == "waiting"

    def _host_live(self) -> bool:
        """Whether the run is still live, per the Agent6TUI host's dir status
        (which knows a dead worker and a parked run) and falling back to this
        screen's event-derived tracking on a pushed viewer with no host."""
        live_fn = getattr(self.app, "session_controllable", None)
        return bool(live_fn()) if callable(live_fn) else self._live

    def _sync_input(self) -> None:
        """Show the composer bar (steer when live, continue when finished on the
        primary view) and keep its labels matching the run's state. The
        liveness and context readout come from the Agent6TUI host when there is
        one -- its dir status knows a dead worker and a parked run, which the
        event stream alone cannot (this screen's own _live stays True over a
        corpse, so a typed steer would go to a run that never reads it); pushed
        viewers on another host keep the event-derived tracking."""
        with contextlib.suppress(NoMatches):
            bar = self.query_one("#conv-input", SteerInput)
            shown = self._bar_shown()
            if bar.display != shown:  # a same-value write still costs a relayout
                bar.display = shown
            mode: ComposerMode = "steer" if self._host_live() else "resume"
            self.query_one("#conv-preset", ResumePreset).show(shown and mode == "resume")
            if bar.display:
                if not bar.policy:  # folded once: the manifest does not change mid-run
                    bar.policy = session_policy(self._logs_path.parent).short()
                pct_fn = getattr(self.app, "context_pct", None)
                pct = pct_fn() if callable(pct_fn) else None
                fork = getattr(self.app, "continue_as", "")
                bar.set_mode(
                    mode=mode,
                    ctx_pct=pct if isinstance(pct, int) else None,
                    continue_as=fork if isinstance(fork, str) else "",
                )

    def refresh_liveness(self) -> None:
        """Relabel the composer for a liveness change that came with no event
        to poll (the host's dir-status probe: a worker died, a parked run got
        resumed) -- called by the Agent6TUI host, covered or not, so the two
        run views can never disagree about the bar's mode."""
        self._sync_input()
        with contextlib.suppress(NoMatches):
            self._render_live()

    def focus_bar(self) -> None:
        """Focus the composer bar if it is shown (an external steer request
        routes here instead of popping a dialog)."""
        with contextlib.suppress(NoMatches):
            bar = self.query_one("#conv-input", SteerInput)
            if bar.display:
                bar.focus()

    def _focus_default(self) -> None:
        """Open with the composer bar focused when it is shown (type to steer a
        live run, or a follow-up to resume a finished one, immediately); a
        historical viewer focuses the scrollback for keyboard nav."""
        if self._bar_shown():
            with contextlib.suppress(NoMatches):
                self.query_one("#conv-input", SteerInput).focus()
                return
        self._scroll().focus()

    def on_steer_input_submitted(self, message: SteerInput.Submitted) -> None:
        """A line typed into the composer bar. With an Agent6TUI host, the host
        owns the routing (submit_instruction): live steer vs resume by its dir
        status, plus the `/compact [focus]` parse: routed by this screen's own
        live path, '/compact …' would reach the model as a literal steer while
        the bar's title advertises it. A pushed viewer on another host keeps
        the direct live-steer bridge."""
        submit = getattr(self.app, "submit_instruction", None)  # the Agent6TUI host
        if callable(submit):
            submit(message.text)
            return
        if message.text.strip() == "/restate":
            text = restate(list(tail_events(self._logs_path, follow=False)))
            self.app.push_screen(TextModal("since your last message", text))
            return
        if message.text.strip() == "/shells":
            self.app.push_screen(
                TextModal("background commands", shells_text(self._logs_path.parent))
            )
            return
        if self._live:
            submit_steer(self._logs_path.parent, message.text)
            self.notify("steering this session…")

    def action_history_search(self) -> None:
        open_history_search(self, self.query_one("#conv-input", SteerInput), self._logs_path)

    def on_text_area_changed(self, event: TextArea.Changed) -> None:
        if event.text_area.id != "conv-input":
            return
        with contextlib.suppress(NoMatches):
            self.query_one("#conv-suggest", SteerSuggest).show_for(
                event.text_area.text, mode="steer" if self._host_live() else "resume"
            )

    # -- copy ---------------------------------------------------------------
    def _emit(self, seq: str) -> None:
        """Write a raw terminal escape (an OSC 52 clipboard-set) through the driver."""
        driver = self.app._driver  # pyright: ignore[reportPrivateUsage]
        if driver is not None:
            driver.write(seq)

    def _selected_or_all(self) -> tuple[str, str]:
        """The current transcript-body selection if any, else the whole transcript."""
        body_selection = self._body_selection()
        if body_selection and body_selection.strip():
            return body_selection, "selection"
        return self._content.plain, "whole transcript"

    def _body_selection(self) -> str | None:
        """Selected text from the transcript BODY only (its chunk Statics, in
        document order), so a drag that strays over the footer or live pane
        never copies their text -- Textual's screen-wide get_selected_text()
        would otherwise include them (they are chrome)."""
        parts: list[str] = []
        for chunk in self.query(".conv-chunk"):
            selection = self.selections.get(chunk)
            if selection is None:
                continue
            grabbed = chunk.get_selection(selection)
            if grabbed is not None:
                parts.append(grabbed[0])
        return "\n".join(parts) if parts else None

    def get_selected_text(self) -> str | None:
        """Textual's copy entry point, restricted to the transcript body."""
        return self._body_selection()

    def _copy_text(self, text: str, *, method: str) -> str:
        """Copy *text* using the resolved *method*; returns a short status."""
        return clipboard.emit_clipboard(text, clipboard.resolve_method(method), self._emit)

    def action_copy(self) -> None:
        text, what = self._selected_or_all()
        if not text:
            self.notify("nothing to copy yet")
            return
        try:
            status = self._copy_text(text, method=get_copy_method())
        except (OSError, subprocess.CalledProcessError) as exc:
            self.notify(f"copy failed: {exc}", severity="error")
            return
        self.notify(f"copied {what} ({status})")

    def action_write_file(self) -> None:
        path = clipboard.write_transcript_file(self._content.plain)
        self.notify(f"wrote transcript to {path}")

    def action_suspend_copy(self) -> None:
        """Drop to the native terminal, print the text (scroll + select + copy with
        the terminal, which always works), Enter to return."""
        text, what = self._selected_or_all()
        with self.app.suspend():
            print(f"\n===== COPY BELOW ({what}): select and copy in your terminal =====\n")
            print(text)
            print("\n===== END (press Enter to return) =====")
            with contextlib.suppress(EOFError):
                input()

    def action_pager(self) -> None:
        """Open the text in $PAGER (scroll + select + copy natively)."""
        text, _ = self._selected_or_all()
        pager = os.environ.get("PAGER") or "less"
        cmd = [pager, "-R"] if Path(pager).name.startswith("less") else [pager]
        with self.app.suspend():
            try:
                subprocess.run(cmd, input=text, text=True, check=False)
            except OSError as exc:
                print(f"pager {pager!r} failed: {exc}\nPress Enter to return.")
                with contextlib.suppress(EOFError):
                    input()

    def action_reload(self) -> None:
        self._reload()

    def action_view_logs(self) -> None:
        """Open the raw event log of this run (the audit-log companion view)."""
        self.app.push_screen(LogScreen(self._logs_path, title=lambda: self._title("logs")))

    def action_cycle_detail(self) -> None:
        """Cycle the transcript's detail level (hidden -> collapsed -> expanded), keeping
        the block at the top of the viewport anchored across the re-render."""
        self._reload_keeping_place(lambda: setattr(self, "_detail", _DETAIL_CYCLE[self._detail]))

    def _item_visual_starts(self) -> list[int]:
        """The visual (wrapped) row where each rendered item begins, at the current
        body width. One pass over the content, so it is cheap enough for a
        user-initiated re-render even on a long transcript."""
        width = max(1, self._tail_widget().content_size.width)
        starts: list[int] = []
        visual = 0
        nxt = 0
        for logical, line in enumerate(self._content.split("\n")):
            while nxt < len(self._item_starts) and self._item_starts[nxt] == logical:
                starts.append(visual)
                nxt += 1
            visual += max(1, -(-line.cell_len // width))  # ceil(cell_len / width)
        starts.extend([visual] * (len(self._item_starts) - nxt))
        return starts

    def _reload_keeping_place(self, flip: Callable[[], None]) -> None:
        """Apply *flip*, re-render, and restore the reading position: pinned to the
        bottom if we were following, else anchored to the block that was at the top of
        the viewport (kept at the same viewport offset across the re-render, so a block
        expanding above doesn't carry your place away)."""
        scroll = self._scroll()
        following = self._at_bottom(scroll)
        top = scroll.scroll_y
        old_visual = self._item_visual_starts()
        anchor = bisect.bisect_right(old_visual, top) - 1 if old_visual else -1
        offset = top - old_visual[anchor] if 0 <= anchor < len(old_visual) else 0.0
        flip()
        self._reload()  # rebuilds _content + _item_starts, ending scrolled to the bottom
        if following or not (0 <= anchor < len(self._item_starts)):
            return
        self._scroll().scroll_to(y=self._item_visual_starts()[anchor] + offset, animate=False)

    def action_scroll_top(self) -> None:
        self._scroll().scroll_home(animate=False)

    def action_scroll_bottom(self) -> None:
        self._scroll().scroll_end(animate=False)

    def action_page_up(self) -> None:
        self._scroll().scroll_page_up(animate=False)  # instant: animation reads as lag

    def action_page_down(self) -> None:
        self._scroll().scroll_page_down(animate=False)

    def action_close(self) -> None:
        if self._primary:
            # The run app's main screen: Back means leave the run view entirely
            # (the Agent6TUI host's to_hub exits with the back-to-hub code).
            handler = getattr(self.app, "action_to_hub", None)
            if callable(handler):
                handler()
            return
        self.dismiss()

    def action_quit_hub(self) -> None:
        handler = getattr(self.app, "action_quit_hub", None)  # the Agent6TUI host
        if callable(handler):
            handler()

    def action_toggle_dashboard(self) -> None:
        if not self._primary:
            return
        handler = getattr(self.app, "action_toggle_dashboard", None)  # the Agent6TUI host
        if callable(handler):
            handler()

    def check_action(self, action: str, parameters: tuple[object, ...]) -> bool | None:
        del parameters
        if action == "toggle_dashboard":
            return self._primary  # pushed viewers have no dashboard to toggle
        return True
