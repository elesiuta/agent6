# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The composer every conversation surface shares: the steer input and its
mode labels, the slash-command suggestions, the resume preset picker, the
history search, and the inline approval row."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, ClassVar, Literal

from rich.markup import escape
from rich.text import Text
from textual import events
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal
from textual.message import Message
from textual.screen import Screen
from textual.widgets import Select, Static, TextArea

from agent6.directive import STEER_COMMANDS
from agent6.ui.tui.menubar import (
    Menu,
    MenuItem,
)
from agent6.ui.tui.modals import HistorySearchModal
from agent6.viewmodel.tail import tail_events
from agent6.viewmodel.transcript import (
    operator_inputs,
)

ComposerMode = Literal["steer", "resume", "start"]


def composer_labels(mode: ComposerMode, *, continue_as: str = "") -> tuple[str, str]:
    """(border title, key hint) for the composer.

    One conversation view serves runs, plans and asks, so it says "session":
    a fixed "the run" is wrong two times in three. *continue_as* names the
    fork an undone run continues as (Enter resumes THAT session).
    """
    if mode == "steer":
        return ("steer this session (/pin, /compact [focus])", "Enter sends · Ctrl-J newline")
    if mode == "resume":
        title = f"continue as {continue_as}" if continue_as else "continue this session"
        return (title, "Enter resumes · Ctrl-J newline")
    return ("new task", "Enter starts · Ctrl-J newline")


def steer_suggestion_rows(text: str, *, mode: ComposerMode) -> list[tuple[str, str]]:
    """The steer directives matching the composer's first word while it is
    still being typed (`/…`, no whitespace yet): (command, help) rows, empty
    for ordinary text. /compact acts only on a live session, so a resume
    composer does not offer it; a draft offers only /parallel (the fan-out
    is the one directive a start understands)."""
    if not text.startswith("/") or any(ch.isspace() for ch in text):
        return []
    if mode == "start":
        offered = {c: h for c, h in STEER_COMMANDS.items() if c == "/parallel"}
    elif mode == "resume":
        offered = {c: h for c, h in STEER_COMMANDS.items() if c not in ("/compact", "/btw")}
    else:
        offered = STEER_COMMANDS
    return [(c, h) for c, h in offered.items() if c.startswith(text)]


def complete_steer(text: str, *, mode: ComposerMode) -> str | None:
    """Tab in a composer: the completed command word, or None when Tab should
    keep its focus-move meaning. A unique match completes with a trailing
    space; several matches advance to their longest common prefix, returning
    *text* unchanged when there is no progress so Tab never yanks focus away
    mid-command."""
    rows = steer_suggestion_rows(text, mode=mode)
    if not rows:
        return None
    if len(rows) == 1:
        return rows[0][0] + " "
    lcp = os.path.commonprefix([c for c, _h in rows])
    return lcp if len(lcp) > len(text) else text


class SteerSuggest(Static):
    """The command hints above a composer (the run views' analogue of the
    hub's model-suggestion line): one row per matching steer directive while
    the first word is being typed, hidden otherwise. Tab in the composer
    completes (see SteerInput.on_key)."""

    ALLOW_SELECT = False
    DEFAULT_CSS = """
    SteerSuggest { display: none; height: auto; padding: 0 1; background: $surface; }
    """

    def show_for(self, text: str, *, mode: ComposerMode) -> None:
        rows = steer_suggestion_rows(text, mode=mode)
        body: Text | None = None
        if rows:
            body = Text()
            for i, (cmd, help_) in enumerate(rows):
                if i:
                    body.append("\n")
                body.append(cmd, style="bold")
                body.append(f"  {help_}", style="dim")
        self.show_text(body)

    def show_text(self, body: Text | None) -> None:
        """Show *body* as the hint line, or hide the line for None."""
        if body is not None:
            self.update(body)
        show = body is not None
        if self.display != show:
            self.display = show


_INPUT_MAX_ROWS = 6  # the steer bar grows to this many rows, then scrolls internally

# The preset picker's first entry: "" => resume under the preset the run
# recorded (no --preset).
RECORDED_PRESET_LABEL = "(as recorded)"


class ResumePreset(Horizontal):
    """The row above a resume composer: pick the config preset the next leg
    continues under (`agent6 resume --preset`). A preset touches any setting,
    so it changes only between legs; the picker shows only while the composer
    resumes, and the choice lives on the host app (`resume_preset`), so the
    conversation and the dashboard composers agree."""

    DEFAULT_CSS = """
    ResumePreset { display: none; height: 3; padding: 0 1; }
    ResumePreset .resume-label { width: auto; padding: 1 1 0 0; color: $text-muted; }
    ResumePreset Select { width: 1fr; max-width: 40; }
    """

    def __init__(self, presets: list[str], **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._presets = presets

    def compose(self) -> ComposeResult:
        yield Static("continue under preset", classes="resume-label")
        yield Select(
            [(RECORDED_PRESET_LABEL, ""), *((p, p) for p in self._presets)],
            value="",
            allow_blank=False,
        )

    def on_select_changed(self, event: Select.Changed) -> None:
        setattr(self.app, "resume_preset", str(event.value))  # noqa: B010

    def show(self, shown: bool) -> None:
        if self.display != shown:
            self.display = shown
        if shown:
            # After the refresh: on the first paint the Select is not mounted
            # yet, and a value written before its mount leaves its label blank.
            self.call_after_refresh(self._sync)

    def _sync(self) -> None:
        wanted = getattr(self.app, "resume_preset", "")
        for picker in self.query(Select):
            if picker.value != wanted and wanted in ("", *self._presets):
                picker.value = wanted


# The run-control menu, shared verbatim by the two run views (this primary
# conversation and the dashboard) so they cannot drift. Every action resolves on
# the Agent6TUI app (the menu bar's dispatcher falls back to app actions).
RUN_MENU = Menu(
    "Run",
    (
        MenuItem("Search past messages…", "history_search", "ctrl+r"),
        MenuItem("Compact context now", "compact"),
        MenuItem("Stop after this step", "stop_step"),
        MenuItem("Stop now", "stop_now"),
        MenuItem("Resume this session", "resume"),
        MenuItem("Run this plan", "run_plan"),
        MenuItem("Fork this session", "fork"),
        MenuItem("Delete this session…", "delete_session"),
    ),
)


# The answers an open approval offers, in the vocabulary the CLI prompt and the
# modal speak ("yes" / "no" / "session" / "session-deny"): (key, answer, label,
# style). The row renders them and answers a click; the composer binds the keys.
APPROVAL_ANSWERS: tuple[tuple[str, str, str, str], ...] = (
    ("a", "yes", "allow", "bold green"),
    ("s", "session", "allow all (session)", "green"),
    ("d", "no", "deny", "bold red"),
    ("x", "session-deny", "deny all", "red"),
)
# Offered only by a standing approval (one the operator may answer for the session).
_STANDING_ANSWERS = frozenset({"session", "session-deny"})


class SteerInput(TextArea):
    """The bottom composer bar: a TextArea that submits on Enter (Ctrl+J /
    Shift+Enter insert a newline instead) and grows with its content up to
    _INPUT_MAX_ROWS. Two modes (set_mode): steer a LIVE run, or type the
    follow-up instruction a FINISHED run is resumed with. While an approval
    row is on the screen and the composer is empty, the row's keys answer it
    (check_action); anything typed makes them letters again."""

    ALLOW_MAXIMIZE = False  # a full-screen composer is never what Maximize means

    BINDINGS: ClassVar = [
        # TextArea's own undo stack; ctrl+z is the app's Detach (see Agent6TUI).
        Binding("ctrl+underscore", "undo", "Undo", show=False),
        # Priority: a plain letter is otherwise text before any binding runs.
        *(
            Binding(key, f"answer('{answer}')", label, priority=True, show=False)
            for key, answer, label, _style in APPROVAL_ANSWERS
        ),
    ]

    class Submitted(Message):
        def __init__(self, text: str) -> None:
            self.text = text
            super().__init__()

    def on_mount(self) -> None:
        self.set_mode(mode=self.mode)
        self._resize()

    policy = ""  # viewmodel.session_policy(...).short(), set once the run dir is known
    mode: ComposerMode = "steer"  # which directives apply (see steer_suggestion_rows)

    def set_mode(
        self, *, mode: ComposerMode, ctx_pct: int | None = None, continue_as: str = ""
    ) -> None:
        """Relabel for the session's state: steering (live), resuming
        (finished; *continue_as* names the fork an undone run resumes as), or
        starting (a draft), plus the context-window fill when known, right
        where you type. Only writes on a real change: this runs on every
        heartbeat, and same-value style writes still cost a refresh."""
        self.mode = mode
        title, keys = composer_labels(mode, continue_as=continue_as)
        ctx = f"ctx {ctx_pct}% · " if ctx_pct is not None else ""
        # The run's policy sits where the eye already goes for status, from the
        # same fold the CLI banner and the web header read.
        policy = f"{self.policy} · " if self.policy else ""
        # Border titles are markup: `[focus]` in a label or a bracket in a
        # model id would vanish (or crash) unescaped.
        title = escape(title)
        subtitle = escape(f"{policy}{ctx}{keys}")
        if self.border_title != title:
            self.border_title = title
        if self.border_subtitle != subtitle:
            self.border_subtitle = subtitle

    def on_key(self, event: events.Key) -> None:
        if event.key == "enter":
            event.prevent_default()
            event.stop()
            text = self.text.strip()
            if text:
                self.post_message(self.Submitted(text))
                self.clear()
        elif event.key in ("ctrl+j", "shift+enter"):
            event.prevent_default()
            event.stop()
            self.insert("\n")
        elif event.key == "tab":
            completed = complete_steer(self.text, mode=self.mode)
            if completed is not None:  # else Tab keeps its focus-move meaning
                event.prevent_default()
                event.stop()
                if completed != self.text:
                    self.load_text(completed)
                    self.move_cursor(self.document.end)

    def check_action(self, action: str, parameters: tuple[object, ...]) -> bool | None:
        if action == "answer":
            # A declined key falls through as the letter it is.
            rows = self.screen.query(ApprovalRow)
            return bool(rows) and rows.first().offers(str(parameters[0])) and not self.text

        return True

    def action_answer(self, answer: str) -> None:
        self.post_message(ApprovalRow.Answered(answer))

    def on_text_area_changed(self, _event: TextArea.Changed) -> None:
        self._resize()

    def _resize(self) -> None:
        rows = min(max(self.document.line_count, 1), _INPUT_MAX_ROWS)
        height = rows + 2  # + the rounded border
        current = self.styles.height
        if current is None or current.value != height:  # only relayout on a real change
            self.styles.height = height


def open_history_search(screen: Screen[Any], field: SteerInput, logs_path: Path) -> None:
    """Ctrl-R on a composer: pick one of this session's past messages (the
    task, then every steer -- journal-read, so resumes and other surfaces'
    steers appear) into *field* for editing. Newest first, flattened to one
    line each, repeats collapsed: the same list every surface's search shows."""
    if not field.display:
        screen.notify("this view has no composer to fill", severity="warning")
        return
    recorded = operator_inputs(tail_events(logs_path, follow=False))
    entries = list(dict.fromkeys(" ".join(t.split()) for t in reversed(recorded)))
    if not entries:
        screen.notify("no past messages this session yet", severity="warning")
        return

    def fill(text: str | None) -> None:
        if text:
            field.load_text(text)
            field.move_cursor(field.document.end)
            field.focus()

    screen.app.push_screen(HistorySearchModal(entries), fill)


class _AnswerLabel(Static):
    """One answer of the row: `[key] label`; a click answers."""

    def __init__(self, key: str, answer: str, label: str, style: str) -> None:
        super().__init__(Text(f"[{key}] {label}", style=style), classes=f"answer-{answer}")
        self.answer = answer

    def on_click(self) -> None:
        self.post_message(ApprovalRow.Answered(self.answer))


class ApprovalRow(Horizontal):
    """The answer row docked above the composer while an approval is open.
    A label answers on click from any focus; its key answers from the
    composer, which keeps focus, while the composer is empty (SteerInput's
    bindings), so a typed message never answers."""

    DEFAULT_CSS = """
    ApprovalRow { height: auto; padding: 0 1; background: $surface; }
    ApprovalRow Static { width: auto; padding: 0 2 0 0; }
    """

    class Answered(Message):
        def __init__(self, answer: str) -> None:
            super().__init__()
            self.answer = answer

    def __init__(self, *, standing: bool) -> None:
        super().__init__()  # no fixed id: a superseded row may still be unmounting
        self._standing = standing

    def compose(self) -> ComposeResult:
        for key, answer, label, style in APPROVAL_ANSWERS:
            if self.offers(answer):
                yield _AnswerLabel(key, answer, label, style)
        yield Static(Text("(keys work while the composer is empty; or click)", style="dim"))

    def offers(self, answer: str) -> bool:
        return self._standing or answer not in _STANDING_ANSWERS
