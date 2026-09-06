# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Shared TUI form widgets: the `[x]`/`[ ]` chooser (:class:`ChoiceField`)
and the flat action label (:class:`ActionItem`). Kept in their own module so
every dialog (config editor, provider form, theme picker) uses the exact same
accent-driven, arrow-navigable controls with no per-screen drift."""

from __future__ import annotations

try:
    from rich.color import Color
    from rich.text import Text
    from textual import events
    from textual.containers import ScrollableContainer
    from textual.geometry import Region
    from textual.message import Message
    from textual.widget import Widget
    from textual.widgets import Input, Static
except ImportError as e:  # pragma: no cover - clear runtime message
    raise SystemExit("The TUI widgets need textual: pip install 'agent6[tui]'") from e


def _scroll_row_into_view(widget: Widget, row: int) -> None:
    """Scroll *widget*'s nearest scrollable ancestor so its content *row* is
    visible (the widget isn't itself scrollable, so scroll_to_region on it
    no-ops; the ancestor must be driven directly)."""
    for node in widget.ancestors:
        if isinstance(node, ScrollableContainer):
            content_y = (
                widget.content_region.y
                + row
                - node.scrollable_content_region.y
                + node.scroll_offset.y
            )
            node.scroll_to_region(
                Region(0, content_y, 1, 1), animate=False, force=True, x_axis=False
            )
            return


def focus_neighbor(widget: Widget, direction: int) -> None:
    """Move focus to the next/previous *control* (ChoiceField / TypeaheadField /
    Input / ActionItem)
    in the dialog, skipping scroll containers and NOT wrapping — so the top and
    bottom of a dialog are hard stops, never a jump to a focusable scroll box (or
    to the far end) that strands the arrows."""
    kinds = (ChoiceField, TypeaheadField, Input, ActionItem)
    nav = [w for w in widget.screen.focus_chain if isinstance(w, kinds)]
    for i, w in enumerate(nav):
        if w is widget:
            j = i + direction
            if 0 <= j < len(nav):
                nav[j].focus()
            return


def _selection_bar(primary: str) -> str:
    """A rich style for a full-row selection bar in *primary*, with black/white
    ink chosen by luminance so the text stays readable on ANY theme's color."""
    rgb = Color.parse(primary).get_truecolor()
    lum = 0.299 * rgb.red + 0.587 * rgb.green + 0.114 * rgb.blue
    ink = "#11111b" if lum > 140 else "#f8f8f2"
    return f"bold {ink} on {primary}"


class ChoiceField(Widget, can_focus=True):
    """A natural terminal chooser: a vertical `[x]`/`[ ]` list. ↑↓ move a
    HIGHLIGHT (the selection does NOT follow, so arrowing through to the next
    field never corrupts the value); Space (or Enter) selects the highlighted
    row; ↑↓ hand off focus at the top/bottom edge so a dialog reads as one
    continuous ↑↓ chain. A single focusable widget -- Tab-ing onto it never
    changes the value.

    With `allow_custom` the last row is an inline text field: highlight it and
    type (typing selects it). Click selects. Posts :class:`Changed` when the
    SELECTION changes (live-preview dialogs, e.g. the theme picker). Space
    consumes the key; Enter also bubbles, so a dialog can confirm on Enter."""

    DEFAULT_CSS = """
    ChoiceField { height: auto; width: 1fr; text-wrap: nowrap; text-overflow: ellipsis; }
    """

    class Changed(Message):
        def __init__(self, field: ChoiceField) -> None:
            self.field = field
            super().__init__()

    def __init__(
        self,
        options: tuple[str, ...],
        current: str,
        *,
        allow_custom: bool = False,
        id: str | None = None,
        classes: str | None = None,
    ) -> None:
        super().__init__(id=id, classes=classes)
        self._options = list(options)
        self._allow_custom = allow_custom
        in_list = current in self._options
        self._custom_text = "" if (in_list or not allow_custom) else current
        if in_list:
            sel = self._options.index(current)
        elif allow_custom:
            sel = len(self._options)  # the inline custom row
        else:
            sel = 0
        self._sel = sel  # the chosen ([x]) row
        self._cursor = sel  # the highlighted row
        self._hover = -1  # the mouse-hovered row (-1 = none), like a DataTable
        self._pos = len(self._custom_text)  # caret within the custom text

    @property
    def _row_count(self) -> int:
        return len(self._options) + (1 if self._allow_custom else 0)

    @property
    def _custom_row(self) -> int:
        return len(self._options) if self._allow_custom else -1

    @property
    def index(self) -> int:
        return self._sel

    @property
    def value(self) -> str:
        if self._sel == self._custom_row:
            return self._custom_text
        if 0 <= self._sel < len(self._options):
            return self._options[self._sel]
        return ""

    def select_value(self, value: str) -> None:
        """Programmatically select `value` if it is one of the fixed options
        (a no-op otherwise, including the custom row). Silent -- posts no
        `Changed` -- so a dialog can prefill this field from a chosen preset
        without retriggering its own change handlers."""
        if value in self._options:
            self._sel = self._cursor = self._options.index(value)
            self.refresh(layout=True)

    def render(self) -> Text:
        focused = self.has_focus
        width = max(self.size.width, 1)
        try:
            bar = _selection_bar(self.app.current_theme.primary)
        except Exception:  # pragma: no cover - defensive (teardown/no theme)
            bar = "bold reverse"
        # A subtle hover bar ($panel) for the mouse row -- weaker than the primary
        # cursor bar, matching the DataTable's hover (so the picker isn't "dead"
        # under the mouse). $boost would be transparent, so use the resolved panel.
        hover_bg = ""
        if self._hover >= 0:
            try:
                hover_bg = f"on {self.app.get_css_variables()['panel']}"
            except Exception:  # pragma: no cover - defensive
                hover_bg = ""
        out = Text()
        for i in range(self._row_count):
            is_option = i < len(self._options)
            mark = "[x]" if i == self._sel else "[ ]"
            label = self._options[i] if is_option else (self._custom_text or "custom…")
            line = Text(f"{mark} ")
            line.append(label, style="" if (is_option or self._custom_text) else "dim")
            if (not is_option) and focused and i == self._cursor:
                line.append("▌")  # caret on the (highlighted) custom row
            line.pad_right(max(0, width - line.cell_len))
            if focused and i == self._cursor:
                line.stylize(bar)  # the moving highlight (keyboard)
            else:
                if hover_bg and i == self._hover:
                    line.stylize(hover_bg)  # the mouse row (subtle)
                if i == self._sel:
                    line.stylize("bold")  # the chosen row, when not highlighted
            out.append_text(line)
            if i < self._row_count - 1:
                out.append("\n")
        return out

    def on_key(self, event: events.Key) -> None:
        key = event.key
        if key == "up":
            event.stop()
            if self._cursor <= 0:
                focus_neighbor(self, -1)
            else:
                self._cursor -= 1
                self._moved()
        elif key == "down":
            event.stop()
            if self._cursor >= self._row_count - 1:
                focus_neighbor(self, 1)
            else:
                self._cursor += 1
                self._moved()
        elif key == "space":
            event.stop()
            self._select()
        elif key == "enter":
            self._select()  # also bubble -> a dialog may confirm/close on Enter
        elif self._cursor == self._custom_row:
            self._edit_custom(event)

    def _edit_custom(self, event: events.Key) -> None:
        """The highlighted custom row is a tiny inline text field; typing it also
        selects it. ←/→ move the caret here (not navigate)."""
        key = event.key
        if event.is_printable and event.character:
            event.stop()
            self._custom_text = (
                self._custom_text[: self._pos] + event.character + self._custom_text[self._pos :]
            )
            self._pos += 1
            self._sel = self._custom_row
            self._changed()
        elif key == "backspace" and self._pos > 0:
            event.stop()
            self._custom_text = self._custom_text[: self._pos - 1] + self._custom_text[self._pos :]
            self._pos -= 1
            self._sel = self._custom_row
            self._changed()
        elif key == "left":
            event.stop()
            self._pos = max(0, self._pos - 1)
            self.refresh()
        elif key == "right":
            event.stop()
            self._pos = min(len(self._custom_text), self._pos + 1)
            self.refresh()

    def _select(self) -> None:
        self._sel = self._cursor
        if self._sel == self._custom_row:
            self._pos = len(self._custom_text)
        self._changed()

    def _changed(self) -> None:
        self.refresh(layout=True)
        self.post_message(self.Changed(self))

    def _moved(self) -> None:
        # Highlight moved (no selection change); keep it in view for long lists.
        self.refresh()
        _scroll_row_into_view(self, self._cursor)

    def on_click(self, event: events.Click) -> None:
        # offset.y is from the widget region, which includes any padding-top
        # (.edit-gap) -- subtract it. A click highlights AND selects.
        row = int(event.offset.y) - self.styles.padding.top
        if 0 <= row < self._row_count:
            self._cursor = row
            self.focus()
            self._select()

    def on_mouse_move(self, event: events.MouseMove) -> None:
        # Track the mouse row so render() can highlight it (refresh only on a
        # change -- mouse moves fire per cell).
        row = int(event.offset.y) - self.styles.padding.top
        row = row if 0 <= row < self._row_count else -1
        if row != self._hover:
            self._hover = row
            self.refresh()

    def on_leave(self, event: events.Leave) -> None:
        if self._hover != -1:
            self._hover = -1
            self.refresh()


class TypeaheadField(Widget, can_focus=True):
    """A type-to-narrow picker for big lists (e.g. model ids): an editable text
    line plus, while focused, the top matching suggestions. Type to narrow; ↓
    highlights a suggestion (↑ back); Enter saves the current value -- the
    highlighted suggestion, or the typed text if none. Hands off ↑/↓ at its edges
    like ChoiceField. `value` is whatever would be saved. Suggestions can be
    swapped in later (e.g. a background fetch) via `set_suggestions`."""

    MAX_SHOWN = 8
    DEFAULT_CSS = """
    TypeaheadField { height: auto; width: 1fr; background: $panel; }
    """

    class Changed(Message):
        def __init__(self, field: TypeaheadField) -> None:
            self.field = field
            super().__init__()

    def __init__(
        self,
        current: str,
        suggestions: list[str],
        *,
        id: str | None = None,
        classes: str | None = None,
    ) -> None:
        super().__init__(id=id, classes=classes)
        self._text = current
        self._cursor = len(current)
        self._all = list(suggestions)
        self._index = -1  # -1 == editing the text; >=0 == a highlighted match
        # "Fresh" = still showing the current value, untouched: list ALL
        # suggestions (so ↓ browses everything) and let the first keystroke
        # replace it (you're searching for a new value, not appending to it).
        self._fresh = bool(current)

    def set_suggestions(self, suggestions: list[str]) -> None:
        matches = self._matches
        highlighted = matches[self._index] if 0 <= self._index < len(matches) else None
        self._all = list(suggestions)
        if highlighted is not None:
            matches = self._matches
            self._index = matches.index(highlighted) if highlighted in matches else -1
        if self.is_mounted:
            self.refresh(layout=True)

    def on_focus(self) -> None:
        self.refresh(layout=True)  # grow to show the suggestion list

    def on_blur(self) -> None:
        self.refresh(layout=True)  # shrink back to the single text line

    @property
    def _matches(self) -> list[str]:
        q = "" if self._fresh else self._text.strip().lower()
        if not q:
            shown = self._all
        else:
            starts = [m for m in self._all if m.lower().startswith(q)]
            rest = [m for m in self._all if q in m.lower() and not m.lower().startswith(q)]
            shown = starts + rest
        return shown[: self.MAX_SHOWN]

    @property
    def value(self) -> str:
        matches = self._matches
        if 0 <= self._index < len(matches):
            return matches[self._index]
        return self._text

    def render(self) -> Text:
        focused = self.has_focus
        width = max(self.size.width, 1)
        try:
            bar = _selection_bar(self.app.current_theme.primary)
        except Exception:  # pragma: no cover - defensive
            bar = "bold reverse"
        out = Text()
        # The editable text line (the field's $panel background, set in CSS,
        # gives it the input affordance; rich can't parse $-vars here).
        editing = focused and self._index < 0
        text = Text(self._text) if self._text else Text("type to search…", style="dim")
        if editing:
            text.append("▌")
        out.append_text(text)
        if focused:
            matches = self._matches
            q = "" if self._fresh else self._text.strip().lower()
            total = sum(1 for m in self._all if q in m.lower())
            for i, m in enumerate(matches):
                out.append("\n")
                # One row per match even for long entries (history search offers
                # whole steers): the value stays intact, only the display clips.
                row = Text(f"  {m}", no_wrap=True, overflow="ellipsis")
                row.pad_right(max(0, width - row.cell_len))
                if i == self._index:
                    row.stylize(bar)
                out.append_text(row)
            if total > len(matches):
                out.append("\n")
                out.append(f"  +{total - len(matches)} more, keep typing", style="dim")
            elif not matches:
                out.append("\n")
                out.append("  (no matches; saved as typed)", style="dim")
        return out

    def _moved(self) -> None:
        # layout=True: the visible match count changes, so re-measure the height.
        self.refresh(layout=True)
        self.call_after_refresh(_scroll_row_into_view, self, max(self._index + 1, 0))
        self.post_message(self.Changed(self))

    def on_key(self, event: events.Key) -> None:
        key = event.key
        if key == "down":
            event.stop()
            if self._index < len(self._matches) - 1:
                self._index += 1
                self._moved()
            else:
                focus_neighbor(self, 1)
        elif key == "up":
            event.stop()
            if self._index >= 0:
                self._index -= 1
                self._moved()
            else:
                focus_neighbor(self, -1)
        elif key == "space" and self._index >= 0:
            event.stop()
            self._accept()  # lock in the highlighted match
        elif key == "enter":
            self._accept()  # also bubble -> a dialog may confirm on Enter
        else:
            self._edit(event)

    def _accept(self) -> None:
        """Commit the highlighted match into the text line (value as-typed)."""
        matches = self._matches
        if 0 <= self._index < len(matches):
            self._text = matches[self._index]
            self._cursor = len(self._text)
            self._fresh = False
            self._index = -1
            self.refresh(layout=True)
            self.post_message(self.Changed(self))

    def _edit(self, event: events.Key) -> None:
        """Text-line editing: backspace / ←→ / typing (the first keystroke
        replaces the current value -- you're searching, not appending)."""
        key = event.key
        if key == "left":
            event.stop()
            self._fresh = False
            self._cursor = max(0, self._cursor - 1)
            self.refresh()
        elif key == "right":
            event.stop()
            self._fresh = False
            self._cursor = min(len(self._text), self._cursor + 1)
            self.refresh()
        elif key == "backspace":
            event.stop()
            self._fresh = False
            if self._cursor > 0:
                self._text = self._text[: self._cursor - 1] + self._text[self._cursor :]
                self._cursor -= 1
                self._index = -1
                self._moved()
        elif event.is_printable and event.character:
            event.stop()
            if self._fresh:
                self._text = ""
                self._cursor = 0
                self._fresh = False
            self._text = self._text[: self._cursor] + event.character + self._text[self._cursor :]
            self._cursor += 1
            self._index = -1
            self._moved()

    def on_click(self, event: events.Click) -> None:
        row = int(event.offset.y) - self.styles.padding.top  # account for padding-top
        matches = self._matches if self.has_focus else []
        if row == 0:
            self.focus()
        elif 1 <= row <= len(matches):
            self._index = row - 1
            self.focus()
            self._moved()


class ActionItem(Static):
    """A flat, focusable, clickable action label (Save / Unset / Cancel) -- the
    natural terminal equivalent of a button. Enter or click activates it; the
    dialog moves ←/→ between items. (Textual's Button assumes a 3-row box and
    won't render its label flat at height 1.)"""

    can_focus = True

    class Activated(Message):
        def __init__(self, action: str) -> None:
            self.action = action
            super().__init__()

    def __init__(self, label: str, action: str) -> None:
        super().__init__(label, classes="action")
        self._action = action

    def on_click(self) -> None:
        self.post_message(self.Activated(self._action))

    def on_key(self, event: events.Key) -> None:
        if event.key == "enter":
            event.stop()
            self.post_message(self.Activated(self._action))


# Shared CSS for the flat actions + inline inputs + chooser, so every form-style
# dialog includes one definition (no per-modal drift).
FORM_CSS = """
.edit-gap { height: auto; padding-top: 1; }
.edit-label { color: $text-muted; padding-top: 1; }
/* $boost resolves to transparent in every theme, so a field filled with it was
   invisible. Use $panel (a raised neutral that adapts light<->dark) for resting
   fields, and let focus tint primary. Flat actions stay text-only until hover. */
.edit-input { border: none; background: $panel; height: 1; padding: 0 1; }
.edit-input:focus { background: $primary 25%; }
ChoiceField { padding: 0 1; }
.action {
    width: auto; height: 1; padding: 0 2; margin-right: 1;
    background: transparent; color: $accent;
}
.action:focus { background: $primary; color: $text; text-style: bold; }
.action:hover { background: $primary 30%; }  /* transparent tint, like menu hover */
"""
