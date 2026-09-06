# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""TUI theming: the branded themes (`agent6-dark` / `agent6-light`) plus
four extra built-ins (`alice`, `snow`, `rose`, `grimm`), a
live-previewing picker (every registered theme, sorted alphabetically) reachable
from the View menu, and the wiring that loads the saved theme on startup and
persists any change.

Design: keep one quiet accent for focus and a calm, low-contrast resting state
(the lazygit/openapi-tui feel). All widget CSS across the TUI already uses
Textual theme variables ($primary, $accent, $surface, $panel, $text…), so
switching the theme re-skins everything for free — this module only chooses the
palettes and remembers the choice (in `ui.toml`, never the agent config).
"""

from __future__ import annotations

from math import ceil
from typing import Any, ClassVar, cast

try:
    from rich.color import Color
    from rich.segment import Segment, Segments
    from rich.style import Style
    from rich.text import Text
    from textual import events, on
    from textual.app import App, ComposeResult
    from textual.binding import Binding
    from textual.containers import Vertical, VerticalScroll
    from textual.notifications import SeverityLevel
    from textual.screen import ModalScreen
    from textual.scrollbar import ScrollBar, ScrollBarRender
    from textual.theme import Theme
    from textual.widgets import Static
except ImportError as e:  # pragma: no cover - clear runtime message
    raise SystemExit(
        "The TUI theme support needs textual, a required dependency; reinstall agent6."
    ) from e

from agent6.ui.tui.clipboard import mux_passthrough
from agent6.ui.tui.settings import DEFAULT_THEME, get_theme, save_theme
from agent6.ui.tui.widgets import FORM_CSS, ChoiceField
from agent6.viewmodel.format import StatusLevel, status_level

# Branded defaults: a deep, low-saturation dark and a soft light, both with a
# green focus accent over a blue selection primary.
AGENT6_DARK = Theme(
    name="agent6-dark",
    primary="#7AA2F7",  # selection / cursor / resting card borders
    secondary="#9ECE6A",  # green; no agent6 widget styles it directly (textual palette slot)
    accent="#06F5F3",  # focus borders, button/action text, key hints -- a vivid neon cyan
    foreground="#C0CAF5",
    # Near-black teal (the cyan brand's neutral): screen < card < panel, so
    # tables/panels read as raised surfaces over an almost-black background.
    background="#0f1414",
    surface="#161c1c",
    panel="#1e2626",
    success="#9ECE6A",
    warning="#E0AF68",
    error="#F7768E",
    dark=True,
    # The footer the baseline had: warm amber keys + neutral labels (reads more
    # "modern" than green keys on lavender text).
    variables={
        "footer-key-foreground": "#FFA62B",
        "footer-foreground": "#E0E0E0",
        "footer-description-foreground": "#E0E0E0",
        # Scrollbar tracks default to near-black (#000002); match the surface so the
        # track meshes with its panel (only the thumb shows) instead of a black gap.
        "scrollbar-background": "#161c1c",  # == surface
        "scrollbar-background-hover": "#161c1c",
        "scrollbar-background-active": "#161c1c",
        "scrollbar-corner-color": "#161c1c",
    },
)

AGENT6_LIGHT = Theme(
    name="agent6-light",
    primary="#2E5BA8",
    secondary="#1E6FA8",
    accent="#4C7A2F",
    foreground="#2A2E3F",
    background="#F4F5F8",
    surface="#EAECF2",
    panel="#DEE1EA",
    success="#4C7A2F",
    warning="#9A6E00",
    error="#C0392B",
    dark=False,
    variables={
        "footer-key-foreground": "#C2410C",  # warm orange keys, readable on light
        "scrollbar-background": "#EAECF2",  # == surface, so tracks mesh (see agent6-dark)
        "scrollbar-background-hover": "#EAECF2",
        "scrollbar-background-active": "#EAECF2",
        "scrollbar-corner-color": "#EAECF2",
    },
)

# Extra built-ins: two light storybook palettes (alice, snow) and two dark
# (rose, grimm).
ALICE = Theme(
    name="alice",
    primary="#D3A129",
    secondary="#3B78D0",
    accent="#6F87A8",
    foreground="#25272C",
    background="#FFFDF7",
    surface="#FFF7E6",
    panel="#F4EBCF",
    success="#2E7D57",
    warning="#B77900",
    error="#B83F48",
    dark=False,
    variables={
        "border": "#B77900",
        "border-blurred": "#D8CBA7",
        "footer-key-foreground": "#3B78D0",
        "input-selection-background": "#3B78D0 25%",
        "block-cursor-background": "#D3A129",
        "block-cursor-foreground": "#25272C",
    },
)

SNOW = Theme(
    name="snow",
    primary="#4D91B2",
    secondary="#315D8C",
    accent="#7CC6D8",
    foreground="#152733",
    background="#F6FBFD",
    surface="#EAF4F8",
    panel="#D9EAF1",
    success="#3B8C73",
    warning="#A96E21",
    error="#B8444A",
    dark=False,
    variables={
        "border": "#315D8C",
        "border-blurred": "#B4CCD8",
        "footer-key-foreground": "#315D8C",
        "input-selection-background": "#7CC6D8 35%",
        "block-cursor-background": "#315D8C",
        "block-cursor-foreground": "#F6FBFD",
    },
)

ROSE = Theme(
    name="rose",
    primary="#D4494F",
    secondary="#58AE9E",
    accent="#4E86C4",
    foreground="#E8ECEF",
    background="#171A1E",
    surface="#20252A",
    panel="#2A3036",
    success="#58AE9E",
    warning="#D1A33B",
    error="#E2555A",
    dark=True,
    variables={
        "border": "#D4494F",
        "border-blurred": "#465057",
        "footer-key-foreground": "#D1A33B",
        "input-selection-background": "#4E86C4 35%",
        "block-cursor-background": "#58AE9E",
        "block-cursor-foreground": "#171A1E",
    },
)

GRIMM = Theme(
    name="grimm",
    primary="#C34D55",
    secondary="#D2A33F",
    accent="#8E5A78",
    foreground="#F2E9E4",
    background="#160F13",
    surface="#24161C",
    panel="#311C24",
    success="#4C936B",
    warning="#D2A33F",
    error="#E45A62",
    dark=True,
    variables={
        "border": "#C34D55",
        "border-blurred": "#59343F",
        "footer-key-foreground": "#D2A33F",
        "input-selection-background": "#C34D55 35%",
        "block-cursor-background": "#D2A33F",
        "block-cursor-foreground": "#160F13",
    },
)

# Restyle textual's built-in command palette (Ctrl+P) to match our dialogs: a
# rounded $accent-framed $surface card, not the default flat $panel-darken box with
# black keyline borders. Add to an App's CSS (the palette is pushed on the App).
# Targets textual-internal ids (#--input etc.), so revisit if textual changes them.
PALETTE_CSS = """
CommandPalette > Vertical { background: $surface; border: round $accent; }
CommandPalette #--input { border: none; background: $panel; }
CommandPalette #--input.--list-visible { border: none; }
"""

# The upstream render_bar signature's defaults, as module singletons (B008).
_SCROLLBAR_BACK = Color.parse("#555555")
_SCROLLBAR_BAR = Color.parse("bright_magenta")


class ThinScrollBarRender(ScrollBarRender):
    """Horizontal scrollbar thumbs at HALF cell height: a terminal cell is about
    twice as tall as it is wide, so textual's full-cell horizontal thumb reads
    twice as heavy as a one-cell-wide vertical bar. The thumb body is a lower
    half-block band with quadrant end caps for half-cell granularity; vertical
    bars keep the default rendering."""

    @classmethod
    def render_bar(
        cls,
        size: int = 25,
        virtual_size: float = 50,
        window_size: float = 20,
        position: float = 0,
        thickness: int = 1,
        vertical: bool = True,
        back_color: Color = _SCROLLBAR_BACK,
        bar_color: Color = _SCROLLBAR_BAR,
    ) -> Segments:
        if vertical or not (window_size and size and virtual_size and size != virtual_size):
            return super().render_bar(
                size=size,
                virtual_size=virtual_size,
                window_size=window_size,
                position=position,
                thickness=thickness,
                vertical=vertical,
                back_color=back_color,
                bar_color=bar_color,
            )
        bar_ratio = virtual_size / size
        thumb_size = max(1.0, window_size / bar_ratio)
        position_ratio = position / (virtual_size - window_size)
        start = int((size - thumb_size) * position_ratio * 2)  # half-cell units
        end = start + max(2, ceil(thumb_size * 2))  # the thumb spans >= one cell
        before = {"@mouse.down": "scroll_up"}
        after = {"@mouse.down": "scroll_down"}
        grab = {"@mouse.down": "grab"}
        segments: list[Segment] = []
        for cell in range(int(size)):
            left, right = 2 * cell, 2 * cell + 1
            left_in = start <= left < end
            right_in = start <= right < end
            if left_in or right_in:
                glyph = "▄" if (left_in and right_in) else ("▗" if right_in else "▖")
                segments.append(
                    Segment(glyph, Style(color=bar_color, bgcolor=back_color, meta=grab))
                )
            else:
                meta = before if right < start else after
                segments.append(Segment(" ", Style(bgcolor=back_color, meta=meta)))
        return Segments([*segments, Segment.line()] * thickness, new_lines=False)


# The Rich style for each `viewmodel.format.status_level`; the CLI's SGR map
# and the web's pill classes are the sibling palettes.
STATUS_LEVEL_STYLE: dict[StatusLevel, str] = {
    "ok": "green",
    "info": "#b48ead",  # mauve, matching the web pill
    "active": "bold cyan",
    "warn": "yellow",
    "error": "bold red",
    "neutral": "",
}


def status_style(status: str) -> str:
    """The Rich style a status word renders in, on every TUI surface."""
    return STATUS_LEVEL_STYLE[status_level(status)]


class PlainNotify:
    """Mix into an App (before App in the bases): notifications carry text,
    never markup. Every toast here relays a message assembled elsewhere (a
    refusal naming `[git].dirty_tree`, a path, an error), and textual's
    default markup parse ate the bracketed parts ("set .dirty_tree=stash")."""

    def notify(
        self,
        message: str,
        *,
        title: str = "",
        severity: SeverityLevel = "information",
        timeout: float | None = None,
        markup: bool = False,
    ) -> None:
        App.notify(
            cast(App[Any], self),
            message,
            title=title,
            severity=severity,
            timeout=timeout,
            markup=markup,
        )


class MuxPointerShapes:
    """Mix into an App (before App in the bases): re-emits the kitty
    pointer-shape OSC (`ESC ] 22 ; <shape> BEL`) wrapped for tmux/screen
    passthrough. textual writes it bare, which a multiplexer swallows -- the
    same lesson as bare OSC 52 copy -- so the I-beam over text never reached
    the outer terminal under byobu."""

    def _set_pointer_shape(self, shape: str) -> None:
        driver = getattr(self, "_driver", None)
        if driver is not None:
            driver.write(mux_passthrough(f"\x1b]22;{shape}\x07"))


def setup_theme(app: App[Any]) -> None:
    """Register the built-in themes, apply the saved one, and persist changes.

    Call from `App.on_mount`. Subscribing to `theme_changed_signal` means
    EVERY path that changes the theme — the View>Theme picker, the built-in
    Ctrl+P "change theme" palette — is remembered, with no extra wiring.
    Also installs the half-height horizontal scrollbar renderer (a class-level
    hook, so one assignment restyles every bar in the process)."""
    ScrollBar.renderer = ThinScrollBarRender
    for theme in (AGENT6_DARK, AGENT6_LIGHT, ALICE, SNOW, ROSE, GRIMM):
        if theme.name not in app.available_themes:
            app.register_theme(theme)
    wanted = get_theme()
    app.theme = wanted if wanted in app.available_themes else DEFAULT_THEME
    app.theme_changed_signal.subscribe(app, lambda theme: save_theme(theme.name))


def open_theme_picker(app: App[Any]) -> None:
    """Push the theme picker (the View>Theme handler)."""
    app.push_screen(ThemePicker())


class ThemePicker(ModalScreen[None]):
    """A small, live-previewing theme chooser: the same `[x]`/`[ ]` chooser
    the config dialogs use. Arrow through to preview; the previewed theme is kept
    on close (Enter or Esc both just dismiss). The choice is persisted by
    `setup_theme`'s signal hook, so nothing here writes to disk directly."""

    BINDINGS: ClassVar = [
        Binding("escape", "cancel", "Cancel"),
        Binding("enter", "confirm", "Use theme"),
    ]
    CSS = (
        FORM_CSS
        + """
    ThemePicker { align: center middle; }
    #theme-box {
        width: 44; height: auto; max-height: 90%;
        border: round $accent; padding: 1 2; background: $surface;
    }
    #theme-title { text-style: bold; }
    /* The list scrolls (all themes) while the title + hint stay put. */
    #theme-scroll { height: auto; max-height: 16; scrollbar-size-vertical: 1; }
    #theme-hint { color: $text-muted; padding-top: 1; }
    """
    )

    def on_mount(self) -> None:
        # Focus without auto-scroll so the list opens at the top (focusing a list
        # taller than the dialog would otherwise scroll it).
        self.query_one(ChoiceField).focus(scroll_visible=False)

    def compose(self) -> ComposeResult:
        current = self.app.theme
        # Every registered theme, sorted alphabetically (includes ansi-dark/-light,
        # the "transparent" terminal-native options). The active one is guaranteed
        # present, so the chooser always opens on a real selection.
        names = sorted(self.app.available_themes)
        if current not in names:
            names.insert(0, current)
        with Vertical(id="theme-box"):
            yield Static("Theme", id="theme-title")
            # Just the scrollable list -- no button below (it added a cross-scroll
            # focus stop). Close with Esc or a click outside (handled below).
            with VerticalScroll(id="theme-scroll"):
                yield ChoiceField(tuple(names), current, id="theme-list")
            # Two balanced lines: the 44-wide box would wrap one line mid-phrase.
            yield Static(
                Text("↑↓ highlight · Space select\nEsc or click outside closes", style="dim"),
                id="theme-hint",
            )

    @on(ChoiceField.Changed)
    def _preview(self, event: ChoiceField.Changed) -> None:
        self.app.theme = event.field.value  # apply the selected theme (live)

    def action_confirm(self) -> None:
        self.dismiss(None)  # Enter: keep the applied theme + close

    def action_cancel(self) -> None:
        self.dismiss(None)  # Esc: just close, keeping whatever was previewed

    def on_click(self, event: events.Click) -> None:
        # Click on the backdrop (outside the dialog) = close, like Esc. A mouse +
        # key-swallowing-terminal alternative to Esc.
        if event.widget is self:
            self.action_cancel()
