# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The TUI config viewer/editor — a thin renderer over the shared config
view-model (`viewmodel.config_view.build_config_view`) and the shared edit
path (`config.write.set_config_value` / `unset_config_value`). All config
logic lives in those layers, so this page and the web editor never drift.

Discoverability is driven by ONE action registry (:data:`CONFIG_ACTIONS`): the
same list generates the on-screen action bar (clickable + keyboard-navigable
buttons), the key bindings shown in the footer, the help/keys overlay, and the
command-palette entries.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

try:
    from rich.markup import escape
    from rich.text import Text
    from textual import events, on
    from textual.app import ComposeResult
    from textual.binding import Binding
    from textual.containers import Horizontal, Vertical, VerticalScroll
    from textual.geometry import Region
    from textual.screen import ModalScreen, Screen
    from textual.widget import Widget
    from textual.widgets import (
        Collapsible,
        DataTable,
        Footer,
        Input,
        Static,
    )
except ImportError as e:  # pragma: no cover - clear runtime message
    raise SystemExit("The config page needs textual: pip install 'agent6[tui]'") from e

from agent6.app.confine import resolved_config_values
from agent6.config import ConfigError
from agent6.config.io import ConfigLeafValue, format_toml_value
from agent6.config.layer import EffectiveConfig, load_effective
from agent6.config.write import (
    PROVIDER_DEFAULTS,
    provider_choices,
    set_config_table,
    set_config_value,
    unset_config_value,
)
from agent6.errors import OperatorError
from agent6.models.cache import cached_models
from agent6.models.choices import config_value_choices, model_role_provider
from agent6.ui.tui.menubar import Menu, MenuBar, MenuItem, menu_bindings
from agent6.ui.tui.screen_chrome import (
    MenuCommands,
    PaletteCommand,
    ScreenChrome,
    menu_palette_commands,
)
from agent6.ui.tui.widgets import (
    FORM_CSS,
    ActionItem,
    ChoiceField,
    TypeaheadField,
    focus_neighbor,
)
from agent6.viewmodel.config_view import (
    ConfigSetting,
    ConfigView,
    build_config_view,
    display_value,
    format_value,
    plain_description,
)


@dataclass(frozen=True, slots=True)
class Action:
    """A user action, reachable three ways from one definition: a labelled
    button (mouse + Tab/Enter), an optional key binding (shown in the footer),
    and a command-palette entry. The help overlay lists them all."""

    id: str
    label: str
    description: str
    key: str | None = None


# The single source of truth for every Config-page action.
CONFIG_ACTIONS: tuple[Action, ...] = (
    Action("search", "Filter", "Filter settings by name", key="/"),
    Action("toggle_modified", "Modified only", "Show only settings a config layer set", key="m"),
    Action("edit", "Edit", "Edit the selected setting (dropdown for choices)", key="e"),
    Action("add_provider", "Add provider…", "Add a [providers.<name>] entry via a form", key="a"),
    # `r` is a harmless Refresh here (re-read config), matching `r`=Refresh on
    # the home hub; Reset (which UNSETS a setting) stays off `r` and lives on
    # `d` (default). Label is "Refresh" (not "Reload") + "Help" (not
    # "Help / keys") to match the home/run footers.
    Action("reset", "Reset", "Reset the selected setting to its default (unset)", key="d"),
    Action("reload", "Refresh", "Re-read config from disk", key="r"),
    # No key (View-menu / palette only, like the home hub): key=None is skipped by
    # the BINDINGS comprehension so it adds no footer binding, but palette_commands
    # still lists it -- so the live-preview Theme… picker stays reachable from the
    # config Ctrl+P palette (the built-in "Theme" is filtered out app-wide).
    Action("choose_theme", "Theme…", "Choose a colour theme", key=None),
    Action("help", "Help", "Show all actions and shortcuts", key="question_mark"),
    Action("close", "Back", "Back to the hub", key="escape"),
)


# Fixed (label, width) columns shared by the single pinned header and every
# section table, so all the section tables align under that one header.
_COLUMNS: tuple[tuple[str, int], ...] = (("setting", 26), ("value", 28), ("source", 12))


class _NavTable(DataTable[str]):
    """A settings table whose arrow nav doesn't dead-end at the section edge: at
    the bottom row, Down moves to the next section's header; at the top row, Up
    moves to this section's header. With the headers themselves arrow-navigable
    (see ConfigScreen._nav_sections), the whole config reads as one list."""

    def action_cursor_down(self) -> None:
        if self.cursor_row >= self.row_count - 1:
            self._hand_off(1)
        else:
            super().action_cursor_down()
            self._keep_in_view()

    def action_cursor_up(self) -> None:
        if self.cursor_row <= 0:
            self._hand_off(-1)
        else:
            super().action_cursor_up()
            self._keep_in_view()

    def _keep_in_view(self) -> None:
        if isinstance(self.screen, ConfigScreen):
            self.screen.scroll_focused_into_view()

    def _hand_off(self, direction: int) -> None:
        screen = self.screen
        section = self.id[4:] if self.id else ""
        if isinstance(screen, ConfigScreen):
            screen.nav_from_table(section, direction)


def _provider_preset_base_url(key: str) -> str:
    """The preset base_url for a `providers.<name>.base_url` setting whose
    `<name>` is a known provider (else ""). Lets the single-setting editor
    offer e.g. openrouter.ai instead of the generic api.openai.com that
    `config._default_base_url` fills in for any unset openai-format provider."""
    parts = key.split(".")
    if len(parts) == 3 and parts[0] == "providers" and parts[2] == "base_url":
        return PROVIDER_DEFAULTS.get(parts[1], {}).get("base_url", "")
    return ""


class _FormModal[ResultT](ModalScreen[ResultT]):
    """Shared form-modal keys. One vertical ↑↓ chain: ChoiceField hands off at
    its own edges; Inputs and action items release ↑↓ here so it flows through
    them too. ←/→ stay text-editing keys inside an Input and move between the
    flat action items; an activated ActionItem dispatches to `action_<id>`."""

    def on_key(self, event: events.Key) -> None:
        focused = self.focused
        if event.key in ("left", "right") and isinstance(focused, ActionItem):
            actions = list(self.query(ActionItem))
            step = 1 if event.key == "right" else -1
            actions[(actions.index(focused) + step) % len(actions)].focus()
            event.stop()
        elif event.key == "up" and isinstance(focused, (Input, ActionItem)):
            focus_neighbor(focused, -1)
            event.stop()
        elif event.key == "down" and isinstance(focused, Input):
            focus_neighbor(focused, 1)
            event.stop()

    @on(ActionItem.Activated)
    def _action_activated(self, event: ActionItem.Activated) -> None:
        getattr(self, f"action_{event.action}")()


class EditModal(_FormModal[tuple[str, str, bool] | None]):
    """Edit one setting with a natural terminal chooser: a [x]/[ ] list (↑↓ select
    as they move) for enum choices and bools -- with an inline "custom" row for
    values the choices don't cover -- a text box otherwise. The action row (Save
    · Unset → default · Cancel) is flat text, ←/→ navigable + clickable. Returns
    `(action, value, to_repo)` (action "save"/"unset") or None on cancel."""

    # No enter->save: Space/Enter on a chooser SELECTS the highlighted option, so
    # Enter must not also save (you'd save while just picking). Save via the Save
    # action (Enter on it / click). A plain text field still saves on Enter
    # (Input.Submitted) since there's nothing to "select" there.
    BINDINGS: ClassVar = [Binding("escape", "cancel", "Cancel")]
    CSS = (
        FORM_CSS
        + """
    EditModal { align: center middle; }
    #edit-box {
        width: 70; height: auto;
        border: round $accent; padding: 1 2; background: $surface;
    }
    #edit-title { text-style: bold; }
    #edit-description { padding-top: 1; color: $text-muted; }
    #edit-actions { padding-top: 1; height: auto; }
    """
    )

    def __init__(
        self,
        setting: ConfigSetting,
        *,
        typeahead: list[str] | None = None,
        fetch: Callable[[], list[str]] | None = None,
    ) -> None:
        super().__init__()
        self._setting = setting
        self._done = False
        # For big open lists (model ids): the cached suggestions to show now, and
        # an optional (blocking) fetch to refresh them from the live listing.
        self._typeahead = typeahead
        self._fetch = fetch

    def on_mount(self) -> None:
        self.query_one("#edit-value").focus()  # arrows work immediately
        if self._fetch is not None:
            self.run_worker(self._fetch_worker, thread=True)

    def _fetch_worker(self) -> None:
        models = self._fetch() if self._fetch else []
        if models:
            self.app.call_from_thread(self._apply_suggestions, models)

    def _apply_suggestions(self, models: list[str]) -> None:
        field = self.query("#edit-value").first()
        if isinstance(field, TypeaheadField):
            field.set_suggestions(models)

    def compose(self) -> ComposeResult:
        s = self._setting
        with VerticalScroll(id="edit-box"):
            yield Static(f"Edit {s.key}", id="edit-title")
            yield Static(
                Text(
                    f"type={s.py_type}  ·  default={format_value(s.default)}  ·  source={s.source}",
                    style="dim",
                )
            )
            if s.description:
                # What the leaf means, as the docs table and the web editor
                # say it; Text, never markup (a description names `[git]`).
                yield Static(Text(plain_description(s.description)), id="edit-description")
            # The DISPLAY formatter (_fmt) renders lists unquoted ([uv, run,
            # pytest]) -- friendly in the table, but not valid TOML, so an
            # untouched Save of a list/dict field failed revalidation ("Input
            # should be a valid tuple"). Prefill the edit box with the exact
            # inverse of parse_cli_value instead; scalars stay bare.
            raw = s.value if s.value is not None else s.default
            current = (
                format_toml_value(raw)
                if isinstance(raw, (list, tuple, dict))
                else format_value(raw)
            )
            if self._typeahead is not None:
                # Big open list (e.g. model ids): type to narrow over suggestions.
                yield TypeaheadField(
                    "" if s.value is None else current,
                    self._typeahead,
                    id="edit-value",
                    classes="edit-gap",
                )
            elif s.choices is not None:
                # Inline custom row: pick "custom" and type the value right there.
                yield ChoiceField(
                    tuple(s.choices),
                    current,
                    allow_custom=True,
                    id="edit-value",
                    classes="edit-gap",
                )
            elif s.py_type == "bool":
                yield ChoiceField(
                    ("true", "false"),
                    current if current in ("true", "false") else "false",
                    id="edit-value",
                    classes="edit-gap",
                )
            else:
                initial = "" if s.value is None else current
                # A known provider whose base_url is still the generic default
                # (api.openai.com, filled by _default_base_url): offer its preset
                # host so an unset openrouter/ollama is one Save from correct.
                if not s.modified and (preset_url := _provider_preset_base_url(s.key)):
                    initial = preset_url
                yield Input(
                    value=initial,
                    placeholder=str(format_value(s.default)),
                    id="edit-value",
                    classes="edit-input edit-gap",
                )
            yield Static("save to", classes="edit-label")
            yield ChoiceField(("global config", "repo config"), "global config", id="edit-target")
            with Horizontal(id="edit-actions"):
                yield ActionItem("Save", "save")
                yield ActionItem("Unset → default", "unset")
                yield ActionItem("Cancel", "cancel")
            yield Static(
                Text("↑↓ highlight · Space select · Tab field · Esc cancel", style="dim"),
                classes="edit-label",
            )

    def _new_value(self) -> str:
        w = self.query_one("#edit-value")
        if isinstance(w, (ChoiceField, TypeaheadField)):
            return w.value
        if isinstance(w, Input):
            return w.value
        return ""

    @on(Input.Submitted)
    def _input_submitted(self, event: Input.Submitted) -> None:
        focus_neighbor(event.input, 1)  # Enter in a text field advances (like Tab/↓)

    def action_save(self) -> None:
        if self._done:
            return  # one save (Enter may also reach an action item / Submitted)
        self._done = True
        to_repo = self.query_one("#edit-target", ChoiceField).index == 1
        self.dismiss(("save", self._new_value(), to_repo))

    def action_unset(self) -> None:
        if not self._done:
            self._done = True
            self.dismiss(("unset", "", False))

    def action_cancel(self) -> None:
        self.dismiss(None)

    def on_click(self, event: events.Click) -> None:
        if event.widget is self:  # click on the backdrop (outside the dialog) = cancel
            self.action_cancel()


class ProviderModal(_FormModal[None]):
    """Add a `[providers.<name>]` entry via a form instead of hand-editing a
    TOML dict: dropdowns for the fixed-choice fields (api_format, deployment,
    from the schema) and inputs for the free ones (name, base_url, api_key_env).
    base_url/auth default from the format+deployment when blank, so a minimal
    entry needs only a name + format. Writes + validates on Add (errors stay in
    the form to fix); the page reloads after."""

    # No enter->add (Space/Enter selects a chooser option); add via the Add action
    # or Enter in a text field (Input.Submitted).
    BINDINGS: ClassVar = [Binding("escape", "cancel", "Cancel")]
    CSS = (
        FORM_CSS
        + """
    ProviderModal { align: center middle; }
    #prov-box {
        width: 80; height: auto;
        border: round $accent; padding: 1 2; background: $surface;
    }
    #prov-title { text-style: bold; }
    #prov-actions { padding: 1 0 0 0; height: auto; }
    """
    )

    def __init__(self, repo_root: Path) -> None:
        super().__init__()
        self._repo = repo_root
        self._autofilled_baseurl = ""  # the base_url we last prefilled (never clobber a typed one)

    def on_mount(self) -> None:
        self.query_one("#prov-name", Input).focus()

    def compose(self) -> ComposeResult:
        choices = provider_choices()
        with VerticalScroll(id="prov-box"):
            yield Static("Add provider", id="prov-title")
            # Split at the sentence boundary: one line is wider than the box
            # (74 cells inside) and would word-wrap mid-phrase.
            yield Static(
                Text(
                    "A [providers.<name>] block.\n"
                    "base_url/auth default from the format + deployment when left blank.",
                    style="dim",
                )
            )
            yield Input(
                placeholder="name  (e.g. openrouter, my-azure)",
                id="prov-name",
                classes="edit-input edit-gap",
            )
            yield Static("api_format", classes="edit-label")
            yield ChoiceField(
                tuple(choices["api_format"]), choices["api_format"][0], id="prov-format"
            )
            yield Static("deployment", classes="edit-label")
            yield ChoiceField(
                tuple(choices["deployment"]), choices["deployment"][0], id="prov-deployment"
            )
            yield Static("base_url", classes="edit-label")
            yield Input(
                placeholder="blank = default for the format/deployment",
                id="prov-baseurl",
                classes="edit-input",
            )
            yield Static("api_key_env", classes="edit-label")
            yield Input(
                placeholder="blank = secrets.toml by provider name",
                id="prov-keyenv",
                classes="edit-input",
            )
            yield Static("save to", classes="edit-label")
            yield ChoiceField(("global config", "repo config"), "global config", id="prov-target")
            with Horizontal(id="prov-actions"):
                yield ActionItem("Add", "add")
                yield ActionItem("Cancel", "cancel")
            yield Static(
                Text("↑↓ highlight · Space select · Tab field · Esc cancel", style="dim"),
                classes="edit-label",
            )

    def _selected(self, widget_id: str, fallback: str) -> str:
        field = self.query_one(widget_id, ChoiceField)
        return field.value or fallback

    @on(Input.Submitted)
    def _input_submitted(self, event: Input.Submitted) -> None:
        focus_neighbor(event.input, 1)  # Enter in a text field advances (like Tab/↓)

    @on(Input.Changed, "#prov-name")
    def _prefill_from_preset(self, event: Input.Changed) -> None:
        """Typing a known provider name prefills its api_format + base_url (parity
        with `agent6 connect`) so e.g. 'openrouter' lands on openrouter.ai rather
        than the api.openai.com fallback in config._default_base_url. Visible and
        editable; only overwrites a base_url we ourselves autofilled (or a blank
        one), never a URL the user typed."""
        preset = PROVIDER_DEFAULTS.get(event.value.strip())
        if preset is None:
            return
        self.query_one("#prov-format", ChoiceField).select_value(preset["api_format"])
        baseurl = self.query_one("#prov-baseurl", Input)
        if baseurl.value in ("", self._autofilled_baseurl):
            self._autofilled_baseurl = preset.get("base_url", "")
            baseurl.value = self._autofilled_baseurl

    def action_add(self) -> None:
        name = self.query_one("#prov-name", Input).value.strip()
        if not name:
            self.notify("Enter a provider name.", severity="warning")
            return
        fields: dict[str, ConfigLeafValue] = {
            "api_format": self._selected("#prov-format", "anthropic")
        }
        dep = self._selected("#prov-deployment", "direct")
        if dep != "direct":
            fields["deployment"] = dep
        base = self.query_one("#prov-baseurl", Input).value.strip()
        if base:
            fields["base_url"] = base
        keyenv = self.query_one("#prov-keyenv", Input).value.strip()
        if keyenv:
            fields["api_key_env"] = keyenv
        to_repo = self.query_one("#prov-target", ChoiceField).index == 1
        try:
            err = set_config_table(self._repo, f"providers.{name}", fields, to_repo=to_repo)
        except OperatorError as exc:
            err = str(exc)  # an unwritable config file is a form error, not a crash
        if err:
            self.notify(f"Invalid: {err}", severity="error", timeout=8.0)
            return  # stay in the form so the user can fix it
        self.notify(f"Added provider '{name}'.")
        self.dismiss(None)

    def action_cancel(self) -> None:
        self.dismiss(None)

    def on_click(self, event: events.Click) -> None:
        if event.widget is self:  # click on the backdrop (outside the dialog) = cancel
            self.action_cancel()


class ConfigScreen(ScreenChrome, Screen[None]):
    """Full config viewer/editor: collapsible per-section tables, search, a
    modified-only filter, provenance, and edit/reset — all reachable by button,
    key, or the command palette."""

    CSS = """
    ConfigScreen { layers: base dropdown; background: $surface; }
    /* One slim row: inline filter (left) + count/modified tag (right). */
    #topbar { height: 1; padding: 0 1; }
    #search { width: 1fr; height: 1; border: none; background: transparent; padding: 0; }
    #search:focus { background: $primary 25%; }  /* $boost is transparent */
    #status { width: auto; height: 1; color: $text-muted; content-align: right middle; }
    /* Sections sit on a surface card; the panel-coloured bars (the one pinned
       column header + each section's title row) carry the structure -- the same
       colour as the top menu bar and footer. No zebra. */
    /* The header + sections share ONE rounded card (matching the home runs table
       and the dashboard panels); the border tracks focus via :focus-within. */
    #config-card { height: 1fr; border: round $primary; background: $surface; }
    #config-card:focus-within { border: round $accent; }
    #settings { height: 1fr; background: $surface; }
    .section-table { height: auto; margin: 0; background: transparent; }
    #status { height: auto; padding: 0 1; color: $text-muted; }
    /* The ONE pinned column header (a panel bar) -- section tables hide theirs. */
    #col-header { height: 1; background: $panel; }
    #col-header > .datatable--header {
        background: $panel; color: $foreground; text-style: bold;
    }
    /* A selection bar marks only the focused table's row. */
    .section-table > .datatable--cursor { background: transparent; color: $foreground; }
    .section-table:focus > .datatable--cursor {
        background: $primary 40%; color: $text; text-style: bold;
    }
    /* Flat sections; each title is a panel bar (matching the column header) that
       separates the sections. The title gets its OWN focus bar so arrow nav onto
       a header stays visible (otherwise focus there looks "lost"). */
    ConfigScreen Collapsible {
        background: transparent; border: none; padding: 0; margin: 0;
    }
    ConfigScreen CollapsibleTitle {
        padding: 0 1; color: $foreground; text-style: bold; background: $panel;
    }
    ConfigScreen CollapsibleTitle:focus { background: $primary 40%; color: $text; }
    ConfigScreen Collapsible > Contents { padding: 0; }
    """
    MENUS: ClassVar = (
        Menu(
            "Config",
            (
                MenuItem("Refresh", "reload", "r"),
                MenuItem("Back", "close", "Esc/q"),
                MenuItem("Quit", "quit", "ctrl+q"),
            ),
        ),
        Menu(
            "Edit",
            (
                MenuItem("Edit setting…", "edit", "e"),
                MenuItem("Add provider…", "add_provider", "a"),
                MenuItem("Reset to default", "reset", "d"),
            ),
        ),
        Menu(
            "View",
            (
                MenuItem("Filter", "search", "/"),
                MenuItem("Modified only", "toggle_modified", "m"),
                MenuItem("Theme…", "choose_theme"),
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
    # Footer order: page actions first, then the meta tail Help, Back, Menu --
    # same order as the home/run footers (the root hub shows Quit in that slot
    # instead, since it is the only screen that quits on q). Help + Back close out
    # CONFIG_ACTIONS; Menu is appended below.
    BINDINGS: ClassVar = (
        [
            Binding(
                a.key,
                a.id,
                a.label,
                show=a.id in {"search", "edit", "toggle_modified", "reload", "help", "close"},
                # Back responds to both Esc and q -- shown as one "Esc/q" footer entry.
                key_display="Esc/q" if a.id == "close" else None,
            )
            for a in CONFIG_ACTIONS
            if a.key is not None
        ]
        # Config is one level below the hub, so q (like Esc) backs out -- only the
        # root hub quits on q. (q is typeable in #search: the focused Input eats it
        # first.) Ctrl+Q is the app-wide hard quit; Quit is in the menu as ^Q.
        + [Binding("q", "close", "Back", show=False)]
        + menu_bindings(MENUS)
    )
    COMMANDS: ClassVar = Screen.COMMANDS | {MenuCommands}
    HELP_TITLE: ClassVar = "agent6 config — keys & actions"
    HELP_HINTS: ClassVar = ("Enter edits the selected setting",)

    def __init__(self, repo_root: Path, config_path: Path | None = None) -> None:
        super().__init__()
        self.repo_root = repo_root
        self.config_path = config_path
        self._eff: EffectiveConfig | None = None
        self._view: ConfigView | None = None
        self._table_rows: dict[str, list[ConfigSetting]] = {}  # per section, in row order
        self._modified_only = False

    def palette_commands(self) -> Iterator[PaletteCommand]:
        """The menu actions with CONFIG_ACTIONS' descriptions as their help
        (the menu title alone says less than the registry does)."""
        descriptions = {a.id: a.description for a in CONFIG_ACTIONS}
        for label, handler, menu_title in menu_palette_commands(self, self.MENUS):
            action = next((i.action for m in self.MENUS for i in m.items if i.label == label), "")
            yield label, handler, descriptions.get(action, menu_title)

    def compose(self) -> ComposeResult:
        # Load the view first so the (fixed) set of sections is known, then
        # compose one Collapsible+DataTable per section up front. Reloads only
        # repopulate rows -- the section structure never needs remounting.
        self._rebuild_view()
        yield MenuBar(self.MENUS)  # the top row: menus + "agent6 — <path>"
        # One slim row: the inline filter on the left, the count (+ "modified
        # only" tag) on the right. No separate pop-up search box.
        with Horizontal(id="topbar"):
            yield Input(placeholder="/  filter settings…", id="search")
            yield Static("", id="status")
        # ONE column header, pinned above the scroll (the per-section tables hide
        # theirs and share these fixed column widths, so everything lines up under
        # this single header instead of repeating "setting value source").
        with Vertical(id="config-card"):
            yield DataTable(id="col-header")
            with VerticalScroll(id="settings"):
                for section in self._sections():
                    table = _NavTable(id=f"tbl-{section}", classes="section-table")
                    yield Collapsible(
                        table, title=escape(f"[{section}]"), collapsed=False, id=f"sec-{section}"
                    )
        yield Footer()

    def _sections(self) -> tuple[str, ...]:
        return self._view.sections if self._view is not None else ()

    def _rebuild_view(self) -> None:
        eff = load_effective(self.repo_root, self.config_path)
        self._eff = eff
        self._view = build_config_view(eff, resolved=resolved_config_values(eff.config))

    def _reload(self) -> None:
        # Every interactive re-read funnels through here. An external hand-edit
        # can invalidate the on-disk config while the page is open; keep the
        # last-good view and point at the fix instead of crashing the TUI out
        # of the action handler (the hub's open guard documents that intent).
        try:
            self._rebuild_view()
        except ConfigError as exc:
            self.notify(
                f"config is invalid on disk; showing the last-loaded values."
                f" Run `agent6 config fix` (or edit the file). {exc}",
                severity="error",
                timeout=10.0,
            )
            return
        self._refresh()

    def on_mount(self) -> None:
        # The single pinned header carries the column labels; section tables hide
        # theirs but share the exact same fixed widths so they line up under it.
        header = self.query_one("#col-header", DataTable)
        header.show_cursor = False
        header.can_focus = False
        for label, width in _COLUMNS:
            header.add_column(label, width=width)
        for section in self._sections():
            table = self.query_one(f"#tbl-{section}", DataTable)
            table.cursor_type = "row"
            table.show_header = False
            for label, width in _COLUMNS:
                table.add_column(label, width=width)
        self._refresh()
        # Show "agent6 — config · <repo>" in the menu-bar title for this screen.
        self.app.sub_title = f"config · {self.repo_root}"
        # Focus the first SECTION table (a _NavTable), not the pinned col-header
        # (which is can_focus=False) -- opening config shouldn't look like a menu
        # is half-activated. Alt+letter still reaches the menu bar.
        tables = list(self.query(_NavTable))
        if tables:
            tables[0].focus()

    def _matches(self, s: ConfigSetting, query: str) -> bool:
        if self._modified_only and not s.modified:
            return False
        return query in s.key.lower() if query else True

    def _refresh(self) -> None:
        if self._view is None:
            return
        query = self.query_one("#search", Input).value.strip().lower()
        by_section: dict[str, list[ConfigSetting]] = {}
        for s in self._view.settings:
            if self._matches(s, query):
                by_section.setdefault(s.section, []).append(s)
        shown = 0
        for section in self._view.sections:
            rows = by_section.get(section, [])
            self._table_rows[section] = rows
            table = self.query_one(f"#tbl-{section}", DataTable)
            table.clear()
            for s in rows:
                leaf = s.key.split(".", 1)[1] if "." in s.key else s.key
                src = s.source + (" *" if s.modified else "")
                # Values/keys may carry brackets (lists, regexes) -> render as
                # Text so they are never parsed as Rich markup.
                table.add_row(Text(leaf), Text(display_value(s)), Text(src), key=s.key)
            # Pin each table to its full row count so it never scrolls internally --
            # only #settings scrolls. (DataTable's height:auto otherwise gets clamped
            # to the viewport in a short window, giving a table scrollbar AND the
            # config scrollbar: a double scrollbar.)
            table.styles.height = max(1, len(rows))
            self.query_one(f"#sec-{section}", Collapsible).display = bool(rows)
            shown += len(rows)
        flt = "   ·   modified only" if self._modified_only else ""
        self.query_one("#status", Static).update(f"{shown} settings{flt}")

    def _current_setting(self) -> ConfigSetting | None:
        focused = self.focused
        if isinstance(focused, DataTable) and focused.id and focused.id.startswith("tbl-"):
            section = focused.id[4:]
            rows = self._table_rows.get(section, [])
            row = focused.cursor_row
            if 0 <= row < len(rows):
                return rows[row]
        return None

    # --- continuous arrow nav across section headers + their rows -----------

    def _ordered_sections(self) -> list[str]:
        """Sections with a visible (non-empty) Collapsible, in display order."""
        return [s for s in self._sections() if self.query_one(f"#sec-{s}", Collapsible).display]

    def _section_has_rows(self, section: str) -> bool:
        col = self.query_one(f"#sec-{section}", Collapsible)
        return not col.collapsed and self.query_one(f"#tbl-{section}", _NavTable).row_count > 0

    def _focus_title(self, section: str) -> None:
        col = self.query_one(f"#sec-{section}", Collapsible)
        title = next(iter(col.query("CollapsibleTitle")), None)
        if title is not None:
            title.focus(scroll_visible=False)  # we do the (minimal) scroll ourselves
            self.scroll_focused_into_view(title)  # pass it: .focused updates async

    def _focus_table(self, section: str, *, top: bool) -> None:
        table = self.query_one(f"#tbl-{section}", _NavTable)
        table.move_cursor(row=0 if top else table.row_count - 1)
        table.focus(scroll_visible=False)
        self.scroll_focused_into_view(table)

    def scroll_focused_into_view(self, target: Widget | None = None) -> None:
        """Scroll #settings just enough to show *target* (the focused row -- a table
        cursor or a section header). Pass the target explicitly when you've just
        focused it: Widget.focus() updates self.focused asynchronously, so reading
        it here would scroll the OLD row. Textual's focus auto-scroll brings the
        whole section into view (jumps at edges); this scrolls a single row."""
        settings = self.query_one("#settings", VerticalScroll)
        focused = target if target is not None else self.focused
        if focused is None:
            return
        if isinstance(focused, _NavTable):
            screen_y = focused.region.y + focused.cursor_row  # header hidden -> row 0 at top
        elif isinstance(focused.parent, Collapsible):
            # The first VISIBLE section's header is the topmost row. scroll_to_region
            # won't pull that last row flush to the top (leaves it ~1 line off, so Up
            # off the first setting looked like it skipped the header) -- pin to home.
            first = next((c for c in self.query("#settings Collapsible") if c.display), None)
            if focused.parent is first:
                settings.scroll_home(animate=False)
                return
            screen_y = focused.region.y  # a focused CollapsibleTitle
        else:
            return
        content_y = screen_y - settings.region.y + settings.scroll_offset.y
        settings.scroll_to_region(Region(0, content_y, 1, 1), animate=False)

    def nav_from_table(self, section: str, direction: int) -> None:
        """A table edge handed off nav: Down -> next section's header; Up -> this
        section's own header (so the header sits between the two sections)."""
        order = self._ordered_sections()
        if section not in order:
            return
        i = order.index(section)
        if direction > 0:
            if i + 1 < len(order):
                self._focus_title(order[i + 1])
        else:
            self._focus_title(section)

    def _nav_from_title(self, section: str, direction: int) -> None:
        """Arrow on a section header: Down -> into its rows (or the next header
        if it's collapsed/empty); Up -> the previous section's rows (or header)."""
        order = self._ordered_sections()
        if section not in order:
            return
        i = order.index(section)
        if direction > 0:
            if self._section_has_rows(section):
                self._focus_table(section, top=True)
            elif i + 1 < len(order):
                self._focus_title(order[i + 1])
        elif i - 1 >= 0:
            prev = order[i - 1]
            if self._section_has_rows(prev):
                self._focus_table(prev, top=False)
            else:
                self._focus_title(prev)
        else:  # Up at the topmost header -> back to the filter box
            self.query_one("#search", Input).focus()

    def on_key(self, event: events.Key) -> None:
        focused = self.focused
        # Down steps out of the filter box into the settings (keeping the filter).
        if isinstance(focused, Input) and focused.id == "search":
            if event.key == "down":
                self._focus_first_setting()
                event.stop()
            return
        # On a focused section HEADER (a CollapsibleTitle under our #sec-* block),
        # Up/Down flow through the sections and Space toggles it (Enter already
        # does, built-in).
        parent = getattr(focused, "parent", None)
        if not (isinstance(parent, Collapsible) and parent.id and parent.id.startswith("sec-")):
            return
        if event.key in ("up", "down"):
            self._nav_from_title(parent.id[4:], 1 if event.key == "down" else -1)
            event.stop()
        elif event.key == "space":
            parent.collapsed = not parent.collapsed
            event.stop()

    # --- actions (one handler per registry entry; button + key both land here)

    def action_search(self) -> None:
        self.query_one("#search", Input).focus()  # the inline filter box

    def _focus_first_setting(self) -> None:
        """Focus the first visible section's first row (used to step out of the
        filter box into the settings)."""
        order = self._ordered_sections()
        if order:
            self._focus_table(order[0], top=True)
        else:
            tables = list(self.query(_NavTable))
            if tables:
                tables[0].focus()

    def _cancel_search(self) -> bool:
        """Esc in/with an active filter clears it and drops back to the settings;
        returns True so Esc backs out of the filter before it closes the page."""
        box = self.query_one("#search", Input)
        if not box.value and self.focused is not box:
            return False  # nothing to back out of -> Esc closes the page
        box.value = ""
        self._refresh()
        self._focus_first_setting()
        return True

    def action_toggle_modified(self) -> None:
        self._modified_only = not self._modified_only
        self._refresh()

    def action_reload(self) -> None:
        self._reload()
        self.notify("Config reloaded.")

    def action_quit(self) -> None:
        # The menu's "Quit" (^Q) quits the whole app. On a Screen `quit` isn't
        # built-in and doesn't bubble, so call exit() directly. (q backs out
        # instead -- see action_close; only the root hub quits on q.)
        self.app.exit()

    def action_close(self) -> None:
        # Esc backs out of an open search/filter first; only then leaves the page.
        if self._cancel_search():
            return
        self.dismiss(None)

    def action_add_provider(self) -> None:
        self.app.push_screen(ProviderModal(self.repo_root), lambda _: self._reload())

    def on_data_table_row_selected(self, _event: DataTable.RowSelected) -> None:
        self.action_edit()  # Enter / double-click a setting row edits it

    def action_edit(self) -> None:
        setting = self._current_setting()
        if setting is None:
            self.notify(
                "Select a setting first (click a row or Tab into a table).", severity="warning"
            )
            return

        def _done(result: tuple[str, str, bool] | None) -> None:
            if result is None:
                return
            action, raw, to_repo = result
            if action == "unset":
                self._unset(setting)
                return
            try:
                err = set_config_value(self.repo_root, setting.key, raw, to_repo=to_repo)
            except OperatorError as exc:
                err = str(exc)
            if err:
                self.notify(err, severity="error", timeout=8.0)
            else:
                self.notify(f"Set {setting.key}")
                self._reload()

        # A model-id field gets a type-to-narrow picker over the provider's
        # models (the cache now, refreshed live in a worker); everything else
        # the plain edit, with the view's choices as its picker.
        provider = model_role_provider(self._eff, setting.key) if self._eff is not None else None
        if provider is not None:
            repo, overlay, key = self.repo_root, self.config_path, setting.key
            modal = EditModal(
                setting,
                typeahead=cached_models(provider),
                fetch=lambda: config_value_choices(load_effective(repo, overlay), key),
            )
        else:
            modal = EditModal(setting)
        self.app.push_screen(modal, _done)

    def action_reset(self) -> None:
        setting = self._current_setting()
        if setting is None:
            self.notify("Select a setting first.", severity="warning")
            return
        self._unset(setting)

    def _unset(self, setting: ConfigSetting) -> None:
        """Reset *setting* to its default in the layer that set it, and say so."""
        # "Already at its default" is only truthful when it IS the default; a
        # leaf from any other layer (a preset's synthesized [presets.<name>], a
        # `--config` file) is modified but outside the two files an unset writes.
        if not setting.modified:
            self.notify(f"{setting.key} is already at its default.")
            return
        if setting.source not in ("global", "repo"):
            self.notify(
                f"{setting.key} comes from the {setting.source} layer;"
                " unset edits only the global and repo config.",
                severity="warning",
            )
            return
        try:
            err = unset_config_value(
                self.repo_root, setting.key, to_repo=setting.source == "repo"
            ).error
        except OperatorError as exc:
            err = str(exc)
        if err:
            self.notify(err, severity="error", timeout=8.0)
        else:
            self.notify(f"Reset {setting.key} to default")
            self._reload()

    @on(Input.Changed, "#search")
    def _on_search(self) -> None:
        self._refresh()

    @on(Input.Submitted, "#search")
    def _on_search_submit(self) -> None:
        self._focus_first_setting()  # Enter steps into the settings (keeps filter)
