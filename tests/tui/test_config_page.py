# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Headless drive of the TUI config page (textual Pilot).

Covers the bits a human would otherwise have to eyeball: the page loads + renders
every section's settings, search narrows them, the modified-only filter narrows
to overridden settings, and Help opens — all over the shared config view-model.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

import pytest
from textual.app import App
from textual.widgets import DataTable, Input, OptionList

from agent6.config import OpenAIProviderEntry
from agent6.config.layer import load_effective
from agent6.ui.tui.config_page import ConfigScreen, EditModal
from agent6.ui.tui.menubar import HelpScreen, MenuBar
from agent6.viewmodel.config_view import build_config_view

_GLOBAL = """\
[providers.anthropic]
api_format = "anthropic"

[models.worker]
provider = "anthropic"
model = "claude-sonnet-4-5"

[sandbox]
run_commands = "yes"
"""


class _Host(App[None]):
    def __init__(self, repo_root: Path, config_path: Path | None = None) -> None:
        super().__init__()
        self._repo = repo_root
        self._config_path = config_path

    def on_mount(self) -> None:
        self.push_screen(ConfigScreen(self._repo, self._config_path))


@pytest.fixture
def repo(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    gdir = tmp_path / "g"
    gdir.mkdir()
    (gdir / "config.toml").write_text(_GLOBAL, encoding="utf-8")
    monkeypatch.setenv("AGENT6_CONFIG_HOME", str(gdir))
    monkeypatch.setenv("AGENT6_STATE_HOME", str(tmp_path / "state"))
    monkeypatch.setenv("AGENT6_CACHE_HOME", str(tmp_path / "cache"))
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    return repo_root


def _row_total(screen: ConfigScreen) -> int:
    return sum(t.row_count for t in screen.query(DataTable))


def test_config_page_view_search_filter_help(repo: Path) -> None:
    async def scenario() -> None:
        app = _Host(repo)
        async with app.run_test() as pilot:
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, ConfigScreen)

            # Every section renders; the whole effective config is shown.
            total = _row_total(screen)
            assert total > 10
            # run_commands is set in the (global) config -> present + sourced.
            sandbox = screen.query_one("#tbl-sandbox", DataTable)
            assert any(
                "run_commands" in str(sandbox.get_row_at(r)[0]) for r in range(sandbox.row_count)
            )

            # Search narrows to matching keys.
            screen.query_one("#search", Input).value = "run_commands"
            screen._refresh()  # pyright: ignore[reportPrivateUsage]
            await pilot.pause()
            narrowed = _row_total(screen)
            assert 0 < narrowed < total

            # Modified-only filter: clear search, show only overridden settings.
            screen.query_one("#search", Input).value = ""
            screen.action_toggle_modified()
            await pilot.pause()
            modified = _row_total(screen)
            assert 0 < modified < total  # fewer than everything, but >0 (run_commands)

            # Help overlay opens from its action (also a button + ? key).
            screen.action_help()
            await pilot.pause()
            assert isinstance(app.screen, HelpScreen)

    asyncio.run(scenario())


def test_config_page_adaptive_value_shown(repo: Path) -> None:
    async def scenario() -> None:
        app = _Host(repo)
        async with app.run_test() as pilot:
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, ConfigScreen)
            ctx = screen.query_one("#tbl-context", DataTable)
            # adaptive compaction (worker=claude-sonnet-4-5 -> 200k window) shows
            # its resolved number tagged "(adaptive)", not "(unset)".
            cells = [str(ctx.get_row_at(r)[1]) for r in range(ctx.row_count)]
            assert any("(adaptive)" in c for c in cells)

    asyncio.run(scenario())


def test_config_page_edit_persists(repo: Path) -> None:
    """Select a row -> Edit -> [x]/[ ] chooser -> arrow to a new value -> Save
    writes through the shared edit path. The whole edit ask, end to end."""

    async def scenario() -> None:
        from agent6.ui.tui.config_page import ChoiceField

        app = _Host(repo)
        async with app.run_test() as pilot:
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, ConfigScreen)
            tbl = screen.query_one("#tbl-sandbox", DataTable)
            tbl.focus()
            ridx = next(
                r for r in range(tbl.row_count) if "run_commands" in str(tbl.get_row_at(r)[0])
            )
            tbl.move_cursor(row=ridx)
            await pilot.pause()
            screen.action_edit()
            await pilot.pause()
            modal = app.screen
            assert isinstance(modal, EditModal)
            # run_commands is an enum -> a [x]/[ ] chooser, focused, current "yes".
            field = modal.query_one("#edit-value", ChoiceField)
            assert field.value == "yes"
            await pilot.press("down")  # highlight "no" (selection unchanged)
            await pilot.pause()
            assert field.value == "yes"  # arrows only highlight now
            await pilot.press("space")  # select "no"
            await pilot.pause()
            assert field.value == "no"
            modal.action_save()  # equivalent to the Save action
            await pilot.pause()
            # Persisted through config_layer.set_config_value (global config).
            assert load_effective(repo).config.sandbox.run_commands == "no"

    asyncio.run(scenario())


def test_edit_unset_reverts_to_default(repo: Path) -> None:
    """The edit modal's "Unset → default" returns a setting to its default by
    removing the override (not by writing the default value back)."""

    async def scenario() -> None:
        from agent6.config.layer import effective_leaf

        app = _Host(repo)
        async with app.run_test() as pilot:
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, ConfigScreen)
            # run_commands is set to "yes" in the (global) config fixture.
            eff = load_effective(repo)
            assert effective_leaf(eff, "sandbox.run_commands") == ("yes", "global")
            tbl = screen.query_one("#tbl-sandbox", DataTable)
            tbl.focus()
            ridx = next(
                r for r in range(tbl.row_count) if "run_commands" in str(tbl.get_row_at(r)[0])
            )
            tbl.move_cursor(row=ridx)
            await pilot.pause()
            screen.action_edit()
            await pilot.pause()
            modal = app.screen
            assert isinstance(modal, EditModal)
            modal.action_unset()
            await pilot.pause()
            # Override removed -> back to the default, sourced as default.
            assert effective_leaf(load_effective(repo), "sandbox.run_commands") == (
                "ask",
                "default",
            )

    asyncio.run(scenario())


def test_edit_custom_value_inline(repo: Path) -> None:
    """A choice setting's last chooser row is an inline custom field: arrow down
    onto it and type the value right there (no separate box), and that text is
    the value -- no jump to a popped-up box below."""

    async def scenario() -> None:
        from agent6.ui.tui.config_page import ChoiceField

        app = _Host(repo)
        async with app.run_test() as pilot:
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, ConfigScreen)
            tbl = screen.query_one("#tbl-sandbox", DataTable)
            tbl.focus()
            ridx = next(
                r for r in range(tbl.row_count) if "run_commands" in str(tbl.get_row_at(r)[0])
            )
            tbl.move_cursor(row=ridx)
            await pilot.pause()
            screen.action_edit()
            await pilot.pause()
            modal = app.screen
            assert isinstance(modal, EditModal)
            field = modal.query_one("#edit-value", ChoiceField)
            # Highlight down to the custom row (yes -> no -> ask -> custom), then
            # type in place -- typing the custom row selects it.
            for _ in range(3):
                await pilot.press("down")
                await pilot.pause()
            for ch in ("z", "z", "z"):
                await pilot.press(ch)
            await pilot.pause()
            assert field.value == "zzz"
            assert modal._new_value() == "zzz"  # pyright: ignore[reportPrivateUsage]

    asyncio.run(scenario())


def test_edit_action_arrows_navigate(repo: Path) -> None:
    """When a flat action (Save / Unset / Cancel) is focused, Left/Right move
    between them (wrapping) -- arrow nav that depends on what's in focus."""

    async def scenario() -> None:
        from agent6.ui.tui.config_page import ActionItem

        app = _Host(repo)
        async with app.run_test() as pilot:
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, ConfigScreen)
            tbl = screen.query_one("#tbl-sandbox", DataTable)
            tbl.focus()
            tbl.move_cursor(row=0)
            await pilot.pause()
            screen.action_edit()
            await pilot.pause()
            modal = app.screen
            assert isinstance(modal, EditModal)
            items = list(modal.query(ActionItem))
            assert len(items) == 3  # Save, Unset, Cancel
            items[0].focus()
            await pilot.pause()
            await pilot.press("right")
            await pilot.pause()
            assert modal.focused is items[1]
            await pilot.press("left")
            await pilot.pause()
            assert modal.focused is items[0]
            await pilot.press("left")  # wrap past the start
            await pilot.pause()
            assert modal.focused is items[2]

    asyncio.run(scenario())


def test_provider_field_is_a_picker_of_configured_providers(repo: Path) -> None:
    """Editing models.<role>.provider shows a chooser of the configured provider
    names (a picker, not a blank text box)."""

    async def scenario() -> None:
        from agent6.ui.tui.config_page import ChoiceField

        app = _Host(repo)
        async with app.run_test(size=(100, 44)) as pilot:
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, ConfigScreen)
            tbl = screen.query_one("#tbl-models", DataTable)
            tbl.focus()
            ridx = next(
                r
                for r in range(tbl.row_count)
                if "worker" in str(tbl.get_row_at(r)[0]) and "provider" in str(tbl.get_row_at(r)[0])
            )
            tbl.move_cursor(row=ridx)
            await pilot.pause()
            screen.action_edit()
            await pilot.pause()
            modal = app.screen
            assert isinstance(modal, EditModal)
            # A ChoiceField (not a plain Input) => the configured providers were
            # injected as choices. The fixture configures "anthropic".
            field = modal.query_one("#edit-value", ChoiceField)
            assert field.value == "anthropic"

    asyncio.run(scenario())


def test_model_field_is_a_typeahead_picker(repo: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Editing models.<role>.model opens a type-to-narrow picker over the
    provider's models (cached + a live refresh), not a blank box."""
    import agent6.ui.tui.config_page as cp

    models = ["claude-opus-4-8", "claude-sonnet-4-6", "claude-sonnet-4-5", "claude-haiku-4-5"]

    def _models(*_a: object, **_k: object) -> list[str]:
        return models

    monkeypatch.setattr(cp, "cached_models", _models)
    monkeypatch.setattr(cp, "config_value_choices", _models)  # mock the live fetch

    async def scenario() -> None:
        from agent6.ui.tui.widgets import TypeaheadField

        app = _Host(repo)
        async with app.run_test(size=(100, 44)) as pilot:
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, ConfigScreen)
            tbl = screen.query_one("#tbl-models", DataTable)
            tbl.focus()
            ridx = next(
                r
                for r in range(tbl.row_count)
                if str(tbl.get_row_at(r)[0]).strip() == "worker.model"
            )
            tbl.move_cursor(row=ridx)
            await pilot.pause()
            screen.action_edit()
            await pilot.pause()
            modal = app.screen
            assert isinstance(modal, EditModal)
            field = modal.query_one("#edit-value", TypeaheadField)
            assert field.value == "claude-sonnet-4-5"  # the current model
            # First keystroke replaces + narrows; arrow highlights a match.
            await pilot.press("h")
            await pilot.pause()
            await pilot.press("down")
            await pilot.pause()
            assert field.value == "claude-haiku-4-5"

    asyncio.run(scenario())


def test_list_setting_prefill_saves_back_unchanged(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A list-valued setting must prefill the edit box in a form that SAVES:
    the display formatter rendered [uv, run, pytest] (unquoted, not TOML), so
    an untouched Save failed revalidation ("Input should be a valid tuple") --
    there was no format in which the shown value saved. The box now prefills
    the exact inverse of parse_cli_value."""
    gdir = tmp_path / "g"
    gdir.mkdir()
    (gdir / "config.toml").write_text(
        _GLOBAL + '\n[workflow]\nverify_command = ["uv", "run", "pytest"]\n',
        encoding="utf-8",
    )
    monkeypatch.setenv("AGENT6_CONFIG_HOME", str(gdir))
    monkeypatch.setenv("AGENT6_STATE_HOME", str(tmp_path / "state"))
    monkeypatch.setenv("AGENT6_CACHE_HOME", str(tmp_path / "cache"))
    repo_root = tmp_path / "repo"
    repo_root.mkdir()

    async def scenario() -> None:
        app = _Host(repo_root)
        async with app.run_test(size=(100, 44)) as pilot:
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, ConfigScreen)
            tbl = screen.query_one("#tbl-workflow", DataTable)
            tbl.focus()
            ridx = next(
                r
                for r in range(tbl.row_count)
                if str(tbl.get_row_at(r)[0]).strip() == "verify_command"
            )
            tbl.move_cursor(row=ridx)
            await pilot.pause()
            screen.action_edit()
            await pilot.pause()
            modal = app.screen
            assert isinstance(modal, EditModal)
            field = modal.query_one("#edit-value", Input)
            assert field.value == '["uv", "run", "pytest"]'  # round-trippable TOML
            modal.action_save()  # untouched Save must succeed, not error
            await pilot.pause()
            assert isinstance(app.screen, ConfigScreen)

    asyncio.run(scenario())
    saved = load_effective(repo_root, None).config.workflow.verify_command
    assert saved == ("uv", "run", "pytest")  # unchanged, not corrupted to a str


def test_editing_a_model_survives_a_broken_secrets_file(
    repo: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """secrets.toml with unsafe perms (0644, e.g. restored from a backup) must
    not crash the WHOLE TUI when the edit modal's background model-list fetch
    runs: the thread worker's SecretsError hit textual's default exit_on_error
    and tore the app down. The fetch (`models.choices`) degrades to a keyless
    attempt instead (the sanctioned models/validate.py pattern)."""
    import agent6.ui.tui.config_page as cp
    from agent6.models import choices

    gdir = Path(os.environ["AGENT6_CONFIG_HOME"])
    secrets = gdir / "secrets.toml"
    secrets.write_text('[anthropic]\napi_key = "sk-x"\n', encoding="utf-8")
    secrets.chmod(0o644)  # group/other-readable -> load_secrets raises

    models = ["claude-sonnet-4-5"]

    def _models(*_a: object, **_k: object) -> list[str]:
        return models

    monkeypatch.setattr(cp, "cached_models", _models)
    monkeypatch.setattr(choices, "list_models", _models)  # the live fetch, keyless here

    async def scenario() -> None:
        app = _Host(repo)
        async with app.run_test(size=(100, 44)) as pilot:
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, ConfigScreen)
            tbl = screen.query_one("#tbl-models", DataTable)
            tbl.focus()
            ridx = next(
                r
                for r in range(tbl.row_count)
                if str(tbl.get_row_at(r)[0]).strip() == "worker.model"
            )
            tbl.move_cursor(row=ridx)
            await pilot.pause()
            screen.action_edit()
            # Let the thread worker run; the app must survive it.
            for _ in range(6):
                await pilot.pause(0.05)
            assert isinstance(app.screen, EditModal)  # still open, app alive

    asyncio.run(scenario())


def test_edit_modal_up_at_top_is_a_hard_stop(repo: Path) -> None:
    """↑ at the top of the first chooser must STAY there, not escape to the
    focusable scroll container (which stranded the arrows). Regression guard."""

    async def scenario() -> None:
        from agent6.ui.tui.config_page import ChoiceField

        app = _Host(repo)
        async with app.run_test() as pilot:
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, ConfigScreen)
            tbl = screen.query_one("#tbl-sandbox", DataTable)
            tbl.focus()
            ridx = next(
                r for r in range(tbl.row_count) if "run_commands" in str(tbl.get_row_at(r)[0])
            )
            tbl.move_cursor(row=ridx)
            await pilot.pause()
            screen.action_edit()
            await pilot.pause()
            modal = app.screen
            assert isinstance(modal, EditModal)
            field = modal.query_one("#edit-value", ChoiceField)
            assert modal.focused is field and field._cursor == 0  # pyright: ignore[reportPrivateUsage]
            await pilot.press("up")  # at the top edge
            await pilot.pause()
            assert modal.focused is field  # stayed put (didn't escape to the scroll box)
            await pilot.press("down")  # highlight still moves afterwards
            await pilot.pause()
            assert modal.focused is field and field._cursor == 1  # pyright: ignore[reportPrivateUsage]

    asyncio.run(scenario())


def test_q_backs_out_from_config_but_types_in_search(repo: Path) -> None:
    """Option 3: q backs out of the Config screen (only the root hub quits on q),
    yet still types normally in the search box — the focused Input eats it first.
    The menu's Quit (^Q -> action_quit) still exits the app."""

    async def scenario() -> None:
        app = _Host(repo)
        async with app.run_test() as pilot:
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, ConfigScreen)
            exits: list[int] = []
            orig = app.exit
            app.exit = lambda *a, **k: exits.append(1) or orig(*a, **k)  # type: ignore[assignment]

            # In the search Input, q types (no quit).
            screen.action_search()
            await pilot.pause()
            await pilot.press("q")
            await pilot.pause()
            assert screen.query_one("#search", Input).value == "q"
            assert not exits

            # Out on a table, q backs out (dismiss) instead of quitting.
            screen._cancel_search()  # pyright: ignore[reportPrivateUsage]
            screen.query_one("#tbl-sandbox", DataTable).focus()
            await pilot.pause()
            await pilot.press("q")
            await pilot.pause()
            assert not exits  # q did NOT quit
            assert not isinstance(app.screen, ConfigScreen)  # it backed out

            # The menu's Quit (^Q) path still quits the app.
            screen.action_quit()
            await pilot.pause()
            assert exits

    asyncio.run(scenario())


def test_view_menu_opens_theme_picker(repo: Path) -> None:
    """The View>Theme item (and action_choose_theme) opens the theme picker."""

    async def scenario() -> None:
        from agent6.ui.tui.theme import ThemePicker

        app = _Host(repo)
        async with app.run_test() as pilot:
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, ConfigScreen)
            screen.action_choose_theme()
            await pilot.pause()
            assert isinstance(app.screen, ThemePicker)

    asyncio.run(scenario())


def test_menu_bar_opens_and_dispatches(repo: Path) -> None:
    """Open a menu (mouse/Alt/F-key all route here), see its items, and picking
    one runs the same action_<id> as the key binding and command palette."""

    async def scenario() -> None:
        app = _Host(repo)
        async with app.run_test() as pilot:
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, ConfigScreen)
            mb = screen.query_one(MenuBar)

            # Open the View menu; its items carry their action ids. The dropdown
            # mounts on the screen (not the 1-row bar, which would clip it).
            mb.open("v")
            await pilot.pause()
            dd = next(iter(screen.query(OptionList)))
            ids = [dd.get_option_at_index(i).id for i in range(dd.option_count)]
            assert "search" in ids and "toggle_modified" in ids

            # Pick "Modified only" -> OptionSelected -> MenuBar.Selected ->
            # screen action_toggle_modified. The whole dispatch chain.
            assert screen._modified_only is False  # pyright: ignore[reportPrivateUsage]
            idx = next(
                i
                for i in range(dd.option_count)
                if dd.get_option_at_index(i).id == "toggle_modified"
            )
            dd.highlighted = idx
            await pilot.press("enter")
            await pilot.pause()
            assert screen._modified_only is True  # pyright: ignore[reportPrivateUsage]

    asyncio.run(scenario())


def test_menu_reopen_no_duplicate(repo: Path) -> None:
    """Switching/re-opening menus must not raise DuplicateIds (the dropdown no
    longer reuses a fixed id) and converges to exactly one open menu."""

    async def scenario() -> None:
        app = _Host(repo)
        async with app.run_test() as pilot:
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, ConfigScreen)
            mb = screen.query_one(MenuBar)
            for m in ("v", "e", "v", "v", "c"):  # used to crash with DuplicateIds
                mb.open(m)
                await pilot.pause()
            assert len(list(screen.query(OptionList))) == 1  # exactly one menu open
            mb.open("c")  # opening the open menu toggles it shut
            await pilot.pause()
            assert len(list(screen.query(OptionList))) == 0

    asyncio.run(scenario())


def test_menu_opens_on_mouse_click(repo: Path) -> None:
    """A mouse click on a title opens its menu and the dropdown is visible (not
    clipped by the 1-row bar). events.Click carries no .widget, so each title
    handles its own click."""

    async def scenario() -> None:
        app = _Host(repo)
        async with app.run_test() as pilot:
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, ConfigScreen)
            await pilot.click("#menu-v")  # click the View title
            await pilot.pause()
            dds = list(screen.query(OptionList))
            assert len(dds) == 1
            dd = dds[0]
            # Floated on the screen (overlay:screen + absolute_offset), so it
            # shows below the bar at full height rather than clipped to one row.
            assert dd.region.height > 1
            assert dd.region.y >= 1

    asyncio.run(scenario())


def test_menu_toggle_switch_and_click_away(repo: Path) -> None:
    """Mouse: a title opens its menu, clicking it again toggles it shut (the
    on_blur/open re-open race is fixed), clicking another switches, and clicking
    the body closes."""

    async def scenario() -> None:
        app = _Host(repo)
        async with app.run_test(size=(120, 30)) as pilot:
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, ConfigScreen)

            def n() -> int:
                return len(list(screen.query(OptionList)))

            await pilot.click("#menu-e")
            await pilot.pause()
            assert n() == 1
            await pilot.click("#menu-e")  # same title again -> toggle shut
            await pilot.pause()
            assert n() == 0
            await pilot.click("#menu-e")
            await pilot.pause()
            assert n() == 1
            await pilot.click("#menu-v")  # different title -> switch
            await pilot.pause()
            assert n() == 1
            assert screen.query_one("#menu-v").has_class("-open")
            # Click away: a click elsewhere moves focus off the dropdown, which
            # closes it (via on_blur). Drive that focus change directly -- a real
            # click's pixel target depends on layout; the close mechanism is the
            # focus loss, exercised here and confirmed under tmux.
            screen.query_one("#tbl-sandbox", DataTable).focus()
            await pilot.pause()
            assert n() == 0
            assert not screen.query_one("#menu-v").has_class("-open")

    asyncio.run(scenario())


def test_menu_left_right_switches_open_menu(repo: Path) -> None:
    """Left/Right move between menus while one is open (classic menu-bar feel)."""

    async def scenario() -> None:
        app = _Host(repo)
        async with app.run_test(size=(120, 30)) as pilot:
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, ConfigScreen)
            screen.query_one(MenuBar).open("e")
            await pilot.pause()
            assert screen.query_one("#menu-e").has_class("-open")
            await pilot.press("right")
            await pilot.pause()
            assert screen.query_one("#menu-v").has_class("-open")
            await pilot.press("left")
            await pilot.pause()
            assert screen.query_one("#menu-e").has_class("-open")

    asyncio.run(scenario())


def test_open_menu_title_stays_highlighted(repo: Path) -> None:
    """The open menu's title carries the -open class (so it reads as active) and
    drops it on close."""

    async def scenario() -> None:
        app = _Host(repo)
        async with app.run_test() as pilot:
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, ConfigScreen)
            mb = screen.query_one(MenuBar)
            mb.open("e")
            await pilot.pause()
            assert screen.query_one("#menu-e").has_class("-open")
            mb.close_menu()
            await pilot.pause()
            assert not screen.query_one("#menu-e").has_class("-open")

    asyncio.run(scenario())


def test_config_actions_in_command_palette(repo: Path) -> None:
    """Every Config action is searchable in the Ctrl+P palette -- discovery by
    typing, no memorizing. Labels are the descriptive MENUS form (matching the menu
    bar + the home/run palettes), not the footer's terse CONFIG_ACTIONS labels."""

    async def scenario() -> None:
        app = _Host(repo)
        async with app.run_test() as pilot:
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, ConfigScreen)
            labels = [name for name, _, _ in screen.palette_commands()]
            for expected in (
                "Filter",
                "Modified only",
                "Edit setting…",
                "Reset to default",
                "Refresh",
                "Keys & actions",
            ):
                assert expected in labels
            # the terse footer labels do NOT leak into the palette
            assert "Help" not in labels and "Edit" not in labels

    asyncio.run(scenario())


def test_enter_on_setting_row_opens_editor(repo: Path) -> None:
    """Enter (or double-click) on a setting row opens the edit modal. The
    DataTable consumes Enter for its own RowSelected, so it's wired via that
    event, not the screen's `enter` binding."""

    async def scenario() -> None:
        app = _Host(repo)
        async with app.run_test() as pilot:
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, ConfigScreen)
            tbl = screen.query_one("#tbl-sandbox", DataTable)
            tbl.focus()
            tbl.move_cursor(row=0)
            await pilot.press("enter")
            await pilot.pause()
            assert isinstance(app.screen, EditModal)

    asyncio.run(scenario())


def test_esc_clears_filter_before_closing(repo: Path) -> None:
    """Esc backs out of an active filter first (clears it + drops back to the
    settings, stays on the page); a later Esc closes the page."""

    async def scenario() -> None:
        app = _Host(repo)
        async with app.run_test() as pilot:
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, ConfigScreen)
            search = screen.query_one("#search", Input)
            screen.action_search()  # / focuses the inline filter
            await pilot.pause()
            assert screen.focused is search
            search.value = "run_commands"
            screen._refresh()  # pyright: ignore[reportPrivateUsage]
            await pilot.pause()
            await pilot.press("escape")
            await pilot.pause()
            assert isinstance(app.screen, ConfigScreen)  # NOT closed
            assert search.value == ""  # filter cleared
            assert screen.focused is not search  # focus dropped into the settings

    asyncio.run(scenario())


def test_filter_arrow_in_and_out(repo: Path) -> None:
    """Down/Enter step out of the filter into the settings (keeping the filter);
    Up from the topmost header returns to the filter box."""

    async def scenario() -> None:
        from agent6.ui.tui.config_page import _NavTable

        app = _Host(repo)
        async with app.run_test(size=(100, 44)) as pilot:
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, ConfigScreen)
            search = screen.query_one("#search", Input)
            screen.action_search()
            await pilot.pause()
            await pilot.press("down")  # step out into the settings
            await pilot.pause()
            assert isinstance(screen.focused, _NavTable)
            # Up from the first row -> header -> Up again returns to the filter.
            await pilot.press("up")
            await pilot.pause()
            await pilot.press("up")
            await pilot.pause()
            assert screen.focused is search

    asyncio.run(scenario())


def test_arrows_flow_through_section_headers(repo: Path) -> None:
    """Arrows flow as one list THROUGH the section headers: Down at a section's
    last row lands on the next header, Down again enters its rows; Up retraces.
    Enter on a header collapses/expands it."""

    async def scenario() -> None:
        from textual.widgets import Collapsible

        from agent6.ui.tui.config_page import _NavTable

        app = _Host(repo)
        async with app.run_test(size=(110, 44)) as pilot:
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, ConfigScreen)
            tables = [t for t in screen.query(_NavTable) if t.row_count]

            def on_header() -> bool:
                p = getattr(app.focused, "parent", None)
                return isinstance(p, Collapsible) and bool(p.id and p.id.startswith("sec-"))

            tables[0].focus()
            tables[0].move_cursor(row=tables[0].row_count - 1)
            await pilot.press("down")
            await pilot.pause()
            assert on_header()  # landed on the next section's header
            await pilot.press("down")
            await pilot.pause()
            assert app.focused is tables[1]  # then into its rows
            await pilot.press("up")
            await pilot.pause()
            assert on_header()  # back up onto the header
            # Enter on the header toggles its section.
            section = app.focused.parent.id[4:]  # type: ignore[union-attr]
            col = screen.query_one(f"#sec-{section}", Collapsible)
            was = col.collapsed
            await pilot.press("enter")
            await pilot.pause()
            assert col.collapsed is not was

    asyncio.run(scenario())


def test_add_provider_via_form_persists(repo: Path) -> None:
    """The Add-provider form writes a validated [providers.<name>] block --
    dropdowns for api_format/deployment, inputs for name/base_url/api_key_env --
    and the page reflects it. No hand-editing a TOML dict."""

    async def scenario() -> None:
        from agent6.ui.tui.config_page import ChoiceField, ProviderModal

        app = _Host(repo)
        async with app.run_test(size=(110, 44)) as pilot:
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, ConfigScreen)
            screen.action_add_provider()
            await pilot.pause()
            modal = app.screen
            assert isinstance(modal, ProviderModal)
            modal.query_one("#prov-name", Input).value = "openrouter"
            fmt = modal.query_one("#prov-format", ChoiceField)
            # Typing the known preset name prefilled openai; exercise the
            # chooser with a round trip (up to anthropic, back down, Space)
            # so the assertion holds however many formats the union grows.
            fmt.focus()
            await pilot.pause()
            await pilot.press("up")
            await pilot.press("down")
            await pilot.press("space")
            await pilot.pause()
            assert fmt.value == "openai"
            modal.query_one("#prov-baseurl", Input).value = "https://openrouter.ai/api/v1"
            await pilot.pause()
            modal.action_add()  # equivalent to the Add action
            await pilot.pause()
            assert isinstance(app.screen, ConfigScreen)  # closed on success
            cfg = load_effective(repo).config
            assert "openrouter" in cfg.providers
            entry = cfg.providers["openrouter"]
            assert isinstance(entry, OpenAIProviderEntry)
            assert entry.base_url == "https://openrouter.ai/api/v1"

    asyncio.run(scenario())


def test_add_provider_prefills_known_preset_base_url(repo: Path) -> None:
    """Regression: typing a known provider name (openrouter) in the Add-provider
    form prefills its api_format + base_url from PROVIDER_DEFAULTS, so submitting
    WITHOUT hand-typing a URL lands on openrouter.ai -- not the api.openai.com
    fallback in config._default_base_url. The form used to ignore the presets
    that `agent6 connect` applies, silently pointing openrouter at OpenAI."""

    async def scenario() -> None:
        from agent6.ui.tui.config_page import ChoiceField, ProviderModal

        app = _Host(repo)
        async with app.run_test(size=(110, 44)) as pilot:
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, ConfigScreen)
            screen.action_add_provider()
            await pilot.pause()
            modal = app.screen
            assert isinstance(modal, ProviderModal)
            # Type only the name; leave api_format + base_url untouched.
            modal.query_one("#prov-name", Input).value = "openrouter"
            await pilot.pause()
            # Live prefill flipped the format dropdown and filled the URL field.
            assert modal.query_one("#prov-format", ChoiceField).value == "openai"
            assert modal.query_one("#prov-baseurl", Input).value == "https://openrouter.ai/api/v1"
            modal.action_add()
            await pilot.pause()
            assert isinstance(app.screen, ConfigScreen)  # written + validated, modal closed
            cfg = load_effective(repo).config
            entry = cfg.providers["openrouter"]
            assert isinstance(entry, OpenAIProviderEntry)
            assert entry.base_url == "https://openrouter.ai/api/v1"

    asyncio.run(scenario())


def test_add_provider_prefill_keeps_user_typed_base_url(repo: Path) -> None:
    """The name-based prefill never overwrites a base_url the user typed: set a
    custom URL first, then type a known name -- the custom URL stays."""

    async def scenario() -> None:
        from agent6.ui.tui.config_page import ProviderModal

        app = _Host(repo)
        async with app.run_test(size=(110, 44)) as pilot:
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, ConfigScreen)
            screen.action_add_provider()
            await pilot.pause()
            modal = app.screen
            assert isinstance(modal, ProviderModal)
            modal.query_one("#prov-baseurl", Input).value = "https://my.proxy/v1"
            await pilot.pause()
            modal.query_one("#prov-name", Input).value = "openrouter"
            await pilot.pause()
            # Their URL is preserved (only our own autofill, or a blank, is replaced).
            assert modal.query_one("#prov-baseurl", Input).value == "https://my.proxy/v1"

    asyncio.run(scenario())


def test_edit_base_url_prefills_preset_for_known_provider(repo: Path) -> None:
    """UX parity with the Add form + `agent6 connect`: when a known provider's
    base_url is still the generic default (api.openai.com, filled for any unset
    openai-format provider), opening its base_url editor offers the name's preset
    URL prefilled -- so re-setting an unset openrouter is one Save, no need to
    know the host."""

    async def scenario() -> None:
        from agent6.config.write import set_config_table
        from agent6.ui.tui.config_page import EditModal

        # An openrouter provider with NO base_url -> effective default api.openai.com.
        assert set_config_table(repo, "providers.openrouter", {"api_format": "openai"}) is None

        app = _Host(repo)
        async with app.run_test(size=(110, 44)) as pilot:
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, ConfigScreen)
            tbl = screen.query_one("#tbl-providers", DataTable)
            tbl.focus()
            ridx = next(
                r
                for r in range(tbl.row_count)
                if "openrouter" in str(tbl.get_row_at(r)[0])
                and "base_url" in str(tbl.get_row_at(r)[0])
            )
            tbl.move_cursor(row=ridx)
            await pilot.pause()
            screen.action_edit()
            await pilot.pause()
            modal = app.screen
            assert isinstance(modal, EditModal)
            # Prefilled with the preset host, not the generic api.openai.com default.
            assert modal.query_one("#edit-value", Input).value == "https://openrouter.ai/api/v1"
            modal.action_save()
            await pilot.pause()
            cfg = load_effective(repo).config
            entry = cfg.providers["openrouter"]
            assert isinstance(entry, OpenAIProviderEntry)
            assert entry.base_url == "https://openrouter.ai/api/v1"

    asyncio.run(scenario())


def test_up_off_first_setting_reveals_top_header_then_filter(repo: Path) -> None:
    """Regression: in a short window, Up off the first setting must focus AND reveal
    the first section's header (the smooth-scroll left the very top row a line
    off-screen, so it looked like Up skipped the header straight to the filter),
    then Up again reaches the #search filter."""
    from textual.containers import VerticalScroll
    from textual.widgets._collapsible import CollapsibleTitle

    async def scenario() -> None:
        app = _Host(repo)
        async with app.run_test(size=(100, 10)) as pilot:  # short: #settings scrolls
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, ConfigScreen)
            settings = screen.query_one("#settings", VerticalScroll)
            screen._focus_first_setting()
            await pilot.pause()
            for _ in range(8):  # scroll down off the top
                await pilot.press("down")
                await pilot.pause()
            header = None
            for _ in range(20):  # walk back up to the [agent6] header
                await pilot.press("up")
                await pilot.pause()
                f = screen.focused
                if isinstance(f, CollapsibleTitle) and getattr(f.parent, "id", "") == "sec-agent6":
                    header = f
                    break
            assert header is not None, "never reached the [agent6] header going up"
            top, bottom = settings.region.y, settings.region.y + settings.region.height
            assert top <= header.region.y < bottom, "top header focused but scrolled off-screen"
            await pilot.press("up")
            await pilot.pause()
            assert isinstance(screen.focused, Input)  # Up off the top header -> filter

    asyncio.run(scenario())


def test_reset_on_a_profile_sourced_setting_tells_the_truth(repo: Path) -> None:
    """A [presets.<name>] leaf renders modified with source "preset", and no
    config-file unset can revert it; Reset must say the preset owns it, not
    lie "already at its default"."""
    gdir = Path(os.environ["AGENT6_CONFIG_HOME"])
    (gdir / "config.toml").write_text(
        'preset = "fast"\n' + _GLOBAL + '\n[presets.fast.review]\ntrigger = "off"\n',
        encoding="utf-8",
    )

    async def scenario() -> None:
        app = _Host(repo)
        async with app.run_test() as pilot:
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, ConfigScreen)
            setting = next(
                s
                for s in build_config_view(load_effective(repo, None)).settings
                if s.key == "review.trigger"
            )
            assert setting.source == "preset" and setting.modified
            screen._current_setting = lambda: setting  # type: ignore[method-assign]
            screen.action_reset()
            await pilot.pause()
            notes = [str(n.message) for n in app._notifications]  # pyright: ignore[reportPrivateUsage]
            assert notes, "no notification fired"
            assert "already at its default" not in notes[-1]
            assert "preset" in notes[-1]

    asyncio.run(scenario())


def test_reset_on_a_flag_sourced_setting_names_the_flag_layer(repo: Path, tmp_path: Path) -> None:
    """A leaf a `--config FILE` layer set is not a preset leaf: Reset names the
    layer it came from, so the operator edits the file they passed."""
    overlay = tmp_path / "overlay.toml"
    overlay.write_text('[review]\ntrigger = "off"\n', encoding="utf-8")

    async def scenario() -> None:
        app = _Host(repo, overlay)
        async with app.run_test() as pilot:
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, ConfigScreen)
            setting = next(
                s
                for s in build_config_view(load_effective(repo, overlay)).settings
                if s.key == "review.trigger"
            )
            assert setting.source == "flag" and setting.modified
            screen._current_setting = lambda: setting  # type: ignore[method-assign]
            screen.action_reset()
            await pilot.pause()
            notes = [str(n.message) for n in app._notifications]  # pyright: ignore[reportPrivateUsage]
            assert notes, "no notification fired"
            assert "preset" not in notes[-1]
            assert "flag" in notes[-1]

    asyncio.run(scenario())


def test_reload_on_an_invalid_on_disk_config_keeps_the_last_good_view(repo: Path) -> None:
    """Hand-editing the config invalid in another terminal then pressing r must
    notify with the `agent6 config fix` pointer and keep the last-good table,
    not crash the whole TUI out of the action handler (the hub's open guard
    already documents that intent)."""

    async def scenario() -> None:
        app = _Host(repo)
        async with app.run_test() as pilot:
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, ConfigScreen)
            baseline = _row_total(screen)
            assert baseline > 10
            gdir = Path(os.environ["AGENT6_CONFIG_HOME"])
            (gdir / "config.toml").write_text(
                _GLOBAL + '\n[workflow]\nplan = "yess"\n', encoding="utf-8"
            )
            await pilot.press("r")
            await pilot.pause()
            assert isinstance(app.screen, ConfigScreen)  # still alive
            assert _row_total(screen) == baseline  # last-good view retained
            notes = [str(n.message) for n in app._notifications]  # pyright: ignore[reportPrivateUsage]
            assert any("config fix" in m for m in notes), notes
            # Edit must not crash either (the choice helpers no longer re-read disk).
            await pilot.press("e")
            await pilot.pause()

    asyncio.run(scenario())


def test_setting_description_lives_in_the_edit_modal_only(repo: Path) -> None:
    """The edit modal explains the highlighted leaf; the page itself carries
    no detail pane (a pane under the table resized with every highlight and
    made scrolling janky, so the operator removed it)."""
    from textual.widgets import Static

    async def scenario() -> None:
        app = _Host(repo)
        async with app.run_test(size=(120, 44)) as pilot:
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, ConfigScreen)
            assert not screen.query("#detail")
            tbl = screen.query_one("#tbl-sandbox", DataTable)
            tbl.focus()
            ridx = next(
                r
                for r in range(tbl.row_count)
                if str(tbl.get_row_at(r)[0]).strip() == "run_commands"
            )
            tbl.move_cursor(row=ridx)
            await pilot.pause()
            screen.action_edit()
            await pilot.pause()
            modal = app.screen
            assert isinstance(modal, EditModal)
            shown = str(modal.query_one("#edit-description", Static).render())
            assert "run_command" in shown and "**" not in shown

    asyncio.run(scenario())
