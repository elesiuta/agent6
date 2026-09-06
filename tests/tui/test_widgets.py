# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Shared form widgets keep their keyboard and mouse state visible."""

from __future__ import annotations

import asyncio

from textual.app import App, ComposeResult
from textual.containers import VerticalScroll
from textual.widgets import Static

from agent6.ui.tui.widgets import ChoiceField, TypeaheadField


class _ChoiceScrollHost(App[None]):
    CSS = "#scroll { height: 6; width: 40; } #before { height: 10; }"

    def compose(self) -> ComposeResult:
        with VerticalScroll(id="scroll"):
            yield Static("before", id="before")
            yield ChoiceField(tuple(f"option-{i}" for i in range(12)), "option-0")

    def on_mount(self) -> None:
        self.query_one(ChoiceField).focus()


def test_choice_cursor_stays_visible_below_prior_content() -> None:
    async def scenario() -> None:
        app = _ChoiceScrollHost()
        async with app.run_test(size=(50, 15)) as pilot:
            await pilot.pause()
            await pilot.press("down")
            await pilot.pause()
            field = app.query_one(ChoiceField)
            scroll = app.query_one("#scroll", VerticalScroll)
            cursor_y = field.content_region.y + field._cursor  # pyright: ignore[reportPrivateUsage]
            assert (
                scroll.scrollable_content_region.y
                <= cursor_y
                < scroll.scrollable_content_region.bottom
            )

    asyncio.run(scenario())


class _ChoiceWidthHost(App[None]):
    CSS = "ChoiceField { width: 12; }"

    def compose(self) -> ComposeResult:
        yield ChoiceField(("abcdefghijklmnopqrstuvwxyz", "second"), "abcdefghijklmnopqrstuvwxyz")


def test_choice_options_each_use_one_screen_row() -> None:
    async def scenario() -> None:
        app = _ChoiceWidthHost()
        async with app.run_test(size=(30, 10)) as pilot:
            await pilot.pause()
            field = app.query_one(ChoiceField)
            assert field.region.height == field._row_count  # pyright: ignore[reportPrivateUsage]

    asyncio.run(scenario())


class _TypeaheadScrollHost(App[None]):
    CSS = "#scroll { height: 6; width: 40; } #before { height: 10; }"

    def compose(self) -> ComposeResult:
        with VerticalScroll(id="scroll"):
            yield Static("before", id="before")
            yield TypeaheadField("", [f"option-{i}" for i in range(10)])

    def on_mount(self) -> None:
        self.query_one(TypeaheadField).focus()


def test_typeahead_highlight_stays_visible_when_suggestions_expand() -> None:
    async def scenario() -> None:
        app = _TypeaheadScrollHost()
        async with app.run_test(size=(50, 15)) as pilot:
            await pilot.pause()
            await pilot.press("down")
            await pilot.pause()
            field = app.query_one(TypeaheadField)
            scroll = app.query_one("#scroll", VerticalScroll)
            highlight_y = (
                field.content_region.y + field._index + 1  # pyright: ignore[reportPrivateUsage]
            )
            assert (
                scroll.scrollable_content_region.y
                <= highlight_y
                < scroll.scrollable_content_region.bottom
            )

    asyncio.run(scenario())


class _TypeaheadRefreshHost(App[None]):
    def compose(self) -> ComposeResult:
        yield TypeaheadField("current", ["alpha", "beta"])

    def on_mount(self) -> None:
        self.query_one(TypeaheadField).focus()


def test_typeahead_live_refresh_preserves_highlighted_value() -> None:
    async def scenario() -> None:
        app = _TypeaheadRefreshHost()
        async with app.run_test() as pilot:
            await pilot.pause()
            field = app.query_one(TypeaheadField)
            await pilot.press("down")
            assert field.value == "alpha"
            field.set_suggestions(["zeta", "alpha", "beta"])
            await pilot.pause()
            assert field.value == "alpha"

    asyncio.run(scenario())
