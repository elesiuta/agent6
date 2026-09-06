# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The fish-style pause-menu line reader: Tab previews and cycles commands,
typing steers, history recalls; input()'s EOF/interrupt contract holds."""

from __future__ import annotations

import io
import os
import re
import sys

import pytest

from agent6.ui.cli._menu_input import (
    _read_key,  # pyright: ignore[reportPrivateUsage]
    menu_input,
)
from agent6.ui.cli._steer_menu import MENU_COMMANDS
from agent6.ui.cli._terminal_guard import ScrubbedStream


def _chars(text: str) -> list[str]:
    return [f"char:{c}" for c in text]


def _run(keys: list[str], history: list[str] | None = None) -> tuple[str, str]:
    """Drive menu_input with scripted keys; returns (line, everything written)."""
    out: list[str] = []
    it = iter(keys)
    line = menu_input(
        "P> ",
        MENU_COMMANDS,
        history if history is not None else [],
        read_key=lambda: next(it),
        write=out.append,
    )
    return line, "".join(out)


def test_typed_line_returns_verbatim_and_lands_in_history() -> None:
    history: list[str] = []
    line, _ = _run([*_chars("focus on tests"), "enter"], history)
    assert line == "focus on tests"
    assert history == ["focus on tests"]
    # Accepting the same line again does not duplicate the history entry.
    line, _ = _run([*_chars("focus on tests"), "enter"], history)
    assert history == ["focus on tests"]


def test_tab_on_empty_line_previews_all_commands_and_cycles() -> None:
    line, out = _run(["tab", "enter"])
    assert line == "/status"  # first candidate selected
    # The menu rendered every command with its description.
    assert "/detach" in out and MENU_COMMANDS["/detach"] in out
    assert "\x1b[7m" in out  # the selection is highlighted
    line, _ = _run(["tab", "tab", "enter"])
    assert line == "/tasks"  # second candidate
    line, _ = _run(["tab", "backtab", "enter"])
    assert line == "/help"  # backwards wraps to the last


def test_tab_prefix_filters_and_arrows_move_selection() -> None:
    line, out = _run([*_chars("/st"), "tab", "enter"])
    assert line == "/status"
    assert "/stop" in out  # both matches were previewed
    assert "/tasks" not in out  # non-matches stay out of the menu
    line, _ = _run([*_chars("/st"), "tab", "down", "enter"])
    assert line == "/stop"
    line, _ = _run([*_chars("/st"), "tab", "down", "up", "enter"])
    assert line == "/status"


def test_unique_prefix_completes_without_a_menu() -> None:
    line, out = _run([*_chars("/sta"), "tab", "enter"])
    assert line == "/status"
    assert "\x1b[7m" not in out  # no menu, no highlight


def test_esc_restores_the_typed_stem_and_typing_keeps_the_candidate() -> None:
    line, _ = _run([*_chars("/st"), "tab", "esc", "enter"])
    assert line == "/st"
    # Typing after cycling keeps the selected candidate and edits from there.
    line, _ = _run(["tab", *_chars("x"), "enter"])
    assert line == "/statusx"


def test_tab_is_inert_inside_steer_text() -> None:
    line, out = _run([*_chars("fix it"), "tab", "enter"])
    assert line == "fix it"
    assert "\a" in out  # rang the bell instead of opening a menu
    line, out = _run([*_chars("q"), "tab", "enter"])
    assert line == "q"  # a non-slash word never completes
    assert "\a" in out


def test_editing_keys() -> None:
    line, _ = _run([*_chars("ab"), "left", *_chars("X"), "enter"])
    assert line == "aXb"
    line, _ = _run([*_chars("ab"), "backspace", "enter"])
    assert line == "a"
    line, _ = _run([*_chars("ab"), "kill-line", *_chars("c"), "enter"])
    assert line == "c"
    line, _ = _run([*_chars("keep this word"), "kill-word", "enter"])
    assert line == "keep this "
    line, _ = _run([*_chars("ab"), "home", "delete", "enter"])
    assert line == "b"


def test_history_recall_with_draft() -> None:
    history = ["first", "second"]
    line, _ = _run(["up", "enter"], history)
    assert line == "second"
    line, _ = _run(["up", "up", "enter"], history)
    assert line == "first"
    # Down past the newest entry restores the unsubmitted draft.
    line, _ = _run([*_chars("dra"), "up", "down", *_chars("ft"), "enter"], history)
    assert line == "draft"


def test_ctrl_r_searches_history_and_fills_the_line() -> None:
    entries = ["fix the parser", "run the suite", "fix the tests"]
    # Newest match first; Enter keeps it for editing (a second Enter sends).
    line, out = _run(
        ["history-search", *_chars("fix"), "enter", *_chars("!"), "enter"], list(entries)
    )
    assert line == "fix the tests!"
    assert "search: " in out
    assert "\x1b[7m" in out  # the selected match is highlighted
    # Ctrl-R again moves to the next older match, skipping the non-match.
    line, _ = _run(
        ["history-search", *_chars("fix"), "history-search", "enter", "enter"], list(entries)
    )
    assert line == "fix the parser"


def test_search_esc_restores_and_no_match_keeps_the_query() -> None:
    history = ["deploy it"]
    line, _ = _run([*_chars("draft"), "history-search", *_chars("dep"), "esc", "enter"], history)
    assert line == "draft"
    line, out = _run(["history-search", *_chars("zzz"), "enter", "enter"], history)
    assert line == "zzz"
    assert "(no match)" in out


def test_search_needs_history_and_caps_the_rows() -> None:
    line, out = _run(["history-search", *_chars("x"), "enter"], [])
    assert line == "x" and "\a" in out  # no history: bell, stay in normal input
    many = [f"steer {i}" for i in range(12)]
    _, out = _run(["history-search", "esc", "enter"], many)
    assert "steer 11" in out  # newest first
    assert "… 4 more" in out  # 12 matches, 8 rendered
    assert "steer 3" not in out
    # Repeats collapse to their newest occurrence: one match, no overflow row.
    _, out = _run(["history-search", "esc", "enter"], ["same"] * 12)
    assert "more" not in out


def test_eof_contract() -> None:
    with pytest.raises(EOFError):
        _run(["eof"])
    # Ctrl-D on a non-empty line is a bell, not EOF.
    line, out = _run([*_chars("x"), "eof", "enter"])
    assert line == "x" and "\a" in out


def test_interrupt_raises_keyboard_interrupt() -> None:
    with pytest.raises(KeyboardInterrupt):
        _run([*_chars("half typed"), "interrupt"])


def test_menu_rows_clamp_to_narrow_terminals(monkeypatch: pytest.MonkeyPatch) -> None:
    """Rows stay one terminal row wide (wrapping breaks the cursor-up math):
    descriptions truncate, the command labels survive."""
    monkeypatch.setattr("agent6.ui.cli._menu_input._width", lambda: 20)
    _, out = _run(["tab", "enter"])
    assert "/detach" in out  # labels intact
    for row in out.split("\r\n")[1:]:
        visible = row.split("\x1b[2m")[-1].split("\x1b[22m")[0]
        assert len(visible) < 20  # descriptions clamped


def test_input_row_never_exceeds_the_terminal_width(monkeypatch: pytest.MonkeyPatch) -> None:
    """The 57-col real prompt overflowed narrow terminals (only the typed line
    was windowed, with an 8-col floor): the wrapped row broke the cursor-up
    math and every keystroke walked a garbled menu down the screen. The WHOLE
    row -- prompt clamped first, line windowed into the remainder -- must fit
    width-1, like the menu rows already do."""
    from agent6.ui.cli._menu_input import _Reader  # pyright: ignore[reportPrivateUsage]
    from agent6.ui.cli._steer_menu import PROMPT

    for width in (20, 50, 60, 67, 120):
        monkeypatch.setattr("agent6.ui.cli._menu_input._width", lambda w=width: w)
        r = _Reader(PROMPT, MENU_COMMANDS, [])
        r.line = "/status"
        r.cur = len(r.line)
        out: list[str] = []
        r.render(out.append)
        first = "".join(out).split("\r\n")[0]
        printable = re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", first).replace("\r", "")
        assert len(printable) <= width - 1, (width, len(printable), printable)


def test_read_key_decodes_bytes_from_a_pipe() -> None:
    """The raw decoder: control keys, CSI sequences, bare Esc, UTF-8 text."""
    r, w = os.pipe()
    try:
        cases = [
            (b"\t", "tab"),
            (b"\r", "enter"),
            (b"\x7f", "backspace"),
            (b"\x03", "interrupt"),
            (b"\x1b[A", "up"),
            (b"\x1b[Z", "backtab"),
            (b"\x1b[3~", "delete"),
            (b"\x12", "history-search"),
            (b"q", "char:q"),
            ("é".encode(), "char:é"),
        ]
        for raw, expected in cases:
            os.write(w, raw)
            assert _read_key(r) == expected, raw
        os.write(w, b"\x1b")  # bare Esc: resolved by the 30ms poll timing out
        assert _read_key(r) == "esc"
    finally:
        os.close(r)
        os.close(w)


def test_the_default_writer_reaches_the_terminal_under_the_guard(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`cli_main` wraps stdout in a scrubber that drops cursor movement, and
    the composer's menu lost its cursor-up (every render garbled). The
    composer writes to the stream under the wrapper, its rows scrubbed."""
    raw = io.StringIO()
    monkeypatch.setattr(sys, "stdout", ScrubbedStream(raw))
    it = iter(["tab", "enter"])
    menu_input("P> ", {"/a": "x\x1b]52;c;aGVsbG8=\x07y", "/b": "z"}, [], read_key=lambda: next(it))
    out = raw.getvalue()
    assert re.search(r"\x1b\[\d+A", out), out
    assert "xy" in out and "\x1b]" not in out and "\x07" not in out
