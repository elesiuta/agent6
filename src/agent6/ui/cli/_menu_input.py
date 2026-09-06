# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""A small fish-style line reader for the pause menu.

Type to steer; Tab previews the matching slash commands in a menu below the
line (with their descriptions) and cycles through them, Shift-Tab cycles
backwards, Up/Down move the selection, Esc restores what was typed. Without
a menu open, Up/Down recall history, and Ctrl-R searches it: matches render
below the line like the Tab menu, typing narrows the query, Ctrl-R/arrows
move the selection, Enter or Tab keeps the highlighted message for editing,
Esc restores what was typed. Enter accepts, Ctrl-C stops the run, Ctrl-D on
an empty line continues it.

Hand-rolled on termios because neither readline flavor can render this: GNU
readline's menu-complete cycles blind (no list until a second Tab, never with
descriptions) and libedit -- what uv-managed CPython 3.12 links -- supports
only plain rl_complete. A dependency would be out of proportion for one
prompt. Unix-only by design: callers gate on :func:`menu_capable` and fall
back to a plain prompt elsewhere.

Rendering uses CR, erase-below, cursor-up/right, reverse and dim only, safe
under tmux/byobu. The input row (prompt clamped, line windowed into the
remainder) fits the terminal width so the redraw never wraps (wrapping would
break the cursor-up arithmetic).
"""

from __future__ import annotations

import os
import select
import sys
from collections.abc import Callable
from typing import TextIO

from agent6.ui.cli._terminal_guard import raw_stream
from agent6.viewmodel.transcript import scrub_terminal_controls

try:  # unix only; Windows callers gate on menu_capable()
    import termios
    import tty
except ImportError:  # pragma: no cover - exercised only on Windows
    termios = None  # type: ignore[assignment]
    tty = None  # type: ignore[assignment]

_CSI_FINAL = {
    b"A": "up",
    b"B": "down",
    b"C": "right",
    b"D": "left",
    b"Z": "backtab",
    b"H": "home",
    b"F": "end",
}
_TILDE_SEQ = {b"1~": "home", b"7~": "home", b"4~": "end", b"8~": "end", b"3~": "delete"}
_CTRL = {
    b"\r": "enter",
    b"\n": "enter",
    b"\t": "tab",
    b"\x7f": "backspace",
    b"\x08": "backspace",
    b"\x03": "interrupt",
    b"\x04": "eof",
    b"\x01": "home",  # Ctrl-A
    b"\x05": "end",  # Ctrl-E
    b"\x15": "kill-line",  # Ctrl-U
    b"\x17": "kill-word",  # Ctrl-W
    b"\x0b": "kill-to-end",  # Ctrl-K
    b"\x0c": "redraw",  # Ctrl-L
    b"\x12": "history-search",  # Ctrl-R
}

# History search renders only the newest matches; typing narrows to the rest.
_SEARCH_ROWS = 8
_SEARCH_PROMPT = "search: "


def menu_capable() -> bool:
    """True when the fish-style reader can own the terminal line: termios
    exists (Unix) and both std streams are the interactive terminal."""
    return termios is not None and sys.stdin.isatty() and sys.stdout.isatty()


class LineSuperseded(Exception):
    """The line being read was answered by another route: the reader's
    *until* held while it waited for a key."""


def read_line_until(stream: TextIO, fd: int, until: Callable[[], bool] | None) -> str | None:
    """One line from *stream*, or None at EOF or once *until* holds (polled
    every 0.2 s while nothing is typed; a line already complete wins)."""
    if until is not None:
        while not until():
            ready, _, _ = select.select([fd], [], [], 0.2)
            if ready:
                break
        else:
            return None
    line = stream.readline()
    return line.rstrip("\n") if line else None


def _read_key_until(fd: int, until: Callable[[], bool]) -> str:
    """`_read_key`, polling *until* every 0.2 s while nothing is typed."""
    while not select.select([fd], [], [], 0.2)[0]:
        if until():
            raise LineSuperseded
    return _read_key(fd)


def _read_escape(fd: int) -> str:
    """The rest of an ESC-initiated key: a name from the tables above, "esc"
    for a bare Escape (or Alt-chord, whose modified key is dropped), "" for
    sequences we ignore."""
    # Distinguish a bare Esc from an escape sequence by a short poll.
    ready, _, _ = select.select([fd], [], [], 0.03)
    if not ready:
        return "esc"
    lead = os.read(fd, 1)
    if lead not in (b"[", b"O"):
        return "esc"
    seq = b""
    while len(seq) < 8:
        ch = os.read(fd, 1)
        if not ch:
            break
        seq += ch
        if 0x40 <= ch[0] <= 0x7E:  # a CSI final byte
            break
    if seq[-1:] == b"~":
        return _TILDE_SEQ.get(seq, "")
    return _CSI_FINAL.get(seq[-1:], "")


def _read_key(fd: int) -> str:
    """One logical key from *fd*: a name from the tables above, `char:<c>`
    for text (UTF-8 decoded), `""` for keys we ignore."""
    data = os.read(fd, 1)
    if not data:
        return "eof"
    if data in _CTRL:
        return _CTRL[data]
    b = data[0]
    if b == 0x1B:
        return _read_escape(fd)
    if b < 0x20:
        return ""  # other control keys: ignore
    if b >= 0xC0:  # UTF-8 lead byte: read the continuation bytes
        n = 1 if b < 0xE0 else 2 if b < 0xF0 else 3
        data += os.read(fd, n)
    return "char:" + data.decode("utf-8", errors="replace")


def _width() -> int:
    try:
        cols = os.get_terminal_size(sys.stdout.fileno()).columns
    except OSError:
        return 80
    # A pty can report 0x0 (no winsize set); treat it like no terminal at all.
    return cols if cols > 0 else 80


class _Reader:
    """One menu_input call's state; split from the loop so each key handler
    stays a small method instead of one long branch pile."""

    def __init__(self, prompt: str, commands: dict[str, str], history: list[str]) -> None:
        self.prompt = prompt
        self.commands = commands
        self.history = history
        self.line = ""
        self.cur = 0  # cursor index into self.line
        self.menu: list[str] | None = None
        self.sel = 0
        self.stem = ""  # what was typed before the menu opened (Esc restores)
        self.hist_idx = len(history)
        self.draft = ""  # the unsubmitted line saved when history recall starts
        self.searching = False  # Ctrl-R mode: the line is the query, hits render below
        self.hits: list[str] = []
        self.more = 0  # matches beyond the rendered cap
        self.saved = ""  # the line before search began (Esc restores)

    # -- rendering ---------------------------------------------------------

    def render(self, write: Callable[[str], None]) -> None:
        width = _width()
        # Budget the WHOLE input row to width-1 like the menu rows: clamp the
        # prompt first, then window the line into the remainder. An overflow
        # floor here wrapped the row on narrow terminals, and the wrapped row
        # broke the cursor-up arithmetic (garbling the menu every keystroke).
        prompt = (_SEARCH_PROMPT if self.searching else self.prompt)[: width - 1]
        avail = max(0, width - 1 - len(prompt))
        start = 0 if self.cur < avail else self.cur - avail + 1
        visible = self.line[start : start + avail]
        out = ["\r\x1b[J", prompt, visible]
        rows, highlight = self._rows()
        # A row's text is data (a history line, a command's blurb), scrubbed
        # here: this writer bypasses the stream's scrubber for its movement.
        rows = [
            (scrub_terminal_controls(label), scrub_terminal_controls(dim)) for label, dim in rows
        ]
        if rows:
            pad = max(len(label) for label, _dim in rows)
            for i, (label, dim) in enumerate(rows):
                # Clamp the VISIBLE text to one row (a wrapped row would break
                # the cursor-up arithmetic); the SGR codes take no columns and
                # must never be sliced through.
                cell = "  " if not label else f"  {label:<{pad}}  "[: width - 1]
                tail = dim[: max(0, width - 1 - len(cell))]
                row = f"{cell}\x1b[2m{tail}\x1b[22m"
                if i == highlight:
                    row = f"\x1b[7m{row}\x1b[27m"
                out.append("\r\n" + row)
            out.append(f"\x1b[{len(rows)}A")
        col = len(prompt) + (self.cur - start)
        out.append("\r" + (f"\x1b[{col}C" if col else ""))
        write("".join(out))

    def _rows(self) -> tuple[list[tuple[str, str]], int]:
        """The rows under the input line as (label, dim tail) pairs, plus the
        highlighted index (-1 none). Search markers are all-dim: empty label."""
        if self.searching:
            if not self.hits:
                return [("", "(no match)")], -1
            rows: list[tuple[str, str]] = [(hit, "") for hit in self.hits]
            if self.more:
                rows.append(("", f"… {self.more} more (type to narrow)"))
            return rows, self.sel
        if self.menu is not None:
            return [(cmd, self.commands[cmd]) for cmd in self.menu], self.sel
        return [], -1

    def close_rows(self, write: Callable[[str], None]) -> None:
        """Leave the accepted/abandoned line in scrollback with the menu erased."""
        write(f"\r\x1b[J{self.prompt}{self.line}\r\n")

    # -- completion menu ---------------------------------------------------

    def open_menu(self, write: Callable[[str], None]) -> None:
        # Same word rule as the dispatch: only the first (and only) word of a
        # line completes; inside steer text Tab is inert.
        if " " in self.line or (self.line and not self.line.startswith("/")):
            write("\a")
            return
        matches = [c for c in self.commands if c.startswith(self.line)]
        if not matches:
            write("\a")
            return
        if len(matches) == 1:
            self.line = matches[0]
            self.cur = len(self.line)
            return
        self.stem = self.line
        self.menu = matches
        self.select(0)

    def select(self, i: int) -> None:
        assert self.menu is not None
        self.sel = i % len(self.menu)
        self.line = self.menu[self.sel]
        self.cur = len(self.line)

    def dismiss_menu(self, *, restore: bool) -> None:
        if restore:
            self.line = self.stem
            self.cur = len(self.line)
        self.menu = None

    # -- history search ----------------------------------------------------

    def open_search(self, write: Callable[[str], None]) -> None:
        """Ctrl-R: search history, with the line as the live query."""
        if not self.history:
            write("\a")
            return
        if self.menu is not None:
            self.dismiss_menu(restore=False)
        self.saved = self.line
        self.line = ""
        self.cur = 0
        self.searching = True
        self.refilter()

    def refilter(self) -> None:
        """Newest-first case-insensitive substring matches; repeats collapse."""
        q = self.line.lower()
        matches = list(dict.fromkeys(h for h in reversed(self.history) if q in h.lower()))
        self.hits = matches[:_SEARCH_ROWS]
        self.more = len(matches) - len(self.hits)
        self.sel = 0

    def close_search(self, line: str) -> None:
        self.line = line
        self.cur = len(line)
        self.searching = False
        self.hits = []
        self.more = 0

    def _search_key(self, key: str) -> None:
        """A key while searching: the line is the query. Enter/Tab keep the
        highlighted match (the query itself when nothing matches); Esc and
        Ctrl-D restore the pre-search line."""
        if key in ("enter", "tab"):
            self.close_search(self.hits[self.sel] if self.hits else self.line)
        elif key in ("esc", "eof"):
            self.close_search(self.saved)
        elif key in ("history-search", "down"):
            if self.hits:
                self.sel = (self.sel + 1) % len(self.hits)
        elif key in ("up", "backtab"):
            if self.hits:
                self.sel = (self.sel - 1) % len(self.hits)
        elif key.startswith("char:"):
            self.insert(key[5:])
            self.refilter()
        elif key in self._EDIT_KEYS:
            self.edit(key)
            self.refilter()
        # anything else (redraw, unknown): repaint only

    # -- editing -----------------------------------------------------------

    def insert(self, text: str) -> None:
        self.line = self.line[: self.cur] + text + self.line[self.cur :]
        self.cur += len(text)

    def edit(self, key: str) -> None:
        if key == "backspace" and self.cur:
            self.line = self.line[: self.cur - 1] + self.line[self.cur :]
            self.cur -= 1
        elif key == "delete":
            self.line = self.line[: self.cur] + self.line[self.cur + 1 :]
        elif key == "left":
            self.cur = max(0, self.cur - 1)
        elif key == "right":
            self.cur = min(len(self.line), self.cur + 1)
        elif key == "home":
            self.cur = 0
        elif key == "end":
            self.cur = len(self.line)
        elif key == "kill-line":
            self.line = self.line[self.cur :]
            self.cur = 0
        elif key == "kill-to-end":
            self.line = self.line[: self.cur]
        elif key == "kill-word":
            head = self.line[: self.cur].rstrip()
            cut = head.rfind(" ") + 1
            self.line = self.line[:cut] + self.line[self.cur :]
            self.cur = cut

    def recall(self, step: int) -> None:
        if not self.history:
            return
        if self.hist_idx == len(self.history):
            self.draft = self.line
        self.hist_idx = max(0, min(len(self.history), self.hist_idx + step))
        self.line = (
            self.draft if self.hist_idx == len(self.history) else self.history[self.hist_idx]
        )
        self.cur = len(self.line)

    # -- key dispatch --------------------------------------------------------

    _EDIT_KEYS = (
        "backspace",
        "delete",
        "left",
        "right",
        "home",
        "end",
        "kill-line",
        "kill-to-end",
        "kill-word",
    )

    def handle_key(self, key: str, write: Callable[[str], None]) -> bool:
        """Apply one key. True when Enter accepted the line (in `self.line`);
        raises KeyboardInterrupt/EOFError for Ctrl-C / Ctrl-D-on-empty."""
        if key == "interrupt":
            write("\r\n\x1b[J")
            raise KeyboardInterrupt
        if self.searching:
            self._search_key(key)
            return False
        if key == "eof":
            if self.menu is not None or self.line:
                write("\a")
                return False
            self.close_rows(write)
            raise EOFError
        if key == "enter":
            self.menu = None
            self.close_rows(write)
            if self.line.strip() and (not self.history or self.history[-1] != self.line):
                self.history.append(self.line)
            return True
        self._apply(key, write)
        return False

    def _apply(self, key: str, write: Callable[[str], None]) -> None:
        """A non-terminal key: menu navigation, history recall, or an edit."""
        if key == "history-search":
            self.open_search(write)
        elif key == "tab" or (key == "backtab" and self.menu is None):
            if self.menu is None:
                self.open_menu(write)
            else:
                self.select(self.sel + 1)
        elif self.menu is not None and key in ("backtab", "up", "down", "esc"):
            if key == "esc":
                self.dismiss_menu(restore=True)
            else:
                self.select(self.sel + (1 if key == "down" else -1))
        elif key in ("up", "down"):
            self.recall(-1 if key == "up" else 1)
        elif key.startswith("char:") or key in self._EDIT_KEYS:
            # Typing keeps the selected candidate and edits from there.
            self.dismiss_menu(restore=False)
            if key.startswith("char:"):
                self.insert(key[5:])
            else:
                self.edit(key)
        # "", "esc" without a menu, "redraw": nothing to apply, just repaint


def menu_input(
    prompt: str,
    commands: dict[str, str],
    history: list[str],
    *,
    read_key: Callable[[], str] | None = None,
    write: Callable[[str], None] | None = None,
    until: Callable[[], bool] | None = None,
) -> str:
    """Read one line with a fish-style command preview.

    Matches `input()`'s contract: returns the line without the newline,
    raises EOFError on Ctrl-D at an empty line, KeyboardInterrupt on Ctrl-C
    (via SIGINT in cbreak mode, or the `\\x03` byte where signals are off),
    and :class:`LineSuperseded` once *until* holds while nothing is typed.
    Accepted non-empty lines are appended to *history* (deduped against the
    last entry); Ctrl-R searches *history* (Enter/Tab keep the highlighted
    match for editing, Esc cancels). *read_key*/*write* are injectable for
    tests; the real terminal is put in cbreak mode only when *read_key* is
    None.
    """

    def restore() -> None:
        return None

    if read_key is None:
        assert termios is not None and tty is not None, "menu_input needs a Unix terminal"
        tio, drain = termios, termios.TCSADRAIN  # narrowed bindings for the closure
        fd = sys.stdin.fileno()
        old_attrs = tio.tcgetattr(fd)
        tty.setcbreak(fd, drain)

        def restore() -> None:
            tio.tcsetattr(fd, drain, old_attrs)

        def terminal_key() -> str:
            return _read_key(fd) if until is None else _read_key_until(fd, until)

        read_key = terminal_key

    if write is None:

        def _stdout_write(text: str) -> None:
            out = raw_stream(sys.stdout)
            out.write(text)
            out.flush()

        write = _stdout_write

    r = _Reader(prompt, commands, history)
    try:
        r.render(write)
        while True:
            if r.handle_key(read_key(), write):
                return r.line
            r.render(write)
    except (KeyboardInterrupt, LineSuperseded):
        # A signal-delivered Ctrl-C (cbreak keeps ISIG) and a superseded line
        # both raise from inside the read, skipping handle_key's cleanup:
        # erase the menu rows so they don't linger under whatever prints next.
        write("\r\n\x1b[J")
        raise
    finally:
        restore()
