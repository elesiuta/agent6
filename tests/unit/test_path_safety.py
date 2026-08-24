# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The in-process tools' path containment (`tools/_path_safety`)."""

from __future__ import annotations

import os
import signal
from pathlib import Path

import pytest

from agent6.tools._path_safety import (
    SafePath,
    Workspace,
    contain,
    list_contained,
    open_contained,
    read_contained,
    write_contained,
)
from agent6.tools.dispatch import ToolError


def test_contain_refuses_an_uncontained_relative_path(tmp_path: Path) -> None:
    """Containment is the walk, and the walk cannot express `..`: every caller
    resolves first, so the invariant was a nine-caller convention. Held by the
    SafePath now, so a caller that forgets is refused instead of walking out."""
    (tmp_path / "root").mkdir()
    (tmp_path / "outside.txt").write_text("host\n", encoding="utf-8")
    with pytest.raises(ToolError, match=r"\.\."):
        contain(tmp_path / "root", "../outside.txt")


def test_contain_refuses_an_absolute_path(tmp_path: Path) -> None:
    """An absolute rel_path drops the base entirely (pathlib's join rule), so
    the fd would be on a host file no containment check ever saw."""
    (tmp_path / "root").mkdir()
    with pytest.raises(ToolError, match="Absolute"):
        contain(tmp_path / "root", "/etc/hostname")


@pytest.mark.parametrize("rel", ["/etc/hostname", "../outside.txt"])
def test_open_contained_re_checks_a_hand_built_safe_path(tmp_path: Path, rel: str) -> None:
    """The walk keeps its own `..`/absolute guard rather than trusting the
    SafePath: containment must hold even for one built directly, since the type
    is constructible without going through `contain` or a `Workspace`."""
    (tmp_path / "root").mkdir()
    forged = SafePath(base=tmp_path / "root", rel_path=Path(rel), abs_path=Path(rel))
    with pytest.raises(ToolError):
        open_contained(forged, os.O_RDONLY)


def test_open_contained_reads_a_contained_path(tmp_path: Path) -> None:
    (tmp_path / "root" / "sub").mkdir(parents=True)
    (tmp_path / "root" / "sub" / "f.txt").write_text("ok\n", encoding="utf-8")
    fd = open_contained(contain(tmp_path / "root", "sub/f.txt"), os.O_RDONLY)
    with os.fdopen(fd, encoding="utf-8") as handle:
        assert handle.read() == "ok\n"


def test_a_leaf_swapped_for_a_fifo_cannot_block_a_read(tmp_path: Path) -> None:
    """`is_file()` then open is two lookups, and O_NOFOLLOW stops a symlink but
    not a FIFO.

    A jailed background command can swap a regular file for a FIFO in that
    window; the read tools run IN-PROCESS as the operator, so opening one with
    no writer parked the whole agent with nothing to show for it. The open is
    O_NONBLOCK now and the kind is checked by `fstat` on the descriptor just
    opened -- one lookup, so there is no window to swap in.
    """
    root = tmp_path / "ws"
    root.mkdir()
    os.mkfifo(root / "notes.txt")
    ws = Workspace(root=root)

    def _timeout(*_a: object) -> None:
        raise AssertionError("the contained read blocked on a FIFO")

    signal.signal(signal.SIGALRM, _timeout)
    signal.alarm(5)
    try:
        with pytest.raises(ToolError, match="Not a regular file"):
            read_contained(ws.resolve_read("notes.txt"))
        # The write side blocks the same way (O_WRONLY on a reader-less FIFO).
        with pytest.raises(ToolError, match="Not a regular file"):
            write_contained(ws.resolve_write("notes.txt"), "payload")
    finally:
        signal.alarm(0)

    # Regular files and directory listings are untouched.
    (root / "ok.txt").write_text("hello\n", encoding="utf-8")
    assert read_contained(ws.resolve_read("ok.txt")) == "hello\n"
    (root / "sub").mkdir()
    assert [e.name for e in list_contained(ws.resolve_read("sub"))] == []


def test_a_harness_owned_file_is_readable_but_never_writable(tmp_path: Path) -> None:
    """DECISIONS.md sits inside the memory grant (the model may read it) but
    is harness-owned: an in-process write refuses, loudly."""
    from agent6.config import Config
    from agent6.tools import ToolError
    from agent6.tools.policy import workspace_for

    mem = tmp_path / "state" / "memory"
    mem.mkdir(parents=True)
    (mem / "DECISIONS.md").write_text("- ruling\n", encoding="utf-8")
    (mem / "MEMORY.md").write_text("", encoding="utf-8")
    ws = workspace_for(Config(), tmp_path, memory_dir=mem)
    assert ws.resolve_read(str(mem / "DECISIONS.md")).abs_path == (mem / "DECISIONS.md").resolve()
    ws.resolve_write(str(mem / "MEMORY.md"))
    with pytest.raises(ToolError, match="harness-owned"):
        ws.resolve_write(str(mem / "DECISIONS.md"))
