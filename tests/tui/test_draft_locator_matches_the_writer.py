# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The TUI finds a draft where `machine create` writes one.

Creating a machine from the hub spawns `agent6 machine create` and locates the
new draft by diffing the draft directory before and after. That is a two-sided
contract with nothing checking it: move the write and the locator sees no new
directory, so the TUI reports "could not start machine create" over a draft
that started fine.
"""

from __future__ import annotations

import inspect
from pathlib import Path

from agent6.app.machine import create as create_mod
from agent6.sessions.layout import bucket_dir
from agent6.types import session_bucket
from agent6.ui.tui.machines import _list_drafts  # pyright: ignore[reportPrivateUsage]


def test_the_locator_reads_the_directory_the_writer_writes(tmp_path: Path) -> None:
    written = bucket_dir(tmp_path, session_bucket("machine")) / "eager-forge-AAAAAA"
    written.mkdir(parents=True)

    assert _list_drafts(tmp_path) == [written]


def test_the_writer_still_derives_that_directory_the_same_way() -> None:
    """Reading the source, because the write happens deep inside a command that
    spawns an authoring agent: the point is that neither side spells the path
    out for itself."""
    source = inspect.getsource(create_mod)
    assert "bucket_dir(state, bucket)" in source
    assert 'session_bucket("machine")' in source


def test_an_empty_state_dir_lists_nothing(tmp_path: Path) -> None:
    assert _list_drafts(tmp_path) == []
