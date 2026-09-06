# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""`sessions transcript --seq` windows: N, N-M, and a reversed M-N refuses
instead of printing an empty transcript with exit 0."""

from __future__ import annotations

import pytest

from agent6.ui.cli.history_cmds import _parse_seq_window  # pyright: ignore[reportPrivateUsage]


def test_seq_window_shapes() -> None:
    assert _parse_seq_window("") is None
    assert _parse_seq_window("5") == (5, 5)
    assert _parse_seq_window("3-7") == (3, 7)


def test_a_reversed_seq_window_is_an_error() -> None:
    with pytest.raises(ValueError, match="reversed"):
        _parse_seq_window("9-3")
