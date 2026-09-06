# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""`agent6` line-buffers a redirected stdout: a run's log file fills as the
run prints, not at exit (a hub-spawned run's log stayed empty for minutes)."""

from __future__ import annotations

import io
import sys
from pathlib import Path

import pytest

from agent6.ui.cli import cli_main


def test_the_cli_line_buffers_a_redirected_stdout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    monkeypatch.chdir(tmp_path)
    out = io.TextIOWrapper(io.BytesIO(), encoding="utf-8", line_buffering=False)
    monkeypatch.setattr(sys, "stdout", out)
    assert cli_main(["sessions", "dir"]) == 0
    assert out.line_buffering is True
