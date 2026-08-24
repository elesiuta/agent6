# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The session snapshot carries the background-shell roster: the web's
Background shells card shows what the CLI `/shells` prints."""

from __future__ import annotations

import json
from pathlib import Path

from agent6.viewmodel.snapshot import session_snapshot


def test_session_snapshot_carries_the_shell_roster(tmp_path: Path) -> None:
    d = tmp_path / "s1"
    shell = d / "shells" / "bg-1"
    shell.mkdir(parents=True)
    (shell / "meta.json").write_text(json.dumps({"command": "make -j"}), encoding="utf-8")
    assert session_snapshot(d)["shells"] == [
        "[bg-1] still running (or the run that owns it ended): make -j"
    ]
    (shell / "result.json").write_text('{"stopped": true}\n', encoding="utf-8")
    assert session_snapshot(d)["shells"] == ["[bg-1] stopped: make -j"]


def test_session_snapshot_without_shells_is_an_empty_roster(tmp_path: Path) -> None:
    d = tmp_path / "s2"
    d.mkdir()
    assert session_snapshot(d)["shells"] == []
