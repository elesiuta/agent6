# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The detached spawners' environment contracts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from agent6.ui import spawn as spawn_mod


def test_hub_machine_spawn_carries_the_wait_away_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`spawn_and_confirm` is the hub surfaces' machine launcher (web + TUI)
    and the detached resume's; without the away marker every prompt raced the
    viewer's registration and the headless default answered instead of the
    operator. A caller's additions ride on top of it."""
    captured: dict[str, Any] = {}

    class _Proc:
        pid = 4242

        def poll(self) -> int | None:
            return None

    def fake_popen(argv: list[str], **kw: Any) -> _Proc:
        captured.update(kw)
        return _Proc()

    monkeypatch.setattr(spawn_mod.subprocess, "Popen", fake_popen)
    err = spawn_mod.spawn_and_confirm(
        ["agent6", "machine", "run"], tmp_path, started=lambda _p: True
    )
    assert err == ""
    assert captured["env"]["AGENT6_DETACHED_AWAY"] == "wait"
    err = spawn_mod.spawn_and_confirm(
        ["agent6", "resume", "r"],
        tmp_path,
        started=lambda _p: True,
        extra_env={"AGENT6_STREAM_TO_LOG": "1"},
    )
    assert err == ""
    assert captured["env"]["AGENT6_DETACHED_AWAY"] == "wait"
    assert captured["env"]["AGENT6_STREAM_TO_LOG"] == "1"
