# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The web composer's slash completion mirrors directive.STEER_COMMANDS
verbatim (the client is static JS, so the shared table is pinned, not
imported), and /compact, /btw and /now stay live-session offers."""

from __future__ import annotations

from importlib import resources

from agent6.directive import STEER_COMMANDS

CLIENT_JS = resources.files("agent6.ui.web").joinpath("client.js").read_text(encoding="utf-8")


def test_client_mirrors_the_steer_commands_verbatim() -> None:
    for cmd, help_ in STEER_COMMANDS.items():
        assert f"['{cmd}', '{help_}']" in CLIENT_JS, f"client.js drifted from {cmd}"


def test_compact_is_gated_on_live() -> None:
    assert "liveNow() || (c !== '/compact' && c !== '/btw' && c !== '/now')" in CLIENT_JS
    assert "() => finished === false" in CLIENT_JS  # the composer's live truth feeds the gate
