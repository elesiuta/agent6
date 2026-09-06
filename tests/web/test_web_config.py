# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The `[web]` config section: secure by default (loopback), non-loopback opt-in."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from agent6.config import Config, WebConfig


def test_web_defaults_are_loopback() -> None:
    w = WebConfig()
    assert w.host == "127.0.0.1"
    assert w.port == 7658
    assert w.allow_non_loopback is False


def test_config_carries_web_section() -> None:
    assert Config().web.host == "127.0.0.1"


@pytest.mark.parametrize("host", ["127.0.0.1", "127.0.0.2", "::1", "[::1]", "LOCALHOST"])
def test_loopback_hosts_need_no_optin(host: str) -> None:
    assert WebConfig(host=host).host == host


def test_non_loopback_rejected_without_optin() -> None:
    with pytest.raises(ValidationError, match="allow_non_loopback"):
        WebConfig(host="0.0.0.0")


def test_non_loopback_allowed_with_optin() -> None:
    w = WebConfig(host="0.0.0.0", allow_non_loopback=True)
    assert w.host == "0.0.0.0"


def test_port_flag_is_held_to_the_same_bounds_as_the_config_leaf(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """`--port` goes through `[web].port`'s bounds.

    The flag skipped the schema the leaf is held to, so `--port 99999` reached
    `bind()` as an OverflowError crash report and `--port 0` bound an ephemeral
    port while printing the unreachable `:0` as the URL."""
    from agent6.ui.cli.web_cmds import _cmd_web  # pyright: ignore[reportPrivateUsage]

    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "g"))
    monkeypatch.chdir(tmp_path)
    for bad in (99999, 0, -1):
        assert _cmd_web("", config_path=None, host=None, port=bad, allow_non_loopback=False) == 2
        assert "--port" in capsys.readouterr().err
