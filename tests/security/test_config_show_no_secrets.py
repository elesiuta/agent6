# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""`agent6 config show` never prints secret material: not the api_key from
secrets.toml, not the value of an `api_key_env` variable. Structurally the
secrets file never enters `Config`; this pins that a future show/render change
cannot start leaking it."""

from __future__ import annotations

from pathlib import Path

import pytest

from agent6.ui.cli.config_cmds import _cmd_config_show  # pyright: ignore[reportPrivateUsage]

_SECRET = "sk-super-secret-do-not-print"
_ENV_SECRET = "env-secret-also-hidden"


@pytest.fixture
def config_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / "agent6-config"
    (home / "agent6").mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(home))
    monkeypatch.chdir(tmp_path)  # no repo config in play
    (home / "agent6" / "config.toml").write_text(
        "\n".join(
            (
                "[agent6]",
                "config_version = 1",
                "[providers.anthropic]",
                'api_format = "anthropic"',
                'api_key_env = "AGENT6_TEST_KEY"',
            )
        ),
        encoding="utf-8",
    )
    secrets = home / "agent6" / "secrets.toml"
    secrets.write_text(f'[providers.anthropic]\napi_key = "{_SECRET}"\n', encoding="utf-8")
    secrets.chmod(0o600)
    monkeypatch.setenv("AGENT6_TEST_KEY", _ENV_SECRET)
    return home


def test_config_show_prints_no_secret_values(
    config_home: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    for as_json in (False, True):
        assert _cmd_config_show(None, as_json=as_json) == 0
        out = capsys.readouterr().out
        assert _SECRET not in out
        assert _ENV_SECRET not in out
        assert "api_key_env" in out  # the POINTER to the secret stays visible
