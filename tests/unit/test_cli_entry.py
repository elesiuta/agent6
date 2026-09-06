# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The guarded console-script entry point `cli_main`: the boundary that sorts
failures by fault. An OperatorError refuses at exit 2 with no traceback;
anything else is a bug and crash-reports at exit 1."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from agent6.config import ConfigError
from agent6.errors import OperatorError
from agent6.ui import cli
from agent6.ui.cli import cli_main

_CRASH_MARKERS = ("unexpected", "full traceback", "report this")


def test_cli_main_passes_through_return_code(monkeypatch: pytest.MonkeyPatch) -> None:
    def _ok(_argv: list[str] | None = None) -> int:
        return 3

    monkeypatch.setattr(cli, "main", _ok)
    assert cli_main() == 3


def test_cli_main_converts_unexpected_exception_to_friendly_error(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def _boom(_argv: list[str] | None = None) -> int:
        raise RuntimeError("kaboom")

    monkeypatch.setattr(cli, "main", _boom)
    monkeypatch.delenv("AGENT6_DEBUG", raising=False)
    rc = cli_main()
    assert rc == 1
    err = capsys.readouterr().err
    assert "ERROR: unexpected RuntimeError: kaboom" in err
    # Points at a saved traceback that actually exists and contains the stack.
    tb_line = next(line for line in err.splitlines() if "full traceback:" in line)
    tb_path = Path(tb_line.split("full traceback:", 1)[1].strip())
    assert tb_path.is_file()
    assert "RuntimeError: kaboom" in tb_path.read_text(encoding="utf-8")
    tb_path.unlink()


def test_an_operator_error_refuses_without_a_crash_report(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The whole contract: a bad value or unreadable file from the operator
    raises OperatorError; cli_main turns that into `ERROR:` + exit 2, and
    everything else into a crash report."""

    def _bad(_argv: list[str] | None = None) -> int:
        raise OperatorError("no such machine file: overlay.toml")

    monkeypatch.setattr(cli, "main", _bad)
    monkeypatch.delenv("AGENT6_DEBUG", raising=False)
    assert cli_main() == 2
    captured = capsys.readouterr()
    assert captured.err == "ERROR: no such machine file: overlay.toml\n"


def test_a_config_error_is_an_operator_error(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A malformed config is the operator's, not a bug in agent6: ConfigError
    subclasses OperatorError, so every reader that raises it gets the refusal
    surface without its own except arm."""

    def _bad(_argv: list[str] | None = None) -> int:
        raise ConfigError("Config file is not valid TOML (/x/config.toml): line 1")

    monkeypatch.setattr(cli, "main", _bad)
    monkeypatch.delenv("AGENT6_DEBUG", raising=False)
    assert cli_main() == 2
    err = capsys.readouterr().err
    assert err.startswith("ERROR: ")
    assert "not valid TOML" in err
    assert not any(marker in err for marker in _CRASH_MARKERS)


def test_an_unreadable_config_file_refuses_end_to_end(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Root-owned after a sudo run, or plain chmod 000: the named file and the
    OS reason reach the operator, with no crash-report language anywhere."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "g"))
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "s"))
    monkeypatch.delenv("AGENT6_DEBUG", raising=False)
    bad = tmp_path / "c.toml"
    bad.write_text("x = 1\n", encoding="utf-8")
    bad.chmod(0o000)
    try:
        rc = cli_main(["--config", str(bad), "config", "show"])
    finally:
        bad.chmod(0o600)
    assert rc == 2
    err = capsys.readouterr().err
    assert err.startswith("ERROR: ")
    assert "c.toml" in err
    assert not any(marker in err for marker in _CRASH_MARKERS)


def test_a_bad_budget_flag_refuses_end_to_end(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """`run --max-usd inf` names the flag it refuses, at exit 2, not a saved
    ValidationError traceback and an invitation to file a bug."""
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)  # past the git wall
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "g"))
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "s"))
    monkeypatch.delenv("AGENT6_DEBUG", raising=False)
    rc = cli_main(["run", "task", "--max-usd", "inf"])
    assert rc == 2
    err = capsys.readouterr().err
    assert err.startswith("ERROR: ")
    assert "--max-usd" in err
    assert not any(marker in err for marker in _CRASH_MARKERS)


def test_cli_main_reraises_under_debug(monkeypatch: pytest.MonkeyPatch) -> None:
    def _boom(_argv: list[str] | None = None) -> int:
        raise RuntimeError("kaboom")

    monkeypatch.setattr(cli, "main", _boom)
    monkeypatch.setenv("AGENT6_DEBUG", "1")
    with pytest.raises(RuntimeError, match="kaboom"):
        cli_main()


def test_cli_main_handles_keyboard_interrupt(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def _interrupt(_argv: list[str] | None = None) -> int:
        raise KeyboardInterrupt

    monkeypatch.setattr(cli, "main", _interrupt)
    assert cli_main() == 130
    assert "interrupted" in capsys.readouterr().err
