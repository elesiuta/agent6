# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""`agent6 review` says which of its flags it cannot honour."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from agent6.config import ConfigError
from agent6.ui.cli import cli_main


def test_personas_without_reviewers_is_said_to_be_ignored(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """`--personas` is read only under `--reviewers N`: alone it ran the single
    freeform review with the named seats silently dropped. The sibling
    `model` command prints a note for a flag it cannot use; so does this."""
    subprocess.run(["git", "init", "-q", "-b", "main", str(tmp_path)], check=True)
    monkeypatch.chdir(tmp_path)

    def stop(*_a: object, **_k: object) -> object:
        raise ConfigError("stop here")

    monkeypatch.setattr("agent6.ui.cli.review_cmds.load_effective", stop)
    rc = cli_main(["review", "--personas", "security,tests"])
    err = capsys.readouterr().err
    assert rc == 2 and "stop here" in err
    assert "note: --personas ignored (no --reviewers N" in err

    rc = cli_main(["review", "--personas", "security,tests", "--reviewers", "2"])
    assert rc == 2 and "--personas ignored" not in capsys.readouterr().err


def test_personas_under_configured_seats_is_said_to_be_ignored(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """`[review].seats` names the roster outright, as the flag's help says;
    the flag beside it was dropped in silence."""
    from types import SimpleNamespace

    from agent6.config import Config

    subprocess.run(["git", "init", "-q", "-b", "main", str(tmp_path)], check=True)
    monkeypatch.chdir(tmp_path)
    cfg = Config.model_validate({"review": {"seats": ["security@openrouter/some-model"]}})

    def loaded(*_a: object, **_k: object) -> SimpleNamespace:
        return SimpleNamespace(config=cfg)

    monkeypatch.setattr("agent6.ui.cli.review_cmds.load_effective", loaded)
    rc = cli_main(["review", "--personas", "tests", "--reviewers", "2"])
    err = capsys.readouterr().err
    assert rc == 2 and "note: --personas ignored ([review].seats names the roster)." in err
