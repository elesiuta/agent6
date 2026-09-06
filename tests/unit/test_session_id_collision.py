# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""An explicit --session-id held by another bucket is refused up front.

Ids are one public namespace. Before this refusal, `run --session-id demo`
beside an existing plan `demo` started fine, then every read surface fell
apart: the CLI resolver refused the id as ambiguous, listings showed two
identical rows, and the web silently picked whichever bucket came first.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from agent6.config import Config
from agent6.paths import state_dir


def _init_repo(path: Path) -> None:
    subprocess.run(["git", "init", "-q", "-b", "main", str(path)], check=True)
    subprocess.run(["git", "-C", str(path), "config", "user.email", "t@t"], check=True)
    subprocess.run(["git", "-C", str(path), "config", "user.name", "t"], check=True)
    (path / "README.md").write_text("hi\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(path), "add", "README.md"], check=True)
    subprocess.run(["git", "-C", str(path), "commit", "-q", "-m", "init"], check=True)


@pytest.fixture
def repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    gdir = tmp_path / "cfg"
    (gdir / "agent6").mkdir(parents=True, exist_ok=True)
    (gdir / "agent6" / "config.toml").write_text(
        '[providers.anthropic]\napi_format = "anthropic"\n'
        '[models.worker]\nprovider = "anthropic"\nmodel = "claude-x"\n',
        encoding="utf-8",
    )
    monkeypatch.setenv("XDG_CONFIG_HOME", str(gdir))
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    repo = tmp_path / "repo"
    _init_repo(repo)
    monkeypatch.chdir(repo)
    return repo


def _load_cfg() -> Config:
    from agent6.config.layer import load_effective

    return load_effective(Path.cwd(), None).config


def test_run_refuses_an_explicit_id_held_by_another_bucket(
    repo: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from agent6.app.run import run_task

    state = state_dir(repo)
    (state / "sessions" / "plans" / "demo").mkdir(parents=True)

    rc = run_task(_load_cfg(), "do a thing", frontend=MagicMock(), session_id="demo", mode="run")

    assert rc == 2
    err = capsys.readouterr().err
    assert "plans/" in err and "unique across every bucket" in err
    # Nothing was created under runs/: the refusal fired before any state.
    assert not (state / "sessions" / "runs" / "demo").exists()
