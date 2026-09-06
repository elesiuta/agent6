# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The `[parallel]` config section: defaults + repo override."""

from __future__ import annotations

from pathlib import Path

import pytest

from agent6.config import Config, ParallelConfig
from agent6.config.layer import load_effective
from agent6.paths import repo_config_path


def _write_repo_config(repo: Path, toml: str) -> None:
    p = repo_config_path(repo)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(toml, encoding="utf-8")


@pytest.fixture
def repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    r = tmp_path / "repo"
    r.mkdir()
    return r


def test_parallel_defaults() -> None:
    cfg = Config()
    assert cfg.parallel.max_lanes == 4
    assert cfg.parallel.workdir == ""


def test_parallel_max_lanes_bounds() -> None:
    # le=1024: the cap itself is capped, or a huge max_lanes re-opens the
    # huge-count allocation the spec grammar refuses against.
    with pytest.raises(ValueError):
        ParallelConfig(max_lanes=0)
    with pytest.raises(ValueError):
        ParallelConfig(max_lanes=1025)
    assert ParallelConfig(max_lanes=1024).max_lanes == 1024


def test_parallel_override_via_repo_config(repo: Path) -> None:
    _write_repo_config(repo, '[parallel]\nmax_lanes = 8\nworkdir = "/tmp/lanes"\n')
    cfg = load_effective(repo).config
    assert cfg.parallel.max_lanes == 8
    assert cfg.parallel.workdir == "/tmp/lanes"
