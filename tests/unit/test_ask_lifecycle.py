# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""An ask is a session like any other: findable, resumable, still an ask."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent6.config import Config
from agent6.paths import state_dir
from agent6.sessions.layout import session_layout
from agent6.sessions.manifest import ManifestError, read_manifest


def _session(state: Path, bucket: str, sid: str, mode: str) -> Path:
    d = state / "sessions" / bucket / sid
    d.mkdir(parents=True)
    (d / "manifest.json").write_text(
        json.dumps({"version": 3, "mode": mode, "session_id": sid}), encoding="utf-8"
    )
    return d


def test_an_ask_id_resolves_to_its_own_bucket(tmp_path: Path) -> None:
    """Resume looked only under runs/, so an ask could not be continued at all."""
    _session(tmp_path, "asks", "quiet-fox-AAAAAA", "ask")
    _session(tmp_path, "runs", "brave-elk-BBBBBB", "run")
    ask = session_layout(tmp_path, "quiet-fox-AAAAAA")
    run = session_layout(tmp_path, "brave-elk-BBBBBB")
    assert ask is not None and ask.subdir == "asks"
    assert run is not None and run.subdir == "runs"
    assert ask.session_dir.is_dir()


def test_a_unique_prefix_resolves_across_buckets(tmp_path: Path) -> None:
    _session(tmp_path, "asks", "quiet-fox-AAAAAA", "ask")
    found = session_layout(tmp_path, "quiet-fox")
    assert found is not None and found.session_id == "quiet-fox-AAAAAA"


def test_an_ambiguous_prefix_resolves_to_nothing_rather_than_guessing(tmp_path: Path) -> None:
    _session(tmp_path, "asks", "quiet-fox-AAAAAA", "ask")
    _session(tmp_path, "runs", "quiet-fox-BBBBBB", "run")
    assert session_layout(tmp_path, "quiet-fox") is None


def test_an_unknown_id_resolves_to_nothing(tmp_path: Path) -> None:
    assert session_layout(tmp_path, "nope") is None
    assert session_layout(tmp_path, "") is None


def test_ask_is_a_mode_resume_and_fork_may_act_on(tmp_path: Path) -> None:
    """The privilege gate refused "ask" outright, so an ask was a dead end: no
    resume, no fork. It is LESS privileged than plan, not unknown."""
    d = _session(tmp_path, "asks", "quiet-fox-AAAAAA", "ask")
    assert read_manifest(d).session_mode() == "ask"


def test_an_unknown_mode_is_still_refused(tmp_path: Path) -> None:
    """The gate's whole point: a damaged manifest must not fall open to the
    privileged write mode."""
    d = _session(tmp_path, "asks", "odd-AAAAAA", "wat")
    with pytest.raises(ManifestError, match="unknown session mode"):
        read_manifest(d).session_mode()


def test_a_resumed_ask_is_still_clamped() -> None:
    """The clamp lives with the mode, not with one lifecycle, so continuing an
    ask cannot hand it the auto-approval a fresh one never had."""
    from agent6.app._setup import session_config

    cfg = Config.model_validate({"sandbox": {"run_commands": "yes"}})
    assert session_config(cfg, "ask").sandbox.run_commands == "ask"
    assert session_config(cfg, "run").sandbox.run_commands == "yes"


def test_a_run_can_be_seeded_from_an_ask(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The direction the operator asked for: work something out in an ask, then run it.
    The ask is untouched -- seeding starts a NEW session, unlike fork."""
    import json

    from agent6.config import Config
    from agent6.ui.cli.run import _compose_task  # pyright: ignore[reportPrivateUsage]

    monkeypatch.chdir(tmp_path)
    ask = state_dir(tmp_path) / "sessions" / "asks" / "quiet-fox-AAAAAA"
    ask.mkdir(parents=True)
    (ask / "manifest.json").write_text(
        json.dumps({"version": 3, "mode": "ask", "user_task": "how do I convert h264"}),
        encoding="utf-8",
    )
    (ask / "logs.jsonl").write_text(
        json.dumps({"type": "session.end", "reason": "answered", "iterations": 1}) + "\n",
        encoding="utf-8",
    )
    task, err = _compose_task("do it", Config(), skills=(), seed_from="quiet-fox-AAAAAA")
    assert err == ""
    assert "how do I convert h264" in task  # the ask's context came across
    assert task.endswith("do it")  # the operator's new task is what it ends on
    assert (ask / "manifest.json").exists()  # the source is untouched


def test_seeding_from_an_unknown_session_fails_loudly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from agent6.config import Config
    from agent6.ui.cli.run import _compose_task  # pyright: ignore[reportPrivateUsage]

    monkeypatch.chdir(tmp_path)
    _task, err = _compose_task("do it", Config(), skills=(), seed_from="nope")
    assert "could not seed" in err
