# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""An ask's session digest over a run whose merge added nothing."""

from __future__ import annotations

import subprocess
from pathlib import Path

from agent6.sessions.manifest import NO_MERGE_COMMIT, MergeStamp, SessionManifest
from agent6.ui.cli._ask import _diff_via_merge_stamp  # pyright: ignore[reportPrivateUsage]


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args], check=True, capture_output=True, text=True
    ).stdout.strip()


def test_an_ask_over_a_noop_merged_run_diffs_its_tip(tmp_path: Path) -> None:
    """A merge that added nothing stamps the all-zero sentinel; the digest
    read it as a merge commit and asked git for the range 000...^..000...,
    which names no commit at all."""
    _git(tmp_path, "init", "-q", "-b", "main")
    _git(tmp_path, "config", "user.email", "t@t")
    _git(tmp_path, "config", "user.name", "t")
    (tmp_path / "README.md").write_text("base\n", encoding="utf-8")
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-q", "-m", "init")
    base = _git(tmp_path, "rev-parse", "HEAD")
    (tmp_path / "a.txt").write_text("a\n", encoding="utf-8")
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-q", "-m", "add a")
    tip = _git(tmp_path, "rev-parse", "HEAD")
    manifest = SessionManifest(
        session_id="run-NOOP11",
        base_sha=base,
        run_branch="agent6/run-NOOP11",
        merged=MergeStamp(into="main", sha=NO_MERGE_COMMIT, tip=tip),
    )

    got = _diff_via_merge_stamp(tmp_path, manifest, base, "agent6/run-NOOP11")

    assert got is not None
    label, rc, diff, _err = got
    assert rc == 0 and "a.txt" in diff, (rc, diff)
    assert "merged without a commit" in label and NO_MERGE_COMMIT[:12] not in label


def test_a_noop_stamp_with_no_tip_names_nothing_to_diff(tmp_path: Path) -> None:
    """An older record carries the sentinel and no tip: an empty range end
    would have git diff the base against its own HEAD under a "merged" label."""
    _git(tmp_path, "init", "-q", "-b", "main")
    _git(tmp_path, "config", "user.email", "t@t")
    _git(tmp_path, "config", "user.name", "t")
    (tmp_path / "README.md").write_text("base\n", encoding="utf-8")
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-q", "-m", "init")
    base = _git(tmp_path, "rev-parse", "HEAD")
    manifest = SessionManifest(
        session_id="run-OLD11",
        base_sha=base,
        run_branch="agent6/run-OLD11",
        merged=MergeStamp(into="main", sha=NO_MERGE_COMMIT),
    )
    assert _diff_via_merge_stamp(tmp_path, manifest, base, "agent6/run-OLD11") is None
