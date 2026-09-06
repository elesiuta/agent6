# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Tests for the friendly run-id module."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from agent6.sessions.id import (
    SessionIdError,
    friendly_token,
    resolve_session,
    validate_explicit_session_id,
)

_PATTERN = re.compile(r"^[a-z]+-[a-z]+-[0-9A-Z]{6}$")


def test_validate_explicit_run_id_rejects_traversal() -> None:
    for bad in ("../escape", "..", ".", "a/b", "/abs/path", "x\\y", ""):
        with pytest.raises(SessionIdError):
            validate_explicit_session_id(bad)
    # A normal slug (and the generated shape) passes through unchanged.
    assert validate_explicit_session_id("my-run-1") == "my-run-1"
    assert validate_explicit_session_id(friendly_token())


def test_validate_explicit_run_id_rejects_git_forbidden_names() -> None:
    """The id becomes a branch (`agent6/<id>`) and a chain ref
    (`refs/agent6/<id>/head`); a value git's ref grammar rejects must be refused
    up front, not accepted into a run whose every commit's `update-ref` then
    fails while it reports success. The traversal check alone misses all of
    these (no separator, no dot name)."""
    for bad in (
        "has space",
        "ti~lde",
        "ca^ret",
        "col:on",
        "quest?ion",
        "star*x",
        "brack[et",
        "end.lock",
        "trailing.",
        "dou..ble",
        "at@{brace",
        "-leading",
        ".leading",
    ):
        with pytest.raises(SessionIdError):
            validate_explicit_session_id(bad)


def test_validate_explicit_run_id_accepts_only_ids_git_can_ref(tmp_path: Path) -> None:
    """Whatever the validator accepts must actually work as BOTH refs the run
    builds -- the guarantee the traversal-only check could not make."""
    import subprocess

    def git_accepts(ref: str) -> bool:
        return (
            subprocess.run(
                ["git", "check-ref-format", ref], capture_output=True, cwd=tmp_path, check=False
            ).returncode
            == 0
        )

    for good in ("my-run-1", "sunny-otter-K4Q7B2", "machine-foo", "a.b.c", "UPPER_case-1"):
        assert validate_explicit_session_id(good) == good
        assert git_accepts(f"refs/heads/agent6/{good}"), good  # the run branch
        assert git_accepts(f"refs/agent6/{good}/head"), good  # the chain ref


def test_friendly_token_shape() -> None:
    for _ in range(50):
        rid = friendly_token()
        assert _PATTERN.match(rid), rid


def test_friendly_token_varies() -> None:
    """Catches a constant or an unseeded generator. NOT a uniqueness guarantee:
    within one millisecond the space is ~30M, so 500 draws collide about once
    in 200 -- which is what made the old 500-draw assertion flaky. What must
    never collide is the DIRECTORY, and `_unused_session_id` owns that
    (tests/unit/test_generated_id_collision.py)."""
    seen = {friendly_token() for _ in range(20)}
    assert len(seen) == 20


def test_friendly_token_suffix_time_sortable() -> None:
    """Suffixes from ids minted in order should sort in time order."""
    import time

    suffixes: list[str] = []
    for _ in range(10):
        suffixes.append(friendly_token().rsplit("-", 1)[1])
        time.sleep(0.002)
    assert suffixes == sorted(suffixes)


def _bucket(state_dir: Path, *ids: str) -> None:
    for sid in ids:
        (state_dir / "sessions" / "plans" / sid).mkdir(parents=True)


def test_resolve_exact_match(tmp_path: Path) -> None:
    _bucket(tmp_path, "sunny-otter-K4Q7B2")
    layout = resolve_session(tmp_path, "sunny-otter-K4Q7B2", buckets=("plans",))
    assert (layout.session_id, layout.subdir) == ("sunny-otter-K4Q7B2", "plans")


def test_resolve_unambiguous_prefix(tmp_path: Path) -> None:
    _bucket(tmp_path, "sunny-otter-K4Q7B2", "calm-river-AAAA11")
    assert resolve_session(tmp_path, "sunny", buckets=("plans",)).session_id == "sunny-otter-K4Q7B2"
    assert (
        resolve_session(tmp_path, "calm-riv", buckets=("plans",)).session_id == "calm-river-AAAA11"
    )


def test_resolve_ambiguous_prefix(tmp_path: Path) -> None:
    _bucket(tmp_path, "sunny-otter-K4Q7B2", "sunny-otter-AAAA11")
    with pytest.raises(SessionIdError, match="ambiguous"):
        resolve_session(tmp_path, "sunny", buckets=("plans",))


def test_resolve_no_match(tmp_path: Path) -> None:
    _bucket(tmp_path, "sunny-otter-K4Q7B2")
    with pytest.raises(SessionIdError, match="no session matches"):
        resolve_session(tmp_path, "nope", buckets=("plans",))


def test_a_bucket_scoped_query_ignores_the_other_buckets(tmp_path: Path) -> None:
    """`plan show`'s resolver walked only plans/ through a twin of the shared
    resolver whose wording had drifted; the shared one takes the buckets, so a
    plans-only prefix is not ambiguous against a run of the same prefix."""
    _bucket(tmp_path, "sunny-otter-K4Q7B2")
    (tmp_path / "sessions" / "runs" / "sunny-otter-AAAA11").mkdir(parents=True)
    assert resolve_session(tmp_path, "sunny", buckets=("plans",)).session_id == "sunny-otter-K4Q7B2"
    with pytest.raises(SessionIdError, match="ambiguous"):
        resolve_session(tmp_path, "sunny")


def test_resolve_empty_query(tmp_path: Path) -> None:
    with pytest.raises(SessionIdError, match="empty"):
        resolve_session(tmp_path, "")
