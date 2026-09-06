# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""A husk session refuses at the resolver; only `sessions rm` still names one."""

from __future__ import annotations

from pathlib import Path

import pytest

from agent6.paths import state_dir
from agent6.sessions.id import SessionIdError
from agent6.sessions.layout import bucket_dir
from agent6.ui.cli._common import (  # pyright: ignore[reportPrivateUsage]
    resolve_or_newest_layout,
    resolve_session_layout,
)


def _husk(repo: Path, session_id: str = "husky-one-AAAAAA") -> Path:
    """A session dir with neither manifest.json nor logs.jsonl: it crashed
    before it ever started."""
    d = bucket_dir(state_dir(repo), "runs") / session_id
    d.mkdir(parents=True)
    return d


def test_an_explicit_husk_id_refuses_with_the_remedy(tmp_path: Path) -> None:
    """`attach`/`sessions show` presented a husk as real and advised a resume
    that fails; the resolver answers once, for every surface."""
    _husk(tmp_path)
    with pytest.raises(SessionIdError, match="crashed before it ever started"):
        resolve_session_layout(tmp_path, "husky-one-AAAAAA")
    with pytest.raises(SessionIdError, match="sessions rm"):
        resolve_session_layout(tmp_path, "husky")  # by prefix too


def test_rm_still_resolves_a_husk(tmp_path: Path) -> None:
    """Cleanup must keep working: rm is the surface that deletes exactly this."""
    d = _husk(tmp_path)
    layout = resolve_session_layout(tmp_path, "husky-one-AAAAAA", allow_husk=True)
    assert layout.session_dir == d
    from_newest = resolve_or_newest_layout(tmp_path, "husky-one-AAAAAA", allow_husk=True)
    assert from_newest is not None and from_newest.session_dir == d


def test_a_real_session_still_resolves(tmp_path: Path) -> None:
    d = bucket_dir(state_dir(tmp_path), "runs") / "realy-two-BBBBBB"
    d.mkdir(parents=True)
    (d / "logs.jsonl").write_text('{"type":"session.start","mode":"run"}\n', encoding="utf-8")
    assert resolve_session_layout(tmp_path, "realy-two-BBBBBB").session_dir == d
