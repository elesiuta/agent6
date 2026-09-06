# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""An id that resolves nothing is not "no run".

These messages are what an operator sees from `agent6 sessions show <id>`,
`attach <id>`, `resume <id>` -- all of which reach every bucket. Calling the
miss a "run" is wrong for the plan or ask they just tried to open, and it reads
as "that is not a run" rather than "there is nothing by that id".
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agent6.sessions.id import SessionIdError, resolve_session
from agent6.sessions.layout import bucket_dir
from agent6.ui.cli._common import resolve_session_layout  # pyright: ignore[reportPrivateUsage]


def test_the_cross_bucket_resolver_says_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SessionIdError) as caught:
        resolve_session_layout(tmp_path, "nope-nope-NOPE00")
    assert "no session matches" in str(caught.value), str(caught.value)


def test_the_bucket_scoped_resolver_says_session(tmp_path: Path) -> None:
    bucket = bucket_dir(tmp_path, "runs")
    bucket.mkdir(parents=True)
    with pytest.raises(SessionIdError) as caught:
        resolve_session(tmp_path, "nope-nope-NOPE00", buckets=("runs",))
    assert "no session matches" in str(caught.value), str(caught.value)
    assert str(tmp_path) in str(caught.value)  # the state dir searched
