# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The one capped streaming read every HTTP body goes through."""

from __future__ import annotations

import time
from collections.abc import Iterator

import pytest

from agent6.tools.http_body import BodyRefused, read_capped


class _Response:
    def __init__(self, chunks: list[bytes], *, encoding: str = "", slow: float = 0.0) -> None:
        self._chunks = chunks
        self._slow = slow
        self.headers = {"content-encoding": encoding} if encoding else {}

    def iter_raw(self) -> Iterator[bytes]:
        for chunk in self._chunks:
            time.sleep(self._slow)
            yield chunk


def test_the_read_refuses_compression_a_cap_and_a_dribble() -> None:
    """Three callers each spelled this loop and one drifted (no total
    deadline), so a server dribbling a byte at a time held `skills install`
    for as long as it liked. One reader, three refusals, each named."""
    later = time.monotonic() + 5
    assert read_capped(_Response([b"ab", b"c"]), cap=3, deadline=later, timeout_s=5) == b"abc"
    with pytest.raises(BodyRefused, match="content-encoding 'gzip'"):
        read_capped(_Response([b"x"], encoding="gzip"), cap=3, deadline=later, timeout_s=5)
    with pytest.raises(BodyRefused, match="larger than 3 bytes"):
        read_capped(_Response([b"ab", b"cd"]), cap=3, deadline=later, timeout_s=5)
    with pytest.raises(BodyRefused, match=r"still arriving after 0\.01s"):
        soon = time.monotonic() + 0.01
        read_capped(_Response([b"a", b"b"], slow=0.02), cap=9, deadline=soon, timeout_s=0.01)
