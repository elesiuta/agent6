# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""One capped read of a streamed HTTP body, for every caller that fetches
one: the fetch tool, the MCP HTTP transport and the skill installer.

The cap counts what arrives, so compression is refused (a decoded stream
expands past the cap before any check) and the body is read chunk by chunk
against the cap and a total deadline (httpx's timeout is per read; a server
dribbling a byte at a time never trips it).
"""

from __future__ import annotations

import time
from collections.abc import Iterable, Mapping
from typing import Protocol


class _Response(Protocol):
    @property
    def headers(self) -> Mapping[str, str]: ...

    def iter_raw(self) -> Iterable[bytes]: ...


class BodyRefused(Exception):
    """The body was not read to the end; the message names why."""


def read_capped(response: _Response, *, cap: int, deadline: float, timeout_s: float) -> bytes:
    """The raw body of *response*, or `BodyRefused` when it is compressed,
    passes *cap* bytes, or is still arriving at *deadline* (a `time.monotonic`
    instant, *timeout_s* after the request started, named in the refusal)."""
    encoding = response.headers.get("content-encoding", "")
    if encoding.lower() not in ("", "identity"):
        raise BodyRefused(f"refusing content-encoding {encoding!r}: only identity is read")
    body = bytearray()
    for chunk in response.iter_raw():
        body += chunk
        if len(body) > cap:
            raise BodyRefused(f"response is larger than {cap} bytes")
        if time.monotonic() > deadline:
            raise BodyRefused(f"response was still arriving after {timeout_s:g}s")
    return bytes(body)
