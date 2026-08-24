# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Every path the page fetches is a path the server routes.

The page is one JS file talking to one server, and nothing type-checks the seam
between them: renaming a route (`/api/run/` -> `/api/session/`) or a payload key
(`runs` -> `sessions`) leaves the server's own tests green while the browser
shows an empty page.
"""

from __future__ import annotations

import json
import re
import threading
from collections.abc import Iterator
from http.client import HTTPConnection
from pathlib import Path

import pytest

from agent6.config.layer import resolved_state_dir
from agent6.ui.web import model
from agent6.ui.web.page import CLIENT_JS
from agent6.ui.web.server import WebServer

# `<id>` stands in for whatever the page interpolates; the fixture creates it.
_ID = "brave-oak-AAAAAA"


@pytest.fixture
def served(tmp_path: Path) -> Iterator[int]:
    session = resolved_state_dir(tmp_path) / "sessions" / "runs" / _ID
    session.mkdir(parents=True)
    (session / "logs.jsonl").write_text(
        json.dumps({"type": "session.start", "mode": "run", "user_task": "t"}) + "\n",
        encoding="utf-8",
    )
    srv = WebServer(("127.0.0.1", 0), tmp_path, "")
    port = int(srv.server_address[1])
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    try:
        yield port
    finally:
        srv.shutdown()
        srv.server_close()


def _client_api_paths() -> set[str]:
    """The `/api/...` paths client.js builds, with interpolation collapsed."""
    source = CLIENT_JS
    found: set[str] = set()
    for raw in re.findall(r"'(/api/[^']*)'", source):
        # `'/api/session/' + encodeURIComponent(id) + '/steer'` arrives as two
        # fragments; a trailing slash means an id follows.
        found.add(raw)
    return found


def _unrouted(port: int, path: str, method: str) -> bool:
    """True when the SERVER has no route for *path*.

    Distinct from a routed path answering "no such session": the router's own
    miss is the only one that says `not found: <path>`, and it is the one a
    renamed route produces.
    """
    conn = HTTPConnection("127.0.0.1", port, timeout=10)
    try:
        body = b"{}" if method == "POST" else None
        headers = {"Content-Type": "application/json"} if body else {}
        conn.request(method, path, body=body, headers=headers)
        resp = conn.getresponse()
        payload = resp.read()
        return resp.status == 404 and f"not found: {path}".encode() in payload
    finally:
        conn.close()


def test_every_api_path_the_page_calls_is_routed(served: int) -> None:
    """A 404 "not found: <path>" is the server saying it has no such route --
    which for the page means a dead button, not an error anyone sees."""
    missing: list[str] = []
    for fragment in sorted(_client_api_paths()):
        path = fragment if not fragment.endswith("/") else f"{fragment}{_ID}"
        if all(_unrouted(served, path, method) for method in ("GET", "POST")):
            missing.append(path)
    assert not missing, f"the page calls paths the server does not route: {missing}"


def test_the_page_reads_the_keys_the_hub_actually_sends(tmp_path: Path) -> None:
    """`d.runs` survived the rename as a silently-undefined lookup: the hub kept
    answering 200 and the list rendered empty.

    Scoped to the hub's own `build(d)` body, because `d` is the page's generic
    name for any decoded response.
    """
    source = CLIENT_JS
    start = source.index("const build = (d) => {")
    body = source[start : source.index("\n  };", start)]
    read = set(re.findall(r"\bd\.([a-z_]+)\b", body))
    assert read, "the hub render body moved; this test is no longer reading it"

    payload = model.hub_payload(tmp_path)
    missing = sorted(key for key in read if key not in payload)
    assert not missing, f"the page reads hub keys the hub does not send: {missing}"
