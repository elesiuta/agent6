# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Render snapshot: the composed web page is pinned byte-for-byte.

`PAGE_HTML` splices the `client.js` / `styles.css` resources into the HTML
template at import time; pinning its sha256 makes any byte drift in the page or
its assets a deliberate, visible change reviewed alongside the edit that moved
it.
"""

from __future__ import annotations

import hashlib
from importlib import resources

from agent6.ui.web.page import PAGE_HTML

# sha256 of PAGE_HTML.encode("utf-8"). An edit to page.py, client.js, or
# styles.css moves it; update it in the same commit as that edit.
PAGE_SHA256 = "3255921514461dfa9db5e7154987d0819f50d5b4f724ee8a39728bca11fd0bbe"


def test_rendered_page_bytes_are_pinned() -> None:
    got = hashlib.sha256(PAGE_HTML.encode("utf-8")).hexdigest()
    assert got == PAGE_SHA256, (
        f"page bytes changed (sha256 {got}); if intended, update PAGE_SHA256 in this test"
    )


def test_page_assets_load_non_empty() -> None:
    # Guards a packaging regression (an asset missing from the wheel) that the
    # build-time wheel check would otherwise catch only at release.
    web = resources.files("agent6.ui.web")
    for name in ("client.js", "styles.css"):
        assert web.joinpath(name).read_text(encoding="utf-8").strip(), f"{name} is empty"
