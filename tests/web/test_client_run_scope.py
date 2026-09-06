# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""`paintRun` reads the session id it was given, never a free `id`.

The step picker fetched `/api/session/<id>/diff` with a bare `id`, which is a
parameter of `renderRun`, not of `paintRun`: under "use strict" every step
selection threw ReferenceError and rendered "id is not defined", so per-step
diffs and the as-of panels were unreachable on the web.
"""

from __future__ import annotations

import re

from agent6.ui.web.page import CLIENT_JS


def _paint_run_body() -> str:
    start = CLIENT_JS.index("function paintRun(")
    end = CLIENT_JS.index("function renderDiff(", start)
    return CLIENT_JS[start:end]


def test_paint_run_reads_no_free_id() -> None:
    body = _paint_run_body()
    # A bare `id` token: not a property (`.id`), not `_id`, not a key (`id:`),
    # not the string 'id'.
    free = [m.group(0) for m in re.finditer(r"(?<![.\w'\"])id(?![\w:'\"])", body)]
    assert free == [], f"paintRun references a free `id` {len(free)} time(s)"


def test_the_step_picker_fetches_with_the_cards_own_id() -> None:
    body = _paint_run_body()
    assert body.count("encodeURIComponent(cards._id)") >= 2
