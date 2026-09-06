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


def test_the_machine_watch_gates_on_the_shared_refusals() -> None:
    """A live machine blocked on an approval reads status "waiting", so gating
    the prompt boxes on the status word hid the box the machine was blocked on
    from the page that had claimed it as answer front-end."""
    from agent6.ui.web.page import CLIENT_JS as js

    start = js.index("function paintMachine(")
    body = js[start : js.index("\nfunction ", start + 1)]
    assert "m.refusals" in body, "the machine watch derives its own gating"
    assert "notRunning" not in body
    assert "canAnswer ? (data.reasoning || {}) : {}" in body


def test_the_web_approval_box_offers_every_answer() -> None:
    """`session-deny` is a first-class answer (the CLI's `x`, the TUI's "Deny
    all", the endpoint's own Literal); the box offered three of the four."""
    from agent6.ui.web.page import CLIENT_JS as js

    start = js.index("for (const ap of")
    body = js[start : js.index("for (const q of", start)]
    for answer in ("'yes'", "'no'", "'session'", "'session-deny'"):
        assert f"send({answer})" in body, answer


def test_a_failure_toast_holds_until_it_is_dismissed() -> None:
    """Every message was one 4-second toast at one fixed position: two
    overlapped, and a captured CLI refusal (several lines) was gone before it
    could be read."""
    from agent6.ui.web.page import CLIENT_JS as js
    from agent6.ui.web.page import PAGE_HTML

    start = js.index("function toast(")
    body = js[start : js.index("\nfunction ", start + 1)]
    assert "setTimeout" in body, "a confirmation still clears itself"
    assert "if (bad)" in body and "t.remove()" in body, "a failure has no dismiss"
    assert "#toasts" in PAGE_HTML, "the stack has no container style"


def test_the_machine_page_offers_stop() -> None:
    """`agent6 machine stop` and the TUI's `x` park a machine at its next
    transition; the page had no way to ask, though its route was there."""
    from agent6.ui.web.page import CLIENT_JS as js

    start = js.index("async function renderMachine")
    body = js[start : js.index("function paintMachine(", start)]
    assert "'/stop'" in body or "+ '/stop'" in body
    assert "cards._stop_btn" in js


def test_the_in_flight_mark_needs_a_live_run() -> None:
    """A killed worker leaves a `role.call` with no `role.result`, so
    `in_flight` stays true forever: the Overview card printed the working
    ellipsis beside the word "stale"."""
    body = _paint_run_body()
    assert "r.in_flight && s.live" in body, "the in-flight mark must read liveness too"


def test_the_web_tool_row_counts_the_args_lines_it_drops() -> None:
    """The TUI row folds every args line (`clip_cell`); the web row showed
    line one and dropped the rest unmarked, counting only the result's."""
    start = CLIENT_JS.index("// tools: one clipped line per call")
    body = CLIENT_JS[start : CLIENT_JS.index("// shells:", start)]
    extra = body[body.index("const extra") : body.index("\n", body.index("const extra"))]
    assert "args_preview" in extra, extra
