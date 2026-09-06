# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The composer's Ctrl-R history search: the client wiring the payload key,
scoped to the focused composer so the browser keeps its reload elsewhere."""

from __future__ import annotations

from agent6.ui.web.page import CLIENT_JS  # the concatenated page-family files


def test_composer_intercepts_ctrl_r_only() -> None:
    # The intercept lives in the composer's own keydown (fires only while the
    # textarea holds focus) and requires the bare Ctrl chord.
    assert "e.key === 'r' && e.ctrlKey && !e.metaKey && !e.altKey && !e.shiftKey" in CLIENT_JS
    # That chord is the ONLY ctrlKey site: no document-level hook, so the
    # browser keeps its reload everywhere outside the composer.
    # ... plus the composer's own Ctrl+Enter (`/now`), inside the same keydown.
    assert CLIENT_JS.count("ctrlKey") == 2
    assert "openHistorySearch" in CLIENT_JS


def test_history_reads_the_payload_key_and_advertises_the_chord() -> None:
    assert "operator_inputs" in CLIENT_JS  # the conversation payload key
    # every composer hint advertises it: steer, resume, and resume-needs-work
    assert CLIENT_JS.count("Ctrl-R past messages") == 3


def test_enter_keeps_the_typed_text_when_nothing_matches() -> None:
    # One accept rule on every surface: the highlighted match, else the query
    # itself (the CLI and TUI searches behave the same way).
    assert "pick(items.length ? items[active] : field.value)" in CLIENT_JS
