# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The web session view says what kind of session it is showing.

The page opens for any session, but its snapshot carried no mode -- so the
details panel was headed a hard-coded "Run" and the composer said "continue the
run" over a plan or an ask. The heading is exactly where the mode belongs, and
it was stating the opposite.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent6.sessions.layout import bucket_dir
from agent6.ui.web.page import CLIENT_JS
from agent6.viewmodel import session_snapshot


def _session(state: Path, bucket: str, session_id: str, mode: str) -> Path:
    session = bucket_dir(state, bucket) / session_id
    session.mkdir(parents=True)
    (session / "manifest.json").write_text(
        json.dumps({"version": 3, "session_id": session_id, "mode": mode, "user_task": "t"}),
        encoding="utf-8",
    )
    (session / "logs.jsonl").write_text(
        json.dumps({"type": "session.start", "mode": mode, "user_task": "t"}) + "\n",
        encoding="utf-8",
    )
    return session


@pytest.mark.parametrize(("bucket", "mode"), [("runs", "run"), ("plans", "plan"), ("asks", "ask")])
def test_the_snapshot_carries_the_mode(tmp_path: Path, bucket: str, mode: str) -> None:
    session = _session(tmp_path, bucket, "brave-oak-AAAAAA", mode)
    assert session_snapshot(session)["mode"] == mode


def test_the_page_heads_the_panel_with_the_mode_not_a_fixed_word() -> None:
    """A hard-coded 'Run' is right one time in three: paintRun must write the
    snapshot's mode into the heading (the old absence check passed forever
    without saying what the page does instead)."""
    client = CLIENT_JS
    assert "cards._head_title.textContent = s.mode" in client


def test_the_session_view_is_the_one_conversation_page() -> None:
    """A second route rendered the same conversation with its own stream
    handler, which had already drifted (no `/undo` branch), and nothing
    linked to it. The session view is the conversation page."""
    assert "renderConversation" not in CLIENT_JS
    assert "parts[0] === 'conversation'" not in CLIENT_JS


def test_the_session_view_paints_the_prompts_it_claims_to_answer() -> None:
    """Opening the session view's stream claims the run as an answer front-end
    (`WebServer.claim_session`), so `paintRun` must paint its prompts: a run
    blocked on an approval would otherwise wait on the page that took the
    claim while it showed nothing."""
    start = CLIENT_JS.index("function paintRun(")
    body = CLIENT_JS[start : CLIENT_JS.index("function renderDiff(", start)]
    assert "paintPrompts(cards, isDead ? {} : s)" in body


def test_the_run_crumb_carries_the_state_word() -> None:
    """A phone shows one widget at a time and opens on the conversation, so the
    state was on a card the operator had to go find: a run could be waiting on
    an approval, or dead, and the page it opened said neither. The crumb sits in
    the fixed header on every widget page."""
    client = CLIENT_JS
    assert "setCrumb(runState(s) + ' · ' + cards._crumb)" in client
    # One owner for the word: the state row reads the same helper.
    assert "add('state', runState(s))" in client


def test_the_run_card_shows_the_task_as_sessions_show_prints_it() -> None:
    """A task seeded from a plan carries the whole plan below its title, and
    the card printed all of it into one cell (`# Plan: ...  ## Original task
    ...` inline). `sessions show` prints the first line; the card does the
    same."""
    client = CLIENT_JS
    assert "add('task', (s.user_task || '').split('\\n')[0] || '(none)')" in client
    assert "add('task', s.user_task || '(none)')" not in client
