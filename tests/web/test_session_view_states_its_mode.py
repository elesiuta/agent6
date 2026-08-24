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


def test_conversation_route_paints_the_prompts_it_claims_to_answer() -> None:
    """#/conversation/<id> registers as the run's answering front-end the
    moment its stream opens, but never painted the prompts: a run blocked on
    an approval waited on a page that could not show it. The route's builder
    paints prompts like the run view -- seeded from the snapshot, then per
    frame."""
    client = CLIENT_JS
    conv = client[client.index("async function renderConversation") :]
    conv = conv[: conv.index("// --- machine watch")]
    assert conv.count("paintPrompts(") >= 2, "the conversation route paints no prompts"
