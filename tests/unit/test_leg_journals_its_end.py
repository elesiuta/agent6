# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""A leg that dies before the loop starts journals session.end BEFORE the
tui_session scope closes: that scope's exit is `_live.tui_session`'s
`proc.wait()`, which blocks on a dashboard that leaves only on a session.end.
"""

from __future__ import annotations

import contextlib
import json
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import agent6.app._leg as leg_mod
from agent6.app._leg import LegInputs, run_leg
from agent6.app.frontend import FrontendCapabilities
from agent6.app.reporter import Reporter
from agent6.config import Config
from agent6.events import EventSink
from agent6.sessions.layout import SessionLayout
from agent6.ui.acp.frontend import acp_frontend
from agent6.ui.steer import SteerState
from agent6.workflows._session_state import SNAPSHOT_VERSION

# The snapshot resume.py's preflight accepts (load_session_snapshot passes) and
# Conversation.from_wire rejects one leg deeper: a tool_result with no tool_use.
TORN = {
    "version": SNAPSHOT_VERSION,
    "system": "s",
    "messages": [
        {"role": "user", "content": [{"type": "text", "text": "TASK:\nx"}]},
        {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "toolu_gone", "content": "ok"}],
        },
    ],
    "tool_calls": 3,
    "next_iteration": 4,
    "root_task_id": None,
    "original_task": "x",
    "verify_command": [],
}


def _returning(value: object) -> Callable[..., object]:
    def stub(*_a: object, **_k: object) -> object:
        return value

    return stub


def test_a_resume_error_journals_session_end_before_the_tui_is_waited_on(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A ResumeError left the loop with no session.end, and the outer handlers
    that journal one for an interrupt or a crash sit past the `tui_session`
    scope, whose exit waits on a dashboard that leaves only on an end it can
    see: `resume --tui` on a torn snapshot hung on its own TUI."""
    state = tmp_path / "state"
    layout = SessionLayout(state_dir=state, session_id="sess-AAAA11")
    layout.session_dir.mkdir(parents=True)
    snap = layout.session_dir / "loop_state.json"
    snap.write_text(json.dumps(TORN), encoding="utf-8")
    events = EventSink(layout.logs_path)

    # What the co-process TUI could see at the moment `_live.tui_session`'s
    # finally calls proc.wait().
    seen_at_exit: list[list[str]] = []

    class _Recorder(contextlib.AbstractContextManager[None]):
        def __enter__(self) -> None:
            return None

        def __exit__(self, *_exc: object) -> bool:
            lines = (
                layout.logs_path.read_text(encoding="utf-8").splitlines()
                if layout.logs_path.exists()
                else []
            )
            seen_at_exit.append([json.loads(x)["type"] for x in lines if x.strip()])
            return False

    def _tui_session(_dir: Path, _enabled: bool) -> _Recorder:
        return _Recorder()

    def _steer_state(*_a: object) -> SteerState:
        return SteerState(
            requested=lambda: False,
            clear=lambda: None,
            prompt=lambda: None,
            restore=lambda: None,
            abort_pending=lambda: False,
            interrupt=lambda: False,
            reset_stage=lambda: None,
        )

    frontend = replace(
        acp_frontend(
            ask=lambda _p, _o, _s, _c, _u=None: None,
            capabilities=FrontendCapabilities(can_ask=False),
            agent6_exe=lambda: "agent6",
            spawn_detached_resume=lambda _cwd, _sid, _flags: "",
        ),
        tui_session=_tui_session,
        make_steer_state=_steer_state,
    )

    session = SimpleNamespace(
        budget=MagicMock(),
        rm_role=SimpleNamespace(model="m", provider="p"),
        provider=MagicMock(),
        summariser_provider=None,
        review_seats=[],
        close=lambda: None,
    )
    tools = SimpleNamespace(
        curator=None,
        dispatcher=MagicMock(),
        compact_drop_at_chars=1,
        compact_summarise_at_chars=1,
        keep_recent_chars=1,
        cfg=Config(),
    )
    monkeypatch.setattr(leg_mod, "build_session_providers", _returning(session))
    monkeypatch.setattr(leg_mod, "build_prompt_reviser_provider", _returning(None))
    monkeypatch.setattr(leg_mod, "build_session_tools", _returning(tools))
    monkeypatch.setattr(leg_mod, "start_mcp_manager_if_enabled", _returning(None))
    monkeypatch.setattr(leg_mod, "wants_session_network", _returning(False))
    monkeypatch.setattr(leg_mod, "chown_to_real_user", _returning(None))

    inputs = LegInputs(
        session_id=layout.session_id,
        mode="run",
        role="worker",
        isolation="hardened",
        tui_enabled=True,
        interactive=False,
        task=None,  # a resumed leg: wf.resume()
        gate=lambda c, _b: c,
        chain_branch=None,
        base_sha="",
        untracked_at_start=frozenset(),
        resume_state_path=snap,
        undo_forker=lambda: None,
        prompts=MagicMock(),
        ask_transcript_task=None,
        resuming=True,
    )
    end = run_leg(
        Config(),
        layout,
        inputs,
        frontend=frontend,
        reporter=Reporter(out=lambda _m: None, err=lambda _m: None),
        events=events,
        transcript_sink=MagicMock(),
        cwd=tmp_path,
        state_dir=state,
    )
    assert end.rc == 1
    assert seen_at_exit, "the tui_session scope never closed"
    assert "session.end" in seen_at_exit[0], seen_at_exit[0]
