# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""`_leg.detach_to_background`: the one hand-off both lifecycles make after a
`/detach` (ask about approvals while away, spawn the background resume under
the invocation's flags, then say so)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

from agent6.app._leg import detach_to_background
from agent6.app.frontend import FrontendCapabilities, SessionFrontend
from agent6.app.reporter import Reporter
from agent6.config import Config
from agent6.sessions.layout import SessionLayout
from agent6.ui.acp.frontend import acp_frontend


def _frontend(calls: list[tuple[str, Any]], *, spawn_err: str = "") -> SessionFrontend:
    def _spawn(_cwd: Path, sid: str, flags: Sequence[str]) -> str:
        calls.append(("spawn", (sid, list(flags))))
        return spawn_err

    def _ask_away(_dir: Path, scopes: tuple[str, ...]) -> None:
        calls.append(("ask", scopes))

    front = acp_frontend(
        ask=lambda _p, _o, _s, _c: None,
        capabilities=FrontendCapabilities(),
        agent6_exe=lambda: "agent6",
        spawn_detached_resume=_spawn,
    )
    # The ACP front-end has no away-mode prompt; record when the lifecycle asks.
    return replace(front, prompt_detach_away_mode=_ask_away)


def test_ask_policy_is_asked_before_the_spawn_and_the_flags_ride_along(tmp_path: Path) -> None:
    calls: list[tuple[str, Any]] = []
    said: list[str] = []
    layout = SessionLayout(state_dir=tmp_path, session_id="runny-one-AAAAAA")
    layout.ensure()
    detach_to_background(
        frontend=_frontend(calls),
        cfg=Config(),  # run_commands = ask, nothing granted
        layout=layout,
        cwd=tmp_path,
        flags=["--max-usd", "0.25"],
        reporter=Reporter(out=said.append, err=said.append),
    )
    assert [c[0] for c in calls] == ["ask", "spawn"]
    assert calls[1][1] == ("runny-one-AAAAAA", ["--max-usd", "0.25"])
    assert any("continues in the background" in s for s in said)
    assert any("agent6 attach runny-one-AAAAAA" in s for s in said)


def test_a_failed_spawn_is_reported_and_never_called_a_continuation(tmp_path: Path) -> None:
    """The reattach line used to print before the spawn; a spawn that failed
    left "continues in the background" said of a run nothing was driving."""
    calls: list[tuple[str, Any]] = []
    said: list[str] = []
    layout = SessionLayout(state_dir=tmp_path, session_id="runny-one-AAAAAA")
    layout.ensure()
    cfg = Config.model_validate({"sandbox": {"run_commands": "yes"}})
    detach_to_background(
        frontend=_frontend(calls, spawn_err="agent6 exe not found"),
        cfg=cfg,
        layout=layout,
        cwd=tmp_path,
        flags=[],
        reporter=Reporter(out=said.append, err=said.append),
    )
    assert [c[0] for c in calls] == ["spawn"]  # yes-policy: nothing to ask
    assert said == ["[agent6] agent6 exe not found"]
