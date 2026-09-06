# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""`_leg.detach_to_background`: the one hand-off both lifecycles make after a
`/detach` (ask about approvals while away, spawn the background resume under
the invocation's flags, then say so)."""

from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from agent6.app._leg import detach_to_background
from agent6.app.frontend import FrontendCapabilities, SessionFrontend
from agent6.app.reporter import Reporter
from agent6.config import Config
from agent6.sessions.ipc import read_worker_pid, write_worker_pid
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


def test_a_recorded_away_mode_is_the_runs_away_answer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A run detached from a terminal (or spawned by the hub) carries the
    operator's choice in `approvals/away.mode`. The preflight read only the
    env, so a later resume from cron, CI or a script was refused as
    unanswerable -- saying the run had no away-mode while its own dir had one.
    The approver has always read the file."""
    from agent6.sessions.ipc import effective_away, set_away_mode

    monkeypatch.delenv("AGENT6_DETACHED_AWAY", raising=False)
    session_dir = tmp_path / "run"
    session_dir.mkdir()

    assert effective_away(session_dir) == ""

    set_away_mode(session_dir, "wait")
    assert effective_away(session_dir) == "wait"

    # A launcher's env still wins: it is this invocation's own intent.
    monkeypatch.setenv("AGENT6_DETACHED_AWAY", "deny")
    assert effective_away(session_dir) == "deny"


def test_a_resume_names_what_the_tree_holds_that_no_commit_does(tmp_path: Path) -> None:
    """A fresh run asks about the operator's uncommitted changes; a resume
    swept them into the run's next auto-commit, under the agent's identity and
    into what `sessions diff` and `merge` present as the run's own work, with
    nothing said. The crashed leg's own uncommitted tail lands there too, so
    this names them rather than refusing."""
    import subprocess as sp

    from agent6.git_ops import chain_commit, chain_dirty_paths, chain_ref_for

    repo = tmp_path / "repo"
    repo.mkdir()
    sp.run(["git", "init", "-q", "-b", "main"], cwd=repo, check=True)
    (repo / "a.py").write_text("x = 1\n", encoding="utf-8")
    sp.run(["git", "add", "-A"], cwd=repo, check=True)
    sp.run(
        ["git", "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "seed"],
        cwd=repo,
        check=True,
    )
    base = sp.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True, check=True
    ).stdout.strip()
    ref = chain_ref_for("resume-run-A1")
    (repo / "a.py").write_text("x = 2\n", encoding="utf-8")  # the run's own leg-1 work
    chain_commit(repo, "iter 1", ref=ref, fallback_parent=base)

    assert chain_dirty_paths(repo, ref, base, 5) == []

    (repo / "a.py").write_text("x = 2\n# the operator's note\n", encoding="utf-8")

    assert chain_dirty_paths(repo, ref, base, 5) == ["a.py"]


def test_the_worker_pid_survives_the_handoff_and_goes_when_it_fails(tmp_path: Path) -> None:
    """`sessions` reads a run with no worker pid as "stale (crashed or killed)".
    Clearing it before the spawn put every detaching run in that state for the
    second the background `resume` takes to claim it."""
    calls: list[tuple[str, Any]] = []
    layout = SessionLayout(state_dir=tmp_path, session_id="runny-one-AAAAAA")
    layout.ensure()
    cfg = Config.model_validate({"sandbox": {"run_commands": "yes"}})
    write_worker_pid(layout.session_dir, os.getpid())

    detach_to_background(
        frontend=_frontend(calls),
        cfg=cfg,
        layout=layout,
        cwd=tmp_path,
        flags=[],
        reporter=Reporter(out=lambda _s: None, err=lambda _s: None),
    )
    assert read_worker_pid(layout.session_dir) == os.getpid()  # the child overwrites it

    detach_to_background(
        frontend=_frontend(calls, spawn_err="agent6 exe not found"),
        cfg=cfg,
        layout=layout,
        cwd=tmp_path,
        flags=[],
        reporter=Reporter(out=lambda _s: None, err=lambda _s: None),
    )
    assert read_worker_pid(layout.session_dir) is None  # nothing took over: really dead
