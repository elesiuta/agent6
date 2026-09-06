# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""`_leg.detach_to_background`: the one hand-off both lifecycles make after a
`/detach` (ask about approvals while away, spawn the background resume under
the invocation's flags, then say so)."""

from __future__ import annotations

import os
from collections.abc import Callable, Sequence
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import agent6.app._leg as leg_mod
from agent6.app._leg import LegInputs, detach_to_background, run_leg
from agent6.app.frontend import FrontendCapabilities, SessionFrontend
from agent6.app.reporter import Reporter
from agent6.config import Config
from agent6.events import EventSink
from agent6.sessions.ipc import read_worker_pid, write_worker_pid
from agent6.sessions.layout import SessionLayout
from agent6.tools.operator_prompts import OperatorPrompts
from agent6.ui.acp.frontend import acp_frontend
from agent6.workflows.loop import SessionResult


def _frontend(calls: list[tuple[str, Any]], *, spawn_err: str = "") -> SessionFrontend:
    def _spawn(_cwd: Path, sid: str, flags: Sequence[str]) -> str:
        calls.append(("spawn", (sid, list(flags))))
        return spawn_err

    def _ask_away(_dir: Path, scopes: tuple[str, ...]) -> None:
        calls.append(("ask", scopes))

    front = acp_frontend(
        ask=lambda _p, _o, _s, _c, _u=None: None,
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


def _returning(value: object) -> Callable[..., object]:
    def stub(*_a: object, **_k: object) -> object:
        return value

    return stub


def _stub_leg_internals(
    monkeypatch: pytest.MonkeyPatch, result: SessionResult, built: dict[str, Any] | None = None
) -> None:
    class _Workflow:
        iterations_reached = 3

        def __init__(self, **kw: Any) -> None:
            if built is not None:
                built.update(kw)
            self._undo_forker: Callable[[], tuple[str, str] | None] = kw["undo_forker"]

        def run(self, _task: str) -> SessionResult:
            if result.reason == "undone":
                self._undo_forker()  # what the loop does before an `undone` end
            return result

    session = SimpleNamespace(
        budget=SimpleNamespace(
            format_summary=lambda: "[agent6] cost $0.01", estimate_usd=lambda: (0.01, False)
        ),
        rm_role=SimpleNamespace(model="fake/model"),
        provider=None,
        summariser_provider=None,
        review_seats=[],
        close=lambda: None,
    )
    tools = SimpleNamespace(
        curator=None,
        dispatcher=SimpleNamespace(settle_background=lambda: None, close=lambda: None),
        compact_drop_at_chars=1,
        compact_summarise_at_chars=1,
        keep_recent_chars=1,
        cfg=Config(),
    )
    monkeypatch.setattr(leg_mod, "build_session_providers", _returning(session))
    monkeypatch.setattr(leg_mod, "build_prompt_reviser_provider", _returning(None))
    monkeypatch.setattr(leg_mod, "wants_session_network", _returning(False))
    monkeypatch.setattr(leg_mod, "start_mcp_manager_if_enabled", _returning(None))
    monkeypatch.setattr(leg_mod, "build_session_tools", _returning(tools))
    monkeypatch.setattr(leg_mod, "Workflow", _Workflow)
    monkeypatch.setattr(leg_mod, "session_facts_provider", _returning(lambda: None))


def test_a_detached_ask_leg_hands_the_run_over_instead_of_answering_with_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`/detach` at the pause menu is offered in every mode. The ask branch ran
    before the detach check, so a detached ask printed the loop's bookkeeping
    line ("operator detached at iter N") as the model's answer, saved it as the
    transcript, exited 1, and never asked the caller to spawn the continuation."""
    _stub_leg_internals(
        monkeypatch,
        SessionResult(
            completed=False,
            reason="detached",
            summary="operator detached at iter 3; resuming in the background",
            iterations=3,
            tool_calls=7,
        ),
    )
    layout = SessionLayout(state_dir=tmp_path / "state", session_id="asky-one-AAAAAA")
    layout.ensure()
    saved: list[tuple[str, str]] = []
    said: list[str] = []

    def _save(_layout: SessionLayout, question: str, answer: str) -> None:
        saved.append((question, answer))

    front = acp_frontend(
        ask=lambda _p, _o, _s, _c, _u=None: None,
        capabilities=FrontendCapabilities(),
        agent6_exe=lambda: "agent6",
        spawn_detached_resume=lambda _cwd, _sid, _flags: "",
    )
    cwd = tmp_path / "repo"
    cwd.mkdir()

    end = run_leg(
        Config(),
        layout,
        LegInputs(
            session_id=layout.session_id,
            mode="ask",
            role="worker",
            isolation="hardened",
            tui_enabled=False,
            interactive=False,
            task="what does this repo do?",
            gate=lambda cfg, _b: cfg,
            chain_branch=None,
            base_sha="",
            untracked_at_start=frozenset(),
            resume_state_path=layout.session_dir / "loop_state.json",
            undo_forker=lambda: None,
            prompts=OperatorPrompts(session_dir=layout.session_dir),
            ask_transcript_task="what does this repo do?",
        ),
        frontend=replace(front, save_ask_transcript=_save),
        reporter=Reporter(out=said.append, err=said.append),
        events=EventSink(layout.logs_path),
        transcript_sink=None,  # type: ignore[arg-type]
        cwd=cwd,
        state_dir=tmp_path / "state",
    )

    assert (end.rc, end.detach_requested) == (0, True)
    assert saved == []
    assert not any("operator detached at iter" in s for s in said)


def test_an_undone_ask_leg_names_the_fork_instead_of_answering_with_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`/undo` at the pause menu is offered in every mode. The ask branch ran
    before the undo check, so an undone ask printed the loop's bookkeeping
    line as the model's answer, saved it as the transcript, and never named
    the fork to continue from."""
    _stub_leg_internals(
        monkeypatch,
        SessionResult(
            completed=False,
            reason="undone",
            summary="operator undid the last message at iter 3",
            iterations=3,
            tool_calls=7,
        ),
    )
    layout = SessionLayout(state_dir=tmp_path / "state", session_id="asky-two-AAAAAA")
    layout.ensure()
    saved: list[tuple[str, str]] = []
    said: list[str] = []

    def _save(_layout: SessionLayout, question: str, answer: str) -> None:
        saved.append((question, answer))

    front = acp_frontend(
        ask=lambda _p, _o, _s, _c, _u=None: None,
        capabilities=FrontendCapabilities(),
        agent6_exe=lambda: "agent6",
        spawn_detached_resume=lambda _cwd, _sid, _flags: "",
    )
    cwd = tmp_path / "repo"
    cwd.mkdir()

    end = run_leg(
        Config(),
        layout,
        LegInputs(
            session_id=layout.session_id,
            mode="ask",
            role="worker",
            isolation="hardened",
            tui_enabled=False,
            interactive=False,
            task="what does this repo do?",
            gate=lambda cfg, _b: cfg,
            chain_branch=None,
            base_sha="",
            untracked_at_start=frozenset(),
            resume_state_path=layout.session_dir / "loop_state.json",
            undo_forker=lambda: ("fork-two-BBBBBB", "what does this repo do?"),
            prompts=OperatorPrompts(session_dir=layout.session_dir),
            ask_transcript_task="what does this repo do?",
        ),
        frontend=replace(front, save_ask_transcript=_save),
        reporter=Reporter(out=said.append, err=said.append),
        events=EventSink(layout.logs_path),
        transcript_sink=None,  # type: ignore[arg-type]
        cwd=cwd,
        state_dir=tmp_path / "state",
    )

    assert end.rc == 0
    assert saved == []
    assert any("continue as fork-two-BBBBBB" in s for s in said)
    assert not any("operator undid" in s for s in said)


def test_a_surface_without_the_revise_choice_skips_revision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The ACP front-end has no terminal to ask the revise_prompt choice on;
    its selector answered None, which read as the operator's quit and ended
    the run "stopped" for an act nobody performed. A surface with no selector
    skips revision, as a leg under the TUI does."""
    built: dict[str, Any] = {}
    _stub_leg_internals(
        monkeypatch,
        SessionResult(
            completed=True, reason="finish_session", summary="ok", iterations=1, tool_calls=0
        ),
        built,
    )
    layout = SessionLayout(state_dir=tmp_path / "state", session_id="acp-one-AAAAAA")
    layout.ensure()
    said: list[str] = []
    front = acp_frontend(
        ask=lambda _p, _o, _s, _c, _u=None: None,
        capabilities=FrontendCapabilities(),
        agent6_exe=lambda: "agent6",
        spawn_detached_resume=lambda _cwd, _sid, _flags: "",
    )
    assert front.select_revised_prompt is None
    cwd = tmp_path / "repo"
    cwd.mkdir()

    end = run_leg(
        Config.model_validate({"prompt": {"revise_prompt": "interactive"}}),
        layout,
        LegInputs(
            session_id=layout.session_id,
            mode="run",
            role="worker",
            isolation="hardened",
            tui_enabled=False,
            interactive=False,
            task="do the thing",
            gate=lambda cfg, _b: cfg,
            chain_branch=None,
            base_sha="",
            untracked_at_start=frozenset(),
            resume_state_path=layout.session_dir / "loop_state.json",
            undo_forker=lambda: None,
            prompts=OperatorPrompts(session_dir=layout.session_dir),
            ask_transcript_task=None,
        ),
        frontend=front,
        reporter=Reporter(out=said.append, err=said.append),
        events=EventSink(layout.logs_path),
        transcript_sink=None,  # type: ignore[arg-type]
        cwd=cwd,
        state_dir=tmp_path / "state",
    )

    assert end.rc == 0
    assert built["revise_prompt"] == "off"
    assert any("this surface has none" in s for s in said)
