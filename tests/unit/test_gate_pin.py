# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The manifest's gate pin: who writes it, and what keeps it true.

Every viewer, the baseline check and the next leg read the gate from here, so a
pin that goes stale is a surface that lies about what judged the run.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agent6.app.manifest import pin_gate, write_session_manifest
from agent6.app.reporter import STDIO_REPORTER, Reporter
from agent6.config import Config
from agent6.events import EventSink
from agent6.sessions.layout import SessionLayout
from agent6.sessions.manifest import read_manifest


def _layout(tmp_path: Path) -> SessionLayout:
    layout = SessionLayout(state_dir=tmp_path, session_id="brave-elk-BBBBBB")
    layout.ensure()
    write_session_manifest(
        layout,
        session_id=layout.session_id,
        user_task="t",
        base_sha="0" * 40,
        base_branch="main",
        run_branch=None,
        cfg=Config(),
    )
    return layout


def _sink(tmp_path: Path) -> EventSink:
    return EventSink(tmp_path / "logs.jsonl")


def _quiet() -> tuple[Reporter, list[str]]:
    said: list[str] = []
    return Reporter(out=said.append, err=said.append), said


def test_a_gate_adopted_mid_leg_re_pins(tmp_path: Path) -> None:
    """The stamp and the re-stamp were separate wiring, present only on a fresh
    run: a RESUMED leg that adopted a gate left a manifest reading gateless
    while a gate was live."""
    layout = _layout(tmp_path)
    events = _sink(tmp_path)
    reporter, _said = _quiet()
    pin_gate(layout.session_dir, (), "", events=events, reporter=reporter)
    assert read_manifest(layout.session_dir).workflow.verify_command == ()

    events.emit("loop.verify_inferred", command=["pytest", "-q"], source="agents_md", adopted_at=3)

    pinned = read_manifest(layout.session_dir).workflow
    assert pinned.verify_command == ("pytest", "-q")
    assert pinned.verify_origin == "adopted"


def test_an_un_adopted_gate_re_pins_gateless(tmp_path: Path) -> None:
    """The un-adopt rides the same event with an empty command: the manifest
    reads gateless again, labelled as such."""
    layout = _layout(tmp_path)
    events = _sink(tmp_path)
    reporter, _said = _quiet()
    pin_gate(layout.session_dir, (), "", events=events, reporter=reporter)
    events.emit("loop.verify_inferred", command=["pytest", "-q"], source="agents_md", adopted_at=3)
    events.emit("loop.verify_inferred", command=[], source="unadopted", adopted_at=5)
    pinned = read_manifest(layout.session_dir).workflow
    assert pinned.verify_command == () and pinned.verify_origin == "unadopted"


def test_a_preflight_inference_is_not_an_adoption(tmp_path: Path) -> None:
    """The same event fires at run start with no `adopted_at`; re-pinning on it
    would relabel a configured gate."""
    layout = _layout(tmp_path)
    events = _sink(tmp_path)
    reporter, _said = _quiet()
    pin_gate(layout.session_dir, ("make", "check"), "configured", events=events, reporter=reporter)
    events.emit("loop.verify_inferred", command=["pytest"], source="repo_signals")
    pinned = read_manifest(layout.session_dir).workflow
    assert pinned.verify_command == ("make", "check")
    assert pinned.verify_origin == "configured"


def test_a_pin_that_cannot_be_written_is_reported(tmp_path: Path) -> None:
    """EventSink swallows a listener's exceptions so a UI consumer cannot break
    the run -- which silently ate the re-pin's failure too."""
    layout = _layout(tmp_path)
    events = _sink(tmp_path)
    reporter, said = _quiet()
    pin_gate(layout.session_dir, (), "", events=events, reporter=reporter)
    layout.manifest_path.unlink()
    events.emit("loop.verify_inferred", command=["pytest"], source="agents_md", adopted_at=1)
    assert any("could not record this run's verify gate" in line for line in said)


def test_a_fork_inherits_the_gate_its_source_was_judged_by(tmp_path: Path) -> None:
    """Derived from the current config instead, a source whose gate was inferred
    or adopted forked to a run every surface called gateless."""
    dst = SessionLayout(state_dir=tmp_path, session_id="quiet-fox-AAAAAA")
    dst.ensure()
    write_session_manifest(
        dst,
        session_id=dst.session_id,
        user_task="t",
        base_sha="0" * 40,
        base_branch="main",
        run_branch=None,
        cfg=Config(),  # no verify_command configured, as the source had none
        gate=(("pytest", "-q"), "adopted"),
    )
    pinned = read_manifest(dst.session_dir).workflow
    assert pinned.verify_command == ("pytest", "-q")
    assert pinned.verify_origin == "adopted"


def test_nothing_runs_a_second_gate_at_the_end_of_a_run(tmp_path: Path) -> None:
    """The whole feature: a second full gate in the teardown produced nine
    findings across two audit rounds -- holding the repo and worker locks, a
    Ctrl-C during it replacing the run's exit code, gating a fork's PARENT
    base, running with no PATH so every real gate exited 127. The answer is
    observed for free during the run instead."""
    import agent6.app.finalize as finalize_mod
    import agent6.app.resume as resume_mod
    import agent6.app.run as run_mod

    for module in (finalize_mod, run_mod, resume_mod):
        src = Path(module.__file__ or "").read_text(encoding="utf-8")
        assert "gate_on_base" not in src, f"{module.__name__} still runs a second gate"


def test_a_red_gate_nobody_checked_says_so_and_names_the_check(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A second full gate run used to answer this in the teardown, holding the
    checkout for up to verify_timeout_s after the run visibly ended -- and its
    own failures answered it wrong more than once. A run whose first verify saw
    a clean tree already knows for free; this is the other case, and saying so
    beats guessing."""
    import json

    from agent6.app import finalize
    from agent6.budget import BudgetTracker
    from agent6.workflows._session_state import SessionResult

    rd = tmp_path / "sessions" / "runs" / "r1"
    rd.mkdir(parents=True)
    (rd / "logs.jsonl").write_text(
        json.dumps({"type": "session.end", "reason": "finish_session", "all_passed": False}) + "\n",
        encoding="utf-8",
    )
    (rd / "manifest.json").write_text(
        json.dumps(
            {
                "version": 3,
                "session_id": "r1",
                "mode": "run",
                "base_sha": "a" * 40,
                "workflow": {"verify_command": ["uv", "run", "pytest"]},
            }
        ),
        encoding="utf-8",
    )
    finalize.print_session_end(
        SessionResult(
            completed=True,
            reason="finish_session",
            summary="s",
            iterations=1,
            tool_calls=1,
            verified="failed",
        ),
        layout=SessionLayout(state_dir=tmp_path, session_id="r1"),
        budget=BudgetTracker(max_usd=-1, max_tokens_fallback=-1, max_percent=-1),
        console_stream=False,
        reporter=STDIO_REPORTER,
    )
    out = capsys.readouterr().out
    assert "nothing checked it before this run started" in out
    assert "uv run pytest" in out


def test_a_run_records_the_isolation_it_actually_ran_under(tmp_path: Path) -> None:
    """`auto` degrades. A manifest stamping the knob told every surface "auto",
    which says nothing about whether the run was confined."""
    layout = SessionLayout(state_dir=tmp_path, session_id="quiet-fox-AAAAAA")
    layout.ensure()
    write_session_manifest(
        layout,
        session_id=layout.session_id,
        user_task="t",
        base_sha="0" * 40,
        base_branch="main",
        run_branch=None,
        cfg=Config(),  # sandbox.isolation defaults to "auto"
        isolation="hardened",
    )
    assert read_manifest(layout.session_dir).policy.isolation == "hardened"


def test_an_empty_gate_never_carries_an_origin(tmp_path: Path) -> None:
    """`configured` beside `()` is self-contradictory on disk, and the next
    leg reads that origin back."""
    layout = _layout(tmp_path)
    reporter, _said = _quiet()
    pin_gate(layout.session_dir, (), "", events=_sink(tmp_path), reporter=reporter)
    pinned = read_manifest(layout.session_dir).workflow
    assert (pinned.verify_command, pinned.verify_origin) == ((), "")
