# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""`/undo` resolution: the newest checkpoint whose restored conversation ends
before the last operator message, following fork lineage past a fork's one
seed checkpoint."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from agent6.app.manifest import write_session_manifest
from agent6.app.undo import undo_target
from agent6.config import Config
from agent6.sessions.layout import SessionLayout
from agent6.workflows._session_state import SessionSnapshot

_WRAP = "OPERATOR STEERING (mid-run instruction; incorporate this into your next step):\n"


def _task(text: str) -> dict[str, Any]:
    return {"role": "user", "content": text}


def _steer(text: str) -> dict[str, Any]:
    return {"role": "user", "content": [{"type": "text", "text": _WRAP + text}]}


def _assistant(text: str) -> dict[str, Any]:
    return {"role": "assistant", "content": [{"type": "text", "text": text}]}


def _tool_result(text: str) -> dict[str, Any]:
    block = {"type": "tool_result", "tool_use_id": "x", "content": text}
    return {"role": "user", "content": [block]}


def _layout(tmp_path: Path, sid: str) -> SessionLayout:
    layout = SessionLayout(state_dir=tmp_path, session_id=sid, subdir="runs")
    layout.ensure()
    return layout


def _checkpoint(layout: SessionLayout, turn: int, messages: list[dict[str, Any]]) -> None:
    snap = SessionSnapshot(
        system="s",
        messages=messages,
        tool_calls=0,
        next_iteration=turn,
        root_task_id=None,
        original_task="do the thing",
        verify_command=(),
    )
    layout.checkpoint_path(turn).write_text(snap.model_dump_json(), encoding="utf-8")


def test_undo_walks_back_one_operator_message(tmp_path: Path) -> None:
    layout = _layout(tmp_path, "run-a")
    _checkpoint(layout, 1, [_task("do the thing"), _assistant("working")])
    _checkpoint(
        layout,
        2,
        [_task("do the thing"), _assistant("working"), _steer("focus the parser")],
    )
    _checkpoint(
        layout,
        3,
        [
            _task("do the thing"),
            _assistant("working"),
            _steer("focus the parser"),
            _tool_result("read 12 bytes"),
            _assistant("done"),
        ],
    )
    target = undo_target(tmp_path, "run-a")
    assert target is not None
    assert (target.source_session_id, target.at_turn) == ("run-a", 1)
    assert target.undone_text == "focus the parser"


def test_undo_with_only_the_task_restarts_from_the_first_checkpoint(tmp_path: Path) -> None:
    """The composer gets the operator's words back, never the skill block or
    digest composed in front of them (here with no manifest to read them
    from, so they come out of the checkpoint's opening message)."""
    from agent6.task_text import SKILLS_PREAMBLE

    composed = f'{SKILLS_PREAMBLE}\n<skill name="tidy">be tidy</skill>\n---\ndo the thing'
    layout = _layout(tmp_path, "run-b")
    _checkpoint(layout, 1, [_task(composed)])
    _checkpoint(layout, 2, [_task(composed), _assistant("lots of work")])
    target = undo_target(tmp_path, "run-b")
    assert target is not None
    assert (target.source_session_id, target.at_turn) == ("run-b", 1)
    assert target.undone_text == "do the thing"


def test_undo_refuses_at_the_opening_message(tmp_path: Path) -> None:
    layout = _layout(tmp_path, "run-c")
    _checkpoint(layout, 1, [_task("do the thing")])
    said: list[str] = []
    from agent6.app.reporter import Reporter

    reporter = Reporter(out=said.append, err=said.append)
    assert undo_target(tmp_path, "run-c", reporter=reporter) is None
    assert any("nothing to undo" in line for line in said)


def test_repeated_undo_follows_the_fork_lineage(tmp_path: Path) -> None:
    """A fork carries one seed checkpoint; the next /undo resolves in the
    parent it was cut from -- the walk-back that makes B -> C -> D chains work."""
    parent = _layout(tmp_path, "run-p")
    _checkpoint(parent, 1, [_task("do the thing")])
    _checkpoint(parent, 2, [_task("do the thing"), _steer("first steer")])
    _checkpoint(parent, 3, [_task("do the thing"), _steer("first steer"), _steer("second steer")])
    child = _layout(tmp_path, "run-q")
    # The seed a /undo of run-p would have cut: turn 2's conversation.
    _checkpoint(child, 0, [_task("do the thing"), _steer("first steer")])
    write_session_manifest(
        child,
        session_id="run-q",
        user_task="do the thing",
        base_sha="",
        base_branch="",
        run_branch=None,
        cfg=Config(),
        mode="run",
        effective_preset="",
        preset_from_flag=False,
        parent_session_id="run-p",
    )
    target = undo_target(tmp_path, "run-q")
    assert target is not None
    assert (target.source_session_id, target.at_turn) == ("run-p", 1)
    assert target.undone_text == "first steer"


def test_a_cyclic_lineage_does_not_crash(tmp_path: Path) -> None:
    """A corrupt/hand-edited manifest whose parent_session_id points at itself
    (or forms a cycle) must not recurse forever: the lineage walk ends on a
    revisited id and /undo refuses cleanly instead of a RecursionError."""
    layout = _layout(tmp_path, "cyclic-run")
    # Only the task in the checkpoint, so the resolver must walk to the parent.
    _checkpoint(layout, 0, [_task("do the thing"), _steer("focus the parser")])
    write_session_manifest(
        layout,
        session_id="cyclic-run",
        user_task="do the thing",
        base_sha="",
        base_branch="",
        run_branch=None,
        cfg=Config(),
        mode="run",
        effective_preset="",
        preset_from_flag=False,
        parent_session_id="cyclic-run",  # points at itself
    )
    said: list[str] = []
    from agent6.app.reporter import Reporter

    reporter = Reporter(out=said.append, err=said.append)
    assert undo_target(tmp_path, "cyclic-run", reporter=reporter) is None
    assert any("nothing to undo" in line for line in said)
