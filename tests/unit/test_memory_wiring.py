# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Memory injection and the tool-side grant.

The index is the recall surface (injected per mode); the files are reached
with the ordinary in-process tools through a narrow grant that lifts the
state-dir denial for exactly the memory dir and nothing else."""

from __future__ import annotations

from pathlib import Path

import pytest

from agent6.config import Config
from agent6.memory import add, memory_dir
from agent6.tools.dispatch import ToolDispatcher
from agent6.tools.errors import ToolError
from agent6.types import RepoSummary
from agent6.workflows import _prompt_blocks as pb


def _repo(root: Path) -> RepoSummary:
    return RepoSummary(
        root=root,
        branch="main",
        head_sha="0" * 40,
        file_count=0,
        top_level=(),
        agents_md="",
        recent_log="",
    )


def _build(mode: str, index: str, tmp_path: Path) -> str:
    return pb.build_system_prompt(
        config=Config.model_validate({"workflow": {"verify_command": ["true"]}}),
        repo=_repo(tmp_path),
        mode=mode,  # pyright: ignore[reportArgumentType]
        memory_index=index,
        memory_dir_path="/state/memory",
        skills=None,
    )


def test_run_mode_always_carries_the_memory_header(tmp_path: Path) -> None:
    out = _build("run", "", tmp_path)
    assert "<memory>" in out
    assert "(none recorded yet)" in out
    assert "apply_edit" in out.split("<memory>")[1].split("</memory>")[0]


def test_index_content_renders_and_clips(tmp_path: Path) -> None:
    out = _build("run", "- fact: the hook", tmp_path)
    assert "- fact: the hook" in out
    big = "\n".join(f"- fact-{i}: {'x' * 80}" for i in range(200))
    out = _build("run", big, tmp_path)
    assert "index clipped" in out


def test_readonly_modes_render_only_with_content(tmp_path: Path) -> None:
    assert "<memory>" not in _build("plan", "", tmp_path)
    assert "<memory>" in _build("plan", "- fact: hook", tmp_path)
    assert "<memory>" not in _build("ask", "", tmp_path)


def test_agent_mode_never_sees_memory(tmp_path: Path) -> None:
    assert "<memory>" not in _build("agent", "- fact: hook", tmp_path)


def _dispatcher(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[ToolDispatcher, Path]:
    """The REAL topology: the state dir sits under the hidden state home, so
    the memory grant must beat the denial (it silently did not, once -- the
    resolve-path denial check bypassed the exemption, and a tmp state dir
    outside the hidden set let this file's tests pass vacuously)."""
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "code.py").write_text("x = 1\n")
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "statehome"))
    state = tmp_path / "statehome" / "repo-id"
    add(state, "seeded", "a seeded fact")
    cfg = Config.model_validate({"sandbox": {"isolation": "none"}})
    return ToolDispatcher(root=repo, config=cfg, state_dir=state), state


def test_tools_reach_the_memory_dir_and_nothing_else_in_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The carve-out pin: memory files are readable and editable through the
    ordinary tools by absolute path, while the rest of the state dir stays
    refused (denied beats every grant, except this one exempt subtree)."""
    d, state = _dispatcher(tmp_path, monkeypatch)
    mem = memory_dir(state)

    out = d.dispatch("read_file", {"path": str(mem / "MEMORY.md")}).to_wire()
    assert "seeded" in out["content"]

    d.dispatch(
        "apply_edit",
        {
            "path": str(mem / "new-fact.md"),
            "edits": [{"kind": "create", "old_string": "", "new_string": "learned\n"}],
        },
    )
    assert (mem / "new-fact.md").read_text() == "learned\n"

    (state / "secretish.txt").write_text("run state\n")
    with pytest.raises(ToolError):
        d.dispatch("read_file", {"path": str(state / "secretish.txt")})
    with pytest.raises(ToolError):
        d.dispatch(
            "apply_edit",
            {
                "path": str(state / "planted.md"),
                "edits": [{"kind": "create", "old_string": "", "new_string": "x"}],
            },
        )


def test_memory_grant_absent_without_state_dir(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    cfg = Config.model_validate({"sandbox": {"isolation": "none"}})
    d = ToolDispatcher(root=repo, config=cfg)
    with pytest.raises(ToolError):
        d.dispatch("read_file", {"path": str(tmp_path / "state" / "memory" / "MEMORY.md")})


def test_the_memory_dir_exists_the_moment_the_grant_does(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The model cannot mkdir outside the jail, so a fresh repo's first
    organic memory write (apply_edit into <state>/memory/) failed ENOENT
    until the dispatcher created the dir it grants (caught live). Fresh
    means NO prior CLI write: the store must not exist beforehand."""
    repo = tmp_path / "repo"
    repo.mkdir()
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "statehome"))
    state = tmp_path / "statehome" / "repo-id"
    state.mkdir(parents=True)
    cfg = Config.model_validate({"sandbox": {"isolation": "none"}})
    assert not memory_dir(state).exists()
    ToolDispatcher(root=repo, config=cfg, state_dir=state)
    assert memory_dir(state).is_dir()


def test_a_memory_write_does_not_withdraw_a_green_verify(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """EditResult.path is store-relative, so the old exclusion (which tested
    result.path for an absolute prefix) never matched: the model's memory
    write after a green verify counted as a tree edit, and the run ended
    gate_red_at_base with all_passed=false over the very suite it had just
    fixed (caught live). The predicate now judges the model's INPUT path."""
    from unittest.mock import MagicMock

    from agent6.memory import memory_dir
    from agent6.workflows.loop import (
        LoopState,
        TurnState,
        Workflow,
    )

    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "statehome"))
    state_dir = tmp_path / "statehome" / "repo-id"
    state_dir.mkdir(parents=True)

    wf = Workflow.__new__(Workflow)
    wf.state_dir = state_dir
    state = LoopState(original_task="t", tool_calls=0)
    state.verify.note_pass()
    turn = MagicMock(spec=TurnState)
    target = memory_dir(state_dir) / "fact.md"

    result = MagicMock()
    result.path = "fact.md"  # the store-relative spelling EditResult uses
    wf._note_tool_effects(  # pyright: ignore[reportPrivateUsage]
        state, turn, "apply_edit", result, {"path": str(target), "edits": []}
    )
    assert state.memory_written is True
    assert state.verify.green_and_untouched  # the green survived
    # A workspace edit still marks the tree.
    wf._note_tool_effects(  # pyright: ignore[reportPrivateUsage]
        state, turn, "apply_edit", result, {"path": "src/x.py", "edits": []}
    )
    assert not state.verify.green_and_untouched
