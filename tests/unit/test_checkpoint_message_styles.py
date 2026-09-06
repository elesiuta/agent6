# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""`[git.commit.checkpoint].message` styles at the loop's auto-commit."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from agent6.config import Config
from agent6.tools.dispatch import ToolDispatcher
from agent6.workflows import loop as loopmod
from agent6.workflows.loop import (
    TurnState,
    Workflow,
)


def _wf(tmp_path: Path, style: str, provider: Any = None, logger: Any = print) -> Workflow:
    cfg = Config.model_validate({"git": {"commit": {"checkpoint": {"message": style}}}})
    return Workflow(
        root=tmp_path,
        config=cfg,
        provider=provider or MagicMock(),
        dispatcher=ToolDispatcher(root=tmp_path, config=cfg),
        logger=logger,
    )


def _turn(text: str) -> TurnState:
    return TurnState(iteration=3, resp=MagicMock(text=text), assistant=MagicMock())


def test_agent6_style_is_the_default_and_unchanged(tmp_path: Path) -> None:
    wf = _wf(tmp_path, "agent6")
    got = wf._checkpoint_subject(  # pyright: ignore[reportPrivateUsage]
        _turn("Add the unified write path.\nmore prose"), fallback="verify passed"
    )
    assert got == "agent6 iter 3: Add the unified write path."


def test_conventional_style_derives_from_the_worktree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _one_added(_p: Path) -> tuple[tuple[str, str], ...]:
        return (("A", "src/agent6/config/write.py"),)

    monkeypatch.setattr(loopmod, "worktree_name_status", _one_added)
    wf = _wf(tmp_path, "conventional")
    got = wf._checkpoint_subject(  # pyright: ignore[reportPrivateUsage]
        _turn("Add the unified write path."), fallback="verify passed"
    )
    assert got == "feat(config): add the unified write path"


def test_model_style_uses_the_provider_text(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _one_modified(_p: Path) -> tuple[tuple[str, str], ...]:
        return (("M", "a.py"),)

    monkeypatch.setattr(loopmod, "worktree_name_status", _one_modified)
    provider = MagicMock()
    provider.call.return_value = MagicMock(text=" fix: tighten the resolver \n")
    wf = _wf(tmp_path, "model", provider=provider)
    got = wf._checkpoint_subject(_turn("prose"), fallback="verify passed")  # pyright: ignore[reportPrivateUsage]
    assert got == "fix: tighten the resolver"


def test_model_style_degrades_to_agent6_with_a_warning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _one_modified(_p: Path) -> tuple[tuple[str, str], ...]:
        return (("M", "a.py"),)

    monkeypatch.setattr(loopmod, "worktree_name_status", _one_modified)
    provider = MagicMock()
    provider.call.side_effect = RuntimeError("no endpoint")
    logged: list[str] = []
    wf = _wf(tmp_path, "model", provider=provider, logger=logged.append)
    got = wf._checkpoint_subject(_turn("Fix the thing."), fallback="verify passed")  # pyright: ignore[reportPrivateUsage]
    assert got == "agent6 iter 3: Fix the thing."
    assert any("model commit message failed" in m for m in logged)
