# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""`agent6 prompt show` assembles the prompt a run receives: the recorded
rulings (the `<decisions>` block) included, not only the memory index."""

from __future__ import annotations

import subprocess as sp
from pathlib import Path

from agent6.config import Config
from agent6.memory import record_decision
from agent6.workflows import system_prompt_for


def test_prompt_show_carries_the_recorded_decisions(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    sp.run(["git", "init", "-q"], cwd=root, check=True)
    state_dir = tmp_path / "state"
    record_decision(state_dir, question="Which greeting?", answer="Hi NAME", session="s1")
    cfg = Config.model_validate({"prompt": {"structural_priors": False}})
    prompt = system_prompt_for(cfg, root, "run", state_dir=state_dir)
    assert "<decisions>" in prompt
    assert "A: Hi NAME" in prompt
