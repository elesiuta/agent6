# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Truthfulness pin: plan/ask do not promise a read-only WORKSPACE.

Both modes withhold the structured edit tools (apply_edit/apply_patch), commit
tools, and the task DAG, but a jailed `run_command` can still write the
workspace as a side effect. So an absolute "you CANNOT change anything" or a
"read-only guarantee" is a lie; the prompts state the truth: commands run
jailed, a probe's writes land in the workspace, and nothing carries them
forward.

plan is clamped like ask: a standing run_commands="yes" becomes "ask", so a
write during planning is operator-approved, never auto-run.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from agent6.config import Config, load_config
from agent6.types import RepoSummary
from agent6.workflows.loop import build_system_prompt  # pyright: ignore[reportPrivateUsage]

_INTERACTIVE: tuple[Literal["plan", "ask"], ...] = ("plan", "ask")

_CONFIG_TOML = "\n".join(
    (
        "[agent6]",
        "config_version = 1",
        "[providers.anthropic]",
        'api_format = "anthropic"',
        'api_key_env = "ANTHROPIC_API_KEY"',
        "[models.worker]",
        'provider = "anthropic"',
        'model = "x"',
    )
)


def _prompt(tmp_path: Path, mode: Literal["plan", "ask"]) -> str:
    cfg_path = tmp_path / "agent6.toml"
    cfg_path.write_text(_CONFIG_TOML, encoding="utf-8")
    repo = RepoSummary(
        root=tmp_path,
        branch="",
        head_sha="",
        file_count=0,
        top_level=("notes.txt",),
        agents_md="",
        recent_log="",
        is_git=False,
    )
    return build_system_prompt(config=load_config(cfg_path), repo=repo, mode=mode, skills=None)


def _norm(tmp_path: Path, mode: Literal["plan", "ask"]) -> str:
    """The prompt lower-cased with runs of whitespace collapsed, so a substring
    check does not depend on where the prose happens to wrap."""
    return " ".join(_prompt(tmp_path, mode).lower().split())


def test_plan_and_ask_prompts_do_not_promise_a_read_only_workspace(tmp_path: Path) -> None:
    for mode in _INTERACTIVE:
        low = _norm(tmp_path, mode)
        # A jailed run_command can write the workspace, so no false absolute.
        assert "cannot change anything" not in low, mode
        assert "cannot edit files" not in low, mode
        assert "read-only guarantee" not in low, mode
        # The truth is stated: commands run jailed and their writes land in
        # the workspace.
        assert "jailed" in low, mode
        assert "land in the workspace" in low, mode


def test_plan_and_ask_prompts_state_that_probe_writes_go_nowhere(tmp_path: Path) -> None:
    """Withhold the false guarantee and state the consequence: nothing carries
    a probe's writes forward, so an edit the answer or plan needs is described
    or recorded, never applied."""
    for mode in _INTERACTIVE:
        low = _norm(tmp_path, mode)
        assert "nothing carries them forward" in low, mode


def test_plan_clamps_run_commands_like_ask(tmp_path: Path) -> None:
    """plan runs with the operator present, so a standing
    run_commands="yes" is clamped to "ask" exactly as in ask -- a write during
    planning is approved per call, never auto-run. The clamp only tightens (a
    configured "no" stays "no"), and an allow-all session answer upgrades the
    clamped "ask" back to "yes"."""
    from agent6.app._setup import session_config
    from agent6.sessions.ipc import COMMAND_SCOPE, effective_run_commands, set_session_allow

    yes = Config.model_validate({"sandbox": {"run_commands": "yes"}})
    # Both interactive modes clamp yes->ask; run keeps the operator's yes.
    assert session_config(yes, "plan").sandbox.run_commands == "ask"
    assert session_config(yes, "ask").sandbox.run_commands == "ask"
    assert session_config(yes, "run").sandbox.run_commands == "yes"
    # Only tightens: a withheld "no" is never loosened by the clamp.
    no = Config.model_validate({"sandbox": {"run_commands": "no"}})
    assert session_config(no, "plan").sandbox.run_commands == "no"
    # The clamped "ask" still upgrades to "yes" for the session on allow-all.
    assert effective_run_commands("ask", tmp_path) == "ask"
    set_session_allow(tmp_path, COMMAND_SCOPE)
    assert effective_run_commands("ask", tmp_path) == "yes"
