# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Startup-warning helpers in app/preflight."""

from __future__ import annotations

from pathlib import Path

import pytest

from agent6.app.preflight import headless_approval_refusal
from agent6.config import Config


def _ask_cfg() -> Config:
    return Config.model_validate({"sandbox": {"run_commands": "ask"}})


def test_a_run_that_cannot_be_asked_refuses_instead_of_hanging(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`ask` needs someone to answer. With no terminal, no TUI and no away-mode
    the first command waits forever -- and since the verify gate is a command
    too, that is essentially every run, every /parallel lane included. It used
    to print a note and hang anyway."""
    refusal = headless_approval_refusal(_ask_cfg(), tui_enabled=False, away="", can_ask=False)
    assert refusal is not None
    assert "would wait forever" in refusal
    assert "--auto-approve" in refusal  # the fix is named


def test_a_clamped_session_kind_names_the_flag_not_the_config_value() -> None:
    """plan and ask clamp a standing run_commands = "yes" to ask: the remedy
    names --auto-approve and the clamp, never the value that is already set."""
    refusal = headless_approval_refusal(
        _ask_cfg(), tui_enabled=False, away="", can_ask=False, clamped=True
    )
    assert refusal is not None
    assert "clamps a standing sandbox.run_commands = 'yes'" in refusal
    assert "sandbox.run_commands = 'yes' (or --auto-approve)" not in refusal


@pytest.mark.parametrize(
    ("tui", "away", "commands", "can_ask"),
    [
        (True, "", "ask", False),  # a TUI can answer
        (False, "wait", "ask", False),  # an away-mode says what an absent operator meant
        (False, "deny", "ask", False),  # ... including "deny", which a btw uses
        (False, "", "yes", False),  # nothing to approve
        (False, "", "no", False),  # commands withheld entirely
        (False, "", "ask", True),  # the front-end asks out of band (a terminal, ACP)
    ],
)
def test_answerable_runs_are_not_refused(
    tui: bool, away: str, commands: str, can_ask: bool
) -> None:
    cfg = Config.model_validate({"sandbox": {"run_commands": commands}})
    assert headless_approval_refusal(cfg, tui_enabled=tui, away=away, can_ask=can_ask) is None


def test_the_lifecycle_sets_the_repos_hook_policy_itself(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`ui/cli` set it and resume set it, but `run_task` did not -- so a
    front-end that calls the lifecycle directly (`agent6 acp`) left a repo that
    opted into its own hooks with them silently off. It fails SAFE, which is
    how a knob `config show` reports went ignored on one surface unnoticed."""
    from agent6.app import preflight as preflight_mod
    from agent6.app import run as lifecycle
    from agent6.app.frontend import FrontendCapabilities

    seen: list[bool] = []

    def _capture(captured: Config) -> None:
        seen.append(captured.git.run_repo_hooks)

    monkeypatch.setattr(preflight_mod, "apply_git_ops_policy", _capture)
    monkeypatch.chdir(tmp_path)
    cfg = Config.model_validate({"git": {"run_repo_hooks": True}})
    # It refuses immediately after (no git identity here); the policy is set
    # before anything git touches the repo, which is the point.
    from agent6.app.reporter import Reporter
    from agent6.ui.acp.frontend import acp_frontend

    front = acp_frontend(
        ask=lambda _p, _o, _s: None,
        capabilities=FrontendCapabilities(),
        agent6_exe=lambda: "agent6",
        spawn_detached_resume=lambda _cwd, _rid, _flags: "",
    )
    said: list[str] = []
    lifecycle.run_task(
        cfg, "t", frontend=front, reporter=Reporter(out=said.append, err=said.append)
    )
    assert seen == [True], said
