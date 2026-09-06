# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""One effective command policy, from three inputs, read the same way everywhere."""

from __future__ import annotations

from pathlib import Path

import pytest

from agent6.config import Config
from agent6.sessions.ipc import (
    COMMAND_SCOPE,
    effective_run_commands,
    set_away_mode,
    set_session_allow,
    set_session_deny,
)
from agent6.tools.dispatch import ToolDispatcher
from agent6.tools.operator_prompts import ApprovalAnswer, ApprovalRequest, OperatorPrompts

_COMMAND_TOOLS = {"run_command", "run_verify_command", "stop_background"}


@pytest.mark.parametrize("configured", ["yes", "no"])
def test_a_standing_policy_is_not_movable_in_run(tmp_path: Path, configured: str) -> None:
    """Only "ask" is a question. A configured yes or no is the operator's
    standing policy, and no in-run choice overrides it."""
    set_session_allow(tmp_path, COMMAND_SCOPE)
    set_session_deny(tmp_path, COMMAND_SCOPE)
    set_away_mode(tmp_path, "deny")
    assert effective_run_commands(configured, tmp_path) == configured


def test_ask_is_what_the_session_choice_moves(tmp_path: Path) -> None:
    assert effective_run_commands("ask", tmp_path) == "ask"
    set_session_allow(tmp_path, COMMAND_SCOPE)
    assert effective_run_commands("ask", tmp_path) == "yes"


def test_deny_for_the_session_is_the_mirror_of_allow(tmp_path: Path) -> None:
    """A single no answers one call, exactly as a single yes approves one; only
    the session choices persist, and denying withdraws rather than refuses."""
    set_session_deny(tmp_path, COMMAND_SCOPE)
    assert effective_run_commands("ask", tmp_path) == "no"


def test_an_away_mode_of_deny_withdraws_the_tools(tmp_path: Path) -> None:
    """Same wiring: "deny while away" and "deny for the session" and
    `run_commands = "no"` all mean the tools are gone, not refused per call."""
    set_away_mode(tmp_path, "deny")
    assert effective_run_commands("ask", tmp_path) == "no"


def test_waiting_is_still_a_question(tmp_path: Path) -> None:
    set_away_mode(tmp_path, "wait")
    assert effective_run_commands("ask", tmp_path) == "ask"


def test_withdrawn_tools_leave_the_model_s_surface(tmp_path: Path) -> None:
    """The point of withdrawing rather than refusing: the model never sees a
    door it cannot open, so it stops spending turns on one."""
    # A gate must be configured, or run_verify_command is hidden for its own
    # reason (a gateless run is not offered a tool that would only error).
    cfg = Config.model_validate(
        {"sandbox": {"run_commands": "ask"}, "workflow": {"verify_command": ["true"]}}
    )
    d = ToolDispatcher(root=tmp_path, config=cfg, session_dir=tmp_path)
    assert set(d.available_tool_names()) >= _COMMAND_TOOLS
    set_session_deny(tmp_path, COMMAND_SCOPE)
    assert _COMMAND_TOOLS.isdisjoint(d.available_tool_names())


def test_the_policy_is_re_read_not_cached(tmp_path: Path) -> None:
    """An operator who allows for the session stops being prompted from the
    next call, without restarting anything."""
    cfg = Config.model_validate({"sandbox": {"run_commands": "ask"}})
    d = ToolDispatcher(root=tmp_path, config=cfg, session_dir=tmp_path)
    assert d.command_policy() == "ask"
    set_session_allow(tmp_path, COMMAND_SCOPE)
    assert d.command_policy() == "yes"


@pytest.mark.parametrize(
    ("commands", "refused"),
    [("ask", True), ("yes", False), ("no", False)],
)
def test_parallel_makes_the_operator_decide_once(commands: str, refused: bool) -> None:
    """ "Wait for someone to approve" is incoherent across detached lanes: it
    would mean attaching a front-end to each in turn, which is most of what
    running them in parallel was for. So `ask` refuses at launch and names the
    two coherent choices."""
    from agent6.ui.cli.parallel import (
        _parallel_approval_refusal,  # pyright: ignore[reportPrivateUsage]
    )

    cfg = Config.model_validate({"sandbox": {"run_commands": commands}})
    err = _parallel_approval_refusal(cfg)
    assert (err is not None) is refused
    if err is not None:
        assert "--auto-approve" in err and "--no-commands" in err
        # A hub (TUI/web new task) relays this refusal and has no flags to
        # pass; the config remedy is the one it can act on.
        assert "agent6 config set sandbox.run_commands" in err


def test_a_single_no_refuses_one_call_and_withdraws_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The asymmetry that matters: "no" to THIS command is not "no commands".
    Only deny-for-session, `run_commands = "no"` and `--no-commands` withdraw
    the tools; a single answer -- either way -- decides a single call."""
    from agent6.sandbox.jail import CommandResult
    from agent6.sessions.ipc import session_deny_set
    from agent6.types import JailPolicy

    cfg = Config.model_validate(
        {"sandbox": {"run_commands": "ask"}, "workflow": {"verify_command": ["true"]}}
    )
    answers = iter(["no", "no", "yes"])

    def _answer(_request: ApprovalRequest, /) -> ApprovalAnswer:
        return ApprovalAnswer(next(answers) == "yes", "stdin")

    prompts = OperatorPrompts(approver=_answer, session_dir=tmp_path)
    d = ToolDispatcher(root=tmp_path, config=cfg, session_dir=tmp_path, prompts=prompts)
    for _ in range(2):
        with pytest.raises(Exception, match="not approved"):
            d.dispatch("run_command", {"argv": ["true"]})
        assert not session_deny_set(tmp_path, COMMAND_SCOPE)
        assert d.command_policy() == "ask"
        assert set(d.available_tool_names()) >= _COMMAND_TOOLS

    # The mirror: the third call's single "yes" runs exactly that call (the
    # jail stubbed out) and widens nothing either -- still "ask" for the next.
    def _ran(policy: JailPolicy, **_kw: object) -> CommandResult:
        return CommandResult(
            argv=tuple(policy.argv), returncode=0, stdout="", stderr="", duration_s=0.01
        )

    monkeypatch.setattr("agent6.tools.dispatch.run_in_jail", _ran)
    assert d.dispatch("run_command", {"argv": ["true"]}).to_wire()["returncode"] == 0
    assert d.command_policy() == "ask"
    assert not session_deny_set(tmp_path, COMMAND_SCOPE)


def test_a_stop_during_the_approval_wait_is_named_as_such(tmp_path: Path) -> None:
    """`sessions stop` reaches a run blocked on an approval by breaking the
    wait; the tool result then said "not approved (run_commands='ask')" as
    if the policy had refused. With a stop request pending it names the stop."""
    from agent6.sessions.ipc import request_stop

    cfg = Config.model_validate(
        {"sandbox": {"run_commands": "ask"}, "workflow": {"verify_command": ["true"]}}
    )

    def _wait_broken(_request: ApprovalRequest, /) -> ApprovalAnswer:
        request_stop(tmp_path)  # the stop lands while the approval waits
        return ApprovalAnswer(False, "stdin")

    prompts = OperatorPrompts(approver=_wait_broken, session_dir=tmp_path)
    d = ToolDispatcher(root=tmp_path, config=cfg, session_dir=tmp_path, prompts=prompts)
    with pytest.raises(Exception, match="asked to stop while awaiting approval"):
        d.dispatch("run_command", {"argv": ["true"]})
    with pytest.raises(Exception, match="asked to stop while awaiting approval"):
        d.dispatch("run_verify_command", {})
