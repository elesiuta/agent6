# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta

"""A tool withheld from the model's list is refused when called anyway.

Withdrawal is not refusal. The list is rebuilt per turn, but the model still
carries the previous turn's list in its context, and a hallucinated name costs
nothing to emit -- so every reason `available_tool_names` drops a tool has to
be mirrored at the call gate. One reason (an MCP server the operator denied for
the session) was mirrored only in the listing, and calling its tool ran it.

This pins the property rather than the list: a new withholding reason that
forgets its call-gate guard fails here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import pytest

from agent6.config import Config
from agent6.sessions.ipc import set_session_deny
from agent6.tools.dispatch import ToolDispatcher
from agent6.tools.errors import ToolDenied, ToolError
from agent6.tools.mcp_client import MCPManager, MCPServerSpec
from agent6.tools.operator_prompts import ApprovalAnswer, ApprovalRequest, OperatorPrompts
from agent6.tools.schema import ALL_TOOLS
from agent6.workflows._toolset import tool_definitions
from tests.unit.test_mcp_client import _fake_server_argv  # pyright: ignore[reportPrivateUsage]

Mode = Literal["run", "plan", "ask", "machine", "agent"]

# The guards' own vocabulary. A refusal in these words came from a gate; a
# refusal in any other words (an argument that failed validation, a missing
# state dir) would pass this test without proving anything.
GUARD_WORDS = ("not available", "is disabled", "denied for this session", "Unknown tool")


def _always_approve(_request: ApprovalRequest, /) -> ApprovalAnswer:
    return ApprovalAnswer(True, "stdin")


def _assert_withheld_are_refused(d: ToolDispatcher, expected: set[str], mode: Mode = "run") -> None:
    """*expected* must be absent from the list the MODEL is handed
    (`tool_definitions`, the mode's surface filtered by the dispatcher), and
    calling each anyway must hit a gate."""
    offered = {t.name for t in tool_definitions(d, mode=mode)}
    withheld = {cls.TOOL_NAME for cls in ALL_TOOLS} - offered
    assert expected <= withheld, f"expected these withheld: {expected - withheld}"
    for name in sorted(expected):
        with pytest.raises((ToolError, ToolDenied)) as caught:
            d.dispatch(name, {})
        message = str(caught.value)
        assert any(w in message for w in GUARD_WORDS), (
            f"{name} was withheld but the call was refused for an unrelated reason: {message}"
        )


def test_a_command_tool_withheld_by_policy_is_refused(tmp_path: Path) -> None:
    d = ToolDispatcher(
        root=tmp_path,
        config=Config.model_validate({"sandbox": {"run_commands": "no"}}),
        isolation="none",
    )
    _assert_withheld_are_refused(d, {"run_command", "stop_background"})


def test_fetch_withheld_because_commands_have_the_network_is_refused(tmp_path: Path) -> None:
    d = ToolDispatcher(
        root=tmp_path,
        config=Config.model_validate({"sandbox": {"network": "host"}}),
        isolation="none",
    )
    _assert_withheld_are_refused(d, {"fetch"})


def test_tools_withheld_by_a_bench_switch_are_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A measurement the model can contaminate measures nothing."""
    monkeypatch.setenv("AGENT6_SYMBOL_TOOLS", "none")
    monkeypatch.setenv("AGENT6_DISABLE_APPLY_EDIT", "1")
    d = ToolDispatcher(root=tmp_path, config=Config(), isolation="none")
    _assert_withheld_are_refused(d, {"outline", "find_definition", "find_references", "apply_edit"})


def test_tools_withheld_by_the_mode_are_refused(tmp_path: Path) -> None:
    """`agent6 ask` edits nothing, and that has to be true of the dispatcher,
    not only of the list it advertises. (It keeps `run_command`: read-only,
    approval-gated investigation is the mode's whole job.)"""
    d = ToolDispatcher(root=tmp_path, config=Config(), isolation="none", mode="ask")
    _assert_withheld_are_refused(d, {"apply_edit", "apply_patch", "stop_background"}, mode="ask")


def test_a_machine_state_reaches_neither_the_repo_nor_the_network(tmp_path: Path) -> None:
    """The strictest surface: its deliverable is the finish_session payload."""
    d = ToolDispatcher(root=tmp_path, config=Config(), isolation="none", mode="machine")
    _assert_withheld_are_refused(
        d,
        {"apply_edit", "run_command", "run_verify_command", "fetch", "read_session"},
        mode="machine",
    )


def test_an_mcp_server_denied_for_the_session_is_refused(tmp_path: Path) -> None:
    session_dir = tmp_path / "session"
    session_dir.mkdir()
    mgr = MCPManager.start(
        [
            MCPServerSpec(
                name="fake", command=_fake_server_argv(), startup_timeout_s=5.0, call_timeout_s=5.0
            )
        ]
    )
    try:
        d = ToolDispatcher(
            root=tmp_path,
            config=Config.model_validate(
                {"mcp": {"enabled": True, "servers": {"fake": {"command": ["true"]}}}}
            ),
            isolation="none",
            mcp_manager=mgr,
            session_dir=session_dir,
            prompts=OperatorPrompts(approver=_always_approve),
        )
        assert "mcp__fake__echo" in d.available_tool_names()
        set_session_deny(session_dir, "mcp.fake")
        assert "mcp__fake__echo" not in d.available_tool_names()
        with pytest.raises(ToolError, match="denied for this session"):
            d.dispatch("mcp__fake__echo", {"text": "hi"})
    finally:
        mgr.close()
