# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The first user message's operational header matches the mode's REAL tool
surface.

Ask used to fall into run's else-branch, telling the model to edit, run
verify, and call `finish_session` -- none of which ask exposes -- and a
`run_commands = "no"` run was told to run a verify gate that withholding
commands removed. A paid live ask reproduced the confusion: the model spent a
call trying to comply with instructions for tools it lacked."""

from __future__ import annotations

import re

import pytest

from agent6.tools.schema import mode_tools
from agent6.workflows._prompt_blocks import initial_instructions


def test_ask_gets_direct_answer_instructions() -> None:
    text = initial_instructions("ask", "ask", has_gate=True)
    assert "answer" in text.lower()
    for phantom in ("finish_session", "make edits", "run_verify_command"):
        assert phantom not in text


def test_a_no_commands_run_is_not_told_to_run_verify() -> None:
    """`run_commands = "no"` withholds the command tools and the verify gate
    with them (the config field's own contract)."""
    assert "run_verify_command" not in initial_instructions("run", "no", has_gate=True)
    assert "finish_session" in initial_instructions("run", "no", has_gate=True)
    assert "run_verify_command" in initial_instructions("run", "ask", has_gate=True)
    assert "run_verify_command" in initial_instructions("run", "yes", has_gate=True)


def test_a_gateless_run_is_not_told_to_run_verify() -> None:
    """A gateless run has no verify gate however commands are configured;
    its header said "run_verify_command" anyway and the no-verify block had to
    disarm it. The header keys on the gate."""
    assert "run_verify_command" not in initial_instructions("run", "yes", has_gate=False)
    assert "finish_session" in initial_instructions("run", "yes", has_gate=False)


@pytest.mark.parametrize("mode", ["run", "plan", "ask", "agent"])
def test_no_instruction_names_a_tool_outside_the_mode_surface(mode: str) -> None:
    """The drift guard: every backticked tool name in a mode's header must be
    one that mode actually exposes, so the ladder and tools/schema.py cannot
    disagree again."""
    all_tools = set().union(
        *(mode_tools(m).permitted for m in ("run", "plan", "ask", "machine", "agent"))
    )
    permitted = mode_tools(mode).permitted
    for rc in ("yes", "ask", "no"):
        named = set(re.findall(r"`([a-z_0-9.]+)`", initial_instructions(mode, rc, has_gate=True)))
        misfits = {n for n in named if n in all_tools and n not in permitted}
        assert not misfits, f"{mode} header names tools outside its surface: {sorted(misfits)}"
