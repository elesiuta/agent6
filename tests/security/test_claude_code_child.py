# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The claude_code child's boundary: argv carries operator config and literals
only, never a widening flag; the environment carries no credential or Claude
Code session variable from the operator shell."""

from __future__ import annotations

from pathlib import Path

import pytest

from agent6.providers._claude_code_wire import CLAUDE_CODE_ENV, child_env, claude_argv

# Flags that would widen the child: its own tools, a permission mode, a settings
# source, another system prompt, a resumed session, or a second budget meter.
_FORBIDDEN = (
    "--dangerously-skip-permissions",
    "--allow-dangerously-skip-permissions",
    "--permission-mode",
    "--permission-prompt-tool",
    "--bare",
    "--safe-mode",
    "--restricted",
    "--add-dir",
    "--settings",
    "--append-system-prompt",
    "--append-system-prompt-file",
    "--system-prompt",
    "--resume",
    "--continue",
    "--session-id",
    "--max-budget-usd",
    "--max-turns",
    "--autocompact",
    "--fallback-model",
    "--plugin-dir",
)


def test_argv_carries_the_default_deny_flags_and_no_widening_flag() -> None:
    argv = claude_argv("claude", "claude-sonnet-4-5", "high", Path("/private/system_prompt.txt"))
    assert not set(argv) & set(_FORBIDDEN)
    assert argv[argv.index("--tools") + 1] == ""
    assert argv[argv.index("--setting-sources") + 1] == ""
    assert argv[argv.index("--allowedTools") + 1] == "mcp__agent6"
    assert "--strict-mcp-config" in argv and "--disable-slash-commands" in argv
    assert "--no-session-persistence" in argv
    assert argv[argv.index("--system-prompt-file") + 1] == "/private/system_prompt.txt"
    assert argv[argv.index("--effort") + 1] == "high"
    assert "--effort" not in claude_argv("claude", "m", None, Path("/p"))


def test_child_env_carries_no_credential_or_claude_session_variable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in (
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_AUTH_TOKEN",
        "ANTHROPIC_BASE_URL",
        "CLAUDE_CODE_OAUTH_TOKEN",
        "CLAUDE_CODE_USE_BEDROCK",
        "CLAUDE_CODE_USE_VERTEX",
        "CLAUDECODE",
        "CLAUDE_CODE_SESSION_ID",
        "CLAUDE_CODE_MESSAGING_TOKEN",
        "DBUS_SESSION_BUS_ADDRESS",
        "XDG_RUNTIME_DIR",
    ):
        monkeypatch.setenv(name, "leak")
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", "/cfg")
    monkeypatch.setenv("HOME", "/home/op")
    env = child_env()
    assert not any(v == "leak" for v in env.values())
    assert not any(k.startswith("ANTHROPIC") for k in env)
    assert {k for k in env if k.startswith("CLAUDE")} == {
        "CLAUDE_CONFIG_DIR",
        *(k for k in CLAUDE_CODE_ENV if k.startswith("CLAUDE")),
    }
    assert env["HOME"] == "/home/op"
    for name, value in CLAUDE_CODE_ENV.items():
        assert env[name] == value
