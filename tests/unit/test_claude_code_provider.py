# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The claude_code provider against `fake_claude.py`, a stand-in binary that
speaks the stream-json protocol in the exact line order Claude Code 2.1.251
uses: argv and environment, the MCP handshake, one round per call with the
tool calls answered by the next call, the continuation rule, the restart
replay, plan-metered budgeting, and every failure mapped to ProviderError."""

from __future__ import annotations

import io
import json
import os
import select
import subprocess
import sys
import time
from pathlib import Path
from typing import IO, Any
from unittest.mock import MagicMock

import pytest

from agent6.app.providers import InstrumentedProvider, build_role_provider
from agent6.budget import BudgetTracker
from agent6.config import Config
from agent6.providers import (
    ProviderAborted,
    ProviderError,
    ProviderInterrupted,
    ToolDefinition,
    TranscriptSink,
    call_for_text,
    claude_code,
)
from agent6.providers._claude_code_wire import (
    CLAUDE_CODE_ENV,
    CLAUDE_CODE_RESULT_CAP_CHARS,
    claude_argv,
    plan_usage_from_rate_limit,
    render_history,
)
from agent6.providers.claude_code import ClaudeCodeProvider, login_status
from agent6.viewmodel.transcript_render import fold_conversation

FAKE = Path(__file__).with_name("fake_claude.py")
TOOLS = [
    ToolDefinition(
        name="read_file",
        description="Read a file.",
        input_schema={
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
            "additionalProperties": False,
        },
    ),
    ToolDefinition(
        name="finish_session",
        description="Finish.",
        input_schema={
            "type": "object",
            "properties": {
                "summary": {"type": "string"},
                "result": {"anyOf": [{"type": "object"}, {"type": "null"}]},
            },
            "additionalProperties": False,
        },
    ),
]
USER0: list[dict[str, Any]] = [
    {"role": "user", "content": [{"type": "text", "text": "TASK:\nwrite hello"}]}
]


def _install(tmp_path: Path, scenario: dict[str, Any]) -> tuple[str, Path]:
    """The fake as `<tmp>/bin/claude`; the stub carries the scenario and capture
    paths itself because the provider's curated env passes nothing else."""
    scen = tmp_path / "scenario.json"
    scen.write_text(json.dumps(scenario), encoding="utf-8")
    cap = tmp_path / "capture.jsonl"
    bindir = tmp_path / "bin"
    bindir.mkdir(exist_ok=True)
    stub = bindir / "claude"
    stub.write_text(
        f"#!/bin/sh\nFAKE_CLAUDE_SCENARIO={scen} FAKE_CLAUDE_CAPTURE={cap}"
        f' exec {sys.executable} {FAKE} "$@"\n',
        encoding="utf-8",
    )
    stub.chmod(0o755)
    return str(stub), cap


def _rescenario(tmp_path: Path, scenario: dict[str, Any]) -> None:
    (tmp_path / "scenario.json").write_text(json.dumps(scenario), encoding="utf-8")


def _captured(cap: Path) -> list[dict[str, Any]]:
    if not cap.exists():
        return []
    return [json.loads(line) for line in cap.read_text(encoding="utf-8").splitlines() if line]


def _spawns(cap: Path) -> list[dict[str, Any]]:
    return [c for c in _captured(cap) if "argv" in c]


def _stdin(cap: Path) -> list[dict[str, Any]]:
    return [c["stdin"] for c in _captured(cap) if "stdin" in c]


def _user_texts(cap: Path) -> list[str]:
    return [
        "".join(b.get("text", "") for b in line["message"]["content"])
        for line in _stdin(cap)
        if line.get("type") == "user"
    ]


def _tool_answers(cap: Path) -> list[list[dict[str, Any]]]:
    out: list[list[dict[str, Any]]] = []
    for line in _stdin(cap):
        rpc = ((line.get("response") or {}).get("response") or {}).get("mcp_response") or {}
        result = rpc.get("result") or {}
        if "content" in result:
            out.append(result["content"])
    return out


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def _budget() -> BudgetTracker:
    return BudgetTracker(max_usd=-1, max_tokens_fallback=-1, max_percent=-1)


def _provider(binary: str, **kw: Any) -> ClaudeCodeProvider:
    return ClaudeCodeProvider(model="claude-haiku-4-5", binary=binary, budget=_budget(), **kw)


def _round(**kw: Any) -> dict[str, Any]:
    return kw


def _usage(inp: int, out: int, read: int = 0, create: int = 0) -> dict[str, int]:
    return {
        "input_tokens": inp,
        "output_tokens": out,
        "cache_read_input_tokens": read,
        "cache_creation_input_tokens": create,
    }


def test_argv_is_operator_config_only_and_the_private_dir_is_removed_on_close(
    tmp_path: Path,
) -> None:
    binary, cap = _install(tmp_path, {"turns": [[_round(text="hi")]]})
    provider = _provider(binary, effort="low")
    system = "SYSTEM TEXT SECRET_SYS"
    provider.call(
        system=system,
        messages=[{"role": "user", "content": [{"type": "text", "text": "TASK SECRET_TASK"}]}],
        tools=TOOLS,
    )
    spawn = _spawns(cap)[0]
    argv = spawn["argv"]
    prompt_file = Path(argv[argv.index("--system-prompt-file") + 1])
    assert (binary, *argv) == claude_argv(binary, "claude-haiku-4-5", "low", prompt_file)
    assert all("SECRET_SYS" not in a and "SECRET_TASK" not in a for a in argv)
    assert spawn["system_prompt"] == system
    assert spawn["system_prompt_mode"] == 0o600
    assert spawn["cwd_entries"] == ["system_prompt.txt"]
    assert Path(spawn["cwd"]).name.startswith("agent6-claude-")
    assert _user_texts(cap) == ["TASK SECRET_TASK"]  # a one-message history is verbatim
    assert _alive(spawn["pid"])  # a tool-passing call keeps its session
    provider.close()
    assert not _alive(spawn["pid"])
    assert not Path(spawn["cwd"]).exists()


def test_child_env_is_curated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_BASE_URL",
        "CLAUDE_CODE_OAUTH_TOKEN",
        "CLAUDECODE",
        "CLAUDE_CODE_SESSION_ID",
        "CLAUDE_CODE_USE_BEDROCK",
        "DBUS_SESSION_BUS_ADDRESS",
        "XDG_RUNTIME_DIR",
    ):
        monkeypatch.setenv(name, "leak")
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path / "cc-config"))
    binary, cap = _install(tmp_path, {})
    _provider(binary).call(system="s", messages=USER0, tools=None)
    env = _spawns(cap)[0]["env"]
    assert not any(v == "leak" for v in env.values())
    assert "HOME" in env and "PATH" in env
    assert env["CLAUDE_CONFIG_DIR"] == str(tmp_path / "cc-config")
    for name, value in CLAUDE_CODE_ENV.items():
        assert env[name] == value


def test_handshake_answers_mcp_initialize_first_and_advertises_tools_verbatim(
    tmp_path: Path,
) -> None:
    binary, cap = _install(tmp_path, {"turns": [[_round(text="hi")]]})
    provider = _provider(binary)
    provider.call(system="s", messages=USER0, tools=TOOLS)
    lines = _stdin(cap)
    kinds = [line["type"] for line in lines]
    # our initialize, the MCP initialize answer (before the CLI answered ours), the user
    # message, then the notifications/initialized ack and the tools/list answer
    assert kinds[:3] == ["control_request", "control_response", "user"]
    listing = lines[4]["response"]["response"]["mcp_response"]["result"]["tools"]
    assert listing == [
        {"name": td.name, "description": td.description, "inputSchema": td.input_schema}
        for td in TOOLS
    ]
    provider.close()


def test_account_email_is_scrubbed_and_never_recorded(tmp_path: Path) -> None:
    email = "leak@example.test"
    binary, _ = _install(
        tmp_path,
        {
            "account_email": email,
            "turns": [[_round(thinking=f"user is {email}", text=f"Hi {email}, done")]],
        },
    )
    sink = TranscriptSink(tmp_path / "transcripts")
    provider = _provider(binary, transcript_sink=sink)
    resp = provider.call(system="s", messages=USER0, tools=None)
    assert resp.text == "Hi <operator-email>, done"
    assert resp.raw["content"][0]["thinking"] == "user is <operator-email>"
    for path in (tmp_path / "transcripts").glob("*.json"):
        assert email not in path.read_text(encoding="utf-8")
    assert "Leak Org" not in json.dumps(resp.raw)


def test_system_init_audit_refuses_a_foreign_tool_or_an_api_key(tmp_path: Path) -> None:
    binary, cap = _install(
        tmp_path,
        {"init": {"tools": ["mcp__agent6__read_file", "mcp__agent6__finish_session", "Bash"]}},
    )
    provider = _provider(binary)
    with pytest.raises(ProviderError, match="extra: Bash") as exc:
        provider.call(system="s", messages=USER0, tools=TOOLS)
    assert exc.value.fatal
    assert not _alive(_spawns(cap)[0]["pid"])
    _rescenario(tmp_path, {"init": {"apiKeySource": "ANTHROPIC_API_KEY"}})
    with pytest.raises(ProviderError, match="API key source") as exc:
        provider.call(system="s", messages=USER0, tools=TOOLS)
    assert exc.value.fatal


def test_tool_round_returns_at_message_stop_with_bare_names_and_plan_metered_budget(
    tmp_path: Path,
) -> None:
    binary, cap = _install(
        tmp_path,
        {
            "rate_limit": {"five_hour": 0.32, "seven_day": 0.51},
            "turns": [
                [
                    _round(
                        thinking="think",
                        tool_uses=[{"id": "toolu_1", "name": "read_file", "input": {"path": "a"}}],
                        usage=_usage(1200, 40, read=300),
                    )
                ]
            ],
        },
    )
    provider = _provider(binary)
    resp = provider.call(system="s", messages=USER0, tools=TOOLS)
    assert resp.stop_reason == "tool_use"
    assert resp.tool_uses == ({"id": "toolu_1", "name": "read_file", "input": {"path": "a"}},)
    assert resp.raw["content"] == [
        {"type": "thinking", "thinking": "think", "signature": "sig"},
        {"type": "tool_use", "id": "toolu_1", "name": "read_file", "input": {"path": "a"}},
    ]
    assert (resp.input_tokens, resp.output_tokens, resp.cache_read_tokens) == (1200, 40, 300)
    assert resp.cost_usd == 0.0
    assert _tool_answers(cap) == []  # the CLI is still blocked on tools/call
    budget = provider.budget
    assert budget is not None
    assert budget.estimate_usd() == (0.0, False)
    snap = budget.snapshot()
    assert snap.plan_latest is not None
    windows = {w.name: (w.used_percent, w.window_minutes) for w in snap.plan_latest.windows}
    assert windows == {"five_hour": (32.0, 300), "seven_day": (51.0, 10080)}
    assert snap.plan_latest.binding.name == "seven_day"
    assert snap.input_total == 1200 and snap.output_total == 40
    provider.close()


def test_next_call_answers_pending_calls_in_order_and_folds_notices(tmp_path: Path) -> None:
    binary, cap = _install(
        tmp_path,
        {
            "turns": [
                [
                    _round(
                        tool_uses=[
                            {"id": "toolu_1", "name": "read_file", "input": {"path": "a"}},
                            {"id": "toolu_2", "name": "read_file", "input": {"path": "b"}},
                        ]
                    ),
                    _round(text="after"),
                ]
            ]
        },
    )
    provider = _provider(binary)
    first = provider.call(system="s", messages=USER0, tools=TOOLS)
    history = [
        *USER0,
        {"role": "assistant", "content": first.raw["content"]},
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "toolu_1", "content": "R1"},
                {"type": "tool_result", "tool_use_id": "toolu_2", "content": "R2"},
                {"type": "text", "text": "NOTICE A"},
            ],
        },
        {"role": "user", "content": [{"type": "text", "text": "NOTICE B"}]},
    ]
    second = provider.call(system="s", messages=history, tools=TOOLS)
    assert second.text == "after" and second.stop_reason == "end_turn"
    assert _tool_answers(cap) == [
        [{"type": "text", "text": "R1"}],
        [
            {"type": "text", "text": "R2"},
            {"type": "text", "text": "NOTICE A"},
            {"type": "text", "text": "NOTICE B"},
        ],
    ]
    assert len(_spawns(cap)) == 1  # no respawn
    assert _user_texts(cap) == ["TASK:\nwrite hello"]  # notices rode the tool answer
    provider.close()


def test_prose_round_reads_result_then_a_notice_is_a_user_message_and_a_popped_turn_continues(
    tmp_path: Path,
) -> None:
    binary, cap = _install(
        tmp_path,
        {"turns": [[_round(text="first")], [_round(text="second")], [_round(text="third")]]},
    )
    provider = _provider(binary)
    first = provider.call(system="s", messages=USER0, tools=TOOLS)
    assert first.text == "first"
    history = [
        *USER0,
        {"role": "assistant", "content": first.raw["content"]},
        {"role": "user", "content": [{"type": "text", "text": "nudge"}]},
    ]
    assert provider.call(system="s", messages=history, tools=TOOLS).text == "second"
    # The loop popped the quiet "second" turn and nudged again.
    history.append({"role": "user", "content": [{"type": "text", "text": "nudge2"}]})
    assert provider.call(system="s", messages=history, tools=TOOLS).text == "third"
    assert _user_texts(cap) == ["TASK:\nwrite hello", "nudge", "nudge2"]
    assert len(_spawns(cap)) == 1
    provider.close()


def test_non_continuation_restarts_with_the_rendered_history(tmp_path: Path) -> None:
    binary, cap = _install(tmp_path, {"turns": [[_round(text="ok")]]})
    provider = _provider(binary)
    provider.call(system="s", messages=USER0, tools=TOOLS)
    resumed = [
        *USER0,
        {
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "hidden", "signature": "x"},
                {"type": "tool_use", "id": "toolu_x", "name": "read_file", "input": {"path": "a"}},
            ],
        },
        {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "toolu_x", "content": "R"}],
        },
    ]
    pid_before = _spawns(cap)[0]["pid"]
    provider.call(system="s", messages=resumed, tools=TOOLS)
    assert len(_spawns(cap)) == 2 and not _alive(pid_before)
    assert _user_texts(cap)[-1] == render_history(resumed)
    assert "hidden" not in _user_texts(cap)[-1]
    provider.call(
        system="s",
        messages=[
            *resumed,
            {"role": "assistant", "content": [{"type": "text", "text": "ok"}]},
            {"role": "user", "content": [{"type": "text", "text": "n"}]},
        ],
        tools=TOOLS[:1],
    )
    assert len(_spawns(cap)) == 3  # the tool list changed
    provider.call(system="other", messages=USER0, tools=TOOLS[:1])
    assert len(_spawns(cap)) == 4  # the system prompt changed
    provider.close()


def test_live_context_past_the_window_reserve_restarts(tmp_path: Path) -> None:
    binary, cap = _install(
        tmp_path,
        {"turns": [[_round(text="a", usage=_usage(5000, 10))], [_round(text="b")]]},
    )
    provider = _provider(binary, context_tokens=20_000)  # reserve 16_384 -> 3_616 left
    first = provider.call(system="s", messages=USER0, tools=TOOLS)
    history = [
        *USER0,
        {"role": "assistant", "content": first.raw["content"]},
        {"role": "user", "content": [{"type": "text", "text": "n"}]},
    ]
    provider.call(system="s", messages=history, tools=TOOLS)
    assert len(_spawns(cap)) == 2
    provider.close()


def test_budget_sums_rounds_and_fails_closed_without_a_reading_or_usage(tmp_path: Path) -> None:
    binary, _ = _install(
        tmp_path,
        {
            "turns": [
                [
                    _round(
                        tool_uses=[{"id": "toolu_1", "name": "read_file", "input": {}}],
                        usage=_usage(1000, 10),
                    ),
                    _round(text="done", usage=_usage(2000, 20)),
                ]
            ]
        },
    )
    provider = _provider(binary)
    first = provider.call(system="s", messages=USER0, tools=TOOLS)
    provider.call(
        system="s",
        messages=[
            *USER0,
            {"role": "assistant", "content": first.raw["content"]},
            {
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": "toolu_1", "content": "R"}],
            },
        ],
        tools=TOOLS,
    )
    budget = provider.budget
    assert budget is not None
    snap = budget.snapshot()
    assert (snap.input_total, snap.output_total) == (3000, 30)
    provider.close()

    _rescenario(tmp_path, {"rate_limit": {"when": "never"}, "turns": [[_round(text="x")]]})
    with pytest.raises(ProviderError, match="no plan window") as exc:
        _provider(binary).call(system="s", messages=USER0, tools=None)
    assert exc.value.fatal  # a retry replays the round into the same absence

    _rescenario(tmp_path, {"turns": [[_round(text="x", usage=_usage(0, 5))]]})
    with pytest.raises(ProviderError, match="no usage input tokens"):
        _provider(binary).call(system="s", messages=USER0, tools=None)


def test_abort_and_interrupt_kill_the_child_and_the_next_call_respawns(tmp_path: Path) -> None:
    binary, cap = _install(tmp_path, {"hang_s": 30, "turns": [[_round(text="x")]]})
    provider = _provider(binary)
    started = time.monotonic()
    with pytest.raises(ProviderAborted):
        provider.call(system="s", messages=USER0, tools=TOOLS, should_abort=lambda: True)
    assert time.monotonic() - started < 5
    assert not _alive(_spawns(cap)[0]["pid"])
    with pytest.raises(ProviderInterrupted):
        provider.call(system="s", messages=USER0, tools=TOOLS, should_interrupt=lambda: True)
    assert not _alive(_spawns(cap)[1]["pid"])
    _rescenario(tmp_path, {"turns": [[_round(text="x")]]})
    assert provider.call(system="s", messages=USER0, tools=TOOLS).text == "x"
    assert len(_spawns(cap)) == 3
    provider.close()


def test_idle_child_is_killed_after_the_stream_timeout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(claude_code, "STREAM_FIRST_DATA_TIMEOUT_S", 0.6)
    binary, cap = _install(tmp_path, {"hang_s": 10, "turns": [[_round(text="x")]]})
    provider = _provider(binary)
    with pytest.raises(ProviderError, match="produced no output") as exc:
        provider.call(system="s", messages=USER0, tools=TOOLS)
    assert not exc.value.fatal
    assert not _alive(_spawns(cap)[0]["pid"])


def test_failures_map_to_provider_errors(tmp_path: Path) -> None:
    with pytest.raises(ProviderError, match="not found on PATH") as exc:
        _provider(str(tmp_path / "missing")).call(system="s", messages=USER0, tools=None)
    assert exc.value.fatal

    binary, _ = _install(tmp_path, {"die_in_round": 1, "turns": [[_round(text="x")]]})
    with pytest.raises(ProviderError, match="exited 3: boom") as exc:
        _provider(binary).call(system="s", messages=USER0, tools=None)
    assert not exc.value.fatal


def test_a_dying_childs_stderr_reaches_the_error_when_the_drain_lags(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The stderr drain is a thread. On a loaded box it can still be scheduled
    out when stdout closes and the exit status lands, and the error then read
    `no stderr` for a message the child had already written. The exit path
    waits for the drain to reach EOF first."""
    import agent6.providers.claude_code as module

    binary, _ = _install(
        tmp_path, {"die_in_round": 1, "die_message": "boom late", "turns": [[_round(text="x")]]}
    )
    real = module._drain_stderr  # pyright: ignore[reportPrivateUsage]

    def lagging(pipe: IO[bytes], keep: list[bytes]) -> None:
        # Scheduled out AFTER the child has written: wait for its stderr to
        # become readable, then sleep well inside the exit path's join cap.
        select.select([pipe], [], [], 5.0)
        time.sleep(0.2)
        real(pipe, keep)

    monkeypatch.setattr(module, "_drain_stderr", lagging)
    with pytest.raises(ProviderError, match="exited 3: boom late"):
        _provider(binary).call(system="s", messages=USER0, tools=None)

    _rescenario(
        tmp_path,
        {
            "result": {"subtype": "error_during_execution", "is_error": True},
            "turns": [[_round(text="x")]],
        },
    )
    with pytest.raises(ProviderError, match="error_during_execution") as exc:
        _provider(binary).call(system="s", messages=USER0, tools=None)
    assert not exc.value.fatal

    _rescenario(tmp_path, {"synthetic_error": {"text": "Not logged in · Please run /login"}})
    with pytest.raises(ProviderError, match="claude auth login") as exc:
        _provider(binary).call(system="s", messages=USER0, tools=None)
    assert exc.value.fatal and exc.value.status_code is None


def test_a_stray_can_use_tool_is_allowed(tmp_path: Path) -> None:
    binary, cap = _install(
        tmp_path,
        {
            "can_use_tool": True,
            "turns": [
                [
                    _round(tool_uses=[{"id": "toolu_1", "name": "read_file", "input": {}}]),
                    _round(text="done"),
                ]
            ],
        },
    )
    provider = _provider(binary)
    first = provider.call(system="s", messages=USER0, tools=TOOLS)
    second = provider.call(
        system="s",
        messages=[
            *USER0,
            {"role": "assistant", "content": first.raw["content"]},
            {
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": "toolu_1", "content": "R"}],
            },
        ],
        tools=TOOLS,
    )
    assert second.text == "done"
    allows = [
        line
        for line in _stdin(cap)
        if ((line.get("response") or {}).get("response") or {}).get("behavior") == "allow"
    ]
    assert len(allows) == 1
    provider.close()


def test_side_call_is_one_process_and_call_for_text_returns_the_text(tmp_path: Path) -> None:
    binary, cap = _install(tmp_path, {"turns": [[_round(text="LGTM")]]})
    provider = _provider(binary)
    resp = provider.call(
        system="review", messages=[{"role": "user", "content": "DIFF"}], tools=None
    )
    assert resp.text == "LGTM"
    assert not _alive(_spawns(cap)[0]["pid"])
    assert call_for_text(provider, system="s", user="u", max_tokens=10) == "LGTM"
    assert len(_spawns(cap)) == 2 and not _alive(_spawns(cap)[1]["pid"])
    assert _user_texts(cap) == ["DIFF", "u"]


def test_transcript_round_is_anthropic_shaped(tmp_path: Path) -> None:
    binary, _ = _install(tmp_path, {"turns": [[_round(text="hello")]]})
    sink = TranscriptSink(tmp_path / "transcripts")
    _provider(binary, transcript_sink=sink).call(system="sys", messages=USER0, tools=None)
    files = sorted((tmp_path / "transcripts").glob("*.json"))
    assert len(files) == 1
    payload = json.loads(files[0].read_text(encoding="utf-8"))
    assert payload["request"]["url"] == "claude-code://fake-session-1"
    assert payload["response"]["body"]["role"] == "assistant"
    turns = fold_conversation([payload])
    assert any("hello" in getattr(t, "text", "") for t in turns)


def test_login_status_reads_logged_in_only(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "leak")
    binary, cap = _install(tmp_path, {"auth": {"loggedIn": False, "rc": 1}})
    remedy = login_status(binary)
    assert remedy is not None and "claude auth login" in remedy
    assert "leak@example.test" not in remedy and "org_leak" not in remedy
    assert "ANTHROPIC_API_KEY" not in _captured(cap)[0]["env"]
    _rescenario(tmp_path, {"auth": {"loggedIn": True, "rc": 0}})
    assert login_status(binary) is None
    missing = login_status(str(tmp_path / "missing"))
    assert missing is not None and "not found on PATH" in missing


def test_render_history_is_verbatim_for_one_message_and_labelled_after() -> None:
    assert render_history(USER0) == "TASK:\nwrite hello"
    history = [
        *USER0,
        {
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "hidden", "signature": "x"},
                {"type": "text", "text": "ok"},
                {"type": "tool_use", "id": "t1", "name": "read_file", "input": {"path": "a"}},
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "t1", "content": "R"},
                {"type": "text", "text": "nudge"},
            ],
        },
    ]
    text = render_history(history)
    assert text.startswith("TASK:\nwrite hello\n\n[harness] This session continues")
    assert "### assistant\nok" in text
    assert '[tool_use read_file] {"path": "a"}' in text
    assert "### tool_result read_file\nR" in text
    assert "### user\nnudge" in text
    assert "hidden" not in text


def test_plan_usage_from_rate_limit_maps_the_windows() -> None:
    info = {
        "status": "rejected",
        "overageStatus": "allowed",
        "unifiedWindows": {
            "five_hour": {"utilization": 0.1, "resetsAt": 100},
            "seven_day_opus": {"utilization": 1.2, "resetsAt": 200},
        },
    }
    plan = plan_usage_from_rate_limit(info)
    assert plan is not None
    assert [(w.name, w.used_percent, w.window_minutes, w.resets_at) for w in plan.windows] == [
        ("five_hour", 10.0, 300, 100.0),
        ("seven_day_opus", 120.0, 10080, 200.0),
    ]
    assert plan.limit_reached and plan.has_credits and plan.window_exhausted
    assert plan_usage_from_rate_limit({"status": "allowed"}) is None


def test_factory_builds_the_provider_and_refuses_effort_off(tmp_path: Path) -> None:
    def cfg(effort: str | None) -> Config:
        role: dict[str, Any] = {"provider": "claude", "model": "claude-haiku-4-5"}
        if effort:
            role["effort"] = effort
        return Config.model_validate(
            {
                "providers": {"claude": {"api_format": "claude_code", "binary": "/opt/claude"}},
                "models": {"worker": role},
            }
        )

    sink = TranscriptSink(tmp_path / "t")
    provider = build_role_provider(cfg("low"), "worker", transcript_sink=sink, budget=_budget())
    assert isinstance(provider, ClaudeCodeProvider)
    assert (provider.binary, provider.effort, provider.model) == (
        "/opt/claude",
        "low",
        "claude-haiku-4-5",
    )
    assert provider.context_tokens == 200_000
    with pytest.raises(ProviderError, match="effort = off") as exc:
        build_role_provider(cfg("off"), "worker", transcript_sink=sink, budget=_budget())
    assert exc.value.fatal


def test_instrumented_provider_close_forwards_to_the_inner_close() -> None:
    inner = MagicMock()
    wrapper = InstrumentedProvider(
        inner=inner, role="worker", model="m", provider_name="claude", events=None, budget=_budget()
    )
    wrapper.close()
    inner.close.assert_called_once_with()
    http_like = MagicMock(spec=["call"])
    bare = InstrumentedProvider(
        inner=http_like, role="worker", model="m", provider_name="p", events=None, budget=_budget()
    )
    bare.close()  # an HTTP provider has no close
    assert not hasattr(http_like, "close")


def test_close_terminates_a_child_blocked_on_a_tool_call(tmp_path: Path) -> None:
    """A leg ends on an unanswered tools/call, where the CLI ignores stdin EOF;
    close() sends SIGTERM, which the CLI handles (exit 143, socket removed),
    before any SIGKILL."""
    marker = tmp_path / "up"
    binary, cap = _install(
        tmp_path,
        {
            "term_marker": str(marker),
            "turns": [[_round(tool_uses=[{"id": "toolu_1", "name": "read_file", "input": {}}])]],
        },
    )
    provider = _provider(binary)
    provider.call(system="s", messages=USER0, tools=TOOLS)
    assert marker.exists()
    provider.close()
    assert not marker.exists()
    assert not _alive(_spawns(cap)[0]["pid"])


def test_an_unclosed_session_is_reaped_at_interpreter_exit(tmp_path: Path) -> None:
    """A caller that never closes (a machine agent's worker) still leaves no
    child and no private directory behind: the session's finalizer runs at
    exit."""
    binary, cap = _install(
        tmp_path,
        {"turns": [[_round(tool_uses=[{"id": "toolu_1", "name": "read_file", "input": {}}])]]},
    )
    script = (
        "from agent6.providers.claude_code import ClaudeCodeProvider\n"
        "from agent6.providers.types import ToolDefinition\n"
        f"provider = ClaudeCodeProvider(model='m', binary={binary!r})\n"
        "tools = [ToolDefinition(name='read_file', description='d', input_schema={})]\n"
        "provider.call(system='s', messages=[{'role': 'user', 'content': 'go'}], tools=tools)\n"
    )
    subprocess.run([sys.executable, "-c", script], check=True, timeout=60)
    spawn = _spawns(cap)[0]
    deadline = time.monotonic() + 5
    while _alive(spawn["pid"]) and time.monotonic() < deadline:
        time.sleep(0.05)
    assert not _alive(spawn["pid"])
    assert not Path(spawn["cwd"]).exists()


def test_a_side_call_leaves_the_live_worker_session_untouched(tmp_path: Path) -> None:
    """A no-tools call on the worker's provider (a model-drafted commit
    message) runs in its own throwaway process; the worker's session and its
    pending tools/call survive, so the next worker call continues instead of
    replaying."""
    binary, cap = _install(
        tmp_path,
        {
            "turns": [
                [
                    _round(tool_uses=[{"id": "toolu_1", "name": "read_file", "input": {}}]),
                    _round(text="after"),
                ]
            ]
        },
    )
    provider = _provider(binary)
    first = provider.call(system="s", messages=USER0, tools=TOOLS)
    worker_pid = _spawns(cap)[0]["pid"]
    call_for_text(provider, system="commit", user="files", max_tokens=10)
    assert _alive(worker_pid)
    assert len(_spawns(cap)) == 2 and not _alive(_spawns(cap)[1]["pid"])
    second = provider.call(
        system="s",
        messages=[
            *USER0,
            {"role": "assistant", "content": first.raw["content"]},
            {
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": "toolu_1", "content": "R"}],
            },
        ],
        tools=TOOLS,
    )
    assert second.text == "after"
    assert len(_spawns(cap)) == 2  # no replay
    provider.close()


def test_a_rewritten_prefix_restarts_even_at_the_same_length(tmp_path: Path) -> None:
    """A tier-2 restart replaces the consumed prefix with the first turn plus
    a summary; at the same length only the content tells it from a
    continuation, and the summary must reach a fresh process."""
    binary, cap = _install(
        tmp_path,
        {
            "turns": [
                [
                    _round(tool_uses=[{"id": "toolu_1", "name": "read_file", "input": {}}]),
                    _round(text="after"),
                ]
            ]
        },
    )
    provider = _provider(binary)
    history = [*USER0, {"role": "user", "content": [{"type": "text", "text": "NOTICE"}]}]
    first = provider.call(system="s", messages=history, tools=TOOLS)
    compacted = [
        USER0[0],
        {"role": "user", "content": [{"type": "text", "text": "SUMMARY"}]},
        {"role": "assistant", "content": first.raw["content"]},
        {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "toolu_1", "content": "R"}],
        },
    ]
    provider.call(system="s", messages=compacted, tools=TOOLS)
    assert len(_spawns(cap)) == 2
    assert _user_texts(cap)[-1] == render_history(compacted)
    provider.close()


def test_tier1_rewrites_and_thinking_strips_keep_the_process(tmp_path: Path) -> None:
    """Tier-1 elision rewrites a consumed tool_result in place and thinking
    strips drop a consumed turn's thinking; neither changes what the process
    was sent, so the session continues."""
    binary, cap = _install(
        tmp_path,
        {
            "turns": [
                [
                    _round(
                        thinking="think",
                        tool_uses=[{"id": "toolu_1", "name": "read_file", "input": {}}],
                    ),
                    _round(tool_uses=[{"id": "toolu_2", "name": "read_file", "input": {}}]),
                    _round(text="done"),
                ]
            ]
        },
    )
    provider = _provider(binary)
    first = provider.call(system="s", messages=USER0, tools=TOOLS)
    history: list[dict[str, Any]] = [
        *USER0,
        {"role": "assistant", "content": first.raw["content"]},
        {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "toolu_1", "content": "LONG"}],
        },
    ]
    second = provider.call(system="s", messages=history, tools=TOOLS)
    history[1] = {
        "role": "assistant",
        "content": [b for b in first.raw["content"] if b["type"] != "thinking"],
    }
    history[2] = {
        "role": "user",
        "content": [{"type": "tool_result", "tool_use_id": "toolu_1", "content": "<elided>"}],
    }
    history += [
        {"role": "assistant", "content": second.raw["content"]},
        {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "toolu_2", "content": "R2"}],
        },
    ]
    assert provider.call(system="s", messages=history, tools=TOOLS).text == "done"
    assert len(_spawns(cap)) == 1
    provider.close()


def test_a_later_rounds_plan_reading_is_recorded_for_that_round(tmp_path: Path) -> None:
    """A round that moved a window is followed by its reading right after
    message_stop; that round records it, not the previous reading."""
    binary, _ = _install(
        tmp_path,
        {
            "rate_limit": {"five_hour": 0.32},
            "turns": [
                [
                    _round(tool_uses=[{"id": "toolu_1", "name": "read_file", "input": {}}]),
                    _round(
                        tool_uses=[{"id": "toolu_2", "name": "read_file", "input": {}}],
                        rate_limit={"five_hour": 0.4},
                    ),
                ]
            ],
        },
    )
    provider = _provider(binary)
    budget = provider.budget
    assert budget is not None

    def five_hour() -> float:
        plan = budget.snapshot().plan_latest
        assert plan is not None
        return next(w.used_percent for w in plan.windows if w.name == "five_hour")

    first = provider.call(system="s", messages=USER0, tools=TOOLS)
    assert five_hour() == 32.0
    provider.call(
        system="s",
        messages=[
            *USER0,
            {"role": "assistant", "content": first.raw["content"]},
            {
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": "toolu_1", "content": "R"}],
            },
        ],
        tools=TOOLS,
    )
    assert five_hour() == 40.0
    provider.close()


def test_streamed_deltas_never_carry_the_account_email(tmp_path: Path) -> None:
    """The scrub holds back the tail that could be the start of an email
    split across deltas and flushes it at content_block_stop, so the stream
    reads the same as the settled block."""
    email = "leak@example.test"
    binary, _ = _install(
        tmp_path,
        {
            "account_email": email,
            "delta_chars": 4,
            "turns": [[_round(thinking=f"user is {email}!", text=f"Hi {email}, done")]],
        },
    )
    text: list[str] = []
    thinking: list[str] = []
    resp = _provider(binary).call(
        system="s",
        messages=USER0,
        tools=None,
        text_delta_callback=text.append,
        thinking_delta_callback=thinking.append,
    )
    assert "".join(text) == resp.text == "Hi <operator-email>, done"
    assert "".join(thinking) == "user is <operator-email>!"
    assert all(email not in piece for piece in text + thinking)


def test_result_and_stderr_error_text_is_scrubbed(tmp_path: Path) -> None:
    """A failed turn's result text and the child's stderr tail reach the
    ProviderError (and the journal) with the account email replaced."""
    email = "leak@example.test"
    binary, _ = _install(
        tmp_path,
        {
            "account_email": email,
            "result": {"subtype": "error_during_execution", "is_error": True, "text": f"x {email}"},
            "turns": [[_round(text="x")]],
        },
    )
    with pytest.raises(ProviderError, match="error_during_execution") as exc:
        _provider(binary).call(system="s", messages=USER0, tools=None)
    assert email not in str(exc.value) and "<operator-email>" in str(exc.value)
    _rescenario(
        tmp_path,
        {
            "account_email": email,
            "die_in_round": 1,
            "die_message": f"boom {email}",
            "turns": [[_round(text="x")]],
        },
    )
    with pytest.raises(ProviderError, match="exited 3") as exc:
        _provider(binary).call(system="s", messages=USER0, tools=None)
    assert email not in str(exc.value) and "<operator-email>" in str(exc.value)


def test_a_result_with_an_api_error_status_carries_it(tmp_path: Path) -> None:
    """The CLI's result line names the API status of a failed turn
    (`api_error_status`, 404 for an unknown model); the ProviderError carries
    it, so the loop's retry ladder skips the permanent ones."""
    binary, _ = _install(
        tmp_path,
        {
            "result": {
                "subtype": "success",
                "is_error": True,
                "text": "There's an issue with the selected model (claude-nope).",
                "api_error_status": 404,
            },
            "turns": [[_round(text="x")]],
        },
    )
    with pytest.raises(ProviderError, match="HTTP 404") as exc:
        _provider(binary).call(system="s", messages=USER0, tools=None)
    assert exc.value.status_code == 404 and not exc.value.fatal


def test_an_mcp_ping_is_answered_with_an_empty_result(tmp_path: Path) -> None:
    binary, cap = _install(tmp_path, {"ping": True, "turns": [[_round(text="x")]]})
    _provider(binary).call(system="s", messages=USER0, tools=TOOLS)
    answers = [
        rpc
        for line in _stdin(cap)
        if (rpc := ((line.get("response") or {}).get("response") or {}).get("mcp_response"))
        and rpc.get("id") == 2
    ]
    assert answers == [{"jsonrpc": "2.0", "id": 2, "result": {}}]


def test_the_child_stdout_is_read_buffered(tmp_path: Path) -> None:
    """An unbuffered pipe makes readline one read syscall per byte (741 ms
    for a 1 MiB line, measured); the reader gets a BufferedReader."""
    binary, _ = _install(tmp_path, {"turns": [[_round(text="x")]]})
    provider = _provider(binary)
    provider.call(system="s", messages=USER0, tools=TOOLS)
    session = provider._cell[0]  # pyright: ignore[reportPrivateUsage]
    assert session is not None
    assert isinstance(session.proc.stdout, io.BufferedReader)
    provider.close()


def test_an_oversize_tool_result_is_refused_before_claude_code_persists_it(
    tmp_path: Path,
) -> None:
    """A result over the 50,000-byte threshold would be written under
    ~/.claude/projects and reach the model as a preview: the provider refuses
    it (fatal) instead of lying about what the model saw."""
    binary, _cap = _install(
        tmp_path,
        {"turns": [[_round(tool_uses=[{"id": "toolu_1", "name": "read_file", "input": {}}])]]},
    )
    provider = _provider(binary)
    first = provider.call(system="s", messages=USER0, tools=TOOLS)
    history = [
        *USER0,
        {"role": "assistant", "content": first.raw["content"]},
        {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "toolu_1", "content": "x" * 50_001}],
        },
    ]
    with pytest.raises(ProviderError, match="50000-byte threshold") as exc:
        provider.call(system="s", messages=history, tools=TOOLS)
    assert exc.value.fatal
    provider.close()


def test_the_loop_caps_results_tighter_for_a_claude_code_worker() -> None:
    from agent6.app._session import tool_result_cap_chars
    from agent6.workflows._compaction import TOOL_RESULT_CHAR_CAP

    cc = Config.model_validate(
        {
            "providers": {"claude": {"api_format": "claude_code"}},
            "models": {"worker": {"provider": "claude", "model": "claude-haiku-4-5"}},
        }
    )
    assert tool_result_cap_chars(cc) == CLAUDE_CODE_RESULT_CAP_CHARS < TOOL_RESULT_CHAR_CAP
    assert tool_result_cap_chars(Config()) == TOOL_RESULT_CHAR_CAP
