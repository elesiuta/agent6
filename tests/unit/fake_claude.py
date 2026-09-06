# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""A stand-in `claude` binary speaking the stream-json protocol exactly as
Claude Code 2.1.251 does on the wire (line order included): the provider's
unit fixture.

Driven by `FAKE_CLAUDE_SCENARIO` (a JSON file); everything it sees goes to
`FAKE_CLAUDE_CAPTURE` as JSON lines: argv, env, cwd, the system-prompt file's
content and mode, and every stdin line (`{"stdin": ...}`).

Scenario keys (all optional):

- `auth`: `{"loggedIn": bool, "rc": int}` for `auth status --json`; the body
  also carries an email and org, as the real one does.
- `account_email`: what the initialize response's account block carries.
- `init`: `{"model", "apiKeySource", "tools" (null = the advertised set),
  "version"}` for the `system/init` line.
- `rate_limit`: `{"five_hour", "seven_day", "status", "overageStatus",
  "when": "after_stop" | "before_round" | "never"}`; emitted once, on the
  first round, like the CLI does when nothing moves afterwards. A round
  carrying its own `rate_limit` emits one after its message_stop (a window
  moved).
- `turns`: one list of rounds per user line received; a round is
  `{"thinking", "text", "tool_uses": [{"id", "name", "input"}],
  "stop_reason", "usage"}`. A tool round waits for every answer (the next
  `tools/call` only after the previous answer), then the next round plays.
  When a turn's rounds run out after a tool round, a `DONE` round follows.
- `delta_chars`: split every text and thinking delta into pieces of this
  many characters (default: one delta per block).
- `result`: `{"subtype", "is_error", "text", "api_error_status"}` for the
  result line.
- `can_use_tool`: ask `can_use_tool` before every `tools/call`.
- `ping`: send an MCP `ping` request after `tools/list`.
- `hang_s`: sleep this long before each round's `message_start`.
- `die_in_round`: exit 3 with `die_message` (default `boom`) on stderr after
  that round's `message_start` (rounds count across turns).
- `synthetic_error`: `{"text"}`: the signed-out shape, a synthetic
  assistant line then a `result` with `is_error` true and subtype success.
- `exit_code`: the exit status on stdin EOF between turns; EOF while a
  `tools/call` awaits its answer is ignored, as the CLI does.
- `term_marker`: a file created at start and removed by the SIGTERM handler
  (exit 143).
"""

from __future__ import annotations

import json
import os
import signal
import stat
import sys
import time
from pathlib import Path
from typing import Any

SESSION_ID = "fake-session-1"


def _capture(obj: dict[str, Any]) -> None:
    path = os.environ.get("FAKE_CLAUDE_CAPTURE")
    if path:
        with Path(path).open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(obj) + "\n")


def _emit(obj: dict[str, Any]) -> None:
    sys.stdout.buffer.write((json.dumps(obj) + "\n").encode())
    sys.stdout.buffer.flush()


class _Fake:
    def __init__(self, scenario: dict[str, Any]) -> None:
        self.scenario = scenario
        self.rpc_id = 0
        self.round_no = 0
        self.tools_listed: list[dict[str, Any]] = []
        self.rate_limit_sent = False

    def read_line(self, *, hold_on_eof: bool = False) -> dict[str, Any]:
        raw = sys.stdin.buffer.readline()
        if not raw:
            # The CLI stays up on EOF while a tools/call awaits its answer. Short
            # sleeps: a signal landing between the EOF and a long sleep would
            # run its Python handler only when the sleep ends.
            while hold_on_eof:
                time.sleep(0.05)
            sys.exit(int(self.scenario.get("exit_code", 0)))
        obj = json.loads(raw)
        _capture({"stdin": obj})
        return obj

    def request(self, message: dict[str, Any]) -> str:
        rid = f"req-{self.rpc_id}"
        self.rpc_id += 1
        _emit(
            {
                "type": "control_request",
                "request_id": rid,
                "request": {"subtype": "mcp_message", "server_name": "agent6", "message": message},
            }
        )
        return rid

    def wait_response(self, rid: str, *, hold_on_eof: bool = False) -> dict[str, Any]:
        while True:
            obj = self.read_line(hold_on_eof=hold_on_eof)
            if obj.get("type") == "control_response" and obj["response"].get("request_id") == rid:
                return obj

    def rate_limit(self, override: dict[str, Any] | None = None) -> None:
        rl = {**self.scenario.get("rate_limit", {}), **(override or {})}
        if override is None:
            if rl.get("when", "after_stop") == "never" or self.rate_limit_sent:
                return
            self.rate_limit_sent = True
        _emit(
            {
                "type": "rate_limit_event",
                "rate_limit_info": {
                    "status": rl.get("status", "allowed"),
                    "resetsAt": 1788088200,
                    "rateLimitType": "five_hour",
                    "overageStatus": rl.get("overageStatus", "rejected"),
                    "isUsingOverage": False,
                    "unifiedWindows": {
                        "five_hour": {
                            "utilization": rl.get("five_hour", 0.32),
                            "resetsAt": 1788088200,
                        },
                        "seven_day": {
                            "utilization": rl.get("seven_day", 0.51),
                            "resetsAt": 1788087600,
                        },
                    },
                },
                "uuid": "u",
                "session_id": SESSION_ID,
            }
        )

    def handshake(self) -> None:
        first = self.read_line()
        assert first["type"] == "control_request" and first["request"]["subtype"] == "initialize"
        # The MCP initialize arrives BEFORE the CLI answers the caller's initialize.
        rid = self.request(
            {
                "method": "initialize",
                "params": {
                    "protocolVersion": "2025-11-25",
                    "capabilities": {},
                    "clientInfo": {"name": "claude-code", "version": "2.1.251"},
                },
                "jsonrpc": "2.0",
                "id": 0,
            }
        )
        self.wait_response(rid)
        _emit(
            {
                "type": "control_response",
                "response": {
                    "subtype": "success",
                    "request_id": first["request_id"],
                    "response": {
                        "commands": [],
                        "agents": [],
                        "models": [],
                        "account": {
                            "email": self.scenario.get("account_email", "leak@example.test"),
                            "organization": "Leak Org",
                            "subscriptionType": "Claude Max",
                        },
                        "pid": os.getpid(),
                    },
                },
            }
        )

    def first_turn_setup(self) -> None:
        rid = self.request({"method": "notifications/initialized", "jsonrpc": "2.0"})
        self.wait_response(rid)
        rid = self.request({"method": "tools/list", "jsonrpc": "2.0", "id": 1})
        resp = self.wait_response(rid)
        self.tools_listed = resp["response"]["response"]["mcp_response"]["result"]["tools"]
        if self.scenario.get("ping"):
            rid = self.request({"method": "ping", "jsonrpc": "2.0", "id": 2})
            self.wait_response(rid)

    def system_init(self) -> None:
        init = self.scenario.get("init", {})
        tools = init.get("tools")
        if tools is None:
            tools = [f"mcp__agent6__{t['name']}" for t in self.tools_listed]
        _emit(
            {
                "type": "system",
                "subtype": "init",
                "cwd": str(Path.cwd()),
                "session_id": SESSION_ID,
                "tools": tools,
                "mcp_servers": [{"name": "agent6", "status": "connected"}],
                "model": init.get("model", "claude-haiku-4-5-20251001"),
                "permissionMode": "default",
                "slash_commands": [],
                "apiKeySource": init.get("apiKeySource", "none"),
                "claude_code_version": init.get("version", "2.1.251"),
                "skills": [],
                "plugins": [],
                "memory_paths": None,
                "uuid": "u",
            }
        )

    def assistant_line(self, message_id: str, block: dict[str, Any], usage: dict[str, Any]) -> None:
        _emit(
            {
                "type": "assistant",
                "message": {
                    "model": "claude-haiku-4-5-20251001",
                    "id": message_id,
                    "type": "message",
                    "role": "assistant",
                    "content": [block],
                    "stop_reason": None,
                    "usage": {**usage, "output_tokens": 3},
                },
                "parent_tool_use_id": None,
                "session_id": SESSION_ID,
                "uuid": "u",
            }
        )

    def stream(self, event: dict[str, Any]) -> None:
        _emit({"type": "stream_event", "event": event, "session_id": SESSION_ID, "uuid": "u"})

    def deltas(self, index: int, kind: str, key: str, text: str) -> None:
        size = int(self.scenario.get("delta_chars", 0)) or max(len(text), 1)
        for i in range(0, len(text), size):
            self.stream(
                {
                    "type": "content_block_delta",
                    "index": index,
                    "delta": {"type": kind, key: text[i : i + size]},
                }
            )

    def tools_call(self, tool_use_id: str, tool_use: dict[str, Any]) -> str:
        if self.scenario.get("can_use_tool"):
            rid = f"req-{self.rpc_id}"
            self.rpc_id += 1
            _emit(
                {
                    "type": "control_request",
                    "request_id": rid,
                    "request": {
                        "subtype": "can_use_tool",
                        "tool_name": f"mcp__agent6__{tool_use['name']}",
                        "input": tool_use.get("input", {}),
                        "tool_use_id": tool_use_id,
                    },
                }
            )
            self.wait_response(rid)
        self.rpc_id += 1
        return self.request(
            {
                "method": "tools/call",
                "params": {
                    "name": tool_use["name"],
                    "arguments": tool_use.get("input", {}),
                    "_meta": {"claudecode/toolUseId": tool_use_id, "progressToken": self.rpc_id},
                },
                "jsonrpc": "2.0",
                "id": self.rpc_id,
            }
        )

    def play_round(self, rnd: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
        """Stream one API round; return the tool calls it left pending."""
        self.round_no += 1
        hang = float(self.scenario.get("hang_s", 0))
        if hang:
            time.sleep(hang)
        rl = self.scenario.get("rate_limit", {})
        if rl.get("when") == "before_round":
            self.rate_limit()
        _emit(
            {
                "type": "system",
                "subtype": "status",
                "status": "requesting",
                "session_id": SESSION_ID,
            }
        )
        message_id = f"msg_{self.round_no}"
        usage = dict(
            rnd.get(
                "usage",
                {
                    "input_tokens": 1000,
                    "cache_creation_input_tokens": 0,
                    "cache_read_input_tokens": 0,
                    "output_tokens": 50,
                },
            )
        )
        start_usage = {k: v for k, v in usage.items() if k != "output_tokens"}
        self.stream(
            {
                "type": "message_start",
                "message": {
                    "model": "claude-haiku-4-5-20251001",
                    "id": message_id,
                    "type": "message",
                    "role": "assistant",
                    "content": [],
                    "stop_reason": None,
                    "usage": {**start_usage, "output_tokens": 3},
                },
            }
        )
        if int(self.scenario.get("die_in_round", 0)) == self.round_no:
            sys.stderr.write(str(self.scenario.get("die_message", "boom")) + "\n")
            sys.stderr.flush()
            sys.exit(3)
        index = 0
        if rnd.get("thinking"):
            self.stream(
                {
                    "type": "content_block_start",
                    "index": index,
                    "content_block": {"type": "thinking", "thinking": "", "signature": ""},
                }
            )
            _emit(
                {
                    "type": "system",
                    "subtype": "thinking_tokens",
                    "estimated_tokens": 5,
                    "session_id": SESSION_ID,
                }
            )
            self.deltas(index, "thinking_delta", "thinking", rnd["thinking"])
            self.stream(
                {
                    "type": "content_block_delta",
                    "index": index,
                    "delta": {"type": "signature_delta", "signature": "sig"},
                }
            )
            self.assistant_line(
                message_id,
                {"type": "thinking", "thinking": rnd["thinking"], "signature": "sig"},
                start_usage,
            )
            self.stream({"type": "content_block_stop", "index": index})
            index += 1
        pending: list[tuple[str, dict[str, Any]]] = []
        first_rid: str | None = None
        for n, tool_use in enumerate(rnd.get("tool_uses", [])):
            tool_use_id = tool_use.get("id") or f"toolu_{self.round_no}_{n}"
            block = {
                "type": "tool_use",
                "id": tool_use_id,
                "name": f"mcp__agent6__{tool_use['name']}",
                "input": tool_use.get("input", {}),
                "caller": {"type": "direct"},
            }
            self.stream(
                {
                    "type": "content_block_start",
                    "index": index,
                    "content_block": {**block, "input": {}},
                }
            )
            self.stream(
                {
                    "type": "content_block_delta",
                    "index": index,
                    "delta": {"type": "input_json_delta", "partial_json": ""},
                }
            )
            self.assistant_line(message_id, block, start_usage)
            self.stream({"type": "content_block_stop", "index": index})
            index += 1
            pending.append((tool_use_id, tool_use))
            if len(pending) == 1:
                first_rid = self.tools_call(tool_use_id, tool_use)  # before message_delta
        if rnd.get("text"):
            self.stream(
                {
                    "type": "content_block_start",
                    "index": index,
                    "content_block": {"type": "text", "text": ""},
                }
            )
            self.deltas(index, "text_delta", "text", rnd["text"])
            self.assistant_line(message_id, {"type": "text", "text": rnd["text"]}, start_usage)
            self.stream({"type": "content_block_stop", "index": index})
        stop_reason = rnd.get("stop_reason") or ("tool_use" if pending else "end_turn")
        self.stream(
            {
                "type": "message_delta",
                "delta": {"stop_reason": stop_reason, "stop_sequence": None},
                "usage": usage,
            }
        )
        self.stream({"type": "message_stop"})
        if rl.get("when", "after_stop") == "after_stop":
            self.rate_limit()
        if rnd.get("rate_limit"):
            self.rate_limit(rnd["rate_limit"])
        # Serve the calls one at a time: the next arrives only after the previous answer.
        for i, (tool_use_id, tool_use) in enumerate(pending):
            rid = first_rid if i == 0 else self.tools_call(tool_use_id, tool_use)
            assert rid is not None
            answer = self.wait_response(rid, hold_on_eof=True)
            content = answer["response"]["response"]["mcp_response"]["result"].get("content")
            _emit(
                {
                    "type": "user",
                    "message": {
                        "role": "user",
                        "content": [
                            {"tool_use_id": tool_use_id, "type": "tool_result", "content": content}
                        ],
                    },
                    "session_id": SESSION_ID,
                    "tool_use_result": content,
                }
            )
        return pending

    def result_line(self, text: str, stop_reason: str) -> None:
        res = self.scenario.get("result", {})
        line: dict[str, Any] = {
            "is_error": bool(res.get("is_error", False)),
            "duration_api_ms": 10,
            "num_turns": self.round_no,
            "stop_reason": stop_reason,
            "session_id": SESSION_ID,
            "total_cost_usd": 0.001 * self.round_no,
            "usage": {"input_tokens": 1000 * self.round_no, "output_tokens": 50 * self.round_no},
            "modelUsage": {"claude-haiku-4-5-20251001": {"costUSD": 0.001 * self.round_no}},
            "permission_denials": [],
            "subtype": res.get("subtype", "success"),
            "result": res.get("text") if res.get("text") is not None else text,
            "type": "result",  # late in the object, as the CLI writes it
        }
        if "api_error_status" in res:
            line["api_error_status"] = res["api_error_status"]
        _emit(line)

    def run(self) -> None:
        self.handshake()
        turns: list[list[dict[str, Any]]] = self.scenario.get("turns", [])
        turn_index = 0
        while True:
            msg = self.read_line()
            if msg.get("type") != "user":
                continue
            if turn_index == 0:
                self.first_turn_setup()
            self.system_init()
            synthetic = self.scenario.get("synthetic_error")
            if synthetic:
                self.assistant_line("msg_synth", {"type": "text", "text": synthetic["text"]}, {})
                _emit(
                    {
                        "is_error": True,
                        "subtype": "success",
                        "terminal_reason": "api_error",
                        "result": synthetic["text"],
                        "session_id": SESSION_ID,
                        "total_cost_usd": 0,
                        "modelUsage": {},
                        "type": "result",
                    }
                )
                sys.exit(1)
            rounds = list(turns[turn_index]) if turn_index < len(turns) else [{"text": "DONE"}]
            turn_index += 1
            last_text = ""
            last_stop = "end_turn"
            pending: list[tuple[str, dict[str, Any]]] = []
            for rnd in rounds:
                pending = self.play_round(rnd)
                last_text = rnd.get("text", "")
                last_stop = rnd.get("stop_reason") or ("tool_use" if pending else "end_turn")
            if pending:
                self.play_round({"text": "DONE"})
                last_text, last_stop = "DONE", "end_turn"
            self.result_line(last_text, last_stop)


def main() -> None:
    argv = sys.argv[1:]
    scenario: dict[str, Any] = {}
    path = os.environ.get("FAKE_CLAUDE_SCENARIO")
    if path and Path(path).exists():
        scenario = json.loads(Path(path).read_text(encoding="utf-8"))
    if argv[:2] == ["auth", "status"]:
        auth = scenario.get("auth", {"loggedIn": True, "rc": 0})
        body = {
            "analyticsDisabled": False,
            "apiProvider": "firstParty",
            "authMethod": "claude.ai" if auth.get("loggedIn", True) else "none",
            "email": "leak@example.test",
            "loggedIn": bool(auth.get("loggedIn", True)),
            "orgId": "org_leak_1",
            "orgName": "Leak Org",
            "subscriptionType": "max",
        }
        _capture({"auth_argv": argv, "env": dict(os.environ)})
        print(json.dumps(body))
        sys.exit(int(auth.get("rc", 0)))
    prompt_file = (
        argv[argv.index("--system-prompt-file") + 1] if "--system-prompt-file" in argv else ""
    )
    prompt_text = None
    prompt_mode = None
    if prompt_file and Path(prompt_file).exists():
        prompt_text = Path(prompt_file).read_text(encoding="utf-8")
        prompt_mode = stat.S_IMODE(Path(prompt_file).stat().st_mode)
    _capture(
        {
            "argv": argv,
            "env": dict(os.environ),
            "cwd": str(Path.cwd()),
            "cwd_entries": sorted(p.name for p in Path.cwd().iterdir()),
            "system_prompt": prompt_text,
            "system_prompt_mode": prompt_mode,
            "pid": os.getpid(),
        }
    )
    marker = scenario.get("term_marker")
    if marker:
        Path(marker).write_text("up", encoding="utf-8")

        def _term(_signum: int, _frame: object) -> None:
            Path(marker).unlink(missing_ok=True)
            os._exit(143)

        signal.signal(signal.SIGTERM, _term)
    _Fake(scenario).run()


if __name__ == "__main__":
    main()
