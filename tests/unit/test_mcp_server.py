# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""MCPServer unit tests — handler routing + JSON-RPC framing."""

from __future__ import annotations

import io
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from agent6.config import Config, load_config
from agent6.config.layer import resolved_state_dir
from agent6.graph.models import TaskNode
from agent6.graph.storage import write_node
from agent6.sessions.ipc import register_frontend
from agent6.sessions.layout import SessionLayout
from agent6.sessions.manifest import MANIFEST_VERSION
from agent6.tools.dispatch import ToolError
from agent6.tools.errors import OperatorCommandUnexecutable
from agent6.tools.results import ExecResult, PatchResult, ToolResult
from agent6.ui.mcp_server import MCPServer

_VALID_TOML = """
[agent6]
config_version = 1
[providers.anthropic]
api_format = "anthropic"
api_key_env = "ANTHROPIC_API_KEY"
prompt_caching = true
[models.worker]
provider = "anthropic"
model = "x"
[models.reviewer]
provider = "anthropic"
model = "x"
[sandbox]
isolation = "auto"
run_commands = "no"
protect_git = true
[git]
dirty_tree = "ask"
branch_per_run = true
[workflow]
verify_command = ["true"]
[budget]
max_tokens_fallback = 2000000
"""


def _config(tmp_path: Path, *, run_commands: str = "no") -> Config:
    toml = _VALID_TOML.replace('run_commands = "no"', f'run_commands = "{run_commands}"')
    p = tmp_path / "agent6.toml"
    p.write_text(toml, encoding="utf-8")
    return load_config(p)


def _server(tmp_path: Path, **kwargs: Any) -> MCPServer:
    cfg = _config(tmp_path, **kwargs)
    return MCPServer(
        root=tmp_path,
        config=cfg,
        stdin=io.BytesIO(),
        stdout=io.BytesIO(),
    )


def _roundtrip(server: MCPServer, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Feed messages into the server's stdin, drive serve() to EOF,
    and parse responses from stdout."""
    payload = b"".join(json.dumps(m).encode("utf-8") + b"\n" for m in messages)
    server._stdin = io.BytesIO(payload)  # type: ignore[attr-defined]  # test-only stdin swap
    server._stdout = io.BytesIO()  # type: ignore[attr-defined]
    server.serve()
    server._stdout.seek(0)  # type: ignore[attr-defined]
    out: list[dict[str, Any]] = []
    for line in server._stdout.readlines():  # type: ignore[attr-defined]
        out.append(json.loads(line))
    return out


# ---------------------------------------------------------------------------
# JSON-RPC framing
# ---------------------------------------------------------------------------


def test_initialize_returns_server_info(tmp_path: Path) -> None:
    server = _server(tmp_path)
    resps = _roundtrip(
        server,
        [{"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}}],
    )
    assert len(resps) == 1
    assert resps[0]["id"] == 1
    info = resps[0]["result"]
    assert info["serverInfo"]["name"] == "agent6"
    assert info["protocolVersion"] == "2024-11-05"
    assert "tools" in info["capabilities"]


def test_tools_list_advertises_five_tools(tmp_path: Path) -> None:
    server = _server(tmp_path, run_commands="yes")
    resps = _roundtrip(
        server,
        [{"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}}],
    )
    tools = resps[0]["result"]["tools"]
    names = {t["name"] for t in tools}
    assert names == {
        "run_verify",
        "run_in_sandbox",
        "apply_patch_in_sandbox",
        "query_dag",
        "list_sessions",
    }
    # Every tool advertises a JSON-schema object.
    for t in tools:
        assert t["inputSchema"]["type"] == "object"


def test_withdrawn_command_tools_are_absent_and_named(tmp_path: Path) -> None:
    """Under run_commands = "no" (or the non-interactive "ask" clamp) the
    command tools are GONE from tools/list -- offered-and-failing lied about
    the surface -- and a client calling one by name is told the real reason,
    not "unknown tool"."""
    server = _server(tmp_path)  # the fixture's default is "no"
    resps = _roundtrip(
        server,
        [
            {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}},
            {
                "jsonrpc": "2.0",
                "id": 3,
                "method": "tools/call",
                "params": {"name": "run_verify", "arguments": {}},
            },
        ],
    )
    names = {t["name"] for t in resps[0]["result"]["tools"]}
    assert names == {"query_dag", "list_sessions"}
    err = resps[1]["error"]["message"]
    assert "withdrawn" in err and "run_commands" in err and "'no'" in err


def test_the_gate_tools_are_withdrawn_when_the_workspace_has_no_gate(tmp_path: Path) -> None:
    """With no verify command there is nothing to run: `run_verify` reached the
    jail with an empty argv and answered "tuple index out of range", and
    `apply_patch_in_sandbox` applied the patch and THEN failed the same way,
    leaving the workspace changed under a call reported as failed."""
    p = tmp_path / "agent6.toml"
    p.write_text(
        _VALID_TOML.replace('run_commands = "no"', 'run_commands = "yes"').replace(
            'verify_command = ["true"]', ""
        ),
        encoding="utf-8",
    )
    server = MCPServer(
        root=tmp_path, config=load_config(p), stdin=io.BytesIO(), stdout=io.BytesIO()
    )
    resps = _roundtrip(
        server,
        [
            {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}},
            {
                "jsonrpc": "2.0",
                "id": 3,
                "method": "tools/call",
                "params": {"name": "apply_patch_in_sandbox", "arguments": {}},
            },
        ],
    )

    names = {t["name"] for t in resps[0]["result"]["tools"]}
    assert names == {"run_in_sandbox", "query_dag", "list_sessions"}
    err = resps[1]["error"]["message"]
    assert "withdrawn" in err and "verify_command" in err


def test_unknown_method_returns_rpc_error(tmp_path: Path) -> None:
    server = _server(tmp_path)
    resps = _roundtrip(
        server,
        [{"jsonrpc": "2.0", "id": 3, "method": "nonsense", "params": {}}],
    )
    assert resps[0]["error"]["code"] == -32601
    assert "nonsense" in resps[0]["error"]["message"]


def test_unknown_tool_returns_rpc_error(tmp_path: Path) -> None:
    server = _server(tmp_path)
    resps = _roundtrip(
        server,
        [
            {
                "jsonrpc": "2.0",
                "id": 4,
                "method": "tools/call",
                "params": {"name": "no_such_tool", "arguments": {}},
            }
        ],
    )
    assert resps[0]["error"]["code"] == -32601


def test_notifications_produce_no_response(tmp_path: Path) -> None:
    server = _server(tmp_path)
    resps = _roundtrip(
        server,
        [{"jsonrpc": "2.0", "method": "notifications/initialized", "params": {}}],
    )
    assert resps == []


def test_malformed_json_answers_a_parse_error(tmp_path: Path) -> None:
    """JSON-RPC's answer to an unparseable request is -32700 with a null id
    (silence left the client hanging on a request it believes it sent); the
    next well-formed request still works."""
    server = _server(tmp_path)
    server._stdin = io.BytesIO(  # type: ignore[attr-defined]
        b"not json\n"
        + json.dumps({"jsonrpc": "2.0", "id": 7, "method": "tools/list"}).encode("utf-8")
        + b"\n"
    )
    server._stdout = io.BytesIO()  # type: ignore[attr-defined]
    server.serve()
    server._stdout.seek(0)  # type: ignore[attr-defined]
    replies = [json.loads(line) for line in server._stdout.readlines()]  # type: ignore[attr-defined]
    assert [r.get("id") for r in replies] == [None, 7]
    assert replies[0]["error"]["code"] == -32700


# ---------------------------------------------------------------------------
# Tool handlers that don't need the jail
# ---------------------------------------------------------------------------


def test_list_runs_empty(tmp_path: Path) -> None:
    server = _server(tmp_path)
    resps = _roundtrip(
        server,
        [
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {"name": "list_sessions", "arguments": {}},
            }
        ],
    )
    payload = resps[0]["result"]["structuredContent"]
    assert payload == {"sessions": []}


def test_list_runs_reads_manifests(tmp_path: Path) -> None:
    import os

    runs = resolved_state_dir(tmp_path) / "sessions" / "runs"
    (runs / "run-a").mkdir(parents=True)
    (runs / "run-b").mkdir(parents=True)
    (runs / "run-a" / "manifest.json").write_text(
        json.dumps({"user_task": "alpha"}), encoding="utf-8"
    )
    # run-b has no manifest -> entry without one. It DOES get a log: a dir with
    # neither is a husk, which every listing hides, so a manifest-less session
    # has to be modelled with one or it is not a session at all.
    (runs / "run-b" / "logs.jsonl").write_text(
        json.dumps({"type": "session.start", "mode": "run"}) + "\n", encoding="utf-8"
    )
    # Pin the dir mtimes so the
    # newest-first ordering is deterministic regardless of the filesystem's
    # mtime granularity: writing run-a's manifest bumps run-a's dir mtime, so
    # without this run-a can sort first on a fine-grained fs (and the tie-break
    # is iterdir order on a coarse one) -- which made this flaky in CI.
    os.utime(runs / "run-a", (1000, 1000))
    os.utime(runs / "run-b", (2000, 2000))  # run-b is newest
    server = _server(tmp_path)
    resps = _roundtrip(
        server,
        [
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {"name": "list_sessions", "arguments": {}},
            }
        ],
    )
    sessions_out = resps[0]["result"]["structuredContent"]["sessions"]
    assert [r["session_id"] for r in sessions_out] == ["run-b", "run-a"]
    # Shipped as the typed SessionManifest dump (full shape, defaults filled), not the
    # raw dict: the recorded user_task survives, the version stamp is present.
    assert sessions_out[1]["manifest"]["user_task"] == "alpha"
    assert sessions_out[1]["manifest"]["version"] == MANIFEST_VERSION
    assert "manifest" not in sessions_out[0]
    # The row the hubs share rides on top of the manifest: an editor reads the
    # same status words the CLI and web list, which the raw manifest lacks.
    assert sessions_out[1]["task"] == "alpha" and sessions_out[0]["mode"] == "run"
    assert {"status", "label", "level", "reason", "cost"} <= set(sessions_out[0])


def test_query_dag_missing_run_returns_tool_error(tmp_path: Path) -> None:
    server = _server(tmp_path)
    resps = _roundtrip(
        server,
        [
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {"name": "query_dag", "arguments": {}},
            }
        ],
    )
    assert resps[0]["result"]["isError"] is True
    assert "no sessions" in resps[0]["result"]["content"][0]["text"]


@pytest.mark.parametrize("bad", ["../../elsewhere/runs/x", "/etc", "a/b", ".."])
def test_query_dag_rejects_traversing_run_id(tmp_path: Path, bad: str) -> None:
    """A client-supplied session_id builds a path under the session buckets; a `..` or
    absolute id would read another repo's state (or anywhere). It must be
    rejected as a single-component id, like the web surface's guard."""
    server = _server(tmp_path)
    resps = _roundtrip(
        server,
        [
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {"name": "query_dag", "arguments": {"session_id": bad}},
            }
        ],
    )
    assert resps[0]["result"]["isError"] is True
    assert "invalid session_id" in resps[0]["result"]["content"][0]["text"]


def test_query_dag_reads_persisted_nodes(tmp_path: Path) -> None:
    layout = SessionLayout(state_dir=resolved_state_dir(tmp_path), session_id="r1")
    layout.ensure()
    node = TaskNode(
        id="01ARZ3NDEKTSV4RRFFQ69G5FAV",
        parent_id=None,
        title="root task",
        status="pending",
        rationale="seed",
        acceptance="done",
        relevant_paths=(),
        created_by="planner",
        created_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
    )
    write_node(layout, {node.id: node}, node)
    server = _server(tmp_path)
    resps = _roundtrip(
        server,
        [
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {
                    "name": "query_dag",
                    "arguments": {"session_id": "r1"},
                },
            }
        ],
    )
    payload = resps[0]["result"]["structuredContent"]
    assert payload["session_id"] == "r1"
    assert payload["nodes"]["01ARZ3NDEKTSV4RRFFQ69G5FAV"]["title"] == "root task"


# ---------------------------------------------------------------------------
# Tool handlers that delegate to the jailed dispatcher
# ---------------------------------------------------------------------------


def test_run_verify_delegates_to_dispatcher(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    server = _server(tmp_path, run_commands="yes")
    captured: list[tuple[str, dict[str, Any]]] = []

    def fake_dispatch(name: str, args: dict[str, Any]) -> ToolResult:
        captured.append((name, args))
        return ExecResult(returncode=0, stdout="", stderr="", duration_s=0.0, exec_failed=False)

    monkeypatch.setattr(server._dispatcher, "dispatch", fake_dispatch)  # type: ignore[attr-defined]
    resps = _roundtrip(
        server,
        [
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {"name": "run_verify", "arguments": {}},
            }
        ],
    )
    # The MCP tool is `run_verify`; internally it dispatches the dispatcher's
    # `run_verify_command` (the registered handler name).
    assert captured == [("run_verify_command", {})]
    assert resps[0]["result"]["structuredContent"]["returncode"] == 0


def test_run_in_sandbox_validates_argv(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    server = _server(tmp_path, run_commands="yes")

    def fake_dispatch(name: str, args: dict[str, Any]) -> ToolResult:
        return ExecResult(returncode=0, stdout="ok", stderr="", duration_s=0.0, exec_failed=False)

    monkeypatch.setattr(server._dispatcher, "dispatch", fake_dispatch)  # type: ignore[attr-defined]
    # Empty argv fails the published schema (minItems 1) at the call boundary:
    # an invalid-params JSON-RPC error, not a tool-level isError result.
    resps = _roundtrip(
        server,
        [
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {"name": "run_in_sandbox", "arguments": {"argv": []}},
            },
            {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/call",
                "params": {
                    "name": "run_in_sandbox",
                    "arguments": {"argv": ["echo", "hi"]},
                },
            },
        ],
    )
    assert resps[0]["error"]["code"] == -32602
    assert resps[1]["result"]["structuredContent"]["stdout"] == "ok"


def test_every_published_schema_type_is_one_the_checker_validates(tmp_path: Path) -> None:
    """`_schema_violation` validates the object/array/string subset and silently
    skips any other `type`, so a tool field of an uncovered type (integer, say)
    would advertise `additionalProperties: false` validation it never gets.
    Holds the published table to the checker's covered set; growing the table
    past it means growing the checker first."""
    checked_types = {"object", "array", "string"}

    def _types(schema: dict[str, Any]) -> set[str]:
        found = {schema["type"]} if isinstance(schema.get("type"), str) else set()
        for sub in schema.get("properties", {}).values():
            found |= _types(sub)
        if isinstance(schema.get("items"), dict):
            found |= _types(schema["items"])
        return found

    server = _server(tmp_path, run_commands="yes")
    for name, spec in server._tools.items():  # pyright: ignore[reportPrivateUsage]
        unchecked = _types(spec.input_schema) - checked_types
        assert not unchecked, f"{name} publishes type(s) {unchecked} the checker skips"


def test_tool_arguments_are_checked_against_the_published_schema(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """tools/list advertises each tool's inputSchema with additionalProperties:
    false and typed fields; a client that ignores it is still held to it at the
    call boundary. An unknown field, a wrong-typed element, a missing required
    field, or a wrong scalar type is a -32602 invalid-params error, not a value
    that rides through to the handler and the jail."""
    server = _server(tmp_path, run_commands="yes")

    def fake_dispatch(name: str, args: dict[str, Any]) -> ToolResult:
        raise AssertionError("handler must not run on a schema-invalid call")

    monkeypatch.setattr(server._dispatcher, "dispatch", fake_dispatch)  # type: ignore[attr-defined]
    resps = _roundtrip(
        server,
        [
            {  # unknown field, additionalProperties:false
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {"name": "run_verify", "arguments": {"surprise": 1}},
            },
            {  # argv items must be strings
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/call",
                "params": {"name": "run_in_sandbox", "arguments": {"argv": ["ok", 3]}},
            },
            {  # missing required "patch"
                "jsonrpc": "2.0",
                "id": 3,
                "method": "tools/call",
                "params": {"name": "apply_patch_in_sandbox", "arguments": {"path": "f"}},
            },
            {  # session_id must be a string
                "jsonrpc": "2.0",
                "id": 4,
                "method": "tools/call",
                "params": {"name": "query_dag", "arguments": {"session_id": 5}},
            },
        ],
    )
    for r in resps:
        assert "error" in r and r["error"]["code"] == -32602, r
    assert "unknown field" in resps[0]["error"]["message"]


def test_apply_patch_runs_verify_after(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    server = _server(tmp_path, run_commands="yes")
    calls: list[str] = []

    def fake_dispatch(name: str, args: dict[str, Any]) -> ToolResult:
        calls.append(name)
        if name == "apply_patch":
            return PatchResult(path="foo.py", bytes_written=5)
        if name == "run_verify_command":
            return ExecResult(returncode=0, stdout="", stderr="", duration_s=0.1, exec_failed=False)
        raise AssertionError(name)

    monkeypatch.setattr(server._dispatcher, "dispatch", fake_dispatch)  # type: ignore[attr-defined]
    resps = _roundtrip(
        server,
        [
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {
                    "name": "apply_patch_in_sandbox",
                    "arguments": {"path": "foo.py", "patch": "diff"},
                },
            }
        ],
    )
    assert calls == ["apply_patch", "run_verify_command"]
    payload = resps[0]["result"]["structuredContent"]
    assert payload["apply"]["bytes_written"] == 5
    assert payload["verify"]["returncode"] == 0


def test_apply_patch_surfaces_tool_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    server = _server(tmp_path, run_commands="yes")

    def fake_dispatch(name: str, args: dict[str, Any]) -> ToolResult:
        raise ToolError("patch did not apply")

    monkeypatch.setattr(server._dispatcher, "dispatch", fake_dispatch)  # type: ignore[attr-defined]
    resps = _roundtrip(
        server,
        [
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {
                    "name": "apply_patch_in_sandbox",
                    "arguments": {"path": "foo.py", "patch": "diff"},
                },
            }
        ],
    )
    assert resps[0]["result"]["isError"] is True
    assert "patch did not apply" in resps[0]["result"]["content"][0]["text"]


def test_unexecutable_operator_command_surfaces_as_iserror(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """OperatorCommandUnexecutable is deliberately not a ToolError (the loop
    aborts a run on it), but the MCP server's contract is isError results:
    letting it escape killed the whole `agent6 mcp serve` process, and every
    later client call died on a broken pipe."""
    server = _server(tmp_path, run_commands="yes")

    def fake_dispatch(name: str, args: dict[str, Any]) -> ToolResult:
        raise OperatorCommandUnexecutable("verify command not found on the jail PATH")

    monkeypatch.setattr(server._dispatcher, "dispatch", fake_dispatch)  # type: ignore[attr-defined]
    resps = _roundtrip(
        server,
        [
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {"name": "run_verify", "arguments": {}},
            },
            {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
        ],
    )
    assert resps[0]["result"]["isError"] is True
    assert "jail PATH" in resps[0]["result"]["content"][0]["text"]
    assert resps[1]["id"] == 2  # the server survived and answered the next call


# ---------------------------------------------------------------------------
# Approver
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("configured", "offered"),
    [("ask", False), ("yes", True), ("no", False)],
)
def test_no_one_to_ask_withdraws_rather_than_breaks(configured: str, offered: bool) -> None:
    """There is no human on a JSON-RPC transport, so `ask` cannot be answered.
    Offering a tool that refuses every call was worse than not offering it:
    run_verify and run_in_sandbox failed on every call under the DEFAULT config,
    and apply_patch_in_sandbox applied the patch and then errored on verify --
    leaving the workspace changed and the call failed."""
    from agent6.config import Config
    from agent6.ui.mcp_server import _no_one_to_ask  # pyright: ignore[reportPrivateUsage]

    cfg = Config.model_validate(
        {"sandbox": {"run_commands": configured}, "workflow": {"verify_command": ["true"]}}
    )
    assert (_no_one_to_ask(cfg).sandbox.run_commands == "yes") is offered


def test_most_recent_run_id_uses_log_activity_not_name_or_dir_touch(tmp_path: Path) -> None:
    # Run ids start with a random adjective-noun, so a name sort is not
    # chronological. Front-ends also write frontend.pid into run dirs, so
    # directory mtime is not chronological either. The newest log activity wins.
    import os

    from agent6.ui.mcp_server import _most_recent_session_id  # pyright: ignore[reportPrivateUsage]

    runs = tmp_path / "sessions" / "runs"
    runs.mkdir(parents=True)
    older = runs / "zzz-older-AAA111"  # alphabetically last
    newer = runs / "aaa-newer-BBB222"  # alphabetically first
    older.mkdir()
    newer.mkdir()
    (older / "logs.jsonl").write_text('{"type":"session.start"}\n', encoding="utf-8")
    (newer / "logs.jsonl").write_text('{"type":"session.start"}\n', encoding="utf-8")
    os.utime(older / "logs.jsonl", (1000, 1000))
    os.utime(newer / "logs.jsonl", (2000, 2000))
    register_frontend(older, 12345)
    assert _most_recent_session_id(tmp_path) == "aaa-newer-BBB222"


# ---------------------------------------------------------------------------
# Input bounding
# ---------------------------------------------------------------------------


def test_serve_bounds_every_stdin_read(tmp_path: Path) -> None:
    """serve() reads with an explicit size bound (mirroring the embedded
    client's _read_loop): the old unbounded readline() buffered an entire
    runaway line into memory BEFORE the 4 MiB check, so the cap could not
    prevent memory exhaustion."""
    from agent6.ui import mcp_server as mod

    sizes: list[int | None] = []

    class _RecordingStdin(io.BytesIO):
        def readline(self, size: int | None = -1) -> bytes:
            sizes.append(size)
            return super().readline(size)

    server = _server(tmp_path)
    msg = json.dumps({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}})
    server._stdin = _RecordingStdin(msg.encode() + b"\n")  # type: ignore[attr-defined]
    server._stdout = io.BytesIO()  # type: ignore[attr-defined]
    server.serve()
    assert sizes and all(s == mod._MAX_LINE_BYTES + 1 for s in sizes)  # pyright: ignore[reportPrivateUsage]


def test_serve_drains_oversized_line_and_recovers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An over-limit line is discarded in bounded chunks up to its newline; the
    next request on the stream is served normally."""
    from agent6.ui import mcp_server as mod

    monkeypatch.setattr(mod, "_MAX_LINE_BYTES", 128)
    server = _server(tmp_path)
    junk = b"x" * 500 + b"\n"
    msg = json.dumps({"jsonrpc": "2.0", "id": 7, "method": "initialize", "params": {}}).encode()
    server._stdin = io.BytesIO(junk + msg + b"\n")  # type: ignore[attr-defined]
    server._stdout = io.BytesIO()  # type: ignore[attr-defined]
    server.serve()
    server._stdout.seek(0)  # type: ignore[attr-defined]
    resps = [json.loads(line) for line in server._stdout.readlines()]  # type: ignore[attr-defined]
    assert [r["id"] for r in resps] == [7]


def test_list_sessions_skips_husks_like_every_other_listing(tmp_path: Path) -> None:
    """A husk is a dir a crash orphaned before any manifest or log. Every other
    listing hides it -- `viewmodel.listing` and `sessions list` both filter on
    `is_session_husk` -- because "(no logs)" forever is noise, not a session.

    MCP enumerated every directory, so an editor driving agent6 saw sessions the
    CLI and the web hub denied existed.
    """
    runs = resolved_state_dir(tmp_path) / "sessions" / "runs"
    (runs / "real-run").mkdir(parents=True)
    (runs / "real-run" / "manifest.json").write_text(
        json.dumps({"user_task": "alpha"}), encoding="utf-8"
    )
    (runs / "husk-run").mkdir(parents=True)  # no manifest, no log, no live worker

    server = _server(tmp_path)
    resps = _roundtrip(
        server,
        [
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {"name": "list_sessions", "arguments": {}},
            }
        ],
    )
    ids = {s["session_id"] for s in resps[0]["result"]["structuredContent"]["sessions"]}
    assert "real-run" in ids
    assert "husk-run" not in ids, f"MCP listed a husk the other surfaces hide: {ids}"
