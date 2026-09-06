# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""agent6 as an MCP (Model Context Protocol) server.

Exposes the workspace's verify command, jail, patch tool, and DAG
storage to an external MCP client (Claude Desktop, an editor's agent
mode). Speaks line-delimited JSON-RPC 2.0 over stdio,
the same framing the embedded client in `tools/mcp_client.py`
consumes.

Trust posture: identical to the agent's own tools. Every command-
spawning handler routes through `agent6.sandbox.jail.run_in_jail`
via a `ToolDispatcher` constructed against the loaded config, so
the same Landlock + seccomp + namespace policy applies to anything
the MCP client asks us to run. `run_in_sandbox` honours the
existing `[sandbox].run_commands` gate; `"ask"` mode is treated
as a hard deny because the MCP boundary is non-interactive.

Tool surface:
    run_verify              - run the configured verify command in jail.
    run_in_sandbox          - run arbitrary argv in jail (gated).
    apply_patch_in_sandbox  - apply a unified diff + re-run verify.
    query_dag               - load <run-dir>/graph/*.md as nodes.
    list_sessions               - enumerate sessions (per-repo state dir) with manifest summary.
"""

from __future__ import annotations

import contextlib
import json
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Any

from agent6 import __version__
from agent6.config import Config
from agent6.config.layer import load_effective, resolved_state_dir
from agent6.graph.storage import load_graph
from agent6.sessions.id import SessionIdError, resolve_session
from agent6.sessions.layout import (
    SESSION_BUCKETS,
    is_safe_session_id,
)
from agent6.sessions.manifest import ManifestError, read_manifest
from agent6.tools.dispatch import ToolDispatcher, ToolError
from agent6.tools.errors import OperatorCommandUnexecutable
from agent6.viewmodel import session_dirs

_PROTOCOL_VERSION = "2024-11-05"
_SERVER_NAME = "agent6"
_MAX_LINE_BYTES = 1 << 22  # 4 MiB; mirrors the client-side cap.


# ---------------------------------------------------------------------------
# JSON-RPC error sentinel.
# ---------------------------------------------------------------------------


class _RpcError(Exception):
    """A JSON-RPC level failure (bad method, bad params). Distinct from
    `ToolError`, which is surfaced as a tool-level isError result."""

    def __init__(self, code: int, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


# ---------------------------------------------------------------------------
# Tool spec table.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _ToolSpec:
    name: str
    description: str
    input_schema: dict[str, Any]
    handler: Callable[[dict[str, Any]], dict[str, Any]]


# ---------------------------------------------------------------------------
# Helpers.
# ---------------------------------------------------------------------------


# The command-spawning tool names, withdrawn together when commands are off.
_COMMAND_TOOLS = frozenset({"run_verify", "run_in_sandbox", "apply_patch_in_sandbox"})

# The two that run the workspace's gate. With no verify command there is
# nothing for them to run, and `apply_patch_in_sandbox` must refuse BEFORE
# applying the patch, or the workspace changes under a call the client is
# told failed. The run loop hides
# `run_verify_command` for the same reason.
_GATE_TOOLS = frozenset({"run_verify", "apply_patch_in_sandbox"})


def _schema_violation(value: Any, schema: dict[str, Any], where: str) -> str | None:
    """First way *value* fails *schema*, or None.

    tools/list publishes each tool's inputSchema with `additionalProperties:
    false`; a client that ignores it must still be held to it here, or an
    unknown field rides through and a wrong-typed one reaches the jail. Covers
    the JSON-Schema subset the tool table uses: object (properties, required,
    additionalProperties), array (items, minItems), string (minLength).
    """
    typ = schema.get("type")
    if typ == "object":
        return _object_violation(value, schema, where)
    if typ == "array":
        return _array_violation(value, schema, where)
    if typ == "string":
        return _string_violation(value, schema, where)
    return None


def _object_violation(value: Any, schema: dict[str, Any], where: str) -> str | None:
    if not isinstance(value, dict):
        return f"{where} must be an object"
    props: dict[str, Any] = schema.get("properties", {})
    if schema.get("additionalProperties") is False:
        unknown = sorted(k for k in value if k not in props)
        if unknown:
            return f"{where} has unknown field(s): {', '.join(unknown)}"
    for req in schema.get("required", ()):
        if req not in value:
            return f"{where} is missing required field {req!r}"
    for key, sub in props.items():
        if key in value and (nested := _schema_violation(value[key], sub, f"{where}.{key}")):
            return nested
    return None


def _array_violation(value: Any, schema: dict[str, Any], where: str) -> str | None:
    if not isinstance(value, list):
        return f"{where} must be an array"
    if len(value) < schema.get("minItems", 0):
        return f"{where} must have at least {schema['minItems']} item(s)"
    item_schema = schema.get("items")
    if isinstance(item_schema, dict):
        for i, item in enumerate(value):
            if nested := _schema_violation(item, item_schema, f"{where}[{i}]"):
                return nested
    return None


def _string_violation(value: Any, schema: dict[str, Any], where: str) -> str | None:
    # bool is an int in Python but never a str, so this rejects true/false.
    if not isinstance(value, str):
        return f"{where} must be a string"
    if len(value) < schema.get("minLength", 0):
        return f"{where} must be at least {schema['minLength']} character(s)"
    return None


def _no_one_to_ask(config: Config) -> Config:
    """Withdraw the command tools when the config would prompt for them.

    The MCP transport has no human at the other end, so `"ask"` cannot be
    answered -- and offering a tool that will refuse every call is worse than
    not offering it: `run_verify` and `run_in_sandbox` failed on every call
    under the default config, and `apply_patch_in_sandbox` applied the patch
    and THEN errored on its verify step, leaving the workspace changed and the
    call failed. Same rule as a detached run with an away-mode of "deny": no
    one to ask means the tools are gone, not broken.
    """
    if config.sandbox.run_commands != "ask":
        return config
    return config.with_sandbox_overrides(no_commands=True)


def _listable_sessions(agent6_dir: Path) -> list[Path]:
    """Every session dir under the state base, newest first.

    `list_sessions` is named for what it lists: every bucket, husks excluded,
    the same sessions the CLI and the web hub list. `viewmodel.session_dirs`
    is the one enumeration behind all three.
    """
    return session_dirs(agent6_dir, SESSION_BUCKETS)


def _most_recent_session_id(agent6_dir: Path) -> str | None:
    candidates = _listable_sessions(agent6_dir)
    return candidates[0].name if candidates else None


# ---------------------------------------------------------------------------
# Server.
# ---------------------------------------------------------------------------


class MCPServer:
    """One serve() session. Owns a `ToolDispatcher` and reads/writes
    line-delimited JSON-RPC over the supplied stdio handles."""

    def __init__(
        self,
        *,
        root: Path,
        config: Config,
        stdin: IO[bytes],
        stdout: IO[bytes],
    ) -> None:
        self._root = root.resolve()
        self._config = config
        self._agent6_dir = resolved_state_dir(self._root)
        self._stdin = stdin
        self._stdout = stdout
        self._dispatcher = ToolDispatcher(root=self._root, config=_no_one_to_ask(config))
        # `ask` clamps to no-commands (no one to answer here) and `no` is the
        # operator's own refusal; either way the command tools are GONE from
        # tools/list, not offered-and-failing. _call_tool still names the real
        # reason for a client that calls one by name anyway.
        self._commands_withdrawn = config.sandbox.run_commands in ("ask", "no")
        self._gate_missing = not config.workflow.verify_command
        specs = self._build_tools()
        if self._commands_withdrawn:
            specs = [t for t in specs if t.name not in _COMMAND_TOOLS]
        if self._gate_missing:
            specs = [t for t in specs if t.name not in _GATE_TOOLS]
        self._tools: dict[str, _ToolSpec] = {t.name: t for t in specs}

    # ---- public entry point -----

    def serve(self) -> None:
        """Read JSON-RPC messages from stdin until EOF. Each request is
        answered on stdout. Notifications (no `id`) are ignored."""
        try:
            while True:
                # Bounded read (mirrors tools/mcp_client._read_loop): an
                # unbounded readline() buffers the entire line into memory
                # BEFORE the size check, so the cap could not prevent memory
                # exhaustion by a runaway client.
                line = self._stdin.readline(_MAX_LINE_BYTES + 1)
                if not line:
                    return
                if len(line) > _MAX_LINE_BYTES:
                    # Oversized: drain the rest of this line (up to its
                    # newline) in bounded chunks, discarding, then drop the
                    # whole payload.
                    while line and not line.endswith(b"\n"):
                        line = self._stdin.readline(_MAX_LINE_BYTES + 1)
                    continue
                try:
                    msg = json.loads(line)
                except json.JSONDecodeError:
                    # JSON-RPC's answer to an unparseable request: the parse
                    # error with a null id, never silence.
                    self._reply(None, error={"code": -32700, "message": "parse error"})
                    continue
                if not isinstance(msg, dict):
                    continue
                self._handle(msg)
        finally:
            self._dispatcher.close()

    # ---- tool catalog -----

    def _build_tools(self) -> list[_ToolSpec]:
        return [
            _ToolSpec(
                name="run_verify",
                description=(
                    "Run the workspace's configured verify command inside the agent6"
                    " jail. Returns {command, returncode, stdout, stderr,"
                    " duration_s, exec_failed}."
                ),
                input_schema={
                    "type": "object",
                    "properties": {},
                    "additionalProperties": False,
                },
                handler=self._h_run_verify,
            ),
            _ToolSpec(
                name="run_in_sandbox",
                description=(
                    "Run an arbitrary argv inside the agent6 jail (Landlock + seccomp"
                    " + user namespace). Requires [sandbox].run_commands = 'yes' in"
                    " your config; 'ask' and 'no' are refused at the MCP boundary"
                    " because there is no operator to prompt."
                ),
                input_schema={
                    "type": "object",
                    "properties": {
                        "argv": {
                            "type": "array",
                            "items": {"type": "string"},
                            "minItems": 1,
                        },
                    },
                    "required": ["argv"],
                    "additionalProperties": False,
                },
                handler=self._h_run_in_sandbox,
            ),
            _ToolSpec(
                name="apply_patch_in_sandbox",
                description=(
                    "Apply a unified-diff patch to a single file under the workspace"
                    " root, then re-run the verify command. Returns {apply: {...},"
                    " verify: {...}}. The caller is responsible for reverting on"
                    " verify failure; agent6 does not auto-revert."
                ),
                input_schema={
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "minLength": 1},
                        "patch": {"type": "string", "minLength": 1},
                    },
                    "required": ["path", "patch"],
                    "additionalProperties": False,
                },
                handler=self._h_apply_patch_in_sandbox,
            ),
            _ToolSpec(
                name="query_dag",
                description=(
                    "Load the task graph for a given run id (default: most recent)."
                    " Returns {session_id, nodes: {id: {title, status, parent_id, ...}}}."
                ),
                input_schema={
                    "type": "object",
                    "properties": {"session_id": {"type": "string"}},
                    "additionalProperties": False,
                },
                handler=self._h_query_dag,
            ),
            _ToolSpec(
                name="list_sessions",
                description=(
                    "Enumerate sessions under the per-repo state dir (most-recent first) with"
                    " their manifest summary (task, base_sha, models, ...)."
                ),
                input_schema={
                    "type": "object",
                    "properties": {},
                    "additionalProperties": False,
                },
                handler=self._h_list_sessions,
            ),
        ]

    # ---- request routing -----

    def _handle(self, msg: dict[str, Any]) -> None:
        method = msg.get("method")
        req_id = msg.get("id")
        raw_params = msg.get("params")
        params = raw_params if isinstance(raw_params, dict) else {}
        if not isinstance(method, str):
            return
        # Notifications carry no id and expect no response.
        if req_id is None:
            return
        try:
            result = self._route(method, params)
            self._reply(req_id, result=result)
        except _RpcError as exc:
            self._reply(req_id, error={"code": exc.code, "message": exc.message})

    def _route(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        if method == "initialize":
            return {
                "protocolVersion": _PROTOCOL_VERSION,
                "capabilities": {"tools": {}},
                "serverInfo": {"name": _SERVER_NAME, "version": __version__},
            }
        if method == "tools/list":
            return {
                "tools": [
                    {
                        "name": t.name,
                        "description": t.description,
                        "inputSchema": t.input_schema,
                    }
                    for t in self._tools.values()
                ],
            }
        if method == "tools/call":
            return self._call_tool(params)
        raise _RpcError(-32601, f"unknown method: {method!r}")

    def _call_tool(self, params: dict[str, Any]) -> dict[str, Any]:
        name = params.get("name")
        raw_args = params.get("arguments")
        args = raw_args if isinstance(raw_args, dict) else {}
        if not isinstance(name, str) or name not in self._tools:
            if isinstance(name, str) and name in _COMMAND_TOOLS and self._commands_withdrawn:
                mode = self._config.sandbox.run_commands
                detail = (
                    "no operator answers the MCP boundary"
                    if mode == "ask"
                    else "the operator disabled commands"
                )
                raise _RpcError(
                    -32601,
                    f"{name} is withdrawn: [sandbox].run_commands = {mode!r} ({detail})",
                )
            if isinstance(name, str) and name in _GATE_TOOLS and self._gate_missing:
                raise _RpcError(
                    -32601,
                    f"{name} is withdrawn: this workspace has no [workflow]"
                    " verify_command, so there is no gate to run",
                )
            raise _RpcError(-32601, f"unknown tool: {name!r}")
        if raw_args is not None and not isinstance(raw_args, dict):
            raise _RpcError(-32602, "arguments must be an object")
        violation = _schema_violation(args, self._tools[name].input_schema, "arguments")
        if violation is not None:
            raise _RpcError(-32602, violation)
        try:
            payload = self._tools[name].handler(args)
        except (ToolError, OperatorCommandUnexecutable) as exc:
            # OperatorCommandUnexecutable aborts a RUN loudly by design; here
            # the contract is an isError result -- escaping killed the whole
            # serve process and every later client call died on a broken pipe.
            return {
                "content": [{"type": "text", "text": str(exc)}],
                "isError": True,
            }
        return {
            "content": [{"type": "text", "text": json.dumps(payload, separators=(",", ":"))}],
            "structuredContent": payload,
        }

    def _reply(
        self,
        req_id: Any,
        *,
        result: Any = None,
        error: dict[str, Any] | None = None,
    ) -> None:
        msg: dict[str, Any] = {"jsonrpc": "2.0", "id": req_id}
        if error is not None:
            msg["error"] = error
        else:
            msg["result"] = result
        self._stdout.write(json.dumps(msg, separators=(",", ":")).encode("utf-8") + b"\n")
        self._stdout.flush()

    # ---- tool handlers -----

    def _h_run_verify(self, _args: dict[str, Any]) -> dict[str, Any]:
        return self._dispatcher.dispatch("run_verify_command", {}).to_wire()

    def _h_run_in_sandbox(self, args: dict[str, Any]) -> dict[str, Any]:
        argv = args.get("argv")
        if not isinstance(argv, list) or not argv or not all(isinstance(s, str) for s in argv):
            raise ToolError("argv must be a non-empty list of strings")
        return self._dispatcher.dispatch("run_command", {"argv": list(argv)}).to_wire()

    def _h_apply_patch_in_sandbox(self, args: dict[str, Any]) -> dict[str, Any]:
        path = args.get("path")
        patch = args.get("patch")
        if not isinstance(path, str) or not isinstance(patch, str):
            raise ToolError("path and patch must be strings")
        apply_result = self._dispatcher.dispatch("apply_patch", {"path": path, "patch": patch})
        verify_result = self._dispatcher.dispatch("run_verify_command", {})
        return {"apply": apply_result.to_wire(), "verify": verify_result.to_wire()}

    def _h_query_dag(self, args: dict[str, Any]) -> dict[str, Any]:
        session_id_arg = args.get("session_id")
        if isinstance(session_id_arg, str) and session_id_arg:
            # Client-supplied: reject a traversing/absolute id before it builds a
            # run dir, so a query cannot read another repo's state (or anywhere).
            if not is_safe_session_id(session_id_arg):
                raise ToolError(f"invalid session_id: {session_id_arg!r}")
            session_id = session_id_arg
        else:
            resolved = _most_recent_session_id(self._agent6_dir)
            if resolved is None:
                raise ToolError("no sessions found under the agent6 state dir")
            session_id = resolved
        try:
            layout = resolve_session(self._agent6_dir, session_id)
        except SessionIdError as exc:
            raise ToolError(str(exc)) from exc
        nodes = load_graph(layout)
        return {
            "session_id": session_id,
            "nodes": {nid: node.model_dump(mode="json") for nid, node in nodes.items()},
        }

    def _h_list_sessions(self, _args: dict[str, Any]) -> dict[str, Any]:
        entries: list[dict[str, Any]] = []
        for d in _listable_sessions(self._agent6_dir):
            summary: dict[str, Any] = {"session_id": d.name}
            # A missing/corrupt manifest lists the run without one.
            with contextlib.suppress(ManifestError):
                summary["manifest"] = read_manifest(d).model_dump(mode="json")
            entries.append(summary)
        return {"sessions": entries}


# ---------------------------------------------------------------------------
# CLI entry point.
# ---------------------------------------------------------------------------


def run_server(config_path: Path | None) -> int:
    """`agent6 mcp serve` body. Loads the layered effective config
    (global + repo, plus an optional explicit `config_path`), spawns an
    :class:`MCPServer` against cwd, and serves until stdin EOF. Returns 0
    on clean exit."""
    root = Path.cwd()
    try:
        cfg = load_effective(root, config_path).config
    except Exception as exc:
        print(f"ERROR: failed to load config: {exc}", file=sys.stderr)
        return 2
    server = MCPServer(
        root=root,
        config=cfg,
        stdin=sys.stdin.buffer,
        stdout=sys.stdout.buffer,
    )
    server.serve()
    return 0
