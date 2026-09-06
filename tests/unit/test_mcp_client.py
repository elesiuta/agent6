# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Smoke-tests for the stdio MCP client.

Uses a tiny in-tree Python "MCP server" that talks just enough JSON-RPC
to satisfy ``initialize`` + ``tools/list`` + ``tools/call``. No external
dependency.
"""

from __future__ import annotations

import contextlib
import json
import os
import signal
import subprocess
import sys
import textwrap
import threading
import time
from pathlib import Path

import pytest

from agent6.tools.mcp_client import (
    MCP_TOOL_PREFIX,
    MCPError,
    MCPManager,
    MCPServerSpec,
)
from agent6.types import JailPolicy


def _fake_server_argv(
    *,
    hang: bool = False,
    crash_after_init: bool = False,
    bad_tool: bool = False,
    newline_tool: bool = False,
    single_tool: bool = False,
    sleep_on: str = "",
) -> tuple[str, ...]:
    """Return argv that runs a tiny Python MCP server inline.

    The server speaks line-delimited JSON-RPC 2.0 over stdio:
    * ``initialize``  -> empty result
    * ``tools/list``  -> two tools: ``echo`` and ``shout``
    * ``tools/call``  -> echoes back the args under "content"

    Knobs:
    * ``hang=True``: never responds (forces client timeout).
    * ``crash_after_init=True``: exits 0 right after handshake.
    * ``single_tool=True``: advertises ``echo`` alone.
    * ``sleep_on=TEXT``: a ``tools/call`` with that text sleeps 3s before answering.
    """
    script = textwrap.dedent(
        f"""
        import json, sys, time
        HANG = {hang!r}
        CRASH = {crash_after_init!r}
        BAD_TOOL = {bad_tool!r}
        NEWLINE_TOOL = {newline_tool!r}
        SINGLE_TOOL = {single_tool!r}
        SLEEP_ON = {sleep_on!r}
        def reply(req_id, result):
            sys.stdout.write(json.dumps({{
                "jsonrpc": "2.0", "id": req_id, "result": result,
            }}) + "\\n")
            sys.stdout.flush()
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            msg = json.loads(line)
            method = msg.get("method")
            if method is None:
                continue
            if "id" not in msg:
                continue  # notification
            if HANG:
                continue
            if method == "initialize":
                reply(msg["id"], {{"protocolVersion": "2024-11-05",
                                    "capabilities": {{}},
                                    "serverInfo": {{"name": "fake", "version": "0"}}}})
                if CRASH:
                    sys.exit(0)
                continue
            if method == "tools/list":
                tools = [
                    {{"name": "echo", "description": "echo the input",
                      "inputSchema": {{"type": "object",
                                       "properties": {{"text": {{"type": "string"}}}}}}}},
                    {{"name": "shout", "description": "upper-case echo",
                      "inputSchema": {{"type": "object",
                                       "properties": {{"text": {{"type": "string"}}}}}}}},
                ]
                if SINGLE_TOOL:
                    tools = tools[:1]
                if BAD_TOOL:
                    tools.append({{"name": "has a space", "description": "invalid",
                                   "inputSchema": {{"type": "object"}}}})
                if NEWLINE_TOOL:
                    tools.append({{"name": "sneaky\\n", "description": "trailing newline",
                                   "inputSchema": {{"type": "object"}}}})
                reply(msg["id"], {{"tools": tools}})
                continue
            if method == "tools/call":
                args = msg["params"].get("arguments", {{}})
                tname = msg["params"].get("name")
                if SLEEP_ON and args.get("text") == SLEEP_ON:
                    time.sleep(3)
                if tname == "shout":
                    out = str(args.get("text", "")).upper()
                else:
                    out = str(args.get("text", ""))
                reply(msg["id"], {{"content": [
                    {{"type": "text", "text": out}}
                ]}})
                continue
            reply(msg["id"], {{}})
        """
    )
    return (sys.executable, "-c", script)


def test_manager_starts_and_discovers_tools() -> None:
    mgr = MCPManager.start(
        [
            MCPServerSpec(
                name="fake", command=_fake_server_argv(), startup_timeout_s=5.0, call_timeout_s=5.0
            )
        ],
    )
    try:
        descs = mgr.descriptors()
        names = sorted(d.qualified_name for d in descs)
        assert names == [
            f"{MCP_TOOL_PREFIX}fake__echo",
            f"{MCP_TOOL_PREFIX}fake__shout",
        ]
        for d in descs:
            assert d.input_schema.get("type") == "object"
    finally:
        mgr.close()


def test_manager_skips_tools_with_invalid_names() -> None:
    # A server-advertised tool whose name isn't [A-Za-z0-9_-] can't form a valid
    # provider tool name; it must be skipped, not poison the whole tools array.
    mgr = MCPManager.start(
        [
            MCPServerSpec(
                name="fake",
                command=_fake_server_argv(bad_tool=True),
                startup_timeout_s=5.0,
                call_timeout_s=5.0,
            )
        ]
    )
    try:
        names = sorted(d.qualified_name for d in mgr.descriptors())
        assert names == [f"{MCP_TOOL_PREFIX}fake__echo", f"{MCP_TOOL_PREFIX}fake__shout"]
    finally:
        mgr.close()


def test_a_tool_name_with_a_trailing_newline_is_skipped() -> None:
    """`re.match` against a `^[A-Za-z0-9_-]+$` pattern accepts a terminal
    newline: `$` matches just before it, so a tool advertised as "sneaky\\n"
    passed the filter and registered as mcp__fake__sneaky\\n -- a newline
    spliced into the LLM-visible tool definition. fullmatch admits no trailing
    newline, so the tool is skipped like any other invalid name."""
    mgr = MCPManager.start(
        [
            MCPServerSpec(
                name="fake",
                command=_fake_server_argv(newline_tool=True),
                startup_timeout_s=5.0,
                call_timeout_s=5.0,
            )
        ]
    )
    try:
        names = sorted(d.qualified_name for d in mgr.descriptors())
        assert names == [f"{MCP_TOOL_PREFIX}fake__echo", f"{MCP_TOOL_PREFIX}fake__shout"]
    finally:
        mgr.close()


def test_manager_routes_calls_to_right_server_and_tool() -> None:
    mgr = MCPManager.start(
        [
            MCPServerSpec(
                name="fake", command=_fake_server_argv(), startup_timeout_s=5.0, call_timeout_s=5.0
            )
        ],
    )
    try:
        echo = mgr.call(f"{MCP_TOOL_PREFIX}fake__echo", {"text": "hi"})
        assert echo["content"][0]["text"] == "hi"
        shout = mgr.call(f"{MCP_TOOL_PREFIX}fake__shout", {"text": "hi"})
        assert shout["content"][0]["text"] == "HI"
    finally:
        mgr.close()


def test_call_tool_rejects_unadvertised_tool_name() -> None:
    """The tool name rides in from the LLM. A name the server never advertised
    (one filtered at registration, or a hidden tool the model was told to reach)
    must be refused HERE, before any tools/call leaves agent6 -- otherwise the
    fake server below happily echoes it back as a successful result."""
    mgr = MCPManager.start(
        [
            MCPServerSpec(
                name="fake", command=_fake_server_argv(), startup_timeout_s=5.0, call_timeout_s=5.0
            )
        ]
    )
    try:
        with pytest.raises(MCPError, match="did not advertise"):
            mgr.call(f"{MCP_TOOL_PREFIX}fake__sneaky", {"text": "hi"})
    finally:
        mgr.close()


def test_manager_rejects_non_mcp_name() -> None:
    mgr = MCPManager.start([])
    try:
        with pytest.raises(MCPError, match="not an MCP tool name"):
            mgr.call("not_mcp", {})
    finally:
        mgr.close()


def test_manager_rejects_unknown_server() -> None:
    mgr = MCPManager.start([])
    try:
        with pytest.raises(MCPError, match="unknown MCP server"):
            mgr.call(f"{MCP_TOOL_PREFIX}nope__t", {})
    finally:
        mgr.close()


def test_manager_logs_and_skips_unstartable_server() -> None:
    logs: list[str] = []
    mgr = MCPManager.start(
        [
            MCPServerSpec(
                name="bogus",
                command=("/this/binary/does/not/exist/agent6-test", "x"),
                startup_timeout_s=1.0,
                call_timeout_s=1.0,
            )
        ],
        logger=logs.append,
    )
    try:
        assert mgr.descriptors() == ()
        assert any("failed to start" in m for m in logs)
    finally:
        mgr.close()


def test_manager_times_out_on_hanging_server() -> None:
    # 0.5s startup timeout; the hang server never responds, so start()
    # should log the failure and the manager should end up with zero
    # servers. We do NOT raise from MCPManager.start because the
    # design is "one bad server doesn't take the run down".
    logs: list[str] = []
    mgr = MCPManager.start(
        [
            MCPServerSpec(
                name="hang",
                command=_fake_server_argv(hang=True),
                startup_timeout_s=0.5,
                call_timeout_s=0.5,
            )
        ],
        logger=logs.append,
    )
    try:
        assert mgr.descriptors() == ()
        assert any("timed out" in m for m in logs)
    finally:
        mgr.close()


def test_a_timeout_carries_the_servers_own_words() -> None:
    """A server that logs its reason and then waits on stdin is the common
    shape of a misconfigured one. Its reason sat in the stderr buffer while the
    operator got a bare timeout, and `mcp connect`'s hint sent them to sandbox
    grants that were not the problem -- the same server EXITING says why."""
    logs: list[str] = []
    mgr = MCPManager.start(
        [
            MCPServerSpec(
                name="errlog",
                command=(
                    "/bin/sh",
                    "-c",
                    'echo "FATAL: missing API token" >&2; sleep 30',
                ),
                startup_timeout_s=0.5,
                call_timeout_s=0.5,
            )
        ],
        logger=logs.append,
    )
    try:
        assert any("FATAL: missing API token" in m for m in logs), logs
    finally:
        mgr.close()


def test_a_timed_out_call_restarts_the_server_before_the_next_call() -> None:
    """A stdio server still busy with the call it never answered is wedged
    for the next one, which then timed out too. agent6 owns the spawn: the
    timed-out call's error names the restart, and the next call gets a fresh
    server."""
    mgr = MCPManager.start(
        [
            MCPServerSpec(
                name="slow",
                command=_fake_server_argv(sleep_on="sleep"),
                startup_timeout_s=5.0,
                call_timeout_s=0.5,
            )
        ]
    )
    try:
        with pytest.raises(
            MCPError, match=r"timed out after 0\.5s on tools/call; the server was restarted"
        ):
            mgr.call(f"{MCP_TOOL_PREFIX}slow__echo", {"text": "sleep"})
        started = time.monotonic()
        result = mgr.call(f"{MCP_TOOL_PREFIX}slow__echo", {"text": "hi"})
        assert result == {"content": [{"type": "text", "text": "hi"}]}
        assert time.monotonic() - started < 0.5, "the second call waited on the wedged server"
    finally:
        mgr.close()


def test_the_started_line_counts_one_tool_as_one_tool() -> None:
    logs: list[str] = []
    mgr = MCPManager.start(
        [
            MCPServerSpec(
                name="one",
                command=_fake_server_argv(single_tool=True),
                startup_timeout_s=5.0,
                call_timeout_s=5.0,
            )
        ],
        logger=logs.append,
    )
    mgr.close()
    assert any(line.startswith("[mcp] started 'one' (1 tool, network: ") for line in logs), logs


def test_a_stderr_tail_is_cut_at_a_line_and_says_what_was_dropped() -> None:
    from agent6.portable import stderr_tail

    assert stderr_tail([b"short\n"]) == "short"
    kept = [b"first line\n" + b"x" * 390 + b"\nlast line\n"]
    assert stderr_tail(kept) == "\u2026[agent6: 402 earlier chars cut]\nlast line"
    # One line longer than the limit keeps its end, still marked.
    assert (
        stderr_tail([b"y" * 500], limit=10) == "\u2026[agent6: 490 earlier chars cut]\n" + "y" * 10
    )


def test_manager_close_is_idempotent() -> None:
    mgr = MCPManager.start(
        [
            MCPServerSpec(
                name="fake", command=_fake_server_argv(), startup_timeout_s=5.0, call_timeout_s=5.0
            )
        ]
    )
    mgr.close()
    mgr.close()  # must not raise


def test_concurrent_calls_do_not_interleave_stdin_writes() -> None:
    """tools/call from concurrent threads (explore-review seats share one
    dispatcher across a thread pool) must serialize on the server's stdin:
    pipe writes larger than PIPE_BUF interleave across unlocked writers,
    corrupting the JSON-RPC framing -- the server read malformed JSON and
    died, failing every in-flight call."""
    mgr = MCPManager.start(
        [
            MCPServerSpec(
                name="fake",
                command=_fake_server_argv(),
                startup_timeout_s=10.0,
                call_timeout_s=30.0,
            )
        ]
    )
    try:
        payloads = {i: f"p{i}-" + "x" * 300_000 for i in range(8)}
        results: dict[int, str] = {}
        errors: list[Exception] = []

        def call(i: int) -> None:
            try:
                out = mgr.call(f"{MCP_TOOL_PREFIX}fake__echo", {"text": payloads[i]})
                results[i] = out["content"][0]["text"]
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=call, args=(i,)) for i in payloads]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)
        assert not errors
        assert results == payloads
    finally:
        mgr.close()


def test_a_server_is_not_handed_the_provider_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    """The spawn passed no `env`, so a server inherited the agent's FULL
    environment -- including the keys resolved via `[providers.*].api_key_env`.
    An MCP server is third-party code that may log or forward what it is given.
    Proved by asking the server itself what it can see."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-DECOY")
    monkeypatch.setenv("MCP_PROBE_TOKEN", "named-and-wanted")
    script = (
        "import json,os,sys\n"
        "def w(o): sys.stdout.write(json.dumps(o)+chr(10)); sys.stdout.flush()\n"
        "for line in sys.stdin:\n"
        "    m=json.loads(line)\n"
        "    if m.get('method')=='initialize':\n"
        "        w({'jsonrpc':'2.0','id':m['id'],'result':{'protocolVersion':'2024-11-05',"
        "'capabilities':{},'serverInfo':{'name':'p','version':'1'}}})\n"
        "    elif m.get('method')=='tools/list':\n"
        "        seen=sorted(k for k in os.environ if 'API_KEY' in k or k=='MCP_PROBE_TOKEN')\n"
        "        w({'jsonrpc':'2.0','id':m['id'],'result':{'tools':[{'name':'x',"
        "'description':json.dumps(seen),'inputSchema':{'type':'object'}}]}})\n"
    )
    mgr = MCPManager.start(
        [
            MCPServerSpec(
                name="probe",
                command=(sys.executable, "-c", script),
                startup_timeout_s=10.0,
                call_timeout_s=10.0,
                pass_env=("MCP_PROBE_TOKEN",),
            )
        ]
    )
    try:
        seen = json.loads(mgr.descriptors()[0].description)
    finally:
        mgr.close()
    assert seen == ["MCP_PROBE_TOKEN"], "a server sees only what it named"


def test_oversized_descriptions_and_results_degrade_instead_of_breaking_turns() -> None:
    """LOW finding: under the 8 MiB transport cap, a compromised operator-run
    server could still emit multi-MiB descriptions (riding in EVERY provider
    request) and results (flooding the context), breaking every turn. Both are
    bounded at the trust boundary with a marker."""
    from agent6.tools.mcp_client import (
        _MAX_INLINE_TEXT_CHARS,  # pyright: ignore[reportPrivateUsage]
        _MAX_RESULT_CHARS,  # pyright: ignore[reportPrivateUsage]
        _bounded_inline_text,  # pyright: ignore[reportPrivateUsage]
        _bounded_result,  # pyright: ignore[reportPrivateUsage]
    )

    desc = _bounded_inline_text("d" * (_MAX_INLINE_TEXT_CHARS * 4))
    assert len(desc) <= _MAX_INLINE_TEXT_CHARS + 40 and "truncated" in desc
    assert _bounded_inline_text("short") == "short"

    huge = {
        "content": [
            {"type": "text", "text": "x" * (_MAX_RESULT_CHARS + 100)},
            {"type": "image", "data": "A" * 1000},
        ]
    }
    out = _bounded_result(huge)
    assert len(json.dumps(out)) <= _MAX_RESULT_CHARS + 500
    (block,) = out["content"]
    assert block["type"] == "text" and "everything else dropped" in block["text"]

    small = {"content": [{"type": "text", "text": "ok"}]}
    assert _bounded_result(small) is small


def test_an_oversized_echo_result_comes_back_bounded() -> None:
    """The end-to-end path: a server whose result serializes past the cap
    reaches the model as one bounded text block, not a context flood."""
    from agent6.tools.mcp_client import _MAX_RESULT_CHARS  # pyright: ignore[reportPrivateUsage]

    mgr = MCPManager.start(
        [
            MCPServerSpec(
                name="fake", command=_fake_server_argv(), startup_timeout_s=5.0, call_timeout_s=15.0
            )
        ],
    )
    try:
        big = "y" * (_MAX_RESULT_CHARS + 10)
        result = mgr.call(MCP_TOOL_PREFIX + "fake__echo", {"text": big})
        assert len(json.dumps(result)) <= _MAX_RESULT_CHARS + 500
        (block,) = result["content"]
        assert "kept up to" in block["text"]
    finally:
        mgr.close()


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="setsid sweep is Linux-only")
def test_manager_close_kills_setsid_escapee(tmp_path: Path) -> None:
    """Closing an MCP server must not leave a process behind. A server with no
    PID namespace (`hardened`, or `none` here) that forks a `setsid` child puts
    that child outside the launcher's process group, so it reparents onto the
    agent and survives a kill of the launcher pid alone. close() must run the
    escapee sweep, not just signal the launcher."""
    from agent6.sandbox.jail import _become_subreaper  # pyright: ignore[reportPrivateUsage]

    # So the escapee reparents onto THIS process, where the sweep looks.
    _become_subreaper()
    marker = tmp_path / "escapee.pid"
    script = (
        "import json, os, sys, time\n"
        "if os.fork() == 0:\n"
        "    os.setsid()\n"  # leave the launcher's group: only the sweep catches this
        "    with open(sys.argv[1], 'w') as f:\n"
        "        f.write(str(os.getpid()))\n"
        "    time.sleep(30)\n"
        "    os._exit(0)\n"
        "def reply(i, r):\n"
        "    sys.stdout.write(json.dumps({'jsonrpc': '2.0', 'id': i, 'result': r}) + '\\n')\n"
        "    sys.stdout.flush()\n"
        "for line in sys.stdin:\n"
        "    m = json.loads(line or '{}')\n"
        "    if m.get('method') is None or 'id' not in m:\n"
        "        continue\n"
        "    if m['method'] == 'initialize':\n"
        "        reply(m['id'], {'protocolVersion': '2024-11-05', 'capabilities': {}})\n"
        "    elif m['method'] == 'tools/list':\n"
        "        reply(m['id'], {'tools': []})\n"
        "    else:\n"
        "        reply(m['id'], {})\n"
    )
    argv = (sys.executable, "-c", script, str(marker))
    spec = MCPServerSpec(
        name="esc",
        command=argv,
        startup_timeout_s=10.0,
        call_timeout_s=10.0,
        policy=JailPolicy(
            cwd=tmp_path, argv=argv, isolation="none", network="none", timeout_s=30.0
        ),
    )
    gc_pid: int | None = None
    mgr = MCPManager.start([spec])
    try:
        assert mgr.failures == (), mgr.failures
        for _ in range(100):
            text = marker.read_text().strip() if marker.exists() else ""
            if text:
                gc_pid = int(text)
                break
            time.sleep(0.05)
        assert gc_pid is not None, "the escapee never recorded its pid"
        assert _pid_alive(gc_pid), "the escapee should be running before close"
        mgr.close()
        assert not _pid_alive(gc_pid), f"the setsid escapee {gc_pid} survived MCPManager.close()"
    finally:
        mgr.close()
        if gc_pid is not None:
            with contextlib.suppress(ProcessLookupError, OSError):
                os.kill(gc_pid, signal.SIGKILL)


def test_initialize_sends_the_canonical_version(tmp_path: Path) -> None:
    """clientInfo.version hardcoded "0" while every other public surface
    imports agent6.__version__; the handshake now carries the canonical one."""
    import agent6
    from agent6.tools.mcp_client import _MCPServer  # pyright: ignore[reportPrivateUsage]

    seen = tmp_path / "init.json"
    srv_py = tmp_path / "srv.py"
    srv_py.write_text(
        "import json, sys\n"
        "line = sys.stdin.readline()\n"
        f"open({str(seen)!r}, 'w').write(line)\n"
        "msg = json.loads(line)\n"
        "print(json.dumps({'jsonrpc': '2.0', 'id': msg['id'], 'result': {\n"
        "    'protocolVersion': '2024-11-05', 'capabilities': {},\n"
        "    'serverInfo': {'name': 'fake', 'version': '1'}}}))\n"
        "sys.stdout.flush()\n"
        # Answer the tools/list request it receives, by its id: the client sends
        # notifications/initialized first, and a reply written before the client
        # has registered the request's id is dropped by the reader.
        "while True:\n"
        "    req = json.loads(sys.stdin.readline())\n"
        "    if req.get('method') == 'tools/list':\n"
        "        break\n"
        "print(json.dumps({'jsonrpc': '2.0', 'id': req['id'], 'result': {'tools': []}}))\n"
        "sys.stdout.flush()\n"
        "sys.stdin.read()\n",
        encoding="utf-8",
    )
    srv = _MCPServer(
        name="v",
        command=("/usr/bin/python3", str(srv_py)),
        startup_timeout_s=5.0,
        call_timeout_s=5.0,
        pass_env=(),
        policy=None,
    )
    try:
        srv.start()
    finally:
        srv.close()
    init = json.loads(seen.read_text(encoding="utf-8"))
    assert init["params"]["clientInfo"]["version"] == agent6.__version__


def test_unconfined_server_ties_to_the_agent(monkeypatch: pytest.MonkeyPatch) -> None:
    """The operator-opted-out (no-jail) MCP spawn is the spawner's `none`
    level: the parent-death tie (a stdio server that ignores stdin EOF must
    not outlive a SIGKILLed agent6), its own session, and the pid registered
    so a sibling handle's escapee sweep spares it. A second Popen here
    carried the first two and not the third."""
    from agent6.sandbox import jail as jail_mod
    from agent6.tools import mcp_client

    captured: dict[str, object] = {}

    class _P:
        pid = 4242
        stdin = None
        stdout = None
        stderr = None

    def fake_popen(*args: object, **kwargs: object) -> _P:
        captured.update(kwargs)
        return _P()

    monkeypatch.setattr(jail_mod.subprocess, "Popen", fake_popen)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-secret")
    live = jail_mod._live_launchers  # pyright: ignore[reportPrivateUsage]
    try:
        mcp_client._spawn_server(  # pyright: ignore[reportPrivateUsage]
            ("fake-server",), pass_env=(), policy=None
        )
        assert 4242 in live
    finally:
        live.discard(4242)
    assert captured.get("preexec_fn") is not None
    assert captured.get("start_new_session") is True
    env = captured.get("env")
    assert isinstance(env, dict) and "ANTHROPIC_API_KEY" not in env


def test_a_failed_starts_survivors_reach_the_managers_close(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A server whose handshake fails is closed by the manager's start, and
    the pids that close's sweep could not kill were dropped there."""
    from agent6.tools.mcp_client import MCPManager, MCPServerSpec

    def sweep(exclude: frozenset[int]) -> frozenset[int]:
        return frozenset({4321})

    monkeypatch.setattr("agent6.sandbox.jail._kill_escapees", sweep)
    mgr = MCPManager.start(
        [
            MCPServerSpec(
                name="dead",
                command=("/bin/sh", "-c", "exit 1"),
                startup_timeout_s=2.0,
                call_timeout_s=1.0,
            )
        ],
        logger=lambda _m: None,
    )
    assert [f.name for f in mgr.failures] == ["dead"]
    assert mgr.close() == frozenset({4321})


def test_a_restarted_servers_survivors_accumulate(monkeypatch: pytest.MonkeyPatch) -> None:
    """A server closed twice (a restart, then the teardown) hands back the
    survivors of both closes, not the last one's."""
    from agent6.sandbox.jail import JailedProcess
    from agent6.tools.mcp_client import _MCPServer  # pyright: ignore[reportPrivateUsage]

    pids = iter([4321, 4322])

    def sweep(exclude: frozenset[int]) -> frozenset[int]:
        return frozenset({next(pids)})

    monkeypatch.setattr("agent6.sandbox.jail._kill_escapees", sweep)
    srv = _MCPServer(name="a", command=("x",), startup_timeout_s=1.0, call_timeout_s=1.0)
    procs = [subprocess.Popen(["sleep", "60"], start_new_session=True) for _ in range(2)]
    try:
        srv._proc = JailedProcess(procs[0])  # pyright: ignore[reportPrivateUsage]
        assert srv.close() == frozenset({4321})
        srv._proc = JailedProcess(procs[1])  # pyright: ignore[reportPrivateUsage]
        assert srv.close() == frozenset({4321, 4322})
    finally:
        for proc in procs:
            proc.kill()
            proc.wait(timeout=5)


def test_a_call_cut_short_by_another_callers_restart_is_retried_once() -> None:
    """Two callers on one server (review seats share a dispatcher): A's
    timeout replaces the process under B's in-flight call. B does not time
    out on a server that no longer exists, nor restart the fresh one: it
    goes once more on it. One restart in total."""
    mgr = MCPManager.start(
        [
            MCPServerSpec(
                name="slow",
                command=_fake_server_argv(sleep_on="sleep"),
                startup_timeout_s=5.0,
                call_timeout_s=0.5,
            )
        ]
    )
    outcomes: dict[str, object] = {}

    def call(tag: str, text: str) -> None:
        try:
            outcomes[tag] = mgr.call(f"{MCP_TOOL_PREFIX}slow__echo", {"text": text})
        except MCPError as exc:
            outcomes[tag] = str(exc)

    a = threading.Thread(target=call, args=("a", "sleep"))
    b = threading.Thread(target=call, args=("b", "hi"))
    started = time.monotonic()
    try:
        a.start()
        time.sleep(0.1)  # the server is busy with A's call when B's arrives
        b.start()
        a.join(timeout=10)
        b.join(timeout=10)
        assert outcomes["a"] == (
            "server 'slow' timed out after 0.5s on tools/call; the server was restarted"
        )
        assert outcomes["b"] == {"content": [{"type": "text", "text": "hi"}]}
        assert time.monotonic() - started < 3.0, "a caller waited on a server that was gone"
        server = mgr._servers["slow"]  # pyright: ignore[reportPrivateUsage]
        assert server._generation == 1  # pyright: ignore[reportPrivateUsage]
    finally:
        mgr.close()


def test_the_manager_hands_back_every_survivor_of_its_servers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A server's escapee the sweep could not kill was dropped on the floor by
    the client's close; the leg records what the manager hands back as a jail
    degradation, the way the run's own session close does. Driven through the
    real client: a stand-in with a `close()` of its own pinned only the union."""
    from agent6.sandbox.jail import JailedProcess
    from agent6.tools.mcp_client import (
        MCPManager,
        _MCPServer,  # pyright: ignore[reportPrivateUsage]
    )

    def sweep(exclude: frozenset[int]) -> frozenset[int]:
        return frozenset({4321})

    monkeypatch.setattr("agent6.sandbox.jail._kill_escapees", sweep)
    escapee = subprocess.Popen(["sleep", "60"], start_new_session=True)
    survived = _MCPServer(name="a", command=("x",), startup_timeout_s=1.0, call_timeout_s=1.0)
    survived._proc = JailedProcess(escapee)  # pyright: ignore[reportPrivateUsage]
    clean = _MCPServer(name="b", command=("x",), startup_timeout_s=1.0, call_timeout_s=1.0)
    manager = MCPManager()
    manager._servers = {"a": survived, "b": clean}  # pyright: ignore[reportPrivateUsage]
    try:
        assert manager.close() == frozenset({4321})
        assert manager.close() == frozenset()  # idempotent: nothing left to close
    finally:
        escapee.kill()
        escapee.wait(timeout=5)
