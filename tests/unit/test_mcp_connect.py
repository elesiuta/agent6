# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""`agent6 mcp connect` proves the server works BEFORE writing config.

A server named in config that turns out not to answer is a run that starts,
logs "failed to start", and quietly has fewer tools than the operator thinks --
discovered mid-task, if at all.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from agent6.config.layer import load_effective
from agent6.ui.cli.mcp_connect import cmd_mcp_connect, cmd_mcp_list

# The interpreter a jailed probe can reach: the run's sandbox grants /usr,
# not the venv (a server there needs `read_paths`, which is the point).
_JAIL_PYTHON = "/usr/bin/python3"


def _server_argv(*, tools: bool = True) -> list[str]:
    """A minimal stdio MCP server: handshake, then one tool (or none)."""
    listed = (
        '[{"name":"read_page","description":"Read a page.","inputSchema":{"type":"object"}}]'
        if tools
        else "[]"
    )
    script = (
        "import json,sys\n"
        "def w(o): sys.stdout.write(json.dumps(o)+chr(10)); sys.stdout.flush()\n"
        "for line in sys.stdin:\n"
        "    m=json.loads(line)\n"
        "    if m.get('method')=='initialize':\n"
        "        w({'jsonrpc':'2.0','id':m['id'],'result':{'protocolVersion':'2024-11-05',"
        "'capabilities':{},'serverInfo':{'name':'t','version':'1'}}})\n"
        "    elif m.get('method')=='tools/list':\n"
        f"        w({{'jsonrpc':'2.0','id':m['id'],'result':{{'tools':{listed}}}}})\n"
    )
    return [_JAIL_PYTHON, "-c", script]


def test_a_server_that_answers_is_written_with_its_tools_shown(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("AGENT6_CONFIG_HOME", str(tmp_path / "cfg"))

    rc = cmd_mcp_connect(
        "browser", command=_server_argv(), url="", token_env="", pass_env=[], to_repo=False
    )

    assert rc == 0
    out = capsys.readouterr().out
    assert "mcp__browser__read_page" in out, "the operator sees what they are getting"
    assert "Read a page." in out
    # The master switch is security-relevant and stays the operator's call.
    assert "config set mcp.enabled true" in out

    entry = load_effective(tmp_path).config.mcp.servers["browser"]
    assert entry.command == tuple(_server_argv())
    assert entry.enabled is True


def test_the_enable_hint_names_the_config_the_entry_went_to(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Following the hint after `--repo` must enable MCP for this repository,
    not for every repository on the machine."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("AGENT6_CONFIG_HOME", str(tmp_path / "cfg"))

    rc = cmd_mcp_connect(
        "browser", command=_server_argv(), url="", token_env="", pass_env=[], to_repo=True
    )

    assert rc == 0
    out = capsys.readouterr().out
    assert "written to the repo config" in out
    assert "agent6 config set --repo mcp.enabled true" in out
    assert cmd_mcp_list() == 0
    assert "DISABLED (agent6 config set --repo mcp.enabled true)" in capsys.readouterr().out


def test_a_binary_missing_on_the_host_is_named_without_a_sandbox_hint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A path that exists nowhere is a plain ENOENT: no `read_paths` grant
    would change it, so the sandbox hint stays out."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("AGENT6_CONFIG_HOME", str(tmp_path / "cfg"))

    rc = cmd_mcp_connect(
        "dead",
        command=["/nonexistent/agent6-test-server"],
        url="",
        token_env="",
        pass_env=[],
        to_repo=False,
    )

    assert rc == 1
    err = capsys.readouterr().err
    assert "/nonexistent/agent6-test-server: no such file on this host" in err
    assert "read_paths" not in err


def test_an_existing_file_that_is_not_executable_is_named_as_such(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("AGENT6_CONFIG_HOME", str(tmp_path / "cfg"))
    script = tmp_path / "server.py"
    script.write_text("print(1)\n", encoding="utf-8")
    script.chmod(0o644)

    rc = cmd_mcp_connect(
        "plain", command=[str(script)], url="", token_env="", pass_env=[], to_repo=False
    )

    assert rc == 1
    assert f"{script}: not executable on this host" in capsys.readouterr().err


def test_a_server_that_does_not_answer_writes_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The whole point of the order: config never names a server that failed."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("AGENT6_CONFIG_HOME", str(tmp_path / "cfg"))

    rc = cmd_mcp_connect(
        "dead",
        command=["/nonexistent/agent6-test-server"],
        url="",
        token_env="",
        pass_env=[],
        to_repo=False,
    )

    assert rc == 1
    assert "nothing was written" in capsys.readouterr().err
    assert load_effective(tmp_path).config.mcp.servers == {}


def test_a_server_with_no_tools_writes_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """It handshakes fine and is still useless: naming it would add a server
    the model can never call."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("AGENT6_CONFIG_HOME", str(tmp_path / "cfg"))

    rc = cmd_mcp_connect(
        "empty",
        command=_server_argv(tools=False),
        url="",
        token_env="",
        pass_env=[],
        to_repo=False,
    )

    assert rc == 1
    assert load_effective(tmp_path).config.mcp.servers == {}


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"command": [], "url": ""}, "exactly one"),
        ({"command": ["x"], "url": "https://h/mcp"}, "exactly one"),
        ({"command": ["x"], "url": "", "token_env": "T"}, "--pass-env"),
        ({"command": [], "url": "https://h/mcp", "pass_env": ["V"]}, "--token-env"),
    ],
)
def test_a_mismatched_transport_and_env_flag_is_named(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    kwargs: dict[str, object],
    message: str,
) -> None:
    """Each transport owns one env flag, so the wrong pairing is a mistake
    worth naming rather than a setting that silently does nothing."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("AGENT6_CONFIG_HOME", str(tmp_path / "cfg"))
    args: dict[str, object] = {"token_env": "", "pass_env": [], "to_repo": False, **kwargs}

    rc = cmd_mcp_connect("s", **args)  # pyright: ignore[reportArgumentType]

    assert rc == 2
    assert message in capsys.readouterr().err
    assert load_effective(tmp_path).config.mcp.servers == {}


def test_an_argv_round_trips_through_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Written as a TOML array, not a shell string: a string would validate as
    a tuple of characters and the server would never start again."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("AGENT6_CONFIG_HOME", str(tmp_path / "cfg"))
    argv = [*_server_argv(), "--flag=a b", 'quote"inside']

    assert cmd_mcp_connect("q", command=argv, url="", token_env="", pass_env=[], to_repo=False) == 0
    assert load_effective(tmp_path).config.mcp.servers["q"].command == tuple(argv)


def test_the_listing_says_how_each_server_is_reached(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("AGENT6_CONFIG_HOME", str(tmp_path / "cfg"))
    from agent6.paths import global_config_path

    cfg_path = global_config_path()
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text(
        "[mcp.servers.spawned]\ncommand = ['x', '-y']\n"
        "[mcp.servers.dialled]\nurl = 'https://h/mcp'\ntoken_env = 'T'\n",
        encoding="utf-8",
    )

    assert cmd_mcp_list() == 0
    out = capsys.readouterr().out
    assert "spawn   x -y" in out
    assert "connect https://h/mcp" in out
    assert "token from $T" in out
    assert "DISABLED (agent6 config set mcp.enabled true)" in out, (
        "mcp.enabled is off by default and that is worth saying; the servers are global"
    )


def test_the_listing_of_nothing_says_how_to_add_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("AGENT6_CONFIG_HOME", str(tmp_path / "cfg"))
    assert cmd_mcp_list() == 0
    assert "agent6 mcp connect" in capsys.readouterr().out


def test_the_probe_leaves_no_server_running(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """It starts one to ask what it can do, and must not leak it into the
    operator's session."""
    import os
    import subprocess

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("AGENT6_CONFIG_HOME", str(tmp_path / "cfg"))
    # The token rides the -c script, so a leaked server's /proc cmdline would
    # carry it and pgrep -f would find it; pid-suffixed so parallel runs and
    # stale processes cannot collide. The tool it lists proves the probe really
    # spawned this argv (its workspace is read-only, so no marker file can).
    token = f"agent6-leak-probe-{os.getpid()}"
    argv = _server_argv()
    argv[2] = f"# {token}\n" + argv[2]

    assert cmd_mcp_connect("p", command=argv, url="", token_env="", pass_env=[], to_repo=False) == 0
    assert "mcp__p__read_page" in capsys.readouterr().out, "the probe really did start it"
    left = subprocess.run(["pgrep", "-f", token], capture_output=True, check=False)
    assert left.returncode != 0, f"probe server leaked: pids {left.stdout.decode()!r}"


def test_a_passed_secret_is_redacted_from_a_server_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A third-party server may echo a pass_env credential to stderr; that tail
    rides into MCPError and the durable mcp.server_unavailable event, so the
    passed value must be redacted before it leaves the transport."""
    from agent6.tools.mcp_client import MCPError, _MCPServer  # pyright: ignore[reportPrivateUsage]

    srv_py = tmp_path / "srv.py"
    srv_py.write_text(
        "import os, sys\n"
        "sys.stderr.write('boom: ' + os.environ.get('MY_MCP_SECRET', '') + '\\n')\n"
        "sys.stderr.flush()\n"
        # readline: accept the initialize write before dying, so death is
        # detected on the response wait (the path that carries the stderr
        # tail), never as a write-time EPIPE (which carries no server words).
        "sys.stdin.readline()\n"
        "sys.exit(1)\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("MY_MCP_SECRET", "sk-supersecret-123")
    srv = _MCPServer(
        name="leaky",
        command=("/usr/bin/python3", str(srv_py)),
        startup_timeout_s=3.0,
        call_timeout_s=3.0,
        pass_env=("MY_MCP_SECRET",),
        policy=None,
    )
    try:
        with pytest.raises(MCPError) as exc:
            srv.start()
    finally:
        srv.close()
    msg = str(exc.value)
    assert "sk-supersecret-123" not in msg
    assert "<REDACTED>" in msg


def test_direct_config_also_refuses_a_provider_key_in_pass_env() -> None:
    """The invariant lives in Config, not only `mcp connect`: a direct TOML edit
    naming a provider's api_key_env in a server's pass_env is rejected at load,
    so it cannot hand a third-party server a provider API key."""
    from agent6.config import Config

    with pytest.raises(Exception, match="never passes a provider key"):
        Config.model_validate(
            {
                "providers": {"anthropic": {"api_format": "anthropic", "api_key_env": "ANTH_KEY"}},
                "mcp": {"servers": {"s": {"command": ["srv"], "pass_env": ["ANTH_KEY"]}}},
            }
        )


def test_connect_confirms_a_plaintext_nonloopback_token_and_no_is_the_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """url + token_env over plaintext http to a non-loopback host is
    connectable (a VPN or internal endpoint is a real case) but confirmed
    first, naming the cost. Declining (the default) probes nothing and writes
    nothing."""
    from agent6.tools.mcp_client import MCPServerSpec, MCPToolDescriptor
    from agent6.ui.cli import mcp_connect

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("AGENT6_CONFIG_HOME", str(tmp_path / "cfg"))

    class _Tty:
        def isatty(self) -> bool:
            return True

    def _no_probe(spec: MCPServerSpec) -> tuple[tuple[MCPToolDescriptor, ...], str]:
        pytest.fail("declined connect must not probe")

    monkeypatch.setattr(sys, "stdin", _Tty())
    monkeypatch.setattr("builtins.input", lambda _prompt="": "")
    monkeypatch.setattr(mcp_connect, "_probe", _no_probe)
    rc = cmd_mcp_connect(
        "corp",
        command=[],
        url="http://mcp.corp.internal/mcp",
        token_env="TOK",
        pass_env=[],
        to_repo=False,
    )
    assert rc == 1
    err = capsys.readouterr().err
    assert "WARNING" in err and "readable on the network" in err
    assert not (tmp_path / "cfg" / "config.toml").exists()


def test_connect_headless_warns_and_proceeds_on_plaintext_nonloopback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """With no terminal to ask, the connect proceeds with the same loud warning
    (warn, never refuse: the config is explicit)."""
    import io

    from agent6.tools.mcp_client import MCPServerSpec, MCPToolDescriptor
    from agent6.ui.cli import mcp_connect

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("AGENT6_CONFIG_HOME", str(tmp_path / "cfg"))
    monkeypatch.setattr(sys, "stdin", io.StringIO())

    def _fake_probe(spec: MCPServerSpec) -> tuple[tuple[MCPToolDescriptor, ...], str]:
        return (
            (
                MCPToolDescriptor(
                    server_name="corp", tool_name="t", description="", input_schema={}
                ),
            ),
            "",
        )

    monkeypatch.setattr(mcp_connect, "_probe", _fake_probe)
    rc = cmd_mcp_connect(
        "corp",
        command=[],
        url="http://mcp.corp.internal/mcp",
        token_env="TOK",
        pass_env=[],
        to_repo=False,
    )
    assert rc == 0
    err = capsys.readouterr().err
    assert "WARNING" in err and "readable on the network" in err
    written = load_effective(tmp_path).config.mcp.servers["corp"]
    assert written.url == "http://mcp.corp.internal/mcp" and written.token_env == "TOK"


def test_mcp_connect_argv_does_not_clobber_the_dispatch_verb() -> None:
    """The `connect` positional shared its dest with the root subparser's
    command verb, so `mcp connect files -- npx srv` dispatched on a LIST and
    crashed (unhashable dict key) before any of connect's own validation."""
    from agent6.ui.cli.parser import build_parser

    args = build_parser().parse_args(["mcp", "connect", "files", "--", "npx", "-y", "srv"])
    assert args.command == "mcp"
    assert args.server_command == ["npx", "-y", "srv"]


def test_mcp_connect_without_a_transport_refuses_cleanly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """`mcp connect x` reached the dispatch table with args.command rebound to
    an empty list: "unexpected TypeError", a crash log, exit 1."""
    from agent6.ui.cli import cli_main

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("AGENT6_CONFIG_HOME", str(tmp_path / "cfg"))
    monkeypatch.delenv("AGENT6_DEBUG", raising=False)
    assert cli_main(["mcp", "connect", "x"]) == 2
    err = capsys.readouterr().err
    assert "exactly one" in err
    assert "unexpected" not in err


def test_connect_probes_a_spawned_server_under_the_runs_sandbox(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The handshake proves the server the run will spawn: a spawned server is
    probed with the run's jail policy, a --url server with none."""
    from agent6.tools.mcp_client import MCPServerSpec, MCPToolDescriptor
    from agent6.ui.cli import mcp_connect

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("AGENT6_CONFIG_HOME", str(tmp_path / "cfg"))
    seen: list[MCPServerSpec] = []

    def _fake_probe(spec: MCPServerSpec) -> tuple[tuple[MCPToolDescriptor, ...], str]:
        seen.append(spec)
        return (), "did not start"

    monkeypatch.setattr(mcp_connect, "_probe", _fake_probe)
    assert (
        cmd_mcp_connect(
            "calc", command=["python3", "s.py"], url="", token_env="", pass_env=[], to_repo=False
        )
        == 1
    )
    assert (
        cmd_mcp_connect(
            "web",
            command=[],
            url="http://127.0.0.1:1/mcp",
            token_env="",
            pass_env=[],
            to_repo=False,
        )
        == 1
    )
    spawned, connected = seen
    assert spawned.policy is not None and spawned.policy.cwd == tmp_path
    assert connected.policy is None
