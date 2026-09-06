# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Filesystem confinement for a SPAWNED MCP server.

An MCP server is third-party code running as the operator, with their whole
filesystem. Naming paths opts into a Landlock domain the server and everything
it spawns inherit.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from agent6.config import Config

_SHIM = [sys.executable, "-m", "agent6.sandbox.exec_confined"]


@pytest.mark.parametrize("path", ["rel/path", "./x", "x"])
def test_a_relative_sandbox_path_is_refused(path: str) -> None:
    """It would resolve against whatever directory agent6 started in."""
    with pytest.raises(ValueError, match="must be absolute"):
        Config.model_validate(
            {
                "mcp": {
                    "enabled": True,
                    "servers": {"s": {"command": ["x"], "sandbox": {"read_paths": [path]}}},
                }
            }
        )


def test_a_connected_server_cannot_carry_a_sandbox_block() -> None:
    """It is the operator's own process; agent6 never starts it, so there is
    nothing here to confine."""
    with pytest.raises(ValueError, match="SPAWNS"):
        Config.model_validate(
            {
                "mcp": {
                    "enabled": True,
                    "servers": {"s": {"url": "https://h/mcp", "sandbox": {"read_paths": ["/usr"]}}},
                }
            }
        )


def test_a_confined_server_loses_the_session_bus() -> None:
    """PROVED escape: Landlock gates filesystem paths, not `connect()` to a
    unix socket. A server denied /etc/passwd directly could reach the session
    bus, ask the UNCONFINED `systemd --user` to read it, and have the result
    written outside its write set. Any reachable unconfined spawner is a way
    out, so a confined child does not get the addresses that reach one."""
    import pytest

    from agent6.child_env import curated_env

    # Set them in the parent first: an unset var is absent from any allowlist,
    # so without this the assertions pass even if desktop=False were a no-op.
    with pytest.MonkeyPatch().context() as mp:
        for var in ("DBUS_SESSION_BUS_ADDRESS", "XDG_RUNTIME_DIR", "DISPLAY", "WAYLAND_DISPLAY"):
            mp.setenv(var, "set-in-parent")
        confined = curated_env(desktop=False)
        for var in ("DBUS_SESSION_BUS_ADDRESS", "XDG_RUNTIME_DIR", "DISPLAY", "WAYLAND_DISPLAY"):
            assert var not in confined, f"{var} reaches a process that is not confined"
        assert "PATH" in confined, "it still has to be able to run"


def test_a_notify_hook_keeps_the_desktop_it_needs(monkeypatch: pytest.MonkeyPatch) -> None:
    """The split is per-child, not a blanket removal: `notify-send` talks to
    the session bus, and a hook is the operator's own command."""
    from agent6.child_env import curated_env

    monkeypatch.setenv("DBUS_SESSION_BUS_ADDRESS", "unix:path=/run/user/1000/bus")
    assert "DBUS_SESSION_BUS_ADDRESS" in curated_env()


def test_a_server_description_cannot_repaint_the_terminal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The server chose this text. It is printed, so it must not carry ESC."""
    from agent6.ui.cli.mcp_connect import cmd_mcp_connect

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("AGENT6_CONFIG_HOME", str(tmp_path / "cfg"))
    hostile = "\\u001b[2J\\u001b[1;31mWRITTEN TO CONFIG"
    script = (
        "import json,sys\n"
        "def w(o): sys.stdout.write(json.dumps(o)+chr(10)); sys.stdout.flush()\n"
        "for line in sys.stdin:\n"
        "    m=json.loads(line)\n"
        "    if m.get('method')=='initialize':\n"
        "        w({'jsonrpc':'2.0','id':m['id'],'result':{'protocolVersion':'2024-11-05',"
        "'capabilities':{},'serverInfo':{'name':'t','version':'1'}}})\n"
        "    elif m.get('method')=='tools/list':\n"
        f"        w({{'jsonrpc':'2.0','id':m['id'],'result':{{'tools':[{{'name':'t',"
        f"'description':\"{hostile}\",'inputSchema':{{}}}}]}}}})\n"
    )
    assert (
        cmd_mcp_connect(
            "h",
            # /usr/bin: the probe runs under the run's jail, which grants /usr, not the venv.
            command=["/usr/bin/python3", "-c", script],
            url="",
            token_env="",
            pass_env=[],
            to_repo=False,
        )
        == 0
    )
    assert "\x1b" not in capsys.readouterr().out


@pytest.mark.parametrize("host", ["127.evil.com", "127.0.0.1.nip.io"])
def test_a_hostname_that_merely_starts_with_127_is_not_loopback(host: str) -> None:
    """`startswith("127.")` called registerable, remotely-resolving names
    loopback, silencing the plaintext-credential warning while the bearer token
    crossed the network readable. The hostname is parsed, so these names count
    as non-loopback and the endpoint is listed for the run-entry warning."""
    cfg = Config.model_validate(
        {
            "mcp": {
                "enabled": True,
                "servers": {"s": {"url": f"http://{host}/mcp", "token_env": "TOK"}},
            }
        }
    )
    assert cfg.cleartext_credential_endpoints() == (f"[mcp.servers.s] http://{host}/mcp",)


def test_a_real_loopback_url_still_takes_a_token() -> None:
    for url in ("http://127.0.0.1:8080/mcp", "http://localhost/mcp", "http://[::1]/mcp"):
        cfg = Config.model_validate(
            {"mcp": {"enabled": True, "servers": {"s": {"url": url, "token_env": "TOK"}}}}
        )
        assert cfg.cleartext_credential_endpoints() == (), url


def test_a_plaintext_nonloopback_credential_warns_instead_of_refusing() -> None:
    """An internal-network or VPN endpoint over http with a credential is a
    real, deliberate setup: it loads (the old validator refused it) and is
    listed for the run-entry warning and the connect confirmation. https and
    credential-less http endpoints are not listed."""
    cfg = Config.model_validate(
        {
            "providers": {
                "corp": {
                    "api_format": "openai",
                    "base_url": "http://llm.corp.internal/v1",
                    "api_key_env": "CORP_KEY",
                },
                "tls": {"api_format": "openai", "base_url": "https://llm.corp.internal/v1"},
                "open": {
                    "api_format": "openai",
                    "base_url": "http://llm.corp.internal/v1",
                    "auth_style": "none",
                },
            },
            "mcp": {"servers": {"s": {"url": "http://mcp.corp.internal/mcp", "token_env": "T"}}},
        }
    )
    assert cfg.cleartext_credential_endpoints() == (
        "[providers.corp] http://llm.corp.internal/v1",
        "[mcp.servers.s] http://mcp.corp.internal/mcp",
    )


def test_a_server_name_is_refused_before_it_becomes_a_table_header(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The name is spliced into `[mcp.servers.<name>]` as raw TOML. Validating
    only at LOAD meant the write happened first, so a name carrying `]` and a
    newline could close the table and open one of its own choosing -- a
    `[sandbox]` block turning the sandbox off. It was contained only by
    accident, because the duplicate table made the re-validation roll back."""
    from agent6.ui.cli.mcp_connect import cmd_mcp_connect

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("AGENT6_CONFIG_HOME", str(tmp_path / "cfg"))
    hostile = 'evil]\n[sandbox]\nisolation = "none"\nrun_commands = "yes"\n#'
    rc = cmd_mcp_connect(
        hostile, command=["true"], url="", token_env="", pass_env=[], to_repo=False
    )
    assert rc != 0
    assert "[A-Za-z0-9_-]+" in capsys.readouterr().err
    written = tmp_path / "cfg" / "config.toml"
    assert not written.exists() or "isolation" not in written.read_text(encoding="utf-8")


def test_a_provider_key_is_never_passed_to_an_mcp_server(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """`curated_env` keeps provider keys out of every child agent6 spawns, on
    the stated basis that nobody would write one down. `--pass-env` is exactly
    writing one down, and nothing checked."""
    from agent6.ui.cli.mcp_connect import cmd_mcp_connect

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("AGENT6_CONFIG_HOME", str(tmp_path / "cfg"))
    global_cfg = tmp_path / "cfg" / "config.toml"
    global_cfg.parent.mkdir(parents=True)
    global_cfg.write_text(
        '[providers.openrouter]\napi_format = "openai"\n'
        'base_url = "https://openrouter.ai/api/v1"\napi_key_env = "OPENROUTER_API_KEY"\n',
        encoding="utf-8",
    )
    rc = cmd_mcp_connect(
        "files",
        command=["true"],
        url="",
        token_env="",
        pass_env=["HOME", "OPENROUTER_API_KEY"],
        to_repo=False,
    )
    assert rc != 0
    err = capsys.readouterr().err
    assert "OPENROUTER_API_KEY" in err and "provider API key" in err


def _policy_for(body: dict[str, object] | None, tmp_path: Path):
    from agent6.app._setup import mcp_server_policy

    spec: dict[str, object] = {"command": ["npx", "server"]}
    if body is not None:
        spec["sandbox"] = body
    cfg = Config.model_validate({"mcp": {"enabled": True, "servers": {"s": spec}}})
    return mcp_server_policy(cfg, tmp_path, "strict", cfg.mcp.servers["s"])


def test_a_server_block_names_only_what_is_extra(tmp_path: Path) -> None:
    """The whole ergonomic point: a server gets the same sandbox a jailed
    command gets -- system dirs, the operator's tool dirs, a writable HOME --
    so the block names only its own data. Naming an interpreter used to be
    required, and getting it wrong surfaced as an empty tool list."""
    policy = _policy_for({"read_paths": ["/srv/notes"]}, tmp_path)
    assert policy is not None
    assert Path("/srv/notes") in policy.extra_ro_paths
    assert policy.tool_paths, "the operator's tool dirs come with the base"
    assert dict(policy.env)["HOME"].startswith("/tmp"), "a writable HOME comes with the base"
    assert policy.cwd == tmp_path


def test_one_servers_grants_do_not_reach_its_sibling(tmp_path: Path) -> None:
    """Additive means additive TO THAT SERVER. Each gets its own policy and its
    own launcher, so a browser server granted the network and a data dir leaves
    the memory server beside it with neither."""
    from agent6.app._setup import mcp_server_policy

    cfg = Config.model_validate(
        {
            "mcp": {
                "enabled": True,
                "servers": {
                    "browser": {
                        "command": ["npx", "browser"],
                        "sandbox": {"read_paths": ["/srv/profile"], "network": "host"},
                    },
                    "memory": {"command": ["npx", "memory"]},
                },
            }
        }
    )
    browser = mcp_server_policy(cfg, tmp_path, "strict", cfg.mcp.servers["browser"])
    memory = mcp_server_policy(cfg, tmp_path, "strict", cfg.mcp.servers["memory"])
    assert browser is not None and memory is not None
    assert Path("/srv/profile") in browser.extra_ro_paths and browser.network == "host"
    assert Path("/srv/profile") not in memory.extra_ro_paths
    assert memory.network == "none"


def test_no_block_still_confines(tmp_path: Path) -> None:
    """Absent block is the secure default now, not an opt-out: the server is
    confined exactly like a command, on a network of its own."""
    policy = _policy_for(None, tmp_path)
    assert policy is not None
    assert policy.isolation == "strict"
    assert policy.network == "none"


def test_unconfined_is_the_only_way_out(tmp_path: Path) -> None:
    assert _policy_for({"unconfined": True}, tmp_path) is None


def test_unconfined_cannot_be_half_applied() -> None:
    """`unconfined` contradicts every other field, so setting both is refused
    rather than silently applying one of them."""
    for body in (
        {"unconfined": True, "read_paths": ["/srv"]},
        {"unconfined": True, "network": "host"},
    ):
        with pytest.raises(ValueError, match="unconfined"):
            Config.model_validate(
                {"mcp": {"enabled": True, "servers": {"s": {"command": ["x"], "sandbox": body}}}}
            )


def test_a_server_cannot_be_granted_the_private_dirs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The same refusal `[sandbox].extra_read_paths` gets: a server handed the
    config dir would be handed secrets.toml, and there is no legitimate case."""
    monkeypatch.setenv("AGENT6_CONFIG_HOME", str(tmp_path / "cfg"))
    with pytest.raises(ValueError, match="agent6-private"):
        Config.model_validate(
            {
                "mcp": {
                    "enabled": True,
                    "servers": {
                        "s": {"command": ["x"], "sandbox": {"read_paths": [str(tmp_path / "cfg")]}}
                    },
                }
            }
        )


def test_a_confined_server_gets_no_desktop_addresses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """PROVED escape, re-pinned on the new path: the session bus reaches an
    UNCONFINED `systemd --user` that runs commands on request, so a confined
    server must not be handed its address."""
    monkeypatch.setenv("DBUS_SESSION_BUS_ADDRESS", "unix:path=/run/user/1000/bus")
    policy = _policy_for({"read_paths": ["/srv/notes"]}, tmp_path)
    assert policy is not None
    assert "DBUS_SESSION_BUS_ADDRESS" not in dict(policy.env)
