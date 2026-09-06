# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""A diagnostic never starts a server outside the confinement a run gives it
and never writes anything but the config it was asked to write.

`agent6 check mcp` and the `mcp connect` handshake start each server in the
repository (so a script that lives there resolves, as it does in a run) under
the run's sandbox with the workspace bound read-only. A server a run would
refuse is a FAIL row, never started. A server the read-only probe cannot hold
(`unconfined = true`, any write grant, no jail at all) is a WARN row naming
the leaf, never started; with no jail `mcp connect` writes the entry unproved
and says so.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agent6.config import Config
from agent6.config.layer import load_effective
from agent6.ui.cli import check_cmds
from agent6.ui.cli.mcp_connect import cmd_mcp_connect

# The interpreter a jailed probe can reach: the run's sandbox grants /usr,
# not the venv.
_JAIL_PYTHON = "/usr/bin/python3"

# A well-behaved (not hostile) stdio MCP server that drops a file in its cwd on
# `initialize` -- a cache, say -- then answers initialize + tools/list. It
# tolerates a refused write, as a cache writer does.
_WRITER_SERVER = (
    "import json,sys,pathlib\n"
    "try:\n"
    "    pathlib.Path('wrote-during-check.txt').write_text('x', encoding='utf-8')\n"
    "except OSError:\n"
    "    pass\n"
    "def w(o):\n"
    "    sys.stdout.write(json.dumps(o)+chr(10)); sys.stdout.flush()\n"
    "for line in sys.stdin:\n"
    "    m=json.loads(line); mid=m.get('id')\n"
    "    if m.get('method')=='initialize':\n"
    "        w({'jsonrpc':'2.0','id':mid,'result':{'protocolVersion':'2024-11-05',"
    "'capabilities':{},'serverInfo':{'name':'t','version':'1'}}})\n"
    "    elif m.get('method')=='tools/list':\n"
    "        w({'jsonrpc':'2.0','id':mid,'result':{'tools':[{'name':'ping','inputSchema':{}}]}})\n"
)


def _repo_with_server(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "server.py").write_text(_WRITER_SERVER, encoding="utf-8")
    monkeypatch.chdir(repo)
    monkeypatch.setenv("AGENT6_CONFIG_HOME", str(tmp_path / "cfg"))
    return repo


def _force(monkeypatch: pytest.MonkeyPatch, isolation: str) -> None:
    """Resolve to *isolation* whatever the host offers (the jail, when one is
    started, still runs at that level for real)."""
    monkeypatch.setattr(check_cmds, "detect_env", object)

    def _select(_req: str, _env: object) -> str:
        return isolation

    monkeypatch.setattr(check_cmds, "resolve_isolation", _select)


def _check_mcp(
    servers: dict[str, object], *, sandbox: dict[str, object] | None = None
) -> list[check_cmds._DoctorCheck]:  # pyright: ignore[reportPrivateUsage]
    cfg = Config.model_validate(
        {"sandbox": sandbox or {}, "mcp": {"enabled": True, "servers": servers}}
    )
    return check_cmds._doctor_check_mcp(cfg)  # pyright: ignore[reportPrivateUsage]


def _never_started(monkeypatch: pytest.MonkeyPatch) -> None:
    def _boom(*_a: object, **_k: object) -> None:
        raise AssertionError("the check started a server it must not")

    monkeypatch.setattr(check_cmds.MCPManager, "start", _boom)


@pytest.mark.needs_namespaces
def test_a_startup_write_never_lands_in_the_repo(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The server runs in the repository (its script resolves there) and is
    verified, and the file it writes on startup is refused: the probe's
    workspace is read-only."""
    repo = _repo_with_server(tmp_path, monkeypatch)
    checks = _check_mcp({"notes": {"command": [_JAIL_PYTHON, "server.py"]}})
    assert [(c.name, c.status) for c in checks] == [("mcp.notes", "PASS")], checks
    assert "1 tool," in checks[0].detail
    assert sorted(p.name for p in repo.iterdir()) == ["server.py"], (
        "check wrote into the repo through MCP startup"
    )


@pytest.mark.needs_namespaces
def test_the_workspace_is_read_only_under_hardened_too(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Hardened has no mount namespace to re-bind with; the Landlock carve-out
    (read on cwd, no write grant beneath it) holds the same line."""
    repo = _repo_with_server(tmp_path, monkeypatch)
    _force(monkeypatch, "hardened")
    checks = _check_mcp(
        {"notes": {"command": [_JAIL_PYTHON, "server.py"]}}, sandbox={"network": "host"}
    )
    assert [(c.name, c.status) for c in checks] == [("mcp.notes", "PASS")], checks
    assert "network: host" in checks[0].detail
    assert sorted(p.name for p in repo.iterdir()) == ["server.py"]


@pytest.mark.needs_namespaces
def test_mcp_connect_probes_read_only_too(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The handshake is proved, the repo stays as it was."""
    repo = _repo_with_server(tmp_path, monkeypatch)
    rc = cmd_mcp_connect(
        "notes",
        command=[_JAIL_PYTHON, "server.py"],
        url="",
        token_env="",
        pass_env=[],
        to_repo=False,
    )
    assert rc == 0, capsys.readouterr().err
    assert sorted(p.name for p in repo.iterdir()) == ["server.py"]


def test_mcp_connect_with_no_jail_writes_the_entry_unproved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """With no jail the read-only bind is inert, so the probe would run the
    server unconfined in the repository: it is not run. The entry is written,
    and the operator is told what was not proved."""
    repo = _repo_with_server(tmp_path, monkeypatch)
    monkeypatch.setenv("AGENT6_DANGEROUSLY_DISABLE_SANDBOX", "1")
    rc = cmd_mcp_connect(
        "notes",
        command=[_JAIL_PYTHON, "server.py"],
        url="",
        token_env="",
        pass_env=[],
        to_repo=False,
    )
    assert rc == 0
    out, err = capsys.readouterr()
    assert (
        "WARNING: notes not probed: no jail (AGENT6_DANGEROUSLY_DISABLE_SANDBOX=1 is set);"
        " a run starts it unconfined." in err
    )
    assert "written to the global config" in out and "mcp__notes__" not in out
    assert load_effective(repo).config.mcp.servers["notes"].command == (_JAIL_PYTHON, "server.py")
    assert sorted(p.name for p in repo.iterdir()) == ["server.py"], (
        "connect wrote into the repo through MCP startup"
    )


def test_a_server_it_cannot_hold_read_only_is_reported_not_started(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Each row names the leaf that makes the server unprobeable and says a
    run starts it as configured; nothing is spawned."""
    repo = _repo_with_server(tmp_path, monkeypatch)
    _force(monkeypatch, "strict")
    _never_started(monkeypatch)
    checks = _check_mcp(
        {
            "unconf": {
                "command": [_JAIL_PYTHON, "server.py"],
                "sandbox": {"unconfined": True},
            },
            "writer": {
                "command": [_JAIL_PYTHON, "server.py"],
                "sandbox": {"write_paths": ["/srv/data"]},
            },
        },
        sandbox={"network": "host"},  # no session network to open on a host without one
    )
    by = {c.name: c for c in checks}
    assert by["mcp.unconf"].status == "WARN"
    assert by["mcp.unconf"].detail == (
        "not probed: mcp.servers.unconf.sandbox.unconfined = true; a run starts it as"
        " configured, unconfined"
    )
    assert by["mcp.writer"].status == "WARN"
    assert by["mcp.writer"].detail == (
        "not probed: mcp.servers.writer.sandbox.write_paths grants writes (/srv/data); a run"
        " starts it as configured"
    )
    assert "[WARN] mcp.unconf" in capsys.readouterr().out
    assert sorted(p.name for p in repo.iterdir()) == ["server.py"]


@pytest.mark.parametrize(
    ("leaf", "value"),
    [("extra_write_paths", ["/srv/out"]), ("extra_device_paths", ["/dev/null"])],
)
def test_an_operator_write_grant_holds_the_probe_off_too(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, leaf: str, value: list[str]
) -> None:
    """A `[sandbox]` grant stays writable under the read-only root, so a
    server under one is not probed either: the rule stays absolute."""
    _repo_with_server(tmp_path, monkeypatch)
    _force(monkeypatch, "strict")
    _never_started(monkeypatch)
    checks = _check_mcp(
        {"notes": {"command": [_JAIL_PYTHON, "server.py"]}},
        sandbox={"network": "host", leaf: value},
    )
    assert [(c.name, c.status) for c in checks] == [("mcp.notes", "WARN")], checks
    assert checks[0].detail == (
        f"not probed: sandbox.{leaf} grants writes ({value[0]}); a run starts it as configured"
    )


def test_a_server_a_run_would_refuse_fails_the_check_unstarted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """`network = "none"` on a hardened host is a run refusal; the check
    applies it first rather than starting the server on the host network and
    reporting PASS."""
    _repo_with_server(tmp_path, monkeypatch)
    _force(monkeypatch, "hardened")
    _never_started(monkeypatch)
    checks = _check_mcp(
        {"quiet": {"command": [_JAIL_PYTHON, "server.py"], "sandbox": {"network": "none"}}},
        sandbox={"network": "host"},
    )
    assert [(c.name, c.status) for c in checks] == [("mcp.quiet", "FAIL")], checks
    assert checks[0].detail.startswith("a run would refuse: MCP server 'quiet' sets")
    assert "network: host" not in checks[0].detail
    assert "[FAIL] mcp.quiet: a run would refuse" in capsys.readouterr().out


@pytest.mark.parametrize(
    ("how", "cause"),
    [
        (
            {"env": "AGENT6_DANGEROUSLY_DISABLE_SANDBOX"},
            "AGENT6_DANGEROUSLY_DISABLE_SANDBOX=1 is set",
        ),
        ({"leaf": "none"}, "sandbox.isolation = none"),
    ],
)
def test_no_jail_means_no_probe_and_names_why(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, how: dict[str, str], cause: str
) -> None:
    """With no jail nothing keeps a startup write off the repo, so the check
    starts no spawned server; the row names what took the jail away."""
    repo = _repo_with_server(tmp_path, monkeypatch)
    _never_started(monkeypatch)
    if "env" in how:
        monkeypatch.setenv(how["env"], "1")
    checks = _check_mcp(
        {"notes": {"command": [_JAIL_PYTHON, "server.py"]}},
        sandbox={"isolation": how["leaf"]} if "leaf" in how else None,
    )
    assert [(c.name, c.status) for c in checks] == [("mcp.notes", "WARN")], checks
    assert checks[0].detail == f"not probed: no jail ({cause}); a run starts it unconfined"
    assert sorted(p.name for p in repo.iterdir()) == ["server.py"]
