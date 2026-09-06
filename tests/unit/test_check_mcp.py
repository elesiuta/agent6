# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""`agent6 check mcp` starts each server as a run does.

A probe in a throwaway directory failed every server whose script lives in
the workspace ("can't open file ... No such file or directory", even by
absolute path under strict), while `mcp connect` and a real run started it.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agent6.config import Config
from agent6.ui.cli import check_cmds

# The interpreter a jailed probe can reach: the run's sandbox grants /usr,
# not the venv.
_JAIL_PYTHON = "/usr/bin/python3"

_SERVER = (
    "import json,sys\n"
    "def w(o): sys.stdout.write(json.dumps(o)+chr(10)); sys.stdout.flush()\n"
    "for line in sys.stdin:\n"
    "    m=json.loads(line)\n"
    "    if m.get('method')=='initialize':\n"
    "        w({'jsonrpc':'2.0','id':m['id'],'result':{'protocolVersion':'2024-11-05',"
    "'capabilities':{},'serverInfo':{'name':'t','version':'1'}}})\n"
    "    elif m.get('method')=='tools/list':\n"
    "        w({'jsonrpc':'2.0','id':m['id'],'result':{'tools':[{'name':'ping',"
    "'inputSchema':{'type':'object'}}]}})\n"
)


@pytest.mark.needs_namespaces
def test_a_server_script_inside_the_workspace_is_checked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The workspace root of the check is the repository, as a run's is: a
    relative script path resolves there on every isolation level."""
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "server.py").write_text(_SERVER, encoding="utf-8")
    monkeypatch.chdir(repo)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "cfg"))
    cfg = Config.model_validate(
        {"mcp": {"enabled": True, "servers": {"inrepo": {"command": [_JAIL_PYTHON, "server.py"]}}}}
    )

    checks = check_cmds._doctor_check_mcp(cfg)  # pyright: ignore[reportPrivateUsage]

    assert [(c.name, c.status) for c in checks] == [("mcp.inrepo", "PASS")], checks
    assert "1 tool," in checks[0].detail
    assert "inrepo: 1 tool," in capsys.readouterr().out
