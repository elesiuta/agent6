# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The in-process file boundary (`Workspace`).

`sandbox.hide_paths` was wired only into the jail policy, so the tools -- which
run IN-PROCESS, outside the jail, and ask no approval -- bypassed it: `read_file`
returned a hidden secret, `list_dir` showed it, and `apply_edit` WROTE into one.
With a workspace root containing agent6's own config dir (root=$HOME) that
extended to `secrets.toml` and to `config.toml`, whose next load sets isolation
and run_commands -- cross-run loosening by persistence.

The tools are the front door of the file axis and the jail is the fence, so the
boundary is derived from config VALUES and holds at EVERY isolation level: a
degradation (auto falling back, macOS having no jail) must never widen it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from agent6.config import Config
from agent6.tools.dispatch import ToolDispatcher
from agent6.tools.errors import ToolError

_SECRET = "AWS_SECRET_ACCESS_KEY=leaked-xyz"


def _dispatcher(root: Path, cfg: Config, isolation: str = "none") -> ToolDispatcher:
    return ToolDispatcher(root=root, config=cfg, isolation=isolation)  # pyright: ignore[reportArgumentType]


def _hiding(root: Path, *rel: str) -> Config:
    return Config.model_validate({"sandbox": {"hide_paths": [str(root / r) for r in rel]}})


def _dispatch(root: Path, cfg: Config, tool: str, args: dict[str, Any]) -> dict[str, Any]:
    d = _dispatcher(root, cfg)
    try:
        return d.dispatch(tool, args).to_wire()
    finally:
        d.close()


def test_read_file_refuses_a_hidden_path(tmp_path: Path) -> None:
    (tmp_path / ".env").write_text(_SECRET, encoding="utf-8")
    with pytest.raises(ToolError, match="hidden from this run"):
        _dispatch(tmp_path, _hiding(tmp_path, ".env"), "read_file", {"path": ".env"})


def test_list_dir_hides_the_entry_but_says_how_many(tmp_path: Path) -> None:
    """Filtered, not named: the listing stays true ("something is hidden")
    without disclosing what, and the model stops probing."""
    (tmp_path / ".env").write_text(_SECRET, encoding="utf-8")
    (tmp_path / "main.py").write_text("x = 1\n", encoding="utf-8")
    out = _dispatch(tmp_path, _hiding(tmp_path, ".env"), "list_dir", {"path": "."})
    assert out["entries"] == ["main.py"]
    assert out["hidden"] == 1


def test_list_dir_omits_the_count_when_nothing_is_hidden(tmp_path: Path) -> None:
    (tmp_path / "main.py").write_text("x = 1\n", encoding="utf-8")
    out = _dispatch(tmp_path, Config(), "list_dir", {"path": "."})
    assert out["entries"] == ["main.py"]
    assert "hidden" not in out


def test_apply_edit_refuses_to_write_a_hidden_path(tmp_path: Path) -> None:
    """The write half: refusing the read while allowing the write would leave
    the model able to plant content in a path the operator hid."""
    secret = tmp_path / ".env"
    secret.write_text(_SECRET, encoding="utf-8")
    with pytest.raises(ToolError, match="hidden from this run"):
        _dispatch(
            tmp_path,
            _hiding(tmp_path, ".env"),
            "apply_edit",
            {"path": ".env", "edits": [{"old_string": "leaked-xyz", "new_string": "PWNED"}]},
        )
    assert secret.read_text(encoding="utf-8") == _SECRET


def test_a_normal_path_is_untouched(tmp_path: Path) -> None:
    (tmp_path / "main.py").write_text("x = 1\n", encoding="utf-8")
    out = _dispatch(tmp_path, _hiding(tmp_path, ".env"), "read_file", {"path": "main.py"})
    assert out["content"] == "x = 1\n"


def test_a_hidden_file_never_reaches_the_symbol_index(tmp_path: Path) -> None:
    """find_definition would otherwise leak the symbol NAMES and line numbers
    of a file nothing is allowed to read."""
    (tmp_path / "secrets.py").write_text("def leaked_symbol():\n    pass\n", encoding="utf-8")
    (tmp_path / "main.py").write_text("def public_symbol():\n    pass\n", encoding="utf-8")
    cfg = _hiding(tmp_path, "secrets.py")
    defs = _dispatch(tmp_path, cfg, "find_definition", {"symbol": "leaked_symbol"})
    assert defs["definitions"] == []
    ok = _dispatch(tmp_path, cfg, "find_definition", {"symbol": "public_symbol"})
    assert [d["path"] for d in ok["definitions"]] == ["main.py"]


@pytest.mark.parametrize("isolation", ["strict", "hardened", "none"])
def test_the_boundary_holds_at_every_isolation_level(tmp_path: Path, isolation: str) -> None:
    """The rule: config VALUES define the boundary, never the isolation level.
    `none` has no jail at all (and is what macOS resolves to), so a boundary
    that tracked the level would vanish exactly where it is the only one left.
    """
    (tmp_path / ".env").write_text(_SECRET, encoding="utf-8")
    d = _dispatcher(tmp_path, _hiding(tmp_path, ".env"), isolation)
    try:
        with pytest.raises(ToolError, match="hidden from this run"):
            d.dispatch("read_file", {"path": ".env"})
    finally:
        d.close()


def test_agent6s_own_secrets_are_denied_when_the_workspace_contains_them(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A workspace root that CONTAINS the config dir (root=$HOME) put
    `secrets.toml` -- provider keys -- inside the tree the tools may reach, with
    no hide_paths entry naming it. The builtin private dirs are denied too."""
    root = tmp_path / "home"
    root.mkdir()
    # The same override the suite's own isolation uses; it outranks XDG.
    monkeypatch.setenv("AGENT6_CONFIG_HOME", str(root / ".config" / "agent6"))
    monkeypatch.setenv("AGENT6_STATE_HOME", str(root / ".local" / "state" / "agent6"))
    cfg_dir = root / ".config" / "agent6"
    cfg_dir.mkdir(parents=True)
    (cfg_dir / "secrets.toml").write_text('[fake]\nKEY="fake-not-real"\n', encoding="utf-8")

    with pytest.raises(ToolError, match="hidden from this run"):
        _dispatch(root, Config(), "read_file", {"path": ".config/agent6/secrets.toml"})


def test_the_config_a_later_run_loads_cannot_be_written(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Persistence, not just disclosure: editing `~/.config/agent6/config.toml`
    sets `isolation` / `run_commands` for the NEXT run."""
    root = tmp_path / "home"
    root.mkdir()
    # The same override the suite's own isolation uses; it outranks XDG.
    monkeypatch.setenv("AGENT6_CONFIG_HOME", str(root / ".config" / "agent6"))
    monkeypatch.setenv("AGENT6_STATE_HOME", str(root / ".local" / "state" / "agent6"))
    cfg_dir = root / ".config" / "agent6"
    cfg_dir.mkdir(parents=True)
    conf = cfg_dir / "config.toml"
    conf.write_text('[sandbox]\nisolation = "strict"\n', encoding="utf-8")

    with pytest.raises(ToolError, match="hidden from this run"):
        _dispatch(
            root,
            Config(),
            "apply_edit",
            {
                "path": ".config/agent6/config.toml",
                "edits": [{"old_string": '"strict"', "new_string": '"none"'}],
            },
        )
    assert 'isolation = "strict"' in conf.read_text(encoding="utf-8")


def test_a_workspace_inside_a_private_dir_refuses_at_preflight(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every tool call would refuse, so the run is told why up front instead of
    failing on every path. One exactly-known case, not an enumeration."""
    from agent6.app.confine import check_workspace_outside_private_dirs

    monkeypatch.setenv("AGENT6_CONFIG_HOME", str(tmp_path / "cfg" / "agent6"))
    monkeypatch.setenv("AGENT6_STATE_HOME", str(tmp_path / "state" / "agent6"))
    inside = tmp_path / "state" / "agent6" / "somerepo"
    inside.mkdir(parents=True)
    refusal = check_workspace_outside_private_dirs(inside)
    assert refusal is not None and "private" in refusal

    ordinary = tmp_path / "project"
    ordinary.mkdir()
    assert check_workspace_outside_private_dirs(ordinary) is None


def test_a_state_dir_inside_the_workspace_refuses_at_preflight(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A relocated `[agent6].state_dir` INSIDE the workspace exposes transcripts
    and keys to jailed commands and stages them into commits, so preflight
    refuses the overlap in this direction too (masking alone does not stop the
    auto-commit from staging them)."""
    from agent6.app.confine import check_workspace_outside_private_dirs

    workspace = tmp_path / "project"
    workspace.mkdir()
    cfg_home = tmp_path / "cfg"
    cfg_home.mkdir()
    a6state = workspace / ".a6state"
    (cfg_home / "config.toml").write_text(f'[agent6]\nstate_dir = "{a6state}"\n', encoding="utf-8")
    monkeypatch.setenv("AGENT6_CONFIG_HOME", str(cfg_home))

    refusal = check_workspace_outside_private_dirs(workspace)
    assert refusal is not None and "inside the workspace" in refusal


# --- operator grants ---------------------------------------------------------


def _granting(read: Path | None = None, write: Path | None = None) -> Config:
    sb: dict[str, Any] = {}
    if read is not None:
        sb["extra_read_paths"] = [str(read)]
    if write is not None:
        sb["extra_write_paths"] = [str(write)]
    return Config.model_validate({"sandbox": sb})


def test_an_absolute_path_inside_a_grant_is_readable(tmp_path: Path) -> None:
    """The tools reach the trees the jail mounts for commands. An absolute path
    is the only way to name one, so grants would otherwise be unreachable."""
    root = tmp_path / "repo"
    root.mkdir()
    sdk = tmp_path / "sdk"
    sdk.mkdir()
    (sdk / "h.h").write_text("granted header\n", encoding="utf-8")
    out = _dispatch(root, _granting(read=sdk), "read_file", {"path": str(sdk / "h.h")})
    assert out["content"] == "granted header\n"


@pytest.mark.parametrize("target", ["ungranted", "/etc/passwd"])
def test_an_absolute_path_outside_every_grant_is_refused(tmp_path: Path, target: str) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    sdk = tmp_path / "sdk"
    sdk.mkdir()
    outside = tmp_path / "ungranted"
    outside.mkdir()
    (outside / "secret.txt").write_text("NOT granted\n", encoding="utf-8")
    path = "/etc/passwd" if target.startswith("/") else str(outside / "secret.txt")
    with pytest.raises(ToolError, match="Absolute"):
        _dispatch(root, _granting(read=sdk), "read_file", {"path": path})


def test_a_read_grant_is_not_writable(tmp_path: Path) -> None:
    """`extra_read_paths` grants reading only, exactly as the jail mounts it."""
    root = tmp_path / "repo"
    root.mkdir()
    sdk = tmp_path / "sdk"
    sdk.mkdir()
    target = sdk / "h.h"
    target.write_text("granted header\n", encoding="utf-8")
    with pytest.raises(ToolError, match="Absolute"):
        _dispatch(
            root,
            _granting(read=sdk),
            "apply_edit",
            {"path": str(target), "edits": [{"old_string": "granted", "new_string": "PWNED"}]},
        )
    assert target.read_text(encoding="utf-8") == "granted header\n"


def test_a_write_grant_is_writable_and_readable(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    out_dir = tmp_path / "artifacts"
    out_dir.mkdir()
    target = out_dir / "report.txt"
    target.write_text("before\n", encoding="utf-8")
    cfg = _granting(write=out_dir)
    _dispatch(
        root,
        cfg,
        "apply_edit",
        {"path": str(target), "edits": [{"old_string": "before", "new_string": "after"}]},
    )
    assert target.read_text(encoding="utf-8") == "after\n"
    assert _dispatch(root, cfg, "read_file", {"path": str(target)})["content"] == "after\n"


def test_denied_beats_a_grant(tmp_path: Path) -> None:
    """A hide inside a granted region wins: the same precedence the jail uses
    when it masks a hidden path out of a broader mount."""
    root = tmp_path / "repo"
    root.mkdir()
    granted = tmp_path / "granted"
    (granted / "keys").mkdir(parents=True)
    (granted / "keys" / "id_rsa").write_text("PRIVATE\n", encoding="utf-8")
    cfg = Config.model_validate(
        {
            "sandbox": {
                "extra_read_paths": [str(granted)],
                "hide_paths": [str(granted / "keys")],
            }
        }
    )
    with pytest.raises(ToolError, match="hidden from this run"):
        _dispatch(root, cfg, "read_file", {"path": str(granted / "keys" / "id_rsa")})


def _hide_on_hardened(root: Path, path: str, extra: dict[str, object] | None = None) -> str | None:
    from agent6.app.confine import check_hide_paths_support

    data: dict[str, object] = {"sandbox": {"hide_paths": [path], **(extra or {})}}
    return check_hide_paths_support(Config.model_validate(data), "hardened", root)


def test_hide_paths_refuses_the_launcher_grant_regions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The preflight built its region set from cwd + the extra grants only,
    while the launcher also grants /tmp, the system roots, and the operator
    tool dirs -- a hide_paths entry under those passed preflight and stayed
    readable (a silently-inert explicit setting)."""
    monkeypatch.chdir(tmp_path)
    tool_dir = Path("/nonexistent-tools/bin")
    monkeypatch.setattr(
        "agent6.tools.policy.operator_tool_paths", lambda: ("/usr/bin:/bin", (tool_dir,))
    )
    r = _hide_on_hardened(tmp_path, "/tmp/secret-cache")
    assert r is not None and "/tmp" in r
    r = _hide_on_hardened(tmp_path, "/etc/agent6-private")
    assert r is not None and "system dir" in r
    r = _hide_on_hardened(tmp_path, str(tool_dir / "sub"))
    assert r is not None and "tool dir" in r
    assert _hide_on_hardened(tmp_path, "/nonexistent-elsewhere/private") is None


def test_hide_paths_resolves_aliases_and_refuses_inner_grants(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A literal `..` refuses upstream at config validation; a symlink alias
    resolves before containment; a grant INSIDE the hidden tree exposes part
    of it and refuses too."""
    monkeypatch.chdir(tmp_path)
    with pytest.raises(Exception, match="'\\.\\.'"):
        _hide_on_hardened(tmp_path, "/opt/../etc/shadow-file")
    link = tmp_path / "alias"
    link.symlink_to("/etc")
    assert _hide_on_hardened(tmp_path, str(link / "agent6-private")) is not None
    hidden_tree = Path("/nonexistent-vault")
    r = _hide_on_hardened(
        tmp_path, str(hidden_tree), {"extra_read_paths": [str(hidden_tree / "inner")]}
    )
    assert r is not None and "extra_read_paths" in r


def test_hide_paths_refuses_an_mcp_server_grant(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Each enabled server's own policy regions join the preflight set."""
    from agent6.app.confine import check_hide_paths_support

    monkeypatch.chdir(tmp_path)
    mcp_dir = Path("/nonexistent-mcp-data")
    cfg = Config.model_validate(
        {
            "sandbox": {"hide_paths": [str(mcp_dir / "creds")]},
            "mcp": {
                "enabled": True,
                "servers": {
                    "srv": {
                        "command": ["fake-server"],
                        "sandbox": {"read_paths": [str(mcp_dir)]},
                    }
                },
            },
        }
    )
    r = check_hide_paths_support(cfg, "hardened", tmp_path)
    assert r is not None and "mcp.servers.srv" in r
