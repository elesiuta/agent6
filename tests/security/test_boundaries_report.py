# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""`agent6 check boundaries` tells the truth about what the launcher enforces.

The report's system-path lists are Python mirrors of constants the Rust
launcher owns (SYSTEM_BINDS, the hardened ro_paths base). A drift between the
report and the enforcer would make the report lie, so the mirrors are pinned
against the launcher's source text.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from agent6.config import Config, SandboxConfig
from agent6.paths import jail_cache_home
from agent6.sandbox.jail import ToolMountNotes
from agent6.ui.cli import check_cmds

_MAIN_RS = Path(__file__).resolve().parents[2] / "src" / "agent6" / "jail" / "src" / "main.rs"


def test_strict_system_binds_mirror_the_launcher() -> None:
    src = _MAIN_RS.read_text(encoding="utf-8")
    m = re.search(r"const SYSTEM_BINDS[^=]*=\s*\[(.*?)\];", src, re.S)
    assert m, "SYSTEM_BINDS not found in main.rs"
    rust = tuple(re.findall(r'"([^"]+)"', m.group(1)))
    assert rust == check_cmds._STRICT_SYSTEM_BINDS  # pyright: ignore[reportPrivateUsage]


def test_hardened_ro_base_mirrors_the_launcher() -> None:
    src = _MAIN_RS.read_text(encoding="utf-8")
    m = re.search(r"let mut ro_paths[^=]*=\s*vec!\[(.*?)\];", src, re.S)
    assert m, "hardened ro_paths base not found in main.rs"
    rust = tuple(re.findall(r'PathBuf::from\("([^"]+)"\)', m.group(1)))
    assert rust == check_cmds._HARDENED_SYSTEM_RO  # pyright: ignore[reportPrivateUsage]


def _force(monkeypatch: pytest.MonkeyPatch, isolation: str, reason: str | None = None) -> None:
    monkeypatch.setattr(check_cmds, "detect_env", object)

    def _select(_req: str, _env: object) -> str:
        return isolation

    def _reason(_env: object) -> str | None:
        return reason

    monkeypatch.setattr(check_cmds, "resolve_isolation", _select)
    monkeypatch.setattr(check_cmds, "degrade_reason", _reason)
    monkeypatch.setattr(
        check_cmds, "tool_mount_notes", lambda: ToolMountNotes(exposes_home_dir=("a -> b",))
    )


def test_boundaries_report_covers_every_actor(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """One glance answers: who is confined, which files, which network."""
    _force(monkeypatch, "strict")
    cfg = Config(sandbox=SandboxConfig(hide_paths=("/work/secrets",)))
    checks = check_cmds._check_boundaries_section(cfg)  # pyright: ignore[reportPrivateUsage]
    out = capsys.readouterr().out
    assert [c.status for c in checks] == ["PASS"]
    assert "in-process file tools" in out
    assert "jailed commands" in out
    assert "(the workspace; .git re-bound read-only)" in out
    assert "ro  system: /usr /bin /sbin /lib /lib64 /etc/alternatives" in out
    assert "rw  /tmp/agent6-home  (HOME" in out
    assert "operator tools: 1 resolved bin-dir" in out
    assert "/work/secrets" in out and "masked out of the jail's view" in out
    assert "network  session" in out
    assert "memory   no cap (sandbox.memory_limit_mb = 0)" in out
    assert "mcp servers: none run" in out
    assert "egress is NOT bounded" in out
    assert "secrets.toml" in out


def test_boundaries_report_is_level_aware(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """hardened has no mounts and no private anything; the words must not
    borrow strict's. A degraded auto also names its cause here."""
    _force(monkeypatch, "hardened", reason="userns blocked (test)")
    checks = check_cmds._check_boundaries_section(Config())  # pyright: ignore[reportPrivateUsage]
    out = capsys.readouterr().out
    assert [c.status for c in checks] == ["PASS"]
    assert "not strict: userns blocked (test)" in out
    assert "ro  system (Landlock): /usr /bin /sbin /lib /lib64 /etc /dev" in out
    assert f"rw  {jail_cache_home()}  (HOME" in out and "persists" in out
    assert "denied by Landlock" in out
    assert "re-bound" not in out
    assert "private to the command" not in out
    assert "network  host" in out


def test_boundaries_report_names_the_opted_in_persistent_home(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The jail's HOME is a write grant, so the report lists it: under
    `home = "cache"` strict's is the persistent cache dir, named as such."""
    _force(monkeypatch, "strict")
    check_cmds._check_boundaries_section(  # pyright: ignore[reportPrivateUsage]
        Config(sandbox=SandboxConfig(home="cache"))
    )
    out = capsys.readouterr().out
    assert f"rw  {jail_cache_home()}  (HOME" in out and "sandbox.home = cache" in out
    assert "/tmp/agent6-home" not in out


def test_boundaries_report_names_the_home_under_none(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """`none` has no jail and still a HOME of agent6's own; the report says so
    after the UNCONFINED line, so the HOME grant is listed at every level."""
    _force(monkeypatch, "none")
    check_cmds._check_boundaries_section(Config())  # pyright: ignore[reportPrivateUsage]
    out = capsys.readouterr().out
    assert "UNCONFINED" in out
    assert f"rw  {jail_cache_home()}  (HOME, persists across runs: none has no private /tmp)" in out


def test_boundaries_report_says_withheld_rather_than_unapproved(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """`run_commands = "no"` withholds the command tools from the model. The
    header read "approval: sandbox.run_commands = no", which describes a
    prompting policy for tools the model never sees."""
    _force(monkeypatch, "strict")
    check_cmds._check_boundaries_section(  # pyright: ignore[reportPrivateUsage]
        Config(sandbox=SandboxConfig(run_commands="no"))
    )
    out = capsys.readouterr().out
    assert "withheld from the model (sandbox.run_commands = no)" in out
    assert "approval: sandbox.run_commands" not in out


def test_boundaries_report_names_each_mcp_server(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from agent6.config._sandbox import MCPConfig, MCPSandbox, MCPServerEntry

    _force(monkeypatch, "strict")
    cfg = Config(
        mcp=MCPConfig(
            enabled=True,
            servers={
                "docs": MCPServerEntry(url="http://127.0.0.1:9000/mcp"),
                "fs": MCPServerEntry(
                    command=("mcp-fs",),
                    sandbox=MCPSandbox(read_paths=("/data",), network="none"),
                ),
                "off": MCPServerEntry(command=("x",), enabled=False),
            },
        )
    )
    check_cmds._check_boundaries_section(cfg)  # pyright: ignore[reportPrivateUsage]
    out = capsys.readouterr().out
    assert "docs: http http://127.0.0.1:9000/mcp" in out
    assert "fs: spawned in the jail" in out and "paths ro+1 rw+0, network none" in out
    assert "off: DISABLED" in out
