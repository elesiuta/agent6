# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""`warn_sandbox_gaps`: the run-entry warning when the resolved isolation
confines less than its name promises (`none`, strict without Landlock, or
hardened on Landlock below ABI 3, where truncation is unconfined)."""

from __future__ import annotations

from pathlib import Path

import pytest

from agent6.app.confine import (
    check_network_support,
    warn_cleartext_credential_endpoints,
    warn_sandbox_gaps,
)
from agent6.config import Config, SandboxConfig
from agent6.paths import jail_cache_home
from agent6.sandbox.detect import Environment, KernelInfo
from agent6.sandbox.jail import ToolMountNotes


def _env(landlock_abi: int) -> Environment:
    return Environment(
        in_container=False,
        container_signals=(),
        kernel=KernelInfo(raw="6.14.0", major=6, minor=14),
        userns_supported=True,
        landlock_abi=landlock_abi,
        seccomp_arch_supported=True,
        sandbox_available=True,
    )


def _cfg(tool_network: str = "auto", isolation: str = "auto") -> Config:
    return Config(
        sandbox=SandboxConfig(network=tool_network, isolation=isolation)  # type: ignore[arg-type]
    )


def test_none_warns_unsandboxed(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """The list of what is absent has to be complete, and the memory cap is
    not on it. Measured with `memory_limit_mb = 64` and isolation `none`: a
    400 MB allocation raises MemoryError through the run's jail session (the
    launcher applies the rlimit with confinement off), and succeeds only on
    the one-shot path, which runs a plain subprocess."""
    warn_sandbox_gaps("none", _env(4), _cfg(), root=tmp_path)
    err = capsys.readouterr().err
    assert "UNSANDBOXED" in err
    assert "memory_limit_mb" in err


def test_strict_without_landlock_warns(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """strict on a Landlock-less kernel (ABI 0) silently lost a documented
    layer: the launcher's best-effort ruleset enforces nothing and no surface
    said so, breaking the "no silent downgrade, always loudly" contract."""
    warn_sandbox_gaps("strict", _env(0), _cfg(), root=tmp_path)
    err = capsys.readouterr().err
    assert "WARNING" in err
    assert "Landlock" in err


def test_strict_with_landlock_is_silent(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("agent6.app.confine.tool_mount_notes", ToolMountNotes)
    warn_sandbox_gaps("strict", _env(2), _cfg(), root=tmp_path)
    assert capsys.readouterr().err == ""


def test_unreachable_tool_is_named_once(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A bin symlink whose target sits directly in $HOME cannot be mounted
    (mounting home would hand the jail every credential), so the tool dies in
    the jail with no explanation -- the preflight warning is the explanation."""
    monkeypatch.setattr(
        "agent6.app.confine.tool_mount_notes",
        lambda: ToolMountNotes(unreachable=("/home/op/.local/bin/x -> /home/op/x.sh",)),
    )
    warn_sandbox_gaps("strict", _env(2), _cfg(), root=tmp_path)
    err = capsys.readouterr().err
    assert "/home/op/.local/bin/x -> /home/op/x.sh" in err
    assert "never" in err and "mounted" in err


def test_a_tool_dragging_a_home_dir_into_the_jail_is_not_a_per_run_warning(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """`~/bin/x -> ~/.ssh/helper` mounts ~/.ssh read-only into the jail, which
    stays ALLOWED (the operator placed the symlink, and guessing at which dirs
    hold keys would be enumerating badness). It is not warned per run either:
    on a normal machine every uv-installed tool in ~/.local/bin points into
    ~/.local/share, so this fired a dozen times a run and buried the messages
    that mattered. `agent6 check` lists it, where someone is asking."""
    monkeypatch.setattr(
        "agent6.app.confine.tool_mount_notes",
        lambda: ToolMountNotes(exposes_home_dir=("/home/op/.local/bin/x -> /home/op/.ssh/helper",)),
    )
    warn_sandbox_gaps("strict", _env(2), _cfg(), root=tmp_path)
    assert capsys.readouterr().err == ""


def test_hardened_auto_warns_tool_network_degrade(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """network='auto' (the secure default) can't be offline on hardened
    (no netns), so it degrades to sharing the host network -- and must SAY so,
    never silently."""
    warn_sandbox_gaps("hardened", _env(4), _cfg("auto"), root=tmp_path)
    err = capsys.readouterr().err
    assert "WARNING" in err and "network" in err and "network namespace" in err


@pytest.mark.parametrize("abi", [1, 2])
def test_hardened_below_abi3_warns_truncate_unconfined(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], abi: int
) -> None:
    """Landlock ABI 1/2 does not confine truncate, so on hardened a jailed
    command can truncate files outside its write grants. `auto` keeps resolving
    to hardened on these ABI-1/2 hosts, so the over-promise must be said once
    per run, naming ABI 3 / Linux 6.2."""
    warn_sandbox_gaps("hardened", _env(abi), _cfg("host"), root=tmp_path)
    err = capsys.readouterr().err
    assert "WARNING" in err and "truncat" in err
    assert "ABI 3" in err and "6.2" in err


def test_hardened_abi3_plus_is_silent_on_truncate(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """From ABI 3 up Landlock confines truncation, so no truncate warning."""
    warn_sandbox_gaps("hardened", _env(3), _cfg("host"), root=tmp_path)
    assert "truncat" not in capsys.readouterr().err


def test_hardened_allow_says_nothing_about_the_network(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # An operator who set network='allow' asked for the tool to have the
    # network, so no degrade warning for it. `.git` is a separate degrade and
    # is expected here: hardened cannot protect it at all.
    warn_sandbox_gaps("hardened", _env(4), _cfg("host"), root=tmp_path)
    err = capsys.readouterr().err
    assert "network" not in err.lower().split("cannot protect .git")[-1]
    assert "cannot protect .git" in err


def test_hardened_warning_names_shared_tmp_and_persistent_home(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """'strict' gives each run a private /tmp tmpfs with HOME
    (/tmp/agent6-home) inside it, gone when the run ends. 'hardened' has no
    mount namespace, so /tmp is the host's shared /tmp and HOME is the
    persistent cache dir. The run-entry warnings say so rather than imply
    strict's private tmpfs, and the HOME one stands on its own: it rode the
    .git warning once, so `protect_git = false` lost it."""
    for cfg in (_cfg(), Config(sandbox=SandboxConfig(protect_git=False))):
        warn_sandbox_gaps("hardened", _env(4), cfg, root=tmp_path)
        err = capsys.readouterr().err
        assert str(jail_cache_home()) in err
        assert "/tmp/agent6-home" not in err
        assert "persists across runs" in err and "executable" in err
        assert ("cannot protect .git" in err) == cfg.sandbox.protect_git


def test_strict_cache_home_warns_naming_the_cost(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """`home = "cache"` under strict is an explicit widening: it runs, with a
    loud warning naming the persistence and the executable grant. The default
    strict HOME warns nothing."""
    warn_sandbox_gaps("strict", _env(4), Config(sandbox=SandboxConfig(home="cache")), root=tmp_path)
    err = capsys.readouterr().err
    assert "sandbox.home = 'cache'" in err and str(jail_cache_home()) in err
    assert "persists across runs" in err and "executable" in err
    warn_sandbox_gaps("strict", _env(4), _cfg(), root=tmp_path)
    assert str(jail_cache_home()) not in capsys.readouterr().err


def test_a_cleartext_credential_endpoint_warns_at_run_entry(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """An endpoint sending its credential over plaintext http to a non-loopback
    host is explicit-but-discouraged config: the run warns once per endpoint,
    naming it and the cost, and never refuses. A clean config warns nothing."""
    cfg = Config.model_validate(
        {
            "providers": {
                "corp": {
                    "api_format": "openai",
                    "base_url": "http://llm.corp.internal/v1",
                    "api_key_env": "K",
                }
            }
        }
    )
    warn_cleartext_credential_endpoints(cfg)
    err = capsys.readouterr().err
    assert "WARNING" in err and "[providers.corp]" in err and "plaintext http" in err
    warn_cleartext_credential_endpoints(Config())
    assert capsys.readouterr().err == ""


def test_explicit_block_refuses_on_hardened(tmp_path: Path) -> None:
    """network='session' is an ENFORCE setting: it needs a netns only strict
    provides, so on hardened we refuse (name what's unsupported + the fix)
    rather than run silently under-confined. 'auto' degrades instead."""
    err = check_network_support(_cfg("session"), "hardened")
    assert err is not None
    assert "sandbox.network = 'session'" in err and "auto" in err and "strict" in err
    # auto is NOT refused (it degrades with a warning) -> None.
    assert check_network_support(_cfg("auto"), "hardened") is None
    # On strict, block is enforceable -> no refusal.
    assert check_network_support(_cfg("session"), "strict") is None


def test_scanner_separates_unreachable_from_home_exposing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A symlink resolving DIRECTLY into $HOME is unreachable (home is never
    mounted); one resolving into a home SUBDIR is reachable but drags that
    subdir in; one resolving inside its own bin dir is neither."""
    from agent6.sandbox import jail as jail_mod

    home = tmp_path / "home"
    binf = home / ".local" / "bin"
    binf.mkdir(parents=True)
    (home / "tools").mkdir()
    (home / "x.sh").write_text("#!/bin/sh\n", encoding="utf-8")
    (home / "tools" / "y.sh").write_text("#!/bin/sh\n", encoding="utf-8")
    (binf / "z.sh").write_text("#!/bin/sh\n", encoding="utf-8")
    (binf / "x").symlink_to(home / "x.sh")
    (binf / "y").symlink_to(home / "tools" / "y.sh")
    (binf / "z").symlink_to(binf / "z.sh")
    monkeypatch.setattr(jail_mod.Path, "home", classmethod(lambda _cls: home))

    notes = jail_mod.tool_mount_notes()
    assert notes.unreachable == (f"{binf}/x -> {home}/x.sh",)
    assert notes.exposes_home_dir == (f"{binf}/y -> {home}/tools/y.sh",)


def test_hardened_warns_loudly_when_a_grant_exposes_the_private_dirs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Granting a region containing the config dir is a choice the operator may
    mean: real protection remains on hardened (writes stay confined, seccomp
    applies), so refusing would be paternalism. It warns instead and names what
    becomes readable. Strict masks the same grant and says nothing."""
    from agent6.app.confine import check_hide_paths_support

    home = tmp_path / "home"
    cfg_dir = home / ".config" / "agent6"
    cfg_dir.mkdir(parents=True)
    monkeypatch.setenv("AGENT6_CONFIG_HOME", str(cfg_dir))
    monkeypatch.setenv("AGENT6_STATE_HOME", str(tmp_path / "state"))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("agent6.app.confine.tool_mount_notes", ToolMountNotes)
    cfg = Config(sandbox=SandboxConfig(extra_read_paths=(str(home),)))

    warn_sandbox_gaps("hardened", _env(4), cfg, root=tmp_path)
    err = capsys.readouterr().err
    assert "WARNING" in err and "can read" in err
    assert str(cfg_dir) in err and str(home) in err
    assert check_hide_paths_support(cfg, "hardened", tmp_path) is None  # warned, not refused

    warn_sandbox_gaps("strict", _env(4), cfg, root=tmp_path)
    assert str(cfg_dir) not in capsys.readouterr().err


def test_the_workspace_itself_counts_as_a_granted_region(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Verified live before this existed: with the config dir INSIDE the
    workspace, a jailed `cat` on hardened printed secrets.toml. The workspace
    is granted implicitly, so it has to be checked like any other region."""
    cfg_dir = tmp_path / ".config" / "agent6"
    cfg_dir.mkdir(parents=True)
    monkeypatch.setenv("AGENT6_CONFIG_HOME", str(cfg_dir))
    monkeypatch.setenv("AGENT6_STATE_HOME", str(tmp_path / "state"))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("agent6.app.confine.tool_mount_notes", ToolMountNotes)

    warn_sandbox_gaps("hardened", _env(4), Config(), root=tmp_path)
    assert str(cfg_dir) in capsys.readouterr().err


def test_hardened_refuses_an_explicit_hide_entry_it_cannot_mask(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An operator who wrote hide_paths down asked explicitly, so the rule the
    other knobs follow applies: a default degrades with a warning, an explicit
    value refuses rather than being silently ineffective."""
    from agent6.app.confine import check_hide_paths_support

    monkeypatch.setenv("AGENT6_CONFIG_HOME", str(tmp_path / "cfg"))
    monkeypatch.setenv("AGENT6_STATE_HOME", str(tmp_path / "state"))
    ws = tmp_path / "ws"
    ws.mkdir()
    monkeypatch.chdir(ws)
    hidden = ws / "cred.txt"
    cfg = Config(sandbox=SandboxConfig(hide_paths=(str(hidden),)))
    err = check_hide_paths_support(cfg, "hardened", tmp_path)
    assert err is not None and str(hidden) in err
    assert check_hide_paths_support(cfg, "strict", tmp_path) is None


def test_a_plain_hardened_run_neither_warns_nor_refuses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from agent6.app.confine import check_hide_paths_support

    # Private homes OUTSIDE every hardened grant region (/tmp is granted RW,
    # so a tmp-based home is genuinely exposed there -- the twin below pins
    # that as a true positive). The normal ~/.local layout is this case. The
    # cache stays where the suite put it: the jail's HOME lives there, and the
    # policy build creates it.
    for var in ("CONFIG", "STATE", "DATA"):
        monkeypatch.setenv(f"AGENT6_{var}_HOME", f"/nonexistent-private/{var.lower()}")
    ws = tmp_path / "ws"
    ws.mkdir()
    monkeypatch.chdir(ws)
    monkeypatch.setattr("agent6.app.confine.tool_mount_notes", ToolMountNotes)
    cfg = Config(sandbox=SandboxConfig(network="host", protect_git=False))
    warn_sandbox_gaps("hardened", _env(4), cfg, root=tmp_path)
    err = capsys.readouterr().err
    # hardened's persistent HOME is the one notice every such run carries;
    # nothing here is an exposure.
    assert err.count("WARNING") == 1 and str(jail_cache_home()) in err, err
    assert "can read" not in err
    assert check_hide_paths_support(cfg, "hardened", tmp_path) is None


def test_hardened_warns_when_private_state_sits_in_a_granted_region(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The preflight used to see only cwd + the extra grants, so private
    dirs under the host's shared /tmp (which the hardened launcher grants
    RW) went unwarned -- silently readable by every command."""
    for var in ("CONFIG", "STATE", "DATA", "CACHE"):
        monkeypatch.setenv(f"AGENT6_{var}_HOME", str(tmp_path / var.lower()))
    ws = tmp_path / "ws"
    ws.mkdir()
    monkeypatch.chdir(ws)
    monkeypatch.setattr("agent6.app.confine.tool_mount_notes", ToolMountNotes)
    cfg = Config(sandbox=SandboxConfig(network="host", protect_git=False))
    warn_sandbox_gaps("hardened", _env(4), cfg, root=tmp_path)
    err = capsys.readouterr().err
    assert str(tmp_path).startswith("/tmp"), "the fixture premise: pytest tmp lives under /tmp"
    assert "jailed commands can read" in err and "/tmp" in err


def test_root_on_hardened_names_what_it_costs(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Running as root is the operator's explicit widening, so it warns rather
    than refuses -- but the warning has to name the cost, not just the choice.
    Verified as real uid 0: under hardened a jailed command reads /etc/shadow,
    /etc/sudoers and the host's ssh private keys, because Landlock grants the
    documented read-only system set and root stops file permissions narrowing
    it. The root banner names running as root; it does not name this."""
    monkeypatch.setattr("agent6.app.confine.tool_mount_notes", ToolMountNotes)
    monkeypatch.setattr("agent6.app.confine.is_root", lambda: True)
    warn_sandbox_gaps(
        "hardened", _env(4), Config(sandbox=SandboxConfig(protect_git=False)), root=tmp_path
    )
    err = capsys.readouterr().err
    assert "running as root" in err
    assert "/etc/shadow" in err and "ssh private keys" in err


def test_root_on_strict_says_nothing_about_it(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """strict pivots into a minimal rootfs -- verified as real uid 0, its /etc
    holds a single entry and none of those files exist. Warning there would be
    telling the operator about a cost they are not paying."""
    monkeypatch.setattr("agent6.app.confine.tool_mount_notes", ToolMountNotes)
    monkeypatch.setattr("agent6.app.confine.is_root", lambda: True)
    warn_sandbox_gaps("strict", _env(4), Config(), root=tmp_path)
    assert capsys.readouterr().err == ""


def test_a_normal_user_on_hardened_is_not_told_about_root(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("agent6.app.confine.tool_mount_notes", ToolMountNotes)
    monkeypatch.setattr("agent6.app.confine.is_root", lambda: False)
    warn_sandbox_gaps(
        "hardened", _env(4), Config(sandbox=SandboxConfig(protect_git=False)), root=tmp_path
    )
    assert "running as root" not in capsys.readouterr().err


def test_auto_degrade_warns_with_the_reason(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The degrade ITSELF is loud, not only its consequences: auto landing on
    hardened printed the network/protect_git consequences but never why strict
    was skipped. One owner (detect.degrade_reason) feeds this line, check
    sandbox, and check config."""
    monkeypatch.setattr("agent6.app.confine.tool_mount_notes", ToolMountNotes)

    def _why(_env: object) -> str:
        return "userns blocked (test)"

    monkeypatch.setattr("agent6.app.confine.degrade_reason", _why)
    warn_sandbox_gaps("hardened", _env(4), _cfg(), root=tmp_path)
    err = capsys.readouterr().err
    assert "'auto' selected 'hardened', not 'strict': userns blocked (test)" in err


def test_explicit_hardened_has_no_degrade_line(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """An operator who WROTE hardened chose it; nothing degraded."""
    monkeypatch.setattr("agent6.app.confine.tool_mount_notes", ToolMountNotes)

    def _why(_env: object) -> str:
        return "userns blocked (test)"

    monkeypatch.setattr("agent6.app.confine.degrade_reason", _why)
    warn_sandbox_gaps("hardened", _env(4), _cfg(isolation="hardened"), root=tmp_path)
    err = capsys.readouterr().err
    assert "not 'strict'" not in err


def test_unsandboxed_origin_says_auto_or_the_operator(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The UNSANDBOXED banner attributed `isolation = 'none'` to the operator
    even when `auto` resolved there on a host with no confinement mechanism."""
    monkeypatch.setattr("agent6.app.confine.tool_mount_notes", ToolMountNotes)

    def _why(_env: object) -> str:
        return "nothing here (test)"

    monkeypatch.setattr("agent6.app.confine.degrade_reason", _why)
    warn_sandbox_gaps("none", _env(0), _cfg(), root=tmp_path)
    err = capsys.readouterr().err
    assert "'auto' found no confinement mechanism" in err
    assert "sandbox.isolation = 'none'" not in err
    warn_sandbox_gaps("none", _env(0), _cfg(isolation="none"), root=tmp_path)
    err = capsys.readouterr().err
    assert "sandbox.isolation = 'none'" in err
