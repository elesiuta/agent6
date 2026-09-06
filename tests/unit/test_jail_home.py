# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The jail's HOME is a directory of agent6's own: `/tmp/agent6-home` inside
strict's private tmpfs, or the persistent cache dir where there is no private
/tmp (`hardened`, `none`) or where `[sandbox].home = "cache"` asks for it.
The policy builder creates the persistent one, so every surface that jails
gets a HOME that exists; the preflight refuses one that cannot be agent6's
own."""

from __future__ import annotations

import stat
from dataclasses import replace
from pathlib import Path

import pytest

from agent6.app.confine import check_jail_home, config_refusal, warn_sandbox_gaps
from agent6.app.reporter import Reporter
from agent6.config import Config, SandboxConfig
from agent6.paths import effective_user, jail_cache_home
from agent6.sandbox.detect import Environment, KernelInfo
from agent6.sandbox.jail import JailUnavailableError
from agent6.tools import policy as policy_module
from agent6.tools.policy import JAIL_TMP_HOME, jail_policy


def _warned(isolation: str, cfg: Config, root: Path) -> list[str]:
    env = Environment(
        in_container=False,
        container_signals=(),
        kernel=KernelInfo(raw="6.14.0", major=6, minor=14),
        userns_supported=True,
        landlock_abi=4,
        seccomp_arch_supported=True,
        sandbox_available=True,
    )
    lines: list[str] = []
    reporter = Reporter(out=lines.append, err=lines.append)
    warn_sandbox_gaps(isolation, env, cfg, root=root, reporter=reporter)  # pyright: ignore[reportArgumentType]
    return lines


def test_strict_defaults_to_the_private_tmpfs_home(tmp_path: Path) -> None:
    policy = jail_policy(tmp_path, Config(), "strict", ("true",))
    assert dict(policy.env)["HOME"] == str(JAIL_TMP_HOME) == "/tmp/agent6-home"
    assert jail_cache_home() not in policy.extra_rw_paths
    assert not jail_cache_home().exists()


@pytest.mark.parametrize(
    ("isolation", "cfg"),
    [
        ("hardened", Config()),
        ("none", Config()),
        ("strict", Config(sandbox=SandboxConfig(home="cache"))),
    ],
)
def test_the_builder_creates_and_grants_the_persistent_home(
    tmp_path: Path, isolation: str, cfg: Config
) -> None:
    """Without a private /tmp the HOME persists; strict opts in. The builder
    alone (no preflight ran) creates it 0700 and rides the extra_rw_paths
    grant, so `agent6 exec` or an MCP probe gets a HOME that exists: the
    launcher skips a missing rw path silently."""
    home = jail_cache_home()
    assert not home.exists()
    policy = jail_policy(tmp_path, cfg, isolation, ("true",))  # pyright: ignore[reportArgumentType]
    assert dict(policy.env)["HOME"] == str(home)
    assert home in policy.extra_rw_paths
    assert home.is_dir() and not home.is_symlink()
    assert stat.S_IMODE(home.lstat().st_mode) == 0o700


def test_the_preflight_check_inspects_without_creating(tmp_path: Path) -> None:
    """The refusal check itself writes nothing: creation is the builder's.
    (Under hardened the refusal list also builds the run's policy for the
    exposure scan, and the builder creates the dir there.)"""
    assert check_jail_home(Config(), "hardened", explicitly_set=False) is None
    assert config_refusal(Config(), "strict", tmp_path) is None
    assert not jail_cache_home().exists()


def test_a_symlink_at_the_cache_home_refuses(tmp_path: Path) -> None:
    """A symlink there redirects every jailed write, and the operator's own
    home is the obvious target. The preflight refuses it as a message; a
    surface without a preflight (`agent6 exec`) gets the same words from the
    builder."""
    home = jail_cache_home()
    home.parent.mkdir(parents=True, exist_ok=True)
    home.symlink_to(tmp_path)
    msg = config_refusal(Config(), "hardened", tmp_path)
    assert msg is not None
    assert str(home) in msg and "symlink" in msg
    with pytest.raises(JailUnavailableError, match="symlink"):
        jail_policy(tmp_path, Config(), "hardened", ("true",))


def test_a_dir_owned_by_someone_else_refuses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """On a shared box the path may already be another user's directory: it
    is never bound, and the message names both uids. The other user is faked
    by shifting `effective_user` (a real chown needs root); the ownership
    comparison itself is the real one."""
    home = jail_cache_home()
    home.mkdir(parents=True)
    me = effective_user()
    monkeypatch.setattr(policy_module, "effective_user", lambda: replace(me, uid=me.uid + 1))
    msg = config_refusal(Config(), "hardened", tmp_path)
    assert msg is not None
    assert str(home) in msg and f"uid {me.uid}" in msg
    with pytest.raises(JailUnavailableError, match="owned by"):
        jail_policy(tmp_path, Config(), "hardened", ("true",))


def test_a_cache_home_open_to_others_refuses(tmp_path: Path) -> None:
    """A jailed command owns HOME and may chmod it; a mode with any group or
    other bit lets another local user plant a `~/.gitconfig` or cache content
    the next jailed run consumes. Checked on every build, never silently
    restored: the refusal names the mode found and the fix."""
    home = jail_cache_home()
    home.mkdir(parents=True, mode=0o700)
    home.chmod(0o777)
    msg = config_refusal(Config(), "hardened", tmp_path)
    assert msg is not None
    assert str(home) in msg and "0777" in msg and f"chmod 700 {home}" in msg
    with pytest.raises(JailUnavailableError, match="chmod 700"):
        jail_policy(tmp_path, Config(), "hardened", ("true",))
    assert stat.S_IMODE(home.lstat().st_mode) == 0o777  # the operator sees it, nothing rewrites it
    home.chmod(0o700)
    assert config_refusal(Config(), "hardened", tmp_path) is None
    assert (
        jail_cache_home() in jail_policy(tmp_path, Config(), "hardened", ("true",)).extra_rw_paths
    )


def test_a_cache_home_inside_a_private_dir_refuses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`AGENT6_CACHE_HOME` under the state base would put a writable grant
    inside the always-hidden tree, re-bound through the strict mask; the
    config validator refuses the same for `extra_write_paths`."""
    monkeypatch.setenv("AGENT6_STATE_HOME", str(tmp_path / "state"))
    monkeypatch.setenv("AGENT6_CACHE_HOME", str(tmp_path / "state" / "cache"))
    msg = config_refusal(Config(), "hardened", tmp_path / "ws")
    assert msg is not None
    assert "private dir" in msg and str(tmp_path / "state") in msg
    with pytest.raises(JailUnavailableError, match="private dir"):
        jail_policy(tmp_path / "ws", Config(), "hardened", ("true",))


def test_a_symlinked_ancestor_into_a_private_dir_refuses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The containment check compares resolved paths: an `AGENT6_CACHE_HOME`
    reached through a symlink into the state base is refused before anything
    is created there (the real dir would sit inside the hidden tree), while a
    symlinked ancestor elsewhere is an ordinary cache location."""
    state = tmp_path / "state"
    state.mkdir()
    monkeypatch.setenv("AGENT6_STATE_HOME", str(state))
    (tmp_path / "link").symlink_to(state)
    monkeypatch.setenv("AGENT6_CACHE_HOME", str(tmp_path / "link" / "cache"))
    msg = config_refusal(Config(), "hardened", tmp_path / "ws")
    assert msg is not None
    assert "private dir" in msg and str(state) in msg
    with pytest.raises(JailUnavailableError, match="private dir"):
        jail_policy(tmp_path / "ws", Config(), "hardened", ("true",))
    assert not (state / "cache").exists()
    real = tmp_path / "real"
    real.mkdir()
    (tmp_path / "elsewhere").symlink_to(real)
    monkeypatch.setenv("AGENT6_CACHE_HOME", str(tmp_path / "elsewhere" / "cache"))
    assert config_refusal(Config(), "hardened", tmp_path / "ws") is None
    policy = jail_policy(tmp_path / "ws", Config(), "hardened", ("true",))
    assert tmp_path / "elsewhere" / "cache" / "home" in policy.extra_rw_paths
    assert (real / "cache" / "home").is_dir()


@pytest.mark.parametrize("isolation", ["hardened", "none"])
def test_an_explicit_tmp_home_refuses_where_there_is_no_private_tmp(
    tmp_path: Path, isolation: str
) -> None:
    """`home = "tmp"` is a private tmpfs, which only strict has: a value the
    operator wrote down refuses, naming the resolved level and the fix (the
    `protect_git` rule); the default degrades (below)."""
    msg = check_jail_home(Config(), isolation, explicitly_set=True)  # pyright: ignore[reportArgumentType]
    assert msg is not None
    assert "requires the strict isolation" in msg
    assert f"resolved to '{isolation}'" in msg
    assert "sandbox.home = 'cache'" in msg
    explicit = frozenset({"sandbox.home"})
    assert config_refusal(Config(), isolation, tmp_path, explicit_leaves=explicit) == msg  # pyright: ignore[reportArgumentType]
    assert config_refusal(Config(), isolation, tmp_path) is None  # pyright: ignore[reportArgumentType]
    assert config_refusal(Config(), "strict", tmp_path, explicit_leaves=explicit) is None


def test_the_default_degrades_with_a_warning_naming_the_home(tmp_path: Path) -> None:
    """hardened's start-of-run warning names the persistent HOME and its cost
    on its own, whatever `protect_git` says."""
    assert check_jail_home(Config(), "hardened", explicitly_set=False) is None
    for cfg in (Config(), Config(sandbox=SandboxConfig(protect_git=False))):
        home = str(jail_cache_home())
        lines = [line for line in _warned("hardened", cfg, tmp_path) if home in line]
        assert len(lines) == 1, lines
        assert "persists across runs" in lines[0] and "executable" in lines[0]


def test_strict_cache_is_an_explicit_widening_that_warns(tmp_path: Path) -> None:
    lines = _warned("strict", Config(sandbox=SandboxConfig(home="cache")), tmp_path)
    assert any(
        "sandbox.home = 'cache'" in line and str(jail_cache_home()) in line and "persists" in line
        for line in lines
    ), lines
    assert not any(str(jail_cache_home()) in line for line in _warned("strict", Config(), tmp_path))
