# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""`protect_git` is a read-only bind, so it is strict-only.

Landlock has no deny rules: protecting `.git` on hardened meant not granting
the workspace ROOT, because a grant is recursive and granting the root its own
create/remove rights grants them over `.git` too. That cost every top-level
write, which is more than the protection is worth when the operator can have
the real thing by using strict.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agent6.app.confine import check_protect_git_support, warn_sandbox_gaps
from agent6.app.reporter import Reporter
from agent6.config import Config
from agent6.sandbox.detect import Environment, KernelInfo


def _said() -> tuple[Reporter, list[str]]:
    lines: list[str] = []
    return Reporter(out=lines.append, err=lines.append), lines


def test_a_default_degrades_with_a_warning(tmp_path: Path) -> None:
    """`protect_git` defaults to true, and a default this host cannot honour
    must still run -- loudly, never silently ineffective."""
    reporter, lines = _said()
    env = Environment(
        in_container=False,
        container_signals=(),
        kernel=KernelInfo(raw="7.0.0", major=7, minor=0),
        userns_supported=False,
        landlock_abi=4,
        seccomp_arch_supported=True,
        sandbox_available=True,
    )
    warn_sandbox_gaps("hardened", env, Config(), reporter=reporter, root=tmp_path)
    assert any("cannot protect .git" in line for line in lines)
    assert check_protect_git_support(Config(), "hardened", explicitly_set=False) is None


def test_a_value_the_operator_wrote_down_refuses(tmp_path: Path) -> None:
    """They asked for something specific; running without it would be a lie."""
    message = check_protect_git_support(Config(), "hardened", explicitly_set=True)
    assert message is not None
    assert "requires the strict isolation" in message
    assert "sandbox.protect_git = false" in message  # names the fix


def test_strict_provides_it_either_way(tmp_path: Path) -> None:
    for explicit in (True, False):
        assert check_protect_git_support(Config(), "strict", explicitly_set=explicit) is None


def test_opting_out_never_refuses(tmp_path: Path) -> None:
    off = Config.model_validate({"sandbox": {"protect_git": False}})
    assert check_protect_git_support(off, "hardened", explicitly_set=True) is None


def test_only_strict_carves_git_out_of_the_jail(tmp_path: Path) -> None:
    """The carve-out is what cost the top-level writes. On hardened there is
    now nothing to carve, so `touch newfile` at the workspace root works."""
    from agent6.tools.dispatch import jail_policy

    (tmp_path / ".git").mkdir()
    strict = jail_policy(tmp_path, Config(), "strict", ("true",))
    hardened = jail_policy(tmp_path, Config(), "hardened", ("true",))
    assert (tmp_path / ".git").resolve() in strict.extra_protect_paths
    assert hardened.extra_protect_paths == ()


@pytest.mark.needs_namespaces
def test_a_new_top_level_entry_can_be_created_on_hardened(tmp_path: Path) -> None:
    """The whole point: `mkfifo`, `touch`, `mkdir` at the workspace root all
    failed with a misleading "File exists" while the carve-out was there."""
    from agent6.sandbox.jail import run_in_jail
    from agent6.tools.dispatch import jail_policy

    (tmp_path / ".git").mkdir()
    res = run_in_jail(
        jail_policy(
            tmp_path,
            Config(),
            "hardened",
            ("sh", "-c", "touch newfile && mkdir build && mkfifo pipe && echo made-them"),
        )
    )
    assert "made-them" in res.stdout, res.stdout + res.stderr


def test_the_default_reaches_the_check_as_a_default(tmp_path: Path) -> None:
    """The caller asked `effective.sources` for the explicit leaves, and that
    dict holds EVERY leaf with its layer, so a default read as operator intent:
    on a host without user namespaces (auto -> hardened) `agent6 run`/`ask`
    refused to start at all, against a config nobody had written."""
    from agent6.config.layer import load_effective

    effective = load_effective(tmp_path, None)
    assert effective.config.sandbox.protect_git is True
    assert effective.sources["sandbox.protect_git"] == "default"
    assert "sandbox.protect_git" not in effective.explicit_leaves
    assert (
        check_protect_git_support(
            effective.config,
            "hardened",
            explicitly_set="sandbox.protect_git" in effective.explicit_leaves,
        )
        is None
    )
