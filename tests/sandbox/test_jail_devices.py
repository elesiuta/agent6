# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""`[sandbox].extra_device_paths`: operator-granted device nodes in the jail."""

from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = pytest.mark.needs_namespaces


def _policy(tmp_path: Path, level: str, argv: tuple[str, ...], devices: tuple[str, ...]):
    from agent6.config import Config
    from agent6.tools.dispatch import jail_policy

    base = jail_policy(
        tmp_path,
        Config(),
        level,  # pyright: ignore[reportArgumentType]
        argv,
        network="none",
    )
    import dataclasses

    return dataclasses.replace(base, extra_device_paths=tuple(Path(d) for d in devices))


@pytest.mark.parametrize("level", ["strict", "hardened"])
def test_a_granted_device_node_is_openable(tmp_path: Path, level: str) -> None:
    """The grant is dead unless the node is present AND writable: /dev/tty is
    a char device the strict /dev deliberately omits and hardened's rules do
    not cover, so a granted probe proves the whole path (bind without the
    nodev floor on strict; the Landlock read+write rule on both)."""
    from agent6.sandbox.jail import run_in_jail

    if not Path("/dev/tty").exists():
        pytest.skip("host has no /dev/tty")
    probe = ("sh", "-c", "test -c /dev/tty && echo present || echo absent")
    denied = run_in_jail(_policy(tmp_path, level, probe, ()))
    granted = run_in_jail(_policy(tmp_path, level, probe, ("/dev/tty",)))
    if level == "strict":
        assert "absent" in denied.stdout  # the default /dev has five nodes, no tty
    assert "present" in granted.stdout


@pytest.mark.parametrize("level", ["strict", "hardened"])
def test_a_non_device_grant_refuses_loudly(tmp_path: Path, level: str) -> None:
    """A path under /dev that is absent (or not a char/block device) refuses
    the launch with the path named -- never a silent skip that would leave the
    operator's GPU task failing confusingly later."""
    from agent6.sandbox.jail import JailUnavailableError, run_in_jail

    with pytest.raises(JailUnavailableError, match="extra_device path /dev/nonesuch-node"):
        run_in_jail(_policy(tmp_path, level, ("true",), ("/dev/nonesuch-node",)))
