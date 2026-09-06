# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""A jailed command holds no capabilities, whatever profile launched the jail.

The profile that bites: a launcher holding ambient capabilities WITHOUT
CAP_SETPCAP (a caps-granting container, an elevated service). There the
bounding-set drop fails EPERM, and returning early at that point leaves the
ambient set to flow across exec into the jailed command. The empty capset
needs no privilege and must still run.

The scaffold builds that profile with user-namespace tooling: an outer userns
(self mapped to 0, a subuid range for other uids), then `setpriv` to become a
NON-root uid carrying one ambient capability and no CAP_SETPCAP. The launcher
runs `hardened` (no inner user namespace, so nothing re-grants what the drop
must remove) and the jailed command reports its own sets via capget(2) --
/proc may be out of Landlock's grant, the syscall always answers.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from agent6.config import Config
from agent6.sandbox.jail import (
    _policy_spec,  # pyright: ignore[reportPrivateUsage]
    locate_jail_binary,
)
from agent6.tools.dispatch import jail_policy

pytestmark = pytest.mark.needs_namespaces

_SCAFFOLD = (
    "unshare",
    "-U",
    "--map-user=0",
    "--map-users=auto",
    "--map-groups=auto",
    "--map-group=0",
    "--",
    "setpriv",
    "--reuid",
    "1",
    "--regid",
    "1",
    "--clear-groups",
    "--inh-caps",
    "+net_raw",
    "--ambient-caps",
    "+net_raw",
)

# capget(2), version 3: effective is word 0 of each of the two data structs.
_CAPGET = (
    "import ctypes;l=ctypes.CDLL(None,use_errno=True);"
    "buf=(ctypes.c_uint32*6)();hdr=(ctypes.c_uint32*2)(0x20080522,0);"
    "r=l.capget(hdr,buf);print('capget_rc',r,'eff',hex(buf[0]),hex(buf[3]))"
)


def _scaffold_available() -> bool:
    """The retained-caps profile needs subuid auto-mapping + setpriv ambient
    support; probe the exact scaffold rather than guessing from versions."""
    probe = subprocess.run(
        [*_SCAFFOLD, "sh", "-c", "grep -q '^CapAmb:.*2000' /proc/self/status"],
        capture_output=True,
        timeout=30,
        check=False,
    )
    return probe.returncode == 0


def test_a_launcher_without_cap_setpcap_still_strips_the_child(tmp_path: Path) -> None:
    """Under the ambient-caps/no-CAP_SETPCAP profile the jailed command's
    effective set is empty: the bounding-set EPERM does not skip the capset."""
    binary = locate_jail_binary()
    if binary is None:
        pytest.skip("no agent6-jail binary")
    if not _scaffold_available():
        pytest.skip("host cannot build the ambient-caps profile (subuid/setpriv)")
    # World-traversable cwd: inside the scaffold this test runs as an
    # unprivileged subuid, which cannot enter the pytest tmp dir.
    cwd = Path("/tmp") / f"agent6-capdrop-{tmp_path.name}"
    cwd.mkdir(mode=0o777)
    try:
        policy = json.dumps(
            _policy_spec(
                jail_policy(cwd, Config(), "hardened", ("python3", "-c", _CAPGET), network="none")
            )
        )
        res = subprocess.run(
            [*_SCAFFOLD, str(binary)],
            input=policy + "\n",
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        assert res.returncode == 0, res.stderr
        answer = json.loads(res.stdout.strip().splitlines()[-1])
        assert answer["returncode"] == 0, answer
        assert "capget_rc 0 eff 0x0 0x0" in answer["stdout"], answer
    finally:
        for child in cwd.iterdir():
            child.unlink()
        cwd.rmdir()
