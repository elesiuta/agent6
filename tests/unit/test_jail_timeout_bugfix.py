# SPDX-License-Identifier: Apache-2.0
"""Regression test for the jail launcher timeout hang.

Bug: run_in_jail did not wrap the launcher subprocess in a TimeoutExpired
handler. A launcher that hangs (e.g. because a backgrounded grandchild keeps the
stdout pipe open) would let subprocess's timeout fire and propagate
TimeoutExpired as an opaque error, leaving the orphaned grandchild running.

Fix: launch the launcher with start_new_session, kill its whole process group on
timeout, and return rc=124 (mirroring the unsandboxed path's contract).

We can't run the real Rust jail here (no user namespaces), so we stand in a fake
"launcher" shell script that reproduces the hang: it backgrounds a long-lived
process which inherits the stdout pipe and then blocks forever, so the parent's
communicate() never sees EOF and the timeout must fire.
"""

from __future__ import annotations

import stat
import time
from pathlib import Path

import pytest

from agent6.sandbox import jail
from agent6.types import JailPolicy


def _write_fake_launcher(tmp_path: Path) -> Path:
    # Background a child that holds the stdout fd open, then block forever.
    # marker file lets the test confirm the whole group was killed.
    marker = tmp_path / "grandchild_alive"
    script = tmp_path / "fake-jail.sh"
    script.write_text(
        "#!/bin/sh\n"
        # grandchild keeps stdout (fd 1) open and stays alive; writes a marker
        # repeatedly so we can detect if it survives the group kill.
        f"( while true; do echo alive > '{marker}'; sleep 0.2; done ) &\n"
        # parent launcher blocks forever -> communicate() must time out.
        "sleep 600\n"
    )
    script.chmod(script.stat().st_mode | stat.S_IEXEC | stat.S_IRUSR)
    return script


def test_jail_timeout_returns_124_and_kills_group(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = _write_fake_launcher(tmp_path)
    monkeypatch.setattr(jail, "locate_jail_binary", lambda: fake)

    def _policy_spec(policy: JailPolicy) -> dict[str, object]:
        return {}

    monkeypatch.setattr(jail, "_policy_spec", _policy_spec)

    policy = JailPolicy(
        cwd=tmp_path,
        argv=("/bin/true",),
        isolation="strict",
        network="none",
        # launcher sleeps 600s; timeout is timeout_s + 5.0, so keep it tiny.
        timeout_s=0.5,
    )

    start = time.monotonic()
    result = jail.run_in_jail(policy)
    elapsed = time.monotonic() - start

    # Must return the documented timeout contract, not raise.
    assert result.returncode == 124
    assert result.argv == ("/bin/true",)
    # Bounded: must not block for the full 600s sleep.
    assert elapsed < 30.0

    # The backgrounded grandchild must have been reaped by the group kill: give
    # it a moment, then confirm the marker stops being refreshed.
    marker = tmp_path / "grandchild_alive"
    time.sleep(1.0)
    if marker.exists():
        first = marker.stat().st_mtime
        time.sleep(1.0)
        second = marker.stat().st_mtime
        assert first == second, "grandchild still alive after group kill"
