# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""JailSession.open on a launcher that dies during setup: the failure is a
JailUnavailableError with the launcher's stderr, the child is reaped, and
every pipe is closed at the failure site. Abandoning the Popen instead left
the stdin writer to garbage collection, which retried the flush against the
dead peer and raised unraisable BrokenPipeError noise into run logs (seen in
two bench legs), plus a zombie launcher for the rest of the process."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any, cast

import pytest

from agent6.sandbox import jail
from agent6.sandbox.jail import JailPolicy, JailSession, JailUnavailableError


def _fake_binary(tmp_path: Path, script: str) -> Path:
    fake = tmp_path / "agent6-jail"
    fake.write_text(f"#!/bin/sh\n{script}\n")
    fake.chmod(0o755)
    return fake


def _recording_popen(monkeypatch: pytest.MonkeyPatch) -> list[subprocess.Popen[bytes]]:
    seen: list[subprocess.Popen[bytes]] = []
    real = subprocess.Popen

    def rec(*args: Any, **kwargs: Any) -> subprocess.Popen[bytes]:
        p = cast("subprocess.Popen[bytes]", real(*args, **kwargs))
        seen.append(p)
        return p

    monkeypatch.setattr(jail.subprocess, "Popen", rec)
    return seen


def test_a_launcher_dead_at_setup_is_reaped_with_its_pipes_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(jail, "_require_jail_binary", lambda: _fake_binary(tmp_path, "exit 7"))
    seen = _recording_popen(monkeypatch)
    with pytest.raises(JailUnavailableError, match="died during setup"):
        JailSession.open(JailPolicy(cwd=tmp_path, argv=("/usr/bin/true",)))
    (proc,) = seen
    assert proc.returncode is not None  # reaped: no zombie outlives the failure
    assert proc.stdin is not None and proc.stdin.closed
    assert proc.stdout is not None and proc.stdout.closed
    assert proc.stderr is not None and proc.stderr.closed


def test_a_spec_write_failure_is_the_same_setup_death(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """EPIPE at the spec write/flush (the launcher died before reading) is
    JailUnavailableError like the EOF case, never a raw OSError, and the
    abandoned child is still reaped."""
    monkeypatch.setattr(jail, "_require_jail_binary", lambda: _fake_binary(tmp_path, "exit 7"))
    seen = _recording_popen(monkeypatch)

    class BoomWriter:
        closed = False

        def write(self, b: bytes) -> int:
            return len(b)

        def flush(self) -> None:
            raise BrokenPipeError(32, "Broken pipe")

        def close(self) -> None:
            self.closed = True

    real = jail.subprocess.Popen

    def boom(*args: Any, **kwargs: Any) -> subprocess.Popen[bytes]:
        p = cast("subprocess.Popen[bytes]", real(*args, **kwargs))
        assert p.stdin is not None
        p.stdin.close()
        p.stdin = BoomWriter()  # pyright: ignore[reportAttributeAccessIssue]
        return p

    monkeypatch.setattr(jail.subprocess, "Popen", boom)
    with pytest.raises(JailUnavailableError, match="died during setup"):
        JailSession.open(JailPolicy(cwd=tmp_path, argv=("/usr/bin/true",)))
    (proc,) = seen
    assert proc.returncode is not None
