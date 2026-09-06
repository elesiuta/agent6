# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""`JailSession.close()` must never propagate an exception at teardown.

Found via the wheel CI-mirror leg on Python 3.12: close() used to `stdin.close()`
then `communicate()`, whose flush re-hits the now-closed pipe and raises
`ValueError: flush of closed file`. Python 3.14 (the dev interpreter) tolerates
that, so no gate caught it -- but AGENTS.md supports 3.12+, where it was an
unhandled crash in `ToolDispatcher.close()`. These tests use a fake launcher
proc, so they need no namespaces and run on every interpreter.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any

import pytest

from agent6.sandbox.jail import JailSession, JailUnavailableError


class _FakeProc:
    """A launcher Popen stand-in whose communicate() raises like 3.12's does."""

    def __init__(self, *, communicate_raises: BaseException | None, alive_after: bool) -> None:
        self.pid = 424242
        self.stdin = None  # close() must not depend on stdin being present
        self._communicate_raises = communicate_raises
        self._alive_after = alive_after
        self.communicated = False

    def communicate(self, timeout: float | None = None) -> tuple[bytes, bytes]:
        self.communicated = True
        if self._communicate_raises is not None:
            raise self._communicate_raises
        return (b"", b"")

    def poll(self) -> int | None:
        return None if self._alive_after else 0


def _session(proc: Any, *, pid_namespaced: bool = True) -> JailSession:
    # A real pipe end for the interrupt channel: close() closes it, and a
    # bogus number would make that the failure under test instead of the flush.
    _, interrupt_w = os.pipe()
    return JailSession(
        _proc=proc,
        _binary=Path("/nonexistent"),
        _pid_namespaced=pid_namespaced,
        _interrupt_w=interrupt_w,
        _memory_limit_mb=0,
    )


def test_close_swallows_the_closed_stdin_flush_valueerror() -> None:
    proc = _FakeProc(communicate_raises=ValueError("flush of closed file"), alive_after=False)
    _session(proc).close()  # must not raise
    assert proc.communicated


def test_close_kills_a_launcher_that_outlived_the_drain(monkeypatch: pytest.MonkeyPatch) -> None:
    """A communicate() that times out leaves the launcher alive; close() then
    SIGKILLs its group. The kill must reach the pid, and close() must not raise
    even if the (already-exited) killpg errors."""
    killed: list[int] = []

    def _fake_killpg(pid: int, sig: int) -> None:
        killed.append(pid)

    monkeypatch.setattr("agent6.sandbox.jail.os.killpg", _fake_killpg)
    proc = _FakeProc(
        communicate_raises=subprocess.TimeoutExpired(cmd="jail", timeout=10.0),
        alive_after=True,
    )
    _session(proc).close()  # must not raise
    assert killed == [proc.pid]


def test_a_survivor_of_the_sweep_is_named_by_every_stop(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """`JailedProcess.close`, `LocalJob.stop` and `BackgroundJob.stop` discarded
    the sweep's survivors, so the MCP-server path and `stop_background` under
    `none` and `hardened` answered "stopped" over a process the sweep could
    not kill; `SessionJob.stop` under hardened never swept at all."""
    from agent6.sandbox.jail import (
        BackgroundJob,
        BackgroundStatus,
        JailedProcess,
        LocalJob,
        SessionJob,
    )

    def sweep(exclude: frozenset[int]) -> frozenset[int]:
        return frozenset({4321})

    def no_children() -> dict[int, int]:
        return {}

    monkeypatch.setattr("agent6.sandbox.jail._kill_escapees", sweep)
    monkeypatch.setattr("agent6.sandbox.jail._own_children", no_children)

    def sleeper() -> subprocess.Popen[bytes]:
        return subprocess.Popen(["sleep", "60"], start_new_session=True)

    proc = sleeper()
    assert "4321" in LocalJob(proc, tmp_path / "local").stop() and proc.poll() is not None
    proc = sleeper()
    (tmp_path / "bg").mkdir()  # the outcome dir the shell registry creates
    assert "4321" in BackgroundJob(proc, tmp_path / "bg").stop() and proc.poll() is not None
    # The launcher is gone, so its ending is recorded even though the sweep failed.
    assert json.loads((tmp_path / "bg" / "result.json").read_text())["stopped"] is True
    proc = sleeper()
    assert JailedProcess(proc).close() == frozenset({4321}) and proc.poll() is not None

    def stopped(
        self: JailSession, request: dict[str, object], *, interrupted: object = None
    ) -> dict[str, object]:
        return {"stopped": True, "returncode": 0}

    monkeypatch.setattr(JailSession, "_request", stopped)
    session = _session(_FakeProc(communicate_raises=None, alive_after=False), pid_namespaced=False)
    assert (
        "4321" in SessionJob(session, 7, tmp_path / "job", before=session.child_snapshot()).stop()
    )
    # A job the registry already settled (its command exited) still sweeps on stop.
    settled = SessionJob(session, 8, tmp_path / "job-settled", before=session.child_snapshot())
    settled._final = BackgroundStatus(running=False, returncode=0, error="")  # pyright: ignore[reportPrivateUsage]
    assert "4321" in settled.stop()
    namespaced = _session(_FakeProc(communicate_raises=None, alive_after=False))
    job2 = SessionJob(namespaced, 7, tmp_path / "job2", before=namespaced.child_snapshot())
    assert job2.stop() == ""  # the namespace bounds them


def test_a_survivor_of_the_sweep_fails_the_command_and_comes_back_from_close(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`run_in_jail` fails a command whose escapee the sweep could not kill;
    the session path every run_command takes discarded the sweep's answer in
    `run` and in `close`, so a process outlived the run in silence."""

    def sweep(exclude: frozenset[int]) -> frozenset[int]:
        return frozenset({4321})

    def no_children() -> dict[int, int]:
        return {}

    def answered(
        self: JailSession, request: dict[str, object], *, interrupted: object = None
    ) -> dict[str, object]:
        return {"returncode": 0, "stdout": "", "stderr": ""}

    monkeypatch.setattr("agent6.sandbox.jail._kill_escapees", sweep)
    monkeypatch.setattr("agent6.sandbox.jail._own_children", no_children)
    monkeypatch.setattr(JailSession, "_request", answered)
    session = _session(_FakeProc(communicate_raises=None, alive_after=False), pid_namespaced=False)
    with pytest.raises(JailUnavailableError, match="4321"):
        session.run(("true",))
    assert session.close() == frozenset({4321})
