# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""One serving launcher at EVERY isolation level, `none` included.

A session used to be strict-only, so hardened paid Landlock + seccomp setup on
every command and `none` never reached the launcher at all: three execution
paths for one lifecycle. Only strict has the PID namespace, so the two bounds
it provided for free are explicit elsewhere -- the launcher sweeps what it
backgrounded when its request channel closes, and the Python side sweeps each
command's escapees.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import time
from pathlib import Path

import pytest

from agent6.sandbox.jail import JailSession
from agent6.types import CommandResult, IsolationLevel, JailPolicy

# hardened and none need no namespaces; the strict case is marked per-test.
_NO_NAMESPACE_LEVELS: tuple[IsolationLevel, ...] = ("hardened", "none")


def _session(cwd: Path, isolation: IsolationLevel) -> JailSession:
    return JailSession.open(
        JailPolicy(cwd=cwd, argv=("true",), isolation=isolation, network="host", timeout_s=30.0)
    )


def _running(pid: int) -> bool:
    """Live, not merely present. `os.kill(pid, 0)` succeeds for a ZOMBIE, and
    the agent is a subreaper, so a swept grandchild lingers unreaped and reads
    as alive to a signal probe."""
    try:
        state = Path(f"/proc/{pid}/stat").read_text().rsplit(") ", 1)[1].split()[0]
    except (OSError, IndexError):
        return False
    return state != "Z"


@pytest.mark.parametrize("isolation", _NO_NAMESPACE_LEVELS)
def test_a_session_serves_commands_without_namespaces(
    tmp_path: Path, isolation: IsolationLevel
) -> None:
    session = _session(tmp_path, isolation)
    try:
        res = session.run(("/bin/sh", "-c", "echo out; echo err >&2; exit 3"))
    finally:
        session.close()
    assert isinstance(res, CommandResult), "a run with no check-in always completes"
    assert res.returncode == 3
    assert res.stdout.strip() == "out"
    assert "err" in res.stderr


@pytest.mark.parametrize("isolation", _NO_NAMESPACE_LEVELS)
def test_backgrounding_works_without_a_pid_namespace(
    tmp_path: Path, isolation: IsolationLevel
) -> None:
    """The launcher used to refuse a background request outside strict, which
    is why those levels needed a launcher of their own per command."""
    session = _session(tmp_path, isolation)
    try:
        pid = session.start_background(("/bin/sh", "-c", "sleep 60"))
        assert session.status_background(pid).running
        session.stop_background(pid)
        assert not session.status_background(pid).running
    finally:
        session.close()


@pytest.mark.parametrize("isolation", _NO_NAMESPACE_LEVELS)
def test_a_backgrounded_command_dies_with_the_session(
    tmp_path: Path, isolation: IsolationLevel
) -> None:
    """strict's PID namespace does this by construction; without one the
    launcher sweeps the pids it started when its request channel closes, or a
    server would outlive the run that started it."""
    session = _session(tmp_path, isolation)
    pid = session.start_background(("/bin/sh", "-c", "sleep 60"))
    assert _running(pid)
    session.close()
    time.sleep(1.0)
    still = _running(pid)
    if still:  # never leave one behind, even on failure
        os.kill(pid, signal.SIGKILL)
    assert not still


@pytest.mark.parametrize("isolation", _NO_NAMESPACE_LEVELS)
def test_a_setsid_escapee_does_not_outlive_its_command(
    tmp_path: Path, isolation: IsolationLevel
) -> None:
    """A `setsid` child leaves the command's process group, so the launcher's
    killpg misses it. Per-command launchers used to sweep it; the session does
    the same on the levels with no PID namespace to do it for them."""
    session = _session(tmp_path, isolation)
    marker = tmp_path / "escapee.pid"
    try:
        session.run(("/bin/sh", "-c", f"setsid sh -c 'echo $$ > {marker}; sleep 60' & sleep 0.4"))
        assert marker.exists(), "the escapee never started; the test proves nothing"
        pid = int(marker.read_text().strip())
        still = _running(pid)
        if still:
            os.kill(pid, signal.SIGKILL)
        assert not still
    finally:
        session.close()


@pytest.mark.parametrize("isolation", _NO_NAMESPACE_LEVELS)
def test_a_backgrounded_setsid_daemon_dies_with_the_session(
    tmp_path: Path, isolation: IsolationLevel
) -> None:
    """The two halves together: a BACKGROUND command's `setsid` child leaves
    the group the launcher tracked, and outlives every per-command sweep. With
    no PID namespace nothing else bounded it, so it survived the run -- on
    hardened, with the host's network."""
    session = _session(tmp_path, isolation)
    marker = tmp_path / "daemon.pid"
    session.start_background(
        ("/bin/sh", "-c", f"setsid sh -c 'echo $$ > {marker}; sleep 60' & sleep 0.4")
    )
    time.sleep(1.0)
    assert marker.exists(), "the daemon never started; the test proves nothing"
    pid = int(marker.read_text().strip())
    assert _running(pid)

    session.close()

    time.sleep(1.0)
    still = _running(pid)
    if still:  # never leave one behind, even on failure
        os.kill(pid, signal.SIGKILL)
    assert not still


def test_the_unconfined_level_says_so_on_startup(tmp_path: Path) -> None:
    """`none` reaches the launcher now, so "the launcher ran" no longer implies
    "confinement was applied". It is loud instead: the caller surfaces this as
    `jail.degraded`."""
    session = _session(tmp_path, "none")
    try:
        assert "UNCONFINED" in session.startup_stderr
    finally:
        session.close()


def test_a_confined_level_stays_silent_on_startup(tmp_path: Path) -> None:
    """The unconfined warning must not fire for a level that does confine."""
    session = _session(tmp_path, "hardened")
    try:
        assert "UNCONFINED" not in session.startup_stderr
    finally:
        session.close()


# --- handing a still-running command back ------------------------------------


def _serve_raw(cwd: Path) -> subprocess.Popen[bytes]:
    """A serving launcher driven directly: the check-in is not on the Python
    session API yet, so the request is written by hand."""
    from agent6.sandbox.jail import _require_jail_binary  # pyright: ignore[reportPrivateUsage]

    proc = subprocess.Popen(
        [str(_require_jail_binary())],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    assert proc.stdin is not None and proc.stdout is not None
    spec = {"cwd": str(cwd), "argv": ["true"], "isolation": "none", "mode": "serve"}
    proc.stdin.write((json.dumps(spec) + "\n").encode())
    proc.stdin.flush()
    proc.stdout.readline()  # ready line
    return proc


def _ask(proc: subprocess.Popen[bytes], request: dict[str, object]) -> dict[str, object]:
    assert proc.stdin is not None and proc.stdout is not None
    proc.stdin.write((json.dumps(request) + "\n").encode())
    proc.stdin.flush()
    return json.loads(proc.stdout.readline())


def test_a_command_outliving_the_checkin_is_handed_back_not_killed(tmp_path: Path) -> None:
    """Whether a long command is stuck or working is a judgement, so it goes to
    whoever can make one. The output so far comes back with it, split by stream,
    and the log keeps filling after the hand-back.

    The two early lines are asserted as a SET, not a sequence: two writes
    microseconds apart arrive on two drain threads, so their relative order is
    not guaranteed -- the same limit a shared `2>&1` fd has, where the writer's
    own buffering decides. What IS guaranteed is that output produced a second
    later lands after both, which is the property the log is for.
    """
    logs = tmp_path / "logs"
    logs.mkdir()
    go = tmp_path / "go"
    proc = _serve_raw(tmp_path)
    try:
        answer = _ask(
            proc,
            {
                "kind": "run",
                "argv": [
                    "/bin/sh",
                    "-c",
                    # Gated rather than timed: the hand-back must land after the
                    # early output and before the late output, whatever the host
                    # is doing.
                    f"echo out1; echo err1 >&2; while [ ! -f {go} ]; do sleep 0.05; done;"
                    " echo out2; sleep 20",
                ],
                "timeout_s": 0,
                "checkin_s": 1.0,
                "log_dir": str(logs),
            },
        )
        assert answer["backgrounded"] is True
        # Per-stream renders, so these are exact however the two interleaved.
        assert answer["stdout"] == "out1\n"
        assert answer["stderr"] == "err1\n"
        pid = answer["pid"]
        assert isinstance(pid, int) and _running(pid), "the command was killed, not handed back"
        log = Path(str(answer["log"]))
        assert log.name == f"converted-{pid}.log"

        go.touch()
        deadline = time.monotonic() + 10.0
        lines: list[str] = []
        while time.monotonic() < deadline:
            lines = log.read_text(encoding="utf-8").split()
            if "out2" in lines:
                break
            time.sleep(0.05)
        assert set(lines) == {"out1", "err1", "out2"}, "the log stopped filling"
        assert lines[-1] == "out2", "output written after the hand-back landed out of order"

        # The launcher keeps serving, and the handed-back pid is pollable.
        assert (
            _ask(proc, {"kind": "run", "argv": ["/bin/echo", "next"], "timeout_s": 10})["stdout"]
            == "next\n"
        )
        assert _ask(proc, {"kind": "status", "pid": pid})["running"] is True
        assert _ask(proc, {"kind": "stop", "pid": pid})["stopped"] is True
    finally:
        assert proc.stdin is not None
        proc.stdin.close()
        proc.wait(timeout=10)


def test_a_non_positive_timeout_never_kills(tmp_path: Path) -> None:
    """The wall-clock kill is what the check-in replaces; a positive one still
    kills, so an operator gate that sets a number keeps its meaning."""
    proc = _serve_raw(tmp_path)
    try:
        unbounded = _ask(
            proc,
            {"kind": "run", "argv": ["/bin/sh", "-c", "sleep 1; echo survived"], "timeout_s": 0},
        )
        assert unbounded["returncode"] == 0
        assert unbounded["stdout"] == "survived\n"
        killed = _ask(
            proc, {"kind": "run", "argv": ["/bin/sh", "-c", "sleep 10"], "timeout_s": 0.5}
        )
        assert killed["returncode"] == 124
    finally:
        assert proc.stdin is not None
        proc.stdin.close()
        proc.wait(timeout=10)
