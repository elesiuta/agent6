# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""A model's command is handed back, never killed for taking too long.

A wall-clock timeout has to answer a question it cannot: whether a command that
has run twenty minutes is stuck or working. `[workflow].command_checkin_s`
replaces the kill with a hand-back, so the judgement goes to the model (or the
operator), and the command keeps running either way.

The operator's gate (`run_verify_command`) is deliberately NOT part of this: the
loop needs a verdict from it, not a handle.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import pytest

from agent6.config import Config
from agent6.tools.dispatch import ToolDispatcher
from agent6.tools.errors import ToolError


def _dispatcher(tmp_path: Path, checkin: float) -> ToolDispatcher:
    root = tmp_path / "repo"
    root.mkdir(exist_ok=True)
    session_dir = tmp_path / "session"
    session_dir.mkdir(exist_ok=True)
    cfg = Config.model_validate(
        {
            "sandbox": {"run_commands": "yes"},
            "workflow": {"command_checkin_s": checkin},
        }
    )
    return ToolDispatcher(
        root=root,
        config=cfg,
        isolation="none",
        session_dir=session_dir,
        use_jail_session=True,
    )


def _run(d: ToolDispatcher, script: str) -> dict[str, Any]:
    return d.dispatch("run_command", {"argv": ["/bin/sh", "-c", script]}).to_wire()


def test_a_command_that_finishes_is_an_ordinary_result(tmp_path: Path) -> None:
    d = _dispatcher(tmp_path, checkin=30.0)
    try:
        out = _run(d, "echo fast; exit 2")
    finally:
        d.close()
    assert out["returncode"] == 2
    assert out["stdout"] == "fast\n"
    assert "still_running" not in out
    assert "background_id" not in out


def test_a_command_outliving_the_checkin_comes_back_as_a_background_job(tmp_path: Path) -> None:
    """One ExecResult shape either way: `returncode` is null and a
    `background_id` names where the command went, so nothing has to branch on
    "a result OR a handle"."""
    d = _dispatcher(tmp_path, checkin=0.5)
    try:
        started = time.monotonic()
        out = _run(d, "echo starting; sleep 30; echo never")
        elapsed = time.monotonic() - started

        assert out["returncode"] is None
        assert out["still_running"] is True
        assert out["background_id"] == "bg1"
        # The output from before the hand-back comes back with it.
        assert out["stdout"] == "starting\n"
        assert elapsed < 10, "the hand-back waited for the command instead of returning"

        # It really is still running, and it is an ordinary background job now.
        read = d.dispatch("read_background", {"id": "bg1"}).to_wire()
        assert "running" in str(read["shells"])
        assert "starting" in str(read["output"])
        stopped = d.dispatch("stop_background", {"id": "bg1"}).to_wire()
        assert "stopped" in str(stopped["shells"])
    finally:
        d.close()


def test_a_zero_checkin_waits_for_the_command(tmp_path: Path) -> None:
    """`0` disables the hand-back: correct when a human is watching and can
    interrupt, and the path a run with no background roster falls back to."""
    d = _dispatcher(tmp_path, checkin=0.0)
    try:
        out = _run(d, "sleep 1; echo waited")
    finally:
        d.close()
    assert out["returncode"] == 0
    assert out["stdout"] == "waited\n"
    assert "background_id" not in out


def test_the_verify_gate_is_never_handed_back(tmp_path: Path) -> None:
    """The operator's gate must return a verdict; a handle would leave the loop
    with nothing to decide on."""
    root = tmp_path / "repo"
    root.mkdir(exist_ok=True)
    session_dir = tmp_path / "session"
    session_dir.mkdir(exist_ok=True)
    cfg = Config.model_validate(
        {
            "sandbox": {"run_commands": "yes"},
            "workflow": {
                "command_checkin_s": 0.5,
                "verify_command": ["/bin/sh", "-c", "sleep 2; echo verified"],
                "verify_timeout_s": 30.0,
            },
        }
    )
    d = ToolDispatcher(
        root=root, config=cfg, isolation="none", session_dir=session_dir, use_jail_session=True
    )
    try:
        out = d.dispatch("run_verify_command", {}).to_wire()
    finally:
        d.close()
    assert out["returncode"] == 0
    assert "verified" in out["stdout"]
    assert "background_id" not in out


@pytest.mark.parametrize("checkin", [0.5, 0.0])
def test_nothing_a_handed_back_command_started_outlives_the_run(
    tmp_path: Path, checkin: float
) -> None:
    """Teardown stops the roster, so a command that was handed back dies with
    the run exactly like one the model backgrounded itself."""
    d = _dispatcher(tmp_path, checkin=checkin)
    marker = tmp_path / "pid"
    try:
        _run(d, f"echo $$ > {marker}; sleep 0.2")
    finally:
        d.close()
    assert marker.exists()
    pid = int(marker.read_text().strip())
    try:
        state = Path(f"/proc/{pid}/stat").read_text().rsplit(") ", 1)[1].split()[0]
    except (OSError, IndexError):
        state = "gone"
    assert state in ("gone", "Z")


# --- one exec tool -----------------------------------------------------------


def test_background_true_returns_the_same_shape_immediately(tmp_path: Path) -> None:
    """`background: true` is a check-in of zero, so it is one parameter rather
    than a second tool with a second return shape."""
    d = _dispatcher(tmp_path, checkin=900.0)
    try:
        out = d.dispatch(
            "run_command", {"argv": ["/bin/sh", "-c", "echo up; sleep 30"], "background": True}
        ).to_wire()
        assert out["returncode"] is None
        assert out["still_running"] is True
        assert out["background_id"] == "bg1"
        # The command has only just started, so poll briefly for its first
        # line rather than assuming it has already reached the log.
        deadline = time.monotonic() + 5.0
        output = ""
        while time.monotonic() < deadline and "up" not in output:
            output = str(
                d.dispatch("read_background", {"id": "bg1", "wait_s": 0}).to_wire()["output"]
            )
            if "up" not in output:
                time.sleep(0.1)
        assert "up" in output
        d.dispatch("stop_background", {"id": "bg1"})
    finally:
        d.close()


def test_the_background_flag_replaced_the_second_tool(tmp_path: Path) -> None:
    """One exec tool plus read + kill, matching what models are trained on."""
    d = _dispatcher(tmp_path, checkin=900.0)
    try:
        names = d.available_tool_names()
        assert "run_background" not in names
        assert {"run_command", "read_background", "stop_background"} <= set(names)
    finally:
        d.close()


def test_a_read_only_mode_cannot_background(tmp_path: Path) -> None:
    """Only a session that edits owns a background command's lifetime; every
    other mode is a short read-only pass and would kill it at the end. Derived
    from the same tool set that withholds read_background there."""
    root = tmp_path / "repo"
    root.mkdir(exist_ok=True)
    session_dir = tmp_path / "session"
    session_dir.mkdir(exist_ok=True)
    cfg = Config.model_validate({"sandbox": {"run_commands": "yes"}})
    d = ToolDispatcher(
        root=root,
        config=cfg,
        isolation="none",
        mode="ask",
        session_dir=session_dir,
        use_jail_session=True,
    )
    try:
        with pytest.raises(ToolError, match="not available in ask mode"):
            d.dispatch("run_command", {"argv": ["/bin/echo", "hi"], "background": True})
    finally:
        d.close()


def test_an_operator_stop_cuts_a_wait_short(tmp_path: Path) -> None:
    """Stop is a marker file polled at a STEP boundary, and a tool call in
    flight reaches no boundary -- so a Stop pressed during a wait sat unread for
    the whole wait (measured: the full 10s of a 10s wait, and the default wait
    is 900). The wait was already a poll loop; it just needed a second reason
    to end."""
    from agent6.sessions.ipc import request_stop

    d = _dispatcher(tmp_path, checkin=900.0)
    session_dir = tmp_path / "session"
    try:
        d.dispatch("run_command", {"argv": ["/bin/sh", "-c", "sleep 60"], "background": True})
        request_stop(session_dir)
        started = time.monotonic()
        d.dispatch("read_background", {"id": "bg1", "wait_s": 10})
        waited = time.monotonic() - started
        assert waited < 3.0, f"the stop was ignored for {waited:.1f}s of a 10s wait"
        d.dispatch("stop_background", {"id": "bg1"})
    finally:
        d.close()


def test_a_wait_still_waits_when_nobody_asked_to_stop(tmp_path: Path) -> None:
    """The negative control: without a stop marker the wait runs to the
    command's end, or the early return above would be meaningless."""
    d = _dispatcher(tmp_path, checkin=900.0)
    try:
        d.dispatch("run_command", {"argv": ["/bin/sh", "-c", "sleep 2"], "background": True})
        started = time.monotonic()
        d.dispatch("read_background", {"id": "bg1", "wait_s": 30})
        waited = time.monotonic() - started
        assert 1.0 < waited < 15.0, f"waited {waited:.1f}s; expected to wait for the ~2s command"
    finally:
        d.close()


def test_an_operator_stop_hands_a_running_command_back_at_once(tmp_path: Path) -> None:
    """The sibling of the wait above, and the harder half.

    `read_background`'s wait is a poll loop this side owns. A SYNCHRONOUS
    `run_command` is not: the dispatcher blocks reading the launcher's answer
    pipe, so a Stop sat unread until the check-in elapsed -- measured at 18s of
    a 20s command, and the default check-in is 900. The launcher now takes a
    second reason to hand back, on its own pipe, because the request channel is
    in lockstep and this side is blocked on the answer to the very request
    being interrupted. The command is not killed: it becomes `bg<N>` exactly as
    the check-in would have made it.
    """
    from agent6.sessions.ipc import request_stop

    d = _dispatcher(tmp_path, checkin=900.0)
    session_dir = tmp_path / "session"
    try:
        request_stop(session_dir)
        started = time.monotonic()
        out = _run(d, "sleep 30")
        waited = time.monotonic() - started
        assert waited < 5.0, f"the stop was ignored for {waited:.1f}s of a 900s check-in"
        assert out["returncode"] is None and out["still_running"] is True
        assert out["background_id"] == "bg1"
        d.dispatch("stop_background", {"id": "bg1"})
    finally:
        d.close()


def test_a_command_runs_to_the_end_when_nobody_asked_to_stop(tmp_path: Path) -> None:
    """The negative control: no marker, so the same command returns its own
    result and no handle, or the early hand-back above would prove nothing."""
    d = _dispatcher(tmp_path, checkin=900.0)
    try:
        started = time.monotonic()
        out = _run(d, "sleep 2; echo done")
        waited = time.monotonic() - started
        assert out["returncode"] == 0
        # A finished command carries no handle at all, not an empty one.
        assert "background_id" not in out and "still_running" not in out
        assert "done" in out["stdout"]
        assert waited > 1.5, f"returned in {waited:.1f}s; the command sleeps 2"
    finally:
        d.close()


def test_a_plan_or_ask_command_runs_bounded_instead_of_handing_back(tmp_path: Path) -> None:
    """plan and ask permit `run_command` but withhold `read_background` and
    `stop_background`, so a check-in hand-back there left the model holding a
    handle it could neither poll nor stop, with the command running until
    teardown. Where the hand-back is unusable the command runs bounded."""
    root = tmp_path / "repo"
    root.mkdir(exist_ok=True)
    session_dir = tmp_path / "session"
    session_dir.mkdir(exist_ok=True)
    cfg = Config.model_validate(
        {"sandbox": {"run_commands": "yes"}, "workflow": {"command_checkin_s": 0.3}}
    )
    d = ToolDispatcher(
        root=root,
        config=cfg,
        isolation="none",
        session_dir=session_dir,
        use_jail_session=True,
        mode="plan",
    )
    out = _run(d, "sleep 1; echo done")
    assert out.get("background_id") is None and not out.get("still_running"), out
    assert out["returncode"] == 0 and "done" in out["stdout"]
