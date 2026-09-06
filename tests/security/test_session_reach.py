# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""`agent6 exec` and `agent6 forward`: the operator's way into a run's network.

A run's session network has no way in from outside, which is the point and also
the problem: the dev server the agent started is invisible to the person who
asked for it. These two commands are the door, and they are the operator's --
the model reaches none of this.
"""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

from agent6.config import Config
from agent6.sandbox.jail import SessionNetwork
from agent6.sessions.ipc import listening_ports, write_session_netns_pid, write_worker_pid
from agent6.sessions.layout import SessionLayout
from agent6.tools.dispatch import ToolDispatcher
from agent6.ui.cli.net_cmds import (
    SessionNetworkUnavailable,
    exec_in_session,
    join_session_network,
)

pytestmark = pytest.mark.needs_namespaces

_PORT = 48411


def _serving(
    tmp_path: Path, port: int, session_id: str
) -> tuple[ToolDispatcher, SessionNetwork, SessionLayout]:
    """A live session holding a dev server on `port`, laid out as a run is so
    the CLI's own id resolution finds it."""
    layout = SessionLayout(state_dir=tmp_path, session_id=session_id, subdir="runs")
    session_dir = layout.session_dir
    session_dir.mkdir(parents=True, exist_ok=True)
    write_worker_pid(session_dir, os.getpid())  # exec joins a live run only
    net = SessionNetwork.open()
    write_session_netns_pid(session_dir, net.holder_pid)
    dispatcher = ToolDispatcher(
        root=tmp_path,
        config=Config.model_validate({"sandbox": {"run_commands": "yes"}}),
        isolation="strict",
        session_dir=session_dir,
        use_jail_session=True,
        session_net=net,
    )
    serve = (
        "import http.server,socketserver;"
        f"socketserver.TCPServer(('127.0.0.1',{port}),"
        "http.server.SimpleHTTPRequestHandler).serve_forever()"
    )
    dispatcher.dispatch(
        "run_command", {"argv": ["/usr/bin/python3", "-c", serve], "background": True}
    )
    time.sleep(2.0)
    return dispatcher, net, layout


def test_the_ports_a_run_serves_are_visible_only_from_inside(tmp_path: Path) -> None:
    """Nothing on this machine can see the dev server, so listing its ports has
    to happen in the run's network -- and stop working when the run ends."""
    dispatcher, net, layout = _serving(tmp_path, _PORT, "serving-1")
    try:
        assert listening_ports(layout.session_dir) == [_PORT]
        with pytest.raises(OSError):  # not reachable from this machine
            socket.create_connection(("127.0.0.1", _PORT), timeout=2).close()
    finally:
        dispatcher.close()
        net.close()
    assert listening_ports(layout.session_dir) == [], "the network outlived the run"


def test_exec_runs_where_the_agent_runs(tmp_path: Path) -> None:
    """The whole claim of `agent6 exec`: what you see is what the agent sees."""
    dispatcher, net, layout = _serving(tmp_path, _PORT + 1, "exec-1")
    try:
        code = exec_in_session(
            layout,
            Config.model_validate({"sandbox": {"run_commands": "yes"}}),
            tmp_path,
            (
                "/usr/bin/python3",
                "-c",
                "import urllib.request;urllib.request.urlopen("
                f"'http://127.0.0.1:{_PORT + 1}/', timeout=4)",
            ),
        )
        assert code == 0, "exec could not reach the run's dev server"
    finally:
        dispatcher.close()
        net.close()


def test_joining_a_session_without_a_network_says_why(tmp_path: Path) -> None:
    """A run on the host network has nothing to join, and the refusal names the
    setting rather than failing with a bare errno."""
    with pytest.raises(SessionNetworkUnavailable, match=r"sandbox\.network"):
        join_session_network(tmp_path)


def test_forward_bridges_a_port_to_this_machine(tmp_path: Path) -> None:
    """The dev-server ergonomic: a plain client on this machine reaches a server
    that only exists inside the run."""
    dispatcher, net, layout = _serving(tmp_path, _PORT + 2, "fwd-1")
    local = _PORT + 100
    bridge = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import sys;sys.path.insert(0,'src');"
            "from pathlib import Path;"
            "from agent6.sessions.layout import SessionLayout;"
            "from agent6.ui.cli.net_cmds import forward;"
            f"lay=SessionLayout(state_dir=Path({str(tmp_path)!r}),"
            f" session_id={layout.session_id!r}, subdir='runs');"
            f"forward(lay, {_PORT + 2}, {local})",
        ],
        cwd=Path(__file__).resolve().parents[2],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        deadline = time.time() + 20
        body = b""
        while time.time() < deadline and not body:
            try:
                with socket.create_connection(("127.0.0.1", local), timeout=2) as sock:
                    sock.sendall(b"GET / HTTP/1.0\r\n\r\n")
                    body = sock.recv(64)
            except OSError:
                time.sleep(0.5)
        assert b"HTTP/1.0 200" in body, f"the bridge served nothing: {body!r}"
    finally:
        bridge.kill()
        bridge.wait(timeout=10)
        dispatcher.close()
        net.close()


def test_forward_refuses_a_run_with_no_network_instead_of_waiting(tmp_path: Path) -> None:
    """It used to bind the local port and block in accept(), then drop each
    connection in silence -- the join happens in the per-connection child, so
    nothing failed until someone tried to use it. A bridge to nowhere must say
    so before it looks like a bridge."""
    import io

    from agent6.ui.cli.net_cmds import forward

    layout = SessionLayout(state_dir=tmp_path, session_id="no-net", subdir="runs")
    layout.session_dir.mkdir(parents=True)
    out = io.StringIO()
    assert forward(layout, 3000, 3000, out=out) == 2
    assert "network" in out.getvalue()  # the refusal names the missing network


def test_forward_stops_when_its_session_ends(tmp_path: Path) -> None:
    """A bridge that outlives its run keeps accepting connections and dropping
    them, which reads as a broken server rather than a finished session.
    Observed against a real run: the run ended, `ss` still showed the listener,
    and curl got nothing."""
    import io
    import threading

    from agent6.sandbox.jail import SessionNetwork
    from agent6.sessions.ipc import clear_session_netns_pid, write_session_netns_pid
    from agent6.ui.cli.net_cmds import forward

    layout = SessionLayout(state_dir=tmp_path, session_id="ends-mid-forward", subdir="runs")
    layout.session_dir.mkdir(parents=True)
    net = SessionNetwork.open()
    write_session_netns_pid(layout.session_dir, net.holder_pid)
    out = io.StringIO()
    result: list[int] = []
    thread = threading.Thread(
        target=lambda: result.append(forward(layout, _PORT + 6, _PORT + 106, out=out)),
        daemon=True,
    )
    thread.start()
    time.sleep(1.0)
    assert thread.is_alive(), "the bridge should still be waiting while the run lives"
    net.close()
    clear_session_netns_pid(layout.session_dir)  # the run's teardown
    thread.join(timeout=15)
    assert not thread.is_alive(), "the bridge outlived its session"
    assert result == [0] and "ended" in out.getvalue(), out.getvalue()
