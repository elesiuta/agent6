# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""`agent6 exec` and `agent6 forward`: reach into a live run's session network.

A run's commands share one network with no route off the box, which is what
lets the agent start a dev server and curl it. The same property means nothing
outside the run can reach that server -- including the operator. These two
commands are the way in, and they are the operator's, never the model's:
`exec` runs a command the way the agent would, `forward` bridges one of the
run's ports to a port on this machine so a browser can open it.

A join goes through the holder pid the run publishes (`netns.pid`). `forward`
always joins; `exec` joins only when the run's own commands took the session
network, so a `host` stamp keeps it on this machine's network even while the
run holds a netns for an MCP server scoped to it. Entering a network namespace
needs capabilities in the user namespace that owns it, so each join takes that
one first, exactly as the launcher does.
"""

from __future__ import annotations

import contextlib
import os
import selectors
import socket
import sys
from pathlib import Path
from typing import TextIO

from agent6.app._setup import detect_env
from agent6.config import Config
from agent6.sandbox.detect import resolve_isolation
from agent6.sandbox.jail import JailUnavailableError, SessionNetwork, run_in_jail
from agent6.sessions.ipc import read_session_netns_pid
from agent6.sessions.layout import SessionLayout
from agent6.sessions.manifest import ManifestError, read_manifest
from agent6.tools.policy import jail_policy
from agent6.types import NetworkMode
from agent6.ui.cli._common import error, refuse
from agent6.viewmodel import session_is_live, summarize_session_dir

_JOIN_ORDER = (("user", os.CLONE_NEWUSER), ("net", os.CLONE_NEWNET))


class SessionNetworkUnavailable(Exception):
    """The run has no session network to join, and why."""


def join_session_network(session_dir: Path) -> None:
    """Put THIS process in the run's session network. Irreversible: seccomp
    is not involved, but nothing here ever leaves a namespace it entered."""
    pid = read_session_netns_pid(session_dir)
    if pid is None:
        raise SessionNetworkUnavailable(
            "this session has no network of its own to join. A run only makes one"
            " under the strict isolation with sandbox.network = auto|session;"
            " with network = host its commands are already on this machine's."
        )
    for kind, flag in _JOIN_ORDER:
        try:
            fd = os.open(f"/proc/{pid}/ns/{kind}", os.O_RDONLY)
        except OSError as exc:
            raise SessionNetworkUnavailable(f"the session's network is gone: {exc}") from exc
        try:
            os.setns(fd, flag)
        except OSError as exc:
            raise SessionNetworkUnavailable(
                f"could not join the session's {kind} namespace: {exc}"
            ) from exc
        finally:
            os.close(fd)


def _pump(a: socket.socket, b: socket.socket) -> None:
    """Shuttle bytes both ways until either side hangs up."""
    sel = selectors.DefaultSelector()
    sel.register(a, selectors.EVENT_READ, b)
    sel.register(b, selectors.EVENT_READ, a)
    try:
        while True:
            for key, _ in sel.select():
                src, dst = key.fileobj, key.data
                assert isinstance(src, socket.socket)
                chunk = src.recv(65536)
                if not chunk:
                    return
                dst.sendall(chunk)
    except OSError:
        return
    finally:
        sel.close()


def no_session_network_reason(layout: SessionLayout) -> str:
    """Why `forward` finds no session network to reach into: the run is not
    live (its network lives only while its run does), or a live run made none
    (host network, or an isolation short of strict)."""
    if not session_is_live(layout.session_dir):
        word = summarize_session_dir(layout.session_dir).status
        return f"{layout.session_id} is {word}; a session network exists only while its run does."
    return (
        f"{layout.session_id} has no network of its own to reach into. A run only makes one"
        " under strict isolation, for commands with sandbox.network other than host or for"
        " an MCP server scoped to the session; otherwise its commands are already on this"
        " machine's."
    )


def forward(
    layout: SessionLayout, remote_port: int, local_port: int, out: TextIO = sys.stderr
) -> int:
    """Bridge `remote_port` inside the run to `local_port` on this machine.

    One forked child per connection: it joins the run's network and connects
    there, then shuttles bytes over the socket it inherited. A child cannot
    come back out of a namespace, and the parent must stay outside to keep
    accepting, so the fork is the bridge rather than a design flourish.
    """
    # Refuse before binding, not per connection: the join happens in the
    # per-connection child, so a bind-first flow prints "forwarding" and then
    # drops every connection in silence when there is no network to join.
    if read_session_netns_pid(layout.session_dir) is None:
        print(f"REFUSING: {no_session_network_reason(layout)}", file=out)
        return 2
    # Same number on both sides unless told otherwise: that is what `kubectl
    # port-forward 3000`, `docker -p 3000:3000` and `ssh -L` all mean, and it is
    # the number you are about to type into a browser. A random local port would
    # be the same syntax with a different meaning, which is the surprising kind
    # of different.
    local_port = local_port or remote_port
    listener = socket.socket()
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        listener.bind(("127.0.0.1", local_port))
    except OSError as exc:
        print(
            f"ERROR: cannot listen on 127.0.0.1:{local_port}: {exc}."
            " Pick another with --local-port.",
            file=out,
        )
        return 2
    listener.listen(16)
    # Wake up between connections so the loop can notice the run ending: a
    # bridge that outlives its session accepts connections and drops them,
    # which looks exactly like a broken server on the other side.
    listener.settimeout(2.0)
    bound = listener.getsockname()[1]
    print(
        f"[agent6] forwarding http://127.0.0.1:{bound} -> port {remote_port} inside"
        f" {layout.session_id}. Ctrl-C to stop.",
        file=out,
    )
    try:
        while True:
            try:
                conn, _ = listener.accept()
            except TimeoutError:
                if read_session_netns_pid(layout.session_dir) is None:
                    print(
                        f"[agent6] {layout.session_id} ended; nothing left to reach.",
                        file=out,
                    )
                    return 0
                continue
            child = os.fork()
            if child == 0:  # pragma: no cover - one process per connection
                listener.close()
                code = 0
                try:
                    join_session_network(layout.session_dir)
                    inside = socket.create_connection(("127.0.0.1", remote_port), timeout=10)
                    inside.settimeout(None)  # the 10 s bounds the connect, not the bridge
                    _pump(conn, inside)
                except (SessionNetworkUnavailable, OSError):
                    code = 1
                finally:
                    conn.close()
                os._exit(code)
            conn.close()  # the child owns it now
            with contextlib.suppress(ChildProcessError):  # reap finished bridges
                while os.waitpid(-1, os.WNOHANG)[0]:
                    pass
    except KeyboardInterrupt:
        return 0
    finally:
        listener.close()


def _stamped_policy(layout: SessionLayout) -> tuple[str, NetworkMode | None] | None:
    """The run's recorded (isolation, network), or None when unreadable or
    unstamped. An unknown network word reads as unset rather than guessing."""
    try:
        stamp = read_manifest(layout.session_dir).policy
    except ManifestError:
        return None
    if stamp.isolation not in ("strict", "hardened", "none"):
        return None
    # "auto" (the knob) and "" (unstamped) resolve as None: jail_policy
    # applies its own auto semantics, same as the run did.
    network: NetworkMode | None = (
        stamp.network if stamp.network in ("host", "session", "none") else None
    )
    return stamp.isolation, network


def exec_in_session(layout: SessionLayout, cfg: Config, cwd: Path, argv: tuple[str, ...]) -> int:
    """Run *argv* the way the run's own commands run: same jail, same network.

    The operator's command, not the model's, so it is not approved or logged as
    a tool call -- but it is confined identically, which is the point: what you
    see is what the agent sees.

    Unbounded (`timeout_s=0.0`): a foreground command in the operator's
    terminal, so Ctrl-C is the bound. The policy's default timeout would kill
    the long-lived dependency (a dev server, a tail) that `exec` is for, held
    open inside the run's network.
    """
    # A live run only: the help promises the run's own jail, and a finished
    # run's jail is gone with it (a fresh one built from its recorded policy
    # is a different place, at today's HEAD, with none of its processes).
    if not session_is_live(layout.session_dir):
        refuse(f"{no_session_network_reason(layout)}")
        return 2
    pid = read_session_netns_pid(layout.session_dir)
    # The RUN'S recorded isolation and network, not today's config: an
    # operator who changed [sandbox] since the run started still gets the
    # jail the run's own commands got (mounts stay config-derived; the help
    # says so). A manifest without the stamp falls back to the current
    # config with a warning naming the divergence risk.
    stamped = _stamped_policy(layout)
    if stamped is not None:
        isolation_word, network_word = stamped
        isolation = resolve_isolation(isolation_word, detect_env())
        # The recorded word, not the holder: a run whose commands took the host
        # network can still hold a session netns for an MCP server scoped to
        # it. An "auto" stamp reads as None and follows the holder, as the run did.
        network = network_word if network_word is not None else ("session" if pid else None)
    else:
        print(
            "[agent6] WARNING: this run recorded no launch policy; using the"
            " current config, which may differ from what the run's commands got.",
            file=sys.stderr,
        )
        isolation = resolve_isolation(cfg.sandbox.isolation, detect_env())
        network = "session" if pid else None
    try:
        policy = jail_policy(cwd, cfg, isolation, argv, network=network, timeout_s=0.0)
    except JailUnavailableError as exc:
        error(f"{exc}")
        return 2
    if policy.network == "session" and pid is None:
        # The run recorded the session network but holds none (an isolation
        # short of strict): refuse rather than open /proc/None.
        refuse(f"{no_session_network_reason(layout)}")
        return 2
    if policy.network == "session":
        # The run's network belongs to the run; borrow it through the holder
        # rather than making one of our own, which would be a different place.
        # The holder can exit between the read above and this open.
        userns_fd = -1
        try:
            userns_fd = os.open(f"/proc/{pid}/ns/user", os.O_RDONLY)
            borrowed = SessionNetwork(
                userns_fd=userns_fd,
                netns_fd=os.open(f"/proc/{pid}/ns/net", os.O_RDONLY),
                holder_pid=int(pid or 0),
            )
        except OSError as exc:
            if userns_fd >= 0:
                os.close(userns_fd)
            refuse(f"the session's network is gone: {exc}")
            return 2
    else:
        borrowed = None
    try:
        result = run_in_jail(policy, session_net=borrowed)
    except JailUnavailableError as exc:
        error(f"{exc}")
        return 2
    finally:
        if borrowed is not None:
            borrowed.close()
    sys.stdout.write(result.stdout)
    sys.stderr.write(result.stderr)
    return result.returncode
