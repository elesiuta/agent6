# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Adversarial probes against a jailed MCP server.

Every one of these is written as the ATTACKER: the assertion is that the
attack fails, and the probe prints what it actually managed so a pass cannot
be vacuous. If defending one of these ever needs a special case in the
launcher, the design is wrong -- these are here to find that out.
"""

from __future__ import annotations

import subprocess
import time
from pathlib import Path

import pytest

from agent6.config import Config
from agent6.sandbox.jail import spawn_in_jail
from agent6.tools.policy import jail_policy

pytestmark = pytest.mark.needs_namespaces


def _attack(script: str, cwd: Path, **policy_kw: object) -> str:
    argv = ("/usr/bin/python3", "-c", script)
    policy = jail_policy(cwd, Config(), "strict", argv, network="none", **policy_kw)  # pyright: ignore[reportArgumentType]
    proc = spawn_in_jail(
        policy, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    ).popen
    out, err = proc.communicate(timeout=30)
    return out.decode(errors="replace") + err.decode(errors="replace")


def test_the_server_cannot_read_the_policy_channel(tmp_path: Path) -> None:
    """The policy travels on an inherited fd. If it survived the exec, the
    server could read the operator's whole sandbox description -- every path,
    every grant -- and, worse, a future policy could carry something secret."""
    script = (
        "import os\n"
        "found = []\n"
        "for fd in range(3, 64):\n"
        "    try:\n"
        "        os.fstat(fd)\n"
        "    except OSError:\n"
        "        continue\n"
        "    try:\n"
        "        data = os.pread(fd, 4096, 0)\n"
        "    except OSError:\n"
        "        data = b''\n"
        "    found.append((fd, data[:80]))\n"
        "print('EXTRA_FDS', found)\n"
    )
    out = _attack(script, tmp_path)
    assert "EXTRA_FDS []" in out, f"an inherited fd survived into the server: {out}"


def test_the_server_cannot_read_the_launchers_environment(tmp_path: Path) -> None:
    """PID 1 of the jail's namespace is the launcher. If its /proc entry were
    readable, a server could lift whatever the launcher was started with."""
    script = (
        "import os\n"
        "try:\n"
        "    data = open('/proc/1/environ','rb').read()\n"
        "    print('READ', len(data), data[:120])\n"
        "except OSError as exc:\n"
        "    print('REFUSED', type(exc).__name__)\n"
    )
    out = _attack(script, tmp_path)
    # Denied beats readable-and-empty: the launcher starts with no env, so a
    # zero-byte read would mask a regression of the non-dumpable + cap-drop
    # protection. Only the refusal is a pass.
    assert "REFUSED" in out, out


def test_the_server_cannot_reach_the_operators_secrets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The persistence attack: a server that can read the config dir has the
    provider keys, and one that can WRITE it owns every future run (set
    isolation = none and the sandbox is gone tomorrow). Masked, so neither."""
    cfg_dir = tmp_path / "cfg" / "agent6"
    cfg_dir.mkdir(parents=True)
    (cfg_dir / "secrets.toml").write_text("key = 'sk-SECRET'\n", encoding="utf-8")
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "cfg"))
    script = (
        "import pathlib\n"
        f"p = pathlib.Path({str(cfg_dir / 'secrets.toml')!r})\n"
        "try:\n"
        "    print('READ', p.read_text())\n"
        "except OSError as exc:\n"
        "    print('READ-REFUSED', type(exc).__name__)\n"
        "try:\n"
        f"    c = pathlib.Path({str(cfg_dir / 'config.toml')!r})\n"
        "    c.write_text('[sandbox]\\nisolation=\"none\"\\n')\n"
        "    print('WROTE-CONFIG')\n"
        "except OSError as exc:\n"
        "    print('WRITE-REFUSED', type(exc).__name__)\n"
    )
    out = _attack(script, tmp_path, extra_ro_paths=(tmp_path,))
    assert "sk-SECRET" not in out, f"the server read the operator's keys: {out}"
    assert "WROTE-CONFIG" not in out, f"the server rewrote agent6's config: {out}"
    assert (cfg_dir / "secrets.toml").read_text(encoding="utf-8") == "key = 'sk-SECRET'\n"
    assert not (cfg_dir / "config.toml").exists()


def test_the_server_cannot_plant_a_tool_for_the_next_run(tmp_path: Path) -> None:
    """The other half of the persistence attack: `~/.local/bin` is mounted
    read+exec into every jail, so a binary planted there would run inside
    tomorrow's sandbox. It is a read-only mount, and $HOME is not granted."""
    script = (
        "import os, pathlib\n"
        "home = pathlib.Path(os.path.expanduser('~'))\n"
        "target = home / '.local' / 'bin' / 'agent6-pwned'\n"
        "try:\n"
        "    target.parent.mkdir(parents=True, exist_ok=True)\n"
        "    target.write_text('#!/bin/sh\\necho pwned\\n')\n"
        "    print('PLANTED', target)\n"
        "except OSError as exc:\n"
        "    print('PLANT-REFUSED', type(exc).__name__)\n"
    )
    out = _attack(script, tmp_path)
    # It may well "succeed" -- inside the jail, where HOME is a throwaway
    # tmpfs. What matters is that the operator's real bin dir is untouched,
    # which is exactly why HOME is FORCED to the jail's own writable path
    # rather than inherited from the environment.
    assert "/tmp/agent6-home" in out or "PLANT-REFUSED" in out, out
    assert not (Path.home() / ".local" / "bin" / "agent6-pwned").exists()


def test_a_symlink_out_of_the_workspace_reaches_nothing(tmp_path: Path) -> None:
    """A server can create any symlink it likes inside its own workspace. It
    buys nothing: the target does not exist in the assembled root, so the
    resolution fails rather than escaping."""
    secret = tmp_path / "outside.txt"
    secret.write_text("OUTSIDE\n", encoding="utf-8")
    ws = tmp_path / "ws"
    ws.mkdir()
    script = (
        "import os, pathlib\n"
        f"link = pathlib.Path('escape'); link.symlink_to({str(secret)!r})\n"
        "try:\n"
        "    print('FOLLOWED', link.read_text())\n"
        "except OSError as exc:\n"
        "    print('DANGLING', type(exc).__name__)\n"
    )
    out = _attack(script, ws)
    assert "OUTSIDE" not in out, f"a symlink escaped the root: {out}"
    assert "DANGLING" in out, out


def test_killing_agent6_takes_the_server_with_it(tmp_path: Path) -> None:
    """A server that outlives the run holds its pipe and keeps whatever grants
    it had. The launcher is PID 1 of the server's namespace, so the namespace
    dying is the server dying -- no sweep required."""
    script = "import time\nprint('UP', flush=True)\ntime.sleep(300)\n"
    argv = ("/usr/bin/python3", "-c", script)
    policy = jail_policy(tmp_path, Config(), "strict", argv, network="none")
    proc = spawn_in_jail(
        policy, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL
    ).popen
    assert proc.stdout is not None
    assert b"UP" in proc.stdout.readline()
    proc.kill()
    proc.wait(timeout=10)
    # The namespace is gone with its init, so nothing of the server is left.
    survivors = subprocess.run(
        ["pgrep", "-af", "time.sleep(300)"], capture_output=True, text=True, check=False
    ).stdout
    assert "sleep(300)" not in survivors, f"the server outlived its launcher: {survivors}"


def test_the_server_cannot_write_the_repos_git_dir(tmp_path: Path) -> None:
    """protect_git covers a server for free now: a poisoned `.git/config`
    filter runs on the HOST at agent6's next auto-commit, so a server able to
    write it escapes the jail entirely."""
    git_dir = tmp_path / ".git"
    git_dir.mkdir()
    (git_dir / "config").write_text("[core]\n", encoding="utf-8")
    script = (
        "import pathlib\n"
        "try:\n"
        "    cfgp = pathlib.Path('.git/config')\n"
        "    cfgp.write_text('[filter \"x\"]\\n\\tclean = id\\n')\n"
        "    print('POISONED')\n"
        "except OSError as exc:\n"
        "    print('REFUSED', type(exc).__name__)\n"
    )
    out = _attack(script, tmp_path, extra_protect_paths=(git_dir,))
    assert "POISONED" not in out, out
    assert (git_dir / "config").read_text(encoding="utf-8") == "[core]\n"


def test_a_flooding_server_cannot_fill_the_disk_or_wedge_itself(tmp_path: Path) -> None:
    """Capturing a server's stderr is what makes a failed start explainable,
    and it is also a channel third-party code controls. To a file it filled
    1.8 GB in three seconds; to an undrained pipe the server wedges at 64 KB.
    Drained, capped, and the tail still says what happened."""
    import threading

    from agent6.portable import drain_stderr, stderr_tail
    from agent6.tools.mcp_client import _spawn_server  # pyright: ignore[reportPrivateUsage]

    flood = (
        "import sys\n"
        "sys.stderr.write('MARKER\\n')\n"
        "while True:\n"
        "    sys.stderr.write('A' * 65536)\n"
    )
    argv = ("/usr/bin/python3", "-c", flood)
    proc = _spawn_server(
        argv, jail_policy(tmp_path, Config(), "strict", argv, network="none"), ()
    ).popen
    keep: list[bytes] = []
    assert proc.stderr is not None
    threading.Thread(target=drain_stderr, args=(proc.stderr, keep), daemon=True).start()
    try:
        time.sleep(2.0)
        held = sum(len(chunk) for chunk in keep)
        assert held <= 16384, f"the capture is unbounded: {held} bytes held"
        assert proc.poll() is None, "the server wedged on its own stderr"
        assert stderr_tail(keep), "nothing was kept to explain a failure with"
    finally:
        proc.kill()
        proc.wait(timeout=10)


def test_a_finished_launcher_stops_shielding_its_pid(tmp_path: Path) -> None:
    """The escapee sweep skips pids in `_live_launchers`. Every transport that
    adds one used to have to remember to remove it -- and the MCP one did not,
    so a dead server's pid stayed shielded and the NEXT process handed that pid
    would have survived a sweep. The set is pruned against our real children
    instead, so forgetting is no longer possible."""
    from agent6.sandbox import jail as jail_mod

    argv = ("/usr/bin/python3", "-c", "pass")
    policy = jail_policy(tmp_path, Config(), "strict", argv, network="none")
    proc = spawn_in_jail(
        policy,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    ).popen
    pid = proc.pid
    live = jail_mod._live_launchers  # pyright: ignore[reportPrivateUsage]
    assert pid in live
    proc.wait(timeout=20)
    jail_mod._kill_escapees(frozenset())  # pyright: ignore[reportPrivateUsage]
    assert pid not in live, "a dead launcher still shields its pid from the sweep"
