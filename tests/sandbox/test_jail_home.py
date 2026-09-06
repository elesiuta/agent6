# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The jail's HOME, live: strict's private one leaves nothing on the host, and
the persistent one is bound read-write at its real path with nothing beside
it."""

from __future__ import annotations

import os
import shlex
import shutil
import stat
import tempfile
from collections.abc import Iterator
from pathlib import Path

import pytest

from agent6.config import Config, SandboxConfig
from agent6.paths import jail_cache_home
from agent6.sandbox.jail import run_in_jail
from agent6.tools.policy import jail_policy

pytestmark = pytest.mark.needs_namespaces


@pytest.fixture
def cache_home(monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """A cache dir outside every other grant. pytest's tmp_path sits under
    /tmp, which hardened grants wholesale, so a HOME there would be writable
    with or without its own grant; /var/tmp is granted by nothing."""
    base = Path("/var/tmp")
    if not (base.is_dir() and os.access(base, os.W_OK)):
        pytest.skip("/var/tmp is not writable here")
    scratch = Path(tempfile.mkdtemp(prefix="agent6-jail-home-", dir=base))
    monkeypatch.setenv("XDG_CACHE_HOME", str(scratch / "cache"))
    try:
        yield scratch / "cache" / "agent6" / "home"
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


def _operator_home_probe() -> str:
    """A command that READS the operator's own home (a directory listing or a
    file open, which Landlock gates; a bare stat it does not) and succeeds on
    the host, so its failure inside the jail is a denial."""
    home = Path.home()
    if (home / ".ssh").is_dir():
        return f"ls {shlex.quote(str(home / '.ssh'))}"
    for name in (".bashrc", ".profile"):
        if (home / name).is_file():
            return f"cat {shlex.quote(str(home / name))}"
    pytest.skip("nothing in the operator's home to probe")


@pytest.mark.parametrize(
    ("isolation", "cfg"),
    [("strict", Config(sandbox=SandboxConfig(home="cache"))), ("hardened", Config())],
)
def test_the_persistent_home_is_writable_and_the_operators_home_is_not(
    tmp_path: Path, cache_home: Path, isolation: str, cfg: Config
) -> None:
    """`touch ~/probe` lands on the host cache dir; the operator's own home
    stays out of reach (strict: absent from the rootfs; hardened: no Landlock
    rule covers it). No preflight runs here: the policy builder alone creates
    the dir, as it does for `agent6 exec` and an MCP probe."""
    ws = tmp_path / "ws"
    ws.mkdir()
    assert jail_cache_home() == cache_home
    assert not cache_home.exists()
    script = (
        f'test "$HOME" = {shlex.quote(str(cache_home))} || exit 3; '
        "cd ~ && touch probe || exit 4; "
        f"{_operator_home_probe()} >/dev/null 2>&1 && exit 5; "
        "echo ok"
    )
    policy = jail_policy(ws, cfg, isolation, ("/bin/sh", "-c", script), network="none")  # pyright: ignore[reportArgumentType]
    assert stat.S_IMODE(cache_home.lstat().st_mode) == 0o700
    res = run_in_jail(policy)
    assert res.returncode == 0 and "ok" in res.stdout, (res.returncode, res.stdout, res.stderr)
    assert (cache_home / "probe").is_file()


def test_the_tmpfs_home_goes_with_the_run(tmp_path: Path, cache_home: Path) -> None:
    """strict's default HOME is created by the launcher inside the private
    /tmp: `cd ~` works, the write never reaches the host, and the cache dir
    is left alone."""
    res = run_in_jail(
        jail_policy(
            tmp_path,
            Config(),
            "strict",
            ("/bin/sh", "-c", 'test "$HOME" = /tmp/agent6-home && cd ~ && touch probe && echo ok'),
            network="none",
        )
    )
    assert res.returncode == 0 and "ok" in res.stdout, (res.returncode, res.stdout, res.stderr)
    assert not Path("/tmp/agent6-home/probe").exists()
    assert not cache_home.exists()
