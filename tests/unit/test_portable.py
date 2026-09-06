# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Tests for agent6.portable cross-platform primitives.

These run on every platform: the lock helper guards the machine + graph
single-writer invariants on Windows/macOS/Linux alike.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from agent6 import portable
from agent6.portable import atomic_write, fsync_dir, lock_exclusive, unlock


def test_toml_basic_string_round_trips_including_control_chars() -> None:
    # The rendered literal must be a complete, parseable TOML basic string that
    # round-trips to the input. Control chars are illegal raw in a basic string,
    # so an unescaped one writes a file that fails to parse on the next read.
    import tomllib

    for raw in ["plain", 'has "quotes"', "back\\slash", "tab\tnew\nline\r", "ctrl\x01\x1f\x7f"]:
        rendered = portable.toml_basic_string(raw)
        assert tomllib.loads(f"k = {rendered}")["k"] == raw


def test_exclusive_lock_blocks_second_holder(tmp_path: Path) -> None:
    lock_path = tmp_path / "x.lock"
    fd1 = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o644)
    fd2 = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o644)
    try:
        lock_exclusive(fd1, blocking=False)
        with pytest.raises(OSError):
            lock_exclusive(fd2, blocking=False)
        unlock(fd1)
        # Now the second holder can take it.
        lock_exclusive(fd2, blocking=False)
        unlock(fd2)
    finally:
        os.close(fd1)
        os.close(fd2)


def test_lock_then_unlock_is_reusable(tmp_path: Path) -> None:
    lock_path = tmp_path / "y.lock"
    fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o644)
    try:
        lock_exclusive(fd, blocking=True)
        unlock(fd)
        lock_exclusive(fd, blocking=True)
        unlock(fd)
    finally:
        os.close(fd)


def test_fsync_dir_does_not_raise(tmp_path: Path) -> None:
    # Should be a durable no-op-or-fsync regardless of platform.
    fsync_dir(tmp_path)


def test_atomic_write_fsyncs_file_and_parent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fsynced_files: list[int] = []
    fsynced_dirs: list[Path] = []

    def record_file_fsync(fd: int) -> None:
        fsynced_files.append(fd)

    def record_dir_fsync(path: Path) -> None:
        fsynced_dirs.append(path)

    monkeypatch.setattr(portable.os, "fsync", record_file_fsync)
    monkeypatch.setattr(portable, "fsync_dir", record_dir_fsync)

    target = tmp_path / "state.json"
    atomic_write(target, '{"ok": true}')

    assert target.read_text(encoding="utf-8") == '{"ok": true}'
    assert not (tmp_path / "state.json.tmp").exists()
    assert fsynced_files, "temp file must be fsync'd before replace"
    assert fsynced_dirs == [tmp_path], "parent dir must be fsync'd after replace"


def test_atomic_write_fsyncs_new_parent_dirs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fsynced_dirs: list[Path] = []

    def record_dir_fsync(path: Path) -> None:
        fsynced_dirs.append(path)

    monkeypatch.setattr(portable, "fsync_dir", record_dir_fsync)

    target = tmp_path / "new" / "nested" / "state.json"
    atomic_write(target, "ok")

    assert target.read_text(encoding="utf-8") == "ok"
    assert fsynced_dirs == [tmp_path, tmp_path / "new", tmp_path / "new" / "nested"]


def test_atomic_write_concurrent_writers_do_not_share_temp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import threading

    writers = 8
    target = tmp_path / "state.json"
    barrier = threading.Barrier(writers)
    errors: list[BaseException] = []

    def wait_at_file_fsync(_fd: int) -> None:
        barrier.wait(timeout=5.0)

    def noop_fsync_dir(_path: Path) -> None:
        return None

    def write_payload(n: int) -> None:
        try:
            atomic_write(target, f"payload-{n}")
        except Exception as exc:  # pragma: no cover - assertion reports it
            errors.append(exc)

    monkeypatch.setattr(portable.os, "fsync", wait_at_file_fsync)
    monkeypatch.setattr(portable, "fsync_dir", noop_fsync_dir)

    threads = [threading.Thread(target=write_payload, args=(n,)) for n in range(writers)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert errors == []
    assert target.read_text(encoding="utf-8") in {f"payload-{n}" for n in range(writers)}


@pytest.mark.skipif(os.name == "nt", reason="POSIX permission bits")
def test_atomic_write_new_file_is_owner_only_under_restrictive_umask(tmp_path: Path) -> None:
    import stat

    # A new state file must not be widened to a hardcoded 0o644 (bypassing the
    # umask): these are per-user run/machine state, owner-only by default.
    old = os.umask(0o077)
    try:
        target = tmp_path / "state.json"
        atomic_write(target, "{}")
        assert stat.S_IMODE(target.stat().st_mode) == 0o600
    finally:
        os.umask(old)


@pytest.mark.skipif(os.name == "nt", reason="POSIX permission bits")
def test_atomic_write_preserves_existing_mode(tmp_path: Path) -> None:
    import stat

    target = tmp_path / "state.json"
    atomic_write(target, "first")
    target.chmod(0o640)
    atomic_write(target, "second")  # a re-publish must keep the file's own mode
    assert stat.S_IMODE(target.stat().st_mode) == 0o640


def test_locked_file_is_same_thread_reentrant(tmp_path: Path) -> None:
    """A transaction (write + revalidate + rollback) holds locked_file around
    per-write helpers that each take it too; flock on a second fd of the same
    file self-deadlocks the process, so the nested acquire must be a no-op.
    Run in a worker thread so a regression fails the join instead of hanging
    the suite."""
    import threading

    target = tmp_path / "c.toml"
    entered: list[str] = []

    def nested() -> None:
        with portable.locked_file(target):
            entered.append("outer")
            with portable.locked_file(target):
                entered.append("inner")

    t = threading.Thread(target=nested, daemon=True)
    t.start()
    t.join(timeout=5)
    assert entered == ["outer", "inner"]
    assert not t.is_alive()
    assert not (tmp_path / "c.toml.lock").exists()  # released and cleaned once


def test_locked_file_blocks_other_threads_despite_reentrancy(tmp_path: Path) -> None:
    """Reentrancy is per-thread only: a second thread must still queue.

    The holder records its exit INSIDE the block. A lock orders critical
    sections, not the bookkeeping after them: release unlinks the lock file
    before unlocking it, so a contender arriving during teardown takes a fresh
    file and never queues, and its append can land first.
    """
    import threading
    import time

    target = tmp_path / "c.toml"
    order: list[str] = []
    inside = threading.Event()
    release = threading.Event()

    def holder() -> None:
        with portable.locked_file(target):
            order.append("holder-in")
            inside.set()
            release.wait(timeout=5)
            order.append("holder-out")

    def contender() -> None:
        inside.wait(timeout=5)
        with portable.locked_file(target):
            order.append("contender-in")

    t1 = threading.Thread(target=holder, daemon=True)
    t2 = threading.Thread(target=contender, daemon=True)
    t1.start()
    t2.start()
    inside.wait(timeout=5)
    time.sleep(0.1)  # give the contender time to reach (and block on) the lock
    assert order == ["holder-in"]
    release.set()
    t1.join(timeout=5)
    t2.join(timeout=5)
    assert order == ["holder-in", "holder-out", "contender-in"]


@pytest.mark.skipif(os.name == "nt", reason="POSIX symlinks + O_NOFOLLOW")
def test_locked_file_refuses_a_symlinked_lock_and_fails_open(tmp_path: Path) -> None:
    """A planted symlink at the predictable ``<name>.lock`` path must never be
    followed: an earlier build chowned that fd as root, turning the lock into
    an arbitrary-file ownership-transfer primitive under ``sudo``. O_NOFOLLOW
    refuses it and the body runs unserialized (fail open); the symlink target
    is neither opened for write, chowned, nor unlinked."""
    target = tmp_path / "config.toml"
    secret = tmp_path / "root_secret"
    secret.write_text("do-not-touch", encoding="utf-8")
    (tmp_path / "config.toml.lock").symlink_to(secret)

    ran = False
    with portable.locked_file(target):
        ran = True
    assert ran
    assert secret.read_text(encoding="utf-8") == "do-not-touch"  # target untouched
    assert secret.exists()
    assert (tmp_path / "config.toml.lock").is_symlink()  # the symlink itself survives


@pytest.mark.skipif(os.name == "nt", reason="POSIX symlinks")
def test_locked_file_reentrant_across_atomic_write_of_symlinked_target(tmp_path: Path) -> None:
    """Keying reentrancy on target.resolve() self-deadlocked a symlinked config:
    the first atomic_write replaces the symlink with a regular file, so
    resolve() (and the key) changed and the second nested acquire blocked on
    the thread's own outer lock. The parent-resolved key is stable across the
    write. Run in a worker thread so a regression fails the join, not hangs."""
    import threading

    real = tmp_path / "real.toml"
    real.write_text("x = 0\n", encoding="utf-8")
    target = tmp_path / "config.toml"
    target.symlink_to(real)  # a dotfiles-style symlinked config
    entered: list[str] = []

    def txn() -> None:
        with portable.locked_file(target):
            entered.append("leaf-1")
            atomic_write(target, "x = 1\n")  # replaces the symlink with a regular file
            with portable.locked_file(target):
                entered.append("leaf-2")

    t = threading.Thread(target=txn, daemon=True)
    t.start()
    t.join(timeout=5)
    assert entered == ["leaf-1", "leaf-2"]
    assert not t.is_alive()
    assert not target.is_symlink()  # the write did replace the symlink


@pytest.mark.skipif(os.name == "nt", reason="POSIX permission bits")
def test_locked_file_fails_open_on_an_unreopenable_lock(tmp_path: Path) -> None:
    """A stale lock a killed ``sudo`` writer left root-owned is unreopenable by
    a later non-root process; rather than wedge every later write, the guard
    fails open. Simulated portably with a 0000 lock the owner cannot open."""
    target = tmp_path / "config.toml"
    lock = tmp_path / "config.toml.lock"
    lock.write_text("", encoding="utf-8")
    lock.chmod(0o000)
    try:
        ran = False
        with portable.locked_file(target):
            ran = True
        assert ran  # body ran despite the unopenable lock, no wedge, no raise
    finally:
        lock.chmod(0o600)


def test_locked_file_reports_acquisition(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The fail-open contract is unchanged; the yielded bool exists so a
    transaction that would restore a whole-file snapshot on failure can tell
    a real serialized cycle from a fictional one (a stale root-owned .lock)."""
    import agent6.portable as portable_mod

    target = tmp_path / "c.toml"
    with portable_mod.locked_file(target) as held:
        assert held is True
        with portable_mod.locked_file(target) as inner:
            assert inner is True  # reentrant: the outer acquisition's truth

    def _no_lock(_p: Path) -> int | None:
        return None

    monkeypatch.setattr(portable_mod, "_acquire_lock", _no_lock)
    with portable_mod.locked_file(target) as held:
        assert held is False
        with portable_mod.locked_file(target) as inner:
            assert inner is False


def test_a_cut_stderr_tail_says_it_was_cut() -> None:
    """The claude_code provider's copy cut a diagnostic at 400 chars with no
    marker, so a partial failure read as a complete one; the MCP client's
    version marks the cut and starts at a line, and is now the one owner."""
    from agent6.portable import stderr_tail

    keep = [(f"line {i}: " + "x" * 60 + "\n").encode() for i in range(20)]
    tail = stderr_tail(keep, limit=200)
    assert tail.startswith("…[agent6: ") and "earlier chars cut]" in tail
    assert tail.endswith("x" * 60)
    assert stderr_tail([b"short\n"]) == "short"
