# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Cross-platform primitives for the few places agent6 touches POSIX-only APIs.

Pure stdlib, no agent6 imports. Keeps the platform split contained in one
spot instead of scattering `sys.platform` checks through the graph and
machine journals. The sandbox itself remains Linux-only (see
`agent6.sandbox.detect.sandbox_available`), and native Windows is unsupported
(use WSL); this module keeps the platform-neutral plumbing (file locks,
durable renames) working so the agent can run unsandboxed on macOS.
"""

from __future__ import annotations

import contextlib
import os
import sys
import tempfile
import threading
from collections.abc import Generator
from pathlib import Path
from typing import IO

if sys.platform == "win32":
    import msvcrt
else:
    import fcntl


def lock_shared_nonblocking(fd: int) -> None:
    """Take a SHARED lock on an open file descriptor, or raise OSError when an
    exclusive holder has it. A probe that only asks "is someone writing?" takes
    this one: an exclusive probe excludes the very writer it is asking about,
    so a run acquiring in that window parked as if the checkout were busy.

    Windows has no shared range lock, so the probe there takes the exclusive
    one."""
    if sys.platform == "win32":
        lock_exclusive(fd, blocking=False)
        return
    fcntl.flock(fd, fcntl.LOCK_SH | fcntl.LOCK_NB)


def lock_exclusive(fd: int, *, blocking: bool) -> None:
    """Take an exclusive lock on an open file descriptor.

    When `blocking` is False and another process already holds the lock this
    raises `OSError` immediately. On POSIX this is an advisory whole-file lock
    via `flock(2)`; on Windows it is a mandatory one-byte range lock via
    `msvcrt.locking` (offset 0, which the OS happily locks past EOF).
    """
    if sys.platform == "win32":
        os.lseek(fd, 0, os.SEEK_SET)
        mode = msvcrt.LK_LOCK if blocking else msvcrt.LK_NBLCK
        msvcrt.locking(fd, mode, 1)
    else:
        flags = fcntl.LOCK_EX if blocking else fcntl.LOCK_EX | fcntl.LOCK_NB
        fcntl.flock(fd, flags)


def unlock(fd: int) -> None:
    """Release a lock previously taken by :func:`lock_exclusive`."""
    if sys.platform == "win32":
        os.lseek(fd, 0, os.SEEK_SET)
        msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
    else:
        fcntl.flock(fd, fcntl.LOCK_UN)


def _same_file(fd: int, path: Path) -> bool:
    try:
        a = os.fstat(fd)
        b = path.stat()
    except OSError:
        return False
    return (a.st_dev, a.st_ino) == (b.st_dev, b.st_ino)


# Lock paths the CURRENT THREAD holds via locked_file, for reentrancy.
_HELD_LOCKS = threading.local()


def _acquire_lock(lock_path: Path) -> int | None:
    """Open + flock *lock_path*, returning the held fd, or None when the lock
    cannot be taken (see :func:`locked_file`'s fail-open contract).

    `O_NOFOLLOW` refuses a planted symlink at the predictable lock path
    outright -- never open, chown, or write the thing it points at. Any other
    open/lock failure (a stale root-owned lock a non-root process can't
    reopen) also returns None: the lock is an optimization, never a
    correctness barrier, so a broken one is skipped, not followed or waited
    on."""
    flags = os.O_CREAT | os.O_RDWR | getattr(os, "O_CLOEXEC", 0)
    if sys.platform != "win32":
        flags |= os.O_NOFOLLOW  # a symlinked lock path -> ELOOP -> fail open
    while True:
        try:
            fd = os.open(lock_path, flags, 0o600)
        except OSError:
            return None
        try:
            lock_exclusive(fd, blocking=True)
            if sys.platform == "win32" or _same_file(fd, lock_path):
                return fd
            # The previous holder unlinked this inode after this open; a
            # fresh lock file may already be held by someone else -- retry.
            unlock(fd)
        except OSError:
            os.close(fd)
            return None
        except BaseException:
            os.close(fd)
            raise
        os.close(fd)


@contextlib.contextmanager
def locked_file(target: Path) -> Generator[bool]:
    """Serialize read-modify-write cycles on *target* across processes.

    Yields whether the lock is actually HELD, for the one caller class that
    must NOT act on a fiction of serialization -- a transaction that would restore a
    whole-file snapshot on failure can erase a concurrent writer's
    just-validated update when the cycle never was serialized, so it degrades
    to keep-and-warn instead (see `config.write.keep_or_rollback`).

    Blocks on a sibling `<name>.lock` file, NOT the target: atomic_write
    replaces the target's inode on publish, so a lock taken on the target
    itself would let a waiter queued on the orphaned old inode run
    concurrently with a fresh locker -- exactly the lost update this guards
    against.

    The lock is a concurrency optimization, never a correctness barrier
    (atomic_write already makes each publish all-or-nothing), so it FAILS
    OPEN. If the lock cannot be opened or locked -- a planted symlink
    (refused by `O_NOFOLLOW`), or a stale root-owned lock a killed `sudo`
    writer left that a later non-root process can't reopen -- the body runs
    unserialized rather than wedging or following the symlink. Worst case is
    an unserialized write, which atomic_write already keeps all-or-nothing; a
    lock failure is never a way to redirect or block a write.

    The lock file is unlinked on release (no residue in a config dir or repo
    worktree); the fstat/stat identity check after acquire detects a
    concurrent unlink and retries on the fresh file. Crash-safe: flock dies
    with the process. On Windows the lock file is left in place (an open
    locked file cannot be unlinked), so no identity check needed.

    Same-thread reentrant: a transaction (write + revalidate + rollback)
    holds the lock across its whole cycle while the per-write helpers it
    calls skip re-acquiring -- flock on a second fd of the same file would
    self-deadlock the process. Other threads still block. The reentrancy key
    is the lock path with its PARENT resolved: the parent dir survives an
    atomic_write of the target, but the target's own inode does not, so a
    symlinked config that a write replaces with a regular file does not shift
    the key mid-transaction.
    """
    _ensure_parent_dirs(target.parent)
    lock_path = target.with_name(target.name + ".lock")
    key = str(target.parent.resolve() / lock_path.name)
    held: dict[str, bool] = getattr(_HELD_LOCKS, "paths", {})
    if key in held:
        yield held[key]  # reentrant: the outer acquisition's truth applies
        return
    fd = _acquire_lock(lock_path)
    held[key] = fd is not None
    _HELD_LOCKS.paths = held
    try:
        yield fd is not None
    finally:
        held.pop(key, None)
        if fd is not None:
            # Unlink BEFORE unlock, while still the holder: waiters queued on
            # this inode then fail the identity check and requeue on the fresh
            # file. Unlock-first would let one win the orphaned inode while a
            # newcomer locks a recreated file -- two concurrent "holders".
            if sys.platform != "win32":
                with contextlib.suppress(OSError):
                    lock_path.unlink()
            with contextlib.suppress(OSError):
                unlock(fd)
            os.close(fd)


def fsync_dir(path: Path) -> None:
    """fsync a directory so a rename into it is durable.

    No-op on Windows, which has no directory file descriptors to fsync; the
    `MoveFileEx`/`ReplaceFile` semantics behind `Path.replace` already
    make the rename durable there.
    """
    if sys.platform == "win32":
        return
    fd = os.open(path, os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def atomic_write(path: Path, data: str | bytes) -> None:
    """Write data via temp file + durable rename.

    The temp file lives beside the target, is fsync'd before the rename, and the
    parent directory is fsync'd after the rename so a crash cannot lose the new
    directory entry on POSIX filesystems.
    """
    _ensure_parent_dirs(path.parent)
    fd = -1
    tmp_name = ""
    try:
        fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
        # Preserve an existing target's mode across a re-publish; a NEW file
        # keeps mkstemp's owner-only 0o600 (a hardcoded wider mode would bypass
        # the umask). These are per-user run/machine state files; owner-only is
        # the secure default.
        # chmod the fd before writing (the two mode-specific branches below only
        # differ in text vs binary, which pyright needs narrowed for `fh.write`).
        mode = _existing_mode(path)
        if sys.platform != "win32" and mode is not None:
            os.fchmod(fd, mode)
        if isinstance(data, bytes):
            with os.fdopen(fd, "wb") as fh:
                fd = -1
                fh.write(data)
                fh.flush()
                os.fsync(fh.fileno())
        else:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fd = -1
                fh.write(data)
                fh.flush()
                os.fsync(fh.fileno())
        Path(tmp_name).replace(path)
    except Exception:
        if fd >= 0:
            os.close(fd)
        if tmp_name:
            Path(tmp_name).unlink(missing_ok=True)
        raise
    fsync_dir(path.parent)


_TOML_BASIC_ESCAPES = {
    "\\": "\\\\",
    '"': '\\"',
    "\b": "\\b",
    "\t": "\\t",
    "\n": "\\n",
    "\f": "\\f",
    "\r": "\\r",
}


def toml_basic_string(value: str) -> str:
    """*value* as a TOML basic (double-quoted) string literal, quotes included.

    Escapes backslash, quote, the named control escapes, and `\\uXXXX` for any
    other control char. TOML basic strings forbid literal control chars, so an
    unescaped one (a newline in a pasted key, say) writes a file that fails to
    parse on read while the write reported success. The single owner of this
    escaping: config serialization, `config fill`, and secrets all share it so
    none can drift back to escaping only `\\` and `"`.
    """
    out: list[str] = []
    for ch in value:
        if ch in _TOML_BASIC_ESCAPES:
            out.append(_TOML_BASIC_ESCAPES[ch])
        elif ord(ch) < 0x20 or ch == "\x7f":
            out.append(f"\\u{ord(ch):04X}")
        else:
            out.append(ch)
    return '"' + "".join(out) + '"'


def _ensure_parent_dirs(parent: Path) -> None:
    missing: list[Path] = []
    cur = parent
    while not cur.exists():
        missing.append(cur)
        if cur.parent == cur:
            break
        cur = cur.parent
    parent.mkdir(parents=True, exist_ok=True)
    for directory in reversed(missing):
        fsync_dir(directory.parent)


def _existing_mode(path: Path) -> int | None:
    """The target's current permission bits, or None if it does not exist yet
    (the caller then leaves the temp file at mkstemp's owner-only 0o600)."""
    try:
        return path.stat().st_mode & 0o777
    except OSError:
        return None


# A child's stderr, bounded because the writer is third-party code: capturing
# it to a file let a hostile MCP server write 1.8 GB in three seconds.
STDERR_KEEP_BYTES = 8192


def drain_stderr(pipe: IO[bytes], keep: list[bytes], *, close: bool = False) -> None:
    """Read a child's stderr forever, keeping only the tail.

    Forever, because a pipe nobody reads stops the writer at 64 KB (a child
    that logs would wedge itself). The last STDERR_KEEP_BYTES, because the
    writer has no reason to be polite about volume. Read at the descriptor: a
    buffered pipe's read(4096) returns only at 4 KB or EOF, so what a live
    child said would reach a failure message only after it died. The drain is
    the pipe's only reader: bytes another reader had buffered would be
    skipped."""
    with contextlib.suppress(OSError, ValueError):
        while chunk := os.read(pipe.fileno(), 4096):
            keep.append(chunk)
            if len(keep) > 2:
                keep[:] = [b"".join(keep)[-STDERR_KEEP_BYTES:]]
    if close:
        pipe.close()


def stderr_tail(keep: list[bytes], limit: int = 400) -> str:
    """The last of what a child said, for a failure message: at most *limit*
    chars, cut at a line start, and marked when anything was dropped, so a
    partial diagnostic never reads as a complete one. Best-effort: a
    diagnostic must never raise over the failure it is describing."""
    text = b"".join(keep)[-STDERR_KEEP_BYTES:].decode(errors="replace").strip()
    if len(text) <= limit:
        return text
    tail = text[-limit:]
    nl = tail.find("\n")
    if 0 <= nl < len(tail) - 1:
        tail = tail[nl + 1 :]
    return f"…[agent6: {len(text) - len(tail)} earlier chars cut]\n{tail.strip()}"
