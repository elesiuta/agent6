# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""One authoritative writer per session dir: the `worker.lock` flock.

`agent6 run`/`resume`/`fork` drive one run's shared state (loop_state.json,
checkpoints, the curator DAG, the run branch). The run-level flock (the
analogue of `machine_lock`) refuses a second concurrent writer.
"""

from __future__ import annotations

import contextlib
import os
from pathlib import Path

from agent6.paths import checkout_root, mkdir_for_real_user, repo_id
from agent6.portable import lock_exclusive, lock_shared_nonblocking, unlock


def acquire_single_writer(session_dir: Path) -> int | None:
    """Take a non-blocking exclusive lock on `<session-dir>/worker.lock`.

    One session's shared state (`loop_state.json`, `checkpoints/`, the curator
    DAG, the run branch) has exactly one authoritative writer. A second
    `agent6 run`/`resume`/`fork` targeting the SAME run dir would spawn a
    second curator whose independent in-memory cache silently clobbers the
    first's parent->child links (a lost update), and would interleave commits on
    the run branch. This is the run-level analogue of `machine_lock`.

    Returns the held fd on success (the caller keeps the process alive to hold
    it, and passes it to `release_single_writer` at teardown), or `None`
    when another live process holds it (the caller refuses). A crashed writer
    leaves no lock -- flock releases on process death -- so resume-after-crash is
    never blocked by a stale lock.
    """
    mkdir_for_real_user(session_dir)
    fd = os.open(session_dir / "worker.lock", os.O_CREAT | os.O_RDWR, 0o644)
    try:
        lock_exclusive(fd, blocking=False)
    except OSError:
        os.close(fd)
        return None
    return fd


def release_single_writer(fd: int | None) -> None:
    """Release + close a lock fd from `acquire_single_writer` (no-op on None).

    Explicit close matters: the fd is a raw int (`os.open`), so it does not
    self-close on GC. A leaked fd would keep the flock held and wrongly refuse a
    later same-dir run in the same process (tests, embedding)."""
    if fd is None:
        return
    with contextlib.suppress(OSError):
        unlock(fd)
    with contextlib.suppress(OSError):
        os.close(fd)


SINGLE_WRITER_BUSY = (
    "REFUSING: run {rid!r} is already being driven by another agent6 process "
    "(its worker.lock is held). Concurrent run/resume of the same run would "
    "corrupt its state (a second curator clobbers the task graph, and commits "
    "interleave on the run branch). Wait for that process to finish; a crashed "
    "one releases the lock automatically."
)


def checkout_lock_path(state_dir: Path, checkout: Path) -> Path:
    """The writer lock of the checkout *checkout* is in (its root, wherever
    the caller stood): `<state-dir>/locks/<checkout-id>.lock`. A repository's
    checkouts (its working tree, each linked worktree) share one state dir
    and hold one lock each."""
    return state_dir / "locks" / f"{repo_id(checkout_root(checkout))}.lock"


def acquire_repo_writer(state_dir: Path, checkout: Path, session_id: str) -> int | None:
    """Take a non-blocking exclusive lock on *checkout*'s lock
    (`checkout_lock_path`): one live `run`-mode worker per CHECKOUT.

    Run-mode workers share one working tree, and each commit stages that whole
    tree (into a temp index, `chain_commit`), so a second concurrent run would
    fold the other's in-flight edits into its own chain: the interleaving the
    run-dir lock prevents for one run, at repo scope. plan/ask make no commits
    and never take this lock.

    The holder stamps its run id into the file so a refusal can name the live
    run. Same crash-safety as `acquire_single_writer`: flock releases on
    process death, so a crashed worker never wedges the checkout. Release with
    `release_single_writer`.
    """
    lock_path = checkout_lock_path(state_dir, checkout)
    mkdir_for_real_user(lock_path.parent)
    fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o644)
    try:
        lock_exclusive(fd, blocking=False)
    except OSError:
        os.close(fd)
        return None
    os.ftruncate(fd, 0)
    os.write(fd, f"{session_id}\n".encode())
    return fd


def repo_writer_holder(state_dir: Path, checkout: Path) -> str:
    """The session id the current holder of *checkout*'s lock stamped, or ""
    unknown. Advisory (for refusal messages); the flock is the boundary."""
    try:
        return checkout_lock_path(state_dir, checkout).read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def repo_writer_held(state_dir: Path, checkout: Path) -> bool:
    """True when a live worker holds *checkout*'s lock.

    An advisory probe for front-end preflight (the web hub refuses a New Work
    submission up front instead of spawning a doomed run): it takes a SHARED
    lock, which an exclusive holder blocks and a second probe does not, so
    asking the question never excludes the writer it asks about. The lock
    itself remains the hard boundary -- a race past this probe still parks at
    `acquire_repo_writer`.
    """
    lock_path = checkout_lock_path(state_dir, checkout)
    if not lock_path.exists():
        return False
    try:
        fd = os.open(lock_path, os.O_RDWR)
    except OSError:
        return False
    try:
        lock_shared_nonblocking(fd)
    except OSError:
        os.close(fd)
        return True
    release_single_writer(fd)
    return False
