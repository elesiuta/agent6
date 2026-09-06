# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Containment for in-process filesystem access.

Every tool that reads/writes a path in-process (outside
`agent6.sandbox.jail.run_in_jail`) resolves it through here first: reject an
absolute path or a `..` component, then require the resolved path to still
be under *root*. Shared by the fs handlers (read_file / list_dir /
apply_edit / apply_patch), the navigation handlers (outline / find_*) -- which
all take an untrusted `path` argument -- and the symbol index they query.
"""

from __future__ import annotations

import contextlib
import errno
import os
import stat
from dataclasses import dataclass
from pathlib import Path

from agent6.tools.errors import ToolError


class NotRegularFile(ToolError):
    """The leaf resolved and is inside the boundary, but is a directory, a FIFO
    or a device. Its own type because callers word the two differently: this is
    "wrong kind of file", not the containment refusal every other ToolError from
    :func:`open_contained` reports."""


@dataclass(frozen=True, slots=True)
class SafePath:
    """A path that passed containment, carrying the base it was contained
    against: every read and write walks from `base`, so a `rel_path` can
    never be paired with the wrong tree."""

    base: Path
    rel_path: Path
    abs_path: Path


@dataclass(frozen=True, slots=True)
class ContainedEntry:
    """One entry of a contained listing. `is_dir` follows a symlink, like
    `Path.is_dir`; a caller that recurses checks `is_symlink` too, because
    the walk refuses to traverse one."""

    name: str
    is_dir: bool
    is_symlink: bool


def fold_name(name: str) -> str:
    """A path component as the FILESYSTEM would match it.

    macOS and Windows match names case-insensitively, and macOS runs agent6
    unsandboxed, so these in-process refusals are the only thing protecting
    `.git` and the hidden trees there: comparing exactly, `.GIT/config` opened
    the real `.git/config` (reproduced on a casefolded ext4). Folded on every
    platform rather than per-filesystem -- one rule, and the cost where case
    does matter is refusing a path to a distinct `.GIT`, which nobody has.
    """
    return name.lower()


def path_within(target: Path, prefix: Path) -> bool:
    """*target* IS *prefix* or lies under it, matched the way the filesystem
    matches names. Whole components only, so `.github` never matches `.git`."""
    folded = [fold_name(p) for p in prefix.parts]
    return [fold_name(p) for p in target.parts][: len(folded)] == folded


@dataclass(frozen=True, slots=True)
class Workspace:
    """The file boundary for everything agent6 does IN-PROCESS.

    Not the sandbox: the sandbox confines child PROCESSES. This is the same
    policy enforced at the other place an untrusted model reaches files -- the
    tools, which run in this process and ask nobody's approval. It therefore
    holds at EVERY isolation level, `none` included: the boundary follows the
    operator's config values, never the isolation level, because a degradation
    that widened what the tools may read would invert the whole degrade rule.

    `denied` (`[sandbox].hide_paths` plus agent6's own private dirs) is
    refused for reads and writes alike, and beats every grant. The grants are
    the operator's `extra_read_paths` / `extra_write_paths`: the same values
    the jail mounts for commands, so a tool and a command reach the same trees.
    A relative path is always the workspace's; an absolute one is allowed only
    inside a grant, which is the only way to name a granted tree at all.

    A path is reached THROUGH a workspace; the tools that deliberately read
    another tree (a skill, the bundled docs, the run's own state) name their own
    base with :func:`contain`.
    """

    root: Path
    denied: tuple[Path, ...] = ()
    # extra_read_paths + extra_write_paths (write implies read).
    read_roots: tuple[Path, ...] = ()
    write_roots: tuple[Path, ...] = ()
    # Files inside a write grant that the harness owns: readable, never written.
    read_only: tuple[Path, ...] = ()
    # agent6's own carve-outs from `denied`, not operator surface: today
    # exactly the per-repo memory dir, a state subtree that is model-writable
    # BY DESIGN (memory is model-authored context). An exempt path still needs
    # a grant to be reachable; exemption only lifts the denial.
    exempt: tuple[Path, ...] = ()

    def _denying(self, abs_path: Path) -> Path | None:
        """The denied root covering *abs_path*, or None. ONE owner for the
        denial verdict: exemption is checked here, so no caller can consult
        `denied` without it."""
        if any(path_within(abs_path, e) for e in self.exempt):
            return None
        for d in self.denied:
            if path_within(abs_path, d):
                return d
        return None

    def is_denied(self, abs_path: Path) -> bool:
        return self._denying(abs_path) is not None

    def resolve_read(self, candidate: str) -> SafePath:
        return self._resolve(candidate, (self.root, *self.read_roots))

    def resolve_write(self, candidate: str) -> SafePath:
        sp = self._resolve(candidate, (self.root, *self.write_roots))
        if sp.abs_path in self.read_only:
            raise ToolError(f"Path is harness-owned and read-only: {candidate!r}")
        return sp

    def _resolve(self, candidate: str, bases: tuple[Path, ...]) -> SafePath:
        sp = (
            self._in_grant(candidate, bases)
            if candidate.startswith("/")
            else resolve_in_root(self.root, candidate)
        )
        self._refuse_denied(sp, candidate)
        return sp

    def _in_grant(self, candidate: str, bases: tuple[Path, ...]) -> SafePath:
        """An absolute path, contained against the deepest grant holding it.

        Deepest first, so a grant nested inside another walks from the one that
        really bounds it rather than from an ancestor that also matches.
        """
        target = Path(candidate).resolve()
        for base in sorted(bases, key=lambda b: len(b.parts), reverse=True):
            if path_within(target, base):
                return SafePath(base=base, rel_path=target.relative_to(base), abs_path=target)
        raise ToolError(f"Absolute paths are only allowed inside a granted path: {candidate!r}")

    def _refuse_denied(self, sp: SafePath, candidate: str) -> None:
        # Refused, not answered empty: the jail masks because a command cannot
        # be handed an error, but a tool result can carry one, and inventing
        # "no such file" for a path that is plainly there is the surface lying.
        d = self._denying(sp.abs_path)
        if d is not None:
            raise ToolError(f"Path is hidden from this run: {candidate!r} (under {d})")


def contain(base: Path, candidate: str | Path) -> SafePath:
    """Contain *candidate* under *base* without resolving symlinks: the
    descriptor walk is what enforces it, refusing every symlink hop.

    For the bases that are deliberately NOT the workspace -- a skill's own
    directory, the bundled docs -- where the caller, not the model, chose the
    tree. Workspace paths go through :class:`Workspace` instead.
    """
    rel = Path(candidate)
    if rel.is_absolute():
        raise ToolError(f"Absolute paths not allowed: {str(candidate)!r}")
    if ".." in rel.parts:
        raise ToolError(f"Path contains '..': {str(candidate)!r}")
    return SafePath(base=base, rel_path=rel, abs_path=base / rel)


def resolve_in_root(root: Path, candidate: str) -> SafePath:
    """Resolve *candidate* relative to *root* and ensure it stays inside *root*."""
    if candidate.startswith("/"):
        raise ToolError(f"Absolute paths not allowed: {candidate!r}")
    parts = Path(candidate).parts
    if ".." in parts:
        raise ToolError(f"Path contains '..': {candidate!r}")
    abs_path = (root / candidate).resolve()
    try:
        rel = abs_path.relative_to(root.resolve())
    except ValueError as exc:
        raise ToolError(f"Path escapes repo root: {candidate!r}") from exc
    return SafePath(base=root, rel_path=rel, abs_path=abs_path)


def _open_dir(dir_fd: int, name: str, *, create: bool) -> int:
    """A descriptor for subdirectory *name* of *dir_fd*, created when it is
    missing and *create*."""
    flags = os.O_PATH | os.O_DIRECTORY | os.O_NOFOLLOW
    try:
        return os.open(name, flags, dir_fd=dir_fd)
    except FileNotFoundError:
        if not create:
            raise
    with contextlib.suppress(FileExistsError):
        os.mkdir(name, dir_fd=dir_fd)
    return os.open(name, flags, dir_fd=dir_fd)


def open_contained(sp: SafePath, flags: int, *, create_parents: bool = False) -> int:
    """Open `sp` one component at a time from a descriptor on its base, each
    hop relative to the one before it. Returns an fd the caller owns.

    A :class:`SafePath` resolves and contains a path; opening it again by its
    full path is a second lookup, and a jailed background command's loop can
    swap a component for a symlink out of the workspace in between (the
    workspace is writable, a symlink needs no access to its target, and these
    tools run IN-PROCESS, outside the jail, as the operator). For a write
    (`O_CREAT|O_TRUNC`) the host file is already truncated by the time any
    after-the-fact check can reject it.

    `O_NOFOLLOW` on every component, including the parents this creates,
    contains the walk by construction: no hop can traverse a symlink. `..`
    and an absolute path are refused here as well as at the SafePath, so
    containment holds even for a hand-built one. Honest callers are unaffected,
    including one working through an in-repo symlink, whose resolved path names
    the real target.

    Unless `O_DIRECTORY` is asked for, the leaf must be a REGULAR file, and
    the check is `fstat` on the descriptor just opened -- never a stat by
    name, which is a second lookup. `O_NONBLOCK` makes the open itself
    unable to block: a jailed background command can swap the leaf for a FIFO
    between any check and the open, and `O_NOFOLLOW` stops a symlink but not
    that. The flag is cleared before the caller reads or writes.
    """
    rel_path = sp.rel_path
    if rel_path.is_absolute():
        raise ToolError(f"Path is not relative to the workspace: {rel_path}")
    if ".." in rel_path.parts:
        raise ToolError(f"Path contains '..': {rel_path}")
    dir_fd = os.open(sp.base, os.O_PATH | os.O_DIRECTORY)
    at = "."  # the component the walk is on, for the error path below
    try:
        for at in rel_path.parts[:-1]:
            child = _open_dir(dir_fd, at, create=create_parents)
            os.close(dir_fd)
            dir_fd = child
        # The root itself is the one path with no leaf to name.
        at = rel_path.name or "."
        if flags & os.O_DIRECTORY:
            return os.open(at, flags | os.O_NOFOLLOW, 0o644, dir_fd=dir_fd)
        fd = os.open(at, flags | os.O_NOFOLLOW | os.O_NONBLOCK, 0o644, dir_fd=dir_fd)
        try:
            if not stat.S_ISREG(os.fstat(fd).st_mode):
                raise NotRegularFile(f"Not a regular file: {rel_path}")
            os.set_blocking(fd, True)
        except BaseException:
            os.close(fd)
            raise
        return fd
    except NotADirectoryError as exc:
        # O_NOFOLLOW|O_DIRECTORY on a symlink is ENOTDIR on Linux, not ELOOP:
        # without this probe, a component swapped for a symlink mid-walk (the
        # race this walk exists to contain) read as the bland message below.
        # One lstat, on the error path only.
        with contextlib.suppress(OSError):
            if stat.S_ISLNK(os.lstat(at, dir_fd=dir_fd).st_mode):
                raise ToolError(
                    f"Path became a symlink while it was being used: {rel_path}"
                ) from exc
        raise ToolError(f"Path component is not a directory: {rel_path}") from exc
    except OSError as exc:
        if exc.errno == errno.ELOOP:
            raise ToolError(f"Path became a symlink while it was being used: {rel_path}") from exc
        if exc.errno == errno.ENXIO:
            # O_WRONLY|O_NONBLOCK on a reader-less FIFO: the one non-regular
            # leaf the open rejects itself, so the fstat never sees it.
            raise NotRegularFile(f"Not a regular file: {rel_path}") from exc
        raise
    finally:
        os.close(dir_fd)


def read_contained(sp: SafePath, *, errors: str = "strict", limit_chars: int | None = None) -> str:
    """The file's text, read through a descriptor walked from its base.
    `UnicodeDecodeError` still reaches the caller, which reports it.

    `limit_chars` bounds the read: at most that many characters are pulled
    into memory, so a multi-gigabyte file cannot OOM the (unsandboxed) agent.
    The caller detects truncation by reading `limit_chars + 1` and checking
    the length. None reads the whole file (for callers that must, like the
    symbol index parsing a source file)."""
    fd = open_contained(sp, os.O_RDONLY)
    with os.fdopen(fd, encoding="utf-8", errors=errors) as handle:
        return handle.read() if limit_chars is None else handle.read(limit_chars)


def read_bytes_contained(sp: SafePath) -> bytes:
    """The file's bytes, read through a descriptor walked from its base. For a
    reader that indexes into the source by byte offset (tree-sitter), which the
    newline translation of a text read would shift."""
    fd = open_contained(sp, os.O_RDONLY)
    with os.fdopen(fd, "rb") as handle:
        return handle.read()


def list_contained(sp: SafePath) -> list[ContainedEntry]:
    """The directory's entries, listed through a descriptor walked from its base.

    The same containment as :func:`read_contained`, for the tools that read a
    directory rather than a file: a name resolved a second time is a second
    lookup, so a listing taken by full path can be a host directory's.
    """
    fd = open_contained(sp, os.O_RDONLY | os.O_DIRECTORY)
    try:
        with os.scandir(fd) as entries:
            return [ContainedEntry(e.name, e.is_dir(), e.is_symlink()) for e in entries]
    finally:
        os.close(fd)


def unlink_contained(sp: SafePath) -> None:
    """Remove the file through a descriptor walk of its parents (the walk
    :func:`open_contained` does), unlinking the leaf by name relative to the
    parent's descriptor: a component swapped for a symlink cannot redirect a
    delete any more than a write."""
    if not sp.rel_path.name:
        raise ToolError(f"Not a file: {sp.rel_path}")
    parent = SafePath(sp.base, sp.rel_path.parent, sp.abs_path.parent)
    fd = open_contained(parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.unlink(sp.rel_path.name, dir_fd=fd)
    finally:
        os.close(fd)


def write_contained(sp: SafePath, content: str) -> None:
    """Replace the file's text through a descriptor walked from its base, adding
    any missing parent directories along the same walk."""
    flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
    fd = open_contained(sp, flags, create_parents=True)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(content)
