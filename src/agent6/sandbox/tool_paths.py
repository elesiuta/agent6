# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Operator tool reachability inside the jail: the PATH a jailed command
gets, the real tool dirs mounted read+exec for it, and the notes a
surface prints about tools it cannot reach or mounts that expose more
than a tool.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from agent6.paths import private_dirs

# The jail's baseline PATH is /usr/bin:/bin and it bind-mounts only the system
# roots below. Operator tools (uv, node, ruff, ...) installed elsewhere are
# otherwise unreachable, so a jailed command dies 127. We add the standard bin
# dirs that exist to PATH, and for those outside the system roots (or whose
# symlinks resolve out to one, a pipx `uv` at /usr/local/bin -> /opt/pipx/...)
# pass the real dirs as tool_paths for a real-location RO+exec mount. Read+exec
# only; the jail still confines writes and network, so containment is
# unchanged. Owned here so run_command and verify (tools.dispatch), machine
# tool states (machine.engine), and the host-side probe (`machine check`)
# resolve tools identically.
_JAIL_BASE_PATH_DIRS = ("/usr/bin", "/bin")
_SYSTEM_ROOTS = (
    Path("/usr"),
    Path("/bin"),
    Path("/sbin"),
    Path("/lib"),
    Path("/lib64"),
    Path("/etc"),
    Path("/dev"),
)


def _under_system_root(p: Path) -> bool:
    return any(p.is_relative_to(r) for r in _SYSTEM_ROOTS)


def _never_mounted(p: Path) -> bool:
    """Dirs that never belong in a jail mount, however a tool symlink
    resolves.

    `operator_tool_paths` mounts `real.parent` for every symlink in a bin
    dir, so one resolving into the config dir mounted `secrets.toml` -- the
    provider API keys -- read-only into the jail, and one into the state dir
    mounted memory and transcripts. Containment cuts both ways: a
    mount CONTAINING a private dir grants the same reads from above, and a
    plain `~/.local/bin/x -> ~/x.sh` makes `real.parent` the whole home
    dir. So agent6's private dirs (:func:`agent6.paths.private_dirs`) are
    refused in either direction, and $HOME and its ancestors outright:
    mounting home or a dir above it would hand the jail `~/.ssh` and every
    credential the operator owns. A mount BELOW home (a tool target's own
    subdir) stays allowed; that is what keeps `~/.local/bin` tools working.
    Denied by identity rather than by inspecting contents.
    """
    if Path.home().is_relative_to(p):
        return True
    return any(p.is_relative_to(d) or d.is_relative_to(p) for d in private_dirs())


def operator_tool_paths() -> tuple[str, tuple[Path, ...]]:
    """Return (PATH string, real-location mount dirs) so operator-installed tools
    resolve in the jail. Recomputed per call so a tool the operator (or model)
    just installed is picked up (dirs under a mounted system root only join PATH;
    dirs outside it, and the real dirs symlinks resolve out to, also need the
    RO+exec mount)."""
    path_dirs: list[str] = list(_JAIL_BASE_PATH_DIRS)
    candidates = _tool_bin_dirs()
    mounts: set[Path] = set()
    for d in candidates:
        if not d.is_dir():
            continue
        path_dirs.append(str(d))
        if not _under_system_root(d) and not _never_mounted(d):
            mounts.add(d)  # real binaries in a non-system dir need the dir itself
        try:
            entries = list(d.iterdir())
        except OSError:
            continue
        for entry in entries:
            if not entry.is_symlink():
                continue  # real files are covered by the dir / the /usr mount
            try:
                real = entry.resolve()
            except OSError:
                continue
            if real.is_file() and not _under_system_root(real) and not _never_mounted(real.parent):
                mounts.add(real.parent)  # e.g. /opt/pipx/venvs/uv/bin
    # Interpreter toolchains a repo venv's python may symlink to: uv-managed
    # CPython lives under XDG data, not any bin dir. Without this mount the
    # jail sees such a venv "linked to a non-existent interpreter" and an
    # in-jail `uv run` deletes and recreates the operator's .venv.
    # Mount-only, never a PATH entry.
    data_home = Path(os.environ.get("XDG_DATA_HOME") or Path.home() / ".local/share")
    uv_pythons = data_home / "uv" / "python"
    if uv_pythons.is_dir():
        mounts.add(uv_pythons)
    return ":".join(path_dirs), tuple(sorted(mounts))


def _tool_bin_dirs() -> tuple[Path, ...]:
    """The bin dirs scanned for operator-installed tools (PATH + mounts)."""
    home = Path.home()
    return (
        Path("/usr/local/bin"),
        Path("/usr/local/sbin"),
        home / ".local/bin",
        home / ".cargo/bin",
        Path("/opt/homebrew/bin"),
        Path("/snap/bin"),
    )


@dataclass(frozen=True, slots=True)
class ToolMountNotes:
    """What the operator should know about how their bin dirs resolve into the
    jail, for the once-per-run preflight. Both lists are `"<link> -> <target>"`
    strings; the mount decisions themselves are unchanged and silent."""

    # A symlink whose target's dir is never mounted, so the tool is absent
    # inside the jail (it would die 127 with nothing naming the reason).
    unreachable: tuple[str, ...] = ()
    # A symlink resolving OUT of its bin dir into another dir under $HOME,
    # which is therefore mounted read-only into the jail. Allowed on purpose
    # (it is what keeps ~/.local/bin tools working), but the operator placed
    # one symlink and got a whole directory exposed, so say which.
    exposes_home_dir: tuple[str, ...] = ()


def tool_mount_notes() -> ToolMountNotes:
    """Scan the bin dirs the jail puts on PATH and report both surprises: a
    tool the jail cannot reach, and a tool that drags a home directory into
    the jail with it."""
    home = Path.home()
    bin_dirs = _tool_bin_dirs()
    unreachable: list[str] = []
    exposes: list[str] = []
    for d in bin_dirs:
        try:
            entries = list(d.iterdir()) if d.is_dir() else []
        except OSError:
            continue
        for entry in entries:
            if not entry.is_symlink():
                continue
            try:
                real = entry.resolve()
            except OSError:
                continue
            if not real.is_file() or _under_system_root(real):
                continue
            parent = real.parent
            if _never_mounted(parent):
                unreachable.append(f"{entry} -> {real}")
            elif parent.is_relative_to(home) and parent not in bin_dirs:
                exposes.append(f"{entry} -> {real}")
    return ToolMountNotes(tuple(sorted(unreachable)), tuple(sorted(exposes)))


def jail_search_path() -> str:
    """The PATH a jailed command resolves against, for host-side reachability
    probes (`machine check`): the jail baseline plus the standard bin dirs that
    exist right now. Advisory only; the jail recomputes its own per call."""
    return operator_tool_paths()[0]
