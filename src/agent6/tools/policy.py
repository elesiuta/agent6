# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The one description of what a confined child may do.

Its own module rather than a corner of `dispatch.py`: two tools-layer callers
need it (`dispatch` for commands, `mcp_client` for servers) and `dispatch`
already imports `mcp_client`, so leaving it there would have forced a cycle.
"""

from __future__ import annotations

import os
import stat
from pathlib import Path
from typing import Any

from agent6.config import Config
from agent6.memory import DECISIONS_NAME
from agent6.paths import (
    effective_user,
    hidden_paths,
    jail_cache_home,
    mkdir_for_real_user,
    private_dirs,
)
from agent6.sandbox._tool_paths import operator_tool_paths
from agent6.sandbox.jail import JailUnavailableError
from agent6.tools._path_safety import Workspace
from agent6.tools._result_format import passthrough_env
from agent6.types import IsolationLevel, JailPolicy, NetworkMode

# strict's default HOME: the launcher creates it inside the run's private /tmp
# tmpfs, so it goes with the run.
JAIL_TMP_HOME = Path("/tmp/agent6-home")  # noqa: S108 - resolved inside the jail


def persistent_jail_home(config: Config, isolation: IsolationLevel) -> Path | None:
    """The persistent HOME a jailed command gets, or None where strict's
    private tmpfs one (`JAIL_TMP_HOME`) applies.

    Only `strict` has a private /tmp to put a throwaway HOME in; every other
    level gets :func:`agent6.paths.jail_cache_home`, and strict opts into it
    with `[sandbox].home = "cache"`. `jail_policy` creates it and grants it
    read-write at its real path.
    """
    if isolation == "strict" and config.sandbox.home == "tmp":
        return None
    return jail_cache_home()


def jail_home_refusal(home: Path) -> str | None:
    """Why *home* cannot be the jail's persistent HOME, else None; an absent
    path is fine (`jail_policy` creates it).

    Inspection only: a symlink (a redirect into the operator's own home), a
    non-directory, another user's directory, a mode with any group or other
    bit (a jailed command owns the dir and may chmod it, and an open one lets
    another local user plant what the next jailed run consumes; checked on
    every build, never restored), or a path inside an agent6-private dir,
    which the strict mask would re-bind writable. The config validator
    refuses the same for `extra_write_paths`; this grant comes from
    `AGENT6_CACHE_HOME`, so it is checked here, on resolved paths so a
    symlinked ancestor cannot dodge it.
    """
    real = home.resolve()
    for private in private_dirs():
        if real.is_relative_to(private.resolve()):
            where = "" if real == home else f" (really {str(real)!r})"
            return (
                f"the jail's HOME {str(home)!r}{where} is inside agent6's private dir"
                f" {str(private)!r} (secrets/state). Point AGENT6_CACHE_HOME elsewhere."
            )
    try:
        st = os.lstat(home)
    except FileNotFoundError:
        return None
    except OSError as exc:
        return f"the jail's HOME {str(home)!r} cannot be read: {exc}"
    uid = effective_user().uid
    mode = stat.S_IMODE(st.st_mode)
    fix = "Remove it, or point AGENT6_CACHE_HOME at a directory of your own."
    if stat.S_ISLNK(st.st_mode):
        problem = (
            f"is a symlink (to {str(home.readlink())!r}): jailed commands would write through it"
        )
    elif not stat.S_ISDIR(st.st_mode):
        problem = "is not a directory"
    elif st.st_uid != uid:
        problem = (
            f"is owned by uid {st.st_uid}, not you (uid {uid}):"
            " another user's directory is never bound"
        )
    elif mode & 0o077:
        problem = (
            f"has mode {mode:04o}, open to other users, who could plant a ~/.gitconfig or"
            " cache content for the next jailed run"
        )
        fix = f"Run: chmod 700 {home}"
    else:
        return None
    return f"the jail's HOME {str(home)!r} {problem}. {fix}"


def workspace_for(config: Config, root: Path, *, memory_dir: Path | None = None) -> Workspace:
    """The in-process file boundary for a run rooted at *root*.

    The tools are the FRONT DOOR of the file axis -- an untrusted model reaches
    files through them directly, with no approval -- and the jail is the fence
    that stops a command working around it. Both read the same hidden set
    (:func:`agent6.paths.hidden_paths`), so the two enforcers cannot disagree.

    Derived from config VALUES, never from the isolation level: a degradation
    (auto falling back to hardened or none, macOS having no jail at all) must
    never WIDEN what the tools may touch.
    """
    sb = config.sandbox
    denied = hidden_paths(Path(p) for p in sb.hide_paths)
    writable = tuple(Path(p).resolve() for p in sb.extra_write_paths)
    # The per-repo memory dir: model-writable state BY DESIGN (the memory
    # index and entries are model-authored context). In-process tools only;
    # jail_policy never mounts it, so commands still see nothing.
    mem = (memory_dir.resolve(),) if memory_dir is not None else ()
    return Workspace(
        root=root.resolve(),
        denied=tuple(p.resolve() for p in denied),
        # Write implies read, matching the grants the jail mounts.
        read_roots=(*(Path(p).resolve() for p in sb.extra_read_paths), *writable, *mem),
        write_roots=(*writable, *mem),
        exempt=mem,
        read_only=(memory_dir.resolve() / DECISIONS_NAME,) if memory_dir is not None else (),
    )


def resolve_network(
    config: Config, isolation: IsolationLevel, *, override: NetworkMode | None = None
) -> NetworkMode:
    """The network a child actually gets.

    A caller that answers for itself passes *override*: an MCP server's
    reachability is the operator's per-server choice, not the tool policy.
    Otherwise a command joins the host network only under `network = "host"`,
    and every other setting puts it on the run's own.

    Clamped to what the level can provide: only strict has namespaces, so
    everywhere else the child shares this process's network and the policy says
    so rather than describing a confinement it will not get (preflight has
    already refused an EXPLICIT setting it cannot honour, and warned about an
    automatic one).
    """
    if isolation != "strict":
        return "host"
    if override is not None:
        return override
    return "host" if config.sandbox.network == "host" else "session"


def linked_worktree_git_dir(root: Path) -> Path | None:
    """The repository git dir the linked worktree at *root* points into, or
    None for an ordinary checkout: its `.git` file (`gitdir: <repo>/.git/
    worktrees/<name>`) and that entry's `commondir`, resolved as git does.
    Read to VERIFY a recorded grant, never to make one: both files sit in the
    workspace, writable by a jailed command under hardened."""
    pointer = root / ".git"
    if not pointer.is_file():
        return None
    try:
        text = pointer.read_text(encoding="utf-8", errors="replace").strip()
        if not text.startswith("gitdir:"):
            return None
        admin = Path(text[len("gitdir:") :].strip())
        if not admin.is_absolute():
            admin = root / admin
        common = Path((admin / "commondir").read_text(encoding="utf-8").strip())
        if not common.is_absolute():
            common = admin / common
        return common.resolve()
    except OSError:
        return None


def jail_policy(
    root: Path,
    config: Config,
    isolation: IsolationLevel,
    argv: tuple[str, ...],
    *,
    timeout_s: float | None = None,
    extra_ro_paths: tuple[Path, ...] = (),
    extra_rw_paths: tuple[Path, ...] = (),
    extra_protect_paths: tuple[Path, ...] = (),
    network: NetworkMode | None = None,
    env_base: dict[str, str] | None = None,
    worktree_git_dir: Path | None = None,
) -> JailPolicy:
    """The sandbox policy every LLM-influenced argv runs under.

    *worktree_git_dir* is the repository git dir agent6 recorded when it
    added *root* as a fork's linked worktree (the manifest's
    `worktree_git_dir`): granted read-only, so git works there, once the
    worktree's own `.git` pointer still resolves to it; a pointer that names
    anything else refuses (JailUnavailableError) rather than grant what a
    jailed command wrote. A linked worktree with no record gets no grant.

    One owner, so every caller is confined identically: a foreground command, a
    detached one (`background: true`), the baseline gate re-run, and a spawned
    MCP server all get the same protect paths, env, tool mounts and memory cap.
    A second policy builder would drift: a gate with its own env misses PATH
    and exits 127 (a failure blamed on the tree), and a parallel confinement
    stack ends up without seccomp, a private /proc, or hidden-path masking.

    Callers name only what is EXTRA. Everything a child needs to exist -- the
    system dirs, the operator's tool dirs, a writable HOME -- is here, so
    nobody has to know where their interpreter lives.
    """
    network = resolve_network(config, isolation, override=network)
    protect_paths: list[Path] = []
    # STRICT only. A writable `.git` is not merely "recoverable": a jailed
    # command can plant a `filter.<n>.clean` in `.git/config` plus a
    # `.gitattributes`, and agent6's own auto-commit then executes it on the
    # HOST, outside the jail. Strict re-binds `.git` read-only, which needs a
    # mount namespace.
    #
    # Hardened has none, so the only tool is Landlock -- which has no deny
    # rules. Protecting `.git` there meant not granting the workspace ROOT,
    # because a Landlock grant is recursive and granting the root its own
    # create/remove rights grants them over `.git` too. That cost every
    # top-level write (`touch newfile`, `mkdir build`), which is too much to
    # pay for a protection the operator can have properly by using strict.
    # The in-process edit tools refuse `.git` writes on both isolation levels.
    if config.sandbox.protect_git and isolation == "strict":
        protect_paths.append((root / ".git").resolve())
    protect_paths.extend(extra_protect_paths)
    policy_kwargs: dict[str, Any] = {}
    if timeout_s is not None:
        policy_kwargs["timeout_s"] = timeout_s
    # A command forwards the few host variables that shape output (LANG, TERM);
    # a caller with its own answer -- an MCP server, which gets the curated set
    # plus the names the operator listed -- passes env_base instead. The jail
    # defaults below (PATH, HOME, the uv/bytecode settings) apply either way.
    env = passthrough_env() if env_base is None else dict(env_base)
    # Toolchains need a writable cache root (go test -> $HOME/.cache/go-build,
    # cargo -> $CARGO_HOME, pip/uv likewise). FORCED, like PATH: the operator's
    # HOME is not in the jail, so inheriting one (curated_env carries it) gives
    # the child an unwritable path and every cache write fails. The persistent
    # HOME is a write grant below; the tmpfs one is inside strict's private
    # /tmp, which the launcher populates.
    persistent = persistent_jail_home(config, isolation)
    if persistent is not None:
        # Created where the grant is decided, so every surface that builds a
        # policy (a run, `agent6 exec`, an MCP probe) gets a HOME that exists:
        # the launcher skips a missing rw path silently.
        # Inspected before creation (a path inside a private dir is never
        # created there) and again after it (what sits at the path now).
        refusal = jail_home_refusal(persistent)
        if refusal is None and not os.path.lexists(persistent):
            try:
                mkdir_for_real_user(persistent)
            except OSError as exc:
                raise JailUnavailableError(
                    f"the jail's HOME {str(persistent)!r} cannot be created: {exc}"
                ) from exc
            refusal = jail_home_refusal(persistent)
        if refusal is not None:
            raise JailUnavailableError(refusal)
    env["HOME"] = str(persistent or JAIL_TMP_HOME)
    env.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    # `uv run` inside the jail must use the venv the operator already synced: the
    # jail is offline, so a sync would re-resolve against a cold cache and fail.
    env.setdefault("UV_NO_SYNC", "1")
    # Make operator-installed tools reachable: a controlled PATH extending
    # /usr/bin:/bin with the standard bin dirs, plus their real dirs as RO+exec
    # mounts. Without this a `uv run` verify dies 127.
    tool_path, tool_mounts = operator_tool_paths()
    env["PATH"] = tool_path
    git_dir: Path | None = None
    if worktree_git_dir is not None:
        pointed = linked_worktree_git_dir(root)
        if pointed != worktree_git_dir:
            raise JailUnavailableError(
                f"the worktree's .git at {root / '.git'} points at {pointed}, not the"
                f" repository git dir agent6 recorded for it ({worktree_git_dir});"
                " a rewritten pointer grants nothing. Restore the pointer, or fork again."
            )
        git_dir = worktree_git_dir
    return JailPolicy(
        cwd=root,
        argv=argv,
        isolation=isolation,
        env=tuple(sorted(env.items())),
        network=network,
        extra_protect_paths=tuple(protect_paths),
        extra_ro_paths=(
            *(Path(p) for p in config.sandbox.extra_read_paths),
            *extra_ro_paths,
            *((git_dir,) if git_dir is not None else ()),
        ),
        extra_rw_paths=(
            *(Path(p) for p in config.sandbox.extra_write_paths),
            *extra_rw_paths,
            *(() if persistent is None else (persistent,)),
        ),
        extra_device_paths=tuple(Path(p) for p in config.sandbox.extra_device_paths),
        tool_paths=tool_mounts,
        hide_paths=tuple(Path(p) for p in config.sandbox.hide_paths),
        memory_limit_mb=config.sandbox.memory_limit_mb,
        **policy_kwargs,
    )
