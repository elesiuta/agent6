# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The one description of what a confined child may do.

Its own module rather than a corner of `dispatch.py`: two tools-layer callers
need it (`dispatch` for commands, `mcp_client` for servers) and `dispatch`
already imports `mcp_client`, so leaving it there would have forced a cycle.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from agent6.config import Config
from agent6.memory import DECISIONS_NAME
from agent6.paths import hidden_paths
from agent6.sandbox.jail import operator_tool_paths
from agent6.tools._path_safety import Workspace
from agent6.tools._result_format import passthrough_env
from agent6.types import IsolationLevel, JailPolicy, NetworkMode


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
) -> JailPolicy:
    """The sandbox policy every LLM-influenced argv runs under.

    One owner, so every caller is confined identically: a foreground command, a
    detached one (`background: true`), the baseline gate re-run, and a spawned
    MCP server all get the same protect paths, env, tool mounts and memory cap.
    A second policy builder would drift: a gate with its own env misses PATH
    and exits 127 (a failure blamed on the tree), and a parallel confinement
    stack ends up without seccomp, a private /proc, or hidden-path masking.

    Callers name only what is EXTRA. Everything a child needs to exist -- the
    system dirs, the operator's tool dirs, a writable /tmp as HOME -- is here,
    so nobody has to know where their interpreter lives.
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
    # cargo -> $CARGO_HOME, pip/uv likewise). The jail's /tmp is writable on both
    # isolation levels, so point HOME there. FORCED, like PATH: the host's HOME
    # does not exist inside the jail, so inheriting one (curated_env carries it)
    # gives the child an unwritable path and every cache write fails.
    env["HOME"] = "/tmp/agent6-home"  # noqa: S108 - resolved inside the jail
    env.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    # `uv run` inside the jail must use the venv the operator already synced: the
    # jail is offline and HOME is a fresh tmpfs, so a sync would re-resolve
    # against an empty cache and fail.
    env.setdefault("UV_NO_SYNC", "1")
    # Make operator-installed tools reachable: a controlled PATH extending
    # /usr/bin:/bin with the standard bin dirs, plus their real dirs as RO+exec
    # mounts. Without this a `uv run` verify dies 127.
    tool_path, tool_mounts = operator_tool_paths()
    env["PATH"] = tool_path
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
        ),
        extra_rw_paths=(
            *(Path(p) for p in config.sandbox.extra_write_paths),
            *extra_rw_paths,
        ),
        tool_paths=tool_mounts,
        hide_paths=tuple(Path(p) for p in config.sandbox.hide_paths),
        memory_limit_mb=config.sandbox.memory_limit_mb,
        **policy_kwargs,
    )
