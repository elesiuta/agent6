# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Warnings and cross-checks for what an isolation level does NOT confine.

The agent process itself is never confined, at any isolation level: every
boundary is the jail's, and the levels differ only in
which jail features the launcher enables (docs/security.md owns the model
and the rationale). Nothing bounds the agent's own filesystem or egress; a
partial block on a trusted process reads as a guarantee it cannot keep, so
these checks warn or refuse instead of pretending.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from agent6.app._setup import detect_env, mcp_server_policy
from agent6.app.reporter import STDIO_REPORTER, Reporter
from agent6.config import Config, MCPServerEntry
from agent6.models.registry import resolved_adaptive_values
from agent6.paths import hidden_paths, is_root, jail_cache_home, private_dirs
from agent6.sandbox._tool_paths import tool_mount_notes
from agent6.sandbox.detect import Environment, degrade_reason, resolve_isolation
from agent6.sandbox.jail import JailBinaryError, JailUnavailableError
from agent6.tools.policy import (
    jail_home_refusal,
    jail_policy,
    persistent_jail_home,
    resolve_network,
)
from agent6.types import IsolationLevel


def warn_sandbox_gaps(
    isolation: IsolationLevel,
    env: Environment,
    cfg: Config,
    *,
    root: Path,
    worktree_git_dir: Path | None = None,
    reporter: Reporter = STDIO_REPORTER,
) -> None:
    """Print a prominent warning when the isolation confines less than it promises.

    `none` is reached on a host with no confinement mechanism at all
    (non-Linux, or a Linux kernel offering neither userns nor Landlock), or
    when the operator EXPLICITLY sets `isolation = "none"` (the unsandboxed
    opt-out, intended for inside a container). Either way commands run as
    plain subprocesses with no agent6 confinement, so say so loudly.

    `strict` needs only userns; on a kernel without Landlock the jail's
    best-effort ruleset enforces nothing (`restrict_self` returns NotEnforced)
    while namespaces + the pivoted read-only rootfs + seccomp still confine.
    That is a documented layer going missing, so it is loud too -- here, once
    per run, not in the launcher: a per-spawn stderr warning would land in
    every tool result and prompt the model to fight the sandbox.

    `network = "auto"` DEGRADES on a netns-less isolation: with no network
    namespace there is no session network to give, so a jailed run_command
    shares the host's, and we say so once per run. Explicit `session` never
    reaches here (check_network_support refused it on hardened).

    Landlock below ABI 3 (Linux 6.2) does not confine file truncation, so on
    `hardened` there a jailed command can truncate files OUTSIDE its write
    grants; we warn once per run. Explicit `hardened` never reaches here
    (resolve_isolation refused it below ABI 3).

    `protect_git` degrades the same way: strict-only, because it is a read-only
    bind. An explicitly-set one refuses (check_protect_git_support).

    The persistent HOME is named once per run: an explicit widening under
    `strict` (`home = "cache"`), the level's own shape under `hardened`. `none`
    has no jail, and the unsandboxed warning covers it.

    Running as ROOT is the operator's explicit widening, so it warns rather
    than refuses -- but on `hardened` it says what the widening costs, since
    the granted system set stops being narrowed by file permissions there.
    """
    if cfg.sandbox.isolation == "auto":
        reason = degrade_reason(env)
        if reason is not None:
            # The degrade ITSELF, not just its consequences below: 'auto'
            # landing under strict never happens silently, and the why is the
            # same line check sandbox / check config print (one owner).
            reporter.warn(f"'auto' selected '{isolation}', not 'strict': {reason}.")
    if isolation == "none":
        origin = (
            "sandbox.isolation = 'none'"
            if cfg.sandbox.isolation == "none"
            else "'auto' found no confinement mechanism on this host"
        )
        reporter.warn(
            f"running UNSANDBOXED ({origin}). "
            "Every command runs as a plain subprocess with no filesystem, network, "
            "or syscall confinement: the LLM's run_command, the verify command, and "
            "any spawned MCP server. sandbox.memory_limit_mb still applies: the "
            "launcher enforces it with confinement off. Only the "
            "surrounding environment, a container for instance, bounds what a command "
            "can reach. Use 'auto', 'strict', or 'hardened' for kernel-enforced "
            "isolation."
        )
    elif isolation == "strict" and env.landlock_abi < 1:
        reporter.warn(
            "'strict' is running WITHOUT its Landlock layer: "
            "this kernel offers no Landlock (needs Linux >= 5.13 with the "
            "Landlock LSM enabled). Namespaces, the pivoted read-only rootfs, "
            "and seccomp still confine commands; the in-jail Landlock "
            "defense-in-depth is absent."
        )
    if isolation == "hardened" and is_root():
        # The root banner names running as root; this names what it COSTS at
        # this level, which is where the operator would otherwise find out
        # afterwards. Not a blocklist of sensitive files: the grant is the
        # documented read-only system set, and root simply stops file
        # permissions from narrowing it.
        reporter.warn(
            "running as root under 'hardened': file permissions "
            "no longer narrow what a jailed command reads, so it can read the "
            "root-only files in the granted system set (/etc/shadow, /etc/sudoers, "
            "the host's ssh private keys). 'strict' pivots into a minimal rootfs "
            "where they are absent. Run as your normal user."
        )
    if isolation == "hardened" and cfg.sandbox.protect_git:
        reporter.warn(
            "'hardened' cannot protect .git: the read-only bind "
            "needs a mount namespace, which only 'strict' has. A jailed command can "
            "write .git; the in-process edit tools still refuse. For the same "
            "reason /tmp is the host's shared /tmp. Use 'strict' for a private /tmp "
            "and a protected .git."
        )
    persistent = persistent_jail_home(cfg, isolation)
    if persistent is not None and isolation != "none":
        cause, fix = (
            (
                "sandbox.home = 'cache'",
                "Set sandbox.home = 'tmp' for a HOME that goes with the run.",
            )
            if isolation == "strict"
            else (
                "'hardened' has no private /tmp",
                "Use 'strict' for a HOME that goes with the run.",
            )
        )
        reporter.warn(
            f"{cause}: HOME ({persistent}) persists across runs and is executable, so a "
            "cache poisoned by one run, or a ~/.gitconfig alias, reaches the next jailed "
            f"run (never your own tools). {fix}"
        )
    if isolation == "hardened" and cfg.sandbox.network == "auto":
        reporter.warn(
            "'hardened' has no network namespace, so "
            "sandbox.network = 'auto' cannot give the run its own session "
            "network: jailed commands share this process's host network, which "
            "hardened does not confine. Run on 'strict' for a session "
            "network, or set sandbox.network = 'session' to refuse the run "
            "instead."
        )
    if isolation == "hardened" and env.landlock_abi < 3:
        reporter.warn(
            f"'hardened' on Landlock ABI {env.landlock_abi} (< 3) "
            "does not confine file truncation: a jailed command can truncate "
            "(truncate/ftruncate) files outside its write grants, discarding their "
            "contents. Its other writes stay confined. Full write-confinement needs "
            "Landlock ABI 3 (Linux 6.2): upgrade the kernel, or run on 'strict', "
            "whose mount namespace confines truncation on any ABI."
        )
    for hidden, region, source in unmaskable_exposures(cfg, isolation, root, worktree_git_dir):
        reporter.warn(
            f"jailed commands can read {hidden}: it sits inside"
            f" {region} ({source}), which they are granted, and 'hardened' has no"
            " mount namespace to mask it out. Every command this run starts can"
            " read the provider keys, transcripts, notes, and run history in there."
            " Use 'strict' to keep them masked under the same grant."
        )
    if isolation in ("strict", "hardened"):
        notes = tool_mount_notes()
        for tool in notes.unreachable:
            reporter.warn(
                f"tool {tool} resolves into a dir that is never"
                " mounted into the jail ($HOME itself, or agent6's private dirs),"
                " so it will not run inside sandboxed commands. Move the target"
                " into its own subdirectory."
            )
        # notes.exposes_home_dir is NOT warned per run: on a normal machine
        # every uv-installed tool in ~/.local/bin points into ~/.local/share,
        # so this fired a dozen times a run and buried the messages that
        # mattered. It is the ordinary state of a dev box, not a surprise --
        # `agent6 check` lists it, where someone is asking.


def warn_cleartext_credential_endpoints(
    cfg: Config, *, reporter: Reporter = STDIO_REPORTER
) -> None:
    """Once per run: an endpoint sending its credential over plaintext http to
    a non-loopback host is explicit-but-discouraged config, so it runs with a
    loud warning naming the cost, never a refusal (an internal-network or VPN
    endpoint is a real case)."""
    for label in cfg.cleartext_credential_endpoints():
        reporter.warn(
            f"{label} sends its credential over plaintext http"
            " to a non-loopback host: anyone on the network path can read it."
            " Use https where you can."
        )


def check_workspace_outside_private_dirs(root: Path) -> str | None:
    """A refusal message when the workspace and one of agent6's own private dirs
    (config, state base) OVERLAP in either direction, else None.

    Workspace inside a private dir: the dir is denied to every in-process tool,
    so the run could not read or write its own files. Private dir inside the
    workspace (a relocated `[agent6].state_dir`): its transcripts and keys become
    readable by jailed commands whose cwd is the workspace, and the auto-commit
    stages them into the run's commits. Wrong everywhere, not an isolation-level
    question.
    """
    resolved = root.resolve()
    for private in private_dirs():
        p = private.resolve()
        if resolved == p or resolved.is_relative_to(p):
            return (
                f"the workspace {str(resolved)!r} is inside agent6's own private"
                f" directory {str(p)!r}, which is hidden from every tool: the run"
                " could not read or write its own files. Work in a directory"
                " outside it."
            )
        if p.is_relative_to(resolved):
            return (
                f"agent6's private directory {str(p)!r} is inside the workspace"
                f" {str(resolved)!r}: its transcripts and keys would be readable"
                " by jailed commands and staged into commits. Keep agent6's state"
                " base outside the workspace."
            )
    return None


def check_protect_git_support(
    cfg: Config, isolation: IsolationLevel, *, explicitly_set: bool
) -> str | None:
    """A refusal message when `protect_git` was EXPLICITLY asked for and this
    isolation cannot provide it, else None.

    `protect_git` is strict-only. Strict re-binds `.git` read-only, which needs
    a mount namespace. On hardened there is none, so the only tool is Landlock,
    which has no deny rules: protecting `.git` means NOT granting the workspace
    root itself, and a Landlock grant is recursive, so granting the root its
    own create/remove rights would grant them over `.git` too. Carving it out
    therefore cost every top-level write -- `touch newfile`, `mkdir build`,
    `mkfifo` all failed at the workspace root, which is too much to pay.

    The default DEGRADES with a warning (see `warn_sandbox_gaps`); an explicit
    `protect_git = true` refuses, naming what is unsupported and the fix. The
    in-process edit tools still refuse writes into `.git` at every level.
    """
    if isolation != "hardened" or not (cfg.sandbox.protect_git and explicitly_set):
        return None
    return (
        "sandbox.protect_git = true requires the strict isolation (a read-only"
        " bind of .git), but this run resolved to 'hardened', where Landlock"
        " could only provide it by refusing every write at the workspace root."
        " Set sandbox.protect_git = false to run here, or use strict."
    )


def check_jail_home(cfg: Config, isolation: IsolationLevel, *, explicitly_set: bool) -> str | None:
    """A refusal when the jail's HOME cannot be what the config says, else
    None.

    `home = "tmp"` is a private tmpfs, which only `strict` has: the default
    degrades to the cache dir with a warning (`warn_sandbox_gaps`), an explicit
    one refuses (the `protect_git` rule). The persistent dir is created by the
    policy builder; this refuses what the builder could not make agent6's own
    (`jail_home_refusal`), without creating anything.
    """
    if isolation != "strict" and explicitly_set and cfg.sandbox.home == "tmp":
        return (
            "sandbox.home = 'tmp' requires the strict isolation (a private /tmp tmpfs),"
            f" but this run resolved to {isolation!r}, where HOME is the persistent"
            f" cache dir {str(jail_cache_home())!r}. Set sandbox.home = 'cache' to"
            " run here, or use strict."
        )
    persistent = persistent_jail_home(cfg, isolation)
    return None if persistent is None else jail_home_refusal(persistent)


def _hardened_grant_regions(
    cfg: Config, root: Path, worktree_git_dir: Path | None = None
) -> tuple[tuple[Path, str], ...]:
    """Every region the hardened launcher grants a command, labeled by its
    source. Derived from the SAME builders the run uses (`jail_policy` for the
    command surface, `mcp_server_policy` per enabled server) so preflight and
    enforcement cannot drift; the fixed sets mirror the launcher's hardened
    ruleset (jail/src/main.rs): /tmp is granted RW, the system roots
    read+exec."""
    policy = jail_policy(root, cfg, "hardened", ("true",), worktree_git_dir=worktree_git_dir)
    regions: list[tuple[Path, str]] = [
        (root, "the workspace"),
        (Path("/tmp"), "the host's shared /tmp (hardened has no private tmpfs)"),  # noqa: S108
    ]
    for sysdir in ("/usr", "/bin", "/sbin", "/lib", "/lib64", "/etc", "/dev"):
        regions.append((Path(sysdir), "a system dir every command is granted"))
    for p in policy.extra_ro_paths:
        if Path(p) == worktree_git_dir:
            regions.append(
                (Path(p), "the repository's .git, which this linked worktree points into")
            )
        else:
            regions.append((Path(p), "sandbox.extra_read_paths"))
    home = persistent_jail_home(cfg, "hardened")
    regions += [
        (Path(p), "the jail's HOME" if p == home else "sandbox.extra_write_paths")
        for p in policy.extra_rw_paths
    ]
    regions += [(Path(p), "an operator tool dir (PATH mount)") for p in policy.tool_paths]
    if cfg.mcp.enabled:
        for name, srv in cfg.mcp.servers.items():
            if not srv.enabled:
                continue
            spol = mcp_server_policy(cfg, root, "hardened", srv)
            if spol is None:
                continue  # unconfined by explicit opt-out; its own loud path
            regions += [(Path(p), f"[mcp.servers.{name}] read_paths") for p in spol.extra_ro_paths]
            regions += [(Path(p), f"[mcp.servers.{name}] write_paths") for p in spol.extra_rw_paths]
            regions.append((Path(spol.cwd), f"[mcp.servers.{name}]'s working dir"))
    return tuple(regions)


def unmaskable_exposures(
    cfg: Config, isolation: IsolationLevel, root: Path, worktree_git_dir: Path | None = None
) -> tuple[tuple[Path, Path, str], ...]:
    """`(hidden path, granted region, region source)` triples this isolation
    cannot mask, hidden-path first. Empty on strict (it masks) and on `none`
    (no jail at all; the blanket unsandboxed warning covers that).

    On `hardened` there is no mount namespace and Landlock has no deny rules,
    so a hidden path OVERLAPPING any granted region is exposed in both
    directions: hidden-inside-grant leaves the whole path readable, and a
    grant INSIDE the hidden tree leaves that part readable. Paths are
    resolved before containment so a `..` spelling cannot dodge the check.
    """
    if isolation != "hardened":
        return ()
    regions = _hardened_grant_regions(cfg, root, worktree_git_dir)
    out: list[tuple[Path, Path, str]] = []
    for h in hidden_paths(Path(p) for p in cfg.sandbox.hide_paths):
        hr = h.resolve()
        for region, source in regions:
            rr = region.resolve()
            if hr.is_relative_to(rr) or rr.is_relative_to(hr):
                out.append((h, region, source))
                break  # one exposing region per hidden path carries the point
    return tuple(out)


def check_hide_paths_support(
    cfg: Config, isolation: IsolationLevel, root: Path, worktree_git_dir: Path | None = None
) -> str | None:
    """A refusal message when an EXPLICIT `[sandbox].hide_paths` entry cannot
    be honored here, else None.

    The same rule the other knobs follow: a default degrades with a warning,
    a value the operator wrote down refuses rather than being silently
    ineffective. `hide_paths` is only ever explicit, so an entry hardened
    cannot mask refuses. The always-hidden private dirs are NOT this: the
    operator granting a region that contains them is a choice they may mean
    (real protection remains -- writes stay confined, seccomp still applies),
    so that is a loud warning instead (`warn_sandbox_gaps`).
    """
    if isolation != "hardened":
        return None  # before reading config: every other level masks
    listed = {Path(p) for p in cfg.sandbox.hide_paths}
    for hidden, region, source in unmaskable_exposures(cfg, isolation, root, worktree_git_dir):
        if hidden in listed:
            return (
                f"sandbox.hide_paths lists {str(hidden)!r}, which sits inside"
                f" {str(region)!r} ({source}), granted to jailed commands."
                " Masking it needs the mount namespace only 'strict' has:"
                " use strict, drop the entry, or move one of the two."
            )
    return None


def mcp_network_refusal(name: str, srv: MCPServerEntry, isolation: IsolationLevel) -> str | None:
    """A refusal when *srv* EXPLICITLY named a network this host cannot give
    it, else None.

    Same rule and same vocabulary as `[sandbox].network`, and therefore the
    same guard: a network namespace needs user namespaces, which only `strict`
    has, so `none` and `session` refuse on `hardened` while the `auto` default
    degrades with a warning. Under `none` nothing is confined at all and the
    blanket unsandboxed warning covers it. One owner: the run's preflight
    (`check_mcp_network_support`) and `agent6 check` both ask here.
    """
    if isolation != "hardened" or not srv.enabled:
        return None
    if srv.effective_network not in ("none", "session"):
        return None
    return (
        f"MCP server {name!r} sets sandbox.network = {srv.effective_network!r},"
        " which needs a network namespace and so the strict isolation; this"
        f" host resolved to {isolation!r}. Use 'auto' to run with a warning,"
        " or 'host' to accept the machine's network."
    )


def check_mcp_network_support(cfg: Config, isolation: IsolationLevel) -> str | None:
    """The first server's `mcp_network_refusal`, else None."""
    for name, srv in sorted(cfg.mcp.servers.items()):
        if (refusal := mcp_network_refusal(name, srv, isolation)) is not None:
            return refusal
    return None


def config_refusal(
    cfg: Config,
    isolation: IsolationLevel,
    workspace: Path,
    *,
    explicit_leaves: frozenset[str] = frozenset(),
    worktree_git_dir: Path | None = None,
) -> str | None:
    """The first refusal for a config this host cannot honor, else None.

    The one list every lifecycle runs (`run`/`resume`/`ask` through
    select_isolation, `machine run` after its interactive network fix), so a
    check added here cannot land in one lifecycle and not the other. The
    NETWORK checks stay per-lifecycle: machines route theirs through
    `resolve_network_fix` and allow per-state opt-ins.
    """
    checks: tuple[Callable[[], str | None], ...] = (
        # First, and one at a time: `check_hide_paths_support` builds the run's
        # policy under hardened, which creates the jail's HOME and raises for
        # one it cannot make. Refusing that here keeps it a message.
        lambda: check_jail_home(cfg, isolation, explicitly_set="sandbox.home" in explicit_leaves),
        lambda: check_mcp_network_support(cfg, isolation),
        lambda: check_hide_paths_support(cfg, isolation, workspace, worktree_git_dir),
        lambda: check_workspace_outside_private_dirs(workspace),
        lambda: check_protect_git_support(
            cfg, isolation, explicitly_set="sandbox.protect_git" in explicit_leaves
        ),
    )
    for check in checks:
        try:
            err = check()
        except JailUnavailableError as exc:
            return str(exc)
        if err is not None:
            return err
    return None


def check_network_support(cfg: Config, isolation: IsolationLevel) -> str | None:
    """A refusal message if the network config EXPLICITLY enforces something
    this isolation cannot provide, else None.

    Only jailed commands have a network boundary. ``network =
    "only_explicit_states"` (singling one tool out) and `"session"`` (the
    run's own network, with no route off the box) both need a network
    namespace, which only `strict` provides. On `hardened` we refuse rather than silently
    under-confine, naming what is unsupported and the fix; `"auto"` is the
    secure default that DEGRADES with a warning instead. On `none` the
    unsandboxed warning already covers it.
    """
    if isolation != "hardened":
        return None
    sb = cfg.sandbox
    if sb.network == "only_explicit_states":
        return (
            "sandbox.network = 'only_explicit_states' requires the strict"
            " isolation (network namespaces), but this run resolved to"
            " 'hardened'. Use 'auto' or 'host'."
        )
    if sb.network == "session":
        return (
            "sandbox.network = 'session' requires the strict isolation (a"
            " network namespace), but this run resolved to 'hardened', where"
            " a jailed command shares this process's network. Use 'auto' to run"
            " with a warning, or 'host' to accept it."
        )
    return None


def resolved_config_values(cfg: Config) -> dict[str, object]:
    """Every config leaf whose effective value differs from its raw one, for a
    config view: the adaptive model settings (`resolved_adaptive_values`) and
    the two `auto` sandbox knobs as this host resolves them, so a surface
    prints `auto` beside the level and network a run here would get. With no
    jail binary to probe, the two stay `auto` (a run here refuses, naming
    the binary)."""
    out = resolved_adaptive_values(cfg)
    if cfg.sandbox.isolation == "auto" or cfg.sandbox.network == "auto":
        try:
            selected = resolve_isolation(cfg.sandbox.isolation, detect_env())
        except JailBinaryError:
            return out
        if cfg.sandbox.isolation == "auto":
            out["sandbox.isolation"] = selected
        if cfg.sandbox.network == "auto":
            out["sandbox.network"] = resolve_network(cfg, selected)
    return out
