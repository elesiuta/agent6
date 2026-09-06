# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""`agent6 check`, sandbox + config + MCP + boundaries + verify pre-flight."""

from __future__ import annotations

import contextlib
import shlex
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from agent6.app._setup import (
    check_provider_keys,
    detect_env,
    mcp_server_policy,
    mcp_server_spec,
    no_jail_cause,
    wants_session_network,
)
from agent6.app.confine import check_network_support, config_refusal, mcp_network_refusal
from agent6.app.fork import worktree_owners
from agent6.config import (
    Config,
    ConfigError,
    MCPServerEntry,
)
from agent6.config.layer import (
    load_effective,
)
from agent6.git_ops import GitError, git_common_dir
from agent6.paths import private_dirs, secrets_path, state_dir
from agent6.sandbox import (
    JailUnavailableError,
    landlock_abi,
    run_in_jail,
)
from agent6.sandbox.detect import (
    Environment,
    IsolationUnavailableError,
    degrade_reason,
    resolve_isolation,
    sandbox_disabled_by_env,
)
from agent6.sandbox.jail import SessionNetwork
from agent6.sandbox.tool_paths import tool_mount_notes
from agent6.tools.mcp_client import MCPManager, tool_count
from agent6.tools.policy import (
    JAIL_TMP_HOME,
    Workspace,
    jail_policy,
    persistent_jail_home,
    resolve_network,
    workspace_for,
)
from agent6.types import CommandResult, IsolationLevel, JailPolicy, SandboxReport
from agent6.verify_infer import infer_verify_command, read_agents_md

# Mirrors SYSTEM_BINDS in src/agent6/jail/src/main.rs (the strict rootfs's
# read-only host binds); tests/security pins the two against each other.
_STRICT_SYSTEM_BINDS = ("/usr", "/bin", "/sbin", "/lib", "/lib64", "/etc/alternatives")
# What hardened's Landlock grants read-only (main.rs ro_paths base set).
_HARDENED_SYSTEM_RO = ("/usr", "/bin", "/sbin", "/lib", "/lib64", "/etc", "/dev")


def _isolation_means(isolation: IsolationLevel) -> str:
    """One line on what this level bounds, for someone diagnosing a tool.

    States the boundaries, not their consequences for any particular program: a
    reader who knows a command runs with its own filesystem, its own network and
    a filtered syscall set can work out why it behaves differently here, and
    knows which words to search the docs for.
    """
    if isolation == "strict":
        # The network clause is hedged because this section runs before any
        # config is loaded (`agent6 check sandbox` needs none): strict CAN give
        # the run its own network, and `sandbox.network = "host"` declines it.
        # The config section prints what this project resolved to.
        return (
            "the run's commands share one jail: their own filesystem view (only"
            " granted paths exist), a private /proc, PID namespace and /tmp, a"
            " filtered syscall set, and the run's own network unless"
            " sandbox.network says otherwise. See docs/security.md."
        )
    if isolation == "hardened":
        return (
            "commands work in the host's own filesystem, /tmp, /proc and"
            " network, shared with everything else on it. Landlock path rules"
            " and a filtered syscall set are the whole boundary."
            " See docs/security.md."
        )
    return "Nothing is confined: commands run as you. See docs/security.md."


def _unprobeable(requested: str) -> str:
    """Why there is no jail to probe: the three ways isolation resolves to `none`."""
    if sandbox_disabled_by_env():
        return "AGENT6_DANGEROUSLY_DISABLE_SANDBOX=1: commands run unconfined; skipped"
    if requested == "none":
        return "sandbox.isolation = 'none': commands run unconfined; skipped"
    return "no kernel sandbox on this host; skipped"


def _cmd_check_sandbox(cfg: Config | None = None) -> int:
    """Run the sandbox boundary self-tests on the host's kernel.

    The probes run under the isolation THIS config resolves to
    (`resolve_isolation(sandbox.isolation, ...)`), so they exercise the sandbox
    `agent6 run` would use here. On a host that blocks unprivileged user
    namespaces (default-seccomp Docker, or Ubuntu with
    `kernel.apparmor_restrict_unprivileged_userns=1`) `auto` resolves to
    `hardened`; testing `strict` there would report a spurious FAIL for a
    sandbox the agent never uses. An explicit level this host cannot give is a
    FAIL naming the refusal, since a run would refuse too. Without a config
    (`cfg is None`) the built-in default `auto` applies.
    """
    reports: list[SandboxReport] = []

    # Landlock probe
    abi = landlock_abi()
    reports.append(
        SandboxReport(
            name="landlock_abi",
            ok=abi > 0,
            detail=f"abi={abi}",
        )
    )

    try:
        env = detect_env()
    except JailUnavailableError as exc:
        reports.append(SandboxReport(name="jail_binary", ok=False, detail=str(exc)))
        return _print_sandbox_reports(reports)
    requested = cfg.sandbox.isolation if cfg is not None else "auto"
    try:
        isolation = resolve_isolation(requested, env)
    except IsolationUnavailableError as exc:
        print(f"  sandbox.isolation = {requested!r} cannot run on this host")
        reports.append(SandboxReport(name="isolation", ok=False, detail=str(exc)))
        return _print_sandbox_reports(reports)
    print(f"  effective isolation ({requested}): {isolation}")
    reason = degrade_reason(env)
    if reason is not None:
        # A degraded level never appears without its why (same line the run
        # warning and check config print; one owner in detect.degrade_reason).
        print(f"  not strict: {reason}")
    # What that level GIVES, in general terms rather than a catalogue of cases:
    # someone whose tool misbehaves needs to know which boundaries exist here
    # before they can guess why, and these words are what to search the docs for.
    print(f"  {_isolation_means(isolation)}")
    notes = tool_mount_notes()
    # Under "none" nothing is confined, so grant language about tool dirs
    # would describe a boundary that does not exist; the block is jail-only.
    if notes.exposes_home_dir and isolation != "none":
        # Where someone is actually asking. Not a per-run warning: on a normal
        # machine every uv-installed tool in ~/.local/bin points into
        # ~/.local/share, so it is the ordinary state of a dev box.
        how = (
            "mounted read-only into the jail and readable by"
            if isolation == "strict"
            else "granted read-only (Landlock path rules) and readable by"
        )
        print(
            f"  {len(notes.exposes_home_dir)} tool(s) resolve out of their bin dir, so those"
            f" target directories are\n  {how}"
            " jailed commands:"
        )
        for tool in notes.exposes_home_dir:
            print(f"    {tool}")
    if isolation == "none":
        # Nothing to probe, and running the boundary probes unconfined would let
        # the /etc-write probe actually escape onto the host.
        reports.append(SandboxReport(name="jail", ok=False, detail=_unprobeable(requested)))
        return _print_sandbox_reports(reports)

    cwd = Path.cwd()

    def _jail(*argv: str) -> CommandResult:
        return run_in_jail(
            JailPolicy(cwd=cwd, argv=argv, isolation=isolation, network="none", timeout_s=10.0)
        )

    # Try running `/usr/bin/true` in the jail.
    try:
        res = _jail("/usr/bin/true")
        reports.append(SandboxReport(name="jail_true", ok=res.ok, detail=f"rc={res.returncode}"))
    except JailUnavailableError as exc:
        reports.append(SandboxReport(name="jail_true", ok=False, detail=str(exc)))

    # Confirm the child cannot reach the network. Only meaningful under
    # `strict`, the one level with network namespaces: there a child that did
    # not ask for `host` lands in one with no route out. `hardened` has none to
    # give, so a jailed command shares this process's network and there is
    # nothing to probe -- report n/a rather than a misleading pass/fail.
    if isolation == "strict":
        try:
            res = _jail("/usr/bin/getent", "hosts", "example.com")
            ok = res.returncode != 0
            reports.append(
                SandboxReport(
                    name="jail_blocks_network",
                    ok=ok,
                    detail=f"rc={res.returncode} (nonzero = blocked, as expected)",
                )
            )
        except JailUnavailableError as exc:
            reports.append(SandboxReport(name="jail_blocks_network", ok=False, detail=str(exc)))
    else:
        reports.append(
            SandboxReport(
                name="jail_blocks_network",
                ok=True,
                detail=(
                    "n/a under hardened: no per-command network namespace; jailed"
                    " commands share the host network (sandbox.network degrades with"
                    " a warning)"
                ),
            )
        )

    # Confirm child cannot write outside the workspace.
    try:
        res = _jail("/bin/sh", "-c", "echo x > /etc/agent6-escape || true")
        # /etc is read-only (bind-mounted RO under strict, Landlock-denied under
        # hardened), so the file must not appear on the host.
        ok = not Path("/etc/agent6-escape").exists()
        reports.append(
            SandboxReport(
                name="jail_blocks_etc_write",
                ok=ok,
                detail=f"rc={res.returncode}; host /etc/agent6-escape exists: {not ok}",
            )
        )
    except JailUnavailableError as exc:
        reports.append(SandboxReport(name="jail_blocks_etc_write", ok=False, detail=str(exc)))

    return _print_sandbox_reports(reports)


def _print_sandbox_reports(reports: list[SandboxReport]) -> int:
    overall_ok = True
    for r in reports:
        status = "PASS" if r.ok else "FAIL"
        print(f"[{status}] {r.name}: {r.detail}")
        overall_ok = overall_ok and r.ok
    return 0 if overall_ok else 1


@dataclass(frozen=True, slots=True)
class _DoctorCheck:
    """One summary row. `status` carries through to the summary line unchanged:
    INFO (advisory, e.g. "run `agent6 connect`") must never render as PASS."""

    name: str
    status: Literal["PASS", "FAIL", "WARN", "INFO"]
    detail: str


def _cmd_check(config_path: Path | None, *, section: str) -> int:
    """Consolidated pre-flight (sandbox + config + MCP + verify).

    The command never spawns the agent loop and never writes to the repo:
    MCP servers are started as a run here starts them (the same workspace
    root, sandbox and network) with the workspace bound read-only, just long
    enough to enumerate their tool descriptors, then closed; one a run would
    refuse, or one this diagnostic must not start (`_probe_refusal`), is
    reported, not started. The one network call is the provider's model
    listing, refreshed for pricing when a key resolves (TTL-gated, ~1.5s,
    never fatal).

    Returns 0 when every selected check passes, 1 otherwise.
    """
    print(f"agent6 check: section={section}")
    print()

    checks: list[_DoctorCheck] = []
    # Every section needs the config: the sandbox probes run under the isolation
    # it selects, so they test the jail a run here would use.
    cfg: Config | None = None
    explicit_leaves: frozenset[str] = frozenset()
    load_error: str | None = None
    try:
        effective = load_effective(Path.cwd(), config_path)
        cfg, explicit_leaves = effective.config, effective.explicit_leaves
    except (ConfigError, OSError) as exc:
        load_error = str(exc)

    if section in {"all", "sandbox"}:
        print("== sandbox ==")
        if load_error is not None:
            print(f"  config unreadable ({load_error}); probing the default isolation")
        rc = _cmd_check_sandbox(cfg)
        checks.append(
            _DoctorCheck(
                name="sandbox",
                status="PASS" if rc == 0 else "FAIL",
                detail="all jail probes passed" if rc == 0 else f"check sandbox exit {rc}",
            )
        )
        print()

    if load_error is not None and section in {"all", "mcp", "verify", "config", "boundaries"}:
        print(f"== config ==\n[FAIL] cannot load config: {load_error}\n")
        checks.append(_DoctorCheck(name="config_load", status="FAIL", detail=load_error))

    if cfg is not None and section in {"all", "config"}:
        print("== config ==")
        checks.extend(_check_config_section(cfg, explicit_leaves))
        print()

    if cfg is not None and section in {"all", "boundaries"}:
        print("== boundaries ==")
        checks.extend(_check_boundaries_section(cfg))
        print()

    if cfg is not None and section in {"all", "mcp"}:
        print("== mcp ==")
        checks.extend(_doctor_check_mcp(cfg))
        print()

    if cfg is not None and section in {"all", "verify"}:
        print("== verify ==")
        checks.extend(_doctor_check_verify(cfg))
        print()

    # `check boundaries` alone reports facts and reaches no verdict, so it
    # prints no summary rather than an empty heading that reads "nothing ran".
    if checks:
        print("== summary ==")
    failed = False
    for c in checks:
        print(f"[{c.status}] {c.name}: {c.detail}")
        failed = failed or c.status == "FAIL"
    return 1 if failed else 0


def _check_config_section(
    cfg: Config, explicit_leaves: frozenset[str] = frozenset()
) -> list[_DoctorCheck]:
    """Environment detection + isolation selection + the refusal ladder a run
    would apply + static config checks."""
    try:
        env = detect_env()
    except JailUnavailableError as exc:
        print(f"  [FAIL] jail binary: {exc}")
        failed = _DoctorCheck(name="config.isolation", status="FAIL", detail=str(exc))
        return [failed, *_doctor_check_config(cfg)]
    print(f"  kernel: {env.kernel.raw}")
    print(f"  userns supported: {env.userns_supported}")
    print(f"  sandbox available: {env.sandbox_available}")
    abi_str = str(env.landlock_abi) if env.sandbox_available else "n/a (no Linux sandbox)"
    print(f"  Landlock ABI: {abi_str}")
    print(
        f"  sandbox.isolation = {cfg.sandbox.isolation}"
        f"  network = {cfg.sandbox.network}"
        f"  run_commands = {cfg.sandbox.run_commands}"
    )
    out: list[_DoctorCheck] = []
    try:
        selected = resolve_isolation(cfg.sandbox.isolation, env)
        # The resolved values, not the configured ones: `auto` is the default on
        # both knobs, and what it resolved to on THIS host is the answer someone
        # runs `check` for.
        print(
            f"  -> selected isolation: {selected}"
            f"  commands' network: {resolve_network(cfg, selected)}"
        )
        reason = degrade_reason(env)
        if cfg.sandbox.isolation == "auto" and reason is not None:
            print(f"  -> not strict: {reason}")
        # The tools' file boundary is NOT the selected isolation's: it follows
        # the config values at every level, so print it beside them rather than
        # leaving the operator to infer it from the level.
        ws = workspace_for(cfg, Path.cwd())
        grants = len({*ws.read_roots, *ws.write_roots})
        print(f"  -> tools' files: {ws.root}  (+{grants} granted, -{len(ws.denied)} hidden)")
        out.append(
            _DoctorCheck(name="config.isolation", status="PASS", detail=f"selected {selected}")
        )
        # The same ladder every run, resume and ask applies to the selected
        # level: an explicit knob this host cannot honour refuses there too.
        refusal = check_network_support(cfg, selected) or config_refusal(
            cfg, selected, Path.cwd(), explicit_leaves=explicit_leaves
        )
        if refusal is not None:
            print(f"  [FAIL] a run would refuse: {refusal}")
            out.append(_DoctorCheck(name="config.refusal", status="FAIL", detail=refusal))
        else:
            out.append(
                _DoctorCheck(
                    name="config.refusal",
                    status="PASS",
                    detail=f"every explicit knob is honoured on {selected}",
                )
            )
    except IsolationUnavailableError as exc:
        print(f"  [FAIL] isolation selection: {exc}")
        out.append(_DoctorCheck(name="config.isolation", status="FAIL", detail=str(exc)))
    out.extend(_doctor_check_config(cfg))
    return out


def _grant_lines(ws: Workspace) -> list[str]:
    """The operator's extra path grants, shared verbatim by both actors."""
    read_only = set(ws.read_roots) - set(ws.write_roots)
    out = [f"    ro  {p}  (sandbox.extra_read_paths)" for p in sorted(read_only)]
    out += [f"    rw  {p}  (sandbox.extra_write_paths)" for p in sorted(ws.write_roots)]
    return out


def _device_lines(cfg: Config) -> list[str]:
    """Operator device-node grants: jail-only (no in-process tool reads them)."""
    return [f"    dev {p}  (sandbox.extra_device_paths)" for p in cfg.sandbox.extra_device_paths]


def _home_line(cfg: Config, selected: IsolationLevel) -> str:
    """The jail's HOME, a grant at every level: strict's tmpfs one, or the
    persistent cache dir and why it is that."""
    persistent = persistent_jail_home(cfg, selected)
    if persistent is None:
        return f"    rw  {JAIL_TMP_HOME}  (HOME, inside the private /tmp)"
    why = "sandbox.home = cache" if selected == "strict" else f"{selected} has no private /tmp"
    return f"    rw  {persistent}  (HOME, persists across runs: {why})"


def _fork_git_grant(cfg: Config, ws: Workspace, selected: IsolationLevel) -> Path | None:
    """The repository git dir a jailed command reaches when the workspace is
    a fork's worktree, as the fork's leg grants it, else None: the
    `worktree_git_dir` of the fork manifest naming the workspace, under the
    repository's state dir (the repository is the worktree's git common
    dir's parent), through the leg's own policy builder, which grants it
    once the worktree's `.git` pointer still names it and raises
    JailUnavailableError otherwise."""
    if selected == "none":
        return None
    try:
        repo = git_common_dir(ws.root).parent
    except GitError:
        return None
    for worktree, sessions in worktree_owners(state_dir(repo)).items():
        if worktree.resolve() != ws.root:
            continue
        git_dir = next((m.worktree_git_dir for _d, m in sessions if m.worktree_git_dir), None)
        if git_dir is not None:
            jail_policy(ws.root, cfg, selected, ("true",), worktree_git_dir=git_dir)
        return git_dir
    return None


def _boundaries_commands(
    cfg: Config, ws: Workspace, selected: IsolationLevel, git_grant: Path | None
) -> None:
    # The resolved fact, not the knob's value: "no" WITHHOLDS the command tools
    # from the model rather than prompting for them, and the paths below are
    # then what an operator-driven jailed command (a machine tool state, an
    # MCP server) reaches.
    gate = {
        "yes": "auto-approved",
        "ask": "prompted per call",
        "no": "withheld from the model",
    }[cfg.sandbox.run_commands]
    print(
        "  jailed commands (run_command, the verify gate, the metric command):"
        f" {gate} (sandbox.run_commands = {cfg.sandbox.run_commands})"
    )
    if selected == "none":
        print("    UNCONFINED: no jail on this host; commands run as you.")
        print(_home_line(cfg, selected))
        return
    git_note = (
        "; .git re-bound read-only" if selected == "strict" and cfg.sandbox.protect_git else ""
    )
    print(f"    rw  {ws.root}  (the workspace{git_note})")
    if git_grant is not None:
        print(
            f"    ro  {git_grant}  (the repository's .git, which this linked worktree points into)"
        )
    if selected == "strict":
        print("    rw  /tmp  (a tmpfs the run's commands share, gone at teardown)")
        print(f"    ro  system: {' '.join(_STRICT_SYSTEM_BINDS)} (the rest of /etc is empty)")
    else:
        print(f"    ro  system (Landlock): {' '.join(_HARDENED_SYSTEM_RO)}")
    print(_home_line(cfg, selected))
    notes = tool_mount_notes()
    if notes.exposes_home_dir:
        print(
            f"    ro  operator tools: {len(notes.exposes_home_dir)} resolved bin-dir"
            " targets (`agent6 check sandbox` lists each)"
        )
    for line in _grant_lines(ws):
        print(line)
    for line in _device_lines(cfg):
        print(line)
    masked = {*private_dirs(), *(Path(p) for p in cfg.sandbox.hide_paths)}
    verb = "masked out of the jail's view" if selected == "strict" else "denied by Landlock"
    for p in sorted(masked):
        print(f"    --  {p}  ({verb})")
    net = resolve_network(cfg, selected)
    net_line = {
        "session": "the run's own network, no route off this machine"
        " (reach it: agent6 exec / agent6 forward)",
        "host": "this machine's network, unconfined",
        "none": "no network at all",
    }.get(net, net)
    print(f"    network  {net}: {net_line}")
    mem = cfg.sandbox.memory_limit_mb
    print(
        f"    memory   {mem} MiB per command (sandbox.memory_limit_mb)"
        if mem > 0
        else "    memory   no cap (sandbox.memory_limit_mb = 0)"
    )


def _boundaries_mcp(cfg: Config, root: Path, selected: IsolationLevel) -> None:
    if not cfg.mcp.enabled or not cfg.mcp.servers:
        cause = "[mcp].enabled = false" if not cfg.mcp.enabled else "no servers configured"
        print(f"  mcp servers: none run ({cause})")
        return
    print(f"  mcp servers ({len(cfg.mcp.servers)} configured, approval per server):")
    for name, srv in sorted(cfg.mcp.servers.items()):
        if not srv.enabled:
            print(f"    {name}: DISABLED")
            continue
        sb = srv.sandbox
        if srv.url:
            where = f"http {srv.url}"
            confinement = "network client only; no filesystem grant"
        elif (refusal := mcp_network_refusal(name, srv, selected)) is not None:
            # The network it asked for is one this level cannot give: a run
            # refuses, so there is no network to print.
            print(f"    {name}: a run would refuse: {refusal}")
            continue
        elif (policy := mcp_server_policy(cfg, root, selected, srv)) is None:
            where = "spawned UNCONFINED"
            confinement = "full host access (mcp.servers.*.sandbox.unconfined = true)"
        else:
            where = "spawned in the jail"
            ro = len(sb.read_paths) if sb else 0
            rw = len(sb.write_paths) if sb else 0
            confinement = f"paths ro+{ro} rw+{rw}, network {policy.network}"
        print(f"    {name}: {where}  approve={srv.approve}  {confinement}")


def _check_boundaries_section(cfg: Config) -> list[_DoctorCheck]:
    """Every boundary in one place, grouped by ACTOR: who is confined, what
    files it reaches, which network it gets. Resolved values only (what THIS
    host and config give), one line per fact; informational, no probes."""
    try:
        env = detect_env()
        selected = resolve_isolation(cfg.sandbox.isolation, env)
    except (IsolationUnavailableError, JailUnavailableError) as exc:
        print(f"[FAIL] isolation selection: {exc}")
        return [_DoctorCheck(name="boundaries", status="FAIL", detail=str(exc))]
    print(f"  isolation: {selected}  (sandbox.isolation = {cfg.sandbox.isolation})")
    reason = degrade_reason(env)
    if cfg.sandbox.isolation == "auto" and reason is not None:
        print(f"  not strict: {reason}")

    ws = workspace_for(cfg, Path.cwd())
    print()
    print("  in-process file tools (read_file, list_dir, apply_edit, apply_patch; no approval):")
    print(f"    rw  {ws.root}  (the workspace)")
    for line in _grant_lines(ws):
        print(line)
    for p in sorted(ws.denied):
        print(f"    --  {p}  (hidden: tools refuse it)")

    print()
    try:
        _boundaries_commands(cfg, ws, selected, _fork_git_grant(cfg, ws, selected))
        print()
        _boundaries_mcp(cfg, ws.root, selected)
    except JailUnavailableError as exc:
        print(f"  [FAIL] a run would refuse: {exc}")
        return [_DoctorCheck(name="boundaries", status="FAIL", detail=str(exc))]
    print()
    print(
        "  agent process: its own egress is NOT bounded (documented in"
        " docs/security.md); the jail bounds commands, not the agent"
    )
    print(
        f"  secrets: {secrets_path()}  (0600; never mounted into any jail,"
        " never passed into a child's env)"
    )
    return []


def _probe_refusal(
    cfg: Config, env: Environment, name: str, srv: MCPServerEntry, isolation: IsolationLevel
) -> str | None:
    """Why the diagnostic leaves *srv* unstarted, or None when a read-only
    probe (the run's sandbox, workspace bound read-only) holds it. The rule:
    a diagnostic never starts a server outside the confinement a run gives
    it and never writes anything but the config it was asked to write, so a
    write grant of any kind, or no jail at all, means no probe."""
    if srv.url:
        return None
    if isolation == "none":
        return f"not probed: no jail ({no_jail_cause(cfg, env)}); a run starts it unconfined"
    sb = srv.sandbox
    if sb is not None and sb.unconfined:
        return (
            f"not probed: mcp.servers.{name}.sandbox.unconfined = true; a run starts it"
            " as configured, unconfined"
        )
    grants = (
        (f"mcp.servers.{name}.sandbox.write_paths", sb.write_paths if sb else ()),
        ("sandbox.extra_write_paths", cfg.sandbox.extra_write_paths),
        ("sandbox.extra_device_paths", cfg.sandbox.extra_device_paths),
    )
    for leaf, paths in grants:
        if paths:
            return (
                f"not probed: {leaf} grants writes ({', '.join(paths)}); a run starts it"
                " as configured"
            )
    return None


def _doctor_check_mcp(cfg: Config) -> list[_DoctorCheck]:
    """Start each enabled MCP server as a run here would (the same workspace
    root, sandbox and network) with the workspace read-only, enumerate its
    tools, then close it. A server a run would refuse is a FAIL row with the
    refusal; one this diagnostic must not start (`_probe_refusal`) is a WARN
    row saying why; neither is started. When `[mcp]` is disabled or empty,
    returns a single skip-style PASS so the doctor doesn't fail an
    unconfigured-by-design feature."""
    if not cfg.mcp.enabled or not cfg.mcp.servers:
        print("(MCP disabled or no servers configured; skipping)")
        return [
            _DoctorCheck(
                name="mcp",
                status="PASS",
                detail="not configured (cfg.mcp.enabled=False or empty servers)",
            )
        ]
    try:
        env = detect_env()
        isolation = resolve_isolation(cfg.sandbox.isolation, env)
    except (IsolationUnavailableError, JailUnavailableError) as exc:
        print(f"[FAIL] mcp: {exc}")
        return [_DoctorCheck(name="mcp", status="FAIL", detail=str(exc))]
    root = Path.cwd()
    out: list[_DoctorCheck] = []
    probed: dict[str, MCPServerEntry] = {}
    for name, srv in sorted(cfg.mcp.servers.items()):
        if not srv.enabled:
            continue
        if (refusal := mcp_network_refusal(name, srv, isolation)) is not None:
            detail = f"a run would refuse: {refusal}"
            print(f"  {name}: {detail}")
            out.append(_DoctorCheck(name=f"mcp.{name}", status="FAIL", detail=detail))
        elif (reason := _probe_refusal(cfg, env, name, srv, isolation)) is not None:
            print(f"  {name}: {reason}")
            out.append(_DoctorCheck(name=f"mcp.{name}", status="WARN", detail=reason))
        else:
            probed[name] = srv
    if not probed:
        return out or [_DoctorCheck(name="mcp", status="PASS", detail="no enabled servers")]
    with contextlib.ExitStack() as stack:
        # A server set to `session` joins the run's network, so `check` has to
        # make one the same way a run does -- otherwise checking such a server
        # reports a failure that only `check` would ever see.
        session_net = None
        if wants_session_network(cfg, isolation):
            session_net = stack.enter_context(contextlib.closing(SessionNetwork.open()))
        try:
            specs = [
                mcp_server_spec(cfg, root, isolation, name, srv, readonly=True)
                for name, srv in probed.items()
            ]
        except JailUnavailableError as exc:
            # The policy builder refused: the jail's HOME cannot be made.
            print(f"[FAIL] mcp: {exc}")
            return [*out, _DoctorCheck(name="mcp", status="FAIL", detail=str(exc))]
        manager = MCPManager.start(specs, session_net=session_net)
        stack.callback(manager.close)
        by_server: dict[str, list[str]] = {}
        for d in manager.descriptors():
            by_server.setdefault(d.server_name, []).append(d.tool_name)
        why_missing = {f.name: f.error for f in manager.failures}
        for name in sorted(probed):
            tools = by_server.get(name, [])
            ok = bool(tools)
            # A server that never started is not one that "exposed no tools":
            # the reason the operator needs is the spawn error, not a symptom.
            # `approve` belongs in a pre-flight for the same reason the network
            # does: it is standing consent for every call this server's tools
            # make, and the operator set it once, possibly a while ago.
            detail = (
                f"{tool_count(len(tools))}, network: {manager.networks[name]},"
                f" approve: {cfg.mcp.servers[name].approve}"
                if ok
                else why_missing.get(name, "started but exposed no tools")
            )
            print(f"  {name}: {detail}")
            out.append(
                _DoctorCheck(name=f"mcp.{name}", status="PASS" if ok else "FAIL", detail=detail)
            )
    return out


def _doctor_check_verify(cfg: Config) -> list[_DoctorCheck]:
    """Verify command sanity: argv non-empty and the head executable resolves.

    Does NOT execute the verify command, that would run an arbitrary
    test suite on every doctor call. Operators can do
    `./$(verify_command)` themselves when they want a live run.
    """
    argv = list(cfg.workflow.verify_command)
    if not argv:
        # Optional: `agent6 run`/`plan` infer one (AGENTS.md -> repo signals ->
        # LLM), else run gateless. Say what THIS repo infers, from the
        # deterministic tiers (the LLM tier is a run's own call). Advisory.
        cwd = Path.cwd()
        inferred = infer_verify_command(cwd, read_agents_md(cwd), llm_call=None)
        detail = (
            f"unset; a run here infers {shlex.join(inferred.argv)} (from {inferred.source})"
            if inferred is not None
            else "unset; nothing here to infer from (a run asks the reviewer model over the"
            " manifests, else goes gateless)"
        )
        print(f"  {detail}")
        return [_DoctorCheck(name="verify.argv", status="INFO", detail=detail)]
    head = argv[0]
    resolved = shutil.which(head)
    ok = resolved is not None
    detail = f"resolves to {resolved}" if resolved else f"not found on PATH: {head!r}"
    print(f"  {head}: {detail}")
    print(f"  argv = {argv}")
    print(f"  timeout = {cfg.workflow.verify_timeout_s}s")
    return [_DoctorCheck(name="verify.head", status="PASS" if ok else "FAIL", detail=detail)]


def _doctor_check_config(cfg: Config) -> list[_DoctorCheck]:
    """Static config sanity checks: provider keys + worktree git policy."""
    out: list[_DoctorCheck] = []
    if not cfg.providers:
        # Zero providers configured: "all referenced keys resolve" is vacuously
        # true and would signal "ready", but `agent6 run` will reject. Say so.
        detail_env = (
            "no providers configured yet; run `agent6 connect` (required before `agent6 run`)"
        )
        out.append(_DoctorCheck(name="config.provider_keys", status="INFO", detail=detail_env))
    else:
        env_err = check_provider_keys(cfg)
        ok_env = env_err is None
        detail_env = "all referenced provider keys resolve" if ok_env else env_err or ""
        out.append(
            _DoctorCheck(
                name="config.provider_keys",
                status="PASS" if ok_env else "FAIL",
                detail=detail_env,
            )
        )

    detail_git = "push/--force/history rewrites are refused unconditionally (git_ops, no override)"
    out.append(_DoctorCheck(name="config.git_policy", status="PASS", detail=detail_git))
    return out
