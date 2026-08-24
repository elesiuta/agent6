# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""`agent6 machine run`: compose the engine and drive a machine to completion.

The engine (`agent6.machine`) is a host-netns supervisor; this module resolves
the sandbox isolation, egress viability, provider keys, budget-price and git
identity preflight, builds the per-`agent`-state runner and the `LiveWorld`, and
calls `drive`. Output routes through the injected `MachineFrontend.reporter`; a
hard tool-network refusal is handed to `frontend.resolve_network_fix` (the one
interactive step, held cli-side). The machine ENGINE is unchanged.
"""

from __future__ import annotations

import contextlib
import dataclasses
import os
import shutil
import sys
import uuid
from collections.abc import Callable
from pathlib import Path

from agent6.app._session import resolve_isolation_or_refuse
from agent6.app._setup import check_provider_keys, detect_env
from agent6.app.confine import (
    check_hide_paths_support,
    config_refusal,
    warn_cleartext_credential_endpoints,
    warn_sandbox_gaps,
)
from agent6.app.frontend import apply_spawned_away_default, approval_scopes
from agent6.app.machine._bundle import validate_bundle
from agent6.app.machine._frontend import MachineFrontend
from agent6.app.machine._preflight import (
    build_machine_notify_hook,
    machine_network_refusal,
    machine_protect_paths,
)
from agent6.app.machine._spend import book_crashed_attempt, machine_spend
from agent6.app.machine_agent import build_machine_agent_runner, clone_at_machine_chain
from agent6.app.parallel import subordinate_workdir_root
from agent6.app.preflight import SessionRefused, budget_preflight
from agent6.app.reporter import Reporter
from agent6.config import Config, ConfigError
from agent6.config.layer import load_effective_with_overlay, resolved_state_dir
from agent6.git_ops import (
    CommitIdentity,
    GitError,
    is_git_repo,
    machine_chain_ref_for,
    paths_dirty,
    verify_git_identity,
)
from agent6.machine import (
    AgentExecResult,
    AgentRequest,
    AgentState,
    EngineError,
    JournalError,
    LiveWorld,
    MachineEnd,
    MachineError,
    MachineJournal,
    ToolPolicyFactory,
    ToolState,
    bundle_drift,
    clear_stop_request,
    drive,
    load_machine,
    machine_lock,
    write_bundle,
)
from agent6.sandbox.jail import JailUnavailableError, run_in_jail
from agent6.sessions.ipc import clear_worker_pid, write_worker_pid
from agent6.sessions.layout import machines_root
from agent6.tools.policy import jail_policy, passthrough_env
from agent6.types import CommandResult, IsolationLevel, JailPolicy, NetworkMode
from agent6.viewmodel.format import format_cost
from agent6.workflows.subrun import SubrunError


def _fail(reporter: Reporter, path: Path, problems: list[str], label: str = "") -> int:
    """Print a FAIL header + problem bullets to stderr; always returns 1."""
    suffix = f" ({label})" if label else ""
    reporter.err(f"FAIL: {path}{suffix}")
    for problem in problems:
        reporter.err(f"  - {problem}")
    return 1


def _transitions(n: int) -> str:
    return f"{n} transition{'' if n == 1 else 's'}"


def uncommitted_refusal(path: Path, cwd: Path) -> str | None:
    """A refusal message if the machine's bundle (`.asm.toml` + `scripts/`)
    has uncommitted changes, else None.

    `machine run` only accepts a committed bundle (docs state-machines.md
    §7.1/§9; the `machine create` hint promises it): a tool/agent executes it
    as trusted logic, so an untracked or dirty piece is unreviewed. One rule
    for both pieces; `machine test` is the ungated iteration loop. Skipped
    outside a git repo (nothing to commit against) and for pieces that resolve
    outside the repo tree."""
    if not is_git_repo(cwd):
        return None
    scripts = path.parent / "scripts"
    pieces = [(path, "machine")] + ([(scripts, "scripts bundle")] if scripts.exists() else [])
    for piece, label in pieces:
        try:
            rel = piece.resolve().relative_to(cwd.resolve()).as_posix()
        except ValueError:
            continue
        try:
            dirty = paths_dirty(cwd, (rel,))
        except GitError as exc:
            # Fail-open (this is a review-discipline gate, not a security
            # boundary), but never SILENTLY: a broken-git environment that
            # can't be probed must be visible, not read as "clean".
            print(
                f"[agent6] WARNING: could not check {rel} for uncommitted changes: {exc}",
                file=sys.stderr,
            )
            continue
        if dirty:
            return (
                f"{piece} has uncommitted changes; `machine run` only accepts a"
                f" committed {label}. Review and commit the bundle first."
            )
    return None


def machine_tool_runner(
    cwd: Path, machine_id: str, clone_root: Path
) -> Callable[[JailPolicy], CommandResult]:
    """A jail runner that executes each tool policy in the machine's own tree.

    Run states commit to the machine chain and never touch the checkout the
    policy was built against, so a tool state jailed there cannot see their
    work; each call runs in a fresh clone at the chain tip instead (the same
    tree a run state starts from). Tree writes are scratch, discarded with the
    clone: the durable channels stay the blackboard and
    `$AGENT6_MACHINE_DATA_DIR`, which is exactly what tool tree-writes were
    before, since a run state's clone never contained them either. Bundle
    protect paths under *cwd* are remapped to the clone's own copy, like a run
    state's."""

    def run(policy: JailPolicy) -> CommandResult:
        dest = clone_root / f"tool-{uuid.uuid4().hex[:12]}"
        try:
            clone_at_machine_chain(cwd, dest, machine_chain_ref_for(machine_id))
        except (SubrunError, GitError) as exc:
            shutil.rmtree(dest, ignore_errors=True)
            raise JailUnavailableError(f"machine tree clone failed: {exc}") from exc
        try:
            return run_in_jail(
                dataclasses.replace(
                    policy,
                    cwd=dest,
                    extra_protect_paths=tuple(
                        dest / p.relative_to(cwd) if p.is_relative_to(cwd) else p
                        for p in policy.extra_protect_paths
                    ),
                )
            )
        finally:
            shutil.rmtree(dest, ignore_errors=True)

    return run


def machine_tool_policy_factory(
    cfg: Config,
    cwd: Path,
    isolation: IsolationLevel,
    *,
    protect_paths: tuple[Path, ...],
    data_dir: Path | None,
) -> ToolPolicyFactory:
    """Per-call tool-jail policies for a machine, through the ONE shared
    builder (`jail_policy`) plus the machine deltas: the bundle's protect
    paths and the data dir's RW grant + `$AGENT6_MACHINE_DATA_DIR`. Operator
    grants, `protect_git`, hidden paths, env, and tool mounts therefore hold
    in machine tool jails exactly as in run commands."""
    env_base = passthrough_env()
    extra_rw: tuple[Path, ...] = ()
    if data_dir is not None:
        # Exported to match where the jail mounts the dir: extra_rw_paths mount
        # at their real locations on every isolation level.
        env_base["AGENT6_MACHINE_DATA_DIR"] = str(data_dir)
        extra_rw = (data_dir,)

    def build(argv: tuple[str, ...], timeout_s: float, network: NetworkMode) -> JailPolicy:
        return jail_policy(
            cwd,
            cfg,
            isolation,
            argv,
            timeout_s=timeout_s,
            network=network,
            extra_rw_paths=extra_rw,
            extra_protect_paths=protect_paths,
            env_base=env_base,
        )

    return build


def run_machine(  # noqa: PLR0911, PLR0912, PLR0915
    path: Path,
    frontend: MachineFrontend,
    *,
    config_path: Path | None = None,
    exit_on_wait: bool = False,
    disable_sandbox: bool = False,
    auto_approve: bool = False,
    no_commands: bool = False,
) -> int:
    reporter = frontend.reporter
    if disable_sandbox:
        # Set the env setter so this supervisor's resolve_isolation resolves to
        # none; it then passes that isolation to each agent subprocess in its
        # request (the subprocess takes req.isolation as given, re-checking
        # only that this host supports it).
        # Using the env (vs mutating cfg) is the simplest single knob; the env
        # is operator-controlled and the LLM cannot reach it.
        os.environ["AGENT6_DANGEROUSLY_DISABLE_SANDBOX"] = "1"
    if auto_approve:
        # The operator's per-invocation run_command grant, reaching each agent
        # subprocess the same way the sandbox setter does (env, operator-only,
        # structurally LLM-unreachable). The subprocess applies it through
        # `with_sandbox_overrides`, which upgrades ask -> yes but never
        # resurrects a withheld "no".
        os.environ["AGENT6_AUTO_APPROVE"] = "1"
    if no_commands:
        # The tightening counterpart, carried the same way: each agent state's
        # subprocess withholds every command tool, as `run --no-commands` does.
        os.environ["AGENT6_NO_COMMANDS"] = "1"
    try:
        spec = load_machine(path)
    except MachineError as exc:
        return _fail(reporter, path, list(exc.problems))
    # Re-validate the script bundle before executing anything: `load_machine`
    # does not, and on an isolation level that cannot RO-bind the bundle a `scripts/`
    # symlink escaping it (which `machine check` rejects) would otherwise be read
    # by a tool. Security boundary, so run enforces it too, not just check.
    bundle_problems = validate_bundle(spec, path)
    if bundle_problems:
        return _fail(reporter, path, bundle_problems, "bundle")
    cwd = Path.cwd()
    # Machines are operator artifacts: refuse an uncommitted file before running
    # anything (docs §7.1/§9), so a tool/agent never executes unreviewed logic.
    uncommitted = uncommitted_refusal(path, cwd)
    if uncommitted is not None:
        reporter.refuse(uncommitted)
        return 2
    states = list(spec.states.values())
    has_agent_state = any(getattr(s, "kind", None) == "agent" for s in states)
    # mode="run" agent states edit + commit; they need a resolved git identity.
    has_run_agent = any(isinstance(s, AgentState) and s.mode == "run" for s in states)
    tool_states = [s for s in states if isinstance(s, ToolState)]
    agent_runner: Callable[[AgentRequest, Path | None], AgentExecResult] | None = None
    # Default isolation for confinement-free machines: resolve from the host.
    env = detect_env()
    isolation: IsolationLevel = env.detected_isolation
    # The running machine's own file + scripts bundle are read-only in every
    # run jail, so a tool/agent can't rewrite its own logic or bundled scripts.
    protect_paths = machine_protect_paths(path, cwd)
    # Load the effective config (machine [config] overlay included) for EVERY
    # machine: a pure wait/branch machine still reads [machine] snapshot_keep from
    # it, and validating the overlay up front means a bad overlay or an ignored
    # snapshot_keep never slips through to a pure machine. The agent/tool block
    # below adds the provider/sandbox checks only those state kinds need.
    try:
        eff = load_effective_with_overlay(cwd, spec.config, explicit_path=config_path)
        cfg = eff.config
    except ConfigError as exc:
        reporter.error(str(exc))
        return 2
    cfg = cfg.with_sandbox_overrides(auto_approve=auto_approve, no_commands=no_commands)
    if has_run_agent and cfg.sandbox.run_commands == "ask":
        # Say the dead-end up front: an unattended machine auto-denies every
        # run_command under 'ask' (machine bridges deny when no front-end is
        # attached), so a mode='run' state that shells out burns its budget
        # against denials.
        reporter.note(
            "NOTE: this machine has mode='run' agent state(s) and"
            " sandbox.run_commands='ask'; an unattended machine auto-denies"
            " run_command. Approve for this invocation with --auto-approve, or"
            " set `agent6 config set --repo sandbox.run_commands yes` to always"
            " allow. Edits and the auto-commit need no approval; verify shares"
            " run_command's gate, so an unattended leg ends unverified."
        )
    snapshot_keep = cfg.machine.snapshot_keep
    # One clone base for every state of a machine that writes: the agent
    # states' per-state clones and the tool states' per-call trees.
    clone_root = subordinate_workdir_root(cfg, cwd, f"machine-{spec.machine}")
    if has_agent_state or tool_states:
        try:
            if has_agent_state:
                cfg.require_runnable("worker")
        except ConfigError as exc:
            reporter.error(str(exc))
            return 2
        try:
            isolation = resolve_isolation_or_refuse(cfg, env, reporter=reporter)
        except SessionRefused as refusal:
            return refusal.rc
        snapshot_keep = cfg.machine.snapshot_keep
        refusal = machine_network_refusal(cfg, isolation, tool_states) or check_hide_paths_support(
            cfg, isolation
        )
        if refusal is not None:
            outcome = frontend.resolve_network_fix(
                path, refusal, cfg, isolation, tool_states, cwd, spec.config
            )
            if isinstance(outcome, int):
                return outcome
            cfg, isolation = outcome  # fix applied + re-validated clear; continue
        cfg_err = config_refusal(cfg, isolation, cwd, explicit_leaves=eff.explicit_leaves)
        if cfg_err is not None:
            reporter.refuse(cfg_err)
            return 2
        if has_agent_state:
            # The machine's statically reachable routes include every agent
            # state's provider/model pins; discovering a dead route only when
            # that state fires wasted the run up to it.
            agent_states = [s for s in spec.states.values() if isinstance(s, AgentState)]
            pinned_providers = [s.provider for s in agent_states if s.provider]
            pinned_routes = [
                (s.provider or "", s.model) for s in agent_states if s.model != "inherit"
            ]
            missing = check_provider_keys(cfg, extra_providers=pinned_providers)
            if missing is not None:
                reporter.err(missing)
                return 2
            # After check_provider_keys so the price cache has been refreshed.
            budget_err = budget_preflight(cfg, extra_routes=pinned_routes)
            if budget_err is not None:
                reporter.refuse(budget_err)
                return 2
            # Resolve the commit identity HERE on the host, where global git
            # config is visible, so a mode="run" state's confined agent (which
            # can't read ~/.gitconfig under Landlock) still commits cleanly. A
            # missing identity fails loudly up front, not as mid-loop noise.
            commit_identity: CommitIdentity | None = None
            if has_run_agent:
                base = CommitIdentity(name=cfg.git.commit.name, email=cfg.git.commit.email)
                try:
                    name, email = verify_git_identity(cwd, base)
                except GitError as exc:
                    reporter.error(str(exc))
                    return 2
                commit_identity = CommitIdentity(name=name, email=email)
            root = machines_root(resolved_state_dir(cwd)) / spec.machine
            # The engine is a host-netns supervisor; each agent state runs in
            # its own subprocess.
            agent_runner = build_machine_agent_runner(
                spec.config,
                cwd,
                isolation,
                root / "agent_transcripts",
                protect_paths,
                commit_identity,
                # A machine that writes never touches the checkout: every
                # agent state works a fresh clone at the machine chain's tip,
                # and each mode="run" state lands both the chain (next
                # state's continuation) and the visible agent6/machine-<id>
                # branch (the operator's handle) back per state -- the lane
                # mechanism, sequential where lanes are parallel.
                machine_id=spec.machine if has_run_agent else None,
                clone_root=clone_root if has_run_agent else None,
            )
    warn_sandbox_gaps(isolation, env, cfg, reporter=reporter)
    warn_cleartext_credential_endpoints(cfg, reporter=reporter)
    root = machines_root(resolved_state_dir(cwd)) / spec.machine
    journal = MachineJournal(root, snapshot_keep=snapshot_keep)
    # Persistent, writable scratch for tool scripts (see LiveWorld.data_dir).
    data_dir = root / "data"
    try:
        with machine_lock(root):
            journal.ensure_dirs()
            # Refuse a rerun of an ended instance BEFORE any worker.pid stamp: a
            # terminal journal (last event a MachineEnd) can only be replayed,
            # not advanced. Stamping the pid here would trip spawn_and_confirm's
            # started() (false "started"), and the dead child's zombie pid would
            # then read "running" forever in `machine status`.
            events = journal.read()
            if events and isinstance(events[-1], MachineEnd):
                end = events[-1]
                reporter.refuse(
                    f"{spec.machine} already ended in {end.state!r}"
                    f" ({end.status}: {end.reason})."
                    " Replay it with `agent6 machine replay`, or archive the"
                    f" instance directory to start fresh: {journal.root}"
                )
                return 2
            if journal.exists():
                # A live instance runs the bundle it recorded: continuation
                # holds the working bundle to those bytes, so an edit can
                # never execute under the old instance's identity.
                drift = bundle_drift(root, path)
                if drift is not None:
                    reporter.refuse(
                        f"{drift}. A live instance runs the bundle it"
                        " recorded; archive the instance directory to start"
                        f" fresh with the edited machine: {journal.root}"
                    )
                    return 2
            data_dir.mkdir(parents=True, exist_ok=True)
            # A leftover stop marker from a prior invocation would park this
            # one at its first boundary; starting the machine is the answer to
            # any stale request (mirrors the session-side stale-marker clear).
            clear_stop_request(root)
            # A crash mid-agent-state left its metered spend only in the
            # per-state log; book it into the journal before the drive re-runs
            # the state (which would start a fresh log over it).
            book_crashed_attempt(journal, root)
            # A hub-spawned machine (web/TUI: AGENT6_DETACHED_AWAY=wait) parks
            # its approvals/questions for the front-end instead of the
            # headless deny -- the same detach semantics a spawned run gets.
            apply_spawned_away_default(root, approval_scopes(cfg))
            # Liveness marker for watchers (the web SSE stream probes it to
            # tell a crashed machine from a parked one), mirroring cli/run.py.
            write_worker_pid(root, os.getpid())
            if not journal.exists():
                write_bundle(root, path)
            # Operator argv fired on machine.notify/machine.end, on the host
            # outside the jail (None when [machine.notify].on_event is unset).
            operator_hook = build_machine_notify_hook(cfg, spec.machine, root)

            def surface_notify(kind: str, state: str, message: str, level: str) -> None:
                # The foreground run is its own watcher: a notify message that
                # is journal-only never reaches the operator sitting right here.
                # Presentation must never affect control flow (engine contract),
                # so a dead stderr (EPIPE, full disk on a detached spawn's log)
                # is swallowed rather than killing the machine un-journaled.
                if kind == "notify":
                    with contextlib.suppress(OSError):
                        reporter.note(f"notify [{level}] {state!r}: {message}")
                if operator_hook is not None:
                    operator_hook(kind, state, message, level)

            world = LiveWorld(
                cwd=cwd,
                journal=journal,
                agent_runner=agent_runner,
                tool_policy=machine_tool_policy_factory(
                    cfg, cwd, isolation, protect_paths=protect_paths, data_dir=data_dir
                ),
                # A machine that writes runs its tool states in its own tree,
                # so an edit-then-check loop sees the run states' commits.
                jail_runner=(
                    machine_tool_runner(cwd, spec.machine, clone_root) if has_run_agent else None
                ),
                data_dir=data_dir,
                # Each agent state writes its own watchable logs.jsonl here, so a
                # running machine is followable like a run (pruned to keep recent).
                state_log_root=root / "states",
                state_log_keep=cfg.machine.state_log_keep,
                notify_hook=surface_notify,
            )
            try:
                result = drive(spec, journal, world, live=True, exit_on_wait=exit_on_wait)
            finally:
                # The worker is done with the machine on every exit (ended,
                # waiting, stopped, error): a stale pid file would read
                # "running" wherever the pid number stays alive.
                clear_worker_pid(root)
    except (JournalError, EngineError) as exc:
        reporter.error(str(exc))
        return 1
    if result.status == "waiting":
        reporter.out(
            f"WAITING: {spec.machine} paused in {result.state!r}"
            f" after {_transitions(result.transitions)} ({result.reason})"
        )
        return 0
    if result.status == "stopped":
        reporter.out(
            f"STOPPED: {spec.machine} parked in {result.state!r}"
            f" after {_transitions(result.transitions)} ({result.reason});"
            " resume with `agent6 machine run`."
        )
        return 0
    spend, _ = machine_spend(journal.read(), root, alive=False)
    reporter.out(
        f"{result.status.upper()}: {spec.machine} ended in {result.state!r}"
        f" after {_transitions(result.transitions)} ({result.reason});"
        f" spent {format_cost(spend.usd, partial=spend.partial)}"
    )
    return 0 if result.status == "ok" else 1
