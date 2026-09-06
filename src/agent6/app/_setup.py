# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Run/resume lifecycle setup shared by the front-end adapters: sandbox env
detection, provider-key preflight, per-invocation budget/sandbox override
values, and MCP server startup."""

from __future__ import annotations

import argparse
from collections.abc import Iterable
from dataclasses import dataclass, replace
from pathlib import Path

from pydantic import ValidationError

from agent6.app.reporter import STDIO_REPORTER, Reporter
from agent6.budget import BudgetTracker
from agent6.child_env import curated_env, set_provider_key_env
from agent6.config import (
    AnthropicProviderEntry,
    ChatGPTProviderEntry,
    ClaudeCodeProviderEntry,
    Config,
    ConfigError,
    MCPServerEntry,
    ProviderEntry,
    plan_metered,
)
from agent6.config.layer import EffectiveConfig, load_effective
from agent6.events import EventSink
from agent6.git_ops import set_repo_filter_policy, set_repo_hook_policy
from agent6.models.cache import list_models, refresh_pricing_catalog
from agent6.providers.claude_code import login_status
from agent6.sandbox import strict_namespaces_work
from agent6.sandbox.detect import Environment, degrade_reason, detect, sandbox_disabled_by_env
from agent6.sandbox.jail import SessionNetwork
from agent6.secrets import SecretsError, load_oauth_tokens, load_secrets, resolve_api_key
from agent6.tools.mcp_client import MCPManager, MCPServerSpec
from agent6.tools.mcp_http import HttpTransport
from agent6.tools.policy import jail_policy
from agent6.types import IsolationLevel, JailPolicy, NetworkMode, session_kind
from agent6.workflows.review import parse_seat_spec


def detect_env() -> Environment:
    """`detect()` with an authoritative strict re-check via the jail binary.

    `detect.probe_userns_supported` runs `unshare -U -r true`, which answers a
    narrower question than "can the jail set up a strict sandbox", and is wrong
    in BOTH directions:

    - It under-reports on an AppArmor-restricted host (Ubuntu 24.04+) where a
      profile grants the *agent6-jail* binary userns but not `/usr/bin/unshare`.
    - It over-reports inside Docker with a relaxed seccomp profile, where
      `unshare` succeeds and the default AppArmor profile then denies the jail's
      `mount`. Measured: every command died with a raw "namespace setup failed:
      EACCES" instead of the run degrading to `hardened`.

    So the real jail binary settles it either way. It costs one short jail spawn
    at startup, cached for the process lifetime. A binary the kernel cannot
    execute raises JailBinaryError out of the probe: the callers refuse with
    it, and `auto` never resolves to `hardened` over it.
    """
    env = detect()
    if not env.sandbox_available:
        return env
    works = strict_namespaces_work()
    if works != env.userns_supported:
        return replace(env, userns_supported=works)
    return env


def budget_tracker(cfg: Config, *, max_usd: float | None = None) -> BudgetTracker:
    """A run's meter from `[budget]`, *max_usd* (a flag) overriding the cap."""
    return BudgetTracker(
        max_usd=cfg.budget.max_usd if max_usd is None else max_usd,
        max_percent=cfg.budget.max_percent,
        allow_paid_credits=cfg.budget.allow_paid_credits,
        max_tokens_fallback=cfg.budget.max_tokens_fallback,
    )


@dataclass(frozen=True, slots=True)
class BudgetOverrides:
    """Per-run budget overrides parsed from `--max-*` flags."""

    max_usd: float | None = None
    max_tokens_fallback: int | None = None
    max_percent: float | None = None

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> BudgetOverrides:
        return cls(
            max_usd=getattr(args, "max_usd", None),
            max_tokens_fallback=getattr(args, "max_tokens_fallback", None),
            max_percent=getattr(args, "max_percent", None),
        )

    def apply(self, cfg: Config) -> Config:
        try:
            return cfg.with_budget_overrides(
                max_usd=self.max_usd,
                max_tokens_fallback=self.max_tokens_fallback,
                max_percent=self.max_percent,
            )
        except ValidationError as exc:
            # The schema speaks in config keys; the operator typed a flag. Name
            # what they typed, and refuse the way `config set` refuses rather
            # than escaping to the crash reporter.
            raise ConfigError(self._flag_error(exc)) from exc

    def argv(self) -> list[str]:
        """These overrides as the flags that set them, for a continuation
        this invocation spawns (a detached resume)."""
        out: list[str] = []
        if self.max_usd is not None:
            out += ["--max-usd", str(self.max_usd)]
        if self.max_tokens_fallback is not None:
            out += ["--max-tokens-fallback", str(self.max_tokens_fallback)]
        if self.max_percent is not None:
            out += ["--max-percent", str(self.max_percent)]
        return out

    def _flag_error(self, exc: ValidationError) -> str:
        flags = {
            "max_usd": "--max-usd",
            "max_tokens_fallback": "--max-tokens-fallback",
            "max_percent": "--max-percent",
        }
        parts: list[str] = []
        for err in exc.errors():
            field = str(err["loc"][-1]) if err["loc"] else ""
            parts.append(f"{flags.get(field, field)}: {err['msg']}")
        return "; ".join(parts) or str(exc)


def override_flags(budget: BudgetOverrides | None, sandbox: SandboxOverrides | None) -> list[str]:
    """The CLI flags a continuation this invocation spawns (a detached resume)
    carries so it runs under the same overrides."""
    return [*(budget.argv() if budget else []), *(sandbox.argv() if sandbox else [])]


@dataclass(frozen=True, slots=True)
class SandboxOverrides:
    """Per-invocation sandbox/approval overrides from CLI flags.

    `--dangerously-disable-sandbox` runs unconfined; `--auto-approve`
    auto-approves every jailed command; `--no-commands` withholds them
    entirely (what `/btw` spawns its side question with). The env setter for the sandbox is read in
    `detect.resolve_isolation` (so it also reaches machine subprocesses), so
    `from_args` reads only the flags. Flags and env are structurally
    LLM-unreachable."""

    disable_sandbox: bool = False
    auto_approve: bool = False
    no_commands: bool = False

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> SandboxOverrides:
        return cls(
            disable_sandbox=bool(getattr(args, "dangerously_disable_sandbox", False)),
            auto_approve=bool(getattr(args, "auto_approve", False)),
            no_commands=bool(getattr(args, "no_commands", False)),
        )

    def argv(self) -> list[str]:
        """These overrides as the flags that set them (see BudgetOverrides.argv)."""
        flags = (
            ("--dangerously-disable-sandbox", self.disable_sandbox),
            ("--auto-approve", self.auto_approve),
            ("--no-commands", self.no_commands),
        )
        return [flag for flag, on in flags if on]

    def apply(self, cfg: Config) -> Config:
        return cfg.with_sandbox_overrides(
            disable_sandbox=self.disable_sandbox,
            auto_approve=self.auto_approve,
            no_commands=self.no_commands,
        )


def apply_git_ops_policy(cfg: Config) -> None:
    """Set how agent6's OWN git ops (run outside the jail) treat repo-controlled
    host code and provider secrets, from the run's config. One call per entry
    point (run, resume, merge, machine), so the policy is set the same way
    everywhere; git_ops itself stays config-free.

    - Repo `.git/hooks/*` fire only under `git.run_repo_hooks` (default off): a
      hook is repo-controlled host code, an RCE vector on an untrusted repo.
    - Repo content drivers (`filter.*`, `merge.*.driver`) run only under
      `git.run_repo_filters` (default off) -- same threat, the Git-LFS opt-in.
    - The configured provider-key env vars are stripped from git's environment:
      git never needs a provider key, and a git subprocess (a credential
      helper, a content driver we could not neutralize) should not inherit one.
    """
    set_repo_hook_policy(cfg.git.run_repo_hooks)
    set_repo_filter_policy(cfg.git.run_repo_filters)
    set_provider_key_env(
        p.api_key_env
        for p in cfg.providers.values()
        if not isinstance(p, ClaudeCodeProviderEntry) and p.api_key_env
    )


def session_config(cfg: Config, mode: str, overrides: SandboxOverrides | None = None) -> Config:
    """The effective config for a session of *mode*.

    Both lifecycles call this before anything reads a knob, so a fresh session
    and a resumed one are governed identically. Today it is the interactive-mode
    clamp (ask, plan); anything else mode-dependent belongs here rather than at
    one call site.

    *overrides* are the operator's per-invocation flags, and they land LAST:
    the most specific layer, and the one the LLM cannot reach. The clamp exists
    to catch a STANDING `run_commands = "yes"` that nobody is watching, not an
    explicit `--auto-approve` on this invocation -- clamping that made the flag
    inert and every headless `ask --auto-approve` refused. Tightening still wins
    outright: `--no-commands` pins "no", and `--auto-approve` never resurrects a
    withheld one.
    """
    clamped = cfg.with_run_commands_clamped() if session_kind(mode).clamps_commands else cfg
    return clamped if overrides is None else overrides.apply(clamped)


def load_session_config(
    cwd: Path,
    config_path: Path | None,
    *,
    mode: str,
    preset: str = "",
    budget_overrides: BudgetOverrides | None = None,
    sandbox_overrides: SandboxOverrides | None = None,
) -> EffectiveConfig:
    """The config a session of *mode* starts or resumes under, built the same
    way at every entry point (`agent6 run`, `resume`, an editor's ACP turn):
    the effective layers for *preset*, the git policy set from them, the
    budget flags, `session_config` (the interactive clamp with the sandbox
    flags landing last), checked runnable for the mode's role. Raises
    ConfigError like `load_effective`."""
    effective = load_effective(cwd, config_path, preset=preset)
    cfg = effective.config
    apply_git_ops_policy(cfg)
    if budget_overrides is not None:
        cfg = budget_overrides.apply(cfg)
    cfg = session_config(cfg, mode, sandbox_overrides)
    cfg.require_runnable(session_kind(mode).role)
    return replace(effective, config=cfg)


def check_provider_keys(cfg: Config, extra_providers: Iterable[str] = ()) -> str | None:
    """Return an error message if any referenced provider has no resolvable key.

    A key may come from the env var named by `api_key_env` or from
    `secrets.toml` (via `agent6 connect`). Checked over every provider the
    run can STATICALLY reach: the configured `[models.<role>]` entries, any
    provider a `[review].seats` spec pins, and *extra_providers* (a machine's
    per-state pins) -- a route discovered only mid-run used to fail after
    state existed and spend started. OpenAI-compat providers with no key
    configured at all are skipped (unauthenticated local endpoints like
    Ollama).
    """
    try:
        secrets = load_secrets()
    except SecretsError as exc:
        return str(exc)
    needed = {rm.provider for rm in cfg.models.configured().values()}
    for spec in cfg.review.seats:
        _persona, seat_provider, _model = parse_seat_spec(spec)
        if seat_provider:
            needed.add(seat_provider)
    needed.update(p for p in extra_providers if p)
    if absent := sorted(needed - cfg.providers.keys()):
        return (
            f"no [providers.{absent[0]}] entry, but a model route references it"
            " (a role, a review seat, or a machine state pin). Add the provider"
            " or fix the reference."
        )
    for name in sorted(needed):
        if (err := _provider_refusal(name, cfg.providers[name], secrets)) is not None:
            return err
    if "openrouter" not in needed and any(
        rm.model.startswith("claude-")
        and "/" not in rm.model
        and not plan_metered(cfg.providers.get(rm.provider))
        for rm in cfg.models.configured().values()
    ):
        # Bare claude-* ids price through the OpenRouter catalog (pricing's
        # alias); with no openrouter provider configured nothing above
        # fetched it, and the $ cap would run honestly-but-needlessly
        # unpriced on a cold cache. A plan-metered route is an authoritative
        # $0 and needs no price.
        refresh_pricing_catalog()
    return None


def _provider_refusal(name: str, entry: ProviderEntry, secrets: dict[str, str]) -> str | None:
    """Why one routed provider cannot run, or None: a claude_code binary that is
    not signed in, a chatgpt block with no stored sign-in, an Anthropic block
    with no key. A keyed or local endpoint passes, refreshing its model listing
    on the way (TTL-gated, ~1.5s, never raises): that cache is what feeds model
    completion, context-window sizing, and the PRICES the budget meters with."""
    if isinstance(entry, ClaudeCodeProviderEntry):
        # No key: the binary carries the operator's own login; check it is
        # signed in before any state exists.
        err = login_status(entry.binary)
        return f"[providers.{name}]: {err}" if err is not None else None
    if isinstance(entry, ChatGPTProviderEntry):
        if load_oauth_tokens(name, secrets=secrets) is None:
            return f"no ChatGPT sign-in stored for [providers.{name}]; run `agent6 connect {name}`."
        list_models(name, entry, None)
        return None
    key = resolve_api_key(name, entry.api_key_env, secrets=secrets)
    if key:
        list_models(name, entry, key)
        return None
    if (
        isinstance(entry, AnthropicProviderEntry)
        and not entry.token_command
        and entry.auth_style != "none"
    ):
        return (
            f"no API key for [providers.{name}] (Anthropic). Run"
            f" `agent6 connect` or set the {entry.api_key_env or 'API key'} env var."
        )
    # Minted by a command (checked at call time), not required, or an
    # OpenAI-compatible endpoint (local ones legitimately need no key).
    return None


def wants_session_network(cfg: Config, isolation: IsolationLevel) -> bool:
    """Whether this run needs its own network: any child that would join one.

    Asked once, before anything spawns, because the network has to exist before
    its first member. Only strict can provide one; elsewhere every child shares
    the host's (preflight has already warned or refused).
    """
    if isolation != "strict":
        return False
    if cfg.sandbox.network != "host":
        return True
    return cfg.mcp.enabled and any(
        srv.enabled and srv.effective_network == "session" for srv in cfg.mcp.servers.values()
    )


def mcp_server_policy(
    cfg: Config,
    root: Path,
    isolation: IsolationLevel,
    srv: MCPServerEntry,
    *,
    readonly: bool = False,
) -> JailPolicy | None:
    """The sandbox for one spawned server, or None when the operator opted it
    out with `unconfined = true`.

    The same `jail_policy` a jailed command gets, plus this server's additive
    grants -- so the block names only what is extra and never has to describe
    the interpreter, the tool dirs, or a writable HOME. `readonly` binds the
    workspace read-only on top (the re-bind `.git` gets, applied to the
    root): a diagnostic's probe, which must not write the repository.

    Its env is the CURATED set rather than a command's passthrough: a server
    is third-party code that may log or forward what it was given, so it gets
    the base plus the variables named in `pass_env`, and never the desktop
    addresses (the session bus reaches an unconfined `systemd --user` that
    runs commands on request, which walks straight out of any sandbox).
    """
    sandbox = srv.sandbox
    if sandbox is not None and sandbox.unconfined:
        return None
    read_paths = sandbox.read_paths if sandbox else ()
    write_paths = sandbox.write_paths if sandbox else ()
    # auto and none both mean "a network of its own"; they differ only in what
    # happens when the host cannot provide one (warn vs refuse, which preflight
    # owns). `session` is the run's shared one; `host` is the machine's.
    configured = srv.effective_network
    network: NetworkMode = "none" if configured == "auto" else configured
    return jail_policy(
        root,
        cfg,
        isolation,
        srv.command,
        extra_ro_paths=tuple(Path(p).expanduser() for p in read_paths),
        extra_rw_paths=tuple(Path(p).expanduser() for p in write_paths),
        extra_protect_paths=(root,) if readonly else (),
        network=network,
        env_base=curated_env(passthrough=srv.pass_env, desktop=False),
    )


def mcp_server_spec(
    cfg: Config,
    root: Path,
    isolation: IsolationLevel,
    name: str,
    srv: MCPServerEntry,
    *,
    readonly: bool = False,
) -> MCPServerSpec:
    """What starting *srv* as *name* takes. One builder for a run, `agent6
    check` and `agent6 mcp connect`, so every surface spawns the server the
    same way: the same workspace root, sandbox and network. The two probes
    (`check mcp`, `mcp connect`) pass `readonly` (see `mcp_server_policy`)."""
    return MCPServerSpec(
        name=name,
        command=srv.command,
        startup_timeout_s=srv.startup_timeout_s,
        call_timeout_s=srv.call_timeout_s,
        pass_env=srv.pass_env,
        # A `url` server is the operator's own process: nothing to confine.
        policy=(
            None if srv.url else mcp_server_policy(cfg, root, isolation, srv, readonly=readonly)
        ),
        http=(
            HttpTransport(
                name=name,
                url=srv.url,
                token_env=srv.token_env,
                httpx_trust_env=srv.httpx_trust_env,
            )
            if srv.url
            else None
        ),
    )


def no_jail_cause(cfg: Config, env: Environment) -> str:
    """Why this host resolved to isolation `none`: the env override, the
    config leaf, or what the host lacks (`degrade_reason`)."""
    if sandbox_disabled_by_env():
        return "AGENT6_DANGEROUSLY_DISABLE_SANDBOX=1 is set"
    if cfg.sandbox.isolation == "none":
        return "sandbox.isolation = none"
    return degrade_reason(env) or "this host has no jail"


def start_mcp_manager_if_enabled(
    cfg: Config,
    root: Path,
    isolation: IsolationLevel,
    *,
    reporter: Reporter = STDIO_REPORTER,
    events: EventSink | None = None,
    session_net: SessionNetwork | None = None,
) -> MCPManager | None:
    """Spawn all enabled MCP servers from `cfg.mcp`. Returns None when
    MCP is disabled or no servers are configured (so callers can skip
    teardown entirely). One bad server doesn't poison the run: it is skipped,
    and the run simply does not see its tools.

    A skipped server also becomes an `mcp.server_unavailable` journal event
    when *events* is given. Stderr is only visible from a terminal -- under an
    editor it is a log pane, and the operator sees a run quietly missing the
    tools they configured.
    """
    if not cfg.mcp.enabled or not cfg.mcp.servers:
        return None
    configs = [
        mcp_server_spec(cfg, root, isolation, name, srv)
        for name, srv in cfg.mcp.servers.items()
        if srv.enabled
    ]
    if not configs:
        return None
    _warn_servers_that_keep_the_network(cfg, isolation, reporter=reporter)
    manager = MCPManager.start(configs, logger=reporter.err, session_net=session_net)
    if events is not None:
        for failure in manager.failures:
            events.emit("mcp.server_unavailable", server=failure.name, error=failure.error)
    return manager


def _warn_servers_that_keep_the_network(
    cfg: Config, isolation: IsolationLevel, *, reporter: Reporter
) -> None:
    """`network = "auto"` is the secure default and cannot be honoured without
    a network namespace, so where it degrades it says so -- per server, here,
    where the operator is already being told about their servers. An explicit
    `none` or `session` refused long before this (check_mcp_network_support)."""
    if isolation == "strict":
        return
    for name, srv in sorted(cfg.mcp.servers.items()):
        if srv.enabled and srv.effective_network == "auto":
            reporter.warn(
                f"MCP server {name!r} keeps this host's network:"
                f" taking it away needs the network namespace only 'strict' has, and"
                f" this host resolved to {isolation!r}. On 'hardened', setting its"
                " sandbox.network = 'none' refuses the run instead of connecting it."
            )
