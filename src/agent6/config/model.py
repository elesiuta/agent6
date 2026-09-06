# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Config loading, TOML to pydantic.

This is a trust boundary (untrusted text -> structured types), so we use
pydantic and surface field-pointing errors.

Field policy: **secure by default, fully auditable**. Every field has a
default, and security-sensitive fields default to the *safe* value
(`sandbox.network = "auto"`,
`sandbox.run_commands = "ask"`, `sandbox.protect_git = true`; git push,
`--force`, and history rewrites are refused unconditionally by `git_ops`,
with no config override at all). Configs layer: global `$XDG_CONFIG_HOME`
defaults, then the per-repo config (out of the workspace, under the state
dir), so a repo is zero-config when the global config supplies providers +
models. Use
`agent6 config show` to audit the *effective* value of every field and
exactly where it came from (default / global / repo / flag). The one thing a
run genuinely cannot guess, a provider+key, is checked by
:meth:`Config.require_runnable` with a pointer to `agent6 connect` rather
than a load-time failure, so `config show` always works. The repo's
`verify_command` is optional: `agent6 run`/`plan` infer one per run when it
is unset (see :mod:`agent6.verify_infer`), else run gateless.
"""

from __future__ import annotations

import tomllib
from collections.abc import Callable
from pathlib import Path
from typing import Literal

from pydantic import (
    BaseModel,
    Field,
    ValidationError,
    model_validator,
)

from agent6.config._base import MODEL_CONFIG
from agent6.config._git import GitConfig
from agent6.config._providers import ClaudeCodeProviderEntry, ProviderEntry
from agent6.config._sandbox import MCPConfig, SandboxConfig, is_cleartext_url, is_loopback_url
from agent6.config._surfaces import (
    MachineConfig,
    NotifyConfig,
    ParallelConfig,
    SkillsConfig,
    WebConfig,
)
from agent6.config._workflow import (
    BudgetConfig,
    ContextConfig,
    PromptConfig,
    ReviewConfig,
    WorkflowConfig,
)
from agent6.errors import OperatorError
from agent6.types import RoleName


class ConfigError(OperatorError):
    """Raised when the config file is missing, malformed, or fails validation.

    An OperatorError: the config is the operator's file, so `cli_main`
    presents it as a refusal, never a crash report.
    """


EffortLevel = Literal["off", "low", "medium", "high", "xhigh", "max"]


class RoleModel(BaseModel):
    """One role's `(provider, model)` assignment.

    `provider` is the name (TOML table key) of an entry in `[providers.*]`.

    `temperature` is the sampling temperature agent6 will pin on every
    call for this role. Defaults to `0.0`, agent6's tool-use loop is a
    search-and-act feedback loop and high-temperature sampling causes
    observable degeneration on some open-weights models (caught
    Kimi K2.6 emitting 15997 literal `\\n` escapes in a single
    `old_string` argument before hitting the completion-tokens cap).
    Anthropic and OpenAI models are tuned to behave well at any
    temperature; OpenRouter routes to provider defaults that vary by
    model, so pinning is the only way to make benches reproducible.
    Set to `null` only if you specifically want the provider's default
    behaviour. TOML has no null literal and `temperature = nan` fails the
    0.0-2.0 bounds, so null is reachable only via the Python API; omitting the
    key leaves the `0.0` default, not the provider's default.
    """

    model_config = MODEL_CONFIG

    provider: str = Field(
        min_length=1,
        description="A `[providers.<name>]` entry, by name.",
    )
    model: str = Field(
        min_length=1,
        description="Model id as that provider names it (`agent6 model` lists them).",
    )
    temperature: float | None = Field(
        default=0.0,
        ge=0.0,
        le=2.0,
        description=(
            "Sampling temperature pinned on every call, `0.0` to `2.0`. `0.0` keeps tool use "
            "stable; unset leaves the provider's default."
        ),
    )
    # Reasoning effort for this role. `None` leaves the provider default;
    # `off` disables it explicitly. Mapped per wire: OpenAI-family models
    # receive `reasoning.effort`, Anthropic adaptive models
    # `output_config.effort` (older ones a thinking budget, `xhigh`/`max`
    # collapsing to the top tier). Non-reasoning models ignore it.
    effort: EffortLevel | None = Field(
        default=None,
        description=(
            "Reasoning effort: `off`, `low`, `medium`, `high`, `xhigh`, or `max` (the top tiers "
            "where the model offers them; Anthropic collapses them to its highest). Unset: what "
            "the wire applies, which `agent6 config show` prints resolved (`low` on "
            "openai-compatible reasoning models, no thinking on Anthropic)."
        ),
    )


class ModelsConfig(BaseModel):
    """Per-role provider + model routing.

    Three roles, all optional:

    - `worker` drives the single-loop agent (`agent6 run` / ``agent6
      resume``); its pricing also drives the USD -> token budget
      conversion.
    - `planner` drives `agent6 plan` (the planning pass).
      Unset -> falls back to `worker` (set it to a frontier model + high
      thinking for careful up-front planning).
    - `reviewer` drives the one-shot `agent6 review` subcommand and the
      in-loop review panel. Unset -> falls back to `worker`.

    Any configured provider may serve any role. Leaving every role unset is
    valid (e.g. a global config that only declares providers); a role is
    only *required* for the command that uses it, checked by
    :meth:`Config.require_runnable`.
    """

    model_config = MODEL_CONFIG

    worker: RoleModel | None = Field(
        default=None,
        description=(
            "The `(provider, model)` driving `agent6 run`/`resume`; its pricing also converts "
            "the USD budget to tokens."
        ),
    )
    reviewer: RoleModel | None = Field(
        default=None,
        description=(
            "Drives `agent6 review`, the in-loop review panel, the context summariser and"
            " gister, and the prompt reviser. Unset falls back to `worker`."
        ),
    )
    planner: RoleModel | None = Field(
        default=None,
        description="Drives `agent6 plan` (the planning pass). Unset falls back to `worker`.",
    )

    def configured(self) -> dict[str, RoleModel]:
        """Only the roles explicitly set (used for validation/key checks)."""
        out: dict[str, RoleModel] = {}
        if self.worker is not None:
            out["worker"] = self.worker
        if self.reviewer is not None:
            out["reviewer"] = self.reviewer
        if self.planner is not None:
            out["planner"] = self.planner
        return out

    def resolve(self, role: RoleName) -> RoleModel | None:
        """The effective model for *role*, applying worker fallbacks."""
        if role == "worker":
            return self.worker
        if role == "planner":
            return self.planner or self.worker
        if role == "reviewer":
            return self.reviewer or self.worker
        return None

    def source_role(self, role: RoleName) -> RoleName:
        """The configured entry `resolve(role)` reads: *role* itself when
        explicitly set, else the worker fallback. Lets an error message name
        the config key the user actually wrote (mirrors `resolve` above)."""
        return role if role in self.configured() else "worker"


class Agent6Section(BaseModel):
    model_config = MODEL_CONFIG

    config_version: int = Field(
        ge=1,
        le=1,
        default=1,
        description="Config schema version; only `1` is accepted.",
    )


class Config(BaseModel):
    """The validated effective config: one immutable object per load.

    Frozen at the attribute level; container VALUES (dicts, the tuples'
    contents) are not deep-frozen. The contract is read-only after
    validation: every derived config goes through the `with_*` copiers,
    never in-place mutation."""

    model_config = MODEL_CONFIG

    agent6: Agent6Section = Field(default_factory=Agent6Section)
    providers: dict[str, ProviderEntry] = Field(
        default_factory=dict,
        description=(
            "Provider endpoints by name (`[providers.<name>]`); a `[models.*]` role names one. "
            "`agent6 connect` writes them."
        ),
    )
    models: ModelsConfig = Field(default_factory=ModelsConfig)
    sandbox: SandboxConfig = Field(default_factory=SandboxConfig)
    git: GitConfig = Field(default_factory=GitConfig)
    workflow: WorkflowConfig = Field(default_factory=WorkflowConfig)
    review: ReviewConfig = Field(default_factory=ReviewConfig)
    context: ContextConfig = Field(default_factory=ContextConfig)
    prompt: PromptConfig = Field(default_factory=PromptConfig)
    skills: SkillsConfig = Field(default_factory=SkillsConfig)
    budget: BudgetConfig = Field(default_factory=BudgetConfig)
    machine: MachineConfig = Field(default_factory=MachineConfig)
    notify: NotifyConfig = Field(default_factory=NotifyConfig)
    mcp: MCPConfig = Field(default_factory=MCPConfig)
    web: WebConfig = Field(default_factory=WebConfig)
    parallel: ParallelConfig = Field(default_factory=ParallelConfig)
    # Named strategy PRESET: fills in many settings at once (BUILTIN_PRESETS +
    # user `[presets.<name>]`). "" / "standard" = plain defaults; injection
    # order and stacking rules: `config.layer._apply_preset`.
    preset: str = Field(
        default="",
        description=(
            "The strategy preset in force: `standard` (plain defaults), `quick` (no review panel), "
            "`ultra` (a three-seat panel that advises and vetoes before finish), `paranoid` (five "
            "explore-tier seats), or a `[presets.<name>]` of your own. Fills many settings at once "
            "and overrides every section of the layer that selects it; `--preset` overrides per "
            "run, `resume --preset` per resumed leg. Empty: no preset."
        ),
    )

    @model_validator(mode="after")
    def _cross_validate_provider_routing(self) -> Config:
        # Only configured roles are checked here, and only when their
        # provider is actually present; an empty/partial config is valid
        # at load time (require_runnable enforces completeness per command).
        for role, rm in self.models.configured().items():
            if self.providers and rm.provider not in self.providers:
                known = ", ".join(sorted(self.providers)) or "(none)"
                raise ValueError(
                    f"models.{role}.provider = {rm.provider!r} but"
                    f" [providers.{rm.provider}] is not configured."
                    f" Known providers: {known}."
                )
        return self

    @model_validator(mode="after")
    def _model_git_control_needs_git_writes(self) -> Config:
        """`git.control = "model"` hands git to the model, which cannot manage
        what it cannot write: `sandbox.protect_git = true` contradicts it."""
        if self.git.control == "model" and self.sandbox.protect_git:
            raise ValueError(
                'git.control = "model" needs the model to write .git;'
                ' set sandbox.protect_git = false (or keep control = "agent6").'
            )
        return self

    @model_validator(mode="after")
    def _pass_env_excludes_provider_keys(self) -> Config:
        """No `pass_env` (an MCP server's, the machine allowlist) may name a
        provider's `api_key_env`.

        The invariant lives here so a direct config edit cannot bypass it (mcp
        connect pre-checks the same rule for a friendlier early refusal).
        """
        keys = {
            e.api_key_env
            for e in self.providers.values()
            if not isinstance(e, ClaudeCodeProviderEntry) and e.api_key_env
        }
        lists = [
            (f"[mcp.servers.{name}].pass_env", srv.pass_env, "an MCP server")
            for name, srv in self.mcp.servers.items()
        ]
        lists.append(("[machine].pass_env", self.machine.pass_env, "a machine's tool"))
        for where, names, who in lists:
            leaked = sorted(keys.intersection(names))
            if leaked:
                raise ValueError(
                    f"{where} names provider API key env var(s) {', '.join(leaked)};"
                    f" agent6 never passes a provider key to {who}."
                )
        return self

    def with_budget_overrides(
        self,
        *,
        max_usd: float | None = None,
        max_tokens_fallback: int | None = None,
        max_percent: float | None = None,
    ) -> Config:
        """Return a copy with budget fields overridden (the per-run CLI flags,
        each writing the config field of the same name). `None` keeps the
        existing value."""
        if max_usd is None and max_tokens_fallback is None and max_percent is None:
            return self
        data = self.model_dump(mode="python")
        budget = data.setdefault("budget", {})
        if max_usd is not None:
            budget["max_usd"] = max_usd
        if max_tokens_fallback is not None:
            budget["max_tokens_fallback"] = max_tokens_fallback
        if max_percent is not None:
            budget["max_percent"] = max_percent
        return Config.model_validate(data)

    def with_sandbox_overrides(
        self,
        *,
        disable_sandbox: bool = False,
        auto_approve: bool = False,
        no_commands: bool = False,
    ) -> Config:
        """Return a copy with per-invocation sandbox overrides from CLI flags.

        `disable_sandbox` forces `sandbox.isolation = "none"` (unconfined).
        `auto_approve` upgrades `run_commands` `"ask" -> "yes"` but never
        resurrects a withheld `"no"` (a per-invocation flag must not grant a
        capability the standing policy denied); it covers every MCP server's
        `approve` too, because "do not prompt me this run" that still prompted
        would not be that. `no_commands` pins `run_commands` to `"no"` and
        always may: tightening needs no permission. All are operator-supplied
        (flag/env); the LLM can reach none of them.
        """
        if not disable_sandbox and not auto_approve and not no_commands:
            return self
        data = self.model_dump(mode="python")
        sandbox = data.setdefault("sandbox", {})
        if disable_sandbox:
            sandbox["isolation"] = "none"
        if auto_approve and self.sandbox.run_commands != "no":
            sandbox["run_commands"] = "yes"
        if auto_approve:
            for server in data.get("mcp", {}).get("servers", {}).values():
                server["approve"] = "yes"
        if no_commands:
            sandbox["run_commands"] = "no"
        return Config.model_validate(data)

    def with_machine_agent_overrides(
        self,
        *,
        provider: str | None = None,
        model: str | None = None,
        effort: str | None = None,
        temperature: float | None = None,
        max_usd: float | None = None,
        max_tokens_fallback: int | None = None,
    ) -> Config:
        """Return a copy with a machine `agent` state's per-state knobs applied.

        Overrides the `worker` role (the role machine agent loops run as)
        and the budget ledgers. `None` means "inherit the effective config".
        Re-validates so the provider-name checks run against the merged result."""
        data = self.model_dump(mode="python")
        worker = data.setdefault("models", {}).get("worker")
        if worker is None:
            worker = {}
            data["models"]["worker"] = worker
        if provider is not None:
            worker["provider"] = provider
        if model is not None:
            worker["model"] = model
        if effort is not None:
            worker["effort"] = effort
        if temperature is not None:
            worker["temperature"] = temperature
        budget = data.setdefault("budget", {})
        if max_usd is not None:
            budget["max_usd"] = max_usd
        if max_tokens_fallback is not None:
            budget["max_tokens_fallback"] = max_tokens_fallback
        return Config.model_validate(data)

    def with_verify_command(self, argv: tuple[str, ...]) -> Config:
        """Return a copy whose `workflow.verify_command` is *argv*, `()` for
        a gateless run.

        How `agent6 run`/`plan` inject a verify command inferred at run start,
        and how a run whose policy withholds command tools drops the gate it
        could never execute. IN-MEMORY only -- runs never write config; the
        operator is shown what was picked and can pin it explicitly.
        """
        data = self.model_dump(mode="python")
        data.setdefault("workflow", {})["verify_command"] = list(argv)
        return Config.model_validate(data)

    def cleartext_credential_endpoints(self) -> tuple[str, ...]:
        """Configured endpoints that send a credential over plaintext http to a
        non-loopback host, as `[table] url` labels for the run-entry warning
        and the `mcp connect` confirmation. Loopback http is the normal
        local-server case and never listed; neither is https."""
        out: list[str] = []
        for name, entry in sorted(self.providers.items()):
            if (
                not isinstance(entry, ClaudeCodeProviderEntry)
                and is_cleartext_url(entry.base_url)
                and entry.auth_style != "none"
                and not is_loopback_url(entry.base_url)
            ):
                out.append(f"[providers.{name}] {entry.base_url}")
        for name, srv in sorted(self.mcp.servers.items()):
            if srv.token_env and is_cleartext_url(srv.url) and not is_loopback_url(srv.url):
                out.append(f"[mcp.servers.{name}] {srv.url}")
        return tuple(out)

    def with_run_commands_clamped(self) -> Config:
        """Return a copy with `sandbox.run_commands` clamped for an interactive
        mode (`agent6 ask` / `agent6 plan`).

        Ask and plan run with the operator sitting there, often in a directory
        that is not even a repo, so they must never execute anything unwatched:
        `"yes"` becomes `"ask"`. Only ever tightens -- `"no"` stays refused,
        because a run can never loosen a boundary the operator set. IN-MEMORY
        only, like `with_verify_command`: `config show` keeps reporting what the
        operator actually configured.
        """
        if self.sandbox.run_commands != "yes":
            return self
        data = self.model_dump(mode="python")
        data.setdefault("sandbox", {})["run_commands"] = "ask"
        return Config.model_validate(data)

    def with_decompose(self, value: Literal["on", "off"]) -> Config:
        """Return a copy with `prompt.decompose` pinned to *value*.

        Used by the CLI to resolve `"auto"` (from the model-capability
        registry) before the workflow starts, so the engine only ever sees
        on/off. IN-MEMORY only, like `with_verify_command`.
        """
        data = self.model_dump(mode="python")
        data.setdefault("prompt", {})["decompose"] = value
        return Config.model_validate(data)

    def require_runnable(self, role: RoleName = "worker") -> None:
        """Raise ConfigError unless *role* can actually run.

        Checks (in order) that a provider is configured and the role resolves
        to a model whose provider exists. Messages point at the command that
        fixes the gap so a fresh user is never stuck. `verify_command` is NOT
        required: `agent6 run`/`plan` infer one when unset (and fall back to a
        gateless run if even that fails) -- see `agent6.verify_infer`.
        """
        if not self.providers:
            raise ConfigError(
                "No providers configured. Run `agent6 connect` to add one"
                " (stored in your global config), or add a [providers.*]"
                " block to the per-repo config."
            )
        rm = self.models.resolve(role)
        if rm is None:
            raise ConfigError(
                f"No model configured for the {role!r} role. Run `agent6 model`"
                " to set it, or add a [models.worker] block to your config."
            )
        if rm.provider not in self.providers:
            known = ", ".join(sorted(self.providers)) or "(none)"
            raise ConfigError(
                f"models.{role}.provider = {rm.provider!r} but [providers.{rm.provider}]"
                f" is not configured. Known providers: {known}."
            )


# pydantic reports a provider block with no `api_format` as
# "Unable to extract tag using discriminator", which names neither the key to
# add nor its two values. A hand-written block is a documented way in.
_MISSING_API_FORMAT = (
    'set api_format = "anthropic", "openai", "chatgpt", or "claude_code" (see docs/config.md)'
)


def _format_validation_error(
    err: ValidationError,
    source: str,
    locate: Callable[[str, str], str | None] | None = None,
) -> str:
    lines = [f"Config validation failed: {source}"]
    for issue in err.errors():
        loc = ".".join(str(part) for part in issue["loc"]) or "<root>"
        msg = issue["msg"]
        if issue["type"] == "union_tag_not_found" and loc.startswith("providers."):
            msg = _MISSING_API_FORMAT
        lines.append(f"  - {loc}: {msg} (type={issue['type']})")
        if locate is not None and (where := locate(loc, issue["type"])):
            lines.append(where)
    return "\n".join(lines)


def validate_config(
    raw: dict[str, object],
    *,
    source: str = "<config>",
    locate: Callable[[str, str], str | None] | None = None,
) -> Config:
    """Validate an already-parsed (and possibly layer-merged) config dict.

    Shared by :func:`load_config` and the layered loader
    (`agent6.config.layer`) so both surface identical field-pointing errors.
    `locate` maps (dotted leaf, pydantic error type) to a "which file, how to
    fix" hint appended to its error line, so a stale value in a layered config
    names its own source and the remedy that works for that kind of error.
    """
    try:
        return Config.model_validate(raw)
    except ValidationError as exc:
        raise ConfigError(_format_validation_error(exc, source, locate)) from exc


def load_config(path: Path) -> Config:
    """Load and strictly validate the TOML config at *path*.

    Raises ConfigError on any problem; never returns a partially valid config.
    """
    if not path.is_file():
        raise ConfigError(f"Config file not found: {path}")
    try:
        raw = tomllib.loads(path.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"Config file is not valid TOML ({path}): {exc}") from exc
    except (OSError, UnicodeDecodeError) as exc:
        raise ConfigError(f"Config file cannot be read ({path}): {exc}") from exc
    return validate_config(raw, source=str(path))
