# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Tests for agent6.config — strict pydantic loading from TOML."""

from __future__ import annotations

from pathlib import Path
from typing import get_args

import pytest
from pydantic import ValidationError

from agent6.config import (
    AnthropicProviderEntry,
    Config,
    ConfigError,
    OpenAIProviderEntry,
    load_config,
)

_VALID_TOML = """
[agent6]
config_version = 1

[providers.anthropic]
api_format = "anthropic"
api_key_env = "ANTHROPIC_API_KEY"
prompt_caching = true

[models.worker]
provider = "anthropic"
model = "claude-x"

[models.reviewer]
provider = "anthropic"
model = "claude-x"

[sandbox]
isolation = "auto"
run_commands = "ask"
protect_git = true

[git]
require_clean_worktree = true
auto_stash = false
branch_per_run = true

[workflow]
verify_command = ["true"]
[budget]
max_tokens_fallback = 100000
"""


def _write(tmp_path: Path, body: str) -> Path:
    p = tmp_path / "agent6.toml"
    p.write_text(body, encoding="utf-8")
    return p


def test_loads_valid_config(tmp_path: Path) -> None:
    cfg = load_config(_write(tmp_path, _VALID_TOML))
    assert cfg.agent6.config_version == 1
    assert cfg.sandbox.isolation == "auto"
    assert cfg.workflow.verify_command == ("true",)


def test_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "nope.toml")


def test_extra_key_forbidden(tmp_path: Path) -> None:
    body = _VALID_TOML.replace("[git]", "[git]\nextra_key = true")
    with pytest.raises(ConfigError, match="extra"):
        load_config(_write(tmp_path, body))


def test_security_field_defaults_to_safe_value(tmp_path: Path) -> None:
    # protect_git is a security field; omitting it must default to the SAFE
    # (enabled) value rather than failing to load (secure-by-default).
    body = _VALID_TOML.replace("protect_git = true\n", "")
    cfg = load_config(_write(tmp_path, body))
    assert cfg.sandbox.protect_git is True


def test_with_sandbox_overrides_disable_forces_none() -> None:
    cfg = Config()
    assert cfg.sandbox.isolation == "auto"
    assert cfg.with_sandbox_overrides(disable_sandbox=True).sandbox.isolation == "none"


def test_with_sandbox_overrides_auto_approve_upgrades_ask_only() -> None:
    ask = Config()
    assert ask.sandbox.run_commands == "ask"
    assert ask.with_sandbox_overrides(auto_approve=True).sandbox.run_commands == "yes"
    # A per-invocation flag must not resurrect a withheld capability.
    withheld = Config.model_validate({"sandbox": {"run_commands": "no"}})
    assert withheld.with_sandbox_overrides(auto_approve=True).sandbox.run_commands == "no"


def test_with_sandbox_overrides_noop_returns_self() -> None:
    cfg = Config()
    assert cfg.with_sandbox_overrides() is cfg


def test_invalid_enum_literal(tmp_path: Path) -> None:
    body = _VALID_TOML.replace('isolation = "auto"', 'isolation = "lax"')
    with pytest.raises(ConfigError, match=r"sandbox\.isolation"):
        load_config(_write(tmp_path, body))


def test_auto_merge_works_without_branch_per_run(tmp_path: Path) -> None:
    """auto_merge lands the hidden chain ref when no visible branch exists, so
    the combination is valid config."""
    body = "[git]\nauto_merge = true\nbranch_per_run = false\n"
    cfg = load_config(_write(tmp_path, body))
    assert cfg.git.auto_merge and not cfg.git.branch_per_run


def test_auto_prune_requires_auto_merge(tmp_path: Path) -> None:
    body = "[git]\nauto_prune = true\nauto_merge = false\n"
    with pytest.raises(ConfigError, match="auto_prune requires"):
        load_config(_write(tmp_path, body))


def test_mcp_server_name_rejects_double_underscore(tmp_path: Path) -> None:
    # `__` separates server from tool in the LLM-visible mcp__<server>__<tool>;
    # a server name containing it would break routing, so it's rejected at load.
    body = _VALID_TOML + ('\n[mcp.servers.bad__name]\ncommand = ["true"]\n')
    with pytest.raises(ConfigError, match="__"):
        load_config(_write(tmp_path, body))


def test_mcp_server_name_is_ascii_only() -> None:
    """The stated contract is ASCII `[A-Za-z0-9_-]+`, but `str.isalnum()` also
    accepts Unicode letters and digits, so a name built from a Cyrillic
    homoglyph, a superscript digit, or a trailing newline slipped through the
    check that guards a TOML table header and the mcp__<server>__ prefix."""
    from agent6.config import mcp_server_name_refusal

    assert mcp_server_name_refusal("good-name_9") == ""
    assert mcp_server_name_refusal("\u0430dmin")  # Cyrillic a + "dmin"
    assert mcp_server_name_refusal("srv\u00b2")  # superscript two
    assert mcp_server_name_refusal("name\n")  # terminal newline a bare $ admits
    assert mcp_server_name_refusal("")  # empty stays refused


def test_extra_read_paths_accepts_clean_absolute(tmp_path: Path) -> None:
    body = _VALID_TOML.replace(
        "protect_git = true",
        'protect_git = true\nextra_read_paths = ["/opt/toolchain", "/usr/local/go"]',
    )
    cfg = load_config(_write(tmp_path, body))
    assert cfg.sandbox.extra_read_paths == ("/opt/toolchain", "/usr/local/go")


def test_extra_read_paths_rejects_relative(tmp_path: Path) -> None:
    body = _VALID_TOML.replace(
        "protect_git = true", 'protect_git = true\nextra_read_paths = ["opt/toolchain"]'
    )
    with pytest.raises(ConfigError, match=r"extra_read_paths"):
        load_config(_write(tmp_path, body))


def test_extra_write_paths_accepts_absolute_rejects_relative_and_traversal(
    tmp_path: Path,
) -> None:
    body = _VALID_TOML.replace(
        "protect_git = true", 'protect_git = true\nextra_write_paths = ["/var/cache/shared"]'
    )
    cfg = load_config(_write(tmp_path, body))
    assert cfg.sandbox.extra_write_paths == ("/var/cache/shared",)
    for bad in ("var/cache", "/var/../etc"):
        body = _VALID_TOML.replace(
            "protect_git = true", f'protect_git = true\nextra_write_paths = ["{bad}"]'
        )
        with pytest.raises(ConfigError, match=r"extra_write_paths"):
            load_config(_write(tmp_path, body))


def test_extra_read_paths_rejects_dotdot_traversal(tmp_path: Path) -> None:
    # FINDING 2: extra_read_paths are bind-mounted read+EXECUTE into the jail, so
    # a `..` component (which could traverse outside the apparent target) must be
    # rejected at config validation even though the path is absolute.
    body = _VALID_TOML.replace(
        "protect_git = true", 'protect_git = true\nextra_read_paths = ["/opt/../etc/shadow"]'
    )
    with pytest.raises(ConfigError, match=r"extra_read_paths.*'\.\.'"):
        load_config(_write(tmp_path, body))


def test_memory_limit_defaults_off(tmp_path: Path) -> None:
    """Not a security control: a memory bomb is a DoS on your own machine and
    the kernel handles it, while a cap costs real builds. Off by default; the
    operator sets one to bound a specific task."""
    cfg = load_config(_write(tmp_path, _VALID_TOML))
    assert cfg.sandbox.memory_limit_mb == 0


def test_memory_limit_accepts_a_cap(tmp_path: Path) -> None:
    body = _VALID_TOML.replace("protect_git = true", "protect_git = true\nmemory_limit_mb = 2048")
    assert load_config(_write(tmp_path, body)).sandbox.memory_limit_mb == 2048


def test_memory_limit_rejects_negative(tmp_path: Path) -> None:
    body = _VALID_TOML.replace("protect_git = true", "protect_git = true\nmemory_limit_mb = -1")
    with pytest.raises(ConfigError, match=r"memory_limit_mb"):
        load_config(_write(tmp_path, body))


def test_openai_base_url_accepts_http_and_https(tmp_path: Path) -> None:
    body = _VALID_TOML.replace(
        "[models.worker]",
        '[providers.local]\napi_format = "openai"\nbase_url = "http://localhost:11434/v1"\n\n[models.worker]',
    )
    cfg = load_config(_write(tmp_path, body))
    assert cfg.providers["local"].base_url == "http://localhost:11434/v1"  # type: ignore[union-attr]


def test_openai_base_url_rejects_schemeless(tmp_path: Path) -> None:
    # The classic paste error: an API key dropped into the base_url field.
    body = _VALID_TOML.replace(
        "[models.worker]",
        '[providers.bad]\napi_format = "openai"\nbase_url = "sk-or-v1-not-a-url"\n'
        "\n[models.worker]",
    )
    with pytest.raises(ConfigError, match=r"base_url"):
        load_config(_write(tmp_path, body))


def test_openai_base_url_rejects_hostless(tmp_path: Path) -> None:
    body = _VALID_TOML.replace(
        "[models.worker]",
        '[providers.bad]\napi_format = "openai"\nbase_url = "https://"\n\n[models.worker]',
    )
    with pytest.raises(ConfigError, match=r"base_url"):
        load_config(_write(tmp_path, body))


def test_role_temperature_defaults_to_zero(tmp_path: Path) -> None:
    # Finding C / Amp 2: agent6's tool-use loop is a feedback loop;
    # default temperature is pinned to 0.0 so OpenRouter-routed models
    # don't run at their (often high) provider default.
    cfg = load_config(_write(tmp_path, _VALID_TOML))
    assert cfg.models.worker is not None
    assert cfg.models.reviewer is not None
    assert cfg.models.worker.temperature == 0.0
    assert cfg.models.reviewer.temperature == 0.0


def test_role_temperature_override(tmp_path: Path) -> None:
    body = _VALID_TOML.replace(
        '[models.worker]\nprovider = "anthropic"\nmodel = "claude-x"',
        '[models.worker]\nprovider = "anthropic"\nmodel = "claude-x"\ntemperature = 0.7',
    )
    cfg = load_config(_write(tmp_path, body))
    assert cfg.models.worker is not None
    assert cfg.models.reviewer is not None
    assert cfg.models.worker.temperature == 0.7
    assert cfg.models.reviewer.temperature == 0.0  # unchanged


def test_role_temperature_nan_rejected(tmp_path: Path) -> None:
    # None (the provider's default) is reachable via the Python API; nan and
    # out-of-range floats fail loud.
    from agent6.config import RoleModel

    assert RoleModel(provider="p", model="m", temperature=None).temperature is None
    body = _VALID_TOML.replace(
        '[models.reviewer]\nprovider = "anthropic"\nmodel = "claude-x"',
        '[models.reviewer]\nprovider = "anthropic"\nmodel = "claude-x"\ntemperature = nan',
    )
    # nan is rejected by ge/le bounds; the canonical "use provider default"
    # path is to omit the field (default 0.0) or explicitly set null via
    # the python API. Document that nan / out-of-range floats fail loud.
    with pytest.raises(ConfigError):
        load_config(_write(tmp_path, body))


def test_role_temperature_out_of_range(tmp_path: Path) -> None:
    body = _VALID_TOML.replace(
        '[models.worker]\nprovider = "anthropic"\nmodel = "claude-x"',
        '[models.worker]\nprovider = "anthropic"\nmodel = "claude-x"\ntemperature = 3.0',
    )
    with pytest.raises(ConfigError):
        load_config(_write(tmp_path, body))


def test_empty_verify_command_loads_and_is_runnable(tmp_path: Path) -> None:
    # verify_command is OPTIONAL: an empty one loads AND is runnable. `agent6
    # run`/`plan` infer one (or fall back to a gateless run), so require_runnable
    # must NOT block on it -- only providers/model are required.
    body = _VALID_TOML.replace('verify_command = ["true"]', "verify_command = []")
    cfg = load_config(_write(tmp_path, body))
    assert cfg.workflow.verify_command == ()
    cfg.require_runnable("worker")  # does not raise


def test_with_verify_command_injects_in_memory(tmp_path: Path) -> None:
    # An inferred verify command is injected in-memory for one run, never
    # mutating the original config.
    body = _VALID_TOML.replace('verify_command = ["true"]', "verify_command = []")
    cfg = load_config(_write(tmp_path, body))
    injected = cfg.with_verify_command(("pytest", "-q"))
    assert injected.workflow.verify_command == ("pytest", "-q")
    assert cfg.workflow.verify_command == ()  # original untouched
    assert cfg.with_verify_command(()).workflow.verify_command == ()


def test_verify_timeout_s_defaults_to_600(tmp_path: Path) -> None:
    """Default verify_timeout_s matches jail default (600s)."""
    cfg = load_config(_write(tmp_path, _VALID_TOML))
    assert cfg.workflow.verify_timeout_s == 600.0


def test_verify_timeout_s_overridable(tmp_path: Path) -> None:
    """Bench configs set verify_timeout_s = 30 for fast failure on
    infinite-loop edits."""
    body = _VALID_TOML.replace(
        'verify_command = ["true"]',
        'verify_command = ["true"]\nverify_timeout_s = 30.0',
    )
    cfg = load_config(_write(tmp_path, body))
    assert cfg.workflow.verify_timeout_s == 30.0


def test_verify_timeout_s_must_be_positive(tmp_path: Path) -> None:
    """0 or negative timeout is rejected (gt=0.0 constraint)."""
    body = _VALID_TOML.replace(
        'verify_command = ["true"]',
        'verify_command = ["true"]\nverify_timeout_s = 0.0',
    )
    with pytest.raises(ConfigError):
        load_config(_write(tmp_path, body))


def test_revise_prompt_defaults_off(tmp_path: Path) -> None:
    cfg = load_config(_write(tmp_path, _VALID_TOML))
    assert cfg.prompt.revise_prompt == "off"


@pytest.mark.parametrize("mode", ["off", "auto", "interactive"])
def test_revise_prompt_modes_load(tmp_path: Path, mode: str) -> None:
    body = _VALID_TOML + f'\n[prompt]\nrevise_prompt = "{mode}"\n'
    cfg = load_config(_write(tmp_path, body))
    assert cfg.prompt.revise_prompt == mode


def test_revise_prompt_invalid_mode_rejected(tmp_path: Path) -> None:
    body = _VALID_TOML + '\n[prompt]\nrevise_prompt = "always"\n'
    with pytest.raises(ConfigError):
        load_config(_write(tmp_path, body))


def test_role_routes_to_unconfigured_provider_rejected(tmp_path: Path) -> None:
    body = _VALID_TOML.replace(
        '[models.reviewer]\nprovider = "anthropic"\nmodel = "claude-x"',
        '[models.reviewer]\nprovider = "openrouter"\nmodel = "gpt-x"',
    )
    with pytest.raises(ConfigError, match="openrouter"):
        load_config(_write(tmp_path, body))


def test_no_providers_loads_but_not_runnable(tmp_path: Path) -> None:
    # Secure-by-default: a config with no providers is valid (a global config
    # may define them); require_runnable refuses to start without one.
    body = _VALID_TOML.replace(
        '[providers.anthropic]\napi_format = "anthropic"\n'
        'api_key_env = "ANTHROPIC_API_KEY"\nprompt_caching = true\n',
        "",
    )
    cfg = load_config(_write(tmp_path, body))
    assert cfg.providers == {}
    with pytest.raises(ConfigError):
        cfg.require_runnable("worker")


def test_openai_provider_with_no_api_key_env_loads(tmp_path: Path) -> None:
    """Ollama-style local endpoint: api_key_env is omitted entirely."""
    body = _VALID_TOML.replace(
        '[providers.anthropic]\napi_format = "anthropic"\n'
        'api_key_env = "ANTHROPIC_API_KEY"\nprompt_caching = true\n',
        '[providers.ollama]\napi_format = "openai"\nbase_url = "http://localhost:11434/v1"\n',
    )
    # Re-route every role to the ollama provider since anthropic is now gone.
    body = body.replace('provider = "anthropic"', 'provider = "ollama"')
    cfg = load_config(_write(tmp_path, body))
    ollama = cfg.providers["ollama"]
    from agent6.config import OpenAIProviderEntry

    assert isinstance(ollama, OpenAIProviderEntry)
    assert ollama.api_key_env is None


def test_chatgpt_provider_defaults_and_refusals(tmp_path: Path) -> None:
    """A bare api_format = "chatgpt" entry fills the Codex backend defaults;
    the formats-only knobs (deployment, key sources, auth_style) are refused
    rather than silently ignored."""
    body = _VALID_TOML.replace(
        '[providers.anthropic]\napi_format = "anthropic"\n'
        'api_key_env = "ANTHROPIC_API_KEY"\nprompt_caching = true\n',
        '[providers.chatgpt]\napi_format = "chatgpt"\n',
    )
    body = body.replace('provider = "anthropic"', 'provider = "chatgpt"')
    cfg = load_config(_write(tmp_path, body))
    from agent6.config import ChatGPTProviderEntry

    entry = cfg.providers["chatgpt"]
    assert isinstance(entry, ChatGPTProviderEntry)
    assert entry.base_url == "https://chatgpt.com/backend-api/codex"
    assert entry.auth_style == "bearer"

    for extra, named in (
        ('deployment = "vertex"\nbase_url = "https://x.example/v1"\n', "direct"),
        ('api_key_env = "OPENAI_API_KEY"\n', "api_key_env"),
        ('token_command = ["mint"]\n', "token_command"),
        ('auth_style = "none"\n', "auth_style"),
        ('oauth_issuer = "https://auth.example.com"\n', "oauth_issuer"),
    ):
        bad = body.replace(
            '[providers.chatgpt]\napi_format = "chatgpt"\n',
            '[providers.chatgpt]\napi_format = "chatgpt"\n' + extra,
        )
        with pytest.raises(ConfigError) as exc:
            load_config(_write(tmp_path, bad))
        assert named in str(exc.value)


def test_multiple_openai_providers_load(tmp_path: Path) -> None:
    """Both OpenAI and OpenRouter side-by-side, distinct keys, routed per role."""
    body = _VALID_TOML.replace(
        '[providers.anthropic]\napi_format = "anthropic"\n'
        'api_key_env = "ANTHROPIC_API_KEY"\nprompt_caching = true\n',
        (
            '[providers.openai]\napi_format = "openai"\n'
            'api_key_env = "OPENAI_API_KEY"\n\n'
            '[providers.openrouter]\napi_format = "openai"\n'
            'api_key_env = "OPENROUTER_API_KEY"\n'
            'base_url = "https://openrouter.ai/api/v1"\n'
        ),
    )
    body = body.replace(
        '[models.worker]\nprovider = "anthropic"\nmodel = "claude-x"',
        '[models.worker]\nprovider = "openai"\nmodel = "gpt-x"',
    )
    body = body.replace('provider = "anthropic"', 'provider = "openrouter"')
    cfg = load_config(_write(tmp_path, body))
    assert set(cfg.providers) == {"openai", "openrouter"}
    assert cfg.models.worker is not None
    assert cfg.models.reviewer is not None
    assert cfg.models.worker.provider == "openai"
    assert cfg.models.reviewer.provider == "openrouter"


def test_metric_block_loads(tmp_path: Path) -> None:
    body = _VALID_TOML + (
        "\n[workflow.metric]\n"
        'command = ["/usr/bin/python3", "bench.py"]\n'
        'pattern = "CYCLES:\\\\s*(\\\\d+)"\n'
        'goal = "minimize"\n'
    )
    cfg = load_config(_write(tmp_path, body))
    assert cfg.workflow.metric is not None
    assert cfg.workflow.metric.command == ("/usr/bin/python3", "bench.py")
    assert cfg.workflow.metric.goal == "minimize"


def test_metric_block_absent_is_none(tmp_path: Path) -> None:
    cfg = load_config(_write(tmp_path, _VALID_TOML))
    assert cfg.workflow.metric is None


def test_metric_goal_invalid(tmp_path: Path) -> None:
    body = _VALID_TOML + (
        '\n[workflow.metric]\ncommand = ["true"]\npattern = "x"\ngoal = "sideways"\n'
    )
    with pytest.raises(ConfigError):
        load_config(_write(tmp_path, body))


def test_operational_fields_have_defaults(tmp_path: Path) -> None:
    """Every field has a default (security fields default to the SAFE value),
    so a minimal TOML loads. Completeness is enforced per command by
    require_runnable, never at load time."""
    body = """
[providers.anthropic]
api_format = "anthropic"
api_key_env = "ANTHROPIC_API_KEY"

[models.worker]
provider = "anthropic"
model = "claude-x"

[models.reviewer]
provider = "anthropic"
model = "claude-x"

[sandbox]
isolation = "auto"
run_commands = "ask"
protect_git = true

[git]

[workflow]
verify_command = ["true"]

[budget]
max_tokens_fallback = 100000
"""
    cfg = load_config(_write(tmp_path, body))
    # Defaulted fields:
    assert cfg.agent6.config_version == 1
    assert cfg.git.require_clean_worktree is True
    assert cfg.git.auto_stash is False
    assert cfg.git.branch_per_run is True
    assert cfg.git.merge_strategy == "squash"
    assert cfg.workflow.verify_timeout_s == 600.0
    anthro = cfg.providers["anthropic"]
    from agent6.config import AnthropicProviderEntry

    assert isinstance(anthro, AnthropicProviderEntry)
    assert anthro.prompt_caching is True
    assert anthro.http_timeout_s == 600.0


def test_compaction_defaults(tmp_path: Path) -> None:
    # Default is now None == adaptive (sized from the worker model's context
    # window at run construction; see models_cache.compaction_thresholds).
    cfg = load_config(_write(tmp_path, _VALID_TOML))
    assert cfg.context.drop_at_chars is None
    assert cfg.context.summarise_at_chars is None
    assert cfg.context.summary_max_tokens == 2048


def test_compaction_both_or_neither(tmp_path: Path) -> None:
    # A lone threshold is ambiguous (is the other adaptive or fixed?); the
    # loader must reject setting only one.
    body = _VALID_TOML + "\n[context]\ndrop_at_chars = 100000\n"
    with pytest.raises(ConfigError) as exc:
        load_config(_write(tmp_path, body))
    assert "BOTH" in str(exc.value) or "NEITHER" in str(exc.value)


def test_compaction_thresholds_overridable(tmp_path: Path) -> None:
    body = (
        _VALID_TOML
        + "\n[context]\n"
        + "drop_at_chars = 100000\n"
        + "summarise_at_chars = 300000\n"
        + "summary_max_tokens = 1024\n"
    )
    cfg = load_config(_write(tmp_path, body))
    assert cfg.context.drop_at_chars == 100000
    assert cfg.context.summarise_at_chars == 300000
    assert cfg.context.summary_max_tokens == 1024


def test_compaction_threshold_must_be_positive(tmp_path: Path) -> None:
    body = _VALID_TOML + "\n[context]\ndrop_at_chars = 0\n"
    with pytest.raises(ConfigError):
        load_config(_write(tmp_path, body))


def test_compaction_summarise_must_exceed_drop(tmp_path: Path) -> None:
    # Inverted ordering (tier-2 <= tier-1) is the misconfiguration that made
    # tier-2 unreachable; the loader must reject it.
    body = _VALID_TOML + "\n[context]\ndrop_at_chars = 300000\nsummarise_at_chars = 200000\n"
    with pytest.raises(ConfigError) as exc:
        load_config(_write(tmp_path, body))
    assert "must be greater than" in str(exc.value)


def test_auto_stash_pop_requires_auto_stash(tmp_path: Path) -> None:
    # The same dependent-knob rule as auto_merge/auto_prune: a pop with nothing
    # ever stashed is inert, so reject it with a pointer instead of loading it.
    body = _VALID_TOML.replace("auto_stash = false", "auto_stash = false\nauto_stash_pop = true")
    with pytest.raises(ConfigError, match="auto_stash_pop"):
        load_config(_write(tmp_path, body))


def test_with_budget_overrides(tmp_path: Path) -> None:
    cfg = load_config(_write(tmp_path, _VALID_TOML))
    out = cfg.with_budget_overrides(max_usd=5.0, max_tokens_fallback=7)
    assert out.budget.max_usd == 5.0
    assert out.budget.max_tokens_fallback == 7
    # Original is unchanged (frozen, returns a copy).
    assert cfg.budget.max_tokens_fallback == 100000


def test_with_budget_overrides_noop_returns_self(tmp_path: Path) -> None:
    cfg = load_config(_write(tmp_path, _VALID_TOML))
    assert cfg.with_budget_overrides() is cfg


def test_budget_max_usd_rejects_non_finite(tmp_path: Path) -> None:
    # TOML nan/inf parse as floats, and a non-finite cap never binds (nan
    # fails every comparison; inf exceeds any spend), silently disabling the
    # hard budget -- refused at the boundary like any other bad value.
    for literal in ("nan", "-nan", "inf", "-inf"):
        body = _VALID_TOML.replace("[budget]", "[budget]\nmax_usd = " + literal)
        with pytest.raises(ConfigError, match="finite"):
            load_config(_write(tmp_path, body))


def test_budget_flag_override_rejects_non_finite(tmp_path: Path) -> None:
    # --max-usd routes through the same validator (with_budget_overrides
    # re-validates), so `--max-usd inf` cannot disable the meter either.
    cfg = load_config(_write(tmp_path, _VALID_TOML))
    with pytest.raises(ValidationError, match="finite"):
        cfg.with_budget_overrides(max_usd=float("inf"))


def test_string_for_bool_rejected(tmp_path: Path) -> None:
    # Strict mode: a quoted "true" is a typo, not a bool; lax coercion
    # laundered it into the safe-looking value.
    body = _VALID_TOML.replace("protect_git = true", 'protect_git = "true"')
    with pytest.raises(ConfigError, match=r"protect_git.*valid boolean"):
        load_config(_write(tmp_path, body))


def test_string_for_int_rejected(tmp_path: Path) -> None:
    body = _VALID_TOML.replace("max_tokens_fallback = 100000", 'max_tokens_fallback = "100000"')
    with pytest.raises(ConfigError, match=r"max_tokens_fallback.*valid integer"):
        load_config(_write(tmp_path, body))


def test_bool_for_number_rejected(tmp_path: Path) -> None:
    body = _VALID_TOML.replace("[budget]", "[budget]\nmax_usd = true")
    with pytest.raises(ConfigError, match=r"max_usd.*valid number"):
        load_config(_write(tmp_path, body))


def test_provider_timeout_rejects_non_finite(tmp_path: Path) -> None:
    # TOML parses inf as a float; an infinite HTTP timeout raised raw
    # OverflowError deep in the transport instead of a config error.
    body = _with_openai_provider(
        '[providers.gw]\napi_format = "openai"\nbase_url = "https://gw.example.com/v1"\n'
        "http_timeout_s = inf"
    )
    with pytest.raises(ConfigError, match="http_timeout_s"):
        load_config(_write(tmp_path, body))


def test_extra_body_rejects_a_toml_date(tmp_path: Path) -> None:
    # TOML parses bare dates/times into objects JSON cannot carry; unrefused,
    # the crash came at request serialization mid-run instead of at load.
    body = _with_openai_provider(
        '[providers.gw]\napi_format = "openai"\nbase_url = "https://gw.example.com/v1"\n'
        "[providers.gw.extra_body]\nsince = 2026-01-01\n"
    )
    with pytest.raises(ConfigError, match=r"extra_body\.since.*date"):
        load_config(_write(tmp_path, body))


def test_argv_rejects_blank_element(tmp_path: Path) -> None:
    for literal in (
        'verify_command = ["uv", " "]',
        '[notify]\non_complete = ["notify-send", ""]',
    ):
        body = _VALID_TOML.replace('verify_command = ["true"]', literal)
        with pytest.raises(ConfigError, match="argv elements"):
            load_config(_write(tmp_path, body))


def test_with_machine_agent_overrides(tmp_path: Path) -> None:
    cfg = load_config(_write(tmp_path, _VALID_TOML))
    out = cfg.with_machine_agent_overrides(
        model="claude-y",
        effort="high",
        temperature=0.5,
        max_usd=2.0,
    )
    assert out.models.worker is not None
    assert out.models.worker.model == "claude-y"
    assert out.models.worker.effort == "high"
    assert out.models.worker.temperature == 0.5
    assert out.budget.max_usd == 2.0
    # Provider name untouched when not overridden.
    assert out.models.worker.provider == "anthropic"


def test_with_machine_agent_overrides_provider(tmp_path: Path) -> None:
    cfg = load_config(_write(tmp_path, _VALID_TOML))
    out = cfg.with_machine_agent_overrides(provider="anthropic", model="claude-z")
    assert out.models.worker is not None
    assert out.models.worker.provider == "anthropic"
    assert out.models.worker.model == "claude-z"


def _with_openai_provider(block: str) -> str:
    return _VALID_TOML.replace("[models.worker]", f"{block}\n\n[models.worker]")


def test_token_command_parses(tmp_path: Path) -> None:
    body = _with_openai_provider(
        '[providers.gw]\napi_format = "openai"\nbase_url = "https://gw.example.com/v1"\n'
        'token_command = ["mint-token", "--json"]\ntoken_command_ttl_s = 60.0'
    )
    cfg = load_config(_write(tmp_path, body))
    entry = cfg.providers["gw"]
    assert entry.token_command == ["mint-token", "--json"]  # type: ignore[union-attr]
    assert entry.token_command_ttl_s == 60.0  # type: ignore[union-attr]


def test_token_command_ttl_defaults_to_300(tmp_path: Path) -> None:
    body = _with_openai_provider(
        '[providers.gw]\napi_format = "openai"\ntoken_command = ["mint-token"]'
    )
    cfg = load_config(_write(tmp_path, body))
    assert cfg.providers["gw"].token_command_ttl_s == 300.0  # type: ignore[union-attr]


def test_token_command_rejects_empty_list(tmp_path: Path) -> None:
    body = _with_openai_provider('[providers.gw]\napi_format = "openai"\ntoken_command = []')
    with pytest.raises(ConfigError, match="token_command"):
        load_config(_write(tmp_path, body))


def test_token_command_rejects_blank_arg(tmp_path: Path) -> None:
    body = _with_openai_provider(
        '[providers.gw]\napi_format = "openai"\ntoken_command = ["mint", "  "]'
    )
    with pytest.raises(ConfigError, match="token_command"):
        load_config(_write(tmp_path, body))


def test_token_command_ttl_must_be_positive(tmp_path: Path) -> None:
    body = _with_openai_provider(
        '[providers.gw]\napi_format = "openai"\ntoken_command = ["mint"]\ntoken_command_ttl_s = 0'
    )
    with pytest.raises(ConfigError, match="token_command_ttl_s"):
        load_config(_write(tmp_path, body))


_VERTEX_CLAUDE = (
    "https://us-east5-aiplatform.googleapis.com/v1/projects/p/locations/us-east5"
    "/publishers/anthropic/models"
)


def test_deployment_and_auth_defaults(tmp_path: Path) -> None:
    cfg = load_config(_write(tmp_path, _VALID_TOML))
    a = cfg.providers["anthropic"]
    assert a.deployment == "direct"
    assert a.auth_style == "x_api_key"  # type: ignore[union-attr]
    assert a.base_url == "https://api.anthropic.com/v1"  # type: ignore[union-attr]


def test_vertex_anthropic_defaults_bearer(tmp_path: Path) -> None:
    body = _with_openai_provider(
        '[providers.v]\napi_format = "anthropic"\ndeployment = "vertex"\n'
        f'base_url = "{_VERTEX_CLAUDE}"'
    )
    cfg = load_config(_write(tmp_path, body))
    assert cfg.providers["v"].deployment == "vertex"
    assert cfg.providers["v"].auth_style == "bearer"  # type: ignore[union-attr]


def test_non_direct_deployment_requires_base_url(tmp_path: Path) -> None:
    body = _with_openai_provider('[providers.v]\napi_format = "anthropic"\ndeployment = "vertex"')
    with pytest.raises(ConfigError, match="base_url is required"):
        load_config(_write(tmp_path, body))


def test_azure_requires_openai_format(tmp_path: Path) -> None:
    body = _with_openai_provider(
        '[providers.a]\napi_format = "anthropic"\ndeployment = "azure"\n'
        'base_url = "https://r.openai.azure.com"\nextra_query = { "api-version" = "2024-06-01" }'
    )
    with pytest.raises(ConfigError, match="api_format 'openai'"):
        load_config(_write(tmp_path, body))


def test_azure_requires_api_version_query(tmp_path: Path) -> None:
    body = _with_openai_provider(
        '[providers.a]\napi_format = "openai"\ndeployment = "azure"\nbase_url = "https://r.openai.azure.com"'
    )
    with pytest.raises(ConfigError, match="extra_query"):
        load_config(_write(tmp_path, body))


def test_azure_defaults_api_key_header(tmp_path: Path) -> None:
    body = _with_openai_provider(
        '[providers.a]\napi_format = "openai"\ndeployment = "azure"\n'
        'base_url = "https://r.openai.azure.com"\nextra_query = { "api-version" = "2024-06-01" }'
    )
    cfg = load_config(_write(tmp_path, body))
    assert cfg.providers["a"].auth_style == "api_key_header"  # type: ignore[union-attr]


def test_unknown_deployment_rejected(tmp_path: Path) -> None:
    body = _with_openai_provider(
        '[providers.x]\napi_format = "openai"\ndeployment = "bedrock"\nbase_url = "https://x.example.com"'
    )
    with pytest.raises(ConfigError, match="deployment"):
        load_config(_write(tmp_path, body))


def test_explicit_auth_style_preserved(tmp_path: Path) -> None:
    body = _with_openai_provider('[providers.x]\napi_format = "openai"\nauth_style = "none"')
    cfg = load_config(_write(tmp_path, body))
    assert cfg.providers["x"].auth_style == "none"  # type: ignore[union-attr]


@pytest.mark.parametrize(
    "cred_line",
    ['api_key_env = "OPENAI_API_KEY"', 'token_command = ["mint-token"]'],
)
def test_none_auth_with_a_credential_source_is_refused(tmp_path: Path, cred_line: str) -> None:
    """auth_style = 'none' sends no auth header, so also naming api_key_env or
    token_command is a contradiction: the credential reads as configured yet is
    never sent. Refuse rather than silently ignore it."""
    body = _with_openai_provider(
        f'[providers.x]\napi_format = "openai"\nauth_style = "none"\n{cred_line}'
    )
    with pytest.raises(ConfigError, match="auth_style"):
        load_config(_write(tmp_path, body))


def test_skills_defaults() -> None:
    cfg = Config()
    assert cfg.skills.enabled is True
    assert cfg.skills.extra_dirs == ()
    assert cfg.skills.state == {}


def test_skills_state_map_loads(tmp_path: Path) -> None:
    body = _VALID_TOML + '\n[skills.state]\ncaveman = "always"\ntidy = "disabled"\n'
    cfg = load_config(_write(tmp_path, body))
    assert cfg.skills.state == {"caveman": "always", "tidy": "disabled"}


def test_skills_state_rejects_unknown_value(tmp_path: Path) -> None:
    # one value per skill; only the three states exist (a skill can never be
    # both disabled and always by construction)
    body = _VALID_TOML + '\n[skills.state]\ncaveman = "sometimes"\n'
    with pytest.raises(ConfigError, match="skills"):
        load_config(_write(tmp_path, body))


def test_skills_rejects_unknown_key(tmp_path: Path) -> None:
    body = _VALID_TOML + "\n[skills]\nallow_repo_skills = true\n"
    with pytest.raises(ConfigError, match="skills"):
        load_config(_write(tmp_path, body))


def test_extra_paths_never_target_the_private_dirs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An extra grant AT or INSIDE an agent6-private dir (secrets, state)
    never enters the jail and is refused at config load; a grant merely
    CONTAINING one stays valid (strict masks it out)."""
    cfg_home = tmp_path / "home" / ".config" / "agent6"
    cfg_home.mkdir(parents=True)
    monkeypatch.setenv("AGENT6_CONFIG_HOME", str(cfg_home))
    body = f'[sandbox]\nextra_read_paths = ["{cfg_home}"]\n'
    with pytest.raises(ConfigError, match="agent6-private"):
        load_config(_write(tmp_path, body))
    body = f'[sandbox]\nextra_write_paths = ["{cfg_home / "sub"}"]\n'
    with pytest.raises(ConfigError, match="agent6-private"):
        load_config(_write(tmp_path, body))
    body = f'[sandbox]\nextra_read_paths = ["{tmp_path / "home"}"]\n'
    assert load_config(_write(tmp_path, body)).sandbox.extra_read_paths


def test_hide_paths_validate_like_the_other_path_lists(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="absolute"):
        load_config(_write(tmp_path, '[sandbox]\nhide_paths = ["relative/x"]\n'))
    with pytest.raises(ConfigError, match=r"\.\."):
        load_config(_write(tmp_path, '[sandbox]\nhide_paths = ["/a/../b"]\n'))


def test_the_skills_dir_can_be_granted_to_the_jail(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Installed skills are operator content the model is meant to use, so a
    skill's bundled script must be runnable in the jail: the data dir (and the
    regenerable cache) are grantable, unlike config and state."""
    monkeypatch.setenv("AGENT6_DATA_HOME", str(tmp_path / "data"))
    monkeypatch.setenv("AGENT6_CACHE_HOME", str(tmp_path / "cache"))
    monkeypatch.setenv("AGENT6_CONFIG_HOME", str(tmp_path / "cfg"))
    monkeypatch.setenv("AGENT6_STATE_HOME", str(tmp_path / "state"))
    skills = tmp_path / "data" / "skills"
    body = f'[sandbox]\nextra_read_paths = ["{skills}", "{tmp_path / "cache"}"]\n'
    assert len(load_config(_write(tmp_path, body)).sandbox.extra_read_paths) == 2
    # config and state stay refused.
    body = f'[sandbox]\nextra_read_paths = ["{tmp_path / "state" / "repo"}"]\n'
    with pytest.raises(ConfigError, match="agent6-private"):
        load_config(_write(tmp_path, body))


def test_api_format_discriminates_the_provider_entry(tmp_path: Path) -> None:
    """`api_format` routes a `[providers.*]` block to its entry class. It is
    declared on the shared base so the field leads every entry's order, and
    each subclass's annotation must stay the single-value literal: the union
    discriminates on it and `config/write.py` reflects over it."""
    cfg = load_config(_write(tmp_path, _VALID_TOML))
    assert isinstance(cfg.providers["anthropic"], AnthropicProviderEntry)

    body = _VALID_TOML.replace('api_format = "anthropic"', 'api_format = "openai"').replace(
        "prompt_caching = true\n", ""
    )
    cfg = load_config(_write(tmp_path, body))
    assert isinstance(cfg.providers["anthropic"], OpenAIProviderEntry)

    assert get_args(AnthropicProviderEntry.model_fields["api_format"].annotation) == ("anthropic",)
    assert get_args(OpenAIProviderEntry.model_fields["api_format"].annotation) == ("openai",)
    assert next(iter(AnthropicProviderEntry.model_fields)) == "api_format"
    assert next(iter(OpenAIProviderEntry.model_fields)) == "api_format"


def test_a_provider_block_without_api_format_names_the_key(tmp_path: Path) -> None:
    """pydantic's own text ("Unable to extract tag using discriminator") names
    neither the key nor its values, and hand-writing a provider block is a
    documented way in."""
    cfg = tmp_path / "config.toml"
    cfg.write_text('[providers.anthropic]\napi_key_env = "K"\n', encoding="utf-8")
    with pytest.raises(ConfigError) as exc:
        load_config(cfg)
    assert 'set api_format = "anthropic", "openai", or "chatgpt"' in str(exc.value)


def test_chatgpt_oauth_endpoints_are_constants_not_config() -> None:
    """The issuer and client id are pinned to OpenAI's (a bearer authority is
    never a config knob); a config that tries to set them is refused as an
    unknown key, and the constants carry the pinned values."""
    from agent6.config import ChatGPTProviderEntry
    from agent6.providers.chatgpt_oauth import CHATGPT_CLIENT_ID, CHATGPT_ISSUER

    assert CHATGPT_ISSUER == "https://auth.openai.com"
    assert CHATGPT_CLIENT_ID
    with pytest.raises(ValueError, match="oauth_issuer"):
        ChatGPTProviderEntry.model_validate(
            {"api_format": "chatgpt", "oauth_issuer": "https://auth.example.com"}
        )


def test_extra_device_paths_must_live_under_dev() -> None:
    """A device grant is /dev-only: anywhere else is a file grant wearing a
    device hat (extra_read/write_paths own those), and traversal is refused."""
    from agent6.config import SandboxConfig

    ok = SandboxConfig(extra_device_paths=("/dev/nvidia0", "/dev/nvidiactl"))
    assert ok.extra_device_paths == ("/dev/nvidia0", "/dev/nvidiactl")
    assert SandboxConfig().extra_device_paths == ()
    for bad in ("/etc/passwd", "dev/null", "/dev/../etc"):
        with pytest.raises(ValueError, match="must live under /dev"):
            SandboxConfig(extra_device_paths=(bad,))


def test_model_git_control_requires_git_writes(tmp_path: Path) -> None:
    """git.control = "model" hands git to the model; protect_git = true
    contradicts it and refuses naming both keys."""
    body = _VALID_TOML.replace("[git]\n", '[git]\ncontrol = "model"\n')
    with pytest.raises(ConfigError, match="protect_git"):
        load_config(_write(tmp_path, body))
    ok = load_config(_write(tmp_path, body.replace("protect_git = true", "protect_git = false")))
    assert ok.git.control == "model"


def test_max_iterations_defaults_to_200(tmp_path: Path) -> None:
    cfg = load_config(_write(tmp_path, _VALID_TOML))
    assert cfg.workflow.max_iterations == 200


def test_max_iterations_unlimited_is_minus_one(tmp_path: Path) -> None:
    body = _VALID_TOML.replace(
        'verify_command = ["true"]',
        'verify_command = ["true"]\nmax_iterations = -1',
    )
    assert load_config(_write(tmp_path, body)).workflow.max_iterations == -1


def test_max_iterations_zero_is_rejected(tmp_path: Path) -> None:
    """0 would end every leg before its first call; the sentinel is exactly -1."""
    body = _VALID_TOML.replace(
        'verify_command = ["true"]',
        'verify_command = ["true"]\nmax_iterations = 0',
    )
    with pytest.raises(ConfigError, match="exactly -1 for unlimited"):
        load_config(_write(tmp_path, body))


def test_cleartext_rejection_is_scheme_case_insensitive(tmp_path: Path) -> None:
    """URL schemes are case-insensitive on the wire: `HTTP://` dials cleartext
    exactly like `http://`, so a prefix match would let a mixed-case scheme
    evade the https requirement on the chatgpt endpoints."""
    for field, extra in (("base_url", 'base_url = "HTTP://api.example.com/codex"'),):
        body = _VALID_TOML + f'\n[providers.gpt]\napi_format = "chatgpt"\n{extra}\n'
        d = tmp_path / field
        d.mkdir()
        with pytest.raises(ConfigError, match="https"):
            load_config(_write(d, body))
