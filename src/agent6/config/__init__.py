# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Config package: the models (`model`), file IO (`io`), and the layered
resolve/view/write (`layer`). The Config models are the package's public API and
are re-exported here, so `from agent6.config import Config` keeps working; the IO
and layering live at `agent6.config.io` / `agent6.config.layer`."""

from __future__ import annotations

from agent6.config._git import GitCommitConfig, GitConfig
from agent6.config._providers import (
    AnthropicProviderEntry,
    ChatGPTProviderEntry,
    ClaudeCodeProviderEntry,
    OpenAIProviderEntry,
    ProviderEntry,
    plan_metered,
    validate_base_url,
)
from agent6.config._sandbox import (
    MCPConfig,
    MCPServerEntry,
    SandboxConfig,
    is_cleartext_url,
    is_loopback_url,
    mcp_server_name_refusal,
)
from agent6.config._surfaces import (
    MachineConfig,
    MachineNotifyConfig,
    NotifyConfig,
    ParallelConfig,
    WebConfig,
    is_loopback_host,
)
from agent6.config._workflow import (
    BudgetConfig,
    ContextConfig,
    MetricConfig,
    PromptConfig,
    ReviewConfig,
    ReviewTier,
    WorkflowConfig,
)
from agent6.config.model import (
    Agent6Section,
    Config,
    ConfigError,
    EffortLevel,
    ModelsConfig,
    RoleModel,
    RoleName,
    load_config,
    validate_config,
)

__all__ = [
    "Agent6Section",
    "AnthropicProviderEntry",
    "BudgetConfig",
    "ChatGPTProviderEntry",
    "ClaudeCodeProviderEntry",
    "Config",
    "ConfigError",
    "ContextConfig",
    "EffortLevel",
    "GitCommitConfig",
    "GitConfig",
    "MCPConfig",
    "MCPServerEntry",
    "MachineConfig",
    "MachineNotifyConfig",
    "MetricConfig",
    "ModelsConfig",
    "NotifyConfig",
    "OpenAIProviderEntry",
    "ParallelConfig",
    "PromptConfig",
    "ProviderEntry",
    "ReviewConfig",
    "ReviewTier",
    "RoleModel",
    "RoleName",
    "SandboxConfig",
    "WebConfig",
    "WorkflowConfig",
    "is_cleartext_url",
    "is_loopback_host",
    "is_loopback_url",
    "load_config",
    "mcp_server_name_refusal",
    "plan_metered",
    "validate_base_url",
    "validate_config",
]
