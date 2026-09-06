# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""agent6's own git ops run on the host, outside the jail, and inherit the
environment. A provider API key sitting in the environment (the operator set
`ANTHROPIC_API_KEY` in their shell) has no business reaching git -- a
credential helper or a content driver we could not neutralize would inherit
it. The configured provider-key env vars are stripped from git's environment;
everything git actually needs (PATH, HOME, ...) stays.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from agent6 import child_env, git_ops
from agent6.app._setup import apply_git_ops_policy
from agent6.config import Config


@pytest.fixture(autouse=True)
def _reset_policy() -> object:  # pyright: ignore[reportUnusedFunction]
    yield
    child_env.set_provider_key_env([])  # module-level state; do not leak across tests


def _captured_git_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> dict[str, str]:
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    seen: dict[str, dict[str, str]] = {}
    real = subprocess.Popen

    def spy(*args: object, **kwargs: object):
        env = kwargs.get("env")
        if isinstance(env, dict):
            seen["env"] = env
        return real(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(git_ops.subprocess, "Popen", spy)
    git_ops.status(tmp_path)
    return seen["env"]


def test_a_configured_provider_key_is_stripped_from_gits_environment(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-secret-should-not-reach-git")
    apply_git_ops_policy(
        Config.model_validate(
            {
                "providers": {
                    "anthropic": {"api_format": "anthropic", "api_key_env": "ANTHROPIC_API_KEY"}
                }
            }
        )
    )
    env = _captured_git_env(monkeypatch, tmp_path)
    assert "ANTHROPIC_API_KEY" not in env, "a provider key reached git's environment"
    assert "PATH" in env, "git lost an environment variable it needs"


def test_a_variable_that_is_not_a_provider_key_is_left_alone(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("GIT_SSH_COMMAND", "ssh -i /home/me/.ssh/id_ed25519")
    monkeypatch.setenv("SOME_OTHER_TOKEN", "not-a-configured-key")
    apply_git_ops_policy(
        Config.model_validate(
            {
                "providers": {
                    "anthropic": {"api_format": "anthropic", "api_key_env": "ANTHROPIC_API_KEY"}
                }
            }
        )
    )
    env = _captured_git_env(monkeypatch, tmp_path)
    # Only the CONFIGURED key name is stripped -- not every token-shaped var,
    # which would risk breaking a git setup that reads its own env.
    assert env.get("GIT_SSH_COMMAND") == "ssh -i /home/me/.ssh/id_ed25519"
    assert env.get("SOME_OTHER_TOKEN") == "not-a-configured-key"


def test_no_providers_configured_strips_nothing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Keys living only in secrets.toml never enter the environment, so there
    is nothing to strip and no false positive on a same-named var."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "from-the-shell")
    apply_git_ops_policy(Config())
    env = _captured_git_env(monkeypatch, tmp_path)
    assert env.get("ANTHROPIC_API_KEY") == "from-the-shell"
