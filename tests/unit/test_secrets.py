# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Tests for agent6.secrets (storage, permissions, key resolution)."""

from __future__ import annotations

import stat
import threading
from pathlib import Path

import pytest

from agent6 import secrets
from agent6.secrets import (
    OAuthTokens,
    SecretsError,
    load_oauth_tokens,
    resolve_api_key,
    save_oauth_tokens,
    save_secret,
)


@pytest.fixture
def gcfg(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    monkeypatch.setenv("AGENT6_CONFIG_HOME", str(tmp_path / "g"))
    return tmp_path / "g"


def test_save_secret_is_0600(gcfg: Path) -> None:
    p = secrets.save_secret("anthropic", "sk-ant-xyz")
    assert p.is_file()
    mode = stat.S_IMODE(p.stat().st_mode)
    assert mode == 0o600
    assert secrets.resolve_api_key("anthropic", None) == "sk-ant-xyz"


def test_an_unreadable_secrets_file_is_a_named_refusal(gcfg: Path) -> None:
    """Root-owned after a `sudo connect`, or a plain chmod 000: the operator's
    environment, not a bug in agent6. It escaped as an unexpected
    PermissionError with a saved traceback and an invitation to report it, and
    no run could start."""
    path = secrets.save_secret("anthropic", "sk-ant-xyz")
    path.chmod(0o000)
    try:
        with pytest.raises(SecretsError, match="could not read"):
            secrets.load_secrets()
    finally:
        path.chmod(0o600)


def test_save_secret_preserves_other_providers(gcfg: Path) -> None:
    secrets.save_secret("anthropic", "sk-ant-1")
    secrets.save_secret("openrouter", "sk-or-2")
    assert secrets.resolve_api_key("anthropic", None) == "sk-ant-1"
    assert secrets.resolve_api_key("openrouter", None) == "sk-or-2"


def test_save_secret_escapes_control_chars(gcfg: Path) -> None:
    # A control char in a pasted key must not write unparseable secrets.toml.
    # A raw newline/\x01 in a basic string is illegal TOML, so the whole file
    # fails to parse and EVERY provider's key reads back missing -- while the
    # save reported success.
    secrets.save_secret("openrouter", "sk-or-clean")
    secrets.save_secret("anthropic", "sk-\x01\nbroken")
    assert secrets.resolve_api_key("anthropic", None) == "sk-\x01\nbroken"
    assert secrets.resolve_api_key("openrouter", None) == "sk-or-clean"


def test_env_takes_precedence_over_secrets(gcfg: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    secrets.save_secret("anthropic", "from-secrets")
    monkeypatch.setenv("MY_KEY", "from-env")
    assert secrets.resolve_api_key("anthropic", "MY_KEY") == "from-env"
    # Empty env falls back to secrets.
    monkeypatch.setenv("MY_KEY", "")
    assert secrets.resolve_api_key("anthropic", "MY_KEY") == "from-secrets"


def test_resolve_missing_returns_none(gcfg: Path) -> None:
    assert secrets.resolve_api_key("nope", None) is None


def test_load_secrets_refuses_group_readable(gcfg: Path) -> None:
    p = secrets.save_secret("anthropic", "sk-ant-xyz")
    p.chmod(0o644)
    with pytest.raises(SecretsError, match="unsafe permissions"):
        secrets.load_secrets()


def test_load_secrets_absent_is_empty(gcfg: Path) -> None:
    assert secrets.load_secrets() == {}


def test_save_secret_does_not_follow_a_planted_tmp_symlink(gcfg: Path, tmp_path: Path) -> None:
    """A pre-planted `secrets.toml.tmp` symlink must not redirect the write to
    its target (the sudo-connect symlink-redirect vector). atomic_write uses an
    unpredictable mkstemp name, so a fixed-name symlink is simply ignored."""
    victim = tmp_path / "victim"
    victim.write_text("KEEP ME\n", encoding="utf-8")
    gcfg.mkdir(parents=True, exist_ok=True)
    (gcfg / "secrets.toml.tmp").symlink_to(victim)
    secrets.save_secret("anthropic", "sk-ant-xyz")
    assert victim.read_text(encoding="utf-8") == "KEEP ME\n"  # untouched
    assert secrets.resolve_api_key("anthropic", None) == "sk-ant-xyz"
    assert not (gcfg / "secrets.toml").is_symlink()


def test_concurrent_save_secret_loses_no_provider(gcfg: Path) -> None:
    """Two concurrent connects both read the same base file and the later
    publish silently dropped the earlier provider's credential (lost update).
    save_secret serializes on portable.locked_file, removed on release."""
    n = 8
    barrier = threading.Barrier(n)

    def save(i: int) -> None:
        barrier.wait()
        secrets.save_secret(f"prov{i}", f"sk-{i}")

    threads = [threading.Thread(target=save, args=(i,)) for i in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)
    for i in range(n):
        assert secrets.resolve_api_key(f"prov{i}", None) == f"sk-{i}"
    p = secrets.save_secret("final", "sk-final")
    assert stat.S_IMODE(p.stat().st_mode) == 0o600
    assert not p.with_name(p.name + ".lock").exists()


def test_oauth_tokens_round_trip_beside_api_keys(gcfg: Path) -> None:
    """OAuth tokens replace their provider's entry, preserve siblings, stay 0600."""
    save_secret("anthropic", "sk-ant-123")
    tokens = OAuthTokens(
        access_token="eyJ.access", refresh_token="rt-1", expires_at=1755.5, account_id="acct-9"
    )
    path = save_oauth_tokens("chatgpt", tokens)
    assert (path.stat().st_mode & 0o777) == 0o600
    assert load_oauth_tokens("chatgpt") == tokens
    assert resolve_api_key("anthropic", None) == "sk-ant-123"
    # Re-connect rotates the whole entry; no stale fields survive.
    save_oauth_tokens("chatgpt", OAuthTokens("a2", "r2", 2000.0, "acct-9"))
    loaded = load_oauth_tokens("chatgpt")
    assert loaded is not None and loaded.access_token == "a2" and loaded.refresh_token == "r2"


def test_load_oauth_tokens_absent_or_mangled_is_none(gcfg: Path) -> None:
    """No entry, an api-key-only entry, and an unparseable expiry all read as
    absent (the caller's repair path is `agent6 connect` either way)."""
    assert load_oauth_tokens("chatgpt") is None
    save_secret("chatgpt", "sk-not-oauth")
    assert load_oauth_tokens("chatgpt") is None
    assert (
        load_oauth_tokens(
            "chatgpt",
            secrets={
                "providers": {
                    "chatgpt": {
                        "oauth_access_token": "a",
                        "oauth_refresh_token": "r",
                        "oauth_expires_at": "not-a-float",
                    }
                }
            },
        )
        is None
    )


def test_delete_provider_secrets_preserves_siblings(gcfg: Path) -> None:
    save_secret("anthropic", "sk-1")
    save_oauth_tokens("chatgpt", OAuthTokens("a", "r", 100.0, "id"))
    assert secrets.delete_provider_secrets("chatgpt") is True
    assert secrets.delete_provider_secrets("chatgpt") is False
    assert secrets.load_oauth_tokens("chatgpt") is None
    assert secrets.resolve_api_key("anthropic", None) == "sk-1"
