# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Secret storage for agent6 (provider API keys, OAuth tokens).

Secrets live in `<global-config-dir>/secrets.toml`, separate from the
config so the config can be shared/committed while keys never are. The
file is treated like an SSH private key:

- it MUST be a regular file owned by the operator,
- it MUST be `0600` (no group/other bits) or agent6 refuses to read it,
- it is written atomically with `0600` and `chown`-ed back to the real
  user when agent6 is running through sudo.

Key resolution order for a provider (most explicit first):

1. the environment variable named by `[providers.<name>].api_key_env`
   (when set and non-empty), keeps CI/secret-manager workflows working,
2. `[providers.<name>].api_key` in `secrets.toml`,
3. nothing (the caller raises a "run `agent6 connect`" error).

Secrets are never written to transcripts, never printed by ``config
show`` (always redacted), and never mounted into the jail, provider
calls happen in agent6's own process, outside the sandbox.
"""

from __future__ import annotations

import os
import stat
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agent6.paths import (
    RealUser,
    chown_to_real_user,
    effective_user,
    mkdir_for_real_user,
    secrets_path,
)
from agent6.portable import atomic_write, locked_file, toml_basic_string


class SecretsError(Exception):
    """Raised when the secrets file is malformed or has unsafe permissions."""


def _require_safe_perms(path: Path, user: RealUser) -> None:
    """Refuse to read a secrets file that others can read or that is not ours."""
    st = path.lstat()
    if not stat.S_ISREG(st.st_mode):
        raise SecretsError(f"{path} is not a regular file; refusing to read secrets from it.")
    if st.st_mode & 0o077:
        raise SecretsError(
            f"{path} has unsafe permissions {stat.S_IMODE(st.st_mode):#o}"
            f" (group/other accessible). Run: chmod 600 {path}"
        )
    # When running as the operator (not sudo), the file must be ours. Under
    # sudo the read takes the real user's file as root, which is expected.
    if os.geteuid() != 0 and st.st_uid != user.uid:
        raise SecretsError(
            f"{path} is owned by uid {st.st_uid}, not you (uid {user.uid});"
            " refusing to read secrets you do not own."
        )


def _read_secrets_toml(path: Path) -> dict[str, Any]:
    """Parse `secrets.toml`, as a SecretsError for anything that stops it.

    THE one reader: an unreadable file (root-owned after a `sudo connect`, a
    chmod 000) is the operator's environment and not a bug in agent6, so it
    raises SecretsError naming the path, never an unhandled PermissionError."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise SecretsError(f"could not read {path}: {exc}") from exc
    try:
        return tomllib.loads(text)
    except tomllib.TOMLDecodeError as exc:
        raise SecretsError(f"{path} is not valid TOML: {exc}") from exc


def load_secrets(user: RealUser | None = None) -> dict[str, Any]:
    """Load and validate `secrets.toml`. Returns `{}` when absent."""
    user = user or effective_user()
    path = secrets_path(user)
    if not path.exists():
        return {}
    _require_safe_perms(path, user)
    return _read_secrets_toml(path)


def resolve_api_key(
    provider_name: str,
    api_key_env: str | None,
    *,
    secrets: dict[str, Any] | None = None,
    user: RealUser | None = None,
) -> str | None:
    """Resolve the API key for one provider, env first then secrets.toml."""
    if api_key_env:
        env_val = os.environ.get(api_key_env, "").strip()
        if env_val:
            return env_val
    data = secrets if secrets is not None else load_secrets(user)
    providers = data.get("providers")
    if isinstance(providers, dict):
        entry = providers.get(provider_name)
        if isinstance(entry, dict):
            key = entry.get("api_key")
            if isinstance(key, str) and key.strip():
                return key.strip()
    return None


def save_secret(
    provider_name: str,
    api_key: str,
    *,
    extra: dict[str, str] | None = None,
    user: RealUser | None = None,
) -> Path:
    """Persist `[providers.<name>].api_key` (and any *extra* string fields)."""
    return _save_provider_entry(provider_name, {"api_key": api_key, **(extra or {})}, user)


def _save_provider_entry(provider_name: str, entry: dict[str, str], user: RealUser | None) -> Path:
    """Replace `[providers.<name>]` with *entry*.

    Rewrites the whole file atomically, preserving other providers'
    entries, then forces `0600` and chowns back to the real user. The
    read-merge-publish cycle runs under `locked_file`: two concurrent
    writers (an `agent6 connect` beside a run refreshing OAuth tokens)
    would otherwise read the same base file and the later publish would
    drop the earlier one's credential.
    """
    user = user or effective_user()
    path = secrets_path(user)
    with locked_file(path):
        data: dict[str, Any] = {}
        if path.exists():
            _require_safe_perms(path, user)
            data = _read_secrets_toml(path)
        providers = data.get("providers")
        if not isinstance(providers, dict):
            providers = {}
        providers[provider_name] = entry
        data["providers"] = providers

        mkdir_for_real_user(path.parent, user)
        path.parent.chmod(0o700)
        text = _render_secrets_toml(data)
        # atomic_write uses tempfile.mkstemp: an unpredictable name opened O_EXCL
        # at 0600, so a pre-planted `secrets.toml.tmp` symlink cannot redirect this
        # write. (A fixed `.tmp` opened O_CREAT|O_TRUNC would follow such a symlink
        # -- an unprivileged user retargeting a root write under `sudo connect`.)
        # A new file inherits mkstemp's 0600; an existing one keeps its mode. Force
        # 0600 anyway so a pre-existing wider-mode file is tightened.
        atomic_write(path, text)
        path.chmod(0o600)
    chown_to_real_user(path.parent, user)
    chown_to_real_user(path, user)
    return path


def delete_provider_secrets(provider_name: str, *, user: RealUser | None = None) -> bool:
    """Remove `[providers.<name>]` from `secrets.toml`. True when it existed."""
    user = user or effective_user()
    path = secrets_path(user)
    with locked_file(path):
        if not path.exists():
            return False
        _require_safe_perms(path, user)
        data: dict[str, Any] = _read_secrets_toml(path)
        providers = data.get("providers")
        if not isinstance(providers, dict) or provider_name not in providers:
            return False
        del providers[provider_name]
        atomic_write(path, _render_secrets_toml(data))
        path.chmod(0o600)
    chown_to_real_user(path, user)
    return True


@dataclass(frozen=True, slots=True)
class OAuthTokens:
    """One provider's OAuth grant as stored in `secrets.toml`.

    `expires_at` is a unix timestamp for the access token; `account_id` is
    the backend account the tokens are bound to ("" when the identity token
    carried none).
    """

    access_token: str
    refresh_token: str
    expires_at: float
    account_id: str = ""


def save_oauth_tokens(
    provider_name: str, tokens: OAuthTokens, *, user: RealUser | None = None
) -> Path:
    """Persist `[providers.<name>]` OAuth tokens (replacing the entry)."""
    return _save_provider_entry(
        provider_name,
        {
            "oauth_access_token": tokens.access_token,
            "oauth_refresh_token": tokens.refresh_token,
            "oauth_expires_at": repr(tokens.expires_at),
            "oauth_account_id": tokens.account_id,
        },
        user,
    )


def load_oauth_tokens(
    provider_name: str,
    *,
    secrets: dict[str, Any] | None = None,
    user: RealUser | None = None,
) -> OAuthTokens | None:
    """The stored OAuth tokens for one provider, or None when absent.

    A present-but-mangled entry (missing token, unparseable expiry) reads as
    absent: every caller's None path already says "run `agent6 connect`",
    which is also the repair.
    """
    data = secrets if secrets is not None else load_secrets(user)
    providers = data.get("providers")
    entry = providers.get(provider_name) if isinstance(providers, dict) else None
    if not isinstance(entry, dict):
        return None
    access = entry.get("oauth_access_token")
    refresh = entry.get("oauth_refresh_token")
    if not (isinstance(access, str) and access and isinstance(refresh, str) and refresh):
        return None
    try:
        expires_at = float(str(entry.get("oauth_expires_at", "0")))
    except ValueError:
        return None
    return OAuthTokens(
        access_token=access,
        refresh_token=refresh,
        expires_at=expires_at,
        account_id=str(entry.get("oauth_account_id", "") or ""),
    )


def _render_secrets_toml(data: dict[str, Any]) -> str:
    """Render the secrets dict back to TOML.

    Hand-rolled (no tomli-w dependency) and intentionally narrow: secrets
    are a flat `[providers.<name>]` table of string fields.
    """
    lines = [
        "# agent6 secrets. Written by `agent6 connect`.",
        "# Keep this file private: it is enforced 0600 and owner-only.",
        "",
    ]
    providers = data.get("providers")
    if isinstance(providers, dict):
        for name in sorted(providers):
            entry = providers[name]
            if not isinstance(entry, dict):
                continue
            lines.append(f"[providers.{name}]")
            for field in sorted(entry):
                value = entry[field]
                if isinstance(value, str):
                    lines.append(f"{field} = {toml_basic_string(value)}")
            lines.append("")
    return "\n".join(lines).rstrip("\n") + "\n"
