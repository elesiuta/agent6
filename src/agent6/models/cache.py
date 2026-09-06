# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Live, cached provider model listings for shell completion + interactive prompts.

Model catalogs change constantly (new OpenRouter routes, new Claude/GPT
snapshots), so agent6 never ships a curated static list that would go stale.
Instead it queries each provider's list endpoint on demand and caches the
result under `$XDG_CACHE_HOME/agent6/models/<provider>.json` for a short
TTL, long enough that tab-completion does not hammer the network on every
keystroke, short enough that a freshly-released model shows up within minutes
without the operator hunting for a cache to clear.

This runs in the operator's own shell process (completion / interactive
`agent6 model`), never inside a run sandbox, so a direct HTTP call is fine.
Everything here is best-effort: :func:`list_models` NEVER raises, on a cache
miss + network failure it falls back to the stale cache, then to an empty
list, so completion degrades to free-text rather than breaking the shell.
"""

from __future__ import annotations

import contextlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlsplit

import httpx2

from agent6.config import (
    AnthropicProviderEntry,
    ChatGPTProviderEntry,
    ClaudeCodeProviderEntry,
    OpenAIProviderEntry,
    ProviderEntry,
)
from agent6.models.pricing import Price
from agent6.paths import cache_dir
from agent6.providers.types import ProviderError
from agent6.providers.wire import auth_header
from agent6.secrets import load_oauth_tokens

__all__ = ["cached_context_window", "list_models"]

_ANTHROPIC_VERSION = "2023-06-01"
_CACHE_TTL_S = 600  # 10 minutes
_FETCH_TIMEOUT_S = 1.5  # keep tab-completion snappy


def _cache_path(provider_name: str) -> Path | None:
    """Cache file for *provider_name*, or None when the name is not a safe
    single path component. Provider names are config table keys; guard against
    `/` or `..` so a crafted name can't write the cache outside cache_dir().
    """
    if provider_name in ("", ".", "..") or provider_name != Path(provider_name).name:
        return None
    return cache_dir() / "models" / f"{provider_name}.json"


def _read_cache(path: Path | None) -> list[str] | None:
    if path is None:
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    models = data.get("models") if isinstance(data, dict) else None
    if isinstance(models, list) and all(isinstance(m, str) for m in models):
        return models
    return None


def _write_cache(
    path: Path | None,
    models: list[str],
    pricing: dict[str, Price],
    context: dict[str, int],
) -> None:
    if path is None:
        return
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        body: dict[str, object] = {"models": models}
        if pricing:
            # Consumed by agent6.models.pricing.lookup_price (USD per 1M tokens,
            # [input, output] plus the cache read and write rates where the
            # listing publishes them). Only providers that publish pricing on
            # their models endpoint (OpenRouter does, Anthropic does not) get
            # this key; there is deliberately no static fallback anywhere.
            body["pricing"] = {m: p.as_list() for m, p in pricing.items()}
        if context:
            # Per-model context window in tokens, consumed by `context_window`
            # to size adaptive compaction. Same story as pricing: only providers
            # that publish `context_length` (OpenRouter does) populate it.
            body["context"] = dict(context)
        path.write_text(json.dumps(body), encoding="utf-8")
    except OSError:
        pass  # cache is throwaway; a write failure must not break completion


def _parse_models(payload: object) -> list[str]:
    """Extract model ids from an OpenAI-/Anthropic-style `{"data": [...]}` body."""
    data = payload.get("data") if isinstance(payload, dict) else None
    out: list[str] = []
    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                mid = item.get("id")
                if isinstance(mid, str) and mid:
                    out.append(mid)
    return out


def _per_mtok(pricing: dict[str, Any], key: str) -> float | None:
    """One USD-per-token string of an OpenRouter pricing block as USD per 1M
    tokens; None when absent, boolean (float(True) is $1) or unparseable."""
    raw = pricing.get(key)
    if raw is None or isinstance(raw, bool):
        return None
    try:
        value = float(raw) * 1_000_000
    except (TypeError, ValueError):
        return None
    return value if value >= 0 else None


def _parse_pricing(payload: object) -> dict[str, Price]:
    """Extract per-model pricing from an OpenRouter-style `{"data": [...]}` body.

    OpenRouter reports `pricing.prompt`/`pricing.completion` (and, for models
    that cache, `input_cache_read`/`input_cache_write`) as USD per TOKEN
    strings; normalize to USD per 1M tokens. Models without a usable
    prompt/completion pair are simply absent (unknown beats wrong); the cache
    rates ride along only when both parse."""
    data = payload.get("data") if isinstance(payload, dict) else None
    out: dict[str, Price] = {}
    if not isinstance(data, list):
        return out
    for item in data:
        if not isinstance(item, dict):
            continue
        mid = item.get("id")
        pricing = item.get("pricing")
        if not (isinstance(mid, str) and mid and isinstance(pricing, dict)):
            continue
        in_mtok, out_mtok = _per_mtok(pricing, "prompt"), _per_mtok(pricing, "completion")
        if in_mtok is None or out_mtok is None:
            continue
        read, write = (
            _per_mtok(pricing, "input_cache_read"),
            _per_mtok(pricing, "input_cache_write"),
        )
        if read is None or write is None:
            out[mid] = Price(in_mtok, out_mtok)
        else:
            out[mid] = Price(in_mtok, out_mtok, read, write)
    return out


def _parse_context(payload: object) -> dict[str, int]:
    """Extract per-model context window (tokens) from a `{"data": [...]}` body.

    OpenRouter reports `context_length` per model; Anthropic's listing does
    not, so those models simply fall back to the bundled table. Models without
    a usable positive integer are absent (unknown beats wrong)."""
    data = payload.get("data") if isinstance(payload, dict) else None
    out: dict[str, int] = {}
    if not isinstance(data, list):
        return out
    for item in data:
        if not isinstance(item, dict):
            continue
        mid = item.get("id")
        ctx = item.get("context_length")
        # not-bool: bool subclasses int, so JSON `true` would cache a 1-token
        # context window and collapse the compaction thresholds every turn.
        if (
            isinstance(mid, str)
            and mid
            and isinstance(ctx, int)
            and not isinstance(ctx, bool)
            and ctx > 0
        ):
            out[mid] = ctx
    return out


def _models_endpoint(
    entry: AnthropicProviderEntry | OpenAIProviderEntry, api_key: str | None
) -> tuple[str, dict[str, str]]:
    """The (url, headers) for *entry*'s `/models` listing, auth included.

    Shared by the cache fetch and the `connect` key probe so both hit the
    endpoint the same way the call path authenticates.
    """
    url = entry.base_url.rstrip("/") + "/models"
    headers = dict(entry.extra_headers)
    # Anthropic's direct /models needs the version header; Vertex/Azure have no
    # uniform /models endpoint, so listing there is best-effort (the caller
    # swallows the failure). Auth uses the same style the call path uses.
    if isinstance(entry, AnthropicProviderEntry) and entry.deployment == "direct":
        headers["anthropic-version"] = _ANTHROPIC_VERSION
    authed = auth_header(entry.auth_style, api_key or "")
    if authed is not None:
        headers[authed[0]] = authed[1]
    return url, headers


# The backend hides models newer than the claimed client, keyed on each
# model's minimal_client_version. This pin names the wire feature set agent6
# implements and has verified live (Responses SSE, function tools, reasoning
# summaries with the effort tiers through max); every current model gates at
# or below it. A future model gated ABOVE the pin stays hidden until its wire
# needs are implemented and the pin is raised deliberately: the server's
# compatibility filter is authoritative, never claimed past.
_CHATGPT_CLIENT_VERSION = "1.0.0"


def _chatgpt_models_endpoint(
    provider_name: str, entry: ChatGPTProviderEntry
) -> tuple[str, dict[str, str]]:
    """The subscription backend's own listing, authorized by the stored
    sign-in (best effort: an expired access token just fails the fetch and
    the caller falls back to the cache; runs refresh tokens, listings don't).
    """
    tokens = load_oauth_tokens(provider_name)
    if tokens is None:
        raise ProviderError(
            f"no ChatGPT sign-in stored for {provider_name!r}; run `agent6 connect {provider_name}`"
        )
    url = f"{entry.base_url.rstrip('/')}/models?client_version={_CHATGPT_CLIENT_VERSION}"
    headers = dict(entry.extra_headers)
    headers["authorization"] = f"Bearer {tokens.access_token}"
    if tokens.account_id:
        headers["chatgpt-account-id"] = tokens.account_id
    headers["originator"] = "agent6"
    return url, headers


def _chatgpt_listing(payload: object) -> tuple[list[str], dict[str, int]]:
    """`{"models": [{slug, context_window, visibility}, ...]}` -> (ids, context).

    Hidden entries (internal models) are left out of completion; a typed
    hidden slug still works, the backend is the validator.
    """
    models = payload.get("models") if isinstance(payload, dict) else None
    ids: list[str] = []
    context: dict[str, int] = {}
    if not isinstance(models, list):
        return ids, context
    for item in models:
        if not isinstance(item, dict):
            continue
        slug = item.get("slug")
        if not isinstance(slug, str) or not slug or item.get("visibility") == "hide":
            continue
        ids.append(slug)
        ctx = item.get("context_window")
        if isinstance(ctx, int) and not isinstance(ctx, bool) and ctx > 0:
            context[slug] = ctx
    return ids, context


def _fetch(
    provider_name: str, entry: ProviderEntry, api_key: str | None, timeout_s: float
) -> tuple[list[str], dict[str, Price], dict[str, int]]:
    if isinstance(entry, ClaudeCodeProviderEntry):
        return [], {}, {}  # no endpoint: the binary resolves model names itself
    if isinstance(entry, ChatGPTProviderEntry):
        url, headers = _chatgpt_models_endpoint(provider_name, entry)
        resp = httpx2.get(url, headers=headers, timeout=timeout_s)
        resp.raise_for_status()
        ids, context = _chatgpt_listing(resp.json())
        return ids, {}, context  # a subscription prices nothing
    url, headers = _models_endpoint(entry, api_key)
    resp = httpx2.get(url, headers=headers, timeout=timeout_s)
    resp.raise_for_status()
    payload = resp.json()
    return _parse_models(payload), _parse_pricing(payload), _parse_context(payload)


@dataclass(frozen=True, slots=True)
class KeyProbeResult:
    """Outcome of a `connect` key-validation probe."""

    ok: bool
    status: Literal["ok", "auth_failed", "unreachable", "unsupported"]
    detail: str


def probe_provider_key(
    entry: ProviderEntry, api_key: str, *, timeout_s: float = 10.0
) -> KeyProbeResult:
    """Check whether *api_key* authenticates against *entry*'s `/models`.

    A read-only GET (no remote content is executed), used by `agent6 connect`
    to catch a bad key at setup instead of mid-run. Distinguishes a working key
    (2xx) from a rejected one (401/403) from an unreachable endpoint, unlike
    `list_models` which swallows every failure into an empty list. Vertex/Azure
    have no uniform `/models` listing, so they report `unsupported` rather
    than a misleading failure.

    Caveat: a 401/403 is a reliable "bad key" everywhere, but a 2xx only proves
    validity when `/models` is auth-gated (Anthropic, OpenAI). OpenRouter's
    `/models` is PUBLIC (returns 200 for any key), so for it we probe the
    auth-gated `/key` endpoint instead. A different OpenAI-compatible provider
    with a public `/models` would report a false `ok` -- the negative
    (auth_failed) is the trustworthy signal.
    """
    if (
        isinstance(entry, (ChatGPTProviderEntry, ClaudeCodeProviderEntry))
        or entry.deployment != "direct"
    ):
        if isinstance(entry, ChatGPTProviderEntry):
            detail = "ChatGPT signs in via OAuth, not a key"
        elif isinstance(entry, ClaudeCodeProviderEntry):
            detail = "Claude Code signs in with its own login, not a key"
        else:
            detail = "no /models listing for this deployment"
        return KeyProbeResult(ok=True, status="unsupported", detail=detail)
    try:
        url, headers = _models_endpoint(entry, api_key)
    except ProviderError as exc:
        # A credential auth_header refuses (control char / non-ASCII) is an
        # unusable key: report it as such rather than crashing `connect`.
        return KeyProbeResult(ok=False, status="auth_failed", detail=str(exc)[:200])
    # OpenRouter's /models is public (200 for any key); probe its auth-gated /key
    # instead. Match the parsed host, not a base_url substring (a proxy URL could
    # merely contain the string).
    host = (urlsplit(entry.base_url).hostname or "").lower()
    if host == "openrouter.ai" or host.endswith(".openrouter.ai"):
        url = entry.base_url.rstrip("/") + "/key"
    try:
        resp = httpx2.get(url, headers=headers, timeout=timeout_s)
    except (httpx2.HTTPError, OSError) as exc:
        return KeyProbeResult(ok=False, status="unreachable", detail=str(exc)[:200])
    if resp.status_code in (401, 403):
        return KeyProbeResult(ok=False, status="auth_failed", detail=f"HTTP {resp.status_code}")
    if resp.status_code >= 400:
        return KeyProbeResult(ok=False, status="unreachable", detail=f"HTTP {resp.status_code}")
    try:
        n = len(_parse_models(resp.json()))
        detail = f"provider returned {n} models" if n else "provider accepted the key"
    except (ValueError, json.JSONDecodeError):
        detail = "provider accepted the key"
    return KeyProbeResult(ok=True, status="ok", detail=detail)


def list_models(
    provider_name: str,
    entry: ProviderEntry,
    api_key: str | None,
    *,
    ttl_s: int = _CACHE_TTL_S,
    timeout_s: float = _FETCH_TIMEOUT_S,
) -> list[str]:
    """Best-effort list of model ids offered by *entry*. Never raises.

    Returns a fresh cache when one exists within *ttl_s*; otherwise fetches
    live, rewrites the cache, and returns it. On any failure (no key, network
    error, bad payload) falls back to a stale cache, then an empty list.
    """
    path = _cache_path(provider_name)
    cached = _read_cache(path)
    age = float("inf")
    if path is not None:
        with contextlib.suppress(OSError):
            age = time.time() - path.stat().st_mtime
    if cached is not None and age < ttl_s:
        return cached
    return fetch_models_live(provider_name, entry, api_key, timeout_s=timeout_s) or cached or []


def fetch_models_live(
    provider_name: str,
    entry: ProviderEntry,
    api_key: str | None,
    *,
    timeout_s: float = _FETCH_TIMEOUT_S,
) -> list[str] | None:
    """Fetch *entry*'s live listing NOW (no TTL gate). Never raises.

    On success rewrites the cache and returns the ids; on any failure (network
    error, bad payload, empty listing) returns None so the caller can tell
    fresh evidence from a stale fallback -- `models.validate` hard-refuses only
    on a listing this returned. The TTL-gated read-through is `list_models`.
    """
    try:
        models, pricing, context = _fetch(provider_name, entry, api_key, timeout_s)
    except (httpx2.HTTPError, ValueError, OSError, ProviderError):
        # ProviderError: a malformed credential auth_header refused. It falls
        # back like any other fetch failure, keeping the "Never raises" contract.
        return None
    if not models:
        return None
    _write_cache(_cache_path(provider_name), models, pricing, context)
    return models


# The price source for bare `claude-*` ids (pricing's OpenRouter alias
# path). Public listing, fetched keyless, only when no openrouter provider is
# configured to refresh it with a key.
# Security review note: a fixed, provider-shaped host (the canonical
# OpenRouter base_url) fetched with a keyless GET at preflight; no secret
# leaves the process and nothing from the response is executed.
_PRICING_CATALOG_BASE_URL = "https://openrouter.ai/api/v1"


def refresh_pricing_catalog(*, ttl_s: int = _CACHE_TTL_S) -> None:
    """TTL-gated keyless refresh of the OpenRouter catalog cache.

    Direct-Anthropic model ids are priced through pricing's alias into this
    catalog; a config with only [providers.anthropic] otherwise never fetches
    it and every claude-* run is honestly-but-needlessly unpriced (found by
    an unmetered $ cap inside a cold-cache container)."""
    entry = OpenAIProviderEntry(api_format="openai", base_url=_PRICING_CATALOG_BASE_URL)
    list_models("openrouter", entry, None, ttl_s=ttl_s)


def cached_models(provider_name: str) -> list[str]:
    """Model ids from the on-disk cache only (no network). `[]` if nothing has
    been cached for *provider_name* yet. For instant typeahead suggestions; pair
    with :func:`list_models` (in a worker) to refresh from the live listing."""
    return _read_cache(_cache_path(provider_name)) or []


# --- context-window reads for the capability registry ---------------------


def cached_context_window(provider_name: str, keys: tuple[str, ...]) -> int | None:
    """Read `context_length` from the provider's model cache for the first
    of *keys* that has one, if a listing has been fetched. Best-effort:
    returns None on any miss. The capability layer (`models.registry`) passes
    the raw and normalized model ids; this module only owns the file format.
    """
    path = _cache_path(provider_name)
    if path is None:
        return None
    try:
        body = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    ctx = body.get("context") if isinstance(body, dict) else None
    if not isinstance(ctx, dict):
        return None
    for key in keys:
        val = ctx.get(key)
        if isinstance(val, int) and not isinstance(val, bool) and val > 0:
            return val
    return None
