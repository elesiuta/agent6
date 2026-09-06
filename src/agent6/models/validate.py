# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Pre-spawn model validation: catch a bogus model id before any run or lane
spawns, with a did-you-mean, instead of dying at the first provider call where
the raw upstream 400 leaks.

Two callers, one policy: `validate_configured_model` checks a configured
`models.<role>.model` at run start; `validate_spec_models` checks a `/parallel`
spec's per-lane models (every lane runs on the WORKER provider by construction
-- the lane config overrides only the model -- so the universe is worker-scoped:
the worker's configured model unioned with the worker provider's listing. A
sibling provider's catalog is unrunnable in a lane).

Matching is cache-first: exact id, or the registry's normalization so a
dated/tagged variant of a listed id (`...-20251001`, `...:free`) passes. A
MISS against an existing cache fetches the provider's live listing once (TTL
bypassed, ~1.5s cap) before any hard stop: `refused` always rests on a listing
fetched by this invocation, so a just-pulled local model or a just-published
listing entry is never refused off a stale snapshot. A failed fetch (offline,
provider down) degrades the miss to `warned` and the run proceeds -- the first
provider call is the final arbiter. With no cache at all nothing is fetched and
nothing blocks (a fresh/offline machine, or a provider that lists no models).
Never raises.

Lives in the models layer so all three front-ends and the coordinator's group
dispatcher share one policy without a UI or workflows dependency.
"""

from __future__ import annotations

import difflib
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from agent6.config import ClaudeCodeProviderEntry, Config, ConfigError, RoleName
from agent6.config.layer import load_effective
from agent6.directive import DirectiveError, Segment, parse_spec
from agent6.models.cache import cached_models, fetch_models_live
from agent6.models.registry import normalize_model_id
from agent6.secrets import SecretsError, load_secrets, resolve_api_key

__all__ = [
    "ModelValidation",
    "configured_model_refusal",
    "directive_model_refusal",
    "known_models",
    "refusal_message",
    "validate_configured_model",
    "validate_spec_models",
    "warning_message",
]

_MAX_SUGGESTIONS = 3


def known_models(cfg: Config) -> set[str]:
    """Every model id a `/parallel` lane can actually run, without touching the
    network: the worker's configured model unioned with the worker provider's
    on-disk model-list cache snapshot (lanes inherit the worker provider; only
    the model is overridden per lane). Empty when no worker role is set."""
    worker = cfg.models.worker
    if worker is None:
        return set()
    return {worker.model} | set(cached_models(worker.provider))


@dataclass(frozen=True, slots=True)
class ModelValidation:
    """Outcome of a model check.

    `unknown` lists the named models not found (deduped, in spec order);
    `suggestions` maps each to its closest known ids; `can_validate` is True
    when the miss was judged against real evidence (a matching cache, or a
    listing fetched live by this invocation). `refused` (unknown +
    can_validate) is a hard stop resting on a just-fetched listing; `warned`
    (unknown + no fresh evidence: no cache, or the live re-fetch failed)
    proceeds -- an offline machine is never blocked on a regenerable cache."""

    unknown: tuple[str, ...]
    suggestions: dict[str, tuple[str, ...]]
    can_validate: bool

    @property
    def refused(self) -> bool:
        return bool(self.unknown) and self.can_validate

    @property
    def warned(self) -> bool:
        return bool(self.unknown) and not self.can_validate


def _fresh_listing(cfg: Config, provider_name: str) -> list[str] | None:
    """The provider's LIVE model listing, fetched now (TTL bypassed): the
    evidence a hard refusal needs. None when the fetch fails -- the caller
    degrades to the warn path rather than refuse on a snapshot it could not
    freshen. Keyless (local) providers list without auth; a secrets problem
    just means an unauthenticated attempt."""
    entry = cfg.providers.get(provider_name)
    if entry is None or isinstance(entry, ClaudeCodeProviderEntry):
        return None  # no listing: the binary resolves model names itself
    try:
        secrets = load_secrets()
    except SecretsError:
        secrets = {}
    key = resolve_api_key(provider_name, entry.api_key_env, secrets=secrets)
    return fetch_models_live(provider_name, entry, key)


def _matches(model: str, pool: set[str], norm_pool: set[str]) -> bool:
    """True when *model* is listed: exact id, or its normalized form matches a
    listed id's (a dated/tagged variant of a listed model is provider-plausible,
    so it must never hard-refuse; the call itself is the final arbiter)."""
    return model in pool or normalize_model_id(model) in norm_pool


def _close_ids(typo: str, pool: list[str], bare_to_full: dict[str, list[str]]) -> tuple[str, ...]:
    """Closest known ids to *typo*: matched against the full provider-prefixed ids
    AND against the un-prefixed model segment (the part after the last `/`). The
    bare match catches a short nickname near-miss (`glm`, `kimi-typo`) that scores
    below difflib's cutoff against a full id, because the provider prefix dominates
    the ratio (`glm` vs `z-ai/glm-4.6`). Bare hits map back to full ids (what the
    operator must actually pass); full-id hits keep priority, capped overall."""
    close = list(difflib.get_close_matches(typo, pool, n=_MAX_SUGGESTIONS))
    bare_typo = typo.rsplit("/", 1)[-1]
    for bare in difflib.get_close_matches(bare_typo, sorted(bare_to_full), n=_MAX_SUGGESTIONS):
        close.extend(full for full in bare_to_full[bare] if full not in close)
    return tuple(close[:_MAX_SUGGESTIONS])


def _suggest(unknown: list[str], pool: list[str]) -> dict[str, tuple[str, ...]]:
    """Did-you-mean suggestions for each unknown model, drawn from *pool*."""
    bare_to_full: dict[str, list[str]] = {}
    for full in pool:
        bare_to_full.setdefault(full.rsplit("/", 1)[-1], []).append(full)
    return {model: _close_ids(model, pool, bare_to_full) for model in unknown}


def validate_spec_models(models: list[str | None], cfg: Config) -> ModelValidation:
    """Check per-lane *models* (a `parse_spec` result; `None` = the worker model,
    skipped) against `known_models`. A miss against an existing cache re-checks
    the live listing once before refusing (see module docstring)."""
    known = known_models(cfg)
    norm_known = {normalize_model_id(m) for m in known}
    misses: list[str] = []
    for model in models:
        if model is None or model in misses or _matches(model, known, norm_known):
            continue
        misses.append(model)
    worker = cfg.models.worker
    if not misses:
        can = worker is not None and bool(cached_models(worker.provider))
        return ModelValidation(unknown=(), suggestions={}, can_validate=can)
    if worker is None or not cached_models(worker.provider):
        # No snapshot to judge against: proceed with a warning, never block a
        # fresh/offline machine (and no fetch attempt -- keyed providers already
        # got one in check_provider_keys; a fetchable listing would be cached).
        return ModelValidation(unknown=tuple(misses), suggestions={}, can_validate=False)
    fresh = _fresh_listing(cfg, worker.provider)
    if fresh is None:
        return ModelValidation(unknown=tuple(misses), suggestions={}, can_validate=False)
    known = {worker.model} | set(fresh)
    norm_known = {normalize_model_id(m) for m in known}
    unknown = [m for m in misses if not _matches(m, known, norm_known)]
    if not unknown:
        return ModelValidation(unknown=(), suggestions={}, can_validate=True)
    return ModelValidation(
        unknown=tuple(unknown), suggestions=_suggest(unknown, sorted(known)), can_validate=True
    )


def validate_configured_model(cfg: Config, role: RoleName) -> ModelValidation:
    """Check the CONFIGURED model for *role* against ITS provider's listing, so a
    typo'd `models.<role>.model` is caught at run start.

    Unlike `validate_spec_models` the pool EXCLUDES the model itself -- a
    configured model is trivially in `known_models`, so that check can never
    fail. A miss against an existing cache re-checks the live listing once;
    `refused` always rests on a listing fetched by this invocation, `warned`
    means the re-fetch failed (the caller prints it and proceeds). No cache at
    all (a fresh/offline machine, or a provider that lists no models) stays a
    silent proceed, with no fetch attempt."""
    rm = cfg.models.resolve(role)
    if rm is None:
        return ModelValidation(unknown=(), suggestions={}, can_validate=False)
    cache = set(cached_models(rm.provider))
    if not cache:
        return ModelValidation(unknown=(), suggestions={}, can_validate=False)
    if _matches(rm.model, cache, {normalize_model_id(c) for c in cache}):
        return ModelValidation(unknown=(), suggestions={}, can_validate=True)
    fresh = _fresh_listing(cfg, rm.provider)
    if fresh is None:
        return ModelValidation(unknown=(rm.model,), suggestions={}, can_validate=False)
    fresh_set = set(fresh)
    if _matches(rm.model, fresh_set, {normalize_model_id(c) for c in fresh_set}):
        return ModelValidation(unknown=(), suggestions={}, can_validate=True)
    return ModelValidation(
        unknown=(rm.model,),
        suggestions=_suggest([rm.model], sorted(fresh_set)),
        can_validate=True,
    )


def configured_model_refusal(v: ModelValidation, role: str) -> str:
    """Refusal text for a typo'd CONFIGURED role model (a refused
    `validate_configured_model`): name the bad model, its closest known ids, and
    how to fix it. The listing was re-fetched live before this refusal, so
    refreshing the cache cannot fix it."""
    model = v.unknown[0]
    close = v.suggestions.get(model, ())
    suffix = f" Closest: {', '.join(close)}." if close else ""
    return (
        f"configured models.{role}.model {model!r} is not in its provider's model"
        f" listing (checked live).{suffix} Fix it in your config."
    )


def refusal_message(v: ModelValidation, *, directive: bool) -> str:
    """The refusal text for an `unknown + can_validate` result: one line per
    unknown model with its closest matches. On a directive surface (the composers
    and the coordinator, where the same token could be task text) add the backtick
    hint."""
    lines = [
        f"unknown model {model!r} in /parallel spec"
        + (f"; closest: {', '.join(close)}" if (close := v.suggestions.get(model, ())) else "")
        for model in v.unknown
    ]
    if directive:
        lines.append("backtick the word if you meant it as task text.")
    return "\n".join(lines)


def directive_model_refusal(
    cwd: Path, segments: Sequence[Segment], config_path: Path | None = None
) -> str | None:
    """Refuse a `/parallel` directive that names a model the configured
    providers' cache says doesn't exist, before any spawn (the surface's normal
    error path, nothing spawned). None = every model checks out, or there is no
    cache to check against (a fresh/offline machine proceeds; the detached
    lane's own preflight warns). A malformed or over-`max_lanes` spec surfaces
    its grammar error."""
    try:
        cfg = load_effective(cwd, config_path).config
    except ConfigError:
        return None  # a broken config is its own separate error; don't mask it here
    try:
        cap = cfg.parallel.max_lanes
        models = [m for seg in segments for m in parse_spec(seg.spec, limit=cap)]
    except DirectiveError as exc:
        return str(exc)
    verdict = validate_spec_models(models, cfg)
    return refusal_message(verdict, directive=True) if verdict.refused else None


def warning_message(v: ModelValidation) -> str:
    """The single warning line for an `unknown + not can_validate` result: no
    fresh listing to check against (no cache, or the live re-fetch failed), so
    proceed but name the unvalidated model(s)."""
    return (
        f"unvalidated model(s) {', '.join(v.unknown)}: no fresh provider listing"
        " to check against; proceeding (run `agent6 model` to refresh)."
    )
