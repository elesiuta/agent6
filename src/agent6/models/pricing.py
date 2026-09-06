# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Cache-only model price lookups (USD per 1M tokens, (input, output)).

There is NO static price table and no fallback rate: a price either came from
a provider's own models endpoint (fetched + cached by `agent6.models.cache`,
which stores it alongside the model list under
`$XDG_CACHE_HOME/agent6/models/<provider>.json`) or it is unknown. An
outdated hardcoded price is worse than no price: reports render unknown models
as "$?" and the USD budget conversion does not apply.

Today OpenRouter publishes per-model pricing on its /models endpoint.
Anthropic's models API does not include pricing (verified live 2026-07), so a
direct-Anthropic model id falls back to its OpenRouter listing when one is
cached: `claude-haiku-4-5-20251001` -> `anthropic/claude-haiku-4.5` (strip
the date suffix, dot the trailing version). The price itself is still
live-fetched from a provider endpoint; only the id spelling is derived, and
OpenRouter mirrors Anthropic's list prices. A model the derivation cannot map
stays honestly unpriced, and runs rely on token ceilings (and
OpenRouter-style `usage.cost` reporting where available).

This module is import-light (stdlib + `agent6.paths`) so `agent6.budget`
can use it without dragging in config/httpx2. Reads are cache-file only, never
network. Lookups are memoized for the process lifetime: one CLI invocation is
one run, and mid-run price changes are noise.
"""

from __future__ import annotations

import contextlib
import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from agent6.paths import cache_dir

__all__ = ["lookup_price"]

_CLAUDE_DATE_SUFFIX_RE = re.compile(r"-20\d{6}$")
_CLAUDE_TRAILING_VERSION_RE = re.compile(r"-(\d+)-(\d+)$")


def _openrouter_alias(model: str) -> str | None:
    """OpenRouter listing id for a direct-Anthropic model id, or None.

    Only derives for bare `claude-*` ids (never rewrites an already
    namespaced id): drop a `-YYYYMMDD` snapshot suffix, then dot a trailing
    `-N-M` version (`claude-opus-4-8` -> `anthropic/claude-opus-4.8`).
    Ids the rules don't cover (e.g. legacy version-first `claude-3-5-sonnet`)
    return a candidate that simply misses the price map, keeping them
    honestly unpriced rather than mispriced.
    """
    if "/" in model or not model.startswith("claude-"):
        return None
    base = _CLAUDE_DATE_SUFFIX_RE.sub("", model)
    base = _CLAUDE_TRAILING_VERSION_RE.sub(r"-\1.\2", base)
    return f"anthropic/{base}"


def _models_cache_dir() -> Path | None:
    with contextlib.suppress(OSError, RuntimeError):
        return cache_dir() / "models"
    return None


def _cache_state() -> tuple[tuple[str, float], ...]:
    """(name, mtime) per cache file; the memoization key for the parsed map.

    A fetch that lands mid-process (the CLI preflight refreshes the cache
    AFTER the config was first constructed) bumps an mtime and naturally
    invalidates the memo. Stat-ing a handful of files per lookup is cheap
    next to the provider call each lookup accompanies."""
    root = _models_cache_dir()
    if root is None or not root.is_dir():
        return ()
    out: list[tuple[str, float]] = []
    with contextlib.suppress(OSError):
        for path in sorted(root.glob("*.json")):
            with contextlib.suppress(OSError):
                out.append((path.name, path.stat().st_mtime))
    return tuple(out)


@dataclass(frozen=True, slots=True)
class Price:
    """USD per 1M tokens: fresh input, output, and the cache rates a listing
    publishes (OpenRouter's `input_cache_read` / `input_cache_write`); None
    where the listing carries none, and the cost arithmetic then applies
    Anthropic's multipliers to the input rate."""

    input: float
    output: float
    cache_read: float | None = None
    cache_write: float | None = None

    def as_list(self) -> list[float]:
        """The cache file's row: `[input, output]`, plus both cache rates when known."""
        if self.cache_read is None or self.cache_write is None:
            return [self.input, self.output]
        return [self.input, self.output, self.cache_read, self.cache_write]


@lru_cache(maxsize=4)
def _load_pricing(
    state: tuple[tuple[str, float], ...],
) -> dict[str, dict[str, Price]]:
    """The pricing map of every provider cache file, keyed by the provider
    name the file is named for. Never raises."""
    out: dict[str, dict[str, Price]] = {}
    root = _models_cache_dir()
    if root is None:
        return out
    for name, _mtime in state:
        path = root / name
        with contextlib.suppress(OSError, ValueError, TypeError):
            data = json.loads(path.read_text(encoding="utf-8"))
            pricing = data.get("pricing") if isinstance(data, dict) else None
            if not isinstance(pricing, dict):
                continue
            table = out.setdefault(path.stem, {})
            for model, row in pricing.items():
                if (
                    isinstance(model, str)
                    and isinstance(row, list)
                    and len(row) in (2, 4)
                    and all(isinstance(x, (int, float)) and x >= 0 for x in row)
                ):
                    rates = [float(x) for x in row]
                    table[model] = Price(*rates[:2], *rates[2:])
    return out


def _price_in(table: dict[str, Price], model: str) -> Price | None:
    hit = table.get(model)
    if hit is not None:
        return hit
    alias = _openrouter_alias(model)
    return table.get(alias) if alias is not None else None


def lookup_price(model: str, provider: str = "") -> Price | None:
    """The listed price of *model* (USD per 1M tokens), or None if unknown.

    *provider* names the config entry the call went through: two providers
    can list one model id at different prices, and the route's own listing
    is the price that bills, so a model it does not list is unpriced. With
    no provider named, or a route with no cached listing at all (the
    direct-Anthropic alias), the first listing that has the id answers, by
    file name."""
    tables = _load_pricing(_cache_state())
    if provider and provider in tables:
        return _price_in(tables[provider], model)
    for name in sorted(tables):
        hit = _price_in(tables[name], model)
        if hit is not None:
            return hit
    return None
