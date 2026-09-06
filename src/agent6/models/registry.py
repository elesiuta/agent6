# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Model facts for the run path: context windows (they size adaptive
compaction) and decompose-first families. Entries change only with bench
evidence; the live cache covers windows this table omits. Lookups never
raise and never touch the network.
"""

from __future__ import annotations

import re

from agent6.config import Config
from agent6.models.cache import cached_context_window
from agent6.providers.openai import is_openai_direct_host, sent_reasoning_effort
from agent6.types import RoleName

__all__ = [
    "BUNDLED_CONTEXT_WINDOWS",
    "DECOMPOSE_WIN_MODEL_FAMILIES",
    "compaction_thresholds",
    "context_window",
    "decompose_default",
    "resolved_adaptive_values",
    "role_effort",
]

# Context windows in TOKENS for tested-or-popular models. Bundled because a
# provider listing can omit the window (Anthropic's /models does not report
# it) and the first run has no cache yet; wins over the live cache when both
# know a model. Keep ids canonical (no date/`:tag` suffix --
# `normalize_model_id` strips those before matching).
BUNDLED_CONTEXT_WINDOWS: dict[str, int] = {
    # Anthropic. The 5-family ships 1M as the default AND maximum (no beta
    # header, standard pricing); the 4.x standard window is 200k with the 1M
    # beta opt-in, so pin [context] explicitly if you enable that beta.
    "claude-fable-5": 1_000_000,
    "claude-opus-5": 1_000_000,
    "claude-sonnet-5": 1_000_000,
    "claude-opus-4-8": 200_000,
    "claude-sonnet-4-6": 200_000,
    "claude-sonnet-4-5": 200_000,
    "claude-haiku-4-5": 200_000,
    "claude-3-5-sonnet": 200_000,
    "claude-3-5-haiku": 200_000,
    # OpenRouter open-weights we bench against (cross-checked against the live
    # listing).
    "moonshotai/kimi-k2.6": 262_144,
    "moonshotai/kimi-k2": 131_072,
    "qwen/qwen3-coder": 1_048_576,
    "qwen/qwen3-coder-30b-a3b-instruct": 160_000,
    "z-ai/glm-4.6": 202_752,
    "z-ai/glm-5.2": 1_048_576,
    "deepseek/deepseek-v3.2-exp": 163_840,
}

# Adaptive sizing. tokens ~= chars/4 (matches the loop's `context_chars`
# approximation). Tier-1 elides old tool_results once they pass ~45% of the
# window (Claude Code's continuous local tiers); tier-2 summarise-and-restart
# is a near-edge valve, firing only within a fixed reserve of the window
# (pi's reserveTokens shape; Claude's autocompact likewise). The reserve
# leaves room for the next turn's output and the summary call itself.
_CHARS_PER_TOKEN = 4
_DROP_FRACTION = 0.45
_RESERVE_TOKENS = 16_384
# Used when the window is unknown: the historical fixed defaults, so behaviour
# is unchanged for unsizable models. Mirrors workflows._compaction
# DROP_BLOCKS_AT_CHARS / SUMMARISE_AT_CHARS.
_FALLBACK_DROP_CHARS = 256_000
_FALLBACK_SUMMARISE_CHARS = 768_000


def normalize_model_id(model_id: str) -> str:
    """Strip a trailing `-YYYYMMDD` snapshot date or `:tag` so dated/tagged
    ids (`claude-haiku-4-5-20251001`, `qwen/qwen3-coder:free`) match the
    canonical bundled key."""
    base = model_id.split(":", 1)[0]
    return re.sub(r"-\d{8}$", "", base)


def _bundled_context_window(model_id: str) -> int | None:
    if model_id in BUNDLED_CONTEXT_WINDOWS:
        return BUNDLED_CONTEXT_WINDOWS[model_id]
    return BUNDLED_CONTEXT_WINDOWS.get(normalize_model_id(model_id))


def context_window(provider_name: str, model_id: str) -> int | None:
    """Best-effort context window (tokens) for a configured model. Never raises.

    Bundled table (curated; tested models + Anthropic) first, then the live
    model cache (`context_length` from the provider listing, populated by
    completion / `agent6 model`), then None. Reads only -- never triggers a
    network fetch -- so it is safe and fast on the run path.
    """
    return _bundled_context_window(model_id) or cached_context_window(
        provider_name, (model_id, normalize_model_id(model_id))
    )


def compaction_thresholds(
    provider_name: str,
    model_id: str,
    *,
    drop_override: int | None,
    summarise_override: int | None,
) -> tuple[int, int]:
    """Effective `(compact_drop_at_chars, compact_summarise_at_chars)`.

    Explicit config wins (both set, by construction -- the config validator
    requires both-or-neither). Otherwise size from the model's context window
    (tier-1 at ~45% of it, tier-2 at the window minus a fixed 16k-token
    reserve); if the window is unknown, the fixed 256k/768k defaults. Never
    raises.
    """
    if drop_override is not None and summarise_override is not None:
        return drop_override, summarise_override
    ctx = context_window(provider_name, model_id)
    if ctx is None or ctx <= 0:
        return _FALLBACK_DROP_CHARS, _FALLBACK_SUMMARISE_CHARS
    drop = int(ctx * _CHARS_PER_TOKEN * _DROP_FRACTION)
    summarise = max(drop + 1, (ctx - _RESERVE_TOKENS) * _CHARS_PER_TOKEN)
    return drop, summarise


# Model families with a MEASURED decompose-first win
# (bench/coreagent/FINDINGS.md, thrust 2): forcing up-front decomposition
# converted premature finishes into full component coverage
# (mistral-small-3.2-24b: textkit +0.53, rpn +0.13, ledger +0.18). Every other
# benched model (qwen3-coder-30b, qwen3.6-35b, claude-haiku-4-5) sat at the
# score ceiling and paid a 2-4x iteration tax, so `prompt.decompose = "auto"`
# resolves to on ONLY for these families and unknown models stay off. Grow
# this list with bench evidence, not vibes.
DECOMPOSE_WIN_MODEL_FAMILIES: tuple[str, ...] = ("mistral-small-3.2",)


def decompose_default(model_id: str) -> bool:
    """True when `prompt.decompose = "auto"` should enable decompose-first
    prompting for *model_id*: its family has a measured win in bench/coreagent.
    Family matching ignores the org prefix and any date/`:tag` suffix, so
    `mistralai/mistral-small-3.2-24b-instruct:free` matches
    `mistral-small-3.2`."""
    family = normalize_model_id(model_id).rsplit("/", 1)[-1].lower()
    return family.startswith(DECOMPOSE_WIN_MODEL_FAMILIES)


def role_effort(cfg: Config, role: RoleName) -> str | None:
    """The reasoning effort *role*'s calls carry, or None when agent6 sends no
    effort at all and the provider's own default decides (a non-reasoning
    OpenAI-compatible model, ChatGPT, the Claude Code binary).

    Reads the configured `[models.<role>].effort` when set, else the default
    each wire applies: openai-compatible reasoning models `low`
    (`sent_reasoning_effort`), Anthropic no thinking, which is `off`.
    """
    rm = cfg.models.resolve(role)
    if rm is None:
        return None
    entry = cfg.providers.get(rm.provider)
    if entry is None:
        return None
    match entry.api_format:
        case "anthropic":
            return rm.effort or "off"
        case "chatgpt" | "claude_code":
            return rm.effort
        case _:
            return sent_reasoning_effort(
                rm.model,
                rm.effort,
                direct_openai=is_openai_direct_host(entry.base_url, entry.deployment),
            )


def resolved_adaptive_values(cfg: Config) -> dict[str, object]:
    """Config settings whose effective value is resolved at runtime, so a UI
    (`config show`, the TUI/web config page) can display the real number rather
    than the unset/adaptive placeholder: the adaptive compaction thresholds
    sized from the worker model's context window, the auto decompose decision,
    and each role's unset effort. Empty when nothing resolves."""
    out: dict[str, object] = {}
    for role in ("worker", "reviewer", "planner"):
        role_model = cfg.models.resolve(role)
        if role_model is None or role_model.effort is not None:
            continue
        if (effort := role_effort(cfg, role)) is not None:
            out[f"models.{role}.effort"] = effort
    rm = cfg.models.resolve("worker")
    if rm is None:
        return out
    drop, summarise = compaction_thresholds(
        rm.provider,
        rm.model,
        drop_override=cfg.context.drop_at_chars,
        summarise_override=cfg.context.summarise_at_chars,
    )
    out["context.drop_at_chars"] = drop
    out["context.summarise_at_chars"] = summarise
    if cfg.prompt.decompose == "auto":
        out["prompt.decompose"] = "on" if decompose_default(rm.model) else "off"
    return out
