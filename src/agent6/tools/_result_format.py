# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Post-dispatch helpers: the jail passthrough env, metric-score parsing, and
compacting tool args for the event log.
"""

from __future__ import annotations

import os
import re
from typing import Any

PASSTHROUGH_ENV_KEYS = ("LANG", "LC_ALL", "TERM", "CI")


def passthrough_env() -> dict[str, str]:
    return {k: os.environ[k] for k in PASSTHROUGH_ENV_KEYS if k in os.environ}


def parse_metric_score(stdout: str, stderr: str, *, pattern: str) -> float | None:
    """Apply the metric `pattern` regex to combined stdout+stderr.

    Shared metric parser; centralised so the workflow and tool handler
    scores from the same command output. Returns `None` on regex compile
    failure, no-match, or non-numeric capture group - the caller treats
    that as "no score this turn" and falls back to raw stdout inspection.
    """
    combined = f"{stdout}\n{stderr}"
    try:
        m = re.search(pattern, combined)
    except re.error:
        return None
    if m is None:
        return None
    try:
        return float(m.group(1))
    except (ValueError, IndexError, TypeError):
        # TypeError: an optional/alternation capture group that did not
        # participate in the match yields None, and float(None) raises it.
        return None


def truncate_args(raw: dict[str, Any], *, max_value_chars: int = 200) -> dict[str, Any]:
    """Cheap argument preview for telemetry; truncates strings longer than
    *max_value_chars* and lists longer than 10 items, at every depth (an
    apply_edit's `edits` is a short list of dicts whose strings hold whole
    files)."""
    return {k: _clip_value(v, max_value_chars) for k, v in raw.items()}


def _clip_value(v: Any, max_value_chars: int) -> Any:
    if isinstance(v, str):
        return v if len(v) <= max_value_chars else v[:max_value_chars] + f"… ({len(v)} chars)"
    if isinstance(v, dict):
        return {k: _clip_value(x, max_value_chars) for k, x in v.items()}  # pyright: ignore[reportUnknownVariableType, reportUnknownArgumentType]
    if isinstance(v, list | tuple):
        items = [_clip_value(x, max_value_chars) for x in v[:10]]  # pyright: ignore[reportUnknownVariableType, reportUnknownArgumentType]
        return [*items, f"… ({len(v)} items)"] if len(v) > 10 else items
    return v
