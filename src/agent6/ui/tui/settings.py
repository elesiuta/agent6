# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""UI-only preferences for the TUI (theme, copy method).

Stored in `<global-config-dir>/ui.toml` — a sibling of `config.toml` and
`secrets.toml`, NOT part of the agent config. A theme or a copy method is a
viewer preference, not agent behavior, so it must never go through the config
schema or the (shareable, per-repo) config layers. This module is the whole
contract:

    get_theme() / save_theme(name)
    get_copy_method() / save_copy_method(name)

Everything is best-effort: a missing, unreadable, or corrupt `ui.toml` simply
degrades to the default — a UI preference must never break the TUI. Writes are
atomic and `chown`-ed back to the real user under sudo (same idiom as
`secrets.py`); there's deliberately no `tomli_w` dependency, the writer is a
tiny hand-rolled serializer for the one flat `[ui]` table.
"""

from __future__ import annotations

import tomllib
from typing import Any

from agent6.paths import (
    RealUser,
    chown_to_real_user,
    effective_user,
    mkdir_for_real_user,
    ui_settings_path,
)
from agent6.portable import atomic_write, toml_basic_string

DEFAULT_THEME = "agent6-dark"
DEFAULT_COPY_METHOD = "auto"


def load_ui_settings(user: RealUser | None = None) -> dict[str, Any]:
    """Read `ui.toml`; `{}` if absent, unreadable, or corrupt (never raises)."""
    path = ui_settings_path(user)
    try:
        return tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError):
        return {}


def get_theme(default: str = DEFAULT_THEME) -> str:
    """The persisted theme name, or *default* when unset/invalid."""
    ui = load_ui_settings().get("ui")
    name = ui.get("theme") if isinstance(ui, dict) else None
    return name if isinstance(name, str) and name else default


def _save_ui_key(key: str, value: str, user: RealUser | None = None) -> None:
    """Persist `[ui].<key> = value` atomically. Best-effort: a failed save is
    swallowed so a viewer preference can never break the UI."""
    user = user or effective_user()
    path = ui_settings_path(user)
    data = load_ui_settings(user)
    ui = data.get("ui")
    if not isinstance(ui, dict):
        ui = {}
    ui[key] = value
    data["ui"] = ui
    try:
        mkdir_for_real_user(path.parent, user)
        # atomic_write uses mkstemp (unpredictable name, O_EXCL): a pre-planted
        # `ui.toml.tmp` symlink cannot redirect this write. A fixed `.tmp` +
        # write_text (O_CREAT|O_TRUNC) would follow such a symlink, and this path
        # can run under sudo (it chowns to the real user) -- an arbitrary-file
        # truncate-as-root primitive.
        atomic_write(path, _render_ui_toml(data))
        chown_to_real_user(path.parent, user)
        chown_to_real_user(path, user)
    except OSError:
        pass  # a viewer preference is not worth a crash


def save_theme(name: str, user: RealUser | None = None) -> None:
    """Persist `[ui].theme = name` (best-effort)."""
    _save_ui_key("theme", name, user)


def get_copy_method(default: str = DEFAULT_COPY_METHOD) -> str:
    """The persisted copy method, or *default* when unset/invalid."""
    ui = load_ui_settings().get("ui")
    name = ui.get("copy_method") if isinstance(ui, dict) else None
    return name if isinstance(name, str) and name else default


def save_copy_method(name: str, user: RealUser | None = None) -> None:
    """Persist `[ui].copy_method = name` (best-effort)."""
    _save_ui_key("copy_method", name, user)


def _render_ui_toml(data: dict[str, Any]) -> str:
    """Render the flat `[ui]` table back to TOML (no `tomli_w` dependency)."""
    lines = ["# agent6 UI preferences (theme, etc.). Written by the TUI.", ""]
    ui = data.get("ui")
    if isinstance(ui, dict) and ui:
        lines.append("[ui]")
        for key in sorted(ui):
            value = ui[key]
            if isinstance(value, bool):
                lines.append(f"{key} = {'true' if value else 'false'}")
            elif isinstance(value, str):
                lines.append(f"{key} = {toml_basic_string(value)}")
            elif isinstance(value, int):
                lines.append(f"{key} = {value}")
        lines.append("")
    return "\n".join(lines).rstrip("\n") + "\n"
