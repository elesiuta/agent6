#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Render docs/config.md from config_template.md + the config model (dev tool).

The template is the page: headings, prose, the preset/env/directory tables and
the TOML examples, all hand-written. Where a FIELD table belongs it carries one
marker naming the sections whose leaves fill it:

    <!-- config-table: git.commit.checkpoint git.commit.squash -->

Each row is built from the model -- the key, the default, and
``Field(description=...)`` -- so a renamed field moves its row, a removed one
takes its row with it, and a wrong default is not expressible. The rendered
page carries no markers, so it reads the same on GitHub and on the site.

The gentle-pressure lever: a row that reads badly has a bad description. Fix
the description (or the template), never this script's output.

Regenerate with:
    uv run python docs/gen_config.py

Pinned byte-for-byte by tests/unit/test_config_doc.py.
"""

from __future__ import annotations

import json
import re
import typing
from pathlib import Path
from typing import Any, cast

from pydantic import BaseModel
from pydantic_core import PydanticUndefined

from agent6.config.layer import BUILTIN_PRESET_NOTES, BUILTIN_PRESETS
from agent6.config.model import Config

_ROOT = Path(__file__).resolve().parent.parent
_TEMPLATE = _ROOT / "docs" / "config_template.md"
_OUT = _ROOT / "docs" / "config.md"
REGEN_CMD = "uv run python docs/gen_config.py"

_MARKER = re.compile(r"^<!-- config-table:\s*(.+?)\s*-->$")
_PRESETS_MARKER = "<!-- presets-table -->"

# Fields whose default is resolved at RUNTIME, so the model holds None and only
# the page can say what it becomes. The one hand-declared thing here, for the
# same reason gen_contracts.py declares its registry: a scan cannot know it.
_RUNTIME_DEFAULTS = {
    "context.drop_at_chars": "_adaptive_",
    "context.summarise_at_chars": "_adaptive_",
    "sandbox.memory_limit_mb": "`0` (off)",
    "providers.<name>.base_url": "per (format, deployment)",
    "providers.<name>.auth_style": "per (format, deployment)",
}


def _sections_of(annotation: object) -> list[type[BaseModel]]:
    """Every BaseModel class an annotation can hold, through `Annotated` and
    unions alike -- a discriminated provider entry is both."""
    if isinstance(annotation, type):
        return [annotation] if issubclass(annotation, BaseModel) else []
    found: list[type[BaseModel]] = []
    for arg in typing.get_args(annotation) or ():
        found.extend(_sections_of(arg))
    return found


def leaves() -> dict[str, tuple[str, str]]:
    """Dotted leaf path -> (rendered default, description), in model order.

    A ``dict[str, Section]`` field is one section spelled ``<name>``: the
    config takes any number of them and the page documents the shape once.
    """
    out: dict[str, tuple[str, str]] = {}

    def walk(model: type[BaseModel], prefix: str) -> None:
        for name, field in model.model_fields.items():
            path = f"{prefix}{name}"
            if typing.get_origin(field.annotation) is dict:
                entries = _sections_of(typing.get_args(field.annotation)[1])
                # `dict[str, Section]` is a section map; `dict[str, str]` (extra
                # headers, the skills state) is one leaf holding a table.
                if entries:
                    for entry in entries:
                        walk(entry, f"{path}.<name>.")
                    continue
            nested = _sections_of(field.annotation)
            for section in nested:
                walk(section, f"{path}.")
            if not nested:
                out[path] = (_default_cell(path, field), field.description or "")

    walk(Config, "")
    return out


def _default_cell(path: str, field: object) -> str:
    if path in _RUNTIME_DEFAULTS:
        return _RUNTIME_DEFAULTS[path]
    default = getattr(field, "default", PydanticUndefined)
    factory = getattr(field, "default_factory", None)
    if default is PydanticUndefined and factory is None:
        return "*(required)*"
    value = factory() if factory is not None else default
    if value is None:
        return "none"
    return f"`{json.dumps(list(value) if isinstance(value, tuple) else value)}`"


def _common_parent(sections: list[str]) -> str:
    """The deepest section prefix every section in a table shares. A table over
    one section keys its rows by the bare field name; a table over siblings
    keeps enough of the path to tell them apart (``checkpoint.message``)."""
    parts = [s.split(".") for s in sections]
    shared: list[str] = []
    for piece in zip(*parts, strict=False):
        if len({*piece}) != 1:
            break
        shared.append(piece[0])
    return ".".join(shared)


def render_table(sections: list[str], all_leaves: dict[str, tuple[str, str]]) -> list[str]:
    parent = _common_parent(sections) if len(sections) > 1 else sections[0]
    rows = ["| Field | Default | Meaning |", "|---|---|---|"]
    seen = 0
    for path, (default, description) in all_leaves.items():
        if not any(path == s or path.startswith(f"{s}.") for s in sections):
            continue
        # A nested section's leaves belong to that section's own table.
        owner = path.rsplit(".", 1)[0]
        if owner not in sections:
            continue
        key = path[len(parent) + 1 :] if parent and path.startswith(f"{parent}.") else path
        rows.append(f"| `{key}` | {default} | {description} |")
        seen += 1
    if not seen:
        raise SystemExit(f"config-table marker matched no fields: {' '.join(sections)}")
    return rows


def _flatten(prefix: str, node: dict[str, Any]) -> list[str]:
    out: list[str] = []
    for key, value in node.items():
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            out.extend(_flatten(path, cast(dict[str, Any], value)))
        else:
            out.append(f"`{path} = {json.dumps(value)}`")
    return out


def render_presets_table() -> list[str]:
    """The built-in presets from `BUILTIN_PRESETS` + `BUILTIN_PRESET_NOTES`:
    a renamed or re-tuned preset moves its row."""
    rows = ["| Preset | For | Sets |", "|---|---|---|"]
    for name, overrides in BUILTIN_PRESETS.items():
        sets = ", ".join(_flatten("", overrides)) or "nothing (the defaults)"
        rows.append(f"| `{name}` | {BUILTIN_PRESET_NOTES[name]} | {sets} |")
    return rows


def render(template: str) -> str:
    all_leaves = leaves()
    out: list[str] = [
        # The output must say it is output: a hand edit here is overwritten by
        # the next regeneration and fails the drift test.
        "<!-- Generated from docs/config_template.md by docs/gen_config.py;"
        " edit those, then regenerate. -->",
    ]
    for line in template.splitlines():
        if line.strip() == _PRESETS_MARKER:
            out.extend(render_presets_table())
            continue
        marker = _MARKER.match(line)
        if marker is None:
            out.append(line)
            continue
        out.extend(render_table(marker.group(1).split(), all_leaves))
    return "\n".join(out) + "\n"


def main() -> None:
    page = render(_TEMPLATE.read_text(encoding="utf-8"))
    _OUT.write_text(page, encoding="utf-8")
    print(f"wrote {_OUT.relative_to(_ROOT)} ({len(page.splitlines())} lines)")


if __name__ == "__main__":
    main()
