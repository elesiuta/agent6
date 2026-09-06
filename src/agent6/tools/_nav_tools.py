# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Tree-sitter navigation handlers: outline, find_definition,
find_references.

The lazy-built `SymbolIndex` singleton stays on `ToolDispatcher` (shared
with the non-tool passthroughs `hot_symbols` / `file_outlines` and with
`apply_edit`/`apply_patch`'s change notification); these functions take
the dispatcher's ensure callable and invoke it only after argument/path
validation (an index scan never happens for a rejected call)."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from agent6.tools._path_safety import Workspace
from agent6.tools.errors import ToolError
from agent6.tools.index import SymbolIndex
from agent6.tools.results import DefinitionsResult, OutlineResult, ReferencesResult
from agent6.tools.schema import (
    FindDefinitionInput,
    FindReferencesInput,
    OutlineInput,
)

INDEX_RESULT_CAP = 500


def outline(
    ws: Workspace, ensure_index: Callable[[], SymbolIndex], raw: dict[str, Any]
) -> OutlineResult:
    args = OutlineInput.model_validate(raw)
    sp = ws.resolve_read(args.path)
    if not sp.abs_path.is_file():
        raise ToolError(f"Not a file: {args.path}")
    index = ensure_index()
    # An empty answer over a file full of symbols is the one thing the model
    # acts on, wrongly: name the cause instead.
    if index.language_of(sp.abs_path) is None:
        raise ToolError(f"outline: no parser for {sp.abs_path.suffix or 'a file without a suffix'}")
    if not index.indexes(sp.abs_path):
        raise ToolError(f"outline: {args.path} is outside the indexed workspace")
    syms = index.outline(sp.abs_path)
    out = [{"name": s.name, "kind": s.kind, "line": s.line, "col": s.col} for s in syms]
    truncated = len(out) > INDEX_RESULT_CAP
    return OutlineResult(symbols=tuple(out[:INDEX_RESULT_CAP]), truncated=truncated)


def find_definition(
    ws: Workspace, ensure_index: Callable[[], SymbolIndex], raw: dict[str, Any]
) -> DefinitionsResult:
    args = FindDefinitionInput.model_validate(raw)
    defs = ensure_index().find_definition(args.symbol)
    out: list[dict[str, Any]] = []
    for s in defs:
        try:
            rel = s.path.relative_to(ws.root)
        except ValueError:
            continue
        out.append({"name": s.name, "kind": s.kind, "path": str(rel), "line": s.line, "col": s.col})
    truncated = len(out) > INDEX_RESULT_CAP
    return DefinitionsResult(definitions=tuple(out[:INDEX_RESULT_CAP]), truncated=truncated)


def find_references(
    ws: Workspace, ensure_index: Callable[[], SymbolIndex], raw: dict[str, Any]
) -> ReferencesResult:
    args = FindReferencesInput.model_validate(raw)
    refs = ensure_index().find_references(args.symbol)
    out: list[dict[str, Any]] = []
    for r in refs:
        try:
            rel = r.path.relative_to(ws.root)
        except ValueError:
            continue
        out.append({"name": r.name, "path": str(rel), "line": r.line, "col": r.col})
    truncated = len(out) > INDEX_RESULT_CAP
    return ReferencesResult(references=tuple(out[:INDEX_RESULT_CAP]), truncated=truncated)
