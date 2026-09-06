# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Tree-sitter symbol index for the LLM-visible navigation tools.

Provides a `SymbolIndex` over a project root:

    outline(path)            -> symbol declarations in one file (nested too)
    find_definition(name)    -> every declaration of `name` across the project
    find_references(name)    -> every identifier occurrence of `name` (incl. def)
    hot_symbols()            -> cross-file reference ranking (repo priors)
    file_outlines()          -> per-file symbol lists across the index

The index is built lazily on the first query, then updated incrementally when
the caller marks files changed via `mark_changed(path)` / `mark_deleted(path)`.
Re-parses happen in batch on the next query, so a worker can call `apply_edit`
many times and pay the parse cost only when it next asks for symbol info.

Languages live in `_LANG_TABLE` (extension -> tree-sitter name + a definitions
query); an unlisted extension is silently ignored. Matching is
identifier-level, never inside strings or comments, but cross-file
*resolution* (which `foo` is the same symbol?) needs a real LSP and is out of
scope.

This module is NOT exposed via `ToolDispatcher` here; that wiring lives
in `tools/dispatch.py` + `tools/schema.py` and requires a security review
note when added.
"""

from __future__ import annotations

import threading
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from tree_sitter import Parser, Query, QueryCursor
from tree_sitter_language_pack import get_language

from agent6.tools._path_safety import Workspace, contain, read_bytes_contained


@dataclass(frozen=True, slots=True)
class Symbol:
    """A definition site. `path` is absolute; `line`/`col` are 1-based,
    the convention every emitting surface shares (the LSP twins and the
    system-prompt repo map)."""

    name: str
    kind: str  # 'function' | 'class' | 'method' | 'struct' | 'enum' | ...
    path: Path
    line: int
    col: int


@dataclass(frozen=True, slots=True)
class Reference:
    """An identifier occurrence. `path` is absolute; `line`/`col` are 1-based.

    Includes the definition site itself. Callers that want call-sites-only
    should subtract the result of `find_definition(name)`.
    """

    name: str
    path: Path
    line: int
    col: int


# ---------------------------------------------------------------------------
# Per-language queries
# ---------------------------------------------------------------------------

_PYTHON_DEFS: Final = """
(function_definition name: (identifier) @function)
(class_definition name: (identifier) @class)
"""

_RUST_DEFS: Final = """
(function_item name: (identifier) @function)
(struct_item name: (type_identifier) @struct)
(enum_item name: (type_identifier) @enum)
(trait_item name: (type_identifier) @trait)
(type_item name: (type_identifier) @type)
(mod_item name: (identifier) @module)
(const_item name: (identifier) @const)
(static_item name: (identifier) @const)
(macro_definition name: (identifier) @macro)
"""

_TS_DEFS: Final = """
(function_declaration name: (identifier) @function)
(class_declaration name: (type_identifier) @class)
(method_definition name: (property_identifier) @method)
(interface_declaration name: (type_identifier) @interface)
(type_alias_declaration name: (type_identifier) @type)
(enum_declaration name: (identifier) @enum)
"""

_GO_DEFS: Final = """
(function_declaration name: (identifier) @function)
(method_declaration name: (field_identifier) @method)
(type_spec name: (type_identifier) @struct type: (struct_type))
(type_spec name: (type_identifier) @interface type: (interface_type))
(type_spec name: (type_identifier) @type
  type: [(type_identifier) (function_type) (pointer_type) (array_type)
         (slice_type) (map_type) (channel_type) (qualified_type) (generic_type)])
(type_alias name: (type_identifier) @type)
(const_spec name: (identifier) @const)
"""

_JAVA_DEFS: Final = """
(class_declaration name: (identifier) @class)
(interface_declaration name: (identifier) @interface)
(enum_declaration name: (identifier) @enum)
(record_declaration name: (identifier) @class)
(annotation_type_declaration name: (identifier) @interface)
(method_declaration name: (identifier) @method)
(constructor_declaration name: (identifier) @method)
(annotation_type_element_declaration name: (identifier) @method)
(enum_constant name: (identifier) @const)
"""

_JS_DEFS: Final = """
(function_declaration name: (identifier) @function)
(generator_function_declaration name: (identifier) @function)
(function_expression name: (identifier) @function)
(generator_function name: (identifier) @function)
(class_declaration name: (identifier) @class)
(class name: (identifier) @class)
(method_definition name: (property_identifier) @method)
(method_definition name: (private_property_identifier) @method)
"""

_C_DEFS: Final = """
(function_definition declarator: (function_declarator declarator: (identifier) @function))
(function_definition declarator: (pointer_declarator
  declarator: (function_declarator declarator: (identifier) @function)))
(function_definition declarator: (pointer_declarator
  declarator: (pointer_declarator
    declarator: (function_declarator declarator: (identifier) @function))))
(struct_specifier name: (type_identifier) @struct body: (_))
(union_specifier name: (type_identifier) @struct body: (_))
(enum_specifier name: (type_identifier) @enum body: (_))
(type_definition declarator: (type_identifier) @type)
(type_definition declarator: (pointer_declarator declarator: (type_identifier) @type))
(type_definition declarator: (function_declarator
  declarator: (parenthesized_declarator (pointer_declarator (type_identifier) @type))))
(preproc_def name: (identifier) @const)
(preproc_function_def name: (identifier) @macro)
"""

_CPP_DEFS: Final = """
(function_definition
  declarator: (function_declarator
    declarator: (identifier) @function))
(function_definition
  declarator: (function_declarator
    declarator: (operator_name) @function))
(function_definition
  declarator: (function_declarator
    declarator: (qualified_identifier name: (identifier) @method)))
(function_definition
  declarator: (function_declarator
    declarator: (qualified_identifier name: (operator_name) @method)))
(function_definition
  declarator: (function_declarator
    declarator: (qualified_identifier name: (destructor_name) @method)))
(field_declaration
  declarator: (function_declarator
    declarator: (field_identifier) @method))
(field_declaration
  declarator: (function_declarator
    declarator: (operator_name) @method))
(declaration
  declarator: (function_declarator
    declarator: (identifier) @function))
(declaration
  declarator: (function_declarator
    declarator: (destructor_name) @method))
(class_specifier name: (type_identifier) @class body: (_))
(struct_specifier name: (type_identifier) @struct body: (_))
(enum_specifier name: (type_identifier) @enum body: (_))
(namespace_definition name: (namespace_identifier) @module)
(alias_declaration name: (type_identifier) @type)
(type_definition declarator: (type_identifier) @type)
"""

_CSHARP_DEFS: Final = """
(namespace_declaration name: (identifier) @module)
(namespace_declaration name: (qualified_name) @module)
(file_scoped_namespace_declaration name: (identifier) @module)
(file_scoped_namespace_declaration name: (qualified_name) @module)
(class_declaration name: (identifier) @class)
(record_declaration name: (identifier) @class)
(struct_declaration name: (identifier) @struct)
(interface_declaration name: (identifier) @interface)
(enum_declaration name: (identifier) @enum)
(delegate_declaration name: (identifier) @type)
(method_declaration name: (identifier) @method)
(constructor_declaration name: (identifier) @method)
(destructor_declaration name: (identifier) @method)
(operator_declaration
  ["+" "-" "*" "/" "%" "==" "!=" "<" ">" "<=" ">=" "++" "--"
   "!" "~" "&" "|" "^" "<<" ">>" "true" "false"] @method)
; conversion_operator_declaration (implicit/explicit operator) and
; indexer_declaration (this[...]) are intentionally NOT captured: they have no
; name node, so the only available text is the target type / bracket list, which
; makes a poor, noisy symbol name. The (rarely-navigated) gap is deliberate.
(property_declaration name: (identifier) @method)
(event_declaration name: (identifier) @method)
(field_declaration (modifier "const")
  (variable_declaration (variable_declarator name: (identifier) @const)))
"""

_RUBY_DEFS: Final = """
(method name: [(identifier) (constant) (setter) (operator)] @method)
(singleton_method name: [(identifier) (constant) (setter) (operator)] @method)
(class name: (constant) @class)
(class name: (scope_resolution name: (constant) @class))
(module name: (constant) @module)
(module name: (scope_resolution name: (constant) @module))
(assignment left: (constant) @const)
"""

_PHP_DEFS: Final = """
(function_definition name: (name) @function)
(method_declaration name: (name) @method)
(class_declaration name: (name) @class)
(interface_declaration name: (name) @interface)
(trait_declaration name: (name) @trait)
(enum_declaration name: (name) @enum)
(namespace_definition name: (namespace_name) @module)
(const_element . (name) @const)
(enum_case name: (name) @const)
"""

# Per-language identifier query for references. Different grammars surface
# names under different node types (rust splits `identifier` vs
# `type_identifier`; ts adds `property_identifier`).
_REF_QUERIES: Final[dict[str, str]] = {
    "python": "(identifier) @id",
    "rust": "[(identifier) (type_identifier)] @id",
    "typescript": "[(identifier) (type_identifier) (property_identifier)] @id",
    "tsx": "[(identifier) (type_identifier) (property_identifier)] @id",
    "go": "[(identifier) (type_identifier) (field_identifier) (package_identifier)] @id",
    "java": "(identifier) @id",
    "javascript": "[(identifier) (property_identifier)] @id",
    "c": "[(identifier) (type_identifier) (field_identifier)] @id",
    "cpp": "[(identifier) (type_identifier) (field_identifier) (namespace_identifier)] @id",
    "csharp": "(identifier) @id",
    "ruby": "[(identifier) (constant)] @id",
    "php": "(name) @id",
}

# suffix -> (tree-sitter language name, definitions query)
_LANG_TABLE: Final[dict[str, tuple[str, str]]] = {
    ".py": ("python", _PYTHON_DEFS),
    ".rs": ("rust", _RUST_DEFS),
    ".ts": ("typescript", _TS_DEFS),
    ".tsx": ("tsx", _TS_DEFS),
    ".go": ("go", _GO_DEFS),
    ".java": ("java", _JAVA_DEFS),
    ".js": ("javascript", _JS_DEFS),
    ".jsx": ("javascript", _JS_DEFS),
    ".mjs": ("javascript", _JS_DEFS),
    ".cjs": ("javascript", _JS_DEFS),
    ".c": ("c", _C_DEFS),
    ".h": ("cpp", _CPP_DEFS),
    ".cpp": ("cpp", _CPP_DEFS),
    ".cc": ("cpp", _CPP_DEFS),
    ".cxx": ("cpp", _CPP_DEFS),
    ".hpp": ("cpp", _CPP_DEFS),
    ".hh": ("cpp", _CPP_DEFS),
    ".hxx": ("cpp", _CPP_DEFS),
    ".cs": ("csharp", _CSHARP_DEFS),
    ".rb": ("ruby", _RUBY_DEFS),
    ".php": ("php", _PHP_DEFS),
}

# Directories never indexed. Hard-coded; we are not parsing .gitignore here.
_DEFAULT_EXCLUDES: Final[tuple[str, ...]] = (
    ".git",
    ".venv",
    "venv",
    "node_modules",
    "target",
    "dist",
    "build",
    "__pycache__",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
)


# ---------------------------------------------------------------------------
# The index
# ---------------------------------------------------------------------------


class SymbolIndex:
    """Lazy, incrementally-updated tree-sitter symbol index for a project root."""

    def __init__(
        self,
        ws: Workspace,
        *,
        excludes: tuple[str, ...] = _DEFAULT_EXCLUDES,
    ) -> None:
        self._ws = ws
        self._root = ws.root.resolve()
        self._excludes = excludes
        # path -> per-file caches. Absolute, resolved paths.
        self._symbols: dict[Path, list[Symbol]] = {}
        self._refs: dict[Path, list[Reference]] = {}
        self._scanned = False
        self._dirty: set[Path] = set()
        # path -> (st_mtime_ns, st_size) recorded at parse time. Used to detect
        # out-of-band changes/deletions (run_command formatters, rm, git mv, sed)
        # that never go through mark_changed/mark_deleted.
        self._stamps: dict[Path, tuple[int, int]] = {}
        # lang_name -> (parser, def_query, ref_query). Built on first use.
        self._parsers: dict[str, tuple[Parser, Query, Query]] = {}
        # Guards every public reader/mutator so the index can be shared across
        # the concurrent explore-review seats (one dispatcher, one index, N
        # ThreadPoolExecutor threads). Re-entrant because public methods call
        # each other (e.g. queries rely on _ensure_fresh) and _ensure_fresh is
        # also invoked under the lock.
        self._lock = threading.RLock()

    # ------------------------------------------------------------------
    # Dirty-tracking surface for the dispatcher to call after apply_edit
    # ------------------------------------------------------------------

    def mark_changed(self, path: Path) -> None:
        """Record that `path` was created or modified; re-parsed on next query."""
        with self._lock:
            self._dirty.add(path.resolve())

    def mark_deleted(self, path: Path) -> None:
        """Drop a path from the index immediately."""
        with self._lock:
            p = path.resolve()
            self._symbols.pop(p, None)
            self._refs.pop(p, None)
            self._stamps.pop(p, None)
            self._dirty.discard(p)

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def outline(self, path: Path) -> list[Symbol]:
        """Top-level + nested definitions in one file, in source order."""
        with self._lock:
            self._ensure_fresh()
            p = path.resolve()
            if p not in self._symbols and p.is_file():
                # On-demand parse for a file we hadn't seen at scan time.
                self._reparse(p)
            out = list(self._symbols.get(p, []))
            out.sort(key=lambda s: (s.line, s.col))
            return out

    def find_definition(self, name: str) -> list[Symbol]:
        """All definition sites of `name` across the project, in path order."""
        with self._lock:
            self._ensure_fresh()
            out: list[Symbol] = []
            for syms in self._symbols.values():
                for s in syms:
                    if s.name == name:
                        out.append(s)
            out.sort(key=lambda s: (str(s.path), s.line, s.col))
            return out

    def find_references(self, name: str) -> list[Reference]:
        """All identifier occurrences of `name` (incl. defs), in path order."""
        with self._lock:
            self._ensure_fresh()
            out: list[Reference] = []
            for refs in self._refs.values():
                for r in refs:
                    if r.name == name:
                        out.append(r)
            out.sort(key=lambda r: (str(r.path), r.line, r.col))
            return out

    def hot_symbols(
        self,
        *,
        max_symbols: int = 20,
        min_files_referenced: int = 2,
    ) -> list[tuple[str, str, str, int, int]]:
        """Top symbols by cross-file reference count.

        Returns a list of (name, kind, def_path, def_line, files_referenced)
        tuples, sorted by `files_referenced` descending. Only symbols whose
        identifier appears in at least `min_files_referenced` distinct
        files are included - this filters out file-local helpers and
        surfaces only symbols whose rename/signature change would touch
        multiple files. Definition site is taken as the first symbol's
        path:line; ambiguous names with multiple definitions return the
        alphabetically-first def.

        Cheap planner prior: knowing that "build_kernel" is referenced
        across 4 files lets the planner enumerate those files in
        relevant_paths up-front, the same payoff shape as co-change
        pairs but driven by static analysis instead of git history.
        Works on fresh repos (no history needed).
        """
        with self._lock:
            self._ensure_fresh()
            files_per_name: defaultdict[str, set[Path]] = defaultdict(set)
            ref_count: Counter[str] = Counter()
            for refs in self._refs.values():
                for r in refs:
                    files_per_name[r.name].add(r.path)
                    ref_count[r.name] += 1
            defs_by_name: dict[str, list[Symbol]] = defaultdict(list)
            for syms in self._symbols.values():
                for s in syms:
                    defs_by_name[s.name].append(s)
        qualifying: list[tuple[str, str, str, int, int]] = []
        for name, files in files_per_name.items():
            n_files = len(files)
            if n_files < min_files_referenced:
                continue
            defs = defs_by_name.get(name) or []
            # Some references have no def in the index (e.g. stdlib /
            # third-party names); skip those - the planner can't action
            # them.
            if not defs:
                continue
            d = sorted(defs, key=lambda s: (str(s.path), s.line))[0]
            try:
                rel = d.path.resolve().relative_to(self._root.resolve())
                rel_str = str(rel)
            except ValueError:
                rel_str = str(d.path)
            qualifying.append((name, d.kind, rel_str, d.line, n_files))
        qualifying.sort(key=lambda t: (-t[4], t[0]))
        return qualifying[:max_symbols]

    def file_outlines(self) -> dict[Path, list[Symbol]]:
        """Per-file symbol lists (nested included) across the whole index.

        Returns a fresh dict mapping absolute file path -> in-source-order
        list of Symbol records. Used by the system-prompt repo map to
        give the agent a one-line-per-symbol outline of the codebase
        without round-tripping `outline` for every file.
        """
        with self._lock:
            self._ensure_fresh()
            out: dict[Path, list[Symbol]] = {}
            for path, syms in self._symbols.items():
                out[path] = sorted(syms, key=lambda s: (s.line, s.col))
            return out

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _ensure_fresh(self) -> None:
        # Callers hold self._lock.
        if not self._scanned:
            self._scan_all()
            self._scanned = True
        # Detect out-of-band changes/deletions (run_command formatters, rm,
        # git mv, sed) that never went through mark_changed/mark_deleted.
        # mark_* remain a cheap fast-path; this stat sweep is the source of
        # truth so the index self-heals regardless of who mutated the tree.
        for p in list(self._symbols.keys()):
            try:
                st = p.stat()
                cur = (st.st_mtime_ns, st.st_size)
            except OSError:
                # File vanished -> evict.
                self._symbols.pop(p, None)
                self._refs.pop(p, None)
                self._stamps.pop(p, None)
                self._dirty.discard(p)
                continue
            if self._stamps.get(p) != cur:
                self._dirty.add(p)
        if not self._dirty:
            return
        for p in list(self._dirty):
            self._reparse(p)
        self._dirty.clear()

    def _scan_all(self) -> None:
        for path in self._root.rglob("*"):
            if not path.is_file():
                continue
            if self._included_rel(path) is None:
                continue
            if self._lang_for(path) is None:
                continue
            self._reparse(path)

    def _reparse(self, path: Path) -> None:
        p = path.resolve()
        rel = self._included_rel(p)
        if rel is None or not p.is_file():
            self._symbols.pop(p, None)
            self._refs.pop(p, None)
            self._stamps.pop(p, None)
            return
        lang_name = self._lang_for(p)
        if lang_name is None:
            return
        bits = self._parser_for(lang_name)
        if bits is None:
            return
        parser, def_query, ref_query = bits
        try:
            src = read_bytes_contained(contain(self._root, rel))
        except OSError:
            self._symbols.pop(p, None)
            self._refs.pop(p, None)
            self._stamps.pop(p, None)
            return
        try:
            tree = parser.parse(src)
        except Exception:  # tree-sitter errors are opaque; absorb per-file failures
            # Record the stamp anyway so a persistently-unparseable file is not
            # re-read and re-parsed on every query (the stat-sweep would keep
            # flagging it dirty); drop any now-stale symbols/refs.
            self._symbols.pop(p, None)
            self._refs.pop(p, None)
            self._record_stamp(p)
            return
        root = tree.root_node
        syms: list[Symbol] = []
        for kind, nodes in QueryCursor(def_query).captures(root).items():
            for n in nodes:
                try:
                    name = src[n.start_byte : n.end_byte].decode("utf-8")
                except UnicodeDecodeError:
                    continue
                syms.append(
                    Symbol(
                        name=name,
                        kind=kind,
                        path=p,
                        line=n.start_point[0] + 1,
                        col=n.start_point[1] + 1,
                    )
                )
        refs: list[Reference] = []
        for _, nodes in QueryCursor(ref_query).captures(root).items():
            for n in nodes:
                try:
                    name = src[n.start_byte : n.end_byte].decode("utf-8")
                except UnicodeDecodeError:
                    continue
                refs.append(
                    Reference(
                        name=name,
                        path=p,
                        line=n.start_point[0] + 1,
                        col=n.start_point[1] + 1,
                    )
                )
        self._symbols[p] = syms
        self._refs[p] = refs
        self._record_stamp(p)

    def _record_stamp(self, p: Path) -> None:
        """Remember (mtime_ns, size) so the stat-sweep treats p as processed.
        Recorded on BOTH a successful parse and an absorbed parse failure, so a
        persistently-unparseable file is not re-read on every query."""
        try:
            st = p.stat()
            self._stamps[p] = (st.st_mtime_ns, st.st_size)
        except OSError:
            self._stamps.pop(p, None)

    def language_of(self, path: Path) -> str | None:
        """The grammar that parses *path*, by suffix; None when none does."""
        return self._lang_for(path)

    def indexes(self, path: Path) -> bool:
        """Whether *path* is inside the indexed workspace: under the root,
        outside the excluded directories, and not hidden by the boundary."""
        return self._included_rel(path.resolve()) is not None

    def _included_rel(self, p: Path) -> Path | None:
        """The path relative to root, or None when *p* lies outside it, under
        an excluded directory, or hidden by the workspace boundary. One answer
        for all three, so what is in scope and what the contained read walks
        cannot disagree. Comparing the RELATIVE parts keeps an excluded dirname
        in an outside ancestor from mattering.

        The boundary check belongs here rather than at the query: a hidden file
        that reached the index would leak its symbol NAMES and line numbers
        through find_definition even though nothing could read it.
        """
        try:
            rel = p.relative_to(self._root)
        except ValueError:
            return None
        if any(part in self._excludes for part in rel.parts):
            return None
        if self._ws.is_denied(p):
            return None
        return rel

    def _lang_for(self, path: Path) -> str | None:
        info = _LANG_TABLE.get(path.suffix)
        return info[0] if info else None

    def _parser_for(self, lang_name: str) -> tuple[Parser, Query, Query] | None:
        cached = self._parsers.get(lang_name)
        if cached is not None:
            return cached
        # Find the def query for this language (linear scan; tiny table).
        def_src: str | None = None
        for _, (n, q) in _LANG_TABLE.items():
            if n == lang_name:
                def_src = q
                break
        if def_src is None:
            return None
        try:
            lang = get_language(lang_name)  # pyright: ignore[reportArgumentType]
        except Exception:  # unknown lang name -> skip
            return None
        parser = Parser(lang)
        def_query = Query(lang, def_src)
        ref_query = Query(lang, _REF_QUERIES.get(lang_name, "(identifier) @id"))
        self._parsers[lang_name] = (parser, def_query, ref_query)
        return self._parsers[lang_name]


__all__ = [
    "Reference",
    "Symbol",
    "SymbolIndex",
]
