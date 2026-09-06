#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Derive the data-contract reference from the source tree (dev tool, not CI).

Each registered contract (typed shapes, plus the run/machine wire snapshot
builders) gets a card: name, home module, kind, invariant, who writes it, who
reads it, what pins guard it. The cards are DERIVED, not hand-curated -- the
invariant prose is lifted from the module and class docstrings, the kind and
type counts from the AST, the reader set from an import scan of ``src/agent6``,
and the guarding tests from an import/golden scan of ``tests/``. Only two things
are declared per contract (the ``CONTRACTS`` registry below): the module and the
pins/writers, which are judgement calls a scan cannot make honestly.

The gentle-pressure lever: a card that reads badly has a bad docstring. Fix the
docstring (or the registry), never this script's output.

Two outputs:

- ``docs/data-contracts.md`` -- the mkdocs page, pinned byte-for-byte by
  ``tests/unit/test_data_contracts_doc.py``. Regenerate with:
      uv run python docs/gen_contracts.py
- a self-contained HTML artifact (the tach.toml module graph + the cards) under
  the gitignored ``docs/screenshots/out/``, which the operator publishes by hand.
"""

from __future__ import annotations

import argparse
import ast
import html
import re
import tomllib
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_SRC = _ROOT / "src" / "agent6"
_TESTS = _ROOT / "tests"
_MD_OUT = _ROOT / "docs" / "data-contracts.md"
_HTML_OUT = _ROOT / "docs" / "screenshots" / "out" / "data-contracts.html"
REGEN_CMD = "uv run python docs/gen_contracts.py"

# Source links point at the repo the docs site names (mkdocs repo_url), so the
# cards never drift from where the site says the code lives.
_REPO_URL = re.search(
    r"^repo_url:\s*(\S+)", (_ROOT / "docs" / "mkdocs.yml").read_text(encoding="utf-8"), re.M
)
_BLOB = (_REPO_URL.group(1).rstrip("/") if _REPO_URL else "") + "/blob/master"


def _module_href(dotted: str) -> str:
    return f"{_BLOB}/src/{dotted.replace('.', '/')}.py"


# Inline the primary shape's own field table when it fits a glance; a bigger
# shape gets only the source link (the module IS the reference).
_MAX_INLINE_FIELDS = 24


# --- the registry: the only declared inputs (everything else is derived) -----


@dataclass(frozen=True)
class Contract:
    """One data contract. ``module`` and ``pins``/``writers`` are the judgement
    inputs a scan cannot honestly derive; ``title`` and ``primary`` name the card
    and which classes' docstrings drive the invariant. Adding another contract is
    one more entry."""

    title: str
    module: str  # dotted, e.g. "agent6.sessions.manifest"
    primary: tuple[str, ...]  # contract class/alias name(s) whose docstrings are lifted
    writers: tuple[str, ...]  # who CONSTRUCTS it (not import-derivable), src-relative posix
    pins: tuple[
        str, ...
    ]  # byte/behaviour guards (test files and/or golden fixtures), repo-relative


CONTRACTS: tuple[Contract, ...] = (
    Contract(
        title="Conversation",
        module="agent6.workflows._conversation",
        primary=("Conversation",),
        writers=("workflows/loop.py",),
        pins=("tests/unit/data/golden_loop_wire.json",),
    ),
    Contract(
        title="SessionManifest",
        module="agent6.sessions.manifest",
        primary=("SessionManifest",),
        writers=("app/manifest.py",),
        pins=("tests/unit/test_sessions_manifest.py",),
    ),
    Contract(
        title="SessionSnapshot",
        module="agent6.workflows._session_state",
        primary=("SessionSnapshot",),
        writers=("workflows/loop.py",),
        pins=("tests/unit/data/golden_loop_wire.json",),
    ),
    Contract(
        title="ToolResult family",
        module="agent6.tools.results",
        primary=("ToolResult",),
        writers=(
            "tools/_control_tools.py",
            "tools/_dag_tools.py",
            "tools/_edit_diag.py",
            "tools/_fs_tools.py",
            "tools/_nav_tools.py",
            "tools/_skill_tools.py",
            "tools/dispatch.py",
        ),
        pins=("tests/unit/test_tool_result_wire.py", "tests/unit/test_tool_result_summaries.py"),
    ),
    Contract(
        title="Event union",
        module="agent6.viewmodel.events",
        primary=("Event",),
        # parse_event constructs the union (the raw EventSink writes dicts;
        # the typed shape exists only on the read side).
        writers=("viewmodel/events.py",),
        pins=("tests/unit/data/golden_session_logs.jsonl",),
    ),
    Contract(
        title="MachineSpec",
        module="agent6.machine.model",
        primary=("MachineSpec",),
        # load_machine constructs it from the .asm.toml at the parse boundary.
        writers=("machine/_semantics.py",),
        pins=("tests/unit/test_machine_model.py",),
    ),
    Contract(
        title="JournalEvent",
        module="agent6.machine.journal",
        primary=("JournalEvent",),
        # engine.py builds step/notify/end events; journal.begin builds the
        # begin event.
        writers=("machine/engine.py", "machine/journal.py"),
        pins=("tests/unit/data/golden_journal.jsonl",),
    ),
    Contract(
        title="TaskNode",
        module="agent6.graph.models",
        primary=("TaskNode",),
        # curator mints new nodes; storage reconstructs them from the on-disk .md.
        writers=("graph/curator.py", "graph/storage.py"),
        pins=("tests/unit/test_graph_storage.py",),
    ),
    Contract(
        title="Run/machine wire snapshot",
        module="agent6.viewmodel.state",
        # The dict these builders return IS the payload `attach --json`, the web
        # page, and the SSE stream serialize; their docstrings state the contract.
        primary=("session_state_as_dict",),
        writers=("viewmodel/state.py", "viewmodel/machine_state.py"),
        pins=("tests/unit/data/golden_session_state.json", "tests/unit/test_viewmodel_state.py"),
    ),
)

# The module graph's column families, leftmost first: a tach module lands in the
# column naming its top-level package under agent6 (unplaced ones, including
# `agent6` itself and `_data`, sit with the leaves). A declared input like
# CONTRACTS; the nodes, edges, and row ordering are derived from tach.toml.
COLS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("leaves", ("types", "portable", "paths", "directive", "prompts", "secrets", "events",
                "budget", "init", "verify_infer", "git_ops")),
    ("state & data", ("config", "models", "runs", "memory", "skills", "graph")),
    ("exec & tools", ("sandbox", "tools", "providers")),
    ("engine", ("workflows", "machine")),
    ("read-model", ("viewmodel",)),
    ("composition", ("app",)),
    ("presentation", ("ui",)),
)  # fmt: skip


# --- AST + prose helpers -----------------------------------------------------

_ROLE = re.compile(
    r":[a-z]+:`([^`]+)`"
)  # RST role, e.g. :class:`SessionManifest` -> SessionManifest


def _norm(text: str) -> str:
    """A docstring fragment as one line of markdown-inline prose: collapse
    whitespace, strip RST roles, fold ``code`` to `code`."""
    return _ROLE.sub(r"\1", " ".join(text.split())).replace("``", "`")


def _first_para(doc: str | None) -> str:
    return _norm(doc.strip().split("\n\n", 1)[0]) if doc else ""


def _first_sentence(doc: str | None) -> str:
    para = _first_para(doc)
    cut = para.find(". ")
    return para[: cut + 1] if cut != -1 else para


def _base_name(node: ast.expr) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return ""


def _is_frozen_dataclass(node: ast.ClassDef) -> bool:
    for dec in node.decorator_list:
        if not (isinstance(dec, ast.Call) and _base_name(dec.func) == "dataclass"):
            continue
        return any(
            kw.arg == "frozen" and isinstance(kw.value, ast.Constant) and kw.value.value is True
            for kw in dec.keywords
        )
    return False


def _flatten_bitor(node: ast.expr) -> list[str]:
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
        return _flatten_bitor(node.left) + _flatten_bitor(node.right)
    return [node.id] if isinstance(node, ast.Name) else []


def _union_members(node: ast.expr) -> list[str]:
    """Member names of a module-level union alias, unwrapping the pydantic
    tagged-union form ``Annotated[A | B, Field(discriminator=...)]`` to the
    ``A | B`` inside (a bare ``A | B`` alias flows straight through)."""
    if (
        isinstance(node, ast.Subscript)
        and _base_name(node.value) == "Annotated"
        and isinstance(node.slice, ast.Tuple)
        and node.slice.elts
    ):
        node = node.slice.elts[0]
    return _flatten_bitor(node)


@dataclass(frozen=True)
class ModuleFacts:
    module_doc: str
    class_docs: dict[str, str | None]
    frozen: tuple[str, ...]
    pydantic: tuple[str, ...]
    subclasses: dict[str, tuple[str, ...]]  # base name -> its subclasses in this module
    unions: dict[str, tuple[str, ...]]  # alias name -> union member names
    class_fields: dict[str, tuple[tuple[str, str, str], ...]]  # name -> (field, type, default)


def _field_default(node: ast.AnnAssign) -> str:
    """The field's default as short source text: a pydantic ``Field(...)``
    unwraps to its ``default=`` (or ``factory``); no value means required."""
    value = node.value
    if value is None:
        return "required"
    if isinstance(value, ast.Call) and _base_name(value.func) == "Field":
        for kw in value.keywords:
            if kw.arg == "default":
                return ast.unparse(kw.value)
            if kw.arg == "default_factory":
                return "factory"
        return "required"
    return ast.unparse(value)


def _class_fields(node: ast.ClassDef) -> tuple[tuple[str, str, str], ...]:
    """The class's own ``(name, type, default)`` rows: annotated assignments,
    minus config/private/ClassVar machinery."""
    rows: list[tuple[str, str, str]] = []
    for stmt in node.body:
        if not isinstance(stmt, ast.AnnAssign) or not isinstance(stmt.target, ast.Name):
            continue
        name = stmt.target.id
        anno = ast.unparse(stmt.annotation)
        if name == "model_config" or name.startswith("_") or anno.startswith("ClassVar"):
            continue
        rows.append((name, anno, _field_default(stmt)))
    return tuple(rows)


def _module_facts(dotted: str) -> ModuleFacts:
    path = _SRC.joinpath(*dotted.split(".")[1:]).with_suffix(".py")
    tree = ast.parse(path.read_text(encoding="utf-8"))
    class_docs: dict[str, str | None] = {}
    frozen: list[str] = []
    pydantic: list[str] = []
    subclasses: dict[str, list[str]] = defaultdict(list)
    unions: dict[str, tuple[str, ...]] = {}
    class_fields: dict[str, tuple[tuple[str, str, str], ...]] = {}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            # A wire-form BUILDER (session_state_as_dict) can be a primary too: its
            # docstring states the payload contract a class cannot (the dict it
            # returns IS the frozen shape).
            class_docs[node.name] = ast.get_docstring(node)
        if isinstance(node, ast.ClassDef):
            class_docs[node.name] = ast.get_docstring(node)
            class_fields[node.name] = _class_fields(node)
            bases = [_base_name(b) for b in node.bases]
            if _is_frozen_dataclass(node):
                frozen.append(node.name)
            if "BaseModel" in bases:
                pydantic.append(node.name)
            for base in bases:
                subclasses[base].append(node.name)
        elif isinstance(node, ast.Assign):
            members = _union_members(node.value)
            for tgt in node.targets:
                if isinstance(tgt, ast.Name) and len(members) > 1:
                    unions[tgt.id] = tuple(members)
    return ModuleFacts(
        module_doc=ast.get_docstring(tree) or "",
        class_docs=class_docs,
        frozen=tuple(frozen),
        pydantic=tuple(pydantic),
        subclasses={k: tuple(v) for k, v in subclasses.items()},
        unions=unions,
        class_fields=class_fields,
    )


def _kind(facts: ModuleFacts, primary: str) -> str:
    """A one-line, AST-derived description of the contract's shape and size."""
    subs = facts.subclasses.get(primary, ())
    if subs:
        return f"abstract base + {len(subs)} frozen result types"
    if primary in facts.unions:
        return f"tagged union of {len(facts.unions[primary])} frozen families"
    if primary in facts.pydantic:
        nested = len(facts.pydantic) - 1
        return "pydantic model" + (f" + {nested} nested models" if nested else "")
    if primary in facts.frozen:
        return "frozen dataclass"
    if facts.frozen:  # a plain container over frozen parts, e.g. Conversation over its turns
        return f"mutable container + {len(facts.frozen)} frozen turn types"
    return "plain class"


# --- import / test scans -----------------------------------------------------


def _imported_dotted(tree: ast.Module) -> set[str]:
    """Every dotted module a file imports, including the ``from PKG import NAME``
    form as ``PKG.NAME`` (how viewmodel.events is pulled in)."""
    out: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            out.add(node.module)
            out.update(f"{node.module}.{a.name}" for a in node.names)
        elif isinstance(node, ast.Import):
            out.update(a.name for a in node.names)
    return out


def _reexports(root: Path) -> dict[str, str]:
    """``package.Name -> the module that defines it``, read off each package's
    ``__init__``. Most of the tree reads a contract through its facade
    (``from agent6.viewmodel import SessionSummary``), so without this the
    reader set is only the modules that name the defining module directly."""
    out: dict[str, str] = {}
    for init in sorted(root.rglob("__init__.py")):
        rel = init.parent.relative_to(root).as_posix()
        pkg = "agent6" if rel == "." else f"agent6.{rel.replace('/', '.')}"
        for node in ast.walk(ast.parse(init.read_text(encoding="utf-8"))):
            if not isinstance(node, ast.ImportFrom):
                continue
            if node.level:  # relative: resolve against this package
                target = f"{pkg}.{node.module}" if node.module else pkg
            elif node.module:
                target = node.module
            else:
                continue
            for alias in node.names:
                out[f"{pkg}.{alias.asname or alias.name}"] = target
    return out


_REEXPORTS = _reexports(_SRC)


def _scan(root: Path) -> dict[str, set[str]]:
    scanned: dict[str, set[str]] = {}
    for p in sorted(root.rglob("*.py")):
        dotted = _imported_dotted(ast.parse(p.read_text(encoding="utf-8")))
        dotted |= {_REEXPORTS[d] for d in dotted if d in _REEXPORTS}
        scanned[p.relative_to(root).as_posix()] = dotted
    return scanned


_SRC_IMPORTS = _scan(_SRC)
_TEST_IMPORTS = _scan(_TESTS)


def _importers(dotted: str, imports: dict[str, set[str]]) -> list[str]:
    return sorted(rel for rel, mods in imports.items() if dotted in mods)


def _guard_tests(contract: Contract) -> list[str]:
    """Tests that import the contract module, reference a declared golden fixture,
    or are a declared pin test -- the full guarding set, byte pins included."""
    guards = {Path(p).name for p in contract.pins if p.endswith(".py")}
    goldens = [Path(p).name for p in contract.pins if not p.endswith(".py")]
    for rel, mods in _TEST_IMPORTS.items():
        if contract.module in mods or (
            goldens and any(g in (_TESTS / rel).read_text(encoding="utf-8") for g in goldens)
        ):
            guards.add(Path(rel).name)
    return sorted(guards)


# --- derived model per contract ----------------------------------------------


@dataclass(frozen=True)
class Card:
    title: str
    module: str
    kind: str
    invariant: str
    shapes: tuple[
        tuple[str, str], ...
    ]  # (primary name, first-sentence) pairs that have a docstring
    fields: tuple[tuple[str, str, str], ...]  # the primary's own (name, type, default) rows
    members: tuple[str, ...]  # union members / result subclasses, when the primary is a family
    writers: tuple[str, ...]  # src-relative posix
    readers: tuple[str, ...]  # src-relative posix
    pins: tuple[tuple[str, str], ...]  # (display basename, repo-relative path)
    guard_count: int


def _derive(contract: Contract) -> Card:
    facts = _module_facts(contract.module)
    importers = _importers(contract.module, _SRC_IMPORTS)
    readers = tuple(r for r in importers if r not in contract.writers)
    shapes = tuple(
        (name, _first_sentence(facts.class_docs.get(name)))
        for name in contract.primary
        if facts.class_docs.get(name)
    )
    primary = contract.primary[0]
    fields = facts.class_fields.get(primary, ())
    members = facts.unions.get(primary, ()) or facts.subclasses.get(primary, ())
    if len(fields) > _MAX_INLINE_FIELDS or members:
        fields = ()  # a family's shape is its member list, not one field table
    return Card(
        title=contract.title,
        module=contract.module,
        kind=_kind(facts, primary),
        invariant=_first_para(facts.module_doc),
        shapes=shapes,
        fields=fields,
        members=members,
        writers=contract.writers,
        readers=readers,
        pins=tuple((Path(p).name, p) for p in contract.pins),
        guard_count=len(_guard_tests(contract)),
    )


def _group(paths: tuple[str, ...]) -> str:
    """`workflows/loop.py`, `app/merge.py`, `app/run.py` -> `app/{merge, run},
    workflows/loop`: one package per part, braces only where they group."""
    by_dir: dict[str, list[str]] = defaultdict(list)
    for p in paths:
        parent = str(Path(p).parent)
        by_dir[parent].append(Path(p).stem)
    parts = []
    for parent in sorted(by_dir):
        stems = sorted(by_dir[parent])
        joined = stems[0] if len(stems) == 1 else "{" + ", ".join(stems) + "}"
        parts.append(joined if parent == "." else f"{parent}/{joined}")
    return ", ".join(parts)


def _guard_line_md(card: Card) -> str:
    links = ", ".join(f"[{name}]({_BLOB}/{rel})" for name, rel in card.pins)
    return f"{links} ({card.guard_count} test files exercise it)"


def _guard_line_text(card: Card) -> str:
    return f"{', '.join(name for name, _ in card.pins)} ({card.guard_count} test files exercise it)"


# --- markdown output ---------------------------------------------------------


_MD_HEADER = f"""<!-- Generated by docs/gen_contracts.py; do not edit by hand.
Edit the contract modules' docstrings or the CONTRACTS registry, then run:
    {REGEN_CMD}
tests/unit/test_data_contracts_doc.py fails if this file drifts. -->

# Data contracts

Each of the {len(CONTRACTS)} typed contracts owns a set of facts, with one writer set, a known reader set, and byte-level pins over the frozen surface.
Every card is generated from the module and class docstrings, so edit those rather than this page.
"""


def _md_fields(card: Card) -> list[str]:
    if not card.fields:
        return []

    def cell(text: str) -> str:
        return f"`{text}`"

    rows = [
        "",
        "| field | type | default |",
        "| --- | --- | --- |",
    ]
    rows += [
        f"| {cell(name)} | {cell(ftype)} |"
        f" {'required' if default == 'required' else cell(default)} |"
        for name, ftype, default in card.fields
    ]
    return rows


def _md_card(card: Card) -> str:
    lines = [
        f"## {card.title}",
        "",
        f"[`{card.module}`]({_module_href(card.module)}) &middot; {card.kind}",
        "",
        card.invariant,
    ]
    for name, sentence in card.shapes:
        lines += ["", f"**{name}** &mdash; {sentence}"]
    if card.members:
        lines += ["", "Members: " + ", ".join(f"`{m}`" for m in card.members)]
    lines += _md_fields(card)
    lines += [
        "",
        f"- **Written by:** {_group(card.writers)}",
        f"- **Read by:** {_group(card.readers)}",
        f"- **Guarded by:** {_guard_line_md(card)}",
    ]
    return "\n".join(lines)


def build_markdown() -> str:
    body = "\n\n".join(_md_card(_derive(c)) for c in CONTRACTS)
    return f"{_MD_HEADER}\n{body}\n"


# --- the module graph (derived from tach.toml) ---------------------------------


def _tach_modules() -> dict[str, tuple[str, ...]]:
    data = tomllib.loads((_ROOT / "tach.toml").read_text(encoding="utf-8"))
    return {
        m["path"]: tuple(d if isinstance(d, str) else d["path"] for d in m.get("depends_on", ()))
        for m in data["modules"]
    }


def _col_of(module: str) -> int:
    root = module.removeprefix("agent6.").split(".")[0]
    return next((i for i, (_, roots) in enumerate(COLS) if root in roots), 0)


def _module_graph_svg() -> tuple[str, int, int, int]:
    """The tach.toml dependency graph as a layered SVG: columns from COLS, rows
    ordered by six barycenter passes (each column sorts by the mean position of
    its neighbours) to minimize edge crossings. Returns the svg plus the derived
    (modules, edges, upward-edge) counts for the header line."""
    mods = _tach_modules()
    colidx = {m: _col_of(m) for m in mods}
    edges = [(s, d) for s, deps in mods.items() for d in deps if d in mods]
    order: dict[int, list[str]] = {c: [] for c in range(len(COLS))}
    for m in sorted(mods):
        order[colidx[m]].append(m)
    pos = {m: i for column in order.values() for i, m in enumerate(column)}

    def bary(m: str) -> float:
        near = [pos[d] for s, d in edges if s == m] + [pos[s] for s, d in edges if d == m]
        return sum(near) / len(near) if near else pos[m]

    for _ in range(6):
        for column in order.values():
            column.sort(key=bary)
            for i, m in enumerate(column):
                pos[m] = i

    cw, xpad, nh, vg, top = 200, 64, 26, 10, 70
    width = len(COLS) * (cw + xpad) + xpad
    height = top + max(len(c) for c in order.values()) * (nh + vg) + 40

    def xy(m: str) -> tuple[int, int]:
        c = colidx[m]
        return xpad + c * (cw + xpad), top + order[c].index(m) * (nh + vg)

    out = [
        f'<svg id="modgraph" viewBox="0 0 {width} {height}" style="min-width:{width}px" '
        'xmlns="http://www.w3.org/2000/svg" font-family="ui-monospace,Menlo,monospace">'
    ]
    for i, (name, _) in enumerate(COLS):
        x = xpad + i * (cw + xpad) + cw / 2
        out.append(
            f'<text x="{x}" y="34" text-anchor="middle" class="colhead">{html.escape(name)}</text>'
        )
        out.append(f'<line x1="{x}" y1="46" x2="{x}" y2="{height - 20}" class="colline"/>')
    out.append('<g id="edges">')
    for s, d in edges:
        (sx, sy), (dx, dy) = xy(s), xy(d)
        y1, y2 = sy + nh / 2, dy + nh / 2
        up = colidx[d] > colidx[s]  # a dependency pointing right, against the layering
        if colidx[d] == colidx[s]:  # same column: a short loop off the right edge
            x = sx + cw
            path = f"M {x} {y1} C {x + 34} {y1}, {x + 34} {y2}, {x} {y2}"
        else:
            x1, x2 = (sx + cw, dx) if up else (sx, dx + cw)
            mx = (x1 + x2) / 2
            path = f"M {x1} {y1} C {mx} {y1}, {mx} {y2}, {x2} {y2}"
        cls = "edge up" if up else "edge"
        out.append(f'<path class="{cls}" data-s="{s}" data-d="{d}" d="{path}"/>')
    out.append('</g><g id="nodes">')
    for m in sorted(mods):
        x, y = xy(m)
        fan_out = sum(1 for s, _ in edges if s == m)
        fan_in = sum(1 for _, d in edges if d == m)
        out.append(
            f'<g class="node g{colidx[m]}" data-id="{m}" tabindex="0">'
            f'<rect x="{x}" y="{y}" rx="5" width="{cw}" height="{nh}"/>'
            f'<text x="{x + 10}" y="{y + 17.5}">{html.escape(m.removeprefix("agent6."))}</text>'
            f'<text x="{x + cw - 8}" y="{y + 17.5}" text-anchor="end" class="deg">'
            f"{fan_out}&#8594; &#8592;{fan_in}</text></g>"
        )
    out.append("</g></svg>")
    upward = sum(1 for s, d in edges if colidx[d] > colidx[s])
    return "\n".join(out), len(mods), len(edges), upward


# --- HTML artifact -----------------------------------------------------------

_HTML_STYLE = """
:root{--surface:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--muted:#8a8984;--node:#eef1f5;--nodeline:#c9cdd4;--edge:#b9bcc4;--hi:#2a78d6;--up:#e34948;--g5:#1baf7a;--g6:#2a78d6}
@media(prefers-color-scheme:dark){:root{--surface:#1a1a19;--ink:#fff;--ink2:#c3c2b7;--muted:#87867e;--node:#26262a;--nodeline:#43434a;--edge:#4a4a52;--hi:#3987e5;--up:#e66767;--g5:#199e70;--g6:#3987e5}}
:root[data-theme=dark]{--surface:#1a1a19;--ink:#fff;--ink2:#c3c2b7;--muted:#87867e;--node:#26262a;--nodeline:#43434a;--edge:#4a4a52;--hi:#3987e5;--up:#e66767;--g5:#199e70;--g6:#3987e5}
:root[data-theme=light]{--surface:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--muted:#8a8984;--node:#eef1f5;--nodeline:#c9cdd4;--edge:#b9bcc4;--hi:#2a78d6;--up:#e34948;--g5:#1baf7a;--g6:#2a78d6}
body{background:var(--surface);color:var(--ink);margin:0;font:14px/1.45 system-ui,sans-serif}
header{padding:18px 24px 6px;display:flex;gap:18px;align-items:baseline;flex-wrap:wrap}
h1{font-size:16px;margin:0;font-weight:600}
h2{font-size:14px;margin:22px 24px 4px;font-weight:600}
.sub{color:var(--ink2);font-size:12.5px}
.note{color:var(--ink2);font-size:12.5px;margin:6px 24px 10px;max-width:74ch}
.legend{display:flex;gap:14px;font-size:12px;color:var(--ink2);align-items:center}
.k{display:inline-block;width:18px;height:3px;border-radius:2px;background:var(--edge);vertical-align:middle;margin-right:5px}.k.up{background:var(--up)}.k.hi{background:var(--hi)}
.wrap{overflow:auto;padding:0 12px 8px}.wrap svg{display:block}
.colhead{font:600 12px system-ui,sans-serif;fill:var(--ink2);letter-spacing:.06em;text-transform:uppercase}
.colline{stroke:var(--nodeline);stroke-width:.5;stroke-dasharray:2 6;opacity:.45}
.edge{fill:none;stroke:var(--edge);stroke-width:1;opacity:.42}.edge.up{stroke:var(--up);stroke-width:1.4;stroke-dasharray:5 3;opacity:.9}
.node rect{fill:var(--node);stroke:var(--nodeline);stroke-width:1}.node.g5 rect{stroke:var(--g5);stroke-width:1.6}.node.g6 rect{stroke:var(--g6);stroke-width:1.6}
.node text{font-size:12.5px;fill:var(--ink)}.node .deg{fill:var(--muted);font-size:10px}
svg.focus .edge{opacity:.06}svg.focus .edge.rel{opacity:.95;stroke:var(--hi);stroke-width:1.6}svg.focus .edge.rel.up{stroke:var(--up)}
svg.focus .node{opacity:.28}svg.focus .node.rel,svg.focus .node.self{opacity:1}
.node:focus{outline:none}.node:focus rect{stroke:var(--hi);stroke-width:2}
@media(prefers-reduced-motion:no-preference){.edge,.node{transition:opacity .12s ease}}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(340px,1fr));gap:12px;padding:4px 24px 26px}
.card{border:1px solid var(--nodeline);border-radius:8px;padding:12px 14px;background:var(--node)}
.cname{font-weight:600;font-size:13.5px}
.chome{font:12px ui-monospace,Menlo,monospace;color:var(--ink2);margin:2px 0 6px}
.cinv{font-size:12.5px;margin-bottom:6px}
.cshape{font-size:12px;color:var(--ink2);margin-bottom:6px}.cshape b{color:var(--ink)}
code{font:11.5px ui-monospace,Menlo,monospace;background:rgba(128,128,128,.14);padding:0 3px;border-radius:3px}
dl{margin:0;font-size:12px;display:grid;grid-template-columns:max-content 1fr;gap:2px 10px}
dt{color:var(--muted);text-transform:uppercase;letter-spacing:.05em;font-size:10.5px;padding-top:1px}
dd{margin:0;color:var(--ink2)}
.chome a{color:inherit}
.cfields{border-collapse:collapse;font-size:11.5px;margin:2px 0 8px;width:100%}
.cfields th{text-align:left;color:var(--muted);font-weight:500;text-transform:uppercase;font-size:10px;letter-spacing:.05em;padding:1px 8px 1px 0}
.cfields td{padding:1px 8px 1px 0;color:var(--ink2);vertical-align:top}
"""


# Hover/keyboard isolation for the module graph: entering a node dims everything
# but the node, its edges, and their endpoints (the CSS `svg.focus ... .rel/.self`
# contract). Self-contained; no external requests.
_HTML_SCRIPT = """
(() => {
  const svg = document.getElementById('modgraph');
  if (!svg) return;
  const edges = [...svg.querySelectorAll('.edge')];
  const nodes = [...svg.querySelectorAll('.node')];
  const clear = () => {
    svg.classList.remove('focus');
    edges.forEach(e => e.classList.remove('rel'));
    nodes.forEach(n => n.classList.remove('rel', 'self'));
  };
  for (const node of nodes) {
    const id = node.dataset.id;
    const isolate = () => {
      clear();
      svg.classList.add('focus');
      const rel = new Set();
      for (const e of edges) {
        if (e.dataset.s === id || e.dataset.d === id) {
          e.classList.add('rel');
          rel.add(e.dataset.s);
          rel.add(e.dataset.d);
        }
      }
      node.classList.add('self');
      for (const n of nodes) if (n !== node && rel.has(n.dataset.id)) n.classList.add('rel');
    };
    node.addEventListener('mouseenter', isolate);
    node.addEventListener('focus', isolate);
    node.addEventListener('mouseleave', clear);
    node.addEventListener('blur', clear);
  }
})();
"""


def _inline(text: str) -> str:
    """Markdown-inline prose -> safe HTML: escape, then `code` -> <code>."""
    return re.sub(r"`([^`]+)`", r"<code>\1</code>", html.escape(text))


def _html_card(card: Card) -> str:
    shapes = "".join(
        f'<div class="cshape"><b>{html.escape(name)}</b> &mdash; {_inline(sentence)}</div>'
        for name, sentence in card.shapes
    )
    members = (
        '<div class="cshape">members: '
        + ", ".join(f"<code>{html.escape(m)}</code>" for m in card.members)
        + "</div>"
        if card.members
        else ""
    )
    fields = ""
    if card.fields:
        rows = "".join(
            f"<tr><td><code>{html.escape(n)}</code></td><td><code>{html.escape(t)}</code></td>"
            f"<td>{'required' if d == 'required' else f'<code>{html.escape(d)}</code>'}</td></tr>"
            for n, t, d in card.fields
        )
        fields = (
            '<table class="cfields"><thead><tr><th>field</th><th>type</th><th>default</th>'
            f"</tr></thead><tbody>{rows}</tbody></table>"
        )
    return (
        '<div class="card">'
        f'<div class="cname">{html.escape(card.title)}</div>'
        f'<div class="chome"><a href="{html.escape(_module_href(card.module))}">'
        f"{html.escape(card.module)}</a> &middot; {html.escape(card.kind)}</div>"
        f'<div class="cinv">{_inline(card.invariant)}</div>'
        f"{shapes}{members}{fields}"
        "<dl>"
        f"<dt>written by</dt><dd>{_inline(_group(card.writers))}</dd>"
        f"<dt>read by</dt><dd>{_inline(_group(card.readers))}</dd>"
        f"<dt>guarded by</dt><dd>{_inline(_guard_line_text(card))}</dd>"
        "</dl></div>"
    )


def build_html() -> str:
    cards = "".join(_html_card(_derive(c)) for c in CONTRACTS)
    graph, n_modules, n_edges, n_upward = _module_graph_svg()
    return (
        "<!doctype html><html><head><meta charset=utf-8>"
        '<meta name=viewport content="width=device-width,initial-scale=1">'
        "<title>agent6 structure &amp; data contracts</title>"
        f"<style>{_HTML_STYLE}</style></head><body>"
        "<header><h1>agent6 structure</h1>"
        f'<span class="sub">derived from tach.toml &middot; modules: {n_modules} / '
        f"edges: {n_edges} / upward: {n_upward}</span>"
        '<span class="legend"><span><span class="k"></span>depends on (right&rarr;left)</span>'
        '<span><span class="k up"></span>upward edge</span>'
        '<span><span class="k hi"></span>hover a node to isolate</span></span></header>'
        f'<div class="wrap">{graph}</div>'
        "<h2>Data contracts</h2>"
        '<p class="note">Every card below is generated from the module and class '
        "docstrings, the AST, and an import scan &mdash; a card that reads badly means a "
        "docstring worth fixing.</p>"
        f'<div class="cards">{cards}</div>'
        f"<script>{_HTML_SCRIPT}</script></body></html>"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--md", type=Path, default=_MD_OUT)
    ap.add_argument("--html", type=Path, default=_HTML_OUT)
    args = ap.parse_args()
    args.md.write_text(build_markdown(), encoding="utf-8")
    args.html.parent.mkdir(parents=True, exist_ok=True)
    args.html.write_text(build_html(), encoding="utf-8")
    print(f"gen_contracts: wrote {args.md} and {args.html}")


if __name__ == "__main__":
    main()
