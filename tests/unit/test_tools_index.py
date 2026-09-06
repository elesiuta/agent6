# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Unit tests for `agent6.tools.index.SymbolIndex`."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from agent6.tools._path_safety import Workspace
from agent6.tools.index import Reference, Symbol, SymbolIndex

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def py_project(tmp_path: Path) -> Path:
    (tmp_path / "a.py").write_text(
        textwrap.dedent(
            """\
            def foo(x):
                return x

            class Bar:
                def baz(self):
                    return foo(1)

            CONST = foo(2)
            """
        )
    )
    (tmp_path / "b.py").write_text(
        textwrap.dedent(
            """\
            from a import foo, Bar

            def caller():
                # foo is called here too
                return foo(Bar().baz())
            """
        )
    )
    # An excluded path that must not pollute the index.
    (tmp_path / ".venv").mkdir()
    (tmp_path / ".venv" / "ignored.py").write_text("def should_not_appear(): pass\n")
    return tmp_path


# ---------------------------------------------------------------------------
# outline()
# ---------------------------------------------------------------------------


def test_outline_python_top_level_and_nested(py_project: Path) -> None:
    idx = SymbolIndex(Workspace(root=py_project))
    syms = idx.outline(py_project / "a.py")
    by_name = {(s.name, s.kind): s for s in syms}
    assert ("foo", "function") in by_name
    assert ("Bar", "class") in by_name
    # nested method is captured by the function_definition rule
    assert ("baz", "function") in by_name
    # Source-order
    assert [s.name for s in syms] == ["foo", "Bar", "baz"]


def test_outline_returns_empty_for_unknown_extension(tmp_path: Path) -> None:
    (tmp_path / "x.md").write_text("# hello\n")
    idx = SymbolIndex(Workspace(root=tmp_path))
    assert idx.outline(tmp_path / "x.md") == []


def test_outline_returns_empty_for_missing_file(tmp_path: Path) -> None:
    idx = SymbolIndex(Workspace(root=tmp_path))
    assert idx.outline(tmp_path / "nope.py") == []


# ---------------------------------------------------------------------------
# find_definition()
# ---------------------------------------------------------------------------


def test_find_definition_locates_single_def(py_project: Path) -> None:
    idx = SymbolIndex(Workspace(root=py_project))
    defs = idx.find_definition("Bar")
    assert len(defs) == 1
    assert defs[0].kind == "class"
    assert defs[0].path == (py_project / "a.py").resolve()
    assert defs[0].line == 4


def test_find_definition_returns_empty_for_unknown_name(py_project: Path) -> None:
    idx = SymbolIndex(Workspace(root=py_project))
    assert idx.find_definition("definitely_does_not_exist") == []


def test_find_definition_skips_excluded_dirs(py_project: Path) -> None:
    idx = SymbolIndex(Workspace(root=py_project))
    assert idx.find_definition("should_not_appear") == []


# ---------------------------------------------------------------------------
# find_references()
# ---------------------------------------------------------------------------


def test_find_references_filters_comments_and_strings(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text(
        textwrap.dedent(
            """\
            def foo():
                return 1

            # foo in a comment must not count
            s = "foo in a string must not count either"
            foo()
            """
        )
    )
    idx = SymbolIndex(Workspace(root=tmp_path))
    refs = idx.find_references("foo")
    # Two: definition site + the call.
    assert len(refs) == 2
    assert all(isinstance(r, Reference) for r in refs)
    lines = sorted(r.line for r in refs)
    assert lines == [1, 6]


def test_find_references_spans_files(py_project: Path) -> None:
    idx = SymbolIndex(Workspace(root=py_project))
    refs = idx.find_references("foo")
    paths = {r.path for r in refs}
    assert (py_project / "a.py").resolve() in paths
    assert (py_project / "b.py").resolve() in paths


# ---------------------------------------------------------------------------
# Incremental invalidation
# ---------------------------------------------------------------------------


def test_mark_changed_picks_up_new_symbols(py_project: Path) -> None:
    idx = SymbolIndex(Workspace(root=py_project))
    assert idx.find_definition("brand_new") == []
    target = py_project / "a.py"
    target.write_text(target.read_text() + "\ndef brand_new():\n    return 0\n")
    idx.mark_changed(target)
    defs = idx.find_definition("brand_new")
    assert len(defs) == 1
    assert defs[0].kind == "function"


def test_mark_deleted_drops_file_from_index(py_project: Path) -> None:
    idx = SymbolIndex(Workspace(root=py_project))
    # Prime the index
    assert idx.find_definition("Bar")
    (py_project / "a.py").unlink()
    idx.mark_deleted(py_project / "a.py")
    assert idx.find_definition("Bar") == []


def test_mark_changed_on_deleted_file_silently_removes(py_project: Path) -> None:
    idx = SymbolIndex(Workspace(root=py_project))
    assert idx.find_definition("Bar")
    target = py_project / "a.py"
    target.unlink()
    # Caller uses mark_changed for both edits and deletes by mistake; the
    # next refresh should notice the file is gone and drop it.
    idx.mark_changed(target)
    assert idx.find_definition("Bar") == []


def test_lazy_initial_scan_is_deferred_until_first_query(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("def x(): pass\n")
    idx = SymbolIndex(Workspace(root=tmp_path))
    # Before any query, internal caches are empty.
    assert not idx._symbols  # pyright: ignore[reportPrivateUsage]
    idx.outline(tmp_path / "a.py")
    assert idx._symbols  # pyright: ignore[reportPrivateUsage]


# ---------------------------------------------------------------------------
# Multi-language sanity
# ---------------------------------------------------------------------------


def test_rust_outline(tmp_path: Path) -> None:
    (tmp_path / "lib.rs").write_text(
        textwrap.dedent(
            """\
            pub fn greet(name: &str) -> String {
                format!("hi {name}")
            }

            pub struct Greeter {
                pub prefix: String,
            }

            pub trait Greet {
                fn greet(&self) -> String;
            }
            """
        )
    )
    idx = SymbolIndex(Workspace(root=tmp_path))
    syms = idx.outline(tmp_path / "lib.rs")
    by_kind = {(s.name, s.kind) for s in syms}
    assert ("greet", "function") in by_kind
    assert ("Greeter", "struct") in by_kind
    assert ("Greet", "trait") in by_kind


def test_typescript_outline(tmp_path: Path) -> None:
    (tmp_path / "a.ts").write_text(
        textwrap.dedent(
            """\
            export function greet(name: string): string {
                return `hi ${name}`;
            }

            export class Greeter {
                greet(name: string): string {
                    return greet(name);
                }
            }

            export interface Named {
                name: string;
            }

            export type Maybe<T> = T | null;
            """
        )
    )
    idx = SymbolIndex(Workspace(root=tmp_path))
    syms = idx.outline(tmp_path / "a.ts")
    by_kind = {(s.name, s.kind) for s in syms}
    assert ("greet", "function") in by_kind
    assert ("Greeter", "class") in by_kind
    assert ("greet", "method") in by_kind
    assert ("Named", "interface") in by_kind
    assert ("Maybe", "type") in by_kind


# ---------------------------------------------------------------------------
# Multi-language outline (tree-sitter-language-pack grammars)
# ---------------------------------------------------------------------------

# (filename, source, expected (name, kind) pairs the outline MUST contain).
_LANG_CASES: list[tuple[str, str, set[tuple[str, str]]]] = [
    (
        "s.go",
        "package m\ntype Point struct{ X int }\ntype Shape interface{ Area() float64 }\n"
        "const Max = 10\nfunc Add(a, b int) int { return a + b }\n"
        "func (p Point) Move(d int) {}\n",
        {
            ("Add", "function"),
            ("Move", "method"),
            ("Point", "struct"),
            ("Shape", "interface"),
            ("Max", "const"),
        },
    ),
    (
        "S.java",
        "class Foo {\n  int field;\n  Foo() {}\n  void doIt() {}\n}\n"
        "interface Bar { void run(); }\nenum Color { RED, GREEN }\n",
        {
            ("Foo", "class"),
            ("doIt", "method"),
            ("Bar", "interface"),
            ("Color", "enum"),
            ("RED", "const"),
        },
    ),
    (
        "s.js",
        "function top() {}\nclass Widget {\n  render() {}\n  #priv() {}\n}\n",
        {("top", "function"), ("Widget", "class"), ("render", "method")},
    ),
    (
        "s.c",
        "struct Pt { int x; };\ntypedef int MyInt;\n#define MAXN 8\n"
        "int add(int a, int b) { return a + b; }\n",
        {("add", "function"), ("Pt", "struct"), ("MyInt", "type"), ("MAXN", "const")},
    ),
    (
        "s.cpp",
        "namespace ns { class Animal {\npublic:\n  void speak();\n}; }\n"
        "void ns::Animal::speak() {}\nint freefn() { return 0; }\n",
        {("Animal", "class"), ("ns", "module"), ("freefn", "function"), ("speak", "method")},
    ),
    (
        "S.cs",
        "namespace App {\n  class Svc { public void Do() {} }\n  struct V {}\n"
        "  interface I {}\n  enum E { A }\n}\n",
        {
            ("App", "module"),
            ("Svc", "class"),
            ("Do", "method"),
            ("V", "struct"),
            ("I", "interface"),
            ("E", "enum"),
        },
    ),
    (
        "s.rb",
        "module M\n  class Foo\n    def bar; end\n    def name=(v); end\n  end\nend\n",
        {("M", "module"), ("Foo", "class"), ("bar", "method"), ("name=", "method")},
    ),
    (
        "s.php",
        "<?php\nfunction top() {}\nclass C { public function m() {} }\n"
        "interface I {}\ntrait T {}\n",
        {("top", "function"), ("C", "class"), ("m", "method"), ("I", "interface"), ("T", "trait")},
    ),
]


@pytest.mark.parametrize(
    "filename,source,expected", _LANG_CASES, ids=lambda v: v if isinstance(v, str) else ""
)
def test_outline_supported_languages(
    tmp_path: Path, filename: str, source: str, expected: set[tuple[str, str]]
) -> None:
    f = tmp_path / filename
    f.write_text(source, encoding="utf-8")
    idx = SymbolIndex(Workspace(root=tmp_path))
    got = {(s.name, s.kind) for s in idx.outline(f)}
    assert expected <= got, f"{filename}: missing {expected - got} (got {got})"


def test_outline_c_header_uses_cpp_grammar(tmp_path: Path) -> None:
    # .h maps to the cpp grammar (a C superset), so a C++ class in a header is
    # captured -- the plain C grammar would miss it.
    f = tmp_path / "widget.h"
    f.write_text("class Widget {\npublic:\n  void draw();\n};\n", encoding="utf-8")
    idx = SymbolIndex(Workspace(root=tmp_path))
    got = {(s.name, s.kind) for s in idx.outline(f)}
    assert ("Widget", "class") in got


def test_cpp_skips_forward_decls_and_param_type_uses(tmp_path: Path) -> None:
    # The cpp class/struct/enum queries require a body, so a forward declaration
    # and a bare type-use in a parameter are NOT captured as definitions (which
    # would pollute find_definition with phantoms + duplicates).
    f = tmp_path / "t.hpp"
    f.write_text(
        "class Conn;\nvoid f(Conn *c, struct Pt *p) {}\nclass Conn { int x; };\n",
        encoding="utf-8",
    )
    idx = SymbolIndex(Workspace(root=tmp_path))
    defs = idx.find_definition("Conn")
    assert len(defs) == 1 and defs[0].kind == "class"  # real def only, no forward-decl dup
    assert idx.find_definition("Pt") == []  # bare param type is not a definition


def test_c_header_prototypes_are_functions_not_methods(tmp_path: Path) -> None:
    # A plain-C prototype in a .h (cpp grammar) is a free function, not a method.
    f = tmp_path / "api.h"
    f.write_text("int conn_open(const char *s, int n);\nvoid conn_close(void);\n", encoding="utf-8")
    idx = SymbolIndex(Workspace(root=tmp_path))
    got = {(s.name, s.kind) for s in idx.outline(f)}
    assert ("conn_open", "function") in got and ("conn_close", "function") in got
    assert not any(k == "method" for _, k in got)


def test_csharp_operator_overload_name_is_the_symbol(tmp_path: Path) -> None:
    # An operator overload is named by its operator token (+, ==), not "operator".
    f = tmp_path / "V.cs"
    f.write_text(
        "class V { public static V operator +(V a, V b){return a;}"
        " public static bool operator ==(V a, V b){return true;} }",
        encoding="utf-8",
    )
    idx = SymbolIndex(Workspace(root=tmp_path))
    got = {(s.name, s.kind) for s in idx.outline(f)}
    assert ("+", "method") in got and ("==", "method") in got
    assert ("operator", "method") not in got


def test_find_references_go_filters_to_identifiers(tmp_path: Path) -> None:
    # Cross-language sanity: references use the per-language ref query, so a Go
    # symbol is found by find_references (and never inside strings/comments).
    f = tmp_path / "m.go"
    f.write_text(
        "package m\nfunc helper() {}\nfunc use() { helper() }\n// helper\n", encoding="utf-8"
    )
    idx = SymbolIndex(Workspace(root=tmp_path))
    refs = idx.find_references("helper")
    # definition + call site, but NOT the comment mention.
    assert len(refs) == 2


# ---------------------------------------------------------------------------
# Symbol value type
# ---------------------------------------------------------------------------


def test_symbol_is_frozen_dataclass() -> None:
    s = Symbol(name="x", kind="function", path=Path("/tmp/x.py"), line=0, col=0)
    with pytest.raises(Exception):  # FrozenInstanceError or AttributeError
        s.name = "y"  # type: ignore[misc]


def test_outline_names_why_it_has_no_symbols(tmp_path: Path) -> None:
    """A file no grammar covers, or one outside the indexed workspace (a
    granted read path), answered `symbols: []`: the one thing the model acts
    on, stated wrongly. Each is refused with its cause."""
    from agent6.config import Config
    from agent6.tools.dispatch import ToolDispatcher, ToolError

    (tmp_path / "notes.md").write_text("# heading\n", encoding="utf-8")
    d = ToolDispatcher(root=tmp_path, config=Config())
    with pytest.raises(ToolError, match=r"no parser for \.md"):
        d.dispatch("outline", {"path": "notes.md"})
    granted = tmp_path.parent / (tmp_path.name + "-granted")
    granted.mkdir()
    (granted / "g.py").write_text("def gamma():\n    pass\n", encoding="utf-8")
    cfg = Config.model_validate({"sandbox": {"extra_read_paths": [str(granted)]}})
    d2 = ToolDispatcher(root=tmp_path, config=cfg)
    with pytest.raises(ToolError, match="outside the indexed workspace"):
        d2.dispatch("outline", {"path": str(granted / "g.py")})
