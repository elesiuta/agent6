# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""`agent6 machine check/test/graph`: the offline authoring gate for a machine
file (parse, semantics, bundle, script lint/types, mock tests, dry-run) and
its diagram render. No provider calls, no real network."""

from __future__ import annotations

import ast
import shutil
import sys
import tomllib
from pathlib import Path
from typing import Any

from agent6.app._setup import detect_env
from agent6.app.machine import lint_and_typecheck, run_offline_tests, validate_bundle
from agent6.config import ConfigError
from agent6.config.layer import load_effective_with_overlay
from agent6.errors import OperatorError, read_operator_file
from agent6.machine import (
    DryRunReport,
    MachineError,
    MachineSpec,
    ToolState,
    dry_run,
    fixture_problems,
    load_machine,
    render_dot,
    render_mermaid,
)
from agent6.sandbox.tool_paths import jail_search_path
from agent6.ui.cli._common import plural


def _fail(path: Path, problems: list[str], label: str = "") -> int:
    """Print a FAIL header + problem bullets to stderr; always returns 1."""
    suffix = f" ({label})" if label else ""
    print(f"FAIL: {path}{suffix}", file=sys.stderr)
    for problem in problems:
        print(f"  - {problem}", file=sys.stderr)
    return 1


def _load_validated(path: Path) -> tuple[MachineSpec | None, list[str], str]:
    """Shared `check`/`test` front half: load, structural bundle validation,
    and the effective-config overlay merge `machine run` performs -- so a bad
    `[config]` key fails here, not first at run.

    Returns (spec, problems, label). spec is None when validation failed;
    label names the failing stage for the FAIL header.
    """
    try:
        spec = load_machine(path)
    except MachineError as exc:
        return None, list(exc.problems), ""
    bundle_problems = validate_bundle(spec, path)
    if bundle_problems:
        return None, bundle_problems, "bundle"
    try:
        load_effective_with_overlay(Path.cwd(), spec.config)
    except ConfigError as exc:
        return None, [str(exc)], "config"
    return spec, [], ""


_SUBPROCESS_CALLS = frozenset({"run", "Popen", "call", "check_call", "check_output"})


def _script_binaries(scripts_dir: Path) -> dict[str, str]:
    """Best effort: the literal first-argv string of each subprocess call in the
    bundle's scripts, mapped to one script that makes it. Dynamic argv is
    invisible to this scan, so absence proves nothing; only a HIT feeds the
    reachability warning."""
    out: dict[str, str] = {}
    for py in sorted(scripts_dir.glob("*.py")) if scripts_dir.is_dir() else []:
        try:
            tree = ast.parse(py.read_text(encoding="utf-8"))
        except (OSError, SyntaxError):
            continue  # the script linter owns reporting unparseable scripts
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
            if name not in _SUBPROCESS_CALLS or not node.args:
                continue
            first_arg = node.args[0]
            if isinstance(first_arg, ast.List) and first_arg.elts:
                head = first_arg.elts[0]
                if isinstance(head, ast.Constant) and isinstance(head.value, str):
                    out.setdefault(head.value, py.name)
    return out


def _tool_reachability_warnings(spec: MachineSpec, path: Path) -> list[str]:
    """Binaries this machine will exec that do not resolve on the jail PATH:
    tool-state `command[0]` plus literal subprocess argv in bundle scripts.
    Offline validation mocks subprocess, so without this probe a machine passes
    check/test and dies on its first real state (observed: ruff, exit at
    transition 1). Advisory: the operator may install the tool later."""
    search = jail_search_path()
    sources: dict[str, str] = {}
    for name, state in spec.states.items():
        if isinstance(state, ToolState):
            sources.setdefault(state.command[0], f"[states.{name}] command")
    for binary, script in _script_binaries(path.parent / "scripts").items():
        sources.setdefault(binary, f"scripts/{script}")
    return [
        f"WARNING: `{binary}` ({src}) does not resolve on the jail PATH; that state"
        " will fail at run time. Install it into a standard bin dir"
        " (~/.local/bin, /usr/local/bin), or use an absolute path in the"
        " state's command."
        for binary, src in sorted(sources.items())
        if "/" not in binary and shutil.which(binary, path=search) is None
    ]


def _cmd_machine_check(path: Path) -> int:
    spec, problems, label = _load_validated(path)
    if spec is None:
        return _fail(path, problems, label)
    script_problems = lint_and_typecheck(path.parent / "scripts")
    if script_problems:
        return _fail(path, script_problems, "scripts")
    for warning in _tool_reachability_warnings(spec, path):
        print(warning, file=sys.stderr)
    for name, state in spec.states.items():
        if isinstance(state, ToolState) and state.pass_env:
            print(
                f"[states.{name}] receives the environment variable(s)"
                f" {', '.join(state.pass_env)} when [machine].pass_env allows them"
            )
    print(f"OK: {path} ({spec.machine}, {len(spec.states)} states)")
    return 0


def _cmd_machine_test(path: Path, *, blackboard: Path | None) -> int:
    # `machine test` is the offline simulation: `machine check`'s structural +
    # bundle validation, plus running the bundle's `*_test.py` mocks in a jail
    # (no network), plus a pure dry-run. Reuse the same load + bundle validation
    # so a malformed machine fails the same way.
    spec, problems, label = _load_validated(path)
    if spec is None:
        return _fail(path, problems, label)
    # Static (lint + types) then the offline mock tests in a no-network jail.
    script_problems = lint_and_typecheck(path.parent / "scripts")
    offline = run_offline_tests(path.parent, detect_env().detected_isolation)
    script_problems.extend(offline.problems)
    if script_problems:
        return _fail(path, script_problems, "scripts")
    fixture: dict[str, Any] | None = None
    if blackboard is not None:
        try:
            fixture = tomllib.loads(read_operator_file(blackboard))
        except tomllib.TOMLDecodeError as exc:
            raise OperatorError(f"blackboard fixture is not valid TOML: {exc}") from exc
        fixture_errors = fixture_problems(spec, fixture)
        if fixture_errors:
            return _fail(path, fixture_errors, "blackboard")
    report = dry_run(spec, fixture)
    _print_dry_run_report(spec, report)
    if report.ok:
        # A skip rides the verdict line, not a stderr aside: OK with tests
        # silently unrun read as "tests ran green".
        skipped = (
            f"; {plural(offline.skipped, 'offline script test')} NOT run ({offline.skip_reason})"
            if offline.skipped
            else ""
        )
        print(
            f"\nOK: {path} dry-run passed ({plural(len(report.states), 'state')}, "
            f"{plural(len(report.branches), 'branch', 'branches')}){skipped}"
        )
        return 0
    print(f"\nFAIL: {path} dry-run found problems", file=sys.stderr)
    return 1


def _print_dry_run_report(spec: MachineSpec, report: DryRunReport) -> None:
    """Render the per-state and per-branch dry-run tables."""
    mark = {True: "ok", False: "FAIL"}
    print(f"machine {spec.machine!r}: per-state dry-run")
    print(f"  {'STATE':<16} {'KIND':<9} {'->LABEL':<9} {'GOTO':<14} STATUS  DETAIL")
    for s in report.states:
        print(
            f"  {s.name:<16} {s.kind:<9} {(s.label or '-'):<9} {(s.goto or '-'):<14}"
            f" {mark[s.ok]:<6}  {s.detail}"
        )
    if report.branches:
        print("\nper-branch routing (fixture overlaid on defaults)")
        print(f"  {'STATE':<16} {'CLAUSE':<7} {'GOTO':<14} STATUS  PREDICATE")
        for b in report.branches:
            clause = "-" if b.clause_index is None else f"[{b.clause_index}]"
            pred = b.detail if not b.ok else (b.predicate or "")
            print(f"  {b.name:<16} {clause:<7} {(b.goto or '-'):<14} {mark[b.ok]:<6}  {pred}")


def _cmd_machine_graph(path: Path, *, fmt: str) -> int:
    try:
        spec = load_machine(path)
    except MachineError as exc:
        return _fail(path, exc.problems)
    print((render_dot if fmt == "dot" else render_mermaid)(spec), end="")
    return 0
