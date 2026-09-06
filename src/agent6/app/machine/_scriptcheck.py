# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Validate the helper scripts `machine create` generates so a committed bundle
is production-ready: lint-clean, typed, and proven to *simulate* offline.

Two layers, matching their risk:

* :func:`lint_and_typecheck`, STATIC analysis only (ruff + ty read the files,
  they never run them), so it shells out directly with a fixed argv. ruff runs
  on the real files, so its own config discovery applies: the nearest config
  above the machine file (the bundle's own pyproject.toml/ruff.toml, else the
  repo's) pins the lint rules. ty has no config-isolation flag, so it checks a
  private temp copy; mock-heavy `*_test.py` files trip it on `unittest.mock`
  internals, so they are gated by *execution* instead.
* :func:`run_offline_tests`, EXECUTES each `*_test.py`. Because that runs
  model-authored code, it goes through :func:`run_in_jail` (no network, the same
  confinement a tool state gets), never a bare subprocess.

A missing ruff/ty is skipped silently (a stripped install still produces a
bundle). An unavailable jail is different: it surfaces a diagnostic rather than
silently dropping the offline-test gate, except on isolation `none`, where
there is no jail to run model-authored code in and execution is skipped.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import tomllib
from dataclasses import dataclass
from pathlib import Path

from agent6.sandbox import run_in_jail
from agent6.sandbox.jail import JailUnavailableError
from agent6.types import IsolationLevel, JailPolicy

__all__ = ["OfflineTestOutcome", "available_tools", "lint_and_typecheck", "run_offline_tests"]

_TEST_SUFFIX = "_test.py"
_MAX_DIAG_LINES = 30


def _resolve_tool(name: str) -> list[str] | None:
    """Locate a bundled dev tool (`ruff` / `ty`) as an argv prefix.

    Prefer the console script installed next to the running interpreter (the
    runtime dependency), then anything on `PATH`, then a self-contained
    `uvx <name>`. `None` if the tool can't be found at all (skip it)."""
    local = Path(sys.executable).parent / name
    if local.is_file():
        return [str(local)]
    on_path = shutil.which(name)
    if on_path:
        return [on_path]
    uvx = shutil.which("uvx")
    if uvx:
        return [uvx, name]
    return None


def available_tools() -> list[str]:
    """Which of ruff/ty resolve in this environment (for a 'skipped' note)."""
    return [name for name in ("ruff", "ty") if _resolve_tool(name) is not None]


def _trim(text: str) -> str:
    lines = text.splitlines()
    if len(lines) <= _MAX_DIAG_LINES:
        return text.strip()
    kept = lines[:_MAX_DIAG_LINES]
    return "\n".join(kept).strip() + f"\n... ({len(lines) - _MAX_DIAG_LINES} more lines)"


def _run_static(argv: list[str], cwd: Path, label: str) -> str | None:
    """Run a static checker; return a problem string on failure, else None."""
    # Fixed argv (an operator-installed tool + flags); the only LLM-derived input
    # is the *files* it statically reads, it never executes them. See AGENTS.md.
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    try:
        res = subprocess.run(
            argv, capture_output=True, text=True, timeout=180, cwd=cwd, check=False, env=env
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return f"{label} could not run ({exc})"
    if res.returncode == 0:
        return None
    out = (res.stdout + ("\n" + res.stderr if res.stderr else "")).strip()
    # Diagnostics name the private temp copy; relativize so they read as bundle
    # paths (scripts/...), mirroring the run_offline_tests cleanup.
    out = out.replace(str(cwd.resolve()) + "/", "").replace(str(cwd) + "/", "")
    return f"{label} found problems:\n{_trim(out)}"


def _nearest_ruff_config(start: Path) -> Path | None:
    """The ruff config governing *start*: the nearest `.ruff.toml`, `ruff.toml`,
    or `pyproject.toml` with a `[tool.ruff]` table, walking up. Mirrors ruff's
    own discovery, for the one caller whose files live outside the tree their
    config governs (machine create's scratch bundle)."""
    base = start.resolve()
    for directory in (base, *base.parents):
        for name in (".ruff.toml", "ruff.toml", "pyproject.toml"):
            candidate = directory / name
            if not candidate.is_file():
                continue
            if name == "pyproject.toml":
                try:
                    data = tomllib.loads(candidate.read_text(encoding="utf-8"))
                except (OSError, tomllib.TOMLDecodeError):
                    continue
                if "ruff" not in data.get("tool", {}):
                    continue
            return candidate
    return None


def lint_and_typecheck(
    scripts_dir: Path, *, fix: bool = False, ruff_config_from: Path | None = None
) -> list[str]:
    """Lint (ruff) and type-check (ty) the bundle's Python scripts, no execution.

    Returns human-readable problems (empty = clean / tools absent).
    `*_test.py` files are linted but not type-checked.

    ruff runs on the real files with its own config discovery, so the nearest
    config above the machine file pins the rules: the bundle's own
    pyproject.toml/ruff.toml, else the repo's, else ruff's defaults.
    `ruff_config_from` (machine create only) resolves that config from the
    publish destination instead: the scratch bundle lives under the state dir,
    where discovery would find nothing, and the draft gate must agree with the
    `machine check` the published bundle faces. `--no-cache` keeps the
    operator-facing verbs write-free (no `.ruff_cache`).

    `fix=True` (machine create only, on its OWN generated bundle) applies
    ruff's safe fixes in place and reports only what remains: a whole
    authoring attempt burned on fixable lint otherwise. Operator-facing verbs
    (`machine check`/`test`) never fix -- a check must not mutate the
    operator's files."""
    if not scripts_dir.is_dir() or not any(scripts_dir.rglob("*.py")):
        return []
    problems: list[str] = []
    if ruff := _resolve_tool("ruff"):
        argv = [*ruff, "check", "--no-cache", "--output-format", "concise"]
        if ruff_config_from is not None:
            found = _nearest_ruff_config(ruff_config_from)
            argv += ["--config", str(found)] if found is not None else ["--isolated"]
        if fix:
            argv.append("--fix")
        bundle_dir = scripts_dir.resolve().parent
        problem = _run_static([*argv, scripts_dir.name], bundle_dir, "ruff (lint)")
        if problem:
            problems.append(problem)
    else:
        print("note: ruff not installed; script lint skipped", file=sys.stderr)
    if ty := _resolve_tool("ty"):
        real = sorted(p for p in scripts_dir.rglob("*.py") if not p.name.endswith(_TEST_SUFFIX))
        if real:
            # ty has no config-isolation flag and walks up from the checked
            # files to the nearest pyproject.toml, which could pull in a stray
            # config, so it checks a private temp copy.
            work = Path(tempfile.mkdtemp(prefix="agent6-scriptcheck-"))
            try:
                dst = work / "scripts"
                shutil.copytree(scripts_dir, dst, symlinks=True)
                copies = [str(dst / p.relative_to(scripts_dir)) for p in real]
                problem = _run_static([*ty, "check", *copies], work, "ty (type check)")
                if problem:
                    problems.append(problem)
            finally:
                shutil.rmtree(work, ignore_errors=True)
    else:
        print("note: ty not installed; script type check skipped", file=sys.stderr)
    return problems


@dataclass(frozen=True, slots=True)
class OfflineTestOutcome:
    """`run_offline_tests`' verdict: failures, plus what could NOT run.

    `skipped`/`skip_reason` ride to the caller's own verdict surface -- a
    skip buried in stderr while the verdict read OK looked like tests ran
    green."""

    problems: tuple[str, ...] = ()
    skipped: int = 0
    skip_reason: str = ""


def run_offline_tests(
    bundle_dir: Path, isolation: IsolationLevel, *, timeout_s: float = 30.0
) -> OfflineTestOutcome:
    """Execute every `scripts/**/*_test.py` in a no-network jail (the bundle's
    offline simulation).

    Requires the strict isolation: it is the only one whose network namespace can
    enforce the no-network contract on model-authored code. On `none` (no jail
    at all) and `hardened` (a jail, but no network namespace, so
    `network="none"` cannot be honored and the scripts would reach the host
    network) the tests are counted as skipped with the reason, for the caller
    to render on its verdict; the static checks still apply. Each test gets a
    fresh writable `$AGENT6_MACHINE_DATA_DIR` so record-style scripts can be
    exercised. Tests run under the default `JailPolicy` memory cap (these are
    offline mocks; the operator's `[sandbox].memory_limit_mb` is not
    consulted)."""
    scripts_dir = bundle_dir / "scripts"
    if not scripts_dir.is_dir():
        return OfflineTestOutcome()
    if not sorted(scripts_dir.rglob(f"*{_TEST_SUFFIX}")):
        return OfflineTestOutcome()
    # Run against a private temp COPY, like lint_and_typecheck: the real
    # bundle lives under the per-repo state dir, which the jail masks as a
    # private path, so tests run in place saw an empty tree (python3: can't
    # open file) or the launcher failed rootfs setup outright.
    workdir = Path(tempfile.mkdtemp(prefix="agent6-scripttest-"))
    try:
        bundle_copy = workdir / "bundle"
        # The tests need the bundle, not its history: a drafting workspace is a
        # git repo, and copying `.git` per attempt copies every draft it holds.
        shutil.copytree(
            bundle_dir, bundle_copy, symlinks=True, ignore=shutil.ignore_patterns(".git")
        )
        return _run_offline_tests_in(bundle_copy, isolation, timeout_s=timeout_s)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def _run_offline_tests_in(
    bundle_dir: Path, isolation: IsolationLevel, *, timeout_s: float
) -> OfflineTestOutcome:
    scripts_dir = bundle_dir / "scripts"
    tests = sorted(scripts_dir.rglob(f"*{_TEST_SUFFIX}"))
    if isolation != "strict":
        # none: no jail to confine model-authored code in. hardened: a jail, but
        # no network namespace, so network="none" cannot be honored and the
        # scripts would run with the host network -- exfil or pull-and-exec of
        # model-authored code during `machine create`. Only strict can honor
        # the no-network contract; skipping is the only safe option on the rest.
        reason = "no sandbox" if isolation == "none" else "no network isolation (hardened)"
        return OfflineTestOutcome(skipped=len(tests), skip_reason=reason)
    data_dir = bundle_dir / ".scriptcheck_data"
    problems: list[str] = []
    try:
        for test in tests:
            # Fresh per test, as promised: state a record-style script leaves
            # behind must not leak into the next test's run.
            shutil.rmtree(data_dir, ignore_errors=True)
            data_dir.mkdir(parents=True)
            rel = test.relative_to(bundle_dir).as_posix()
            policy = JailPolicy(
                cwd=bundle_dir,
                argv=("python3", rel),
                isolation=isolation,
                env=(
                    ("AGENT6_MACHINE_DATA_DIR", ".scriptcheck_data"),
                    ("PYTHONDONTWRITEBYTECODE", "1"),
                ),
                network="none",
                extra_rw_paths=(data_dir,),
                timeout_s=timeout_s,
            )
            try:
                res = run_in_jail(policy)
            except JailUnavailableError as exc:
                # The jail is a prerequisite for ANY test here, so fail fast on
                # the first unavailability rather than repeating it per test.
                return OfflineTestOutcome(
                    problems=(
                        f"could not run offline tests in a jail ({exc});"
                        " static checks still applied",
                    )
                )
            if res.returncode != 0:
                detail = (res.stderr or res.stdout or "").strip()
                # Tracebacks name the absolute bundle dir. Relativize so the
                # diagnostic (which is fed back into the authoring prompt and
                # journaled) stays short and free of host paths.
                detail = detail.replace(str(bundle_dir.resolve()) + "/", "").replace(
                    str(bundle_dir) + "/", ""
                )
                problems.append(
                    f"offline test {rel} failed (exit {res.returncode}):\n{_trim(detail)}"
                )
    finally:
        shutil.rmtree(data_dir, ignore_errors=True)
    return OfflineTestOutcome(problems=tuple(problems))
