# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Infer a `verify_command` for a run when none is configured.

agent6's verify command is the success gate, but a brand-new user has not set
one. Rather than block the run, `agent6 run`/`plan`
infers one, cheapest source first:

  1. the `## Verify command` (or `## Test`) section of AGENTS.md, or an
     inline `Verify:`/`Test:` line -- explicit, human-authored intent;
  2. deterministic repo signals (a root `verify.sh`, package.json
     `scripts.test`, a Makefile `test`/`check` target, pyproject/pytest,
     Cargo, go.mod, loose `test_*.py` files);
  3. an LLM call (injected, so this module stays provider-agnostic) given the
     repo's manifest files + AGENTS.md.

The result is used IN-MEMORY for one run and never written to config (runs do
not mutate config). The operator is shown what was picked + how to pin it.

`verify_command` is an argv tuple run with NO shell, so a simple command
tokenises directly; a shell pipeline (`a && b`, `a | b`) is wrapped as
`("sh", "-c", "<pipeline>")` -- `sh` resolves on the jail PATH
(`/usr/bin:/bin` plus the standard bin dirs that exist). Operator tools like
`uv` resolve now, so `uv run pytest` is a fine inferred command (it uses the
already-synced venv; the sandbox cannot sync).
"""

from __future__ import annotations

import json
import os
import re
import shlex
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

# A command line with any of these is a shell construct, not a bare argv; we
# wrap it in `sh -c` rather than mis-tokenising it.
_SHELL_META = re.compile(r"(\|\||&&|[|&;<>`]|\$\()")
_VERIFY_HEADING = re.compile(r"^#{1,6}\s*(verify|test)\b", re.IGNORECASE)
_INLINE_VERIFY = re.compile(r"^\s*(?:verify|test)\s*:\s*(.+)$", re.IGNORECASE)
_MAKE_TARGET = re.compile(r"^([A-Za-z0-9_-]+)\s*:", re.MULTILINE)


@dataclass(frozen=True, slots=True)
class InferredVerify:
    """A verify command agent6 inferred for one run (never persisted)."""

    argv: tuple[str, ...]
    source: str  # "agents_md" | "package.json" | "Makefile:test" | "pyproject" | ... | "llm"


def line_to_argv(cmd: str) -> tuple[str, ...] | None:
    """A single logical command line -> argv, wrapping a shell pipeline in sh -c."""
    cmd = cmd.strip()
    if not cmd:
        return None
    if _SHELL_META.search(cmd):
        return ("sh", "-c", cmd)
    try:
        parts = shlex.split(cmd)
    except ValueError:
        return None
    return tuple(parts) or None


def _block_to_argv(block: list[str]) -> tuple[str, ...] | None:
    """A fenced code block (possibly multi-line, comments, backslash-continued)
    -> one argv. Comment/blank lines are dropped; continued lines are joined."""
    logical: list[str] = []
    for raw in block:
        line = raw.rstrip()
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        logical.append(line)
    if not logical:
        return None
    joined = " ".join(part.rstrip("\\").strip() for part in logical).strip()
    return line_to_argv(joined)


def _first_fenced_block(lines: list[str], start: int) -> list[str] | None:
    """The first ``` fenced block opening within 12 lines of *start*, else None."""
    i = start
    # Allow a short prose gap between the heading and its code fence.
    while i < len(lines) and i < start + 12:
        if lines[i].lstrip().startswith("```"):
            body: list[str] = []
            i += 1
            while i < len(lines) and not lines[i].lstrip().startswith("```"):
                body.append(lines[i])
                i += 1
            return body
        i += 1
    return None


def read_agents_md(repo_root: Path) -> str:
    """The repo's AGENTS.md text for inference; "" when absent or unreadable."""
    path = repo_root / "AGENTS.md"
    if not path.is_file():
        return ""
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def verify_from_agents_md(agents_md: str) -> tuple[str, ...] | None:
    """Parse a verify command out of AGENTS.md.

    Honours a `## Verify command`/`## Test` heading followed by a fenced
    block, or an inline `Verify:`/`Test:` line. Returns argv or None.
    """
    if not agents_md:
        return None
    lines = agents_md.splitlines()
    for i, line in enumerate(lines):
        if _VERIFY_HEADING.match(line.strip()):
            block = _first_fenced_block(lines, i + 1)
            if block is not None and (argv := _block_to_argv(block)) is not None:
                return argv
    for line in lines:
        m = _INLINE_VERIFY.match(line)
        if m and (argv := line_to_argv(m.group(1))) is not None:
            return argv
    return None


def _has_make_target(text: str, target: str) -> bool:
    return any(m.group(1) == target for m in _MAKE_TARGET.finditer(text))


def _python(repo_root: Path) -> str:
    """The interpreter a pytest gate runs with: the project's `.venv/bin/python`
    when it exists (symlinks into /usr, so jail-visible per the documented
    convention), else `python3` on PATH, which is the correct interpreter in
    containers and system-/conda-python setups that have no `.venv`
    (hardcoding the missing `.venv/bin/python` there silently breaks verify).
    The operator can pin one."""
    return ".venv/bin/python" if (repo_root / ".venv" / "bin" / "python").exists() else "python3"


Signal = tuple[tuple[str, ...], str]  # (argv, source)


def _verify_sh(repo_root: Path) -> Signal | None:
    script = repo_root / "verify.sh"
    if not script.is_file():
        return None
    argv = ("./verify.sh",) if os.access(script, os.X_OK) else ("sh", "verify.sh")
    return (argv, "verify.sh")


def _package_json(repo_root: Path) -> Signal | None:
    pkg = repo_root / "package.json"
    if not pkg.is_file():
        return None
    try:
        data = json.loads(pkg.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        data = {}
    scripts = data.get("scripts") if isinstance(data, dict) else None
    if isinstance(scripts, dict) and isinstance(scripts.get("test"), str):
        return (("npm", "test", "--silent"), "package.json")
    return None


def _makefile(repo_root: Path) -> Signal | None:
    for mk in ("Makefile", "makefile", "GNUmakefile"):
        p = repo_root / mk
        if not p.is_file():
            continue
        try:
            txt = p.read_text(encoding="utf-8", errors="replace")
        except OSError:
            txt = ""
        for target in ("test", "check"):
            if _has_make_target(txt, target):
                return (("make", target), f"Makefile:{target}")
    return None


def _python_manifest(repo_root: Path) -> Signal | None:
    manifests = ("pyproject.toml", "pytest.ini", "tox.ini", "setup.cfg", "setup.py")
    if any((repo_root / f).is_file() for f in manifests):
        return ((_python(repo_root), "-m", "pytest", "-q"), "pyproject")
    return None


def _cargo(repo_root: Path) -> Signal | None:
    return (
        (("cargo", "test", "--quiet"), "Cargo.toml")
        if (repo_root / "Cargo.toml").is_file()
        else None
    )


def _go(repo_root: Path) -> Signal | None:
    return (("go", "test", "./..."), "go.mod") if (repo_root / "go.mod").is_file() else None


def _loose_python_tests(repo_root: Path) -> Signal | None:
    if any(repo_root.glob("test_*.py")) or any((repo_root / "tests").glob("test_*.py")):
        return ((_python(repo_root), "-m", "pytest", "-q"), "test_*.py")
    return None


# In order: a root `verify.sh` first (an operator wrote it for exactly this),
# then the manifests, then loose Python tests (`test_*.py` at the root or under
# `tests/`) last, so a Rust or Go repo's `tests/` dir never reads as pytest.
_REPO_SIGNALS = (
    _verify_sh,
    _package_json,
    _makefile,
    _python_manifest,
    _cargo,
    _go,
    _loose_python_tests,
)


def verify_from_repo_signals(repo_root: Path) -> Signal | None:
    """Deterministic detection from the repo's own files (`_REPO_SIGNALS`, in
    order). Returns (argv, source)."""
    return next((found for probe in _REPO_SIGNALS if (found := probe(repo_root))), None)


# Files whose contents most strongly signal the test command, fed to the LLM.
_MANIFEST_FILES = (
    "pyproject.toml",
    "package.json",
    "Makefile",
    "Cargo.toml",
    "go.mod",
    "tox.ini",
    "pytest.ini",
    "noxfile.py",
)

VERIFY_INFER_SYSTEM_PROMPT = (
    "You infer the single command a CI/verify step runs to decide whether a change to THIS"
    " repository passes (build + tests). You are given the repo's manifest files and AGENTS.md.\n\n"
    'Reply with ONLY a JSON array of argv strings and nothing else, e.g. ["pytest","-q"].\n'
    "Hard rules:\n"
    "- The command runs in a locked-down sandbox: PATH is /usr/bin:/bin, the standard bin"
    " dirs that exist (/usr/local/bin, ~/.local/bin, ~/.cargo/bin, ...), and the repo dir,"
    " with an ephemeral $HOME and NO network. Operator tools resolve, so `uv run pytest`"
    " works (it uses the already-synced venv; the sandbox cannot sync). Prefer the"
    " project's real runner: `uv run ...` for a uv project, else a stdlib .venv/bin/python"
    " or /usr/bin/python3, or system cargo/go/node/make.\n"
    "- Prefer the project's real fast test/build command, not a lint-only or syntax check.\n"
    '- If you need a shell pipeline, return ["sh","-c","<pipeline>"].\n'
    "- If you genuinely cannot determine one, return []."
)


def gather_repo_manifests(repo_root: Path, agents_md: str, *, cap: int = 4000) -> str:
    """A clipped context string of manifest files + AGENTS.md for the LLM call."""
    parts: list[str] = []
    try:
        top = sorted(p.name + ("/" if p.is_dir() else "") for p in repo_root.iterdir())
    except OSError:
        top = []
    parts.append("<top-level>\n" + " ".join(top[:80]) + "\n</top-level>")
    for name in _MANIFEST_FILES:
        p = repo_root / name
        if p.is_file():
            try:
                text = p.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            parts.append(f'<file path="{name}">\n{text[:cap]}\n</file>')
    if agents_md.strip():
        parts.append(f"<AGENTS.md>\n{agents_md[:cap]}\n</AGENTS.md>")
    return "\n\n".join(parts)


def parse_llm_verify(text: str) -> tuple[str, ...] | None:
    """Extract a JSON argv array from the model's reply. Returns argv or None."""
    if not text:
        return None
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if match is None:
        return None
    try:
        data = json.loads(match.group(0))
    except ValueError:
        return None
    if not isinstance(data, list) or not data:
        return None
    if not all(isinstance(x, str) and x.strip() for x in data):
        return None
    return tuple(x for x in data)


def infer_verify_command(
    repo_root: Path,
    agents_md: str,
    *,
    llm_call: Callable[[str], str] | None = None,
) -> InferredVerify | None:
    """Infer a verify command (AGENTS.md -> repo signals -> LLM). None if unknown.

    `llm_call` takes the gathered repo context and returns the model's raw
    text; pass None to skip the LLM tier (deterministic-only). The LLM tier
    reads manifest content and AGENTS.md prose; with neither present it is
    skipped: a bare filename listing can only confirm "none" or invent a
    gate, and a project created mid-run is adopted by the mid-run check.
    """
    argv = verify_from_agents_md(agents_md)
    if argv is not None:
        return InferredVerify(argv=argv, source="agents_md")
    sig = verify_from_repo_signals(repo_root)
    if sig is not None:
        return InferredVerify(argv=sig[0], source=sig[1])
    readable = bool(agents_md.strip()) or any(
        (repo_root / name).is_file() for name in _MANIFEST_FILES
    )
    if llm_call is not None and readable:
        context = gather_repo_manifests(repo_root, agents_md)
        try:
            raw = llm_call(context)
        except Exception:  # inference is best-effort; never fail the run on it
            raw = ""
        argv = parse_llm_verify(raw)
        if argv is not None:
            return InferredVerify(argv=argv, source="llm")
    return None
