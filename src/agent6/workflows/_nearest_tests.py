# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Test files nearest a set of changed paths, for the scoped verify fallback:
when the full gate overruns its budget, these are the tests most likely to
judge the change. Pure path heuristics over the worktree; no git, no config.
"""

from __future__ import annotations

from pathlib import Path

_TEST_DIR_NAMES = ("tests", "test")
# Directories never scanned for tests (vendored trees and envs are large and
# judge nothing).
_SKIP_DIRS = {".git", ".venv", "venv", "node_modules", "__pycache__", ".tox", ".eggs"}
_SCAN_CAP = 4000


def diff_changed_paths(diff: str) -> tuple[str, ...]:
    """The b-side paths of a unified diff's `diff --git a/x b/y` lines, in
    order, deduplicated. Deleted files still appear (their b-side names the
    old path); the caller's existence checks drop them."""
    seen: dict[str, None] = {}
    for ln in diff.splitlines():
        if ln.startswith("diff --git ") and " b/" in ln:
            seen.setdefault(ln.split(" b/", 1)[1].strip(), None)
    return tuple(seen)


def is_bare_pytest(command: tuple[str, ...]) -> bool:
    """True when appending test paths to *command* selects them: some token
    is `pytest` (or ends in `/pytest`) and every token after it is an option.
    A `sh -c` script binds appended paths as $0/$1 with the script unchanged,
    and a command that already names a path (`pytest tests`) unions the two,
    so neither takes a selection."""
    for i, tok in enumerate(command):
        if tok == "pytest" or tok.endswith("/pytest"):
            return all(t.startswith("-") for t in command[i + 1 :])
    return False


def _is_test_file(rel: Path) -> bool:
    name = rel.name
    return (name.startswith("test_") and name.endswith(".py")) or name.endswith("_test.py")


def _candidates_for(rel: Path) -> list[Path]:
    """Conventional homes for the tests of one changed source file, most
    specific first: siblings, a tests dir beside it, then a mirrored or flat
    layout under each ancestor's tests dir."""
    stem = rel.stem
    parent = rel.parent
    out = [
        parent / f"test_{stem}.py",
        parent / f"{stem}_test.py",
    ]
    for tdir in _TEST_DIR_NAMES:
        out.append(parent / tdir / f"test_{stem}.py")
    for anc in [*rel.parents][1:]:
        rest = parent.relative_to(anc)
        for tdir in _TEST_DIR_NAMES:
            out.append(anc / tdir / rest / f"test_{stem}.py")
            out.append(anc / tdir / f"test_{stem}.py")
    return out


def _scan_test_dirs(root: Path, stems: set[str]) -> list[Path]:
    """Name-matched test files anywhere under the repo's test dirs, bounded:
    catches layouts the conventions above miss (tests/unit/test_mod.py).
    Scans at most _SCAN_CAP entries; a larger tree yields what was seen."""
    hits: list[Path] = []
    budget = _SCAN_CAP
    wanted = {f"test_{s}.py" for s in stems} | {f"{s}_test.py" for s in stems}
    stack = [p for n in _TEST_DIR_NAMES if (p := root / n).is_dir()]
    while stack and budget > 0:
        d = stack.pop()
        try:
            entries = sorted(d.iterdir())
        except OSError:
            continue
        for e in entries:
            if budget <= 0:
                break
            budget -= 1
            if e.is_dir():
                if e.name not in _SKIP_DIRS:
                    stack.append(e)
            elif e.name in wanted:
                hits.append(e.relative_to(root))
    return hits


def nearest_test_paths(root: Path, changed: tuple[str, ...], *, cap: int = 20) -> tuple[str, ...]:
    """Repo-relative test files most likely to judge the changed paths:
    changed test files themselves, conventional siblings/mirrors of each
    changed .py source, then name matches under the repo's test dirs.
    Existing files only, deduplicated in that order, at most *cap*. A changed
    helper or conftest under a tests dir is neither run nor mirrored. Name
    matches only: a test directory named for the module (tests/frame/ for
    core/frame.py) is not found, and a same-stem test elsewhere
    (plotting/test_frame.py) is picked."""
    picked: dict[str, None] = {}
    sources: list[Path] = []
    for c in changed:
        rel = Path(c)
        if rel.suffix != ".py":
            continue
        if _is_test_file(rel):
            if (root / rel).is_file():
                picked.setdefault(rel.as_posix(), None)
        elif not any(part in _TEST_DIR_NAMES for part in rel.parts[:-1]):
            sources.append(rel)
    for rel in sources:
        for cand in _candidates_for(rel):
            if (root / cand).is_file():
                picked.setdefault(cand.as_posix(), None)
    if sources:
        for hit in _scan_test_dirs(root, {s.stem for s in sources}):
            picked.setdefault(hit.as_posix(), None)
    return tuple(picked)[:cap]
