# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Repo context for the system prompt: AGENTS.md discovery and the
structural repo summary (file map + priors)."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from agent6.budget import BudgetExceeded
from agent6.git_ops import co_change_pairs, is_git_repo, recent_log, status, toplevel, tracked_files
from agent6.types import CoChangePair, HotSymbol, RepoSummary
from agent6.workflows._symbol_outline import build_symbol_outline_block

if TYPE_CHECKING:
    from agent6.tools.dispatch import ToolDispatcher

_REPO_MAP_MAX_LINES = 60
_REPO_MAP_MAX_FILES_PER_DIR = 6
# AGENTS.md is injected whole (pi and Claude Code both do); past this size the
# operator is warned at session start instead of the text being clipped.
AGENTS_MD_WARN_CHARS = 40_000


def _read_text(path: Path) -> str:
    """Tolerant read: a Windows-1252 byte or a permission-denied file must
    degrade, not crash the run AFTER session.start with no session.end (a dead
    run that listed as running)."""
    try:
        return path.read_text(encoding="utf-8", errors="replace") if path.is_file() else ""
    except OSError:
        return ""


def agents_md_text(root: Path) -> str:
    """The AGENTS.md text a session at *root* injects, whole (never clipped).

    When *root* sits below a git toplevel, the toplevel's file loads first and
    *root*'s own (if any) follows under a heading naming its directory, so a
    subdirectory start still carries the repo's conventions (pi and Claude Code
    collect ancestor files the same way)."""
    own = _read_text(root / "AGENTS.md")
    top = toplevel(root)
    if top is None or top == root:
        return own
    root_text = _read_text(top / "AGENTS.md")
    if not root_text:
        return own
    if not own:
        return root_text
    return f"{root_text}\n\n# AGENTS.md in {root.name}/ (this run's working directory)\n\n{own}"


def agents_md_notices(root: Path) -> tuple[str, ...]:
    """Session-start operator lines about the injected AGENTS.md: which files
    load when starting from a subdirectory, and an oversize warning (the text
    is injected whole; the remedy is trimming the file)."""
    out: list[str] = []
    top = toplevel(root)
    if top is not None and top != root and (top / "AGENTS.md").is_file():
        also = " plus this directory's" if (root / "AGENTS.md").is_file() else ""
        out.append(f"loading the repo root's AGENTS.md ({top}){also}")
    total = len(agents_md_text(root))
    if total > AGENTS_MD_WARN_CHARS:
        out.append(
            f"WARNING: AGENTS.md totals {total // 1000}k chars and rides in the"
            " system prompt of every model call; consider trimming it."
        )
    return tuple(out)


def _build_repo_map(tracked: tuple[str, ...]) -> str:
    """Compact `path/  (N files: a, b, ...)` directory map from git ls-files.

    Takes the already-resolved tracked-file list (shared with `file_count` so
    git ls-files runs once). Returns an empty string for an empty list. Output is
    capped at `_REPO_MAP_MAX_LINES` rows (plus one ``... (K more
    directories)`` summary line past the cap) so it never dominates the
    system prompt.
    """
    if not tracked:
        return ""
    by_dir: dict[str, list[str]] = {}
    for rel in tracked:
        parent, _, name = rel.rpartition("/")
        key = parent or "."
        by_dir.setdefault(key, []).append(name)
    keys = sorted(by_dir.keys(), key=lambda k: (k != ".", k))
    rows: list[str] = []
    for idx, key in enumerate(keys):
        files = sorted(by_dir[key])
        shown = files[:_REPO_MAP_MAX_FILES_PER_DIR]
        suffix = (
            ""
            if len(files) <= _REPO_MAP_MAX_FILES_PER_DIR
            else f", +{len(files) - _REPO_MAP_MAX_FILES_PER_DIR} more"
        )
        rows.append(f"  {key}/  ({len(files)} files: {', '.join(shown)}{suffix})")
        if len(rows) >= _REPO_MAP_MAX_LINES:
            remaining = len(keys) - idx - 1
            if remaining > 0:
                rows.append(f"  ... ({remaining} more directories)")
            break
    return "\n".join(rows)


def load_repo_summary(root: Path, *, dispatcher: ToolDispatcher | None = None) -> RepoSummary:
    """Build a `RepoSummary` for the workspace rooted at `root`.

    Base view (layout, AGENTS.md, recent commits, repo map) is shared by the
    implement and plan-mode workflows. When *dispatcher* is given (the run loop,
    and `agent6 prompt show`), ALSO enrich with structural priors: hot symbols
    (cross-file reference hot spots), git co-change pairs, and the tree-sitter
    symbol outline. Enrichment is best-effort -- a parser or git-history hiccup
    must not block the run -- but BudgetExceeded / KeyboardInterrupt propagate so
    the loop's budget guarantee and abort path stay intact.

    Outside a git repository (`agent6 ask` runs anywhere; run/plan refuse up
    front) the git-derived fields stay empty: the top-level listing is the
    model's starting point and it lists/reads deeper on demand. No recursive
    walk substitute: an unbounded crawl of an arbitrary directory (say $HOME)
    is exactly what the tracked-files count exists to avoid.
    """
    in_git = is_git_repo(root)
    st = status(root) if in_git else None
    top = tuple(
        sorted(
            p.name + ("/" if p.is_dir() else "")
            for p in root.iterdir()
            if not p.name.startswith(".")
        )
    )
    # Count git-tracked files: an unfiltered rglob would count .git/.venv/build
    # junk (a misleading number to the model) and traverse the whole tree every
    # startup. tracked is reused by _build_repo_map below.
    tracked = tracked_files(root) if in_git else ()
    file_count = len(tracked)
    agents_md = agents_md_text(root)
    hot: tuple[HotSymbol, ...] = ()
    co_change: tuple[CoChangePair, ...] = ()
    symbol_outline = ""
    if dispatcher is not None:
        try:
            hot = tuple(
                HotSymbol(*t)
                for t in dispatcher.symbol_index().hot_symbols(
                    max_symbols=20, min_files_referenced=2
                )
            )
        except (BudgetExceeded, KeyboardInterrupt):
            raise
        except Exception:
            hot = ()
        if in_git:
            try:
                co_change = tuple(CoChangePair(*t) for t in co_change_pairs(root, n_commits=200))
            except (BudgetExceeded, KeyboardInterrupt):
                raise
            except Exception:
                co_change = ()
        try:
            symbol_outline = build_symbol_outline_block(
                dispatcher.symbol_index().file_outlines(), root=root
            )
        except (BudgetExceeded, KeyboardInterrupt):
            raise
        except Exception:
            symbol_outline = ""
    return RepoSummary(
        root=root,
        branch=st.branch if st is not None else "",
        head_sha=st.head_sha if st is not None else "",
        file_count=file_count,
        top_level=top,
        agents_md=agents_md,
        recent_log=recent_log(root, n=20) if in_git else "",
        repo_map=_build_repo_map(tracked),
        co_change_pairs=co_change,
        hot_symbols=hot,
        symbol_outline=symbol_outline,
        is_git=in_git,
    )
