# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Commit message composition: the trailer line, the condensed message a
squash carries, and the Conventional Commits subject a machine's commit
gets. Pure string work; `git_ops` runs git.
"""

from __future__ import annotations

import re
import textwrap
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import PurePosixPath


def render_commit_trailer(fmt: str, *, models: Sequence[str]) -> str | None:
    """The `[git.commit].trailer` format string as a concrete trailer line, or
    None when unset. {model} names the model(s) that wrote the code, first-seen
    order (the primary worker first), ", "-joined and deduplicated; the model
    that wrote a commit MESSAGE never appears. The config validator pins the
    placeholder set and the "Key: value" shape."""
    if not fmt:
        return None
    return fmt.format(model=", ".join(dict.fromkeys(m for m in models if m)))


@dataclass(frozen=True, slots=True)
class CommitRow:
    """One commit on a run branch (oldest-first), for listing + squash-condensing."""

    sha: str
    subject: str
    message: str  # full %B


_ITER_SUBJECT_RE = re.compile(r"^agent6 iter \d+:\s*", re.IGNORECASE)


def condense_commit_message(rows: tuple[CommitRow, ...], *, subject: str) -> str:
    """Fold per-step commits into one readable message, so a squash reads as a
    single authored commit, not a squashed series.

    *subject* is the run's task (the headline). The body lists the distinct,
    de-noised per-step subjects (the `agent6 iter N:` prefix and checkpoint
    noise stripped). The provenance trailer is the commit emitter's job
    (identity.trailer), not this message's."""
    bullets: list[str] = []
    seen: set[str] = set()
    for row in rows:
        s = _ITER_SUBJECT_RE.sub("", row.subject).strip()
        if not s or s.lower().startswith("checkpoint") or s.lower() in seen:
            continue
        seen.add(s.lower())
        bullets.append(s)
    task = _ITER_SUBJECT_RE.sub("", subject).strip()
    headline = _headline_subject(task) or (bullets[0] if bullets else "agent6 run")
    parts = [headline]
    # If the subject truncated the task, wrap the full task into the body so
    # nothing is lost (git never wraps the subject line itself).
    full = " ".join(task.split())
    if full and full != headline:
        parts.append("")
        parts.extend(textwrap.wrap(full, width=72))
    if bullets:
        parts.append("")
        parts.extend(f"- {b}" for b in bullets)
    return "\n".join(parts)


_SUBJECT_LIMIT = 72  # git's soft subject cap; conventional tooling truncates past it


def _is_testish(p: str) -> bool:
    parts = PurePosixPath(p).parts
    name = parts[-1] if parts else ""
    return parts[:1] == ("tests",) or name.startswith("test_") or name == "conftest.py"


def _is_docish(p: str) -> bool:
    pp = PurePosixPath(p)
    return pp.suffix.lower() in (".md", ".rst") or pp.parts[:1] == ("docs",)


def _conventional_scope(paths: Sequence[str]) -> str:
    """The one common area the change touches, or "" when there is none: the
    package dir under `src/<pkg>/` (the module stem for a file directly under
    the package), else a second-level dir every path shares."""
    parts = [PurePosixPath(p).parts for p in paths if p]
    if not parts:
        return ""
    src_pkgs = [pp for pp in parts if len(pp) >= 3 and pp[0] == "src"]
    if src_pkgs:
        names = {pp[2] if len(pp) > 3 else str(PurePosixPath(pp[2]).stem) for pp in src_pkgs}
        return names.pop() if len(names) == 1 else ""
    tops = {pp[0] for pp in parts}
    if len(tops) != 1:
        return ""
    seconds = {pp[1] for pp in parts if len(pp) >= 3}
    return seconds.pop() if len(seconds) == 1 else ""


def conventional_commit_subject(changes: Sequence[tuple[str, str]], *, summary: str) -> str:
    """A Conventional Commits subject from `(status, path)` pairs, without a
    model call: all-tests -> `test`, all-docs -> `docs`, any added file ->
    `feat`, else `fix` (`chore` when nothing changed). Scope is the one
    common area (:func:`_conventional_scope`); the subject is *summary* with
    its head lowercased and any trailing period stripped, capped at 72."""
    paths = [p for _, p in changes]
    if not paths:
        ctype = "chore"
    elif all(_is_testish(p) for p in paths):
        ctype = "test"
    elif all(_is_docish(p) for p in paths):
        ctype = "docs"
    elif any(status.startswith("A") for status, _ in changes):
        ctype = "feat"
    else:
        ctype = "fix"
    scope = _conventional_scope(paths)
    head = f"{ctype}({scope}): " if scope else f"{ctype}: "
    subject = " ".join(summary.split()).rstrip(".")
    subject = (subject[:1].lower() + subject[1:]) if subject else "update"
    return (head + subject)[:_SUBJECT_LIMIT]


def _headline_subject(task: str, *, limit: int = _SUBJECT_LIMIT) -> str:
    """A short commit subject derived from the task's first clause: its first
    line, up to the first sentence end, whitespace-collapsed, capped at *limit*
    (an ellipsis marks a truncation). A run's whole task text as the subject
    reads as one unwrapped 180-char line that every git tool clips; the full
    task is wrapped into the body by the caller when this truncates it."""
    first_line = next((ln for ln in task.splitlines() if ln.strip()), "")
    match = re.search(r"[.!?](?:\s|$)", first_line)
    clause = first_line[: match.start()] if match else first_line
    clause = " ".join(clause.split())
    if len(clause) <= limit:
        return clause
    return clause[: limit - 1].rstrip() + "…"
