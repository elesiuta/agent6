# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The optional pre-loop prompt-revision pass.

Before the worker loop starts, the reviser model can rewrite a terse task into
an explicit one and surface clarifying questions. This module holds the parse
of its output, the repo-context block fed to it, the effective-task assembly,
and the small text helpers they use. The loop owns running the reviser call.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from agent6.types import RepoSummary

# One leading list marker ("- ", "* ", "1. ", "2) "). A charset lstrip would
# also eat leading digits of the question itself ("- 32-bit ..." -> "bit ...").
# The numeric marker requires trailing whitespace so a bare decimal that opens a
# question keeps it ("0.5s latency budget OK?" must not become "5s ...").
_LIST_MARKER_RE = re.compile(r"^\s*(?:[-*]|\d+[.)]\s)\s*")


@dataclass(frozen=True, slots=True)
class PromptRevision:
    revised_task: str
    clarifying_questions: tuple[str, ...] = ()


class PromptRevisionError(Exception):
    """Raised when the optional prompt-revision pass cannot produce a task."""


class PromptRevisionDeclined(PromptRevisionError):
    """The operator quit at the interactive choice: their stop, not a failure."""


def clip_text(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max(0, max_chars - 40)].rstrip() + "\n...[truncated for prompt revision]"


def tag_body(text: str, tag: str) -> str:
    start_tag = f"<{tag}>"
    end_tag = f"</{tag}>"
    start = text.find(start_tag)
    if start == -1:
        return ""
    start += len(start_tag)
    end = text.find(end_tag, start)
    if end == -1:
        return ""
    return text[start:end].strip()


def parse_prompt_revision(text: str) -> PromptRevision:
    revised = tag_body(text, "revised_task") if "<revised_task>" in text else text.strip()
    questions_raw = tag_body(text, "clarifying_questions")
    questions: list[str] = []
    for raw_line in questions_raw.splitlines():
        line = _LIST_MARKER_RE.sub("", raw_line).strip()
        if not line or line.lower() in {"none", "n/a", "no questions"}:
            continue
        questions.append(line)
    return PromptRevision(revised_task=revised.strip(), clarifying_questions=tuple(questions[:3]))


def format_prompt_revision_context(repo: RepoSummary) -> str:
    if repo.is_git:
        repo_line = (
            f"Repository: branch={repo.branch}, head={repo.head_sha[:12]}, files={repo.file_count}"
        )
    else:
        # Same degrade as the worker prompt: outside git a fake empty header
        # would send the model after branch and history that do not exist.
        repo_line = "Directory (not a git repository; no branch, history, or tracked-file map)."
    parts = [
        repo_line,
        f"Top-level: {', '.join(repo.top_level)}",
    ]
    if repo.agents_md:
        parts.append("AGENTS.md:\n" + clip_text(repo.agents_md, 5000))
    if repo.repo_map:
        parts.append("Repo map:\n" + clip_text(repo.repo_map, 4000))
    if repo.symbol_outline:
        parts.append("Symbol outline:\n" + clip_text(repo.symbol_outline, 5000))
    if repo.co_change_pairs:
        lines = "\n".join(
            f"  {p.file_a} <-> {p.file_b} ({p.count})" for p in repo.co_change_pairs[:15]
        )
        parts.append("Git co-change pairs:\n" + lines)
    if repo.hot_symbols:
        lines = "\n".join(
            f"  {s.name} ({s.kind}) at {s.def_path}:{s.def_line}, {s.files_referenced} files"
            for s in repo.hot_symbols[:12]
        )
        parts.append("Hot symbols:\n" + lines)
    if repo.recent_log:
        parts.append("Recent commits:\n" + clip_text(repo.recent_log, 2000))
    return clip_text("\n\n".join(parts), 20_000)


def format_effective_task(raw_task: str, revision: PromptRevision) -> str:
    pieces = [
        "Revised task prompt:",
        revision.revised_task,
        "Original user task (authoritative if anything conflicts):",
        raw_task,
    ]
    if revision.clarifying_questions:
        pieces.extend(
            [
                "Clarifying questions raised by the revision pass:",
                "\n".join(f"- {q}" for q in revision.clarifying_questions),
                (
                    "Proceed under conservative assumptions if these cannot be answered from"
                    " repository context; do not stop solely because questions exist."
                ),
            ]
        )
    return "\n\n".join(pieces)
