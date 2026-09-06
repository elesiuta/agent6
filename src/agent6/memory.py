# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Per-repo agent memory under `<state_dir>/memory/`.

One fact per markdown file plus a `MEMORY.md` index (one line per entry).
The index is injected into every run's system prompt; the files are read and
edited with the ordinary in-process tools through a narrow path grant, so
recording or correcting a memory is a normal file edit. Model-authored
context: never instructions, never secrets. Repo-only by design; sharing a
memory across repos is the operator copying it.
"""

from __future__ import annotations

import re
import time
from pathlib import Path

from agent6.errors import OperatorError

MEMORY_DIR_NAME = "memory"
INDEX_NAME = "MEMORY.md"
# Operator rulings, harness-written and append-only: every ask_user answer and
# every steer that answered a question, verbatim. The model reads it (it is
# shown first, like the index) and never writes it.
DECISIONS_NAME = "DECISIONS.md"
DECISIONS_INJECT_CAP = 4_096
# The index is injected whole; past the cap it is clipped with a pointer so
# a runaway index cannot flood every prompt in the repo.
INDEX_INJECT_CAP = 4_096

_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,63}$")


class MemoryStoreError(OperatorError):
    """Memory-store operation failed (bad name, unreadable store)."""


def memory_dir(state_dir: Path) -> Path:
    return state_dir / MEMORY_DIR_NAME


def index_path(state_dir: Path) -> Path:
    return memory_dir(state_dir) / INDEX_NAME


def decisions_path(state_dir: Path) -> Path:
    return memory_dir(state_dir) / DECISIONS_NAME


def record_decision(
    state_dir: Path, *, question: str, answer: str, session: str, when: float | None = None
) -> str:
    """Append one operator ruling (question as asked, answer verbatim, the
    session and UTC time) and return the entry written. Append-only: nothing
    here rewrites or removes an earlier entry."""
    stamp = time.strftime("%Y-%m-%d %H:%MZ", time.gmtime(when))
    q = question.strip().replace("\n", "\n  ")
    a = answer.strip().replace("\n", "\n  ")
    entry = f"- {stamp} [{session}] Q: {q}\n  A: {a}\n"
    path = decisions_path(state_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(entry)
    return entry


def merge_decisions(src_state_dir: Path, dst_state_dir: Path) -> int:
    """Append every ruling recorded under *src_state_dir* to *dst_state_dir*'s
    decisions file (a fan-out lane's answers outlive its state dir). Returns
    the number of entries appended; 0 when the source recorded none."""
    try:
        text = decisions_path(src_state_dir).read_text(encoding="utf-8")
    except OSError:
        return 0
    if not text.strip():
        return 0
    path = decisions_path(dst_state_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(text if text.endswith("\n") else text + "\n")
    return sum(1 for line in text.splitlines() if line.startswith("- "))


def decisions_text(state_dir: Path) -> str:
    """The decisions file for injection: whole when it fits the cap, else
    its newest tail behind a pointer; "" when nothing is recorded."""
    try:
        text = decisions_path(state_dir).read_text(encoding="utf-8").strip()
    except OSError:
        return ""
    if len(text) <= DECISIONS_INJECT_CAP:
        return text
    tail = text[-DECISIONS_INJECT_CAP:]
    tail = tail[tail.index("\n- ") + 1 :] if "\n- " in tail else tail
    return f"... (earlier rulings clipped; {DECISIONS_NAME} holds all)\n{tail}"


def index_text(state_dir: Path) -> str:
    """The index body for prompt injection; "" when absent or unreadable
    (memory is context, one stray byte must not kill every run)."""
    try:
        return index_path(state_dir).read_text(encoding="utf-8").strip()
    except (OSError, ValueError):
        return ""


def _check_name(name: str) -> str:
    if not _NAME_RE.match(name):
        raise MemoryStoreError(
            f"bad memory name {name!r}: lowercase letters, digits, and dashes only"
        )
    return name


def _index_has(state_dir: Path, name: str) -> bool:
    pattern = re.compile(rf"^\s*[-*]\s*{re.escape(name)}\s*:")
    return any(pattern.match(ln) for ln in index_text(state_dir).splitlines())


def _append_index_line(state_dir: Path, name: str, hook: str) -> None:
    idx = index_path(state_dir)
    existing = index_text(state_dir)
    line = f"- {name}: {hook}"
    idx.write_text((existing + "\n" if existing else "") + line + "\n", encoding="utf-8")


def add(state_dir: Path, name: str, body: str) -> Path:
    """Operator CLI helper: write `<name>.md` and append its index line.

    The file is written first: an unindexed file is invisible to runs and
    harmless, while an index line without its file is a prompt that lies. A
    fault between the two writes heals on retry: an existing file with no
    index line is re-indexed from its own first line, named loudly.
    """
    body = body.strip()
    if not body:
        raise MemoryStoreError("memory body must be non-empty")
    d = memory_dir(state_dir)
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{_check_name(name)}.md"
    if path.exists():
        if _index_has(state_dir, name):
            raise MemoryStoreError(f"memory {name!r} exists; edit {path} or pick another name")
        first = (path.read_text(encoding="utf-8").strip().splitlines() or [""])[0]
        _append_index_line(state_dir, name, first[:120])
        raise MemoryStoreError(
            f"memory {name!r} existed but was missing from the index; re-indexed it."
            f" The body passed here was not saved; edit {path} to change it."
        )
    path.write_text(body + "\n", encoding="utf-8")
    _append_index_line(state_dir, name, body.splitlines()[0][:120])
    return path


def remove(state_dir: Path, name: str) -> None:
    """Operator CLI helper: delete `<name>.md` and its index line.

    The index line goes first: a file with no line is invisible and a retry
    can still delete it, while a line with no file is a prompt naming a
    memory that will not open. Either remnant alone is removable, so a fault
    between the two writes heals on retry; only a name with neither refuses.
    """
    _check_name(name)
    path = memory_dir(state_dir) / f"{name}.md"
    had_line = _index_has(state_dir, name)
    if not path.is_file() and not had_line:
        raise MemoryStoreError(f"no memory named {name!r}")
    if had_line:
        idx = index_path(state_dir)
        kept = [
            ln
            for ln in index_text(state_dir).splitlines()
            if not re.match(rf"^\s*[-*]\s*{re.escape(name)}\s*:", ln)
        ]
        idx.write_text("\n".join(kept) + ("\n" if kept else ""), encoding="utf-8")
    if path.is_file():
        path.unlink()


def show(state_dir: Path, name: str) -> str:
    _check_name(name)
    path = memory_dir(state_dir) / f"{name}.md"
    try:
        return path.read_text(encoding="utf-8")
    except OSError as exc:
        raise MemoryStoreError(f"no memory named {name!r}") from exc
