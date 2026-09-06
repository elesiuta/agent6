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

import hashlib
import json
import re
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from agent6.errors import OperatorError
from agent6.portable import atomic_write

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
# Beside a lane's store: the sha256 of every file `seed_store` copied in, by
# name, so the import can tell an untouched copy from a lane's edit.
SEED_NAME = "memory-seed.json"

_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,63}$")


class MemoryStoreError(OperatorError):
    """Memory-store operation failed (bad name, unreadable store)."""


def memory_dir(state_dir: Path) -> Path:
    return state_dir / MEMORY_DIR_NAME


def index_path(state_dir: Path) -> Path:
    return memory_dir(state_dir) / INDEX_NAME


def seed_path(state_dir: Path) -> Path:
    return state_dir / SEED_NAME


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


def _entries(text: str) -> list[str]:
    """A decisions file as its `- ` entries, each with its continuation lines."""
    entries: list[str] = []
    for line in text.splitlines():
        if line.startswith("- ") or not entries:
            entries.append(line)
        else:
            entries[-1] += "\n" + line
    return entries


def _ruling(entry: str) -> str:
    """The question and answer of an entry, without its stamp and session tag:
    two lanes answering alike record the same ruling under different tags."""
    return entry.split("] ", 1)[-1]


def merge_decisions(src_state_dir: Path, dst_state_dir: Path) -> tuple[int, int]:
    """Append the rulings recorded under *src_state_dir* to *dst_state_dir*'s
    decisions file (a fan-out lane's answers outlive its state dir). A
    ruling is its question and answer: one the destination already holds,
    however long ago and under whatever session tag, is skipped, and so is a
    repeat within the source. Returns (appended, skipped)."""
    try:
        text = decisions_path(src_state_dir).read_text(encoding="utf-8")
    except OSError:
        return 0, 0
    path = decisions_path(dst_state_dir)
    try:
        existing = path.read_text(encoding="utf-8")
    except OSError:
        existing = ""
    known = {_ruling(e) for e in _entries(existing.strip())}
    entries = _entries(text.strip())
    fresh: list[str] = []
    for entry in entries:
        if (ruling := _ruling(entry)) not in known:
            known.add(ruling)
            fresh.append(entry)
    if fresh:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            if existing and not existing.endswith("\n"):
                fh.write("\n")
            fh.write("\n".join(fresh) + "\n")
    return len(fresh), len(entries) - len(fresh)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass(frozen=True, slots=True)
class MemoryMerge:
    """What a lane's memory left in the origin's store at import, by name."""

    carried: tuple[str, ...] = ()  # new names, landed with their index lines
    updated: tuple[str, ...] = ()  # the lane's edit replaced a copy unchanged since seeding
    deleted: tuple[str, ...] = ()  # the lane's deletion removed such a copy
    held: tuple[str, ...] = ()  # changed on both sides, or a name already taken: kept aside


def merge_memory(src_state_dir: Path, dst_state_dir: Path, *, held_dir: Path) -> MemoryMerge:
    """Carry a lane's memory into the origin's store the way its branch comes
    back. A copy unchanged since seeding is nothing. A change over a copy the
    origin has not touched since seeding lands: an edit replaces the file and
    its index line, a deletion removes both. A change on both sides is held
    back (the lane's version kept under *held_dir*) and named. A new name
    lands with its index line, or is held when the origin holds that name with
    other content (two lanes invented it); a file the lane's own index does
    not list is left where it is. The rulings have `merge_decisions`."""
    src = memory_dir(src_state_dir)
    if not src.is_dir():
        return MemoryMerge()
    dst = memory_dir(dst_state_dir)
    dst.mkdir(parents=True, exist_ok=True)
    try:
        seeds: dict[str, str] = json.loads(seed_path(src_state_dir).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        seeds = {}
    src_index = index_text(src_state_dir).splitlines()
    lane = {p.stem: p for p in src.glob("*.md") if p.name not in (INDEX_NAME, DECISIONS_NAME)}
    landed: dict[str, list[str]] = {"carry": [], "update": [], "delete": [], "hold": []}
    for name in sorted(seeds.keys() | lane.keys()):
        path = lane.get(name)
        origin = dst / f"{name}.md"
        seed = seeds.get(name)
        theirs = _sha256(path) if path is not None else None
        ours = _sha256(origin) if origin.is_file() else None
        hook = _index_hook(src_index, name)
        taken = seed is None and (ours is not None or _index_has(dst_state_dir, name))
        fate = _fate(seed, theirs, ours, hook=hook, taken=taken)
        if fate == "skip":
            continue
        if fate == "delete":
            _drop_index_line(dst_state_dir, name)
            origin.unlink()
        elif fate == "hold":
            if path is not None:
                held_dir.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, held_dir / path.name)
        else:
            assert path is not None and hook is not None  # a carry or update has a file and a line
            shutil.copyfile(path, origin)
            if fate == "carry":
                _append_index_line(dst_state_dir, name, hook)
            else:
                _replace_index_line(dst_state_dir, name, hook)
        landed[fate].append(name)
    return MemoryMerge(*(tuple(landed[k]) for k in ("carry", "update", "delete", "hold")))


def _fate(
    seed: str | None, theirs: str | None, ours: str | None, *, hook: str | None, taken: bool
) -> Literal["skip", "carry", "update", "delete", "hold"]:
    """One name's fate at import, from the digests of the seeded copy, the
    lane's file and the origin's file (None: absent), whether the lane's index
    lists it (*hook*) and whether the origin already uses the name (*taken*)."""
    if theirs in (seed, ours):
        return "skip"  # untouched in the lane, or the same content on both sides
    if seed is None:  # new in the lane
        if hook is None:
            return "skip"  # unindexed there: invisible there, and stays so
        return "hold" if taken else "carry"
    if ours != seed:
        return "hold"  # changed on both sides
    return "delete" if theirs is None else "update"


def seed_store(src_state_dir: Path, dst_state_dir: Path) -> int:
    """Copy the repo's memory (index, facts, recorded rulings) into a fresh
    state dir, leaving anything already there, and record each copied fact's
    digest in `seed_path` for the import. Returns the files copied.

    A `--parallel` lane clones the repo into a workspace of its own, so its
    state dir is new and its memory empty; without this the lanes run blind to
    the rulings every other run on that repo is given. Copies, never a link: a lane
    must not write the origin's store mid-run; `merge_memory` and
    `merge_decisions` carry what it wrote back at import.
    """
    src = memory_dir(src_state_dir)
    if not src.is_dir():
        return 0
    copied = 0
    digests: dict[str, str] = {}
    dst = memory_dir(dst_state_dir)
    dst.mkdir(parents=True, exist_ok=True)
    for path in sorted(src.iterdir()):
        target = dst / path.name
        if not path.is_file() or target.exists():
            continue
        shutil.copyfile(path, target)
        copied += 1
        if path.suffix == ".md" and path.name not in (INDEX_NAME, DECISIONS_NAME):
            digests[path.stem] = _sha256(target)
    atomic_write(seed_path(dst_state_dir), json.dumps(digests, indent=1) + "\n")
    return copied


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
    (memory is context, one stray byte must not kill every run).

    A byte that is not UTF-8 is REPLACED rather than fatal: read strictly, one
    of them would empty the whole index for every run, and the next `memory
    add` would rebuild the file from that empty read.
    An unreadable FILE (a permission, a directory) is still ""."""
    try:
        return index_path(state_dir).read_text(encoding="utf-8", errors="replace").strip()
    except OSError:
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


def _index_hook(index_lines: list[str], name: str) -> str | None:
    """The hook text an index line carries for *name*, None when it has none."""
    pattern = re.compile(rf"^\s*[-*]\s*{re.escape(name)}\s*:")
    line = next((ln for ln in index_lines if pattern.match(ln)), None)
    return None if line is None else line.split(":", 1)[1].strip()


def _index_pattern(name: str) -> re.Pattern[bytes]:
    return re.compile(rb"^\s*[-*]\s*" + re.escape(name.encode("utf-8")) + rb"\s*:")


def _drop_index_line(state_dir: Path, name: str) -> None:
    """Remove the index line naming *name*, over bytes: a rewrite through the
    replacing reader would turn every byte that is not UTF-8 into U+FFFD, in
    lines the operator wrote."""
    idx = index_path(state_dir)
    pattern = _index_pattern(name)
    kept = [ln for ln in idx.read_bytes().split(b"\n") if not pattern.match(ln)]
    atomic_write(idx, b"\n".join(kept))


def _replace_index_line(state_dir: Path, name: str, hook: str) -> None:
    """Rewrite the index line naming *name* in place (over bytes, like
    `_drop_index_line`), appending one when there is none."""
    idx = index_path(state_dir)
    pattern = _index_pattern(name)
    try:
        lines = idx.read_bytes().split(b"\n")
    except OSError:
        lines = []
    if not any(pattern.match(ln) for ln in lines):
        _append_index_line(state_dir, name, hook)
        return
    new = f"- {name}: {hook}".encode()
    atomic_write(idx, b"\n".join(new if pattern.match(ln) else ln for ln in lines))


def _append_index_line(state_dir: Path, name: str, hook: str) -> None:
    """Append one line to the index, never rewriting the lines it holds: a
    rewrite from an unreadable (so empty) read deletes every one of them."""
    idx = index_path(state_dir)
    try:
        tail = idx.read_bytes()[-1:]
    except OSError:
        tail = b""
    with idx.open("a", encoding="utf-8") as fh:
        if tail not in (b"", b"\n"):
            fh.write("\n")
        fh.write(f"- {name}: {hook}\n")


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
        _drop_index_line(state_dir, name)
    if path.is_file():
        path.unlink()


def show(state_dir: Path, name: str) -> str:
    _check_name(name)
    path = memory_dir(state_dir) / f"{name}.md"
    try:
        return path.read_text(encoding="utf-8")
    except OSError as exc:
        raise MemoryStoreError(f"no memory named {name!r}") from exc
