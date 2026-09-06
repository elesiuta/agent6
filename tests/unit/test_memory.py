# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The per-repo memory store: one fact per file plus the MEMORY.md index."""

from __future__ import annotations

from pathlib import Path

import pytest

from agent6.memory import (
    MemoryStoreError,
    add,
    decisions_path,
    index_path,
    index_text,
    memory_dir,
    merge_decisions,
    record_decision,
    remove,
    show,
)


def test_add_writes_file_and_index_line(tmp_path: Path) -> None:
    path = add(tmp_path, "build-quirk", "The build needs FOO=1.\nDetails here.")
    assert path == memory_dir(tmp_path) / "build-quirk.md"
    assert path.read_text() == "The build needs FOO=1.\nDetails here.\n"
    assert index_text(tmp_path) == "- build-quirk: The build needs FOO=1."


def test_add_refuses_duplicate_and_bad_names(tmp_path: Path) -> None:
    add(tmp_path, "one", "fact")
    with pytest.raises(MemoryStoreError, match="exists"):
        add(tmp_path, "one", "other")
    for bad in ("Has-Caps", "sl/ash", "..", "-lead", "a" * 65):
        with pytest.raises(MemoryStoreError, match="bad memory name"):
            add(tmp_path, bad, "x")
    with pytest.raises(MemoryStoreError, match="non-empty"):
        add(tmp_path, "empty", "   ")


def test_remove_deletes_file_and_index_line(tmp_path: Path) -> None:
    add(tmp_path, "keep", "kept fact")
    add(tmp_path, "drop", "dropped fact")
    remove(tmp_path, "drop")
    assert not (memory_dir(tmp_path) / "drop.md").exists()
    assert index_text(tmp_path) == "- keep: kept fact"
    with pytest.raises(MemoryStoreError, match="no memory named"):
        remove(tmp_path, "drop")


def test_add_reindexes_a_file_the_index_lost(tmp_path: Path) -> None:
    """A fault between add's two writes leaves the file present and the index
    line missing: the fact is invisible to runs, and a retry refused with
    "exists" while nothing could ever see it. The retry now re-indexes from
    the file's own first line and says the new body was not saved."""
    d = memory_dir(tmp_path)
    d.mkdir(parents=True)
    (d / "orphan.md").write_text("The original fact.\n", encoding="utf-8")
    with pytest.raises(MemoryStoreError, match="re-indexed"):
        add(tmp_path, "orphan", "a different body")
    assert index_text(tmp_path) == "- orphan: The original fact."
    assert (d / "orphan.md").read_text() == "The original fact.\n"


def test_remove_heals_either_remnant(tmp_path: Path) -> None:
    """A fault between remove's two writes leaves one remnant: a dangling
    index line (a prompt naming a memory that will not open) or an unindexed
    file. Either alone is removable; only a name with neither refuses."""
    add(tmp_path, "dangling", "fact one")
    (memory_dir(tmp_path) / "dangling.md").unlink()
    remove(tmp_path, "dangling")
    assert index_text(tmp_path) == ""

    add(tmp_path, "fileonly", "fact two")
    index_path(tmp_path).write_text("", encoding="utf-8")
    remove(tmp_path, "fileonly")
    assert not (memory_dir(tmp_path) / "fileonly.md").exists()

    with pytest.raises(MemoryStoreError, match="no memory named"):
        remove(tmp_path, "gone")


def test_show_reads_one_entry(tmp_path: Path) -> None:
    add(tmp_path, "fact", "body text")
    assert show(tmp_path, "fact") == "body text\n"
    with pytest.raises(MemoryStoreError, match="no memory named"):
        show(tmp_path, "absent")


def test_index_text_degrades_to_empty(tmp_path: Path) -> None:
    """Memory is context: an absent or unreadable index is "" for injection,
    never an error that kills every run in the repo."""
    assert index_text(tmp_path) == ""
    d = memory_dir(tmp_path)
    d.mkdir(parents=True)
    (d / "MEMORY.md").write_bytes(b"\xff\xfe broken")
    assert index_text(tmp_path) == ""


def test_record_decision_appends_verbatim_and_the_text_clips_to_the_newest(tmp_path: Path) -> None:
    """The harness-owned DECISIONS.md: append-only entries (question, answer
    verbatim with continuation lines indented, session, UTC time); the
    injected text keeps the newest rulings behind a pointer past the cap."""
    from agent6.memory import DECISIONS_INJECT_CAP, decisions_path, decisions_text, record_decision

    assert decisions_text(tmp_path) == ""
    first = record_decision(
        tmp_path, question="Keep the modal?", answer="No.\nInline item.", session="s1", when=0
    )
    second = record_decision(tmp_path, question="Port?", answer="8931", session="s2", when=60)
    text = decisions_path(tmp_path).read_text(encoding="utf-8")
    assert text == first + second
    assert first == "- 1970-01-01 00:00Z [s1] Q: Keep the modal?\n  A: No.\n  Inline item.\n"
    assert decisions_text(tmp_path) == text.strip()
    for i in range(200):
        record_decision(
            tmp_path, question=f"q{i} " + "x" * 40, answer="y" * 40, session="s", when=0
        )
    clipped = decisions_text(tmp_path)
    assert len(clipped) <= DECISIONS_INJECT_CAP + 80
    assert clipped.startswith("... (earlier rulings clipped") and clipped.rstrip().endswith(
        "y" * 40
    )
    assert "q199" in clipped and "Keep the modal" not in clipped


def test_merge_decisions_appends_a_lanes_rulings(tmp_path: Path) -> None:
    """A fan-out lane's rulings land in the coordinator's DECISIONS.md after its
    own; a lane that recorded none writes nothing."""
    lane, origin = tmp_path / "lane", tmp_path / "origin"
    assert merge_decisions(lane, origin) == 0
    assert not decisions_path(origin).exists()
    record_decision(origin, question="q0?", answer="a0", session="run")
    record_decision(lane, question="q1?", answer="a1", session="l1")
    assert merge_decisions(lane, origin) == 1
    text = decisions_path(origin).read_text(encoding="utf-8")
    assert text.index("Q: q0?") < text.index("Q: q1?")
    assert "[l1]" in text
