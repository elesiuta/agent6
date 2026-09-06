# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The per-repo memory store: one fact per file plus the MEMORY.md index."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent6.memory import (
    MemoryMerge,
    MemoryStoreError,
    add,
    decisions_path,
    index_path,
    index_text,
    memory_dir,
    merge_decisions,
    merge_memory,
    record_decision,
    remove,
    seed_path,
    seed_store,
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


def test_a_bad_byte_costs_one_character_not_the_index(tmp_path: Path) -> None:
    """Memory is context: an absent index is "" for injection, never an error
    that kills every run in the repo. A byte that is not UTF-8 costs itself and
    nothing else -- read strictly, one of them emptied the index for every run,
    and the next `memory add` rebuilt the file from that empty read, deleting
    every line the operator had."""
    from agent6.memory import add

    assert index_text(tmp_path) == ""
    add(tmp_path, "alpha", "the parser is generated")
    add(tmp_path, "beta", "the cache is per repo")
    idx = memory_dir(tmp_path) / "MEMORY.md"
    idx.write_bytes(idx.read_bytes() + b"- gamma: caf\xe9 a latin-1 byte\n")

    kept = index_text(tmp_path)

    assert "alpha" in kept and "beta" in kept and "gamma" in kept

    add(tmp_path, "delta", "a third fact")

    after = index_text(tmp_path)
    assert all(name in after for name in ("alpha", "beta", "gamma", "delta")), after


def test_index_lines_stay_adjacent_across_adds(tmp_path: Path) -> None:
    """The append asked the STRIPPED index text for its trailing newline, which
    it never has, so every add after the first opened with a blank line."""
    add(tmp_path, "one", "first fact")
    add(tmp_path, "two", "second fact")
    add(tmp_path, "three", "third fact")
    assert index_path(tmp_path).read_text(encoding="utf-8") == (
        "- one: first fact\n- two: second fact\n- three: third fact\n"
    )


def test_index_add_starts_a_line_of_its_own(tmp_path: Path) -> None:
    """An index edited by hand without a final newline gets one before the
    appended entry, so the two never share a line."""
    idx = index_path(tmp_path)
    idx.parent.mkdir(parents=True)
    idx.write_text("- hand: written by hand", encoding="utf-8")
    add(tmp_path, "two", "second fact")
    assert idx.read_text(encoding="utf-8") == "- hand: written by hand\n- two: second fact\n"


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
    assert merge_decisions(lane, origin) == (0, 0)
    assert not decisions_path(origin).exists()
    record_decision(origin, question="q0?", answer="a0", session="run")
    record_decision(lane, question="q1?", answer="a1\n\nsecond paragraph", session="l1")
    assert merge_decisions(lane, origin) == (1, 0)
    text = decisions_path(origin).read_text(encoding="utf-8")
    assert text.index("Q: q0?") < text.index("Q: q1?")
    assert "[l1]" in text
    assert text.endswith("  A: a1\n  \n  second paragraph\n"), text


def test_merge_decisions_skips_a_ruling_the_origin_already_holds(tmp_path: Path) -> None:
    """N lanes asked the same question and got the same answer: one ruling,
    recorded once, however many session tags it arrived under."""
    l1, l2, origin = tmp_path / "l1", tmp_path / "l2", tmp_path / "origin"
    record_decision(l1, question="Tabs?", answer="spaces", session="fan-l1", when=0)
    record_decision(l2, question="Tabs?", answer="spaces", session="fan-l2", when=60)
    record_decision(l2, question="Tabs?", answer="tabs", session="fan-l2", when=120)
    assert merge_decisions(l1, origin) == (1, 0)
    assert merge_decisions(l2, origin) == (1, 1)
    assert merge_decisions(l2, origin) == (0, 2)
    text = decisions_path(origin).read_text(encoding="utf-8")
    assert text.count("Q: Tabs?") == 2
    assert text.count("A: spaces") == 1 and "[fan-l1]" in text and "A: tabs" in text


def test_merge_decisions_skips_a_repeat_within_the_source(tmp_path: Path) -> None:
    """A lane that recorded one ruling twice carries it over once; a ruling
    the origin recorded long ago, under another session, is a skip too."""
    lane, origin = tmp_path / "lane", tmp_path / "origin"
    record_decision(origin, question="Tabs?", answer="spaces", session="old", when=0)
    record_decision(lane, question="Tabs?", answer="spaces", session="lane", when=3600)
    record_decision(lane, question="Lint?", answer="ruff", session="lane", when=3660)
    record_decision(lane, question="Lint?", answer="ruff", session="lane", when=3720)
    assert merge_decisions(lane, origin) == (1, 2)
    text = decisions_path(origin).read_text(encoding="utf-8")
    assert text.count("Q: Tabs?") == 1 and text.count("Q: Lint?") == 1


def test_merge_decisions_starts_on_its_own_line(tmp_path: Path) -> None:
    """A destination cut short of its trailing newline (a partial write, a
    hand edit) gets one before the first appended entry."""
    lane, origin = tmp_path / "lane", tmp_path / "origin"
    path = decisions_path(origin)
    path.parent.mkdir(parents=True)
    path.write_text("- 2026-01-01 00:00Z [x] Q: a?\n  A: b", encoding="utf-8")
    record_decision(lane, question="c?", answer="d", session="lane", when=0)
    assert merge_decisions(lane, origin) == (1, 0)
    assert path.read_text(encoding="utf-8") == (
        "- 2026-01-01 00:00Z [x] Q: a?\n  A: b\n- 1970-01-01 00:00Z [lane] Q: c?\n  A: d\n"
    )


def _seeded_lane(tmp_path: Path, origin: Path, name: str = "lane") -> Path:
    lane = tmp_path / name
    seed_store(origin, lane)
    return lane


def test_merge_memory_lands_new_facts_and_leaves_untouched_copies_alone(tmp_path: Path) -> None:
    """A lane's store is a copy of the origin's. At import a fact the lane
    added lands with its index line; the copies it never touched are nothing
    to report (every one read as "already recorded" before), and the same
    content on both sides is nothing either."""
    origin = tmp_path / "origin"
    add(origin, "repo-fact", "The build needs BUILD_ID set.")
    lane = _seeded_lane(tmp_path, origin)
    assert list(json.loads(seed_path(lane).read_text(encoding="utf-8"))) == ["repo-fact"]
    add(lane, "lane-fact", "The flaky test is test_clock.")
    held = tmp_path / "held"
    assert merge_memory(lane, origin, held_dir=held) == MemoryMerge(carried=("lane-fact",))
    assert "test_clock" in show(origin, "lane-fact")
    assert index_text(origin).splitlines() == [
        "- repo-fact: The build needs BUILD_ID set.",
        "- lane-fact: The flaky test is test_clock.",
    ]
    assert not held.exists()
    assert merge_memory(lane, origin, held_dir=held) == MemoryMerge()


def test_merge_memory_fast_forwards_a_lanes_edit_and_deletion(tmp_path: Path) -> None:
    """Over a copy the origin has not touched since seeding, the lane's edit
    replaces the file and its index line in place, and its deletion removes
    both: the branch rule, applied to the store. Before, an edit was held
    back silently and the lane's version ended with its state dir."""
    origin = tmp_path / "origin"
    add(origin, "a-fact", "A first.")
    add(origin, "b-fact", "B first.")
    add(origin, "c-fact", "C first.")
    lane = _seeded_lane(tmp_path, origin)
    (memory_dir(lane) / "a-fact.md").write_text("A second, refined.\n", encoding="utf-8")
    idx = index_path(lane)
    idx.write_text(idx.read_text(encoding="utf-8").replace("A first.", "A second, refined."))
    remove(lane, "c-fact")
    held = tmp_path / "held"
    assert merge_memory(lane, origin, held_dir=held) == MemoryMerge(
        updated=("a-fact",), deleted=("c-fact",)
    )
    assert show(origin, "a-fact") == "A second, refined.\n"
    assert not (memory_dir(origin) / "c-fact.md").exists()
    assert index_text(origin).splitlines() == ["- a-fact: A second, refined.", "- b-fact: B first."]
    assert not held.exists()


def test_merge_memory_holds_back_a_change_on_both_sides(tmp_path: Path) -> None:
    """Changed in the lane and in the origin since seeding: the origin keeps
    its version, the lane's is kept under held_dir, and both names are
    reported. A lane deletion over an origin edit is held the same way, with
    nothing to keep."""
    origin = tmp_path / "origin"
    add(origin, "a-fact", "A first.")
    add(origin, "d-fact", "D first.")
    lane = _seeded_lane(tmp_path, origin)
    (memory_dir(lane) / "a-fact.md").write_text("A per the lane.\n", encoding="utf-8")
    (memory_dir(origin) / "a-fact.md").write_text("A per the origin.\n", encoding="utf-8")
    remove(lane, "d-fact")
    (memory_dir(origin) / "d-fact.md").write_text("D per the origin.\n", encoding="utf-8")
    held = tmp_path / "held"
    assert merge_memory(lane, origin, held_dir=held) == MemoryMerge(held=("a-fact", "d-fact"))
    assert show(origin, "a-fact") == "A per the origin.\n"
    assert show(origin, "d-fact") == "D per the origin.\n"
    assert (held / "a-fact.md").read_text(encoding="utf-8") == "A per the lane.\n"
    assert not (held / "d-fact.md").exists()


def test_merge_memory_holds_back_a_name_two_lanes_invented(tmp_path: Path) -> None:
    """Two lanes recording different facts under one new name: the first
    lands, the second is held with its version kept, so neither silently
    overwrites the other."""
    origin = tmp_path / "origin"
    l1, l2 = _seeded_lane(tmp_path, origin, "l1"), _seeded_lane(tmp_path, origin, "l2")
    add(l1, "shared-name", "What lane one saw.")
    add(l2, "shared-name", "What lane two saw.")
    assert merge_memory(l1, origin, held_dir=tmp_path / "h1") == MemoryMerge(
        carried=("shared-name",)
    )
    assert merge_memory(l2, origin, held_dir=tmp_path / "h2") == MemoryMerge(held=("shared-name",))
    assert (tmp_path / "h2" / "shared-name.md").read_text(
        encoding="utf-8"
    ) == "What lane two saw.\n"
    assert show(origin, "shared-name") == "What lane one saw.\n"
