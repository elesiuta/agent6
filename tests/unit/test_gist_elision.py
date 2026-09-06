# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Tier-1 gist elision: a large read_file result decays to a distilled-gist
placeholder (batched distiller call per pass) before the bare marker, hot and
small reads are never gisted, and continued pressure demotes gists so the
byte bound still holds (bench/longhorizon FINDINGS #1)."""

from __future__ import annotations

import json
from typing import Any

from agent6.workflows._compaction import (
    ELISION_GIST_PREFIX,
    ELISION_PREFIX,
    GistRequest,
    compact_old_tool_results,
    elision_gist_placeholder,
    parse_gist_lines,
    read_file_text_from_result,
)
from agent6.workflows._conversation import Conversation, ToolResultItem, UserTurn


def _add_call(conv: Conversation, name: str, tool_input: dict[str, Any], content: str) -> None:
    """One (assistant tool_use, result) exchange."""
    turn = conv.assistant(
        [{"type": "tool_use", "id": f"t{len(conv)}", "name": name, "input": tool_input}]
    )
    conv.results(
        [
            ToolResultItem(
                tool_use_id=turn.tool_uses[0].id, content=content, for_call=turn.tool_uses[0]
            )
        ]
    )


def _add_read(conv: Conversation, path: str, text: str) -> None:
    """A read_file exchange whose result carries the real JSON envelope."""
    _add_call(conv, "read_file", {"path": path}, json.dumps({"content": text, "size": len(text)}))


def _contents(conv: Conversation) -> list[str]:
    return [
        item.content
        for turn in conv.turns
        if isinstance(turn, UserTurn)
        for item in turn.items
        if isinstance(item, ToolResultItem)
    ]


class _SpyGister:
    def __init__(self, replies: dict[str, str]) -> None:
        self.replies = replies
        self.calls: list[tuple[GistRequest, ...]] = []

    def __call__(self, requests: tuple[GistRequest, ...]) -> dict[str, str]:
        self.calls.append(requests)
        return {r.path: self.replies[r.path] for r in requests if r.path in self.replies}


def test_large_read_decays_to_gist_placeholder() -> None:
    doc = "R01 requires headers under 80 chars. END: lines are exempt (R10). " * 60
    conv = Conversation()
    _add_read(conv, "rules/r01.md", doc)
    _add_read(conv, "b.py", "x" * 500)
    _add_read(conv, "c.py", "y" * 500)
    gister = _SpyGister({"rules/r01.md": "R01: headers <80 chars; END: lines exempt (R10)"})
    stats = compact_old_tool_results(conv, max_total_bytes=1500, keep_recent=2, gister=gister)
    assert stats.elided == 1
    assert stats.gisted == 1
    got = _contents(conv)[0]
    assert got.startswith(ELISION_GIST_PREFIX)
    assert "rules/r01.md" in got
    assert "gist: R01: headers <80 chars; END: lines exempt (R10)" in got
    # The distiller saw the unwrapped file text, not the JSON envelope.
    assert gister.calls[0][0].content.startswith("R01 requires headers")


def test_no_gister_and_failed_gister_fall_back_to_bare() -> None:
    doc = "spec " * 1000
    for gister in (None, _SpyGister({})):
        conv = Conversation()
        _add_read(conv, "docs/spec.md", doc)
        _add_read(conv, "b.py", "x" * 500)
        _add_read(conv, "c.py", "y" * 500)
        stats = compact_old_tool_results(conv, max_total_bytes=1500, keep_recent=2, gister=gister)
        assert stats.elided == 1
        assert stats.gisted == 0
        got = _contents(conv)[0]
        assert got.startswith(ELISION_PREFIX)
        assert not got.startswith(ELISION_GIST_PREFIX)


def test_small_protected_and_non_read_results_are_never_gisted() -> None:
    conv = Conversation()
    _add_read(conv, "hot.py", "h" * 5000)  # protected: actively edited
    _add_read(conv, "tiny.md", "t" * 300)  # below GIST_MIN_SOURCE_CHARS
    _add_call(conv, "grep", {"pattern": "q"}, "g" * 5000)  # not a read_file
    _add_read(conv, "b.py", "x" * 500)
    _add_read(conv, "c.py", "y" * 500)
    gister = _SpyGister({"hot.py": "nope", "tiny.md": "nope", "grep": "nope"})
    stats = compact_old_tool_results(
        conv,
        max_total_bytes=1000,
        keep_recent=2,
        protect_paths=frozenset({"hot.py"}),
        gister=gister,
    )
    assert stats.elided == 3
    assert stats.gisted == 0
    assert gister.calls == []  # nothing eligible: the distiller is never dialled


def test_newest_read_per_path_wins_the_gist() -> None:
    doc_v1 = "OLD spec text. " * 300
    doc_v2 = "NEW spec text. " * 300
    conv = Conversation()
    _add_read(conv, "docs/spec.md", doc_v1)
    _add_read(conv, "docs/spec.md", doc_v2)
    _add_read(conv, "b.py", "x" * 500)
    _add_read(conv, "c.py", "y" * 500)
    gister = _SpyGister({"docs/spec.md": "the spec facts"})
    stats = compact_old_tool_results(conv, max_total_bytes=1200, keep_recent=2, gister=gister)
    assert stats.elided == 2
    assert stats.gisted == 1
    assert len(gister.calls) == 1 and len(gister.calls[0]) == 1
    assert gister.calls[0][0].content.startswith("NEW spec text")
    # The newer read keeps the gist; the older one gets the bare marker.
    contents = _contents(conv)
    assert contents[0].startswith(ELISION_PREFIX)
    assert not contents[0].startswith(ELISION_GIST_PREFIX)
    assert contents[1].startswith(ELISION_GIST_PREFIX)


def test_continued_pressure_demotes_gists_oldest_first() -> None:
    doc = "authoritative spec. " * 300
    conv = Conversation()
    _add_read(conv, "docs/spec.md", doc)
    _add_read(conv, "b.py", "x" * 500)
    _add_read(conv, "c.py", "y" * 500)
    gister = _SpyGister({"docs/spec.md": "spec facts " * 30})
    first = compact_old_tool_results(conv, max_total_bytes=1800, keep_recent=2, gister=gister)
    assert (first.gisted, first.demoted) == (1, 0)
    gist_ph = _contents(conv)[0]
    assert gist_ph.startswith(ELISION_GIST_PREFIX)
    # Re-running under the same budget is a no-op (idempotent).
    again = compact_old_tool_results(conv, max_total_bytes=1800, keep_recent=2, gister=gister)
    assert (again.elided, again.gisted, again.demoted) == (0, 0, 0)
    assert _contents(conv)[0] == gist_ph
    # A tighter budget demotes the gist to the bare marker; the bound holds.
    tighter = compact_old_tool_results(conv, max_total_bytes=1100, keep_recent=2, gister=gister)
    assert tighter.demoted == 1
    got = _contents(conv)[0]
    assert got.startswith(ELISION_PREFIX)
    assert not got.startswith(ELISION_GIST_PREFIX)
    assert "docs/spec.md" in got


def test_gist_longer_than_content_stays_bare() -> None:
    # A gist placeholder that would not shrink the block is pointless.
    text = "z" * 2100  # just over GIST_MIN_SOURCE_CHARS
    conv = Conversation()
    # raw (non-JSON) read result: read_file_text_from_result falls back verbatim
    _add_call(conv, "read_file", {"path": "a.md"}, text)
    _add_read(conv, "b.py", "x" * 500)
    _add_read(conv, "c.py", "y" * 500)
    gister = _SpyGister({"a.md": "g" * 3000})  # clipped to GIST_MAX_CHARS, still fits
    stats = compact_old_tool_results(conv, max_total_bytes=2000, keep_recent=2, gister=gister)
    assert stats.gisted == 1
    assert len(_contents(conv)[0]) < 2100


def test_a_gist_the_budget_cannot_hold_is_never_reported_as_kept() -> None:
    """A gist costing more than the plan's headroom was applied, demoted back
    to the bare marker in the same pass, and still counted: the run line read
    "1 kept as distilled gists" over a marker holding none."""
    conv = Conversation()
    _add_call(conv, "read_file", {"path": "a.md"}, "z" * 2100)
    _add_read(conv, "b.py", "x" * 500)
    _add_read(conv, "c.py", "y" * 500)
    gister = _SpyGister({"a.md": "g" * 3000})

    stats = compact_old_tool_results(conv, max_total_bytes=1500, keep_recent=2, gister=gister)

    assert (stats.gisted, stats.demoted) == (0, 0)
    assert not _contents(conv)[0].startswith(ELISION_GIST_PREFIX)


def test_elision_gist_placeholder_shares_the_elision_prefix() -> None:
    ph = elision_gist_placeholder("docs/a.md", "facts")
    assert ph.startswith(ELISION_PREFIX)
    assert ph.startswith(ELISION_GIST_PREFIX)
    assert "docs/a.md" in ph and "gist: facts" in ph


def test_parse_gist_lines_is_tolerant() -> None:
    text = (
        "- `docs/a.md`: A requires X; threshold 80\n"
        "docs/b.md: B forbids Y\n"
        "unknown.md: ignored\n"
        "no separator line\n"
        "docs/c.md:\n"  # empty gist: skipped
    )
    got = parse_gist_lines(text, paths=["docs/a.md", "docs/b.md", "docs/c.md"])
    assert got == {"docs/a.md": "A requires X; threshold 80", "docs/b.md": "B forbids Y"}


def test_read_file_text_from_result_unwraps_shapes() -> None:
    assert read_file_text_from_result(json.dumps({"content": "text", "size": 4})) == "text"
    truncated = json.dumps(
        {"_tool_result_truncated": True, "head": "the head", "tool": "read_file"}
    )
    assert read_file_text_from_result(truncated) == "the head"
    assert read_file_text_from_result(json.dumps({"error": "Not a file: x"})) == ""
    assert read_file_text_from_result("plain non-json payload") == "plain non-json payload"


def test_stats_carry_gist_and_demotion_identities() -> None:
    doc = "authoritative spec. " * 300
    conv = Conversation()
    _add_read(conv, "docs/spec.md", doc)
    _add_read(conv, "b.py", "x" * 500)
    _add_read(conv, "c.py", "y" * 500)
    gister = _SpyGister({"docs/spec.md": "spec facts " * 30})
    first = compact_old_tool_results(conv, max_total_bytes=1800, keep_recent=2, gister=gister)
    assert first.elided_calls == ("read_file docs/spec.md",)
    assert first.gist_paths == ("docs/spec.md",)
    assert first.demoted_paths == ()
    tighter = compact_old_tool_results(conv, max_total_bytes=1100, keep_recent=2, gister=gister)
    assert tighter.demoted == 1
    assert tighter.demoted_paths == ("docs/spec.md",)
    assert tighter.elided_calls == ()
    assert tighter.gist_paths == ()


def test_gist_placeholder_identity_matches_bare_for_long_paths() -> None:
    """The gist placeholder names the call through the same truncated identity
    as the bare marker, so a gist->bare demotion of a >120-char path is not
    re-reported as a fresh elision by the conversation differ."""
    import re

    from agent6.workflows._compaction import call_label, elision_placeholder

    long_path = "docs/" + "d" * 130 + ".md"
    ident = re.compile(r": the result of (.+?) was replaced")
    label = call_label("read_file", {"path": long_path})
    gist_m = ident.search(elision_gist_placeholder(label, "the gist"))
    bare_m = ident.search(elision_placeholder("read_file", {"path": long_path}))
    assert gist_m is not None and bare_m is not None
    assert gist_m.group(1) == bare_m.group(1) == call_label("read_file", {"path": long_path})


def test_gist_placeholder_identity_matches_bare_for_a_ranged_read() -> None:
    """The placeholder rebuilt the identity from the path alone, so a gisted
    read_file with offset/limit carried a DIFFERENT identity than its bare
    marker: the differ read the demotion as a fresh elision and every surface
    reported a second, phantom marker for one read."""
    import re

    from agent6.workflows._compaction import call_label, elision_placeholder

    conv = Conversation()
    _add_call(conv, "read_file", {"path": "a.py", "start_line": 100, "limit": 500}, "z" * 4000)
    _add_read(conv, "b.py", "y" * 500)
    _add_read(conv, "c.py", "y" * 500)
    compact_old_tool_results(
        conv, max_total_bytes=1800, keep_recent=2, gister=_SpyGister({"a.py": "the gist " * 10})
    )

    text = str(conv.turns[1].items[0].content)  # pyright: ignore[reportAttributeAccessIssue]
    assert text.startswith(ELISION_GIST_PREFIX)
    ident = re.compile(r": the result of (.+?) was replaced")
    got = ident.search(text)
    full_input = {"path": "a.py", "start_line": 100, "limit": 500}
    bare = ident.search(elision_placeholder("read_file", full_input))
    assert got is not None and bare is not None
    assert got.group(1) == bare.group(1)  # gist and bare marker share ONE identity
    assert got.group(1) == call_label("read_file", full_input)
    assert "start_line=100" in got.group(1)  # the range is part of the identity
