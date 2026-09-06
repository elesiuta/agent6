# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Unit tests for context compaction (oldest tool_result elision)."""

from __future__ import annotations

from typing import Any

from agent6.workflows._compaction import (
    call_label,
    compact_old_tool_results,
    context_chars,
    elision_placeholder,
    parse_checkoff,
    strip_checkoff,
)
from agent6.workflows._conversation import Conversation, ToolResultItem, UserTurn


def _add_exchange(conv: Conversation, *calls: tuple[str, dict[str, Any], str]) -> None:
    """One assistant turn of (tool name, input, result content) calls plus its
    results turn. Ids are unique per conversation position."""
    base = len(conv)
    turn = conv.assistant(
        [
            {"type": "tool_use", "id": f"t{base}-{i}", "name": name, "input": tool_input}
            for i, (name, tool_input, _content) in enumerate(calls)
        ]
    )
    conv.results(
        [
            ToolResultItem(tool_use_id=tu.id, content=content, for_call=tu)
            for tu, (_name, _input, content) in zip(turn.tool_uses, calls, strict=True)
        ]
    )


def _reads(conv: Conversation, *contents: str, name: str = "read_file") -> None:
    """One single-call exchange per content string."""
    for c in contents:
        _add_exchange(conv, (name, {"path": "x.py"}, c))


def _result_contents(conv: Conversation) -> list[str]:
    return [
        item.content
        for turn in conv.turns
        if isinstance(turn, UserTurn)
        for item in turn.items
        if isinstance(item, ToolResultItem)
    ]


def test_parse_checkoff_valid_block() -> None:
    text = (
        "Progress summary here.\n\n"
        '```checkoff\n{"completed_ids": ["01A", "01B"], "new_tasks": ["fix the parser", ""]}\n```'
    )
    completed, new_tasks = parse_checkoff(text)
    assert completed == ["01A", "01B"]
    assert new_tasks == ["fix the parser"]  # empty title filtered


def test_parse_checkoff_absent_or_malformed() -> None:
    assert parse_checkoff("no block at all") == ([], [])
    assert parse_checkoff("```checkoff\nnot json\n```") == ([], [])
    assert parse_checkoff('```checkoff\n["not", "a", "dict"]\n```') == ([], [])
    # non-string ids/titles are dropped
    assert parse_checkoff('```checkoff\n{"completed_ids": [1, "ok"], "new_tasks": [2]}\n```') == (
        ["ok"],
        [],
    )


def test_parse_checkoff_present_but_non_list_field_is_total() -> None:
    # A present-but-non-list value (null when nothing completed, a number, a
    # bool) must yield [] -- .get(key, []) returns the value as-is, so the old
    # `for s in None` raised TypeError and crashed the run (deterministically,
    # since the summariser runs at temperature 0.0, making resume re-crash).
    for bad in ("null", "0", "false", '"a string"'):
        assert parse_checkoff(
            f'```checkoff\n{{"completed_ids": {bad}, "new_tasks": {bad}}}\n```'
        ) == (
            [],
            [],
        )


def test_strip_checkoff_removes_block() -> None:
    text = 'the summary\n\n```checkoff\n{"completed_ids": []}\n```'
    assert strip_checkoff(text) == "the summary"
    assert strip_checkoff("no block") == "no block"


def test_context_chars_counts_text_tool_use_and_tool_results() -> None:
    # tier-2's trigger must see content tier-1 does NOT cap (assistant prose,
    # tool_use inputs), not just tool_result bytes.
    conv = Conversation()
    conv.notice("abcd")  # 4
    turn = conv.assistant(
        [
            {"type": "text", "text": "hello"},  # 5
            {"type": "tool_use", "id": "t1", "name": "grep", "input": {"q": "x"}},
        ]
    )
    conv.results(
        [ToolResultItem(tool_use_id="t1", content="RESULT", for_call=turn.tool_uses[0])]  # 6
    )
    total = context_chars(conv)
    # Every value of the tool_use block, not a chosen three: its id and name go
    # on the wire with the input, so they count too.
    assert total == 4 + 5 + 6 + len("t1") + len("grep") + len(str({"q": "x"}))
    assert total > 6


def test_compact_skips_tool_result_smaller_than_placeholder() -> None:
    # Eliding a tool_result already smaller than the placeholder would
    # GROW cumulative size, not shrink it. Such blocks must be left intact.
    from agent6.workflows._compaction import (
        ELISION_PLACEHOLDER as PLACEHOLDER,
    )

    tiny = "x" * 50  # smaller than the placeholder, so eliding it would grow the context
    big = "y" * 5000
    # Oldest-first; keep_recent=2 keeps the last two, and the final results
    # turn is exempt, so the eligible blocks are the two in the first turn.
    conv = Conversation()
    _add_exchange(conv, ("grep", {}, tiny), ("grep", {}, big))
    _add_exchange(conv, ("grep", {}, big), ("grep", {}, big))
    compact_old_tool_results(conv, max_total_bytes=100, keep_recent=2)
    contents = _result_contents(conv)
    # The oldest (tiny) block is eligible but must be skipped, not ballooned;
    # its eligible sibling is elided as normal.
    assert contents[0] == tiny
    assert "elided" in contents[1]
    assert len(tiny) < len(PLACEHOLDER)  # the premise the skip guards


def test_compact_noop_when_under_threshold() -> None:
    conv = Conversation()
    _reads(conv, "small")
    stats = compact_old_tool_results(conv, max_total_bytes=1000)
    assert stats.elided == 0
    assert _result_contents(conv) == ["small"]


def test_compact_elides_oldest_when_over_threshold() -> None:
    # Distinct payloads: identical ones would be deduplicated before elision.
    a, b, c = "a" * 1000, "b" * 1000, "c" * 1000
    conv = Conversation()
    _reads(conv, a, b, c)  # oldest first
    stats = compact_old_tool_results(conv, max_total_bytes=1500, keep_recent=2)
    assert stats.elided == 1
    contents = _result_contents(conv)
    # Oldest replaced with marker; the newer two kept.
    assert "elided" in contents[0]
    assert contents[1] == b
    assert contents[2] == c


def test_compact_preserves_keep_recent_floor() -> None:
    """Even when over threshold, the newest `keep_recent` entries
    are never elided."""
    bodies = [ch * 10_000 for ch in "abcde"]  # distinct: dedup must not fire
    conv = Conversation()
    _reads(conv, *bodies)
    stats = compact_old_tool_results(conv, max_total_bytes=100, keep_recent=2)
    # 3 oldest elided, 2 most recent preserved.
    assert stats.elided == 3
    contents = _result_contents(conv)
    assert all("elided" in c for c in contents[:3])
    assert contents[3] == bodies[3]
    assert contents[4] == bodies[4]


def test_compact_idempotent_on_already_elided() -> None:
    """Running compaction twice doesn't double-elide or churn."""
    bodies = [ch * 1000 for ch in "abcd"]  # distinct: dedup must not fire
    conv = Conversation()
    _reads(conv, *bodies)
    e1 = compact_old_tool_results(conv, max_total_bytes=1500, keep_recent=2)
    e2 = compact_old_tool_results(conv, max_total_bytes=1500, keep_recent=2)
    assert e1.elided == 2  # oldest 2 elided
    assert e2.elided == 0  # no further work needed on second pass


def test_compact_never_elides_unseen_results_in_final_turn() -> None:
    """Compaction runs at top-of-iteration, BEFORE the provider call that
    delivers the final turn's tool_results: the model has never seen them.
    A turn with 3+ large results must not have its oldest same-turn results
    replaced by the "re-call the tool" placeholder (which previously sent the
    model into a paid re-call cycle chasing content it never received)."""
    big = "x" * 10_000
    conv = Conversation()
    conv.notice("task")
    _add_exchange(conv, *[("read_file", {}, big)] * 3)
    stats = compact_old_tool_results(conv, max_total_bytes=100, keep_recent=2)
    assert stats.elided == 0
    assert _result_contents(conv) == [big, big, big]


def test_compact_elides_seen_results_but_protects_final_turn() -> None:
    # Results the model has already consumed (a later assistant turn exists)
    # stay eligible; only the undelivered final results turn is exempt.
    seen = [(f"s{i}" * 5_000) for i in range(3)]  # distinct: dedup must not fire
    fresh = [(f"f{i}" * 5_000) for i in range(3)]
    conv = Conversation()
    conv.notice("task")
    _add_exchange(conv, *[("read_file", {}, c) for c in seen])  # seen: answered below
    _add_exchange(conv, *[("read_file", {}, c) for c in fresh])  # unseen: awaiting delivery
    stats = compact_old_tool_results(conv, max_total_bytes=100, keep_recent=2)
    assert stats.elided == 3
    contents = _result_contents(conv)
    assert all("elided" in c for c in contents[:3])
    assert contents[3:] == fresh


def test_compact_never_elides_undelivered_results_behind_a_steer_message() -> None:
    """Undelivered tool_results are not always the final turn: an operator
    steer (or a pre-call nudge) appends a trailing user turn after them, so
    they sit at index -2. They are still unseen (the delivering provider call
    runs after this compaction), so keying on the final index alone let their
    older same-turn blocks be elided into a paid re-call cycle. The exemption
    tracks the last tool_result-bearing turn, which still holds here."""
    big = "x" * 10_000
    conv = Conversation()
    conv.notice("task")
    _add_exchange(conv, *[("read_file", {}, big)] * 3)  # unseen: awaiting delivery
    conv.notice("steer: focus on the parser")
    stats = compact_old_tool_results(conv, max_total_bytes=100, keep_recent=2)
    assert stats.elided == 0
    assert _result_contents(conv) == [big, big, big]


def test_restart_notice_is_dag_aware() -> None:
    """The tier-2 summarise-and-restart notice must point the worker at its
    durable task DAG so cross-compaction task state is recovered."""
    from agent6.prompts.revision import context_restart_notice

    for mode in ("run", "plan"):
        notice = context_restart_notice(mode)
        # The real tool is ``list_tasks`` (no ``dag_`` prefix); the notice must
        # name it exactly or the post-compaction recovery call 404s.
        assert "list_tasks" in notice
        assert "dag_list_tasks" not in notice
        assert "DAG" in notice
        # Still tells the worker not to start over.
        assert "Do NOT start over" in notice
    # ask/machine/agent have no DAG tools: instructing list_tasks there burns a
    # turn on an unknown-tool error, so the DAG paragraph must be absent.
    for mode in ("ask", "machine", "agent"):
        notice = context_restart_notice(mode)
        assert "list_tasks" not in notice
        assert "Do NOT start over" in notice
        assert notice.endswith("PROGRESS SUMMARY:\n")


# --- read-waste reduction: identity placeholders + hot-file protection ------


def test_elision_placeholder_names_the_call() -> None:
    from agent6.workflows._compaction import ELISION_PREFIX, elision_placeholder

    p = elision_placeholder("read_file", {"path": "src/x.py", "start_line": 10, "limit": 50})
    assert p.startswith(ELISION_PREFIX)
    assert "read_file src/x.py" in p and "start_line=10" in p
    g = elision_placeholder("find_definition", {"symbol": "foo"})
    assert "find_definition foo" in g
    # Unknown pairing (orphan result) falls back to the generic marker.
    from agent6.workflows._compaction import ELISION_PLACEHOLDER

    assert elision_placeholder("", None) == ELISION_PLACEHOLDER
    assert elision_placeholder("read_file", "not-a-dict") == ELISION_PLACEHOLDER
    # A pathological arg is clipped, keeping the placeholder short.
    long = elision_placeholder("read_file", {"path": "x" * 5000})
    assert len(long) < 500


def test_recently_edited_paths_extraction() -> None:
    from agent6.workflows._compaction import recently_edited_paths

    unified = "--- a/pkg/mod.py\n+++ b/pkg/mod.py\n@@ -1,1 +1,1 @@\n-a\n+b\n"
    v4a = "*** Begin Patch\n*** Update File: pkg/v4a.py\n@@\n-a\n+b\n*** End Patch\n"
    conv = Conversation()
    _add_exchange(conv, ("apply_edit", {"path": "edited.py", "edits": []}, "ok"))
    _add_exchange(conv, ("apply_patch", {"path": "explicit.py", "patch": "x"}, "ok"))
    _add_exchange(conv, ("apply_patch", {"path": "", "patch": unified}, "ok"))
    _add_exchange(conv, ("apply_patch", {"patch": v4a}, "ok"))
    _add_exchange(conv, ("read_file", {"path": "only-read.py"}, "ok"))
    got = recently_edited_paths(conv)
    assert got == frozenset({"edited.py", "explicit.py", "pkg/mod.py", "pkg/v4a.py"})
    # The window is per assistant TURN: an edit older than last_turns drops out.
    conv2 = Conversation()
    _add_exchange(conv2, ("apply_edit", {"path": "old.py", "edits": []}, "ok"))
    _add_exchange(conv2, ("read_file", {"path": "a"}, "ok"))
    _add_exchange(conv2, ("read_file", {"path": "b"}, "ok"))
    assert recently_edited_paths(conv2, last_turns=2) == frozenset()


def test_compact_elides_protected_reads_last_but_bound_still_holds() -> None:
    def build() -> Conversation:
        conv = Conversation()
        _add_exchange(conv, ("read_file", {"path": "hot.py"}, "H" * 1000))
        _add_exchange(conv, ("read_file", {"path": "cold.py"}, "C" * 1000))
        _add_exchange(conv, ("grep", {"pattern": "x"}, "G" * 1000))
        _add_exchange(conv, ("list_dir", {"path": "."}, "L" * 1000))
        return conv

    # Budget forces ONE elision: with hot.py protected, the (older) hot read
    # survives and the cold read goes first.
    conv = build()
    n = compact_old_tool_results(
        conv, max_total_bytes=3500, keep_recent=2, protect_paths=frozenset({"hot.py"})
    )
    assert n.elided == 1
    contents = _result_contents(conv)
    assert contents[0] == "H" * 1000
    assert "cold.py" in contents[1]
    # Tighter budget: protection is a priority, not an exemption; the hot read
    # is elided too and the bound holds.
    conv2 = build()
    n2 = compact_old_tool_results(
        conv2, max_total_bytes=2500, keep_recent=2, protect_paths=frozenset({"hot.py"})
    )
    assert n2.elided == 2
    assert "hot.py" in _result_contents(conv2)[0]


def test_compact_placeholder_carries_tool_identity() -> None:
    conv = Conversation()
    _add_exchange(conv, ("read_file", {"path": "src/lib.py"}, "X" * 1000))
    _add_exchange(conv, ("grep", {"pattern": "q"}, "Y" * 1000))
    _add_exchange(conv, ("list_dir", {"path": "."}, "Z" * 1000))
    _add_exchange(conv, ("outline", {"path": "a.py"}, "W" * 1000))
    n = compact_old_tool_results(conv, max_total_bytes=3000, keep_recent=2)
    assert n.elided >= 1
    elided = _result_contents(conv)[0]
    assert elided.startswith("<elided by context compaction")
    assert "read_file src/lib.py" in elided


def test_call_label_identities() -> None:
    assert call_label("read_file", {"path": "src/foo.py"}) == "read_file src/foo.py"
    assert (
        call_label("read_file", {"path": "a.py", "start_line": 10, "limit": 40})
        == "read_file a.py (start_line=10, limit=40)"
    )
    assert call_label("list_dir", {"path": "src"}) == "list_dir src"
    # Every tool with an identifying argument, not a hand-listed few: a
    # compacted "run_command" with no command told the model nothing about
    # whether it had already run the suite, and searching moved there.
    assert call_label("run_command", {"argv": ["pytest", "-x", "tests/t.py"]}) == (
        "run_command pytest -x tests/t.py"
    )
    assert call_label("apply_edit", {"path": "src/a.py", "edits": []}) == "apply_edit src/a.py"
    assert call_label("use_skill", {"name": "debugging"}) == "use_skill debugging"
    assert call_label("read_background", {"id": "bg1"}) == "read_background bg1"
    assert call_label("fetch", {"url": "https://x.test/s"}) == "fetch https://x.test/s"
    # An argv is rendered as a command line, so a quoted pattern stays readable.
    assert call_label("run_command", {"argv": ["rg", "-n", "def f"]}) == (
        "run_command rg -n 'def f'"
    )
    # Nothing identifying: the bare name, not an invented hint.
    assert call_label("finish_session", {"summary": "done"}) == "finish_session"
    assert call_label("find_definition", {"symbol": "Foo"}) == "find_definition Foo"
    assert call_label("frobnicate", {"x": 1}) == "frobnicate"
    assert call_label("frobnicate", None) == "frobnicate"
    assert call_label("", None) == ""
    long_path = "d/" * 80
    assert call_label("read_file", {"path": long_path}) == f"read_file {long_path[:120]}..."


def test_elision_placeholder_unchanged_by_label_refactor() -> None:
    """Pins the exact placeholder bytes: the call_label extraction must not
    drift the prompt copy the idempotency walk and the model both key on."""
    assert elision_placeholder("read_file", {"path": "src/foo.py"}) == (
        "<elided by context compaction: the result of read_file src/foo.py was"
        " replaced with this short marker to keep the loop's cumulative input"
        " bounded. If you still need it, re-read only the part you need"
        " (read_file with a targeted start_line/limit); do not re-issue the"
        " identical call.>"
    )


def test_stats_carry_elided_identities() -> None:
    big = "x" * 1000
    conv = Conversation()
    _add_exchange(conv, ("read_file", {"path": "a.py"}, big))
    _add_exchange(conv, ("read_file", {"path": "b.py"}, big))
    _add_exchange(conv, ("read_file", {"path": "c.py"}, big))
    stats = compact_old_tool_results(conv, max_total_bytes=1500, keep_recent=2)
    assert stats.elided == 1
    assert stats.elided_calls == ("read_file a.py",)
    assert stats.gist_paths == ()
    assert stats.demoted_paths == ()


def test_context_chars_counts_a_reasoning_model_s_thinking() -> None:
    """The tier-2 trigger is compared against ~80% of the model's real context
    window, so it has to measure what actually goes on the wire.

    `Conversation.to_wire` sends an assistant turn's raw blocks VERBATIM, and
    nothing strips thinking, so a reasoning model's `{"type": "thinking",
    "thinking": ...}` is in every later request. Counting only text/content/
    tool_use-input scored a turn holding 130,000 chars of thinking as 2, and
    tier-2 summarisation waited on a number that omitted the largest thing in
    the context -- on exactly the models that need it most.
    """
    conv = Conversation()
    conv.assistant(
        [
            {"type": "thinking", "thinking": "R" * 10_000, "signature": "sig"},
            {"type": "text", "text": "ok"},
        ]
    )
    total = context_chars(conv)
    assert total >= 10_000, f"thinking is on the wire but counted as {total}"


def test_context_chars_counts_an_unknown_block_type() -> None:
    """Enumerating known keys is what let thinking go uncounted; a block type
    nobody has met yet must not be free either."""
    conv = Conversation()
    conv.assistant([{"type": "something_new", "payload": "P" * 5_000}])
    assert context_chars(conv) >= 5_000


def test_recent_tail_start_respects_cap_and_boundaries() -> None:
    from agent6.workflows._compaction import recent_tail_start
    from agent6.workflows._conversation import Conversation, ToolResultItem

    conv = Conversation()
    conv.notice("task")
    conv.assistant([{"type": "tool_use", "id": "t1", "name": "read_file", "input": {"path": "a"}}])
    last = conv.turns[-1]
    conv.results([ToolResultItem(tool_use_id="t1", content="x" * 100, for_call=last.tool_uses[0])])  # type: ignore[union-attr]
    conv.assistant([{"type": "text", "text": "y" * 100}])
    turns = conv.turns

    # Cap covering only the final text turn: the tail starts there.
    assert recent_tail_start(turns, 150) == 3
    # Cap reaching the results turn but not its call turn: a results-turn
    # start is unsafe (its call was summarised away), so it advances past it.
    assert recent_tail_start(turns, 210) == 3
    # Cap covering the balanced call+result+text triple keeps all three.
    assert recent_tail_start(turns, 100_000) == 1
    # Cap 0 keeps nothing.
    assert recent_tail_start(turns, 0) == len(turns)
    # A cap smaller than the newest exchange keeps that exchange anyway (its
    # results may be undelivered; paraphrasing them away is the one loss the
    # tail exists to prevent).
    assert recent_tail_start(turns, 50) == 3


def _conv_with_repeated_reads(payload: str) -> Conversation:
    from agent6.workflows._conversation import Conversation, ToolResultItem

    conv = Conversation()
    conv.notice("task")
    for tid in ("t1", "t2", "t3"):
        conv.assistant(
            [{"type": "tool_use", "id": tid, "name": "read_file", "input": {"path": "a.py"}}]
        )
        last = conv.turns[-1]
        conv.results(
            [ToolResultItem(tool_use_id=tid, content=payload, for_call=last.tool_uses[0])]  # type: ignore[union-attr]
        )
    return conv


def test_tier1_dedupes_identical_results_keeping_the_newest() -> None:
    """The same read re-run with identical bytes: older copies become pointer
    placeholders, the newest survives whole. Lossless, so no knob (Claude
    Code dedupes the same way)."""
    from agent6.workflows._compaction import ELISION_PREFIX, compact_old_tool_results
    from agent6.workflows._conversation import ToolResultItem, UserTurn

    payload = "x" * 1_000
    conv = _conv_with_repeated_reads(payload)
    stats = compact_old_tool_results(conv, max_total_bytes=1_500, keep_recent=1)
    # Both older copies dedupe; only the newest (t3) keeps the bytes.
    assert stats.deduped == 2
    result_turns = [t for t in conv.turns if isinstance(t, UserTurn) and t.items]
    contents = [
        item.content for t in result_turns for item in t.items if isinstance(item, ToolResultItem)
    ]
    assert sum(c == payload for c in contents) == 1  # only the newest copy keeps the bytes
    assert any(c.startswith(f"{ELISION_PREFIX} (duplicate)") for c in contents)
    assert stats.deduped_calls == ("read_file a.py", "read_file a.py")


def test_a_duplicate_marker_claims_no_copy_the_same_pass_elides() -> None:
    """The duplicate marker sent the model to "the newer result", which the
    elision pass in the same call can replace with a bare marker: a pointer to
    content nothing holds any more."""
    from agent6.workflows._compaction import ELISION_PREFIX, compact_old_tool_results
    from agent6.workflows._conversation import AssistantTurn

    payload = "x" * 4_000
    conv = Conversation()
    conv.notice("task")
    calls = [("t1", "read_file", {"path": "a.py"}), ("t2", "read_file", {"path": "a.py"})]
    calls += [(f"c{i}", "run_command", {"command": f"echo {i}"}) for i in range(3)]
    for tid, name, args in calls:
        conv.assistant([{"type": "tool_use", "id": tid, "name": name, "input": args}])
        last = conv.turns[-1]
        assert isinstance(last, AssistantTurn)
        conv.results([ToolResultItem(tool_use_id=tid, content=payload, for_call=last.tool_uses[0])])
    # The duplicated read is older than the kept tail, so its newest copy is a
    # candidate for elision in the same pass that deduped the older one.
    compact_old_tool_results(conv, max_total_bytes=9_000, keep_recent=2)

    result_turns = [t for t in conv.turns if isinstance(t, UserTurn) and t.items]
    contents = [
        item.content for t in result_turns for item in t.items if isinstance(item, ToolResultItem)
    ]
    assert any(c.startswith(f"{ELISION_PREFIX} (duplicate)") for c in contents)
    assert payload not in contents[:2], "the deduped read still holds a full copy"
    assert not any("newer result" in c for c in contents)


def test_a_duplicate_marker_never_grows_the_result_it_replaces() -> None:
    """The marker carries the call's arguments, so a long path makes it longer
    than a small result: writing it inflated the conversation while `deduped`
    reported a saving."""
    from agent6.workflows._compaction import compact_old_tool_results
    from agent6.workflows._conversation import AssistantTurn

    payload = "z" * 210  # over _DEDUP_MIN_CHARS, under the marker's own length
    conv = Conversation()
    conv.notice("task")
    for tid in ("t1", "t2", "t3"):
        conv.assistant(
            [{"type": "tool_use", "id": tid, "name": "read_file", "input": {"path": "a.py"}}]
        )
        last = conv.turns[-1]
        assert isinstance(last, AssistantTurn)
        conv.results([ToolResultItem(tool_use_id=tid, content=payload, for_call=last.tool_uses[0])])
    before = sum(
        len(item.content)
        for turn in conv.turns
        if isinstance(turn, UserTurn)
        for item in turn.items
        if isinstance(item, ToolResultItem)
    )

    stats = compact_old_tool_results(conv, max_total_bytes=100, keep_recent=1)

    after = sum(
        len(item.content)
        for turn in conv.turns
        if isinstance(turn, UserTurn)
        for item in turn.items
        if isinstance(item, ToolResultItem)
    )
    assert after <= before, f"compaction grew the conversation: {before} -> {after}"
    assert stats.deduped == 0


def test_tier1_dedup_alone_can_satisfy_the_budget() -> None:
    """When freeing duplicates gets the total under the threshold, nothing
    real is elided."""
    from agent6.workflows._compaction import compact_old_tool_results

    payload = "y" * 1_000
    conv = _conv_with_repeated_reads(payload)
    stats = compact_old_tool_results(conv, max_total_bytes=2_500, keep_recent=1)
    assert stats.deduped >= 1
    assert stats.elided == 0


def test_tier1_dedup_skips_small_and_different_results() -> None:
    from agent6.workflows._compaction import compact_old_tool_results
    from agent6.workflows._conversation import Conversation, ToolResultItem

    conv = Conversation()
    conv.notice("task")
    for tid, body in (("t1", "tiny"), ("t2", "tiny"), ("t3", "z" * 900), ("t4", "w" * 900)):
        conv.assistant(
            [{"type": "tool_use", "id": tid, "name": "read_file", "input": {"path": "b.py"}}]
        )
        last = conv.turns[-1]
        conv.results(
            [ToolResultItem(tool_use_id=tid, content=body, for_call=last.tool_uses[0])]  # type: ignore[union-attr]
        )
    stats = compact_old_tool_results(conv, max_total_bytes=100, keep_recent=1)
    # "tiny" is under the dedup floor; the 900-char bodies differ: no dedup.
    assert stats.deduped == 0


def test_strip_old_thinking_clears_all_but_the_newest_turns() -> None:
    """Claude-side thinking eviction behind the keep_thinking_turns knob: old
    assistant turns lose their thinking blocks, the newest keep theirs
    (Anthropic needs the signed block of a pending tool_use)."""
    from agent6.workflows._compaction import strip_old_thinking
    from agent6.workflows._conversation import AssistantTurn, Conversation

    conv = Conversation()
    conv.notice("task")
    for i in range(3):
        conv.assistant(
            [
                {"type": "thinking", "thinking": f"reasoning {i} " + "t" * 100},
                {"type": "text", "text": f"answer {i}"},
            ]
        )
    n_turns, n_chars = strip_old_thinking(conv, keep_turns=1)
    assert n_turns == 2 and n_chars > 200
    assistants = [t for t in conv.turns if isinstance(t, AssistantTurn)]
    kinds = [[b["type"] for b in t.raw_content] for t in assistants]
    assert kinds == [["text"], ["text"], ["thinking", "text"]]
    # Idempotent: nothing left to strip on the older turns.
    assert strip_old_thinking(conv, keep_turns=1) == (0, 0)


def test_strip_thinking_preserves_tool_use_pairing() -> None:
    from agent6.workflows._conversation import Conversation, ToolResultItem

    conv = Conversation()
    conv.notice("task")
    conv.assistant(
        [
            {"type": "thinking", "thinking": "hmm"},
            {"type": "tool_use", "id": "t1", "name": "read_file", "input": {"path": "a"}},
        ]
    )
    last = conv.turns[-1]
    conv.results(
        [ToolResultItem(tool_use_id="t1", content="body", for_call=last.tool_uses[0])]  # type: ignore[union-attr]
    )
    conv.assistant([{"type": "text", "text": "done"}])
    removed = conv.strip_thinking(1)
    assert removed > 0
    wire = conv.to_wire()
    # The tool_use block and its result still pair on the wire.
    assert any(
        b.get("type") == "tool_use" and b.get("id") == "t1"
        for m in wire
        if isinstance(m.get("content"), list)
        for b in m["content"]
    )


def test_restart_notice_re_shows_the_operator_rulings() -> None:
    """A compaction restart re-shows DECISIONS.md between the pins and the
    summary, so a ruling survives the summary that might have dropped it."""
    from agent6.prompts.revision import context_restart_notice

    notice = context_restart_notice("run", pins=("pin one",), decisions="- Q: modal?\n  A: no")
    head, rulings, summary = notice.partition("OPERATOR RULINGS (recorded, still binding):")
    assert "pin one" in head and rulings and "- Q: modal?\n  A: no" in summary
    assert summary.index("A: no") < summary.index("PROGRESS SUMMARY")
    assert "OPERATOR RULINGS" not in context_restart_notice("run")
