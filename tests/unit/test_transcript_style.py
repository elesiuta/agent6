# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The shared transcript renderer (item_lines): structure, the #1 neutral-detail fix,
and the collapsed/expanded/hidden detail levels."""

from __future__ import annotations

from dataclasses import replace

from agent6.viewmodel.transcript import TranscriptItem
from agent6.viewmodel.transcript_style import TAIL_CLIP, item_lines


def test_failed_tool_detail_is_a_neutral_span_not_the_fail_colour() -> None:
    item = TranscriptItem("tool", name="apply_edit", arg="x.py", ok=False, detail="err " * 100)
    result = item_lines(item, detail="collapsed")[1]
    assert result[0][1] == "fail"  # the RESULT glyph carries the fail colour
    assert result[1][1] == "detail"  # the detail is its OWN neutral span (the #1 fix)
    assert "err" in result[1][0]


def test_multiline_marker_renders_a_headline_plus_indented_detail() -> None:
    # A parallel dispatch/join marker carries detail lines under the divider
    # headline; the renderer keeps the first line as the ── … ── divider and
    # indents the rest (one place, so cli/tui/web all show it the same).
    item = TranscriptItem("marker", body="joined group p1: 2 lane(s)\njoined  l1\nconflict  l2")
    lines = item_lines(item, detail="collapsed")
    assert lines[0] == [("── joined group p1: 2 lane(s) ──", "marker")]
    assert lines[1] == [("   joined  l1", "marker")]
    assert lines[2] == [("   conflict  l2", "marker")]


def test_long_multiline_detail_clips_to_reason_plus_more_note_when_collapsed() -> None:
    detail = "old_string not found\n" + "\n".join(f"line {i}" for i in range(50))
    item = TranscriptItem("tool", name="apply_edit", ok=False, detail=detail)
    result = item_lines(item, detail="collapsed")[1]
    assert result[1] == ("old_string not found", "detail")  # first line only, neutral
    assert result[2][1] == "more" and "50 more lines" in result[2][0]


def test_expanded_shows_the_full_tool_detail() -> None:
    detail = "old_string not found\n" + "\n".join(f"line {i}" for i in range(50))
    item = TranscriptItem("tool", name="apply_edit", ok=False, detail=detail)
    lines = item_lines(item, detail="expanded")
    assert lines[1] == [("  └ ", "fail"), ("old_string not found", "detail")]
    assert all(span[1] == "detail" for line in lines[2:52] for span in line)  # every line neutral
    assert "line 49" in lines[51][0][0]  # the last detail line is present, no "+N more"


def test_short_single_line_detail_is_not_clipped_and_has_no_more_note() -> None:
    item = TranscriptItem("tool", name="read_file", ok=True, detail="2440 bytes")
    result = item_lines(item, detail="collapsed")[1]
    assert result[1] == ("2440 bytes", "detail")
    assert len(result) == 2  # no "more" span


def test_tool_tail_clipped_to_one_length_and_neutral() -> None:
    item = TranscriptItem("tool", name="run_command", ok=False, detail="d", tail="x" * 500)
    tail = item_lines(item, detail="collapsed")[-1]
    assert tail[0][1] == "tail"
    # tail + the 4-space relative indent + the clip ellipsis
    assert len(tail[0][0]) <= TAIL_CLIP + 5
    assert tail[0][0].endswith("…")  # a clipped tail says so


def test_tool_tail_expands_to_full_lines() -> None:
    # Expanded shows the WHOLE captured output line-by-line; without this the
    # detail toggle was a no-op exactly where users want more (command output).
    out = "\n".join(f"line {i}" for i in range(6)) + "\n" + "y" * 300
    item = TranscriptItem("tool", name="run_command", ok=True, detail="d", tail=out)
    collapsed = item_lines(item, detail="collapsed")
    expanded = item_lines(item, detail="expanded")
    assert expanded != collapsed
    tail_text = "".join(span[0] for line in expanded for span in line if span[1] == "tail")
    assert "line 5" in tail_text and "y" * 300 in tail_text  # nothing clipped


def test_thinking_detail_levels() -> None:
    item = TranscriptItem("thinking", body="plan the fix\nb\nc")
    assert item_lines(item, detail="hidden") == []  # omitted entirely
    collapsed = item_lines(item, detail="collapsed")
    # Collapsed = the marker (its own accent span) + the FIRST LINE of the
    # reasoning as a summary + a more-count, so it still says what the model
    # is thinking about.
    assert collapsed[0][0][1] == "think-marker"
    assert collapsed[0][1][1] == "thinking" and "plan the fix" in collapsed[0][1][0]
    assert "b" not in collapsed[0][1][0].split("plan the fix")[-1]  # only the first line
    assert collapsed[0][2][1] == "more" and "+2 more lines" in collapsed[0][2][0]
    # A single-line thought has no more-count; a long first line is clipped.
    single = item_lines(TranscriptItem("thinking", body="only line"), detail="collapsed")
    assert len(single[0]) == 2 and "only line" in single[0][1][0]
    long = item_lines(TranscriptItem("thinking", body="x" * 400), detail="collapsed")
    assert long[0][1][0].endswith("…") and len(long[0][1][0]) < 200
    expanded = item_lines(item, detail="expanded")
    assert expanded[0][0][1] == "think-marker"
    assert expanded[0][1][1] == "thinking" and expanded[0][1][0] == "plan the fix"
    assert [line[0][0].strip() for line in expanded[1:]] == ["b", "c"]


def test_tool_head_is_one_call_span_plus_arg() -> None:
    item = TranscriptItem("tool", name="read_file", arg="a.py", ok=True, detail="ok")
    head = item_lines(item, detail="collapsed")[0]
    assert head[0][1] == "call" and "read_file" in head[0][0]
    assert head[1][1] == "arg" and "a.py" in head[1][0]


def test_every_line_is_a_single_line() -> None:
    """item_lines' contract is one entry per rendered LINE; a span embedding
    newlines desyncs every consumer that counts entries (the TUI scroll-anchor
    math splits the content on newlines and pairs it with per-entry counters).
    Expanded thinking and a multi-line finish summary were the violators."""
    multi = "first thought\nsecond thought\nthird"
    for item in (
        TranscriptItem("thinking", body=multi),
        TranscriptItem("done", ok=True, body="all good\non two lines", detail="finish_session"),
    ):
        for detail in ("expanded", "collapsed", "hidden"):
            for line in item_lines(item, detail=detail):  # type: ignore[arg-type]
                for chunk, _style in line:
                    assert "\n" not in chunk, (item.kind, detail, chunk)


def test_expanded_thinking_renders_every_body_line() -> None:
    body = "alpha\nbeta\ngamma"
    lines = item_lines(TranscriptItem("thinking", body=body), detail="expanded")
    text = ["".join(c for c, _s in line) for line in lines]
    assert any("alpha" in t for t in text)
    assert any("beta" in t for t in text)
    assert any("gamma" in t for t in text)
    assert len(text) == 3


def test_hidden_omits_tool_items_and_cycling_back_restores_them() -> None:
    """The least-noise level reads as pure dialogue: hidden omits tool items
    exactly like thinking. The item itself survives (rendering is a pure
    function of the fold), so cycling back restores it -- nothing is lost."""
    tool = TranscriptItem("tool", name="read_file", arg="a.py", ok=True, detail="12 bytes")
    assert item_lines(tool, detail="hidden") == []
    assert item_lines(tool, detail="collapsed")
    assert item_lines(tool, detail="expanded")
    # The dialogue kinds stay at every level.
    assert item_lines(TranscriptItem("text", body="hello"), detail="hidden")
    assert item_lines(TranscriptItem("operator", body="go on"), detail="hidden")


def test_verify_head_has_its_own_style() -> None:
    """The gate reads differently from ordinary tools: run_verify_command's
    call head styles as "verify", everything else as "call"."""
    verify = TranscriptItem("tool", name="run_verify_command", ok=True, detail="✓ pass · 0.2s")
    assert item_lines(verify, detail="collapsed")[0][0][1] == "verify"
    other = TranscriptItem("tool", name="read_file", arg="a.py", ok=True, detail="ok")
    assert item_lines(other, detail="collapsed")[0][0][1] == "call"


def test_an_in_flight_tool_is_one_running_line() -> None:
    """A call with no result yet renders as its head marked running, at every
    detail level that shows tools; never the fail glyph over an empty result."""
    item = TranscriptItem("tool", name="run_command", arg="sleep 60", ok=None, call_id="7")
    (line,) = item_lines(item, detail="collapsed")
    assert "".join(text for text, _style in line) == "→ run_command  sleep 60  · running"
    assert "fail" not in {style for _text, style in line}
    assert item_lines(item, detail="expanded") == [line]
    assert item_lines(item, detail="hidden") == []
    (line,) = item_lines(replace(item, detail="awaiting approval"), detail="collapsed")
    assert "".join(text for text, _style in line) == "→ run_command  sleep 60  · awaiting approval"


def test_the_done_badge_keeps_the_gates_tri_state() -> None:
    """`all_passed` is null for a run no gate judged and for the operator's own
    stop or undo. Flattening it to a bool painted those in the failure colour
    and rendered a gateless finish byte-identically to a finish over a RED gate
    (which exits 4)."""
    from agent6.viewmodel.transcript import TranscriptItem
    from agent6.viewmodel.transcript_style import item_lines

    def badge(ok: bool | None, name: str) -> tuple[str, str]:
        item = TranscriptItem("done", ok=ok, name=name, detail="1 tool")
        return item_lines(item, detail="collapsed")[1][0]

    assert badge(True, "") == ("● done", "done-ok")
    assert badge(False, "finished") == ("● finished", "done-fail")
    assert badge(None, "finished") == ("● finished", "done-neutral")
    assert badge(None, "stopped") == ("● stopped", "done-neutral")
