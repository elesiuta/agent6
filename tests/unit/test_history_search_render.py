# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""`agent6 history search` renders readable windowed hits, not raw JSON lines."""

from __future__ import annotations

import base64
import json
from pathlib import Path

import pytest

from agent6.ui.cli.history_cmds import (
    _char_span,  # pyright: ignore[reportPrivateUsage]
    _parse_rg_matches,  # pyright: ignore[reportPrivateUsage]
    _render_history_hits,  # pyright: ignore[reportPrivateUsage]
    _session_id_from_path,  # pyright: ignore[reportPrivateUsage]
    _strings_in,  # pyright: ignore[reportPrivateUsage]
    _window,  # pyright: ignore[reportPrivateUsage]
)


def _rg_match(path: str, line: str, col: int) -> str:
    return json.dumps(
        {
            "type": "match",
            "data": {
                "path": {"text": path},
                "lines": {"text": line},
                "submatches": [{"start": col, "end": col + 1}],
            },
        }
    )


def test_window_clips_a_huge_line_around_the_match() -> None:
    line = "x" * 500 + "NEEDLE" + "y" * 500
    out = _window(line, 500)
    assert "NEEDLE" in out
    assert out.startswith("…") and out.endswith("…")
    assert len(out) < 200  # not the whole 1000+ char line


def test_window_collapses_json_escaped_newlines() -> None:
    assert _window("the\\nfirst message here", 0) == "the first message here"


def test_window_decodes_backslashes_not_bare_backslash_space() -> None:
    # A double-encoded newline (a transcript embedding a JSON body) is the chars
    # \\ \\ n; the old naive replace matched the trailing \\n and left the ugly
    # "\ ". Now \\\\ decodes to one backslash first, so no "\ " artifact.
    assert "\\ " not in _window("cmd = tail \\\\nlog NEEDLE", 0)
    # An escaped backslash renders as ONE backslash, and an escaped quote as a
    # quote -- not the raw doubled JSON escapes.
    assert (
        _window('path C:\\\\Users and \\"quoted\\" NEEDLE', 0)
        == 'path C:\\Users and "quoted" NEEDLE'
    )


def test_run_id_from_path_finds_the_run_dir_child() -> None:
    assert _session_id_from_path(Path("/s/runs/deep-poppy-AB/logs.jsonl")) == "deep-poppy-AB"
    assert (
        _session_id_from_path(Path("/s/asks/quiet-fox-CD/transcripts/0003.json")) == "quiet-fox-CD"
    )
    # A state-base ANCESTOR sharing a bucket name must not shadow the real
    # bucket (XDG_STATE_HOME=/mnt/runs/state mislabelled every hit as "state").
    assert (
        _session_id_from_path(Path("/mnt/runs/state/agent6/repo-x/runs/deep-poppy-AB/logs.jsonl"))
        == "deep-poppy-AB"
    )
    assert (
        _session_id_from_path(Path("/mnt/asks/state/agent6/repo-x/asks/quiet-fox-CD/logs.jsonl"))
        == "quiet-fox-CD"
    )


def test_parse_extracts_event_type_and_time_for_logs_jsonl() -> None:
    event = {"ts": "2026-07-12T09:15:30.1Z", "type": "tool.call", "name": "grep"}
    out = _parse_rg_matches(_rg_match("/s/runs/r1/logs.jsonl", json.dumps(event), 40))
    assert len(out) == 1
    assert out[0].session_id == "r1"
    assert out[0].kind == "tool.call"
    assert out[0].when == "09:15:30"


def test_transcripts_share_one_label(capsys: pytest.CaptureFixture[str]) -> None:
    # The same snippet across cumulative transcript snapshots collapses to one
    # (xN) line, labelled "transcript", not per-file.
    lines = "\n".join(
        _rg_match(f"/s/runs/r1/transcripts/000{i}.json", '  "text": "hello NEEDLE"', 12)
        for i in (3, 5, 7)
    )
    hits = _parse_rg_matches(lines)
    assert all(h.kind == "transcript" for h in hits)
    _render_history_hits(hits, Path("/s/runs"))
    out = capsys.readouterr().out
    assert "(x3)" in out  # three identical snapshot hits collapsed
    assert out.count("hello NEEDLE") == 1


def test_event_snippet_windows_inside_the_matched_field(
    capsys: pytest.CaptureFixture[str],
) -> None:
    # A match inside an event's string field snippets the FIELD's prose, not a
    # '"type": "role.thinking_delta", "text": " ...' raw-JSON fragment.
    event = json.dumps(
        {
            "ts": "2026-07-12T09:15:30.1Z",
            "type": "role.thinking_delta",
            "text": "I improved the NEEDLE of one bullet in README.md",
        }
    )
    hits = _parse_rg_matches(_rg_match("/s/runs/r1/logs.jsonl", event, event.index("NEEDLE")))
    assert hits[0].snippet == "I improved the NEEDLE of one bullet in README.md"
    _render_history_hits(hits, Path("/s/runs"))
    assert '"type"' not in capsys.readouterr().out


def test_one_task_in_many_encodings_collapses_to_the_readable_one(
    capsys: pytest.CaptureFixture[str],
) -> None:
    # One task string is stored many ways: the session.start event, manifest.json,
    # the graph's dot label, a per-call transcript body. A search for a word in
    # it must print ONE line per run (the timestamped event) with a count,
    # not the same content in every storage encoding (raw JSON fragments
    # included). The syntax BEFORE the match differs per encoding; the
    # suffix-only content key sees through it.
    task = "Improve the wording of the end banner"
    lines: list[str] = []
    event = json.dumps({"ts": "2026-07-12T07:36:51.1Z", "type": "session.start", "user_task": task})
    lines.append(_rg_match("/s/runs/r1/logs.jsonl", event, event.index("Improve")))
    manifest = json.dumps({"version": 2, "user_task": task})
    lines.append(_rg_match("/s/runs/r1/manifest.json", manifest, manifest.index("Improve")))
    dot = f'"01KX" [label="{task}... [pending]", style=filled]'
    lines.append(_rg_match("/s/runs/r1/graph/graph.dot", dot, dot.index("Improve")))
    body = f'"content": [{{"type": "text", "text": "TASK: {task}"}}]'
    lines.append(_rg_match("/s/runs/r1/transcripts/0003.json", body, body.index("Improve")))

    hits = _parse_rg_matches("\n".join(lines))
    assert len({h.key for h in hits}) == 1  # every encoding shares the content key
    _render_history_hits(hits, Path("/s/runs"))
    out = capsys.readouterr().out
    assert out.count("Improve the wording") == 1  # one representative line
    assert "(x4)" in out  # all four encodings counted
    assert "manifest.json" not in out and "graph.dot" not in out  # internals lose
    assert "session.start" in out  # the timestamped event wins


def _rg_match_bytes(path: str, line: str, needle: str) -> str:
    """An rg-faithful match record: submatch offsets in BYTES, as rg reports."""
    b = line.encode("utf-8")
    start = b.index(needle.encode("utf-8"))
    return json.dumps(
        {
            "type": "match",
            "data": {
                "path": {"text": path},
                "lines": {"text": line},
                "submatches": [{"start": start, "end": start + len(needle.encode("utf-8"))}],
            },
        }
    )


def test_byte_offsets_convert_to_characters_on_non_ascii_lines() -> None:
    # rg reports byte offsets; curly quotes / ellipses earlier in the line made
    # character slicing land off the match, breaking the snippet and the key.
    event = json.dumps(
        {
            "ts": "2026-07-12T09:15:30.1Z",
            "type": "role.text_delta",
            "text": "café ’naïve’ résumé… NEEDLE found mid prose",  # noqa: RUF001
        },
        ensure_ascii=False,
    )
    hits = _parse_rg_matches(_rg_match_bytes("/s/runs/r1/logs.jsonl", event, "NEEDLE"))
    assert "NEEDLE found mid prose" in hits[0].snippet
    assert "needle found mid prose" in hits[0].key


def test_ascii_escaped_and_raw_utf8_encodings_share_one_key() -> None:
    # logs.jsonl is raw UTF-8; manifests/transcripts are ascii-escaped. The
    # \uXXXX decode in the normal form makes both sides one identity.
    task = "Improve the résumé wording NEEDLE of the banner"
    raw_event = json.dumps(
        {"ts": "2026-07-12T07:36:51.1Z", "type": "session.start", "user_task": task},
        ensure_ascii=False,
    )
    escaped_manifest = json.dumps({"version": 2, "user_task": task})  # ascii-escaped
    assert "\\u00e9" in escaped_manifest  # the divergence under test
    hits = _parse_rg_matches(
        _rg_match_bytes("/s/runs/r1/logs.jsonl", raw_event, "NEEDLE")
        + "\n"
        + _rg_match_bytes("/s/runs/r1/manifest.json", escaped_manifest, "NEEDLE")
    )
    assert len(hits) == 2
    assert hits[0].key == hits[1].key


def test_distinct_sentences_ending_with_the_query_stay_distinct(
    capsys: pytest.CaptureFixture[str],
) -> None:
    # A match at end-of-string has no following context; keying on the bare
    # word merged DIFFERENT statements into one line with a wrong (xN).
    e1 = json.dumps(
        {
            "ts": "2026-07-12T08:00:00.1Z",
            "type": "role.text_delta",
            "text": "I deleted the broken NEEDLE",
        }
    )
    e2 = json.dumps(
        {
            "ts": "2026-07-12T09:30:00.1Z",
            "type": "role.text_delta",
            "text": "then I rewrote a completely new NEEDLE",
        }
    )
    hits = _parse_rg_matches(
        _rg_match_bytes("/s/runs/r1/logs.jsonl", e1, "NEEDLE")
        + "\n"
        + _rg_match_bytes("/s/runs/r1/logs.jsonl", e2, "NEEDLE")
    )
    _render_history_hits(hits, Path("/s/runs"))
    out = capsys.readouterr().out
    assert "deleted the broken NEEDLE" in out
    assert "rewrote a completely new NEEDLE" in out
    assert "(x2)" not in out


def test_deeply_nested_json_line_degrades_instead_of_crashing() -> None:
    depth = 100_000
    line = '{"a":' * depth + "1" + "}" * depth
    hits = _parse_rg_matches(_rg_match_bytes("/s/runs/r1/logs.jsonl", line, '"a"'))
    assert len(hits) == 1  # fell back to the raw window, no RecursionError


def test_string_walk_survives_nesting_the_parser_accepted() -> None:
    """json.loads accepts nesting right up to the stack ceiling (where that
    ceiling sits varies by interpreter and arch), so the walk over its result
    must be iterative: one CI leg parsed the 100k-deep line the others refused,
    then blew the recursion limit inside the walker instead."""
    deep: dict[str, object] = {"leaf": "NEEDLE"}
    for _ in range(100_000):
        deep = {"a": deep}
    assert list(_strings_in(deep)) == ["NEEDLE"]


def test_a_line_that_is_not_utf8_still_parses_from_its_bytes() -> None:
    """rg --json carries a line that is not UTF-8 as base64 `bytes` in place
    of `text`; the parser read it as empty, so the hit lost its kind and its
    snippet. The bytes decode with U+FFFD standing in for what is not UTF-8."""
    event = (
        b'{"type": "tool.call", "ts": "2026-09-02T09:15:30+00:00", "name": "run_command",'
        b' "args": {"argv": ["grep", "caf\xe9 NEEDLE"]}}'
    )
    start = event.index(b"NEEDLE")
    rec = {
        "type": "match",
        "data": {
            "path": {"text": "/s/runs/r1/logs.jsonl"},
            "lines": {"bytes": base64.b64encode(event).decode("ascii")},
            "line_number": 1,
            "absolute_offset": 0,
            "submatches": [{"match": {"text": "NEEDLE"}, "start": start, "end": start + 6}],
        },
    }
    out = _parse_rg_matches(json.dumps(rec))
    assert len(out) == 1
    assert (out[0].session_id, out[0].kind) == ("r1", "tool.call")
    assert "NEEDLE" in out[0].snippet and "caf\ufffd" in out[0].snippet


def test_byte_offsets_map_onto_the_decoded_line_past_a_byte_that_is_not_utf8() -> None:
    """The base64 branch decoded with U+FFFD and re-encoded before mapping,
    so every byte that is not UTF-8 grew to three and the match window slid
    two characters early ('\ufffd NEED' for NEEDLE). The mapping counts the
    characters the byte prefix decodes to."""
    line = b"caf\xe9 NEEDLE and \xe2\x80\x9cmore\xe2\x80\x9d"
    start, end = _char_span(line, line.index(b"NEEDLE"), line.index(b"NEEDLE") + 6)
    assert line.decode("utf-8", "replace")[start:end] == "NEEDLE"
    start, end = _char_span(line, line.index(b"more"), line.index(b"more") + 4)
    assert line.decode("utf-8", "replace")[start:end] == "more"
