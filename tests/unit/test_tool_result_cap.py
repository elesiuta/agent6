"""per-tool-result cap must produce well-formed JSON.

, the loop applied a hard `content[:20_000]` slice to every
tool_result payload regardless of the boundary. For tools that return
JSON-serialized dicts (e.g. `read_file` returns
`{"content": "...", "size": N, "lines_total": N}`), this slice cut
through the middle of the JSON string and produced a payload the model
could not parse. Weak models (Kimi K2.6 observed live, May 2026)
concluded the underlying tool returned a partial result, called
`read_file` again with identical arguments expecting "the rest", and
latched the loop-guard with $0.15-0.20 wasted on a futile re-read loop.

The fix wraps over-cap payloads in a fresh, well-formed JSON envelope
that names the truncation explicitly and points the model at the right
next step (offset+limit for read_file, narrower scope for run_command).
"""

from __future__ import annotations

import json

from agent6.workflows._compaction import (
    TOOL_RESULT_CAP_BYTES as _TOOL_RESULT_CAP_BYTES,
)
from agent6.workflows._compaction import (
    cap_tool_result as _cap_tool_result,
)


def test_small_payload_passes_through_unchanged() -> None:
    payload = json.dumps({"content": "hello", "size": 5})
    assert _cap_tool_result(payload, tool_name="read_file") == payload


def test_payload_at_cap_passes_through_unchanged() -> None:
    payload = "x" * _TOOL_RESULT_CAP_BYTES
    assert _cap_tool_result(payload, tool_name="read_file") == payload


def test_oversized_read_file_payload_yields_valid_truncation_envelope() -> None:
    """The big regression: cap a read_file result, parse the output,
    confirm it is valid JSON with explicit truncation signal."""
    big = "A" * (_TOOL_RESULT_CAP_BYTES * 2)
    raw = json.dumps({"content": big, "size": len(big), "lines_total": 1})
    capped = _cap_tool_result(raw, tool_name="read_file")
    parsed = json.loads(capped)  # must be valid JSON, no mid-string cut
    assert parsed["_tool_result_truncated"] is True
    assert parsed["tool"] == "read_file"
    assert parsed["total_chars"] == len(raw)
    assert parsed["shown_chars"] <= _TOOL_RESULT_CAP_BYTES
    assert "start_line" in parsed["guidance"]
    assert "limit" in parsed["guidance"]
    # Head should be a prefix of the original raw payload so the model
    # can see what it did get.
    assert raw.startswith(parsed["head"])


def test_oversized_run_command_payload_guidance_points_at_narrowing() -> None:
    big = "B" * (_TOOL_RESULT_CAP_BYTES + 1)
    capped = _cap_tool_result(big, tool_name="run_command")
    parsed = json.loads(capped)
    assert parsed["_tool_result_truncated"] is True
    assert "narrower" in parsed["guidance"] or "narrower scope" in parsed["guidance"]


def test_cap_total_envelope_size_stays_under_cap() -> None:
    """The envelope itself must respect the cap so we do not silently
    grow the tool_result payload past its budget. The head must be sized
    by ENCODED length: json.dumps re-escapes quotes and backslashes, so a
    raw-char budget overshoots the cap on escape-heavy content (observed
    118k chars emitted against the 60k cap)."""
    for big in (
        "C" * (_TOOL_RESULT_CAP_BYTES * 5),  # no escaping: raw == encoded
        '"\\' * (_TOOL_RESULT_CAP_BYTES * 2),  # every char doubles when encoded
        ('He said "use \\n"\n' * 20_000),  # mixed quotes/backslashes/newlines
    ):
        capped = _cap_tool_result(big, tool_name="grep")
        assert len(capped.encode()) <= _TOOL_RESULT_CAP_BYTES
        parsed = json.loads(capped)  # still a well-formed envelope
        assert parsed["_tool_result_truncated"] is True
        assert parsed["total_chars"] == len(big)
        # A useful amount of head survives; big.startswith proves it is a
        # clean prefix, not a mid-escape cut.
        assert parsed["shown_chars"] > _TOOL_RESULT_CAP_BYTES // 4
        assert big.startswith(parsed["head"])


def test_truncation_envelope_for_unknown_tool_still_well_formed() -> None:
    big = "D" * (_TOOL_RESULT_CAP_BYTES + 100)
    capped = _cap_tool_result(big, tool_name="some_new_tool")
    parsed = json.loads(capped)
    assert parsed["tool"] == "some_new_tool"
    assert parsed["_tool_result_truncated"] is True


def test_the_cap_is_a_parameter() -> None:
    """A provider that hands the model less than the loop's default gets a
    tighter bound through the same envelope."""
    content = json.dumps({"content": "x" * 3_000})
    assert _cap_tool_result(content, tool_name="read_file") == content
    capped = _cap_tool_result(content, tool_name="read_file", cap=2_000)
    assert len(capped) <= 2_000 and json.loads(capped) != json.loads(content)


def test_the_cap_is_a_byte_budget() -> None:
    """Measured in characters, a 45,000-character CJK result passed the cap at
    135,000 bytes and hit Claude Code's 50,000-byte persistence threshold as a
    fatal provider error that ended the run."""
    wide = "\u6f22" * 45_000
    capped = _cap_tool_result(wide, tool_name="read_file", cap=49_000)
    assert len(capped.encode()) <= 49_000
    parsed = json.loads(capped)
    assert parsed["_tool_result_truncated"] is True
    assert parsed["total_chars"] == 45_000 and 0 < parsed["shown_chars"] < 45_000
