# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The typed logs.jsonl read model: parse_event maps raw dicts to the fold families.

Pins the parse boundary itself (the fold's end-to-end behaviour is pinned by
test_viewmodel_state and the golden compat test). The contract: every family the
SessionState fold consumes parses to its typed shape with the fold's historical
coercion, and everything else -- telemetry, unknown/future types, a typeless line
-- degrades to RawEvent so the fold drops it instead of crashing.
"""

from __future__ import annotations

from agent6.viewmodel import events as ev


def test_known_families_parse_to_their_typed_shape() -> None:
    assert ev.parse_event({"type": "session.start", "user_task": "t"}) == ev.SessionStart(
        user_task="t"
    )
    assert ev.parse_event(
        {"type": "session.end", "reason": "finish_session", "all_passed": True}
    ) == (ev.SessionEnd(all_passed=True, reason="finish_session"))
    assert ev.parse_event(
        {"type": "session.end", "reason": "finish_session", "all_passed": True, "scoped": True}
    ) == (ev.SessionEnd(all_passed=True, reason="finish_session", scoped=True))
    assert ev.parse_event({"type": "loop.resume.start"}) == ev.ResumeStart()
    assert ev.parse_event({"type": "session.steer_requested"}) == ev.SteerRequested()


def test_unknown_and_telemetry_and_typeless_become_rawevent() -> None:
    # A telemetry type the fold never consumes, an unknown/future type, and a line
    # with no `type` all fall to RawEvent (the fold's old `case _`).
    samples = ({"type": "loop.auto_commit.failed", "sha": "x"}, {"type": "totally.new"}, {"foo": 1})
    for raw in samples:
        parsed = ev.parse_event(raw)
        assert isinstance(parsed, ev.RawEvent)
        assert parsed.raw is raw  # carries the dict for the log-line renderer


def test_coercion_matches_the_folds_historical_defaults() -> None:
    # Missing fields default, not raise (the fold never validated).
    assert ev.parse_event({"type": "role.call"}) == ev.RoleCall(role="", model="", provider="")
    # as_int swallows a non-numeric token to 0 (role.result context math).
    assert ev.parse_event({"type": "role.result", "tokens_in": "nope"}).tokens_in == 0  # type: ignore[union-attr]
    # ok tolerates the legacy stringified booleans: "True" folds ok, "False"
    # folds FAILED. bool()-coercion read any non-empty string -- "False"
    # included -- as success, so a historical failed tool rendered green in the
    # SessionState surfaces while the conversation fold showed it red.
    assert ev.parse_event({"type": "tool.result", "name": "t", "ok": "True"}).ok is True  # type: ignore[union-attr]
    assert ev.parse_event({"type": "tool.result", "name": "t", "ok": "False"}).ok is False  # type: ignore[union-attr]
    # A non-string cursor drops to None; nodes stays raw for the tree walker.
    gu = ev.parse_event({"type": "graph.update", "nodes": {"a": {}}, "cursor": 123})
    assert isinstance(gu, ev.GraphUpdate) and gu.cursor is None and gu.nodes == {"a": {}}
    # A null node map coerces to {} exactly as the fold's `or {}` did.
    assert ev.parse_event({"type": "graph.update", "nodes": None, "cursor": None}).nodes == {}  # type: ignore[union-attr]


def test_question_prompt_filters_non_dict_entries_and_coerces_options() -> None:
    parsed = ev.parse_event(
        {
            "type": "question.prompt",
            "id": "q1",
            "questions": [{"question": "Which?", "options": ["a", "b"]}, "garbage", {"bad": 1}],
        }
    )
    assert isinstance(parsed, ev.QuestionPrompt)
    assert parsed.id == "q1"
    # The non-dict entry is dropped; the malformed dict keeps defaults.
    assert parsed.questions == (
        ev.EventQuestion(question="Which?", options=("a", "b")),
        ev.EventQuestion(question="", options=()),
    )


def test_malformed_numeric_fields_degrade_to_raw_event() -> None:
    # A torn numeric in a KNOWN family must not raise: the fold runs unwrapped
    # inside live tails (web SSE, TUI reader), so a line an interrupted writer
    # left behind degrades like an unknown type instead of crashing the tail.
    for raw in (
        {"type": "budget.update", "usd_total": "garbage"},
        {"type": "budget.update", "input_total": []},
        {"type": "verify.end", "exit_code": "x"},
        {"type": "verify.end", "duration_s": {}},
    ):
        parsed = ev.parse_event(raw)
        assert isinstance(parsed, ev.RawEvent)
        assert parsed.raw == raw
