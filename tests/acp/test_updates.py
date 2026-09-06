# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""What an editor is told a run did.

Projected from the SAME fold the CLI, TUI and web render, so a fourth surface
cannot disagree with the other three about what happened.
"""

from __future__ import annotations

import json
from typing import Any

from agent6.ui.acp.updates import updates_for
from agent6.viewmodel.transcript import TranscriptFold, TranscriptItem


def updates_for_events(
    events: list[dict[str, Any]], *, acp_session_id: str
) -> list[dict[str, Any]]:
    """One fold instance across the whole sequence, as the runner drives it;
    fresh folds per event would emit each partial message as if it were whole."""
    fold = TranscriptFold()
    out: list[dict[str, Any]] = []
    announced: set[str] = set()
    for event in events:
        for item in fold.feed(event):
            out.extend(
                updates_for(
                    item, acp_session_id=acp_session_id, announced=item.call_id in announced
                )
            )
            if item.kind == "tool":
                announced.add(item.call_id)
    return out


def _kinds(updates: list[dict[str, Any]]) -> list[str]:
    return [u["params"]["update"]["sessionUpdate"] for u in updates]


def test_reasoning_and_answer_are_different_channels() -> None:
    """An editor renders thinking collapsed; conflating them would present the
    model's scratch work as its answer."""
    thinking = updates_for(TranscriptItem("thinking", body="let me look"), acp_session_id="s")
    text = updates_for(TranscriptItem("text", body="the answer"), acp_session_id="s")
    assert _kinds(thinking) == ["agent_thought_chunk"]
    assert _kinds(text) == ["agent_message_chunk"]


def test_the_operators_own_words_echo_back_as_theirs() -> None:
    """A steer is the human speaking. Attributing it to the agent would make
    the transcript lie about who said what."""
    updates = updates_for(TranscriptItem("operator", body="also add a flag"), acp_session_id="s")
    assert _kinds(updates) == ["user_message_chunk"]


def test_a_tool_is_a_call_and_then_an_outcome() -> None:
    """ACP models a tool call as a thing with a lifecycle: the call goes out
    when the fold sees it (in progress), its outcome when the result lands.
    An editor that only ever saw the finished pair could not show work in
    progress -- which for a long verify is the whole point."""
    call = {"type": "tool.call", "name": "run_verify_command", "args": {}, "call_id": 1}
    result = {"type": "tool.result", "name": "run_verify_command", "ok": True, "call_id": 1}
    updates = updates_for_events([call, result], acp_session_id="s")
    assert _kinds(updates) == ["tool_call", "tool_call_update"]
    announced, done = (u["params"]["update"] for u in updates)
    assert announced["toolCallId"] == done["toolCallId"], "the update must pair with its call"
    assert announced["status"] == "in_progress"
    assert done["status"] == "completed"


def test_an_approval_wait_reads_pending_then_in_progress() -> None:
    """ACP keeps `pending` for a call awaiting approval: the fold marks the
    gated call while its prompt is open, and the projection follows it,
    updating the call it announced rather than announcing it again."""
    call = {"type": "tool.call", "name": "run_command", "args": {"argv": ["ls"]}, "call_id": 1}
    prompt = {"type": "approval.prompt", "id": "approval-1", "prompt": "Allow run_command: ls"}
    answer = {"type": "approval.answer", "id": "approval-1", "approved": True}
    result = {"type": "tool.result", "name": "run_command", "ok": True, "call_id": 1}
    updates = updates_for_events([call, prompt, answer, result], acp_session_id="s")
    assert _kinds(updates) == ["tool_call"] + ["tool_call_update"] * 3
    assert [u["params"]["update"]["status"] for u in updates] == [
        "in_progress",
        "pending",
        "in_progress",
        "completed",
    ]
    assert len({u["params"]["update"]["toolCallId"] for u in updates}) == 1


def test_a_failed_tool_says_so() -> None:
    (outcome,) = updates_for(
        TranscriptItem("tool", name="run_command", arg="ls", ok=False), acp_session_id="s"
    )
    assert outcome["params"]["update"]["status"] == "failed"


def test_a_tool_still_running_is_not_reported_failed() -> None:
    """`ok=None` is "no outcome yet", which is neither a failure nor a
    success: the call is announced, in progress, and nothing closes it."""
    updates = updates_for(TranscriptItem("tool", name="grep", arg="x"), acp_session_id="s")
    assert _kinds(updates) == ["tool_call"]
    assert updates[0]["params"]["update"]["status"] == "in_progress"


def test_an_empty_body_produces_nothing() -> None:
    """A blank chunk renders as an empty bubble in the editor."""
    assert updates_for(TranscriptItem("text", body="   "), acp_session_id="s") == []


def test_every_notification_is_addressed_and_well_formed() -> None:
    updates = updates_for(TranscriptItem("text", body="hi"), acp_session_id="sess-1")
    (one,) = updates
    assert one["jsonrpc"] == "2.0"
    assert one["method"] == "session/update"
    assert one["params"]["sessionId"] == "sess-1"
    assert "id" not in one, "a notification expects no reply"
    json.dumps(one)  # it has to survive the wire


def test_deltas_are_folded_once_across_the_whole_run() -> None:
    """The fold is stateful: deltas accumulate and flush at a turn boundary.
    A fresh fold per event would emit every partial message as if it were
    whole."""
    events: list[dict[str, Any]] = [
        {"type": "role.text_delta", "text": "the "},
        {"type": "role.text_delta", "text": "answer"},
        {"type": "role.result", "role": "worker", "ok": True},
    ]
    updates = updates_for_events(events, acp_session_id="s")
    assert _kinds(updates) == ["agent_message_chunk"]
    assert updates[0]["params"]["update"]["content"]["text"] == "the answer"


def test_a_headless_run_still_has_something_to_show() -> None:
    """No streaming means no deltas; the settled text on role.result is what
    the fold falls back to, and it must reach the editor too."""
    events: list[dict[str, Any]] = [
        {"type": "role.result", "role": "worker", "ok": True, "text": "done it"}
    ]
    updates = updates_for_events(events, acp_session_id="s")
    assert updates[0]["params"]["update"]["content"]["text"] == "done it"


def test_a_run_that_failed_does_not_render_as_silence() -> None:
    """The fold sets `body` only for a clean finish, carrying everything else
    in ok/name/detail. Reading body alone made a provider error, a budget stop
    and an iteration cap produce ZERO notifications -- an editor watching a run
    that simply stops."""
    labels = {
        "provider_error": "provider error",
        "budget_exhausted": "budget exhausted",
        "max_iterations": "hit iteration cap",
        "steer_abort": "stopped",
    }
    for reason, label in labels.items():
        events = [{"type": "session.end", "reason": reason, "all_passed": False}]
        updates = updates_for_events(events, acp_session_id="s")
        assert updates, f"{reason} rendered as nothing"
        text = updates[-1]["params"]["update"]["content"]["text"]
        # The reason's own label, in the shared status vocabulary -- never a
        # blanket "did not pass" verdict.
        assert f"Session {label}" in text


def test_a_red_gate_does_not_look_like_a_green_one() -> None:
    """`finish_session` with all_passed=False is a finish over a RED verify."""
    red = updates_for_events(
        [{"type": "session.end", "reason": "finish_session", "all_passed": False}],
        acp_session_id="s",
    )
    green = updates_for_events(
        [{"type": "session.end", "reason": "finish_session", "all_passed": True}],
        acp_session_id="s",
    )
    assert (
        red[-1]["params"]["update"]["content"]["text"]
        != green[-1]["params"]["update"]["content"]["text"]
    )


def test_a_commit_is_not_dropped() -> None:
    """A commit's sha and line count live in `detail`; `body` is empty, so
    keying on body alone dropped every auto-commit."""
    updates = updates_for(
        TranscriptItem("commit", arg="abc1234", detail="3 lines"), acp_session_id="s"
    )
    text = updates[0]["params"]["update"]["content"]["text"]
    assert "abc1234" in text and "3 lines" in text


def test_two_identical_tool_calls_do_not_share_an_id() -> None:
    """ACP models a tool call as ONE thing with a lifecycle. Sharing an id made
    an editor overwrite the first call's FAILURE with the second's success --
    the red run vanished from view."""
    events: list[dict[str, Any]] = [
        {"type": "tool.call", "name": "run_command", "args": {"argv": ["pytest"]}, "call_id": 1},
        {"type": "tool.result", "name": "run_command", "call_id": 1, "ok": False},
        {"type": "tool.call", "name": "run_command", "args": {"argv": ["pytest"]}, "call_id": 2},
        {"type": "tool.result", "name": "run_command", "call_id": 2, "ok": True},
    ]
    ids = {
        u["params"]["update"]["toolCallId"]
        for u in updates_for_events(events, acp_session_id="s")
        if "toolCallId" in u["params"]["update"]
    }
    assert len(ids) == 2, f"the two calls collided on {ids}"


def test_a_tool_call_id_is_unique_across_a_sessions_turns() -> None:
    """One ACP session runs many runs, and the stamped call id is a
    per-DISPATCHER counter that restarts at "1" each time. Without the run id
    an editor keyed on toolCallId -- which is the whole reason the field is
    carried -- overwrites turn 1's FAILED call with turn 2's success."""
    from agent6.ui.acp.updates import updates_for
    from agent6.viewmodel.transcript import TranscriptItem

    item = TranscriptItem(kind="tool", name="run_command", arg="ls", ok=True, call_id="1")
    first = updates_for(item, acp_session_id="s", session_id="brave-oak-AAAAAA")
    second = updates_for(item, acp_session_id="s", session_id="clever-elm-BBBBBB")
    assert first[0]["params"]["update"]["toolCallId"] != second[0]["params"]["update"]["toolCallId"]
    assert first[0]["params"]["update"]["toolCallId"].startswith("brave-oak-AAAAAA:")


def test_a_tools_output_is_wrapped_in_acps_tagged_content() -> None:
    """`ToolCallContent` is a discriminated union, not a ContentBlock array.

    From the published schema: `oneOf` [{type: "content", ...Content}, {type:
    "diff", ...}, {type: "terminal", ...}] with `discriminator.propertyName =
    "type"`. Sending the bare array made a strict client reject the whole
    notification -- so the `completed`/`failed` it carried never arrived and
    the call announced one line earlier stayed `pending` for the rest of the
    session.
    """
    from agent6.ui.acp.updates import updates_for
    from agent6.viewmodel.transcript import TranscriptItem

    item = TranscriptItem(kind="tool", name="run_verify", arg="", ok=False, detail="exit 1")
    (outcome,) = updates_for(item, acp_session_id="s")
    content = outcome["params"]["update"]["content"]
    assert content == [{"type": "content", "content": {"type": "text", "text": "exit 1"}}]


def test_a_failed_tool_carries_the_output_that_explains_it() -> None:
    """The fold fills `tail` with the stderr/stdout of a failure for exactly
    this. Sending only `detail` left an editor showing "failed" and the word
    "exit 1", with the test log that says WHY nowhere on the wire."""
    from agent6.ui.acp.updates import updates_for
    from agent6.viewmodel.transcript import TranscriptItem

    item = TranscriptItem(
        kind="tool", name="run_verify", arg="", ok=False, detail="exit 1", tail="E   assert 1 == 2"
    )
    (outcome,) = updates_for(item, acp_session_id="s")
    text = outcome["params"]["update"]["content"][0]["content"]["text"]
    assert "assert 1 == 2" in text


def test_model_text_cannot_carry_a_terminal_escape_to_the_editor() -> None:
    """Unlike the CLI, the renderer here is a THIRD PARTY, so agent6 does not
    get to assume it treats an escape as inert. The fold strips CSI from
    `detail`/`tail` only -- OSC (the title / clipboard / hyperlink family)
    survived and `body` was never scrubbed at all. Newlines and tabs are real
    content and stay."""
    from agent6.ui.acp.updates import updates_for
    from agent6.viewmodel.transcript import TranscriptItem

    hostile = "hi\x1b]0;pwned\x07 there\x1b[2J\nsecond\tline"
    (update,) = updates_for(TranscriptItem(kind="text", body=hostile), acp_session_id="s")
    text = update["params"]["update"]["content"]["text"]
    assert "\x1b" not in text and "\x07" not in text
    assert "\nsecond\tline" in text, "real whitespace is content, not an escape"


def test_a_tool_call_title_is_scrubbed_like_its_content() -> None:
    """`title` is `salient_arg` -- the model's own argv, path or pattern -- and
    it is not a ContentBlock, so it went round the scrub the sibling `content`
    field on the very next notification already had."""
    from agent6.ui.acp.updates import updates_for
    from agent6.viewmodel.transcript import TranscriptItem

    hostile = "sh -c '\x1b]0;PWNED\x07'"
    (call,) = updates_for(
        TranscriptItem(kind="tool", name="run_command", arg=hostile), acp_session_id="s"
    )
    assert "\x1b" not in call["params"]["update"]["title"]
    assert "\x07" not in call["params"]["update"]["title"]


def test_a_gateless_finish_never_reads_as_a_failed_check() -> None:
    """A deliberate finish that verified nothing is "finished" in the shared
    status vocabulary; "did not pass" implied a check that never existed."""
    updates = updates_for_events(
        [{"type": "session.end", "reason": "finish_session", "all_passed": False}],
        acp_session_id="s",
    )
    text = updates[-1]["params"]["update"]["content"]["text"]
    assert "Session finished" in text
    assert "did not pass" not in text
