# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""TranscriptFold: the event stream folds into the right conversation items."""

from __future__ import annotations

from agent6.viewmodel import fold_transcript, restate, salient_arg


def _read(path: str) -> list[dict[str, object]]:
    return [
        {"type": "role.call", "role": "worker"},
        {"type": "role.thinking_delta", "text": "let me look"},
        {"type": "role.result"},
        {"type": "tool.call", "name": "read_file", "args": {"path": path}},
        {"type": "tool.result", "name": "read_file", "ok": True, "summary": "12 bytes"},
    ]


def test_tool_only_turn_has_no_empty_text_item() -> None:
    # A turn that reasons then calls a tool, emitting NO assistant text, must not
    # produce a blank `text` item -- the bug behind the empty response blocks.
    items = fold_transcript(_read("a.py"))
    kinds = [i.kind for i in items]
    assert kinds == ["thinking", "tool"]
    assert not any(i.kind == "text" and not i.body for i in items)
    tool = items[1]
    assert (tool.name, tool.arg, tool.ok, tool.detail) == ("read_file", "a.py", True, "12 bytes")


def test_verify_badge_folds_into_the_tool_item() -> None:
    events = [
        {"type": "tool.call", "name": "run_verify_command", "args": {}},
        {"type": "verify.start", "cmd": ["pytest"]},
        {"type": "verify.end", "exit_code": 0, "duration_s": 0.2},
        {"type": "tool.result", "name": "run_verify_command", "ok": True, "summary": "exit=0"},
    ]
    (item,) = fold_transcript(events)
    assert item.kind == "tool" and item.ok is True
    assert item.detail == "✓ pass · 0.2s"  # the verify badge, not the raw summary


def test_finish_tool_becomes_the_verdict_not_a_step() -> None:
    events = [
        {"type": "tool.call", "name": "finish_session", "args": {"summary": "all green"}},
        {"type": "tool.result", "name": "finish_session", "ok": True, "summary": "finish_session"},
        {"type": "session.end", "all_passed": True, "reason": "finish_session"},
    ]
    items = fold_transcript(events)
    assert [i.kind for i in items] == ["done"]
    done = items[0]
    assert done.ok is True and done.body == "all green"
    assert done.detail == "0 tools · 0 commits"


def test_a_plans_finish_summary_pairs_with_its_done_line() -> None:
    """finish_planning ends a plan (reason finish_planning); its summary is the
    plan's own deliverable line and paired like a run's, not dropped as if the
    end were a failure (every surface showed a bare "done" for plans)."""
    events = [
        {"type": "tool.call", "name": "finish_planning", "args": {"summary": "Plan seeded."}},
        {"type": "tool.result", "name": "finish_planning", "ok": True, "summary": "ok"},
        {"type": "session.end", "all_passed": True, "reason": "finish_planning"},
    ]
    (done,) = fold_transcript(events)
    assert done.kind == "done" and done.ok is True and done.body == "Plan seeded."


def test_failed_tool_keeps_a_tail() -> None:
    events = [
        {"type": "tool.call", "name": "run_command", "args": {"command": "ls /nope"}},
        {
            "type": "tool.result",
            "name": "run_command",
            "ok": False,
            "summary": "exit=2",
            "stderr_tail": "ls: /nope: No such file",
        },
    ]
    (item,) = fold_transcript(events)
    assert item.ok is False and "No such file" in item.tail


def test_tool_output_ansi_is_stripped_from_the_fold() -> None:
    # The fold is plain data for non-terminal surfaces (web/saved transcripts):
    # colored tool output must not leak escape sequences as literal text.
    events = [
        {"type": "tool.call", "name": "run_command", "args": {"command": "pytest"}},
        {
            "type": "tool.result",
            "name": "run_command",
            "ok": True,
            "summary": "\x1b[32mok\x1b[0m",
            "stdout_tail": "\x1b[36m[Tach]\x1b[0m 10 tests \x1b[1mpass\x1b[0m",
        },
    ]
    (item,) = fold_transcript(events)
    assert "\x1b" not in item.tail and item.tail == "[Tach] 10 tests pass"
    assert "\x1b" not in item.detail and item.detail == "ok"


def test_the_scrub_is_default_deny_not_a_csi_blocklist() -> None:
    """Stripping CSI alone let OSC and DCS through to the CLI terminal -- a
    demonstrated OSC 52 wrote the operator's clipboard from command stdout.
    Every string-carrying escape family goes, whole; stray C0/C1 controls drop
    (keeping \\n and \\t); plain text and the sequences' cut-off payloads
    surface as inert text."""
    from agent6.viewmodel.transcript import scrub_terminal_controls as scrub

    payload = "cGF5bG9hZA=="
    assert scrub(f"\x1b]52;c;{payload}\x07after") == "after"  # OSC 52, BEL-terminated
    assert scrub(f"\x1b]52;c;{payload}\x1b\\after") == "after"  # OSC 52, ST-terminated
    assert scrub("\x1b]0;title\x07x") == "x"  # OSC 0 (window title)
    assert scrub("\x1bPq#payload\x1b\\x") == "x"  # DCS
    assert scrub("\x1b_apc\x1b\\x") == "x" and scrub("\x1b^pm\x1b\\x") == "x"  # APC / PM
    assert scrub("\x1b[31mred\x1b[0m") == "red"  # CSI still goes
    assert scrub("\x9b31mx") == "31mx"  # a C1 byte cannot reopen the door
    assert scrub("a\rb\x07c") == "abc"  # stray \r spoofing and BEL drop
    assert scrub("keep\nthese\ttwo") == "keep\nthese\ttwo"
    # Cut off mid-sequence (a stream chunk boundary): the opener's tail goes,
    # and the continuation is inert text on its own.
    assert scrub(f"\x1b]52;c;{payload[:4]}") == ""
    assert scrub(f"{payload[4:]}\x07done") == f"{payload[4:]}done"
    # Idempotent: re-scrubbing accumulated tails changes nothing.
    once = scrub("\x1b]52;c;x\x07text\x1b[1mbold")
    assert scrub(once) == once == "textbold"


def test_parallel_dispatched_renders_a_truthful_marker() -> None:
    # The dispatched event carries only group + per-segment tasks (lane ids do not
    # exist yet); the fold renders a task count + the task summary, never raw json.
    items = fold_transcript(
        [{"type": "loop.parallel.dispatched", "group": "p1", "tasks": ["fix the bug", "add tests"]}]
    )
    (item,) = items
    assert item.kind == "marker"
    assert "dispatched 2 parallel tasks (group p1)" in item.body
    assert "listed under this session" in item.body
    assert "fix the bug" in item.body and "add tests" in item.body


def test_parallel_joined_names_lane_ids_branches_shas() -> None:
    items = fold_transcript(
        [
            {
                "type": "loop.parallel.joined",
                "group": "p1",
                "lanes": [
                    {"session_id": "co-p1-l1", "branch": "agent6/co-p1-l1", "status": "joined",
                     "sha": "abcdef1234567890"},
                    {"session_id": "co-p1-l2", "branch": "agent6/co-p1-l2", "status": "conflict",
                     "sha": ""},
                ],
            }
        ]
    )  # fmt: skip
    (item,) = items
    assert item.kind == "marker"
    assert "joined group p1: 2 lane(s)" in item.body
    assert "joined  co-p1-l1  agent6/co-p1-l1  abcdef123456" in item.body  # sha clipped to 12
    assert "conflict  co-p1-l2  agent6/co-p1-l2" in item.body


def test_parallel_failed_renders_the_dispatch_error_but_not_the_join_subset() -> None:
    # A dispatch failure (carries `error`) renders; a post-join failure (carries
    # only `lanes`, already shown by the joined marker) is not double-rendered.
    (item,) = fold_transcript(
        [{"type": "loop.parallel.failed", "group": "p2", "error": "dirty tree"}]
    )
    assert item.kind == "marker" and "group p2 dispatch failed: dirty tree" in item.body
    assert fold_transcript([{"type": "loop.parallel.failed", "group": "p2", "lanes": [{}]}]) == []


def test_salient_arg_prefers_a_primary_key() -> None:
    assert salient_arg({"recursive": True, "path": "src/x.py"}) == "src/x.py"
    assert salient_arg({}) == ""
    assert salient_arg({"n": 3}) == "n=3"


def test_salient_arg_renders_argv_as_a_shell_line() -> None:
    # Not a Python list repr: the operator reads it as a command; a token with a
    # space is quoted the way a shell needs.
    assert salient_arg({"argv": ["cargo", "build", "--release"]}) == "cargo build --release"
    assert salient_arg({"argv": ["echo", "a b"]}) == "echo 'a b'"


def test_salient_arg_renders_ask_user_questions_as_text() -> None:
    args = {"questions": [{"question": "Which theme?"}, {"question": "Apply to TUI?"}]}
    assert salient_arg(args) == "Which theme? (+1)"


def test_interleaved_tool_calls_pair_by_name() -> None:
    # A concurrent explore-tier review panel can interleave tool.call/tool.result
    # across tools; each result must pair with its own call by name, not with the
    # next pending call by position.
    events = [
        {"type": "tool.call", "name": "read_file", "args": {"path": "a.py"}},
        {"type": "tool.call", "name": "grep", "args": {"pattern": "def"}},
        {"type": "tool.result", "name": "grep", "ok": False, "summary": "no match"},
        {"type": "tool.result", "name": "read_file", "ok": True, "summary": "12 bytes"},
    ]
    tools = {i.name: i for i in fold_transcript(events) if i.kind == "tool"}
    assert len(tools) == 2  # both paired; none dropped or mislabelled
    assert tools["grep"].ok is False and tools["grep"].detail == "no match"
    assert tools["read_file"].ok is True and tools["read_file"].detail == "12 bytes"


def test_same_name_concurrent_calls_pair_by_call_id() -> None:
    # Two review seats reading different files concurrently: name-keyed pairing
    # rendered one item with crossed arg/summary and dropped the other result.
    # call_id (stamped per dispatch) pairs each result with its own call, even
    # out of call order.
    events = [
        {"type": "tool.call", "name": "read_file", "args": {"path": "a.py"}, "call_id": 1},
        {"type": "tool.call", "name": "read_file", "args": {"path": "b.py"}, "call_id": 2},
        {
            "type": "tool.result",
            "name": "read_file",
            "ok": True,
            "summary": "b bytes",
            "call_id": 2,
        },
        {
            "type": "tool.result",
            "name": "read_file",
            "ok": False,
            "summary": "a boom",
            "call_id": 1,
        },
    ]
    tools = {i.arg: i for i in fold_transcript(events) if i.kind == "tool"}
    assert set(tools) == {"a.py", "b.py"}  # both rendered; neither dropped
    assert tools["b.py"].ok is True and tools["b.py"].detail == "b bytes"
    assert tools["a.py"].ok is False and "a boom" in tools["a.py"].detail


def test_unmatched_tool_result_is_dropped() -> None:
    # A result with no matching pending call must not crash or emit a bogus item.
    assert fold_transcript([{"type": "tool.result", "name": "ghost", "ok": True}]) == []


def test_stopped_run_done_reads_as_stopped_not_failed() -> None:
    # A steer_abort run must render "stopped", not the raw "steer_abort" nor a
    # failure -- the CLI/TUI done line shows item.name for a not-ok run.
    (done,) = fold_transcript(
        [{"type": "session.end", "reason": "steer_abort", "all_passed": False}]
    )
    assert done.kind == "done" and done.ok is False and done.name == "stopped"


def test_interrupted_run_is_in_the_reason_vocabulary_and_labeled() -> None:
    """The app layer emits session.end reason="interrupted" on KeyboardInterrupt;
    the value must live in SessionEndReason (the wire vocabulary of session.end.reason).
    The raw token IS the accepted done-line rendering (it reads fine; the
    label map exists only for unfriendly tokens like steer_abort)."""
    from typing import get_args

    from agent6.workflows._session_state import SessionEndReason

    assert "interrupted" in get_args(SessionEndReason)
    (done,) = fold_transcript(
        [{"type": "session.end", "reason": "interrupted", "all_passed": False}]
    )
    assert done.kind == "done" and done.ok is False and done.name == "interrupted"


def test_operator_steer_text_becomes_an_operator_item() -> None:
    """The loop's steer injection (a typed steer, or the follow-up a resume was
    started with) shows in the conversation as an operator turn; old logs that
    carry only a char count yield nothing."""
    from agent6.viewmodel.transcript import OPERATOR, TranscriptFold
    from agent6.viewmodel.transcript_style import item_lines

    fold = TranscriptFold()
    items = fold.feed({"type": "loop.steer.injected", "chars": 9, "text": "try it\nagain"})
    assert [i.kind for i in items] == ["operator"]
    assert items[0].body == "try it\nagain"
    # Rendered at every detail level, glyph + the operator's own words.
    for level in ("hidden", "collapsed", "expanded"):
        lines = item_lines(items[0], detail=level)
        flat = "".join(chunk for line in lines for chunk, _ in line)
        assert f"{OPERATOR} try it" in flat and "again" in flat
    # An old log without the text field adds no item.
    assert fold.feed({"type": "loop.steer.injected", "chars": 9}) == []


def test_pins_render_once_as_operator_items() -> None:
    """A pin is the operator's own instruction, shown like a steer where it
    enters the conversation: the leg-start announcement (a --pin run, a fork)
    and each /pin. A resume boundary restating the same list adds nothing;
    the conversation carried no pin at all before."""
    from agent6.viewmodel.transcript import TranscriptFold

    fold = TranscriptFold()
    items = fold.feed({"type": "loop.pin.restored", "pins": ["never touch tests"], "count": 1})
    assert [(i.kind, i.body) for i in items] == [("operator", "pinned: never touch tests")]
    items = fold.feed({"type": "loop.pin.added", "text": "keep the API", "chars": 12, "count": 2})
    assert [(i.kind, i.body) for i in items] == [("operator", "pinned: keep the API")]
    restated = {"type": "loop.pin.restored", "pins": ["never touch tests", "keep the API"]}
    assert fold.feed(restated) == []
    assert fold.feed({"type": "loop.pin.restored", "pins": [], "count": 0}) == []


def test_an_internal_side_call_is_not_rendered_as_agent_speech() -> None:
    """Only the role DRIVING the session speaks to the operator.

    agent6 makes side calls with their own roles -- the verify-command inferer
    runs before the loop even starts. Its `role.result` was folded into a
    message like any other, so an editor over ACP and the web conversation both
    opened with a bare "[]": the inferer's raw answer for "no verify command
    found", presented as the agent talking.
    """
    events: list[dict[str, object]] = [
        {"type": "role.call", "role": "verify_inferer", "model": "m"},
        {"type": "role.result", "role": "verify_inferer", "ok": True, "text": "[]"},
        {"type": "session.start", "mode": "run", "user_task": "t"},
        {"type": "role.call", "role": "worker", "model": "m"},
        {"type": "role.result", "role": "worker", "ok": True, "text": "the real answer"},
    ]
    bodies = [i.body for i in fold_transcript(events) if i.kind == "text"]
    assert "the real answer" in bodies, "the driving role must still speak"
    assert "[]" not in bodies, f"a side call was rendered as agent speech: {bodies}"


def test_a_streamed_reply_still_renders_when_the_role_is_unnamed() -> None:
    """Older events and the delta path carry no role; they must keep working.
    The guard drops a side call, not every result."""
    events: list[dict[str, object]] = [
        {"type": "session.start", "mode": "run", "user_task": "t"},
        {"type": "role.call", "role": "worker"},
        {"type": "role.text_delta", "text": "streamed prose"},
        {"type": "role.result"},
    ]
    assert "streamed prose" in [i.body for i in fold_transcript(events) if i.kind == "text"]


def test_streamed_deltas_are_scrubbed_even_when_a_sequence_splits() -> None:
    """The live delta path bypassed the fold's preview scrub entirely, and an
    escape can arrive SPLIT across two deltas: scrubbing per piece would let
    the reassembled whole through. The fold scrubs the concatenation."""
    from agent6.viewmodel.state import apply_event, initial_state

    s = initial_state()
    s = apply_event(s, {"type": "role.call", "role": "worker", "model": "m"})
    s = apply_event(s, {"type": "role.text_delta", "text": "safe \x1b]52;c;cGF5"})
    s = apply_event(s, {"type": "role.text_delta", "text": "bG9hZA==\x07 text"})
    assert s.last_role is not None
    # The opener died with its own delta; the continuation is inert text.
    assert s.last_role.streamed_text == "safe bG9hZA== text"
    assert "\x1b" not in s.last_role.streamed_text
    assert "\x07" not in s.last_role.streamed_text


def test_log_lines_carry_no_terminal_controls() -> None:
    """format_log_line embeds model-authored fields (args, summaries, output
    tails) into every skin's log pane; the finished line is scrubbed."""
    from agent6.viewmodel.log_line import format_log_line

    line = format_log_line(
        {
            "type": "tool.result",
            "name": "run_command",
            "ok": True,
            "summary": "done",
            "stdout_tail": "\x1b]52;c;cGF5bG9hZA==\x07visible",
        }
    )
    assert "\x1b" not in line and "visible" in line


def test_restate_compacts_since_the_last_operator_input() -> None:
    """`/restate`: the last steer's text leads, assistant prose survives whole,
    tools become one line with their outcome, and everything before the last
    operator input stays out of frame."""
    events: list[dict[str, object]] = [
        {"type": "session.start", "user_task": "build the thing"},
        {"type": "role.call", "role": "worker"},
        {"type": "role.text_delta", "text": "early work, out of frame"},
        {"type": "role.result"},
        {"type": "loop.steer.injected", "text": "focus on the parser"},
        {"type": "role.call", "role": "worker"},
        {"type": "role.text_delta", "text": "on it"},
        {"type": "role.result"},
        {"type": "tool.call", "name": "read_file", "args": {"path": "parser.py"}},
        {"type": "tool.result", "name": "read_file", "ok": True, "summary": "12 bytes"},
        {"type": "tool.call", "name": "apply_edit", "args": {"path": "parser.py"}},
        {"type": "tool.result", "name": "apply_edit", "ok": False, "summary": "no match"},
    ]
    text = restate(events)
    assert text.startswith("you said: focus on the parser")
    assert "early work" not in text
    assert "on it" in text
    assert "[ok] read_file parser.py: 12 bytes" in text
    assert "[FAILED] apply_edit parser.py: no match" in text


def test_restate_with_no_operator_input_says_so() -> None:
    assert restate([]).startswith("nothing to restate")


def test_done_item_is_a_receipt_when_the_journal_carries_the_pieces() -> None:
    """The done item ends the story INSIDE the surface: cost, wall time, the
    counts, and the last commit subject, each present only when the journal
    carried it (an old journal folds to the bare counts as before)."""
    events = [
        {"type": "session.start", "ts": "2026-08-09T20:00:00+00:00", "user_task": "t"},
        {"type": "tool.call", "name": "apply_edit", "args": {"path": "a.py"}},
        {"type": "tool.result", "name": "apply_edit", "ok": True, "summary": "ok"},
        {"type": "budget.update", "usd_total": 0.0112},
        {
            "type": "loop.auto_commit",
            "sha": "abc123",
            "subject": "fix median for even length",
        },
        {"type": "diff.updated", "sha": "abc123", "patch": "+x\n-y\n"},
        {"type": "tool.call", "name": "finish_session", "args": {"summary": "Fixed."}},
        {
            "type": "session.end",
            "ts": "2026-08-09T20:00:45+00:00",
            "reason": "finish_session",
            "all_passed": True,
        },
    ]
    done = next(it for it in fold_transcript(events) if it.kind == "done")
    assert done.ok is True
    assert done.body == "Fixed."
    assert done.detail == "$0.0112 · 45s · 1 tool · 1 commit · fix median for even length"


def test_done_item_degrades_to_counts_on_a_journal_without_receipt_fields() -> None:
    events = [
        {"type": "tool.call", "name": "read_file", "args": {"path": "a"}},
        {"type": "tool.result", "name": "read_file", "ok": True, "summary": "1 byte"},
        {"type": "session.end", "reason": "finish_session", "all_passed": True},
    ]
    done = next(it for it in fold_transcript(events) if it.kind == "done")
    assert done.detail == "1 tool · 0 commits"


def test_tool_items_carry_bounded_previews() -> None:
    """A successful read shows its head + true line count, and a successful
    edit shows its hunk (carried from the CALL side, where the journal already
    holds the edit pairs); an old journal without the fields folds to no tail."""
    events = [
        {"type": "tool.call", "name": "read_file", "args": {"path": "a.py"}},
        {
            "type": "tool.result",
            "name": "read_file",
            "ok": True,
            "summary": "961 bytes",
            "head_tail": "def f():\n    return 1",
            "lines_total": 961,
        },
        {
            "type": "tool.call",
            "name": "apply_edit",
            "args": {
                "path": "a.py",
                "edits": [
                    {"old_string": "return 1", "new_string": "return 2"},
                    {"old_string": "x", "new_string": "y"},
                ],
            },
        },
        {"type": "tool.result", "name": "apply_edit", "ok": True, "summary": "ok"},
        {"type": "tool.call", "name": "read_file", "args": {"path": "old.py"}},
        {"type": "tool.result", "name": "read_file", "ok": True, "summary": "12 bytes"},
    ]
    tools = [it for it in fold_transcript(events) if it.kind == "tool"]
    assert tools[0].tail == "def f():\n    return 1\n…(961 lines)"
    assert tools[1].tail == "- return 1\n+ return 2\n…(+1 more edit)"
    assert tools[2].tail == ""  # an old journal: no preview fields, no tail


def test_edit_preview_shows_the_changed_lines_only() -> None:
    """An append re-emits its anchor lines in both old_string and new_string;
    the preview is the diff, so the anchor is not shown twice as - and +."""
    events = [
        {
            "type": "tool.call",
            "name": "apply_edit",
            "args": {
                "path": "calc.py",
                "edits": [
                    {
                        "old_string": "def sub(a, b):\n    return a - b\n",
                        "new_string": (
                            "def sub(a, b):\n    return a - b\n\n\n"
                            "def mul(a, b):\n    return a * b\n"
                        ),
                    }
                ],
            },
        },
        {"type": "tool.result", "name": "apply_edit", "ok": True, "summary": "ok"},
    ]
    (tool,) = [it for it in fold_transcript(events) if it.kind == "tool"]
    assert tool.tail == "+ def mul(a, b):\n+ return a * b"


def test_salient_arg_is_always_one_line() -> None:
    """A multi-line arg value (a raw-arguments blob with embedded newlines)
    split the tool head across lines on every skin; the clip flattens
    whitespace so the head stays one line."""
    arg = salient_arg({"_raw_arguments": '{"argv": [".venv/bin/python", "-c", "\nfrom x"]}'})
    assert "\n" not in arg


def test_an_asks_receipt_carries_no_commit_count() -> None:
    """An ask (or a plan) never commits: its done line counts tools only; a
    run keeps "N commits", and a journal with no session.start keeps the
    counts it always showed."""
    ask = [
        {"type": "session.start", "mode": "ask", "user_task": "why?"},
        {"type": "tool.call", "name": "read_file", "args": {"path": "a"}},
        {"type": "tool.result", "name": "read_file", "ok": True, "summary": "1 byte"},
        {"type": "session.end", "reason": "answered", "all_passed": False},
    ]
    done = next(it for it in fold_transcript(ask) if it.kind == "done")
    assert done.detail == "1 tool"
    run = [{**ask[0], "mode": "run"}, *ask[1:]]
    done = next(it for it in fold_transcript(run) if it.kind == "done")
    assert done.detail == "1 tool · 0 commits"
    # A resumed leg starts with loop.resume.start (never a second
    # session.start): the mode rides on it too, or the resumed plan's receipt
    # reads "0 commits".
    resumed_plan = [
        {"type": "loop.resume.start", "mode": "plan", "iteration": 3},
        {"type": "session.end", "reason": "finish_planning", "all_passed": False},
    ]
    done = next(it for it in fold_transcript(resumed_plan) if it.kind == "done")
    assert done.detail == "0 tools"


def test_every_end_reason_has_a_done_line_label() -> None:
    """The done marker words every end (`● stopped`, never `● steer_exit`):
    a reason added to SessionEndReason without a label leaked its raw token
    to the CLI, TUI and web transcripts."""
    from typing import get_args

    from agent6.viewmodel.transcript import _END_REASON_LABEL  # pyright: ignore[reportPrivateUsage]
    from agent6.workflows._session_state import SessionEndReason

    missing = set(get_args(SessionEndReason)) - set(_END_REASON_LABEL)
    assert not missing, f"end reasons with no done-line label: {sorted(missing)}"


def test_a_resumed_legs_receipt_is_its_own() -> None:
    """The done item of a resumed leg carries that leg's wall clock and
    counts, as it already carried its cost; the first leg's 45 s, tool and
    commit do not ride on a 3 s leg that did nothing."""
    events = [
        {"type": "session.start", "ts": "2026-08-09T20:00:00+00:00", "user_task": "t"},
        {"type": "tool.call", "name": "apply_edit", "args": {"path": "a.py"}},
        {"type": "tool.result", "name": "apply_edit", "ok": True, "summary": "ok"},
        {"type": "loop.auto_commit", "sha": "abc123", "subject": "first"},
        {"type": "diff.updated", "sha": "abc123", "patch": "+x\n"},
        {"type": "budget.update", "usd_total": 0.01},
        {"type": "session.end", "ts": "2026-08-09T20:00:45+00:00", "reason": "stopped"},
        {"type": "loop.resume.start", "ts": "2026-08-09T20:10:00+00:00", "user_task": "t"},
        {"type": "budget.update", "usd_total": 0.002},
        {"type": "tool.call", "name": "finish_session", "args": {"summary": "Done."}},
        {
            "type": "session.end",
            "ts": "2026-08-09T20:10:03+00:00",
            "reason": "finish_session",
            "all_passed": True,
        },
    ]
    dones = [it for it in fold_transcript(events) if it.kind == "done"]
    assert dones[-1].detail == "$0.0020 · 3s · 0 tools · 0 commits"


def test_a_tool_call_is_in_flight_until_its_result_settles_it() -> None:
    """The fold yields a call as soon as it is seen (`ok=None`: "running" on
    every surface), then its settled twin under the same call_id, which
    supersedes it; the batch form keeps one item per call, at the call's place."""
    from agent6.viewmodel.transcript import TranscriptFold

    call = {"type": "tool.call", "name": "run_command", "args": {"argv": ["sleep", "60"]}}
    result = {"type": "tool.result", "name": "run_command", "ok": True, "summary": "exit 0"}
    fold = TranscriptFold()
    (pending,) = fold.feed({**call, "call_id": 7})
    assert (pending.kind, pending.name, pending.arg) == ("tool", "run_command", "sleep 60")
    assert pending.ok is None and pending.call_id == "7"
    (settled,) = fold.feed({**result, "call_id": 7})
    assert settled.call_id == "7" and settled.ok is True and settled.detail == "exit 0"
    # Batch order is stream order: the settled item lands where the stream is
    # when the result arrives, after anything that landed during the call.
    aside = {"type": "btw.answered", "block": "--- btw: why\nbecause"}
    items = fold_transcript([{**call, "call_id": 7}, aside, {**result, "call_id": 7}])
    assert [(i.kind, i.ok) for i in items] == [("marker", None), ("tool", True)]


def test_an_approval_prompt_marks_the_call_it_gates_as_awaiting() -> None:
    """tool.call is journaled before the approval gate, so a gated call is in
    flight while its prompt is open: the fold says it waits (every surface
    reads it from here), and says it runs again once answered."""
    from agent6.viewmodel.transcript import TranscriptFold

    call = {"type": "tool.call", "name": "run_command", "args": {"argv": ["ls"]}, "call_id": 1}
    prompt = {
        "type": "approval.prompt",
        "id": "approval-1",
        "prompt": "Allow run_command: ls",
        "call_id": 1,
    }
    answer = {"type": "approval.answer", "id": "approval-1", "approved": True}
    fold = TranscriptFold()
    fold.feed(call)
    (waiting,) = fold.feed(prompt)
    assert (waiting.ok, waiting.call_id, waiting.detail) == (None, "1", "awaiting approval")
    (running,) = fold.feed(answer)
    assert (running.ok, running.call_id, running.detail) == (None, "1", "")
    (item,) = fold_transcript([call, prompt])
    assert item.detail == "awaiting approval"
    events = [{"type": "session.start", "user_task": "list"}, call, prompt]
    assert "[awaiting approval] run_command ls" in restate(events)
    assert fold_transcript([prompt]) == []  # a prompt gating no call (a pre-run confirmation)


def test_a_prompt_marks_the_call_it_names_not_the_newest_in_flight() -> None:
    """Two calls in flight (a concurrent review seat's read beside the gated
    command): the prompt carries the gated call's id, and only that call
    waits; the answer releases the same call. A prompt naming no call (an
    id-less historical journal, a verify the harness runs itself) marks
    nothing."""
    from agent6.viewmodel.transcript import TranscriptFold

    gated = {"type": "tool.call", "name": "run_command", "args": {"argv": ["ls"]}, "call_id": 1}
    newest = {"type": "tool.call", "name": "read_file", "args": {"path": "x"}, "call_id": 2}
    prompt = {"type": "approval.prompt", "id": "approval-1", "prompt": "Allow", "call_id": 1}
    answer = {"type": "approval.answer", "id": "approval-1", "approved": True}
    fold = TranscriptFold()
    fold.feed(gated)
    fold.feed(newest)
    (waiting,) = fold.feed(prompt)
    assert (waiting.name, waiting.call_id) == ("run_command", "1")
    assert waiting.detail == "awaiting approval"
    (running,) = fold.feed(answer)
    assert (running.name, running.call_id, running.detail) == ("run_command", "1", "")
    items = fold_transcript([gated, newest, prompt])
    assert {i.call_id: i.detail for i in items} == {"1": "awaiting approval", "2": ""}
    unnamed = {"type": "approval.prompt", "id": "approval-2", "prompt": "Allow"}
    (item,) = fold_transcript([gated, unnamed])
    assert item.detail == ""


def test_a_question_prompt_marks_the_ask_user_call_as_awaiting() -> None:
    """ask_user's call is journaled before its prompt, so while the operator
    answers the call is in flight: every surface read it as running (the
    approval pair was marked, the question pair was not)."""
    from agent6.viewmodel.transcript import TranscriptFold

    call = {
        "type": "tool.call",
        "name": "ask_user",
        "args": {"questions": ["which?"]},
        "call_id": 1,
    }
    prompt = {
        "type": "question.prompt",
        "id": "question-1",
        "questions": ["which?"],
        "call_id": 1,
    }
    answer = {"type": "question.answer", "id": "question-1", "answers": ["a"]}
    fold = TranscriptFold()
    fold.feed(call)
    (waiting,) = fold.feed(prompt)
    assert (waiting.ok, waiting.call_id, waiting.detail) == (None, "1", "awaiting answer")
    (running,) = fold.feed(answer)
    assert (running.ok, running.call_id, running.detail) == (None, "1", "")
    (item,) = fold_transcript([call, prompt])
    assert item.detail == "awaiting answer"
    events = [{"type": "session.start", "user_task": "pick"}, call, prompt]
    assert "[awaiting answer] ask_user" in restate(events)
    assert fold_transcript([prompt]) == []  # a question gating no call (a machine's own ask)


def test_a_dead_workers_open_call_settles_for_a_reader_that_knows() -> None:
    """A worker killed without a session.end leaves its last call open with no
    boundary to settle it; the reader that probes the worker settles it."""
    events = [
        {"type": "session.start", "user_task": "x"},
        {"type": "tool.call", "name": "run_command", "args": {"argv": ["sleep", "60"]}},
    ]
    (item,) = fold_transcript(events)
    assert item.ok is None
    (item,) = fold_transcript(events, worker_dead=True)
    assert item.ok is False and item.detail == "no result (the run died)"
    assert "[FAILED] run_command sleep 60: no result (the run died)" in restate(
        events, worker_dead=True
    )


def test_a_second_id_less_call_under_one_name_supersedes_the_first() -> None:
    """A journal with no call ids pairs by name: a second call under the same
    name before the first settles would orphan it in flight for good."""
    events = [
        {"type": "tool.call", "name": "read_file", "args": {"path": "a.py"}},
        {"type": "tool.call", "name": "read_file", "args": {"path": "b.py"}},
        {"type": "tool.result", "name": "read_file", "ok": True, "summary": "b bytes"},
    ]
    assert [(i.arg, i.ok, i.detail) for i in fold_transcript(events)] == [
        ("a.py", False, "no result (superseded)"),
        ("b.py", True, "b bytes"),
    ]


def test_a_leg_boundary_settles_a_call_that_never_returned() -> None:
    """A call still open at session.end (a crash, a kill) or at the next leg's
    start (a resume over one) did not return: it settles as such instead of
    reading "running" for the rest of time."""
    call = {"type": "tool.call", "name": "run_verify_command", "args": {}, "call_id": 3}
    items = fold_transcript([call, {"type": "session.end", "reason": "crashed"}])
    assert [i.kind for i in items] == ["tool", "done"]
    assert items[0].ok is False and items[0].call_id == "3" and "no result" in items[0].detail
    items = fold_transcript([call, {"type": "loop.resume.start"}])
    assert [(i.kind, i.ok) for i in items] == [("tool", False)]


def test_restate_names_a_call_still_running() -> None:
    events = [
        {"type": "session.start", "user_task": "wait a bit"},
        {"type": "tool.call", "name": "run_command", "args": {"argv": ["sleep", "60"]}},
    ]
    assert "[running] run_command sleep 60" in restate(events)


def test_compaction_renders_as_markers_on_the_conversation() -> None:
    """No `loop.compact.*` event produced an item, so the conversation surfaces
    (TUI, CLI stream, web, ACP) showed nothing when tier 2 replaced the history
    the operator was reading, when the summariser failed, or when a `/compact`
    the surface had promised was refused; only the log pane said so."""
    items = fold_transcript(
        [
            {"type": "loop.compact.requested", "focus": "the parser"},
            {"type": "loop.compact.refused", "reason": "too little history to summarise"},
            {"type": "loop.compact.requested", "focus": ""},
            {
                "type": "loop.compact.summarise.done",
                "summary_chars": 1200,
                "summary": "...",
                "kept_turns": 3,
            },
            {"type": "loop.compact.summarise.failed", "error": "429 rate limited"},
        ]
    )
    assert [it.kind for it in items] == ["marker"] * 5
    assert items[0].body == "compaction requested: the parser"
    assert items[1].body == "compaction refused: too little history to summarise"
    assert items[2].body == "compaction requested"
    assert items[3].body == "context compacted: 1,200-char summary, 3 recent turns kept verbatim"
    assert items[4].body == "compaction failed: 429 rate limited"


def test_parallel_compared_renders_the_ranking() -> None:
    """A fan-out's journal records the auto-compare: the marker lists the
    candidates best first with their gate verdict and cost, and names the
    judge or the mechanical fallback."""
    (item,) = fold_transcript(
        [
            {
                "type": "loop.parallel.compared",
                "group": "fan",
                "ranked_by": "judge",
                "ranking": [
                    {"session_id": "fan-l2", "verify": "passed", "cost_usd": 0.05},
                    {"session_id": "fan-l1", "verify": "failed", "cost_usd": 0.07},
                ],
            }
        ]
    )
    assert item.kind == "marker"
    assert "compared group fan: 2 candidate(s), ranked by judge" in item.body
    assert "1. fan-l2  passed  $0.05" in item.body
    assert "2. fan-l1  failed  $0.07" in item.body


def test_parallel_dispatched_counts_lanes_when_the_event_carries_them() -> None:
    """A fan-out (or a group whose lane count is known at dispatch) names its
    lanes, the count every listing shows; the task count alone said
    "dispatched 1 parallel task" over two lanes."""
    (item,) = fold_transcript(
        [{"type": "loop.parallel.dispatched", "group": "fan", "lanes": 2, "tasks": ["t"]}]
    )
    assert "dispatched 2 lanes (group fan)" in item.body
    (item,) = fold_transcript(
        [{"type": "loop.parallel.dispatched", "group": "p1", "lanes": 3, "tasks": ["a", "b"]}]
    )
    assert "dispatched 3 lanes for 2 tasks (group p1)" in item.body
