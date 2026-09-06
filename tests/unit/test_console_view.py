# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""ConsoleView: the live CLI stream shows tools and never a blank response block."""

from __future__ import annotations

from io import StringIO
from typing import Any

import pytest

from agent6.ui.cli._console_view import ConsoleView


def _render(events: list[dict[str, object]]) -> str:
    buf = StringIO()
    view = ConsoleView(buf, color=False)
    for event in events:
        view.feed(event)
    return buf.getvalue()


def test_reasoning_tool_call_and_result_all_render() -> None:
    out = _render(
        [
            {"type": "session.start", "user_task": "fix the failing test"},
            {"type": "role.call", "role": "worker"},
            {"type": "role.thinking_delta", "role": "worker", "text": "let me read the file"},
            {"type": "role.result", "role": "worker"},
            {"type": "tool.call", "name": "read_file", "args": {"path": "a.py"}},
            {"type": "tool.result", "name": "read_file", "ok": True, "summary": "12 bytes"},
            {"type": "session.end", "all_passed": True, "reason": "finish_session"},
        ]
    )
    assert "fix the failing test" in out
    assert "let me read the file" in out  # reasoning shown
    assert "→ read_file" in out and "a.py" in out  # the tool call, invisible before
    assert "└" in out and "12 bytes" in out  # its result
    assert "done" in out


def test_whitespace_only_text_prints_no_empty_block() -> None:
    # The turn streams only whitespace text then calls a tool. The old renderer
    # printed a "── worker: response ──" bar with nothing under it; this must not.
    out = _render(
        [
            {"type": "role.call", "role": "worker"},
            {"type": "role.text_delta", "role": "worker", "text": "  \n "},
            {"type": "role.result", "role": "worker"},
            {"type": "tool.call", "name": "apply_edit", "args": {}},
            {"type": "tool.result", "name": "apply_edit", "ok": True, "summary": "ok"},
        ]
    )
    assert "worker: response" not in out
    non_empty = [ln for ln in out.splitlines() if ln.strip()]
    assert non_empty and non_empty[0].strip().startswith("→ apply_edit")


def test_failed_tool_shows_its_output_tail() -> None:
    out = _render(
        [
            {"type": "tool.call", "name": "run_command", "args": {"command": "ls /nope"}},
            {
                "type": "tool.result",
                "name": "run_command",
                "ok": False,
                "summary": "exit=2",
                "stderr_tail": "ls: /nope: No such file or directory",
            },
        ]
    )
    assert "→ run_command" in out
    assert "No such file" in out


def test_steer_request_closes_open_dim_block() -> None:
    # A Ctrl-C pause message prints to the same terminal; the open dim thinking
    # block must be closed (reset) first so the message doesn't inherit the dim.
    buf = StringIO()
    view = ConsoleView(buf, color=True)
    view.feed({"type": "role.thinking_delta", "text": "pondering the fix"})
    assert not buf.getvalue().endswith("\033[0m\n")  # block still open
    view.feed({"type": "session.steer_requested", "source": "sigint"})
    assert buf.getvalue().endswith("\033[0m\n")  # closed + reset before the message prints


def _graph_event(nodes: dict[str, Any], cursor: str | None = None) -> dict[str, object]:
    return {"type": "graph.update", "nodes": nodes, "cursor": cursor}


def test_plan_block_prints_when_the_dag_is_seeded() -> None:
    nodes = {
        "01A": {
            "title": "root task",
            "status": "in_progress",
            "parent_id": None,
            "children": ["01B", "01C"],
        },
        "01B": {"title": "survey", "status": "passed", "parent_id": "01A", "children": []},
        "01C": {"title": "implement", "status": "pending", "parent_id": "01A", "children": []},
    }
    out = _render([_graph_event(nodes, cursor="01C")])
    assert "plan (3 tasks)" in out
    assert "root task" in out and "survey" in out and "implement" in out
    # Nesting: children are indented under the root.
    assert "  ✓ survey" in out


def test_plan_block_reprints_only_when_the_dag_grows() -> None:
    n1 = {
        "01A": {"title": "root", "status": "pending", "parent_id": None, "children": ["01B"]},
        "01B": {"title": "a", "status": "pending", "parent_id": "01A", "children": []},
    }
    n2 = dict(n1)  # same set of tasks -> no reprint
    n3 = {**n1, "01C": {"title": "b", "status": "pending", "parent_id": "01A", "children": []}}
    n1["01A"]["children"] = ["01B"]
    n3["01A"]["children"] = ["01B", "01C"]
    out = _render([_graph_event(n1), _graph_event(n2), _graph_event(n3)])
    assert out.count("plan (2 tasks)") == 1  # seeded once
    assert out.count("plan (3 tasks)") == 1  # reprinted when it grew, not on the no-op update


def test_single_root_task_is_not_a_plan_block() -> None:
    # A plain run seeds one root task; that is not a decomposition worth a block.
    out = _render(
        [
            _graph_event(
                {"01A": {"title": "t", "status": "in_progress", "parent_id": None, "children": []}}
            )
        ]
    )
    assert "plan (" not in out


class _FakeTTY:
    """A tty-like sink: isatty() True so ConsoleView starts its heartbeat thread."""

    def __init__(self) -> None:
        self.chunks: list[str] = []

    def write(self, s: str) -> int:
        self.chunks.append(s)
        return len(s)

    def flush(self) -> None:
        pass

    def isatty(self) -> bool:
        return True

    def getvalue(self) -> str:
        return "".join(self.chunks)


# Comfortably past the module's stall threshold (1.5s) + a couple 0.5s ticks.
_STALL_WAIT_S = 3.0


def test_cli_heartbeat_shows_working_when_the_stream_stalls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A turn that goes silent mid-flight (a stalled SSE stream) shows a ticking
    'working… Ns' line so the CLI never looks hung (the user's exact symptom).
    Mid-block the real threshold is 10s (see the no-split test below); shrink it
    so the genuine-stall path stays testable without a 10s sleep."""
    import time

    monkeypatch.setattr("agent6.ui.cli._console_view._MID_BLOCK_STALL_S", 1.5)
    out = _FakeTTY()
    view = ConsoleView(out, color=False)  # type: ignore[arg-type]
    try:
        view.feed({"type": "role.call", "role": "worker", "model": "m"})
        view.feed({"type": "role.text_delta", "text": "Let me investigate"})
        time.sleep(_STALL_WAIT_S)  # let the stall register and the spinner tick
        assert "working…" in out.getvalue()
        # Output resuming clears the spinner (\r + erase) and shows the new text.
        view.feed({"type": "role.text_delta", "text": " the theme system"})
        assert "the theme system" in out.getvalue()
        assert "\x1b[2K" in out.getvalue()  # the spinner line was erased
    finally:
        view.close()


def test_cli_heartbeat_does_not_split_a_streaming_block_at_short_gaps() -> None:
    """A few-seconds gap in a flowing prose block must NOT draw the spinner:
    doing so closes the block and the next delta opens a new bullet, visibly
    splitting a streamed word (a file path, mid-token) in two."""
    import time

    out = _FakeTTY()
    view = ConsoleView(out, color=False)  # type: ignore[arg-type]
    try:
        view.feed({"type": "role.call", "role": "worker", "model": "m"})
        view.feed({"type": "role.text_delta", "text": "writing to tests/test"})
        time.sleep(_STALL_WAIT_S)  # over the between-blocks threshold, under 10s
        view.feed({"type": "role.text_delta", "text": "_calc.py now"})
        got = out.getvalue()
        assert "working…" not in got  # no spinner mid-block at a short gap
        assert "tests/test_calc.py" in got  # the streamed path stayed whole
    finally:
        view.close()


def test_cli_heartbeat_spins_during_a_long_tool_run() -> None:
    """A long verify / run_command executes between role.result and the next
    role.call; the heartbeat must still spin so a running test suite doesn't look
    frozen (gap: a role-only flag missed this)."""
    import time

    out = _FakeTTY()
    view = ConsoleView(out, color=False)  # type: ignore[arg-type]
    try:
        view.feed({"type": "role.call", "role": "worker", "model": "m"})
        view.feed({"type": "role.result", "role": "worker"})  # turn done...
        # ...now a tool.call starts a long jail command (no result yet).
        view.feed({"type": "tool.call", "name": "run_verify_command", "args": {}})
        time.sleep(_STALL_WAIT_S)
        assert "working…" in out.getvalue()  # spins through the command, not frozen
    finally:
        view.close()


def test_cli_heartbeat_silent_on_a_non_tty() -> None:
    """No spinner thread (and no spinner bytes) when the sink is not a terminal --
    a piped/redirected run or a test stays clean."""
    import time

    buf = StringIO()
    view = ConsoleView(buf, color=False)
    view.feed({"type": "role.call", "role": "worker", "model": "m"})
    time.sleep(_STALL_WAIT_S)
    assert "working…" not in buf.getvalue()
    view.close()


def test_notice_clears_the_spinner_before_printing() -> None:
    """A workflow notice (auto-commit, review) routes through the ConsoleView so
    it clears the spinner line first and writes to the same stream -- no garble
    with the stderr heartbeat on a shared terminal."""
    import time

    out = _FakeTTY()
    view = ConsoleView(out, color=False)  # type: ignore[arg-type]
    try:
        view.feed({"type": "role.call", "role": "worker", "model": "m"})
        time.sleep(_STALL_WAIT_S)  # spinner up
        assert "working…" in out.getvalue()
        view.notice("[agent6]   auto-commit: abc123")
        v = out.getvalue()
        assert "auto-commit: abc123" in v
        assert "\x1b[2K" in v  # the spinner line was erased before the notice
    finally:
        view.close()


def test_pause_suspends_the_heartbeat_spinner() -> None:
    """An interactive /dev/tty prompt (ask_user, a run_command approval) wraps the
    read in console_view.pause() so the spinner stops erasing the question and the
    operator's keystrokes; it resumes once the prompt returns."""
    import time

    out = _FakeTTY()
    view = ConsoleView(out, color=False)  # type: ignore[arg-type]
    try:
        view.feed({"type": "tool.call", "name": "ask_user", "args": {}})
        time.sleep(_STALL_WAIT_S)  # a tool is in flight + output silent: spinner up
        assert "working…" in out.getvalue()
        with view.pause():
            marker = len(out.chunks)  # after pause() cleared the spinner line
            time.sleep(1.2)  # a couple ticks: the prompt owns the terminal now
            assert "working" not in "".join(out.chunks[marker:])  # no spinner redraws
        resume = len(out.chunks)
        time.sleep(1.2)  # after the prompt the heartbeat is free to spin again
        assert "working" in "".join(out.chunks[resume:])
    finally:
        view.close()


def test_replayed_history_does_not_reset_the_idle_timer() -> None:
    """`agent6 attach` replays the whole log through feed(): each replayed line
    bumped the idle anchor to ARRIVAL time, so a run wedged 40 minutes read
    "working… 3s" -- the timer meant to tell thinking from hung concealed the
    hang. The anchor is the fed event's own ts."""
    import time

    out = _FakeTTY()
    view = ConsoleView(out, color=False)  # type: ignore[arg-type]
    try:
        wedged_at = time.time() - 2400
        view.feed({"type": "role.call", "role": "worker", "model": "m", "ts": wedged_at - 1})
        view.feed({"type": "tool.call", "name": "run_command", "args": {}, "ts": wedged_at})
        idle = time.monotonic() - view._last_output_at  # pyright: ignore[reportPrivateUsage]
        assert idle >= 2399, f"replay reset the idle timer to {idle:.1f}s"
        # A ts-less (live, foreign) event keeps the arrival behaviour.
        view.feed({"type": "tool.result", "name": "run_command", "ok": True, "summary": "ok"})
        idle = time.monotonic() - view._last_output_at  # pyright: ignore[reportPrivateUsage]
        assert idle < 5
    finally:
        view.close()


def test_streamed_model_text_cannot_reach_the_terminal_with_controls() -> None:
    """The live CLI stream printed model deltas raw: the fold's previews were
    scrubbed but this path was not, so OSC 52 in streamed text could write the
    operator's clipboard. A split sequence cannot reassemble: the opener's
    piece loses its tail, and the continuation prints as inert text."""
    out = _FakeTTY()
    view = ConsoleView(out, color=False)  # type: ignore[arg-type]
    try:
        view.feed({"type": "role.call", "role": "worker", "model": "m"})
        view.feed({"type": "role.text_delta", "text": "safe \x1b]52;c;cGF5"})
        view.feed({"type": "role.text_delta", "text": "bG9hZA==\x07 more"})
        got = out.getvalue()
        assert "\x1b]52" not in got and "\x07" not in got
        assert "safe" in got and "more" in got
    finally:
        view.close()


def test_the_policy_line_is_read_when_the_task_prints() -> None:
    """The gate is inferred and pinned AFTER the view is built and BEFORE
    session.start; a policy read at construction said "no verify gate" over a
    run that had one, so the line is read when it prints."""
    buf = StringIO()
    facts = ["kimi · strict · commands ask · no verify gate"]
    view = ConsoleView(buf, color=False, policy=lambda: facts[0])
    facts[0] = "kimi · strict · commands ask · ./verify.sh (inferred)"
    view.feed({"type": "session.start", "user_task": "add mul"})
    view.close()
    assert "./verify.sh (inferred)" in buf.getvalue()
    assert "no verify gate" not in buf.getvalue()


def test_the_task_headline_is_the_first_line_clipped() -> None:
    """A `--from-plan` task carries the whole plan; flattening it made the
    headline one endless line. The first user-authored line, clipped, is the
    headline every other surface shows."""
    out = _render(
        [
            {
                "type": "session.start",
                "user_task": (
                    "Execute the prepared plan: validate inputs\n\n# Plan\n\n1. add\n2. test"
                ),
            },
        ]
    )
    assert "Execute the prepared plan: validate inputs" in out
    assert "# Plan" not in out and "1. add" not in out


def test_the_receipt_reads_the_mode_from_the_start_the_console_prints_itself() -> None:
    """The console prints its own headline for session.start; the fold must
    still see the event (mode, first timestamp), or an ask's receipt reads
    "0 commits" on the live console while every other surface omits it."""
    out = _render(
        [
            {"type": "session.start", "mode": "ask", "user_task": "why?"},
            {"type": "tool.call", "name": "read_file", "args": {"path": "a"}},
            {"type": "tool.result", "name": "read_file", "ok": True, "summary": "1 byte"},
            {"type": "session.end", "reason": "answered", "all_passed": False},
        ]
    )
    assert "1 tool" in out
    assert "0 commits" not in out
    assert out.count("why?") == 1  # the headline once, no operator item for the start


def test_the_cli_prints_a_tool_once_when_it_settles() -> None:
    """The fold announces a call before its result; the CLI's heartbeat covers
    the wait, so the call prints once, head and result together."""
    out = _render(
        [
            {"type": "tool.call", "name": "read_file", "args": {"path": "a.py"}},
            {"type": "tool.result", "name": "read_file", "ok": True, "summary": "12 bytes"},
        ]
    )
    assert out.count("→ read_file") == 1
    assert "running" not in out


def test_a_provider_retry_says_so_instead_of_resetting_the_clock() -> None:
    """The retry event bumps the idle clock, so with nothing rendered the
    "working… Ns" counter restarted with no explanation: a run wedged behind
    four provider failures read as freshly started, and the Ctrl-C hint (which
    needs 20s idle) never appeared."""
    out = _render(
        [
            {
                "type": "loop.provider.retry",
                "attempt": 2,
                "error": "HTTP error calling https://api.example/v1: connection refused",
            }
        ]
    )
    assert "retrying after a provider error (attempt 2)" in out
    assert "connection refused" in out
