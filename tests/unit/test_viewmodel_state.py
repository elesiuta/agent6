# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Tests for the pure event-fold in agent6.viewmodel.state."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from agent6.viewmodel.state import (
    ApprovalPrompt,
    BudgetView,
    SessionState,
    TaskNodeView,
    apply_event,
    fold_session,
    format_log_line,
    initial_state,
    session_state_as_dict,
    session_status_label,
)


def test_format_log_line_keeps_the_compaction_reason() -> None:
    """A failed compaction carries its reason in `error`; with no case for these
    types the log view printed the bare event name and dropped it, so a 429'd
    summariser read as nothing having happened. The operator's /compact focus
    was invisible for the same reason."""
    failed = format_log_line(
        {"type": "loop.compact.summarise.failed", "error": "provider 429: rate limited"}
    )
    assert "429" in failed
    gist = format_log_line({"type": "loop.compact.gist.failed", "error": "boom"})
    assert "boom" in gist
    requested = format_log_line({"type": "loop.compact.requested", "focus": "keep the auth work"})
    assert "keep the auth work" in requested


def test_context_fill_is_the_one_rule_and_rides_the_wire(monkeypatch: pytest.MonkeyPatch) -> None:
    """The `ctx N%` readout every surface shows (TUI header + composer, the
    pause menu, the web budget card) comes from one rule: the last completed
    call's prompt tokens over the model's window; None until both are known.
    The wire dict carries it as `context_pct`."""
    from agent6.viewmodel import state as state_mod
    from agent6.viewmodel.state import apply_event, context_fill, initial_state

    def _known(_provider: str, _model: str) -> int | None:
        return 100_000

    def _unknown(_provider: str, _model: str) -> int | None:
        return None

    monkeypatch.setattr(state_mod, "_window", _known)
    s = initial_state()
    assert context_fill(s) is None  # no call yet
    for ev in (
        {"type": "session.start", "user_task": "x", "mode": "run"},
        {"type": "role.call", "role": "worker", "model": "m", "provider": "p"},
        {
            "type": "role.result",
            "role": "worker",
            "ok": True,
            "tokens_in": 1_000,
            "cache_read": 40_000,
        },
    ):
        s = apply_event(s, ev)
    assert context_fill(s) == 41
    assert session_state_as_dict(s)["context_pct"] == 41
    monkeypatch.setattr(state_mod, "_window", _unknown)
    assert context_fill(s) is None  # an unknown window is not a percent


def test_format_log_line_names_the_pins_in_force() -> None:
    """loop.pin.restored announces the pins in force at leg start, whether a
    fresh run's --pin or a resume's snapshot; the line said "restored from the
    snapshot" for both, false for --pin, and named none of them."""
    line = format_log_line(
        {"type": "loop.pin.restored", "pins": ["never touch test_calc.py"], "count": 1}
    )
    assert "1 pinned: never touch test_calc.py" in line
    assert "snapshot" not in line
    assert "no pinned instructions" in format_log_line(
        {"type": "loop.pin.restored", "pins": [], "count": 0}
    )


def test_format_log_line_tool_result_appends_output_tail() -> None:
    """An execution tool's tool.result line shows a one-line stderr/stdout hint
    (full tail is in the event), while a plain result stays summary-only."""
    line = format_log_line(
        {
            "type": "tool.result",
            "name": "run_command",
            "ok": True,
            "summary": "exit=1 in 0.2s",
            "stderr_tail": "boom: file not found\nsecond line",
        }
    )
    assert "run_command" in line and "exit=1" in line and "boom: file not found" in line
    plain = format_log_line(
        {"type": "tool.result", "name": "read_file", "ok": True, "summary": "40 bytes"}
    )
    assert "|" not in plain  # no tail hint for non-execution tools


def _graph_event(nodes: dict[str, Any], cursor: str | None = None) -> dict[str, Any]:
    return {"type": "graph.update", "nodes": nodes, "cursor": cursor}


def test_initial_state_is_empty() -> None:
    s = initial_state()
    assert s == SessionState()
    assert s.tasks == ()
    assert s.budget == BudgetView()


def test_run_start_records_task() -> None:
    s = apply_event(initial_state(), {"type": "session.start", "user_task": "fix bug"})
    assert s.user_task == "fix bug"


def test_run_start_records_run_id() -> None:
    # The loop now stamps the run dir name into session.start; the fold picks it up,
    # so watch --json / the web snapshot report a real id, not "".
    s = apply_event(
        initial_state(),
        {"type": "session.start", "session_id": "deep-granite-CSSYTJ", "user_task": "t"},
    )
    assert s.session_id == "deep-granite-CSSYTJ"


def test_graph_update_builds_task_tree_dfs_with_depth() -> None:
    # root -> (a -> a1), b  (children order preserved, DFS pre-order, depths)
    nodes = {
        "root": {
            "title": "root",
            "parent_id": None,
            "status": "in_progress",
            "children": ["a", "b"],
        },
        "a": {"title": "task a", "parent_id": "root", "status": "passed", "children": ["a1"]},
        "a1": {"title": "task a1", "parent_id": "a", "status": "pending", "children": []},
        "b": {"title": "task b", "parent_id": "root", "status": "failed", "children": []},
    }
    s = apply_event(initial_state(), _graph_event(nodes, cursor="a1"))
    assert s.cursor_task_id == "a1"
    assert s.tasks == (
        TaskNodeView(id="root", title="root", status="in_progress", depth=0),
        TaskNodeView(id="a", title="task a", status="passed", depth=1),
        TaskNodeView(id="a1", title="task a1", status="pending", depth=2, is_cursor=True),
        TaskNodeView(id="b", title="task b", status="failed", depth=1),
    )


def test_graph_update_latest_snapshot_replaces_prior() -> None:
    s = apply_event(
        initial_state(), _graph_event({"r": {"title": "r", "parent_id": None, "children": []}})
    )
    assert len(s.tasks) == 1
    s = apply_event(s, _graph_event({"r2": {"title": "r2", "parent_id": None, "children": []}}))
    assert tuple(t.id for t in s.tasks) == ("r2",)


def test_graph_update_guards_cycles() -> None:
    # a -> b -> a (cycle) must not infinite-loop; each node appears once.
    nodes = {
        "a": {"title": "a", "parent_id": None, "status": "pending", "children": ["b"]},
        "b": {"title": "b", "parent_id": "a", "status": "pending", "children": ["a"]},
    }
    s = apply_event(initial_state(), _graph_event(nodes))
    assert tuple(t.id for t in s.tasks) == ("a", "b")


def test_tool_call_then_result_pairs_up() -> None:
    s = apply_event(
        initial_state(),
        {"type": "tool.call", "name": "read_file", "args": {"path": "foo.py"}},
    )
    assert len(s.tool_calls) == 1
    assert s.tool_calls[0].name == "read_file"
    assert s.tool_calls[0].ok is None
    s = apply_event(
        s, {"type": "tool.result", "name": "read_file", "ok": True, "summary": "100 bytes"}
    )
    assert s.tool_calls[0].ok is True
    assert s.tool_calls[0].result_summary == "100 bytes"


def test_apply_edit_args_render_the_edit_kinds_not_a_dict_repr() -> None:
    # The drawer + TUI tool table read args_preview; apply_edit's `edits` list
    # used to leak `[{'kind': 'replace', 'old_string': ...}]` as a Python repr.
    s = apply_event(
        initial_state(),
        {
            "type": "tool.call",
            "name": "apply_edit",
            "args": {
                "path": "calc.py",
                "edits": [
                    {"kind": "replace", "old_string": "a", "new_string": "b"},
                    {"kind": "create", "old_string": "", "new_string": "x"},
                ],
            },
        },
    )
    preview = s.tool_calls[0].args_preview
    assert "old_string" not in preview and "{" not in preview
    assert "path=calc.py" in preview and "edits=replace, create" in preview


def test_dict_valued_tool_summary_renders_as_json_not_python_repr() -> None:
    # A malformed dict summary must not leak `{'unexpected': ...}` (single-quoted
    # Python repr) into the tool table or the log tail.
    s = apply_event(initial_state(), {"type": "tool.call", "name": "weird", "args": {}})
    s = apply_event(
        s, {"type": "tool.result", "name": "weird", "ok": True, "summary": {"unexpected": 1}}
    )
    assert s.tool_calls[0].result_summary == '{"unexpected": 1}'
    line = format_log_line(
        {"type": "tool.result", "name": "weird", "ok": True, "summary": {"unexpected": 1}}
    )
    assert '{"unexpected": 1}' in line and "'unexpected'" not in line


def test_tool_result_with_mismatched_name_does_not_overwrite() -> None:
    s = apply_event(initial_state(), {"type": "tool.call", "name": "read_file", "args": {}})
    s = apply_event(s, {"type": "tool.result", "name": "other", "ok": True, "summary": "x"})
    assert s.tool_calls[0].ok is None


def test_role_call_then_result_clears_in_flight() -> None:
    s = apply_event(
        initial_state(),
        {"type": "role.call", "role": "worker", "model": "gpt-5", "provider": "openai"},
    )
    assert s.last_role is not None
    assert s.last_role.in_flight is True
    s = apply_event(s, {"type": "role.result", "role": "worker", "tokens_in": 10, "tokens_out": 20})
    assert s.last_role is not None
    assert s.last_role.in_flight is False


def test_budget_update_populates_view() -> None:
    s = apply_event(
        initial_state(),
        {
            "type": "budget.update",
            "input_total": 100,
            "output_total": 50,
            "usd_cap": 10.0,
            "tokens_unmetered": 30,
            "tokens_fallback_cap": 1000,
        },
    )
    assert s.budget.input_total == 100
    assert s.budget.usd_cap == 10.0
    assert s.budget.tokens_unmetered == 30
    assert s.budget.tokens_fallback_cap == 1000
    # USD fields default cleanly when the event omits them.
    assert s.budget.usd_total == 0.0
    assert s.budget.usd_partial is False
    # budget.update never carries a per-model token breakdown (no emitter ever
    # wrote one, and no surface reads it); the fold must not resurrect a
    # permanently-empty field. The per-model breakdown is surfaced from the
    # budget snapshot directly, not the viewmodel.
    assert not hasattr(s.budget, "per_model_tokens")


def test_budget_update_carries_usd_total() -> None:
    s = apply_event(
        initial_state(),
        {
            "type": "budget.update",
            "input_total": 100,
            "output_total": 50,
            "input_cap": 1000,
            "output_cap": 500,
            "usd_total": 0.1234,
            "usd_partial": True,
        },
    )
    assert s.budget.usd_total == 0.1234
    assert s.budget.usd_partial is True


def test_budget_usd_cumulative_across_resume_legs() -> None:
    # Each resume leg's budget.update restarts usd_total from 0; the view banks
    # the finished leg on loop.resume.start so "cost" stays the cumulative
    # spend -- the same rule the hub scanner applies (listing.scan_session_log),
    # keeping the hub row and the run view in agreement.
    def _update(usd: float, *, partial: bool = False) -> dict[str, object]:
        return {"type": "budget.update", "usd_total": usd, "usd_partial": partial}

    s = apply_event(initial_state(), _update(0.02, partial=True))
    s = apply_event(s, {"type": "session.end", "reason": "finish_session", "all_passed": True})
    s = apply_event(s, {"type": "loop.resume.start"})
    # Banked, and the header keeps the old total until the new leg reports.
    assert s.budget.usd_total == 0.02
    s = apply_event(s, _update(0.005))
    assert s.budget.usd_total == pytest.approx(0.025)
    # partial is sticky: leg 1's unpriced spend keeps the total an under-estimate.
    assert s.budget.usd_partial is True
    # A second resume banks the cumulative, not just the last leg.
    s = apply_event(s, {"type": "loop.resume.start"})
    s = apply_event(s, _update(0.001))
    assert s.budget.usd_total == pytest.approx(0.026)


def test_verify_lifecycle() -> None:
    s = apply_event(initial_state(), {"type": "verify.start", "cmd": ["pytest", "-q"]})
    assert s.last_verify is not None
    assert s.last_verify.exit_code is None
    s = apply_event(
        s,
        {
            "type": "verify.end",
            "cmd": ["pytest", "-q"],
            "exit_code": 0,
            "duration_s": 1.5,
            "stdout_tail": "ok",
            "stderr_tail": "",
        },
    )
    assert s.last_verify is not None
    assert s.last_verify.exit_code == 0
    assert s.last_verify.duration_s == 1.5


def test_approval_prompt_then_answer() -> None:
    s = apply_event(
        initial_state(),
        {"type": "approval.prompt", "id": "a001", "prompt": "Allow run_command?"},
    )
    assert len(s.pending_approvals) == 1
    assert s.pending_approvals[0] == ApprovalPrompt(id="a001", prompt="Allow run_command?")
    assert s.pending_approvals[0].head == "Allow run_command?"
    assert s.pending_approvals[0].payload == ""
    s = apply_event(s, {"type": "approval.answer", "id": "a001", "approved": True})
    assert s.pending_approvals[0].answered is True
    assert s.pending_approvals[0].approved is True


def test_diff_updated_stores_latest_patch() -> None:
    s = apply_event(initial_state(), {"type": "diff.updated", "sha": "abc", "patch": "diff text"})
    assert s.latest_diff == "diff text"
    # a newer diff replaces the prior one
    s = apply_event(s, {"type": "diff.updated", "sha": "def", "patch": "newer"})
    assert s.latest_diff == "newer"


def test_run_end_marks_finished() -> None:
    s = apply_event(initial_state(), {"type": "session.end", "all_passed": True})
    assert s.finished is True
    assert s.all_passed is True


def test_unknown_event_type_still_appends_to_log() -> None:
    s = apply_event(initial_state(), {"type": "totally.new", "x": 1})
    # Unknown events should not change SessionState identity-relevant fields
    # but DO go into the log tail.
    assert s.tasks == ()
    assert len(s.log_tail) == 1


def test_tool_history_is_bounded() -> None:
    s = initial_state()
    for i in range(200):
        s = apply_event(s, {"type": "tool.call", "name": f"t{i}", "args": {}})
    assert len(s.tool_calls) == 50  # _MAX_TOOL_HISTORY


def test_full_run_trace_replay() -> None:
    """End-to-end: feed a plausible event sequence and assert final state."""
    tasks = {
        "one": {"title": "one", "parent_id": None, "status": "passed", "children": []},
        "two": {"title": "two", "parent_id": None, "status": "failed", "children": []},
    }
    events = [
        {"type": "session.start", "user_task": "do thing"},
        {"type": "graph.update", "nodes": tasks, "cursor": "two"},
        {"type": "role.call", "role": "worker", "model": "gpt-5"},
        {"type": "tool.call", "name": "apply_patch", "args": {"path": "x.py"}},
        {"type": "tool.result", "name": "apply_patch", "ok": True, "summary": "applied=1"},
        {"type": "role.result", "role": "worker", "tokens_in": 50, "tokens_out": 100},
        {
            "type": "budget.update",
            "input_total": 50,
            "output_total": 100,
            "input_cap": 1000,
            "output_cap": 1000,
        },
        {"type": "diff.updated", "sha": "abc", "patch": "+ added"},
        {"type": "session.end", "all_passed": False},
    ]
    s = initial_state()
    for e in events:
        s = apply_event(s, e)
    assert s.finished is True
    assert s.all_passed is False
    assert s.tasks[0].status == "passed"
    assert s.tasks[1].status == "failed"
    assert s.cursor_task_id == "two"
    assert s.latest_diff == "+ added"
    assert s.budget.input_total == 50


def test_log_count_is_monotonic_past_window_cap() -> None:
    # log_tail is a sliding window (MAX_LOG_TAIL); log_count must keep growing
    # so a live viewer can diff on it. A length-based diff freezes once the
    # window saturates -- this is the bug log_count fixes.
    from agent6.viewmodel.state import MAX_LOG_TAIL

    s = initial_state()
    n = MAX_LOG_TAIL + 50
    for i in range(n):
        s = apply_event(s, {"type": "loop.note", "msg": f"line {i}"})
    assert len(s.log_tail) == MAX_LOG_TAIL  # window stays capped
    assert s.log_count == n  # but the count keeps climbing


def test_fold_run_reduces_events_to_a_snapshot() -> None:
    state = fold_session(
        [
            {"type": "session.start", "user_task": "do it"},
            {"type": "role.call", "role": "worker", "model": "m"},
            {"type": "role.text_delta", "text": "hi"},
            {"type": "session.end", "all_passed": True},
        ]
    )
    assert state.user_task == "do it"
    assert state.finished and state.all_passed
    assert state.last_role is not None and state.last_role.streamed_text == "hi"


def test_run_state_as_dict_is_json_serializable() -> None:
    import json

    state = fold_session(
        [
            {"type": "session.start", "user_task": "t"},
            {"type": "tool.call", "name": "grep", "args": {"q": "x"}},
        ]
    )
    d = session_state_as_dict(state)
    assert d["user_task"] == "t"
    assert d["tool_calls"][0]["name"] == "grep"  # tuple -> list, dataclass -> dict
    json.dumps(d)  # the wire form must serialize


def test_run_state_as_dict_owns_the_dir_backed_identity(tmp_path: Path) -> None:
    """A resumed/forked leg's log can start at loop.resume.start (no session.start),
    folding session_id/user_task empty. With the dir in hand THE wire owner fills
    them (dir name + manifest task) so no consumer patches its own copy."""
    import json

    session_dir = tmp_path / "sunny-otter-K4Q7B2"
    session_dir.mkdir()
    (session_dir / "manifest.json").write_text(
        json.dumps({"version": 2, "user_task": "queued work"}), encoding="utf-8"
    )
    d = session_state_as_dict(fold_session([]), session_dir)
    assert d["session_id"] == "sunny-otter-K4Q7B2"
    assert d["user_task"] == "queued work"
    assert d["live"] is False  # no worker: the dir word is not live


def test_run_state_as_dict_always_carries_live(tmp_path: Path) -> None:
    # `live` is part of the wire shape: None (unknowable) without a dir, a real
    # bool with one -- never an absent key a client must typeof-probe.
    assert session_state_as_dict(fold_session([]))["live"] is None
    d = tmp_path / "r"
    d.mkdir()
    assert isinstance(session_state_as_dict(fold_session([]), d)["live"], bool)


def test_run_state_as_dict_flags_operator_blocked() -> None:
    """The wire carries operator_blocked from the fold so a DIR-LESS consumer (the
    machine watch, which folds an agent-state log with no session_dir) can quiet its
    heartbeat when the agent is blocked on a prompt, not paint 'agent working…'."""
    idle = session_state_as_dict(fold_session([{"type": "session.start", "user_task": "t"}]))
    assert idle["operator_blocked"] is False
    blocked = session_state_as_dict(
        fold_session(
            [
                {"type": "session.start", "user_task": "t"},
                {"type": "approval.prompt", "id": "a1", "prompt": "run cmd?"},
            ]
        )
    )
    assert blocked["operator_blocked"] is True


def test_run_status_label_distinguishes_stop_finish_error() -> None:
    # All of these set finished=True; the reason is what a user needs to tell them
    # apart. A stopped run must never read as a bare "finished" (looks completed).
    def end(reason: str, all_passed: bool) -> SessionState:
        s = apply_event(initial_state(), {"type": "session.start", "user_task": "t"})
        return apply_event(s, {"type": "session.end", "reason": reason, "all_passed": all_passed})

    assert session_status_label(initial_state()) == "running"
    assert session_status_label(end("steer_abort", False)) == "stopped"
    assert session_status_label(end("finish_session", True)) == "passed"
    assert session_status_label(end("finish_session", False)) == "finished"
    assert session_status_label(end("provider_error", False)) == "failed · provider error"
    # and the computed label rides along on the wire dict for the web client
    assert session_state_as_dict(end("steer_abort", False))["status_label"] == "stopped"
    # the raw status WORD rides along too, so a client can branch on it (the web
    # heartbeat goes quiet on "waiting" instead of painting "working" over a run
    # blocked on the operator).
    assert session_state_as_dict(initial_state())["status"] == "running"
    assert session_state_as_dict(end("steer_abort", False))["status"] == "stopped"
    assert session_state_as_dict(end("finish_session", True))["status"] == "passed"


def test_resume_start_unfinishes_the_run() -> None:
    # A resume restarts a finished/stopped run in place; the header must show it
    # running again (else steer/stop stay disabled on a live run).
    s = apply_event(initial_state(), {"type": "session.end", "reason": "steer_abort"})
    assert s.finished and s.end_reason == "steer_abort"
    s = apply_event(s, {"type": "loop.resume.start"})
    assert not s.finished and s.end_reason == ""
    assert session_status_label(s) == "running"


def test_role_result_tracks_context_tokens_and_provider() -> None:
    """role.call carries the provider; role.result folds the call's full prompt
    (fresh + cache read + cache write) into ctx_tokens -- the context size the
    ctx% readout is computed from. The value survives the next role.call (no
    per-turn blink) and an error result without usage keeps the last known."""
    from agent6.viewmodel.state import apply_event, initial_state

    s = initial_state()
    s = apply_event(s, {"type": "role.call", "role": "worker", "model": "m", "provider": "p"})
    assert s.last_role is not None and s.last_role.provider == "p"
    assert s.last_role.ctx_tokens == 0  # nothing measured yet
    s = apply_event(
        s,
        {
            "type": "role.result",
            "role": "worker",
            "ok": True,
            "tokens_in": 1_000,
            "cache_read": 40_000,
            "cache_creation": 2_000,
        },
    )
    assert s.last_role is not None and s.last_role.ctx_tokens == 43_000
    s = apply_event(s, {"type": "role.call", "role": "worker", "model": "m", "provider": "p"})
    assert s.last_role is not None and s.last_role.ctx_tokens == 43_000  # carried over
    s = apply_event(s, {"type": "role.result", "role": "worker", "ok": False, "error": "boom"})
    assert s.last_role is not None and s.last_role.ctx_tokens == 43_000  # kept on error


def test_run_start_after_run_end_unfinishes_without_banking() -> None:
    """The ask REPL re-enters wf.run() per follow-up, emitting a fresh session.start
    on the same log with no resume marker. A session.start begins a leg: it must
    clear the terminal state (or the streaming follow-up renders "answered").
    It must NOT bank usd like ResumeStart: the REPL reuses one BudgetTracker,
    so usd_total is already cumulative and banking would double-count."""
    s = initial_state()
    s = apply_event(s, {"type": "session.start", "user_task": "q"})
    s = apply_event(s, {"type": "budget.update", "usd_total": 0.02})
    s = apply_event(s, {"type": "session.end", "all_passed": False, "reason": "answered"})
    s = apply_event(s, {"type": "session.start", "user_task": "follow-up"})
    s = apply_event(s, {"type": "role.call", "role": "worker", "model": "m"})
    assert s.finished is False
    assert s.end_reason == ""
    assert session_status_label(s) == "running"
    s = apply_event(s, {"type": "budget.update", "usd_total": 0.03})
    assert s.budget.usd_total == pytest.approx(0.03)
    assert s.budget.usd_prior_legs == pytest.approx(0.0)


def test_resume_resets_the_leg_token_counters() -> None:
    """ResumeStart banks usd but must also drop the dead leg's token counters
    and caps: BudgetView documents them as the CURRENT leg's, and scan_session_log
    already resets -- until the resumed leg's first budget.update the header
    would otherwise render the finished leg's ~100%%."""
    s = initial_state()
    s = apply_event(s, {"type": "session.start", "user_task": "t"})
    s = apply_event(
        s,
        {
            "type": "budget.update",
            "input_total": 9500,
            "output_total": 480,
            "usd_total": 0.2,
            "usd_cap": 10.0,
            "tokens_unmetered": 700,
            "tokens_fallback_cap": 1000,
        },
    )
    s = apply_event(s, {"type": "session.end", "all_passed": False, "reason": "budget_exhausted"})
    s = apply_event(s, {"type": "loop.resume.start"})
    assert s.budget.input_total == 0
    assert s.budget.output_total == 0
    assert s.budget.usd_cap == 0.0
    assert s.budget.tokens_unmetered == 0
    assert s.budget.tokens_fallback_cap == 0
    assert s.budget.usd_total == pytest.approx(0.2)
    assert s.budget.usd_prior_legs == pytest.approx(0.2)


def test_concurrent_same_name_results_pair_by_call_id() -> None:
    """Two review seats call read_file concurrently through the shared
    dispatcher; last-entry name pairing cross-stamped the summaries and left
    one row in-flight forever. Results pair on the stamped call_id."""
    s = initial_state()
    s = apply_event(
        s, {"type": "tool.call", "name": "read_file", "args": {"path": "a.py"}, "call_id": 1}
    )
    s = apply_event(
        s, {"type": "tool.call", "name": "read_file", "args": {"path": "b.py"}, "call_id": 2}
    )
    s = apply_event(
        s, {"type": "tool.result", "name": "read_file", "ok": True, "summary": "sa", "call_id": 1}
    )
    s = apply_event(
        s, {"type": "tool.result", "name": "read_file", "ok": True, "summary": "sb", "call_id": 2}
    )
    a, b = s.tool_calls
    assert (a.result_summary, a.ok) == ("sa", True)
    assert (b.result_summary, b.ok) == ("sb", True)


def test_interleaved_result_for_an_earlier_call_pairs_by_call_id() -> None:
    """call A, call B, result A: name-vs-last matching dropped A's result and
    left A in-flight forever; with call_id it lands on A while B stays open."""
    s = initial_state()
    s = apply_event(
        s, {"type": "tool.call", "name": "read_file", "args": {"path": "a.py"}, "call_id": 1}
    )
    s = apply_event(
        s, {"type": "tool.call", "name": "grep", "args": {"pattern": "x"}, "call_id": 2}
    )
    s = apply_event(
        s, {"type": "tool.result", "name": "read_file", "ok": True, "summary": "sa", "call_id": 1}
    )
    a, b = s.tool_calls
    assert (a.result_summary, a.ok) == ("sa", True)
    assert b.ok is None  # still in flight


def test_compaction_events_fold_into_elision_counters() -> None:
    """/status truth source: counts of elided markers / live gists in the
    CURRENT context (a demoted gist is no longer held as a gist)."""
    s = initial_state()
    s = apply_event(
        s, {"type": "loop.compact.dropped", "n": 3, "calls": ["read_file a.py", "grep 'q'"]}
    )
    s = apply_event(
        s,
        {
            "type": "loop.compact.gists",
            "gisted": 2,
            "demoted": 1,
            "paths": ["a.py", "b.py"],
            "demoted_paths": ["c.py"],
        },
    )
    s = apply_event(s, {"type": "loop.compact.dropped", "n": 1, "calls": ["read_file d.py"]})
    assert s.compact_elided == 4
    assert s.compact_gists_live == 1


def test_format_log_line_compaction_identities() -> None:
    dropped = format_log_line(
        {"type": "loop.compact.dropped", "n": 2, "calls": ["read_file a.py", "grep 'q'"]}
    )
    assert "elided 2" in dropped and "read_file a.py" in dropped and "grep 'q'" in dropped
    gists = format_log_line(
        {
            "type": "loop.compact.gists",
            "gisted": 1,
            "demoted": 1,
            "paths": ["a.py"],
            "demoted_paths": ["b.py"],
        }
    )
    assert "1 distilled (a.py)" in gists and "1 demoted (b.py)" in gists
    done = format_log_line(
        {"type": "loop.compact.summarise.done", "summary_chars": 2341, "summary": "did things"}
    )
    assert "2341-char progress summary" in done


def test_pin_added_events_accumulate() -> None:
    s = initial_state()
    s = apply_event(
        s, {"type": "loop.pin.added", "text": "never touch schema", "chars": 18, "count": 1}
    )
    s = apply_event(s, {"type": "loop.pin.added", "text": "ship X first", "chars": 12, "count": 2})
    assert s.pins == ("never touch schema", "ship X first")


def test_tier2_restart_resets_elision_counters() -> None:
    """A summarise-and-restart wipes every elision marker and gist from the
    model's context; the /status counters must reset with it or the surface
    claims markers the model no longer holds."""
    s = initial_state()
    s = apply_event(s, {"type": "loop.compact.dropped", "n": 9, "calls": ["read_file a.py"]})
    s = apply_event(
        s,
        {
            "type": "loop.compact.gists",
            "gisted": 3,
            "demoted": 0,
            "paths": ["a.py", "b.py", "c.py"],
            "demoted_paths": [],
        },
    )
    s = apply_event(
        s, {"type": "loop.compact.summarise.done", "summary_chars": 900, "summary": "s"}
    )
    assert s.compact_elided == 0
    assert s.compact_gists_live == 0


def test_pins_restored_event_replaces_not_appends() -> None:
    """loop.pin.restored carries the full restored list: a plain resume (whose
    log already holds the pin.added events) must not double-count, and a fork
    (fresh log, snapshot-only pins) must show them at all."""
    s = initial_state()
    s = apply_event(s, {"type": "loop.pin.added", "text": "keep A", "chars": 6, "count": 1})
    s = apply_event(s, {"type": "loop.pin.restored", "pins": ["keep A"], "count": 1})
    assert s.pins == ("keep A",)  # replace, not append
    fork = apply_event(
        initial_state(), {"type": "loop.pin.restored", "pins": ["keep A", "ship X"], "count": 2}
    )
    assert fork.pins == ("keep A", "ship X")


# One wrong-shaped field per known family, each syntactically valid JSON: the
# containers the fold reads without a type check. None is producible by
# agent6's own writers; a corrupted, hand-edited, or foreign-version log is.
_WRONG_SHAPE_EVENTS: list[dict[str, Any]] = [
    {"type": "graph.update", "nodes": [1]},
    {"type": "graph.update", "nodes": 7},
    {"type": "tool.call", "name": "x", "args": [1]},
    {"type": "tool.call", "name": "x", "args": "not-a-dict"},
    {"type": "question.prompt", "questions": {"question": "q"}},
    {"type": "question.answer", "id": "q1", "answers": 5},
    {"type": "loop.compact.dropped", "n": 1, "calls": 5},
    {"type": "loop.compact.gists", "gisted": 1, "paths": True, "demoted": 1, "demoted_paths": 1},
]


@pytest.mark.parametrize(
    "bad", _WRONG_SHAPE_EVENTS, ids=lambda e: f"{e['type']}-{type(sorted(e)[0]).__name__}"
)
def test_fold_is_total_for_wrong_shaped_containers(bad: dict[str, Any]) -> None:
    """Any syntactically valid JSON object must fold, not raise: the fold runs
    unwrapped inside live tails (attach, TUI, web SSE), so one wrong-shaped
    field in a corrupted or foreign log crashed every viewer and turned the
    web run endpoint into a 500."""
    state = fold_session(
        [
            {"type": "session.start", "user_task": "t"},
            bad,
            {"type": "session.end", "reason": "finish_session", "all_passed": True},
        ]
    )
    assert state.finished  # the fold survived the bad line and kept folding


def test_a_question_asked_while_no_model_runs_is_the_harness_s() -> None:
    """agent6 asks its own start questions (the dirty-tree gate) before
    session.start, and again before a resumed leg starts (after the last
    session.end); a question while the model runs is the model's. The TUI
    modal and the web prompt box name the asker from this flag."""
    q = {"type": "question.prompt", "id": "question-1", "questions": [{"question": "stash?"}]}
    before = fold_session([q])
    assert before.pending_questions[0].from_harness is True
    during = fold_session([{"type": "session.start", "user_task": "t"}, q])
    assert during.pending_questions[0].from_harness is False
    after = fold_session(
        [
            {"type": "session.start", "user_task": "t"},
            {"type": "session.end", "reason": "finish_session", "all_passed": True},
            {**q, "id": "question-2"},
        ]
    )
    assert after.pending_questions[-1].from_harness is True


def test_run_state_as_dict_carries_what_the_run_serves(tmp_path: Path) -> None:
    """`ports` is the wire form of `listening_ports`: the ports the run's
    network listens on ([] with no network), so the web/TUI headers can name
    a dev server the agent started, as `sessions show` and `forward` do."""
    import os
    import socket

    from agent6.sessions.ipc import write_session_netns_pid

    d = tmp_path / "r"
    d.mkdir()
    assert session_state_as_dict(fold_session([]), d)["ports"] == []
    with socket.socket() as srv:
        srv.bind(("127.0.0.1", 0))
        srv.listen(1)
        port = srv.getsockname()[1]
        write_session_netns_pid(d, os.getpid())  # this process stands in for the holder
        assert port in session_state_as_dict(fold_session([]), d)["ports"]


def test_approval_parts_is_the_one_shape_every_surface_renders() -> None:
    """A dispatch prompt is "Allow <tool>: <payload>"; the head is the question
    and the payload the command (possibly several lines). A prompt with no
    payload is all head, and the web JSON carries both parts."""
    from agent6.viewmodel import approval_parts

    assert approval_parts("Allow run_command: pytest -q tests") == (
        "Allow run_command",
        "pytest -q tests",
    )
    assert approval_parts("Allow run_command: sh -c 'a\nb'") == (
        "Allow run_command",
        "sh -c 'a\nb'",
    )
    assert approval_parts("Allow fetch?") == ("Allow fetch?", "")


def test_auto_commits_fold_into_the_step_list() -> None:
    """Every per-step commit lands in `steps` (oldest first) with its iteration
    and subject: the dashboards' step selector; a sha-less auto_commit (nothing
    to commit) adds no step."""
    from agent6.viewmodel.state import apply_event, initial_state

    s = apply_event(
        initial_state(),
        {"type": "loop.auto_commit", "iteration": 3, "sha": "a" * 40, "subject": "fix parser"},
    )
    s = apply_event(s, {"type": "loop.auto_commit", "iteration": 4, "sha": ""})
    s = apply_event(
        s, {"type": "loop.auto_commit", "iteration": 5, "sha": "b" * 40, "subject": "add test"}
    )
    assert [(st.iteration, st.sha[:1], st.subject) for st in s.steps] == [
        (3, "a", "fix parser"),
        (5, "b", "add test"),
    ]
