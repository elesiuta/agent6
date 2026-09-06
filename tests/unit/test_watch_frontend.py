# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Interactive `agent6 attach` attaches as a CLI front-end: an unanswered
run_command approval / ask_user question in the streamed log is prompted on the
terminal and the answer is written back over the file bridge. Historical and
already-answered prompts are not re-asked on the replay."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from agent6.sessions.ipc import approvals_dir, questions_dir
from agent6.ui.cli import plan_watch


def _view() -> Any:
    class _V:
        def pause(self) -> Any:
            import contextlib

            return contextlib.nullcontext()

    return _V()


def _write_log(path: Path, events: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(e) + "\n" for e in events), encoding="utf-8")


def test_open_prompt_at_attach_is_answered_and_written(tmp_path: Path, monkeypatch: Any) -> None:
    # A run already waiting at approval-1 when you attach: the front-end prompts
    # and writes the answer the worker is blocked reading.
    def _yes(_prompt: str, *, standing: bool = True) -> str:
        return "yes"

    monkeypatch.setattr(plan_watch, "default_stdin_approver", _yes)
    log = tmp_path / "logs.jsonl"
    _write_log(
        log,
        [
            {"type": "session.start"},
            {"type": "approval.prompt", "id": "approval-1", "prompt": "run `ls`?"},
        ],
    )
    fe = plan_watch._CliFrontEnd(tmp_path, _view())  # pyright: ignore[reportPrivateUsage]
    opens = fe.open_prompts_at_attach(log)
    assert [(e["type"], e["id"]) for e in opens] == [("approval.prompt", "approval-1")]
    for event in opens:
        fe.handle(event)
    assert (approvals_dir(tmp_path) / "approval-1.answer").read_text() == "yes"


def test_already_answered_prompt_is_not_reasked(tmp_path: Path, monkeypatch: Any) -> None:
    # approval-1 was emitted AND answered in history: not open at attach, and the
    # replay must not re-prompt it (the approver would fail the test if called).
    def _forbidden(_p: object, *, standing: bool = True) -> str:
        raise AssertionError("must not prompt for an already-answered approval")

    monkeypatch.setattr(plan_watch, "default_stdin_approver", _forbidden)
    log = tmp_path / "logs.jsonl"
    _write_log(
        log,
        [
            {"type": "approval.prompt", "id": "approval-1", "prompt": "x"},
            {"type": "approval.answer", "id": "approval-1", "approved": True},
        ],
    )
    fe = plan_watch._CliFrontEnd(tmp_path, _view())  # pyright: ignore[reportPrivateUsage]
    assert fe.open_prompts_at_attach(log) == []
    # replay through react(): the historical prompt is skipped (answered).
    replay: list[dict[str, Any]] = [
        {"type": "approval.prompt", "id": "approval-1", "prompt": "x"},
        {"type": "approval.answer", "id": "approval-1", "approved": True},
    ]
    for ev in replay:
        fe.react(ev)  # no exception == not re-prompted


def test_react_answers_a_new_live_question(tmp_path: Path, monkeypatch: Any) -> None:
    def _beta(_qs: object) -> tuple[str, ...]:
        return ("beta",)

    monkeypatch.setattr(plan_watch, "default_stdin_questioner", _beta)
    log = tmp_path / "logs.jsonl"
    history: list[dict[str, Any]] = [{"type": "session.start"}]
    _write_log(log, history)
    fe = plan_watch._CliFrontEnd(tmp_path, _view())  # pyright: ignore[reportPrivateUsage]
    fe.open_prompts_at_attach(log)
    for ev in history:  # the follow loop replays the scanned prefix first
        fe.react(ev)
    event: dict[str, Any] = {
        "type": "question.prompt",
        "id": "question-1",
        "questions": [{"question": "which?", "options": ["alpha", "beta"]}],
    }
    fe.react(event)
    assert json.loads((questions_dir(tmp_path) / "question-1.answer").read_text()) == ["beta"]


def test_attach_replay_does_not_reask_an_answered_prompt(tmp_path: Path, monkeypatch: Any) -> None:
    """The follow loop replays the WHOLE log through react() after the pre-scan,
    and every real log opens with session.start. Clearing the answered ids at that
    boundary threw away the pre-scan's knowledge, so attaching to any run that
    had ever answered a prompt re-asked it and blocked on stdin."""
    asked: list[str] = []

    def _yes(prompt: str, *, standing: bool = True) -> str:
        asked.append(prompt)
        return "yes"

    monkeypatch.setattr(plan_watch, "default_stdin_approver", _yes)
    log = tmp_path / "logs.jsonl"
    history: list[dict[str, Any]] = [
        {"type": "session.start", "user_task": "t"},
        {"type": "approval.prompt", "id": "approval-1", "prompt": "ANSWERED LONG AGO"},
        {"type": "approval.answer", "id": "approval-1", "approved": True},
        {"type": "role.call", "role": "worker", "model": "m"},
    ]
    _write_log(log, history)

    fe = plan_watch._CliFrontEnd(tmp_path, _view())  # pyright: ignore[reportPrivateUsage]
    assert fe.open_prompts_at_attach(log) == []  # nothing open
    for ev in history:  # _watch_transcript replays from the start
        fe.react(ev)
    assert asked == []  # the answered prompt is history, not a question


def test_resumed_leg_reuses_prompt_ids_and_is_still_answered(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """Prompt ids are per-leg counters, so a resumed leg re-emits approval-1 /
    question-1. The answered-set must clear at the session boundary: it did not,
    so an attached CLI silently dropped the new leg's first prompt and the run
    hung forever on a front-end that would never answer (the TUI already resets
    its seen-set on SESSION_START_EVENTS)."""
    asked: list[str] = []

    def _yes(prompt: str, *, standing: bool = True) -> str:
        asked.append(prompt)
        return "yes"

    def _answer(_qs: Any) -> tuple[str, ...]:
        asked.append("question")
        return ("a",)

    monkeypatch.setattr(plan_watch, "default_stdin_approver", _yes)
    monkeypatch.setattr(plan_watch, "default_stdin_questioner", _answer)
    log = tmp_path / "logs.jsonl"
    leg1: list[dict[str, Any]] = [
        {"type": "session.start"},
        {"type": "approval.prompt", "id": "approval-1", "prompt": "leg 1 ok?"},
        {"type": "approval.answer", "id": "approval-1", "approved": True},
        {"type": "question.prompt", "id": "question-1", "questions": [{"question": "q?"}]},
        {"type": "question.answer", "id": "question-1", "answers": ["x"]},
        {"type": "session.end", "reason": "steer_abort"},
    ]
    _write_log(log, leg1)
    fe = plan_watch._CliFrontEnd(tmp_path, _view())  # pyright: ignore[reportPrivateUsage]
    assert fe.open_prompts_at_attach(log) == []  # leg 1 is fully answered
    for ev in leg1:  # the follow loop replays the whole log first
        fe.react(ev)
    assert asked == []  # nothing historical is re-asked

    # The resumed leg restarts the id counters and prompts again, live.
    leg2: tuple[dict[str, object], ...] = (
        {"type": "loop.resume.start", "iteration": 2},
        {"type": "approval.prompt", "id": "approval-1", "prompt": "leg 2 ok?"},
        {"type": "question.prompt", "id": "question-1", "questions": [{"question": "q2?"}]},
    )
    for ev in leg2:
        fe.react(ev)
    assert asked == ["leg 2 ok?", "question"]  # both prompted, neither swallowed
    assert (approvals_dir(tmp_path) / "approval-1.answer").read_text(encoding="utf-8") == "yes"
    assert (questions_dir(tmp_path) / "question-1.answer").exists()


def test_an_unscoped_approval_offers_no_session_choice(tmp_path: Path, monkeypatch: Any) -> None:
    """A gate with no scope to grant (`fetch`) journals `standing: false`, and
    the foreground prompt then offers no "allow all" answer, since
    `record_answer` drops a session grant it cannot scope. The attached
    front-end asked with the default and offered one anyway."""
    seen: list[bool] = []

    def _approver(_prompt: str, *, standing: bool = True) -> str:
        seen.append(standing)
        return "no"

    monkeypatch.setattr(plan_watch, "default_stdin_approver", _approver)
    log = tmp_path / "logs.jsonl"
    history: list[dict[str, Any]] = [
        {"type": "session.start"},
        {
            "type": "approval.prompt",
            "id": "approval-1",
            "prompt": "Allow fetch: example.invalid /data",
            "standing": False,
        },
    ]
    _write_log(log, history)
    live = plan_watch._CliFrontEnd(tmp_path, _view())  # pyright: ignore[reportPrivateUsage]
    live.react(history[1])
    attached = plan_watch._CliFrontEnd(tmp_path, _view())  # pyright: ignore[reportPrivateUsage]
    for event in attached.open_prompts_at_attach(log):
        attached.handle(event)
    assert seen == [False, False]
