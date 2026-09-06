# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Regression tests for three loop/resume bugs:

#3  end-of-iteration resume snapshot (don't replay already-executed tools)
#12 completion-relevant scalars survive a resume (metric / verify-settled)
#10 final checkpoint commits a dirty worktree on a gated-run success exit
"""

from __future__ import annotations

import json
import subprocess as sp
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest import mock
from unittest.mock import MagicMock

from agent6.tools.results import ExecResult, RawResult
from agent6.workflows._conversation import Conversation
from agent6.workflows._metric import MetricSample as _MetricSample
from agent6.workflows._session_state import SessionSnapshot, load_session_snapshot
from agent6.workflows.loop import (
    LoopState,
    Workflow,
)

# The `[git]` surface the loop reads: the checkpoint message and the commit
# identity (`_commit_identity`), empty as a real Config carries it unset.
_GIT_STUB = SimpleNamespace(
    control="agent6",
    commit=SimpleNamespace(
        checkpoint=SimpleNamespace(message="agent6"), name="", email="", trailer=""
    ),
)


def _silent(_: str) -> None:
    return None


def _wf(**kw: Any) -> Workflow:
    defaults: dict[str, Any] = {
        "root": Path("/tmp"),
        "config": MagicMock(
            git=_GIT_STUB,
            budget=SimpleNamespace(max_usd=10.0, max_tokens_fallback=2_000_000),
            prompt=MagicMock(system_prompt_file=""),
            workflow=MagicMock(verify_command=(), verify_when="never", verify_retries=2),
        ),
        "provider": MagicMock(),
        "dispatcher": MagicMock(),
        "logger": _silent,
    }
    defaults.update(kw)
    if "chain_fallback_parent" not in kw and "root" in kw:
        # Mirror run.py's wiring: the chain's first parent is HEAD at start.
        head = sp.run(
            ["git", "-C", str(kw["root"]), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        ).stdout.strip()
        defaults.setdefault("chain_ref", "refs/agent6/test")
        defaults.setdefault("chain_fallback_parent", head or None)
    return Workflow(**defaults)


def _git_repo(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    sp.run(["git", "init", "-q", "-b", "main"], cwd=path, check=True)
    sp.run(["git", "config", "user.email", "t@example.com"], cwd=path, check=True)
    sp.run(["git", "config", "user.name", "t"], cwd=path, check=True)
    (path / "seed.txt").write_text("seed\n")
    sp.run(["git", "add", "seed.txt"], cwd=path, check=True)
    sp.run(["git", "commit", "-q", "-m", "init"], cwd=path, check=True)


# --- #12: completion-relevant scalars round-trip + restore -----------------


def test_snapshot_persists_completion_scalars(tmp_path: Path) -> None:
    """verify_ever_passed / gateless_ever_committed / metric summary are written
    and load back, instead of resetting to their fresh-run defaults."""
    snap = tmp_path / "loop_state.json"
    config = SimpleNamespace(
        git=_GIT_STUB,
        budget=SimpleNamespace(max_usd=10.0, max_tokens_fallback=2_000_000),
        workflow=SimpleNamespace(
            verify_when="never",
            verify_retries=2,
            verify_command=(),
            verify_infer=True,
            metric=SimpleNamespace(goal="maximize"),
        ),
    )
    wf = _wf(resume_state_path=snap, config=config)
    state = LoopState(original_task="t", tool_calls=2)
    state.verify.ever_passed = True
    state.verify.scoped = True
    state.gateless_ever_committed = True
    state.metric_history.append(_MetricSample(label="x", score=27.0, returncode=0, at_ceiling=True))
    wf._save_resume_snapshot(  # pyright: ignore[reportPrivateUsage]
        system="s", messages=[], tool_calls=2, next_iteration=4, root_task_id=None, state=state
    )
    loaded = load_session_snapshot(snap)
    assert loaded.verify_ever_passed is True
    assert loaded.verify_scoped is True
    assert loaded.gateless_ever_committed is True
    assert loaded.metric_best_score == 27.0
    assert loaded.metric_at_ceiling is True


def test_completed_prose_turn_is_snapshotted_before_the_boundary(tmp_path: Path) -> None:
    """A prose turn plus its nudge is a completed iteration; an operator stop
    at its boundary must resume from AFTER it. The post-turn snapshot only ran
    on tool turns, so a stop after a prose answer left loop_state.json at the
    PRE-call snapshot: resume re-paid the provider call and the nudge never
    existed in the resumed history."""
    from agent6.providers import ProviderResponse
    from agent6.workflows._session_state import SessionResult

    repo = tmp_path / "repo"
    _git_repo(repo)
    snap_path = tmp_path / "loop_state.json"
    provider = MagicMock()
    provider.call.return_value = ProviderResponse(
        text="answer in prose",
        tool_uses=(),
        stop_reason="end_turn",
        input_tokens=1,
        output_tokens=1,
        cache_read_tokens=0,
        cache_creation_tokens=0,
        raw={"content": [{"type": "text", "text": "answer in prose"}]},
    )
    wf = _wf(
        root=repo,
        provider=provider,
        resume_state_path=snap_path,
        provider_retry_count=0,
        provider_retry_delay_s=0.0,
        max_iterations=5,
    )
    stopped = SessionResult(
        completed=False, reason="interactive_stop", summary="", iterations=1, tool_calls=0
    )

    def stop_at_boundary(*_a: object, **_k: object) -> SessionResult:
        return stopped

    with mock.patch.object(Workflow, "_operator_boundary", stop_at_boundary):
        result = wf.run("do the task")
    assert result.reason == "interactive_stop"
    loaded = load_session_snapshot(snap_path)
    assert loaded.next_iteration == 2  # the prose turn is a COMPLETED iteration
    dumped = json.dumps(loaded.messages)
    assert "answer in prose" in dumped  # the model's turn survives the stop
    assert "[harness]" in dumped  # and so does the nudge that answered it


def test_snapshot_persists_and_restores_parallel_group_counter(tmp_path: Path) -> None:
    """The /parallel group counter is run-lifetime state: lane ids embed it
    (`<run>-p<N>-l<i>`) and so do the imported branches. In-memory only, every
    resume restarted at p1, so the first post-resume dispatch rebuilt the exact
    ids of a prior group -- clone dirs collided or, cache clean, the lanes ran
    to completion and then failed import on the already-existing branch,
    stranding paid work. Persist it like the sibling completion scalars."""
    from agent6.workflows.loop import (
        restore_completion_state,
    )

    snap = tmp_path / "loop_state.json"
    config = SimpleNamespace(
        git=_GIT_STUB,
        budget=SimpleNamespace(max_usd=10.0, max_tokens_fallback=2_000_000),
        workflow=SimpleNamespace(
            verify_when="never",
            verify_retries=2,
            verify_command=(),
            verify_infer=True,
            metric=SimpleNamespace(goal="maximize"),
        ),
    )
    wf = _wf(resume_state_path=snap, config=config)
    state = LoopState(original_task="t", tool_calls=0)
    state.parallel_groups_dispatched = 2
    wf._save_resume_snapshot(  # pyright: ignore[reportPrivateUsage]
        system="s", messages=[], tool_calls=0, next_iteration=4, root_task_id=None, state=state
    )
    loaded = load_session_snapshot(snap)
    assert loaded.parallel_groups_dispatched == 2

    fresh = LoopState(original_task="t", tool_calls=0)
    restore_completion_state(fresh, loaded)
    assert fresh.parallel_groups_dispatched == 2  # the next dispatch is p3, not p1


def test_snapshot_persists_and_restores_pins(tmp_path: Path) -> None:
    """Operator /pin instructions are run-lifetime state: every tier-2 restart
    re-injects them verbatim, so a resume must carry them like the sibling
    completion scalars. A version-2 snapshot written BEFORE pins existed loads
    with none (additive defaulted field, same as the /parallel counter)."""
    from agent6.workflows.loop import (
        restore_completion_state,
    )

    snap = tmp_path / "loop_state.json"
    config = SimpleNamespace(
        git=_GIT_STUB,
        budget=SimpleNamespace(max_usd=10.0, max_tokens_fallback=2_000_000),
        workflow=SimpleNamespace(
            verify_when="never",
            verify_retries=2,
            verify_command=(),
            verify_infer=True,
            metric=SimpleNamespace(goal="maximize"),
        ),
    )
    wf = _wf(resume_state_path=snap, config=config)
    state = LoopState(original_task="t", tool_calls=0)
    state.pins.extend(["never touch schema files", "goal:\nship X"])
    wf._save_resume_snapshot(  # pyright: ignore[reportPrivateUsage]
        system="s", messages=[], tool_calls=0, next_iteration=4, root_task_id=None, state=state
    )
    loaded = load_session_snapshot(snap)
    assert loaded.pins == ("never touch schema files", "goal:\nship X")

    fresh = LoopState(original_task="t", tool_calls=0)
    restore_completion_state(fresh, loaded)
    assert fresh.pins == ["never touch schema files", "goal:\nship X"]

    # Pre-pins snapshot (no `pins` key) still loads: additive default.
    raw = json.loads(snap.read_text(encoding="utf-8"))
    del raw["pins"]
    snap.write_text(json.dumps(raw), encoding="utf-8")
    assert load_session_snapshot(snap).pins == ()


def test_pre_version_bump_snapshot_refused_loudly(tmp_path: Path) -> None:
    """An in-flight run written before a state-format change (an older
    SNAPSHOT_VERSION) must refuse to resume/fork with a clear reason -- never a
    garbage parse or a silent default. Deliberately fabricates the OLD shape."""
    import pytest

    snap = tmp_path / "loop_state.json"
    snap.write_text(
        json.dumps(
            {
                "version": 1,
                "system": "s",
                "messages": [],
                "tool_calls": 0,
                "next_iteration": 1,
                "root_task_id": None,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="predates a state-format change"):
        load_session_snapshot(snap)


def test_malformed_snapshot_shapes_fail_loud(tmp_path: Path) -> None:
    """A valid-JSON but wrong-shape snapshot (null / list / scalar / missing key
    / non-list messages) raises a clean ValueError, not an AttributeError or a
    deferred mid-loop crash the resume callers don't catch."""
    import pytest

    snap = tmp_path / "loop_state.json"
    for bad in ("null", "[]", "123", '"str"'):
        snap.write_text(bad, encoding="utf-8")
        with pytest.raises(ValueError, match="expected a JSON object"):
            load_session_snapshot(snap)
    # current version, wrong internals: missing required keys, and a non-list messages
    snap.write_text(json.dumps({"version": 2, "system": "s"}), encoding="utf-8")
    with pytest.raises(ValueError, match="malformed run-state snapshot"):
        load_session_snapshot(snap)
    snap.write_text(
        json.dumps(
            {"version": 2, "system": "s", "messages": "oops", "tool_calls": 0, "next_iteration": 1}
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="messages"):
        load_session_snapshot(snap)


def test_resume_seeds_state_from_snapshot_scalars() -> None:
    """_drive_loop restores verify_ever_passed and a synthetic at-ceiling metric
    sample so the metric/verify-settled stop logic doesn't regress on resume.

    Drives a single iteration that immediately finishes; the assertion is that
    the loop saw the restored at-ceiling history (no early-finish rejection).
    """
    config = SimpleNamespace(
        git=_GIT_STUB,
        budget=SimpleNamespace(max_usd=10.0, max_tokens_fallback=2_000_000),
        workflow=SimpleNamespace(
            verify_when="never",
            verify_retries=2,
            verify_command=(),
            verify_infer=True,
            metric=SimpleNamespace(goal="maximize"),
        ),
    )
    provider = MagicMock()
    provider.call.return_value = SimpleNamespace(
        text="",
        tool_uses=({"id": "t1", "name": "finish_session", "input": {"summary": "done"}},),
        stop_reason="tool_use",
        input_tokens=1,
        output_tokens=1,
        raw={
            "content": [
                {
                    "type": "tool_use",
                    "id": "t1",
                    "name": "finish_session",
                    "input": {"summary": "done"},
                }
            ]
        },
    )
    dispatcher = MagicMock()
    dispatcher.dispatch.return_value = RawResult({"ok": True})
    wf = _wf(provider=provider, dispatcher=dispatcher, config=config, mode="run")

    captured: dict[str, Any] = {}
    orig = wf._metric_at_ceiling  # pyright: ignore[reportPrivateUsage]

    def _spy(history: list[Any]) -> bool:
        captured["at_ceiling"] = orig(history)
        return captured["at_ceiling"]

    wf._metric_at_ceiling = _spy  # type: ignore[method-assign]
    result = wf._drive_loop(  # pyright: ignore[reportPrivateUsage]
        system="s",
        conversation=Conversation.from_wire(
            [{"role": "user", "content": [{"type": "text", "text": "go"}]}]
        ),
        tool_calls=0,
        start_iteration=3,
        root_task_id=None,
        original_task="go",
        resume_from=SessionSnapshot(
            system="s",
            messages=[],
            tool_calls=0,
            next_iteration=3,
            root_task_id=None,
            original_task="go",
            verify_command=(),
            metric_best_score=27.0,
            metric_at_ceiling=True,
        ),
    )
    assert result.completed is True
    assert result.reason == "finish_session"
    # The early-finish guard consulted the restored at-ceiling history.
    assert captured.get("at_ceiling") is True


class _EventCapture:
    def __init__(self, path: Path = Path("logs.jsonl")) -> None:
        self.events: list[dict[str, Any]] = []
        self.path = path  # EventSink.path: the log file the emits land in

    def emit(self, event_type: str, /, **fields: Any) -> None:
        self.events.append({"type": event_type, **fields})


def test_resume_reannounces_restored_pins_for_the_read_model() -> None:
    """A resumed leg emits loop.pin.restored with the snapshot's pins: a fork's
    fresh logs.jsonl has no pin.added events, so without this the surfaces show
    zero pins while the engine still re-injects them at every restart."""
    config = SimpleNamespace(
        git=_GIT_STUB,
        budget=SimpleNamespace(max_usd=10.0, max_tokens_fallback=2_000_000),
        workflow=SimpleNamespace(
            verify_when="never",
            verify_retries=2,
            verify_command=(),
            verify_infer=True,
            metric=SimpleNamespace(goal="maximize"),
        ),
    )
    provider = MagicMock()
    provider.call.return_value = SimpleNamespace(
        text="",
        tool_uses=({"id": "t1", "name": "finish_session", "input": {"summary": "done"}},),
        stop_reason="tool_use",
        input_tokens=1,
        output_tokens=1,
        raw={
            "content": [
                {
                    "type": "tool_use",
                    "id": "t1",
                    "name": "finish_session",
                    "input": {"summary": "done"},
                }
            ]
        },
    )
    dispatcher = MagicMock()
    dispatcher.dispatch.return_value = RawResult({"ok": True})
    ev = _EventCapture()
    wf = _wf(provider=provider, dispatcher=dispatcher, config=config, mode="run", events=ev)
    wf._drive_loop(  # pyright: ignore[reportPrivateUsage]
        system="s",
        conversation=Conversation.from_wire(
            [{"role": "user", "content": [{"type": "text", "text": "go"}]}]
        ),
        tool_calls=0,
        start_iteration=3,
        root_task_id=None,
        original_task="go",
        resume_from=SessionSnapshot(
            system="s",
            messages=[],
            tool_calls=0,
            next_iteration=3,
            root_task_id=None,
            original_task="go",
            verify_command=(),
            pins=("keep A", "ship X"),
        ),
    )
    restored = [e for e in ev.events if e["type"] == "loop.pin.restored"]
    assert restored and restored[0]["pins"] == ["keep A", "ship X"]
    assert restored[0]["count"] == 2


def test_resume_start_carries_the_leg_identity(tmp_path: Path) -> None:
    """loop.resume.start opens a resumed/forked leg's log; it stamps session_id and
    mode like session.start so the leg's log identifies itself (the manifest owns
    the task). An identity-less leg log left every fold empty and each consumer
    patching its own copy."""
    from agent6.workflows._session_state import SessionSnapshot as _Snap

    session_dir = tmp_path / "sessions" / "runs" / "tidy-otter-AB12CD"
    session_dir.mkdir(parents=True)
    snap_path = session_dir / "loop_state.json"
    snap_path.write_text(
        _Snap(
            system="s",
            messages=[{"role": "user", "content": [{"type": "text", "text": "go"}]}],
            tool_calls=0,
            next_iteration=3,
            root_task_id=None,
            original_task="go",
            verify_command=(),
        ).model_dump_json(),
        encoding="utf-8",
    )
    config = SimpleNamespace(
        git=_GIT_STUB,
        budget=SimpleNamespace(max_usd=10.0, max_tokens_fallback=2_000_000),
        workflow=SimpleNamespace(
            verify_when="never",
            verify_retries=2,
            verify_command=(),
            verify_infer=True,
            metric=SimpleNamespace(goal="maximize"),
        ),
    )
    provider = MagicMock()
    provider.call.return_value = SimpleNamespace(
        text="",
        tool_uses=({"id": "t1", "name": "finish_session", "input": {"summary": "done"}},),
        stop_reason="tool_use",
        input_tokens=1,
        output_tokens=1,
        raw={
            "content": [
                {
                    "type": "tool_use",
                    "id": "t1",
                    "name": "finish_session",
                    "input": {"summary": "done"},
                }
            ]
        },
    )
    dispatcher = MagicMock()
    dispatcher.dispatch.return_value = RawResult({"ok": True})
    ev = _EventCapture(path=session_dir / "logs.jsonl")
    wf = _wf(
        provider=provider,
        dispatcher=dispatcher,
        config=config,
        mode="run",
        events=ev,
        resume_state_path=snap_path,
    )
    wf.resume()
    (start,) = [e for e in ev.events if e["type"] == "loop.resume.start"]
    assert start["session_id"] == "tidy-otter-AB12CD"
    assert start["mode"] == "run"


def test_resume_with_no_pins_still_corrects_a_stale_pin_added() -> None:
    """A pin added and then lost to a crash (loop.pin.added reached logs.jsonl,
    the snapshot that would carry it never did) leaves the fold holding a pin the
    engine does not have: the resumed leg appends to the SAME log, so /status and
    /pin keep listing it while no restart will ever re-inject it. The corrective
    event (which the fold REPLACES on) must fire even when the snapshot is
    empty -- guarding it on a non-empty list is what let the stale one stand."""
    config = SimpleNamespace(
        git=_GIT_STUB,
        budget=SimpleNamespace(max_usd=10.0, max_tokens_fallback=2_000_000),
        workflow=SimpleNamespace(
            verify_when="never",
            verify_retries=2,
            verify_command=(),
            verify_infer=True,
            metric=SimpleNamespace(goal="maximize"),
        ),
    )
    provider = MagicMock()
    provider.call.return_value = SimpleNamespace(
        text="",
        tool_uses=({"id": "t1", "name": "finish_session", "input": {"summary": "done"}},),
        stop_reason="tool_use",
        input_tokens=1,
        output_tokens=1,
        raw={
            "content": [
                {
                    "type": "tool_use",
                    "id": "t1",
                    "name": "finish_session",
                    "input": {"summary": "d"},
                }
            ]
        },
    )
    dispatcher = MagicMock()
    dispatcher.dispatch.return_value = RawResult({"ok": True})
    ev = _EventCapture()
    wf = _wf(provider=provider, dispatcher=dispatcher, config=config, mode="run", events=ev)
    wf._drive_loop(  # pyright: ignore[reportPrivateUsage]
        system="s",
        conversation=Conversation.from_wire(
            [{"role": "user", "content": [{"type": "text", "text": "go"}]}]
        ),
        tool_calls=0,
        start_iteration=3,
        root_task_id=None,
        original_task="go",
        resume_from=SessionSnapshot(
            system="s",
            messages=[],
            tool_calls=0,
            next_iteration=3,
            root_task_id=None,
            original_task="go",
            verify_command=(),
            pins=(),
        ),
    )
    restored = [e for e in ev.events if e["type"] == "loop.pin.restored"]
    assert restored, "an empty restore must still be announced"
    assert restored[0]["pins"] == []
    assert restored[0]["count"] == 0


# --- #3: end-of-iteration snapshot (no replay of executed tools) -----------


def test_snapshot_written_after_tool_dispatch_advances_iteration(tmp_path: Path) -> None:
    """After a full iteration (assistant turn + tool dispatch + tool_results),
    the snapshot must advance to next_iteration and include the executed turn,
    so a crash before the next pre-call snapshot resumes AFTER the tools."""
    repo = tmp_path / "repo"
    _git_repo(repo)
    snap = repo / "loop_state.json"
    config = SimpleNamespace(
        git=_GIT_STUB,
        budget=SimpleNamespace(max_usd=10.0, max_tokens_fallback=2_000_000),
        workflow=SimpleNamespace(
            verify_when="never",
            verify_retries=2,
            verify_command=(),
            verify_infer=True,
            metric=SimpleNamespace(goal=None),
        ),
    )
    provider = MagicMock()
    # Iter 1: a run_command tool_use (non-idempotent side effect).
    # Iter 2: finish_session.
    provider.call.side_effect = [
        SimpleNamespace(
            text="",
            tool_uses=({"id": "a1", "name": "run_command", "input": {"command": "echo hi"}},),
            stop_reason="tool_use",
            input_tokens=1,
            output_tokens=1,
            raw={
                "content": [
                    {
                        "type": "tool_use",
                        "id": "a1",
                        "name": "run_command",
                        "input": {"command": "echo hi"},
                    }
                ]
            },
        ),
        SimpleNamespace(
            text="",
            tool_uses=({"id": "f1", "name": "finish_session", "input": {"summary": "done"}},),
            stop_reason="tool_use",
            input_tokens=1,
            output_tokens=1,
            raw={
                "content": [
                    {
                        "type": "tool_use",
                        "id": "f1",
                        "name": "finish_session",
                        "input": {"summary": "x"},
                    }
                ]
            },
        ),
    ]
    dispatcher = MagicMock()
    dispatcher.dispatch.return_value = ExecResult(
        returncode=0, stdout="hi", stderr="", duration_s=0.0, exec_failed=False
    )
    dispatcher.set_run_root_node_id = MagicMock()

    events: list[dict[str, Any]] = []
    wf = _wf(
        root=repo,
        provider=provider,
        dispatcher=dispatcher,
        config=config,
        mode="run",
        resume_state_path=snap,
    )
    orig_save = wf._save_resume_snapshot  # pyright: ignore[reportPrivateUsage]
    orig_call = wf._call_with_retry  # pyright: ignore[reportPrivateUsage]
    orig_compact = wf._maybe_compact  # pyright: ignore[reportPrivateUsage]

    def _spy_save(**kw: Any) -> None:
        orig_save(**kw)
        events.append(
            {
                "kind": "save",
                "next_iteration": kw["next_iteration"],
                "messages": json.loads(json.dumps(kw["messages"])),
            }
        )

    def _spy_call(*a: Any, **kw: Any) -> Any:
        events.append({"kind": "provider_call"})
        return orig_call(*a, **kw)

    def _spy_compact(msgs: Any, state: Any, **kw: Any) -> bool:
        events.append({"kind": "compact"})
        return orig_compact(msgs, state, **kw)

    wf._save_resume_snapshot = _spy_save  # type: ignore[method-assign]
    wf._call_with_retry = _spy_call  # type: ignore[method-assign]
    wf._maybe_compact = _spy_compact  # type: ignore[method-assign]
    wf._drive_loop(  # pyright: ignore[reportPrivateUsage]
        system="s",
        conversation=Conversation.from_wire(
            [{"role": "user", "content": [{"type": "text", "text": "go"}]}]
        ),
        tool_calls=0,
        start_iteration=1,
        root_task_id=None,
        original_task="go",
    )

    # The KEY guarantee: a snapshot advancing to next_iteration=2 (with the
    # executed iter-1 turn) must be written at the END of iter 1 -- i.e. AFTER
    # the first provider call but BEFORE iter 2's compaction/pre-call snapshot.
    # That closes the crash window between tool dispatch and iter-2's pre-call
    # save. On the old code the FIRST save after provider-call-1 was iter-2's
    # OWN pre-call save, which happens AFTER iter-2's compaction.
    kinds = [ev["kind"] for ev in events]
    first_call = kinds.index("provider_call")
    second_compact = next(i for i, k in enumerate(kinds) if k == "compact" and i > first_call)
    end_of_iter_saves = [
        ev
        for i, ev in enumerate(events)
        if first_call < i < second_compact and ev["kind"] == "save"
    ]
    assert end_of_iter_saves, (
        "expected an end-of-iteration snapshot between the 1st provider call"
        " and iter-2's compaction (the post-tool-dispatch crash window)"
    )
    post = [s for s in end_of_iter_saves if s["next_iteration"] == 2]
    assert post, "end-of-iteration snapshot must advance next_iteration to 2"
    msgs = post[0]["messages"]
    assert any(m.get("role") == "assistant" for m in msgs), "assistant turn must be snapshotted"
    has_tool_result = any(
        m.get("role") == "user"
        and isinstance(m.get("content"), list)
        and any(isinstance(b, dict) and b.get("type") == "tool_result" for b in m["content"])
        for m in msgs
    )
    assert has_tool_result, "executed tool_result must be in the advanced snapshot"


# --- #10: final checkpoint commits a dirty worktree on a gated run ---------


def test_final_checkpoint_commits_dirty_worktree_on_gated_run(tmp_path: Path) -> None:
    """A run_command-authored edit left uncommitted on a gated run is captured
    by _final_checkpoint so it isn't lost from git history at exit."""
    repo = tmp_path / "repo"
    _git_repo(repo)
    config = SimpleNamespace(
        git=_GIT_STUB,
        budget=SimpleNamespace(max_usd=10.0, max_tokens_fallback=2_000_000),
        workflow=SimpleNamespace(
            verify_when="never",
            verify_retries=2,
            verify_command=("pytest", "-q"),
            metric=SimpleNamespace(goal=None),
        ),
    )
    emitted: list[tuple[str, dict[str, Any]]] = []

    class _Sink:
        def emit(self, event_type: str, **fields: Any) -> None:
            emitted.append((event_type, fields))

    wf = _wf(root=repo, config=config, mode="run", events=_Sink())
    # Worker edited a file via run_command; never re-verified, never committed.
    (repo / "edit.txt").write_text("a real edit\n")
    head_before = sp.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True, check=True
    ).stdout.strip()

    wf._final_checkpoint(5)  # pyright: ignore[reportPrivateUsage]

    head_after = sp.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True, check=True
    ).stdout.strip()
    assert head_after == head_before, "the operator's HEAD never moves"
    chain = sp.run(
        ["git", "rev-parse", "refs/agent6/test"],
        cwd=repo,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    assert chain != head_before, "the chain must capture the edit on exit"
    shown = sp.run(
        ["git", "show", "refs/agent6/test:edit.txt"],
        cwd=repo,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    assert shown == "a real edit\n"
    subject = sp.run(
        ["git", "log", "-1", "--pretty=%s", "refs/agent6/test"],
        cwd=repo,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    assert "checkpoint" in subject
    # The commit must be COUNTABLE by the folds: a diff.updated is emitted, not
    # only loop.auto_commit, so web/TUI/CLI don't read the final work as 0 commits.
    kinds = [k for k, _ in emitted]
    assert "loop.auto_commit" in kinds and "diff.updated" in kinds


def test_final_checkpoint_noop_when_clean_or_not_run_mode(tmp_path: Path) -> None:
    """No commit when the tree is clean, and never in non-run mode."""
    repo = tmp_path / "repo"
    _git_repo(repo)
    config = SimpleNamespace(
        git=_GIT_STUB,
        budget=SimpleNamespace(max_usd=10.0, max_tokens_fallback=2_000_000),
        workflow=SimpleNamespace(
            verify_when="never",
            verify_retries=2,
            verify_command=("pytest",),
            metric=SimpleNamespace(goal=None),
        ),
    )
    head = sp.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True, check=True
    ).stdout.strip()

    wf_clean = _wf(root=repo, config=config, mode="run")
    wf_clean._final_checkpoint(1)  # pyright: ignore[reportPrivateUsage]
    assert (
        sp.run(
            ["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True, check=True
        ).stdout.strip()
        == head
    )

    # Dirty tree, but plan mode -> still no commit.
    (repo / "edit.txt").write_text("plan-mode edit\n")
    wf_plan = _wf(root=repo, config=config, mode="plan")
    wf_plan._final_checkpoint(1)  # pyright: ignore[reportPrivateUsage]
    assert (
        sp.run(
            ["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True, check=True
        ).stdout.strip()
        == head
    )


def test_a_forked_leg_reports_the_elisions_its_context_carries() -> None:
    """A fork copies the checkpoint but NOT logs.jsonl, so the child's log has no
    compact.dropped events to fold: /status reported "0 elided" over a restored
    context full of elision markers, contradicting the field's own "markers in
    the CURRENT context" contract. Same shape as the pin re-announce."""
    from agent6.workflows._compaction import ELISION_GIST_PREFIX, ELISION_PREFIX

    config = SimpleNamespace(
        git=_GIT_STUB,
        budget=SimpleNamespace(max_usd=10.0, max_tokens_fallback=2_000_000),
        workflow=SimpleNamespace(
            verify_when="never",
            verify_retries=2,
            verify_command=(),
            verify_infer=True,
            metric=SimpleNamespace(goal="maximize"),
        ),
    )
    provider = MagicMock()
    provider.call.return_value = SimpleNamespace(
        text="",
        tool_uses=({"id": "t1", "name": "finish_session", "input": {"summary": "done"}},),
        stop_reason="tool_use",
        input_tokens=1,
        output_tokens=1,
        raw={"content": [{"type": "tool_use", "id": "t1", "name": "finish_session", "input": {}}]},
    )
    dispatcher = MagicMock()
    dispatcher.dispatch.return_value = RawResult({"ok": True})
    ev = _EventCapture()
    wf = _wf(provider=provider, dispatcher=dispatcher, config=config, mode="run", events=ev)
    # A restored context carrying two bare elisions and one distilled gist.
    restored = Conversation.from_wire(
        [
            {"role": "user", "content": [{"type": "text", "text": "go"}]},
            {
                "role": "assistant",
                "content": [{"type": "tool_use", "id": "a", "name": "read_file", "input": {}}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "a", "content": f"{ELISION_PREFIX}: x"}
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "tool_use", "id": "b", "name": "read_file", "input": {}}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "b", "content": f"{ELISION_PREFIX}: y"}
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "tool_use", "id": "c", "name": "read_file", "input": {}}],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "c",
                        "content": f"{ELISION_GIST_PREFIX}: z",
                    }
                ],
            },
        ]
    )
    wf._drive_loop(  # pyright: ignore[reportPrivateUsage]
        system="s",
        conversation=restored,
        tool_calls=0,
        start_iteration=3,
        root_task_id=None,
        original_task="go",
        resume_from=SessionSnapshot(
            system="s",
            messages=[],
            tool_calls=0,
            next_iteration=3,
            root_task_id=None,
            original_task="go",
            verify_command=(),
        ),
    )
    restored_ev = [e for e in ev.events if e["type"] == "loop.compact.restored"]
    assert restored_ev, "the restored context's elisions were never announced"
    assert restored_ev[0]["elided"] == 3
    assert restored_ev[0]["gists"] == 1


def test_initial_pins_seed_a_fresh_run_out_of_band() -> None:
    """A /parallel lane inherits the coordinator's pins via --pin, NOT a task
    prefix (the prefix became the lane's manifest user_task, so listings and
    the judge's brief led with the pin header). Seeding emits the same
    replace-fold event a restore does, renders the same PINNED block into the
    conversation, and keeps the pins in state so a later compaction restart
    re-shows them."""
    from agent6.providers import ProviderResponse

    provider = MagicMock()
    provider.call.return_value = ProviderResponse(
        text="",
        tool_uses=({"id": "t1", "name": "finish_session", "input": {"summary": "done"}},),
        stop_reason="tool_use",
        input_tokens=1,
        output_tokens=1,
        cache_read_tokens=0,
        cache_creation_tokens=0,
        raw={
            "content": [
                {
                    "type": "tool_use",
                    "id": "t1",
                    "name": "finish_session",
                    "input": {"summary": "d"},
                }
            ]
        },
    )
    dispatcher = MagicMock()
    dispatcher.dispatch.return_value = RawResult({"ok": True})
    config = MagicMock(
        prompt=MagicMock(system_prompt_file=""),
        workflow=MagicMock(verify_command=(), verify_when="never", verify_retries=2),
    )
    ev = _EventCapture()
    wf = _wf(
        provider=provider,
        dispatcher=dispatcher,
        config=config,
        mode="run",
        events=ev,
        initial_pins=("never touch schema files",),
    )
    conversation = Conversation.from_wire(
        [{"role": "user", "content": [{"type": "text", "text": "go"}]}]
    )
    wf._drive_loop(  # pyright: ignore[reportPrivateUsage]
        system="s",
        conversation=conversation,
        tool_calls=0,
        start_iteration=1,
        original_task="go",
        root_task_id=None,
    )
    restored = [e for e in ev.events if e["type"] == "loop.pin.restored"]
    assert restored and restored[0]["pins"] == ["never touch schema files"]
    # The block the worker sees is the SAME one a restart re-shows.
    first_call = provider.call.call_args_list[0]
    messages = first_call.kwargs.get("messages") or first_call.args[1]
    flat = str(messages)
    assert "PINNED operator instructions (verbatim):" in flat
    assert "never touch schema files" in flat


def test_initial_pins_honor_the_cap_and_skip_empties() -> None:
    """--pin seeded state.pins DIRECTLY, bypassing the PINS_MAX_CHARS cap (a
    huge --pin then rode every restart and permanently wedged /pin) and the
    non-empty check (--pin '' seeded a blank pin). Seeding now goes through the
    same _try_pin owner /pin uses."""
    from agent6.providers import ProviderResponse
    from agent6.workflows.loop import PINS_MAX_CHARS  # pyright: ignore[reportPrivateUsage]

    provider = MagicMock()
    provider.call.return_value = ProviderResponse(
        text="",
        tool_uses=({"id": "t1", "name": "finish_session", "input": {"summary": "d"}},),
        stop_reason="tool_use",
        input_tokens=1,
        output_tokens=1,
        cache_read_tokens=0,
        cache_creation_tokens=0,
        raw={
            "content": [
                {
                    "type": "tool_use",
                    "id": "t1",
                    "name": "finish_session",
                    "input": {"summary": "d"},
                }
            ]
        },
    )
    dispatcher = MagicMock()
    dispatcher.dispatch.return_value = RawResult({"ok": True})
    config = MagicMock(
        prompt=MagicMock(system_prompt_file=""),
        workflow=MagicMock(verify_command=(), verify_when="never", verify_retries=2),
    )
    ev = _EventCapture()
    huge = "x" * (PINS_MAX_CHARS + 1)
    wf = _wf(
        provider=provider,
        dispatcher=dispatcher,
        config=config,
        mode="run",
        events=ev,
        initial_pins=("keep this", "", huge, "   "),  # 1 good, 1 empty, 1 over-cap, 1 blank
    )
    conversation = Conversation.from_wire(
        [{"role": "user", "content": [{"type": "text", "text": "go"}]}]
    )
    wf._drive_loop(  # pyright: ignore[reportPrivateUsage]
        system="s",
        conversation=conversation,
        tool_calls=0,
        start_iteration=1,
        original_task="go",
        root_task_id=None,
    )
    restored = [e for e in ev.events if e["type"] == "loop.pin.restored"]
    assert restored and restored[0]["pins"] == ["keep this"]  # only the fitting one
    refused = [e for e in ev.events if e["type"] == "loop.pin.refused"]
    assert len(refused) == 3  # the empty, the over-cap, and the blank


def test_a_gate_swapped_between_legs_is_announced_to_the_worker(tmp_path: Path) -> None:
    """The system prompt is the RUN's, frozen at its start. Config that gains a
    verify command between legs swaps what judges the work while the
    instructions still name the old gate, so the worker runs one command and is
    graded on another. Silence there is the worst case: it looks like it worked."""
    from agent6.workflows._session_state import SessionSnapshot as _Snap

    session_dir = tmp_path / "sessions" / "runs" / "tidy-otter-AB12CD"
    session_dir.mkdir(parents=True)
    snap_path = session_dir / "loop_state.json"
    snap_path.write_text(
        _Snap(
            system="s",
            messages=[{"role": "user", "content": [{"type": "text", "text": "go"}]}],
            tool_calls=0,
            next_iteration=3,
            root_task_id=None,
            original_task="go",
            verify_command=("pytest", "-q"),
        ).model_dump_json(),
        encoding="utf-8",
    )
    config = SimpleNamespace(
        git=_GIT_STUB,
        budget=SimpleNamespace(max_usd=10.0, max_tokens_fallback=2_000_000),
        workflow=SimpleNamespace(
            verify_when="never",
            verify_retries=2,
            verify_command=("make", "check"),  # the operator pinned one since
            metric=SimpleNamespace(goal="maximize"),
        ),
    )
    provider = MagicMock()
    provider.call.return_value = SimpleNamespace(
        text="",
        tool_uses=({"id": "t1", "name": "finish_session", "input": {"summary": "done"}},),
        stop_reason="tool_use",
        input_tokens=1,
        output_tokens=1,
        raw={
            "content": [
                {
                    "type": "tool_use",
                    "id": "t1",
                    "name": "finish_session",
                    "input": {"summary": "done"},
                }
            ]
        },
    )
    dispatcher = MagicMock()
    dispatcher.dispatch.return_value = RawResult({"ok": True})
    ev = _EventCapture(path=session_dir / "logs.jsonl")
    wf = _wf(
        provider=provider,
        dispatcher=dispatcher,
        config=config,
        mode="run",
        events=ev,
        resume_state_path=snap_path,
    )
    wf.resume()

    (swap,) = [e for e in ev.events if e["type"] == "loop.verify_swapped"]
    assert swap["was"] == ["pytest", "-q"] and swap["now"] == ["make", "check"]
    sent = provider.call.call_args.kwargs["messages"]
    told = json.dumps(sent)
    assert "was `pytest -q`" in told and "now `make check`" in told
