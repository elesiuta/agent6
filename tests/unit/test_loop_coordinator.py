# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Coordinator dispatch: `/parallel` steer fans out subordinate lanes.

Drives Workflow._drive_loop with a fake provider (steer fires once after the
first turn) and a fake GROUP spawner that fabricates real, mergeable branches in
the coordinator's tmp repo -- so the loop's dispatch phase (parse, dirty-tree
gate, sequential join, DAG stamping, events, summary message) is exercised
end-to-end without spawning real runs. A fake curator records DAG mutations.
"""

from __future__ import annotations

import subprocess as _sp
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest

from agent6.config import Config
from agent6.providers import ProviderResponse
from agent6.tools.results import RawResult
from agent6.ui.cli.parallel import build_coordinator_spawner
from agent6.workflows.loop import Workflow
from agent6.workflows.subrun import LaneResult, LaneSpec, LaneTask


def _silent(_msg: str) -> None:
    return None


# The `/parallel` grammar itself is covered in tests/unit/test_directive.py; this
# file drives the coordinator's dispatch phase end-to-end.


# ---------------------------------------------------------------------------
# Loop-driving harness
# ---------------------------------------------------------------------------


def _git(repo: Path, *args: str) -> str:
    return _sp.run(
        ["git", "-C", str(repo), *args], check=True, capture_output=True, text=True
    ).stdout.strip()


def _init_repo(repo: Path) -> None:
    repo.mkdir(parents=True, exist_ok=True)
    _git(repo, "init", "-q", "-b", "main")
    _git(repo, "config", "user.email", "t@example.com")
    _git(repo, "config", "user.name", "t")
    (repo / "README.md").write_text("hi\n", encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "init")


def _head(repo: Path) -> str:
    return _git(repo, "rev-parse", "HEAD")


def _resp_tool(name: str, args: dict[str, Any], tu_id: str) -> ProviderResponse:
    block = {"type": "tool_use", "id": tu_id, "name": name, "input": args}
    return ProviderResponse(
        text="",
        tool_uses=({"id": tu_id, "name": name, "input": args},),
        stop_reason="tool_use",
        input_tokens=1,
        output_tokens=1,
        cache_read_tokens=0,
        cache_creation_tokens=0,
        raw={"content": [block]},
    )


class _OneShotSteer:
    """Steer that fires exactly once (the first boundary poll), returning *text*."""

    def __init__(self, text: str) -> None:
        self.text = text
        self._fired = False

    def requested(self) -> bool:
        return not self._fired

    def prompt(self) -> str:
        return self.text

    def clear(self) -> None:
        self._fired = True


class _FakeGraph:
    """A minimal in-memory curator: records add/update/record_commit, so the
    loop's DAG stamping is observable without a real curator."""

    def __init__(self) -> None:
        self._seq = 0
        self.nodes: dict[str, dict[str, Any]] = {}
        self.status_calls: list[tuple[str, str, str]] = []
        self.commit_calls: list[tuple[str, str]] = []

    def add_subtask(self, intent: Any) -> Any:
        self._seq += 1
        nid = f"N{self._seq:025d}"  # 26 chars, matches TaskNode.id width
        self.nodes[nid] = {
            "parent_id": intent.parent_id,
            "title": intent.draft.title,
            "status": "pending",
            "commit_sha": "",
            "created_by": intent.draft.created_by,
        }
        return SimpleNamespace(id=nid)

    def update_status(self, intent: Any) -> Any:
        self.status_calls.append((intent.id, intent.new_status, intent.note))
        self.nodes[intent.id]["status"] = intent.new_status
        return SimpleNamespace(id=intent.id)

    def record_commit(self, intent: Any) -> Any:
        self.commit_calls.append((intent.id, intent.sha))
        self.nodes[intent.id]["commit_sha"] = intent.sha
        return SimpleNamespace(id=intent.id)


class _FakeEvents:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.emitted: list[tuple[str, dict[str, Any]]] = []

    def emit(self, event_type: str, **fields: Any) -> None:
        self.emitted.append((event_type, fields))

    def of(self, event_type: str) -> list[dict[str, Any]]:
        return [f for t, f in self.emitted if t == event_type]


def _make_branch(
    repo: Path, branch: str, base_sha: str, fname: str, content: str, wt_dir: Path
) -> None:
    """Create *branch* at a divergent commit off *base_sha* via a throwaway
    worktree, so it exists as a real ref the coordinator can merge -- without
    touching the main worktree."""
    _git(repo, "worktree", "add", "-b", branch, str(wt_dir), base_sha)
    (wt_dir / fname).write_text(content, encoding="utf-8")
    _git(wt_dir, "add", "-A")
    _git(wt_dir, "commit", "-q", "-m", f"{branch} work")
    _git(repo, "worktree", "remove", "--force", str(wt_dir))


class _FakeGroupSpawner:
    """A synchronous GroupLaneSpawner stand-in: for each lane fabricate a real
    mergeable branch in the coordinator repo and return a LaneResult. Records the
    (lanes, group) it was handed so a test can assert the parse + expansion."""

    def __init__(
        self,
        repo: Path,
        coord_id: str,
        wt_root: Path,
        *,
        fail: set[int] | None = None,
        conflict: set[int] | None = None,
        base_by_lane: dict[int, str] | None = None,
    ) -> None:
        self.repo = repo
        self.coord_id = coord_id
        self.wt_root = wt_root
        self.fail = fail or set()
        self.conflict = conflict or set()
        self.base_by_lane = base_by_lane or {}
        self.calls: list[tuple[list[LaneTask], str]] = []

    def tasks(self, call: int = 0) -> list[str]:
        return [lane.task for lane in self.calls[call][0]]

    def __call__(
        self, lanes: list[LaneTask], group: str, *, at: str | None = None
    ) -> list[LaneResult]:
        self.calls.append((list(lanes), group))
        results: list[LaneResult] = []
        for i, lane in enumerate(lanes, start=1):
            session_id = f"{self.coord_id}-{group}-l{i}"
            branch = f"agent6/{session_id}"
            spec = LaneSpec(
                lane=i, session_id=session_id, workdir=self.wt_root / session_id, model=lane.model
            )
            if i in self.fail:
                results.append(
                    LaneResult(
                        spec=spec,
                        session_dir=spec.workdir,
                        branch=branch,
                        ok=False,
                        error="lane boom",
                    )
                )
                continue
            base = self.base_by_lane.get(i, _head(self.repo))
            if i in self.conflict:
                _make_branch(
                    self.repo, branch, base, "conflict.txt", "lane version", self.wt_root / f"wt{i}"
                )
            else:
                _make_branch(
                    self.repo, branch, base, f"lane{i}.txt", f"lane {i}\n", self.wt_root / f"wt{i}"
                )
            results.append(
                LaneResult(spec=spec, session_dir=spec.workdir, branch=branch, ok=True, error="")
            )
        return results


def _build_wf(
    repo: Path,
    provider: MagicMock,
    *,
    steer_text: str,
    lane_spawner: Any = None,
    graph: Any = None,
    events: Any = None,
    dispatcher: MagicMock | None = None,
    verify_command: tuple[str, ...] = (),
    verify_when: str = "finish",
    max_iterations: int = 2,
) -> Workflow:
    steer = _OneShotSteer(steer_text)
    disp = dispatcher if dispatcher is not None else MagicMock()
    if dispatcher is None:
        disp.dispatch.return_value = RawResult({"content": "hi\n"})
    return Workflow(
        root=repo,
        chain_ref="refs/agent6/coord",
        chain_fallback_parent=_head(repo),
        config=MagicMock(
            budget=SimpleNamespace(max_usd=10.0, max_tokens_fallback=2_000_000),
            prompt=MagicMock(system_prompt_file=""),
            workflow=MagicMock(
                verify_command=verify_command, verify_when=verify_when, verify_retries=2
            ),
            # A real int, not a Mock: segment_lanes compares the spec's lane
            # count against this cap.
            parallel=SimpleNamespace(max_lanes=4),
        ),
        provider=provider,
        dispatcher=disp,
        logger=_silent,
        events=events,
        curator=graph,
        lane_spawner=lane_spawner,
        provider_retry_count=0,
        provider_retry_delay_s=0.0,
        max_iterations=max_iterations,
        steer_requested=steer.requested,
        steer_prompt=steer.prompt,
        steer_clear=steer.clear,
    )


def _user_texts(messages: list[dict[str, Any]]) -> list[str]:
    out: list[str] = []
    for msg in messages:
        if msg.get("role") != "user":
            continue
        content = msg.get("content", [])
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    out.append(str(block.get("text", "")))
    return out


def _final_messages(provider: MagicMock) -> list[dict[str, Any]]:
    last = provider.call.call_args_list[-1]
    return last.kwargs.get("messages") or last.args[1]


# ---------------------------------------------------------------------------
# Dispatch behaviour
# ---------------------------------------------------------------------------


def test_none_spawner_answers_with_feedback_and_continues(tmp_path: Path) -> None:
    """No lane_spawner (default / headless) -> the directive is answered with a
    'not available' notice and the run continues; never a crash."""
    repo = tmp_path / "repo"
    _init_repo(repo)
    provider = MagicMock()
    provider.call.side_effect = [
        _resp_tool("read_file", {"path": "README.md"}, "t1"),
        _resp_tool("read_file", {"path": "README.md"}, "t2"),
    ]
    wf = _build_wf(repo, provider, steer_text="/parallel do a thing", lane_spawner=None)
    result = wf.run("start")

    assert provider.call.call_count == 2  # ran to max_iterations, no crash
    assert result.reason == "max_iterations"
    texts = _user_texts(_final_messages(provider))
    assert any("parallel dispatch is not available" in t for t in texts)


def test_parallel_lookalike_steer_flows_through_as_plain_steer(tmp_path: Path) -> None:
    """A steer beginning `/parallelfoo ...` is NOT a directive: it reaches the
    model verbatim as an OPERATOR STEERING message, and nothing is dispatched or
    answered with parse feedback."""
    repo = tmp_path / "repo"
    _init_repo(repo)
    spawner = _FakeGroupSpawner(repo, "run-lk", tmp_path / "wt")
    provider = MagicMock()
    provider.call.side_effect = [
        _resp_tool("read_file", {"path": "README.md"}, "t1"),
        _resp_tool("read_file", {"path": "README.md"}, "t2"),
    ]
    wf = _build_wf(repo, provider, steer_text="/parallelfoo do x", lane_spawner=spawner)
    wf.run("start")

    assert spawner.calls == []  # never parsed as a directive
    texts = _user_texts(_final_messages(provider))
    assert any("OPERATOR STEERING" in t and "/parallelfoo do x" in t for t in texts)
    assert not any(t.startswith("[parallel]") for t in texts)


def test_dispatch_joins_in_order_and_stamps_dag(tmp_path: Path) -> None:
    """Two clean lanes -> both branches join in dispatch order, each DAG node is
    passed with its join sha, and dispatched/joined events fire."""
    repo = tmp_path / "repo"
    _init_repo(repo)
    coord_id = "run-abc"
    events = _FakeEvents(tmp_path / coord_id / "logs.jsonl")
    graph = _FakeGraph()
    spawner = _FakeGroupSpawner(repo, coord_id, tmp_path / "wt")

    provider = MagicMock()
    provider.call.side_effect = [
        _resp_tool("read_file", {"path": "README.md"}, "t1"),
        _resp_tool("read_file", {"path": "README.md"}, "t2"),
    ]
    wf = _build_wf(
        repo,
        provider,
        steer_text="/parallel task one\n/parallel task two",
        lane_spawner=spawner,
        graph=graph,
        events=events,
    )
    wf.run("start")

    # The spawner saw the parsed sibling group under a p1 group id.
    assert spawner.tasks() == ["task one", "task two"]
    assert spawner.calls[0][1] == "p1"
    # Both lane branches merged onto the coordinator's chain, in order; the
    # operator's HEAD never moves.
    log = _git(repo, "log", "--oneline", "refs/agent6/coord")
    assert "agent6/run-abc-p1-l1" in log
    assert "agent6/run-abc-p1-l2" in log
    assert "l1" not in _git(repo, "log", "--oneline")
    assert log.index("l1") > log.index("l2")  # l1 merged first => older => lower in log
    # Two steering nodes were added (besides the seeded root) and both passed
    # with a recorded commit sha.
    steering = [n for n in graph.nodes.values() if n["created_by"] == "steering"]
    assert len(steering) == 2
    assert all(n["status"] == "passed" and n["commit_sha"] for n in steering)
    assert len(graph.commit_calls) == 2
    # Events render the fan-out truthfully: dispatched carries tasks + group
    # (lane ids do not exist yet); joined names the REAL ids from the results.
    dispatched = events.of("loop.parallel.dispatched")
    joined = events.of("loop.parallel.joined")
    assert dispatched and dispatched[0] == {"group": "p1", "tasks": ["task one", "task two"]}
    assert joined and [ln["status"] for ln in joined[0]["lanes"]] == ["joined", "joined"]
    assert [ln["session_id"] for ln in joined[0]["lanes"]] == ["run-abc-p1-l1", "run-abc-p1-l2"]
    # The join names the group the lanes were stamped with -- the id
    # `sessions compare` takes -- not the loop's local counter.
    assert joined[0]["group"] == "run-abc-p1"
    assert not events.of("loop.parallel.failed")
    # One summary message names both joined lanes.
    summary = [
        t for t in _user_texts(_final_messages(provider)) if t.startswith("[parallel] group")
    ]
    assert len(summary) == 1
    assert "joined at" in summary[0]


def test_model_list_spec_expands_to_one_lane_per_model(tmp_path: Path) -> None:
    """`/parallel m1,m2 <task>` -> two lanes of that task, one per model, under a
    SINGLE segment DAG node that records the last joined sha."""
    repo = tmp_path / "repo"
    _init_repo(repo)
    coord_id = "run-mdl"
    events = _FakeEvents(tmp_path / coord_id / "logs.jsonl")
    graph = _FakeGraph()
    spawner = _FakeGroupSpawner(repo, coord_id, tmp_path / "wt")
    provider = MagicMock()
    provider.call.side_effect = [
        _resp_tool("read_file", {"path": "README.md"}, "t1"),
        _resp_tool("read_file", {"path": "README.md"}, "t2"),
    ]
    wf = _build_wf(
        repo,
        provider,
        steer_text="/parallel kimi,glm refactor the parser",
        lane_spawner=spawner,
        graph=graph,
        events=events,
    )
    wf.run("start")

    # Two lanes, same task, one per model, in list order.
    lanes = spawner.calls[0][0]
    assert [(ln.task, ln.model) for ln in lanes] == [
        ("refactor the parser", "kimi"),
        ("refactor the parser", "glm"),
    ]
    # ONE DAG node for the segment (not one per lane), passed with a join sha, and
    # its note names both lanes.
    steering = [n for n in graph.nodes.values() if n["created_by"] == "steering"]
    assert len(steering) == 1
    assert steering[0]["status"] == "passed" and steering[0]["commit_sha"]
    note = next(note for _id, _status, note in graph.status_calls)
    assert f"{coord_id}-p1-l1" in note and f"{coord_id}-p1-l2" in note
    # Both lanes still surface per-lane in the joined event.
    joined = events.of("loop.parallel.joined")[0]
    assert [ln["session_id"] for ln in joined["lanes"]] == [
        f"{coord_id}-p1-l1",
        f"{coord_id}-p1-l2",
    ]


def test_lane_count_spec_expands_to_n_default_lanes(tmp_path: Path) -> None:
    """`/parallel 3 <task>` -> three lanes on the default model, one segment node."""
    repo = tmp_path / "repo"
    _init_repo(repo)
    coord_id = "run-cnt"
    graph = _FakeGraph()
    spawner = _FakeGroupSpawner(repo, coord_id, tmp_path / "wt")
    provider = MagicMock()
    provider.call.side_effect = [
        _resp_tool("read_file", {"path": "README.md"}, "t1"),
        _resp_tool("read_file", {"path": "README.md"}, "t2"),
    ]
    wf = _build_wf(
        repo,
        provider,
        steer_text="/parallel 3 tidy the imports",
        lane_spawner=spawner,
        graph=graph,
    )
    wf.run("start")

    lanes = spawner.calls[0][0]
    assert [(ln.task, ln.model) for ln in lanes] == [("tidy the imports", None)] * 3
    steering = [n for n in graph.nodes.values() if n["created_by"] == "steering"]
    assert len(steering) == 1  # one node for the segment, not three


def test_join_conflict_emits_event_message_and_continues(tmp_path: Path) -> None:
    """A lane whose branch conflicts -> join returns None -> node failed, a
    loop.parallel.failed event fires, the summary tells the model to merge
    manually, and the run continues."""
    repo = tmp_path / "repo"
    _init_repo(repo)
    base = _head(repo)  # before conflict.txt exists
    (repo / "conflict.txt").write_text("main version", encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "main adds conflict.txt")

    coord_id = "run-cf"
    events = _FakeEvents(tmp_path / coord_id / "logs.jsonl")
    graph = _FakeGraph()
    # Lane 1 clean; lane 2 conflicts (cut from `base`, before conflict.txt).
    spawner = _FakeGroupSpawner(
        repo, coord_id, tmp_path / "wt", conflict={2}, base_by_lane={2: base}
    )
    provider = MagicMock()
    provider.call.side_effect = [
        _resp_tool("read_file", {"path": "README.md"}, "t1"),
        _resp_tool("read_file", {"path": "README.md"}, "t2"),
    ]
    wf = _build_wf(
        repo,
        provider,
        steer_text="/parallel clean lane\n/parallel conflicting lane",
        lane_spawner=spawner,
        graph=graph,
        events=events,
    )
    result = wf.run("start")

    assert provider.call.call_count == 2  # run continued past the conflict
    assert result.reason == "max_iterations"
    # The clean lane's file reached the worktree (chain sync); the conflicted
    # merge touched nothing.
    assert (repo / "lane1.txt").read_text(encoding="utf-8") == "lane 1\n"
    assert (repo / "conflict.txt").read_text(encoding="utf-8") == "main version"
    joined = events.of("loop.parallel.joined")[0]
    assert [ln["status"] for ln in joined["lanes"]] == ["joined", "conflict"]
    failed = events.of("loop.parallel.failed")
    assert failed and [ln["session_id"] for ln in failed[0]["lanes"]] == [f"{coord_id}-p1-l2"]
    # The conflicting node is marked failed (NodeStatus has no "blocked").
    steering = [n for n in graph.nodes.values() if n["created_by"] == "steering"]
    assert sorted(n["status"] for n in steering) == ["failed", "passed"]
    summary = next(t for t in _user_texts(_final_messages(provider)) if t.startswith("[parallel]"))
    assert "CONFLICT" in summary
    assert "git merge agent6/run-cf-p1-l2" in summary


def test_failed_lane_is_reported_truthfully(tmp_path: Path) -> None:
    """A lane the spawner could not run (ok=False) -> node failed, summary says
    FAILED with the reason, run continues."""
    repo = tmp_path / "repo"
    _init_repo(repo)
    coord_id = "run-fl"
    events = _FakeEvents(tmp_path / coord_id / "logs.jsonl")
    graph = _FakeGraph()
    spawner = _FakeGroupSpawner(repo, coord_id, tmp_path / "wt", fail={1})
    provider = MagicMock()
    provider.call.side_effect = [
        _resp_tool("read_file", {"path": "README.md"}, "t1"),
        _resp_tool("read_file", {"path": "README.md"}, "t2"),
    ]
    wf = _build_wf(
        repo,
        provider,
        steer_text="/parallel doomed lane\n/parallel good lane",
        lane_spawner=spawner,
        graph=graph,
        events=events,
    )
    result = wf.run("start")

    assert result.reason == "max_iterations"  # did not crash
    joined = events.of("loop.parallel.joined")[0]
    assert [ln["status"] for ln in joined["lanes"]] == ["failed", "joined"]
    steering = [n for n in graph.nodes.values() if n["created_by"] == "steering"]
    assert sorted(n["status"] for n in steering) == ["failed", "passed"]
    summary = next(t for t in _user_texts(_final_messages(provider)) if t.startswith("[parallel]"))
    assert "FAILED -- lane boom" in summary


def test_spawner_raising_mid_group_never_aborts_the_run(tmp_path: Path) -> None:
    """The group spawner raising (mkdir OSError, a pool-propagated spawn fault,
    a result-count mismatch) must not abort the coordinator: the run continues,
    loop.parallel.failed fires, a truthful feedback message is injected, and no
    steering node is left pending."""
    repo = tmp_path / "repo"
    _init_repo(repo)
    coord_id = "run-boom"
    events = _FakeEvents(tmp_path / coord_id / "logs.jsonl")
    graph = _FakeGraph()

    calls: list[str] = []

    def exploding_spawner(
        lanes: list[LaneTask], group: str, *, at: str | None = None
    ) -> list[LaneResult]:
        calls.append(group)
        raise OSError("disk full while cloning lane 2")

    provider = MagicMock()
    provider.call.side_effect = [
        _resp_tool("read_file", {"path": "README.md"}, "t1"),
        _resp_tool("read_file", {"path": "README.md"}, "t2"),
    ]
    wf = _build_wf(
        repo,
        provider,
        steer_text="/parallel task one\n/parallel task two",
        lane_spawner=exploding_spawner,
        graph=graph,
        events=events,
    )
    result = wf.run("start")

    assert calls == ["p1"]  # the dispatch was attempted
    assert provider.call.call_count == 2  # the run CONTINUED past the fault
    assert result.reason == "max_iterations"
    # The failure is rendered truthfully on every surface.
    failed = events.of("loop.parallel.failed")
    assert failed and failed[0]["group"] == "p1"
    assert "disk full" in str(failed[0])
    texts = _user_texts(_final_messages(provider))
    assert any("disk full while cloning lane 2" in t for t in texts)
    # No steering node is orphaned as pending.
    steering = [n for n in graph.nodes.values() if n["created_by"] == "steering"]
    assert len(steering) == 2
    assert all(n["status"] == "failed" for n in steering)


def test_bare_parallel_directive_dispatches_nothing(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _init_repo(repo)
    spawner = _FakeGroupSpawner(repo, "run-x", tmp_path / "wt")
    provider = MagicMock()
    provider.call.side_effect = [
        _resp_tool("read_file", {"path": "README.md"}, "t1"),
        _resp_tool("read_file", {"path": "README.md"}, "t2"),
    ]
    wf = _build_wf(repo, provider, steer_text="/parallel", lane_spawner=spawner)
    wf.run("start")
    assert spawner.calls == []  # nothing dispatched
    texts = _user_texts(_final_messages(provider))
    assert any("nothing dispatched" in t for t in texts)


def test_dirty_tree_is_auto_committed_then_dispatched(tmp_path: Path) -> None:
    """A changed worktree at the boundary is chain-committed (lanes cut from
    the chain tip only), then dispatch proceeds."""
    repo = tmp_path / "repo"
    _init_repo(repo)
    coord_id = "run-dirty"
    events = _FakeEvents(tmp_path / coord_id / "logs.jsonl")
    spawner = _FakeGroupSpawner(repo, coord_id, tmp_path / "wt")

    # The first turn's tool "edits" the tree (an uncommitted run_command write).
    dispatcher = MagicMock()

    def dispatch(_name: str, _input: dict[str, Any]) -> RawResult:
        (repo / "wip.txt").write_text("uncommitted work\n", encoding="utf-8")
        return RawResult({"content": "wrote wip.txt"})

    dispatcher.dispatch.side_effect = dispatch

    provider = MagicMock()
    provider.call.side_effect = [
        _resp_tool("run_command", {"command": "echo hi > wip.txt"}, "t1"),
        _resp_tool("read_file", {"path": "README.md"}, "t2"),
    ]
    wf = _build_wf(
        repo,
        provider,
        steer_text="/parallel keep going",
        lane_spawner=spawner,
        events=events,
        dispatcher=dispatcher,
        # A model-run gate: the wip edit stays uncommitted until dispatch.
        verify_command=("true",),
        verify_when="never",
    )
    wf.run("start")

    assert spawner.calls and spawner.calls[0][1] == "p1"  # dispatched
    # The wip edit was captured by the pre-dispatch checkpoint -- on the chain.
    assert _git(repo, "show", "refs/agent6/coord:wip.txt") == "uncommitted work"
    chain_log = _git(repo, "log", "--oneline", "refs/agent6/coord")
    assert "checkpoint before /parallel dispatch" in chain_log
    assert events.of("loop.auto_commit")  # a checkpoint commit was emitted


def test_dirty_tree_that_cannot_be_cleaned_refuses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If the tree stays dirty after the auto-commit attempt, dispatch is refused
    (never clone stale work) and the run continues."""
    import agent6.workflows.loop as loop_mod

    # chain_commit becomes a no-op, so the tree stays dirty after the attempt.
    def _noop_commit(*_a: object, **_k: object) -> None:
        return None

    monkeypatch.setattr(loop_mod, "chain_commit", _noop_commit)

    repo = tmp_path / "repo"
    _init_repo(repo)
    spawner = _FakeGroupSpawner(repo, "run-nd", tmp_path / "wt")

    dispatcher = MagicMock()

    def dispatch(_name: str, _input: dict[str, Any]) -> RawResult:
        (repo / "wip.txt").write_text("uncommitted\n", encoding="utf-8")
        return RawResult({"content": "ok"})

    dispatcher.dispatch.side_effect = dispatch

    provider = MagicMock()
    provider.call.side_effect = [
        _resp_tool("run_command", {"command": "x"}, "t1"),
        _resp_tool("read_file", {"path": "README.md"}, "t2"),
    ]
    wf = _build_wf(
        repo,
        provider,
        steer_text="/parallel keep going",
        lane_spawner=spawner,
        dispatcher=dispatcher,
        verify_command=("true",),  # a model-run gate: the wip edit stays uncommitted
        verify_when="never",
    )
    wf.run("start")

    assert spawner.calls == []  # refused: nothing dispatched
    texts = _user_texts(_final_messages(provider))
    assert any("could not be" in t and "auto-committed" in t for t in texts)


# ---------------------------------------------------------------------------
# run.py / resume.py wiring gate (depth 1 by construction)
# ---------------------------------------------------------------------------


def test_coordinator_spawner_gate_under_subrun(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = Config()
    monkeypatch.delenv("AGENT6_SUBRUN", raising=False)
    # A write run outside a lane gets a real dispatcher.
    assert callable(build_coordinator_spawner(cfg, tmp_path, tmp_path, mode="run", session_id="r"))
    # plan/ask make no commits to clone -> no dispatcher.
    assert build_coordinator_spawner(cfg, tmp_path, tmp_path, mode="plan", session_id="r") is None
    # Inside a subordinate lane -> no dispatcher (depth 1).
    monkeypatch.setenv("AGENT6_SUBRUN", "1")
    assert build_coordinator_spawner(cfg, tmp_path, tmp_path, mode="run", session_id="r") is None
