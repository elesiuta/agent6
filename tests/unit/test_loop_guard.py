# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""degenerate-loop guard in Workflow._drive_loop.

When the worker calls the same (tool_name, args) back-to-back >=3 times,
the workflow appends a one-shot "loop-guard" text block to the next user
turn telling the worker the result has not changed and to pivot. Behaviour
observed live with Kimi K2.6 on the perf takehome: 15 consecutive
`read_file(path="problem.py")` calls returning the same 19826 bytes,
followed by went_quiet.
"""

from __future__ import annotations

import itertools
import subprocess as _sp
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

from agent6.providers import ProviderResponse
from agent6.tools.results import RawResult
from agent6.workflows.loop import Workflow


def _silent(_msg: str) -> None:
    return None


def _resp_with_tool(name: str, args: dict[str, Any], tu_id: str = "tu1") -> ProviderResponse:
    """Provider response with a single tool_use."""
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


def _resp_text(text: str = "done") -> ProviderResponse:
    return ProviderResponse(
        text=text,
        tool_uses=(),
        stop_reason="end_turn",
        input_tokens=1,
        output_tokens=1,
        cache_read_tokens=0,
        cache_creation_tokens=0,
    )


def _init_repo(repo: Path) -> None:
    repo.mkdir()
    _sp.run(["git", "init", "-q", "-b", "main"], cwd=repo, check=True)
    _sp.run(["git", "config", "user.email", "t@example.com"], cwd=repo, check=True)
    _sp.run(["git", "config", "user.name", "t"], cwd=repo, check=True)
    (repo / "x.txt").write_text("hi\n")
    _sp.run(["git", "add", "x.txt"], cwd=repo, check=True)
    _sp.run(["git", "commit", "-q", "-m", "init"], cwd=repo, check=True)


def _build_wf(repo: Path, provider: MagicMock, dispatcher: MagicMock) -> Workflow:
    return Workflow(
        root=repo,
        chain_ref="refs/agent6/guard",
        chain_fallback_parent=_sp.run(
            ["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True, check=True
        ).stdout.strip(),
        config=MagicMock(
            budget=SimpleNamespace(max_usd=10.0, max_tokens_fallback=2_000_000),
            prompt=MagicMock(system_prompt_file=""),
            workflow=MagicMock(verify_command=(), verify_when="never", verify_retries=2),
        ),
        provider=provider,
        dispatcher=dispatcher,
        logger=_silent,
        provider_retry_count=0,
        provider_retry_delay_s=0.0,
        max_iterations=10,
    )


def _loop_guard_blocks(messages: list[dict[str, Any]]) -> list[str]:
    """Extract the text of every [loop-guard] block injected into user turns."""
    out: list[str] = []
    for msg in messages:
        if msg.get("role") != "user":
            continue
        content = msg.get("content", [])
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict) or block.get("type") != "text":
                continue
            text = block.get("text", "")
            if text.startswith("[loop-guard]"):
                out.append(text)
    return out


def test_loop_guard_fires_on_three_identical_calls(tmp_path: Path) -> None:
    """3 back-to-back identical tool calls -> one loop-guard notice appended."""
    repo = tmp_path / "repo"
    _init_repo(repo)

    provider = MagicMock()
    # Turns 1-3: same read_file call. Turn 4: silent finish.
    provider.call.side_effect = [
        _resp_with_tool("read_file", {"path": "x.txt"}, tu_id="t1"),
        _resp_with_tool("read_file", {"path": "x.txt"}, tu_id="t2"),
        _resp_with_tool("read_file", {"path": "x.txt"}, tu_id="t3"),
        _resp_text("ok"),
    ]
    dispatcher = MagicMock(operator_wait_s=0.0)
    dispatcher.dispatch.return_value = RawResult({"content": "hi\n"})

    wf = _build_wf(repo, provider, dispatcher)
    result = wf.run("read the file")

    assert result.completed is True
    assert provider.call.call_count == 4

    # Reconstruct messages from the provider call history (last call's
    # messages arg holds the full conversation).
    last_args = provider.call.call_args_list[-1]
    final_messages: list[dict[str, Any]] = last_args.kwargs.get("messages") or last_args.args[1]
    notices = _loop_guard_blocks(final_messages)
    assert len(notices) == 1, f"expected exactly one notice, got {len(notices)}: {notices}"
    assert "read_file" in notices[0]
    assert "3 times" in notices[0]


def test_polling_a_growing_result_is_not_a_spiral(tmp_path: Path) -> None:
    """`run_command`'s own description tells the model to poll a background
    job with `read_background`, whose args never change until the job ends.
    Counted as a repeat it drew three nudges and then killed the run for
    following the instruction. A poll is not a repeat -- and a repeat of any
    OTHER tool still is, however its bytes differ (a duration, a timestamp):
    re-running the same failing command is the spiral the guard exists for."""
    repo = tmp_path / "repo"
    _init_repo(repo)

    provider = MagicMock()
    provider.call.side_effect = [
        *(_resp_with_tool("read_background", {"id": "bg-1"}, tu_id=f"t{i}") for i in range(1, 10)),
        _resp_text("ok"),
    ]
    dispatcher = MagicMock(operator_wait_s=0.0)
    tail = iter(f"line {i}\n" * i for i in range(1, 20))

    def _growing_tail(*_args: object, **_kwargs: object) -> RawResult:
        return RawResult({"output": next(tail)})

    dispatcher.dispatch.side_effect = _growing_tail

    wf = _build_wf(repo, provider, dispatcher)
    result = wf.run("watch the build")

    assert result.completed is True, result.reason
    assert result.reason != "loop_guard_killed"
    last_args = provider.call.call_args_list[-1]
    final_messages: list[dict[str, Any]] = last_args.kwargs.get("messages") or last_args.args[1]
    assert _loop_guard_blocks(final_messages) == []

    # The same shape on a NON-poll tool is a spiral, whatever its bytes do: a
    # command re-run with identical arguments whose only difference is its own
    # duration is the case the guard was written for.
    spiraller = MagicMock()
    spiraller.call.side_effect = itertools.chain(
        (_resp_with_tool("run_command", {"argv": ["pytest"]}, tu_id=f"s{i}") for i in range(1, 10)),
        itertools.repeat(_resp_text("ok")),
    )
    spins = MagicMock(operator_wait_s=0.0)
    runs = iter(f"1 failed in {i}.0s\n" for i in range(1, 20))

    def _timestamped(*_args: object, **_kwargs: object) -> RawResult:
        return RawResult({"stdout": next(runs)})

    spins.dispatch.side_effect = _timestamped
    wf2 = _build_wf(repo, spiraller, spins)
    wf2.run("make the tests pass")
    last2 = spiraller.call.call_args_list[-1]
    msgs2: list[dict[str, Any]] = last2.kwargs.get("messages") or last2.args[1]
    assert _loop_guard_blocks(msgs2), "a repeated command with a changing duration is a spiral"


def test_loop_guard_does_not_fire_when_args_change(tmp_path: Path) -> None:
    """Different args every turn -> no loop-guard notice."""
    repo = tmp_path / "repo"
    _init_repo(repo)

    provider = MagicMock()
    provider.call.side_effect = [
        _resp_with_tool("read_file", {"path": "a.txt"}, tu_id="t1"),
        _resp_with_tool("read_file", {"path": "b.txt"}, tu_id="t2"),
        _resp_with_tool("read_file", {"path": "c.txt"}, tu_id="t3"),
        _resp_with_tool("read_file", {"path": "d.txt"}, tu_id="t4"),
        _resp_text("ok"),
    ]
    dispatcher = MagicMock(operator_wait_s=0.0)
    dispatcher.dispatch.return_value = RawResult({"content": "x"})

    wf = _build_wf(repo, provider, dispatcher)
    result = wf.run("read several files")

    assert result.completed is True
    last_args = provider.call.call_args_list[-1]
    final_messages: list[dict[str, Any]] = last_args.kwargs.get("messages") or last_args.args[1]
    assert _loop_guard_blocks(final_messages) == []


def test_loop_guard_does_not_re_fire_back_to_back(tmp_path: Path) -> None:
    """Once notice is emitted at iter N, do not emit again at iter N+1 even if streak continues."""
    repo = tmp_path / "repo"
    _init_repo(repo)

    provider = MagicMock()
    # 5 identical calls then finish.
    provider.call.side_effect = [
        _resp_with_tool("read_file", {"path": "x.txt"}, tu_id=f"t{i}") for i in range(5)
    ] + [_resp_text("ok")]
    dispatcher = MagicMock(operator_wait_s=0.0)
    dispatcher.dispatch.return_value = RawResult({"content": "hi\n"})

    wf = _build_wf(repo, provider, dispatcher)
    wf.run("loop")

    last_args = provider.call.call_args_list[-1]
    final_messages: list[dict[str, Any]] = last_args.kwargs.get("messages") or last_args.args[1]
    notices = _loop_guard_blocks(final_messages)
    # The guard fires once when streak hits 3. The
    # `spiral.warned_at_iteration < iteration - 1` gate suppresses
    # re-emission at iter 4 (consecutive) but allows re-emission at
    # iter 5 (one-iteration gap) if the streak persists. So we expect
    # 1 or 2 notices, but NOT one per iteration.
    assert 1 <= len(notices) <= 2, f"expected 1-2 notices, got {len(notices)}"


def test_loop_guard_kills_run_when_streak_passes_threshold(tmp_path: Path) -> None:
    """Notice is advisory; when the streak reaches
    `loop_guard_kill_threshold` the run terminates."""
    repo = tmp_path / "repo"
    _init_repo(repo)

    provider = MagicMock()
    # 12 identical calls. Threshold=5 -> kill at iter 5.
    provider.call.side_effect = [
        _resp_with_tool("read_file", {"path": "x.txt"}, tu_id=f"t{i}") for i in range(12)
    ] + [_resp_text("never reached")]
    dispatcher = MagicMock(operator_wait_s=0.0)
    dispatcher.dispatch.return_value = RawResult({"content": "hi\n"})

    wf = Workflow(
        root=repo,
        config=MagicMock(
            budget=SimpleNamespace(max_usd=10.0, max_tokens_fallback=2_000_000),
            prompt=MagicMock(system_prompt_file=""),
            workflow=MagicMock(verify_command=(), verify_when="never", verify_retries=2),
        ),
        provider=provider,
        dispatcher=dispatcher,
        logger=_silent,
        provider_retry_count=0,
        provider_retry_delay_s=0.0,
        max_iterations=20,
        loop_guard_kill_threshold=5,
    )
    result = wf.run("loop forever")

    assert result.completed is False
    assert result.reason == "loop_guard_killed"
    assert provider.call.call_count == 5
    assert "read_file" in result.summary
    assert "5x" in result.summary or "5 " in result.summary


def test_loop_guard_kill_disabled_when_threshold_zero(tmp_path: Path) -> None:
    """`loop_guard_kill_threshold = 0` restores notice-only behaviour."""
    repo = tmp_path / "repo"
    _init_repo(repo)

    provider = MagicMock()
    provider.call.side_effect = [
        _resp_with_tool("read_file", {"path": "x.txt"}, tu_id=f"t{i}") for i in range(6)
    ] + [_resp_text("done")]
    dispatcher = MagicMock(operator_wait_s=0.0)
    dispatcher.dispatch.return_value = RawResult({"content": "hi\n"})

    wf = Workflow(
        root=repo,
        config=MagicMock(
            budget=SimpleNamespace(max_usd=10.0, max_tokens_fallback=2_000_000),
            prompt=MagicMock(system_prompt_file=""),
            workflow=MagicMock(verify_command=(), verify_when="never", verify_retries=2),
        ),
        provider=provider,
        dispatcher=dispatcher,
        logger=_silent,
        provider_retry_count=0,
        provider_retry_delay_s=0.0,
        max_iterations=20,
        loop_guard_kill_threshold=0,
    )
    result = wf.run("loop")

    assert result.completed is True
    assert provider.call.call_count == 7


def _gated_wf(repo: Path, provider: MagicMock, dispatcher: MagicMock, **kw: Any) -> Workflow:
    """A GATED workflow (verify_command set), where the per-turn auto-commit
    fires only on a green verify -- so a run_command-authored edit stays in the
    worktree and only a final checkpoint can get it into git history."""
    return Workflow(
        root=repo,
        chain_ref="refs/agent6/guard",
        chain_fallback_parent=_sp.run(
            ["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True, check=True
        ).stdout.strip(),
        config=MagicMock(
            budget=SimpleNamespace(max_usd=10.0, max_tokens_fallback=2_000_000),
            prompt=MagicMock(system_prompt_file=""),
            workflow=MagicMock(verify_command=("true",), verify_when="never", verify_retries=2),
        ),
        provider=provider,
        dispatcher=dispatcher,
        logger=_silent,
        provider_retry_count=0,
        provider_retry_delay_s=0.0,
        **kw,
    )


def _dirtying_dispatcher(repo: Path) -> MagicMock:
    """A dispatcher whose tool leaves an uncommitted edit, as run_command does."""
    dispatcher = MagicMock(operator_wait_s=0.0)

    def dispatch(*_args: Any, **_kwargs: Any) -> RawResult:
        (repo / "edit.txt").write_text("run_command wrote this\n")
        return RawResult({"content": "hi\n"})

    dispatcher.dispatch.side_effect = dispatch
    return dispatcher


def _git_log(repo: Path) -> str:
    """The run's commit line: the chain ref (checkpoints never touch HEAD)."""
    return _sp.run(
        ["git", "log", "--oneline", "refs/agent6/guard"],
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
    ).stdout


def test_loop_guard_kill_checkpoints_the_dirty_worktree(tmp_path: Path) -> None:
    """A harness-initiated stop must not drop run_command-authored edits: every
    agent6 surface (runs diff, merge, resume, score) reads git history, not the
    worktree, so a kill that leaves the tree dirty loses the work from all of
    them. The sibling max_iterations stop already checkpoints."""
    repo = tmp_path / "repo"
    _init_repo(repo)
    provider = MagicMock()
    provider.call.side_effect = [
        _resp_with_tool("read_file", {"path": "x.txt"}, tu_id=f"t{i}") for i in range(12)
    ] + [_resp_text("never reached")]

    wf = _gated_wf(
        repo,
        provider,
        _dirtying_dispatcher(repo),
        max_iterations=20,
        loop_guard_kill_threshold=5,
    )
    result = wf.run("loop forever")

    assert result.reason == "loop_guard_killed"
    assert "checkpoint (iter 5)" in _git_log(repo)


def test_max_iterations_stop_checkpoints_the_dirty_worktree(tmp_path: Path) -> None:
    """The contract the loop-guard kill has to match."""
    repo = tmp_path / "repo"
    _init_repo(repo)
    provider = MagicMock()
    provider.call.side_effect = [
        _resp_with_tool("read_file", {"path": "x.txt"}, tu_id=f"t{i}") for i in range(12)
    ]

    wf = _gated_wf(
        repo,
        provider,
        _dirtying_dispatcher(repo),
        max_iterations=3,
        loop_guard_kill_threshold=0,
    )
    result = wf.run("keep going")

    assert result.reason == "max_iterations"
    assert "checkpoint (iter 3)" in _git_log(repo)


def test_budget_exhausted_checkpoints_the_dirty_worktree(tmp_path: Path) -> None:
    """Every harness-initiated end must checkpoint (the loop-guard rule): a
    BudgetExceeded on the next provider call ended the run with the prior
    turn's run_command edit only in the worktree, invisible to runs
    diff/merge/score."""
    from agent6.budget import BudgetExceeded

    repo = tmp_path / "repo"
    _init_repo(repo)
    provider = MagicMock()
    provider.call.side_effect = [
        _resp_with_tool("read_file", {"path": "x.txt"}),
        BudgetExceeded("input cap reached"),
    ]
    wf = _gated_wf(repo, provider, _dirtying_dispatcher(repo), max_iterations=5)
    result = wf.run("do the thing")
    assert result.reason == "budget_exhausted"
    assert "checkpoint (iter" in _git_log(repo)


def test_provider_error_checkpoints_the_dirty_worktree(tmp_path: Path) -> None:
    """A fatal provider error (permanent status / retries exhausted) is a
    harness-initiated end too; the run is resumable and its edits must be in
    git history like every sibling stop."""
    from agent6.providers import ProviderError

    repo = tmp_path / "repo"
    _init_repo(repo)
    provider = MagicMock()
    provider.call.side_effect = [
        _resp_with_tool("read_file", {"path": "x.txt"}),
        ProviderError("HTTP 401", status_code=401),
    ]
    wf = _gated_wf(repo, provider, _dirtying_dispatcher(repo), max_iterations=5)
    result = wf.run("do the thing")
    assert result.reason == "provider_error"
    assert "checkpoint (iter" in _git_log(repo)


def test_went_quiet_checkpoints_the_dirty_worktree(tmp_path: Path, monkeypatch: Any) -> None:
    """A model that starves into empty turns after making real edits must not
    lose them from git history on the way out."""
    monkeypatch.delenv("AGENT6_WENT_QUIET_MAX_NUDGES", raising=False)
    repo = tmp_path / "repo"
    _init_repo(repo)
    provider = MagicMock()
    provider.call.side_effect = [
        _resp_with_tool("read_file", {"path": "x.txt"}),
        _resp_text(""),  # no text, no tool_use -> went_quiet
    ]
    wf = _gated_wf(
        repo, provider, _dirtying_dispatcher(repo), max_iterations=5, went_quiet_max_nudges=0
    )
    result = wf.run("do the thing")
    assert result.reason == "went_quiet"
    assert "checkpoint (iter" in _git_log(repo)


def test_unexecutable_verify_abort_checkpoints_the_dirty_worktree(tmp_path: Path) -> None:
    """The worst sibling: the operator's verify command cannot execute in the
    jail, so verify can NEVER go green and the per-turn auto-commit never
    fires -- ALL of the run's edits existed only in the worktree at the
    abort."""
    from agent6.tools.dispatch import OperatorCommandUnexecutable

    repo = tmp_path / "repo"
    _init_repo(repo)
    provider = MagicMock()
    provider.call.side_effect = [
        _resp_with_tool("read_file", {"path": "x.txt"}),
        _resp_with_tool("run_verify_command", {}, tu_id="tu2"),
        _resp_text("never reached"),
    ]
    dispatcher = MagicMock(operator_wait_s=0.0)

    def dispatch(name: str, *_a: Any, **_k: Any) -> RawResult:
        if name == "run_verify_command":
            raise OperatorCommandUnexecutable("verify binary missing from the jail PATH")
        (repo / "edit.txt").write_text("run_command wrote this\n")
        return RawResult({"content": "hi\n"})

    dispatcher.dispatch.side_effect = dispatch
    wf = _gated_wf(repo, provider, dispatcher, max_iterations=5)
    result = wf.run("do the thing")
    assert result.reason == "verify_command_unexecutable"
    assert "checkpoint (iter" in _git_log(repo)


def _stagnation_blocks(messages: list[dict[str, Any]]) -> list[str]:
    """Every [stagnation] block injected into user turns."""
    out: list[str] = []
    for msg in messages:
        if msg.get("role") != "user":
            continue
        content = msg.get("content", [])
        if not isinstance(content, list):
            continue
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                text = block.get("text", "")
                if text.startswith("[stagnation]"):
                    out.append(text)
    return out


def _final_messages(provider: MagicMock) -> list[dict[str, Any]]:
    last_args = provider.call.call_args_list[-1]
    return last_args.kwargs.get("messages") or last_args.args[1]


def test_stagnation_notice_fires_once_without_attempts(tmp_path: Path) -> None:
    """Wall clock past the threshold with zero edit and zero verify calls
    injects the notice exactly once, however many turns follow. Recall
    spirals make 3-10 total calls with long reasoning between them, so the
    identical-signature guard structurally never sees them (P1: 5 of 6
    spiral empties ended by timeout, not guard)."""
    repo = tmp_path / "repo"
    _init_repo(repo)
    provider = MagicMock()
    provider.call.side_effect = [
        _resp_with_tool("read_file", {"path": "a.txt"}, tu_id="t1"),
        _resp_with_tool("read_file", {"path": "b.txt"}, tu_id="t2"),
        # An attemptless prose end gets one silent-no-work nudge round first.
        _resp_text("ok"),
        _resp_text("ok"),
    ]
    dispatcher = MagicMock(operator_wait_s=0.0)
    dispatcher.dispatch.return_value = RawResult({"content": "x"})
    wf = _build_wf(repo, provider, dispatcher)
    wf.stagnation_notice_after_s = 1e-9
    result = wf.run("investigate")
    assert result.completed is True
    notices = _stagnation_blocks(_final_messages(provider))
    assert len(notices) == 1, notices
    # This harness configures no verify command, so the notice names no gate:
    # sending a gateless run after `run_verify_command` names a tool the same
    # run's prompt says it does not have.
    assert "nothing edited yet" in notices[0]
    assert "verify" not in notices[0]

    # A gate the POLICY withholds is not a gate either: `run_commands = "no"`
    # takes run_verify_command away, and the same run's prompt says so.
    denied = MagicMock()
    denied.call.side_effect = itertools.chain(
        [_resp_with_tool("read_file", {"path": "x.txt"}, tu_id="d1")],
        itertools.repeat(_resp_text("ok")),
    )
    no_commands = MagicMock(operator_wait_s=0.0)
    no_commands.dispatch.return_value = RawResult({"content": "x"})
    no_commands.command_policy.return_value = "no"
    wf3 = _build_wf(repo, denied, no_commands)
    wf3.config.workflow.verify_command = ("true",)
    wf3.stagnation_notice_after_s = 1e-9
    wf3.run("investigate")
    assert "nothing edited yet" in _stagnation_blocks(_final_messages(denied))[0]

    # With a gate the run can actually reach, the same notice names it.
    gated = MagicMock()
    gated.call.side_effect = itertools.chain(
        [_resp_with_tool("read_file", {"path": "x.txt"}, tu_id="g1")],
        itertools.repeat(_resp_text("ok")),
    )
    wf2 = _build_wf(repo, gated, dispatcher)
    wf2.config.workflow.verify_command = ("true",)
    wf2.stagnation_notice_after_s = 1e-9
    wf2.run("investigate")
    assert "no edit and no verify" in _stagnation_blocks(_final_messages(gated))[0]


def test_stagnation_ignores_time_blocked_on_the_operator(tmp_path: Path) -> None:
    """The stagnation clock is the model's own time: an hour spent waiting on
    an approval never reads as an hour of research."""
    repo = tmp_path / "repo"
    _init_repo(repo)
    provider = MagicMock()
    provider.call.side_effect = [
        _resp_with_tool("read_file", {"path": "a.txt"}, tu_id="t1"),
        _resp_with_tool("read_file", {"path": "b.txt"}, tu_id="t2"),
        _resp_text("ok"),
        _resp_text("ok"),
    ]
    dispatcher = MagicMock(operator_wait_s=3600.0)
    dispatcher.dispatch.return_value = RawResult({"content": "x"})
    wf = _build_wf(repo, provider, dispatcher)
    wf.stagnation_notice_after_s = 1e-9
    result = wf.run("investigate")
    assert result.completed is True
    assert _stagnation_blocks(_final_messages(provider)) == []


def test_stagnation_notice_suppressed_by_an_edit(tmp_path: Path) -> None:
    """An edit attempt before the threshold crossing means no notice: the
    guard targets attemptless runs only."""
    repo = tmp_path / "repo"
    _init_repo(repo)
    provider = MagicMock()
    provider.call.side_effect = [
        _resp_with_tool("apply_edit", {"path": "x.txt", "edits": []}, tu_id="t1"),
        _resp_with_tool("read_file", {"path": "x.txt"}, tu_id="t2"),
        _resp_text("ok"),
    ]
    dispatcher = MagicMock(operator_wait_s=0.0)
    dispatcher.dispatch.return_value = RawResult({"content": "x"})
    wf = _build_wf(repo, provider, dispatcher)
    wf.stagnation_notice_after_s = 1e-9
    result = wf.run("fix it")
    assert result.completed is True
    assert _stagnation_blocks(_final_messages(provider)) == []


def test_stagnation_notice_zero_disables(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _init_repo(repo)
    provider = MagicMock()
    provider.call.side_effect = [
        _resp_with_tool("read_file", {"path": "x.txt"}, tu_id="t1"),
        _resp_text("ok"),
        _resp_text("ok"),
        _resp_text("ok"),
    ]
    dispatcher = MagicMock(operator_wait_s=0.0)
    dispatcher.dispatch.return_value = RawResult({"content": "x"})
    wf = _build_wf(repo, provider, dispatcher)
    wf.stagnation_notice_after_s = 0.0
    result = wf.run("look around")
    assert result.completed is True
    assert _stagnation_blocks(_final_messages(provider)) == []


def test_unlimited_iterations_is_minus_one(tmp_path: Path) -> None:
    """[workflow].max_iterations = -1 runs unbounded. The pre-knob loop fed -1
    into range(start, 0), which is EMPTY: the run exited max_iterations at
    zero iterations without a single provider call."""
    repo = tmp_path / "repo"
    _init_repo(repo)
    provider = MagicMock()
    provider.call.side_effect = [
        *(_resp_with_tool("read_file", {"path": f"x{i}.txt"}, tu_id=f"t{i}") for i in range(12)),
        _resp_text("ok"),
    ]
    dispatcher = MagicMock(operator_wait_s=0.0)
    dispatcher.dispatch.return_value = RawResult({"content": "hi\n"})

    wf = _build_wf(repo, provider, dispatcher)
    wf.max_iterations = -1
    result = wf.run("read the files")

    assert result.completed is True
    assert provider.call.call_count == 13


def test_resume_leg_rearms_the_iteration_allowance(tmp_path: Path) -> None:
    """A resumed leg gets a fresh max_iterations window relative to its own
    start. The counter used to be absolute: a run capped at max_iterations
    resumed into range(cap+1, cap+1) and made ZERO provider calls (observed
    live: a standing run's resume leg ended instantly)."""
    repo = tmp_path / "repo"
    _init_repo(repo)
    provider = MagicMock()
    provider.call.side_effect = [_resp_text("done")]
    dispatcher = MagicMock(operator_wait_s=0.0)
    dispatcher.dispatch.return_value = RawResult({"content": "hi\n"})

    wf = _build_wf(repo, provider, dispatcher)
    wf.max_iterations = 5
    wf.resume_state_path = tmp_path / "loop_state.json"
    from agent6.workflows.loop import LoopState

    wf._save_resume_snapshot(  # pyright: ignore[reportPrivateUsage]
        system="s",
        messages=[],
        tool_calls=0,
        next_iteration=6,
        root_task_id=None,
        state=LoopState(original_task="t", tool_calls=0),
    )
    result = wf.resume()

    assert result.completed is True
    assert provider.call.call_count == 1
