# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""went_quiet nudge-and-retry instead of immediate run termination.

Weak open-weights models (observed live with Kimi K2.6) sometimes emit a
single empty assistant turn (no text, no tool_use) mid-run. The
harness would terminate the run on the first such turn, throwing away
the work done so far and the budget that produced it. injects a
short synthetic user prompt and retries up to ``went_quiet_max_nudges``
times PER STREAK; on any non-empty turn the counter resets.
"""

from __future__ import annotations

import json
import subprocess as _sp
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

from agent6.events import EventSink
from agent6.providers import ProviderResponse
from agent6.tools.results import RawResult
from agent6.workflows.loop import Workflow


def _silent(_msg: str) -> None:
    return None


def _empty_resp() -> ProviderResponse:
    return ProviderResponse(
        text="",
        tool_uses=(),
        stop_reason="end_turn",
        input_tokens=1,
        output_tokens=0,
        cache_read_tokens=0,
        cache_creation_tokens=0,
        raw={"content": []},
    )


def _resp_with_tool(name: str, args: dict[str, Any], tu_id: str = "tu1") -> ProviderResponse:
    return ProviderResponse(
        text="",
        tool_uses=({"id": tu_id, "name": name, "input": args},),
        stop_reason="tool_use",
        input_tokens=1,
        output_tokens=1,
        cache_read_tokens=0,
        cache_creation_tokens=0,
        raw={"content": [{"type": "tool_use", "id": tu_id, "name": name, "input": args}]},
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
        raw={"content": [{"type": "text", "text": text}]},
    )


def _starved_resp() -> ProviderResponse:
    """A reasoning-starvation turn: stop_reason=length, all output spent
    on a thinking block, no text and no tool_use."""
    return ProviderResponse(
        text="",
        tool_uses=(),
        stop_reason="length",
        input_tokens=1,
        output_tokens=32768,
        cache_read_tokens=0,
        cache_creation_tokens=0,
        raw={"content": [{"type": "thinking", "thinking": "x" * 4096}]},
    )


def _init_repo(repo: Path) -> None:
    repo.mkdir()
    _sp.run(["git", "init", "-q", "-b", "main"], cwd=repo, check=True)
    _sp.run(["git", "config", "user.email", "t@example.com"], cwd=repo, check=True)
    _sp.run(["git", "config", "user.name", "t"], cwd=repo, check=True)
    (repo / "x.txt").write_text("hi\n")
    _sp.run(["git", "add", "x.txt"], cwd=repo, check=True)
    _sp.run(["git", "commit", "-q", "-m", "init"], cwd=repo, check=True)


def _build_wf(repo: Path, provider: MagicMock, **kwargs: Any) -> Workflow:
    dispatcher = MagicMock()
    dispatcher.dispatch.return_value = RawResult({"content": "hi\n"})
    return Workflow(
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
        max_iterations=10,
        **kwargs,
    )


def _nudge_blocks(messages: list[dict[str, Any]]) -> list[str]:
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
                if text.startswith("[harness]"):
                    out.append(text)
    return out


def test_went_quiet_nudges_then_succeeds(tmp_path: Path) -> None:
    """Empty turn -> nudge injected -> model recovers and finishes."""
    repo = tmp_path / "repo"
    _init_repo(repo)

    provider = MagicMock()
    provider.call.side_effect = [
        _empty_resp(),
        # early prose finishes on an untouched tree bounce twice off the
        # no-work gate before one is honored
        _resp_text("done"),
        _resp_text("done"),
        _resp_text("done"),
    ]
    wf = _build_wf(repo, provider, went_quiet_max_nudges=2)
    result = wf.run("do something")

    assert result.completed is True
    assert result.reason == "silent_finish"
    assert provider.call.call_count == 4
    last_args = provider.call.call_args_list[-1]
    final_messages: list[dict[str, Any]] = last_args.kwargs.get("messages") or last_args.args[1]
    nudges = _nudge_blocks(final_messages)
    assert any("empty" in n.lower() for n in nudges)


def test_starvation_injects_nudge_without_suppressing_reasoning(tmp_path: Path) -> None:
    """After a reasoning_starvation turn the harness injects the
    starvation-specific nudge but does NOT force reasoning_effort='off'.
    An N=8 K2.6 perf batch showed forcing reasoning off on recovery turns
    hurt win-rate (its big speedups come from reasoning), so the automatic
    loop-level suppression was removed; the nudge text remains."""
    repo = tmp_path / "repo"
    _init_repo(repo)

    provider = MagicMock()
    provider.call.side_effect = [
        _starved_resp(),
        _resp_text("done"),
        _resp_text("done"),
        _resp_text("done"),
    ]
    wf = _build_wf(repo, provider, went_quiet_max_nudges=2)
    result = wf.run("do something")

    assert result.completed is True
    assert provider.call.call_count == 4
    # The loop must never drive reasoning_effort itself anymore.
    efforts = [c.kwargs.get("reasoning_effort") for c in provider.call.call_args_list]
    assert efforts == [None, None, None, None]
    # The starvation-specific nudge is still injected.
    last_args = provider.call.call_args_list[-1]
    final_messages: list[dict[str, Any]] = last_args.kwargs.get("messages") or last_args.args[1]
    nudges = _nudge_blocks(final_messages)
    assert any("stop reasoning" in n.lower() for n in nudges)


def test_went_quiet_drops_empty_assistant_turn(tmp_path: Path) -> None:
    """The empty assistant turn must be popped before the nudge so
    Anthropic doesn't reject the next call."""
    repo = tmp_path / "repo"
    _init_repo(repo)

    provider = MagicMock()
    provider.call.side_effect = [
        _empty_resp(),
        _resp_text("done"),
        _resp_text("done"),
        _resp_text("done"),
    ]
    wf = _build_wf(repo, provider, went_quiet_max_nudges=1)
    wf.run("task")

    last_args = provider.call.call_args_list[-1]
    final_messages: list[dict[str, Any]] = last_args.kwargs.get("messages") or last_args.args[1]
    # No assistant message in the final transcript should have empty
    # content (we popped the empty one before injecting the nudge).
    for msg in final_messages:
        if msg.get("role") == "assistant":
            assert msg.get("content"), f"empty assistant turn leaked: {msg}"


def test_went_quiet_exhausts_nudges_then_fails(tmp_path: Path) -> None:
    """After `went_quiet_max_nudges` consecutive empty turns, give up."""
    repo = tmp_path / "repo"
    _init_repo(repo)

    provider = MagicMock()
    # 3 empty turns; with max_nudges=2 the third gives up.
    provider.call.side_effect = [_empty_resp(), _empty_resp(), _empty_resp()]
    wf = _build_wf(repo, provider, went_quiet_max_nudges=2)
    result = wf.run("task")

    assert result.completed is False
    assert result.reason == "went_quiet"
    # 1 original + 2 retries = 3 calls.
    assert provider.call.call_count == 3


def test_went_quiet_disabled_when_max_nudges_zero(tmp_path: Path) -> None:
    """`went_quiet_max_nudges = 0` restores the fail-fast behaviour."""
    repo = tmp_path / "repo"
    _init_repo(repo)

    provider = MagicMock()
    provider.call.side_effect = [_empty_resp(), _resp_text("never reached")]
    wf = _build_wf(repo, provider, went_quiet_max_nudges=0)
    result = wf.run("task")

    assert result.completed is False
    assert result.reason == "went_quiet"
    assert provider.call.call_count == 1


def test_went_quiet_nudges_reset_after_successful_turn(tmp_path: Path) -> None:
    """After a non-empty turn the nudge counter refills so a later
    streak of empties also gets the full nudge budget."""
    repo = tmp_path / "repo"
    _init_repo(repo)

    provider = MagicMock()
    # empty, tool_use (resets), empty, empty (uses budget), text.
    provider.call.side_effect = [
        _empty_resp(),
        _resp_with_tool("read_file", {"path": "x.txt"}, tu_id="t1"),
        _empty_resp(),
        _empty_resp(),
        _resp_text("done"),
    ]
    wf = _build_wf(repo, provider, went_quiet_max_nudges=2)
    result = wf.run("task")

    assert result.completed is True
    last_args = provider.call.call_args_list[-1]
    final_messages: list[dict[str, Any]] = last_args.kwargs.get("messages") or last_args.args[1]
    # Three nudges total: one for the first empty, two for the second streak.
    assert len(_nudge_blocks(final_messages)) == 3


def test_went_quiet_budget_refills_on_a_bounced_prose_turn(tmp_path: Path) -> None:
    """The refill contract is "reset on any NON-EMPTY turn", not only tool_use
    turns. Interleaving quiet streaks with bounced prose turns (the silent-
    no-work gate) drained one shared budget and ended the run as went_quiet
    although no streak reached the cap. Each bounced prose turn must refill
    the budget for the next streak."""
    repo = tmp_path / "repo"
    _init_repo(repo)

    provider = MagicMock()
    provider.call.side_effect = [
        _resp_text("I think the answer is..."),  # iter 1: prose, bounced (early stall)
        _empty_resp(),  # iter 2: quiet -> nudge 1/1
        _resp_text("Let me explain more..."),  # iter 3: prose, bounced -> REFILL
        _empty_resp(),  # iter 4: quiet -> nudge 1/1 again (old code: budget dead -> went_quiet)
        _resp_with_tool("read_file", {"path": "x.txt"}, tu_id="t1"),  # iter 5
        _resp_text("done"),  # iter 6: legitimate silent finish
    ]
    wf = _build_wf(repo, provider, went_quiet_max_nudges=1)
    result = wf.run("task")
    assert result.reason != "went_quiet"
    assert result.completed is True


def test_a_billed_empty_turn_says_so(tmp_path: Path) -> None:
    """An empty turn the provider still charged output tokens for is not a
    model that chose silence (reasoning that never surfaced, or a tool call
    the upstream dropped): the log line and the nudge event carry the count,
    so the transcript file is not the only place the difference shows."""
    repo = tmp_path / "repo"
    _init_repo(repo)
    billed = ProviderResponse(
        text="\n\n",
        tool_uses=(),
        stop_reason="stop",
        input_tokens=1,
        output_tokens=128,
        cache_read_tokens=0,
        cache_creation_tokens=0,
        raw={"content": []},
    )
    provider = MagicMock()
    provider.call.side_effect = [billed, _resp_text("done"), _resp_text("done"), _resp_text("done")]
    lines: list[str] = []
    wf = _build_wf(repo, provider, went_quiet_max_nudges=2)
    wf.logger = lines.append
    wf.events = EventSink(tmp_path / "logs.jsonl")
    wf.run("do something")

    quiet = [ln for ln in lines if "went_quiet at iter" in ln]
    # Non-subscription run: dollar-metered, so the word is "billed".
    assert quiet and "128 output tokens billed on it" in quiet[0], quiet
    events = [json.loads(ln) for ln in (tmp_path / "logs.jsonl").read_text().splitlines()]
    nudge = next(e for e in events if e["type"] == "loop.went_quiet.nudge")
    assert nudge["output_tokens"] == 128


def test_a_plan_metered_empty_turn_says_spent_not_billed(tmp_path: Path) -> None:
    """On a subscription plan the tokens cost $0; the went-quiet line must not
    claim they were "billed" (a dollar word). Same count, honest verb."""
    from agent6.budget import BudgetTracker, PlanUsage

    repo = tmp_path / "repo"
    _init_repo(repo)
    billed = ProviderResponse(
        text="\n\n",
        tool_uses=(),
        stop_reason="stop",
        input_tokens=1,
        output_tokens=128,
        cache_read_tokens=0,
        cache_creation_tokens=0,
        raw={"content": []},
    )
    provider = MagicMock()
    provider.call.side_effect = [billed, _resp_text("done"), _resp_text("done"), _resp_text("done")]
    lines: list[str] = []
    wf = _build_wf(repo, provider, went_quiet_max_nudges=2)
    wf.logger = lines.append
    wf.events = EventSink(tmp_path / "logs.jsonl")
    wf.budget = BudgetTracker(max_usd=-1, max_tokens_fallback=-1, max_percent=-1)
    wf.budget.record(
        model="gpt-5.6-sol",
        input_tokens=1,
        output_tokens=1,
        cache_read_tokens=0,
        cache_creation_tokens=0,
        plan_usage=PlanUsage.single(used_percent=2.0, window_minutes=10080, resets_at=2e9),
    )
    wf.run("do something")

    quiet = [ln for ln in lines if "went_quiet at iter" in ln]
    assert quiet and "128 output tokens spent on it" in quiet[0], quiet
    assert "billed" not in quiet[0]


def test_unrunnable_signature_names_only_the_adopted_runner() -> None:
    """Exit 127 and the adopted `-m` module missing are the unrunnable
    signatures; a different missing module or an ordinary red (exit 1 with
    test output) is not."""
    from agent6.workflows._nudges import unrunnable_signature

    argv = ("python3", "-m", "pytest", "-q")
    assert unrunnable_signature(argv, 127, "", "") == "exit 127, the command is not found"
    assert unrunnable_signature(argv, 1, "", "No module named pytest") == "no module named pytest"
    assert unrunnable_signature(argv, 1, "", "No module named numpy") == ""
    assert unrunnable_signature(argv, 1, "3 failed, 2 passed", "") == ""
    assert unrunnable_signature(("pytest",), 1, "", "No module named pytest") == ""
