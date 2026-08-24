# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Unit tests for the Workflow loop: provider retry, operator steering, the
tool-error ladder, finish gates, and the other drive-loop mechanics, driven
directly with scripted providers and dispatchers. Termination-reason
distinctions are exercised end-to-end in the integration suite."""

from __future__ import annotations

import subprocess
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal
from unittest.mock import MagicMock, patch

import pytest

from agent6.providers import ProviderError, ProviderResponse
from agent6.tools.mcp_client import MCPToolDescriptor
from agent6.tools.results import ExecResult, MetricResult, RawResult, ToolResult
from agent6.workflows._conversation import AssistantTurn, Conversation, Notice
from agent6.workflows._verify_verdict import VerifyVerdict
from agent6.workflows.loop import Workflow

# The `[git]` surface the loop reads: the checkpoint message and the commit
# identity (`_commit_identity`), which the real Config carries as empty
# strings when the operator sets neither.
_GIT_STUB = SimpleNamespace(
    control="agent6",
    commit=SimpleNamespace(
        checkpoint=SimpleNamespace(message="agent6"), name="", email="", trailer=""
    ),
)


class _StubDispatcher:
    """The dispatcher surface the loop reads besides `dispatch`.

    The loop rebuilds its tool list every turn (a gate adopted mid-run, or a
    policy denied mid-run, changes what the worker has), so a stub that answers
    only `dispatch` no longer models the real thing. The defaults here keep the
    behaviour these tests were written against: no filtered tools -- the
    provider stubs ignore the list -- and a policy that withholds nothing.
    """

    def available_tool_names(self) -> tuple[str, ...]:
        return ()

    def mcp_descriptors(self) -> tuple[MCPToolDescriptor, ...]:
        """No MCP servers, so nothing to add to the per-turn tool list."""
        return ()

    def skills_available(self) -> bool:
        return False

    def command_policy(self) -> str:
        return "ask"

    def settle_background(self) -> None:
        """The turn boundary observes background commands; these tests start
        none, so there is nothing to write down."""


def _silent(_msg: str) -> None:
    return None


def _wf(**kw: Any) -> Workflow:
    """Construct a Workflow with mocks for everything not under test.

    Caller-supplied kwargs win over the defaults so a test can pass its
    own provider / steer callables without colliding on the keyword.
    """
    defaults: dict[str, Any] = {
        "root": Path("/tmp"),
        "config": MagicMock(
            git=_GIT_STUB,
            budget=SimpleNamespace(max_usd=10.0, max_tokens_fallback=2_000_000),
            prompt=MagicMock(system_prompt_file=""),
            workflow=MagicMock(
                verify_command=(), verify_when="never", verify_retries=2, standing_patience=-1
            ),
        ),
        "provider": MagicMock(),
        "dispatcher": MagicMock(),
        "logger": _silent,
        # A live chain so the auto-commit paths run; tests patch loop.chain_commit.
        "chain_ref": "refs/agent6/test",
        "provider_retry_delay_s": 0.01,  # keep tests fast
    }
    defaults.update(kw)
    if "chain_fallback_parent" not in kw and "root" in kw:
        # Mirror run.py's wiring: the chain's first parent is HEAD at start.
        head = subprocess.run(
            ["git", "-C", str(kw["root"]), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        ).stdout.strip()
        defaults.setdefault("chain_fallback_parent", head or None)
    return Workflow(**defaults)


def _state(**kw: Any) -> Any:
    """Minimal _LoopState for _save_resume_snapshot call sites."""
    from agent6.workflows.loop import _LoopState  # pyright: ignore[reportPrivateUsage]

    defaults: dict[str, Any] = {"original_task": "t", "tool_calls": 0}
    defaults.update(kw)
    return _LoopState(**defaults)


_T0 = datetime(2026, 1, 1, tzinfo=UTC)


def _tn(node_id: str, **fields: Any) -> Any:
    """A TaskNode test fixture from a partial dict of node fields. Uses
    model_construct so short readable ids ("a", "b") survive (id-sort =
    creation order, which first_ready_subtask relies on)."""
    from agent6.graph.models import TaskNode

    base: dict[str, Any] = {
        "id": node_id,
        "parent_id": None,
        "title": "t",
        "rationale": "",
        "acceptance": "",
        "relevant_paths": (),
        "depends_on": (),
        "children": (),
        "status": "pending",
        "created_at": _T0,
        "updated_at": _T0,
        "created_by": "planner",
        "commit_sha": "",
        "notes": "",
    }
    base.update(fields)
    return TaskNode.model_construct(**base)


def _typed(nodes: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Convert the readable dict-of-dicts node literals the tests author into the
    typed dict[str, TaskNode] the curator now hands consumers."""
    return {nid: _tn(nid, **d) for nid, d in nodes.items()}


def test_ask_silent_finish_ends_as_answered_not_silent_finish() -> None:
    # In ask mode a prose answer with no tool call is the normal success: it must
    # end as "answered", not the failure-sounding "silent_finish".
    wf = _wf(mode="ask")
    result = wf._handle_silent_finish(  # pyright: ignore[reportPrivateUsage]
        "The answer is 42.", Conversation(), _state(), iteration=2
    )
    assert result is not None
    assert result.reason == "answered"
    assert result.completed is True
    assert result.summary == "The answer is 42."  # ask keeps the whole answer


def test_run_silent_finish_stays_silent_finish() -> None:
    # In run mode (engaged: edited + verified), a no-tool prose turn is still an
    # implicit silent_finish, not "answered".
    wf = _wf(mode="run")
    result = wf._handle_silent_finish(  # pyright: ignore[reportPrivateUsage]
        "Done.",
        Conversation(),
        _state(ever_edited=True, verify=VerifyVerdict(ever_passed=True)),
        iteration=5,
    )
    assert result is not None
    assert result.reason == "silent_finish"


class _EventCapture:
    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []

    def emit(self, event_type: str, /, **fields: Any) -> None:
        self.events.append({"type": event_type, **fields})


def _cfg_with_verify() -> Any:
    return MagicMock(
        prompt=MagicMock(system_prompt_file=""),
        workflow=MagicMock(
            verify_command=("true",),
            verify_when="never",
            verify_retries=2,
            metric=SimpleNamespace(goal=None),
        ),
    )


def test_run_silent_finish_over_red_verify_is_not_passed() -> None:
    """A run-mode silent finish (prose, no tool_use) over a RED or stale verify
    must emit session.end all_passed=False — the same honest-finish rule as the
    explicit finish_session path, so no surface renders the failed run as 'passed'."""
    ev = _EventCapture()
    wf = _wf(mode="run", config=_cfg_with_verify(), events=ev)
    result = wf._handle_silent_finish(  # pyright: ignore[reportPrivateUsage]
        "I tried, but the tests still fail.",
        Conversation(),
        _state(ever_edited=True, verify=VerifyVerdict(ever_passed=True, last_ok=False)),
        iteration=5,
    )
    assert result is not None and result.reason == "silent_finish"
    ends = [e for e in ev.events if e["type"] == "session.end"]
    assert ends and ends[-1]["all_passed"] is False


def test_run_silent_finish_over_green_verify_stays_passed() -> None:
    """The mirror: a clean green tree still ends passed (no false negative)."""
    ev = _EventCapture()
    wf = _wf(mode="run", config=_cfg_with_verify(), events=ev)
    wf._handle_silent_finish(  # pyright: ignore[reportPrivateUsage]
        "Done, all green.",
        Conversation(),
        _state(
            ever_edited=True,
            verify=VerifyVerdict(ever_passed=True, last_ok=True, edited_since=False),
        ),
        iteration=5,
    )
    ends = [e for e in ev.events if e["type"] == "session.end"]
    assert ends and ends[-1]["all_passed"] is True


def test_run_silent_finish_gateless_is_ungated_not_passed() -> None:
    """A GATELESS run's silent finish emitted all_passed=True and every surface
    read "passed" for a tree nothing verified. The end carries the ungated
    tri-state (all_passed None), which words as "finished"."""
    from agent6.viewmodel.listing import status_word

    ev = _EventCapture()
    wf = _wf(mode="run", events=ev)  # _wf's default config has no verify_command
    result = wf._handle_silent_finish(  # pyright: ignore[reportPrivateUsage]
        "READY",
        Conversation(),
        _state(ever_edited=True),
        iteration=5,
    )
    assert result is not None and result.reason == "silent_finish"
    ends = [e for e in ev.events if e["type"] == "session.end"]
    assert ends and ends[-1]["all_passed"] is None
    assert status_word(finished=True, all_passed=None, end_reason="silent_finish") == (
        "finished",
        "",
    )


def _turn(**kw: Any) -> Any:
    """A bare _TurnState for direct turn-phase method tests."""
    from agent6.workflows.loop import _TurnState  # pyright: ignore[reportPrivateUsage]

    defaults: dict[str, Any] = {
        "iteration": 1,
        "resp": _resp(""),
        "assistant": AssistantTurn((), ()),
    }
    defaults.update(kw)
    return _TurnState(**defaults)


def test_finish_planning_salvages_a_title_only_plan(tmp_path: Path) -> None:
    # Weak models leave plan_markdown a bare title and put the plan in `summary`;
    # the fold must produce a plan.md with real content, not a title-only stub.
    plan_path = tmp_path / "plan.md"
    wf = _wf(mode="plan", plan_output_path=plan_path)
    wf._capture_finish(  # pyright: ignore[reportPrivateUsage]
        _turn(),
        "finish_planning",
        {
            "summary": "1. Add the --count flag. 2. Update the parser help. 3. Add a test.",
            "plan_markdown": "# Plan: Add --count flag",
        },
    )
    text = plan_path.read_text(encoding="utf-8")
    assert "# Plan: Add --count flag" in text  # the title is kept
    assert "Add the --count flag" in text  # the summary was folded in as the body


def test_finish_planning_keeps_a_real_plan_markdown(tmp_path: Path) -> None:
    # A proper plan_markdown is written verbatim; the summary is NOT appended.
    plan_path = tmp_path / "plan.md"
    wf = _wf(mode="plan", plan_output_path=plan_path)
    wf._capture_finish(  # pyright: ignore[reportPrivateUsage]
        _turn(),
        "finish_planning",
        {"summary": "short blurb", "plan_markdown": "# Plan: X\n\n1. real step\n2. another"},
    )
    text = plan_path.read_text(encoding="utf-8")
    assert "real step" in text and "short blurb" not in text


def _resp(text: str = "ok") -> ProviderResponse:
    return ProviderResponse(
        text=text,
        tool_uses=(),
        stop_reason="end_turn",
        input_tokens=1,
        output_tokens=1,
        cache_read_tokens=0,
        cache_creation_tokens=0,
    )


def _tool_resp(
    name: str,
    tool_input: dict[str, Any] | None = None,
    *,
    tool_id: str = "tool-1",
) -> ProviderResponse:
    payload = tool_input or {}
    block = {"type": "tool_use", "id": tool_id, "name": name, "input": payload}
    return ProviderResponse(
        text="",
        tool_uses=({"id": tool_id, "name": name, "input": payload},),
        stop_reason="tool_use",
        input_tokens=1,
        output_tokens=1,
        cache_read_tokens=0,
        cache_creation_tokens=0,
        raw={"content": [block]},
    )


# --- _call_with_retry -----------------------------------------------------


def test_call_with_retry_first_try_returns() -> None:
    """No ProviderError -> single call, returns immediately."""
    provider = MagicMock()
    provider.call.return_value = _resp("first")
    wf = _wf(provider=provider)
    out = wf._call_with_retry(system="s", messages=[], tools=[], max_tokens=16384)  # pyright: ignore[reportPrivateUsage]
    assert out.text == "first"
    assert provider.call.call_count == 1


def test_call_with_retry_succeeds_on_retry() -> None:
    """ProviderError on first call, success on retry -> returns the retry."""
    provider = MagicMock()
    provider.call.side_effect = [ProviderError("transient 529"), _resp("retried")]
    wf = _wf(provider=provider, provider_retry_count=1)
    out = wf._call_with_retry(system="s", messages=[], tools=[], max_tokens=16384)  # pyright: ignore[reportPrivateUsage]
    assert out.text == "retried"
    assert provider.call.call_count == 2


def test_call_with_retry_reraises_after_retries_exhausted() -> None:
    """Two ProviderErrors with retry_count=1 -> bubble the last error."""
    provider = MagicMock()
    provider.call.side_effect = [ProviderError("flake 1"), ProviderError("flake 2")]
    wf = _wf(provider=provider, provider_retry_count=1)
    with pytest.raises(ProviderError, match="flake 2"):
        wf._call_with_retry(system="s", messages=[], tools=[], max_tokens=16384)  # pyright: ignore[reportPrivateUsage]
    assert provider.call.call_count == 2


def test_call_with_retry_never_retries_an_abort() -> None:
    """ProviderAborted (operator stop) bubbles immediately, never retried."""
    from agent6.providers import ProviderAborted

    provider = MagicMock()
    provider.call.side_effect = [ProviderAborted("stopped"), _resp("late")]
    wf = _wf(provider=provider, provider_retry_count=3)
    with pytest.raises(ProviderAborted):
        wf._call_with_retry(system="s", messages=[], tools=[], max_tokens=16384)  # pyright: ignore[reportPrivateUsage]
    assert provider.call.call_count == 1  # not retried
    # and should_abort is threaded to the provider
    assert provider.call.call_args.kwargs["should_abort"] is wf.should_abort


def test_call_with_retry_never_retries_a_steer_interrupt() -> None:
    """ProviderInterrupted (steer mid-stream) bubbles immediately -- the loop shows
    the steer menu; retrying would just re-hit the interrupt."""
    from agent6.providers import ProviderInterrupted

    provider = MagicMock()
    provider.call.side_effect = [ProviderInterrupted("steer"), _resp("late")]
    wf = _wf(provider=provider, provider_retry_count=3)
    with pytest.raises(ProviderInterrupted):
        wf._call_with_retry(system="s", messages=[], tools=[], max_tokens=16384)  # pyright: ignore[reportPrivateUsage]
    assert provider.call.call_count == 1  # not retried
    assert provider.call.call_args.kwargs["should_interrupt"] is wf.should_interrupt


def test_call_with_retry_honors_retry_after(monkeypatch: pytest.MonkeyPatch) -> None:
    """A 429 carrying retry_after_s waits at least that long, not the (shorter)
    self-computed backoff."""
    slept: list[float] = []
    monkeypatch.setattr("agent6.workflows.loop.time.sleep", slept.append)
    provider = MagicMock()
    provider.call.side_effect = [
        ProviderError("429 rate limited", status_code=429, retry_after_s=50.0),
        _resp("ok"),
    ]
    wf = _wf(provider=provider, provider_retry_count=1)  # _wf backoff is 0.01s
    out = wf._call_with_retry(system="s", messages=[], tools=[], max_tokens=16384)  # pyright: ignore[reportPrivateUsage]
    assert out.text == "ok"
    assert slept and slept[0] >= 50.0  # honored the server's window, not ~0.01


def test_call_with_retry_clamps_retry_after_to_ceiling(monkeypatch: pytest.MonkeyPatch) -> None:
    """A hostile/buggy Retry-After can't hang the run: clamp to the ceiling."""
    slept: list[float] = []
    monkeypatch.setattr("agent6.workflows.loop.time.sleep", slept.append)
    provider = MagicMock()
    provider.call.side_effect = [
        ProviderError("429", status_code=429, retry_after_s=9999.0),
        _resp("ok"),
    ]
    wf = _wf(provider=provider, provider_retry_count=1)
    wf._call_with_retry(system="s", messages=[], tools=[], max_tokens=16384)  # pyright: ignore[reportPrivateUsage]
    assert slept and slept[0] <= 120.0  # _RETRY_AFTER_CEILING_S


def _empty_tool_call_resp() -> ProviderResponse:
    """A self-contradictory response: stop_reason=tool_calls but no tool_use/text
    (the GLM-via-OpenRouter post-restart flake)."""
    return ProviderResponse(
        text="",
        tool_uses=(),
        stop_reason="tool_calls",
        input_tokens=1,
        output_tokens=20,
        cache_read_tokens=0,
        cache_creation_tokens=0,
    )


def test_call_with_retry_retries_empty_tool_call_response() -> None:
    """An empty finish=tool_calls response (no tool_use, no text) is retried; the
    recovered real response is returned."""
    provider = MagicMock()
    provider.call.side_effect = [_empty_tool_call_resp(), _tool_resp("read_file", {"path": "x"})]
    wf = _wf(provider=provider, provider_retry_count=4, provider_retry_delay_s=0.001)
    out = wf._call_with_retry(system="s", messages=[], tools=[], max_tokens=16384)  # pyright: ignore[reportPrivateUsage]
    assert out.tool_uses  # recovered to a real tool call
    assert provider.call.call_count == 2


def test_call_with_retry_returns_last_empty_after_exhausting() -> None:
    """If every attempt is empty, return the last empty response (the loop's
    went_quiet handler takes over) -- never raise / assert-fail."""
    provider = MagicMock()
    provider.call.return_value = _empty_tool_call_resp()
    wf = _wf(provider=provider, provider_retry_count=2, provider_retry_delay_s=0.001)
    out = wf._call_with_retry(system="s", messages=[], tools=[], max_tokens=16384)  # pyright: ignore[reportPrivateUsage]
    assert out.stop_reason == "tool_calls" and not out.tool_uses
    assert provider.call.call_count == 3  # 1 initial + 2 retries


def test_is_empty_tool_call_response_discriminates() -> None:
    from agent6.workflows.loop import (
        is_empty_tool_call_response,  # pyright: ignore[reportPrivateUsage]
    )

    assert is_empty_tool_call_response(_empty_tool_call_resp())
    assert not is_empty_tool_call_response(_resp("hi"))  # has text -> a silent finish
    assert not is_empty_tool_call_response(_tool_resp("read_file"))  # has a tool_use
    # length-truncated reasoning starvation is handled separately, not retried here.
    starved = ProviderResponse(
        text="",
        tool_uses=(),
        stop_reason="length",
        input_tokens=1,
        output_tokens=20,
        cache_read_tokens=0,
        cache_creation_tokens=0,
    )
    assert not is_empty_tool_call_response(starved)


def test_call_with_retry_default_rides_out_multiple_flaps() -> None:
    """The default retry budget survives more than one consecutive transient
    disconnect. Regression: a single retry (the old default) aborted long,
    expensive runs on a multi-second Anthropic 'Server disconnected' flap."""
    provider = MagicMock()
    disconnect = ProviderError("Server disconnected without sending a response")
    provider.call.side_effect = [disconnect, disconnect, disconnect, _resp("recovered")]
    wf = _wf(provider=provider)  # uses the default provider_retry_count
    out = wf._call_with_retry(system="s", messages=[], tools=[], max_tokens=16384)  # pyright: ignore[reportPrivateUsage]
    assert out.text == "recovered"
    assert provider.call.call_count == 4


def test_call_with_retry_zero_retries_no_retry() -> None:
    """provider_retry_count=0 -> single attempt, no retry on error."""
    provider = MagicMock()
    provider.call.side_effect = [ProviderError("nope")]
    wf = _wf(provider=provider, provider_retry_count=0)
    with pytest.raises(ProviderError, match="nope"):
        wf._call_with_retry(system="s", messages=[], tools=[], max_tokens=16384)  # pyright: ignore[reportPrivateUsage]
    assert provider.call.call_count == 1


def test_call_with_retry_does_not_swallow_non_provider_errors() -> None:
    """RuntimeError (etc.) must propagate without retry."""
    provider = MagicMock()
    provider.call.side_effect = [RuntimeError("not a provider error")]
    wf = _wf(provider=provider, provider_retry_count=3)
    with pytest.raises(RuntimeError, match="not a provider error"):
        wf._call_with_retry(system="s", messages=[], tools=[], max_tokens=16384)  # pyright: ignore[reportPrivateUsage]
    assert provider.call.call_count == 1


def test_call_with_retry_skips_retry_on_permanent_status() -> None:
    """A permanent client error (402 insufficient credits) re-raises on the
    first failure without consuming a retry. Observed live: a 402 was
    otherwise retried on every remaining turn, burning wall-time."""
    provider = MagicMock()
    provider.call.side_effect = [
        ProviderError("OpenAI API error 402: Insufficient credits", status_code=402),
        _resp("should-never-be-reached"),
    ]
    wf = _wf(provider=provider, provider_retry_count=3)
    with pytest.raises(ProviderError, match="402"):
        wf._call_with_retry(system="s", messages=[], tools=[], max_tokens=16384)  # pyright: ignore[reportPrivateUsage]
    assert provider.call.call_count == 1


@pytest.mark.parametrize("status", [400, 401, 402, 403, 404, 422])
def test_call_with_retry_skips_retry_on_all_permanent_statuses(status: int) -> None:
    """Every status in _NON_RETRYABLE_HTTP_STATUSES re-raises on the first
    failure without consuming a retry (not just the 402 observed live)."""
    provider = MagicMock()
    provider.call.side_effect = [
        ProviderError(f"provider error {status}", status_code=status),
        _resp("should-never-be-reached"),
    ]
    wf = _wf(provider=provider, provider_retry_count=3)
    with pytest.raises(ProviderError, match=str(status)):
        wf._call_with_retry(system="s", messages=[], tools=[], max_tokens=16384)  # pyright: ignore[reportPrivateUsage]
    assert provider.call.call_count == 1


def test_call_with_retry_still_retries_transient_5xx() -> None:
    """A 503 carries a status_code but is NOT in the permanent set, so the
    normal single-retry path still applies."""
    provider = MagicMock()
    provider.call.side_effect = [
        ProviderError("OpenAI API error 503: upstream", status_code=503),
        _resp("recovered"),
    ]
    wf = _wf(provider=provider, provider_retry_count=1)
    out = wf._call_with_retry(system="s", messages=[], tools=[], max_tokens=16384)  # pyright: ignore[reportPrivateUsage]
    assert out.text == "recovered"
    assert provider.call.call_count == 2


# --- exponential backoff with jitter -------------------------------------


def test_call_with_retry_exponential_backoff() -> None:
    """Retry delays grow exponentially: attempt N sleeps
    provider_retry_delay_s * 2 ** (attempt - 1), scaled by the jitter factor."""
    provider = MagicMock()
    provider.call.side_effect = [
        ProviderError("flake 1"),
        ProviderError("flake 2"),
        ProviderError("flake 3"),
        _resp("success"),
    ]
    wf = _wf(
        provider=provider,
        provider_retry_count=3,
        provider_retry_delay_s=2.0,
        provider_retry_max_delay_s=30.0,
    )
    sleep_calls: list[float] = []
    with (
        patch("time.sleep", side_effect=sleep_calls.append),
        patch("random.uniform", return_value=0.75),
    ):
        out = wf._call_with_retry(system="s", messages=[], tools=[], max_tokens=16384)  # pyright: ignore[reportPrivateUsage]
    assert out.text == "success"
    assert provider.call.call_count == 4
    assert sleep_calls[0] == pytest.approx(1.5)  # 2.0 * 2**0 * 0.75
    assert sleep_calls[1] == pytest.approx(3.0)  # 2.0 * 2**1 * 0.75
    assert sleep_calls[2] == pytest.approx(6.0)  # 2.0 * 2**2 * 0.75


def test_call_with_retry_backoff_capped_at_max_delay() -> None:
    """Exponential backoff is capped at provider_retry_max_delay_s."""
    provider = MagicMock()
    provider.call.side_effect = [
        ProviderError("flake 1"),
        ProviderError("flake 2"),
        ProviderError("flake 3"),
        ProviderError("flake 4"),
        _resp("success"),
    ]
    wf = _wf(
        provider=provider,
        provider_retry_count=4,
        provider_retry_delay_s=2.0,
        provider_retry_max_delay_s=5.0,
    )
    sleep_calls: list[float] = []
    with (
        patch("time.sleep", side_effect=sleep_calls.append),
        patch("random.uniform", return_value=1.0),
    ):
        out = wf._call_with_retry(system="s", messages=[], tools=[], max_tokens=16384)  # pyright: ignore[reportPrivateUsage]
    assert out.text == "success"
    assert provider.call.call_count == 5
    assert sleep_calls[0] == pytest.approx(2.0)  # min(2.0 * 2**0, 5.0)
    assert sleep_calls[1] == pytest.approx(4.0)  # min(2.0 * 2**1, 5.0)
    assert sleep_calls[2] == pytest.approx(5.0)  # min(2.0 * 2**2, 5.0) capped
    assert sleep_calls[3] == pytest.approx(5.0)  # min(2.0 * 2**3, 5.0) capped


def test_call_with_retry_backoff_skips_sleep_on_permanent_status() -> None:
    """A permanent status re-raises immediately with no sleep at all,
    even though provider_retry_count would otherwise allow retries."""
    provider = MagicMock()
    provider.call.side_effect = [
        ProviderError("Insufficient credits", status_code=402),
    ]
    wf = _wf(provider=provider, provider_retry_count=3, provider_retry_delay_s=10.0)
    sleep_calls: list[float] = []
    with (
        patch("time.sleep", side_effect=sleep_calls.append),
        pytest.raises(ProviderError, match="Insufficient credits"),
    ):
        wf._call_with_retry(system="s", messages=[], tools=[], max_tokens=16384)  # pyright: ignore[reportPrivateUsage]
    assert provider.call.call_count == 1
    assert sleep_calls == []


# --- temperature wiring (Amp 2) -----------------------------------


def test_call_with_retry_pins_default_temperature_to_zero() -> None:
    """Default Workflow.temperature is 0.0; every provider.call must
    receive it. agent6 used to pass temperature=None
    so OpenRouter routed to the model's (often high) provider default,
    which produced observable degeneration on Kimi K2.6."""
    provider = MagicMock()
    provider.call.return_value = _resp("ok")
    wf = _wf(provider=provider)
    wf._call_with_retry(system="s", messages=[], tools=[], max_tokens=16384)  # pyright: ignore[reportPrivateUsage]
    assert provider.call.call_args.kwargs["temperature"] == 0.0


def test_call_with_retry_honours_overridden_temperature() -> None:
    """Operators who set `[models.worker].temperature = 0.7` get it
    threaded through verbatim."""
    provider = MagicMock()
    provider.call.return_value = _resp("ok")
    wf = _wf(provider=provider, temperature=0.7)
    wf._call_with_retry(system="s", messages=[], tools=[], max_tokens=16384)  # pyright: ignore[reportPrivateUsage]
    assert provider.call.call_args.kwargs["temperature"] == 0.7


def test_call_with_retry_passes_through_none_temperature() -> None:
    """Explicit `temperature = None` reverts to the previous behaviour
    (let the provider pick), for operators who specifically want it."""
    provider = MagicMock()
    provider.call.return_value = _resp("ok")
    wf = _wf(provider=provider, temperature=None)
    wf._call_with_retry(system="s", messages=[], tools=[], max_tokens=16384)  # pyright: ignore[reportPrivateUsage]
    assert provider.call.call_args.kwargs["temperature"] is None


# --- automatic metric feedback ------------------------------------------


def test_drive_loop_auto_runs_metric_after_verify_pass(tmp_path: Path) -> None:
    """Metric-configured runs should not rely on the worker remembering to
    call run_metric_command. After a green verify, the harness runs it and
    injects a compact history block into the next worker turn.
    """

    class ProviderStub:
        def __init__(self) -> None:
            self.calls: list[list[dict[str, Any]]] = []
            self.saw_metric_feedback = False

        def call(self, **kwargs: Any) -> ProviderResponse:
            messages = kwargs["messages"]
            self.calls.append(messages)
            if len(self.calls) == 1:
                return _tool_resp("run_verify_command")
            rendered = str(messages[-1])
            self.saw_metric_feedback = (
                "[harness metric]" in rendered
                and "score=42" in rendered
                and "first parsed metric sample" in rendered
            )
            return _tool_resp("finish_session", {"summary": "done"}, tool_id="tool-2")

    class DispatcherStub(_StubDispatcher):
        def __init__(self) -> None:
            self.calls: list[str] = []

        def dispatch(self, name: str, raw_input: dict[str, Any]) -> ToolResult:
            self.calls.append(name)
            if name == "run_verify_command":
                return ExecResult(
                    returncode=0, stdout="", stderr="", duration_s=0.1, exec_failed=False
                )
            if name == "run_metric_command":
                return MetricResult(
                    returncode=0,
                    stdout="CYCLES: 42\n",
                    stderr="",
                    duration_s=0.1,
                    exec_failed=False,
                    score=42.0,
                )
            if name == "finish_session":
                return RawResult({"acknowledged": True, "summary": raw_input["summary"]})
            raise AssertionError(f"unexpected tool: {name}")

    provider = ProviderStub()
    dispatcher = DispatcherStub()
    config = SimpleNamespace(
        git=_GIT_STUB,
        budget=SimpleNamespace(max_usd=10.0, max_tokens_fallback=2_000_000),
        workflow=SimpleNamespace(
            verify_when="never",
            verify_retries=2,
            verify_command=("true",),
            metric=SimpleNamespace(goal="minimize"),
        ),
    )
    wf = _wf(
        root=tmp_path,
        config=config,
        provider=provider,
        dispatcher=dispatcher,
        max_iterations=3,
    )
    messages = [{"role": "user", "content": [{"type": "text", "text": "TASK:\noptimize"}]}]

    with patch("agent6.workflows.loop.chain_commit", return_value="abc1234567890"):
        result = wf._drive_loop(  # pyright: ignore[reportPrivateUsage]
            system="system",
            conversation=Conversation.from_wire(messages),
            tool_calls=0,
            start_iteration=1,
            root_task_id=None,
            original_task="t",
        )

    assert result.completed is True
    assert result.reason == "finish_session"
    assert provider.saw_metric_feedback is True
    assert dispatcher.calls == ["run_verify_command", "run_metric_command", "finish_session"]


def test_drive_loop_tracks_iterations_reached(tmp_path: Path) -> None:
    """The loop records the absolute iteration it is driving on the Workflow, so
    the app-level KeyboardInterrupt fallbacks in run/resume can emit a session.end
    carrying a truthful iteration count (matching the loop's own session.end shape).
    Uses a resumed start_iteration to prove it is the absolute number, not a
    zero-based counter."""

    class ProviderStub:
        def __init__(self) -> None:
            self.n = 0

        def call(self, **kwargs: Any) -> ProviderResponse:
            del kwargs
            self.n += 1
            if self.n == 1:
                return _tool_resp("run_verify_command")
            return _tool_resp("finish_session", {"summary": "done"}, tool_id="tool-2")

    class DispatcherStub(_StubDispatcher):
        def dispatch(self, name: str, raw_input: dict[str, Any]) -> ToolResult:
            if name == "run_verify_command":
                return ExecResult(
                    returncode=0, stdout="", stderr="", duration_s=0.1, exec_failed=False
                )
            if name == "finish_session":
                return RawResult({"acknowledged": True, "summary": raw_input["summary"]})
            raise AssertionError(f"unexpected tool: {name}")

    config = SimpleNamespace(
        git=_GIT_STUB,
        budget=SimpleNamespace(max_usd=10.0, max_tokens_fallback=2_000_000),
        workflow=SimpleNamespace(
            verify_when="never",
            verify_retries=2,
            verify_command=("true",),
            metric=SimpleNamespace(goal=None),
        ),
    )
    wf = _wf(
        root=tmp_path,
        config=config,
        provider=ProviderStub(),
        dispatcher=DispatcherStub(),
        max_iterations=20,
    )
    assert wf.iterations_reached == 0  # untouched before the loop runs
    messages = [{"role": "user", "content": [{"type": "text", "text": "TASK:\ngo"}]}]

    with patch("agent6.workflows.loop.chain_commit", return_value="abc1234567890"):
        result = wf._drive_loop(  # pyright: ignore[reportPrivateUsage]
            system="system",
            conversation=Conversation.from_wire(messages),
            tool_calls=0,
            start_iteration=7,  # a resumed run picks up mid-way
            root_task_id=None,
            original_task="t",
        )

    assert result.completed is True
    # verify ran at iter 7, finish_session at iter 8 -> the loop reached iteration 8.
    assert wf.iterations_reached == 8


def test_provider_error_summary_is_concise_not_the_raw_body(tmp_path: Path) -> None:
    """A permanent provider error's raw upstream body (which can carry a noisy
    account user_id) belongs in the ONE diagnostic log line, not echoed again in
    the end-block summary. The summary stays concise (failure + HTTP status)."""
    raw_body = 'OpenRouter API error 400: {"error":"bad model","user_id":"user_SECRET"}'

    class ProviderStub:
        def call(self, **kwargs: Any) -> ProviderResponse:
            del kwargs
            raise ProviderError(raw_body, status_code=400)

    config = SimpleNamespace(
        git=_GIT_STUB,
        budget=SimpleNamespace(max_usd=10.0, max_tokens_fallback=2_000_000),
        workflow=SimpleNamespace(
            verify_when="never",
            verify_retries=2,
            verify_command=("true",),
            metric=SimpleNamespace(goal=None),
        ),
    )
    logs: list[str] = []
    wf = _wf(
        root=tmp_path,
        config=config,
        provider=ProviderStub(),
        dispatcher=MagicMock(),
        logger=logs.append,
        max_iterations=3,
    )
    messages = [{"role": "user", "content": [{"type": "text", "text": "TASK:\nx"}]}]
    result = wf._drive_loop(  # pyright: ignore[reportPrivateUsage]
        system="system",
        conversation=Conversation.from_wire(messages),
        tool_calls=0,
        start_iteration=1,
        root_task_id=None,
        original_task="t",
    )
    assert result.reason == "provider_error"
    assert "provider error" in result.summary and "HTTP 400" in result.summary
    assert "user_SECRET" not in result.summary  # the raw blob is NOT re-echoed here
    assert any("user_SECRET" in line for line in logs)  # kept once, in the log line


class _OneShotSteer:
    """A file-bridge steer stand-in that fires once, returning *text*."""

    def __init__(self, text: str) -> None:
        self.text = text
        self._fired = False

    def requested(self) -> bool:
        return not self._fired

    def prompt(self) -> str:
        return self.text

    def clear(self) -> None:
        self._fired = True


def _resume_snapshot(**kw: Any) -> Any:
    from agent6.workflows._session_state import SessionSnapshot

    defaults: dict[str, Any] = {
        "system": "system",
        "messages": [{"role": "user", "content": [{"type": "text", "text": "TASK:\nmean"}]}],
        "tool_calls": 4,
        "next_iteration": 5,
        "root_task_id": None,
        "original_task": "add a mean() function",
        "verify_command": (),
    }
    defaults.update(kw)
    return SessionSnapshot(**defaults)


def test_resume_seeded_steer_drives_a_finished_run(tmp_path: Path) -> None:
    """`resume --steer` on an already-FINISHED run: the seeded follow-up is
    injected BEFORE the first provider call and drives at least one more
    iteration. Without the fix the resumed conversation silent-finishes on
    iteration 1 (before the end-of-iteration steer poll), dropping the follow-up
    and reporting the original task as done."""

    class ProviderStub:
        def __init__(self) -> None:
            self.calls: list[str] = []

        def call(self, **kwargs: Any) -> ProviderResponse:
            rendered = str(kwargs["messages"])
            self.calls.append(rendered)
            if "median" not in rendered:
                # The buggy path: the follow-up never reached the model, so it
                # re-confirms the finished task in prose (a silent finish).
                return _resp("The mean() function is already done.")
            # The follow-up is present: act on it, then finish.
            if len(self.calls) == 1:
                return _tool_resp("run_command", {"command": "add median"}, tool_id="m1")
            return _tool_resp("finish_session", {"summary": "added median()"}, tool_id="m2")

    class DispatcherStub(_StubDispatcher):
        def dispatch(self, name: str, raw_input: dict[str, Any]) -> ToolResult:
            if name == "finish_session":
                return RawResult({"acknowledged": True, "summary": raw_input["summary"]})
            return RawResult({"content": "ok"})

    steer = _OneShotSteer("add a median() function too")
    provider = ProviderStub()
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
    wf = _wf(
        root=tmp_path,
        config=config,
        provider=provider,
        dispatcher=DispatcherStub(),
        max_iterations=10,
        steer_requested=steer.requested,
        steer_prompt=steer.prompt,
        steer_clear=steer.clear,
    )
    snapshot = _resume_snapshot()

    with patch("agent6.workflows.loop.chain_commit", return_value="abc1234567890"):
        result = wf._drive_loop(  # pyright: ignore[reportPrivateUsage]
            system=snapshot.system,
            conversation=Conversation.from_wire(snapshot.messages),
            tool_calls=snapshot.tool_calls,
            start_iteration=snapshot.next_iteration,
            root_task_id=snapshot.root_task_id,
            original_task=snapshot.original_task,
            resume_from=snapshot,
        )

    # The seeded steer entered the conversation BEFORE the first provider call.
    assert "median" in provider.calls[0]
    # It drove the run to a real finish, not a dropped-steer silent finish.
    assert result.reason == "finish_session"
    assert result.completed is True
    assert len(provider.calls) >= 2  # at least one more iteration than the silent finish


def test_resume_without_steer_does_not_poll_up_front(tmp_path: Path) -> None:
    """The up-front resume steer check is inert when no steer is seeded: a resume
    with no `--steer` puts no phantom OPERATOR STEERING block on the wire."""
    captured: list[str] = []

    class ProviderStub:
        def call(self, **kwargs: Any) -> ProviderResponse:
            captured.append(str(kwargs["messages"]))
            return _resp("done")

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
    # No steer callables: the Workflow's default steer_requested() is False, so the
    # up-front resume check is a no-op -- exactly a resume with no `--steer`.
    wf = _wf(
        root=tmp_path,
        config=config,
        provider=ProviderStub(),
        dispatcher=MagicMock(),
        max_iterations=5,
    )
    snapshot = _resume_snapshot(
        messages=[{"role": "user", "content": [{"type": "text", "text": "TASK:\nengaged"}]}],
        verify_ever_passed=True,
    )

    result = wf._drive_loop(  # pyright: ignore[reportPrivateUsage]
        system=snapshot.system,
        conversation=Conversation.from_wire(snapshot.messages),
        tool_calls=snapshot.tool_calls,
        start_iteration=snapshot.next_iteration,
        root_task_id=snapshot.root_task_id,
        original_task=snapshot.original_task,
        resume_from=snapshot,
    )

    assert result.reason == "silent_finish"  # unchanged behaviour without a seeded steer
    # The named property: nothing was injected ahead of the first resumed call.
    assert captured and "OPERATOR STEERING" not in captured[0]


def test_drive_loop_auto_metric_unexecutable_aborts_gracefully(tmp_path: Path) -> None:
    """An unexecutable metric command must abort the run the SAME graceful way
    whether the model called run_metric_command or the auto-after-verify path
    did. Pins the crash where the auto path's `except ToolError` could not catch
    OperatorCommandUnexecutable (a sibling of ToolError, not a subclass), so the
    misconfiguration escaped as an uncaught traceback out of the whole run."""
    from agent6.tools.dispatch import OperatorCommandUnexecutable

    class ProviderStub:
        # Always pass verify; never call run_metric_command itself, so the AUTO
        # path is what triggers the unexecutable metric command.
        def call(self, **kwargs: Any) -> ProviderResponse:
            del kwargs
            return _tool_resp("run_verify_command")

    class DispatcherStub(_StubDispatcher):
        def __init__(self) -> None:
            self.calls: list[str] = []

        def dispatch(self, name: str, raw_input: dict[str, Any]) -> ToolResult:
            del raw_input
            self.calls.append(name)
            if name == "run_verify_command":
                return ExecResult(
                    returncode=0, stdout="", stderr="", duration_s=0.1, exec_failed=False
                )
            if name == "run_metric_command":
                raise OperatorCommandUnexecutable("metric command '/x/uv' not in jail")
            raise AssertionError(f"unexpected tool: {name}")

    provider = ProviderStub()
    dispatcher = DispatcherStub()
    config = SimpleNamespace(
        git=_GIT_STUB,
        budget=SimpleNamespace(max_usd=10.0, max_tokens_fallback=2_000_000),
        workflow=SimpleNamespace(
            verify_when="never",
            verify_retries=2,
            verify_command=("true",),
            metric=SimpleNamespace(goal="minimize"),
        ),
    )
    wf = _wf(
        root=tmp_path, config=config, provider=provider, dispatcher=dispatcher, max_iterations=5
    )
    messages = [{"role": "user", "content": [{"type": "text", "text": "TASK:\noptimize"}]}]

    with patch("agent6.workflows.loop.chain_commit", return_value="abc1234567890"):
        result = wf._drive_loop(  # pyright: ignore[reportPrivateUsage]
            system="system",
            conversation=Conversation.from_wire(messages),
            tool_calls=0,
            start_iteration=1,
            root_task_id=None,
            original_task="t",
        )

    assert result.completed is False
    assert result.reason == "verify_command_unexecutable"
    # The auto path triggered it: verify ran, then the auto metric raised.
    assert dispatcher.calls == ["run_verify_command", "run_metric_command"]


def test_drive_loop_no_verified_commit_when_edit_follows_verify_in_turn(tmp_path: Path) -> None:
    """A turn that runs verify (green) and THEN edits must not auto-commit the
    edited tree labeled 'verify passed': the edit changed the tree the verify
    validated. Pins the verify_just_passed latch where an unverified edit was
    committed as green. (edit-then-verify, the normal order, still commits.)"""

    def _multi(*names: str) -> ProviderResponse:
        tus = tuple({"id": f"t{i}", "name": n, "input": {}} for i, n in enumerate(names))
        return ProviderResponse(
            text="",
            tool_uses=tus,
            stop_reason="tool_use",
            input_tokens=1,
            output_tokens=1,
            cache_read_tokens=0,
            cache_creation_tokens=0,
            raw={"content": [{"type": "tool_use", **tu} for tu in tus]},
        )

    class ProviderStub:
        def __init__(self) -> None:
            self.n = 0

        def call(self, **kwargs: Any) -> ProviderResponse:
            del kwargs
            self.n += 1
            if self.n == 1:
                # verify (green) THEN edit, in that order, in ONE turn.
                return _multi("run_verify_command", "apply_edit")
            return _tool_resp("finish_session", {"summary": "done"}, tool_id="fin")

    class DispatcherStub(_StubDispatcher):
        def dispatch(self, name: str, raw_input: dict[str, Any]) -> ToolResult:
            if name == "run_verify_command":
                return ExecResult(
                    returncode=0, stdout="", stderr="", duration_s=0.1, exec_failed=False
                )
            if name == "apply_edit":
                return RawResult({"ok": True})
            if name == "finish_session":
                return RawResult({"acknowledged": True, "summary": raw_input["summary"]})
            raise AssertionError(f"unexpected tool: {name}")

    provider = ProviderStub()
    config = SimpleNamespace(
        git=_GIT_STUB,
        budget=SimpleNamespace(max_usd=10.0, max_tokens_fallback=2_000_000),
        workflow=SimpleNamespace(
            verify_when="never",
            verify_retries=2,
            verify_command=("true",),
            metric=None,
        ),
        prompt=SimpleNamespace(decompose=False),
    )
    wf = _wf(
        root=tmp_path,
        config=config,
        provider=provider,
        dispatcher=DispatcherStub(),
        max_iterations=3,
    )
    messages = [{"role": "user", "content": [{"type": "text", "text": "TASK:\nx"}]}]

    commits: list[str] = []

    def _fake_commit(root: Any, subject: str) -> str:
        del root
        commits.append(subject)
        return f"sha{len(commits)}"

    with patch("agent6.workflows.loop.chain_commit", side_effect=_fake_commit):
        result = wf._drive_loop(  # pyright: ignore[reportPrivateUsage]
            system="s",
            conversation=Conversation.from_wire(messages),
            tool_calls=0,
            start_iteration=1,
            root_task_id=None,
            original_task="t",
        )

    # The verify->edit turn produced no 'verify passed' commit (old code did).
    assert commits == []
    assert result.reason == "finish_session"


def test_worker_max_tokens_starvation_backoff() -> None:
    """A metric run uses the lifted ceiling until the worker has gone quiet on 2
    CONSECUTIVE turns, then backs off to per_call_max_tokens to break a
    reasoning-binge spiral. A one-off quiet keeps the full recovery room;
    non-metric runs are unaffected."""
    metric_cfg = SimpleNamespace(
        workflow=SimpleNamespace(
            verify_when="never",
            verify_retries=2,
            verify_command=("true",),
            metric=SimpleNamespace(goal="minimize"),
        )
    )
    wf = _wf(config=metric_cfg)
    wmt = wf._worker_max_tokens  # pyright: ignore[reportPrivateUsage]
    full = max(wf.per_call_max_tokens, wf.metric_task_max_tokens)
    assert wmt(_state(went_quiet_nudges_used=0)) == full
    assert wmt(_state(went_quiet_nudges_used=1)) == full  # one-off quiet: full room
    assert wmt(_state(went_quiet_nudges_used=2)) == wf.per_call_max_tokens  # spiral: back off
    assert wmt(_state(went_quiet_nudges_used=3)) == wf.per_call_max_tokens

    # Non-metric run: always per_call, regardless of the quiet streak.
    plain = _wf(
        config=SimpleNamespace(
            git=_GIT_STUB,
            budget=SimpleNamespace(max_usd=10.0, max_tokens_fallback=2_000_000),
            workflow=SimpleNamespace(
                verify_when="never",
                verify_retries=2,
                verify_command=("true",),
                metric=None,
            ),
        )
    )
    pwmt = plain._worker_max_tokens  # pyright: ignore[reportPrivateUsage]
    assert pwmt(_state(went_quiet_nudges_used=0)) == plain.per_call_max_tokens
    assert pwmt(_state(went_quiet_nudges_used=2)) == plain.per_call_max_tokens


def test_drive_loop_starvation_backoff_breaks_the_spiral(tmp_path: Path) -> None:
    """End-to-end: a model that goes quiet at the full metric ceiling but ACTS
    once the cap is tightened recovers via the starvation backoff instead of
    dying on went_quiet. The stub's behaviour is keyed on the cap it receives, so
    this proves the backoff changes the loop's OUTCOME, not just the number:
    turns 1-2 see the lifted ceiling and go quiet; turn 3 (>= 2 consecutive
    quiets) gets per_call_max_tokens and the stub finishes. Without the backoff
    the cap would stay lifted, the stub would keep going quiet, and the run would
    die on went_quiet -- exactly GLM 5.2's observed spiral."""

    class ProviderStub:
        def __init__(self) -> None:
            self.caps_seen: list[int] = []

        def call(self, **kwargs: Any) -> ProviderResponse:
            cap = kwargs["max_tokens"]
            self.caps_seen.append(cap)
            if cap >= 65536:
                # Full ceiling: a reasoning binge that emits nothing actionable.
                return ProviderResponse(
                    text="",
                    tool_uses=(),
                    stop_reason="end_turn",
                    input_tokens=1,
                    output_tokens=1,
                    cache_read_tokens=0,
                    cache_creation_tokens=0,
                    raw={"content": []},
                )
            # Tightened cap: the model is forced to act.
            return _tool_resp("finish_session", {"summary": "done"}, tool_id="fin")

    class DispatcherStub(_StubDispatcher):
        def dispatch(self, name: str, raw_input: dict[str, Any]) -> ToolResult:
            if name == "finish_session":
                return RawResult({"acknowledged": True, "summary": raw_input["summary"]})
            raise AssertionError(f"unexpected tool: {name}")

    provider = ProviderStub()
    config = SimpleNamespace(
        git=_GIT_STUB,
        budget=SimpleNamespace(max_usd=10.0, max_tokens_fallback=2_000_000),
        workflow=SimpleNamespace(
            verify_when="never",
            verify_retries=2,
            verify_command=("true",),
            metric=SimpleNamespace(goal="minimize"),
        ),
    )
    wf = _wf(
        root=tmp_path,
        config=config,
        provider=provider,
        dispatcher=DispatcherStub(),
        max_iterations=10,
        per_call_max_tokens=16384,
        metric_task_max_tokens=65536,
    )
    messages = [{"role": "user", "content": [{"type": "text", "text": "TASK:\noptimize"}]}]
    result = wf._drive_loop(  # pyright: ignore[reportPrivateUsage]
        system="s",
        conversation=Conversation.from_wire(messages),
        tool_calls=0,
        start_iteration=1,
        root_task_id=None,
        original_task="t",
    )
    # Recovered (finished) rather than dying on went_quiet.
    assert result.reason == "finish_session"
    # Two quiet turns at the lifted ceiling, then the backoff to per_call.
    assert provider.caps_seen[:3] == [65536, 65536, 16384]


def test_drive_loop_finishes_on_metric_plateau(tmp_path: Path) -> None:
    class ProviderStub:
        def __init__(self) -> None:
            self.calls = 0

        def call(self, **kwargs: Any) -> ProviderResponse:
            del kwargs
            self.calls += 1
            return _tool_resp("run_verify_command", tool_id=f"verify-{self.calls}")

    class DispatcherStub(_StubDispatcher):
        def __init__(self) -> None:
            self.calls: list[str] = []
            # Improves to 50, then ties it. The plateau detector fires once
            # >=5 parsed samples exist, but the loop now answers the first
            # _METRIC_PLATEAU_PATIENCE (3) plateaus with a pivot nudge and
            # only stops on the 4th, so we need four tied samples at the end.
            self.scores = iter([100.0, 80.0, 60.0, 50.0, 50.0, 50.0, 50.0, 50.0])

        def dispatch(self, name: str, raw_input: dict[str, Any]) -> ToolResult:
            del raw_input
            self.calls.append(name)
            if name == "run_verify_command":
                return ExecResult(
                    returncode=0, stdout="", stderr="", duration_s=0.1, exec_failed=False
                )
            if name == "run_metric_command":
                score = next(self.scores)
                return MetricResult(
                    returncode=0,
                    stdout=f"CYCLES: {score:g}\n",
                    stderr="",
                    duration_s=0.1,
                    exec_failed=False,
                    score=score,
                )
            raise AssertionError(f"unexpected tool: {name}")

    provider = ProviderStub()
    dispatcher = DispatcherStub()
    config = SimpleNamespace(
        git=_GIT_STUB,
        budget=SimpleNamespace(max_usd=10.0, max_tokens_fallback=2_000_000),
        workflow=SimpleNamespace(
            verify_when="never",
            verify_retries=2,
            verify_command=("true",),
            metric=SimpleNamespace(goal="minimize"),
        ),
    )
    wf = _wf(
        root=tmp_path,
        config=config,
        provider=provider,
        dispatcher=dispatcher,
        max_iterations=10,
    )
    messages = [{"role": "user", "content": [{"type": "text", "text": "TASK:\noptimize"}]}]

    with patch(
        "agent6.workflows.loop.chain_commit",
        side_effect=["sha1", "sha2", "sha3", "sha4", "sha5", "sha6", "sha7", "sha8"],
    ):
        result = wf._drive_loop(  # pyright: ignore[reportPrivateUsage]
            system="system",
            conversation=Conversation.from_wire(messages),
            tool_calls=0,
            start_iteration=1,
            root_task_id=None,
            original_task="t",
        )

    assert result.completed is True
    assert result.reason == "metric_plateau"
    assert "performance per dollar" in result.summary
    # 8 verify+metric pairs: samples 5-7 each draw a pivot nudge, sample 8 stops.
    assert dispatcher.calls == ["run_verify_command", "run_metric_command"] * 8


def test_drive_loop_plateau_nudges_before_stopping(tmp_path: Path) -> None:
    """The first plateau should not stop the run: the loop injects a pivot
    nudge and keeps going, so a worker that changes strategy can recover the
    remaining budget instead of quitting at a local optimum."""

    class ProviderStub:
        def __init__(self) -> None:
            self.calls = 0
            self.saw_plateau_nudge = False

        def call(self, **kwargs: Any) -> ProviderResponse:
            self.calls += 1
            rendered = str(kwargs["messages"][-1])
            if "[harness plateau]" in rendered:
                self.saw_plateau_nudge = True
                return _tool_resp("finish_session", {"summary": "pivoted"}, tool_id="fin")
            return _tool_resp("run_verify_command", tool_id=f"verify-{self.calls}")

    class DispatcherStub(_StubDispatcher):
        def __init__(self) -> None:
            self.calls: list[str] = []
            self.scores = iter([100.0, 80.0, 60.0, 50.0, 50.0])

        def dispatch(self, name: str, raw_input: dict[str, Any]) -> ToolResult:
            self.calls.append(name)
            if name == "run_verify_command":
                return ExecResult(
                    returncode=0, stdout="", stderr="", duration_s=0.1, exec_failed=False
                )
            if name == "run_metric_command":
                score = next(self.scores)
                return MetricResult(
                    returncode=0,
                    stdout=f"CYCLES: {score:g}\n",
                    stderr="",
                    duration_s=0.1,
                    exec_failed=False,
                    score=score,
                )
            if name == "finish_session":
                return RawResult({"acknowledged": True, "summary": raw_input["summary"]})
            raise AssertionError(f"unexpected tool: {name}")

    provider = ProviderStub()
    dispatcher = DispatcherStub()
    config = SimpleNamespace(
        git=_GIT_STUB,
        budget=SimpleNamespace(max_usd=10.0, max_tokens_fallback=2_000_000),
        workflow=SimpleNamespace(
            verify_when="never",
            verify_retries=2,
            verify_command=("true",),
            metric=SimpleNamespace(goal="minimize"),
        ),
    )
    wf = _wf(
        root=tmp_path,
        config=config,
        provider=provider,
        dispatcher=dispatcher,
        max_iterations=10,
    )
    messages = [{"role": "user", "content": [{"type": "text", "text": "TASK:\noptimize"}]}]

    with patch(
        "agent6.workflows.loop.chain_commit",
        side_effect=["sha1", "sha2", "sha3", "sha4", "sha5"],
    ):
        result = wf._drive_loop(  # pyright: ignore[reportPrivateUsage]
            system="system",
            conversation=Conversation.from_wire(messages),
            tool_calls=0,
            start_iteration=1,
            root_task_id=None,
            original_task="t",
        )

    # The plateau at the 5th sample injected a pivot nudge instead of
    # stopping; the worker saw it and finished on its own terms.
    assert provider.saw_plateau_nudge is True
    assert result.reason == "finish_session"


def test_drive_loop_plateau_final_nudge_fires_in_final_budget_slice(tmp_path: Path) -> None:
    """On a REAL-budget run, ties while budget is high must not exhaust the
    plateau patience: the escalating FINAL ("make your one best bet") nudge has
    to still fire once the budget enters the final slice. Pins the bug where
    `plateau_nudges_used` accrued on high-budget ties, so the run stopped the
    instant the budget crossed the threshold and the FINAL nudge never showed."""
    from agent6.workflows._metric import (
        METRIC_PLATEAU_NUDGE_FINAL as _METRIC_PLATEAU_NUDGE_FINAL,
    )

    class ProviderStub:
        def __init__(self) -> None:
            self.calls = 0
            self.saw_final_nudge = False

        def call(self, **kwargs: Any) -> ProviderResponse:
            self.calls += 1
            if _METRIC_PLATEAU_NUDGE_FINAL in str(kwargs["messages"][-1]):
                self.saw_final_nudge = True
            # Vary the call signature each turn so the repeat-loop-guard (which
            # kills at 10 identical back-to-back calls) does not fire; a real
            # worker varies its edits between verifies. This isolates the plateau
            # logic under test from the orthogonal loop-guard.
            return _tool_resp(
                "run_verify_command", {"n": self.calls}, tool_id=f"verify-{self.calls}"
            )

    class DispatcherStub(_StubDispatcher):
        def __init__(self) -> None:
            self.calls: list[str] = []
            self.metric_count = 0
            # Improve to 50, then tie it for many rounds. Plateau fires from the
            # 5th sample; ties 5-8 land while budget is high (runway), 9+ land in
            # the final slice. With the fix, runway ties do not consume patience,
            # so the FINAL nudge fires on samples 9/10/11 and the run stops on 12.
            self.scores = iter([100.0, 80.0, 60.0, 50.0] + [50.0] * 8)

        def dispatch(self, name: str, raw_input: dict[str, Any]) -> ToolResult:
            del raw_input
            self.calls.append(name)
            if name == "run_verify_command":
                return ExecResult(
                    returncode=0, stdout="", stderr="", duration_s=0.1, exec_failed=False
                )
            if name == "run_metric_command":
                self.metric_count += 1
                score = next(self.scores)
                return MetricResult(
                    returncode=0,
                    stdout=f"CYCLES: {score:g}\n",
                    stderr="",
                    duration_s=0.1,
                    exec_failed=False,
                    score=score,
                )
            raise AssertionError(f"unexpected tool: {name}")

    provider = ProviderStub()
    dispatcher = DispatcherStub()
    config = SimpleNamespace(
        git=_GIT_STUB,
        budget=SimpleNamespace(max_usd=10.0, max_tokens_fallback=2_000_000),
        workflow=SimpleNamespace(
            verify_when="never",
            verify_retries=2,
            verify_command=("true",),
            metric=SimpleNamespace(goal="minimize"),
        ),
    )
    wf = _wf(
        root=tmp_path,
        config=config,
        provider=provider,
        dispatcher=dispatcher,
        max_iterations=20,
    )
    # Drive the budget fraction off the dispatcher's measurement count (robust to
    # _budget_fraction_remaining being read more than once per iteration): samples
    # 5-8 see 80% left (runway), 9+ see 10% left (final slice, FINAL nudge tier).
    wf._budget_fraction_remaining = lambda: 0.8 if dispatcher.metric_count <= 8 else 0.1  # type: ignore[method-assign]  # pyright: ignore[reportPrivateUsage]
    messages = [{"role": "user", "content": [{"type": "text", "text": "TASK:\noptimize"}]}]

    with patch(
        "agent6.workflows.loop.chain_commit",
        side_effect=[f"sha{i}" for i in range(20)],
    ):
        result = wf._drive_loop(  # pyright: ignore[reportPrivateUsage]
            system="system",
            conversation=Conversation.from_wire(messages),
            tool_calls=0,
            start_iteration=1,
            root_task_id=None,
            original_task="t",
        )

    # The FINAL nudge must have fired in the final slice before the run stopped.
    assert provider.saw_final_nudge is True
    assert result.reason == "metric_plateau"
    # Runway ties (samples 5-8) did not consume patience, so the run kept going
    # well past the point the old code stopped (sample 9): >=12 metric samples.
    assert dispatcher.metric_count >= 12


def test_drive_loop_plan_finish_nudge_fires_once_at_iter_cap(tmp_path: Path) -> None:
    """A verbose planner that never calls finish_planning gets a single harness
    'finish now' nudge once it hits the plan turn cap -- not before, not again.
    This is the lever that makes Kimi K2.6 actually land a plan; pins the
    off-by-one (iteration - start + 1 >= cap) and the one-shot latch."""
    from agent6.workflows.loop import (
        PLAN_BUDGET_NUDGE,  # pyright: ignore[reportPrivateUsage]
        PLAN_NUDGE_AFTER_ITERS,  # pyright: ignore[reportPrivateUsage]
    )

    class ProviderStub:
        def __init__(self) -> None:
            self.calls = 0
            self.nudged_on: list[int] = []

        def call(self, **kwargs: Any) -> ProviderResponse:
            self.calls += 1
            if PLAN_BUDGET_NUDGE[:24] in str(kwargs["messages"][-1]):
                self.nudged_on.append(self.calls)
            # never finish on our own -> the loop must force the issue
            return _tool_resp("read_file", {"path": f"f{self.calls}.py"}, tool_id=f"r-{self.calls}")

    class DispatcherStub(_StubDispatcher):
        def dispatch(self, name: str, raw_input: dict[str, Any]) -> ToolResult:
            assert name == "read_file"
            return RawResult({"content": "..."})

    provider = ProviderStub()
    wf = _wf(
        root=tmp_path,
        mode="plan",
        provider=provider,
        dispatcher=DispatcherStub(),
        max_iterations=PLAN_NUDGE_AFTER_ITERS + 3,
    )
    messages = [{"role": "user", "content": [{"type": "text", "text": "TASK:\nplan a feature"}]}]
    wf._drive_loop(  # pyright: ignore[reportPrivateUsage]
        system="s",
        conversation=Conversation.from_wire(messages),
        tool_calls=0,
        start_iteration=1,
        root_task_id=None,
        original_task="t",
    )
    # Injected exactly once, on the turn-cap iteration (mode stays "plan" on
    # every later turn, so the latch is what keeps it to one).
    assert provider.nudged_on == [PLAN_NUDGE_AFTER_ITERS]


def test_drive_loop_plan_finish_nudge_fires_on_low_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The nudge also fires early when the token budget runs low (not only on
    the turn cap) -- e.g. a planner reading large files burns budget fast."""
    from agent6.workflows import loop as loopmod
    from agent6.workflows.loop import PLAN_BUDGET_NUDGE  # pyright: ignore[reportPrivateUsage]

    def _low_budget(_self: object) -> float:
        return 0.2

    monkeypatch.setattr(loopmod.Workflow, "_budget_fraction_remaining", _low_budget)

    class ProviderStub:
        def __init__(self) -> None:
            self.calls = 0
            self.nudged_on: list[int] = []

        def call(self, **kwargs: Any) -> ProviderResponse:
            self.calls += 1
            if PLAN_BUDGET_NUDGE[:24] in str(kwargs["messages"][-1]):
                self.nudged_on.append(self.calls)
            return _tool_resp("read_file", {"path": f"f{self.calls}.py"}, tool_id=f"r-{self.calls}")

    class DispatcherStub(_StubDispatcher):
        def dispatch(self, name: str, raw_input: dict[str, Any]) -> ToolResult:
            return RawResult({"content": "..."})

    provider = ProviderStub()
    wf = _wf(
        root=tmp_path,
        mode="plan",
        provider=provider,
        dispatcher=DispatcherStub(),
        max_iterations=5,
    )
    messages = [{"role": "user", "content": [{"type": "text", "text": "TASK:\nplan"}]}]
    wf._drive_loop(  # pyright: ignore[reportPrivateUsage]
        system="s",
        conversation=Conversation.from_wire(messages),
        tool_calls=0,
        start_iteration=1,
        root_task_id=None,
        original_task="t",
    )
    # Budget already below the threshold -> nudge on the very first turn, once.
    assert provider.nudged_on == [1]


def test_drive_loop_run_budget_nudge_forces_verify_and_finish(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A non-metric `run` gets a one-shot wrap-up nudge when budget runs low.
    Observed live: the worker solves the task but never re-verifies or calls
    finish_session, so the budget dies on read-only commands."""
    from agent6.workflows import loop as loopmod
    from agent6.workflows.loop import RUN_BUDGET_NUDGE  # pyright: ignore[reportPrivateUsage]

    def _low_budget(_self: object) -> float:
        return 0.2

    monkeypatch.setattr(loopmod.Workflow, "_budget_fraction_remaining", _low_budget)

    class ProviderStub:
        def __init__(self) -> None:
            self.calls = 0
            self.nudged_on: list[int] = []

        def call(self, **kwargs: Any) -> ProviderResponse:
            self.calls += 1
            if RUN_BUDGET_NUDGE[:24] in str(kwargs["messages"][-1]):
                self.nudged_on.append(self.calls)
            return _tool_resp("list_dir", {"path": "."}, tool_id=f"l-{self.calls}")

    class DispatcherStub(_StubDispatcher):
        def dispatch(self, name: str, raw_input: dict[str, Any]) -> ToolResult:
            return RawResult({"content": "..."})

    provider = ProviderStub()
    wf = _wf(
        root=tmp_path,
        mode="run",
        provider=provider,
        dispatcher=DispatcherStub(),
        max_iterations=4,
    )
    messages = [{"role": "user", "content": [{"type": "text", "text": "TASK:\nfix"}]}]
    wf._drive_loop(  # pyright: ignore[reportPrivateUsage]
        system="s",
        conversation=Conversation.from_wire(messages),
        tool_calls=0,
        start_iteration=1,
        root_task_id=None,
        original_task="t",
    )
    # fires once, on the first turn at/below the threshold, and only once.
    assert provider.nudged_on == [1]


def test_drive_loop_verify_settled_nudges_then_stops(tmp_path: Path) -> None:
    """A run-mode worker that keeps spinning after verify already passed (no new
    commit, no edit) gets one finish nudge, then the loop stops it with
    reason='verify_settled' — the positive completion signal a non-metric run
    otherwise lacks (Kimi K2.6 observed running 128 iters when done at ~45)."""
    from agent6.workflows.loop import VERIFY_SETTLED_NUDGE  # pyright: ignore[reportPrivateUsage]

    class ProviderStub:
        def __init__(self) -> None:
            self.calls = 0
            self.saw_nudge = False

        def call(self, **kwargs: Any) -> ProviderResponse:
            self.calls += 1
            if VERIFY_SETTLED_NUDGE[:24] in str(kwargs["messages"][-1]):
                self.saw_nudge = True
            if self.calls == 1:
                return _tool_resp("run_verify_command", tool_id="v1")  # -> verify passes
            # then spin on read-only commands forever (no edit, no commit)
            return _tool_resp("run_command", {"cmd": f"ls {self.calls}"}, tool_id=f"c{self.calls}")

    class DispatcherStub(_StubDispatcher):
        def dispatch(self, name: str, raw_input: dict[str, Any]) -> ToolResult:
            return ExecResult(
                returncode=0, stdout="ok", stderr="", duration_s=0.1, exec_failed=False
            )

    provider = ProviderStub()
    config = SimpleNamespace(
        git=_GIT_STUB,
        budget=SimpleNamespace(max_usd=10.0, max_tokens_fallback=2_000_000),
        workflow=SimpleNamespace(
            verify_when="never",
            verify_retries=2,
            verify_command=("true",),
            metric=SimpleNamespace(goal=None),
        ),
    )
    wf = _wf(
        root=tmp_path,
        config=config,
        mode="run",
        provider=provider,
        dispatcher=DispatcherStub(),
        max_iterations=30,
    )
    messages = [{"role": "user", "content": [{"type": "text", "text": "TASK:\ndo it"}]}]
    with patch("agent6.workflows.loop.chain_commit", return_value="sha1"):
        result = wf._drive_loop(  # pyright: ignore[reportPrivateUsage]
            system="s",
            conversation=Conversation.from_wire(messages),
            tool_calls=0,
            start_iteration=1,
            root_task_id=None,
            original_task="t",
        )
    assert provider.saw_nudge is True
    assert result.reason == "verify_settled"
    assert result.completed is True


def test_drive_loop_settle_after_unreverified_edits_is_not_passed(tmp_path: Path) -> None:
    """A green verify followed by edits that never re-verify must not settle as
    'verify passed': the settle end grounds on the same tree probe as
    finish_session, so it downgrades to reason='settled' with the stale-green
    summary (all_passed=False)."""

    class ProviderStub:
        def __init__(self) -> None:
            self.calls = 0

        def call(self, **kwargs: Any) -> ProviderResponse:
            self.calls += 1
            if self.calls == 1:
                return _tool_resp("run_verify_command", tool_id="v1")  # green
            if self.calls == 2:  # then an edit nothing re-verifies
                return _tool_resp(
                    "apply_edit",
                    {"path": "a.py", "edits": [{"kind": "create", "new_string": "x = 2\n"}]},
                    tool_id="e1",
                )
            return _tool_resp("run_command", {"cmd": f"ls {self.calls}"}, tool_id=f"c{self.calls}")

    class DispatcherStub(_StubDispatcher):
        def dispatch(self, name: str, raw_input: dict[str, Any]) -> ToolResult:
            return ExecResult(
                returncode=0, stdout="ok", stderr="", duration_s=0.1, exec_failed=False
            )

    config = SimpleNamespace(
        git=_GIT_STUB,
        budget=SimpleNamespace(max_usd=10.0, max_tokens_fallback=2_000_000),
        workflow=SimpleNamespace(
            verify_when="never",
            verify_retries=2,
            verify_command=("true",),
            metric=SimpleNamespace(goal=None),
        ),
    )
    events: list[dict[str, Any]] = []

    class _Events:
        def emit(self, event_type: str, /, **fields: Any) -> None:
            events.append({"type": event_type, **fields})

    wf = _wf(
        root=tmp_path,
        config=config,
        mode="run",
        provider=ProviderStub(),
        dispatcher=DispatcherStub(),
        max_iterations=30,
        events=_Events(),
    )
    messages = [{"role": "user", "content": [{"type": "text", "text": "TASK:\ndo it"}]}]
    with patch("agent6.workflows.loop.chain_commit", return_value="sha1"):
        result = wf._drive_loop(  # pyright: ignore[reportPrivateUsage]
            system="s",
            conversation=Conversation.from_wire(messages),
            tool_calls=0,
            start_iteration=1,
            root_task_id=None,
            original_task="t",
        )
    assert result.reason == "settled"
    assert "never re-verified" in result.summary
    ends = [e for e in events if e["type"] == "session.end"]
    assert ends and ends[-1]["all_passed"] is False


def test_drive_loop_verify_settled_does_not_fire_before_first_verify(tmp_path: Path) -> None:
    """The settled detector must stay dormant until verify has passed at least
    once — a worker still reading toward its first green build must not be
    stopped early."""
    from agent6.workflows.loop import VERIFY_SETTLED_NUDGE  # pyright: ignore[reportPrivateUsage]

    class ProviderStub:
        def __init__(self) -> None:
            self.calls = 0
            self.saw_nudge = False

        def call(self, **kwargs: Any) -> ProviderResponse:
            self.calls += 1
            if VERIFY_SETTLED_NUDGE[:24] in str(kwargs["messages"][-1]):
                self.saw_nudge = True
            if self.calls >= 6:
                return _tool_resp("finish_session", {"summary": "done"}, tool_id="fin")
            return _tool_resp("read_file", {"path": f"f{self.calls}.py"}, tool_id=f"r{self.calls}")

    class DispatcherStub(_StubDispatcher):
        def dispatch(self, name: str, raw_input: dict[str, Any]) -> ToolResult:
            if name == "finish_session":
                return RawResult({"acknowledged": True, "summary": raw_input["summary"]})
            return RawResult({"content": "..."})

    provider = ProviderStub()
    config = SimpleNamespace(
        git=_GIT_STUB,
        budget=SimpleNamespace(max_usd=10.0, max_tokens_fallback=2_000_000),
        workflow=SimpleNamespace(
            verify_when="never",
            verify_retries=2,
            verify_command=("true",),
            metric=SimpleNamespace(goal=None),
        ),
    )
    wf = _wf(
        root=tmp_path, config=config, mode="run", provider=provider, dispatcher=DispatcherStub()
    )
    messages = [{"role": "user", "content": [{"type": "text", "text": "TASK:\ndo it"}]}]
    result = wf._drive_loop(  # pyright: ignore[reportPrivateUsage]
        system="s",
        conversation=Conversation.from_wire(messages),
        tool_calls=0,
        start_iteration=1,
        root_task_id=None,
        original_task="t",
    )
    # never verified -> never nudged/stopped by the settled detector
    assert provider.saw_nudge is False
    assert result.reason == "finish_session"


def test_drive_loop_verify_settled_neutral_on_reverify(tmp_path: Path) -> None:
    """Re-running verify on an already-green tree (which the prompt encourages
    between reads) is active work, not idle — it must NOT accrue toward the
    verify-settled hard-stop, or a legit run gets truncated."""

    class ProviderStub:
        def __init__(self) -> None:
            self.calls = 0

        def call(self, **kwargs: Any) -> ProviderResponse:
            self.calls += 1
            return _tool_resp("run_verify_command", tool_id=f"v{self.calls}")  # always re-verify

    class DispatcherStub(_StubDispatcher):
        def dispatch(self, name: str, raw_input: dict[str, Any]) -> ToolResult:
            return ExecResult(
                returncode=0, stdout="ok", stderr="", duration_s=0.1, exec_failed=False
            )

    provider = ProviderStub()
    config = SimpleNamespace(
        git=_GIT_STUB,
        budget=SimpleNamespace(max_usd=10.0, max_tokens_fallback=2_000_000),
        workflow=SimpleNamespace(
            verify_when="never",
            verify_retries=2,
            verify_command=("true",),
            metric=SimpleNamespace(goal=None),
        ),
    )
    wf = _wf(
        root=tmp_path,
        config=config,
        mode="run",
        provider=provider,
        dispatcher=DispatcherStub(),
        max_iterations=10,
    )
    messages = [{"role": "user", "content": [{"type": "text", "text": "TASK:\ndo it"}]}]
    with patch("agent6.workflows.loop.chain_commit", return_value=""):
        result = wf._drive_loop(  # pyright: ignore[reportPrivateUsage]
            system="s",
            conversation=Conversation.from_wire(messages),
            tool_calls=0,
            start_iteration=1,
            root_task_id=None,
            original_task="t",
        )
    assert result.reason != "verify_settled"


def test_drive_loop_verify_settled_dormant_on_metric_runs(tmp_path: Path) -> None:
    """On a metric run, post-verify measure/analyse/read iterations legitimately
    make no commit; completion is owned by the metric early-finish + plateau
    logic, so the verify-settled detector must NOT hard-stop them."""

    class ProviderStub:
        def __init__(self) -> None:
            self.calls = 0

        def call(self, **kwargs: Any) -> ProviderResponse:
            self.calls += 1
            if self.calls == 1:
                return _tool_resp("run_verify_command", tool_id="v1")  # verify passes
            return _tool_resp("run_command", {"cmd": f"ls {self.calls}"}, tool_id=f"c{self.calls}")

    class DispatcherStub(_StubDispatcher):
        def dispatch(self, name: str, raw_input: dict[str, Any]) -> ToolResult:
            if name == "run_metric_command":
                return MetricResult(
                    returncode=0,
                    stdout="ok",
                    stderr="",
                    duration_s=0.1,
                    exec_failed=False,
                    score=None,
                )
            return ExecResult(
                returncode=0, stdout="ok", stderr="", duration_s=0.1, exec_failed=False
            )

    provider = ProviderStub()
    # goal set -> this is a metric run (still mode=="run")
    config = SimpleNamespace(
        git=_GIT_STUB,
        budget=SimpleNamespace(max_usd=10.0, max_tokens_fallback=2_000_000),
        workflow=SimpleNamespace(
            verify_when="never",
            verify_retries=2,
            verify_command=("true",),
            metric=SimpleNamespace(goal="minimize"),
        ),
    )
    wf = _wf(
        root=tmp_path,
        config=config,
        mode="run",
        provider=provider,
        dispatcher=DispatcherStub(),
        max_iterations=8,
    )
    messages = [{"role": "user", "content": [{"type": "text", "text": "TASK:\noptimize"}]}]
    with patch("agent6.workflows.loop.chain_commit", return_value=""):
        result = wf._drive_loop(  # pyright: ignore[reportPrivateUsage]
            system="s",
            conversation=Conversation.from_wire(messages),
            tool_calls=0,
            start_iteration=1,
            root_task_id=None,
            original_task="t",
        )
    # would have been killed at idle 6 without the metric gate
    assert result.reason != "verify_settled"


def test_metric_plateau_nudge_escalates_with_budget_pressure() -> None:
    from agent6.workflows._metric import (
        METRIC_PLATEAU_NUDGE_EXPLORE as _METRIC_PLATEAU_NUDGE_EXPLORE,
    )
    from agent6.workflows._metric import (
        METRIC_PLATEAU_NUDGE_FINAL as _METRIC_PLATEAU_NUDGE_FINAL,
    )
    from agent6.workflows._metric import (
        METRIC_PLATEAU_NUDGE_PIVOT as _METRIC_PLATEAU_NUDGE_PIVOT,
    )
    from agent6.workflows._metric import (
        metric_plateau_nudge as _metric_plateau_nudge,
    )

    # No budget signal -> explore tier (keep trying new directions).
    assert _metric_plateau_nudge(None) is _METRIC_PLATEAU_NUDGE_EXPLORE
    # Plenty of runway -> explore.
    assert _metric_plateau_nudge(0.80) is _METRIC_PLATEAU_NUDGE_EXPLORE
    # Boundary at 0.5 is still "more than half" only when strictly above.
    assert _metric_plateau_nudge(0.50) is _METRIC_PLATEAU_NUDGE_PIVOT
    # Mid budget -> decisive pivot.
    assert _metric_plateau_nudge(0.40) is _METRIC_PLATEAU_NUDGE_PIVOT
    # Final slice -> single best bet.
    assert _metric_plateau_nudge(0.20) is _METRIC_PLATEAU_NUDGE_FINAL
    # Every tier keeps the greppable marker.
    for tier in (
        _METRIC_PLATEAU_NUDGE_EXPLORE,
        _METRIC_PLATEAU_NUDGE_PIVOT,
        _METRIC_PLATEAU_NUDGE_FINAL,
    ):
        assert tier.startswith("[harness plateau]")


def test_drive_loop_plateau_keeps_nudging_while_budget_high(tmp_path: Path) -> None:
    """With most of the budget unspent, a metric plateau must NOT terminate
    the run even after the fixed nudge patience is exhausted — the loop keeps
    pivoting until the budget enters its final slice."""
    from agent6.budget import BudgetTracker

    class ProviderStub:
        def __init__(self) -> None:
            self.calls = 0
            self.plateau_nudges_seen = 0

        def call(self, **kwargs: Any) -> ProviderResponse:
            self.calls += 1
            rendered = str(kwargs["messages"][-1])
            if "[harness plateau]" in rendered:
                self.plateau_nudges_seen += 1
            return _tool_resp("run_verify_command", tool_id=f"verify-{self.calls}")

    class DispatcherStub(_StubDispatcher):
        def __init__(self) -> None:
            self.calls: list[str] = []
            # Plateaus at the 5th sample and stays flat thereafter.
            self.scores = iter([100.0, 80.0, 60.0, 50.0] + [50.0] * 20)

        def dispatch(self, name: str, raw_input: dict[str, Any]) -> ToolResult:
            del raw_input
            self.calls.append(name)
            if name == "run_verify_command":
                return ExecResult(
                    returncode=0, stdout="", stderr="", duration_s=0.1, exec_failed=False
                )
            if name == "run_metric_command":
                score = next(self.scores)
                return MetricResult(
                    returncode=0,
                    stdout=f"CYCLES: {score:g}\n",
                    stderr="",
                    duration_s=0.1,
                    exec_failed=False,
                    score=score,
                )
            raise AssertionError(f"unexpected tool: {name}")

    provider = ProviderStub()
    dispatcher = DispatcherStub()
    config = SimpleNamespace(
        git=_GIT_STUB,
        budget=SimpleNamespace(max_usd=10.0, max_tokens_fallback=2_000_000),
        workflow=SimpleNamespace(
            verify_when="never",
            verify_retries=2,
            verify_command=("true",),
            metric=SimpleNamespace(goal="minimize"),
        ),
    )
    # Fresh budget with huge ceilings -> fraction_remaining stays ~1.0, well
    # above the final-slice threshold, so the plateau never becomes terminal.
    budget = BudgetTracker(max_usd=-1, max_tokens_fallback=-1, max_percent=-1)
    max_iters = 12
    wf = _wf(
        root=tmp_path,
        config=config,
        provider=provider,
        dispatcher=dispatcher,
        budget=budget,
        max_iterations=max_iters,
        loop_guard_kill_threshold=0,
    )
    messages = [{"role": "user", "content": [{"type": "text", "text": "TASK:\noptimize"}]}]

    with patch(
        "agent6.workflows.loop.chain_commit",
        side_effect=[f"sha{i}" for i in range(1, max_iters + 2)],
    ):
        result = wf._drive_loop(  # pyright: ignore[reportPrivateUsage]
            system="system",
            conversation=Conversation.from_wire(messages),
            tool_calls=0,
            start_iteration=1,
            root_task_id=None,
            original_task="t",
        )

    # Ran out the iteration cap rather than stopping on the plateau, and
    # kept nudging past the fixed patience of 3.
    assert result.reason == "max_iterations"
    assert provider.plateau_nudges_seen > 3


def test_drive_loop_rejects_early_finish_while_budget_high(tmp_path: Path) -> None:
    """A finish_session on a metric run with most of the budget unspent is rejected
    and nudged a few times before the loop honours it."""
    from agent6.budget import BudgetTracker

    class ProviderStub:
        def __init__(self) -> None:
            self.calls = 0
            self.finish_nudges_seen = 0

        def call(self, **kwargs: Any) -> ProviderResponse:
            self.calls += 1
            rendered = str(kwargs["messages"][-1])
            if "[harness budget]" in rendered:
                self.finish_nudges_seen += 1
            # Vary the summary so the loop-guard repeat detector stays quiet.
            return _tool_resp(
                "finish_session",
                {"summary": f"done-{self.calls}"},
                tool_id=f"finish-{self.calls}",
            )

    class DispatcherStub(_StubDispatcher):
        def dispatch(self, name: str, raw_input: dict[str, Any]) -> ToolResult:
            del name, raw_input
            return RawResult({"ok": True})

    provider = ProviderStub()
    config = SimpleNamespace(
        git=_GIT_STUB,
        budget=SimpleNamespace(max_usd=10.0, max_tokens_fallback=2_000_000),
        workflow=SimpleNamespace(
            verify_when="never",
            verify_retries=2,
            verify_command=("true",),
            metric=SimpleNamespace(goal="minimize"),
        ),
    )
    # Huge ceilings keep fraction_remaining ~1.0, well above the final slice.
    budget = BudgetTracker(max_usd=-1, max_tokens_fallback=-1, max_percent=-1)
    wf = _wf(
        root=tmp_path,
        config=config,
        provider=provider,
        dispatcher=DispatcherStub(),
        budget=budget,
        max_iterations=20,
        loop_guard_kill_threshold=0,
    )
    messages = [{"role": "user", "content": [{"type": "text", "text": "TASK:\noptimize"}]}]

    result = wf._drive_loop(  # pyright: ignore[reportPrivateUsage]
        system="system",
        conversation=Conversation.from_wire(messages),
        tool_calls=0,
        start_iteration=1,
        root_task_id=None,
        original_task="t",
    )

    # Rejected for the fixed patience of 3, then honoured on the 4th call.
    assert result.reason == "finish_session"
    assert provider.finish_nudges_seen == 3
    assert provider.calls == 4


def test_drive_loop_honors_finish_without_budget_signal(tmp_path: Path) -> None:
    """With no budget tracker wired in, an early finish_session is honoured at once
    so the guard can never deadlock a run that lacks a budget signal."""

    class ProviderStub:
        def __init__(self) -> None:
            self.calls = 0
            self.finish_nudges_seen = 0

        def call(self, **kwargs: Any) -> ProviderResponse:
            self.calls += 1
            rendered = str(kwargs["messages"][-1])
            if "[harness budget]" in rendered:
                self.finish_nudges_seen += 1
            return _tool_resp("finish_session", {"summary": "done"}, tool_id=f"finish-{self.calls}")

    class DispatcherStub(_StubDispatcher):
        def dispatch(self, name: str, raw_input: dict[str, Any]) -> ToolResult:
            del name, raw_input
            return RawResult({"ok": True})

    provider = ProviderStub()
    config = SimpleNamespace(
        git=_GIT_STUB,
        budget=SimpleNamespace(max_usd=10.0, max_tokens_fallback=2_000_000),
        workflow=SimpleNamespace(
            verify_when="never",
            verify_retries=2,
            verify_command=("true",),
            metric=SimpleNamespace(goal="minimize"),
        ),
    )
    wf = _wf(
        root=tmp_path,
        config=config,
        provider=provider,
        dispatcher=DispatcherStub(),
        budget=None,
        max_iterations=20,
        loop_guard_kill_threshold=0,
    )
    messages = [{"role": "user", "content": [{"type": "text", "text": "TASK:\noptimize"}]}]

    result = wf._drive_loop(  # pyright: ignore[reportPrivateUsage]
        system="system",
        conversation=Conversation.from_wire(messages),
        tool_calls=0,
        start_iteration=1,
        root_task_id=None,
        original_task="t",
    )

    assert result.reason == "finish_session"
    assert provider.finish_nudges_seen == 0
    assert provider.calls == 1


def test_tool_calls_after_finish_session_are_not_executed(tmp_path: Path) -> None:
    """The finish tools say "tool calls after it are not executed": a
    run_command emitted after finish_session in the same message is answered
    with an error result and never dispatched."""

    class ProviderStub:
        def __init__(self) -> None:
            self.calls = 0

        def call(self, **kwargs: Any) -> ProviderResponse:
            del kwargs
            self.calls += 1
            uses = (
                {"id": "f1", "name": "finish_session", "input": {"summary": "done"}},
                {"id": "c1", "name": "run_command", "input": {"command": "rm -rf build"}},
            )
            return ProviderResponse(
                text="",
                tool_uses=uses,
                stop_reason="tool_use",
                input_tokens=1,
                output_tokens=1,
                cache_read_tokens=0,
                cache_creation_tokens=0,
                raw={"content": [{"type": "tool_use", **u} for u in uses]},
            )

    class DispatcherStub(_StubDispatcher):
        def __init__(self) -> None:
            self.calls: list[str] = []

        def dispatch(self, name: str, raw_input: dict[str, Any]) -> ToolResult:
            del raw_input
            self.calls.append(name)
            return RawResult({"ok": True})

    provider = ProviderStub()
    dispatcher = DispatcherStub()
    config = SimpleNamespace(
        git=_GIT_STUB,
        budget=SimpleNamespace(max_usd=10.0, max_tokens_fallback=2_000_000),
        workflow=SimpleNamespace(
            verify_when="never",
            verify_retries=2,
            verify_command=("true",),
            metric=SimpleNamespace(goal="minimize"),
        ),
    )
    wf = _wf(
        root=tmp_path,
        config=config,
        provider=provider,
        dispatcher=dispatcher,
        budget=None,
        max_iterations=20,
        loop_guard_kill_threshold=0,
    )
    messages = [{"role": "user", "content": [{"type": "text", "text": "TASK:\noptimize"}]}]
    result = wf._drive_loop(  # pyright: ignore[reportPrivateUsage]
        system="system",
        conversation=Conversation.from_wire(messages),
        tool_calls=0,
        start_iteration=1,
        root_task_id=None,
        original_task="t",
    )
    assert result.reason == "finish_session"
    assert dispatcher.calls == ["finish_session"]
    assert result.tool_calls == 1


def test_metric_at_fraction_ceiling_detects_maxed_score() -> None:
    from agent6.workflows._metric import (
        metric_at_fraction_ceiling as _metric_at_fraction_ceiling,
    )

    # Maxed-out fraction: numerator == score == denominator.
    assert _metric_at_fraction_ceiling("SCORE: 27/27\n", 27.0, pattern=r"SCORE: (\d+)") is True
    assert _metric_at_fraction_ceiling("passed 5 / 5 checks", 5.0, pattern=r"passed (\d+)") is True
    # Partial score is not the ceiling.
    assert _metric_at_fraction_ceiling("SCORE: 26/27\n", 26.0, pattern=r"SCORE: (\d+)") is False
    # Score that does not match the numerator is ignored.
    assert _metric_at_fraction_ceiling("SCORE: 27/27\n", 26.0, pattern=r"SCORE: (\d+)") is False
    # Unbounded metric (raw count, no denominator) never trips the ceiling.
    assert _metric_at_fraction_ceiling("CYCLES: 1487\n", 1487.0, pattern=r"CYCLES: (\d+)") is False


def test_metric_at_fraction_ceiling_scans_only_the_score_line() -> None:
    from agent6.workflows._metric import (
        metric_at_fraction_ceiling as _metric_at_fraction_ceiling,
    )

    # tqdm progress in stderr prints an incidental 100/100 equal to the score;
    # the real score line has no denominator, so the ceiling must NOT latch.
    text = "SCORE: 100\n100%|##########| 100/100 [00:03<00:00, 33.1it/s]\n"
    assert _metric_at_fraction_ceiling(text, 100.0, pattern=r"SCORE: (\d+)") is False
    # A genuine maxed fraction ON the score-pattern line still trips it.
    assert _metric_at_fraction_ceiling("junk 3/3\nSCORE: 27/27\n", 27.0, pattern=r"SCORE: (\d+)")
    # No score-pattern match at all -> conservative False.
    assert _metric_at_fraction_ceiling("100/100\n", 100.0, pattern=r"SCORE: (\d+)") is False


def test_drive_loop_honors_finish_at_metric_ceiling(tmp_path: Path) -> None:
    """A finish_session on a maximize metric that is already at its provable
    ceiling (SCORE: N/N) is honoured immediately — even with most of the
    budget unspent — instead of being rejected and nudged. This is the guard
    against weak models burning their whole budget re-deriving a solved task.
    """
    from agent6.budget import BudgetTracker

    class ProviderStub:
        def __init__(self) -> None:
            self.calls = 0
            self.finish_nudges_seen = 0

        def call(self, **kwargs: Any) -> ProviderResponse:
            self.calls += 1
            rendered = str(kwargs["messages"][-1])
            if "[harness budget]" in rendered:
                self.finish_nudges_seen += 1
            # First turn: pass verify (auto-metric will report the ceiling).
            # Subsequent turns: try to finish.
            if self.calls == 1:
                return _tool_resp("run_verify_command", tool_id=f"verify-{self.calls}")
            return _tool_resp(
                "finish_session",
                {"summary": f"done-{self.calls}"},
                tool_id=f"finish-{self.calls}",
            )

    class DispatcherStub(_StubDispatcher):
        def dispatch(self, name: str, raw_input: dict[str, Any]) -> ToolResult:
            del raw_input
            if name == "run_verify_command":
                return ExecResult(
                    returncode=0, stdout="", stderr="", duration_s=0.1, exec_failed=False
                )
            if name == "run_metric_command":
                return MetricResult(
                    returncode=0,
                    stdout="SCORE: 27/27\n",
                    stderr="",
                    duration_s=0.1,
                    exec_failed=False,
                    score=27.0,
                )
            return RawResult({"ok": True})

    provider = ProviderStub()
    config = SimpleNamespace(
        git=_GIT_STUB,
        budget=SimpleNamespace(max_usd=10.0, max_tokens_fallback=2_000_000),
        workflow=SimpleNamespace(
            verify_when="never",
            verify_retries=2,
            verify_command=("true",),
            metric=SimpleNamespace(goal="maximize", pattern=r"SCORE:\s*([\d.]+)"),
        ),
    )
    # Huge ceilings keep fraction_remaining ~1.0: without the ceiling guard
    # the early-finish guard would reject the finish here.
    budget = BudgetTracker(max_usd=-1, max_tokens_fallback=-1, max_percent=-1)
    wf = _wf(
        root=tmp_path,
        config=config,
        provider=provider,
        dispatcher=DispatcherStub(),
        budget=budget,
        max_iterations=20,
        loop_guard_kill_threshold=0,
    )
    messages = [{"role": "user", "content": [{"type": "text", "text": "TASK:\noptimize"}]}]

    with patch(
        "agent6.workflows.loop.chain_commit",
        side_effect=[f"sha{i}" for i in range(1, 22)],
    ):
        result = wf._drive_loop(  # pyright: ignore[reportPrivateUsage]
            system="system",
            conversation=Conversation.from_wire(messages),
            tool_calls=0,
            start_iteration=1,
            root_task_id=None,
            original_task="t",
        )

    # Honoured on the very first finish_session, with no budget nudges.
    assert result.reason == "finish_session"
    assert provider.finish_nudges_seen == 0
    assert provider.calls == 2


# --- tier-aware metric targets --------------------------------------------


def test_extract_metric_targets_ignores_arrow_output() -> None:
    """Grader progress arrows ('epoch 2 -> 27.0') are not thresholds: the bare
    '>' alternative also matched the second char of '->', fabricating an
    unmeetable 'drive the metric above <current>' directive from the grader's
    own echo of the score."""
    from agent6.workflows._metric import (
        extract_metric_targets as _extract_metric_targets,
    )

    text = "epoch 1 -> 12.0\nepoch 2 -> 27.0\nbest => 30\nSCORE: 27\n"
    assert _extract_metric_targets(text, goal="maximize") == ()
    # Real assert-style thresholds still extract.
    assert _extract_metric_targets("assert score > 25", goal="maximize") == (25.0,)


def test_extract_metric_targets_minimize_picks_upper_bounds() -> None:
    from agent6.workflows._metric import (
        extract_metric_targets as _extract_metric_targets,
    )

    text = (
        "assert cycles() < 18532\n"
        "assert cycles() < 1_487\n"
        "assert cycles() < 1579\n"
        "some unrelated > 99 noise\n"
    )
    targets = _extract_metric_targets(text, goal="minimize")
    # Only `<`/`<=` bounds, de-duplicated, order preserved.
    assert targets == (18532.0, 1487.0, 1579.0)


def test_extract_metric_targets_maximize_picks_lower_bounds() -> None:
    from agent6.workflows._metric import (
        extract_metric_targets as _extract_metric_targets,
    )

    text = "assert score > 0.80\nassert score >= 0.95\nassert other < 5\n"
    targets = _extract_metric_targets(text, goal="maximize")
    assert targets == (0.80, 0.95)


def test_next_metric_target_minimize_returns_nearest_unmet() -> None:
    from agent6.workflows._metric import next_metric_target as _next_metric_target

    targets = (147734.0, 18532.0, 1579.0, 1487.0)
    # At 8256 we've cleared 18532/147734; nearest unmet is the largest
    # threshold still below the current score.
    assert _next_metric_target(targets, 8256.0, "minimize") == 1579.0
    # Once under everything, no target remains.
    assert _next_metric_target(targets, 1000.0, "minimize") is None


def test_next_metric_target_maximize_returns_nearest_unmet() -> None:
    from agent6.workflows._metric import next_metric_target as _next_metric_target

    targets = (0.50, 0.80, 0.95)
    assert _next_metric_target(targets, 0.83, "maximize") == 0.95
    assert _next_metric_target(targets, 0.99, "maximize") is None


def test_next_metric_target_equality_is_unmet() -> None:
    # Thresholds are harvested from strict comparisons (`assert x < N`), which
    # still FAIL at x == N; a score sitting exactly on the threshold has not
    # met it and must keep it as the next target.
    from agent6.workflows._metric import next_metric_target as _next_metric_target

    assert _next_metric_target((1487.0,), 1487.0, "minimize") == 1487.0
    assert _next_metric_target((0.95,), 0.95, "maximize") == 0.95
    # Strictly beyond the threshold in the improving direction -> met.
    assert _next_metric_target((1487.0,), 1486.0, "minimize") is None
    assert _next_metric_target((0.95,), 0.96, "maximize") is None


def test_format_metric_feedback_shows_next_target() -> None:
    from agent6.workflows._metric import (
        MetricSample as _MetricSample,
    )
    from agent6.workflows._metric import (
        format_metric_feedback as _format_metric_feedback,
    )

    history = [
        _MetricSample(label="a", score=20000.0, returncode=0),
        _MetricSample(
            label="b",
            score=8256.0,
            returncode=0,
            targets=(18532.0, 1579.0, 1487.0),
        ),
    ]
    text = _format_metric_feedback(history, goal="minimize")
    assert "next target: drive the metric below 1579" in text
    assert "current 8256" in text


def test_worker_max_tokens_lifts_cap_on_metric_runs() -> None:
    config = SimpleNamespace(
        git=_GIT_STUB,
        budget=SimpleNamespace(max_usd=10.0, max_tokens_fallback=2_000_000),
        workflow=SimpleNamespace(
            verify_when="never",
            verify_retries=2,
            verify_command=("true",),
            metric=SimpleNamespace(goal="minimize"),
        ),
    )
    wf = _wf(
        config=config,
        mode="run",
        per_call_max_tokens=16384,
        metric_task_max_tokens=32768,
    )
    assert wf._worker_max_tokens(_state()) == 32768  # pyright: ignore[reportPrivateUsage]


def test_worker_max_tokens_keeps_default_without_metric() -> None:
    config = SimpleNamespace(
        git=_GIT_STUB,
        budget=SimpleNamespace(max_usd=10.0, max_tokens_fallback=2_000_000),
        workflow=SimpleNamespace(
            verify_when="never",
            verify_retries=2,
            verify_command=("true",),
            metric=SimpleNamespace(goal=None),
        ),
    )
    wf = _wf(
        config=config,
        mode="run",
        per_call_max_tokens=16384,
        metric_task_max_tokens=32768,
    )
    assert wf._worker_max_tokens(_state()) == 16384  # pyright: ignore[reportPrivateUsage]


def test_worker_max_tokens_keeps_default_in_plan_mode() -> None:
    config = SimpleNamespace(
        git=_GIT_STUB,
        budget=SimpleNamespace(max_usd=10.0, max_tokens_fallback=2_000_000),
        workflow=SimpleNamespace(
            verify_when="never",
            verify_retries=2,
            verify_command=("true",),
            metric=SimpleNamespace(goal="minimize"),
        ),
    )
    wf = _wf(
        config=config,
        mode="plan",
        per_call_max_tokens=16384,
        metric_task_max_tokens=32768,
    )
    assert wf._worker_max_tokens(_state()) == 16384  # pyright: ignore[reportPrivateUsage]


# --- tier-2 summarise-and-restart compaction ------------------------------


def _long_history(n_pairs: int) -> list[dict[str, Any]]:
    """An original task message followed by ``n_pairs`` assistant tool_use /
    user tool_result turns with bulky payloads."""
    msgs: list[dict[str, Any]] = [
        {"role": "user", "content": [{"type": "text", "text": "TASK:\noptimize the kernel"}]}
    ]
    for i in range(n_pairs):
        msgs.append(
            {
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "id": f"t{i}", "name": "read_file", "input": {"i": i}}
                ],
            }
        )
        msgs.append(
            {
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": f"t{i}", "content": "X" * 5000}],
            }
        )
    return msgs


def _restart_via_wire(wf: Workflow, messages: list[dict[str, Any]], *, state: Any = None) -> None:
    conversation = Conversation.from_wire(messages)
    wf._summarise_and_restart(  # pyright: ignore[reportPrivateUsage]
        conversation, state if state is not None else _state()
    )
    messages[:] = conversation.to_wire()


def _compact_via_wire(wf: Workflow, messages: list[dict[str, Any]], *, state: Any = None) -> bool:
    conversation = Conversation.from_wire(messages)
    out = wf._maybe_compact(  # pyright: ignore[reportPrivateUsage]
        conversation, state if state is not None else _state()
    )
    messages[:] = conversation.to_wire()
    return out


def _read_history(*reads: tuple[str, str]) -> list[dict[str, Any]]:
    """An original task message plus one read_file exchange per (path, content)."""
    msgs: list[dict[str, Any]] = [
        {"role": "user", "content": [{"type": "text", "text": "TASK:\nt"}]}
    ]
    for i, (path, content) in enumerate(reads):
        msgs.append(
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": f"t{i}",
                        "name": "read_file",
                        "input": {"path": path},
                    }
                ],
            }
        )
        msgs.append(
            {
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": f"t{i}", "content": content}],
            }
        )
    return msgs


def test_tier1_compact_event_names_what_was_elided() -> None:
    """loop.compact.dropped carries the elided call identities, not just a count."""
    ev = _EventCapture()
    wf = _wf(
        events=ev,
        compact_drop_at_chars=1500,
        compact_summarise_at_chars=10**9,
        compact_elision_gists=False,
    )
    msgs = _read_history(("a.py", "X" * 1000), ("b.py", "X" * 1000), ("c.py", "X" * 1000))
    _compact_via_wire(wf, msgs)
    dropped = [e for e in ev.events if e["type"] == "loop.compact.dropped"]
    assert dropped and dropped[-1]["calls"] == ["read_file a.py"]


def test_tier1_gist_event_carries_paths() -> None:
    import json

    ev = _EventCapture()
    summariser = MagicMock()
    summariser.call.return_value = _resp("docs/spec.md: spec facts distilled")
    wf = _wf(
        events=ev,
        summariser_provider=summariser,
        compact_drop_at_chars=1800,
        compact_summarise_at_chars=10**9,
        compact_elision_gists=True,
    )
    doc = json.dumps({"content": "authoritative spec. " * 300})
    msgs = _read_history(("docs/spec.md", doc), ("b.py", "x" * 500), ("c.py", "y" * 500))
    _compact_via_wire(wf, msgs)
    gists = [e for e in ev.events if e["type"] == "loop.compact.gists"]
    assert gists
    assert gists[-1]["paths"] == ["docs/spec.md"]
    assert gists[-1]["demoted_paths"] == []


def test_summarise_done_event_carries_summary_text() -> None:
    """The full summary rides the event so surfaces can show what the model now
    works from; summary_chars alone hid the post-restart worldview."""
    ev = _EventCapture()
    summariser = MagicMock()
    summariser.call.return_value = _resp("done: tried A; best=42 at sha9")
    wf = _wf(events=ev, summariser_provider=summariser)
    _restart_via_wire(wf, _long_history(6))
    done = [e for e in ev.events if e["type"] == "loop.compact.summarise.done"]
    assert done and done[-1]["summary"] == "done: tried A; best=42 at sha9"


def test_forced_compact_threads_focus_to_summariser() -> None:
    """/compact <focus>: the marker's text reaches the summariser prompt and the
    loop.compact.requested event, so the operator steers WHAT the summary keeps."""
    ev = _EventCapture()
    summariser = MagicMock()
    summariser.call.return_value = _resp("s")
    cleared: list[bool] = []
    wf = _wf(
        events=ev,
        summariser_provider=summariser,
        compact_requested=lambda: "weigh the auth decisions",
        compact_clear=lambda: cleared.append(True),
        compact_drop_at_chars=10**9,
        compact_summarise_at_chars=10**9,
    )
    assert _compact_via_wire(wf, _long_history(3)) is True
    assert cleared == [True]
    sent = str(summariser.call.call_args)
    assert "Operator focus for this summary" in sent
    assert "weigh the auth decisions" in sent
    req = [e for e in ev.events if e["type"] == "loop.compact.requested"]
    assert req and req[-1]["focus"] == "weigh the auth decisions"


def test_forced_compact_below_the_floor_says_it_was_refused() -> None:
    """The history floor is deliberate -- a restart below it loses more than it
    saves -- but the request was consumed and cleared with no second event: the
    front-end had already said "applies before the next model call", so the
    operator saw a success toast, lost the focus text, and had to guess."""
    ev = _EventCapture()
    summariser = MagicMock()
    cleared: list[bool] = []
    wf = _wf(
        events=ev,
        summariser_provider=summariser,
        compact_requested=lambda: "keep the auth work",
        compact_clear=lambda: cleared.append(True),
        compact_drop_at_chars=10**9,
        compact_summarise_at_chars=10**9,
    )
    assert _compact_via_wire(wf, _long_history(1)) is False
    assert cleared == [True]
    summariser.call.assert_not_called()
    refused = [e for e in ev.events if e["type"] == "loop.compact.refused"]
    assert refused, "the consumed request must say why nothing happened"
    assert "history" in str(refused[-1].get("reason", ""))


def test_forced_compact_plain_keeps_prompt_unfocused() -> None:
    """A plain /compact ("" focus) forces tier-2 with the byte-identical
    summariser prompt of an automatic tier-2 (no focus clause)."""
    summariser = MagicMock()
    summariser.call.return_value = _resp("s")
    wf = _wf(
        summariser_provider=summariser,
        compact_requested=lambda: "",
        compact_drop_at_chars=10**9,
        compact_summarise_at_chars=10**9,
    )
    assert _compact_via_wire(wf, _long_history(3)) is True
    assert "Operator focus" not in str(summariser.call.call_args)


def test_summarise_and_restart_reinjects_pins_verbatim() -> None:
    """Pins are re-shown verbatim in the restart message (before the summary
    label, as standing orders), and the summariser is told not to restate them."""
    summariser = MagicMock()
    summariser.call.return_value = _resp("progress summary text")
    wf = _wf(summariser_provider=summariser)
    st = _state(pins=["never touch schema files", "goal:\nship X"])
    messages = _long_history(6)
    _restart_via_wire(wf, messages, state=st)
    text = messages[1]["content"][0]["text"]
    assert "PINNED operator instructions (verbatim):" in text
    assert "1. never touch schema files" in text
    assert "2. goal:\nship X" in text
    assert text.index("PINNED operator") < text.index("PROGRESS SUMMARY:")
    assert "do NOT restate" in str(summariser.call.call_args)


def test_summarise_and_restart_replaces_history() -> None:
    summariser = MagicMock()
    summariser.call.return_value = _resp("done: tried A (kept), B (reverted); best=42 at sha9")
    wf = _wf(summariser_provider=summariser)
    messages = _long_history(6)
    original = messages[0]

    _restart_via_wire(wf, messages)

    # Collapsed to (original task, restart-with-summary).
    assert len(messages) == 2
    assert messages[0] == original
    text = messages[1]["content"][0]["text"]
    assert "[harness context restart]" in text
    assert "best=42 at sha9" in text
    # The summariser saw the worker provider's content, not the worker itself.
    summariser.call.assert_called_once()


def test_summarise_and_restart_applies_dag_checkoff() -> None:
    """At tier-2 compaction agent6 asks the summariser which tasks finished and
    applies it to the curator (passes completed, queues discovered), strips the
    bookkeeping block from the restart, and ignores hallucinated task ids."""

    class _FakeClient:
        def __init__(self) -> None:
            self._nodes = {
                "01ROOT": {"parent_id": None, "status": "in_progress", "title": "review repo"},
                "01DONE": {"parent_id": "01ROOT", "status": "pending", "title": "audit providers"},
                "01OPEN": {"parent_id": "01ROOT", "status": "pending", "title": "audit sandbox"},
            }
            self.passed: list[str] = []
            self.added: list[tuple[str | None, str]] = []

        def nodes(self) -> dict[str, Any]:
            return _typed(self._nodes)

        def update_status(self, intent: Any) -> None:
            self.passed.append(intent.id)
            self._nodes[intent.id]["status"] = intent.new_status

        def add_subtask(self, intent: Any) -> Any:
            self.added.append((intent.parent_id, intent.draft.title))
            return MagicMock()

    fake = _FakeClient()
    summariser = MagicMock()
    summariser.call.return_value = _resp(
        "Progress: finished the providers audit.\n\n"
        '```checkoff\n{"completed_ids": ["01DONE", "01HALLUCINATED"], '
        '"new_tasks": ["fix the budget rounding bug"]}\n```'
    )
    wf = _wf(summariser_provider=summariser, curator=fake)
    messages = _long_history(6)
    _restart_via_wire(wf, messages)

    assert fake.passed == ["01DONE"]  # valid completed id passed; hallucinated id ignored
    assert fake.added == [("01ROOT", "fix the budget rounding bug")]  # queued under the root
    restart_text = messages[1]["content"][0]["text"]
    assert "providers audit" in restart_text
    assert "checkoff" not in restart_text  # bookkeeping block stripped from the restart


class _FakeGraph:
    def __init__(self, nodes: dict[str, dict[str, Any]]) -> None:
        self._nodes = nodes

    def nodes(self) -> dict[str, Any]:
        return _typed(self._nodes)


def test_task_finish_gate_nudges_open_subtasks_then_caps() -> None:
    """The finish-gate nudges while a SUBTASK is open, naming it, and stops after
    _TASK_FINISH_PATIENCE so a stuck worker can't bounce the loop forever."""
    from agent6.workflows.loop import TASK_FINISH_PATIENCE  # pyright: ignore[reportPrivateUsage]

    nodes = {
        "root": {"parent_id": None, "status": "in_progress", "title": "review repo"},
        "sub1": {"parent_id": "root", "status": "pending", "title": "audit providers"},
        "sub2": {"parent_id": "root", "status": "passed", "title": "audit sandbox"},  # done
    }
    wf = _wf(curator=_FakeGraph(nodes))
    st = _state()
    for i in range(1, TASK_FINISH_PATIENCE + 1):
        nudge = wf._task_finish_gate_nudge(st)  # pyright: ignore[reportPrivateUsage]
        assert nudge is not None and "audit providers" in nudge
        assert "audit sandbox" not in nudge  # passed subtask not listed
        assert st.task_finish_nudges_used == i
    # Cap reached -> finish is honoured (no further nudges).
    assert wf._task_finish_gate_nudge(st) is None  # pyright: ignore[reportPrivateUsage]


def test_task_finish_gate_allows_finish_without_open_subtasks() -> None:
    """Only SUBTASKS gate. The always-pending auto-root alone must NOT block a
    finish (else every run deadlocks); no curator -> no gate either."""
    root_only = _FakeGraph({"root": {"parent_id": None, "status": "pending", "title": "t"}})
    assert _wf(curator=root_only)._task_finish_gate_nudge(_state()) is None  # pyright: ignore[reportPrivateUsage]
    assert _wf(curator=None)._task_finish_gate_nudge(_state()) is None  # pyright: ignore[reportPrivateUsage]


# --- surface-current-task -------------------------------------------------


def test_current_task_id_prefers_open_cursor() -> None:
    """The cursor wins when it still points at an open subtask, even if an
    earlier subtask is also open (the worker's explicit focus choice is kept)."""
    from agent6.workflows.loop import current_task_id  # pyright: ignore[reportPrivateUsage]

    nodes = {
        "root": {"parent_id": None, "status": "in_progress", "title": "r"},
        "a": {"parent_id": "root", "status": "pending", "title": "a"},
        "b": {"parent_id": "root", "status": "in_progress", "title": "b"},
    }
    assert current_task_id(_typed(nodes), "b") == "b"  # cursor respected
    assert current_task_id(_typed(nodes), None) == "a"  # no cursor -> first open subtask
    # Stale cursor (points at a closed task) -> recompute the frontier.
    nodes["b"]["status"] = "passed"
    assert current_task_id(_typed(nodes), "b") == "a"
    # Cursor on the auto-root is not a focus target -> first open subtask.
    assert current_task_id(_typed(nodes), "root") == "a"


def test_first_ready_subtask_respects_deps_and_order() -> None:
    """The frontier skips a subtask whose dependency is not yet done, and a
    passed/obsolete dependency unblocks it; roots and done tasks never surface."""
    from agent6.workflows._dag_focus import first_ready_subtask as _first_ready_subtask

    nodes = {
        "root": {"parent_id": None, "status": "in_progress", "title": "r"},
        "a": {"parent_id": "root", "status": "passed", "title": "a"},  # done
        "b": {"parent_id": "root", "status": "pending", "title": "b", "depends_on": ["c"]},
        "c": {"parent_id": "root", "status": "pending", "title": "c"},
    }
    # b is blocked on c (pending) -> c is the first ready subtask.
    assert _first_ready_subtask(_typed(nodes)) == "c"
    # Once c is done, b unblocks.
    nodes["c"]["status"] = "obsolete"
    assert _first_ready_subtask(_typed(nodes)) == "b"
    # Everything done -> nothing ready (the finish-gate, not this, ends the run).
    nodes["b"]["status"] = "passed"
    assert _first_ready_subtask(_typed(nodes)) is None


def test_first_ready_subtask_prefers_leaf_over_decomposed_parent() -> None:
    """A subtask with open children is a container -- the frontier surfaces its
    first ready leaf, not the parent, so a decompose moves focus forward. A cursor
    still pointing at the parent falls through to the leaf too."""
    from agent6.workflows._dag_focus import (
        current_task_id as _current_task_id,
    )
    from agent6.workflows._dag_focus import (
        first_ready_subtask as _first_ready_subtask,
    )

    nodes = {
        "root": {"parent_id": None, "status": "in_progress", "title": "r", "children": ["a", "b"]},
        "a": {"parent_id": "root", "status": "in_progress", "title": "a", "children": ["a1", "a2"]},
        "a1": {"parent_id": "a", "status": "pending", "title": "a1"},
        "a2": {"parent_id": "a", "status": "pending", "title": "a2"},
        "b": {"parent_id": "root", "status": "pending", "title": "b"},
    }
    assert _first_ready_subtask(_typed(nodes)) == "a1"  # the parent 'a' is skipped as a container
    assert _current_task_id(_typed(nodes), "a") == "a1"  # stale cursor on the parent falls through
    # Once the children are done, the parent becomes a focusable leaf again.
    nodes["a1"]["status"] = "passed"
    nodes["a2"]["status"] = "passed"
    assert _first_ready_subtask(_typed(nodes)) == "a"


def test_current_task_banner_carries_title_acceptance_paths() -> None:
    from agent6.workflows.loop import current_task_banner  # pyright: ignore[reportPrivateUsage]

    banner = current_task_banner(
        "01TASK",
        _tn("01TASK", title="audit providers", acceptance="no bugs left", relevant_paths=("a.py",)),
    )
    assert "Current task (01TASK): audit providers" in banner
    assert "Acceptance: no bugs left" in banner
    assert "Relevant paths: a.py" in banner
    assert "ONE task to completion" in banner
    # Absent acceptance/paths are simply omitted, not rendered empty.
    bare = current_task_banner("01X", _tn("01X", title="t"))
    assert "Acceptance:" not in bare and "Relevant paths:" not in bare
    # Decompose invites a finer plan for a large childless task (recursion);
    # off by default, and never once the task already has children.
    assert "child subtasks" not in bare
    rec = current_task_banner("01X", _tn("01X", title="t"), decompose=True)
    assert "child subtasks under it (parent_id=01X)" in rec
    has_kids = current_task_banner("01X", _tn("01X", title="t", children=("01Y",)), decompose=True)
    assert "child subtasks" not in has_kids


def test_graph_update_snapshot_payload_is_wire_stable(tmp_path: Path) -> None:
    """FROZEN wire surface: the graph.update event the loop emits (consumed by
    the viewmodel fold, web and TUI, and on-disk in old run dirs) projects each
    node to exactly {title, status, parent_id, children} plus a top-level
    cursor, with children a JSON list. Interface-independent: drives a real
    curator + real Workflow, so it pins the emitted bytes regardless of how the
    curator hands state to the loop internally."""
    from agent6.graph.curator import GraphCurator
    from agent6.graph.models import (
        AddSubtaskIntent,
        SetCursorIntent,
        TaskNodeDraft,
        UpdateStatusIntent,
    )
    from agent6.sessions.layout import SessionLayout

    cur = GraphCurator(SessionLayout(state_dir=tmp_path / ".agent6", session_id="run1"))
    root = cur.add_subtask(
        AddSubtaskIntent(parent_id=None, draft=TaskNodeDraft(title="root", created_by="planner"))
    )
    child = cur.add_subtask(
        AddSubtaskIntent(parent_id=root.id, draft=TaskNodeDraft(title="child", created_by="worker"))
    )
    cur.update_status(UpdateStatusIntent(id=child.id, new_status="in_progress"))
    cur.set_cursor(SetCursorIntent(id=child.id))

    captured: list[tuple[str, dict[str, Any]]] = []

    class _Events:
        def emit(self, event_type: str, /, **fields: Any) -> None:
            captured.append((event_type, fields))

    wf = _wf(curator=cur, events=_Events())
    wf._emit_graph_snapshot()  # pyright: ignore[reportPrivateUsage]

    assert len(captured) == 1
    etype, fields = captured[0]
    assert etype == "graph.update"
    assert fields == {
        "nodes": {
            root.id: {
                "title": "root",
                "status": "pending",
                "parent_id": None,
                "children": [child.id],
            },
            child.id: {
                "title": "child",
                "status": "in_progress",
                "parent_id": root.id,
                "children": [],
            },
        },
        "cursor": child.id,
    }
    # children must serialize as a JSON list (model_dump(mode="json") shape), not
    # a tuple -- the frozen on-disk/wire contract old run dirs already hold.
    assert isinstance(fields["nodes"][root.id]["children"], list)


def test_decompose_prompt_describes_nested_phases() -> None:
    from agent6.prompts.loop import DAG_RULES_DECOMPOSE, dag_rules_block

    assert dag_rules_block(True) == DAG_RULES_DECOMPOSE
    # Phases with child subtasks (parent_id), and the re-plan-when-large rule.
    assert "phases" in DAG_RULES_DECOMPOSE.lower()
    assert "parent_id" in DAG_RULES_DECOMPOSE
    assert "large" in DAG_RULES_DECOMPOSE.lower()


class _FakeCurator:
    """In-memory GraphCurator stand-in: nodes / cursor / set_cursor / update_status."""

    def __init__(self, nodes: dict[str, dict[str, Any]], cursor: str | None = None) -> None:
        self._nodes = nodes
        self._cursor = cursor
        self.cursor_sets: list[str | None] = []
        self.status_sets: list[tuple[str, str]] = []

    def nodes(self) -> dict[str, Any]:
        return _typed(self._nodes)

    def cursor(self) -> str | None:
        return self._cursor

    def set_cursor(self, intent: Any) -> None:
        self._cursor = intent.id
        self.cursor_sets.append(intent.id)

    def update_status(self, intent: Any) -> None:
        self.status_sets.append((intent.id, intent.new_status))
        self._nodes[intent.id]["status"] = intent.new_status


def _surface(wf: Workflow, st: Any, messages: list[dict[str, Any]]) -> None:
    """Wire-in/wire-out driver so the tests keep asserting on message dicts."""
    conversation = Conversation.from_wire(messages)
    wf._maybe_surface_current_task(conversation, st)  # pyright: ignore[reportPrivateUsage]
    messages[:] = conversation.to_wire()


def test_surface_current_task_surfaces_advances_then_quiets() -> None:
    """First call surfaces the focus banner, advances the cursor onto the task,
    and marks it in_progress; a repeat call with the same focus stays quiet (the
    banner survives tier-1 elision); marking it passed advances to the next."""
    nodes = {
        "root": {"parent_id": None, "status": "in_progress", "title": "review repo"},
        "a": {"parent_id": "root", "status": "pending", "title": "audit providers"},
        "b": {"parent_id": "root", "status": "pending", "title": "audit sandbox"},
    }
    cur = _FakeCurator(nodes)
    wf = _wf(curator=cur)
    st = _state()
    messages: list[dict[str, Any]] = []

    _surface(wf, st, messages)
    assert len(messages) == 1
    assert "audit providers" in messages[0]["content"][0]["text"]
    assert cur.cursor_sets == ["a"]  # cursor advanced onto the focus task
    assert cur.status_sets == [("a", "in_progress")]  # reflected as being worked
    assert st.surfaced_task_id == "a"

    # Same focus -> no new banner, no redundant cursor/status writes.
    _surface(wf, st, messages)
    assert len(messages) == 1
    assert cur.cursor_sets == ["a"]
    assert cur.status_sets == [("a", "in_progress")]  # no second write for the same task

    # Worker finishes task a -> next turn focus advances to b.
    nodes["a"]["status"] = "passed"
    _surface(wf, st, messages)
    assert len(messages) == 2
    assert "audit sandbox" in messages[1]["content"][0]["text"]
    assert cur.cursor_sets == ["a", "b"]
    assert cur.status_sets == [("a", "in_progress"), ("b", "in_progress")]
    assert st.surfaced_task_id == "b"


def test_surface_current_task_skips_status_write_when_already_in_progress() -> None:
    """The in_progress-only guard: a current task already in_progress is surfaced
    WITHOUT a redundant update_status write (only pending -> in_progress writes).
    Pins the negative branch of the sole conditional curator write."""
    cur = _FakeCurator(
        {
            "root": {"parent_id": None, "status": "in_progress", "title": "r"},
            "a": {"parent_id": "root", "status": "in_progress", "title": "audit providers"},
        },
        cursor="a",
    )
    wf = _wf(curator=cur)
    messages: list[dict[str, Any]] = []
    _surface(wf, _state(), messages)
    assert len(messages) == 1  # banner still surfaced
    assert cur.status_sets == []  # already in_progress -> no redundant status write
    assert cur.cursor_sets == []  # cursor already on it -> no redundant set_cursor


def test_surface_current_task_resurfaces_after_compaction_reset() -> None:
    """A tier-2 restart resets surfaced_task_id to None; the next surface call
    re-injects the focus banner into the fresh context."""
    nodes = {
        "root": {"parent_id": None, "status": "in_progress", "title": "r"},
        "a": {"parent_id": "root", "status": "pending", "title": "audit providers"},
    }
    wf = _wf(curator=_FakeCurator(nodes))
    st = _state()
    messages: list[dict[str, Any]] = []
    _surface(wf, st, messages)
    assert len(messages) == 1
    st.surfaced_task_id = None  # what the loop does on a tier-2 restart
    _surface(wf, st, messages)
    assert len(messages) == 2  # re-surfaced after the restart wiped the banner


def test_surface_current_task_noop_cases() -> None:
    """No-op without open subtasks (root only), without a curator, or outside run
    mode -- nothing is appended and no cursor/status write happens."""
    root_only = _FakeCurator({"root": {"parent_id": None, "status": "pending", "title": "t"}})
    msgs: list[dict[str, Any]] = []
    _surface(_wf(curator=root_only), _state(), msgs)
    assert msgs == [] and root_only.cursor_sets == []

    _surface(_wf(curator=None), _state(), msgs)
    assert msgs == []

    open_sub = _FakeCurator(
        {
            "root": {"parent_id": None, "status": "pending", "title": "t"},
            "a": {"parent_id": "root", "status": "pending", "title": "a"},
        }
    )
    _surface(_wf(curator=open_sub, mode="plan"), _state(), msgs)
    assert msgs == [] and open_sub.cursor_sets == []  # plan mode does not surface


def _stuck_count(messages: list[dict[str, Any]]) -> int:
    return sum(1 for m in messages if "without concluding it" in m["content"][0]["text"])


def test_surface_current_task_stuck_nudge_fires_periodically_then_caps() -> None:
    """The split/pass/skip nudge re-fires every _STUCK_ON_TASK_AFTER turns on the
    same stuck task (a weak model ignored a single nudge live), but caps at
    _STUCK_NUDGE_MAX so it cannot nag forever."""
    from agent6.workflows.loop import (
        STUCK_NUDGE_MAX,  # pyright: ignore[reportPrivateUsage]
        STUCK_ON_TASK_AFTER,  # pyright: ignore[reportPrivateUsage]
    )

    cur = _FakeCurator(
        {
            "root": {"parent_id": None, "status": "in_progress", "title": "r"},
            "a": {"parent_id": "root", "status": "pending", "title": "audit providers"},
        }
    )
    wf = _wf(curator=cur)
    st = _state()
    messages: list[dict[str, Any]] = []
    # One nudge after the first period, but not before it.
    for _ in range(STUCK_ON_TASK_AFTER):
        _surface(wf, st, messages)
    assert _stuck_count(messages) == 0  # turns_on_task is _STUCK_ON_TASK_AFTER-1 here
    _surface(wf, st, messages)
    assert _stuck_count(messages) == 1  # crossed the first period
    # Keep grinding well past the cap; it re-fires periodically then stops.
    for _ in range((STUCK_NUDGE_MAX + 2) * STUCK_ON_TASK_AFTER):
        _surface(wf, st, messages)
    assert _stuck_count(messages) == STUCK_NUDGE_MAX
    assert st.stuck_nudges_fired == STUCK_NUDGE_MAX


def test_surface_current_task_stuck_nudge_resets_on_progress() -> None:
    """Forward motion (a task marked passed -> focus advances) resets the grind
    counter, so the stuck nudge does not fire."""
    from agent6.workflows.loop import STUCK_ON_TASK_AFTER  # pyright: ignore[reportPrivateUsage]

    nodes = {
        "root": {"parent_id": None, "status": "in_progress", "title": "r"},
        "a": {"parent_id": "root", "status": "pending", "title": "a"},
        "b": {"parent_id": "root", "status": "pending", "title": "b"},
    }
    wf = _wf(curator=_FakeCurator(nodes))
    st = _state()
    messages: list[dict[str, Any]] = []
    for _ in range(STUCK_ON_TASK_AFTER - 1):  # grind almost to the threshold on a
        _surface(wf, st, messages)
    assert _stuck_count(messages) == 0
    nodes["a"]["status"] = "passed"  # progress -> focus advances to b
    for _ in range(3):
        _surface(wf, st, messages)
    assert _stuck_count(messages) == 0
    assert st.last_focus_id == "b" and st.turns_on_task < STUCK_ON_TASK_AFTER


def test_surface_current_task_stuck_counter_survives_compaction() -> None:
    """A tier-2 restart resets the banner (surfaced_task_id) but NOT the grind
    counter -- compaction is not progress on the task."""
    wf = _wf(
        curator=_FakeCurator(
            {
                "root": {"parent_id": None, "status": "in_progress", "title": "r"},
                "a": {"parent_id": "root", "status": "pending", "title": "a"},
            }
        )
    )
    st = _state()
    messages: list[dict[str, Any]] = []
    for _ in range(5):
        _surface(wf, st, messages)
    assert st.turns_on_task == 4
    st.surfaced_task_id = None  # what the loop does on a tier-2 restart
    _surface(wf, st, messages)
    assert st.turns_on_task == 5  # kept climbing across the restart
    assert st.last_focus_id == "a"


def test_surface_decompose_resets_grind_counter() -> None:
    """Obeying the nudge -- decomposing the focus task with add_task -- moves focus
    to the first new leaf and resets the grind counter (the fix for the
    self-defeating-nudge bug)."""
    nodes: dict[str, dict[str, Any]] = {
        "root": {"parent_id": None, "status": "in_progress", "title": "r", "children": ["a"]},
        "a": {"parent_id": "root", "status": "pending", "title": "a", "children": []},
    }
    wf = _wf(curator=_FakeCurator(nodes))
    st = _state()
    messages: list[dict[str, Any]] = []
    for _ in range(5):
        _surface(wf, st, messages)
    assert st.last_focus_id == "a" and st.turns_on_task == 4
    # Worker splits 'a' into a child -> 'a' becomes a container, focus moves to a1.
    nodes["a"]["status"] = "in_progress"
    nodes["a"]["children"] = ["a1"]
    nodes["a1"] = {"parent_id": "a", "status": "pending", "title": "a1"}
    _surface(wf, st, messages)
    assert st.last_focus_id == "a1"  # focus advanced to the new leaf
    assert st.turns_on_task == 0  # grind counter reset by the decompose


def test_maybe_compact_returns_restart_signal() -> None:
    """_maybe_compact returns True only when a tier-2 restart actually replaced
    the history (the loop's cue to re-surface the focus banner)."""
    summariser = MagicMock()
    summariser.call.return_value = _resp("progress summary")
    wf = _wf(summariser_provider=summariser, compact_summarise_at_chars=500_000)
    # Below the tier-2 threshold -> no restart, returns False.
    short = [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]
    assert _compact_via_wire(wf, short) is False
    # Over the threshold -> restart, returns True.
    big = _big_text_history("TASK: x", blocks=8, block_chars=100_000)
    assert _compact_via_wire(wf, big) is True
    # History was replaced: [task, restart+summary, verbatim recent tail].
    assert len(big) == 3


def test_compact_request_forces_a_tier2_restart() -> None:
    """An operator compact.request (the TUI's "Compact now") forces the tier-2
    summarise-and-restart at the next boundary even far below the size
    thresholds, and consumes the marker: one request, one compaction."""
    summariser = MagicMock()
    summariser.call.return_value = _resp("progress summary")
    pending = {"req": True}
    wf = _wf(
        summariser_provider=summariser,
        compact_summarise_at_chars=500_000,
        compact_requested=lambda: "" if pending["req"] else None,
        compact_clear=lambda: pending.__setitem__("req", False),
    )
    small = _big_text_history("TASK: x", blocks=2, block_chars=100)  # nowhere near tier 2
    assert _compact_via_wire(wf, small) is True
    assert pending["req"] is False  # the marker was consumed
    assert len(small) == 2  # history replaced by (task + summary)
    # No re-trigger without a fresh request (and still below the threshold).
    assert _compact_via_wire(wf, small) is False


def test_stop_request_ends_the_run_at_the_step_boundary(tmp_path: Path) -> None:
    """A front-end's stop.request ("stop after this step") ends the run at the
    completed-iteration boundary -- after the step's tool results land -- with
    the resumable steer_abort shape, and consumes the marker."""
    from agent6.events import EventSink

    class ProviderStub:
        def __init__(self) -> None:
            self.calls = 0

        def call(self, **kwargs: Any) -> ProviderResponse:
            del kwargs
            self.calls += 1
            tid = f"t{self.calls}"
            return ProviderResponse(
                text="working",
                tool_uses=({"id": tid, "name": "noop", "input": {}},),
                stop_reason="tool_use",
                input_tokens=1,
                output_tokens=1,
                cache_read_tokens=0,
                cache_creation_tokens=0,
                raw={
                    "content": [
                        {"type": "text", "text": "working"},
                        {"type": "tool_use", "id": tid, "name": "noop", "input": {}},
                    ]
                },
            )

    class DispatcherStub(_StubDispatcher):
        def dispatch(self, name: str, raw_input: dict[str, Any]) -> ToolResult:
            del name, raw_input
            return RawResult({"ok": True})

    provider = ProviderStub()
    pending = {"stop": True}
    config = SimpleNamespace(
        git=_GIT_STUB,
        budget=SimpleNamespace(max_usd=10.0, max_tokens_fallback=2_000_000),
        workflow=SimpleNamespace(
            verify_when="never",
            verify_retries=2,
            verify_command=("true",),
            metric=None,
        ),
        prompt=SimpleNamespace(decompose=False),
    )
    wf = _wf(
        root=tmp_path,
        config=config,
        provider=provider,
        dispatcher=DispatcherStub(),
        events=EventSink(tmp_path / "logs.jsonl"),
        stop_requested=lambda: pending["stop"],
        stop_clear=lambda: pending.__setitem__("stop", False),
        max_iterations=30,
        loop_guard_kill_threshold=0,
    )
    messages = [{"role": "user", "content": [{"type": "text", "text": "TASK: x"}]}]
    with patch("agent6.workflows.loop.chain_commit", return_value="abc1234567890"):
        result = wf._drive_loop(  # pyright: ignore[reportPrivateUsage]
            system="system",
            conversation=Conversation.from_wire(messages),
            tool_calls=0,
            start_iteration=1,
            root_task_id=None,
            original_task="t",
        )
    assert result.reason == "steer_abort"  # the resumable stopped shape
    assert "stopped the run after step 1" in result.summary
    assert provider.calls == 1  # the step completed; no second turn started
    assert pending["stop"] is False  # the marker was consumed


def test_drive_loop_resurfaces_current_task_after_compaction(tmp_path: Path) -> None:
    """Integration: a tier-2 restart mid-run wipes the focus banner, and the loop's
    `if self._maybe_compact(messages): state.surfaced_task_id = None` edge makes the
    next nudge pass RE-SURFACE the current task into the fresh context. Pins that
    edge -- dropping the reset (or inverting the _maybe_compact bool) leaves no
    loop.task.surfaced after the restart, which is exactly the regression the
    surface/check-off/finish-gate trio exists to prevent."""
    import json

    from agent6.events import EventSink

    class ProviderStub:
        def __init__(self) -> None:
            self.calls = 0

        def call(self, **kwargs: Any) -> ProviderResponse:
            del kwargs
            self.calls += 1
            if self.calls >= 6:
                return _tool_resp("finish_session", {"summary": "done"}, tool_id=f"f{self.calls}")
            big = "y" * 3000  # accumulates each turn so tier-2 fires mid-run
            tid = f"t{self.calls}"
            return ProviderResponse(
                text=big,
                tool_uses=({"id": tid, "name": "noop", "input": {}},),
                stop_reason="tool_use",
                input_tokens=1,
                output_tokens=1,
                cache_read_tokens=0,
                cache_creation_tokens=0,
                raw={
                    "content": [
                        {"type": "text", "text": big},
                        {"type": "tool_use", "id": tid, "name": "noop", "input": {}},
                    ]
                },
            )

    class SummariserStub:
        def call(self, **kwargs: Any) -> ProviderResponse:
            del kwargs
            return _resp("SUMMARY of progress so far")  # no checkoff block

    class DispatcherStub(_StubDispatcher):
        def dispatch(self, name: str, raw_input: dict[str, Any]) -> ToolResult:
            if name == "finish_session":
                return RawResult({"acknowledged": True, "summary": raw_input.get("summary", "")})
            return RawResult({"ok": True})

    events = EventSink(tmp_path / "logs.jsonl")
    cur = _FakeCurator(
        {
            "root": {"parent_id": None, "status": "in_progress", "title": "review"},
            "a": {"parent_id": "root", "status": "pending", "title": "audit providers"},
        }
    )
    config = SimpleNamespace(
        git=_GIT_STUB,
        budget=SimpleNamespace(max_usd=10.0, max_tokens_fallback=2_000_000),
        workflow=SimpleNamespace(
            verify_when="never",
            verify_retries=2,
            verify_command=("true",),
            metric=None,
        ),
        prompt=SimpleNamespace(decompose=False),
    )
    wf = _wf(
        root=tmp_path,
        config=config,
        provider=ProviderStub(),
        dispatcher=DispatcherStub(),
        summariser_provider=SummariserStub(),
        events=events,
        curator=cur,
        compact_drop_at_chars=256_000,
        compact_summarise_at_chars=5_000,  # low so tier-2 fires mid-run
        budget=None,
        max_iterations=30,
        loop_guard_kill_threshold=0,
    )
    messages = [{"role": "user", "content": [{"type": "text", "text": "TASK: review"}]}]
    with patch("agent6.workflows.loop.chain_commit", return_value="abc1234567890"):
        wf._drive_loop(  # pyright: ignore[reportPrivateUsage]
            system="system",
            conversation=Conversation.from_wire(messages),
            tool_calls=0,
            start_iteration=1,
            root_task_id="root",
            original_task="t",
        )
    types = [
        json.loads(line)["type"]
        for line in (tmp_path / "logs.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert "loop.compact.summarise.done" in types  # tier-2 restart happened
    assert "loop.task.surfaced" in types
    # The focus banner re-surfaces AFTER the restart wiped it.
    restart_at = types.index("loop.compact.summarise.done")
    assert "loop.task.surfaced" in types[restart_at + 1 :]


def test_summarise_and_restart_falls_back_to_worker_provider() -> None:
    worker = MagicMock()
    worker.call.return_value = _resp("summary text")
    wf = _wf(provider=worker, summariser_provider=None)
    messages = _long_history(4)

    _restart_via_wire(wf, messages)

    assert len(messages) == 2
    worker.call.assert_called_once()


def test_summarise_and_restart_keeps_history_on_empty_summary() -> None:
    summariser = MagicMock()
    summariser.call.return_value = _resp("   ")
    wf = _wf(summariser_provider=summariser)
    messages = _long_history(5)
    before = list(messages)

    _restart_via_wire(wf, messages)

    # Empty summary -> message list untouched (fail-safe).
    assert messages == before


def test_summarise_and_restart_keeps_history_on_provider_error() -> None:
    summariser = MagicMock()
    summariser.call.side_effect = ProviderError("boom")
    wf = _wf(summariser_provider=summariser)
    messages = _long_history(5)
    before = list(messages)

    _restart_via_wire(wf, messages)

    assert messages == before


# --- _maybe_handle_steer --------------------------------------------------


def _steer_via_wire(
    wf: Workflow, messages: list[dict[str, Any]], *, iteration: int, state: Any
) -> str | None:
    conversation = Conversation.from_wire(messages)
    try:
        return wf._maybe_handle_steer(  # pyright: ignore[reportPrivateUsage]
            conversation, iteration, state
        )
    finally:
        messages[:] = conversation.to_wire()


def test_steer_noop_when_not_requested() -> None:
    """steer_requested() returns False -> _maybe_handle_steer is a no-op."""
    wf = _wf()  # default steer_requested = lambda: False
    messages: list[dict[str, Any]] = []
    result = _steer_via_wire(wf, messages, iteration=1, state=_state())
    assert result is None
    assert messages == []


def test_steer_injects_instruction() -> None:
    """Requested + non-empty prompt text -> instruction appended to messages."""
    cleared: list[bool] = []
    wf = _wf(
        steer_requested=lambda: True,
        steer_clear=lambda: cleared.append(True),
        steer_prompt=lambda: "focus on perf_takehome.py first",
    )
    messages: list[dict[str, Any]] = []
    result = _steer_via_wire(wf, messages, iteration=3, state=_state())
    assert result is None
    assert cleared == [True], "steer_clear must be called even on success"
    assert len(messages) == 1
    msg = messages[0]
    assert msg["role"] == "user"
    block = msg["content"][0]
    assert block["type"] == "text"
    assert "OPERATOR STEERING" in block["text"]
    assert "focus on perf_takehome.py first" in block["text"]


def test_steer_empty_text_continues_without_inject() -> None:
    """Operator answered blank/whitespace -> continue with no message."""
    cleared: list[bool] = []
    wf = _wf(
        steer_requested=lambda: True,
        steer_clear=lambda: cleared.append(True),
        steer_prompt=lambda: "   ",
    )
    messages: list[dict[str, Any]] = []
    result = _steer_via_wire(wf, messages, iteration=2, state=_state())
    assert result is None
    assert cleared == [True]
    assert messages == []


def test_steer_none_text_continues_without_inject() -> None:
    """Operator EOF'd (None) -> continue with no message."""
    cleared: list[bool] = []
    wf = _wf(
        steer_requested=lambda: True,
        steer_clear=lambda: cleared.append(True),
        steer_prompt=lambda: None,
    )
    messages: list[dict[str, Any]] = []
    result = _steer_via_wire(wf, messages, iteration=2, state=_state())
    assert result is None
    assert cleared == [True]
    assert messages == []


def test_steer_pin_records_and_injects_marked_notice() -> None:
    ev = _EventCapture()
    st = _state()
    wf = _wf(
        events=ev,
        steer_requested=lambda: True,
        steer_clear=lambda: None,
        steer_prompt=lambda: "/pin never touch the schema files",
    )
    messages: list[dict[str, Any]] = []
    assert _steer_via_wire(wf, messages, iteration=3, state=st) is None
    assert st.pins == ["never touch the schema files"]
    block = messages[0]["content"][0]["text"]
    assert "PINNED" in block and "survives context compaction" in block
    assert "never touch the schema files" in block
    added = [e for e in ev.events if e["type"] == "loop.pin.added"]
    assert added and added[-1]["text"] == "never touch the schema files"
    assert added[-1]["count"] == 1


def test_steer_pin_over_cap_delivers_as_ordinary_steer() -> None:
    """A pin past the total cap still reaches the model NOW as a plain steer;
    only the survives-compaction durability is refused, loudly."""
    from agent6.workflows.loop import PINS_MAX_CHARS

    ev = _EventCapture()
    st = _state(pins=["x" * (PINS_MAX_CHARS - 10)])
    wf = _wf(
        events=ev,
        steer_requested=lambda: True,
        steer_clear=lambda: None,
        steer_prompt=lambda: "/pin " + "y" * 100,
    )
    messages: list[dict[str, Any]] = []
    assert _steer_via_wire(wf, messages, iteration=3, state=st) is None
    assert len(st.pins) == 1  # the oversized pin was NOT recorded
    text = messages[0]["content"][0]["text"]
    assert "OPERATOR STEERING" in text and "y" * 100 in text
    assert "pin refused" in text  # the refusal is visible on every surface
    assert "PINNED" not in text
    refused = [e for e in ev.events if e["type"] == "loop.pin.refused"]
    assert refused and refused[-1]["limit"] == PINS_MAX_CHARS


def test_steer_bare_pin_answers_with_feedback() -> None:
    st = _state()
    wf = _wf(
        steer_requested=lambda: True,
        steer_clear=lambda: None,
        steer_prompt=lambda: "/pin   ",
    )
    messages: list[dict[str, Any]] = []
    assert _steer_via_wire(wf, messages, iteration=1, state=st) is None
    assert st.pins == []
    assert messages and "nothing pinned" in messages[0]["content"][0]["text"]


def test_steer_abort_signal() -> None:
    """Operator typed 'abort' (case-insensitive) -> returns 'abort'."""
    for typed in ("abort", "ABORT", "Abort"):
        cleared: list[bool] = []

        def _record(c: list[bool] = cleared) -> None:
            c.append(True)

        def _typed(t: str = typed) -> str:
            return t

        wf = _wf(
            steer_requested=lambda: True,
            steer_clear=_record,
            steer_prompt=_typed,
        )
        messages: list[dict[str, Any]] = []
        result = _steer_via_wire(wf, messages, iteration=5, state=_state())
        assert result == "abort", f"typed={typed!r}"
        assert cleared == [True]
        assert messages == [], "abort must not inject a message"


def test_steer_detach_signal() -> None:
    """Operator chose 'detach' -> returns 'detach' (the caller backgrounds the run)."""
    cleared: list[bool] = []
    wf = _wf(
        steer_requested=lambda: True,
        steer_clear=lambda: cleared.append(True),
        steer_prompt=lambda: "detach",
    )
    messages: list[dict[str, Any]] = []
    result = _steer_via_wire(wf, messages, iteration=4, state=_state())
    assert result == "detach"
    assert cleared == [True]
    assert messages == [], "detach must not inject a message"


def test_steer_clear_called_even_when_prompt_raises() -> None:
    """A misbehaving steer_prompt must not leave the flag set."""
    cleared: list[bool] = []

    def boom() -> str | None:
        raise RuntimeError("input EOF")

    wf = _wf(
        steer_requested=lambda: True,
        steer_clear=lambda: cleared.append(True),
        steer_prompt=boom,
    )
    messages: list[dict[str, Any]] = []
    with pytest.raises(RuntimeError, match="input EOF"):
        _steer_via_wire(wf, messages, iteration=1, state=_state())
    assert cleared == [True], "finally must run steer_clear even on prompt failure"


# --- resume: snapshot save/load and resume() behaviour ------------


def test_save_resume_snapshot_noop_when_path_unset(tmp_path: Path) -> None:
    """resume_state_path=None -> no file written, no exception."""
    wf = _wf()
    wf._save_resume_snapshot(  # pyright: ignore[reportPrivateUsage]
        system="s", messages=[], tool_calls=0, next_iteration=1, root_task_id=None, state=_state()
    )
    # tmp_path should still be empty.
    assert list(tmp_path.iterdir()) == []


def test_save_and_load_run_snapshot_round_trip(tmp_path: Path) -> None:
    """Snapshot written by _save_resume_snapshot loads back identically."""
    from agent6.workflows.loop import load_session_snapshot  # pyright: ignore[reportPrivateUsage]

    snap_path = tmp_path / "loop_state.json"
    wf = _wf(resume_state_path=snap_path)
    msgs: list[dict[str, Any]] = [
        {"role": "user", "content": [{"type": "text", "text": "hello"}]},
        {"role": "assistant", "content": [{"type": "text", "text": "hi back"}]},
    ]
    wf._save_resume_snapshot(  # pyright: ignore[reportPrivateUsage]
        system="SYSTEM PROMPT",
        messages=msgs,
        tool_calls=3,
        next_iteration=7,
        root_task_id="task-abc",
        state=_state(tool_calls=3),
    )
    assert snap_path.is_file()
    loaded = load_session_snapshot(snap_path)
    assert loaded.system == "SYSTEM PROMPT"
    assert loaded.messages == msgs
    assert loaded.tool_calls == 3
    assert loaded.next_iteration == 7
    assert loaded.root_task_id == "task-abc"


def test_save_resume_snapshot_atomic_no_partial_tmp(tmp_path: Path) -> None:
    """After save, no .tmp file remains: the final snapshot + its per-turn
    checkpoint are the only artifacts (both written atomically)."""
    snap_path = tmp_path / "loop_state.json"
    wf = _wf(resume_state_path=snap_path)
    wf._save_resume_snapshot(  # pyright: ignore[reportPrivateUsage]
        system="s",
        messages=[],
        tool_calls=0,
        next_iteration=1,
        root_task_id=None,
        state=_state(),
        write_checkpoint=True,
    )
    assert snap_path.is_file()
    # The per-turn checkpoint lands under checkpoints/; nothing else (no .tmp).
    assert (tmp_path / "checkpoints" / "0001.json").is_file()
    leftovers = sorted(p.name for p in tmp_path.iterdir() if p.name != snap_path.name)
    assert leftovers == ["checkpoints"], f"unexpected leftover files: {leftovers}"
    cp_leftovers = [p.name for p in (tmp_path / "checkpoints").iterdir()]
    assert cp_leftovers == ["0001.json"], f"unexpected checkpoint leftovers: {cp_leftovers}"


def test_save_resume_snapshot_uses_durable_atomic_writer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Resume state must go through the durable writer, not plain write_text."""
    writes: list[Path] = []

    def _fake_atomic_write(path: Path, data: str | bytes) -> None:
        writes.append(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(data, bytes):
            path.write_bytes(data)
        else:
            path.write_text(data, encoding="utf-8")

    monkeypatch.setattr("agent6.workflows.loop.atomic_write", _fake_atomic_write)
    snap_path = tmp_path / "loop_state.json"
    wf = _wf(resume_state_path=snap_path)

    wf._save_resume_snapshot(  # pyright: ignore[reportPrivateUsage]
        system="s",
        messages=[],
        tool_calls=0,
        next_iteration=9,
        root_task_id=None,
        state=_state(),
        write_checkpoint=True,
    )

    assert writes == [tmp_path / "checkpoints" / "0009.json", snap_path]


def test_load_run_snapshot_rejects_version_mismatch(tmp_path: Path) -> None:
    """A snapshot with a wrong version must raise ValueError."""
    import json as _json

    from agent6.workflows.loop import load_session_snapshot  # pyright: ignore[reportPrivateUsage]

    snap_path = tmp_path / "loop_state.json"
    snap_path.write_text(
        _json.dumps(
            {
                "version": 999,
                "system": "s",
                "messages": [],
                "tool_calls": 0,
                "next_iteration": 1,
                "root_task_id": None,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="is version 999, not 2"):
        load_session_snapshot(snap_path)


def test_resume_raises_when_path_unset() -> None:
    """resume() with resume_state_path=None must raise ResumeError."""
    from agent6.workflows.loop import ResumeError

    wf = _wf()
    with pytest.raises(ResumeError, match="resume_state_path"):
        wf.resume()


def test_resume_raises_on_missing_snapshot(tmp_path: Path) -> None:
    """resume() with a nonexistent snapshot file must raise ResumeError."""
    from agent6.workflows.loop import ResumeError

    wf = _wf(resume_state_path=tmp_path / "nope.json")
    with pytest.raises(ResumeError, match="failed to load"):
        wf.resume()


def test_resume_drives_loop_from_snapshot(tmp_path: Path) -> None:
    """resume() loads snapshot, calls provider once, finishes via silent_finish."""
    snap_path = tmp_path / "loop_state.json"
    # Pre-seed the snapshot as if a prior run had just completed iter 4
    # and was about to start iter 5.
    snap_path.write_text(
        '{"version": 2, "system": "S", "messages": [{"role": "user", '
        '"content": [{"type": "text", "text": "go"}]}], "tool_calls": 2, '
        '"next_iteration": 5, "root_task_id": null, "original_task": "go", '
        '"verify_command": []}',
        encoding="utf-8",
    )
    seeded_mtime_ns = snap_path.stat().st_mtime_ns

    provider = MagicMock()
    provider.call.return_value = _resp("all done")  # no tool_uses -> silent_finish

    dispatcher = MagicMock()
    dispatcher.set_run_root_node_id = MagicMock()

    wf = _wf(provider=provider, dispatcher=dispatcher, resume_state_path=snap_path)
    result = wf.resume()

    assert result.completed is True
    assert result.reason == "silent_finish"
    assert result.iterations == 5, "must resume at snapshot's next_iteration"
    assert result.tool_calls == 2, "must carry forward snapshot's tool_calls"
    # The pre-call save REWROTE the seeded snapshot (not merely left it there).
    assert snap_path.stat().st_mtime_ns > seeded_mtime_ns


def test_resume_restores_root_task_id_on_dispatcher(tmp_path: Path) -> None:
    """A non-null root_task_id in the snapshot must be re-set on dispatcher."""
    snap_path = tmp_path / "loop_state.json"
    snap_path.write_text(
        '{"version": 2, "system": "S", "messages": [{"role": "user", '
        '"content": [{"type": "text", "text": "go"}]}], "tool_calls": 0, '
        '"next_iteration": 1, "root_task_id": "task-xyz", "original_task": "go", '
        '"verify_command": []}',
        encoding="utf-8",
    )
    provider = MagicMock()
    provider.call.return_value = _resp("done")
    dispatcher = MagicMock()
    wf = _wf(provider=provider, dispatcher=dispatcher, resume_state_path=snap_path)
    wf.resume()
    dispatcher.set_run_root_node_id.assert_called_once_with("task-xyz")


# --- crash-and-resume: snapshot survives a provider crash mid-run ---


def test_crash_mid_run_then_resume_continues_from_snapshot(tmp_path: Path) -> None:
    """Simulate a provider crash mid-loop: snapshot must allow a clean resume.

    The v2 contract is: a snapshot is written BEFORE each LLM call, so a
    crash at any point (network, OOM, SIGKILL) leaves the run resumable
    from exactly that iteration with the prior turn's messages intact.
    Here we use a fake provider that raises on the first call to simulate
    the crash, then a fresh provider on resume that drives the loop to a
    clean finish.
    """
    import subprocess as _sp

    # Real git repo so _load_repo_summary() succeeds.
    repo = tmp_path / "repo"
    repo.mkdir()
    _sp.run(["git", "init", "-q", "-b", "main"], cwd=repo, check=True)
    _sp.run(["git", "config", "user.email", "t@example.com"], cwd=repo, check=True)
    _sp.run(["git", "config", "user.name", "t"], cwd=repo, check=True)
    (repo / "x.txt").write_text("hi\n")
    _sp.run(["git", "add", "x.txt"], cwd=repo, check=True)
    _sp.run(["git", "commit", "-q", "-m", "init"], cwd=repo, check=True)

    snap_path = repo / "loop_state.json"

    # First "process": crash on the first LLM call.
    crashing_provider = MagicMock()
    crashing_provider.call.side_effect = ProviderError("simulated network drop / SIGKILL window")
    dispatcher = MagicMock()
    dispatcher.set_run_root_node_id = MagicMock()
    wf1 = _wf(
        root=repo,
        provider=crashing_provider,
        dispatcher=dispatcher,
        resume_state_path=snap_path,
        provider_retry_count=0,  # don't mask the crash with a retry
    )
    # The first .run() ends with provider_error (v2's clean-shutdown path
    # for provider crashes). The snapshot was written BEFORE the call, so
    # the run is resumable from exactly that iteration.
    result1 = wf1.run("do a thing")
    assert result1.completed is False
    assert result1.reason == "provider_error"

    # Snapshot must exist after the crash and be loadable.
    assert snap_path.is_file(), "snapshot must be written before every LLM call"
    from agent6.workflows.loop import load_session_snapshot  # pyright: ignore[reportPrivateUsage]

    snap = load_session_snapshot(snap_path)
    # The user's task message survived in the snapshot.
    user_text = "".join(
        block.get("text", "")
        for msg in snap.messages
        if msg["role"] == "user"
        for block in msg["content"]
        if isinstance(block, dict) and block.get("type") == "text"
    )
    assert "do a thing" in user_text, "user task message must be preserved in the snapshot"

    # Second "process": new provider, drives to silent_finish.
    fresh_provider = MagicMock()
    fresh_provider.call.return_value = _resp("done now")
    wf2 = _wf(
        root=repo,
        provider=fresh_provider,
        dispatcher=dispatcher,
        resume_state_path=snap_path,
    )
    result = wf2.resume()
    assert result.completed is True
    assert result.reason == "silent_finish"


# --- tier-2 summarise-and-restart compaction -------------------------------
# Synthetic exercise driving context past compact_summarise_at_chars to confirm
# tier-2 actually summarises-and-restarts (the path that was unreachable before
# it measured the whole context via _context_chars).


def _ctx_chars(messages: list[dict[str, Any]]) -> int:
    from agent6.workflows._compaction import context_chars

    return context_chars(Conversation.from_wire(messages))


def _big_text_history(task: str, *, blocks: int, block_chars: int) -> list[dict[str, Any]]:
    # Assistant TEXT accumulates across a long run and tier-1 never elides it
    # (it only drops tool_results), so this is exactly what tier-2 must catch.
    big = "x" * block_chars
    msgs: list[dict[str, Any]] = [{"role": "user", "content": [{"type": "text", "text": task}]}]
    for _ in range(blocks):
        msgs.append({"role": "assistant", "content": [{"type": "text", "text": big}]})
        msgs.append({"role": "user", "content": [{"type": "text", "text": "keep going"}]})
    return msgs


def test_tier2_summarise_fires_and_restarts_past_threshold(tmp_path: Path) -> None:
    class SummariserStub:
        def __init__(self) -> None:
            self.calls = 0

        def call(self, **kwargs: Any) -> ProviderResponse:
            del kwargs
            self.calls += 1
            return _resp("PROGRESS SUMMARY: explored modules, applied 3 patches.")

    summ = SummariserStub()
    wf = _wf(
        root=tmp_path,
        summariser_provider=summ,
        compact_drop_at_chars=256_000,
        compact_summarise_at_chars=500_000,
    )
    messages = _big_text_history("TASK: optimize the kernel", blocks=8, block_chars=100_000)
    assert _ctx_chars(messages) > 500_000  # over the tier-2 threshold

    _compact_via_wire(wf, messages)

    assert summ.calls == 1  # tier-2 summariser ran
    # Restarted to [original task, restart+summary, verbatim recent tail]:
    # the trailing small turn fits keep_recent_chars and survives verbatim.
    assert len(messages) == 3
    assert messages[0]["content"][0]["text"] == "TASK: optimize the kernel"
    assert "PROGRESS SUMMARY" in messages[1]["content"][0]["text"]
    assert messages[2]["content"][0]["text"] == "keep going"
    assert _ctx_chars(messages) < 500_000  # context actually shrank


def test_tier2_summarise_failsafe_keeps_context_on_empty_summary(tmp_path: Path) -> None:
    class EmptySummariser:
        def call(self, **kwargs: Any) -> ProviderResponse:
            del kwargs
            return _resp("")  # empty -> fail-safe: keep the (tier-1-elided) context

    wf = _wf(
        root=tmp_path,
        summariser_provider=EmptySummariser(),
        compact_summarise_at_chars=500_000,
    )
    messages = _big_text_history("TASK", blocks=8, block_chars=100_000)
    n_before = len(messages)

    _compact_via_wire(wf, messages)

    assert len(messages) == n_before  # unchanged; the run continues on tier-1 elision


def test_drive_loop_summarises_midrun_then_completes(tmp_path: Path) -> None:
    import json

    from agent6.events import EventSink

    class ProviderStub:
        def __init__(self) -> None:
            self.calls = 0

        def call(self, **kwargs: Any) -> ProviderResponse:
            del kwargs
            self.calls += 1
            if self.calls >= 6:
                return _tool_resp("finish_session", {"summary": "done"}, tool_id=f"f{self.calls}")
            # Large assistant text accumulates each turn; tier-1 can't elide it.
            big = "y" * 3000
            tid = f"t{self.calls}"
            return ProviderResponse(
                text=big,
                tool_uses=({"id": tid, "name": "noop", "input": {}},),
                stop_reason="tool_use",
                input_tokens=1,
                output_tokens=1,
                cache_read_tokens=0,
                cache_creation_tokens=0,
                raw={
                    "content": [
                        {"type": "text", "text": big},
                        {"type": "tool_use", "id": tid, "name": "noop", "input": {}},
                    ]
                },
            )

    class SummariserStub:
        def __init__(self) -> None:
            self.calls = 0

        def call(self, **kwargs: Any) -> ProviderResponse:
            del kwargs
            self.calls += 1
            return _resp("SUMMARY of progress so far")

    class DispatcherStub(_StubDispatcher):
        def dispatch(self, name: str, raw_input: dict[str, Any]) -> ToolResult:
            if name == "finish_session":
                return RawResult({"acknowledged": True, "summary": raw_input.get("summary", "")})
            return RawResult({"ok": True})

    events = EventSink(tmp_path / "logs.jsonl")
    summ = SummariserStub()
    config = SimpleNamespace(
        git=_GIT_STUB,
        budget=SimpleNamespace(max_usd=10.0, max_tokens_fallback=2_000_000),
        workflow=SimpleNamespace(
            verify_when="never",
            verify_retries=2,
            verify_command=("true",),
            metric=None,
        ),
        prompt=SimpleNamespace(decompose=False),
    )
    wf = _wf(
        root=tmp_path,
        config=config,
        provider=ProviderStub(),
        dispatcher=DispatcherStub(),
        summariser_provider=summ,
        events=events,
        compact_drop_at_chars=256_000,
        compact_summarise_at_chars=5_000,  # low so it fires mid-run
        budget=None,
        max_iterations=30,
        loop_guard_kill_threshold=0,
    )
    messages = [{"role": "user", "content": [{"type": "text", "text": "TASK: optimize"}]}]

    with patch("agent6.workflows.loop.chain_commit", return_value="abc1234567890"):
        result = wf._drive_loop(  # pyright: ignore[reportPrivateUsage]
            system="system",
            conversation=Conversation.from_wire(messages),
            tool_calls=0,
            start_iteration=1,
            root_task_id=None,
            original_task="t",
        )

    assert result.completed is True
    assert result.reason == "finish_session"
    assert summ.calls >= 1  # tier-2 fired mid-run
    types = [
        json.loads(line)["type"]
        for line in (tmp_path / "logs.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert "loop.compact.summarise.done" in types  # summarise-and-restart happened cleanly


def test_pass_pending_root_tasks_passes_only_pending_roots() -> None:
    """_pass_pending_root_tasks marks pending/in-progress ROOT tasks passed and
    leaves everything else (already-terminal roots, non-root subtasks) alone --
    so a finish_session-only ask/run reads N/N, not 0/1."""

    class _FakeClient:
        def __init__(self, nodes: dict[str, dict[str, Any]]) -> None:
            self._nodes = nodes
            self.passed: list[str] = []

        def nodes(self) -> dict[str, Any]:
            return _typed(self._nodes)

        def update_status(self, intent: Any) -> None:
            self.passed.append(intent.id)
            self._nodes[intent.id]["status"] = intent.new_status

    nodes: dict[str, dict[str, Any]] = {
        "root1": {"parent_id": None, "status": "pending"},
        "root2": {"parent_id": None, "status": "passed"},  # already done -> skip
        "child": {"parent_id": "root1", "status": "pending"},  # not a root -> skip
        "root3": {"parent_id": None, "status": "in_progress"},
        "root4": {"parent_id": None, "status": "failed"},  # failed -> leave honest
    }
    fake = _FakeClient(nodes)
    wf = _wf(curator=fake)
    wf._pass_pending_root_tasks()  # pyright: ignore[reportPrivateUsage]
    assert set(fake.passed) == {"root1", "root3"}


def test_pass_pending_root_tasks_noop_without_curator() -> None:
    """No curator wired (e.g. ask without a DAG) -> the auto-pass is a no-op."""
    wf = _wf(curator=None)
    wf._pass_pending_root_tasks()  # pyright: ignore[reportPrivateUsage]  (must not raise)


def test_drive_loop_gateless_settles_after_commit(tmp_path: Path) -> None:
    """A GATELESS run (no verify_command) has no green verify to seed the
    idle-stop net. Once an edit is committed it must still settle: spinning on
    read-only commands after the commit stops the run, so a gateless run can't
    burn budget to exhaustion when the worker is done. The reason is 'settled'
    (never 'verify_settled': nothing was verified, and the old label put
    'passed / verify passed' on every surface)."""

    class ProviderStub:
        def __init__(self) -> None:
            self.calls = 0

        def call(self, **kwargs: Any) -> ProviderResponse:
            self.calls += 1
            if self.calls == 1:
                # an edit -> gateless auto-commit -> seeds gateless_ever_committed
                return _tool_resp("apply_edit", {"path": "x", "edits": []}, tool_id="e1")
            # then spin on read-only commands (no edit, no commit)
            return _tool_resp("run_command", {"cmd": f"ls {self.calls}"}, tool_id=f"c{self.calls}")

    class DispatcherStub(_StubDispatcher):
        def dispatch(self, name: str, raw_input: dict[str, Any]) -> ToolResult:
            return ExecResult(
                returncode=0, stdout="ok", stderr="", duration_s=0.1, exec_failed=False
            )

    provider = ProviderStub()
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
    wf = _wf(
        root=tmp_path,
        config=config,
        mode="run",
        provider=provider,
        dispatcher=DispatcherStub(),
        max_iterations=30,
    )
    messages = [{"role": "user", "content": [{"type": "text", "text": "TASK:\ndo it"}]}]
    with patch("agent6.workflows.loop.chain_commit", return_value="sha1"):
        result = wf._drive_loop(  # pyright: ignore[reportPrivateUsage]
            system="s",
            conversation=Conversation.from_wire(messages),
            tool_calls=0,
            start_iteration=1,
            root_task_id=None,
            original_task="t",
        )
    assert result.reason == "settled"
    assert result.completed is True
    assert provider.calls < 30  # stopped well before max_iterations, not burned to the cap


def test_resume_snapshot_carries_verify_command(tmp_path: Path) -> None:
    """The snapshot stores the run's resolved verify_command so resume reuses it
    rather than re-inferring (which could diverge from the frozen prompt). A
    gateless run stores [] and loads back as ()."""
    from agent6.workflows._session_state import load_session_snapshot

    snap = tmp_path / "loop_state.json"
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
    wf = _wf(resume_state_path=snap, config=config)
    wf._save_resume_snapshot(  # pyright: ignore[reportPrivateUsage]
        system="s", messages=[], tool_calls=0, next_iteration=1, root_task_id=None, state=_state()
    )
    assert load_session_snapshot(snap).verify_command == ("pytest", "-q")

    config.workflow.verify_command = ()  # gateless run -> stored as [] -> loads as ()
    wf._save_resume_snapshot(  # pyright: ignore[reportPrivateUsage]
        system="s", messages=[], tool_calls=0, next_iteration=1, root_task_id=None, state=_state()
    )
    assert load_session_snapshot(snap).verify_command == ()


def test_provider_error_hint_for_auth_and_quota() -> None:
    from agent6.workflows.loop import provider_error_hint  # pyright: ignore[reportPrivateUsage]

    assert "agent6 connect" in provider_error_hint(401)
    assert "agent6 connect" in provider_error_hint(403)
    # The failing provider's own config key, when the wrapper stamped it.
    assert "[providers.openrouter].api_key_env" in provider_error_hint(401, "openrouter")
    assert "[providers.<name>].api_key_env" in provider_error_hint(401)
    assert "credits" in provider_error_hint(402).lower()
    # Transient / unknown statuses get no hint (don't mislead).
    assert provider_error_hint(429) == ""
    assert provider_error_hint(500) == ""
    assert provider_error_hint(None) == ""


def test_save_resume_snapshot_degrades_on_unwritable_state_dir(tmp_path: Path) -> None:
    # A full disk / read-only state dir disables resume/fork but must not abort
    # the run. Simulate by pointing the snapshot under a path whose parent is a
    # regular file, so mkdir raises OSError.
    blocker = tmp_path / "blocker"
    blocker.write_text("x", encoding="utf-8")
    snap = blocker / "loop_state.json"  # parent "blocker" is a file -> mkdir fails
    logs: list[str] = []
    config = SimpleNamespace(
        git=_GIT_STUB,
        budget=SimpleNamespace(max_usd=10.0, max_tokens_fallback=2_000_000),
        workflow=SimpleNamespace(
            verify_when="never",
            verify_retries=2,
            verify_command=(),
            metric=None,
        ),
    )
    wf = _wf(resume_state_path=snap, config=config, logger=logs.append)
    # Must not raise, twice (the second call must not re-warn).
    for _ in range(2):
        wf._save_resume_snapshot(  # pyright: ignore[reportPrivateUsage]
            system="s",
            messages=[],
            tool_calls=0,
            next_iteration=1,
            root_task_id=None,
            state=_state(),
        )
    warnings = [m for m in logs if "could not persist resume snapshot" in m]
    assert len(warnings) == 1, "warn exactly once, then stay quiet"
    assert not snap.exists()


def test_open_tasks_for_checkoff_excludes_auto_root() -> None:
    # The tier-2 compaction check-off must never offer the auto-root (parent_id
    # is None): a summariser listing it would mark the whole run passed mid-run.
    curator = MagicMock()
    curator.nodes.return_value = _typed(
        {
            "root": {"status": "in_progress", "title": "the whole run", "parent_id": None},
            "01A": {"status": "pending", "title": "subtask A", "parent_id": "root"},
            "01B": {"status": "in_progress", "title": "subtask B", "parent_id": "root"},
            "01C": {"status": "passed", "title": "done subtask", "parent_id": "root"},
        }
    )
    wf = _wf(curator=curator)
    ids = {nid for nid, _ in wf._open_tasks_for_checkoff()}  # pyright: ignore[reportPrivateUsage]
    assert ids == {"01A", "01B"}  # root excluded; passed subtask excluded


def test_run_result_docstring_enumerates_every_loop_reason() -> None:
    # SessionResult.reason is a free-form str whose docstring is the enumeration
    # operators grep against; it silently drifted to omit five reasons. Pin it
    # to the literal `reason=` values loop.py actually constructs.
    import ast
    import inspect

    import agent6.workflows.loop as loopmod
    from agent6.workflows._session_state import SessionResult

    reasons: set[str] = set()
    for node in ast.walk(ast.parse(inspect.getsource(loopmod))):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)):
            continue
        if node.func.id != "SessionResult":
            continue
        for kw in node.keywords:
            if kw.arg == "reason" and isinstance(kw.value, ast.Constant):
                assert isinstance(kw.value.value, str)
                reasons.add(kw.value.value)
    # `reason=finish_kind` is the one non-literal construction; its Literal type
    # covers exactly these two.
    reasons |= {"finish_session", "finish_planning"}
    assert reasons >= {
        "loop_guard_killed",
        "verify_settled",
        "verify_command_unexecutable",
        "interactive_stop",
        "finish_planning",
    }  # the five the docstring omitted
    doc = SessionResult.__doc__ or ""
    undocumented = {r for r in reasons if r not in doc}
    assert not undocumented, f"SessionResult docstring omits reasons: {sorted(undocumented)}"


def test_question_nudge_then_accept(tmp_path: Path) -> None:
    """A run-mode turn that ends by asking a prose question with no tool call is
    nudged ONCE to call ask_user; if the model then acts it recovers, and if it
    keeps asking the run accepts silent_finish (bounded, no loop)."""
    from agent6.workflows.loop import QUESTION_NUDGE  # pyright: ignore[reportPrivateUsage]

    class ProviderStub:
        def __init__(self) -> None:
            self.turns = 0
            self.saw_nudge = False

        def call(self, **kwargs: Any) -> ProviderResponse:
            self.turns += 1
            # After the nudge, the last user message carries _QUESTION_NUDGE.
            msgs = kwargs.get("messages", [])
            last = msgs[-1] if msgs else {}
            blocks = last.get("content", []) if isinstance(last, dict) else []
            text = " ".join(b.get("text", "") for b in blocks if isinstance(b, dict))
            if QUESTION_NUDGE in text:
                self.saw_nudge = True
                return _tool_resp("ask_user", {"questions": [{"question": "A?"}]}, tool_id="q1")
            if self.turns == 1:
                return _resp("Which theme should I add?")  # prose question, no tool
            return _resp("Anything else you want?")  # would-be second question

    class DispatcherStub(_StubDispatcher):
        def dispatch(self, name: str, raw_input: dict[str, Any]) -> ToolResult:
            if name == "ask_user":
                return RawResult({"answers": ["dracula"]})
            raise AssertionError(f"unexpected tool: {name}")

    provider = ProviderStub()
    config = SimpleNamespace(
        git=_GIT_STUB,
        budget=SimpleNamespace(max_usd=10.0, max_tokens_fallback=2_000_000),
        workflow=SimpleNamespace(
            verify_when="never",
            verify_retries=2,
            verify_command=(),
            metric=None,
        ),
    )
    wf = _wf(
        root=tmp_path,
        config=config,
        provider=provider,
        dispatcher=DispatcherStub(),
        max_iterations=10,
    )
    messages = [{"role": "user", "content": [{"type": "text", "text": "TASK:\nadd a theme"}]}]
    result = wf._drive_loop(  # pyright: ignore[reportPrivateUsage]
        system="s",
        conversation=Conversation.from_wire(messages),
        tool_calls=0,
        start_iteration=1,
        root_task_id=None,
        original_task="t",
    )
    # Turn 1 asked a question -> nudged -> turn 2 called ask_user -> turn 3 asked
    # again, but the one-shot nudge is spent, so it silently finished.
    assert provider.saw_nudge
    assert result.reason == "silent_finish"


def test_ends_with_question_detection() -> None:
    from agent6.workflows.loop import ends_with_question  # pyright: ignore[reportPrivateUsage]

    assert ends_with_question("I found two options.\nWhich do you prefer?")
    assert not ends_with_question("Done. All tests pass.")
    assert not ends_with_question("")
    assert ends_with_question("Should I proceed?  ")  # trailing space tolerated


def test_drive_loop_no_progress_nudges_on_identical_failures(tmp_path: Path) -> None:
    """A worker whose edits keep producing the SAME verify failure (observed:
    mistral-small repeating one failure nine times) gets a root-cause nudge at
    the 4th identical consecutive failure and one escalation at the 7th; the
    signature ignores cosmetic drift like line numbers."""
    from agent6.workflows.loop import (
        NO_PROGRESS_ESCALATION,  # pyright: ignore[reportPrivateUsage]
        NO_PROGRESS_NUDGE,  # pyright: ignore[reportPrivateUsage]
    )

    class ProviderStub:
        def __init__(self) -> None:
            self.calls = 0
            self.nudges = 0
            self.escalations = 0

        def call(self, **kwargs: Any) -> ProviderResponse:
            self.calls += 1
            last = str(kwargs["messages"][-1])
            if NO_PROGRESS_NUDGE[:28] in last:
                self.nudges += 1
            if NO_PROGRESS_ESCALATION[:28] in last:
                self.escalations += 1
            if self.calls >= 18:
                return _tool_resp("finish_session", {"summary": "stuck"}, tool_id="f")
            if self.calls % 2 == 1:
                return _tool_resp(
                    "apply_edit",
                    {"path": "f.py", "edits": [{"old_string": "a", "new_string": "b"}]},
                    tool_id=f"e{self.calls}",
                )
            return _tool_resp("run_verify_command", tool_id=f"v{self.calls}")

    class DispatcherStub(_StubDispatcher):
        def __init__(self) -> None:
            self.verifies = 0

        def dispatch(self, name: str, raw_input: dict[str, Any]) -> ToolResult:
            if name == "run_verify_command":
                self.verifies += 1
                return ExecResult(
                    returncode=1,
                    stdout="",
                    stderr=f'File "t.py", line {40 + self.verifies}\nAssertionError: want 3 got 2',
                    duration_s=0.1,
                    exec_failed=False,
                )
            if name == "apply_edit":
                return RawResult({"applied": True, "path": "f.py"})
            return RawResult({"ok": True})

    provider = ProviderStub()
    config = SimpleNamespace(
        git=_GIT_STUB,
        budget=SimpleNamespace(max_usd=10.0, max_tokens_fallback=2_000_000),
        workflow=SimpleNamespace(
            verify_when="never",
            verify_retries=2,
            verify_command=("true",),
            metric=SimpleNamespace(goal=None),
        ),
    )
    wf = _wf(
        root=tmp_path,
        config=config,
        mode="run",
        provider=provider,
        dispatcher=DispatcherStub(),
        max_iterations=40,
    )
    messages = [{"role": "user", "content": [{"type": "text", "text": "TASK:\nfix"}]}]
    with patch("agent6.workflows.loop.chain_commit", return_value="sha1"):
        result = wf._drive_loop(  # pyright: ignore[reportPrivateUsage]
            system="s",
            conversation=Conversation.from_wire(messages),
            tool_calls=0,
            start_iteration=1,
            root_task_id=None,
            original_task="t",
        )
    assert provider.nudges == 1
    assert provider.escalations == 1
    assert result.completed is True


def test_drive_loop_no_progress_silent_when_failures_differ(tmp_path: Path) -> None:
    """Distinct failures mean real progress through the error list; the guard
    must stay quiet."""
    from agent6.workflows.loop import NO_PROGRESS_NUDGE  # pyright: ignore[reportPrivateUsage]

    class ProviderStub:
        def __init__(self) -> None:
            self.calls = 0
            self.nudges = 0

        def call(self, **kwargs: Any) -> ProviderResponse:
            self.calls += 1
            if NO_PROGRESS_NUDGE[:28] in str(kwargs["messages"][-1]):
                self.nudges += 1
            if self.calls >= 20:
                return _tool_resp("finish_session", {"summary": "done"}, tool_id="f")
            return _tool_resp("run_verify_command", tool_id=f"v{self.calls}")

    class DispatcherStub(_StubDispatcher):
        def __init__(self) -> None:
            self.verifies = 0

        def dispatch(self, name: str, raw_input: dict[str, Any]) -> ToolResult:
            if name == "run_verify_command":
                self.verifies += 1
                return ExecResult(
                    returncode=1,
                    stdout="",
                    stderr=f"AssertionError: case {self.verifies} failed",
                    duration_s=0.1,
                    exec_failed=False,
                )
            return RawResult({"ok": True})

    provider = ProviderStub()
    config = SimpleNamespace(
        git=_GIT_STUB,
        budget=SimpleNamespace(max_usd=10.0, max_tokens_fallback=2_000_000),
        workflow=SimpleNamespace(
            verify_when="never",
            verify_retries=2,
            verify_command=("true",),
            metric=SimpleNamespace(goal=None),
        ),
    )
    wf = _wf(
        root=tmp_path,
        config=config,
        mode="run",
        provider=provider,
        dispatcher=DispatcherStub(),
        max_iterations=30,
    )
    messages = [{"role": "user", "content": [{"type": "text", "text": "TASK:\nfix"}]}]
    with patch("agent6.workflows.loop.chain_commit", return_value="sha1"):
        wf._drive_loop(  # pyright: ignore[reportPrivateUsage]
            system="s",
            conversation=Conversation.from_wire(messages),
            tool_calls=0,
            start_iteration=1,
            root_task_id=None,
            original_task="t",
        )
    assert provider.nudges == 0


def test_verify_failure_signature_normalizes_cosmetics() -> None:
    from agent6.workflows._nudges import verify_failure_signature

    a = verify_failure_signature("", 'File "t.py", line 41\nAssertionError: want 3 got 2')
    b = verify_failure_signature("", 'File "t.py", line 97\nAssertionError: want 3 got 2')
    c = verify_failure_signature("", 'File "t.py", line 41\nAssertionError: want 5 got 1')
    assert a == b
    assert a != c
    d = verify_failure_signature("ran in 3.21s at 0x7f01ab", "")
    e = verify_failure_signature("ran in 0.07s at 0x9921cd", "")
    assert d == e


def test_drive_loop_no_progress_stops_after_unheeded_interventions(tmp_path: Path) -> None:
    """Ten consecutive identical failures with both nudges delivered ends the
    run honestly (reason=no_progress) instead of burning to the iteration cap
    (measured: nudged mistral runs ran to 77 iters at score 0 without this)."""

    class ProviderStub:
        def __init__(self) -> None:
            self.calls = 0

        def call(self, **kwargs: Any) -> ProviderResponse:
            self.calls += 1
            if self.calls % 2 == 1:
                return _tool_resp(
                    "apply_edit",
                    {"path": "f.py", "edits": [{"old_string": "a", "new_string": "b"}]},
                    tool_id=f"e{self.calls}",
                )
            return _tool_resp("run_verify_command", tool_id=f"v{self.calls}")

    class DispatcherStub(_StubDispatcher):
        def dispatch(self, name: str, raw_input: dict[str, Any]) -> ToolResult:
            if name == "run_verify_command":
                return ExecResult(
                    returncode=1,
                    stdout="",
                    stderr="AssertionError: want 3 got 2",
                    duration_s=0.1,
                    exec_failed=False,
                )
            if name == "apply_edit":
                return RawResult({"applied": True, "path": "f.py"})
            return RawResult({"ok": True})

    provider = ProviderStub()
    config = SimpleNamespace(
        git=_GIT_STUB,
        budget=SimpleNamespace(max_usd=10.0, max_tokens_fallback=2_000_000),
        workflow=SimpleNamespace(
            verify_when="never",
            verify_retries=2,
            verify_command=("true",),
            metric=SimpleNamespace(goal=None),
        ),
    )
    wf = _wf(
        root=tmp_path,
        config=config,
        mode="run",
        provider=provider,
        dispatcher=DispatcherStub(),
        max_iterations=60,
    )
    messages = [{"role": "user", "content": [{"type": "text", "text": "TASK:\nfix"}]}]
    with patch("agent6.workflows.loop.chain_commit", return_value="sha1"):
        result = wf._drive_loop(  # pyright: ignore[reportPrivateUsage]
            system="s",
            conversation=Conversation.from_wire(messages),
            tool_calls=0,
            start_iteration=1,
            root_task_id=None,
            original_task="t",
        )
    assert result.completed is False
    assert result.reason == "no_progress"
    assert result.iterations < 30  # stopped well before the 60-iteration cap


def test_drive_loop_silent_finish_on_untouched_tree_is_nudged(tmp_path: Path) -> None:
    """A prose-only turn before ANY edit or green verify is a stall, not an
    implicit finish (observed: kimi answering a SWE-bench problem statement
    in prose at iteration 2, ending the run patchless). Two nudges steer back
    to the tools; a third prose turn is then honored as silent_finish."""
    from agent6.workflows.loop import SILENT_NO_WORK_NUDGE  # pyright: ignore[reportPrivateUsage]

    class ProviderStub:
        def __init__(self) -> None:
            self.calls = 0
            self.nudges = 0

        def call(self, **kwargs: Any) -> ProviderResponse:
            self.calls += 1
            if SILENT_NO_WORK_NUDGE[:22] in str(kwargs["messages"][-1]):
                self.nudges += 1
            return _resp("Here is my analysis of the problem.")

    provider = ProviderStub()
    config = SimpleNamespace(
        git=_GIT_STUB,
        budget=SimpleNamespace(max_usd=10.0, max_tokens_fallback=2_000_000),
        workflow=SimpleNamespace(
            verify_when="never",
            verify_retries=2,
            verify_command=("true",),
            metric=SimpleNamespace(goal=None),
        ),
    )
    wf = _wf(
        root=tmp_path,
        config=config,
        mode="run",
        provider=provider,
        dispatcher=MagicMock(),
        max_iterations=10,
    )
    messages = [{"role": "user", "content": [{"type": "text", "text": "TASK:\nfix"}]}]
    with patch("agent6.workflows.loop.chain_commit", return_value="sha1"):
        result = wf._drive_loop(  # pyright: ignore[reportPrivateUsage]
            system="s",
            conversation=Conversation.from_wire(messages),
            tool_calls=0,
            start_iteration=1,
            root_task_id=None,
            original_task="t",
        )
    assert provider.nudges == 2
    assert result.reason == "silent_finish"
    assert result.completed is True


def test_drive_loop_silent_finish_after_real_work_is_honored(tmp_path: Path) -> None:
    """Once an edit has landed, a prose wrap-up is the normal implicit finish
    and must not be bounced by the no-work gate."""
    from agent6.workflows.loop import SILENT_NO_WORK_NUDGE  # pyright: ignore[reportPrivateUsage]

    class ProviderStub:
        def __init__(self) -> None:
            self.calls = 0
            self.nudges = 0

        def call(self, **kwargs: Any) -> ProviderResponse:
            self.calls += 1
            if SILENT_NO_WORK_NUDGE[:22] in str(kwargs["messages"][-1]):
                self.nudges += 1
            if self.calls == 1:
                return _tool_resp(
                    "apply_edit",
                    {"path": "f.py", "edits": [{"old_string": "a", "new_string": "b"}]},
                    tool_id="e1",
                )
            return _resp("Done: applied the fix.")

    class DispatcherStub(_StubDispatcher):
        def dispatch(self, name: str, raw_input: dict[str, Any]) -> ToolResult:
            return RawResult({"applied": True, "path": "f.py"})

    provider = ProviderStub()
    config = SimpleNamespace(
        git=_GIT_STUB,
        budget=SimpleNamespace(max_usd=10.0, max_tokens_fallback=2_000_000),
        workflow=SimpleNamespace(
            verify_when="never",
            verify_retries=2,
            verify_command=("true",),
            metric=SimpleNamespace(goal=None),
        ),
    )
    wf = _wf(
        root=tmp_path,
        config=config,
        mode="run",
        provider=provider,
        dispatcher=DispatcherStub(),
        max_iterations=10,
    )
    messages = [{"role": "user", "content": [{"type": "text", "text": "TASK:\nfix"}]}]
    with patch("agent6.workflows.loop.chain_commit", return_value="sha1"):
        result = wf._drive_loop(  # pyright: ignore[reportPrivateUsage]
            system="s",
            conversation=Conversation.from_wire(messages),
            tool_calls=0,
            start_iteration=1,
            root_task_id=None,
            original_task="t",
        )
    assert provider.nudges == 0
    assert result.reason == "silent_finish"


def test_drive_loop_no_progress_defers_to_metric_runs(tmp_path: Path) -> None:
    """On a metric-optimization run, repeated identical verify failures during
    search are expected, and the metric plateau/early-finish machinery owns
    when the run stops. The no-progress guard must NOT fire (it would truncate
    the budgeted search and end the run completed=false)."""
    from agent6.workflows.loop import NO_PROGRESS_NUDGE  # pyright: ignore[reportPrivateUsage]

    class ProviderStub:
        def __init__(self) -> None:
            self.calls = 0
            self.nudges = 0

        def call(self, **kwargs: Any) -> ProviderResponse:
            self.calls += 1
            if NO_PROGRESS_NUDGE[:28] in str(kwargs["messages"][-1]):
                self.nudges += 1
            if self.calls >= 18:
                return _tool_resp("finish_session", {"summary": "done"}, tool_id="f")
            if self.calls % 2 == 1:
                return _tool_resp(
                    "apply_edit",
                    {"path": "f.py", "edits": [{"old_string": "a", "new_string": "b"}]},
                    tool_id=f"e{self.calls}",
                )
            return _tool_resp("run_verify_command", tool_id=f"v{self.calls}")

    class DispatcherStub(_StubDispatcher):
        def dispatch(self, name: str, raw_input: dict[str, Any]) -> ToolResult:
            if name == "run_verify_command":
                return ExecResult(
                    returncode=1,
                    stdout="",
                    stderr="AssertionError: want 3 got 2",
                    duration_s=0.1,
                    exec_failed=False,
                )
            if name == "apply_edit":
                return RawResult({"applied": True, "path": "f.py"})
            return RawResult({"ok": True})

    provider = ProviderStub()
    # metric configured -> this is an optimization run
    config = SimpleNamespace(
        git=_GIT_STUB,
        budget=SimpleNamespace(max_usd=10.0, max_tokens_fallback=2_000_000),
        workflow=SimpleNamespace(
            verify_when="never",
            verify_retries=2,
            verify_command=("true",),
            metric=SimpleNamespace(goal="minimize"),
        ),
    )
    wf = _wf(
        root=tmp_path,
        config=config,
        mode="run",
        provider=provider,
        dispatcher=DispatcherStub(),
        max_iterations=40,
    )
    messages = [{"role": "user", "content": [{"type": "text", "text": "TASK:\noptimize"}]}]
    with patch("agent6.workflows.loop.chain_commit", return_value="sha1"):
        result = wf._drive_loop(  # pyright: ignore[reportPrivateUsage]
            system="s",
            conversation=Conversation.from_wire(messages),
            tool_calls=0,
            start_iteration=1,
            root_task_id=None,
            original_task="t",
        )
    assert provider.nudges == 0
    assert result.reason != "no_progress"


def test_drive_loop_dedupes_identical_back_to_back_tool_results(tmp_path: Path) -> None:
    """A back-to-back identical (name,args) call whose result bytes are
    identical to the previous one is served a short stub instead of the full
    payload (observed: kimi re-serving a 60KB read_file result 10-12x, growing
    context to 125K tokens). The call still dispatches; a CHANGED result is
    served in full."""

    class ProviderStub:
        def __init__(self) -> None:
            self.calls = 0

        def call(self, **kwargs: Any) -> ProviderResponse:
            self.calls += 1
            if self.calls <= 3:
                return _tool_resp("read_file", {"path": "big.py"}, tool_id=f"r{self.calls}")
            return _tool_resp("finish_session", {"summary": "done"}, tool_id="f")

    big = "X" * 4000

    class DispatcherStub(_StubDispatcher):
        def dispatch(self, name: str, raw_input: dict[str, Any]) -> ToolResult:
            if name == "read_file":
                return RawResult({"content": big, "size": len(big)})
            return RawResult({"ok": True})

    provider = ProviderStub()
    config = SimpleNamespace(
        git=_GIT_STUB,
        budget=SimpleNamespace(max_usd=10.0, max_tokens_fallback=2_000_000),
        workflow=SimpleNamespace(
            verify_when="never",
            verify_retries=2,
            verify_command=("true",),
            metric=SimpleNamespace(goal=None),
        ),
    )
    wf = _wf(
        root=tmp_path,
        config=config,
        mode="run",
        provider=provider,
        dispatcher=DispatcherStub(),
        max_iterations=10,
    )
    messages = [{"role": "user", "content": [{"type": "text", "text": "TASK:\nread"}]}]
    conversation = Conversation.from_wire(messages)
    with patch("agent6.workflows.loop.chain_commit", return_value="sha1"):
        wf._drive_loop(  # pyright: ignore[reportPrivateUsage]
            system="s",
            conversation=conversation,
            tool_calls=0,
            start_iteration=1,
            root_task_id=None,
            original_task="t",
        )
    # collect the served tool_result contents for read_file
    served = []
    for m in conversation.to_wire():
        if m.get("role") == "user" and isinstance(m.get("content"), list):
            for b in m["content"]:
                if isinstance(b, dict) and b.get("type") == "tool_result":
                    served.append(b["content"])
    # first read served in full (contains the payload); the 2nd/3rd deduped
    full = [c for c in served if big[:200] in c]
    stubs = [c for c in served if "identical" in c.lower() and big[:200] not in c]
    assert len(full) == 1, f"expected exactly one full payload, got {len(full)}"
    assert len(stubs) >= 1, f"expected the repeats deduped to a stub, got {stubs}"


def test_drive_loop_tool_error_ladder_nudges_then_stops(tmp_path: Path) -> None:
    """A run that keeps issuing a call failing with the SAME error (a runaway
    grep tripping 'not valid JSON' repeatedly) is nudged, escalated, then
    stopped as reason=tool_error_stuck instead of looping to the cap
    (observed: kimi re-issuing malformed grep until timeout)."""
    from agent6.workflows.loop import (
        TOOL_ERROR_ESCALATION,  # pyright: ignore[reportPrivateUsage]
        TOOL_ERROR_NUDGE,  # pyright: ignore[reportPrivateUsage]
    )

    class ProviderStub:
        def __init__(self) -> None:
            self.calls = 0
            self.nudges = 0
            self.escs = 0

        def call(self, **kwargs: Any) -> ProviderResponse:
            self.calls += 1
            last = str(kwargs["messages"][-1])
            if TOOL_ERROR_NUDGE[:26] in last:
                self.nudges += 1
            if TOOL_ERROR_ESCALATION[:26] in last:
                self.escs += 1
            # keep issuing the same tool with a slightly different (runaway) arg
            # each time — same ERROR signature, different args
            return _tool_resp("read_file", {"path": "x/" * self.calls}, tool_id=f"g{self.calls}")

    from agent6.tools.errors import ToolError as _TE

    class DispatcherStub(_StubDispatcher):
        def dispatch(self, name: str, raw_input: dict[str, Any]) -> ToolResult:
            raise _TE("read_file: the arguments were not valid JSON. Resend the call.")

    provider = ProviderStub()
    config = SimpleNamespace(
        git=_GIT_STUB,
        budget=SimpleNamespace(max_usd=10.0, max_tokens_fallback=2_000_000),
        workflow=SimpleNamespace(
            verify_when="never",
            verify_retries=2,
            verify_command=("true",),
            metric=SimpleNamespace(goal=None),
        ),
    )
    wf = _wf(
        root=tmp_path,
        config=config,
        mode="run",
        provider=provider,
        dispatcher=DispatcherStub(),
        max_iterations=40,
    )
    messages = [{"role": "user", "content": [{"type": "text", "text": "TASK:\nsearch"}]}]
    with patch("agent6.workflows.loop.chain_commit", return_value="sha1"):
        result = wf._drive_loop(  # pyright: ignore[reportPrivateUsage]
            system="s",
            conversation=Conversation.from_wire(messages),
            tool_calls=0,
            start_iteration=1,
            root_task_id=None,
            original_task="t",
        )
    assert provider.nudges == 1
    assert provider.escs == 1
    assert result.reason == "tool_error_stuck"
    assert result.completed is False
    assert result.iterations < 20


def test_drive_loop_denial_streak_gets_policy_nudge_not_malformed(tmp_path: Path) -> None:
    """A streak of policy refusals (ToolDenied) is nudged as 'refused, stop
    retrying', never 'your call is malformed', and the stale binary a REAL
    exec failure recorded first (git at streak 1; the note fires at 2) must
    not be resurfaced by what is pure policy."""
    from agent6.tools.errors import ToolDenied as _TD
    from agent6.workflows.loop import (
        TOOL_DENIED_NUDGE,  # pyright: ignore[reportPrivateUsage]
        TOOL_ERROR_NUDGE,  # pyright: ignore[reportPrivateUsage]
    )

    class ProviderStub:
        def __init__(self) -> None:
            self.calls = 0
            self.denial_nudges = 0
            self.malformed_nudges = 0
            self.reach_notes = 0

        def call(self, **kwargs: Any) -> ProviderResponse:
            self.calls += 1
            last = str(kwargs["messages"][-1])
            if TOOL_DENIED_NUDGE[:30] in last:
                self.denial_nudges += 1
            if TOOL_ERROR_NUDGE[:26] in last:
                self.malformed_nudges += 1
            if "installed on this machine" in last:
                self.reach_notes += 1
            return _tool_resp(
                "run_command",
                {"argv": ["git", "status", f"-{self.calls}"]},
                tool_id=f"c{self.calls}",
            )

    class DispatcherStub(_StubDispatcher):
        def __init__(self) -> None:
            self.calls = 0

        def dispatch(self, name: str, raw_input: dict[str, Any]) -> ToolResult:
            self.calls += 1
            if self.calls == 1:
                # A real exec failure (the jail's 127 shape) records
                # argv[0]="git" in the reachability tracker; a raised ToolError
                # would record nothing (denials/errors never entered the jail).
                return ExecResult(
                    returncode=127,
                    stdout="",
                    stderr="git: command not found or not executable",
                    duration_s=0.0,
                    exec_failed=True,
                )
            raise _TD("run_command not approved (sandbox.run_commands='ask')")

    provider = ProviderStub()
    config = SimpleNamespace(
        git=_GIT_STUB,
        budget=SimpleNamespace(max_usd=10.0, max_tokens_fallback=2_000_000),
        workflow=SimpleNamespace(
            verify_when="never",
            verify_retries=2,
            verify_command=("true",),
            metric=SimpleNamespace(goal=None),
        ),
    )
    wf = _wf(
        root=tmp_path,
        config=config,
        mode="run",
        provider=provider,
        dispatcher=DispatcherStub(),
        max_iterations=40,
    )
    messages = [{"role": "user", "content": [{"type": "text", "text": "TASK:\nship"}]}]
    with patch("agent6.workflows.loop.chain_commit", return_value="sha1"):
        result = wf._drive_loop(  # pyright: ignore[reportPrivateUsage]
            system="s",
            conversation=Conversation.from_wire(messages),
            tool_calls=0,
            start_iteration=1,
            root_task_id=None,
            original_task="t",
        )
    assert provider.denial_nudges >= 1  # the policy wording reached the model
    assert provider.malformed_nudges == 0  # never told its call shape is wrong
    assert provider.reach_notes == 0  # no stale-binary jail misdiagnosis
    assert result.reason == "tool_error_stuck"


def test_drive_loop_tool_error_streak_resets_on_success(tmp_path: Path) -> None:
    """A successful tool call between errors clears the streak, so intermittent
    errors never trip the ladder."""
    from agent6.tools.errors import ToolError as _TE
    from agent6.workflows.loop import TOOL_ERROR_NUDGE  # pyright: ignore[reportPrivateUsage]

    class ProviderStub:
        def __init__(self) -> None:
            self.calls = 0
            self.nudges = 0

        def call(self, **kwargs: Any) -> ProviderResponse:
            self.calls += 1
            if TOOL_ERROR_NUDGE[:26] in str(kwargs["messages"][-1]):
                self.nudges += 1
            if self.calls >= 12:
                return _tool_resp("finish_session", {"summary": "ok"}, tool_id="f")
            return _tool_resp("read_file", {"path": "p"}, tool_id=f"g{self.calls}")

    class DispatcherStub(_StubDispatcher):
        def __init__(self) -> None:
            self.n = 0

        def dispatch(self, name: str, raw_input: dict[str, Any]) -> ToolResult:
            self.n += 1
            if self.n % 2 == 0:  # alternate error / success
                return RawResult({"content": "ok"})
            raise _TE("read_file: bad path")

    provider = ProviderStub()
    config = SimpleNamespace(
        git=_GIT_STUB,
        budget=SimpleNamespace(max_usd=10.0, max_tokens_fallback=2_000_000),
        workflow=SimpleNamespace(
            verify_when="never",
            verify_retries=2,
            verify_command=("true",),
            metric=SimpleNamespace(goal=None),
        ),
    )
    wf = _wf(
        root=tmp_path,
        config=config,
        mode="run",
        provider=provider,
        dispatcher=DispatcherStub(),
        max_iterations=20,
    )
    messages = [{"role": "user", "content": [{"type": "text", "text": "TASK:\ngo"}]}]
    with patch("agent6.workflows.loop.chain_commit", return_value="sha1"):
        result = wf._drive_loop(  # pyright: ignore[reportPrivateUsage]
            system="s",
            conversation=Conversation.from_wire(messages),
            tool_calls=0,
            start_iteration=1,
            root_task_id=None,
            original_task="t",
        )
    assert provider.nudges == 0
    assert result.reason != "tool_error_stuck"


def test_note_verify_result_flags_a_dead_verify(tmp_path: Path) -> None:
    """A failing verify that exited instantly because the runner is absent is
    flagged once with the verify-broken nudge (observed: sympy `python -m
    pytest` with pytest missing, exit 1 in 0.0s); a legitimate slow test
    failure is not flagged."""
    from agent6.workflows.loop import VERIFY_BROKEN_NUDGE  # pyright: ignore[reportPrivateUsage]

    wf = _wf(root=tmp_path, config=MagicMock(), provider=MagicMock(), dispatcher=MagicMock())
    st = _state()
    turn = _turn(iteration=1)
    # dead verify: instant, "No module named pytest"
    wf._note_verify_result(  # pyright: ignore[reportPrivateUsage]
        st,
        turn,
        ExecResult(
            returncode=1,
            stdout="",
            stderr="No module named pytest",
            duration_s=0.02,
            exec_failed=False,
        ),
    )
    texts = [it.text for it in turn.tool_results if isinstance(it, Notice)]
    assert any(VERIFY_BROKEN_NUDGE[:24] in t for t in texts)
    assert st.verify.broken_warned is True

    # a second dead verify does not re-warn
    turn2 = _turn(iteration=2)
    wf._note_verify_result(  # pyright: ignore[reportPrivateUsage]
        st,
        turn2,
        ExecResult(
            returncode=1,
            stdout="",
            stderr="No module named pytest",
            duration_s=0.02,
            exec_failed=False,
        ),
    )
    assert not any(
        isinstance(it, Notice) and "verify-broken" in it.text for it in turn2.tool_results
    )


def test_note_verify_result_does_not_flag_real_failure(tmp_path: Path) -> None:
    from agent6.workflows.loop import VERIFY_BROKEN_NUDGE  # pyright: ignore[reportPrivateUsage]

    wf = _wf(root=tmp_path, config=MagicMock(), provider=MagicMock(), dispatcher=MagicMock())
    st = _state()
    turn = _turn(iteration=1)
    # a real test failure: took real time, ordinary assertion output
    wf._note_verify_result(  # pyright: ignore[reportPrivateUsage]
        st,
        turn,
        ExecResult(
            returncode=1,
            stdout="5 failed, 200 passed",
            stderr="AssertionError: x != y",
            duration_s=12.4,
            exec_failed=False,
        ),
    )
    texts = [it.text for it in turn.tool_results if isinstance(it, Notice)]
    assert not any(VERIFY_BROKEN_NUDGE[:24] in t for t in texts)
    assert st.verify.broken_warned is False


def test_tool_error_spiral_stops_without_blaming_the_sandbox(tmp_path: Path) -> None:
    """A run_command ToolError spiral climbs the nudge ladder and stops, but
    never gets the sandbox-reachability note even for a host-present binary: a
    ToolError never entered the jail, so it says nothing about reachability
    (only repeated exec_failed results do; see the reachability tests)."""

    class ProviderStub:
        def __init__(self) -> None:
            self.calls = 0
            self.reach_hits = 0

        def call(self, **kwargs: Any) -> ProviderResponse:
            self.calls += 1
            if "sandbox-\n" in str(kwargs["messages"][-1]) or "reachability" in str(
                kwargs["messages"][-1]
            ):
                self.reach_hits += 1
            return _tool_resp(
                "run_command", {"argv": ["python3", "-c", "x"]}, tool_id=f"c{self.calls}"
            )

    from agent6.tools.errors import ToolError as _TE

    class DispatcherStub(_StubDispatcher):
        def dispatch(self, name: str, raw_input: dict[str, Any]) -> ToolResult:
            raise _TE("python3: boom in the sandbox")

    provider = ProviderStub()
    config = SimpleNamespace(
        git=_GIT_STUB,
        budget=SimpleNamespace(max_usd=10.0, max_tokens_fallback=2_000_000),
        workflow=SimpleNamespace(
            verify_when="never",
            verify_retries=2,
            verify_command=("true",),
            metric=SimpleNamespace(goal=None),
        ),
    )
    wf = _wf(
        root=tmp_path,
        config=config,
        mode="run",
        provider=provider,
        dispatcher=DispatcherStub(),
        max_iterations=20,
    )
    messages = [{"role": "user", "content": [{"type": "text", "text": "TASK:\ngo"}]}]
    with patch("agent6.workflows.loop.chain_commit", return_value="sha1"):
        result = wf._drive_loop(  # pyright: ignore[reportPrivateUsage]
            system="s",
            conversation=Conversation.from_wire(messages),
            tool_calls=0,
            start_iteration=1,
            root_task_id=None,
            original_task="t",
        )
    assert provider.reach_hits == 0  # no sandbox blame for a non-jail failure
    assert result.reason == "tool_error_stuck"


def test_drive_loop_gateless_settle_never_claims_verify_passed(tmp_path: Path) -> None:
    """A gateless run (no verify command configured or inferable) that commits
    work and goes idle settles as reason='settled' with all_passed=False and an
    honest summary. It ended 'passed / verify passed' before, with zero verify
    executions in the whole run (observed live on an empty-repo build)."""

    class ProviderStub:
        def __init__(self) -> None:
            self.calls = 0

        def call(self, **kwargs: Any) -> ProviderResponse:
            self.calls += 1
            if self.calls == 1:
                return _tool_resp(
                    "apply_edit",
                    {"path": "a.py", "edits": [{"kind": "create", "new_string": "x = 1\n"}]},
                    tool_id="e1",
                )
            # Then the worker goes idle: read-only calls, no edits, no finish.
            return _tool_resp("read_file", {"path": "a.py"}, tool_id=f"r{self.calls}")

    class DispatcherStub(_StubDispatcher):
        def dispatch(self, name: str, raw_input: dict[str, Any]) -> ToolResult:
            return ExecResult(
                returncode=0, stdout="ok", stderr="", duration_s=0.1, exec_failed=False
            )

    provider = ProviderStub()
    config = SimpleNamespace(
        git=_GIT_STUB,
        budget=SimpleNamespace(max_usd=10.0, max_tokens_fallback=2_000_000),
        workflow=SimpleNamespace(
            verify_when="never",
            verify_retries=2,
            verify_command=(),  # GATELESS
            verify_infer=True,
            metric=SimpleNamespace(goal=None),
        ),
    )
    events: list[dict[str, Any]] = []

    class _Events:
        def emit(self, event_type: str, /, **fields: Any) -> None:
            events.append({"type": event_type, **fields})

    wf = _wf(
        root=tmp_path,
        config=config,
        mode="run",
        provider=provider,
        dispatcher=DispatcherStub(),
        max_iterations=40,
        events=_Events(),
    )
    messages = [{"role": "user", "content": [{"type": "text", "text": "TASK:\nbuild"}]}]
    with patch("agent6.workflows.loop.chain_commit", return_value="sha1"):
        result = wf._drive_loop(  # pyright: ignore[reportPrivateUsage]
            system="s",
            conversation=Conversation.from_wire(messages),
            tool_calls=0,
            start_iteration=1,
            root_task_id=None,
            original_task="t",
        )
    assert result.completed is True
    assert result.reason == "settled"
    assert "no verify" in result.summary
    assert "verify passed" not in result.summary
    ends = [e for e in events if e["type"] == "session.end"]
    assert ends and ends[-1]["reason"] == "settled" and ends[-1]["all_passed"] is False


def test_drive_loop_interactive_stop_never_ends_passed(tmp_path: Path) -> None:
    """The REPL hook's "stop" ends the run deliberately, not as verified
    success: reason='interactive_stop' with all_passed=False on the session.end
    event (it used to route through the passed emitter with zero verifies)."""

    class ProviderStub:
        def call(self, **kwargs: Any) -> ProviderResponse:
            return _tool_resp(
                "apply_edit",
                {"path": "a.py", "edits": [{"kind": "create", "new_string": "x = 1\n"}]},
                tool_id="e1",
            )

    class DispatcherStub(_StubDispatcher):
        def dispatch(self, name: str, raw_input: dict[str, Any]) -> ToolResult:
            return ExecResult(
                returncode=0, stdout="ok", stderr="", duration_s=0.1, exec_failed=False
            )

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
    events: list[dict[str, Any]] = []

    class _Events:
        def emit(self, event_type: str, /, **fields: Any) -> None:
            events.append({"type": event_type, **fields})

    def _stop_hook(_i: int, _sha: str) -> Literal["continue", "stop"]:
        return "stop"

    wf = _wf(
        root=tmp_path,
        config=config,
        mode="run",
        provider=ProviderStub(),
        dispatcher=DispatcherStub(),
        max_iterations=10,
        events=_Events(),
        after_auto_commit=_stop_hook,
    )
    messages = [{"role": "user", "content": [{"type": "text", "text": "TASK:\nt"}]}]
    with patch("agent6.workflows.loop.chain_commit", return_value="sha1"):
        result = wf._drive_loop(  # pyright: ignore[reportPrivateUsage]
            system="s",
            conversation=Conversation.from_wire(messages),
            tool_calls=0,
            start_iteration=1,
            root_task_id=None,
            original_task="t",
        )
    assert result.completed is True
    assert result.reason == "interactive_stop"
    ends = [e for e in events if e["type"] == "session.end"]
    assert ends and ends[-1]["reason"] == "interactive_stop"
    assert ends[-1]["all_passed"] is False  # a stop is deliberate, never "passed"


def test_drive_loop_repl_undo_takes_the_steer_undo_path(tmp_path: Path) -> None:
    """The REPL hook's "undo" is the loop's own /undo (fork back before the
    last message): the run ends `undone` naming the fork, exactly as a steer
    /undo does, and never touches git itself."""

    class ProviderStub:
        def call(self, **kwargs: Any) -> ProviderResponse:
            return _tool_resp(
                "apply_edit",
                {"path": "a.py", "edits": [{"kind": "create", "new_string": "x = 1\n"}]},
                tool_id="e1",
            )

    class DispatcherStub(_StubDispatcher):
        def dispatch(self, name: str, raw_input: dict[str, Any]) -> ToolResult:
            return ExecResult(
                returncode=0, stdout="ok", stderr="", duration_s=0.1, exec_failed=False
            )

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
    events: list[dict[str, Any]] = []

    class _Events:
        def emit(self, event_type: str, /, **fields: Any) -> None:
            events.append({"type": event_type, **fields})

    def _undo_hook(_i: int, _sha: str) -> Literal["continue", "stop", "undo"]:
        return "undo"

    wf = _wf(
        root=tmp_path,
        config=config,
        mode="run",
        provider=ProviderStub(),
        dispatcher=DispatcherStub(),
        max_iterations=10,
        events=_Events(),
        after_auto_commit=_undo_hook,
    )
    wf.undo_forker = lambda: ("forked-child-ID", "t")
    messages = [{"role": "user", "content": [{"type": "text", "text": "TASK:\nt"}]}]
    with patch("agent6.workflows.loop.chain_commit", return_value="sha1"):
        result = wf._drive_loop(  # pyright: ignore[reportPrivateUsage]
            system="s",
            conversation=Conversation.from_wire(messages),
            tool_calls=0,
            start_iteration=1,
            root_task_id=None,
            original_task="t",
        )
    assert result.reason == "undone"
    assert "forked-child-ID" in result.summary
    undone = [e for e in events if e["type"] == "session.undone"]
    assert undone and undone[-1]["new_session_id"] == "forked-child-ID"
    # An undo is the operator's own end: it journals a session.end (reason
    # undone -> the listings' "stopped"), or the run read "stale" (dead worker,
    # no end) the moment the fork was cut.
    ends = [e for e in events if e["type"] == "session.end"]
    assert ends and ends[-1]["reason"] == "undone" and ends[-1]["all_passed"] is False


def test_drive_loop_gateless_run_adopts_verify_when_the_repo_materializes(
    tmp_path: Path,
) -> None:
    """Preflight inference on an empty repo finds nothing; the run then creates
    a recognizable project. At the next gateless auto-commit the deterministic
    tiers re-run and the verify is ADOPTED: config and dispatcher pick it up
    and the model is told, so the rest of the run is gated instead of
    finishing a whole build unverified."""
    from agent6.config import Config

    # What the run "just created" before its first commit.
    (tmp_path / "pyproject.toml").write_text('[project]\nname = "x"\n', encoding="utf-8")

    class ProviderStub:
        def __init__(self) -> None:
            self.calls = 0
            self.adoption_notices = 0

        def call(self, **kwargs: Any) -> ProviderResponse:
            self.calls += 1
            if "verify command was adopted" in str(kwargs["messages"][-1]):
                self.adoption_notices += 1
            if self.calls == 1:
                return _tool_resp(
                    "apply_edit",
                    {"path": "pyproject.toml", "edits": [{"kind": "create", "new_string": "x"}]},
                    tool_id="e1",
                )
            return _tool_resp("read_file", {"path": "pyproject.toml"}, tool_id=f"r{self.calls}")

    class DispatcherStub(_StubDispatcher):
        def __init__(self) -> None:
            self.adopted: tuple[str, ...] | None = None

        def adopt_verify_command(self, argv: tuple[str, ...]) -> bool:
            self.adopted = argv
            return True

        def dispatch(self, name: str, raw_input: dict[str, Any]) -> ToolResult:
            return ExecResult(
                returncode=0, stdout="ok", stderr="", duration_s=0.1, exec_failed=False
            )

    provider = ProviderStub()
    dispatcher = DispatcherStub()
    wf = _wf(
        root=tmp_path,
        config=Config(),  # real config: verify_command defaults empty (gateless)
        mode="run",
        provider=provider,
        dispatcher=dispatcher,
        max_iterations=40,
    )
    messages = [{"role": "user", "content": [{"type": "text", "text": "TASK:\nbuild"}]}]
    with patch("agent6.workflows.loop.chain_commit", return_value="sha1"):
        result = wf._drive_loop(  # pyright: ignore[reportPrivateUsage]
            system="s",
            conversation=Conversation.from_wire(messages),
            tool_calls=0,
            start_iteration=1,
            root_task_id=None,
            original_task="t",
        )
    assert dispatcher.adopted is not None  # the dispatcher gates run_verify now
    assert tuple(wf.config.workflow.verify_command) == dispatcher.adopted
    assert provider.adoption_notices >= 1  # the gate flip was said to the model
    assert result.completed is True
    # The worker then idled without ever running the adopted verify; the
    # settle summary must not claim no command existed (one demonstrably did).
    assert result.reason == "settled"
    assert "adopted verify never passed" in result.summary
    assert "no verify command existed" not in result.summary


def test_drive_loop_gateless_adoption_declines_an_unexecutable_verify(
    tmp_path: Path,
) -> None:
    """When the dispatcher refuses the inferred command (its binary is not on
    the jail PATH), the run stays gateless: adopting a gate the sandbox cannot
    execute would turn the honest settle into an unexecutable-verify abort."""
    from agent6.config import Config

    (tmp_path / "pyproject.toml").write_text('[project]\nname = "x"\n', encoding="utf-8")

    class ProviderStub:
        def __init__(self) -> None:
            self.calls = 0
            self.adoption_notices = 0

        def call(self, **kwargs: Any) -> ProviderResponse:
            self.calls += 1
            if "verify command was adopted" in str(kwargs["messages"][-1]):
                self.adoption_notices += 1
            if self.calls == 1:
                return _tool_resp(
                    "apply_edit",
                    {"path": "pyproject.toml", "edits": [{"kind": "create", "new_string": "x"}]},
                    tool_id="e1",
                )
            return _tool_resp("read_file", {"path": "pyproject.toml"}, tool_id=f"r{self.calls}")

    class DispatcherStub(_StubDispatcher):
        def adopt_verify_command(self, argv: tuple[str, ...]) -> bool:
            return False  # the jail cannot execute the inferred runner

        def dispatch(self, name: str, raw_input: dict[str, Any]) -> ToolResult:
            return ExecResult(
                returncode=0, stdout="ok", stderr="", duration_s=0.1, exec_failed=False
            )

    provider = ProviderStub()
    wf = _wf(
        root=tmp_path,
        config=Config(),
        mode="run",
        provider=provider,
        dispatcher=DispatcherStub(),
        max_iterations=40,
    )
    messages = [{"role": "user", "content": [{"type": "text", "text": "TASK:\nbuild"}]}]
    with patch("agent6.workflows.loop.chain_commit", return_value="sha1"):
        result = wf._drive_loop(  # pyright: ignore[reportPrivateUsage]
            system="s",
            conversation=Conversation.from_wire(messages),
            tool_calls=0,
            start_iteration=1,
            root_task_id=None,
            original_task="t",
        )
    assert tuple(wf.config.workflow.verify_command) == ()  # still gateless
    assert provider.adoption_notices == 0  # no false gate-flip message
    assert result.reason == "settled"
    assert "no verify command existed" in result.summary


def _run_command_provider(calls_before_idle: int) -> Any:
    """A provider that issues `calls_before_idle` run_command calls, then goes
    read-only, counting reachability NOTEs it is served along the way."""

    class ProviderStub:
        def __init__(self) -> None:
            self.calls = 0
            self.reachability_notes = 0

        def call(self, **kwargs: Any) -> ProviderResponse:
            self.calls += 1
            if "the sandbox cannot execute it" in str(kwargs["messages"][-1]):
                self.reachability_notes += 1
            if self.calls <= calls_before_idle:
                return _tool_resp(
                    "run_command", {"argv": ["sh", "-c", "true"]}, tool_id=f"c{self.calls}"
                )
            return _tool_resp("read_file", {"path": "a"}, tool_id=f"r{self.calls}")

    return ProviderStub()


def test_reachability_note_fires_on_repeated_jail_exec_failure(tmp_path: Path) -> None:
    """Two consecutive jail exec failures (exec_failed, not a nonzero exit) of
    the same host-present binary emit loop.sandbox_tool_unreachable ONCE and
    tell the model once; finalize's operator warning reads that event."""

    class DispatcherStub(_StubDispatcher):
        def dispatch(self, name: str, raw_input: dict[str, Any]) -> ToolResult:
            if name == "run_command":
                return ExecResult(
                    returncode=127,
                    stdout="",
                    stderr="sh: command not found or not executable",
                    duration_s=0.0,
                    exec_failed=True,
                )
            return ExecResult(
                returncode=0, stdout="ok", stderr="", duration_s=0.0, exec_failed=False
            )

    provider = _run_command_provider(calls_before_idle=4)
    events: list[dict[str, Any]] = []

    class _Events:
        def emit(self, event_type: str, /, **fields: Any) -> None:
            events.append({"type": event_type, **fields})

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
    wf = _wf(
        root=tmp_path,
        config=config,
        mode="run",
        provider=provider,
        dispatcher=DispatcherStub(),
        max_iterations=8,
        events=_Events(),
    )
    messages = [{"role": "user", "content": [{"type": "text", "text": "TASK:\nt"}]}]
    wf._drive_loop(  # pyright: ignore[reportPrivateUsage]
        system="s",
        conversation=Conversation.from_wire(messages),
        tool_calls=0,
        start_iteration=1,
        root_task_id=None,
        original_task="t",
    )
    unreachable = [e for e in events if e["type"] == "loop.sandbox_tool_unreachable"]
    assert [e["binary"] for e in unreachable] == ["sh"]  # once, at the 2nd failure
    assert provider.reachability_notes >= 1  # the model was told


def test_reachability_note_never_fires_on_a_validation_error(tmp_path: Path) -> None:
    """A run_command rejected at input validation never entered the jail;
    it must not seed the reachability diagnosis (observed live: an `env`
    extra-input rejection produced a finalize warning blaming the sandbox
    for a binary that later ran fine)."""

    from agent6.tools.errors import ToolError as _TE

    class DispatcherStub(_StubDispatcher):
        def dispatch(self, name: str, raw_input: dict[str, Any]) -> ToolResult:
            if name == "run_command":
                raise _TE("1 validation error for RunCommandInput: env extra_forbidden")
            return ExecResult(
                returncode=0, stdout="ok", stderr="", duration_s=0.0, exec_failed=False
            )

    provider = _run_command_provider(calls_before_idle=4)
    events: list[dict[str, Any]] = []

    class _Events:
        def emit(self, event_type: str, /, **fields: Any) -> None:
            events.append({"type": event_type, **fields})

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
    wf = _wf(
        root=tmp_path,
        config=config,
        mode="run",
        provider=provider,
        dispatcher=DispatcherStub(),
        max_iterations=8,
        events=_Events(),
    )
    messages = [{"role": "user", "content": [{"type": "text", "text": "TASK:\nt"}]}]
    wf._drive_loop(  # pyright: ignore[reportPrivateUsage]
        system="s",
        conversation=Conversation.from_wire(messages),
        tool_calls=0,
        start_iteration=1,
        root_task_id=None,
        original_task="t",
    )
    assert not any(e["type"] == "loop.sandbox_tool_unreachable" for e in events)
    assert provider.reachability_notes == 0


def test_load_repo_summary_tolerates_a_broken_agents_md(tmp_path: Path) -> None:
    """A non-UTF-8 (Windows-1252 curly quote) or unreadable AGENTS.md degrades
    to a replaced/empty read; unguarded, it raised AFTER session.start with no
    session.end -- a dead run listed "running" then "stale". The tolerant pattern
    already existed for the loop's own reads; the startup summary was the
    outlier."""
    from agent6.workflows._context import load_repo_summary

    (tmp_path / "AGENTS.md").write_bytes(b"Style: use \x93smart quotes\x94\n")
    summary = load_repo_summary(tmp_path)
    assert "smart quotes" in summary.agents_md  # lossy, present, no crash


def test_refused_finish_tool_is_not_captured_as_a_finish() -> None:
    """A finish tool the dispatcher REFUSED (ToolError -- e.g. a hallucinated
    finish_planning in run mode, which the mode backstop blocks) must not be
    captured as a finish signal: the refusal is an error tool_result the model
    reads and recovers from. Capturing it anyway ended the run completed=True
    -- for finish_planning even all_passed=True -- bypassing every finish
    gate."""
    from agent6.tools.dispatch import ToolError
    from agent6.workflows._conversation import ToolUse
    from agent6.workflows.loop import _TurnState  # pyright: ignore[reportPrivateUsage]

    dispatcher = MagicMock()
    dispatcher.dispatch.side_effect = ToolError("finish_planning is not available in run mode")
    wf = _wf(mode="run", dispatcher=dispatcher)
    turn = _TurnState(
        iteration=1,
        resp=MagicMock(),
        assistant=AssistantTurn(
            raw_content=(),
            tool_uses=(
                ToolUse(
                    id="tu1",
                    name="finish_planning",
                    input={"summary": "all done", "plan_markdown": "# x"},
                ),
            ),
        ),
    )
    out = wf._turn_dispatch_tools(_state(), turn)  # pyright: ignore[reportPrivateUsage]
    assert out is None  # the refusal is served as an error result, not an abort
    assert turn.finish_signal is None  # and never captured as a finish


def test_finish_dispatch_is_not_work_for_the_standing_streak() -> None:
    """A dispatched finish_session must not advance ok_tool_calls: a standing
    goal's revoked finish would otherwise reset the fruitless streak every
    round, and standing_patience could never engage (the run span a
    3-second finish->revoke->finish loop until killed)."""
    from agent6.tools.results import FinishSessionResult
    from agent6.workflows._conversation import ToolUse
    from agent6.workflows.loop import _TurnState  # pyright: ignore[reportPrivateUsage]

    dispatcher = MagicMock()
    dispatcher.dispatch.return_value = FinishSessionResult(summary_text="done", result=None)
    wf = _wf(mode="run", dispatcher=dispatcher)
    state = _state()
    turn = _TurnState(
        iteration=1,
        resp=MagicMock(),
        assistant=AssistantTurn(
            raw_content=(),
            tool_uses=(ToolUse(id="tu1", name="finish_session", input={"summary": "done"}),),
        ),
    )
    wf._turn_dispatch_tools(state, turn)  # pyright: ignore[reportPrivateUsage]
    assert state.ok_tool_calls == 0  # a control verb is not work

    worked = _TurnState(
        iteration=2,
        resp=MagicMock(),
        assistant=AssistantTurn(
            raw_content=(),
            tool_uses=(ToolUse(id="tu2", name="read_file", input={"path": "x"}),),
        ),
    )
    wf._turn_dispatch_tools(state, worked)  # pyright: ignore[reportPrivateUsage]
    assert state.ok_tool_calls == 1


def test_stop_request_honored_after_a_prose_turn(tmp_path: Path) -> None:
    """ "Stop after this step" is honored at the end of EVERY completed
    iteration, including one with no tool_use: the boundary poll only ran on
    the tool path, so a model answering in prose kept the run calling the
    provider with the stop marker pending forever."""

    calls = {"n": 0}

    class ProviderStub:
        def call(self, **kwargs: Any) -> ProviderResponse:
            del kwargs
            calls["n"] += 1
            return ProviderResponse(
                text="still thinking about the approach",
                tool_uses=(),
                stop_reason="end_turn",
                input_tokens=1,
                output_tokens=1,
                cache_read_tokens=0,
                cache_creation_tokens=0,
                raw={"content": [{"type": "text", "text": "still thinking about the approach"}]},
            )

    cleared = {"n": 0}
    config = SimpleNamespace(
        git=_GIT_STUB,
        budget=SimpleNamespace(max_usd=10.0, max_tokens_fallback=2_000_000),
        workflow=SimpleNamespace(
            verify_when="never",
            verify_retries=2,
            verify_command=("true",),
            metric=SimpleNamespace(goal=None),
        ),
    )
    wf = _wf(
        root=tmp_path,
        config=config,
        provider=ProviderStub(),
        dispatcher=MagicMock(),
        max_iterations=5,
        stop_requested=lambda: True,
        stop_clear=lambda: cleared.__setitem__("n", cleared["n"] + 1),
    )
    messages = [{"role": "user", "content": [{"type": "text", "text": "TASK:\ngo"}]}]
    result = wf._drive_loop(  # pyright: ignore[reportPrivateUsage]
        system="system",
        conversation=Conversation.from_wire(messages),
        tool_calls=0,
        start_iteration=1,
        root_task_id=None,
        original_task="t",
    )
    assert result.reason == "steer_abort"
    assert result.completed is False
    assert calls["n"] == 1  # stopped at the FIRST boundary; no further provider calls
    assert cleared["n"] == 1  # the marker was consumed, not left pending


def test_metric_plateau_over_a_stale_verify_is_not_passed() -> None:
    """The plateau stop grounds on the tree like its sibling clean ends
    (finish_session, verify_settled): a same-turn edit AFTER the green verify
    means nothing verified the FINAL tree, so the end must not claim
    all_passed=True."""
    from agent6.workflows._conversation import ToolUse
    from agent6.workflows.loop import _TurnState  # pyright: ignore[reportPrivateUsage]

    ev = _EventCapture()
    wf = _wf(mode="run", config=_cfg_with_verify(), events=ev, root=Path("/tmp"))
    turn = _TurnState(
        iteration=7,
        resp=MagicMock(),
        assistant=AssistantTurn(
            raw_content=(),
            tool_uses=(ToolUse(id="tu1", name="apply_edit", input={}),),
        ),
        plateau_should_stop=True,
        metric_plateau_finish="score plateaued at 10",
    )
    state = _state(
        ever_edited=True,
        # The green verify predates the last edit.
        verify=VerifyVerdict(ever_passed=True, last_ok=True, edited_since=True),
    )
    with patch.object(wf, "_worktree_dirty", return_value=False):
        result = wf._turn_stop_checks(state, turn, Conversation())  # pyright: ignore[reportPrivateUsage]
    assert result is not None and result.reason == "metric_plateau"
    ends = [e for e in ev.events if e["type"] == "session.end"]
    assert ends and ends[-1]["all_passed"] is False


def test_metric_plateau_over_a_green_tree_stays_passed() -> None:
    """The mirror: a verified-green tree at the plateau still ends passed."""
    from agent6.workflows._conversation import ToolUse
    from agent6.workflows.loop import _TurnState  # pyright: ignore[reportPrivateUsage]

    ev = _EventCapture()
    wf = _wf(mode="run", config=_cfg_with_verify(), events=ev, root=Path("/tmp"))
    turn = _TurnState(
        iteration=7,
        resp=MagicMock(),
        assistant=AssistantTurn(
            raw_content=(),
            tool_uses=(ToolUse(id="tu1", name="run_verify_command", input={}),),
        ),
        plateau_should_stop=True,
        metric_plateau_finish="score plateaued at 10",
    )
    state = _state(
        ever_edited=True, verify=VerifyVerdict(ever_passed=True, last_ok=True, edited_since=False)
    )
    with patch.object(wf, "_worktree_dirty", return_value=False):
        result = wf._turn_stop_checks(state, turn, Conversation())  # pyright: ignore[reportPrivateUsage]
    assert result is not None and result.reason == "metric_plateau"
    ends = [e for e in ev.events if e["type"] == "session.end"]
    assert ends and ends[-1]["all_passed"] is True


def test_a_red_verify_finish_still_passes_its_root_tasks() -> None:
    """A deliberate end over a red verify passed its roots on the settled path
    and not on the finish_session/metric_plateau path, so the same epistemic state
    (completed, not verify-green) left one of them reading `tasks 0/1` forever.
    The DAG tracks work items; the run-level word carries the verify truth."""

    class _FakeClient:
        def __init__(self) -> None:
            self._nodes: dict[str, dict[str, Any]] = {
                "root1": {"parent_id": None, "status": "pending"}
            }
            self.passed: list[str] = []

        def nodes(self) -> dict[str, Any]:
            return _typed(self._nodes)

        def update_status(self, intent: Any) -> None:
            self.passed.append(intent.id)
            self._nodes[intent.id]["status"] = intent.new_status

    fake = _FakeClient()
    wf = _wf(
        curator=fake,
        config=MagicMock(
            git=_GIT_STUB,
            budget=SimpleNamespace(max_usd=10.0, max_tokens_fallback=2_000_000),
            prompt=MagicMock(system_prompt_file=""),
            workflow=MagicMock(verify_command=("false",), verify_when="never", verify_retries=2),
        ),
    )
    state = _state()
    state.verify.last_ok = False  # red tree: all_passed must stay False
    state.verify.edited_since = True
    events: list[dict[str, Any]] = []
    wf._emit = lambda event_type, **fields: events.append({"type": event_type, **fields})  # pyright: ignore[reportPrivateUsage]

    wf._emit_run_end_grounded(reason="finish_session", iteration=3, state=state)  # pyright: ignore[reportPrivateUsage]

    (end,) = [e for e in events if e["type"] == "session.end"]
    assert end["all_passed"] is False  # the verify truth is unchanged...
    assert fake.passed == ["root1"]  # ...and the work item is no longer pending


def test_an_operator_stop_names_the_worktree_it_leaves_dirty(tmp_path: Path) -> None:
    """An operator stop deliberately does NOT checkpoint -- committing over
    someone taking over would remove their choice to discard -- but it said
    nothing, so uncommitted work was invisible to `sessions diff` and `sessions merge`
    with no hint it existed. A clean tree adds nothing."""
    import subprocess

    subprocess.run(["git", "init", "-q", "-b", "main", str(tmp_path)], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "config", "user.email", "t@t"], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "config", "user.name", "t"], check=True)
    (tmp_path / "a.txt").write_text("x\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(tmp_path), "add", "-A"], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "commit", "-qm", "init"], check=True)

    wf = _wf(root=tmp_path, mode="run")
    assert wf._dirty_tree_note() == ""  # pyright: ignore[reportPrivateUsage]

    (tmp_path / "a.txt").write_text("edited\n", encoding="utf-8")
    (tmp_path / "new.txt").write_text("untracked\n", encoding="utf-8")
    note = wf._dirty_tree_note()  # pyright: ignore[reportPrivateUsage]
    assert "worktree left dirty" in note
    assert "2 file" in note  # the real count, not a capped one


def test_parallel_group_counter_reaches_disk_before_the_group_runs(tmp_path: Path) -> None:
    """The counter names each group's lanes (<run-id>-p<seq>-l<i>), but it was
    bumped inside the operator boundary -- which runs AFTER the iteration's
    snapshot -- so it stayed in RAM for the whole time the group blocked. A
    crash there resumed with the old value and the next /parallel re-used p1:
    either the lane clones already existed and every lane failed, or (once the
    first group's clones were cleaned up) the lanes ran and BILLED before
    import_run refused their already-existing branches."""
    import json

    from agent6.directive import Segment
    from agent6.workflows.subrun import LaneResult, LaneSpec

    snap = tmp_path / "loop_state.json"
    at_spawn: dict[str, Any] = {}

    def spawner(lanes: Any, group: str, *, at: str | None = None) -> list[Any]:
        at_spawn["group"] = group
        at_spawn["persisted"] = json.loads(snap.read_text(encoding="utf-8"))[
            "parallel_groups_dispatched"
        ]
        return [
            LaneResult(
                spec=LaneSpec(
                    lane=i, session_id=f"run-{group}-l{i}", workdir=tmp_path, model=lane.model
                ),
                session_dir=tmp_path,
                branch="b",
                ok=False,
                error="lane failed",
            )
            for i, lane in enumerate(lanes, start=1)
        ]

    wf = _wf(root=tmp_path, mode="run", lane_spawner=spawner, resume_state_path=snap)
    conversation = Conversation.from_wire(
        [{"role": "user", "content": [{"type": "text", "text": "go"}]}]
    )
    wf._dispatch_parallel(  # pyright: ignore[reportPrivateUsage]
        conversation, 3, _state(), [Segment(spec="", task="do the thing")]
    )
    assert at_spawn["group"] == "p1"
    assert at_spawn["persisted"] == 1, "the bump must be on disk before the group blocks"


def test_steer_abort_names_the_dirty_worktree_like_its_siblings(tmp_path: Path) -> None:
    """The pause-menu / front-end Stop consumed at the boundary is the fourth
    operator end, and the only one that said nothing about the worktree it
    leaves uncommitted. The same Stop delivered mid-stream did say so, so one
    operator action reported two different truths depending on timing."""
    import subprocess

    subprocess.run(["git", "init", "-q", "-b", "main", str(tmp_path)], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "config", "user.email", "t@t"], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "config", "user.name", "t"], check=True)
    (tmp_path / "a.txt").write_text("x\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(tmp_path), "add", "-A"], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "commit", "-qm", "init"], check=True)
    (tmp_path / "a.txt").write_text("edited after the last checkpoint\n", encoding="utf-8")

    wf = _wf(root=tmp_path, mode="run")
    res = wf._steer_outcome("abort", 4, _state())  # pyright: ignore[reportPrivateUsage]
    assert res is not None
    assert res.reason == "steer_abort"
    assert "worktree left dirty" in res.summary


def test_a_second_restart_carries_the_first_summary_forward() -> None:
    """The prior restart's summary rides at the HEAD of the post-restart history
    and the summariser's transcript is tail-clipped, so it was the first thing
    dropped: the second summary began at the first restart while the preamble
    told the worker everything it had done was captured below. It must reach the
    summariser out-of-band, like pins."""
    from agent6.prompts.revision import context_restart_notice

    summariser = MagicMock()
    summariser.call.return_value = _resp("second summary")
    wf = _wf(
        summariser_provider=summariser,
        compact_drop_at_chars=10**9,
        compact_summarise_at_chars=10**9,
        compact_requested=lambda: "",
    )
    # A conversation that already carries one restart, then plenty of new work
    # so the tail clip has something to prefer over the notice.
    restart = context_restart_notice("run") + "SUMMARY-1: found the parser bug in a.md"
    history: list[dict[str, Any]] = [
        {"role": "user", "content": [{"type": "text", "text": "TASK:\noptimize"}]},
        {"role": "user", "content": [{"type": "text", "text": restart}]},
    ]
    # Enough work since that restart to overflow the summariser's 60k tail
    # clip -- which is exactly when the notice at the head gets dropped.
    history += _long_history(120)[1:]

    assert _compact_via_wire(wf, history) is True
    sent = str(summariser.call.call_args)
    assert "SUMMARY-1" in sent, "the first restart's summary never reached the summariser"


def test_the_frontier_executes_the_order_the_children_list_shows() -> None:
    """`list_tasks` and every task tree render a parent's children in its
    `children` order, but the frontier walked node ids (creation order), so a
    reordered or positionally-inserted child was shown in one order and
    executed in another."""
    from agent6.workflows._dag_focus import (
        first_ready_subtask,  # pyright: ignore[reportPrivateUsage]
    )

    nodes = _typed(
        {
            "root": {"parent_id": None, "status": "in_progress", "children": ("b", "a")},
            "a": {"parent_id": "root", "status": "pending"},
            "b": {"parent_id": "root", "status": "pending"},
        }
    )
    assert first_ready_subtask(nodes) == "b", "the frontier ignored the children order"


def test_the_frontier_walks_depth_first_through_children() -> None:
    """A decomposed child's own leaves come before its later siblings, the
    order the tree shows top to bottom."""
    from agent6.workflows._dag_focus import (
        first_ready_subtask,  # pyright: ignore[reportPrivateUsage]
    )

    nodes = _typed(
        {
            "root": {"parent_id": None, "status": "in_progress", "children": ("p", "z")},
            "p": {"parent_id": "root", "status": "in_progress", "children": ("p2", "p1")},
            "p1": {"parent_id": "p", "status": "pending"},
            "p2": {"parent_id": "p", "status": "pending"},
            "z": {"parent_id": "root", "status": "pending"},
        }
    )
    assert first_ready_subtask(nodes) == "p2"


def test_steer_undo_signal() -> None:
    """`/undo` typed as a steer -> the "undo" sentinel, no injected message."""
    cleared: list[bool] = []
    wf = _wf(
        steer_requested=lambda: True,
        steer_clear=lambda: cleared.append(True),
        steer_prompt=lambda: "/undo",
    )
    messages: list[dict[str, Any]] = []
    result = _steer_via_wire(wf, messages, iteration=4, state=_state())
    assert result == "undo"
    assert cleared == [True]
    assert messages == [], "/undo must not inject a message"


def _standing_nodes() -> Any:
    """root -> one standing child, ready (the queue is empty)."""
    return _typed(
        {
            "a": {"children": ("b",)},
            "b": {"parent_id": "a", "standing": True},
        }
    )


def test_standing_default_never_self_quits_and_escalates() -> None:
    """At the default standing_patience -1, a fruitless quiet round never
    ends the run by itself: every re-entry lands, and fruitless ones carry
    the escalating dig-deeper nudge (the run ends on budget/cap/operator).
    A refused tool call is not work; an executed one resets the streak."""
    curator = MagicMock()
    curator.nodes.return_value = _standing_nodes()
    wf = _wf(mode="run", curator=curator, budget=None)
    conv = Conversation()
    state = _state(ever_edited=True, verify=VerifyVerdict(ever_passed=True))
    first = wf._handle_silent_finish("Done.", conv, state, iteration=3)  # pyright: ignore[reportPrivateUsage]
    assert first is None
    assert "standing task" in conv.to_wire()[-1]["content"][0]["text"]
    # Quiet again with no executed call: still absorbed, nudge escalates.
    state.tool_calls += 1  # a REFUSED call is not work
    second = wf._handle_silent_finish("Done.", conv, state, iteration=4)  # pyright: ignore[reportPrivateUsage]
    assert second is None
    text = conv.to_wire()[-1]["content"][0]["text"]
    assert "different approach" in text and "fruitless round 1" in text
    third = wf._handle_silent_finish("Done.", conv, state, iteration=5)  # pyright: ignore[reportPrivateUsage]
    assert third is None
    assert "fruitless round 2" in conv.to_wire()[-1]["content"][0]["text"]
    # Work landing resets the streak: the next absorb is the plain nudge.
    state.ok_tool_calls += 1
    fourth = wf._handle_silent_finish("Done.", conv, state, iteration=6)  # pyright: ignore[reportPrivateUsage]
    assert fourth is None
    assert "fruitless" not in conv.to_wire()[-1]["content"][0]["text"]
    assert state.standing_fruitless == 0


def test_standing_patience_bounds_fruitless_reentries() -> None:
    """standing_patience = N absorbs N fruitless rounds, then honours the
    end; 0 restores give-up-on-first-fruitless."""
    curator = MagicMock()
    curator.nodes.return_value = _standing_nodes()
    wf = _wf(mode="run", curator=curator, budget=None)
    wf.config.workflow.standing_patience = 1
    conv = Conversation()
    state = _state(ever_edited=True, verify=VerifyVerdict(ever_passed=True))
    assert wf._handle_silent_finish("Done.", conv, state, iteration=3) is None  # pyright: ignore[reportPrivateUsage]
    assert wf._handle_silent_finish("Done.", conv, state, iteration=4) is None  # pyright: ignore[reportPrivateUsage]
    ended = wf._handle_silent_finish("Done.", conv, state, iteration=5)  # pyright: ignore[reportPrivateUsage]
    assert ended is not None and ended.reason == "silent_finish"

    wf0 = _wf(mode="run", curator=curator, budget=None)
    wf0.config.workflow.standing_patience = 0
    state0 = _state(ever_edited=True, verify=VerifyVerdict(ever_passed=True))
    assert wf0._handle_silent_finish("Done.", Conversation(), state0, iteration=3) is None  # pyright: ignore[reportPrivateUsage]
    ended0 = wf0._handle_silent_finish("Done.", Conversation(), state0, iteration=4)  # pyright: ignore[reportPrivateUsage]
    assert ended0 is not None and ended0.reason == "silent_finish"


def test_standing_task_gates_finish_session_and_soft_stops() -> None:
    curator = MagicMock()
    curator.nodes.return_value = _standing_nodes()
    wf = _wf(mode="run", curator=curator, budget=None)
    state = _state()
    turn = _turn(iteration=2)
    turn.finish_signal = "all done"
    turn.finish_kind = "finish_session"
    wf._gate_standing_finish(state, turn)  # pyright: ignore[reportPrivateUsage]
    assert turn.finish_signal is None  # revoked: the goal continues
    assert any("standing task" in getattr(n, "text", "") for n in turn.tool_results)
    # Soft stop: verify_settled absorbs and clears its streak.
    state.ok_tool_calls += 1
    turn2 = _turn(iteration=3)
    turn2.verify_settled_stop = True
    state.verify_settled_idle = 9
    conv = Conversation()
    wf._absorb_soft_stop(state, turn2, conv)  # pyright: ignore[reportPrivateUsage]
    assert turn2.verify_settled_stop is False
    assert state.verify_settled_idle == 0
    assert "standing task" in conv.to_wire()[-1]["content"][0]["text"]


def test_standing_absorb_refuses_without_a_ready_standing_task() -> None:
    curator = MagicMock()
    curator.nodes.return_value = _typed({"a": {}})  # no standing node
    wf = _wf(mode="run", curator=curator, budget=None)
    assert wf._standing_absorb(_state(), reason="silent_finish", iteration=1) is None  # pyright: ignore[reportPrivateUsage]


def test_standing_goal_seeds_a_standing_child_under_the_root() -> None:
    """`run --standing` reaches the graph: one standing child under the root,
    created as steering (the operator's word, not the worker's)."""
    curator = MagicMock()
    root = _tn("a")
    curator.add_subtask.side_effect = [root, _tn("b", parent_id="a", standing=True)]
    curator.nodes.return_value = _typed({"a": {}})
    provider = MagicMock()
    provider.call.return_value = _resp("done")
    wf = _wf(
        mode="run",
        curator=curator,
        standing_goal="keep hunting bugs",
        budget=None,
        provider=provider,
    )
    wf.run("t")
    drafts = [c.args[0].draft for c in curator.add_subtask.call_args_list]
    assert len(drafts) == 2  # the root, then the standing goal
    assert drafts[1].standing is True
    assert drafts[1].title == "keep hunting bugs"
    assert drafts[1].created_by == "steering"


def test_session_start_carries_the_operators_words_under_a_seed() -> None:
    """`run --from` composes a `<prior-run>` digest ahead of the operator's
    task; the session.start event (every headline's source) carries the
    operator's words, not the digest's opening tag."""
    events: list[dict[str, Any]] = []

    class _Events:
        path = Path("/tmp/x/logs.jsonl")

        def emit(self, event_type: str, /, **fields: Any) -> None:
            events.append({"type": event_type, **fields})

    provider = MagicMock()
    provider.call.return_value = _resp("done")
    wf = _wf(mode="run", budget=None, provider=provider, events=_Events())
    composed = (
        '<prior-run id="agile-echo-H2EWX5">\nThis question is about a PRIOR agent6 run.\n'
        "## Run task\nhow many functions?\n</prior-run>\n\n"
        "add a module docstring to calc.py"
    )
    wf.run(composed)
    start = next(e for e in events if e["type"] == "session.start")
    assert start["user_task"] == "add a module docstring to calc.py"


def test_interactive_quiet_turn_parks_and_a_steer_continues_the_conversation() -> None:
    """G: interactively, going quiet is a TURN BOUNDARY. The run parks (same
    in-memory conversation) and the operator's steer continues it -- no
    resume leg; an "abort" steer ends it as steer_abort."""
    steers = iter(["keep going: also cover sub()"])
    wf = _wf(
        mode="run",
        interactive=True,
        steer_requested=lambda: True,
        steer_prompt=lambda: next(steers),
    )
    conv = Conversation()
    state = _state(ever_edited=True, verify=VerifyVerdict(ever_passed=True))
    parked = wf._handle_silent_finish("Done.", conv, state, iteration=4)  # pyright: ignore[reportPrivateUsage]
    assert parked is None  # steered onward, same conversation
    wire = conv.to_wire()
    assert "keep going: also cover sub()" in wire[-1]["content"][0]["text"]

    aborts = iter(["abort"])
    wf2 = _wf(
        mode="run",
        interactive=True,
        steer_requested=lambda: True,
        steer_prompt=lambda: next(aborts),
    )
    ended = wf2._handle_silent_finish(  # pyright: ignore[reportPrivateUsage]
        "Done.",
        Conversation(),
        _state(ever_edited=True, verify=VerifyVerdict(ever_passed=True)),
        iteration=4,
    )  # pyright: ignore[reportPrivateUsage]
    assert ended is not None and ended.reason == "steer_abort"


def test_non_interactive_quiet_turn_still_ends_and_standing_outranks_the_park() -> None:
    # Non-interactive: unchanged silent_finish end.
    wf = _wf(mode="run", interactive=False)
    ended = wf._handle_silent_finish(  # pyright: ignore[reportPrivateUsage]
        "Done.",
        Conversation(),
        _state(ever_edited=True, verify=VerifyVerdict(ever_passed=True)),
        iteration=4,
    )  # pyright: ignore[reportPrivateUsage]
    assert ended is not None and ended.reason == "silent_finish"
    # A standing goal outranks the park: autonomy first, the absorb nudge (not
    # a park) continues the run.
    curator = MagicMock()
    curator.nodes.return_value = _standing_nodes()
    wf2 = _wf(mode="run", interactive=True, curator=curator, budget=None)
    conv = Conversation()
    out = wf2._handle_silent_finish(  # pyright: ignore[reportPrivateUsage]
        "Done.", conv, _state(ever_edited=True, verify=VerifyVerdict(ever_passed=True)), iteration=4
    )  # pyright: ignore[reportPrivateUsage]
    assert out is None
    assert "standing task" in conv.to_wire()[-1]["content"][0]["text"]


def test_tier2_growth_floor_prevents_zero_growth_refire(tmp_path: Path) -> None:
    """A restart that lands ABOVE the threshold (tiny explicit thresholds, a
    large summary) must not re-summarise every iteration: tier-2 re-fires only
    after the context grew 25% past the last restart's size."""

    class SummariserStub:
        def __init__(self) -> None:
            self.calls = 0

        def call(self, **kwargs: Any) -> ProviderResponse:
            del kwargs
            self.calls += 1
            return _resp("PROGRESS SUMMARY: " + "s" * 4_000)  # bigger than the threshold

    summ = SummariserStub()
    wf = _wf(
        root=tmp_path,
        summariser_provider=summ,
        compact_drop_at_chars=2_000,
        compact_summarise_at_chars=3_000,
    )
    state = _state()
    messages = _big_text_history("TASK: t", blocks=4, block_chars=1_000)
    assert _compact_via_wire(wf, messages, state=state) is True
    assert summ.calls == 1
    # The restarted context already exceeds the threshold (the 4k summary),
    # but with zero growth the next pass must NOT re-summarise.
    assert _compact_via_wire(wf, messages, state=state) is False
    assert summ.calls == 1
    # Real growth past the floor re-arms tier-2.
    for _ in range(6):
        messages.append({"role": "assistant", "content": [{"type": "text", "text": "y" * 1_000}]})
        messages.append({"role": "user", "content": [{"type": "text", "text": "go on"}]})
    assert _compact_via_wire(wf, messages, state=state) is True
    assert summ.calls == 2


def test_auto_commit_with_nothing_changed_emits_no_event(tmp_path: Path) -> None:
    """A green verify with no new edits makes chain_commit return "" (nothing
    changed since the tip); an event or log line for it would claim a commit
    that never happened (a live run printed `auto-commit: ` with a blank
    sha)."""
    events: list[dict[str, Any]] = []
    wf = _wf(root=tmp_path, mode="run", commit_per_step=True)

    def _capture(_type: str, **f: Any) -> None:
        events.append({"type": _type, **f})

    wf.events = MagicMock()
    wf.events.emit = _capture  # type: ignore[method-assign]
    turn = _turn(iteration=3)
    turn.verify_just_passed = True
    turn.edit_since_verify_pass = False
    with patch.object(wf, "_chain_commit", return_value=""):
        wf._turn_auto_commit_and_metric(_state(), turn)  # pyright: ignore[reportPrivateUsage]
    assert [e for e in events if e["type"] == "loop.auto_commit"] == []
    assert turn.committed is False


def test_auto_commit_failure_surface_tells_the_truth(tmp_path: Path) -> None:
    """The failure reporter's two directions: a benign nothing-changed variant
    stays silent (no failure event for a non-failure), and a real GitError
    emits loop.auto_commit.failed carrying the error and the subject. A
    regression in the benign filter would spam failure events on every clean
    green, or hide real failures."""
    from agent6.git_ops import GitError

    events: list[dict[str, Any]] = []
    wf = _wf(root=tmp_path, mode="run", commit_per_step=True)

    def _capture(_type: str, **f: Any) -> None:
        events.append({"type": _type, **f})

    wf.events = MagicMock()
    wf.events.emit = _capture  # type: ignore[method-assign]

    for benign in ("nothing to commit, working tree clean", "no changes added to commit"):
        wf._report_auto_commit_failure(GitError(benign), "s", iteration=1)  # pyright: ignore[reportPrivateUsage]
    assert events == []  # a non-failure never claims to be one

    wf._report_auto_commit_failure(  # pyright: ignore[reportPrivateUsage]
        GitError("fatal: unable to write new index file"), "agent6 iter 2: fix", iteration=2
    )
    (evt,) = [e for e in events if e["type"] == "loop.auto_commit.failed"]
    assert evt["iteration"] == 2
    assert "unable to write" in evt["error"]
    assert evt["commit_subject"] == "agent6 iter 2: fix"


def test_turn_marker_covers_dispatch_and_clears_after_the_snapshot(tmp_path: Path) -> None:
    """The mid-turn-crash marker is on disk WHILE tools dispatch (a crash in
    the dispatch->snapshot window leaves it at the re-run iteration for resume
    to ask about) and gone once the after-tools snapshot advanced (a clean
    turn leaves nothing; a later resume never falsely prompts)."""
    from agent6.workflows._session_state import TURN_IN_FLIGHT_NAME, read_turn_marker

    marker = tmp_path / TURN_IN_FLIGHT_NAME
    seen: list[tuple[int, tuple[str, ...]] | None] = []

    class ProviderStub:
        def __init__(self) -> None:
            self.calls = 0

        def call(self, **kwargs: Any) -> ProviderResponse:
            self.calls += 1
            if self.calls == 1:
                return _tool_resp("run_verify_command")
            return _tool_resp("finish_session", {"summary": "done"}, tool_id="tool-2")

    class DispatcherStub(_StubDispatcher):
        def dispatch(self, name: str, raw_input: dict[str, Any]) -> ToolResult:
            seen.append(read_turn_marker(marker))
            if name == "run_verify_command":
                return ExecResult(
                    returncode=0, stdout="", stderr="", duration_s=0.1, exec_failed=False
                )
            return RawResult({"acknowledged": True, "summary": raw_input.get("summary", "")})

    config = SimpleNamespace(
        git=_GIT_STUB,
        budget=SimpleNamespace(max_usd=10.0, max_tokens_fallback=2_000_000),
        workflow=SimpleNamespace(
            verify_when="never", verify_retries=2, verify_command=("true",), metric=None
        ),
    )
    wf = _wf(
        root=tmp_path,
        config=config,
        provider=ProviderStub(),
        dispatcher=DispatcherStub(),
        max_iterations=3,
        resume_state_path=tmp_path / "loop_state.json",
    )
    messages = [{"role": "user", "content": [{"type": "text", "text": "TASK:\nt"}]}]
    with patch("agent6.workflows.loop.chain_commit", return_value="abc1234567890"):
        result = wf._drive_loop(  # pyright: ignore[reportPrivateUsage]
            system="system",
            conversation=Conversation.from_wire(messages),
            tool_calls=0,
            start_iteration=1,
            root_task_id=None,
            original_task="t",
        )
    assert result.completed is True
    assert seen[0] == (1, ("run_verify_command",))  # live during dispatch
    assert seen[-1] is not None and seen[-1][0] == 2  # second turn's own marker
    assert not marker.exists()  # a clean end leaves nothing


def test_turn_replay_allowed_marker_semantics(tmp_path: Path) -> None:
    """No marker proceeds; a stale one proceeds and clears silently; a
    matching one asks -- decline keeps the marker (resume again to be asked
    again), accept clears it so one crash asks once."""
    from agent6.app.resume import turn_replay_allowed
    from agent6.workflows._session_state import TURN_IN_FLIGHT_NAME, write_turn_marker

    marker = tmp_path / TURN_IN_FLIGHT_NAME
    asked: list[tuple[int, tuple[str, ...]]] = []

    def _no(iteration: int, tools: tuple[str, ...]) -> bool:
        asked.append((iteration, tools))
        return False

    def _yes(iteration: int, tools: tuple[str, ...]) -> bool:
        asked.append((iteration, tools))
        return True

    assert turn_replay_allowed(tmp_path, 5, _no) is True  # no marker, never asks
    write_turn_marker(marker, 3, ("apply_patch",))
    assert turn_replay_allowed(tmp_path, 5, _no) is True  # stale: cleared, no ask
    assert not marker.exists()
    assert asked == []
    write_turn_marker(marker, 5, ("run_command",))
    assert turn_replay_allowed(tmp_path, 5, _no) is False  # matching + decline
    assert marker.exists()  # stays for the next resume to ask again
    assert turn_replay_allowed(tmp_path, 5, _yes) is True  # matching + accept
    assert not marker.exists()  # asks once
    assert asked == [(5, ("run_command",)), (5, ("run_command",))]


def test_steer_exit_ends_steer_exit_and_suppresses_the_follow_up() -> None:
    """The pause menu's /exit stops the run with its own end reason: the
    listing reads "stopped" and `follow_up_on_offer` skips the "next:"
    prompt (stop-then-type-/exit was the only way to leave before)."""
    import json

    from agent6.ui.cli._session_prompt import follow_up_on_offer
    from agent6.viewmodel.listing import status_word

    ev = _EventCapture()
    wf = _wf(
        mode="run",
        events=ev,
        steer_requested=lambda: True,
        steer_prompt=lambda: "exit",
    )
    result = wf._maybe_handle_steer(  # pyright: ignore[reportPrivateUsage]
        Conversation(), 3, _state()
    )
    assert result == "exit"
    out = wf._steer_outcome("exit", 3, _state())  # pyright: ignore[reportPrivateUsage]
    assert out is not None and out.reason == "steer_exit" and out.completed is False
    ends = [e for e in ev.events if e["type"] == "session.end"]
    assert ends and ends[-1]["reason"] == "steer_exit"
    assert status_word(finished=True, all_passed=False, end_reason="steer_exit") == ("stopped", "")

    # The log-derived follow-up gate: steer_exit never re-opens the prompt.
    import tempfile
    from pathlib import Path as _P

    with tempfile.TemporaryDirectory() as td:
        d = _P(td)
        (d / "logs.jsonl").write_text(
            json.dumps({"type": "session.start", "mode": "run", "user_task": "t"})
            + "\n"
            + json.dumps({"type": "session.end", "reason": "steer_exit", "all_passed": False})
            + "\n",
            encoding="utf-8",
        )
        assert follow_up_on_offer(d) is False
        (d / "logs.jsonl").write_text(
            json.dumps({"type": "session.start", "mode": "run", "user_task": "t"})
            + "\n"
            + json.dumps({"type": "session.end", "reason": "steer_abort", "all_passed": False})
            + "\n",
            encoding="utf-8",
        )
        assert follow_up_on_offer(d) is True


def test_an_adopted_gate_that_cannot_run_is_un_adopted(tmp_path: Path) -> None:
    """Shape (b) of the adoption probe: the first adopted-verify run whose
    failure is an unrunnable signature (exit 127, or the adopted `-m` module
    missing) drops the gate again, tells the model, re-pins the manifest
    gateless, and never re-adopts that argv; a configured gate stays red."""
    from agent6.config import Config
    from agent6.workflows.loop import VERIFY_UNADOPTED_NOTICE  # pyright: ignore[reportPrivateUsage]

    argv = ("python3", "-m", "pytest", "-q")
    events: list[dict[str, Any]] = []

    def emit(kind: str, **kw: Any) -> None:
        events.append({"type": kind, **kw})

    dispatcher = MagicMock()
    wf = _wf(
        root=tmp_path,
        config=Config().with_verify_command(argv),
        provider=MagicMock(),
        dispatcher=dispatcher,
        events=MagicMock(emit=emit),
    )
    st = _state()
    st.verify.adopted = argv
    turn = _turn(iteration=4)
    wf._note_verify_result(  # pyright: ignore[reportPrivateUsage]
        st,
        turn,
        ExecResult(
            returncode=1,
            stdout="",
            stderr="/usr/bin/python3: No module named pytest",
            duration_s=0.03,
            exec_failed=False,
        ),
    )
    assert wf.config.workflow.verify_command == ()
    dispatcher.drop_verify_command.assert_called_once()
    assert st.verify.adopted == () and argv in st.verify.unadoptable
    texts = [it.text for it in turn.tool_results if isinstance(it, Notice)]
    assert any(t.startswith(VERIFY_UNADOPTED_NOTICE[:30]) for t in texts)
    assert not st.verify.broken_warned
    assert any(
        e.get("command") == [] and e.get("source") == "unadopted" and e.get("adopted_at") == 4
        for e in events
    )

    # A configured (not adopted) gate with the same failure stays a red
    # verify: the broken nudge fires, nothing is un-adopted.
    wf2 = _wf(
        root=tmp_path,
        config=Config().with_verify_command(argv),
        provider=MagicMock(),
        dispatcher=MagicMock(),
    )
    st2 = _state()
    turn2 = _turn(iteration=2)
    wf2._note_verify_result(  # pyright: ignore[reportPrivateUsage]
        st2,
        turn2,
        ExecResult(
            returncode=1,
            stdout="",
            stderr="No module named pytest",
            duration_s=0.03,
            exec_failed=False,
        ),
    )
    assert wf2.config.workflow.verify_command == argv and st2.verify.broken_warned


def test_operator_answers_become_recorded_rulings(tmp_path: Path) -> None:
    """The decisions file is written by the harness, not the model: an
    ask_user answer lands as a ruling with its question, a steer that answers
    the model's trailing question lands with that question, an ordinary steer
    does not, and the finish-time check finds them all in the file."""
    from agent6.memory import decisions_path
    from agent6.tools.results import AnswersResult
    from agent6.workflows._conversation import Conversation

    state_dir = tmp_path / "state"
    state_dir.mkdir()
    events = MagicMock(path=tmp_path / "sessions" / "runs" / "tidy-fox-1" / "logs.jsonl")
    prompts = iter(["Use the inline item.", "unrelated instruction"])
    wf = _wf(
        root=tmp_path,
        state_dir=state_dir,
        provider=MagicMock(),
        dispatcher=MagicMock(),
        events=events,
        steer_requested=lambda: True,
        steer_prompt=lambda: next(prompts),
        steer_clear=lambda: None,
    )
    st = _state()
    turn = _turn(iteration=1)
    wf._note_tool_effects(  # pyright: ignore[reportPrivateUsage]
        st,
        turn,
        "ask_user",
        AnswersResult(answers=("8931", "no")),
        {"questions": [{"question": "Which port?"}, {"question": "Keep the modal?"}]},
    )
    conv = Conversation()
    conv.notice("task")
    conv.assistant([{"type": "text", "text": "Two shapes fit.\nDrop the modal or keep it?"}])
    assert wf._maybe_handle_steer(conv, 2, st) is None  # pyright: ignore[reportPrivateUsage]
    conv.assistant([{"type": "text", "text": "Done with the item."}])
    assert wf._maybe_handle_steer(conv, 3, st) is None  # pyright: ignore[reportPrivateUsage]
    text = decisions_path(state_dir).read_text(encoding="utf-8")
    assert text.count("[tidy-fox-1]") == 3
    assert "Q: Which port?\n  A: 8931\n" in text and "Q: Keep the modal?\n  A: no\n" in text
    assert "Q: Drop the modal or keep it?\n  A: Use the inline item.\n" in text
    assert "unrelated instruction" not in text
    assert len(st.decisions_recorded) == 3
    # The check reads the file, not the capped injection view: a leg whose
    # rulings outgrow the cap still finds every one of them on disk.
    with decisions_path(state_dir).open("a", encoding="utf-8") as fh:
        fh.write("- 2026-08-23T00:00:00Z [other] Q: pad\n  A: " + "x" * 5000 + "\n")
    wf._check_decisions_recorded(st)  # pyright: ignore[reportPrivateUsage]
    assert not any(
        c.kwargs.get("missing")
        for c in events.emit.call_args_list
        if c.args[:1] == ("loop.decision.unrecorded",)
    )


def test_a_skill_command_steer_expands_in_the_loop(tmp_path: Path) -> None:
    """`/<skill> [args]` from any composer: the loop injects the skill's full
    text as the instruction (the CLI menu passes the line through), so every
    surface means the same thing; a slash word that is no skill stays an
    ordinary steer."""
    from agent6.skills import ResolvedSkills, Skill

    skill = Skill(name="caveman", description="Use when grunting.", dir=tmp_path, text="GRUNT")
    prompts = iter(["/caveman lite", "/nosuch thing"])
    dispatcher = MagicMock()
    dispatcher.resolved_skills.return_value = ResolvedSkills(
        enabled=(skill,), always=(), warnings=()
    )
    wf = _wf(
        root=tmp_path,
        provider=MagicMock(),
        dispatcher=dispatcher,
        steer_requested=lambda: True,
        steer_prompt=lambda: next(prompts),
        steer_clear=lambda: None,
    )
    wf.mode = "run"
    st = _state()
    conv = MagicMock()
    assert wf._maybe_handle_steer(conv, 1, st) is None  # pyright: ignore[reportPrivateUsage]
    injected = conv.notice.call_args.args[0]
    assert "Apply the operator-installed skill 'caveman'" in injected
    assert (
        "Skill arguments: lite" in injected
        and '<skill name="caveman">\nGRUNT\n</skill>' in injected
    )
    assert wf._maybe_handle_steer(conv, 2, st) is None  # pyright: ignore[reportPrivateUsage]
    assert "/nosuch thing" in conv.notice.call_args.args[0]
