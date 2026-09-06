# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Golden pin of the provider wire + persisted messages for a scripted loop run.

The messages list handed to ``provider.call`` is frozen LLM I/O: same
conversation history => byte-identical dicts, key order, and rolling
``cache_control`` placement (the breakpoint roll is measured perf). The same
list persists verbatim inside ``loop_state.json``. This drives one
representative run through a scripted provider and pins, per worker call, the
exact ``json.dumps`` of the messages received AND the raw pre-call
``loop_state.json`` bytes, plus the summariser side-calls and a resume from a
mid-run snapshot.

The scenario covers every history-shaping path: tool_use/tool_result pairs, an
interleaved harness notice inside a results turn (broken-verify), a went-quiet
assistant turn popped from history, operator steering injection, tier-1
elision with a distilled gist, a forced tier-2 summarise-and-restart, the
rolling cache breakpoints across all of it, and resume re-entering the saved
history.

Regenerate only with a deliberate, reviewed wire change:

    uv run python tests/unit/test_loop_wire_golden.py
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from agent6.providers import ProviderResponse
from agent6.tools.mcp_client import MCPToolDescriptor
from agent6.tools.results import ExecResult, RawResult, ToolResult
from agent6.workflows._conversation import Conversation
from agent6.workflows.loop import Workflow

_GOLDEN = Path(__file__).parent / "data" / "golden_loop_wire.json"

_TASK = "Fix the parser bug in a.md"


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

    def metric_configured(self) -> bool:
        return True

    def skills_available(self) -> bool:
        return False

    def command_policy(self) -> str:
        return "ask"

    def tool_is_withheld(self, name: str) -> bool:
        """ "ask" withholds nothing: the operator is asked at call time."""
        return False

    def settle_background(self) -> None:
        """The turn boundary observes background commands; this scenario starts
        none, so there is nothing to write down."""


def _resp(
    *,
    text: str = "",
    thinking: str = "",
    tool_uses: tuple[tuple[str, str, dict[str, Any]], ...] = (),
    stop_reason: str = "end_turn",
) -> ProviderResponse:
    """A provider response whose raw content mirrors what real providers build:
    thinking / text / tool_use blocks, with tool_uses parsed from the same."""
    blocks: list[dict[str, Any]] = []
    if thinking:
        blocks.append({"type": "thinking", "thinking": thinking})
    if text:
        blocks.append({"type": "text", "text": text})
    for tu_id, name, tool_input in tool_uses:
        blocks.append({"type": "tool_use", "id": tu_id, "name": name, "input": tool_input})
    return ProviderResponse(
        text=text,
        tool_uses=tuple({"id": i, "name": n, "input": inp} for i, n, inp in tool_uses),
        stop_reason=stop_reason,
        input_tokens=3,
        output_tokens=7,
        cache_read_tokens=0,
        cache_creation_tokens=0,
        raw={"content": blocks},
    )


class _WorkerScript:
    """Scripted worker provider: captures each call's messages (deep-copied:
    the loop mutates the live history) and the pre-call loop_state bytes."""

    def __init__(self, responses: list[ProviderResponse], snap_path: Path) -> None:
        self._responses = responses
        self._snap_path = snap_path
        self.captured: list[dict[str, str]] = []

    def call(self, **kwargs: Any) -> ProviderResponse:
        self.captured.append(
            {
                "messages": json.dumps(copy.deepcopy(kwargs["messages"]), ensure_ascii=False),
                "loop_state": self._snap_path.read_text(encoding="utf-8"),
            }
        )
        return self._responses[len(self.captured) - 1]


class _SummariserScript:
    """Scripted summariser seat: first call is the gist distiller, second is
    the tier-2 restart summary. Captures both requests (they embed the
    transcript-tail renderer's output, pinning that renderer too)."""

    def __init__(self) -> None:
        self.captured: list[dict[str, str]] = []

    def call(self, **kwargs: Any) -> ProviderResponse:
        self.captured.append(
            {
                "system_head": str(kwargs["system"])[:60],
                "messages": json.dumps(kwargs["messages"], ensure_ascii=False),
            }
        )
        if len(self.captured) == 1:
            return _resp(text="a.md: parser spec; headers under 80 chars; END lines exempt")
        return _resp(text="PROGRESS: read a.md and b.md, verify runner is broken, grep found it.")


class _Dispatcher(_StubDispatcher):
    """Scripted tool results. Serving the grep (iteration 4) arms the manual
    compact request so the next pre-call forces the tier-2 restart."""

    def __init__(self, compact_flag: list[bool]) -> None:
        self._compact_flag = compact_flag

    def set_run_root_node_id(self, node_id: str) -> None:  # pragma: no cover - resume leg
        return None

    def resolved_skills(self) -> Any:  # pragma: no cover - not used by _drive_loop
        return SimpleNamespace(warnings=[], enabled=[], always=[])

    def dispatch(self, name: str, tool_input: dict[str, Any]) -> ToolResult:
        if name == "read_file":
            path = str(tool_input.get("path", ""))
            body = {"a.md": "A" * 4000, "b.md": "B" * 600}[path]
            return RawResult({"content": body, "size": len(body)})
        if name == "run_verify_command":
            return ExecResult(
                returncode=1,
                stdout="",
                stderr="sh: 1: pytest: command not found",
                duration_s=0.05,
                exec_failed=False,
            )
        if name == "list_dir":
            self._compact_flag[0] = True
            return RawResult({"entries": ["b.md"]})
        if name == "finish_session":
            return RawResult({"acknowledged": True})
        raise AssertionError(f"unexpected tool: {name}")


class _SteerOnce:
    """One steering request after the first completed iteration."""

    def __init__(self) -> None:
        self.armed = True

    def requested(self) -> bool:
        return self.armed

    def prompt(self) -> str:
        return "focus on the parser first"

    def clear(self) -> None:
        self.armed = False


def _config() -> Any:
    return SimpleNamespace(
        workflow=SimpleNamespace(
            verify_command=("pytest", "-q"),
            verify_when="never",
            verify_retries=2,
            metric=None,
        ),
        prompt=SimpleNamespace(decompose="off"),
    )


_RESPONSES = [
    # iter 1: prose + a read and a (broken) verify -- paired results with an
    # interleaved [harness] notice inside the same user turn.
    _resp(
        thinking="scan the repo first",
        text="Reading a.md and running verify.",
        tool_uses=(
            ("tu-1a", "read_file", {"path": "a.md"}),
            ("tu-1b", "run_verify_command", {}),
        ),
        stop_reason="tool_use",
    ),
    # iter 2: went quiet (thinking only) -- the empty assistant turn is popped
    # from history and a [harness] nudge is appended instead.
    _resp(thinking="pondering silently"),
    # iter 3: another large read (feeds tier-1 pressure).
    _resp(tool_uses=(("tu-3", "read_file", {"path": "b.md"}),), stop_reason="tool_use"),
    # iter 4: list_dir; serving it arms the manual compact marker.
    _resp(tool_uses=(("tu-4", "list_dir", {"path": "."}),), stop_reason="tool_use"),
    # iter 5 (post tier-2 restart): finish.
    _resp(tool_uses=(("tu-5", "finish_session", {"summary": "done"}),), stop_reason="tool_use"),
]


def _run_scenario(tmp_dir: Path) -> dict[str, Any]:
    snap_path = tmp_dir / "loop_state.json"
    compact_flag = [False]
    worker = _WorkerScript(list(_RESPONSES), snap_path)
    summariser = _SummariserScript()
    steer = _SteerOnce()
    pre_restart_state: list[str] = []

    def _compact_requested() -> str | None:
        if compact_flag[0] and not pre_restart_state:
            # Capture the richest on-disk snapshot (post-tools iteration 4:
            # gist placeholder + interleaved notice + steer + nudge) before
            # the forced restart replaces the history; the resume leg re-enters
            # from these bytes.
            pre_restart_state.append(snap_path.read_text(encoding="utf-8"))
        return "" if compact_flag[0] else None  # a plain /compact carries no focus

    def _compact_clear() -> None:
        compact_flag[0] = False

    wf = Workflow(
        root=tmp_dir,
        config=_config(),
        provider=worker,  # type: ignore[arg-type]
        dispatcher=_Dispatcher(compact_flag),  # type: ignore[arg-type]
        logger=lambda _msg: None,
        summariser_provider=summariser,  # type: ignore[arg-type]
        compact_drop_at_chars=2_000,
        resume_state_path=snap_path,
        steer_requested=steer.requested,
        steer_prompt=steer.prompt,
        steer_clear=steer.clear,
        compact_requested=_compact_requested,
        compact_clear=_compact_clear,
    )
    initial = {"role": "user", "content": [{"type": "text", "text": f"TASK:\n{_TASK}\n\nBegin."}]}
    result = wf._drive_loop(  # pyright: ignore[reportPrivateUsage]
        system="SYSTEM",
        conversation=Conversation.from_wire([initial]),
        tool_calls=0,
        start_iteration=1,
        root_task_id=None,
        original_task=_TASK,
    )
    assert result.completed is True and result.reason == "finish_session"
    assert len(worker.captured) == len(_RESPONSES)
    assert len(summariser.captured) == 2
    assert len(pre_restart_state) == 1

    # Resume leg: re-enter from the richest mid-run snapshot. The pre-call
    # snapshot this resume writes must reproduce the loaded messages exactly
    # (save -> load -> save stability), which the captured loop_state pins.
    resume_snap = tmp_dir / "resume" / "loop_state.json"
    resume_snap.parent.mkdir(parents=True, exist_ok=True)
    resume_snap.write_text(pre_restart_state[0], encoding="utf-8")
    resume_worker = _WorkerScript(
        [
            _resp(
                tool_uses=(("tu-r", "finish_session", {"summary": "done"}),), stop_reason="tool_use"
            )
        ],
        resume_snap,
    )
    wf2 = Workflow(
        root=tmp_dir,
        config=_config(),
        provider=resume_worker,  # type: ignore[arg-type]
        dispatcher=_Dispatcher([False]),  # type: ignore[arg-type]
        logger=lambda _msg: None,
        compact_drop_at_chars=2_000,
        resume_state_path=resume_snap,
    )
    resumed = wf2.resume()
    assert resumed.completed is True and resumed.reason == "finish_session"

    return {
        "worker_calls": worker.captured,
        "summariser_calls": summariser.captured,
        "pre_restart_loop_state": pre_restart_state[0],
        "final_loop_state": snap_path.read_text(encoding="utf-8"),
        "resume_call": resume_worker.captured[0],
    }


def test_loop_wire_matches_golden(tmp_path: Path) -> None:
    got = _run_scenario(tmp_path)
    want = json.loads(_GOLDEN.read_text(encoding="utf-8"))
    # Compare piecewise so a mismatch names the drifted surface, not a wall.
    assert got["summariser_calls"] == want["summariser_calls"]
    for i, (g, w) in enumerate(zip(got["worker_calls"], want["worker_calls"], strict=True)):
        assert g["messages"] == w["messages"], f"worker call {i + 1} messages drifted"
        assert g["loop_state"] == w["loop_state"], f"worker call {i + 1} loop_state drifted"
    assert got["pre_restart_loop_state"] == want["pre_restart_loop_state"]
    assert got["final_loop_state"] == want["final_loop_state"]
    assert got["resume_call"] == want["resume_call"]


def test_scenario_exercises_the_shaping_paths(tmp_path: Path) -> None:
    """Guard the scenario itself: the pin is only as strong as what the run
    actually walked through."""
    got = _run_scenario(tmp_path)
    calls = [json.loads(c["messages"]) for c in got["worker_calls"]]
    # In-turn notice: the broken-verify text rides the results turn AFTER the
    # results (a text block ahead of a tool_result 400s the anthropic wire).
    results_turn = calls[1][2]["content"]
    assert [b["type"] for b in results_turn] == ["tool_result", "tool_result", "text"]
    # Steering injected; the went-quiet turn is thinking-only, so a pop
    # regression would leak a NON-empty assistant turn: assert the popped
    # content is gone and every surviving assistant turn is substantive.
    assert "focus on the parser first" in json.dumps(calls[2])
    assert "pondering silently" not in json.dumps(calls[2])
    assert all(
        any(b["type"] in ("text", "tool_use") for b in m["content"])
        for m in calls[2]
        if m["role"] == "assistant"
    )
    # Tier-1 gist elision landed before call 4.
    assert "distilled" in json.dumps(calls[3])
    # Tier-2 restart: call 5 sees (original task + restart notice) only.
    assert len(calls[4]) == 2
    assert "PROGRESS: read a.md and b.md" in json.dumps(calls[4][1])
    # Rolling cache breakpoints: at most two marks, and call 2 keeps call 1's.
    for msgs in calls:
        marks = [
            b["cache_control"]
            for m in msgs
            if isinstance(m["content"], list)
            for b in m["content"]
            if "cache_control" in b
        ]
        assert len(marks) <= 2
        assert all(mark == {"type": "ephemeral"} for mark in marks)
    call1_marks = json.loads(got["worker_calls"][0]["messages"])[0]["content"][0]
    assert "cache_control" in call1_marks
    assert "cache_control" in json.loads(got["worker_calls"][1]["messages"])[0]["content"][0]


def _regenerate() -> None:
    """Rewrite the golden from the current code. Manual, reviewed only."""
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        got = _run_scenario(Path(td))
    _GOLDEN.write_text(json.dumps(got, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    _regenerate()
    print(f"regenerated {_GOLDEN}")
