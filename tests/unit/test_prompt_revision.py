# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Unit tests for prompt revision."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

from agent6.config import Config, load_config
from agent6.providers import ProviderResponse
from agent6.tools.results import ExecResult, RawResult
from agent6.types import RepoSummary
from agent6.workflows import loop as loopmod
from agent6.workflows.loop import Workflow

_VALID_TOML = """
[agent6]
config_version = 1
[providers.anthropic]
api_format = "anthropic"
api_key_env = "ANTHROPIC_API_KEY"
prompt_caching = true
[models.worker]
provider = "anthropic"
model = "x"
[models.reviewer]
provider = "anthropic"
model = "x"
[sandbox]
isolation = "auto"
run_commands = "no"
protect_git = true
[git]
dirty_tree = "ask"
branch_per_run = true
[workflow]
verify_command = ["true"]
[budget]
max_tokens_fallback = 2000000
"""


def _silent(_msg: str) -> None:
    return None


def _config(tmp_path: Path) -> Config:
    path = tmp_path / "agent6.toml"
    path.write_text(_VALID_TOML, encoding="utf-8")
    return load_config(path)


def _repo(tmp_path: Path) -> RepoSummary:
    return RepoSummary(
        root=tmp_path,
        branch="main",
        head_sha="0" * 40,
        file_count=3,
        top_level=("src", "tests"),
        agents_md="Use ruff and pytest.",
        recent_log="abc123 feat: add thing",
        repo_map="src/  (1 files: foo.py)",
        symbol_outline="src/foo.py:\n  function frob:12",
    )


def _text_resp(text: str) -> ProviderResponse:
    return ProviderResponse(
        text=text,
        tool_uses=(),
        stop_reason="end_turn",
        input_tokens=10,
        output_tokens=20,
        cache_read_tokens=0,
        cache_creation_tokens=0,
        raw={"content": [{"type": "text", "text": text}]},
    )


def _finish_resp(summary: str) -> ProviderResponse:
    tool_use = {
        "type": "tool_use",
        "id": "finish-1",
        "name": "finish_session",
        "input": {"summary": summary},
    }
    return ProviderResponse(
        text="",
        tool_uses=(tool_use,),
        stop_reason="tool_use",
        input_tokens=10,
        output_tokens=20,
        cache_read_tokens=0,
        cache_creation_tokens=0,
        raw={"content": [tool_use]},
    )


def _wf(tmp_path: Path, **kw: Any) -> Workflow:
    dispatcher = kw.pop("dispatcher", MagicMock())
    dispatcher.available_tool_names.return_value = []
    dispatcher.dispatch.return_value = RawResult({"acknowledged": True})
    # The harness certifies the finish (`verify_when = "finish"`): a green gate.
    dispatcher.run_verify.return_value = ExecResult(
        returncode=0, stdout="", stderr="", duration_s=0.1, exec_failed=False
    )
    defaults: dict[str, Any] = {
        "root": tmp_path,
        "config": _config(tmp_path),
        "provider": MagicMock(),
        "dispatcher": dispatcher,
        "logger": _silent,
        "provider_retry_delay_s": 0.01,
    }
    defaults.update(kw)
    return Workflow(**defaults)


def test_parse_prompt_revision_tagged_output() -> None:
    parsed = loopmod.parse_prompt_revision(  # pyright: ignore[reportPrivateUsage]
        "<revised_task>Fix foo and verify with pytest.</revised_task>\n"
        "<clarifying_questions>\n- Which API version?\n- none\n</clarifying_questions>"
    )
    assert parsed.revised_task == "Fix foo and verify with pytest."
    assert parsed.clarifying_questions == ("Which API version?",)


def test_parse_prompt_revision_falls_back_to_plain_text() -> None:
    parsed = loopmod.parse_prompt_revision(  # pyright: ignore[reportPrivateUsage]
        "Fix the failing test with the smallest change."
    )
    assert parsed.revised_task == "Fix the failing test with the smallest change."
    assert parsed.clarifying_questions == ()


def test_parse_prompt_revision_keeps_leading_digits_in_questions() -> None:
    # Only the list marker is stripped. The old charset lstrip("-*0123456789. ")
    # also ate leading digits of the question itself ("- 32-bit support
    # needed?" became "bit support needed?").
    parsed = loopmod.parse_prompt_revision(  # pyright: ignore[reportPrivateUsage]
        "<revised_task>t</revised_task>\n"
        "<clarifying_questions>\n"
        "- 32-bit support needed?\n"
        "1. 64-bit only, or both?\n"
        "2) 100ms latency budget acceptable?\n"
        "</clarifying_questions>"
    )
    assert parsed.clarifying_questions == (
        "32-bit support needed?",
        "64-bit only, or both?",
        "100ms latency budget acceptable?",
    )


def test_parse_prompt_revision_unmarked_question_passes_through() -> None:
    parsed = loopmod.parse_prompt_revision(  # pyright: ignore[reportPrivateUsage]
        "<revised_task>t</revised_task>\n"
        "<clarifying_questions>\n3rd-party deps allowed?\n* none\n</clarifying_questions>"
    )
    # No marker: the line (digits included) is kept verbatim; "none" after a
    # marker is still filtered.
    assert parsed.clarifying_questions == ("3rd-party deps allowed?",)


def test_parse_prompt_revision_keeps_leading_decimal_in_question() -> None:
    # A question that opens with a bare decimal ("0.5s ...") is NOT a numbered
    # list item ("0." has no trailing space); its leading digits must survive.
    parsed = loopmod.parse_prompt_revision(  # pyright: ignore[reportPrivateUsage]
        "<revised_task>t</revised_task>\n"
        "<clarifying_questions>\n0.5s latency budget acceptable?\n</clarifying_questions>"
    )
    assert parsed.clarifying_questions == ("0.5s latency budget acceptable?",)


def test_workflow_auto_revises_task_before_worker_call(tmp_path: Path) -> None:
    worker = MagicMock()
    worker.call.return_value = _finish_resp("done")
    reviser = MagicMock()
    reviser.call.return_value = _text_resp(
        "<revised_task>Fix the bug in src/foo.py and run verify.</revised_task>\n"
        "<clarifying_questions>none</clarifying_questions>"
    )
    wf = _wf(
        tmp_path,
        provider=worker,
        prompt_reviser_provider=reviser,
        revise_prompt="auto",
    )

    with patch.object(Workflow, "_load_repo_summary", return_value=_repo(tmp_path)):
        result = wf.run("fix it")

    assert result.reason == "finish_session"
    assert reviser.call.call_count == 1
    first_worker_messages = worker.call.call_args.kwargs["messages"]
    task_text = first_worker_messages[0]["content"][0]["text"]
    assert "Revised task prompt:" in task_text
    assert "Fix the bug in src/foo.py" in task_text
    assert "Original user task" in task_text
    assert "fix it" in task_text


def test_workflow_prompt_revision_empty_response_fails_before_worker(tmp_path: Path) -> None:
    worker = MagicMock()
    reviser = MagicMock()
    reviser.call.return_value = _text_resp("<revised_task>   </revised_task>")
    wf = _wf(
        tmp_path,
        provider=worker,
        prompt_reviser_provider=reviser,
        revise_prompt="auto",
    )

    with patch.object(Workflow, "_load_repo_summary", return_value=_repo(tmp_path)):
        result = wf.run("fix it")

    assert result.completed is False
    assert result.reason == "prompt_revision_failed"
    assert worker.call.call_count == 0


def test_workflow_interactive_selector_can_use_original(tmp_path: Path) -> None:
    def select_original(original: str, _revised: str, _questions: tuple[str, ...]) -> str:
        return original

    worker = MagicMock()
    worker.call.return_value = _finish_resp("done")
    reviser = MagicMock()
    reviser.call.return_value = _text_resp("<revised_task>Rewrite everything.</revised_task>")
    wf = _wf(
        tmp_path,
        provider=worker,
        prompt_reviser_provider=reviser,
        revise_prompt="interactive",
        prompt_revision_selector=select_original,
    )

    with patch.object(Workflow, "_load_repo_summary", return_value=_repo(tmp_path)):
        result = wf.run("keep this exact task")

    assert result.reason == "finish_session"
    first_worker_messages = worker.call.call_args.kwargs["messages"]
    task_text = first_worker_messages[0]["content"][0]["text"]
    assert "keep this exact task" in task_text
    assert "Rewrite everything" not in task_text
