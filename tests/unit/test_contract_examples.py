# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""`[prompt].contract_examples`: one worker call before the first turn appends
the derived input -> output examples to the task; off leaves it untouched and
a failed call leaves it as it was."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

from agent6.config import Config, load_config
from agent6.providers import ProviderResponse
from agent6.providers.types import ProviderError
from agent6.tools.results import RawResult
from agent6.workflows.loop import Workflow

_TOML = """
[agent6]
config_version = 1
[providers.anthropic]
api_format = "anthropic"
api_key_env = "ANTHROPIC_API_KEY"
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
require_clean_worktree = true
auto_stash = false
branch_per_run = true
[workflow]
verify_command = ["true"]
"""


def _config(tmp_path: Path) -> Config:
    path = tmp_path / "agent6.toml"
    path.write_text(_TOML, encoding="utf-8")
    return load_config(path)


def _text(text: str) -> ProviderResponse:
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


def _wf(tmp_path: Path, provider: Any, **kw: Any) -> Workflow:
    dispatcher = MagicMock()
    dispatcher.available_tool_names.return_value = []
    dispatcher.dispatch.return_value = RawResult({"acknowledged": True})
    return Workflow(
        root=tmp_path,
        config=_config(tmp_path),
        provider=provider,
        dispatcher=dispatcher,
        logger=lambda _m: None,
        **kw,
    )


def test_the_examples_are_appended_as_a_contract_block(tmp_path: Path) -> None:
    provider = MagicMock()
    provider.call.return_value = _text("input: f(1) -> expected: 2\ninput: f(0) -> expected: 1")
    wf = _wf(tmp_path, provider, contract_examples=True)
    out = wf._maybe_derive_contract("Fix f")  # pyright: ignore[reportPrivateUsage]
    assert out.startswith("Fix f\n\n<contract>\n") and out.endswith("\n</contract>")
    assert "input: f(1) -> expected: 2" in out
    assert provider.call.call_args.kwargs["tools"] == []
    assert provider.call.call_args.kwargs["messages"][0]["content"] == "TASK:\nFix f"


def test_off_and_a_failed_call_leave_the_task_as_it_was(tmp_path: Path) -> None:
    provider = MagicMock()
    provider.call.return_value = _text("input: x -> expected: y")
    assert _wf(tmp_path, provider)._maybe_derive_contract("Fix f") == "Fix f"  # pyright: ignore[reportPrivateUsage]
    provider.call.assert_not_called()
    provider.call.side_effect = ProviderError("down")
    wf = _wf(tmp_path, provider, contract_examples=True)
    assert wf._maybe_derive_contract("Fix f") == "Fix f"  # pyright: ignore[reportPrivateUsage]
    provider.call.side_effect = None
    provider.call.return_value = _text("   ")
    assert wf._maybe_derive_contract("Fix f") == "Fix f"  # pyright: ignore[reportPrivateUsage]
