# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""`agent6 prompt show` assembles the real system prompt for the current repo."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from agent6.ui.cli.prompt_cmds import _cmd_prompt_show  # pyright: ignore[reportPrivateUsage]


def _git_repo(tmp_path: Path) -> Path:
    p = tmp_path / "repo"
    p.mkdir()
    (p / "f.py").write_text("x = 1\n", encoding="utf-8")
    (p / "AGENTS.md").write_text("# conventions\n- be terse here\n", encoding="utf-8")
    env = {
        "GIT_AUTHOR_NAME": "t",
        "GIT_AUTHOR_EMAIL": "t@e",
        "GIT_COMMITTER_NAME": "t",
        "GIT_COMMITTER_EMAIL": "t@e",
        "PATH": os.environ.get("PATH", ""),
    }
    subprocess.run(["git", "init", "-q"], cwd=p, check=True)
    subprocess.run(["git", "add", "-A"], cwd=p, check=True)
    subprocess.run(["git", "commit", "-qm", "init"], cwd=p, env=env, check=True)
    return p


def _isolate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, repo: Path) -> None:
    monkeypatch.chdir(repo)
    # isolate from the developer's real global config / state
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "cfg"))
    monkeypatch.setenv("AGENT6_STATE_HOME", str(tmp_path / "state"))


def test_prompt_show_run_mode_injects_agents_md(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    repo = _git_repo(tmp_path)
    _isolate(tmp_path, monkeypatch, repo)
    rc = _cmd_prompt_show(None, mode="run")
    out = capsys.readouterr().out
    assert rc == 0
    # static structural blocks + the per-repo priors block
    assert "<agent6>" in out and "<repo-priors>" in out
    # the repo's AGENTS.md is injected verbatim into the prompt
    assert "be terse here" in out


def test_prompt_show_plan_mode_differs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    repo = _git_repo(tmp_path)
    _isolate(tmp_path, monkeypatch, repo)
    rc = _cmd_prompt_show(None, mode="plan")
    out = capsys.readouterr().out
    assert rc == 0 and "PLAN mode" in out


def test_prompt_show_includes_recorded_memories(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """`prompt show` claims to print what the worker actually receives, but it
    never passed the memories (or skills) the run loop injects: an operator
    checking whether a recorded memory would reach future runs saw '(none
    recorded yet)' while the real prompt carried it."""
    from agent6.config.layer import resolved_state_dir
    from agent6.memory import add

    repo = _git_repo(tmp_path)
    _isolate(tmp_path, monkeypatch, repo)
    state = resolved_state_dir(repo)
    state.mkdir(parents=True, exist_ok=True)
    add(state, "facts", "the deploy script needs sudo")

    assert _cmd_prompt_show(None, mode="run") == 0
    out = capsys.readouterr().out
    assert "the deploy script needs sudo" in out


def test_prompt_show_prints_the_tools_and_the_first_message(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The system prompt is half of what the model receives: the tool
    definitions travel in the API's `tools` field and the task rides a
    first-message header. `prompt show` printed the system prompt alone, so
    an operator reading it saw no run_command guidance and judged the model
    blind. It prints all three now, and `--json` the same as one object,
    with the tool list this config actually exposes."""
    import json

    repo = _git_repo(tmp_path)
    _isolate(tmp_path, monkeypatch, repo)
    assert _cmd_prompt_show(None, mode="run") == 0
    out = capsys.readouterr().out
    assert "=== tools (" in out and "--- run_command" in out and "--- finish_session" in out
    assert "=== first user message ===" in out and "`finish_session` ends the run" in out

    assert _cmd_prompt_show(None, mode="ask", as_json=True) == 0
    exchange = json.loads(capsys.readouterr().out)
    names = [t["name"] for t in exchange["tools"]]
    assert "read_file" in names and "agent6_docs" in names
    assert "apply_edit" not in names and "finish_session" not in names  # ask has no edit/finish
    assert set(exchange) == {"mode", "system", "tools", "first_message", "mcp_tools_pending"}
    assert all("input_schema" in t and "description" in t for t in exchange["tools"])


def test_prompt_show_infers_the_gate_a_run_would_infer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A run infers its gate before assembling the prompt, and the gate decides
    the `<verify-command>` block, the commit rule and whether
    `run_verify_command` is offered. Skipping it printed "this run has no
    verify command" for every repo whose gate is inferred."""
    repo = _git_repo(tmp_path)
    (repo / "AGENTS.md").write_text(
        "# agents\n\nbe terse here\n\n## Verify command\n\n```bash\npytest -q\n```\n",
        encoding="utf-8",
    )
    _isolate(tmp_path, monkeypatch, repo)

    assert _cmd_prompt_show(None, mode="run") == 0

    out = capsys.readouterr().out
    assert "pytest -q" in out
    assert "no verify command" not in out
    assert "run_verify_command" in out


def test_a_withheld_tool_gets_no_block_and_no_offer(tmp_path: Path) -> None:
    """`run_commands = "no"` withholds every command tool, and a metric with no
    `[workflow.metric]` can only error. A prompt block describing a tool the
    model does not have is one it cannot act on, and the metric tool was
    offered unconditionally while `run_verify_command` was already hidden."""
    import tempfile

    from agent6.config import Config
    from agent6.tools.dispatch import ToolDispatcher
    from agent6.workflows import model_exchange_for
    from agent6.workflows._toolset import tool_definitions  # pyright: ignore[reportPrivateUsage]

    withheld = Config.model_validate(
        {
            "sandbox": {"run_commands": "no"},
            "workflow": {
                "verify_command": ["pytest", "-q"],
                "metric": {"command": ["m"], "pattern": r"x:(\d+)", "goal": "minimize"},
            },
        }
    )
    exchange = model_exchange_for(withheld, tmp_path, "run", state_dir=tmp_path)
    assert "<verify-command>" not in exchange.system
    assert "<metric-command>" not in exchange.system
    assert "<no-verify-command>" in exchange.system

    with tempfile.TemporaryDirectory() as td:
        plain = ToolDispatcher(root=Path(td), config=Config())
        names = [t.name for t in tool_definitions(plain, mode="run")]
    assert "run_metric_command" not in names, "offered with no [workflow.metric]"
