# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""`agent6 machine create`: authoring prompts and the CLI flow."""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Iterable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from agent6.app import machine_agent
from agent6.app.machine import create as _create
from agent6.config import Config
from agent6.config.layer import resolved_state_dir
from agent6.machine import (
    AgentExecResult,
    AgentRequest,
    build_authoring_prompt,
)
from agent6.ui.cli import main

VALID_MACHINE = """\
machine = "greeter"
version = 1
initial = "say"

[budget]
max_usd = 1.0
max_transitions = 10

[states.say]
kind = "tool"
command = ["echo", "hi"]
timeout_secs = 5
on = { ok = "done", nonzero = "fail", timeout = "fail" }

[states.done]
kind = "terminal"
status = "ok"
reason = "greeted"

[states.fail]
kind = "terminal"
status = "failed"
reason = "echo failed"
"""

INVALID_MACHINE = """\
machine = "greeter"
version = 1
initial = "nowhere"

[budget]
max_usd = 1.0
max_transitions = 10

[states.done]
kind = "terminal"
status = "ok"
reason = "x"
"""


# --- pure pieces -----------------------------------------------------------


def test_build_authoring_prompt_first_attempt() -> None:
    prompt = build_authoring_prompt("Poll a queue", attempt=1)
    assert "authoring guide" in prompt
    assert "Poll a queue" in prompt
    assert "finish_session" in prompt
    assert "apply_edit" in prompt, "the draft is written into the workspace"
    assert "fix the draft in this workspace" not in prompt


def test_authoring_guide_describes_the_metered_budget() -> None:
    # One budget story for every draft: max_usd caps metered spend; unpriced
    # models fall to the operator's max_tokens_fallback. No per-draft steering.
    prompt = build_authoring_prompt("Poll a queue", attempt=1)
    assert "max_usd" in prompt and "max_tokens_fallback" in prompt
    assert "best_effort_usd_limit" not in prompt


def test_build_authoring_prompt_retry_carries_only_the_diagnostics() -> None:
    """The draft is in the workspace, so a retry names the problems and sends
    the agent back to its own files instead of re-pasting them into the prompt."""
    prompt = build_authoring_prompt(
        "Poll a queue",
        attempt=2,
        diagnostics=["initial 'nowhere' names no state"],
    )
    assert "Attempt 2: fix the draft in this workspace" in prompt
    assert "initial 'nowhere' names no state" in prompt
    assert "Read the files you wrote" in prompt


# --- CLI flow --------------------------------------------------------------


def _stub_preflight(monkeypatch: pytest.MonkeyPatch) -> None:
    def _load(_root: object, _explicit: object = None) -> object:
        # A REAL Config: the authoring leg materializes it as its overlay
        # (`cfg.model_dump`), so a stand-in namespace would dodge that path.
        cfg = Config.model_validate(
            {
                "sandbox": {"isolation": "none"},
                "budget": {"max_usd": 10.0},
                "providers": {"openrouter": {"api_format": "openai", "api_key_env": "OR_KEY"}},
                "models": {"worker": {"provider": "openrouter", "model": "test-model"}},
            }
        )
        return SimpleNamespace(config=cfg, explicit_leaves=frozenset())

    def _keys_ok(_cfg: object) -> str | None:
        return None

    def _no_preflight(_cfg: object, **_kw: object) -> str:
        # The isolation preflight is the run lifecycle's (select_isolation),
        # tested there; a stand-in config has none of what it reads.
        return "none"

    monkeypatch.setattr(_create, "load_effective", _load)
    monkeypatch.setattr(_create, "check_provider_keys", _keys_ok)
    monkeypatch.setattr(_create, "select_isolation", _no_preflight)


def _stub_runner(
    monkeypatch: pytest.MonkeyPatch,
    drafts: Iterable[tuple[dict[str, str], AgentExecResult]],
) -> None:
    """Each attempt writes its files into the drafting workspace, as the real
    authoring leg does with `apply_edit`, then returns its result."""
    seq = iter(drafts)

    def fake_build(
        cfg: object, root: Path, isolation: object, transcript_dir: Path, **_kw: object
    ) -> Callable[[AgentRequest], AgentExecResult]:
        def run(_request: AgentRequest, _events_log: object = None) -> AgentExecResult:
            files, result = next(seq)
            for rel, content in files.items():
                target = root / rel
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(content, encoding="utf-8")
            return result

        return run

    monkeypatch.setattr(_create, "build_machine_agent_runner", fake_build)


def _write_draft(root: Path, files: dict[str, str]) -> None:
    """Lay a draft down in the workspace, as the authoring leg's edits do."""
    for rel, content in files.items():
        target = root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")


def _draft(machine: str, **scripts: str) -> dict[str, str]:
    """A bundle as the agent would leave it in the workspace."""
    return {"greeter.asm.toml": machine, **{f"scripts/{n}": s for n, s in scripts.items()}}


def test_create_inherits_worker_model(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The authoring agent must INHERIT the worker model (model=None), not get
    an empty-string override. `model=""` overwrote the worker model with "" and
    failed min_length validation, making every `machine create` attempt error
    out -- a path the request-ignoring stub runner never exercised."""
    monkeypatch.chdir(tmp_path)
    _stub_preflight(monkeypatch)
    captured: list[AgentRequest] = []

    def fake_build(
        cfg: object, root: Path, isolation: object, transcript_dir: Path, **_kw: object
    ) -> Callable[[AgentRequest], AgentExecResult]:
        def run(request: AgentRequest, _events_log: object = None) -> AgentExecResult:
            captured.append(request)
            _write_draft(root, _draft(VALID_MACHINE))
            return AgentExecResult(payload=None, reason="finish_session", usd=0.0)

        return run

    monkeypatch.setattr(_create, "build_machine_agent_runner", fake_build)
    code = main(["machine", "create", "Greet the user"])
    assert code == 0
    assert captured, "runner was never invoked"
    assert captured[0].model is None  # inherit, not "" (which would fail to validate)
    # mode="machine" -> authoring system prompt + read-only tools. If the
    # plumbing dropped it the authoring agent would silently fall back to the
    # 29k coding prompt with no test catching it.
    assert captured[0].mode == "run"


def test_create_writes_default_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    _stub_preflight(monkeypatch)
    _stub_runner(
        monkeypatch,
        [(_draft(VALID_MACHINE), AgentExecResult(payload=None, reason="finish_session", usd=0.02))],
    )
    code = main(["machine", "create", "Greet the user"])
    assert code == 0
    out = capsys.readouterr()
    written = tmp_path / "greeter.asm.toml"
    assert written.exists()
    assert written.read_text(encoding="utf-8").startswith('machine = "greeter"')
    assert "wrote draft" in out.err
    assert "spent ~$0.0200" in out.err


def test_create_writes_watchable_event_log(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """machine create writes a logs.jsonl in the draft dir (session.start carrying the
    NL task + session.end) and points the agent runner at that same path, so the TUI
    can open the dashboard on the draft and follow the authoring live, like a run."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("AGENT6_STATE_HOME", str(tmp_path / "state"))
    _stub_preflight(monkeypatch)
    captured_log: list[object] = []

    def fake_build(
        cfg: object, root: Path, isolation: object, transcript_dir: Path, **kw: object
    ) -> Callable[[AgentRequest, object], AgentExecResult]:
        def run(_request: AgentRequest, events_log: object = None) -> AgentExecResult:
            captured_log.append(events_log)  # events_log is now per CALL, not per build
            _write_draft(root, _draft(VALID_MACHINE))
            return AgentExecResult(payload=None, reason="finish_session", usd=0.0)

        return run

    monkeypatch.setattr(_create, "build_machine_agent_runner", fake_build)
    assert main(["machine", "create", "Greet the user"]) == 0

    logs = list((tmp_path / "state").glob("**/sessions/machines/*/logs.jsonl"))
    assert len(logs) == 1
    # The runner was pointed at that same log (so the subprocess appends to it).
    assert captured_log and str(captured_log[0]) == str(logs[0])
    events = [json.loads(line) for line in logs[0].read_text(encoding="utf-8").splitlines()]
    assert events[0]["type"] == "session.start"
    assert events[0]["user_task"] == "Greet the user"  # the dashboard header
    end = next(e for e in events if e["type"] == "session.end")
    # session.end carries the one shape every emitter agrees on: reason + iterations
    # (authoring attempts) + all_passed. One attempt succeeded here.
    assert {"reason", "iterations", "all_passed"} <= end.keys()
    assert end["iterations"] == 1
    assert end["all_passed"] is True


def test_create_logs_the_cumulative_spend_across_attempts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Each attempt's subprocess logs its OWN reset budget.update, so the fold's
    last one showed only the last attempt. create emits the true cumulative total
    at the end, so the watchable draft's cost is the real spend, not the last try."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("AGENT6_STATE_HOME", str(tmp_path / "state"))
    _stub_preflight(monkeypatch)
    _stub_runner(
        monkeypatch,
        [
            # First attempt returns no draft (fails), second succeeds.
            (
                {},
                AgentExecResult(
                    payload=None,
                    reason="finish_session",
                    usd=0.01,
                    input_tokens=100,
                    output_tokens=20,
                ),
            ),
            (
                _draft(VALID_MACHINE),
                AgentExecResult(
                    payload=None,
                    reason="finish_session",
                    usd=0.02,
                    input_tokens=150,
                    output_tokens=30,
                ),
            ),
        ],
    )
    assert main(["machine", "create", "Greet the user", "--max-attempts", "2"]) == 0
    logs = next((tmp_path / "state").glob("**/sessions/machines/*/logs.jsonl"))
    events = [json.loads(line) for line in logs.read_text(encoding="utf-8").splitlines()]
    budgets = [e for e in events if e["type"] == "budget.update"]
    assert budgets, "no cumulative budget.update emitted"
    last = budgets[-1]
    assert abs(last["usd_total"] - 0.03) < 1e-9  # 0.01 + 0.02, not the last 0.02
    assert last["input_total"] == 250 and last["output_total"] == 50


def test_create_saves_the_prompt(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The natural-language task is saved to the draft dir as prompt.txt, so the
    draft is self-describing (otherwise the task only survives embedded inside the
    authoring transcript)."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("AGENT6_STATE_HOME", str(tmp_path / "state"))
    _stub_preflight(monkeypatch)
    _stub_runner(
        monkeypatch,
        [(_draft(VALID_MACHINE), AgentExecResult(payload=None, reason="finish_session", usd=0.0))],
    )
    code = main(["machine", "create", "Greet the user warmly"])
    assert code == 0
    prompts = list((tmp_path / "state").glob("**/sessions/machines/*/prompt.txt"))
    assert len(prompts) == 1
    assert prompts[0].read_text(encoding="utf-8") == "Greet the user warmly"


def test_create_retries_then_succeeds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    _stub_preflight(monkeypatch)
    _stub_runner(
        monkeypatch,
        [
            (
                _draft(INVALID_MACHINE),
                AgentExecResult(payload=None, reason="finish_session", usd=0.01),
            ),
            (
                _draft(VALID_MACHINE),
                AgentExecResult(payload=None, reason="finish_session", usd=0.03),
            ),
        ],
    )
    code = main(["machine", "create", "Greet the user"])
    assert code == 0
    out = capsys.readouterr()
    assert (tmp_path / "greeter.asm.toml").exists()
    # both attempts' spend summed
    assert "spent ~$0.0400" in out.err
    assert "attempt 2/3" in out.err


def test_create_refuses_to_overwrite_default_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    existing = tmp_path / "greeter.asm.toml"
    existing.write_text("# do not clobber\n", encoding="utf-8")
    _stub_preflight(monkeypatch)
    _stub_runner(
        monkeypatch,
        [(_draft(VALID_MACHINE), AgentExecResult(payload=None, reason="finish_session", usd=0.0))],
    )
    code = main(["machine", "create", "Greet the user"])
    assert code == 2  # a refusal
    out = capsys.readouterr()
    # untouched
    assert existing.read_text(encoding="utf-8") == "# do not clobber\n"
    assert "REFUSING to overwrite" in out.err
    # validated draft dumped to stdout
    assert out.out.startswith('machine = "greeter"')


def test_create_collision_refusal_ends_the_watchable_log_as_failed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The collision refusal exits 1 and writes nothing, but session.end had
    already said machine_created / all_passed=true -- a failed create rendered
    as done on every watch surface. The refusal ends the log as its own
    failure token instead."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("AGENT6_STATE_HOME", str(tmp_path / "state"))
    (tmp_path / "greeter.asm.toml").write_text("# do not clobber\n", encoding="utf-8")
    _stub_preflight(monkeypatch)
    _stub_runner(
        monkeypatch,
        [(_draft(VALID_MACHINE), AgentExecResult(payload=None, reason="finish_session", usd=0.0))],
    )
    assert main(["machine", "create", "Greet the user"]) == 2  # a refusal
    capsys.readouterr()
    logs = list((tmp_path / "state").glob("**/sessions/machines/*/logs.jsonl"))
    assert len(logs) == 1
    events = [json.loads(line) for line in logs[0].read_text(encoding="utf-8").splitlines()]
    end = next(e for e in events if e["type"] == "session.end")
    assert end["all_passed"] is False
    assert end["reason"] == "output_collision"


def test_create_write_failure_ends_the_watchable_log_as_failed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """machine_created was emitted before the bundle writes, so a write that
    fails (read-only target dir) raised out of the CLI with the log already
    claiming success. The write failure ends the log as its own failure token,
    keeps the paid-for draft on stdout, and exits 1."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("AGENT6_STATE_HOME", str(tmp_path / "state"))
    _stub_preflight(monkeypatch)
    _stub_runner(
        monkeypatch,
        [(_draft(VALID_MACHINE), AgentExecResult(payload=None, reason="finish_session", usd=0.0))],
    )

    real_write = _create._write_scripts  # pyright: ignore[reportPrivateUsage]

    def denied(base_dir: Path, scripts: dict[str, str]) -> None:
        if str(base_dir).startswith(str(tmp_path / "state")):
            real_write(base_dir, scripts)  # scratch-validation writes proceed
            return
        raise PermissionError("scripts/: Permission denied")  # the output dir only

    monkeypatch.setattr(_create, "_write_scripts", denied)
    assert main(["machine", "create", "Greet the user"]) == 1
    out = capsys.readouterr()
    assert "could not write" in out.err
    assert out.out.startswith('machine = "greeter"')  # the draft is not lost
    logs = list((tmp_path / "state").glob("**/sessions/machines/*/logs.jsonl"))
    assert len(logs) == 1
    events = [json.loads(line) for line in logs[0].read_text(encoding="utf-8").splitlines()]
    end = next(e for e in events if e["type"] == "session.end")
    assert end["all_passed"] is False
    assert end["reason"] == "write_failed"


def test_create_output_flag_overwrites(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    target = tmp_path / "custom.asm.toml"
    target.write_text("# old\n", encoding="utf-8")
    _stub_preflight(monkeypatch)
    _stub_runner(
        monkeypatch,
        [(_draft(VALID_MACHINE), AgentExecResult(payload=None, reason="finish_session", usd=0.0))],
    )
    code = main(["machine", "create", "Greet the user", "-o", str(target)])
    assert code == 0
    assert target.read_text(encoding="utf-8").startswith('machine = "greeter"')


def test_create_never_valid_exits_1(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    _stub_preflight(monkeypatch)
    _stub_runner(
        monkeypatch,
        [
            (
                _draft(INVALID_MACHINE),
                AgentExecResult(payload=None, reason="finish_session", usd=0.01),
            ),
            (
                _draft(INVALID_MACHINE),
                AgentExecResult(payload=None, reason="finish_session", usd=0.01),
            ),
        ],
    )
    code = main(["machine", "create", "Greet the user", "--max-attempts", "2"])
    assert code == 1
    out = capsys.readouterr()
    assert "no valid machine after 2 attempt(s)" in out.err
    assert "Last diagnostics:" in out.err
    # the failed draft stays whole in the workspace, which the failure names
    assert "The last draft is in " in out.err
    assert not (tmp_path / "greeter.asm.toml").exists()


def test_create_surfaces_a_reason_per_failed_attempt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # Each failed attempt logs a one-line reason, not a bare "attempt N/M" with
    # the only diagnostics buried at the very end.
    monkeypatch.chdir(tmp_path)
    _stub_preflight(monkeypatch)
    _stub_runner(
        monkeypatch,
        [
            (
                {},
                AgentExecResult(payload=None, reason="max_iterations", usd=0.0),
            ),  # returned no draft
            (  # a structurally invalid draft
                _draft(INVALID_MACHINE),
                AgentExecResult(payload=None, reason="finish_session", usd=0.01),
            ),
        ],
    )
    code = main(["machine", "create", "Greet the user", "--max-attempts", "2"])
    assert code == 1
    err = capsys.readouterr().err
    assert "attempt 1 failed: No .asm.toml in the workspace" in err
    assert "attempt 2 failed:" in err  # a concrete reason, not silence


def test_attempt_reason_pulls_the_error_from_an_introducing_block() -> None:
    # 'offline test x failed (exit 1):' alone explains nothing; the block's
    # last line carries the actual error (a traceback or test dump ends on it).
    from agent6.app.machine.create import _attempt_reason  # pyright: ignore[reportPrivateUsage]

    block = "offline test scripts/t.py failed (exit 1):\nusage: run.py <pkg>\nAssertionError"
    assert _attempt_reason([block]) == "offline test scripts/t.py failed (exit 1): AssertionError"
    assert _attempt_reason(["plain reason", "extra"]) == "plain reason (+1 more)"


def test_create_no_payload_gives_diagnostic_and_retries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    _stub_preflight(monkeypatch)
    _stub_runner(
        monkeypatch,
        [
            ({}, AgentExecResult(payload=None, reason="max_iterations", usd=0.0)),
            (
                _draft(VALID_MACHINE),
                AgentExecResult(payload=None, reason="finish_session", usd=0.01),
            ),
        ],
    )
    code = main(["machine", "create", "Greet the user"])
    assert code == 0
    assert (tmp_path / "greeter.asm.toml").exists()


def test_create_rejects_bad_max_attempts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    code = main(["machine", "create", "Greet the user", "--max-attempts", "0"])
    assert code == 2
    assert "--max-attempts must be >= 1" in capsys.readouterr().err


def test_create_output_flag_creates_parent_dirs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # -o into a directory that does not exist yet must not crash.
    monkeypatch.chdir(tmp_path)
    _stub_preflight(monkeypatch)
    _stub_runner(
        monkeypatch,
        [(_draft(VALID_MACHINE), AgentExecResult(payload=None, reason="finish_session", usd=0.0))],
    )
    target = tmp_path / "new" / "deep" / "m.asm.toml"
    code = main(["machine", "create", "Greet the user", "-o", str(target)])
    assert code == 0
    assert target.read_text(encoding="utf-8").startswith('machine = "greeter"')


def test_create_publishes_the_bytes_the_lint_gate_passed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The gate lints with `--fix` on a scratch copy, so the source that PASSED
    is ruff's repaired one. Publishing the model's original bytes wrote a bundle
    that failed the `agent6 machine check` this command points the operator at
    (observed live: 4 fixable errors in a freshly created bundle)."""
    from agent6.app.machine import _scriptcheck as scriptcheck

    if "ruff" not in scriptcheck.available_tools():
        pytest.skip("ruff not installed")
    monkeypatch.chdir(tmp_path)
    _stub_preflight(monkeypatch)
    unused_import = "import json\nprint('hi')\n"  # F401: ruff fixes it safely

    def fake_build(
        cfg: object, root: Path, isolation: object, transcript_dir: Path, **_kw: object
    ) -> Callable[[AgentRequest], AgentExecResult]:
        def run(_request: AgentRequest, _events_log: object = None) -> AgentExecResult:
            _write_draft(
                root, {"greeter.asm.toml": SCRIPT_MACHINE, "scripts/run.py": unused_import}
            )
            return AgentExecResult(payload=None, reason="finish_session", usd=0.01)

        return run

    monkeypatch.setattr(_create, "build_machine_agent_runner", fake_build)
    assert main(["machine", "create", "Run a script"]) == 0
    written = (tmp_path / "scripts" / "run.py").read_text(encoding="utf-8")
    assert "import json" not in written, written
    assert scriptcheck.lint_and_typecheck(tmp_path / "scripts") == []


def test_create_retry_prompt_names_the_script_problem(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A draft that fails the script gate gets the problems in the next
    # attempt's prompt; the source itself stays in the workspace, where the
    # agent reads and patches its own file.
    from agent6.app.machine import _scriptcheck as scriptcheck

    if "ruff" not in scriptcheck.available_tools():
        pytest.skip("ruff not installed")
    monkeypatch.chdir(tmp_path)
    _stub_preflight(monkeypatch)
    prompts: list[str] = []
    bad = "print(undefined_name)\n"  # F821
    good = "import json\nprint(json.dumps({}))\n"
    responses = iter(
        [
            (
                {"greeter.asm.toml": SCRIPT_MACHINE, "scripts/run.py": bad},
                AgentExecResult(payload=None, reason="finish_session", usd=0.01),
            ),
            (
                {"greeter.asm.toml": SCRIPT_MACHINE, "scripts/run.py": good},
                AgentExecResult(payload=None, reason="finish_session", usd=0.01),
            ),
        ]
    )

    def fake_build(
        cfg: object, root: Path, isolation: object, transcript_dir: Path, **_kw: object
    ) -> Callable[[AgentRequest], AgentExecResult]:
        def run(request: AgentRequest, _events_log: object = None) -> AgentExecResult:
            prompts.append(request.prompt)
            files, result = next(responses)
            _write_draft(root, files)
            return result

        return run

    monkeypatch.setattr(_create, "build_machine_agent_runner", fake_build)
    code = main(["machine", "create", "Run a script"])
    assert code == 0
    assert len(prompts) == 2
    assert "undefined_name" not in prompts[0]
    assert "scripts/run.py" in prompts[1], "the failing file is named"
    assert "F821" in prompts[1], "and the problem with it"
    assert "Read the files you wrote" in prompts[1]


# --- script bundle: the agent emits helper scripts alongside the .asm.toml ---

SCRIPT_MACHINE = """\
machine = "scripted"
version = 1
initial = "go"

[budget]
max_usd = 1.0
max_transitions = 10

[states.go]
kind = "tool"
command = ["python3", "scripts/run.py"]
timeout_secs = 5
on = { ok = "done", nonzero = "fail", timeout = "fail" }

[states.done]
kind = "terminal"
status = "ok"
reason = "ran"

[states.fail]
kind = "terminal"
status = "failed"
reason = "failed"
"""
SCRIPT_BODY = "import json\nprint(json.dumps({}))"


def test_create_attempts_share_one_budget_ledger(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Each attempt's subprocess is otherwise a fresh budget tracker, so N
    retries could bill N full budgets. Attempts share one ledger: every
    request carries the REMAINING cap, and a spent-out create stops instead
    of paying for another attempt."""
    monkeypatch.chdir(tmp_path)
    _stub_preflight(monkeypatch)
    caps: list[float | None] = []

    def fake_build(
        cfg: object, root: Path, isolation: object, transcript_dir: Path, **_kw: object
    ) -> Callable[[AgentRequest], AgentExecResult]:
        def run(request: AgentRequest, _events_log: object = None) -> AgentExecResult:
            caps.append(request.max_usd)
            return AgentExecResult(payload=None, reason="finish_session", usd=6.0)

        return run

    monkeypatch.setattr(_create, "build_machine_agent_runner", fake_build)
    code = main(["machine", "create", "Run a script"])
    assert code == 1  # no valid draft, and no third full-budget attempt
    assert caps == [10.0, 4.0]
    assert "exhausted" in capsys.readouterr().err


def test_create_writes_script_bundle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    _stub_preflight(monkeypatch)
    _stub_runner(
        monkeypatch,
        [
            (
                {"greeter.asm.toml": SCRIPT_MACHINE, "scripts/run.py": SCRIPT_BODY},
                AgentExecResult(payload=None, reason="finish_session", usd=0.02),
            )
        ],
    )
    code = main(["machine", "create", "Run a script"])
    assert code == 0
    out = capsys.readouterr()
    assert (tmp_path / "scripted.asm.toml").exists()
    script = tmp_path / "scripts" / "run.py"
    assert script.exists()
    # The published bytes are the gate's (ruff --fix ran on the copy that
    # passed), so this compares content rather than the model's exact source.
    written = script.read_text(encoding="utf-8")
    assert written.startswith("import json") and "print(json.dumps({}))" in written
    assert "1 script(s)" in out.err


def test_create_refuses_to_overwrite_existing_script(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The default (no -o) path is documented as clobbering NOTHING; that must
    cover the whole bundle. An operator's pre-existing scripts/run.py whose
    name collides with an LLM-chosen bundle script was silently replaced
    (unrecoverable if uncommitted) while the sibling .asm.toml got a refusal."""
    monkeypatch.chdir(tmp_path)
    sentinel = "# operator-authored, do not clobber\n"
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "run.py").write_text(sentinel, encoding="utf-8")
    _stub_preflight(monkeypatch)
    _stub_runner(
        monkeypatch,
        [
            (
                {"greeter.asm.toml": SCRIPT_MACHINE, "scripts/run.py": SCRIPT_BODY},
                AgentExecResult(payload=None, reason="finish_session", usd=0.02),
            )
        ],
    )
    code = main(["machine", "create", "Run a script"])
    assert code == 2  # a refusal
    out = capsys.readouterr()
    assert "REFUSING to overwrite" in out.err
    assert "scripts/run.py" in out.err  # the clashing path is named
    assert (tmp_path / "scripts" / "run.py").read_text(encoding="utf-8") == sentinel
    assert not (tmp_path / "scripted.asm.toml").exists()  # no half-written bundle


def test_create_rejects_missing_script_then_succeeds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A TOML that runs scripts/run.py but ships no scripts must fail bundle
    validation (the user's bug), then succeed once the agent supplies it."""
    monkeypatch.chdir(tmp_path)
    _stub_preflight(monkeypatch)
    _stub_runner(
        monkeypatch,
        [
            # attempt 1: references the script but omits it -> rejected.
            (
                {"greeter.asm.toml": SCRIPT_MACHINE},
                AgentExecResult(payload=None, reason="finish_session", usd=0.01),
            ),
            # attempt 2: now ships it -> accepted.
            (
                {"greeter.asm.toml": SCRIPT_MACHINE, "scripts/run.py": SCRIPT_BODY},
                AgentExecResult(payload=None, reason="finish_session", usd=0.02),
            ),
        ],
    )
    code = main(["machine", "create", "Run a script"])
    assert code == 0
    out = capsys.readouterr()
    assert (tmp_path / "scripts" / "run.py").exists()
    assert "attempt 2/3" in out.err


def test_create_rejects_lint_bad_script(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A structurally-valid machine whose script has a lint error must NOT be
    written — ruff/ty run in the create loop and the failure is a diagnostic."""
    from agent6.app.machine import _scriptcheck as scriptcheck

    if "ruff" not in scriptcheck.available_tools():
        pytest.skip("ruff not installed")
    monkeypatch.chdir(tmp_path)
    _stub_preflight(monkeypatch)
    bad = "import json\nprint(undefined_name)\n"  # F821 undefined name
    _stub_runner(
        monkeypatch,
        [
            (
                {"greeter.asm.toml": SCRIPT_MACHINE, "scripts/run.py": bad},
                AgentExecResult(payload=None, reason="finish_session", usd=0.01),
            )
        ],
    )
    code = main(["machine", "create", "Run a script", "--max-attempts", "1"])
    assert code == 1
    out = capsys.readouterr()
    assert "ruff" in out.err
    assert not (tmp_path / "scripted.asm.toml").exists()


def test_create_publish_validates_the_destination_before_claiming_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The scratch validation ran on a clean copy; the destination can differ
    (a pre-existing escaping symlink under scripts/). rc 0 with machine_created
    plus a "won't run yet" warning was a success banner over a broken bundle;
    a published bundle that fails validation is now a FAILED outcome."""
    monkeypatch.chdir(tmp_path)
    _stub_preflight(monkeypatch)
    _stub_runner(
        monkeypatch,
        [
            (
                {"greeter.asm.toml": SCRIPT_MACHINE, "scripts/run.py": SCRIPT_BODY},
                AgentExecResult(payload=None, reason="finish_session", usd=0.01),
            )
        ],
    )
    out_dir = tmp_path / "dest"
    (out_dir / "scripts").mkdir(parents=True)
    outside = tmp_path / "outside.py"
    outside.write_text("print('x')\n", encoding="utf-8")
    (out_dir / "scripts" / "evil.py").symlink_to(outside)
    code = main(["machine", "create", "Run a script", "-o", str(out_dir / "m.asm.toml")])
    assert code == 1
    err = capsys.readouterr().err
    assert "FAILED" in err and "does not validate" in err
    assert "OK: wrote draft" not in err


def test_create_writes_scripts_before_the_machine_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The .asm is the bundle's commit point: a death mid-publish must leave
    inert scripts, never a machine file whose scripts are missing. A script
    write failure therefore leaves no machine file behind."""
    monkeypatch.chdir(tmp_path)
    _stub_preflight(monkeypatch)
    _stub_runner(
        monkeypatch,
        [
            (
                {"greeter.asm.toml": SCRIPT_MACHINE, "scripts/run.py": SCRIPT_BODY},
                AgentExecResult(payload=None, reason="finish_session", usd=0.01),
            )
        ],
    )

    real_write = _create._write_scripts  # pyright: ignore[reportPrivateUsage]

    def boom(base_dir: Path, scripts: dict[str, str]) -> None:
        if base_dir == tmp_path:  # the publish call; scratch validation proceeds
            raise OSError("disk full")
        real_write(base_dir, scripts)

    monkeypatch.setattr(_create, "_write_scripts", boom)
    code = main(["machine", "create", "Run a script"])
    assert code == 1
    assert not (tmp_path / "scripted.asm.toml").exists()


def test_create_never_ships_script_exits_1(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    _stub_preflight(monkeypatch)
    _stub_runner(
        monkeypatch,
        [
            (
                {"greeter.asm.toml": SCRIPT_MACHINE},
                AgentExecResult(payload=None, reason="finish_session", usd=0.01),
            )
        ],
    )
    code = main(["machine", "create", "Run a script", "--max-attempts", "1"])
    assert code == 1
    out = capsys.readouterr()
    assert "not found in bundle" in out.err
    # no half-written bundle left behind.
    assert not (tmp_path / "scripted.asm.toml").exists()
    assert not (tmp_path / "scripts").exists()


def test_timed_out_agent_state_salvages_spend_from_its_event_log(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A SIGKILLed (timed-out) agent subprocess never writes result.json, but
    its event log carries the loop's running budget totals. The runner must
    book that real spend, or a 24/7 machine full of weak-model timeouts burns
    money against a $0 ledger and its budget guard never trips."""
    import subprocess as sp

    events_log = tmp_path / "logs.jsonl"

    class _HungProc:
        pid = 424242
        returncode = None

        def wait(self, timeout: float | None = None) -> int:
            if timeout is not None:
                raise sp.TimeoutExpired(cmd="agent", timeout=timeout)
            return 0

    def _popen(*_args: Any, **_kwargs: Any) -> _HungProc:
        # The subprocess writes its budget.update lines DURING the call --
        # after run_agent captured the log offset (the offset scopes a shared
        # draft log to this call's own events; see the double-book fix).
        # Writing at spawn time mirrors that, where a pre-seeded log would
        # simulate a PRIOR call's spend and correctly salvage $0.
        with events_log.open("a", encoding="utf-8") as fh:
            fh.write(
                json.dumps({"type": "budget.update", "input_total": 100, "output_total": 5}) + "\n"
            )
            fh.write(
                json.dumps(
                    {
                        "type": "budget.update",
                        "input_total": 66084,
                        "output_total": 838,
                        "usd_total": 0.0588752,
                    }
                )
                + "\n"
            )
        return _HungProc()

    def _getpgid(_pid: int) -> int:
        return 424242

    def _killpg(_pgid: int, _sig: int) -> None:
        return None

    monkeypatch.setattr(machine_agent.subprocess, "Popen", _popen)
    monkeypatch.setattr(machine_agent.os, "getpgid", _getpgid)
    monkeypatch.setattr(machine_agent.os, "killpg", _killpg)

    runner = machine_agent.build_machine_agent_runner({}, tmp_path, "strict", tmp_path / "tr")
    res = runner(AgentRequest(prompt="p", timeout_s=1.0, mode="agent"), events_log)
    assert res.reason == "timeout"
    assert res.usd == pytest.approx(0.0588752)
    assert (res.input_tokens, res.output_tokens) == (66084, 838)


def test_create_failure_end_reason_names_the_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A create that never produced a valid machine folds to ("failed", reason),
    and that reason is the detail every listing prints beside the word. It has to
    name the failure like every other emitter's token, not read as success prose
    under a failed status."""
    from agent6.viewmodel.listing import status_word

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("AGENT6_STATE_HOME", str(tmp_path / "state"))
    _stub_preflight(monkeypatch)
    _stub_runner(
        monkeypatch,
        [({}, AgentExecResult(payload=None, reason="max_iterations", usd=0.0))],
    )
    assert main(["machine", "create", "Greet the user", "--max-attempts", "1"]) == 1

    logs = list((tmp_path / "state").glob("**/sessions/machines/*/logs.jsonl"))
    events = [json.loads(line) for line in logs[0].read_text(encoding="utf-8").splitlines()]
    end = next(e for e in events if e["type"] == "session.end")
    assert end["all_passed"] is False
    word, detail = status_word(finished=True, all_passed=False, end_reason=end["reason"])
    assert word == "failed"
    assert " " not in detail  # a token, like every other session.end reason
    assert "finished" not in detail  # never success prose under a failed status


def test_create_stamps_a_liveness_marker_on_the_draft(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The draft dir is watchable -- the hub lists it and the SSE endpoints
    stream it -- but stamped no worker.pid, so a draft whose process died read
    "running" until the 10-minute log-silence window expired, holding its stream
    open the whole time. Every other watchable run-style dir records one."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("AGENT6_STATE_HOME", str(tmp_path / "state"))
    _stub_preflight(monkeypatch)
    _stub_runner(
        monkeypatch,
        [(_draft(VALID_MACHINE), AgentExecResult(payload=None, reason="finish_session", usd=0.0))],
    )
    assert main(["machine", "create", "Greet the user"]) == 0

    pids = list((tmp_path / "state").glob("**/sessions/machines/*/worker.pid"))
    assert pids, "the draft recorded no liveness marker"
    assert pids[0].read_text(encoding="utf-8").split()[0] == str(os.getpid())


def test_create_runs_the_shared_isolation_preflight(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """`machine create` goes through the run lifecycle's preflight
    (select_isolation), so the shared refusal list applies: a state base
    inside the workspace refuses here as it does for `agent6 run` and
    `machine run`. Its own copy checked only the network and hidden paths."""
    from agent6.app import _session as session_mod
    from agent6.config import Config, ModelsConfig, OpenAIProviderEntry, RoleModel
    from agent6.config.layer import EffectiveConfig
    from agent6.sandbox.detect import Environment, KernelInfo

    monkeypatch.setenv("AGENT6_CONFIG_HOME", str(tmp_path / "cfg"))
    monkeypatch.setenv("AGENT6_STATE_HOME", str(tmp_path / "inside-state"))
    monkeypatch.chdir(tmp_path)
    cfg = Config(
        providers={
            "openrouter": OpenAIProviderEntry(
                api_format="openai", base_url="https://openrouter.ai/api/v1"
            )
        },
        models=ModelsConfig(worker=RoleModel(provider="openrouter", model="kimi")),
    )

    def _load(*_a: object, **_k: object) -> EffectiveConfig:
        return EffectiveConfig(config=cfg, sources={}, layers=())

    def _keys_ok(*_a: object, **_k: object) -> None:
        return None

    def _env() -> Environment:
        return Environment(
            in_container=False,
            container_signals=(),
            kernel=KernelInfo(raw="6.14.0", major=6, minor=14),
            userns_supported=True,
            landlock_abi=4,
            seccomp_arch_supported=True,
            sandbox_available=True,
        )

    def _strict(_req: object, _env: object) -> str:
        return "strict"

    monkeypatch.setattr(_create, "load_effective", _load)
    monkeypatch.setattr(_create, "check_provider_keys", _keys_ok)
    monkeypatch.setattr(session_mod, "detect_env", _env)
    monkeypatch.setattr(session_mod, "resolve_isolation", _strict)
    assert main(["machine", "create", "a nightly loop"]) == 2
    err = capsys.readouterr().err
    assert "REFUSING" in err and "private directory" in err


def test_a_structural_failure_still_reports_the_scripts_lint_problems(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """One attempt reveals every problem class: a draft whose TOML fails
    validation AND whose script fails lint gets both in the same retry
    diagnostics, instead of schema-then-lint costing an attempt each (the
    serial reveal burned sol's whole budget on a simple machine)."""
    monkeypatch.chdir(tmp_path)
    _stub_preflight(monkeypatch)
    bad_toml = 'machine = "x"\nversion = 1\ninitial = "missing_state"\n'
    bad_script = "import subprocess\nsubprocess.run(['ls'])\n"  # PLW1510: no check=
    seen_prompts: list[str] = []

    def fake_build(
        cfg: object, root: Path, isolation: object, transcript_dir: Path, **_kw: object
    ) -> Callable[[AgentRequest], AgentExecResult]:
        def run(request: AgentRequest, _events_log: object = None) -> AgentExecResult:
            seen_prompts.append(request.prompt)
            (root / "greeter.asm.toml").write_text(bad_toml, encoding="utf-8")
            (root / "scripts").mkdir(exist_ok=True)
            (root / "scripts" / "helper.py").write_text(bad_script, encoding="utf-8")
            return AgentExecResult(payload=None, reason="finish_session", usd=0.0)

        return run

    monkeypatch.setattr(_create, "build_machine_agent_runner", fake_build)
    code = main(["machine", "create", "--max-attempts", "2", "Doomed draft"])
    assert code == 1
    retry_prompt = seen_prompts[1]
    assert "initial" in retry_prompt  # the schema problem
    assert "PLW1510" in retry_prompt  # the lint problem, in the same attempt's diagnostics


def test_the_authoring_agent_drafts_in_a_workspace_of_its_own(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The bundle is files in a workspace, not a finish payload: a ~20KB TOML
    string inside tool-call JSON defeated kimi-k2.6's emitter three times in a
    row. The leg drafts with the edit tools and no shell -- the validators need
    agent6 itself, which no jailed command can reach."""
    monkeypatch.chdir(tmp_path)
    _stub_preflight(monkeypatch)
    captured: list[tuple[AgentRequest, Path, object]] = []

    seen: list[bool] = []

    def fake_build(
        cfg: object, root: Path, isolation: object, transcript_dir: Path, **_kw: object
    ) -> Callable[[AgentRequest], AgentExecResult]:
        def run(request: AgentRequest, _events_log: object = None) -> AgentExecResult:
            captured.append((request, root, cfg))
            # Read while the leg is running: a published create removes the
            # workspace, so nothing about it survives to assert on afterwards.
            seen.append((root / ".git").exists())
            for rel, content in _draft(VALID_MACHINE).items():
                (root / rel).write_text(content, encoding="utf-8")
            return AgentExecResult(payload=None, reason="finish_session", usd=0.0)

        return run

    monkeypatch.setattr(_create, "build_machine_agent_runner", fake_build)
    assert main(["machine", "create", "Greet the user"]) == 0
    assert captured, "runner was never invoked"
    req, root, cfg = captured[0]
    assert req.mode == "run", "the authoring agent writes files"
    assert req.output_schema is None, "no finish payload: the bundle is on disk"
    assert seen == [True], "the workspace is a repo, like any run's"
    assert not root.exists(), "and it is gone once its bundle is published"
    assert root != tmp_path, "never the operator's checkout"
    # The jail masks the state dir, so a workspace inside it would be hidden
    # from the run that must write there (a live create burned 200 iterations
    # on `list_dir: Path is hidden from this run`).
    state_dir = resolved_state_dir(tmp_path)
    assert state_dir not in root.parents, "the workspace sits outside the state dir"
    assert isinstance(cfg, dict)
    # The operator's own settings ride as the overlay: the workspace's per-repo
    # layer is empty, so without them a repo-pinned worker model is invisible
    # to the leg and every attempt fails.
    assert cfg["models"] == {"worker": {"provider": "openrouter", "model": "test-model"}}
    assert cfg["sandbox"]["run_commands"] == "no", "and it still carries no command tool"
    assert "state_dir" not in cfg.get("agent6", {}), "global-only; the overlay forbids it"
    # A workspace that never ran has no state dir to remove, which rmtree
    # reports like a failure: every published create printed that it stayed.
    assert "the drafting workspace stays" not in capsys.readouterr().err


def test_create_publishes_the_files_a_referenced_script_needs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A referenced script imports modules and reads data files, and none of
    those appear in a `tool` command. Publishing only the referenced set gave
    the operator a bundle that fails at its first state, after the workspace
    holding the missing file was gone."""
    monkeypatch.chdir(tmp_path)
    _stub_preflight(monkeypatch)
    _stub_runner(
        monkeypatch,
        [
            (
                {
                    "greeter.asm.toml": SCRIPT_MACHINE,
                    "scripts/run.py": "from common import helper\n\nprint(helper())\n",
                    "scripts/common.py": "def helper() -> int:\n    return 1\n",
                    "scripts/data.json": "{}\n",
                },
                AgentExecResult(payload=None, reason="finish_session", usd=0.01),
            )
        ],
    )

    assert main(["machine", "create", "Run a script", "--max-attempts", "1"]) == 0

    assert (tmp_path / "scripts" / "common.py").exists(), "the module it imports"
    assert (tmp_path / "scripts" / "data.json").exists(), "the data file it reads"
