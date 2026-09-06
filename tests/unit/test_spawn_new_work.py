# SPDX-License-Identifier: Apache-2.0
"""`ui.spawn.spawn_new_work`: the one start every hub (TUI, web) makes for a
new run / plan / ask, its argv, its `/parallel` fan-out, and its refusals."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from agent6.ui import spawn


def _capture_locate(monkeypatch: pytest.MonkeyPatch) -> list[list[str]]:
    captured: list[list[str]] = []

    def _fake_locate(argv: list[str], cwd: Path, **_k: object) -> tuple[Path | None, str]:
        captured.append(list(argv))
        return None, "not started"

    monkeypatch.setattr(spawn, "spawn_and_locate", _fake_locate)
    return captured


def test_argv_ends_options_before_task_and_carries_the_preset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`agent6 <mode> [--preset P] -- <task>`: a task that looks like a flag rides
    behind `--`; the "(config default)" choice (preset="") adds no flag."""
    captured = _capture_locate(monkeypatch)
    spawn.spawn_new_work(tmp_path, "run", "--allow-root pwn", preset="quick")
    assert captured[-1][1:] == ["run", "--preset", "quick", "--", "--allow-root pwn"]
    spawn.spawn_new_work(tmp_path, "plan", "do it")
    assert captured[-1][1:] == ["plan", "--", "do it"]


def test_config_path_stamps_the_argv_head(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    captured = _capture_locate(monkeypatch)
    spawn.spawn_new_work(tmp_path, "ask", "why?", config_path=tmp_path / "c.toml")
    assert captured[-1][1:] == ["--config", str(tmp_path / "c.toml"), "ask", "--", "why?"]


def test_unknown_mode_and_empty_task_are_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured = _capture_locate(monkeypatch)
    assert spawn.spawn_new_work(tmp_path, "machine", "x") == (None, "unknown mode 'machine'")
    assert spawn.spawn_new_work(tmp_path, "run", "  ") == (None, "empty task")
    assert captured == []


def test_detached_env_streams_and_waits(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The child streams its reasoning to logs.jsonl (a live view renders it)
    and its approvals / questions wait for a front-end; the rest of the
    environment (PATH) is inherited."""
    captured_env: dict[str, str] = {}

    class _FakeProc:
        pid = 424242
        returncode = 0

        def poll(self) -> int:
            return 0

    def _fake_popen(argv: list[str], **kw: object) -> _FakeProc:
        env = kw.get("env")
        if isinstance(env, dict):
            captured_env.update(env)
        return _FakeProc()

    monkeypatch.setattr(subprocess, "Popen", _fake_popen)
    spawn.spawn_new_work(tmp_path, "ask", "why?")
    assert captured_env["AGENT6_STREAM_TO_LOG"] == "1"
    assert captured_env["AGENT6_DETACHED_AWAY"] == "wait"
    assert "PATH" in captured_env


# --- the `/parallel` new-work directive: fan out lanes ------------------------


def test_parallel_lane_count(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    captured = _capture_locate(monkeypatch)
    spawn.spawn_new_work(tmp_path, "run", "/parallel 2 add a greeting")
    assert captured[-1][1:] == ["run", "--parallel", "2", "--", "add a greeting"]


def test_parallel_model_list_with_preset(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    captured = _capture_locate(monkeypatch)
    spawn.spawn_new_work(tmp_path, "run", "/parallel gpt-5,opus refactor", preset="quick")
    assert captured[-1][1:] == [
        "run", "--preset", "quick", "--parallel", "gpt-5,opus", "--", "refactor",
    ]  # fmt: skip


def test_malformed_parallel_is_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    captured = _capture_locate(monkeypatch)
    session_dir, err = spawn.spawn_new_work(tmp_path, "run", "/parallel")
    assert session_dir is None
    assert "/parallel" in err  # the directive error surfaces, no spawn happens
    assert captured == []


class _Eff:
    def __init__(self, cfg: object) -> None:
        self.config = cfg


def _provider_cfg() -> object:
    from agent6.config import Config

    return Config.model_validate(
        {
            "providers": {"o": {"api_format": "openai", "base_url": "https://x/v1"}},
            "models": {"worker": {"provider": "o", "model": "moonshotai/kimi-k2.6"}},
        }
    )


def test_parallel_refuses_unknown_model_before_spawn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A cache exists to validate against -> a typo'd model is the composer's normal
    # error path, nothing spawned.
    cache = tmp_path / "cache" / "agent6" / "models"
    cache.mkdir(parents=True)
    (cache / "o.json").write_text(
        json.dumps({"models": ["moonshotai/kimi-k2.6"]}), encoding="utf-8"
    )
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))

    # The miss re-checks the live listing before refusing; stub it with the
    # same ids so the refusal rests on "fresh" evidence (no real network).
    from agent6.models import validate as models_validate

    def _listing(*_a: object) -> list[str] | None:
        return ["moonshotai/kimi-k2.6"]

    monkeypatch.setattr(models_validate, "_fresh_listing", _listing)

    def _eff(_cwd: object, _cp: object = None) -> _Eff:
        return _Eff(_provider_cfg())

    monkeypatch.setattr(models_validate, "load_effective", _eff)
    captured = _capture_locate(monkeypatch)
    session_dir, err = spawn.spawn_new_work(
        tmp_path, "run", "/parallel moonshotai/kimi-k2.7 fix it"
    )
    assert session_dir is None
    assert "unknown model 'moonshotai/kimi-k2.7'" in err
    assert "closest: moonshotai/kimi-k2.6" in err
    assert captured == []  # nothing spawned


def test_parallel_unknown_model_no_cache_proceeds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # No cache to validate against -> never block; the detached lane's own
    # preflight warns. The spawn happens.
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "empty-cache"))

    from agent6.models import validate as models_validate

    def _eff(_cwd: object, _cp: object = None) -> _Eff:
        return _Eff(_provider_cfg())

    monkeypatch.setattr(models_validate, "load_effective", _eff)
    captured = _capture_locate(monkeypatch)
    spawn.spawn_new_work(tmp_path, "run", "/parallel made-up/model fix it")
    assert captured[-1][1:] == ["run", "--parallel", "made-up/model", "--", "fix it"]


def test_parallel_only_for_run_mode(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # plan/ask cannot fan out: the text is a literal task (rides behind `--`).
    captured = _capture_locate(monkeypatch)
    spawn.spawn_new_work(tmp_path, "plan", "/parallel 2 add a greeting")
    assert captured[-1][1:] == ["plan", "--", "/parallel 2 add a greeting"]


def test_parallel_omitted_spec_is_one_lane(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # No spec -> one isolated lane: --parallel 1 (a clone lane, not an in-place run).
    captured = _capture_locate(monkeypatch)
    spawn.spawn_new_work(tmp_path, "run", "/parallel refactor the parser")
    assert captured[-1][1:] == ["run", "--parallel", "1", "--", "refactor the parser"]


def test_parallel_multi_segment_spawns_one_fanout_per_segment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured = _capture_locate(monkeypatch)
    spawn.spawn_new_work(tmp_path, "run", "/parallel 2 task A /parallel gpt-5,opus task B")
    assert [c[1:] for c in captured] == [
        ["run", "--parallel", "2", "--", "task A"],
        ["run", "--parallel", "gpt-5,opus", "--", "task B"],
    ]


def test_parallel_partial_spawn_failure_surfaces(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One lane failing to spawn fails the whole message: the surfaces open
    the run XOR show the error, so a swallowed failure must not navigate."""

    def fake_spawn(
        cwd: Path, mode: str, task: str, *, preset: str, spec: str, config_path: object = None
    ) -> tuple[Path | None, str]:
        if "task B" in task:
            return None, "boom"
        return tmp_path / "run-A", ""

    monkeypatch.setattr(spawn, "_spawn_run", fake_spawn)

    def no_refusal(cwd: Path, segments: object, config_path: object = None) -> None:
        return None

    monkeypatch.setattr(spawn, "directive_model_refusal", no_refusal)
    session_dir, err = spawn.spawn_new_work(
        tmp_path, "run", "/parallel 2 task A /parallel 3 task B"
    )
    assert session_dir is None
    assert "boom" in err and "task B" in err
    # The lane that started is named, so a resend of the message does not
    # launch it a second time.
    assert err.index("lane 1 (task A): running as run-A") < err.index("lane 2")  # lane order


def test_multi_segment_malformed_spawns_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # all-or-nothing: a later empty segment refuses the whole message, no spawn.
    captured = _capture_locate(monkeypatch)
    session_dir, err = spawn.spawn_new_work(tmp_path, "run", "/parallel 2 good task /parallel")
    assert session_dir is None and "/parallel" in err
    assert captured == []


def test_a_busy_checkout_is_refused_at_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A run into a checkout another run is driving would park after the
    locate wait; the hub says so at once instead (plan/ask spawn freely)."""
    captured = _capture_locate(monkeypatch)

    def _held(_state: Path) -> bool:
        return True

    def _holder(_state: Path) -> str:
        return "busy-run"

    monkeypatch.setattr(spawn, "repo_writer_held", _held)
    monkeypatch.setattr(spawn, "repo_writer_holder", _holder)
    session_dir, err = spawn.spawn_new_work(tmp_path, "run", "do it")
    assert session_dir is None and "busy-run" in err
    assert captured == []
    spawn.spawn_new_work(tmp_path, "ask", "why?")
    assert captured[-1][1:] == ["ask", "--", "why?"]


def test_detached_resume_refuses_a_malformed_steer_before_spawning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A composer's continue on a finished run rides a detached `resume
    --steer`; a bare `/pin` (a `/parallel` with no task) is refused HERE, with
    the message the child would print to a stdio nobody reads while the
    composer says "resuming"."""

    def _must_not_spawn(*_a: object, **_k: object) -> object:
        pytest.fail("nothing may be spawned for a malformed steer")

    monkeypatch.setattr(subprocess, "Popen", _must_not_spawn)
    err = spawn.spawn_detached_resume(tmp_path, "runny-one-AAAAAA", steer="/pin")
    assert "pin needs an instruction" in err
    assert "needs a task" in spawn.spawn_detached_resume(
        tmp_path, "runny-one-AAAAAA", steer="/parallel 3"
    )


def test_a_timeout_says_what_it_knows(tmp_path: Path) -> None:
    """A child still starting after the wait is not known to have failed: a
    slow resume preflight read as "has not started" while the run went on."""
    err = spawn.spawn_and_confirm(
        ["sleep", "3"], tmp_path, started=lambda pid: False, timeout_s=0.5
    )
    assert "has not reported starting within 0s (`agent6 ps` shows whether it is running)" in err
