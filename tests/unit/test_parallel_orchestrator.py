# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The `agent6 run --parallel` fan-out orchestrator (ui/cli/parallel.py).

Driven with a fake spawner that, for each LaneSpec, really clones the origin,
commits on the lane's `agent6/<id>` branch, and fabricates a finished run dir
(manifest.json + logs.jsonl) -- so the orchestrator's clone-independent behavior
(symlink live view, import, lineage stamp, ranked report, resilience to a failed
lane) is exercised on real tmp git repos without spawning real runs.
"""

from __future__ import annotations

import json
import os
from collections.abc import Generator, Mapping
from dataclasses import replace
from pathlib import Path

import pytest

from agent6 import memory
from agent6.app import _lane_watch as lane_watch
from agent6.app import parallel
from agent6.app.compare import RankOutcome
from agent6.app.parallel import (
    LaneRuntime,
    ParallelError,
    build_lane_specs,
    run_parallel,
)
from agent6.app.reporter import STDIO_REPORTER
from agent6.config import Config
from agent6.directive import DirectiveError
from agent6.git_ops import branch_exists, commit_all, create_branch
from agent6.memory import decisions_text, record_decision
from agent6.paths import state_dir
from agent6.ui.cli import parallel as parallel_cmd
from agent6.ui.cli.parallel import lane_runtime
from agent6.workflows.subrun import LaneResult, LaneSpec, LaneTask, clone_workspace


def _git(repo: Path, *args: str) -> None:
    import subprocess

    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)


def _init_repo(repo: Path) -> None:
    repo.mkdir(parents=True, exist_ok=True)
    _git(repo, "init", "-q", "-b", "main")
    _git(repo, "config", "user.email", "t@t")
    _git(repo, "config", "user.name", "t")
    (repo / "README.md").write_text("hi\n", encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "init")


def _write_fake_run(
    session_dir: Path,
    task: str,
    *,
    status: str,
    cost: float,
    parallel_id: str | None = None,
    lane: int | None = None,
) -> None:
    session_dir.mkdir(parents=True)
    # A real lane's manifest records its lineage from birth (the spawn env);
    # the fakes mirror that for `<fanout>-l<N>` ids so post-import assertions
    # hold for the same reason they do live.
    manifest: dict[str, object] = {
        "version": 2,
        "session_id": session_dir.name,
        "mode": "run",
        "user_task": task,
    }
    fanout, sep, lane_s = session_dir.name.rpartition("-l")
    if parallel_id is not None:
        manifest["parallel_id"] = parallel_id
        manifest["lane"] = lane
    elif sep and fanout and lane_s.isdigit():
        manifest["parallel_id"] = fanout
        manifest["lane"] = int(lane_s)
    (session_dir / "manifest.json").write_text(json.dumps(manifest) + "\n", encoding="utf-8")
    events: list[dict[str, object]] = [
        {"type": "session.start", "mode": "run", "user_task": task},
        {"type": "budget.update", "usd_total": cost},
    ]
    if status == "passed":
        events.append({"type": "session.end", "reason": "finish_session", "all_passed": True})
    elif status == "failed":
        events.append({"type": "session.end", "reason": "provider_error", "all_passed": False})
    elif status == "stale":
        # Died without a session.end (OOM/SIGKILL): a recorded pid that is gone.
        (session_dir / "worker.pid").write_text("999999999", encoding="utf-8")
    else:  # "finished": a deliberate finish without all-passed
        events.append({"type": "session.end", "reason": "finish_session", "all_passed": False})
    (session_dir / "logs.jsonl").write_text(
        "\n".join(json.dumps(e) for e in events) + "\n", encoding="utf-8"
    )


class _FakeSpawner:
    """A synchronous stand-in for the bridge spawner: clone, commit on the lane
    branch, and fabricate a finished run dir. Records what it observed so a test
    can assert the orchestrator's symlink-then-replace behavior."""

    def __init__(
        self,
        origin: Path,
        origin_state: Path,
        state_root: Path,
        *,
        fail: set[int] | None = None,
        status_by_lane: dict[int, str] | None = None,
        cost_by_lane: dict[int, float] | None = None,
        pid_lanes: set[int] | None = None,
        fanout_id: str | None = None,
    ) -> None:
        self.origin = origin
        self.origin_state = origin_state
        self.state_root = state_root
        self.fail = fail or set()
        self.status_by_lane = status_by_lane or {}
        self.cost_by_lane = cost_by_lane or {}
        # Lanes whose fabricated run dir carries a LIVE worker.pid (this test
        # process), simulating the teardown window where session.end is already in
        # logs.jsonl but the lane process has not yet cleared its pid.
        self.pid_lanes = pid_lanes or set()
        # The group/fanout id the real bridge spawner stamps into the lane env;
        # the fabricated manifest mirrors it (None = derive from the id).
        self.fanout_id = fanout_id
        self.prior_link_was_symlink: dict[int, bool] = {}
        self.tasks: list[str] = []

    def __call__(self, spec: LaneSpec, task: str) -> LaneResult:
        self.tasks.append(task)
        branch = f"agent6/{spec.session_id}"
        if spec.lane > 1:  # observe the previous lane's live symlink
            prefix = spec.session_id.rsplit("-l", 1)[0]
            prior = self.origin_state / "sessions" / "runs" / f"{prefix}-l{spec.lane - 1}"
            self.prior_link_was_symlink[spec.lane] = prior.is_symlink()
        if spec.lane in self.fail:
            return LaneResult(
                spec=spec, session_dir=spec.workdir, branch=branch, ok=False, error="boom"
            )
        clone_workspace(self.origin, spec.workdir)
        create_branch(spec.workdir, branch)
        (spec.workdir / f"lane{spec.lane}.txt").write_text(f"lane {spec.lane}\n", encoding="utf-8")
        commit_all(spec.workdir, f"lane {spec.lane} work")
        session_dir = self.state_root / f"lane{spec.lane}" / "sessions" / "runs" / spec.session_id
        _write_fake_run(
            session_dir,
            task,
            parallel_id=self.fanout_id,
            lane=spec.lane if self.fanout_id is not None else None,
            status=self.status_by_lane.get(spec.lane, "passed"),
            cost=self.cost_by_lane.get(spec.lane, 0.05),
        )
        if spec.lane in self.pid_lanes:
            (session_dir / "worker.pid").write_text(str(os.getpid()), encoding="utf-8")
        return LaneResult(spec=spec, session_dir=session_dir, branch=branch, ok=True, error="")


@pytest.fixture
def origin(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    o = tmp_path / "origin"
    _init_repo(o)
    return o


@pytest.fixture
def runtime() -> LaneRuntime:
    """The real front-end LaneRuntime the pipeline drives (detached process spawn
    + reviewer/judging wiring). Tests faking one primitive use `dataclasses.replace`
    on it (e.g. a fake `spawn`), or `monkeypatch` the module-level `worker_is_alive`
    (the run-dir bridge, imported directly -- no longer a LaneRuntime field)."""
    return lane_runtime()


def _specs(tmp_path: Path, cfg: Config, fanout_id: str, spec: str) -> list[LaneSpec]:
    return build_lane_specs(
        spec,
        cfg=cfg,
        origin=tmp_path,
        fanout_id=fanout_id,
        workdir_root=tmp_path / "work" / fanout_id,
    )


# ---------------------------------------------------------------------------
# Lane planning
# ---------------------------------------------------------------------------


def test_build_lane_specs_int_layout(tmp_path: Path) -> None:
    lanes = _specs(tmp_path, Config(), "fan", "3")
    assert [(ln.lane, ln.session_id, ln.model) for ln in lanes] == [
        (1, "fan-l1", None),
        (2, "fan-l2", None),
        (3, "fan-l3", None),
    ]
    assert lanes[0].workdir == tmp_path / "work" / "fan" / "lane-1"


def test_build_lane_specs_model_list(tmp_path: Path) -> None:
    lanes = _specs(tmp_path, Config(), "fan", "kimi,glm")
    assert [(ln.lane, ln.model) for ln in lanes] == [(1, "kimi"), (2, "glm")]


def test_build_lane_specs_over_cap_refused(tmp_path: Path) -> None:
    cfg = Config.model_validate({"parallel": {"max_lanes": 2}})
    with pytest.raises(DirectiveError, match="max_lanes"):
        _specs(tmp_path, cfg, "fan", "5")


def test_build_lane_specs_rejects_zero(tmp_path: Path) -> None:
    # Every spec-shape error, the cap included, comes from the shared grammar:
    # parse_spec refuses over-limit BEFORE building the lane list.
    with pytest.raises(DirectiveError):
        _specs(tmp_path, Config(), "fan", "0")


# ---------------------------------------------------------------------------
# Pre-spawn model validation (B3): a bogus model refuses before any clone.
# ---------------------------------------------------------------------------


def _write_models_cache(cache_home: Path, provider: str, models: list[str]) -> None:
    p = cache_home / "models" / f"{provider}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"models": models}), encoding="utf-8")


def _provider_cfg(model: str = "moonshotai/kimi-k2.6", run_commands: str = "yes") -> Config:
    # run_commands defaults to "yes" here: `--parallel` refuses under "ask",
    # since nobody can answer a detached lane, and these tests are about the
    # dispatch itself. The refusal has its own test.
    return Config.model_validate(
        {
            "providers": {"o": {"api_format": "openai", "base_url": "https://x/v1"}},
            "models": {"worker": {"provider": "o", "model": model}},
            "sandbox": {"run_commands": run_commands},
        }
    )


def test_dispatch_parallel_refuses_unknown_model_before_any_clone(
    origin: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    _write_models_cache(
        tmp_path / "cache" / "agent6", "o", ["moonshotai/kimi-k2.6", "z-ai/glm-4.6"]
    )

    # The miss now re-checks the live listing before refusing; stub it with the
    # same ids so the refusal rests on "fresh" evidence (no real network).
    from agent6.models import validate as models_validate

    def _listing(*_a: object) -> list[str] | None:
        return ["moonshotai/kimi-k2.6", "z-ai/glm-4.6"]

    monkeypatch.setattr(models_validate, "_fresh_listing", _listing)

    def _boom(*_a: object, **_k: object) -> int:
        raise AssertionError("run_parallel must not be reached on a refusal")

    monkeypatch.setattr(parallel_cmd, "run_parallel", _boom)
    rc = parallel_cmd.dispatch_parallel(
        _provider_cfg(), "fix the bug", "moonshotai/kimi-k2.7", cwd=origin
    )
    assert rc == 2
    err = capsys.readouterr().err
    assert "REFUSING" in err
    assert "unknown model 'moonshotai/kimi-k2.7'" in err
    assert "closest: moonshotai/kimi-k2.6" in err


def test_dispatch_parallel_unknown_model_no_cache_warns_and_proceeds(
    origin: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))  # empty: no snapshot

    reached: list[str] = []

    def _fake_run(task: str, lanes: object, **_k: object) -> int:
        reached.append(task)
        return 0

    monkeypatch.setattr(parallel_cmd, "run_parallel", _fake_run)
    rc = parallel_cmd.dispatch_parallel(_provider_cfg(), "fix the bug", "made-up/model", cwd=origin)
    assert rc == 0
    assert reached == ["fix the bug"]  # not blocked offline
    assert "WARNING" in capsys.readouterr().err


def test_dispatch_parallel_forwards_auto_approve_to_run_parallel(
    origin: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The CLI's `--auto-approve` must reach `run_parallel`, same as --max-usd."""
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    captured: list[object] = []

    def _fake_run(task: str, lanes: object, **kw: object) -> int:
        captured.append(kw.get("auto_approve"))
        return 0

    monkeypatch.setattr(parallel_cmd, "run_parallel", _fake_run)
    parallel_cmd.dispatch_parallel(
        _provider_cfg(), "fix the bug", "made-up/model", cwd=origin, auto_approve=True
    )
    parallel_cmd.dispatch_parallel(_provider_cfg(), "fix the bug", "made-up/model", cwd=origin)

    assert captured == [True, False]


def test_dispatch_parallel_forwards_pins_to_run_parallel(
    origin: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`run --parallel --pin X` must reach run_parallel: the CLI fan-out
    returns before run_task, so my C5 threaded --pin only through the in-loop
    /parallel path -- the flag's own help promised the CLI fan-out."""
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    captured: list[object] = []

    def _fake_run(task: str, lanes: object, **kw: object) -> int:
        captured.append(kw.get("pins"))
        return 0

    monkeypatch.setattr(parallel_cmd, "run_parallel", _fake_run)
    parallel_cmd.dispatch_parallel(
        _provider_cfg(), "fix", "made-up/model", cwd=origin, pins=("never touch schema",)
    )
    assert captured == [("never touch schema",)]


def test_coordinator_dispatch_refuses_unknown_model(
    origin: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, runtime: LaneRuntime
) -> None:
    """The ui-built group dispatcher validates before cloning: an unknown model
    raises, and the loop's group-failure feedback (its `except Exception`) carries
    the message to the coordinator -- so workflows needs no models dependency."""
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    _write_models_cache(tmp_path / "cache" / "agent6", "o", ["moonshotai/kimi-k2.6"])

    # The miss now re-checks the live listing before refusing; stub it with the
    # same ids so the refusal rests on "fresh" evidence (no real network).
    from agent6.models import validate as models_validate

    def _listing(*_a: object) -> list[str] | None:
        return ["moonshotai/kimi-k2.6"]

    monkeypatch.setattr(models_validate, "_fresh_listing", _listing)
    origin_state = tmp_path / "ostate"
    origin_state.mkdir()

    def _boom(*_a: object, **_k: object) -> None:
        raise AssertionError("clone must not happen before validation")

    monkeypatch.setattr(parallel, "clone_workspace", _boom)
    dispatch = parallel.build_lane_spawner(
        _provider_cfg(), origin, origin_state, coordinator_session_id="coord", runtime=runtime
    )
    with pytest.raises(ParallelError, match=r"unknown model 'moonshotai/kimi-k2\.7'"):
        dispatch([LaneTask(task="do it", model="moonshotai/kimi-k2.7")], "p1")


def test_coordinator_dispatch_aborts_promptly_on_a_lane_thread_raise(
    origin: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, runtime: LaneRuntime
) -> None:
    """A lane thread that RAISES (a bug, not a lane failure -- those return
    ok=False) must abort the group NOW: awaiting futures in submission order
    left the raise unobserved (and hard_stop unset) until every
    earlier-submitted lane happened to finish on its own."""
    import time

    origin_state = tmp_path / "ostate"
    origin_state.mkdir()
    seen: dict[str, bool] = {}

    def fake_lane(spec: LaneSpec, task: str, **kwargs: object) -> LaneResult:
        import threading

        hard_stop = kwargs["hard_stop"]
        assert isinstance(hard_stop, threading.Event)
        if spec.lane == 1:
            # Stands in for a long-running healthy lane: it unblocks only when
            # the dispatcher propagates the other lane's failure.
            seen["lane1_released_by_stop"] = hard_stop.wait(timeout=8.0)
            return LaneResult(
                spec=spec, session_dir=tmp_path / "l1", branch="b1", ok=True, error=""
            )
        raise RuntimeError("lane thread blew up")

    monkeypatch.setattr(parallel, "run_lane_to_completion", fake_lane)
    dispatch = parallel.build_lane_spawner(
        _provider_cfg(), origin, origin_state, coordinator_session_id="coord", runtime=runtime
    )
    start = time.monotonic()
    with pytest.raises(RuntimeError, match="lane thread blew up"):
        dispatch([LaneTask(task="a", model=None), LaneTask(task="b", model=None)], "p1")
    assert time.monotonic() - start < 4.0  # not after lane 1's own sweet time
    # The dispatcher re-raises without joining still-running lane threads
    # (deliberate: teardown must not block on lanes), so give lane 1's thread
    # a bounded moment to record that hard_stop released it.
    deadline = time.monotonic() + 2.0
    while "lane1_released_by_stop" not in seen and time.monotonic() < deadline:
        time.sleep(0.01)
    assert seen.get("lane1_released_by_stop") is True  # hard_stop actually fired


def test_bridge_spawner_argv_ends_options_before_task(
    origin: Path, tmp_path: Path, runtime: LaneRuntime
) -> None:
    """The lane spawner puts every flag before `--` and the task after it, so a
    task that looks like a flag can never be parsed as one (matches web/TUI). The
    agent6 executable is folded into the injected `spawn`, so the argv it receives
    starts at the subcommand."""
    captured: list[list[str]] = []

    def fake_spawn(argv: list[str], workdir: Path, **_k: object) -> tuple[Path, str]:
        captured.append(list(argv))
        return workdir, ""

    cfg = Config()
    spec = LaneSpec(lane=1, session_id="fan-l1", workdir=tmp_path / "work" / "lane-1", model=None)
    parallel.bridge_spawner(
        spec, "--allow-root pwn", cfg=cfg, origin=origin, max_usd=2.0,
        fanout_id="fan", runtime=replace(runtime, spawn=fake_spawn),
    )  # fmt: skip

    argv = captured[-1]
    assert argv[:1] == ["run"]  # the exe is prepended inside the injected spawn
    dd = argv.index("--")
    assert {"--session-id", "--config", "--max-usd"} <= set(argv[:dd])  # flags precede `--`
    assert argv[dd + 1 :] == ["--allow-root pwn"]  # task is the sole element after


def test_a_lane_is_seeded_with_the_repos_memory(
    origin: Path, tmp_path: Path, runtime: LaneRuntime
) -> None:
    """A lane clones the repo, so its state dir is new and its memory empty: the
    lanes ran blind to the facts and rulings every other run on that repo gets."""
    cfg = Config()
    origin_state = state_dir(origin)
    memory.add(origin_state, "house-style", "Docstrings end in a period.")
    record_decision(origin_state, question="Ship the rename?", answer="yes", session="s", when=0.0)

    def fake_spawn(_argv: list[str], workdir: Path, **_k: object) -> tuple[Path, str]:
        return workdir, ""

    spec = LaneSpec(lane=1, session_id="fan-l1", workdir=tmp_path / "work" / "lane-1", model=None)
    parallel.bridge_spawner(
        spec, "task", cfg=cfg, origin=origin, max_usd=None,
        fanout_id="fan", runtime=replace(runtime, spawn=fake_spawn),
    )  # fmt: skip

    lane_state = state_dir(spec.workdir)
    assert "house-style" in memory.index_text(lane_state)
    assert "Docstrings end in a period." in memory.show(lane_state, "house-style")
    assert "Ship the rename?" in decisions_text(lane_state)


def test_bridge_spawner_argv_includes_auto_approve_when_set(
    origin: Path, tmp_path: Path, runtime: LaneRuntime
) -> None:
    """A coordinator/fan-out started with --auto-approve must forward it to the
    lane, or the lane sits on run_commands=ask with nothing to answer it."""
    captured: list[list[str]] = []

    def fake_spawn(argv: list[str], workdir: Path, **_k: object) -> tuple[Path, str]:
        captured.append(list(argv))
        return workdir, ""

    cfg = Config()
    spec = LaneSpec(lane=1, session_id="fan-l1", workdir=tmp_path / "work" / "lane-1", model=None)
    parallel.bridge_spawner(
        spec, "do it", cfg=cfg, origin=origin, max_usd=None, auto_approve=True,
        fanout_id="fan", runtime=replace(runtime, spawn=fake_spawn),
    )  # fmt: skip

    argv = captured[-1]
    dd = argv.index("--")
    assert "--auto-approve" in argv[:dd]  # precedes the `--` separator, like --max-usd


def test_bridge_spawner_argv_omits_auto_approve_by_default(
    origin: Path, tmp_path: Path, runtime: LaneRuntime
) -> None:
    captured: list[list[str]] = []

    def fake_spawn(argv: list[str], workdir: Path, **_k: object) -> tuple[Path, str]:
        captured.append(list(argv))
        return workdir, ""

    cfg = Config()
    spec = LaneSpec(lane=1, session_id="fan-l1", workdir=tmp_path / "work" / "lane-1", model=None)
    parallel.bridge_spawner(
        spec, "do it", cfg=cfg, origin=origin, max_usd=None,
        fanout_id="fan", runtime=replace(runtime, spawn=fake_spawn),
    )  # fmt: skip

    assert "--auto-approve" not in captured[-1]


def test_run_lane_to_completion_forwards_auto_approve_to_the_default_spawner(
    origin: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, runtime: LaneRuntime
) -> None:
    """When no *spawner* is injected, `run_lane_to_completion` builds the real
    bridge spawner itself (the coordinator's path); auto_approve must reach it
    exactly like max_usd already does."""
    captured: list[dict[str, object]] = []

    def fake_bridge(spec: LaneSpec, task: str, **kw: object) -> LaneResult:
        captured.append(kw)
        return LaneResult(
            spec=spec, session_dir=spec.workdir, branch="agent6/x", ok=False, error="stub"
        )

    monkeypatch.setattr(parallel, "bridge_spawner", fake_bridge)
    cfg = Config()
    spec = LaneSpec(
        lane=1, session_id="co-p1-l1", workdir=tmp_path / "work" / "co-p1-l1", model=None
    )

    parallel.run_lane_to_completion(
        spec,
        "do it",
        cfg=cfg,
        origin=origin,
        origin_state=tmp_path / "ostate",
        group="p1",
        runtime=runtime,
        auto_approve=True,
    )

    assert captured[-1]["auto_approve"] is True


def test_run_parallel_forwards_pins_to_the_default_spawner(
    origin: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, runtime: LaneRuntime
) -> None:
    """run_parallel must carry --pin into every lane's bridge spawner (which
    turns them into repeatable --pin argv, already tested)."""
    from agent6.paths import state_dir

    origin_state = state_dir(origin)
    cfg = Config()
    lanes = _specs(tmp_path, cfg, "fan", "1")
    captured: list[dict[str, object]] = []

    def fake_bridge(spec: LaneSpec, task: str, **kw: object) -> LaneResult:
        captured.append(kw)
        return LaneResult(
            spec=spec, session_dir=spec.workdir, branch="agent6/x", ok=False, error="s"
        )

    monkeypatch.setattr(parallel, "bridge_spawner", fake_bridge)
    run_parallel(
        "t", lanes, cfg=cfg, origin=origin, origin_state=origin_state,
        runtime=runtime, fanout_id="fan", pins=("keep it",),
    )  # fmt: skip
    assert captured[-1]["pins"] == ("keep it",)


# ---------------------------------------------------------------------------
# Pending-ask probe: a "running" lane blocked on an approval/question
# ---------------------------------------------------------------------------


def _lane_logs(session_dir: Path, *events: Mapping[str, object]) -> None:
    session_dir.mkdir(parents=True, exist_ok=True)
    (session_dir / "logs.jsonl").write_text(
        "".join(json.dumps(e) + "\n" for e in events), encoding="utf-8"
    )


def test_pending_prompt_reads_last_prompt_answer_event(tmp_path: Path) -> None:
    start = {"type": "session.start", "mode": "run", "user_task": "t"}
    q = tmp_path / "q"
    _lane_logs(q, start, {"type": "question.prompt", "id": "question-1"})
    assert lane_watch.pending_prompt(q) == "a question"

    a = tmp_path / "a"
    _lane_logs(a, start, {"type": "approval.prompt", "id": "approval-1"})
    assert lane_watch.pending_prompt(a) == "approval"

    # answered -> not waiting; and a dir with no prompt events -> "".
    answered = tmp_path / "answered"
    _lane_logs(
        answered,
        start,
        {"type": "question.prompt", "id": "question-1"},
        {"type": "question.answer", "id": "question-1", "answers": ["yes"]},
    )
    assert lane_watch.pending_prompt(answered) == ""
    plain = tmp_path / "plain"
    _lane_logs(plain, start, {"type": "tool.call", "name": "read_file"})
    assert lane_watch.pending_prompt(plain) == ""


def test_await_lanes_status_line_flags_a_waiting_lane(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    runtime: LaneRuntime,
) -> None:
    """Since the status unification a lane blocked on an unanswered prompt
    reads "waiting", not "running" -- the word the hint was keyed on, so the
    fan-out sat on a bare "waiting" forever with no pointer at the hub. The
    hint must fire on the real word, with pending_prompt supplying only the
    approval-vs-question wording."""
    from agent6.viewmodel import SessionSummary

    lane = tmp_path / "lane"
    _lane_logs(
        lane,
        {"type": "session.start", "mode": "run", "user_task": "t"},
        {"type": "question.prompt", "id": "question-1"},
    )
    spec = LaneSpec(lane=1, session_id="fan-l1", workdir=tmp_path / "wd", model=None)
    res = LaneResult(spec=spec, session_dir=lane, branch="agent6/fan-l1", ok=True, error="")

    # One poll only: "waiting" prints the hint, and the dead worker makes that
    # same poll terminal (lane_terminal), so a second status is never read.
    statuses = iter(["waiting"])

    def fake_summary(rd: Path) -> SessionSummary:
        return SessionSummary(
            session_id=rd.name, mode="run", task="t", status=next(statuses),
            reason="", cost_usd=0.0, usd_partial=False, mtime=0.0,
        )  # fmt: skip

    def fake_worker_is_alive(_session_dir: Path) -> bool:
        return False

    def fake_sleep(*_args: object) -> None:
        return None

    monkeypatch.setattr(lane_watch, "summarize_session_dir", fake_summary)
    monkeypatch.setattr(lane_watch.time, "sleep", fake_sleep)
    monkeypatch.setattr(lane_watch, "worker_is_alive", fake_worker_is_alive)

    assert lane_watch.await_lanes([res]) is False
    assert "waiting on a question (answer via agent6 attach" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def test_run_parallel_imports_branches_and_stamps_lineage(
    origin: Path, tmp_path: Path, runtime: LaneRuntime
) -> None:
    from agent6.paths import state_dir

    origin_state = state_dir(origin)
    cfg = Config()
    lanes = _specs(tmp_path, cfg, "fan", "2")
    spawner = _FakeSpawner(origin, origin_state, tmp_path / "lane-state")

    rc = run_parallel(
        "do the task",
        lanes,
        cfg=cfg,
        origin=origin,
        origin_state=origin_state,
        runtime=runtime,
        spawner=spawner,
        fanout_id="fan",
    )

    assert rc == 0
    # Both lane branches landed in the origin.
    assert branch_exists(origin, "agent6/fan-l1")
    assert branch_exists(origin, "agent6/fan-l2")
    # The live symlink was replaced by the real imported dir.
    imported = origin_state / "sessions" / "runs" / "fan-l1"
    assert imported.is_dir() and not imported.is_symlink()
    # Lineage was stamped post-import.
    manifest = json.loads((imported / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["parallel_id"] == "fan"
    assert manifest["lane"] == 1
    assert (
        json.loads(
            (origin_state / "sessions" / "runs" / "fan-l2" / "manifest.json").read_text(
                encoding="utf-8"
            )
        )["lane"]
        == 2
    )


def test_a_lanes_memory_files_are_carried_into_the_origin_at_import(
    origin: Path, tmp_path: Path, runtime: LaneRuntime, capsys: pytest.CaptureFixture[str]
) -> None:
    """The harness nudges a lane to write memory like any run, and the import
    carried only its rulings before tearing the lane's state dir down: every
    memory file the lane wrote, and its index line, went with it. A name the
    origin already holds with other content (written here before the lane was
    seeded, so not a seeded copy) is held back, kept in the lane's imported
    run dir, and the note says where."""
    from agent6.paths import state_dir

    origin_state = state_dir(origin)
    cfg = Config()
    memory.add(origin_state, "repo-fact", "The build needs BUILD_ID set.")
    lanes = _specs(tmp_path, cfg, "fan", "1")
    lane_state = state_dir(lanes[0].workdir)
    memory.add(lane_state, "lane-fact", "The flaky test is test_clock.")
    memory.add(lane_state, "repo-fact", "The build needs BUILD_ID set, and BUILD_NO.")
    spawner = _FakeSpawner(origin, origin_state, tmp_path / "lane-state")

    assert (
        run_parallel(
            "do the task",
            lanes,
            cfg=cfg,
            origin=origin,
            origin_state=origin_state,
            runtime=runtime,
            spawner=spawner,
            fanout_id="fan",
        )
        == 0
    )
    assert "The flaky test is test_clock." in memory.show(origin_state, "lane-fact")
    assert "lane-fact" in memory.index_text(origin_state)
    assert (
        memory.index_text(origin_state).count("repo-fact") == 1
    )  # the seeded copy is not a second line
    assert "BUILD_NO" not in memory.show(origin_state, "repo-fact")
    held = origin_state / "sessions" / "runs" / lanes[0].session_id / "memory-held" / "repo-fact.md"
    assert "BUILD_NO" in held.read_text(encoding="utf-8")
    err = capsys.readouterr().err
    assert (
        "lane 1: memory 1 carried, held back (changed here too, or the name is taken):"
        f" repo-fact, kept at {held.parent}"
    ) in err


def test_run_parallel_forwards_auto_approve_to_the_default_spawner(
    origin: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, runtime: LaneRuntime
) -> None:
    """`run --parallel --auto-approve` must reach the lane's own default (real)
    bridge spawner, same plumbing as --max-usd."""
    from agent6.paths import state_dir

    origin_state = state_dir(origin)
    cfg = Config()
    lanes = _specs(tmp_path, cfg, "fan", "1")
    captured: list[dict[str, object]] = []

    def fake_bridge(spec: LaneSpec, task: str, **kw: object) -> LaneResult:
        captured.append(kw)
        return LaneResult(
            spec=spec, session_dir=spec.workdir, branch="agent6/x", ok=False, error="s"
        )

    monkeypatch.setattr(parallel, "bridge_spawner", fake_bridge)

    run_parallel(
        "t", lanes, cfg=cfg, origin=origin, origin_state=origin_state,
        runtime=runtime, fanout_id="fan", auto_approve=True,
    )  # fmt: skip

    assert captured[-1]["auto_approve"] is True


def test_compare_outcome_stamped_into_each_lane_manifest(
    origin: Path, tmp_path: Path, runtime: LaneRuntime
) -> None:
    """The fan-out's auto-compare stamps a `compare` block into EVERY imported
    lane's manifest (winner + loser), recording rank/of/winner and, with no
    reviewer configured, ranked_by="mechanical" with an empty rationale."""
    from agent6.paths import state_dir

    origin_state = state_dir(origin)
    cfg = Config()  # no reviewer -> mechanical ranking
    lanes = _specs(tmp_path, cfg, "fan", "2")
    # Lane 2 passes verify, lane 1 finishes without it -> lane 2 wins (rank 1)
    # mechanically. The loser is a deliberate finish, not a "failed" run: a
    # failure is imported but never ranked, so it would carry no stamp at all.
    spawner = _FakeSpawner(
        origin, origin_state, tmp_path / "lane-state", status_by_lane={1: "finished", 2: "passed"}
    )

    run_parallel(
        "t", lanes, cfg=cfg, origin=origin, origin_state=origin_state,
        runtime=runtime, spawner=spawner, fanout_id="fan",
    )  # fmt: skip

    m1 = json.loads(
        (origin_state / "sessions" / "runs" / "fan-l1" / "manifest.json").read_text("utf-8")
    )
    m2 = json.loads(
        (origin_state / "sessions" / "runs" / "fan-l2" / "manifest.json").read_text("utf-8")
    )
    assert m2["compare"] == {
        "rank": 1, "of": 2, "winner": True,
        "ranked_by": "mechanical", "rationale": "", "judge_cost_usd": 0.0,
        "judge_cost_partial": False,
    }  # fmt: skip
    assert m1["compare"] == {
        "rank": 2, "of": 2, "winner": False,
        "ranked_by": "mechanical", "rationale": "", "judge_cost_usd": 0.0,
        "judge_cost_partial": False,
    }  # fmt: skip
    # The lineage stamp is untouched by the compare stamp (shared rewrite merges).
    assert m1["parallel_id"] == "fan" and m1["lane"] == 1


def test_compare_stamp_records_judge_rationale_truncated(
    origin: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, runtime: LaneRuntime
) -> None:
    """When the judge ranks (not the mechanical fallback), every lane records
    ranked_by="judge", the SAME rationale, truncated to bound the manifest, and
    the SAME group judge cost."""
    from agent6.paths import state_dir

    origin_state = state_dir(origin)
    cfg = Config()
    lanes = _specs(tmp_path, cfg, "fan", "2")
    spawner = _FakeSpawner(origin, origin_state, tmp_path / "lane-state")
    long_rationale = "x" * 3000

    def fake_rank(*_a: object, **_k: object) -> RankOutcome:
        return RankOutcome(("fan-l1", "fan-l2"), long_rationale, "judge", judge_cost_usd=0.0123)

    monkeypatch.setattr(parallel, "rank", fake_rank)

    run_parallel(
        "t", lanes, cfg=cfg, origin=origin, origin_state=origin_state,
        runtime=runtime, spawner=spawner, fanout_id="fan",
    )  # fmt: skip

    m1 = json.loads(
        (origin_state / "sessions" / "runs" / "fan-l1" / "manifest.json").read_text("utf-8")
    )
    m2 = json.loads(
        (origin_state / "sessions" / "runs" / "fan-l2" / "manifest.json").read_text("utf-8")
    )
    assert m1["compare"]["ranked_by"] == "judge" and m1["compare"]["winner"] is True
    assert m1["compare"]["rank"] == 1 and m2["compare"]["rank"] == 2
    assert len(m1["compare"]["rationale"]) == 2000  # truncated ~2000
    assert m2["compare"]["rationale"] == m1["compare"]["rationale"]  # same group rationale
    # The judge call's group cost lands on every lane, same figure (never summed).
    assert m1["compare"]["judge_cost_usd"] == m2["compare"]["judge_cost_usd"] == 0.0123


def test_run_parallel_removes_its_emptied_workdir_levels(
    origin: Path, tmp_path: Path, runtime: LaneRuntime
) -> None:
    """`run --parallel` clones sit at `<base>/<repo>/<fan-out>/lane-N`: the
    fan-out dir and the per-repo dir go once empty, the base stays."""
    from agent6.paths import state_dir

    origin_state = state_dir(origin)
    cfg = Config()
    per_repo = parallel.workdir_base(cfg, origin)
    lanes = build_lane_specs(
        "2", cfg=cfg, origin=origin, fanout_id="fan", workdir_root=per_repo / "fan"
    )
    spawner = _FakeSpawner(origin, origin_state, tmp_path / "lane-state")

    rc = run_parallel(
        "t",
        lanes,
        cfg=cfg,
        origin=origin,
        origin_state=origin_state,
        runtime=runtime,
        spawner=spawner,
        fanout_id="fan",
    )

    assert rc == 0
    assert not (per_repo / "fan").exists()
    assert not per_repo.exists()
    assert per_repo.parent.is_dir()


def test_run_parallel_symlink_appears_before_import(
    origin: Path, tmp_path: Path, runtime: LaneRuntime
) -> None:
    from agent6.paths import state_dir

    origin_state = state_dir(origin)
    cfg = Config()
    lanes = _specs(tmp_path, cfg, "fan", "2")
    spawner = _FakeSpawner(origin, origin_state, tmp_path / "lane-state")

    run_parallel(
        "t",
        lanes,
        cfg=cfg,
        origin=origin,
        origin_state=origin_state,
        runtime=runtime,
        spawner=spawner,
        fanout_id="fan",
    )

    # While lane 2 spawned, lane 1 was already visible as a live symlink...
    assert spawner.prior_link_was_symlink[2] is True
    # ...and after completion every lane is a real dir, no symlink left behind.
    for i in (1, 2):
        link = origin_state / "sessions" / "runs" / f"fan-l{i}"
        assert link.is_dir() and not link.is_symlink()


def test_failed_lane_does_not_stop_others(
    origin: Path, tmp_path: Path, runtime: LaneRuntime
) -> None:
    from agent6.paths import state_dir

    origin_state = state_dir(origin)
    cfg = Config()
    lanes = _specs(tmp_path, cfg, "fan", "3")
    spawner = _FakeSpawner(origin, origin_state, tmp_path / "lane-state", fail={2})

    rc = run_parallel(
        "t",
        lanes,
        cfg=cfg,
        origin=origin,
        origin_state=origin_state,
        runtime=runtime,
        spawner=spawner,
        fanout_id="fan",
    )

    assert rc == 0  # lanes 1 and 3 still produced candidates
    assert branch_exists(origin, "agent6/fan-l1")
    assert branch_exists(origin, "agent6/fan-l3")
    assert not branch_exists(origin, "agent6/fan-l2")
    assert (origin_state / "sessions" / "runs" / "fan-l1").is_dir()
    assert not (origin_state / "sessions" / "runs" / "fan-l2").exists()


def test_report_ranks_passing_lane_first(
    origin: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str], runtime: LaneRuntime
) -> None:
    from agent6.paths import state_dir

    origin_state = state_dir(origin)
    cfg = Config()  # no reviewer model -> mechanical ranking
    lanes = _specs(tmp_path, cfg, "fan", "2")
    # Lane 1 fails verify but is cheaper; lane 2 passes. Verify-pass wins.
    spawner = _FakeSpawner(
        origin,
        origin_state,
        tmp_path / "lane-state",
        status_by_lane={1: "failed", 2: "passed"},
        cost_by_lane={1: 0.01, 2: 0.09},
    )

    run_parallel(
        "t",
        lanes,
        cfg=cfg,
        origin=origin,
        origin_state=origin_state,
        runtime=runtime,
        spawner=spawner,
        fanout_id="fan",
    )

    out = capsys.readouterr().out
    assert "ranked candidates" in out
    # The passing lane ranks first despite costing more.
    assert out.index("fan-l2") < out.index("fan-l1")
    assert "agent6 sessions merge fan-l2" in out


def test_run_parallel_all_failed_returns_1(
    origin: Path, tmp_path: Path, runtime: LaneRuntime
) -> None:
    from agent6.paths import state_dir

    origin_state = state_dir(origin)
    cfg = Config()
    lanes = _specs(tmp_path, cfg, "fan", "2")
    spawner = _FakeSpawner(origin, origin_state, tmp_path / "lane-state", fail={1, 2})

    rc = run_parallel(
        "t",
        lanes,
        cfg=cfg,
        origin=origin,
        origin_state=origin_state,
        runtime=runtime,
        spawner=spawner,
        fanout_id="fan",
    )
    assert rc == 1


def test_lineage_stamp_oserror_does_not_abort_import_loop(
    origin: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    runtime: LaneRuntime,
) -> None:
    """An atomic_write OSError while stamping lineage (disk full / read-only
    mount) must not abort the import loop mid-way: each lane's import stands, the
    degradation prints, and the remaining lanes still import + report."""
    from agent6.paths import state_dir

    origin_state = state_dir(origin)
    cfg = Config()
    lanes = _specs(tmp_path, cfg, "fan", "2")
    spawner = _FakeSpawner(origin, origin_state, tmp_path / "lane-state")

    def boom(_path: Path, _m: object) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(parallel, "write_manifest", boom)

    rc = run_parallel(
        "t",
        lanes,
        cfg=cfg,
        origin=origin,
        origin_state=origin_state,
        runtime=runtime,
        spawner=spawner,
        fanout_id="fan",
    )

    assert rc == 0  # both lanes still imported despite the stamp failure
    assert branch_exists(origin, "agent6/fan-l1")
    assert branch_exists(origin, "agent6/fan-l2")
    assert (origin_state / "sessions" / "runs" / "fan-l1").is_dir()
    assert (origin_state / "sessions" / "runs" / "fan-l2").is_dir()
    err = capsys.readouterr().err
    assert "lineage" in err and "disk full" in err


def test_ctrl_c_during_spawn_loop_stops_imports_and_reports(
    origin: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    runtime: LaneRuntime,
) -> None:
    """A KeyboardInterrupt while still spawning (before the await) routes into the
    same stop-grace + import-what-exists + report path: the already-started lane
    is imported, the run exits 130, and lanes never spawned are simply absent."""
    from agent6.paths import state_dir

    origin_state = state_dir(origin)
    cfg = Config()
    lanes = _specs(tmp_path, cfg, "fan", "3")
    base = _FakeSpawner(origin, origin_state, tmp_path / "lane-state")

    def interrupting_spawner(spec: LaneSpec, task: str) -> LaneResult:
        if spec.lane == 2:  # interrupt AFTER lane 1 has started
            raise KeyboardInterrupt
        return base(spec, task)

    monkeypatch.setattr(lane_watch, "POLL_INTERVAL_S", 0.01)

    rc = run_parallel(
        "t",
        lanes,
        cfg=cfg,
        origin=origin,
        origin_state=origin_state,
        runtime=runtime,
        spawner=interrupting_spawner,
        fanout_id="fan",
    )

    assert rc == 130
    # Lane 1 (started before the interrupt) was stopped + imported...
    assert branch_exists(origin, "agent6/fan-l1")
    assert (origin_state / "sessions" / "runs" / "fan-l1").is_dir()
    # ...lanes 2 and 3 never produced a candidate.
    assert not branch_exists(origin, "agent6/fan-l2")
    assert not branch_exists(origin, "agent6/fan-l3")
    assert "interrupted; stopping lanes" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# Teardown race + cleanup safety
# ---------------------------------------------------------------------------


def test_await_waits_for_worker_pid_to_clear(
    origin: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    runtime: LaneRuntime,
) -> None:
    """session.end lands in logs.jsonl BEFORE the lane's teardown clears worker.pid.
    The await gate must keep waiting through that window (terminal = non-running
    status AND pid cleared/dead); importing inside it would misread the lane as
    still running and cleanup would destroy its only copy."""
    from agent6.paths import state_dir

    origin_state = state_dir(origin)
    cfg = Config()
    lanes = _specs(tmp_path, cfg, "fan", "1")
    spawner = _FakeSpawner(origin, origin_state, tmp_path / "lane-state", pid_lanes={1})

    # Clear the live pid only on the SECOND status poll of the lane's run dir,
    # so poll 1 exercises the race window deterministically.
    real_summarize = parallel.summarize_session_dir
    polls = {"n": 0}

    def summarize_then_clear_pid(session_dir: Path) -> object:
        summary = real_summarize(session_dir)
        if session_dir.name == "fan-l1":
            polls["n"] += 1
            if polls["n"] >= 2:
                (session_dir / "worker.pid").unlink(missing_ok=True)
        return summary

    monkeypatch.setattr(parallel, "summarize_session_dir", summarize_then_clear_pid)
    monkeypatch.setattr(lane_watch, "summarize_session_dir", summarize_then_clear_pid)
    monkeypatch.setattr(lane_watch, "POLL_INTERVAL_S", 0.01)

    rc = run_parallel(
        "t",
        lanes,
        cfg=cfg,
        origin=origin,
        origin_state=origin_state,
        runtime=runtime,
        spawner=spawner,
        fanout_id="fan",
    )

    assert rc == 0
    assert polls["n"] >= 2  # the gate really held through the live-pid poll
    assert branch_exists(origin, "agent6/fan-l1")
    imported = origin_state / "sessions" / "runs" / "fan-l1"
    assert imported.is_dir() and not imported.is_symlink()
    assert "failed lanes" not in capsys.readouterr().out


def test_cleanup_preserves_unimported_lane(
    origin: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str], runtime: LaneRuntime
) -> None:
    """A lane whose import is refused keeps its clone, run state, and live
    symlink (the clone holds the only copy of its branch), and the report names
    what was kept. Imported lanes are still cleaned up."""
    from agent6.paths import state_dir

    origin_state = state_dir(origin)
    cfg = Config()
    lanes = _specs(tmp_path, cfg, "fan", "2")
    # Pre-existing branch in the origin makes lane 1's import refuse.
    create_branch(origin, "agent6/fan-l1")
    _git(origin, "checkout", "main")
    spawner = _FakeSpawner(origin, origin_state, tmp_path / "lane-state")

    rc = run_parallel(
        "t",
        lanes,
        cfg=cfg,
        origin=origin,
        origin_state=origin_state,
        runtime=runtime,
        spawner=spawner,
        fanout_id="fan",
    )

    assert rc == 0  # lane 2 still imported
    # Lane 1 kept: clone (with its branch), fabricated run state, live symlink.
    assert lanes[0].workdir.is_dir()
    assert branch_exists(lanes[0].workdir, "agent6/fan-l1")
    assert (tmp_path / "lane-state" / "lane1" / "sessions" / "runs" / "fan-l1").is_dir()
    assert (origin_state / "sessions" / "runs" / "fan-l1").is_symlink()
    # Lane 2 imported and cleaned: real dir in origin state, clone gone.
    assert (origin_state / "sessions" / "runs" / "fan-l2").is_dir()
    assert not (origin_state / "sessions" / "runs" / "fan-l2").is_symlink()
    assert not lanes[1].workdir.exists()
    # The report names the kept clone so the operator can act on it.
    out = capsys.readouterr().out
    assert str(lanes[0].workdir) in out


def test_await_uses_real_run_dir_not_symlink(
    origin: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    runtime: LaneRuntime,
) -> None:
    """The symlink is a view for the hub, not the source of truth: with symlink
    creation failing entirely, the lane is still awaited on its REAL run dir
    (its true status is observed, not '?') and imported."""
    from agent6.paths import state_dir

    origin_state = state_dir(origin)
    cfg = Config()
    lanes = _specs(tmp_path, cfg, "fan", "1")
    spawner = _FakeSpawner(origin, origin_state, tmp_path / "lane-state")

    def _no_symlink(*_a: object, **_k: object) -> None:
        return None

    monkeypatch.setattr(parallel, "symlink_lane", _no_symlink)

    rc = run_parallel(
        "t",
        lanes,
        cfg=cfg,
        origin=origin,
        origin_state=origin_state,
        runtime=runtime,
        spawner=spawner,
        fanout_id="fan",
    )

    assert rc == 0
    assert branch_exists(origin, "agent6/fan-l1")
    assert (origin_state / "sessions" / "runs" / "fan-l1").is_dir()
    # The lane's real terminal status was observed (not the missing-link "?").
    assert "lane 1 [fan-l1]: passed" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# Coordinator plumbing: run_lane_to_completion + the group spawner
# ---------------------------------------------------------------------------


def test_run_lane_to_completion_imports_and_stamps(
    origin: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    runtime: LaneRuntime,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """One lane fully: spawn (fake), symlink it live into the origin's runs/,
    await to terminal, import its branch + run dir into the origin, and stamp
    `<group>` lineage. The live symlink is visible while the lane runs (so a hub
    can see + answer it) and is replaced by the real dir after import."""
    from agent6.paths import state_dir

    origin_state = state_dir(origin)
    cfg = Config()
    spawner = _FakeSpawner(origin, origin_state, tmp_path / "lane-state", fanout_id="p1")
    spec = LaneSpec(
        lane=1, session_id="co-p1-l1", workdir=tmp_path / "work" / "co-p1-l1", model=None
    )
    memory.add(state_dir(spec.workdir), "lane-fact", "The flaky test is test_clock.")

    # The await polls summarize_session_dir; observe the origin link state then -- it
    # must be a live symlink while the lane is still running.
    link = origin_state / "sessions" / "runs" / "co-p1-l1"
    real_summarize = parallel.summarize_session_dir
    seen: dict[str, bool] = {}

    def observe(session_dir: Path) -> object:
        seen.setdefault("symlink_during_life", link.is_symlink())
        return real_summarize(session_dir)

    monkeypatch.setattr(parallel, "summarize_session_dir", observe)
    monkeypatch.setattr(lane_watch, "summarize_session_dir", observe)

    res = parallel.run_lane_to_completion(
        spec,
        "do it",
        cfg=cfg,
        origin=origin,
        origin_state=origin_state,
        group="p1",
        runtime=runtime,
        spawner=spawner,
        poll_interval_s=0.01,
    )

    assert res.ok
    assert seen["symlink_during_life"] is True  # a hub could see + answer the lane
    assert branch_exists(origin, "agent6/co-p1-l1")
    # The lane's memory came back before its state dir went (the coordinator
    # path cleaned up with no carry at all).
    assert "The flaky test is test_clock." in memory.show(origin_state, "lane-fact")
    assert "lane 1: memory 1 carried" in capsys.readouterr().err
    imported = origin_state / "sessions" / "runs" / "co-p1-l1"
    assert imported.is_dir() and not imported.is_symlink()  # replaced by the real dir
    assert res.session_dir == imported
    manifest = json.loads((imported / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["parallel_id"] == "p1"
    assert manifest["lane"] == 1


def test_run_lane_to_completion_failed_spawn_imports_nothing(
    origin: Path, tmp_path: Path, runtime: LaneRuntime
) -> None:
    from agent6.paths import state_dir

    origin_state = state_dir(origin)
    cfg = Config()
    spawner = _FakeSpawner(origin, origin_state, tmp_path / "lane-state", fail={1})
    spec = LaneSpec(
        lane=1, session_id="co-p1-l1", workdir=tmp_path / "work" / "co-p1-l1", model=None
    )

    res = parallel.run_lane_to_completion(
        spec,
        "do it",
        cfg=cfg,
        origin=origin,
        origin_state=origin_state,
        group="p1",
        runtime=runtime,
        spawner=spawner,
    )

    assert not res.ok and res.error == "boom"
    assert not branch_exists(origin, "agent6/co-p1-l1")


def test_a_failed_lane_never_joins_the_coordinator(
    origin: Path, tmp_path: Path, runtime: LaneRuntime
) -> None:
    """`/parallel` dispatch gated only on died-without-end, so a lane that ended
    `provider_error` (folded "failed") came back ok=True: join_lane_result
    merged its half-done branch into the coordinator's checkout and told the
    model "joined at <sha>". Candidacy is ONE question (`produced_result`),
    the same one the fan-out asks."""
    from agent6.paths import state_dir

    origin_state = state_dir(origin)
    cfg = Config()
    spawner = _FakeSpawner(
        origin, origin_state, tmp_path / "lane-state", status_by_lane={1: "failed"}
    )
    spec = LaneSpec(
        lane=1, session_id="co-pf-l1", workdir=tmp_path / "work" / "co-pf-l1", model=None
    )

    res = parallel.run_lane_to_completion(
        spec,
        "do it",
        cfg=cfg,
        origin=origin,
        origin_state=origin_state,
        group="pf",
        runtime=runtime,
        spawner=spawner,
        poll_interval_s=0.01,
    )

    assert res.ok is False
    assert "failed" in res.error and "provider error" in res.error
    # The work is not lost: the branch was imported before the verdict.
    assert "agent6/co-pf-l1" in res.error
    assert branch_exists(origin, "agent6/co-pf-l1")


def test_a_crashed_lane_never_joins_the_coordinator(
    origin: Path, tmp_path: Path, runtime: LaneRuntime
) -> None:
    """The died-without-end half of the same gate (a lane with no session.end
    folds "stale"): imported, named in the error, never joined."""
    from agent6.paths import state_dir

    origin_state = state_dir(origin)
    cfg = Config()
    spawner = _FakeSpawner(
        origin, origin_state, tmp_path / "lane-state", status_by_lane={1: "stale"}
    )
    spec = LaneSpec(
        lane=1, session_id="co-ps-l1", workdir=tmp_path / "work" / "co-ps-l1", model=None
    )

    res = parallel.run_lane_to_completion(
        spec,
        "do it",
        cfg=cfg,
        origin=origin,
        origin_state=origin_state,
        group="ps",
        runtime=runtime,
        spawner=spawner,
        poll_interval_s=0.01,
    )

    assert res.ok is False
    assert "stale" in res.error and "agent6/co-ps-l1" in res.error
    assert branch_exists(origin, "agent6/co-ps-l1")


def test_build_lane_spawner_over_cap_refused(
    origin: Path, monkeypatch: pytest.MonkeyPatch, runtime: LaneRuntime
) -> None:
    """[parallel].max_lanes is documented as a hard cap per fan-out, and a live
    /parallel steer is a fan-out: it must refuse over-cap BEFORE any clone or
    spawn, exactly as `run --parallel` does in build_lane_specs. The loop turns
    the raise into 'group dispatch failed' coordinator feedback."""
    from agent6.paths import state_dir

    cfg = Config.model_validate({"parallel": {"max_lanes": 2}})
    called: list[str] = []

    def fake_rltc(spec: LaneSpec, task: str, **kw: object) -> LaneResult:
        called.append(spec.session_id)
        return LaneResult(spec=spec, session_dir=spec.workdir, branch="b", ok=True, error="")

    monkeypatch.setattr(parallel, "run_lane_to_completion", fake_rltc)
    dispatch = parallel.build_lane_spawner(
        cfg, origin, state_dir(origin), coordinator_session_id="co", runtime=runtime
    )
    with pytest.raises(ParallelError, match="max_lanes"):
        dispatch([LaneTask(task="t", model=None)] * 3, "p1")
    assert called == []  # nothing cloned or spawned


def test_run_lane_to_completion_cleans_up_imported_clone(
    origin: Path, tmp_path: Path, runtime: LaneRuntime
) -> None:
    """The coordinator path must honor the module contract 'clones + lane state
    are torn down after import': every /parallel group otherwise leaked one full
    repo clone + state dir + lane config per lane, forever. Its clones sit at
    `<base>/<repo>/<coordinator>/<group>/lane-N`: every emptied level up to and
    including the per-repo dir goes, the base above it stays. Only the imported
    (ok=True) lane is cleaned; a failed import keeps its clone (it may hold the
    only copy of the branch)."""
    from agent6.paths import state_dir

    origin_state = state_dir(origin)
    cfg = Config()
    spawner = _FakeSpawner(origin, origin_state, tmp_path / "lane-state")
    per_repo = parallel.workdir_base(cfg, origin)
    spec = LaneSpec(
        lane=1, session_id="co-p9-l1", workdir=per_repo / "co" / "grp" / "lane-1", model=None
    )
    res = parallel.run_lane_to_completion(
        spec,
        "do it",
        cfg=cfg,
        origin=origin,
        origin_state=origin_state,
        group="p9",
        runtime=runtime,
        spawner=spawner,
        poll_interval_s=0.01,
    )
    assert res.ok
    assert not spec.workdir.exists()  # the clone is gone
    assert not (per_repo / "co" / "grp").exists()  # the emptied group dir is rmdir'd
    assert not (per_repo / "co").exists()  # the coordinator's dir above it
    assert not per_repo.exists()  # and the per-repo dir
    assert per_repo.parent.is_dir()  # never the base

    # Import refused (branch already exists): the lane keeps its clone.
    create_branch(origin, "agent6/co-p8-l1")
    create_branch(origin, "main")
    spawner2 = _FakeSpawner(origin, origin_state, tmp_path / "lane-state-2")
    spec2 = LaneSpec(
        lane=1, session_id="co-p8-l1", workdir=tmp_path / "work" / "grp8" / "lane-1", model=None
    )
    res2 = parallel.run_lane_to_completion(
        spec2,
        "do it",
        cfg=cfg,
        origin=origin,
        origin_state=origin_state,
        group="p8",
        runtime=runtime,
        spawner=spawner2,
        poll_interval_s=0.01,
    )
    assert not res2.ok
    assert spec2.workdir.exists()  # unimported lane retains its workspace


def test_build_lane_spawner_builds_specs_and_preserves_order(
    origin: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, runtime: LaneRuntime
) -> None:
    """The group dispatcher names lanes `<coord>-<group>-l<i>`, puts them under a
    per-group workdir, and returns results in dispatch order despite the pool."""
    from agent6.paths import state_dir

    origin_state = state_dir(origin)
    cfg = Config()
    seen: list[tuple[int, str, str, str, str]] = []

    def fake_rltc(spec: LaneSpec, task: str, **kw: object) -> LaneResult:
        seen.append((spec.lane, spec.session_id, task, str(kw["group"]), str(spec.workdir)))
        return LaneResult(
            spec=spec,
            session_dir=spec.workdir,
            branch=f"agent6/{spec.session_id}",
            ok=True,
            error="",
        )

    monkeypatch.setattr(parallel, "run_lane_to_completion", fake_rltc)
    dispatch = parallel.build_lane_spawner(
        cfg, origin, origin_state, coordinator_session_id="co", runtime=runtime
    )
    lanes = [LaneTask(task="task a", model="kimi"), LaneTask(task="task b", model=None)]
    results = dispatch(lanes, "p2")

    assert [r.spec.session_id for r in results] == ["co-p2-l1", "co-p2-l2"]
    assert [r.spec.model for r in results] == ["kimi", None]  # per-lane model threaded through
    assert sorted(s[0] for s in seen) == [1, 2]  # every lane ran once
    # One name for the group: `adopt_orphan_lane` and the clone sweep both
    # derive a lane's clone from the lineage its manifest recorded, so the dir
    # has to be exactly that. A bare "p2" put the clones a level deeper, and an
    # orphaned lane was never adopted.
    assert all(group == "co-p2" for (_l, _r, _t, group, _w) in seen)
    assert all(
        str(parallel.subordinate_workdir_root(cfg, origin, group) / f"lane-{lane}") == workdir
        for (lane, _r, _t, group, workdir) in seen
    )


def test_build_lane_spawner_forwards_auto_approve(
    origin: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, runtime: LaneRuntime
) -> None:
    from agent6.paths import state_dir

    origin_state = state_dir(origin)
    cfg = Config()
    seen: list[object] = []

    def fake_rltc(spec: LaneSpec, task: str, **kw: object) -> LaneResult:
        seen.append(kw["auto_approve"])
        return LaneResult(spec=spec, session_dir=spec.workdir, branch="agent6/x", ok=True, error="")

    monkeypatch.setattr(parallel, "run_lane_to_completion", fake_rltc)
    dispatch = parallel.build_lane_spawner(
        cfg, origin, origin_state, coordinator_session_id="co", runtime=runtime, auto_approve=True
    )
    dispatch([LaneTask(task="task a", model=None)], "p3")

    assert seen == [True]


def test_build_coordinator_spawner_forwards_auto_approve(
    origin: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, runtime: LaneRuntime
) -> None:
    """A coordinator started with --auto-approve dispatches lanes that inherit
    it; one started without does not (build_coordinator_spawner -> build_lane_
    spawner, same param as max_usd)."""
    origin_state = tmp_path / "ostate"
    origin_state.mkdir()
    cfg = Config()
    captured: list[object] = []

    def fake_build_lane_spawner(*_a: object, **kw: object) -> object:
        captured.append(kw.get("auto_approve"))
        return "dispatcher"

    monkeypatch.setattr(parallel, "build_lane_spawner", fake_build_lane_spawner)

    parallel.build_coordinator_spawner(
        cfg, origin, origin_state, mode="run", session_id="co", runtime=runtime, auto_approve=True
    )
    parallel.build_coordinator_spawner(
        cfg, origin, origin_state, mode="run", session_id="co", runtime=runtime
    )

    assert captured == [True, False]


# ---------------------------------------------------------------------------
# Interrupting a lane await (coordinator stop channel / group teardown)
# ---------------------------------------------------------------------------


def test_await_lane_returns_when_should_stop_fires(tmp_path: Path, runtime: LaneRuntime) -> None:
    """The single-lane await honors should_stop: without it, the poll loop
    blocked until the lane ended on its own, so a coordinator stop (or Ctrl-C
    teardown) sat on a lane that might run for hours."""
    import os as _os

    lane_dir = tmp_path / "lane-run"
    lane_dir.mkdir()
    _write_fake_run(lane_dir / "sub", "t", status="running", cost=0.0)
    from agent6.sessions.ipc import write_worker_pid

    session_dir = lane_dir / "sub"
    write_worker_pid(session_dir, _os.getpid())  # a live lane: never terminal
    spec = LaneSpec(lane=1, session_id="co-p1-l1", workdir=tmp_path / "w", model=None)
    res = LaneResult(spec=spec, session_dir=session_dir, branch="agent6/x", ok=True, error="")
    calls = {"n": 0}

    def stop_after_two() -> bool:
        calls["n"] += 1
        return calls["n"] >= 2

    assert lane_watch.await_lane(res, poll_interval_s=0.01, should_stop=stop_after_two) is False


def test_run_lane_to_completion_interrupted_stops_lane_and_skips_import(
    origin: Path, tmp_path: Path, runtime: LaneRuntime
) -> None:
    """An interrupted await requests a clean stop on the lane and returns
    ok=False WITHOUT importing (the lane keeps running detached); with
    hard_stop set the bounded grace is skipped so teardown is prompt."""
    import os as _os
    import threading as _threading

    from agent6.sessions.ipc import stop_request_pending, write_worker_pid

    lane_dir = tmp_path / "lane-run" / "sub"
    _write_fake_run(lane_dir, "t", status="running", cost=0.0)
    write_worker_pid(lane_dir, _os.getpid())  # live forever from the test's view
    spec = LaneSpec(lane=1, session_id="co-p1-l1", workdir=tmp_path / "w", model=None)

    def fake_spawner(spec: LaneSpec, task: str) -> LaneResult:
        return LaneResult(spec=spec, session_dir=lane_dir, branch="agent6/x", ok=True, error="")

    hard_stop = _threading.Event()
    hard_stop.set()
    res = parallel.run_lane_to_completion(
        spec,
        "do it",
        cfg=Config(),
        origin=origin,
        origin_state=tmp_path / "ostate",
        group="p1",
        runtime=runtime,
        spawner=fake_spawner,
        poll_interval_s=0.01,
        should_stop=hard_stop.is_set,
        hard_stop=hard_stop,
    )
    assert res.ok is False
    assert "interrupted" in res.error
    assert stop_request_pending(lane_dir) is True  # the lane was asked to stop
    assert lane_dir.is_dir()  # nothing was moved/imported


def test_crashed_lane_is_not_a_rankable_candidate(
    origin: Path, tmp_path: Path, runtime: LaneRuntime, capsys: pytest.CaptureFixture[str]
) -> None:
    """A lane that died without a session.end folds to "stale", which verify_ok maps
    to None -- the same tri-state as a clean unverified finish. Mechanical
    ranking then sorts by cost, so the cheapest (earliest-crashing) lane ranked
    first and was stamped compare.winner=true, wearing the winner glyph in every
    listing while the report called it "no-verify"."""
    from agent6.paths import state_dir

    origin_state = state_dir(origin)
    cfg = Config()
    lanes = _specs(tmp_path, cfg, "crsh", "2")
    spawner = _FakeSpawner(
        origin,
        origin_state,
        tmp_path / "lane-state",
        status_by_lane={1: "stale", 2: "finished"},
        cost_by_lane={1: 0.001, 2: 2.0},
    )

    rc = run_parallel(
        "do the task",
        lanes,
        cfg=cfg,
        origin=origin,
        origin_state=origin_state,
        runtime=runtime,
        spawner=spawner,
        fanout_id="crsh",
    )
    assert rc == 0
    out = "".join(capsys.readouterr())
    # The crashed lane never wins, and the surface names the missing result.
    crashed = json.loads(
        (origin_state / "sessions" / "runs" / "crsh-l1" / "manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert (crashed.get("compare") or {}).get("winner") is not True
    survivor = json.loads(
        (origin_state / "sessions" / "runs" / "crsh-l2" / "manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert survivor["compare"]["winner"] is True
    assert "crsh-l1" in out and "no result (stale)" in out


def test_lane_config_forces_a_run_branch(tmp_path: Path) -> None:
    """A lane's branch is how its work is imported (bridge_spawner fetches
    agent6/<session_id>), but the origin's [git].branch_per_run=false materialized
    into the lane config, so the branch was never cut: every lane completed and
    billed, then failed at import with a raw git 'couldn't find remote ref'."""
    import tomllib

    from agent6.app.parallel import _write_lane_config  # pyright: ignore[reportPrivateUsage]

    base = Config()
    cfg = base.model_copy(update={"git": base.git.model_copy(update={"branch_per_run": False})})
    spec = LaneSpec(lane=1, session_id="fan-l1", workdir=tmp_path / "clone", model=None)
    path = _write_lane_config(cfg, spec)
    written = tomllib.loads(path.read_text(encoding="utf-8"))
    assert written["git"]["branch_per_run"] is True


def test_a_fanout_where_every_lane_failed_crowns_nobody(
    origin: Path, tmp_path: Path, runtime: LaneRuntime, capsys: pytest.CaptureFixture[str]
) -> None:
    """A failed lane is imported but never ranked.

    Only lanes that died WITHOUT a session.end were excluded, so a lane that
    ended `provider_error` sailed into the candidate set: the fan-out stamped one
    `compare.winner=true`, wore the star in every listing, printed a merge
    command for a branch with nothing on it, and exited 0. `failed` is reserved
    for a run that did not finish deliberately -- a deliberate finish over a red
    gate is `finished` and still ranks, which the sibling test above covers."""
    from agent6.paths import state_dir

    origin_state = state_dir(origin)
    cfg = Config()
    lanes = _specs(tmp_path, cfg, "allf", "2")
    spawner = _FakeSpawner(
        origin,
        origin_state,
        tmp_path / "lane-state",
        status_by_lane={1: "failed", 2: "failed"},
        cost_by_lane={1: 0.001, 2: 2.0},
    )

    rc = run_parallel(
        "do the task",
        lanes,
        cfg=cfg,
        origin=origin,
        origin_state=origin_state,
        runtime=runtime,
        spawner=spawner,
        fanout_id="allf",
    )
    # Nothing to merge, and the exit code says so.
    assert rc == 1
    out = "".join(capsys.readouterr())
    assert "merge with" not in out
    for lane_id in ("allf-l1", "allf-l2"):
        m = json.loads(
            (origin_state / "sessions" / "runs" / lane_id / "manifest.json").read_text(
                encoding="utf-8"
            )
        )
        assert (m.get("compare") or {}).get("winner") is not True
        # The work is not lost: the branch is named in the failure report.
        assert lane_id in out


def test_fanout_exit_reflects_the_gate_verdicts() -> None:
    """An all-red fan-out exited 0: every lane finished over a red gate, one
    was still crowned rank 1, and the fan-out read as success to every script.
    The exit now mirrors session_exit_code: 4 when gates ran and none passed,
    0 when some lane verified green or no lane had a gate, 1 for no candidates."""
    from agent6.app.parallel import fanout_exit_code
    from agent6.workflows.judge import CandidateBrief

    def _cand(verify_ok: bool | None) -> CandidateBrief:
        return CandidateBrief(session_id="s", task="t", diff="", verify_ok=verify_ok, cost_usd=0.0)

    assert fanout_exit_code([]) == 1
    assert fanout_exit_code([_cand(True), _cand(False)]) == 0
    assert fanout_exit_code([_cand(None), _cand(None)]) == 0  # gateless fan-out
    assert fanout_exit_code([_cand(False), _cand(False)]) == 4
    assert fanout_exit_code([_cand(False), _cand(None)]) == 4


def test_the_judge_is_capped_like_a_lane(tmp_path: Path) -> None:
    """The fan-out advertises "$X/lane x N + judge = $Y total", but the
    judge's tracker took the full config budget, so the effective ceiling
    quietly exceeded the printed one. rank caps the judge at the lane cap
    when one is given; the config budget stays the fallback."""
    from contextlib import contextmanager

    from agent6.app.compare import rank
    from agent6.budget import BudgetTracker
    from agent6.providers import ProviderError
    from agent6.workflows.judge import CandidateBrief

    seen: list[BudgetTracker] = []

    def build(cfg: Config, sink: object, budget: BudgetTracker) -> object:
        seen.append(budget)
        raise ProviderError("captured; fall back to mechanical")

    @contextmanager
    def status() -> Generator[None]:
        yield

    cfg = Config.model_validate(
        {
            "providers": {"p": {"api_format": "openai", "api_key_env": "K"}},
            "models": {
                "worker": {"provider": "p", "model": "m"},
                "reviewer": {"provider": "p", "model": "judge"},
            },
            "budget": {"max_usd": 10.0},
        }
    )
    briefs = [
        CandidateBrief(session_id=f"s{i}", task="t", diff="", verify_ok=True, cost_usd=0.0)
        for i in range(2)
    ]
    outcome = rank(
        cfg,
        briefs,
        transcript_dir=tmp_path,
        build_provider=build,  # pyright: ignore[reportArgumentType]
        judging_status=status,
        max_usd=0.25,
    )
    assert outcome.ranked_by == "mechanical"
    assert [b.max_usd for b in seen] == [0.25]  # the lane cap, not the $10 config budget


def test_lane_is_self_describing_from_birth(
    origin: Path, tmp_path: Path, runtime: LaneRuntime, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The spawn env carries the fan-out lineage and the manifest's one writer
    records it, so the grouping survives a coordinator death -- the old
    post-import stamp existed only while the coordinator lived, leaving
    orphaned lanes listed as unrelated runs."""
    from agent6.app.manifest import write_session_manifest
    from agent6.sessions.layout import SessionLayout
    from agent6.sessions.manifest import read_manifest

    captured: list[dict[str, str]] = []

    def fake_spawn(argv: list[str], workdir: Path, **kw: object) -> tuple[Path, str]:
        env = kw.get("env")
        assert isinstance(env, dict)
        captured.append(env)
        return workdir, ""

    spec = LaneSpec(lane=2, session_id="fan-l2", workdir=tmp_path / "work" / "lane-2", model=None)
    parallel.bridge_spawner(
        spec, "do it", cfg=Config(), origin=origin, max_usd=None,
        fanout_id="fan", runtime=replace(runtime, spawn=fake_spawn),
    )  # fmt: skip
    assert captured[-1]["AGENT6_PARALLEL_LINEAGE"] == "fan:2"

    # The writer records it from the env, exactly as the spawned lane would.
    monkeypatch.setenv("AGENT6_PARALLEL_LINEAGE", "fan:2")
    layout = SessionLayout(state_dir=tmp_path / "state", session_id="fan-l2")
    layout.ensure()
    write_session_manifest(
        layout,
        session_id="fan-l2",
        user_task="do it",
        base_sha="s",
        base_branch="main",
        run_branch="agent6/fan-l2",
        cfg=Config(),
    )
    m = read_manifest(layout.session_dir)
    assert m.parallel_id == "fan"
    assert m.lane == 2

    # An ordinary run records no lineage.
    monkeypatch.delenv("AGENT6_PARALLEL_LINEAGE")
    layout2 = SessionLayout(state_dir=tmp_path / "state", session_id="plain-run")
    layout2.ensure()
    write_session_manifest(
        layout2,
        session_id="plain-run",
        user_task="t",
        base_sha="s",
        base_branch="main",
        run_branch=None,
        cfg=Config(),
    )
    assert read_manifest(layout2.session_dir).parallel_id is None


def test_sweep_keeps_a_clone_holding_unmerged_commits(
    origin: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The sweep is content-safe by commit proof: a fan-out dir whose lane
    clone holds any commit the origin lacks is kept whole (the clone may be
    the only copy); one whose every lane tip the origin holds is deleted.
    The scan is scoped to this repo's `<repo-id>` subdir: another repo's
    clones can neither be swept by a commit proof made against the wrong
    origin nor kept forever with advice that cannot apply here."""
    from agent6.app.parallel import sweep_fanout_clones
    from agent6.paths import repo_id

    workdir = tmp_path / "cache"
    cfg = Config.model_validate({"parallel": {"workdir": str(workdir)}})
    scoped = workdir / repo_id(origin)

    unmerged = scoped / "fan-a" / "lane-1"
    unmerged.parent.mkdir(parents=True)
    clone_workspace(origin, unmerged)
    create_branch(unmerged, "agent6/fan-a-l1")
    (unmerged / "x.txt").write_text("x\n", encoding="utf-8")
    commit_all(unmerged, "lane work")

    merged = scoped / "fan-b" / "lane-1"
    merged.parent.mkdir(parents=True)
    clone_workspace(origin, merged)
    create_branch(merged, "agent6/fan-b-l1")  # tip == origin HEAD: nothing unique

    foreign = workdir / "other-repo-0" / "fan-c" / "lane-1"
    foreign.mkdir(parents=True)
    clone_workspace(origin, foreign)
    create_branch(foreign, "agent6/fan-c-l1")

    swept, kept = sweep_fanout_clones(origin, cfg)
    assert (swept, kept) == (1, 1)  # the foreign repo's dir is not counted
    assert (scoped / "fan-a").is_dir()  # unique commits: kept whole
    assert not (scoped / "fan-b").exists()
    assert (foreign / ".git").exists()  # out of scope: untouched


def test_sweep_keeps_a_clone_whose_tip_the_origin_cannot_reach(
    origin: Path, tmp_path: Path
) -> None:
    """The proof was `rev-parse <sha>^{commit}` in the origin, which succeeds
    for a loose object no ref reaches -- exactly what the operator is left with
    after the `git branch -D` prune's own message tells them to run. The clone
    was then deleted and the work went with the next `git gc`."""
    import subprocess as sp

    from agent6.app.parallel import sweep_fanout_clones
    from agent6.paths import repo_id

    workdir = tmp_path / "cache"
    cfg = Config.model_validate({"parallel": {"workdir": str(workdir)}})
    scoped = workdir / repo_id(origin)
    lane = scoped / "fan-x" / "lane-1"
    lane.parent.mkdir(parents=True)
    clone_workspace(origin, lane)
    create_branch(lane, "agent6/fan-x-l1")
    (lane / "x.txt").write_text("lane work\n", encoding="utf-8")
    commit_all(lane, "lane work")
    tip = sp.run(
        ["git", "-C", str(lane), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    # The origin has the OBJECT (fetched) but no ref reaching it.
    sp.run(["git", "-C", str(origin), "fetch", str(lane), tip], check=True, capture_output=True)

    swept, kept = sweep_fanout_clones(origin, cfg)

    assert (swept, kept) == (0, 1)
    assert lane.is_dir(), "the last ref reaching the lane's work was deleted"


def test_sweep_leaves_a_dir_that_is_not_a_fan_out_group_alone(
    origin: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Only a group dir holding `lane-*` clones is agent6's to judge by commit
    proof. A directory the operator put under the workdir scope (a clone
    named otherwise, a plain dir with files) holds no lane clone, and the
    sweep read that as "nothing to prove" and deleted it."""
    from agent6.app.parallel import sweep_fanout_clones
    from agent6.paths import repo_id

    workdir = tmp_path / "cache"
    cfg = Config.model_validate({"parallel": {"workdir": str(workdir)}})
    scoped = workdir / repo_id(origin)
    theirs = scoped / "my-clone"
    theirs.parent.mkdir(parents=True)
    clone_workspace(origin, theirs)
    (theirs / "draft.txt").write_text("uncommitted\n", encoding="utf-8")
    notes = scoped / "notes"
    notes.mkdir()
    (notes / "todo.md").write_text("mine\n", encoding="utf-8")

    swept, kept = sweep_fanout_clones(origin, cfg)
    assert (swept, kept) == (0, 0)
    assert (theirs / "draft.txt").read_text(encoding="utf-8") == "uncommitted\n"
    assert (notes / "todo.md").read_text(encoding="utf-8") == "mine\n"


def test_carry_back_names_the_kept_dir_only_when_it_holds_something(
    origin: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A held-back deletion keeps nothing, so a note pointing at memory-held/
    named a directory that did not exist."""
    from agent6.paths import state_dir

    origin_state = state_dir(origin)
    memory.add(origin_state, "d-fact", "D first.")
    lane_state = tmp_path / "lane-state"
    memory.seed_store(origin_state, lane_state)
    memory.remove(lane_state, "d-fact")
    (memory.memory_dir(origin_state) / "d-fact.md").write_text("D per the origin.\n")
    dest = tmp_path / "dest"
    parallel.carry_back(lane_state, origin_state, dest, lane=2, reporter=STDIO_REPORTER)
    err = capsys.readouterr().err
    assert "lane 2: memory held back (changed here too, or the name is taken): d-fact" in err
    assert "kept at" not in err and not (dest / "memory-held").exists()
