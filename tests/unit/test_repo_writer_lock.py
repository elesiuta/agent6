# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""One live run-mode worker per checkout: the repo.lock + park-and-resume flow.

Two concurrent run-mode workers share one working tree; each auto-commit is a
`git add -A` on whatever HEAD points at, so whichever run's branch was checked
out last received BOTH runs' commits. The repo-scoped flock refuses the second
worker up front, and the refused submission is PARKED (the manifest saves the
verbatim task) so the typed prompt is never dropped.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from agent6.config import Config
from agent6.config.layer import resolved_state_dir
from agent6.sessions.layout import SessionLayout
from agent6.sessions.lock import (
    acquire_repo_writer,
    acquire_single_writer,
    release_single_writer,
    repo_writer_held,
    repo_writer_holder,
)
from agent6.sessions.manifest import read_manifest


def test_repo_writer_second_acquire_refused_and_holder_named(tmp_path: Path) -> None:
    fd = acquire_repo_writer(tmp_path, "run-A")
    assert fd is not None
    try:
        assert acquire_repo_writer(tmp_path, "run-B") is None
        assert repo_writer_holder(tmp_path) == "run-A"
        assert repo_writer_held(tmp_path) is True
    finally:
        release_single_writer(fd)
    # Released: the checkout is free again and the probe agrees.
    assert repo_writer_held(tmp_path) is False
    fd2 = acquire_repo_writer(tmp_path, "run-B")
    assert fd2 is not None
    release_single_writer(fd2)


def test_one_probe_does_not_read_another_probe_as_a_live_run(tmp_path: Path) -> None:
    """The probe took the EXCLUSIVE lock to ask whether anyone held it, so two
    at once (a web hub and a TUI, or two hub tabs) each reported the other as
    a run driving the checkout and refused a submission nothing was blocking."""
    import os
    import threading

    from agent6.portable import lock_shared_nonblocking

    lock_path = tmp_path / "repo.lock"
    created = acquire_repo_writer(tmp_path, "run-A")  # creates the file
    assert created is not None
    release_single_writer(created)  # the checkout is free: only a probe holds anything
    probing = threading.Event()
    done = threading.Event()

    def _hold_probe() -> None:
        fd = os.open(lock_path, os.O_RDWR)
        lock_shared_nonblocking(fd)
        probing.set()
        done.wait(5.0)
        release_single_writer(fd)

    thread = threading.Thread(target=_hold_probe)
    thread.start()
    try:
        assert probing.wait(5.0)
        assert repo_writer_held(tmp_path) is False, "a probe read as a live writer"
    finally:
        done.set()
        thread.join(5.0)


def test_repo_writer_probe_does_not_hold(tmp_path: Path) -> None:
    # The advisory probe must not itself keep the lock (it acquires + releases).
    assert repo_writer_held(tmp_path) is False  # no lock file yet
    fd = acquire_repo_writer(tmp_path, "run-A")
    assert fd is not None
    release_single_writer(fd)
    assert repo_writer_held(tmp_path) is False
    fd2 = acquire_repo_writer(tmp_path, "run-C")
    assert fd2 is not None
    release_single_writer(fd2)


def _init_repo(path: Path) -> None:
    subprocess.run(["git", "init", "-q", "-b", "main", str(path)], check=True)
    subprocess.run(["git", "-C", str(path), "config", "user.email", "t@t"], check=True)
    subprocess.run(["git", "-C", str(path), "config", "user.name", "t"], check=True)
    (path / "README.md").write_text("hi\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(path), "add", "-A"], check=True)
    subprocess.run(["git", "-C", str(path), "commit", "-q", "-m", "init"], check=True)


@pytest.fixture
def repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A tmp git repo with isolated state + a minimal runnable global config."""
    gdir = tmp_path / "cfg"
    gdir.mkdir()
    (gdir / "config.toml").write_text(
        '[providers.anthropic]\napi_format = "anthropic"\n'
        '[models.worker]\nprovider = "anthropic"\nmodel = "claude-x"\n',
        encoding="utf-8",
    )
    monkeypatch.setenv("AGENT6_CONFIG_HOME", str(gdir))
    monkeypatch.setenv("AGENT6_STATE_HOME", str(tmp_path / "state"))
    repo = tmp_path / "repo"
    _init_repo(repo)
    monkeypatch.chdir(repo)
    return repo


def _load_cfg() -> Config:
    from agent6.config.layer import load_effective

    return load_effective(Path.cwd(), None).config


def test_second_run_parks_with_the_verbatim_task(repo: Path) -> None:
    """While a live worker holds the checkout, a second `run` submission is
    refused, but the exact typed prompt is saved as a parked, resumable run —
    with no tree mutation (no stash, no branch cut)."""
    from agent6.app.run import run_task

    state = resolved_state_dir(repo)
    long_task = "fix the thing " + "x" * 5000  # > the 4000-char display cap
    holder_fd = acquire_repo_writer(state, "run-LIVE")
    try:
        rc = run_task(
            _load_cfg(),
            long_task,
            frontend=MagicMock(),
            session_id="run-PARKED",
            mode="run",
        )
    finally:
        release_single_writer(holder_fd)
    assert rc == 2
    layout = SessionLayout(state_dir=state, session_id="run-PARKED")
    m = read_manifest(layout.session_dir)
    assert m.parked_task == long_task  # verbatim, not the truncated display twin
    assert m.run_branch is None
    # No branch was cut and the tree is untouched.
    branches = subprocess.run(
        ["git", "-C", str(repo), "branch", "--list", "agent6/run-PARKED"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    assert branches.strip() == ""
    # The parked dir survives (it is a saved run, not a discardable husk) and
    # the listing tells the truth about it.
    from agent6.viewmodel.listing import summarize_session_dir

    row = summarize_session_dir(layout.session_dir)
    assert (row.status, row.reason) == ("parked", "checkout busy")


def test_resume_starts_a_parked_run_with_the_saved_task(
    repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`agent6 resume <parked-id>` delegates to a fresh run_task with the
    verbatim saved task under the same run id (releasing its own locks first,
    so the fresh start can take them)."""
    from agent6.app import resume as resume_mod
    from agent6.app.manifest import stamp_parked, write_session_manifest

    state = resolved_state_dir(repo)
    layout = SessionLayout(state_dir=state, session_id="run-PARKED2")
    layout.ensure()
    write_session_manifest(
        layout,
        session_id="run-PARKED2",
        user_task="do the saved thing",
        base_sha="",
        base_branch="main",
        run_branch=None,
        cfg=_load_cfg(),
        mode="run",
    )
    stamp_parked(layout.session_dir, task="do the saved thing", reason="checkout busy")
    called: dict[str, Any] = {}

    def fake_run_task(cfg: Config, task: str, **kw: Any) -> int:
        called["task"] = task
        called["session_id"] = kw.get("session_id")
        called["mode"] = kw.get("mode")
        return 0

    monkeypatch.setattr(resume_mod, "run_task", fake_run_task)
    rc = resume_mod.resume_task(None, "run-PARKED2", frontend=MagicMock(), force=False)
    assert rc == 0
    assert called == {"task": "do the saved thing", "session_id": "run-PARKED2", "mode": "run"}
    # The delegation released the run-dir lock before handing off, so a real
    # run_task can re-acquire it: prove the lock is free.
    from agent6.sessions.lock import acquire_single_writer

    fd = acquire_single_writer(layout.session_dir)
    assert fd is not None
    release_single_writer(fd)


def test_resume_refuses_while_another_run_drives_the_checkout(
    repo: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Resuming run B while run A's worker is live in the same checkout must
    refuse (a resumed worker drives the tree exactly like a fresh one)."""
    from agent6.app import resume as resume_mod

    state = resolved_state_dir(repo)
    layout = SessionLayout(state_dir=state, session_id="run-B")
    layout.ensure()
    layout.manifest_path.write_text(
        json.dumps(
            {
                "version": 2,
                "session_id": "run-B",
                "mode": "run",
                "base_sha": "",
                "base_branch": "main",
                "run_branch": "agent6/run-B",
                "user_task": "t",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    holder_fd = acquire_repo_writer(state, "run-A")
    try:
        rc = resume_mod.resume_task(None, "run-B", frontend=MagicMock(), force=False)
    finally:
        release_single_writer(holder_fd)
    assert rc == 2
    err = capsys.readouterr().err
    assert "run-A" in err and "checkout" in err


def test_hub_new_work_preflight_refuses_while_checkout_busy(
    repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A hub refuses a New Work `run` submission up front (naming the live
    run) instead of spawning a detached run that parks and times out the locate;
    plan submissions are read-only and spawn freely."""
    from agent6.ui import spawn

    def must_not_spawn(*a: object, **k: object) -> tuple[Path | None, str]:
        raise AssertionError("must not spawn")

    monkeypatch.setattr(spawn, "spawn_and_locate", must_not_spawn)
    state = resolved_state_dir(repo)
    holder_fd = acquire_repo_writer(state, "run-LIVE")
    try:
        session_dir, err = spawn.spawn_new_work(repo, "run", "another task")
    finally:
        release_single_writer(holder_fd)
    assert session_dir is None
    assert "run-LIVE" in err and "checkout" in err


def test_hub_new_work_fans_out_while_checkout_busy(
    repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A fan-out takes no writer lock (its lanes clone the checkout), and the
    plain-run refusal told the operator to /parallel instead, yet the same
    check refused the /parallel message before parsing it."""
    from agent6.ui import spawn

    spawned: list[str] = []

    def fake_spawn(
        cwd: Path, mode: str, task: str, *, preset: str, spec: str, config_path: object = None
    ) -> tuple[Path | None, str]:
        spawned.append(f"{spec}:{task}")
        return repo / "run-P", ""

    monkeypatch.setattr(spawn, "_spawn_run", fake_spawn)

    def no_refusal(cwd: Path, segments: object, config_path: object = None) -> None:
        return None

    monkeypatch.setattr(spawn, "directive_model_refusal", no_refusal)
    state = resolved_state_dir(repo)
    holder_fd = acquire_repo_writer(state, "run-LIVE")
    try:
        session_dir, err = spawn.spawn_new_work(repo, "run", "/parallel 2 another task")
    finally:
        release_single_writer(holder_fd)
    assert (session_dir, err) == (repo / "run-P", "")
    assert spawned == ["2:another task"]


def test_runs_show_reports_a_parked_run_as_parked(
    repo: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The refusal hands the operator a run id to resume, so `sessions show` on it has
    to lead with the same word the listing uses. A parked run is a saved,
    resumable submission -- never "unknown (no events yet)", which reads as a
    broken husk and hides the one action that starts it."""
    from agent6.app.run import run_task
    from agent6.ui.cli.sessions_show import _cmd_status  # pyright: ignore[reportPrivateUsage]

    state = resolved_state_dir(repo)
    holder_fd = acquire_repo_writer(state, "run-LIVE")
    try:
        rc = run_task(
            _load_cfg(),
            "add a retry to the fetch helper",
            frontend=MagicMock(),
            session_id="run-PARKED",
            mode="run",
        )
    finally:
        release_single_writer(holder_fd)
    assert rc == 2
    capsys.readouterr()  # drop the refusal message

    assert _cmd_status("run-PARKED") == 0
    out = capsys.readouterr().out
    assert "parked" in out
    assert "unknown" not in out


def test_parked_manifest_records_the_config_profile_not_the_sandbox_one(repo: Path) -> None:
    """The parked manifest's workflow.preset is what resume feeds back to
    load_effective; the park path stamped the SANDBOX preset there
    ('strict'/'hardened'/'none'), so `agent6 resume <parked-id>` died with
    "CONFIG ERROR: unknown preset 'strict'" on every sandboxed host."""
    from agent6.app.run import run_task
    from agent6.config.layer import load_effective

    state = resolved_state_dir(repo)
    holder_fd = acquire_repo_writer(state, "run-LIVE")
    try:
        rc = run_task(
            _load_cfg(),
            "do the thing",
            frontend=MagicMock(),
            session_id="run-PROF",
            mode="run",
        )
    finally:
        release_single_writer(holder_fd)
    assert rc == 2
    m = read_manifest(SessionLayout(state_dir=state, session_id="run-PROF").session_dir)
    assert m.workflow.preset == _load_cfg().preset  # the CONFIG preset ("")
    # The exact call resume makes with it must not blow up on a sandbox word.
    load_effective(repo, None, preset=m.workflow.preset)


def test_parked_resume_passes_the_steer_through_to_run_task(
    repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`resume --steer` on a PARKED run: the bridge files resume seeds are
    wiped by run_task's own stale-state clear, so the follow-up must ride the
    delegation (initial_steer) instead of dying on the floor."""
    from agent6.app import resume as resume_mod
    from agent6.app.manifest import stamp_parked, write_session_manifest

    state = resolved_state_dir(repo)
    layout = SessionLayout(state_dir=state, session_id="run-PSTEER")
    layout.ensure()
    write_session_manifest(
        layout,
        session_id="run-PSTEER",
        user_task="do the saved thing",
        base_sha="",
        base_branch="main",
        run_branch=None,
        cfg=_load_cfg(),
        mode="run",
    )
    stamp_parked(layout.session_dir, task="do the saved thing", reason="checkout busy")
    called: dict[str, Any] = {}

    def fake_run_task(cfg: Config, task: str, **kw: Any) -> int:
        called["initial_steer"] = kw.get("initial_steer")
        return 0

    monkeypatch.setattr(resume_mod, "run_task", fake_run_task)
    rc = resume_mod.resume_task(
        None, "run-PSTEER", frontend=MagicMock(), force=False, steer="also update the docs"
    )
    assert rc == 0
    assert called["initial_steer"] == "also update the docs"


def test_run_task_seeds_initial_steer_on_the_bridge(repo: Path) -> None:
    """run_task's initial_steer lands on the bridge before the loop starts, so
    its first boundary poll finds it."""
    from agent6.app.run import run_task
    from agent6.sessions.ipc import read_steer_answer, steer_request_pending

    state = resolved_state_dir(repo)
    holder_fd = acquire_repo_writer(state, "run-LIVE")
    try:
        rc = run_task(
            _load_cfg(),
            "do the thing",
            frontend=MagicMock(),
            session_id="run-STEERSEED",
            mode="run",
            initial_steer="focus on tests",
        )
    finally:
        release_single_writer(holder_fd)
    assert rc == 2  # parked (checkout busy) -- but the steer already landed
    d = SessionLayout(state_dir=state, session_id="run-STEERSEED").session_dir
    assert steer_request_pending(d)
    assert read_steer_answer(d) == "focus on tests"


def test_teardown_raise_still_releases_both_writer_locks(
    repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A raise inside run_task's teardown must still release both writer
    flocks. A CLI exit drops them with the process anyway, but the ACP
    front-end calls run_task IN-PROCESS and outlives the run, where a leaked
    flock refused every later run on the session/checkout until the server
    restarted."""
    from agent6.app import run as run_mod

    def boom(*a: Any, **kw: Any) -> None:
        raise RuntimeError("fail with both writer locks held")

    # The first call past BOTH lock acquisitions on the clean-tree path.
    monkeypatch.setattr(run_mod, "write_session_manifest", boom)
    frontend = MagicMock()
    frontend.close_console_view.side_effect = OSError("teardown raise")
    with pytest.raises(OSError, match="teardown raise"):
        run_mod.run_task(
            _load_cfg(), "do a thing", frontend=frontend, session_id="run-TD", mode="run"
        )
    # Both flocks are free for the next in-process run: the checkout's...
    state = resolved_state_dir(repo)
    fd = acquire_repo_writer(state, "run-NEXT")
    assert fd is not None
    release_single_writer(fd)
    # ...and the run dir's.
    fd2 = acquire_single_writer(SessionLayout(state_dir=state, session_id="run-TD").session_dir)
    assert fd2 is not None
    release_single_writer(fd2)


def test_resume_keeps_a_stop_request_pending_after_the_previous_leg_ended(
    repo: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """resume_task's sweep of the previous leg's bridge state dropped a stop
    marker written after that leg ended (an editor's cancel while this leg was
    coming up); a marker older than the journal's last line is the stale one."""
    import os

    from agent6.app import resume as resume_mod
    from agent6.sessions.ipc import request_stop, stop_request_pending

    state = resolved_state_dir(repo)
    layout = SessionLayout(state_dir=state, session_id="run-C")
    layout.ensure()
    layout.manifest_path.write_text(
        json.dumps(
            {
                "version": 2,
                "session_id": "run-C",
                "mode": "run",
                "base_sha": "",
                "base_branch": "main",
                "run_branch": "agent6/run-C",
                "user_task": "t",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    layout.logs_path.write_text(
        '{"type": "session.start", "mode": "run", "user_task": "t"}\n'
        '{"type": "session.end", "reason": "budget_exhausted", "all_passed": false}\n',
        encoding="utf-8",
    )
    leg_end = layout.logs_path.stat().st_mtime
    (layout.session_dir / "steer.request").write_text("", encoding="utf-8")
    os.utime(layout.session_dir / "steer.request", (leg_end - 10, leg_end - 10))
    request_stop(layout.session_dir)
    os.utime(layout.session_dir / "stop.request", (leg_end + 1, leg_end + 1))
    holder_fd = acquire_repo_writer(state, "run-A")
    try:
        rc = resume_mod.resume_task(None, "run-C", frontend=MagicMock(), force=False)
    finally:
        release_single_writer(holder_fd)
    assert rc == 2  # the checkout is busy: refused after the sweep
    assert "run-A" in capsys.readouterr().err
    assert stop_request_pending(layout.session_dir)
    assert not (layout.session_dir / "steer.request").exists()


def test_a_reused_ask_dir_drops_the_previous_legs_markers_and_keeps_this_legs(
    repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An ask session reuses its dir under the same id (transient Q&A), so a
    run_task that sweeps nothing starts the second leg on the first leg's
    leftover stop marker and ends at its first boundary. A bridge file older
    than the journal's last write is the previous leg's; a younger one was
    written for this leg."""
    import os

    from agent6.app import run as run_mod
    from agent6.app._leg import LegEnd
    from agent6.sessions.ipc import request_stop, steer_request_pending, stop_request_pending

    state = resolved_state_dir(repo)
    layout = SessionLayout(state_dir=state, session_id="chat", subdir="asks")
    layout.ensure()
    layout.logs_path.write_text(
        '{"type": "session.start", "mode": "ask", "user_task": "q"}\n'
        '{"type": "session.end", "reason": "finish_session", "all_passed": true}\n',
        encoding="utf-8",
    )
    leg_end = layout.logs_path.stat().st_mtime
    (layout.session_dir / "steer.request").write_text("", encoding="utf-8")
    os.utime(layout.session_dir / "steer.request", (leg_end - 10, leg_end - 10))
    request_stop(layout.session_dir)
    os.utime(layout.session_dir / "stop.request", (leg_end + 1, leg_end + 1))
    seen: list[tuple[bool, bool]] = []

    def _leg(*_a: object, **_k: object) -> LegEnd:
        d = layout.session_dir
        seen.append((steer_request_pending(d), stop_request_pending(d)))
        return LegEnd(rc=0)

    monkeypatch.setattr(run_mod, "run_leg", _leg)
    rc = run_mod.run_task(
        _load_cfg(), "again?", frontend=MagicMock(), session_id="chat", mode="ask"
    )
    assert rc == 0
    assert seen == [(False, True)]
