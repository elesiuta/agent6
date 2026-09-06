# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Tests for the shared spawn+locate helper behind the hub's "start a run" and the
machines page's "create" -- both spawn the CLI detached, then watch the new log dir."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from agent6.paths import state_dir
from agent6.sessions.layout import bucket_dir
from agent6.ui import spawn


def test_spawn_and_locate_finds_new_log_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base = tmp_path / "dirs"
    base.mkdir()

    class _Proc:
        pid = 424242
        # A detached child is registered with the escapee sweep by pid, so a
        # stub without one no longer models a real spawn.
        pid = 424242

        def __init__(self) -> None:
            # The "child" produces a new dir with a logs.jsonl the moment it starts.
            (base / "new").mkdir()
            (base / "new" / "logs.jsonl").write_text("", encoding="utf-8")

        def poll(self) -> int | None:
            return None  # still running

    def _popen(*_a: object, **_k: object) -> _Proc:
        return _Proc()

    monkeypatch.setattr(spawn.subprocess, "Popen", _popen)
    found, err = spawn.spawn_and_locate(
        ["agent6", "x"],
        tmp_path,
        before=set(),
        list_dirs=lambda: [p for p in base.iterdir() if p.is_dir()],
    )
    assert err == ""
    assert found == base / "new"


def test_spawn_and_locate_ignores_preexisting_dirs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base = tmp_path / "dirs"
    base.mkdir()
    (base / "old").mkdir()
    (base / "old" / "logs.jsonl").write_text("", encoding="utf-8")
    before = {base / "old"}

    class _Proc:
        pid = 424242
        returncode = 0

        def poll(self) -> int:
            return 0  # exits immediately without producing a NEW dir

    def _popen(*_a: object, **_k: object) -> _Proc:
        return _Proc()

    monkeypatch.setattr(spawn.subprocess, "Popen", _popen)
    found, err = spawn.spawn_and_locate(
        ["agent6", "machine", "create", "x"],
        tmp_path,
        before=before,
        list_dirs=lambda: [p for p in base.iterdir() if p.is_dir()],
    )
    assert found is None  # the only dir was already in `before`
    assert "exited" in err  # surfaced the early exit, not a 25s timeout


def test_spawn_and_locate_surfaces_spawn_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _boom(*_a: object, **_k: object) -> object:
        raise OSError("no exec")

    monkeypatch.setattr(spawn.subprocess, "Popen", _boom)
    found, err = spawn.spawn_and_locate(
        ["agent6", "run", "x"], tmp_path, before=set(), list_dirs=list
    )
    assert found is None
    assert "failed to start agent6 run" in err


# --- spawn_and_confirm: the machine-run launch with early-exit stderr capture --


def test_spawn_and_confirm_surfaces_refusal_stderr(tmp_path: Path) -> None:
    # A child that prints a refusal and exits nonzero before taking ownership
    # (lock held, network refusal) must surface its stderr, not "" (started).
    argv = [sys.executable, "-c", "import sys; sys.stderr.write('lock held'); sys.exit(2)"]
    err = spawn.spawn_and_confirm(argv, tmp_path, started=lambda _pid: False, timeout_s=10.0)
    assert err == "lock held"  # the child's own words, no plumbing prefix


def test_spawn_and_confirm_returns_clean_once_started(tmp_path: Path) -> None:
    # started(pid) flipping true ends the wait with "" while the child runs on.
    marker = tmp_path / "worker.pid"
    # The detached child exits on its own shortly after the started() signal.
    code = (
        "import os, time, pathlib, sys; "
        f"pathlib.Path({str(marker)!r}).write_text(str(os.getpid())); "
        "time.sleep(5)"
    )

    def started(pid: int) -> bool:
        try:
            return int(marker.read_text()) == pid
        except (OSError, ValueError):
            return False

    err = spawn.spawn_and_confirm([sys.executable, "-c", code], tmp_path, started=started)
    assert err == ""


def test_spawn_and_confirm_clean_fast_exit_is_ok(tmp_path: Path) -> None:
    # Exit 0 without the signal is a clean fast completion (an already-ended
    # machine re-run), not an error.
    err = spawn.spawn_and_confirm(
        [sys.executable, "-c", "raise SystemExit(0)"], tmp_path, started=lambda _pid: False
    )
    assert err == ""


# --- spawn_detached_resume: the resume launch over the same early-exit capture --


def _run_dir(cwd: Path, session_id: str) -> Path:
    d = bucket_dir(state_dir(cwd), "runs") / session_id
    d.mkdir(parents=True)
    return d


def _fake_agent6(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, script: str) -> None:
    exe = tmp_path / "agent6"
    exe.write_text("#!/bin/sh\n" + script, encoding="utf-8")
    exe.chmod(0o755)
    monkeypatch.setattr(spawn, "agent6_exe", lambda: str(exe))


def test_spawn_detached_resume_reports_the_childs_early_exit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A resume child that refuses at once (a preflight refusal, a crash) was
    spawned with its stderr on /dev/null and reported as "" (resuming): the web
    composer and the TUI said "resuming" over a run nothing was resuming."""
    _run_dir(tmp_path, "tidy-owl-9Z3AAA")
    _fake_agent6(tmp_path, monkeypatch, "echo 'REFUSING: the checkout is busy' >&2\nexit 2\n")
    err = spawn.spawn_detached_resume(tmp_path, "tidy-owl-9Z3AAA")
    assert err == "REFUSING: the checkout is busy"


def test_spawn_detached_resume_argv_env_and_owning_signal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """ "" once the child owns the run: its pid is the run's worker.pid. The
    child runs detached with the bridge environment (deltas streamed to the log
    for a later attach, asks and approvals waiting for a front-end)."""
    run = _run_dir(tmp_path, "tidy-owl-9Z3AAA")
    seen = tmp_path / "seen"
    _fake_agent6(
        tmp_path,
        monkeypatch,
        f"printf '%s\\n' \"$@\" > {seen}\n"
        f"env | grep '^AGENT6_' >> {seen}\n"
        f"echo $$ > {run / 'worker.pid'}\n"
        "sleep 5\n",
    )
    err = spawn.spawn_detached_resume(tmp_path, "tidy-owl-9Z3AAA", steer="go on", preset="quick")
    assert err == ""
    lines = seen.read_text(encoding="utf-8").splitlines()
    assert lines[:4] == ["resume", "tidy-owl-9Z3AAA", "--preset=quick", "--steer=go on"]
    assert "AGENT6_STREAM_TO_LOG=1" in lines and "AGENT6_DETACHED_AWAY=wait" in lines


def test_spawn_detached_resume_names_an_unknown_session(tmp_path: Path) -> None:
    """Resolved before the spawn, in the CLI's own words: the child would refuse
    the same way on a stdio nobody reads."""
    err = spawn.spawn_detached_resume(tmp_path, "no-such-run-AAAAAA")
    assert "no session matches 'no-such-run-AAAAAA'" in err


def test_spawn_detached_resume_reports_oserror(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _run_dir(tmp_path, "tidy-owl-9Z3AAA")

    def _boom(*_a: object, **_k: object) -> object:
        raise OSError("no exec")

    monkeypatch.setattr(spawn.subprocess, "Popen", _boom)
    err = spawn.spawn_detached_resume(tmp_path, "tidy-owl-9Z3AAA")
    assert err == "failed to start agent6 resume: no exec"


# --- diagnostics wording: subcommand labels + captured-output cleanup ----------


def test_subcommand_label_names_the_full_subcommand() -> None:
    assert spawn.subcommand_label(["a6", "machine", "run", "m.asm.toml"]) == "machine run"
    assert spawn.subcommand_label(["a6", "sessions", "prune"]) == "sessions prune"
    assert spawn.subcommand_label(["a6", "config", "set", "--", "k", "v"]) == "config set"
    # One-word subcommands never swallow the value that follows them.
    assert spawn.subcommand_label(["a6", "run", "--", "fix the bug"]) == "run"
    assert spawn.subcommand_label(["a6", "plan", "--preset", "p", "--", "t"]) == "plan"


def test_run_cli_capture_strips_console_prefixes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The CLI brands its own console lines "[agent6] " to stand apart from
    # pass-through git output; in a front-end toast that prefix is noise.
    class _Done:
        returncode = 0
        stdout = "[agent6] merged a into b\n\n[agent6] deleted branch a\n"
        stderr = "[agent6] skipped c (checked out)\n"

    def _fake_run(*_a: object, **_k: object) -> _Done:
        return _Done()

    monkeypatch.setattr(spawn.subprocess, "run", _fake_run)
    ok, msg = spawn.run_cli_capture(["a6", "sessions", "prune"], tmp_path)
    assert ok
    assert msg == "merged a into b\ndeleted branch a\nskipped c (checked out)"


def test_capture_message_drops_the_error_prefix_a_failure_field_already_states() -> None:
    """`ERROR: ` is the console's failure marker; an API `error` field or a red
    toast carried it verbatim (`"error": "ERROR: unknown config key ..."`)."""
    assert (
        spawn.capture_message("", "ERROR: unknown config key 'x.y'") == "unknown config key 'x.y'"
    )
    assert spawn.capture_message("[agent6] ERROR: no branch\n", "") == "no branch"
    # A refusal keeps its kind: REFUSING / PARKED say what happened, not that it did.
    assert spawn.capture_message("REFUSING: run r is live") == "REFUSING: run r is live"


def test_agent6_exe_finds_the_binary_beside_the_interpreter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A view started as `python -m agent6.ui.tui` has a module path in
    argv[0]; the binary of the same install sits beside its interpreter."""
    binary = tmp_path / "bin" / "agent6"
    binary.parent.mkdir()
    binary.write_text("#!/bin/sh\n", encoding="utf-8")
    monkeypatch.setattr(spawn.sys, "argv", [str(tmp_path / "ui" / "tui" / "__main__.py")])
    monkeypatch.setattr(spawn.sys, "executable", str(tmp_path / "bin" / "python3"))
    assert spawn.agent6_exe() == str(binary.resolve())


def test_stderr_tail_starts_at_a_line(tmp_path: Path) -> None:
    """A long refusal is clipped at a line start, never mid-word."""
    f = tmp_path / "err"
    f.write_text("REFUSING: " + "x" * 50 + "\n" + "y" * 30 + "\n", encoding="utf-8")
    with f.open("r+", encoding="utf-8") as fh:
        assert spawn._stderr_tail(fh, limit=40) == "y" * 30 + "\n"  # pyright: ignore[reportPrivateUsage]
        assert spawn._stderr_tail(fh) == f.read_text(encoding="utf-8")  # pyright: ignore[reportPrivateUsage]
