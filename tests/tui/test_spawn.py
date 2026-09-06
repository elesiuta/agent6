# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Tests for the shared spawn+locate helper behind the hub's "start a run" and the
machines page's "create" -- both spawn the CLI detached, then watch the new log dir."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

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


def test_spawn_detached_resume_argv_and_stream_env(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, object] = {}

    class _Proc:
        pid = 424242

    def _popen(argv: list[str], **kw: object) -> _Proc:
        seen["argv"] = argv
        seen["kw"] = kw
        return _Proc()

    monkeypatch.setattr(spawn.subprocess, "Popen", _popen)
    monkeypatch.setattr(spawn, "agent6_exe", lambda: "/opt/agent6")
    err = spawn.spawn_detached_resume(Path("/repo"), "tidy-owl-9Z3")
    assert err == ""
    assert seen["argv"] == ["/opt/agent6", "resume", "tidy-owl-9Z3"]
    kw = seen["kw"]
    assert isinstance(kw, dict)
    assert kw["start_new_session"] is True
    env = kw["env"]
    assert isinstance(env, dict)
    assert env["AGENT6_STREAM_TO_LOG"] == "1"  # headless child still emits deltas for watch
    # No terminal, so the child must WAIT for a front-end at an ask/approval
    # instead of fabricating an empty answer (parity with the fresh-run spawns).
    assert env["AGENT6_DETACHED_AWAY"] == "wait"


def test_spawn_detached_resume_reports_oserror(monkeypatch: pytest.MonkeyPatch) -> None:
    def _boom(*_a: object, **_k: object) -> object:
        raise OSError("no exec")

    monkeypatch.setattr(spawn.subprocess, "Popen", _boom)
    err = spawn.spawn_detached_resume(Path("/repo"), "tidy-owl-9Z3")
    assert "could not spawn" in err


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
