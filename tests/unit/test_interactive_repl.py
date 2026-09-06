# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Tests for interactive REPL plumbing.

Covers:
* ``cli._build_repl_hook`` slash-command dispatch:
  - empty input / ``/continue`` -> ``"continue"``
  - ``/quit`` -> ``"stop"``
  - EOF -> ``"stop"``
  - ``/cost`` invokes ``budget.format_summary`` then re-prompts
  - ``/undo`` -> ``"undo"`` (the loop's fork-back undo; the chain never
    moves HEAD, so a `git revert HEAD` would have reverted the checkout's
    own commit)
  - unknown command re-prompts
* ``Workflow`` exits cleanly with ``reason="interactive_stop"`` when
  the hook returns ``"stop"`` after an auto-commit, and takes the undo
  fork on ``"undo"``.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from agent6.budget import BudgetTracker
from agent6.paths import state_dir
from agent6.ui.cli._repl import REPL_HELP
from agent6.ui.cli.run import build_repl_hook  # pyright: ignore[reportPrivateUsage]


def _init_repo(path: Path) -> None:
    subprocess.run(["git", "init", "-q", "-b", "main", str(path)], check=True)
    subprocess.run(["git", "-C", str(path), "config", "user.email", "t@t"], check=True)
    subprocess.run(["git", "-C", str(path), "config", "user.name", "t"], check=True)
    (path / "a.txt").write_text("v1\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(path), "add", "-A"], check=True)
    subprocess.run(["git", "-C", str(path), "commit", "-q", "-m", "init"], check=True)


def _commit(path: Path, name: str, body: str, msg: str) -> str:
    (path / name).write_text(body, encoding="utf-8")
    subprocess.run(["git", "-C", str(path), "add", "-A"], check=True)
    subprocess.run(["git", "-C", str(path), "commit", "-q", "-m", msg], check=True)
    out = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return out.stdout.strip()


# --- _build_repl_hook dispatch -------------------------------------------


def _budget() -> BudgetTracker:
    return BudgetTracker(max_usd=-1, max_tokens_fallback=-1, max_percent=-1)


def test_hook_pauses_the_console_heartbeat_while_prompting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The REPL prompt must sit inside the console view's pause(): the run is
    waiting on the OPERATOR, and without the pause the heartbeat's per-tick
    line-erase wiped the "agent6> " prompt and the typed characters, replacing
    them with a lying "working…" spinner (keystrokes were submitted blind)."""
    states: list[str] = []

    class _FakePause:
        def __enter__(self) -> None:
            states.append("paused")

        def __exit__(self, *exc: object) -> None:
            states.append("resumed")

    class _FakeConsole:
        def pause(self) -> _FakePause:
            return _FakePause()

    def _input(_p: str = "") -> str:
        assert states == ["paused"], "input() ran outside the console pause"
        return ""

    monkeypatch.setattr("builtins.input", _input)
    hook = build_repl_hook(tmp_path, _budget(), console_view=_FakeConsole())  # type: ignore[arg-type]
    assert hook(1, "a" * 40) == "continue"
    assert states == ["paused", "resumed"]


def test_hook_empty_input_continues(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("builtins.input", lambda _p="": "")
    hook = build_repl_hook(tmp_path, _budget())
    assert hook(1, "deadbeefcafe1234") == "continue"


def test_hook_slash_continue(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("builtins.input", lambda _p="": "/continue")
    hook = build_repl_hook(tmp_path, _budget())
    assert hook(2, "abc") == "continue"


def test_hook_quit_stops(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("builtins.input", lambda _p="": "/quit")
    hook = build_repl_hook(tmp_path, _budget())
    assert hook(3, "abc") == "stop"


def test_hook_exit_is_the_loops_exit_directive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`/exit` stops AND leaves: the loop ends the run `steer_exit`, which the
    follow-up prompt skips, unlike `/quit`'s stop that re-opens "next:"."""
    monkeypatch.setattr("builtins.input", lambda _p="": "/exit")
    hook = build_repl_hook(tmp_path, _budget())
    assert hook(3, "abc") == "exit"
    assert "/exit" in REPL_HELP


def test_hook_eof_stops(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def _raise(_p: str = "") -> str:
        raise EOFError

    monkeypatch.setattr("builtins.input", _raise)
    hook = build_repl_hook(tmp_path, _budget())
    assert hook(1, "abc") == "stop"


def test_hook_cost_reprompts_then_continues(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    answers = iter(["/cost", ""])
    monkeypatch.setattr("builtins.input", lambda _p="": next(answers))
    hook = build_repl_hook(tmp_path, _budget())
    assert hook(1, "abc") == "continue"


def test_hook_unknown_reprompts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    answers = iter(["/wat", "/quit"])
    monkeypatch.setattr("builtins.input", lambda _p="": next(answers))
    hook = build_repl_hook(tmp_path, _budget())
    assert hook(1, "abc") == "stop"


def test_hook_undo_is_the_loops_undo_directive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`/undo` hands the loop its own undo (fork back before the last message)
    and touches no git: the chain never moves HEAD, so reverting HEAD would
    revert the checkout's own commit, never the auto-commit."""
    _init_repo(tmp_path)
    head = _commit(tmp_path, "b.txt", "theirs\n", "the operator's commit")
    answers = iter(["/undo"])
    monkeypatch.setattr("builtins.input", lambda _p="": next(answers))
    hook = build_repl_hook(tmp_path, _budget())
    assert hook(1, "abc") == "undo"
    assert (tmp_path / "b.txt").exists()
    argv = ["git", "-C", str(tmp_path), "rev-parse", "HEAD"]
    assert subprocess.run(argv, check=True, capture_output=True, text=True).stdout.strip() == head


# --- Workflow integration ----------------------------------------------


def test_after_auto_commit_default_continues() -> None:
    """Default hook is a no-op lambda returning "continue"."""
    from agent6.workflows.loop import Workflow

    wf = Workflow(
        root=Path("/tmp"),
        config=MagicMock(
            prompt=MagicMock(system_prompt_file=""),
            workflow=MagicMock(verify_command=(), verify_when="never", verify_retries=2),
        ),
        provider=MagicMock(),
        dispatcher=MagicMock(),
        logger=lambda _m: None,
    )
    # Field exists and defaults to the no-op shape.
    assert wf.after_auto_commit(1, "abc") == "continue"


def test_after_auto_commit_field_is_overridable() -> None:
    """Custom hook is honoured (called with iteration + sha)."""
    from agent6.workflows.loop import Workflow

    calls: list[tuple[int, str]] = []

    def hook(it: int, sha: str) -> Any:
        calls.append((it, sha))
        return "stop"

    wf = Workflow(
        root=Path("/tmp"),
        config=MagicMock(
            prompt=MagicMock(system_prompt_file=""),
            workflow=MagicMock(verify_command=(), verify_when="never", verify_retries=2),
        ),
        provider=MagicMock(),
        dispatcher=MagicMock(),
        logger=lambda _m: None,
        after_auto_commit=hook,
    )
    assert wf.after_auto_commit(7, "deadbeef") == "stop"
    assert calls == [(7, "deadbeef")]


# --- steer marker self-heals on a dismissed/timed-out TUI modal ------------


def _tui_live(_session_dir: Path) -> bool:
    return True


def _answer_none(_session_dir: Path) -> str | None:
    return None  # modal dismissed / read_steer_answer timed out


def _answer_text(_session_dir: Path) -> str | None:
    return "do the thing"


def test_steer_prompt_clears_request_marker_on_no_answer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A TUI-initiated steer whose modal is dismissed (read_steer_answer -> None
    on timeout) must clear the `steer.request` marker so the run does NOT
    re-enter the 600s blocking prompt at every later boundary."""
    from agent6.sessions.ipc import request_steer, steer_request_pending
    from agent6.ui.cli import _steer

    session_dir = tmp_path
    request_steer(session_dir)  # TUI `s`-key dropped the marker
    assert steer_request_pending(session_dir)

    # TUI is live but the modal yields no answer (dismissed / 600s timeout).
    monkeypatch.setattr(_steer, "frontend_is_live", _tui_live)
    monkeypatch.setattr(_steer, "read_steer_answer", _answer_none)

    state = _steer.install_steer_sigint(MagicMock(), session_dir)
    try:
        assert state.requested() is True  # marker seen -> would prompt
        assert state.prompt() is None  # dismissed modal
        # The marker is gone, so the next boundary does NOT re-trigger a steer.
        assert not steer_request_pending(session_dir)
        assert state.requested() is False
    finally:
        state.restore()


def test_steer_prompt_keeps_marker_on_real_answer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A genuinely-answered steer still works: prompt() returns the answer and
    leaves clearing to the caller's clear() (which consumes request+answer)."""
    from agent6.sessions.ipc import request_steer, steer_request_pending
    from agent6.ui.cli import _steer

    session_dir = tmp_path
    request_steer(session_dir)
    monkeypatch.setattr(_steer, "frontend_is_live", _tui_live)
    monkeypatch.setattr(_steer, "read_steer_answer", _answer_text)

    state = _steer.install_steer_sigint(MagicMock(), session_dir)
    try:
        assert state.prompt() == "do the thing"
        # prompt() must NOT clear on the answered path (caller's clear() owns it).
        assert steer_request_pending(session_dir)
        state.clear()  # caller clears after consuming the answer
        assert not steer_request_pending(session_dir)
    finally:
        state.restore()


def test_watch_shows_audit_events_not_streaming_fragments(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """`/watch` printed the last N raw log lines, mostly one-word
    role.thinking_delta fragments of a single turn; it shows the audit lines
    every other log view shows (deltas and the loop's mirrors skipped)."""
    import json

    from agent6.sessions.layout import SessionLayout
    from agent6.ui.cli._repl import repl_show_recent_events

    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    layout = SessionLayout(
        state_dir=state_dir(tmp_path), session_id="watchy-run-AAAAAA", subdir="runs"
    )
    layout.session_dir.mkdir(parents=True)
    events: list[dict[str, object]] = [
        {"type": "session.start", "ts": "2026-01-01T00:00:00Z", "user_task": "t"},
        {"type": "tool.call", "ts": "2026-01-01T00:00:01Z", "name": "read_file", "args": {}},
        {"type": "loop.tool.call", "ts": "2026-01-01T00:00:01Z", "name": "read_file"},
        *(
            {"type": "role.thinking_delta", "ts": "2026-01-01T00:00:02Z", "text": f"w{i}"}
            for i in range(30)
        ),
        {"type": "tool.result", "ts": "2026-01-01T00:00:03Z", "name": "read_file", "ok": True},
    ]
    layout.logs_path.write_text("".join(json.dumps(e) + "\n" for e in events), encoding="utf-8")
    repl_show_recent_events(tmp_path, "watchy-run-AAAAAA", n=20)
    out = capsys.readouterr()
    assert "thinking_delta" not in out.out and "loop.tool.call" not in out.out
    assert "tool.call" in out.out and "tool.result" in out.out and "session.start" in out.out
    assert "last 3 events" in out.err


def test_i_on_a_pipe_refuses_up_front(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """-i promises a stdin REPL ("Requires a TTY"); on a pipe the REPL's first
    prompt read EOF and stopped the run mid-task after its first commit. The
    explicit-but-unhonourable flag refuses before anything runs, for run and
    resume alike."""
    from agent6.ui import cli

    monkeypatch.setattr(cli.sys.stdin, "isatty", lambda: False)

    def _must_not_run(*_a: object, **_k: object) -> int:
        pytest.fail("the run must not start")

    monkeypatch.setattr("agent6.ui.cli.run._cmd_run", _must_not_run)
    monkeypatch.setattr("agent6.ui.cli.resume._cmd_resume", _must_not_run)
    assert cli.main(["run", "-i", "do the thing"]) == 2
    assert "-i needs a TTY" in capsys.readouterr().err
    assert cli.main(["resume", "some-run-AAAAAA", "-i"]) == 2
    assert "-i needs a TTY" in capsys.readouterr().err


def test_init_wizard_ctrl_c_aborts_init_not_the_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Ctrl-C at an /init wizard question aborts /init and returns to the
    REPL. The prompt session is idle, so the leg's escalating steer handler
    must not own SIGINT inside the wizard's nested input(): it printed
    "pausing after this step" for a step that does not exist, and its third
    press escaped the hook and ended the whole run as interrupted."""
    import os
    import signal

    seen: dict[str, Any] = {}

    def _leg_handler(_signum: int, _frame: Any) -> None:
        seen["leg_handler_ran"] = True

    def _wizard(*_a: Any, **_kw: Any) -> int:
        os.kill(os.getpid(), signal.SIGINT)  # the operator's Ctrl-C at the y/n question
        return 0

    monkeypatch.setattr("agent6.ui.cli._repl.init_workspace", _wizard)
    answers = iter(["/init", "/continue"])
    monkeypatch.setattr("builtins.input", lambda _p="": next(answers))

    previous = signal.signal(signal.SIGINT, _leg_handler)
    try:
        directive = build_repl_hook(tmp_path, _budget())(1, "a" * 40)
    except KeyboardInterrupt:
        pytest.fail("Ctrl-C in the /init wizard escaped the REPL hook: the run ends")
    finally:
        signal.signal(signal.SIGINT, previous)
    assert directive == "continue"
    assert "leg_handler_ran" not in seen
    assert "/init cancelled." in capsys.readouterr().err


def test_diff_ctrl_c_aborts_diff_not_the_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Ctrl-C while /diff prints aborts /diff and returns to the REPL: a
    KeyboardInterrupt walked past the /diff failure handler, out of the hook,
    and the leg journaled the run as interrupted."""
    import signal

    def _diff(**_kw: Any) -> int:
        raise KeyboardInterrupt  # the operator's Ctrl-C while the patch prints

    monkeypatch.setattr("agent6.ui.cli._repl._cmd_diff", _diff)
    answers = iter(["/diff", "/continue"])
    monkeypatch.setattr("builtins.input", lambda _p="": next(answers))

    previous = signal.signal(signal.SIGINT, signal.default_int_handler)
    try:
        directive = build_repl_hook(tmp_path, _budget(), session_id="run-AAAAAA")(1, "a" * 40)
    except KeyboardInterrupt:
        pytest.fail("Ctrl-C during /diff escaped the REPL hook: the run ends interrupted")
    finally:
        signal.signal(signal.SIGINT, previous)
    assert directive == "continue"
    assert "/diff cancelled." in capsys.readouterr().err
