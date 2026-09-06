# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Steering a run that has no controlling terminal (detached spawn).

Regression: make_steer_state used to return a null steer without /dev/tty, so
a run spawned from the TUI hub or the web UI never polled steer.request and
every front-end steer was silently dropped.
"""

from __future__ import annotations

import builtins
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from agent6.events import EventSink
from agent6.sessions.ipc import (
    request_steer,
    steer_request_pending,
    submit_steer,
    write_steer_answer,
)
from agent6.ui.cli._steer import file_bridge_steer, install_steer_sigint, make_steer_state


def test_prompt_consumes_bridged_answer(tmp_path: Path) -> None:
    steer = file_bridge_steer(tmp_path)
    assert steer.requested() is False
    write_steer_answer(tmp_path, "focus on the tests")
    request_steer(tmp_path)
    assert steer.requested() is True
    # The ruled default: a plain steer waits for the step boundary (aborting
    # the in-flight call wastes the streamed tokens); only the `now` urgency
    # (`steer --now`) interrupts.
    assert steer.interrupt() is False
    assert steer.prompt() == "focus on the tests"
    steer.clear()
    assert steer.requested() is False

    write_steer_answer(tmp_path, "wrap up")
    request_steer(tmp_path, now=True)
    assert steer.requested() is True
    assert steer.interrupt() is True
    assert steer.prompt() == "wrap up"
    steer.clear()


def test_prompt_without_answer_clears_request(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A dead/abandoned front-end yields None; the request marker must go with
    # it or the next loop boundary re-triggers another blocking read forever.
    def no_answer(session_dir: Path) -> str | None:
        return None

    monkeypatch.setattr("agent6.ui.cli._steer.read_steer_answer", no_answer)
    request_steer(tmp_path)
    steer = file_bridge_steer(tmp_path)
    assert steer.prompt() is None
    assert steer_request_pending(tmp_path) is False


def test_make_steer_state_without_tty_uses_bridge(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real_open = builtins.open

    def fake_open(file: object, *args: object, **kwargs: object):  # type: ignore[no-untyped-def]
        if file == "/dev/tty":
            raise OSError("no controlling terminal")
        return real_open(file, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr("builtins.open", fake_open)
    events = EventSink(tmp_path / "logs.jsonl")
    steer = make_steer_state(events, tmp_path)
    # The old null steer answered False here even with a request pending.
    request_steer(tmp_path)
    assert steer.requested() is True


def test_steer_answer_is_abort_peeks_without_consuming(tmp_path: Path) -> None:
    """The non-blocking stop peek: True only for abort/stop, and it never consumes
    the answer (the between-step boundary still handles it)."""
    from agent6.sessions.ipc import steer_answer_is_abort

    assert not steer_answer_is_abort(tmp_path)  # no answer file yet
    write_steer_answer(tmp_path, "focus on the parser")
    assert not steer_answer_is_abort(tmp_path)  # a steering instruction is not a stop
    write_steer_answer(tmp_path, "stop")
    # Even "stop" is a steer instruction, not a stop -- the Stop button writes
    # "abort", and the between-step boundary stops only on "abort". Consistency.
    assert not steer_answer_is_abort(tmp_path)
    write_steer_answer(tmp_path, "  ABORT  ")
    assert steer_answer_is_abort(tmp_path)  # exactly the Stop contract, case/space-insensitive
    assert (tmp_path / "steer.answer").exists()  # peek did not consume it
    # A non-UTF-8 answer must read as "no abort", never raise -- a raising peek
    # would kill the streaming watchdog thread (and its idle-hang detection).
    (tmp_path / "steer.answer").write_bytes(b"\xff\xfe not utf8")
    assert not steer_answer_is_abort(tmp_path)


def _silent_banner(text: str) -> None:
    """tty_message stand-in: keep test output off the developer's terminal."""


def test_sigint_escalates_boundary_interrupt_stop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The Ctrl-C stages: 1st pauses at the next between-step boundary (the
    in-flight call finishes), 2nd interrupts the in-flight call, 3rd stops."""
    import signal

    monkeypatch.setattr("agent6.ui.cli._steer.tty_message", _silent_banner)
    events = EventSink(tmp_path / "logs.jsonl")
    steer = install_steer_sigint(events, tmp_path)
    try:
        assert steer.requested() is False
        signal.raise_signal(signal.SIGINT)  # 1st: graceful pause
        assert steer.requested() is True
        assert steer.interrupt() is False
        signal.raise_signal(signal.SIGINT)  # 2nd: abort the in-flight call
        assert steer.interrupt() is True
        with pytest.raises(KeyboardInterrupt):  # 3rd: stop the run
            signal.raise_signal(signal.SIGINT)
        steer.clear()
        assert steer.requested() is False
        assert steer.interrupt() is False
    finally:
        steer.restore()


def test_sigint_at_the_pause_prompt_stops(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """At the pause prompt itself a Ctrl-C stops the run outright, whatever the
    stage: the banner promised it, and there is nothing in flight to interrupt."""
    import signal

    monkeypatch.setattr("agent6.ui.cli._steer.tty_message", _silent_banner)
    monkeypatch.setattr("agent6.ui.cli._steer.menu_capable", lambda: False)

    def prompt_hit_by_ctrl_c(text: str, **_kw: object) -> str | None:
        signal.raise_signal(signal.SIGINT)
        return ""

    monkeypatch.setattr("agent6.ui.cli._steer.tty_prompt", prompt_hit_by_ctrl_c)
    events = EventSink(tmp_path / "logs.jsonl")
    steer = install_steer_sigint(events, tmp_path)
    try:
        signal.raise_signal(signal.SIGINT)  # stage 1: the boundary pause
        with pytest.raises(KeyboardInterrupt):
            steer.prompt()
    finally:
        steer.restore()


def test_a_seeded_steer_is_the_answer_on_the_terminal_too(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`resume --steer` and the end-of-session follow-up seed the answer file
    before the loop's first boundary; the terminal steer path consumed only a
    live front-end's answer and otherwise opened the pause menu, so the text
    the operator had just typed was asked for again."""
    monkeypatch.setattr("agent6.ui.cli._steer.menu_capable", lambda: True)

    def no_menu(session_dir: Path, **_kw: object) -> str | None:
        pytest.fail("the menu opened over a seeded steer")

    monkeypatch.setattr("agent6.ui.cli._steer.pause_menu", no_menu)
    events = EventSink(tmp_path / "logs.jsonl")
    submit_steer(tmp_path, "also add a test that mul(2, 0) == 0")
    steer = install_steer_sigint(events, tmp_path)
    try:
        assert steer.requested()
        assert steer.prompt() == "also add a test that mul(2, 0) == 0"
        steer.clear()
    finally:
        steer.restore()
    assert not (tmp_path / "steer.answer").exists()


def test_prompt_pauses_the_console_spinner(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The pause menu runs inside ConsoleView.pause(): the heartbeat spinner's
    per-tick line-erase otherwise wipes the pause-menu line and its Tab preview."""
    import contextlib
    from collections.abc import Generator
    from typing import cast

    from agent6.ui.cli._console_view import ConsoleView

    calls: list[str] = []

    class FakeView:
        @contextlib.contextmanager
        def pause(self) -> Generator[None]:
            calls.append("pause")
            yield
            calls.append("resume")

    monkeypatch.setattr("agent6.ui.cli._steer.menu_capable", lambda: True)

    def fake_menu(session_dir: Path, **_kw: object) -> str | None:
        calls.append("prompt")
        return "steer text"

    monkeypatch.setattr("agent6.ui.cli._steer.pause_menu", fake_menu)
    events = EventSink(tmp_path / "logs.jsonl")
    steer = install_steer_sigint(events, tmp_path, cast(ConsoleView, FakeView()))
    try:
        assert steer.prompt() == "steer text"
    finally:
        steer.restore()
    assert calls == ["pause", "prompt", "resume"]


def test_revision_selector_pauses_the_console_spinner(monkeypatch: pytest.MonkeyPatch) -> None:
    """select_revised_prompt blocks on input() for as long as the operator
    reads the proposal; without ConsoleView.pause() the heartbeat erases the
    choice prompt (and the typed echo) every 0.5s -- the one foreground-CLI
    interactive prompt that was never handed the view."""
    import contextlib
    from collections.abc import Generator
    from typing import cast

    from agent6.ui.cli._console_view import ConsoleView
    from agent6.ui.cli._steer import select_revised_prompt

    calls: list[str] = []

    class FakeView:
        @contextlib.contextmanager
        def pause(self) -> Generator[None]:
            calls.append("pause")
            yield
            calls.append("resume")

    def fake_input(prompt: str = "") -> str:
        calls.append("prompt")
        return "a"

    monkeypatch.setattr("builtins.input", fake_input)
    out = select_revised_prompt("orig", "rev", (), cast("ConsoleView", FakeView()))
    assert out == "rev"
    assert calls == ["pause", "prompt", "resume"]
    # And the bare (console_view=None) path still works.
    assert select_revised_prompt("orig", "rev", ()) == "rev"


def test_reset_stage_disarms_without_touching_the_markers(tmp_path: Path) -> None:
    """A stage armed in one leg must not leak into the next (phantom pause
    menu; stage 2 aborts the next leg's first call). reset_stage zeroes ONLY
    the SIGINT stage: the steer marker files stay, because resume --steer
    seeds the next leg through them."""
    import signal

    from agent6.sessions.ipc import request_steer, steer_request_pending, write_steer_answer
    from agent6.ui.cli._steer import install_steer_sigint

    events = MagicMock()
    steer = install_steer_sigint(events, tmp_path)
    try:
        handler = signal.getsignal(signal.SIGINT)
        assert callable(handler)
        handler(signal.SIGINT, None)  # 1st Ctrl-C arms stage 1
        assert steer.requested()
        steer.reset_stage()
        assert not steer.requested()  # the armed stage is gone
        # A front-end marker steer is the OTHER requested() source and must
        # survive a leg boundary (a web steer typed near leg end reaches the
        # next leg); reset_stage leaves both files alone.
        write_steer_answer(tmp_path, "carry on")
        request_steer(tmp_path)
        steer.reset_stage()
        assert (tmp_path / "steer.answer").exists()
        assert steer_request_pending(tmp_path)
        assert steer.requested()  # marker-driven, by design
    finally:
        steer.restore()


def test_workflow_run_resets_the_steer_stage_at_leg_entry() -> None:
    """Each wf.run() leg starts with no armed Ctrl-C (the ask REPL re-enters
    run() per follow-up under one installed handler): the reset fires at the
    very top of run(), before any other leg work."""
    import contextlib

    from agent6.workflows.loop import Workflow

    resets: list[bool] = []

    def spy() -> None:
        resets.append(True)

    wf = Workflow(
        root=Path("/tmp"),
        config=MagicMock(),
        provider=MagicMock(),
        dispatcher=MagicMock(),
        mode="ask",
        steer_reset=spy,
    )
    for expected in (1, 2):
        with contextlib.suppress(Exception):  # mocks explode later in the leg
            wf.run("q")
        assert len(resets) == expected


def test_the_turn_boundary_settles_background_commands(tmp_path: Path) -> None:
    """A background command's ending reaches disk when someone observes it, and
    a model that starts one and never asks again left `/shells` -- which reads
    off disk, at this very boundary -- reporting it maybe-running for the rest
    of the run. The boundary observes once per turn."""
    from agent6.providers import ProviderResponse
    from agent6.workflows.loop import Workflow

    repo = tmp_path / "repo"
    repo.mkdir()
    provider = MagicMock()
    provider.call.return_value = ProviderResponse(
        text="done",
        tool_uses=(),
        stop_reason="end_turn",
        input_tokens=1,
        output_tokens=1,
        cache_read_tokens=0,
        cache_creation_tokens=0,
        raw={"content": [{"type": "text", "text": "done"}]},
    )
    dispatcher = MagicMock()
    wf = Workflow(
        root=repo,
        config=MagicMock(
            budget=SimpleNamespace(max_usd=10.0, max_tokens_fallback=2_000_000),
            prompt=MagicMock(system_prompt_file=""),
            workflow=MagicMock(verify_command=(), verify_when="never", verify_retries=2),
        ),
        provider=provider,
        dispatcher=dispatcher,
        logger=lambda _msg: None,
        provider_retry_count=0,
        provider_retry_delay_s=0.0,
        max_iterations=3,
    )
    wf.run("do something")
    assert dispatcher.settle_background.called, "no turn boundary observed the background commands"


def test_compact_request_carries_focus(tmp_path: Path) -> None:
    """The compact marker body is the operator's optional summary focus:
    "" = plain compact, None = no request pending."""
    from agent6.sessions.ipc import clear_compact_request, read_compact_request, request_compact

    assert read_compact_request(tmp_path) is None
    request_compact(tmp_path)
    assert read_compact_request(tmp_path) == ""
    request_compact(tmp_path, focus="weigh the auth decisions")
    assert read_compact_request(tmp_path) == "weigh the auth decisions"
    clear_compact_request(tmp_path)
    assert read_compact_request(tmp_path) is None


def test_compact_request_reports_a_failed_write(tmp_path: Path) -> None:
    """A marker that could not be written must read as a failure. The write was
    wrapped in suppress(OSError) while every front-end reported "compaction
    requested" unconditionally, so a read-only or full state dir looked like
    success and nothing ever compacted."""
    from agent6.sessions.ipc import read_compact_request, request_compact

    assert request_compact(tmp_path) is True
    # A run dir that is really a file: the publish cannot succeed.
    blocked = tmp_path / "not-a-dir"
    blocked.write_text("x", encoding="utf-8")
    assert request_compact(blocked) is False
    assert read_compact_request(blocked) is None


def test_compact_request_publishes_atomically(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """request_compact publishes via tmp+rename (portable.atomic_write). The run
    polls read_compact_request every boundary, so a plain write_text exposed an
    empty/partial focus the run then consumed -- and clear_compact_request
    deleted the real one before it was ever read."""
    from agent6.sessions import ipc

    calls: list[tuple[Path, str]] = []
    real = ipc.atomic_write

    def spy(path: Path, data: str) -> None:
        calls.append((path, data))
        real(path, data)

    monkeypatch.setattr(ipc, "atomic_write", spy)
    ipc.request_compact(tmp_path, focus="pin the auth decisions")
    assert (tmp_path / "compact.request", "pin the auth decisions") in calls
    assert ipc.read_compact_request(tmp_path) == "pin the auth decisions"


def test_the_fallback_pause_prompt_takes_a_steer_written_while_it_waits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The plain prompt (no menu-capable terminal) reads the steer file while
    it waits, like the menu: a front-end's steer ends the prompt and is the
    answer, not a line the operator never typed."""
    monkeypatch.setattr("agent6.ui.cli._steer.tty_message", _silent_banner)
    monkeypatch.setattr("agent6.ui.cli._steer.menu_capable", lambda: False)

    def prompt_superseded(text: str, **kw: object) -> str | None:
        until = kw.get("until")
        assert callable(until) and not until()
        submit_steer(tmp_path, "from the web")
        assert until()  # the prompt would now end
        return None

    monkeypatch.setattr("agent6.ui.cli._steer.tty_prompt", prompt_superseded)
    events = EventSink(tmp_path / "logs.jsonl")
    steer = install_steer_sigint(events, tmp_path)
    try:
        import signal

        signal.raise_signal(signal.SIGINT)  # stage 1: the boundary pause
        assert steer.prompt() == "from the web"
        steer.clear()
    finally:
        steer.restore()


def test_edit_survives_an_unparsable_editor(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """An $EDITOR with unbalanced quoting is a choose-again, like a missing
    binary: shlex.split's ValueError escaped every guard up to the leg's
    `except Exception`, ending the run as crashed."""
    import io
    import sys

    from agent6.ui.cli._steer import _select_revised_prompt  # pyright: ignore[reportPrivateUsage]

    monkeypatch.setenv("EDITOR", 'code --wait "')
    monkeypatch.setattr(sys, "stdin", io.StringIO("e\na\n"))
    assert _select_revised_prompt("orig", "revised", ()) == "revised"
    assert "$EDITOR" in capsys.readouterr().err


def test_edit_survives_a_non_utf8_save(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """An editor that writes back non-UTF-8 bytes is a choose-again; the
    UnicodeDecodeError from reading the file back ended the run."""
    import io
    import sys

    from agent6.ui.cli._steer import _select_revised_prompt  # pyright: ignore[reportPrivateUsage]

    script = tmp_path / "fake_editor.sh"
    script.write_text("#!/bin/sh\nprintf '\\377\\376x' > \"$1\"\n", encoding="utf-8")
    script.chmod(0o755)
    monkeypatch.setenv("EDITOR", str(script))
    monkeypatch.setattr(sys, "stdin", io.StringIO("e\na\n"))
    assert _select_revised_prompt("orig", "revised", ()) == "revised"
    assert "not UTF-8" in capsys.readouterr().err


def test_one_ctrl_c_at_the_revise_prompt_leaves_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One Ctrl-C leaves the revise_prompt choice, as at every other idle CLI
    prompt. The leg installs the run's escalating steer handler before the
    loop reaches this prompt, so without `repl_prompt_sigint` the press was
    absorbed by the retried input(): three presses to leave, a "pausing after
    this step" line with no step in flight, and an armed stage that opened a
    pause menu at the run's first boundary."""
    import contextlib
    import os
    import signal
    import sys
    import threading
    import time
    from typing import Any

    from agent6.ui.cli._steer import select_revised_prompt

    read_fd, write_fd = os.pipe()
    reader = os.fdopen(read_fd, encoding="utf-8")
    blocked = threading.Event()

    class NotifyingStdin:
        def readline(self) -> str:
            blocked.set()
            return reader.readline()

        def __getattr__(self, name: str) -> Any:
            return getattr(reader, name)

    monkeypatch.setattr(sys, "stdin", NotifyingStdin())
    steer = install_steer_sigint(MagicMock(), tmp_path)

    def press_then_rescue() -> None:
        blocked.wait(10.0)
        os.kill(os.getpid(), signal.SIGINT)
        # A swallowed press leaves the prompt blocked: answer it so the test
        # reports the wrong result instead of hanging the suite.
        time.sleep(1.0)
        with contextlib.suppress(OSError):
            os.write(write_fd, b"a\n")

    presser = threading.Thread(target=press_then_rescue)
    presser.start()
    try:
        assert select_revised_prompt("original", "revised", ()) is None
        assert steer.armed() is False
    finally:
        steer.restore()
        presser.join()
        os.close(write_fd)
