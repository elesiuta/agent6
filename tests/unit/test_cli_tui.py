# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""run_command approver bridge + TUI auto-spawn gating.

The textual TUI was fully built (modal, writes `approvals/<id>.answer`) but the
workflow side never read those answers and never auto-spawned the dashboard.
These cover the wiring that fixes that.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

import pytest

from agent6.events import EventSink
from agent6.sessions.ipc import COMMAND_SCOPE
from agent6.tools.operator_prompts import OperatorPrompts
from agent6.tools.schema import UserQuestion
from agent6.ui.cli import _interact as interactmod
from agent6.ui.cli import _live as livemod
from agent6.ui.steer import SteerState


def _events_of(log: Path, type_: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for line in log.read_text(encoding="utf-8").splitlines():
        obj = json.loads(line)
        if obj.get("type") == type_:
            out.append(obj)
    return out


def _prompts(
    session_dir: Path, events: EventSink, steer_cell: list[SteerState | None] | None = None
) -> OperatorPrompts:
    """The gate over the CLI's own approver and questioner: the pairing a
    run wires, journaling into *events*."""
    return OperatorPrompts(
        approver=interactmod.build_approver(session_dir, None, steer_cell),
        questioner=interactmod.build_questioner(session_dir),
        journal=events.emit,
        session_dir=session_dir,
    )


def _live(_d: object) -> bool:
    return True


def _dead(_d: object) -> bool:
    return False


def _ans_yes(_d: object, _pid: object, **_k: object) -> str:
    return "yes"


def _ans_none(_d: object, _pid: object, **_k: object) -> str | None:
    return None


def _stdin_no(_p: object, **_k: object) -> str:
    return "no"


def _stdin_yes(_p: object, **_k: object) -> str:
    return "yes"


def _stdin_forbidden(_p: object, **_k: object) -> str:
    pytest.fail("stdin approver must not be used")


def _stdin_session(_p: object, **_k: object) -> str:
    return "session"


def test_approver_uses_tui_answer_when_live(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    log = tmp_path / "logs.jsonl"
    events = EventSink(log)
    monkeypatch.setattr(interactmod, "frontend_is_live", _live)
    monkeypatch.setattr(interactmod, "read_answer", _ans_yes)
    monkeypatch.setattr(interactmod, "default_stdin_approver", _stdin_forbidden)
    approve = _prompts(tmp_path, events).approve
    assert approve("run `ls`?", scope=COMMAND_SCOPE) is True
    assert _events_of(log, "approval.prompt")
    ans = _events_of(log, "approval.answer")[0]
    assert ans["approved"] is True
    assert ans["source"] == "frontend"


def test_approver_does_not_consume_an_answer_written_before_the_prompt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A premature /api/session/<id>/approve (ids are predictable counters) pre-writes
    # approvals/approval-1.answer before the run reaches its first approval. The
    # gate must clear that stale slot before journaling the prompt, so it is
    # not silently consumed as an auto-approval. Uses the REAL read_answer (short
    # timeout) so this exercises the actual file-bridge ordering.
    import functools

    from agent6.sessions.ipc import read_answer, write_answer

    log = tmp_path / "logs.jsonl"
    events = EventSink(log)
    monkeypatch.setattr(interactmod, "frontend_is_live", _live)
    monkeypatch.setattr(
        interactmod, "read_answer", functools.partial(read_answer, timeout_s=0.4, poll_s=0.05)
    )
    monkeypatch.setattr(interactmod, "_has_controlling_tty", _tty)  # foreground stdin path
    monkeypatch.setattr(interactmod, "default_stdin_approver", _stdin_no)
    write_answer(tmp_path, "approval-1", "yes")  # the premature POST
    approve = _prompts(tmp_path, events).approve
    # The premature "yes" is cleared before the prompt; read_answer finds nothing
    # and times out, so it falls back to stdin (which denies) -- NOT auto-approved.
    assert approve("run `curl evil`?", scope=COMMAND_SCOPE) is False
    assert _events_of(log, "approval.answer")[0]["source"] == "stdin"


def test_approver_consumes_an_answer_written_after_the_prompt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The legitimate path: the front-end writes the answer only after it renders
    # the emitted prompt. A writer thread does exactly that; the answer is honored.
    import functools
    import threading
    import time

    from agent6.sessions.ipc import read_answer, write_answer

    log = tmp_path / "logs.jsonl"
    events = EventSink(log)
    monkeypatch.setattr(interactmod, "frontend_is_live", _live)
    monkeypatch.setattr(
        interactmod, "read_answer", functools.partial(read_answer, timeout_s=3.0, poll_s=0.05)
    )
    monkeypatch.setattr(interactmod, "default_stdin_approver", _stdin_no)

    def writer() -> None:
        time.sleep(0.3)  # after the prompt is emitted and the poll starts
        write_answer(tmp_path, "approval-1", "yes")

    t = threading.Thread(target=writer, daemon=True)
    t.start()
    approve = _prompts(tmp_path, events).approve
    assert approve("run `ls`?", scope=COMMAND_SCOPE) is True
    t.join(timeout=2)
    assert _events_of(log, "approval.answer")[0]["source"] == "frontend"


def _tty(_: object = None) -> bool:
    return True  # simulate a controlling terminal (foreground stdin path)


def test_approver_falls_back_to_stdin_without_tui(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    log = tmp_path / "logs.jsonl"
    events = EventSink(log)
    monkeypatch.setattr(interactmod, "frontend_is_live", _dead)
    monkeypatch.setattr(interactmod, "_has_controlling_tty", _tty)  # foreground
    monkeypatch.setattr(interactmod, "default_stdin_approver", _stdin_no)
    approve = _prompts(tmp_path, events).approve
    assert approve("x", scope=COMMAND_SCOPE) is False
    assert _events_of(log, "approval.answer")[0]["source"] == "stdin"


def test_approver_headless_no_frontend_waits_not_denies(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # No terminal, no away-mode, no front-end attached right now: the run
    # WAITS for a front-end rather than denying (the ruled default: a park
    # spends nothing and stays answerable late via attach; deny would let the
    # run burn tokens on a path it may not be able to finish). A context that
    # can never be attended declares AGENT6_DETACHED_AWAY=deny (the bench
    # containers do). A writer thread attaches + answers after a beat.
    import threading
    import time

    from agent6.sessions.ipc import register_frontend, write_answer

    log = tmp_path / "logs.jsonl"
    events = EventSink(log)
    # Use the REAL frontend_is_live: nothing is attached at approve() time (the
    # writer sleeps first), so the approver reaches the wait path; once the
    # writer registers its front-end claim, the wait picks up its answer.
    monkeypatch.setattr(interactmod, "_has_controlling_tty", lambda: False)  # headless
    monkeypatch.setattr(interactmod, "default_stdin_approver", _stdin_forbidden)  # never stdin

    def attach_and_answer() -> None:
        time.sleep(0.3)
        register_frontend(tmp_path, os.getpid())
        write_answer(tmp_path, "approval-1", "yes")

    threading.Thread(target=attach_and_answer, daemon=True).start()
    approve = _prompts(tmp_path, events).approve
    assert approve("rm -rf build", scope=COMMAND_SCOPE) is True
    assert _events_of(log, "approval.answer")[0]["source"] == "await-frontend"


def test_approver_session_allows_every_later_command(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # "allow session" (stdin returns "session") approves this command AND every
    # later one without prompting again -- across the run.
    log = tmp_path / "logs.jsonl"
    events = EventSink(log)
    monkeypatch.setattr(interactmod, "frontend_is_live", _dead)
    monkeypatch.setattr(interactmod, "_has_controlling_tty", _tty)  # foreground
    monkeypatch.setattr(interactmod, "default_stdin_approver", _stdin_session)
    approve = _prompts(tmp_path, events).approve
    assert approve("first?", scope=COMMAND_SCOPE) is True
    # A second prompt must NOT reach the stdin approver -- the session marker auto-passes.
    monkeypatch.setattr(interactmod, "default_stdin_approver", _stdin_forbidden)
    assert approve("second?", scope=COMMAND_SCOPE) is True
    assert _events_of(log, "approval.answer")[-1]["source"] == "session"


def test_approver_tui_timeout_falls_back_to_stdin(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    log = tmp_path / "logs.jsonl"
    events = EventSink(log)
    monkeypatch.setattr(interactmod, "frontend_is_live", _live)
    monkeypatch.setattr(interactmod, "read_answer", _ans_none)  # TUI died / timed out
    monkeypatch.setattr(interactmod, "_has_controlling_tty", _tty)  # foreground
    monkeypatch.setattr(interactmod, "default_stdin_approver", _stdin_yes)
    approve = _prompts(tmp_path, events).approve
    assert approve("x", scope=COMMAND_SCOPE) is True
    assert _events_of(log, "approval.answer")[0]["source"] == "stdin"


class _FakeStdout:
    def __init__(self, *, tty: bool) -> None:
        self._tty = tty

    def isatty(self) -> bool:
        return self._tty


def _yes() -> bool:
    return True


def _no() -> bool:
    return False


def test_should_spawn_tui_gating(monkeypatch: pytest.MonkeyPatch) -> None:
    def should(**kw: Any) -> bool:
        return livemod.should_spawn_tui(**kw)

    monkeypatch.setattr(livemod, "_tui_available", _yes)
    monkeypatch.setattr(livemod.sys, "stdout", _FakeStdout(tty=True))
    # Headless by default: no --tui -> never spawn.
    assert should(tui=False, interactive=False, mode="run") is False
    # --tui on a TTY with textual + run mode -> spawn.
    assert should(tui=True, interactive=False, mode="run") is True
    # --tui asked for but can't honour -> warn and stay headless.
    assert should(tui=True, interactive=True, mode="run") is False
    # A planning run opens the same view (the hub already views plans there);
    # an ask stays text: its answer is the deliverable.
    assert should(tui=True, interactive=False, mode="plan") is True
    assert should(tui=True, interactive=False, mode="ask") is False
    # textual not installed.
    monkeypatch.setattr(livemod, "_tui_available", _no)
    assert should(tui=True, interactive=False, mode="run") is False
    # non-TTY (benches / CI / pipes).
    monkeypatch.setattr(livemod, "_tui_available", _yes)
    monkeypatch.setattr(livemod.sys, "stdout", _FakeStdout(tty=False))
    assert should(tui=True, interactive=False, mode="run") is False


def test_stream_modes(monkeypatch: pytest.MonkeyPatch) -> None:
    def modes(*, tui_enabled: bool) -> tuple[bool, bool]:
        return livemod.stream_modes(tui_enabled=tui_enabled)

    monkeypatch.delenv("AGENT6_FORCE_STREAM", raising=False)
    monkeypatch.delenv("AGENT6_STREAM_TO_LOG", raising=False)

    # Headless, no env: the audited non-streaming path, no console echo.
    monkeypatch.setattr(livemod.sys, "stderr", _FakeStdout(tty=False))
    assert modes(tui_enabled=False) == (False, False)

    # Interactive stderr TTY: stream; echo only when the TUI does NOT own the term.
    monkeypatch.setattr(livemod.sys, "stderr", _FakeStdout(tty=True))
    assert modes(tui_enabled=False) == (True, True)  # plain ask/plan
    assert modes(tui_enabled=True) == (True, False)  # the TUI renders the deltas

    # AGENT6_FORCE_STREAM (bench/CI): stream AND echo even when headless.
    monkeypatch.setattr(livemod.sys, "stderr", _FakeStdout(tty=False))
    monkeypatch.setenv("AGENT6_FORCE_STREAM", "1")
    assert modes(tui_enabled=False) == (True, True)
    monkeypatch.delenv("AGENT6_FORCE_STREAM")

    # AGENT6_STREAM_TO_LOG (hub-watched headless run): emit the delta EVENTS only,
    # NO console echo -- the dashboard renders them; the stderr temp is discarded.
    monkeypatch.setenv("AGENT6_STREAM_TO_LOG", "1")
    assert modes(tui_enabled=False) == (True, False)


def test_tui_session_disabled_is_noop(tmp_path: Path) -> None:
    # enabled=False must not spawn anything or touch stdout.
    with livemod.tui_session(tmp_path, enabled=False):
        pass
    assert not (tmp_path / "tui_console.log").exists()


def test_spawned_away_default_sets_wait_from_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A front-end launcher (web/TUI hub) sets AGENT6_DETACHED_AWAY so a spawned,
    # terminal-less run WAITS for a viewer instead of fabricating empty answers.
    from agent6.app.frontend import apply_spawned_away_default
    from agent6.sessions.ipc import away_mode

    monkeypatch.setenv("AGENT6_DETACHED_AWAY", "wait")
    apply_spawned_away_default(tmp_path, (COMMAND_SCOPE,))
    assert away_mode(tmp_path) == "wait"


def test_spawned_away_default_approve_reuses_session_allow(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """AGENT6_DETACHED_AWAY=approve maps to the session-allow marker, like the
    interactive detach prompt. Writing "approve" into away.mode put it outside
    the file's deny|wait vocabulary, so the reader fell into the wait branch and
    the spawn BLOCKED on every approval instead of approving."""
    from agent6.app.frontend import apply_spawned_away_default
    from agent6.sessions.ipc import COMMAND_SCOPE, away_mode, session_allow_set

    monkeypatch.setenv("AGENT6_DETACHED_AWAY", "approve")
    apply_spawned_away_default(tmp_path, (COMMAND_SCOPE,))
    assert session_allow_set(tmp_path, COMMAND_SCOPE) is True
    assert away_mode(tmp_path) == ""  # approve is never stored in away.mode


def test_set_away_mode_rejects_values_outside_its_vocabulary(tmp_path: Path) -> None:
    # away.mode's contract is deny|wait; anything else must fail loudly at the
    # writer, never land on disk for readers to misinterpret.
    from agent6.sessions.ipc import set_away_mode

    with pytest.raises(ValueError, match="deny"):
        set_away_mode(tmp_path, "approve")


def test_spawned_away_default_is_noop_without_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A pure headless run (no launcher, no env) is untouched, keeping its
    # non-hanging default so CI never blocks on an unanswerable question.
    from agent6.app.frontend import apply_spawned_away_default
    from agent6.sessions.ipc import away_mode

    monkeypatch.delenv("AGENT6_DETACHED_AWAY", raising=False)
    apply_spawned_away_default(tmp_path, (COMMAND_SCOPE,))
    assert away_mode(tmp_path) == ""


def test_approver_away_deny_auto_denies(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Detach chose "deny all": every run_command is denied without prompting.
    from agent6.sessions.ipc import set_away_mode

    log = tmp_path / "logs.jsonl"
    events = EventSink(log)
    monkeypatch.setattr(interactmod, "default_stdin_approver", _stdin_forbidden)  # must NOT prompt
    set_away_mode(tmp_path, "deny")
    approve = _prompts(tmp_path, events).approve
    assert approve("rm -rf /", scope=COMMAND_SCOPE) is False
    assert _events_of(log, "approval.answer")[0]["source"] == "away-deny"


def test_approver_live_front_end_wins_over_away_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A live front-end (a re-attached watch/TUI/web) is always asked, in its own
    # UI, regardless of the detach away-mode -- away-mode governs only the window
    # when nothing is attached. Even under away="deny", a live front-end answers.
    from agent6.sessions.ipc import set_away_mode

    log = tmp_path / "logs.jsonl"
    events = EventSink(log)
    monkeypatch.setattr(interactmod, "frontend_is_live", _live)  # a front-end is attached
    monkeypatch.setattr(interactmod, "read_answer", _ans_yes)  # and it approved
    monkeypatch.setattr(interactmod, "default_stdin_approver", _stdin_forbidden)  # no stdin fall
    set_away_mode(tmp_path, "deny")  # would deny if the front-end did NOT win
    approve = _prompts(tmp_path, events).approve
    assert (
        approve("ls", scope=COMMAND_SCOPE) is True
    )  # the attached front-end approved despite away=deny
    assert _events_of(log, "approval.answer")[0]["source"] == "frontend"


def test_approver_away_wait_blocks_for_a_front_end_when_none_attached(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # away="wait" with NOTHING attached: block until a front-end attaches and
    # answers. A writer thread attaches (a front-end claim) + answers after a beat.
    import threading
    import time

    from agent6.sessions.ipc import register_frontend, set_away_mode, write_answer

    log = tmp_path / "logs.jsonl"
    events = EventSink(log)
    monkeypatch.setattr(interactmod, "default_stdin_approver", _stdin_forbidden)  # never stdin
    set_away_mode(tmp_path, "wait")

    def attach_and_answer() -> None:
        time.sleep(0.3)
        register_frontend(tmp_path, os.getpid())  # a front-end re-attaches
        write_answer(tmp_path, "approval-1", "yes")  # and answers

    threading.Thread(target=attach_and_answer, daemon=True).start()
    approve = _prompts(tmp_path, events).approve
    assert approve("ls", scope=COMMAND_SCOPE) is True
    assert _events_of(log, "approval.answer")[0]["source"] == "await-frontend"


def test_a_stop_request_ends_an_away_wait(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """`sessions stop` drops the stop marker; a run blocked in an away-wait
    (a pre-start question, an approval nobody attached to answer) has no step
    to stop after, so the marker breaks the wait: the questioner returns empty
    answers and the run parks or denies instead of waiting on forever."""
    import threading
    import time

    from agent6.sessions.ipc import request_stop, set_away_mode

    log = tmp_path / "logs.jsonl"
    events = EventSink(log)
    monkeypatch.setattr(interactmod, "default_stdin_approver", _stdin_forbidden)
    set_away_mode(tmp_path, "wait")

    def stop_it() -> None:
        time.sleep(0.3)
        request_stop(tmp_path)

    threading.Thread(target=stop_it, daemon=True).start()
    ask = _prompts(tmp_path, events).ask
    started = time.monotonic()
    answers = ask((UserQuestion(question="stash?", options=("stash", "cancel")),))
    assert answers == ("",)
    assert time.monotonic() - started < 10


def test_spawned_away_default_does_not_overwrite_the_operators_choice(tmp_path: Path) -> None:
    """On detach the operator picks the while-away policy ('deny' stops
    run_commands until they reattach), then the background resume is spawned
    with AGENT6_DETACHED_AWAY=wait. Applying that as a DEFAULT must not clobber
    the explicit choice -- it silently upgraded 'deny' to 'wait', so the run sat
    blocked on an approval nobody was there to give instead of denying and
    carrying on."""
    import os

    from agent6.app.frontend import apply_spawned_away_default
    from agent6.sessions.ipc import away_mode, set_away_mode

    session_dir = tmp_path / "run"
    session_dir.mkdir()
    set_away_mode(session_dir, "deny")  # the operator's detach answer
    old = os.environ.get("AGENT6_DETACHED_AWAY")
    os.environ["AGENT6_DETACHED_AWAY"] = "wait"  # what the spawned resume carries
    try:
        apply_spawned_away_default(session_dir, (COMMAND_SCOPE,))
        assert away_mode(session_dir) == "deny"
        # With nothing chosen, the launcher's default still applies.
        other = tmp_path / "other"
        other.mkdir()
        apply_spawned_away_default(other, (COMMAND_SCOPE,))
        assert away_mode(other) == "wait"
    finally:
        if old is None:
            del os.environ["AGENT6_DETACHED_AWAY"]
        else:
            os.environ["AGENT6_DETACHED_AWAY"] = old


def test_approver_wait_consumes_a_claimless_answer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The web UI writes answer files without ever registering a front-end
    claim. The wait loop gated READING on a live claim, so a web-answered
    approval sat unconsumed while the run waited out its whole window
    (caught live: a web-spawned run wedged on an "answered" approval)."""
    import threading
    import time

    from agent6.sessions.ipc import write_answer, write_steer_answer

    log = tmp_path / "logs.jsonl"
    events = EventSink(log)
    monkeypatch.setattr(interactmod, "_has_controlling_tty", lambda: False)
    monkeypatch.setattr(interactmod, "default_stdin_approver", _stdin_forbidden)

    def answer_never_claiming() -> None:
        time.sleep(0.3)
        write_answer(tmp_path, "approval-1", "yes")  # no register_frontend

    def abort_if_wedged() -> None:
        # Pre-fix the wait loop never reads a claim-less answer; break it so
        # the test fails fast instead of hanging the suite.
        time.sleep(15)
        write_steer_answer(tmp_path, "abort")

    threading.Thread(target=answer_never_claiming, daemon=True).start()
    threading.Thread(target=abort_if_wedged, daemon=True).start()
    approve = _prompts(tmp_path, events).approve
    assert approve("run_verify_command", scope=COMMAND_SCOPE) is True
    assert _events_of(log, "approval.answer")[0]["source"] == "await-frontend"


def test_stdin_approver_renders_the_command_on_its_own_lines(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The approval prompt glued the command to the question on one line; a
    long argv wrapped into the [y/N/a/d] suffix and the input point drowned.
    The payload renders indented on its own lines with a blank line before
    the answer line."""
    from agent6.ui.cli import _interact as interactmod

    seen: list[str] = []
    plains: list[object] = []

    def _capture(rendered: str, **kw: object) -> str:
        seen.append(rendered)
        plains.append(kw.get("plain"))
        return "y"

    monkeypatch.setattr(interactmod, "tty_prompt", _capture)
    assert interactmod.default_stdin_approver("Allow run_command: git log --stat -5") == "yes"
    rendered = seen[0]
    plain = re.sub(r"\x1b\[[0-9;]*m", "", rendered)
    assert plain.startswith("? Allow run_command:\n\n    git log --stat -5\n\n  [y/N/a/d]")
    # The console vocabulary: a bold yellow ? marks the question.
    assert "\x1b[1m\x1b[33m?" in rendered
    # The stdin fallback (stdout may be a pipe) gets the same text unstyled.
    assert plains[0] == plain
    # A prompt without the "<head>: <payload>" shape keeps the one-line form.
    seen.clear()
    interactmod.default_stdin_approver("Proceed?", standing=False)
    assert seen[0] == "Proceed? [y/N]: "


def test_approval_with_a_pause_armed_opens_the_menu_after_the_answer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An operator prompt counts as a Ctrl-C boundary: with a pause armed, the
    approval prompt says so and the pause menu runs right after the answer
    (its action seeds the steer the next boundary consumes). Before, the armed
    pause waited out the rest of the step in silence."""
    from agent6.events import EventSink
    from agent6.ui.cli import _interact as interactmod
    from agent6.ui.steer import SteerState

    events = EventSink(tmp_path / "logs.jsonl")
    notices: list[str] = []
    calls: list[str] = []

    def _steer(armed: bool) -> SteerState:
        return SteerState(
            requested=lambda: armed,
            clear=lambda: None,
            prompt=lambda: None,
            restore=lambda: None,
            abort_pending=lambda: False,
            interrupt=lambda: False,
            reset_stage=lambda: None,
            armed=lambda: armed,
            prompt_now=lambda: calls.append("menu"),
        )

    def _not_live(_d: Path) -> bool:
        return False

    def _no_away(_d: Path) -> str | None:
        return None

    def _approve_yes(_p: str, **_k: object) -> str:
        return "yes"

    monkeypatch.setattr(interactmod, "frontend_is_live", _not_live)
    monkeypatch.setattr(interactmod, "away_mode", _no_away)
    monkeypatch.setattr(interactmod, "_has_controlling_tty", lambda: True)
    monkeypatch.setattr(interactmod, "tty_message", notices.append)
    monkeypatch.setattr(interactmod, "default_stdin_approver", _approve_yes)

    approve = _prompts(tmp_path, events, [_steer(True)]).approve
    assert approve("Allow run_command: ls", scope="command") is True
    assert calls == ["menu"]
    assert any("pause armed" in n for n in notices)

    calls.clear()
    notices.clear()
    approve = _prompts(tmp_path, events, [_steer(False)]).approve
    assert approve("Allow run_command: ls", scope="command") is True
    assert calls == [] and notices == []


def test_the_prompts_pause_a_console_view_attached_after_they_were_built(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The lifecycle builds the gate before the leg attaches the live console
    view, so the approver and questioner read the view at prompt time: with it
    captured at build time they paused nothing, and the heartbeat's per-tick
    line-erase wiped the tty prompt and the operator's keystrokes."""
    from agent6.ui.cli._console_view import ConsoleView
    from agent6.ui.cli.run import session_frontend

    paused: list[ConsoleView] = []
    real_pause = ConsoleView.pause

    def _pause(self: ConsoleView) -> Any:
        paused.append(self)
        return real_pause(self)

    monkeypatch.setattr(ConsoleView, "pause", _pause)
    monkeypatch.setattr(interactmod, "frontend_is_live", _dead)
    monkeypatch.setattr(interactmod, "_has_controlling_tty", _tty)
    monkeypatch.setattr(interactmod, "default_stdin_approver", _stdin_yes)

    def _first(_q: tuple[UserQuestion, ...], **_k: object) -> tuple[str, ...]:
        return ("a",)

    monkeypatch.setattr(interactmod, "default_stdin_questioner", _first)
    fe = session_frontend()
    events = EventSink(tmp_path / "logs.jsonl")
    prompts = OperatorPrompts(
        approver=fe.build_approver(tmp_path),
        questioner=fe.build_questioner(tmp_path),
        journal=events.emit,
        session_dir=tmp_path,
    )
    fe.attach_console_view(events)  # the leg attaches the view after the gate exists
    try:
        assert prompts.approve("Allow run_command: ls", scope=COMMAND_SCOPE) is True
        assert prompts.ask((UserQuestion(question="pick?", options=("a", "b")),)) == ("a",)
    finally:
        fe.close_console_view()
    assert len(paused) == 2, "both prompts pause the view the leg attached"
