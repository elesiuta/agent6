# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The `/btw` runner returns at once and delivers the answer later."""

from __future__ import annotations

import io
import json
import time
from pathlib import Path

import pytest

from agent6.events import EventSink
from agent6.ui.btw import make_btw_runner
from agent6.ui.cli._console_view import ConsoleView
from agent6.ui.cli._steer_menu import (
    MENU_COMMANDS,
    _run_info_command,  # pyright: ignore[reportPrivateUsage]
)


def _answered_ask(root: Path, name: str, answer: str) -> Path:
    d = root / name
    d.mkdir(parents=True)
    (d / "manifest.json").write_text(json.dumps({"version": 3, "mode": "ask"}), encoding="utf-8")
    (d / "logs.jsonl").write_text(
        "\n".join(
            [
                json.dumps({"type": "session.start", "user_task": "q"}),
                json.dumps({"type": "role.result", "text": answer}),
                json.dumps({"type": "session.end", "reason": "answered", "all_passed": True}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return d


def test_the_run_is_never_blocked_and_the_answer_arrives_later(tmp_path: Path) -> None:
    """The point of asking beside a run: `/btw` returns immediately, and the
    answer lands at the next turn boundary."""
    asks = tmp_path / "sessions" / "asks"
    asks.mkdir(parents=True)
    out = io.StringIO()
    view = ConsoleView(out, color=False)
    events = EventSink(tmp_path / "logs.jsonl")
    events.subscribe(view.feed)

    def launch(cwd: Path, argv: list[str], env: dict[str, str]) -> str:
        _answered_ask(asks, "quiet-fox-AAAAAA", "use ffmpeg -c:v libx265")
        return ""

    runner = make_btw_runner(
        "parent-BBBBBB",
        launch=launch,
        list_asks=lambda: [d for d in asks.iterdir() if d.is_dir()],
        events=events,
    )
    started = time.monotonic()
    _opened, line = runner("why h265", tmp_path)
    assert time.monotonic() - started < 2.0  # returned, did not wait for an answer
    assert "quiet-fox-AAAAAA" in line

    deadline = time.monotonic() + 15.0
    while "libx265" not in out.getvalue() and time.monotonic() < deadline:
        view.feed({"type": "role.result"})  # turn boundaries keep arriving
        time.sleep(0.2)
    text = out.getvalue()
    assert "--- btw: why h265" in text
    assert "use ffmpeg -c:v libx265" in text
    assert "agent6 resume quiet-fox-AAAAAA" in text


def test_an_answer_survives_a_surface_that_cannot_print_it(tmp_path: Path) -> None:
    """Handed straight to the console view, a btw answer was DROPPED under
    --tui and the web (there is none) and lost when the parent exited first --
    after the model had already been paid for. The journal is where every
    surface reads, and it outlives the process."""
    import json as _json

    asks = tmp_path / "sessions" / "asks"
    asks.mkdir(parents=True)
    events = EventSink(tmp_path / "logs.jsonl")

    def launch(cwd: Path, argv: list[str], env: dict[str, str]) -> str:
        _answered_ask(asks, "quiet-fox-AAAAAA", "use ffmpeg")
        return ""

    runner = make_btw_runner(
        "parent-BBBBBB",
        launch=launch,
        list_asks=lambda: [d for d in asks.iterdir() if d.is_dir()],
        events=events,  # no view at all
    )
    runner("why h265", tmp_path)
    deadline = time.monotonic() + 15.0
    answered: dict[str, object] = {}
    while not answered and time.monotonic() < deadline:
        for line in (tmp_path / "logs.jsonl").read_text(encoding="utf-8").splitlines():
            event = _json.loads(line)
            if event["type"] == "btw.answered":
                answered = event
        time.sleep(0.2)
    assert answered.get("btw_id") == "quiet-fox-AAAAAA"
    assert "use ffmpeg" in str(answered.get("block"))


def test_btw_is_offered_in_the_menu() -> None:
    assert "/btw" in MENU_COMMANDS


def test_an_unwired_btw_says_so_rather_than_failing_obscurely(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A detached run has no console view to print an answer into."""
    _run_info_command("/btw why", tmp_path, None)
    assert "needs a live run" in capsys.readouterr().out


def test_a_bare_btw_asks_for_a_question(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Opening an empty session would be worse than saying nothing was asked."""
    _run_info_command("/btw", tmp_path, None)
    assert "ask something" in capsys.readouterr().out


def test_a_btw_with_a_question_reaches_the_runner_and_never_the_loop(tmp_path: Path) -> None:
    """The bug that made /btw dead: the menu special-cased only /compact and
    skills for lines WITH arguments, so `/btw why...` fell through and was
    returned as STEER TEXT -- sent to the loop, which is exactly what a btw
    must never be."""
    from agent6.ui.cli._steer_menu import pause_menu

    asked: list[str] = []

    def runner(question: str, session_dir: Path) -> tuple[bool, str]:
        asked.append(question)
        return True, "[agent6] btw opened"

    lines = iter(["/btw why is the broker slow?", "/continue"])
    action = pause_menu(tmp_path, input_fn=lambda _p: next(lines), btw_runner=runner)
    assert asked == ["why is the broker slow?"]
    assert action == "", "a btw must not become a steer instruction"


def test_ordinary_text_is_still_a_steer(tmp_path: Path) -> None:
    from agent6.ui.cli._steer_menu import pause_menu

    lines = iter(["make it faster"])
    assert pause_menu(tmp_path, input_fn=lambda _p: next(lines)) == "make it faster"


def test_a_btw_answer_renders_in_the_shared_fold(tmp_path: Path) -> None:
    """The fold is what the TUI and the web render from; without it the answer
    reached the journal and still showed nowhere but the CLI."""
    from agent6.viewmodel.transcript import TranscriptFold

    fold = TranscriptFold()
    fold.feed({"type": "role.text_delta", "text": "working"})
    items = fold.feed({"type": "btw.answered", "btw_id": "x", "block": "--- btw: why\nbecause"})
    kinds = [i.kind for i in items]
    assert kinds == ["text", "marker"], "a btw must not join the turn's own prose"
    assert "because" in items[-1].body


def test_btw_is_not_offered_where_nothing_can_spawn_it(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """It was listed on every surface and answered "needs a live run" only once
    the operator had typed it. A surface that knows what it cannot do never
    offers it."""
    from agent6.ui.cli._steer_menu import _run_info_command  # pyright: ignore[reportPrivateUsage]

    _run_info_command("/help", tmp_path, None)
    assert "/btw" not in capsys.readouterr().out

    _run_info_command("/help", tmp_path, lambda _q, _d: (True, ""))
    assert "/btw" in capsys.readouterr().out


def test_open_btw_serves_every_composer(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """`/btw` from the TUI or web composer opens the same side ask the CLI
    menu does, with the run's own journal as the answer channel; a bare
    `/btw` is told what to type."""
    import agent6.ui.btw as btw_mod

    session_dir = tmp_path / "sessions" / "runs" / "parent-BBBBBB"
    session_dir.mkdir(parents=True)
    (session_dir / "logs.jsonl").write_text("", encoding="utf-8")

    def launch(cwd: Path, argv: list[str], env: dict[str, str]) -> str:
        _answered_ask(tmp_path / "sessions" / "asks", "quiet-fox-CCCCCC", "yes")
        return ""

    monkeypatch.setattr(btw_mod, "direct_launch", launch)
    assert btw_mod.open_btw(session_dir, "") == (False, "[agent6] ask something: `/btw <question>`")
    opened, line = btw_mod.open_btw(session_dir, "is it safe?")
    assert opened and "quiet-fox-CCCCCC opened" in line
    deadline = time.monotonic() + 5
    log = ""
    while time.monotonic() < deadline:
        log = (session_dir / "logs.jsonl").read_text(encoding="utf-8")
        if "btw.answered" in log:
            break
        time.sleep(0.05)
    assert '"btw.opened"' in log and '"btw.answered"' in log and "yes" in log
