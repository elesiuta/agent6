# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Every line the CLI prints passes one scrubber at the terminal seam."""

from __future__ import annotations

import io
import json
import os
import pty
import select
import sys
from pathlib import Path

import pytest

from agent6.sessions.layout import bucket_dir
from agent6.ui.cli import cli_main
from agent6.ui.cli._console_view import ConsoleView
from agent6.ui.cli._steer import tty_message, tty_prompt
from agent6.ui.cli._terminal_guard import ScrubbedStream, guarded_terminal, raw_stream

OSC52 = "\x1b]52;c;aGVsbG8=\x07"


class _Tty(io.StringIO):
    def isatty(self) -> bool:
        return True


def test_the_stream_drops_foreign_controls_and_keeps_the_clis_own() -> None:
    raw = _Tty()
    out = ScrubbedStream(raw)
    text = f"a{OSC52}b \x1b[2mdim\x1b[0m \r\x1b[2Kspin \x1b[6n cr\rx \x1b[8mhid\x1b[1;08m\x1b[0m\n"
    assert out.write(text) == len(raw.getvalue())
    assert raw.getvalue() == "ab \x1b[2mdim\x1b[0m spin  crx hid\x1b[0m\n"
    out.writelines([f"1{OSC52}", "2\n"])
    assert raw.getvalue().endswith("12\n")
    assert out.isatty() is True and raw_stream(out) is raw


def test_the_spinner_erases_its_line_under_the_wrapper() -> None:
    """The erase idiom was allowlisted for the spinner, so a file name carrying
    it forged the line it sat on; the spinner writes it under the wrapper."""
    raw = io.StringIO()
    view = ConsoleView(ScrubbedStream(raw), color=False)  # type: ignore[arg-type]
    view._status_active = True  # pyright: ignore[reportPrivateUsage]
    view._clear_status()  # pyright: ignore[reportPrivateUsage]
    assert raw.getvalue() == "\r\x1b[2K"


def test_the_guard_wraps_both_streams_for_the_block(capsys: pytest.CaptureFixture[str]) -> None:
    with guarded_terminal():
        print(f"out{OSC52}", end="")
        print(f"err{OSC52}", end="", file=sys.stderr)
    print("after", end="")
    captured = capsys.readouterr()
    assert captured.out == "outafter" and captured.err == "err"


def test_the_tty_writers_scrub_like_stdout(monkeypatch: pytest.MonkeyPatch) -> None:
    """`tty_message` and `tty_prompt` reach the controlling terminal past the
    wrapped streams; each write passes the same scrubber."""
    master, slave = pty.openpty()
    try:
        monkeypatch.setattr("agent6.ui.cli._steer.TTY_PATH", os.ttyname(slave))
        tty_message(f"[agent6] {OSC52}note\n")
        assert tty_prompt(f"Allow {OSC52}rm? \x1b[2m[y/N]\x1b[0m: ", until=lambda: True) is None
        seen = b""
        while select.select([master], [], [], 0.5)[0]:
            seen += os.read(master, 4096)
    finally:
        os.close(master)
        os.close(slave)
    text = seen.decode()
    assert "[agent6] note" in text and "Allow rm? \x1b[2m[y/N]\x1b[0m: " in text
    assert "\x1b]" not in text and "\x07" not in text


def _session_with_task(root: Path, task: str) -> None:
    from agent6.paths import state_dir

    session = bucket_dir(state_dir(root), "runs") / "brave-oak-AAAAAA"
    session.mkdir(parents=True)
    (session / "manifest.json").write_text(
        json.dumps(
            {"version": 3, "session_id": "brave-oak-AAAAAA", "mode": "run", "user_task": task}
        ),
        encoding="utf-8",
    )
    (session / "logs.jsonl").write_text(
        json.dumps({"type": "session.start", "mode": "run", "user_task": task}) + "\n",
        encoding="utf-8",
    )


def test_a_task_a_model_wrote_cannot_drive_the_terminal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A run's task can come from a plan the model wrote, a file name from a
    command inside the jail, a subject from a commit the model made; every
    listing printed them raw, and a terminal obeys an OSC 52 wherever it sits
    in a line (a clipboard write, then a paste). The seam is the process's
    stdout and stderr, not each print."""
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    monkeypatch.chdir(tmp_path)
    _session_with_task(tmp_path, f"rename the {OSC52}widget")
    assert cli_main(["sessions", "show", "brave-oak-AAAAAA"]) == 0
    out = capsys.readouterr().out
    assert "rename the widget" in out
    assert "\x1b]" not in out and "\x07" not in out


def test_a_crash_report_naming_model_text_prints_scrubbed(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The handlers that print a refusal or a crash report run under the
    guard: an exception quoting a name a command chose reaches stderr
    without its sequence."""

    def crash(_argv: list[str] | None = None) -> int:
        raise RuntimeError(f"file {OSC52}gone")

    monkeypatch.setattr("agent6.ui.cli.main", crash)
    monkeypatch.delenv("AGENT6_DEBUG", raising=False)
    assert cli_main(["sessions", "dir"]) == 1
    err = capsys.readouterr().err
    assert "unexpected RuntimeError: file gone" in err
    assert "\x1b]" not in err and "\x07" not in err
