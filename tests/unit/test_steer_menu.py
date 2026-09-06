# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The mid-run Ctrl-C menu maps operator input to a canonical steer action."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from agent6.ui.cli._steer_menu import pause_line


def test_both_prompts_answer_a_line_the_same_way(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The plain prompt took bare words (`q`, `exit`, `d`) the rich menu had
    already dropped, then a slash table of its own that swallowed `/parallel`
    and sent `/shells` to the model: one dispatcher answers a typed line at
    both prompts, and the plain one continues where the menu asks again."""
    assert pause_line("/stop", tmp_path) == "abort"
    assert pause_line(" /EXIT ", tmp_path) == "exit"
    assert pause_line("/detach", tmp_path) == "detach"
    assert pause_line("/continue", tmp_path) == ""
    assert pause_line("/undo", tmp_path) == "/undo"
    # The loop's directives travel verbatim, arguments and all, the word
    # lowercased for its case-sensitive parsers.
    assert pause_line("/parallel 2 fix the tests", tmp_path) == "/parallel 2 fix the tests"
    assert pause_line("/PIN always gate", tmp_path) == "/pin always gate"
    # Any other line with spaces travels verbatim, case and spacing included.
    assert pause_line("/Users/eric/Notes.md   has it", tmp_path) == "/Users/eric/Notes.md   has it"
    assert pause_line("/h check the logs", tmp_path) == "/h check the logs"
    # A command that prints continues the run; an unknown one says so.
    assert pause_line("/shells", tmp_path) == ""
    assert pause_line("/statsu", tmp_path) == ""
    out = capsys.readouterr().out
    assert "background commands" in out and "unknown command '/statsu'" in out
    for word in ("q", "Q", "quit", "stop", "abort", "d", "detach", "exit"):
        assert pause_line(word, tmp_path) == word


def test_blank_continues(tmp_path: Path) -> None:
    assert pause_line("", tmp_path) == ""
    assert pause_line("   ", tmp_path) == ""


def test_none_stays_none(tmp_path: Path) -> None:
    assert pause_line(None, tmp_path) is None


def test_instruction_passes_through(tmp_path: Path) -> None:
    assert pause_line("focus on the parser", tmp_path) == "focus on the parser"
    # a sentence that merely starts with a keyword is an instruction, not a command
    assert pause_line("abort the current plan", tmp_path) == "abort the current plan"


def _feed(lines: list[str]) -> Callable[[str], str]:
    """An input_fn that replays *lines* then raises EOF (menu -> continue)."""
    it = iter(lines)

    def fn(_prompt: str) -> str:
        try:
            return next(it)
        except StopIteration:
            raise EOFError from None

    return fn


def test_pause_menu_slash_commands(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Info commands print and re-prompt; action commands return the canonical
    steer values; free text passes through as the instruction."""
    import json
    import os

    from agent6.ui.cli._steer_menu import pause_menu

    # The menu IS Ctrl-C on a live attach, so the worker is alive: without its
    # pid on disk the status line reads the run as one that exited.
    (tmp_path / "worker.pid").write_text(str(os.getpid()), encoding="utf-8")
    (tmp_path / "logs.jsonl").write_text(
        "".join(
            json.dumps(e) + "\n"
            for e in (
                {"type": "session.start", "user_task": "polish the TUI", "mode": "run"},
                {
                    "type": "graph.update",
                    "cursor": "t1",
                    "nodes": {
                        "t1": {
                            "title": "fix the bars",
                            "parent_id": None,
                            "status": "in_progress",
                            "children": [],
                        }
                    },
                },
                {"type": "tool.call", "name": "read_file", "args": {"path": "a.py"}},
                {"type": "tool.result", "name": "read_file", "ok": True, "summary": "12 bytes"},
            )
        ),
        encoding="utf-8",
    )
    # /help + /status + /tasks print, then the free text is the steer.
    out = pause_menu(tmp_path, input_fn=_feed(["/help", "/status", "/tasks", "focus on tests"]))
    assert out == "focus on tests"
    printed = capsys.readouterr().out
    assert "/detach" in printed  # help listed the commands
    assert "running" in printed and "1 tool " in printed  # status line, singular at 1
    assert "fix the bars" in printed  # the task graph

    assert pause_menu(tmp_path, input_fn=_feed(["/stop"])) == "abort"
    assert pause_menu(tmp_path, input_fn=_feed(["/detach"])) == "detach"
    assert pause_menu(tmp_path, input_fn=_feed(["/continue"])) == ""
    # Bare keywords are gone: a plain word is a steering instruction now.
    assert pause_menu(tmp_path, input_fn=_feed(["q"])) == "q"
    # Unknown slash command re-prompts (does not steer with a typo).
    out = pause_menu(tmp_path, input_fn=_feed(["/statsu", "real steer"]))
    assert out == "real steer"
    assert "unknown command" in capsys.readouterr().out
    # EOF (Ctrl-D) means continue.
    assert pause_menu(tmp_path, input_fn=_feed([])) is None


def test_pause_menu_status_clips_the_task_like_every_listing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """/status ends a long task with an ellipsis and skips a seeded run's
    `<prior-run>` block, the shared snippet rule; a bare 80-char slice cut
    mid-word and read as the whole task."""
    import json
    import os

    from agent6.ui.cli._steer_menu import pause_menu

    (tmp_path / "worker.pid").write_text(str(os.getpid()), encoding="utf-8")
    task = "<prior-run>\nseeded context\n</prior-run>\n" + "add a function " * 12
    (tmp_path / "logs.jsonl").write_text(
        json.dumps({"type": "session.start", "user_task": task, "mode": "run"}) + "\n",
        encoding="utf-8",
    )
    pause_menu(tmp_path, input_fn=_feed(["/status"]))
    status = next(ln for ln in capsys.readouterr().out.splitlines() if "task:" in ln)
    shown = status.split("task: ", 1)[1]
    assert shown.endswith("…") and len(shown) == 80
    assert shown.startswith("add a function") and "<prior-run" not in shown


def test_pause_menu_help_names_parallel(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """The mid-run steer help names `/parallel`, the directive the loop dispatches
    sibling lanes for (see agent6.directive.parse_directive)."""
    from agent6.ui.cli._steer_menu import pause_menu

    (tmp_path / "logs.jsonl").write_text("", encoding="utf-8")
    assert pause_menu(tmp_path, input_fn=_feed(["/help", "go"])) == "go"
    assert "/parallel" in capsys.readouterr().out


def test_pause_menu_bare_parallel_explains_and_reprompts(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """`/parallel` is a menu command like the other directives (listed,
    completed from a unique prefix); bare, it names the missing task and
    re-prompts instead of reaching the loop as an empty directive."""
    from agent6.ui.cli._steer_menu import MENU_COMMANDS, pause_menu

    assert "/parallel" in MENU_COMMANDS
    (tmp_path / "logs.jsonl").write_text("", encoding="utf-8")
    assert pause_menu(tmp_path, input_fn=_feed(["/para", "go"])) == "go"
    assert "needs a task" in capsys.readouterr().out


def test_pause_menu_parallel_directive_passes_through_verbatim(tmp_path: Path) -> None:
    """`/parallel <task>` has a space, so the pause menu sends it to the run
    verbatim (the loop's _maybe_handle_steer parses it); it is never swallowed as
    a menu command. This is why mid-run `/parallel` needs no composer change."""
    from agent6.ui.cli._steer_menu import pause_menu

    (tmp_path / "logs.jsonl").write_text("", encoding="utf-8")
    assert pause_menu(tmp_path, input_fn=_feed(["/parallel 2 add a greeting"])) == (
        "/parallel 2 add a greeting"
    )


def test_pause_menu_prefixes_and_word_rule(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A unique prefix fires the command, an ambiguous one re-asks, and a line
    with spaces is always a steering instruction (no quoting needed)."""
    import json

    # A run mid-pause has session.start AND a live worker.pid on disk (the menu
    # is Ctrl-C on a live attach), which is what reads "running".
    import os

    from agent6.ui.cli._steer_menu import pause_menu

    (tmp_path / "worker.pid").write_text(str(os.getpid()), encoding="utf-8")
    (tmp_path / "logs.jsonl").write_text(
        json.dumps({"type": "session.start", "user_task": "t", "mode": "run"}) + "\n",
        encoding="utf-8",
    )
    # /sta is uniquely /status; /st matches /status and /stop -> re-ask.
    assert pause_menu(tmp_path, input_fn=_feed(["/sta", "/st", "/stop"])) == "abort"
    printed = capsys.readouterr().out
    assert "running" in printed  # /sta printed the status line
    assert "ambiguous" in printed and "/status" in printed and "/stop" in printed
    # A multi-word line starting with "/" is a steer, never a command.
    assert pause_menu(tmp_path, input_fn=_feed(["/stop hammering the API"])) == (
        "/stop hammering the API"
    )
    # /h is the /help alias.
    assert pause_menu(tmp_path, input_fn=_feed(["/h", "go"])) == "go"
    assert "/detach" in capsys.readouterr().out


def test_pause_menu_compact_requests_compaction(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from agent6.sessions.ipc import read_compact_request
    from agent6.ui.cli._steer_menu import pause_menu

    (tmp_path / "logs.jsonl").write_text("", encoding="utf-8")
    assert pause_menu(tmp_path, input_fn=_feed(["/compact"])) is None  # EOF -> continue
    assert read_compact_request(tmp_path) == ""  # marker pending, no focus
    assert "compaction requested" in capsys.readouterr().out


def test_pause_menu_status_tells_the_truth_about_a_dead_worker(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """/status in the pause menu of an ATTACHED run whose worker died prints
    the hub's word ('stale'), not 'running' -- the fold-only label sent the
    operator back to waiting on a run nothing was executing."""
    import json

    from agent6.ui.cli._steer_menu import pause_menu

    (tmp_path / "logs.jsonl").write_text(
        json.dumps({"type": "session.start", "user_task": "t", "mode": "run"}) + "\n",
        encoding="utf-8",
    )
    (tmp_path / "worker.pid").write_text("999999999", encoding="utf-8")  # dead
    assert pause_menu(tmp_path, input_fn=_feed(["/status", "/continue"])) == ""
    printed = capsys.readouterr().out
    assert "stale" in printed
    assert "running" not in printed


def test_pause_menu_status_shows_ctx_and_profile(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """/status includes the context fill (tokens + % of the model window) and
    the sandbox profile the run started with."""
    import json

    from agent6.ui.cli._steer_menu import pause_menu

    (tmp_path / "logs.jsonl").write_text(
        "".join(
            json.dumps(e) + "\n"
            for e in (
                {"type": "session.start", "user_task": "polish", "mode": "run"},
                {
                    "type": "role.call",
                    "role": "worker",
                    "provider": "anthropic",
                    "model": "claude-sonnet-4-5",
                },
                {"type": "role.result", "role": "worker", "tokens_in": 90_000, "tokens_out": 10},
            )
        ),
        encoding="utf-8",
    )
    (tmp_path / "manifest.json").write_text(
        json.dumps({"workflow": {"preset": "paranoid"}}), encoding="utf-8"
    )
    assert pause_menu(tmp_path, input_fn=_feed(["/status"])) is None
    printed = capsys.readouterr().out
    assert "ctx 90,000 tok" in printed
    assert "(45%)" in printed  # 90k of the 200k sonnet window
    assert "preset paranoid" in printed
    assert "elided" not in printed  # no compaction yet: no elision suffix


def test_pause_menu_status_shows_compaction_truth(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Once compaction has elided results, /status says how many left the
    model's context (and how many survive as distilled gists)."""
    import json

    from agent6.ui.cli._steer_menu import pause_menu

    (tmp_path / "logs.jsonl").write_text(
        "".join(
            json.dumps(e) + "\n"
            for e in (
                {"type": "session.start", "user_task": "polish", "mode": "run"},
                {"type": "loop.compact.dropped", "n": 9, "calls": ["read_file a.py"]},
                {
                    "type": "loop.compact.gists",
                    "gisted": 3,
                    "demoted": 0,
                    "paths": ["a.py", "b.py", "c.py"],
                    "demoted_paths": [],
                },
            )
        ),
        encoding="utf-8",
    )
    assert pause_menu(tmp_path, input_fn=_feed(["/status"])) is None
    printed = capsys.readouterr().out
    assert "elided 9 (3 gists)" in printed


# --- skill slash commands ----------------------------------------------------


def _skill_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *names: str) -> None:
    """Install fake skills into an isolated data dir and chdir to tmp."""
    monkeypatch.setenv("AGENT6_DATA_HOME", str(tmp_path / "data"))
    monkeypatch.setenv("AGENT6_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.setenv("AGENT6_STATE_HOME", str(tmp_path / "state"))
    monkeypatch.chdir(tmp_path)
    for name in names:
        d = tmp_path / "data" / "skills" / name
        d.mkdir(parents=True)
        (d / "SKILL.md").write_text(
            f"---\nname: {name}\ndescription: Use when testing {name}.\n---\n\nGRUNT {name}\n",
            encoding="utf-8",
        )


def test_skill_command_whole_line(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from agent6.ui.cli._steer_menu import pause_menu

    _skill_env(tmp_path, monkeypatch, "caveman")
    # The menu passes a skill command through as typed; the loop expands it
    # (one owner for every composer).
    assert pause_menu(tmp_path, input_fn=_feed(["/caveman"])) == "/caveman"


def test_skill_command_with_args(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from agent6.ui.cli._steer_menu import pause_menu

    _skill_env(tmp_path, monkeypatch, "caveman")
    assert pause_menu(tmp_path, input_fn=_feed(["/caveman lite"])) == "/caveman lite"


def test_non_skill_line_with_spaces_stays_verbatim(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from agent6.ui.cli._steer_menu import pause_menu

    _skill_env(tmp_path, monkeypatch, "caveman")
    assert pause_menu(tmp_path, input_fn=_feed(["/focus on tests"])) == "/focus on tests"
    assert pause_menu(tmp_path, input_fn=_feed(["fix the parser"])) == "fix the parser"


def test_builtin_wins_name_collision(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from agent6.ui.cli._steer_menu import pause_menu

    _skill_env(tmp_path, monkeypatch, "status")
    # /status must still be the built-in info command (prints, re-prompts, EOF)
    assert pause_menu(tmp_path, input_fn=_feed(["/status"])) is None


def test_disabled_skill_absent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from agent6.ui.cli._steer_menu import pause_menu

    _skill_env(tmp_path, monkeypatch, "caveman")
    (tmp_path / "config").mkdir(exist_ok=True)
    (tmp_path / "config" / "config.toml").write_text(
        '[skills.state]\ncaveman = "disabled"\n', encoding="utf-8"
    )
    out = pause_menu(tmp_path, input_fn=_feed(["/caveman", "steer text"]))
    # unknown command message printed, then the steer line is returned
    assert out == "steer text"


def test_skill_menu_table_lists_enabled(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from agent6.ui.cli._steer_menu import skill_menu_table

    _skill_env(tmp_path, monkeypatch, "caveman", "tidy")
    table = skill_menu_table()
    assert set(table) == {"/caveman", "/tidy"}
    assert table["/caveman"][0] == "Use when testing caveman."


def test_pause_menu_status_and_bare_pin_list_pins(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """/status counts pins; a bare /pin lists them with usage (the /pin <text>
    form is a steer directive the loop parses, so it must stay a verbatim steer)."""
    import json

    from agent6.ui.cli._steer_menu import pause_menu

    (tmp_path / "logs.jsonl").write_text(
        "".join(
            json.dumps(e) + "\n"
            for e in (
                {"type": "session.start", "user_task": "polish", "mode": "run"},
                {"type": "loop.pin.added", "text": "never touch schema", "chars": 18, "count": 1},
                {"type": "loop.pin.added", "text": "goal:\nship X", "chars": 12, "count": 2},
            )
        ),
        encoding="utf-8",
    )
    assert pause_menu(tmp_path, input_fn=_feed(["/status", "/pin", "/stop"])) == "abort"
    printed = capsys.readouterr().out
    assert "pins 2" in printed  # /status
    assert "1. never touch schema" in printed  # bare /pin lists them
    assert "2. goal:" in printed
    assert "/pin <text>" in printed  # usage line
    # /pin with text stays a verbatim steer for the loop's parser
    assert (
        pause_menu(tmp_path, input_fn=_feed(["/pin keep the API stable"]))
        == "/pin keep the API stable"
    )


def test_pause_menu_compact_accepts_focus(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """`/compact <focus>` routes to the compact request with the focus text;
    an ambiguous prefix with args stays a verbatim steer."""
    from agent6.sessions.ipc import read_compact_request
    from agent6.ui.cli._steer_menu import pause_menu

    (tmp_path / "logs.jsonl").write_text("", encoding="utf-8")
    assert pause_menu(tmp_path, input_fn=_feed(["/compact keep the auth decisions"])) is None
    assert read_compact_request(tmp_path) == "keep the auth decisions"
    assert "compaction requested" in capsys.readouterr().out
    # unique prefix with args routes too
    assert pause_menu(tmp_path, input_fn=_feed(["/comp focus on the parser"])) is None
    assert read_compact_request(tmp_path) == "focus on the parser"
    # /c is ambiguous (/compact, /continue): the line stays a steer
    assert pause_menu(tmp_path, input_fn=_feed(["/c keep it"])) == "/c keep it"
    # /pin with args is the loop's directive, never a menu route
    assert pause_menu(tmp_path, input_fn=_feed(["/pin keep it"])) == "/pin keep it"


def test_pause_menu_seeds_recall_from_the_journal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The Up/Ctrl-R history is seeded once per session from logs.jsonl (task,
    then steers, newlines flattened for the one-line reader), so recall spans
    process exits and other surfaces' steers; lines accepted in this process
    survive a later pause, and a different session reseeds."""
    import json

    from agent6.ui.cli import _steer_menu
    from agent6.ui.cli._steer_menu import pause_menu

    (tmp_path / "logs.jsonl").write_text(
        "".join(
            json.dumps(e) + "\n"
            for e in (
                {"type": "session.start", "user_task": "polish the\nTUI", "mode": "run"},
                {"type": "loop.steer.injected", "chars": 14, "text": "focus on tests"},
            )
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(_steer_menu, "_RECALL", _steer_menu._Recall())  # pyright: ignore[reportPrivateUsage]
    seen: list[list[str]] = []

    def fake_menu_input(
        prompt: str, commands: dict[str, str], history: list[str], **_kw: object
    ) -> str:
        seen.append(list(history))
        history.append("/status")  # what accepting a line does
        return "go"

    monkeypatch.setattr(_steer_menu, "menu_input", fake_menu_input)
    assert pause_menu(tmp_path) == "go"
    assert seen[0] == ["polish the TUI", "focus on tests"]
    # A later pause of the same session must not reseed away in-process lines.
    assert pause_menu(tmp_path) == "go"
    assert seen[1][-1] == "/status"
    # A different session dir reseeds.
    other = tmp_path / "other"
    other.mkdir()
    (other / "logs.jsonl").write_text(
        json.dumps({"type": "session.start", "user_task": "other task", "mode": "run"}) + "\n",
        encoding="utf-8",
    )
    assert pause_menu(other) == "go"
    assert seen[2] == ["other task"]


def test_ctrl_z_shows_status_and_cancels_an_armed_pause(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Ctrl-C's first stage now carries the facts a CLI operator cannot
    otherwise see, and Ctrl-Z shows the same line without arming anything --
    de-escalating an armed pause so checking on a run costs it nothing. It also
    replaces SIGTSTP's default: suspending a run mid-step would freeze it
    holding its worker lock and its egress broker."""
    import signal

    from agent6.app.frontend import SessionFacts
    from agent6.events import EventSink
    from agent6.ui.cli import _steer

    printed: list[str] = []
    monkeypatch.setattr(_steer, "tty_message", printed.append)
    monkeypatch.setattr(_steer, "frontend_is_live", lambda _d: False)  # type: ignore[misc]

    facts = SessionFacts(
        spend_usd=1.42,
        spend_partial=False,
        model="claude-sonnet-4-6",
        run_commands="ask",
        isolation="strict",
    )
    state = _steer.install_steer_sigint(
        EventSink(tmp_path / "logs.jsonl"), tmp_path, None, lambda: facts
    )
    try:
        sigint = signal.getsignal(signal.SIGINT)
        sigtstp = signal.getsignal(signal.SIGTSTP)
        assert callable(sigint) and callable(sigtstp)

        sigint(signal.SIGINT, None)  # first Ctrl-C: arm + show the facts
        assert state.requested() is True
        assert "$1.42 · claude-sonnet-4-6 · commands ask · strict" in printed[0]

        sigtstp(signal.SIGTSTP, None)  # Ctrl-Z: same facts, and stand down
        assert "$1.42" in printed[1] and "pause cancelled" in printed[1]

        # Disarmed: the next Ctrl-C arms again rather than escalating to an
        # interrupt the operator never asked for.
        printed.clear()
        sigint(signal.SIGINT, None)
        assert "pausing after this step" in printed[0]
    finally:
        state.restore()


def test_exit_maps_to_exit_and_stop_stays_abort(tmp_path: Path) -> None:
    """`/exit` is stop-AND-leave: the menu returns the distinct 'exit' action
    (the loop ends the run `steer_exit` and the CLI skips the follow-up
    prompt), while /stop keeps returning 'abort'. Before, an operator had to
    /stop and then type /exit at the "next:" prompt to actually leave."""
    from agent6.ui.cli._steer_menu import pause_menu

    assert pause_line("/exit", tmp_path) == "exit"
    # a sentence starting with the word stays an instruction
    assert pause_line("exit the retry loop early", tmp_path) == "exit the retry loop early"
    assert pause_menu(tmp_path, input_fn=_feed(["/exit"])) == "exit"
    assert pause_menu(tmp_path, input_fn=_feed(["/stop"])) == "abort"
