# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""A CLI run/plan session ends by asking for the next input (`/exit` finishes)."""

from __future__ import annotations

import argparse
from collections.abc import Callable
from pathlib import Path

import pytest

from agent6.app._setup import BudgetOverrides, SandboxOverrides
from agent6.paths import state_dir
from agent6.sessions.layout import SessionLayout
from agent6.ui.cli import _session_prompt as prompt_mod


def _seed_session(
    repo_root: Path, monkeypatch: pytest.MonkeyPatch, session_id: str = "test-run-AAAAAA"
) -> SessionLayout:
    """A real run dir under repo_root's state home, so resolution reaches the
    tty guard rather than short-circuiting on SessionIdError."""

    monkeypatch.setenv("XDG_STATE_HOME", str(repo_root / ".state"))
    layout = SessionLayout(state_dir=state_dir(repo_root), session_id=session_id, subdir="runs")
    layout.session_dir.mkdir(parents=True, exist_ok=True)
    (layout.session_dir / "logs.jsonl").write_text(
        '{"type": "session.start", "ts": "2026-01-01T00:00:00Z"}\n'
        '{"type": "session.end", "reason": "finish_session", "all_passed": true}\n',
        encoding="utf-8",
    )
    return layout


def _run_args(**overrides: object) -> argparse.Namespace:
    """`agent6 run` flags as argparse hands them over, defaults unless overridden."""
    fields: dict[str, object] = {
        "config": None,
        "max_usd": None,
        "max_tokens_fallback": None,
        "dangerously_disable_sandbox": False,
        "auto_approve": False,
        "no_commands": False,
    }
    fields.update(overrides)
    return argparse.Namespace(**fields)


def _seen_resumes(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, str]]:
    calls: list[tuple[str, str]] = []

    def fake_resume(_cfg: Path | None, session_id: str, **kw: object) -> int:
        calls.append((session_id, str(kw.get("steer", ""))))
        return 0

    monkeypatch.setattr(prompt_mod, "_cmd_resume", fake_resume)
    return calls


def test_follow_up_legs_run_under_the_invocations_flags(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`agent6 run --max-usd 0.10 ...` then a follow-up at "next:": the leg
    carries the same overrides. Dropping them ran the follow-up under the
    config's $10 default, silently, after the operator capped the run."""
    from agent6.ui import cli

    layout = _seed_session(tmp_path, monkeypatch)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("agent6.ui.cli._session_prompt.prompting_is_possible", lambda: True)
    seen: list[dict[str, object]] = []

    def fake_resume(_cfg: Path | None, session_id: str, **kw: object) -> int:
        seen.append(dict(kw))
        return 0

    monkeypatch.setattr(prompt_mod, "_cmd_resume", fake_resume)
    answers = iter(["and a test", "/exit"])
    monkeypatch.setattr("builtins.input", lambda _p="": next(answers))
    args = _run_args(max_usd=0.10, auto_approve=True)
    assert cli._prompt_for_the_next_input(args, 0, layout.session_id) == 0  # pyright: ignore[reportPrivateUsage]
    (leg,) = seen
    assert leg["steer"] == "and a test"
    assert leg["budget_overrides"] == BudgetOverrides.from_args(args)
    assert leg["sandbox_overrides"] == SandboxOverrides.from_args(args)


def test_free_text_becomes_the_next_leg_then_exit(monkeypatch: pytest.MonkeyPatch) -> None:
    """Each answer is the next turn's operator instruction -- exactly what
    --steer carries -- so the session continues without retyping `resume`."""
    calls = _seen_resumes(monkeypatch)
    answers = iter(["now add the tests", "  ", "/exit"])
    rc = prompt_mod.end_of_session_prompt(
        rc=0, session_id="runny-one-AAAAAA", ask=lambda _p: next(answers)
    )
    assert rc == 0
    assert calls == [("runny-one-AAAAAA", "now add the tests")]


def test_a_malformed_directive_re_prompts_instead_of_spending_a_leg(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A bare `/pin` typed here started a resume leg the loop could only
    decline; the model answered without tools and the passed run read
    "failed · silent finish". The prompt names the problem and asks again."""
    calls = _seen_resumes(monkeypatch)
    answers = iter(["/pin", "/pin keep the API stable", "/exit"])
    rc = prompt_mod.end_of_session_prompt(
        rc=0, session_id="runny-one-AAAAAA", ask=lambda _p: next(answers)
    )
    assert rc == 0
    assert calls == [("runny-one-AAAAAA", "/pin keep the API stable")]
    assert "pin needs an instruction" in capsys.readouterr().err


def test_exit_leaves_the_session_resumable(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """/exit ends the prompting, never the session: nothing is sealed, so the
    printed line is the one that picks it back up."""
    _seen_resumes(monkeypatch)
    rc = prompt_mod.end_of_session_prompt(
        rc=3, session_id="runny-one-AAAAAA", ask=lambda _p: "/exit"
    )
    assert rc == 3
    assert "agent6 resume runny-one-AAAAAA" in capsys.readouterr().out


def test_eof_ends_like_exit(monkeypatch: pytest.MonkeyPatch) -> None:
    """Walking away mid-prompt (Ctrl-D) is not an instruction."""
    calls = _seen_resumes(monkeypatch)

    def eof(_p: str) -> str:
        raise EOFError

    assert prompt_mod.end_of_session_prompt(rc=0, session_id="r-AAAAAA", ask=eof) == 0
    assert not calls


def test_a_failing_leg_stops_the_loop(monkeypatch: pytest.MonkeyPatch) -> None:
    """A resume that refuses (bad config, dirty tree) returns its own code
    rather than re-prompting over the failure."""

    def failing(_cfg: Path | None, _session_id: str, **_kw: object) -> int:
        return 2

    monkeypatch.setattr(prompt_mod, "_cmd_resume", failing)
    asked: list[str] = []

    def ask(prompt: str) -> str:
        asked.append(prompt)
        return "keep going"

    assert prompt_mod.end_of_session_prompt(rc=0, session_id="r-AAAAAA", ask=ask) == 2
    assert len(asked) == 1


def test_no_terminal_ends_the_session_as_before(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A headless run (CI, a detached spawn) has nobody to type: it must end,
    not block on a prompt nothing will answer."""
    from agent6.ui.cli import _prompt_for_the_next_input  # pyright: ignore[reportPrivateUsage]

    # A real session dir so the ONLY short-circuit under test is the tty guard;
    # patch the bindings _prompt_for_the_next_input actually calls (imported
    # into `cli`, not the source module).
    layout = _seed_session(tmp_path, monkeypatch)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("agent6.ui.cli._session_prompt.prompting_is_possible", lambda: False)
    called: list[str] = []

    def spy(**_kw: object) -> int:
        called.append("asked")
        return 0

    monkeypatch.setattr("agent6.ui.cli._session_prompt.end_of_session_prompt", spy)
    assert _prompt_for_the_next_input(_run_args(), 0, layout.session_id) == 0
    assert not called


def test_ask_sessions_do_not_prompt() -> None:
    """`agent6 ask` answers a question; a one-shot that becomes a conversation
    is a different feature: this is scoped to run and plan sessions."""
    import inspect

    from agent6.ui.cli import _dispatch_ask  # pyright: ignore[reportPrivateUsage]

    assert "_prompt_for_the_next_input" not in inspect.getsource(_dispatch_ask)


def test_a_backgrounded_run_is_not_stopped_by_the_prompt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`agent6 run ... &` keeps a tty on stdin, so isatty() alone said "someone
    is there". Reading the terminal from a BACKGROUND process group raises
    SIGTTIN, which stops the job: the run suspended at the end instead of
    finishing, and needed `fg`. The same shape blocks forever wherever a tty is
    allocated with nobody at it (`docker run -t`, some CI runners).
    """
    monkeypatch.setattr(prompt_mod.sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(prompt_mod.sys.stdin, "fileno", lambda: 0)
    monkeypatch.setattr(prompt_mod.os, "getpgrp", lambda: 4242)

    def owner_is(pgrp: int) -> Callable[[int], int]:
        def tcgetpgrp(_fd: int) -> int:
            return pgrp

        return tcgetpgrp

    monkeypatch.setattr(prompt_mod.os, "tcgetpgrp", owner_is(1717))
    assert not prompt_mod.prompting_is_possible(), "prompted from a background process group"

    monkeypatch.setattr(prompt_mod.os, "tcgetpgrp", owner_is(4242))
    assert prompt_mod.prompting_is_possible(), "the foreground job must still prompt"


def test_a_refused_runs_discarded_id_ends_quietly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A refusal discards its husk, so the minted id matches nothing on disk;
    the follow-up prompt must end with the refusal's exit code, not crash on
    the resolver's SessionIdError."""
    from agent6.ui import cli

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("agent6.ui.cli._session_prompt.prompting_is_possible", lambda: True)
    assert cli._prompt_for_the_next_input(_run_args(), 2, "gone-run-QQQQQQ") == 2  # pyright: ignore[reportPrivateUsage]


@pytest.mark.parametrize(("mode", "asks"), [("run", True), ("plan", True), ("ask", False)])
def test_a_resumed_leg_ends_by_asking_like_a_fresh_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str, asks: bool
) -> None:
    """`agent6 resume <id>` ended without the "next:" prompt a run ends with;
    a resumed run or plan asks the same way (a resumed ask stays a one-shot)."""
    import json

    from agent6.ui import cli

    layout = _seed_session(tmp_path, monkeypatch, session_id="resumed-run-AAAAAA")
    (layout.session_dir / "manifest.json").write_text(
        json.dumps({"version": 3, "session_id": layout.session_id, "mode": mode}),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("agent6.ui.cli._session_prompt.prompting_is_possible", lambda: True)

    def _resumed(*_a: object, **_k: object) -> int:
        return 0

    monkeypatch.setattr("agent6.ui.cli.resume._cmd_resume", _resumed)
    asked: list[str] = []

    def spy(**kw: object) -> int:
        asked.append(str(kw["session_id"]))
        return 0

    monkeypatch.setattr("agent6.ui.cli._session_prompt.end_of_session_prompt", spy)
    args = _run_args(session_id="resumed-run", force=False, tui=False, preset="", steer="")
    assert cli._dispatch_resume(args) == 0  # pyright: ignore[reportPrivateUsage]
    assert asked == (["resumed-run-AAAAAA"] if asks else [])


def test_a_leg_that_undoes_or_detaches_ends_the_asking(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Inside the prompt loop a follow-up leg can end by /undo (the fork it
    named is the continuation) or by /detach (the run went on in the
    background); the loop asked "next:" again for a run that takes no
    follow-up here, and an answer would have collided or been refused."""
    layout = _seed_session(tmp_path, monkeypatch, session_id="undone-run-AAAAAA")
    monkeypatch.chdir(tmp_path)
    asked: list[str] = []

    def fake_resume(_cfg: Path | None, _sid: str, **kw: object) -> int:
        # The leg forks back and ends the run as undone.
        (layout.session_dir / "logs.jsonl").write_text(
            '{"type": "session.start"}\n{"type": "session.end", "reason": "undone"}\n',
            encoding="utf-8",
        )
        return 0

    monkeypatch.setattr(prompt_mod, "_cmd_resume", fake_resume)

    def ask(prompt: str) -> str:
        asked.append(prompt)
        return "/undo" if len(asked) == 1 else pytest.fail("asked again after the undo")

    rc = prompt_mod.end_of_session_prompt(
        rc=0, session_id=layout.session_id, session_dir=layout.session_dir, ask=ask
    )
    assert rc == 0 and len(asked) == 1


def test_a_detached_run_is_not_followed_by_the_prompt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`/detach` hands the run to a background resume (its reattach line was
    printed); the leg here did not end, so there is nothing to follow up on.
    Asking "next:" offered a leg that would collide with the live one."""
    from agent6.ui import cli

    layout = _seed_session(tmp_path, monkeypatch, session_id="detached-run-AAAAAA")
    (layout.session_dir / "logs.jsonl").write_text(
        '{"type": "session.start", "ts": "2026-01-01T00:00:00Z"}\n'
        '{"type": "loop.steer.detached"}\n',
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("agent6.ui.cli._session_prompt.prompting_is_possible", lambda: True)

    def _must_not_prompt(**_kw: object) -> int:
        pytest.fail("prompted")

    monkeypatch.setattr("agent6.ui.cli._session_prompt.end_of_session_prompt", _must_not_prompt)
    args = _run_args()
    assert cli._prompt_for_the_next_input(args, 0, layout.session_id) == 0  # pyright: ignore[reportPrivateUsage]


def test_a_parked_start_is_not_followed_by_the_prompt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A start that parked (busy checkout, uncommitted changes) never ran; the
    resume line it printed is the next step. Asking "next:" there offered a
    follow-up to a leg that does not exist and re-parked on the same cause."""
    import json

    from agent6.ui import cli

    layout = _seed_session(tmp_path, monkeypatch, session_id="parked-run-AAAAAA")
    (layout.session_dir / "manifest.json").write_text(
        json.dumps(
            {
                "version": 3,
                "session_id": layout.session_id,
                "mode": "run",
                "user_task": "t",
                "parked_task": "t",
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("agent6.ui.cli._session_prompt.prompting_is_possible", lambda: True)
    asked: list[str] = []

    def spy(**_kw: object) -> int:
        asked.append("asked")
        return 0

    monkeypatch.setattr("agent6.ui.cli._session_prompt.end_of_session_prompt", spy)
    assert cli._prompt_for_the_next_input(_run_args(), 2, layout.session_id) == 2  # pyright: ignore[reportPrivateUsage]
    assert asked == []


def test_an_undone_run_is_not_followed_by_the_prompt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """/undo forks back and names the fork as the continuation; asking "next:"
    on the undone run offered a follow-up to the abandoned one (and its /exit
    printed a resume line for it)."""
    from agent6.ui import cli

    layout = _seed_session(tmp_path, monkeypatch, session_id="undone-run-AAAAAA")
    with (layout.session_dir / "logs.jsonl").open("a", encoding="utf-8") as fh:
        fh.write('{"type": "session.undone", "new_session_id": "fork-BBBBBB"}\n')
        fh.write('{"type": "session.end", "reason": "undone", "all_passed": false}\n')
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("agent6.ui.cli._session_prompt.prompting_is_possible", lambda: True)
    asked: list[str] = []

    def spy(**_kw: object) -> int:
        asked.append("asked")
        return 0

    monkeypatch.setattr("agent6.ui.cli._session_prompt.end_of_session_prompt", spy)
    assert cli._prompt_for_the_next_input(_run_args(), 0, layout.session_id) == 0  # pyright: ignore[reportPrivateUsage]
    assert asked == []


def test_a_lone_slash_word_is_refused_not_sent_as_a_task(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """`/shells` typed here spent a model call answering the literal text as a
    new leg's task; a live-run command re-prompts with steer_problem's pointer,
    an unknown lone slash word (a typo, a REPL verb) with the prompt's own.
    Multi-word slash input still rides as the leg's instruction (directives
    like `/pin <text>` are the loop's to parse)."""
    calls = _seen_resumes(monkeypatch)
    answers = iter(["/shells", "/cost", "now add the tests", "/exit"])
    rc = prompt_mod.end_of_session_prompt(
        rc=0, session_id="runny-one-AAAAAA", ask=lambda _p: next(answers)
    )
    assert rc == 0
    assert calls == [("runny-one-AAAAAA", "now add the tests")]
    err = capsys.readouterr().err
    assert "/shells acts in a composer or the pause menu" in err
    assert "'/cost' is not sent as a task" in err
