# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""`agent6 plan show/edit` and `agent6 run --from` CLI smoke."""

from __future__ import annotations

from pathlib import Path

import pytest

from agent6.errors import OperatorError
from agent6.paths import state_dir
from agent6.ui.cli import cli_main, main


def _seed_plan(tmp_path: Path, session_id: str, body: str) -> Path:
    plan_dir = state_dir(tmp_path) / "sessions" / "plans" / session_id
    plan_dir.mkdir(parents=True)
    plan = plan_dir / "plan.md"
    plan.write_text(body, encoding="utf-8")
    return plan


def test_plan_show_prints_plan(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(tmp_path)
    _seed_plan(tmp_path, "happy-tree-abcd", "# Plan: foo\n\nbody\n")
    rc = main(["plan", "show", "happy-tree-abcd"])
    assert rc == 0
    assert "# Plan: foo" in capsys.readouterr().out


def test_plan_show_resolves_prefix(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(tmp_path)
    _seed_plan(tmp_path, "happy-tree-abcd", "# Plan: foo\n")
    rc = main(["plan", "show", "happy"])
    assert rc == 0
    assert "# Plan: foo" in capsys.readouterr().out


def test_plan_show_prefix_ignores_a_run_of_the_same_prefix(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """`plan show` resolves inside plans/: a run sharing the prefix is not a
    second match."""
    monkeypatch.chdir(tmp_path)
    _seed_plan(tmp_path, "happy-tree-abcd", "# Plan: foo\n")
    (state_dir(tmp_path) / "sessions" / "runs" / "happy-tree-zzzz").mkdir(parents=True)
    assert main(["plan", "show", "happy"]) == 0
    assert "# Plan: foo" in capsys.readouterr().out


def test_plan_show_omit_id_uses_most_recent_plan(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # Matches the omit-for-latest convention of the runs commands.
    monkeypatch.chdir(tmp_path)
    _seed_plan(tmp_path, "only-plan-abcd", "# Plan: the latest\n\nsteps\n")
    rc = main(["plan", "show"])
    assert rc == 0
    assert "# Plan: the latest" in capsys.readouterr().out


def test_from_plan_task_leads_with_the_plan_title() -> None:
    # The run's task (shown in listings / DAG root) must read as the plan, not
    # the 'The following plan was prepared...' boilerplate.
    from agent6.ui.cli import _from_plan_task  # pyright: ignore[reportPrivateUsage]

    task = _from_plan_task("# Plan: Add a --count flag\n\n1. do it", "serene-geyser-NP20")
    assert task.startswith("Execute the prepared plan: Add a --count flag")
    assert "1. do it" in task  # the full plan is still fed to the agent


def test_plan_show_omit_id_with_no_plans_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(tmp_path)
    assert main(["plan", "show"]) == 2
    assert "no plans yet" in capsys.readouterr().err


def test_plan_show_missing_run_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(tmp_path)
    (state_dir(tmp_path) / "sessions" / "plans").mkdir(parents=True)
    rc = main(["plan", "show", "nonexistent"])
    assert rc == 2
    assert "ERROR" in capsys.readouterr().err


def test_plan_show_no_plan_md_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(tmp_path)
    (state_dir(tmp_path) / "sessions" / "plans" / "happy-tree-abcd").mkdir(parents=True)
    rc = main(["plan", "show", "happy-tree-abcd"])
    assert rc == 2
    err = capsys.readouterr().err
    assert "plan.md" in err


def test_plan_requires_task_or_show(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(tmp_path)
    # Bare `plan` (no task, no verb) injects the `run` verb and reports the
    # missing-task error rather than the most-recent-plan prompt (no runs here).
    rc = main(["plan"])
    assert rc == 2
    assert "ERROR" in capsys.readouterr().err


def test_plan_edit_invokes_editor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    plan = _seed_plan(tmp_path, "happy-tree-abcd", "original\n")
    marker = tmp_path / "editor_ran"
    script = tmp_path / "fake_editor.sh"
    script.write_text(f"#!/bin/sh\necho edited >> $1\ntouch {marker}\n", encoding="utf-8")
    script.chmod(0o755)
    monkeypatch.setenv("EDITOR", str(script))
    rc = main(["plan", "edit", "happy-tree-abcd"])
    assert rc == 0
    assert marker.exists()
    assert "edited" in plan.read_text(encoding="utf-8")


def test_plan_edit_honors_a_multi_word_editor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """$EDITOR may be a command with flags ("code --wait"); the steer editor
    already splits it, but plan edit spawned the whole string as one binary
    name and failed every time for VS Code/emacsclient users."""
    monkeypatch.chdir(tmp_path)
    plan = _seed_plan(tmp_path, "happy-tree-efgh", "original\n")
    marker = tmp_path / "editor_ran"
    script = tmp_path / "fake_editor.sh"
    script.write_text(
        f'#!/bin/sh\n[ "$1" = --wait ] || exit 3\nshift\necho edited >> $1\ntouch {marker}\n',
        encoding="utf-8",
    )
    script.chmod(0o755)
    monkeypatch.setenv("EDITOR", f"{script} --wait")
    rc = main(["plan", "edit", "happy-tree-efgh"])
    assert rc == 0
    assert marker.exists()
    assert "edited" in plan.read_text(encoding="utf-8")


def test_run_from_a_plan_with_no_task_runs_that_plan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`--from` folded into `--from`: a plan id alone is the task (the
    plan's own text), and the seed is not digested a second time."""
    from agent6.ui import cli

    monkeypatch.chdir(tmp_path)
    _seed_plan(tmp_path, "happy-tree-abcd", "# Plan: do it\n\n1. step\n")
    seen: dict[str, object] = {}

    def _fake_run(_cfg: object, task: str, **kw: object) -> int:
        seen.update(kw, task=task)
        return 0

    def _no_prompt(_args: object, rc: int, _sid: str) -> int:
        return rc

    monkeypatch.setattr("agent6.ui.cli.run._cmd_run", _fake_run)
    monkeypatch.setattr(cli, "_prompt_for_the_next_input", _no_prompt)
    assert main(["run", "--from", "happy-tree-abcd"]) == 0
    assert "do it" in str(seen["task"]) and "1. step" in str(seen["task"])
    assert seen["seed_from"] == ""


def test_run_from_a_run_with_no_task_names_what_it_needs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    run_dir = state_dir(tmp_path) / "sessions" / "runs" / "busy-fox-abcd"
    run_dir.mkdir(parents=True)
    (run_dir / "manifest.json").write_text('{"mode": "run"}', encoding="utf-8")
    assert main(["run", "--from", "busy-fox-abcd"]) == 2
    assert "needs a task" in capsys.readouterr().err


def test_seeding_from_a_plan_carries_its_text(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With a task, `--from <plan>` digests the plan session, plan.md included;
    the digest of a session that wrote no code said nothing about the plan."""
    import json

    from agent6.ui.cli._ask import build_ask_session_digest

    monkeypatch.chdir(tmp_path)
    plan = _seed_plan(tmp_path, "happy-tree-abcd", "# Plan: do it\n\n1. step\n")
    (plan.parent / "manifest.json").write_text(
        json.dumps({"mode": "plan", "user_task": "plan it"}), encoding="utf-8"
    )
    digest = build_ask_session_digest(tmp_path, "happy-tree-abcd", latest=False)
    assert digest is not None and "## Plan" in digest and "1. step" in digest


def test_an_unreadable_plan_refuses_rather_than_crashing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A file-permission problem is the operator's, not an agent6 defect.

    The reader raises OperatorError (no bespoke except arm at the call site);
    cli_main is the one place that turns it into `ERROR:` + exit 2."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("AGENT6_DEBUG", raising=False)
    plan = _seed_plan(tmp_path, "quiet-fox-abcd", "# Plan: do it\n")
    plan.chmod(0o000)
    try:
        with pytest.raises(OperatorError, match="could not read"):
            main(["run", "--from", "quiet-fox-abcd"])
        assert cli_main(["run", "--from", "quiet-fox-abcd"]) == 2
        err = capsys.readouterr().err
        assert "could not read" in err
        assert str(plan) in err
        assert "report this" not in err
        assert "full traceback" not in err
    finally:
        plan.chmod(0o600)


def test_an_unreadable_plan_refuses_in_plan_show_too(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """`plan show` reads the same operator file `--from` does; the same
    refusal, from the same shared reader."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("AGENT6_DEBUG", raising=False)
    plan = _seed_plan(tmp_path, "quiet-owl-abcd", "# Plan: do it\n")
    plan.chmod(0o000)
    try:
        assert cli_main(["plan", "show", "quiet-owl-abcd"]) == 2
        err = capsys.readouterr().err
        assert "could not read" in err
        assert "report this" not in err
    finally:
        plan.chmod(0o600)


def test_plan_takes_tui_like_run(monkeypatch: pytest.MonkeyPatch) -> None:
    """`agent6 plan --tui "<task>"` opens the TUI on the planning run as `run
    --tui` does on a run (it read "unrecognized arguments: --tui")."""
    from agent6.ui import cli

    seen: dict[str, object] = {}

    def _fake_run(_cfg: object, task: str, **kw: object) -> int:
        seen.update(kw, task=task)
        return 0

    monkeypatch.setattr("agent6.ui.cli.run._cmd_run", _fake_run)

    def _no_prompt(_args: object, rc: int, _sid: str) -> int:
        return rc

    monkeypatch.setattr(cli, "_prompt_for_the_next_input", _no_prompt)
    assert main(["plan", "--tui", "lay out the work"]) == 0
    assert seen["task"] == "lay out the work"
    assert seen["mode"] == "plan" and seen["tui"] is True


def test_plan_edit_reports_an_editor_that_failed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """An editor exiting non-zero (a crash, a refused lock) came back through
    `plan edit` as its bare code, with no word about it."""
    monkeypatch.chdir(tmp_path)
    _seed_plan(tmp_path, "happy-tree-ijkl", "original\n")
    script = tmp_path / "failing_editor.sh"
    script.write_text("#!/bin/sh\nexit 3\n", encoding="utf-8")
    script.chmod(0o755)
    monkeypatch.setenv("EDITOR", str(script))
    assert main(["plan", "edit", "happy-tree-ijkl"]) == 1
    assert "exited 3" in capsys.readouterr().err
