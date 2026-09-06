# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Parser builders for the run/resume/fork family: start a run, resume a
paused one from its snapshot, or fork a new run off a prior checkpoint."""

from __future__ import annotations

import argparse

from agent6.config.layer import BUILTIN_PRESETS
from agent6.ui.cli._common import _add_budget_flags, _add_config_flag, _add_sandbox_flags, _sub
from agent6.ui.cli.completers import (
    _complete_parallel_models,
    _complete_plan_session_ids,
    _complete_presets,
    _complete_resumable_ids,
    _complete_session_ids,
    _complete_skills,
)


def _add_run_parser(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    run_p = _sub(sub, "run", help="Run the single-loop agent on a task.")
    run_p.add_argument(
        "task",
        nargs="?",
        default="",
        help="Task description (in quotes). Omit to execute the most recent plan.",
    )
    run_p.add_argument(
        "--session-id", default="", help="Explicit session id (default: generate one)."
    )
    run_from = run_p.add_argument(
        "--from",
        dest="seed_from",
        default="",
        metavar="SESSION_ID",
        help=(
            "Seed a new run from another session (a run, a plan or an ask):"
            " its task, outcome, diff and key events. The source is untouched;"
            " use `fork` to clone a session at a past turn instead."
        ),
    )
    run_from.completer = _complete_session_ids  # type: ignore[attr-defined]
    run_p.add_argument(
        "--pin",
        dest="pins",
        action="append",
        default=[],
        metavar="TEXT",
        help="Pin an instruction before the run starts (repeatable). Like /pin:"
        " it survives context compaction and is restated to the model for the whole run."
        " A /parallel lane inherits the coordinator's pins through this.",
    )
    run_profile = run_p.add_argument(
        "--preset",
        default="",
        help=(
            f"Strategy preset ({'/'.join(BUILTIN_PRESETS)}, or a custom [presets.<name>])."
            " Overrides the top-level `preset` key and your config files; an explicit"
            " --config FILE or individual flags still win."
        ),
    )
    run_profile.completer = _complete_presets  # type: ignore[attr-defined]
    _add_config_flag(run_p)
    run_p.add_argument(
        "-i",
        "--interactive",
        action="store_true",
        help=(
            "REPL mode: after each successful auto-commit, prompt on stdin for"
            " one of /continue (default), /diff, /cost, /undo (fork back before the last message),"
            " /watch, /mcp, /init, /help, /quit. Requires a TTY."
        ),
    )
    run_p.add_argument(
        "--tui",
        action="store_true",
        help=(
            "Open the full-screen TUI on the run (the conversation view; Ctrl+D"
            " toggles the dashboard) instead of the default headless CLI stream."
            " Needs a TTY; mutually exclusive with -i."
            " (Or run `agent6 tui` and start the run from there.)"
        ),
    )
    run_from_plan = run_p.add_argument(
        "--from-plan",
        default="",
        metavar="RUN_ID",
        help=(
            "Use the plan.md from a prior `agent6 plan` run (resolved"
            " under the per-repo run-state dir, exact or unambiguous prefix) as the"
            " task description. Mutually exclusive with a positional task."
        ),
    )
    run_from_plan.completer = _complete_plan_session_ids  # type: ignore[attr-defined]
    run_p.add_argument(
        "--decompose",
        action="store_true",
        help=(
            "Plan-first: the agent lays the task out as ordered DAG subtasks"
            " (add_task) before editing, then works them one at a time, with no"
            " approval step. Same as setting [prompt].decompose for this run."
            " Helps on multi-part tasks and smaller models; a capable model"
            " decomposes implicitly, so measure before leaving it on."
        ),
    )
    run_skill = run_p.add_argument(
        "--skill",
        action="append",
        default=[],
        metavar="NAME",
        help="Prepend an installed skill's instructions to the task (repeatable).",
    )
    run_skill.completer = _complete_skills  # type: ignore[attr-defined]
    run_parallel_flag = run_p.add_argument(
        "--parallel",
        default="",
        metavar="N|m1,m2,...",
        help=(
            "Fan out isolated lanes: an integer N runs N lanes on the worker model,"
            " a comma-separated model list runs one lane per model. Each lane clones"
            " the repo, runs independently, and lands its own branch; results are"
            " auto-compared and ranked (nothing is merged). Capped by"
            " [parallel].max_lanes; combine with --max-usd for a per-lane budget."
        ),
    )
    run_parallel_flag.completer = _complete_parallel_models  # type: ignore[attr-defined]
    run_p.add_argument(
        "--standing",
        default="",
        metavar="GOAL",
        help=(
            "A standing goal for this run: a never-finishing fallback task the run"
            " re-enters whenever the ordinary queue drains or the worker tries to"
            " stop. New work always outranks it. The run still ends on its budget,"
            " an operator stop, or the iteration cap (workflow.standing_patience"
            " can additionally end it after N fruitless re-entries; default never)."
        ),
    )
    _add_budget_flags(run_p)
    _add_sandbox_flags(run_p)


def _add_resume_parser(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    resume_p = _sub(sub, "resume", help="Resume a paused run from its snapshot.")
    resume_run = resume_p.add_argument(
        "session_id",
        nargs="?",
        default="",
        help="Session id under the per-repo state dir (omit for the most recent).",
    )
    resume_run.completer = _complete_resumable_ids  # type: ignore[attr-defined]
    _add_config_flag(resume_p)
    resume_preset = resume_p.add_argument(
        "--preset",
        default="",
        help=(
            "Continue under another strategy preset (a preset touches any setting, so it"
            " changes only between legs); recorded on the run, so later resumes keep it."
        ),
    )
    resume_preset.completer = _complete_presets  # type: ignore[attr-defined]
    resume_p.add_argument(
        "--force",
        action="store_true",
        help="Resume even if the run's commit chain diverged from its last snapshot "
        "(a rewritten or replaced agent6/<id> ref; the run's own forward commits resume "
        "without this flag).",
    )
    resume_p.add_argument(
        "--tui",
        action="store_true",
        help="Open the full-screen TUI instead of the headless stream (like `run --tui`).",
    )
    resume_p.add_argument(
        "-i",
        "--interactive",
        action="store_true",
        help=(
            "Interactive resume, like `run -i`: when the model goes quiet the"
            " run parks for your steer instead of ending."
        ),
    )
    resume_p.add_argument(
        "--steer",
        default="",
        metavar="TEXT",
        help=(
            "Inject TEXT as an operator steering instruction at the resumed"
            " session's first safe boundary (the TUI composer bar's follow-up"
            " uses this)."
        ),
    )
    _add_budget_flags(resume_p)
    _add_sandbox_flags(resume_p)


def _add_fork_parser(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    fork_p = _sub(
        sub,
        "fork",
        help=(
            "Clone a run, rolled back to a checkpoint, into a new run in its own git"
            " worktree and continue it (the source run and your checkout are never touched)."
        ),
    )
    fork_src = fork_p.add_argument(
        "session_id",
        nargs="?",
        default="",
        help="Source run id or unambiguous prefix to fork from (omit for the most recent run).",
    )
    fork_src.completer = _complete_resumable_ids  # type: ignore[attr-defined]
    fork_p.add_argument(
        "--at-turn",
        type=int,
        default=None,
        metavar="N",
        dest="at_turn",
        help="Checkpoint turn to fork from (default: the latest checkpoint).",
    )
    fork_p.add_argument(
        "--session-id",
        default="",
        dest="new_session_id",
        help="Explicit id for the new (forked) session (default: generate one).",
    )
    fork_p.add_argument(
        "--no-run",
        action="store_true",
        help="Only create the fork (its run dir and worktree); resume it later.",
    )
    _add_config_flag(fork_p)
    fork_p.add_argument(
        "--tui",
        action="store_true",
        help="Open the full-screen TUI instead of the headless stream (like `run --tui`).",
    )
    fork_p.add_argument(
        "--steer",
        default="",
        metavar="TEXT",
        help=(
            "Inject TEXT as an operator steering instruction at the forked"
            " session's first safe boundary. Not with --no-run; use"
            " `resume --steer` afterwards."
        ),
    )
    _add_budget_flags(fork_p)
    # A fork without --no-run CONTINUES a run, so it is a paid command like the
    # rest and carries the same approval/sandbox overrides.
    _add_sandbox_flags(fork_p)
