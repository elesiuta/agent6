# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Parser builders for `plan` and `ask`: alternate single-loop modes (planning-
only, Q&A) alongside the main `run`, each with its own default-verb
subcommand tree (see `_inject_default_verb`)."""

from __future__ import annotations

import argparse
import os

from agent6.ui.cli._common import _add_budget_flags, _add_config_flag, _add_sandbox_flags, _sub
from agent6.ui.cli.completers import (
    _complete_plan_session_ids,
    _complete_presets,
    _complete_session_ids,
)


def _add_plan_parser(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    plan_p = _sub(
        sub,
        "plan",
        help=(
            "Planning pass: same loop, no edit tools, writes plan.md."
            ' A configured `run_commands = "yes"` clamps to `"ask"`.'
            " Pair with `agent6 run --from-plan <run-id>` to execute."
            " Inspect with `plan show <id>` / `plan edit <id>`."
        ),
    )
    # `plan <task>` is the bare planning run; `plan show/edit <id>` inspect a
    # prior plan. `run` is the implicit default verb injected by
    # `_inject_default_verb` when the first token isn't a known plan verb, so
    # `plan "fix the bug"` and `plan run "fix the bug"` are the same.
    plan_sub = plan_p.add_subparsers(dest="plan_command", required=True, metavar="<subcommand>")
    plan_run = _sub(plan_sub, "run", help="Run a planning pass on a task.")
    plan_run.add_argument(
        "task",
        nargs="?",
        default="",
        help="Task to plan (in quotes). Required; `plan show/edit <id>` inspect prior plans.",
    )
    plan_run.add_argument(
        "--session-id", default="", help="Explicit session id (default: generate one)."
    )
    plan_run.add_argument(
        "--tui",
        action="store_true",
        help=(
            "Open the full-screen TUI on the planning run (the conversation view; Ctrl+D"
            " toggles the dashboard) instead of the default headless CLI stream. Needs a"
            " TTY. (Or run `agent6 tui` and start the plan from there.)"
        ),
    )
    plan_profile = plan_run.add_argument(
        "--preset", default="", help="Strategy preset (see `agent6 run --preset`)."
    )
    plan_profile.completer = _complete_presets  # type: ignore[attr-defined]
    _add_config_flag(plan_run)
    _add_budget_flags(plan_run)
    _add_sandbox_flags(plan_run)
    plan_show = _sub(plan_sub, "show", help="Print the plan.md for a prior plan run and exit.")
    plan_show_id = plan_show.add_argument(
        "session_id",
        nargs="?",
        default="",
        help="Plan run id (or unambiguous prefix); omit for the most recent plan.",
    )
    plan_show_id.completer = _complete_plan_session_ids  # type: ignore[attr-defined]
    plan_edit = _sub(
        plan_sub,
        "edit",
        help=(
            "Open the plan.md for a prior plan run in $EDITOR"
            f" (currently: {os.environ.get('EDITOR', '') or 'vi'}) and exit."
        ),
    )
    plan_edit_id = plan_edit.add_argument(
        "session_id",
        nargs="?",
        default="",
        help="Plan run id (or unambiguous prefix); omit for the most recent plan.",
    )
    plan_edit_id.completer = _complete_plan_session_ids  # type: ignore[attr-defined]


def _add_ask_parser(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    ask_p = _sub(
        sub,
        "ask",
        help=(
            "Q&A in prose: no edit tools, no commits, no repo required."
            ' A configured `run_commands = "yes"` clamps to `"ask"`.'
            " `ask list` shows saved asks."
        ),
    )
    # `ask <question>` runs a Q&A; `ask list` enumerates saved asks. `query` is
    # the implicit default verb injected by `_inject_default_verb` when the first
    # token isn't a known ask verb, so `ask "why ..."` == `ask query "why ..."`.
    ask_sub = ask_p.add_subparsers(dest="ask_command", required=True, metavar="<subcommand>")
    ask_query = _sub(ask_sub, "query", help="Ask a question (the default verb).")
    ask_profile = ask_query.add_argument(
        "--preset", default="", help="Strategy preset (see `agent6 run --preset`)."
    )
    ask_profile.completer = _complete_presets  # type: ignore[attr-defined]
    ask_query.add_argument(
        "task",
        nargs="?",
        default="",
        help='Question (in quotes), e.g. "why does the retry loop double the timeout?".',
    )
    _add_config_flag(ask_query)
    # One seed, named one way: the two spellings are mutually exclusive.
    seed = ask_query.add_mutually_exclusive_group()
    ask_session = seed.add_argument(
        "--from",
        dest="ask_session",
        default="",
        metavar="SESSION_ID",
        help=(
            "Seed this question from another session (a run, a plan or an ask):"
            " its task, outcome, diff and key events (exact id or unambiguous"
            " prefix)."
        ),
    )
    ask_session.completer = _complete_session_ids  # type: ignore[attr-defined]
    seed.add_argument(
        "--from-latest",
        dest="ask_session_latest",
        action="store_true",
        help="Like --from, but seed the most recent run or ask (plans are skipped).",
    )
    ask_query.add_argument(
        "--file",
        dest="ask_files",
        action="append",
        default=[],
        metavar="PATH",
        help="Seed a file's contents into the question (repeatable; like an inline @path).",
    )
    ask_query.add_argument(
        "-i",
        "--interactive",
        action="store_true",
        help=(
            "Interactive REPL: keep asking follow-ups in one session (the prior"
            " Q&A is carried as context). /cost, /reset, /quit. Also the default"
            " when no question is given and stdin is a TTY."
        ),
    )
    _add_budget_flags(ask_query)
    _add_sandbox_flags(ask_query)
    _sub(
        ask_sub,
        "list",
        help="List saved asks under the per-repo state dir (asks subdir, newest first) and exit.",
    )
