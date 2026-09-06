# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
# PYTHON_ARGCOMPLETE_OK
"""agent6 command-line interface."""

from __future__ import annotations

import argparse
import contextlib
import io
import os
import sys
import tempfile
import traceback
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

import argcomplete

from agent6.errors import OperatorError
from agent6.events import EventWriteError
from agent6.ui.cli._common import _enforce_root_policy, error, refuse
from agent6.ui.cli.parser import _command_index, _inject_default_verb, build_parser

if TYPE_CHECKING:
    from agent6.sessions.layout import SessionLayout


def _first_markdown_line(text: str, max_len: int = 80) -> str:
    """First non-empty line of a markdown doc (a plan title), `#`/bullet stripped."""
    for raw in text.splitlines():
        line = raw.strip().lstrip("#").lstrip("-*").strip()
        if line:
            return line[:max_len]
    return "(untitled plan)"


def _from_plan_task(plan_md: str, session_id: str) -> str:
    """The execution prompt for `run --from <plan>`, LEADING with the plan title so
    a listing (the runs table, the DAG root, attach --json) shows the plan, not
    the 'The following plan was prepared...' boilerplate as the run's task."""
    title = _first_markdown_line(plan_md)
    if title.lower().startswith("plan:"):  # the '# Plan: <title>' convention
        title = title[len("plan:") :].strip() or title
    return f"Execute the prepared plan: {title}\n\n(from planning pass {session_id})\n\n{plan_md}"


def cli_main(argv: list[str] | None = None) -> int:
    """Console-script entry point: the boundary that sorts failures by fault.

    An `OperatorError` (a bad flag value, an unreadable operator file, an
    invalid config) prints as an `ERROR:` refusal at exit 2, no traceback.
    Anything else is a bug in agent6: a one-line `ERROR: unexpected ...` plus
    a pointer to a saved traceback, exit 1. Set `AGENT6_DEBUG=1` to re-raise
    the full traceback inline (for bug reports). `main` itself is left
    unguarded so tests and `python -m` see real tracebacks. argparse's
    `SystemExit` (bad args / --help) is not an `Exception` and passes
    through untouched.
    """
    try:
        return main(argv)
    except KeyboardInterrupt:
        print("\nagent6: interrupted.", file=sys.stderr)
        return 130
    except OperatorError as exc:
        error(f"{exc}")
        return 2
    except Exception as exc:  # top-level last resort; re-raised under AGENT6_DEBUG
        if os.environ.get("AGENT6_DEBUG") == "1":
            raise
        error(f"unexpected {type(exc).__name__}: {exc}")
        try:
            fd, path = tempfile.mkstemp(prefix="agent6-crash-", suffix=".log")
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                traceback.print_exc(file=fh)
            print(f"  full traceback: {path}", file=sys.stderr)
        except OSError:
            pass  # never let crash-reporting itself crash the exit path
        print(
            "  re-run with AGENT6_DEBUG=1 to see it inline; if it persists, report it:"
            " https://github.com/agent6-dev/agent6/issues",
            file=sys.stderr,
        )
        return 1


def _dispatch_run(args: argparse.Namespace) -> int:  # noqa: PLR0911, PLR0912
    from agent6.app._setup import BudgetOverrides, SandboxOverrides  # noqa: PLC0415
    from agent6.errors import read_operator_file  # noqa: PLC0415
    from agent6.ui.cli._common import _plans_dir  # noqa: PLC0415
    from agent6.ui.cli.plan_watch import (  # noqa: PLC0415
        _most_recent_plan_session_id,
    )
    from agent6.ui.cli.run import _cmd_run  # noqa: PLC0415

    if args.interactive and not sys.stdin.isatty():
        # -i is explicit and needs the terminal its help names; run on a pipe,
        # the REPL's first prompt reads EOF and stops the run mid-task after
        # the first commit.
        error("-i needs a TTY on stdin (the REPL reads it); drop -i for a headless run.")
        return 2
    if getattr(args, "parallel", "") and (args.interactive or args.tui):
        error("--parallel cannot combine with -i or --tui (each lane runs headless and detached).")
        return 2
    if getattr(args, "parallel", "") and args.session_id:
        error("--parallel cannot combine with --session-id (each lane mints its own id).")
        return 2
    seed_from = getattr(args, "seed_from", "")
    if not args.task and seed_from:
        # A plan id alone runs that plan: its text is the task, and digesting
        # the same text as a seed would double it.
        from agent6.config.layer import resolved_state_dir  # noqa: PLC0415
        from agent6.sessions.id import SessionIdError, resolve_session  # noqa: PLC0415

        try:
            layout = resolve_session(resolved_state_dir(Path.cwd()), seed_from)
        except SessionIdError as exc:
            error(f"{exc}")
            return 2
        plan_path = layout.session_dir / "plan.md"
        if not plan_path.is_file():
            error("'run' needs a task; --from <id> seeds one, and a plan id alone runs that plan.")
            return 2
        task = _from_plan_task(read_operator_file(plan_path), layout.session_id)
        seed_from = ""
    elif not args.task:
        # No task: fall back to the most recent plan run, the common
        # "I just ran `agent6 plan`, now execute it" flow. At a TTY,
        # confirm before editing; non-interactively, refuse (a bare
        # `run` in a script should not silently start mutating).
        last_plan = _most_recent_plan_session_id(_plans_dir(Path.cwd()))
        if last_plan is None:
            error("'run' needs a task (or --from <plan-id>); no prior plan found to execute.")
            return 2
        plan_md = read_operator_file(_plans_dir(Path.cwd()) / last_plan / "plan.md")
        title = _first_markdown_line(plan_md)
        if not sys.stdin.isatty():
            error(
                f"'run' needs a task. Most recent plan is {last_plan}"
                f" ({title}); execute it with: agent6 run --from {last_plan}"
            )
            return 2
        print(f"[agent6] No task given. Most recent plan: {last_plan}  ({title})")
        try:
            ans = input("Execute it now? [Y/n]: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            ans = "n"
        if ans in ("n", "no"):
            print(f"Aborted. Run it later: agent6 run --from {last_plan}")
            return 0
        task = _from_plan_task(plan_md, last_plan)
    else:
        task = args.task
    session_id = _minted_session_id(args.session_id, "run")
    rc = _cmd_run(
        args.config,
        task,
        session_id=session_id,
        interactive=args.interactive,
        tui=args.tui,
        decompose=args.decompose,
        seed_from=seed_from,
        skills=tuple(args.skill),
        budget_overrides=BudgetOverrides.from_args(args),
        sandbox_overrides=SandboxOverrides.from_args(args),
        preset=getattr(args, "preset", ""),
        parallel_spec=getattr(args, "parallel", ""),
        standing_goal=getattr(args, "standing", ""),
        pins=tuple(args.pins),
    )
    # A fan-out ends in its own compare summary and the TUI owns its screen,
    # so neither hands the terminal back to a prompt.
    if getattr(args, "parallel", "") or args.tui:
        return rc
    return _prompt_for_the_next_input(args, rc, session_id)


def _minted_session_id(explicit: str, mode: str) -> str:
    """This invocation's session id, minted here when the operator named none.

    Through the same owner ACP mints with, and BEFORE the run: the dispatcher
    then knows which session it created, so the end-of-session prompt offers
    that one rather than whatever the repo's newest happens to be. Minting
    reserves nothing on disk, so a run that refuses leaves no session behind.
    """
    from agent6.config.layer import resolved_state_dir  # noqa: PLC0415
    from agent6.sessions.id import unused_session_id  # noqa: PLC0415
    from agent6.types import session_bucket  # noqa: PLC0415

    if explicit:
        return explicit
    return unused_session_id(resolved_state_dir(Path.cwd()), session_bucket(mode))


def _prompt_for_the_next_input(args: argparse.Namespace, rc: int, session_id: str) -> int:
    """Ask for the next input instead of ending, when someone is there to type.

    `run` and `plan` sessions end this way; `ask` does not (a one-shot question
    that becomes a conversation is a different feature). Without a terminal the
    session ends as it always did, with the resume line already printed.

    Only ever THIS invocation's session, and only once it exists on disk: the
    refusal paths above return before any session is created, and resolving the
    repo's newest one instead offered to continue -- then continued -- a
    session this run had nothing to do with. Every follow-up leg runs under
    this invocation's flags (`--max-usd`, `--auto-approve`, ...): the operator
    set them for the run, and a leg that dropped them ran under the config's
    defaults instead.
    """
    from agent6.app._setup import BudgetOverrides, SandboxOverrides  # noqa: PLC0415
    from agent6.sessions.id import SessionIdError  # noqa: PLC0415
    from agent6.sessions.manifest import ManifestError, read_manifest  # noqa: PLC0415
    from agent6.ui.cli._common import resolve_or_newest_layout  # noqa: PLC0415
    from agent6.ui.cli._session_prompt import (  # noqa: PLC0415
        end_of_session_prompt,
        follow_up_on_offer,
        prompting_is_possible,
    )

    if not prompting_is_possible():
        return rc
    try:
        layout = resolve_or_newest_layout(Path.cwd(), session_id)
    except SessionIdError:
        # A refused run discarded its husk, so its minted id matches nothing;
        # there is no session to continue.
        return rc
    if layout is None or not layout.session_dir.is_dir():
        return rc
    # A parked start never ran: its next step is the resume line already
    # printed (once the checkout is free or the changes settled), not a
    # follow-up to a leg that does not exist yet.
    with contextlib.suppress(ManifestError):
        if read_manifest(layout.session_dir).parked_task:
            return rc
    if not follow_up_on_offer(layout.session_dir):
        return rc
    return end_of_session_prompt(
        rc=rc,
        session_id=layout.session_id,
        session_dir=layout.session_dir,
        ask=input,
        config_path=args.config,
        budget_overrides=BudgetOverrides.from_args(args),
        sandbox_overrides=SandboxOverrides.from_args(args),
    )


def _dispatch_plan(args: argparse.Namespace) -> int:
    from agent6.app._setup import BudgetOverrides, SandboxOverrides  # noqa: PLC0415
    from agent6.ui.cli.plan_watch import _cmd_plan_edit, _cmd_plan_show  # noqa: PLC0415
    from agent6.ui.cli.run import _cmd_run  # noqa: PLC0415

    if args.plan_command == "show":
        return _cmd_plan_show(args.session_id)
    if args.plan_command == "edit":
        return _cmd_plan_edit(args.session_id)
    if not args.task:
        error("'plan' needs a task argument (or `plan show/edit <id>`).")
        return 2
    session_id = _minted_session_id(args.session_id, "plan")
    rc = _cmd_run(
        args.config,
        args.task,
        session_id=session_id,
        mode="plan",
        tui=args.tui,
        budget_overrides=BudgetOverrides.from_args(args),
        sandbox_overrides=SandboxOverrides.from_args(args),
        preset=getattr(args, "preset", ""),
    )
    return _prompt_for_the_next_input(args, rc, session_id)


def _dispatch_ask(args: argparse.Namespace) -> int:
    from agent6.app._setup import BudgetOverrides, SandboxOverrides  # noqa: PLC0415
    from agent6.ui.cli._ask import build_ask_session_digest, seed_files  # noqa: PLC0415
    from agent6.ui.cli.run import _cmd_run  # noqa: PLC0415

    # REPL when -i is given, or no question + an interactive stdin.
    repl = args.interactive or (not args.task and sys.stdin.isatty())
    if not args.task and not repl:
        error("'ask' needs a question (in quotes), or -i for the REPL.")
        return 2
    question = args.task
    prefix: list[str] = []
    if args.ask_session_latest or args.ask_session:
        digest = build_ask_session_digest(
            Path.cwd(), args.ask_session, latest=args.ask_session_latest
        )
        if digest is None:
            return 2
        prefix.append(digest)
    if args.ask_files:
        seeds = seed_files(Path.cwd(), args.ask_files)
        if seeds:
            prefix.append(seeds)
    if prefix:
        question = "\n\n".join([*prefix, question]) if question else "\n\n".join(prefix)
    return _cmd_run(
        args.config,
        question,
        mode="ask",
        interactive=repl,
        budget_overrides=BudgetOverrides.from_args(args),
        sandbox_overrides=SandboxOverrides.from_args(args),
        preset=getattr(args, "preset", ""),
    )


def _dispatch_attach(args: argparse.Namespace) -> int:
    from agent6.ui.cli.watch import _cmd_watch_target  # noqa: PLC0415

    return _cmd_watch_target(
        args.target,
        tui=args.tui,
        json_out=args.json,
        since=args.since,
        raw=args.raw,
        config_path=args.config,
    )


def _dispatch_steer(args: argparse.Namespace) -> int:
    from agent6.ui.cli.steer_cmd import _cmd_steer  # noqa: PLC0415

    return _cmd_steer(args.target, args.text, now=args.now)


def _dispatch_answer(args: argparse.Namespace) -> int:
    from agent6.ui.cli.answer_cmd import _cmd_answer  # noqa: PLC0415

    return _cmd_answer(args.target, tuple(args.answers))


def _resolve_target(target: str) -> SessionLayout | None:
    """The named session, or the newest when the operator omitted one, through
    the resolution `attach` and the `sessions` verbs use -- so an ambiguous
    prefix reads as ambiguous here too, and a husk names itself. Prints the
    reason and returns None when there is nothing to act on."""
    from agent6.config.layer import resolved_state_dir  # noqa: PLC0415
    from agent6.sessions.id import SessionIdError  # noqa: PLC0415
    from agent6.ui.cli._common import (  # noqa: PLC0415
        print_no_session_match,
        resolve_or_newest_layout,
    )

    try:
        layout = resolve_or_newest_layout(Path.cwd(), target)
    except SessionIdError as exc:
        error(f"{exc}")
        return None
    if layout is None:
        print_no_session_match(target, resolved_state_dir(Path.cwd()))
    return layout


def _dispatch_exec(args: argparse.Namespace) -> int:
    from agent6.config import ConfigError  # noqa: PLC0415
    from agent6.config.layer import load_effective  # noqa: PLC0415
    from agent6.ui.cli.net_cmds import exec_in_session  # noqa: PLC0415

    # `[SESSION --] CMD...`: only the FIRST `--` separates the optional session
    # from the command, and the command rides verbatim (a later `--`, as in
    # `git log -- path`, belongs to it). No `--` at all = the whole tail is the
    # command, run in the newest session.
    rest: list[str] = list(args.rest)
    target = ""
    if "--" in rest:
        split = rest.index("--")
        before, argv = rest[:split], tuple(rest[split + 1 :])
        if len(before) > 1:
            error(f"at most one session id before `--`, got {' '.join(before)!r}.")
            return 2
        target = before[0] if before else ""
    else:
        argv = tuple(rest)
    if not argv:
        error("give a command (after `--` when naming a session).")
        return 2
    layout = _resolve_target(target)
    if layout is None:
        return 2
    try:
        cfg = load_effective(Path.cwd(), args.config).config
    except ConfigError as exc:
        error(f"{exc}")
        return 2
    return exec_in_session(layout, cfg, Path.cwd(), argv)


def _dispatch_forward(args: argparse.Namespace) -> int:
    from agent6.sessions.ipc import listening_ports, read_session_netns_pid  # noqa: PLC0415
    from agent6.ui.cli.net_cmds import forward, no_session_network_reason  # noqa: PLC0415

    target, port = args.target, args.port
    if port is None and target.isdigit():
        # `forward 8000` means "port 8000 of the newest session": a bare number
        # is a port (the help says so; a numeric session id needs both args).
        target, port = "", int(target)
    layout = _resolve_target(target)
    if layout is None:
        return 2
    if port is None:
        ports = listening_ports(layout.session_dir)
        if not ports:
            reason = (
                f"{layout.session_id} is listening on nothing yet."
                if read_session_netns_pid(layout.session_dir) is not None
                else no_session_network_reason(layout)
            )
            refuse(f"{reason}")
            return 1
        print(f"{layout.session_id} is listening on: {', '.join(str(p) for p in ports)}")
        return 0
    return forward(layout, port, args.local_port)


def _dispatch_sessions(args: argparse.Namespace) -> int:  # noqa: PLR0911
    from agent6.ui.cli.history_cmds import (  # noqa: PLC0415
        _cmd_history_graph,
        _cmd_history_transcript,
    )
    from agent6.ui.cli.sessions_cmds import (  # noqa: PLC0415
        _cmd_commits,
        _cmd_diff,
        _cmd_list,
        _cmd_sessions_dir,
        _cmd_sessions_rm,
        _cmd_stop,
    )
    from agent6.ui.cli.sessions_compare import _cmd_compare  # noqa: PLC0415
    from agent6.ui.cli.sessions_merge import _cmd_merge, _cmd_prune  # noqa: PLC0415
    from agent6.ui.cli.sessions_show import _cmd_status  # noqa: PLC0415

    if args.sessions_command == "list":
        return _cmd_list(as_json=args.list_json)
    if args.sessions_command == "show":
        return _cmd_status(args.session_id, as_json=args.json)
    if args.sessions_command == "diff":
        return _cmd_diff(session_id=args.session_id, stat=args.stat, paths=tuple(args.paths))
    if args.sessions_command == "merge":
        return _cmd_merge(
            session_id=args.session_id,
            strategy=args.strategy,
            into=args.into,
            message=args.message,
            config_path=args.config,
        )
    if args.sessions_command == "compare":
        return _cmd_compare(
            session_ids=tuple(args.session_ids),
            config_path=args.config,
            rejudge=args.rejudge,
        )
    if args.sessions_command == "commits":
        return _cmd_commits(session_id=args.session_id)
    if args.sessions_command == "stop":
        return _cmd_stop(session_id=args.session_id)
    if args.sessions_command == "prune":
        return _cmd_prune(delete_squashed=args.delete_squashed, config_path=args.config)
    if args.sessions_command == "dir":
        return _cmd_sessions_dir(args.session_id)
    if args.sessions_command == "rm":
        return _cmd_sessions_rm(session_id=args.session_id, asks=args.asks)
    if args.sessions_command == "transcript":
        return _cmd_history_transcript(
            args.session_id,
            as_json=args.as_json,
            no_thinking=args.no_thinking,
            tools=args.tools,
            seq=args.seq,
        )
    if args.sessions_command == "graph":
        return _cmd_history_graph(args.session_id)
    raise AssertionError("unreachable")  # pragma: no cover -- earlier branches cover every verb


def _dispatch_tui(args: argparse.Namespace) -> int:
    from agent6.ui.cli.plan_watch import _cmd_tui  # noqa: PLC0415
    from agent6.ui.cli.watch import _cmd_watch_target  # noqa: PLC0415

    if args.target:
        return _cmd_watch_target(
            args.target, tui=True, json_out=False, since=0, raw=False, config_path=args.config
        )
    return _cmd_tui(args.config)


def _dispatch_completions(args: argparse.Namespace) -> int:
    from agent6.ui.cli.completions_cmd import cmd_completions  # noqa: PLC0415

    return cmd_completions(args.shell, print_only=args.print_only)


def _dispatch_web(args: argparse.Namespace) -> int:
    from agent6.ui.cli.web_cmds import _cmd_web  # noqa: PLC0415

    return _cmd_web(
        args.target,
        config_path=args.config,
        host=args.host,
        port=args.port,
        allow_non_loopback=args.allow_non_loopback,
    )


def _dispatch_prompt(args: argparse.Namespace) -> int:
    from agent6.ui.cli.prompt_cmds import _cmd_prompt_show  # noqa: PLC0415

    if args.prompt_command == "show":
        return _cmd_prompt_show(args.config, mode=args.mode, as_json=args.json)
    raise AssertionError("unreachable")  # pragma: no cover -- prompt subparser is required


def _dispatch_resume(args: argparse.Namespace) -> int:
    from agent6.app._setup import BudgetOverrides, SandboxOverrides  # noqa: PLC0415
    from agent6.sessions.id import SessionIdError  # noqa: PLC0415
    from agent6.sessions.manifest import ManifestError, read_manifest  # noqa: PLC0415
    from agent6.ui.cli._common import resolve_or_newest_layout  # noqa: PLC0415
    from agent6.ui.cli.resume import _cmd_resume  # noqa: PLC0415

    if getattr(args, "interactive", False) and not sys.stdin.isatty():
        # Same terminal need as `run -i` (the REPL reads stdin).
        error("-i needs a TTY on stdin (the REPL reads it); drop -i for a headless resume.")
        return 2
    rc = _cmd_resume(
        args.config,
        args.session_id,
        force=args.force,
        tui=args.tui,
        budget_overrides=BudgetOverrides.from_args(args),
        sandbox_overrides=SandboxOverrides.from_args(args),
        preset=args.preset,
        steer=args.steer,
        interactive=getattr(args, "interactive", False),
    )
    # A resumed leg ends the way a fresh one does: asking for the next input
    # (the TUI owns its screen; an ask stays a one-shot).
    if args.tui:
        return rc
    try:
        layout = resolve_or_newest_layout(Path.cwd(), args.session_id)
    except SessionIdError:
        return rc
    if layout is None:
        return rc
    with contextlib.suppress(ManifestError):
        if read_manifest(layout.session_dir).mode == "ask":
            return rc
    return _prompt_for_the_next_input(args, rc, layout.session_id)


def _dispatch_fork(args: argparse.Namespace) -> int:
    from agent6.app._setup import BudgetOverrides, SandboxOverrides  # noqa: PLC0415
    from agent6.ui.cli.fork import _cmd_fork  # noqa: PLC0415

    return _cmd_fork(
        args.config,
        args.session_id,
        at_turn=args.at_turn,
        new_session_id=args.new_session_id,
        no_run=args.no_run,
        tui=args.tui,
        budget_overrides=BudgetOverrides.from_args(args),
        sandbox_overrides=SandboxOverrides.from_args(args),
        steer=args.steer,
    )


def _dispatch_config(args: argparse.Namespace) -> int:  # noqa: PLR0911
    from agent6.ui.cli.config_cmds import (  # noqa: PLC0415
        _cmd_config_fill,
        _cmd_config_fix,
        _cmd_config_get,
        _cmd_config_path,
        _cmd_config_presets,
        _cmd_config_set,
        _cmd_config_show,
        _cmd_config_unset,
        _config_list_edit,
    )

    if args.config_command == "show":
        return _cmd_config_show(
            args.config,
            as_json=args.as_json,
            keys=args.keys,
            descriptions=args.descriptions,
            machine=args.machine_file,
        )
    if args.config_command == "fill":
        return _cmd_config_fill(force=args.force)
    if args.config_command == "path":
        return _cmd_config_path()
    if args.config_command == "presets":
        return _cmd_config_presets(args.config)
    if args.config_command == "get":
        return _cmd_config_get(args.config, args.key, machine=args.machine_file)
    if args.config_command == "set":
        return _cmd_config_set(
            args.key, args.value, repo=args.repo, machine=args.machine_file, config_path=args.config
        )
    if args.config_command == "unset":
        return _cmd_config_unset(
            args.key, repo=args.repo, machine=args.machine_file, config_path=args.config
        )
    if args.config_command in ("add", "remove"):
        return _config_list_edit(
            args.key,
            args.value,
            repo=args.repo,
            machine=args.machine_file,
            add=args.config_command == "add",
        )
    if args.config_command == "fix":
        return _cmd_config_fix(machine=args.machine_file)
    raise AssertionError("unreachable")  # pragma: no cover -- config subparser is required


def _dispatch_check(args: argparse.Namespace) -> int:
    from agent6.ui.cli.check_cmds import _cmd_check  # noqa: PLC0415

    return _cmd_check(args.config, section=args.section)


def _dispatch_connect(args: argparse.Namespace) -> int:
    from agent6.ui.cli.connect import _cmd_connect  # noqa: PLC0415

    return _cmd_connect(
        provider=args.provider, to_repo=args.repo, verify=args.verify, logout=args.logout
    )


def _dispatch_model(args: argparse.Namespace) -> int:
    from agent6.ui.cli.model import _cmd_model  # noqa: PLC0415

    return _cmd_model(
        args.config,
        role=args.role,
        provider=args.provider,
        model=args.model,
        effort=args.effort,
        to_repo=args.repo,
    )


def _dispatch_memory(args: argparse.Namespace) -> int:
    from agent6.ui.cli.memory_cmds import (  # noqa: PLC0415
        _cmd_memory_add,
        _cmd_memory_decisions,
        _cmd_memory_list,
        _cmd_memory_rm,
        _cmd_memory_show,
    )

    if args.memory_command == "add":
        return _cmd_memory_add(args.name, args.body)
    if args.memory_command == "list":
        return _cmd_memory_list()
    if args.memory_command == "show":
        return _cmd_memory_show(args.name)
    if args.memory_command == "rm":
        return _cmd_memory_rm(args.name)
    if args.memory_command == "decisions":
        return _cmd_memory_decisions()
    raise AssertionError("unreachable")  # pragma: no cover -- memory subparser is required


def _dispatch_skills(args: argparse.Namespace) -> int:
    from agent6.ui.cli.skills_cmds import (  # noqa: PLC0415
        _cmd_skills_disable,
        _cmd_skills_enable,
        _cmd_skills_install,
        _cmd_skills_list,
        _cmd_skills_remove,
        _cmd_skills_update,
    )

    if args.skills_command == "install":
        return _cmd_skills_install(args.url, force=args.force, config_path=args.config)
    if args.skills_command == "update":
        return _cmd_skills_update(args.name)
    if args.skills_command == "list":
        return _cmd_skills_list(args.config)
    if args.skills_command == "enable":
        return _cmd_skills_enable(
            args.name, always=args.always, repo=args.repo, config_path=args.config
        )
    if args.skills_command == "disable":
        return _cmd_skills_disable(args.name, repo=args.repo, config_path=args.config)
    if args.skills_command == "remove":
        return _cmd_skills_remove(args.name, args.config)
    raise AssertionError("unreachable")  # pragma: no cover -- skills subparser is required


def _dispatch_ps(args: argparse.Namespace) -> int:
    from agent6.ui.cli.ps_cmd import cmd_ps  # noqa: PLC0415

    return cmd_ps(as_json=args.json)


def _dispatch_history(args: argparse.Namespace) -> int:
    from agent6.ui.cli.history_cmds import _cmd_history_search  # noqa: PLC0415

    if args.history_command == "search":
        if not args.query:
            error("'history' needs a query, the pattern to search for.")
            return 2
        return _cmd_history_search(args.query, fixed=not args.regex, session_id=args.session)
    raise AssertionError("unreachable")  # pragma: no cover -- history subparser is required


def _dispatch_init(args: argparse.Namespace) -> int:
    from agent6.ui.cli.init_cmds import _cmd_init  # noqa: PLC0415

    return _cmd_init(ecosystem=args.ecosystem, assume_yes=args.yes, config_path=args.config)


def _dispatch_review(args: argparse.Namespace) -> int:
    from agent6.ui.cli.review_cmds import _cmd_review  # noqa: PLC0415

    return _cmd_review(
        args.config,
        base=args.base,
        head=args.head,
        paths=tuple(args.paths),
        model_override=args.model,
        reviewers=args.reviewers,
        personas=args.personas,
    )


def _dispatch_mcp(args: argparse.Namespace) -> int:
    from agent6.ui.cli.mcp_connect import (  # noqa: PLC0415
        cmd_mcp_connect,
        cmd_mcp_list,
        cmd_mcp_remove,
    )
    from agent6.ui.mcp_server import run_server  # noqa: PLC0415

    if args.mcp_command == "serve":
        return run_server(args.config)
    if args.mcp_command == "connect":
        return cmd_mcp_connect(
            args.name,
            command=args.server_command,
            url=args.url,
            token_env=args.token_env,
            pass_env=args.pass_env,
            to_repo=args.to_repo,
            config_path=args.config,
        )
    if args.mcp_command == "remove":
        return cmd_mcp_remove(args.name, to_repo=args.to_repo, config_path=args.config)
    return cmd_mcp_list(args.config)


def _dispatch_machine(args: argparse.Namespace) -> int:  # noqa: PLR0911
    from agent6.ui.cli.machine_check import (  # noqa: PLC0415
        _cmd_machine_check,
        _cmd_machine_graph,
        _cmd_machine_test,
    )
    from agent6.ui.cli.machine_cmds import (  # noqa: PLC0415
        _cmd_machine_create,
        _cmd_machine_list,
        _cmd_machine_poke,
        _cmd_machine_replay,
        _cmd_machine_run,
        _cmd_machine_status,
        _cmd_machine_stop,
    )

    if args.machine_command == "list":
        return _cmd_machine_list()
    if args.machine_command == "check":
        return _cmd_machine_check(args.file)
    if args.machine_command == "test":
        return _cmd_machine_test(args.file, blackboard=args.blackboard)
    if args.machine_command == "graph":
        return _cmd_machine_graph(args.file, fmt=args.format)
    if args.machine_command == "run":
        return _cmd_machine_run(
            args.file,
            config_path=args.config,
            exit_on_wait=args.exit_on_wait,
            disable_sandbox=args.dangerously_disable_sandbox,
            auto_approve=args.auto_approve,
            no_commands=args.no_commands,
        )
    if args.machine_command == "status":
        return _cmd_machine_status(args.machine_id)
    if args.machine_command == "poke":
        return _cmd_machine_poke(args.machine_id, data=args.data, message=args.message)
    if args.machine_command == "stop":
        return _cmd_machine_stop(args.machine_id)
    if args.machine_command == "replay":
        return _cmd_machine_replay(args.machine_id)
    if args.machine_command == "create":
        return _cmd_machine_create(
            args.task, output=args.output, max_attempts=args.max_attempts, config_path=args.config
        )
    raise AssertionError("unreachable")  # pragma: no cover -- machine subparser is required


def _dispatch_acp(args: argparse.Namespace) -> int:
    from agent6.ui.acp import serve_acp  # noqa: PLC0415

    return serve_acp(config_path=args.config)


def _dispatch_system(args: argparse.Namespace) -> int:
    from agent6.ui.cli.system_cmds import _cmd_system_apparmor  # noqa: PLC0415

    if args.system_command == "apparmor":
        return _cmd_system_apparmor(args.action)
    raise AssertionError("unreachable")  # pragma: no cover -- system subparser is required


# command -> per-family dispatcher. Mirrors the `_*_args.py` parser grouping:
# one handler per top-level command, each fanning out over its own subcommands.
_DISPATCH: dict[str, Callable[[argparse.Namespace], int]] = {
    "run": _dispatch_run,
    "plan": _dispatch_plan,
    "ask": _dispatch_ask,
    "attach": _dispatch_attach,
    "steer": _dispatch_steer,
    "answer": _dispatch_answer,
    "exec": _dispatch_exec,
    "forward": _dispatch_forward,
    "sessions": _dispatch_sessions,
    "tui": _dispatch_tui,
    "completions": _dispatch_completions,
    "web": _dispatch_web,
    "prompt": _dispatch_prompt,
    "resume": _dispatch_resume,
    "fork": _dispatch_fork,
    "config": _dispatch_config,
    "check": _dispatch_check,
    "connect": _dispatch_connect,
    "model": _dispatch_model,
    "memory": _dispatch_memory,
    "skills": _dispatch_skills,
    "history": _dispatch_history,
    "ps": _dispatch_ps,
    "init": _dispatch_init,
    "review": _dispatch_review,
    "machine": _dispatch_machine,
    "mcp": _dispatch_mcp,
    "acp": _dispatch_acp,
    "system": _dispatch_system,
}


def main(argv: list[str] | None = None) -> int:
    # A redirected stdout is block-buffered, so a run's log file would stay
    # empty until exit: line-buffer it so every line lands as it is printed.
    if isinstance(sys.stdout, io.TextIOWrapper):
        sys.stdout.reconfigure(line_buffering=True)
    parser = build_parser()
    argcomplete.autocomplete(parser)
    raw = sys.argv[1:] if argv is None else argv
    # Bare `agent6` (no command, no -h/--version): print help rather than the
    # terse argparse "required: <command>" error. The boring, expected thing.
    if _command_index(raw) is None and not any(a in ("-h", "--help", "--version") for a in raw):
        parser.print_help()
        return 0
    args = parser.parse_args(_inject_default_verb(raw))
    # `agent6 system ...` is a privileged host-setup command that legitimately
    # runs as root (it writes /etc and reloads AppArmor); it does not run the
    # LLM, so it is exempt from the "no LLM agent as root" gate.
    if args.command != "system":
        root_rc = _enforce_root_policy(getattr(args, "allow_root", False))
        if root_rc is not None:
            return root_rc
    handler = _DISPATCH.get(args.command)
    if handler is None:  # pragma: no cover -- the top-level subparser is required
        parser.error("unknown command")
    try:
        return handler(args)
    except EventWriteError as exc:
        # A lifecycle stopped because the durable run journal could not be
        # appended; its finally already released locks and egress. One report
        # here beats a per-command arm in every lifecycle.
        error(f"{exc}")
        return 1
