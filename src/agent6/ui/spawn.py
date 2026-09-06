# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Find the agent6 executable and spawn it detached.

Shared by every front-end (TUI hub, machines page, web server) so a UI action
shells out to the same CLI a user would run, never doing the work in-process:
one argv head, one detached environment, one new-work spawn."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import IO

from agent6.directive import DirectiveError, parse_directive, steer_problem
from agent6.models.validate import directive_model_refusal
from agent6.paths import state_dir
from agent6.sandbox.jail import keep_out_of_the_sweep
from agent6.sessions.id import SessionIdError, resolve_session
from agent6.sessions.ipc import read_worker_pid
from agent6.sessions.layout import LOGS_NAME
from agent6.sessions.lock import repo_writer_held, repo_writer_holder
from agent6.types import OPERATOR_MODES
from agent6.viewmodel.listing import session_dirs


def agent6_exe() -> str:
    """The agent6 executable that launched this TUI (so a spawned child uses the
    same install), falling back to the entry on PATH."""
    argv0 = Path(sys.argv[0])
    if argv0.name.startswith("agent6") and argv0.exists():
        return str(argv0.resolve())
    # A view started as `python -m agent6.ui.tui`: the binary of the same
    # install sits beside its interpreter.
    beside = Path(sys.executable).with_name("agent6")
    if beside.exists():
        return str(beside.resolve())
    return shutil.which("agent6") or "agent6"


def agent6_argv(config_path: Path | None) -> list[str]:
    """The argv head every spawn starts from: the exe, plus `--config F` when
    the front-end runs under an explicit config, so spawned work runs under
    the config the operator gave the front-end."""
    argv = [agent6_exe()]
    if config_path is not None:
        argv += ["--config", str(config_path)]
    return argv


# The environment of work a front-end drives over the bridge: approvals and
# questions WAIT for a front-end instead of the headless default's fabricated
# empty answer (`spawn_and_confirm` sets it for every child it starts), and a
# run's headless child streams its reasoning deltas to logs.jsonl so a live
# view renders them.
DETACHED_AWAY_ENV: dict[str, str] = {"AGENT6_DETACHED_AWAY": "wait"}
DETACHED_RUN_ENV: dict[str, str] = {"AGENT6_STREAM_TO_LOG": "1", **DETACHED_AWAY_ENV}


def spawn_new_work(  # noqa: PLR0911
    cwd: Path, mode: str, task: str, *, preset: str = "", config_path: Path | None = None
) -> tuple[Path | None, str]:
    """Start `agent6 <mode> [--preset P] -- <task>` detached from a hub and
    return the new session's dir to open, or `(None, why)`.

    A `/parallel [spec] <task> ...` message (run mode only) fans out one
    detached `agent6 run --parallel <spec>` per segment (omitted spec = one
    isolated lane); a malformed directive is refused before any spawn, any
    segment's failure fails the whole message, naming the lanes already
    running (they keep running), and the first segment's dir is returned. A
    plain run into a checkout another run is driving is refused here, at once,
    rather than parked by the child after the locate wait; a fan-out takes no
    such lock (its lanes clone the checkout)."""
    if mode not in OPERATOR_MODES:
        return None, f"unknown mode {mode!r}"
    if not task.strip():
        return None, "empty task"
    segments = None
    if mode == "run":
        try:
            segments = parse_directive(task)
        except DirectiveError as exc:
            return None, str(exc)
    if segments is None:
        if mode == "run" and repo_writer_held(state := state_dir(cwd)):
            holder = repo_writer_holder(state) or "another run"
            return None, (
                f"run {holder} is already driving this checkout; steer it with this task"
                " (or /parallel it) from its run view, or wait for it to finish"
            )
        return _spawn_run(cwd, mode, task, preset=preset, spec="", config_path=config_path)
    refusal = directive_model_refusal(cwd, segments, config_path)
    if refusal is not None:
        return None, refusal
    first: Path | None = None
    lines: list[str] = []
    failed = False
    for i, seg in enumerate(segments, 1):
        session_dir, err = _spawn_run(
            cwd, "run", seg.task, preset=preset, spec=seg.spec or "1", config_path=config_path
        )
        if session_dir is None:
            lines.append(f"lane {i} ({seg.task}): {err}")
            failed = True
            continue
        lines.append(f"lane {i} ({seg.task}): running as {session_dir.name}")
        if first is None:
            first = session_dir
    if failed:
        # Open the run XOR show the error: a partial failure must not vanish
        # behind a surviving lane, and a resend must not double-launch one.
        return None, "\n".join(lines)
    assert first is not None  # no failures => every segment produced a dir
    return first, ""


def _spawn_run(
    cwd: Path, mode: str, task: str, *, preset: str, spec: str, config_path: Path | None
) -> tuple[Path | None, str]:
    """One detached `agent6 <mode> [--preset P] [--parallel S] -- <task>`,
    located by its new session dir. `--` ends option parsing: a task starting
    with `-` is never read as a flag."""
    argv = [*agent6_argv(config_path), mode]
    if preset:
        argv += ["--preset", preset]
    if spec:
        argv += ["--parallel", spec]
    argv += ["--", task]
    state = state_dir(cwd)
    return spawn_and_locate(
        argv,
        cwd,
        before=set(session_dirs(state)),
        list_dirs=lambda: session_dirs(state),
        env={**os.environ, **DETACHED_RUN_ENV},
    )


def spawn_detached_resume(
    cwd: Path,
    session_id: str,
    *,
    steer: str = "",
    preset: str = "",
    config_path: Path | None = None,
    flags: Sequence[str] = (),
) -> str:
    """Start a detached `agent6 resume <session_id>` (new session, no stdio)
    so a run keeps going in the background after the operator detaches, and
    return "" once the child owns the run, else why it did not.

    Owning the run = the child's pid is the run's worker.pid, which `resume`
    writes once its preflight passed (locks, snapshot, git guards, config,
    isolation and provider checks); a child that exits before that hands back
    its own refusal through `spawn_and_confirm`, the early-exit capture every
    hub spawn shares. The caller must have released the run's worker lock first,
    so the child acquires it cleanly. *cwd* is the checkout whose state dir
    holds the session (a fork's origin, never its worktree), as `agent6
    resume` itself reads it.

    A non-empty *steer* rides along as `--steer=TEXT` (the `=` form, so a
    follow-up starting with `-` cannot read as an option): the resume injects
    it as the first steering instruction. Operator-typed text, never LLM output.
    A malformed directive as *steer* is refused here, with the message the
    child would print. A non-empty *preset* is the `--preset` the leg
    continues under; *flags* are further `resume` options the leg runs under
    (the detaching invocation's own overrides). argv is the agent6 exe + the
    run id (never LLM output)."""
    if steer and (problem := steer_problem(steer)) is not None:
        return problem
    try:
        session_dir = resolve_session(state_dir(cwd), session_id).session_dir
    except SessionIdError as exc:
        return str(exc)
    argv = [*agent6_argv(config_path), "resume", session_id]
    if preset:
        argv.append(f"--preset={preset}")
    if steer:
        argv.append(f"--steer={steer}")
    argv.extend(flags)
    return spawn_and_confirm(
        argv,
        cwd,
        started=lambda pid: read_worker_pid(session_dir) == pid,
        extra_env=DETACHED_RUN_ENV,
    )


# Subcommand groups whose verb is the SECOND argv word ("machine run",
# "sessions prune", "config set"); everything else is a one-word subcommand whose
# next arg is already a value.
_COMMAND_GROUPS = frozenset({"machine", "sessions", "config"})


def subcommand_label(argv: list[str]) -> str:
    """The agent6 subcommand named by *argv*, for diagnostics: "machine run",
    not a bare "machine" (or worse, "run" with the task word attached)."""
    if len(argv) < 2:
        return argv[0]
    label = argv[1]
    if label in _COMMAND_GROUPS and len(argv) > 2 and not argv[2].startswith("-"):
        return f"{label} {argv[2]}"
    return label


def capture_message(*streams: str) -> str:
    """Captured CLI output as front-end message text, its console decorations
    dropped: "[agent6] " marks agent6's own lines among pass-through git
    output, "ERROR: " marks a failure, and a toast or an API error field
    already says both."""
    lines = [
        ln.removeprefix("[agent6] ").removeprefix("ERROR: ").strip()
        for ln in "\n".join(streams).splitlines()
    ]
    return "\n".join(ln for ln in lines if ln)


def _child_exit_message(label: str, rc: int | None, captured: str) -> str:
    """What a front-end shows when a spawned child ended before it began: the
    child's own words (a REFUSING / PARKED line, or its error with the
    `ERROR: ` marker dropped), or the exit code when it said nothing."""
    said = capture_message(captured)
    return said or f"agent6 {label} exited {rc} without a word"


def _not_started_message(label: str, timeout_s: float, captured: str) -> str:
    said = capture_message(captured)
    return (
        f"agent6 {label} has not reported starting within {timeout_s:.0f}s"
        " (`agent6 ps` shows whether it is running)" + (f":\n{said}" if said else "")
    )


def run_cli_capture(argv: list[str], cwd: Path, *, timeout_s: float = 120.0) -> tuple[bool, str]:
    """Run a quick agent6 subcommand synchronously, capturing its output, and
    return `(ok, message)`. For the fast, foreground CLI ops a front-end drives
    the same way a user would: `sessions merge`, `sessions prune`, `config set`. argv is
    fixed (the agent6 exe + operator-chosen args), never LLM output."""
    try:
        proc = subprocess.run(
            argv,
            cwd=str(cwd),
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, f"failed to run agent6 {subcommand_label(argv)}: {exc}"
    message = capture_message(proc.stdout, proc.stderr)
    return proc.returncode == 0, message or f"exit {proc.returncode}"


def spawn_and_confirm(
    argv: list[str],
    cwd: Path,
    *,
    started: Callable[[int], bool],
    extra_env: Mapping[str, str] | None = None,
    timeout_s: float = 25.0,
) -> str:
    """Spawn *argv* detached and return "" once *started(child_pid)* reports
    the child took ownership of its work, else why it did not (`_spawn_and_wait`).
    A child that exits 0 without the signal is a clean fast completion.

    The child runs under this process's environment plus the away marker
    (`DETACHED_AWAY_ENV`: every child started here is driven from a hub, so
    its asks and approvals wait for that front-end) plus *extra_env*.

    The pid-signalled analogue of `spawn_and_locate`, behind `machine run` and
    a detached `resume`: their refusals (lock held, network refusal, bad
    bundle, a finished run) print to stderr and exit nonzero without ever
    starting, which a fire-and-forget spawn (stderr to /dev/null) silently
    swallowed."""
    _, err = _spawn_and_wait(
        argv,
        cwd,
        ready=lambda pid: True if started(pid) else None,
        env={**os.environ, **DETACHED_AWAY_ENV, **(extra_env or {})},
        timeout_s=timeout_s,
        clean_exit=True,
    )
    return err


def _stderr_tail(err: IO[str], limit: int = 2000) -> str:
    """The end of a spawn's captured-stderr temp file: at most *limit* chars,
    cut at a line start so a refusal never begins mid-word."""
    err.flush()
    text = Path(err.name).read_text(encoding="utf-8", errors="replace")
    if len(text) <= limit:
        return text
    tail = text[-limit:]
    nl = tail.find("\n")
    return tail[nl + 1 :] if 0 <= nl < len(tail) - 1 else tail


def _located(list_dirs: Callable[[], list[Path]], before: set[Path]) -> Path | None:
    """The newest dir from *list_dirs* not in *before* whose logs.jsonl exists."""
    for d in list_dirs():
        if d not in before and (d / LOGS_NAME).exists():
            return d
    return None


def spawn_and_locate(
    argv: list[str],
    cwd: Path,
    *,
    before: set[Path],
    list_dirs: Callable[[], list[Path]],
    env: dict[str, str] | None = None,
    timeout_s: float = 25.0,
) -> tuple[Path | None, str]:
    """Spawn *argv* detached, then poll *list_dirs* for a NEW dir (not in *before*)
    whose `logs.jsonl` exists, and return `(dir, "")` so the caller can hand it
    to the dashboard; `(None, message)` on any failure (`_spawn_and_wait`).

    The shared launch+watch path behind both "start a run" (hub) and "create a
    machine" (machines page): spawn the same CLI a user would, then watch the new
    log dir live."""
    return _spawn_and_wait(
        argv, cwd, ready=lambda _pid: _located(list_dirs, before), env=env, timeout_s=timeout_s
    )


def _spawn_and_wait[T](
    argv: list[str],
    cwd: Path,
    *,
    ready: Callable[[int], T | None],
    env: dict[str, str] | None,
    timeout_s: float,
    clean_exit: T | None = None,
) -> tuple[T | None, str]:
    """Spawn *argv* detached (non-TTY stdio, new session, so the child never
    opens its own TUI) with an early-exit stderr capture, and poll
    *ready(child_pid)* until it answers: `(answer, "")`. A child that exits
    first hands back its stderr tail (its own refusal, or the exit code when it
    said nothing), unless it exited 0 and *clean_exit* stands in for the
    answer; nothing by *timeout_s* hands back what the child said so far."""
    label = subcommand_label(argv)
    err = tempfile.NamedTemporaryFile(  # noqa: SIM115 - closed in finally
        mode="w+", suffix=".agent6-launch.err", delete=False
    )
    try:
        try:
            proc = subprocess.Popen(
                argv,
                cwd=str(cwd),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=err,
                start_new_session=True,
                env=env,
            )
        except OSError as exc:
            return None, f"failed to start agent6 {label}: {exc}"
        keep_out_of_the_sweep(proc.pid)
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if (found := ready(proc.pid)) is not None:
                return found, ""
            rc = proc.poll()
            if rc is not None:
                # Recheck once: the answer may have landed in the same instant.
                if (found := ready(proc.pid)) is not None:
                    return found, ""
                if rc == 0 and clean_exit is not None:
                    return clean_exit, ""
                return None, _child_exit_message(label, rc, _stderr_tail(err))
            time.sleep(0.2)
        return None, _not_started_message(label, timeout_s, _stderr_tail(err))
    finally:
        # The detached child keeps the unlinked-but-open inode as its stderr
        # until it exits; its real output is its own log, and this capture
        # only feeds the early-exit and timeout diagnostics.
        err.close()
        Path(err.name).unlink(missing_ok=True)
