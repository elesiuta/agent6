# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Cross-cutting CLI helpers: run dirs, budget flags, key/root checks."""

from __future__ import annotations

import argparse
import os
import shlex
import sys
from pathlib import Path

from agent6.app.reporter import STDIO_REPORTER
from agent6.paths import (
    effective_user,
    is_root,
    root_optin_enabled,
    state_dir,
)
from agent6.sessions.id import SessionIdError, resolve_session
from agent6.sessions.layout import (
    SESSION_BUCKETS,
    SessionLayout,
    bucket_dir,
    layout_of,
)


def _sub(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
    name: str,
    *,
    help: str,
) -> argparse.ArgumentParser:
    """`add_parser` with *help* as the leaf's own description; the parent's
    command list shows its first sentence, so `agent6 --help` reads as a list
    of commands and the detail waits in `agent6 <command> --help`."""
    summary, _, rest = help.partition(". ")
    return subparsers.add_parser(name, help=summary + "." if rest else summary, description=help)


SESSION_ID = "Session id or unambiguous prefix"
SESSION_ID_HELP = f"{SESSION_ID}; omit for the newest."


def _add_session_id(
    parser: argparse.ArgumentParser, completer: object, *, help_text: str = SESSION_ID_HELP
) -> None:
    """The positional a verb acting on a session takes: an id or unambiguous
    prefix, omitted for the newest; *completer* offers the ids the verb
    accepts (passed in: `completers` imports this module)."""
    arg = parser.add_argument("session_id", nargs="?", default="", help=help_text)
    arg.completer = completer  # type: ignore[attr-defined]


def _add_config_flag(parser: argparse.ArgumentParser) -> None:
    """A subcommand's `--config FILE`. Its default is SUPPRESS, not None: the
    subparser sets `config` only when the flag follows the subcommand, so both
    `agent6 --config F run` and `agent6 run --config F` work and the top-level
    flag supplies the always-present default."""
    parser.add_argument(
        "--config",
        type=Path,
        default=argparse.SUPPRESS,
        metavar="FILE",
        help="Explicit config file (layered over global + repo configs).",
    )


def _add_budget_flags(parser: argparse.ArgumentParser) -> None:
    """Add per-run budget override flags (override `[budget]` config)."""
    parser.add_argument(
        "--max-usd",
        type=float,
        default=None,
        metavar="USD",
        help="Override [budget].max_usd for this run (-1 unlimited, 0 refuses metered calls).",
    )
    parser.add_argument(
        "--max-percent",
        type=float,
        default=None,
        metavar="PCT",
        help=(
            "Override [budget].max_percent for this run: the plan percentage points a"
            " subscription run may consume (-1 unlimited, 0 refuses plan-metered calls)."
        ),
    )
    parser.add_argument(
        "--max-tokens-fallback",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Override [budget].max_tokens_fallback for this run: the input+output"
            " token cap for calls with no price data (-1 unlimited, 0 refuses"
            " unmetered calls)."
        ),
    )


def _add_sandbox_flags(parser: argparse.ArgumentParser) -> None:
    """Add the per-invocation sandbox/approval override flags (every paid
    command carries both: run/plan/ask/resume and machine run).

    `--dangerously-disable-sandbox` runs the agent's commands unconfined on
    the host (equivalent to a one-off `sandbox.isolation = "none"`); the env
    `AGENT6_DANGEROUSLY_DISABLE_SANDBOX=1` does the same. `--auto-approve`
    auto-approves `run_command` for this invocation: it upgrades
    `sandbox.run_commands` from ask to yes and never resurrects a withheld
    no. Approval is skipped; confinement still depends on `sandbox.isolation`.
    """
    parser.add_argument(
        "--dangerously-disable-sandbox",
        action="store_true",
        help=(
            "Run the agent's commands unconfined on the host (no Landlock/"
            "seccomp/namespaces). Only for a disposable or already-isolated"
            " machine; the host becomes the only boundary."
        ),
    )
    approval = parser.add_mutually_exclusive_group()
    approval.add_argument(
        "--auto-approve",
        action="store_true",
        help=(
            "Auto-approve every jailed command for this run instead of prompting."
            " Raises sandbox.run_commands to `yes` unless it is `no`, which the"
            " flag never overrides. Confinement still depends on"
            " sandbox.isolation; combined with --dangerously-disable-sandbox it"
            " hands the agent unprompted host access."
        ),
    )
    approval.add_argument(
        "--no-commands",
        action="store_true",
        help=(
            "Withhold every jailed command for this session (sets"
            " sandbox.run_commands = no): no run_command, no verify gate, no"
            " background commands. What `/btw` asks its side question with."
        ),
    )


def editor_argv() -> list[str] | None:
    """$EDITOR as argv (default: vi), or None after printing the refusal when its
    quoting is unbalanced."""
    editor = os.environ.get("EDITOR", "vi")
    try:
        return shlex.split(editor) or ["vi"]
    except ValueError as exc:
        error(f"$EDITOR {editor!r} is not a valid command: {exc}")
        return None


def sgr(text: str, code: str) -> str:
    """Wrap *text* in an ANSI style, tty only, so piped output stays plain.
    The one place the CLI's faded/bold hints are styled."""
    return f"\x1b[{code}m{text}\x1b[0m" if sys.stdout.isatty() else text


def _runs_dir(repo_root: Path) -> Path:
    """The `runs/` directory under the per-repo state dir."""
    return bucket_dir(state_dir(repo_root), "runs")


def _plans_dir(repo_root: Path) -> Path:
    """The `plans/` directory under the per-repo state dir."""
    return bucket_dir(state_dir(repo_root), "plans")


# What a fresh install is told when it has nothing yet. One string: the same
# first contact whichever command the operator happened to type, and it names
# the way out rather than the directory that is missing.


def nothing_yet(what: str = "sessions") -> str:
    return f'no {what} yet. Start one with `agent6 run "<task>"`.'


# The stderr conventions belong to app.reporter (REFUSING:, ERROR:, [agent6]
# WARNING:); every CLI message goes through these, so the wording has one owner.
error = STDIO_REPORTER.error
refuse = STDIO_REPORTER.refuse
warn = STDIO_REPORTER.warn

# The one sentence every command's id argument prints; a command whose default
# differs appends its own clause.
MACHINE_ID_HELP = "Machine id (a directory under the per-repo state dir's machines/)."
REPO_FLAG_HELP = "Write to the per-repo config instead of the global config."


def print_nothing_yet(what: str = "sessions") -> None:
    """Say there is nothing yet, and how to change that.

    An empty state dir is not a fault, so it must not read as one: an ERROR
    about a missing directory would tell a new operator their install is
    broken.
    """
    print(nothing_yet(what), file=sys.stderr)


def print_no_session_match(query: str, state: Path) -> None:
    """The one missing-session error, shared by every command that resolves one:
    name the query and where it looked (never the bucket-layout internals), or
    the same first-contact copy as `sessions` when there is nothing at all."""
    if query:
        print(f"ERROR: no session matches {query!r} (looked under {state})", file=sys.stderr)
    else:
        print_nothing_yet()


def session_bucket_dirs(repo_root: Path) -> list[Path]:
    """The session bucket dirs under `sessions/` in the state
    dir, the cross-bucket scope for latest-run resolution and history. A missing
    bucket is still listed; iterators skip non-dirs."""
    state = state_dir(repo_root)
    return [bucket_dir(state, subdir) for subdir in SESSION_BUCKETS]


def all_session_dirs(repo_root: Path) -> list[Path]:
    """Every run directory across all SESSION_BUCKETS. So latest-run resolution and
    history search cover every bucket, not just runs/ (a bare `attach`
    or `history search` right after an `ask` must find that ask)."""
    dirs: list[Path] = []
    for bucket in session_bucket_dirs(repo_root):
        if bucket.is_dir():
            dirs.extend(p for p in bucket.iterdir() if p.is_dir())
    return dirs


def resolve_session_layout(
    repo_root: Path, query: str, *, allow_husk: bool = False
) -> SessionLayout:
    """Resolve a run id (or unique prefix) across every run-style bucket --
    one per mode under `sessions/` -- returning a `SessionLayout`
    with the matching subdir.

    `agent6 run` lives under `runs/`, `plan` under `plans/`, `agent6 ask`
    under `asks/`, and `machine create` authoring logs under
    `sessions/machines/`; read-only commands (`sessions show`/`attach`/
    `history search`) use this so anything
    a listing shows is also inspectable by id. Raises `SessionIdError` if no run
    matches in any bucket.

    A HUSK (no manifest, no log: it crashed before it ever started) refuses
    with the remedy, so every surface says the same thing instead of showing
    an empty session and advising a resume that fails. `allow_husk` is for
    `sessions rm`, whose whole job is deleting one.
    """
    layout = resolve_session(state_dir(repo_root), query)
    from agent6.viewmodel import is_session_husk  # noqa: PLC0415

    if not allow_husk and is_session_husk(layout.session_dir):
        raise SessionIdError(
            f"session {layout.session_id} crashed before it ever started (no log, nothing"
            f" to resume); `agent6 sessions rm {layout.session_id}` removes it"
        )
    return layout


def resolve_target(target: str) -> SessionLayout | None:
    """The named session, or the newest when the operator omitted one, for a
    verb run from the checkout: an ambiguous prefix reads as ambiguous, a husk
    names itself, and nothing to act on prints why and returns None."""
    try:
        layout = resolve_or_newest_layout(Path.cwd(), target)
    except SessionIdError as exc:
        error(f"{exc}")
        return None
    if layout is None:
        print_no_session_match(target, state_dir(Path.cwd()))
    return layout


def newest_layout_holding(repo_root: Path, child: str) -> SessionLayout | None:
    """The newest session across every bucket whose dir holds *child*.

    `history graph` / `history transcript` each scanned runs/ and then built a
    runs/ layout from the name, so a session in any other bucket was both
    invisible and, if named explicitly, resolved to a directory that does not
    exist.
    """
    candidates = [d for d in all_session_dirs(repo_root) if (d / child).is_dir()]
    if not candidates:
        return None
    from agent6.viewmodel import session_mtime  # noqa: PLC0415

    return layout_of(max(candidates, key=session_mtime))


def resolve_or_newest_layout(
    repo_root: Path, session_id: str, *, allow_husk: bool = False
) -> SessionLayout | None:
    """Resolve an explicit *session_id* across every run-style bucket, or fall back to
    the newest run across all buckets when *session_id* is empty.

    Returns the resolved `SessionLayout`. Returns None only for the empty-*session_id*
    "no sessions exist" case, so the caller phrases its own 'none yet' message. Raises
    `SessionIdError` (`.no_match` set only when nothing matched) when an explicit id has
    no or many matches. The one 'a run by id, or the latest' resolution behind
    `attach` / `sessions stop` / `sessions show`: a new such command resolves the
    same way instead of re-deriving the id-or-newest glue.
    """
    if session_id:
        return resolve_session_layout(repo_root, session_id, allow_husk=allow_husk)
    from agent6.viewmodel import newest_session_dir  # noqa: PLC0415

    newest = newest_session_dir(session_bucket_dirs(repo_root))
    if newest is None:
        return None
    return layout_of(newest)


def _enforce_root_policy(allow_root: bool) -> int | None:
    """Gate running as root behind an explicit opt-in.

    Returns a non-zero exit code (to refuse) when running as root without
    `--allow-root` / `AGENT6_ALLOW_ROOT=1`; returns None to proceed. When
    proceeding as root it prints a loud banner. Privileges are not dropped:
    under sudo the LLM's verify/run commands need to run as root inside the
    jail, so the jail is the boundary.
    """
    if not is_root():
        return None
    if not root_optin_enabled(allow_root):
        print(
            "REFUSING: running as root. An LLM-driven agent as root is dangerous;"
            " if a task genuinely needs it, re-run as `agent6 --allow-root <command> ...`"
            " (the flag goes before the command), or set AGENT6_ALLOW_ROOT=1.",
            file=sys.stderr,
        )
        return 2
    user = effective_user()
    who = f" on behalf of {user.name} (uid {user.uid})" if user.via_sudo else ""
    print(
        f"[agent6] WARNING: running as root{who}. The LLM's commands execute as"
        " root inside the jail; files agent6 writes under the repo are chowned"
        " back to you when invoked via sudo. Proceed with care.",
        file=sys.stderr,
    )
    return None


# The ANSI SGR for each `viewmodel.format.status_level`, tty only: a listing
# where a provider_error death reads as plain text is how dead runs went
# unnoticed. The TUI's Rich map and the web's pill classes are the siblings.
_LEVEL_SGR: dict[str, str] = {
    "ok": "32",
    "info": "35",  # magenta (mauve on the TUI/web)
    "active": "1;36",
    "warn": "33",
    "error": "1;31",
    "neutral": "",
}


def styled_status(
    status: str, reason: str, *, color: bool, label: str | None = None
) -> tuple[str, str]:
    """(possibly-colored label, plain label) for a listing row -- the plain form
    drives width math. *label* overrides the text (the sessions listing's
    mode-folded cell); the colour always follows the status word."""
    from agent6.viewmodel.format import status_label, status_level  # noqa: PLC0415

    text = status_label(status, reason) if label is None else label
    sgr_code = _LEVEL_SGR[status_level(status)]
    if color and sgr_code:
        return f"\x1b[{sgr_code}m{text}\x1b[0m", text
    return text, text


def plural(n: int, singular: str, plural: str | None = None) -> str:
    """'1 transition' / '3 transitions': no '1 branches' in user-facing counts."""
    word = singular if n == 1 else (plural or singular + "s")
    return f"{n} {word}"


def home_contracted(path: str) -> str:
    """*path* with `$HOME` shortened to `~`, only at a path boundary (a
    sibling directory whose name merely starts with $HOME's stays whole)."""
    home = str(Path.home())
    return "~" + path[len(home) :] if path == home or path.startswith(home + "/") else path
