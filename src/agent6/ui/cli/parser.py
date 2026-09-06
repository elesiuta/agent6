# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Assembles the `agent6` argparse parser (subcommands, flags, completers)."""

from __future__ import annotations

import argparse
from pathlib import Path

from agent6 import __version__
from agent6.paths import cache_dir, data_dir, effective_user, global_config_path, state_base
from agent6.ui.cli._common import _add_config_flag, _sub
from agent6.ui.cli._config_args import _add_config_parser, _add_connect_parser, _add_model_parser
from agent6.ui.cli._machine_args import _add_machine_parser
from agent6.ui.cli._mcp_args import _add_mcp_server_parsers
from agent6.ui.cli._plan_args import _add_ask_parser, _add_plan_parser
from agent6.ui.cli._review_args import _add_check_parser, _add_review_parser, _add_system_parser
from agent6.ui.cli._run_args import _add_fork_parser, _add_resume_parser, _add_run_parser
from agent6.ui.cli._sessions_args import _add_sessions_parser
from agent6.ui.cli._skills_args import _add_skills_parser
from agent6.ui.cli._watch_args import (
    _add_answer_parser,
    _add_attach_parser,
    _add_net_parsers,
    _add_steer_parser,
    _add_tui_parser,
    _add_web_parser,
)
from agent6.ui.cli.completers import _complete_session_ids

# Commands with a default verb: `plan <task>` == `plan run <task>`, `ask <q>`
# == `ask query <q>`, and a bare group whose obvious action is its read-only
# listing (`skills` == `skills list`, `config` == `config show`, like a bare
# `sessions`). _inject_default_verb rewrites argv so a bare task isn't
# mistaken for a subcommand name. The explicit forms (`plan run`, `ask query`,
# `history search`) cover the rare query whose first word is a verb name. A
# test pins each verb set to the parser's real subcommands.
_DEFAULT_VERBS: dict[str, tuple[str, frozenset[str]]] = {
    "plan": ("run", frozenset({"run", "show", "edit"})),
    "ask": ("query", frozenset({"query", "list"})),
    "history": ("search", frozenset({"search"})),
    "skills": ("list", frozenset({"install", "update", "list", "enable", "disable", "remove"})),
    "memory": ("list", frozenset({"add", "list", "show", "rm", "decisions"})),
    "mcp": ("list", frozenset({"connect", "list", "remove", "serve"})),
    "config": (
        "show",
        frozenset(
            {"show", "fill", "path", "presets", "get", "set", "unset", "add", "remove", "fix"}
        ),
    ),
    "prompt": ("show", frozenset({"show"})),
    "sessions": (
        "list",
        frozenset(
            {
                "commits",
                "compare",
                "diff",
                "dir",
                "graph",
                "list",
                "merge",
                "prune",
                "rm",
                "show",
                "stop",
                "transcript",
            }
        ),
    ),
    "machine": (
        "list",
        frozenset(
            {"list", "check", "test", "graph", "run", "status", "poke", "stop", "replay", "create"}
        ),
    ),
}

# The groups whose default verb takes no positional: a bare word after them is
# a mistyped verb, left for argparse to name the choices.
_BARE_DEFAULT_GROUPS: frozenset[str] = frozenset(
    {"skills", "memory", "mcp", "prompt", "machine", "sessions"}
)


# Top-level options that may precede the subcommand. `--config` takes a value;
# the rest are flags. _inject_default_verb skips past these to find the command.
_GLOBAL_VALUE_OPTS = frozenset({"--config"})
_GLOBAL_FLAG_OPTS = frozenset({"--allow-root"})


def _shell_default_help() -> str:
    """The completions `shell` help, naming what detection resolves to RIGHT
    NOW so the default reads as a fact, not a mechanism. Detection walks the
    process tree (a fish inside bash detects fish); unknown keeps generic
    wording."""
    from agent6.ui.cli.completions_cmd import detect_shell  # noqa: PLC0415

    detected = detect_shell()
    if detected in ("bash", "zsh", "fish", "xonsh"):
        return f"Target shell (default: detected {detected})."
    return "Target shell (default: detect the running shell)."


def _command_index(argv: list[str]) -> int | None:
    """Index of the subcommand token, skipping leading global options.

    `["--config", "c.toml", "plan", ...]` -> 2. Returns None if a global help
    or version flag appears first (argparse handles those) or no command is
    found.
    """
    i = 0
    while i < len(argv):
        tok = argv[i]
        if tok in ("-h", "--help", "--version"):
            return None
        if tok in _GLOBAL_VALUE_OPTS:
            i += 2
            continue
        if tok.startswith("--") and "=" in tok and tok.split("=", 1)[0] in _GLOBAL_VALUE_OPTS:
            i += 1
            continue
        if tok in _GLOBAL_FLAG_OPTS:
            i += 1
            continue
        return i
    return None


def _inject_default_verb(argv: list[str]) -> list[str]:
    """Insert the implicit verb for `plan`/`ask` when the next token isn't one.

    `["plan", "fix the bug"]` -> `["plan", "run", "fix the bug"]`;
    `["ask", "why?"]` -> `["ask", "query", "why?"]`. Leading global options
    (`--config FILE`, `--allow-root`) are skipped to find the command. A bare
    `plan`/`ask`, an explicit verb, or `-h`/`--help` is left untouched.
    """
    ci = _command_index(argv)
    if ci is None or argv[ci] not in _DEFAULT_VERBS:
        return argv
    default_verb, verbs = _DEFAULT_VERBS[argv[ci]]
    rest = argv[ci + 1 :]
    # A bare `plan`/`ask` also gets the default verb so the no-task path (offer
    # the most recent plan / start the ask REPL) still runs; only an explicit
    # verb or -h/--help is left alone.
    if rest and (rest[0] in verbs or rest[0] in ("-h", "--help")):
        return argv
    if rest and argv[ci] in _BARE_DEFAULT_GROUPS and not rest[0].startswith("-"):
        return argv
    return [*argv[: ci + 1], default_verb, *rest]


def _directories_epilog() -> str:
    """Where agent6 keeps things, resolved, for the bottom of `--help`.

    Four XDG bases each holding a different kind of thing is correct and
    unguessable; naming them here is the difference between "where did my run
    history go" and reading the docs. Paths only (no file contents), and each
    is a plain env/home lookup, so building the parser stays cheap.
    """
    user = effective_user()
    rows = (
        ("config", global_config_path(user).parent, "config.toml, secrets.toml (0600)"),
        ("state", state_base(user), "per-repo run history, memory, reviews"),
        ("data", data_dir(user), "installed skill packs (skills/)"),
        ("cache", cache_dir(user), "regenerable model lists"),
    )
    width = max(len(str(p)) for _n, p, _w in rows)
    lines = [f"  {name:<6} {path!s:<{width}}  {what}" for name, path, what in rows]
    return "\n".join(
        [
            "directories (XDG; override each with AGENT6_<NAME>_HOME):",
            *lines,
            "",
            "`agent6 config path` adds this repo's own state dir and the config files.",
        ]
    )


def build_parser() -> argparse.ArgumentParser:  # noqa: PLR0915
    parser = argparse.ArgumentParser(
        prog="agent6",
        description="Sandboxed coding agent.",
        epilog=_directories_epilog(),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=f"agent6 {__version__}")
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        metavar="FILE",
        help=(
            "Explicit config file, layered on top of the global config (its path"
            " is printed below) and the per-repo config (out of the workspace,"
            " under the state dir). Default: use only those two layers + built-in"
            " defaults."
        ),
    )
    parser.add_argument(
        "--allow-root",
        action="store_true",
        help=(
            "Permit running as root (also AGENT6_ALLOW_ROOT=1). Off by default:"
            " running an LLM-driven agent as root is dangerous. Under sudo,"
            " agent6 reads your config/secrets and chowns new files back to you."
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True, metavar="<command>")

    _add_run_parser(sub)

    _add_plan_parser(sub)

    _add_ask_parser(sub)

    _add_attach_parser(sub)
    _add_steer_parser(sub)
    _add_answer_parser(sub)
    _add_net_parsers(sub)

    _add_sessions_parser(sub)

    _add_tui_parser(sub)

    completions_p = _sub(
        sub,
        "completions",
        help=(
            "Install shell tab-completion for agent6 (detects the shell you"
            " are running; bash/zsh get a guarded source line in their rc, fish and"
            " xonsh a native auto-loaded file). --print emits the script"
            " instead, for `eval` or manual setup."
        ),
    )
    completions_p.add_argument(
        "shell",
        nargs="?",
        # None (not ""): argparse validates a *string* default against choices,
        # and an empty-string choice leaks into completion output as a bogus
        # description-only candidate.
        default=None,
        choices=["bash", "zsh", "fish", "xonsh"],
        metavar="{bash,zsh,fish,xonsh}",
        help=_shell_default_help(),
    )
    completions_p.add_argument(
        "--print",
        dest="print_only",
        action="store_true",
        help="Print the completion script to stdout instead of installing it.",
    )

    _add_web_parser(sub)

    prompt_p = _sub(
        sub,
        "prompt",
        help="Inspect the assembled system prompt for this repo + config.",
    )
    prompt_sub = prompt_p.add_subparsers(
        dest="prompt_command", required=True, metavar="<subcommand>"
    )
    prompt_show = _sub(
        prompt_sub,
        "show",
        help=(
            "Print everything the model receives on a run's first call here: the"
            " system prompt (static blocks + the per-repo <repo-priors> block), the"
            " tool definitions this config exposes (the API's `tools` field), and"
            " the first user message around the task."
        ),
    )
    prompt_show.add_argument(
        "--mode",
        choices=("run", "plan", "ask", "agent"),
        default="run",
        help="Which mode's exchange to assemble (default: run).",
    )
    prompt_show.add_argument(
        "--json",
        action="store_true",
        help=(
            "One JSON object (mode, system, tools, first_message, mcp_tools_pending)"
            " instead of text."
        ),
    )

    _add_resume_parser(sub)

    _add_fork_parser(sub)

    _add_config_parser(sub)

    _add_check_parser(sub)

    _add_connect_parser(sub)

    _add_system_parser(sub)

    _add_model_parser(sub)

    mem_p = _sub(
        sub,
        "memory",
        help=(
            "Manage the repo's agent memory (one fact per file + index); a bare"
            " `agent6 memory` lists it."
        ),
    )
    mem_sub = mem_p.add_subparsers(dest="memory_command", required=True, metavar="<subcommand>")
    mem_add = _sub(mem_sub, "add", help="Write <name>.md and its index line.")
    mem_add.add_argument("name", help="Memory name (lowercase letters, digits, dashes).")
    mem_add.add_argument("body", help="The fact (in quotes; first line becomes the index hook).")
    _sub(mem_sub, "list", help="Print the MEMORY.md index.")
    mem_show = _sub(mem_sub, "show", help="Print one memory file.")
    mem_show.add_argument("name", help="Memory name.")
    mem_rm = _sub(mem_sub, "rm", help="Delete a memory file and its index line.")
    mem_rm.add_argument("name", help="Memory name.")
    _sub(
        mem_sub,
        "decisions",
        help="Print the operator rulings the harness recorded (memory/DECISIONS.md).",
    )

    _add_skills_parser(sub)

    _sub(
        sub,
        "ps",
        help=(
            "Live agent6 sessions across every repository on this machine"
            " (directory, id, mode, status, pid, front-end; per-repo views: `sessions`)."
        ),
    )

    hist_p = _sub(
        sub,
        "history",
        help=(
            "Cross-session search over persisted transcripts and session data"
            " (per-session views: `sessions`)."
        ),
    )
    hist_sub = hist_p.add_subparsers(dest="history_command", required=True, metavar="<subcommand>")
    hist_search = _sub(hist_sub, "search", help="ripgrep-backed search over all runs.")
    hist_search.add_argument(
        "query",
        nargs="?",
        default="",
        help="Pattern (passed to rg --fixed-strings by default).",
    )
    hist_search.add_argument(
        "--regex", action="store_true", help="Interpret query as a regex instead of fixed string."
    )
    hist_search_session = hist_search.add_argument(
        "--session",
        default="",
        metavar="SESSION_ID",
        help="Restrict to a single session id (default: all sessions).",
    )
    hist_search_session.completer = _complete_session_ids  # type: ignore[attr-defined]

    init_p = _sub(
        sub,
        "init",
        help="Optional setup wizard: per-repo config, verify_command, .gitignore, AGENTS.md.",
    )
    init_p.add_argument(
        "--yes",
        action="store_true",
        help="Skip the interactive prompts and accept the defaults for every step"
        " (nothing existing is ever overwritten).",
    )
    init_p.add_argument(
        # Named --ecosystem: `run/plan/ask --preset` is the strategy preset
        # (quick/ultra/...), a different concept.
        "--ecosystem",
        dest="ecosystem",
        choices=("py", "rust", "node"),
        default="",
        help=(
            "Ecosystem for the .gitignore build-artifact entries. Auto-detected"
            " from the repo's manifests when omitted (py/rust/node)."
        ),
    )

    _add_review_parser(sub)

    mcp_p = _sub(
        sub,
        "mcp",
        help=(
            "MCP (Model Context Protocol): add a server, list them, or serve; a bare"
            " `agent6 mcp` lists them."
        ),
    )
    mcp_sub = mcp_p.add_subparsers(dest="mcp_command", required=True, metavar="<subcommand>")
    _add_mcp_server_parsers(mcp_sub)
    mcp_serve = _sub(
        mcp_sub,
        "serve",
        help=(
            "Run agent6 as an MCP stdio server over the cwd's agent6 config:"
            " query_dag and list_sessions always, run_in_sandbox only where"
            ' sandbox.run_commands = "yes" (the default "ask" withholds it: nothing'
            " here can answer an approval), and run_verify and apply_patch_in_sandbox"
            " where it also sets a verify command. Speaks line-delimited"
            " JSON-RPC on stdin/stdout; configure an MCP-aware client to spawn"
            " this command."
        ),
    )
    _add_config_flag(mcp_serve)

    _sub(
        sub,
        "acp",
        help=(
            "Run agent6 as an ACP (Agent Client Protocol) agent, driven by an"
            " editor. Speaks line-delimited JSON-RPC on stdin/stdout; the"
            " editor spawns this, so nothing else may write to stdout. Config"
            " comes from each session's own directory."
        ),
    )

    _add_machine_parser(sub)

    return parser
