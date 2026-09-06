# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Parser builder for `sessions` and its subcommands: list this repo's sessions,
or inspect one (show/diff/merge/compare/commits/stop/prune/transcript/graph)."""

from __future__ import annotations

import argparse

from agent6.ui.cli._common import SESSION_ID, _add_session_id, _sub
from agent6.ui.cli.completers import _complete_live_session_ids, _complete_session_ids


def _add_sessions_parser(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    sessions_p = _sub(
        sub,
        "sessions",
        help=(
            "List this repo's sessions (`agent6 sessions`, or `sessions list`) or"
            " inspect one: show (liveness/progress), diff, compare, transcript, graph."
            " The session id is a positional everywhere (exact or unambiguous prefix;"
            " omit for the most recent). To follow one live, use `agent6 attach`."
        ),
    )
    # A bare `sessions` is `sessions list` (parser._DEFAULT_VERBS).
    sessions_sub = sessions_p.add_subparsers(
        dest="sessions_command", required=True, metavar="<subcommand>"
    )

    sessions_list = _sub(
        sessions_sub,
        "list",
        help="List sessions newest-first by update time: updated, status, cost, id, task.",
    )
    sessions_list.add_argument(
        "--json",
        dest="list_json",
        action="store_true",
        help="Emit the rows as a JSON array (session_id, mode, status, label, level,"
        " reason, unmerged, verify_ok, cost_usd, usd_partial, mtime, winner, task).",
    )

    sessions_show = _sub(
        sessions_sub,
        "show",
        help="One-shot liveness + progress of a session, then exit (`agent6 attach` follows live).",
    )
    _add_session_id(sessions_show, _complete_session_ids)
    sessions_show.add_argument(
        "--json",
        action="store_true",
        help="Emit the status as a single JSON object (for scripts/monitoring).",
    )

    sessions_diff = _sub(
        sessions_sub,
        "diff",
        help="Print the git diff produced by a session (manifest.base_sha -> HEAD of run branch).",
    )
    _add_session_id(sessions_diff, _complete_session_ids)
    sessions_diff.add_argument(
        "--stat",
        action="store_true",
        help="Show --stat summary instead of the full patch.",
    )
    sessions_diff.add_argument(
        "--path",
        dest="paths",
        action="append",
        default=[],
        metavar="PATH",
        help="Restrict the diff to PATH (repeatable).",
    )

    sessions_merge = _sub(
        sessions_sub,
        "merge",
        help="Merge a session's branch into a target (default: the branch it was cut from).",
    )
    _add_session_id(sessions_merge, _complete_session_ids)
    sessions_merge.add_argument(
        "--strategy",
        choices=("squash", "merge", "ff"),
        default=None,
        help="Override git.merge_strategy for this merge.",
    )
    sessions_merge.add_argument(
        "--into",
        default=None,
        metavar="BRANCH",
        help="Target branch to merge into (default: the session's base branch).",
    )
    sessions_merge.add_argument(
        "--message",
        "-m",
        default=None,
        help="Commit message for squash or merge (default: a condensed session summary).",
    )

    sessions_compare = _sub(
        sessions_sub,
        "compare",
        help=(
            "Advisory ranked comparison across >=2 sessions (verify+cost, judged by the"
            " reviewer model when configured): the report `--parallel`'s auto-compare"
            " prints. A fan-out id prints that report as recorded. Never merges."
        ),
    )
    sessions_compare.add_argument(
        "--rejudge",
        action="store_true",
        help=(
            "Judge again instead of printing a fan-out's recorded verdict:"
            " a fresh call, which can rank differently than the listings show."
        ),
    )
    sessions_compare_ids = sessions_compare.add_argument(
        "session_ids",
        nargs="+",
        metavar="SESSION_ID",
        help=(
            "2 or more session ids (or unique prefixes) to compare, or one --parallel"
            " fan-out id (compares its lanes)."
        ),
    )
    sessions_compare_ids.completer = _complete_session_ids  # type: ignore[attr-defined]

    sessions_commits = _sub(
        sessions_sub,
        "commits",
        help="List the per-step commits on a session's branch.",
    )
    _add_session_id(sessions_commits, _complete_session_ids)

    sessions_stop = _sub(
        sessions_sub,
        "stop",
        help="Ask a running detached session to stop cleanly after its current step (resumable).",
    )
    _add_session_id(sessions_stop, _complete_live_session_ids)

    sessions_dir = _sub(
        sessions_sub,
        "dir",
        help="Print the directory this repo's session history lives in, or a session's own"
        " directory when given its id (one line, scriptable).",
    )
    _add_session_id(
        sessions_dir,
        _complete_session_ids,
        help_text=f"{SESSION_ID}; omit for the repo's session root.",
    )

    sessions_rm = _sub(
        sessions_sub,
        "rm",
        help=(
            "Delete a session's history from the state dir, its chain ref, and a fork's"
            " worktree; its branch, if any, is left alone."
        ),
    )
    _add_session_id(sessions_rm, _complete_session_ids)
    sessions_rm.add_argument(
        "--asks",
        action="store_true",
        help="Delete this directory's saved asks instead of one session (an ask is keyed by "
        "the directory it ran in; asks elsewhere are untouched).",
    )

    sessions_prune = _sub(
        sessions_sub,
        "prune",
        help=(
            "Delete safely merged agent6/* run branches, their chain refs, and the fan-out"
            " clones and fork worktrees whose work has landed; report the rest."
        ),
    )
    sessions_prune.add_argument(
        "--delete-squashed",
        action="store_true",
        help=(
            "Also force-delete run branches confirmed squash-merged into their base"
            " (git branch -d refuses these; the content is safe in the base commit)."
            " Each deletion prints an undelete command."
        ),
    )

    sessions_tr = _sub(
        sessions_sub,
        "transcript",
        help="Render a session's full LLM conversation (the lossless transcripts) as Markdown.",
    )
    _add_session_id(sessions_tr, _complete_session_ids)
    sessions_tr.add_argument(
        "--json",
        dest="as_json",
        action="store_true",
        help="Emit the raw transcript array (the per-call request/response objects) instead.",
    )
    sessions_tr.add_argument(
        "--no-thinking", action="store_true", help="Omit the model's reasoning/thinking blocks."
    )
    sessions_tr.add_argument(
        "--tools",
        choices=("both", "calls", "none"),
        default="both",
        help="Show tool calls + results (both), calls only, or neither.",
    )
    sessions_tr.add_argument(
        "--seq",
        default="",
        help="Restrict to a round-trip seq window, e.g. 3 or 3-7 (default: all).",
    )

    sessions_graph = _sub(
        sessions_sub,
        "graph",
        help="Render the persisted task graph for a session as a DFS tree.",
    )
    _add_session_id(sessions_graph, _complete_session_ids)
