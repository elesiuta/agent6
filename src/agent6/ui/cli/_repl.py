# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The interactive in-run REPL hook fired after each auto-commit: show diffs,
recent events, MCP tools, (re)init the workspace, or steer the next step.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agent6.ui.cli._console_view import ConsoleView

from agent6.budget import BudgetTracker
from agent6.config.layer import repo_config_path_for, resolved_state_dir
from agent6.init import init_workspace
from agent6.sessions.id import SessionIdError, resolve_session
from agent6.tools.mcp_client import MCPManager
from agent6.types import AutoCommitDirective
from agent6.ui.cli._interact import _pause
from agent6.ui.cli._steer import repl_prompt_sigint
from agent6.ui.cli.plan_watch import (
    event_epoch,
    format_plain_event,
)
from agent6.ui.cli.sessions_cmds import _cmd_diff
from agent6.viewmodel.state import LOG_NOISE_EVENTS, STREAM_DELTA_EVENTS

REPL_HELP = (
    "  /continue  (empty enter) - let the agent take another iteration\n"
    "  /cost                    - print the running token + USD summary\n"
    "  /diff                    - git diff: base_sha -> the run branch's tip\n"
    "                              (read-only; `agent6 sessions diff`, no pager)\n"
    "  /watch                   - print the last 20 audit events from this run\n"
    "                              (snapshot, no streaming deltas; not a live tail)\n"
    "  /mcp                     - list MCP servers + tools currently wired\n"
    "                              into the agent's tool surface\n"
    "  /init                    - run the `agent6 init` setup wizard in the\n"
    "                              current cwd (prompts; never overwrites files)\n"
    "  /undo                    - fork back before the last message (the\n"
    "                              same /undo as steering, the TUI and web):\n"
    "                              this run ends, the fork holds the state\n"
    "                              before it, resume it with the message\n"
    "                              edited. Nothing is rewritten.\n"
    "  /help                    - show this help\n"
    "  /quit                    - stop the agent cleanly after this commit\n"
)


def build_repl_hook(
    root: Path,
    budget: BudgetTracker,
    *,
    session_id: str = "",
    mcp_manager: MCPManager | None = None,
    console_view: ConsoleView | None = None,
) -> Callable[[int, str], AutoCommitDirective]:
    """Build the after_auto_commit hook for `agent6 run -i`.

    Captures the budget tracker (for `/cost`), the repo root (for
    `/diff` and `/init`), the current run id (for `/diff` and
    `/watch`), and the live MCP manager (for `/mcp`) in a closure
    so Workflow stays agnostic of the CLI's extra state. `/undo` is the
    loop's own undo (fork back before the last message), returned as a
    directive.
    """

    def hook(iteration: int, sha: str) -> AutoCommitDirective:
        # The whole prompt session sits inside the console-view pause: the run
        # is waiting on the OPERATOR, and the heartbeat's per-tick line-erase
        # would otherwise wipe the "agent6> " prompt and the typed characters,
        # replacing them with a lying "working…" spinner (same wiring as the
        # approval/question prompts).
        with _pause(console_view):
            return _prompt_loop(iteration, sha)

    def _prompt_loop(iteration: int, sha: str) -> AutoCommitDirective:
        print(
            f"\n[agent6] iter {iteration} committed {sha[:12]}. "
            f"REPL: /continue /cost /diff /watch /mcp /init /undo /help /quit",
            file=sys.stderr,
        )
        while True:
            try:
                with repl_prompt_sigint():
                    raw = input("agent6> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("[agent6] EOF - stopping interactively.", file=sys.stderr)
                return "stop"
            cmd = raw.lower()
            if cmd in {"", "/continue", "/c"}:
                return "continue"
            if cmd in {"/quit", "/q", "/stop", "/exit"}:
                return "stop"
            if cmd in {"/help", "/h", "?"}:
                print(REPL_HELP, file=sys.stderr)
                continue
            if cmd == "/cost":
                print(budget.format_summary(), file=sys.stderr)
                continue
            if cmd == "/diff":
                repl_run_diff(session_id)
                continue
            if cmd == "/watch":
                repl_show_recent_events(root, session_id, n=20)
                continue
            if cmd == "/mcp":
                repl_list_mcp(mcp_manager)
                continue
            if cmd == "/init":
                repl_run_init(root)
                continue
            if cmd == "/undo":
                return "undo"
            print(
                f"[agent6] unknown command {raw!r}; try /help",
                file=sys.stderr,
            )

    return hook


def repl_run_diff(session_id: str) -> None:
    """REPL /diff: print the run's diff (`sessions diff`), no pager: a pager
    would take over the prompt loop's terminal."""
    try:
        _cmd_diff(session_id=session_id, stat=False, paths=(), paginate=False)
    except Exception as exc:
        print(f"[agent6] /diff failed: {exc}", file=sys.stderr)


def repl_show_recent_events(root: Path, session_id: str, *, n: int) -> None:
    """REPL /watch: snapshot the last n events from this run's logs.jsonl.

    Intentionally NOT a live tail - the REPL is between turns of the
    agent loop; a tail would block the next iteration. Operators who
    want continuous tail use `agent6 attach` in another shell.
    """
    if not session_id:
        print("[agent6] /watch: no run id available", file=sys.stderr)
        return
    # Across buckets: the REPL runs inside an ask, whose dir is asks/ -- a
    # runs/-only path never found the session's own log.
    try:
        layout = resolve_session(resolved_state_dir(root), session_id)
    except SessionIdError as exc:
        print(f"[agent6] /watch: {exc}", file=sys.stderr)
        return
    events_path = layout.logs_path
    if not events_path.is_file():
        print(f"[agent6] /watch: no logs.jsonl at {events_path}", file=sys.stderr)
        return
    try:
        lines = events_path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        print(f"[agent6] /watch failed: {exc}", file=sys.stderr)
        return
    session_start_ts: float | None = None
    if lines:
        try:
            obj0 = json.loads(lines[0])
            if isinstance(obj0, dict):
                session_start_ts = event_epoch(obj0.get("ts"))
        except json.JSONDecodeError:
            session_start_ts = None

    # The audit-log lines every other log view shows: streaming deltas and the
    # loop's mirrors would fill the window with fragments of one turn.
    def _audit_line(raw: str) -> bool:
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError:
            return False
        etype = obj.get("type") if isinstance(obj, dict) else None
        return etype not in STREAM_DELTA_EVENTS and etype not in LOG_NOISE_EVENTS

    tail = [raw for raw in lines if _audit_line(raw)][-n:]
    print(f"[agent6] /watch: last {len(tail)} events from {session_id}", file=sys.stderr)
    for raw in tail:
        print(format_plain_event(raw, session_start_ts=session_start_ts))


def repl_list_mcp(mcp_manager: MCPManager | None) -> None:
    """REPL /mcp: print configured MCP servers + their tool surface."""
    if mcp_manager is None:
        print(
            "[agent6] /mcp: no MCP servers configured (set [mcp] in your config)",
            file=sys.stderr,
        )
        return
    descriptors = mcp_manager.descriptors()
    if not descriptors:
        print("[agent6] /mcp: 0 tools (servers started but exposed nothing)", file=sys.stderr)
        return
    by_server: dict[str, list[str]] = {}
    for d in descriptors:
        by_server.setdefault(d.server_name, []).append(d.tool_name)
    print(f"[agent6] /mcp: {len(descriptors)} tools across {len(by_server)} server(s)")
    for server, tools in sorted(by_server.items()):
        print(f"  {server}: {len(tools)} tool(s)")
        for t in sorted(tools):
            print(f"    - {t}")


def repl_run_init(root: Path) -> None:
    """REPL /init: run the setup wizard. Prompts on a TTY (the REPL is
    interactive) and never overwrites existing files; the ecosystem is
    auto-detected (no hard-coded isolation)."""
    try:
        rc = init_workspace(
            root,
            repo_config_target=repo_config_path_for(root),
            interactive=sys.stdin.isatty(),
        )
    except Exception as exc:
        print(f"[agent6] /init failed: {exc}", file=sys.stderr)
        return
    print("[agent6] /init: ok" if rc == 0 else f"[agent6] /init: exit {rc}", file=sys.stderr)
