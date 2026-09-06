# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""One-line renderings of a run's events for the log views: the argument
preview every tool call carries, and the `format_log_line` row."""

from __future__ import annotations

import shlex
from typing import Any

from agent6.viewmodel import events
from agent6.viewmodel.transcript import scrub_terminal_controls


def _edit_kind(edit: dict[str, object]) -> str:
    """The kind an `apply_edit` pair resolves to, as the tool resolves it: an
    omitted discriminator follows the pair's shape, so a bare create rendered
    as "replace" in every log view."""
    kind = str(edit.get("kind") or "")
    if kind:
        return kind
    return "replace" if edit.get("old_string") else "create"


def _render_arg_value(key: str, value: Any) -> str:
    """One arg value, human-shaped: argv as a shell line, ask_user's questions as
    their text, apply_edit's edits as their kinds, everything else as its string
    / repr."""
    if key == "argv" and isinstance(value, (list, tuple)) and value:
        return shlex.join(str(a) for a in value)
    if key == "questions" and isinstance(value, (list, tuple)) and value:
        first = value[0]
        q = first.get("question", "") if isinstance(first, dict) else str(first)
        return str(q) + (f" (+{len(value) - 1})" if len(value) > 1 else "")
    if key == "edits" and isinstance(value, (list, tuple)) and value:
        # apply_edit: the kinds (replace/create), not the raw {old_string, ...}
        # dict repr that flooded the drawer + TUI tool table.
        return ", ".join(_edit_kind(e) if isinstance(e, dict) else str(e) for e in value)
    return value if isinstance(value, str) else repr(value)


def _as_dict(value: Any) -> dict[str, Any]:
    """An untrusted event field as a dict; the raw log renderer must be total."""
    return value if isinstance(value, dict) else {}


def _as_list(value: Any) -> list[Any]:
    """An untrusted event field as a list; the raw log renderer must be total."""
    return list(value) if isinstance(value, (list, tuple)) else []


def render_args(args: dict[str, Any], *, max_value: int = 80) -> str:
    """Render an args dict as `k=v, ...`, truncating each value to *max_value*
    chars. The inline table uses the tight default; the detail modal renders with
    a generous cap so a long arg (a command, a path, a payload) is readable while
    one pathological value still can't bloat the bounded history."""
    pairs: list[str] = []
    for k, v in args.items():
        s = _render_arg_value(k, v)
        if len(s) > max_value:
            s = s[:max_value] + "…"
        pairs.append(f"{k}={s}")
    return ", ".join(pairs)


def format_log_line(event: dict[str, Any]) -> str:  # noqa: PLR0912, PLR0915
    ts = str(event.get("ts", ""))
    etype = str(event.get("type", "?"))
    # Compact one-line representation: timestamp, type, salient field.
    salient = ""
    match etype:
        case "graph.update":
            nodes = event.get("nodes", {})
            salient = f"{len(nodes)} tasks" if isinstance(nodes, dict) else ""
        case "diff.updated":
            salient = f"{len(str(event.get('patch', '')).splitlines())} lines"
        case "loop.auto_commit":
            salient = f"{str(event.get('sha', ''))[:12]} {event.get('subject', '')}".strip()
        case "tool.call":
            salient = f"{event.get('name', '')}({render_args(_as_dict(event.get('args')))})"
        case "tool.result":
            summ = events.readable_summary(event.get("summary", ""))
            salient = f"{event.get('name', '')} ok={event.get('ok')} {summ}"
            # Execution tools carry capped output tails; show a one-line hint of
            # the latest stderr (else stdout) so a command's outcome reads in the
            # log without opening the transcript. The full tail is in the event.
            tail = str(event.get("stderr_tail") or event.get("stdout_tail") or "")
            snippet = " ".join(tail.split())[:100]
            if snippet:
                salient = f"{salient.rstrip()} | {snippet}"
        case "role.call":
            salient = f"{event.get('role', '')}/{event.get('model', '')}"
        case "role.result":
            role = event.get("role", "")
            if event.get("error"):
                # The error is how a dead run gets diagnosed from the log view.
                salient = f"{role} error: {str(event.get('error'))[:160]}"
            else:
                tin = event.get("tokens_in")
                tout = event.get("tokens_out")
                salient = f"{role} in={tin} out={tout}"
        case "loop.provider.retry":
            salient = f"attempt {event.get('attempt')}: {str(event.get('error', ''))[:160]}"
        case "loop.pin.added":
            salient = f"pinned ({event.get('chars')} chars): {str(event.get('text', ''))[:80]}"
        case "loop.pin.refused":
            salient = f"pin refused: over the {event.get('limit')}-char cap"
        case "loop.pin.restored":
            # The pins in force at leg start: --pin on a fresh run, or a
            # resume/fork's snapshot. The event never says which.
            pins = [str(p) for p in _as_list(event.get("pins"))]
            salient = (
                f"{len(pins)} pinned: " + " | ".join(p[:80] for p in pins)
                if pins
                else "no pinned instructions"
            )
        case "loop.compact.dropped":
            calls = _as_list(event.get("calls"))
            named = ", ".join(str(c) for c in calls)
            salient = f"elided {event.get('n')} old tool results"
            if named:
                salient += f": {named[:160]}"
        case "loop.compact.deduped":
            calls = _as_list(event.get("calls"))
            named = ", ".join(str(c) for c in calls)
            salient = f"deduplicated {event.get('n')} identical tool results"
            if named:
                salient += f": {named[:160]}"
        case "loop.compact.thinking_dropped":
            salient = (
                f"dropped thinking from {event.get('turns')} old turns ({event.get('chars')} chars)"
            )
        case "loop.compact.gists":
            parts = []
            if event.get("gisted"):
                paths = ", ".join(str(p) for p in _as_list(event.get("paths")))
                parts.append(f"{event.get('gisted')} distilled ({paths[:120]})")
            if event.get("demoted"):
                dem = ", ".join(str(p) for p in _as_list(event.get("demoted_paths")))
                parts.append(f"{event.get('demoted')} demoted ({dem[:120]})")
            salient = "; ".join(parts)
        case "loop.compact.summarise.done":
            salient = f"restarted on a {event.get('summary_chars')}-char progress summary"
        case "loop.compact.summarise.failed" | "loop.compact.gist.failed":
            # Without the reason a 429'd summariser reads as nothing having
            # happened.
            salient = str(event.get("error", ""))[:160]
        case "loop.compact.requested":
            focus = str(event.get("focus", ""))
            salient = f"focus: {focus[:120]}" if focus else "no focus"
        case "loop.compact.restored":
            salient = f"{event.get('elided')} elided, {event.get('gists')} gists in context"
        case "loop.compact.refused":
            salient = str(event.get("reason", ""))[:160]
        case "mcp.server_unavailable":
            salient = f"{event.get('server')} unavailable: {str(event.get('error', ''))[:120]}"
        case "loop.skills.warning":
            salient = str(event.get("warning", ""))[:160]
        case "loop.resume.start":
            salient = f"iteration={event.get('iteration')} messages={event.get('messages')}"
        case "budget.update":
            usd = event.get("usd_total")
            # Not format_cost: this is the raw log view, but a float-repr tail
            # ($0.015091189999999999) is noise, not truth.
            usd_s = f"${usd:.4f}" if isinstance(usd, (int, float)) else f"${usd}"
            salient = f"in={event.get('input_total')} out={event.get('output_total')} {usd_s}"
        case "session.start":
            salient = str(event.get("user_task", ""))[:80]
        case "verify.end":
            dur = event.get("duration_s")
            dur_s = f"{dur:.1f}s" if isinstance(dur, (int, float)) else f"{dur}s"
            salient = f"exit={event.get('exit_code')} dur={dur_s}"
        case "approval.prompt":
            salient = str(event.get("prompt", ""))[:80]
        case "approval.answer":
            salient = f"id={event.get('id')} approved={event.get('approved')}"
        case "question.prompt":
            qs = _as_list(event.get("questions"))
            first = str(qs[0].get("question", "")) if qs and isinstance(qs[0], dict) else ""
            salient = (f"[{len(qs)}] " if len(qs) > 1 else "") + first[:80]
        case "question.answer":
            ans = _as_list(event.get("answers"))
            salient = f"id={event.get('id')} answers={len(ans)}"
        case "session.end":
            salient = f"{event.get('reason', '')} all_passed={event.get('all_passed')}"
        case _:
            salient = ""
    line = f"{ts[11:23] if len(ts) > 23 else ts}  {etype:<18}"
    # The salient text embeds model-authored fields (args, summaries, output
    # tails): scrub the finished line so no skin's log pane relays an escape.
    return scrub_terminal_controls(f"{line} {salient}") if salient else line
