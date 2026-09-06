# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""One renderer for a folded `TranscriptItem`, shared by the CLI stream and the
TUI conversation view.

`item_lines` produces medium-agnostic styled lines: each line is a list of
`(text, style)` spans, where *style* is a SEMANTIC name (not an ANSI code or a
Rich style). Each front-end maps those names to its own output -- the CLI to ANSI,
the TUI to a Rich `Text` -- so the structure and the styling decisions live in
ONE place and the two skins cannot drift.

Relative indent is baked into the line text (a tool's result/tail sit under its
call); each front-end adds its own left margin (the CLI a two-space gutter, the
TUI its CSS padding).
"""

from __future__ import annotations

from typing import Literal

from agent6.viewmodel.transcript import (
    CALL,
    COMMIT,
    DONE,
    OPERATOR,
    RESULT,
    THINK,
    TranscriptItem,
)

StyleName = Literal[
    "thinking",
    "think-marker",
    "text",
    "call",
    "verify",
    "arg",
    "ok",
    "fail",
    "detail",
    "tail",
    "more",
    "commit",
    "marker",
    "done-ok",
    "done-fail",
    "body",
    "done-detail",
    "operator",
]
Span = tuple[str, StyleName]
Line = list[Span]

# One detail level, cycled by a single shortcut in the TUI:
#   hidden    -- thinking AND tool items omitted entirely: just the dialogue
#                (prose, operator, commits, verdict)
#   collapsed -- thinking as a one-line marker, tool detail clipped (the default)
#   expanded  -- thinking and tool detail both in full
DetailLevel = Literal["hidden", "collapsed", "expanded"]

TAIL_CLIP = 120  # chars of a failed tool's captured output tail shown inline
DETAIL_CLIP = 120  # chars of a tool result's first line shown inline (the rest -> "+N more")


def _tool_lines(item: TranscriptItem, *, expanded: bool) -> list[Line]:
    """A tool call's lines: the call head, then the result; an in-flight call
    is its head alone, marked running. The RESULT glyph carries the pass/fail
    colour; the detail is its OWN neutral span, never tinted by the fail
    colour. Collapsed, a long multi-line detail (a failed tool's error dump)
    is clipped to its first line + a "+N more lines" note so it can't
    dominate; expanded, the full detail is shown, indented and still neutral."""
    head_style: StyleName = "verify" if item.name == "run_verify_command" else "call"
    head: Line = [(f"{CALL} {item.name}", head_style)]
    if item.arg:
        head.append((f"  {item.arg}", "arg"))
    if item.ok is None:  # in flight: the head alone, marked; its result replaces it
        head.append((f"  · {item.detail or 'running'}", "more"))
        return [head]
    glyph: StyleName = "ok" if item.ok else "fail"
    detail_lines = item.detail.split("\n")
    long = len(detail_lines) > 1 or len(detail_lines[0]) > DETAIL_CLIP
    if expanded and long:
        lines: list[Line] = [head, [(f"  {RESULT} ", glyph), (detail_lines[0], "detail")]]
        lines.extend([(f"      {ln}", "detail")] for ln in detail_lines[1:])
    else:
        reason = detail_lines[0]
        if len(reason) > DETAIL_CLIP:
            reason = reason[: DETAIL_CLIP - 1] + "…"
        result: Line = [(f"  {RESULT} ", glyph), (reason, "detail")]
        extra = len(detail_lines) - 1
        if extra:
            result.append((f"  (+{extra} more line{'' if extra == 1 else 's'})", "more"))
        lines = [head, result]
    if item.tail:
        if expanded:
            # Full captured output, line-structured -- without this, expanded ==
            # collapsed for tool items and the web/TUI detail toggle is a no-op
            # exactly where the user wants more (a run_command's output).
            lines.extend([(f"    {ln}", "tail")] for ln in item.tail.split("\n"))
        else:
            flat = " ".join(item.tail.split())
            clip = flat[:TAIL_CLIP] + ("…" if len(flat) > TAIL_CLIP else "")
            lines.append([(f"    {clip}", "tail")])
    return lines


def _thinking_lines(item: TranscriptItem, *, expanded: bool) -> list[Line]:
    """A reasoning block's lines. Expanded: one entry per body line (the
    contract every consumer's line arithmetic relies on), continuation lines
    aligned under the glyph. Collapsed: one line OF the reasoning as the
    summary (first non-empty line, clipped) with a more-count, so collapsed
    still says what the model is thinking about."""
    if expanded:
        body_lines = item.body.split("\n")
        out: list[Line] = [[(f"{THINK} ", "think-marker"), (body_lines[0], "thinking")]]
        out.extend([(f"  {ln}", "thinking")] for ln in body_lines[1:])
        return out
    n = item.body.count("\n") + 1
    first = next((ln.strip() for ln in item.body.split("\n") if ln.strip()), "")
    if len(first) > DETAIL_CLIP:
        first = first[: DETAIL_CLIP - 1] + "…"
    line: Line = [(f"{THINK} ", "think-marker"), (first, "thinking")]
    if n > 1:
        line.append((f"  (+{n - 1} more line{'' if n == 2 else 's'})", "more"))
    return [line]


def item_lines(item: TranscriptItem, *, detail: DetailLevel) -> list[Line]:
    """The styled lines for one folded conversation item (both skins render these).
    `detail` is the one detail level cycled in the TUI (see DetailLevel)."""
    if detail == "hidden" and item.kind in ("thinking", "tool"):
        # The least-noise level reads as pure dialogue; the items survive in
        # the fold, so cycling back restores them.
        return []
    lines: list[Line] = []
    if item.kind == "thinking":
        lines.extend(_thinking_lines(item, expanded=detail == "expanded"))
    elif item.kind == "text":
        lines.extend([(ln, "text")] for ln in item.body.split("\n"))
    elif item.kind == "tool":
        lines.extend(_tool_lines(item, expanded=detail == "expanded"))
    elif item.kind == "operator":
        # The operator's own words (steer / resume follow-up): always shown in
        # full at every detail level -- it is the other half of the dialogue.
        body_lines = item.body.split("\n")
        lines.append([(f"{OPERATOR} ", "operator"), (body_lines[0], "operator")])
        lines.extend([(f"  {ln}", "operator")] for ln in body_lines[1:])
    elif item.kind == "commit":
        lines.append([(f"{COMMIT} commit  {item.detail}", "commit")])
    elif item.kind == "marker":
        # First line is the divider headline; any further lines (a parallel
        # dispatch's tasks, a join's per-lane rows) sit under it, indented.
        body_lines = item.body.split("\n")
        lines.append([(f"── {body_lines[0]} ──", "marker")])
        lines.extend([(f"   {ln}", "marker")] for ln in body_lines[1:])
    elif item.kind == "done":
        badge: Line = (
            [(f"{DONE} done", "done-ok")]
            if item.ok
            else [(f"{DONE} {item.name or 'stopped'}", "done-fail")]
        )
        body_lines = item.body.split("\n") if item.body else []
        if body_lines:
            badge.append((f"  {body_lines[0]}", "body"))
        lines.append([])  # a blank line sets the verdict apart
        lines.append(badge)
        lines.extend([(f"  {ln}", "body")] for ln in body_lines[1:])
        lines.append([(item.detail, "done-detail")])
    return lines
