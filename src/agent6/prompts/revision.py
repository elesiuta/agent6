# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Auxiliary agent-loop prompts.

The prompt-revision pass, the context summariser, the per-file gist
distiller, and the post-compaction restart notice. Pure text; the loop owns
running each call.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

PROMPT_REVISION_SYSTEM_PROMPT = """\
You revise raw coding-agent tasks before the main worker loop starts.

Goal: transform a terse, vague, or under-specified task into a clear task
specification the worker can act on immediately. Preserve every explicit
constraint from the raw task. Do not invent requirements. Use repo context only
to name likely files, conventions, verification commands, and success criteria.

If the raw task is already crisp, still restate it compactly rather than adding
new scope. If important ambiguity remains, list at most 3 clarifying questions;
the downstream worker may have to proceed under conservative assumptions, so the
revised task must remain actionable without answers.

Output exactly this shape, with no preamble:
<revised_task>
...plain text revised task...
</revised_task>
<clarifying_questions>
- question, or "none"
</clarifying_questions>
"""


CONTEXT_SUMMARY_SYSTEM_PROMPT = (
    "You are compacting a long autonomous-coding-agent transcript so the agent"
    " can keep working with a smaller context window. Produce a dense, factual"
    " progress summary that lets the agent resume WITHOUT re-reading the"
    " elided history. Cover, in order:\n"
    "1. The goal, in one line.\n"
    "2. What has been tried and the outcome of each attempt — which edits were"
    " kept, which were reverted, and which directions turned out to be dead"
    " ends (so the agent does not repeat them).\n"
    "3. The current state: files changed so far, the best result/score"
    " achieved, and the latest verified commit sha.\n"
    "4. The concrete next steps the agent intended to take.\n"
    "Be specific about file paths, function names, numbers, and commit shas."
    " Do not include pleasantries or meta-commentary. Output only the summary."
)

GIST_DISTILL_SYSTEM_PROMPT = (
    "You are compacting an autonomous coding agent's context. Each file below"
    " is about to be dropped from that context. For EACH file output exactly"
    " one line:\n"
    "the file's path, a colon, then the facts the agent needs to keep working"
    " correctly without re-reading the file: exact requirements, constraints,"
    " thresholds, edge cases, interfaces, and numbers, in the file's own"
    " terms.\n"
    "One line per file, every file, in the order given, at most 350 characters"
    " per line. No commentary, no markdown, no blank lines."
)


# Prepended to the post-compaction restart message so the worker knows the
# history was summarised rather than lost, and continues rather than restarting.
_CONTEXT_RESTART_HEAD = (
    "[harness context restart] The earlier conversation was compacted to free"
    " up context. Everything you had done up to this point is captured in the"
    " progress summary below — trust it for prior results and continue the task"
    " from here. Do NOT start over."
)
_CONTEXT_RESTART_DAG = (
    "Your task DAG is durable curator-owned state and was NOT compacted: call"
    " `list_tasks` to recover the full task breakdown, each task's status,"
    " and the current cursor, then resume from the first unfinished task."
    " Treat the DAG as the authoritative record of what is done vs. pending —"
    " the summary below is only a narrative supplement."
)


def pinned_block(pins: Sequence[str]) -> str:
    """The operator's `/pin` instructions as a numbered verbatim block, or ""
    when none. Rendered into every tier-2 restart so pinned instructions are
    never squeezed through the summariser."""
    if not pins:
        return ""
    lines = "\n".join(f"{i}. {pin}" for i, pin in enumerate(pins, start=1))
    return f"PINNED operator instructions (verbatim):\n{lines}"


# Appended to the summariser's request when pins exist: the restart re-shows
# them verbatim, so a summary that restates them would double-spend the chars.
PINS_NO_RESTATE_CLAUSE = (
    "\n\nThe operator PINNED these instructions; they are re-shown verbatim"
    " after the restart — do NOT restate them in the summary:\n"
)


def progress_summary_from_notice(text: str) -> str:
    """The progress summary a restart notice carries, or "" if *text* is not one.

    Parser beside the builder so the two cannot drift. The caller carries the
    prior restart's summary into the NEXT summariser request out-of-band: the
    notice sits at the head of the post-restart history and the summariser's
    transcript is tail-clipped, so a second summary reading only the history
    would begin at the first restart while claiming to cover everything.
    """
    if not text.startswith(_CONTEXT_RESTART_HEAD[:40]):
        return ""
    _, sep, tail = text.partition("PROGRESS SUMMARY:\n")
    return tail.strip() if sep else ""


def context_restart_notice(
    mode: Literal["run", "plan", "ask", "machine", "agent"],
    pins: Sequence[str] = (),
    decisions: str = "",
) -> str:
    """The post-compaction restart preamble. The DAG-recovery paragraph is
    included only for modes whose tool surface has the DAG tools (run, plan):
    in ask/machine/agent `list_tasks` does not exist, so instructing the worker
    to call it burns a turn on an unknown-tool error. Operator pins render
    between the preamble and the summary label, as standing orders."""
    parts = [_CONTEXT_RESTART_HEAD]
    if mode in ("run", "plan"):
        parts.append(_CONTEXT_RESTART_DAG)
    if block := pinned_block(pins):
        parts.append(block)
    if decisions.strip():
        parts.append(f"OPERATOR RULINGS (recorded, still binding):\n{decisions.strip()}\n")
    parts.append("PROGRESS SUMMARY:\n")
    return "\n\n".join(parts)
