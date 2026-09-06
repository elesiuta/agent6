# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Context-window management for the agent loop.

Two tiers keep a long run inside the model's context window:

- tier 1 (`compact_old_tool_results`): at `DROP_BLOCKS_AT_CHARS` the oldest
  tool_result blocks are replaced by `ELISION_PLACEHOLDER`; large read_file
  results decay through a distilled-gist placeholder first when the caller
  provides a `gister` (see below).
- tier 2 (`context_chars` vs `SUMMARISE_AT_CHARS`): the elided history is
  summarised and the conversation restarts from (task + summary).

`cap_tool_result` separately bounds a single tool_result so one huge payload
cannot blow the budget on the turn it arrives. Everything here is a pure
function of the conversation; the loop owns the policy of when to call them
and supplies the one impure seam (the `gister` callable that distills
about-to-be-elided file reads with the summariser model).
"""

from __future__ import annotations

import json
import re
import shlex
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Final

from agent6.providers import CLAUDE_CODE_PERSIST_BYTES
from agent6.providers.types import ToolDefinition
from agent6.tools.schema import AskUserInput
from agent6.workflows._conversation import (
    AssistantTurn,
    Conversation,
    ToolResultItem,
    Turn,
    UserTurn,
)
from agent6.workflows._panel import REVIEW_NOTICE_BYTES
from agent6.workflows._verify_gate import VERIFY_TAIL_CHARS

# Stable prefix shared by every placeholder variant: idempotency checks and
# tests key on it.
ASK_USER_TOOL = AskUserInput.TOOL_NAME
ELISION_PREFIX = "<elided by context compaction"

ELISION_PLACEHOLDER = (
    "<elided by context compaction: this tool_result has been replaced "
    "with this short marker to keep the loop's cumulative input bounded. "
    "If you still need it, re-read only the part you need with a targeted "
    "read_file start_line/limit; do not re-issue the identical call.>"
)

# Gist placeholders share ELISION_PREFIX (idempotency walks key on it) but are
# distinguishable so continued pressure can demote them to the bare marker.
ELISION_GIST_PREFIX = ELISION_PREFIX + " (distilled)"

# How much of a tool arg the placeholder echoes. Placeholders stay in context,
# so the identity hint must stay short.
_ELISION_HINT_MAX_CHARS = 120


# The argument that identifies a call, tried in order. Named tools are not
# enumerated here: anything carrying one of these gets a label, so a tool added
# later is never silently anonymous in a compacted transcript. `run_command` is
# why this matters most -- searching moved there, so a placeholder reading just
# "run_command" leaves the model unable to tell whether it already ran the suite.
_IDENTIFYING_KEYS: Final = ("path", "argv", "symbol", "name", "id", "url")


def call_label(tool_name: str, tool_input: Any) -> str:
    """Short identity for a tool call ("read_file src/foo.py").

    The placeholder hint, shared with the `loop.compact.*` event payloads so
    every surface can say WHAT left the model's context, not just how much.
    """
    if not tool_name or not isinstance(tool_input, dict):
        return tool_name
    hint = ""
    if tool_name == "read_file":
        hint = str(tool_input.get("path", ""))
        start_line = tool_input.get("start_line")
        limit = tool_input.get("limit")
        if start_line or limit:
            hint += f" (start_line={start_line}, limit={limit})"
    else:
        for key in _IDENTIFYING_KEYS:
            value = tool_input.get(key)
            if not value:  # absent, or present-but-empty: no identity to show
                continue
            hint = (
                shlex.join(str(a) for a in value) if isinstance(value, list | tuple) else str(value)
            )
            break
    if len(hint) > _ELISION_HINT_MAX_CHARS:
        hint = hint[:_ELISION_HINT_MAX_CHARS] + "..."
    return f"{tool_name} {hint}".rstrip()


def elision_placeholder(tool_name: str, tool_input: Any) -> str:
    """Identity-bearing tier-1 placeholder.

    Names the elided call (tool + its key argument) so the model can re-issue
    or skip it without scanning up for the paired tool_use block; a bare
    marker made weak models lose track of WHAT was elided and re-read the
    wrong files. Unknown tool (orphan result) falls back to the generic
    marker.
    """
    if not tool_name or not isinstance(tool_input, dict):
        return ELISION_PLACEHOLDER
    described = call_label(tool_name, tool_input)
    # Only read_file takes a range, so only it can be told to re-read one;
    # start_line/limit named for a run_command result leaves the model no
    # legal move next to "do not re-issue the identical call".
    retry = (
        "re-read only the part you need (read_file with a targeted start_line/limit)"
        if tool_name == "read_file"
        else "re-run it with a narrower scope"
    )
    return (
        f"{ELISION_PREFIX}: the result of {described} was replaced with this "
        f"short marker to keep the loop's cumulative input bounded. If you "
        f"still need it, {retry}; do not re-issue the identical call.>"
    )


# Distilled-gist elision. Measured (bench/longhorizon FINDINGS #1): under a
# small-window regime tier-1 elision of reference docs halves a retention
# task's score (0.921 -> 0.425) and every redundant read is post-drop, while
# code files are cheaply re-readable. So a large read_file result about to be
# elided decays in two stages: first to a placeholder carrying a model-written
# gist of the file, then (under continued pressure) to
# the bare identity marker, so the hard byte bound always holds. The caps
# bound the distiller call per drop event; hot files (protect_paths) are never
# gisted because their content is changing under edits and a stale gist would
# mislead.
GIST_MIN_SOURCE_CHARS = 2_000  # below this the content is nearly gist-sized
GIST_MAX_CHARS = 400  # per gist, clipped
GIST_FILE_SLICE_CHARS = 8_000  # per-file head sent to the distiller
GIST_INPUT_CAP_CHARS = 24_000  # total distiller input per drop event
GIST_MAX_FILES_PER_CALL = 12


@dataclass(frozen=True, slots=True)
class GistRequest:
    """One file whose about-to-be-elided read_file content should be distilled."""

    path: str
    content: str


# The impure seam: called once per drop event with the batch of eligible
# reads; returns path -> distilled gist (missing paths fall back to the bare
# placeholder). The loop binds this to the summariser model; on provider
# failure it returns {}.
Gister = Callable[[tuple[GistRequest, ...]], Mapping[str, str]]


@dataclass(frozen=True, slots=True)
class CompactionStats:
    """One tier-1 pass: the identities (`call_label` strings, read paths) of
    the identical results deduplicated, the old tool_results elided, the
    gists among them, and the gist placeholders demoted to the bare marker.
    A count is the length of its tuple."""

    elided_calls: tuple[str, ...] = ()
    gist_paths: tuple[str, ...] = ()
    demoted_paths: tuple[str, ...] = ()
    deduped_calls: tuple[str, ...] = ()


def elision_gist_placeholder(described: str, gist: str) -> str:
    """Tier-1 placeholder that keeps a distilled gist of the elided read.

    Takes the caller's `call_label` rather than rebuilding one, so the gist
    and bare markers carry the SAME identity (the conversation differ dedupes a
    gist->bare demotion on it). Rebuilt from the path alone it would drop a
    ranged read's start_line/limit, and every demotion would re-report as a
    fresh elision.
    """
    return (
        f"{ELISION_GIST_PREFIX}: the result of {described} was replaced "
        f"by this distilled gist; if the gist is not enough, re-read only "
        f"the part you need (read_file with a targeted start_line/limit).\ngist: {gist}>"
    )


def read_file_text_from_result(raw: str) -> str:
    """The file text inside a serialized read_file tool_result.

    Unwraps the {"content": ...} result shape (and the truncation envelope's
    "head") so the distiller sees file text, not JSON escapes. An error result
    returns "" (nothing worth distilling); any other shape falls back to the
    raw payload.
    """
    try:
        data = json.loads(raw)
    except (ValueError, TypeError):
        return raw
    if isinstance(data, dict):
        content = data.get("content")
        if isinstance(content, str):
            return content
        head = data.get("head")
        if data.get("_tool_result_truncated") and isinstance(head, str):
            return head
        if "error" in data:
            return ""
    return raw


def parse_gist_lines(text: str, paths: Sequence[str]) -> dict[str, str]:
    """path -> gist from the distiller's one-line-per-file reply.

    Tolerant of list markers and backticks around the path; a file the reply
    misses simply keeps the bare placeholder, and unknown paths are ignored.
    """
    wanted = set(paths)
    out: dict[str, str] = {}
    for line in text.splitlines():
        stripped = line.strip().lstrip("-* ").strip()
        head, sep, rest = stripped.partition(":")
        if not sep:
            continue
        head = head.strip().strip("`")
        rest = rest.strip()
        if head in wanted and rest:
            out[head] = rest
    return out


# Per-tool-result cap: 60_000 bytes of UTF-8 (~15k tokens) fits most source
# files whole. Anything over it is wrapped by cap_tool_result in a well-formed
# JSON truncation notice: a raw mid-JSON slice reads as a malformed result,
# and a weak model re-calls `read_file` to "see the rest" until the loop-guard
# latches. Bytes, because the bound that displaces this default for a Claude
# Code worker (its 50,000-byte persistence threshold) is measured in bytes.
TOOL_RESULT_CAP_BYTES = 60_000

# A Claude Code worker's turn carries the capped result and the turn's
# trailing notices in one tool_result, and the whole stays under the
# provider's persist threshold with the notices at their largest: a verify
# tail of VERIFY_TAIL_CHARS four-byte characters, a review critique of
# REVIEW_NOTICE_BYTES, and the nudges and framing around them.
CLAUDE_CODE_NOTICE_ROOM_BYTES = 4 * VERIFY_TAIL_CHARS + REVIEW_NOTICE_BYTES + 4_000
CLAUDE_CODE_RESULT_CAP_BYTES = CLAUDE_CODE_PERSIST_BYTES - CLAUDE_CODE_NOTICE_ROOM_BYTES

# compaction thresholds (chars, not tokens - approximate; tokens
# are roughly chars/4 for English-shaped content).
DROP_BLOCKS_AT_CHARS = 256_000  # ~64k tokens of tool_result content
SUMMARISE_AT_CHARS = 768_000  # ~192k tokens: full context restart


def cap_tool_result(content: str, *, tool_name: str, cap: int = TOOL_RESULT_CAP_BYTES) -> str:
    """Cap a serialized tool_result payload at *cap* bytes of UTF-8 (the loop's
    `TOOL_RESULT_CAP_BYTES`, or a provider's tighter bound) without producing
    malformed JSON. If the payload is over the
    cap, wrap it in a new JSON envelope that tells the model:
    (a) the result was truncated, (b) how many chars were shown vs
    total, (c) the head of the original content, (d) actionable next
    steps. This prevents weak models from inferring "the tool itself
    returned a partial result, let me call it again"."""
    if len(content.encode()) <= cap:
        return content
    if tool_name == "read_file":
        guidance = (
            "Use `read_file` again with `start_line` and `limit` to read the rest"
            " of the file in chunks. Do NOT re-call with identical arguments"
            " expecting a different result - you will get the same truncated"
            " head and waste budget."
        )
    elif tool_name in ("run_command", "run_verify_command"):
        guidance = (
            "Re-run with a narrower scope (e.g. a single test, a narrower"
            " search, head/tail) to get a result that fits. Do NOT re-call"
            " with identical arguments expecting different output."
        )
    else:
        guidance = (
            "Re-call with arguments that produce less output. Do NOT re-call"
            " with identical arguments expecting different output."
        )

    def envelope(head_len: int) -> str:
        head = content[:head_len]
        return json.dumps(
            {
                "_tool_result_truncated": True,
                "tool": tool_name,
                "shown_chars": len(head),
                "total_chars": len(content),
                "head": head,
                "guidance": guidance,
            },
            ensure_ascii=False,
        )

    # Size the head by ENCODED length: json.dumps re-escapes quotes/backslashes
    # and a wide character is several bytes, so a raw-char budget overshoots
    # the cap on escape-heavy or CJK content (observed 118k emitted against
    # the 60k cap). Encoded length is monotone in head length and the empty
    # head always fits, so bisect for the largest head whose envelope fits
    # (~16 dumps passes).
    lo, hi = 0, min(len(content), cap)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if len(envelope(mid).encode()) <= cap:
            lo = mid
        else:
            hi = mid - 1
    return envelope(lo)


_CHECKOFF_FENCE_RE = re.compile(r"```checkoff\s*\n(.*?)\n```", re.DOTALL)


def parse_checkoff(text: str) -> tuple[list[str], list[str]]:
    """Extract a tier-2 compaction check-off from the summariser's output.

    The summariser is asked to append a fenced ```checkoff block holding
    `{"completed_ids": [...], "new_tasks": [...]}` so agent6 can mark finished
    tasks done and queue newly-discovered ones in the curator-owned DAG (the
    model rarely calls update_task itself). Returns
    `(completed_ids, new_task_titles)`. Best-effort and total: a missing or
    malformed block yields `([], [])` so a bad summary never breaks the run.
    """
    m = _CHECKOFF_FENCE_RE.search(text)
    if m is None:
        return [], []
    try:
        data = json.loads(m.group(1))
    except (ValueError, TypeError):
        return [], []
    if not isinstance(data, dict):
        return [], []
    return _nonempty_strs(data.get("completed_ids")), _nonempty_strs(data.get("new_tasks"))


def _nonempty_strs(value: object) -> list[str]:
    """The stripped, non-empty strings in a JSON *value*, or [] if it is not a
    list. Keeps parse_checkoff total: a present-but-non-list field (`null`
    when nothing completed, a number, a bool -- all natural summariser output)
    yields [] rather than raising when iterated."""
    if not isinstance(value, list):
        return []
    return [s.strip() for s in value if isinstance(s, str) and s.strip()]


def strip_checkoff(text: str) -> str:
    """Remove the ```checkoff block from a summary before it re-enters context;
    it is agent6 bookkeeping, not narrative the restarted worker should re-read."""
    return _CHECKOFF_FENCE_RE.sub("", text).strip()


def context_chars(conversation: Conversation) -> int:
    """Approximate the full character size of the conversation context.

    Sums notice text, tool_result content, and -- for assistant turns -- every
    value of every raw block, because `Conversation.to_wire` sends those
    blocks VERBATIM: whatever is in them is in each later request. Used as the
    tier-2 (summarise-and-restart) trigger, which must measure something tier-1
    elision does not already cap, against ~80% of the model's real context
    window.

    Whole blocks rather than a list of known keys: counting only text/content/
    tool_use-input scored a reasoning model's `{"type": "thinking", ...}` as
    ZERO, so tier-2 waited on a number that omitted the largest thing in the
    context. A block type nobody has met yet must not be free either.
    """
    return sum(turn_chars(turn) for turn in conversation.turns)


def turn_chars(turn: Turn) -> int:
    """One turn's contribution to :func:`context_chars`."""
    if isinstance(turn, AssistantTurn):
        total = 0
        for item in turn.raw_content:
            if not isinstance(item, dict):
                total += len(str(item))
                continue
            # "type" is the discriminator, not payload; everything else is.
            total += sum(
                len(v if isinstance(v, str) else str(v))
                for k, v in item.items()
                if k != "type" and v is not None
            )
        return total
    return sum(
        len(item.content if isinstance(item, ToolResultItem) else item.text) for item in turn.items
    )


# Verbatim recent-history tail kept through a tier-2 restart, sized to pi's
# keepRecentTokens default (20k tokens ~= 80k chars). `[context]
# keep_recent_chars` overrides.
KEEP_RECENT_CHARS = 80_000


def strip_old_thinking(conversation: Conversation, *, keep_turns: int) -> tuple[int, int]:
    """Drop thinking blocks from assistant turns older than the newest
    *keep_turns* assistant turns (Claude Code clears old thinking the same
    way). The newest stay: Anthropic requires the signed thinking block of a
    tool_use still being answered, so callers pass `keep_turns >= 1`.
    Returns (turns stripped, chars removed)."""
    assistant_idxs = [
        i for i, turn in enumerate(conversation.turns) if isinstance(turn, AssistantTurn)
    ]
    n = chars = 0
    for idx in assistant_idxs[:-keep_turns]:
        removed = conversation.strip_thinking(idx)
        if removed:
            n += 1
            chars += removed
    return n, chars


def request_prefix_chars(system: str, tools: Sequence[ToolDefinition]) -> int:
    """The chars every request carries besides the conversation: the system
    prompt and the tool definitions.

    The model's window bounds the WHOLE request, so a threshold measured on the
    conversation alone leaves a band, exactly the size of this prefix, where
    the loop sees room and the provider answers 400 (prompt too long), and a
    resumed leg re-issues the same over-window request.
    The system prompt is the unbounded half: AGENTS.md rides in it whole."""
    return len(system) + sum(
        len(t.name) + len(t.description) + len(json.dumps(t.input_schema, separators=(",", ":")))
        for t in tools
    )


def recent_tail_start(turns: Sequence[Turn], cap_chars: int) -> int:
    """The index where a tier-2 restart's verbatim tail begins: the largest
    tail of whole turns within *cap_chars* that starts on a wire-safe
    boundary. Returns `len(turns)` when nothing is kept (cap 0, or no safe
    boundary fits).

    A safe start is any turn except a user turn carrying tool_results: that
    turn answers the assistant turn BEFORE it, which the restart summarised
    away, and an unanswered pairing is a provider refusal. Turn 0 (the task)
    is never part of the tail; the restart always keeps it separately.
    """
    if cap_chars <= 0:
        return len(turns)
    total = 0
    start = len(turns)
    i = len(turns) - 1
    while i >= 1:
        size = turn_chars(turns[i])
        if total + size > cap_chars:
            break
        total += size
        start = i
        i -= 1
    while start < len(turns) and _starts_with_results(turns[start]):
        start += 1
    if start == len(turns):
        # The newest exchange alone exceeds the cap: keep it anyway. It holds
        # the model's freshest (possibly still undelivered) results, and
        # paraphrasing those away is the one loss the tail exists to prevent.
        assistant_idxs = [i for i in range(1, len(turns)) if isinstance(turns[i], AssistantTurn)]
        if assistant_idxs:
            start = assistant_idxs[-1]
    return start


def _starts_with_results(turn: Turn) -> bool:
    return isinstance(turn, UserTurn) and any(
        isinstance(item, ToolResultItem) for item in turn.items
    )


# First target header in a unified diff (`+++ b/PATH`) or a v4a patch
# (`*** Update|Add File: PATH`). apply_patch is one-file-per-call, so the
# first match is the file.
_PATCH_TARGET_RE = re.compile(
    r"^(?:\+\+\+ b/(?P<u>\S+)|\*\*\* (?:Update|Add) File: (?P<v>.+))$", re.MULTILINE
)


def recently_edited_paths(conversation: Conversation, *, last_turns: int = 8) -> frozenset[str]:
    """Paths targeted by apply_edit / apply_patch in the last *last_turns*
    assistant turns: the files the worker is actively editing. Tier-1
    elision deprioritises their read_file results (see
    `compact_old_tool_results`), because a placeholder there triggers a paid
    re-read before the very next edit. Best-effort: an apply_patch without a
    `path` argument falls back to the patch headers; an unparseable patch
    just goes unprotected.
    """
    out: set[str] = set()
    seen_assistant = 0
    for turn in reversed(conversation.turns):
        if not isinstance(turn, AssistantTurn):
            continue
        seen_assistant += 1
        if seen_assistant > last_turns:
            break
        for tu in turn.tool_uses:
            if tu.name not in ("apply_edit", "apply_patch") or not isinstance(tu.input, dict):
                continue
            path = str(tu.input.get("path", "") or "")
            if not path and tu.name == "apply_patch":
                m = _PATCH_TARGET_RE.search(str(tu.input.get("patch", "")))
                path = ((m.group("u") or m.group("v") or "") if m else "").strip()
            if path:
                out.add(path)
    return frozenset(out)


def _tool_result_pointers(
    conversation: Conversation,
) -> tuple[list[tuple[int, int, int]], int]:
    """((turn_idx, item_idx, size) per tool_result, total size) in order."""
    pointers: list[tuple[int, int, int]] = []
    total = 0
    for turn_idx, turn in enumerate(conversation.turns):
        if isinstance(turn, AssistantTurn):
            continue
        for item_idx, item in enumerate(turn.items):
            if not isinstance(item, ToolResultItem):
                continue
            pointers.append((turn_idx, item_idx, len(item.content)))
            total += len(item.content)
    return pointers, total


def count_elisions(conversation: Conversation) -> tuple[int, int]:
    """The count of elision markers in the context, and of live gists among them.

    A resumed or forked leg re-announces these: a fork's fresh logs.jsonl has
    no compact.dropped events to fold, so the status surfaces would otherwise
    report zero over a restored context full of markers.
    """
    elided = gists = 0
    for turn in conversation.turns:
        for item in getattr(turn, "items", ()):
            body = getattr(item, "content", "")
            if isinstance(body, str) and body.startswith(ELISION_PREFIX):
                elided += 1
                gists += body.startswith(ELISION_GIST_PREFIX)
    return elided, gists


def compact_old_tool_results(
    conversation: Conversation,
    *,
    max_total_bytes: int,
    keep_recent: int = 2,
    protect_paths: frozenset[str] = frozenset(),
    gister: Gister | None = None,
) -> CompactionStats:
    """Elide old tool_result blocks once cumulative content exceeds the
    threshold. Walks the conversation oldest-first, replaces each tool_result's
    `content` with a short identity-bearing placeholder, stops once total
    size is back under `max_total_bytes`. The most recent `keep_recent`
    are always preserved, as is every tool_result in the last
    tool_result-bearing turn: the loop compacts at top-of-iteration, before
    the provider call that would deliver that batch, so the model has never
    seen it and the placeholder's "re-call the tool" guidance would trigger a
    paid re-call cycle. (Keying on the final turn alone is not enough: a
    trailing steer or nudge user turn pushes the fresh, still undelivered
    results off the final index, and one turn can carry several such blocks.)

    `protect_paths` (the actively-edited set from `recently_edited_paths`)
    deprioritises rather than exempts: read_file results for those paths are
    elided only after every other candidate, so the hot file's content
    survives as long as the budget allows but the hard bound still holds.

    With a `gister`, each large unprotected read_file victim decays to a
    placeholder carrying a distilled gist of the file (one batched distiller
    call per pass, newest read per path, caps above); everything else gets the
    bare marker. Gists make the pass land slightly OVER the bare-accounting
    plan, so when the applied total still exceeds the budget, existing gist
    placeholders are demoted oldest-first to the bare marker: content decays
    content -> gist -> bare marker, and the spec facts survive the longest
    while the byte bound still holds (in the limit everything is bare, exactly
    the pre-gist behavior). Demotion runs after even the protected reads are
    elided: losing a gist costs correctness (the file is gone from context),
    losing a hot read costs one paid re-read.

    Idempotent on already-elided entries.
    """
    pointers, total = _tool_result_pointers(conversation)
    if total <= max_total_bytes or len(pointers) <= keep_recent:
        return CompactionStats()

    # Dedup first: freeing duplicate bytes is lossless, and may spare real
    # content from elision below (or make it unnecessary).
    deduped_calls = _dedupe_identical_results(conversation, pointers, keep_recent=keep_recent)
    if deduped_calls:
        pointers, total = _tool_result_pointers(conversation)
        if total <= max_total_bytes:
            return CompactionStats(deduped_calls=deduped_calls)

    def _is_protected(turn_idx: int, item_idx: int) -> bool:
        call = _result_at(conversation, turn_idx, item_idx).for_call
        if call.name != "read_file" or not isinstance(call.input, dict):
            return False
        return str(call.input.get("path", "")) in protect_paths

    # The undelivered batch is always in the last tool_result-bearing turn:
    # at top-of-iteration only text-only steer/nudge user turns can trail the
    # fresh results, and the delivering provider call runs after this compaction.
    # Exempt that whole turn.
    last_turn = max(turn_idx for turn_idx, _, _ in pointers)
    candidates = [
        c
        for c in pointers[:-keep_recent]
        if c[0] != last_turn and not _is_operator_answer(conversation, c)
    ]
    if protect_paths:
        # Protected reads go last, each group staying oldest-first.
        candidates = [c for c in candidates if not _is_protected(c[0], c[1])] + [
            c for c in candidates if _is_protected(c[0], c[1])
        ]

    walk = _Tier1Pass(
        conversation=conversation,
        max_total_bytes=max_total_bytes,
        protect_paths=protect_paths,
        candidates=candidates,
        total=total,
    )
    walk.plan()
    if gister is not None:
        walk.distill(gister)
    walk.apply()
    walk.demote()
    return CompactionStats(
        elided_calls=tuple(walk.elided_calls),
        gist_paths=tuple(walk.gist_paths),
        demoted_paths=tuple(walk.demoted_paths),
        deduped_calls=deduped_calls,
    )


def _is_operator_answer(conversation: Conversation, pointer: tuple[int, int, int]) -> bool:
    """Whether this result is the operator's answer to an `ask_user`.

    Exempt from elision: it is a binding ruling that exists nowhere else in the
    model's context, and the placeholder's advice ("re-run it") means
    interrupting the operator to re-ask a question they have already answered.
    A handful of answers costs less than the re-ask."""
    return _result_at(conversation, pointer[0], pointer[1]).for_call.name == ASK_USER_TOOL


def _result_at(conversation: Conversation, turn_idx: int, item_idx: int) -> ToolResultItem:
    turn = conversation.turns[turn_idx]
    assert not isinstance(turn, AssistantTurn)
    item = turn.items[item_idx]
    assert isinstance(item, ToolResultItem)  # pointers only ever index tool_results
    return item


# Below this a duplicate's pointer placeholder is barely smaller than the
# content it replaces.
_DEDUP_MIN_CHARS = 200


def _dedupe_identical_results(
    conversation: Conversation,
    pointers: list[tuple[int, int, int]],
    *,
    keep_recent: int,
) -> tuple[str, ...]:
    """History-wide identical-result dedup, the tier-1 pass's first step.

    When the same call (name + input) produced byte-identical content more
    than once, every copy but the newest becomes a short placeholder. It runs
    only here, where history is being rewritten anyway, so it adds no new
    cache-invalidation points. The placeholder points at no other
    block, because the elision pass below can take the newest copy in the same
    call. Claude Code dedupes the same way; pi, which only ever compacts at the
    context edge, has no tier this could live in.

    The undelivered final batch, the `keep_recent` newest results,
    already-elided placeholders, and sub-`_DEDUP_MIN_CHARS` results are
    never rewritten.
    """
    if len(pointers) <= keep_recent:
        return ()
    last_turn = max(turn_idx for turn_idx, _, _ in pointers)
    exempt = {(t, i) for t, i, _ in pointers[-keep_recent:]}
    by_key: dict[tuple[str, str, str], list[tuple[int, int]]] = {}
    for turn_idx, item_idx, _size in pointers:
        item = _result_at(conversation, turn_idx, item_idx)
        if item.content.startswith(ELISION_PREFIX) or len(item.content) < _DEDUP_MIN_CHARS:
            continue
        call = item.for_call
        try:
            input_key = json.dumps(call.input, sort_keys=True, default=str)
        except (TypeError, ValueError):
            continue
        by_key.setdefault((call.name, input_key, item.content), []).append((turn_idx, item_idx))
    labels: list[str] = []
    for locs in by_key.values():
        for turn_idx, item_idx in locs[:-1]:  # every copy but the newest
            if turn_idx == last_turn or (turn_idx, item_idx) in exempt:
                continue
            item = _result_at(conversation, turn_idx, item_idx)
            label = call_label(item.for_call.name, item.for_call.input)
            marker = (
                f"{ELISION_PREFIX} (duplicate): {label} returned byte-identical"
                " content again later in this conversation. If you still need it,"
                " re-read only the part you need; do not re-issue the identical call.>"
            )
            if len(marker) >= len(item.content):
                # The label carries the call's arguments, so a long path can
                # make the marker bigger than the result it replaces: writing
                # it would GROW the total, as the elision pass also refuses to.
                continue
            conversation.set_result_content(turn_idx, item_idx, marker)
            labels.append(label)
    return tuple(labels)


@dataclass(slots=True)
class _Tier1Pass:
    """State shared by the phases of one tier-1 pass (the loop's `TurnState`
    pattern: one mutable object instead of six hand-threaded locals)."""

    conversation: Conversation
    max_total_bytes: int
    protect_paths: frozenset[str]
    candidates: list[tuple[int, int, int]]
    total: int
    victims: list[tuple[int, int, int]] = field(default_factory=list)
    gist_headroom: int = 0
    gists: dict[tuple[int, int], str] = field(default_factory=dict)
    elided_calls: list[str] = field(default_factory=list)
    gist_paths: list[str] = field(default_factory=list)
    demoted_paths: list[str] = field(default_factory=list)

    def _item(self, turn_idx: int, item_idx: int) -> ToolResultItem:
        return _result_at(self.conversation, turn_idx, item_idx)

    def plan(self) -> None:
        """Pick the victim set under bare-placeholder accounting (the maximum
        shrink); nothing is mutated yet so the distiller can still read the
        content."""
        planned = self.total
        for turn_idx, item_idx, size in self.candidates:
            if planned <= self.max_total_bytes:
                break
            item = self._item(turn_idx, item_idx)
            if item.content.startswith(ELISION_PREFIX):
                continue
            placeholder = elision_placeholder(item.for_call.name, item.for_call.input)
            if size <= len(placeholder):
                # Replacing content already smaller than the placeholder would
                # GROW the total, defeating the point; skip it.
                continue
            self.victims.append((turn_idx, item_idx, size))
            planned -= size - len(placeholder)
        # What a gist may add back on top of the bare-placeholder plan.
        self.gist_headroom = self.max_total_bytes - planned

    def distill(self, gister: Gister) -> None:
        """One batched distiller call over the eligible victims: large
        unprotected read_file results, the newest read per path, largest files
        first under the input caps."""
        newest_by_path: dict[str, tuple[int, int, int]] = {}
        for turn_idx, item_idx, size in self.victims:
            call = self._item(turn_idx, item_idx).for_call
            if call.name != "read_file" or not isinstance(call.input, dict):
                continue
            path = str(call.input.get("path", ""))
            if not path or path in self.protect_paths or size < GIST_MIN_SOURCE_CHARS:
                continue
            newest_by_path[path] = (turn_idx, item_idx, size)  # victims are oldest-first
        batch: list[GistRequest] = []
        keys: dict[str, tuple[int, int]] = {}
        input_budget = GIST_INPUT_CAP_CHARS
        by_size = sorted(newest_by_path.items(), key=lambda kv: kv[1][2], reverse=True)
        for path, (turn_idx, item_idx, _size) in by_size:
            if len(batch) >= GIST_MAX_FILES_PER_CALL or input_budget <= 0:
                break
            text = read_file_text_from_result(self._item(turn_idx, item_idx).content)
            if len(text) < GIST_MIN_SOURCE_CHARS:
                continue
            excerpt = text[: min(GIST_FILE_SLICE_CHARS, input_budget)]
            input_budget -= len(excerpt)
            batch.append(GistRequest(path=path, content=excerpt))
            keys[path] = (turn_idx, item_idx)
        if not batch:
            return
        for path, gist in gister(tuple(batch)).items():
            flat = " ".join(gist.split())
            if path in keys and flat:
                self.gists[keys[path]] = flat[:GIST_MAX_CHARS]

    def _landing_gists(self) -> dict[tuple[int, int], str]:
        """The distilled placeholders that land, chosen NEWEST-first: the newest
        read of a path is the one a later turn needs, and `demote` drops gists
        oldest-first for the same reason. A gist no smaller than the content it
        replaces never lands, nor does one costing more than the plan's headroom
        (`demote` would strip it before this same pass returned). A gist shorter
        than the bare marker costs nothing, so it always lands."""
        headroom = max(self.gist_headroom, 0)
        landing: dict[tuple[int, int], str] = {}
        for turn_idx, item_idx, size in reversed(self.victims):
            gist = self.gists.get((turn_idx, item_idx))
            call = self._item(turn_idx, item_idx).for_call
            if gist is None or not isinstance(call.input, dict):
                continue
            candidate = elision_gist_placeholder(call_label(call.name, call.input), gist)
            extra = len(candidate) - len(elision_placeholder(call.name, call.input))
            if len(candidate) < size and extra <= headroom:
                headroom -= max(extra, 0)
                landing[(turn_idx, item_idx)] = candidate
        return landing

    def apply(self) -> None:
        """Apply the whole plan (`plan` already chose the minimal set; gist
        placeholders only add back what the plan's headroom holds)."""
        landing = self._landing_gists()
        for turn_idx, item_idx, size in self.victims:
            call = self._item(turn_idx, item_idx).for_call
            gist_placeholder = landing.get((turn_idx, item_idx))
            placeholder = gist_placeholder or elision_placeholder(call.name, call.input)
            if gist_placeholder is not None and isinstance(call.input, dict):
                self.gist_paths.append(str(call.input.get("path", "")))
            self.conversation.set_result_content(turn_idx, item_idx, placeholder)
            self.total -= size - len(placeholder)
            self.elided_calls.append(call_label(call.name, call.input))

    def demote(self) -> None:
        """Still over budget (gist extras, or a shrunken budget with nothing
        fresh left): demote gist placeholders oldest-first to the bare marker
        until the bound holds or none remain."""
        if self.total <= self.max_total_bytes:
            return
        for turn_idx, item_idx, _size in self.candidates:
            if self.total <= self.max_total_bytes:
                break
            item = self._item(turn_idx, item_idx)
            if not item.content.startswith(ELISION_GIST_PREFIX):
                continue
            bare = elision_placeholder(item.for_call.name, item.for_call.input)
            if len(item.content) <= len(bare):
                continue
            self.conversation.set_result_content(turn_idx, item_idx, bare)
            self.total -= len(item.content) - len(bare)
            call = item.for_call
            self.demoted_paths.append(
                str(call.input.get("path", "")) if isinstance(call.input, dict) else ""
            )
