# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Fold a session's event stream into an ordered conversation of `TranscriptItem`s.

The medium-agnostic half of live conversation rendering. `TranscriptFold` walks
`logs.jsonl` events in emission order and yields the things worth showing as
plain data: a reasoning block, an assistant message, a tool call (in flight,
then settled with its result), a commit, the final verdict. Each front-end (the
CLI ANSI stream, the TUI conversation, the web SPA, ACP) maps these items to
its own styling; the glyphs and content helpers here are shared so they never
drift.

One order everywhere: an item lands when it completes, so a tool call's
settled item lands after anything that landed during the call. The call shows
in flight (`ok=None`) from its `tool.call` until then, in the surface's
progress spot (the TUI pane, the web items, an ACP status).

`fold_transcript` is the batch form (whole stream at once); the CLI/TUI live
tailers feed the same `TranscriptFold` one event at a time.
"""

from __future__ import annotations

import difflib
import re
import shlex
from collections.abc import Callable, Iterable
from dataclasses import dataclass, replace
from typing import Any, Literal

from agent6.types import SESSION_KINDS
from agent6.viewmodel.events import SESSION_START_EVENTS, as_int, event_epoch, tool_result_ok
from agent6.viewmodel.format import format_usd, lane_count

# Terminal control sequences in MODEL-AUTHORED text and command output.
# Default-deny, not a CSI-only blocklist: stripping CSI alone left OSC intact
# (a demonstrated OSC 52 writes the terminal's clipboard) and DCS/SOS/PM/APC
# carry arbitrary payloads; a C1 byte opens the same doors 8-bit. Sequences are
# removed whole (a payload cut off at a chunk boundary surfaces as inert text);
# stray C0 controls (BEL, \r spoofing) and DEL drop too, keeping \n and \t.
# Consumed by every skin: the fold's previews and log lines, the streamed
# deltas, and the CLI's live stream. The TUI's own OSC 52 copy feature is
# agent6-authored output and never passes through here.
_CONTROL_RE = re.compile(
    r"\x1b\[[0-9;?]*[ -/]*[@-~]"  # CSI
    r"|\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)?"  # OSC, BEL- or ST-terminated (or cut off)
    r"|\x1b[PX^_][^\x1b]*(?:\x1b\\)?"  # DCS / SOS / PM / APC, ST-terminated
    r"|\x1b."  # any other escape, with its final byte
    r"|[\x00-\x08\x0b-\x1f\x7f\x80-\x9f]",  # stray C0 (keep \t \n) + DEL + C1
    re.DOTALL,
)


def scrub_terminal_controls(text: str) -> str:
    """Drop every terminal control sequence and stray control character from
    model-authored text (see `_CONTROL_RE`). Idempotent, so accumulating
    stream tails re-scrub for free."""
    return _CONTROL_RE.sub("", text)


# What the CLI's stdout, stderr and /dev/tty let through, whoever wrote it:
# SGR styling, conceal (SGR 8, which hides the text after it) excepted; a
# 256-colour grey (`38;5;8`) drops with it, and its text stays. A
# colour cannot move the cursor or rewrite a line; the approval prompt drops
# styling from the text under judgment all the same. The spinners' erase idiom
# and the composer's cursor movement go under the wrapper
# (`ui/cli/_terminal_guard.raw_stream`), since a carriage return and an erase
# in a file name would forge the line. Anything else the process prints (a
# commit subject, a summary, a task) is text, wherever it came from.
_TERMINAL_RE = re.compile(
    r"(?P<own>\x1b\[(?!(?:[0-9;]*;)?0*8(?:;|m))[0-9;]*m)|" + _CONTROL_RE.pattern, re.DOTALL
)


def scrub_terminal_output(text: str) -> str:
    """`scrub_terminal_controls` for everything the CLI writes to its terminal:
    SGR styling passes (conceal excepted), every other control sequence and
    stray control character drops."""
    return _TERMINAL_RE.sub(lambda m: m.group("own") or "", text)


# Shared glyph vocabulary (text characters, not graphics, so every terminal font
# renders them). One place so cli/tui/web agree.
CALL = "→"  # a tool call
RESULT = "└"  # its result, on the line below (U+2514: base box-drawing, renders in every mono font)
COMMIT = "✎"  # an auto-commit
THINK = "·"  # a reasoning block
DONE = "●"  # run start / final verdict
OPERATOR = "❯"  # noqa: RUF001 -- deliberate prompt glyph, not a mistyped >

# Tool names the loop treats as terminal; their call is folded into the final
# verdict rather than shown as an ordinary step. Kept as literals so viewmodel
# stays free of a tools import (layering).
_FINISH_TOOLS = frozenset({"finish_session", "finish_planning"})

# Friendly word for a session.end reason on the terminal/TUI "done" line, so a stop
# reads as "stopped" (not the raw "steer_abort") and an error names itself.
_END_REASON_LABEL = {
    "finish_session": "finished",
    "finish_planning": "planned",
    "answered": "answered",
    "silent_finish": "finished (no finish call)",
    "went_quiet": "went quiet",
    "budget_exhausted": "budget exhausted",
    "provider_error": "provider error",
    "metric_plateau": "metric plateaued",
    "verify_settled": "verify settled",
    "settled": "settled (unverified)",
    "no_progress": "no progress",
    "tool_error_stuck": "stuck on tool errors",
    "verify_command_unexecutable": "verify command could not run",
    "loop_guard_killed": "loop guard killed the run",
    "interactive_stop": "stopped interactively",
    "interrupted": "interrupted",
    "crashed": "crashed",
    "steer_abort": "stopped",
    "steer_exit": "stopped",
    "undone": "undone (forked back)",
    "detached": "detached",
    "prompt_revision_failed": "prompt revision failed",
    "plan_unreadable": "plan unreadable",
    "max_iterations": "hit iteration cap",
    "ask_repl_empty": "ask ended (empty input)",
    "gate_stale": "finished over a stale gate",
    "gate_red_at_base": "gate was already red before this run",
    "no_lane_result": "no lane produced a result",
    "no_lane_passed": "no lane passed its gate",
}

ItemKind = Literal["thinking", "text", "tool", "commit", "marker", "done", "operator"]


# Events that render BETWEEN turns rather than as part of one: {type: (kind,
# the field holding the text)}. An empty field renders nothing.
_BETWEEN_TURNS: dict[str, tuple[ItemKind, str]] = {
    # The operator's typed instruction: a steer, or the follow-up a resume
    # started with.
    "loop.steer.injected": ("operator", "text"),
    # A side question's answer: not the run's output, so it never joins its prose.
    "btw.answered": ("marker", "block"),
}

# Where the operator's own words live in the journal: the opening task, then
# every steer. Sibling of `_BETWEEN_TURNS` (renders steers) and of
# `tools.sessions._SPEAKER` (quotes them across sessions).
_OPERATOR_TEXT = {"session.start": "user_task", "loop.steer.injected": "text"}


def worker_models(events: Iterable[dict[str, Any]]) -> tuple[str, ...]:
    """The models that wrote code in a session: every `role.call` worker
    model, first-seen order (the primary worker first), deduplicated. Commit
    trailers read this; a message-writing model never joins the list."""
    seen: dict[str, None] = {}
    for event in events:
        if event.get("type") == "role.call" and event.get("role") == "worker":
            model = str(event.get("model", "")).strip()
            if model:
                seen.setdefault(model, None)
    return tuple(seen)


def operator_inputs(events: Iterable[dict[str, Any]]) -> list[str]:
    """The operator's typed messages, oldest first, consecutive repeats
    collapsed. Fed from the on-disk journal, so it spans resume legs and every
    surface's steers; input-history recall and search read it."""
    out: list[str] = []
    for event in events:
        field = _OPERATOR_TEXT.get(str(event.get("type", "")))
        if field is None:
            continue
        text = str(event.get(field, "")).strip()
        if text and (not out or out[-1] != text):
            out.append(text)
    return out


@dataclass(frozen=True, slots=True)
class TranscriptItem:
    """One rendered conversation step. Only the fields its `kind` needs are set."""

    kind: ItemKind
    body: str = ""  # thinking / text / marker prose; the final summary for `done`
    name: str = ""  # tool name
    arg: str = ""  # the tool's salient argument (path, pattern, command, ...)
    ok: bool | None = None  # tool or run outcome (None = not applicable / in flight)
    # Tool result summary, verify badge, or commit/done metadata; for a call in
    # flight, why it waits ("awaiting approval"), empty while it runs.
    detail: str = ""
    tail: str = ""  # a failed tool's captured output tail
    # The provider's stamped call_id, for a surface that pairs a tool's start
    # with its outcome by identity. Reconstructing one from name+arg made two
    # identical calls collide, and an editor keyed on it overwrote the first
    # call's FAILURE with the second's success.
    call_id: str = ""


_PRIMARY_ARGS = ("path", "file", "pattern", "query", "command", "cmd", "url", "title", "summary")


def _clip(text: str, n: int = 60) -> str:
    # One LINE by contract: every caller puts the clip on a single rendered
    # line, and an embedded newline (a multi-line arg value) split the tool
    # head in two on every skin.
    text = " ".join(text.split())
    return text if len(text) <= n else text[: n - 3] + "…"


def _call_preview(name: str, args: Any) -> str:
    """A bounded preview carried from the CALL side of a tool whose substance
    is in its arguments: apply_edit's first hunk (the journal carries the edit
    pairs nearly whole). Other tools carry none."""
    if name != "apply_edit" or not isinstance(args, dict):
        return ""
    edits = args.get("edits")
    if not isinstance(edits, (list, tuple)) or not edits:
        return ""
    first = edits[0]
    if not isinstance(first, dict):
        return ""
    old = str(first.get("old_string", ""))
    new = str(first.get("new_string", ""))
    # The changed lines only (an anchor line the edit re-emits unchanged is
    # not news), each clipped: the preview is multi-line by design (the tail
    # renders line-structured), while _clip's one-line contract caps each row.
    changed = [
        ln
        for ln in difflib.ndiff(old.splitlines(), new.splitlines())
        if ln[:1] in "+-" and ln[2:].strip()
    ]
    lines = [_clip(ln, 120) for ln in changed[:4]]
    if len(changed) > 4:
        lines.append(f"…(+{len(changed) - 4} more changed lines)")
    more = len(edits) - 1
    if more > 0:
        lines.append(f"…(+{more} more edit{'' if more == 1 else 's'})")
    return "\n".join(lines)


def salient_arg(args: Any) -> str:
    """The one argument worth showing beside a tool name (best effort). Takes
    untrusted event data, so a non-dict `args` is tolerated, not assumed away."""
    if not isinstance(args, dict) or not args:
        return ""
    # argv (run_command): a shell-style line, not a Python list repr.
    argv = args.get("argv")
    if isinstance(argv, (list, tuple)) and argv:
        return _clip(shlex.join(str(a) for a in argv))
    # ask_user: the question text, not the nested {questions:[{...}]} repr.
    questions = args.get("questions")
    if isinstance(questions, (list, tuple)) and questions:
        first = questions[0]
        q = first.get("question", "") if isinstance(first, dict) else str(first)
        more = f" (+{len(questions) - 1})" if len(questions) > 1 else ""
        return _clip(str(q)) + more
    for key in _PRIMARY_ARGS:
        value = args.get(key)
        if isinstance(value, (str, int)):
            return _clip(str(value))
    key, value = next(iter(args.items()))
    return _clip(f"{key}={value}")


def _parallel_group_label(event: dict[str, Any]) -> str:
    group = str(event.get("group", "")).strip()
    return f"group {group}" if group else "parallel"


def _parallel_dispatched_body(event: dict[str, Any]) -> str:
    """The coordinator dispatched a group: how many lanes (the count every
    listing shows) for how many tasks, and which tasks; a journal from
    before the event carried the lane count names the tasks alone (lane ids
    do not exist yet: the spawner names them)."""
    tasks_raw = event.get("tasks")
    tasks = [str(t).strip() for t in tasks_raw] if isinstance(tasks_raw, list) else []
    n = len(tasks)
    lanes = event.get("lanes")
    if isinstance(lanes, int) and not isinstance(lanes, bool):
        what = lane_count(lanes) + (f" for {n} tasks" if n > 1 else "")
    else:
        what = f"{n} parallel task{'' if n == 1 else 's'}"
    head = (
        f"dispatched {what} ({_parallel_group_label(event)});"
        " lanes run detached, listed under this session"
    )
    return "\n".join([head, *(f"• {_clip(t, 80)}" for t in tasks if t)])


def _parallel_compared_body(event: dict[str, Any]) -> str:
    """A fan-out ranked its candidates: best first, each with its gate
    verdict and cost, and who ranked them (the judge or the mechanical
    fallback)."""
    raw = event.get("ranking")
    rows = [r for r in raw if isinstance(r, dict)] if isinstance(raw, list) else []
    by = str(event.get("ranked_by", "")).strip() or "?"
    head = f"compared {_parallel_group_label(event)}: {len(rows)} candidate(s), ranked by {by}"
    lines = [head]
    for rank, r in enumerate(rows, start=1):
        cost = r.get("cost_usd")
        cost_s = format_usd(float(cost)) if isinstance(cost, (int, float)) else "?"
        lines.append(f"{rank}. {r.get('session_id', '?')}  {r.get('verify', '?')}  {cost_s}")
    return "\n".join(lines)


def _parallel_joined_body(event: dict[str, Any]) -> str:
    """The coordinator joined a group's lanes back: one line per lane naming its
    id, branch, status, and (when it landed) short sha."""
    lanes_raw = event.get("lanes")
    lanes = [ln for ln in lanes_raw if isinstance(ln, dict)] if isinstance(lanes_raw, list) else []
    head = f"joined {_parallel_group_label(event)}: {len(lanes)} lane(s)"
    rows: list[str] = []
    for ln in lanes:
        status = str(ln.get("status", "?"))
        parts = [str(ln.get("session_id", "?"))]
        branch = str(ln.get("branch", "")).strip()
        if branch:
            parts.append(branch)
        sha = str(ln.get("sha", "")).strip()
        if sha:
            parts.append(sha[:12])
        rows.append(f"{status}  {'  '.join(parts)}")
    return "\n".join([head, *rows])


def _parallel_failed_body(event: dict[str, Any]) -> str | None:
    """A `/parallel` dispatch failure (nothing was joined): name the group + error.

    Two shapes: a dispatch failure carries `error`; a post-join failure carries
    only `lanes` -- a subset of the joined event, which already showed each
    lane's status, so it renders nothing rather than a redundant marker.
    """
    error = str(event.get("error", "")).strip()
    if not error:
        return None
    return f"{_parallel_group_label(event)} dispatch failed: {error}"


def _mcp_unavailable_body(event: dict[str, Any]) -> str:
    """A configured MCP server that did not start: why, and what it costs.

    The tools it would have carried are simply absent, so without this the run
    looks normal and quietly cannot do what the operator configured it for. The
    error already names the server (every startup-path MCPError does), so the
    line adds only the consequence.
    """
    error = str(event.get("error", "")).strip()
    if not error:
        server = str(event.get("server", "")).strip() or "(unnamed)"
        return f"MCP server {server!r} is unavailable; its tools are missing"
    return f"{error}; its tools are missing"


# Events that render as a marker BETWEEN turns, each composing its own body.
# A builder returning None renders nothing.
def _compact_requested_body(event: dict[str, Any]) -> str:
    focus = str(event.get("focus", "")).strip()
    return f"compaction requested: {focus}" if focus else "compaction requested"


def _compact_done_body(event: dict[str, Any]) -> str:
    """Tier 2 replaced the history above this line with a summary."""
    chars = as_int(event.get("summary_chars"))
    kept = as_int(event.get("kept_turns"))
    return f"context compacted: {chars:,}-char summary, {kept} recent turns kept verbatim"


def _compact_failed_body(event: dict[str, Any]) -> str:
    error = str(event.get("error", "")).strip()
    return f"compaction failed: {error}" if error else "compaction failed"


def _compact_refused_body(event: dict[str, Any]) -> str:
    reason = str(event.get("reason", "")).strip()
    return f"compaction refused: {reason}" if reason else "compaction refused"


def _jail_degraded_body(event: dict[str, Any]) -> str:
    """The sandbox came up weaker than asked, or a stop left a process behind:
    the reason is the notice."""
    detail = " ".join(str(event.get("detail", "")).split())
    return f"sandbox degraded: {detail}" if detail else "sandbox degraded"


_MARKER_BODIES: dict[str, Callable[[dict[str, Any]], str | None]] = {
    "mcp.server_unavailable": _mcp_unavailable_body,
    "jail.degraded": _jail_degraded_body,
    "loop.parallel.dispatched": _parallel_dispatched_body,
    "loop.parallel.joined": _parallel_joined_body,
    "loop.parallel.failed": _parallel_failed_body,
    "loop.parallel.compared": _parallel_compared_body,
    # Compaction rewrites the history the reader is looking at: the surface
    # that promised a `/compact` "applies before the next model call" is the
    # one that says it did, failed, or was refused.
    "loop.compact.requested": _compact_requested_body,
    "loop.compact.summarise.done": _compact_done_body,
    "loop.compact.summarise.failed": _compact_failed_body,
    "loop.compact.refused": _compact_refused_body,
}


def _pending_key(event: dict[str, Any], name: str) -> int | str:
    """The pairing key for a tool.call/tool.result: the stamped call_id, or the
    name for id-less historical events."""
    cid = event.get("call_id")
    return cid if isinstance(cid, int) else name


# The roles whose output IS the session talking. Derived from the code table so
# it cannot drift: everything else (verify_inferer, summariser,
# reviewer) is a side call whose raw answer is not addressed to the operator.
DRIVING_ROLES: frozenset[str] = frozenset(k.role for k in SESSION_KINDS.values())


class TranscriptFold:
    """Incremental event -> `TranscriptItem` fold. Feed events in order; each
    `feed` returns the items that event produced (usually zero or one).

    A tool call is several items under one `call_id`, each superseding the
    last: in flight at `tool.call` (`ok=None`), marked awaiting while the
    prompt naming it (`approval.prompt` / `question.prompt`, by `call_id`) is
    open, settled at `tool.result`. A consumer keeping a list drops the
    superseded one (`fold_transcript` does). A leg boundary settles every
    call still open; a reader that knows the worker died calls
    `settle_open_calls` itself.
    """

    def __init__(self) -> None:
        self._thinking: list[str] = []
        self._text: list[str] = []
        # Calls awaiting their result, as (the in-flight item, the call-side
        # preview), keyed by the per-dispatch call_id: a concurrent
        # explore-tier review panel shares one dispatcher across threads, so
        # same-name calls interleave. An id-less historical event falls back
        # to its name key (sequential pairing).
        self._pending: dict[int | str, tuple[TranscriptItem, str]] = {}
        # The call each open prompt holds, by prompt id: the answer event
        # names only the prompt.
        self._gated: dict[str, int] = {}
        self._verify: tuple[bool, str] | None = None  # (ok, badge) for run_verify_command
        self._finish = ""  # summary from the terminal finish tool
        self._tools = 0
        self._commits = 0
        self._mode = ""  # from session.start; an ask or a plan never commits
        # Receipt state for the done item: cost from budget.update, wall time
        # from the first/last event ts, the last auto-commit's subject. Each
        # degrades to absent on a journal that never carried it.
        self._usd = 0.0
        self._first_ep: float | None = None
        self._last_ep: float | None = None
        self._commit_subject = ""
        # Pins already shown: a pin renders once, where it enters the
        # conversation (a /pin, a --pin at leg start), never again at a resume
        # boundary that restates the list.
        self._pins_shown: set[str] = set()

    def _fold_receipt(self, event: dict[str, Any], etype: str) -> bool:
        """Track the done item's receipt pieces (wall-clock span, cost, last
        commit subject); True when the event carried only receipt state."""
        if (ep := event_epoch(event.get("ts"))) is not None:
            self._last_ep = ep
            if self._first_ep is None:
                self._first_ep = ep
        if etype in SESSION_START_EVENTS:
            self._mode = str(event.get("mode", "")) or self._mode
            # The receipt is the leg's: a resumed leg's wall clock and counts
            # start at its own start event, as its cost already does.
            self._first_ep = ep
            self._tools = 0
            self._commits = 0
            self._commit_subject = ""
        if etype == "budget.update":
            self._usd = float(event.get("usd_total", 0) or 0)
            return True
        if etype == "loop.auto_commit":
            self._commit_subject = str(event.get("subject", "")).strip()
            return True
        return False

    def _receipt_detail(self) -> str:
        """The done item's detail: cost · wall · counts · commit subject, each
        piece present only when the journal carried it, so the story ends
        inside the surface instead of as post-exit shell text."""
        tools = f"{self._tools} tool{'' if self._tools == 1 else 's'}"
        commits = f"{self._commits} commit{'' if self._commits == 1 else 's'}"
        parts = []
        if self._usd:
            parts.append(f"${self._usd:.4f}")
        if self._first_ep is not None and self._last_ep is not None:
            parts.append(f"{max(0, round(self._last_ep - self._first_ep))}s")
        # An ask or a plan never commits: "0 commits" there is noise, not a fact
        # worth a receipt line.
        counts = (
            tools if self._mode in ("ask", "plan") and not self._commits else f"{tools} · {commits}"
        )
        parts.append(counts)
        if self._commit_subject:
            parts.append(_clip(self._commit_subject, 60))
        return " · ".join(parts)

    def _pin_items(self, event: dict[str, Any], etype: str) -> list[TranscriptItem]:
        """The operator's pins as an operator item (the other half of the
        dialogue, like a steer): the ones this fold has not shown yet."""
        raw = [event.get("text", "")] if etype == "loop.pin.added" else event.get("pins") or ()
        texts = [str(t).strip() for t in raw] if isinstance(raw, (list, tuple)) else []
        fresh = [t for t in texts if t and t not in self._pins_shown]
        if not fresh:
            return []
        self._pins_shown.update(fresh)
        out = self._flush_message()
        out.append(TranscriptItem("operator", body="pinned: " + " | ".join(fresh)))
        return out

    def feed(self, event: dict[str, Any]) -> list[TranscriptItem]:  # noqa: PLR0911, PLR0912
        etype = event.get("type", "")
        if self._fold_receipt(event, etype):
            return []
        if etype == "role.call":
            self._thinking.clear()
            self._text.clear()
            return []
        if etype in ("role.thinking_delta", "role.text_delta"):
            buffer = self._thinking if etype == "role.thinking_delta" else self._text
            buffer.append(str(event.get("text", "")))
            return []
        if etype == "role.result":
            # The settled text, used only when no deltas arrived: a streaming
            # leg already has the same prose in `self._text`.
            #
            # Only the role DRIVING the session speaks. agent6 makes side calls
            # with their own roles -- the verify-command inferer runs before the
            # loop starts -- and folding their results as messages opened an ACP
            # editor and the web conversation with a bare "[]", the inferer's
            # answer for "no verify command found", looking like the agent.
            settled = "" if self._is_side_call(event) else str(event.get("text", ""))
            return self._flush_message(settled=settled)
        if etype == "tool.call":
            out = self._flush_message()  # a turn's prose precedes its calls
            out.extend(self._start_tool(event))
            return out
        if etype in ("approval.prompt", "question.prompt"):
            why = "awaiting approval" if etype == "approval.prompt" else "awaiting answer"
            return self._mark_gated_call(event, why)
        if etype in ("approval.answer", "question.answer"):
            return self._release_gated_call(event)
        if etype == "verify.end":
            code = event.get("exit_code")
            dur = float(event.get("duration_s", 0) or 0)
            badge = "✓ pass" if code == 0 else f"✗ exit {code}"
            self._verify = (code == 0, f"{badge} · {dur:.1f}s")
            return []
        if etype == "tool.result":
            return self._complete_tool(event)
        if etype == "diff.updated":
            self._commits += 1
            n = len(str(event.get("patch", "")).splitlines())
            sha = str(event.get("sha", ""))[:12]
            return [TranscriptItem("commit", detail=f"{sha} · {n} lines" if sha else f"{n} lines")]
        build = _MARKER_BODIES.get(etype)
        if build is not None:
            body = build(event)
            if body is None:
                return []
            out = self._flush_message()  # a turn's prose precedes the marker
            out.append(TranscriptItem("marker", body=body))
            return out
        if etype in ("loop.pin.added", "loop.pin.restored"):
            return self._pin_items(event, etype)
        aside = _BETWEEN_TURNS.get(etype)
        if aside is not None:
            kind, field = aside
            out = self._flush_message()
            body = str(event.get(field, "")).strip()
            if body:
                out.append(TranscriptItem(kind, body=body))
            return out
        if etype in SESSION_START_EVENTS:
            return self.settle_open_calls("the run ended")
        if etype == "session.end":
            out = self.settle_open_calls("the run ended")
            out.extend(self._flush_message())
            counts = self._receipt_detail()
            reason = str(event.get("reason", ""))
            # Pair the finish summary with the done line ONLY on a clean finish
            # (a run's finish_session, a plan's finish_planning). On a
            # failure/stop the summary is from an EARLIER finish call and
            # pairing it (e.g. "provider error  Plan seeded.") misreads as success.
            body = self._finish if reason in ("", "finish_session", "finish_planning") else ""
            out.append(
                TranscriptItem(
                    "done",
                    body=body,
                    # The gate's tri-state, not a bool: null (no gate ran, or
                    # the operator ended the run) is neither pass nor fail;
                    # flattened, `stopped` and a gateless finish would take the
                    # failure colour of a finish over a RED gate, which exits 4.
                    ok=all_passed
                    if isinstance(all_passed := event.get("all_passed"), bool)
                    else None,
                    detail=counts,
                    name=_END_REASON_LABEL.get(reason, reason),
                )
            )
            return out
        return []

    def _is_side_call(self, event: dict[str, Any]) -> bool:
        """Whether this result belongs to a role other than one that DRIVES a
        session -- an inferer, summariser or reviewer.

        Allowlisted from the SessionKind table rather than listing the side
        roles, so a new driving mode is covered and a new side call is silent by
        default. An unnamed role is not a side call: older events carry none,
        and a streamed leg keeps its prose in the deltas anyway.
        """
        role = str(event.get("role", ""))
        return bool(role) and role not in DRIVING_ROLES

    def _flush_message(self, *, settled: str = "") -> list[TranscriptItem]:
        out: list[TranscriptItem] = []
        thinking = "".join(self._thinking).strip()
        self._thinking.clear()
        if thinking:
            out.append(TranscriptItem("thinking", body=thinking))
        text = "".join(self._text).strip() or settled.strip()
        self._text.clear()
        if text:  # only when non-empty: no more blank response blocks
            out.append(TranscriptItem("text", body=text))
        return out

    def _start_tool(self, event: dict[str, Any]) -> list[TranscriptItem]:
        """A dispatched call: its in-flight item, kept until the result. A
        finish tool's summary is the done line's, never an item."""
        name = str(event.get("name", ""))
        if name in _FINISH_TOOLS:
            self._finish = str((event.get("args") or {}).get("summary", "")).strip()
            return []
        self._tools += 1
        args = event.get("args") or {}
        key = _pending_key(event, name)
        out: list[TranscriptItem] = []
        if key in self._pending:
            # An id-less journal pairs by name: a second call under the name
            # takes the key, and the first would never settle.
            first, _preview = self._pending.pop(key)
            out.append(replace(first, ok=False, detail="no result (superseded)"))
        pending = TranscriptItem("tool", name=name, arg=salient_arg(args), call_id=str(key))
        self._pending[key] = (pending, _call_preview(name, args))
        self._verify = None
        out.append(pending)
        return out

    def _mark_gated_call(self, event: dict[str, Any], why: str) -> list[TranscriptItem]:
        """The call the prompt names (its `call_id`) re-emitted with *why* it
        waits; nothing for a prompt naming no call in flight (a question
        before the run starts, a verify the harness runs itself)."""
        key = event.get("call_id")
        if not isinstance(key, int) or key not in self._pending:
            return []
        self._gated[str(event.get("id", ""))] = key
        return [self._redetail(key, why)]

    def _release_gated_call(self, event: dict[str, Any]) -> list[TranscriptItem]:
        """The answered prompt's call re-emitted as running again."""
        key = self._gated.pop(str(event.get("id", "")), None)
        if key is None or key not in self._pending:
            return []
        return [self._redetail(key, "")]

    def _redetail(self, key: int, detail: str) -> TranscriptItem:
        pending, preview = self._pending[key]
        marked = replace(pending, detail=detail)
        self._pending[key] = (marked, preview)
        return marked

    def _complete_tool(self, event: dict[str, Any]) -> list[TranscriptItem]:
        name = str(event.get("name", ""))
        key = _pending_key(event, name)
        if key not in self._pending:  # a finish tool's result, or an unmatched one
            return []
        pending, call_preview = self._pending.pop(key)
        if name == "run_verify_command" and self._verify is not None:
            ok, detail = self._verify
            self._verify = None
        else:
            ok = tool_result_ok(event.get("ok"))
            detail = str(event.get("summary", "")).strip()
        # A failed tool shows why (stderr, else stdout). On success the tail is
        # the item's substance: command output (the operator ran it to SEE it),
        # a read's head preview with the true line count, or the edit's hunk
        # carried from the call side. Absent fields degrade to no tail.
        if not ok:
            tail = str(event.get("stderr_tail") or event.get("stdout_tail") or "").strip()
        elif name in ("run_command", "run_metric_command"):
            tail = str(event.get("stdout_tail") or "").strip()
        elif name == "read_file":
            head = str(event.get("head_tail") or "").strip("\n")
            total = event.get("lines_total")
            tail = f"{head}\n…({total} lines)" if head and total else head
        else:
            tail = call_preview
        return [
            replace(
                pending,
                ok=ok,
                detail=scrub_terminal_controls(detail),
                tail=scrub_terminal_controls(tail),
            )
        ]

    def settle_open_calls(self, why: str) -> list[TranscriptItem]:
        """Every call still in flight settled as one that never returned,
        with *why* ("the run ended" at a leg boundary; "the run died" from a
        reader whose worker probe found the worker gone)."""
        out = [
            replace(pending, ok=False, detail=f"no result ({why})")
            for pending, _preview in self._pending.values()
        ]
        self._pending.clear()
        self._gated.clear()
        return out


def _land(out: list[TranscriptItem], item: TranscriptItem) -> None:
    """Append *item*; a tool item first drops the in-flight one it supersedes
    (near the end: calls in flight are recent)."""
    if item.kind == "tool":
        for i in range(len(out) - 1, -1, -1):
            earlier = out[i]
            if earlier.kind == "tool" and earlier.ok is None and earlier.call_id == item.call_id:
                del out[i]
                break
    out.append(item)


def fold_transcript(
    events: list[dict[str, Any]], *, worker_dead: bool = False
) -> list[TranscriptItem]:
    """Fold a whole event stream into its ordered conversation items, one per
    tool call. *worker_dead* (the caller probed the worker and it is gone)
    settles the calls still open at the end: nothing will."""
    fold = TranscriptFold()
    out: list[TranscriptItem] = []
    for event in events:
        for item in fold.feed(event):
            _land(out, item)
    if worker_dead:
        for item in fold.settle_open_calls("the run died"):
            _land(out, item)
    return out


def _outcome_word(item: TranscriptItem) -> str:
    """A tool item's bracket word in a restatement: its verdict, or (in
    flight) why it waits, else "running"."""
    if item.ok is None:
        return item.detail or "running"
    return "ok" if item.ok else "FAILED"


def restate(events: list[dict[str, Any]], *, worker_dead: bool = False) -> str:
    """The conversation since the operator's last prompt or steer, compacted:
    their words, then assistant prose kept whole with tool calls and markers
    one line each. Rendered from the journal, never a model call, so every
    surface answers `/restate` locally and free."""
    last: int | None = None
    for i, event in enumerate(events):
        if str(event.get("type", "")) in _OPERATOR_TEXT:
            last = i
    if last is None:
        return "nothing to restate: this session has no operator input yet"
    anchor = events[last]
    said = str(anchor.get(_OPERATOR_TEXT[str(anchor["type"])], "")).strip()
    lines = [f"you said: {_clip(said, 200)}"]
    for item in fold_transcript(events[last:], worker_dead=worker_dead):
        if item.kind in ("thinking", "operator"):
            continue
        if item.kind == "text":
            body = item.body.strip()
            if body:
                lines.extend(("", body))
        elif item.kind == "tool":
            arg = f" {item.arg}" if item.arg else ""
            detail = f": {_clip(item.detail, 80)}" if item.detail else ""
            lines.append(f"  [{_outcome_word(item)}] {item.name}{arg}{detail}")
        else:  # commit / marker / done
            body = (item.body or item.detail).strip()
            if body:
                lines.append(f"  {body}")
    if len(lines) == 1:
        lines.append("(nothing has happened since)")
    return "\n".join(lines)
