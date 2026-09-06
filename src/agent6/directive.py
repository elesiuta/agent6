# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The steer-directive grammars (`/parallel`, `/pin`, `/compact`, `/btw`), shared by
the coordinator steer parser (`workflows/loop.py`) and the web + TUI composers.

    /parallel [spec] <task text> [/parallel [spec] <task text>]...
    /pin <instruction that must survive context compaction>
    /compact [focus text for the summary]

- `spec` is a positive int (lane count) or a comma-separated model list, and
  is OPTIONAL: omitted means one lane on the configured worker model. A
  segment's first token counts as a spec when it contains a comma OR a slash
  (model ids are provider/model shaped, e.g. `moonshotai/kimi-k2.6`); a bare
  comma-less slash-less model name (`opus`) intentionally stays task text --
  it is indistinguishable from a task word. The flip side: a task whose FIRST
  word is a path (`src/foo.py`) parses as a bogus model spec, refused
  pre-spawn with a did-you-mean (`models.validate`) when a model cache exists
  to check against, else it runs and fails at the provider call; start with a verb.
- The exact token `/parallel`, whitespace-delimited, separates tasks. A
  message is a directive only when it STARTS with the exact `/parallel` token;
  `/parallelfoo ...` stays ordinary text, byte-for-byte. A mid-task
  `/parallel` inside a word or path (not whitespace-delimited) is ordinary text
  too.
- Newlines are ordinary task characters, so a task can span multiple lines.

One parser per directive, imported by `workflows` (the coordinator) and
`ui` (the composers, and the CLI `--parallel` value via
:func:`parse_spec`). Pure stdlib string parsing, no agent6 imports -- a leaf
both layers sit above."""

from __future__ import annotations

import re
from dataclasses import dataclass

# A `/parallel` token that is whitespace-delimited: preceded by string start or
# whitespace, followed by whitespace or string end. NOT re.MULTILINE -- a newline
# is task text; `\s` already covers it, so a bare whitespace-delimited /parallel
# IS a separator while `foo/parallel/bar` (in a path) is not. `\A`/`\Z` anchor to
# the whole string, never to line boundaries.
_SEPARATOR = re.compile(r"(?:\A|(?<=\s))/parallel(?=\s|\Z)")


class DirectiveError(ValueError):
    """A `/parallel` directive or spec was malformed: a bare token, a segment
    with no task, a non-positive or over-`max_lanes` lane count, or an empty
    model list."""


@dataclass(frozen=True, slots=True)
class Segment:
    """One parsed `/parallel` task: its optional `spec` (`""` = one default
    lane) and the `task` text (internal whitespace and newlines preserved)."""

    spec: str
    task: str


def parse_spec(spec: str, *, limit: int) -> list[str | None]:
    """A spec string -> one entry per lane: `None` = the configured worker
    model, else a per-lane model override. `""` (omitted) is one default lane.

    A positive integer `N` is N default lanes; a comma-separated list is one
    lane per named model (a single model id, e.g. `provider/model`, is a
    one-lane list). *limit* is the caller's `[parallel].max_lanes`; an
    over-limit count refuses BEFORE the lane list is built, so a mistyped huge
    count cannot allocate it. Raises DirectiveError on a non-positive or
    over-limit count or a list that names no models. Single source for the
    directive spec AND the CLI `run --parallel <spec>` value grammar."""
    s = spec.strip()
    if not s:
        return [None]
    # isdecimal, not isdigit: isdigit() is True for superscripts/circled
    # digits ('\u00b2') that int() rejects, so the guard raised a bare
    # ValueError past every DirectiveError-catching caller (the coordinator's
    # never-end-the-run contract included). isdecimal() is exactly the set
    # int() parses for a stripped, sign-less string.
    if s.isdecimal():
        n = int(s)
        if n < 1:
            raise DirectiveError("parallel lane count must be >= 1")
        if n > limit:
            raise DirectiveError(_over_limit(n, limit))
        return [None] * n
    models = [m.strip() for m in s.split(",") if m.strip()]
    if not models:
        raise DirectiveError(f"parallel spec {spec!r} names no models")
    if len(models) > limit:
        raise DirectiveError(_over_limit(len(models), limit))
    return list(models)


def _over_limit(requested: int, limit: int) -> str:
    return (
        f"parallel spec requests {requested} lanes but [parallel].max_lanes = {limit}."
        " Request fewer, or raise [parallel].max_lanes."
    )


# A leading `/pin` token, whitespace-delimited: optional leading whitespace,
# then the exact token, then whitespace or end. Same discipline as _SEPARATOR
# (mid-text or glued tokens are ordinary steer text), but /pin never splits a
# message: everything after the token is one pinned instruction.
_PIN_TOKEN = re.compile(r"\A\s*/pin(?=\s|\Z)")


def parse_pin(text: str) -> str | None:
    """The instruction a `/pin` steer carries, or `None` when *text* is not a
    pin directive (does not start with the exact `/pin` token). Internal
    newlines are instruction text. Raises DirectiveError on a bare `/pin`."""
    m = _PIN_TOKEN.match(text)
    if m is None:
        return None
    instruction = text[m.end() :].strip()
    if not instruction:
        raise DirectiveError("pin needs an instruction: /pin <text that must survive compaction>")
    return instruction


# A leading `/compact` token, same discipline as _PIN_TOKEN. Parsed by the
# composers (web/TUI) and the CLI pause menu, NOT by the loop: a compact
# request is an out-of-band marker, not steer text.
_COMPACT_TOKEN = re.compile(r"\A\s*/compact(?=\s|\Z)")


def parse_compact(text: str) -> str | None:
    """The summary focus a `/compact` composer message carries ("" for a bare
    /compact), or `None` when *text* is not a compact directive."""
    m = _COMPACT_TOKEN.match(text)
    if m is None:
        return None
    return text[m.end() :].strip()


# A leading `/btw` token, same discipline as the two above. A btw is a
# QUESTION asked beside the run, never steer text: it must not reach the loop.
_BTW_TOKEN = re.compile(r"\A\s*/btw(?=\s|\Z)")


def parse_btw(text: str) -> str | None:
    """The question a `/btw` composer message carries, or `None` when *text*
    is not a btw directive. A bare `/btw` carries "" -- there is nothing to
    ask, and the caller says so rather than opening an empty session."""
    m = _BTW_TOKEN.match(text)
    if m is None:
        return None
    return text[m.end() :].strip()


# A leading `/now` token: the urgency the CLI spells `steer --now`. Parsed by
# the composers (web/TUI), never by the loop: the request marker carries it.
_NOW_TOKEN = re.compile(r"\A\s*/now(?=\s|\Z)")


def parse_now(text: str) -> str | None:
    """The steer a `/now` composer message carries, to be taken by aborting the
    in-flight model call, or `None` when *text* is not a now directive. A bare
    `/now` carries "": there is nothing to steer with, and the caller says so."""
    m = _NOW_TOKEN.match(text)
    if m is None:
        return None
    return text[m.end() :].strip()


# The spec token of a `/parallel` directive still being typed: the message
# is `/parallel <token>` with nothing after it yet (a following space = task
# text has begun, so stop suggesting).
_SPEC_TAIL = re.compile(r"/parallel[^\S\n]+(\S*)\Z")


def spec_fragment(text: str) -> str | None:
    """The comma-separated model fragment under construction at the end of a
    `/parallel` spec (a composer's autocomplete key), or None when the caret
    has left the spec or the token is a bare lane count."""
    m = _SPEC_TAIL.match(text)
    if m is None:
        return None
    token = m.group(1)
    if token.isdigit():
        return None
    return token.rsplit(",", 1)[-1]


def steer_problem(text: str) -> str | None:
    """Why *text* cannot START a leg as its steer: a malformed directive (a
    bare `/pin`, a `/parallel` with no task) or a live-run command (`/compact`,
    `/btw`, `/restate`: a composer or the pause menu acts on those; the loop
    would hand the bare token to the model as an instruction). None for
    ordinary text and a well-formed directive. A leg spent on a directive the
    loop can only decline reads as a silent finish and flips a passed run to
    failed."""
    live_only = "acts on a live run (from a composer or the pause menu); it cannot start a leg"
    if parse_compact(text) is not None:
        return f"/compact {live_only}"
    if parse_btw(text) is not None:
        return f"/btw {live_only}"
    if _RESTATE_TOKEN.match(text):
        return f"/restate {live_only}"
    if _SHELLS_TOKEN.match(text):
        return f"/shells {live_only}"
    try:
        parse_pin(text)
        parse_directive(text)
    except DirectiveError as exc:
        return str(exc)
    return None


_RESTATE_TOKEN = re.compile(r"\A\s*/restate(?=\s|\Z)")
_SHELLS_TOKEN = re.compile(r"\A\s*/shells(?=\s|\Z)")


def parse_directive(text: str) -> list[Segment] | None:
    """Split a `/parallel` message into its task segments, or `None` when *text*
    is not a directive (does not start with the exact `/parallel` token).

    Each whitespace-delimited `/parallel` token starts a new segment. Within a
    segment, the first whitespace-delimited token is the spec when it is a
    positive int or contains a comma or slash (a model list / model id), else
    the whole segment is the task. Raises DirectiveError on a segment with no
    task (a bare `/parallel`, or a spec with nothing after it) --
    all-or-nothing, so a later empty segment fails the whole parse."""
    body = text.lstrip()
    matches = list(_SEPARATOR.finditer(body))
    if not matches or matches[0].start() != 0:
        return None
    segments: list[Segment] = []
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(body)
        segments.append(_parse_segment(body[m.end() : end]))
    return segments


def _is_spec_token(token: str) -> bool:
    """A leading token is a spec iff it is a positive integer, a comma list, or
    contains a slash (a provider/model id; no natural task starts with a
    slash-containing word -- see the module docstring for the path caveat). A
    bare word (`fix`, a single model name with no comma or slash) is task
    text."""
    return token.isdecimal() or "," in token or "/" in token


def _parse_segment(raw: str) -> Segment:
    body = raw.strip()
    if not body:
        raise DirectiveError(
            "/parallel needs a task, e.g. `/parallel fix the bug` or `/parallel 2 fix the bug`"
        )
    parts = body.split(None, 1)
    if _is_spec_token(parts[0]):
        spec, task = parts[0], (parts[1] if len(parts) > 1 else "")
    else:
        spec, task = "", body
    if not task:
        raise DirectiveError(
            f"/parallel {parts[0]} needs a task, e.g. `/parallel {parts[0]} fix the bug`"
        )
    return Segment(spec=spec, task=task)


# The directives a composer can complete, with one-line help: exactly what the
# TUI/web composers and the loop parse out of steer text (`/btw` stays a CLI
# pause-menu command, and that menu keeps its own richer table). The web client
# mirrors these strings verbatim, drift-pinned by tests/web.
STEER_COMMANDS: dict[str, str] = {
    "/pin": "pin an instruction that survives compaction: /pin <text>",
    "/compact": "compact the context now; /compact <focus> steers the summary",
    "/parallel": "fan out lanes: /parallel [N|models] <task> (repeat to queue more)",
    "/restate": "restate the conversation since your last message (local, no model call)",
    "/undo": "fork back to before your last message (the text returns to edit and resend)",
    "/btw": "ask a question beside the run: /btw <question> (answers inline, later)",
    "/now": "steer at once, aborting the call in flight: /now <text> (Ctrl+Enter on the web)",
    "/shells": "background commands this run started, and how they ended",
}
