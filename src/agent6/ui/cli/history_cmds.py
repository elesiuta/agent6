# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""`agent6 history search` and the `sessions graph` / `sessions transcript` verbs."""

from __future__ import annotations

import base64
import json
import shutil
import subprocess
import sys
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from agent6.config.layer import resolved_state_dir
from agent6.graph.storage import load_graph
from agent6.sessions.id import SessionIdError
from agent6.sessions.layout import LOGS_NAME, SESSION_BUCKETS, SessionLayout
from agent6.ui.cli._common import (
    all_session_dirs,
    error,
    newest_layout_holding,
    resolve_session_layout,
    sgr,
)
from agent6.ui.cli._task_tree import task_tree_lines
from agent6.viewmodel.transcript_render import (
    conversation_transcripts,
    fold_conversation,
    load_transcripts,
    render_markdown,
)


def _cmd_history_search(query: str, *, fixed: bool, session_id: str) -> int:
    rg = shutil.which("rg")
    if rg is None:
        error(
            "`rg` (ripgrep) is required for `agent6 history search`. "
            "Install ripgrep (https://github.com/BurntSushi/ripgrep) and retry."
        )
        return 2
    cwd = Path.cwd()
    if session_id:
        # Resolve across every bucket so an ask's logs/transcript are searchable.
        try:
            targets = [resolve_session_layout(cwd, session_id).session_dir]
        except SessionIdError as exc:
            error(f"{exc}")
            return 2
    else:
        # No id: search every session across every bucket, so a
        # search right after an `ask` finds it (matching what `agent6 sessions` lists).
        targets = all_session_dirs(cwd)
        if not targets:
            print("[agent6] no sessions to search yet.")
            return 1
    argv: list[str] = [rg, "--json"]
    if fixed:
        argv.append("--fixed-strings")
    argv.extend(["--", query, *(str(t) for t in targets)])
    completed = subprocess.run(argv, check=False, capture_output=True, text=True)
    if completed.returncode not in (0, 1):  # 1 == no matches, not an error
        sys.stderr.write(completed.stderr)
        return completed.returncode
    hits = _parse_rg_matches(completed.stdout)
    _render_history_hits(hits, resolved_state_dir(cwd))
    return 0 if hits else 1


# A search hit rendered readably: which run, when, the event type, and a snippet
# windowed around the match -- never the whole (possibly 400KB) JSON event line.
@dataclass(frozen=True, slots=True)
class _SearchHit:
    session_id: str
    when: str  # short clock time, or "" if the line is not a timestamped event
    kind: str  # event type, or the file's basename for non-event files
    snippet: str
    # Content identity for collapsing the same text across storage encodings:
    # the matched text + following context (normalized), or the snippet when
    # the context adds nothing (a match at end-of-string would otherwise merge
    # DIFFERENT sentences that merely end with the query word).
    key: str


_SNIPPET_HALF = 70  # chars kept either side of the match in a hit's snippet
_CORE_TAIL = 25  # chars of following context in a hit's content-identity key


def _normalize(text: str) -> str:
    """Decoded, lowercased, alphanumerics+spaces only: the normal form both
    sides of an identity comparison reduce to."""
    decoded = _collapse_escapes(text).lower()
    return "".join(ch for ch in decoded if ch.isalnum() or ch == " ").strip()


def _match_core(text: str, start: int, end: int) -> str:
    """A hit's content identity: the matched text plus a little FOLLOWING
    context, decoded and reduced to lowercase alphanumerics. One task string is
    stored in many encodings (the session.start event, manifest.json, the graph's
    dot labels, per-call transcripts); they differ in the syntax BEFORE the
    match ('"user_task": "' vs 'label="'), while the text after it is the same
    content everywhere, so a suffix-only key sees through the encodings.
    Empty when the following context adds nothing beyond the match itself --
    the caller must then key on the snippet instead of merging."""
    hi = min(len(text), end + _CORE_TAIL)
    core = _normalize(text[start:hi])
    return core if len(core) > len(_normalize(text[start:end])) else ""


def _strings_in(obj: object) -> Iterator[str]:
    """Every string value nested anywhere in a decoded JSON object.

    Iterative: json.loads accepts nesting right up to the interpreter's stack
    ceiling, so a recursive walk over what it returns can be the thing that
    blows up."""
    stack = [obj]
    while stack:
        item = stack.pop()
        if isinstance(item, str):
            yield item
        elif isinstance(item, dict):
            stack.extend(reversed(list(item.values())))
        elif isinstance(item, list):
            stack.extend(reversed(item))


def _field_snippet(raw: str, start: int, end: int) -> str | None:
    """When the matched line is a JSON object (a logs.jsonl event, a per-call
    transcript line), window inside the STRING FIELD holding the match: the
    snippet then reads as prose instead of a raw
    `"type": "role.thinking_delta", "text": " ...` fragment. None when the
    line is not a JSON object or the match sits on syntax/keys (the caller
    falls back to the raw-line window)."""
    try:
        event = json.loads(raw)
    except (ValueError, RecursionError):  # deep nesting raises RecursionError
        return None
    if not isinstance(event, dict):
        return None
    matched = _collapse_escapes(raw[start:end])
    if not matched.strip():
        return None
    for value in _strings_in(event):
        idx = value.find(matched)
        if idx >= 0:
            return _window(value, idx)
    return None


def _parse_rg_matches(rg_json: str) -> list[_SearchHit]:
    """Turn `rg --json` output into readable hits. Each match line is parsed as a
    logs.jsonl event when it is one (for its type + timestamp), and the snippet is
    a whitespace-collapsed window around the first match (inside the matched
    string field when the line is JSON), so a match buried in a huge tool/diff
    blob prints a short prose excerpt, not the entire line."""
    hits: list[_SearchHit] = []
    for line in rg_json.splitlines():
        try:
            rec = json.loads(line)
        except (ValueError, RecursionError):
            continue
        if rec.get("type") != "match":
            continue
        data = rec.get("data", {})
        path = Path(_rg_bytes(data.get("path")).decode("utf-8", "replace"))
        raw_bytes = _rg_bytes(data.get("lines")).rstrip(b"\n")
        raw = raw_bytes.decode("utf-8", "replace")
        subs = data.get("submatches") or []
        b_start = subs[0].get("start", 0) if subs else 0
        b_end = subs[0].get("end", b_start) if subs else b_start
        start, end = _char_span(raw_bytes, b_start, b_end)
        session_id = _session_id_from_path(path)
        when, kind = _event_when_kind(path, raw)
        snippet = _field_snippet(raw, start, end) or _window(raw, start)
        hits.append(
            _SearchHit(
                session_id=session_id,
                when=when,
                kind=kind,
                snippet=snippet,
                key=_match_core(raw, start, end) or snippet,
            )
        )
    return hits


def _kind_rank(hit: _SearchHit) -> int:
    """Readability order when the same content collapses across encodings: a
    timestamped event line beats the transcript record, which beats a rendered
    .md, which beats raw internals (manifest.json, graph.dot, per-call JSON)."""
    if hit.when:
        return 0
    if hit.kind == "transcript":
        return 1
    if hit.kind.endswith(".md"):
        return 2
    return 3


def _rg_bytes(field: object) -> bytes:
    """rg --json encodes a path or line as {"text": ...}, or {"bytes": <base64>}
    when it is not UTF-8: the bytes either way, so rg's byte offsets apply."""
    if isinstance(field, dict):
        if "bytes" in field:
            return base64.b64decode(str(field["bytes"]))
        return str(field.get("text", "")).encode("utf-8")
    return str(field or "").encode("utf-8")


def _char_span(line: bytes, b_start: int, b_end: int) -> tuple[int, int]:
    """rg's byte offsets as character offsets into the line decoded with
    U+FFFD for what is not UTF-8: the prefix decodes to the same characters
    the whole line does, so non-ASCII prose (curly quotes, ellipses) and a
    stray byte before the match shift nothing."""
    start = len(line[:b_start].decode("utf-8", "replace"))
    return start, start + len(line[b_start:b_end].decode("utf-8", "replace"))


def _session_id_from_path(path: Path) -> str:
    """The session id owning a match file: the child of the DEEPEST bucket
    segment (a state-base ancestor may reuse a bucket name, e.g.
    XDG_STATE_HOME=/mnt/runs/state)."""
    parts = path.parts
    anchors = set(SESSION_BUCKETS)
    for i in range(len(parts) - 2, -1, -1):
        if parts[i] in anchors:
            return parts[i + 1]
    return path.parent.name


def _event_when_kind(path: Path, raw: str) -> tuple[str, str]:
    """(clock-time, event-type) when the matched line is a logs.jsonl event;
    otherwise ("", a short label) for a transcript snapshot, plan.md, etc. The
    transcript snapshots are cumulative, so they get one shared "transcript"
    label to collapse the same text repeated across snapshots."""
    if path.name == LOGS_NAME:
        try:
            event = json.loads(raw)
        except (ValueError, RecursionError):
            return "", path.name
        ts = str(event.get("ts", ""))
        return ts[11:19] if len(ts) >= 19 else "", str(event.get("type", "event"))
    if path.parent.name == "transcripts":
        return "", "transcript"
    return "", path.name


def _collapse_escapes(s: str) -> str:
    """Render a JSON-encoded fragment readably, scanning left-to-right so a real
    escaped backslash (`\\\\`) is never mistaken for the start of a `\\n`.

    The old naive `str.replace("\\n", " ")` matched the `n` of a
    double-encoded newline (`\\\\n` in a transcript that embeds a JSON body),
    splitting the `\\\\` and leaving the ugly `\\ ` the operator saw. Here
    the whitespace escapes (`\\n` `\\t` `\\r`) become spaces,
    `\\\\` / `\\"` / `\\/` decode to their literal char, and `\\uXXXX`
    decodes to its character (surrogate pairs combined): transcripts are
    written ascii-escaped while logs.jsonl is raw UTF-8, and the identity key
    must see one form or the same content never collapses. An unknown,
    window-clipped, or lone-surrogate escape keeps its literal backslash text
    (printing a lone surrogate would raise on encode)."""
    out: list[str] = []
    i, n = 0, len(s)
    while i < n:
        c = s[i]
        if c == "\\" and i + 1 < n:
            nxt = s[i + 1]
            if nxt in "ntr":
                out.append(" ")
                i += 2
                continue
            if nxt in '\\"/':
                out.append(nxt)
                i += 2
                continue
            if nxt == "u":
                decoded = _decode_u_escape(s, i)
                if decoded is not None:
                    ch, consumed = decoded
                    out.append(ch)
                    i += consumed
                    continue
        out.append(c)
        i += 1
    return "".join(out)


def _hex4(s: str, i: int) -> int | None:
    """`int(s[i:i+4], 16)`, or None when truncated or not hex."""
    if i + 4 > len(s):
        return None
    try:
        return int(s[i : i + 4], 16)
    except ValueError:
        return None


def _decode_u_escape(s: str, i: int) -> tuple[str, int] | None:
    """Decode the `\\uXXXX` escape at `s[i]`, combining a surrogate PAIR
    into its real character; None keeps the literal text (malformed hex,
    truncated, or a lone surrogate)."""
    cp = _hex4(s, i + 2)
    if cp is None or 0xDC00 <= cp <= 0xDFFF:
        return None  # malformed/truncated, or a lone low surrogate
    if 0xD800 <= cp <= 0xDBFF:
        lo = _hex4(s, i + 8) if s[i + 6 : i + 8] == "\\u" else None
        if lo is None or not 0xDC00 <= lo <= 0xDFFF:
            return None  # a high surrogate without its pair
        return chr(0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00)), 12
    return chr(cp), 6


def _window(text: str, start: int) -> str:
    """A cleaned excerpt of *text* around byte offset *start*, capped at
    ~2*_SNIPPET_HALF chars with leading/trailing ellipses when it was clipped.
    JSON escapes inside transcript strings are decoded and real whitespace
    collapses to single spaces so the snippet reads as one line."""
    lo = max(0, start - _SNIPPET_HALF)
    hi = min(len(text), start + _SNIPPET_HALF)
    excerpt = " ".join(_collapse_escapes(text[lo:hi]).split())
    return f"{'…' if lo > 0 else ''}{excerpt}{'…' if hi < len(text) else ''}"


def _render_history_hits(hits: list[_SearchHit], target: Path) -> None:
    """Group hits by run, print a faded run header once, then one line per hit.
    Identical snippets within a run (the same system-prompt boilerplate matched
    in every transcript) collapse to one line with an `(xN)` count."""
    if not hits:
        print(f"[agent6] no matches under {target}.")
        return
    grouped: dict[str, list[_SearchHit]] = {}
    for hit in hits:
        grouped.setdefault(hit.session_id, []).append(hit)
    total = 0
    for i, (session_id, run_hits) in enumerate(grouped.items()):
        print("" if i == 0 else "\n", end="")
        print(sgr(session_id, "1"))
        # Dedup by content identity, not by file kind: one task string lives in
        # many storage encodings (session.start event, manifest, graph labels,
        # per-call transcripts) and cumulative transcript snapshots repeat the
        # same text; each collapses to ONE line, the most readable encoding
        # (see _kind_rank), with an (xN) count.
        counts: dict[str, int] = {}
        best: dict[str, _SearchHit] = {}
        for hit in run_hits:
            counts[hit.key] = counts.get(hit.key, 0) + 1
            cur = best.get(hit.key)
            if cur is None or _kind_rank(hit) < _kind_rank(cur):
                best[hit.key] = hit
        for key, hit in best.items():
            n = counts[key]
            # Show the timestamp only for a unique hit; a collapsed group spans
            # several times, so a count is clearer than any one of them.
            meta = "  ".join(p for p in (hit.when, hit.kind) if p) if n == 1 else hit.kind
            tag = f" {sgr(f'(x{n})', '2')}" if n > 1 else ""
            print(f"  {sgr(meta, '2')}  {hit.snippet}{tag}")
        total += len(run_hits)
    print(sgr(f"\n{total} match{'es' if total != 1 else ''} in {len(grouped)} run(s)", "2"))


def _cmd_history_graph(session_id: str) -> int:
    """Render the persisted TaskNode tree for a run as a DFS-ordered listing."""

    cwd = Path.cwd()
    if session_id:
        # Resolve across runs/ + asks/ so an ask's graph is findable too.
        try:
            layout = resolve_session_layout(cwd, session_id)
        except SessionIdError as exc:
            error(f"{exc}")
            return 2
    else:
        found = newest_layout_holding(cwd, "graph")
        if found is None:
            error(f"no sessions with a graph under {resolved_state_dir(cwd)}")
            return 2
        layout = found
        print(
            f"[agent6] showing graph for most recent session: {layout.session_id}",
            file=sys.stderr,
        )

    target_id = layout.session_id
    nodes = load_graph(layout)
    if not nodes:
        error(f"run {target_id} has no persisted graph nodes")
        return 2

    print(f"Session id: {target_id}")
    print()
    for line in task_tree_lines({nid: nodes[nid].model_dump() for nid in sorted(nodes)}):
        print(line)
    return 0


def _parse_seq_window(spec: str) -> tuple[int, int] | None:
    """`""` -> None (all); `"5"` -> (5,5); `"3-7"` -> (3,7). Raises ValueError on junk."""
    spec = spec.strip()
    if not spec:
        return None
    if "-" in spec:
        a, b = spec.split("-", 1)
        lo, hi = int(a), int(b)
        if lo > hi:
            raise ValueError(f"reversed window {spec!r}")
        return lo, hi
    n = int(spec)
    return n, n


def _transcript_layout(cwd: Path, session_id: str) -> SessionLayout | int:
    """Resolve the run whose transcripts to render: by id, else the most recent
    run that has a transcripts/ dir. An int is the exit code of a printed error."""
    if session_id:
        try:
            return resolve_session_layout(cwd, session_id)
        except SessionIdError as exc:
            error(f"{exc}")
            return 2
    found = newest_layout_holding(cwd, "transcripts")
    if found is None:
        error(f"no sessions with transcripts under {resolved_state_dir(cwd)}")
        return 2
    print(f"[agent6] transcript for most recent session: {found.session_id}", file=sys.stderr)
    return found


def _cmd_history_transcript(
    session_id: str, *, as_json: bool, no_thinking: bool, tools: str, seq: str
) -> int:
    """Render a run's full LLM conversation from its lossless per-call transcripts.

    The transcripts (`<run>/transcripts/*.json`) are the complete, self-
    contained record -- no join with logs.jsonl is needed. This is the CONVERSATION
    view (assistant text/thinking + every tool call with full I/O); for the terse
    EVENT timeline use `agent6 attach` / `agent6 history search`.
    """
    layout = _transcript_layout(Path.cwd(), session_id)
    if isinstance(layout, int):
        return layout

    try:
        window = _parse_seq_window(seq)
    except ValueError:
        error(f"--seq expects N or N-M with N <= M, got {seq!r}")
        return 2

    transcripts = load_transcripts(layout.transcripts_dir)
    if not transcripts:
        error(f"run {layout.session_id} has no transcripts")
        return 2

    if as_json:
        if window is not None:
            lo, hi = window
            transcripts = [t for t in transcripts if lo <= int(t.get("seq", 0)) <= hi]
        print(json.dumps(transcripts, indent=2, ensure_ascii=False))
        return 0

    if not conversation_transcripts(transcripts):
        # e.g. a review-only dir: every round-trip is a side-call seat, so the
        # conversation fold would print nothing at all.
        print(
            f"run {layout.session_id} has only side-call transcripts (review seats /"
            " compaction); --json dumps them raw.",
            file=sys.stderr,
        )
        return 2

    # Fold the FULL set (the per-seq walk needs every call), then window the turns.
    turns = fold_conversation(transcripts)
    if window is not None:
        lo, hi = window
        turns = [t for t in turns if lo <= t.seq <= hi]
    print(
        render_markdown(
            turns, session_id=layout.session_id, show_thinking=not no_thinking, tools=tools
        ),
        end="",
    )
    return 0
