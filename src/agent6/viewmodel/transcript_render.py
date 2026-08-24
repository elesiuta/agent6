# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Render a session's per-call provider transcripts into a readable conversation.

agent6 writes one JSON file per LLM round-trip under `<run>/transcripts/` --
the full, lossless `{request, response}` (secrets redacted). Each request
carries the whole conversation up to that call, so the sequence is a complete,
self-contained record (no join with `logs.jsonl` needed). This module folds
that sequence, across the Chat Completions, Anthropic, and Responses wire
shapes, into an ordered list of conversation turns and renders them as
Markdown.

`agent6 sessions transcript` is the CLI front end (`--json` returns the raw
transcript array instead). The fold walks transcripts in seq order, emitting
only newly-introduced messages per call, so the cumulative-snapshot growth is
not double-printed and a mid-run context-compaction reset shows as a marker.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Matches ELISION_PREFIX in workflows/_compaction.py -- duplicated so the
# read-model needs no runtime import of the engine; a test pins the equality
# and the placeholder bytes themselves are pinned in the compaction tests.
ELISION_MARKER_PREFIX = "<elided by context compaction"
_GIST_MARKER_PREFIX = ELISION_MARKER_PREFIX + " (distilled)"
_ELIDED_IDENTITY_RE = re.compile(r": the result of (.+?) was replaced")


@dataclass
class Turn:
    """One normalized conversation turn (provider-agnostic).

    Deliberately mutable: `fold_conversation` builds a turn in a shape helper
    that does not know the call it came from, then stamps `seq` on it. Freezing
    would force threading seq through every builder for no gain.
    """

    role: str  # "system" | "user" | "assistant" | "tool" | "marker"
    text: str = ""
    thinking: str = ""
    tool_calls: list[tuple[str, str]] = field(default_factory=list)  # (name, args_json)
    tool_name: str = ""  # for role == "tool"
    seq: int = 0


# The seats whose round-trips ARE the conversation: the loop's driving provider,
# whose role differs by mode ("planner" in plan mode). Everything else shares the
# run's sink but is a side-call -- the gist distiller, the tier-2 summariser, a
# review seat -- and a side-call's one-message request reads as a
# compaction restart to the fold below, which then printed a phantom "context
# summarised" marker, rendered its scratch prompt as a turn, and re-emitted the
# history behind it. A transcript written before seats were stamped has none and
# is the driving seat's by default.
CONVERSATION_SEATS = frozenset({"worker", "planner"})


def load_transcripts(transcripts_dir: Path) -> list[dict[str, Any]]:
    """Every transcript JSON object under a session's transcripts/ dir, in seq
    order -- ALL seats. The raw list is `sessions transcript --json`'s output, the
    one CLI surface for a side-call's actual request/response; the conversation
    fold filters for itself (`conversation_transcripts`)."""
    if not transcripts_dir.is_dir():
        return []
    out: list[dict[str, Any]] = []
    for p in sorted(transcripts_dir.glob("*.json")):
        try:
            obj = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if isinstance(obj, dict):
            out.append(obj)
    out.sort(key=lambda t: t.get("seq", 0))
    return out


def conversation_transcripts(transcripts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Only the CONVERSATION_SEATS' round-trips (see the comment above)."""
    return [t for t in transcripts if str(t.get("seat", "") or "worker") in CONVERSATION_SEATS]


def _as_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except ValueError:
            return {}
    return value if isinstance(value, dict) else {}


def _request_body(t: dict[str, Any]) -> dict[str, Any]:
    return _as_dict(_as_dict(t.get("request")).get("body"))


def _response_body(t: dict[str, Any]) -> dict[str, Any]:
    return _as_dict(_as_dict(t.get("response")).get("body"))


def _shape(req: dict[str, Any], resp: dict[str, Any]) -> str:
    """Detect the provider wire shape of one transcript."""
    if isinstance(resp.get("choices"), list):
        return "openai"
    if isinstance(resp.get("content"), list) and resp.get("role"):
        return "anthropic"
    if isinstance(resp.get("output"), list) or isinstance(req.get("input"), list):
        return "responses"
    # Fall back on the request: Anthropic carries a top-level `system` and
    # content-block messages; OpenAI uses a system *message* + flat strings.
    return "anthropic" if "system" in req else "openai"


def _request_items(req: dict[str, Any], shape: str) -> list[Any]:
    """The request's conversation list: Responses `input`, else `messages`."""
    return (req.get("input") if shape == "responses" else req.get("messages")) or []


def _item_text(item: dict[str, Any]) -> str:
    """A Responses message item's text parts, joined."""
    content = item.get("content")
    if isinstance(content, str):
        return content
    return "".join(
        str(part.get("text", ""))
        for part in content or []
        if isinstance(part, dict) and part.get("type") in ("input_text", "output_text", "text")
    )


def _responses_turns(items: list[Any], names: dict[str, str]) -> list[Turn]:
    """Responses items -> turns. One model response spans several items
    (reasoning, a message, function calls), so consecutive assistant-side items
    fold into ONE assistant turn; a user message or a call output ends it."""
    turns: list[Turn] = []
    current: Turn | None = None

    def flush() -> None:
        nonlocal current
        if current is not None:
            turns.append(current)
            current = None

    for item in items:
        if not isinstance(item, dict):
            continue
        kind = item.get("type")
        if kind == "message" and item.get("role") != "assistant":
            flush()
            turns.append(Turn(role=str(item.get("role") or "user"), text=_item_text(item)))
            continue
        if kind == "function_call_output":
            flush()
            call_id = str(item.get("call_id", ""))
            output = str(item.get("output", ""))
            turns.append(Turn(role="tool", text=output, tool_name=names.get(call_id, "")))
            continue
        if current is None:
            current = Turn(role="assistant")
        if kind == "reasoning":
            summary = "\n".join(
                str(part.get("text", ""))
                for part in item.get("summary") or []
                if isinstance(part, dict) and part.get("text")
            )
            if summary:
                current.thinking = f"{current.thinking}\n{summary}".strip()
        elif kind == "message":
            text = _item_text(item)
            if text:
                current.text = f"{current.text}\n{text}".strip()
        elif kind == "function_call":
            name = str(item.get("name", ""))
            call_id = str(item.get("call_id") or item.get("id") or "")
            if call_id:
                names[call_id] = name
            current.tool_calls.append((name, _pretty_args(item.get("arguments", ""))))
    flush()
    return turns


def _same_item(a: Any, b: Any) -> bool:
    """Whether a replayed Responses input item is the recorded output item."""
    if not isinstance(a, dict) or not isinstance(b, dict) or a.get("type") != b.get("type"):
        return False
    key = "call_id" if a.get("type") == "function_call" else "id"
    return a.get(key) == b.get(key) if (a.get(key) or b.get(key)) else a == b


def _pretty_args(raw: Any) -> str:
    """Tool-call arguments -> compact one-line JSON (best effort)."""
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except ValueError:
            return raw.strip()
    try:
        return json.dumps(raw, ensure_ascii=False, sort_keys=True)
    except (TypeError, ValueError):
        return str(raw)


def _openai_turns(m: dict[str, Any], names: dict[str, str]) -> list[Turn]:
    role = m.get("role", "")
    if role == "tool":
        name = names.get(str(m.get("tool_call_id", "")), "")
        return [Turn(role="tool", text=str(m.get("content", "")), tool_name=name)]
    if role == "assistant":
        calls: list[tuple[str, str]] = []
        for tc in m.get("tool_calls") or []:
            fn = tc.get("function", {}) if isinstance(tc, dict) else {}
            name = str(fn.get("name", ""))
            calls.append((name, _pretty_args(fn.get("arguments", ""))))
            if isinstance(tc, dict) and tc.get("id"):
                names[str(tc["id"])] = name
        return [
            Turn(
                role="assistant",
                text=str(m.get("content") or ""),
                thinking=str(m.get("reasoning_content") or ""),
                tool_calls=calls,
            )
        ]
    return [Turn(role=role or "user", text=str(m.get("content") or ""))]


def _anthropic_turns(m: dict[str, Any], names: dict[str, str]) -> list[Turn]:
    role = m.get("role", "user")
    content = m.get("content")
    if isinstance(content, str):
        return [Turn(role=role, text=content)]
    text_parts: list[str] = []
    thinking_parts: list[str] = []
    calls: list[tuple[str, str]] = []
    tool_results: list[Turn] = []
    for b in content or []:
        if not isinstance(b, dict):
            continue
        match b.get("type"):
            case "text":
                text_parts.append(str(b.get("text", "")))
            case "thinking":
                thinking_parts.append(str(b.get("thinking", "")))
            case "tool_use":
                nm = str(b.get("name", ""))
                calls.append((nm, _pretty_args(b.get("input", {}))))
                if b.get("id"):
                    names[str(b["id"])] = nm
            case "tool_result":
                raw = b.get("content")
                txt = raw if isinstance(raw, str) else json.dumps(raw, ensure_ascii=False)
                nm = names.get(str(b.get("tool_use_id", "")), "")
                tool_results.append(Turn(role="tool", text=txt, tool_name=nm))
            case _:
                pass
    if role == "assistant":
        return [
            Turn(
                role="assistant",
                text="\n".join(text_parts).strip(),
                thinking="\n".join(thinking_parts).strip(),
                tool_calls=calls,
            )
        ]
    # A user message is either prose or a batch of tool_result blocks.
    if tool_results and not "".join(text_parts).strip():
        return tool_results
    return [Turn(role=role, text="\n".join(text_parts).strip())]


def _message_turns(m: dict[str, Any], shape: str, names: dict[str, str]) -> list[Turn]:
    return _openai_turns(m, names) if shape == "openai" else _anthropic_turns(m, names)


def _response_turns(resp: dict[str, Any], shape: str, names: dict[str, str]) -> list[Turn]:
    if shape == "responses":
        return _responses_turns(resp.get("output") or [], names)
    if shape == "openai":
        choices = resp.get("choices") or []
        if not choices:
            return []
        # A response message IS the assistant's, so stamp the role rather than
        # trusting the body to carry it (the streaming path synthesises the
        # message without one): without this the model's words rendered as the
        # user's, and its tool_calls and reasoning were dropped entirely.
        message = {**_as_dict(choices[0].get("message")), "role": "assistant"}
        return _openai_turns(message, names)
    if resp.get("content") is not None:
        return _anthropic_turns({"role": "assistant", "content": resp.get("content")}, names)
    return []


def _elided_strings(msg: dict[str, Any]) -> list[str]:
    """Every elision-placeholder string one wire message carries (either shape:
    an OpenAI `role: tool` string content, or Anthropic `tool_result` items)."""
    out: list[str] = []
    content = msg.get("content") if "content" in msg else msg.get("output")
    if isinstance(content, str):
        if content.startswith(ELISION_MARKER_PREFIX):
            out.append(content)
    elif isinstance(content, list):
        for item in content:
            if isinstance(item, dict):
                inner = item.get("content")
                if isinstance(inner, str) and inner.startswith(ELISION_MARKER_PREFIX):
                    out.append(inner)
    return out


def _elision_identity(placeholder: str) -> str:
    """The elided call's identity, recovered from the placeholder's own copy."""
    m = _ELIDED_IDENTITY_RE.search(placeholder)
    return m.group(1) if m else "a tool result"


def _elision_label(placeholder: str) -> str:
    label = _elision_identity(placeholder)
    if placeholder.startswith(_GIST_MARKER_PREFIX):
        label += " (distilled gist kept)"
    return label


def _elision_marker(prev: list[Any], msgs: list[Any], upto: int) -> str:
    """Marker text when old tool_results were mutated into elision placeholders
    between two request snapshots, or "" when none were. The conversation view
    keeps showing the original results; this line is the truth about what the
    MODEL still sees. Compares identity COUNTS, not placeholder bytes: a gist
    demoting to the bare marker is not re-reported, while a second result of
    the same identity elided in a later pass still is."""
    labels: list[str] = []
    for i in range(min(upto, len(prev), len(msgs))):
        cur_m, prev_m = msgs[i], prev[i]
        if not isinstance(cur_m, dict) or not isinstance(prev_m, dict):
            continue
        before = Counter(_elision_identity(s) for s in _elided_strings(prev_m))
        for s in _elided_strings(cur_m):
            ident = _elision_identity(s)
            if before[ident] > 0:
                before[ident] -= 1
            else:
                labels.append(_elision_label(s))
    if not labels:
        return ""
    shown = ", ".join(labels[:6]) + (f", +{len(labels) - 6} more" if len(labels) > 6 else "")
    noun = "result" if len(labels) == 1 else "results"
    return (
        f"context compaction: elided {len(labels)} older tool {noun}"
        f" from the model's context: {shown}"
    )


def fold_conversation(transcripts: list[dict[str, Any]]) -> list[Turn]:
    """Fold per-call transcripts into one ordered conversation (no double-print).

    Reconciles each request against the PRIOR one instead of predicting: a
    recorded response only reappears as the next request's `msgs[prev_len]`
    when the history actually grew. Error transcripts (a 5xx body) and
    empty-response retries re-send the identical message list, so blindly
    assuming one committed assistant message per transcript misread every
    provider retry as a compaction restart and re-printed the whole history.

    Folds only the conversation seats: a side-call's one-message request reads
    as a restart here (see `CONVERSATION_SEATS`).
    """
    transcripts = conversation_transcripts(transcripts)
    turns: list[Turn] = []
    names: dict[str, str] = {}  # tool_call/use id -> tool name (to label results)
    prev_len = 0  # messages of the prior request already emitted
    prev_msgs: list[Any] = []  # the prior request's messages, for elision diffing
    pending_response = False  # the prior transcript's response yielded turns
    prev_output: list[Any] = []  # a Responses call's output items, echoed by the next request
    for t in transcripts:
        seq = int(t.get("seq", 0))
        req = _request_body(t)
        resp = _response_body(t)
        shape = _shape(req, resp)
        msgs = _request_items(req, shape)
        # Anthropic and Responses keep the system prompt out of the message
        # list; surface it once.
        sys = req.get("system") if shape == "anthropic" else req.get("instructions")
        if shape in ("anthropic", "responses") and prev_len == 0 and sys:
            turns.append(
                Turn(role="system", seq=seq, text=sys if isinstance(sys, str) else json.dumps(sys))
            )
        if len(msgs) < prev_len:  # a context-compaction restart shrank the history
            turns.append(Turn(role="marker", text="context summarised / restarted", seq=seq))
            prev_len = 0
            pending_response = False
        elif marker := _elision_marker(prev_msgs, msgs, prev_len):
            turns.append(Turn(role="marker", text=marker, seq=seq))
        # The previously-emitted response is msgs[prev_len] only when the
        # history grew; a retry that re-sends the identical list skips nothing.
        # A Responses call's output is several items, echoed back verbatim.
        start = prev_len
        if shape == "responses":
            while (
                start - prev_len < len(prev_output)
                and start < len(msgs)
                and _same_item(msgs[start], prev_output[start - prev_len])
            ):
                start += 1
        elif pending_response and len(msgs) > prev_len:
            start = prev_len + 1
        fresh = msgs[start:]
        if shape == "responses":
            new_turns = _responses_turns(fresh, names)
        else:
            new_turns = [
                tt for m in fresh if isinstance(m, dict) for tt in _message_turns(m, shape, names)
            ]
        for tt in new_turns:
            tt.seq = seq
            turns.append(tt)
        response_turns = _response_turns(resp, shape, names)  # this call's assistant output
        for rt in response_turns:
            rt.seq = seq
            turns.append(rt)
        prev_len = len(msgs)
        prev_msgs = list(msgs)
        pending_response = bool(response_turns)
        prev_output = list(resp.get("output") or []) if shape == "responses" else []
    return turns


def _clip(s: str, n: int) -> str:
    return s if len(s) <= n else s[:n] + f"… (+{len(s) - n} chars)"


def render_markdown(
    turns: list[Turn],
    *,
    session_id: str,
    show_thinking: bool = True,
    tools: str = "both",
    result_cap: int = 4000,
) -> str:
    """Render folded turns as a Markdown conversation. `tools` in both|calls|none."""
    out: list[str] = [f"# Transcript: {session_id}", ""]
    for tn in turns:
        if tn.role == "marker":
            out.append(f"\n--- {tn.text} ---\n")
            continue
        if tn.role == "tool":
            if tools != "both":
                continue
            label = f" {tn.tool_name}" if tn.tool_name else ""
            out.append(f"  <-{label}: {_clip(tn.text, result_cap)}")
            out.append("")
            continue
        header = f"## {tn.role}"
        if tn.role == "assistant":
            header += f"  (seq {tn.seq})"
        out.append(header)
        if tn.thinking and show_thinking:
            out.append(f"<thinking>\n{tn.thinking}\n</thinking>")
        if tn.text:
            out.append(tn.text)
        if tn.tool_calls and tools != "none":
            out.extend(f"-> {name}({args})" for name, args in tn.tool_calls)
        out.append("")
    return "\n".join(out).rstrip() + "\n"
