# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Project the shared transcript fold into ACP `session/update` notifications.

The fold (`viewmodel.transcript`) is what the CLI, the TUI and the web already
render. Projecting it -- rather than reading the journal again with ACP's own
rules -- is what keeps a fourth surface from disagreeing with the other three
about what happened in a run.

Pure: events in, notification bodies out. Nothing here touches the wire, so a
test can assert the exact JSON an editor would receive.
"""

from __future__ import annotations

from typing import Any

from agent6.viewmodel.transcript import TranscriptItem

# Which ACP update a fold item becomes. `thinking` is the model's reasoning and
# ACP has a distinct channel for it; an editor renders it collapsed rather than
# as the answer. `operator` is the human's own words -- a steer, or the
# follow-up a resume began with -- so it echoes back as a user message, not as
# something the agent said.
_CHUNK_KIND = {
    "thinking": "agent_thought_chunk",
    "text": "agent_message_chunk",
    "operator": "user_message_chunk",
    # Harness prose: a compaction, a btw answer, an operator notice. Not the
    # model speaking, but it IS what the run said.
    "marker": "agent_message_chunk",
}


def updates_for(
    item: TranscriptItem,
    *,
    acp_session_id: str,
    wire_id: str = "",
    announced: bool = False,
) -> list[dict[str, Any]]:
    """The `session/update` notifications one fold item becomes.

    A tool call is announced once (`tool_call`, from its first in-flight item)
    and updated after that (`tool_call_update`: awaiting approval, running
    again, settled), paired by *wire_id* (`tool_call_id`; the leg's own stamp
    when none is given); *announced* says the editor already has the call.
    ACP models a tool call as a thing with a lifecycle, and an editor that
    only sees the finished one cannot show work in progress, which for a
    long verify is the whole point.
    """
    if item.kind == "done":
        return [
            _update(
                acp_session_id,
                {"sessionUpdate": "agent_message_chunk", "content": _text(ending(item))},
            )
        ]
    if item.kind == "commit":
        # `body` is empty on a commit; the sha and the line count live in
        # `detail`. Keying on body alone dropped every auto-commit.
        text = " ".join(part for part in ("committed", item.arg, item.detail) if part)
        return [
            _update(
                acp_session_id, {"sessionUpdate": "agent_message_chunk", "content": _text(text)}
            )
        ]
    if item.kind == "tool":
        wire_id = wire_id or _leg_call_id(item)
        if item.ok is None and not announced:
            return [
                _update(acp_session_id, {"sessionUpdate": "tool_call", **_tool_call(item, wire_id)})
            ]
        return [
            _update(
                acp_session_id,
                {
                    "sessionUpdate": "tool_call_update",
                    "toolCallId": wire_id,
                    "status": _tool_status(item),
                    **({"content": _tool_content(item)} if _tool_content(item) else {}),
                },
            )
        ]
    chunk = _CHUNK_KIND.get(item.kind)
    body = item.body.strip()
    if chunk is None or not body:
        return []
    return [_update(acp_session_id, {"sessionUpdate": chunk, "content": _text(body)})]


def ending(item: TranscriptItem) -> str:
    """How a run ended, in words.

    The fold sets `body` only for a clean `finish_session`, carrying everything
    else in `ok`/`name`/`detail`. Reading `body` alone made a provider error, a
    budget stop and an iteration cap render as SILENCE -- an editor watching a
    run that simply stops -- and made a finish over a red gate look identical
    to a green one.

    The words are the status vocabulary every other surface uses: "passed"
    only for all-gates-green, otherwise the end reason's own label --
    "finished" is a deliberate finish that verified nothing (a gateless run),
    never a failure verdict like "did not pass" implied.
    """
    word = "passed" if item.ok else (item.name or "ended")
    parts = [f"Session {word}"]
    if item.detail:
        parts.append(f"- {item.detail}")
    ending = " ".join(parts)
    return f"{item.body}\n\n{ending}" if item.body.strip() else ending


def message_update(acp_session_id: str, text: str) -> dict[str, Any]:
    """One line of agent6's own prose, as a `session/update`.

    What the harness says when there is no run to say it: a cancel for a
    session that does not exist, a run that died before it had a journal. Both
    otherwise wrote zero bytes, and an editor cannot render silence. Marked as
    agent6's own, because the model did not say it.
    """
    return _update(
        acp_session_id,
        {"sessionUpdate": "agent_message_chunk", "content": _text(f"[agent6] {text}")},
    )


def _update(acp_session_id: str, update: dict[str, Any]) -> dict[str, Any]:
    return {
        "jsonrpc": "2.0",
        "method": "session/update",
        "params": {"sessionId": acp_session_id, "update": update},
    }


def printable(text: str) -> str:
    """Model-authored text, with control characters dropped.

    Every string this front-end puts on the wire that the MODEL had a hand in
    goes through here. Applying it only to ContentBlocks left two ways past
    it: a tool call's `title` (the model's own argv, via `salient_arg`) and a
    permission request's title and option names -- the latter being the one
    surface an operator MUST read before granting a command.
    """
    return "".join(c for c in text if c.isprintable() or c in "\n\t")


def _text(text: str) -> dict[str, Any]:
    """A ContentBlock, with control characters dropped.

    Every string this module puts on the wire goes through here, and most of
    them are model-authored. The fold scrubs its own previews and deltas
    (viewmodel.transcript.scrub_terminal_controls), but the renderer here is a
    THIRD PARTY: agent6 does not get to assume it treats an escape as inert,
    so this layer scrubs everything it emits regardless. `isprintable` is
    false for every C0/C1 control, so a sequence loses its ESC and becomes
    the literal text it was pretending not to be.
    """
    return {"type": "text", "text": printable(text)}


def _tool_status(item: TranscriptItem) -> str:
    """ACP's status for the fold's item: a call in flight is `in_progress`,
    or `pending` while it waits on an approval or an ask_user answer (the
    fold's mark in `detail`); a settled one carries its verdict."""
    if item.ok is None:
        return "pending" if item.detail else "in_progress"
    return "completed" if item.ok else "failed"


def _tool_content(item: TranscriptItem) -> list[dict[str, Any]]:
    """What the tool produced, in ACP's TAGGED shape.

    `ToolCallContent` is a discriminated union (`content` | `diff` |
    `terminal`), not a bare ContentBlock array. Sending the bare array made a
    strict client reject the whole notification, so the `completed`/`failed`
    it carried never arrived and the call announced a line earlier stayed
    `pending` for the rest of the session.

    `tail` is the failure's actual output -- a red gate's test log, a command's
    stderr. The fold fills it for exactly this, and dropping it left an editor
    showing "failed" with no reason.
    """
    body = "\n".join(part for part in (item.detail, item.tail) if part)
    return [{"type": "content", "content": _text(body)}] if body else []


def wire_call_id(session_id: str, turn: int, within_leg: str) -> str:
    """One tool call's id on the wire, `<run>:<turn>:<call>`: unique for the
    life of the ACP session, which is what an editor keys a call's lifecycle
    on. *within_leg* is the dispatcher's stamp, a per-leg counter that starts
    at 1 in every turn, so the run id and the turn join it."""
    return f"{session_id}:{turn}:{within_leg}" if session_id else within_leg


def _leg_call_id(item: TranscriptItem) -> str:
    """A fold item's stamped call id, which makes every call its own entity
    (two identical calls share a name+arg key); the name+arg fall-back is for
    historical events with no stamp."""
    return item.call_id or (f"{item.name}:{item.arg}" if item.arg else item.name)


def tool_call_id(item: TranscriptItem, session_id: str, turn: int) -> str:
    """`wire_call_id` for a fold item."""
    return wire_call_id(session_id, turn, _leg_call_id(item))


def _tool_call(item: TranscriptItem, wire_id: str) -> dict[str, Any]:
    # The model wrote `arg` (its own argv / path / pattern), so it is scrubbed
    # like any other model text; `content` next door already was.
    title = printable(f"{item.name} {item.arg}".strip())
    return {
        "toolCallId": wire_id,
        "title": title,
        # ACP's `kind` drives the editor's icon. agent6's own tool names are
        # the honest source; guessing a finer category from them would be a
        # second vocabulary to keep in sync.
        "kind": "other",
        "status": _tool_status(item),
    }
