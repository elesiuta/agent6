# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Every notification we emit, validated against the ACP protocol schema.

Hand-written shape assertions pin what the author BELIEVED the protocol says.
Three bugs this front-end shipped were written from the prose docs and caught
only by reading `schema.json`: a `ToolCallContent` sent as a bare ContentBlock
array rather than the tagged union (which makes a strict client drop the tool
outcome entirely), a status ACP does not define for an in-flight tool, and a
permission request that no client could answer.

The schema is vendored (`data/acp-schema.json`, from the protocol's published
release) so the suite stays offline and a protocol change arrives as a
deliberate update rather than a surprise.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from jsonschema import Draft202012Validator

from agent6.ui.acp.updates import message_update, updates_for
from agent6.viewmodel.transcript import TranscriptFold, TranscriptItem

_SCHEMA = json.loads(
    (Path(__file__).parent / "data" / "acp-schema.json").read_text(encoding="utf-8")
)
_RECORDED = Path(__file__).parent.parent / "unit" / "data" / "golden_session_logs.jsonl"


def _validator(definition: str) -> Draft202012Validator:
    """A validator for one `$defs` entry, resolving refs against the whole doc."""
    # Only the $defs, never the whole document: its top-level `anyOf` (every
    # message the protocol defines) applies ALONGSIDE a sibling $ref in Draft
    # 2020-12, so it would reject every payload.
    return Draft202012Validator({"$defs": _SCHEMA["$defs"], "$ref": f"#/$defs/{definition}"})


def _errors(validator: Draft202012Validator, payload: Any) -> list[str]:
    return [f"{list(e.absolute_path)}: {e.message}" for e in validator.iter_errors(payload)]


def _notifications() -> list[dict[str, Any]]:
    """What an editor receives for the recorded run the fold's golden test uses.

    The real journal, not hand-written events: a fabricated shape the engine
    never emits is how a surface validates green while rendering nothing.
    """
    fold = TranscriptFold()
    out: list[dict[str, Any]] = []
    announced: set[str] = set()
    for line in _RECORDED.read_text(encoding="utf-8").splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue  # the fixture's trailing malformed lines
        if isinstance(event, dict):
            for item in fold.feed(event):
                out.extend(
                    updates_for(
                        item,
                        acp_session_id="s",
                        session_id="brave-oak-AAAAAA",
                        announced=item.call_id in announced,
                    )
                )
                if item.kind == "tool":
                    announced.add(item.call_id)
    return out


def test_the_recorded_run_produces_only_valid_session_updates() -> None:
    notifications = _notifications()
    assert notifications, "the projection emitted nothing at all"
    validator = _validator("SessionNotification")
    for body in notifications:
        assert body["method"] == "session/update"
        assert not _errors(validator, body["params"]), (
            f"invalid session/update: {json.dumps(body['params'])}"
        )


@pytest.mark.parametrize(
    "item",
    [
        TranscriptItem(kind="tool", name="run_verify", arg="", ok=False, detail="exit 1"),
        TranscriptItem(kind="tool", name="grep", arg="x", ok=None),
        TranscriptItem(kind="tool", name="read_file", arg="a.py", ok=True, tail="4 lines"),
        TranscriptItem(kind="done", ok=False, name="budget", detail="stopped"),
        TranscriptItem(kind="commit", arg="abc1234", detail="+3 -1"),
        TranscriptItem(kind="thinking", body="hmm"),
        TranscriptItem(kind="operator", body="do the other thing"),
    ],
)
def test_each_fold_item_projects_to_a_valid_update(item: TranscriptItem) -> None:
    """Item kinds the recorded run does not happen to contain."""
    validator = _validator("SessionNotification")
    for body in updates_for(item, acp_session_id="s", session_id="r"):
        assert not _errors(validator, body["params"]), json.dumps(body["params"])


def test_the_harness_message_is_a_valid_update() -> None:
    body = message_update("s", "the run could not start: boom")
    assert not _errors(_validator("SessionNotification"), body["params"])


def test_the_handshake_answer_is_a_valid_initialize_response() -> None:
    import io

    from agent6.ui.acp.server import ACPServer

    server = ACPServer(stdin=io.BytesIO(), stdout=io.BytesIO())
    result = server._initialize({"clientCapabilities": {}}, None)  # pyright: ignore[reportPrivateUsage]
    assert not _errors(_validator("InitializeResponse"), result), json.dumps(result)


def test_a_permission_request_is_one_a_client_can_answer() -> None:
    """The params an editor is asked to render. Sending a shape it rejects
    means the approval never appears and the run waits out its timeout."""
    import io

    from agent6.ui.acp.runner import RunBridge
    from agent6.ui.acp.server import ACPServer
    from agent6.ui.acp.session import Session

    sent: list[dict[str, Any]] = []
    bridge = RunBridge(server=ACPServer(stdin=io.BytesIO(), stdout=io.BytesIO()))

    def _capture(_method: str, params: dict[str, Any], **_kw: object) -> dict[str, Any]:
        sent.append(params)
        return {}

    bridge.server.request = _capture  # pyright: ignore[reportAttributeAccessIssue]
    bridge.ask(
        Session(acp_id="s", cwd=Path("/x")), "Allow run_command: ls", ("allow", "deny"), True
    )
    bridge.ask(Session(acp_id="s", cwd=Path("/x")), "Theme?", ("dark", "light"), None)
    assert len(sent) == 2
    validator = _validator("RequestPermissionRequest")
    for params in sent:
        assert not _errors(validator, params), json.dumps(params)
