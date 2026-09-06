# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Tests for OpenAIProvider tool-use translation.

Validates the Anthropic-shape <-> OpenAI Chat Completions translation
in both directions: outgoing messages (tool_use -> tool_calls,
tool_result -> role=tool) and incoming responses (tool_calls ->
tool_uses tuple in Anthropic shape).

Uses a stub `httpx2.post` so no network call is made; the test asserts
on the request body and synthesises an OpenAI-shape response.
"""

from __future__ import annotations

import json
from typing import Any, ClassVar

import httpx2
import pytest

from agent6.providers import OpenAIProvider, ToolDefinition
from agent6.providers._openai_messages import anthropic_to_openai_messages, tools_to_openai
from agent6.providers._openai_recovery import (
    coerce_text_tool_calls as _coerce_text_tool_calls,
)


class _FakeResponse:
    def __init__(self, *, status_code: int, payload: dict[str, Any]):
        self.status_code = status_code
        self.headers: dict[str, str] = {}
        self._payload = payload
        self.text = ""

    def json(self) -> dict[str, Any]:
        return self._payload


def _ok_response(message: dict[str, Any]) -> dict[str, Any]:
    """Build an OpenAI-shape response with the given assistant message."""
    return {
        "choices": [{"message": message, "finish_reason": "tool_calls"}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5},
    }


def test_translate_text_only_user_message() -> None:
    """The simple-text case: string content stays a string."""
    out = anthropic_to_openai_messages(
        "sys",
        [{"role": "user", "content": "hello"}],
    )
    assert out == [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "hello"},
    ]


def test_translate_assistant_tool_use_becomes_tool_calls() -> None:
    """Anthropic tool_use block in assistant content -> OpenAI tool_calls."""
    msgs = [
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "Let me read that file."},
                {
                    "type": "tool_use",
                    "id": "t1",
                    "name": "read_file",
                    "input": {"path": "foo.py"},
                },
            ],
        }
    ]
    out = anthropic_to_openai_messages("sys", msgs)
    assert out[0] == {"role": "system", "content": "sys"}
    assert out[1]["role"] == "assistant"
    assert out[1]["content"] == "Let me read that file."
    assert out[1]["tool_calls"] == [
        {
            "id": "t1",
            "type": "function",
            "function": {
                "name": "read_file",
                "arguments": json.dumps({"path": "foo.py"}),
            },
        }
    ]


def test_translate_user_tool_result_becomes_role_tool_message() -> None:
    """tool_result in user content -> separate role=tool message."""
    msgs = [
        {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "file content"}],
        }
    ]
    out = anthropic_to_openai_messages("sys", msgs)
    # system + role=tool message; no user message (no text).
    assert len(out) == 2
    assert out[1] == {"role": "tool", "tool_call_id": "t1", "content": "file content"}


def test_translate_user_text_plus_tool_result_emits_both() -> None:
    """Mixed content: tool_result MUST come first (OpenAI requires
    role=tool to immediately follow the assistant's tool_calls), then
    the user text as a follow-up turn.

    Emitted text first then tool_result, which both
    violated the OpenAI ordering rule AND caused harness-injected
    notices ([loop-guard], [harness], [review]) to arrive BEFORE the
    tool result they were commenting on. Weak models lost the causal
    link and ignored the notice entirely - observed live with Kimi K2.6
    looping on `read_file` 10x in a row despite three loop-guard
    notices being injected.
    """
    msgs = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "[loop-guard] stop re-reading"},
                {"type": "tool_result", "tool_use_id": "t1", "content": "data"},
            ],
        }
    ]
    out = anthropic_to_openai_messages("sys", msgs)
    # tool_result first, then the user text.
    assert out[1] == {"role": "tool", "tool_call_id": "t1", "content": "data"}
    assert out[2] == {"role": "user", "content": "[loop-guard] stop re-reading"}


def test_translate_loop_guard_notice_lands_after_tool_results() -> None:
    """regression: when the harness injects a [loop-guard] /
    [harness] / [review] notice into a user turn that also carries
    tool_results, the notice must land in a SEPARATE user message
    AFTER all the role=tool messages so weak models see it as a fresh
    instruction rather than something the tool said. Tests with
    multiple tool_results to confirm ordering."""
    msgs = [
        {
            "role": "assistant",
            "content": [
                {"type": "tool_use", "id": "t1", "name": "read_file", "input": {"path": "a"}},
                {"type": "tool_use", "id": "t2", "name": "read_file", "input": {"path": "b"}},
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "t1", "content": "data_a"},
                {"type": "tool_result", "tool_use_id": "t2", "content": "data_b"},
                {"type": "text", "text": "[loop-guard] you have looped 3x; pivot"},
            ],
        },
    ]
    out = anthropic_to_openai_messages("sys", msgs)
    # system, assistant, tool t1, tool t2, user notice
    assert len(out) == 5
    assert out[1]["role"] == "assistant"
    assert out[2] == {"role": "tool", "tool_call_id": "t1", "content": "data_a"}
    assert out[3] == {"role": "tool", "tool_call_id": "t2", "content": "data_b"}
    assert out[4]["role"] == "user"
    assert "[loop-guard]" in out[4]["content"]


def test_translate_tool_result_list_content_flattened() -> None:
    """Anthropic tool_result content can be a list of blocks - flatten to string."""
    msgs = [
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "t1",
                    "content": [
                        {"type": "text", "text": "first "},
                        {"type": "text", "text": "second"},
                    ],
                }
            ],
        }
    ]
    out = anthropic_to_openai_messages("sys", msgs)
    assert out[1]["content"] == "first second"


def test_tools_to_openai_translation() -> None:
    """ToolDefinition -> OpenAI function-tool entries."""
    tools = [
        ToolDefinition(
            name="read_file",
            description="Read a file",
            input_schema={"type": "object", "properties": {"path": {"type": "string"}}},
        ),
        ToolDefinition(
            name="grep",
            description="Search",
            input_schema={"type": "object", "properties": {"pattern": {"type": "string"}}},
        ),
    ]
    out = tools_to_openai(tools)
    assert out == [
        {
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Read a file",
                "parameters": {"type": "object", "properties": {"path": {"type": "string"}}},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "grep",
                "description": "Search",
                "parameters": {
                    "type": "object",
                    "properties": {"pattern": {"type": "string"}},
                },
            },
        },
    ]


def test_call_with_tools_translates_request_and_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """End-to-end: caller passes Anthropic-shape inputs, gets Anthropic-shape
    response back. Tools go out as OpenAI function-tools and come back as
    tool_uses tuple."""
    captured: dict[str, Any] = {}

    def fake_post(url: str, **kwargs: Any) -> _FakeResponse:
        captured["url"] = url
        captured["body"] = json.loads(kwargs["content"].decode("utf-8"))
        return _FakeResponse(
            status_code=200,
            payload=_ok_response(
                {
                    "role": "assistant",
                    "content": "Found it.",
                    "tool_calls": [
                        {
                            "id": "call_42",
                            "type": "function",
                            "function": {
                                "name": "read_file",
                                "arguments": '{"path": "src/foo.py"}',
                            },
                        }
                    ],
                }
            ),
        )

    monkeypatch.setattr("agent6.providers._transport.http_post", fake_post)
    provider = OpenAIProvider(api_key="k", model="gpt-x")
    tools = [
        ToolDefinition(
            name="read_file",
            description="Read a file",
            input_schema={"type": "object", "properties": {"path": {"type": "string"}}},
        )
    ]
    resp = provider.call(
        system="sys",
        messages=[{"role": "user", "content": "find the bug"}],
        tools=tools,
        max_tokens=100,
    )
    # Request body has tools as OpenAI function entries.
    assert captured["body"]["tools"][0]["type"] == "function"
    assert captured["body"]["tools"][0]["function"]["name"] == "read_file"
    # Response was translated to Anthropic-shape ProviderResponse.
    assert resp.text == "Found it."
    assert len(resp.tool_uses) == 1
    tu = resp.tool_uses[0]
    assert tu["id"] == "call_42"
    assert tu["name"] == "read_file"
    assert tu["input"] == {"path": "src/foo.py"}
    # raw.content mirrors Anthropic's response shape so worker_loop's
    # `assistant_blocks = resp.raw.get("content")` works uniformly.
    assert {"type": "text", "text": "Found it."} in resp.raw["content"]
    assert any(
        b.get("type") == "tool_use" and b.get("name") == "read_file" for b in resp.raw["content"]
    )


def test_response_with_malformed_tool_arguments_doesnt_crash(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When the model returns invalid JSON in tool arguments, surface as
    `_raw_arguments` so debugging is possible but don't blow up the
    parser."""

    def fake_post(url: str, **kwargs: Any) -> _FakeResponse:
        return _FakeResponse(
            status_code=200,
            payload=_ok_response(
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "read_file",
                                # Invalid JSON
                                "arguments": '{"path": "foo.py',
                            },
                        }
                    ],
                }
            ),
        )

    monkeypatch.setattr("agent6.providers._transport.http_post", fake_post)
    provider = OpenAIProvider(api_key="k", model="gpt-x")
    resp = provider.call(system="s", messages=[{"role": "user", "content": "u"}])
    assert resp.tool_uses[0]["input"] == {"_raw_arguments": '{"path": "foo.py'}


def test_huge_malformed_tool_arguments_are_capped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Finding C: a degenerate model (Kimi K2.6 was observed live)
    can emit a 30+ KB tool-arg payload of repeating escape sequences that
    exhausts the completion-token cap mid-string. Surfacing the entire
    raw blob in `_raw_arguments` lets that toxic content survive into the
    next tool-error round-trip and primes the same degeneration on the
    next turn. Cap the diagnostic at 500 chars + an origin marker."""
    huge = '{"edits": [{"kind":"replace","old_string":"' + ("\\n" * 15000)

    def fake_post(url: str, **kwargs: Any) -> _FakeResponse:
        return _FakeResponse(
            status_code=200,
            payload=_ok_response(
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_x",
                            "type": "function",
                            "function": {
                                "name": "apply_edit",
                                "arguments": huge,
                            },
                        }
                    ],
                }
            ),
        )

    monkeypatch.setattr("agent6.providers._transport.http_post", fake_post)
    provider = OpenAIProvider(api_key="k", model="gpt-x")
    resp = provider.call(system="s", messages=[{"role": "user", "content": "u"}])
    parsed = resp.tool_uses[0]["input"]
    assert "_raw_arguments" in parsed
    raw = parsed["_raw_arguments"]
    # Capped well below the original payload size.
    assert len(raw) < 700, f"expected ~500-char cap; got {len(raw)}"
    assert f"original was {len(huge)} chars" in raw
    # Cap kicks in by 500 chars; the original 15k `\n` escapes must not
    # survive verbatim.
    assert raw.count("\\n") < 300


def test_extended_thinking_silently_ignored(monkeypatch: pytest.MonkeyPatch) -> None:
    """Anthropic-shape extended_thinking param doesn't translate to OpenAI;
    silently drop so cross-provider workflow code doesn't have to branch."""
    captured: dict[str, Any] = {}

    def fake_post(url: str, **kwargs: Any) -> _FakeResponse:
        captured["body"] = json.loads(kwargs["content"].decode("utf-8"))
        return _FakeResponse(
            status_code=200,
            payload=_ok_response({"role": "assistant", "content": "ok"}),
        )

    monkeypatch.setattr("agent6.providers._transport.http_post", fake_post)
    provider = OpenAIProvider(api_key="k", model="gpt-x")
    provider.call(
        system="s",
        messages=[{"role": "user", "content": "u"}],
        extended_thinking={"type": "enabled", "budget_tokens": 1000},
    )
    assert "effort" not in captured["body"]
    assert "extended_thinking" not in captured["body"]


def test_no_tools_path_still_works(monkeypatch: pytest.MonkeyPatch) -> None:
    """Back-compat: when no tools are passed, behave as before."""
    captured: dict[str, Any] = {}

    def fake_post(url: str, **kwargs: Any) -> _FakeResponse:
        captured["body"] = json.loads(kwargs["content"].decode("utf-8"))
        return _FakeResponse(
            status_code=200,
            payload=_ok_response({"role": "assistant", "content": "hi"}),
        )

    monkeypatch.setattr("agent6.providers._transport.http_post", fake_post)
    provider = OpenAIProvider(api_key="k", model="gpt-x")
    resp = provider.call(system="s", messages=[{"role": "user", "content": "u"}])
    assert "tools" not in captured["body"]
    assert resp.text == "hi"
    assert resp.tool_uses == ()


def test_full_loop_message_roundtrip(monkeypatch: pytest.MonkeyPatch) -> None:
    """Simulates one turn of a worker_loop-style conversation: assistant
    emits a tool_use (which we get back as Anthropic shape), then the
    workflow appends a tool_result in user content, and the NEXT call's
    OpenAI request has the right role=tool message."""
    captured_bodies: list[dict[str, Any]] = []

    def fake_post(url: str, **kwargs: Any) -> _FakeResponse:
        captured_bodies.append(json.loads(kwargs["content"].decode("utf-8")))
        # Always reply with a finish tool_call so the test is deterministic.
        return _FakeResponse(
            status_code=200,
            payload=_ok_response(
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_done",
                            "type": "function",
                            "function": {
                                "name": "finish_step",
                                "arguments": '{"edit": {"edits": [], "notes": "ok"}}',
                            },
                        }
                    ],
                }
            ),
        )

    monkeypatch.setattr("agent6.providers._transport.http_post", fake_post)
    provider = OpenAIProvider(api_key="k", model="gpt-x")

    # First call: simple user prompt.
    provider.call(
        system="s",
        messages=[{"role": "user", "content": "read foo.py"}],
        tools=[
            ToolDefinition(
                name="read_file",
                description="x",
                input_schema={"type": "object"},
            )
        ],
    )

    # Second call: simulating the loop after one read_file round-trip.
    provider.call(
        system="s",
        messages=[
            {"role": "user", "content": "read foo.py"},
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "t1",
                        "name": "read_file",
                        "input": {"path": "foo.py"},
                    }
                ],
            },
            {
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "x = 1"}],
            },
        ],
        tools=[
            ToolDefinition(
                name="read_file",
                description="x",
                input_schema={"type": "object"},
            )
        ],
    )
    # Second request body has 4 messages: system, user-text, assistant
    # (with tool_calls), tool (role=tool with result).
    second_body = captured_bodies[1]
    msgs = second_body["messages"]
    assert msgs[0]["role"] == "system"
    assert msgs[1] == {"role": "user", "content": "read foo.py"}
    assert msgs[2]["role"] == "assistant"
    assert msgs[2]["tool_calls"][0]["function"]["name"] == "read_file"
    assert msgs[3] == {"role": "tool", "tool_call_id": "t1", "content": "x = 1"}


# --- Text-emitted tool-call recovery (small local models) ----------------
#
# Some Ollama / llama.cpp chat templates don't parse the model's tool
# call into native `tool_calls`; the call leaks into the assistant text.
# `_parse_response` recovers it ONLY when no native tool_calls are present
# AND the recovered name matches an offered tool. These tests pin that
# behaviour and guard against regressions for well-behaved models.

_READ_FILE_TOOL = ToolDefinition(
    name="read_file",
    description="Read a file",
    input_schema={"type": "object", "properties": {"path": {"type": "string"}}},
)


def _call_with_text_content(
    monkeypatch: pytest.MonkeyPatch,
    content: str,
    *,
    tools: list[ToolDefinition] | None = None,
) -> Any:
    def fake_post(url: str, **kwargs: Any) -> _FakeResponse:
        return _FakeResponse(
            status_code=200,
            payload={
                "choices": [
                    {"message": {"role": "assistant", "content": content}, "finish_reason": "stop"}
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5},
            },
        )

    monkeypatch.setattr("agent6.providers._transport.http_post", fake_post)
    provider = OpenAIProvider(api_key="k", model="qwen2.5-coder")
    return provider.call(
        system="s",
        messages=[{"role": "user", "content": "read foo"}],
        tools=[_READ_FILE_TOOL] if tools is None else tools,
    )


def test_bare_json_tool_call_in_text_is_recovered(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A model that emits the call as a bare JSON object in `content`."""
    resp = _call_with_text_content(
        monkeypatch, '{"name": "read_file", "arguments": {"path": "calc.py"}}'
    )
    assert len(resp.tool_uses) == 1
    tu = resp.tool_uses[0]
    assert tu["name"] == "read_file"
    assert tu["input"] == {"path": "calc.py"}
    # The consumed JSON is stripped from the visible text so it isn't
    # echoed back into the model's context on the next turn.
    assert resp.text == ""


def test_hermes_tool_call_tags_are_recovered(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Qwen/Hermes `<tool_call>...</tool_call>` wrapper, with prose around."""
    content = (
        "Let me read the file.\n"
        '<tool_call>\n{"name": "read_file", "arguments": {"path": "calc.py"}}\n</tool_call>'
    )
    resp = _call_with_text_content(monkeypatch, content)
    assert len(resp.tool_uses) == 1
    assert resp.tool_uses[0]["input"] == {"path": "calc.py"}
    assert resp.text == "Let me read the file."


def test_fenced_json_tool_call_is_recovered(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A ```json fenced tool call."""
    content = '```json\n{"name": "read_file", "arguments": {"path": "calc.py"}}\n```'
    resp = _call_with_text_content(monkeypatch, content)
    assert len(resp.tool_uses) == 1
    assert resp.tool_uses[0]["input"] == {"path": "calc.py"}


def test_native_tool_calls_take_precedence_over_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When native tool_calls ARE present, the text fallback never fires —
    even if the content also contains tool-call-shaped JSON."""

    def fake_post(url: str, **kwargs: Any) -> _FakeResponse:
        return _FakeResponse(
            status_code=200,
            payload=_ok_response(
                {
                    "role": "assistant",
                    "content": '{"name": "read_file", "arguments": {"path": "ignore.py"}}',
                    "tool_calls": [
                        {
                            "id": "call_native",
                            "type": "function",
                            "function": {
                                "name": "read_file",
                                "arguments": '{"path": "real.py"}',
                            },
                        }
                    ],
                }
            ),
        )

    monkeypatch.setattr("agent6.providers._transport.http_post", fake_post)
    provider = OpenAIProvider(api_key="k", model="gpt-x")
    resp = provider.call(
        system="s",
        messages=[{"role": "user", "content": "u"}],
        tools=[_READ_FILE_TOOL],
    )
    assert len(resp.tool_uses) == 1
    assert resp.tool_uses[0]["id"] == "call_native"
    assert resp.tool_uses[0]["input"] == {"path": "real.py"}


def test_plain_json_answer_is_not_misread_as_tool_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A model legitimately answering with a JSON object whose `name`
    is NOT an offered tool must stay plain text — no false coercion."""
    content = '{"name": "Alice", "arguments": {"age": 30}}'
    resp = _call_with_text_content(monkeypatch, content)
    assert resp.tool_uses == ()
    assert resp.text == content


def test_text_coercion_disabled_when_no_tools_offered(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With no tools offered, tool-call-shaped JSON stays text."""
    content = '{"name": "read_file", "arguments": {"path": "calc.py"}}'
    resp = _call_with_text_content(monkeypatch, content, tools=[])
    assert resp.tool_uses == ()
    assert resp.text == content


# --- Qwen-Coder `<function=...><parameter=...>` XML tool form --------------
# Distinct from the Hermes `<tool_call>{json}</tool_call>` shape: the body is
# NOT JSON. Observed live with qwen3-coder-30b via OpenRouter (Novita), which
# returns finish_reason="stop" + this XML in `content` and an empty
# `tool_calls`, silently killing the run before this recovery existed.

_APPLY_EDIT_TOOL = ToolDefinition(
    name="apply_edit",
    description="Edit a file",
    input_schema={
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "edits": {"type": "array"},
            "offset": {"type": "integer"},
        },
    },
)


def test_qwen_function_xml_tool_call_is_recovered(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The exact qwen3-coder leakage: `<function=NAME><parameter=KEY>` plus a
    stray unmatched `</tool_call>`, with prose before it."""
    content = (
        "I'll start by reading the file.\n\n"
        "<function=read_file>\n<parameter=path>\ninterp.py\n</parameter>\n</function>\n</tool_call>"
    )
    resp = _call_with_text_content(monkeypatch, content)
    assert len(resp.tool_uses) == 1
    assert resp.tool_uses[0]["name"] == "read_file"
    assert resp.tool_uses[0]["input"] == {"path": "interp.py"}
    assert resp.text == "I'll start by reading the file."


def test_qwen_function_xml_allows_space_around_equals(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    content = "<function = read_file>\n<parameter = path>\ninterp.py\n</parameter>\n</function>"

    resp = _call_with_text_content(monkeypatch, content)

    assert resp.tool_uses[0]["name"] == "read_file"
    assert resp.tool_uses[0]["input"] == {"path": "interp.py"}
    assert resp.text == ""


def test_qwen_function_xml_structured_param_is_typed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An `array`-typed parameter value is JSON-parsed per the tool schema."""
    edits = '[{"kind": "replace", "old_string": "a", "new_string": "b"}]'
    content = (
        "<function=apply_edit>\n<parameter=path>\nf.py\n</parameter>\n"
        f"<parameter=edits>\n{edits}\n</parameter>\n</function>"
    )
    resp = _call_with_text_content(monkeypatch, content, tools=[_APPLY_EDIT_TOOL])
    assert resp.tool_uses[0]["input"]["path"] == "f.py"
    assert resp.tool_uses[0]["input"]["edits"] == [
        {"kind": "replace", "old_string": "a", "new_string": "b"}
    ]


def test_qwen_function_xml_does_not_turn_an_invalid_boolean_false() -> None:
    text = "<function=toggle><parameter=enabled>maybe</parameter></function>"
    schemas = {"toggle": {"properties": {"enabled": {"type": "boolean"}}}}

    calls, _ = _coerce_text_tool_calls(text, frozenset({"toggle"}), schemas)

    assert calls[0]["input"]["enabled"] == "maybe"


def test_qwen_function_xml_unclosed_params_keep_all(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression: with `</parameter>` closers MISSING (truncation), each param's
    body must stop at the NEXT `<parameter=` via a lookahead, not consume it --
    else the following param is silently dropped (here `edits`), making apply_edit
    fail pydantic validation and wasting a turn. Open-weight models emit this."""
    edits = '[{"kind": "replace", "old_string": "a", "new_string": "b"}]'
    content = (  # no </parameter> closers at all
        f"<function=apply_edit>\n<parameter=path>\nf.py\n<parameter=edits>\n{edits}\n</function>"
    )
    resp = _call_with_text_content(monkeypatch, content, tools=[_APPLY_EDIT_TOOL])
    assert len(resp.tool_uses) == 1
    assert resp.tool_uses[0]["input"]["path"] == "f.py"
    assert resp.tool_uses[0]["input"]["edits"] == [
        {"kind": "replace", "old_string": "a", "new_string": "b"}
    ]


def test_qwen_function_xml_string_param_not_mangled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A string-typed code param whose value happens to be JSON-shaped stays a
    byte-exact string (schema type wins over JSON-parsing)."""
    edits = json.dumps([{"kind": "create", "old_string": "", "new_string": '{"a": 1}'}])
    content = (
        "<function=apply_edit>\n<parameter=path>\nf.py\n</parameter>\n"
        f"<parameter=edits>\n{edits}\n</parameter>\n</function>"
    )
    resp = _call_with_text_content(monkeypatch, content, tools=[_APPLY_EDIT_TOOL])
    assert resp.tool_uses[0]["input"]["edits"][0]["new_string"] == '{"a": 1}'


def test_qwen_function_xml_prose_mentioning_tool_is_not_a_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Prose that names a tool but has no `<function=...>` block stays text."""
    resp = _call_with_text_content(
        monkeypatch, "I will use read_file next, but first let me think."
    )
    assert resp.tool_uses == ()


# --- Gemini / Gemma ```tool_code Python-call blocks --------------------------


def test_gemma_tool_code_block_is_recovered(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The exact gemma-3 leakage: a ```tool_code fence holding a Python list of
    calls in `content`, with empty native tool_calls. Without recovery the loop
    sees no tool_use and silent-finishes."""
    content = "Okay, I'll read the spec.\n```tool_code\n[read_file(path='spec.md')]\n```"
    resp = _call_with_text_content(monkeypatch, content)
    assert len(resp.tool_uses) == 1
    assert resp.tool_uses[0]["name"] == "read_file"
    assert resp.tool_uses[0]["input"] == {"path": "spec.md"}
    assert resp.text == "Okay, I'll read the spec."


def test_gemma_tool_code_typed_kwargs_and_print_wrapper(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A print()-wrapped call with int kwargs: ast.literal_eval keeps the types,
    and the inner tool call is mined out of the wrapper."""
    content = "```tool_code\nprint(read_file(path='shapes.py', offset=10, limit=50))\n```"
    resp = _call_with_text_content(monkeypatch, content)
    assert resp.tool_uses[0]["input"] == {"path": "shapes.py", "offset": 10, "limit": 50}


def test_gemma_tool_code_prose_mentioning_tool_is_not_a_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No ```tool_code fence -> prose naming a tool stays text (no false call)."""
    resp = _call_with_text_content(monkeypatch, "Next I will read_file(path='x') to check.")
    assert resp.tool_uses == ()


def test_gemma_tool_code_preserves_source_order_mixed_depth(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A list mixing a print-wrapped call and a bare call must keep SOURCE order
    (ast.walk's breadth-first order would put the shallower bare call first and
    invert read-before-edit intent)."""
    content = "```tool_code\n[print(read_file(path='FIRST')), apply_edit(path='SECOND')]\n```"
    resp = _call_with_text_content(monkeypatch, content, tools=[_READ_FILE_TOOL, _APPLY_EDIT_TOOL])
    assert [c["name"] for c in resp.tool_uses] == ["read_file", "apply_edit"]
    assert resp.tool_uses[0]["input"]["path"] == "FIRST"


def test_gemma_tool_code_keeps_an_unrecovered_sibling_fence_visible() -> None:
    text = "```tool_code\n[read_file(path='a.py')]\n```\n```tool_code\nnot valid Python (\n```"

    calls, remaining = _coerce_text_tool_calls(text, frozenset({"read_file"}))

    assert calls == [{"name": "read_file", "input": {"path": "a.py"}}]
    assert "not valid Python" in remaining
    assert "read_file" not in remaining


# --- blank-name native tool_call (poisons strict backends with a 400) -------


def test_blank_name_native_tool_call_is_dropped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A native tool_call with an empty `function.name` (observed live with
    qwen3-coder-30b) is dropped; valid calls in the same turn survive."""

    def fake_post(url: str, **kwargs: Any) -> _FakeResponse:
        return _FakeResponse(
            status_code=200,
            payload={
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "ok",
                            "tool_calls": [
                                {
                                    "id": "c1",
                                    "type": "function",
                                    "function": {
                                        "name": "read_file",
                                        "arguments": '{"path": "a.py"}',
                                    },
                                },
                                {
                                    "id": "c2",
                                    "type": "function",
                                    "function": {"name": "", "arguments": "{}"},
                                },
                            ],
                        },
                        "finish_reason": "tool_calls",
                    }
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5},
            },
        )

    monkeypatch.setattr("agent6.providers._transport.http_post", fake_post)
    provider = OpenAIProvider(api_key="k", model="qwen3-coder-30b")
    resp = provider.call(
        system="s",
        messages=[{"role": "user", "content": "go"}],
        tools=[_READ_FILE_TOOL],
    )
    assert [tu["name"] for tu in resp.tool_uses] == ["read_file"]


def test_blank_name_tool_use_and_orphan_result_dropped_in_translation() -> None:
    """Serialization defense: a blank-name assistant tool_use already in
    history (e.g. a resumed snapshot) and its orphaned tool_result are both
    dropped so the request stays well-formed for strict backends."""
    from agent6.providers._openai_messages import anthropic_to_openai_messages

    history = [
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "ok"},
                {"type": "tool_use", "id": "c1", "name": "read_file", "input": {"path": "a.py"}},
                {"type": "tool_use", "id": "c2", "name": "", "input": {}},
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "c1", "content": "{}"},
                {
                    "type": "tool_result",
                    "tool_use_id": "c2",
                    "content": '{"error": "Unknown tool: "}',
                },
            ],
        },
    ]
    msgs = anthropic_to_openai_messages("sys", history)
    asst = next(m for m in msgs if m["role"] == "assistant")
    tool_msgs = [m for m in msgs if m["role"] == "tool"]
    assert [tc["function"]["name"] for tc in asst["tool_calls"]] == ["read_file"]
    assert [m["tool_call_id"] for m in tool_msgs] == ["c1"]


# --- review findings: translation, 400 adaptation, usage nulls, recovery -----


def test_translate_thinking_only_assistant_sends_empty_content() -> None:
    # A reasoning-starved turn (thinking block, no text, no tool_use) must not
    # become {"content": null} without tool_calls - strict backends 400 on it.
    out = anthropic_to_openai_messages(
        "sys",
        [
            {"role": "user", "content": "go"},
            {"role": "assistant", "content": [{"type": "effort", "effort": "hmm"}]},
            {"role": "user", "content": "continue"},
        ],
    )
    assistant = next(m for m in out if m["role"] == "assistant")
    assert assistant["content"] == ""
    assert "tool_calls" not in assistant


def test_max_completion_tokens_400_adapts_and_latches(monkeypatch: pytest.MonkeyPatch) -> None:
    bodies: list[dict[str, Any]] = []
    responses = [
        _FakeResponse(status_code=400, payload={}),
        _FakeResponse(
            status_code=200, payload=_ok_response({"role": "assistant", "content": "ok"})
        ),
        _FakeResponse(
            status_code=200, payload=_ok_response({"role": "assistant", "content": "ok"})
        ),
    ]
    responses[0].text = (
        "Unsupported parameter: 'max_tokens' is not supported with this model."
        " Use 'max_completion_tokens' instead."
    )

    def fake_post(url: str, **kwargs: Any) -> _FakeResponse:
        bodies.append(json.loads(kwargs["content"].decode("utf-8")))
        return responses[len(bodies) - 1]

    monkeypatch.setattr("agent6.providers._transport.http_post", fake_post)
    provider = OpenAIProvider(api_key="k", model="my-azure-gpt5", deployment="azure")
    resp = provider.call(system="s", messages=[{"role": "user", "content": "x"}], max_tokens=64)
    assert resp.text == "ok"
    assert "max_tokens" in bodies[0] and "max_completion_tokens" not in bodies[0]
    assert "max_completion_tokens" in bodies[1] and "max_tokens" not in bodies[1]
    # Latched: the next call builds the right body first time.
    provider.call(system="s", messages=[{"role": "user", "content": "x"}], max_tokens=64)
    assert "max_completion_tokens" in bodies[2] and "max_tokens" not in bodies[2]


def test_temperature_400_adapts_and_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    bodies: list[dict[str, Any]] = []
    responses = [
        _FakeResponse(status_code=400, payload={}),
        _FakeResponse(
            status_code=200, payload=_ok_response({"role": "assistant", "content": "ok"})
        ),
    ]
    responses[0].text = "temperature is not supported with this model"

    def fake_post(url: str, **kwargs: Any) -> _FakeResponse:
        bodies.append(json.loads(kwargs["content"].decode("utf-8")))
        return responses[len(bodies) - 1]

    monkeypatch.setattr("agent6.providers._transport.http_post", fake_post)
    provider = OpenAIProvider(api_key="k", model="my-deployment", deployment="azure")
    resp = provider.call(
        system="s", messages=[{"role": "user", "content": "x"}], max_tokens=64, temperature=0.0
    )
    assert resp.text == "ok"
    assert "temperature" in bodies[0]
    assert "temperature" not in bodies[1]


def test_null_usage_fields_do_not_crash(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = {
        "choices": [{"message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": None, "completion_tokens": None},
    }

    def fake_post(url: str, **kwargs: Any) -> _FakeResponse:
        return _FakeResponse(status_code=200, payload=payload)

    monkeypatch.setattr("agent6.providers._transport.http_post", fake_post)
    provider = OpenAIProvider(api_key="k", model="m")
    resp = provider.call(system="s", messages=[{"role": "user", "content": "x"}], max_tokens=64)
    assert resp.input_tokens == 0
    assert resp.output_tokens == 0


def test_fence_recovery_preserves_other_json_fences() -> None:
    text = (
        'Call:\n```json\n{"name": "read_file", "arguments": {"path": "a.py"}}\n```\n'
        'Reference config:\n```json\n{"key": "value"}\n```'
    )
    calls, remaining = _coerce_text_tool_calls(text, frozenset({"read_file"}))
    assert [c["name"] for c in calls] == ["read_file"]
    assert '"key": "value"' in remaining  # the second fence is content, not a call


def test_fence_recovery_finds_a_call_after_a_content_fence() -> None:
    text = (
        'Reference config:\n```json\n{"key": "value"}\n```\n'
        'Call:\n```json\n{"name": "read_file", "arguments": {"path": "a.py"}}\n```'
    )

    calls, remaining = _coerce_text_tool_calls(text, frozenset({"read_file"}))

    assert calls == [{"name": "read_file", "input": {"path": "a.py"}}]
    assert '"key": "value"' in remaining
    assert '"name": "read_file"' not in remaining


def test_tool_call_tag_recovery_keeps_malformed_tag_visible() -> None:
    text = (
        '<tool_call>{"name": "read_file", "arguments": {"path": "a.py"}}</tool_call>\n'
        "<tool_call>{not json}</tool_call>"
    )
    calls, remaining = _coerce_text_tool_calls(text, frozenset({"read_file"}))
    assert len(calls) == 1
    # The malformed second call stays visible so the model can see it failed.
    assert "{not json}" in remaining


def test_streaming_indexless_parallel_tool_calls_get_separate_slots(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lines = [
        "data: "
        + json.dumps(
            {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {
                                    "id": "a",
                                    "type": "function",
                                    "function": {
                                        "name": "read_file",
                                        "arguments": '{"path": "a.py"}',
                                    },
                                },
                                {
                                    "id": "b",
                                    "type": "function",
                                    "function": {
                                        "name": "read_file",
                                        "arguments": '{"path": "b.py"}',
                                    },
                                },
                            ]
                        },
                        "finish_reason": None,
                    }
                ]
            }
        ),
        'data: {"choices": [{"delta": {}, "finish_reason": "tool_calls"}],'
        ' "usage": {"prompt_tokens": 1, "completion_tokens": 1}}',
        "data: [DONE]",
    ]

    class _Stream:
        status_code = 200
        headers: ClassVar[dict[str, str]] = {}

        def __enter__(self) -> _Stream:
            return self

        def __exit__(self, *exc: object) -> None:
            return None

        def iter_lines(self) -> list[str]:
            return lines

        def read(self) -> bytes:
            return b""

        def close(self) -> None:
            return None

    def fake_stream(method: str, url: str, **kwargs: Any) -> _Stream:
        return _Stream()

    monkeypatch.setattr(httpx2, "stream", fake_stream)
    provider = OpenAIProvider(api_key="k", model="m")
    resp = provider.call(
        system="s",
        messages=[{"role": "user", "content": "x"}],
        max_tokens=64,
        text_delta_callback=lambda _: None,
    )
    assert len(resp.tool_uses) == 2
    assert {tu["id"] for tu in resp.tool_uses} == {"a", "b"}
    assert resp.tool_uses[0]["input"] == {"path": "a.py"}
    assert resp.tool_uses[1]["input"] == {"path": "b.py"}


def test_qwen_function_xml_keeps_unmatched_call_visible() -> None:
    """Same rule as the <tool_call> branch (which pins it in
    test_tool_call_tag_recovery_keeps_malformed_tag_visible): remove ONLY the
    calls that were recovered. Branch 0's blanket scrub deleted EVERY
    <function=...> block, so a hallucinated/misspelled second call vanished
    without a trace and the model assumed it happened."""
    text = (
        "<function=read_file><parameter=path>a.py</parameter></function>\n"
        "<function=write_notes><parameter=text>x</parameter></function>"
    )
    calls, remaining = _coerce_text_tool_calls(text, frozenset({"read_file"}))
    assert [c["name"] for c in calls] == ["read_file"]
    assert "<function=write_notes>" in remaining  # the unrecovered call stays visible
