# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Tests for AnthropicProvider SSE streaming.

Validate that when a ``text_delta_callback`` is supplied, the provider:

* takes the streaming code path (sets ``stream: true`` in the body);
* fans text deltas to the callback as they arrive;
* reassembles a ProviderResponse with shape identical to the
  non-streaming path (text + tool_uses + usage + stop_reason);
* records a transcript identical in shape to non-streaming;
* handles tool_use blocks whose ``input_json_delta`` chunks span
  multiple SSE events.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx2
import pytest

from agent6.budget import BudgetTracker
from agent6.providers import AnthropicProvider, ProviderError, TranscriptSink
from agent6.providers.token_command import CommandToken


class FakeStreamResponse:
    """Mimics the subset of httpx2 streaming Response we use."""

    def __init__(
        self,
        *,
        status_code: int,
        lines: list[str],
        error_body: str = "",
        headers: dict[str, str] | None = None,
    ) -> None:
        self.status_code = status_code
        self._lines = lines
        self._error_body = error_body
        # The real httpx2 Response always exposes headers; the error path reads
        # Retry-After from them.
        self.headers: dict[str, str] = headers or {}

    def __enter__(self) -> FakeStreamResponse:
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def iter_lines(self) -> list[str]:
        return self._lines

    def read(self) -> bytes:
        return self._error_body.encode("utf-8")


def _sse(events: list[tuple[str, dict[str, Any]]]) -> list[str]:
    """Turn (event_type, data) pairs into the raw line list httpx2
    .iter_lines() would yield. SSE frames are separated by a blank line."""
    out: list[str] = []
    for et, data in events:
        out.append(f"event: {et}")
        out.append(f"data: {json.dumps(data)}")
        out.append("")
    return out


def _basic_text_stream() -> list[str]:
    return _sse(
        [
            (
                "message_start",
                {
                    "type": "message_start",
                    "message": {
                        "id": "msg_1",
                        "role": "assistant",
                        "content": [],
                        "usage": {
                            "input_tokens": 42,
                            "cache_read_input_tokens": 7,
                            "cache_creation_input_tokens": 0,
                        },
                    },
                },
            ),
            (
                "content_block_start",
                {
                    "type": "content_block_start",
                    "index": 0,
                    "content_block": {"type": "text", "text": ""},
                },
            ),
            (
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "text_delta", "text": "hello"},
                },
            ),
            (
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "text_delta", "text": " world"},
                },
            ),
            (
                "content_block_stop",
                {"type": "content_block_stop", "index": 0},
            ),
            (
                "message_delta",
                {
                    "type": "message_delta",
                    "delta": {"stop_reason": "end_turn"},
                    "usage": {"output_tokens": 9},
                },
            ),
            ("message_stop", {"type": "message_stop"}),
        ]
    )


def _tool_use_stream() -> list[str]:
    """tool_use whose input arrives in 3 input_json_delta chunks."""
    return _sse(
        [
            (
                "message_start",
                {
                    "message": {
                        "usage": {
                            "input_tokens": 10,
                            "cache_read_input_tokens": 0,
                            "cache_creation_input_tokens": 0,
                        }
                    }
                },
            ),
            (
                "content_block_start",
                {
                    "index": 0,
                    "content_block": {
                        "type": "tool_use",
                        "id": "tu_xyz",
                        "name": "list_dir",
                        "input": {},
                    },
                },
            ),
            (
                "content_block_delta",
                {
                    "index": 0,
                    "delta": {"type": "input_json_delta", "partial_json": '{"pa'},
                },
            ),
            (
                "content_block_delta",
                {
                    "index": 0,
                    "delta": {"type": "input_json_delta", "partial_json": 'th": '},
                },
            ),
            (
                "content_block_delta",
                {
                    "index": 0,
                    "delta": {"type": "input_json_delta", "partial_json": '"."}'},
                },
            ),
            ("content_block_stop", {"index": 0}),
            (
                "message_delta",
                {"delta": {"stop_reason": "tool_use"}, "usage": {"output_tokens": 4}},
            ),
            ("message_stop", {}),
        ]
    )


def test_streaming_calls_back_on_each_text_delta(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    sink = TranscriptSink(tmp_path / "transcripts")
    provider = AnthropicProvider(
        api_key="sk-test", model="claude-test", prompt_caching=False, transcript_sink=sink
    )

    captured_bodies: list[dict[str, Any]] = []

    def fake_stream(method: str, url: str, **kwargs: Any) -> FakeStreamResponse:
        assert method == "POST"
        body = json.loads(kwargs["content"])
        captured_bodies.append(body)
        return FakeStreamResponse(status_code=200, lines=_basic_text_stream())

    monkeypatch.setattr(httpx2, "stream", fake_stream)

    pieces: list[str] = []
    resp = provider.call(
        system="sys",
        messages=[{"role": "user", "content": "x"}],
        text_delta_callback=pieces.append,
    )

    assert captured_bodies[0]["stream"] is True
    assert pieces == ["hello", " world"]
    assert resp.text == "hello world"
    assert resp.stop_reason == "end_turn"
    assert resp.input_tokens == 42
    assert resp.output_tokens == 9
    assert resp.cache_read_tokens == 7
    # raw must be shaped like a non-streaming response so downstream
    # assistant-block reconstruction in Workflow keeps working.
    assert resp.raw["content"] == [{"type": "text", "text": "hello world"}]
    assert resp.raw["stop_reason"] == "end_turn"

    # Transcript records the synthesised response, not raw SSE.
    files = list((tmp_path / "transcripts").glob("*.json"))
    assert len(files) == 1
    doc = json.loads(files[0].read_text(encoding="utf-8"))
    assert doc["response"]["status"] == 200
    assert doc["response"]["body"]["content"] == [{"type": "text", "text": "hello world"}]


def test_streaming_reassembles_tool_use_input_across_deltas(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    sink = TranscriptSink(tmp_path / "transcripts")
    provider = AnthropicProvider(
        api_key="sk-test", model="claude-test", prompt_caching=False, transcript_sink=sink
    )

    def fake_stream(method: str, url: str, **kwargs: Any) -> FakeStreamResponse:
        return FakeStreamResponse(status_code=200, lines=_tool_use_stream())

    monkeypatch.setattr(httpx2, "stream", fake_stream)

    pieces: list[str] = []
    resp = provider.call(
        system="sys",
        messages=[{"role": "user", "content": "x"}],
        text_delta_callback=pieces.append,
    )

    assert pieces == []  # tool_use has no text_delta callbacks
    assert resp.text == ""
    assert len(resp.tool_uses) == 1
    tu = resp.tool_uses[0]
    assert tu["name"] == "list_dir"
    assert tu["id"] == "tu_xyz"
    assert tu["input"] == {"path": "."}
    assert resp.stop_reason == "tool_use"


def test_streaming_callback_exception_does_not_break_stream(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A buggy TUI callback must NOT take the run down. The callback
    surface is cosmetic; the loop must complete and return the full
    response either way."""
    sink = TranscriptSink(tmp_path / "transcripts")
    provider = AnthropicProvider(
        api_key="sk-test", model="claude-test", prompt_caching=False, transcript_sink=sink
    )

    def fake_stream(method: str, url: str, **kwargs: Any) -> FakeStreamResponse:
        return FakeStreamResponse(status_code=200, lines=_basic_text_stream())

    monkeypatch.setattr(httpx2, "stream", fake_stream)

    def boom(_piece: str) -> None:
        raise RuntimeError("renderer exploded")

    resp = provider.call(
        system="sys",
        messages=[{"role": "user", "content": "x"}],
        text_delta_callback=boom,
    )

    assert resp.text == "hello world"
    assert resp.stop_reason == "end_turn"


def _truncated_text_stream() -> list[str]:
    """A stream cut off mid-message: message_start + a text delta, then a clean
    EOF with no content_block_stop and no message_stop."""
    return _sse(
        [
            (
                "message_start",
                {
                    "message": {
                        "usage": {
                            "input_tokens": 500,
                            "cache_read_input_tokens": 100,
                            "cache_creation_input_tokens": 0,
                        }
                    }
                },
            ),
            (
                "content_block_start",
                {"index": 0, "content_block": {"type": "text", "text": ""}},
            ),
            (
                "content_block_delta",
                {
                    "index": 0,
                    "delta": {"type": "text_delta", "text": "I will now edit the file"},
                },
            ),
        ]
    )


def test_streaming_premature_end_without_message_stop_raises(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A clean EOF before `message_stop` is a cut-off turn, not a completed one:
    it must raise a (retryable) ProviderError so the loop re-issues the request,
    not return the partial content and record input tokens as if it finished."""
    sink = TranscriptSink(tmp_path / "transcripts")
    provider = AnthropicProvider(
        api_key="sk-test", model="claude-test", prompt_caching=False, transcript_sink=sink
    )

    def fake_stream(method: str, url: str, **kwargs: Any) -> FakeStreamResponse:
        return FakeStreamResponse(status_code=200, lines=_truncated_text_stream())

    monkeypatch.setattr(httpx2, "stream", fake_stream)

    pieces: list[str] = []
    with pytest.raises(ProviderError, match="ended prematurely"):
        provider.call(
            system="sys",
            messages=[{"role": "user", "content": "x"}],
            text_delta_callback=pieces.append,
        )
    # The delta was fanned to the callback before the cut, but the call itself
    # must fail rather than report success.
    assert pieces == ["I will now edit the file"]


def test_streaming_error_event_raises_and_records_transcript(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A mid-stream `error` event must surface as a retryable ProviderError AND
    leave the error frame in the transcript for audit (parity with OpenAI)."""
    sink = TranscriptSink(tmp_path / "transcripts")
    provider = AnthropicProvider(
        api_key="sk-test", model="claude-test", prompt_caching=False, transcript_sink=sink
    )
    lines = _sse(
        [
            ("message_start", {"message": {"usage": {"input_tokens": 5}}}),
            (
                "error",
                {"type": "error", "error": {"type": "overloaded_error", "message": "busy"}},
            ),
        ]
    )

    def fake_stream(method: str, url: str, **kwargs: Any) -> FakeStreamResponse:
        return FakeStreamResponse(status_code=200, lines=lines)

    monkeypatch.setattr(httpx2, "stream", fake_stream)

    with pytest.raises(ProviderError, match="overloaded_error"):
        provider.call(
            system="sys",
            messages=[{"role": "user", "content": "x"}],
            text_delta_callback=lambda _p: None,
        )
    files = list((tmp_path / "transcripts").glob("*.json"))
    assert len(files) == 1
    doc = json.loads(files[0].read_text(encoding="utf-8"))
    assert doc["response"]["status"] == 0
    assert "overloaded_error" in doc["response"]["body"]


def test_streaming_propagates_http_error(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    sink = TranscriptSink(tmp_path / "transcripts")
    provider = AnthropicProvider(
        api_key="sk-test", model="claude-test", prompt_caching=False, transcript_sink=sink
    )

    def fake_stream(method: str, url: str, **kwargs: Any) -> FakeStreamResponse:
        return FakeStreamResponse(
            status_code=429,
            lines=[],
            error_body='{"error":{"type":"rate_limit"}}',
        )

    monkeypatch.setattr(httpx2, "stream", fake_stream)

    with pytest.raises(ProviderError):
        provider.call(
            system="sys",
            messages=[{"role": "user", "content": "x"}],
            text_delta_callback=lambda _p: None,
        )


def test_streaming_429_captures_retry_after(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    sink = TranscriptSink(tmp_path / "transcripts")
    provider = AnthropicProvider(
        api_key="sk-test", model="claude-test", prompt_caching=False, transcript_sink=sink
    )

    def fake_stream(method: str, url: str, **kwargs: Any) -> FakeStreamResponse:
        return FakeStreamResponse(
            status_code=429,
            lines=[],
            error_body='{"error":{"type":"rate_limit"}}',
            headers={"retry-after": "30"},
        )

    monkeypatch.setattr(httpx2, "stream", fake_stream)

    with pytest.raises(ProviderError) as exc_info:
        provider.call(
            system="sys",
            messages=[{"role": "user", "content": "x"}],
            text_delta_callback=lambda _p: None,
        )
    assert exc_info.value.status_code == 429
    assert exc_info.value.retry_after_s == 30.0  # threaded from the header


def test_non_streaming_path_unchanged_when_callback_is_none(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The default behaviour must NOT call httpx2.stream. Bench runs
    rely on the audited non-streaming code path."""
    sink = TranscriptSink(tmp_path / "transcripts")
    provider = AnthropicProvider(
        api_key="sk-test", model="claude-test", prompt_caching=False, transcript_sink=sink
    )

    stream_called = False

    def fake_stream(*_a: Any, **_kw: Any) -> FakeStreamResponse:
        nonlocal stream_called
        stream_called = True
        return FakeStreamResponse(status_code=200, lines=[])

    class _R:
        status_code = 200
        text = ""

        def json(self) -> dict[str, Any]:
            return {
                "content": [{"type": "text", "text": "ok"}],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 1, "output_tokens": 1},
            }

    def fake_post(*_a: Any, **_kw: Any) -> _R:
        return _R()

    monkeypatch.setattr(httpx2, "stream", fake_stream)
    monkeypatch.setattr("agent6.providers._transport.http_post", fake_post)

    resp = provider.call(system="sys", messages=[{"role": "user", "content": "x"}])
    assert resp.text == "ok"
    assert stream_called is False


def test_streaming_refreshes_token_command_on_401(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Regression: a Vertex-Anthropic (token_command) stream whose bearer expired
    # must refresh + retry once on a 401, not die. Requires the streaming 401
    # raise to carry status_code so the retry guard fires.
    counter = tmp_path / "n"
    script = (
        f'n=$(cat "{counter}" 2>/dev/null || echo 0); '
        f'n=$((n + 1)); printf %s "$n" > "{counter}"; printf "tok%s" "$n"'
    )
    base = (
        "https://us-east5-aiplatform.googleapis.com/v1/projects/p/locations/us-east5"
        "/publishers/anthropic/models"
    )
    provider = AnthropicProvider(
        api_key="",
        model="claude-x",
        base_url=base,
        deployment="vertex",
        auth_style="bearer",
        prompt_caching=False,
        credential=CommandToken(["sh", "-c", script], ttl_s=1000.0),
    )
    seen_auth: list[str | None] = []
    responses = [
        FakeStreamResponse(status_code=401, lines=[], error_body='{"error":"expired"}'),
        FakeStreamResponse(status_code=200, lines=_basic_text_stream()),
    ]

    def fake_stream(method: str, url: str, **kwargs: Any) -> FakeStreamResponse:
        seen_auth.append(kwargs["headers"].get("authorization"))
        return responses[len(seen_auth) - 1]

    monkeypatch.setattr(httpx2, "stream", fake_stream)
    resp = provider.call(
        system="sys",
        messages=[{"role": "user", "content": "x"}],
        text_delta_callback=lambda _p: None,
    )
    assert seen_auth == ["Bearer tok1", "Bearer tok2"]  # refreshed bearer on retry
    assert "hello" in resp.text


def test_streaming_with_budget_requires_usage_tokens(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    budget = BudgetTracker(max_usd=-1, max_tokens_fallback=1, max_percent=-1)
    provider = AnthropicProvider(api_key="sk-test", model="claude-test", budget=budget)
    lines = _sse(
        [
            (
                "message_start",
                {
                    "type": "message_start",
                    "message": {
                        "id": "msg_1",
                        "role": "assistant",
                        "content": [],
                    },
                },
            ),
            (
                "content_block_start",
                {
                    "type": "content_block_start",
                    "index": 0,
                    "content_block": {"type": "text", "text": ""},
                },
            ),
            (
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "text_delta", "text": "ok"},
                },
            ),
            ("content_block_stop", {"type": "content_block_stop", "index": 0}),
            ("message_delta", {"type": "message_delta", "delta": {"stop_reason": "end_turn"}}),
            ("message_stop", {"type": "message_stop"}),
        ]
    )

    def fake_stream(method: str, url: str, **kwargs: Any) -> FakeStreamResponse:
        return FakeStreamResponse(status_code=200, lines=lines)

    monkeypatch.setattr(httpx2, "stream", fake_stream)
    with pytest.raises(ProviderError) as exc_info:
        provider.call(
            system="sys",
            messages=[{"role": "user", "content": "x"}],
            text_delta_callback=lambda _p: None,
        )
    assert exc_info.value.status_code == 422
    assert budget.snapshot().per_model == {}


def test_foreign_opaque_blocks_never_reach_the_wire() -> None:
    """A cross-provider resume can carry another wire's opaque replay state
    (the ChatGPT reasoning item); sent verbatim it would 400 this API."""
    from agent6.providers.anthropic import drop_foreign_blocks

    messages = [
        {
            "role": "assistant",
            "content": [
                {
                    "type": "thinking",
                    "thinking": "plan",
                    "chatgpt_reasoning": {"type": "reasoning", "id": "rs_1"},
                },
                {"type": "text", "text": "hi"},
            ],
        },
        {"role": "user", "content": "next"},
    ]
    out = drop_foreign_blocks(messages)
    assert out[0]["content"] == [{"type": "text", "text": "hi"}]
    assert out[1] is messages[1]
    original = messages[0]["content"]
    assert isinstance(original, list)
    assert "chatgpt_reasoning" in original[0]  # caller list untouched
