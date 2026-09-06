# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Tests for OpenAIProvider SSE streaming.

Mirrors ``test_anthropic_streaming.py`` for the OpenAI Chat Completions
SSE shape. Validates that when a ``text_delta_callback`` is supplied
the provider:

* takes the streaming code path (``stream: true`` +
  ``stream_options.include_usage`` in the body);
* fans text deltas to the callback as they arrive;
* reassembles tool_call arguments spanning multiple chunks (indexed
  by the chunk-level ``index`` field, not the call id);
* reads ``usage`` from the terminal ``choices: []`` chunk;
* ignores SSE comment heartbeats (``:OPENROUTER PROCESSING``) and
  the ``data: [DONE]`` sentinel;
* synthesises a non-streaming-shape response so ``_parse_response``
  yields the same ProviderResponse it would for a normal call;
* surfaces HTTP errors as ``ProviderError``;
* never calls ``httpx2.stream`` when the callback is None.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from unittest import mock

import httpx2
import pytest

from agent6.budget import BudgetTracker
from agent6.providers import ProviderError
from agent6.providers.openai import OpenAIProvider


class _FakeStreamResponse:
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

    def __enter__(self) -> _FakeStreamResponse:
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def iter_lines(self) -> list[str]:
        return self._lines

    def read(self) -> bytes:
        return self._error_body.encode("utf-8")

    def iter_bytes(self) -> Iterator[bytes]:
        yield self._error_body.encode("utf-8")


def _chunk(data: dict[str, Any]) -> list[str]:
    return [f"data: {json.dumps(data)}", ""]


def _text_stream() -> list[str]:
    out: list[str] = []
    # Leading heartbeat (OpenRouter style).
    out += [":OPENROUTER PROCESSING", ""]
    out += _chunk(
        {
            "id": "c1",
            "choices": [
                {"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}
            ],
        }
    )
    out += _chunk({"choices": [{"index": 0, "delta": {"content": "hello"}, "finish_reason": None}]})
    out += [":OPENROUTER PROCESSING", ""]
    out += _chunk(
        {"choices": [{"index": 0, "delta": {"content": " world"}, "finish_reason": None}]}
    )
    out += _chunk({"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]})
    out += _chunk(
        {
            "choices": [],
            "usage": {
                "prompt_tokens": 42,
                "completion_tokens": 9,
                "prompt_tokens_details": {"cached_tokens": 7},
            },
        }
    )
    out += ["data: [DONE]", ""]
    return out


def _tool_stream() -> list[str]:
    """Tool call whose arguments arrive in 3 chunks; id+name on first."""
    out: list[str] = []
    out += _chunk(
        {
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_xyz",
                                "type": "function",
                                "function": {"name": "list_dir", "arguments": ""},
                            }
                        ],
                    },
                    "finish_reason": None,
                }
            ]
        }
    )
    out += _chunk(
        {
            "choices": [
                {
                    "index": 0,
                    "delta": {"tool_calls": [{"index": 0, "function": {"arguments": '{"pa'}}]},
                    "finish_reason": None,
                }
            ]
        }
    )
    out += _chunk(
        {
            "choices": [
                {
                    "index": 0,
                    "delta": {"tool_calls": [{"index": 0, "function": {"arguments": 'th": '}}]},
                    "finish_reason": None,
                }
            ]
        }
    )
    out += _chunk(
        {
            "choices": [
                {
                    "index": 0,
                    "delta": {"tool_calls": [{"index": 0, "function": {"arguments": '"."}'}}]},
                    "finish_reason": "tool_calls",
                }
            ]
        }
    )
    out += _chunk({"choices": [], "usage": {"prompt_tokens": 10, "completion_tokens": 4}})
    out += ["data: [DONE]", ""]
    return out


def test_streaming_calls_back_on_each_text_delta() -> None:
    provider = OpenAIProvider(api_key="sk-test", model="kimi")
    captured_bodies: list[dict[str, Any]] = []

    def fake_stream(method: str, url: str, **kwargs: Any) -> _FakeStreamResponse:
        assert method == "POST"
        captured_bodies.append(json.loads(kwargs["content"]))
        return _FakeStreamResponse(status_code=200, lines=_text_stream())

    pieces: list[str] = []
    with mock.patch("httpx2.stream", side_effect=fake_stream):
        resp = provider.call(
            system="sys",
            messages=[{"role": "user", "content": "x"}],
            text_delta_callback=pieces.append,
        )

    assert captured_bodies[0]["stream"] is True
    assert captured_bodies[0]["stream_options"] == {"include_usage": True}
    assert pieces == ["hello", " world"]
    assert resp.text == "hello world"
    assert resp.stop_reason == "stop"
    # input_tokens is fresh (non-cached) only; prompt_tokens=42 with
    # cached_tokens=7 means 35 fresh + 7 cached.
    assert resp.input_tokens == 35
    assert resp.output_tokens == 9
    assert resp.cache_read_tokens == 7


def test_streaming_reassembles_tool_call_arguments_across_chunks() -> None:
    provider = OpenAIProvider(api_key="sk-test", model="kimi")

    def fake_stream(method: str, url: str, **kwargs: Any) -> _FakeStreamResponse:
        return _FakeStreamResponse(status_code=200, lines=_tool_stream())

    with mock.patch("httpx2.stream", side_effect=fake_stream):
        resp = provider.call(
            system="sys",
            messages=[{"role": "user", "content": "x"}],
            text_delta_callback=lambda _p: None,
        )

    assert resp.text == ""
    assert resp.stop_reason == "tool_calls"
    assert len(resp.tool_uses) == 1
    tu = resp.tool_uses[0]
    assert tu["id"] == "call_xyz"
    assert tu["name"] == "list_dir"
    assert tu["input"] == {"path": "."}
    assert resp.input_tokens == 10
    assert resp.output_tokens == 4


def test_streaming_swallows_callback_exception() -> None:
    provider = OpenAIProvider(api_key="sk-test", model="kimi")

    def boom(_p: str) -> None:
        raise RuntimeError("ui exploded")

    def fake_stream(method: str, url: str, **kwargs: Any) -> _FakeStreamResponse:
        return _FakeStreamResponse(status_code=200, lines=_text_stream())

    with mock.patch("httpx2.stream", side_effect=fake_stream):
        resp = provider.call(
            system="sys",
            messages=[{"role": "user", "content": "x"}],
            text_delta_callback=boom,
        )

    assert resp.text == "hello world"


def test_streaming_http_error_raises_provider_error() -> None:
    provider = OpenAIProvider(api_key="sk-test", model="kimi")

    def fake_stream(method: str, url: str, **kwargs: Any) -> _FakeStreamResponse:
        return _FakeStreamResponse(status_code=429, lines=[], error_body='{"error": "rate limit"}')

    with (
        mock.patch("httpx2.stream", side_effect=fake_stream),
        pytest.raises(ProviderError, match="429"),
    ):
        provider.call(
            system="sys",
            messages=[{"role": "user", "content": "x"}],
            text_delta_callback=lambda _p: None,
        )


def test_streaming_httpx_transport_error_raises_provider_error() -> None:
    provider = OpenAIProvider(api_key="sk-test", model="kimi")

    def boom(method: str, url: str, **kwargs: Any) -> _FakeStreamResponse:
        raise httpx2.ReadTimeout("timed out")

    with (
        mock.patch("httpx2.stream", side_effect=boom),
        pytest.raises(ProviderError, match="HTTP error streaming from"),
    ):
        provider.call(
            system="sys",
            messages=[{"role": "user", "content": "x"}],
            text_delta_callback=lambda _p: None,
        )


def test_streaming_mid_stream_error_frame_raises_not_silent() -> None:
    """OpenRouter/OpenAI deliver an upstream error as a mid-stream `error` frame
    then end the stream. It must surface as a ProviderError (not be swallowed) AND
    carry the upstream status like the non-streaming 2xx-envelope path -- streaming
    is the default, so a permanent code delivered mid-stream would otherwise be
    retried every turn. A 502 stays retryable; insufficient_quota is permanent."""
    from agent6.workflows._provider_call import NON_RETRYABLE_HTTP_STATUSES

    provider = OpenAIProvider(api_key="sk-test", model="kimi")

    def _run(err: dict[str, Any]) -> ProviderError:
        lines = _chunk(
            {"choices": [{"index": 0, "delta": {"content": "partial"}, "finish_reason": None}]}
        ) + _chunk({"error": err})

        def fake_stream(method: str, url: str, **kwargs: Any) -> _FakeStreamResponse:
            return _FakeStreamResponse(status_code=200, lines=lines)

        with (
            mock.patch("httpx2.stream", side_effect=fake_stream),
            pytest.raises(ProviderError) as ei,
        ):
            provider.call(
                system="sys",
                messages=[{"role": "user", "content": "x"}],
                text_delta_callback=lambda _p: None,
            )
        return ei.value

    transient = _run({"code": 502, "message": "upstream gateway error"})
    assert "stream error: 502" in str(transient)
    assert transient.status_code == 502  # carried, not dropped to None
    assert transient.status_code not in NON_RETRYABLE_HTTP_STATUSES  # retryable

    permanent = _run({"code": "insufficient_quota", "message": "no credits"})
    assert permanent.status_code == 402  # permanent string code classified
    assert permanent.status_code in NON_RETRYABLE_HTTP_STATUSES  # not retried


def test_streaming_premature_end_without_done_or_finish_raises() -> None:
    """A stream that ends without `[DONE]` and without any `finish_reason` was
    cut off mid-generation; its partial content must not be returned as a
    finished turn."""
    provider = OpenAIProvider(api_key="sk-test", model="kimi")
    lines = _chunk(
        {"choices": [{"index": 0, "delta": {"content": "half a sentence"}, "finish_reason": None}]}
    )

    def fake_stream(method: str, url: str, **kwargs: Any) -> _FakeStreamResponse:
        return _FakeStreamResponse(status_code=200, lines=lines)

    with (
        mock.patch("httpx2.stream", side_effect=fake_stream),
        pytest.raises(ProviderError, match="ended prematurely"),
    ):
        provider.call(
            system="sys",
            messages=[{"role": "user", "content": "x"}],
            text_delta_callback=lambda _p: None,
        )


def test_streaming_finish_reason_without_done_is_complete() -> None:
    """A gateway that sends a real `finish_reason` but omits `[DONE]` is a
    completed turn, not a premature end."""
    provider = OpenAIProvider(api_key="sk-test", model="kimi")
    lines = _chunk(
        {"choices": [{"index": 0, "delta": {"content": "done"}, "finish_reason": "stop"}]}
    )

    def fake_stream(method: str, url: str, **kwargs: Any) -> _FakeStreamResponse:
        return _FakeStreamResponse(status_code=200, lines=lines)

    with mock.patch("httpx2.stream", side_effect=fake_stream):
        resp = provider.call(
            system="sys",
            messages=[{"role": "user", "content": "x"}],
            text_delta_callback=lambda _p: None,
        )
    assert resp.text == "done"
    assert resp.stop_reason == "stop"


def test_streaming_with_budget_requires_usage_trailer() -> None:
    """A completed stream ([DONE]) with no usage trailer still fails closed --
    but RETRYABLE: at the call site a permanently misconfigured gateway is
    indistinguishable from one mangled stream (a degenerate stream the
    gateway cut dropped the trailer live, killing a $1 run at $0.09), and
    the bounded retry lane converts the permanent case into at-most-N
    attempts while saving the transient one."""
    provider = OpenAIProvider(
        api_key="sk-test",
        model="kimi",
        budget=BudgetTracker(max_usd=-1, max_tokens_fallback=1, max_percent=-1),
    )
    lines = [
        *_chunk({"choices": [{"index": 0, "delta": {"content": "done"}, "finish_reason": "stop"}]}),
        "data: [DONE]",
    ]

    def fake_stream(method: str, url: str, **kwargs: Any) -> _FakeStreamResponse:
        return _FakeStreamResponse(status_code=200, lines=lines)

    with (
        mock.patch("httpx2.stream", side_effect=fake_stream),
        pytest.raises(ProviderError) as exc_info,
    ):
        provider.call(
            system="sys",
            messages=[{"role": "user", "content": "x"}],
            text_delta_callback=lambda _p: None,
        )
    assert exc_info.value.status_code is None
    assert "usage accounting" in str(exc_info.value)
    assert provider.budget is not None
    assert provider.budget.snapshot().per_model == {}


def test_streaming_cut_before_the_usage_trailer_is_retryable() -> None:
    """stream_options.include_usage is always set, so the usage chunk arrives
    AFTER finish_reason and before [DONE]. A connection dropped in that window
    is a truncated stream -- it was classified as a permanent 422 (the
    no-usage-accounting error), so the run died on a blip every other
    truncation retries through."""
    provider = OpenAIProvider(
        api_key="sk-test",
        model="kimi",
        budget=BudgetTracker(max_usd=-1, max_tokens_fallback=-1, max_percent=-1),
    )
    lines = _chunk(  # finish_reason, then the connection drops: no usage, no [DONE]
        {"choices": [{"index": 0, "delta": {"content": "done"}, "finish_reason": "stop"}]}
    )

    def fake_stream(method: str, url: str, **kwargs: Any) -> _FakeStreamResponse:
        return _FakeStreamResponse(status_code=200, lines=lines)

    with (
        mock.patch("httpx2.stream", side_effect=fake_stream),
        pytest.raises(ProviderError) as exc_info,
    ):
        provider.call(
            system="sys",
            messages=[{"role": "user", "content": "x"}],
            text_delta_callback=lambda _p: None,
        )
    assert exc_info.value.status_code != 422  # retryable, not the permanent shape
    assert "truncat" in str(exc_info.value).lower() or "cut off" in str(exc_info.value).lower()


def test_no_callback_does_not_stream() -> None:
    provider = OpenAIProvider(api_key="sk-test", model="kimi")

    def fake_post(*_a: Any, **kw: Any) -> httpx2.Response:
        return httpx2.Response(
            status_code=200,
            request=httpx2.Request("POST", "https://api.openai.com/v1/chat/completions"),
            content=json.dumps(
                {
                    "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1},
                }
            ).encode("utf-8"),
            headers={"content-type": "application/json"},
        )

    with (
        mock.patch("agent6.providers._transport.http_post", side_effect=fake_post) as post_mock,
        mock.patch("httpx2.stream") as stream_mock,
    ):
        resp = provider.call(
            system="sys",
            messages=[{"role": "user", "content": "x"}],
        )

    assert resp.text == "ok"
    assert post_mock.called
    assert not stream_mock.called


def test_streaming_idle_watchdog_kills_heartbeat_only_stream(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A stream that emits only ``:`` heartbeats must trip
    the idle watchdog. Without this guard, OpenRouter sessions where
    the upstream model wedges can pin the harness for 800+ seconds
    while heartbeats keep httpx2's read-timeout reset indefinitely.
    """
    import threading
    import time

    from agent6.providers import _stream as stream_mod

    # Only heartbeats, no data line ever: this is the prefill-wedge case, so the
    # FIRST-data timeout governs (the mid-stream idle timeout only applies after
    # tokens start).
    monkeypatch.setattr(stream_mod, "STREAM_FIRST_DATA_TIMEOUT_S", 0.3)
    monkeypatch.setattr(stream_mod, "STREAM_WATCHDOG_TICK_S", 0.05)

    provider = OpenAIProvider(api_key="sk-test", model="kimi")

    class _BlockingHeartbeatResponse:
        def __init__(self) -> None:
            self.status_code = 200
            self._closed = threading.Event()

        def __enter__(self) -> _BlockingHeartbeatResponse:
            return self

        def __exit__(self, *exc: object) -> None:
            return None

        def close(self) -> None:
            self._closed.set()

        def read(self) -> bytes:
            return b""

        def iter_lines(self):
            # Emit heartbeats every 50ms until the watchdog closes us.
            while not self._closed.is_set():
                yield ":OPENROUTER PROCESSING"
                yield ""
                # Block briefly; on close(), raise to mimic httpx2.
                if self._closed.wait(0.05):
                    raise httpx2.ReadError("connection closed by watchdog")
            raise httpx2.ReadError("connection closed by watchdog")

    def fake_stream(method: str, url: str, **kwargs: Any) -> _BlockingHeartbeatResponse:
        return _BlockingHeartbeatResponse()

    started = time.monotonic()
    with (
        mock.patch("httpx2.stream", side_effect=fake_stream),
        pytest.raises(ProviderError, match="idle for"),
    ):
        provider.call(
            system="sys",
            messages=[{"role": "user", "content": "x"}],
            text_delta_callback=lambda _p: None,
        )
    elapsed = time.monotonic() - started
    # Should fire well under 2s with a 0.3s threshold + 0.05s tick.
    assert elapsed < 2.0, f"watchdog took {elapsed:.2f}s"


def test_streaming_idle_watchdog_mid_stream_uses_the_short_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Once tokens have started, a stall trips the SHORT mid-stream timeout, not
    the long prefill budget -- the user's case (text streamed, then wedged). The
    prefill timeout is set long here to prove the short one is what fired."""
    import threading
    import time

    from agent6.providers import _stream as stream_mod

    monkeypatch.setattr(stream_mod, "STREAM_IDLE_TIMEOUT_S", 0.3)
    monkeypatch.setattr(stream_mod, "STREAM_FIRST_DATA_TIMEOUT_S", 30.0)
    monkeypatch.setattr(stream_mod, "STREAM_WATCHDOG_TICK_S", 0.05)

    provider = OpenAIProvider(api_key="sk-test", model="kimi")

    class _DataThenStallResponse:
        def __init__(self) -> None:
            self.status_code = 200
            self._closed = threading.Event()

        def __enter__(self) -> _DataThenStallResponse:
            return self

        def __exit__(self, *exc: object) -> None:
            return None

        def close(self) -> None:
            self._closed.set()

        def read(self) -> bytes:
            return b""

        def iter_lines(self):
            # One real data line (tokens started -> short timeout now applies),
            # then only heartbeats until the watchdog closes us.
            yield 'data: {"choices":[{"index":0,"delta":{"content":"hello"}}]}'
            yield ""
            while not self._closed.is_set():
                yield ":OPENROUTER PROCESSING"
                yield ""
                if self._closed.wait(0.05):
                    raise httpx2.ReadError("connection closed by watchdog")
            raise httpx2.ReadError("connection closed by watchdog")

    def fake_stream(method: str, url: str, **kwargs: Any) -> _DataThenStallResponse:
        return _DataThenStallResponse()

    started = time.monotonic()
    with (
        mock.patch("httpx2.stream", side_effect=fake_stream),
        pytest.raises(ProviderError, match="mid-stream"),
    ):
        provider.call(
            system="sys",
            messages=[{"role": "user", "content": "x"}],
            text_delta_callback=lambda _p: None,
        )
    # Fires on the 0.3s mid-stream timeout, not the 30s prefill budget.
    assert time.monotonic() - started < 3.0


def test_lenient_json_object_recovers_common_malformations() -> None:
    """Weak/open models emit args strict JSON rejects; the lenient re-parse
    recovers the safe cases (raw newline, trailing junk) so the tool just runs,
    and refuses the ambiguous ones so the _raw_arguments sentinel is kept."""
    from agent6.providers._openai_recovery import lenient_json_object as _lenient_json_object

    # Raw newline inside a string value (a multiline code param).
    assert _lenient_json_object('{"new_string": "a\nb"}') == {"new_string": "a\nb"}
    # Trailing junk after a valid object (a leaked closing tag / prose).
    assert _lenient_json_object('{"path": "a.py"} </invoke>') == {"path": "a.py"}
    # A bad regex escape (\d) is a hard JSON error -> keep the sentinel (None).
    assert _lenient_json_object(r'{"pattern": "\d+"}') is None
    # A scalar / array is not tool args -> None.
    assert _lenient_json_object("42") is None
    assert _lenient_json_object('["a"]') is None
    assert _lenient_json_object("") is None


def test_empty_role_delta_stays_in_the_prefill_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    """An empty role delta arrives immediately but is NOT real output, so it must
    not flip to the short mid-stream idle timeout: a model that emits the role
    delta then reasons silently gets the generous prefill budget, not a 45s kill
    (regression guard for the two-phase watchdog)."""
    import threading
    import time

    from agent6.providers import _stream as stream_mod

    monkeypatch.setattr(stream_mod, "STREAM_FIRST_DATA_TIMEOUT_S", 0.6)
    monkeypatch.setattr(stream_mod, "STREAM_IDLE_TIMEOUT_S", 0.2)
    monkeypatch.setattr(stream_mod, "STREAM_WATCHDOG_TICK_S", 0.05)
    provider = OpenAIProvider(api_key="sk-test", model="kimi")

    class _RoleThenSilent:
        def __init__(self) -> None:
            self.status_code = 200
            self._closed = threading.Event()

        def __enter__(self) -> _RoleThenSilent:
            return self

        def __exit__(self, *exc: object) -> None:
            return None

        def close(self) -> None:
            self._closed.set()

        def read(self) -> bytes:
            return b""

        def iter_lines(self):
            # Empty role delta (immediate), then only heartbeats: no real content.
            yield 'data: {"choices":[{"index":0,"delta":{"role":"assistant","content":""}}]}'
            yield ""
            while not self._closed.is_set():
                yield ":OPENROUTER PROCESSING"
                yield ""
                if self._closed.wait(0.03):
                    raise httpx2.ReadError("closed by watchdog")
            raise httpx2.ReadError("closed by watchdog")

    def fake_stream(method: str, url: str, **kwargs: Any) -> _RoleThenSilent:
        return _RoleThenSilent()

    started = time.monotonic()
    with (
        mock.patch("httpx2.stream", side_effect=fake_stream),
        pytest.raises(ProviderError, match="prefill"),
    ):
        provider.call(
            system="s",
            messages=[{"role": "user", "content": "x"}],
            text_delta_callback=lambda _p: None,
        )
    # It survived past the 0.2s mid-stream budget and fired on the 0.6s prefill
    # budget -> the empty role delta did not flip the phase.
    assert time.monotonic() - started >= 0.45


def test_a_stream_cut_before_usage_is_recorded_as_truncated(tmp_path: Path) -> None:
    """The retryable raise landed AFTER the 200 was written, so the transcript
    showed a clean successful call for an attempt that was actually cut and
    re-issued -- while the sibling truncation one branch up records status 0.
    An audit of a retried run could not see what happened."""
    from agent6.providers.types import TranscriptSink

    provider = OpenAIProvider(
        api_key="sk-test",
        model="kimi",
        budget=BudgetTracker(max_usd=-1, max_tokens_fallback=-1, max_percent=-1),
        transcript_sink=TranscriptSink(tmp_path),
    )
    lines = _chunk(  # finish_reason, then the connection drops: no usage, no [DONE]
        {"choices": [{"index": 0, "delta": {"content": "done"}, "finish_reason": "stop"}]}
    )

    def fake_stream(method: str, url: str, **kwargs: Any) -> _FakeStreamResponse:
        return _FakeStreamResponse(status_code=200, lines=lines)

    with (
        mock.patch("httpx2.stream", side_effect=fake_stream),
        pytest.raises(ProviderError),
    ):
        provider.call(
            system="sys",
            messages=[{"role": "user", "content": "x"}],
            text_delta_callback=lambda _p: None,
        )

    recorded = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(tmp_path.glob("*.json"))]
    assert recorded, "the cut attempt was not recorded at all"
    assert [r["response"]["status"] for r in recorded] == [0]  # never a clean 200
    assert "truncat" in str(recorded[0]["response"]["body"]).lower()
