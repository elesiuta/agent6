# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Shared request transport for the provider call paths.

Both providers execute one API call the same way: an attempt loop with
per-attempt auth headers (a `token_command` credential mints a short-lived
bearer, and a 401/403 refreshes it once and retries), one-shot 4xx body
adaptation (each provider decides which parameter-rejection 400s it can fix
by rewriting the body and latching), transcript recording, a retryable error
for a 2xx with a non-JSON body, usage metering, and the budget charge.
:class:`ProviderCall` owns that loop; request-body construction, header
composition, 400 adaptation, metering rules, and response parsing stay
per-provider via the hook fields.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import httpx2

from agent6.budget import BudgetTracker
from agent6.providers.types import (
    BearerCredential,
    ProviderError,
    ProviderResponse,
    TranscriptRecorder,
    parse_retry_after,
    scrub_secret_values,
)

# TCP+TLS handshake bound. A healthy endpoint connects in well under this;
# only a blackhole (dropped SYN, dead proxy) takes longer, and *timeout*'s
# 600s default would sit on it for ten minutes -- the stream watchdog cannot
# help there, it has no response to close until the connect returns.
CONNECT_TIMEOUT_S = 20.0


def granular_timeout(timeout: float) -> httpx2.Timeout:
    """*timeout* for read/write/pool, `CONNECT_TIMEOUT_S` for connect."""
    return httpx2.Timeout(timeout, connect=min(CONNECT_TIMEOUT_S, timeout))


# A full-window reply serializes to a few MiB; 64 MiB never clips a real
# response while stopping a pathological or hostile endpoint from being
# buffered whole into this process (fetch and MCP bound their reads the same
# way).
MAX_RESPONSE_BYTES = 64 * 1024 * 1024


def http_post(
    url: str, *, headers: dict[str, str], content: bytes, timeout: float
) -> httpx2.Response:
    """POST seam: tests stub this name, never `httpx2` globally.

    The body is read incrementally under `MAX_RESPONSE_BYTES`; an endpoint
    exceeding it raises a retryable `ProviderError` instead of an unbounded
    buffer."""
    with httpx2.stream(
        "POST", url, headers=headers, content=content, timeout=granular_timeout(timeout)
    ) as resp:
        body = bytearray()
        for chunk in resp.iter_bytes():
            body.extend(chunk)
            if len(body) > MAX_RESPONSE_BYTES:
                raise ProviderError(
                    f"provider response exceeded {MAX_RESPONSE_BYTES} bytes; refusing to buffer it"
                )
        # iter_bytes() decoded the body: the wire's representation headers
        # (content-encoding, content-length) no longer describe the content,
        # and carrying them over makes httpx2 run the decoder again over
        # plaintext.
        response_headers = [
            (k, v)
            for k, v in httpx2.Headers(resp.headers).multi_items()
            if k not in ("content-encoding", "content-length")
        ]
        return httpx2.Response(
            resp.status_code,
            headers=response_headers,
            content=bytes(body),
            request=resp.request,
        )


def _has_assistant_output(data: dict[str, Any]) -> bool:
    """Whether a 2xx body carries a REAL assistant response, so a top-level
    `error` key beside it is incidental rather than an error envelope. Covers
    both wire shapes: OpenAI `choices[].message` (content or tool_calls),
    Anthropic top-level `content`. A placeholder choice with null content is
    not output."""
    choices = data.get("choices")
    if isinstance(choices, list):
        for ch in choices:
            msg = ch.get("message") if isinstance(ch, dict) else None
            if isinstance(msg, dict) and (msg.get("content") or msg.get("tool_calls")):
                return True
    return isinstance(data.get("content"), list) and bool(data.get("content"))


# STRING error codes/types that are PERMANENT: retrying one wastes the whole
# budget (a quota/auth/not-found failure never clears mid-run). Map each to the
# terminal HTTP status NON_RETRYABLE_HTTP_STATUSES already treats as permanent.
# Transient strings (rate_limit_exceeded/_error, server_error, api_error,
# overloaded_error) are deliberately absent -> None -> retryable, the safe
# default. Covers both wire families: OpenAI's `code` and Anthropic's `type`.
_PERMANENT_ERROR_CODE_STATUS: dict[str, int] = {
    # OpenAI-family `code`
    "insufficient_quota": 402,
    "invalid_api_key": 401,
    "model_not_found": 404,
    # Anthropic `type`
    "invalid_request_error": 400,
    "authentication_error": 401,
    "permission_error": 403,
    "not_found_error": 404,
}


def envelope_status(err: object) -> int | None:
    """The upstream HTTP status carried in an error envelope, if it is a real
    4xx/5xx; else None (retryable). Threading it into `ProviderError.status_code`
    lets `NON_RETRYABLE_HTTP_STATUSES` classify a 402 as permanent while a
    429/5xx stays retryable.

    Reads a numeric `code` (int or all-digit string) directly, and maps a
    known-permanent STRING `code`/`type` (`insufficient_quota`, ...) to its
    terminal status -- a gateway that reports a quota/auth failure as a 200 body
    with a string code would otherwise be retried every turn; this path must
    classify the same failure set `require_metered` treats as permanent on
    real HTTP statuses."""
    if not isinstance(err, dict):
        return None
    code = err.get("code")
    if isinstance(code, bool):
        code = None
    if isinstance(code, int):
        return code if 400 <= code <= 599 else None
    if isinstance(code, str) and code.isdigit() and 400 <= int(code) <= 599:
        return int(code)
    for label in (code, err.get("type")):
        if isinstance(label, str) and label in _PERMANENT_ERROR_CODE_STATUS:
            return _PERMANENT_ERROR_CODE_STATUS[label]
    return None


def _envelope_detail(err: object) -> str:
    """A readable `code: message` for an error envelope, tolerating a bare
    string error and an empty object."""
    if isinstance(err, dict):
        label = err.get("code") or err.get("type") or "error"
        return f"{label}: {err.get('message') or err}"
    return str(err)


@dataclass(frozen=True, slots=True)
class ProviderCall:
    """One provider API call: the attempt loop around a built request body.

    `adapt_400` receives `(status, error_text, body)` and returns True
    after mutating `body` (and latching provider state) so the next attempt
    sends the adapted request; `adapt_attempts` reserves one extra attempt
    per adaptation the provider considers possible for this body. `stream`,
    when set, replaces the non-streaming POST and receives the per-attempt
    headers; errors it raises flow through the same adapt/refresh logic.
    """

    api_label: str  # "OpenAI" / "Anthropic"; leads API-error messages
    api_format: str  # "openai" / "anthropic"; names the wire format
    url: str
    body: dict[str, Any]
    timeout_s: float
    api_key: str
    credential: BearerCredential | None
    transcript_sink: TranscriptRecorder | None
    budget: BudgetTracker | None
    model: str
    build_headers: Callable[[str], dict[str, str]]
    adapt_400: Callable[[int | None, str, dict[str, Any]], bool]
    adapt_attempts: int
    require_metered: Callable[[dict[str, Any]], None]
    parse: Callable[[dict[str, Any]], ProviderResponse]
    stream: Callable[[dict[str, str]], ProviderResponse] | None = None

    def record(self, headers: dict[str, str], status: int, response: dict[str, Any] | str) -> None:
        """Write one transcript entry for this request (no-op without a sink)."""
        if self.transcript_sink is not None:
            self.transcript_sink.record(
                url=self.url,
                request_headers=headers,
                request_body=self.body,
                response_status=status,
                response_body=response,
            )

    def run(self) -> ProviderResponse:
        cred = self.credential
        # A credential reserves one refresh + retry for an expired bearer;
        # each possible one-shot body adaptation reserves one more attempt.
        max_attempts = (2 if cred is not None else 1) + self.adapt_attempts
        for attempt in range(max_attempts):
            token = cred.token() if cred is not None else self.api_key
            headers = self.build_headers(token)

            if self.stream is not None:
                try:
                    return self.stream(headers)
                except ProviderError as exc:
                    if attempt + 1 < max_attempts and self.adapt_400(
                        exc.status_code, str(exc), self.body
                    ):
                        continue
                    if (
                        cred is not None
                        and attempt + 1 < max_attempts
                        and exc.status_code in (401, 403)
                        and cred.invalidate(exc.status_code)
                    ):
                        continue
                    raise

            try:
                resp = http_post(
                    self.url,
                    headers=headers,
                    content=json.dumps(self.body).encode("utf-8"),
                    timeout=self.timeout_s,
                )
            except httpx2.HTTPError as exc:
                self.record(headers, 0, f"HTTPError: {exc}")
                raise ProviderError(
                    f"HTTP error calling {self.url} ({self.api_format} format): {exc}"
                ) from exc
            if cred is not None and attempt + 1 < max_attempts and resp.status_code in (401, 403):
                # Record BEFORE refreshing: this 401/403 hit the wire, and the
                # transcript contract is one file per round-trip (the streaming
                # path already records it; only this branch dropped it).
                self.record(headers, resp.status_code, resp.text[:8192])
                if cred.invalidate(resp.status_code):
                    continue
            if resp.status_code >= 400:
                self.record(headers, resp.status_code, resp.text[:8192])
                if attempt + 1 < max_attempts and self.adapt_400(
                    resp.status_code, resp.text, self.body
                ):
                    continue
                raise ProviderError(
                    f"{self.api_label} API error {resp.status_code}: "
                    f"{scrub_secret_values(resp.text, headers)[:500]}",
                    status_code=resp.status_code,
                    retry_after_s=parse_retry_after(resp.headers),
                )
            return self._decode_success(headers, resp)
        raise ProviderError(f"{self.api_label} auth retry exhausted")  # pragma: no cover

    def _decode_success(self, headers: dict[str, str], resp: httpx2.Response) -> ProviderResponse:
        """A 2xx body -> ProviderResponse: decode, record, meter, budget."""
        try:
            # Annotated Any: json() returns whatever the body holds; the
            # dict shape is PROVEN by the guard below, not assumed.
            data: Any = resp.json()
        except (json.JSONDecodeError, ValueError) as exc:
            # A 2xx with a non-JSON body (transient proxy/gateway glitch)
            # would otherwise raise a JSONDecodeError that the retry loop
            # doesn't catch (it only handles ProviderError), aborting the
            # run. Convert to a retryable ProviderError. Leaving
            # status_code unset marks it retryable.
            self.record(headers, resp.status_code, resp.text[:8192])
            raise ProviderError(
                f"non-JSON response from {self.api_label} "
                f"(status {resp.status_code}): {scrub_secret_values(resp.text, headers)[:500]}"
            ) from exc
        if not isinstance(data, dict):
            # A 2xx whose valid JSON is not an object (array/string from a
            # glitching gateway): every consumer downstream assumes a dict,
            # and the AttributeError it would raise bypasses the loop's
            # ProviderError-only retry. Same retryable conversion as the
            # non-JSON branch above.
            self.record(headers, resp.status_code, resp.text[:8192])
            raise ProviderError(
                f"{self.api_label} returned a non-object JSON body "
                f"(status {resp.status_code}): {scrub_secret_values(resp.text, headers)[:500]}"
            )
        self.record(headers, resp.status_code, data)
        # An in-band error envelope on a 2xx (OpenRouter/LiteLLM deliver an
        # upstream 5xx/429/4xx this way; Anthropic's error object has the same
        # top-level key). require_metered below finds no usage and blamed
        # agent6's own accounting ("no usage input tokens", 422), so a transient
        # upstream failure killed the run with no retry AND a permanent one
        # (402 "Insufficient credits") was RETRIED every turn because the status
        # was dropped. Key on "no usable assistant output" -- a placeholder
        # `choices` entry with null content is not output, a bare string `error`
        # is still an envelope -- and carry the upstream code as the status so
        # NON_RETRYABLE_HTTP_STATUSES makes 4xx permanent, 429/5xx retryable.
        err = data.get("error")
        if err and not _has_assistant_output(data):
            raise ProviderError(
                f"{self.api_label} error in 2xx body: "
                f"{scrub_secret_values(_envelope_detail(err), headers)}",
                status_code=envelope_status(err),
            )
        if self.budget is not None:
            self.require_metered(data)
        try:
            parsed = self.parse(data)
        except (AttributeError, KeyError, TypeError, ValueError, IndexError) as exc:
            # A malformed 2xx body (a flaky gateway's null/renamed fields) is a
            # retryable provider fault, never a raw traceback that bypasses
            # the loop's retry wrapper. The one parse seam both providers use.
            raise ProviderError(
                f"{self.api_label} 2xx body did not match the wire shape: {exc!r}"
            ) from exc
        if self.budget is not None:
            self.budget.record(
                model=self.model,
                input_tokens=parsed.input_tokens,
                output_tokens=parsed.output_tokens,
                cache_read_tokens=parsed.cache_read_tokens,
                cache_creation_tokens=parsed.cache_creation_tokens,
                cost_usd=parsed.cost_usd,
            )
        # The upstream's own failure signal, seen from OpenRouter as a 200 whose
        # choice carries `finish_reason: "error"`, a null content and nothing
        # else. Returned as a finished turn it spends a went-quiet nudge on an
        # error and abstains a review seat as if the model had answered; raised
        # here it retries, like a stream that ends without [DONE]. AFTER the
        # record: the provider billed those tokens either way.
        if (
            parsed.stop_reason.strip().lower() == "error"
            and not parsed.text.strip()
            and not parsed.tool_uses
        ):
            raise ProviderError(
                f"{self.api_label} response carries finish_reason='error' with no content:"
                " the upstream failed this completion"
            )
        return parsed
