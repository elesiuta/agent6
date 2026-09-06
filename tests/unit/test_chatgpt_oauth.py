# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Tests for agent6.providers.chatgpt_oauth (PKCE, grants, credential)."""

from __future__ import annotations

import base64
import json
import time
from collections.abc import Generator
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, urlsplit

import pytest

from agent6.providers import chatgpt_oauth
from agent6.providers.chatgpt_oauth import (
    REDIRECT_URI,
    ChatGPTCredential,
    TokenGrant,
    account_id_of,
    authorize_url,
    exchange_code,
    jwt_claims,
    parse_callback,
    pkce_challenge,
    pkce_pair,
    refresh_grant,
    tokens_from_grant,
)
from agent6.providers.types import ProviderError
from agent6.secrets import OAuthTokens, load_oauth_tokens, save_oauth_tokens


@pytest.fixture
def gcfg(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "g"))
    return tmp_path / "g"


class _Resp:
    def __init__(self, status_code: int, body: object) -> None:
        self.status_code = status_code
        self._body = body
        self.text = body if isinstance(body, str) else json.dumps(body)

    def json(self) -> object:
        if isinstance(self._body, str):
            return json.loads(self._body)
        return self._body


def _jwt(claims: dict[str, Any]) -> str:
    payload = base64.urlsafe_b64encode(json.dumps(claims).encode()).rstrip(b"=").decode()
    return f"h.{payload}.s"


_AUTH_CLAIM = "https://api.openai.com/auth"


def test_pkce_challenge_matches_rfc7636_vector() -> None:
    """RFC 7636 appendix B: the S256 transform of the sample verifier."""
    assert (
        pkce_challenge("dBjftJeZ4CVP-mB92K27uhbUJU1p1r_wW1gFWFOEjXk")
        == "E9Melhoa2OwvFrEMTJguCHaoeK1t8URWbuGJSstw-cM"
    )
    verifier, challenge = pkce_pair()
    assert 43 <= len(verifier) <= 128 and "=" not in challenge
    assert challenge == pkce_challenge(verifier)


def test_authorize_url_carries_the_registered_params() -> None:
    url = authorize_url("https://auth.example/", "app_X", challenge="C", state="S")
    parts = urlsplit(url)
    assert (parts.hostname, parts.path) == ("auth.example", "/oauth/authorize")
    q = dict(parse_qsl(parts.query))
    assert q["response_type"] == "code" and q["client_id"] == "app_X"
    assert q["redirect_uri"] == REDIRECT_URI
    assert q["code_challenge"] == "C" and q["code_challenge_method"] == "S256"
    assert q["state"] == "S" and q["scope"] == "openid profile email offline_access"
    assert q["codex_cli_simplified_flow"] == "true" and q["originator"] == "agent6"


def test_parse_callback_accepts_url_or_query_and_checks_state() -> None:
    assert parse_callback(f"{REDIRECT_URI}?code=abc&state=S", state="S") == "abc"
    assert parse_callback("code=abc&state=S", state="S") == "abc"
    with pytest.raises(ValueError, match="state mismatch"):
        parse_callback(f"{REDIRECT_URI}?code=abc&state=OTHER", state="S")
    with pytest.raises(ValueError, match="no `code`"):
        parse_callback(f"{REDIRECT_URI}?state=S", state="S")
    with pytest.raises(ValueError, match="access_denied"):
        parse_callback(f"{REDIRECT_URI}?error=access_denied&state=S", state="S")


def test_account_id_prefers_access_token_claim() -> None:
    access = _jwt({_AUTH_CLAIM: {"chatgpt_account_id": "acct-access"}})
    id_tok = _jwt({_AUTH_CLAIM: {"chatgpt_account_id": "acct-id"}})
    assert account_id_of(TokenGrant(access, "r", 60.0, id_token=id_tok)) == "acct-access"
    assert account_id_of(TokenGrant("opaque-token", "r", 60.0, id_token=id_tok)) == "acct-id"
    assert account_id_of(TokenGrant("garbage", "r", 60.0)) == ""
    assert jwt_claims("not-a-jwt") == {}


def test_exchange_and_refresh_post_the_right_grants(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, dict[str, str]]] = []

    def fake_post(url: str, data: dict[str, str], timeout_s: float) -> _Resp:
        calls.append((url, data))
        return _Resp(
            200,
            {"access_token": "AT", "refresh_token": "RT", "expires_in": 1200, "id_token": "IT"},
        )

    monkeypatch.setattr("agent6.providers.chatgpt_oauth._post_form", fake_post)
    grant = exchange_code(
        "https://auth.example", "app_X", code="C0", verifier="V0", provider="chatgpt"
    )
    assert grant == TokenGrant("AT", "RT", 1200.0, id_token="IT")
    url, data = calls[0]
    assert url == "https://auth.example/oauth/token"
    assert data["grant_type"] == "authorization_code"
    assert data["code_verifier"] == "V0" and data["redirect_uri"] == REDIRECT_URI

    refresh_grant("https://auth.example", "app_X", "RT", provider="chatgpt")
    _, data = calls[1]
    assert data == {"grant_type": "refresh_token", "refresh_token": "RT", "client_id": "app_X"}


def test_dead_refresh_token_names_connect(monkeypatch: pytest.MonkeyPatch) -> None:
    """`refresh_token_expired` (and any 401) is permanent: the message names
    `agent6 connect chatgpt` and carries a 401 so the loop never retries it."""

    def dead(url: str, data: dict[str, str], timeout_s: float) -> _Resp:
        return _Resp(400, {"error": {"code": "refresh_token_expired"}})

    monkeypatch.setattr("agent6.providers.chatgpt_oauth._post_form", dead)
    with pytest.raises(ProviderError) as exc:
        refresh_grant("https://auth.example", "app_X", "RT", provider="chatgpt")
    assert "agent6 connect chatgpt" in str(exc.value) and exc.value.status_code == 401

    def down(url: str, data: dict[str, str], timeout_s: float) -> _Resp:
        return _Resp(503, "upstream down")

    monkeypatch.setattr("agent6.providers.chatgpt_oauth._post_form", down)
    with pytest.raises(ProviderError) as exc:
        refresh_grant("https://auth.example", "app_X", "RT", provider="chatgpt")
    assert exc.value.status_code == 503


def test_every_remedy_names_the_provider_it_diagnosed(
    gcfg: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A chatgpt-format provider under another name was told `agent6 connect
    chatgpt` by four of its five remedies, which signs in a different
    provider and leaves the broken one untouched."""

    def dead(url: str, data: dict[str, str], timeout_s: float) -> _Resp:
        return _Resp(400, {"error": {"code": "refresh_token_expired"}})

    monkeypatch.setattr("agent6.providers.chatgpt_oauth._post_form", dead)
    cred = ChatGPTCredential("codex", issuer="https://auth.example", client_id="app_X")
    with pytest.raises(ProviderError, match="agent6 connect codex"):
        cred.token()  # nothing stored
    save_oauth_tokens("codex", OAuthTokens("AT0", "RT1", time.time() + 3600, "acct-1"))
    assert cred.token() == "AT0"
    save_oauth_tokens("codex", OAuthTokens("AT1", "RT1", time.time() + 3600, "acct-2"))
    cred.invalidate()
    with pytest.raises(ProviderError, match="agent6 connect codex"):
        cred.token()  # the stored sign-in moved to another account
    with pytest.raises(ProviderError, match="agent6 connect codex"):
        refresh_grant("https://auth.example", "app_X", "RT", provider="codex")  # dead grant


def test_tokens_from_grant_keeps_previous_on_partial_refresh() -> None:
    prev = OAuthTokens("old-a", "old-r", 1.0, account_id="acct-1")
    fresh = tokens_from_grant(TokenGrant("new-a", "", 600.0), previous=prev)
    assert fresh.access_token == "new-a"
    assert fresh.refresh_token == "old-r" and fresh.account_id == "acct-1"
    assert fresh.expires_at > time.time() + 500


def test_credential_caches_refreshes_and_persists(
    gcfg: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    refreshes: list[str] = []

    def fake_post(url: str, data: dict[str, str], timeout_s: float) -> _Resp:
        refreshes.append(data["refresh_token"])
        return _Resp(
            200, {"access_token": f"AT{len(refreshes)}", "refresh_token": "RT2", "expires_in": 3600}
        )

    monkeypatch.setattr("agent6.providers.chatgpt_oauth._post_form", fake_post)
    cred = ChatGPTCredential("chatgpt", issuer="https://auth.example", client_id="app_X")
    with pytest.raises(ProviderError, match="agent6 connect chatgpt"):
        cred.token()

    save_oauth_tokens("chatgpt", OAuthTokens("AT0", "RT1", time.time() + 3600, "acct"))
    assert cred.token() == "AT0" and refreshes == []

    save_oauth_tokens("chatgpt", OAuthTokens("AT0", "RT1", time.time() + 10, "acct"))
    cred2 = ChatGPTCredential("chatgpt", issuer="https://auth.example", client_id="app_X")
    assert cred2.token() == "AT1" and refreshes == ["RT1"]
    stored = load_oauth_tokens("chatgpt")
    assert stored is not None and stored.refresh_token == "RT2" and stored.account_id == "acct"
    assert cred2.token() == "AT1" and len(refreshes) == 1  # cached until expiry
    assert cred2.account_id() == "acct"

    cred2.invalidate()
    assert cred2.token() == "AT2" and refreshes == ["RT1", "RT2"]


def test_credential_adopts_a_sibling_process_rotation(
    gcfg: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When another process already rotated the (single-use) refresh token,
    the credential adopts the stored tokens instead of replaying the old
    refresh token into a `refresh_token_reused` dead end."""

    def never(url: str, data: dict[str, str], timeout_s: float) -> _Resp:
        pytest.fail("refresh must not run")

    monkeypatch.setattr("agent6.providers.chatgpt_oauth._post_form", never)
    clock = {"now": 1000.0}
    fake_time = type("T", (), {"time": staticmethod(lambda: clock["now"])})
    monkeypatch.setattr("agent6.providers.chatgpt_oauth.time", fake_time)
    cred = ChatGPTCredential("chatgpt", issuer="https://auth.example", client_id="app_X")
    save_oauth_tokens("chatgpt", OAuthTokens("stale", "RT1", 5000.0, "acct"))
    assert cred.token() == "stale"
    # The cached copy ages out; a sibling has meanwhile stored a fresher grant.
    clock["now"] = 4800.0
    save_oauth_tokens("chatgpt", OAuthTokens("rotated", "RT2", 9000.0, "acct"))
    assert cred.token() == "rotated"


def test_device_auth_start_and_poll(monkeypatch: pytest.MonkeyPatch) -> None:
    """The device flow: usercode POST starts it (404 = disabled -> None),
    the poll treats 403/pending codes as waiting and slow_down as back-off,
    and success exchanges the issuer-minted code with the DEVICE redirect."""
    from agent6.providers.chatgpt_oauth import poll_device_auth, start_device_auth

    posts: list[tuple[str, dict[str, str]]] = []
    replies = [
        _Resp(200, {"device_auth_id": "da_1", "user_code": "AB-12", "interval": "5"}),
        _Resp(403, "pending"),
        _Resp(400, {"error": {"code": "deviceauth_authorization_pending"}}),
        _Resp(400, {"error": {"code": "slow_down"}}),
        _Resp(200, {"authorization_code": "AC", "code_verifier": "SERVER-V"}),
    ]

    def fake_json(url: str, data: dict[str, str], timeout_s: float) -> _Resp:
        posts.append((url, data))
        return replies.pop(0)

    exchanges: list[dict[str, str]] = []

    def fake_form(url: str, data: dict[str, str], timeout_s: float) -> _Resp:
        exchanges.append(data)
        return _Resp(200, {"access_token": "AT", "refresh_token": "RT", "expires_in": 60})

    monkeypatch.setattr("agent6.providers.chatgpt_oauth._post_json", fake_json)
    monkeypatch.setattr("agent6.providers.chatgpt_oauth._post_form", fake_form)
    naps: list[float] = []

    device = start_device_auth("https://auth.example", "app_X")
    assert device is not None and device.user_code == "AB-12" and device.interval_s == 5.0
    grant = poll_device_auth(
        "https://auth.example", "app_X", device, provider="chatgpt", sleep=naps.append
    )
    assert grant.access_token == "AT"
    assert posts[0] == (
        "https://auth.example/api/accounts/deviceauth/usercode",
        {"client_id": "app_X"},
    )
    assert posts[1][1] == {"device_auth_id": "da_1", "user_code": "AB-12"}
    assert naps == [5.0, 5.0, 10.0]  # two pendings, then slow_down backs off
    assert exchanges[0]["code_verifier"] == "SERVER-V"
    assert exchanges[0]["redirect_uri"] == "https://auth.example/deviceauth/callback"


def test_device_auth_disabled_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    from agent6.providers.chatgpt_oauth import start_device_auth

    def gone(url: str, data: dict[str, str], timeout_s: float) -> _Resp:
        return _Resp(404, "not enabled")

    monkeypatch.setattr("agent6.providers.chatgpt_oauth._post_json", gone)
    assert start_device_auth("https://auth.example", "app_X") is None


def test_refresh_error_scrubs_an_echoed_refresh_token(monkeypatch: pytest.MonkeyPatch) -> None:
    """A token-endpoint error that echoes the received credential (a proxy or
    a debug body) must not carry it into ProviderError text: those messages
    land in retry events and logs. The model wire scrubs this class via
    scrub_secret_values; the oauth wire scrubs its own in-flight values."""
    secret = "rt-veryverysecretvalue123"

    def echoing_post(url: str, data: dict[str, str], timeout_s: float) -> _Resp:
        return _Resp(500, {"error": "boom", "received": secret})

    monkeypatch.setattr(chatgpt_oauth, "_post_form", echoing_post)
    with pytest.raises(ProviderError) as ei:
        chatgpt_oauth.refresh_grant("https://auth.openai.com", "cid", secret, provider="chatgpt")
    assert secret not in str(ei.value)
    assert "<REDACTED>" in str(ei.value)


def test_revoke_warning_scrubs_the_token(monkeypatch: pytest.MonkeyPatch) -> None:
    tok = "at-echoedtokenvalue456789"

    def echoing_post(*args: object, **kwargs: object) -> _Resp:
        return _Resp(500, {"error": "boom", "received": tok})

    monkeypatch.setattr(chatgpt_oauth.httpx2, "post", echoing_post)
    tokens = chatgpt_oauth.OAuthTokens(access_token=tok, refresh_token="", expires_at=0.0)
    warn = chatgpt_oauth.revoke_tokens("https://auth.openai.com", "cid", tokens)
    assert warn is not None and tok not in warn and "<REDACTED>" in warn


def test_account_id_never_guesses_from_user_id() -> None:
    """`user_id` is the ChatGPT USER id, not an account id: a grant whose
    claims carry only user_id reads as account-less (the caller demands a
    re-connect) instead of sending a guessed `chatgpt-account-id` header."""
    tok = _jwt({_AUTH_CLAIM: {"user_id": "user-123"}})
    assert chatgpt_oauth.account_id_of(chatgpt_oauth.TokenGrant(tok, "", 100.0, tok)) == ""
    good = _jwt({_AUTH_CLAIM: {"chatgpt_account_id": "acct-9"}})
    assert chatgpt_oauth.account_id_of(chatgpt_oauth.TokenGrant(good, "", 100.0, good)) == "acct-9"


def test_credential_refuses_an_account_swap(gcfg: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The first read pins the account; a stored grant bound to a DIFFERENT
    account (a login from another process) refuses with the connect hint
    instead of riding under the old `chatgpt-account-id` header."""
    clock = {"now": 1000.0}
    fake_time = type("T", (), {"time": staticmethod(lambda: clock["now"])})
    monkeypatch.setattr("agent6.providers.chatgpt_oauth.time", fake_time)
    cred = ChatGPTCredential("chatgpt", issuer="https://auth.example", client_id="app_X")
    save_oauth_tokens("chatgpt", OAuthTokens("tokA", "RT1", 5000.0, "acct-A"))
    assert cred.token() == "tokA"
    save_oauth_tokens("chatgpt", OAuthTokens("tokB", "RT2", 9000.0, "acct-B"))
    cred.invalidate(401)
    with pytest.raises(ProviderError, match="different account"):
        cred.token()


def test_post_401_recovery_adopts_a_sibling_grant_first(
    gcfg: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """After a 401 the credential adopts a NEWER stored grant and retries
    with it; it only rotates the refresh token when no fresher grant exists.
    The old path refused adoption under force-refresh and burned the
    sibling's just-rotated (single-use) token again."""

    def never(url: str, data: dict[str, str], timeout_s: float) -> _Resp:
        pytest.fail("refresh must not run when a fresh sibling grant exists")

    monkeypatch.setattr("agent6.providers.chatgpt_oauth._post_form", never)
    clock = {"now": 1000.0}
    fake_time = type("T", (), {"time": staticmethod(lambda: clock["now"])})
    monkeypatch.setattr("agent6.providers.chatgpt_oauth.time", fake_time)
    cred = ChatGPTCredential("chatgpt", issuer="https://auth.example", client_id="app_X")
    save_oauth_tokens("chatgpt", OAuthTokens("revoked", "RT1", 5000.0, "acct"))
    assert cred.token() == "revoked"
    save_oauth_tokens("chatgpt", OAuthTokens("fresh", "RT2", 9000.0, "acct"))
    cred.invalidate(401)
    assert cred.token() == "fresh"


def test_403_does_not_arm_a_refresh(gcfg: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A 403 is permission or entitlement: neither a sibling grant nor a
    rotation changes what the account may do, so the credential keeps its
    cached bearer instead of burning a single-use refresh token."""

    def never(url: str, data: dict[str, str], timeout_s: float) -> _Resp:
        pytest.fail("a 403 must not trigger a refresh")

    monkeypatch.setattr("agent6.providers.chatgpt_oauth._post_form", never)
    clock = {"now": 1000.0}
    fake_time = type("T", (), {"time": staticmethod(lambda: clock["now"])})
    monkeypatch.setattr("agent6.providers.chatgpt_oauth.time", fake_time)
    cred = ChatGPTCredential("chatgpt", issuer="https://auth.example", client_id="app_X")
    save_oauth_tokens("chatgpt", OAuthTokens("tok", "RT1", 5000.0, "acct"))
    assert cred.token() == "tok"
    cred.invalidate(403)
    assert cred.token() == "tok"


def test_invalid_grant_is_a_dead_signin(gcfg: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The standard OAuth `{"error": "invalid_grant"}` (HTTP 400) means the
    grant is expired or revoked: the error names the repair (connect), not a
    generic HTTP 400 the retry policy would hammer."""

    def dead(url: str, data: dict[str, str], timeout_s: float) -> _Resp:
        return _Resp(400, {"error": "invalid_grant"})

    monkeypatch.setattr("agent6.providers.chatgpt_oauth._post_form", dead)
    clock = {"now": 6000.0}
    fake_time = type("T", (), {"time": staticmethod(lambda: clock["now"])})
    monkeypatch.setattr("agent6.providers.chatgpt_oauth.time", fake_time)
    cred = ChatGPTCredential("chatgpt", issuer="https://auth.example", client_id="app_X")
    save_oauth_tokens("chatgpt", OAuthTokens("old", "RT1", 5000.0, "acct"))
    with pytest.raises(ProviderError, match="no longer valid"):
        cred.token()


def test_reused_rotation_rereads_once(gcfg: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """`refresh_token_reused` with a fresher grant on disk (a process on
    ANOTHER host won the rotation and synced) adopts that grant instead of
    declaring the sign-in dead."""

    def reused(url: str, data: dict[str, str], timeout_s: float) -> _Resp:
        # A sibling's rotation lands between our read and the endpoint's answer.
        save_oauth_tokens("chatgpt", OAuthTokens("winner", "RT9", 9000.0, "acct"))
        return _Resp(401, {"error": {"code": "refresh_token_reused"}})

    monkeypatch.setattr("agent6.providers.chatgpt_oauth._post_form", reused)

    def no_sleep(_s: float) -> None:
        return None

    clock = {"now": 6000.0}
    fake_time = type(
        "T",
        (),
        {"time": staticmethod(lambda: clock["now"]), "sleep": staticmethod(no_sleep)},
    )
    monkeypatch.setattr("agent6.providers.chatgpt_oauth.time", fake_time)
    cred = ChatGPTCredential("chatgpt", issuer="https://auth.example", client_id="app_X")
    save_oauth_tokens("chatgpt", OAuthTokens("old", "RT1", 5000.0, "acct"))
    assert cred.token() == "winner"


def test_callback_state_checked_before_the_error_param() -> None:
    """A request the sign-in did not start gets nothing processed or
    reflected from its parameters, error path included: the state check
    outranks the error param."""
    with pytest.raises(ValueError, match="state mismatch"):
        parse_callback("error=x&error_description=<script>alert(1)</script>", state="S")


def test_callback_error_page_escapes_the_description() -> None:
    """The 400 page renders the refusal escaped: an attacker-supplied
    error_description must not run script on the localhost callback origin."""
    import urllib.error
    import urllib.request

    from agent6.ui.cli.connect import _CallbackServer  # pyright: ignore[reportPrivateUsage]

    srv = _CallbackServer("STATE", port=0)
    try:
        port = srv.port
        url = (
            f"http://127.0.0.1:{port}/auth/callback"
            "?state=STATE&error=x&error_description=<script>alert(1)</script>"
        )
        with pytest.raises(urllib.error.HTTPError) as ei:
            urllib.request.urlopen(url, timeout=5)
        body = ei.value.read().decode()
        assert "<script>" not in body
        assert "&lt;script&gt;" in body
        assert ei.value.headers.get("content-security-policy") == "default-src 'none'"
    finally:
        srv.close()


def test_unheld_refresh_lock_refuses_the_rotation(
    gcfg: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The refresh token is single-use: with the interprocess lock NOT held,
    the credential refuses (retryable) rather than risking a rotation that
    kills the sign-in for every process."""
    from contextlib import contextmanager

    @contextmanager
    def unheld(_path: Path) -> Generator[bool]:
        yield False

    monkeypatch.setattr("agent6.providers.chatgpt_oauth.locked_file", unheld)
    clock = {"now": 6000.0}
    fake_time = type("T", (), {"time": staticmethod(lambda: clock["now"])})
    monkeypatch.setattr("agent6.providers.chatgpt_oauth.time", fake_time)
    cred = ChatGPTCredential("chatgpt", issuer="https://auth.example", client_id="app_X")
    save_oauth_tokens("chatgpt", OAuthTokens("old", "RT1", 5000.0, "acct"))
    with pytest.raises(ProviderError, match="refresh lock"):
        cred.token()


def test_stored_account_must_match_the_tokens_own_claim(
    gcfg: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An entry written by an older parser can hold a USER id in the account
    field; the stored id is checked against the access token's own claim and
    refuses with the connect hint instead of sending a wrong header."""
    clock = {"now": 1000.0}
    fake_time = type("T", (), {"time": staticmethod(lambda: clock["now"])})
    monkeypatch.setattr("agent6.providers.chatgpt_oauth.time", fake_time)
    tok = _jwt({_AUTH_CLAIM: {"chatgpt_account_id": "acct-real"}})
    save_oauth_tokens("chatgpt", OAuthTokens(tok, "RT1", 5000.0, "user-legacy"))
    cred = ChatGPTCredential("chatgpt", issuer="https://auth.example", client_id="app_X")
    with pytest.raises(ProviderError, match="does not match its own token"):
        cred.token()


def test_403_reports_no_retry_worthwhile() -> None:
    cred = ChatGPTCredential("chatgpt", issuer="https://auth.example", client_id="app_X")
    assert cred.invalidate(403) is False
    assert cred.invalidate(401) is True


@pytest.mark.parametrize("expires_in", ["soon", int("1" + "0" * 400)])
def test_an_unusable_expires_in_is_a_provider_error(
    monkeypatch: pytest.MonkeyPatch, expires_in: object
) -> None:
    """A token body whose `expires_in` cannot be a float (text, or an integer
    too large for a double) fails like its two neighbours (a non-JSON body, a
    missing access_token): a ProviderError the sign-in prints at exit 2, not a
    bare ValueError or OverflowError that `_chatgpt_sign_in` never catches."""

    def odd(url: str, data: dict[str, str], timeout_s: float) -> _Resp:
        return _Resp(200, {"access_token": "AT", "refresh_token": "RT", "expires_in": expires_in})

    monkeypatch.setattr("agent6.providers.chatgpt_oauth._post_form", odd)
    with pytest.raises(ProviderError, match="expires_in"):
        exchange_code("https://auth.example", "app_X", code="C", verifier="V", provider="chatgpt")
    with pytest.raises(ProviderError, match="expires_in"):
        refresh_grant("https://auth.example", "app_X", "RT", provider="chatgpt")
