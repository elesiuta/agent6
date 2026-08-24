# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""ChatGPT sign-in: PKCE authorization-code OAuth and a refreshing credential.

`agent6 connect chatgpt` owns the interaction (browser, local callback,
paste fallback); this module owns the protocol: the authorize URL, the code
exchange, the refresh grant, and the :class:`ChatGPTCredential` the provider
holds per call. The issuer, the client id, and the redirect
(`localhost:1455`, pinned by the client registration) are constants, not
knobs: the ChatGPT profile dials only OpenAI's hosts (tests inject a
loopback issuer through the function parameters).

Token requests go to `<issuer>/oauth/token` from agent6's own process;
nothing a remote returns is executed. Tokens live in `secrets.toml` (0600)
and never reach transcripts (the recorder redacts the Authorization header)
or the jail.
"""

from __future__ import annotations

import base64
import hashlib
import json
import secrets as pysecrets
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit

import httpx2

from agent6.paths import secrets_path
from agent6.portable import locked_file
from agent6.providers.types import ProviderError
from agent6.secrets import OAuthTokens, load_oauth_tokens, save_oauth_tokens

CHATGPT_ISSUER = "https://auth.openai.com"
# The Codex CLI's public client registration, whose redirect is pinned to
# localhost:1455 below.
CHATGPT_CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"
REDIRECT_URI = "http://localhost:1455/auth/callback"
CALLBACK_PORT = 1455
# The device flow's fixed pieces: where the person enters the code, and the
# redirect the issuer pairs with device-issued authorization codes.
DEVICE_VERIFY_PATH = "/codex/device"
_DEVICE_USERCODE_PATH = "/api/accounts/deviceauth/usercode"
_DEVICE_TOKEN_PATH = "/api/accounts/deviceauth/token"  # noqa: S105 - a URL path, not a secret
_DEVICE_REDIRECT_PATH = "/deviceauth/callback"
_DEVICE_TIMEOUT_S = 15 * 60.0
OAUTH_SCOPE = "openid profile email offline_access"
# The namespaced JWT claim OpenAI tokens carry the ChatGPT identity under.
_CLAIMS_KEY = "https://api.openai.com/auth"
# Refresh this long before nominal expiry so a token never dies mid-call.
_REFRESH_SKEW_S = 300.0
_TOKEN_TIMEOUT_S = 30.0
# Token-endpoint error codes that mean the refresh token itself is dead
# (re-consent is the only repair); everything else is worth retrying.
_PERMANENT_REFRESH_CODES = frozenset(
    {"refresh_token_expired", "refresh_token_reused", "refresh_token_invalidated", "invalid_grant"}
)


@dataclass(frozen=True, slots=True)
class TokenGrant:
    """One `/oauth/token` response (exchange or refresh)."""

    access_token: str
    refresh_token: str
    expires_in: float
    id_token: str = ""


def pkce_challenge(verifier: str) -> str:
    """The RFC 7636 S256 challenge for a verifier."""
    digest = hashlib.sha256(verifier.encode("ascii")).digest()
    return base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")


def pkce_pair() -> tuple[str, str]:
    """A fresh RFC 7636 `(code_verifier, S256 code_challenge)` pair."""
    verifier = pysecrets.token_urlsafe(64)
    return verifier, pkce_challenge(verifier)


def authorize_url(issuer: str, client_id: str, *, challenge: str, state: str) -> str:
    """The browser URL that starts the sign-in.

    The extra `id_token_add_organizations` / `codex_cli_simplified_flow`
    params are what the issuer expects from this client registration; without
    them workspace accounts get an id_token with no account claim.
    """
    query = urlencode(
        [
            ("response_type", "code"),
            ("client_id", client_id),
            ("redirect_uri", REDIRECT_URI),
            ("scope", OAUTH_SCOPE),
            ("code_challenge", challenge),
            ("code_challenge_method", "S256"),
            ("state", state),
            ("id_token_add_organizations", "true"),
            ("codex_cli_simplified_flow", "true"),
            ("originator", "agent6"),
        ]
    )
    return f"{issuer.rstrip('/')}/oauth/authorize?{query}"


def parse_callback(pasted: str, *, state: str) -> str:
    """The authorization code carried by a callback URL (or bare query).

    Accepts the full `http://localhost:1455/auth/callback?...` line the
    browser lands on, or just its query string. Raises `ValueError` naming
    the problem: an `error` param, a missing code, or a state mismatch (a
    response agent6's own sign-in did not start).
    """
    text = pasted.strip()
    query = urlsplit(text).query if "?" in text else text
    params = dict(parse_qsl(query, keep_blank_values=True))
    # State first, for the error path too: a request agent6's own sign-in did
    # not start gets nothing reflected or processed from its parameters.
    if params.get("state", "") != state:
        raise ValueError("state mismatch: this callback is not from the sign-in agent6 started")
    if params.get("error"):
        detail = params.get("error_description") or params["error"]
        raise ValueError(f"sign-in was refused: {detail}")
    code = params.get("code", "")
    if not code:
        raise ValueError("no `code` parameter found; paste the full URL the browser landed on")
    return code


def _post_form(url: str, data: dict[str, str], timeout_s: float) -> httpx2.Response:
    """Token-endpoint POST seam: tests stub this name, never `httpx2` globally."""
    return httpx2.post(
        url,
        headers={"content-type": "application/x-www-form-urlencoded"},
        content=urlencode(data).encode("ascii"),
        timeout=timeout_s,
    )


def _post_json(url: str, data: dict[str, str], timeout_s: float) -> httpx2.Response:
    """Device-endpoint POST seam: tests stub this name, never `httpx2` globally."""
    return httpx2.post(url, json=data, timeout=timeout_s)


def _grant_from_response(resp: httpx2.Response, *, operation: str) -> TokenGrant:
    try:
        data: Any = resp.json()
    except ValueError as exc:
        raise ProviderError(f"ChatGPT token {operation} returned a non-JSON body") from exc
    access = data.get("access_token") if isinstance(data, dict) else None
    if not isinstance(access, str) or not access:
        raise ProviderError(f"ChatGPT token {operation} response carried no access_token")
    return TokenGrant(
        access_token=access,
        refresh_token=str(data.get("refresh_token") or ""),
        expires_in=float(data.get("expires_in") or 3600.0),
        id_token=str(data.get("id_token") or ""),
    )


def _scrub(text: str, secrets: tuple[str, ...]) -> str:
    """Replace the in-flight credential values wherever the issuer's response
    text echoes them (the same class `scrub_secret_values` covers on the model
    wire; these endpoints have their own). Raw and JSON-escaped spellings;
    values under 8 chars are ignored."""
    for value in secrets:
        if len(value) < 8:
            continue
        for spelling in {value, json.dumps(value)[1:-1]}:
            text = text.replace(spelling, "<REDACTED>")
    return text


def _token_error(
    resp: httpx2.Response, *, operation: str, secrets: tuple[str, ...] = ()
) -> ProviderError:
    """A classified error for a non-2xx token response. The body's error code
    decides permanence: a dead refresh token names the repair (`agent6
    connect chatgpt`); anything else keeps its status for the retry policy.
    *secrets* are the request's credential values, scrubbed from any echoed
    body text."""
    body = _scrub(resp.text[:2000], secrets)
    code = ""
    try:
        err = json.loads(body).get("error")
        code = str(err.get("code") if isinstance(err, dict) else err or "")
    except (ValueError, AttributeError):
        pass
    if resp.status_code == 401 or code in _PERMANENT_REFRESH_CODES:
        return ProviderError(
            f"ChatGPT sign-in is no longer valid ({code or f'HTTP {resp.status_code}'});"
            " run `agent6 connect chatgpt` to sign in again.",
            status_code=401,
        )
    return ProviderError(
        f"ChatGPT token {operation} failed: HTTP {resp.status_code}: {body[:300]}",
        status_code=resp.status_code,
    )


def exchange_code(
    issuer: str,
    client_id: str,
    *,
    code: str,
    verifier: str,
    redirect_uri: str = REDIRECT_URI,
    timeout_s: float = _TOKEN_TIMEOUT_S,
) -> TokenGrant:
    """Exchange an authorization code for the token grant. *redirect_uri*
    must match the flow that minted the code (the localhost callback, or the
    issuer's device-flow callback)."""
    url = f"{issuer.rstrip('/')}/oauth/token"
    try:
        resp = _post_form(
            url,
            {
                "grant_type": "authorization_code",
                "code": code,
                "redirect_uri": redirect_uri,
                "client_id": client_id,
                "code_verifier": verifier,
            },
            timeout_s,
        )
    except httpx2.HTTPError as exc:
        raise ProviderError(f"could not reach {url}: {exc}") from exc
    if resp.status_code >= 400:
        raise _token_error(resp, operation="exchange", secrets=(code, verifier))
    return _grant_from_response(resp, operation="exchange")


@dataclass(frozen=True, slots=True)
class DeviceAuth:
    """A started device-code sign-in: what the person types, how we poll."""

    device_auth_id: str
    user_code: str
    interval_s: float


def start_device_auth(issuer: str, client_id: str) -> DeviceAuth | None:
    """Begin the code-entry sign-in; None when the issuer has it disabled
    (a 404 -- the caller falls back to pasting the callback URL)."""
    url = f"{issuer.rstrip('/')}{_DEVICE_USERCODE_PATH}"
    try:
        resp = _post_json(url, {"client_id": client_id}, _TOKEN_TIMEOUT_S)
    except httpx2.HTTPError as exc:
        raise ProviderError(f"could not reach {url}: {exc}") from exc
    if resp.status_code == 404:
        return None
    if resp.status_code >= 400:
        raise ProviderError(f"device sign-in refused: HTTP {resp.status_code}: {resp.text[:200]}")
    try:
        data: Any = resp.json()
        return DeviceAuth(
            device_auth_id=str(data["device_auth_id"]),
            user_code=str(data["user_code"]),
            interval_s=max(5.0, float(data.get("interval") or 5.0)),
        )
    except (ValueError, KeyError, TypeError) as exc:
        raise ProviderError(f"device sign-in response was malformed: {exc!r}") from exc


def poll_device_auth(
    issuer: str,
    client_id: str,
    device: DeviceAuth,
    *,
    timeout_s: float = _DEVICE_TIMEOUT_S,
    sleep: Callable[[float], None] = time.sleep,
) -> TokenGrant:
    """Wait for the person to enter the code, then exchange the grant.

    The issuer answers pending as 403/404 or `deviceauth_authorization_pending`
    and hands back `{authorization_code, code_verifier}` once approved; the
    exchange then runs with the issuer's own verifier and device redirect.
    Raises ProviderError on refusal or when the code expires unentered.
    """
    url = f"{issuer.rstrip('/')}{_DEVICE_TOKEN_PATH}"
    deadline = time.monotonic() + timeout_s
    interval = device.interval_s
    while time.monotonic() < deadline:
        try:
            resp = _post_json(
                url,
                {"device_auth_id": device.device_auth_id, "user_code": device.user_code},
                _TOKEN_TIMEOUT_S,
            )
        except httpx2.HTTPError as exc:
            raise ProviderError(f"could not reach {url}: {exc}") from exc
        if resp.status_code < 400:
            try:
                data: Any = resp.json()
                code = str(data["authorization_code"])
                verifier = str(data["code_verifier"])
            except (ValueError, KeyError, TypeError) as exc:
                raise ProviderError(f"device sign-in response was malformed: {exc!r}") from exc
            return exchange_code(
                issuer,
                client_id,
                code=code,
                verifier=verifier,
                redirect_uri=f"{issuer.rstrip('/')}{_DEVICE_REDIRECT_PATH}",
            )
        detail = _error_code_of(resp)
        if resp.status_code in (403, 404) or detail == "deviceauth_authorization_pending":
            sleep(interval)
            continue
        if detail == "slow_down":
            interval += 5.0
            sleep(interval)
            continue
        raise ProviderError(
            "device sign-in failed: "
            f"HTTP {resp.status_code}: {_scrub(resp.text[:200], (device.device_auth_id,))}"
        )
    raise ProviderError("device sign-in expired before the code was entered; run connect again")


def _error_code_of(resp: httpx2.Response) -> str:
    try:
        err = resp.json().get("error")
    except (ValueError, AttributeError):
        return ""
    return str(err.get("code") if isinstance(err, dict) else err or "")


def refresh_grant(
    issuer: str,
    client_id: str,
    refresh_token: str,
    *,
    timeout_s: float = _TOKEN_TIMEOUT_S,
) -> TokenGrant:
    """Trade a refresh token for a fresh grant (tokens rotate)."""
    url = f"{issuer.rstrip('/')}/oauth/token"
    try:
        resp = _post_form(
            url,
            {
                "grant_type": "refresh_token",
                "refresh_token": refresh_token,
                "client_id": client_id,
            },
            timeout_s,
        )
    except httpx2.HTTPError as exc:
        raise ProviderError(f"could not reach {url}: {exc}") from exc
    if resp.status_code >= 400:
        raise _token_error(resp, operation="refresh", secrets=(refresh_token,))
    return _grant_from_response(resp, operation="refresh")


def revoke_tokens(issuer: str, client_id: str, tokens: OAuthTokens) -> str | None:
    """Best-effort revocation at `<issuer>/oauth/revoke` for a sign-out.

    Prefers the refresh token (killing the whole grant), falls back to the
    access token. Returns an error description instead of raising: the caller
    removes the local tokens either way, matching the endpoint's own
    semantics (revoking an already-dead token is a success).
    """
    token, hint = (
        (tokens.refresh_token, "refresh_token")
        if tokens.refresh_token
        else (tokens.access_token, "access_token")
    )
    body: dict[str, str] = {"token": token, "token_type_hint": hint}
    if hint == "refresh_token":
        body["client_id"] = client_id
    url = f"{issuer.rstrip('/')}/oauth/revoke"
    try:
        resp = httpx2.post(url, json=body, timeout=_TOKEN_TIMEOUT_S)
    except httpx2.HTTPError as exc:
        return f"could not reach {url}: {exc}"
    if resp.status_code >= 400:
        return f"HTTP {resp.status_code}: {_scrub(resp.text[:200], (token,))}"
    return None


def jwt_claims(token: str) -> dict[str, Any]:
    """The payload claims of a JWT, `{}` on any malformation.

    No signature check: agent6 is the OAuth client, not a verifier; the
    tokens arrive over the issuer's own TLS channel and are only read back
    for the account id.
    """
    parts = token.split(".")
    if len(parts) != 3:
        return {}
    payload = parts[1] + "=" * (-len(parts[1]) % 4)
    try:
        decoded: Any = json.loads(base64.urlsafe_b64decode(payload.encode("ascii")))
    except (ValueError, UnicodeDecodeError):
        return {}
    return decoded if isinstance(decoded, dict) else {}


def account_id_of(grant: TokenGrant) -> str:
    """The ChatGPT account id a grant is bound to, "" when absent.

    The claim rides in the access token and (for workspace accounts) the
    id_token; the backend requires it back as the `chatgpt-account-id`
    header.
    """
    for token in (grant.access_token, grant.id_token):
        auth = jwt_claims(token).get(_CLAIMS_KEY)
        if isinstance(auth, dict):
            # Only the real account claim: `user_id` is the ChatGPT USER id,
            # not an account id, and a guessed header is worse than none.
            account = auth.get("chatgpt_account_id")
            if isinstance(account, str) and account:
                return account
    return ""


def plan_type_of(grant: TokenGrant) -> str:
    """The ChatGPT plan the grant reports ("plus", "pro", ...), "" if absent."""
    for token in (grant.id_token, grant.access_token):
        auth = jwt_claims(token).get(_CLAIMS_KEY)
        if isinstance(auth, dict):
            plan = auth.get("chatgpt_plan_type")
            if isinstance(plan, str) and plan:
                return plan
    return ""


def tokens_from_grant(grant: TokenGrant, *, previous: OAuthTokens | None = None) -> OAuthTokens:
    """The storable tokens for a grant. A refresh response may omit the
    rotated refresh token or the identity claim; both carry over from
    *previous* rather than being erased."""
    account = account_id_of(grant) or (previous.account_id if previous else "")
    refresh = grant.refresh_token or (previous.refresh_token if previous else "")
    return OAuthTokens(
        access_token=grant.access_token,
        refresh_token=refresh,
        expires_at=time.time() + grant.expires_in,
        account_id=account,
    )


class ChatGPTCredential:
    """Cached, refreshing bearer over the stored ChatGPT OAuth tokens.

    The `token()` / `invalidate()` twin of :class:`CommandToken`, so the
    shared transport refreshes it once after an auth failure. Thread-safe in
    process; the reload-refresh-save transaction also holds an interprocess
    flock beside `secrets.toml`, because the refresh token is SINGLE-USE and
    rotates: two processes submitting the same one trips
    `refresh_token_reused` and kills the sign-in for both.

    The first account id read PINS the credential to that account: a stored
    or refreshed grant bound to a different account refuses with the connect
    hint (fail closed) rather than sending a bearer under a stale
    `chatgpt-account-id` header.

    After a 401 (`invalidate`), recovery adopts a NEWER stored grant first
    and only refreshes when none exists; on `refresh_token_reused` it
    re-reads once after a beat, in case a process on another host completed
    the rotation. A 403 never refreshes: that is entitlement or policy, and
    rotating a working token cannot fix it.
    """

    __slots__ = (
        "_account",
        "_client_id",
        "_force_refresh",
        "_issuer",
        "_last_returned",
        "_lock",
        "_provider",
        "_tokens",
    )

    def __init__(
        self,
        provider_name: str,
        *,
        issuer: str = CHATGPT_ISSUER,
        client_id: str = CHATGPT_CLIENT_ID,
    ) -> None:
        self._provider = provider_name
        self._issuer = issuer
        self._client_id = client_id
        self._lock = threading.Lock()
        self._tokens: OAuthTokens | None = None
        self._force_refresh = False
        self._account = ""
        self._last_returned = ""

    def _stored(self) -> OAuthTokens:
        tokens = load_oauth_tokens(self._provider)
        if tokens is None:
            raise ProviderError(
                f"No ChatGPT sign-in stored for provider {self._provider!r};"
                " run `agent6 connect chatgpt`.",
                status_code=401,
            )
        # The stored id must match the token's own claim: an entry written by
        # an older parser can hold a USER id where the account id belongs, and
        # trusting it would send a wrong chatgpt-account-id header. The repair
        # is a reconnect, never a silent migration.
        claimed = account_id_of(TokenGrant(tokens.access_token, "", 0.0, ""))
        if claimed and tokens.account_id and claimed != tokens.account_id:
            raise ProviderError(
                f"The stored ChatGPT sign-in for {self._provider!r} carries an account id"
                " that does not match its own token; run `agent6 connect chatgpt` to sign"
                " in again.",
                status_code=401,
            )
        return self._same_account(tokens)

    def _same_account(self, tokens: OAuthTokens) -> OAuthTokens:
        """Pin on first sight; refuse a grant bound to another account."""
        account = tokens.account_id
        if not self._account:
            self._account = account
        elif account and account != self._account:
            raise ProviderError(
                f"The stored ChatGPT sign-in for {self._provider!r} now belongs to a"
                f" different account than this run started under;"
                " run `agent6 connect chatgpt` to sign in again.",
                status_code=401,
            )
        return tokens

    def _adopt(self, tokens: OAuthTokens) -> str:
        self._tokens = tokens
        self._force_refresh = False
        self._last_returned = tokens.access_token
        return tokens.access_token

    def token(self) -> str:
        with self._lock:
            tokens = self._tokens or self._stored()
            if not self._force_refresh and time.time() < tokens.expires_at - _REFRESH_SKEW_S:
                return self._adopt(tokens)
            # Interprocess: the refresh token is SINGLE-USE, so the whole
            # reload-refresh-save transaction serializes on the secrets lock
            # (reentrant with save_oauth_tokens' own take). Unlike config
            # writes (atomic either way), an unserialized rotation can kill
            # the sign-in for every process, so an unheld lock REFUSES with a
            # retryable error instead of proceeding on a fiction.
            with locked_file(secrets_path()) as held:
                if not held:
                    raise ProviderError(
                        "could not take the credential-refresh lock beside"
                        f" {secrets_path()}; refusing to rotate the single-use"
                        " ChatGPT refresh token (remove a stale .lock sibling"
                        " if one is left over)"
                    )
                # Re-read UNDER the lock: a sibling that finished first is
                # adopted (after a 401 that means retry with its token, not
                # burn another rotation on a grant that may already be fresh).
                stored = self._stored()
                fresh_enough = time.time() < stored.expires_at - _REFRESH_SKEW_S
                if fresh_enough and stored.access_token != self._last_returned:
                    return self._adopt(stored)
                tokens = stored
                if not tokens.refresh_token:
                    raise ProviderError(
                        f"Stored ChatGPT sign-in for {self._provider!r} has no refresh token;"
                        " run `agent6 connect chatgpt`.",
                        status_code=401,
                    )
                try:
                    grant = refresh_grant(self._issuer, self._client_id, tokens.refresh_token)
                except ProviderError as exc:
                    grant = None
                    if "refresh_token_reused" not in str(exc):
                        raise
                    # Another HOST may have rotated (the flock covers only this
                    # one). One beat, one re-read; a fresh sibling grant wins.
                    time.sleep(1.0)
                    rescued = self._stored()
                    if rescued.access_token == tokens.access_token:
                        raise
                    return self._adopt(rescued)
                fresh = self._same_account(tokens_from_grant(grant, previous=tokens))
                save_oauth_tokens(self._provider, fresh)
                return self._adopt(fresh)

    def invalidate(self, status: int = 401) -> bool:
        """Arm recovery for the next `token()` after an auth failure. Only a
        401 means the bearer itself is bad; a 403 is permission or
        entitlement, and neither a newer sibling grant nor a rotation can
        change what the account is allowed to do -- recovery changes
        nothing, so the caller gets False and does not retry."""
        if status != 401:
            return False
        with self._lock:
            self._force_refresh = True
        return True

    def account_id(self) -> str:
        """The account id the backend requires as `chatgpt-account-id`."""
        with self._lock:
            tokens = self._tokens or self._stored()
            self._tokens = tokens
        if tokens.account_id:
            return tokens.account_id
        return account_id_of(TokenGrant(tokens.access_token, "", 0.0))
