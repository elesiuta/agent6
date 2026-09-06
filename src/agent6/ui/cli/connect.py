# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""`agent6 connect`, add a provider + API key."""

from __future__ import annotations

import contextlib
import getpass
import html as html_module
import http.server
import os
import re
import secrets as pysecrets
import sys
import threading
import time
import webbrowser
from pathlib import Path
from urllib.parse import urlsplit

from pydantic import ValidationError

from agent6.config import (
    AnthropicProviderEntry,
    OpenAIProviderEntry,
    ProviderEntry,
    validate_base_url,
)
from agent6.config.layer import repo_config_path_for
from agent6.config.write import PROVIDER_DEFAULTS, ConfigLeafValue, set_config_leaves
from agent6.models.cache import probe_provider_key
from agent6.paths import global_config_path
from agent6.providers.chatgpt_oauth import (
    CALLBACK_PORT,
    CHATGPT_CLIENT_ID,
    CHATGPT_ISSUER,
    DEVICE_VERIFY_PATH,
    TokenGrant,
    authorize_url,
    exchange_code,
    parse_callback,
    pkce_pair,
    plan_type_of,
    poll_device_auth,
    revoke_tokens,
    start_device_auth,
    tokens_from_grant,
)
from agent6.providers.claude_code import login_status
from agent6.providers.types import ProviderError
from agent6.secrets import (
    SecretsError,
    delete_provider_secrets,
    load_oauth_tokens,
    save_oauth_tokens,
    save_secret,
)


def _prompt_api_key(name: str) -> str:
    """Prompt for an API key without leaking it.

    On Python 3.14+ `getpass` accepts `echo_char` so we mask each
    keystroke with `*`, live feedback that the paste landed, without ever
    revealing the key. On 3.12/3.13 input stays fully hidden and we print a
    post-entry summary (length + last four chars) so the operator can still
    tell a partial/garbled paste from a clean one. The key itself is never
    logged.
    """
    prompt = f"API key for {name} (input hidden, blank for none): "
    if not sys.stdin.isatty():
        # No controlling terminal (piped/scripted connect): getpass would fall
        # back to an unmasked read AND print a GetPassWarning about echo. Read a
        # plain line instead -- echo is moot without a terminal, and the scary
        # warning is suppressed.
        try:
            return input(prompt).strip()
        except EOFError:
            return ""
    masked = False
    try:
        api_key = getpass.getpass(prompt, echo_char="*").strip()  # type: ignore[call-arg]
        masked = True
    except TypeError:
        # Python < 3.14: no echo_char parameter.
        api_key = getpass.getpass(prompt).strip()
    except EOFError:
        return ""
    if api_key and not masked:
        tail = f", ending …{api_key[-4:]}" if len(api_key) >= 8 else ""
        print(f"Captured key: {len(api_key)} chars{tail}.")
    return api_key


def _prompt_base_url(default_url: str) -> str:
    """Prompt for an OpenAI-compatible base URL and validate it.

    Validates before any secret/config write so a scheme-less value (e.g. an
    API key pasted into the wrong prompt) is rejected up front rather than
    persisted and surfaced later as an opaque HTTP error. Raises `ValueError`
    on an invalid URL (same check as the `OpenAIProviderEntry.base_url`
    validator).
    """
    try:
        url = input(f"Base URL [{default_url}]: ").strip() or default_url
    except EOFError:
        url = default_url
    validate_base_url(url)
    return url


def _resolve_provider_name(provider: str) -> str | None:
    """Resolve + validate the provider name; print an error and return None if bad.

    The name becomes a TOML table key `[providers.<name>]`; a non-bare-key
    name (space, dot, bracket, …) would be written verbatim and corrupt the
    whole config file, which `connect` -- unlike `model`/`config set` --
    does not re-validate after writing. So reject it before any write.
    """
    name = provider.strip()
    if not name:
        print("Known presets: " + ", ".join(sorted(PROVIDER_DEFAULTS)) + " (or any custom name).")
        try:
            name = input("Provider name [anthropic]: ").strip() or "anthropic"
        except EOFError:
            print("ERROR: no input.", file=sys.stderr)
            return None
    if not re.fullmatch(r"[A-Za-z0-9_-]+", name):
        print(
            f"ERROR: provider name {name!r} is not a valid TOML bare key"
            " (use only letters, digits, '-', '_').",
            file=sys.stderr,
        )
        return None
    return name


def _verify_key(*, api_format: str, base_url: str, api_key: str) -> None:
    """Probe the provider's /models endpoint to confirm the key authenticates.

    A read-only GET, so it does not violate connect's no-remote-execution rule.
    Prints the outcome; never raises (a probe failure must not fail connect, the
    key is already saved). Skipped for offline/local endpoints via --no-verify.
    """
    try:
        entry: ProviderEntry = (
            AnthropicProviderEntry(api_format="anthropic")
            if api_format == "anthropic"
            else OpenAIProviderEntry(
                api_format="openai", base_url=base_url or "https://api.openai.com/v1"
            )
        )
    except ValidationError as exc:
        print(f"  (skipped key check: {exc})", file=sys.stderr)
        return
    print("Checking the key against the provider...")
    result = probe_provider_key(entry, api_key)
    if result.status == "ok":
        print(f"  Key validated: {result.detail}.")
    elif result.status == "auth_failed":
        print(
            f"  WARNING: the provider REJECTED this key ({result.detail}). It was saved anyway;\n"
            "  re-run `agent6 connect` with the correct key (or pass --no-verify for a local"
            " endpoint).",
            file=sys.stderr,
        )
    elif result.status == "unsupported":
        print(f"  (key check skipped: {result.detail})")
    else:  # unreachable
        print(
            f"  NOTE: could not reach the provider to validate the key ({result.detail}); saved"
            " anyway.",
            file=sys.stderr,
        )


class _CallbackServer:
    """One-shot localhost receiver for the OAuth redirect.

    Binds 127.0.0.1:1455 (the client registration pins the port) and serves
    until the `/auth/callback` hit arrives; every other path 404s. The
    authorization code is held in memory only, never logged.
    """

    def __init__(self, state: str, *, port: int = CALLBACK_PORT) -> None:
        self._code: str | None = None
        self._got = threading.Event()
        outer = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # BaseHTTPRequestHandler API name
                parts = urlsplit(self.path)
                if parts.path != "/auth/callback":
                    self.send_error(404)
                    return
                try:
                    outer._code = parse_callback(parts.query, state=state)
                    body = b"<html><body>Signed in. Return to the terminal.</body></html>"
                    status = 200
                except ValueError as exc:
                    # Escaped: the query is attacker-reachable while this
                    # listener is up, and a reflected error_description would
                    # otherwise run script on this localhost origin. The
                    # terminal keeps the exact detail; the page stays generic
                    # for the refusal text itself.
                    body = f"<html><body>{html_module.escape(str(exc))}</body></html>".encode()
                    status = 400
                self.send_response(status)
                self.send_header("content-type", "text/html; charset=utf-8")
                self.send_header("content-security-policy", "default-src 'none'")
                self.send_header("x-content-type-options", "nosniff")
                self.send_header("cache-control", "no-store")
                self.send_header("content-length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                if status == 200:
                    outer._got.set()

            def log_message(self, format: str, *args: object) -> None:
                del format, args  # silent: the code must not reach the terminal

        self._server = http.server.HTTPServer(("127.0.0.1", port), Handler)
        self._thread = threading.Thread(
            target=self._server.serve_forever, name="agent6-oauth-callback", daemon=True
        )
        self._thread.start()

    @property
    def port(self) -> int:
        return int(self._server.server_address[1])

    def wait(self, timeout_s: float) -> str | None:
        """The code, or None on timeout. Polls so Ctrl-C lands promptly."""
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if self._got.wait(0.25):
                return self._code
        return None

    def close(self) -> None:
        self._server.shutdown()
        self._server.server_close()


def _gui_browser_available() -> bool:
    """Auto-open the sign-in URL only where a GUI browser can take it.

    On a display-less Linux box `webbrowser.open` falls back to a console
    browser (w3m/lynx), which takes over the very terminal the sign-in
    prompt lives on; there the printed URL is the flow.
    """
    if sys.platform in ("darwin", "win32"):
        return True
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


def _code_via_callback_server(url: str, state: str) -> str | None:
    """The GUI path: open the browser, wait on localhost for the redirect.
    None on timeout, Ctrl-C, or an unbindable port (the caller falls back)."""
    try:
        server = _CallbackServer(state)
    except OSError as exc:
        print(f"(no local callback: port {CALLBACK_PORT} unavailable: {exc})")
        return None
    with contextlib.suppress(Exception):
        webbrowser.open(url)
    print(f"Waiting for the sign-in redirect on localhost:{CALLBACK_PORT}")
    print("(Ctrl-C to paste the callback URL by hand instead)")
    try:
        return server.wait(timeout_s=300.0)
    except KeyboardInterrupt:
        print()
        return None
    finally:
        server.close()


def _grant_via_device_code(issuer: str, client_id: str) -> TokenGrant | None:
    """The no-display path: show a short code, poll while the person enters
    it at the issuer's device page from any browser (nothing to forward over
    SSH). None when the issuer has the flow disabled, on refusal, or on
    Ctrl-C -- the caller falls back to pasting the callback URL."""
    try:
        device = start_device_auth(issuer, client_id)
    except ProviderError as exc:
        print(f"(device sign-in unavailable: {exc})")
        return None
    if device is None:
        return None
    print(f"On any device, open  {issuer.rstrip('/')}{DEVICE_VERIFY_PATH}")
    print(f"and enter the code:  {device.user_code}")
    print("(waiting; Ctrl-C to paste the callback URL by hand instead)")
    try:
        return poll_device_auth(issuer, client_id, device)
    except KeyboardInterrupt:
        print()
        return None
    except ProviderError as exc:
        print(f"(device sign-in failed: {exc})")
        return None


def _chatgpt_sign_in(name: str) -> int:
    """The ChatGPT OAuth sign-in.

    Three ways in, picked by the environment: a GUI machine gets the browser
    + localhost callback; a display-less terminal gets the code-entry device
    flow; pasting the callback URL always works (and is the whole flow for a
    piped stdin). Never executes anything the remote returns; the only
    inputs read back are the authorization code (state-checked) and the
    token JSON.
    """
    issuer, client_id = CHATGPT_ISSUER, CHATGPT_CLIENT_ID
    verifier, challenge = pkce_pair()
    state = pysecrets.token_urlsafe(24)
    url = authorize_url(issuer, client_id, challenge=challenge, state=state)
    print("Open this URL to sign in with your ChatGPT account:\n\n  " + url + "\n")

    grant: TokenGrant | None = None
    code: str | None = None
    if sys.stdin.isatty() and _gui_browser_available():
        code = _code_via_callback_server(url, state)
    elif sys.stdin.isatty():
        grant = _grant_via_device_code(issuer, client_id)
    if grant is None and code is None:
        try:
            pasted = input("Paste the callback URL the browser landed on: ").strip()
        except EOFError:
            print("ERROR: no callback input.", file=sys.stderr)
            return 2
        try:
            code = parse_callback(pasted, state=state)
        except ValueError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 2

    if grant is None:
        assert code is not None
        try:
            grant = exchange_code(issuer, client_id, code=code, verifier=verifier)
        except ProviderError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 2
    tokens = tokens_from_grant(grant)
    try:
        saved = save_oauth_tokens(name, tokens)
    except SecretsError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    plan = plan_type_of(grant)
    print(f"Signed in{f' ({plan} plan)' if plan else ''}; tokens saved to {saved} (0600).")
    if not tokens.account_id:
        print(
            "WARNING: the sign-in carried no ChatGPT account id; runs will refuse until a"
            " sign-in with a ChatGPT plan succeeds.",
            file=sys.stderr,
        )
    print(
        "\nNote: whether these conversations train OpenAI's models follows the ChatGPT\n"
        "account's own data controls (Settings > Data controls > 'Improve the model for\n"
        "everyone'); agent6 cannot change that setting, and it never sends feedback or\n"
        "ratings, which would opt those turns in regardless of it."
    )
    return 0


def _claude_code_check(name: str) -> None:
    """No secret to store: the binary carries the operator's own login. Checked
    now so a signed-out install is named here, not at the first run; connect
    checks `claude` on PATH, the run preflight and `agent6 model` check
    `[providers.<name>].binary`."""
    err = login_status("claude")
    if err is None:
        print("Claude Code (`claude` on PATH): signed in.")
        return
    print(
        f"WARNING: {err}\n  [providers.{name}] is written but not usable yet.",
        file=sys.stderr,
    )


def _prompt_api_format(name: str, preset_format: str) -> str | None:
    """The api_format for *name*: the preset's, else the operator's answer;
    None (after printing why) on no input or an unknown value."""
    api_format = preset_format
    if not api_format:
        try:
            api_format = (
                input(f"API format for {name!r} [anthropic/openai/chatgpt/claude_code]: ").strip()
                or "anthropic"
            )
        except EOFError:
            return None
    if api_format not in ("anthropic", "openai", "chatgpt", "claude_code"):
        print(
            f"ERROR: unknown api_format {api_format!r}"
            " (expected anthropic, openai, chatgpt, or claude_code).",
            file=sys.stderr,
        )
        return None
    return api_format


def _cmd_logout(name: str, api_format: str) -> int:
    """Remove a provider's stored credentials (`connect --logout`).

    For a chatgpt-format provider the OAuth grant is revoked at the issuer
    first (best effort: local removal proceeds regardless, and revoking an
    already-dead token is a success). The `[providers.<name>]` config block
    stays; only credentials are removed. A claude_code provider holds none
    here: its login belongs to the binary.
    """
    if api_format == "claude_code":
        print(
            "agent6 stores no Claude Code credentials; `claude auth logout` signs out.",
            file=sys.stderr,
        )
        return 2
    tokens = load_oauth_tokens(name)
    if tokens is not None:
        err = revoke_tokens(CHATGPT_ISSUER, CHATGPT_CLIENT_ID, tokens)
        if err is None:
            print(f"Revoked the ChatGPT sign-in for {name!r} at {CHATGPT_ISSUER}.")
        else:
            print(f"WARNING: revocation failed ({err}); removing local tokens anyway.")
    try:
        removed = delete_provider_secrets(name)
    except SecretsError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(
        f"Removed stored credentials for {name!r} from secrets.toml."
        if removed
        else f"No stored credentials for {name!r}."
    )
    return 0


def _cmd_connect(*, provider: str, to_repo: bool, verify: bool = True, logout: bool = False) -> int:  # noqa: PLR0911, PLR0912
    """Interactively add a provider + API key.

    Security: this command NEVER executes anything supplied by a remote. It
    only prompts locally (key via getpass, hidden, or masked with `*` on
    Python 3.14+), stores the key in the 0600 secrets file, writes a minimal
    `[providers.<name>]` block, and (unless `verify` is False) makes one
    read-only GET to the provider's `/models` endpoint to confirm the key
    authenticates.
    """
    name = _resolve_provider_name(provider)
    if name is None:
        return 2
    preset = PROVIDER_DEFAULTS.get(name)
    preset_format = preset["api_format"] if preset else ""
    if logout:
        return _cmd_logout(name, preset_format)
    print("agent6 connect: add a provider and API key.\n")
    api_format = _prompt_api_format(name, preset_format)
    if api_format is None:
        return 2
    base_url = (preset or {}).get("base_url", "")
    if api_format == "openai":
        try:
            base_url = _prompt_base_url(base_url or "https://api.openai.com/v1")
        except ValueError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 2

    if api_format == "claude_code":
        _claude_code_check(name)
        api_key = ""
    elif api_format == "chatgpt":
        rc = _chatgpt_sign_in(name)
        if rc != 0:
            return rc
        api_key = ""
    else:
        try:
            api_key = _prompt_api_key(name)
        except EOFError:
            api_key = ""
    if api_key:
        try:
            saved = save_secret(name, api_key)
        except SecretsError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 2
        print(f"Saved key to {saved} (0600).")
        if verify:
            _verify_key(api_format=api_format, base_url=base_url, api_key=api_key)
    elif api_format == "anthropic":
        # The Anthropic api_format always sends a key; a keyless block is
        # unusable and `agent6 run` would later fail with "no API key". Say so
        # now rather than contradicting ourselves one command later.
        print(
            f"WARNING: no key entered, but the Anthropic API format requires one.\n"
            f"  [providers.{name}] is written but not usable yet; rerun"
            " `agent6 connect`\n  (or set the api_key_env var) before `agent6 run`."
        )
    elif api_format == "openai":
        print("No key entered; assuming an unauthenticated/local endpoint.")

    target = repo_config_path_for(Path.cwd()) if to_repo else global_config_path()
    fields: dict[str, ConfigLeafValue] = {"api_format": api_format}
    if api_format == "openai" and base_url and base_url != "https://api.openai.com/v1":
        fields["base_url"] = base_url
    # Leaf surgery, not a whole-block replace: connect is the documented
    # add/UPDATE path, and a re-run (key rotation, base_url fix) must preserve
    # hand-added sibling keys and comments. Revalidates the merged config and
    # rolls the file back on failure so a bad endpoint never leaves config.toml
    # broken (the key, saved above, is a harmless orphan until a valid retry).
    err = set_config_leaves(Path.cwd(), f"providers.{name}", fields, to_repo=to_repo)
    if err is not None:
        print(f"Refusing: that would make the config invalid:\n{err}", file=sys.stderr)
        return 2
    print(f"Wrote [providers.{name}] to {target}.")
    print(
        "\nNext: `agent6 model worker "
        f"{name} <model>` to route a role here, then `agent6 config show`."
    )
    return 0
