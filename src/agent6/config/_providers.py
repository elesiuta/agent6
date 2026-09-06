# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The `[providers.*]` model: one entry per endpoint, discriminated by wire format."""

from __future__ import annotations

from typing import Annotated, Any, Literal
from urllib.parse import urlsplit

from pydantic import BaseModel, Discriminator, Field, field_validator, model_validator

from agent6.config._base import MODEL_CONFIG, Argv
from agent6.config._sandbox import is_cleartext_url, is_loopback_url

ApiFormat = Literal["anthropic", "openai", "chatgpt", "claude_code"]
Deployment = Literal["direct", "vertex", "azure"]
AuthStyle = Literal["x_api_key", "bearer", "api_key_header", "none"]


def validate_base_url(url: str, field: str = "base_url") -> None:
    """Reject a `[providers.*].base_url` that is not an http(s) URL with a host.

    A provider's
    `base_url` is the host+path prefix the HTTP client posts to (the
    deployment profile appends `/chat/completions`, `/messages`, etc.), so
    it must carry an explicit
    `http://` / `https://` scheme and a host. The common paste error this
    catches is dropping an API key (or a bare host) into the field, which would
    otherwise be accepted and only fail much later as an opaque HTTP error.
    """
    try:
        parts = urlsplit(url)
        port = parts.port  # urlsplit raises ValueError on an out-of-range port
    except ValueError as exc:
        raise ValueError(f"invalid {field} {url!r}: {exc}") from exc
    if parts.scheme not in ("http", "https"):
        raise ValueError(f"{field} {url!r} must start with http:// or https://")
    if not parts.hostname:
        raise ValueError(f"{field} {url!r} has no host")
    if port is not None and not (1 <= port <= 65535):
        raise ValueError(f"{field} {url!r} has an invalid port")


_ANTHROPIC_DEFAULT_BASE_URL = "https://api.anthropic.com/v1"
_OPENAI_DEFAULT_BASE_URL = "https://api.openai.com/v1"
_CHATGPT_DEFAULT_BASE_URL = "https://chatgpt.com/backend-api/codex"


def _default_base_url(api_format: str, deployment: str) -> str | None:
    """Default `base_url` for a (format, deployment), or None if required.

    Only the `direct` deployment has a sensible fixed endpoint; vertex/azure
    (and future bedrock) carry project/resource/region in the URL, so the
    operator must supply `base_url`.
    """
    if deployment != "direct":
        return None
    if api_format == "anthropic":
        return _ANTHROPIC_DEFAULT_BASE_URL
    return _CHATGPT_DEFAULT_BASE_URL if api_format == "chatgpt" else _OPENAI_DEFAULT_BASE_URL


def _default_auth_style(api_format: str, deployment: str) -> str:
    """Default `auth_style` for a (format, deployment)."""
    if deployment == "azure":
        return "api_key_header"
    if deployment == "vertex":
        return "bearer"
    return "x_api_key" if api_format == "anthropic" else "bearer"


_API_FORMAT_DESCRIPTION = (
    "The wire format: `anthropic` (the Messages API), `openai` (Chat Completions: OpenAI, "
    "OpenRouter, Ollama, vLLM, LM Studio, llama.cpp, Gemini's OpenAI endpoint), `chatgpt` "
    "(the ChatGPT-subscription Codex backend, Responses API), or `claude_code` (the installed, "
    "signed-in Claude Code binary on a Claude subscription; no HTTP endpoint, no key)."
)


def _require_json_shaped(value: Any, path: str) -> None:
    """Refuse any value JSON cannot carry (TOML also parses dates/times)."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return
    if isinstance(value, list):
        for i, item in enumerate(value):
            _require_json_shaped(item, f"{path}[{i}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            _require_json_shaped(item, f"{path}.{key}")
        return
    raise ValueError(
        f"extra_body{path} holds a {type(value).__name__}, which JSON cannot carry"
        " (a TOML date/time is the usual cause); quote it as a string"
    )


class _ProviderBase(BaseModel):
    """Transport + auth fields shared by every provider, independent of format.

    Three orthogonal concerns: `api_format` (the discriminator) selects the
    wire dialect; `deployment` selects the URL /
    model-placement profile; and the auth fields (`auth_style` + a static
    `api_key_env` or a refreshable `token_command`) select the credential.
    They compose freely -- e.g. Claude-on-Vertex and Gemini-on-Vertex differ
    only in `api_format` (both `deployment = "vertex"`). `base_url` and
    `auth_style` default from (api_format, deployment) in `_fill_defaults` so
    a minimal entry (just `api_format`) is fully usable. Each block is
    one endpoint; configure as many as you like under any names and reference
    them from `[models.*]`.
    """

    model_config = MODEL_CONFIG

    # Declared on the base only to fix the FIELD ORDER: a redeclared field
    # keeps its base position, so api_format leads every subclass's
    # model_fields (the docs table and `config show` print that order). Each
    # subclass narrows it to its own literal, which is what discriminates.
    api_format: ApiFormat
    deployment: Deployment = Field(
        default="direct",
        description=(
            "`direct`, `vertex` (Google Vertex AI), or `azure` (Azure OpenAI; `openai` format "
            "only): the URL shape and where the model name and API version go."
        ),
    )
    # Resolved by _fill_defaults from (api_format, deployment) when omitted;
    # never empty post-validation. The host also feeds the egress allow-list.
    base_url: str = Field(
        default="",
        description=(
            "The endpoint's host and path prefix (`https://api.anthropic.com/v1`); required for "
            "`vertex` and `azure`. The provider API destination; a ChatGPT sign-in also dials "
            "its fixed OAuth authority."
        ),
    )
    # Auth header style; defaults from (api_format, deployment) in _fill_defaults.
    auth_style: AuthStyle = Field(
        default="bearer",
        description=(
            "How the key is sent: `x_api_key` (Anthropic), `bearer` (`Authorization: Bearer`, the "
            "OpenAI style), `api_key_header` (Azure), or `none` (an unauthenticated local "
            "endpoint). `agent6 connect` sets it."
        ),
    )
    # Static key: env var name (falls back to secrets.toml by provider name).
    # Secrets live here, never in base_url/extra_headers/extra_query.
    api_key_env: str | None = Field(
        default=None,
        min_length=1,
        description=(
            "The environment variable holding the API key; it wins over `secrets.toml`. Unset for "
            "a key `agent6 connect` stored, or an unauthenticated local endpoint."
        ),
    )
    token_command: Argv = Field(
        default=(),
        description=(
            "A command (argv) that prints a short-lived bearer token to stdout, re-run when "
            "`token_command_ttl_s` expires and once after a `401` or `403`. Wins over "
            "`api_key_env`."
        ),
    )
    token_command_ttl_s: float = Field(
        gt=0.0,
        default=300.0,
        description="Seconds a `token_command` token is reused before the command runs again.",
    )
    extra_headers: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Extra HTTP headers on every request to this provider. Never a secret: the config file "
            "is not `0600`."
        ),
    )
    extra_body: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Provider-specific JSON merged last into every request body, so tuning keys "
            "(`max_tokens`, `temperature`) win; the structural keys agent6 owns (messages, model, "
            "stream, tools, tool choice, response shape) are filtered out. Values must be "
            "JSON-shaped (a TOML date or time is refused). OpenRouter's routing options go here."
        ),
    )
    extra_query: dict[str, str] = Field(
        default_factory=dict,
        description="Extra URL query parameters on every request (Azure's `api-version`).",
    )
    # Per-HTTP-call read/write budget in seconds; the connect phase is bounded
    # separately (providers._transport.CONNECT_TIMEOUT_S) so a blackholed
    # connect fails in seconds, not this. Default 600s streams a long response;
    # lower it on benches that should fail fast.
    http_timeout_s: float = Field(
        gt=0.0,
        default=600.0,
        description=(
            "Seconds one HTTP call may take to read or write; the connect phase is bounded at 20 s "
            "regardless."
        ),
    )

    @model_validator(mode="before")
    @classmethod
    def _fill_defaults(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        fmt = data.get("api_format")
        dep = data.get("deployment", "direct")
        if fmt == "anthropic" and dep == "azure":
            raise ValueError("deployment 'azure' requires api_format 'openai'")
        if fmt == "chatgpt" and dep != "direct":
            raise ValueError("api_format 'chatgpt' supports deployment 'direct' only")
        if not data.get("base_url"):
            default = _default_base_url(fmt, dep) if isinstance(fmt, str) else None
            if default is None:
                raise ValueError(f"base_url is required for deployment {dep!r}")
            data["base_url"] = default
        if not data.get("auth_style") and isinstance(fmt, str):
            data["auth_style"] = _default_auth_style(fmt, dep)
        if dep == "azure" and "api-version" not in (data.get("extra_query") or {}):
            raise ValueError("deployment 'azure' requires extra_query['api-version']")
        return data

    @field_validator("base_url")
    @classmethod
    def _check_base_url(cls, v: str) -> str:
        if v:
            validate_base_url(v)
        return v

    @field_validator("extra_body")
    @classmethod
    def _check_extra_body_json_shaped(cls, v: dict[str, Any]) -> dict[str, Any]:
        """TOML also parses dates and times, which JSON cannot carry: caught
        here, at load, instead of a serialization crash mid-request."""
        for key, value in v.items():
            _require_json_shaped(value, f".{key}")
        return v

    @model_validator(mode="after")
    def _none_auth_takes_no_credential(self) -> _ProviderBase:
        """`auth_style = "none"` sends no auth header, so a credential source
        named beside it is dead config that reads as authenticated; refuse
        rather than silently ignore the key."""
        if self.auth_style == "none" and (self.api_key_env or self.token_command):
            named = "api_key_env" if self.api_key_env else "token_command"
            raise ValueError(
                f"auth_style = 'none' sends no auth header, so {named} would never"
                " be used; drop one or the other"
            )
        return self


class AnthropicProviderEntry(_ProviderBase):
    """`api_format = "anthropic"` -- the Anthropic Messages wire format.

    `deployment = "direct"` (default) hits api.anthropic.com; `"vertex"`
    is Claude-on-Vertex (model id in the URL, `anthropic_version` in the body,
    a Google-OAuth bearer via `token_command`).
    """

    # The narrowing override is sound: the model is frozen, so the attribute
    # can never be written back through the wider base type.
    api_format: Literal["anthropic"] = (  # pyright: ignore[reportIncompatibleVariableOverride]
        Field(description=_API_FORMAT_DESCRIPTION)
    )
    prompt_caching: bool = Field(
        default=True,
        description=(
            "Anthropic prompt caching: the system prompt, the tools, and the growing conversation "
            "are re-read at 0.1x the input price. `anthropic` format only."
        ),
    )


class OpenAIProviderEntry(_ProviderBase):
    """`api_format = "openai"` -- any OpenAI Chat Completions wire format.

    `deployment = "direct"` works against OpenAI, OpenRouter, Ollama, vLLM,
    LM Studio, llama.cpp, Gemini's OpenAI-compatible endpoint, GitHub Copilot,
    etc.; `"vertex"` is Gemini's Vertex OpenAPI endpoint; `"azure"` is Azure
    OpenAI (deployment-name in the URL, api-version query param, `api-key`
    header).
    """

    api_format: Literal["openai"] = (  # pyright: ignore[reportIncompatibleVariableOverride]
        Field(description=_API_FORMAT_DESCRIPTION)
    )


class ChatGPTProviderEntry(_ProviderBase):
    """`api_format = "chatgpt"` -- the ChatGPT-subscription Codex backend.

    The Responses wire format at `chatgpt.com/backend-api/codex`, authorized
    by the OAuth tokens `agent6 connect <name>` stores in `secrets.toml`
    (no API key). Usage draws on the account's ChatGPT plan limits.
    This provider dials only `base_url` and OpenAI's fixed OAuth authority
    (the issuer and client id are constants, not config).
    """

    api_format: Literal["chatgpt"] = (  # pyright: ignore[reportIncompatibleVariableOverride]
        Field(description=_API_FORMAT_DESCRIPTION)
    )

    @field_validator("base_url")
    @classmethod
    def _chatgpt_base_url_is_https(cls, v: str) -> str:
        # The bearer and account id ride every request; unlike a generic
        # base_url (where plain http serves LAN Ollama), cleartext for this
        # backend is never right off-loopback.
        if is_cleartext_url(v) and not is_loopback_url(v):
            raise ValueError(
                "a chatgpt base_url must use https (plain http is allowed only"
                " for a loopback test endpoint)"
            )
        return v

    @field_validator("extra_headers")
    @classmethod
    def _reserved_headers_stay_structural(cls, v: dict[str, str]) -> dict[str, str]:
        # authorization / chatgpt-account-id / originator / session-id are the
        # authentication structure; an overlay replacing them would silently
        # re-route or mislabel every call.
        reserved = {"authorization", "chatgpt-account-id", "originator", "session-id"}
        clash = sorted(k for k in v if k.lower() in reserved)
        if clash:
            raise ValueError(
                f"extra_headers may not override the chatgpt auth headers: {', '.join(clash)}"
            )
        return v

    @model_validator(mode="after")
    def _oauth_takes_no_key_source(self) -> ChatGPTProviderEntry:
        """The chatgpt format authenticates with the connect-stored OAuth
        tokens; a static key source beside them is dead config."""
        if self.api_key_env or self.token_command:
            named = "api_key_env" if self.api_key_env else "token_command"
            raise ValueError(
                f"api_format 'chatgpt' authenticates with the OAuth tokens"
                f" `agent6 connect <name>` stores, so {named} would never be used;"
                " drop it"
            )
        if self.auth_style != "bearer":
            raise ValueError(
                "api_format 'chatgpt' always sends its OAuth token as"
                f" `Authorization: Bearer`; auth_style {self.auth_style!r} is not honoured,"
                " drop it"
            )
        return self


class ClaudeCodeProviderEntry(BaseModel):
    """`api_format = "claude_code"` -- the operator's installed Claude Code binary.

    Not a `_ProviderBase`: it dials no endpoint and holds no credential, so the
    transport and auth fields do not exist on it (`extra="forbid"` refuses each
    by name). The binary carries the operator's own Claude login; usage draws
    on that subscription's plan windows.
    """

    model_config = MODEL_CONFIG

    api_format: Literal["claude_code"] = Field(description=_API_FORMAT_DESCRIPTION)
    binary: str = Field(
        default="claude",
        min_length=1,
        description=(
            "The Claude Code executable: a name on PATH or an absolute path. `claude_code`"
            " format only."
        ),
    )


ProviderEntry = Annotated[
    AnthropicProviderEntry | OpenAIProviderEntry | ChatGPTProviderEntry | ClaudeCodeProviderEntry,
    Discriminator("api_format"),
]


def plan_metered(entry: object) -> bool:
    """Whether calls through *entry* draw on a subscription plan: metered in
    plan percent with an authoritative $0, never priced per token."""
    return isinstance(entry, (ChatGPTProviderEntry, ClaudeCodeProviderEntry))
