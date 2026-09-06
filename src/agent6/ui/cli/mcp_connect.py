# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""`agent6 mcp connect`, add an MCP server after proving it works.

The order is the whole point: handshake, list the tools, show them, and only
then write config. A server named in config that turns out not to answer is a
run that starts, logs "failed to start", and quietly has fewer tools than the
operator thinks -- discovered mid-task, if at all.

Nothing the server returns is ever executed. Its tool names and descriptions
are printed as text and stored nowhere.
"""

from __future__ import annotations

import shlex
import sys
from pathlib import Path

from agent6.app._setup import detect_env, mcp_server_policy
from agent6.config import (
    Config,
    MCPServerEntry,
    is_cleartext_url,
    is_loopback_url,
    mcp_server_name_refusal,
)
from agent6.config.layer import load_effective
from agent6.config.write import ConfigLeafValue, set_config_leaves
from agent6.sandbox.detect import resolve_isolation
from agent6.sandbox.jail import JailUnavailableError
from agent6.tools.mcp_client import MCPError, MCPServerSpec, MCPToolDescriptor, _MCPServer
from agent6.tools.mcp_http import HttpTransport

# Long enough for a cold `npx` to fetch and boot a server, which is the slow
# case an operator actually hits; the per-run default stays 10s.
_CONNECT_TIMEOUT_S = 60.0


def _probe(spec: MCPServerSpec) -> tuple[tuple[MCPToolDescriptor, ...], str]:
    """Start the server, take its tool list, stop it. Returns (tools, error)."""
    server = _MCPServer(
        name=spec.name,
        command=spec.command,
        startup_timeout_s=spec.startup_timeout_s,
        call_timeout_s=spec.call_timeout_s,
        pass_env=spec.pass_env,
        policy=spec.policy,
        http=spec.http,
    )
    try:
        server.start()
        return server.tools, ""
    except MCPError as exc:
        return (), str(exc)
    finally:
        server.close()


def _refuse_bad_flags(
    *, name: str, command: list[str], url: str, token_env: str, pass_env: list[str], cfg: Config
) -> str:
    """Why this invocation cannot be acted on, or "". Each transport owns one
    env flag, so the wrong pairing is a mistake worth naming rather than a
    setting that silently does nothing."""
    if bool(command) == bool(url):
        return (
            "give exactly one of a command to spawn or --url to connect to.\n"
            "  spawn:   agent6 mcp connect files -- npx -y"
            " @modelcontextprotocol/server-filesystem .\n"
            "  connect: agent6 mcp connect browser --url http://127.0.0.1:8931/mcp"
        )
    if token_env and not url:
        return "--token-env is for --url servers; a spawned one uses --pass-env"
    if pass_env and url:
        return "--pass-env is for spawned servers; a --url one uses --token-env"
    name_refusal = mcp_server_name_refusal(name)
    if name_refusal:
        # BEFORE the write: the name becomes a TOML table header.
        return name_refusal
    keys = {str(getattr(e, "api_key_env", "")) for e in cfg.providers.values()} - {""}
    leaked = sorted(keys.intersection(pass_env))
    if leaked:
        # An MCP server is third-party code running as the operator. A provider
        # key is the one thing agent6 keeps out of every child it spawns, and
        # `--pass-env` is the only way to hand one over by name.
        return (
            f"{', '.join(leaked)} holds a provider API key; agent6 does not pass one"
            " to an MCP server.\n"
            "  If the server needs its own credential, give it a different variable."
        )
    return ""


def _cleartext_token_go_ahead(url: str, token_env: str) -> bool:
    """True to proceed past the plaintext-non-loopback confirmation.

    Explicit-but-discouraged config, so the cost is named and never refused (an
    internal-network or VPN endpoint is a real case): interactive asks, default
    no; headless warns and proceeds.
    """
    if not (token_env and is_cleartext_url(url) and not is_loopback_url(url)):
        return True
    print(
        f"WARNING: {url} is plaintext http to a non-loopback host: the token"
        f" from ${token_env} will be readable on the network path.",
        file=sys.stderr,
    )
    if not sys.stdin.isatty():
        return True
    return input("Connect anyway? [y/N]: ").strip().lower() in ("y", "yes")


def _describe(spec: MCPServerSpec) -> str:
    if spec.http is not None:
        return f"connecting to {spec.http.url}"
    return f"spawning {shlex.join(spec.command)}"


def cmd_mcp_connect(  # noqa: PLR0911
    name: str,
    *,
    command: list[str],
    url: str,
    token_env: str,
    pass_env: list[str],
    to_repo: bool,
    config_path: Path | None = None,
) -> int:
    """Prove the server answers, then write it into config. Returns an exit code."""
    cfg = load_effective(Path.cwd(), config_path).config
    refusal = _refuse_bad_flags(
        name=name, command=command, url=url, token_env=token_env, pass_env=pass_env, cfg=cfg
    )
    if refusal:
        print(f"ERROR: {refusal}", file=sys.stderr)
        return 2
    if not _cleartext_token_go_ahead(url, token_env):
        print("nothing was written to config.", file=sys.stderr)
        return 1

    # The sandbox the run gives this server, so the handshake proves the
    # server the run will actually spawn (a script outside the workspace is
    # invisible inside it).
    isolation = resolve_isolation(cfg.sandbox.isolation, detect_env())
    entry = MCPServerEntry.model_validate(
        {"command": command, "url": url, "token_env": token_env, "pass_env": pass_env}
    )
    try:
        policy = None if url else mcp_server_policy(cfg, Path.cwd(), isolation, entry)
    except JailUnavailableError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    spec = MCPServerSpec(
        name=name,
        command=tuple(command),
        startup_timeout_s=_CONNECT_TIMEOUT_S,
        call_timeout_s=_CONNECT_TIMEOUT_S,
        pass_env=tuple(pass_env),
        policy=policy,
        http=HttpTransport(name=name, url=url, token_env=token_env) if url else None,
    )
    print(f"[agent6] {_describe(spec)} ...", file=sys.stderr)
    tools, error = _probe(spec)
    if error:
        print(f"ERROR: {name} did not answer: {error}", file=sys.stderr)
        if spec.policy is not None:
            print(
                f"       (probed under the run's {isolation} sandbox: a server outside the"
                f" workspace needs [mcp.servers.{name}.sandbox] read_paths, or"
                " unconfined = true)",
                file=sys.stderr,
            )
        print("       nothing was written to config.", file=sys.stderr)
        return 1
    if not tools:
        print(f"ERROR: {name} started but exposed no tools; nothing was written.", file=sys.stderr)
        return 1

    print(f"\n{name}: {len(tools)} tool{'' if len(tools) == 1 else 's'}")
    for tool in tools:
        # The server chose this text. Collapsing whitespace stops a forged
        # extra line; dropping the other control characters stops ESC
        # sequences repainting the operator's terminal.
        summary = "".join(c for c in " ".join(tool.description.split()) if c.isprintable())[:80]
        print(f"  mcp__{name}__{tool.tool_name}{'  ' + summary if summary else ''}")

    # Values, not TOML text: `format_toml_value` serializes each one, so a
    # list stays a list. Handing it a pre-quoted string wrote an argv as one
    # long string, which then validated as a tuple of characters.
    fields: dict[str, ConfigLeafValue] = {"enabled": True}
    if command:
        fields["command"] = command
    else:
        fields["url"] = url
    if token_env:
        fields["token_env"] = token_env
    if pass_env:
        fields["pass_env"] = pass_env
    written = set_config_leaves(Path.cwd(), f"mcp.servers.{name}", fields, to_repo=to_repo)
    if written is not None:
        print(f"ERROR: {written}", file=sys.stderr)
        return 2
    # The master switch is separate and off by default, so say so rather than
    # flipping a security-relevant default on the operator's behalf.
    print(f"\nwritten to {'the repo' if to_repo else 'the global'} config.")
    print("enable MCP for runs with:  agent6 config set mcp.enabled true")
    return 0


def cmd_mcp_list(config_path: Path | None = None) -> int:
    """The configured servers and how each is reached. Reads config only: it
    never starts anything, so it answers instantly and says nothing about
    whether a server currently works (`agent6 check mcp` does that)."""
    cfg = load_effective(Path.cwd(), config_path).config
    if not cfg.mcp.servers:
        print("no MCP servers configured. Add one with `agent6 mcp connect <name> ...`.")
        return 0
    state = "enabled" if cfg.mcp.enabled else "DISABLED (agent6 config set mcp.enabled true)"
    print(f"MCP is {state}\n")
    for name, srv in sorted(cfg.mcp.servers.items()):
        how = f"connect {srv.url}" if srv.url else f"spawn   {shlex.join(srv.command)}"
        off = "" if srv.enabled else "  [disabled]"
        print(f"  {name:<16} {how}{off}")
        if srv.token_env:
            print(f"  {'':<16} token from ${srv.token_env}")
        if srv.pass_env:
            print(f"  {'':<16} env {' '.join(srv.pass_env)}")
    return 0
