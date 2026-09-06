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
import shutil
import sys
from pathlib import Path

from pydantic import ValidationError

from agent6.app._setup import detect_env, mcp_server_spec, no_jail_cause
from agent6.config import (
    Config,
    MCPServerEntry,
    is_cleartext_url,
    is_loopback_url,
    mcp_server_name_refusal,
)
from agent6.config.layer import EffectiveConfig, load_effective, repo_config_path_for
from agent6.config.write import ConfigLeafValue, set_config_leaves, unset_config_table
from agent6.paths import global_config_path
from agent6.sandbox.detect import resolve_isolation
from agent6.sandbox.jail import JailUnavailableError
from agent6.tools.mcp_client import MCPManager, MCPServerSpec, MCPToolDescriptor, tool_count
from agent6.types import IsolationLevel
from agent6.ui.cli._common import error, warn

# Long enough for a cold `npx` to fetch and boot a server, which is the slow
# case an operator actually hits; the per-run default stays 10s.
_CONNECT_TIMEOUT_S = 60.0


def _probe(spec: MCPServerSpec) -> tuple[tuple[MCPToolDescriptor, ...], str]:
    """Start the server as a run does, take its tool list, stop it. Returns
    (tools, failure)."""
    manager = MCPManager.start([spec])
    try:
        return manager.descriptors(), manager.failures[0].error if manager.failures else ""
    finally:
        manager.close()


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
    warn(
        f"{url} is plaintext http to a non-loopback host: the token"
        f" from ${token_env} will be readable on the network path."
    )
    if not sys.stdin.isatty():
        return True
    return input("Connect anyway? [y/N]: ").strip().lower() in ("y", "yes")


def _describe(spec: MCPServerSpec) -> str:
    if spec.http is not None:
        return f"connecting to {spec.http.url}"
    return f"spawning {shlex.join(spec.command)}"


def _report_no_answer(
    name: str, command: list[str], isolation: IsolationLevel, failure: str
) -> None:
    """The failure and the one hint that applies: a binary missing on the
    host needs no sandbox grant; one present here but not in the jail does."""
    error(f"{name} did not answer: {failure}")
    if command and shutil.which(command[0]) is None:
        head = command[0]
        what = (
            "not executable"
            if Path(head).exists()
            else "no such file"
            if "/" in head
            else "not on PATH"
        )
        print(f"       {head}: {what} on this host.", file=sys.stderr)
    elif command and isolation != "none":
        print(
            f"       (probed under the run's {isolation} sandbox: a server outside the"
            f" workspace needs [mcp.servers.{name}.sandbox] read_paths, or"
            " unconfined = true)",
            file=sys.stderr,
        )
    print("       nothing was written to config.", file=sys.stderr)


def cmd_mcp_connect(
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
    effective = load_effective(Path.cwd(), config_path)
    cfg = effective.config
    refusal = _refuse_bad_flags(
        name=name, command=command, url=url, token_env=token_env, pass_env=pass_env, cfg=cfg
    )
    if refusal:
        error(f"{refusal}")
        return 2
    if not _cleartext_token_go_ahead(url, token_env):
        print("nothing was written to config.", file=sys.stderr)
        return 1

    try:
        entry = MCPServerEntry.model_validate(
            {
                "command": command,
                "url": url,
                "token_env": token_env,
                "pass_env": pass_env,
                "startup_timeout_s": _CONNECT_TIMEOUT_S,
            }
        )
    except ValidationError as exc:
        # These are operator flag values, and the entry's own rules (the URL
        # shape above all -- a dropped scheme is the likeliest typo here) live
        # in the model, and reach the operator as one line each rather than a
        # pydantic dump with a saved traceback.
        detail = "; ".join(
            f"{'.'.join(str(part) for part in issue['loc']) or 'entry'}: {issue['msg']}"
            for issue in exc.errors()
        )
        error(f"{name}: {detail}")
        return 2
    env = detect_env()
    isolation = resolve_isolation(cfg.sandbox.isolation, env)
    if command and isolation == "none":
        # No jail means no read-only workspace to probe under, and a probe
        # never runs a server unconfined in the repository: the entry is
        # written unproved, said out loud.
        warn(f"{name} not probed: no jail ({no_jail_cause(cfg, env)}); a run starts it unconfined.")
    else:
        rc = _prove(cfg, name, entry, isolation)
        if rc is not None:
            return rc

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
        error(f"{written}")
        return 2
    print(f"\n{_written_line(effective, name, to_repo)}")
    # The master switch is separate and off by default, so say so rather than
    # flipping a security-relevant default on the operator's behalf.
    print(f"enable MCP for runs with:  {_enable_command(to_repo)}")
    return 0


def _prove(cfg: Config, name: str, entry: MCPServerEntry, isolation: IsolationLevel) -> int | None:
    """Start the server as a run would (its sandbox, the workspace bound
    read-only: a probe never writes the repository) and print its tools; the
    exit code when it gave no proof, else None."""
    try:
        spec = mcp_server_spec(cfg, Path.cwd(), isolation, name, entry, readonly=True)
    except JailUnavailableError as exc:
        error(f"{exc}")
        return 2
    print(f"[agent6] {_describe(spec)} ...", file=sys.stderr)
    tools, failure = _probe(spec)
    if failure:
        _report_no_answer(name, list(entry.command), isolation, failure)
        return 1
    if not tools:
        error(f"{name} started but exposed no tools; nothing was written.")
        return 1
    print(f"\n{name}: {tool_count(len(tools))}")
    for tool in tools:
        # The server chose this text. Collapsing whitespace stops a forged
        # extra line; dropping the other control characters stops ESC
        # sequences repainting the operator's terminal.
        summary = "".join(c for c in " ".join(tool.description.split()) if c.isprintable())[:80]
        print(f"  mcp__{name}__{tool.tool_name}{'  ' + summary if summary else ''}")
    return None


def _repo_flag(to_repo: bool) -> str:
    """The `--repo ` a command line needs to target the repo config, or ""."""
    return "--repo " if to_repo else ""


def _layers_holding(effective: EffectiveConfig, name: str) -> set[str]:
    """The config layers whose own file declares `[mcp.servers.<name>]`."""
    return {
        layer.name
        for layer in effective.layers
        if name in layer.data.get("mcp", {}).get("servers", {})
    }


def _written_line(effective: EffectiveConfig, name: str, to_repo: bool) -> str:
    """Where the entry went, and what that means beside an entry the other
    layer holds: the repo layer wins over the global one."""
    holders = _layers_holding(effective, name)
    target, other = ("repo", "global") if to_repo else ("global", "repo")
    if target in holders:
        note = f", replacing {name}"
    elif other in holders and to_repo:
        note = f"; it shadows the global config's entry for {name}"
    elif other in holders:
        note = f"; the repo config's entry for {name} keeps winning"
    else:
        note = ""
    return f"written to the {target} config{note}."


def _enable_command(to_repo: bool) -> str:
    return f"agent6 config set {_repo_flag(to_repo)}mcp.enabled true"


def cmd_mcp_remove(name: str, *, to_repo: bool = False, config_path: Path | None = None) -> int:
    """Drop `[mcp.servers.<name>]` from the global (or `--repo`) config.

    The inverse of `connect`, and the only way to drop a server: the entry is a
    table rather than a leaf, so `config unset` cannot name it, and unsetting
    its `command`/`url` is refused because an entry needs exactly one of them.
    """
    effective = load_effective(Path.cwd(), config_path)
    holders = _layers_holding(effective, name)
    target, other = ("repo", "global") if to_repo else ("global", "repo")
    if target not in holders:
        elsewhere = (
            f"the {other} config declares it: agent6 mcp remove {_repo_flag(not to_repo)}{name}"
            if other in holders
            else "agent6 mcp list shows the configured servers"
        )
        error(f"no {name!r} in the {target} config ({elsewhere}).")
        return 2
    res = unset_config_table(Path.cwd(), f"mcp.servers.{name}", to_repo=to_repo)
    if res.error is not None:
        error(f"removing {name} left an invalid config:\n{res.error}")
        return 2
    if not res.removed:
        # The layer declares it, but not as a `[mcp.servers.<name>]` header the
        # line surgery can delete (a dotted key, an inline table), so the entry
        # is still live and the operator is told so.
        path = repo_config_path_for(Path.cwd()) if to_repo else global_config_path()
        error(
            f"{name} is not written as a [mcp.servers.{name}] table in {path};"
            " it lives in a dotted key or an inline table, which this verb does not"
            " rewrite. Edit that file by hand."
        )
        return 2
    print(f"removed {name} from the {target} config")
    if other in holders:
        print(f"note: the {other} config's entry for {name} applies now")
    return 0


def cmd_mcp_list(config_path: Path | None = None) -> int:
    """The configured servers and how each is reached. Reads config only: it
    never starts anything, so it answers instantly and says nothing about
    whether a server currently works (`agent6 check mcp` does that)."""
    effective = load_effective(Path.cwd(), config_path)
    cfg = effective.config
    if not cfg.mcp.servers:
        print("no MCP servers configured. Add one with `agent6 mcp connect <name> ...`.")
        return 0
    # The switch goes where the servers are: a repo-only setup is enabled
    # for this repository, never for every repository on the machine.
    layers = {
        layer
        for leaf, layer in effective.sources.items()
        if leaf.startswith("mcp.servers.") and layer != "default"
    }
    state = "enabled" if cfg.mcp.enabled else f"DISABLED ({_enable_command(layers == {'repo'})})"
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
