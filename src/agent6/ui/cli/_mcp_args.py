# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Parser builder for `mcp` and its subcommands."""

from __future__ import annotations

import argparse

from agent6.ui.cli._common import _sub
from agent6.ui.cli.completers import _complete_mcp_servers


def _add_mcp_server_parsers(mcp_sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """`connect` and `list`, added to the existing `mcp` group beside `serve`."""
    connect = _sub(
        mcp_sub,
        "connect",
        help="Add an MCP server, after proving it answers and listing its tools.",
    )
    connect.add_argument("name", help="The tool prefix: its tools appear as mcp__<name>__<tool>.")
    connect.add_argument(
        # Not named "command": a positional's name is its dest, and the root
        # parser's subcommand verb already owns args.command.
        "server_command",
        nargs="*",
        metavar="ARGV",
        help=("argv for a server to spawn (put it after `--`). Exactly one of this or --url."),
    )
    connect.add_argument(
        "--url",
        default="",
        metavar="URL",
        help=(
            "An http(s) endpoint of a server you run, which agent6 only connects"
            " to. Exactly one of this or a command."
        ),
    )
    connect.add_argument(
        "--token-env",
        dest="token_env",
        default="",
        metavar="VAR",
        help=(
            "For --url: the environment variable holding the bearer token."
            " The config records the name; the value never enters it."
        ),
    )
    connect.add_argument(
        "--pass-env",
        dest="pass_env",
        action="append",
        default=[],
        metavar="VAR",
        help=(
            "For a spawned server: an environment variable it needs, by name"
            " (repeatable). Everything else is agent6's curated base, which"
            " never carries a provider key."
        ),
    )
    connect.add_argument(
        "--repo",
        dest="to_repo",
        action="store_true",
        help="Write to the per-repo config instead of the global one.",
    )

    remove = _sub(
        mcp_sub,
        "remove",
        help="Drop a configured MCP server (the inverse of connect).",
    )
    remove_name = remove.add_argument("name", help="The server to remove, as `mcp list` names it.")
    remove_name.completer = _complete_mcp_servers  # type: ignore[attr-defined]
    remove.add_argument(
        "--repo",
        dest="to_repo",
        action="store_true",
        help="Remove from the per-repo config instead of the global one.",
    )

    _sub(mcp_sub, "list", help="The configured MCP servers and how each is reached.")
