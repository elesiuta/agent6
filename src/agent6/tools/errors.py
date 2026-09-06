# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Tool-layer exceptions, homed here so the handler modules and mcp_server
import them without importing the whole dispatch module."""

from __future__ import annotations


class ToolError(Exception):
    """The LLM tried something the tool layer refused."""


class ToolDenied(ToolError):
    """A tool call refused by POLICY before it ran: the approval gate did not
    approve (a human said no, or the ask-policy auto-denied an unattended run).
    A command, an MCP server's tool and a `fetch` whose host is outside
    `sandbox.fetch_hosts` all raise it. Nothing executed, so the loop's
    sandbox-reachability heuristic must not count it as a tool that "fails in
    the jail", and the repeat-error nudge says "refused, stop retrying" instead
    of "your call is malformed"."""


class OperatorCommandUnexecutable(Exception):
    """An operator-configured verify/metric command could not be executed in the
    jail (not found on PATH /usr/bin:/bin, or a path that escapes the sandbox).

    Distinct from ToolError (which the loop surfaces to the model and continues):
    the model cannot fix the operator's config, so the loop must abort loudly
    rather than let the worker flail against a verify that never actually runs.
    """
