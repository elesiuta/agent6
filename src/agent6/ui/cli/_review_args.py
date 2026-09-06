# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Parser builders for `review`/`check`/`system`: read-only introspection of
the repo (diff review, pre-flight checks) and privileged host/OS setup
(the AppArmor profile) that a strict sandbox needs."""

from __future__ import annotations

import argparse

from agent6.ui.cli._common import _add_config_flag, _sub


def _add_check_parser(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    check_p = _sub(
        sub,
        "check",
        help=(
            "Pre-flight checks: sandbox, config (provider keys included), boundaries,"
            " MCP, verify_command. Read-only; safe on any clean repo."
        ),
    )
    check_p.add_argument(
        "section",
        nargs="?",
        default="all",
        choices=("all", "sandbox", "config", "boundaries", "mcp", "verify"),
        help=(
            "Limit the report to one section. 'all' (default) runs every check"
            " and prints a single PASS/FAIL summary."
        ),
    )
    _add_config_flag(check_p)


def _add_system_parser(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    # `agent6 system <component> <action>`: privileged host/OS setup (uses sudo).
    system_p = _sub(
        sub,
        "system",
        help="Host/OS setup that needs privileges, and uses sudo to get them."
        " Installs or removes the AppArmor profile the strict sandbox needs.",
    )
    system_sub = system_p.add_subparsers(
        dest="system_command", required=True, metavar="<subcommand>"
    )
    apparmor_p = _sub(
        system_sub,
        "apparmor",
        help="Install/remove the agent6-jail AppArmor profile (Ubuntu 24.04+: lets the"
        " strict sandbox use user namespaces).",
    )
    apparmor_p.add_argument(
        "action",
        choices=("install", "remove", "status"),
        help="install the profile and reload AppArmor, remove it, or report its state.",
    )


def _add_review_parser(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    review_p = _sub(
        sub,
        "review",
        help="Read-only code review of a diff (working tree, branch-vs-base, or arbitrary range).",
    )
    review_p.add_argument(
        "--base",
        default="",
        help="Base ref. Default: review uncommitted changes (working tree vs HEAD).",
    )
    review_p.add_argument(
        "--head",
        default="HEAD",
        help="Head ref (default: HEAD). Only used when --base is set.",
    )
    review_p.add_argument(
        "--path",
        dest="paths",
        action="append",
        default=[],
        metavar="PATH",
        help="Restrict the diff to PATH (repeatable; forwarded to `git diff -- PATH...`).",
    )
    review_p.add_argument(
        "--model",
        default="",
        help=(
            "Override the reviewer model for this one-shot review, under the"
            " [models.reviewer] route's provider (e.g. claude-sonnet-4-5 for a cheaper"
            " read). Default: that route's model."
        ),
    )
    review_p.add_argument(
        "--reviewers",
        type=int,
        default=0,
        metavar="N",
        help=(
            "Run an adversarial review panel of N grounded reviewers instead of one"
            " freeform review. Findings are grounded against the diff (only real,"
            " block-eligible problems gate). 0 (default) = the classic single review."
            " With [review].seats configured, N only turns the panel on: the seats"
            " decide the reviewers."
        ),
    )
    review_p.add_argument(
        "--personas",
        default="",
        help=(
            "Comma-separated adversarial stances for the panel seats, cycled across"
            " --reviewers (e.g. 'security,correctness,tests'). Default: a built-in set;"
            " ignored when [review].seats is configured."
        ),
    )
