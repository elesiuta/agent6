# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Parser builder for `skills` and its subcommands: install/update/list/
enable/disable/remove operator-installed skill packs (SKILL.md)."""

from __future__ import annotations

import argparse

from agent6.ui.cli._common import REPO_FLAG_HELP, _sub
from agent6.ui.cli.completers import _complete_skills


def _add_skills_parser(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    skills_p = _sub(
        sub,
        "skills",
        help=(
            "Manage operator-installed skills (SKILL.md packs); a bare `agent6 skills` lists them."
        ),
    )
    skills_sub = skills_p.add_subparsers(
        dest="skills_command", required=True, metavar="<subcommand>"
    )
    sk_install = _sub(
        skills_sub,
        "install",
        help="Install skills from a SKILL.md URL, a git repository URL, or a local path.",
    )
    sk_install.add_argument("url", help="Direct SKILL.md URL, git repo URL, or local path.")
    sk_install.add_argument(
        "--force", action="store_true", help="Replace an already-installed skill of the same name."
    )
    sk_update = _sub(skills_sub, "update", help="Re-fetch installed skills from their origins.")
    sk_update_name = sk_update.add_argument(
        "name", nargs="?", default="", help="Skill to update (default: all with an origin)."
    )
    sk_update_name.completer = _complete_skills  # type: ignore[attr-defined]
    _sub(skills_sub, "list", help="List installed skills with state and origin.")
    sk_enable = _sub(skills_sub, "enable", help="Re-enable a skill (or promote it to always-on).")
    sk_enable_name = sk_enable.add_argument("name", help="Skill name.")
    sk_enable_name.completer = _complete_skills  # type: ignore[attr-defined]
    sk_enable.add_argument(
        "--always",
        action="store_true",
        help="Inject the skill's full text into every run's system prompt instead of the index.",
    )
    sk_enable.add_argument("--repo", action="store_true", help=REPO_FLAG_HELP)
    sk_disable = _sub(skills_sub, "disable", help="Drop a skill from the index and use_skill.")
    sk_disable_name = sk_disable.add_argument("name", help="Skill name.")
    sk_disable_name.completer = _complete_skills  # type: ignore[attr-defined]
    sk_disable.add_argument("--repo", action="store_true", help=REPO_FLAG_HELP)
    sk_remove = _sub(skills_sub, "remove", help="Delete an installed skill from the data dir.")
    sk_remove_name = sk_remove.add_argument("name", help="Skill name.")
    sk_remove_name.completer = _complete_skills  # type: ignore[attr-defined]
