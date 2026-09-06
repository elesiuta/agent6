# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""One sentence per shared argument across the CLI.

Fifteen commands take a session id and nine wordings described it (`unique
prefix`, `exact or prefix`, `Defaults to the most recent`...), and `sessions
show` named no prefix rule while accepting one; six write verbs carried five
wordings of `--repo`. The constants in `ui/cli/_common.py` are the one owner,
and this walk keeps every site on them.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterator

from agent6.ui.cli._common import MACHINE_ID_HELP, REPO_FLAG_HELP, SESSION_ID
from agent6.ui.cli.parser import build_parser

_REPO_HELPS = {REPO_FLAG_HELP, "Remove from the per-repo config instead of the global config."}


def _walk(parser: argparse.ArgumentParser, path: str) -> Iterator[tuple[str, argparse.Action]]:
    for action in parser._actions:  # pyright: ignore[reportPrivateUsage]
        if isinstance(action, argparse._SubParsersAction):  # pyright: ignore[reportPrivateUsage]
            for name, sub in action.choices.items():
                yield from _walk(sub, f"{path} {name}")
        else:
            yield path, action


def test_every_id_and_repo_argument_shares_one_sentence() -> None:
    seen = {"session_id": 0, "machine_id": 0, "--repo": 0}
    for path, action in _walk(build_parser(), "agent6"):
        help_text = action.help or ""
        if action.dest in ("session_id", "target") and not action.option_strings:
            # fork names its source run and plan show its plan; the rule is the same.
            assert "or unambiguous prefix" in help_text, (path, help_text)
            assert help_text.startswith(SESSION_ID) or path.startswith(
                ("agent6 fork", "agent6 plan")
            ), (
                path,
                help_text,
            )
            seen["session_id"] += 1
        elif action.dest == "machine_id" and not action.option_strings:
            assert help_text == MACHINE_ID_HELP, (path, help_text)
            seen["machine_id"] += 1
        elif "--repo" in action.option_strings:
            assert help_text in _REPO_HELPS, (path, help_text)
            seen["--repo"] += 1
    assert seen["session_id"] >= 12 and seen["machine_id"] >= 4 and seen["--repo"] >= 6, seen
