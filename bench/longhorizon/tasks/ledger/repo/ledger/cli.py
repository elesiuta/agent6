"""Dispatch: the command table is generated into ledger/_commands.py."""

from __future__ import annotations

import importlib
import sys

from ledger._commands import COMMANDS


def run(argv: list[str]) -> int:
    if not argv or argv[0] in ("-h", "--help"):
        for name, (_handler, help_) in COMMANDS.items():
            print(f"  {name:12} {help_}")
        return 0
    name, *rest = argv
    if name not in COMMANDS:
        print(f"unknown command {name!r}", file=sys.stderr)
        return 2
    module, func = COMMANDS[name][0].split(":")
    handler = getattr(importlib.import_module(module), func)
    for line in handler(rest):
        print(line)
    return 0


if __name__ == "__main__":
    sys.exit(run(sys.argv[1:]))
