# SPDX-License-Identifier: Apache-2.0
"""Stdlib-only checker for the code-fixer machine.

Imports the ``stats`` module from the tree it runs in and checks ``median``
against a few cases. Prints one JSON line ``{"passed": bool, "summary": str}``
and always exits 0, so the machine's tool state routes on the captured
``passed`` flag, not on the process exit code. In a machine with run states,
tool states run in the machine's own tree, so this sees the agent's committed
fix with no extra plumbing.
"""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path
from typing import Any

CASES: list[tuple[list[float], float]] = [
    ([3, 1, 2], 2.0),
    ([1, 2, 3, 4], 2.5),
    ([5], 5.0),
    ([4, 1], 2.5),
    ([10, 2, 8, 4, 6], 6.0),
]


def _emit(passed: bool, summary: str) -> int:
    print(json.dumps({"passed": passed, "summary": summary}))
    return 0


def main() -> int:
    sys.path.insert(0, str(Path.cwd()))
    try:
        module = importlib.import_module("stats")
    except (ImportError, SyntaxError) as exc:
        return _emit(False, f"could not import stats: {exc}")
    median: Any = getattr(module, "median", None)
    if not callable(median):
        return _emit(False, "stats.median is missing or not callable")
    for xs, want in CASES:
        try:
            got = median(list(xs))
        except Exception as exc:  # noqa: BLE001 - the candidate may raise anything
            return _emit(False, f"median({xs}) raised {exc!r}")
        if got != want:
            return _emit(False, f"median({xs}) = {got!r}, want {want!r}")
    return _emit(True, f"all {len(CASES)} median cases pass")


if __name__ == "__main__":
    raise SystemExit(main())
