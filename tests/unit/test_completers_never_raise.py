# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""No completer raises into the operator's shell.

argcomplete runs these on Tab, inside the shell, with nowhere to show an error:
an exception there is a traceback dumped over the command line. Several guarded
themselves ad hoc and several did not, which is the same gap in as many places.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest

from agent6.ui.cli import completers

_COMPLETERS = [
    (name, fn)
    for name, fn in vars(completers).items()
    if name.startswith("_complete_") and inspect.isfunction(fn)
]


def test_there_are_completers_to_check() -> None:
    assert len(_COMPLETERS) >= 10, [n for n, _ in _COMPLETERS]


# The completers that consult the per-repo state dir when called with a bare
# prefix. The other nine never reach `state_dir` (they return early or read
# config only), so parametrizing them here forced nothing; the decorator test
# below carries their never-raise promise.
_STATE_DIR_CONSUMERS = [
    "_complete_session_ids",
    "_complete_resumable_ids",
    "_complete_plan_session_ids",
    "_complete_machine_ids",
    "_complete_watch_targets",
    "_complete_machine_files",
]


@pytest.mark.parametrize("name", _STATE_DIR_CONSUMERS)
def test_an_unresolvable_state_dir_does_not_reach_the_shell(
    name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The realistic failure: the config does not parse, so resolving the state
    dir raises. Forced directly -- pointing cwd at a bad config passed without
    ever reaching the raising path, which proved nothing.
    """
    from agent6.config import ConfigError
    from agent6.ui.cli import _common

    calls: list[Path] = []

    def _boom(root: Path) -> Path:
        calls.append(root)
        raise ConfigError("config is not valid TOML")

    monkeypatch.setattr(completers, "state_dir", _boom)
    monkeypatch.setattr(_common, "state_dir", _boom)

    fn = getattr(completers, name)
    result = fn("", parsed_args=None)
    assert isinstance(result, list), f"{name} returned {result!r}"
    assert calls, f"{name} never consulted the state dir; drop it from _STATE_DIR_CONSUMERS"


def test_any_completer_bug_yields_no_suggestions_not_a_traceback() -> None:
    """The decorator's promise is its name: never an exception, not "never
    the three exception types someone predicted". A KeyError from a bug is a
    traceback over the command line all the same."""
    from agent6.ui.cli.completers import _never_raises  # pyright: ignore[reportPrivateUsage]

    @_never_raises
    def boom(prefix: str, **_kw: object) -> list[str]:
        raise KeyError("bug")

    assert boom("") == []
