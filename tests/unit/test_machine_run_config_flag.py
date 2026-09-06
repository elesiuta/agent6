# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""`agent6 --config FILE machine run` honours the flag layer."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

_MACHINE = """\
machine = "tiny"
version = 1
initial = "done"

[budget]
max_transitions = 5

[states.done]
kind   = "terminal"
status = "ok"
reason = "nothing to do"
"""


def test_machine_run_reads_the_explicit_config_layer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """docs/config.md presents --config as a general layer; machine run
    resolved without it, so the file the operator named was ignored."""
    from agent6.app.machine.run import run_machine

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "g"))
    explicit = tmp_path / "explicit.toml"
    explicit.write_text("[machine]\nsnapshot_keep = 41\n", encoding="utf-8")
    mfile = tmp_path / "tiny.asm.toml"
    mfile.write_text(_MACHINE, encoding="utf-8")

    seen: list[int] = []
    from agent6.app.machine import run as run_mod

    real = run_mod.load_effective_with_overlay

    def spy(repo_root: Path, overlay: dict[str, object], **kw: object):
        eff = real(repo_root, overlay, **kw)  # pyright: ignore[reportArgumentType]
        seen.append(eff.config.machine.snapshot_keep)
        return eff

    monkeypatch.setattr(run_mod, "load_effective_with_overlay", spy)
    frontend = MagicMock()
    frontend.reporter = MagicMock()
    run_machine(mfile, frontend, config_path=explicit)
    assert seen and seen[0] == 41, "the --config layer never reached machine run"
