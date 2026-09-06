# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""`agent6 tui <target>` opens what `attach --tui <target>` opens.

`web` took a target and `tui` did not: `agent6 tui <id>` was an argparse
refusal and completion offered only `-h`."""

from __future__ import annotations

import pytest

from agent6.ui.cli import main


def test_tui_with_a_target_opens_it_and_without_one_opens_the_hub(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[tuple[str, object]] = []

    def _watch(target: str, **kw: object) -> int:
        seen.append(("watch", (target, kw.get("tui"))))
        return 0

    def _hub(_config: object) -> int:
        seen.append(("hub", None))
        return 0

    monkeypatch.setattr("agent6.ui.cli.watch._cmd_watch_target", _watch)
    monkeypatch.setattr("agent6.ui.cli.plan_watch._cmd_tui", _hub)
    assert main(["tui", "brisk-otter-AAAAAA"]) == 0
    assert main(["tui"]) == 0
    assert seen == [("watch", ("brisk-otter-AAAAAA", True)), ("hub", None)]
