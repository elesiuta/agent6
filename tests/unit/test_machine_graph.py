# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Tests for agent6.machine.graph — mermaid + dot rendering."""

from __future__ import annotations

import re
from pathlib import Path

from agent6.machine._semantics import load_machine
from agent6.machine.graph import render_dot, render_mermaid
from tests.unit.test_machine_model import VALID_MACHINE


def _spec(tmp_path: Path):  # type: ignore[no-untyped-def]
    path = tmp_path / "m.asm.toml"
    path.write_text(VALID_MACHINE, encoding="utf-8")
    return load_machine(path)


def test_mermaid_has_entry_and_terminal(tmp_path: Path) -> None:
    out = render_mermaid(_spec(tmp_path))
    assert out.startswith("stateDiagram-v2\n")
    assert "[*] --> poll" in out
    assert "halt --> [*]" in out
    assert "scan --> have_items: ok" in out
    # A branch clause's predicate is the label as written; `else = true` reads "else".
    assert "have_items --> poll: len(pending) == 0" in out
    assert "have_items --> classify: else" in out


def test_dot_has_start_point_and_terminal_shape(tmp_path: Path) -> None:
    out = render_dot(_spec(tmp_path))
    assert out.startswith('digraph "item-classifier" {')
    assert "__start__ [shape=point];" in out
    assert '"halt" [shape=doublecircle];' in out
    assert '__start__ -> "poll";' in out
    assert '"scan" -> "have_items" [label="ok"];' in out
    assert '"have_items" -> "poll" [label="len(pending) == 0"];' in out
    assert '"have_items" -> "classify" [label="else"];' in out


_DOC = Path(__file__).resolve().parents[2] / "docs" / "state-machines.md"


def test_documented_graph_is_the_rendered_one(tmp_path: Path) -> None:
    """docs/state-machines.md's mermaid block is `machine graph`'s own output
    over the worked example above it. Hand-drawn, it drifted: `nonzero` and
    `timeout` merged into one edge, `urgent and confident` stood in for the
    branch predicate, and `signal` was absent."""
    text = _DOC.read_text(encoding="utf-8")
    diagram = re.search(r"```mermaid\n(.*?)```", text, re.S)
    assert diagram, "no mermaid block on the page"
    blocks = re.findall(r"```toml\n(.*?)```", text[: diagram.start()], re.S)
    example = [b for b in blocks if 'machine = "item-classifier"' in b][-1]
    path = tmp_path / "item-classifier.asm.toml"
    path.write_text(example, encoding="utf-8")
    assert render_mermaid(load_machine(path)) == diagram.group(1)
