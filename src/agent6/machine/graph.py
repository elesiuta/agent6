# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Render a validated `MachineSpec` as a state diagram.

Two pure renderers over the same validated graph: `mermaid`
(`stateDiagram-v2`, the default) and Graphviz `dot`. Both consume the
edges already computed by `agent6.machine.model`, so a diagram is just a
render of a machine that has already passed `machine check`.
"""

from __future__ import annotations

from agent6.machine.model import MachineSpec, TerminalState, edges

__all__ = ["render_dot", "render_mermaid"]


def _terminals(spec: MachineSpec) -> list[str]:
    return [name for name, state in spec.states.items() if isinstance(state, TerminalState)]


def _clean_label(label: str) -> str:
    return " ".join(label.split())


def render_mermaid(spec: MachineSpec) -> str:
    lines = ["stateDiagram-v2", f"    [*] --> {spec.initial}"]
    for edge in edges(spec):
        lines.append(f"    {edge.src} --> {edge.dst}: {_clean_label(edge.label)}")
    for terminal in _terminals(spec):
        lines.append(f"    {terminal} --> [*]")
    return "\n".join(lines) + "\n"


def _dot_escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace('"', '\\"')


def render_dot(spec: MachineSpec) -> str:
    lines = [
        f'digraph "{_dot_escape(spec.machine)}" {{',
        "    rankdir=LR;",
        "    __start__ [shape=point];",
    ]
    for terminal in _terminals(spec):
        lines.append(f'    "{_dot_escape(terminal)}" [shape=doublecircle];')
    lines.append(f'    __start__ -> "{_dot_escape(spec.initial)}";')
    for edge in edges(spec):
        label = _dot_escape(_clean_label(edge.label))
        lines.append(
            f'    "{_dot_escape(edge.src)}" -> "{_dot_escape(edge.dst)}" [label="{label}"];'
        )
    lines.append("}")
    return "\n".join(lines) + "\n"
