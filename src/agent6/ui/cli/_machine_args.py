# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Parser builder for `machine` and its subcommands: author-time tooling for
agent6 state machines (.asm.toml) -- list/check/test/graph/run/status/poke/
replay/create."""

from __future__ import annotations

import argparse
from pathlib import Path

from agent6.ui.cli._common import MACHINE_ID_HELP, _add_sandbox_flags, _sub
from agent6.ui.cli.completers import _complete_machine_ids


def _add_machine_parser(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    machine_p = _sub(
        sub,
        "machine",
        help=(
            "Author-time tooling for agent6 state machines (.asm.toml); bare `machine`"
            " lists this repo's machines."
        ),
    )
    machine_sub = machine_p.add_subparsers(
        dest="machine_command", required=True, metavar="<subcommand>"
    )
    _sub(
        machine_sub,
        "list",
        help=(
            "List this repo's machines (`agent6 machine`, or `machine list`): each"
            " instance's status and state, and each authored .asm.toml's spec validity."
        ),
    )
    machine_check = _sub(
        machine_sub,
        "check",
        help=(
            "Validate a .asm.toml machine file: parse, type-check, reachability,"
            " bundle paths, and static script lint/types (ruff + ty). No execution."
        ),
    )
    machine_check.add_argument("file", type=Path, help="Path to the .asm.toml machine file.")
    machine_test = _sub(
        machine_sub,
        "test",
        help=(
            "Simulate a machine offline: everything `check` does, plus run the"
            " bundle's scripts/*_test.py mocks in a no-network jail, plus a pure"
            " dry-run (synthesized facts, branch routing against a fixture)."
            " No provider calls, no real network."
        ),
    )
    machine_test.add_argument("file", type=Path, help="Path to the .asm.toml machine file.")
    machine_test.add_argument(
        "--blackboard",
        type=Path,
        default=None,
        metavar="FIXTURE.toml",
        help="TOML fixture of variable values, overlaid on defaults for branch routing.",
    )
    machine_graph = _sub(
        machine_sub,
        "graph",
        help="Emit the machine as a state diagram (mermaid or Graphviz dot).",
    )
    machine_graph.add_argument("file", type=Path, help="Path to the .asm.toml machine file.")
    machine_graph.add_argument(
        "--format",
        choices=("mermaid", "dot"),
        default="mermaid",
        help="Diagram format (default: mermaid).",
    )
    machine_run = _sub(
        machine_sub,
        "run",
        help="Run (or resume) a machine, driving its states to a terminal one.",
    )
    machine_run.add_argument("file", type=Path, help="Path to the .asm.toml machine file.")
    machine_run.add_argument(
        "--exit-on-wait",
        action="store_true",
        help=(
            "Persist the next wake instant and exit 0 (status 'waiting') at the first"
            " not-ready wait instead of blocking, for an external scheduler to resume."
        ),
    )
    # --auto-approve is the operator's per-invocation run_command grant, the
    # same flag `run` carries: an unattended machine otherwise auto-denies
    # (machine [config] overlays must not set sandbox.*, so the grant can only
    # come from the operator at the keyboard or the repo config).
    _add_sandbox_flags(machine_run)
    machine_status = _sub(
        machine_sub,
        "status",
        help="Report a machine instance's current state, spend, and next wake. Read-only.",
    )
    machine_status_id = machine_status.add_argument("machine_id", help=MACHINE_ID_HELP)
    machine_status_id.completer = _complete_machine_ids  # type: ignore[attr-defined]
    machine_poke = _sub(
        machine_sub,
        "poke",
        help="Signal a waiting machine to wake on its next check (drops a signal file).",
    )
    machine_poke_id = machine_poke.add_argument("machine_id", help=MACHINE_ID_HELP)
    machine_poke_id.completer = _complete_machine_ids  # type: ignore[attr-defined]
    machine_poke_payload = machine_poke.add_mutually_exclusive_group()
    machine_poke_payload.add_argument(
        "--data",
        metavar="JSON",
        help="A JSON value delivered to the waking wait as its poke payload"
        " (readable by the next tool at $AGENT6_MACHINE_DATA_DIR/poke.json).",
    )
    machine_poke_payload.add_argument(
        "--message",
        metavar="TEXT",
        help="Shorthand for --data with a JSON string payload.",
    )
    machine_stop = _sub(
        machine_sub,
        "stop",
        help=(
            "Park a running machine at its next transition boundary; it finishes the step"
            " in flight."
        ),
    )
    machine_stop_id = machine_stop.add_argument("machine_id", help=MACHINE_ID_HELP)
    machine_stop_id.completer = _complete_machine_ids  # type: ignore[attr-defined]
    machine_replay = _sub(
        machine_sub,
        "replay",
        help="Deterministically replay a machine's journal offline (no world I/O).",
    )
    machine_replay_id = machine_replay.add_argument("machine_id", help=MACHINE_ID_HELP)
    machine_replay_id.completer = _complete_machine_ids  # type: ignore[attr-defined]

    machine_create = _sub(
        machine_sub,
        "create",
        help="Draft a .asm.toml machine from a natural-language task (LLM-assisted).",
    )
    machine_create.add_argument("task", help="Natural-language description of the loop to author.")
    machine_create.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help=(
            "Write the draft here (overwriting freely). Default: <machine-name>.asm.toml"
            " in cwd, which is never overwritten."
        ),
    )
    machine_create.add_argument(
        "--max-attempts",
        type=int,
        default=3,
        metavar="N",
        help="Maximum draft->check->fix attempts before giving up (default: 3).",
    )
