# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Prompt scaffolding for `agent6 machine create`.

`machine create` is an ordinary jailed agent6 loop whose job is to *draft*
a `.asm.toml` state machine from a natural-language task. This module holds
the prompt-assembly pieces of that flow: the per-attempt draft→check→fix
prompt, built around the grammar reference in `agent6.prompts.machine`.

It deliberately imports nothing from the workflow stack: the orchestration
(running the agent loop, validating with `load_machine`, writing the draft)
lives in `app/machine/create.py`, which already depends on both
`agent6.machine` and `agent6.workflows`. Keeping this module pure keeps the
tach graph acyclic.
"""

from __future__ import annotations

from agent6.prompts.machine import MACHINE_AUTHOR_GUIDE

__all__ = ["MACHINE_AUTHOR_GUIDE", "build_authoring_prompt"]


# The keys the authoring agent uses to return its draft: the `.asm.toml` source
# and the helper scripts its `tool` states reference (a map of bundle-relative
# path -> file content). Both are written by `machine create`.
def build_authoring_prompt(
    task: str,
    *,
    attempt: int,
    diagnostics: list[str] | None = None,
) -> str:
    """Assemble the user-task prompt for one draft→check→fix attempt.

    The first attempt carries the grammar guide, the operator's task, and the
    production-readiness rules. On a retry only the validation diagnostics are
    appended: the draft itself is in the workspace, where the agent reads and
    patches the files it wrote rather than re-deriving them from a transcript.
    """
    parts = [
        MACHINE_AUTHOR_GUIDE,
        "",
        "## Your task",
        "",
        "Author ONE complete, valid `.asm.toml` machine for this request:",
        "",
        task.strip(),
        "",
    ]
    parts += [
        "## Where to write it",
        "",
        "This workspace is yours and starts empty: write the machine file and its"
        ' scripts here with `apply_edit`, `kind="create"` for each new file,'
        " and read them back when you need to."
        " Write exactly ONE `<machine-name>.asm.toml` at the workspace root, plus"
        " every `scripts/...` file its `tool` states reference (and, for any script"
        " with a seam, its `scripts/<name>_test.py` companion). Call `finish_session`"
        " when the bundle is complete; agent6 validates what is on disk and hands"
        " you back any problems.",
        "",
        "Make each script PRODUCTION-READY for the real task: it reads live inputs"
        " from their real source (real HTTP via stdlib `urllib`), reads any"
        " secrets from the environment (never hard-coded), sets"
        ' `network = "host"` on its state if it touches the network, prints'
        " ONE JSON object on stdout matching its `output_schema`, and exits 0 on"
        " success. Type-annotate it and keep it lint-clean: `machine create` runs"
        " ruff + ty and rejects it otherwise. For every script with an external"
        " seam (network/clock/files), ALSO write a `scripts/<name>_test.py` that"
        " mocks the seam and asserts the contract; these run offline in a"
        " no-network jail so the operator can simulate the machine without live"
        " services. Put a one-line rationale per state in `summary`.",
    ]
    if diagnostics:
        joined = "\n".join(f"  - {problem}" for problem in diagnostics)
        parts.extend(
            [
                "",
                f"## Attempt {attempt}: fix the draft in this workspace",
                "",
                "Your draft did not pass validation. The diagnostics were:",
                "",
                joined,
                "",
                "Read the files you wrote, change ONLY what the diagnostics name,"
                " and finish again.",
            ]
        )
    return "\n".join(parts)
