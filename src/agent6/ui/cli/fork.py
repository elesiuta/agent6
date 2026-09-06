# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""`agent6 fork`: adapt argv, materialize the fork (`agent6.app.fork`), then
(unless `--no-run`) continue the new run from its forked turn over the resume
path."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from agent6.app._setup import BudgetOverrides, SandboxOverrides
from agent6.app.fork import create_fork
from agent6.app.preflight import headless_approval_refusal
from agent6.app.resume import resume_task
from agent6.config import Config
from agent6.types import session_kind
from agent6.ui.cli.run import session_frontend


def _cmd_fork(
    config_path: Path | None,
    source_session_id: str,
    *,
    at_turn: int | None = None,
    new_session_id: str = "",
    no_run: bool = False,
    tui: bool = False,
    budget_overrides: BudgetOverrides | None = None,
    sandbox_overrides: SandboxOverrides | None = None,
    steer: str = "",
) -> int:
    """Create a new run cloned from *source_session_id* at checkpoint *at_turn*.

    Default: fork from the latest checkpoint and immediately continue the new run
    from that turn (resume-like); `--steer` seeds the fresh direction at its
    first safe boundary. `--no-run` just creates the fork dir.
    """
    if no_run and steer.strip():
        print(
            "ERROR: --steer seeds the immediate continuation, which --no-run skips."
            " Drop --no-run, or start the fork later with `agent6 resume <id> --steer ...`.",
            file=sys.stderr,
        )
        return 2
    frontend = session_frontend(config_path)

    def refuse_continuation(cfg: Config, mode: str) -> str | None:
        # The resume below would refuse the same way, after the fork existed.
        return headless_approval_refusal(
            cfg,
            tui_enabled=frontend.should_spawn_tui(tui, False, mode),
            away=os.environ.get("AGENT6_DETACHED_AWAY", ""),
            can_ask=frontend.capabilities.can_ask,
            clamped=session_kind(mode).clamps_commands,
        )

    child_id, rc = create_fork(
        config_path,
        source_session_id,
        at_turn=at_turn,
        new_session_id=new_session_id,
        cwd=Path.cwd(),
        sandbox_overrides=None if no_run else sandbox_overrides,
        refuse_continuation=None if no_run else refuse_continuation,
    )
    if rc != 0:
        return rc

    if no_run:
        print(f"[agent6] fork created (not started): {child_id}", file=sys.stderr)
        print(f"  resume it with: agent6 resume {child_id}", file=sys.stderr)
        return 0

    # Continue the new run from turn N by reusing the resume path. The fork just
    # cloned the checkpoint (its head_sha) and cut agent6/<child> at that same
    # sha, so the resume head guard passes by construction; force stays off so a
    # real mismatch (a broken fork) still refuses.
    return resume_task(
        config_path,
        child_id,
        frontend=frontend,
        force=False,
        tui=tui,
        budget_overrides=budget_overrides,
        sandbox_overrides=sandbox_overrides,
        steer=steer,
    )
