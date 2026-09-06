# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""CLI adapter for `agent6 run --parallel` and the coordinator `/parallel`
dispatch.

The fan-out / coordinator pipeline is headless in `agent6.app.parallel`; this
module is the front-end seam. It supplies the `LaneRuntime` the pipeline drives
(the detached process spawn from `ui.spawn`, and the reviewer provider +
judging spinner from `_compare`), and holds the CLI-side preflight +
refusal messages for `run --parallel`. run.py routes here; run.py / resume.py
wire the coordinator spawner from here.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path

from agent6.app.parallel import (
    LaneRuntime,
    ParallelError,
    build_lane_specs,
    run_parallel,
)
from agent6.app.preflight import budget_preflight
from agent6.config import Config
from agent6.directive import DirectiveError
from agent6.git_ops import GitError, modified_paths
from agent6.models.validate import refusal_message, validate_spec_models, warning_message
from agent6.paths import state_dir
from agent6.sessions.id import friendly_token
from agent6.ui.cli._common import error, refuse, warn
from agent6.ui.cli._compare import _judging_status, _reviewer_provider
from agent6.ui.spawn import agent6_exe, spawn_and_locate


def lane_runtime() -> LaneRuntime:
    """The front-end primitives the parallel pipeline drives: the detached
    process spawn (`ui.spawn`) and the reviewer provider + judging spinner
    (`_compare`). Injected so `agent6.app` never imports `agent6.ui`. Lane
    liveness/stop is the run-dir bridge, which `agent6.app.parallel` imports
    directly (not part of this seam)."""

    def spawn(
        argv: list[str],
        cwd: Path,
        *,
        before: set[Path],
        list_dirs: Callable[[], list[Path]],
        env: dict[str, str],
    ) -> tuple[Path | None, str]:
        return spawn_and_locate(
            [agent6_exe(), *argv], cwd, before=before, list_dirs=list_dirs, env=env
        )

    return LaneRuntime(
        spawn=spawn,
        build_provider=_reviewer_provider,
        judging_status=_judging_status,
    )


def _parallel_approval_refusal(cfg: Config) -> str | None:
    """Refuse `--parallel` under `ask`, naming the two coherent choices.

    Lanes run detached and at the same time, so "wait for someone to approve"
    would mean attaching a front-end to each lane in turn -- most of what
    running them in parallel was for. The decision is made once, at launch.
    """
    if cfg.sandbox.run_commands != "ask":
        return None
    return (
        "sandbox.run_commands = 'ask' cannot drive parallel lanes: each lane runs"
        " detached, with nobody to answer its prompts.\n"
        "  --auto-approve   approve every command in every lane\n"
        "  --no-commands    withhold commands from every lane\n"
        "  (a hub has no flags: agent6 config set sandbox.run_commands yes|no, --repo"
        " for this checkout)"
    )


def dispatch_parallel(
    cfg: Config,
    task: str,
    spec: str,
    *,
    cwd: Path,
    max_usd: float | None = None,
    auto_approve: bool = False,
    pins: Sequence[str] = (),
) -> int:
    """Preflight and route `agent6 run --parallel`: refuse an unenforceable
    --max-usd or a dirty origin (lanes clone committed HEAD only), plan the
    lanes, then hand off to the headless `run_parallel`. Called from `run.py`.
    `auto_approve` forwards `--auto-approve` to every lane, same as `max_usd`."""
    origin = cwd
    origin_state = state_dir(origin)
    for err in (budget_preflight(cfg), _parallel_approval_refusal(cfg)):
        if err is not None:
            refuse(f"{err}")
            return 2
    try:
        modified = modified_paths(origin)
    except GitError as exc:
        error(f"{exc}")
        return 2
    if modified and cfg.git.dirty_tree == "ask":
        listed = "\n".join(f"    {p}" for p in modified[:10])
        more = f"\n    ... {len(modified) - 10} more" if len(modified) > 10 else ""
        n = len(modified)
        refuse(
            f"{n} tracked {'file has' if n == 1 else 'files have'} uncommitted"
            f" changes:\n{listed}{more}\n"
            "Lanes clone committed HEAD, so those changes would not reach them. Commit or"
            ' stash them first, or set [git].dirty_tree to "stash" or "include" to fan out'
            " without them."
        )
        return 2

    fanout_id = friendly_token()
    try:
        lanes = build_lane_specs(spec, cfg=cfg, origin=origin, fanout_id=fanout_id)
    except (DirectiveError, ParallelError) as exc:
        refuse(f"{exc}")
        return 2
    # Validate the named models before any clone/spawn (lanes are plain specs so
    # far, no workdir touched): refuse a typo when a cache exists to check
    # against, else warn and proceed (a fresh/offline machine is never blocked).
    verdict = validate_spec_models([ln.model for ln in lanes], cfg)
    if verdict.refused:
        refuse(f"{refusal_message(verdict, directive=False)}")
        return 2
    if verdict.warned:
        warn(f"{warning_message(verdict)}")
    return run_parallel(
        task,
        lanes,
        cfg=cfg,
        origin=origin,
        origin_state=origin_state,
        runtime=lane_runtime(),
        max_usd=max_usd,
        fanout_id=fanout_id,
        auto_approve=auto_approve,
        pins=pins,
    )
