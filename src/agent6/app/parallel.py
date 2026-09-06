# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Fan-out orchestrator for `agent6 run --parallel` and the coordinator's
`/parallel` dispatch.

Spawn N isolated lanes -- each a disposable clone of the repo running its own
detached `agent6 run` -- symlink the live lanes into `agent6 sessions` for
visibility, await them, import each finished lane's branch + run dir back into
the origin, then auto-compare and print a ranked report. Nothing is merged: the
operator picks a winner and runs `agent6 sessions merge <id>`.

The origin repo is never mutated (no branch cut, no run dir, no commits) until
`import_run` lands a lane's branch. Clones + lane state are torn down after
import. The heavy git plumbing lives in `workflows.subrun`; the ranking in
`app.compare` over `workflows.judge`; this module orchestrates them over a
`LaneRuntime` -- the process-spawn + run-dir bridge the front-end injects so this
pipeline never imports `agent6.ui`.
"""

from __future__ import annotations

import contextlib
import functools
import json
import os
import shutil
import threading
import time
from collections.abc import Callable, Sequence
from concurrent.futures import FIRST_EXCEPTION, ThreadPoolExecutor
from concurrent.futures import wait as futures_wait
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from agent6.app.compare import (
    BuildProvider,
    JudgingStatus,
    RankOutcome,
    manifest_task,
    print_ranked_candidates,
    rank,
)
from agent6.app.finalize import EXIT_VERIFY_FAILED
from agent6.app.manifest import write_manifest
from agent6.app.reporter import STDIO_REPORTER, Reporter
from agent6.config import Config
from agent6.config.layer import materialize
from agent6.directive import parse_spec
from agent6.git_ops import (
    GitError,
    branch_exists,
    chain_tip,
    checkout_detached,
    diff_since,
    list_chain_refs,
    list_run_branches,
    run_branch_for,
)
from agent6.git_ops import status as git_status
from agent6.memory import merge_decisions
from agent6.models.validate import refusal_message, validate_spec_models, warning_message
from agent6.paths import cache_dir, repo_id, state_dir
from agent6.sessions.ipc import request_stop, steer_answer_is_abort, worker_is_alive
from agent6.sessions.layout import LOGS_NAME, SessionLayout, bucket_dir
from agent6.sessions.manifest import CompareStamp, ManifestError, SessionManifest, read_manifest
from agent6.viewmodel import produced_result, summarize_session_dir
from agent6.viewmodel.format import format_cost, status_label
from agent6.workflows.judge import CandidateBrief
from agent6.workflows.subrun import (
    GroupLaneSpawner,
    LaneResult,
    LaneSpawner,
    LaneSpec,
    LaneTask,
    SubrunError,
    clone_workspace,
    import_run,
)

# How often the await loop polls lane liveness, and how long Ctrl+C waits for a
# stop-requested lane to finish its in-flight step before giving up on it.
_POLL_INTERVAL_S = 2.0
_STOP_GRACE_S = 30.0


class ParallelError(Exception):
    """The fan-out could not be set up (over the [parallel].max_lanes cap)."""


class SpawnRun(Protocol):
    """Spawn a detached `agent6 <argv>` in *cwd* and return its located run dir.

    The front-end's process-spawn primitive (ui.spawn's
    `spawn_and_locate` with `agent6_exe` prepended); *argv* is the agent6
    subcommand + flags, WITHOUT the executable. Returns `(session_dir, "")` once the
    new run dir with a logs.jsonl appears, else `(None, error)`."""

    def __call__(
        self,
        argv: list[str],
        cwd: Path,
        *,
        before: set[Path],
        list_dirs: Callable[[], list[Path]],
        env: dict[str, str],
    ) -> tuple[Path | None, str]: ...


@dataclass(frozen=True, slots=True)
class LaneRuntime:
    """The front-end (`ui/cli`) primitives the parallel pipeline drives, injected
    so `agent6.app` never imports `agent6.ui`:

    - `spawn`: launch a detached `agent6` run and locate its run dir.
    - `build_provider` / `judging_status`: the reviewer provider + judge-progress
      status the fan-out auto-compare's `rank` uses (same wiring `sessions compare`
      uses). The coordinator dispatch path leaves these unexercised (it never
      compares its lanes).

    Lane liveness (`worker_is_alive`) and stop requests (`request_stop`) are the
    run-dir bridge itself (`agent6.sessions.ipc`), imported at module top: `app`
    already depends on it (`run.py`, `machine_agent.py`), so the seam does not
    re-export them."""

    spawn: SpawnRun
    build_provider: BuildProvider
    judging_status: JudgingStatus


# ---------------------------------------------------------------------------
# Lane planning
# ---------------------------------------------------------------------------


def subordinate_workdir_root(cfg: Config, origin: Path, group: str) -> Path:
    """Base dir for a group of subordinate working trees: `[parallel].workdir`
    (or `<cache_dir>/parallel`) / `<repo-id>` / `<group>`. Groups are fan-out
    ids (lane clones), `machine-<id>` (a run-state's per-state clone), and
    fork ids (the fork's linked worktree is the group dir itself); one
    location, so one prune sweep covers them all. Scoped by `repo_id(origin)`
    like the state dir: the sweep proves commit reachability against
    *origin*, so another repo's clones must never enter its scan."""
    return workdir_base(cfg, origin) / group


def workdir_base(cfg: Config, origin: Path) -> Path:
    """The per-repo dir every subordinate working tree of *origin* sits in."""
    base = Path(cfg.parallel.workdir) if cfg.parallel.workdir else cache_dir() / "parallel"
    return base / repo_id(origin)


def adopt_orphan_lane(
    origin: Path, cfg: Config, layout: SessionLayout, manifest: SessionManifest
) -> str | None:
    """Import an orphaned fan-out lane so an ordinary merge can land it.

    A coordinator death leaves a finished lane's branch only in its clone
    under `[parallel].workdir`, with the origin state still holding the
    live-view symlink. When the manifest carries fan-out lineage, the branch
    is absent from the origin, and the clone still holds it, run the
    coordinator's own import: fetch the branch, then replace the symlink with
    the real run dir. Returns the printable note, or None when this session
    is not that shape. Raises SubrunError when the attempt fails (the symlink
    is restored)."""
    if (
        not manifest.run_branch
        or manifest.parallel_id is None
        or manifest.lane is None
        or branch_exists(origin, manifest.run_branch)
        or not layout.session_dir.is_symlink()
    ):
        return None
    clone = subordinate_workdir_root(cfg, origin, manifest.parallel_id) / f"lane-{manifest.lane}"
    if not (clone / ".git").exists() or not branch_exists(clone, manifest.run_branch):
        return None
    real = layout.session_dir.resolve()
    layout.session_dir.unlink()
    try:
        import_run(origin, clone, manifest.run_branch, real, layout.state_dir)
    except SubrunError:
        layout.session_dir.symlink_to(real)
        raise
    return f"imported orphaned lane branch {manifest.run_branch} from {clone}"


def sweep_fanout_clones(origin: Path, cfg: Config) -> tuple[int, int]:
    """Delete fan-out clone dirs whose every lane branch tip already exists in
    *origin* (content-safe by commit proof, the prune --delete-squashed
    philosophy). Returns (swept, kept). A lane clone holding any commit the
    origin lacks keeps its whole fan-out dir: the clone may be the only copy.
    A fork's worktree (a group dir whose `.git` is a file) is
    `fork.sweep_fork_worktrees`'s, not a fan-out group."""
    base = workdir_base(cfg, origin)
    if not base.is_dir():
        return 0, 0
    swept = kept = 0
    for fanout in sorted(p for p in base.iterdir() if p.is_dir()):
        if (fanout / ".git").is_file():
            continue
        safe = True
        for clone in sorted(fanout.glob("lane-*")):
            if not (clone / ".git").exists():
                continue
            try:
                tips = [chain_tip(clone, br) for br in list_run_branches(clone)]
                # A machine-state clone's work rides its chain ref, which is
                # not a branch: prove those tips too, or the sweep deletes
                # the only copy of chain-only commits.
                tips += [sha for _ref, sha in list_chain_refs(clone)]
            except GitError:
                safe = False
                break
            if any(tip is not None and chain_tip(origin, tip) is None for tip in tips):
                safe = False
                break
        if safe:
            shutil.rmtree(fanout, ignore_errors=True)
            swept += 1
        else:
            kept += 1
    return swept, kept


def build_lane_specs(
    spec: str, *, cfg: Config, origin: Path, fanout_id: str, workdir_root: Path | None = None
) -> list[LaneSpec]:
    """Plan the lanes for a `--parallel` fan-out, refusing over-cap up front.
    *workdir_root* defaults to this fan-out's `subordinate_workdir_root` (the CLI adapter
    relies on that so it needn't reach the private helper)."""
    if workdir_root is None:
        workdir_root = subordinate_workdir_root(cfg, origin, fanout_id)
    models = parse_spec(spec, limit=cfg.parallel.max_lanes)
    return [
        LaneSpec(
            lane=i,
            session_id=f"{fanout_id}-l{i}",
            workdir=workdir_root / f"lane-{i}",
            model=model,
        )
        for i, model in enumerate(models, start=1)
    ]


# ---------------------------------------------------------------------------
# The real (bridge) spawner: clone, write a lane config, spawn detached, locate
# ---------------------------------------------------------------------------


def _write_lane_config(cfg: Config, spec: LaneSpec) -> Path:
    """Materialize the origin's effective config (worker model overridden for a
    per-lane model) to a file the lane loads with `--config`.

    The clone's path-keyed repo id yields an EMPTY per-repo config, so the lane
    would otherwise lose every origin repo setting; a full materialized config
    layered over the (shared) global config restores them. `for_repo=True` drops
    the global-only `[agent6].state_dir`, which `--config` forbids. Global config
    + secrets apply automatically."""
    lane_cfg = cfg.with_machine_agent_overrides(model=spec.model) if spec.model else cfg
    # A lane's branch IS how its work comes back (the import fetches
    # agent6/<session_id>), so the origin's branch_per_run=false must not ride
    # along: with it, every lane runs to completion, bills, and then fails the
    # import with a raw git "couldn't find remote ref".
    lane_cfg = lane_cfg.model_copy(
        update={"git": lane_cfg.git.model_copy(update={"branch_per_run": True})}
    )
    config_path = spec.workdir.parent / f"lane-{spec.lane}-config.toml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(materialize(lane_cfg, for_repo=True), encoding="utf-8")
    return config_path


def bridge_spawner(
    spec: LaneSpec,
    task: str,
    *,
    pins: Sequence[str] = (),
    cfg: Config,
    origin: Path,
    max_usd: float | None,
    auto_approve: bool = False,
    at: str | None = None,
    fanout_id: str,
    runtime: LaneRuntime,
) -> LaneResult:
    """Clone the origin, spawn a detached `agent6 run` in the clone, and return a
    LaneResult once its run dir is located (the run keeps going in the
    background). `ok=False` when the clone or spawn fails; the orchestrator
    records it and moves on. `auto_approve` forwards the coordinator/fan-out's
    own `--auto-approve` to the lane's argv, same as `max_usd`. The detached
    spawn + run-dir locate is *runtime*.spawn (the front-end's primitive)."""
    branch = run_branch_for(spec.session_id)
    try:
        clone_workspace(origin, spec.workdir)
        if at is not None:
            # Coordinator dispatch cuts lanes at the run's chain tip, not the
            # cloned HEAD (the operator's checkout, which the run never moves).
            # A local clone hardlinks the whole odb, so the chain's commits are
            # present without their refs.
            checkout_detached(spec.workdir, at)
    except (SubrunError, GitError) as exc:
        return LaneResult(
            spec=spec, session_dir=spec.workdir, branch=branch, ok=False, error=str(exc)
        )
    config_path = _write_lane_config(cfg, spec)
    lane_runs = bucket_dir(state_dir(spec.workdir, cfg.agent6.state_dir), "runs")

    def list_dirs() -> list[Path]:
        if not lane_runs.is_dir():
            return []
        return [p for p in lane_runs.iterdir() if p.is_dir()]

    argv = [
        "run",
        "--session-id",
        spec.session_id,
        "--config",
        str(config_path),
    ]
    if max_usd is not None:
        argv += ["--max-usd", f"{max_usd:g}"]
    if auto_approve:
        argv += ["--auto-approve"]
    for pin in pins:
        # Coordinator pins ride out-of-band of the task (see segment_lanes):
        # the lane seeds its own pin state from these instead of a task prefix
        # that became its manifest user_task.
        argv += ["--pin", pin]
    # `--` before the task so a task that looks like a flag (`--allow-root ...`)
    # is never parsed as one. Flags all precede it.
    argv += ["--", task]
    # AGENT6_SUBRUN marks the lane as a subordinate run: run.py leaves its
    # coordinator `lane_spawner` unwired and the `--parallel` flag refuses under
    # it, so a lane can never itself fan out or dispatch (depth 1 by construction,
    # for both the CLI fan-out and the coordinator's `/parallel` groups).
    markers = {
        "AGENT6_STREAM_TO_LOG": "1",
        "AGENT6_DETACHED_AWAY": "wait",
        "AGENT6_SUBRUN": "1",
        # The lane is self-describing from birth: its own manifest records the
        # fan-out lineage, so a coordinator death cannot orphan the grouping
        # (the old post-import stamp existed only while the coordinator lived).
        "AGENT6_PARALLEL_LINEAGE": f"{fanout_id}:{spec.lane}",
    }
    session_dir, err = runtime.spawn(
        argv, spec.workdir, before=set(), list_dirs=list_dirs, env={**os.environ, **markers}
    )
    if session_dir is None:
        return LaneResult(
            spec=spec, session_dir=lane_runs / spec.session_id, branch=branch, ok=False, error=err
        )
    return LaneResult(spec=spec, session_dir=session_dir, branch=branch, ok=True, error="")


# ---------------------------------------------------------------------------
# Coordinator dispatch: one lane to completion + a group spawner for the loop
# ---------------------------------------------------------------------------


def _lane_terminal(session_dir: Path, status: str, worker_is_alive: Callable[[Path], bool]) -> bool:
    """Terminal gate for an awaited lane: the fold left "running" AND the worker
    pid is cleared/dead. session.end lands in logs.jsonl before the lane's teardown
    clears worker.pid, so status alone races the teardown, and importing inside
    that window would misread a finished lane as still running. A lane that dies
    WITHOUT a session.end cannot hang this gate: the fold flips a dead recorded pid
    to "stale" at once, a pid-less silent lane to "stale" after its bounded
    silence window, and a lane that never wrote logs reads "?" (see
    `summarize_session_dir`)."""
    return status != "running" and not worker_is_alive(session_dir)


def _await_lane(
    res: LaneResult,
    *,
    runtime: LaneRuntime,
    poll_interval_s: float = _POLL_INTERVAL_S,
    should_stop: Callable[[], bool] | None = None,
) -> bool:
    """Block until *res*'s lane is terminal (True), awaited on its REAL run
    dir, or until *should_stop* goes true first (False): the coordinator's
    abort channel must be able to interrupt a group await that otherwise
    blocks until every lane ends. Same gate as the fan-out's `_await_lanes`,
    for a single lane."""
    while True:
        summary = summarize_session_dir(res.session_dir)
        if _lane_terminal(res.session_dir, summary.status, worker_is_alive):
            return True
        if should_stop is not None and should_stop():
            return False
        time.sleep(poll_interval_s)


def _drain_lane(
    res: LaneResult, *, poll_interval_s: float, hard_stop: threading.Event | None
) -> bool:
    """Bounded post-stop grace (mirrors the fan-out's stop_and_drain): True when
    the lane lands terminal in time, so its finished work still imports; False
    to leave it running un-imported. A hard stop (process teardown) skips the
    wait."""
    deadline = time.monotonic() + _STOP_GRACE_S
    while time.monotonic() < deadline:
        if hard_stop is not None and hard_stop.is_set():
            return False
        summary = summarize_session_dir(res.session_dir)
        if _lane_terminal(res.session_dir, summary.status, worker_is_alive):
            return True
        if hard_stop is not None:
            if hard_stop.wait(poll_interval_s):
                return False
        else:
            time.sleep(poll_interval_s)
    return False


def run_lane_to_completion(
    spec: LaneSpec,
    task: str,
    *,
    pins: Sequence[str] = (),
    cfg: Config,
    origin: Path,
    origin_state: Path,
    group: str,
    runtime: LaneRuntime,
    max_usd: float | None = None,
    auto_approve: bool = False,
    spawner: LaneSpawner | None = None,
    at: str | None = None,
    import_lock: threading.Lock | None = None,
    poll_interval_s: float = _POLL_INTERVAL_S,
    reporter: Reporter = STDIO_REPORTER,
    should_stop: Callable[[], bool] | None = None,
    hard_stop: threading.Event | None = None,
) -> LaneResult:
    """Run ONE subordinate lane fully and import it into *origin*.

    Clone + spawn (via *spawner*, default the bridge spawner), await the lane to
    terminal, then import its branch + run dir into the coordinator's repo and
    stamp `<group>` lineage. Returns a LaneResult whose `session_dir` is the imported
    dir on success; `ok=False` (nothing imported, *origin* untouched for this
    lane) when the lane failed to start, was still running at teardown, or its
    import was refused -- and also for an imported lane that produced no result
    (`produced_result`): its branch is safe in the origin but never joins as a
    success. The coordinator runs a group of these on a thread pool, so
    each is self-contained per lane; *import_lock*, when given, serializes the
    git-mutating import step across that group (concurrent fetches into one repo
    race on refs/objects). Tests inject a fake *spawner*.

    *should_stop* interrupts the await (the coordinator's abort channel or the
    group teardown): the lane gets a clean stop request, then a bounded grace
    (skipped once *hard_stop* is set) so a finishing lane still imports;
    otherwise it is left running detached and NOT imported (`ok=False`)."""
    if spawner is None:
        spawner = functools.partial(
            bridge_spawner,
            pins=pins,
            cfg=cfg,
            origin=origin,
            max_usd=max_usd,
            auto_approve=auto_approve,
            at=at,
            fanout_id=group,
            runtime=runtime,
        )
    res = spawner(spec, task)
    if not res.ok:
        return res
    # Symlink the live lane into the origin's runs/ (same as the fan-out path) so
    # a hub can see it and answer its approvals/asks while it runs, not just at
    # import. Dropped just before import so import_run can place the real dir.
    _symlink_lane(origin_state, res)
    if not _await_lane(
        res, runtime=runtime, poll_interval_s=poll_interval_s, should_stop=should_stop
    ):
        request_stop(res.session_dir)
        if not _drain_lane(res, poll_interval_s=poll_interval_s, hard_stop=hard_stop):
            # Still running: keep the clone + live symlink (they hold the only
            # copy of its branch until an import) and report the truth.
            return LaneResult(
                spec=spec,
                session_dir=res.session_dir,
                branch=res.branch,
                ok=False,
                error="interrupted; lane was asked to stop and keeps running"
                " detached; not imported",
            )
    lock = import_lock if import_lock is not None else contextlib.nullcontext()
    link = _lane_link(origin_state, res.spec.session_id)
    had_link = link.is_symlink()
    with contextlib.suppress(FileNotFoundError):
        link.unlink()
    try:
        with lock:
            dest = import_run(origin, spec.workdir, res.branch, res.session_dir, origin_state)
    except SubrunError as exc:
        if had_link:
            _symlink_lane(origin_state, res)  # restore the live view; nothing moved
        return LaneResult(
            spec=spec, session_dir=res.session_dir, branch=res.branch, ok=False, error=str(exc)
        )
    # The module contract ("clones + lane state are torn down after import")
    # applies to this path too; the fan-out's teardown lives in run_parallel.
    # Success only: the early ok=False returns above keep everything, since an
    # unimported lane's clone may hold the only copy of its branch. Thread-pool
    # safe: each lane removes only its own dirs, and the group-dir rmdir inside
    # _cleanup succeeds only for whichever lane empties it last.
    _cleanup([spec], workdir_root=spec.workdir.parent, cfg=cfg)
    summary = summarize_session_dir(dest)
    if not produced_result(summary.status):
        # The branch is imported (nothing lost), but only a deliberate end may
        # join the coordinator as a success.
        return LaneResult(
            spec=spec,
            session_dir=dest,
            branch=res.branch,
            ok=False,
            error=f"no result ({status_label(summary.status, summary.reason)});"
            f" branch imported as {res.branch}",
        )
    return LaneResult(spec=spec, session_dir=dest, branch=res.branch, ok=True, error="")


def build_lane_spawner(
    cfg: Config,
    origin: Path,
    origin_state: Path,
    *,
    coordinator_session_id: str,
    runtime: LaneRuntime,
    max_usd: float | None = None,
    auto_approve: bool = False,
    reporter: Reporter = STDIO_REPORTER,
) -> GroupLaneSpawner:
    """Build the coordinator's group dispatcher: the `GroupLaneSpawner` the loop
    calls at a `/parallel` steer boundary.

    One call clones + spawns each lane on its own model, awaits them all to
    terminal on a thread pool (one thread per lane, like the review panel's
    seats), imports each into *origin* (serialized by a shared lock), and returns
    per-lane LaneResults in dispatch order. Lane run ids are
    `<coordinator_session_id>-<group>-l<i>`; lane workspaces live under the same
    per-repo `[parallel].workdir` cache the fan-out uses, in a `<group>` subdir. The bridge
    spawner tags each lane `AGENT6_SUBRUN=1`, so a lane can never itself dispatch
    (depth 1 by construction). `auto_approve` forwards the coordinator's own
    `--auto-approve` to every lane, same as `max_usd`."""

    def dispatch(lanes: list[LaneTask], group: str, *, at: str | None = None) -> list[LaneResult]:
        # The documented hard cap on lanes per fan-out binds here too: the
        # /parallel steer uses the same grammar as run --parallel (which
        # build_lane_specs refuses up front), so a fat-fingered `/parallel 40`
        # must not clone the repo 40x and spawn 40 detached runs. First, before
        # any model-cache lookup or spec build; the loop's group-failure
        # feedback delivers the refusal to the coordinator.
        if len(lanes) > cfg.parallel.max_lanes:
            raise ParallelError(
                f"/parallel requests {len(lanes)} lanes but [parallel].max_lanes ="
                f" {cfg.parallel.max_lanes}. Request fewer, or raise [parallel].max_lanes."
            )
        # Validate the per-lane models before any clone: a refusal raises, and the
        # loop's group-failure feedback delivers the message to the coordinator
        # (keeping workflows free of a models dependency); no cache = warn + proceed.
        verdict = validate_spec_models([lane.model for lane in lanes], cfg)
        if verdict.refused:
            raise ParallelError(refusal_message(verdict, directive=True))
        if verdict.warned:
            reporter.warn(warning_message(verdict))
        workdir_root = subordinate_workdir_root(cfg, origin, coordinator_session_id) / group
        specs = [
            LaneSpec(
                lane=i,
                session_id=f"{coordinator_session_id}-{group}-l{i}",
                workdir=workdir_root / f"lane-{i}",
                model=lane.model,
            )
            for i, lane in enumerate(lanes, start=1)
        ]
        bucket_dir(origin_state, "runs").mkdir(parents=True, exist_ok=True)
        import_lock = threading.Lock()
        coord_dir = bucket_dir(origin_state, "runs") / coordinator_session_id
        hard_stop = threading.Event()

        def should_stop() -> bool:
            # The coordinator's immediate-stop channel: a front-end Stop (or
            # Ctrl-C steer -> abort) writes the "abort" steer answer, which the
            # loop consumes at the boundary AFTER this dispatch returns.
            # Without this poll, a stop during a /parallel group would sit
            # blocked until every lane finishes on its own.
            return hard_stop.is_set() or steer_answer_is_abort(coord_dir)

        def one(pair: tuple[LaneSpec, LaneTask]) -> LaneResult:
            spec, lane = pair
            return run_lane_to_completion(
                spec,
                lane.task,
                pins=lane.pins,
                at=at,
                cfg=cfg,
                origin=origin,
                origin_state=origin_state,
                group=group,
                runtime=runtime,
                max_usd=max_usd,
                auto_approve=auto_approve,
                import_lock=import_lock,
                reporter=reporter,
                should_stop=should_stop,
                hard_stop=hard_stop,
            )

        pairs = list(zip(specs, lanes, strict=True))
        if len(pairs) > 1:
            # Not a with-block: on KeyboardInterrupt __exit__ would join
            # every lane await, each blocked until its lane ends on its own.
            pool = ThreadPoolExecutor(max_workers=len(pairs))
            try:
                futures = [pool.submit(one, p) for p in pairs]
                # FIRST_EXCEPTION: a lane thread that RAISES (a bug, not a lane
                # failure -- those return ok=False) aborts the group now, not
                # after every earlier-submitted lane happens to finish.
                done, _ = futures_wait(futures, return_when=FIRST_EXCEPTION)
                for f in done:
                    exc = f.exception()
                    if exc is not None:
                        raise exc
                results = [f.result() for f in futures]  # submit order = lane order
            except BaseException:
                # Lane threads notice hard_stop within a poll tick, request a
                # clean stop on their lanes, and exit; the lanes themselves
                # keep running detached.
                hard_stop.set()
                pool.shutdown(wait=False, cancel_futures=True)
                raise
            pool.shutdown(wait=True)
            return results
        return [one(p) for p in pairs]

    return dispatch


def build_coordinator_spawner(
    cfg: Config,
    origin: Path,
    origin_state: Path,
    *,
    mode: str,
    session_id: str,
    runtime: LaneRuntime,
    max_usd: float | None = None,
    auto_approve: bool = False,
    reporter: Reporter = STDIO_REPORTER,
) -> GroupLaneSpawner | None:
    """The `/parallel` group dispatcher to wire into a run's loop, or None when
    dispatch is unavailable: a non-write mode (plan/ask make no commits to clone),
    or a run already inside a subordinate lane (`AGENT6_SUBRUN` set), which keeps
    dispatch depth 1 by construction. run.py / resume.py call this to build the
    loop's `lane_spawner`, passing the coordinator run's own effective
    `--auto-approve` (same as `max_usd`) so a lane never sits on an approval
    nothing detached can answer."""
    if mode != "run" or os.environ.get("AGENT6_SUBRUN"):
        return None
    return build_lane_spawner(
        cfg,
        origin,
        origin_state,
        coordinator_session_id=session_id,
        runtime=runtime,
        max_usd=max_usd,
        auto_approve=auto_approve,
        reporter=reporter,
    )


# ---------------------------------------------------------------------------
# Live view + await
# ---------------------------------------------------------------------------


def _lane_link(origin_state: Path, session_id: str) -> Path:
    return bucket_dir(origin_state, "runs") / session_id


def _symlink_lane(origin_state: Path, res: LaneResult) -> None:
    """Symlink a located lane's (clone-side) run dir into the origin's `runs/` so
    `agent6 sessions`/hub shows it live. Replaced by the real imported dir at import."""
    link = _lane_link(origin_state, res.spec.session_id)
    link.parent.mkdir(parents=True, exist_ok=True)
    with contextlib.suppress(FileNotFoundError):
        link.unlink()
    with contextlib.suppress(OSError):
        link.symlink_to(res.session_dir)


def _await_lanes(
    started: list[LaneResult],
    *,
    runtime: LaneRuntime,
    already_interrupted: bool = False,
    reporter: Reporter = STDIO_REPORTER,
) -> bool:
    """Poll every started lane's REAL run dir (in the clone's state; the origin
    symlink is a view for the hub, never the source of truth) until it is
    terminal (`_lane_terminal`), printing one line per lane on a status/cost
    change. Returns True if interrupted (Ctrl+C): request a clean stop on each
    still-running lane, wait a bounded grace for them to finish their in-flight
    step, then return so the caller imports what landed.

    `already_interrupted=True` (a Ctrl+C the spawn loop caught before the await
    even began) skips the normal poll and goes straight into that same stop-grace
    path, so a mid-spawn interrupt stops the already-started lanes identically."""
    pending = {r.spec.session_id: r for r in started}
    seen: dict[str, tuple[str, str, float]] = {}

    def poll_once() -> None:
        for rid, res in list(pending.items()):
            summary = summarize_session_dir(res.session_dir)
            # A "waiting" lane is blocked on an approval/question no detached
            # lane can answer; point the operator at the hub. _pending_prompt
            # supplies only the approval-vs-question wording.
            waiting = _pending_prompt(res.session_dir) if summary.status == "waiting" else ""
            key = (summary.status, waiting, round(summary.cost_usd, 4))
            if seen.get(rid) != key:
                seen[rid] = key
                _print_lane_status(
                    res.spec, summary.status, summary.cost_usd, waiting=waiting, reporter=reporter
                )
            if _lane_terminal(res.session_dir, summary.status, worker_is_alive):
                pending.pop(rid)

    def stop_and_drain() -> None:
        reporter.err("\n[agent6] interrupted; stopping lanes...")
        for res in pending.values():
            request_stop(res.session_dir)
        deadline = time.monotonic() + _STOP_GRACE_S
        with contextlib.suppress(KeyboardInterrupt):
            while pending and time.monotonic() < deadline:
                poll_once()
                if pending:
                    time.sleep(_POLL_INTERVAL_S)

    if already_interrupted:
        stop_and_drain()
        return True
    try:
        while pending:
            poll_once()
            if pending:
                time.sleep(_POLL_INTERVAL_S)
        return False
    except KeyboardInterrupt:
        stop_and_drain()
        return True


# The two prompt/answer event pairs a lane can block on, for `_pending_prompt`.
_PROMPT_KIND = {"approval.prompt": "approval", "question.prompt": "a question"}
_ANSWER_EVENTS = frozenset({"approval.answer", "question.answer"})


def _pending_prompt(session_dir: Path) -> str:
    """ "approval" / "a question" if the lane is blocked on an unanswered prompt,
    else "". The worker emits `approval.prompt`/`question.prompt` then BLOCKS on
    its `*.answer` (lanes run with AGENT6_DETACHED_AWAY=wait, so a prompt with no
    hub attached waits rather than denies), so the LAST prompt/answer event in
    logs.jsonl decides it -- a cheap trailing scan, no `*.request` marker exists
    for approvals/questions. Deliberately not the heavyweight SessionState fold; the
    fan-out status line needs only this one bit."""
    try:
        lines = (session_dir / LOGS_NAME).read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return ""
    for raw in reversed(lines):
        if "approval." not in raw and "question." not in raw:
            continue  # fast reject before json.loads
        try:
            ev = json.loads(raw)
        except ValueError:
            continue
        etype = ev.get("type") if isinstance(ev, dict) else None
        if etype in _ANSWER_EVENTS:
            return ""
        if etype in _PROMPT_KIND:
            return _PROMPT_KIND[etype]
    return ""


def _print_lane_status(
    spec: LaneSpec,
    status: str,
    cost: float,
    *,
    waiting: str = "",
    reporter: Reporter = STDIO_REPORTER,
) -> None:
    model = f" ({spec.model})" if spec.model else ""
    cost_s = f"  {format_cost(cost)}" if cost > 0 else ""
    state = (
        f"waiting on {waiting} (answer via agent6 attach {spec.session_id}, the web or TUI hub)"
        if waiting
        else status
    )
    reporter.note(f"lane {spec.lane} [{spec.session_id}]{model}: {state}{cost_s}")


# ---------------------------------------------------------------------------
# Import + auto-compare
# ---------------------------------------------------------------------------


def _stamp(session_dir: Path, **updates: object) -> str | None:
    """Apply typed field *updates* to an imported lane's manifest (read the model,
    `model_copy`, atomic rewrite). Returns an error string when the manifest
    cannot be read/parsed or written (the import itself stands; the caller reports
    the degradation). The one stamping helper: `_stamp_compare_outcomes`
    (post-ranking) goes through it; lineage itself is written by the LANE at
    birth (the spawn env), so a coordinator death cannot orphan the grouping."""
    mpath = session_dir / "manifest.json"
    try:
        m = read_manifest(session_dir)
    except ManifestError as exc:
        return f"could not read {mpath}: {exc}"
    try:
        write_manifest(mpath, m.model_copy(update=updates))
    except (OSError, ManifestError) as exc:
        # Disk full / read-only mount, or a manifest newer than this binary can
        # rewrite: the import already stands, so report the degradation and let
        # the loop keep importing/stamping the remaining lanes.
        return f"could not write {mpath}: {exc}"
    return None


def _stamp_compare_outcomes(
    candidates: list[CandidateBrief],
    outcome: RankOutcome,
    *,
    origin_state: Path,
    reporter: Reporter = STDIO_REPORTER,
) -> None:
    """Stamp the auto-compare outcome into EACH ranked lane's manifest, so every
    run view can show where a lane placed and why. ONE writer: only the fan-out's
    auto-compare stamps this (`sessions compare` stays stateless; the coordinator
    never compares its lanes). The imported lanes sit at `<origin_state>/sessions/runs/<id>`
    (import_run's contract); the same rationale and judge cost are recorded on
    every lane (both describe the judge's ranking of the whole group), the
    rationale truncated to bound the manifest and empty for a mechanical ranking.
    A per-lane stamp failure degrades loudly and never blocks the others."""
    of = len(candidates)
    text = outcome.rationale[:2000] if outcome.ranked_by == "judge" else ""
    for rank_pos, session_id in enumerate(outcome.ranking, start=1):
        compare = CompareStamp(
            rank=rank_pos,
            of=of,
            winner=rank_pos == 1,
            ranked_by=outcome.ranked_by,
            rationale=text,
            judge_cost_usd=outcome.judge_cost_usd,
            judge_cost_partial=outcome.judge_cost_partial,
        )
        err = _stamp(_lane_link(origin_state, session_id), compare=compare)
        if err is not None:
            reporter.note(f"lane [{session_id}]: imported, but the compare stamp failed: {err}")


def _import_lanes(
    results: list[LaneResult],
    *,
    origin: Path,
    origin_state: Path,
    state_base: str | None,
    base_sha: str,
    fanout_id: str,
    task: str,
    runtime: LaneRuntime,
    reporter: Reporter = STDIO_REPORTER,
) -> tuple[list[CandidateBrief], list[tuple[LaneResult, str]], list[LaneSpec]]:
    """Import each finished lane's branch + run dir into the origin, stamp its
    lineage, and build a candidate brief from it -- for lanes that produced a
    result; an imported lane without one (`produced_result`) is recorded as
    failed instead, its work safe in the origin. Returns (candidates, failed,
    imported specs); only imported lanes are safe to clean up. A failed-to-start,
    still-running, or import-refused lane is recorded and never blocks the
    others; its clone, state, and live symlink stay in place (they may hold the
    only copy of its work). Candidate diffs come from the clone (still on the
    run branch) before cleanup.
    """
    candidates: list[CandidateBrief] = []
    failed: list[tuple[LaneResult, str]] = []
    imported: list[LaneSpec] = []
    for res in results:
        if not res.ok:
            failed.append((res, res.error))
            continue
        link = _lane_link(origin_state, res.spec.session_id)
        if worker_is_alive(res.session_dir):
            failed.append(
                (
                    res,
                    "still running; left in place"
                    f" (watch: agent6 attach {res.spec.session_id};"
                    f" stop: agent6 sessions stop {res.spec.session_id})",
                )
            )
            continue
        had_link = link.is_symlink()
        with contextlib.suppress(FileNotFoundError):
            link.unlink()  # drop the live symlink so import can place the real dir
        try:
            dest = import_run(origin, res.spec.workdir, res.branch, res.session_dir, origin_state)
        except SubrunError as exc:
            if had_link:
                _symlink_lane(origin_state, res)  # restore the live view; nothing moved
            failed.append((res, str(exc)))
            continue
        imported.append(res.spec)
        # A lane's operator rulings outlive its state dir (torn down after import).
        carried = merge_decisions(state_dir(res.spec.workdir, state_base), origin_state)
        if carried:
            reporter.note(f"lane {res.spec.lane}: {carried} recorded decision(s) carried over")
        summary = summarize_session_dir(dest)
        if not produced_result(summary.status):
            # Imported (its branch is safe in the origin) but not a candidate:
            # only a deliberate end is rankable work.
            failed.append(
                (
                    res,
                    f"no result ({status_label(summary.status, summary.reason)});"
                    f" branch imported as {res.branch}, not ranked",
                )
            )
            continue
        candidates.append(
            CandidateBrief(
                session_id=res.spec.session_id,
                task=manifest_task(dest, task),
                diff=diff_since(res.spec.workdir, base_sha),
                verify_ok=summary.verify_ok,
                cost_usd=summary.cost_usd,
            )
        )
    return candidates, failed, imported


def _cleanup(imported: list[LaneSpec], *, workdir_root: Path, cfg: Config) -> None:
    """Tear down clone + state dir + lane config for IMPORTED lanes only; a lane
    that did not import keeps everything it has (its clone may hold the only
    copy of its branch, and a live lane must never lose its workspace). The
    fan-out workdir root is removed only once it is empty. Best-effort: a
    leftover clone is disk waste, never corruption."""
    for spec in imported:
        shutil.rmtree(state_dir(spec.workdir, cfg.agent6.state_dir), ignore_errors=True)
        shutil.rmtree(spec.workdir, ignore_errors=True)
        (spec.workdir.parent / f"lane-{spec.lane}-config.toml").unlink(missing_ok=True)
    with contextlib.suppress(OSError):
        workdir_root.rmdir()  # only succeeds when nothing was kept


def fanout_exit_code(candidates: list[CandidateBrief]) -> int:
    """The fan-out's exit, from the lanes' gate verdicts: 1 nothing rankable,
    0 some lane verified green (or no lane had a gate), 4 gates ran and none
    passed: a distinct code, so a script never reads an all-red fan-out as
    success."""
    if not candidates:
        return 1
    if any(c.verify_ok is True for c in candidates):
        return 0
    return EXIT_VERIFY_FAILED if any(c.verify_ok is False for c in candidates) else 0


def _print_report(
    candidates: list[CandidateBrief],
    outcome: RankOutcome,
    failed: list[tuple[LaneResult, str]],
    *,
    fanout_id: str,
    reporter: Reporter = STDIO_REPORTER,
) -> None:
    """Print the ranked candidate table + a `sessions merge` line per candidate, and
    list any failed lanes. Nothing is merged automatically."""
    reporter.out(
        f"\n[agent6] parallel fan-out {fanout_id} complete: {len(candidates)} candidate(s)"
    )
    print_ranked_candidates(candidates, outcome, reporter=reporter)
    if failed:
        reporter.out("\nfailed lanes (nothing of theirs was deleted):")
        for res, err in failed:
            reporter.out(f"  - lane {res.spec.lane} [{res.spec.session_id}]: {err}")
            kept = [p for p in (res.spec.workdir, res.session_dir) if p.exists()]
            if kept:
                reporter.out(f"    kept: {', '.join(str(p) for p in kept)}")


# ---------------------------------------------------------------------------
# Orchestrator entry point
# ---------------------------------------------------------------------------


def run_parallel(
    task: str,
    lanes: list[LaneSpec],
    *,
    cfg: Config,
    origin: Path,
    origin_state: Path,
    runtime: LaneRuntime,
    spawner: LaneSpawner | None = None,
    max_usd: float | None = None,
    fanout_id: str | None = None,
    auto_approve: bool = False,
    pins: Sequence[str] = (),
    reporter: Reporter = STDIO_REPORTER,
) -> int:
    """Run *lanes* to completion, import them, and print a ranked comparison.

    Returns `fanout_exit_code` over the lanes (0 a lane verified green or no
    lane had a gate, 1 nothing rankable, 4 gates ran and none passed), 2 with
    no lanes or an unreadable origin, 130 on Ctrl+C.
    *spawner* defaults to the real bridge spawner; tests inject a fake.
    `auto_approve` forwards to every lane's argv, same as `max_usd`.
    """
    if not lanes:
        reporter.error("no lanes to run")
        return 2
    if fanout_id is None:
        fanout_id = lanes[0].session_id.rsplit("-l", 1)[0]
    if spawner is None:
        spawner = functools.partial(
            bridge_spawner,
            pins=pins,
            cfg=cfg,
            origin=origin,
            max_usd=max_usd,
            auto_approve=auto_approve,
            fanout_id=fanout_id,
            runtime=runtime,
        )
    try:
        base_sha = git_status(origin).head_sha
    except GitError as exc:
        reporter.error(str(exc))
        return 2

    bucket_dir(origin_state, "runs").mkdir(parents=True, exist_ok=True)
    reporter.note(f"parallel fan-out {fanout_id}: {len(lanes)} lanes")
    if max_usd is not None:
        # The judge is one more capped call series, so the advertised total
        # includes it; without that the effective ceiling quietly exceeded
        # the printed one.
        reporter.note(
            f"budget: ${max_usd:g}/lane x {len(lanes)} + judge"
            f" = ${max_usd * (len(lanes) + 1):g} total"
        )

    results: list[LaneResult] = []
    try:
        for spec in lanes:
            res = spawner(spec, task)
            results.append(res)
            if res.ok:
                _symlink_lane(origin_state, res)
                _print_lane_status(spec, "started", 0.0, reporter=reporter)
            else:
                reporter.note(f"lane {spec.lane} [{spec.session_id}]: FAILED to start: {res.error}")
        interrupted = _await_lanes([r for r in results if r.ok], runtime=runtime, reporter=reporter)
    except KeyboardInterrupt:
        # Ctrl+C mid-spawn (before the await): route the already-started lanes
        # into the same stop-grace path, then import-what-exists + report below.
        interrupted = _await_lanes(
            [r for r in results if r.ok],
            runtime=runtime,
            already_interrupted=True,
            reporter=reporter,
        )

    candidates, failed, imported = _import_lanes(
        results,
        origin=origin,
        origin_state=origin_state,
        state_base=cfg.agent6.state_dir,
        base_sha=base_sha,
        fanout_id=fanout_id,
        task=task,
        runtime=runtime,
        reporter=reporter,
    )
    _cleanup(imported, workdir_root=lanes[0].workdir.parent, cfg=cfg)

    outcome = rank(
        cfg,
        candidates,
        transcript_dir=origin_state / "parallel" / fanout_id,
        build_provider=runtime.build_provider,
        judging_status=runtime.judging_status,
        max_usd=max_usd,
        reporter=reporter,
    )
    _stamp_compare_outcomes(
        candidates,
        outcome,
        origin_state=origin_state,
        reporter=reporter,
    )
    _print_report(
        candidates,
        outcome,
        failed,
        fanout_id=fanout_id,
        reporter=reporter,
    )

    if interrupted:
        return 130
    return fanout_exit_code(candidates)
