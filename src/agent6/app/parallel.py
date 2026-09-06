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
import os
import shutil
import threading
from collections.abc import Callable, Sequence
from concurrent.futures import FIRST_EXCEPTION, ThreadPoolExecutor
from concurrent.futures import wait as futures_wait
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from agent6.app._lane_watch import (
    POLL_INTERVAL_S,
    await_lane,
    await_lanes,
    drain_lane,
    lane_link,
    print_lane_status,
    symlink_lane,
)
from agent6.app.compare import (
    BuildProvider,
    JudgingStatus,
    RankOutcome,
    manifest_task,
    print_ranked_candidates,
    rank,
    verify_word,
)
from agent6.app.finalize import EXIT_VERIFY_FAILED
from agent6.app.manifest import write_manifest, write_session_manifest
from agent6.app.reporter import STDIO_REPORTER, Reporter
from agent6.config import Config
from agent6.config.layer import materialize
from agent6.directive import parse_spec
from agent6.events import EventSink, EventWriteError
from agent6.git_ops import (
    GitError,
    branch_exists,
    chain_tip,
    checkout_detached,
    commit_is_reachable,
    diff_since,
    list_chain_refs,
    list_run_branches,
    run_branch_for,
)
from agent6.git_ops import status as git_status
from agent6.memory import merge_decisions, merge_memory, seed_store
from agent6.models.validate import refusal_message, validate_spec_models, warning_message
from agent6.paths import cache_dir, mkdir_for_real_user, repo_id, state_dir
from agent6.sessions.ipc import (
    AwayMode,
    clear_stop_request,
    clear_worker_pid,
    emit_session_start,
    request_stop,
    steer_answer_is_abort,
    stop_request_pending,
    worker_is_alive,
)
from agent6.sessions.layout import SessionLayout, bucket_dir, session_layout
from agent6.sessions.manifest import (
    MANIFEST_NAME,
    CompareStamp,
    FanoutStamp,
    ManifestError,
    SessionManifest,
    read_manifest,
)
from agent6.types import session_bucket
from agent6.viewmodel import produced_result, summarize_session_dir
from agent6.viewmodel.format import status_label
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
        or manifest.parallel is None
        or branch_exists(origin, manifest.run_branch)
        or not layout.session_dir.is_symlink()
    ):
        return None
    lineage = manifest.parallel
    clone = subordinate_workdir_root(cfg, origin, lineage.group) / f"lane-{lineage.lane}"
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


def _recorded_merged(origin: Path, clone: Path, tip: str) -> bool:
    """Whether some session's manifest records *tip* as merged into a branch
    that still holds it. The evidence `--delete-squashed` requires, read here
    so a squash-merged lane's clone (whose commits are unreachable by design)
    is still sweepable."""
    state = state_dir(origin)
    for branch in list_run_branches(clone):
        layout = session_layout(state, branch.removeprefix("agent6/"))
        if layout is None:
            continue
        try:
            merged = read_manifest(layout.session_dir).merged
        except ManifestError:
            continue
        if merged is not None and merged.tip == tip and branch_exists(origin, merged.into):
            return True
    return False


def sweep_fanout_clones(origin: Path, cfg: Config) -> tuple[int, int]:
    """Delete fan-out clone dirs whose every lane branch tip already exists in
    *origin* (content-safe by commit proof, the prune --delete-squashed
    philosophy). Returns (swept, kept). A lane clone holding any commit the
    origin lacks keeps its whole fan-out dir: the clone may be the only copy.
    Only a dir holding `lane-*` clones is a fan-out group: anything else
    under the scope (a fork's worktree, which `fork.sweep_fork_worktrees`
    owns through its manifest; a directory the operator put there) is left
    alone."""
    base = workdir_base(cfg, origin)
    if not base.is_dir():
        return 0, 0
    swept = kept = 0
    for fanout in sorted(p for p in base.iterdir() if p.is_dir()):
        clones = [c for c in sorted(fanout.glob("lane-*")) if (c / ".git").is_dir()]
        if not clones:
            continue
        safe = True
        for clone in clones:
            try:
                tips = [chain_tip(clone, br) for br in list_run_branches(clone)]
                # A machine-state clone's work rides its chain ref, which is
                # not a branch: prove those tips too, or the sweep deletes
                # the only copy of chain-only commits.
                tips += [sha for _ref, sha in list_chain_refs(clone)]
            except GitError:
                safe = False
                break
            # Reachability, not existence: a tip the origin holds only as a
            # loose object (its branch deleted, as the prune's own message
            # tells the operator to do) is one `git gc` from gone, and the
            # clone is the last ref that reaches it. A tip a run's own manifest
            # records as merged is proven a second way: a squash leaves the
            # content in the base and the commits unreachable by design.
            if any(
                tip is not None
                and not commit_is_reachable(origin, tip)
                and not _recorded_merged(origin, clone, tip)
                for tip in tips
            ):
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
    layered over the (shared) global config restores them. Global config
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
    config_path.write_text(materialize(lane_cfg), encoding="utf-8")
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
    lane_away: AwayMode = "wait",
    at: str | None = None,
    fanout_id: str,
    coordinator: str,
    runtime: LaneRuntime,
) -> LaneResult:
    """Clone the origin, spawn a detached `agent6 run` in the clone, and return a
    LaneResult once its run dir is located (the run keeps going in the
    background). `ok=False` when the clone or spawn fails; the orchestrator
    records it and moves on. `auto_approve` forwards the coordinator/fan-out's
    own `--auto-approve` to the lane's argv, same as `max_usd`; `lane_away` is
    the lane's away-mode, "wait" (a question parks for `agent6 attach`) or
    "deny" (nobody attends: a question gets empty answers). The detached
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
    lane_state = state_dir(spec.workdir)
    # The clone is a repo of its own, so the lane's state dir is empty: without
    # this the lanes run blind to the memory and rulings the origin's own runs
    # are given. New rulings come back with `merge_decisions` at import.
    seed_store(state_dir(origin), lane_state)
    lane_runs = bucket_dir(lane_state, "runs")

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
        "AGENT6_DETACHED_AWAY": lane_away,
        "AGENT6_SUBRUN": "1",
        # The lane is self-describing from birth: its own manifest records the
        # fan-out lineage (the coordinator every listing nests it under, the
        # group, the lane), so a coordinator death cannot orphan the grouping.
        "AGENT6_PARALLEL_LINEAGE": f"{coordinator}:{fanout_id}:{spec.lane}",
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


def run_lane_to_completion(
    spec: LaneSpec,
    task: str,
    *,
    pins: Sequence[str] = (),
    cfg: Config,
    origin: Path,
    origin_state: Path,
    group: str,
    coordinator: str,
    runtime: LaneRuntime,
    max_usd: float | None = None,
    auto_approve: bool = False,
    lane_away: AwayMode = "wait",
    spawner: LaneSpawner | None = None,
    at: str | None = None,
    import_lock: threading.Lock | None = None,
    poll_interval_s: float = POLL_INTERVAL_S,
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
            lane_away=lane_away,
            at=at,
            fanout_id=group,
            coordinator=coordinator,
            runtime=runtime,
        )
    res = spawner(spec, task)
    if not res.ok:
        return res
    # Symlink the live lane into the origin's runs/ (same as the fan-out path) so
    # a hub can see it and answer its approvals/asks while it runs, not just at
    # import. Dropped just before import so import_run can place the real dir.
    symlink_lane(origin_state, res)
    if not await_lane(res, poll_interval_s=poll_interval_s, should_stop=should_stop):
        asked = request_stop(res.session_dir)
        if not drain_lane(res, poll_interval_s=poll_interval_s, hard_stop=hard_stop):
            # Still running: keep the clone + live symlink (they hold the only
            # copy of its branch until an import) and report the truth.
            return LaneResult(
                spec=spec,
                session_dir=res.session_dir,
                branch=res.branch,
                ok=False,
                error="interrupted; lane "
                + ("was asked to stop and" if asked else "could not be asked to stop and")
                + " keeps running detached; not imported",
            )
    try:
        dest = _import_lane(res, origin, origin_state, import_lock)
    except SubrunError as exc:
        return LaneResult(
            spec=res.spec, session_dir=res.session_dir, branch=res.branch, ok=False, error=str(exc)
        )
    carry_back(
        state_dir(res.spec.workdir),
        origin_state,
        dest,
        lane=res.spec.lane,
        reporter=reporter,
    )
    # The module contract ("clones + lane state are torn down after import")
    # applies to this path too; the fan-out's teardown lives in run_parallel.
    # Success only: the early ok=False returns above keep everything, since an
    # unimported lane's clone may hold the only copy of its branch. Thread-pool
    # safe: each lane removes only its own dirs, and the group-dir rmdir inside
    # _cleanup succeeds only for whichever lane empties it last.
    _cleanup(
        [res.spec], workdir_root=res.spec.workdir.parent, base=workdir_base(cfg, origin), cfg=cfg
    )
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
    lane_away: AwayMode = "wait",
    reporter: Reporter = STDIO_REPORTER,
) -> GroupLaneSpawner:
    """Build the coordinator's group dispatcher: the `GroupLaneSpawner` the loop
    calls at a `/parallel` steer boundary.

    One call clones + spawns each lane on its own model, awaits them all to
    terminal on a thread pool (one thread per lane, like the review panel's
    seats), imports each into *origin* (serialized by a shared lock), and returns
    per-lane LaneResults in dispatch order. Lane run ids are
    `<coordinator_session_id>-<group>-l<i>`; lane workspaces live under the same
    per-repo `[parallel].workdir` cache the fan-out uses, in a dir of that name. The bridge
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
        # ONE name for this group: the workdir dir, the lane ids and the
        # lineage each lane's manifest records. A group whose clones sit a level
        # deeper than the name they are stamped with is invisible to
        # `adopt_orphan_lane` and to `sweep_fanout_clones`, which both derive
        # the clone as `subordinate_workdir_root(...) / lane-<n>`.
        group_id = f"{coordinator_session_id}-{group}"
        workdir_root = subordinate_workdir_root(cfg, origin, group_id)
        specs = [
            LaneSpec(
                lane=i,
                session_id=f"{group_id}-l{i}",
                workdir=workdir_root / f"lane-{i}",
                model=lane.model,
            )
            for i, lane in enumerate(lanes, start=1)
        ]
        mkdir_for_real_user(bucket_dir(origin_state, "runs"))
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
                group=group_id,
                coordinator=coordinator_session_id,
                runtime=runtime,
                max_usd=max_usd,
                auto_approve=auto_approve,
                lane_away=lane_away,
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
    lane_away: AwayMode = "wait",
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
        lane_away=lane_away,
        reporter=reporter,
    )


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
    mpath = session_dir / MANIFEST_NAME
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
        err = _stamp(lane_link(origin_state, session_id), compare=compare)
        if err is not None:
            reporter.note(f"lane [{session_id}]: imported, but the compare stamp failed: {err}")


def carry_back(
    lane_state: Path, origin_state: Path, dest: Path, *, lane: int, reporter: Reporter
) -> None:
    """A lane's operator rulings and memories outlive its state dir (torn
    down after import): the harness nudges a lane to write memory like any
    run, so what it wrote lands in the origin's store, held-back files under
    the imported run dir *dest*, and the note says what did."""
    carried, known = merge_decisions(lane_state, origin_state)
    if carried or known:
        already = f", {known} already recorded" if known else ""
        reporter.note(f"lane {lane}: {carried} recorded decision(s) carried over{already}")
    held_dir = dest / "memory-held"
    merge = merge_memory(lane_state, origin_state, held_dir=held_dir)
    parts = [
        f"{len(names)} {word}"
        for word, names in (
            ("carried", merge.carried),
            ("updated", merge.updated),
            ("deleted", merge.deleted),
        )
        if names
    ]
    if merge.held:
        kept = f", kept at {held_dir}" if held_dir.is_dir() else ""
        parts.append(
            f"held back (changed here too, or the name is taken): {', '.join(merge.held)}{kept}"
        )
    if parts:
        reporter.note(f"lane {lane}: memory {', '.join(parts)}")


def _import_lane(
    res: LaneResult, origin: Path, origin_state: Path, lock: threading.Lock | None
) -> Path:
    """Land a finished lane's run dir in the origin under *lock* when a group
    shares one: the live symlink drops so the real dir can take its place, and
    comes back when the import refuses (nothing moved). Raises SubrunError."""
    link = lane_link(origin_state, res.spec.session_id)
    had_link = link.is_symlink()
    with contextlib.suppress(FileNotFoundError):
        link.unlink()
    try:
        with lock if lock is not None else contextlib.nullcontext():
            return import_run(origin, res.spec.workdir, res.branch, res.session_dir, origin_state)
    except SubrunError:
        if had_link:
            symlink_lane(origin_state, res)
        raise


def _import_lanes(
    results: list[LaneResult],
    *,
    origin: Path,
    origin_state: Path,
    base_sha: str,
    task: str,
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
        try:
            dest = _import_lane(res, origin, origin_state, None)
        except SubrunError as exc:
            failed.append((res, str(exc)))
            continue
        imported.append(res.spec)
        carry_back(
            state_dir(res.spec.workdir),
            origin_state,
            dest,
            lane=res.spec.lane,
            reporter=reporter,
        )
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


def _cleanup(imported: list[LaneSpec], *, workdir_root: Path, base: Path, cfg: Config) -> None:
    """Tear down clone + state dir + lane config for IMPORTED lanes only; a lane
    that did not import keeps everything it has (its clone may hold the only
    copy of its branch, and a live lane must never lose its workspace). Every
    level from the group dir up to and including the per-repo *base*
    (`workdir_base`) is removed once empty, never the dir above it. Best-effort:
    a leftover clone is disk waste, never corruption."""
    for spec in imported:
        shutil.rmtree(state_dir(spec.workdir), ignore_errors=True)
        shutil.rmtree(spec.workdir, ignore_errors=True)
        (spec.workdir.parent / f"lane-{spec.lane}-config.toml").unlink(missing_ok=True)
    level = workdir_root
    while level.is_relative_to(base):
        with contextlib.suppress(OSError):
            level.rmdir()  # only succeeds when nothing was kept
        if level == base:
            break
        level = level.parent


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
    lane_away: AwayMode = "wait",
    pins: Sequence[str] = (),
    spec: str = "",
    reporter: Reporter = STDIO_REPORTER,
) -> int:
    """Run *lanes* to completion, import them, and print a ranked comparison.

    The fan-out is a session of its own under the origin's runs (id
    *fanout_id*): a manifest carrying the fan-out stamp (*spec* is the
    `--parallel` argument as typed) and no run branch, a journal (the
    dispatch, the ranking, the end), and a worker pid while the lanes run
    and the judge ranks them, so every listing shows it live and nests the
    lanes under it. `sessions stop <fanout_id>` ends it the way Ctrl+C does.

    Returns `fanout_exit_code` over the lanes (0 a lane verified green or no
    lane had a gate, 1 nothing rankable, 4 gates ran and none passed), 2 with
    no lanes or an unreadable origin, 130 on Ctrl+C or a stop request.
    *spawner* defaults to the real bridge spawner; tests inject a fake.
    `auto_approve` forwards to every lane's argv, same as `max_usd`, and
    `lane_away` is every lane's away-mode (see `bridge_spawner`).
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
            lane_away=lane_away,
            fanout_id=fanout_id,
            coordinator=fanout_id,
            runtime=runtime,
        )
    try:
        origin_status = git_status(origin)
    except GitError as exc:
        reporter.error(str(exc))
        return 2
    base_sha = origin_status.head_sha

    layout = SessionLayout(
        state_dir=origin_state, session_id=fanout_id, subdir=session_bucket("run")
    )
    layout.ensure()
    write_session_manifest(
        layout,
        session_id=fanout_id,
        user_task=task,
        base_sha=base_sha,
        base_branch=origin_status.branch,
        run_branch=None,
        cfg=cfg,
        fanout=FanoutStamp(lanes=len(lanes), spec=spec),
    )
    events = EventSink(layout.logs_path)
    emit_session_start(
        events,
        layout.session_dir,
        "session.start",
        session_id=fanout_id,
        user_task=task[:200],
        mode="run",
    )
    events.emit("loop.parallel.dispatched", group=fanout_id, lanes=len(lanes), tasks=[task[:200]])
    try:
        return _drive_fanout(
            task,
            lanes,
            spawner=spawner,
            cfg=cfg,
            origin=origin,
            origin_state=origin_state,
            runtime=runtime,
            max_usd=max_usd,
            fanout_id=fanout_id,
            base_sha=base_sha,
            events=events,
            coordinator_dir=layout.session_dir,
            reporter=reporter,
        )
    except KeyboardInterrupt:
        # Ctrl+C past the await (the judge call, the report): the operator's
        # own act, journaled so the record reads stopped, never stale.
        with contextlib.suppress(EventWriteError):
            events.emit("session.end", reason="interrupted", iterations=0, all_passed=False)
        raise
    except Exception:
        with contextlib.suppress(EventWriteError):
            events.emit("session.end", reason="crashed", iterations=0, all_passed=False)
        raise
    finally:
        clear_worker_pid(layout.session_dir)


def _drive_fanout(
    task: str,
    lanes: list[LaneSpec],
    *,
    spawner: LaneSpawner,
    cfg: Config,
    origin: Path,
    origin_state: Path,
    runtime: LaneRuntime,
    max_usd: float | None,
    fanout_id: str,
    base_sha: str,
    events: EventSink,
    coordinator_dir: Path,
    reporter: Reporter,
) -> int:
    """Spawn, await, import, rank and report the lanes, journaling the ranking
    and the end into the fan-out's own session (`run_parallel` opened it)."""
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
                symlink_lane(origin_state, res)
                print_lane_status(spec, "started", 0.0, reporter=reporter)
            else:
                reporter.note(f"lane {spec.lane} [{spec.session_id}]: FAILED to start: {res.error}")
        interrupted = await_lanes(
            [r for r in results if r.ok],
            should_stop=lambda: stop_request_pending(coordinator_dir),
            reporter=reporter,
        )
    except KeyboardInterrupt:
        # Ctrl+C mid-spawn (before the await): route the already-started lanes
        # into the same stop-grace path, then import-what-exists + report below.
        interrupted = await_lanes(
            [r for r in results if r.ok],
            already_interrupted=True,
            reporter=reporter,
        )

    # The stop request, honoured or arrived too late to matter, is consumed
    # with the session: nothing outlives the fan-out in its dir.
    clear_stop_request(coordinator_dir)
    candidates, failed, imported = _import_lanes(
        results,
        origin=origin,
        origin_state=origin_state,
        base_sha=base_sha,
        task=task,
        reporter=reporter,
    )
    _cleanup(
        imported, workdir_root=lanes[0].workdir.parent, base=workdir_base(cfg, origin), cfg=cfg
    )

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
    by_id = {c.session_id: c for c in candidates}
    events.emit(
        "loop.parallel.compared",
        group=fanout_id,
        ranked_by=outcome.ranked_by,
        ranking=[
            {
                "session_id": rid,
                "verify": verify_word(by_id[rid].verify_ok),
                "cost_usd": by_id[rid].cost_usd,
            }
            for rid in outcome.ranking
        ],
        judge_cost_usd=outcome.judge_cost_usd,
        judge_cost_partial=outcome.judge_cost_partial,
    )
    if outcome.judge_cost_usd > 0 or outcome.judge_cost_partial:
        # The judge is the fan-out's own spend; each lane's rides its own row.
        events.emit(
            "budget.update",
            usd_total=outcome.judge_cost_usd,
            usd_partial=outcome.judge_cost_partial,
        )
    rc = 130 if interrupted else fanout_exit_code(candidates)
    reason, all_passed = _fanout_end(rc, candidates)
    events.emit("session.end", reason=reason, iterations=0, all_passed=all_passed)
    return rc


def _fanout_end(rc: int, candidates: list[CandidateBrief]) -> tuple[str, bool | None]:
    """The `session.end` a fan-out's exit code reads as: interrupted (130),
    finished (0: passed when a lane's gate went green, ungated when none had
    one), or failed for want of a result (1) or a green gate (4). The
    fan-out's own gate verdict is its best lane's: it ran no gate itself."""
    if rc == 130:
        return "interrupted", False
    if rc == 0:
        return "finish_session", True if any(c.verify_ok for c in candidates) else None
    return ("no_lane_passed" if rc == EXIT_VERIFY_FAILED else "no_lane_result"), False
