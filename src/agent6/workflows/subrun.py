# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Subordinate-run mechanics: clone a disposable lane workspace, import a
finished lane's branch and run dir back into the origin, and join a
subordinate branch into the current branch.

Pure git plumbing over `agent6.git_ops` -- no LLM, no UI, no process
spawning. `app.parallel` drives a `LaneSpawner` over these.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from agent6.git_ops import GitError, branch_exists, clone_repo, fetch_branch
from agent6.sessions.layout import bucket_dir


class SubrunError(Exception):
    """Subordinate-run mechanics (clone/import/join) failed."""


@dataclass(frozen=True, slots=True)
class LaneSpec:
    """One subordinate lane to run: its own workspace clone, run id, and
    model (None = the configured worker model)."""

    lane: int
    session_id: str
    workdir: Path
    model: str | None


@dataclass(frozen=True, slots=True)
class LaneResult:
    """Outcome of running a lane: where its run state lives, its branch, and
    whether it succeeded (`error` set on failure)."""

    spec: LaneSpec
    session_dir: Path
    branch: str
    ok: bool
    error: str


@dataclass(frozen=True, slots=True)
class LaneTask:
    """One lane to dispatch: the task text and an optional per-lane model
    (`None` = the configured worker model). The coordinator expands each
    `/parallel` segment into these (spec=3 -> three, one per model in a list)."""

    task: str
    model: str | None
    # Operator pins delivered OUT-OF-BAND of the task (the spawner's --pin
    # channel): folding one into `task` would make it the lane's manifest
    # user_task, so every listing would lead with the pin header.
    pins: tuple[str, ...] = ()


class LaneSpawner(Protocol):
    def __call__(self, spec: LaneSpec, task: str) -> LaneResult: ...


class GroupLaneSpawner(Protocol):
    """Dispatch a sibling group of subordinate lanes and return their results in
    dispatch order (one `LaneResult` per `LaneTask` in *lanes*).

    One call is synchronous-complete: clone + spawn each lane on its own model,
    await them all to terminal, and import each finished branch + run dir into the
    coordinator's repo. All spawn/await/import machinery is `app.parallel`'s
    (over the front-end's `LaneRuntime`); the coordinator loop supplies only
    the per-lane tasks
    and a *group* id (`p<seq>`), so `workflows` never imports ui. On a lane that
    failed to start, is still running at teardown, or whose import was refused,
    that lane's `LaneResult.ok` is False and the coordinator's repo is left
    untouched for it."""

    def __call__(
        self, lanes: list[LaneTask], group: str, *, at: str | None = None
    ) -> list[LaneResult]: ...


def clone_workspace(origin: Path, dest: Path) -> None:
    """Clone *origin* into *dest*, a disposable lane workspace.

    Plain `git clone` on a filesystem path (git's local-clone optimization:
    hardlinks same-filesystem, copies cross-device). Raises SubrunError on
    failure, e.g. *dest* already exists or *origin* is not a repo.
    """
    try:
        clone_repo(origin, dest)
    except GitError as exc:
        raise SubrunError(f"clone {origin} -> {dest} failed: {exc}") from exc


def import_run(
    origin: Path,
    lane_repo: Path,
    branch: str,
    lane_session_dir: Path,
    origin_state: Path,
) -> Path:
    """Land a finished lane's *branch* in *origin* and move `lane_session_dir`
    under `<origin_state>/runs/`. Returns the imported run dir.

    Refuses (SubrunError) to overwrite an existing branch in *origin* or an
    existing run dir at the destination -- checked before either the fetch or
    the move, so a refusal touches neither.

    A lane that never committed has no branch to land, and its record imports
    all the same, so the reason it stopped stays reachable under
    `<origin_state>/runs/`.
    """
    if branch_exists(origin, branch):
        raise SubrunError(f"branch {branch!r} already exists in {origin}")
    dest_session_dir = bucket_dir(origin_state, "runs") / lane_session_dir.name
    if dest_session_dir.exists():
        raise SubrunError(f"run dir already exists: {dest_session_dir}")
    if branch_exists(lane_repo, branch):
        try:
            fetch_branch(origin, lane_repo, f"{branch}:{branch}")
        except GitError as exc:
            raise SubrunError(f"fetch {branch!r} from {lane_repo} failed: {exc}") from exc
    dest_session_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(lane_session_dir), str(dest_session_dir))
    return dest_session_dir
