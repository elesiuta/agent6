# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""`agent6 ps`: live agent6 sessions across every repository on this machine.

`sessions list` is per-repo (the cwd's state dir); this walks the whole state
base so a detached run is findable from anywhere, with the directory to cd to
and the id to attach."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from agent6.paths import repo_root_of_id, state_base
from agent6.sessions.ipc import frontend_is_live, read_worker_pid, worker_is_alive
from agent6.sessions.layout import SESSION_BUCKETS
from agent6.ui.cli._common import home_contracted
from agent6.viewmodel.format import lane_count, lane_id_cell, status_label
from agent6.viewmodel.listing import ListingRow, SessionSummary, nested_rows, summarize_session_dir
from agent6.viewmodel.machine_state import summarize_machine_dir


@dataclass(frozen=True, slots=True)
class _Row:
    directory: str | None  # None when the id does not lead back to a checkout
    repo_id: str
    id: str
    mode: str
    status: str
    pid: int | None
    attached: bool
    coordinator: str = ""  # a lane: the live coordinator row it nests under
    lanes: tuple[_Row, ...] = ()

    def cells(self, id_cell: str) -> tuple[str, ...]:
        where = self.directory if self.directory is not None else f"? ({self.repo_id})"
        return (
            where,
            id_cell,
            self.mode,
            self.status,
            str(self.pid) if self.pid is not None else "?",
            "attached" if self.attached else "",
        )


@dataclass(frozen=True, slots=True)
class _Live:
    """The live sessions (rows and their summaries, by id) and machine rows."""

    rows: dict[str, _Row]
    summaries: dict[str, SessionSummary]
    machines: list[_Row]

    def nested(self) -> list[_Row]:
        """The session rows with a fan-out's live lanes under it (the listing
        fold, `nested_rows`), then the machines."""

        def tree(row: ListingRow) -> _Row:
            own = self.rows[row.summary.session_id]
            return replace(own, lanes=tuple(tree(lane) for lane in row.lanes))

        return [*(tree(r) for r in nested_rows(self.summaries.values())), *self.machines]


def _live_rows() -> _Live:
    """Every live session and machine instance under the state base, one row each."""
    base = state_base()
    # Keyed on the real session dir: a fan-out lane is linked under its
    # coordinator repo as well as its own, and the origin's view (the link)
    # wins, so the lane lists beside the coordinator it nests under.
    rows_by_dir: dict[Path, _Row] = {}
    summaries_by_dir: dict[Path, SessionSummary] = {}
    rows: list[_Row] = []
    if base.is_dir():
        for repo_dir in sorted(base.iterdir()):
            root = repo_root_of_id(repo_dir.name)
            # An elided-hash id is not reversible to a path: the cell says so
            # instead of offering a state-dir name the cd line cannot use.
            where = home_contracted(str(root)) if root is not None else None
            for bucket in SESSION_BUCKETS:
                bucket_path = repo_dir / "sessions" / bucket
                if not bucket_path.is_dir():
                    continue
                for sdir in sorted(bucket_path.iterdir()):
                    real = sdir.resolve()
                    if (
                        not sdir.is_dir()
                        or not worker_is_alive(sdir)
                        or (real in rows_by_dir and not sdir.is_symlink())
                    ):
                        continue
                    summary = summarize_session_dir(sdir)
                    summaries_by_dir[real] = summary
                    rows_by_dir[real] = _Row(
                        where,
                        repo_dir.name,
                        sdir.name,
                        summary.mode,
                        status_label(summary.status, summary.reason),
                        read_worker_pid(sdir),
                        frontend_is_live(sdir),
                        coordinator=summary.coordinator,
                    )
            # A machine instance is a live session too (a repo may hold only
            # machines, and no sessions/ at all): its worker.pid sits at the
            # instance root, one dir per machine name. Its status is the
            # word every machine surface shows (a live worker in a wait or
            # blocked on an approval is "waiting").
            machines = repo_dir / "machines"
            if machines.is_dir():
                for mdir in sorted(machines.iterdir()):
                    if not mdir.is_dir() or not worker_is_alive(mdir):
                        continue
                    machine = summarize_machine_dir(mdir)
                    rows.append(
                        _Row(
                            where,
                            repo_dir.name,
                            mdir.name,
                            "machine",
                            status_label(machine.status, machine.reason),
                            read_worker_pid(mdir),
                            False,
                        )
                    )
    return _Live(
        rows={r.id: r for r in rows_by_dir.values()},
        summaries={s.session_id: s for s in summaries_by_dir.values()},
        machines=rows,
    )


def cmd_ps(*, as_json: bool = False, lanes: bool = False) -> int:
    """Print one row per live session: directory, id, mode, status, pid, and
    whether a front end is attached. Liveness is the worker-pid rule every
    listing uses (a foreign-owned or reused pid reads dead). A fan-out's live
    lanes nest under its row: folded into a count, listed indented with
    *lanes*; the JSON row nests them always."""
    rows = _live_rows().nested()
    if as_json:
        print(json.dumps([asdict(r) for r in rows], indent=2))
        return 0
    if not rows:
        print("no live agent6 sessions.")
        return 0
    headers = ("directory", "id", "mode", "status", "pid", "front-end")
    cells: list[tuple[str, ...]] = []

    def emit(r: _Row, depth: int) -> None:
        if depth:
            cells.append(r.cells(lane_id_cell(r.id, depth)))
        else:
            folded = f" ({lane_count(len(r.lanes))})" if r.lanes and not lanes else ""
            cells.append(r.cells(r.id + folded))
        if lanes:
            for lane in r.lanes:
                emit(lane, depth + 1)

    for r in rows:
        emit(r, 0)
    widths = [max(len(headers[i]), *(len(c[i]) for c in cells)) for i in range(len(headers))]
    print("  ".join(h.ljust(widths[i]) for i, h in enumerate(headers)).rstrip())
    for c in cells:
        print("  ".join(v.ljust(widths[i]) for i, v in enumerate(c)).rstrip())
    print(
        "\nattach with: cd <directory> && agent6 attach <id>"
        "  (a machine: agent6 machine status <id>;"
        " ? = directory not recoverable from the id)"
    )
    return 0
