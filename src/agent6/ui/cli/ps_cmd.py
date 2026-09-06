# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""`agent6 ps`: live agent6 sessions across every repository on this machine.

`sessions list` is per-repo (the cwd's state dir); this walks the whole state
base so a detached run is findable from anywhere, with the directory to cd to
and the id to attach."""

from __future__ import annotations

from pathlib import Path

from agent6.paths import repo_root_of_id, state_base
from agent6.sessions.ipc import frontend_is_live, read_worker_pid, worker_is_alive
from agent6.sessions.layout import SESSION_BUCKETS
from agent6.ui.cli._common import home_contracted
from agent6.viewmodel.format import status_label
from agent6.viewmodel.listing import summarize_session_dir
from agent6.viewmodel.machine_state import summarize_machine_dir


def cmd_ps() -> int:
    """Print one row per live session: directory, id, mode, status, pid, and
    whether a front end is attached. Liveness is the worker-pid rule every
    listing uses (a foreign-owned or reused pid reads dead)."""
    base = state_base()
    # Keyed on the real session dir: a fan-out lane is linked under its
    # coordinator repo as well as its own, and the row with a repo wins.
    rows_by_dir: dict[Path, tuple[str, str, str, str, str, str]] = {}
    rows: list[tuple[str, str, str, str, str, str]] = []
    if base.is_dir():
        for repo_dir in sorted(base.iterdir()):
            root = repo_root_of_id(repo_dir.name)
            # An elided-hash id is not reversible to a path: the cell says so
            # instead of offering a state-dir name the cd line cannot use.
            where = home_contracted(str(root)) if root is not None else f"? ({repo_dir.name})"
            for bucket in SESSION_BUCKETS:
                bucket_path = repo_dir / "sessions" / bucket
                if not bucket_path.is_dir():
                    continue
                for sdir in sorted(bucket_path.iterdir()):
                    real = sdir.resolve()
                    if (
                        not sdir.is_dir()
                        or not worker_is_alive(sdir)
                        or (real in rows_by_dir and root is None)
                    ):
                        continue
                    summary = summarize_session_dir(sdir)
                    pid = read_worker_pid(sdir)
                    rows_by_dir[real] = (
                        where,
                        sdir.name,
                        summary.mode,
                        status_label(summary.status, summary.reason),
                        str(pid) if pid is not None else "?",
                        "attached" if frontend_is_live(sdir) else "",
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
                    pid = read_worker_pid(mdir)
                    rows.append(
                        (
                            where,
                            mdir.name,
                            "machine",
                            status_label(machine.status, machine.reason),
                            str(pid) if pid is not None else "?",
                            "",
                        )
                    )
    rows = [*rows_by_dir.values(), *rows]
    if not rows:
        print("no live agent6 sessions.")
        return 0
    headers = ("directory", "id", "mode", "status", "pid", "front-end")
    widths = [max(len(headers[i]), *(len(r[i]) for r in rows)) for i in range(len(headers))]
    print("  ".join(h.ljust(widths[i]) for i, h in enumerate(headers)).rstrip())
    for r in rows:
        print("  ".join(c.ljust(widths[i]) for i, c in enumerate(r)).rstrip())
    print(
        "\nattach with: cd <directory> && agent6 attach <id>"
        "  (a machine: agent6 machine status <id>;"
        " ? = directory not recoverable from the id)"
    )
    return 0
