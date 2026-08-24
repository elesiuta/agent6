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
from agent6.viewmodel.format import status_label
from agent6.viewmodel.listing import summarize_session_dir


def _home_contracted(path: Path) -> str:
    home = str(Path.home())
    s = str(path)
    return "~" + s[len(home) :] if s == home or s.startswith(home + "/") else s


def cmd_ps() -> int:
    """Print one row per live session: directory, id, mode, status, pid, and
    whether a front end is attached. Liveness is the worker-pid rule every
    listing uses (a foreign-owned or reused pid reads dead)."""
    base = state_base()
    rows: list[tuple[str, str, str, str, str, str]] = []
    if base.is_dir():
        for repo_dir in sorted(base.iterdir()):
            sessions = repo_dir / "sessions"
            if not sessions.is_dir():
                continue
            root = repo_root_of_id(repo_dir.name)
            # An elided-hash id is not reversible to a path: the cell says so
            # instead of offering a state-dir name the cd line cannot use.
            where = _home_contracted(root) if root is not None else f"? ({repo_dir.name})"
            for bucket in SESSION_BUCKETS:
                bucket_path = sessions / bucket
                if not bucket_path.is_dir():
                    continue
                for sdir in sorted(bucket_path.iterdir()):
                    if not sdir.is_dir() or not worker_is_alive(sdir):
                        continue
                    summary = summarize_session_dir(sdir)
                    pid = read_worker_pid(sdir)
                    rows.append(
                        (
                            where,
                            sdir.name,
                            summary.mode,
                            status_label(summary.status, summary.reason),
                            str(pid) if pid is not None else "?",
                            "attached" if frontend_is_live(sdir) else "",
                        )
                    )
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
        "  (? = directory not recoverable from the id)"
    )
    return 0
