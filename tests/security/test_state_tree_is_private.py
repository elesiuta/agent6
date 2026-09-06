# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Every directory a writer creates under the state base is 0700 whatever the
process umask. The base shields transcripts, memory and run history from
other local users, and a writer that reached a fresh base through a plain
`mkdir(parents=True)` (`agent6 init`, the first command on a new machine)
left it at the umask's 755 for good: `mkdir_for_real_user` never re-chmods
a directory that exists."""

from __future__ import annotations

import os
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest

from agent6 import memory, paths
from agent6.events import EventSink
from agent6.graph.storage import append_jsonl
from agent6.init import init_workspace
from agent6.machine.journal import MachineJournal, machine_lock, write_source, write_stop_request
from agent6.providers.types import TranscriptSink
from agent6.sessions.ipc import approvals_dir, questions_dir, register_frontend, set_session_allow
from agent6.sessions.layout import SessionLayout, machines_root
from agent6.tools.background import BackgroundShells


def _session(state: Path) -> Path:
    return SessionLayout(state, "s1").session_dir


def _machine_lock(root: Path) -> None:
    with machine_lock(root):
        pass


WRITERS: dict[str, Callable[[Path, Path], object]] = {
    "init": lambda repo, state: init_workspace(repo),
    "memory add": lambda repo, state: memory.add(state, "note", "body"),
    "memory decision": lambda repo, state: memory.record_decision(
        state, question="q", answer="a", session="s1"
    ),
    "session layout": lambda repo, state: SessionLayout(state, "s1").ensure(),
    "approvals dir": lambda repo, state: approvals_dir(_session(state)),
    "questions dir": lambda repo, state: questions_dir(_session(state)),
    "session grant": lambda repo, state: set_session_allow(_session(state), "command"),
    "frontend claim": lambda repo, state: register_frontend(_session(state), os.getpid()),
    "event sink": lambda repo, state: EventSink(SessionLayout(state, "s1").logs_path).emit("x"),
    "transcripts": lambda repo, state: TranscriptSink(SessionLayout(state, "s1").transcripts_dir),
    "background shells": lambda repo, state: BackgroundShells(_session(state)),
    "graph append": lambda repo, state: append_jsonl(
        SessionLayout(state, "s1").graph_dir / "g.jsonl", {"k": 1}
    ),
    "machine journal": lambda repo, state: MachineJournal(machines_root(state) / "m").ensure_dirs(),
    "machine lock": lambda repo, state: _machine_lock(machines_root(state) / "m"),
    "machine source": lambda repo, state: write_source(machines_root(state) / "m", "x = 1\n"),
    "machine stop": lambda repo, state: write_stop_request(machines_root(state) / "m"),
}


@pytest.fixture
def umask_022() -> Iterator[None]:
    old = os.umask(0o022)
    try:
        yield
    finally:
        os.umask(old)


def _open_dirs(base: Path) -> list[str]:
    return [
        str(p.relative_to(base.parent))
        for p in (base, *base.rglob("*"))
        if p.is_dir() and (p.stat().st_mode & 0o777) != 0o700
    ]


@pytest.mark.parametrize("writer", sorted(WRITERS))
def test_every_dir_a_writer_creates_under_the_state_base_is_0700(
    writer: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, umask_022: None
) -> None:
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "cfg"))
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / ".git").mkdir()
    state = paths.state_dir(repo)
    assert not paths.state_base().exists()
    WRITERS[writer](repo, state)
    assert paths.state_base().is_dir(), "the writer created nothing under the state base"
    assert _open_dirs(paths.state_base()) == []
