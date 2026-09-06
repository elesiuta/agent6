# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Unit regressions for background command lifecycle bookkeeping."""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any, cast

import pytest

from agent6.sandbox.jail import BackgroundStatus, Stopped
from agent6.tools import background
from agent6.tools.background import BackgroundError, BackgroundShells, roster_from_dir
from agent6.types import BackgroundHandoff, ChildSnapshot, JailPolicy


class _Job:
    def __init__(self, *, stop_error: str = "") -> None:
        self.stopped = False
        self.running = True
        self.stop_error = stop_error

    def status(self) -> BackgroundStatus:
        return BackgroundStatus(
            running=self.running, returncode=None if self.running else -9, error=""
        )

    def stop(self) -> str:
        self.stopped = True
        if not self.stop_error:
            self.running = False
        return self.stop_error


class _Session:
    def __init__(self, survivors: frozenset[int] = frozenset()) -> None:
        self.stopped: list[int] = []
        self.survivors = survivors

    def open_job(self, _pid: int, _before: ChildSnapshot) -> None:
        pass

    def status_background(self, _pid: int) -> BackgroundStatus:
        return BackgroundStatus(running=True, returncode=None, error="")

    def stop_background(self, pid: int) -> Stopped:
        self.stopped.append(pid)
        return Stopped(returncode=-9, survivors=self.survivors)

    def sweep_for(self, _pid: int, _before: ChildSnapshot) -> frozenset[int]:
        return frozenset()


def _fail_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    real_write = Path.write_text

    def write_text(self: Path, *args: Any, **kwargs: Any) -> int:
        if self.name == "meta.json":
            raise OSError("disk full")
        return real_write(self, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", write_text)


def test_start_stops_a_command_when_its_metadata_cannot_be_recorded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failed roster write must not leave a started command unreachable."""
    job = _Job()

    def start_in_jail(*_args: object, **_kwargs: object) -> _Job:
        return job

    monkeypatch.setattr(background, "start_in_jail", start_in_jail)
    _fail_metadata(monkeypatch)
    shells = BackgroundShells(tmp_path / "shells")

    with pytest.raises(BackgroundError, match="could not record"):
        shells.start(("sleep", "60"), lambda _a, _rw: cast(JailPolicy, object()))

    assert job.stopped
    assert shells.roster() == []


def test_a_command_is_still_reachable_when_registration_and_its_stop_fail(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failed cleanup must stay in memory so teardown can retry it."""
    job = _Job(stop_error="pid 42 survived SIGKILL")

    def start_in_jail(*_args: object, **_kwargs: object) -> _Job:
        return job

    monkeypatch.setattr(background, "start_in_jail", start_in_jail)
    _fail_metadata(monkeypatch)
    shells = BackgroundShells(tmp_path / "shells")

    with pytest.raises(BackgroundError, match="stopping it failed"):
        shells.start(("sleep", "60"), lambda _a, _rw: cast(JailPolicy, object()))

    assert [view.state for view in shells.roster()] == ["stop failed"]
    job.stop_error = ""
    assert shells.stop("bg1").state == "stopped"


def test_stop_all_closes_every_log_descriptor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Run teardown releases the raw descriptors held for safe output reads."""
    job = _Job()

    def start_in_jail(*_args: object, **_kwargs: object) -> _Job:
        return job

    monkeypatch.setattr(background, "start_in_jail", start_in_jail)
    shells = BackgroundShells(tmp_path / "shells")
    view = shells.start(("sleep", "60"), lambda _a, _rw: cast(JailPolicy, object()))
    log_fd = shells._get(view.id).log_fd  # pyright: ignore[reportPrivateUsage]
    os.fstat(log_fd)

    shells.stop_all()

    with pytest.raises(OSError):
        os.fstat(log_fd)
    assert shells.stop_all() == []


def test_the_disk_roster_skips_metadata_that_is_not_an_object(tmp_path: Path) -> None:
    """One malformed shell record must not break every roster surface."""
    shell = tmp_path / "shells" / "bg1"
    shell.mkdir(parents=True)
    (shell / "meta.json").write_text("[]", encoding="utf-8")

    assert roster_from_dir(tmp_path / "shells") == []


def test_the_disk_roster_tolerates_its_root_disappearing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Concurrent session cleanup between the existence check and listing is harmless."""
    root = tmp_path / "shells"
    root.mkdir()
    real_iterdir = Path.iterdir

    def iterdir(self: Path) -> Iterator[Path]:
        if self == root:
            raise FileNotFoundError(root)
        return real_iterdir(self)

    monkeypatch.setattr(Path, "iterdir", iterdir)

    assert roster_from_dir(root) == []


def test_adopt_stops_a_command_when_its_metadata_cannot_be_recorded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A handed-back command is live before its roster write, like a fresh start."""
    log = tmp_path / "handoff.log"
    log.write_text("", encoding="utf-8")
    session = _Session()
    _fail_metadata(monkeypatch)
    shells = BackgroundShells(tmp_path / "shells")
    handoff = BackgroundHandoff(
        argv=("sleep", "60"),
        pid=42,
        log=str(log),
        stdout="",
        stderr="",
        duration_s=900.0,
        before=ChildSnapshot(1, frozenset()),
    )

    with pytest.raises(BackgroundError, match="could not record"):
        shells.adopt(handoff, session=cast(Any, session))

    assert session.stopped == [42]
    assert shells.roster() == []


def test_adopt_retains_a_command_when_its_log_and_cleanup_fail(tmp_path: Path) -> None:
    """A failed cleanup remains reachable for a later stop and teardown retry."""
    session = _Session(frozenset({777}))
    shells = BackgroundShells(tmp_path / "shells")
    handoff = BackgroundHandoff(
        argv=("sleep", "60"),
        pid=42,
        log=str(tmp_path / "missing.log"),
        stdout="",
        stderr="",
        duration_s=900.0,
        before=ChildSnapshot(1, frozenset()),
    )

    with pytest.raises(BackgroundError, match="stopping it failed"):
        shells.adopt(handoff, session=cast(Any, session))

    assert [view.state for view in shells.roster()] == ["stop failed"]
    session.survivors = frozenset()
    shells.stop("bg1")
    assert session.stopped == [42, 42]
