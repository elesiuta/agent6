# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Append-only journal, blackboard snapshots, and the single-writer lock for one
machine instance. The journal is the source of truth: the pure reducer validates
each impure observation, the validated fact is appended as a JournalEvent, and
the returned blackboard replaces the current one only then. Replaying the events
reproduces the reducer's path exactly.

The recorded observations are a tool's exit code and stdout, a wait's resolved
wake instant, and a branch's chosen clause (§5.1); the reducer reads them back
instead of re-touching the world.

Events are read back from disk, so they re-enter at a trust boundary and are
re-validated by pydantic (`extra="forbid", frozen=True`), exactly like the
machine spec itself. Snapshots are an optimisation for human inspection and
fast status; correctness depends only on the journal.

Layout under the per-repo state dir (`machines/<id>/`) (§5.3)::

    machine.asm.toml     # the exact source the run was started from (for replay)
    journal.jsonl        # append-only, fsync'd, one event per line
    snapshots/<n>.json   # blackboard + current state, atomic temp+rename
    machine.lock         # flock single-writer guard
    signal               # optional operator poke consumed by a `wait` state
    wait.json            # persisted next-wake instant for --exit-on-wait mode
"""

from __future__ import annotations

import json
import os
import shutil
from collections.abc import Generator
from contextlib import contextmanager, suppress
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError
from pydantic_core import PydanticSerializationError

from agent6.machine.model import MachineError
from agent6.paths import mkdir_for_real_user
from agent6.portable import atomic_write, lock_exclusive, unlock

__all__ = [
    "AgentFact",
    "BranchFact",
    "Fact",
    "JournalError",
    "JournalEvent",
    "MachineBegin",
    "MachineEnd",
    "MachineJournal",
    "MachineNotify",
    "PendingWait",
    "Snapshot",
    "StepEvent",
    "ToolFact",
    "WaitFact",
    "machine_lock",
    "read_source",
    "write_source",
]

_MODEL_CONFIG = ConfigDict(extra="forbid", frozen=True)


class JournalError(MachineError):
    """Raised when on-disk journal state (journal, pending wait, source, lock) is
    missing, corrupt, or unusable.

    A `MachineError` subclass so every surface that degrades on a
    broken machine file (hub listing, machine page, SSE stream) degrades the
    same way on a broken journal instead of crashing.
    """

    def __init__(self, message: str) -> None:
        super().__init__([message])


def _now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="microseconds")


# --------------------------------------------------------------------------
# Facts, the impure observation a single state execution produced.
# --------------------------------------------------------------------------


class ToolFact(BaseModel):
    model_config = _MODEL_CONFIG

    kind: Literal["tool"] = "tool"
    exit_code: int
    stdout: str
    timed_out: bool
    # The tool's captured stderr, so a failing machine tool is debuggable from
    # the journal (routing keys off exit_code/stdout only, so this never affects
    # the reducer). Additive with a default: journal lines written before this
    # field still parse (extra="forbid" only rejects UNKNOWN keys, not a missing
    # defaulted one), keeping old instances replayable.
    stderr: str = ""


class WaitFact(BaseModel):
    model_config = _MODEL_CONFIG

    kind: Literal["wait"] = "wait"
    # `None` for a wait with no timer (parks until a `signal` poke, §4.3).
    wake_epoch: float | None = None
    woke_by: Literal["tick", "signal"]
    # The poke payload delivered by a `signal` wake, journaled so a replay
    # re-reads the identical input. `None` for a bare poke or a `tick`.
    payload: Any = None


class BranchFact(BaseModel):
    model_config = _MODEL_CONFIG

    kind: Literal["branch"] = "branch"
    clause_index: int = Field(ge=0)


class AgentFact(BaseModel):
    model_config = _MODEL_CONFIG

    kind: Literal["agent"] = "agent"
    outcome: Literal["ok", "failed", "budget_exhausted", "timeout"]
    reason: str
    payload: dict[str, Any] | None = None
    usd: float = 0.0
    # True when `usd` is a known under-estimate (an unpriced model contributed
    # $0); machine status renders it with the shared '~' marker. Defaults False
    # so old journals parse unchanged.
    usd_partial: bool = False
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)


Fact = Annotated[ToolFact | WaitFact | BranchFact | AgentFact, Field(discriminator="kind")]


# --------------------------------------------------------------------------
# Events, one journal line each.
# --------------------------------------------------------------------------


class MachineBegin(BaseModel):
    model_config = _MODEL_CONFIG

    type: Literal["machine.begin"] = "machine.begin"
    ts: str
    machine: str
    version: int


class StepEvent(BaseModel):
    model_config = _MODEL_CONFIG

    type: Literal["step"] = "step"
    ts: str
    seq: int = Field(ge=0)
    state: str
    label: str
    goto: str
    fact: Fact


class MachineNotify(BaseModel):
    """A state's `notify` message, journaled on entry (§4.3).

    Presentation only: it adds no edge and does not affect the reducer or
    routing. Front-ends render it as an ephemeral notification; the operator
    notify hook fires on it out-of-band.
    """

    model_config = _MODEL_CONFIG

    type: Literal["machine.notify"] = "machine.notify"
    ts: str
    state: str
    message: str
    level: Literal["info", "warn", "error"] = "info"


class MachineEnd(BaseModel):
    model_config = _MODEL_CONFIG

    type: Literal["machine.end"] = "machine.end"
    ts: str
    status: Literal["ok", "failed"]
    reason: str
    state: str
    transitions: int = Field(ge=0)
    # Spend of a slice that ended WITHOUT a StepEvent to book it. A capture that
    # cannot be reduced halts before journaling the step (a poison fact would
    # re-crash every later replay), which also discarded the agent's real usd and
    # tokens: `machine run` then reported $0.0000 for a state that burned money.
    usd: float = 0.0
    usd_partial: bool = False
    input_tokens: int = 0
    output_tokens: int = 0


class AttemptSpend(BaseModel):
    """Metered spend of a state attempt the crash window orphaned.

    A supervisor death mid-agent-state leaves real provider spend with no
    StepEvent to book it and no MachineEnd to carry it; the per-state log
    still holds the totals. The RESUMING supervisor journals this before
    re-running the state, so the budget and every spend surface keep the
    billed slice. Bookkeeping only: it adds no edge and never moves the
    reducer."""

    model_config = _MODEL_CONFIG

    type: Literal["attempt.spend"] = "attempt.spend"
    ts: str
    seq: int = Field(ge=0)
    state: str
    usd: float = 0.0
    usd_partial: bool = False
    input_tokens: int = 0
    output_tokens: int = 0


JournalEvent = Annotated[
    MachineBegin | StepEvent | MachineNotify | MachineEnd | AttemptSpend,
    Field(discriminator="type"),
]

_EVENT_ADAPTER: TypeAdapter[Any] = TypeAdapter(JournalEvent)


# --------------------------------------------------------------------------
# Snapshot, blackboard + position, written after every transition.
# --------------------------------------------------------------------------


class Snapshot(BaseModel):
    model_config = _MODEL_CONFIG

    seq: int = Field(ge=0)
    state: str
    blackboard: dict[str, Any]


class PendingWait(BaseModel):
    """A `wait` armed by `--exit-on-wait` but not yet fired (§6).

    The absolute `wake_epoch` is computed once, when the wait is first
    reached, and persisted so that re-invocations by an external scheduler
    compare against the *same* instant rather than re-arming `every_secs`
    from a fresh `now` each tick. Deleted once the wait fires.
    """

    model_config = _MODEL_CONFIG

    state: str
    # `None` for a wait with no timer: it fires only on a `signal` poke, never
    # on a wake instant, so `--exit-on-wait` parks it until the operator pokes.
    wake_epoch: float | None = None

    @property
    def wake_at(self) -> str:
        """The wake instant as an ISO-8601 UTC timestamp, "" when there is none.

        Every surface that shows an operator when a parked machine wakes reads
        this, so `machine run --exit-on-wait` and `machine status` cannot render
        the same instant differently.
        """
        if self.wake_epoch is None:
            return ""
        return datetime.fromtimestamp(self.wake_epoch, tz=UTC).isoformat()


# --------------------------------------------------------------------------
# The journal directory.
# --------------------------------------------------------------------------


def scrub_lone_surrogates(value: Any) -> Any:
    """A parsed-JSON value with any lone surrogate replaced.

    Applied at the two trust boundaries that produce them -- a tool's captured
    stdout and a `machine poke` payload -- so the blackboard never holds one.
    Sanitizing only the journal writers moved the crash one step downstream
    instead of removing it: the next agent state serializes the blackboard into
    its request payload, and `model_dump_json` raises a
    `PydanticSerializationError` that no handler on that path catches.
    """
    try:
        json.dumps(value, ensure_ascii=False).encode("utf-8")
    except UnicodeEncodeError:
        clean = json.dumps(value, ensure_ascii=False).encode("utf-8", "replace").decode("utf-8")
        return json.loads(clean)
    return value


def dump_json(model: BaseModel, *, indent: int | None = None) -> str:
    """One journal/snapshot record as JSON, lone-surrogate safe.

    `json.loads` legally yields lone surrogates from `\\udXXX` escapes, and
    they reach these writers from a tool's captured stdout and a `machine poke`
    payload. `model_dump_json` raises on them, which crashed the run before it
    could journal a MachineEnd and re-crashed on every restart. Replace them
    (the same call EventSink makes for logs.jsonl) so the audit trail is written
    and stays valid UTF-8 for every reader."""
    try:
        return model.model_dump_json(indent=indent)
    except PydanticSerializationError:
        raw = json.dumps(
            model.model_dump(mode="json"),
            ensure_ascii=False,
            indent=indent,
            separators=None if indent is not None else (",", ":"),
            default=str,
        )
        return raw.encode("utf-8", "replace").decode("utf-8")


# How far back a torn-tail heal looks for the last committed newline before it
# falls back to reading the file. A journal line is one event; a tool fact
# carrying a command's output is the long case.
_TAIL_WINDOW = 1 << 20


class MachineJournal:
    """Append-only event log plus snapshots for one machine instance."""

    def __init__(self, root: Path, *, snapshot_keep: int = 5) -> None:
        # Number of recent snapshots to retain (0 = keep all); see
        # `[machine] snapshot_keep` in the config.
        self.snapshot_keep = snapshot_keep
        self.root = root
        self.journal_path = root / "journal.jsonl"
        self.snapshots_dir = root / "snapshots"
        self.source_path = root / "machine.asm.toml"
        self.signal_path = root / "signal"
        self.wait_path = root / "wait.json"

    def ensure_dirs(self) -> None:
        mkdir_for_real_user(self.snapshots_dir)

    def exists(self) -> bool:
        return self.journal_path.is_file()

    def begin(self, *, machine: str, version: int) -> None:
        self.append(MachineBegin(ts=_now_iso(), machine=machine, version=version))

    def append(self, event: BaseModel) -> None:
        """Append one event as a JSON line, fsync'd.

        Heals a torn previous append first: a committed line always ends in
        `\\n`, so a file that does not is a crash mid-write. Truncating the
        partial line off keeps this event on its own line instead of
        concatenating onto the fragment (which `read` would then reject).
        """
        self._heal_torn_tail()
        line = dump_json(event)
        with self.journal_path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")
            fh.flush()
            os.fsync(fh.fileno())

    def _heal_torn_tail(self) -> None:
        if not self.journal_path.is_file():
            return
        # Cheap common path: peek the last byte only.
        with self.journal_path.open("rb") as fh:
            if fh.seek(0, os.SEEK_END) == 0:
                return
            fh.seek(-1, os.SEEK_END)
            if fh.read(1) == b"\n":
                return
        # Truncate in place: reading the whole journal and writing it back
        # opens a window where a kill (or a concurrent reader) sees an EMPTY
        # journal -- the file every machine's correctness rests on. `truncate`
        # leaves it at either the old length or the new one.
        with self.journal_path.open("rb") as fh:
            size = fh.seek(0, os.SEEK_END)
            window = min(size, _TAIL_WINDOW)
            fh.seek(size - window)
            tail = fh.read(window)
        cut = tail.rfind(b"\n")
        if cut < 0:
            # No newline in the tail window: fall back to the whole file, which
            # is the only way to find the last committed line.
            cut = self.journal_path.read_bytes().rfind(b"\n")
            if cut < 0:
                os.truncate(self.journal_path, 0)  # one torn line, nothing committed
                return
            os.truncate(self.journal_path, cut + 1)
            return
        os.truncate(self.journal_path, size - window + cut + 1)

    def read(self) -> list[Any]:
        """Parse and validate every journal line in order."""
        if not self.journal_path.is_file():
            return []
        raw_lines = self.journal_path.read_bytes().split(b"\n")
        # split(b"\n"), NOT splitlines(): splitlines() also breaks on U+2028 /
        # U+2029 / U+0085 after decode, which `model_dump_json` writes literally
        # inside JSON strings, so a captured value containing one would shred a
        # single line into unparseable fragments and brick the instance.
        #
        # Split bytes before decoding: a crash can tear the final line in the
        # middle of a multibyte UTF-8 sequence. Dropping that byte tail first
        # keeps the committed prefix readable.
        if raw_lines and raw_lines[-1] != b"":
            raw_lines.pop()
        events: list[Any] = []
        for lineno, raw in enumerate(raw_lines, start=1):
            if not raw.strip():
                continue
            try:
                events.append(_EVENT_ADAPTER.validate_json(raw))
            except ValidationError as exc:
                raise JournalError(
                    f"corrupt journal line {lineno} in {self.journal_path}: {exc}"
                ) from exc
        return events

    def write_snapshot(self, snapshot: Snapshot) -> None:
        """Write a snapshot atomically (temp file + rename), pruning old ones.

        Recovery only ever reads `latest_snapshot` and replay rebuilds from
        the journal, so old snapshots are dead weight: a 10-minute-loop machine
        would otherwise accumulate ~150k files a year. Keep a short fixed tail
        (paranoia against a corrupt latest) and delete the rest.
        """
        mkdir_for_real_user(self.snapshots_dir)
        dest = self.snapshots_dir / f"{snapshot.seq}.json"
        atomic_write(dest, dump_json(snapshot, indent=2) + "\n")
        if self.snapshot_keep <= 0:
            return
        with suppress(OSError):
            for entry in self.snapshots_dir.iterdir():
                if (
                    entry.suffix == ".json"
                    and entry.stem.isdigit()
                    and int(entry.stem) <= snapshot.seq - self.snapshot_keep
                ):
                    with suppress(OSError):
                        entry.unlink()

    def latest_snapshot(self) -> Snapshot | None:
        """The newest readable snapshot, falling back through the retained tail.

        Snapshots are an inspection optimization (the journal is authoritative),
        and `write_snapshot` keeps a short tail expressly "against a corrupt
        latest". So a torn newest snapshot falls back to the next-older one, and
        only when none are readable do we return None instead of raising -- a
        single bad snapshot must not make `machine status` fail.
        """
        if not self.snapshots_dir.is_dir():
            return None
        seqs = sorted(
            (
                int(entry.stem)
                for entry in self.snapshots_dir.iterdir()
                if entry.suffix == ".json" and entry.stem.isdigit()
            ),
            reverse=True,
        )
        for seq in seqs:
            path = self.snapshots_dir / f"{seq}.json"
            try:
                return Snapshot.model_validate_json(path.read_text(encoding="utf-8"))
            except (ValidationError, OSError):
                continue
        return None

    def take_signal(self) -> tuple[bool, Any]:
        """Consume a pending operator poke, if any.

        Returns `(present, payload)`: `present` is True when a signal file was
        consumed; `payload` is the JSON the poke carried (`None` for a bare
        poke, an empty file, or an unparseable one -- a hand-touched signal is a
        valid bare wake).

        Claims the signal by renaming it to a private consume path first: `poke`
        renames a fresh signal into place from another process, so a
        read-then-unlink would destroy a poke that landed in between.

        The claim file OUTLIVES this call: it is deleted by `ack_signal` once
        the wake's StepEvent is durable, never here. A consume path already
        present is therefore an unacked claim (machine_lock guarantees no live
        second consumer): read IT rather than renaming over it, so a death
        anywhere before the ack re-delivers the same poke on restart.
        Delivery is at-least-once across the whole claim-to-step window, which
        a wake tolerates (a bare poke is a valid wake).
        """
        consume = self.signal_path.with_suffix(".consuming")
        if not consume.exists():
            try:
                self.signal_path.rename(consume)
            except FileNotFoundError:
                return False, None
        try:
            raw = consume.read_text(encoding="utf-8")
        except OSError:
            raw = ""
        if not raw.strip():
            return True, None
        try:
            return True, scrub_lone_surrogates(json.loads(raw))
        except json.JSONDecodeError:
            return True, None

    def ack_signal(self) -> None:
        """Discard the claimed poke once its wake's StepEvent is durable.

        Deleting on take made the poke's only remaining trace an un-fsynced
        return value, so a death between the take and the step append lost it
        with nothing to re-deliver."""
        self.signal_path.with_suffix(".consuming").unlink(missing_ok=True)

    def read_pending_poke(self) -> tuple[bool, Any]:
        """A poke not yet acked, without consuming it: `(present, payload)`.

        Reads the signal file, or the claim file of a take whose step is not
        yet durable; both are a poke the machine has still to act on."""
        for path in (self.signal_path.with_suffix(".consuming"), self.signal_path):
            try:
                raw = path.read_text(encoding="utf-8")
            except FileNotFoundError:
                continue
            if not raw.strip():
                return True, None
            try:
                return True, scrub_lone_surrogates(json.loads(raw))
            except json.JSONDecodeError:
                return True, None
        return False, None

    def poke(self, payload: Any = None) -> None:
        """Drop a signal file so a blocked or armed `wait` wakes (§6 signal-poke).

        The optional *payload* travels to the waking `wait` as its `signal`
        payload (journaled, replay-safe) for the next tool to read.

        Atomic (temp + fsync + rename) like every other journal write: the
        engine's `take_signal` polls from another process, and a plain write
        exposes an empty/partial file it would consume as a bare poke,
        dropping the payload.
        """
        mkdir_for_real_user(self.root)
        atomic_write(self.signal_path, json.dumps(payload))

    def read_pending_wait(self) -> PendingWait | None:
        if not self.wait_path.is_file():
            return None
        try:
            return PendingWait.model_validate_json(self.wait_path.read_text(encoding="utf-8"))
        except ValidationError as exc:
            # The engine cannot guess a wake instant from this: firing early or
            # skipping the wait are both worse than refusing. Name the remedy,
            # like every other refusal -- deleting the file re-arms the wait
            # from the state itself on the next run.
            raise JournalError(
                f"corrupt pending wait {self.wait_path}: {exc}\n"
                f"  delete it to re-arm the wait from the machine's own state:"
                f" rm {self.wait_path}"
            ) from exc

    def write_pending_wait(self, pending: PendingWait) -> None:
        """Persist the armed next-wake instant atomically (temp file + rename)."""
        mkdir_for_real_user(self.root)
        atomic_write(self.wait_path, dump_json(pending, indent=2) + "\n")

    def clear_pending_wait(self) -> None:
        self.wait_path.unlink(missing_ok=True)


@contextmanager
def machine_lock(root: Path) -> Generator[None]:
    """Single-writer guard for one machine id (§6). Refuses a second runner."""
    mkdir_for_real_user(root)
    lock_path = root / "machine.lock"
    fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o644)
    try:
        try:
            lock_exclusive(fd, blocking=False)
        except OSError as exc:
            raise JournalError(f"machine is already running (lock held): {lock_path}") from exc
        try:
            yield
        finally:
            unlock(fd)
    finally:
        os.close(fd)


def write_source(root: Path, text: str) -> None:
    """Persist the exact `.asm.toml` source the run started from (for replay)."""
    mkdir_for_real_user(root)
    atomic_write(root / "machine.asm.toml", text)


def read_source(root: Path) -> str:
    path = root / "machine.asm.toml"
    if not path.is_file():
        raise JournalError(f"no persisted machine source at {path}")
    return path.read_text(encoding="utf-8")


def write_stop_request(root: Path) -> None:
    """Ask the live machine to park at its next transition boundary.

    A marker, not a kill (the `sessions stop` semantics): the state in flight
    finishes and journals its fact, then the engine returns a "stopped" result
    without a MachineEnd -- the instance stays resumable."""
    mkdir_for_real_user(root)
    (root / "stop").touch()


def stop_requested(root: Path) -> bool:
    return (root / "stop").is_file()


def clear_stop_request(root: Path) -> None:
    with suppress(FileNotFoundError):
        (root / "stop").unlink()


def write_bundle(root: Path, machine_path: Path) -> None:
    """Persist the exact executable bundle the instance starts from: the
    `.asm.toml` source plus its `scripts/` tree. Replay evidence, and the
    baseline `bundle_drift` holds every continuation to."""
    write_source(root, machine_path.read_text(encoding="utf-8"))
    dst = root / "scripts"
    shutil.rmtree(dst, ignore_errors=True)
    scripts = machine_path.parent / "scripts"
    if scripts.is_dir():
        shutil.copytree(scripts, dst)


def bundle_drift(root: Path, machine_path: Path) -> str | None:
    """The first difference between the working bundle and the instance's
    recorded one, or None when they match byte for byte.

    A live instance runs the logic it recorded; an edit takes effect on a new
    instance. Byte comparison against the recorded copy keeps that copy the
    single source of truth -- no digest to go stale, no mtime heuristics."""
    recorded_asm = root / "machine.asm.toml"
    if not recorded_asm.is_file():
        return f"no recorded machine source at {recorded_asm}"
    if recorded_asm.read_bytes() != machine_path.read_bytes():
        return f"{machine_path.name} differs from the recorded {recorded_asm}"
    working = _tree_files(machine_path.parent / "scripts")
    recorded = _tree_files(root / "scripts")
    for rel in sorted(recorded.keys() - working.keys()):
        return f"scripts/{rel} was removed after the instance began"
    for rel in sorted(working.keys() - recorded.keys()):
        return f"scripts/{rel} was added after the instance began"
    for rel in sorted(working):
        if working[rel] != recorded[rel]:
            return f"scripts/{rel} differs from the instance's recorded copy"
    return None


def _tree_files(base: Path) -> dict[str, bytes]:
    if not base.is_dir():
        return {}
    return {
        p.relative_to(base).as_posix(): p.read_bytes()
        for p in sorted(base.rglob("*"))
        if p.is_file()
    }
