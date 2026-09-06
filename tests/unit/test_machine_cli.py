# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""CLI tests for Phase 4 machine ergonomics: status, poke, run --exit-on-wait."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

from agent6.config.layer import resolved_state_dir
from agent6.machine import MachineJournal
from agent6.sessions.ipc import clear_worker_pid, write_worker_pid
from agent6.ui.cli import main


def _git_init(path: Path) -> None:
    subprocess.run(["git", "init", "-q", "-b", "main", str(path)], check=True)
    subprocess.run(["git", "-C", str(path), "config", "user.email", "t@t"], check=True)
    subprocess.run(["git", "-C", str(path), "config", "user.name", "t"], check=True)


WAITER_DELAYED = """
machine = "waiter_delayed"
version = 1
initial = "poll"

[budget]
max_usd = 1.0
max_transitions = 100

[vars.operator]
secs = { type = "int", value = 3600 }

[states.poll]
kind = "wait"
every_secs = "{{ secs }}"
on = { tick = "done", signal = "woken" }

[states.done]
kind = "terminal"
status = "ok"
reason = "ticked"

[states.woken]
kind = "terminal"
status = "ok"
reason = "signalled"
"""


def _write_machine(tmp_path: Path) -> Path:
    f = tmp_path / "waiter.asm.toml"
    f.write_text(WAITER_DELAYED, encoding="utf-8")
    return f


def test_run_exit_on_wait_yields_waiting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    f = _write_machine(tmp_path)
    code = main(["machine", "run", str(f), "--exit-on-wait"])
    assert code == 0
    out = capsys.readouterr().out
    assert "WAITING" in out
    # The wait was armed and persisted.
    root = resolved_state_dir(tmp_path) / "machines" / "waiter_delayed"
    pending = MachineJournal(root).read_pending_wait()
    assert pending is not None
    assert pending.state == "poll"


def test_run_prints_a_notify_on_the_foreground_terminal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # A notify was journal-only: the foreground run is its own watcher, so the
    # message must land on the terminal too (attach and the web already showed
    # it). The operator [machine.notify].on_event hook is unset here.
    monkeypatch.chdir(tmp_path)
    f = tmp_path / "waiter.asm.toml"
    f.write_text(
        WAITER_DELAYED.replace(
            'kind = "wait"',
            'kind = "wait"\nnotify = { message = "parked, awaiting a poke", level = "warn" }',
        ),
        encoding="utf-8",
    )
    code = main(["machine", "run", str(f), "--exit-on-wait"])
    assert code == 0
    err = capsys.readouterr().err
    assert "[agent6] notify [warn] 'poll': parked, awaiting a poke" in err


def test_status_reports_waiting_state_and_spend(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    f = _write_machine(tmp_path)
    assert main(["machine", "run", str(f), "--exit-on-wait"]) == 0
    capsys.readouterr()  # drop run output
    # `machine run --exit-on-wait` exits the process, so the worker pid is dead;
    # in-process it is this live pytest, so clear it to model the parked reality.
    root = resolved_state_dir(tmp_path) / "machines" / "waiter_delayed"
    clear_worker_pid(root)
    code = main(["machine", "status", "waiter_delayed"])
    assert code == 0
    out = capsys.readouterr().out
    assert "waiter_delayed" in out
    # A parked instance reads "waiting" (the word run --exit-on-wait/web use), not
    # the engine's raw "incomplete".
    assert "status: waiting" in out
    # A timed wait wakes on its own; the poke is offered as the way to wake it NOW.
    assert "waiting in 'poll': wakes at " in out
    assert "a poke wakes it now: agent6 machine poke waiter_delayed" in out
    assert "spend: $0.0000" in out


def test_status_hints_poke_for_a_live_foreground_wait(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A foreground `machine run` blocked in a wait persists the wait record
    BEFORE it sleeps (the same wait.json --exit-on-wait leaves), so a live
    worker in a wait carries one. The readout gated the poke line on the
    record's absence, so a blocking wait never printed it."""
    monkeypatch.chdir(tmp_path)
    f = _write_machine(tmp_path)
    assert main(["machine", "run", str(f), "--exit-on-wait"]) == 0
    capsys.readouterr()  # drop run output
    root = resolved_state_dir(tmp_path) / "machines" / "waiter_delayed"
    assert MachineJournal(root).read_pending_wait() is not None
    # The run cleared its own pid on exit; re-stamp a live worker (this pytest).
    write_worker_pid(root, os.getpid())
    code = main(["machine", "status", "waiter_delayed"])
    assert code == 0
    out = capsys.readouterr().out
    assert "status: waiting" in out
    assert "waiting in 'poll': wakes at " in out
    assert "a poke wakes it now: agent6 machine poke waiter_delayed" in out


def test_status_shows_a_pending_poke_until_it_is_acked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A poke the machine has not acted on is state the readout owes the
    operator: its payload shows from the poke until the wake's step is acked,
    a claimed-but-unacked take included."""
    monkeypatch.chdir(tmp_path)
    f = _write_machine(tmp_path)
    assert main(["machine", "run", str(f), "--exit-on-wait"]) == 0
    capsys.readouterr()
    root = resolved_state_dir(tmp_path) / "machines" / "waiter_delayed"
    clear_worker_pid(root)
    assert main(["machine", "poke", "waiter_delayed", "--message", "go"]) == 0
    capsys.readouterr()
    assert main(["machine", "status", "waiter_delayed"]) == 0
    assert "poke pending: 'go'" in capsys.readouterr().out
    journal = MachineJournal(root)
    assert journal.take_signal() == (True, "go")  # claimed, the step not yet durable
    assert main(["machine", "status", "waiter_delayed"]) == 0
    assert "poke pending: 'go'" in capsys.readouterr().out
    journal.ack_signal()
    assert main(["machine", "status", "waiter_delayed"]) == 0
    assert "poke pending" not in capsys.readouterr().out


def test_status_of_an_alive_but_parked_instance_reads_waiting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """machine_word_for_dir (the shared status-word owner) checks `parked` BEFORE
    `alive`, so an alive-but-parked instance (a persisted wait written while the
    worker is still live -- a teardown race) must read "waiting" like the watch
    screen / web pill, not the CLI alive-branch's hardcoded "running"."""
    monkeypatch.chdir(tmp_path)
    f = _write_machine(tmp_path)
    assert main(["machine", "run", str(f), "--exit-on-wait"]) == 0
    capsys.readouterr()  # drop run output
    root = resolved_state_dir(tmp_path) / "machines" / "waiter_delayed"
    write_worker_pid(root, os.getpid())  # a LIVE worker alongside the persisted wait

    assert main(["machine", "status", "waiter_delayed"]) == 0
    out = capsys.readouterr().out
    assert "status: waiting" in out  # parked wins over alive
    assert "status: running" not in out


def test_status_tolerates_a_corrupt_pending_wait(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A corrupt wait.json must not abort the whole readout: the shared dir word
    (machine_word_for_dir -> machine_is_parked) tolerates it as parked, so status
    mirrors that -- it notes the wait file is unreadable and still prints the
    state / spend, instead of ERROR + exit 1 (the CLI was off that shared rule)."""
    monkeypatch.chdir(tmp_path)
    f = _write_machine(tmp_path)
    assert main(["machine", "run", str(f), "--exit-on-wait"]) == 0
    capsys.readouterr()  # drop run output
    root = resolved_state_dir(tmp_path) / "machines" / "waiter_delayed"
    clear_worker_pid(root)
    MachineJournal(root).wait_path.write_text("{ not valid json", encoding="utf-8")

    code = main(["machine", "status", "waiter_delayed"])
    assert code == 0  # not aborted
    out = capsys.readouterr().out
    assert "pending wait: unreadable" in out  # truthful, not silently swallowed
    assert "state:" in out  # the rest of the readout still prints


CRASHER = """
machine = "crasher"
version = 1
initial = "one"

[budget]
max_transitions = 100

[states.one]
kind = "tool"
command = ["true"]
timeout_secs = 60
on = { ok = "two", nonzero = "bad", timeout = "bad" }

[states.two]
kind = "tool"
command = ["true"]
timeout_secs = 60
on = { ok = "done", nonzero = "bad", timeout = "bad" }

[states.done]
kind = "terminal"
status = "ok"
reason = "done"

[states.bad]
kind = "terminal"
status = "failed"
reason = "bad"
"""


def test_status_reports_stopped_for_a_crashed_instance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # A worker that died mid-state (a step recorded, no MachineEnd, no armed
    # wait, dead pid) is "stopped" -- the word machine watch, the TUI header, and
    # the web pill all show via machine_status_word, the one owner of the
    # distinction -- not the engine's raw "incomplete", which only that owner
    # translates.
    from agent6.machine.journal import StepEvent, ToolFact

    monkeypatch.chdir(tmp_path)
    root = resolved_state_dir(tmp_path) / "machines" / "crasher"
    root.mkdir(parents=True)
    (root / "machine.asm.toml").write_text(CRASHER, encoding="utf-8")
    journal = MachineJournal(root)
    journal.ensure_dirs()
    journal.begin(machine="crasher", version=1)
    journal.append(
        StepEvent(
            ts="t",
            seq=0,
            state="one",
            label="ok",
            goto="two",
            fact=ToolFact(exit_code=0, stdout="", timed_out=False),
        )
    )
    # No worker.pid file -> not alive; no pending wait -> not parked.
    assert main(["machine", "status", "crasher"]) == 0
    out = capsys.readouterr().out
    assert "status: stopped" in out


def test_status_missing_instance_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    code = main(["machine", "status", "nope"])
    assert code == 2
    assert "no machine instance" in capsys.readouterr().err


def test_uncommitted_refusal_logs_a_git_error_instead_of_silently_failing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # The dirty-file gate fails OPEN on a GitError (it is review-discipline, not
    # security) but must never do so SILENTLY: a broken-git env stays visible.
    import agent6.app.machine.run as machine_run
    from agent6.git_ops import GitError

    _git_init(tmp_path)
    f = tmp_path / "m.asm.toml"
    f.write_text('machine="m"\nversion=1\ninitial="s"\n[states.s]\nkind="terminal"\n')

    def _boom(*_a: object, **_k: object) -> bool:
        raise GitError("git index is corrupt")

    monkeypatch.setattr(machine_run, "paths_dirty", _boom)
    assert machine_run.uncommitted_refusal(f, tmp_path) is None  # fail-open preserved
    err = capsys.readouterr().err
    assert "could not check" in err and "git index is corrupt" in err


def test_status_asm_file_path_hints_the_instance_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # `machine run` takes a FILE (waiter.asm.toml); status/replay/poke/watch take
    # the instance ID (waiter_delayed). Passing the file where the id belongs must
    # suggest the id, not dead-end.
    monkeypatch.chdir(tmp_path)
    f = _write_machine(tmp_path)  # machine = "waiter_delayed"
    assert main(["machine", "run", str(f), "--exit-on-wait"]) == 0
    capsys.readouterr()
    clear_worker_pid(resolved_state_dir(tmp_path) / "machines" / "waiter_delayed")
    code = main(["machine", "status", "waiter.asm.toml"])
    assert code == 2
    err = capsys.readouterr().err
    assert "no machine instance" in err
    assert "waiter_delayed" in err  # the did-you-mean names the real instance id


def test_status_on_an_invalid_machine_file_names_it_as_a_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """An unparsable .asm.toml where an id belongs is still a file, and the
    hint says so; the load failure fell through to the near-miss id search."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "bad.asm.toml").write_text('machine = "bad"\n', encoding="utf-8")
    assert main(["machine", "status", "bad.asm.toml"]) == 2
    err = capsys.readouterr().err
    assert "no machine instance" in err
    assert "machine file" in err, err
    assert "agent6 machine check bad.asm.toml" in err, err  # `machine run` would refuse it too


# A no-I/O machine that reaches a terminal immediately (branch -> terminal), so
# `agent6 attach` on it takes the finished path (overview + end) without blocking
# in the follow loop and without needing a model or the jail.
TINY = """
machine = "tiny"
version = 1
initial = "route"

[budget]
max_transitions = 10

[vars.code]
n = { type = "int", default = 0 }

[states.route]
kind = "branch"
when = [
  { if = "n == 0", goto = "done" },
  { else = true, goto = "done" },
]

[states.done]
kind = "terminal"
status = "ok"
reason = "routed"
"""


def test_watch_finished_instance_shows_overview_and_end(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    f = tmp_path / "tiny.asm.toml"
    f.write_text(TINY, encoding="utf-8")
    assert main(["machine", "run", str(f)]) == 0
    capsys.readouterr()  # drop run output
    # A finished instance has a journaled MachineEnd, so the unified `agent6 attach`
    # (which routes a machine name to the machine follower) prints the overview +
    # the final state and returns instead of entering the (blocking) follow loop.
    code = main(["attach", "tiny"])
    assert code == 0
    out = capsys.readouterr().out
    assert "machine: tiny" in out
    assert "▸ done" in out  # current state marked (machine_state_mark)
    assert "· route" in out  # a visited state marked
    assert "OK: ended in 'done'" in out


def _stalled_instance(tmp_path: Path, *, parked: bool) -> None:
    """An instance whose journal has begun but not ended, with a dead worker
    pid -- plus an armed pending wait when *parked*."""
    from agent6.machine.journal import MachineJournal, PendingWait

    inst = resolved_state_dir(tmp_path) / "machines" / "tiny"
    inst.mkdir(parents=True)
    (tmp_path / "tiny.asm.toml").write_text(TINY, encoding="utf-8")
    (inst / "machine.asm.toml").write_text(TINY, encoding="utf-8")
    (inst / "journal.jsonl").write_text(
        '{"type":"machine.begin","ts":"2026-07-12T00:00:00+00:00","machine":"tiny","version":1}\n',
        encoding="utf-8",
    )
    (inst / "worker.pid").write_text("999999", encoding="utf-8")
    if parked:
        MachineJournal(inst).write_pending_wait(PendingWait(state="route", wake_epoch=None))


def _watch_in_thread(timeout_s: float) -> tuple[list[int], bool]:
    import threading

    result: list[int] = []
    t = threading.Thread(target=lambda: result.append(main(["attach", "tiny"])), daemon=True)
    t.start()
    t.join(timeout=timeout_s)
    return result, t.is_alive()


def test_watch_exits_on_a_parked_machine(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A parked (--exit-on-wait) machine has an armed wait and no worker: the
    docstring promises watch "exits when the machine ends/waits", but the loop
    only ever ended on MachineEnd and spun silently forever."""
    monkeypatch.chdir(tmp_path)
    _stalled_instance(tmp_path, parked=True)
    result, still_running = _watch_in_thread(5.0)
    assert not still_running, "watch loop failed to terminate on a parked machine"
    assert result == [0]
    out = capsys.readouterr().out
    assert "WAITING" in out and "machine poke tiny" in out


def test_watch_follows_a_live_machine_in_a_wait(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A live worker blocked in a foreground wait still writes: attach follows
    it and exits only once the worker is gone. It exited at once on the word
    "waiting", live or not."""
    import threading

    monkeypatch.chdir(tmp_path)
    _stalled_instance(tmp_path, parked=True)
    inst = resolved_state_dir(tmp_path) / "machines" / "tiny"
    write_worker_pid(inst, os.getpid())
    result: list[int] = []
    t = threading.Thread(target=lambda: result.append(main(["attach", "tiny"])), daemon=True)
    t.start()
    t.join(timeout=2.0)
    assert t.is_alive(), f"attach left a live machine: exit {result}"
    clear_worker_pid(inst)  # the worker goes: the parked instance has nothing to follow
    t.join(timeout=5.0)
    assert result == [0]
    assert "WAITING" in capsys.readouterr().out


def test_watch_exits_on_a_crashed_machine(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A crashed worker (stale worker.pid, no MachineEnd, no armed wait) left
    watch presenting "watching..." forever over a dead machine."""
    monkeypatch.chdir(tmp_path)
    _stalled_instance(tmp_path, parked=False)
    result, still_running = _watch_in_thread(5.0)
    assert not still_running, "watch loop failed to terminate on a crashed machine"
    assert result == [1]
    assert "STOPPED" in capsys.readouterr().err


def test_replay_pluralizes_the_transition_count(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """`machine replay` counts "1 transition" (singular), matching `machine run`."""
    monkeypatch.chdir(tmp_path)
    f = tmp_path / "tiny.asm.toml"
    f.write_text(TINY, encoding="utf-8")
    assert main(["machine", "run", str(f)]) == 0
    capsys.readouterr()
    assert main(["machine", "replay", "tiny"]) == 0
    out = capsys.readouterr().out
    assert "after 1 transition (" in out and "1 transitions" not in out


def test_run_refuses_uncommitted_machine(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # docs §7.1/§9: `machine run` only accepts a committed machine. An untracked
    # .asm.toml is refused before any execution.
    monkeypatch.chdir(tmp_path)
    _git_init(tmp_path)
    f = tmp_path / "tiny.asm.toml"
    f.write_text(TINY, encoding="utf-8")
    code = main(["machine", "run", str(f)])
    assert code == 2  # a refusal, like every REFUSING
    err = capsys.readouterr().err
    assert "uncommitted" in err and "committed machine" in err
    # Refused before touching the state dir: no instance journal was created.
    root = resolved_state_dir(tmp_path) / "machines" / "tiny"
    assert not (root / "journal.jsonl").exists()


def test_uncommitted_refusal_tracks_git_state(tmp_path: Path) -> None:
    from agent6.app.machine.run import uncommitted_refusal

    # Outside a git repo the gate never fires (nothing to commit against).
    f = tmp_path / "tiny.asm.toml"
    f.write_text(TINY, encoding="utf-8")
    assert uncommitted_refusal(f, tmp_path) is None
    _git_init(tmp_path)
    assert uncommitted_refusal(f, tmp_path) is not None  # untracked
    subprocess.run(["git", "-C", str(tmp_path), "add", "tiny.asm.toml"], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "commit", "-q", "-m", "add"], check=True)
    assert uncommitted_refusal(f, tmp_path) is None  # committed clean
    f.write_text(TINY + "\n", encoding="utf-8")
    assert uncommitted_refusal(f, tmp_path) is not None  # modified again


def test_uncommitted_refusal_covers_the_scripts_bundle(tmp_path: Path) -> None:
    """One committed-bundle rule: a tool executes `scripts/` as trusted logic
    exactly like the .asm.toml, so a dirty bundle REFUSES (not a warning a
    scrolling launch buries); `machine test` stays the ungated iteration
    loop."""
    from agent6.app.machine.run import uncommitted_refusal

    f = tmp_path / "tiny.asm.toml"
    f.write_text(TINY, encoding="utf-8")
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    (scripts / "do.py").write_text("print('hi')\n", encoding="utf-8")
    # No git repo: no gate (nothing to commit against).
    assert uncommitted_refusal(f, tmp_path) is None
    _git_init(tmp_path)
    assert uncommitted_refusal(f, tmp_path) is not None  # untracked bundle
    subprocess.run(["git", "-C", str(tmp_path), "add", "-A"], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "commit", "-q", "-m", "add"], check=True)
    assert uncommitted_refusal(f, tmp_path) is None  # committed clean
    (scripts / "do.py").write_text("print('changed')\n", encoding="utf-8")
    refusal = uncommitted_refusal(f, tmp_path)
    assert refusal is not None and "scripts" in refusal  # modified script refuses


def test_first_run_records_the_bundle_and_drift_refuses_continuation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A live instance runs the bundle it recorded: the first run persists the
    .asm.toml + scripts tree under the instance root, an edited script refuses
    continuation by name (never executes under the old instance identity), and
    a restored bundle continues cleanly."""
    monkeypatch.chdir(tmp_path)
    _git_init(tmp_path)
    f = _write_machine(tmp_path)  # waiter: parks WAITING under --exit-on-wait
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    (scripts / "do.py").write_text("print('hi')\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(tmp_path), "add", "-A"], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "commit", "-q", "-m", "add"], check=True)

    assert main(["machine", "run", str(f), "--exit-on-wait"]) == 0
    assert "WAITING" in capsys.readouterr().out
    root = resolved_state_dir(tmp_path) / "machines" / "waiter_delayed"
    recorded = root / "scripts" / "do.py"
    assert recorded.read_text(encoding="utf-8") == "print('hi')\n"

    (scripts / "do.py").write_text("print('changed')\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(tmp_path), "commit", "-qam", "edit"], check=True)
    assert main(["machine", "run", str(f), "--exit-on-wait"]) == 2  # a refusal
    err = capsys.readouterr().err
    assert "scripts/do.py" in err and "archive the instance" in err

    (scripts / "do.py").write_text("print('hi')\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(tmp_path), "commit", "-qam", "restore"], check=True)
    assert main(["machine", "run", str(f), "--exit-on-wait"]) == 0
    assert "WAITING" in capsys.readouterr().out


def test_continuation_refuses_an_edited_machine_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """An edited .asm.toml with the same name/version drifts from the recorded
    source and refuses continuation -- identity strings alone let an
    incompatible edit land on the old journal."""
    monkeypatch.chdir(tmp_path)
    _git_init(tmp_path)
    f = _write_machine(tmp_path)
    subprocess.run(["git", "-C", str(tmp_path), "add", "-A"], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "commit", "-q", "-m", "add"], check=True)
    assert main(["machine", "run", str(f), "--exit-on-wait"]) == 0
    capsys.readouterr()
    f.write_text(WAITER_DELAYED.replace('reason = "ticked"', 'reason = "changed"'), "utf-8")
    subprocess.run(["git", "-C", str(tmp_path), "commit", "-qam", "edit"], check=True)
    assert main(["machine", "run", str(f), "--exit-on-wait"]) == 2  # a refusal
    err = capsys.readouterr().err
    assert "differs from the recorded" in err and "archive the instance" in err


def test_run_refuses_rerun_of_ended_instance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # An ended instance can only be replayed, never advanced. A rerun must refuse
    # BEFORE stamping worker.pid, so a dead machine never reads "running".
    from agent6.machine import drive, load_machine
    from agent6.sessions.ipc import read_worker_pid, write_worker_pid

    monkeypatch.chdir(tmp_path)
    f = tmp_path / "tiny.asm.toml"
    f.write_text(TINY, encoding="utf-8")
    assert main(["machine", "run", str(f)]) == 0  # runs to a terminal
    capsys.readouterr()
    root = resolved_state_dir(tmp_path) / "machines" / "tiny"
    # Stand in for the previous worker having exited: a pid that is never alive.
    sentinel = 10**9
    write_worker_pid(root, sentinel)
    code = main(["machine", "run", str(f)])
    assert code == 2  # a refusal
    err = capsys.readouterr().err
    assert "already ended" in err
    assert str(root) in err  # the archive remedy names the instance dir
    # worker.pid was NOT re-stamped with the (live) rerun process pid.
    assert read_worker_pid(root) == sentinel
    # The journal still reads terminal, unchanged.
    result = drive(load_machine(root / "machine.asm.toml"), MachineJournal(root), None, live=False)
    assert result.status == "ok"


def test_poke_drops_signal_for_waiting_machine(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    f = _write_machine(tmp_path)
    assert main(["machine", "run", str(f), "--exit-on-wait"]) == 0
    capsys.readouterr()
    code = main(["machine", "poke", "waiter_delayed"])
    assert code == 0
    assert "poked" in capsys.readouterr().out
    # The signal is now pending for the next take_signal().
    root = resolved_state_dir(tmp_path) / "machines" / "waiter_delayed"
    assert MachineJournal(root).take_signal() == (True, None)


def test_poke_carries_data_payload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    f = _write_machine(tmp_path)
    assert main(["machine", "run", str(f), "--exit-on-wait"]) == 0
    capsys.readouterr()
    assert main(["machine", "poke", "waiter_delayed", "--data", '{"cmd": "go"}']) == 0
    root = resolved_state_dir(tmp_path) / "machines" / "waiter_delayed"
    j = MachineJournal(root)
    assert j.take_signal() == (True, {"cmd": "go"})
    j.ack_signal()
    # --message wraps a plain string.
    assert main(["machine", "poke", "waiter_delayed", "--message", "hello"]) == 0
    assert j.take_signal() == (True, "hello")


def test_poke_rejects_invalid_json_data(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    f = _write_machine(tmp_path)
    assert main(["machine", "run", str(f), "--exit-on-wait"]) == 0
    capsys.readouterr()
    assert main(["machine", "poke", "waiter_delayed", "--data", "{not json}"]) == 2
    assert "not valid JSON" in capsys.readouterr().err


def test_poke_refuses_ended_machine(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # A terminal machine consumes no signals; poking it would sit unread, so the
    # CLI refuses instead of claiming "it will wake on its next signal check".
    monkeypatch.chdir(tmp_path)
    f = tmp_path / "tiny.asm.toml"
    f.write_text(TINY, encoding="utf-8")
    assert main(["machine", "run", str(f)]) == 0
    capsys.readouterr()
    code = main(["machine", "poke", "tiny"])
    assert code == 1
    assert "already ended" in capsys.readouterr().err
    root = resolved_state_dir(tmp_path) / "machines" / "tiny"
    assert not (root / "signal").exists()  # no signal was dropped


def test_poke_missing_instance_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    code = main(["machine", "poke", "nope"])
    assert code == 2
    assert "no machine instance" in capsys.readouterr().err


AGENT_MACHINE_HARD = """
machine = "hard-usd"
version = 1
initial = "judge"

[budget]
max_usd = 1.0
max_transitions = 10

[schemas.r]
ok = "bool"

[vars.agent]
out = { type = "r", default = {} }

[states.judge]
kind = "agent"
prompt = "judge"
output_schema = "r"
capture = { finish_json = "out" }
timeout_secs = 60
on = { ok = "done", failed = "done", budget_exhausted = "done", timeout = "done" }

[states.done]
kind = "terminal"
status = "ok"
reason = "done"
"""


AGENT_RUN_MACHINE = AGENT_MACHINE_HARD.replace(
    'machine = "hard-usd"', 'machine = "run-warn"'
).replace('kind = "agent"', 'kind = "agent"\nmode = "run"', 1)


WAIT_THEN_RUN = (
    WAITER_DELAYED.replace('machine = "waiter_delayed"', 'machine = "wait-then-run"')
    .replace('on = { tick = "done", signal = "woken" }', 'on = { tick = "work", signal = "work" }')
    .replace(
        "[states.done]",
        """[schemas.r]
ok = "bool"

[vars.agent]
out = { type = "r", default = {} }

[states.work]
kind = "agent"
mode = "run"
prompt = "fix it"
output_schema = "r"
capture = { finish_json = "out" }
timeout_secs = 60
on = { ok = "done", failed = "done", budget_exhausted = "done", timeout = "done" }

[states.done]""",
    )
    .replace('[states.woken]\nkind = "terminal"\nstatus = "ok"\nreason = "signalled"\n', "")
)


def test_run_says_where_a_machines_work_landed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A machine with run states commits to `agent6/machine-<id>` and never
    touches the checkout, so "tests passing" was reported over a tree whose
    tests still fail, with nothing naming where the work went."""
    cfg_home = tmp_path.parent / (tmp_path.name + "-cfg")  # outside the workspace
    monkeypatch.setenv("AGENT6_CONFIG_HOME", str(cfg_home))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    cfg_home.mkdir(parents=True)
    (cfg_home / "config.toml").write_text(
        "\n".join(
            (
                "[agent6]",
                "config_version = 1",
                "[providers.anthropic]",
                'api_format = "anthropic"',
                'api_key_env = "ANTHROPIC_API_KEY"',
                "[models.worker]",
                'provider = "anthropic"',
                'model = "x"',
            )
        ),
        encoding="utf-8",
    )
    _git_init(tmp_path)
    monkeypatch.chdir(tmp_path)
    (tmp_path / "f.txt").write_text("x", encoding="utf-8")
    subprocess.run(["git", "-C", str(tmp_path), "add", "f.txt"], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "commit", "-qm", "init"], check=True)
    subprocess.run(
        ["git", "-C", str(tmp_path), "branch", "agent6/machine-wait-then-run"], check=True
    )
    f = tmp_path / "wtr.asm.toml"
    f.write_text(WAIT_THEN_RUN, encoding="utf-8")
    subprocess.run(["git", "-C", str(tmp_path), "add", "wtr.asm.toml"], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "commit", "-qm", "machine"], check=True)

    assert main(["machine", "run", str(f), "--exit-on-wait", "--auto-approve"]) == 0

    out = capsys.readouterr().out
    assert "WAITING" in out
    assert "changes are on agent6/machine-wait-then-run" in out
    assert "git merge agent6/machine-wait-then-run" in out


def test_run_warns_on_mode_run_states_under_ask_policy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # An unattended machine auto-denies run_command under 'ask'; a mode='run'
    # state burns its budget against denials, so machine run says so up front
    # and names both remedies. No provider is configured here, so the run then
    # refuses at require_runnable, which keeps this test spend-free; the note
    # must already have printed.
    monkeypatch.setenv("AGENT6_CONFIG_HOME", str(tmp_path / "cfg"))
    monkeypatch.chdir(tmp_path)
    f = tmp_path / "runwarn.asm.toml"
    f.write_text(AGENT_RUN_MACHINE, encoding="utf-8")
    code = main(["machine", "run", str(f)])
    err = capsys.readouterr().err
    assert code == 2  # no worker configured: refused right after the note
    assert "auto-denies" in err and "--auto-approve" in err


def test_run_auto_approve_suppresses_the_warning_and_sets_the_env_grant(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import os

    monkeypatch.setenv("AGENT6_CONFIG_HOME", str(tmp_path / "cfg"))
    monkeypatch.delenv("AGENT6_AUTO_APPROVE", raising=False)  # snapshot: restored at teardown
    monkeypatch.chdir(tmp_path)
    f = tmp_path / "runwarn.asm.toml"
    f.write_text(AGENT_RUN_MACHINE, encoding="utf-8")
    code = main(["machine", "run", str(f), "--auto-approve"])
    err = capsys.readouterr().err
    assert code == 2  # still refused on the missing worker; that is fine
    assert "auto-denies" not in err  # the grant removed the dead-end
    # The grant reaches each agent subprocess the way the sandbox setter does.
    assert os.environ.get("AGENT6_AUTO_APPROVE") == "1"


def test_a_fresh_instance_over_a_stale_chain_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A machine chain ref outlives an archived instance dir; a fresh instance
    silently continued the dead instance's tree (a live leg saw its fix and
    reported tests passed over a broken repo). The run refuses instead, naming
    the branch and both remedies."""
    import subprocess

    monkeypatch.setenv("AGENT6_CONFIG_HOME", str(tmp_path / "cfg"))
    monkeypatch.setenv("AGENT6_STATE_HOME", str(tmp_path / ".state"))
    monkeypatch.setenv("AGENT6_AUTO_APPROVE", "1")
    repo = tmp_path / "repo"
    repo.mkdir()
    monkeypatch.chdir(repo)
    f = repo / "runwarn.asm.toml"
    f.write_text(AGENT_RUN_MACHINE, encoding="utf-8")
    for argv in (
        ["git", "init", "-q"],
        ["git", "add", "runwarn.asm.toml"],
        ["git", "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-q", "-m", "seed"],
        ["git", "update-ref", "refs/agent6/machine-run-warn/head", "HEAD"],
    ):
        subprocess.run(argv, cwd=repo, check=True)
    cfg = tmp_path / "agent6.toml"
    cfg.write_text(
        """[providers.p]
api_format = "openai"
base_url = "http://127.0.0.1:9"
api_key_env = "AGENT6_TEST_KEY"

[models.worker]
provider = "p"
model = "m"
""",
        encoding="utf-8",
    )
    monkeypatch.setenv("AGENT6_TEST_KEY", "x")
    code = main(["--config", str(cfg), "machine", "run", str(f)])
    err = capsys.readouterr().err
    assert code == 2
    assert "chain branch 'agent6/machine-run-warn' exists" in err
    assert "git branch -D agent6/machine-run-warn" in err


def test_run_no_commands_withholds_them_from_the_machine(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`--no-commands` reached `machine run`'s parser and stopped there: the
    dispatch never read it, so an operator running an unfamiliar machine with
    it got the machine's full command surface."""
    import os

    from agent6.app.machine_agent import (
        _apply_operator_env_grants,  # pyright: ignore[reportPrivateUsage]
    )
    from agent6.config import Config

    monkeypatch.setenv("AGENT6_CONFIG_HOME", str(tmp_path / "cfg"))
    # setenv, not delenv: monkeypatch records an absent var as nothing to
    # restore, so `main` setting it would leak into the next test.
    monkeypatch.setenv("AGENT6_NO_COMMANDS", "")
    monkeypatch.chdir(tmp_path)
    f = tmp_path / "nocmd.asm.toml"
    f.write_text(AGENT_RUN_MACHINE, encoding="utf-8")
    main(["machine", "run", str(f), "--no-commands"])
    assert os.environ.get("AGENT6_NO_COMMANDS") == "1"
    yes = Config.model_validate({"sandbox": {"run_commands": "yes"}})
    assert _apply_operator_env_grants(yes).sandbox.run_commands == "no"


def test_apply_operator_env_grants_upgrades_ask_never_no(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from agent6.app.machine_agent import (
        _apply_operator_env_grants,  # pyright: ignore[reportPrivateUsage]
    )
    from agent6.config import Config

    monkeypatch.setenv("AGENT6_AUTO_APPROVE", "1")
    ask = Config.model_validate({"sandbox": {"run_commands": "ask"}})
    assert _apply_operator_env_grants(ask).sandbox.run_commands == "yes"
    no = Config.model_validate({"sandbox": {"run_commands": "no"}})
    assert _apply_operator_env_grants(no).sandbox.run_commands == "no"  # never resurrected
    monkeypatch.delenv("AGENT6_AUTO_APPROVE")
    assert _apply_operator_env_grants(ask).sandbox.run_commands == "ask"  # no grant, no change


TOOL_PROBE_MACHINE = """
machine = "probe-check"
version = 1
initial = "lint"

[budget]
max_usd = 1.0
max_transitions = 10

[states.lint]
kind = "tool"
command = ["definitely-not-a-binary-xyz", "check"]
timeout_secs = 5
on = { ok = "done", nonzero = "fail", timeout = "fail" }

[states.done]
kind = "terminal"
status = "ok"
reason = "clean"

[states.fail]
kind = "terminal"
status = "failed"
reason = "lint"
"""


def test_check_validates_the_config_overlay_run_will_merge(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """`machine run` merges the file's [config] table into the effective
    config; check and test skipped that merge, so an unknown config key
    returned OK from both and failed first at run."""
    monkeypatch.chdir(tmp_path)
    f = tmp_path / "bad.asm.toml"
    f.write_text(
        TINY + '\n[config.workflow]\nnonsense_key = "x"\n',
        encoding="utf-8",
    )
    assert main(["machine", "check", str(f)]) == 1
    err = capsys.readouterr().err
    assert "FAIL" in err and "nonsense_key" in err
    assert main(["machine", "test", str(f)]) == 1


def test_check_warns_on_binaries_unreachable_in_the_jail(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # Offline validation mocks subprocess, so a machine whose tool calls a
    # binary absent from the jail PATH passed check/test and died on its first
    # real transition. The probe covers tool-state command[0] AND literal
    # subprocess argv inside bundle scripts; reachable binaries stay quiet and
    # the warnings are advisory (check still exits 0).
    monkeypatch.chdir(tmp_path)
    f = tmp_path / "probe.asm.toml"
    f.write_text(TOOL_PROBE_MACHINE, encoding="utf-8")
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    (scripts / "helper.py").write_text(
        '"""Bundle helper."""\n\nimport subprocess\n\n\n'
        "def main() -> int:\n"
        '    subprocess.run(["also-missing-tool-abc", "x"], check=False)\n'
        '    subprocess.run(["python3", "-c", "pass"], check=False)\n'
        "    return 0\n",
        encoding="utf-8",
    )
    code = main(["machine", "check", str(f)])
    out = capsys.readouterr()
    assert code == 0  # advisory: the operator may install the tool later
    assert "OK:" in out.out
    assert "`definitely-not-a-binary-xyz` ([states.lint] command)" in out.err
    assert "`also-missing-tool-abc` (scripts/helper.py)" in out.err
    assert "python3" not in out.err  # reachable binaries stay quiet


def test_machine_stop_marks_a_running_worker_and_refuses_a_dead_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """`machine stop` writes the durable marker only for a live worker; a
    parked/dead instance gets a refusal, never a marker that would ambush the
    next `machine run` at its first boundary."""

    from agent6.viewmodel import machine_state as machine_state_mod

    monkeypatch.chdir(tmp_path)
    f = tmp_path / "tiny.asm.toml"
    f.write_text(TINY, encoding="utf-8")
    assert main(["machine", "run", str(f)]) == 0
    capsys.readouterr()
    root = resolved_state_dir(tmp_path) / "machines" / "tiny"
    assert main(["machine", "stop", "tiny"]) == 1  # ended: nothing to stop
    assert "already ended" in capsys.readouterr().err
    assert not (root / "stop").exists()

    w = _write_machine(tmp_path)  # waiter: parks WAITING, journal not ended
    assert main(["machine", "run", str(w), "--exit-on-wait"]) == 0
    capsys.readouterr()
    wroot = resolved_state_dir(tmp_path) / "machines" / "waiter_delayed"
    assert main(["machine", "stop", "waiter_delayed"]) == 1  # parked, worker dead
    assert "not running" in capsys.readouterr().err
    assert not (wroot / "stop").exists()

    def _alive(_root: Path) -> bool:
        return True

    monkeypatch.setattr(machine_state_mod, "worker_is_alive", _alive)  # the verb gate's owner
    assert main(["machine", "stop", "waiter_delayed"]) == 0
    assert "stop requested" in capsys.readouterr().out
    assert (wroot / "stop").is_file()


def test_run_start_clears_a_stale_stop_marker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A leftover stop marker must not park the next invocation at its first
    boundary: starting the machine is the answer to any stale request."""
    monkeypatch.chdir(tmp_path)
    f = tmp_path / "tiny.asm.toml"
    f.write_text(TINY, encoding="utf-8")
    root = resolved_state_dir(tmp_path) / "machines" / "tiny"
    root.mkdir(parents=True)
    (root / "stop").touch()
    assert main(["machine", "run", str(f)]) == 0
    out = capsys.readouterr().out
    assert "OK:" in out and "STOPPED" not in out
    assert not (root / "stop").exists()


def test_hub_spawn_away_mode_reaches_the_instance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A hub-spawned machine (AGENT6_DETACHED_AWAY=wait in the spawn env)
    records "wait" on its instance dir at run start, so every agent state's
    bridges park prompts for the front-end regardless of when its viewer
    registers."""
    from agent6.sessions.ipc import away_mode

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("AGENT6_DETACHED_AWAY", "wait")
    f = tmp_path / "tiny.asm.toml"
    f.write_text(TINY, encoding="utf-8")
    assert main(["machine", "run", str(f)]) == 0
    root = resolved_state_dir(tmp_path) / "machines" / "tiny"
    assert away_mode(root) == "wait"


def test_attach_degrades_a_corrupt_journal_like_status(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """`machine status` prints a clean ERROR for a corrupt journal; the watch
    (attach's machine arm) must degrade identically, never propagate the
    JournalError as a traceback."""
    monkeypatch.chdir(tmp_path)
    f = tmp_path / "tiny.asm.toml"
    f.write_text(TINY, encoding="utf-8")
    assert main(["machine", "run", str(f)]) == 0
    capsys.readouterr()
    root = resolved_state_dir(tmp_path) / "machines" / "tiny"
    (root / "journal.jsonl").write_text('{"type": "machine.begin"\n', encoding="utf-8")
    code = main(["attach", "tiny"])
    assert code == 1
    err = capsys.readouterr().err
    assert "ERROR:" in err


def test_run_refuses_an_explicit_protect_git_the_host_cannot_enforce(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """`machine run` had its own hand-assembled preflight and skipped the
    protect_git check `run`/`ask` make, so on hardened an explicit
    `protect_git = true` warned and ran instead of refusing (docs/security.md
    states the refusal without qualification). Both lifecycles now run
    `config_refusal`."""
    from agent6.app import _session as session_mod
    from agent6.app.machine import run as run_mod

    cfg_home = tmp_path / "cfg"
    cfg_home.mkdir()
    (cfg_home / "config.toml").write_text("[sandbox]\nprotect_git = true\n", encoding="utf-8")
    monkeypatch.setenv("AGENT6_CONFIG_HOME", str(cfg_home))
    monkeypatch.setenv("AGENT6_STATE_HOME", str(tmp_path / "state"))
    workspace = tmp_path / "repo"
    workspace.mkdir()
    monkeypatch.chdir(workspace)

    class _Env:
        detected_isolation = "hardened"

    monkeypatch.setattr(run_mod, "detect_env", _Env)

    def _hardened(_req: str, _env: object) -> str:
        return "hardened"

    monkeypatch.setattr(session_mod, "resolve_isolation", _hardened)
    f = workspace / "probe.asm.toml"
    f.write_text(TOOL_PROBE_MACHINE, encoding="utf-8")
    assert main(["machine", "run", str(f)]) == 2
    err = capsys.readouterr().err
    assert "REFUSING" in err and "protect_git" in err


def test_run_refuses_a_state_dir_inside_the_workspace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """`agent6 run` refuses a state base inside the workspace (jailed commands
    could read transcripts, and commits would stage them); `machine run` did
    not, so the same config ran there."""
    from agent6.app import _session as session_mod
    from agent6.app.machine import run as run_mod

    monkeypatch.setenv("AGENT6_CONFIG_HOME", str(tmp_path / "cfg"))
    monkeypatch.setenv("AGENT6_STATE_HOME", str(tmp_path / "inside-state"))
    monkeypatch.chdir(tmp_path)

    class _Env:
        detected_isolation = "strict"

    monkeypatch.setattr(run_mod, "detect_env", _Env)

    def _strict(_req: str, _env: object) -> str:
        return "strict"

    monkeypatch.setattr(session_mod, "resolve_isolation", _strict)
    f = tmp_path / "probe.asm.toml"
    f.write_text(TOOL_PROBE_MACHINE, encoding="utf-8")
    assert main(["machine", "run", str(f)]) == 2
    err = capsys.readouterr().err
    assert "REFUSING" in err and "private directory" in err


def test_list_joins_instances_with_their_files_and_names_the_rest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """`agent6 machine` (== `machine list`) is the CLI's machines page: each
    instance's status and current state (the web hub's words) joined with the
    authored file that declares it (the TUI page's spec column), then the
    files no instance has run, and an unparsable file kept by path alone."""
    monkeypatch.chdir(tmp_path)
    assert main(["machine"]) == 0
    assert "no machines yet" in capsys.readouterr().out
    (tmp_path / "tiny.asm.toml").write_text(TINY, encoding="utf-8")
    (tmp_path / "waiter.asm.toml").write_text(WAITER_DELAYED, encoding="utf-8")
    (tmp_path / "broken.asm.toml").write_text("not toml {{", encoding="utf-8")
    (tmp_path / "broken2.asm.toml").write_text("also not toml {{", encoding="utf-8")
    assert main(["machine", "run", str(tmp_path / "tiny.asm.toml")]) == 0
    capsys.readouterr()
    assert main(["machine"]) == 0
    out = capsys.readouterr().out
    lines = out.splitlines()
    assert lines[0].split() == ["updated", "status", "state", "machine", "spec", "file"]
    tiny = next(line for line in lines if "tiny.asm.toml" in line)
    assert tiny.split()[2:] == ["ok", "done", "tiny", "valid", "tiny.asm.toml"]  # [2:]: the when
    waiter = next(line for line in lines if "waiter.asm.toml" in line)
    assert waiter.split() == ["-", "-", "waiter_delayed", "valid", "waiter.asm.toml"]
    broken = next(line for line in lines if "broken.asm.toml" in line)
    assert broken.split() == ["-", "-", "-", "invalid", "broken.asm.toml"]
    # Two unparsable files (both named "-") keep two rows.
    assert any("broken2.asm.toml" in line for line in lines)


def test_status_and_list_name_a_parked_approval(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A live worker whose agent state holds an unanswered approval reads
    "waiting" on both surfaces, and status names the state to answer in."""
    monkeypatch.chdir(tmp_path)
    f = _write_machine(tmp_path)
    assert main(["machine", "run", str(f), "--exit-on-wait"]) == 0
    capsys.readouterr()
    root = resolved_state_dir(tmp_path) / "machines" / "waiter_delayed"
    MachineJournal(root).clear_pending_wait()
    write_worker_pid(root, os.getpid())
    leg = root / "states" / "0001-attempt"
    leg.mkdir(parents=True)
    (leg / "logs.jsonl").write_text(
        json.dumps({"type": "approval.prompt", "id": "a1", "prompt": "Allow run_command: x"})
        + "\n",
        encoding="utf-8",
    )
    assert main(["machine", "status", "waiter_delayed"]) == 0
    out = capsys.readouterr().out
    assert "status: waiting" in out
    assert "an approval open in 0001-attempt" in out
    assert main(["machine", "list"]) == 0
    row = next(line for line in capsys.readouterr().out.splitlines() if "waiter_delayed" in line)
    assert "waiting" in row and "0001-attempt" in row
