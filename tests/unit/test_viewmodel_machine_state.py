# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Tests for the pure machine-journal fold in agent6.viewmodel.machine_state."""

from __future__ import annotations

import json
import os
from pathlib import Path

from agent6.machine import load_machine
from agent6.machine.journal import (
    BranchFact,
    MachineEnd,
    MachineJournal,
    MachineNotify,
    PendingWait,
    StepEvent,
)
from agent6.viewmodel.machine_state import (
    _NOTIFY_KEEP,  # pyright: ignore[reportPrivateUsage]
    NotificationView,
    fold_machine,
    machine_is_parked,
    machine_state_as_dict,
    machine_status_word,
    machine_word_for_dir,
    newest_state_log,
    notification_key,
)

# A branch -> terminal machine: two states, no I/O, valid to load.
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


def _spec(tmp_path: Path):
    f = tmp_path / "tiny.asm.toml"
    f.write_text(TINY, encoding="utf-8")
    return load_machine(f)


def test_fold_empty_starts_at_initial(tmp_path: Path) -> None:
    # No journal yet: the machine is at its initial state, nothing visited.
    ms = fold_machine(_spec(tmp_path), [])
    assert (ms.machine, ms.version, ms.initial, ms.current) == ("tiny", 1, "route", "route")
    assert ms.transitions == ()
    assert ms.ended is None
    assert [s.name for s in ms.states] == ["route", "done"]  # spec order preserved
    route = next(s for s in ms.states if s.name == "route")
    assert route.is_current and not route.is_visited and route.kind == "branch"
    assert all(not s.is_visited for s in ms.states)


def test_fold_tracks_position_transitions_and_end(tmp_path: Path) -> None:
    events = [
        StepEvent(
            ts="t", seq=0, state="route", label="else", goto="done", fact=BranchFact(clause_index=1)
        ),
        MachineEnd(ts="t", status="ok", reason="routed", state="done", transitions=1),
    ]
    ms = fold_machine(_spec(tmp_path), events)
    by = {s.name: s for s in ms.states}
    # current = goto of the last transition; both endpoints are visited.
    assert ms.current == "done"
    assert by["done"].is_current and not by["route"].is_current
    assert by["route"].is_visited and by["done"].is_visited
    path = [(t.seq, t.state, t.label, t.goto) for t in ms.transitions]
    assert path == [(0, "route", "else", "done")]
    assert ms.ended is not None
    assert (ms.ended.status, ms.ended.reason, ms.ended.state, ms.ended.transitions) == (
        "ok",
        "routed",
        "done",
        1,
    )


def test_machine_status_word_distinguishes_waiting_from_running(tmp_path: Path) -> None:
    spec = _spec(tmp_path)
    ended = fold_machine(
        spec, [MachineEnd(ts="t", status="failed", reason="boom", state="done", transitions=1)]
    )
    # A terminal instance reports its end status regardless of liveness probes.
    assert machine_status_word(ended, parked=True, alive=True) == "failed"

    live = fold_machine(spec, [])  # not ended
    # Parked (an armed --exit-on-wait wait) reads waiting, even if a stale pid
    # probe were to lie alive; running only when live and not parked; a dead pid
    # that is neither parked nor ended is stopped.
    assert machine_status_word(live, parked=True, alive=False) == "waiting"
    assert machine_status_word(live, parked=True, alive=True) == "waiting"
    assert machine_status_word(live, parked=False, alive=True) == "running"
    assert machine_status_word(live, parked=False, alive=False) == "stopped"

    # A live worker blocked in a foreground `wait` state is "waiting", not
    # "running" (the default `machine run` persists no PendingWait).
    wf = tmp_path / "w.asm.toml"
    wf.write_text(
        'machine = "w"\nversion = 1\ninitial = "poll"\n[budget]\nmax_transitions = 10\n'
        '[states.poll]\nkind = "wait"\nevery_secs = "3600"\n'
        'on = { tick = "done", signal = "done" }\n'
        '[states.done]\nkind = "terminal"\nstatus = "ok"\nreason = "d"\n',
        encoding="utf-8",
    )
    waiting = fold_machine(load_machine(wf), [])
    assert machine_status_word(waiting, parked=False, alive=True) == "waiting"


def test_machine_word_for_dir_pairs_the_dir_probes(tmp_path: Path) -> None:
    """The dir-level owner: the pure word fed the armed-wait and worker-pid
    probes, so surfaces cannot pair them differently."""
    spec = _spec(tmp_path)
    live = fold_machine(spec, [])
    d = tmp_path / "inst"
    d.mkdir()
    assert machine_word_for_dir(live, d) == "stopped"  # no wait, no worker
    MachineJournal(d).write_pending_wait(PendingWait(state="w", wake_epoch=None))
    assert machine_word_for_dir(live, d) == "waiting"  # armed wait, no worker
    MachineJournal(d).clear_pending_wait()
    (d / "worker.pid").write_text(str(os.getpid()), encoding="utf-8")
    assert machine_word_for_dir(live, d) == "running"
    ended = fold_machine(
        spec, [MachineEnd(ts="t", status="ok", reason="routed", state="done", transitions=1)]
    )
    assert machine_word_for_dir(ended, d) == "ok"  # the end outranks the probes


def test_machine_state_as_dict_stamps_the_dir_word(tmp_path: Path) -> None:
    # With a dir in hand the wire form carries the dir-aware word; a genuinely
    # dir-less stream keeps the bare fold (no fabricated liveness claim).
    spec = _spec(tmp_path)
    live = fold_machine(spec, [])
    d = tmp_path / "inst"
    d.mkdir()
    MachineJournal(d).write_pending_wait(PendingWait(state="w", wake_epoch=None))
    assert machine_state_as_dict(live, d)["status"] == "waiting"
    assert "status" not in machine_state_as_dict(live)


def test_the_wire_form_carries_every_verb_refusal(tmp_path: Path) -> None:
    """A front-end paints all four verbs, so it asks once. Deriving them from
    the status word instead read a LIVE machine blocked on an approval (word:
    "waiting") as not running, and the web hid the box it was blocked on."""
    spec = _spec(tmp_path)
    live = fold_machine(spec, [])
    d = tmp_path / "inst"
    d.mkdir()
    (d / "worker.pid").write_text(str(os.getpid()), encoding="utf-8")

    refusals = machine_state_as_dict(live, d)["refusals"]

    assert refusals == {"stop": "", "poke": "", "steer": "", "answer": ""}
    MachineJournal(d).write_pending_wait(PendingWait(state="w", wake_epoch=None))
    parked = machine_state_as_dict(live, d)
    assert parked["status"] == "waiting"
    assert parked["refusals"]["answer"] == "", "a live machine still takes its answer"
    assert "reads no steer" in parked["refusals"]["steer"]


def test_a_corrupt_wait_file_counts_as_parked(tmp_path: Path) -> None:
    # Better to render "waiting" than to guess "stopped"/close the stream over
    # an unreadable wait record; the one rule every surface shares.
    spec = _spec(tmp_path)
    live = fold_machine(spec, [])
    d = tmp_path / "inst"
    d.mkdir()
    (d / "wait.json").write_text("{ not json", encoding="utf-8")
    assert machine_is_parked(d) is True
    assert machine_word_for_dir(live, d) == "waiting"


def test_newest_state_log_picks_highest_seq(tmp_path: Path) -> None:
    states = tmp_path / "states"
    for name in ("0000-greet", "0002-review", "0001-greet"):
        (states / name).mkdir(parents=True)
        (states / name / "logs.jsonl").write_text("{}\n", encoding="utf-8")
    # A dir without a log yet (the agent hasn't written) must be ignored.
    (states / "0009-pending").mkdir()
    assert newest_state_log(tmp_path) == states / "0002-review" / "logs.jsonl"
    assert newest_state_log(tmp_path / "absent") is None


def test_fold_collects_notifications(tmp_path: Path) -> None:
    events = [
        MachineNotify(ts="t1", state="route", message="starting", level="info"),
        StepEvent(
            ts="t", seq=0, state="route", label="else", goto="done", fact=BranchFact(clause_index=1)
        ),
        MachineNotify(ts="t2", state="done", message="all done", level="warn"),
        MachineEnd(ts="t", status="ok", reason="routed", state="done", transitions=1),
    ]
    ms = fold_machine(_spec(tmp_path), events)
    assert [(n.state, n.message, n.level) for n in ms.notifications] == [
        ("route", "starting", "info"),
        ("done", "all done", "warn"),
    ]


def test_notifications_are_a_capped_sliding_window(tmp_path: Path) -> None:
    # notifications is capped to the recent tail: a front-end must dedup by
    # notification_key, NOT by a count index (which would miss every one past
    # the cap once the window slides).
    events = [
        MachineNotify(ts=f"t{i}", state="route", message=f"n{i}", level="info")
        for i in range(_NOTIFY_KEEP + 5)
    ]
    ms = fold_machine(_spec(tmp_path), events)
    assert len(ms.notifications) == _NOTIFY_KEEP
    assert ms.notifications[-1].message == f"n{_NOTIFY_KEEP + 4}"  # newest kept
    assert ms.notifications[0].message == "n5"  # oldest dropped


def test_notification_key_is_stable_identity() -> None:
    n = NotificationView(ts="t1", state="poll", message="hi", level="warn")
    assert notification_key(n) == ("t1", "poll", "hi")


def test_machine_state_as_dict_is_json_serializable(tmp_path: Path) -> None:
    import json

    from agent6.viewmodel.machine_state import machine_state_as_dict

    events = [
        StepEvent(
            ts="t", seq=0, state="route", label="else", goto="done", fact=BranchFact(clause_index=1)
        ),
        MachineEnd(ts="t", status="ok", reason="routed", state="done", transitions=1),
    ]
    d = machine_state_as_dict(fold_machine(_spec(tmp_path), events))
    assert d["machine"] == "tiny" and d["current"] == "done"
    assert d["states"][0]["name"] == "route"  # tuple -> list, dataclass -> dict
    assert d["ended"]["status"] == "ok"
    json.dumps(d)  # the wire form must serialize


def test_machine_verb_refusal_is_one_reading_per_state_and_verb(tmp_path: Path) -> None:
    """The one gate every surface's stop/poke/steer/answer runs: an unknown
    machine is named as unknown; an ended one takes nothing; a stopped one
    takes only a poke (it wakes it); a live one in a wait state takes a stop
    and an answer but reads no steer; a running one takes everything."""
    from agent6.viewmodel.machine_state import machine_verb_refusal

    verbs = ("stop", "poke", "steer", "answer")
    missing = tmp_path / "ghost"
    assert all(machine_verb_refusal(missing, "ghost", v) == "no machine 'ghost'" for v in verbs)

    d = tmp_path / "inst"
    d.mkdir()
    (d / "machine.asm.toml").write_text(TINY, encoding="utf-8")
    journal = MachineJournal(d)
    journal.begin(machine="tiny", version=1)
    # Stopped: no worker, no wait. Only a poke goes through.
    assert machine_verb_refusal(d, "tiny", "poke") == ""
    for verb in ("stop", "steer", "answer"):
        assert "is not running" in machine_verb_refusal(d, "tiny", verb), verb
    # Live in an armed wait: a stop and an answer reach it; a steer does not.
    (d / "worker.pid").write_text(str(os.getpid()), encoding="utf-8")
    journal.write_pending_wait(PendingWait(state="route", wake_epoch=None))
    assert machine_verb_refusal(d, "tiny", "stop") == ""
    assert machine_verb_refusal(d, "tiny", "answer") == ""
    assert "reads no steer" in machine_verb_refusal(d, "tiny", "steer")
    journal.clear_pending_wait()
    # Running: everything reaches it.
    assert all(machine_verb_refusal(d, "tiny", v) == "" for v in verbs)
    # Ended: nothing does, and the end is named.
    journal.append(MachineEnd(ts="t", status="ok", reason="routed", state="done", transitions=1))
    for verb in verbs:
        msg = machine_verb_refusal(d, "tiny", verb)
        assert "already ended in 'done' (ok: routed)" in msg, (verb, msg)


def test_an_open_prompt_in_the_newest_state_blocks_the_machine(tmp_path: Path) -> None:
    """The newest state log's unanswered approval names the state the machine
    waits on; an answered one does not, and a live blocked worker is "waiting"."""
    from agent6.viewmodel.machine_state import machine_operator_blocked, machine_status_word

    states = tmp_path / "states"
    (states / "0001-attempt").mkdir(parents=True)
    log = states / "0001-attempt" / "logs.jsonl"
    prompt = {"type": "approval.prompt", "id": "a1", "prompt": "Allow run_command: pytest"}
    log.write_text(json.dumps(prompt) + "\n", encoding="utf-8")
    assert machine_operator_blocked(tmp_path) == "0001-attempt"
    answer = {"type": "approval.answer", "id": "a1", "approved": True}
    log.write_text(json.dumps(prompt) + "\n" + json.dumps(answer) + "\n", encoding="utf-8")
    assert machine_operator_blocked(tmp_path) == ""
    ms = fold_machine(_spec(tmp_path), [])
    assert machine_status_word(ms, parked=False, alive=True, blocked=True) == "waiting"
    assert machine_status_word(ms, parked=False, alive=True) == "running"
