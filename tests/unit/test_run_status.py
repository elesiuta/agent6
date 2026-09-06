# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""`agent6 sessions show`: one-shot liveness + progress of a run from its run dir."""

from __future__ import annotations

import datetime as dt
import json
import os
import subprocess
from pathlib import Path
from typing import Any

import pytest

from agent6.sessions.ipc import worker_is_alive, write_worker_pid
from agent6.ui.cli._common import _runs_dir  # pyright: ignore[reportPrivateUsage]
from agent6.ui.cli.sessions_show import _cmd_status  # pyright: ignore[reportPrivateUsage]


def _ts(off_s: float) -> str:
    return (dt.datetime.now(dt.UTC) - dt.timedelta(seconds=off_s)).isoformat()


def _make_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, events: list[dict[str, object]]
) -> Path:
    monkeypatch.setenv("AGENT6_STATE_HOME", str(tmp_path / "state"))
    repo = tmp_path / "repo"
    repo.mkdir()
    monkeypatch.chdir(repo)
    runs = _runs_dir(repo)
    runs.mkdir(parents=True, exist_ok=True)
    d = runs / "winsome-dawn-YWH5ZS"
    d.mkdir()
    (d / "manifest.json").write_text(
        json.dumps({"mode": "run", "models": {"driver": {"model": "claude-opus-4-8"}}})
    )
    (d / "logs.jsonl").write_text("\n".join(json.dumps(e) for e in events) + "\n")
    return d


def test_status_running_with_live_pid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    d = _make_run(
        tmp_path,
        monkeypatch,
        [
            {"ts": _ts(40), "type": "session.start", "mode": "run"},
            {"ts": _ts(3), "type": "loop.tool.call", "iteration": 3},
        ],
    )
    write_worker_pid(d, os.getpid())  # this test process is genuinely alive
    assert worker_is_alive(d)
    assert _cmd_status("winsome-dawn-YWH5ZS") == 0
    out = capsys.readouterr().out
    assert "running" in out
    assert "claude-opus-4-8" in out
    assert "iteration:  3" in out


def test_status_json_is_machine_readable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    d = _make_run(tmp_path, monkeypatch, [{"ts": _ts(5), "type": "session.start", "mode": "run"}])
    write_worker_pid(d, os.getpid())
    assert _cmd_status("", as_json=True) == 0  # "" -> most recent run
    obj = json.loads(capsys.readouterr().out)
    assert obj["session_id"] == "winsome-dawn-YWH5ZS"
    assert obj["alive"] is True
    assert obj["status"] == "running"
    # A live run keeps elapsing while it waits (its last event is 5 s old and
    # it wrote nothing since); a dead one stops at its last event.
    assert obj["elapsed_s"] >= 4


def test_status_elapsed_of_a_fork_leg_runs_from_its_first_event(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A fork's log opens with loop.resume.start and never carries a
    session.start, so its finished leg read `elapsed: -`. With no
    session.start the scan's start is the first event's timestamp."""
    _make_run(
        tmp_path,
        monkeypatch,
        [
            {"type": "loop.resume.start", "ts": "2026-01-01T00:00:00+00:00", "iteration": 2},
            {"type": "tool.call", "ts": "2026-01-01T00:00:10+00:00", "iteration": 2},
            {
                "type": "session.end",
                "ts": "2026-01-01T00:00:30+00:00",
                "reason": "finish_session",
                "all_passed": True,
            },
        ],
    )
    assert _cmd_status("winsome-dawn-YWH5ZS", as_json=True) == 0
    assert json.loads(capsys.readouterr().out)["elapsed_s"] == 30.0
    assert _cmd_status("winsome-dawn-YWH5ZS") == 0
    assert "elapsed:    30s" in capsys.readouterr().out


def test_status_waiting_when_blocked_on_an_operator_answer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A live run blocked on an unanswered approval/question must read
    "waiting (needs answer)" -- the same first-class status `agent6 sessions`
    gives it -- not "running (long step, likely a provider call)", which sent
    the operator off to wait on a provider while the run sat blocked on THEM."""
    d = _make_run(
        tmp_path,
        monkeypatch,
        [
            {"ts": _ts(400), "type": "session.start", "mode": "run"},
            {"ts": _ts(300), "type": "approval.prompt", "id": "approval-1", "prompt": "rm -rf?"},
        ],
    )
    write_worker_pid(d, os.getpid())
    assert _cmd_status("winsome-dawn-YWH5ZS") == 0
    out = capsys.readouterr().out
    assert "waiting" in out and "needs answer" in out
    assert "provider call" not in out
    assert _cmd_status("winsome-dawn-YWH5ZS", as_json=True) == 0
    obj = json.loads(capsys.readouterr().out)
    assert (obj["status"], obj["detail"]) == ("waiting", "needs answer; attach to respond")


def test_status_crashed_when_pid_dead_and_no_run_end(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A dead worker without a session.end leads with the hub's word ("stale") plus
    this surface's diagnostic detail. The old lead word was "stopped" -- the
    hub's word for an OPERATOR stop (steer_abort), so the same run read as
    deliberately stopped in one surface and crashed in the other."""
    d = _make_run(
        tmp_path,
        monkeypatch,
        [
            {"ts": _ts(40), "type": "session.start"},
            {"ts": _ts(30), "type": "loop.tool.call", "iteration": 1},
        ],
    )
    (d / "worker.pid").write_text("999999")  # almost certainly not a live pid
    assert not worker_is_alive(d)
    _cmd_status("winsome-dawn-YWH5ZS")
    out = capsys.readouterr().out
    assert "state:      stale (no worker, no session.end: likely crashed or killed)" in out
    assert "stopped" not in out


def test_status_words_lead_with_the_listing_word_in_every_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """`sessions show --json` "status" is exactly the word the hub row shows
    for the SAME dir, for every non-session.end state: "created" (was "unknown"),
    "starting" (was a bare "running" while the hub said starting), "waiting",
    "stale". One decision -- status_for_session_dir -- so the two can't drift."""
    from agent6.viewmodel.listing import summarize_session_dir

    d = _make_run(tmp_path, monkeypatch, [{"ts": _ts(5), "type": "session.start", "mode": "run"}])

    def state_word() -> str:
        assert _cmd_status("winsome-dawn-YWH5ZS", as_json=True) == 0
        word = json.loads(capsys.readouterr().out)["status"]
        assert word == summarize_session_dir(d).status
        return word

    write_worker_pid(d, os.getpid())
    assert state_word() == "running"
    (d / "logs.jsonl").unlink()
    assert state_word() == "starting"
    # A dead pid file with no session.start: a worker died LAUNCHING (the pid
    # survives a kill; a clean refusal clears it) -- not the never-ran word.
    (d / "worker.pid").write_text("999999999", encoding="utf-8")
    assert state_word() == "stale"
    (d / "worker.pid").unlink()
    assert state_word() == "created"
    (d / "logs.jsonl").write_text(
        json.dumps({"ts": _ts(30), "type": "session.start", "mode": "run"})
        + "\n"
        + json.dumps({"ts": _ts(9), "type": "approval.prompt", "id": "a1", "prompt": "ok?"})
        + "\n",
        encoding="utf-8",
    )
    write_worker_pid(d, os.getpid())
    assert state_word() == "waiting"
    (d / "worker.pid").write_text("999999999", encoding="utf-8")
    assert state_word() == "stale"


def test_status_leads_with_the_listing_word_then_the_raw_reason(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # `sessions show` must agree with `sessions list`: a finish_session+all_passed run reads
    # "passed", not the opposite "finished" the raw reason alone used to print.
    # The raw reason stays in parens as the diagnostic.
    _make_run(
        tmp_path,
        monkeypatch,
        [
            {"ts": _ts(40), "type": "session.start"},
            {"ts": _ts(1), "type": "session.end", "reason": "finish_session", "all_passed": True},
        ],
    )
    _cmd_status("winsome-dawn-YWH5ZS")
    assert "passed (finish_session)" in capsys.readouterr().out


def test_status_of_a_scoped_green_names_the_scoped_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A pass certified by a scoped gate reads `passed · scoped gate` here as on
    every listing (the label is `status_label`'s); the raw reason stays in parens."""
    _make_run(
        tmp_path,
        monkeypatch,
        [
            {"ts": _ts(40), "type": "session.start"},
            {
                "ts": _ts(1),
                "type": "session.end",
                "reason": "finish_session",
                "all_passed": True,
                "scoped": True,
            },
        ],
    )
    assert _cmd_status("winsome-dawn-YWH5ZS", as_json=True) == 0
    obj = json.loads(capsys.readouterr().out)
    assert (obj["status"], obj["label"], obj["detail"]) == (
        "passed",
        "passed · scoped gate",
        "finish_session",
    )
    assert _cmd_status("winsome-dawn-YWH5ZS") == 0
    assert "state:      passed · scoped gate (finish_session)\n" in capsys.readouterr().out


def test_status_finish_without_all_passed_reads_finished(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _make_run(
        tmp_path,
        monkeypatch,
        [
            {"ts": _ts(40), "type": "session.start"},
            {"ts": _ts(1), "type": "session.end", "reason": "finish_session", "all_passed": False},
        ],
    )
    _cmd_status("winsome-dawn-YWH5ZS")
    assert "finished (finish_session)" in capsys.readouterr().out


def test_status_error_reason_reads_failed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _make_run(
        tmp_path,
        monkeypatch,
        [
            {"ts": _ts(40), "type": "session.start"},
            {"ts": _ts(1), "type": "session.end", "reason": "provider_error", "all_passed": False},
        ],
    )
    # The label already carries the reason; the parenthetical is not repeated.
    assert _cmd_status("winsome-dawn-YWH5ZS") == 0
    assert "state:      failed · provider error\n" in capsys.readouterr().out


def test_status_shows_fan_out_compare_outcome(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """`sessions show` prints where a lane placed in its fan-out (+ the judge's
    rationale), and the JSON carries the raw compare block."""
    d = _make_run(tmp_path, monkeypatch, [{"ts": _ts(5), "type": "session.start", "mode": "run"}])
    manifest = json.loads((d / "manifest.json").read_text("utf-8"))
    manifest["compare"] = {
        "group": "fan", "rank": 1, "of": 2, "winner": True,
        "ranked_by": "judge", "rationale": "cleanest diff, all tests pass",
    }  # fmt: skip
    (d / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    _cmd_status("winsome-dawn-YWH5ZS")
    out = capsys.readouterr().out
    assert "compare:    rank 1/2 · winner · judge" in out
    assert "judge: cleanest diff, all tests pass" in out

    _cmd_status("winsome-dawn-YWH5ZS", as_json=True)
    obj = json.loads(capsys.readouterr().out)
    assert obj["compare"]["winner"] is True and obj["compare"]["rank"] == 1


def test_worker_pid_clear(tmp_path: Path) -> None:
    from agent6.sessions.ipc import clear_worker_pid, read_worker_pid

    write_worker_pid(tmp_path, os.getpid())
    assert read_worker_pid(tmp_path) == os.getpid()
    clear_worker_pid(tmp_path)
    assert read_worker_pid(tmp_path) is None
    assert not worker_is_alive(tmp_path)


def test_status_shows_usage_from_budget_update_event(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # `sessions show` must read budget.update (the authoritative post-call totals),
    # not loop.budget (emitted BEFORE each call: lags one call, 0 on iter 1).
    # A stray loop.budget must NOT override the real usage.
    d = _make_run(
        tmp_path,
        monkeypatch,
        [
            {"ts": _ts(40), "type": "session.start"},
            {
                "ts": _ts(30),
                "type": "loop.budget",
                "iteration": 1,
                "input_tokens": 0,
                "output_tokens": 0,
                "cost_usd": 0.0,
            },
            {
                "ts": _ts(20),
                "type": "budget.update",
                "input_total": 1000,
                "output_total": 200,
                "usd_total": 0.0123,
                "usd_partial": False,
            },
            {
                "ts": _ts(5),
                "type": "budget.update",
                "input_total": 4200,
                "output_total": 800,
                "usd_total": 0.0456,
                "usd_partial": False,
            },
        ],
    )
    write_worker_pid(d, os.getpid())
    _cmd_status("winsome-dawn-YWH5ZS")
    out = capsys.readouterr().out
    assert "in=4200 out=800" in out  # latest budget.update wins, not the 0/0 loop.budget
    assert "$0.0456" in out
    # json carries the same
    _cmd_status("winsome-dawn-YWH5ZS", as_json=True)
    obj = json.loads(capsys.readouterr().out)
    assert obj["input_tokens"] == 4200 and obj["cost_usd"] == 0.0456


def test_status_names_the_pins_in_force(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A run's pinned instructions (--pin, /pin) bind for the whole run; the
    show page names them, one per line, and the JSON carries the list. The
    leg-start announcement replaces the list, /pin appends (the fold's rule)."""
    _make_run(
        tmp_path,
        monkeypatch,
        [
            {"ts": _ts(40), "type": "session.start"},
            {"ts": _ts(30), "type": "loop.pin.restored", "pins": ["never touch tests"]},
            {"ts": _ts(20), "type": "loop.pin.added", "text": "keep the API stable"},
        ],
    )
    _cmd_status("winsome-dawn-YWH5ZS")
    out = capsys.readouterr().out
    assert "pins:       never touch tests\n            keep the API stable\n" in out
    _cmd_status("winsome-dawn-YWH5ZS", as_json=True)
    assert json.loads(capsys.readouterr().out)["pins"] == [
        "never touch tests",
        "keep the API stable",
    ]


def test_status_cost_cumulative_and_unfinished_across_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # A resume leg restarts the budget from 0 and un-finishes the run; `runs
    # show` banks legs (same rule as `sessions list` and the run view) and must
    # not report leg 1's session.end for a run that is live again. A valid-JSON
    # non-object line is skipped, not a crash.
    d = _make_run(
        tmp_path,
        monkeypatch,
        [
            {"ts": _ts(60), "type": "session.start"},
            {
                "ts": _ts(50),
                "type": "budget.update",
                "input_total": 1000,
                "output_total": 200,
                "usd_total": 0.02,
                "usd_partial": True,
            },
            {"ts": _ts(40), "type": "session.end", "reason": "finish_session", "all_passed": True},
            {"ts": _ts(30), "type": "loop.resume.start", "iteration": 4},
            {
                "ts": _ts(5),
                "type": "budget.update",
                "input_total": 300,
                "output_total": 50,
                "usd_total": 0.005,
                "usd_partial": False,
            },
        ],
    )
    logs = d / "logs.jsonl"
    logs.write_text(logs.read_text(encoding="utf-8") + "42\n", encoding="utf-8")
    write_worker_pid(d, os.getpid())
    _cmd_status("winsome-dawn-YWH5ZS", as_json=True)
    obj = json.loads(capsys.readouterr().out)
    assert obj["cost_usd"] == pytest.approx(0.025)
    assert obj["usd_partial"] is True  # sticky: leg 1's unpriced spend
    assert obj["status"] == "running"  # not leg 1's "passed (finish_session)"
    assert obj["input_tokens"] == 300  # token gauges stay per-leg


def test_status_missing_id_and_empty_state_speak_human(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # A bad id names itself and where it looked, without leaking the
    # bucket alternation under sessions/; an empty state dir gets
    # the same first-contact copy as `runs`.
    monkeypatch.setenv("AGENT6_STATE_HOME", str(tmp_path / "state"))
    repo = tmp_path / "repo"
    repo.mkdir()
    monkeypatch.chdir(repo)
    assert _cmd_status("zzz", as_json=False) == 2
    err = capsys.readouterr().err
    assert "no session matches 'zzz'" in err and "machines" not in err
    assert _cmd_status("", as_json=False) == 2
    assert 'no sessions yet. Start one with `agent6 run "<task>"`.' in capsys.readouterr().err


def test_status_text_labels_leg_scoped_figures_on_a_resumed_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # Cost is banked across legs, token counters are the latest leg's; the
    # usage line must say which scope each figure describes once they differ.
    d = _make_run(
        tmp_path,
        monkeypatch,
        [
            {"ts": _ts(90), "type": "session.start", "mode": "run", "user_task": "t"},
            {
                "ts": _ts(80),
                "type": "budget.update",
                "input_total": 9000,
                "output_total": 500,
                "usd_total": 0.02,
            },
            {"ts": _ts(70), "type": "session.end", "reason": "finish_session", "all_passed": True},
            {"ts": _ts(60), "type": "loop.resume.start", "iteration": 4},
            {
                "ts": _ts(10),
                "type": "budget.update",
                "input_total": 300,
                "output_total": 50,
                "usd_total": 0.005,
            },
            {"ts": _ts(5), "type": "session.end", "reason": "finish_session", "all_passed": True},
        ],
    )
    write_worker_pid(d, 999999999)
    _cmd_status("winsome-dawn-YWH5ZS")
    out = capsys.readouterr().out
    assert "in=300 out=50 (latest leg)" in out
    assert "cost $0.0250 (all 2 legs)" in out


def test_worker_is_alive_reads_a_foreign_owned_pid_as_dead(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A worker is always spawned by the probing user, so PermissionError on
    the recorded pid means the worker died and the kernel reused the number
    for another user's process. Reading it as alive rendered a crashed run
    "running" forever and hung the /parallel lane await permanently."""
    if os.geteuid() == 0:
        pytest.skip("root can signal any pid; the foreign-owner probe needs a non-root euid")
    monkeypatch.setenv("AGENT6_STATE_HOME", str(tmp_path / "state"))
    d = tmp_path / "run"
    d.mkdir()
    write_worker_pid(d, 1)  # init: exists, foreign-owned -> PermissionError
    assert not worker_is_alive(d)
    write_worker_pid(d, os.getpid())
    assert worker_is_alive(d)


def test_concurrent_answer_writers_do_not_race_on_the_temp(tmp_path: Path) -> None:
    """Two concurrently-live front-ends (attach + web) answering the same
    prompt both wrote the SAME sibling .tmp: the loser hit FileNotFoundError
    after the winner's rename -- a 500 on an answer that actually landed. The
    durable write now uses a unique mkstemp temp per call."""
    import threading

    from agent6.sessions.ipc import write_answer

    d = tmp_path / "run"
    d.mkdir()
    barrier = threading.Barrier(2)
    errors: list[Exception] = []

    def _answer() -> None:
        try:
            barrier.wait(timeout=5)
            for _ in range(50):
                write_answer(d, "approval-1", "yes")
        except Exception as exc:
            errors.append(exc)

    t1, t2 = threading.Thread(target=_answer), threading.Thread(target=_answer)
    t1.start()
    t2.start()
    t1.join(timeout=15)
    t2.join(timeout=15)
    assert errors == []
    assert (d / "approvals" / "approval-1.answer").read_text(encoding="utf-8") == "yes"


def test_status_ambiguous_prefix_names_the_candidates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """An ambiguous id prefix must say so and name the matches, as `attach` and
    `sessions stop` do. `sessions show` swallowed the resolver's error and printed
    "no session matches 't'" -- telling the operator no such run exists while two
    did."""
    d = _make_run(tmp_path, monkeypatch, [{"ts": _ts(5), "type": "session.start", "mode": "run"}])
    sibling = d.parent / "winsome-dusk-AAAAAA"
    sibling.mkdir()
    (sibling / "logs.jsonl").write_text(
        json.dumps({"ts": _ts(9), "type": "session.start", "mode": "run"}) + "\n", encoding="utf-8"
    )

    assert _cmd_status("winsome-d") == 2
    err = capsys.readouterr().err
    assert "ambiguous" in err
    assert "winsome-dawn-YWH5ZS" in err and "winsome-dusk-AAAAAA" in err
    assert "no session matches" not in err


def test_a_nonpositive_recorded_pid_never_reads_alive(tmp_path: Path) -> None:
    """`os.kill(0, 0)` signals the process group and `os.kill(-1, 0)` every
    process, so both succeed: a worker.pid holding 0 or -1 read ALIVE forever,
    refusing resume and hanging the /parallel lane await -- the exact symptom
    the identity record exists to kill. The front-end probe guards this; the
    worker probe did not."""
    for junk in ("0", "-1"):
        (tmp_path / "worker.pid").write_text(junk, encoding="utf-8")
        assert worker_is_alive(tmp_path) is False, junk


def test_worker_pid_is_published_atomically(tmp_path: Path) -> None:
    """The last polled state file written with plain write_text: it truncates,
    then writes, so a reader in that window sees a PREFIX of the pid with the
    start-time identity stripped -- and a prefix that happens to name a live
    process you own reads alive with nothing left to refute it, which is the
    recycled-pid lie the identity was added to kill."""
    from agent6.sessions import ipc

    seen: list[str] = []
    real = ipc.atomic_write

    def spy(path: Path, text: str) -> None:
        seen.append(path.name)
        real(path, text)

    monkey = pytest.MonkeyPatch()
    monkey.setattr(ipc, "atomic_write", spy)
    try:
        write_worker_pid(tmp_path, os.getpid())
    finally:
        monkey.undo()
    assert seen == ["worker.pid"]
    assert worker_is_alive(tmp_path) is True  # still a valid record


def test_a_started_session_with_no_pid_file_is_not_running(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A worker writes its pid BEFORE session.start, so a started session with
    no pid file cleared it on the way out: it is gone, whatever the log says.

    Reachable by an ordinary `agent6 run ... | head`: SIGPIPE unwinds through
    the finally, which clears the pid but writes no session.end. Treating the
    absence as weaker evidence than a dead pid inverted the two -- `kill -9`
    leaves the pid file and read "stale" at once, while the tidier death read
    "running" for the whole 600s silence window.
    """
    d = _make_run(
        tmp_path,
        monkeypatch,
        [
            {"ts": _ts(20), "type": "session.start", "mode": "run"},
            {"ts": _ts(2), "type": "loop.tool.call", "iteration": 2},
        ],
    )
    assert not (d / "worker.pid").exists()

    assert _cmd_status("winsome-dawn-YWH5ZS") == 0
    out = capsys.readouterr().out
    assert "running" not in out, "a session with no live worker must never render as running"
    assert "stale" in out


def test_a_start_event_is_never_readable_before_the_pid_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The invariant `_running_is_stale` rests on: no pid file under a started
    session means the worker cleared it, not that it has not written it yet.
    Emitting session.start first opened a window where a live session read as
    dead on every surface. The start emitter owns the order now, so no entry
    point (the loop's two starts, machine create's header) can invert it."""
    from agent6.events import EventSink
    from agent6.sessions.ipc import emit_session_start

    sdir = tmp_path / "sess"
    sdir.mkdir()
    pid_present_at_emit: list[bool] = []
    real_emit = EventSink.emit

    def spy(self: EventSink, event_type: str, /, **fields: Any) -> None:
        pid_present_at_emit.append((sdir / "worker.pid").exists())
        real_emit(self, event_type, **fields)

    monkeypatch.setattr(EventSink, "emit", spy)
    emit_session_start(EventSink(sdir / "logs.jsonl"), sdir, "session.start", mode="run")
    assert pid_present_at_emit == [True]


def _stamp_manifest(d: Path, **fields: Any) -> None:
    (d / "manifest.json").write_text(
        json.dumps({"mode": "run", "models": {"driver": {"model": "m"}}, **fields})
    )


def test_status_says_where_the_changes_are(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """`sessions show` carries the run's branch fact the web header shows and
    the end-of-run footer said: on the run branch awaiting its merge (with
    the command), merged into the base, or a branch no commit reached. The
    JSON carries run_branch / base_branch / merged_into."""
    d = _make_run(tmp_path, monkeypatch, [{"ts": _ts(5), "type": "session.start", "mode": "run"}])
    subprocess.run(["git", "init", "-q", "-b", "main"], cwd=Path.cwd(), check=True)
    subprocess.run(
        ["git", "commit", "-q", "--allow-empty", "-m", "base"],
        cwd=Path.cwd(),
        check=True,
        env={
            **os.environ,
            "GIT_AUTHOR_NAME": "t",
            "GIT_AUTHOR_EMAIL": "t@t",
            "GIT_COMMITTER_NAME": "t",
            "GIT_COMMITTER_EMAIL": "t@t",
        },
    )
    branch = "agent6/winsome-dawn-YWH5ZS"

    def show() -> str:
        assert _cmd_status("winsome-dawn-YWH5ZS") == 0
        return capsys.readouterr().out

    _stamp_manifest(d, run_branch=branch, base_branch="main")
    assert f"changes:    {branch} (no commits)" in show()
    subprocess.run(["git", "branch", branch], cwd=Path.cwd(), check=True)
    assert (
        f"changes:    {branch} → merges into main; merge with: agent6 sessions merge"
        " winsome-dawn-YWH5ZS"
    ) in show()
    tip = subprocess.run(
        ["git", "rev-parse", branch], cwd=Path.cwd(), check=True, capture_output=True, text=True
    ).stdout.strip()
    _stamp_manifest(
        d, run_branch=branch, base_branch="main", merged={"into": "main", "sha": tip, "tip": tip}
    )
    assert f"changes:    {branch} (merged into main)" in show()
    assert _cmd_status("winsome-dawn-YWH5ZS", as_json=True) == 0
    obj = json.loads(capsys.readouterr().out)
    assert (obj["run_branch"], obj["base_branch"], obj["merged_into"]) == (branch, "main", "main")
    _stamp_manifest(d, mode="ask")
    assert "changes:" not in show()
    # The branch deleted (no merge stamp) while the chain ref keeps the
    # commits: the ref is named, and the merge still offered.
    _stamp_manifest(d, run_branch=branch, base_branch="main")
    chain = "refs/agent6/winsome-dawn-YWH5ZS/head"
    subprocess.run(["git", "update-ref", chain, tip], cwd=Path.cwd(), check=True)
    subprocess.run(["git", "branch", "-D", branch], cwd=Path.cwd(), check=True, capture_output=True)
    out = show()
    assert "changes:    refs/agent6/winsome-dawn-YWH5ZS/head (" in out
    assert "is gone; the commits are kept); merge with: agent6 sessions merge" in out


def test_status_of_an_undone_run_does_not_offer_a_merge(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The changes line of an undone run agrees with the listing: the branch
    was taken back by /undo, so no merge is offered. It read the end reason
    not at all and offered `merge with: agent6 sessions merge <id>`."""
    d = _make_run(
        tmp_path,
        monkeypatch,
        [
            {"ts": _ts(5), "type": "session.start", "mode": "run"},
            {"ts": _ts(1), "type": "session.end", "reason": "undone", "all_passed": False},
        ],
    )
    subprocess.run(["git", "init", "-q", "-b", "main"], cwd=Path.cwd(), check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.email=t@t",
            "-c",
            "user.name=t",
            "commit",
            "-q",
            "--allow-empty",
            "-m",
            "b",
        ],
        cwd=Path.cwd(),
        check=True,
    )
    branch = "agent6/winsome-dawn-YWH5ZS"
    subprocess.run(["git", "branch", branch], cwd=Path.cwd(), check=True)
    _stamp_manifest(d, run_branch=branch, base_branch="main")

    assert _cmd_status("winsome-dawn-YWH5ZS") == 0
    out = capsys.readouterr().out
    assert "state:      undone" in out
    assert f"changes:    {branch} (taken back by /undo)" in out
    assert "merge with:" not in out


def test_status_of_an_ask_does_not_repeat_its_word(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """An ask ends with reason "answered", the listing word too: one word, no
    "answered (answered)"."""
    _make_run(
        tmp_path,
        monkeypatch,
        [
            {"ts": _ts(9), "type": "session.start", "mode": "ask"},
            {"ts": _ts(5), "type": "session.end", "reason": "answered", "all_passed": False},
        ],
    )
    assert _cmd_status("winsome-dawn-YWH5ZS") == 0
    assert "state:      answered\n" in capsys.readouterr().out


def test_show_json_label_matches_the_listing_cell(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """One vocabulary across the two surfaces: `show --json`'s label was the
    bare word while the listing folds the mode in, so a script could not match
    a run's row to its detail view."""
    from agent6.viewmodel.format import listing_status_label
    from agent6.viewmodel.listing import summarize_session_dir

    d = _make_run(tmp_path, monkeypatch, [{"ts": _ts(5), "type": "session.start", "mode": "plan"}])
    (d / "manifest.json").write_text(json.dumps({"mode": "plan"}), encoding="utf-8")
    write_worker_pid(d, os.getpid())

    assert _cmd_status("winsome-dawn-YWH5ZS", as_json=True) == 0

    obj = json.loads(capsys.readouterr().out)
    s = summarize_session_dir(d)
    assert obj["label"] == listing_status_label(s.mode, s.status, s.reason, unmerged=s.unmerged)
    assert obj["label"] == "plan · running", "the mode is folded in, as the listing folds it"
