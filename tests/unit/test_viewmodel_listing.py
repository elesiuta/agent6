# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The shared run-listing helpers (session_mtime, task_snippet, summarize_session_dir)."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest

from agent6.sessions.manifest import CompareStamp
from agent6.viewmodel import (
    is_session_husk,
    is_winner,
    manifest_branches,
    session_compare,
    session_is_live,
    session_mtime,
    summarize_session_dir,
    task_snippet,
)
from agent6.viewmodel.format import format_branch, format_compare, format_lineage


def test_run_mtime_prefers_log_over_dir(tmp_path: Path) -> None:
    d = tmp_path / "run"
    d.mkdir()
    log = d / "logs.jsonl"
    log.write_text("{}\n", encoding="utf-8")
    os.utime(log, (1000.0, 1000.0))
    os.utime(d, (5000.0, 5000.0))  # dir bumped later (a viewer wrote frontend.pid)
    assert session_mtime(d) == 1000.0  # keyed off the log, not the dir


def test_run_mtime_falls_back_to_dir(tmp_path: Path) -> None:
    d = tmp_path / "run"
    d.mkdir()
    os.utime(d, (2000.0, 2000.0))
    assert session_mtime(d) == 2000.0  # no log yet -> dir mtime


def test_task_snippet_skips_seeded_file_block() -> None:
    task = (
        "# agent6 ask\n\n## Question\n\n"
        '<file path="a.py">\ndef f(): pass\nSHOULD NOT SHOW\n</file>\n\n'
        "why is the broker slow?\n\n## Answer\n"
    )
    assert task_snippet(task) == "why is the broker slow?"


def test_task_snippet_is_the_operators_words_under_a_from_seed_and_skills() -> None:
    """`run --from` prepends a `<prior-run>` digest and `--skill` a preamble +
    `<skill>` blocks + `---`; the headline everywhere is what the operator
    typed. A clipped copy that cuts inside the block (the 200-char event of
    the older writers) drops the open block instead of showing its opener."""
    from agent6.task_text import operator_task_text

    seeded = (
        '<prior-run id="agile-echo-H2EWX5">\nThis question is about a PRIOR agent6 run.\n'
        "## Run task\nhow many functions?\n</prior-run>\n\n"
        "add a module docstring to calc.py"
    )
    assert task_snippet(seeded) == "add a module docstring to calc.py"
    skilled = (
        "Apply the operator-installed skill(s) below to this task.\n\n"
        '<skill name="tdd">\nwrite the test first\n</skill>\n\n---\n\n'
        "add a --json flag\nmore detail"
    )
    assert task_snippet(skilled) == "add a --json flag"
    assert operator_task_text(skilled) == "add a --json flag\nmore detail"
    assert operator_task_text(seeded[:60]) == ""  # clipped inside the block: nothing invented
    assert task_snippet("plain words") == "plain words"


def test_task_snippet_plain_task() -> None:
    assert task_snippet("add a --json flag\nmore detail") == "add a --json flag"


def test_task_snippet_falls_back_to_stripped_text() -> None:
    assert task_snippet("   ") == ""


def _stamp(session_dir: Path, compare: object) -> None:
    session_dir.mkdir(parents=True, exist_ok=True)
    (session_dir / "manifest.json").write_text(json.dumps({"compare": compare}), encoding="utf-8")


def test_run_compare_and_is_winner_read_the_manifest_block(tmp_path: Path) -> None:
    # The fixture writes the legacy `group` key; the model ignores it (old-shape
    # compat), so a fan-out lane recorded before the dedup still reads its stamp.
    win = tmp_path / "win"
    _stamp(win, {"group": "fan", "rank": 1, "of": 2, "winner": True, "ranked_by": "judge"})
    assert is_winner(win) is True
    assert isinstance(session_compare(win), CompareStamp)
    loser = tmp_path / "loser"
    _stamp(loser, {"group": "fan", "rank": 2, "of": 2, "winner": False, "ranked_by": "judge"})
    assert is_winner(loser) is False
    # A run outside any fan-out (no manifest / no compare block) reads as None.
    plain = tmp_path / "plain"
    plain.mkdir()
    assert session_compare(plain) is None and is_winner(plain) is False


def test_format_compare_headline_and_rationale() -> None:
    won = format_compare(
        CompareStamp(rank=1, of=3, winner=True, ranked_by="judge", rationale="cleanest diff")
    )
    assert won == ("rank 1/3 · winner · judge", "cleanest diff")
    # A loser, mechanical, no rationale.
    lost = format_compare(
        CompareStamp(rank=2, of=3, winner=False, ranked_by="mechanical", rationale="")
    )
    assert lost == ("rank 2/3 · mechanical", "")
    # No stamp -> None.
    assert format_compare(None) is None


def test_format_branch_is_the_one_wording_and_manifest_branches_carries_it(tmp_path: Path) -> None:
    """The run header's branch line (TUI, web, `sessions show`): the run branch
    merged into its base, else the base a merge lands on; "" without a run
    branch. `manifest_branches` hands it to every header as `branch_line`."""
    import json

    assert format_branch("agent6/x", "main", "") == "agent6/x → merges into main"
    assert format_branch("agent6/x", "main", "main") == "agent6/x (merged into main)"
    assert format_branch("agent6/x", "", "") == "agent6/x"
    assert format_branch("", "main", "") == ""
    d = tmp_path / "run-x"
    d.mkdir()
    (d / "manifest.json").write_text(
        json.dumps({"mode": "run", "run_branch": "agent6/x", "base_branch": "main"})
    )
    assert manifest_branches(d) == {
        "run_branch": "agent6/x",
        "base_branch": "main",
        "branch_line": "agent6/x → merges into main",
    }
    (d / "manifest.json").write_text(json.dumps({"mode": "ask"}))
    assert manifest_branches(d) == {}


def test_manifest_header_carries_the_fork_lineage_in_one_wording(tmp_path: Path) -> None:
    """`forked_from` is the line every header shows for a run `agent6 fork`
    made (`sessions show`, the TUI dashboard, the web run header):
    `<parent>@turn <n> (<sha12>)`; absent for a run that is not a fork."""
    import json

    from agent6.viewmodel import manifest_header

    assert format_lineage("orig-run-AAAAAA", 3, "a" * 40) == "orig-run-AAAAAA@turn 3 (aaaaaaaaaaaa)"
    assert format_lineage("orig-run-AAAAAA", 3, None) == "orig-run-AAAAAA@turn 3"
    assert format_lineage(None, None, None) == ""
    d = tmp_path / "fork-x"
    d.mkdir()
    (d / "manifest.json").write_text(
        json.dumps(
            {
                "mode": "run",
                "parent_session_id": "orig-run-AAAAAA",
                "forked_from_turn": 3,
                "forked_from_sha": "b" * 40,
            }
        )
    )
    assert manifest_header(d)["forked_from"] == "orig-run-AAAAAA@turn 3 (bbbbbbbbbbbb)"
    (d / "manifest.json").write_text(json.dumps({"mode": "run"}))
    assert "forked_from" not in manifest_header(d)


def test_manifest_branches_claims_merged_only_while_the_stamp_holds(tmp_path: Path) -> None:
    """A run resumed after its merge commits past the stamp: with the repo at
    hand the header says the branch awaits a merge again (the web Merge button
    read the raw stamp and stayed disabled over unmerged commits)."""
    import json
    import subprocess

    repo = tmp_path / "repo"
    repo.mkdir()
    env = {"GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t"}
    env["GIT_COMMITTER_EMAIL"] = "t@t"
    git = ["git", "-C", str(repo)]
    subprocess.run([*git, "init", "-q", "-b", "main"], check=True)
    subprocess.run([*git, "commit", "-q", "--allow-empty", "-m", "base"], check=True, env=env)
    subprocess.run([*git, "branch", "agent6/x"], check=True)
    tip = subprocess.run(
        [*git, "rev-parse", "agent6/x"], check=True, capture_output=True, text=True
    ).stdout.strip()
    d = tmp_path / "run-x"
    d.mkdir()
    stamped = {
        "mode": "run",
        "run_branch": "agent6/x",
        "base_branch": "main",
        "merged": {"into": "main", "sha": tip, "tip": tip},
    }
    (d / "manifest.json").write_text(json.dumps(stamped))
    assert manifest_branches(d, repo=repo)["branch_line"] == "agent6/x (merged into main)"
    # A later commit on the run branch: the stamp no longer describes it.
    subprocess.run([*git, "checkout", "-q", "agent6/x"], check=True)
    subprocess.run([*git, "commit", "-q", "--allow-empty", "-m", "more"], check=True, env=env)
    got = manifest_branches(d, repo=repo)
    assert "merged_into" not in got
    assert got["branch_line"] == "agent6/x → merges into main"
    # Without the repo the manifest fact stands as recorded.
    assert manifest_branches(d)["merged_into"] == "main"


def test_manifest_branches_names_a_branch_only_once_it_exists(tmp_path: Path) -> None:
    """The manifest names the run branch at run start; git creates it at the
    first commit. A run stopped before one (or parked before starting) had a
    header reading `agent6/x → merges into main` and an enabled Merge that the
    CLI then refused with "no branch to merge"; `sessions show --json` already
    reported `run_branch` null. One rule, `existing_run_branch`, for both."""
    import json
    import subprocess

    from agent6.sessions.manifest import read_manifest
    from agent6.viewmodel import existing_run_branch, session_snapshot

    repo = tmp_path / "repo"
    repo.mkdir()
    git = ["git", "-C", str(repo)]
    subprocess.run([*git, "init", "-q", "-b", "main"], check=True)
    d = tmp_path / "run-x"
    d.mkdir()
    (d / "manifest.json").write_text(
        json.dumps({"mode": "run", "run_branch": "agent6/x", "base_branch": "main"})
    )
    (d / "logs.jsonl").write_text(
        json.dumps({"type": "session.start", "user_task": "t"})
        + "\n"
        + json.dumps({"type": "session.end", "reason": "steer_abort"})
        + "\n"
    )
    assert existing_run_branch(read_manifest(d), repo) == ""
    assert manifest_branches(d, repo=repo) == {"base_branch": "main"}
    snap = session_snapshot(d, repo=repo)
    assert "run_branch" not in snap and "branch_line" not in snap
    # The first commit creates it: from here the header names it and Merge lands.
    env = {"GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t"}
    env["GIT_COMMITTER_EMAIL"] = "t@t"
    subprocess.run([*git, "commit", "-q", "--allow-empty", "-m", "base"], check=True, env=env)
    subprocess.run([*git, "branch", "agent6/x"], check=True)
    assert existing_run_branch(read_manifest(d), repo) == "agent6/x"
    assert session_snapshot(d, repo=repo)["branch_line"] == "agent6/x → merges into main"


# --- summarize_session_dir / status_word (shared by TUI hub, web hub, runs list) --


def _write_run(base: Path, sub: str, session_id: str, events: list[dict[str, object]]) -> Path:
    """A session dir as one really looks on disk: a started session has a LIVE
    worker.pid, because the worker writes it before emitting its start event.
    Tests that model a death overwrite or unlink it."""
    import json
    import os

    rd = base / "sessions" / sub / session_id
    rd.mkdir(parents=True)
    (rd / "logs.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events), encoding="utf-8")
    if any(e.get("type") in ("session.start", "loop.resume.start") for e in events):
        (rd / "worker.pid").write_text(str(os.getpid()), encoding="utf-8")
    return rd


def test_summary_reads_mode_task_and_passed(tmp_path: Path) -> None:
    rd = _write_run(
        tmp_path,
        "runs",
        "r1",
        [
            {"type": "session.start", "mode": "run", "user_task": "fix [the] bug"},
            {"type": "tool.call", "name": "read_file"},
            {"type": "budget.update", "usd_total": 0.12},
            {"type": "session.end", "all_passed": True, "reason": "finish_session"},
        ],
    )
    s = summarize_session_dir(rd)
    assert (s.mode, s.task, s.status, s.reason) == ("run", "fix [the] bug", "passed", "")
    assert s.cost_usd == 0.12


def test_verify_verdict_reads_the_gate_facts_not_the_status_word(tmp_path: Path) -> None:
    """The judge's verify tri-state came from the folded status word, and
    finish_session over a red gate folds to "finished": the compare table and
    the judge called a RED gate "no verify", so an all-red fan-out crowned a
    rank 1 and exited 0. The verdict now reads the gate facts: the last
    verify.end this leg, and the end's all_passed."""
    red_finish: list[dict[str, object]] = [
        {"type": "session.start", "mode": "run", "user_task": "t"},
        {"type": "verify.end", "cmd": ["pytest"], "exit_code": 1},
        {"type": "session.end", "all_passed": False, "reason": "finish_session"},
    ]
    rd = _write_run(tmp_path, "runs", "r-red", red_finish)
    assert summarize_session_dir(rd).verify_ok is False, "a red gate read as no-verify"

    green: list[dict[str, object]] = [
        {"type": "session.start", "mode": "run", "user_task": "t"},
        {"type": "verify.end", "cmd": ["pytest"], "exit_code": 1},
        {"type": "verify.end", "cmd": ["pytest"], "exit_code": 0},
        {"type": "session.end", "all_passed": True, "reason": "finish_session"},
    ]
    assert summarize_session_dir(_write_run(tmp_path, "runs", "r-green", green)).verify_ok is True

    gateless: list[dict[str, object]] = [
        {"type": "session.start", "mode": "run", "user_task": "t"},
        {"type": "session.end", "all_passed": False, "reason": "settled"},
    ]
    assert summarize_session_dir(_write_run(tmp_path, "runs", "r-none", gateless)).verify_ok is None

    # A plan never runs its (inferred) gate: no verdict to claim.
    plan: list[dict[str, object]] = [
        {"type": "session.start", "mode": "plan", "user_task": "t"},
        {"type": "session.end", "all_passed": True, "reason": "finish_planning"},
    ]
    assert summarize_session_dir(_write_run(tmp_path, "plans", "p1", plan)).verify_ok is None

    # A prior leg's red is not this leg's: the observation is leg-scoped, like
    # the token counters (the resumed leg may never run the gate at all).
    resumed: list[dict[str, object]] = [
        {"type": "session.start", "mode": "run", "user_task": "t"},
        {"type": "verify.end", "cmd": ["pytest"], "exit_code": 1},
        {"type": "loop.resume.start"},
        {"type": "session.end", "all_passed": False, "reason": "finish_session"},
    ]
    rd = _write_run(tmp_path, "runs", "r-legs", resumed)
    assert summarize_session_dir(rd).verify_ok is None


def test_summary_ask_reads_answered_not_passed(tmp_path: Path) -> None:
    # An ask verifies nothing; "passed" for a Q&A is a category error. The ask
    # flow's own banner already says "answered", so listings must agree.
    rd = _write_run(
        tmp_path,
        "asks",
        "a1",
        [
            {"type": "session.start", "mode": "ask", "user_task": "what does x do?"},
            {"type": "session.end", "all_passed": True, "reason": "answered"},
        ],
    )
    s = summarize_session_dir(rd)
    assert (s.mode, s.status, s.reason) == ("ask", "answered", "")


def test_summary_failure_carries_its_reason(tmp_path: Path) -> None:
    """The core truth fix: a provider_error death reads 'failed · provider_error',
    never a neutral 'done' the operator scrolls past."""
    rd = _write_run(
        tmp_path,
        "runs",
        "r1",
        [
            {"type": "session.start", "mode": "run", "user_task": "t"},
            {"type": "session.end", "all_passed": False, "reason": "provider_error"},
        ],
    )
    s = summarize_session_dir(rd)
    assert (s.status, s.reason) == ("failed", "provider_error")


def test_summary_stop_is_not_a_failure(tmp_path: Path) -> None:
    rd = _write_run(
        tmp_path,
        "runs",
        "r1",
        [
            {"type": "session.start", "mode": "run", "user_task": "t"},
            {"type": "session.end", "all_passed": False, "reason": "steer_abort"},
        ],
    )
    assert summarize_session_dir(rd).status == "stopped"


def test_summary_interrupt_reads_as_stopped(tmp_path: Path) -> None:
    # A Ctrl-C interrupt is the operator's own act, like steer_abort -- not a
    # failure the listing should flag red.
    rd = _write_run(
        tmp_path,
        "runs",
        "r1",
        [
            {"type": "session.start", "mode": "run", "user_task": "t"},
            {"type": "session.end", "all_passed": False, "reason": "interrupted"},
        ],
    )
    assert summarize_session_dir(rd).status == "stopped"


def test_summary_undone_reads_undone_and_never_unmerged(tmp_path: Path) -> None:
    """/undo ends a run with reason "undone": its own word on every surface
    (the listing folded it into "stopped" while the console said "undone
    (forked back)"), and never the unmerged mark, whatever its branch holds
    (see SessionSummary.unmerged)."""
    from agent6.viewmodel.format import listing_status_label

    rd = _write_run(
        tmp_path,
        "runs",
        "r-undone",
        [
            {"type": "session.start", "mode": "run", "user_task": "t"},
            {"type": "session.end", "reason": "undone", "all_passed": False},
        ],
    )
    (rd / "manifest.json").write_text(
        json.dumps(
            {"mode": "run", "user_task": "t", "base_sha": "b" * 40, "run_branch": "agent6/r"}
        ),
        encoding="utf-8",
    )
    s = summarize_session_dir(rd, branch_tips={"agent6/r": "c" * 40})
    assert (s.status, s.reason, s.unmerged) == ("undone", "", False)
    assert listing_status_label(s.mode, s.status, s.reason, unmerged=s.unmerged) == "undone"


def test_summary_resume_unfinishes(tmp_path: Path) -> None:
    """A detached resume appends past the first session.end; the run is running
    again, not whatever it last ended as."""
    rd = _write_run(
        tmp_path,
        "runs",
        "r1",
        [
            {"type": "session.start", "mode": "run", "user_task": "t"},
            {"type": "session.end", "all_passed": False, "reason": "steer_abort"},
            {"type": "loop.resume.start", "iteration": 2},
        ],
    )
    assert summarize_session_dir(rd).status == "running"


def test_summary_running_and_stale(tmp_path: Path) -> None:
    """Liveness is the worker, not log silence: the pid file present-and-live
    is the whole difference between "running" and "stale"."""
    rd = _write_run(tmp_path, "runs", "r2", [{"type": "session.start", "mode": "plan"}])
    assert summarize_session_dir(rd).status == "running"
    (rd / "worker.pid").unlink()  # the worker's finally cleared it on the way out
    assert summarize_session_dir(rd).status == "stale"


def test_summary_unanswered_approval_reads_waiting(tmp_path: Path) -> None:
    # A live run whose LAST event is an unanswered approval (or ask_user
    # question) is blocked on the operator; "running" read as busy, and an
    # approval-parked lane sat invisible in every hub for hours.
    rd = _write_run(
        tmp_path,
        "runs",
        "r5",
        [
            {"type": "session.start", "mode": "run", "user_task": "t"},
            {"type": "approval.prompt", "id": "a1", "prompt": "Allow run_command: pytest"},
        ],
    )
    s = summarize_session_dir(rd)
    assert (s.status, s.reason) == ("waiting", "needs answer")
    # Once answered, the run is running again (the approver appends the answer).
    with (rd / "logs.jsonl").open("a", encoding="utf-8") as fh:
        fh.write('{"type": "approval.answer", "id": "a1", "approved": true}\n')
    assert summarize_session_dir(rd).status == "running"


def test_summary_dead_worker_reads_stale_at_once(tmp_path: Path) -> None:
    # A killed run (worker.pid points at a dead process, no session.end) must not
    # read "running" for the whole silence window; the pid probe settles it now.
    rd = _write_run(tmp_path, "runs", "r3", [{"type": "session.start", "mode": "run"}])
    (rd / "worker.pid").write_text("999999999", encoding="utf-8")  # beyond pid_max: never alive
    assert summarize_session_dir(rd).status == "stale"


def test_summary_live_worker_with_a_silent_log_stays_running(tmp_path: Path) -> None:
    # The converse: a live worker blocked in a long provider call emits no
    # events for minutes, and must not read stale for it.
    import os

    rd = _write_run(tmp_path, "runs", "r4", [{"type": "session.start", "mode": "run"}])
    (rd / "worker.pid").write_text(str(os.getpid()), encoding="utf-8")
    assert summarize_session_dir(rd).status == "running"


def test_summary_carries_the_partial_cost_marker(tmp_path: Path) -> None:
    """LogScan's sticky usd_partial must reach SessionSummary: listings printed an
    exact $0.0123 while the run page printed ~$0.0123 for the same run."""
    rd = _write_run(
        tmp_path,
        "runs",
        "r1",
        [
            {"type": "session.start", "mode": "run", "user_task": "t"},
            {"type": "budget.update", "usd_total": 0.0123, "usd_partial": True},
            {"type": "session.end", "all_passed": True, "reason": "finish_session"},
        ],
    )
    assert summarize_session_dir(rd).usd_partial is True
    clean = _write_run(
        tmp_path,
        "runs",
        "r2",
        [
            {"type": "session.start", "mode": "run", "user_task": "t"},
            {"type": "budget.update", "usd_total": 0.0123},
            {"type": "session.end", "all_passed": True, "reason": "finish_session"},
        ],
    )
    assert summarize_session_dir(clean).usd_partial is False


def test_run_is_live_finished_run_with_lingering_pid_is_not_live(tmp_path: Path) -> None:
    """A finished run whose worker.pid survives into teardown is NOT live: the
    loop has exited, so a steer/compact/answer marker written now is read by
    nobody. session_is_live must fold the log facts; fed empty facts it degenerates
    to worker_is_alive under a new name (the exact question it exists to
    replace) and called this run "starting"."""
    rd = _write_run(
        tmp_path,
        "runs",
        "r1",
        [
            {"type": "session.start", "mode": "run", "user_task": "t"},
            {"type": "session.end", "all_passed": True, "reason": "finish_session"},
        ],
    )
    (rd / "worker.pid").write_text(str(os.getpid()), encoding="utf-8")
    assert session_is_live(rd) is False


def test_run_is_live_waiting_on_an_answer_is_live(tmp_path: Path) -> None:
    # Blocked on an unanswered approval with a live worker: the answer WILL be
    # read, so the prompt buttons and the composer stay live.
    rd = _write_run(
        tmp_path,
        "runs",
        "r1",
        [
            {"type": "session.start", "mode": "run", "user_task": "t"},
            {"type": "approval.prompt", "id": "a1", "prompt": "Allow run_command: pytest"},
        ],
    )
    (rd / "worker.pid").write_text(str(os.getpid()), encoding="utf-8")
    assert session_is_live(rd) is True


def test_run_is_live_dead_worker_is_not_live(tmp_path: Path) -> None:
    rd = _write_run(tmp_path, "runs", "r1", [{"type": "session.start", "mode": "run"}])
    (rd / "worker.pid").write_text("999999999", encoding="utf-8")  # beyond pid_max
    assert session_is_live(rd) is False


def test_run_is_live_unstarted_dirs(tmp_path: Path) -> None:
    # A parked submission or fork --no-run dir: nothing polls markers -> not
    # live (resume is the offer). A launching worker (pid, no events yet) is.
    rd = tmp_path / "sessions" / "runs" / "parked"
    rd.mkdir(parents=True)
    (rd / "manifest.json").write_text(
        json.dumps({"version": 2, "parked_task": "queued work"}), encoding="utf-8"
    )
    assert session_is_live(rd) is False
    live = _write_run(tmp_path, "runs", "launching", [])
    (live / "worker.pid").write_text(str(os.getpid()), encoding="utf-8")
    assert session_is_live(live) is True


def test_summary_ask_task_comes_from_transcript(tmp_path: Path) -> None:
    rd = _write_run(
        tmp_path,
        "asks",
        "a1",
        [
            {"type": "session.start", "mode": "ask", "user_task": '<file path="a.py">\nx'},
            {"type": "session.end", "all_passed": True},
        ],
    )
    (rd / "transcript.md").write_text(
        "# agent6 ask\n\n## Question\n\nwhat is the default port?\n", encoding="utf-8"
    )
    s = summarize_session_dir(rd)
    assert task_snippet(s.task) == "what is the default port?"


def test_summary_no_logs(tmp_path: Path) -> None:
    rd = tmp_path / "sessions" / "runs" / "empty"
    rd.mkdir(parents=True)
    s = summarize_session_dir(rd)
    assert (s.status, s.task) == ("created", "(no logs)")


def test_summary_plan_reads_planned_not_passed(tmp_path: Path) -> None:
    # A plan pass ends via finish_planning (its only clean exit) with
    # all_passed=True; it gates nothing, so it must read "planned", not "passed".
    rd = _write_run(
        tmp_path,
        "runs",
        "p1",
        [
            {"type": "session.start", "mode": "plan", "user_task": "plan the refactor"},
            {"type": "session.end", "all_passed": True, "reason": "finish_planning"},
        ],
    )
    s = summarize_session_dir(rd)
    assert (s.mode, s.status, s.reason) == ("plan", "planned", "")
    # A real run still reads "passed" (finish_session + all_passed) -- unchanged.
    rd2 = _write_run(
        tmp_path,
        "runs",
        "r1",
        [
            {"type": "session.start", "mode": "run", "user_task": "t"},
            {"type": "session.end", "all_passed": True, "reason": "finish_session"},
        ],
    )
    assert summarize_session_dir(rd2).status == "passed"


def test_summary_manifest_only_fork_shows_mode_and_task(tmp_path: Path) -> None:
    # A `fork --no-run` fork has a manifest (mode + task) but no logs yet; the
    # listing must show them, not a blank "? ? (no logs)".
    rd = tmp_path / "sessions" / "runs" / "child"
    rd.mkdir(parents=True)
    (rd / "manifest.json").write_text(
        json.dumps({"mode": "plan", "user_task": "carry this forward"}), encoding="utf-8"
    )
    s = summarize_session_dir(rd)
    assert (s.mode, s.task, s.status) == ("plan", "carry this forward", "created")


def test_summary_launching_run_reads_starting(tmp_path: Path) -> None:
    # A run with no verify_command spends ~80s inferring one BEFORE session.start.
    # During it the log has a role.call (the inference LLM call) but no session.start,
    # and the worker is alive -- it must read "starting" (its real mode+task from
    # the manifest), not a blank "? / (no task) / running" that looks missing.
    rd = _write_run(tmp_path, "runs", "boot", [{"type": "role.call", "role": "verify_inferer"}])
    (rd / "manifest.json").write_text(
        json.dumps({"mode": "run", "user_task": "refactor the loop"}), encoding="utf-8"
    )
    (rd / "worker.pid").write_text(str(os.getpid()), encoding="utf-8")  # a live worker
    s = summarize_session_dir(rd)
    assert (s.mode, s.task, s.status) == ("run", "refactor the loop", "starting")


def test_summary_pre_start_dead_worker_says_it_died_launching(tmp_path: Path) -> None:
    """A worker killed during preflight leaves its pid file (a clean refusal
    clears it) and preflight events with real spend. Reading it as "created" --
    the fork --no-run word -- hid the death; a bare "stale" would overclaim
    ("was running, crashed"), so the word carries its own reason. A dir with NO
    pid file ever (fork --no-run) stays "created"."""
    rd = _write_run(tmp_path, "runs", "dead", [{"type": "role.call", "role": "verify_inferer"}])
    (rd / "manifest.json").write_text(
        json.dumps({"mode": "run", "user_task": "t"}), encoding="utf-8"
    )
    (rd / "worker.pid").write_text("999999999", encoding="utf-8")  # never alive
    s = summarize_session_dir(rd)
    assert (s.status, s.reason) == ("stale", "died launching")

    never_launched = _write_run(tmp_path, "runs", "husk", [])
    (never_launched / "worker.pid").unlink(missing_ok=True)
    assert summarize_session_dir(never_launched).status == "created"


def test_a_forks_single_leg_is_one_leg(tmp_path: Path) -> None:
    """A fork's log OPENS with loop.resume.start (resume() drives it; no
    session.start ever lands), and the unconditional leg increment counted its
    single leg as two: `sessions show` labelled its cost "(all 2 legs)" and its
    tokens "(latest leg)". The first leg-start of any kind begins leg 1."""
    from agent6.viewmodel.listing import scan_session_log

    rd = _write_run(
        tmp_path,
        "runs",
        "fork-1",
        [
            {"type": "loop.resume.start", "iteration": 1},
            {"type": "budget.update", "usd_total": 0.05},
            {"type": "session.end", "all_passed": True, "reason": "finish_session"},
        ],
    )
    scan = scan_session_log(rd / "logs.jsonl")
    assert scan.legs == 1
    assert scan.cost_usd == 0.05

    # A real second leg still counts (and banks the first leg's spend).
    rd2 = _write_run(
        tmp_path,
        "runs",
        "fork-2",
        [
            {"type": "loop.resume.start", "iteration": 1},
            {"type": "budget.update", "usd_total": 0.05},
            {"type": "loop.resume.start", "iteration": 5},
            {"type": "budget.update", "usd_total": 0.01},
        ],
    )
    scan2 = scan_session_log(rd2 / "logs.jsonl")
    assert scan2.legs == 2
    assert scan2.cost_usd == pytest.approx(0.06)


def test_summary_cost_sums_across_resume_legs(tmp_path: Path) -> None:
    # Each resume leg starts a fresh budget (usd_total resets to 0). The listing
    # total must be the cumulative spend across legs, not just the latest leg's.
    rd = _write_run(
        tmp_path,
        "runs",
        "r1",
        [
            {"type": "session.start", "mode": "run", "user_task": "t"},
            {"type": "budget.update", "usd_total": 0.01},
            {"type": "budget.update", "usd_total": 0.02},  # leg 1 ends at $0.02
            {"type": "session.end", "all_passed": False, "reason": "budget_exhausted"},
            {"type": "loop.resume.start", "iteration": 3},
            {"type": "budget.update", "usd_total": 0.003},
            {"type": "budget.update", "usd_total": 0.007},  # leg 2 ends at $0.007
            {"type": "session.end", "all_passed": True, "reason": "finish_session"},
        ],
    )
    s = summarize_session_dir(rd)
    assert abs(s.cost_usd - 0.027) < 1e-9  # 0.02 (leg 1) + 0.007 (leg 2), not 0.007


def test_is_run_husk(tmp_path: Path) -> None:
    # Neither manifest nor logs: never started, a husk.
    husk = tmp_path / "husk"
    husk.mkdir()
    assert is_session_husk(husk)
    # Either file makes it a real run.
    with_logs = tmp_path / "with-logs"
    with_logs.mkdir()
    (with_logs / "logs.jsonl").write_text("", encoding="utf-8")
    assert not is_session_husk(with_logs)
    with_manifest = tmp_path / "with-manifest"
    with_manifest.mkdir()
    (with_manifest / "manifest.json").write_text("{}", encoding="utf-8")
    assert not is_session_husk(with_manifest)
    # A dir with neither file but a LIVE worker.pid is a launching run in its
    # pre-manifest preflight window, not a husk -- keep it listed (as "starting").
    launching = tmp_path / "launching"
    launching.mkdir()
    (launching / "worker.pid").write_text(str(os.getpid()), encoding="utf-8")
    assert not is_session_husk(launching)
    # ... but a dead worker.pid with no files is still a husk.
    dead = tmp_path / "dead-husk"
    dead.mkdir()
    (dead / "worker.pid").write_text("999999999", encoding="utf-8")
    assert is_session_husk(dead)


def test_summary_survives_a_valid_json_non_object_line(tmp_path: Path) -> None:
    # A valid-JSON line that isn't an object (a torn or adversarial writer) must
    # not crash the listing fold -- one bad line otherwise took down the whole
    # hub / `sessions list` / TUI home. It's skipped like an unparseable line.
    rd = tmp_path / "sessions" / "runs" / "weird"
    rd.mkdir(parents=True)
    (rd / "logs.jsonl").write_text(
        json.dumps({"type": "session.start", "user_task": "do a thing"})
        + "\n"
        + "[1, 2, 3]\n"  # valid JSON, not a dict
        + '"a bare string"\n'
        + json.dumps({"type": "session.end", "all_passed": True, "reason": "finish_session"})
        + "\n",
        encoding="utf-8",
    )
    s = summarize_session_dir(rd)  # must not raise
    assert s.task == "do a thing"
    assert s.status == "passed"


def test_summary_survives_a_malformed_usd_total(tmp_path: Path) -> None:
    # budget.update is agent6-written, but a torn write or hand-edited log can
    # leave usd_total non-numeric; the scan keeps the last good figure instead
    # of aborting the whole listing (same degradation the typed fold applies).
    # Falsy junk ('', False) counts: an `or 0.0` fallback silently reset it.
    rd = tmp_path / "sessions" / "runs" / "torn-usd"
    rd.mkdir(parents=True)
    (rd / "logs.jsonl").write_text(
        json.dumps({"type": "session.start", "user_task": "t"})
        + "\n"
        + json.dumps({"type": "budget.update", "usd_total": 0.25})
        + "\n"
        + json.dumps({"type": "budget.update", "usd_total": "garbage"})
        + "\n"
        + json.dumps({"type": "budget.update", "usd_total": [1, 2]})
        + "\n"
        + json.dumps({"type": "budget.update", "usd_total": ""})
        + "\n"
        + json.dumps({"type": "budget.update", "usd_total": False})
        + "\n"
        + json.dumps({"type": "session.end", "all_passed": True, "reason": "finish_session"})
        + "\n",
        encoding="utf-8",
    )
    s = summarize_session_dir(rd)  # must not raise
    assert s.status == "passed"
    assert s.cost_usd == 0.25  # the last good figure, not 0 and not a crash


def test_summary_gateless_settle_reads_finished_unverified(tmp_path: Path) -> None:
    # A gateless run's quiet finish committed real work but nothing verified
    # it: "finished · unverified", deliberately neither green nor "failed".
    # ("unverified", not "no verify": a command may exist via mid-run adoption
    # and simply never have passed.)
    rd = _write_run(
        tmp_path,
        "runs",
        "g1",
        [
            {"type": "session.start", "mode": "run", "user_task": "build it"},
            {"type": "session.end", "all_passed": False, "reason": "settled"},
        ],
    )
    s = summarize_session_dir(rd)
    assert (s.status, s.reason) == ("finished", "unverified")


def test_summary_second_run_start_reads_running(tmp_path: Path) -> None:
    """An ask REPL follow-up re-runs on the same log via a plain session.start; the
    hub row must read "running" while the follow-up leg streams, not the prior
    leg's "answered"."""
    rd = _write_run(
        tmp_path,
        "asks",
        "ask-repl",
        [
            {"type": "session.start", "mode": "ask", "user_task": "q"},
            {"type": "session.end", "all_passed": True, "reason": "answered"},
            {"type": "session.start", "mode": "ask", "user_task": "q2"},
            {"type": "role.call", "role": "worker", "model": "m"},
        ],
    )
    assert summarize_session_dir(rd).status == "running"


def test_newest_run_dir_skips_husks_that_no_listing_shows(tmp_path: Path) -> None:
    """A husk (a dir a crash orphaned before any manifest or log) is hidden by
    every listing, but the recency query returned it, so a bare `attach` /
    `sessions show` / `sessions stop` targeted a phantom the operator cannot see -- and
    could miss a live run whose log was quiet during a long provider call."""
    from agent6.viewmodel.listing import newest_session_dir

    bucket = tmp_path / "sessions" / "runs"
    bucket.mkdir(parents=True)
    real = bucket / "real-run-0001"
    real.mkdir()
    (real / "logs.jsonl").write_text(
        json.dumps({"type": "session.start", "mode": "run", "user_task": "t"}) + "\n",
        encoding="utf-8",
    )
    os.utime(real, (time.time() - 7200, time.time() - 7200))
    os.utime(real / "logs.jsonl", (time.time() - 7200, time.time() - 7200))
    husk = bucket / "zz-husk-0002"  # newer, but nothing ever ran
    husk.mkdir()

    assert newest_session_dir([bucket]) == real


def test_summary_forked_leg_reads_mode_and_task_from_manifest(tmp_path: Path) -> None:
    """A fork/resumed leg's log holds only loop.resume.start, which sets
    saw_start=True but records no mode/task (only session.start carries them). Gating
    the manifest fallback on saw_start therefore blanked the row to "? (no logs)";
    gate on the missing mode instead so the row shows the run's real work."""
    rd = _write_run(tmp_path, "runs", "forked-0001", [{"type": "loop.resume.start"}])
    (rd / "manifest.json").write_text(
        json.dumps(
            {"version": 2, "session_id": "forked-0001", "mode": "run", "user_task": "carry on"}
        ),
        encoding="utf-8",
    )
    s = summarize_session_dir(rd)
    assert (s.mode, s.task) == ("run", "carry on")


def test_scan_counts_a_non_string_prompt_id_as_blocking(tmp_path: Path) -> None:
    """The answer side discards str(id), but the prompt side only registered
    string ids -- so an int id (events.py coerces ids to str) left a run blocked
    on the operator reading as plain "running". Coerce on the prompt side too."""
    from agent6.viewmodel.listing import scan_session_log

    log = tmp_path / "logs.jsonl"
    log.write_text(
        json.dumps({"type": "session.start", "mode": "run", "user_task": "t"})
        + "\n"
        + json.dumps({"type": "approval.prompt", "id": 7})
        + "\n",
        encoding="utf-8",
    )
    assert scan_session_log(log).operator_blocked  # the int id still registers as unanswered

    log.write_text(
        log.read_text(encoding="utf-8") + json.dumps({"type": "approval.answer", "id": 7}) + "\n",
        encoding="utf-8",
    )
    assert not scan_session_log(log).operator_blocked  # answered by the same int id


def test_a_crashed_run_reads_dead_at_once(tmp_path: Path) -> None:
    """A run whose loop escaped with a fault records session.end reason=crashed, so
    every surface calls it failed immediately. Without that record the dying
    process still cleared worker.pid -- the only immediate liveness evidence --
    and the fold fell back to the silence window, so `sessions list`, `sessions show`,
    attach, the web hub and the TUI all showed a dead run as "running" for ten
    minutes. (A SIGKILLed run leaves its pid file, which is why that case
    always read stale at once.)"""
    session_dir = tmp_path / "sessions" / "runs" / "gone"
    session_dir.mkdir(parents=True)
    (session_dir / "logs.jsonl").write_text(
        json.dumps({"type": "session.start", "mode": "run", "user_task": "t"})
        + "\n"
        + json.dumps({"type": "session.end", "reason": "crashed", "all_passed": False})
        + "\n",
        encoding="utf-8",
    )
    summary = summarize_session_dir(session_dir)
    assert (summary.status, summary.reason) == ("failed", "crashed")
    assert session_is_live(session_dir) is False


def test_summary_ungated_end_reads_finished_and_absent_key_reads_as_before(
    tmp_path: Path,
) -> None:
    """session.end's all_passed is a tri-state on the wire: an explicit null
    (ungated: no verify command) words as "finished" whatever the reason -- a
    gateless silent finish listed as "passed" for a tree nothing verified. An
    ABSENT key (a pre-tri-state log) still reads False, so old failure ends
    keep their word."""
    ungated = _write_run(
        tmp_path,
        "runs",
        "r-ungated",
        [
            {"type": "session.start", "mode": "run", "user_task": "t"},
            {"type": "session.end", "reason": "silent_finish", "all_passed": None},
        ],
    )
    assert (summarize_session_dir(ungated).status, summarize_session_dir(ungated).reason) == (
        "finished",
        "",
    )
    legacy = _write_run(
        tmp_path,
        "runs",
        "r-legacy",
        [
            {"type": "session.start", "mode": "run", "user_task": "t"},
            {"type": "session.end", "reason": "went_quiet"},
        ],
    )
    assert summarize_session_dir(legacy).status == "failed"
