# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""`agent6 resume` preflight ordering: the snapshot-version refusal must land
BEFORE the egress broker is spawned (like `fork`, which refuses instantly), so a
v1-snapshot resume never spawns a broker + netns or prints the egress preamble.
"""

from __future__ import annotations

import json
import subprocess as sp
from pathlib import Path

import pytest

import agent6.app._session as session_mod
import agent6.app._setup as setup_mod
import agent6.app.resume as resume_mod
from agent6.paths import state_dir
from agent6.sessions.layout import SessionLayout
from agent6.ui.cli.resume import _cmd_resume  # pyright: ignore[reportPrivateUsage]
from agent6.workflows._session_state import SNAPSHOT_VERSION


def _git_repo(path: Path) -> None:
    sp.run(["git", "init", "-q", "-b", "main"], cwd=path, check=True)
    sp.run(["git", "config", "user.email", "t@example.com"], cwd=path, check=True)
    sp.run(["git", "config", "user.name", "t"], cwd=path, check=True)
    (path / "seed.txt").write_text("seed\n")
    sp.run(["git", "add", "seed.txt"], cwd=path, check=True)
    sp.run(["git", "commit", "-q", "-m", "init"], cwd=path, check=True)


def test_parked_resume_does_not_replay_a_config_selected_profile_as_a_flag(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The parked branch is the SECOND preset replay site: it handed the raw
    stamped name to load_effective, where _select_preset treats it as a flag
    that outranks every config layer -- so a parked submission under a
    config-selected preset started under a config its original submission
    never had. The snapshot-resume path already replays via replay_preset."""
    repo = tmp_path / "repo"
    repo.mkdir()
    _git_repo(repo)
    monkeypatch.chdir(repo)
    session_dir = state_dir(repo) / "sessions" / "runs" / "parked-AAAA11"
    session_dir.mkdir(parents=True)
    (session_dir / "manifest.json").write_text(
        json.dumps(
            {
                "version": 2,
                "session_id": "parked-AAAA11",
                "mode": "run",
                "user_task": "queued work",
                "parked_task": "queued work",
                "workflow": {"preset": "t", "preset_from_flag": False},
            }
        ),
        encoding="utf-8",
    )
    seen: list[str] = []

    def _capture_load_effective(*_a: object, preset: str = "", **_k: object) -> object:
        from agent6.config import ConfigError

        seen.append(preset)
        raise ConfigError("stop before run_task")  # short-circuit the branch

    monkeypatch.setattr(setup_mod, "load_effective", _capture_load_effective)
    rc = _cmd_resume(None, "parked-AAAA11", force=False)
    assert rc == 2
    # A config-selected preset re-resolves from the config files; only a
    # --preset flag is replayed (WorkflowStamp.replay_preset's contract).
    assert seen == [""]


def test_resume_refuses_a_malformed_steer_directive_before_any_leg(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """`resume --steer "/pin"` (every front-end's continue lands here) is
    refused before a session is even resolved: a leg spent on a directive
    the loop declines ends as a silent finish and flips a passed run to
    failed."""
    monkeypatch.chdir(tmp_path)
    assert _cmd_resume(None, "any-run-AAAAAA", force=False, steer="/pin") == 2
    assert "pin needs an instruction" in capsys.readouterr().err


def _park_manifest(session_dir: Path, *, preset: str, from_flag: bool) -> None:
    session_dir.mkdir(parents=True)
    (session_dir / "manifest.json").write_text(
        json.dumps(
            {
                "version": 2,
                "session_id": session_dir.name,
                "mode": "run",
                "user_task": "queued work",
                "parked_task": "queued work",
                "workflow": {"preset": preset, "preset_from_flag": from_flag},
            }
        ),
        encoding="utf-8",
    )


def _stub_start_of_run(
    resume: object, monkeypatch: pytest.MonkeyPatch, tmp: Path
) -> dict[str, object]:
    """Let a parked resume reach `run_task`; capture the kwargs it hands over."""
    _stub_load_effective(monkeypatch, _PLANNER_AND_WORKER, tmp)
    captured: dict[str, object] = {}

    def _capture_run_task(*_a: object, **k: object) -> int:
        captured.update(k)
        return 0

    monkeypatch.setattr(resume, "run_task", _capture_run_task)
    return captured


def test_parked_resume_carries_the_original_flag_selected_profile_stamp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A parked leg never ran, but its manifest recorded a FLAG-selected preset.
    Restarting it must re-stamp the SAME (name, from_flag) so a later resume/fork
    replays the flag precedence; deriving the stamp from the (empty) resume
    `preset` dropped the from_flag bit and silently downgraded a flag-selected
    preset's blocking veto on the next leg."""
    repo = tmp_path / "repo"
    repo.mkdir()
    _git_repo(repo)
    monkeypatch.chdir(repo)
    _park_manifest(
        state_dir(repo) / "sessions" / "runs" / "parked-BBBB22",
        preset="strict",
        from_flag=True,
    )
    captured = _stub_start_of_run(resume_mod, monkeypatch, tmp_path)

    assert _cmd_resume(None, "parked-BBBB22", force=False) == 0
    assert captured["preset_stamp"] == ("strict", True)


def test_parked_resume_with_its_own_profile_flag_lets_run_task_derive_the_stamp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A resume that DOES pass --preset is a fresh flag choice for this leg, so
    it must NOT pin the manifest's old stamp -- run_task derives from `preset`."""
    repo = tmp_path / "repo"
    repo.mkdir()
    _git_repo(repo)
    monkeypatch.chdir(repo)
    _park_manifest(
        state_dir(repo) / "sessions" / "runs" / "parked-CCCC33",
        preset="strict",
        from_flag=True,
    )
    captured = _stub_start_of_run(resume_mod, monkeypatch, tmp_path)

    assert _cmd_resume(None, "parked-CCCC33", force=False, preset="none") == 0
    assert captured["preset_stamp"] is None
    assert captured["preset"] == "none"


def test_parked_resume_of_a_config_selected_profile_re_derives_the_stamp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A CONFIG-selected preset (from_flag False) re-resolves from the CURRENT
    config on restart, so pinning the manifest's OLD name would show a stale
    preset if the config changed since. Pass preset_stamp=None so run_task
    derives from the re-resolved cfg, like a fresh run -- only a FLAG-selected
    preset (whose blocking veto must survive) is pinned."""
    repo = tmp_path / "repo"
    repo.mkdir()
    _git_repo(repo)
    monkeypatch.chdir(repo)
    _park_manifest(
        state_dir(repo) / "sessions" / "runs" / "parked-DDDD44",
        preset="hardened",
        from_flag=False,
    )
    captured = _stub_start_of_run(resume_mod, monkeypatch, tmp_path)

    assert _cmd_resume(None, "parked-DDDD44", force=False) == 0
    assert captured["preset_stamp"] is None  # re-derives, not the stale manifest name


class _Stop(Exception):
    """Sentinel: the resume path reached the seam past the assertion point."""


def _plan_session_dir(repo: Path, session_id: str) -> None:
    session_dir = state_dir(repo) / "sessions" / "runs" / session_id
    session_dir.mkdir(parents=True)
    (session_dir / "manifest.json").write_text(
        json.dumps({"version": 2, "session_id": session_id, "mode": "plan", "user_task": "t"}),
        encoding="utf-8",
    )
    (session_dir / "loop_state.json").write_text(
        json.dumps(
            {
                "version": SNAPSHOT_VERSION,
                "system": "s",
                "messages": [],
                "tool_calls": 0,
                "next_iteration": 1,
                "root_task_id": None,
                "original_task": "t",
                "verify_command": [],
            }
        ),
        encoding="utf-8",
    )


_PROVIDER_TOML = """
[agent6]
config_version = 1

[providers.anthropic]
api_format = "anthropic"
api_key_env = "ANTHROPIC_API_KEY"

[models.reviewer]
provider = "anthropic"
model = "planner-model"
"""

_PLANNER_ONLY = (
    _PROVIDER_TOML
    + """
[models.planner]
provider = "anthropic"
model = "planner-model"
"""
)

_PLANNER_AND_WORKER = (
    _PLANNER_ONLY
    + """
[models.worker]
provider = "anthropic"
model = "worker-model"
"""
)


def _stub_load_effective(monkeypatch: pytest.MonkeyPatch, toml_body: str, tmp: Path) -> None:
    from agent6.config import load_config
    from agent6.config.layer import EffectiveConfig

    cfg_path = tmp / "cfg.toml"
    cfg_path.write_text(toml_body, encoding="utf-8")
    cfg = load_config(cfg_path)

    # The real type: preflight reads `explicit_leaves` off it to tell a DEFAULT
    # this host cannot honour (degrade) from a value the operator wrote down
    # (refuse), and a stand-in cannot answer for that.
    def _load(*_a: object, **_k: object) -> EffectiveConfig:
        return EffectiveConfig(config=cfg, sources={}, layers=())

    monkeypatch.setattr(setup_mod, "load_effective", _load)


def test_plan_resume_requires_the_planner_role(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A plan run resumes under the planner role. Resume hard-coded "worker" at
    its readiness gate, so a planner-only config could START a plan (fresh
    preflight passes require_runnable("planner")) but never resume it."""
    repo = tmp_path / "repo"
    repo.mkdir()
    _git_repo(repo)
    monkeypatch.chdir(repo)
    _plan_session_dir(repo, "plan-AAAA11")
    _stub_load_effective(monkeypatch, _PLANNER_ONLY, tmp_path)

    def _stop(*_a: object, **_k: object) -> object:
        raise _Stop()

    # Reaching detect_env means the readiness gate accepted the planner-only
    # config; the old hard-coded require_runnable("worker") returned 2 first.
    monkeypatch.setattr(session_mod, "detect_env", _stop)
    with pytest.raises(_Stop):
        _cmd_resume(None, "plan-AAAA11", force=False)


def test_resume_preset_flag_is_recorded_for_later_legs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`resume --preset X` continues the run under X and stamps it as the run's
    flag-selected preset, so a later plain resume replays X and every listing
    names it; without the flag the stamp is untouched."""
    from agent6.sessions.manifest import read_manifest

    repo = tmp_path / "repo"
    repo.mkdir()
    _git_repo(repo)
    monkeypatch.chdir(repo)
    _plan_session_dir(repo, "plan-PRESET1")
    _stub_load_effective(monkeypatch, _PLANNER_ONLY, tmp_path)
    session_dir = state_dir(repo) / "sessions" / "runs" / "plan-PRESET1"

    def _stop(*_a: object, **_k: object) -> object:
        raise _Stop()

    monkeypatch.setattr(session_mod, "detect_env", _stop)
    with pytest.raises(_Stop):
        _cmd_resume(None, "plan-PRESET1", force=False, preset="quick")
    stamp = read_manifest(session_dir).workflow
    assert (stamp.preset, stamp.preset_from_flag, stamp.replay_preset) == ("quick", True, "quick")
    with pytest.raises(_Stop):
        _cmd_resume(None, "plan-PRESET1", force=False)
    assert read_manifest(session_dir).workflow.replay_preset == "quick"


def test_resume_writes_its_worker_pid_only_after_the_preflight_passed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A hub's detached resume reads the run's worker.pid as the child owning
    the run (spawn_and_confirm) and `sessions show` reads it as a live worker:
    written before the preflight, every refusal past that point (the checkout
    lock, a missing snapshot, the git guards, config, isolation) still read
    "resuming" from the hub and "alive" from the listing."""
    from agent6.app._leg import LegEnd
    from agent6.app.preflight import SessionRefused

    repo = tmp_path / "repo"
    repo.mkdir()
    _git_repo(repo)
    monkeypatch.chdir(repo)
    _plan_session_dir(repo, "plan-PIDORDER")
    _stub_load_effective(monkeypatch, _PLANNER_ONLY, tmp_path)
    monkeypatch.setenv("AGENT6_DETACHED_AWAY", "deny")  # run_commands="ask" with no tty refuses
    session_dir = state_dir(repo) / "sessions" / "runs" / "plan-PIDORDER"
    order: list[str] = []
    real_write = resume_mod.write_worker_pid

    def _write(session_dir: Path, pid: int) -> None:
        order.append("pid")
        real_write(session_dir, pid)

    monkeypatch.setattr(resume_mod, "write_worker_pid", _write)

    def _refuse(*_a: object, **_k: object) -> str:
        order.append("isolation")
        raise SessionRefused(2)

    monkeypatch.setattr(resume_mod, "select_isolation", _refuse)
    assert _cmd_resume(None, "plan-PIDORDER", force=False) == 2
    assert order == ["isolation"]
    assert not (session_dir / "worker.pid").exists()

    def _select(*_a: object, **_k: object) -> str:
        order.append("isolation")
        return "none"

    def _none(*_a: object, **_k: object) -> None:
        return None

    def _leg(*_a: object, **_k: object) -> LegEnd:
        order.append("leg")
        assert (session_dir / "worker.pid").is_file()  # owned before the leg runs
        return LegEnd(rc=0)

    order.clear()
    monkeypatch.setattr(resume_mod, "select_isolation", _select)
    monkeypatch.setattr(resume_mod, "check_provider_keys", _none)  # no key in a unit test
    monkeypatch.setattr(resume_mod, "run_leg", _leg)
    assert _cmd_resume(None, "plan-PIDORDER", force=False) == 0
    assert order == ["isolation", "pid", "leg"]


def _unconfined(*_a: object, **_k: object) -> str:
    return "none"


def _nothing(*_a: object, **_k: object) -> None:
    return None


def _finished_leg(*_a: object, **_k: object) -> object:
    from agent6.app._leg import LegEnd

    return LegEnd(0)


def test_a_parked_resumes_detach_leaves_the_pid_with_the_spawned_child(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A parked submission resumed by `agent6 resume` runs through run_task,
    whose teardown keeps worker.pid through a detach (the spawned child then
    holds the file). resume_task's own teardown then cleared it: every listing
    read the live child as stale until its loop wrote the pid again."""
    import subprocess
    from unittest.mock import MagicMock

    import agent6.app.run as run_mod
    from agent6.app._leg import LegEnd
    from agent6.sessions.ipc import read_worker_pid, write_worker_pid

    repo = tmp_path / "repo"
    repo.mkdir()
    _git_repo(repo)
    monkeypatch.chdir(repo)
    monkeypatch.delenv("AGENT6_DETACHED_AWAY", raising=False)
    session_dir = state_dir(repo) / "sessions" / "runs" / "parked-DETACH"
    _park_manifest(session_dir, preset="", from_flag=False)
    _stub_load_effective(monkeypatch, _PLANNER_AND_WORKER, tmp_path)
    child = subprocess.Popen(["sleep", "60"])
    try:

        def _leg(*_a: object, events: object, **_k: object) -> LegEnd:
            events.emit("session.start", session_id="parked-DETACH", mode="run", user_task="t")  # type: ignore[attr-defined]
            return LegEnd(0, detach_requested=True)

        monkeypatch.setattr(run_mod, "run_leg", _leg)
        monkeypatch.setattr(run_mod, "select_isolation", _unconfined)
        frontend = MagicMock()

        def _spawn(_cwd: Path, _sid: str, _flags: object) -> str:
            write_worker_pid(session_dir, child.pid)  # the child claimed the run
            return ""

        frontend.spawn_detached_resume.side_effect = _spawn
        assert resume_mod.resume_task(None, "parked-DETACH", frontend=frontend, force=False) == 0
        assert read_worker_pid(session_dir) == child.pid
    finally:
        child.kill()
        child.wait()


def test_the_resume_note_leaves_the_untracked_at_start_files_out(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """ "the tree holds changes no commit has ...; this leg's next commit takes
    them" named the operator's files untracked when the run started, which
    every chain commit leaves out."""
    from unittest.mock import MagicMock

    from agent6.sessions.layout import write_untracked_at_start

    repo = tmp_path / "repo"
    repo.mkdir()
    _git_repo(repo)
    monkeypatch.chdir(repo)
    monkeypatch.delenv("AGENT6_DETACHED_AWAY", raising=False)
    base = sp.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True, check=True
    ).stdout.strip()
    session_dir = state_dir(repo) / "sessions" / "runs" / "note-UNTRACKED"
    session_dir.mkdir(parents=True)
    (session_dir / "manifest.json").write_text(
        json.dumps(
            {
                "version": 3,
                "session_id": "note-UNTRACKED",
                "mode": "run",
                "user_task": "t",
                "base_sha": base,
                "base_branch": "main",
                "run_branch": "agent6/note-UNTRACKED",
            }
        ),
        encoding="utf-8",
    )
    (session_dir / "loop_state.json").write_text(
        json.dumps(
            {
                "version": SNAPSHOT_VERSION,
                "system": "s",
                "messages": [],
                "tool_calls": 0,
                "next_iteration": 1,
                "root_task_id": None,
                "original_task": "t",
                "verify_command": [],
            }
        ),
        encoding="utf-8",
    )
    write_untracked_at_start(session_dir, {"notes.md"})
    (repo / "notes.md").write_text("the operator's, since before the run\n", encoding="utf-8")
    (repo / "seed.txt").write_text("edited between legs\n", encoding="utf-8")
    _stub_load_effective(monkeypatch, _PLANNER_AND_WORKER, tmp_path)
    monkeypatch.setattr(resume_mod, "select_isolation", _unconfined)
    monkeypatch.setattr(resume_mod, "check_provider_keys", _nothing)
    monkeypatch.setattr(resume_mod, "run_leg", _finished_leg)
    assert resume_mod.resume_task(None, "note-UNTRACKED", frontend=MagicMock(), force=False) == 0
    notes = [line for line in capsys.readouterr().err.splitlines() if "no commit has" in line]
    assert notes == [
        "[agent6] the tree holds changes no commit has (seed.txt);"
        " this leg's next commit takes them"
    ]


def test_plan_resume_builds_the_planner_provider(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The resumed leg's DRIVING provider is the planner route: with both roles
    configured, the old path silently switched a plan run to the worker model
    on its second leg (and stamped the transcript seat "worker")."""
    import dataclasses

    import agent6.ui.cli.resume as cli_resume_mod
    from agent6.ui.cli.run import session_frontend

    repo = tmp_path / "repo"
    repo.mkdir()
    _git_repo(repo)
    monkeypatch.chdir(repo)
    _plan_session_dir(repo, "plan-BBBB22")
    _stub_load_effective(monkeypatch, _PLANNER_AND_WORKER, tmp_path)
    # The default run_commands="ask" with no tty now REFUSES rather than
    # hanging; this test is about which provider drives the leg.
    monkeypatch.setenv("AGENT6_DETACHED_AWAY", "deny")

    def _yes(*_a: object) -> bool:
        return True

    def _frontend(_cp: object = None) -> object:
        return dataclasses.replace(session_frontend(), confirm_unconfined_autorun=_yes)

    def _none(*_a: object, **_k: object) -> None:
        return None

    def _strict(*_a: object, **_k: object) -> str:
        return "strict"

    monkeypatch.setattr(cli_resume_mod, "session_frontend", _frontend)
    monkeypatch.setattr(session_mod, "detect_env", object)
    monkeypatch.setattr(session_mod, "resolve_isolation", _strict)
    monkeypatch.setattr(session_mod, "warn_sandbox_gaps", _none)
    monkeypatch.setattr(session_mod, "check_network_support", _none)
    monkeypatch.setattr(resume_mod, "check_provider_keys", _none)
    monkeypatch.setattr(session_mod, "budget_preflight", _none)
    monkeypatch.setattr(resume_mod, "verify_git_identity", _none)

    captured: list[str] = []

    def _capture_role(_cfg: object, role: str, **_k: object) -> object:
        captured.append(role)
        raise _Stop()

    monkeypatch.setattr(session_mod, "build_role_provider", _capture_role)
    with pytest.raises(_Stop):
        _cmd_resume(None, "plan-BBBB22", force=False)
    assert captured == ["planner"]


def _session_dir(state: Path, bucket: str, sid: str, mode: str) -> Path:
    d = state / "sessions" / bucket / sid
    d.mkdir(parents=True)
    (d / "manifest.json").write_text(
        json.dumps({"version": 3, "session_id": sid, "mode": mode, "user_task": "t"}),
        encoding="utf-8",
    )
    return d


def test_an_id_matching_two_buckets_is_refused_by_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The fallback re-resolved inside runs/ only, so a prefix matching BOTH a
    run and an ask silently resumed the run."""
    repo = tmp_path / "repo"
    repo.mkdir()
    _git_repo(repo)
    monkeypatch.chdir(repo)
    state = state_dir(repo)
    _session_dir(state, "runs", "quiet-fox-AAAAAA", "run")
    _session_dir(state, "asks", "quiet-fox-BBBBBB", "ask")

    rc = _cmd_resume(None, "quiet-fox-", force=False)

    assert rc == 2
    err = capsys.readouterr().err
    assert "ambiguous" in err
    assert "quiet-fox-AAAAAA" in err and "quiet-fox-BBBBBB" in err


def test_a_session_resume_cannot_continue_is_left_untouched(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """resume reaches every bucket, and it locked + cleared the target's state
    on the way to discovering it could not continue it -- killing a live machine
    draft's worker.pid."""
    repo = tmp_path / "repo"
    repo.mkdir()
    _git_repo(repo)
    monkeypatch.chdir(repo)
    state = state_dir(repo)
    draft = _session_dir(state, "machines", "brave-elk-CCCCCC", "machine")
    (draft / "worker.pid").write_text("4242\n", encoding="utf-8")
    (draft / "answer_1.json").write_text("{}", encoding="utf-8")

    rc = _cmd_resume(None, "brave-elk-CCCCCC", force=False)

    assert rc == 2
    assert (draft / "worker.pid").read_text(encoding="utf-8") == "4242\n"
    assert (draft / "answer_1.json").is_file()


def test_a_resumed_ask_needs_no_repo_and_answers_where_a_fresh_one_does(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A fresh ask is read-only and may run outside a git repo; resuming one
    refused with talk of branches an ask never cuts, and a leg that DID run
    printed no answer and left transcript.md holding the first leg's."""
    from agent6.ui.cli._ask import save_ask_transcript

    outside = tmp_path / "notarepo"
    outside.mkdir()
    monkeypatch.chdir(outside)
    state = state_dir(outside)
    ask = _session_dir(state, "asks", "quiet-fox-AAAAAA", "ask")

    # No snapshot: the refusal that follows proves the git preflight was skipped
    # (it ran BEFORE the snapshot check and would have refused first).
    rc = _cmd_resume(None, "quiet-fox-AAAAAA", force=False)
    assert rc == 2
    assert "no resume snapshot" in capsys.readouterr().err

    layout = SessionLayout(state_dir=state, session_id="quiet-fox-AAAAAA", subdir="asks")
    save_ask_transcript(layout, question="q", answer="first")
    save_ask_transcript(layout, question="q", answer="second")
    text = (ask / "transcript.md").read_text(encoding="utf-8")
    assert "first" in text and "second" in text, "a later leg overwrote the answer"


def test_resuming_a_finished_run_without_a_steer_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A run the agent ENDED has nothing to continue.

    Resuming one spent a provider call, got prose with no tool use, recorded a
    `silent_finish`, and left a run that PASSED reading as failed for a tree
    nobody touched. Refused before anything is spent or cleared; new work is
    what `--steer` is for, and every other ending stays freely resumable.
    """
    repo = tmp_path / "repo"
    repo.mkdir()
    _git_repo(repo)
    monkeypatch.chdir(repo)
    session_dir = state_dir(repo) / "sessions" / "runs" / "done-BBBB22"
    session_dir.mkdir(parents=True)
    (session_dir / "manifest.json").write_text(
        json.dumps({"version": 2, "session_id": "done-BBBB22", "mode": "run", "user_task": "t"}),
        encoding="utf-8",
    )
    (session_dir / "logs.jsonl").write_text(
        json.dumps({"type": "session.start", "mode": "run", "task": "t"})
        + "\n"
        + json.dumps({"type": "session.end", "reason": "finish_session", "all_passed": True})
        + "\n",
        encoding="utf-8",
    )
    # A stale answer file proves the refusal lands before the stale-state
    # clear: a refused resume must not touch the run's shared state.
    approvals = session_dir / "approvals"
    approvals.mkdir()
    (approvals / "a1.answer").write_text("yes", encoding="utf-8")

    rc = _cmd_resume(None, "done-BBBB22", force=False)

    assert rc == 2
    err = capsys.readouterr().err
    assert "already finished" in err
    assert "--steer" in err
    assert (approvals / "a1.answer").exists()


def test_a_finished_run_still_resumes_with_a_steer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The refusal is narrow: `--steer` is new work, so it goes straight
    through -- pinned by the run getting all the way to the snapshot check."""
    repo = tmp_path / "repo"
    repo.mkdir()
    _git_repo(repo)
    monkeypatch.chdir(repo)
    session_dir = state_dir(repo) / "sessions" / "runs" / "done-CCCC33"
    session_dir.mkdir(parents=True)
    (session_dir / "manifest.json").write_text(
        json.dumps({"version": 2, "session_id": "done-CCCC33", "mode": "run", "user_task": "t"}),
        encoding="utf-8",
    )
    (session_dir / "logs.jsonl").write_text(
        json.dumps({"type": "session.end", "reason": "finish_session", "all_passed": True}) + "\n",
        encoding="utf-8",
    )
    rc = _cmd_resume(None, "done-CCCC33", force=False, steer="do more")

    assert rc == 2  # this fixture has no snapshot; the point is WHICH refusal
    err = capsys.readouterr().err
    assert "already finished" not in err
    assert "no resume snapshot" in err
