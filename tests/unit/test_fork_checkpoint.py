# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Per-turn checkpoint store + `agent6 fork` (clone-to-new-session recovery).

Covers:
- the loop's pre-call save writes the append-only `checkpoints/<NNNN>.json`
  carrying head_sha + graph_version (every save advances loop_state.json),
- `agent6 fork` clones state, writes lineage manifest fields, cuts the branch,
  and appends `lineage.jsonl`,
- forking a pre-checkpoint (old) run degrades gracefully.
"""

from __future__ import annotations

import json
import subprocess as sp
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest

from agent6.app._setup import SandboxOverrides
from agent6.git_ops import chain_ref_for
from agent6.graph.storage import list_checkpoint_turns, load_graph
from agent6.paths import global_config_dir, state_dir
from agent6.sessions.layout import SessionLayout
from agent6.types import session_bucket
from agent6.ui.cli.fork import _cmd_fork  # pyright: ignore[reportPrivateUsage]
from agent6.ui.cli.resume import _cmd_resume  # pyright: ignore[reportPrivateUsage]
from agent6.workflows._session_state import SNAPSHOT_VERSION, load_session_snapshot
from agent6.workflows.loop import (
    LoopState,
    Workflow,
)


def _silent(_: str) -> None:
    return None


def _git_repo(path: Path) -> str:
    path.mkdir(parents=True, exist_ok=True)
    sp.run(["git", "init", "-q", "-b", "main"], cwd=path, check=True)
    sp.run(["git", "config", "user.email", "t@example.com"], cwd=path, check=True)
    sp.run(["git", "config", "user.name", "t"], cwd=path, check=True)
    (path / "seed.txt").write_text("seed\n")
    sp.run(["git", "add", "seed.txt"], cwd=path, check=True)
    sp.run(["git", "commit", "-q", "-m", "init"], cwd=path, check=True)
    return sp.run(
        ["git", "rev-parse", "HEAD"], cwd=path, capture_output=True, text=True, check=True
    ).stdout.strip()


def _wf(**kw: Any) -> Workflow:
    defaults: dict[str, Any] = {
        "root": Path("/tmp"),
        "config": MagicMock(
            prompt=MagicMock(system_prompt_file=""),
            workflow=MagicMock(verify_command=(), verify_when="never", verify_retries=2),
        ),
        "provider": MagicMock(),
        "dispatcher": MagicMock(),
        "logger": _silent,
    }
    defaults.update(kw)
    return Workflow(**defaults)


# --- checkpoint store -------------------------------------------------------


def test_save_snapshot_writes_per_turn_checkpoint(tmp_path: Path) -> None:
    """`_save_resume_snapshot` writes both loop_state.json AND a per-turn
    checkpoints/<NNNN>.json carrying head_sha + graph_version."""
    repo = tmp_path / "repo"
    head = _git_repo(repo)
    session_dir = tmp_path / "run"
    session_dir.mkdir()
    snap = session_dir / "loop_state.json"

    curator = MagicMock()
    curator.graph_version = 7
    config = SimpleNamespace(
        workflow=SimpleNamespace(
            verify_when="never",
            verify_retries=2,
            verify_command=(),
            metric=SimpleNamespace(goal=None),
        )
    )
    wf = _wf(root=repo, config=config, resume_state_path=snap, curator=curator)
    state = LoopState(original_task="t", tool_calls=0)

    wf._save_resume_snapshot(  # pyright: ignore[reportPrivateUsage]
        system="s",
        messages=[],
        tool_calls=0,
        next_iteration=3,
        root_task_id=None,
        state=state,
        write_checkpoint=True,
    )

    # checkpoints live next to loop_state.json (the run dir).
    cp = session_dir / "checkpoints" / "0003.json"
    assert snap.is_file()
    assert cp.is_file(), "per-turn checkpoint must be written"

    loaded = load_session_snapshot(cp)
    assert loaded.next_iteration == 3
    assert loaded.head_sha == head
    assert loaded.graph_version == 7
    # loop_state.json holds the identical SessionSnapshot bytes as the checkpoint.
    assert json.loads(snap.read_text())["head_sha"] == head


def test_checkpoints_are_append_only(tmp_path: Path) -> None:
    """Each turn writes a distinct checkpoint; older ones are never overwritten."""
    repo = tmp_path / "repo"
    _git_repo(repo)
    session_dir = tmp_path / "run"
    session_dir.mkdir()
    snap = session_dir / "loop_state.json"
    config = SimpleNamespace(
        workflow=SimpleNamespace(
            verify_when="never",
            verify_retries=2,
            verify_command=(),
            metric=SimpleNamespace(goal=None),
        )
    )
    wf = _wf(root=repo, config=config, resume_state_path=snap)
    state = LoopState(original_task="t", tool_calls=0)
    for turn in (1, 2, 3):
        wf._save_resume_snapshot(  # pyright: ignore[reportPrivateUsage]
            system="s",
            messages=[{"role": "user", "content": f"turn {turn}"}],
            tool_calls=0,
            next_iteration=turn,
            root_task_id=None,
            state=state,
            write_checkpoint=True,
        )
    cp_dir = session_dir / "checkpoints"
    assert sorted(p.name for p in cp_dir.glob("*.json")) == ["0001.json", "0002.json", "0003.json"]
    # Turn 1's payload was not clobbered by later turns.
    assert load_session_snapshot(cp_dir / "0001.json").messages[0]["content"] == "turn 1"


def test_only_the_pre_call_save_writes_the_numbered_checkpoint(tmp_path: Path) -> None:
    """A turn's number used to be written by up to three saves (pre-call,
    post-tools, the parallel-group bump), each with a different state, so
    `fork --at-turn N` meant whichever write came last. Only the pre-call save
    (write_checkpoint=True) names a checkpoint; every save still advances
    loop_state.json, the pointer resume and default fork follow."""
    repo = tmp_path / "repo"
    _git_repo(repo)
    session_dir = tmp_path / "run"
    session_dir.mkdir()
    snap = session_dir / "loop_state.json"
    config = SimpleNamespace(
        workflow=SimpleNamespace(
            verify_when="never",
            verify_retries=2,
            verify_command=(),
            metric=SimpleNamespace(goal=None),
        )
    )
    wf = _wf(root=repo, config=config, resume_state_path=snap)
    state = LoopState(original_task="t", tool_calls=0)

    def save(content: str, turn: int, *, checkpoint: bool) -> None:
        wf._save_resume_snapshot(  # pyright: ignore[reportPrivateUsage]
            system="s",
            messages=[{"role": "user", "content": content}],
            tool_calls=0,
            next_iteration=turn,
            root_task_id=None,
            state=state,
            write_checkpoint=checkpoint,
        )

    save("pre-call of turn 2", 2, checkpoint=True)
    save("after turn 2's tools", 3, checkpoint=False)
    save("pre-call of turn 3", 3, checkpoint=True)
    save("after turn 3's tools", 4, checkpoint=False)

    cp_dir = session_dir / "checkpoints"
    assert sorted(p.name for p in cp_dir.glob("*.json")) == ["0002.json", "0003.json"]
    # The checkpoint holds what turn 3's call consumed, not a later overwrite.
    assert (
        load_session_snapshot(cp_dir / "0003.json").messages[0]["content"] == "pre-call of turn 3"
    )
    # Every save advanced the pointer.
    assert load_session_snapshot(snap).messages[0]["content"] == "after turn 3's tools"


def test_list_checkpoint_turns_empty_for_old_run(tmp_path: Path) -> None:
    """A run dir with no checkpoints/ dir lists no turns (old-run detection)."""
    layout = SessionLayout(state_dir=tmp_path, session_id="old")
    (tmp_path / "sessions" / "runs" / "old").mkdir(parents=True)
    assert list_checkpoint_turns(layout) == []


def test_load_run_snapshot_rejects_malformed_shapes(tmp_path: Path) -> None:
    """A wrong-shape checkpoint (null / list / missing key) fails with a clean
    ValueError, which fork's loader catches, instead of an AttributeError."""
    cp = tmp_path / "0001.json"
    for bad in ("null", "[]", '"x"'):
        cp.write_text(bad, encoding="utf-8")
        with pytest.raises(ValueError, match="expected a JSON object"):
            load_session_snapshot(cp)
    cp.write_text(
        json.dumps({"version": SNAPSHOT_VERSION}), encoding="utf-8"
    )  # missing required keys
    with pytest.raises(ValueError, match="malformed run-state snapshot"):
        load_session_snapshot(cp)
    # A torn file is the likeliest corruption of all (a full disk, a power
    # loss), and its refusal named neither the run nor the file: just a JSON
    # position, where both of its siblings above carry the path.
    cp.write_text('{"messages": [{"role":', encoding="utf-8")
    with pytest.raises(ValueError, match=r"unreadable run-state snapshot at .*0001\.json"):
        load_session_snapshot(cp)


# --- fork command -----------------------------------------------------------


def _seed_graph(layout: SessionLayout) -> tuple[str, str]:
    """A two-node DAG through the real curator: root (graph_version 1), child
    (2), child passed (3). Returns (root_id, child_id)."""
    from agent6.graph.curator import GraphCurator
    from agent6.graph.models import AddSubtaskIntent, TaskNodeDraft, UpdateStatusIntent

    curator = GraphCurator(layout)
    root = curator.add_subtask(
        AddSubtaskIntent(parent_id=None, draft=TaskNodeDraft(title="root task", created_by="user"))
    )
    child = curator.add_subtask(
        AddSubtaskIntent(
            parent_id=root.id, draft=TaskNodeDraft(title="late subtask", created_by="worker")
        )
    )
    curator.update_status(UpdateStatusIntent(id=child.id, new_status="passed"))
    return root.id, child.id


def _seed_source_run(
    state_dir: Path,
    session_id: str,
    *,
    head_sha: str,
    turns: tuple[int, ...],
    mode: str = "run",
    workflow_profile: str = "",
    preset_from_flag: bool | None = None,
) -> SessionLayout:
    """Lay down a source session dir with a manifest, graph DAG, and checkpoints."""
    layout = SessionLayout(state_dir=state_dir, session_id=session_id, subdir=session_bucket(mode))
    layout.ensure()
    layout.manifest_path.write_text(
        json.dumps(
            {
                "version": 2,
                "session_id": session_id,
                "mode": mode,
                "user_task": "do the thing",
                "base_sha": "basesha000",
                "base_branch": "main",
                "run_branch": f"agent6/{session_id}",
                "workflow": {
                    "review_trigger": "off",
                    "revise_prompt": "off",
                    "preset": workflow_profile,
                    # Default: a seeded preset is one the fork replays (flag-selected);
                    # pass preset_from_flag=False for a config-selected source.
                    "preset_from_flag": (
                        bool(workflow_profile) if preset_from_flag is None else preset_from_flag
                    ),
                },
            }
        ),
        encoding="utf-8",
    )
    # A REAL curator DAG, so a fork exercises the journal replay: the root lands
    # at graph_version 1, the child at 2, and the child passes at 3 -- matching
    # the per-turn graph_version the checkpoints below record.
    _seed_graph(layout)
    for turn in turns:
        payload = {
            "version": SNAPSHOT_VERSION,
            "system": "sys",
            "messages": [{"role": "user", "content": f"turn {turn}"}],
            "tool_calls": 0,
            "next_iteration": turn,
            "root_task_id": "root",
            "original_task": "do the thing",
            "verify_command": [],
            "head_sha": head_sha,
            "graph_version": turn,
        }
        layout.checkpoint_path(turn).write_text(json.dumps(payload), encoding="utf-8")
    # loop_state.json mirrors the latest checkpoint.
    layout.session_dir.joinpath("loop_state.json").write_text(
        layout.checkpoint_path(turns[-1]).read_text(encoding="utf-8"), encoding="utf-8"
    )
    return layout


def test_fork_preserves_source_run_mode(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Forking a plan run must resume in plan mode. Stamping mode="run" would pair
    # the frozen planning system prompt (which drives finish_planning) with
    # run-mode mutating tools and auto-commits.
    repo = tmp_path / "repo"
    head = _git_repo(repo)
    monkeypatch.chdir(repo)
    state = state_dir(repo)
    _seed_source_run(state, "plan-src-AAAA11", head_sha=head, turns=(1, 2), mode="plan")

    rc = _cmd_fork(None, "plan-src", new_session_id="plan-fork-BBBB22", no_run=True)
    assert rc == 0

    # The fork inherits mode="plan", so its dir belongs in plans/ -- not in the
    # runs/ bucket the default layout would have put it in.
    dst = SessionLayout(state_dir=state, session_id="plan-fork-BBBB22", subdir="plans")
    assert json.loads(dst.manifest_path.read_text(encoding="utf-8"))["mode"] == "plan"
    assert not (state / "sessions" / "runs" / "plan-fork-BBBB22").exists()


def test_fork_refuses_an_explicit_id_held_by_any_bucket(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Ids are one public namespace, so a fork --session-id that any bucket
    already holds is refused up front, naming the holder, before any child
    state exists."""
    repo = tmp_path / "repo"
    head = _git_repo(repo)
    monkeypatch.chdir(repo)
    state = state_dir(repo)
    _seed_source_run(state, "src-AAAA11", head_sha=head, turns=(1,))
    (state / "sessions" / "asks" / "taken-CCCC33").mkdir(parents=True)

    rc = _cmd_fork(None, "src", new_session_id="taken-CCCC33", no_run=True)

    assert rc == 2
    err = capsys.readouterr().err
    assert "asks/" in err and "unique across every bucket" in err
    assert not (state / "sessions" / "runs" / "taken-CCCC33").exists()


def test_fork_preserves_source_run_profile(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # A fork carries the source run's effective preset forward so `resume`
    # re-applies the same strategy; dropping it (writing preset="") would
    # silently change how the forked run behaves.
    repo = tmp_path / "repo"
    head = _git_repo(repo)
    monkeypatch.chdir(repo)
    state = state_dir(repo)
    _seed_source_run(state, "src-AAAA11", head_sha=head, turns=(1, 2), workflow_profile="paranoid")

    rc = _cmd_fork(None, "src", new_session_id="child-BBBB22", no_run=True)
    assert rc == 0

    dst = SessionLayout(state_dir=state, session_id="child-BBBB22")
    manifest = json.loads(dst.manifest_path.read_text(encoding="utf-8"))
    assert manifest["workflow"]["preset"] == "paranoid"


def test_fork_stamps_the_child_manifest_from_the_profiled_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The fork's child runs under the SOURCE's preset (resume replays it), so
    the child manifest's models stamp must come from that profiled config --
    not the base config, which permanently recorded a worker model the forked
    run never used."""

    gdir = global_config_dir()
    gdir.mkdir(parents=True, exist_ok=True)
    (gdir / "config.toml").write_text(
        "[providers.anthropic]\n"
        'api_format = "anthropic"\n'
        "[models.worker]\n"
        'provider = "anthropic"\n'
        'model = "claude-base"\n'
        "[presets.fast.models.worker]\n"
        'provider = "anthropic"\n'
        'model = "claude-fast"\n',
        encoding="utf-8",
    )
    repo = tmp_path / "repo"
    head = _git_repo(repo)
    monkeypatch.chdir(repo)
    state = state_dir(repo)
    _seed_source_run(state, "src-PROF11", head_sha=head, turns=(1,), workflow_profile="fast")

    rc = _cmd_fork(None, "src-PROF11", new_session_id="child-PROF22", no_run=True)
    assert rc == 0
    dst = SessionLayout(state_dir=state, session_id="child-PROF22")
    manifest = json.loads(dst.manifest_path.read_text(encoding="utf-8"))
    assert manifest["workflow"]["preset"] == "fast"
    assert manifest["models"]["driver"]["model"] == "claude-fast"  # not claude-base


def test_fork_of_a_config_selected_profile_stamps_the_current_config_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A CONFIG-selected preset (from_flag False) re-resolves from the CURRENT
    config on fork, so the child stamps the current config's preset name, not the
    source manifest's possibly-stale one -- the fork sibling of the parked-resume
    stamp fix. Only a FLAG-selected preset is pinned by name."""

    global_config_dir().mkdir(parents=True, exist_ok=True)
    (global_config_dir() / "config.toml").write_text('preset = "quick"\n', encoding="utf-8")
    repo = tmp_path / "repo"
    head = _git_repo(repo)
    monkeypatch.chdir(repo)
    state = state_dir(repo)
    # A config-selected source whose stamped name is now STALE (config says quick).
    _seed_source_run(
        state,
        "src-CFG11",
        head_sha=head,
        turns=(1,),
        workflow_profile="stale-old-name",
        preset_from_flag=False,
    )

    assert _cmd_fork(None, "src-CFG11", new_session_id="child-CFG22", no_run=True) == 0
    manifest = json.loads(
        SessionLayout(state_dir=state, session_id="child-CFG22").manifest_path.read_text(
            encoding="utf-8"
        )
    )
    assert manifest["workflow"]["preset"] == "quick"  # re-derived, not "stale-old-name"
    assert manifest["workflow"]["preset_from_flag"] is False


def test_fork_snapshots_the_dag_under_the_source_curator_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A live source's curator atomic-renames and prunes node files; an
    unlocked copy could hit a vanishing file mid-copytree or produce a torn,
    mixed-instant DAG. The copy must run under the source's per-mutation
    curator flock (the same <run>/.lock every write_node holds)."""
    from agent6.app import fork as fork_mod

    repo = tmp_path / "repo"
    head = _git_repo(repo)
    monkeypatch.chdir(repo)
    state = state_dir(repo)
    src = _seed_source_run(state, "src-LOCK11", head_sha=head, turns=(1,))

    locked: list[Path] = []
    real_flock = fork_mod.flock

    @contextmanager
    def recording_flock(path: Path) -> Generator[None]:
        locked.append(path)
        with real_flock(path):
            yield

    monkeypatch.setattr(fork_mod, "flock", recording_flock)
    rc = _cmd_fork(None, "src-LOCK11", new_session_id="child-LOCK22", no_run=True)
    assert rc == 0
    assert locked == [src.lock_path]  # the copy held the source curator lock


def test_fork_fails_loud_on_a_bad_source_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A missing, corrupt-JSON, or non-object source manifest must refuse the
    # fork (exit 2, no run dir, no branch), never fall open to the privileged
    # default mode="run" -- `mode` gates write-tool access (same contract as
    # resume's fail-loud manifest read).
    repo = tmp_path / "repo"
    head = _git_repo(repo)
    monkeypatch.chdir(repo)
    state = state_dir(repo)
    src = _seed_source_run(state, "src-AAAA11", head_sha=head, turns=(1,), mode="plan")

    for bad in (None, "{not json", "[]"):  # missing / corrupt JSON / non-object
        if bad is None:
            src.manifest_path.unlink()
        else:
            src.manifest_path.write_text(bad, encoding="utf-8")
        rc = _cmd_fork(None, "src", new_session_id="child-BBBB22", no_run=True)
        assert rc == 2, f"manifest shape {bad!r} must refuse the fork"
        assert not SessionLayout(state_dir=state, session_id="child-BBBB22").session_dir.exists()
        branches = sp.run(
            ["git", "branch", "--list", "agent6/child-BBBB22"],
            cwd=repo,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        assert branches == "", f"manifest shape {bad!r} must not cut a fork branch"


def test_fork_cleans_up_run_dir_when_branch_cut_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # If the fork branch already exists at a DIFFERENT sha, create_branch_at
    # refuses (we never move a branch) -- the just-materialized run dir must be
    # cleaned up, not left orphaned.
    repo = tmp_path / "repo"
    head = _git_repo(repo)
    monkeypatch.chdir(repo)
    state = state_dir(repo)
    _seed_source_run(state, "src-AAAA11", head_sha=head, turns=(1,))
    # A second commit, and pre-create the fork branch pointing at it (≠ head).
    (repo / "b.txt").write_text("y\n")
    sp.run(["git", "add", "-A"], cwd=repo, check=True)
    sp.run(["git", "commit", "-qm", "c2"], cwd=repo, check=True)
    other = sp.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True, check=True
    ).stdout.strip()
    sp.run(["git", "branch", "agent6/child-BBBB22", other], cwd=repo, check=True)

    rc = _cmd_fork(None, "src", new_session_id="child-BBBB22", no_run=True)
    assert rc == 1
    assert not SessionLayout(state_dir=state, session_id="child-BBBB22").session_dir.exists()


def test_fork_clones_state_writes_lineage_and_branch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`agent6 fork --no-run` clones the checkpoint + DAG into a new run, writes
    the lineage manifest fields, cuts agent6/<new> at the checkpoint sha, and
    appends lineage.jsonl."""
    repo = tmp_path / "repo"
    head = _git_repo(repo)
    monkeypatch.chdir(repo)
    state = state_dir(repo)
    src = _seed_source_run(state, "sunny-otter-AAAA11", head_sha=head, turns=(1, 2, 3))

    rc = _cmd_fork(None, "sunny-otter", new_session_id="brave-yak-BBBB22", no_run=True)
    assert rc == 0

    dst = SessionLayout(state_dir=state, session_id="brave-yak-BBBB22")
    assert dst.session_dir.is_dir()
    # loop_state.json + seed checkpoint 0000.json carry the latest (turn 3) state.
    seed = load_session_snapshot(dst.checkpoint_path(0))
    assert seed.messages[0]["content"] == "turn 3"
    assert (dst.session_dir / "loop_state.json").is_file()
    # DAG rebuilt at the latest checkpoint's graph_version (3): both nodes, the
    # child still passed, and the journal prefix that produced that version.
    from agent6.graph.storage import load_graph

    forked_nodes = load_graph(dst)
    assert {n.title for n in forked_nodes.values()} == {"root task", "late subtask"}
    assert [n.status for n in forked_nodes.values() if n.title == "late subtask"] == ["passed"]
    assert len(dst.journal_path.read_text(encoding="utf-8").splitlines()) == 3

    # Lineage manifest fields.
    manifest = json.loads(dst.manifest_path.read_text(encoding="utf-8"))
    assert manifest["parent_session_id"] == "sunny-otter-AAAA11"
    assert manifest["forked_from_turn"] == 3
    assert manifest["forked_from_sha"] == head
    assert manifest["base_sha"] == "basesha000"
    assert manifest["base_branch"] == "main"
    assert manifest["run_branch"] == "agent6/brave-yak-BBBB22"

    # Branch cut at the checkpoint sha, WITHOUT moving HEAD off main.
    branch_sha = sp.run(
        ["git", "rev-parse", "agent6/brave-yak-BBBB22"],
        cwd=repo,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    assert branch_sha == head
    current = sp.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        cwd=repo,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    assert current == "main", "fork must not move the operator's checkout"

    # lineage.jsonl appended under the state dir root.
    lineage = (state / "lineage.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(lineage) == 1
    ev = json.loads(lineage[0])
    assert ev["child"] == "brave-yak-BBBB22"
    assert ev["parent"] == "sunny-otter-AAAA11"
    assert ev["turn"] == 3
    assert ev["sha"] == head
    assert ev["ts"]

    # Source run is untouched: no new checkpoints, manifest unchanged.
    assert sorted(list_checkpoint_turns(src)) == [1, 2, 3]


def test_latest_fork_uses_loop_state_when_checkpoint_is_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Default fork mirrors resume's latest pointer even if a crash left the
    matching per-turn checkpoint absent."""
    repo = tmp_path / "repo"
    head = _git_repo(repo)
    monkeypatch.chdir(repo)
    state = state_dir(repo)
    src = _seed_source_run(state, "src-AAAA11", head_sha=head, turns=(1, 2))
    latest_payload = {
        "version": SNAPSHOT_VERSION,
        "system": "sys",
        "messages": [{"role": "user", "content": "turn 3 from loop_state"}],
        "tool_calls": 0,
        "next_iteration": 3,
        "root_task_id": "root",
        "original_task": "do the thing",
        "verify_command": [],
        "head_sha": head,
        "graph_version": 3,
    }
    src.session_dir.joinpath("loop_state.json").write_text(
        json.dumps(latest_payload), encoding="utf-8"
    )

    rc = _cmd_fork(None, "src", new_session_id="child-BBBB22", no_run=True)

    assert rc == 0
    dst = SessionLayout(state_dir=state, session_id="child-BBBB22")
    assert load_session_snapshot(dst.checkpoint_path(0)).messages[0]["content"] == (
        "turn 3 from loop_state"
    )


def test_latest_fork_does_not_run_ahead_of_loop_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A crash after checkpoint write but before loop_state write leaves a newer
    checkpoint on disk; default fork must still mirror resume."""
    repo = tmp_path / "repo"
    head = _git_repo(repo)
    monkeypatch.chdir(repo)
    state = state_dir(repo)
    src = _seed_source_run(state, "src-AAAA11", head_sha=head, turns=(1, 2, 3))
    src.session_dir.joinpath("loop_state.json").write_text(
        src.checkpoint_path(2).read_text(encoding="utf-8"), encoding="utf-8"
    )

    rc = _cmd_fork(None, "src", new_session_id="child-BBBB22", no_run=True)

    assert rc == 0
    dst = SessionLayout(state_dir=state, session_id="child-BBBB22")
    assert load_session_snapshot(dst.checkpoint_path(0)).messages[0]["content"] == "turn 2"


def test_fork_at_turn_selects_that_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`--at-turn N` forks from checkpoint N, not the latest."""
    repo = tmp_path / "repo"
    head = _git_repo(repo)
    monkeypatch.chdir(repo)
    state = state_dir(repo)
    _seed_source_run(state, "sunny-otter-AAAA11", head_sha=head, turns=(1, 2, 3))

    rc = _cmd_fork(None, "sunny-otter", at_turn=2, new_session_id="kid-CCCC33", no_run=True)
    assert rc == 0
    dst = SessionLayout(state_dir=state, session_id="kid-CCCC33")
    assert load_session_snapshot(dst.checkpoint_path(0)).messages[0]["content"] == "turn 2"
    assert json.loads(dst.manifest_path.read_text(encoding="utf-8"))["forked_from_turn"] == 2


def test_fork_unknown_turn_errors(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """`--at-turn` with no matching checkpoint is a clean error, no fork dir."""
    repo = tmp_path / "repo"
    head = _git_repo(repo)
    monkeypatch.chdir(repo)
    state = state_dir(repo)
    _seed_source_run(state, "sunny-otter-AAAA11", head_sha=head, turns=(1, 2, 3))

    rc = _cmd_fork(None, "sunny-otter", at_turn=99, new_session_id="kid-DDDD44", no_run=True)
    assert rc == 2
    assert not (state / "sessions" / "runs" / "kid-DDDD44").exists()


def test_fork_at_turn_refuses_without_checkpoint_store(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`--at-turn` selects only from `checkpoints/`: a run with an empty store
    refuses (rc 2, no fork dir) rather than silently substituting the rolling
    snapshot; the default fork (no `--at-turn`) still follows loop_state.json."""
    repo = tmp_path / "repo"
    head = _git_repo(repo)
    monkeypatch.chdir(repo)
    state = state_dir(repo)
    layout = SessionLayout(state_dir=state, session_id="bare-run-EEEE55")
    layout.ensure()
    layout.manifest_path.write_text(
        json.dumps(
            {
                "session_id": "bare-run-EEEE55",
                "mode": "run",
                "base_sha": "x",
                "base_branch": "m",
            }
        ),
        encoding="utf-8",
    )
    layout.session_dir.joinpath("loop_state.json").write_text(
        json.dumps(
            {
                "version": SNAPSHOT_VERSION,
                "system": "s",
                "messages": [{"role": "user", "content": "rolling"}],
                "tool_calls": 0,
                "next_iteration": 4,
                "root_task_id": None,
                "original_task": "task",
                "verify_command": [],
                "head_sha": head,
            }
        ),
        encoding="utf-8",
    )

    rc = _cmd_fork(None, "bare-run", at_turn=4, new_session_id="kid-EEEE55", no_run=True)
    assert rc == 2
    assert not (state / "sessions" / "runs" / "kid-EEEE55").exists()

    rc = _cmd_fork(None, "bare-run", new_session_id="fresh-FFFF66", no_run=True)
    assert rc == 0
    dst = SessionLayout(state_dir=state, session_id="fresh-FFFF66")
    seed = load_session_snapshot(dst.checkpoint_path(0))
    assert seed.messages[0]["content"] == "rolling"
    assert seed.next_iteration == 4


# --- resume gets onto the run branch ---------------------------------------


def _current_branch(repo: Path) -> str:
    return sp.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        cwd=repo,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def test_fork_without_id_forks_most_recent_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = tmp_path / "repo"
    head = _git_repo(repo)
    monkeypatch.chdir(repo)
    state = state_dir(repo)
    _seed_source_run(state, "only-run-AAAA11", head_sha=head, turns=(1,))
    rc = _cmd_fork(None, "", new_session_id="child-BBBB22", no_run=True)
    assert rc == 0
    dst = SessionLayout(state_dir=state, session_id="child-BBBB22")
    manifest = json.loads(dst.manifest_path.read_text(encoding="utf-8"))
    assert manifest["parent_session_id"] == "only-run-AAAA11"  # the only/most-recent run


def test_fork_continue_resumes_without_force(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The default `agent6 fork` continue path just cloned the checkpoint and cut
    # the branch at its head_sha, so the resume head guard passes by
    # construction. force stays OFF so a genuinely misaligned fork still
    # refuses instead of resuming against the wrong worktree.
    repo = tmp_path / "repo"
    head = _git_repo(repo)
    monkeypatch.chdir(repo)
    state = state_dir(repo)
    _seed_source_run(state, "src-AAAA11", head_sha=head, turns=(1,))
    captured: dict[str, Any] = {}

    def _fake_resume(config_path: object, session_id: str, *, force: bool, **_kw: object) -> int:
        captured["force"] = force
        captured["session_id"] = session_id
        return 0

    monkeypatch.setattr("agent6.ui.cli.fork.resume_task", _fake_resume)
    rc = _cmd_fork(  # default: continue; approvable headless, so the continuation is reached
        None,
        "src",
        new_session_id="child-BBBB22",
        sandbox_overrides=SandboxOverrides(auto_approve=True),
    )
    assert rc == 0
    assert captured["force"] is False
    assert captured["session_id"] == "child-BBBB22"


def test_fork_refuses_an_unanswerable_continuation_before_creating_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A continuing fork under run_commands = "ask" with no terminal, no TUI and
    no away-mode refuses like `run` does: BEFORE the fork exists. Refused after
    materializing, a never-started fork stayed listed with its branch cut, and
    the retry with the fix made a second one. `--no-run` continues nothing, so
    it still creates the fork."""
    repo = tmp_path / "repo"
    head = _git_repo(repo)
    monkeypatch.chdir(repo)
    monkeypatch.delenv("AGENT6_DETACHED_AWAY", raising=False)
    monkeypatch.setattr("sys.stdin", SimpleNamespace(isatty=lambda: False))
    state = state_dir(repo)
    _seed_source_run(state, "src-AAAA11", head_sha=head, turns=(1,))
    child = SessionLayout(state_dir=state, session_id="child-BBBB22", subdir="runs")

    rc = _cmd_fork(None, "src", new_session_id="child-BBBB22")
    assert rc == 2
    assert "needs someone to answer" in capsys.readouterr().err
    assert not child.session_dir.exists()
    assert not (state / "lineage.jsonl").exists()
    refs = sp.run(
        ["git", "for-each-ref", "refs/heads/agent6/", "refs/agent6/"],
        cwd=repo,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    assert "child-BBBB22" not in refs

    assert _cmd_fork(None, "src", new_session_id="child-BBBB22", no_run=True) == 0
    assert child.session_dir.is_dir()


def test_fork_without_id_and_no_runs_errors_cleanly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    repo = tmp_path / "repo"
    _git_repo(repo)
    monkeypatch.chdir(repo)
    rc = _cmd_fork(None, "")  # no id, no runs -> clean error, not a crash
    assert rc == 2
    assert "nothing to fork" in capsys.readouterr().err


def test_resume_without_id_and_no_runs_errors_cleanly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    repo = tmp_path / "repo"
    _git_repo(repo)
    monkeypatch.chdir(repo)
    rc = _cmd_resume(None, "", force=False)
    assert rc == 2
    assert "nothing to resume" in capsys.readouterr().err


def test_resume_config_refusal_leaves_checkout_untouched(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # No providers configured: resume refuses BEFORE any workspace mutation.
    # It used to check out agent6/<id> first and leave the operator parked
    # there on every preflight refusal.
    repo = tmp_path / "repo"
    head = _git_repo(repo)
    monkeypatch.chdir(repo)
    state = state_dir(repo)
    _seed_source_run(state, "cfgfail-AAAA11", head_sha=head, turns=(1,))
    rc = _cmd_resume(None, "cfgfail-AAAA11", force=False)
    assert rc == 2
    assert "No providers configured" in capsys.readouterr().err
    assert _current_branch(repo) == "main"
    missing = sp.run(
        ["git", "rev-parse", "--verify", "--quiet", "refs/heads/agent6/cfgfail-AAAA11"],
        cwd=repo,
        capture_output=True,
        check=False,
    )
    assert missing.returncode != 0  # the run branch was never created either


def test_resume_diverged_branch_refuses_without_checkout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # The head guard reads the CHAIN ref's tip: a rewritten chain refuses
    # with the operator's checkout untouched.
    repo = tmp_path / "repo"
    base = _git_repo(repo)
    sp.run(["git", "update-ref", chain_ref_for("divg-AAAA11"), base], cwd=repo, check=True)
    (repo / "seed.txt").write_text("moved on\n")
    sp.run(["git", "commit", "-aqm", "advance main"], cwd=repo, check=True)
    new_head = sp.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True, check=True
    ).stdout.strip()
    monkeypatch.chdir(repo)
    state = state_dir(repo)
    _seed_source_run(state, "divg-AAAA11", head_sha=new_head, turns=(1,))
    rc = _cmd_resume(None, "divg-AAAA11", force=False)
    assert rc == 2  # a refusal
    assert "diverged" in capsys.readouterr().err
    assert _current_branch(repo) == "main"


def test_fork_steer_passes_through_to_the_continuation(monkeypatch: pytest.MonkeyPatch) -> None:
    """Forking exists to try an alternative direction; --steer seeds it at the
    forked session's first safe boundary. Without the pass-through the only
    route was fork --no-run followed by resume --steer, defeating the default
    immediate continuation."""
    import agent6.ui.cli.fork as fork_cli

    def _fake_fork(*_a: object, **_k: object) -> tuple[str, int]:
        return ("kid-AAAA11", 0)

    captured: dict[str, object] = {}

    def _capture_resume(*_a: object, **k: object) -> int:
        captured.update(k)
        return 0

    monkeypatch.setattr(fork_cli, "create_fork", _fake_fork)
    monkeypatch.setattr(fork_cli, "resume_task", _capture_resume)
    rc = fork_cli._cmd_fork(  # pyright: ignore[reportPrivateUsage]
        None, "src-run", steer="try the lock-free design instead"
    )
    assert rc == 0
    assert captured["steer"] == "try the lock-free design instead"


def test_forking_a_finished_run_with_no_new_work_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The child would continue a conversation that already ended: a paid call,
    a nudge, a silent finish, a new branch, and a listing row offering a merge
    of the parent's own tree. `resume` refuses exactly this and cannot see it
    from the child, whose log is empty by construction."""
    import agent6.ui.cli.fork as fork_cli

    monkeypatch.chdir(tmp_path)
    layout = SessionLayout(state_dir=state_dir(tmp_path), session_id="done-run-AAAA11")
    layout.ensure()
    layout.logs_path.write_text(
        json.dumps({"type": "session.start", "mode": "run", "user_task": "t"})
        + "\n"
        + json.dumps({"type": "session.end", "reason": "finish_session", "all_passed": True})
        + "\n",
        encoding="utf-8",
    )

    def _must_not_fork(*_a: object, **_k: object) -> tuple[str, int]:
        pytest.fail("create_fork must not run when the fork has nothing to do")

    monkeypatch.setattr(fork_cli, "create_fork", _must_not_fork)

    assert fork_cli._cmd_fork(None, "done-run") == 2  # pyright: ignore[reportPrivateUsage]

    err = capsys.readouterr().err
    assert "--steer" in err and "--at-turn" in err


def test_fork_steer_with_no_run_is_refused(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """--steer only means something for the immediate continuation; with
    --no-run nothing ever ran to receive it, so refuse up front (before any
    fork dir is created) instead of dropping the instruction silently."""
    import agent6.ui.cli.fork as fork_cli

    def _must_not_fork(*_a: object, **_k: object) -> tuple[str, int]:
        pytest.fail("create_fork must not run when the flag combo is refused")

    monkeypatch.setattr(fork_cli, "create_fork", _must_not_fork)
    rc = fork_cli._cmd_fork(  # pyright: ignore[reportPrivateUsage]
        None, "src-run", no_run=True, steer="x"
    )
    assert rc == 2
    assert "--steer" in capsys.readouterr().err


def test_a_past_turn_fork_starts_on_the_task_that_turn_was_on(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The cursor is the run's focus, and a fork replays it like every other
    fact. Falling back to the SOURCE's current cursor when the prefix set none
    handed the child the last turn's task: it worked the newest task first and
    the earlier ones after it."""
    from agent6.graph.curator import GraphCurator
    from agent6.graph.models import SetCursorIntent

    repo = tmp_path / "repo"
    head = _git_repo(repo)
    monkeypatch.chdir(repo)
    state = state_dir(repo)
    src = _seed_source_run(state, "sunny-otter-AAAA11", head_sha=head, turns=(1, 2, 3))
    # The focus lands AFTER turn 1's graph_version: turn 1 had no cursor.
    late = next(n for n in load_graph(src).values() if n.title == "late subtask")
    GraphCurator(src).set_cursor(SetCursorIntent(id=late.id))
    assert json.loads(src.cursor_path.read_text(encoding="utf-8"))["node_id"] == late.id

    assert _cmd_fork(None, "sunny-otter", at_turn=1, new_session_id="kid-DDDD44", no_run=True) == 0

    dst = SessionLayout(state_dir=state, session_id="kid-DDDD44")
    assert json.loads(dst.cursor_path.read_text(encoding="utf-8"))["node_id"] is None


def test_fork_at_past_turn_rebuilds_the_graph_of_that_turn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A past-turn fork gets the DAG as it stood at that turn, not the run's
    latest. Copying the newest graph handed the forked session tasks it never
    created and `passed` statuses for work absent from its tree -- and, since
    DAG statuses drive the focus frontier and the finish gate, a turn-1 fork
    resumed with an already-satisfied gate."""
    from agent6.graph.storage import load_graph

    repo = tmp_path / "repo"
    head = _git_repo(repo)
    monkeypatch.chdir(repo)
    state = state_dir(repo)
    # graph_version 1 = root only; 2 = root + child; 3 = child passed.
    src = _seed_source_run(state, "sunny-otter-AAAA11", head_sha=head, turns=(1, 2, 3))
    assert {n.title for n in load_graph(src).values()} == {"root task", "late subtask"}

    assert _cmd_fork(None, "sunny-otter", at_turn=1, new_session_id="kid-CCCC33", no_run=True) == 0

    dst = SessionLayout(state_dir=state, session_id="kid-CCCC33")
    nodes = load_graph(dst)
    # The turn-3 subtask did not exist at turn 1, and the root had not passed.
    assert [n.title for n in nodes.values()] == ["root task"]
    assert [n.status for n in nodes.values()] == ["pending"]
    assert [n.children for n in nodes.values()] == [()]
    # The cursor cannot point into the future: the run held none at turn 1, and
    # inheriting the SOURCE's current one started the child on the last turn's
    # task with every earlier one pending. `in (None, first)` accepted that
    # while the graph had one node.
    assert json.loads(dst.cursor_path.read_text(encoding="utf-8"))["node_id"] is None
    versions = [
        json.loads(line)["graph_version"]
        for line in dst.journal_path.read_text(encoding="utf-8").splitlines()
    ]
    assert versions == [1]


def test_a_past_turn_fork_reopens_without_a_lost_tail_warning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The rebuild undoes the mutations stamped after the forked turn; a node
    kept its newer stamp, so the fork's curator read its (correctly shorter)
    journal as one that lost its tail, warned so on every open, and resumed
    numbering from the source's future. Stamps clamp to the forked version."""
    from agent6.graph.curator import GraphCurator
    from agent6.graph.storage import load_graph

    repo = tmp_path / "repo"
    head = _git_repo(repo)
    monkeypatch.chdir(repo)
    state = state_dir(repo)
    _seed_source_run(state, "sunny-otter-AAAA11", head_sha=head, turns=(1, 2, 3))
    assert _cmd_fork(None, "sunny-otter", at_turn=1, new_session_id="kid-CCCC33", no_run=True) == 0

    dst = SessionLayout(state_dir=state, session_id="kid-CCCC33")
    assert [n.graph_version for n in load_graph(dst).values()] == [1]
    capsys.readouterr()
    curator = GraphCurator(dst)
    assert "lost its tail" not in capsys.readouterr().err
    assert curator.graph_version == 1


def test_fork_copies_the_dag_when_the_checkpoint_has_no_graph_version(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """graph_version 0 means the checkpoint predates the stamp (or the curator
    was unreadable when it was written). With no version to rebuild at, the
    fork copies the DAG verbatim rather than rebuilding to an empty graph."""
    from agent6.graph.storage import load_graph

    repo = tmp_path / "repo"
    head = _git_repo(repo)
    monkeypatch.chdir(repo)
    state = state_dir(repo)
    src = _seed_source_run(state, "old-run-AAAA11", head_sha=head, turns=(1,))
    payload = json.loads(src.checkpoint_path(1).read_text(encoding="utf-8"))
    payload["graph_version"] = 0
    src.checkpoint_path(1).write_text(json.dumps(payload), encoding="utf-8")

    assert _cmd_fork(None, "old-run", at_turn=1, new_session_id="kid-DDDD44", no_run=True) == 0

    dst = SessionLayout(state_dir=state, session_id="kid-DDDD44")
    assert {n.title for n in load_graph(dst).values()} == {"root task", "late subtask"}


def test_an_auto_minted_fork_id_skips_a_taken_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A fork names a session DIRECTORY, so its auto-minted id goes through the
    owner that checks the bucket. Minting a raw token instead wrote into
    whatever already stood there."""
    from agent6.sessions import id as id_mod

    repo = tmp_path / "repo"
    head = _git_repo(repo)
    monkeypatch.chdir(repo)
    state = state_dir(repo)
    _seed_source_run(state, "forky-src-AAAA11", head_sha=head, turns=(1, 2))
    taken = state / "sessions" / "runs" / "taken-one-AAAAAA"
    taken.mkdir(parents=True)
    (taken / "marker.txt").write_text("do not clobber\n", encoding="utf-8")
    minted = iter(["taken-one-AAAAAA", "freed-two-BBBBBB"])
    monkeypatch.setattr(id_mod, "friendly_token", lambda: next(minted))

    assert _cmd_fork(None, "forky-src", no_run=True) == 0

    assert (taken / "marker.txt").read_text(encoding="utf-8") == "do not clobber\n"
    assert (state / "sessions" / "runs" / "freed-two-BBBBBB" / "manifest.json").is_file()


def test_fork_manifest_stamps_the_resolved_isolation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A fork's policy stamp is the level it would run at on this host, like
    a run's, never the `auto` knob (which tells `exec` and the pages nothing)."""
    repo = tmp_path / "repo"
    head = _git_repo(repo)
    monkeypatch.chdir(repo)
    state = state_dir(repo)
    _seed_source_run(state, "iso-src-AAAA11", head_sha=head, turns=(1, 2))

    def _hardened(_knob: str, _env: object) -> str:
        return "hardened"

    monkeypatch.setattr("agent6.app.fork.resolve_isolation", _hardened)
    assert _cmd_fork(None, "iso-src", new_session_id="iso-fork-BBBB22", no_run=True) == 0
    dst = SessionLayout(state_dir=state, session_id="iso-fork-BBBB22", subdir="runs")
    manifest = json.loads(dst.manifest_path.read_text(encoding="utf-8"))
    assert manifest["policy"]["isolation"] == "hardened"


# --- a fork's own worktree ------------------------------------------------------


def _commit_all(repo: Path, message: str) -> str:
    sp.run(["git", "add", "-A"], cwd=repo, check=True)
    sp.run(["git", "commit", "-q", "-m", message], cwd=repo, check=True)
    return sp.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True, check=True
    ).stdout.strip()


def _fork_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, str, str]:
    """A repo whose checkout moved past the source run's turn-1 sha (a later
    commit, an uncommitted edit, an operator file), and that source run.
    Returns (repo, turn-1 sha, HEAD sha)."""
    repo = tmp_path / "repo"
    turn1 = _git_repo(repo)
    (repo / "seed.txt").write_text("later\n", encoding="utf-8")
    head = _commit_all(repo, "second")
    (repo / "seed.txt").write_text("dirty\n", encoding="utf-8")
    (repo / "notes.md").write_text("mine\n", encoding="utf-8")
    monkeypatch.chdir(repo)
    _seed_source_run(state_dir(repo), "src-AAAA11", head_sha=turn1, turns=(1,))
    return repo, turn1, head


def test_a_fork_gets_its_own_worktree_and_commits_only_its_own_edits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`fork --at-turn N` gives the fork a linked worktree detached at the
    checkpoint sha, recorded in its manifest; a chain commit made there records
    the fork's own edit and nothing of the source checkout, which stays as it
    was. Seeding only the refs left the fork sharing the checkout, so its
    first commit snapshotted the source's later content as its own work."""
    from agent6.git_ops import chain_commit, tree_diff_paths

    repo, turn1, head = _fork_fixture(tmp_path, monkeypatch)
    state = state_dir(repo)

    assert _cmd_fork(None, "src", at_turn=1, new_session_id="child-BBBB22", no_run=True) == 0

    dst = SessionLayout(state_dir=state, session_id="child-BBBB22")
    manifest = json.loads(dst.manifest_path.read_text(encoding="utf-8"))
    worktree = Path(manifest["worktree"])
    assert (worktree / ".git").is_file(), "a linked worktree of the repository"
    assert manifest["worktree_git_dir"] == str((repo / ".git").resolve())
    assert (worktree / "seed.txt").read_text(encoding="utf-8") == "seed\n"
    assert not (worktree / "notes.md").exists()
    rev = ["git", "rev-parse", "HEAD"]
    assert sp.run(rev, cwd=worktree, capture_output=True, text=True, check=True).stdout.strip() == (
        turn1
    )

    (worktree / "fork.txt").write_text("fork\n", encoding="utf-8")
    sha = chain_commit(
        worktree, "fork step", ref=chain_ref_for("child-BBBB22"), fallback_parent=turn1
    )
    assert sha is not None
    assert tree_diff_paths(repo, turn1, sha) == ["fork.txt"]

    # The source checkout: HEAD, its uncommitted edit, and the operator's file.
    assert sp.run(rev, cwd=repo, capture_output=True, text=True, check=True).stdout.strip() == head
    assert (repo / "seed.txt").read_text(encoding="utf-8") == "dirty\n"
    assert (repo / "notes.md").read_text(encoding="utf-8") == "mine\n"
    assert not (repo / "fork.txt").exists()


def test_resume_of_a_fork_runs_its_leg_in_the_worktree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`agent6 resume <fork>` from the repo drives the leg with the fork's
    worktree as its checkout (the process cwd stays the repo: its state dir
    and config are the repo's). Run in the repo instead, the fork committed
    the operator's checkout."""

    from agent6.app import resume as resume_mod
    from agent6.app._leg import LegEnd, LegInputs

    global_config_dir().mkdir(parents=True, exist_ok=True)
    (global_config_dir() / "config.toml").write_text(
        '[providers.anthropic]\napi_format = "anthropic"\n'
        '[models.worker]\nprovider = "anthropic"\nmodel = "claude-x"\n',
        encoding="utf-8",
    )
    repo, _turn1, _head = _fork_fixture(tmp_path, monkeypatch)
    state = state_dir(repo)
    assert _cmd_fork(None, "src", at_turn=1, new_session_id="child-BBBB22", no_run=True) == 0
    manifest = json.loads(
        SessionLayout(state_dir=state, session_id="child-BBBB22").manifest_path.read_text(
            encoding="utf-8"
        )
    )
    worktree = Path(manifest["worktree"])
    seen: dict[str, Any] = {}

    def _fake_leg(cfg: Any, layout: Any, inputs: LegInputs, **kw: Any) -> LegEnd:
        seen["cwd"] = kw["cwd"]
        seen["state_dir"] = kw["state_dir"]
        seen["process_cwd"] = Path.cwd()
        return LegEnd(0)

    def _no_missing(_cfg: object) -> None:
        return None

    def _strict(*_a: object, **_k: object) -> str:
        return "strict"

    monkeypatch.setattr(resume_mod, "run_leg", _fake_leg)
    monkeypatch.setattr(resume_mod, "check_provider_keys", _no_missing)
    monkeypatch.setattr(resume_mod, "select_isolation", _strict)
    rc = resume_mod.resume_task(None, "child-BBBB22", frontend=MagicMock(), force=False)
    assert rc == 0
    assert seen["cwd"] == worktree and seen["process_cwd"] == repo
    assert seen["state_dir"] == state
    assert Path.cwd() == repo


def test_resume_of_a_fork_whose_worktree_is_gone_refuses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A pruned or deleted worktree is named, not silently replaced by the
    operator's checkout."""
    import shutil

    from agent6.app import resume as resume_mod

    repo, _turn1, _head = _fork_fixture(tmp_path, monkeypatch)
    state = state_dir(repo)
    assert _cmd_fork(None, "src", at_turn=1, new_session_id="child-BBBB22", no_run=True) == 0
    manifest = json.loads(
        SessionLayout(state_dir=state, session_id="child-BBBB22").manifest_path.read_text(
            encoding="utf-8"
        )
    )
    shutil.rmtree(manifest["worktree"])

    rc = resume_mod.resume_task(None, "child-BBBB22", frontend=MagicMock(), force=False)
    assert rc == 2
    err = capsys.readouterr().err
    assert manifest["worktree"] in err and "agent6 fork child-BBBB22" in err
    assert Path.cwd() == repo


def test_resume_of_a_pruned_fork_points_at_the_merge_that_landed_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A merged fork whose branch and worktree are gone (deleted here by hand,
    as `sessions prune --delete-squashed` and the worktree sweep leave them):
    the refusal points at the merge stamp, what `sessions commits` prints,
    never at the branch it named unchecked."""
    import shutil

    from agent6.app import resume as resume_mod

    repo, _turn1, head = _fork_fixture(tmp_path, monkeypatch)
    state = state_dir(repo)
    assert _cmd_fork(None, "src", at_turn=1, new_session_id="child-BBBB22", no_run=True) == 0
    layout = SessionLayout(state_dir=state, session_id="child-BBBB22")
    manifest = json.loads(layout.manifest_path.read_text(encoding="utf-8"))
    manifest["merged"] = {"into": "main", "sha": head, "tip": manifest["forked_from_sha"]}
    layout.manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    sp.run(["git", "branch", "-D", "agent6/child-BBBB22"], cwd=repo, check=True)
    shutil.rmtree(manifest["worktree"])
    capsys.readouterr()

    assert resume_mod.resume_task(None, "child-BBBB22", frontend=MagicMock(), force=False) == 2
    err = capsys.readouterr().err
    assert f"merged into main as {head[:12]}" in err
    assert "agent6/child-BBBB22" not in err and "refs/agent6" not in err


def test_resume_of_a_pruned_fork_names_the_chain_ref_past_its_stamp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A fork merged, then resumed (a later commit on its branch and chain),
    then its branch deleted and its worktree removed: the stamp covers only
    the earlier tip, so the refusal names the chain ref that holds the later
    commit, never the merge (which trusted the stamp unchecked)."""
    import shutil

    from agent6.app import resume as resume_mod

    repo, turn1, head = _fork_fixture(tmp_path, monkeypatch)
    state = state_dir(repo)
    assert _cmd_fork(None, "src", at_turn=1, new_session_id="child-BBBB22", no_run=True) == 0
    layout = SessionLayout(state_dir=state, session_id="child-BBBB22")
    manifest = json.loads(layout.manifest_path.read_text(encoding="utf-8"))
    manifest["merged"] = {"into": "main", "sha": head, "tip": turn1}
    layout.manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    later = sp.run(
        ["git", "commit-tree", f"{turn1}^{{tree}}", "-p", turn1, "-m", "a later leg"],
        cwd=repo,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    sp.run(["git", "update-ref", chain_ref_for("child-BBBB22"), later], cwd=repo, check=True)
    sp.run(["git", "branch", "-D", "agent6/child-BBBB22"], cwd=repo, check=True)
    shutil.rmtree(manifest["worktree"])
    capsys.readouterr()

    assert resume_mod.resume_task(None, "child-BBBB22", frontend=MagicMock(), force=False) == 2
    err = capsys.readouterr().err
    assert f"its commits are on {chain_ref_for('child-BBBB22')}" in err
    assert "merged into main" not in err


def test_a_steered_fork_takes_the_steer_as_its_own_task(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A fork carries its source's task until the operator sends it elsewhere.

    Left as the source's, the steer's work merged under the source's subject
    and `sessions list` showed the fork under a task it was never given.
    """
    repo = tmp_path / "repo"
    head = _git_repo(repo)
    monkeypatch.chdir(repo)
    state = state_dir(repo)
    _seed_source_run(state, "sunny-otter-AAAA11", head_sha=head, turns=(1,))
    assert _cmd_fork(None, "sunny-otter", new_session_id="brave-yak-BBBB22", no_run=True) == 0
    dst = SessionLayout(state_dir=state, session_id="brave-yak-BBBB22")
    assert json.loads(dst.manifest_path.read_text(encoding="utf-8"))["user_task"] == "do the thing"

    # No providers configured, so the leg never starts; the steer is queued and
    # the task stamped before that refusal, as they are for a normal resume.
    assert _cmd_resume(None, "brave-yak-BBBB22", force=False, steer="create README.md only") == 2

    manifest = json.loads(dst.manifest_path.read_text(encoding="utf-8"))
    assert manifest["user_task"] == "create README.md only"
    assert manifest["parent_session_id"] == "sunny-otter-AAAA11", "lineage still names the source"


def test_only_the_first_steer_names_a_fork(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The steer that sends a fork elsewhere is its task; a later one is a
    follow-up within that task, as it is for a run of its own. Stamping every
    steer re-titled a fork on each resume, and ACP passes every editor prompt
    as one."""
    repo = tmp_path / "repo"
    head = _git_repo(repo)
    monkeypatch.chdir(repo)
    state = state_dir(repo)
    _seed_source_run(state, "sunny-otter-AAAA11", head_sha=head, turns=(1,))
    assert _cmd_fork(None, "sunny-otter", new_session_id="brave-yak-BBBB22", no_run=True) == 0
    dst = SessionLayout(state_dir=state, session_id="brave-yak-BBBB22")

    assert _cmd_resume(None, "brave-yak-BBBB22", force=False, steer="create README.md only") == 2
    assert _cmd_resume(None, "brave-yak-BBBB22", force=False, steer="also fix the typo") == 2

    manifest = json.loads(dst.manifest_path.read_text(encoding="utf-8"))
    assert manifest["user_task"] == "create README.md only"


def test_a_steered_ordinary_run_keeps_its_task(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`resume --steer` on a run of its own is a follow-up within that task,
    not a new one: the headline it has carried all along stays."""
    repo = tmp_path / "repo"
    head = _git_repo(repo)
    monkeypatch.chdir(repo)
    state = state_dir(repo)
    src = _seed_source_run(state, "sunny-otter-AAAA11", head_sha=head, turns=(1,))

    assert _cmd_resume(None, "sunny-otter-AAAA11", force=False, steer="and add tests") == 2

    assert json.loads(src.manifest_path.read_text(encoding="utf-8"))["user_task"] == "do the thing"


def test_a_fork_records_the_untracked_files_of_its_own_checkout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A fork's worktree is a fresh checkout of the checkpoint sha, so the
    source's untracked paths are not in it. Inheriting that set described the
    wrong checkout and left a file dropped into the fork's worktree unexcluded,
    so its first commit swept the operator's file in."""
    from agent6.sessions.layout import read_untracked_at_start

    repo = tmp_path / "repo"
    head = _git_repo(repo)
    monkeypatch.chdir(repo)
    state = state_dir(repo)
    src = _seed_source_run(state, "sunny-otter-AAAA11", head_sha=head, turns=(1,))
    (repo / "notes.txt").write_text("the operator's own file\n", encoding="utf-8")
    (src.session_dir / "untracked-at-start").write_bytes(b"notes.txt")

    assert _cmd_fork(None, "sunny-otter", new_session_id="brave-yak-BBBB22", no_run=True) == 0

    dst = SessionLayout(state_dir=state, session_id="brave-yak-BBBB22")
    worktree = Path(json.loads(dst.manifest_path.read_text(encoding="utf-8"))["worktree"])
    assert not (worktree / "notes.txt").exists(), "a fresh checkout has none of it"
    assert read_untracked_at_start(dst.session_dir) == frozenset()


def test_a_refused_fork_leaves_no_chain_ref_behind(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """`sessions rm` keeps a run's branch, so forking onto a reused id hits the
    branch refusal. The chain ref was written first and never removed, and a
    later run with that id would build its chain on the dead fork's tip."""
    from agent6.git_ops import chain_ref_for, chain_tip, set_ref

    repo = tmp_path / "repo"
    head = _git_repo(repo)
    monkeypatch.chdir(repo)
    state = state_dir(repo)
    _seed_source_run(state, "sunny-otter-AAAA11", head_sha=head, turns=(1,))
    # The branch a removed run left behind, pointing at a different commit.
    (repo / "other.txt").write_text("other\n")
    sp.run(["git", "add", "other.txt"], cwd=repo, check=True)
    sp.run(["git", "commit", "-q", "-m", "other"], cwd=repo, check=True)
    sp.run(["git", "branch", "agent6/brave-yak-BBBB22", "HEAD"], cwd=repo, check=True)

    assert _cmd_fork(None, "sunny-otter", new_session_id="brave-yak-BBBB22", no_run=True) == 1

    assert "could not cut fork refs" in capsys.readouterr().err
    assert chain_tip(repo, chain_ref_for("brave-yak-BBBB22")) is None
    assert not SessionLayout(state_dir=state, session_id="brave-yak-BBBB22").session_dir.exists()

    # The same refusal over an id whose chain ref already holds commits: that
    # ref is the earlier run's anchor and must survive.
    anchor = sp.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True, check=True
    ).stdout.strip()
    set_ref(repo, chain_ref_for("brave-yak-BBBB22"), anchor)

    assert _cmd_fork(None, "sunny-otter", new_session_id="brave-yak-BBBB22", no_run=True) == 1

    assert chain_tip(repo, chain_ref_for("brave-yak-BBBB22")) == anchor


def test_fork_refuses_a_run_the_model_controls_git_in(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A fork runs in a linked worktree whose `.git` the jail grants read-only;
    under `[git].control = "model"` the prompt would tell the model to commit
    and the sandbox would refuse every commit. The fork refuses up front, with
    nothing materialized."""

    global_config_dir().mkdir(parents=True, exist_ok=True)
    (global_config_dir() / "config.toml").write_text(
        '[git]\ncontrol = "model"\n[sandbox]\nprotect_git = false\n', encoding="utf-8"
    )
    repo = tmp_path / "repo"
    head = _git_repo(repo)
    monkeypatch.chdir(repo)
    state = state_dir(repo)
    _seed_source_run(state, "src-MOD11", head_sha=head, turns=(1,))

    rc = _cmd_fork(None, "src-MOD11", new_session_id="child-MOD22", no_run=True)

    assert rc == 2
    assert 'control = "model"' in capsys.readouterr().err
    assert not SessionLayout(state_dir=state, session_id="child-MOD22").manifest_path.exists()
