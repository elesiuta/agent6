# SPDX-License-Identifier: Apache-2.0
"""Regression tests: how `agent6 run` starts on a working tree that is not
clean.

Untracked files are the operator's: a run starts on them without a word,
records them as `untracked-at-start`, and never commits them (an earlier
shape refused every such tree as "dirty" and, with `require_clean_worktree`
off, swept them into the first auto-commit). Uncommitted changes to tracked
files ask the operator (stash / include / cancel) over the same channel as
`ask_user`; a run nobody can answer refuses BEFORE any session dir exists,
`auto_stash` stashes without asking, and `require_clean_worktree = false`
includes without asking.
"""

from __future__ import annotations

import dataclasses
import subprocess
from pathlib import Path

import pytest

import agent6.app._leg as leg_mod
import agent6.app._setup as setup_mod
import agent6.app.preflight as preflight_mod
import agent6.app.run as app_run_mod
import agent6.ui.cli.run as run_mod
from agent6.config import (
    Config,
    GitConfig,
    ModelsConfig,
    OpenAIProviderEntry,
    RoleModel,
    SandboxConfig,
)
from agent6.config.layer import EffectiveConfig
from agent6.git_ops import status as git_status
from agent6.sessions.layout import read_untracked_at_start
from agent6.sessions.manifest import read_manifest
from agent6.tools.operator_prompts import QuestionAnswer, QuestionRequest
from agent6.tools.schema import UserQuestion


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args], check=True, capture_output=True, text=True
    ).stdout


def _init_repo(repo: Path) -> None:
    _git(repo, "init", "-q")
    _git(repo, "config", "user.name", "Test")
    _git(repo, "config", "user.email", "test@example.com")
    (repo / "seed.txt").write_text("seed\n", encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "initial")


def _runnable_cfg(git_cfg: GitConfig) -> Config:
    return Config(
        providers={
            "openrouter": OpenAIProviderEntry(
                api_format="openai",
                base_url="https://openrouter.ai/api/v1",
            )
        },
        models=ModelsConfig(worker=RoleModel(provider="openrouter", model="kimi")),
        git=git_cfg,
        # Answerable, so the worktree policy under test is what decides: the
        # `ask` default has nobody to answer it here and is refused first.
        sandbox=SandboxConfig(run_commands="yes"),
    )


class _Stop(Exception):
    """Raised by a stub to end the run right after the tree policy ran."""


# Whether the tracked files read unmodified at the stop point, per stopped run.
_seen_at_stop: list[bool] = []


def _patch_common(monkeypatch: pytest.MonkeyPatch, cfg: Config, *, stop_after_policy: bool) -> None:
    def _load_effective(*a: object, **k: object) -> EffectiveConfig:
        return EffectiveConfig(config=cfg, sources={}, layers=())

    def _noop(*a: object, **k: object) -> None:
        return None

    # The fake worker model ("kimi") isn't in the real on-disk model cache, so
    # the configured-model preflight would refuse it before the tree policy
    # under test. Bypass it here (its own validation is covered separately).
    def _model_ok(*a: object, **k: object) -> object:
        from agent6.models.validate import ModelValidation

        return ModelValidation(unknown=(), suggestions={}, can_validate=False)

    def _stop(*_a: object, **_k: object) -> object:
        _seen_at_stop.append(git_status(Path.cwd()).modified_count == 0)
        raise _Stop

    monkeypatch.setattr(setup_mod, "load_effective", _load_effective)
    monkeypatch.setattr(preflight_mod, "apply_git_ops_policy", _noop)
    monkeypatch.setattr(run_mod, "validate_configured_model", _model_ok)
    monkeypatch.setattr(preflight_mod, "verify_git_identity", _noop)
    if stop_after_policy:
        # The first step after the tree policy and the untracked snapshot: the
        # leg body's provider session.
        monkeypatch.setattr(leg_mod, "build_session_providers", _stop)


def _answering_frontend(monkeypatch: pytest.MonkeyPatch, answer: str) -> list[UserQuestion]:
    """A front-end that can ask and answers every question with *answer*;
    returns the list the asked questions land in."""
    asked: list[UserQuestion] = []
    real = run_mod.session_frontend

    def _frontend(config_path: Path | None = None) -> object:
        fe = real(config_path)

        def _questioner(_sd: Path) -> object:
            def _ask(request: QuestionRequest, /) -> QuestionAnswer:
                asked.extend(request.questions)
                return QuestionAnswer(tuple(answer for _ in request.questions), "stdin")

            return _ask

        return dataclasses.replace(
            fe,
            capabilities=dataclasses.replace(fe.capabilities, can_ask=True),
            build_questioner=_questioner,
        )

    monkeypatch.setattr(run_mod, "session_frontend", _frontend)
    return asked


def _session_dirs(state: Path) -> list[Path]:
    return sorted(p for p in (state / "sessions" / "runs").glob("*") if p.is_dir())


def test_untracked_files_are_not_dirt_and_are_recorded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_repo(repo)
    (repo / "notes.txt").write_text("mine\n", encoding="utf-8")
    monkeypatch.chdir(repo)
    _patch_common(monkeypatch, _runnable_cfg(GitConfig()), stop_after_policy=True)

    with pytest.raises(_Stop):
        run_mod._cmd_run(None, "do a thing")  # pyright: ignore[reportPrivateUsage]

    err = capsys.readouterr().err
    assert "REFUSING" not in err and "PARKED" not in err
    assert (repo / "notes.txt").read_text(encoding="utf-8") == "mine\n"
    (session_dir,) = _session_dirs(app_run_mod.resolved_state_dir(repo))
    assert read_untracked_at_start(session_dir) == {"notes.txt"}
    assert not read_manifest(session_dir).parked_task


def test_modified_tracked_files_refuse_when_nobody_can_answer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_repo(repo)
    (repo / "seed.txt").write_text("edited\n", encoding="utf-8")
    monkeypatch.chdir(repo)
    # stdin is not a terminal under pytest and no away-mode is set: nobody to ask.
    _patch_common(monkeypatch, _runnable_cfg(GitConfig()), stop_after_policy=True)

    rc = run_mod._cmd_run(None, "do a thing")  # pyright: ignore[reportPrivateUsage]

    assert rc == 2
    err = capsys.readouterr().err
    assert "REFUSING: 1 tracked file has uncommitted changes:\n- seed.txt" in err
    assert "no terminal and no front-end" in err
    assert "[git].auto_stash = true" in err and "[git].require_clean_worktree = false" in err
    # The operator's edit is untouched and no session dir survives the refusal.
    assert (repo / "seed.txt").read_text(encoding="utf-8") == "edited\n"
    assert _session_dirs(app_run_mod.resolved_state_dir(repo)) == []


@pytest.mark.parametrize("answer", ["stash", "stash: set them aside", "STASH"])
def test_answer_stash_stashes_tracked_changes_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, answer: str
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_repo(repo)
    (repo / "seed.txt").write_text("edited\n", encoding="utf-8")
    (repo / "notes.txt").write_text("mine\n", encoding="utf-8")
    monkeypatch.chdir(repo)
    _patch_common(monkeypatch, _runnable_cfg(GitConfig()), stop_after_policy=True)
    asked = _answering_frontend(monkeypatch, answer)

    with pytest.raises(_Stop):
        run_mod._cmd_run(None, "do a thing")  # pyright: ignore[reportPrivateUsage]

    assert len(asked) == 1
    assert asked[0].options == ("stash", "include", "cancel")
    assert "seed.txt" in asked[0].question
    # Stashed for the run (the tree read clean while it ran, the untracked
    # file left alone), and restored when it ended: "stash" promises both.
    assert _seen_at_stop[-1] is True
    assert (repo / "seed.txt").read_text(encoding="utf-8") == "edited\n"
    assert (repo / "notes.txt").read_text(encoding="utf-8") == "mine\n"
    assert _git(repo, "stash", "list") == ""
    (session_dir,) = _session_dirs(app_run_mod.resolved_state_dir(repo))
    assert read_untracked_at_start(session_dir) == {"notes.txt"}


def test_answer_include_starts_with_the_changes_in_place(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_repo(repo)
    (repo / "seed.txt").write_text("edited\n", encoding="utf-8")
    monkeypatch.chdir(repo)
    _patch_common(monkeypatch, _runnable_cfg(GitConfig()), stop_after_policy=True)
    _answering_frontend(monkeypatch, "include")

    with pytest.raises(_Stop):
        run_mod._cmd_run(None, "do a thing")  # pyright: ignore[reportPrivateUsage]

    assert (repo / "seed.txt").read_text(encoding="utf-8") == "edited\n"
    assert _git(repo, "stash", "list") == ""


@pytest.mark.parametrize("answer", ["cancel", "", "no idea"])
def test_answer_cancel_parks_the_run_with_its_task(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], answer: str
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_repo(repo)
    (repo / "seed.txt").write_text("edited\n", encoding="utf-8")
    monkeypatch.chdir(repo)
    _patch_common(monkeypatch, _runnable_cfg(GitConfig()), stop_after_policy=True)
    _answering_frontend(monkeypatch, answer)

    rc = run_mod._cmd_run(None, "do a thing")  # pyright: ignore[reportPrivateUsage]

    assert rc == 2
    err = capsys.readouterr().err
    assert "PARKED: the working tree has uncommitted changes to tracked files" in err
    (session_dir,) = _session_dirs(app_run_mod.resolved_state_dir(repo))
    assert f"agent6 resume {session_dir.name}" in err
    manifest = read_manifest(session_dir)
    assert manifest.parked_task == "do a thing"
    assert manifest.parked_reason == "uncommitted changes"
    assert (repo / "seed.txt").read_text(encoding="utf-8") == "edited\n"


def test_auto_stash_stashes_without_asking(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_repo(repo)
    (repo / "seed.txt").write_text("edited\n", encoding="utf-8")
    monkeypatch.chdir(repo)
    _patch_common(monkeypatch, _runnable_cfg(GitConfig(auto_stash=True)), stop_after_policy=True)
    asked = _answering_frontend(monkeypatch, "cancel")

    with pytest.raises(_Stop):
        run_mod._cmd_run(None, "do a thing")  # pyright: ignore[reportPrivateUsage]

    assert asked == []
    # auto_stash without auto_stash_pop: stashed for the run, left stashed after.
    assert _seen_at_stop[-1] is True
    assert git_status(repo).is_clean
    assert "agent6 auto-stash before run" in _git(repo, "stash", "list")


def test_require_clean_worktree_off_includes_without_asking(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_repo(repo)
    (repo / "seed.txt").write_text("edited\n", encoding="utf-8")
    monkeypatch.chdir(repo)
    cfg = _runnable_cfg(GitConfig(require_clean_worktree=False))
    _patch_common(monkeypatch, cfg, stop_after_policy=True)
    asked = _answering_frontend(monkeypatch, "cancel")

    with pytest.raises(_Stop):
        run_mod._cmd_run(None, "do a thing")  # pyright: ignore[reportPrivateUsage]

    assert asked == []
    assert (repo / "seed.txt").read_text(encoding="utf-8") == "edited\n"


def test_the_last_runs_unmerged_work_is_named_as_such(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A run's edits sit uncommitted on the checkout until its branch merges;
    the next run's dirty-tree text then names that run and its merge, instead
    of calling agent6's own work the operator's uncommitted changes."""
    from agent6.git_ops import chain_commit, chain_ref_for
    from agent6.sessions.layout import bucket_dir

    repo = tmp_path / "repo"
    repo.mkdir()
    _init_repo(repo)
    monkeypatch.chdir(repo)
    state = app_run_mod.resolved_state_dir(repo)
    prior = bucket_dir(state, "runs") / "prior-run-AAAAAA"
    prior.mkdir(parents=True)
    (prior / "logs.jsonl").write_text('{"type": "session.start", "user_task": "t"}\n')
    (repo / "seed.txt").write_text("edited by the prior run\n", encoding="utf-8")
    head = _git(repo, "rev-parse", "HEAD").strip()
    assert chain_commit(repo, "agent6 iter 1", ref=chain_ref_for(prior.name), fallback_parent=head)
    _patch_common(monkeypatch, _runnable_cfg(GitConfig()), stop_after_policy=True)

    assert run_mod._cmd_run(None, "do a thing") == 2  # pyright: ignore[reportPrivateUsage]
    err = capsys.readouterr().err
    assert "the unmerged work of run prior-run-AAAAAA, on agent6/prior-run-AAAAAA" in err
    assert "agent6 sessions merge prior-run-AAAAAA" in err

    # Answered "cancel", the parked message names the merge as well.
    _answering_frontend(monkeypatch, "cancel")
    assert run_mod._cmd_run(None, "do a thing") == 2  # pyright: ignore[reportPrivateUsage]
    parked = capsys.readouterr().err
    assert "PARKED" in parked and "agent6 sessions merge prior-run-AAAAAA" in parked, parked

    # A commit that landed on the base since (another file) does not hide the
    # match: the comparison is per modified file, not whole-tree.
    (repo / "other.txt").write_text("later\n", encoding="utf-8")
    _git(repo, "add", "other.txt")
    _git(repo, "commit", "-q", "-m", "later work on main")
    assert run_mod._cmd_run(None, "do a thing") == 2  # pyright: ignore[reportPrivateUsage]
    assert "the unmerged work of run prior-run-AAAAAA" in capsys.readouterr().err

    # A further edit of the operator's own is not the run's work.
    (repo / "seed.txt").write_text("edited by the prior run\nand by me\n", encoding="utf-8")
    assert run_mod._cmd_run(None, "do a thing") == 2  # pyright: ignore[reportPrivateUsage]
    assert "unmerged work" not in capsys.readouterr().err
