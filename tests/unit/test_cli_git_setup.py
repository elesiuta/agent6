# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""`agent6 run`/`init` git pre-flight: nice no-repo error + the init git offer."""

from __future__ import annotations

import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

from agent6.app.preflight import headless_approval_refusal, require_git_repo
from agent6.config import Config
from agent6.git_ops import init_repo, is_git_repo
from agent6.ui.cli import main
from agent6.ui.cli.init_cmds import _offer_git_setup  # pyright: ignore[reportPrivateUsage]


def test_headless_approval_refusal(monkeypatch: pytest.MonkeyPatch) -> None:
    """Duplicate coverage of the refusal, from the CLI's angle: the same helper
    both lifecycles call before starting anything."""
    ask = cast(Config, SimpleNamespace(sandbox=SimpleNamespace(run_commands="ask")))
    assert headless_approval_refusal(ask, tui_enabled=False, away="", can_ask=False) is not None
    # Answerable: a TUI, a front-end that can ask, an away-mode, or nothing to approve.
    assert headless_approval_refusal(ask, tui_enabled=True, away="", can_ask=False) is None
    assert headless_approval_refusal(ask, tui_enabled=False, away="deny", can_ask=False) is None
    assert headless_approval_refusal(ask, tui_enabled=False, away="", can_ask=True) is None
    yes = cast(Config, SimpleNamespace(sandbox=SimpleNamespace(run_commands="yes")))
    assert headless_approval_refusal(yes, tui_enabled=False, away="", can_ask=False) is None


def test_run_surfaces_git_wall_before_provider_wall(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # Non-git scratch dir with no provider configured (conftest isolates config):
    # the not-a-git-repo error surfaces first, not after the provider/key walls.
    monkeypatch.chdir(tmp_path)
    rc = main(["run", "do a thing"])
    assert rc == 2
    err = capsys.readouterr().err
    assert "not a git repository" in err
    assert "No providers configured" not in err  # git wall comes first now


def test_require_git_repo_errors_outside_repo(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert require_git_repo(tmp_path) is False
    err = capsys.readouterr().err
    assert "not a git repository" in err
    assert "agent6 init" in err  # points at the guided fix


def test_require_git_repo_ok_inside_repo(tmp_path: Path) -> None:
    init_repo(tmp_path)
    assert require_git_repo(tmp_path) is True


def test_offer_git_setup_noninteractive_just_notes(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _offer_git_setup(tmp_path, (tmp_path / "AGENTS.md",), interactive=False)
    out = capsys.readouterr().out
    assert "not a git repository" in out
    assert is_git_repo(tmp_path) is False  # did NOT create a repo non-interactively


def test_offer_git_setup_interactive_inits_and_commits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _yes(_prompt: str) -> str:
        return "y"

    monkeypatch.setattr("builtins.input", _yes)
    # GIT_* env supplies the commit identity without touching global config.
    for k in ("GIT_AUTHOR_NAME", "GIT_COMMITTER_NAME"):
        monkeypatch.setenv(k, "Test")
    for k in ("GIT_AUTHOR_EMAIL", "GIT_COMMITTER_EMAIL"):
        monkeypatch.setenv(k, "t@t.t")
    (tmp_path / ".gitignore").write_text("__pycache__/\n", encoding="utf-8")
    (tmp_path / "AGENTS.md").write_text("# AGENTS\n", encoding="utf-8")
    # The per-repo config lives OUT of the workspace. Passing such a path must
    # not crash _offer_git_setup (it filters to paths under the repo) and it is
    # never committed.
    out_of_repo_cfg = tmp_path.parent / "a6-state" / "config.toml"
    out_of_repo_cfg.parent.mkdir(parents=True, exist_ok=True)
    out_of_repo_cfg.write_text("# cfg\n", encoding="utf-8")

    _offer_git_setup(
        tmp_path,
        (out_of_repo_cfg, tmp_path / "AGENTS.md", tmp_path / ".gitignore"),
        interactive=True,
    )

    assert is_git_repo(tmp_path) is True
    tracked = subprocess.run(
        ["git", "-C", str(tmp_path), "ls-files"], capture_output=True, text=True, check=True
    ).stdout.split()
    assert "AGENTS.md" in tracked
    assert ".gitignore" in tracked
    assert "config.toml" not in tracked  # out-of-repo config is never committed


def _existing_repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A repo with one commit and an identity from GIT_* env."""
    for k in ("GIT_AUTHOR_NAME", "GIT_COMMITTER_NAME"):
        monkeypatch.setenv(k, "Test")
    for k in ("GIT_AUTHOR_EMAIL", "GIT_COMMITTER_EMAIL"):
        monkeypatch.setenv(k, "t@t.t")
    init_repo(tmp_path)
    (tmp_path / "README.md").write_text("x\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(tmp_path), "add", "-A"], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "commit", "-qm", "init"], check=True)


def _porcelain(tmp_path: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(tmp_path), "status", "--porcelain"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout


def test_offer_git_setup_existing_repo_commits_scaffold_noninteractive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # In an already-git repo, init used to leave the scaffold uncommitted, so
    # the advertised `agent6 run "<task>"` refused on a dirty tree.
    # Non-interactive means --yes (non-TTY without --yes is refused earlier).
    _existing_repo(tmp_path, monkeypatch)
    (tmp_path / "AGENTS.md").write_text("# AGENTS\n", encoding="utf-8")
    (tmp_path / ".gitignore").write_text(".env\n", encoding="utf-8")
    _offer_git_setup(tmp_path, (tmp_path / "AGENTS.md", tmp_path / ".gitignore"), interactive=False)
    assert _porcelain(tmp_path) == ""  # scaffold committed, tree clean


def test_offer_git_setup_existing_repo_declined_prints_exact_command(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _existing_repo(tmp_path, monkeypatch)
    (tmp_path / "AGENTS.md").write_text("# AGENTS\n", encoding="utf-8")

    def _decline(_prompt: str) -> str:
        return "n"

    monkeypatch.setattr("builtins.input", _decline)
    _offer_git_setup(tmp_path, (tmp_path / "AGENTS.md",), interactive=True)
    out = capsys.readouterr().out
    assert "git add AGENTS.md && git commit -m" in out
    assert _porcelain(tmp_path) != ""  # nothing committed


def test_offer_git_setup_scaffold_committed_but_tree_dirty_elsewhere(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # Re-running init in a repo where the scaffold is ALREADY committed but the
    # worktree is dirty for an unrelated reason must not attempt a path-limited
    # commit of the (unchanged) scaffold paths -- that fails "nothing to commit"
    # and used to print a false "commit failed" and a remediation that also fails.
    _existing_repo(tmp_path, monkeypatch)
    scaffold = (tmp_path / "AGENTS.md", tmp_path / ".gitignore")
    scaffold[0].write_text("# AGENTS\n", encoding="utf-8")
    scaffold[1].write_text(".env\n", encoding="utf-8")
    _offer_git_setup(tmp_path, scaffold, interactive=False)  # first: commits scaffold
    (tmp_path / "README.md").write_text("wip edit\n", encoding="utf-8")  # unrelated dirt

    _offer_git_setup(tmp_path, scaffold, interactive=False)  # second: must be a no-op
    out = capsys.readouterr().out
    assert "commit failed" not in out
    assert _porcelain(tmp_path) == " M README.md\n"  # unrelated dirt untouched, not swept


def test_offer_git_setup_existing_clean_repo_is_silent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # Scaffold already committed: no prompt, no output, nothing to do.
    _existing_repo(tmp_path, monkeypatch)
    _offer_git_setup(tmp_path, (tmp_path / "README.md",), interactive=True)
    assert capsys.readouterr().out == ""


def test_init_refuses_without_tty_or_yes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # `echo n | agent6 init` used to take every default and write files.
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    rc = main(["init"])
    assert rc == 2
    assert "ERROR: no input" in capsys.readouterr().err
    assert not (tmp_path / "AGENTS.md").exists()
    assert not (tmp_path / ".gitignore").exists()


def test_an_unanswerable_run_creates_no_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A refused start must leave nothing behind, or it poisons its own id.

    The refusal names the fix (`--auto-approve`); applying it to the same
    `--session-id` then answered "already exists, use resume", and resume found
    no snapshot -- the id was unusable. The dir also sat in `agent6 sessions`
    forever as a run that never ran. Refusing before anything is created is the
    whole fix, so the pin is on the absence.
    """
    from agent6.config.layer import resolved_state_dir

    init_repo(tmp_path)
    for key, value in (("user.email", "t@example.com"), ("user.name", "t")):
        subprocess.run(["git", "-C", str(tmp_path), "config", key, value], check=True)
    (tmp_path / "agent6.toml").write_text(
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
    # Committed, so the dirty-tree refusal (which fires FIRST in the old order)
    # cannot stand in for the one under test.
    subprocess.run(["git", "-C", str(tmp_path), "add", "-A"], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "commit", "-qm", "seed"], check=True)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    monkeypatch.delenv("AGENT6_DETACHED_AWAY", raising=False)
    monkeypatch.chdir(tmp_path)

    rc = main(
        [
            "--config",
            str(tmp_path / "agent6.toml"),
            "run",
            "do a thing",
            "--session-id",
            "picked-id",
        ]
    )

    assert rc == 2
    assert "needs someone to answer" in capsys.readouterr().err
    assert not (resolved_state_dir(tmp_path) / "sessions" / "runs" / "picked-id").exists()


def test_init_never_commits_the_operators_own_edits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """`agent6 init` committed AGENTS.md by path, so an edit the operator had
    in flight landed inside "chore: scaffold agent6 config" -- and the summary
    line named AGENTS.md even when init had left it untouched."""
    repo = tmp_path / "repo"
    repo.mkdir()
    init_repo(repo)
    (repo / "AGENTS.md").write_text("theirs v1\n", encoding="utf-8")
    (repo / ".gitignore").write_text("x\n", encoding="utf-8")
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
    subprocess.run(
        ["git", "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "seed"],
        cwd=repo,
        check=True,
    )
    (repo / "AGENTS.md").write_text("theirs v2, still being written\n", encoding="utf-8")

    monkeypatch.chdir(repo)
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    assert main(["init", "--yes"]) == 0

    out = capsys.readouterr().out
    assert "committed the agent6 scaffold (.gitignore)" in out
    assert "left uncommitted (already edited): AGENTS.md" in out
    head = subprocess.run(
        ["git", "show", "--name-only", "--format=", "HEAD"],
        cwd=repo,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()
    assert head == [".gitignore"], head
    assert (repo / "AGENTS.md").read_text(encoding="utf-8") == "theirs v2, still being written\n"


def test_init_in_a_fresh_repo_leaves_the_operators_files_out(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The non-repo path runs `git init` and commits the scaffold, where every
    file is untracked and therefore looks like agent6's. An AGENTS.md the
    operator wrote before ever running agent6 is not."""
    repo = tmp_path / "fresh"
    repo.mkdir()
    # Carries a verify section already, so init leaves the file untouched.
    (repo / "AGENTS.md").write_text(
        "theirs\n\n## Verify command\n\n```bash\npytest\n```\n", encoding="utf-8"
    )
    monkeypatch.chdir(repo)
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)

    def _yes(*_a: object, **_k: object) -> bool:
        return True

    monkeypatch.setattr("agent6.ui.cli.init_cmds._ask", _yes)
    monkeypatch.setattr("agent6.init._ask", _yes)
    assert main(["init"]) == 0

    head = subprocess.run(
        ["git", "show", "--name-only", "--format=", "HEAD"],
        cwd=repo,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()
    assert head == [".gitignore"], head
    assert (repo / "AGENTS.md").read_text(encoding="utf-8").startswith("theirs\n")


def test_init_does_not_call_a_committed_scaffold_uncommitted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """AGENTS.md committed before init and untouched since: init leaves it
    alone, and said "left uncommitted (already edited)" over a clean file."""
    repo = tmp_path / "repo"
    repo.mkdir()
    init_repo(repo)
    (repo / "AGENTS.md").write_text("theirs v1\n", encoding="utf-8")
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
    subprocess.run(
        ["git", "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "seed"],
        cwd=repo,
        check=True,
    )

    monkeypatch.chdir(repo)
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    assert main(["init", "--yes"]) == 0

    out = capsys.readouterr().out
    assert "committed the agent6 scaffold (.gitignore)" in out
    assert "left uncommitted" not in out
