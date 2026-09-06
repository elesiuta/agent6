# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""`agent6 review` says which of its flags it cannot honour."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import pytest

from agent6.config import ConfigError
from agent6.ui.cli import cli_main


def test_personas_without_reviewers_is_said_to_be_ignored(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """`--personas` is read only under `--reviewers N`: alone it ran the single
    freeform review with the named seats silently dropped. The sibling
    `model` command prints a note for a flag it cannot use; so does this."""
    subprocess.run(["git", "init", "-q", "-b", "main", str(tmp_path)], check=True)
    monkeypatch.chdir(tmp_path)

    def stop(*_a: object, **_k: object) -> object:
        raise ConfigError("stop here")

    monkeypatch.setattr("agent6.ui.cli.review_cmds.load_effective", stop)
    rc = cli_main(["review", "--personas", "security,tests"])
    err = capsys.readouterr().err
    assert rc == 2 and "stop here" in err
    assert "note: --personas ignored (no --reviewers N" in err

    rc = cli_main(["review", "--personas", "security,tests", "--reviewers", "2"])
    assert rc == 2 and "--personas ignored" not in capsys.readouterr().err


def test_personas_under_configured_seats_is_said_to_be_ignored(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """`[review].seats` names the roster outright, as the flag's help says;
    the flag beside it was dropped in silence."""
    from types import SimpleNamespace

    from agent6.config import Config

    subprocess.run(["git", "init", "-q", "-b", "main", str(tmp_path)], check=True)
    monkeypatch.chdir(tmp_path)
    cfg = Config.model_validate({"review": {"seats": ["security@openrouter/some-model"]}})

    def loaded(*_a: object, **_k: object) -> SimpleNamespace:
        return SimpleNamespace(config=cfg)

    monkeypatch.setattr("agent6.ui.cli.review_cmds.load_effective", loaded)
    rc = cli_main(["review", "--personas", "tests", "--reviewers", "2"])
    err = capsys.readouterr().err
    assert rc == 2 and "note: --personas ignored ([review].seats names the roster)." in err


def _two_commits(repo: Path) -> tuple[str, str]:
    """Commit A defines `f(x)` with a caller `f(1)`; commit B widens the
    signature and updates the caller. Returns (A, B), checked out at A."""

    def git(*args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=repo, check=True, capture_output=True, text=True
        ).stdout.strip()

    git("init", "-q", "-b", "main")
    git("config", "user.email", "t@example.com")
    git("config", "user.name", "t")
    (repo / "lib.py").write_text("def f(x):\n    return x\n", encoding="utf-8")
    (repo / "caller.py").write_text("from lib import f\n\nprint(f(1))\n", encoding="utf-8")
    git("add", "lib.py", "caller.py")
    git("commit", "-qm", "A")
    a = git("rev-parse", "HEAD")
    (repo / "lib.py").write_text("def f(x, y):\n    return x + y\n", encoding="utf-8")
    (repo / "caller.py").write_text("from lib import f\n\nprint(f(1, 2))\n", encoding="utf-8")
    git("add", "lib.py", "caller.py")
    git("commit", "-qm", "B")
    b = git("rev-parse", "HEAD")
    git("checkout", "-q", a)
    return a, b


def _explore_review(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, base: str, head: str
) -> tuple[int, dict[str, Any]]:
    """`agent6 review --reviewers 1` under `review.tier = "explore"` with one
    fake seat whose panel reads `caller.py` the way the explore prompt tells it
    to; returns the exit code and what the seat read."""
    from types import SimpleNamespace

    from agent6.config import Config
    from agent6.ui.cli import review_cmds
    from agent6.workflows._panel import PanelResult
    from agent6.workflows._review import ReviewSeat

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    cfg = Config.model_validate({"review": {"tier": "explore"}})
    seen: dict[str, Any] = {}

    def _fake_effective(*_a: object, **_k: object) -> SimpleNamespace:
        return SimpleNamespace(config=cfg)

    def _runnable(_self: Config, _role: str) -> None:
        return None

    def _no_key_error(_cfg: Config) -> None:
        return None

    def _fake_seats(_cfg: Config, **_k: Any) -> list[ReviewSeat]:
        return [
            ReviewSeat(
                persona="correctness",
                model="fake",
                provider=None,  # pyright: ignore[reportArgumentType]
                tier="explore",
            )
        ]

    def _fake_panel(_seats: Any, _ctx: Any, **kw: Any) -> PanelResult:
        seen["read_file"] = kw["dispatch"]("read_file", {"path": "caller.py"}).content
        return PanelResult(
            panel_id="cli",
            decision="advisory",
            blocked=False,
            merged_findings=(),
            per_seat=(),
            n_block=0,
            n_abstain=0,
        )

    monkeypatch.setattr(review_cmds, "load_effective", _fake_effective)
    monkeypatch.setattr(Config, "require_runnable", _runnable)
    monkeypatch.setattr(review_cmds, "check_provider_keys", _no_key_error)
    monkeypatch.setattr(review_cmds, "build_review_seats", _fake_seats)
    monkeypatch.setattr(review_cmds, "run_panel", _fake_panel)
    rc = review_cmds._cmd_review(  # pyright: ignore[reportPrivateUsage]
        None, base=base, head=head, paths=(), reviewers=1
    )
    return rc, seen


def test_explore_tier_gates_on_the_head_being_the_checkout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """`review.tier = "explore"` hands a seat read-only tools over the CHECKOUT,
    so reviewing a `--head` that is not checked out fed it file contents the
    diff contradicts: the diff said the caller became `f(1, 2)` while
    `read_file` returned the old `f(1)`, the false break the explore prompt
    tells a seat to BLOCK on. Both directions: the ordinary `--head HEAD` on
    the checked-out commit still runs its seat."""
    base, head = _two_commits(tmp_path)
    rc, seen = _explore_review(tmp_path, monkeypatch, base=base, head=head)
    err = capsys.readouterr().err
    assert seen == {}, f"an explore seat read the checkout, not --head: {seen}"
    assert rc == 2
    assert "--head" in err and "explore" in err

    subprocess.run(["git", "checkout", "-q", head], cwd=tmp_path, check=True)
    (tmp_path / "scratch.log").write_text("build output\n", encoding="utf-8")  # untracked
    rc, seen = _explore_review(tmp_path, monkeypatch, base=base, head="HEAD")
    assert rc == 0
    assert seen == {"read_file": "from lib import f\n\nprint(f(1, 2))\n"}


def test_explore_tier_refuses_a_dirty_checkout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The default `--head HEAD` names the checked-out commit whatever the tree
    holds, so an uncommitted edit to a file the `base..HEAD` diff describes fed
    a seat the same false break a wrong `--head` does."""
    base, head = _two_commits(tmp_path)
    subprocess.run(["git", "checkout", "-q", head], cwd=tmp_path, check=True)
    (tmp_path / "caller.py").write_text("from lib import f\n\nprint(f(1))\n", encoding="utf-8")
    rc, seen = _explore_review(tmp_path, monkeypatch, base=base, head="HEAD")
    err = capsys.readouterr().err
    assert seen == {}, f"an explore seat read a dirty tree the diff contradicts: {seen}"
    assert rc == 2
    assert "--head" in err and "explore" in err
