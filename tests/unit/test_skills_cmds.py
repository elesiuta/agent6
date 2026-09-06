# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""`agent6 skills` CLI: install (file/dir/git), update, list, state, remove."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from agent6.config.write import set_config_value
from agent6.errors import OperatorError
from agent6.ui.cli.skills_cmds import (
    _cmd_skills_disable,  # pyright: ignore[reportPrivateUsage]
    _cmd_skills_enable,  # pyright: ignore[reportPrivateUsage]
    _cmd_skills_install,  # pyright: ignore[reportPrivateUsage]
    _cmd_skills_list,  # pyright: ignore[reportPrivateUsage]
    _cmd_skills_remove,  # pyright: ignore[reportPrivateUsage]
    _cmd_skills_update,  # pyright: ignore[reportPrivateUsage]
    resolved_skill_names_for_completion,
)

SKILL_MD = """---
name: {name}
description: Use when testing {name}.
---

Body of {name}.
"""


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Hermetic data/config/state homes; returns the tmp root."""
    monkeypatch.setenv("AGENT6_DATA_HOME", str(tmp_path / "data"))
    monkeypatch.setenv("AGENT6_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.setenv("AGENT6_STATE_HOME", str(tmp_path / "state"))
    monkeypatch.chdir(tmp_path / "cwd" if (tmp_path / "cwd").exists() else tmp_path)
    return tmp_path


def _write_skill_file(path: Path, name: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(SKILL_MD.format(name=name), encoding="utf-8")
    return path


def _installed(tmp_path: Path, name: str) -> Path:
    return tmp_path / "data" / "skills" / name


class TestInstall:
    def test_local_skill_md_file(self, env: Path, capsys: pytest.CaptureFixture[str]) -> None:
        src = _write_skill_file(env / "src" / "SKILL.md", "tidy")
        assert _cmd_skills_install(str(src), force=False) == 0
        assert (_installed(env, "tidy") / "SKILL.md").is_file()
        assert (_installed(env, "tidy") / ".origin.toml").is_file()
        out = capsys.readouterr().out
        assert "Installed tidy" in out
        assert "Use when testing tidy." in out

    def test_traversing_frontmatter_name_refused(self, env: Path) -> None:
        """The install target is `<skills>/<name>` (and, under --force, an rmtree
        target). A SKILL.md `name` with `..` or an absolute path from an untrusted
        source must be refused, not used verbatim to write/delete outside the dir."""
        outside = env / "precious"
        outside.mkdir()
        (outside / "keep.txt").write_text("do not delete", encoding="utf-8")
        src = env / "src" / "SKILL.md"
        src.parent.mkdir(parents=True)
        src.write_text(
            "---\nname: ../../precious\ndescription: evil traversal skill.\n---\nbody\n",
            encoding="utf-8",
        )
        with pytest.raises(OperatorError, match="invalid skill name"):
            _cmd_skills_install(str(src), force=True)
        assert (outside / "keep.txt").read_text() == "do not delete"  # untouched

    def test_local_repo_with_skills_dir(self, env: Path) -> None:
        repo = env / "pack"
        _write_skill_file(repo / "skills" / "aa" / "SKILL.md", "aa")
        _write_skill_file(repo / "skills" / "bb" / "SKILL.md", "bb")
        (repo / "skills" / "aa" / "references").mkdir()
        (repo / "skills" / "aa" / "references" / "x.md").write_text("REF\n", encoding="utf-8")
        assert _cmd_skills_install(str(repo), force=False) == 0
        assert (_installed(env, "aa") / "references" / "x.md").read_text() == "REF\n"
        assert (_installed(env, "bb") / "SKILL.md").is_file()

    def test_git_repo_install(self, env: Path) -> None:
        repo = env / "gitpack"
        _write_skill_file(repo / "skills" / "gg" / "SKILL.md", "gg")
        subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
        subprocess.run(["git", "config", "user.email", "t@t"], cwd=repo, check=True)
        subprocess.run(["git", "config", "user.name", "t"], cwd=repo, check=True)
        subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
        subprocess.run(["git", "commit", "-qm", "seed"], cwd=repo, check=True)
        # a git URL that is not an existing local path exercises the clone path;
        # file:// URLs hit git's local transport, no network involved
        assert _cmd_skills_install(f"file://{repo}", force=False) == 0
        assert (_installed(env, "gg") / "SKILL.md").is_file()
        origin = (_installed(env, "gg") / ".origin.toml").read_text()
        assert 'kind = "git"' in origin
        assert "source_sha" in origin

    def test_a_symlinked_skill_file_is_not_installed_as_its_target(self, env: Path) -> None:
        """`copytree` defaults to `symlinks=False`, which copies the CONTENT a
        link points at. A skill shipping `reference.md -> secrets.toml` then
        installs as a real file holding the operator's provider keys, and
        `use_skill` serves it to the model: the containment check that refuses
        a link has nothing left to catch once install dereferenced it. A
        directory link is the same hole one level up."""
        secrets = env / "config" / "agent6" / "secrets.toml"
        secrets.parent.mkdir(parents=True)
        secrets.write_text('api_key = "sk-OPERATOR-SECRET"\n', encoding="utf-8")
        src = env / "src"
        _write_skill_file(src / "SKILL.md", "leaky")
        (src / "reference.md").symlink_to(secrets)
        (src / "refs").symlink_to(secrets.parent, target_is_directory=True)

        assert _cmd_skills_install(str(src), force=False) == 0

        installed = _installed(env, "leaky")
        real_files = [p for p in installed.rglob("*") if p.is_file() and not p.is_symlink()]
        leaked = [
            str(p.relative_to(installed)) for p in real_files if "sk-OPERATOR" in p.read_text()
        ]
        assert not leaked, f"install copied the operator's secrets into the skill: {leaked}"
        # The links survive, so use_skill's containment check has something to refuse.
        assert (installed / "reference.md").is_symlink()
        assert (installed / "refs").is_symlink()

    def test_conflict_refused_then_forced(self, env: Path) -> None:
        src = _write_skill_file(env / "src" / "SKILL.md", "tidy")
        assert _cmd_skills_install(str(src), force=False) == 0
        with pytest.raises(OperatorError, match="already installed"):
            _cmd_skills_install(str(src), force=False)
        assert _cmd_skills_install(str(src), force=True) == 0

    def test_missing_frontmatter_rejected(self, env: Path) -> None:
        bad = env / "bad.md"
        bad.write_text("no frontmatter\n", encoding="utf-8")
        with pytest.raises(OperatorError, match="frontmatter"):
            _cmd_skills_install(str(bad), force=False)

    def test_unreadable_source_refuses_in_the_shared_voice(
        self, env: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """`skills install` kept its own except arm and `SKILLS ERROR:` voice
        after the one-error-boundary commit deleted that shape everywhere else.
        An unreadable operator source refuses through the boundary: `ERROR:` at
        exit 2, one voice, no crash report."""
        from agent6.ui.cli import cli_main

        src = _write_skill_file(env / "src" / "SKILL.md", "tidy")
        src.chmod(0o000)
        try:
            assert cli_main(["skills", "install", str(src)]) == 2
        finally:
            src.chmod(0o600)
        err = capsys.readouterr().err
        assert err.startswith("ERROR: could not read")
        assert "SKILLS ERROR" not in err


class TestUpdate:
    def test_update_reports_changed_and_unchanged(
        self, env: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        src = _write_skill_file(env / "src" / "SKILL.md", "tidy")
        assert _cmd_skills_install(str(src), force=False) == 0
        assert _cmd_skills_update("tidy") == 0
        out = capsys.readouterr().out
        assert "tidy" in out and "unchanged" in out
        src.write_text(SKILL_MD.format(name="tidy") + "\nMore.\n", encoding="utf-8")
        assert _cmd_skills_update("tidy") == 0
        assert "updated" in capsys.readouterr().out
        assert "More." in (_installed(env, "tidy") / "SKILL.md").read_text()

    def test_update_unknown_name(self, env: Path) -> None:
        with pytest.raises(OperatorError, match="not installed"):
            _cmd_skills_update("ghost")

    def test_update_skips_when_local_source_gone(
        self, env: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        src = _write_skill_file(env / "src" / "SKILL.md", "tidy")
        assert _cmd_skills_install(str(src), force=False) == 0
        src.unlink()  # the source file the operator installed from is deleted
        capsys.readouterr()
        assert _cmd_skills_update("tidy") == 0  # a vanished source is a skip, not an abort
        out = capsys.readouterr().out
        assert "skipped" in out and "gone from origin" in out
        assert (_installed(env, "tidy") / "SKILL.md").is_file()  # left intact

    def test_update_dir_install_reinstalls(
        self, env: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        repo = env / "pack"
        _write_skill_file(repo / "SKILL.md", "dd")
        assert _cmd_skills_install(str(repo), force=False) == 0
        capsys.readouterr()
        assert _cmd_skills_update("dd") == 0  # dir-kind origin must not read_text a directory
        assert "unchanged" in capsys.readouterr().out

    def test_update_repo_style_dir_install(
        self, env: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        # a repo with skills/*/SKILL.md records the repo root as each skill's
        # origin; update must find the skill in its subdir, not the root.
        repo = env / "pack"
        _write_skill_file(repo / "skills" / "aa" / "SKILL.md", "aa")
        _write_skill_file(repo / "skills" / "bb" / "SKILL.md", "bb")
        assert _cmd_skills_install(str(repo), force=False) == 0
        capsys.readouterr()
        assert _cmd_skills_update("") == 0
        out = capsys.readouterr().out
        assert "aa" in out and "bb" in out and "unchanged" in out
        assert "gone from origin" not in out


class TestStateCommands:
    def _install_tidy(self, env: Path) -> None:
        src = _write_skill_file(env / "src" / "SKILL.md", "tidy")
        assert _cmd_skills_install(str(src), force=False) == 0

    def test_disable_writes_global_state(self, env: Path) -> None:
        self._install_tidy(env)
        assert _cmd_skills_disable("tidy", repo=False) == 0
        cfg = (env / "config" / "config.toml").read_text()
        assert 'tidy = "disabled"' in cfg

    def test_enable_always_and_back(self, env: Path) -> None:
        self._install_tidy(env)
        assert _cmd_skills_enable("tidy", always=True, repo=False) == 0
        assert 'tidy = "always"' in (env / "config" / "config.toml").read_text()
        assert _cmd_skills_enable("tidy", always=False, repo=False) == 0
        assert "tidy" not in (env / "config" / "config.toml").read_text()

    def test_unknown_skill_refused(self, env: Path) -> None:
        with pytest.raises(OperatorError, match="unknown skill"):
            _cmd_skills_disable("ghost", repo=False)
        with pytest.raises(OperatorError, match="unknown skill"):
            _cmd_skills_enable("ghost", always=False, repo=False)

    def test_disable_over_a_headerless_state_table_errors_not_crashes(self, env: Path) -> None:
        """A hand-written inline `state` table under [skills] can't take a single
        leaf; the surgery refuses with an operator error the boundary presents,
        not a 'please report this' traceback."""
        self._install_tidy(env)
        cfg = env / "config" / "config.toml"
        cfg.parent.mkdir(parents=True, exist_ok=True)
        before = '[skills]\nstate = { tidy = "always" }\n'
        cfg.write_text(before, encoding="utf-8")

        with pytest.raises(OperatorError, match="cannot be set on its own"):
            _cmd_skills_disable("tidy", repo=False)
        assert cfg.read_text(encoding="utf-8") == before  # untouched


class TestRemoveListComplete:
    def test_remove_installed(self, env: Path) -> None:
        src = _write_skill_file(env / "src" / "SKILL.md", "tidy")
        assert _cmd_skills_install(str(src), force=False) == 0
        assert _cmd_skills_remove("tidy") == 0
        assert not _installed(env, "tidy").exists()
        with pytest.raises(OperatorError, match="not installed"):
            _cmd_skills_remove("tidy")

    def test_remove_refuses_a_traversal_name(self, env: Path) -> None:
        """The name becomes an rmtree target; a `../` name must be refused before
        any path op, not delete a sibling outside the managed skills dir."""
        (env / "data" / "skills").mkdir(parents=True, exist_ok=True)
        victim = env / "data" / "victim"
        victim.mkdir()
        (victim / "keep.txt").write_text("important", encoding="utf-8")
        with pytest.raises(OperatorError, match="invalid skill name"):
            _cmd_skills_remove("../victim")
        assert victim.is_dir()  # never touched

    def test_list_shows_state_and_origin(
        self, env: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        src = _write_skill_file(env / "src" / "SKILL.md", "tidy")
        assert _cmd_skills_install(str(src), force=False) == 0
        assert _cmd_skills_disable("tidy", repo=False) == 0
        assert _cmd_skills_list() == 0
        out = capsys.readouterr().out
        assert "tidy" in out
        assert "[disabled]" in out
        assert "Use when testing tidy." in out

    def test_completion_names(self, env: Path) -> None:
        src = _write_skill_file(env / "src" / "SKILL.md", "tidy")
        assert _cmd_skills_install(str(src), force=False) == 0
        assert resolved_skill_names_for_completion(Path.cwd()) == ["tidy"]


class TestSkillsTaskPrefix:
    def test_prefix_contains_skill_and_unknown_errors(self, env: Path) -> None:
        from agent6.config.layer import load_effective
        from agent6.ui.cli.run import _skills_task_prefix  # pyright: ignore[reportPrivateUsage]

        src = _write_skill_file(env / "src" / "SKILL.md", "tidy")
        assert _cmd_skills_install(str(src), force=False) == 0
        cfg = load_effective(Path.cwd()).config
        prefix, err = _skills_task_prefix(cfg, ("tidy",))
        assert err == ""
        assert '<skill name="tidy">' in prefix
        assert "Body of tidy." in prefix
        _, err2 = _skills_task_prefix(cfg, ("ghost",))
        assert "ghost" in err2
        assert "tidy" in err2

    def test_the_master_switch_covers_the_skill_flag_and_the_listing(
        self, env: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """`[skills].enabled` is the master switch: off means no skills
        anywhere. `--skill` resolved discovery for itself and injected the text
        regardless, and `skills list` printed the installed set with nothing
        saying no run would load any of it."""
        from agent6.config.layer import load_effective
        from agent6.ui.cli.run import _skills_task_prefix  # pyright: ignore[reportPrivateUsage]

        src = _write_skill_file(env / "src" / "SKILL.md", "tidy")
        assert _cmd_skills_install(str(src), force=False) == 0
        assert set_config_value(Path.cwd(), "skills.enabled", "false") is None
        cfg = load_effective(Path.cwd()).config

        prefix, err = _skills_task_prefix(cfg, ("tidy",))

        assert prefix == ""
        # The refusal names the switch, not "(none installed)": the skill IS
        # installed, and the listing beside it says exactly that.
        assert "skills are disabled" in err and "skills.enabled true" in err
        assert _cmd_skills_list() == 0
        assert "skills are DISABLED" in capsys.readouterr().out


class TestAtomicMultiInstall:
    def test_repo_conflict_refuses_whole_install(
        self, env: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        # zz already installed; a repo carrying aa+zz must install NOTHING.
        # The conflict is the LAST entry in sort order on purpose: with the
        # conflict first, the install aborts before reaching the sibling and a
        # pre-check narrowed to the first dir would still look atomic.
        src = _write_skill_file(env / "one" / "SKILL.md", "zz")
        assert _cmd_skills_install(str(src), force=False) == 0
        repo = env / "pack"
        _write_skill_file(repo / "skills" / "aa" / "SKILL.md", "aa")
        _write_skill_file(repo / "skills" / "zz" / "SKILL.md", "zz")
        with pytest.raises(OperatorError, match="nothing was installed"):
            _cmd_skills_install(str(repo), force=False)
        assert not _installed(env, "aa").exists()  # the pre-conflict skill too
        # --force replaces and installs both
        assert _cmd_skills_install(str(repo), force=True) == 0
        assert _installed(env, "aa").exists() and _installed(env, "zz").exists()


def test_force_reinstall_survives_a_copy_fault(env: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """--force removed the old install BEFORE copying, so a copy fault
    destroyed the good skill and left nothing (or a partial dir) behind. The
    replacement is staged beside the target and swapped in only when fully
    built; a fault leaves the old install untouched and no staging litter."""
    import shutil as _shutil

    from agent6.ui.cli import skills_cmds

    src = env / "src-skill"
    src.mkdir()
    (src / "SKILL.md").write_text(
        "---\nname: keeper\ndescription: good\n---\nold body\n", encoding="utf-8"
    )
    assert _cmd_skills_install(str(src), force=False) == 0
    installed = env / "data" / "skills" / "keeper" / "SKILL.md"
    assert "old body" in installed.read_text(encoding="utf-8")

    def boom(*a: object, **k: object) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(skills_cmds.shutil, "copytree", boom)
    with pytest.raises(OperatorError, match="could not install"):
        _cmd_skills_install(str(src), force=True)
    monkeypatch.setattr(skills_cmds.shutil, "copytree", _shutil.copytree)
    assert "old body" in installed.read_text(encoding="utf-8")  # the good install survived
    assert not list((env / "data" / "skills").glob(".staging-*"))


def test_origin_toml_round_trips_a_quoted_source(env: Path) -> None:
    """The origin was hand-built without escaping, so a quote in a source
    path produced unparseable TOML and update lost its origin."""
    from agent6.ui.cli.skills_cmds import (
        _read_origin,  # pyright: ignore[reportPrivateUsage]
        _write_origin,  # pyright: ignore[reportPrivateUsage]
    )

    skill = env / "data" / "skills" / "quoty"
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text("---\nname: quoty\ndescription: d\n---\n", encoding="utf-8")
    url = 'file:///tmp/we"ird\\path/skill'
    _write_origin(skill, url=url, kind="dir", source_sha="")
    origin = _read_origin(skill)
    assert origin is not None and origin["url"] == url


def test_repo_skill_state_honors_the_custom_state_base(
    env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """--repo skill state must write the config the effective loader READS.
    The raw path helper ignored the global [agent6].state_dir override, so a
    custom-state setup wrote the default tree, printed success, and the skill
    stayed enabled."""
    from agent6.ui.cli.skills_cmds import (
        _state_target,  # pyright: ignore[reportPrivateUsage]
    )

    custom = env / "custom-state"
    gdir = env / "config"  # AGENT6_CONFIG_HOME names the agent6 config dir itself
    gdir.mkdir(parents=True, exist_ok=True)
    (gdir / "config.toml").write_text(f'[agent6]\nstate_dir = "{custom}"\n', encoding="utf-8")
    target = _state_target(repo=True)
    assert target.is_relative_to(custom), f"wrote {target}, outside the custom base"


def test_update_follows_an_upstream_rename(env: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """A skillmd origin whose frontmatter now declares a new name is a rename:
    the skill reinstalls under the new name, the old directory goes (never two
    live copies), and the row says what happened."""
    src = _write_skill_file(env / "src" / "SKILL.md", "old-name")
    assert _cmd_skills_install(str(src), force=False) == 0
    _write_skill_file(src, "new-name")
    capsys.readouterr()
    assert _cmd_skills_update("old-name") == 0
    out = capsys.readouterr().out
    assert "renamed to new-name" in out
    assert _installed(env, "new-name").is_dir()
    assert not _installed(env, "old-name").exists()


def test_install_names_a_surviving_disabled_state(
    env: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """`skills.state.<name> = "disabled"` outlives remove; a reinstall under
    that name must not claim "Enabled and active now"."""
    src = _write_skill_file(env / "src" / "SKILL.md", "sleeper")
    assert _cmd_skills_install(str(src), force=False) == 0
    _cmd_skills_disable("sleeper", repo=False)
    assert _cmd_skills_remove("sleeper") == 0
    out = capsys.readouterr().out
    assert 'skills.state.sleeper = "disabled" remains' in out
    assert _cmd_skills_install(str(src), force=False) == 0
    out = capsys.readouterr().out
    assert "Enabled and active now" not in out
    assert 'skills.state.sleeper = "disabled"' in out


def test_enable_clears_a_state_leaf_whose_skill_is_gone(
    env: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """remove deletes the install, never the operator's config -- so the CLI
    that wrote the leaf must still be able to clear it afterwards."""
    src = _write_skill_file(env / "src" / "SKILL.md", "ghost")
    assert _cmd_skills_install(str(src), force=False) == 0
    _cmd_skills_disable("ghost", repo=False)
    assert _cmd_skills_remove("ghost") == 0
    capsys.readouterr()
    assert _cmd_skills_enable("ghost", always=False, repo=False) == 0
    out = capsys.readouterr().out
    assert "Unset skills.state.ghost" in out
    # A name with no leaf and no skill is still a typo, not cleanup.
    with pytest.raises(OperatorError, match="unknown skill"):
        _cmd_skills_enable("nonexistent", always=False, repo=False)
