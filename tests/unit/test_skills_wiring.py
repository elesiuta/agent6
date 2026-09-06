# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Skills wiring: the <skills> system-prompt block and its assembly hook."""

from __future__ import annotations

from pathlib import Path

from agent6.config import Config, load_config
from agent6.skills import ResolvedSkills, Skill
from agent6.types import RepoSummary
from agent6.workflows._prompt_blocks import build_system_prompt, skills_block

_VALID_TOML = """
[agent6]
config_version = 1
[providers.anthropic]
api_format = "anthropic"
api_key_env = "ANTHROPIC_API_KEY"
[models.worker]
provider = "anthropic"
model = "x"
[models.reviewer]
provider = "anthropic"
model = "x"
"""


def _config(tmp_path: Path, extra: str = "") -> Config:
    p = tmp_path / "agent6.toml"
    p.write_text(_VALID_TOML + extra, encoding="utf-8")
    return load_config(p)


def _repo(tmp_path: Path) -> RepoSummary:
    return RepoSummary(
        root=tmp_path,
        branch="main",
        head_sha="0" * 40,
        file_count=0,
        top_level=(),
        agents_md="",
        recent_log="",
    )


def _skill(name: str, description: str = "Use when testing.", text: str = "") -> Skill:
    body = text or f"---\nname: {name}\ndescription: {description}\n---\n\nBody of {name}.\n"
    return Skill(name=name, description=description, dir=Path(f"/skills/{name}"), text=body)


def _resolved(enabled: tuple[Skill, ...] = (), always: tuple[Skill, ...] = ()) -> ResolvedSkills:
    return ResolvedSkills(enabled=enabled, always=always, warnings=())


class TestSkillsBlock:
    def test_empty_renders_nothing(self) -> None:
        assert skills_block(_resolved()) == ""

    def test_index_lists_name_and_description(self) -> None:
        block = skills_block(_resolved(enabled=(_skill("tidy", "Use when output is verbose."),)))
        assert block.startswith("<skills>")
        assert block.rstrip().endswith("</skills>")
        assert "- tidy — Use when output is verbose." in block
        assert "use_skill" in block  # loading guidance

    def test_always_injects_full_text(self) -> None:
        sk = _skill("caveman", text="---\nname: caveman\ndescription: d\n---\n\nGRUNT RULES\n")
        block = skills_block(_resolved(always=(sk,)))
        assert '<skill name="caveman">' in block
        assert "GRUNT RULES" in block
        # an always skill is not ALSO indexed
        assert "- caveman —" not in block

    def test_long_description_clipped(self) -> None:
        block = skills_block(_resolved(enabled=(_skill("big", "words " * 100),)))
        line = next(ln for ln in block.splitlines() if ln.startswith("- big"))
        assert len(line) <= 220

    def test_oversized_always_text_clipped(self) -> None:
        sk = _skill("huge", text="x" * 60_000)
        block = skills_block(_resolved(always=(sk,)))
        assert len(block) < 40_000
        assert "[clipped]" in block

    def test_index_bounded_with_elision_note(self) -> None:
        many = tuple(_skill(f"skill-{i:02d}") for i in range(400))
        block = skills_block(_resolved(enabled=many))
        assert len(block) < 12_000
        assert "elided" in block


class TestBuildSystemPromptSkills:
    def test_skills_part_appended_in_run_mode(self, tmp_path: Path) -> None:
        out = build_system_prompt(
            config=_config(tmp_path),
            repo=_repo(tmp_path),
            mode="run",
            skills=_resolved(enabled=(_skill("tidy"),)),
        )
        assert "<skills>" in out
        assert "- tidy —" in out

    def test_no_skills_no_block(self, tmp_path: Path) -> None:
        out = build_system_prompt(
            config=_config(tmp_path),
            repo=_repo(tmp_path),
            mode="run",
            skills=None,
        )
        assert "<skills>" not in out

    def test_skills_survive_system_prompt_file_override(self, tmp_path: Path) -> None:
        override = tmp_path / "base.md"
        override.write_text("CUSTOM BASE\n", encoding="utf-8")
        cfg = _config(tmp_path, f'[prompt]\nsystem_prompt_file = "{override}"\n')
        out = build_system_prompt(
            config=cfg,
            repo=_repo(tmp_path),
            mode="run",
            skills=_resolved(enabled=(_skill("tidy"),)),
        )
        assert out.startswith("CUSTOM BASE")
        assert "<skills>" in out


# --- use_skill tool -----------------------------------------------------------


def _skills_env(tmp_path: Path, monkeypatch: object) -> Path:
    """Point the installed-skills dir at an isolated tmp location."""
    import pytest

    assert isinstance(monkeypatch, pytest.MonkeyPatch)
    data = tmp_path / "data"
    monkeypatch.setenv("XDG_DATA_HOME", str(data))
    return data / "agent6" / "skills"


def _install(skills_dir: Path, name: str, text: str = "") -> Path:
    d = skills_dir / name
    d.mkdir(parents=True)
    body = text or f"---\nname: {name}\ndescription: Use when testing {name}.\n---\n\nDo {name}.\n"
    (d / "SKILL.md").write_text(body, encoding="utf-8")
    return d


class TestUseSkillTool:
    def test_serves_skill_md(self, tmp_path: Path, monkeypatch: object) -> None:
        from agent6.tools.dispatch import ToolDispatcher

        sd = _skills_env(tmp_path, monkeypatch)
        _install(sd, "tidy")
        d = ToolDispatcher(root=tmp_path, config=_config(tmp_path))
        out = d.dispatch("use_skill", {"name": "tidy"}).to_wire()
        assert out["skill"] == "tidy"
        assert out["file"] == "SKILL.md"
        assert "Do tidy." in out["content"]

    def test_serves_supplementary_file(self, tmp_path: Path, monkeypatch: object) -> None:
        from agent6.tools.dispatch import ToolDispatcher

        sd = _skills_env(tmp_path, monkeypatch)
        skill_dir = _install(sd, "tidy")
        (skill_dir / "references").mkdir()
        (skill_dir / "references" / "extra.md").write_text("EXTRA CONTENT\n", encoding="utf-8")
        d = ToolDispatcher(root=tmp_path, config=_config(tmp_path))
        out = d.dispatch("use_skill", {"name": "tidy", "file": "references/extra.md"}).to_wire()
        assert out["content"] == "EXTRA CONTENT\n"

    def test_traversal_refused(self, tmp_path: Path, monkeypatch: object) -> None:
        from agent6.tools.dispatch import ToolDispatcher
        from agent6.tools.errors import ToolError

        sd = _skills_env(tmp_path, monkeypatch)
        _install(sd, "tidy")
        (tmp_path / "data" / "secret.txt").write_text("outside\n", encoding="utf-8")
        d = ToolDispatcher(root=tmp_path, config=_config(tmp_path))
        import pytest

        with pytest.raises(ToolError, match="escapes"):
            d.dispatch("use_skill", {"name": "tidy", "file": "../secret.txt"})

    def test_symlink_out_refused(self, tmp_path: Path, monkeypatch: object) -> None:
        from agent6.tools.dispatch import ToolDispatcher
        from agent6.tools.errors import ToolError

        sd = _skills_env(tmp_path, monkeypatch)
        skill_dir = _install(sd, "tidy")
        outside = tmp_path / "outside.txt"
        outside.write_text("outside\n", encoding="utf-8")
        (skill_dir / "link.md").symlink_to(outside)
        d = ToolDispatcher(root=tmp_path, config=_config(tmp_path))
        import pytest

        with pytest.raises(ToolError, match="escapes"):
            d.dispatch("use_skill", {"name": "tidy", "file": "link.md"})

    def test_unknown_skill_lists_available(self, tmp_path: Path, monkeypatch: object) -> None:
        from agent6.tools.dispatch import ToolDispatcher
        from agent6.tools.errors import ToolError

        sd = _skills_env(tmp_path, monkeypatch)
        _install(sd, "tidy")
        d = ToolDispatcher(root=tmp_path, config=_config(tmp_path))
        import pytest

        with pytest.raises(ToolError, match="tidy"):
            d.dispatch("use_skill", {"name": "ghost"})

    def test_disabled_skill_not_servable(self, tmp_path: Path, monkeypatch: object) -> None:
        from agent6.tools.dispatch import ToolDispatcher
        from agent6.tools.errors import ToolError

        sd = _skills_env(tmp_path, monkeypatch)
        _install(sd, "tidy")
        cfg = _config(tmp_path, '[skills.state]\ntidy = "disabled"\n')
        d = ToolDispatcher(root=tmp_path, config=cfg)
        import pytest

        with pytest.raises(ToolError, match="unknown or disabled"):
            d.dispatch("use_skill", {"name": "tidy"})

    def test_run_mode_only(self, tmp_path: Path, monkeypatch: object) -> None:
        from agent6.tools.dispatch import ToolDispatcher
        from agent6.tools.errors import ToolError

        sd = _skills_env(tmp_path, monkeypatch)
        _install(sd, "tidy")
        d = ToolDispatcher(root=tmp_path, config=_config(tmp_path), mode="plan")
        import pytest

        with pytest.raises(ToolError, match="not available in plan mode"):
            d.dispatch("use_skill", {"name": "tidy"})


class TestUseSkillGating:
    def test_hidden_when_no_skills(self, tmp_path: Path, monkeypatch: object) -> None:
        from agent6.tools.dispatch import ToolDispatcher
        from agent6.workflows._toolset import tool_definitions

        _skills_env(tmp_path, monkeypatch)  # empty data dir
        d = ToolDispatcher(root=tmp_path, config=_config(tmp_path))
        assert "use_skill" not in {t.name for t in tool_definitions(d, mode="run")}

    def test_exposed_when_skills_installed(self, tmp_path: Path, monkeypatch: object) -> None:
        from agent6.tools.dispatch import ToolDispatcher
        from agent6.workflows._toolset import tool_definitions

        sd = _skills_env(tmp_path, monkeypatch)
        _install(sd, "tidy")
        d = ToolDispatcher(root=tmp_path, config=_config(tmp_path))
        assert "use_skill" in {t.name for t in tool_definitions(d, mode="run")}

    def test_hidden_when_subsystem_disabled(self, tmp_path: Path, monkeypatch: object) -> None:
        from agent6.tools.dispatch import ToolDispatcher
        from agent6.workflows._toolset import tool_definitions

        sd = _skills_env(tmp_path, monkeypatch)
        _install(sd, "tidy")
        cfg = _config(tmp_path, "[skills]\nenabled = false\n")
        d = ToolDispatcher(root=tmp_path, config=cfg)
        assert "use_skill" not in {t.name for t in tool_definitions(d, mode="run")}

    def test_never_exposed_outside_run(self, tmp_path: Path, monkeypatch: object) -> None:
        from agent6.tools.dispatch import ToolDispatcher
        from agent6.workflows._toolset import tool_definitions

        sd = _skills_env(tmp_path, monkeypatch)
        _install(sd, "tidy")
        d = ToolDispatcher(root=tmp_path, config=_config(tmp_path))
        for mode in ("plan", "ask", "machine", "agent"):
            assert "use_skill" not in {t.name for t in tool_definitions(d, mode=mode)}, mode


class TestWorkflowLoadSkills:
    def test_run_mode_loads_from_dispatcher(self, tmp_path: Path, monkeypatch: object) -> None:
        from agent6.tools.dispatch import ToolDispatcher
        from agent6.workflows.loop import Workflow

        sd = _skills_env(tmp_path, monkeypatch)
        _install(sd, "tidy")
        cfg = _config(tmp_path)
        d = ToolDispatcher(root=tmp_path, config=cfg)
        wf = Workflow(root=tmp_path, config=cfg, provider=None, dispatcher=d, logger=lambda _: None)  # type: ignore[arg-type]
        resolved = wf._load_skills()  # pyright: ignore[reportPrivateUsage]
        assert resolved is not None
        assert [s.name for s in resolved.enabled] == ["tidy"]

    def test_non_run_mode_returns_none(self, tmp_path: Path, monkeypatch: object) -> None:
        from agent6.tools.dispatch import ToolDispatcher
        from agent6.workflows.loop import Workflow

        sd = _skills_env(tmp_path, monkeypatch)
        _install(sd, "tidy")
        cfg = _config(tmp_path)
        d = ToolDispatcher(root=tmp_path, config=cfg, mode="plan")
        wf = Workflow(
            root=tmp_path,
            config=cfg,
            provider=None,  # type: ignore[arg-type]
            dispatcher=d,
            logger=lambda _: None,
            mode="plan",
            plan_output_path=tmp_path / "plan.md",
        )
        assert wf._load_skills() is None  # pyright: ignore[reportPrivateUsage]
