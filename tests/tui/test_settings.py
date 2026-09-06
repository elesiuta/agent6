# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""UI-only preferences store (ui.toml), separate from the agent config."""

from __future__ import annotations

from pathlib import Path

import pytest

from agent6.ui.tui.settings import DEFAULT_THEME, get_theme, load_ui_settings, save_theme


@pytest.fixture
def cfg(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    home = tmp_path / "agent6"
    home.mkdir()
    return home


def test_theme_roundtrips_in_own_file(cfg: Path) -> None:
    assert get_theme() == DEFAULT_THEME  # default when ui.toml is absent
    save_theme("nord")
    assert get_theme() == "nord"
    # Lands in its own file (a sibling of config.toml), not the agent config.
    ui = cfg / "ui.toml"
    assert ui.is_file()
    assert "theme" in ui.read_text(encoding="utf-8")
    assert not (cfg / "config.toml").exists()


def test_corrupt_or_missing_file_degrades_to_default(cfg: Path) -> None:
    (cfg / "ui.toml").write_text("this is [ not valid toml", encoding="utf-8")
    assert load_ui_settings() == {}  # never raises
    assert get_theme() == DEFAULT_THEME


def test_save_preserves_other_keys(cfg: Path) -> None:
    (cfg / "ui.toml").write_text('[ui]\ntheme = "nord"\nshow_x = true\n', encoding="utf-8")
    save_theme("dracula")
    data = load_ui_settings()["ui"]
    assert data["theme"] == "dracula"
    assert data["show_x"] is True  # unrelated keys survive the rewrite


@pytest.mark.skipif(__import__("os").name == "nt", reason="POSIX symlinks")
def test_save_does_not_follow_a_planted_tmp_symlink(cfg: Path) -> None:
    """The old fixed `ui.toml.tmp` + write_text followed a planted symlink; since
    the save chowns back to the real user it can run under sudo, making that an
    arbitrary-file truncate-as-root primitive. atomic_write's mkstemp temp is
    unpredictable, so a planted `ui.toml.tmp` is simply ignored."""
    secret = cfg / "root_secret"
    secret.write_text("do-not-truncate", encoding="utf-8")
    (cfg / "ui.toml.tmp").symlink_to(secret)
    save_theme("nord")
    assert secret.read_text(encoding="utf-8") == "do-not-truncate"  # untouched
    assert get_theme() == "nord"  # the real save still landed


def test_a_control_character_in_a_ui_value_round_trips(tmp_path: Path) -> None:
    """The TUI's own TOML writer escaped only backslash and quote, the drift
    `toml_basic_string`'s docstring forbids: a newline in a value wrote a file
    that failed to parse on read after the write reported success."""
    import tomllib

    from agent6.ui.tui.settings import _render_ui_toml  # pyright: ignore[reportPrivateUsage]

    text = _render_ui_toml({"ui": {"copy_method": "osc52\nrogue", "theme": "dark"}})
    assert tomllib.loads(text)["ui"]["copy_method"] == "osc52\nrogue"
