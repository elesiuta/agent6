# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Config presets: a named preset injected just above the config layer that
selected it, so the preset OVERRIDES that config (a more-specific config layer
or flag still wins); most-specific preset source wins, presets never stack."""

from __future__ import annotations

from pathlib import Path

import pytest

from agent6.config import ConfigError
from agent6.config.layer import load_effective, repo_config_path_for


def _write_repo_config(repo: Path, toml: str) -> None:
    p = repo_config_path_for(repo)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(toml, encoding="utf-8")


@pytest.fixture
def repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    r = tmp_path / "repo"
    r.mkdir()
    return r


def test_preset_via_preset_field_expands_review_knobs(repo: Path) -> None:
    _write_repo_config(repo, 'preset = "ultra"\n')
    cfg = load_effective(repo).config
    assert cfg.review.trigger == "before_finish"
    assert cfg.review.seats == ("security", "correctness", "tests")
    assert cfg.review.decision == "veto"
    assert cfg.review.concurrency == 3  # seats run in parallel, not in series


def test_preset_via_flag_overrides_field(repo: Path) -> None:
    _write_repo_config(repo, 'preset = "quick"\n')
    cfg = load_effective(repo, preset="paranoid").config  # flag wins over the field
    assert len(cfg.review.seats) == 5
    assert cfg.review.tier == "explore"
    assert cfg.review.decision == "veto"
    assert cfg.review.concurrency == 5  # seats run in parallel, not in series


def test_repo_selected_preset_beats_same_layer_setting(repo: Path) -> None:
    # The preset selected by the repo's top-level `preset` is injected ABOVE the
    # repo config, so it OVERRIDES a conflicting value set in the SAME repo config.
    _write_repo_config(repo, 'preset = "ultra"\n\n[review]\ndecision = "advisory"\n')
    cfg = load_effective(repo).config
    assert cfg.review.decision == "veto"  # repo-selected preset wins
    assert cfg.review.seats == ("security", "correctness", "tests")  # rest of the preset applies


def test_custom_user_preset(repo: Path) -> None:
    _write_repo_config(
        repo,
        'preset = "myteam"\n\n[presets.myteam.review]\n'
        'trigger = "before_finish"\nconcurrency = 2\n',
    )
    cfg = load_effective(repo).config
    assert cfg.review.concurrency == 2 and cfg.review.trigger == "before_finish"


def test_only_a_flag_selected_preset_is_replayed_on_resume(repo: Path) -> None:
    """A resumed/forked leg re-applies --preset but must NOT hand a
    config-selected name back as an override: _select_preset would call it a
    flag, which outranks every config layer, so a run whose repo config beat a
    global preset came back from resume with the preset winning -- gaining a
    blocking review veto the original never had. Only the name was stamped, so
    the two cases were indistinguishable."""
    from agent6.sessions.manifest import WorkflowStamp

    assert WorkflowStamp(preset="t", preset_from_flag=True).replay_preset == "t"
    assert WorkflowStamp(preset="t").replay_preset == ""  # config-selected: re-resolves

    # Why it matters: the SAME files resolve differently when the name arrives
    # as a flag, which is exactly what the old replay did.
    _write_repo_config(repo, f'preset = "t"\n\n[review]\nconcurrency = 3\n\n{_PROFILE_T}')
    assert load_effective(repo).config.review.concurrency == 5  # repo-selected preset wins
    _write_repo_config(repo, f"[review]\nconcurrency = 3\n\n{_PROFILE_T}")
    assert load_effective(repo).config.review.concurrency == 3  # no selection: config wins
    assert load_effective(repo, None, preset="t").config.review.concurrency == 5  # as a flag


def test_user_preset_named_standard_replaces_the_builtin(repo: Path) -> None:
    """A user table named after a built-in replaces it wholesale (docs/config.md,
    and resolve_preset's own "user presets win over built-ins" contract). The
    name "standard" short-circuited to the empty built-in before the user table
    was ever consulted, so its overrides were silently dropped -- while
    `agent6 config presets` reported it selected and applied."""
    _write_repo_config(
        repo,
        'preset = "standard"\n\n[presets.standard]\nreview = { trigger = "before_finish",'
        " concurrency = 4 }\n",
    )
    cfg = load_effective(repo).config
    assert cfg.review.trigger == "before_finish"
    assert cfg.review.concurrency == 4


def test_unknown_preset_errors(repo: Path) -> None:
    _write_repo_config(repo, 'preset = "nope"\n')
    with pytest.raises(ConfigError, match="unknown preset"):
        load_effective(repo)


def test_preset_table_instead_of_string_is_clear_error(repo: Path) -> None:
    """A `[preset]` TABLE (e.g. from a typo'd `config set preset.porifle x`)
    must fail as "preset must be a string", not str()-coerce the dict into
    `unknown preset "{'porifle': 'ultra'}"`."""
    _write_repo_config(repo, '[preset]\nporifle = "ultra"\n')
    with pytest.raises(ConfigError, match="must be a preset name string"):
        load_effective(repo)


def test_no_preset_is_plain_defaults(repo: Path) -> None:
    _write_repo_config(repo, "[review]\n")
    cfg = load_effective(repo).config
    assert cfg.review.trigger == "off" and cfg.review.concurrency == 1


# ---------------------------------------------------------------------------
# Scope-nested precedence: a preset OVERRIDES config at its scope, but a
# more-specific config layer (or flag) overrides the preset; most-specific
# preset source wins, presets never stack.
#
# `review.concurrency` is the observable knob: default 1, the custom presets
# below set it to 5, and config layers set it to other distinct values.
# ---------------------------------------------------------------------------

# A custom preset [presets.t] that sets review.concurrency = 5 (distinct from
# both the default 1 and the config values used in each test).
_PROFILE_T = "[presets.t.review]\nconcurrency = 5\n"


@pytest.fixture
def global_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point the global config at an isolated dir and return its path."""
    gdir = tmp_path / "global"
    (gdir / "agent6").mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(gdir))
    return gdir / "agent6" / "config.toml"


def test_global_selected_preset_loses_to_repo_config(repo: Path, global_config: Path) -> None:
    # Preset selected by GLOBAL top-level `preset` sits between global and repo
    # config, so a conflicting value in REPO config (more specific) wins.
    global_config.write_text(f'preset = "t"\n\n{_PROFILE_T}', encoding="utf-8")
    _write_repo_config(repo, "[review]\nconcurrency = 3\n")
    cfg = load_effective(repo).config
    assert cfg.review.concurrency == 3  # repo config beats global-selected preset


def test_repo_selected_preset_beats_same_repo_config(repo: Path) -> None:
    # Preset selected by REPO top-level `preset` sits ABOVE the repo config, so
    # a conflicting value in the SAME repo config loses to the preset.
    _write_repo_config(repo, f'preset = "t"\n\n[review]\nconcurrency = 3\n\n{_PROFILE_T}')
    cfg = load_effective(repo).config
    assert cfg.review.concurrency == 5  # repo-selected preset wins


def test_flag_selected_preset_beats_config(repo: Path) -> None:
    # --preset FLAG injects the preset above all config, so it beats a
    # conflicting value in config.
    _write_repo_config(repo, f"[review]\nconcurrency = 3\n\n{_PROFILE_T}")
    cfg = load_effective(repo, preset="t").config
    assert cfg.review.concurrency == 5  # flag-selected preset wins


def test_flag_preset_loses_to_explicit_config_file(repo: Path, tmp_path: Path) -> None:
    # --preset FLAG + an explicit --config FILE setting the same field: the
    # --config FILE sits ABOVE the flag-selected preset, so the file wins.
    _write_repo_config(repo, _PROFILE_T)  # custom preset defined in repo config
    explicit = tmp_path / "explicit.toml"
    explicit.write_text("[review]\nconcurrency = 7\n", encoding="utf-8")
    cfg = load_effective(repo, explicit, preset="t").config
    assert cfg.review.concurrency == 7  # explicit --config FILE beats the preset


def test_no_stacking_only_most_specific_preset_applies(repo: Path, global_config: Path) -> None:
    # Different presets at global (sets field X) and repo (sets field Y): only
    # the REPO preset applies; X falls back to its DEFAULT (no stacking).
    global_config.write_text(
        'preset = "g"\n\n[presets.g.review]\ntrigger = "before_finish"\n',
        encoding="utf-8",
    )
    _write_repo_config(
        repo,
        'preset = "r"\n\n[presets.r.review]\nconcurrency = 5\n',
    )
    cfg = load_effective(repo).config
    assert cfg.review.concurrency == 5  # the repo preset applies
    assert cfg.review.trigger == "off"  # the global preset does NOT stack (default)


def test_no_preset_anywhere_is_plain_config(repo: Path, global_config: Path) -> None:
    # Regression: with no preset selected anywhere, the result is identical to
    # plain config (the global/repo layers merge normally, preset is a no-op).
    global_config.write_text("[review]\nconcurrency = 4\n", encoding="utf-8")
    _write_repo_config(repo, '[review]\ntrigger = "before_finish"\n')
    cfg = load_effective(repo).config
    assert cfg.review.concurrency == 4  # from global config
    assert cfg.review.trigger == "before_finish"  # from repo config
