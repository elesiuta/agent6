# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""`agent6 config show` renders TOML an operator can copy back out."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent6.config import Config
from agent6.config.layer import EffectiveConfig, load_effective
from agent6.models.registry import resolved_adaptive_values
from agent6.viewmodel.config_view import render_key_detail, render_show


def test_a_top_level_scalar_is_not_dressed_as_a_table(tmp_path: Path) -> None:
    """`preset` is a bare top-level key, not a `[preset]` table.

    The renderer grouped every leaf by its first dotted segment, so a key with
    no dot became its own one-row "section" under a `[preset]` header. Copying
    that into a config file writes invalid TOML, and `config fill` -- the other
    half of the same feature -- already emits top-level scalars correctly.
    """
    out = render_show(load_effective(tmp_path, preset="quick"))

    assert "[preset]" not in out, "a scalar rendered as a TOML table header"
    assert "preset" in out, "the setting itself must still be shown"
    # It belongs above the tables, exactly where TOML requires it.
    assert out.index("preset") < out.index("["), "a top-level scalar must precede every section"


def test_config_presets_reads_the_explicit_config_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """`--config FILE` is a global flag every config subcommand honours -- except
    `presets`, which hardcoded None and silently listed only the built-ins.

    Silently: the file parsed, the preset was there, and the listing simply did
    not mention it.
    """
    from agent6.ui.cli import main

    cfg = tmp_path / "custom.toml"
    cfg.write_text('[presets.myfast.sandbox]\nrun_commands = "yes"\n', encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    assert main(["--config", str(cfg), "config", "presets"]) == 0
    assert "myfast" in capsys.readouterr().out, "presets ignored the explicit config file"


def test_a_filled_config_can_be_used_as_an_explicit_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`config fill` snapshots every effective value into one explicit file. That
    file has to be a config agent6 will actually load.

    It emitted the top-level `preset` selector, which the layer REFUSES from an
    explicit `--config` file -- so `agent6 config fill` produced a file that
    `agent6 --config <it>` rejected. `--parallel` was collateral: the
    orchestrator materializes each lane's config the same way, so every lane
    died before starting.

    A preset SELECTS other leaves; once they are materialized the selector is
    both redundant and, for a named preset, would apply twice.
    """
    from agent6.config.layer import materialize

    filled = tmp_path / "filled.toml"
    filled.write_text(materialize(load_effective(tmp_path).config), encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    # The point: this must not raise.
    reloaded = load_effective(tmp_path, filled).config
    assert reloaded.agent6.config_version == 1


def test_config_fill_keeps_the_presets_the_file_defines(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`config fill` rewrites the operator's own config file. A `[presets.*]`
    table in it is meta-config -- stripped before validation, so absent from the
    `Config` the snapshot is rendered from -- and the rewrite dropped it.

    Silently, and with `--force` there is no earlier copy: `config presets`
    listed `myfast` before the fill and only the built-ins after. The leaves the
    preset selected survive (they are materialized), the definition did not, so
    `--preset myfast` stopped resolving at all.
    """
    from agent6.ui.cli import main

    cfg_home = tmp_path / "cfg"
    cfg_home.mkdir()
    monkeypatch.setenv("AGENT6_CONFIG_HOME", str(cfg_home))
    (cfg_home / "config.toml").write_text(
        'preset = "myfast"\n\n[presets.myfast.sandbox]\nrun_commands = "yes"\n',
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)

    assert main(["config", "fill", "--force"]) == 0

    after = load_effective(tmp_path)
    assert after.config.sandbox.run_commands == "yes", "the preset stopped applying"
    text = (cfg_home / "config.toml").read_text(encoding="utf-8")
    assert "[presets.myfast" in text, f"config fill deleted the operator's preset:\n{text}"
    # The SELECTOR survives, and the preset's EFFECT is not baked: the filled
    # leaf is the default, with the preset still applying over it at runtime.
    # Baking it froze the old values while the selector -- what the operator
    # edits -- was dropped, so later preset edits did nothing.
    assert 'preset = "myfast"' in text
    assert 'run_commands = "ask"' in text, f"the preset's effect was baked in:\n{text}"


def test_descriptions_mode_prints_the_meaning_under_each_row() -> None:
    """`--descriptions` adds each leaf's meaning; the default stays values-only."""
    eff = EffectiveConfig(config=Config(), sources={}, layers=())
    assert "Cap on the metered spend" not in render_show(eff)
    assert "Cap on the metered spend" in render_show(eff, descriptions=True)


def test_key_detail_always_carries_the_meaning() -> None:
    """`config show <key>` is a deliberate ask about one key, so the meaning is
    part of the answer, no flag needed."""
    eff = EffectiveConfig(config=Config(), sources={}, layers=())
    detail = render_key_detail(eff, ["budget.max_usd"])
    assert "meaning: Cap on the metered spend" in detail


def test_key_detail_takes_several_keys_in_the_order_asked() -> None:
    """`config show a b` prints a's leaves then b's (a section prefix expands
    to its leaves, a leaf named twice prints once); a key matching nothing
    raises KeyError naming it, so the command can refuse by name."""
    eff = EffectiveConfig(config=Config(), sources={}, layers=())
    detail = render_key_detail(eff, ["sandbox.network", "budget", "sandbox.network"])
    heads = [line.strip() for line in detail.splitlines() if not line.startswith("    ")]
    assert heads[0] == "sandbox.network"
    assert "budget.max_usd" in heads
    assert heads.index("budget.max_usd") > 0
    assert heads.count("sandbox.network") == 1
    with pytest.raises(KeyError, match="nope"):
        render_key_detail(eff, ["budget.max_usd", "nope"])
    as_json = json.loads(render_key_detail(eff, ["sandbox.network"], as_json=True))
    assert list(as_json) == ["sandbox.network"]


def _effort_config(tmp_path: Path, body: str) -> EffectiveConfig:
    cfg = tmp_path / "config.toml"
    cfg.write_text(body, encoding="utf-8")
    return load_effective(tmp_path, cfg)


def test_an_unset_effort_shows_what_the_openai_wire_actually_sends(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`(unset)` claimed nothing was chosen while openai-compatible reasoning
    models were getting `low` on every call."""
    monkeypatch.delenv("AGENT6_REASONING_EFFORT", raising=False)
    eff = _effort_config(
        tmp_path,
        '[providers.openrouter]\napi_format = "openai"\n'
        'base_url = "https://openrouter.ai/api/v1"\n'
        '[models.worker]\nprovider = "openrouter"\nmodel = "moonshotai/kimi-k2.6"\n',
    )

    resolved = resolved_adaptive_values(eff.config)

    assert resolved["models.worker.effort"] == "low"
    assert "low" in render_show(eff, resolved=resolved)


def test_a_model_that_takes_no_reasoning_knob_keeps_the_unset_row(tmp_path: Path) -> None:
    """Only a resolution the wire really applies replaces `(unset)`."""
    eff = _effort_config(
        tmp_path,
        '[providers.openrouter]\napi_format = "openai"\n'
        'base_url = "https://openrouter.ai/api/v1"\n'
        '[models.worker]\nprovider = "openrouter"\nmodel = "qwen/qwen3-coder"\n',
    )

    assert "models.worker.effort" not in resolved_adaptive_values(eff.config)


def test_an_unset_anthropic_effort_resolves_to_off(tmp_path: Path) -> None:
    """Anthropic sends no thinking at all when the role leaves effort unset."""
    eff = _effort_config(
        tmp_path,
        '[providers.anthropic]\napi_format = "anthropic"\n'
        '[models.worker]\nprovider = "anthropic"\nmodel = "claude-opus-5"\n',
    )

    assert resolved_adaptive_values(eff.config)["models.worker.effort"] == "off"


def test_a_configured_effort_is_not_marked_resolved(tmp_path: Path) -> None:
    """The row shows the operator's own value, with its layer, not a default."""
    eff = _effort_config(
        tmp_path,
        '[providers.openrouter]\napi_format = "openai"\n'
        'base_url = "https://openrouter.ai/api/v1"\n'
        '[models.worker]\nprovider = "openrouter"\nmodel = "moonshotai/kimi-k2.6"\n'
        'effort = "high"\n',
    )

    assert "models.worker.effort" not in resolved_adaptive_values(eff.config)


def test_the_env_override_is_the_value_shown(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """AGENT6_REASONING_EFFORT sits below the config and above the built-in
    default; `config show` reads the same resolver the request does."""
    monkeypatch.setenv("AGENT6_REASONING_EFFORT", "medium")
    eff = _effort_config(
        tmp_path,
        '[providers.openrouter]\napi_format = "openai"\n'
        'base_url = "https://openrouter.ai/api/v1"\n'
        '[models.worker]\nprovider = "openrouter"\nmodel = "moonshotai/kimi-k2.6"\n',
    )

    assert resolved_adaptive_values(eff.config)["models.worker.effort"] == "medium"


def test_an_empty_string_default_renders_a_visible_token(tmp_path: Path) -> None:
    """`preset`, `git.commit.trailer`, `prompt.system_prompt_file` and
    `parallel.workdir` default to "" and rendered a blank cell, which reads as
    a rendering failure next to `(unset)`, `[]` and `{}`."""
    eff = _effort_config(tmp_path, "")
    rows = render_show(eff, resolved=resolved_adaptive_values(eff.config)).splitlines()
    preset = next(line for line in rows if line.split()[:1] == ["preset"])
    assert "(empty)" in preset, preset


def test_the_auto_sandbox_leaves_show_what_this_host_resolves_them_to(tmp_path: Path) -> None:
    """`sandbox.isolation = auto` and `sandbox.network = auto` resolved at one
    place, `agent6 check config`; the config views on every surface showed
    `auto` with no resolution, so a browser-only operator never learned what
    a run here would get."""
    from agent6.app.confine import resolved_config_values
    from agent6.viewmodel.config_view import build_config_view

    eff = _effort_config(tmp_path, "")
    view = build_config_view(eff, resolved=resolved_config_values(eff.config))
    rows = {s.key: s for s in view.settings}
    assert rows["sandbox.isolation"].is_adaptive
    assert rows["sandbox.isolation"].effective_value in ("strict", "hardened", "none")
    assert rows["sandbox.network"].is_adaptive
    assert "(adaptive)" in render_show(eff, resolved=resolved_config_values(eff.config))

    explicit = _effort_config(tmp_path, '[sandbox]\nisolation = "none"\n')
    view = build_config_view(explicit, resolved=resolved_config_values(explicit.config))
    assert not {s.key: s for s in view.settings}["sandbox.isolation"].is_adaptive
