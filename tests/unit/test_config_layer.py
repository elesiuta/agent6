# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Tests for agent6.config.layer (layering, source map, show, fill)."""

from __future__ import annotations

from pathlib import Path

import pytest

from agent6 import paths as paths_mod
from agent6.config import (
    AnthropicProviderEntry,
    ConfigError,
    OpenAIProviderEntry,
    load_config,
)
from agent6.config.layer import (
    load_effective,
    materialize,
)
from agent6.config.write import set_config_value, unset_config_value
from agent6.errors import OperatorError
from agent6.paths import repo_config_path
from agent6.viewmodel.config_view import (
    ConfigSetting,
    ConfigView,
    build_config_view,
    render_show,
)

_GLOBAL = """\
[providers.anthropic]
api_format = "anthropic"

[models.worker]
provider = "anthropic"
model = "claude-sonnet-4-5"

[sandbox]
run_commands = "ask"
"""

_REPO = """\
[workflow]
verify_command = ["pytest", "-q"]

[sandbox]
run_commands = "yes"
"""


@pytest.fixture
def repo(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    gdir = tmp_path / "g"
    (gdir / "agent6").mkdir(parents=True, exist_ok=True)
    (gdir / "agent6" / "config.toml").write_text(_GLOBAL, encoding="utf-8")
    monkeypatch.setenv("XDG_CONFIG_HOME", str(gdir))
    repo_root = tmp_path / "repo"
    repo_root.mkdir(parents=True)
    rcfg = repo_config_path(repo_root)  # out of the workspace, under the state base
    rcfg.parent.mkdir(parents=True, exist_ok=True)
    rcfg.write_text(_REPO, encoding="utf-8")
    return repo_root


def test_layering_merges_global_and_repo(repo: Path) -> None:
    eff = load_effective(repo)
    cfg = eff.config
    # From global:
    assert cfg.models.worker is not None
    assert cfg.models.worker.model == "claude-sonnet-4-5"
    # From repo:
    assert cfg.workflow.verify_command == ("pytest", "-q")
    # Repo overrides global on the same field:
    assert cfg.sandbox.run_commands == "yes"


def test_an_unknown_key_points_at_config_fix(repo: Path) -> None:
    """`agent6 config set` refuses a key `Config` has no field for, so pointing
    an extra_forbidden leaf at it sends the operator to a second error. The
    remedy that works is `agent6 config fix`, which drops the key. A bad VALUE
    on a real key still points at `config set`."""
    gcfg = Path(repo).parent / "g" / "agent6" / "config.toml"
    gcfg.write_text('[sandbox]\nnonexistent_key = 1\nisolation = "srtict"\n', encoding="utf-8")
    with pytest.raises(ConfigError) as exc:
        load_effective(repo)
    text = str(exc.value)
    assert "sandbox.nonexistent_key" in text and "fix: agent6 config fix" in text
    assert "fix: agent6 config set sandbox.isolation <value>" in text


def test_source_map_attribution(repo: Path) -> None:
    eff = load_effective(repo)
    assert eff.sources["models.worker.model"] == "global"
    assert eff.sources["workflow.verify_command"] == "repo"
    assert eff.sources["sandbox.run_commands"] == "repo"  # repo wins
    # Untouched secure default:
    assert eff.sources["git.run_repo_hooks"] == "default"


def test_render_show_marks_overrides(repo: Path) -> None:
    eff = load_effective(repo)
    text = render_show(eff)
    assert "global" in text and "repo" in text
    assert "* = set by a config layer (see the source column)" in text
    # A defaulted field is unmarked; an overridden one is marked.
    assert "* models.worker.model" in text


def test_render_show_json(repo: Path) -> None:
    eff = load_effective(repo)
    import json

    data = json.loads(render_show(eff, as_json=True))
    assert data["workflow.verify_command"]["source"] == "repo"


# --- the UI-agnostic config view-model (shared by config show / TUI / web) ---


def _by_key(view: ConfigView) -> dict[str, ConfigSetting]:
    return {s.key: s for s in view.settings}


def test_build_config_view_provenance_type_choices(repo: Path) -> None:
    settings = _by_key(build_config_view(load_effective(repo)))
    rc = settings["sandbox.run_commands"]
    assert rc.source == "repo" and rc.modified is True
    # enum field -> a dropdown's worth of choices, typed "choice"
    assert rc.py_type == "choice" and rc.choices is not None and "yes" in rc.choices
    ap = settings["git.run_repo_hooks"]
    assert ap.source == "default" and ap.modified is False
    assert ap.py_type == "bool" and ap.default is False


def test_build_config_view_adaptive_resolution(repo: Path) -> None:
    view = build_config_view(load_effective(repo), resolved={"context.drop_at_chars": 999_999})
    s = _by_key(view)["context.drop_at_chars"]
    assert s.value is None  # raw: unset -> adaptive
    assert s.effective_value == 999_999
    assert s.is_adaptive is True
    assert s.modified is False  # an adaptive default is not a user modification


def test_render_show_json_is_full_view(repo: Path) -> None:
    import json

    data = json.loads(render_show(load_effective(repo), as_json=True))
    entry = data["sandbox.run_commands"]
    assert set(entry) >= {
        "value",
        "effective",
        "default",
        "source",
        "modified",
        "adaptive",
        "type",
        "choices",
    }
    assert entry["type"] == "choice" and "yes" in entry["choices"]


def test_render_show_text_marks_adaptive(repo: Path) -> None:
    text = render_show(load_effective(repo), resolved={"context.drop_at_chars": 471859})
    assert "(adaptive)" in text and "471859" in text


# --- shared edit path (the CLI + TUI/web editors write through this) ---


def test_config_write_keeps_the_edit_when_another_layer_was_already_invalid(
    repo: Path, tmp_path: Path
) -> None:
    """An edit is rolled back only when IT broke a valid config. Rolling back on
    any error meant a stale value in an unedited layer refused every write --
    and `agent6 connect` saves the API key before writing the provider block, so
    it exited having stored a key with no provider stanza to use it, and nothing
    said `agent6 config fix`."""
    # A pre-existing, unrelated error in the GLOBAL layer.
    (tmp_path / "g" / "agent6" / "config.toml").write_text('[cli]\ninput = "x"\n', encoding="utf-8")

    err = set_config_value(repo, "sandbox.run_commands", "no", to_repo=True)

    assert err is None, "a pre-existing error elsewhere must not refuse this edit"
    assert "run_commands" in repo_config_path(repo).read_text(encoding="utf-8")


def test_set_then_unset_config_value(repo: Path) -> None:
    # repo config starts with run_commands="yes"; global has "ask".
    err = set_config_value(repo, "sandbox.run_commands", "no", to_repo=True)
    assert err is None
    eff = load_effective(repo)
    assert eff.config.sandbox.run_commands == "no"
    assert eff.sources["sandbox.run_commands"] == "repo"
    # unset removes the repo override -> falls through to the global "ask".
    res = unset_config_value(repo, "sandbox.run_commands", to_repo=True)
    assert res.removed and res.error is None
    assert load_effective(repo).config.sandbox.run_commands == "ask"


def test_unset_reports_whether_anything_was_removed(repo: Path) -> None:
    """`config unset` says "nothing to unset" only when nothing was removed; a
    bare None return conflated that with a successful removal."""
    res = unset_config_value(repo, "sandbox.run_commands", to_repo=True)
    assert res.removed and res.error is None
    again = unset_config_value(repo, "sandbox.run_commands", to_repo=True)
    assert not again.removed and again.error is None


def test_unset_refuses_a_shape_the_surgery_cannot_carve(repo: Path) -> None:
    """A dotted top-level key has no [table] header to match: the refusal is an
    OperatorError for the one boundary, never a returned string a caller would
    print as a revalidation failure."""
    rcfg = repo_config_path(repo)
    before = 'sandbox.run_commands = "yes"\n'
    rcfg.write_text(before, encoding="utf-8")
    with pytest.raises(OperatorError):
        unset_config_value(repo, "sandbox.run_commands", to_repo=True)
    assert rcfg.read_text(encoding="utf-8") == before


def test_set_config_value_invalid_rolls_back(repo: Path) -> None:
    err = set_config_value(repo, "sandbox.run_commands", "bogus_value", to_repo=True)
    assert err is not None  # invalid enum -> rejected
    # the repo file was rolled back to its prior contents (run_commands="yes").
    assert load_effective(repo).config.sandbox.run_commands == "yes"


def test_set_config_value_rejects_a_value_masked_by_a_higher_layer(repo: Path) -> None:
    """An engine writer (set_config_*) must reject a value that is invalid on its
    own even when a HIGHER layer masks it in the merge -- else it lands the bad
    value and the config explodes once the mask is gone. The repo layer sets
    sandbox.run_commands="yes", so a GLOBAL write of a bad enum merges valid; only
    the standalone written-value check catches it. Shares the CLI's guard now, so
    the TUI/web/init/connect writers validate identically (the promised contract)."""
    gpath = repo.parent / "g" / "agent6" / "config.toml"
    before = gpath.read_text(encoding="utf-8")

    err = set_config_value(repo, "sandbox.run_commands", "garbage_not_an_enum", to_repo=False)

    assert err is not None and "sandbox.run_commands" in err
    assert gpath.read_text(encoding="utf-8") == before  # the masked bad value rolled back
    assert load_effective(repo).config.sandbox.run_commands == "yes"  # repo layer intact


def test_set_config_value_rejects_a_masked_invalid_provider_base_url(repo: Path) -> None:
    """A provider leaf rejected by a @field_validator (base_url's http(s) check),
    not a Field constraint, must still be caught on a masked write. The check
    validates the leaf against the provider MODEL, which runs the validator; a
    bare TypeAdapter of the annotation dropped it and let the bad value land."""
    repo_config_path(repo).write_text(
        '[providers.x]\napi_format = "openai"\nbase_url = "https://good.example/v1"\n',
        encoding="utf-8",
    )
    gpath = repo.parent / "g" / "agent6" / "config.toml"
    before = gpath.read_text(encoding="utf-8")

    err = set_config_value(repo, "providers.x.base_url", "not a url", to_repo=False)

    assert err is not None and "base_url" in err
    assert gpath.read_text(encoding="utf-8") == before  # the masked bad base_url rolled back


def test_written_value_error_catches_an_invalid_container_element() -> None:
    """A container's per-element error sits UNDER the key
    (`sandbox.fetch_hosts.0`), not at it: an error anywhere inside the written
    value is the written value's own, else a masked bad list lands and explodes
    only once the mask is gone."""
    from agent6.config.write import written_value_error

    assert written_value_error("sandbox.fetch_hosts", [5]) is not None
    assert written_value_error("providers.x.token_command", [1]) is not None
    assert written_value_error("sandbox.fetch_hosts", ["ok.example"]) is None
    assert written_value_error("providers.x.token_command", ["gcloud"]) is None


def test_a_scalar_written_to_a_list_leaf_names_both_ways_to_write_one() -> None:
    """`config set workflow.verify_command "python -m pytest"` answered with
    pydantic's "Input should be a valid tuple", which names neither the array
    form nor `config add`."""
    from agent6.config.write import written_value_error

    err = written_value_error("workflow.verify_command", "python -m pytest")
    assert err is not None
    assert "expected a list" in err
    assert "config add workflow.verify_command" in err


def test_setting_a_section_keeps_its_other_leaves_and_comments(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """`config set context '{ ... }'` is the form a sibling-rule refusal
    recommends. Written as one key it replaced the whole `[context]` table,
    silently taking every other leaf and comment with it."""
    from agent6.config.write import set_config_value

    gdir = tmp_path / "g"
    (gdir / "agent6").mkdir(parents=True, exist_ok=True)
    (gdir / "agent6" / "config.toml").write_text(
        "[context]\n# my tuning\nkeep_recent_chars = 50000\nsummary_max_tokens = 4096\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("XDG_CONFIG_HOME", str(gdir))
    repo_root = tmp_path / "repo"
    repo_root.mkdir()

    err = set_config_value(
        repo_root, "context", "{ drop_at_chars = 200000, summarise_at_chars = 400000 }"
    )

    assert err is None, err
    text = (gdir / "agent6" / "config.toml").read_text(encoding="utf-8")
    assert "# my tuning" in text
    assert "keep_recent_chars = 50000" in text
    assert "summary_max_tokens = 4096" in text
    # Both halves of the pair land under one revalidation, so the rule spanning
    # them sees its sibling instead of refusing each leaf on its own.
    assert "drop_at_chars = 200000" in text and "summarise_at_chars = 400000" in text


def test_a_dict_typed_leaf_is_replaced_whole(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """`providers.<name>.extra_body` is one VALUE, not a table: writing it leaf
    by leaf merged into the old value where it landed at all, and elsewhere
    refused with "set it as a whole" -- the command the operator had run."""
    from agent6.config.write import set_config_value

    gdir = tmp_path / "g"
    (gdir / "agent6").mkdir(parents=True, exist_ok=True)
    (gdir / "agent6" / "config.toml").write_text(
        "[providers.openrouter]\n"
        'api_format = "openai"\n'
        'base_url = "https://openrouter.ai/api/v1"\n'
        'extra_body = { provider = { sort = "throughput" } }  # prefer fast backends\n',
        encoding="utf-8",
    )
    monkeypatch.setenv("XDG_CONFIG_HOME", str(gdir))
    repo_root = tmp_path / "repo"
    repo_root.mkdir()

    err = set_config_value(repo_root, "providers.openrouter.extra_body", '{ order = ["x"] }')

    assert err is None, err
    text = (gdir / "agent6" / "config.toml").read_text(encoding="utf-8")
    assert 'extra_body = { order = ["x"] }' in text
    assert "sort" not in text, "the old value was merged into, not replaced"
    assert "# prefer fast backends" in text


def test_a_write_that_breaks_the_toml_is_rolled_back(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The revalidation reads the file it just wrote; when that read raises,
    the rollback it exists for must still run, or the operator is left with a
    config no command can read."""
    from agent6.config.write import set_config_value

    gdir = tmp_path / "g"
    (gdir / "agent6").mkdir(parents=True, exist_ok=True)
    before = '[sandbox]\nnetwork = "none"\n'
    (gdir / "agent6" / "config.toml").write_text(before, encoding="utf-8")
    monkeypatch.setenv("XDG_CONFIG_HOME", str(gdir))
    repo_root = tmp_path / "repo"
    repo_root.mkdir()

    err = set_config_value(repo_root, "providers.openai.extra_headers.X Title", "b")

    assert err is not None and "invalid TOML" in err
    assert (gdir / "agent6" / "config.toml").read_text(encoding="utf-8") == before


def test_written_value_error_catches_a_section_wide_rule() -> None:
    """A rule spanning two keys is a model_validator, and pydantic reports it at
    the SECTION -- a PARENT of the written key. Accepting a parent loc only for
    extra_forbidden let every such rule through: `config set
    context.drop_at_chars` from a repo whose layer set both halves wrote a
    half-set [context] to the GLOBAL file, exit 0, and every other repo on the
    machine then failed to load any config at all.

    The standalone dict holds only the written key, so a complaint about the
    section it sits in can only be about this write.
    """
    from agent6.config.write import written_value_error

    for key, value in (
        ("context.drop_at_chars", 200_000),  # pair: both or neither
        ("git.auto_stash_pop", True),  # needs auto_stash
        ("web.host", "0.0.0.0"),  # non-loopback needs the opt-in
    ):
        assert written_value_error(key, value) is not None, f"{key} slipped through"
    # A provider filled in over several sets still validates field by field.
    assert written_value_error("providers.x.base_url", "https://api.example") is None


def test_set_config_table_rejects_a_masked_invalid_leaf(repo: Path) -> None:
    """set_config_table writes a whole [table]; it must validate each LEAF, not the
    table dict as one. written_value_error only flags an error at loc == key, so a
    whole (key, dict) dropped every LEAF-level error and a masked-invalid leaf
    still landed (the TUI provider editor and `agent6 model` write through this)."""
    from agent6.config.write import set_config_table

    # The repo layer masks models.worker.effort with a valid value, so only the
    # standalone per-leaf check catches a bad `thinking` written to global.
    repo_config_path(repo).write_text(
        '[models.worker]\nprovider = "anthropic"\nmodel = "claude"\nthinking = "off"\n',
        encoding="utf-8",
    )
    gpath = repo.parent / "g" / "agent6" / "config.toml"
    before = gpath.read_text(encoding="utf-8")

    err = set_config_table(
        repo,
        "models.worker",
        {"provider": "anthropic", "model": "claude", "effort": "garbage_level"},
        to_repo=False,
    )

    assert err is not None and "effort" in err
    assert gpath.read_text(encoding="utf-8") == before  # the masked bad leaf rolled back


def test_flag_layer_wins(repo: Path, tmp_path: Path) -> None:
    flag = tmp_path / "flag.toml"
    flag.write_text('[sandbox]\nrun_commands = "no"\n', encoding="utf-8")
    eff = load_effective(repo, flag)
    assert eff.config.sandbox.run_commands == "no"
    assert eff.sources["sandbox.run_commands"] == "flag"


def test_overlay_is_highest_layer(repo: Path) -> None:
    from agent6.config.layer import load_effective_with_overlay

    overlay = {"sandbox": {"run_commands": "no"}, "review": {"trigger": "periodic"}}
    eff = load_effective_with_overlay(repo, overlay)
    # Overlay beats the repo value.
    assert eff.config.sandbox.run_commands == "no"
    assert eff.sources["sandbox.run_commands"] == "machine"
    # Overlay sets a brand-new value.
    assert eff.config.review.trigger == "periodic"
    assert eff.sources["review.trigger"] == "machine"
    # Lower layers still read through where the overlay is silent.
    assert eff.config.workflow.verify_command == ("pytest", "-q")


def test_empty_overlay_matches_load_effective(repo: Path) -> None:
    from agent6.config.layer import load_effective_with_overlay

    eff = load_effective_with_overlay(repo, {})
    assert eff.config.sandbox.run_commands == "yes"


def test_deep_merge_replaces_provider_when_kind_changes() -> None:
    # A lower layer's kind-specific keys must not survive a kind change, or they
    # surface as a confusing extra_forbidden error under the new kind.
    from agent6.config.layer import _deep_merge  # pyright: ignore[reportPrivateUsage]

    base = {"providers": {"p": {"api_format": "anthropic", "api_key_env": "X"}}}
    override = {"providers": {"p": {"api_format": "openai", "base_url": "Y"}}}
    merged = _deep_merge(base, override)
    assert merged["providers"]["p"] == {"api_format": "openai", "base_url": "Y"}


def test_deep_merge_still_merges_when_kind_unchanged() -> None:
    from agent6.config.layer import _deep_merge  # pyright: ignore[reportPrivateUsage]

    base = {"providers": {"p": {"api_format": "openai", "base_url": "Y", "api_key_env": "X"}}}
    override = {"providers": {"p": {"base_url": "Z"}}}
    merged = _deep_merge(base, override)
    assert merged["providers"]["p"] == {"api_format": "openai", "base_url": "Z", "api_key_env": "X"}


def test_materialize_roundtrips(repo: Path, tmp_path: Path) -> None:
    eff = load_effective(repo)
    text = materialize(eff.config)
    out = tmp_path / "full.toml"
    out.write_text(text, encoding="utf-8")
    # The materialized file must be a complete, valid config on its own.
    reloaded = load_config(out)
    assert reloaded.workflow.verify_command == ("pytest", "-q")
    assert reloaded.sandbox.run_commands == "yes"
    assert reloaded.providers["anthropic"].api_format == "anthropic"


def test_materialize_roundtrips_nested_objects_in_arrays(repo: Path, tmp_path: Path) -> None:
    """Dict-valued fields inside array items were dropped by the emitters, and
    a dict inside a plain list printed as Python repr -- so `config fill` (and
    a --parallel lane's snapshot) silently changed a valid provider request.
    Every JSON-shaped extra_body value must survive materialize -> parse."""
    gpath = repo.parent / "g" / "agent6" / "config.toml"
    gpath.write_text(
        gpath.read_text(encoding="utf-8")
        + "\n[providers.gw]\n"
        + 'api_format = "openai"\n'
        + 'base_url = "https://gw.example.com/v1"\n'
        + "[providers.gw.extra_body]\n"
        + 'models = [{name = "a", options = {weight = 2, tags = ["x"]}}, {name = "b"}]\n'
        + "mixed = [1, {flag = true}]\n",
        encoding="utf-8",
    )
    eff = load_effective(repo)
    out = tmp_path / "full.toml"
    out.write_text(materialize(eff.config), encoding="utf-8")
    reloaded = load_config(out)
    gw = reloaded.providers["gw"]
    assert isinstance(gw, OpenAIProviderEntry)
    body = gw.extra_body
    assert body["models"] == [
        {"name": "a", "options": {"weight": 2, "tags": ["x"]}},
        {"name": "b"},
    ]
    assert body["mixed"] == [1, {"flag": True}]


def test_missing_flag_file_errors(repo: Path, tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="not found"):
        load_effective(repo, tmp_path / "does-not-exist.toml")


def test_provenance_survives_a_format_changing_provider_replace(repo: Path) -> None:
    """_deep_merge wholesale-REPLACES a provider entry when api_format flips
    between layers; the old separate provenance pass kept the discarded lower
    layer's stale source entries, so `config show` attributed refilled model
    DEFAULTS (base_url, timeouts) to a file holding different values -- with
    the operator-set marker. Provenance is now stamped in the same walk as
    the merge."""
    gpath = repo.parent / "g" / "agent6" / "config.toml"
    gpath.write_text(
        gpath.read_text(encoding="utf-8")
        + "\n[providers.foo]\n"
        + 'api_format = "openai"\n'
        + 'base_url = "https://x.example/v1"\n'
        + "http_timeout_s = 30.0\n",
        encoding="utf-8",
    )
    rpath = repo_config_path(repo)
    rpath.write_text(
        rpath.read_text(encoding="utf-8") + '\n[providers.foo]\napi_format = "anthropic"\n',
        encoding="utf-8",
    )
    eff = load_effective(repo)
    foo = eff.config.providers["foo"]
    assert foo.api_format == "anthropic"
    assert foo.base_url == "https://api.anthropic.com/v1"  # refilled default
    assert eff.sources["providers.foo.api_format"] == "repo"
    # The refilled defaults are DEFAULTS, not phantom global values.
    assert eff.sources["providers.foo.base_url"] == "default"
    assert eff.sources["providers.foo.http_timeout_s"] == "default"


def test_profile_key_is_rejected_in_flag_and_machine_layers(repo: Path, tmp_path: Path) -> None:
    """Only global/repo config (and --preset) can SELECT a preset; the key
    still merged from a --config FILE or machine overlay, so config show
    displayed preset=<name> as effective while the preset silently never
    applied -- and resume then replayed the stamped name as a real selection,
    making the resumed run behave differently from the original. Reject the
    key loudly in the layers that cannot select it."""
    from agent6.config.layer import load_effective_with_overlay

    explicit = tmp_path / "ci.toml"
    explicit.write_text('preset = "ultra"\n', encoding="utf-8")
    with pytest.raises(ConfigError, match="preset"):
        load_effective(repo, explicit)
    with pytest.raises(ConfigError, match="preset"):
        load_effective_with_overlay(repo, {"preset": "ultra"})


def test_materialize_quotes_non_bare_keys(repo: Path, tmp_path: Path) -> None:
    """A provider hand-named with a space or dot is valid input, so the
    serializer must quote it: raw interpolation emitted an unparseable
    `[providers.my provider]` header (or a silently re-nested dotted one),
    and `config fill --force` then replaced the operator's working config
    with the broken output."""
    from agent6.config import Config, load_config
    from agent6.config.layer import materialize

    cfg = Config.model_validate(
        {
            "providers": {
                "my provider": {"api_format": "openai", "base_url": "https://x.example/v1"},
                "openrouter.free": {"api_format": "openai", "base_url": "https://o.example/v1"},
            },
            "skills": {"state": {"org.some.skill": "enabled"}},
        }
    )
    out = tmp_path / "materialized.toml"
    out.write_text(materialize(cfg), encoding="utf-8")
    reloaded = load_config(out)
    assert "my provider" in reloaded.providers
    assert "openrouter.free" in reloaded.providers  # not silently re-nested
    assert reloaded.skills.state == {"org.some.skill": "enabled"}


def test_materialize_escapes_control_chars_in_values(repo: Path, tmp_path: Path) -> None:
    """A control char in a config string value must serialize to valid TOML.
    The old escape (backslash + quote only) emitted the raw char, so the file
    failed to parse on the next read while `config fill` reported success."""
    from agent6.config import Config, load_config
    from agent6.config.layer import materialize

    cfg = Config.model_validate({"workflow": {"verify_command": ["echo", "a\x01b\nc"]}})
    out = tmp_path / "materialized.toml"
    out.write_text(materialize(cfg), encoding="utf-8")
    reloaded = load_config(out)
    assert list(reloaded.workflow.verify_command) == ["echo", "a\x01b\nc"]


def test_concurrent_rollback_does_not_erase_a_valid_write(
    repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Writer A publishes an invalid value and rolls back after revalidation;
    writer B lands a valid value in between. A's rollback republished its
    pre-B snapshot: B's update was silently erased (and B's own revalidate,
    seeing A's junk still in the file, spuriously rejected B's write). The
    whole write+revalidate+rollback cycle now holds locked_file, so B queues
    until A has rolled back and then lands cleanly on the restored base."""
    import threading
    import time

    from agent6.config import write as write_mod

    a_in_revalidate = threading.Event()
    b_attempted = threading.Event()
    real_load = write_mod.load_effective
    calls = {"n": 0}

    def gated_load(root: Path, flag: Path | None) -> object:
        calls["n"] += 1
        if calls["n"] == 1:  # A's revalidate: hold the transaction open
            a_in_revalidate.set()
            b_attempted.wait(timeout=5)
            time.sleep(0.4)  # window for B to (old code) land / (fixed) queue
        return real_load(root, flag)

    monkeypatch.setattr(write_mod, "load_effective", gated_load)
    results: dict[str, str | None] = {}

    def writer_a() -> None:
        results["a"] = set_config_value(repo, "sandbox.run_commands", "bogus", to_repo=True)

    def writer_b() -> None:
        a_in_revalidate.wait(timeout=5)
        b_attempted.set()
        results["b"] = set_config_value(repo, "git.dirty_tree", "stash", to_repo=True)

    ta = threading.Thread(target=writer_a, daemon=True)
    tb = threading.Thread(target=writer_b, daemon=True)
    ta.start()
    tb.start()
    ta.join(timeout=10)
    tb.join(timeout=10)
    assert results["a"] is not None  # the invalid write was rejected
    assert results["b"] is None  # ...without taking B's valid write down with it
    eff = load_effective(repo)
    assert eff.config.git.dirty_tree == "stash"  # B's update survived A's rollback
    assert eff.config.sandbox.run_commands == "yes"  # A rolled back to the prior value


def test_prepare_write_target_hands_back_the_created_state_base(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A sudo config write on a fresh machine creates the whole state base;
    chowning only the deepest dir left `<base>` root-owned, and the next
    repo's non-root write then died creating its sibling dir there."""
    import os

    from agent6.config import write as write_mod

    # Through sudo the XDG vars are root's and ignored: the base is the real
    # user's `~/.local/state/agent6`, two levels of which do not exist yet.
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setattr(
        paths_mod,
        "effective_user",
        lambda: paths_mod.RealUser(uid=1234, gid=1234, name="op", home=home, via_sudo=True),
    )
    base = home / ".local" / "state" / "agent6"
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    monkeypatch.setattr(os, "geteuid", lambda: 0)
    monkeypatch.setenv("SUDO_UID", "1234")
    monkeypatch.setenv("SUDO_GID", "1234")
    chowned: list[Path] = []

    def _record(*a: object) -> None:
        chowned.append(Path(str(a[0])))

    def _record_at(target: object, _uid: int, _gid: int, **kw: object) -> None:
        chowned.append(Path(f"/proc/self/fd/{kw['dir_fd']}").readlink() / str(target))

    monkeypatch.setattr(os, "lchown", _record)
    monkeypatch.setattr(os, "chown", _record_at)
    target = write_mod._prepare_write_target(repo_root, to_repo=True)  # pyright: ignore[reportPrivateUsage]
    assert target.parent.is_dir()
    assert base in chowned  # the created base is handed back...
    assert home / ".local" in chowned  # ...and every created level above it


def test_config_write_hands_the_dir_over_before_writing(
    repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Under `sudo` the config dir is created as root. Handing it back only
    after a SUCCESSFUL write stranded it root-owned whenever the write failed
    or the writer was killed inside the lock, and every later non-root write
    then died PermissionError creating its atomic-write temp file there."""
    from agent6.config import write as write_mod

    handed: list[Path] = []
    monkeypatch.setattr(write_mod, "mkdir_for_real_user", handed.append)

    def killed(*_args: object, **_kwargs: object) -> None:
        raise KeyboardInterrupt  # stands in for the operator killing the writer

    monkeypatch.setattr(write_mod, "upsert_toml_leaf", killed)
    with pytest.raises(KeyboardInterrupt):
        set_config_value(repo, "git.dirty_tree", "stash", to_repo=True)
    assert handed[0] == repo_config_path(repo).parent  # before the write, not after it


def test_config_write_hands_the_file_over_after_a_rejected_edit(
    repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A rejected edit rolls back through atomic_write, i.e. republishes the
    file as a NEW inode owned by root under `sudo`, so the handover cannot be
    conditional on the edit being valid."""
    from agent6.config import write as write_mod

    handed: list[Path] = []
    monkeypatch.setattr(write_mod, "chown_to_real_user", handed.append)
    assert set_config_value(repo, "sandbox.run_commands", "bogus_value", to_repo=True) is not None
    assert repo_config_path(repo) in handed


def test_engine_writers_refuse_a_write_into_an_unparseable_target(
    repo: Path, tmp_path: Path
) -> None:
    """Line surgery on a file that does not parse only appends to the damage
    (a malformed header is invisible to the lookups, so the write lands as a
    duplicate table): every writer refuses up front with the parse error, the
    same refusal the CLI always gave."""
    gcfg = tmp_path / "g" / "agent6" / "config.toml"
    gcfg.write_text("[sandbox\nprotect_git = true\n", encoding="utf-8")  # missing ]
    before = gcfg.read_text(encoding="utf-8")

    with pytest.raises(ConfigError):
        set_config_value(repo, "sandbox.run_commands", "no")

    assert gcfg.read_text(encoding="utf-8") == before


def test_no_lock_rollback_keeps_the_write_and_says_so(
    repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When the config lock fails open (a stale root-owned .lock a killed sudo
    writer left), _revalidate's whole-file restore could erase a concurrent
    writer's just-validated update -- the snapshot predates it. Without the
    lock the write is KEPT and the error says so, narrowing the exposure back
    to the unlocked RMW the fail-open always tolerated."""
    import agent6.portable as portable_mod

    def _no_lock(_p: Path) -> int | None:
        return None

    monkeypatch.setattr(portable_mod, "_acquire_lock", _no_lock)
    err = set_config_value(repo, "sandbox.run_commands", "bogus_value", to_repo=True)
    assert err is not None
    assert "kept as written" in err and "lock" in err
    # NOT restored: the invalid value is still in the file for the operator.
    text = repo_config_path(repo).read_text(encoding="utf-8")
    assert 'run_commands = "bogus_value"' in text


def test_an_optional_section_is_written_leaf_by_leaf(repo: Path) -> None:
    """`models.worker` and `workflow.metric` are `[table]`s whose type is
    optional; read as leaves they were written inline under a `[models]` header
    of their own, which declares the same key the existing `[models.worker]`
    block does -- refused as "invalid TOML", blaming a file that parses."""
    rcfg = repo_config_path(repo)
    rcfg.write_text(
        '[models.worker]\nprovider = "anthropic"\nmodel = "claude-sonnet-4-5"\n', encoding="utf-8"
    )

    assert set_config_value(repo, "models.worker", '{ model = "gpt-y" }', to_repo=True) is None

    text = rcfg.read_text(encoding="utf-8")
    assert "[models.worker]" in text
    assert 'model = "gpt-y"' in text
    assert 'provider = "anthropic"' in text, "a section's other leaves survive"


def test_a_name_keyed_table_is_written_entry_by_entry(repo: Path) -> None:
    """`providers` and `mcp.servers` are tables of entries, not one value:
    written whole, a `config set providers '{...}'` replaced every provider the
    operator had, with their keys and their comments, at exit 0."""
    rcfg = repo_config_path(repo)
    rcfg.write_text(
        '[providers.anthropic]\napi_format = "anthropic"\napi_key_env = "A"\n', encoding="utf-8"
    )

    err = set_config_value(
        repo,
        "providers",
        '{ ollama = { api_format = "openai", base_url = "http://localhost:11434/v1" } }',
        to_repo=True,
    )

    assert err is None
    providers = load_effective(repo).config.providers
    assert set(providers) >= {"anthropic", "ollama"}
    kept = providers["anthropic"]
    assert isinstance(kept, AnthropicProviderEntry)
    assert kept.api_key_env == "A"


def test_a_table_valued_leaf_replaces_the_block_it_already_has(repo: Path) -> None:
    """A dict-typed leaf is one value, written whole -- and the other shape it
    can already have on disk is its own `[table.leaf]` block, which the inline
    write must replace rather than declare twice."""
    rcfg = repo_config_path(repo)
    rcfg.write_text('[skills.state]\nalpha = "enabled"\n', encoding="utf-8")

    assert set_config_value(repo, "skills.state", '{ gamma = "always" }', to_repo=True) is None

    text = rcfg.read_text(encoding="utf-8")
    assert "gamma" in text and "alpha" not in text, text
    assert load_effective(repo).config.skills.state == {"gamma": "always"}


def test_an_invalid_value_is_refused_even_where_its_section_was_broken(repo: Path) -> None:
    """A section rule that a SIBLING breaks is not this edit's fault, and the
    write stands. This edit's own value being invalid is, whatever else in the
    section was already wrong -- it landed with a warning that blamed a value
    "in another layer" and exit 0."""
    rcfg = repo_config_path(repo)
    before = '[web]\nhost = "0.0.0.0"\n'  # already invalid: non-loopback, not opted in
    rcfg.write_text(before, encoding="utf-8")

    err = set_config_value(repo, "web.port", "abc", to_repo=True)

    assert err is not None and "valid integer" in err
    assert rcfg.read_text(encoding="utf-8") == before


def test_set_config_leaves_refuses_a_headerless_ancestor(
    repo: Path,
) -> None:
    """`agent6 connect` / init / the TUI write providers through set_config_leaves.
    A leaf whose ancestor is a header-less (inline) table cannot be set on its own;
    the surgery's refusal is an OperatorError -- a printable message at the one
    boundary, never a traceback -- and the file is untouched."""
    from agent6.config.write import set_config_leaves

    rcfg = repo_config_path(repo)
    before = '[providers]\nanthropic = { api_format = "anthropic" }\n'
    rcfg.write_text(before, encoding="utf-8")

    with pytest.raises(OperatorError, match="not a plain"):
        set_config_leaves(repo, "providers.anthropic", {"base_url": "https://x/v1"}, to_repo=True)

    assert rcfg.read_text(encoding="utf-8") == before  # nothing partially written


def test_set_config_leaves_rolls_back_a_partial_multi_leaf_write(
    repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One revalidate+rollback wraps ALL the leaf writes: when a later leaf raises,
    the earlier leaves that already landed roll back to the prior file rather than
    leaving a half-applied provider block."""
    from agent6.config import write as write_mod
    from agent6.config.write import set_config_leaves

    rcfg = repo_config_path(repo)
    before = '[providers.anthropic]\napi_format = "anthropic"\n'
    rcfg.write_text(before, encoding="utf-8")

    real = write_mod.upsert_toml_leaf
    calls = {"n": 0}

    def _fail_second(path: Path, key: str, value: object) -> None:
        calls["n"] += 1
        if calls["n"] == 2:
            raise ConfigError("second leaf refused")
        real(path, key, value)

    monkeypatch.setattr(write_mod, "upsert_toml_leaf", _fail_second)

    with pytest.raises(OperatorError, match="second leaf refused"):
        set_config_leaves(
            repo,
            "providers.anthropic",
            {"base_url": "https://x/v1", "api_key_env": "KEY"},
            to_repo=True,
        )

    assert rcfg.read_text(encoding="utf-8") == before  # the first leaf's write rolled back


def test_leaves_partial_write_without_the_lock_is_kept_and_says_so(
    repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When the lock failed open, restoring the prior file could erase a
    concurrent writer's update, so a partial multi-leaf write is KEPT and the
    refusal says so -- the same keep-and-warn every writer applies."""
    import agent6.portable as portable_mod
    from agent6.config import write as write_mod
    from agent6.config.write import set_config_leaves

    def _no_lock(_p: Path) -> int | None:
        return None

    monkeypatch.setattr(portable_mod, "_acquire_lock", _no_lock)
    rcfg = repo_config_path(repo)
    before = '[providers.anthropic]\napi_format = "anthropic"\n'
    rcfg.write_text(before, encoding="utf-8")
    real = write_mod.upsert_toml_leaf
    calls = {"n": 0}

    def _fail_second(path: Path, key: str, value: object) -> None:
        calls["n"] += 1
        if calls["n"] == 2:
            raise ConfigError("second leaf refused")
        real(path, key, value)

    monkeypatch.setattr(write_mod, "upsert_toml_leaf", _fail_second)

    with pytest.raises(OperatorError, match="kept as written"):
        set_config_leaves(
            repo,
            "providers.anthropic",
            {"base_url": "https://x/v1", "api_key_env": "KEY"},
            to_repo=True,
        )

    assert "base_url" in rcfg.read_text(encoding="utf-8")  # the landed leaf was kept


def test_load_config_wraps_an_unreadable_file(tmp_path: Path) -> None:
    """The single-file loader caught the TOML parse error but not the OSError
    its layered sibling wraps: chmod-000 escaped as a raw PermissionError."""
    p = tmp_path / "c.toml"
    p.write_text("[review]\nperiod = 7\n", encoding="utf-8")
    p.chmod(0o000)
    try:
        with pytest.raises(ConfigError, match="cannot be read"):
            load_config(p)
    finally:
        p.chmod(0o600)


def test_provider_members_are_derived_from_the_union() -> None:
    """A hand-listed member tuple drifts silently: a new provider entry type
    would be validated by nothing, so a bad leaf on it would land."""
    from typing import get_args

    from agent6.config import ProviderEntry
    from agent6.config.write import PROVIDER_MEMBERS

    declared = get_args(get_args(ProviderEntry)[0])
    assert set(PROVIDER_MEMBERS) == set(declared)
    assert len(PROVIDER_MEMBERS) == len(declared) >= 2
