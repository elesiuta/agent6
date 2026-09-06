# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Tests for `agent6 config get/set/unset/add/remove` + allow_urls egress wiring."""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

from agent6.config.layer import resolved_state_dir


@pytest.fixture
def iso(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Isolated global config home + cwd inside a fresh repo."""
    monkeypatch.setenv("AGENT6_CONFIG_HOME", str(tmp_path / "g"))
    monkeypatch.chdir(tmp_path)
    return tmp_path


def _run(args: list[str]) -> int:
    from agent6.ui.cli import main

    return main(args)


def _refuse(args: list[str]) -> int:
    """Run through the guarded entry point: operator errors present as
    `ERROR:` + exit 2 there, while `main` raises them."""
    from agent6.ui.cli import cli_main

    return cli_main(args)


def _global_toml(tmp_path: Path) -> dict[str, object]:
    return tomllib.loads((tmp_path / "g" / "config.toml").read_text(encoding="utf-8"))


# --- set / get / unset (scalars) -------------------------------------------


def test_set_bool_is_typed_not_string(iso: Path) -> None:
    assert _run(["config", "set", "sandbox.protect_git", "false"]) == 0
    sandbox = _global_toml(iso)["sandbox"]
    assert isinstance(sandbox, dict)
    assert sandbox["protect_git"] is False  # parsed as bool, not the string "false"


def test_get_default_source_for_unset(iso: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert _run(["config", "get", "sandbox.protect_git"]) == 0
    out = capsys.readouterr().out
    assert "sandbox.protect_git = true" in out
    assert "[default]" in out


def test_get_unknown_key_errors(iso: Path) -> None:
    assert _run(["config", "get", "sandbox.nope"]) == 2


def test_machine_get_on_malformed_toml_is_clean_error(
    iso: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # A malformed --machine-file must produce a clean ERROR (exit 2),
    # not an uncaught TOMLDecodeError traceback.
    bad = tmp_path / "broken.asm.toml"
    bad.write_text("this is = not valid [[[\n", encoding="utf-8")
    assert _refuse(["config", "get", "git.merge_strategy", "--machine-file", str(bad)]) == 2
    err = capsys.readouterr().err
    assert "invalid TOML" in err
    assert "report this" not in err


def test_unset_refuses_a_key_that_is_not_a_leaf(
    iso: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """`config unset nope.nope` refuses like get and set do; "nothing to unset"
    with exit 0 read as a no-op on a key that never existed."""
    assert _run(["config", "unset", "nope.nope"]) == 2
    assert "is not a config leaf" in capsys.readouterr().err


def test_unset_reverts_to_default(iso: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _run(["config", "set", "sandbox.protect_git", "false"])
    assert _run(["config", "unset", "sandbox.protect_git"]) == 0
    capsys.readouterr()
    _run(["config", "get", "sandbox.protect_git"])
    assert "[default]" in capsys.readouterr().out


def test_unset_last_leaf_drops_the_empty_table(iso: Path) -> None:
    # Unsetting a section's only key must not leave a dangling [sandbox]
    # header accreting in the file; a sibling key keeps the section.
    from agent6.paths import global_config_path

    _run(["config", "set", "git.auto_stash", "true"])
    _run(["config", "set", "sandbox.run_commands", "yes"])
    assert _run(["config", "unset", "sandbox.run_commands"]) == 0
    text = global_config_path().read_text(encoding="utf-8")
    assert "[sandbox]" not in text
    assert "[git]" in text  # untouched sibling section survives
    _run(["config", "set", "git.run_repo_hooks", "true"])
    assert _run(["config", "unset", "git.auto_stash"]) == 0
    text = global_config_path().read_text(encoding="utf-8")
    assert "[git]" in text  # still holds run_repo_hooks
    assert "run_repo_hooks" in text and "auto_stash" not in text


# --- top-level `preset` (the one section-less leaf) -------------------------


def test_set_top_level_profile_and_get(iso: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert _run(["config", "set", "preset", "ultra"]) == 0
    assert _global_toml(iso)["preset"] == "ultra"
    capsys.readouterr()
    assert _run(["config", "get", "preset"]) == 0
    out = capsys.readouterr().out
    assert "preset = ultra" in out
    assert "[global]" in out


def test_set_unknown_profile_name_reverts(iso: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert _run(["config", "set", "preset", "ultra"]) == 0
    assert _run(["config", "set", "preset", "porifle"]) == 2
    assert "unknown preset" in capsys.readouterr().err
    assert _global_toml(iso)["preset"] == "ultra"  # rolled back to the prior value


def test_unset_top_level_profile(iso: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _run(["config", "set", "preset", "quick"])
    assert _run(["config", "unset", "preset"]) == 0
    assert "preset" not in _global_toml(iso)
    capsys.readouterr()
    _run(["config", "get", "preset"])
    assert "[default]" in capsys.readouterr().out


def test_set_profile_heals_a_profile_table_typo(iso: Path) -> None:
    # A leftover `[preset]` TABLE (from `config set preset.<name>`) breaks the
    # config; the advertised fix `config set preset <name>` must heal it in one
    # step, not stack a bare key on top of the table (unparseable TOML, kept by
    # the lenient already-invalid path).
    (iso / "g").mkdir(parents=True, exist_ok=True)
    (iso / "g" / "config.toml").write_text('[preset]\nporifle = "ultra"\n', encoding="utf-8")
    assert _run(["config", "set", "preset", "ultra"]) == 0
    assert _global_toml(iso) == {"preset": "ultra"}


def test_set_profile_table_typo_reports_profile_error(
    iso: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # `config set preset.porifle x` over a valid config must fail with the
    # preset-must-be-a-string message and roll back, even when the bare
    # `preset` key it collides with is already set.
    assert _run(["config", "set", "preset", "ultra"]) == 0
    assert _run(["config", "set", "preset.porifle", "x"]) == 2
    assert "must be a preset name string" in capsys.readouterr().err
    assert _global_toml(iso) == {"preset": "ultra"}


def test_set_profile_repo_targets_repo_config(iso: Path) -> None:
    assert _run(["config", "set", "preset", "quick", "--repo"]) == 0
    repo_cfg = (resolved_state_dir(iso) / "config.toml").read_text(encoding="utf-8")
    assert 'preset = "quick"' in repo_cfg
    assert not (iso / "g" / "config.toml").is_file()


def test_set_profile_machine_file_is_refused(
    iso: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # A machine [config] overlay must not smuggle a preset selection
    # (_forbid_layer_preset); the write is rolled back.
    mf = tmp_path / "m.asm.toml"
    mf.write_text("[config]\n", encoding="utf-8")
    assert _run(["config", "set", "preset", "ultra", "--machine-file", str(mf)]) == 2
    assert "preset" in capsys.readouterr().err
    assert "preset" not in mf.read_text(encoding="utf-8").replace("[config]", "")


# --- presets listing ---------------------------------------------------------


def test_config_profiles_lists_builtins_and_user(
    iso: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (iso / "g").mkdir(parents=True, exist_ok=True)
    (iso / "g" / "config.toml").write_text(
        'preset = "ultra"\n\n[presets.myteam.review]\nconcurrency = 2\n', encoding="utf-8"
    )
    assert _run(["config", "presets"]) == 0
    out = capsys.readouterr().out
    for builtin in ("standard", "quick", "ultra", "paranoid"):
        assert builtin in out
    assert "selected" in out  # ultra marked as the selection, with its source
    assert "global" in out
    assert "review.concurrency = 3" in out  # ultra's contents are shown
    assert "myteam" in out  # user preset listed with its contents
    assert "review.concurrency = 2" in out


def test_config_profiles_none_selected(iso: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert _run(["config", "presets"]) == 0
    out = capsys.readouterr().out
    assert "no preset selected" in out
    assert "standard" in out  # built-ins still listed


def test_config_profiles_user_shadow_replaces_builtin(
    iso: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # A user [presets.ultra] REPLACES the built-in wholesale; the listing must
    # show the user's contents (not the dead built-in's) and say so.
    (iso / "g").mkdir(parents=True, exist_ok=True)
    (iso / "g" / "config.toml").write_text(
        "[presets.ultra.review]\nconcurrency = 9\n", encoding="utf-8"
    )
    assert _run(["config", "presets"]) == 0
    out = capsys.readouterr().out
    assert "review.concurrency = 9" in out
    assert "review.concurrency = 3" not in out  # the built-in body is dead, not shown
    assert "replaces the built-in" in out


# --- repo target ------------------------------------------------------------


# --- add / remove (list field: allow_urls) ----------------------------------


# --- machine [config] overlay target ----------------------------------------


def _machine_file(tmp_path: Path) -> Path:
    p = tmp_path / "demo.asm.toml"
    p.write_text(
        '[machine]\nname = "demo"\nentry = "s"\n\n[states.s]\nkind = "terminal"\noutcome = "ok"\n',
        encoding="utf-8",
    )
    return p


def test_machine_overlay_set_and_get(iso: Path, capsys: pytest.CaptureFixture[str]) -> None:
    mf = _machine_file(iso)
    # A non-security knob is fine in a machine overlay (review tuning).
    assert (
        _run(["config", "set", "review.trigger", "on_verify_fail", "--machine-file", str(mf)]) == 0
    )
    data = tomllib.loads(mf.read_text(encoding="utf-8"))
    assert data["config"] == {"review": {"trigger": "on_verify_fail"}}  # type: ignore[comparison-overlap]
    # The original machine tables survive the edit.
    assert data["machine"]["name"] == "demo"  # type: ignore[index]
    capsys.readouterr()
    assert _run(["config", "get", "review.trigger", "--machine-file", str(mf)]) == 0
    assert "[machine]" in capsys.readouterr().out


def test_machine_overlay_rejects_providers(iso: Path) -> None:
    mf = _machine_file(iso)
    assert _run(["config", "set", "providers.x.kind", "anthropic", "--machine-file", str(mf)]) == 2


# --- egress endpoint wiring -------------------------------------------------


def test_config_fill_has_no_repo_flag(iso: Path) -> None:
    """Filling the repo layer makes it explicitly set everything, shadowing the
    global config permanently -- future edits included -- which defeats the
    layering `config show` exists to explain. The flag stays gone; the global
    fill keeps resolving defaults plus global, never the repo layer."""
    with pytest.raises(SystemExit) as exc:
        _run(["config", "fill", "--repo"])
    assert exc.value.code == 2


def test_config_fill_serializes_against_a_concurrent_set(iso: Path) -> None:
    """`config fill` read the effective config, then published it with an
    unlocked, non-atomic write_text; a `config set` landing between the read
    and the write was overwritten by the stale snapshot (lost update). fill now
    holds the target's lock across load+publish, so a concurrent set blocks and
    lands after -- its value survives."""
    import threading
    import time
    from unittest import mock

    from agent6.ui.cli import config_cmds

    assert _run(["config", "set", "sandbox.memory_limit_mb", "512"]) == 0

    fill_holds_lock = threading.Event()
    release_fill = threading.Event()
    real_load = config_cmds.load_global_only
    results: dict[str, object] = {}

    def gated_load() -> object:
        fill_holds_lock.set()  # reached inside `with locked_file(target)`
        release_fill.wait(timeout=5)
        return real_load()

    def run_fill() -> None:
        with mock.patch.object(config_cmds, "load_global_only", gated_load):
            results["fill"] = _run(["config", "fill", "--force"])

    def run_set() -> None:
        fill_holds_lock.wait(timeout=5)
        results["set"] = _run(["config", "set", "sandbox.memory_limit_mb", "1234"])
        results["set_done"] = True

    tf = threading.Thread(target=run_fill, daemon=True)
    ts = threading.Thread(target=run_set, daemon=True)
    tf.start()
    ts.start()
    assert fill_holds_lock.wait(timeout=5)
    time.sleep(0.3)  # let the set reach (and block on) the target lock
    assert results.get("set_done") is None  # the set is queued behind fill
    release_fill.set()
    tf.join(timeout=10)
    ts.join(timeout=10)
    assert results["fill"] == 0 and results["set"] == 0
    sandbox = _global_toml(iso)["sandbox"]
    assert isinstance(sandbox, dict)
    assert sandbox["memory_limit_mb"] == 1234  # the set survived


def test_unset_refuses_a_leaf_inside_an_undeclared_table(
    iso: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """`sandbox.protect_git = false` written as a dotted top-level key: unset
    said "nothing to unset" rc=0 while `config get` showed the leaf set -- the
    one write surface that lied about this shape (set refuses it, fix reports
    it stuck)."""
    (iso / "g").mkdir(parents=True, exist_ok=True)
    cfg = iso / "g" / "config.toml"
    cfg.write_text("sandbox.protect_git = false\n", encoding="utf-8")
    rc = _refuse(["config", "unset", "sandbox.protect_git"])
    assert rc == 2
    assert "cannot be unset on its own" in capsys.readouterr().err
    # The file is untouched: nothing was silently dropped or rewritten.
    assert cfg.read_text(encoding="utf-8") == "sandbox.protect_git = false\n"


def test_add_rejects_a_value_masked_by_a_higher_layer(
    iso: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """`config add` writes the whole new list through the same standalone value
    check as `config set`: a repo overlay masking the leaf must not let a bad
    global element land, to explode only once the mask is gone."""
    assert _run(["config", "set", "--repo", "sandbox.fetch_hosts", '["ok.example"]']) == 0
    capsys.readouterr()
    rc = _refuse(["config", "add", "sandbox.fetch_hosts", "5"])
    assert rc == 2
    assert "fetch_hosts" in capsys.readouterr().err
    gcfg = iso / "g" / "config.toml"
    assert not gcfg.is_file() or "5" not in gcfg.read_text(encoding="utf-8")


def test_set_warns_when_another_layer_is_still_broken(
    iso: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A valid write over a config broken in ANOTHER layer lands, exits 0, and
    warns with the other layer's error; delegating the CLI writers to the
    engine must not silence the warning."""
    (iso / "g").mkdir(parents=True, exist_ok=True)
    (iso / "g" / "config.toml").write_text('[cli]\ninput = "x"\n', encoding="utf-8")
    rc = _run(["config", "set", "--repo", "sandbox.protect_git", "false"])
    assert rc == 0
    assert "a value this edit did not write" in capsys.readouterr().err


def test_get_honours_the_global_config_flag(iso: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """`--config FILE` reaches `config get`, not just `config show`.

    `get` answered from the default/global stack while `show` reported the
    flag layer, so the two config readers disagreed about the same leaf."""
    explicit = iso / "x.toml"
    explicit.write_text("[review]\nperiod = 77\n", encoding="utf-8")
    assert _run(["--config", str(explicit), "config", "get", "review.period"]) == 0
    out = capsys.readouterr().out
    assert "review.period = 77" in out
    assert "[flag]" in out


def test_get_refuses_a_missing_global_config_file(iso: Path) -> None:
    """A `--config` file that does not exist is refused, as `config show`
    refuses it: answering from the defaults reports a value the named file
    never set."""
    assert _refuse(["--config", str(iso / "nope.toml"), "config", "get", "review.period"]) == 2


def test_get_refuses_a_machine_file_that_does_not_exist(
    iso: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A missing overlay path read as an EMPTY overlay, so a typo'd
    --machine-file answered confidently from the stack below it at exit 0."""
    assert (
        _refuse(["config", "get", "--machine-file", str(iso / "nope.asm.toml"), "review.period"])
        == 2
    )
    assert "no such machine file" in capsys.readouterr().err


def test_a_provider_leaf_error_names_every_valid_value(
    iso: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """`api_format` is a discriminator with two legal values, and only the first
    member's complaint was reported -- telling someone configuring an
    OpenAI-compatible provider that 'anthropic' was the only option."""
    assert _run(["config", "set", "providers.p.api_format", "nonsense"]) == 2
    err = capsys.readouterr().err
    assert "anthropic" in err
    assert "openai" in err


def test_an_unreadable_config_refuses_rather_than_crashing(iso: Path) -> None:
    """A root-owned config after a sudo run is the operator's file, not a defect.

    The reader wrapped a TOML parse error but not an OSError, so a permission
    problem escaped as "unexpected PermissionError" with a saved traceback,
    "please report this", and exit 1."""
    gdir = iso / "g"
    gdir.mkdir(parents=True, exist_ok=True)
    cfg = gdir / "config.toml"
    cfg.write_text("[review]\nperiod = 7\n", encoding="utf-8")
    cfg.chmod(0o000)
    try:
        assert _refuse(["config", "show"]) == 2
    finally:
        cfg.chmod(0o600)


_CRASH_MARKERS = ("unexpected", "full traceback", "report this")


@pytest.mark.parametrize(
    "argv",
    [
        ["config", "set", "workflow.max_iterations", "7"],
        ["config", "unset", "review.period"],
        ["config", "add", "sandbox.allow_urls", "https://example.com"],
        ["config", "remove", "sandbox.allow_urls", "https://example.com"],
    ],
    ids=["set", "unset", "add", "remove"],
)
def test_write_commands_refuse_an_unreadable_target(
    iso: Path, capsys: pytest.CaptureFixture[str], argv: list[str]
) -> None:
    """The unreadable-config fix landed in the readers while every write command
    still read the target directly first: `config set`/`unset` crashed through
    the bug reporter at exit 1 on a root-owned config the readers refused."""
    gdir = iso / "g"
    gdir.mkdir(parents=True, exist_ok=True)
    cfg = gdir / "config.toml"
    cfg.write_text("[review]\nperiod = 7\n", encoding="utf-8")
    cfg.chmod(0o000)
    try:
        assert _refuse(argv) == 2
    finally:
        cfg.chmod(0o600)
    err = capsys.readouterr().err
    assert err.startswith("ERROR: ")
    assert "config.toml" in err
    assert not any(marker in err for marker in _CRASH_MARKERS)


def test_a_write_command_bug_still_crash_reports(
    iso: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Routing operator errors to the boundary must not soften real bugs: an
    unexpected exception inside `config set` keeps the crash report at exit 1."""
    from agent6.config import write as write_mod

    def _boom(*_a: object, **_k: object) -> None:
        raise RuntimeError("kaboom")

    monkeypatch.setattr(write_mod, "upsert_toml_leaf", _boom)
    monkeypatch.delenv("AGENT6_DEBUG", raising=False)
    assert _refuse(["config", "set", "workflow.max_iterations", "7"]) == 1
    err = capsys.readouterr().err
    assert "unexpected RuntimeError" in err
    tb_line = next(line for line in err.splitlines() if "full traceback:" in line)
    Path(tb_line.split("full traceback:", 1)[1].strip()).unlink()


def test_set_of_an_unserializable_cli_value_refuses(
    iso: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """parse_cli_value reads `2024-01-01` as a TOML date, which the writer cannot
    serialize. That refusal used to live in a per-command except arm; it must
    survive the arm's deletion as a refusal, never become a crash report."""
    assert _refuse(["config", "set", "workflow.max_iterations", "2024-01-01"]) == 2
    err = capsys.readouterr().err
    assert err.startswith("ERROR: ")
    assert not any(marker in err for marker in _CRASH_MARKERS)


def test_commit_trailer_validates_placeholders_and_shape(
    iso: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """[git.commit].trailer takes a git trailer line with {model} as its one
    placeholder; an unknown placeholder or a shapeless string is refused at
    config set, not at commit time."""
    assert _run(["config", "set", "git.commit.trailer", "Assisted-by: agent6:{model}"]) == 0
    capsys.readouterr()
    assert _refuse(["config", "set", "git.commit.trailer", "Assisted-by: {agent}"]) == 2
    assert "agent" in capsys.readouterr().err
    assert _refuse(["config", "set", "git.commit.trailer", "By: {model} ({role})"]) == 2
    assert "role" in capsys.readouterr().err
    assert _refuse(["config", "set", "git.commit.trailer", "no trailer shape"]) == 2
    assert "Key: value" in capsys.readouterr().err


def test_checkpoint_style_refuses_combine(iso: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """combine is git's own squash message; a checkpoint has nothing to
    combine, so the checkpoint table refuses it while squash accepts it."""
    assert _run(["config", "set", "git.commit.squash.message", "combine"]) == 0
    assert _refuse(["config", "set", "git.commit.checkpoint.message", "combine"]) == 2


def test_coauthor_is_gone(iso: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Replaced by the trailer format string; no migration, pre-1.0."""
    assert _refuse(["config", "set", "git.commit.coauthor", "A <a@b>"]) == 2
    assert "trailer" in capsys.readouterr().err  # the did-you-mean points at it


def test_the_paired_context_thresholds_are_settable_together(iso: Path) -> None:
    """Both leaves must move together, so neither can be set alone. The
    inline-table form writes the pair in ONE validated upsert."""
    inline = "{ drop_at_chars = 200000, summarise_at_chars = 400000 }"
    assert _run(["config", "set", "context", inline]) == 0
    ctx = _global_toml(iso)["context"]
    assert isinstance(ctx, dict)
    assert ctx == {"drop_at_chars": 200000, "summarise_at_chars": 400000}


def test_setting_one_threshold_names_the_command_that_works(
    iso: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A refusal that does not name the working form leaves the operator with
    no way in: both single-leaf orderings refuse."""
    assert _refuse(["config", "set", "context.drop_at_chars", "200000"]) == 2
    err = capsys.readouterr().err
    assert "config set context '{ drop_at_chars =" in err
    assert "summarise_at_chars =" in err


def _fill_and_read(iso: Path) -> dict[str, object]:
    assert _run(["config", "fill", "--force"]) == 0
    return _global_toml(iso)


def test_fill_never_bakes_the_repo_layer_into_the_global_config(iso: Path) -> None:
    """fill writes the GLOBAL file; loading the full merge baked this repo's
    overrides into it, so a value set for one repo followed the operator
    everywhere."""
    assert _run(["config", "set", "--repo", "sandbox.memory_limit_mb", "4321"]) == 0
    filled = _fill_and_read(iso)
    sandbox = filled["sandbox"]
    assert isinstance(sandbox, dict)
    assert sandbox["memory_limit_mb"] != 4321, "the repo layer leaked into the global config"


def test_fill_keeps_the_preset_selector_and_does_not_bake_its_effects(iso: Path) -> None:
    """A selected preset stays selected: baking its effects and dropping the
    selector froze the old values and made later preset edits do nothing."""
    assert _run(["config", "set", "preset", "quick"]) == 0
    filled = _fill_and_read(iso)
    assert filled.get("preset") == "quick", "the selector was dropped"
    review = filled["review"]
    assert isinstance(review, dict)
    # `quick` sets review.trigger = "off"; the filled value must be the
    # DEFAULT, with the preset still applying over it at runtime.
    from agent6.config import Config

    assert review["trigger"] == Config().review.trigger


def test_fill_keeps_authored_preset_bodies(iso: Path) -> None:
    (iso / "g").mkdir(parents=True, exist_ok=True)
    (iso / "g" / "config.toml").write_text(
        "[presets.myteam.review]\nconcurrency = 2\n", encoding="utf-8"
    )
    filled = _fill_and_read(iso)
    presets = filled["presets"]
    assert isinstance(presets, dict)
    assert "myteam" in presets


def test_config_path_lists_every_directory_agent6_writes_to(
    iso: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Four XDG bases each holding a different thing is correct and
    unguessable, so one command answers "where did agent6 put that": the
    config/repo/secrets files plus every directory, this repo's state dir
    included."""
    monkeypatch.setenv("AGENT6_STATE_HOME", str(iso / "st"))
    monkeypatch.setenv("AGENT6_DATA_HOME", str(iso / "dt"))
    monkeypatch.setenv("AGENT6_CACHE_HOME", str(iso / "ch"))
    assert _run(["config", "path"]) == 0
    out = capsys.readouterr().out
    for label in ("global config", "repo config", "secrets", "config dir", "cache"):
        assert f"{label}" in out
    assert str(iso / "st") in out  # state base
    assert str(resolved_state_dir(iso)) in out  # this repo's own dir
    assert str(iso / "dt" / "skills") in out
    assert str(iso / "ch") in out


def test_top_level_help_names_the_directories(
    iso: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Discoverable without knowing `config path` exists: `agent6 --help` ends
    with the four XDG dirs, resolved, and points at the fuller listing."""
    monkeypatch.setenv("AGENT6_STATE_HOME", str(iso / "st"))
    monkeypatch.setenv("AGENT6_DATA_HOME", str(iso / "dt"))
    monkeypatch.setenv("AGENT6_CACHE_HOME", str(iso / "ch"))
    with pytest.raises(SystemExit) as exc:
        _run(["--help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "directories" in out
    assert str(iso / "g") in out and str(iso / "st") in out
    assert str(iso / "dt") in out and str(iso / "ch") in out
    assert "agent6 config path" in out
