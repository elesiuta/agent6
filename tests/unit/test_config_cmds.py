# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""`config set/add/remove --machine` re-validates the whole machine spec."""

from __future__ import annotations

from pathlib import Path

import pytest

from agent6.ui.cli import config_cmds as cc


def _noop_overlay(*_a: object, **_k: object) -> None:
    # Stub for load_effective_with_overlay so the test isolates machine-spec
    # validation from the cwd-dependent [config]-overlay validation.
    return None


def test_extra_body_value_completer_offers_routing_presets() -> None:
    # TAB after `config set providers.<name>.extra_body` suggests the routing
    # presets for any provider name (matched by suffix).
    import argparse

    from agent6.ui.cli.completers import (
        _complete_config_values,  # pyright: ignore[reportPrivateUsage]
    )

    args = argparse.Namespace(key="providers.openrouter.extra_body")
    out = _complete_config_values("", args)  # pyright: ignore[reportPrivateUsage]
    assert '{ provider = { sort = "throughput" } }' in out
    # a non-extra_body key is unaffected
    enum_args = argparse.Namespace(key="sandbox.isolation")
    assert _complete_config_values("", enum_args) == [  # pyright: ignore[reportPrivateUsage]
        "auto",
        "strict",
        "hardened",
    ]


def test_profile_value_completer_offers_profile_names(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # TAB after `config set preset` offers the selectable names: built-ins
    # plus user [presets.*] tables.
    import argparse

    from agent6.ui.cli.completers import (
        _complete_config_values,  # pyright: ignore[reportPrivateUsage]
    )

    gdir = tmp_path / "g"
    gdir.mkdir()
    (gdir / "config.toml").write_text("[presets.myteam.review]\npanel_size = 2\n")
    monkeypatch.setenv("AGENT6_CONFIG_HOME", str(gdir))
    monkeypatch.chdir(tmp_path)
    args = argparse.Namespace(key="preset")
    out = _complete_config_values("", args)  # pyright: ignore[reportPrivateUsage]
    assert "ultra" in out and "myteam" in out
    assert _complete_config_values("ul", args) == ["ultra"]  # pyright: ignore[reportPrivateUsage]


def test_config_key_completer_offers_user_profile_paths(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # `config set presets.<TAB>` completes leaf paths for USER-defined presets
    # only. Built-in names never complete in the KEY position: writing
    # presets.ultra.* creates a user table that REPLACES the built-in
    # wholesale, a footgun TAB should not put one keystroke away (the same
    # rule keeps `none` out of sandbox.isolation completion).
    from agent6.ui.cli.completers import (
        _complete_config_keys,  # pyright: ignore[reportPrivateUsage]
    )

    gdir = tmp_path / "g"
    gdir.mkdir()
    (gdir / "config.toml").write_text("[presets.myteam.review]\npanel_size = 2\n")
    monkeypatch.setenv("AGENT6_CONFIG_HOME", str(gdir))
    monkeypatch.chdir(tmp_path)
    out = _complete_config_keys("presets.")  # pyright: ignore[reportPrivateUsage]
    assert any(k.startswith("presets.myteam.review.") for k in out)
    assert not any(k.startswith("presets.ultra") for k in out)
    # the top-level `preset` leaf itself is offered alongside presets.*
    assert "preset" in _complete_config_keys("preset")  # pyright: ignore[reportPrivateUsage]
    # a bare TAB (empty prefix) is not flooded with the generated paths
    assert not any(
        k.startswith("presets.")
        for k in _complete_config_keys("")  # pyright: ignore[reportPrivateUsage]
    )


def test_parallel_models_completer_completes_after_last_comma(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # TAB on `run --parallel` completes model ids for the WORKER provider only
    # (lanes inherit it; only the model is overridden per lane), and completes
    # only the token AFTER the last comma so a `m1,m2,...` list grows member by
    # member (the head is preserved on each completion).
    from agent6.config import Config
    from agent6.ui.cli import completers

    cfg = Config.model_validate(
        {
            "providers": {
                "w": {"api_format": "openai", "base_url": "https://w.example/v1"},
                "s": {"api_format": "openai", "base_url": "https://s.example/v1"},
            },
            "models": {"worker": {"provider": "w", "model": "gpt-5"}},
        }
    )

    class _Eff:
        config = cfg

    def _eff(*_a: object, **_k: object) -> _Eff:
        return _Eff()

    def _models(_cp: object, provider: object) -> list[str]:
        # Per-provider catalogs: the sibling's must never be offered.
        return {"w": ["gpt-5", "gpt-5-mini", "opus"], "s": ["gpt-sibling-only"]}[str(provider)]

    monkeypatch.setattr(completers, "load_effective", _eff)
    monkeypatch.setattr("agent6.ui.cli.model._models_for", _models)
    assert completers._complete_parallel_models("gpt") == [  # pyright: ignore[reportPrivateUsage]
        "gpt-5",
        "gpt-5-mini",
    ]
    assert completers._complete_parallel_models("opus,gpt") == [  # pyright: ignore[reportPrivateUsage]
        "opus,gpt-5",
        "opus,gpt-5-mini",
    ]


_GOOD = (
    'machine = "m"\nversion = 1\ninitial = "s"\n'
    "[budget]\nmax_usd = 1.0\nmax_transitions = 10\n"
    '[states.s]\nkind = "terminal"\nstatus = "ok"\nreason = "done"\n'
)
# Same machine but with an unknown state kind -> a complete-but-invalid spec.
_BAD = (
    'machine = "m"\nversion = 1\ninitial = "s"\n'
    "[budget]\nmax_usd = 1.0\nmax_transitions = 10\n"
    '[states.s]\nkind = "bogus"\n'
)


def test_config_set_names_the_inline_table_a_leaf_lives_in(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """`config show` and TAB both offer the leaves inside an inline table (the
    routing preset agent6 itself suggests writes one), but the leaf surgery only
    knows [table] headers, so setting one emitted a header that collides with
    the inline parent. The write is refused either way; say WHICH value owns the
    leaf instead of leaking `Cannot declare (...) twice`."""
    from agent6.ui.cli import cli_main

    gdir = tmp_path / "g"
    gdir.mkdir()
    cfg = gdir / "config.toml"
    cfg.write_text(
        "[providers.openrouter]\n"
        'api_format = "openai"\n'
        'base_url = "https://o.example/v1"\n'
        'extra_body = { provider = { sort = "throughput" } }\n',
        encoding="utf-8",
    )
    before = cfg.read_text(encoding="utf-8")
    # The whole layer stack must read this same file, not just the write path.
    monkeypatch.setenv("AGENT6_CONFIG_HOME", str(gdir))
    monkeypatch.chdir(tmp_path)

    rc = cli_main(["config", "set", "providers.openrouter.extra_body.provider.sort", "price"])
    err = capsys.readouterr().err
    assert rc == 2
    assert "providers.openrouter.extra_body" in err
    assert "Cannot declare" not in err, "the raw TOML error is not an explanation"
    assert cfg.read_text(encoding="utf-8") == before


def test_config_set_refuses_a_target_that_does_not_parse(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """`config set` never parsed the file it edits. Over a target with a typo'd
    bracket it appended line-surgery output, could not find the malformed header,
    and left the file still unparseable -- then reported success, because the
    "config was already invalid, so blame another layer" branch swallows a parse
    error in the very file just written. Repeating it kept appending."""
    from agent6.ui.cli import cli_main

    cfg = tmp_path / "config.toml"
    cfg.write_text("[sandbox\nprotect_git = true\n", encoding="utf-8")  # missing ]

    def _global_path(*_a: object, **_k: object) -> Path:
        return cfg

    from agent6.config import write as write_mod

    monkeypatch.setattr(cc, "global_config_path", _global_path)
    monkeypatch.setattr(write_mod, "global_config_path", _global_path)

    rc = cli_main(["config", "set", "sandbox.run_commands", "yes"])
    out = capsys.readouterr()
    assert rc == 2, "a target that does not parse must not report success"
    assert "Set sandbox.run_commands" not in out.out
    # And the file is untouched: no surgery appended into a file we cannot read.
    assert cfg.read_text(encoding="utf-8") == "[sandbox\nprotect_git = true\n"


def test_reject_machine_protected_covers_every_spec_forbidden_key(tmp_path: Path) -> None:
    """The CLI guard documents itself as mirroring the MachineSpec validator but
    checked only providers/sandbox, so `config set --machine-file` wrote
    presets.*, machine.notify.*, and git.run_repo_hooks into an overlay the
    loader then always rejects -- and the compensating load re-check is skipped
    while the file is not yet a valid machine (a `machine create` draft), so the
    operator got a success and a file that can never load."""
    m = tmp_path / "m.asm.toml"
    for key in (
        "providers.openai.base_url",
        "sandbox.run_commands",
        "presets.ultra.sandbox.run_commands",
        "machine.notify.on_event",
        "mcp.servers",
        "notify.on_complete",
        "git.run_repo_hooks",
    ):
        assert cc._reject_machine_protected(key, m) is not None, key  # pyright: ignore[reportPrivateUsage]
    # Benign overlay keys stay writable (the forbid is surgical).
    assert cc._reject_machine_protected("git.commit.name", m) is None  # pyright: ignore[reportPrivateUsage]
    assert cc._reject_machine_protected("review.panel_size", m) is None  # pyright: ignore[reportPrivateUsage]


def test_revalidate_machine_rejects_invalid_spec_and_rolls_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Isolate the machine-spec validation from the cwd-dependent [config]-overlay
    # validation by stubbing the latter.
    monkeypatch.setattr(cc, "load_effective_with_overlay", _noop_overlay)
    target = tmp_path / "m.asm.toml"
    target.write_text(_BAD, encoding="utf-8")

    err = cc._revalidate_machine(target, _GOOD)  # pyright: ignore[reportPrivateUsage]

    assert err is not None  # the invalid machine was caught (not silently left)
    assert target.read_text(encoding="utf-8") == _GOOD  # and the file was rolled back


def test_revalidate_machine_accepts_valid_spec(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cc, "load_effective_with_overlay", _noop_overlay)
    target = tmp_path / "m.asm.toml"
    target.write_text(_GOOD, encoding="utf-8")

    # prior_text=_GOOD makes _machine_is_valid true, so load_machine(target)
    # actually runs (None would skip the machine check and pass vacuously).
    assert cc._revalidate_machine(target, _GOOD) is None  # pyright: ignore[reportPrivateUsage]
    assert target.read_text(encoding="utf-8") == _GOOD  # untouched


def test_config_show_unknown_key_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from agent6.ui.cli import main

    monkeypatch.chdir(tmp_path)
    assert main(["config", "show", "nope.nope"]) == 2
    assert "no config key matches" in capsys.readouterr().err


def test_config_set_keeps_a_valid_write_despite_a_stale_value_elsewhere(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # A value left invalid by a schema change (prompt.decompose = true, once a bool,
    # now Literal["auto","on","off"]) must NOT block setting an unrelated valid key:
    # the write is kept (not reverted) and a WARNING names the exact file + command
    # to fix the stale value, so it is self-service. The old strict behaviour made a
    # broken config impossible to fix through `config set`.
    from agent6.paths import global_config_path
    from agent6.ui.cli import main

    gpath = global_config_path()
    gpath.write_text("[prompt]\ndecompose = true\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    rc = main(["config", "set", "budget.max_usd", "5"])
    captured = capsys.readouterr()
    assert rc == 0  # the valid write is kept...
    assert "Set budget" in captured.out  # ...it succeeded,
    assert "prompt.decompose" in captured.err  # ...and a warning names the stale value,
    assert str(gpath) in captured.err  # the exact file,
    assert "config set prompt.decompose <value>" in captured.err  # and how to fix it.

    # Overwriting the offending value clears the warning; the write is clean.
    assert main(["config", "set", "prompt.decompose", "off"]) == 0
    assert "WARNING" not in capsys.readouterr().err


def test_config_set_rejects_a_masked_invalid_value(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # A repo overlay masks the key in the merged view; setting an INVALID value in
    # a lower layer must still be rejected, or it lands unvalidated and only
    # explodes later where the mask is absent.
    import subprocess

    from agent6.ui.cli import main

    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    monkeypatch.chdir(tmp_path)
    assert main(["config", "set", "--repo", "sandbox.run_commands", "yes"]) == 0  # the mask
    capsys.readouterr()
    # Global set of an invalid value -> rejected despite the repo mask.
    rc = main(["config", "set", "sandbox.run_commands", "bogus"])
    assert rc == 2
    err = capsys.readouterr().err
    assert "sandbox.run_commands" in err  # a friendly per-field error, not a merge dump
    # A valid global set still succeeds.
    assert main(["config", "set", "sandbox.run_commands", "no"]) == 0


def test_config_set_rejects_a_newly_invalid_value(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # Setting an invalid VALUE still fails loud and reverts, so a typo cannot land in
    # the config even though a stale value elsewhere no longer blocks a valid write.
    from agent6.paths import global_config_path
    from agent6.ui.cli import main

    monkeypatch.chdir(tmp_path)
    rc = main(["config", "set", "prompt.decompose", "bogus"])
    assert rc == 2  # the write itself is invalid -> reverted + fail loud
    assert "prompt.decompose" in capsys.readouterr().err
    gpath = global_config_path()
    assert not gpath.is_file() or "decompose" not in gpath.read_text(encoding="utf-8")


def test_a_refused_write_still_hands_the_config_back_to_the_operator(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Under sudo the config dir is created as root and every write publishes a
    fresh root-owned inode -- the rollback of a refused value included. The
    handover ran only after a successful write, so `sudo agent6 config set` with
    a bad value left the operator's own config owned by root."""
    from agent6.config import write as write_mod
    from agent6.paths import global_config_path
    from agent6.ui.cli import main

    handed: list[Path] = []
    monkeypatch.setattr(write_mod, "chown_to_real_user", handed.append)
    monkeypatch.setattr(write_mod, "mkdir_for_real_user", handed.append)  # the dir handover
    gpath = global_config_path()
    gpath.write_text("[budget]\nmax_usd = 5.0\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    assert main(["config", "set", "prompt.decompose", "bogus"]) == 2  # rolled back
    assert gpath in handed  # the rolled-back file is the operator's again
    assert gpath.parent in handed  # and so is the dir the write may have created


def test_config_set_reverts_a_write_that_trips_a_non_pydantic_check(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # A write that trips a STANDALONE ConfigError (not a pydantic per-leaf error) --
    # e.g. a non-absolute agent6.state_dir -- must still revert. Such errors carry no
    # "  - <leaf>:" line, so the before/after comparison must count full error content
    # (else the invalid write is silently kept and bricks the config).
    from agent6.ui.cli import main

    monkeypatch.chdir(tmp_path)
    rc = main(["config", "set", "agent6.state_dir", "not-absolute"])
    assert rc == 2
    assert "absolute" in capsys.readouterr().err.lower()


def test_config_set_keeps_a_write_on_an_already_invalid_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # config set never lands a known-INVALID written value (it validates the value
    # standalone), but a config left invalid by a stale value elsewhere STAYS
    # fixable: a VALID write still succeeds. So on an already-broken config,
    # writing another invalid value is rejected, while a valid value lands + clears.
    from agent6.paths import global_config_path
    from agent6.ui.cli import main

    gpath = global_config_path()
    gpath.write_text("[prompt]\ndecompose = true\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    assert main(["config", "set", "prompt.decompose", "enabled"]) == 2  # invalid value: rejected
    assert "prompt.decompose" in capsys.readouterr().err  # friendly per-field error
    assert main(["config", "set", "prompt.decompose", "on"]) == 0  # a valid value clears it
    assert "WARNING" not in capsys.readouterr().err


def test_config_set_unknown_leaf_gets_a_did_you_mean(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # An unknown key under a known section spoke pydantic ("Extra inputs are
    # not permitted"); a typo deserves the near-miss and the show pointer.
    from agent6.ui.cli import main

    monkeypatch.chdir(tmp_path)
    rc = main(["config", "set", "sandbox.run_command", "yes"])  # missing 's'
    assert rc == 2
    err = capsys.readouterr().err
    assert "unknown config key 'sandbox.run_command'" in err
    assert "'sandbox.run_commands'" in err  # the did-you-mean
    assert "Extra inputs" not in err


def test_config_set_unknown_section_gets_the_same_friendly_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # An unknown TOP-LEVEL section errors at the section loc (a parent of the
    # written key); it used to fall through to the merged-layer dump with
    # "(merged config layers)" and a raw type=extra_forbidden.
    from agent6.paths import global_config_path
    from agent6.ui.cli import main

    monkeypatch.chdir(tmp_path)
    rc = main(["config", "set", "bogus.key", "foo"])
    assert rc == 2
    err = capsys.readouterr().err
    assert "unknown config key 'bogus.key'" in err
    assert "merged config layers" not in err and "extra_forbidden" not in err
    gpath = global_config_path()
    assert not gpath.is_file() or "bogus" not in gpath.read_text(encoding="utf-8")


def test_config_set_accepts_a_profiles_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # [presets.*] is meta-config stripped before validation (_apply_preset),
    # so the schema forbids it by design; the unknown-key reroute must not
    # reject a legitimate, documented preset write (it did: rc 2 + revert).
    from agent6.paths import global_config_path
    from agent6.ui.cli import main

    monkeypatch.chdir(tmp_path)
    rc = main(["config", "set", "presets.mine.review.trigger", "before_finish"])
    assert rc == 0
    text = global_config_path().read_text(encoding="utf-8")
    assert "[presets.mine.review]" in text
    assert "unknown config key" not in capsys.readouterr().err


def test_config_set_bool_error_speaks_human(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from agent6.ui.cli import main

    monkeypatch.chdir(tmp_path)
    rc = main(["config", "set", "sandbox.protect_git", "notabool"])
    assert rc == 2
    assert "sandbox.protect_git: expected true or false, got 'notabool'" in capsys.readouterr().err


def test_config_set_global_keeps_a_valid_write_shadowed_by_a_stale_repo_layer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # The exact motivating case: prompt.decompose is stale in the REPO layer; setting a
    # VALID value GLOBALLY (which the repo still shadows) must be KEPT + warn, NEVER
    # reverted -- the leaf appears in the merged error but this write is not its cause.
    from agent6.config.layer import repo_config_path_for
    from agent6.paths import global_config_path
    from agent6.ui.cli import main

    repo_cfg = repo_config_path_for(tmp_path)
    repo_cfg.parent.mkdir(parents=True, exist_ok=True)
    repo_cfg.write_text("[prompt]\ndecompose = true\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    rc = main(["config", "set", "prompt.decompose", "auto"])
    captured = capsys.readouterr()
    assert rc == 0  # the valid global write is KEPT, not reverted over the repo's stale value
    assert "WARNING" in captured.err  # ...but warns the repo layer still shadows it
    assert '"auto"' in global_config_path().read_text(encoding="utf-8")


def test_config_set_sub_leaf_on_an_existing_provider(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # providers.<name> is a discriminated union on api_format, so a leaf's isolated dict
    # lacks the union tag and errors on the parent providers.<name>; the pre-check must
    # not attribute that to the written child, or every providers.<name>.* set on an
    # already-complete provider would be rejected.
    from agent6.paths import global_config_path
    from agent6.ui.cli import main

    gpath = global_config_path()
    gpath.parent.mkdir(parents=True, exist_ok=True)
    gpath.write_text(
        '[providers.op]\napi_format = "openai"\nbase_url = "https://x.test/v1"\n', encoding="utf-8"
    )
    monkeypatch.chdir(tmp_path)
    assert main(["config", "set", "providers.op.base_url", "https://y.test/v1"]) == 0


def test_config_set_submodel_inline_table_completed_by_a_lower_layer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # An inline-table value for a submodel key that a LOWER layer completes must be
    # accepted: the isolated pre-check sees only the partial table (missing a required
    # child), so it must not attribute that descendant error to the written key.
    from agent6.paths import global_config_path
    from agent6.ui.cli import main

    gpath = global_config_path()
    gpath.parent.mkdir(parents=True, exist_ok=True)
    gpath.write_text('[models.worker]\nmodel = "m"\n', encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    assert main(["config", "set", "--repo", "models.worker", '{ provider = "p" }']) == 0


# --- `config fix`: drop invalid entries, print what was dropped and where -------


def test_config_fix_drops_a_bad_value_and_keeps_valid_ones(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # prompt.decompose = true is invalid (now Literal["auto","on","off"]); the valid
    # budget entry beside it must survive the repair.
    from agent6.paths import global_config_path
    from agent6.ui.cli import main

    gpath = global_config_path()
    gpath.parent.mkdir(parents=True, exist_ok=True)
    gpath.write_text("[prompt]\ndecompose = true\n[budget]\nmax_usd = 5.0\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    rc = main(["config", "fix"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "prompt.decompose" in out and "global" in out  # named the entry + its layer
    text = gpath.read_text(encoding="utf-8")
    assert "decompose" not in text  # the invalid entry is gone
    assert "max_usd" in text  # the valid one stays
    assert main(["config", "show"]) == 0  # config is valid now


def test_config_fix_drops_an_unknown_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from agent6.paths import global_config_path
    from agent6.ui.cli import main

    gpath = global_config_path()
    gpath.parent.mkdir(parents=True, exist_ok=True)
    gpath.write_text("[sandbox]\nprotct_git = true\n", encoding="utf-8")  # typo of protect_git
    monkeypatch.chdir(tmp_path)

    rc = main(["config", "fix"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "sandbox.protct_git" in out
    assert "protct_git" not in gpath.read_text(encoding="utf-8")


def test_config_fix_labels_a_repo_layer_entry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from agent6.config.layer import repo_config_path_for
    from agent6.ui.cli import main

    monkeypatch.chdir(tmp_path)
    rpath = repo_config_path_for(tmp_path)
    rpath.parent.mkdir(parents=True, exist_ok=True)
    rpath.write_text("[prompt]\ndecompose = true\n", encoding="utf-8")

    rc = main(["config", "fix"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "prompt.decompose" in out and "repo" in out
    assert "decompose" not in rpath.read_text(encoding="utf-8")


def test_config_fix_on_valid_config_reports_nothing_to_fix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from agent6.paths import global_config_path
    from agent6.ui.cli import main

    gpath = global_config_path()
    gpath.parent.mkdir(parents=True, exist_ok=True)
    before = "[budget]\nmax_usd = 5.0\n"
    gpath.write_text(before, encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    rc = main(["config", "fix"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "nothing to fix" in out.lower()
    assert gpath.read_text(encoding="utf-8") == before  # untouched


def test_config_fix_repairs_both_layers_and_labels_each(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from agent6.config.layer import repo_config_path_for
    from agent6.paths import global_config_path
    from agent6.ui.cli import main

    gpath = global_config_path()
    gpath.parent.mkdir(parents=True, exist_ok=True)
    gpath.write_text("[prompt]\ndecompose = true\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    rpath = repo_config_path_for(tmp_path)
    rpath.parent.mkdir(parents=True, exist_ok=True)
    rpath.write_text('[sandbox]\nrun_commands = "bogus"\n', encoding="utf-8")

    rc = main(["config", "fix"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "prompt.decompose" in out and "global" in out
    assert "sandbox.run_commands" in out and "repo" in out
    assert "decompose" not in gpath.read_text(encoding="utf-8")
    assert "bogus" not in rpath.read_text(encoding="utf-8")


def test_config_fix_machine_overlay_leaves_the_spec_untouched(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from agent6.ui.cli import main

    monkeypatch.chdir(tmp_path)
    mfile = tmp_path / "m.asm.toml"
    mfile.write_text(_GOOD + "[config.prompt]\ndecompose = true\n", encoding="utf-8")

    rc = main(["config", "fix", "--machine-file", str(mfile)])
    out = capsys.readouterr().out
    assert rc == 0
    assert "prompt.decompose" in out
    text = mfile.read_text(encoding="utf-8")
    assert "decompose" not in text  # the invalid overlay entry is gone
    assert 'machine = "m"' in text  # the machine spec itself is untouched


def test_config_fix_reports_an_entry_it_cannot_auto_remove(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # A non-absolute agent6.state_dir is rejected before the model loads (it locates
    # the per-repo config dir), so fix cannot drop it as a plain leaf. It must SAY so
    # and exit non-zero, never silently report a still-broken config as fixed.
    from agent6.paths import global_config_path
    from agent6.ui.cli import main

    gpath = global_config_path()
    gpath.parent.mkdir(parents=True, exist_ok=True)
    gpath.write_text('[agent6]\nstate_dir = "not-absolute"\n', encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    rc = main(["config", "fix"])
    err = capsys.readouterr().err
    assert rc == 2
    assert "state_dir" in err


def test_config_set_unknown_provider_key_speaks_human(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # A provider entry is a discriminated union; the standalone minimal dict
    # cannot resolve a member, so every provider-key error used to fall through
    # to the merged-layer pydantic dump (with the member tag leaking into the
    # displayed loc). The member models answer directly now.
    from agent6.ui.cli import main

    monkeypatch.chdir(tmp_path)
    assert main(["config", "set", "providers.p.api_format", "anthropic"]) == 0
    capsys.readouterr()
    rc = main(["config", "set", "providers.p.bogus_key", "x"])
    assert rc == 2
    err = capsys.readouterr().err
    assert "unknown provider key 'providers.p.bogus_key'" in err
    assert "Extra inputs" not in err and "anthropic.bogus" not in err
    rc = main(["config", "set", "providers.p.api_key_enw", "MY_KEY"])  # typo
    assert rc == 2
    assert "'api_key_env'" in capsys.readouterr().err  # the did-you-mean


def test_config_set_invalid_provider_value_names_the_field(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from agent6.ui.cli import main

    monkeypatch.chdir(tmp_path)
    assert main(["config", "set", "providers.p.api_format", "anthropic"]) == 0
    capsys.readouterr()
    rc = main(["config", "set", "providers.p.http_timeout_s", "not-a-number"])
    assert rc == 2
    err = capsys.readouterr().err
    assert "providers.p.http_timeout_s" in err
    assert "merged config layers" not in err
    # A Field-constraint violation (gt=0) gets the same member answer, not the
    # merged dump with the discriminator tag leaking into the loc.
    rc = main(["config", "set", "providers.p.http_timeout_s", "-5"])
    assert rc == 2
    err = capsys.readouterr().err
    assert "greater than 0" in err
    assert "merged config layers" not in err and ".anthropic." not in err
    # A partial-entry write some member accepts still lands (never reverted).
    assert main(["config", "set", "providers.p.base_url", "https://x.example/v1"]) == 0


def test_config_fix_skips_an_entry_another_writer_already_fixed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """find_invalid_entries reads unlocked and removal deletes by key NAME, so a
    `config set` that replaced the offending key with a VALID value in between
    had it deleted -- after that writer was told it had been saved."""
    from agent6.config.layer import ConfigDiagnosis, InvalidEntry

    cfg = tmp_path / "config.toml"
    cfg.write_text('[sandbox]\nrun_commands = "ask"\n', encoding="utf-8")

    # Diagnosis saw the OLD, invalid value; the file already holds the fixed one.
    stale = InvalidEntry(
        leaf="sandbox.run_commands",
        value="maybe",  # what the (unlocked) diagnosis read
        layer="global",
        path=cfg,
        file_key="sandbox.run_commands",
    )
    calls = {"n": 0}

    def _diag(*_a: object, **_k: object) -> ConfigDiagnosis:
        calls["n"] += 1
        return ConfigDiagnosis(removable=(stale,) if calls["n"] == 1 else (), blocked=None)

    monkeypatch.setattr(cc, "find_invalid_entries", _diag)
    cc._cmd_config_fix(machine=None)  # pyright: ignore[reportPrivateUsage]

    assert 'run_commands = "ask"' in cfg.read_text(encoding="utf-8"), (
        "the concurrent writer's value was deleted by a stale diagnosis"
    )


def test_config_fix_removes_a_nan_entry_instead_of_claiming_valid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """TOML `nan` never compares equal to itself, so the under-lock staleness
    re-check read a still-present nan as "replaced by a concurrent writer",
    skipped it, and the no-progress break printed "Config is valid; nothing
    to fix." rc=0 over a config every next command still refuses."""
    from agent6.paths import global_config_path
    from agent6.ui.cli import main

    gpath = global_config_path()
    gpath.parent.mkdir(parents=True, exist_ok=True)
    gpath.write_text("[sandbox]\nbogus_entry = nan\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    rc = main(["config", "fix"])
    out = capsys.readouterr().out
    assert "nothing to fix" not in out  # the config was NOT valid
    assert rc == 0
    assert "bogus_entry" in out  # removed and named
    assert "bogus_entry" not in gpath.read_text(encoding="utf-8")
    assert main(["config", "show"]) == 0


def test_equal_tolerating_nan_matches_nan_at_every_depth() -> None:
    from math import nan

    assert cc._equal_tolerating_nan(nan, nan)  # pyright: ignore[reportPrivateUsage]
    assert cc._equal_tolerating_nan({"x": nan}, {"x": nan})  # pyright: ignore[reportPrivateUsage]
    assert cc._equal_tolerating_nan([1.0, nan], [1.0, nan])  # pyright: ignore[reportPrivateUsage]
    assert not cc._equal_tolerating_nan({"x": nan}, {"x": 1.0})  # pyright: ignore[reportPrivateUsage]
    assert not cc._equal_tolerating_nan([nan], [nan, nan])  # pyright: ignore[reportPrivateUsage]


def test_revalidate_machine_no_lock_keeps_the_write_and_says_so(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without the config lock a whole-file restore could clobber a concurrent
    writer, so the machine path (like the layer writers) KEEPS the invalid write
    and says the lock was missing, rather than rolling back over a snapshot that
    predates the other writer's update."""
    monkeypatch.setattr(cc, "load_effective_with_overlay", _noop_overlay)
    target = tmp_path / "m.asm.toml"
    target.write_text(_BAD, encoding="utf-8")

    err = cc._revalidate_machine(target, _GOOD, held=False)  # pyright: ignore[reportPrivateUsage]

    assert err is not None and "kept as written" in err
    assert target.read_text(encoding="utf-8") == _BAD  # NOT rolled back


def test_config_set_names_the_flag_file_that_shadows_the_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """`agent6 --config F config set` writes the global config while F keeps
    overriding the key for every invocation that carries the flag; without the
    note the successful edit reads as ineffective."""
    from agent6.ui.cli import main

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("AGENT6_CONFIG_HOME", str(tmp_path / "cfg"))
    monkeypatch.setenv("AGENT6_STATE_HOME", str(tmp_path / "state"))
    flag = tmp_path / "f.toml"
    flag.write_text('[sandbox]\nrun_commands = "yes"\n', encoding="utf-8")
    rc = main(["--config", str(flag), "config", "set", "sandbox.run_commands", "no"])
    out = capsys.readouterr().out
    assert rc == 0
    assert f"--config {flag} overrides sandbox.run_commands" in out
    # A key the flag file does not set carries no note.
    rc = main(["--config", str(flag), "config", "set", "sandbox.protect_git", "true"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "overrides sandbox.protect_git" not in out


def test_config_show_legend_names_the_flag_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The source column says "flag"; the legend must read back to the one
    path the operator typed."""
    from agent6.ui.cli import main

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("AGENT6_CONFIG_HOME", str(tmp_path / "cfg"))
    monkeypatch.setenv("AGENT6_STATE_HOME", str(tmp_path / "state"))
    flag = tmp_path / "f.toml"
    flag.write_text('[sandbox]\nrun_commands = "yes"\n', encoding="utf-8")
    rc = main(["--config", str(flag), "config", "show"])
    out = capsys.readouterr().out
    assert rc == 0
    assert f"flag = {flag}" in out


def test_a_machine_overlay_refusal_reads_like_every_other_writer() -> None:
    """`config set --machine-file` prints the leaf lines the global and repo
    writers print: no validation header, no error type."""
    from agent6.ui.cli.config_cmds import (
        _leaf_problems,  # pyright: ignore[reportPrivateUsage]
    )

    raw = (
        "Config validation failed: (merged config layers + machine overlay)\n"
        "  - workflow.verify_retries: Input should be greater than or equal to 0"
        " (type=greater_than_equal)"
    )
    assert _leaf_problems(raw) == (
        "workflow.verify_retries: Input should be greater than or equal to 0"
    )
    assert _leaf_problems("machine file unreadable") == "machine file unreadable"


def test_config_unset_on_an_mcp_server_names_the_verb_that_removes_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """`[mcp.servers.<name>]` is valid only whole, so unsetting a key of it
    leaves a config that does not load; the refusal names `mcp remove`."""
    monkeypatch.chdir(tmp_path)
    cfg = tmp_path / "config.toml"
    cfg.write_text(
        '[agent6]\nconfig_version = 1\n\n[mcp.servers.calc]\ncommand = ["true"]\n',
        encoding="utf-8",
    )

    rc = cc._cmd_config_unset(  # pyright: ignore[reportPrivateUsage]
        "mcp.servers.calc", repo=False, machine=None, config_path=cfg
    )

    assert rc == 2
    assert "agent6 mcp remove calc" in capsys.readouterr().err


def test_config_unset_on_an_unknown_key_keeps_the_generic_pointer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    cfg = tmp_path / "config.toml"
    cfg.write_text("[agent6]\nconfig_version = 1\n", encoding="utf-8")

    rc = cc._cmd_config_unset(  # pyright: ignore[reportPrivateUsage]
        "mcp.servers.nope", repo=False, machine=None, config_path=cfg
    )

    assert rc == 2
    assert "is not a config leaf" in capsys.readouterr().err
