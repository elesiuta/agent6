# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Tests for `agent6 connect` / `agent6 model` / `agent6 config` CLI flows."""

from __future__ import annotations

import stat
from collections.abc import Callable
from pathlib import Path

import pytest

from agent6 import secrets
from agent6.config.layer import resolved_state_dir
from agent6.models.cache import KeyProbeResult
from agent6.ui.cli import main


@pytest.fixture
def iso(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    monkeypatch.setenv("AGENT6_CONFIG_HOME", str(tmp_path / "g"))
    monkeypatch.chdir(tmp_path)
    # Default to an interactive terminal: the getpass-path tests below assert
    # the masked-input behaviour, which `connect` only takes when stdin is a
    # TTY. Under pytest stdin reports non-TTY; the non-TTY path has its own test.
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    # Keep connect hermetic by default: stub the post-save key probe so no test
    # makes a real network call. Tests of the probe behaviour re-patch it.
    monkeypatch.setattr(
        "agent6.ui.cli.connect.probe_provider_key",
        lambda *a, **k: KeyProbeResult(  # type: ignore[misc]
            ok=True, status="ok", detail="provider returned 1 models"
        ),
    )
    return tmp_path / "g"


def test_connect_stores_key_and_provider_and_never_execs(
    iso: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr("agent6.ui.cli.connect.getpass.getpass", lambda prompt="": "sk-ant-FAKE")
    # Security: connect must NEVER run a subprocess (no remote-supplied command).
    calls: list[object] = []

    def _record_run(*args: object, **kwargs: object) -> None:
        calls.append(args)

    monkeypatch.setattr("subprocess.run", _record_run)

    rc = main(["connect", "anthropic"])
    assert rc == 0

    sp = tmp_path / "g" / "secrets.toml"
    assert sp.is_file()
    assert stat.S_IMODE(sp.stat().st_mode) == 0o600
    assert secrets.resolve_api_key("anthropic", None) == "sk-ant-FAKE"

    gc = (tmp_path / "g" / "config.toml").read_text(encoding="utf-8")
    assert "[providers.anthropic]" in gc
    assert calls == []


def test_connect_preserves_hand_edited_provider_keys(
    iso: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """connect is the documented add/UPDATE path; re-running it for a key
    rotation must not erase the operator's hand-added sibling keys (it replaced
    the whole [providers.<name>] block)."""
    gc = tmp_path / "g" / "config.toml"
    gc.parent.mkdir(parents=True, exist_ok=True)
    gc.write_text(
        "[providers.openrouter]\n"
        'api_format = "openai"\n'
        'base_url = "https://openrouter.ai/api/v1"\n'
        "# hand-added\n"
        "http_timeout_s = 120\n"
        'extra_body = { provider = { sort = "throughput" } }\n',
        encoding="utf-8",
    )
    monkeypatch.setattr("agent6.ui.cli.connect.getpass.getpass", lambda prompt="": "sk-or-FAKE")
    monkeypatch.setattr("builtins.input", lambda prompt="": "")  # accept the preset base_url
    rc = main(["connect", "openrouter"])
    assert rc == 0
    text = gc.read_text(encoding="utf-8")
    assert "[providers.openrouter]" in text
    assert 'api_format = "openai"' in text
    assert "http_timeout_s = 120" in text
    assert "extra_body" in text
    assert "# hand-added" in text


def test_connect_validates_key_and_reports_ok(
    iso: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr("agent6.ui.cli.connect.getpass.getpass", lambda prompt="": "sk-ant-REAL")
    rc = main(["connect", "anthropic"])  # iso stubs the probe -> ok
    assert rc == 0
    out = capsys.readouterr().out
    assert "Checking the key against the provider" in out
    assert "Key validated" in out


def test_connect_warns_when_provider_rejects_key(
    iso: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr("agent6.ui.cli.connect.getpass.getpass", lambda prompt="": "sk-ant-BAD")
    monkeypatch.setattr(
        "agent6.ui.cli.connect.probe_provider_key",
        lambda *a, **k: KeyProbeResult(ok=False, status="auth_failed", detail="HTTP 401"),  # type: ignore[misc]
    )
    rc = main(["connect", "anthropic"])
    assert rc == 0  # the key is saved anyway; the warning is advisory
    err = capsys.readouterr().err
    assert "REJECTED this key" in err
    assert "HTTP 401" in err
    # The key was still written (the user may fix it later).
    assert secrets.resolve_api_key("anthropic", None) == "sk-ant-BAD"


def test_connect_no_verify_skips_the_probe(
    iso: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr("agent6.ui.cli.connect.getpass.getpass", lambda prompt="": "sk-ant-REAL")

    def _boom(*_a: object, **_k: object) -> KeyProbeResult:
        raise AssertionError("--no-verify must not probe the provider")

    monkeypatch.setattr("agent6.ui.cli.connect.probe_provider_key", _boom)
    rc = main(["connect", "anthropic", "--no-verify"])
    assert rc == 0
    assert "Checking the key" not in capsys.readouterr().out


def test_connect_non_tty_reads_plain_input_without_getpass(
    iso: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Scripted/piped connect (no controlling terminal): getpass would print a
    # GetPassWarning + "input may be echoed" line. The non-TTY path reads a
    # plain line via input() instead, never touching getpass.
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)

    def _boom(*_a: object, **_k: object) -> str:
        raise AssertionError("getpass must not be called on the non-TTY path")

    monkeypatch.setattr("agent6.ui.cli.connect.getpass.getpass", _boom)
    monkeypatch.setattr("builtins.input", lambda prompt="": "sk-ant-PIPED")

    rc = main(["connect", "anthropic"])
    assert rc == 0
    assert secrets.resolve_api_key("anthropic", None) == "sk-ant-PIPED"


def test_connect_rejects_non_bare_key_provider_name(
    iso: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # A name with a space would corrupt `[providers.<name>]` in the TOML; reject
    # it before writing anything (connect doesn't re-validate the file).
    rc = main(["connect", "my provider"])
    assert rc == 2
    err = capsys.readouterr().err
    assert "not a valid TOML bare key" in err
    # Nothing was written.
    assert not (tmp_path / "g" / "config.toml").exists()
    assert not (tmp_path / "g" / "secrets.toml").exists()


def test_connect_prints_post_entry_key_summary(
    iso: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # Simulate Python < 3.14 getpass (no echo_char): the helper must print a
    # length + last-four summary so the operator can tell the paste landed.
    def _fake_getpass(prompt: str = "", **kwargs: object) -> str:
        if "echo_char" in kwargs:
            raise TypeError("echo_char unsupported")
        return "sk-ant-0123456789wxyz"

    monkeypatch.setattr("agent6.ui.cli.connect.getpass.getpass", _fake_getpass)
    rc = main(["connect", "anthropic"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "Captured key: 21 chars, ending …wxyz" in out
    # The key itself is never echoed in full.
    assert "sk-ant-0123456789wxyz" not in out


def test_connect_short_key_summary_omits_tail(
    iso: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def _fake_getpass(prompt: str = "", **kwargs: object) -> str:
        if "echo_char" in kwargs:
            raise TypeError("echo_char unsupported")
        return "short"

    monkeypatch.setattr("agent6.ui.cli.connect.getpass.getpass", _fake_getpass)
    rc = main(["connect", "anthropic"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "Captured key: 5 chars." in out
    assert "ending" not in out


def test_connect_masked_echo_skips_summary(
    iso: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # Simulate Python 3.14+ getpass that accepts echo_char: no post-entry
    # summary is printed because the keystrokes were already masked live.
    def _fake_getpass(prompt: str = "", **kwargs: object) -> str:
        return "sk-ant-0123456789wxyz"

    monkeypatch.setattr("agent6.ui.cli.connect.getpass.getpass", _fake_getpass)
    rc = main(["connect", "anthropic"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "Captured key:" not in out


def test_connect_local_endpoint_no_key(
    iso: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr("agent6.ui.cli.connect.getpass.getpass", lambda prompt="": "")
    monkeypatch.setattr("builtins.input", lambda prompt="": "")  # accept default base_url
    rc = main(["connect", "ollama"])
    assert rc == 0
    gc = (tmp_path / "g" / "config.toml").read_text(encoding="utf-8")
    assert "[providers.ollama]" in gc
    # No key entered -> no secrets file required.
    assert not (tmp_path / "g" / "secrets.toml").is_file()


def test_model_set_and_show(iso: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    rc = main(["model", "worker", "anthropic", "claude-x", "--effort", "medium"])
    assert rc == 0
    gc = (tmp_path / "g" / "config.toml").read_text(encoding="utf-8")
    assert "[models.worker]" in gc
    assert "claude-x" in gc
    assert "medium" in gc

    capsys.readouterr()
    rc = main(["model"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "worker" in out
    assert "claude-x" in out


def test_model_all_sets_every_role(
    iso: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # "all" is a pseudo-role: one command writes planner/worker/reviewer alike.
    rc = main(["model", "all", "anthropic", "claude-x"])
    assert rc == 0
    gc = (tmp_path / "g" / "config.toml").read_text(encoding="utf-8")
    assert "[models.planner]" in gc
    assert "[models.worker]" in gc
    assert "[models.reviewer]" in gc
    assert gc.count("claude-x") == 3
    assert "all roles" in capsys.readouterr().out


def test_model_invalid_provider_refuses_and_rolls_back(
    iso: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Regression: setting a role to an unconfigured provider makes the merged
    # config invalid. The set must REFUSE (rc 2) and leave config.toml byte-for-
    # byte as it was, not write the broken value -- which previously bricked every
    # later command. (The provider cross-check is active only once a provider is
    # configured, so connect one first.)
    monkeypatch.setattr("agent6.ui.cli.connect.getpass.getpass", lambda prompt="": "sk-ant-FAKE")
    assert main(["connect", "anthropic"]) == 0
    assert main(["model", "worker", "anthropic", "good-x"]) == 0
    cfg = tmp_path / "g" / "config.toml"
    before = cfg.read_text(encoding="utf-8")
    assert main(["model", "worker", "missing-prov", "gpt"]) == 2
    after = cfg.read_text(encoding="utf-8")
    assert after == before  # rolled back exactly
    assert "missing-prov" not in after
    assert main(["model"]) == 0  # config still loads (not bricked)


def test_model_rejects_unknown_role(iso: Path) -> None:
    # argparse `choices` validates the role positional (and feeds argcomplete).
    with pytest.raises(SystemExit) as exc:
        main(["model", "bogus", "anthropic", "claude-x"])
    assert exc.value.code == 2


def _models_stub(models: list[str]) -> Callable[..., list[str]]:
    """A typed list_models stand-in (strict pyright rejects bare lambdas here)."""

    def _list(*_a: object, **_k: object) -> list[str]:
        return models

    return _list


def _key_stub(key: str | None) -> Callable[..., str | None]:
    """A typed resolve_api_key stand-in returning a fixed key (or None)."""

    def _resolve(*_a: object, **_k: object) -> str | None:
        return key

    return _resolve


def test_model_piped_without_model_lists_the_catalog(
    iso: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # No tty and no model named: the invocation is a listing (one id per line,
    # exit 0), not a 344-line prompt dump that ends in an EOF error. The set
    # hint goes to stderr so stdout stays pipe-clean.
    (tmp_path / "g").mkdir(parents=True, exist_ok=True)
    (tmp_path / "g" / "config.toml").write_text(
        '[providers.anthropic]\napi_format = "anthropic"\n', encoding="utf-8"
    )
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    monkeypatch.setattr("agent6.models.choices.list_models", _models_stub(["claude-a", "claude-b"]))
    rc = main(["model", "worker", "anthropic"])
    assert rc == 0
    captured = capsys.readouterr()
    assert captured.out == "claude-a\nclaude-b\n"
    assert "set one with: agent6 model worker anthropic <model>" in captured.err
    # Nothing was written: the listing never touches config.
    assert "[models.worker]" not in (tmp_path / "g" / "config.toml").read_text(encoding="utf-8")


def test_model_set_warns_when_the_provider_has_no_key(
    iso: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # Setting a role to a configured-but-keyless provider succeeds (config is
    # just config) but the first run would refuse; the note closes that loop at
    # set time. README's own quickstart line hits this on a keyless machine.
    (tmp_path / "g").mkdir(parents=True, exist_ok=True)
    (tmp_path / "g" / "config.toml").write_text(
        '[providers.anthropic]\napi_format = "anthropic"\n', encoding="utf-8"
    )
    monkeypatch.setattr("agent6.ui.cli.model.resolve_api_key", _key_stub(None))
    rc = main(["model", "worker", "anthropic", "claude-x"])
    assert rc == 0
    err = capsys.readouterr().err
    assert "provider 'anthropic' has no stored API key" in err
    assert "agent6 connect" in err


def test_model_set_stays_quiet_when_the_key_resolves(
    iso: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (tmp_path / "g").mkdir(parents=True, exist_ok=True)
    (tmp_path / "g" / "config.toml").write_text(
        '[providers.anthropic]\napi_format = "anthropic"\n', encoding="utf-8"
    )
    monkeypatch.setattr("agent6.ui.cli.model.resolve_api_key", _key_stub("sk-x"))
    rc = main(["model", "worker", "anthropic", "claude-x"])
    assert rc == 0
    assert "note:" not in capsys.readouterr().err


def test_model_piped_unknown_provider_errors(
    iso: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    monkeypatch.setattr("agent6.models.choices.list_models", _models_stub([]))
    rc = main(["model", "worker", "nosuch"])
    assert rc == 2
    assert "no known models for nosuch" in capsys.readouterr().err


def test_model_stdout_piped_lists_even_with_a_tty_stdin(
    iso: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # `agent6 model worker anthropic | grep x` keeps stdin a tty; the listing
    # must trigger on the piped stdout, not park the pipe on an invisible
    # numbered prompt. (iso leaves stdin.isatty True; captured stdout is not
    # a tty, exactly the pipe shape.)
    (tmp_path / "g").mkdir(parents=True, exist_ok=True)
    (tmp_path / "g" / "config.toml").write_text(
        '[providers.anthropic]\napi_format = "anthropic"\n', encoding="utf-8"
    )
    monkeypatch.setattr("agent6.models.choices.list_models", _models_stub(["claude-a"]))
    rc = main(["model", "worker", "anthropic"])
    assert rc == 0
    assert capsys.readouterr().out == "claude-a\n"


def test_model_piped_without_provider_errors_without_prompt_dump(
    iso: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # The provider prompt is interactive-only; piped it dumped "Connected
    # providers: ..." plus a prompt, then died on EOF.
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    rc = main(["model", "worker"])
    assert rc == 2
    captured = capsys.readouterr()
    assert "no provider given" in captured.err
    assert "Connected providers" not in captured.out


def test_model_piped_listing_notes_an_ignored_thinking_flag(
    iso: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (tmp_path / "g").mkdir(parents=True, exist_ok=True)
    (tmp_path / "g" / "config.toml").write_text(
        '[providers.anthropic]\napi_format = "anthropic"\n', encoding="utf-8"
    )
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    monkeypatch.setattr("agent6.models.choices.list_models", _models_stub(["claude-a"]))
    rc = main(["model", "worker", "anthropic", "--effort", "high"])
    assert rc == 0
    assert "--effort ignored" in capsys.readouterr().err


def test_model_aborts_without_provider(iso: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Role given but provider omitted and none connected: the prompt gets an
    # empty answer and the command refuses rather than writing a bad config.
    monkeypatch.setattr("builtins.input", lambda prompt="": "")
    assert main(["model", "worker"]) == 2


def test_model_interactive_prefill(
    iso: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # A provider is connected; the model list is served live (mocked). The
    # operator picks the provider by default and the model by number.
    (tmp_path / "g").mkdir(parents=True, exist_ok=True)
    (tmp_path / "g" / "config.toml").write_text(
        '[providers.anthropic]\napi_format = "anthropic"\n', encoding="utf-8"
    )

    def fake_list_models(*a: object, **k: object) -> list[str]:
        return ["claude-a", "claude-b"]

    monkeypatch.setattr("agent6.models.choices.list_models", fake_list_models)
    monkeypatch.setattr("sys.stdout.isatty", lambda: True)  # interactive = both ttys
    answers = iter(["", "2"])  # provider default, then model #2
    monkeypatch.setattr("builtins.input", lambda prompt="": next(answers))
    rc = main(["model", "worker"])
    assert rc == 0
    gc = (tmp_path / "g" / "config.toml").read_text(encoding="utf-8")
    assert "[models.worker]" in gc
    assert "claude-b" in gc


def test_model_all_interactive_prompts_once(
    iso: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # "all" prompts ONCE for provider/model (not per role), then applies to each.
    (tmp_path / "g").mkdir(parents=True, exist_ok=True)
    (tmp_path / "g" / "config.toml").write_text(
        '[providers.anthropic]\napi_format = "anthropic"\n', encoding="utf-8"
    )

    def _models(*_a: object, **_k: object) -> list[str]:
        return ["claude-a", "claude-b"]

    monkeypatch.setattr("agent6.models.choices.list_models", _models)
    monkeypatch.setattr("sys.stdout.isatty", lambda: True)  # interactive = both ttys
    calls = {"n": 0}
    answers = iter(["", "2"])  # provider default, model #2 — once, not 3x

    def _input(prompt: str = "") -> str:
        calls["n"] += 1
        return next(answers)

    monkeypatch.setattr("builtins.input", _input)
    rc = main(["model", "all"])
    assert rc == 0
    assert calls["n"] == 2  # one provider prompt + one model prompt, total
    gc = (tmp_path / "g" / "config.toml").read_text(encoding="utf-8")
    for role in ("planner", "worker", "reviewer"):
        assert f"[models.{role}]" in gc
    assert gc.count("claude-b") == 3


def test_model_repo_scope_writes_repo(iso: Path, tmp_path: Path) -> None:
    rc = main(["model", "reviewer", "anthropic", "claude-o", "--repo"])
    assert rc == 0
    repo_cfg = (resolved_state_dir(tmp_path) / "config.toml").read_text(encoding="utf-8")
    assert "[models.reviewer]" in repo_cfg


def test_config_fill_writes_global(iso: Path, tmp_path: Path) -> None:
    rc = main(["config", "fill"])
    assert rc == 0
    gc = (tmp_path / "g" / "config.toml").read_text(encoding="utf-8")
    assert "[sandbox]" in gc
    assert "[budget]" in gc


def test_config_show_runs(iso: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["config", "show"]) == 0
    out = capsys.readouterr().out
    assert "[sandbox]" in out
    assert "source:" in out


def test_config_path_runs(iso: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["config", "path"]) == 0
    out = capsys.readouterr().out
    assert "global config" in out
    assert "secrets" in out


def _grant_jwt() -> str:
    import base64
    import json as _json

    claims = {
        "https://api.openai.com/auth": {
            "chatgpt_account_id": "acct-7",
            "chatgpt_plan_type": "plus",
        }
    }
    payload = base64.urlsafe_b64encode(_json.dumps(claims).encode()).rstrip(b"=").decode()
    return f"h.{payload}.s"


class _TokenResp:
    status_code = 200
    text = ""

    @staticmethod
    def json() -> dict[str, object]:
        return {
            "access_token": _grant_jwt(),
            "refresh_token": "RT1",
            "expires_in": 3600,
            "id_token": "",
        }


def test_connect_chatgpt_paste_flow_signs_in_and_writes_config(
    iso: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Headless connect: paste the callback URL, exchange the code, store
    tokens 0600, write the provider block, print the training-data notice.
    Never executes a subprocess and never opens a browser."""
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    monkeypatch.setattr("agent6.ui.cli.connect.pysecrets.token_urlsafe", lambda n=24: "STATE1")
    exchanges: list[dict[str, str]] = []

    def fake_post(url: str, data: dict[str, str], timeout_s: float) -> _TokenResp:
        exchanges.append(data)
        return _TokenResp()

    monkeypatch.setattr("agent6.providers.chatgpt_oauth._post_form", fake_post)
    monkeypatch.setattr(
        "builtins.input",
        lambda prompt="": "http://localhost:1455/auth/callback?code=C1&state=STATE1",
    )
    opened: list[str] = []
    monkeypatch.setattr("webbrowser.open", opened.append)
    execs: list[object] = []

    def record_run(*args: object, **kwargs: object) -> None:
        execs.append(args)

    monkeypatch.setattr("subprocess.run", record_run)

    rc = main(["connect", "chatgpt"])
    assert rc == 0
    assert opened == [] and execs == []
    assert exchanges[0]["grant_type"] == "authorization_code"
    assert exchanges[0]["code"] == "C1"

    sp = tmp_path / "g" / "secrets.toml"
    assert stat.S_IMODE(sp.stat().st_mode) == 0o600
    tokens = secrets.load_oauth_tokens("chatgpt")
    assert tokens is not None and tokens.account_id == "acct-7"

    gc = (tmp_path / "g" / "config.toml").read_text(encoding="utf-8")
    assert "[providers.chatgpt]" in gc and 'api_format = "chatgpt"' in gc
    out = capsys.readouterr().out
    assert "Signed in (plus plan)" in out
    assert "Data controls" in out and "never sends feedback" in out


def test_connect_claude_writes_the_format_only_and_stores_no_secret(
    iso: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """`connect claude` checks the binary's sign-in, writes `api_format` and no
    secret; a signed-out binary warns with the remedy; `--logout` refuses since
    agent6 holds no Claude Code credential."""

    def signed_in(binary: str) -> str | None:
        return None

    def signed_out(binary: str) -> str | None:
        return "not signed in; run `claude auth login`"

    monkeypatch.setattr("agent6.ui.cli.connect.login_status", signed_in)
    assert main(["connect", "claude"]) == 0
    gc = (tmp_path / "g" / "config.toml").read_text(encoding="utf-8")
    assert "[providers.claude]" in gc and 'api_format = "claude_code"' in gc
    assert not (tmp_path / "g" / "secrets.toml").exists()
    assert "Claude Code (`claude` on PATH): signed in." in capsys.readouterr().out

    monkeypatch.setattr("agent6.ui.cli.connect.login_status", signed_out)
    assert main(["connect", "claude"]) == 0
    err = capsys.readouterr().err
    assert "WARNING: not signed in; run `claude auth login`" in err and "not usable yet" in err

    assert main(["connect", "claude", "--logout"]) == 2
    assert "claude auth logout" in capsys.readouterr().err


def test_connect_chatgpt_state_mismatch_refuses(iso: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    monkeypatch.setattr("agent6.ui.cli.connect.pysecrets.token_urlsafe", lambda n=24: "STATE1")
    monkeypatch.setattr(
        "builtins.input",
        lambda prompt="": "http://localhost:1455/auth/callback?code=C1&state=EVIL",
    )
    rc = main(["connect", "chatgpt"])
    assert rc == 2
    assert secrets.load_oauth_tokens("chatgpt") is None


def test_oauth_callback_server_round_trip() -> None:
    """The localhost receiver answers the redirect, hands over the code, and
    404s every other path; a state-mismatch hit is a 400, not a capture."""
    import urllib.error
    import urllib.request

    from agent6.ui.cli.connect import _CallbackServer  # pyright: ignore[reportPrivateUsage]

    srv = _CallbackServer("S1", port=0)
    try:
        base = f"http://127.0.0.1:{srv.port}"
        with pytest.raises(urllib.error.HTTPError) as e404:
            urllib.request.urlopen(f"{base}/favicon.ico", timeout=5)
        assert e404.value.code == 404
        with pytest.raises(urllib.error.HTTPError) as e400:
            urllib.request.urlopen(f"{base}/auth/callback?code=X&state=EVIL", timeout=5)
        assert e400.value.code == 400
        assert srv.wait(timeout_s=0.3) is None
        with urllib.request.urlopen(f"{base}/auth/callback?code=OK&state=S1", timeout=5) as resp:
            assert resp.status == 200
        assert srv.wait(timeout_s=5.0) == "OK"
    finally:
        srv.close()


def test_connect_logout_revokes_and_removes_tokens(
    iso: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """--logout revokes a ChatGPT grant at the issuer (best effort) and
    removes the provider's secrets entry; a repeat run reports nothing left."""
    import time as _time

    secrets.save_oauth_tokens(
        "chatgpt", secrets.OAuthTokens("AT", "RT", _time.time() + 3600, "acct")
    )
    revoked: list[dict[str, object]] = []

    class _RevokeResp:
        status_code = 200
        text = ""

    def fake_post(url: str, *, json: dict[str, object], timeout: float) -> _RevokeResp:
        revoked.append({"url": url, **json})
        return _RevokeResp()

    monkeypatch.setattr("agent6.providers.chatgpt_oauth.httpx2.post", fake_post)
    rc = main(["connect", "chatgpt", "--logout"])
    assert rc == 0
    assert revoked[0]["url"] == "https://auth.openai.com/oauth/revoke"
    assert revoked[0]["token"] == "RT" and revoked[0]["token_type_hint"] == "refresh_token"
    assert secrets.load_oauth_tokens("chatgpt") is None
    assert "Removed stored credentials" in capsys.readouterr().out

    rc = main(["connect", "chatgpt", "--logout"])
    assert rc == 0
    assert "No stored credentials" in capsys.readouterr().out


def test_connect_chatgpt_headless_terminal_uses_the_device_flow(
    iso: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A tty without a display signs in by code entry: no browser, no
    localhost server, no paste prompt; the polled grant is saved."""
    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    monkeypatch.setattr("sys.platform", "linux")

    from agent6.providers.chatgpt_oauth import DeviceAuth, TokenGrant

    def fake_start(issuer: str, client_id: str) -> DeviceAuth:
        return DeviceAuth(device_auth_id="da_1", user_code="AB-12", interval_s=5.0)

    def fake_poll(issuer: str, client_id: str, device: DeviceAuth) -> TokenGrant:
        return TokenGrant(_grant_jwt(), "RT9", 3600.0)

    monkeypatch.setattr("agent6.ui.cli.connect.start_device_auth", fake_start)
    monkeypatch.setattr("agent6.ui.cli.connect.poll_device_auth", fake_poll)

    def no_input(prompt: str = "") -> str:
        pytest.fail("device path must not prompt for a paste")

    monkeypatch.setattr("builtins.input", no_input)
    opened: list[str] = []
    monkeypatch.setattr("webbrowser.open", opened.append)

    rc = main(["connect", "chatgpt"])
    assert rc == 0 and opened == []
    tokens = secrets.load_oauth_tokens("chatgpt")
    assert tokens is not None and tokens.refresh_token == "RT9"
    out = capsys.readouterr().out
    assert "enter the code:  AB-12" in out and "/codex/device" in out


def test_connect_chatgpt_device_flow_disabled_falls_back_to_paste(
    iso: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    monkeypatch.setattr("sys.platform", "linux")

    def disabled(issuer: str, client_id: str) -> None:
        return None

    monkeypatch.setattr("agent6.ui.cli.connect.start_device_auth", disabled)
    monkeypatch.setattr("agent6.ui.cli.connect.pysecrets.token_urlsafe", lambda n=24: "STATE1")

    def fake_post(url: str, data: dict[str, str], timeout_s: float) -> _TokenResp:
        return _TokenResp()

    monkeypatch.setattr("agent6.providers.chatgpt_oauth._post_form", fake_post)
    monkeypatch.setattr(
        "builtins.input",
        lambda prompt="": "http://localhost:1455/auth/callback?code=C1&state=STATE1",
    )
    rc = main(["connect", "chatgpt"])
    assert rc == 0
    assert secrets.load_oauth_tokens("chatgpt") is not None
