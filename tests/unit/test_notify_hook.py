# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Tests for the post-run notify hook."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent6.app.finalize import fire_notify_hook
from agent6.app.machine import (
    build_machine_notify_hook,
)
from agent6.app.reporter import STDIO_REPORTER
from agent6.config import Config, NotifyConfig, load_config


def test_notify_noop_when_unconfigured(tmp_path: Path) -> None:
    """An empty `on_complete` tuple is a no-op (no subprocess, no error)."""
    notify = NotifyConfig()
    # Should return without raising, without doing anything.
    fire_notify_hook(
        notify,
        session_id="abcdef0123456789",
        session_dir=tmp_path,
        ok=True,
        reason="finish_session",
        verified="passed",
        reporter=STDIO_REPORTER,
    )


def test_notify_failure_does_not_raise(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """A failing argv (nonexistent binary) logs but does not raise."""
    notify = NotifyConfig(on_complete=("/nonexistent/agent6-notify-binary",), timeout_s=5.0)
    fire_notify_hook(
        notify,
        session_id="run-xyz",
        session_dir=tmp_path,
        ok=False,
        reason="budget_exhausted",
        verified="passed",
        reporter=STDIO_REPORTER,
    )
    captured = capsys.readouterr()
    assert "notify.on_complete failed" in captured.err


def test_both_hooks_run_the_same_way(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Two runners for one job drifted in both directions: the run hook
    swallowed a non-zero exit (a hook that fails silently stops notifying with
    nobody the wiser), and the machine hook let the hook's stdout into the
    parent's -- under `agent6 acp` that is the JSON-RPC stream."""
    argv = ("sh", "-c", "echo HOOK_STDOUT; exit 3")
    fire_notify_hook(
        NotifyConfig(on_complete=argv, timeout_s=5.0),
        session_id="run-xyz",
        session_dir=tmp_path,
        ok=True,
        reason="finish_session",
        verified="passed",
        reporter=STDIO_REPORTER,
    )
    cfg = Config.model_validate({"machine": {"notify": {"on_event": list(argv)}}})
    fire = build_machine_notify_hook(cfg, "machine-1", tmp_path)
    assert fire is not None
    fire("machine.end", "done", "", "info")

    captured = capsys.readouterr()
    assert "notify.on_complete exited 3" in captured.err
    assert "machine.notify hook exited 3" in captured.err
    assert "HOOK_STDOUT" not in captured.out


def test_notify_ok_zero_when_failed(tmp_path: Path) -> None:
    """ok=False sets AGENT6_SESSION_OK=0."""
    out = tmp_path / "ok.txt"
    argv = (
        "sh",
        "-c",
        f'printf "%s" "$AGENT6_SESSION_OK" > {out}',
    )
    notify = NotifyConfig(on_complete=argv, timeout_s=5.0)
    fire_notify_hook(
        notify,
        session_id="r",
        session_dir=tmp_path,
        ok=False,
        reason="provider_error",
        verified="passed",
        reporter=STDIO_REPORTER,
    )
    assert out.read_text(encoding="utf-8") == "0"


_MACHINE_CFG_BODY = """
[agent6]
config_version = 1

[providers.anthropic]
api_format = "anthropic"
api_key_env = "ANTHROPIC_API_KEY"

[models.worker]
provider = "anthropic"
model = "claude-sonnet-4-5"

[models.reviewer]
provider = "anthropic"
model = "claude-opus-4-5"

[workflow]
verify_command = ["true"]

[machine.notify]
on_event = ["python3", "-c", "PLACEHOLDER"]
timeout_s = 10.0
"""


def test_machine_notify_hook_fires_with_env(tmp_path: Path) -> None:
    out = tmp_path / "machine-notify.json"
    script = (
        "import json,os,sys; json.dump({"
        "'id': os.environ['AGENT6_MACHINE_ID'], "
        "'dir': os.environ['AGENT6_MACHINE_DIR'], "
        "'event': os.environ['AGENT6_MACHINE_EVENT'], "
        "'state': os.environ['AGENT6_MACHINE_STATE'], "
        "'message': os.environ['AGENT6_MACHINE_MESSAGE'], "
        "'level': os.environ['AGENT6_MACHINE_LEVEL']"
        "}, open(sys.argv[1], 'w'))"
    )
    body = _MACHINE_CFG_BODY.replace('"PLACEHOLDER"', f'"{script}", "{out}"')
    cfg_path = tmp_path / "agent6.toml"
    cfg_path.write_text(body, encoding="utf-8")
    cfg = load_config(cfg_path)
    hook = build_machine_notify_hook(cfg, "mymachine", tmp_path / "inst")
    assert hook is not None
    hook("notify", "poll", "attention needed", "warn")
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload == {
        "id": "mymachine",
        "dir": str(tmp_path / "inst"),
        "event": "notify",
        "state": "poll",
        "message": "attention needed",
        "level": "warn",
    }


def test_machine_notify_hook_nonzero_exit_is_reported(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """check=False keeps the hook non-fatal, but a nonzero exit was discarded
    entirely: notifications silently stopped arriving. The exit is named on
    stderr."""
    body = _MACHINE_CFG_BODY.replace('"PLACEHOLDER"', '"raise SystemExit(3)"')
    cfg_path = tmp_path / "agent6.toml"
    cfg_path.write_text(body, encoding="utf-8")
    hook = build_machine_notify_hook(load_config(cfg_path), "m", tmp_path / "inst")
    assert hook is not None
    hook("notify", "poll", "msg", "warn")
    assert "hook exited 3" in capsys.readouterr().err


def test_machine_notify_hook_none_when_unconfigured(tmp_path: Path) -> None:
    body = _MACHINE_CFG_BODY.replace(
        '\n[machine.notify]\non_event = ["python3", "-c", "PLACEHOLDER"]\ntimeout_s = 10.0\n', ""
    )
    cfg_path = tmp_path / "agent6.toml"
    cfg_path.write_text(body, encoding="utf-8")
    cfg = load_config(cfg_path)
    assert build_machine_notify_hook(cfg, "m", tmp_path) is None


def test_notify_in_config_loads(tmp_path: Path) -> None:
    """[notify] section round-trips through the config loader."""

    body = """
[agent6]
config_version = 1

[providers.anthropic]
api_format = "anthropic"
api_key_env = "ANTHROPIC_API_KEY"
prompt_caching = true

[models.worker]
provider = "anthropic"
model = "claude-sonnet-4-5"

[models.reviewer]
provider = "anthropic"
model = "claude-opus-4-5"

[sandbox]
isolation = "auto"
run_commands = "ask"
protect_git = true

[git]

[workflow]
verify_command = ["true"]

[budget]
max_tokens_fallback = 2000000

[notify]
on_complete = ["notify-send", "agent6 done"]
timeout_s = 12.5
"""
    cfg_path = tmp_path / "agent6.toml"
    cfg_path.write_text(body, encoding="utf-8")
    cfg = load_config(cfg_path)
    assert cfg.notify.on_complete == ("notify-send", "agent6 done")
    assert cfg.notify.timeout_s == 12.5


def test_notify_hook_env_carries_no_secrets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The hook child got the operator's WHOLE environment -- provider bearer
    tokens included via `[providers.*].api_key_env` -- where docs/security.md
    promises only the AGENT6_* set. The env is now the minimal hook_env base:
    a hook that logs or forwards its environment cannot carry a key with it."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-super-secret")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-secret")
    out = tmp_path / "env.json"
    argv = (
        "python3",
        "-c",
        "import json,os,sys; json.dump({"
        "'key': os.environ.get('ANTHROPIC_API_KEY'), "
        "'or_key': os.environ.get('OPENROUTER_API_KEY'), "
        "'path': bool(os.environ.get('PATH')), "
        "'home': bool(os.environ.get('HOME')), "
        "'id': os.environ['AGENT6_SESSION_ID']"
        "}, open(sys.argv[1], 'w'))",
        str(out),
    )
    notify = NotifyConfig(on_complete=argv, timeout_s=10.0)
    fire_notify_hook(
        notify,
        session_id="r1",
        session_dir=tmp_path,
        ok=True,
        reason="finish_session",
        verified="passed",
        reporter=STDIO_REPORTER,
    )
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload == {"key": None, "or_key": None, "path": True, "home": True, "id": "r1"}


def test_machine_notify_hook_env_carries_no_secrets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The machine hook goes through the same hook_env owner.
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-super-secret")
    out = tmp_path / "env2.json"
    script = (
        "import json,os,sys; json.dump({"
        "'key': os.environ.get('ANTHROPIC_API_KEY'), "
        "'path': bool(os.environ.get('PATH')), "
        "'id': os.environ['AGENT6_MACHINE_ID']"
        "}, open(sys.argv[1], 'w'))"
    )
    body = _MACHINE_CFG_BODY.replace('"PLACEHOLDER"', f'"{script}", "{out}"')
    cfg_path = tmp_path / "agent6.toml"
    cfg_path.write_text(body, encoding="utf-8")
    cfg = load_config(cfg_path)
    hook = build_machine_notify_hook(cfg, "m1", tmp_path / "inst")
    assert hook is not None
    hook("notify", "poll", "msg", "info")
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload == {"key": None, "path": True, "id": "m1"}


def test_hook_env_separates_deliberate_from_verified(tmp_path: Path) -> None:
    """AGENT6_SESSION_OK says the agent stopped deliberately; AGENT6_SESSION_VERIFIED
    says what the gate said. A finish over a red verify is OK=1 VERIFIED=failed
    -- a hook that wants "green" reads the second, because the first is true
    for a finish the verify never passed."""
    script = tmp_path / "hook.sh"
    out = tmp_path / "env.txt"
    script.write_text(
        f'#!/bin/sh\necho "$AGENT6_SESSION_OK $AGENT6_SESSION_VERIFIED" > {out}\n', encoding="utf-8"
    )
    script.chmod(0o755)
    fire_notify_hook(
        NotifyConfig(on_complete=(str(script),)),
        session_id="r1",
        session_dir=tmp_path,
        ok=True,
        reason="finish_session",
        verified="failed",
        reporter=STDIO_REPORTER,
    )
    assert out.read_text(encoding="utf-8").strip() == "1 failed"


def test_the_hook_env_names_the_session_not_the_run(tmp_path: Path) -> None:
    """`run` is the verb for the agentic coding loop; the SESSION is the thing a
    hook is told about, and it may be a run, a plan, an ask or a machine. The
    old `AGENT6_SESSION_*` names said the wrong one three times in four."""
    out = tmp_path / "env.json"
    argv = (
        "python3",
        "-c",
        "import json,os,sys; "
        "json.dump({k: v for k, v in os.environ.items() if k.startswith('AGENT6_')},"
        " open(sys.argv[1], 'w'))",
        str(out),
    )
    fire_notify_hook(
        NotifyConfig(on_complete=argv),
        session_id="brave-oak-AAAAAA",
        session_dir=tmp_path,
        ok=True,
        reason="finish_session",
        verified="passed",
        reporter=STDIO_REPORTER,
    )
    env = json.loads(out.read_text(encoding="utf-8"))

    assert env["AGENT6_SESSION_ID"] == "brave-oak-AAAAAA"
    assert env["AGENT6_SESSION_DIR"] == str(tmp_path)
    assert env["AGENT6_SESSION_OK"] == "1"
    assert env["AGENT6_SESSION_REASON"] == "finish_session"
    assert env["AGENT6_SESSION_VERIFIED"] == "passed"
    assert not [k for k in env if k.startswith("AGENT6_RUN_")], sorted(env)
