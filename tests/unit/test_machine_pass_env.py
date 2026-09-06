# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""A machine `tool` state receives an operator secret only through two
declarations: its own `pass_env` names the variable, the operator's
`[machine].pass_env` allows it (global/repo config, never the machine
overlay), and the run refuses at startup when the two disagree."""

from __future__ import annotations

from pathlib import Path

import pytest

from agent6.app.machine import machine_pass_env_refusal
from agent6.app.machine.run import machine_tool_policy_factory
from agent6.config import Config
from agent6.machine import MachineError, ToolState, load_machine
from agent6.ui.cli.machine_check import _cmd_machine_check  # pyright: ignore[reportPrivateUsage]

MACHINE = """
machine = "secretdemo"
version = 1
initial = "fetch"

[budget]
max_usd = 1.0
max_transitions = 100

[states.fetch]
kind = "tool"
command = ["fetch"]
timeout_secs = 5
pass_env = ["X_TOKEN", "X_REGION"]
on = { ok = "stop_ok", nonzero = "stop_fail", timeout = "stop_fail" }

[states.stop_ok]
kind = "terminal"
status = "ok"
reason = "done"

[states.stop_fail]
kind = "terminal"
status = "failed"
reason = "failed"
"""


def _write(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "secret.asm.toml"
    path.write_text(body, encoding="utf-8")
    return path


def test_a_tool_state_declares_the_variables_it_wants(tmp_path: Path) -> None:
    spec = load_machine(_write(tmp_path, MACHINE))
    fetch = spec.states["fetch"]
    assert isinstance(fetch, ToolState)
    assert fetch.pass_env == ("X_TOKEN", "X_REGION")
    with pytest.raises(MachineError, match="invalid environment variable name"):
        load_machine(_write(tmp_path, MACHINE.replace('"X_REGION"', '"BAD NAME"')))


def test_the_operator_allowlist_is_never_a_machine_overlay_or_a_provider_key(
    tmp_path: Path,
) -> None:
    with pytest.raises(MachineError, match=r"machine\.pass_env"):
        load_machine(_write(tmp_path, MACHINE + '\n[config.machine]\npass_env = ["X_TOKEN"]\n'))
    with pytest.raises(ValueError, match="never passes a provider key to a machine's tool"):
        Config.model_validate(
            {
                "providers": {"anthropic": {"api_format": "anthropic", "api_key_env": "MY_KEY"}},
                "machine": {"pass_env": ["MY_KEY"]},
            }
        )


def test_a_variable_the_operator_has_not_allowed_refuses_the_run(tmp_path: Path) -> None:
    spec = load_machine(_write(tmp_path, MACHINE))
    refusal = machine_pass_env_refusal(Config(), spec.states)
    assert refusal is not None and "[states.fetch] asks for X_TOKEN, X_REGION" in refusal
    assert "[machine].pass_env does not allow" in refusal
    allowed = Config.model_validate({"machine": {"pass_env": ["X_TOKEN", "X_REGION"]}})
    assert machine_pass_env_refusal(allowed, spec.states) is None


def test_a_pass_env_refusal_never_enters_the_network_fix_flow(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The refusal was handed to the frontend's network-fix flow, which on
    `hardened` offered `sandbox.network = host` (a false cause), re-checked
    the network alone, and let the run proceed with the variable copied into
    the jail. A pass_env refusal is refused outright: its fix is an allowlist
    entry, never a network change."""
    from unittest.mock import MagicMock

    from agent6.app.machine.run import run_machine

    monkeypatch.chdir(tmp_path)
    mfile = _write(tmp_path, MACHINE)
    frontend = MagicMock()
    frontend.reporter = MagicMock()
    frontend.resolve_network_fix.side_effect = AssertionError("the network-fix flow was entered")
    assert run_machine(mfile, frontend) == 2
    said = " ".join(str(c.args[0]) for c in frontend.reporter.refuse.call_args_list)
    assert "[states.fetch] asks for X_TOKEN, X_REGION" in said


def test_the_jail_gets_only_the_declared_variables_that_are_set(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("X_TOKEN", "t0k3n")
    monkeypatch.delenv("X_REGION", raising=False)
    monkeypatch.setenv("OTHER_SECRET", "no")
    factory = machine_tool_policy_factory(
        Config(),
        tmp_path,
        "strict",  # type: ignore[arg-type]
        protect_paths=(),
        data_dir=None,
    )
    env = dict(factory(("fetch",), 5.0, "none", ("X_TOKEN", "X_REGION")).env)
    assert env.get("X_TOKEN") == "t0k3n"
    assert "X_REGION" not in env and "OTHER_SECRET" not in env
    assert "X_TOKEN" not in dict(factory(("fetch",), 5.0, "none", ()).env)


def test_machine_check_names_every_declared_variable(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert _cmd_machine_check(_write(tmp_path, MACHINE)) == 0
    out = capsys.readouterr().out
    assert "[states.fetch] receives the environment variable(s) X_TOKEN, X_REGION" in out
