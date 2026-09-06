# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""sandbox.network: isolation compatibility, machine refusals, and
the supervisor subprocess that runs a machine `agent` state self-confined."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest

from agent6.app import machine_agent
from agent6.app.confine import (
    check_network_support,
)
from agent6.app.machine import (
    machine_network_refusal,
)
from agent6.config import Config, validate_config
from agent6.git_ops import CommitIdentity
from agent6.machine import AgentRequest
from agent6.machine.model import ToolState
from agent6.types import IsolationLevel


def _cfg(network: str = "session") -> Config:
    return validate_config({"sandbox": {"network": network}})


# --- _is_loopback ----------------------------------------------------------


# --- check_network_support (isolation compatibility) ------------------------


@pytest.mark.parametrize("isolation", ["strict", "none"])
def test_check_network_support_allows_off_hardened(isolation: IsolationLevel) -> None:
    # local/only_explicit_states only refused on hardened; strict supports them,
    # none is unsandboxed (warned elsewhere), so neither refuses here.
    assert check_network_support(_cfg("session"), isolation) is None
    assert check_network_support(_cfg("only_explicit_states"), isolation) is None


def test_check_network_support_refuses_only_explicit_states_on_hardened() -> None:
    msg = check_network_support(_cfg("only_explicit_states"), "hardened")
    assert msg is not None and "only_explicit_states" in msg


# --- _machine_network_refusal ----------------------------------------------

_TOOL = ToolState(kind="tool", command=("x",), timeout_secs=5, on={"ok": "s"})
_NET_TOOL = ToolState(kind="tool", command=("x",), timeout_secs=5, on={"ok": "s"}, network="host")
_BLOCK_TOOL = ToolState(kind="tool", command=("x",), timeout_secs=5, on={"ok": "s"}, network="none")


def test_refusal_networked_tool_under_block() -> None:
    r = machine_network_refusal(_cfg("session"), "strict", [_NET_TOOL])
    assert r is not None and r.fix == (("sandbox.network", "only_explicit_states"),)


def test_refusal_providers_explicit_states_strict_ok() -> None:
    # The headline combo: confined agent + audited networked tool, on strict.
    assert machine_network_refusal(_cfg("only_explicit_states"), "strict", [_NET_TOOL]) is None


def test_refusal_block_tools_on_hardened() -> None:
    r = machine_network_refusal(_cfg("session"), "hardened", [_TOOL])
    assert r is not None and "strict" in r.message


def test_refusal_explicit_none_state_on_hardened() -> None:
    # sandbox.network = host runs auto/host tools on hardened, but a state that
    # explicitly demands `none` cannot be honoured there -> refuse.
    r = machine_network_refusal(_cfg("host"), "hardened", [_BLOCK_TOOL])
    assert r is not None and "none" in r.message and r.fix == ()


def test_refusal_networked_tool_under_the_auto_default() -> None:
    """`auto` is the DEFAULT sandbox.network, and it intends no tool network, so a
    state demanding network="host" is refused on both isolation levels -- and the
    message names the ACTUAL value. Every other case here pins block/allow/
    only_explicit_states, leaving the default path unexercised."""
    for isolation in ("strict", "hardened"):
        r = machine_network_refusal(_cfg("auto"), isolation, [_NET_TOOL])
        assert r is not None and "network" in r.message
        assert "'auto'" in r.message  # not a hardcoded "session"


def test_refusal_allow_auto_tools_on_hardened_ok() -> None:
    assert machine_network_refusal(_cfg("host"), "hardened", [_TOOL]) is None


# --- supervisor subprocess: machine_agent.run_one -------------------------


@pytest.fixture
def iso(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    gdir = tmp_path / "g"
    (gdir / "agent6").mkdir(parents=True, exist_ok=True)
    (gdir / "agent6" / "config.toml").write_text(
        '[providers.anthropic]\napi_format = "anthropic"\n'
        '[models.worker]\nprovider = "anthropic"\nmodel = "claude-x"\n',
        encoding="utf-8",
    )
    monkeypatch.setenv("XDG_CONFIG_HOME", str(gdir))
    return tmp_path


def test_run_one_returns_finish_payload(
    iso: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from agent6.workflows.loop import SessionResult

    class _FakeWf:
        def __init__(self, **_kw: object) -> None:
            pass

        def run(self, _prompt: str) -> SessionResult:
            return SessionResult(
                reason="finish_session",
                completed=True,
                summary="done",
                iterations=1,
                tool_calls=0,
                finish_payload={"label": "ok"},
            )

    def _fake(*_a: object, **_k: object) -> object:
        return object()

    monkeypatch.setattr(machine_agent, "Workflow", _FakeWf)
    monkeypatch.setattr(machine_agent, "build_role_provider", _fake)
    monkeypatch.setattr(machine_agent, "reviewer_seat_provider", _fake)
    monkeypatch.setattr(machine_agent, "ToolDispatcher", _fake)

    req = machine_agent.MachineAgentRequest(
        cwd=iso,
        root=iso,
        overlay={},
        isolation="none",  # no real sandbox: landlock is a no-op
        transcript_dir=tmp_path / "t",
        request=AgentRequest(model="claude-x", prompt="go", timeout_s=5.0, provider="anthropic"),
    )
    out = machine_agent.run_one(req)
    assert out.reason == "finish_session"
    assert out.payload == {"label": "ok"}


def _stub_loop(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Stub the agent loop in machine_agent; return a dict capturing dispatcher kwargs."""
    from agent6.workflows.loop import SessionResult

    class _FakeWf:
        def __init__(self, **_kw: object) -> None:
            pass

        def run(self, _prompt: str) -> SessionResult:
            return SessionResult(
                reason="finish_session",
                completed=True,
                summary="d",
                iterations=1,
                tool_calls=0,
                finish_payload={},
            )

    captured: dict[str, Any] = {}

    def _disp(**kw: object) -> object:
        captured.update(kw)
        return object()

    def _prov(*_a: object, **_k: object) -> object:
        return object()

    monkeypatch.setattr(machine_agent, "Workflow", _FakeWf)
    monkeypatch.setattr(machine_agent, "build_role_provider", _prov)
    monkeypatch.setattr(machine_agent, "reviewer_seat_provider", _prov)
    monkeypatch.setattr(machine_agent, "ToolDispatcher", _disp)
    return captured


def test_run_one_drops_out_of_cwd_protect_paths(
    iso: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured = _stub_loop(monkeypatch)
    inside = iso / "m.asm.toml"
    inside.write_text("x", encoding="utf-8")
    outside = tmp_path.parent / "evil.asm.toml"
    outside.write_text("x", encoding="utf-8")
    req = machine_agent.MachineAgentRequest(
        cwd=iso,
        root=iso,
        overlay={},
        isolation="none",
        transcript_dir=tmp_path / "t",
        protect_paths=(inside, outside),
        request=AgentRequest(model="claude-x", prompt="go", timeout_s=5.0, provider="anthropic"),
    )
    machine_agent.run_one(req)
    # Only the in-cwd path survives the subprocess-boundary re-validation.
    assert captured["extra_protect_paths"] == (inside.resolve(),)


def test_run_one_exports_commit_identity(
    iso: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stub_loop(monkeypatch)
    for key in (
        "GIT_AUTHOR_NAME",
        "GIT_COMMITTER_NAME",
        "GIT_AUTHOR_EMAIL",
        "GIT_COMMITTER_EMAIL",
    ):
        monkeypatch.delenv(key, raising=False)
    req = machine_agent.MachineAgentRequest(
        cwd=iso,
        root=iso,
        overlay={},
        isolation="none",
        transcript_dir=tmp_path / "t",
        commit_identity=CommitIdentity(name="Machine Bot", email="bot@example.com"),
        request=AgentRequest(
            model="claude-x", prompt="go", timeout_s=5.0, provider="anthropic", mode="run"
        ),
    )
    out = machine_agent.run_one(req)
    assert out.reason == "finish_session"
    assert os.environ["GIT_AUTHOR_NAME"] == "Machine Bot"
    assert os.environ["GIT_COMMITTER_NAME"] == "Machine Bot"
    assert os.environ["GIT_AUTHOR_EMAIL"] == "bot@example.com"
    assert os.environ["GIT_COMMITTER_EMAIL"] == "bot@example.com"
