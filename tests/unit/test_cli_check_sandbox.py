# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""`agent6 check sandbox` runs its probes under the host's *effective* isolation.

Pure-logic tests: the jail itself is stubbed out, so these run on any host
(no namespaces required). They pin the behaviour that on a host that can only
run `hardened` (default-seccomp Docker, AppArmor-restricted Ubuntu) the check
PASSES rather than spuriously failing against a `strict` jail the agent would
never use there.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from agent6.config import Config, SandboxConfig
from agent6.sandbox.detect import IsolationUnavailableError
from agent6.sandbox.jail import JailUnavailableError
from agent6.types import CommandResult, JailPolicy
from agent6.ui.cli import check_cmds


def _fake_result(argv: tuple[str, ...], rc: int) -> CommandResult:
    return CommandResult(argv=argv, returncode=rc, stdout="", stderr="", duration_s=0.0)


@pytest.fixture
def stub_jail(monkeypatch: pytest.MonkeyPatch) -> list[JailPolicy]:
    """Stub landlock_abi + run_in_jail; record every policy the check builds."""
    seen: list[JailPolicy] = []
    monkeypatch.setattr(check_cmds, "landlock_abi", lambda: 8)

    def fake_run(policy: JailPolicy) -> CommandResult:
        seen.append(policy)
        # getent (network probe) "fails" (blocked); everything else succeeds.
        rc = 2 if policy.argv[0].endswith("getent") else 0
        return _fake_result(policy.argv, rc)

    monkeypatch.setattr(check_cmds, "run_in_jail", fake_run)
    return seen


def _force_profile(
    monkeypatch: pytest.MonkeyPatch, isolation: str, reason: str | None = None
) -> None:
    monkeypatch.setattr(check_cmds, "detect_env", object)  # returns a throwaway env stub

    def _reason(_env: object) -> str | None:
        return reason

    monkeypatch.setattr(check_cmds, "degrade_reason", _reason)

    def fake_select(_req: str, _env: object) -> str:
        return isolation

    monkeypatch.setattr(check_cmds, "resolve_isolation", fake_select)


def _honour_request(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stub the resolver to return exactly what the config asked for."""
    monkeypatch.setattr(check_cmds, "detect_env", object)

    def _reason(_env: object) -> str | None:
        return None

    def _resolve(requested: str, _env: object) -> str:
        return requested

    monkeypatch.setattr(check_cmds, "degrade_reason", _reason)
    monkeypatch.setattr(check_cmds, "resolve_isolation", _resolve)


def test_check_sandbox_hardened_passes_and_skips_network(
    monkeypatch: pytest.MonkeyPatch, stub_jail: list[JailPolicy], capsys: pytest.CaptureFixture[str]
) -> None:
    _force_profile(monkeypatch, "hardened")
    rc = check_cmds._cmd_check_sandbox()  # pyright: ignore[reportPrivateUsage]
    out = capsys.readouterr().out
    assert rc == 0, out
    assert "effective isolation (auto): hardened" in out
    # Network probe is reported n/a, not run, under hardened.
    assert "jail_blocks_network: n/a under hardened" in out
    assert all(p.isolation == "hardened" for p in stub_jail)
    assert not any(p.argv[0].endswith("getent") for p in stub_jail)


def test_check_sandbox_strict_runs_network_probe(
    monkeypatch: pytest.MonkeyPatch, stub_jail: list[JailPolicy], capsys: pytest.CaptureFixture[str]
) -> None:
    _force_profile(monkeypatch, "strict")
    rc = check_cmds._cmd_check_sandbox()  # pyright: ignore[reportPrivateUsage]
    out = capsys.readouterr().out
    assert rc == 0, out
    assert "effective isolation (auto): strict" in out
    # The network probe actually runs under strict, with isolation=strict.
    getent = [p for p in stub_jail if p.argv[0].endswith("getent")]
    assert len(getent) == 1
    assert getent[0].isolation == "strict"
    assert getent[0].network == "none"


def test_check_sandbox_none_skips_probes(
    monkeypatch: pytest.MonkeyPatch, stub_jail: list[JailPolicy], capsys: pytest.CaptureFixture[str]
) -> None:
    from agent6.sandbox._tool_paths import ToolMountNotes

    _force_profile(monkeypatch, "none")
    monkeypatch.setattr(
        check_cmds,
        "tool_mount_notes",
        lambda: ToolMountNotes(exposes_home_dir=("~/.local/bin/x -> ~/.local/share/x",)),
    )
    rc = check_cmds._cmd_check_sandbox()  # pyright: ignore[reportPrivateUsage]
    out = capsys.readouterr().out
    # No kernel sandbox -> reported FAIL, and no jail invocations attempted.
    assert rc == 1, out
    assert "effective isolation (auto): none" in out
    assert stub_jail == []
    # Nothing is confined under "none": grant language about tool dirs would
    # describe a boundary that does not exist, so the block is absent.
    assert "granted read-only" not in out
    assert "mounted read-only" not in out


def test_check_sandbox_degraded_names_why(
    monkeypatch: pytest.MonkeyPatch, stub_jail: list[JailPolicy], capsys: pytest.CaptureFixture[str]
) -> None:
    """A degraded level never appears without its cause. Reproduced on a
    userns-blocked host (user.max_user_namespaces = 0): the line read
    `effective isolation (auto): hardened` and nothing said why, while
    `check config` did."""
    from agent6.sandbox._tool_paths import ToolMountNotes

    why = "unprivileged user namespaces are disabled (user.max_user_namespaces = 0)"
    _force_profile(monkeypatch, "hardened", reason=why)
    monkeypatch.setattr(
        check_cmds,
        "tool_mount_notes",
        lambda: ToolMountNotes(exposes_home_dir=("~/.local/bin/x -> ~/.local/share/x",)),
    )
    rc = check_cmds._cmd_check_sandbox()  # pyright: ignore[reportPrivateUsage]
    out = capsys.readouterr().out
    assert rc == 0, out
    assert f"not strict: {why}" in out
    # Under hardened nothing is MOUNTED (no mount namespace): the tool-dir
    # exposure is a Landlock read grant and the words must say so.
    assert "granted read-only (Landlock path rules)" in out
    assert "mounted read-only into the jail" not in out


def test_check_sandbox_probes_the_isolation_the_config_selects(
    monkeypatch: pytest.MonkeyPatch, stub_jail: list[JailPolicy], capsys: pytest.CaptureFixture[str]
) -> None:
    """The probes exercise the jail a run here would use. With
    `sandbox.isolation = "hardened"` configured on a strict-capable host the
    section reported `(auto): strict` and probed strict, contradicting the
    `check config` section of the same command."""
    _honour_request(monkeypatch)
    cfg = Config(sandbox=SandboxConfig(isolation="hardened"))
    rc = check_cmds._cmd_check_sandbox(cfg)  # pyright: ignore[reportPrivateUsage]
    out = capsys.readouterr().out
    assert rc == 0, out
    assert "effective isolation (hardened): hardened" in out
    assert stub_jail and all(p.isolation == "hardened" for p in stub_jail)


def test_check_names_a_jail_binary_it_cannot_run(
    monkeypatch: pytest.MonkeyPatch, stub_jail: list[JailPolicy], capsys: pytest.CaptureFixture[str]
) -> None:
    """An unusable AGENT6_JAIL_BIN read as "this host blocks user namespaces"
    (`userns supported: False`, `auto` resolved to hardened): the environment
    probe hands back the binary's own refusal, and each section prints it."""
    refusal = "agent6-jail at /opt/agent6-jail cannot be executed: Exec format error. Reinstall it"

    def _binary_refusal() -> object:
        raise JailUnavailableError(refusal)

    monkeypatch.setattr(check_cmds, "detect_env", _binary_refusal)
    rc = check_cmds._cmd_check_sandbox(None)  # pyright: ignore[reportPrivateUsage]
    out = capsys.readouterr().out
    assert rc == 1, out
    assert f"[FAIL] jail_binary: {refusal}" in out
    assert "user namespaces" not in out
    assert stub_jail == []
    checks = check_cmds._check_config_section(Config())  # pyright: ignore[reportPrivateUsage]
    assert (checks[0].name, checks[0].status, checks[0].detail) == (
        "config.isolation",
        "FAIL",
        refusal,
    )
    assert "user namespaces" not in capsys.readouterr().out


def test_check_sandbox_fails_on_an_isolation_this_host_refuses(
    monkeypatch: pytest.MonkeyPatch, stub_jail: list[JailPolicy], capsys: pytest.CaptureFixture[str]
) -> None:
    """An explicit level the host cannot give is a FAIL naming the refusal (a
    run would refuse too), never probes run under some other level."""
    monkeypatch.setattr(check_cmds, "detect_env", object)

    def _refuse(req: str, _env: object) -> str:
        raise IsolationUnavailableError(f"sandbox.isolation = {req!r} requires user namespaces")

    monkeypatch.setattr(check_cmds, "resolve_isolation", _refuse)
    rc = check_cmds._cmd_check_sandbox(  # pyright: ignore[reportPrivateUsage]
        Config(sandbox=SandboxConfig(isolation="strict"))
    )
    out = capsys.readouterr().out
    assert rc == 1, out
    assert "requires user namespaces" in out
    assert stub_jail == []


def test_check_sandbox_names_which_opt_out_left_nothing_to_probe(
    monkeypatch: pytest.MonkeyPatch, stub_jail: list[JailPolicy], capsys: pytest.CaptureFixture[str]
) -> None:
    """`none` from the config is the operator's opt-out, not a platform without
    a sandbox: the skip line must not blame the platform."""
    _honour_request(monkeypatch)
    rc = check_cmds._cmd_check_sandbox(  # pyright: ignore[reportPrivateUsage]
        Config(sandbox=SandboxConfig(isolation="none"))
    )
    out = capsys.readouterr().out
    assert rc == 1, out
    assert "sandbox.isolation = 'none': commands run unconfined" in out
    assert "no kernel sandbox" not in out
    assert stub_jail == []


def test_check_config_runs_the_refusal_ladder_a_run_applies(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """`check config` FAILS on an explicit knob the selected isolation cannot
    honour, with the run's own refusal text: hardened + network = session."""
    env = SimpleNamespace(
        kernel=SimpleNamespace(raw="6.8"),
        userns_supported=True,
        sandbox_available=True,
        landlock_abi=5,
    )

    def _no_reason(_env: object) -> str | None:
        return None

    def _as_requested(requested: str, _env: object) -> str:
        return requested

    monkeypatch.setattr(check_cmds, "detect_env", lambda: env)
    monkeypatch.setattr(check_cmds, "degrade_reason", _no_reason)
    monkeypatch.setattr(check_cmds, "resolve_isolation", _as_requested)
    cfg = Config.model_validate({"sandbox": {"isolation": "hardened", "network": "session"}})
    checks = check_cmds._check_config_section(cfg)  # pyright: ignore[reportPrivateUsage]
    out = capsys.readouterr().out
    refusal = next(c for c in checks if c.name == "config.refusal")
    assert refusal.status == "FAIL"
    assert "sandbox.network = 'session' requires the strict isolation" in refusal.detail
    assert "[FAIL] a run would refuse" in out
    ok = Config.model_validate({"sandbox": {"isolation": "hardened"}})
    checks = check_cmds._check_config_section(ok)  # pyright: ignore[reportPrivateUsage]
    assert next(c for c in checks if c.name == "config.refusal").status == "PASS"
