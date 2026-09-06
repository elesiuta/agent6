# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Machine tool egress: per-state network, bundle validation, and the
running machine's files made read-only in run jails (immutability)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from agent6.app.machine import (
    machine_protect_paths,
    validate_bundle,
)
from agent6.app.machine.run import machine_tool_policy_factory
from agent6.config import Config
from agent6.machine import MachineJournal, ToolState, drive, load_machine
from agent6.machine.engine import LiveWorld, ToolExecResult
from agent6.paths import jail_cache_home
from agent6.types import NetworkMode
from agent6.ui.cli.machine_cmds import (
    _resolve_network_refusal,  # pyright: ignore[reportPrivateUsage]
    _suggested_network_fix,  # pyright: ignore[reportPrivateUsage]
)

# A two-tool machine: the first tool opts into the network, the second does not.
NET_MACHINE = """
machine = "netdemo"
version = 1
initial = "fetch"

[budget]
max_usd = 1.0
max_transitions = 100

[states.fetch]
kind = "tool"
command = ["scripts/fetch.sh"]
timeout_secs = 5
network = "host"
on = { ok = "store", nonzero = "stop_fail", timeout = "stop_fail" }

[states.store]
kind = "tool"
command = ["store"]
timeout_secs = 5
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

TOOL_ONLY_MACHINE = """
machine = "toolonly"
version = 1
initial = "check"

[budget]
max_transitions = 5

[states.check]
kind = "tool"
command = ["true"]
timeout_secs = 5
on = { ok = "done", nonzero = "fail", timeout = "fail" }

[states.done]
kind = "terminal"
status = "ok"
reason = "checked"

[states.fail]
kind = "terminal"
status = "failed"
reason = "failed"
"""


def _write(tmp_path: Path, text: str, name: str = "m.asm.toml") -> Path:
    f = tmp_path / name
    f.write_text(text, encoding="utf-8")
    return f


# --- ToolState.network field -----------------------------------------


def test_tool_network_defaults_auto(tmp_path: Path) -> None:
    text = NET_MACHINE.replace('network = "host"\n', "")
    spec = load_machine(_write(tmp_path, text))
    fetch = spec.states["fetch"]
    assert isinstance(fetch, ToolState)
    assert fetch.network == "auto"


def test_tool_network_roundtrips(tmp_path: Path) -> None:
    spec = load_machine(_write(tmp_path, NET_MACHINE))
    fetch = spec.states["fetch"]
    store = spec.states["store"]
    assert isinstance(fetch, ToolState) and fetch.network == "host"
    assert isinstance(store, ToolState) and store.network == "auto"


# --- engine threads network through to the World ----------------------


@dataclass
class _RecordingWorld:
    net_calls: list[tuple[tuple[str, ...], NetworkMode]]

    def run_tool(
        self,
        argv: tuple[str, ...],
        timeout_s: float,
        *,
        network: NetworkMode = "none",
        pass_env: tuple[str, ...] = (),
    ) -> ToolExecResult:
        self.net_calls.append((argv, network))
        return ToolExecResult(exit_code=0, stdout="", timed_out=False)

    def run_agent(self, request: Any) -> Any:  # pragma: no cover - no agent states here
        raise AssertionError("no agent states")

    def now(self) -> float:
        return 1000.0

    def sleep_until(self, wake_epoch: float | None) -> Any:  # pragma: no cover
        from agent6.machine.engine import WaitWake

        return WaitWake("tick")

    def materialize_poke(self, payload: Any) -> None:  # pragma: no cover
        pass

    def notify(self, kind: str, state: str, message: str, level: str) -> None:  # pragma: no cover
        pass


def test_engine_passes_per_state_network(tmp_path: Path) -> None:
    spec = load_machine(_write(tmp_path, NET_MACHINE))
    journal = MachineJournal(tmp_path / "inst")
    world = _RecordingWorld(net_calls=[])
    result = drive(spec, journal, world, live=True)
    assert result.status == "ok"
    # fetch opted in (True); store did not (False).
    assert world.net_calls == [(("scripts/fetch.sh",), "host"), (("store",), "none")]


# --- LiveWorld (supervisor) honors the per-state network flag --------
# The engine is the host-netns supervisor; whether an opt-in is permitted at
# all is gated at machine-run startup (sandbox.network), so LiveWorld just
# passes the per-state flag straight through to the jail.


@dataclass
class _FakeJailResult:
    returncode: int = 0
    stdout: str = ""
    stderr: str = ""


def _patch_jail(monkeypatch: pytest.MonkeyPatch) -> list[Any]:
    """Capture every JailPolicy run_tool builds, without forking a real jail."""
    seen: list[Any] = []

    def fake_run_in_jail(policy: Any) -> _FakeJailResult:
        seen.append(policy)
        return _FakeJailResult()

    monkeypatch.setattr("agent6.machine.engine.run_in_jail", fake_run_in_jail)
    return seen


def _world(
    tmp_path: Path,
    isolation: str,
    *,
    cfg: Config | None = None,
    protect_paths: tuple[Path, ...] = (),
    data_dir: Path | None = None,
) -> LiveWorld:
    """A LiveWorld wired exactly as run_machine wires it: through the shared
    policy builder, so these pins hold the REAL machine confinement."""
    factory = machine_tool_policy_factory(
        cfg or Config(),
        tmp_path,
        isolation,  # type: ignore[arg-type]
        protect_paths=protect_paths,
        data_dir=data_dir,
    )
    return LiveWorld(
        cwd=tmp_path,
        journal=MachineJournal(tmp_path / "i"),
        tool_policy=factory,
        data_dir=data_dir,
    )


def test_machine_tool_jail_carries_operator_grants_and_protect_git(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The machine tool jail is built by the same policy builder as run
    commands, so an operator's extra read/write grants, hide_paths, and
    protect_git reach it. The hand-built policy carried none of them: a
    configured grant failed only in machines, and a strict machine tool could
    write .git while ordinary runs protected it."""
    seen = _patch_jail(monkeypatch)
    (tmp_path / ".git").mkdir()
    cfg = Config.model_validate(
        {
            "sandbox": {
                "extra_read_paths": ["/srv/ro"],
                "extra_write_paths": ["/srv/rw"],
                "hide_paths": ["/srv/secret"],
                "protect_git": True,
            }
        }
    )
    world = _world(tmp_path, "strict", cfg=cfg)
    world.run_tool(("true",), 5.0, network="none")
    policy = seen[-1]
    assert Path("/srv/ro") in policy.extra_ro_paths
    assert Path("/srv/rw") in policy.extra_rw_paths
    assert Path("/srv/secret") in policy.hide_paths
    assert (tmp_path / ".git").resolve() in policy.extra_protect_paths


def test_liveworld_non_network_tool_is_isolated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen = _patch_jail(monkeypatch)
    world = _world(tmp_path, "strict")
    world.run_tool(("true",), 5.0, network="none")
    assert seen[-1].network == "none"


def test_liveworld_grants_data_dir_rw_and_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A machine's data dir is RW in every tool jail + exported as
    # $AGENT6_MACHINE_DATA_DIR, so a tool script can persist on hardened too.
    seen = _patch_jail(monkeypatch)
    data = tmp_path / "i" / "data"
    world = _world(tmp_path, "hardened", data_dir=data)
    world.run_tool(("true",), 5.0, network="none")
    policy = seen[-1]
    assert data in policy.extra_rw_paths
    # Exported to match where the jail mounts the dir: the real host abspath on
    # hardened (real filesystem). The data dir lives OUTSIDE cwd by design, so a
    # relative-to-cwd path can never reach it. (strict maps it to /rw<abspath>;
    # see test_data_dir_env_matches_jail_mount.)
    assert ("AGENT6_MACHINE_DATA_DIR", str(data)) in policy.env


def test_liveworld_no_data_dir_grants_only_the_home(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen = _patch_jail(monkeypatch)
    world = _world(tmp_path, "hardened")
    world.run_tool(("true",), 5.0, network="none")
    # hardened's persistent HOME is the only extra grant: no data dir, no data grant.
    assert seen[-1].extra_rw_paths == (jail_cache_home(),)
    assert all(k != "AGENT6_MACHINE_DATA_DIR" for k, _ in seen[-1].env)


def test_liveworld_disables_python_bytecode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen = _patch_jail(monkeypatch)
    world = _world(tmp_path, "hardened")
    world.run_tool(("python3", "-m", "unittest"), 5.0, network="none")
    assert ("PYTHONDONTWRITEBYTECODE", "1") in seen[-1].env


def test_liveworld_passes_protect_paths_to_jail(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen = _patch_jail(monkeypatch)
    guarded = (tmp_path / "m.asm.toml", tmp_path / "scripts")
    world = _world(tmp_path, "strict", protect_paths=guarded)
    world.run_tool(("true",), 5.0)
    assert guarded[0] in seen[-1].extra_protect_paths
    assert guarded[1] in seen[-1].extra_protect_paths


# --- machine-file immutability (_machine_protect_paths) --------------------


def test_protect_paths_include_machine_file_and_scripts(tmp_path: Path) -> None:
    f = _write(tmp_path, NET_MACHINE)
    (tmp_path / "scripts").mkdir()
    got = machine_protect_paths(f, tmp_path)
    assert f.resolve() in got
    assert (tmp_path / "scripts").resolve() in got


def test_protect_paths_skip_missing_scripts(tmp_path: Path) -> None:
    f = _write(tmp_path, NET_MACHINE)  # no scripts/ dir
    got = machine_protect_paths(f, tmp_path)
    assert got == (f.resolve(),)


def test_protect_paths_exclude_machine_outside_cwd(tmp_path: Path) -> None:
    # A machine file outside the jail-mounted cwd isn't in the child's view, so
    # it isn't (and can't be) protected.
    outside = tmp_path.parent / "outside.asm.toml"
    outside.write_text(NET_MACHINE, encoding="utf-8")
    sub = tmp_path / "repo"
    sub.mkdir()
    assert machine_protect_paths(outside, sub) == ()


# --- bundle / script-path validation ---------------------------------------


def test_bundle_ok_when_script_exists(tmp_path: Path) -> None:
    f = _write(tmp_path, NET_MACHINE)
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "fetch.sh").write_text("#!/bin/sh\n", encoding="utf-8")
    spec = load_machine(f)
    assert validate_bundle(spec, f) == []


def test_bundle_flags_missing_script(tmp_path: Path) -> None:
    f = _write(tmp_path, NET_MACHINE)  # references scripts/fetch.sh, never created
    spec = load_machine(f)
    problems = validate_bundle(spec, f)
    assert any("not found in bundle" in p for p in problems)


def test_bundle_flags_escaping_command_ref(tmp_path: Path) -> None:
    text = NET_MACHINE.replace(
        'command = ["scripts/fetch.sh"]', 'command = ["scripts/../../etc/x"]'
    )
    f = _write(tmp_path, text)
    spec = load_machine(f)
    problems = validate_bundle(spec, f)
    assert any("escapes the bundle" in p for p in problems)


def test_bundle_flags_symlink_escape(tmp_path: Path) -> None:
    f = _write(tmp_path, NET_MACHINE)
    (tmp_path / "scripts").mkdir()
    outside = tmp_path.parent / "outside_secret"
    outside.write_text("secret", encoding="utf-8")
    (tmp_path / "scripts" / "fetch.sh").symlink_to(outside)
    spec = load_machine(f)
    problems = validate_bundle(spec, f)
    assert any("outside the bundle" in p for p in problems)


def test_bundle_reports_circular_symlink_in_scripts(tmp_path: Path) -> None:
    # A circular symlink makes Path.resolve() raise RuntimeError; the validator
    # must report it as a problem, not crash.
    f = _write(tmp_path, NET_MACHINE)
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "loop").symlink_to(tmp_path / "scripts" / "loop")
    spec = load_machine(f)
    problems = validate_bundle(spec, f)  # must not raise
    assert any("loop" in p for p in problems)


def test_bundle_reports_circular_symlink_command_ref(tmp_path: Path) -> None:
    text = NET_MACHINE.replace('command = ["scripts/fetch.sh"]', 'command = ["scripts/loop"]')
    f = _write(tmp_path, text)
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "loop").symlink_to(tmp_path / "scripts" / "loop")
    spec = load_machine(f)
    problems = validate_bundle(spec, f)  # must not raise
    assert any("fetch" not in p and "loop" in p for p in problems)


def test_machine_check_fails_on_bad_bundle(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from agent6.ui.cli import main

    text = NET_MACHINE.replace('command = ["scripts/fetch.sh"]', 'command = ["scripts/../escape"]')
    f = _write(tmp_path, text)
    monkeypatch.chdir(tmp_path)
    assert main(["machine", "check", str(f)]) == 1


def test_machine_check_passes_with_valid_bundle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from agent6.ui.cli import main

    f = _write(tmp_path, NET_MACHINE)
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "fetch.sh").write_text("#!/bin/sh\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    assert main(["machine", "check", str(f)]) == 0


def test_machine_run_refuses_escaping_bundle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Security: `machine run` must re-validate the bundle, not only `check`. On a
    # isolation that can't RO-bind the bundle, a `scripts/` symlink escaping it
    # would otherwise be executed; run must refuse before touching the world.
    from agent6.ui.cli import main

    f = _write(tmp_path, NET_MACHINE)
    (tmp_path / "scripts").mkdir()
    outside = tmp_path.parent / "outside_secret_run"
    outside.write_text("secret", encoding="utf-8")
    (tmp_path / "scripts" / "fetch.sh").symlink_to(outside)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("AGENT6_STATE_HOME", str(tmp_path / ".state"))
    assert main(["machine", "run", str(f)]) == 1


def test_machine_run_validates_config_overlay_for_pure_machine(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # B10: a pure wait/terminal machine has no agent/tool state, but its [config]
    # overlay must still be validated (and [machine] snapshot_keep honored). A
    # bogus overlay key fails the run with a config refusal instead of being
    # silently ignored.
    from agent6.ui.cli import main

    pure = (
        'machine = "pure"\nversion = 1\ninitial = "go"\n'
        "[budget]\nmax_transitions = 5\n"
        "[config.workflow]\nbogus_key = 42\n"
        '[states.go]\nkind = "wait"\nuntil = "2020-01-01T00:00:00Z"\n'
        'on = { tick = "done", signal = "done" }\n'
        '[states.done]\nkind = "terminal"\nstatus = "ok"\nreason = "x"\n'
    )
    f = _write(tmp_path, pure)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("AGENT6_STATE_HOME", str(tmp_path / ".state"))
    assert main(["machine", "run", str(f)]) == 2


def test_suggested_network_fix_block_is_unfixable(tmp_path: Path) -> None:
    # network="none" REQUIRES isolation only strict provides -> no config fix.
    text = NET_MACHINE.replace('network = "host"', 'network = "none"')
    spec = load_machine(_write(tmp_path, text))
    fetch = spec.states["fetch"]
    assert isinstance(fetch, ToolState)
    assert _suggested_network_fix(Config.model_validate({}), "hardened", [fetch]) is None


def test_resolve_network_refusal_unfixable_points_to_simulate(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    text = NET_MACHINE.replace('network = "host"', 'network = "none"')
    spec = load_machine(_write(tmp_path, text))
    fetch = spec.states["fetch"]
    assert isinstance(fetch, ToolState)
    code = _resolve_network_refusal(
        tmp_path / "m.asm.toml",
        "needs strict",
        Config.model_validate({}),
        "hardened",
        [fetch],
        tmp_path,
        {},
    )
    assert code == 2
    assert "machine test" in capsys.readouterr().err
