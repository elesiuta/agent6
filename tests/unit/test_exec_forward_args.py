# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""`exec` and `forward` command grammars.

The optional session positional used to eat the first command word
(`agent6 exec -- echo hi` treated `echo` as the session), `forward 8000`
read the port as a session id, and dispatch stripped EVERY literal `--`
from the command, corrupting valid argv like `git log -- path`. The
contract now: only the FIRST `--` separates an optional session from the
command, the command rides verbatim, and a bare number to `forward` is a
port of the newest session."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest

from agent6.sessions.ipc import write_worker_pid
from agent6.sessions.layout import SessionLayout
from agent6.ui import cli


@pytest.fixture
def seen(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> dict[str, Any]:
    calls: dict[str, Any] = {}

    def _resolve(target: str) -> SessionLayout:
        return SessionLayout(state_dir=tmp_path, session_id=target or "newest-run")

    def _exec(layout: SessionLayout, cfg: Any, cwd: Path, argv: tuple[str, ...]) -> int:
        calls.update(target=layout.session_id, argv=argv)
        return 0

    def _forward(layout: SessionLayout, port: int, local_port: int) -> int:
        calls.update(target=layout.session_id, port=port)
        return 0

    def _effective(*args: Any, **kwargs: Any) -> Any:
        return type("E", (), {"config": None})()

    monkeypatch.setattr("agent6.ui.cli._common.resolve_target", _resolve)
    monkeypatch.setattr("agent6.ui.cli.net_cmds.exec_in_session", _exec)
    monkeypatch.setattr("agent6.ui.cli.net_cmds.forward", _forward)
    monkeypatch.setattr("agent6.config.layer.load_effective", _effective)
    return calls


def test_exec_command_after_separator_runs_in_the_newest_session(seen: dict[str, Any]) -> None:
    assert cli.main(["exec", "--", "echo", "hi"]) == 0
    assert seen == {"target": "newest-run", "argv": ("echo", "hi")}


def test_exec_names_a_session_before_the_separator(seen: dict[str, Any]) -> None:
    assert cli.main(["exec", "brave-otter", "--", "echo", "hi"]) == 0
    assert seen == {"target": "brave-otter", "argv": ("echo", "hi")}


def test_exec_keeps_a_later_separator_in_the_command(seen: dict[str, Any]) -> None:
    assert cli.main(["exec", "brave-otter", "--", "git", "log", "--", "p"]) == 0
    assert seen["argv"] == ("git", "log", "--", "p")


def test_exec_without_separator_is_all_command(seen: dict[str, Any]) -> None:
    assert cli.main(["exec", "ls", "-la"]) == 0
    assert seen == {"target": "newest-run", "argv": ("ls", "-la")}


def test_exec_refuses_two_tokens_before_the_separator(
    seen: dict[str, Any], capsys: pytest.CaptureFixture[str]
) -> None:
    assert cli.main(["exec", "a", "b", "--", "cmd"]) == 2
    assert "at most one session id" in capsys.readouterr().err
    assert seen == {}


def test_forward_bare_number_is_a_port_of_the_newest_session(seen: dict[str, Any]) -> None:
    assert cli.main(["forward", "8000"]) == 0
    assert seen == {"target": "newest-run", "port": 8000}


def test_forward_session_and_port(seen: dict[str, Any]) -> None:
    assert cli.main(["forward", "brave-otter", "8000"]) == 0
    assert seen == {"target": "brave-otter", "port": 8000}


def test_exec_refuses_a_session_network_nobody_holds(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """`[sandbox].network = "session"` + an ended run used to reach
    `os.open("/proc/None/ns/user")` -- an unexpected traceback instead of a
    refusal naming the situation. exec joins a LIVE session's network only.
    The isolation seam is pinned to strict so the policy derives "session"
    on every host this suite runs on."""
    from agent6.config import Config
    from agent6.ui.cli import net_cmds

    def _strict(req: str, env: Any) -> str:
        return "strict"

    monkeypatch.setattr(net_cmds, "resolve_isolation", _strict)
    (tmp_path / "run").mkdir()
    layout = SessionLayout(state_dir=tmp_path, session_id="run")
    write_worker_pid(layout.session_dir, os.getpid())  # live, but holding no network
    cfg = Config.model_validate({"sandbox": {"network": "session"}})
    rc = net_cmds.exec_in_session(layout, cfg, tmp_path, ("true",))
    err = capsys.readouterr().err
    assert rc == 2
    assert "no network of its own to reach into" in err


def test_attach_presentation_modes_are_mutually_exclusive(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """--json silently won over --raw/--tui when combined; one presentation
    at a time, refused by the parser."""
    with pytest.raises(SystemExit) as exc:
        cli.main(["attach", "--json", "--raw"])
    assert exc.value.code == 2
    assert "not allowed with" in capsys.readouterr().err


def test_attach_since_needs_raw(capsys: pytest.CaptureFixture[str]) -> None:
    """--since replays event lines only the --raw tail renders; it was
    silently ignored elsewhere."""
    rc = cli.main(["attach", "--since", "5"])
    assert rc == 2
    assert "--since applies to --raw only" in capsys.readouterr().err


def test_exec_uses_the_runs_recorded_policy_over_current_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The run's manifest records its resolved isolation and network; exec
    reproduces THEM after a config change (run strict, config later flipped
    to none: exec must not run unconfined against "same jail"). A run with no
    stamp falls back to the current config with a warning."""
    from types import SimpleNamespace

    from agent6.config import Config
    from agent6.sessions.layout import SessionLayout
    from agent6.ui.cli import net_cmds

    layout = SessionLayout(state_dir=tmp_path / "state", session_id="r1")
    layout.ensure()
    write_worker_pid(layout.session_dir, os.getpid())  # exec joins a live run only
    (layout.session_dir / "manifest.json").write_text(
        json.dumps(
            {
                "session_id": "r1",
                "mode": "run",
                "policy": {"run_commands": "yes", "isolation": "none", "network": "host"},
            }
        ),
        encoding="utf-8",
    )
    captured: dict[str, Any] = {}

    def fake_jail_policy(cwd: Path, cfg: Config, isolation: str, argv: Any, **kw: Any) -> Any:
        captured["isolation"] = isolation
        captured["network"] = kw.get("network")
        raise RuntimeError("stop before running anything")

    def _no_pid(_d: Path) -> None:
        return None

    def _env() -> Any:
        return SimpleNamespace(sandbox_available=True)

    def _resolve(word: str, _env_v: Any) -> str:
        return word

    monkeypatch.setattr(net_cmds, "jail_policy", fake_jail_policy)
    monkeypatch.setattr(net_cmds, "read_session_netns_pid", _no_pid)
    monkeypatch.setattr(net_cmds, "detect_env", _env)
    monkeypatch.setattr(net_cmds, "resolve_isolation", _resolve)

    cfg = Config.model_validate({"sandbox": {"isolation": "strict"}})
    with pytest.raises(RuntimeError, match="stop before"):
        net_cmds.exec_in_session(layout, cfg, tmp_path, ("true",))
    assert captured["isolation"] == "none"  # the stamp, not the config
    assert captured["network"] == "host"

    # No stamp: current config with a loud warning.
    (layout.session_dir / "manifest.json").write_text(
        json.dumps({"session_id": "r1", "mode": "run"}), encoding="utf-8"
    )
    with pytest.raises(RuntimeError, match="stop before"):
        net_cmds.exec_in_session(layout, cfg, tmp_path, ("true",))
    assert captured["isolation"] == "strict"
    assert "recorded no launch policy" in capsys.readouterr().err


def test_exec_and_forward_resolve_a_session_the_way_every_other_verb_does(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Each rolled its own id lookup, so an ambiguous prefix -- which `attach`
    and `sessions show` name as ambiguous -- read as "no session 'ambig'",
    which is false: two matched."""
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    repo = tmp_path / "repo"
    repo.mkdir()
    monkeypatch.chdir(repo)
    from agent6.config.layer import resolved_state_dir

    for name in ("ambig-one11", "ambig-two11"):
        layout = SessionLayout(state_dir=resolved_state_dir(repo), session_id=name)
        layout.ensure()
        layout.manifest_path.write_text(
            json.dumps({"version": 3, "session_id": name, "mode": "run", "user_task": "t"}) + "\n",
            encoding="utf-8",
        )
        layout.logs_path.write_text(
            json.dumps({"type": "session.start", "mode": "run", "user_task": "t"}) + "\n",
            encoding="utf-8",
        )

    assert cli.main(["exec", "ambig", "--", "true"]) == 2
    assert cli.main(["forward", "ambig", "8000"]) == 2

    err = capsys.readouterr().err
    assert err.count("is ambiguous (2 matches)") == 2, err


def test_forward_names_a_finished_run_instead_of_blaming_the_config(tmp_path: Path) -> None:
    """A run's session network lives only while the run does; `forward` on a
    finished run used to explain isolation levels and network modes as if
    the config were the reason. It says the run is finished; the config
    explanation is kept for a LIVE run that made no network."""
    import io
    import json
    import os

    from agent6.sessions.ipc import write_worker_pid
    from agent6.ui.cli import net_cmds

    layout = SessionLayout(state_dir=tmp_path, session_id="done-run-AAAAAA")
    layout.ensure()
    run = layout.session_dir
    (run / "logs.jsonl").write_text(
        json.dumps({"type": "session.start", "mode": "run", "user_task": "t"})
        + "\n"
        + json.dumps({"type": "session.end", "reason": "finish_session", "all_passed": True})
        + "\n",
        encoding="utf-8",
    )
    out = io.StringIO()
    assert net_cmds.forward(layout, 8765, 0, out=out) == 2
    assert "done-run-AAAAAA is passed; a session network exists only while its run does" in (
        out.getvalue()
    )
    assert "strict isolation" not in out.getvalue()

    # Live (a worker holds it) but without a network of its own: the config explanation.
    (run / "logs.jsonl").write_text(
        json.dumps({"type": "session.start", "mode": "run", "user_task": "t"}) + "\n",
        encoding="utf-8",
    )
    write_worker_pid(run, os.getpid())
    out = io.StringIO()
    assert net_cmds.forward(layout, 8765, 0, out=out) == 2
    assert "strict isolation" in out.getvalue()


def test_exec_refuses_a_run_that_is_over(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The help promises the run's own jail; a finished run's is gone, and
    `exec` built a fresh one (today's HEAD, none of the run's processes) and
    ran the command there in silence, refusing only a run that had recorded
    the session network, with a remedy that made it worse."""
    import json
    import os

    from agent6.config import Config
    from agent6.sessions.layout import SessionLayout
    from agent6.types import JailPolicy
    from agent6.ui.cli.net_cmds import exec_in_session

    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    repo = tmp_path / "repo"
    repo.mkdir()
    from agent6.config.layer import resolved_state_dir

    layout = SessionLayout(state_dir=resolved_state_dir(repo), session_id="over-run-AAAA11")
    layout.ensure()
    layout.logs_path.write_text(
        json.dumps({"type": "session.start", "mode": "run", "user_task": "t"})
        + "\n"
        + json.dumps({"type": "session.end", "reason": "finish_session", "all_passed": True})
        + "\n",
        encoding="utf-8",
    )
    (layout.session_dir / "worker.pid").write_text("999999999", encoding="utf-8")
    ran: list[tuple[str, ...]] = []

    def fake_run(policy: JailPolicy, **_kw: object) -> int:
        ran.append(tuple(policy.argv))
        return os.EX_OK

    monkeypatch.setattr("agent6.ui.cli.net_cmds.run_in_jail", fake_run)

    rc = exec_in_session(layout, Config(), repo, ("pwd",))

    assert rc == 2 and ran == []
    err = capsys.readouterr().err
    assert "REFUSING: over-run-AAAA11 is " in err and "exists only while its run does" in err
