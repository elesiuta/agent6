# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""One ACP prompt becoming one agent6 run, driven over a real connection."""

from __future__ import annotations

import io
import json
import os
import select
import subprocess
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from agent6.app.reporter import Reporter
from agent6.config import Config
from agent6.config.layer import EffectiveConfig
from agent6.config.model import ConfigError
from agent6.sessions.layout import SessionLayout
from agent6.tools.operator_prompts import OperatorPrompts
from agent6.ui.acp import runner
from agent6.ui.acp import session as session_mod
from agent6.ui.acp.runner import Announced, RunBridge, forwarding_reporter, option_kind, stop_reason
from agent6.ui.acp.server import ACPServer


class _Wire:
    """A live connection: messages in one pipe, replies out the other."""

    def __init__(self) -> None:
        self._in_r, self._in_w = os.pipe()
        self._out_r, self._out_w = os.pipe()
        self.server = ACPServer(
            stdin=os.fdopen(self._in_r, "rb"),
            stdout=os.fdopen(self._out_w, "wb"),
        )
        self.server.sessions = RunBridge(server=self.server).sessions()
        # Unbuffered, so `select` telling us nothing is waiting is the truth.
        self._reader = os.fdopen(self._out_r, "rb", buffering=0)
        self._thread = threading.Thread(target=self.server.serve, daemon=True)
        self._thread.start()

    def send(self, **message: Any) -> None:
        os.write(self._in_w, json.dumps({"jsonrpc": "2.0", **message}).encode() + b"\n")

    def recv(self, timeout: float = 5.0) -> dict[str, Any]:
        if not select.select([self._out_r], [], [], timeout)[0]:
            raise AssertionError("the editor got nothing back")
        return json.loads(self._reader.readline())

    def until(self, method: str, timeout: float = 5.0) -> dict[str, Any]:
        for _ in range(50):
            message = self.recv(timeout=timeout)
            if message.get("method") == method or method == "":
                return message
        raise AssertionError(f"no {method} arrived")

    def close(self) -> None:
        os.close(self._in_w)
        self._thread.join(timeout=5.0)

    def new_session(self, cwd: Path) -> str:
        self.send(id=1, method="initialize", params={"clientCapabilities": {}})
        self.recv()
        self.send(id=2, method="session/new", params={"cwd": str(cwd)})
        return str(self.recv()["result"]["sessionId"])

    def prompt(self, session_id: str, text: str, *, req_id: int = 3) -> None:
        self.send(
            id=req_id,
            method="session/prompt",
            params={"sessionId": session_id, "prompt": [{"type": "text", "text": text}]},
        )


def _ignore(_path: Path) -> bool:
    """A cancel whose marker landed (nothing here reads the run dir)."""
    return True


def _repo(path: Path) -> Path:
    """A session's cwd has to pass the same git-repo wall `agent6 run` uses."""
    subprocess.run(["git", "init", "-q", "-b", "main", str(path)], check=True)
    return path


def _loaded(*_a: object, **_k: object) -> EffectiveConfig:
    """The loader's result for a test that stubs `run_task`: the real type (the
    bridge reads `explicit_leaves` off it), unconfigured (the real loader
    would refuse a config with no model to run)."""
    return EffectiveConfig(config=Config(), sources={}, layers=())


def test_the_reporter_never_writes_to_stdout(capsys: pytest.CaptureFixture[str]) -> None:
    """stdout IS the protocol stream. One status line on it desynchronises the
    connection irrecoverably, and no editor recovers from that. The same line
    reaches the editor as agent6's own prose, marked once."""
    wire = io.BytesIO()
    said: list[str] = []
    reporter = forwarding_reporter(ACPServer(stdin=io.BytesIO(), stdout=wire), "s", said)
    reporter.out("a status line")
    reporter.note("a note")
    captured = capsys.readouterr()
    assert captured.out == "", "the wire must carry nothing but JSON-RPC"
    assert "a status line" in captured.err and "[agent6] a note" in captured.err
    texts = [
        json.loads(line)["params"]["update"]["content"]["text"]
        for line in wire.getvalue().decode().splitlines()
    ]
    assert texts == ["[agent6] a status line", "[agent6] a note"]
    assert said == ["a status line", "[agent6] a note"]


def test_a_cancel_reaches_the_run_it_names(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The run id is minted BEFORE the run starts, so the session has a handle
    to address. Letting the lifecycle mint its own left `session_id` empty: the
    cancel reported success while the run continued to completion, spending
    budget and making commits."""
    stopped: list[Path] = []

    def _record(path: Path) -> bool:
        stopped.append(path)
        return True

    monkeypatch.setattr(session_mod, "request_stop", _record)
    monkeypatch.chdir(tmp_path)

    started, release = threading.Event(), threading.Event()

    def _blocking_run(*_a: object, **kw: object) -> int:
        started.set()
        release.wait(timeout=5.0)
        return 0

    monkeypatch.setattr(runner, "run_task", _blocking_run)
    monkeypatch.setattr(runner, "load_session_config", _loaded)

    wire = _Wire()
    try:
        session_id = wire.new_session(_repo(tmp_path))
        wire.prompt(session_id, "do the thing")
        assert started.wait(timeout=5.0), "the run never started"
        wire.send(method="session/cancel", params={"sessionId": session_id})
        for _ in range(100):
            if stopped:
                break
            threading.Event().wait(0.05)
        assert stopped, "the stop marker never reached a run directory"
        assert stopped[0].parent.name == "runs", stopped[0]
        assert stopped[0].name, "the run id was empty, so the cancel addressed nothing"
    finally:
        release.set()
        wire.close()


def test_a_cancelled_turn_says_so(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(session_mod, "request_stop", _ignore)
    monkeypatch.chdir(tmp_path)
    started, release = threading.Event(), threading.Event()

    def _blocking_run(*_a: object, **kw: object) -> int:
        started.set()
        release.wait(timeout=5.0)
        return 0

    monkeypatch.setattr(runner, "run_task", _blocking_run)
    monkeypatch.setattr(runner, "load_session_config", _loaded)
    wire = _Wire()
    try:
        session_id = wire.new_session(_repo(tmp_path))
        wire.prompt(session_id, "do the thing")
        assert started.wait(timeout=5.0)
        wire.send(method="session/cancel", params={"sessionId": session_id})
        release.set()
        answer = wire.until("")
        assert answer["result"]["stopReason"] == "cancelled"
    finally:
        release.set()
        wire.close()


def test_the_runs_journal_streams_out_as_session_update(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The tail is the whole live view: without it an editor sees a turn that
    starts, says nothing for minutes, and then answers.

    Driven by the RECORDED journal the fold's golden test uses, not by
    hand-written events: a fabricated shape the engine never emits is how a
    surface tests green while rendering nothing.
    """
    monkeypatch.chdir(tmp_path)
    recorded = Path(__file__).parent.parent / "unit" / "data" / "golden_session_logs.jsonl"

    def _writing_run(*_a: object, **kw: object) -> int:
        session_id = kw["session_id"]
        assert isinstance(session_id, str)
        layout = SessionLayout(state_dir=runner.state_dir(tmp_path), session_id=session_id)
        layout.session_dir.mkdir(parents=True, exist_ok=True)
        layout.logs_path.write_bytes(recorded.read_bytes())
        return 0

    monkeypatch.setattr(runner, "run_task", _writing_run)
    monkeypatch.setattr(runner, "load_session_config", _loaded)
    wire = _Wire()
    try:
        session_id = wire.new_session(_repo(tmp_path))
        wire.prompt(session_id, "do the thing")
        seen: list[str] = []
        for _ in range(40):
            message = wire.recv()
            if message.get("method") != "session/update":
                continue
            assert message["params"]["sessionId"] == session_id
            seen.append(json.dumps(message["params"]["update"]))
            if any("Session passed" in body for body in seen):
                break
        assert any("Let me read the file." in body for body in seen), "no thinking reached it"
        assert any('"tool_call"' in body for body in seen), "no tool call reached it"
        assert any("Session passed" in body for body in seen), "the ending never reached it"
    finally:
        wire.close()


def test_a_fault_after_the_journal_opened_still_reaches_the_editor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The bare-refusal guard assumed a journal means the fold reported the
    ending, but a fault between the journal's first line and its session.end
    ended the turn as `{"stopReason": "refusal"}` with no words at all."""
    from agent6.paths import state_dir
    from agent6.sessions.layout import SessionLayout

    monkeypatch.chdir(tmp_path)

    def _dies_mid_run(*_a: object, session_id: str = "", **_kw: object) -> int:
        layout = SessionLayout(state_dir(tmp_path), session_id)
        layout.ensure()
        layout.logs_path.write_text(
            json.dumps({"type": "session.start", "mode": "run", "user_task": "t"}) + "\n",
            encoding="utf-8",
        )
        raise RuntimeError("the provider client exploded")

    monkeypatch.setattr(runner, "run_task", _dies_mid_run)
    monkeypatch.setattr(runner, "load_session_config", _loaded)
    wire = _Wire()
    try:
        session_id = wire.new_session(_repo(tmp_path))
        wire.prompt(session_id, "do the thing")
        said = wire.until("session/update")
        text = said["params"]["update"]["content"]["text"]
        assert "the run failed" in text and "exploded" in text, text
        assert wire.until("")["result"]["stopReason"] == "refusal"
    finally:
        wire.close()
    # A bug's traceback reaches stderr, the way the CLI's crash path saves one.
    assert "RuntimeError: the provider client exploded" in capsys.readouterr().err


def test_a_run_that_cannot_start_says_why(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A broken config is the ordinary case, and it raises before the run has a
    journal to carry the reason. The editor would otherwise see a turn end with
    a stop reason and no words at all."""
    monkeypatch.chdir(tmp_path)

    def _broken(*_a: object, **_kw: object) -> object:
        raise ConfigError("Config file is not valid TOML (agent6.toml)")

    monkeypatch.setattr(runner, "load_session_config", _broken)
    wire = _Wire()
    try:
        session_id = wire.new_session(_repo(tmp_path))
        wire.prompt(session_id, "do the thing")
        said = wire.until("session/update")
        text = said["params"]["update"]["content"]["text"]
        assert "could not start" in text and "agent6.toml" in text, text
        assert wire.until("")["result"]["stopReason"] == "refusal"
    finally:
        wire.close()


def _acp_front(*, reply: str | None):
    """The real ACP frontend, with the ask seam captured."""
    from agent6.app.frontend import FrontendCapabilities
    from agent6.ui.acp.frontend import acp_frontend

    asked: list[tuple[str, tuple[str, ...], bool | None]] = []

    def _ask(
        prompt: str,
        options: tuple[str, ...],
        standing: bool | None,
        _call_id: int | None,
        until: Callable[[], bool] | None = None,
    ) -> str | None:
        asked.append((prompt, options, standing))
        return reply

    front = acp_frontend(
        ask=_ask,
        capabilities=FrontendCapabilities(),
        agent6_exe=lambda: "agent6",
        spawn_detached_resume=lambda _cwd, _rid, _flags: "",
    )
    return front, asked


def _bridge(answer: dict[str, Any]) -> RunBridge:
    bridge = RunBridge(server=ACPServer(stdin=io.BytesIO(), stdout=io.BytesIO()))

    def _answer(*_a: object, **_kw: object) -> dict[str, Any]:
        return answer

    bridge.server.request = _answer  # pyright: ignore[reportAttributeAccessIssue]
    return bridge


def test_an_approval_round_trips_through_the_editor() -> None:
    bridge = _bridge({"outcome": {"outcome": "selected", "optionId": "0"}})
    session = session_mod.Session(acp_id="s", cwd=Path("/x"))
    assert (
        bridge.ask(
            session, Announced(turn=1), "Allow run_command: ls", ("allow", "deny"), True, None
        )
        == "allow"
    )


@pytest.mark.parametrize(
    "answer",
    [
        {"outcome": {"outcome": "cancelled"}},
        {},
        {"outcome": {"outcome": "selected", "optionId": "99"}},
        {"outcome": {"outcome": "selected"}},
    ],
)
def test_only_an_option_we_offered_is_an_answer(answer: dict[str, Any]) -> None:
    """A timeout, a cancel and an echoed string are all "no answer". Treating
    an unknown string as one would let it become an allow by prefix, and the
    seam reads a None as the cautious answer."""
    bridge = _bridge(answer)
    session = session_mod.Session(acp_id="s", cwd=Path("/x"))
    assert (
        bridge.ask(
            session, Announced(turn=1), "Allow run_command: rm -rf /", ("allow", "deny"), True, None
        )
        is None
    )


def test_the_option_kinds_carry_what_the_editor_may_remember() -> None:
    """`allow once` is the fetch tool's off-list host, where an editor that
    remembers the answer would silently cover a different host."""
    assert option_kind("allow", True) == "allow_always"
    assert option_kind("allow once", False) == "allow_once"
    assert option_kind("deny", True) == "reject_once"
    assert option_kind("dark", None) == "allow_once"
    # The MODEL writes a question's options. Keying on the text let it name one
    # "allow" and have it advertised as a permission the editor may REMEMBER.
    assert option_kind("allow", None) == "allow_once"


def test_the_stop_reason_is_one_acp_defines() -> None:
    assert stop_reason(0) == "end_turn"
    assert stop_reason(1) == "refusal"
    assert stop_reason(2) == "refusal"
    assert stop_reason(130) == "cancelled"


def test_a_deliberate_finish_over_a_red_gate_is_end_turn() -> None:
    """Exit 4 is "finished deliberately, gate not green": the agent answered,
    so the editor must not be told the turn was refused (a live smoke saw a
    committed, summarised fix reported as stopReason=refusal)."""
    assert stop_reason(4) == "end_turn"


def test_an_editor_driven_run_is_not_refused_for_lack_of_a_terminal() -> None:
    """`agent6 acp`'s stdin is the protocol pipe, never a tty.

    The refusal used to test the tty rather than the surface's own declaration,
    so with the stock `run_commands = "ask"` EVERY editor-driven run was
    refused before it started -- and the whole `session/request_permission`
    path it refused on behalf of was unreachable.
    """
    from agent6.app.preflight import headless_approval_refusal
    from agent6.config import Config
    from agent6.ui.acp.server import capabilities_from

    cfg = Config.model_validate({"sandbox": {"run_commands": "ask"}})
    caps = capabilities_from({})  # a client that declared nothing
    assert caps.can_ask is True, "every ACP client must answer session/request_permission"
    assert headless_approval_refusal(cfg, tui_enabled=False, away="", can_ask=caps.can_ask) is None


def test_a_turn_cancelled_while_queued_never_starts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Runs are serialised, so a second turn can wait minutes for the lock.

    Minting the run id inside the lock left that whole window addressing
    nothing: the cancel wrote no stop marker, the queued turn then ran to
    completion spending budget and making commits, and the editor was told
    "cancelled" the entire time.
    """
    monkeypatch.setattr(session_mod, "request_stop", _ignore)
    monkeypatch.chdir(tmp_path)
    ran: list[str] = []
    first_started, release = threading.Event(), threading.Event()

    def _blocking_run(*_a: object, **kw: object) -> int:
        ran.append(str(kw["session_id"]))
        first_started.set()
        release.wait(timeout=5.0)
        return 0

    monkeypatch.setattr(runner, "run_task", _blocking_run)
    monkeypatch.setattr(runner, "load_session_config", _loaded)
    wire = _Wire()
    try:
        first = wire.new_session(_repo(tmp_path))
        wire.send(id=9, method="session/new", params={"cwd": str(tmp_path)})
        second = str(wire.recv()["result"]["sessionId"])
        wire.prompt(first, "the long one", req_id=3)
        assert first_started.wait(timeout=5.0)
        wire.prompt(second, "the queued one", req_id=4)
        wire.send(method="session/cancel", params={"sessionId": second})
        release.set()
        answers: dict[int, str] = {}
        while len(answers) < 2:
            reply = wire.until("")  # replies, past the queued turn's own notice
            if "result" in reply:
                answers[reply["id"]] = reply["result"]["stopReason"]
        assert len(ran) == 1, f"the cancelled turn started anyway: {ran}"
        assert answers[4] == "cancelled", answers
        assert answers[3] == "end_turn", answers
    finally:
        release.set()
        wire.close()


def test_a_session_outside_a_git_repo_is_refused(tmp_path: Path) -> None:
    """`cwd` arrives over the wire and becomes what the jail mounts WRITABLE.

    `agent6 run` walls this with the same check; the ACP path was the one
    caller that never ran it, so a client could point a run at any absolute
    path -- and `$HOME` on a machine with dotfiles under git would hand the
    model the whole home directory as its workspace.
    """
    wire = _Wire()
    try:
        wire.send(id=1, method="initialize", params={"clientCapabilities": {}})
        wire.recv()
        wire.send(id=2, method="session/new", params={"cwd": str(tmp_path)})
        reply = wire.recv()
        assert "result" not in reply, "a non-repo directory became a workspace"
        assert "not a git repository" in reply["error"]["message"]
    finally:
        wire.close()


def test_a_relative_or_missing_cwd_is_refused(tmp_path: Path) -> None:
    wire = _Wire()
    try:
        wire.send(id=1, method="initialize", params={"clientCapabilities": {}})
        wire.recv()
        for cwd in ("relative/path", None):
            wire.send(id=2, method="session/new", params={"cwd": cwd})
            assert "absolute" in wire.recv()["error"]["message"]
    finally:
        wire.close()


def test_a_refusal_that_never_reached_a_journal_still_says_why(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """About a dozen lifecycle paths `return 2` after writing their reason to
    the reporter and nowhere else. With no `session.end` the fold produces no
    ending, so the editor saw a turn stop with a stop reason and no words."""
    monkeypatch.chdir(tmp_path)

    def _refusing_run(*_a: object, **kw: object) -> int:
        reporter = kw["reporter"]
        assert isinstance(reporter, Reporter)
        reporter.err("REFUSING: another writer holds this repository")
        return 2

    monkeypatch.setattr(runner, "run_task", _refusing_run)
    monkeypatch.setattr(runner, "load_session_config", _loaded)
    wire = _Wire()
    try:
        session_id = wire.new_session(_repo(tmp_path))
        wire.prompt(session_id, "do the thing")
        said = wire.until("session/update")["params"]["update"]["content"]["text"]
        assert "another writer holds this repository" in said
        assert wire.until("")["result"]["stopReason"] == "refusal"
    finally:
        wire.close()


def test_a_closed_editor_stops_waiting_for_answers_it_will_never_get() -> None:
    """The read loop is the only thing that delivers a client's answer, so once
    it is gone a worker parked on an approval waits the full permission
    timeout -- far longer than the EOF grace, so the process always exited and
    killed the run it was trying to let finish."""
    server = ACPServer(stdin=io.BytesIO(), stdout=io.BytesIO())
    answered: list[dict[str, Any]] = []
    asking = threading.Thread(
        target=lambda: answered.append(
            server.request("session/request_permission", {}, timeout_s=30.0)
        ),
        daemon=True,
    )
    asking.start()
    for _ in range(100):  # let it register its pending slot
        if server._pending:  # pyright: ignore[reportPrivateUsage]
            break
        threading.Event().wait(0.01)
    server.abandon_pending()
    asking.join(timeout=5.0)
    assert answered == [{}], "the worker was left waiting"


def test_a_question_with_no_buttons_is_not_put_to_the_editor() -> None:
    """ACP v1 carries a question as a permission request, whose options ARE
    the buttons. A free-form `ask_user` has none, so there was nothing to
    press: it stalled the full 300s timeout and then answered "said nothing"
    anyway -- up to eight times in one call, sequentially."""
    sent: list[tuple[str, dict[str, Any]]] = []
    bridge = _bridge({"outcome": {"outcome": "cancelled"}})

    def _record(method: str, params: dict[str, Any], **_kw: object) -> dict[str, Any]:
        sent.append((method, params))
        return {}

    bridge.server.request = _record  # pyright: ignore[reportAttributeAccessIssue]
    session = session_mod.Session(acp_id="s", cwd=Path("/x"))
    assert (
        bridge.ask(session, Announced(turn=1), "What should the theme be?", (), None, None) is None
    )
    assert sent == [], "an unanswerable prompt was still shown"
    assert bridge.ask(session, Announced(turn=1), "Theme?", ("dark", "light"), None, None) is None
    assert len(sent) == 1, "a question WITH options still goes out"


def test_an_approval_closes_the_tool_call_it_announced() -> None:
    """`toolCall` is required on a permission request, so an ask announces one.
    ACP models a tool call as an entity with a lifecycle, so an editor kept one
    PENDING entry per approval for the life of the session."""
    sent: list[dict[str, Any]] = []
    bridge = _bridge({"outcome": {"outcome": "selected", "optionId": "0"}})
    bridge.server.notify_raw = sent.append  # pyright: ignore[reportAttributeAccessIssue]
    session = session_mod.Session(acp_id="s", cwd=Path("/x"))
    assert (
        bridge.ask(
            session, Announced(turn=1), "Allow run_command: ls", ("allow", "deny"), True, None
        )
        == "allow"
    )
    closes = [m for m in sent if m["params"]["update"]["sessionUpdate"] == "tool_call_update"]
    assert len(closes) == 1 and closes[0]["params"]["update"]["status"] == "completed"


def test_a_malformed_frame_cannot_answer_an_outstanding_approval() -> None:
    """`_deliver` keyed on "has an id and no method", so any junk carrying an
    outstanding id became that approval's answer -- and an unreadable answer
    denies, so a stray frame could silently refuse a command."""
    server = ACPServer(stdin=io.BytesIO(), stdout=io.BytesIO())
    answered: list[dict[str, Any]] = []
    asking = threading.Thread(
        target=lambda: answered.append(
            server.request("session/request_permission", {}, timeout_s=3)
        ),
        daemon=True,
    )
    asking.start()
    for _ in range(100):
        if server._pending:  # pyright: ignore[reportPrivateUsage]
            break
        threading.Event().wait(0.01)
    req_id = next(iter(server._pending))  # pyright: ignore[reportPrivateUsage]
    assert server._deliver(req_id, {"id": req_id}) is False  # pyright: ignore[reportPrivateUsage]
    assert server._pending, "the slot was consumed by a non-response"  # pyright: ignore[reportPrivateUsage]
    server.abandon_pending()
    asking.join(timeout=5.0)


def test_the_approval_dialog_is_scrubbed_too() -> None:
    """The one surface an operator MUST read before granting a command, and it
    never went through the scrub: the prompt embeds the model's own argv, and
    a `UserQuestion`'s option strings are model-written outright."""
    sent: list[dict[str, Any]] = []
    bridge = _bridge({"outcome": {"outcome": "cancelled"}})

    def _record(_method: str, params: dict[str, Any], **_kw: object) -> dict[str, Any]:
        sent.append(params)
        return {}

    bridge.server.request = _record  # pyright: ignore[reportAttributeAccessIssue]
    session = session_mod.Session(acp_id="s", cwd=Path("/x"))
    bridge.ask(
        session,
        Announced(turn=1),
        "Allow run_command: \x1b]0;PWNED\x07ls",
        ("allow\x1b[2J", "deny"),
        True,
        None,
    )
    wire = json.dumps(sent[0])
    assert "\\u001b" not in wire and "\\u0007" not in wire, wire


def test_the_unsandboxed_gate_is_never_offered_as_remember_me() -> None:
    """docs/security.md documents it as a ONE-TIME gate. ACP's `allow_always`
    is exactly the button that would let one click silence it forever."""
    from agent6.config import Config

    front, asked = _acp_front(reply="allow once")
    dangerous = Config.model_validate({"sandbox": {"isolation": "none", "run_commands": "yes"}})
    assert front.confirm_unconfined_autorun("none", dangerous) is True
    assert asked[-1][2] is False, "the sandbox-off gate must not be a standing approval"


def test_a_cwd_that_does_not_exist_is_refused_by_name(tmp_path: Path) -> None:
    """A stale workspace path is the ordinary editor mistake. Asking git first
    meant `subprocess` could not chdir into it, and the FileNotFoundError
    surfaced as `{"code": -32603, "message": "FileNotFoundError"}` -- an
    internal error where a named refusal belongs."""
    wire = _Wire()
    try:
        wire.send(id=1, method="initialize", params={"clientCapabilities": {}})
        wire.recv()
        wire.send(id=2, method="session/new", params={"cwd": str(tmp_path / "gone")})
        error = wire.recv()["error"]
        assert error["code"] != -32603, error
        assert "not a directory" in error["message"]
    finally:
        wire.close()


def test_a_second_prompt_resumes_the_same_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An ACP session is one conversation: the first prompt mints a run id and
    starts a run; the next prompt resumes that run with its text as the steer
    seed, and only a session with no snapshot starts fresh."""

    calls: list[tuple[str, str, str]] = []
    monkeypatch.chdir(tmp_path)

    def _state_dir(_cwd: Path) -> Path:
        return tmp_path / "state"

    def _minted(*_a: object, **_k: object) -> str:
        return "run-AAAA11"

    monkeypatch.setattr(runner, "state_dir", _state_dir)
    monkeypatch.setattr(runner, "unused_session_id", _minted)
    monkeypatch.setattr(runner, "load_session_config", _loaded)

    def _run_task(_config: object, text: str, **kw: Any) -> int:
        calls.append(("run", str(kw["session_id"]), text))
        return 0

    def _resume_task(_config_path: object, session_id: str, **kw: Any) -> int:
        calls.append(("resume", session_id, str(kw["steer"])))
        return 0

    monkeypatch.setattr(runner, "run_task", _run_task)
    monkeypatch.setattr(runner, "resume_task", _resume_task)
    bridge = RunBridge(server=ACPServer(stdin=io.BytesIO(), stdout=io.BytesIO()))
    session = session_mod.Session(acp_id="s", cwd=tmp_path)

    assert bridge.run(session, "first task") == "end_turn"
    assert calls == [("run", "run-AAAA11", "first task")]

    # The run left a snapshot: the next prompt continues it, same id.
    layout = session.layout(tmp_path / "state")
    layout.session_dir.mkdir(parents=True, exist_ok=True)
    (layout.session_dir / "loop_state.json").write_text("{}", encoding="utf-8")
    assert bridge.run(session, "and now this") == "end_turn"
    assert calls[1] == ("resume", "run-AAAA11", "and now this")


def test_a_fault_on_a_resumed_turn_still_reaches_the_editor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The bare-refusal guard read the journal's `finished`, which the
    previous turn's session.end leaves True on every resumed turn, so a fault
    in a resume's preflight ended the turn as a bare refusal again. The guard
    reads whether the tail told the editor THIS turn's ending."""
    monkeypatch.chdir(tmp_path)

    def _state_dir(_cwd: Path) -> Path:
        return tmp_path / "state"

    def _minted(*_a: object, **_k: object) -> str:
        return "run-AAAA11"

    def _ended_run(_config: object, text: str, **kw: Any) -> int:
        layout = SessionLayout(tmp_path / "state", str(kw["session_id"]))
        layout.ensure()
        layout.logs_path.write_text(
            json.dumps({"type": "session.start", "mode": "run", "user_task": text})
            + "\n"
            + json.dumps({"type": "session.end", "reason": "finish_session", "all_passed": True})
            + "\n",
            encoding="utf-8",
        )
        (layout.session_dir / "loop_state.json").write_text("{}", encoding="utf-8")
        return 0

    def _dies_in_preflight(_config_path: object, session_id: str, **kw: Any) -> int:
        raise RuntimeError("the resume preflight exploded")

    monkeypatch.setattr(runner, "state_dir", _state_dir)
    monkeypatch.setattr(runner, "unused_session_id", _minted)
    monkeypatch.setattr(runner, "load_session_config", _loaded)
    monkeypatch.setattr(runner, "run_task", _ended_run)
    monkeypatch.setattr(runner, "resume_task", _dies_in_preflight)
    out = io.BytesIO()
    bridge = RunBridge(server=ACPServer(stdin=io.BytesIO(), stdout=out))
    session = session_mod.Session(acp_id="s", cwd=tmp_path)
    assert bridge.run(session, "first task") == "end_turn"
    assert bridge.run(session, "and now this") == "refusal"
    said = out.getvalue().decode()
    assert "the run failed: the resume preflight exploded" in said, said


def _journal_types(layout: SessionLayout) -> list[str]:
    return [
        json.loads(line)["type"]
        for line in layout.logs_path.read_text(encoding="utf-8").splitlines()
    ]


def test_a_gated_call_reads_pending_on_the_wire(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A call blocked on an approval: the editor is asked under the CALL's own
    id, after the call was announced, and the call reads pending until the
    answer lands. The journaled prompt/answer pair is what lets the fold say
    so, on this wire and on every other surface."""
    from agent6.events import EventSink

    monkeypatch.chdir(tmp_path)
    layouts: list[SessionLayout] = []

    def _gated_run(*_a: object, **kw: Any) -> int:
        layout = SessionLayout(
            state_dir=runner.state_dir(tmp_path), session_id=str(kw["session_id"])
        )
        layouts.append(layout)
        layout.session_dir.mkdir(parents=True, exist_ok=True)
        events = EventSink(layout.logs_path)
        prompts = OperatorPrompts(  # built before the loop, as the leg does
            approver=kw["frontend"].build_approver(layout.session_dir),
            journal=events.emit,
            session_dir=layout.session_dir,
        )
        events.emit("session.start", session_id=layout.session_id, mode="run", user_task="t")
        events.emit("tool.call", name="run_command", args={"argv": ["ls"]}, call_id=1)
        approved = prompts.approve("Allow run_command: ls", scope="command", call_id=1)
        events.emit("tool.result", name="run_command", ok=approved, summary="ok", call_id=1)
        events.emit("session.end", reason="finish_session", iterations=1, all_passed=True)
        return 0

    monkeypatch.setattr(runner, "run_task", _gated_run)
    monkeypatch.setattr(runner, "load_session_config", _loaded)
    wire = _Wire()
    try:
        session_id = wire.new_session(_repo(tmp_path))
        wire.prompt(session_id, "do the thing")
        seen: list[tuple[str, str]] = []  # (toolCallId, status), in wire order
        announced: list[str] = []  # the `tool_call` announcements' ids
        asked: dict[str, Any] | None = None
        announced_before_ask: list[str] = []
        for _ in range(60):
            message = wire.recv()
            if message.get("method") == "session/request_permission":
                asked = message["params"]["toolCall"]
                announced_before_ask = list(announced)
                wire.send(
                    id=message["id"],
                    result={"outcome": {"outcome": "selected", "optionId": "0"}},
                )
            elif message.get("method") == "session/update":
                update = message["params"]["update"]
                if update["sessionUpdate"] == "tool_call":
                    announced.append(update["toolCallId"])
                if "toolCallId" in update:
                    seen.append((update["toolCallId"], update["status"]))
            elif "result" in message:
                assert message["result"]["stopReason"] == "end_turn"
                break
        assert asked is not None, "the editor was never asked"
        assert announced == [asked["toolCallId"]], (
            f"the request must name the one call it gates: {announced} vs {asked['toolCallId']}"
        )
        assert announced_before_ask, "the request reached the editor before the call it gates"
        assert [s for _, s in seen] == ["in_progress", "pending", "in_progress", "completed"]
        assert {i for i, _ in seen} == {announced[0]}, "an update addressed another call"
        types = _journal_types(layouts[0])
        assert "approval.prompt" in types and "approval.answer" in types
    finally:
        wire.close()


def test_a_request_waits_for_the_announcement_only_while_the_tail_reads() -> None:
    """The wait ends on the announcement, when the tail stops reading (it
    closes the register), or when the turn is cancelled; never on a clock."""
    import time

    announced = Announced(turn=1)
    abandoned = threading.Event()
    returned: list[float] = []

    def _wait() -> None:
        started = time.monotonic()
        announced.wait_for("run-x:1:1", abandoned=abandoned.is_set)
        returned.append(time.monotonic() - started)

    waiter = threading.Thread(target=_wait, daemon=True)
    waiter.start()
    waiter.join(timeout=2.5)
    assert waiter.is_alive(), f"the wait ended on its own after {returned}"
    announced.add("run-x:1:1")
    waiter.join(timeout=5.0)
    assert returned, "the announcement did not release the wait"

    closing = Announced(turn=1)
    closer = threading.Thread(
        target=lambda: closing.wait_for("run-x:1:2", abandoned=lambda: False), daemon=True
    )
    closer.start()
    closing.close()
    closer.join(timeout=5.0)
    assert not closer.is_alive(), "a closed register must release every wait"

    cancelled = Announced(turn=1)
    quitter = threading.Thread(
        target=lambda: cancelled.wait_for("run-x:1:3", abandoned=abandoned.is_set), daemon=True
    )
    quitter.start()
    abandoned.set()
    quitter.join(timeout=5.0)
    assert not quitter.is_alive(), "a cancelled turn must release the wait"


def test_a_tool_call_id_is_unique_across_a_sessions_turns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A later prompt resumes the same run under a fresh dispatcher, whose
    stamped call ids restart at 1: turn 2's first call overwrote turn 1's in
    an editor keyed on toolCallId."""
    from agent6.events import EventSink

    monkeypatch.chdir(tmp_path)

    def _layout(session_id: str) -> SessionLayout:
        return SessionLayout(state_dir=runner.state_dir(tmp_path), session_id=session_id)

    def _run_task(*_a: object, **kw: Any) -> int:
        layout = _layout(str(kw["session_id"]))
        layout.session_dir.mkdir(parents=True, exist_ok=True)
        events = EventSink(layout.logs_path)
        events.emit("session.start", session_id=layout.session_id, mode="run", user_task="t")
        events.emit("tool.call", name="run_command", args={"argv": ["ls"]}, call_id=1)
        events.emit("tool.result", name="run_command", ok=False, summary="no", call_id=1)
        events.emit("session.end", reason="finish_session", iterations=1, all_passed=True)
        (layout.session_dir / "loop_state.json").write_text("{}", encoding="utf-8")
        return 0

    def _resume_task(_config_path: object, session_id: str, **_kw: Any) -> int:
        events = EventSink(_layout(session_id).logs_path)
        events.emit("loop.resume.start", session_id=session_id, mode="run", iteration=2)
        events.emit("tool.call", name="run_command", args={"argv": ["ls"]}, call_id=1)
        events.emit("tool.result", name="run_command", ok=True, summary="ok", call_id=1)
        events.emit("session.end", reason="finish_session", iterations=2, all_passed=True)
        return 0

    monkeypatch.setattr(runner, "run_task", _run_task)
    monkeypatch.setattr(runner, "resume_task", _resume_task)
    monkeypatch.setattr(runner, "load_session_config", _loaded)
    wire = _Wire()
    try:
        session_id = wire.new_session(_repo(tmp_path))
        announced: list[str] = []
        for req_id in (3, 4):
            wire.prompt(session_id, "do the thing", req_id=req_id)
            for _ in range(60):
                message = wire.recv()
                if message.get("method") == "session/update":
                    update = message["params"]["update"]
                    if update["sessionUpdate"] == "tool_call":
                        announced.append(update["toolCallId"])
                elif message.get("id") == req_id:
                    break
        assert len(announced) == 2, announced
        assert announced[0] != announced[1], f"turn 2's first call reused {announced[0]}"
    finally:
        wire.close()


def test_a_late_tail_keeps_its_own_turn(tmp_path: Path) -> None:
    """A tail that outlives its turn's join reads the turn it was started for;
    the next turn's increment must not restamp its late items."""
    import time

    from agent6.events import EventSink

    sent: list[dict[str, Any]] = []
    server = ACPServer(stdin=io.BytesIO(), stdout=io.BytesIO())
    server.notify_raw = sent.append  # pyright: ignore[reportAttributeAccessIssue]
    bridge = RunBridge(server=server)
    session = session_mod.Session(acp_id="s", cwd=tmp_path, session_id="run-x", turn=1)
    log = tmp_path / "logs.jsonl"
    log.write_text("", encoding="utf-8")
    done = threading.Event()
    tail = threading.Thread(
        target=bridge._stream,  # pyright: ignore[reportPrivateUsage]
        args=(session, log, done.is_set, False, Announced(turn=1)),
        daemon=True,
    )
    tail.start()
    session.turn = 2  # the next turn began while this tail still reads
    EventSink(log).emit("tool.call", name="run_command", args={"argv": ["ls"]}, call_id=1)
    for _ in range(100):
        if sent:
            break
        time.sleep(0.05)
    done.set()
    tail.join(timeout=5.0)
    assert sent and sent[0]["params"]["update"]["toolCallId"] == "run-x:1:1", sent


def test_the_runs_notices_reach_the_editor(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The stash notice and the where-are-my-changes footer are facts the CLI
    prints at the end of a run. Over ACP they went to stderr only once a
    journal existed, so stashes accumulated invisibly."""
    from agent6.events import EventSink

    monkeypatch.chdir(tmp_path)

    def _noticing_run(*_a: object, **kw: Any) -> int:
        layout = SessionLayout(
            state_dir=runner.state_dir(tmp_path), session_id=str(kw["session_id"])
        )
        layout.session_dir.mkdir(parents=True, exist_ok=True)
        events = EventSink(layout.logs_path)
        events.emit("session.start", session_id=layout.session_id, mode="run", user_task="t")
        events.emit("session.end", reason="finish_session", iterations=1, all_passed=True)
        reporter = kw["reporter"]
        assert isinstance(reporter, Reporter)
        reporter.out("\nchanges are on agent6/run-x")
        reporter.out("  merge with:  agent6 sessions merge run-x")
        reporter.note("pre-run changes left stashed; restore with: git stash apply abc123")
        return 0

    monkeypatch.setattr(runner, "run_task", _noticing_run)
    monkeypatch.setattr(runner, "load_session_config", _loaded)
    wire = _Wire()
    try:
        session_id = wire.new_session(_repo(tmp_path))
        wire.prompt(session_id, "do the thing")
        said: list[str] = []
        for _ in range(60):
            message = wire.recv()
            if message.get("method") == "session/update":
                update = message["params"]["update"]
                if update["sessionUpdate"] == "agent_message_chunk":
                    said.append(update["content"]["text"])
            elif "result" in message:
                assert message["result"]["stopReason"] == "end_turn"
                break
        text = "\n".join(said)
        assert "git stash apply abc123" in text, text
        assert "merge with:  agent6 sessions merge run-x" in text, text
    finally:
        wire.close()


def test_the_editor_gets_each_ending_fact_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The fold's done item carries the summary and the cost. The lifecycle's
    cost receipt and the ending go to stderr (the editor's agent log), so
    the editor reads each fact once and the log keeps its headline."""
    from agent6.events import EventSink

    monkeypatch.chdir(tmp_path)

    def _ending_run(*_a: object, **kw: Any) -> int:
        layout = SessionLayout(
            state_dir=runner.state_dir(tmp_path), session_id=str(kw["session_id"])
        )
        layout.session_dir.mkdir(parents=True, exist_ok=True)
        events = EventSink(layout.logs_path)
        events.emit("session.start", session_id=layout.session_id, mode="run", user_task="t")
        events.emit("budget.update", usd_total=0.0028)
        events.emit("session.end", reason="finish_session", iterations=1, all_passed=True)
        reporter = kw["reporter"]
        assert isinstance(reporter, Reporter)
        reporter.cost("Token + cost summary:\n  TOTAL: in=839 out=253 cost=$0.0028 of $0.0500")
        reporter.out("\nchanges are on agent6/run-x")
        return 0

    monkeypatch.setattr(runner, "run_task", _ending_run)
    monkeypatch.setattr(runner, "load_session_config", _loaded)
    wire = _Wire()
    try:
        session_id = wire.new_session(_repo(tmp_path))
        wire.prompt(session_id, "do the thing")
        said: list[str] = []
        for _ in range(60):
            message = wire.recv()
            if message.get("method") == "session/update":
                update = message["params"]["update"]
                if update["sessionUpdate"] == "agent_message_chunk":
                    said.append(update["content"]["text"])
            elif "result" in message:
                break
    finally:
        wire.close()
    assert sum("$0.0028" in text for text in said) == 1, said
    assert not any("Token + cost summary" in text for text in said), said
    assert any("changes are on agent6/run-x" in text for text in said), said
    err = capsys.readouterr().err
    assert "Token + cost summary" in err and "Session passed" in err, err


def _two_sessions_one_blocked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[_Wire, str, str, threading.Event]:
    """Two sessions on one connection, the first's turn holding the run lock."""
    monkeypatch.setattr(session_mod, "request_stop", _ignore)
    monkeypatch.chdir(tmp_path)
    first_started, release = threading.Event(), threading.Event()

    def _blocking_run(*_a: object, **_kw: object) -> int:
        first_started.set()
        release.wait(timeout=10.0)
        return 0

    monkeypatch.setattr(runner, "run_task", _blocking_run)
    monkeypatch.setattr(runner, "load_session_config", _loaded)
    wire = _Wire()
    first = wire.new_session(_repo(tmp_path))
    wire.send(id=9, method="session/new", params={"cwd": str(tmp_path)})
    second = str(wire.recv()["result"]["sessionId"])
    wire.prompt(first, "the long one", req_id=3)
    assert first_started.wait(timeout=5.0)
    wire.prompt(second, "the queued one", req_id=4)
    return wire, first, second, release


def test_a_queued_prompt_says_what_it_waits_for(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Runs are serialised on the connection, so a second session's prompt can
    wait for minutes. It waited in silence: the editor saw a turn that had
    started and said nothing."""
    wire, first, second, release = _two_sessions_one_blocked(tmp_path, monkeypatch)
    try:
        said = wire.until("session/update", timeout=3.0)
        assert said["params"]["sessionId"] == second
        text = said["params"]["update"]["content"]["text"]
        assert "waiting" in text and first in text, text
    finally:
        release.set()
        wire.close()


def test_a_cancel_of_a_queued_prompt_answers_at_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A queued turn blocked on the run lock noticed its cancel only when the
    other turn ended, so the editor's stop button answered minutes later."""
    wire, _first, _second, release = _two_sessions_one_blocked(tmp_path, monkeypatch)
    try:
        wire.send(method="session/cancel", params={"sessionId": _second})
        reply = wire.until("", timeout=3.0)
        while reply.get("id") != 4:
            reply = wire.until("", timeout=3.0)
        assert reply["result"]["stopReason"] == "cancelled"
    finally:
        release.set()
        wire.close()


def test_a_request_names_the_call_it_gates_not_the_newest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two calls in flight (a concurrent seat's read beside the gated
    command): the request names the call the journaled prompt carries, and
    only that call reads pending on the wire."""
    from agent6.events import EventSink

    monkeypatch.chdir(tmp_path)

    def _gated_run(*_a: object, **kw: Any) -> int:
        layout = SessionLayout(
            state_dir=runner.state_dir(tmp_path), session_id=str(kw["session_id"])
        )
        layout.session_dir.mkdir(parents=True, exist_ok=True)
        events = EventSink(layout.logs_path)
        prompts = OperatorPrompts(
            approver=kw["frontend"].build_approver(layout.session_dir),
            journal=events.emit,
            session_dir=layout.session_dir,
        )
        events.emit("session.start", session_id=layout.session_id, mode="run", user_task="t")
        events.emit("tool.call", name="run_command", args={"argv": ["ls"]}, call_id=1)
        events.emit("tool.call", name="read_file", args={"path": "x"}, call_id=2)
        approved = prompts.approve("Allow run_command: ls", scope="command", call_id=1)
        events.emit("tool.result", name="read_file", ok=True, summary="ok", call_id=2)
        events.emit("tool.result", name="run_command", ok=approved, summary="ok", call_id=1)
        events.emit("session.end", reason="finish_session", iterations=1, all_passed=True)
        return 0

    monkeypatch.setattr(runner, "run_task", _gated_run)
    monkeypatch.setattr(runner, "load_session_config", _loaded)
    wire = _Wire()
    try:
        session_id = wire.new_session(_repo(tmp_path))
        wire.prompt(session_id, "do the thing")
        announced: list[str] = []
        pending: list[str] = []
        asked: str | None = None
        for _ in range(80):
            message = wire.recv()
            if message.get("method") == "session/request_permission":
                asked = message["params"]["toolCall"]["toolCallId"]
                wire.send(
                    id=message["id"],
                    result={"outcome": {"outcome": "selected", "optionId": "0"}},
                )
            elif message.get("method") == "session/update":
                update = message["params"]["update"]
                if update["sessionUpdate"] == "tool_call":
                    announced.append(update["toolCallId"])
                elif update.get("status") == "pending":
                    pending.append(update["toolCallId"])
            elif "result" in message:
                break
        assert len(announced) == 2, announced
        assert asked == announced[0], "the request names the gated call, not the newest in flight"
        assert pending == [announced[0]], "only the gated call waits"
    finally:
        wire.close()


def test_an_answer_file_ends_the_editors_pending_request(tmp_path: Path) -> None:
    """An editor over ACP is asked through a request that blocks until it
    replies; `agent6 answer` and the web write the session's answer file,
    which nothing in that wait read, so they reported "answered" to a run
    that kept waiting on the editor."""
    from agent6.app.frontend import FrontendCapabilities
    from agent6.sessions.ipc import write_answer, write_question_answers
    from agent6.tools.operator_prompts import ApprovalRequest, QuestionRequest
    from agent6.tools.schema import UserQuestion
    from agent6.ui.acp.frontend import acp_frontend

    def _editor_never_replies(
        prompt: str,
        options: tuple[str, ...],
        standing: bool | None,
        call_id: int | None,
        until: Callable[[], bool] | None = None,
    ) -> str | None:
        assert until is not None
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            if until():
                return None
            time.sleep(0.05)
        pytest.fail("the wait never looked at the answer file")

    front = acp_frontend(
        ask=_editor_never_replies,
        capabilities=FrontendCapabilities(can_ask=True),
        agent6_exe=lambda: "agent6",
        spawn_detached_resume=lambda _cwd, _rid, _flags: "",
    )
    write_answer(tmp_path, "approval-1", "yes")
    approver = front.build_approver(tmp_path)
    verdict = approver(
        ApprovalRequest(id="approval-1", prompt="Allow run_command: ls", scope="command", call_id=1)
    )
    assert (verdict.approved, verdict.source) == (True, "frontend")

    write_question_answers(tmp_path, "question-1", ["9090"])
    questioner = front.build_questioner(tmp_path)
    answer = questioner(
        QuestionRequest(id="question-1", questions=(UserQuestion(question="port?"),), call_id=2)
    )
    assert (answer.answers, answer.source) == (("9090",), "frontend")


def test_the_lifecycles_lines_take_their_place_in_journal_order(tmp_path: Path) -> None:
    """The lifecycle speaks from the run thread while the tail projects the
    journal a poll behind: an ending line ("no changes were committed") reached
    the editor before the turn's last tool calls, and a stop notice before the
    work it stopped."""
    import time

    from agent6.events import EventSink
    from agent6.ui.acp.runner import ProseOrder

    sent: list[dict[str, Any]] = []
    server = ACPServer(stdin=io.BytesIO(), stdout=io.BytesIO())
    server.notify_raw = sent.append  # pyright: ignore[reportAttributeAccessIssue]
    bridge = RunBridge(server=server)
    session = session_mod.Session(acp_id="s", cwd=tmp_path, session_id="run-x", turn=1)
    log = tmp_path / "logs.jsonl"
    events = EventSink(log)
    events.emit("tool.call", name="run_command", args={"argv": ["ls"]}, call_id=1)
    events.emit("tool.call", name="run_command", args={"argv": ["true"]}, call_id=2)
    order = ProseOrder(server, "s", log)
    said: list[str] = []
    reporter = forwarding_reporter(server, "s", said, order=order)
    reporter.out("no changes were committed")  # said after both calls, before the tail read them
    assert sent == []
    done = threading.Event()
    tail = threading.Thread(
        target=bridge._stream,  # pyright: ignore[reportPrivateUsage]
        args=(session, log, done.is_set, False, Announced(turn=1), order),
        daemon=True,
    )
    tail.start()
    for _ in range(100):
        if len(sent) >= 3:
            break
        time.sleep(0.05)
    done.set()
    tail.join(timeout=5.0)
    kinds = [s["params"]["update"].get("sessionUpdate") for s in sent]
    assert kinds[-1] == "agent_message_chunk" and "no changes were committed" in json.dumps(
        sent[-1]
    )
    assert all(k == "tool_call" for k in kinds[:2]), kinds


def test_a_turn_that_cannot_choose_its_run_says_why(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The id is chosen before the run's try, and a global config that cannot
    be read raises there: the turn ended with a bare "refusal" and not a word,
    while the same failure a step later was reported."""
    monkeypatch.chdir(tmp_path)

    def _broken(*_a: object, **_kw: object) -> object:
        raise ConfigError("Config file cannot be read (config.toml)")

    monkeypatch.setattr(runner, "state_dir", _broken)
    wire = _Wire()
    try:
        session_id = wire.new_session(_repo(tmp_path))
        wire.prompt(session_id, "do the thing")
        said = wire.until("session/update")
        text = said["params"]["update"]["content"]["text"]
        assert "could not start" in text and "config.toml" in text, text
        assert wire.until("")["result"]["stopReason"] == "refusal"
    finally:
        wire.close()


def test_an_internal_error_keeps_its_reason(monkeypatch: pytest.MonkeyPatch) -> None:
    """A handler bug answered the editor with the exception's class name alone
    (`KeyError`) and wrote nothing to stderr."""
    from agent6.ui.acp.session import Sessions

    def _boom(self: object, params: object) -> object:
        raise KeyError("no such row")

    monkeypatch.setattr(Sessions, "get", _boom)
    wire = _Wire()
    try:
        wire.send(id=1, method="initialize", params={"clientCapabilities": {}})
        wire.recv()
        wire.send(id=2, method="session/cancel", params={"sessionId": "s"})
        error = wire.recv()["error"]
        assert error["code"] == -32603
        assert "KeyError" in error["message"] and "no such row" in error["message"]
    finally:
        wire.close()
