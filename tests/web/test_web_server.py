# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Integration tests for the `agent6 web` server.

Starts the stdlib server on an ephemeral loopback port and drives it with
`http.client`, asserting the JSON endpoints emit the same wire form as
`agent6 attach --json` and that SSE streams a folded snapshot. No browser."""

from __future__ import annotations

import json
import os
import socket
import threading
import time
from collections.abc import Callable, Iterator
from http.client import HTTPConnection
from pathlib import Path
from typing import Any, cast

import pytest

from agent6.config.layer import resolved_state_dir
from agent6.sessions.ipc import write_worker_pid
from agent6.ui.cli import main
from agent6.ui.web.server import (
    WebServer,
    _create_web_server,  # pyright: ignore[reportPrivateUsage]
)

TINY = """
machine = "tiny"
version = 1
initial = "route"

[budget]
max_transitions = 10

[vars.code]
n = { type = "int", default = 0 }

[states.route]
kind = "branch"
when = [
  { if = "n == 0", goto = "done" },
  { else = true, goto = "done" },
]

[states.done]
kind = "terminal"
status = "ok"
reason = "routed"
"""


def _make_run(cwd: Path, session_id: str, events: list[dict[str, object]]) -> None:
    runs = resolved_state_dir(cwd) / "sessions" / "runs" / session_id
    runs.mkdir(parents=True)
    body = "".join(json.dumps(e) + "\n" for e in events)
    (runs / "logs.jsonl").write_text(body, encoding="utf-8")


@pytest.fixture
def server(tmp_path: Path) -> Iterator[tuple[WebServer, int]]:
    """A WebServer bound to an ephemeral loopback port, serving from tmp_path."""
    srv = WebServer(("127.0.0.1", 0), tmp_path, "")
    port = srv.server_address[1]
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    try:
        yield srv, port
    finally:
        srv.shutdown()
        srv.server_close()


def _get(port: int, path: str) -> tuple[int, bytes, str]:
    conn = HTTPConnection("127.0.0.1", port, timeout=10)
    try:
        conn.request("GET", path)
        resp = conn.getresponse()
        return resp.status, resp.read(), resp.getheader("Content-Type", "")
    finally:
        conn.close()


def _post(port: int, path: str, body: dict[str, object]) -> tuple[int, dict[str, object]]:
    conn = HTTPConnection("127.0.0.1", port, timeout=10)
    try:
        payload = json.dumps(body).encode()
        conn.request("POST", path, payload, {"Content-Type": "application/json"})
        resp = conn.getresponse()
        return resp.status, json.loads(resp.read())
    finally:
        conn.close()


def _post_raw(
    port: int, path: str, body: bytes, headers: dict[str, str]
) -> tuple[int, dict[str, object]]:
    """POST with caller-controlled headers (for the CSRF checks)."""
    conn = HTTPConnection("127.0.0.1", port, timeout=10)
    try:
        conn.request("POST", path, body, headers)
        resp = conn.getresponse()
        return resp.status, json.loads(resp.read())
    finally:
        conn.close()


def test_page_served(server: tuple[WebServer, int]) -> None:
    _srv, port = server
    status, body, ctype = _get(port, "/")
    assert status == 200
    assert "text/html" in ctype
    assert b"<title>agent6</title>" in body


@pytest.mark.parametrize("host", ["::1", "[::1]"])
def test_ipv6_loopback_bind_uses_ipv6_socket(tmp_path: Path, host: str) -> None:
    srv = _create_web_server(host, 0, tmp_path, "")  # pyright: ignore[reportPrivateUsage]
    try:
        assert srv.address_family == socket.AF_INET6
    finally:
        srv.server_close()


def test_explicit_config_reaches_the_server(tmp_path: Path) -> None:
    """`agent6 --config F web` threads F to the server object every route
    reads (`self.config_path`); the constructor used to drop it, so the whole
    browser surface ran on the default layers while binding the configured
    port."""
    cfg = tmp_path / "f.toml"
    srv = _create_web_server("127.0.0.1", 0, tmp_path, "", cfg)  # pyright: ignore[reportPrivateUsage]
    try:
        assert srv.config_path == cfg
    finally:
        srv.server_close()


def test_run_snapshot_matches_watch_json(
    server: tuple[WebServer, int],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _srv, port = server
    _make_run(
        tmp_path,
        "willing-glen-001",
        [
            {"type": "session.start", "user_task": "demo"},
            {"type": "tool.call", "name": "grep", "args": {"q": "x"}},
            {"type": "tool.result", "name": "grep", "ok": True, "summary": "1 hit"},
        ],
    )
    # The web GET must equal `agent6 attach <id> --json` byte-for-byte in content.
    status, body, ctype = _get(port, "/api/session/willing-glen-001")
    assert status == 200
    assert "application/json" in ctype
    from_web = json.loads(body)

    monkeypatch.chdir(tmp_path)
    assert main(["attach", "willing-glen-001", "--json"]) == 0
    from_cli = json.loads(capsys.readouterr().out)
    assert from_web == from_cli
    assert from_web["tool_calls"][0]["name"] == "grep"
    # Even for a log whose session.start predates the session_id field, both surfaces
    # stamp the authoritative id from the dir (never an empty session_id).
    assert from_web["session_id"] == "willing-glen-001"


def test_hub_lists_runs(server: tuple[WebServer, int], tmp_path: Path) -> None:
    _srv, port = server
    _make_run(tmp_path, "run-a", [{"type": "session.start", "mode": "run", "user_task": "task a"}])
    _make_run(
        tmp_path,
        "run-b",
        [
            {"type": "session.start", "mode": "run", "user_task": "task b"},
            {"type": "session.end", "all_passed": True},
        ],
    )
    status, body, _ = _get(port, "/api/hub")
    assert status == 200
    hub = json.loads(body)
    ids = {r["id"] for r in hub["sessions"]}
    assert ids == {"run-a", "run-b"}
    by_id = {r["id"]: r for r in hub["sessions"]}
    assert by_id["run-b"]["status"] == "passed"
    assert by_id["run-b"]["task"] == "task b"


def test_machine_snapshot_matches_watch_json(
    server: tuple[WebServer, int],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _srv, port = server
    monkeypatch.chdir(tmp_path)
    (tmp_path / "tiny.asm.toml").write_text(TINY, encoding="utf-8")
    assert main(["machine", "run", str(tmp_path / "tiny.asm.toml")]) == 0
    capsys.readouterr()

    status, body, _ = _get(port, "/api/machine/tiny")
    assert status == 200
    from_web = json.loads(body)

    assert main(["attach", "tiny", "--json"]) == 0
    from_cli = json.loads(capsys.readouterr().out)
    assert from_web == from_cli
    assert from_web["machine"] == "tiny"
    assert from_web["ended"]["status"] == "ok"


def test_unknown_run_is_404(server: tuple[WebServer, int]) -> None:
    _srv, port = server
    status, body, _ = _get(port, "/api/session/nope")
    assert status == 404
    assert "no session" in json.loads(body)["error"]


def test_meta_resolves_the_target_kind(tmp_path: Path) -> None:
    # `agent6 web <target>` deep-links on load; the page asks /api/meta what
    # kind of view the target names.
    runs = resolved_state_dir(tmp_path) / "sessions" / "runs" / "run-t"
    runs.mkdir(parents=True)
    (runs / "logs.jsonl").write_text('{"type": "session.start"}\n', encoding="utf-8")
    srv = WebServer(("127.0.0.1", 0), tmp_path, "run-t")
    port = srv.server_address[1]
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    try:
        status, body, _ = _get(port, "/api/meta")
        assert status == 200
        meta = json.loads(body)
        assert meta["target"] == "run-t"
        assert meta["target_kind"] == "session"
    finally:
        srv.shutdown()
        srv.server_close()


def test_resume_spawns_a_detached_resume_with_the_follow_up(
    server: tuple[WebServer, int], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from agent6.ui.web import actions

    _srv, port = server
    _make_run(tmp_path, "run-r", [{"type": "session.start"}, {"type": "session.end"}])
    calls: list[tuple[Path, str, str, str]] = []

    def fake_resume(
        cwd: Path,
        session_id: str,
        *,
        steer: str = "",
        preset: str = "",
        config_path: object = None,
    ) -> str:
        calls.append((cwd, session_id, steer, preset))
        return ""

    monkeypatch.setattr(actions, "spawn_detached_resume", fake_resume)
    status, data = _post(
        port, "/api/session/run-r/resume", {"text": "also fix the docs", "preset": "quick"}
    )
    assert status == 200 and data["ok"] is True
    assert calls == [(tmp_path, "run-r", "also fix the docs", "quick")]


def test_resume_refused_while_the_worker_is_alive(
    server: tuple[WebServer, int], tmp_path: Path
) -> None:
    import os

    _srv, port = server
    _make_run(tmp_path, "run-l", [{"type": "session.start"}])
    runs = resolved_state_dir(tmp_path) / "sessions" / "runs" / "run-l"
    (runs / "worker.pid").write_text(str(os.getpid()), encoding="utf-8")
    status, data = _post(port, "/api/session/run-l/resume", {"text": ""})
    assert status == 422
    assert "still live" in str(data["error"])


def test_stop_step_and_compact_drop_markers_on_a_live_run(
    server: tuple[WebServer, int], tmp_path: Path
) -> None:
    import os

    _srv, port = server
    _make_run(tmp_path, "run-m", [{"type": "session.start"}])
    runs = resolved_state_dir(tmp_path) / "sessions" / "runs" / "run-m"
    (runs / "worker.pid").write_text(str(os.getpid()), encoding="utf-8")
    status, data = _post(port, "/api/session/run-m/stop_step", {})
    assert status == 200 and data["ok"] is True
    assert (runs / "stop.request").exists()
    status, data = _post(port, "/api/session/run-m/compact", {})
    assert status == 200 and data["ok"] is True
    assert (runs / "compact.request").exists()


def test_stop_step_refused_on_a_dead_run(server: tuple[WebServer, int], tmp_path: Path) -> None:
    _srv, port = server
    _make_run(tmp_path, "run-d", [{"type": "session.start"}, {"type": "session.end"}])
    status, data = _post(port, "/api/session/run-d/stop_step", {})
    assert status == 422
    assert "not live" in str(data["error"])


def test_run_conversation_endpoint(server: tuple[WebServer, int], tmp_path: Path) -> None:
    _srv, port = server
    _make_run(
        tmp_path,
        "run-c",
        [
            {"type": "session.start", "mode": "run", "user_task": "task c"},
            {"type": "tool.call", "name": "read_file", "args": {"path": "a.py"}},
            {"type": "tool.result", "name": "read_file", "ok": True, "summary": "12 bytes"},
            {"type": "session.end", "all_passed": True, "reason": "finish_session"},
        ],
    )
    status, body, _ = _get(port, "/api/session/run-c/conversation")
    assert status == 200
    payload = json.loads(body)
    assert payload["session_id"] == "run-c"
    kinds = [it["kind"] for it in payload["items"]]
    assert kinds == ["tool", "done"]
    tool = payload["items"][0]
    flat = "".join(text for line in tool["lines"] for text, _style in line)
    assert "read_file" in flat and "12 bytes" in flat


def test_config_endpoint(server: tuple[WebServer, int]) -> None:
    _srv, port = server
    status, body, _ = _get(port, "/api/config")
    assert status == 200
    cfg = json.loads(body)
    # A per-leaf view keyed by dotted key, each carrying provenance.
    assert any(k.startswith("sandbox.") for k in cfg)
    sample = next(iter(cfg.values()))
    assert {"value", "effective", "default", "source", "modified"} <= set(sample)


def test_approve_writes_answer_file(server: tuple[WebServer, int], tmp_path: Path) -> None:
    _srv, port = server
    session_dir = resolved_state_dir(tmp_path) / "sessions" / "runs" / "appr-run"
    session_dir.mkdir(parents=True)
    (session_dir / "logs.jsonl").write_text("", encoding="utf-8")
    write_worker_pid(session_dir, os.getpid())  # a prompt is answerable only while live
    status, body = _post(port, "/api/session/appr-run/approve", {"id": "p1", "answer": "yes"})
    assert status == 200
    assert body["ok"] is True
    assert (session_dir / "approvals" / "p1.answer").read_text(encoding="utf-8") == "yes"


def test_answer_writes_question_file(server: tuple[WebServer, int], tmp_path: Path) -> None:
    _srv, port = server
    session_dir = resolved_state_dir(tmp_path) / "sessions" / "runs" / "q-run"
    session_dir.mkdir(parents=True)
    (session_dir / "logs.jsonl").write_text("", encoding="utf-8")
    write_worker_pid(session_dir, os.getpid())  # a prompt is answerable only while live
    status, body = _post(port, "/api/session/q-run/answer", {"id": "q1", "answers": ["option B"]})
    assert status == 200 and body["ok"] is True
    assert (session_dir / "questions" / "q1.answer").read_text(encoding="utf-8") == json.dumps(
        ["option B"]
    )


def test_steer_writes_answer_and_request(server: tuple[WebServer, int], tmp_path: Path) -> None:
    _srv, port = server
    session_dir = resolved_state_dir(tmp_path) / "sessions" / "runs" / "steer-run"
    session_dir.mkdir(parents=True)
    (session_dir / "logs.jsonl").write_text("", encoding="utf-8")
    (session_dir / "worker.pid").write_text(str(os.getpid()), encoding="utf-8")
    status, body = _post(port, "/api/session/steer-run/steer", {"text": "focus on tests"})
    assert status == 200 and body["ok"] is True
    assert (session_dir / "steer.answer").read_text(encoding="utf-8") == "focus on tests"
    assert (session_dir / "steer.request").exists()


def test_steer_refused_on_a_dead_run(server: tuple[WebServer, int], tmp_path: Path) -> None:
    """A crashed run (no session.end, dead worker) folds as unfinished, so the
    composer offers steer; the action must refuse like stop_step/compact do
    instead of toasting "steer sent" for a marker nothing will ever read (the
    next resume even deletes it via clear_pending_answers)."""
    _srv, port = server
    _make_run(tmp_path, "run-sd", [{"type": "session.start"}, {"type": "session.end"}])
    status, data = _post(port, "/api/session/run-sd/steer", {"text": "abort"})
    assert status == 422
    assert "not live" in str(data["error"])
    session_dir = resolved_state_dir(tmp_path) / "sessions" / "runs" / "run-sd"
    assert not (session_dir / "steer.answer").exists()
    assert not (session_dir / "steer.request").exists()


def test_approve_id_traversal_is_contained(server: tuple[WebServer, int], tmp_path: Path) -> None:
    # A malicious answer id must not escape the run's approvals/ dir.
    _srv, port = server
    session_dir = resolved_state_dir(tmp_path) / "sessions" / "runs" / "trav-run"
    session_dir.mkdir(parents=True)
    (session_dir / "logs.jsonl").write_text("", encoding="utf-8")
    write_worker_pid(session_dir, os.getpid())  # a prompt is answerable only while live
    escape = tmp_path / "pwned.answer"
    status, _ = _post(
        port, "/api/session/trav-run/approve", {"id": "../../../../pwned", "answer": "yes"}
    )
    assert status != 200
    assert not escape.exists()
    # a normal id still works
    ok_status, ok_body = _post(port, "/api/session/trav-run/approve", {"id": "p1", "answer": "yes"})
    assert ok_status == 200 and ok_body["ok"] is True


def _make_machine_with_state(
    cwd: Path, name: str, seq_state: str, *, running: bool = False
) -> tuple[Path, Path]:
    """A machine instance dir + one per-state agent-log dir. Returns (instance,
    state). ``running`` records this test process as the machine's worker, so
    steer (which refuses a machine no state is executing under) is offered."""
    inst = resolved_state_dir(cwd) / "machines" / name
    inst.mkdir(parents=True)
    (inst / "machine.asm.toml").write_text(TINY, encoding="utf-8")
    (inst / "journal.jsonl").write_text("", encoding="utf-8")
    state = inst / "states" / seq_state
    state.mkdir(parents=True)
    (state / "logs.jsonl").write_text("", encoding="utf-8")
    if running:
        write_worker_pid(inst, os.getpid())
    return inst, state


def test_machine_poke_writes_signal(server: tuple[WebServer, int], tmp_path: Path) -> None:
    _srv, port = server
    inst, _ = _make_machine_with_state(tmp_path, "pokable", "0000-review")
    status, body = _post(port, "/api/machine/pokable/poke", {"message": "reload"})
    assert status == 200 and body["ok"] is True
    assert json.loads((inst / "signal").read_text(encoding="utf-8")) == "reload"


def test_machine_poke_json_data_payload(server: tuple[WebServer, int], tmp_path: Path) -> None:
    _srv, port = server
    inst, _ = _make_machine_with_state(tmp_path, "datapoke", "0000-review")
    status, body = _post(port, "/api/machine/datapoke/poke", {"data": {"cmd": "go", "n": 2}})
    assert status == 200 and body["ok"] is True
    assert json.loads((inst / "signal").read_text(encoding="utf-8")) == {"cmd": "go", "n": 2}


def test_machine_approve_and_steer_target_per_state_dir(
    server: tuple[WebServer, int], tmp_path: Path
) -> None:
    _srv, port = server
    _inst, state = _make_machine_with_state(tmp_path, "acter", "0001-work", running=True)
    _post(port, "/api/machine/acter/approve", {"id": "approval-1", "answer": "no"})
    assert (state / "approvals" / "approval-1.answer").read_text(encoding="utf-8") == "no"
    _post(port, "/api/machine/acter/steer", {"text": "focus"})
    assert (state / "steer.answer").read_text(encoding="utf-8") == "focus"
    assert (state / "steer.request").exists()


def test_machine_answer_id_traversal_is_contained(
    server: tuple[WebServer, int], tmp_path: Path
) -> None:
    _srv, port = server
    # running=True so the liveness gate passes and the id-component check is
    # the only thing that can refuse -- otherwise the traversal is never tested.
    _inst, _state = _make_machine_with_state(tmp_path, "travm", "0000-review", running=True)
    escape = tmp_path / "pwned.answer"
    status, _ = _post(port, "/api/machine/travm/answer", {"id": "../../pwned", "answers": ["x"]})
    assert status != 200
    assert not escape.exists()


def test_pwa_assets_served(server: tuple[WebServer, int]) -> None:
    _srv, port = server
    st, body, ctype = _get(port, "/manifest.webmanifest")
    assert st == 200 and "manifest" in ctype
    assert json.loads(body)["name"] == "agent6"
    assert _get(port, "/sw.js")[0] == 200
    assert _get(port, "/icon.svg")[0] == 200


def test_favicon_matches_the_docs_asset(server: tuple[WebServer, int]) -> None:
    # The tab favicon is docs/assets/favicon.svg embedded verbatim (the padded
    # /icon.svg tile is only for the PWA surfaces); this pins the copy in sync.
    _srv, port = server
    st, body, ctype = _get(port, "/favicon.svg")
    assert st == 200 and "svg" in ctype
    docs_svg = Path(__file__).parents[2] / "docs" / "assets" / "favicon.svg"
    assert body == docs_svg.read_bytes()


def test_run_id_traversal_is_404(server: tuple[WebServer, int]) -> None:
    _srv, port = server
    status, _body, _ = _get(port, "/api/session/..")
    assert status == 404


def test_extra_api_path_segments_are_404(server: tuple[WebServer, int], tmp_path: Path) -> None:
    _srv, port = server
    _make_run(tmp_path, "seg-run", [{"type": "session.start", "user_task": "x"}])
    machine = resolved_state_dir(tmp_path) / "machines" / "tiny"
    machine.mkdir(parents=True)
    (machine / "machine.asm.toml").write_text(TINY, encoding="utf-8")
    draft = resolved_state_dir(tmp_path) / "sessions" / "machines" / "drafty"
    draft.mkdir(parents=True)
    (draft / "logs.jsonl").write_text('{"type": "session.start"}\n', encoding="utf-8")

    assert _get(port, "/api/session/seg-run/events/extra")[0] == 404
    assert _get(port, "/api/machine/tiny/events/extra")[0] == 404
    assert _get(port, "/api/draft/drafty/events/extra")[0] == 404


def test_draft_snapshot_folds_the_draft_log(server: tuple[WebServer, int], tmp_path: Path) -> None:
    # A machine-create draft is watched through the run endpoints.
    _srv, port = server
    draft = resolved_state_dir(tmp_path) / "sessions" / "machines" / "brave-otter"
    draft.mkdir(parents=True)
    (draft / "logs.jsonl").write_text(
        '{"type": "session.start", "user_task": "author a fixer machine"}\n', encoding="utf-8"
    )
    status, body, _ = _get(port, "/api/draft/brave-otter")
    assert status == 200
    assert json.loads(body)["user_task"] == "author a fixer machine"
    # traversal rejected
    assert _get(port, "/api/draft/..")[0] == 404


def test_web_refuses_non_loopback_host_without_optin(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # `--host 0.0.0.0` must be refused before binding unless opted in.
    monkeypatch.chdir(tmp_path)
    assert main(["web", "--host", "0.0.0.0"]) == 2
    assert "refusing to bind non-loopback" in capsys.readouterr().err


def test_new_work_empty_task_rejected(server: tuple[WebServer, int]) -> None:
    _srv, port = server
    status, body = _post(port, "/api/new", {"mode": "run", "task": "   "})
    assert status == 422
    assert body["ok"] is False


def test_machine_run_rejects_unknown_file(server: tuple[WebServer, int]) -> None:
    _srv, port = server
    status, body = _post(port, "/api/machine/run", {"file": "/etc/passwd"})
    assert status == 422
    assert "unknown machine file" in str(body["error"])


def test_bad_post_body_is_400_with_the_field_named(server: tuple[WebServer, int]) -> None:
    """One human line per failed field, not the repr of pydantic's error list
    (`[{'type': 'missing', 'loc': ('task',), ...}]`) in the toast."""
    _srv, port = server
    status, body = _post(port, "/api/new", {"mode": "run"})
    assert (status, body["error"]) == (400, "task: field required")
    # extra="forbid": an unknown field fails validation loudly.
    status, body = _post(port, "/api/new", {"mode": "run", "task": "x", "bogus": 1})
    assert (status, body["error"]) == (400, "bogus: extra inputs are not permitted")


def test_a_post_on_an_unknown_session_or_machine_is_404_like_its_get(
    server: tuple[WebServer, int],
) -> None:
    """The verbs answered 422 (`no session 'x'`) where the GET of the same id
    answers 404: one status per fact."""
    _srv, port = server
    status, body = _post(port, "/api/session/nope/steer", {"text": "x"})
    assert (status, body["error"]) == (404, "no session 'nope'")
    status, body = _post(port, "/api/machine/nope/stop", {})
    assert (status, body["error"]) == (404, "no machine 'nope'")


def _read_until(
    resp: Any, cond: Callable[[dict[str, object]], bool], *, deadline_s: float = 10.0
) -> dict[str, object]:
    """Read SSE data frames until *cond*(snapshot) is true; return that
    snapshot. The stream does not close on a finished run (a resume keeps
    painting into it), so tests read to a condition, never to EOF."""
    buf = b""
    deadline = time.monotonic() + deadline_s
    while time.monotonic() < deadline:
        chunk = resp.read1(4096)
        if not chunk:
            break
        buf += chunk
        *complete, buf = buf.split(b"\n\n")
        for f in complete:
            if not f.startswith(b"data:"):
                continue
            snap = json.loads(f[len(b"data:") :].strip())
            if cond(snap):
                return snap
    raise AssertionError("no SSE frame matched before the deadline")


def test_sse_run_streams_snapshot(server: tuple[WebServer, int], tmp_path: Path) -> None:
    _srv, port = server
    _make_run(
        tmp_path,
        "stream-run",
        [
            {"type": "session.start", "user_task": "streamed"},
            {"type": "session.end", "all_passed": True},
        ],
    )
    conn = HTTPConnection("127.0.0.1", port, timeout=10)
    try:
        conn.request("GET", "/api/session/stream-run/events")
        resp = conn.getresponse()
        assert resp.status == 200
        assert "text/event-stream" in resp.getheader("Content-Type", "")
        snap = _read_until(resp, lambda s: s.get("finished") is True)
        assert snap["user_task"] == "streamed"
    finally:
        conn.close()


def test_sse_run_stream_survives_a_finish_and_follows_the_resumed_leg(
    server: tuple[WebServer, int], tmp_path: Path
) -> None:
    """The tailer opened with stop_when_finished=True and the client closed on
    `finished`, so a run resumed from ANOTHER surface left this page frozen on
    "stopped" while the hub said "running", indefinitely. The stream now stays
    open across a finish and paints the resumed leg (the TUI already did)."""
    _srv, port = server
    _make_run(
        tmp_path,
        "resume-run",
        [
            {"type": "session.start", "user_task": "leg one"},
            {"type": "session.end", "all_passed": True},
        ],
    )
    logs = resolved_state_dir(tmp_path) / "sessions" / "runs" / "resume-run" / "logs.jsonl"
    conn = HTTPConnection("127.0.0.1", port, timeout=10)
    try:
        conn.request("GET", "/api/session/resume-run/events")
        resp = conn.getresponse()
        _read_until(resp, lambda s: s.get("finished") is True)
        # A resume (from the CLI, say) appends to the same log.
        with logs.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({"type": "loop.resume.start", "ts": time.time()}) + "\n")
            fh.write(
                json.dumps({"type": "role.call", "role": "worker", "model": "m", "ts": time.time()})
                + "\n"
            )
        snap = _read_until(resp, lambda s: s.get("finished") is False)
        assert snap["user_task"] == "leg one"
        # The frame carries a server-computed idle age so the browser's
        # "working… Ns" needs no clock agreement (and replay reads its true age).
        age = snap["last_event_age_s"]
        assert isinstance(age, (int, float)) and 0 <= age < 60
    finally:
        conn.close()


def test_sse_run_frame_carries_the_compare_outcome(
    server: tuple[WebServer, int], tmp_path: Path
) -> None:
    """The run view paints from the SSE frame, not the one-shot snapshot; the
    frame must carry the manifest's compare block (branch facts + compare share
    one manifest_header helper so the two endpoints can never drift)."""
    _srv, port = server
    _make_run(tmp_path, "cmp-run", [{"type": "session.start", "user_task": "x"},
                                    {"type": "session.end", "all_passed": True}])  # fmt: skip
    session_dir = resolved_state_dir(tmp_path) / "sessions" / "runs" / "cmp-run"
    (session_dir / "manifest.json").write_text(
        json.dumps({"compare": {"rank": 1, "of": 2, "winner": True, "ranked_by": "judge"}}),
        encoding="utf-8",
    )
    conn = HTTPConnection("127.0.0.1", port, timeout=10)
    try:
        conn.request("GET", "/api/session/cmp-run/events")
        resp = conn.getresponse()
        snap = _read_until(resp, lambda s: s.get("finished") is True)
        compare = cast("dict[str, Any]", snap["compare"])
        assert compare["winner"] is True and compare["rank"] == 1
    finally:
        conn.close()


# --- a corrupt journal degrades, never 500s / kills the stream ---------------


def _corrupt_journal(inst: Path) -> None:
    (inst / "journal.jsonl").write_text('{"type": "step", "bogus": true}\n', encoding="utf-8")


def test_corrupt_journal_hub_shows_unreadable(
    server: tuple[WebServer, int], tmp_path: Path
) -> None:
    # One corrupt journal line must not 500 the whole landing page; the entry
    # stays listed with an unreadable status.
    _srv, port = server
    inst, _ = _make_machine_with_state(tmp_path, "sick", "0000-review")
    _corrupt_journal(inst)
    _make_run(tmp_path, "healthy-run", [{"type": "session.start", "user_task": "x"}])
    status, body, _ = _get(port, "/api/hub")
    assert status == 200
    hub = json.loads(body)
    (entry,) = [m for m in hub["machines"] if m["name"] == "sick"]
    assert entry["status"] == "unreadable"


def test_hub_parked_instance_reads_waiting(server: tuple[WebServer, int], tmp_path: Path) -> None:
    # A parked --exit-on-wait instance (an armed wait, no live worker) must read
    # "waiting" on the hub, not "running": a paused machine never looks busy.
    from agent6.machine.journal import MachineJournal, PendingWait

    _srv, port = server
    inst, _ = _make_machine_with_state(tmp_path, "parked", "0000-poll")
    MachineJournal(inst).write_pending_wait(PendingWait(state="poll", wake_epoch=None))
    status, body, _ = _get(port, "/api/hub")
    assert status == 200
    (entry,) = [m for m in json.loads(body)["machines"] if m["name"] == "parked"]
    assert entry["status"] == "waiting"


def test_corrupt_journal_machine_snapshot_is_422(
    server: tuple[WebServer, int], tmp_path: Path
) -> None:
    _srv, port = server
    inst, _ = _make_machine_with_state(tmp_path, "sick2", "0000-review")
    _corrupt_journal(inst)
    status, body, _ = _get(port, "/api/machine/sick2")
    assert status == 422
    assert "corrupt journal" in json.loads(body)["error"]


def test_corrupt_journal_machine_sse_sends_error_frame(
    server: tuple[WebServer, int], tmp_path: Path
) -> None:
    # The SSE stream must emit an in-band error frame and close, never write a
    # second HTTP status line into the open stream.
    _srv, port = server
    inst, _ = _make_machine_with_state(tmp_path, "sick3", "0000-review")
    _corrupt_journal(inst)
    conn = HTTPConnection("127.0.0.1", port, timeout=10)
    try:
        conn.request("GET", "/api/machine/sick3/events")
        resp = conn.getresponse()
        assert resp.status == 200
        seen = resp.read()  # stream closes after the error frame
        frames = [f for f in seen.split(b"\n\n") if f.startswith(b"data:")]
        assert len(frames) == 1
        assert "corrupt journal" in json.loads(frames[0][len(b"data:") :].strip())["error"]
        assert b"HTTP/1" not in seen  # no second status line inside the stream
    finally:
        conn.close()


# --- SSE catch-up folds history into one frame --------------------------------


def test_sse_run_catchup_folds_history_into_few_frames(
    server: tuple[WebServer, int], tmp_path: Path
) -> None:
    # Connecting to a run with a long history must not emit one full SessionState
    # frame per historical event (13 MB probed on a 502-event run): the backlog
    # folds into (almost) one snapshot.
    _srv, port = server
    events: list[dict[str, object]] = [{"type": "session.start", "user_task": "big"}]
    for i in range(150):
        events.append({"type": "tool.call", "name": f"t{i}", "args": {}})
        events.append({"type": "tool.result", "name": f"t{i}", "ok": True, "summary": "ok"})
    events.append({"type": "session.end", "all_passed": True})
    _make_run(tmp_path, "big-run", events)
    conn = HTTPConnection("127.0.0.1", port, timeout=10)
    try:
        conn.request("GET", "/api/session/big-run/events")
        resp = conn.getresponse()
        assert resp.status == 200
        seen_frames = 0

        def _caught_up(s: dict[str, object]) -> bool:
            nonlocal seen_frames
            seen_frames += 1
            return s.get("finished") is True and s.get("log_count") == len(events)

        _read_until(resp, _caught_up)
        assert seen_frames <= 5  # was ~1 per historical event
    finally:
        conn.close()


@pytest.mark.filterwarnings("ignore::pytest.PytestUnhandledThreadExceptionWarning")
def test_sse_run_closes_even_if_tailer_dies(
    server: tuple[WebServer, int], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The tail thread must ALWAYS enqueue its None sentinel: if it raises (the
    # injected raise below is intentionally unhandled in that thread), the
    # stream sends the folded snapshot and closes instead of hanging until the
    # client gives up.
    import agent6.ui.web._sse as sse_mod

    def _boom(*_a: object, **_k: object) -> object:
        raise RuntimeError("tailer died")

    monkeypatch.setattr(sse_mod, "tail_events", _boom)
    _srv, port = server
    _make_run(tmp_path, "dead-tail", [{"type": "session.start", "user_task": "x"}])
    conn = HTTPConnection("127.0.0.1", port, timeout=5)
    try:
        conn.request("GET", "/api/session/dead-tail/events")
        resp = conn.getresponse()
        assert resp.status == 200
        seen = resp.read()  # must reach EOF, not time out
        frames = [f for f in seen.split(b"\n\n") if f.startswith(b"data:")]
        assert len(frames) == 1  # the final (initial-state) snapshot
    finally:
        conn.close()


def test_sse_run_dead_worker_frame_is_terminal(
    server: tuple[WebServer, int], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A run whose worker died without a session.end must close its SSE stream with
    a TERMINAL frame carrying the dedicated transport bit: stream_dead=True +
    status_label="stale". `finished` stays the fold truth (False -- a crashed
    run is stale, not finished); the client closes on either signal, so the
    tab never reconnect-refolds forever over a dead run."""
    import agent6.ui.web._sse as sse_mod

    monkeypatch.setattr(sse_mod, "HEARTBEAT_S", 0.2)
    _make_run(
        tmp_path,
        "dead-worker",
        [{"type": "session.start", "user_task": "x"}, {"type": "role.call", "role": "worker"}],
    )
    session_dir = resolved_state_dir(tmp_path) / "sessions" / "runs" / "dead-worker"
    (session_dir / "worker.pid").write_text("999999999", encoding="utf-8")
    _srv, port = server
    conn = HTTPConnection("127.0.0.1", port, timeout=5)
    try:
        conn.request("GET", "/api/session/dead-worker/events")
        resp = conn.getresponse()
        assert resp.status == 200
        seen = resp.read()  # must reach EOF (the close sticks), not hang
    finally:
        conn.close()
    frames = [f for f in seen.split(b"\n\n") if f.startswith(b"data:")]
    last = json.loads(frames[-1][len(b"data:") :])
    assert last["stream_dead"] is True
    assert last["finished"] is False  # the fold truth is not overwritten
    assert last["status_label"] == "stale"


def test_sse_run_pidless_stale_frame_is_terminal(
    server: tuple[WebServer, int], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A crashed run that never recorded worker.pid (killed in preflight, or
    the pid file cleaned) heartbeated forever: the idle close only probed a
    RECORDED pid. The one dir decision (summarize_session_dir) already calls a
    pid-less run silent past its window "stale"; the stream must close on it
    with the same terminal frame as the recorded-dead-pid case."""
    import agent6.ui.web._sse as sse_mod

    monkeypatch.setattr(sse_mod, "HEARTBEAT_S", 0.2)
    _make_run(
        tmp_path,
        "pidless-stale",
        [{"type": "session.start", "user_task": "x"}, {"type": "role.call", "role": "worker"}],
    )
    session_dir = resolved_state_dir(tmp_path) / "sessions" / "runs" / "pidless-stale"
    assert not (session_dir / "worker.pid").exists()
    old = 1_000_000_000.0  # silent far past the stale window
    os.utime(session_dir / "logs.jsonl", (old, old))
    _srv, port = server
    conn = HTTPConnection("127.0.0.1", port, timeout=1)
    seen = b""
    eof = False
    try:
        conn.request("GET", "/api/session/pidless-stale/events")
        resp = conn.getresponse()
        assert resp.status == 200
        # Bounded read: the buggy stream never closes but keeps heartbeating,
        # so a plain read() would hang forever on live ping bytes.
        import time as _time

        deadline = _time.monotonic() + 4.0
        while _time.monotonic() < deadline:
            try:
                chunk = resp.read1(65536)
            except TimeoutError:
                continue
            if chunk == b"":
                eof = True  # the close stuck
                break
            seen += chunk
    finally:
        conn.close()
    assert eof, "stream never closed for a pid-less stale run"
    frames = [f for f in seen.split(b"\n\n") if f.startswith(b"data:")]
    last = json.loads(frames[-1][len(b"data:") :])
    assert last["stream_dead"] is True
    assert last["finished"] is False  # the fold truth is not overwritten
    assert last["status_label"] == "stale"


def test_sse_run_created_frame_is_terminal(
    server: tuple[WebServer, int], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A `created` run (worker killed in preflight before session.start, or a
    fork --no-run) reaches terminal WITHOUT a session.end, but the close only
    checked "stale": the tailer pinged forever, holding the thread and the
    frontends/ claim. The close now asks the codebase's own died_without_end,
    and the terminal frame keeps the truthful label ("created", not a
    hardcoded "stale")."""
    import agent6.ui.web._sse as sse_mod

    monkeypatch.setattr(sse_mod, "HEARTBEAT_S", 0.2)
    session_dir = resolved_state_dir(tmp_path) / "sessions" / "runs" / "created-run"
    session_dir.mkdir(parents=True)
    (session_dir / "logs.jsonl").write_text("", encoding="utf-8")  # no events, no pid
    _srv, port = server
    conn = HTTPConnection("127.0.0.1", port, timeout=1)
    seen = b""
    eof = False
    try:
        conn.request("GET", "/api/session/created-run/events")
        resp = conn.getresponse()
        assert resp.status == 200
        import time as _time

        deadline = _time.monotonic() + 4.0
        while _time.monotonic() < deadline:
            try:
                chunk = resp.read1(65536)
            except TimeoutError:
                continue
            if chunk == b"":
                eof = True
                break
            seen += chunk
    finally:
        conn.close()
    assert eof, "stream never closed for a created run"
    frames = [f for f in seen.split(b"\n\n") if f.startswith(b"data:")]
    last = json.loads(frames[-1][len(b"data:") :])
    assert last["stream_dead"] is True
    assert last["finished"] is False  # created, never started: not "finished"
    assert last["status_label"] == "created"
    assert last["live"] is False


def test_sse_run_parked_keeps_streaming(
    server: tuple[WebServer, int], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`parked` also never reached session.end, but its stream deliberately stays
    open: a parked submission the operator resumes starts logging into this
    same stream. Pin the exclusion so the died_without_end close cannot
    swallow it."""
    import agent6.ui.web._sse as sse_mod

    monkeypatch.setattr(sse_mod, "HEARTBEAT_S", 0.2)
    session_dir = resolved_state_dir(tmp_path) / "sessions" / "runs" / "parked-run"
    session_dir.mkdir(parents=True)
    (session_dir / "logs.jsonl").write_text("", encoding="utf-8")
    (session_dir / "manifest.json").write_text(
        json.dumps({"mode": "run", "user_task": "queued", "parked_task": "queued"}),
        encoding="utf-8",
    )
    _srv, port = server
    conn = HTTPConnection("127.0.0.1", port, timeout=1)
    eof = False
    try:
        conn.request("GET", "/api/session/parked-run/events")
        resp = conn.getresponse()
        assert resp.status == 200
        import time as _time

        deadline = _time.monotonic() + 1.5  # several heartbeats
        while _time.monotonic() < deadline:
            try:
                chunk = resp.read1(65536)
            except TimeoutError:
                continue
            if chunk == b"":
                eof = True
                break
    finally:
        conn.close()
    assert not eof, "the parked run's stream must stay open for a future resume"


def test_sse_machine_frame_carries_the_idle_age(
    server: tuple[WebServer, int],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The machine pane anchored its "agent working… Ns" timer to when the FRAME
    arrived, so a state wedged for forty minutes read as three seconds of work
    every time one landed. The frame carries a server-computed age, as the run
    stream's does, and the client ticks from that."""
    import agent6.ui.web._sse as sse_mod

    monkeypatch.setattr(sse_mod, "MACHINE_POLL_S", 0.05)
    monkeypatch.chdir(tmp_path)
    (tmp_path / "tiny.asm.toml").write_text(TINY, encoding="utf-8")
    assert main(["machine", "run", str(tmp_path / "tiny.asm.toml")]) == 0
    capsys.readouterr()
    inst = resolved_state_dir(tmp_path) / "machines" / "tiny"
    log = next(inst.glob("states/*/logs.jsonl"), None)
    if log is None:  # a pure wait/branch machine runs no agent state
        log = inst / "states" / "0001-work" / "logs.jsonl"
        log.parent.mkdir(parents=True, exist_ok=True)
    # An event 15 minutes old: the age must report the event's, not the frame's.
    log.write_text(
        json.dumps({"type": "role.call", "role": "agent", "model": "m", "ts": time.time() - 900})
        + "\n",
        encoding="utf-8",
    )
    _srv, port = server
    conn = HTTPConnection("127.0.0.1", port, timeout=5)
    try:
        conn.request("GET", "/api/machine/tiny/events")
        resp = conn.getresponse()
        seen = resp.read()
    finally:
        conn.close()
    frames = [f for f in seen.split(b"\n\n") if f.startswith(b"data:")]
    assert frames, "the machine stream sent no frame"
    payload = json.loads(frames[0][len(b"data:") :])
    age = payload["reasoning"]["last_event_age_s"]
    assert isinstance(age, (int, float)) and age >= 880, f"age reads as fresh: {age}"


def test_sse_machine_dead_worker_frame_is_terminal(
    server: tuple[WebServer, int],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A machine that died mid-state (no MachineEnd) must close its SSE stream
    with a DISTINCT worker_lost frame: supervisor loss is not a journaled end
    (the instance is resumable), so a fabricated `ended` styled it terminal;
    `ended` stays reserved for a durable MachineEnd, and a bare return left
    the tab reconnecting forever over a "running" machine."""
    import agent6.ui.web._sse as sse_mod

    monkeypatch.setattr(sse_mod, "MACHINE_POLL_S", 0.05)
    monkeypatch.chdir(tmp_path)
    (tmp_path / "tiny.asm.toml").write_text(TINY, encoding="utf-8")
    assert main(["machine", "run", str(tmp_path / "tiny.asm.toml")]) == 0
    capsys.readouterr()
    inst = resolved_state_dir(tmp_path) / "machines" / "tiny"
    # Un-end the journal (drop the MachineEnd line): the machine now reads as
    # mid-state, and its recorded worker pid points at a dead process.
    journal = inst / "journal.jsonl"
    lines = journal.read_text(encoding="utf-8").splitlines()
    assert "end" in lines[-1]
    journal.write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")
    (inst / "worker.pid").write_text("999999999", encoding="utf-8")
    _srv, port = server
    conn = HTTPConnection("127.0.0.1", port, timeout=5)
    try:
        conn.request("GET", "/api/machine/tiny/events")
        resp = conn.getresponse()
        assert resp.status == 200
        seen = resp.read()  # must reach EOF, not hang
    finally:
        conn.close()
    frames = [f for f in seen.split(b"\n\n") if f.startswith(b"data:")]
    last = json.loads(frames[-1][len(b"data:") :])
    assert last["machine"]["ended"] is None  # no journaled end was invented
    assert "died" in last["machine"]["worker_lost"]["reason"]


# --- POST hardening -----------------------------------------------------------


def test_oversize_post_body_is_413(server: tuple[WebServer, int]) -> None:
    # Headers only: the server refuses on Content-Length alone, before any body
    # bytes arrive (actually streaming 1 MiB races the server's early close).
    _srv, port = server
    conn = HTTPConnection("127.0.0.1", port, timeout=10)
    try:
        conn.putrequest("POST", "/api/new")
        conn.putheader("Content-Type", "application/json")
        conn.putheader("Content-Length", str((1 << 20) + 100))
        conn.endheaders()
        resp = conn.getresponse()
        assert resp.status == 413
        assert "body larger" in json.loads(resp.read())["error"]
    finally:
        conn.close()


def test_prune_body_is_drained_so_keepalive_is_not_poisoned(
    server: tuple[WebServer, int],
) -> None:
    # The client posts `{}` to prune. If the route does not read that body, the
    # 2 bytes sit on the keep-alive socket and the next pipelined request line is
    # parsed with them prepended -> 400 Bad Request. Pipeline prune + a GET on a
    # single socket and require the GET to be answered cleanly.
    _srv, port = server
    sock = socket.create_connection(("127.0.0.1", port), timeout=10)
    try:
        body = b"{}"
        prune = (
            b"POST /api/sessions/prune HTTP/1.1\r\n"
            b"Host: 127.0.0.1\r\n"
            b"Content-Type: application/json\r\n"
            b"Content-Length: " + str(len(body)).encode() + b"\r\n\r\n" + body
        )
        follow = b"GET /api/hub HTTP/1.1\r\nHost: 127.0.0.1\r\nConnection: close\r\n\r\n"
        sock.sendall(prune + follow)
        chunks = []
        while True:
            data = sock.recv(4096)
            if not data:
                break
            chunks.append(data)
        raw = b"".join(chunks)
    finally:
        sock.close()
    # Both requests were answered (prune then GET), the GET returned the hub
    # payload, and nothing was a 400 framing error: the prune body was drained.
    # Undrained, the GET line would parse as `{}GET /api/hub...` -> 400 and no
    # hub JSON.
    assert raw.count(b"HTTP/1.1 ") == 2, raw
    assert b" 400 " not in raw, raw
    assert b'"sessions":' in raw, raw  # the GET /api/hub payload came back intact


def test_negative_content_length_is_rejected(server: tuple[WebServer, int]) -> None:
    # A negative Content-Length must not reach rfile.read(n) (which would read to
    # EOF and park the worker); reject it up front.
    _srv, port = server
    status, body = _post_raw(
        port,
        "/api/new",
        b"",
        {"Content-Type": "application/json", "Content-Length": "-1"},
    )
    assert status == 400
    assert "Content-Length" in str(body["error"])


def test_chunked_post_body_is_refused(server: tuple[WebServer, int]) -> None:
    # Only Content-Length bodies are read; a chunked body would sit unread on
    # the connection exactly like an undrained early-error body.
    _srv, port = server
    status, body = _post_raw(
        port,
        "/api/sessions/prune",
        b"",
        {"Transfer-Encoding": "chunked", "Content-Type": "application/json"},
    )
    assert status == 411
    assert "chunked" in str(body["error"])


def test_unknown_post_verb_does_not_poison_keepalive(
    server: tuple[WebServer, int], tmp_path: Path
) -> None:
    # A 404 that leaves the body undrained poisoned the keep-alive connection:
    # the next request parsed the leftover body as its request line (probed
    # garbage 400). The server now closes; the client reconnects cleanly.
    _srv, port = server
    _make_run(tmp_path, "ka-run", [{"type": "session.start", "user_task": "x"}])
    conn = HTTPConnection("127.0.0.1", port, timeout=10)
    try:
        payload = json.dumps({"text": "hello"}).encode()
        conn.request(
            "POST", "/api/session/ka-run/bogusverb", payload, {"Content-Type": "application/json"}
        )
        resp = conn.getresponse()
        assert resp.status == 404
        resp.read()
        # Same client object again: must yield a clean 200, not body-garbage 400.
        conn.request("GET", "/api/hub")
        resp2 = conn.getresponse()
        assert resp2.status == 200
        assert json.loads(resp2.read())["sessions"]
    finally:
        conn.close()


# --- CSRF: cross-site state-changing POSTs are refused -----------------------


def test_cross_origin_post_refused(server: tuple[WebServer, int], tmp_path: Path) -> None:
    _srv, port = server
    _make_machine_with_state(tmp_path, "csrf1", "0000-review")
    status, body = _post_raw(
        port,
        "/api/machine/csrf1/poke",
        json.dumps({"message": "x"}).encode(),
        {
            "Content-Type": "application/json",
            "Host": f"127.0.0.1:{port}",
            "Origin": "https://evil.example",
        },
    )
    assert status == 403
    assert "cross-origin" in str(body.get("error", ""))


def test_non_json_content_type_post_refused(server: tuple[WebServer, int], tmp_path: Path) -> None:
    _srv, port = server
    inst, _ = _make_machine_with_state(tmp_path, "csrf2", "0000-review")
    # A JSON body smuggled in as a CORS-simple text/plain request is refused,
    # and the signal file is NOT written.
    status, _ = _post_raw(
        port,
        "/api/machine/csrf2/poke",
        json.dumps({"message": "x"}).encode(),
        {"Content-Type": "text/plain", "Host": f"127.0.0.1:{port}"},
    )
    assert status == 403
    assert not (inst / "signal").exists()


def test_same_origin_post_allowed(server: tuple[WebServer, int], tmp_path: Path) -> None:
    _srv, port = server
    inst, _ = _make_machine_with_state(tmp_path, "csrf3", "0000-review")
    status, body = _post_raw(
        port,
        "/api/machine/csrf3/poke",
        json.dumps({"message": "ok"}).encode(),
        {
            "Content-Type": "application/json",
            "Host": f"127.0.0.1:{port}",
            "Origin": f"http://127.0.0.1:{port}",
        },
    )
    assert status == 200 and body["ok"] is True
    assert (inst / "signal").exists()


# --- machine answers route to the rendered state, not the newest -------------


def test_machine_answer_routes_to_named_state_not_newest(
    server: tuple[WebServer, int], tmp_path: Path
) -> None:
    _srv, port = server
    # Two agent states, each with its own approval-1. The operator was shown the
    # OLDER state's prompt; the machine has since advanced to a newer state.
    inst, old_state = _make_machine_with_state(tmp_path, "adv", "0001-work", running=True)
    new_state = inst / "states" / "0002-review"
    new_state.mkdir(parents=True)
    (new_state / "logs.jsonl").write_text("", encoding="utf-8")
    status, body = _post(
        port,
        "/api/machine/adv/approve",
        {"id": "approval-1", "answer": "yes", "state": "0001-work"},
    )
    assert status == 200 and body["ok"] is True
    # The answer landed in the state the prompt was rendered from, NOT the newest.
    assert (old_state / "approvals" / "approval-1.answer").read_text(encoding="utf-8") == "yes"
    assert not (new_state / "approvals" / "approval-1.answer").exists()


def test_machine_answer_defaults_to_newest_state_without_hint(
    server: tuple[WebServer, int], tmp_path: Path
) -> None:
    _srv, port = server
    inst, _old = _make_machine_with_state(tmp_path, "adv2", "0001-work", running=True)
    new_state = inst / "states" / "0002-review"
    new_state.mkdir(parents=True)
    (new_state / "logs.jsonl").write_text("", encoding="utf-8")
    status, body = _post(port, "/api/machine/adv2/answer", {"id": "question-1", "answers": ["hi"]})
    assert status == 200 and body["ok"] is True
    assert (new_state / "questions" / "question-1.answer").read_text(
        encoding="utf-8"
    ) == json.dumps(["hi"])


def test_machine_answer_state_hint_traversal_is_contained(
    server: tuple[WebServer, int], tmp_path: Path
) -> None:
    _srv, port = server
    # running=True so the refusal rests on the state-hint component check.
    _make_machine_with_state(tmp_path, "adv3", "0001-work", running=True)
    escape = tmp_path / "pwned.answer"
    status, _ = _post(
        port,
        "/api/machine/adv3/approve",
        {"id": "approval-1", "answer": "yes", "state": "../../../../pwned"},
    )
    assert status != 200
    assert not escape.exists()


def test_config_suggest_endpoint(server: tuple[WebServer, int]) -> None:
    # Best-effort value suggestions; an env with no providers suggests nothing
    # but the endpoint always answers.
    _srv, port = server
    st, body, _ = _get(port, "/api/config/suggest/models.worker.provider")
    assert st == 200
    assert json.loads(body) == {"values": []}


def test_steer_compact_directive_routes_to_compact_request(
    server: tuple[WebServer, int], tmp_path: Path
) -> None:
    """A composer `/compact <focus>` on a live run becomes a compact request
    carrying the focus -- never a steer message the loop would read as text."""
    _srv, port = server
    session_dir = resolved_state_dir(tmp_path) / "sessions" / "runs" / "compact-run"
    session_dir.mkdir(parents=True)
    (session_dir / "logs.jsonl").write_text("", encoding="utf-8")
    (session_dir / "worker.pid").write_text(str(os.getpid()), encoding="utf-8")
    status, body = _post(
        port, "/api/session/compact-run/steer", {"text": "/compact keep the auth decisions"}
    )
    assert status == 200 and body["ok"] is True
    assert "compaction requested" in str(body["message"])
    assert (session_dir / "compact.request").read_text(
        encoding="utf-8"
    ) == "keep the auth decisions"
    assert not (session_dir / "steer.answer").exists()
    assert not (session_dir / "steer.request").exists()


def test_client_disconnects_are_quiet(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """A client vanishing mid-request is routine (a browser sends RST on
    navigate-away, reload, or an abandoned body), and it surfaces at the
    request-line read, outside every handler try. The stdlib handle_error
    printed a full traceback to stderr for each one; real handler errors
    keep their report."""
    srv = WebServer(("127.0.0.1", 0), tmp_path, "")
    try:
        for quiet_exc in (ConnectionResetError(104, "reset by peer"), BrokenPipeError()):
            try:
                raise quiet_exc
            except OSError:
                srv.handle_error(None, ("127.0.0.1", 12345))
        assert capsys.readouterr().err == ""
        try:
            raise ValueError("a real handler bug")
        except ValueError:
            srv.handle_error(None, ("127.0.0.1", 12345))
        assert "ValueError" in capsys.readouterr().err
    finally:
        srv.server_close()


def test_steer_btw_opens_a_side_ask(
    server: tuple[WebServer, int], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`/btw <question>` from the web composer opens the side ask (the CLI
    menu's mechanism, shared) instead of steering the run; a bare `/btw` is
    refused with what to type."""
    import agent6.ui.btw as btw_mod

    _srv, port = server
    session_dir = resolved_state_dir(tmp_path) / "sessions" / "runs" / "btw-run"
    session_dir.mkdir(parents=True)
    (session_dir / "logs.jsonl").write_text("", encoding="utf-8")
    (session_dir / "worker.pid").write_text(str(os.getpid()), encoding="utf-8")
    asks = resolved_state_dir(tmp_path) / "sessions" / "asks"

    def launch(cwd: Path, argv: list[str], env: dict[str, str]) -> str:
        d = asks / "quiet-fox-DDDDDD"
        d.mkdir(parents=True)
        (d / "manifest.json").write_text(json.dumps({"version": 3, "mode": "ask"}))
        lines = [
            {"type": "session.start", "user_task": "q"},
            {"type": "role.result", "text": "no"},
            {"type": "session.end", "reason": "answered", "all_passed": True},
        ]
        (d / "logs.jsonl").write_text("".join(json.dumps(e) + "\n" for e in lines))
        return ""

    monkeypatch.setattr(btw_mod, "direct_launch", launch)
    status, body = _post(port, "/api/session/btw-run/steer", {"text": "/btw"})
    assert status == 422 and "ask something" in str(body)
    status, body = _post(port, "/api/session/btw-run/steer", {"text": "/btw is it safe?"})
    assert status == 200 and body["ok"] is True and "opened" in str(body["message"])
    assert not (session_dir / "steer.request").exists()


def _git_chain(repo: Path) -> tuple[str, str, str]:
    """A repo with a base commit and two run commits; returns (base, c1, c2)."""
    import subprocess as sp

    def git(*args: str) -> str:
        return sp.run(
            ["git", "-c", "user.name=t", "-c", "user.email=t@example.invalid", *args],
            cwd=repo,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    git("init", "-q", "-b", "main")
    (repo / "a.txt").write_text("one\n")
    git("add", "-A")
    git("commit", "-q", "-m", "base")
    base = git("rev-parse", "HEAD")
    (repo / "a.txt").write_text("one\ntwo\n")
    git("commit", "-q", "-am", "step one")
    c1 = git("rev-parse", "HEAD")
    (repo / "b.txt").write_text("three\n")
    git("add", "-A")
    git("commit", "-q", "-m", "step two")
    c2 = git("rev-parse", "HEAD")
    return base, c1, c2


def test_step_diff_serves_one_step_or_the_cumulative_chain(
    server: tuple[WebServer, int], tmp_path: Path
) -> None:
    """The diff card's step selector reads `/diff?sha=`: one step's own patch,
    or `base..sha` with `cumulative=1`; a run the model controls has no chain
    and is refused, as is a sha that is not one of its commits."""
    _srv, port = server
    base, c1, c2 = _git_chain(tmp_path)
    session_dir = resolved_state_dir(tmp_path) / "sessions" / "runs" / "steps-run"
    session_dir.mkdir(parents=True)
    (session_dir / "logs.jsonl").write_text("", encoding="utf-8")
    (session_dir / "manifest.json").write_text(
        json.dumps({"version": 3, "mode": "run", "base_sha": base, "git_control": "agent6"}),
        encoding="utf-8",
    )
    status, raw, _ = _get(port, f"/api/session/steps-run/diff?sha={c2}")
    patch = str(json.loads(raw)["patch"])
    assert status == 200 and "three" in patch and "two" not in patch
    status, raw, _ = _get(port, f"/api/session/steps-run/diff?sha={c2}&cumulative=1")
    patch = str(json.loads(raw)["patch"])
    assert status == 200 and "two" in patch and "three" in patch
    status, _raw, _ = _get(port, "/api/session/steps-run/diff?sha=deadbeef")
    assert status == 422
    (session_dir / "manifest.json").write_text(
        json.dumps({"version": 3, "mode": "run", "base_sha": base, "git_control": "model"}),
        encoding="utf-8",
    )
    status, raw, _ = _get(port, f"/api/session/steps-run/diff?sha={c1}")
    assert status == 422 and "owns git" in str(json.loads(raw)["error"])


def test_session_snapshot_as_of_a_step(server: tuple[WebServer, int], tmp_path: Path) -> None:
    """`?step=<sha>` folds the log up to that commit (the Budget and Task graph
    widgets follow the step picked in the Latest commit card) and stamps
    `as_of`; a sha the run never made is refused."""
    _srv, port = server
    session_dir = resolved_state_dir(tmp_path) / "sessions" / "runs" / "asof-run"
    session_dir.mkdir(parents=True)
    events = [
        {"type": "session.start", "session_id": "asof-run", "mode": "run", "user_task": "t"},
        {"type": "loop.auto_commit", "iteration": 1, "sha": "a" * 40, "subject": "one"},
        {"type": "loop.auto_commit", "iteration": 2, "sha": "b" * 40, "subject": "two"},
    ]
    (session_dir / "logs.jsonl").write_text(
        "".join(json.dumps(e) + "\n" for e in events), encoding="utf-8"
    )
    status, raw, _ = _get(port, f"/api/session/asof-run?step={'a' * 40}")
    snap = json.loads(raw)
    assert status == 200 and snap["as_of"] == {"iteration": 1, "sha": "a" * 40}
    assert [st["iteration"] for st in snap["steps"]] == [1]
    status, raw, _ = _get(port, "/api/session/asof-run")
    snap = json.loads(raw)
    assert status == 200 and snap["as_of"] is None and len(snap["steps"]) == 2
    status, _raw, _ = _get(port, f"/api/session/asof-run?step={'c' * 40}")
    assert status == 422


def test_a_malformed_body_is_the_clients_error(server: tuple[WebServer, int]) -> None:
    """A body that is not JSON, or not an object, is a 400 with the reason,
    never a 500."""
    _srv, port = server
    headers = {"Content-Type": "application/json"}
    status, body = _post_raw(port, "/api/new", b"not json", headers)
    assert status == 400 and "bad request" in str(body.get("error"))
    status, body = _post_raw(port, "/api/new", b"[1, 2]", headers)
    assert status == 400 and "JSON object" in str(body.get("error"))
