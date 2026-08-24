# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The `agent6 web` server: a stdlib HTTP front-end over the shared read-side.

Serves web.page to a browser, fed by:
  - plain GET JSON endpoints (the same wire form as `agent6 attach --json`), and
  - SSE (`text/event-stream`) streams that re-fold logs.jsonl / the machine
    journal on each change and push a fresh snapshot.

Uses the stdlib `http.server.ThreadingHTTPServer`. Binds loopback by default; a
non-loopback bind is opt-in (see the `[web]` config section) and widens the
inbound network surface. The server only ever renders folded read-state and (in
the write phase) drives the typed `agent6.sessions.ipc` contracts; it never serves
secrets and never executes arbitrary input.
"""

from __future__ import annotations

import json
import os
import queue
import socket
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from ipaddress import ip_address
from pathlib import Path
from typing import Any, Literal
from urllib.parse import parse_qs, unquote, urlsplit

from pydantic import BaseModel, ConfigDict, ValidationError

from agent6 import __version__
from agent6.config import is_loopback_host
from agent6.machine import MachineError
from agent6.sessions.ipc import (
    read_worker_pid,
    register_frontend,
    unregister_frontend,
    worker_is_alive,
)
from agent6.sessions.layout import LOGS_NAME
from agent6.ui.spawn import spawn_new_work
from agent6.ui.web import actions, model
from agent6.ui.web.page import (
    FAVICON_SVG,
    ICON_SVG,
    MANIFEST_JSON,
    PAGE_HTML,
    SERVICE_WORKER_JS,
)
from agent6.viewmodel import (
    UnknownStepError,
    apply_event,
    died_without_end,
    initial_state,
    machine_is_parked,
    machine_snapshot,
    manifest_header,
    session_snapshot,
    session_state_as_dict,
    summarize_session_dir,
    tail_events,
)

# SSE tuning: coalesce high-frequency streaming deltas, heartbeat idle streams so
# a disconnected client is noticed and its worker thread exits.
_DELTA_COALESCE_S = 0.15
_HEARTBEAT_S = 15.0
_MACHINE_POLL_S = 0.5
_STREAMING_DELTAS = frozenset({"role.text_delta", "role.thinking_delta"})

# POST body cap. The typed bodies are a few strings (a task, an answer, a config
# value); 1 MiB is generous. An uncapped Content-Length would let one request
# buffer arbitrary bytes in this process.
_MAX_BODY_BYTES = 1 << 20


# Typed POST bodies (pydantic only at this HTTP trust boundary; extra keys are
# rejected so a malformed request fails loudly rather than silently ignoring a
# misspelled field).
class _Body(BaseModel):
    model_config = ConfigDict(extra="forbid")


class NewWorkBody(_Body):
    mode: str
    task: str
    preset: str = ""


class SteerBody(_Body):
    text: str = ""
    # For a machine: the per-state dir name the prompt was rendered from, so the
    # answer routes to that state even if the machine has since advanced. Empty
    # (the default, and always for a run) routes to the newest state.
    state: str = ""


class ApproveBody(_Body):
    id: str
    # The operator's literal choice; what a session answer GRANTS is the asking
    # side's to decide (agent6.sessions.ipc.record_answer), not this endpoint's.
    answer: Literal["yes", "no", "session", "session-deny"]
    state: str = ""


class AnswerBody(_Body):
    id: str
    answers: list[str]  # one per question in the ask_user prompt, by index
    state: str = ""


class MergeBody(_Body):
    strategy: str = ""


class ResumeBody(_Body):
    # The follow-up instruction a finished run is resumed with; empty = plain resume.
    text: str = ""
    # The config preset the leg continues under; empty = as the run recorded.
    preset: str = ""


class MachineCreateBody(_Body):
    task: str


class MachineRunBody(_Body):
    file: str


class MachinePokeBody(_Body):
    # A JSON `data` payload wins over a `message` string; neither = a bare wake.
    message: str = ""
    data: Any = None


class ConfigSetBody(_Body):
    key: str
    value: str = ""  # unused (and unrequired) when unset=True
    repo: bool = False
    unset: bool = False  # remove the key from the target layer instead of setting it


class WebServer(ThreadingHTTPServer):
    """A ThreadingHTTPServer that carries the repo cwd its handlers read from,
    and tracks which runs a browser is actively watching so it can register this
    process as an answering front-end (a frontends/ claim) only while someone
    is looking."""

    daemon_threads = True
    allow_reuse_address = True

    def __init__(
        self, addr: tuple[str, int], cwd: Path, target: str, config_path: Path | None = None
    ) -> None:
        super().__init__(addr, _Handler)
        self.cwd = cwd
        self.target = target
        self.config_path = config_path
        self._pid_lock = threading.Lock()
        self._watch_counts: dict[str, int] = {}

    def handle_error(self, request: Any, client_address: Any) -> None:
        # A client vanishing mid-request (navigate-away, reload, an abandoned
        # body) raises at the request-line read, outside every handler try;
        # the stdlib default printed a traceback for each. Real errors keep it.
        exc = sys.exc_info()[1]
        if isinstance(exc, (BrokenPipeError, ConnectionResetError)):
            return
        super().handle_error(request, client_address)

    def claim_session(self, session_dir: Path) -> None:
        """A browser opened this run's stream: register as an answer front-end
        so its approval/question/steer prompts bridge here. Reference-counted
        across concurrent viewers; the claim file is per-process, so other
        front-ends (TUI, attach) are never displaced."""
        key = str(session_dir)
        with self._pid_lock:
            n = self._watch_counts.get(key, 0) + 1
            self._watch_counts[key] = n
            if n == 1:
                register_frontend(session_dir, os.getpid())

    def release_session(self, session_dir: Path) -> None:
        """The last browser watching this run went away: drop our own claim so
        the run falls back to its headless behaviour instead of blocking on
        answers no one gives. The count transition and the claim change share
        the lock so a concurrent claim_session cannot interleave."""
        key = str(session_dir)
        with self._pid_lock:
            n = self._watch_counts.get(key, 1) - 1
            if n > 0:
                self._watch_counts[key] = n
                return
            self._watch_counts.pop(key, None)
            unregister_frontend(session_dir, os.getpid())


class _IPv6WebServer(WebServer):
    address_family = socket.AF_INET6


def _with_idle_age(payload: dict[str, Any]) -> dict[str, Any]:
    """*payload* with the reasoning fold's idle age filled in from its epoch.

    Server-computed, like the run stream's, so a browser on another machine
    needs no clock agreement: the client anchors its "working... Ns" timer to
    (its own now) - age and ticks locally. Anchoring to the frame's ARRIVAL
    instead showed a state wedged for forty minutes as three seconds of work.
    """
    reasoning = payload.get("reasoning") or {}
    ep = reasoning.get("last_event_ep")
    if not isinstance(ep, (int, float)):
        return payload
    fresh = {**reasoning, "last_event_age_s": max(0.0, time.time() - ep)}
    return {**payload, "reasoning": fresh}


def _bind_host(host: str) -> str:
    """Normalize URL-style bracketed IPv6 literals to socket bind addresses."""
    stripped = host.strip()
    if stripped.startswith("[") and stripped.endswith("]"):
        return stripped[1:-1]
    return stripped


def _is_ipv6_literal(host: str) -> bool:
    try:
        return ip_address(_bind_host(host)).version == 6
    except ValueError:
        return False


def _display_host(host: str) -> str:
    if host == "0.0.0.0":  # noqa: S104 - display only
        return "127.0.0.1"
    if host == "::":
        return "[::1]"
    return f"[{host}]" if _is_ipv6_literal(host) else host


def _create_web_server(
    host: str, port: int, cwd: Path, target: str, config_path: Path | None = None
) -> WebServer:
    bind_host = _bind_host(host)
    server_cls: type[WebServer] = _IPv6WebServer if _is_ipv6_literal(bind_host) else WebServer
    return server_cls((bind_host, port), cwd, target, config_path)


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server: WebServer  # type: ignore[assignment]

    def log_message(self, format: str, *args: Any) -> None:  # match the stdlib signature
        pass  # quiet; this is not a logging server

    @property
    def cwd(self) -> Path:
        return self.server.cwd

    @property
    def config_path(self) -> Path | None:
        return self.server.config_path

    # -- routing --------------------------------------------------------------

    def do_GET(self) -> None:  # BaseHTTPRequestHandler dispatch contract (method name fixed)
        path = unquote(urlsplit(self.path).path)
        try:
            self._route(path)
        except (BrokenPipeError, ConnectionResetError):
            pass  # client went away mid-response
        except Exception as exc:  # never take the whole server down for one bad request
            self._send_json({"error": str(exc)}, status=500)

    def do_POST(self) -> None:  # BaseHTTPRequestHandler dispatch contract (method name fixed)
        path = unquote(urlsplit(self.path).path)
        try:
            csrf_err = self._csrf_refusal()
            if csrf_err is not None:
                # Close the connection rather than drain an unread body under
                # HTTP/1.1 keep-alive (a partial read would desync framing).
                self.close_connection = True
                self._send_json({"error": csrf_err}, status=403)
                return
            if self.headers.get("Transfer-Encoding"):
                # Only Content-Length bodies are read; a chunked body would sit
                # unread on the connection like the early-error cases below.
                self.close_connection = True
                self._send_json({"error": "chunked bodies are not supported"}, status=411)
                return
            content_length = int(self.headers.get("Content-Length", "0") or "0")
            if content_length < 0:
                # A negative length would make rfile.read(n) read to EOF, buffering
                # arbitrary bytes and parking the worker thread; the > cap check
                # alone (n < cap) lets it through.
                self.close_connection = True
                self._send_json({"error": "negative Content-Length"}, status=400)
                return
            if content_length > _MAX_BODY_BYTES:
                self.close_connection = True
                self._send_json({"error": f"body larger than {_MAX_BODY_BYTES} bytes"}, status=413)
                return
            self._route_post(path)
        except (BrokenPipeError, ConnectionResetError):
            pass
        except ValidationError as exc:
            # The body was read (validation runs on the parsed body), so the
            # connection framing is intact and may stay open.
            self._send_json({"error": f"bad request: {exc.errors()}"}, status=400)
        except Exception as exc:  # never take the whole server down for one bad request
            # The body may not have been read; a keep-alive reuse would parse the
            # leftover bytes as the next request line. Close instead.
            self.close_connection = True
            self._send_json({"error": str(exc)}, status=500)

    def _csrf_refusal(self) -> str | None:
        """Reason to refuse this state-changing POST as cross-site, or None.

        The web UI has no app-level auth: on the default loopback bind the
        machine is the trust boundary (any local process/user reaches
        127.0.0.1, so a shared box is not confined here), behind `tailscale
        serve` the tailnet is. Neither stops a page on ANOTHER origin in the
        operator's
        browser from POSTing here (classic CSRF). Two standard,
        deployment-agnostic checks close it:

        - Require `Content-Type: application/json` for a body. A cross-site
          `fetch` with that type is not a CORS "simple request", so the
          browser sends a preflight we never answer and the POST is blocked.
          This shuts the hole where a JSON body rides in as `text/plain`.
        - If an `Origin` is present, its host:port must equal `Host`. Our own
          page matches; a cross-site page (Origin: https://evil.example) does
          not. A missing Origin (curl, the CLI) is allowed -- not
          browser-driven, so not a CSRF vector.

        Residual: DNS rebinding (an attacker page rebinds its own hostname to
        127.0.0.1 so its request is same-origin) is not covered here; a Host
        allow-list would break the tailnet-hostname `tailscale serve` path, so
        that vector is left to the network layer."""
        n = int(self.headers.get("Content-Length", "0") or "0")
        if n > 0:
            ctype = (self.headers.get("Content-Type") or "").split(";", 1)[0].strip().lower()
            if ctype != "application/json":
                return f"POST body must be Content-Type: application/json, not {ctype!r}"
        origin = self.headers.get("Origin")
        if origin:
            host = self.headers.get("Host", "")
            if urlsplit(origin).netloc != host:
                return f"cross-origin POST refused (Origin {origin!r} != Host {host!r})"
        return None

    def _read_body(self) -> dict[str, Any]:
        n = int(self.headers.get("Content-Length", "0") or "0")
        raw = self.rfile.read(n) if n > 0 else b""
        if not raw:
            return {}
        obj = json.loads(raw)
        if not isinstance(obj, dict):
            raise ValueError("request body must be a JSON object")
        return obj

    def _route_post(self, path: str) -> None:  # noqa: PLR0911
        parts = path.strip("/").split("/")
        # /api/new  /api/sessions/prune  /api/config  /api/machine/create  /api/machine/run
        if path == "/api/new":
            body = NewWorkBody.model_validate(self._read_body())
            session_dir, err = spawn_new_work(
                self.cwd, body.mode, body.task, preset=body.preset, config_path=self.config_path
            )
            session_id = session_dir.name if session_dir is not None else None
            self._ok_or_err(session_id is not None, {"session_id": session_id}, err)
            return
        if path == "/api/sessions/rm_asks":
            self._read_body()  # drain the `{}` body (keep-alive framing)
            ok, msg = actions.remove_asks(self.cwd, self.config_path)
            self._ok_or_err(ok, {"message": msg}, msg)
            return
        if path == "/api/sessions/prune":
            # Drain the body (the client posts `{}`) even though prune takes no
            # params: an unread body would sit on the keep-alive socket and the
            # next request line would be parsed with it prepended.
            self._read_body()
            ok, msg = actions.prune_sessions(self.cwd, self.config_path)
            self._ok_or_err(ok, {"message": msg}, msg)
            return
        if path == "/api/config":
            body = ConfigSetBody.model_validate(self._read_body())
            if body.unset:
                ok, msg = actions.unset_config(
                    self.cwd, body.key, repo=body.repo, config_path=self.config_path
                )
            else:
                ok, msg = actions.set_config(
                    self.cwd, body.key, body.value, repo=body.repo, config_path=self.config_path
                )
            self._ok_or_err(ok, {"message": msg}, msg)
            return
        if path == "/api/machine/create":
            body = MachineCreateBody.model_validate(self._read_body())
            draft, err = actions.spawn_machine_create(self.cwd, body.task, self.config_path)
            self._ok_or_err(draft is not None, {"draft": draft}, err)
            return
        if path == "/api/machine/run":
            body = MachineRunBody.model_validate(self._read_body())
            ok, msg = actions.spawn_machine_run(self.cwd, body.file, self.config_path)
            self._ok_or_err(ok, {"message": msg}, msg)
            return
        # /api/session/<id>/<verb>
        if len(parts) == 4 and parts[0] == "api" and parts[1] == "session":
            self._route_session_post(parts[2], parts[3])
            return
        # /api/machine/<name>/<verb>
        if len(parts) == 4 and parts[0] == "api" and parts[1] == "machine":
            self._route_machine_post(parts[2], parts[3])
            return
        self._post_not_found(path)

    def _post_not_found(self, what: str) -> None:
        """404 for a POST whose body was never read: close the connection so the
        unread body cannot be parsed as the next request on keep-alive."""
        self.close_connection = True
        self._send_json({"error": f"not found: {what}"}, status=404)

    def _route_session_post(self, session_id: str, verb: str) -> None:
        if verb == "steer":
            body = SteerBody.model_validate(self._read_body())
            ok, msg = actions.steer(self.cwd, session_id, body.text)
        elif verb == "approve":
            ab = ApproveBody.model_validate(self._read_body())
            ok, msg = actions.approve(self.cwd, session_id, ab.id, ab.answer)
        elif verb == "answer":
            qb = AnswerBody.model_validate(self._read_body())
            ok, msg = actions.answer_question(self.cwd, session_id, qb.id, qb.answers)
        elif verb == "merge":
            mb = MergeBody.model_validate(self._read_body())
            ok, msg = actions.merge_run(
                self.cwd, session_id, mb.strategy, config_path=self.config_path
            )
        elif verb == "undo":
            self._read_body()  # no parameters; drain the (empty) body
            payload, err = actions.undo_session(self.cwd, session_id)
            self._ok_or_err(payload is not None, payload or {}, err)
            return
        elif verb == "resume":
            rb = ResumeBody.model_validate(self._read_body())
            ok, msg = actions.resume_run(
                self.cwd, session_id, rb.text, preset=rb.preset, config_path=self.config_path
            )
        elif verb == "stop_step":
            self._read_body()  # drain the `{}` body (keep-alive framing)
            ok, msg = actions.stop_after_step(self.cwd, session_id)
        elif verb == "compact":
            self._read_body()  # drain the `{}` body (keep-alive framing)
            ok, msg = actions.compact_run(self.cwd, session_id)
        elif verb == "rm":
            self._read_body()  # drain the `{}` body (keep-alive framing)
            ok, msg = actions.remove_session(self.cwd, session_id, self.config_path)
        elif verb == "run_plan":
            self._read_body()  # drain the `{}` body (keep-alive framing)
            payload, err = actions.run_plan(self.cwd, session_id, self.config_path)
            self._ok_or_err(payload is not None, payload or {}, err)
            return
        else:
            self._post_not_found(f"run/{session_id}/{verb}")
            return
        self._ok_or_err(ok, {"message": msg}, msg)

    def _route_machine_post(self, name: str, verb: str) -> None:
        if verb == "poke":
            pb = MachinePokeBody.model_validate(self._read_body())
            ok, msg = actions.machine_poke(self.cwd, name, data=pb.data, message=pb.message)
        elif verb == "stop":
            ok, msg = actions.machine_stop(self.cwd, name)
        elif verb == "steer":
            body = SteerBody.model_validate(self._read_body())
            ok, msg = actions.machine_steer(self.cwd, name, body.text, state=body.state)
        elif verb == "approve":
            ab = ApproveBody.model_validate(self._read_body())
            ok, msg = actions.machine_approve(self.cwd, name, ab.id, ab.answer, state=ab.state)
        elif verb == "answer":
            qb = AnswerBody.model_validate(self._read_body())
            ok, msg = actions.machine_answer(self.cwd, name, qb.id, qb.answers, state=qb.state)
        else:
            self._post_not_found(f"machine/{name}/{verb}")
            return
        self._ok_or_err(ok, {"message": msg}, msg)

    def _ok_or_err(self, ok: bool, payload: dict[str, Any], err: str) -> None:
        if ok:
            self._send_json({"ok": True, **payload})
        else:
            self._send_json({"ok": False, "error": err}, status=422)

    def _route(self, path: str) -> None:  # noqa: PLR0911
        if path == "/":
            self._send_bytes(PAGE_HTML.encode("utf-8"), "text/html; charset=utf-8")
            return
        if path == "/manifest.webmanifest":
            self._send_bytes(MANIFEST_JSON.encode("utf-8"), "application/manifest+json")
            return
        if path == "/sw.js":
            self._send_bytes(SERVICE_WORKER_JS.encode("utf-8"), "text/javascript; charset=utf-8")
            return
        if path == "/icon.svg":
            self._send_bytes(ICON_SVG.encode("utf-8"), "image/svg+xml")
            return
        if path == "/favicon.svg":
            self._send_bytes(FAVICON_SVG.encode("utf-8"), "image/svg+xml")
            return
        if path == "/api/meta":
            self._send_json(
                {
                    "version": __version__,
                    "target": self.server.target,
                    "target_kind": self._target_kind(),
                }
            )
            return
        if path == "/api/hub":
            self._send_json(model.hub_payload(self.cwd, self.config_path))
            return
        if path == "/api/config":
            self._send_json(model.config_payload(self.cwd, self.config_path))
            return
        if path.startswith("/api/config/suggest/"):
            key = path.removeprefix("/api/config/suggest/")
            self._send_json({"values": model.config_suggestions(self.cwd, key, self.config_path)})
            return
        parts = path.strip("/").split("/")
        # /api/session/<id>[/conversation|/restate|/events]
        if len(parts) in (3, 4) and parts[0] == "api" and parts[1] == "session":
            self._route_session(parts[2], parts[3] if len(parts) > 3 else "")
            return
        # /api/machine/<name>[/reasoning|/conversation|/events]
        if len(parts) in (3, 4) and parts[0] == "api" and parts[1] == "machine":
            self._route_machine(parts[2], parts[3] if len(parts) > 3 else "")
            return
        # /api/draft/<name>[/events]: a `machine create` draft, watched as a run.
        if len(parts) in (3, 4) and parts[0] == "api" and parts[1] == "draft":
            self._route_draft(parts[2], parts[3] if len(parts) > 3 else "")
            return
        self._send_json({"error": f"not found: {path}"}, status=404)

    def _target_kind(self) -> str:
        """Which view the CLI-given target deep-links to (session / draft / machine),
        or "" when there is no target or it matches nothing. Resolved per request,
        so a target that appears after startup still resolves."""
        t = self.server.target
        if not t:
            return ""
        if model.session_dir_for(self.cwd, t) is not None:
            return "session"
        if model.draft_dir_for(self.cwd, t) is not None:
            return "draft"
        if model.machine_dir_for(self.cwd, t) is not None:
            return "machine"
        return ""

    def _route_draft(self, name: str, sub: str) -> None:
        draft_dir = model.draft_dir_for(self.cwd, name)
        if draft_dir is None:
            self._send_json({"error": f"no draft {name!r}"}, status=404)
            return
        if sub == "":
            self._send_json(session_snapshot(draft_dir))
        elif sub == "conversation":
            self._send_json(model.conversation_payload(draft_dir))
        elif sub == "events":
            self._sse_session(draft_dir)
        else:
            self._send_json({"error": f"not found: draft/{name}/{sub}"}, status=404)

    def _route_session(self, session_id: str, sub: str) -> None:
        session_dir = model.session_dir_for(self.cwd, session_id)
        if session_dir is None:
            self._send_json({"error": f"no session {session_id!r}"}, status=404)
            return
        if sub == "":
            step = (parse_qs(urlsplit(self.path).query).get("step") or [""])[0]
            try:
                self._send_json(session_snapshot(session_dir, repo=self.cwd, step=step))
            except UnknownStepError as e:
                self._send_json({"error": str(e)}, status=422)
        elif sub == "conversation":
            self._send_json(model.conversation_payload(session_dir))
        elif sub == "restate":
            self._send_json(model.restate_payload(session_dir))
        elif sub == "diff":
            q = parse_qs(urlsplit(self.path).query)
            payload, why = model.step_diff_payload(
                self.cwd,
                session_dir,
                (q.get("sha") or [""])[0],
                cumulative=(q.get("cumulative") or ["0"])[0] in ("1", "true"),
            )
            if payload is None:
                self._send_json({"error": why}, status=422)
            else:
                self._send_json(payload)
        elif sub == "events":
            self._sse_session(session_dir)
        else:
            self._send_json({"error": f"not found: /api/session/{session_id}/{sub}"}, status=404)

    def _route_machine(self, name: str, sub: str) -> None:
        machine_dir = model.machine_dir_for(self.cwd, name)
        if machine_dir is None:
            self._send_json({"error": f"no machine {name!r}"}, status=404)
            return
        try:
            if sub == "":
                self._send_json(machine_snapshot(machine_dir))
            elif sub == "reasoning":
                self._send_json(model.machine_reasoning_snapshot(machine_dir))
            elif sub == "conversation":
                self._send_json(model.machine_conversation_payload(machine_dir))
            elif sub == "events":
                self._sse_machine(machine_dir)
            else:
                self._send_json({"error": f"not found: machine/{name}/{sub}"}, status=404)
        except MachineError as exc:
            self._send_json({"error": f"machine {name!r}: {'; '.join(exc.problems)}"}, status=422)

    # -- plain responses ------------------------------------------------------

    def _send_json(self, payload: Any, *, status: int = 200) -> None:
        self._send_bytes(
            json.dumps(payload).encode("utf-8"), "application/json; charset=utf-8", status=status
        )

    def _send_bytes(self, body: bytes, ctype: str, *, status: int = 200) -> None:
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        if self.close_connection:
            # Announce the close (CSRF refusal, unread POST body): without the
            # header a keep-alive client reuses the socket this handler is about to shut.
            self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(body)

    # -- SSE ------------------------------------------------------------------

    def _begin_sse(self) -> None:
        # Close-framed, not keep-alive: an SSE body has no Content-Length, so the
        # socket closing is what tells the client the stream ended (and lets a
        # finished run's EventSource stop). close_connection makes the handler
        # close the socket when the stream loop returns.
        self.close_connection = True
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.send_header("X-Accel-Buffering", "no")  # tell any proxy not to buffer SSE
        self.end_headers()

    def _sse_send(self, obj: Any) -> bool:
        """Write one SSE data frame. Returns False if the client has gone away."""
        try:
            self.wfile.write(f"data: {json.dumps(obj)}\n\n".encode())
            self.wfile.flush()
        except OSError:
            return False
        return True

    def _sse_ping(self) -> bool:
        try:
            self.wfile.write(b": ping\n\n")
            self.wfile.flush()
        except OSError:
            return False
        return True

    def _sse_session(self, session_dir: Path) -> None:
        """Stream a run: fold logs.jsonl incrementally, push a fresh SessionState
        snapshot on each event (coalescing streaming deltas). A background tailer
        feeds a queue so the response loop can heartbeat idle periods and exit
        promptly when the client disconnects. While connected we register as the
        run's answer front-end so its approval/steer prompts bridge to the browser."""
        self._begin_sse()
        self.server.claim_session(session_dir)
        try:
            self._sse_session_loop(session_dir)
        finally:
            self.server.release_session(session_dir)

    def _sse_session_loop(self, session_dir: Path) -> None:
        events: queue.Queue[dict[str, Any] | None] = queue.Queue()
        stop = threading.Event()

        def tail() -> None:
            src = session_dir / LOGS_NAME
            try:
                # NOT stop_when_finished: a finished run resumed from any other
                # surface logs into this same file, and a stream that closed at
                # session.end left the page frozen on "stopped" while the hub
                # said "running", forever. The TUI follows across legs the same
                # way; the client closes only on stream_dead (or navigation).
                for ev in tail_events(
                    src, follow=True, stop_when_finished=False, should_stop=stop.is_set
                ):
                    events.put(ev)
            finally:
                # ALWAYS enqueue the sentinel, even if the tailer raises: without
                # it the response loop would block on heartbeats forever.
                events.put(None)  # run ended (or tail cancelled/failed), tailer done

        threading.Thread(target=tail, daemon=True).start()

        # Manifest-derived header fields (branch facts + the fan-out compare
        # outcome), read once per connection: they are fixed for the run's life
        # (merged_into lands after the run ends; a reopen/reconnect re-reads).
        header = manifest_header(session_dir, repo=self.cwd)

        def frame(*, dead: bool = False) -> dict[str, Any]:
            # session_dir per frame, not once at connect: a parked run the operator
            # resumes starts logging into this same stream, and the label (and
            # `live`) have to follow.
            d = {**session_state_as_dict(state, session_dir), **header}
            if state.last_event_ep is not None:
                # Server-computed so a browser on another machine needs no clock
                # agreement: the client anchors its "working… Ns" timer to
                # (its own now) - age, then ticks locally.
                d["last_event_age_s"] = max(0.0, time.time() - state.last_event_ep)
            if dead:
                # Transport signal, distinct from the fold's `finished`: this
                # stream will send nothing more (dead worker, no session.end), so
                # the client must close instead of letting EventSource retry
                # into a reconnect-refold loop. `finished` stays the fold truth
                # -- a crashed run is stale, not "finished".
                d["stream_dead"] = True
            return d

        try:
            state = initial_state()
            last_delta_emit = 0.0
            while True:
                try:
                    ev: dict[str, Any] | None = events.get(timeout=_HEARTBEAT_S)
                except queue.Empty:
                    if not self._sse_ping():
                        return
                    # A run that reached terminal without its own session.end
                    # (crash, went quiet, killed in preflight) would otherwise
                    # pin this worker forever; ask the codebase's own
                    # died_without_end rather than one word of it. `parked` is
                    # deliberately excluded: a parked submission the operator
                    # resumes starts logging into this same stream.
                    word = summarize_session_dir(session_dir).status
                    if word != "parked" and died_without_end(word):
                        self._sse_send(frame(dead=True))
                        return
                    continue
                # Fold everything already queued into ONE frame. On connect the
                # tailer replays the whole history, and a full SessionState frame per
                # historical event is quadratic (13 MB probed on a 502-event run).
                last_type = ""
                while ev is not None:
                    state = apply_event(state, ev)
                    last_type = str(ev.get("type", ""))
                    try:
                        ev = events.get_nowait()
                    except queue.Empty:
                        break
                if ev is None:  # run ended: send the final snapshot and close
                    self._sse_send(frame())
                    return
                now = time.monotonic()
                if last_type in _STREAMING_DELTAS and (now - last_delta_emit) < _DELTA_COALESCE_S:
                    continue  # coalesce bursts of text/thinking deltas
                if not self._sse_send(frame()):
                    return
                last_delta_emit = now
        finally:
            # cancel the tailer so it exits on disconnect / dead run, not just session.end
            stop.set()

    def _sse_machine(self, machine_dir: Path) -> None:
        """Stream a machine: re-fold the journal + the current agent state's
        reasoning on a poll, pushing the combined snapshot when it changes. While
        connected we register as the answer front-end on the INSTANCE dir, so a
        machine agent state's approval/question/steer prompts bridge to the
        browser (the state's answer files live in its per-state dir; the liveness
        gate probes this instance dir)."""
        self._begin_sse()
        self.server.claim_session(machine_dir)
        try:
            self._sse_machine_loop(machine_dir)
        finally:
            self.server.release_session(machine_dir)

    def _sse_machine_loop(self, machine_dir: Path) -> None:
        prev = ""
        idle = 0.0
        while True:
            try:
                payload = {
                    "machine": machine_snapshot(machine_dir),
                    "reasoning": model.machine_reasoning_snapshot(machine_dir),
                }
            except MachineError as exc:
                self._sse_send({"error": "; ".join(exc.problems)})
                return
            blob = json.dumps(payload, sort_keys=True)
            if blob != prev:
                # The age is derived at SEND time and deliberately outside the
                # comparison above: it changes every poll, so including it would
                # send a frame every poll. The epoch it comes from does not.
                if not self._sse_send(_with_idle_age(payload)):
                    return
                prev = blob
                idle = 0.0
            else:
                idle += _MACHINE_POLL_S
                if idle >= _HEARTBEAT_S and not self._sse_ping():
                    return
                if idle >= _HEARTBEAT_S:
                    idle = 0.0
            if payload["machine"].get("ended") is not None:
                return  # machine terminated: final snapshot sent, close the stream
            # A machine that died mid-state (no MachineEnd) would pin this
            # stream forever: its worker.pid points at a dead process AND no
            # armed wait explains the absence (a parked --exit-on-wait machine
            # legitimately has no live process between scheduler ticks).
            if (
                read_worker_pid(machine_dir) is not None
                and not worker_is_alive(machine_dir)
                and not machine_is_parked(machine_dir)
            ):
                # Supervisor loss is NOT a journaled end: the instance is
                # resumable, and a fabricated `ended` (a status the journal
                # vocabulary does not even hold) styled it terminal. A
                # distinct field closes the stream truthfully; `ended` stays
                # reserved for a durable MachineEnd. A bare return would
                # leave the tab reconnecting forever over a "running" machine.
                payload["machine"]["worker_lost"] = {
                    "reason": "worker died",
                    "state": payload["machine"].get("current", ""),
                }
                self._sse_send(_with_idle_age(payload))
                return
            time.sleep(_MACHINE_POLL_S)


def run_web(
    target: str,
    *,
    host: str,
    port: int,
    cwd: Path | None = None,
    config_path: Path | None = None,
) -> int:
    """Serve the web UI on host:port until interrupted. `target` deep-links the
    page to a run id or machine name on load (empty opens the hub)."""
    workdir = cwd or Path.cwd()
    bind_host = _bind_host(host)
    try:
        server = _create_web_server(bind_host, port, workdir, target, config_path)
    except OSError as exc:
        print(f"agent6 web: cannot bind {bind_host}:{port}: {exc}", file=sys.stderr)
        return 2
    shown = _display_host(bind_host)
    print(f"agent6 web: serving on http://{shown}:{port}  (Ctrl-C to stop)", file=sys.stderr)
    if not is_loopback_host(bind_host):
        print(
            "agent6 web: WARNING bound to a non-loopback address; anyone who can reach"
            f" {bind_host}:{port} can drive this agent. Prefer `tailscale serve` in front of a"
            " loopback bind.",
            file=sys.stderr,
        )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nagent6 web: stopped", file=sys.stderr)
    finally:
        server.server_close()
    return 0
