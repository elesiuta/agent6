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
import socket
import sys
import threading
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
    register_frontend,
    unregister_frontend,
)
from agent6.ui.spawn import spawn_new_work
from agent6.ui.web import actions, model
from agent6.ui.web._sse import SseChannel, stream_machine, stream_session
from agent6.ui.web.page import (
    FAVICON_SVG,
    ICON_SVG,
    MANIFEST_JSON,
    PAGE_HTML,
    SERVICE_WORKER_JS,
)
from agent6.viewmodel import (
    UnknownStepError,
    machine_snapshot,
    session_snapshot,
)

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


class PruneBody(_Body):
    # The CLI's `--delete-squashed`: force-delete a branch the manifest
    # confirms was squash-merged (the default strategy, so without it a
    # merged run's branch stays).
    delete_squashed: bool = False


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


def _validation_message(exc: ValidationError) -> str:
    """The failed fields as one line (`task: field required`), not the repr of
    pydantic's error list."""
    clauses: list[str] = []
    for err in exc.errors():
        field = ".".join(str(part) for part in err["loc"]) or "body"
        msg = str(err["msg"])
        clauses.append(f"{field}: {msg[:1].lower()}{msg[1:]}")
    return "; ".join(clauses)


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
            self._send_json({"error": _validation_message(exc)}, status=400)
        except ValueError as exc:
            # A body that is not JSON, or not an object: `_read_body` consumed
            # it, so the connection may stay open too.
            self._send_json({"error": f"bad request: {exc}"}, status=400)
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
            body = PruneBody.model_validate(self._read_body())
            ok, msg = actions.prune_sessions(
                self.cwd, delete_squashed=body.delete_squashed, config_path=self.config_path
            )
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
        self._post_not_found(f"not found: {path}")

    def _post_not_found(self, message: str) -> None:
        """404 for a POST whose body was never read: close the connection so the
        unread body cannot be parsed as the next request on keep-alive."""
        self.close_connection = True
        self._send_json({"error": message}, status=404)

    def _route_session_post(self, session_id: str, verb: str) -> None:
        if model.session_dir_for(self.cwd, session_id) is None:
            self._post_not_found(f"no session {session_id!r}")  # as its GET answers
            return
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
            self._post_not_found(f"not found: run/{session_id}/{verb}")
            return
        self._ok_or_err(ok, {"message": msg}, msg)

    def _route_machine_post(self, name: str, verb: str) -> None:
        if model.machine_dir_for(self.cwd, name) is None:
            self._post_not_found(f"no machine {name!r}")  # as its GET answers
            return
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
            self._post_not_found(f"not found: machine/{name}/{verb}")
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
        """Stream a run (see `_sse.stream_session`). While connected we register
        as the run's answer front-end so its approval/steer prompts bridge to
        the browser."""
        self._begin_sse()
        self.server.claim_session(session_dir)
        try:
            stream_session(self._channel(), session_dir, repo=self.cwd)
        finally:
            self.server.release_session(session_dir)

    def _sse_machine(self, machine_dir: Path) -> None:
        """Stream a machine (see `_sse.stream_machine`). While connected we
        register as the answer front-end on the INSTANCE dir, so a machine
        agent state's approval/question/steer prompts bridge to the browser
        (the state's answer files live in its per-state dir; the liveness gate
        probes this instance dir)."""
        self._begin_sse()
        self.server.claim_session(machine_dir)
        try:
            stream_machine(self._channel(), machine_dir)
        finally:
            self.server.release_session(machine_dir)

    def _channel(self) -> SseChannel:
        return SseChannel(send=self._sse_send, ping=self._sse_ping)


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
