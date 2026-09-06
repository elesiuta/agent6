# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The `claude_code` provider: agent6's worker on the operator's Claude Code login.

One long-lived `claude -p` process per worker leg, driven over stream-json on
its stdin and stdout. agent6 is the process's only tool source: it serves the
sdk-type MCP server `agent6` on the same pipes, advertising the loop's tools
and answering each `tools/call` with the result the loop dispatched. A round
ends at `message_stop`; a tool round returns while the CLI is blocked on its
first `tools/call`, and the next `call()` answers every pending request in
order from the loop's tool_result blocks, with any later notice text riding
as extra text items on the last answer. A call that passes no tools is one
process per call.

The process continues only when the new history extends what this session
produced: the pending tool results, then text-only user turns. Anything else
(resume, fork, a tier-2 restart, a retried or interrupted call, a tool-list or
system change, a live context near the window) kills it and respawns with the
history rendered as one user message (`render_history`).

Spend is plan-metered: every round records $0 with the plan windows from the
CLI's `rate_limit_event`, and a round without a reading fails closed. The
CLI's cumulative list-price estimate is never recorded.

The child runs unjailed as the operator (its own login under `$HOME`, its own
egress) with every Claude Code capability off: argv is operator config and
literals, the system prompt travels in a 0600 file inside a private empty
cwd, prompts and tool results travel on stdin, and the environment is curated
(`child_env`). The initialize handshake carries the account block; only the
email is kept, in memory, to scrub it from returned text (docs/security.md).
"""

from __future__ import annotations

import contextlib
import json
import os
import queue
import shutil
import signal
import subprocess
import tempfile
import threading
import time
import weakref
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, Any

import agent6
from agent6.budget import BudgetTracker, PlanUsage
from agent6.portable import atomic_write
from agent6.providers._claude_code_wire import (
    CLAUDE_CODE_PERSIST_BYTES,
    CLAUDE_CODE_RESULT_CAP_CHARS,
    MCP_SERVER,
    TOOL_PREFIX,
    Skeleton,
    bare_tool_name,
    child_env,
    claude_argv,
    history_skeleton,
    mcp_answer,
    message_blocks,
    message_texts,
    plan_usage_from_rate_limit,
    render_history,
    tool_results,
    tool_use_ids,
    user_line,
)
from agent6.providers._stream import (
    STREAM_FIRST_DATA_TIMEOUT_S,
    STREAM_IDLE_TIMEOUT_S,
    STREAM_THINKING_IDLE_TIMEOUT_S,
    STREAM_WATCHDOG_TICK_S,
)
from agent6.providers.types import (
    ProviderAborted,
    ProviderError,
    ProviderInterrupted,
    ProviderResponse,
    ToolDefinition,
    TranscriptRecorder,
)
from agent6.sandbox.jail import die_with_parent

EMAIL_PLACEHOLDER = "<operator-email>"
_MAX_LINE_BYTES = 8 * 1024 * 1024
_STDERR_KEEP_BYTES = 8192
# A round that moved a window is followed by its reading within milliseconds
# of message_stop (4 ms measured on 2.1.251): one tick covers it. A process's
# first round is always followed by one; none inside the grace is not meterable.
_PLAN_READING_DRAIN_S = STREAM_WATCHDOG_TICK_S
_PLAN_READING_GRACE_S = 3.0
# The reserve `models.registry.compaction_thresholds` keeps below the window;
# a live context past it restarts the session on the compacted mirror.
_CONTEXT_RESERVE_TOKENS = 16_384
_KILL_GRACE_S = 1.0


def _missing_binary(binary: str) -> str:
    return (
        f"Claude Code binary {binary!r} not found on PATH; install Claude Code and sign in"
        " (`claude auth login`), or set [providers.<name>].binary"
    )


def login_status(binary: str, *, timeout_s: float = 20.0) -> str | None:
    """None when `<binary> auth status --json` reports a signed-in login, else
    the remedy. Only `loggedIn` is read; the body (email, org) is never
    returned, printed, or journaled."""
    argv = [binary, "auth", "status", "--json"]
    try:
        proc = subprocess.run(  # noqa: PLW1510 - a signed-out login exits 1 with a JSON body
            argv, env=child_env(), capture_output=True, timeout=timeout_s
        )
    except FileNotFoundError:
        return _missing_binary(binary)
    except subprocess.TimeoutExpired:
        return f"`{binary} auth status --json` did not answer within {timeout_s:.0f}s"
    except OSError as exc:
        return f"`{binary} auth status --json` failed: {exc}"
    try:
        data = json.loads(proc.stdout.decode(errors="replace"))
    except ValueError:
        tail = proc.stderr.decode(errors="replace").strip()[-400:]
        return (
            f"`{binary} auth status --json` returned no JSON (exit {proc.returncode}):"
            f" {tail or 'no output'}"
        )
    if isinstance(data, dict) and data.get("loggedIn") is True:
        return None
    return (
        f"Claude Code is not signed in for uid {os.getuid()} (HOME={os.environ.get('HOME', '')}):"
        " run `claude auth login` as that user. agent6 stores no Claude credentials and this"
        " provider uses no API key."
    )


def _reap(proc: subprocess.Popen[bytes], private_dir: Path) -> None:
    """End the child: SIGTERM to its process group (the CLI exits at once and
    removes its messaging socket; on stdin EOF it stays up while a tools/call
    awaits its answer), SIGKILL after the grace, then the private directory.
    The reader threads close stdout and stderr at their EOF."""
    with contextlib.suppress(OSError, ValueError):
        if proc.stdin is not None:
            proc.stdin.close()
    if proc.poll() is None:
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.killpg(proc.pid, signal.SIGTERM)
        try:
            proc.wait(timeout=_KILL_GRACE_S)
        except subprocess.TimeoutExpired:
            with contextlib.suppress(ProcessLookupError, PermissionError):
                os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
    shutil.rmtree(private_dir, ignore_errors=True)


def _drain_stderr(pipe: IO[bytes], keep: list[bytes]) -> None:
    with contextlib.suppress(OSError, ValueError):
        while chunk := pipe.read(4096):
            keep.append(chunk)
            while len(keep) > 2:
                keep.pop(0)
    pipe.close()


@dataclass(slots=True)
class _ToolCall:
    """One pending `tools/call`: the control request to answer and its JSON-RPC id."""

    request_id: str
    rpc_id: Any


@dataclass(slots=True, weakref_slot=True)
class _Session:
    """One spawned child and everything the continuation rule needs."""

    proc: subprocess.Popen[bytes]
    private_dir: Path
    system: str
    tools: list[ToolDefinition]
    lines: queue.Queue[dict[str, Any] | None]
    # A line read past a round's end, returned by the next read.
    pushback: list[dict[str, Any] | None] = field(default_factory=list)
    # Ends the child once: on close(), or at interpreter exit for a caller
    # that never closes (a machine agent's worker).
    reap: weakref.finalize[[subprocess.Popen[bytes], Path], _Session] = field(init=False)
    stdin_lock: threading.Lock = field(default_factory=threading.Lock)
    stderr_tail: list[bytes] = field(default_factory=list)
    stdin_log: list[dict[str, Any]] = field(default_factory=list)
    # The caller's history as this session consumed it (`message_skeleton`
    # per message), and the tool_use ids the last round produced (answered in
    # order by the next call).
    consumed: tuple[Skeleton, ...] = ()
    pending: tuple[str, ...] = ()
    calls: dict[str, _ToolCall] = field(default_factory=dict)
    plan: PlanUsage | None = None
    resolved_model: str = ""
    session_id: str = ""
    account_email: str = ""
    restart_next: bool = False

    def __post_init__(self) -> None:
        self.reap = weakref.finalize(self, _reap, self.proc, self.private_dir)

    @property
    def tool_names(self) -> tuple[str, ...]:
        return tuple(td.name for td in self.tools)


@dataclass(frozen=True, slots=True)
class _Tail:
    """What a continuing call sends: the pending results by id and the text
    blocks that follow them (notices, nudges, steer text)."""

    results: dict[str, str]
    texts: tuple[str, ...]


def _continuation(s: _Session, messages: Sequence[Mapping[str, Any]]) -> _Tail | None:
    """The one rule for keeping the process: the new history must extend
    what this session produced. The consumed prefix still has its skeleton
    (a tier-2 restart replaces it; tier-1 rewrites keep it); after it comes
    the assistant turn of the last round (or nothing, when the loop popped a
    quiet one), then user turns only: the pending tool results first, text
    blocks after. None means restart."""
    n = len(s.consumed)
    if history_skeleton(messages[:n]) != s.consumed:
        return None
    tail = list(messages[n:])
    if tail and tail[0].get("role") == "assistant" and tool_use_ids(tail[0]) == s.pending:
        tail = tail[1:]
    elif s.pending:
        return None
    if not tail or any(m.get("role") != "user" for m in tail):
        return None
    results: dict[str, str] = {}
    texts: list[str] = []
    if s.pending:
        results = tool_results(tail[0])
        if tuple(results) != s.pending:
            return None
        texts.extend(message_texts(tail[0]))
        tail = tail[1:]
    if any(tool_results(m) for m in tail):
        return None
    texts.extend(text for m in tail for text in message_texts(m))
    return _Tail(results=results, texts=tuple(texts)) if results or texts else None


def _serve_inline(s: _Session, msg: dict[str, Any]) -> bool:
    """Answer, from the reader thread, what needs no loop state: the MCP
    handshake (`initialize`, `notifications/initialized`, `tools/list`,
    `ping`, an unknown method) and the CLI's own initialize response, whose
    account block is dropped except the email kept in memory for the scrub.
    True when consumed; a `tools/call` or any other line is queued for
    `call()`."""
    kind = msg.get("type")
    if kind == "control_response":
        response = msg.get("response") or {}
        if response.get("request_id") == "init-1":
            account = (response.get("response") or {}).get("account") or {}
            s.account_email = str(account.get("email") or "")
            s.lines.put({"type": "_agent6_handshake_done"})
        return True
    if kind != "control_request":
        return False
    request = msg.get("request") or {}
    if request.get("subtype") != "mcp_message":
        return False
    rpc = request.get("message") or {}
    method, rpc_id = rpc.get("method"), rpc.get("id")
    if method == "tools/call":
        return False
    rid = str(msg.get("request_id", ""))
    if method == "initialize":
        result: dict[str, Any] = {
            "protocolVersion": (rpc.get("params") or {}).get("protocolVersion", ""),
            "capabilities": {"tools": {}},
            "serverInfo": {"name": MCP_SERVER, "version": agent6.__version__},
        }
    elif method == "tools/list":
        result = {
            "tools": [
                {"name": td.name, "description": td.description, "inputSchema": td.input_schema}
                for td in s.tools
            ]
        }
    elif method == "ping" or rpc_id is None:
        result = {}  # a ping, or a notification acknowledged with an empty result
    else:
        _write(
            s,
            {
                "type": "control_response",
                "response": {
                    "subtype": "success",
                    "request_id": rid,
                    "response": {
                        "mcp_response": {
                            "jsonrpc": "2.0",
                            "id": rpc_id,
                            "error": {"code": -32601, "message": "unknown method"},
                        }
                    },
                },
            },
            log=False,
        )
        return True
    _write(s, mcp_answer(rid, rpc_id, result), log=False)
    return True


def _read_stdout(s: _Session) -> None:
    """The reader thread; it owns the pipe (a close from another thread while
    a read is in flight lets that read land on the next child's descriptor)."""
    stream = s.proc.stdout
    assert stream is not None
    while True:
        try:
            raw = stream.readline(_MAX_LINE_BYTES + 1)
        except (OSError, ValueError):
            break
        if not raw:
            break
        if len(raw) > _MAX_LINE_BYTES:
            while raw and not raw.endswith(b"\n"):
                raw = stream.readline(_MAX_LINE_BYTES + 1)
            s.lines.put({"type": "_agent6_error", "text": "claude wrote a line over 8 MiB"})
            continue
        try:
            msg = json.loads(raw.decode("utf-8", errors="replace"))
        except ValueError:
            s.lines.put(
                {"type": "_agent6_error", "text": f"claude wrote a non-JSON line: {raw[:200]!r}"}
            )
            continue
        if not isinstance(msg, dict):
            continue
        try:
            consumed = _serve_inline(s, msg)
        except ProviderError as exc:
            s.lines.put({"type": "_agent6_error", "text": str(exc)})
            break
        if not consumed:
            s.lines.put(msg)
    stream.close()
    s.lines.put(None)


def _write(s: _Session, obj: dict[str, Any], *, log: bool = True) -> None:
    data = json.dumps(obj, separators=(",", ":")).encode() + b"\n"
    with s.stdin_lock:
        stdin = s.proc.stdin
        assert stdin is not None
        try:
            stdin.write(data)
            stdin.flush()
        except (OSError, ValueError) as exc:
            raise ProviderError(f"claude stopped reading stdin: {exc}") from exc
    if log:
        s.stdin_log.append(obj)


def _stderr_tail(s: _Session) -> str:
    text = b"".join(s.stderr_tail)[-_STDERR_KEEP_BYTES:].decode(errors="replace").strip()
    return text[-400:]


def _email_prefix_len(email: str, text: str) -> int:
    """The longest proper prefix of *email* that ends *text*; 0 when none."""
    return next((k for k in range(len(email) - 1, 0, -1) if text.endswith(email[:k])), 0)


def _int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


@dataclass(slots=True)
class _Round:
    """One API round as it streams in: the assistant blocks (Anthropic shape),
    the message_delta usage and stop reason, and whether message_stop landed."""

    blocks: list[dict[str, Any]] = field(default_factory=list)
    usage: dict[str, Any] = field(default_factory=dict)
    stop_reason: str = ""
    message_id: str = ""
    ended: bool = False
    # The open block's delta callback and the streamed tail the scrub holds.
    cb: Callable[[str], None] | None = None
    held: str = ""


class _Watch:
    """The wait loop's per-tick checks: operator stop/steer and the idle clock,
    with the phase-specific thresholds the SSE providers use."""

    def __init__(
        self,
        should_abort: Callable[[], bool] | None,
        should_interrupt: Callable[[], bool] | None,
    ) -> None:
        self._abort = should_abort
        self._interrupt = should_interrupt
        self.limit = STREAM_FIRST_DATA_TIMEOUT_S
        self.last_at = time.monotonic()

    def mark(self) -> None:
        self.last_at = time.monotonic()

    def tick(self) -> None:
        if _poll(self._abort):
            raise ProviderAborted("run stopped by operator")
        if _poll(self._interrupt):
            raise ProviderInterrupted("steer requested mid-turn")
        if time.monotonic() - self.last_at > self.limit:
            raise ProviderError(f"claude produced no output for {self.limit:.0f}s")


def _poll(fn: Callable[[], bool] | None) -> bool:
    """An operator-state poll; a failing poll never kills the watch."""
    if fn is None:
        return False
    try:
        return bool(fn())
    except Exception:
        return False


@dataclass(frozen=True, slots=True)
class ClaudeCodeProvider:
    """The worker on the operator's Claude Code login (module docstring)."""

    model: str
    binary: str = "claude"
    effort: str | None = None
    transcript_sink: TranscriptRecorder | None = None
    budget: BudgetTracker | None = None
    context_tokens: int | None = None
    # The live session (a mutable cell on a frozen dataclass).
    _cell: list[_Session | None] = field(default_factory=lambda: [None])

    def call(
        self,
        *,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[ToolDefinition] | None = None,
        max_tokens: int = 4096,
        temperature: float | None = None,
        reasoning_effort: str | None = None,
        text_delta_callback: Callable[[str], None] | None = None,
        thinking_delta_callback: Callable[[str], None] | None = None,
        should_abort: Callable[[], bool] | None = None,
        should_interrupt: Callable[[], bool] | None = None,
    ) -> ProviderResponse:
        del max_tokens, temperature, reasoning_effort  # the binary owns sampling; effort is argv
        if self.budget is not None:
            self.budget.check()
        tool_list = list(tools or ())
        watch = _Watch(should_abort, should_interrupt)
        s = self._cell[0] if tool_list else None
        try:
            if tool_list:
                s = self._worker_session(s, system, tool_list, messages, watch)
            else:
                # A side call is one process per call; the worker's session,
                # blocked on its tools/call, is untouched.
                s = self._spawn(system, [], messages, watch)
            s.consumed = history_skeleton(messages)
            resp = self._read_round(s, watch, text_delta_callback, thinking_delta_callback)
        except BaseException:
            if s is not None:
                self._close(s)  # an exceptional exit leaves no half-consumed process behind
            raise
        self._record_transcript(s, system, messages, resp)
        if not tool_list:
            s.reap()
        return resp

    def close(self) -> None:
        """Kill the worker's child and remove its private directory."""
        if (s := self._cell[0]) is not None:
            self._close(s)

    def _close(self, s: _Session) -> None:
        s.reap()
        if self._cell[0] is s:
            self._cell[0] = None

    # ---- process lifecycle -------------------------------------------------

    def _worker_session(
        self,
        s: _Session | None,
        system: str,
        tools: list[ToolDefinition],
        messages: Sequence[Mapping[str, Any]],
        watch: _Watch,
    ) -> _Session:
        """The worker's process: continued when the history extends what it
        produced, else respawned on the rendered history."""
        if (
            s is not None
            and s.proc.poll() is None
            and not s.restart_next
            and s.system == system
            and s.tool_names == tuple(td.name for td in tools)
        ):
            tail = _continuation(s, messages)
            if tail is not None:
                self._continue(s, tail, watch)
                return s
        self.close()
        s = self._cell[0] = self._spawn(system, tools, messages, watch)
        return s

    def _spawn(
        self,
        system: str,
        tools: list[ToolDefinition],
        messages: Sequence[Mapping[str, Any]],
        watch: _Watch,
    ) -> _Session:
        """One child through its handshake and the history's `system/init`;
        a failure on the way reaps it."""
        private_dir = Path(tempfile.mkdtemp(prefix="agent6-claude-"))
        prompt_file = private_dir / "system_prompt.txt"
        atomic_write(prompt_file, system)  # a new file lands 0600
        try:
            proc = subprocess.Popen(
                claude_argv(self.binary, self.model, self.effort, prompt_file),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=private_dir,
                env=child_env(),
                start_new_session=True,
                preexec_fn=die_with_parent(os.getpid(), sig=signal.SIGKILL),  # noqa: PLW1509
            )
        except FileNotFoundError as exc:
            shutil.rmtree(private_dir, ignore_errors=True)
            raise ProviderError(_missing_binary(self.binary), fatal=True) from exc
        except OSError as exc:
            shutil.rmtree(private_dir, ignore_errors=True)
            raise ProviderError(f"cannot start {self.binary!r}: {exc}") from exc
        s = _Session(
            proc=proc, private_dir=private_dir, system=system, tools=tools, lines=queue.Queue()
        )
        threading.Thread(
            target=_read_stdout, args=(s,), name="agent6-claude-stdout", daemon=True
        ).start()
        assert proc.stderr is not None
        threading.Thread(
            target=_drain_stderr,
            args=(proc.stderr, s.stderr_tail),
            name="agent6-claude-stderr",
            daemon=True,
        ).start()
        try:
            _write(
                s,
                {
                    "type": "control_request",
                    "request_id": "init-1",
                    "request": {"subtype": "initialize", "hooks": None},
                },
                log=False,
            )
            while self._next_line(s, watch).get("type") != "_agent6_handshake_done":
                pass
            _write(s, user_line(render_history(messages)))
            while True:
                line = self._next_line(s, watch)
                if line.get("type") == "system" and line.get("subtype") == "init":
                    self._audit_init(s, line)
                    return s
                self._absorb(s, line)
        except BaseException:
            s.reap()
            raise

    def _audit_init(self, s: _Session, line: Mapping[str, Any]) -> None:
        """The default-deny check on every `system/init`: the model's tool set
        is exactly what agent6 offered, and the child authenticates with the
        login, never a key."""
        offered = {TOOL_PREFIX + td.name for td in s.tools}
        exposed = {str(t) for t in (line.get("tools") or ())}
        if exposed != offered:
            extra = ", ".join(sorted(exposed - offered)) or "none"
            missing = ", ".join(sorted(offered - exposed)) or "none"
            raise ProviderError(
                f"Claude Code exposed a tool set agent6 did not offer (extra: {extra};"
                f" missing: {missing})",
                fatal=True,
            )
        source = line.get("apiKeySource")
        if source != "none":
            raise ProviderError(
                f"Claude Code resolved an API key source ({source}) instead of the subscription"
                " login; agent6 passes no ANTHROPIC_* variable, so the key comes from the Claude"
                " config dir. Remove it there, or sign in with `claude auth login`.",
                fatal=True,
            )
        s.resolved_model = str(line.get("model") or self.model)
        s.session_id = str(line.get("session_id") or "")

    def _continue(self, s: _Session, tail: _Tail, watch: _Watch) -> None:
        if not tail.results:
            _write(s, user_line("\n\n".join(tail.texts)))
            return
        last = s.pending[-1]
        for tool_use_id in s.pending:
            call = self._await_call(s, tool_use_id, watch)
            content = [{"type": "text", "text": tail.results[tool_use_id]}]
            if tool_use_id == last:
                content.extend({"type": "text", "text": text} for text in tail.texts)
            size = sum(len(str(item["text"]).encode()) for item in content)
            if size > CLAUDE_CODE_PERSIST_BYTES:
                raise ProviderError(
                    f"a {size}-byte tool result is over Claude Code's"
                    f" {CLAUDE_CODE_PERSIST_BYTES}-byte threshold: it would be written under"
                    " ~/.claude/projects and reach the model as a 2 KB preview. The loop caps"
                    f" results at {CLAUDE_CODE_RESULT_CAP_CHARS} characters for this provider;"
                    " this one is wider in bytes than in characters.",
                    fatal=True,
                )
            _write(s, mcp_answer(call.request_id, call.rpc_id, {"content": content}))
        s.pending = ()

    def _await_call(self, s: _Session, tool_use_id: str, watch: _Watch) -> _ToolCall:
        """The `tools/call` for *tool_use_id*: stashed while the round was read,
        or the next one the CLI sends (it serialises calls, one per answer)."""
        watch.limit = STREAM_FIRST_DATA_TIMEOUT_S
        while tool_use_id not in s.calls:
            line = self._next_line(s, watch)
            if line.get("type") in ("stream_event", "assistant", "result"):
                raise ProviderError(
                    f"claude moved on while tool call {tool_use_id} was unanswered"
                    f" ({line.get('type')} arrived)"
                )
            self._absorb(s, line)
        return s.calls.pop(tool_use_id)

    def _absorb(self, s: _Session, line: Mapping[str, Any]) -> None:
        """A line outside a round's stream: stash a tools/call, allow a stray
        can_use_tool, refuse any other control request, take a plan reading,
        drop the CLI's echoes and progress lines."""
        kind = line.get("type")
        if kind == "control_request":
            self._control(s, line)
        elif kind == "rate_limit_event":
            s.plan = plan_usage_from_rate_limit(line.get("rate_limit_info") or {}) or s.plan
        elif kind == "system" and line.get("subtype") == "init":
            self._audit_init(s, line)
        elif kind == "result":
            self._check_result(s, line)

    def _control(self, s: _Session, line: Mapping[str, Any]) -> None:
        request = line.get("request") or {}
        rid = str(line.get("request_id", ""))
        subtype = request.get("subtype")
        if subtype == "mcp_message":
            rpc = request.get("message") or {}
            params = rpc.get("params") or {}
            meta = params.get("_meta") or {}
            tool_use_id = str(meta.get("claudecode/toolUseId") or "")
            if rpc.get("method") != "tools/call" or not tool_use_id:
                raise ProviderError(f"claude sent an unexpected MCP request: {rpc.get('method')!r}")
            s.calls[tool_use_id] = _ToolCall(rid, rpc.get("id"))
        elif subtype == "can_use_tool":
            # Never sent under `--allowedTools mcp__agent6`; agent6's own
            # approval gate already ran, so a deny here would only strand it.
            _write(
                s,
                {
                    "type": "control_response",
                    "response": {
                        "subtype": "success",
                        "request_id": rid,
                        "response": {"behavior": "allow", "updatedInput": request.get("input")},
                    },
                },
            )
        else:
            _write(
                s,
                {
                    "type": "control_response",
                    "response": {
                        "subtype": "error",
                        "request_id": rid,
                        "error": "unsupported control request",
                    },
                },
            )

    def _check_result(self, s: _Session, line: Mapping[str, Any]) -> None:
        subtype = str(line.get("subtype") or "")
        if not line.get("is_error") and not subtype.startswith("error"):
            return
        text = self._scrub(s, str(line.get("result") or subtype))
        if "not logged in" in text.lower():
            raise ProviderError(
                "Claude Code is not signed in; run `claude auth login` as this user", fatal=True
            )
        # `api_error_status` names the API status of a failed turn (404 for an
        # unknown model); the loop's retry ladder skips the permanent ones.
        status = line.get("api_error_status")
        status = status if isinstance(status, int) else None
        where = f" (HTTP {status})" if status else ""
        raise ProviderError(f"claude result {subtype}{where}: {text[:500]}", status_code=status)

    def _next_line(self, s: _Session, watch: _Watch) -> dict[str, Any]:
        while True:
            if s.pushback:
                line = s.pushback.pop()
            else:
                try:
                    line = s.lines.get(timeout=STREAM_WATCHDOG_TICK_S)
                except queue.Empty:
                    watch.tick()
                    continue
            if line is None:
                # stdout closes a moment before the exit status lands.
                try:
                    rc: int | None = s.proc.wait(timeout=_KILL_GRACE_S)
                except subprocess.TimeoutExpired:
                    rc = None
                tail = self._scrub(s, _stderr_tail(s)) or "no stderr"
                raise ProviderError(f"claude exited {rc}: {tail}")
            if line.get("type") == "_agent6_error":
                raise ProviderError(str(line.get("text")))
            watch.mark()
            return line

    # ---- one API round --------------------------------------------------------

    def _read_round(
        self,
        s: _Session,
        watch: _Watch,
        text_cb: Callable[[str], None] | None,
        thinking_cb: Callable[[str], None] | None,
    ) -> ProviderResponse:
        r = _Round()
        plan_before = s.plan
        watch.limit = STREAM_FIRST_DATA_TIMEOUT_S
        while not r.ended:
            line = self._next_line(s, watch)
            kind = line.get("type")
            if kind == "stream_event":
                self._stream_event(s, r, line.get("event") or {}, watch, text_cb, thinking_cb)
            elif kind == "assistant":
                r.blocks.extend(
                    self._block(s, b) for b in message_blocks(line.get("message") or {})
                )
            elif kind == "result":
                # A turn that ended without a stream (the signed-out synthetic reply).
                self._check_result(s, line)
                r.stop_reason = r.stop_reason or str(line.get("stop_reason") or "end_turn")
                r.ended = True
            else:
                self._absorb(s, line)
        ids = tuple(str(b.get("id", "")) for b in r.blocks if b.get("type") == "tool_use")
        if not ids:
            self._read_to_result(s, watch)
        if s.plan is plan_before:
            self._await_plan_reading(s)
        s.pending = ids
        return self._finish_round(s, r)

    def _stream_event(
        self,
        s: _Session,
        r: _Round,
        event: Mapping[str, Any],
        watch: _Watch,
        text_cb: Callable[[str], None] | None,
        thinking_cb: Callable[[str], None] | None,
    ) -> None:
        kind = event.get("type")
        if kind == "message_start":
            r.message_id = str((event.get("message") or {}).get("id", ""))
            watch.limit = STREAM_IDLE_TIMEOUT_S
        elif kind == "content_block_start":
            thinking = (event.get("content_block") or {}).get("type") == "thinking"
            watch.limit = STREAM_THINKING_IDLE_TIMEOUT_S if thinking else STREAM_IDLE_TIMEOUT_S
        elif kind == "content_block_stop":
            watch.limit = STREAM_IDLE_TIMEOUT_S
            self._emit_delta(s, r, "", r.cb, final=True)
        elif kind == "content_block_delta":
            delta = event.get("delta") or {}
            if delta.get("type") == "text_delta":
                self._emit_delta(s, r, str(delta.get("text", "")), text_cb)
            elif delta.get("type") == "thinking_delta":
                self._emit_delta(s, r, str(delta.get("thinking", "")), thinking_cb)
        elif kind == "message_delta":
            r.stop_reason = str((event.get("delta") or {}).get("stop_reason") or "")
            r.usage = dict(event.get("usage") or {})
        elif kind == "message_stop":
            r.ended = True

    def _emit_delta(
        self,
        s: _Session,
        r: _Round,
        piece: str,
        cb: Callable[[str], None] | None,
        *,
        final: bool = False,
    ) -> None:
        """Stream a delta with the email scrubbed across delta boundaries: a
        tail that could start an occurrence waits for the next delta, and
        content_block_stop flushes it."""
        if cb is None:
            return
        r.cb = cb
        out = self._scrub(s, r.held + piece)
        keep = 0 if final else _email_prefix_len(s.account_email, out)
        r.held = out[len(out) - keep :] if keep else ""
        if keep:
            out = out[:-keep]
        if out:
            cb(out)

    def _finish_round(self, s: _Session, r: _Round) -> ProviderResponse:
        input_tokens = _int(r.usage.get("input_tokens"))
        cache_read = _int(r.usage.get("cache_read_input_tokens"))
        cache_creation = _int(r.usage.get("cache_creation_input_tokens"))
        output_tokens = _int(r.usage.get("output_tokens"))
        self._record(s, input_tokens, output_tokens, cache_read, cache_creation)
        live_context = input_tokens + cache_read + cache_creation
        if self.context_tokens and live_context > self.context_tokens - _CONTEXT_RESERVE_TOKENS:
            s.restart_next = True  # the next call replays the compacted mirror
        return ProviderResponse(
            text="".join(str(b.get("text", "")) for b in r.blocks if b.get("type") == "text"),
            tool_uses=tuple(
                {"id": b.get("id"), "name": b.get("name"), "input": b.get("input")}
                for b in r.blocks
                if b.get("type") == "tool_use"
            ),
            stop_reason=r.stop_reason,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_read_tokens=cache_read,
            cache_creation_tokens=cache_creation,
            cost_usd=0.0,
            raw={
                "content": r.blocks,
                "model": s.resolved_model,
                "id": r.message_id,
                "usage": r.usage,
            },
        )

    def _read_to_result(self, s: _Session, watch: _Watch) -> None:
        """A prose round's `result` line belongs to that round: read it so a
        turn error is raised where it happened."""
        watch.limit = STREAM_IDLE_TIMEOUT_S
        while True:
            line = self._next_line(s, watch)
            if line.get("type") == "result":
                self._check_result(s, line)
                return
            if line.get("type") in ("stream_event", "assistant"):
                raise ProviderError(
                    "claude started another round before reporting the turn's result"
                )
            self._absorb(s, line)

    def _await_plan_reading(self, s: _Session) -> None:
        """Wait one drain window for the round's reading (the first-reading
        grace when the process has none yet); a line of the next round is
        pushed back for the next read."""
        before = s.plan
        deadline = time.monotonic() + (
            _PLAN_READING_GRACE_S if before is None else _PLAN_READING_DRAIN_S
        )
        while s.plan is before:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return
            try:
                line = s.lines.get(timeout=min(remaining, STREAM_WATCHDOG_TICK_S))
            except queue.Empty:
                continue
            if line is None or line.get("type") in (
                "_agent6_error",
                "stream_event",
                "assistant",
                "result",
            ):
                s.pushback.append(line)
                return
            self._absorb(s, line)

    def _record(
        self, s: _Session, inp: int, out: int, cache_read: int, cache_creation: int
    ) -> None:
        if self.budget is None:
            return
        if inp + cache_read + cache_creation <= 0:
            raise ProviderError(
                "claude reported no usage input tokens for this round; budgeted runs require"
                " provider usage accounting"
            )
        if s.plan is None:
            # Fatal: a retry replays the history into the same absence, one
            # billed round per attempt.
            raise ProviderError(
                "claude reported no plan window (rate_limit_event) for this round; agent6 meters"
                " this provider by plan window only",
                fatal=True,
            )
        self.budget.record(
            model=s.resolved_model or self.model,
            input_tokens=inp,
            output_tokens=out,
            cache_read_tokens=cache_read,
            cache_creation_tokens=cache_creation,
            cost_usd=0.0,
            plan_usage=s.plan,
        )

    def _block(self, s: _Session, block: Mapping[str, Any]) -> dict[str, Any]:
        """One assistant block in Anthropic shape: bare tool names, the email
        scrubbed from text and thinking, the CLI's `caller` dropped."""
        kind = block.get("type")
        if kind == "text":
            return {"type": "text", "text": self._scrub(s, str(block.get("text", "")))}
        if kind == "thinking":
            out: dict[str, Any] = {
                "type": "thinking",
                "thinking": self._scrub(s, str(block.get("thinking", ""))),
            }
            if block.get("signature"):
                out["signature"] = block["signature"]
            return out
        if kind == "tool_use":
            return {
                "type": "tool_use",
                "id": str(block.get("id", "")),
                "name": bare_tool_name(str(block.get("name", ""))),
                "input": block.get("input"),
            }
        return dict(block)

    @staticmethod
    def _scrub(s: _Session, text: str) -> str:
        return text.replace(s.account_email, EMAIL_PLACEHOLDER) if s.account_email else text

    def _record_transcript(
        self, s: _Session, system: str, messages: list[dict[str, Any]], resp: ProviderResponse
    ) -> None:
        if self.transcript_sink is None:
            s.stdin_log.clear()
            return
        self.transcript_sink.record(
            url=f"claude-code://{s.session_id}",
            request_headers={},
            request_body={"system": system, "messages": messages, "stdin": list(s.stdin_log)},
            response_status=200,
            response_body={
                "id": resp.raw.get("id", ""),
                "role": "assistant",
                "model": s.resolved_model,
                "content": resp.raw.get("content", []),
                "stop_reason": resp.stop_reason,
                "usage": resp.raw.get("usage", {}),
            },
        )
        s.stdin_log.clear()
