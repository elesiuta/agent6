# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""File-based IPC between the workflow process and a front-end.

The workflow process and a front-end (the Textual TUI or the `agent6 web`
server) run as separate OS processes; the front-end just tails JSONL and
answers prompts by writing files. When an approval is needed:

1. The workflow process writes an `approval.prompt` event to logs.jsonl
   and then polls `<session_dir>/approvals/<id>.answer` for a result.
2. If a `<session_dir>/frontends/` claim points at a live process, the
   workflow process waits for the front-end to write the answer file.
   Otherwise it falls back to a plain stdin prompt.
3. The front-end (when present) presents a modal / control, then writes
   `<session_dir>/approvals/<id>.answer` containing the operator's literal
   choice: `yes`, `no`, `session` or `session-deny`. The asking side decides
   what a choice grants (see `Approver`); the front-end reports the click.

We use the filesystem rather than a socket because:
- the JSONL log is already the cross-process contract,
- the front-end may crash without taking the workflow down with it,
- every front-end mirrors the same files (the TUI, the web server, the ACP agent).

Answers are written with `atomic_write` (a unique temp file, fsync, rename):
the reader polls on existence and would consume a torn file as deny or "",
and two live front-ends answering one prompt must not share a temp name.
"""

from __future__ import annotations

import contextlib
import json
import os
import subprocess
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from agent6.events import EventSink
from agent6.paths import mkdir_for_real_user
from agent6.portable import atomic_write

APPROVAL_DIR_NAME = "approvals"
QUESTION_DIR_NAME = "questions"
FRONTENDS_DIR = "frontends"
WORKER_PID_FILE = "worker.pid"  # the run's worker process, for `agent6 sessions show` liveness
# The run's session-network holder. A separate process can only name a namespace
# through a live /proc entry, so `agent6 exec` and `agent6 forward` join through
# this pid. Absent when the run has no session network (a weaker isolation, or
# everything on the host network).
NETNS_PID_FILE = "netns.pid"
STEER_ANSWER_FILE = "steer.answer"

# How long the answer polls keep waiting after the front-end liveness gate goes
# dark before falling back headless (deny / ""). A transient drop (a phone
# locking its browser, a page reload, a web server restart) re-registers within
# seconds; without the grace, one 0.2s poll landing in that gap silently denied
# a pending approval. 30s outlasts a reload while a truly-gone front-end still
# fails over well before the answer timeout.
FRONTEND_DEAD_GRACE_S = 30.0


def approvals_dir(session_dir: Path) -> Path:
    """The approvals dir, created. A READ asks `approvals_path` instead: making
    the dir bumps the session dir's mtime, which the listings sort by."""
    p = session_dir / APPROVAL_DIR_NAME
    mkdir_for_real_user(p)
    return p


def approvals_path(session_dir: Path) -> Path:
    """Where the approvals dir would be; never created."""
    return session_dir / APPROVAL_DIR_NAME


def _contained(directory: Path, filename: str, *, untrusted: str, what: str) -> Path:
    """`<directory>/<filename>`, refusing a name that is not one plain file.

    Both bridge files name themselves after a string from outside: a prompt id
    the web server takes from the request, and an approval scope holding a
    server name parsed out of a tool name the LLM chose. A separator makes a
    directory of one of those, and `..` walks out of the run, so containment
    is a hard check on the write primitive rather than a caller's manners.
    """
    if not untrusted or "/" in untrusted or os.sep in untrusted or "\x00" in untrusted:
        raise ValueError(f"unsafe {what}: {untrusted!r}")
    target = directory / filename
    if not target.resolve().is_relative_to(directory.resolve()):
        raise ValueError(f"unsafe {what}: {untrusted!r}")
    return target


def _answer_path(directory: Path, answer_id: str) -> Path:
    return _contained(directory, f"{answer_id}.answer", untrusted=answer_id, what="answer id")


def clear_pending_answers(session_dir: Path, *, before: float) -> None:
    """Drop the previous leg's bridge state at a leg's START: its `*.answer`
    files (tidiness: a prompt clears its own slot before asking, so a stale
    answer is never read), its steer answer and marker (a phantom steer
    prompt no live front-end answers), its stop and compact markers (an
    instant re-stop or re-compact). Only files older than *before* (a
    timestamp) go: a marker written since belongs to the leg that is
    starting (an editor's cancel that landed while the run was coming up).
    A run passes its journal's last write (`SessionLayout.previous_leg_end`);
    a machine's crash recovery passes now, its per-state dir being this
    execution's own. Best-effort. Front-end claims need no sweep:
    `frontend_is_live` prunes dead ones on every probe, and a live watcher's
    must survive so its modals stay wired up."""
    answer_dirs = (session_dir / APPROVAL_DIR_NAME, session_dir / QUESTION_DIR_NAME)
    markers = (STEER_ANSWER_FILE, STEER_REQUEST_FILE, STOP_REQUEST_FILE, COMPACT_REQUEST_FILE)
    answers = (f for d in answer_dirs for f in d.glob("*.answer"))
    for path in (*answers, *(session_dir / name for name in markers)):
        with contextlib.suppress(OSError):
            if path.stat().st_mtime < before:
                path.unlink()


def register_frontend(session_dir: Path, pid: int) -> None:
    """Register *pid* as a live answering front-end: one claim file per
    front-end (`frontends/<pid>`), so any number can watch concurrently
    (web + TUI + attach, or several of one kind) and none can deregister
    another. The name is the claim; the body is the process start time, which
    is what tells a live front-end from a recycled pid (see
    :func:`frontend_is_live`)."""
    d = session_dir / FRONTENDS_DIR
    mkdir_for_real_user(d)
    (d / str(pid)).write_text(_proc_start_time(pid), encoding="utf-8")


def unregister_frontend(session_dir: Path, pid: int) -> None:
    """Drop *pid*'s own claim; other front-ends' claims are untouched."""
    with contextlib.suppress(OSError):
        (session_dir / FRONTENDS_DIR / str(pid)).unlink()


def pid_alive(pid: int) -> bool:
    """True iff a live process WE OWN has *pid* (signal 0 probes without killing).

    PermissionError reads as DEAD: agent6's workers and front-ends are always
    spawned by the same user that later probes them, so a foreign-owned pid
    can only mean the original process died and the kernel reused the number
    for another user's process; reading that pid as live would render a dead
    run "running" forever and hang the /parallel lane await."""
    try:
        os.kill(pid, 0)
    except (ProcessLookupError, PermissionError, OSError):
        return False
    return True


# /proc exists on Linux; on macOS `ps` answers the same question instead.
_HAS_PROC = Path("/proc").is_dir()


def _ps_start_time(pid: int) -> str:
    """Start-time identity via `ps -o lstart=` ("" for a dead pid or a host
    without ps). Fixed argv over a pid agent6 itself recorded, never LLM
    output; see the subprocess allowlist in docs/security.md."""
    try:
        proc = subprocess.run(
            ["ps", "-p", str(pid), "-o", "lstart="],
            capture_output=True,
            check=False,
            timeout=5.0,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    if proc.returncode != 0:
        return ""
    return proc.stdout.decode(errors="replace").strip()


def _proc_start_time(pid: int) -> str:
    """Start-time identity for *pid*, or "" when it cannot be read (the
    process just exited): field 22 of /proc/<pid>/stat on Linux, `ps` where
    /proc is absent (macOS -- whose small pid_max recycles pids fast, so the
    plain kill-0 probe misread reuse as liveness there too). The comm field
    may contain spaces/parens, so split after the LAST ')'."""
    if not _HAS_PROC:
        return _ps_start_time(pid)
    try:
        stat = Path(f"/proc/{pid}/stat").read_text(encoding="ascii", errors="replace")
    except OSError:
        return ""
    rest = stat.rpartition(")")[2].split()
    return rest[19] if len(rest) > 19 else ""


def write_session_netns_pid(session_dir: Path, pid: int) -> None:
    """Publish the holder of this run's session network, for `agent6 exec`."""
    atomic_write(session_dir / NETNS_PID_FILE, pid_record(pid))


def read_session_netns_pid(session_dir: Path) -> int | None:
    """The live holder of this run's session network, or None.

    A pid whose /proc entry is gone is a run that ended (or never had one), not
    a network to join. Nor is a pid the kernel has since handed to someone else:
    the recorded start time settles that: joining on liveness alone would put
    `agent6 exec` and `agent6 forward` inside an unrelated process's namespaces
    while calling them the run's.
    """
    rec = _parse_pid_record(session_dir / NETNS_PID_FILE)
    if rec is None:
        return None
    pid, recorded_start = rec
    if pid <= 0 or not Path(f"/proc/{pid}/ns/net").exists():
        return None
    if recorded_start and _proc_start_time(pid) != recorded_start:
        return None
    return pid


def listening_ports(session_dir: Path) -> list[int]:
    """The TCP ports something in the run is listening on, or [].

    Read from `/proc/<holder>/net/`, which is that process's OWN view: a
    namespace's sockets are readable without entering it, so this needs no
    fork and no setns (a forking version would warn under any threaded caller,
    the web server included). Every surface's "serving" line reads this.
    """
    pid = read_session_netns_pid(session_dir)
    if pid is None:
        return []
    ports: set[int] = set()
    for name in ("tcp", "tcp6"):
        with contextlib.suppress(OSError):
            for line in Path(f"/proc/{pid}/net/{name}").read_text(encoding="utf-8").splitlines():
                cols = line.split()
                # st == 0A is LISTEN; the local address is host:port in hex.
                if len(cols) > 3 and cols[3] == "0A" and ":" in cols[1]:
                    ports.add(int(cols[1].rsplit(":", 1)[1], 16))
    return sorted(ports)


def clear_session_netns_pid(session_dir: Path) -> None:
    with contextlib.suppress(OSError):
        (session_dir / NETNS_PID_FILE).unlink()


def write_worker_pid(session_dir: Path, pid: int) -> None:
    """Record the session's worker pid so `agent6 sessions show` can probe liveness even
    while the worker is blocked in a long provider call (no events emitted).
    The start-time identity rides along after the pid (/proc ticks on Linux,
    `ps` lstart text elsewhere) so a recycled pid -- same number, different
    process, after a SIGKILL'd worker left the file behind -- cannot make a
    dead run read running forever (blocking resume and the /parallel lane
    await)."""
    # Atomic like every sibling publish: a plain write truncates first, so a
    # reader in that window sees a PREFIX of the pid with the identity stripped
    # -- and a prefix naming a live process you own reads alive with nothing
    # left to refute it, the exact recycled-pid lie this record exists to kill.
    atomic_write(session_dir / WORKER_PID_FILE, pid_record(pid))


def emit_session_start(
    events: EventSink, session_dir: Path, event_type: str, /, **fields: Any
) -> None:
    """Emit a start-family event (`session.start` / `loop.resume.start`)
    with the worker pid already on disk: the status fold reads a started
    session with no pid file as one whose worker exited."""
    write_worker_pid(session_dir, os.getpid())
    events.emit(event_type, **fields)


def clear_worker_pid(session_dir: Path) -> None:
    with contextlib.suppress(FileNotFoundError):
        (session_dir / WORKER_PID_FILE).unlink()


def pid_record(pid: int) -> str:
    """A pid plus the identity that distinguishes it from a later reuse of the
    same number. Every published pid uses this: liveness alone is not identity,
    and the kernel hands the number on."""
    return f"{pid} {_proc_start_time(pid)}".rstrip()


def _parse_pid_record(path: Path) -> tuple[int, str] | None:
    """The recorded `(pid, start_time)`; start_time is "" when none was
    recorded. Split once only: the `ps` lstart identity contains spaces."""
    try:
        tokens = path.read_text(encoding="utf-8").split(maxsplit=1)
        return int(tokens[0]), tokens[1].strip() if len(tokens) > 1 else ""
    except (OSError, ValueError, IndexError):
        return None


def _read_pid_record(session_dir: Path) -> tuple[int, str] | None:
    return _parse_pid_record(session_dir / WORKER_PID_FILE)


def read_worker_pid(session_dir: Path) -> int | None:
    rec = _read_pid_record(session_dir)
    return None if rec is None else rec[0]


def worker_is_alive(session_dir: Path) -> bool:
    """True iff worker.pid points at a live process that IS the recorded worker:
    the pid is alive AND, when a start time was recorded, today's start time
    matches. A recycled pid fails the match and reads dead."""
    rec = _read_pid_record(session_dir)
    if rec is None:
        return False
    pid, recorded_start = rec
    return _still_the_process(pid, recorded_start)


def _still_the_process(pid: int, recorded_start: str) -> bool:
    """Alive AND, when a start time was recorded, still the process that
    recorded it (a recycled pid fails the match). Liveness alone is not
    identity: a front-end that died and had its pid reused by another process
    of ours would read live forever, and an approval would then wait out its
    whole timeout instead of the dead-grace, the stall away-mode exists to
    avoid. No recorded start time is trusted. `os.kill(0, 0)` probes the
    process GROUP and `os.kill(-1, 0)` every process, so both answer alive:
    0 and -1 are not pids."""
    if pid <= 0 or not pid_alive(pid):
        return False
    return not recorded_start or _proc_start_time(pid) == recorded_start


def _claim_is_live(claim: Path, pid: int) -> bool:
    """Whether *claim* still names the front-end that wrote it (the worker's
    own test, `_still_the_process`, over the start time the claim recorded)."""
    try:
        recorded_start = claim.read_text(encoding="utf-8").strip()
    except OSError:
        return False
    return _still_the_process(pid, recorded_start)


def effective_away(session_dir: Path) -> str:
    """This run's away answer: the env a launcher set, else the one recorded on
    the run dir.

    THE one owner: a run detached from a terminal (or spawned by the hub)
    carries its operator's choice in `approvals/away.mode`, so the preflight
    and the approver read the env and the file the same way."""
    return os.environ.get("AGENT6_DETACHED_AWAY", "") or away_mode(session_dir)


def frontend_is_live(session_dir: Path) -> bool:
    """True when ANY registered front-end is a live process agent6 registered. Prunes
    dead claims (hard-killed front-ends, and pids since reused by something
    else) in passing so a stale claim can never block the answer poll and the
    dir stays tidy."""
    try:
        entries = list((session_dir / FRONTENDS_DIR).iterdir())
    except OSError:
        return False
    live = False
    for f in entries:
        try:
            pid = int(f.name)
        except ValueError:
            pid = -1
        if _claim_is_live(f, pid):
            live = True
        else:
            with contextlib.suppress(OSError):
                f.unlink()
    return live


def _consume_answer(target: Path) -> str | None:
    """Read + delete *target* (consume, so it is never re-read on a later
    prompt/resume), or None when absent."""
    try:
        txt = target.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    with contextlib.suppress(FileNotFoundError):
        target.unlink()
    return txt


def _await_answer(
    target: Path, live: Path, *, timeout_s: float, poll_s: float, dead_grace_s: float
) -> str | None:
    """Poll for *target*, consume it, and return its text.

    Returns None when the front-end registered on *live* stays dead for
    *dead_grace_s* consecutive seconds (see FRONTEND_DEAD_GRACE_S) or when
    *timeout_s* elapses. A file that vanishes between polls is not-yet-answered,
    never an error. A final consume runs before either None verdict, so an answer landing
    between the round's read and the verdict is honoured rather than denied
    with its file left on disk."""
    deadline = time.monotonic() + timeout_s
    dead_since: float | None = None
    while time.monotonic() < deadline:
        if (txt := _consume_answer(target)) is not None:
            return txt
        if frontend_is_live(live):
            dead_since = None
        else:
            now = time.monotonic()
            if dead_since is None:
                dead_since = now
            if now - dead_since >= dead_grace_s:
                break
        time.sleep(poll_s)
    return _consume_answer(target)


def await_frontend_reply[T](session_dir: Path, read_once: Callable[[], T | None]) -> T | None:
    """Detach 'wait' mode: block until an answer arrives or a Stop ends the run.

    `read_once` is called even with NO front-end claim registered: a
    claim-less front-end (the web UI answering over HTTP) writes the same
    answer files, and the answer's existence, not a claim, is the proof
    someone answered. `read_once` paces itself (its liveness dead-grace
    caps a claim-less round); the extra sleep paces the no-claim loop. A
    front-end's Stop lands as a steer abort and `sessions stop` as the stop
    marker; either breaks the wait so the run can end (a run parked in a
    pre-start question has no step to stop after: the empty reply parks
    it). Returns the reply, or None on stop."""
    while True:
        if steer_answer_is_abort(session_dir) or stop_request_pending(session_dir):
            return None
        reply = read_once()
        if reply is not None:
            return reply
        if not frontend_is_live(session_dir):
            time.sleep(1.0)


def write_answer(session_dir: Path, prompt_id: str, answer: str) -> None:
    """Called by a front-end (TUI or web) with the operator's literal choice:
    "yes", "no", "session" or "session-deny"."""
    target = _answer_path(approvals_dir(session_dir), prompt_id)
    atomic_write(target, answer)


def answer_written(session_dir: Path, prompt_id: str) -> bool:
    """Whether an answer for *prompt_id* is on disk: a peek, nothing consumed.
    The terminal prompt polls it, so an answer written by another route ends
    the prompt instead of waiting behind it."""
    return _answer_path(approvals_path(session_dir), prompt_id).exists()


def clear_answer(session_dir: Path, prompt_id: str) -> None:
    """Drop any pre-existing answer for *prompt_id* so an answer written BEFORE
    the prompt was emitted is never consumed. Prompt ids are deterministic
    sequential counters (approval-1, ...), so a front-end (or a hostile POST)
    could pre-write approvals/approval-1.answer and the run would silently
    honor it the moment it reached that approval, auto-approving a command the
    operator never saw. The run process clears the slot immediately before
    emitting the prompt (it alone knows the exact emit moment); a legitimate
    answer is only ever written after the front-end renders the prompt, so
    none is lost. Mirrors clear_steer_answer for the steer bridge."""
    with contextlib.suppress(OSError):
        _answer_path(approvals_dir(session_dir), prompt_id).unlink(missing_ok=True)


def clear_question_answers(session_dir: Path, question_id: str) -> None:
    """The ask_user analogue of :func:`clear_answer`: drop a pre-written answer
    for *question_id* before its prompt is emitted."""
    with contextlib.suppress(OSError):
        _answer_path(questions_dir(session_dir), question_id).unlink(missing_ok=True)


# "Allow (or deny) for the rest of the session": one marker file per SCOPE,
# checked before every prompt in that scope. A scope is what the operator was
# answering about, so a standing answer grants what the prompt said and no more.
# The whole vocabulary: the three command tools share one, and each MCP server
# has its own (server names are [A-Za-z0-9_-]+, so a scope is always a safe file
# suffix and two servers never collide).
#
# Markers are NOT `*.answer`s, so clear_pending_answers leaves them in place:
# the choice persists across a detached run's resumes (it keeps going without a
# front-end to prompt), and an interactive start drops the allow markers with
# the away-mode (`clear_session_grants`). They live in the run's approvals dir,
# so other runs are unaffected and a fresh run prompts again.
COMMAND_SCOPE = "command"
MCP_SCOPE_PREFIX = "mcp."
SESSION_ALLOW_FILE = "session.allow"
SESSION_DENY_FILE = "session.deny"


def _marker_path(session_dir: Path, stem: str, scope: str) -> Path:
    """Where one scope's marker lives; the dir is created by the WRITERS, so a
    probe (`session_allow_set`) never makes one."""
    return _contained(
        approvals_path(session_dir), f"{stem}.{scope}", untrusted=scope, what="approval scope"
    )


def set_session_allow(session_dir: Path, scope: str) -> None:
    """Record the operator's 'allow all of *scope* for the session' choice."""
    target = _marker_path(session_dir, SESSION_ALLOW_FILE, scope)
    mkdir_for_real_user(target.parent)
    atomic_write(target, "1")


def session_allow_set(session_dir: Path, scope: str) -> bool:
    return _marker_path(session_dir, SESSION_ALLOW_FILE, scope).exists()


def set_session_deny(session_dir: Path, scope: str) -> None:
    """Record the mirror choice: 'none of *scope* for the rest of the session'.

    A single "no" answers one call, exactly as a single "yes" approves one; only
    the session choices persist. Denying for the session WITHDRAWS the tools
    rather than refusing each call, so the model stops spending turns on a door
    that will not open.
    """
    target = _marker_path(session_dir, SESSION_DENY_FILE, scope)
    mkdir_for_real_user(target.parent)
    atomic_write(target, "1")


def session_deny_set(session_dir: Path, scope: str) -> bool:
    return _marker_path(session_dir, SESSION_DENY_FILE, scope).exists()


def record_answer(session_dir: Path, answer: str, scope: str | None) -> bool:
    """Apply the operator's literal *answer* and return the verdict for THIS call.

    The one place an answer's meaning is decided. A session choice persists only
    when the prompt offered one: `scope=None` is a gate with no standing answer
    (`fetch`), and an "allow all" arriving on one anyway grants nothing beyond
    the call it was clicked on. Anything unrecognised is a deny, so a truncated
    or hand-written answer file cannot approve.
    """
    if scope:
        if answer == "session":
            set_session_allow(session_dir, scope)
        elif answer == "session-deny":
            set_session_deny(session_dir, scope)
    return answer in {"yes", "session"}


def effective_run_commands(configured: str, session_dir: Path) -> str:
    """What the command policy IS right now: "no" | "ask" | "yes".

    One answer from three inputs, so every consumer agrees: the configured
    knob, the operator's session choice, and the away-mode a detached run was
    left with. Only "ask" is movable -- a configured "yes" or "no" is the
    operator's standing policy and no in-run choice overrides it.

    "no" means the tools are WITHDRAWN, not refused per call: that is the same
    wiring for `run_commands = "no"`, `--no-commands`, deny-for-session and an
    away-mode of "deny", so the rules fall out consistently.
    """
    if configured != "ask":
        return configured
    if session_allow_set(session_dir, COMMAND_SCOPE):
        return "yes"
    if session_deny_set(session_dir, COMMAND_SCOPE) or away_mode(session_dir) == "deny":
        return "no"
    return "ask"


# How a DETACHED run (no terminal to prompt) handles run_command approvals and
# ask_user questions: "deny" auto-denies, "wait" blocks until a front-end
# reattaches and answers. "approve" is not stored here -- detach approve-all
# sets the command scope's allow marker. Persists like it (not an *.answer).
AWAY_MODE_FILE = "away.mode"
# What an operator may set AGENT6_DETACHED_AWAY to. "approve" is not stored in
# away.mode: like the interactive detach prompt it sets an allow marker per
# scope. Anything else is a typo, and a typo must not read as "an absent
# operator's intent is known", which would lift the preflight refusal and
# leave the run waiting forever at the first approval.
AWAY_MODES = ("wait", "deny", "approve")


def set_away_mode(session_dir: Path, mode: str) -> None:
    """Record the detach 'while away' choice ("deny" | "wait")."""
    if mode not in ("deny", "wait"):
        raise ValueError(
            f"away.mode is 'deny' or 'wait', got {mode!r} (approve-all reuses session.allow)"
        )
    atomic_write(approvals_dir(session_dir) / AWAY_MODE_FILE, mode)


def away_mode(session_dir: Path) -> str:
    """ "deny", "wait", or "" (unset -- interactive/foreground default flow)."""
    try:
        return (approvals_path(session_dir) / AWAY_MODE_FILE).read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def clear_away_mode(session_dir: Path) -> None:
    """Drop the detach 'while away' choice. Called when an INTERACTIVE (tty) run or
    resume starts: the operator is back at the terminal, so a stale away-mode from a
    prior detach must not keep auto-denying/waiting."""
    with contextlib.suppress(FileNotFoundError):
        (approvals_path(session_dir) / AWAY_MODE_FILE).unlink()


def clear_session_grants(session_dir: Path) -> None:
    """Drop every per-scope approve-all grant (`record_answer`'s "allow all of
    this scope", the detach's answers among them) beside the away-mode: they
    expire together when the operator is back at a terminal. `--auto-approve`
    is the grant that stays."""
    approvals = approvals_path(session_dir)
    if approvals.is_dir():
        for marker in approvals.glob(f"{SESSION_ALLOW_FILE}.*"):
            with contextlib.suppress(FileNotFoundError):
                marker.unlink()


def read_answer(
    session_dir: Path,
    prompt_id: str,
    *,
    timeout_s: float = 600.0,
    poll_s: float = 0.2,
    live_dir: Path | None = None,
    dead_grace_s: float = FRONTEND_DEAD_GRACE_S,
) -> str | None:
    """Called by the workflow. Returns the operator's literal choice ("yes",
    "no", "session", "session-deny"), or None on timeout or once the front-end
    has stayed dead past `dead_grace_s` (a shorter drop keeps waiting).

    `live_dir` overrides which dir the liveness gate probes for front-end claims
    (defaults to `session_dir`). A machine agent state reads answers from its
    per-state dir but the front-end registers on the instance dir, so it passes
    the instance dir here."""
    target = _answer_path(approvals_dir(session_dir), prompt_id)
    txt = _await_answer(
        target,
        live_dir or session_dir,
        timeout_s=timeout_s,
        poll_s=poll_s,
        dead_grace_s=dead_grace_s,
    )
    return None if txt is None else txt.strip().lower()


# --- agent->user question bridge (the `ask_user` tool) -----------------------
# Same shape as approvals, but the answer is a free string (a selected option or
# typed text). The workflow emits `question.prompt`, polls for the answer file;
# the TUI shows a modal and writes it. Falls back to stdin (then a default) when
# no TUI is live, so headless runs never hang.


def questions_dir(session_dir: Path) -> Path:
    p = session_dir / QUESTION_DIR_NAME
    mkdir_for_real_user(p)
    return p


def write_question_answers(session_dir: Path, question_id: str, answers: Sequence[str]) -> None:
    """Called by a front-end when the user answers the question(s). Answers align to
    the prompt's `questions` by index and are stored as a JSON list."""
    atomic_write(_answer_path(questions_dir(session_dir), question_id), json.dumps(list(answers)))


def question_answers_written(session_dir: Path, question_id: str) -> bool:
    """The `ask_user` analogue of :func:`answer_written`."""
    return _answer_path(session_dir / QUESTION_DIR_NAME, question_id).exists()


def read_question_answers(
    session_dir: Path,
    question_id: str,
    *,
    timeout_s: float = 600.0,
    poll_s: float = 0.2,
    live_dir: Path | None = None,
    dead_grace_s: float = FRONTEND_DEAD_GRACE_S,
) -> tuple[str, ...] | None:
    """Called by the workflow. Returns the answers tuple (aligned to the prompt's
    questions), or None on timeout or once the front-end has stayed dead past
    `dead_grace_s`. `live_dir` overrides the liveness-gate dir (see
    :func:`read_answer`)."""
    target = _answer_path(questions_dir(session_dir), question_id)
    raw = _await_answer(
        target,
        live_dir or session_dir,
        timeout_s=timeout_s,
        poll_s=poll_s,
        dead_grace_s=dead_grace_s,
    )
    if raw is None:
        return None
    try:
        data = json.loads(raw)
    except ValueError:
        return (raw,)  # a bare free-text answer (not JSON) -> single answer
    return tuple(str(x) for x in data) if isinstance(data, list) else (str(data),)


# --- mid-run steering bridge (Ctrl-C while the TUI owns the terminal) --------
# Single-slot: only one steer prompt is ever outstanding (the SIGINT handler
# sets a flag the loop drains at its next boundary). The run process triggers a
# steer by emitting `session.steer_requested`; the TUI shows a modal and writes the
# answer here; the run process reads it. The answer is a free string:
# "" = continue, "abort" = stop, anything else = a steering instruction.


def write_steer_answer(session_dir: Path, answer: str) -> None:
    """Called by a front-end when the user answers the steer prompt."""
    atomic_write(session_dir / STEER_ANSWER_FILE, answer)


def clear_steer_answer(session_dir: Path) -> None:
    with contextlib.suppress(FileNotFoundError):
        (session_dir / STEER_ANSWER_FILE).unlink()


def take_steer_answer(session_dir: Path) -> str | None:
    """Consume a steer answer that is already on disk (a `resume --steer`
    seed, an end-of-session follow-up, a front-end's answer that landed before
    the boundary), or None: the tty prompt asks nothing it was already told."""
    return _consume_answer(session_dir / STEER_ANSWER_FILE)


def steer_answer_written(session_dir: Path) -> bool:
    """Whether a steer answer is on disk: a peek, nothing consumed. The pause
    menu polls it, so a steer sent from a front-end while the menu is open
    ends the menu instead of waiting behind it."""
    return (session_dir / STEER_ANSWER_FILE).exists()


def steer_answer_is_abort(session_dir: Path) -> bool:
    """Non-blocking peek: True if a pending steer answer is a stop. Lets a long
    streaming model turn bail immediately instead of only at the between-step
    boundary. Does NOT consume the answer -- the boundary still handles it if the
    stream ends first."""
    try:
        answer = (session_dir / STEER_ANSWER_FILE).read_text(encoding="utf-8").strip().lower()
    except (OSError, ValueError):  # missing/unreadable, or non-UTF-8: not an abort
        return False
    # Exactly the Stop contract: every front-end's Stop writes "abort", and the
    # between-step boundary (_maybe_handle_steer) also stops only on "abort". A
    # typed steer instruction -- even the word "stop" -- is an instruction, not a
    # stop; interrupting mid-stream on it would diverge from the boundary.
    return answer == "abort"


# A steer can also be INITIATED from the TUI (the `s` key) without Ctrl-C: the
# dashboard drops this marker, the run notices it at its next safe boundary (same
# as the SIGINT flag), prompts via the modal, and clears it. Decoupled from
# signals so a watcher process can request a steer the run picks up.
STEER_REQUEST_FILE = "steer.request"


def request_steer(session_dir: Path, *, now: bool = False) -> None:
    """Drop the steer marker the session polls. The default is consumed at
    the next step boundary; `now=True` writes the urgency into the marker and
    the loop aborts the in-flight model call to take it."""
    with contextlib.suppress(OSError):
        (session_dir / STEER_REQUEST_FILE).write_text("now" if now else "", encoding="utf-8")


def steer_request_pending(session_dir: Path) -> bool:
    return (session_dir / STEER_REQUEST_FILE).exists()


def steer_interrupt_pending(session_dir: Path) -> bool:
    """A pending steer whose marker carries the `now` urgency: only this
    aborts an in-flight model call; a plain steer waits for the boundary
    (aborting wastes the streamed tokens and the step's partial work)."""
    try:
        return (session_dir / STEER_REQUEST_FILE).read_text(encoding="utf-8").strip() == "now"
    except OSError:
        return False


def submit_steer(session_dir: Path, text: str, *, now: bool = False) -> None:
    """Queue *text* as the session's next steer (a front-end composer, a
    `--steer` seed): the answer lands before the request marker, so the loop
    finds it the moment it notices the request and never waits on a modal."""
    write_steer_answer(session_dir, text)
    request_steer(session_dir, now=now)


def clear_steer_request(session_dir: Path) -> None:
    with contextlib.suppress(FileNotFoundError):
        (session_dir / STEER_REQUEST_FILE).unlink()


STOP_REQUEST_FILE = "stop.request"


def request_stop(session_dir: Path) -> bool:
    """Front-end "stop after this step": drop a marker the session polls at each
    completed-iteration boundary and honors by ending the run cleanly there
    (the finished step's tool results and auto-commit land first). The
    immediate stop stays the steer "abort" answer, which interrupts mid-turn.

    Returns whether the marker landed, the `request_compact` rule: a failed
    write neither raises into a front-end action nor reads as a stop nothing
    will honor."""
    try:
        (session_dir / STOP_REQUEST_FILE).write_text("", encoding="utf-8")
    except OSError:
        return False
    return True


def stop_request_pending(session_dir: Path) -> bool:
    return (session_dir / STOP_REQUEST_FILE).exists()


def clear_stop_request(session_dir: Path) -> None:
    with contextlib.suppress(FileNotFoundError):
        (session_dir / STOP_REQUEST_FILE).unlink()


COMPACT_REQUEST_FILE = "compact.request"


def request_compact(session_dir: Path, focus: str = "") -> bool:
    """Front-end-initiated manual compaction: drop a marker the session polls at its
    next safe boundary and honors by forcing a context compaction (mirrors
    steer). The marker body is the operator's optional summary *focus*
    (`/compact <focus>`); "" is a plain compact. Published atomically: the run
    polls `read_compact_request` every boundary, so a plain write exposed an
    empty/partial focus it consumed (and then cleared) as the real one.

    Returns whether the marker landed. A failed write must not raise into a TUI
    action or a web handler, and must not read as success either: on a
    read-only or full state dir an unconditional "compaction requested" would
    be a claim nothing ever honors."""
    try:
        atomic_write(session_dir / COMPACT_REQUEST_FILE, focus)
    except OSError:
        return False
    return True


def read_compact_request(session_dir: Path) -> str | None:
    """The pending compact request's focus text, or None when no request is
    pending ("" = a plain compact with no focus)."""
    try:
        return (session_dir / COMPACT_REQUEST_FILE).read_text(encoding="utf-8")
    except OSError:
        return None


def clear_compact_request(session_dir: Path) -> None:
    with contextlib.suppress(FileNotFoundError):
        (session_dir / COMPACT_REQUEST_FILE).unlink()


def read_steer_answer(session_dir: Path, *, live_dir: Path | None = None) -> str | None:
    """Called by the workflow when a front-end is live. Returns the answer
    string (consuming the file), or None after ten minutes or once the
    front-end has stayed dead past `FRONTEND_DEAD_GRACE_S`. `live_dir`
    overrides the liveness-gate dir (see :func:`read_answer`)."""
    return _await_answer(
        session_dir / STEER_ANSWER_FILE,
        live_dir or session_dir,
        timeout_s=600.0,
        poll_s=0.2,
        dead_grace_s=FRONTEND_DEAD_GRACE_S,
    )
