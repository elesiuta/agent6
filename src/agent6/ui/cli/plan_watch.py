# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""`agent6 plan`/`agent6 attach` and run-id resolution helpers."""

from __future__ import annotations

import contextlib
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

from agent6.config.layer import resolved_state_dir
from agent6.errors import read_operator_file
from agent6.sessions.id import SessionIdError, resolve_session_id
from agent6.sessions.ipc import (
    register_frontend,
    unregister_frontend,
    worker_is_alive,
    write_answer,
    write_question_answers,
)
from agent6.sessions.layout import LOGS_NAME
from agent6.tools.schema import UserQuestion
from agent6.ui.cli._common import (
    _plans_dir,
    print_nothing_yet,
    resolve_or_newest_layout,
)
from agent6.ui.cli._console_view import ConsoleView
from agent6.ui.cli._interact import default_stdin_approver, default_stdin_questioner
from agent6.viewmodel import (
    StatusFacts,
    event_epoch,
    scan_session_log,
    session_is_live,
    session_mtime,
    session_policy,
    status_for_session_dir,
    tail_events,
)
from agent6.viewmodel.events import SESSION_START_EVENTS


def _resolve_plan_session_id(session_id: str) -> str | None:
    """Resolve a (possibly prefix) plan id under the per-repo state dir.

    Prints an error and returns None on failure. Used by `run --from`,
    `plan show`, and `plan edit`. An empty *session_id* resolves the most recent
    plan, matching the omit-for-latest convention of the sessions commands.
    """
    plans_dir = _plans_dir(Path.cwd())
    if not session_id:
        latest = _most_recent_plan_session_id(plans_dir)
        if latest is None:
            print("ERROR: no plans yet (start one with `agent6 plan`).", file=sys.stderr)
            return None
        session_id = latest
    try:
        resolved = resolve_session_id(plans_dir, session_id)
    except SessionIdError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return None
    plan = plans_dir / resolved / "plan.md"
    if not plan.is_file():
        print(
            f"ERROR: {resolved} has no plan.md (was it created with `agent6 plan`?)",
            file=sys.stderr,
        )
        return None
    return resolved


def _cmd_plan_show(session_id: str) -> int:
    """Print a planning run's plan.md to stdout."""
    resolved = _resolve_plan_session_id(session_id)
    if resolved is None:
        return 2
    sys.stdout.write(read_operator_file(_plans_dir(Path.cwd()) / resolved / "plan.md"))
    return 0


def _cmd_plan_edit(session_id: str) -> int:
    """Open a planning run's plan.md in $EDITOR (default: vi).

    Operator-controlled argv (the editor name + the resolved plan path),
    not LLM-controlled, so direct subprocess.run is allowed.
    """
    resolved = _resolve_plan_session_id(session_id)
    if resolved is None:
        return 2
    plan = _plans_dir(Path.cwd()) / resolved / "plan.md"
    editor = os.environ.get("EDITOR", "vi")
    # $EDITOR may be a command with flags ("code --wait"); split it.
    argv = shlex.split(editor) or ["vi"]
    try:
        result = subprocess.run([*argv, str(plan)], check=False)
    except OSError as exc:
        print(f"ERROR: failed to spawn editor {editor!r}: {exc}", file=sys.stderr)
        return 1
    return result.returncode


def _most_recent_plan_session_id(plans_dir: Path) -> str | None:
    """Most recently active plan dir that holds a `plan.md`.

    Used by bare `agent6 run` (no task) to offer the latest plan for execution.
    """
    if not plans_dir.is_dir():
        return None
    candidates = sorted(
        (p for p in plans_dir.iterdir() if p.is_dir() and (p / "plan.md").is_file()),
        key=session_mtime,
        reverse=True,
    )
    return candidates[0].name if candidates else None


def _cmd_watch(
    session_id: str,
    *,
    tui: bool = False,
    since: int = 0,
    raw: bool = False,
    config_path: Path | None = None,
) -> int:
    """Read-only live view of a run directory.

    Default follows the run's conversation (the same render as `agent6 run`).
    `--raw` is the no-deps event-line tail; `--tui` the full-screen dashboard.
    """
    cwd = Path.cwd()
    # An explicit id resolves across every run-style bucket (runs/asks/machine-
    # drafts): a listed ask or a `machine create` draft is watchable by id too.
    # Empty most-recent spans every bucket, so a bare `attach` after an `ask`
    # finds it.
    try:
        layout = resolve_or_newest_layout(cwd, session_id)
    except SessionIdError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    if layout is None:
        print_nothing_yet()
        return 2
    target = layout.session_dir
    if not session_id:
        print(f"[agent6] attached to most recent run: {target.name}", file=sys.stderr)
    if not target.is_dir():
        print(f"ERROR: no such run dir: {target}", file=sys.stderr)
        return 2
    if not tui:
        return _cmd_watch_plain(target, since=since) if raw else _watch_transcript(target)
    try:
        from agent6.ui.tui.app import run_tui  # noqa: PLC0415 - lazy: textual is optional
    except ImportError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        print(
            "HINT: drop --tui for a no-deps text tail of logs.jsonl.",
            file=sys.stderr,
        )
        return 3
    return run_tui(target, config_path=config_path)


def _resolve_session_dir(repo_root: Path, session_id: str) -> Path | None:
    """Resolve a run id (or the most-recent run when empty) to its run dir.

    An explicit id resolves across every run-style bucket (runs/, asks/,
    sessions/machines/): anything `agent6 sessions` lists must also be inspectable
    by id. The empty (most-recent) case also spans every bucket, so a bare
    `attach` right after an `ask` finds that ask."""
    try:
        layout = resolve_or_newest_layout(repo_root, session_id)
    except SessionIdError:
        return None
    return layout.session_dir if layout is not None else None


def _cmd_tui(config_path: Path | None = None) -> int:
    """The TUI hub (`agent6 tui`): browse runs and start new work. Loops between
    the home screen and the run view (the conversation; Ctrl+D toggles the
    dashboard), opening a run watches it, then returns here on close."""
    try:
        from agent6.ui.tui.app import (  # noqa: PLC0415 - lazy: textual optional
            QUIT_HUB_CODE,
            run_tui,
        )
        from agent6.ui.tui.home import run_home  # noqa: PLC0415
    except ImportError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        print("HINT: the TUI needs 'textual' (part of the base install).", file=sys.stderr)
        return 3
    cwd = Path.cwd()
    # The STATE dir: every bucket lookup below it goes through `bucket_dir`,
    # which appends `sessions/` itself.
    agent6_dir = resolved_state_dir(cwd)
    while True:
        session_dir = run_home(agent6_dir, cwd, config_path)
        if session_dir is None:
            return 0
        # Esc in the dashboard returns here (reopen home); q quits the hub.
        if run_tui(session_dir, from_hub=True, config_path=config_path) == QUIT_HUB_CODE:
            return 0


def format_plain_event(line: str, *, session_start_ts: float | None) -> str:
    """Pretty-print one logs.jsonl line as `<elapsed> <type> key=val ...`.

    Falls back to the raw line on parse error so a corrupt event doesn't
    abort the tail. `session_start_ts` is the wall-clock timestamp of the
    earliest event seen so far; used to render relative elapsed seconds.
    """
    raw = line.rstrip("\n")
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        return raw
    if not isinstance(obj, dict):
        return raw
    ts = event_epoch(obj.get("ts"))
    event = obj.get("event") or obj.get("type") or "?"
    if ts is not None and session_start_ts is not None:
        elapsed = max(0.0, ts - session_start_ts)
        ts_str = f"+{elapsed:7.1f}s"
    else:
        ts_str = "        "
    skip = {"ts", "event", "type", "session_id"}
    pairs: list[str] = []
    for k, v in obj.items():
        if k in skip:
            continue
        if isinstance(v, str):
            shown = v if len(v) <= 80 else v[:77] + "..."
            pairs.append(f"{k}={shown!r}")
        elif isinstance(v, (int, float, bool)) or v is None:
            pairs.append(f"{k}={v}")
        else:
            blob = json.dumps(v, default=str)
            shown = blob if len(blob) <= 80 else blob[:77] + "..."
            pairs.append(f"{k}={shown}")
    return f"{ts_str} {event:30s} {' '.join(pairs)}"


class _CliFrontEnd:
    """Makes an interactive `agent6 attach` a real run FRONT-END, not just a
    reader. When the streamed log surfaces an unanswered `run_command` approval
    or `ask_user` question, it prompts on the controlling terminal with the SAME
    CLI prompts a foreground run uses and writes the answer back over the file
    bridge -- so watching a detached run is "as if you never detached". The
    caller registers a `frontends/` claim so the worker's approver bridges to it (a
    live front-end always wins over the detach away-mode).

    Prompt ids are deterministic counters, and the log replays from the start on
    attach, so `_answered` (ids with an answer seen) and `_handled` (ids WE
    prompted for) gate re-prompting a historical or already-answered prompt."""

    def __init__(self, session_dir: Path, view: ConsoleView) -> None:
        self._session_dir = session_dir
        self._view = view
        self._answered: set[str] = set()
        self._handled: set[str] = set()
        # Events the attach pre-scan already decided. The follow loop re-reads
        # logs.jsonl FROM THE START, so it hands those same events back to
        # `react`; replaying them must never prompt (see `react`). A count is
        # enough because logs.jsonl is append-only: the follow can only deliver
        # MORE events than the scan saw, never fewer, and the scanned ones
        # always arrive first.
        self._replayed: int = 0

    def open_prompts_at_attach(self, events_path: Path) -> list[tuple[str, str, object]]:
        """Pre-scan the existing log: seed `_answered` and return the prompts
        that are open right now (emitted, not answered) so a run already waiting
        at an approval when you attach is handled at once."""
        open_prompts: dict[str, tuple[str, str, object]] = {}
        scanned = 0
        for ev in tail_events(events_path, follow=False):
            scanned += 1
            etype = str(ev.get("type", ""))
            pid = str(ev.get("id", ""))
            if etype in SESSION_START_EVENTS:
                self._new_session()
                open_prompts.clear()
            if etype == "approval.prompt":
                open_prompts[pid] = ("approval", pid, ev.get("prompt", ""))
            elif etype == "question.prompt":
                open_prompts[pid] = ("question", pid, ev.get("questions", []))
            elif etype in ("approval.answer", "question.answer"):
                self._answered.add(pid)
                open_prompts.pop(pid, None)
        self._replayed = scanned
        return list(open_prompts.values())

    def handle(self, kind: str, prompt_id: str, content: object) -> None:
        """Prompt on the terminal (spinner paused) and write the answer over the
        bridge. Marks the id handled so the follow-loop replay won't re-ask it."""
        if kind == "approval":
            with self._view.pause():
                answer = default_stdin_approver(str(content))
            write_answer(self._session_dir, prompt_id, answer or "no")
        else:
            questions = tuple(
                UserQuestion(
                    question=str(q.get("question", "")),
                    options=tuple(str(o) for o in q.get("options", [])),
                )
                for q in (content if isinstance(content, list) else [])
            )
            with self._view.pause():
                answers = default_stdin_questioner(questions)
            write_question_answers(
                self._session_dir,
                prompt_id,
                answers if answers is not None else tuple("" for _ in questions),
            )
        self._handled.add(prompt_id)

    def _new_session(self) -> None:
        """A session boundary (a fresh run, or a resumed leg) restarts the prompt
        id counters at approval-1/question-1, so the prior leg's ids say nothing
        about the new leg's. Keeping them made the attached front-end swallow the
        resumed leg's first prompt while the worker waited on it forever."""
        self._answered.clear()
        self._handled.clear()

    def react(self, event: dict[str, object]) -> None:
        """Live follow: answer a NEW unanswered prompt; a historical/answered one
        (id in `_answered`/`_handled`) is skipped on the replay."""
        etype = str(event.get("type", ""))
        pid = str(event.get("id", ""))
        if self._replayed > 0:
            # Still inside the pre-scan's window: the follow loop is handing
            # back events `open_prompts_at_attach` already ruled on, so keep the
            # bookkeeping in step but NEVER prompt. Deciding these live re-asked
            # every prompt the run had already answered, because the leg-boundary
            # clear below discarded what the pre-scan knew.
            self._replayed -= 1
            if etype in SESSION_START_EVENTS:
                self._new_session()
            elif etype in ("approval.answer", "question.answer"):
                self._answered.add(pid)
            return
        if etype in SESSION_START_EVENTS:
            self._new_session()
            return
        if etype in ("approval.answer", "question.answer"):
            self._answered.add(pid)
            return
        if pid in self._handled or pid in self._answered:
            return
        if etype == "approval.prompt":
            self.handle("approval", pid, event.get("prompt", ""))
        elif etype == "question.prompt":
            self.handle("question", pid, event.get("questions", []))


def _print_crashed_line(target: Path) -> None:
    print(
        f"[agent6] {target.name}: worker not running and the run never ended"
        f" (crashed or killed); see `agent6 sessions show {target.name}`.",
        file=sys.stderr,
    )


def _render_over_session(target: Path, events_path: Path, *, finished: bool) -> int:
    """A session no worker is driving: nothing more will be appended and no
    answer would be read. Render the log read-only (no front-end, no re-asked
    prompts), then say how it ended.

    *finished* separates the two ways that happens: a run that ended cleanly
    already said its outcome, and calling that "crashed or killed" contradicted
    the `passed` the other surfaces were showing for the same run.
    """
    view = ConsoleView(sys.stdout, policy=lambda: session_policy(target).line())
    try:
        for event in tail_events(events_path, follow=False):
            view.feed(event)
    finally:
        view.close()
    if not finished:
        _print_crashed_line(target)
    return 0


def _install_front_end(target: Path, view: ConsoleView) -> _CliFrontEnd | None:
    """Attach as the answering front-end on an interactive terminal (both
    streams a tty); piped/redirected stays a pure reader (None)."""
    if not (sys.stdin.isatty() and sys.stdout.isatty()):
        print(f"[agent6] following {target.name}. Ctrl-C to exit.", file=sys.stderr)
        return None
    front_end = _CliFrontEnd(target, view)
    register_frontend(target, os.getpid())
    print(
        f"[agent6] attached to {target.name}: approvals and questions prompt here."
        " Ctrl-C to detach.",
        file=sys.stderr,
    )
    return front_end


def _watch_transcript(target: Path) -> int:
    """Follow a run's conversation live and, on an interactive terminal, ATTACH
    to it as a front-end: fold `logs.jsonl` through the same `ConsoleView` as
    `agent6 run` and, when the run asks for a `run_command` approval or an
    `ask_user` answer, prompt on the terminal exactly as the foreground run
    would (see `_CliFrontEnd`). Piped/redirected (no tty) stays a pure reader.
    Renders from the start, tails until the run ends, then returns; Ctrl-C exits.
    A detach emits no session.end, so watching a detached run follows it to its end."""
    events_path = target / LOGS_NAME
    if not events_path.is_file():
        # Not an error: a parked submission, a `fork --no-run`, or a run still
        # launching (egress + the ~80s verify inference run before the first log
        # line) has no log yet. Answer with the same word the listings and
        # `sessions show` use, plus what to do, instead of a raw filesystem message.
        word, reason = status_for_session_dir(target, StatusFacts())
        print(f"{target.name}: {word}" + (f" ({reason})" if reason else ""))
        if word == "starting":
            # A live worker is mid-preflight: it IS running, not resumable.
            # Telling the operator to `resume` would refuse (or fork a second
            # worker); it just has no log to follow yet.
            print("it is starting; run this again in a moment to follow it.")
        else:
            print(f"start it with: agent6 resume {target.name}")
        return 0

    # THE liveness question, answered where every other surface answers it: a
    # second rule here read "no pid" as not-dead, so attach followed a log
    # nothing would append to while `sessions list` called it stale. Whether it
    # ENDED is a separate fact: both stop the follow, only one is a crash.
    if not session_is_live(target):
        scan = scan_session_log(events_path)
        return _render_over_session(target, events_path, finished=scan.finished)

    def worker_dead() -> bool:
        # Per poll, so it stays O(1): once we are following, the session has
        # started and the worker IS the liveness evidence (session_is_live above
        # folds the log once, for the parked/created distinction it needs).
        return not worker_is_alive(target)

    view = ConsoleView(sys.stdout, policy=lambda: session_policy(target).line())
    front_end = _install_front_end(target, view)
    interrupted = False
    try:
        if front_end is not None:
            for kind, prompt_id, content in front_end.open_prompts_at_attach(events_path):
                front_end.handle(kind, prompt_id, content)  # a prompt already pending at attach
        for event in tail_events(
            events_path, follow=True, stop_when_finished=True, should_stop=worker_dead
        ):
            view.feed(event)
            if front_end is not None:
                front_end.react(event)
    except KeyboardInterrupt:
        interrupted = True
        print("\n[agent6] watch: stopped.", file=sys.stderr)
    finally:
        view.close()  # stop the heartbeat thread, clear any spinner line
        if front_end is not None:
            unregister_frontend(target, os.getpid())  # our claim only
    if not interrupted and not _session_has_ended(events_path):
        _print_crashed_line(target)
    return 0


def _line_is_session_end(raw: bytes | str) -> bool:
    """True if a logs.jsonl line is a `session.end` event."""
    text = raw.decode("utf-8", errors="replace") if isinstance(raw, bytes) else raw
    try:
        obj = json.loads(text)
    except (ValueError, TypeError):
        return False
    return isinstance(obj, dict) and obj.get("type") == "session.end"


def _session_has_ended(events_path: Path) -> bool:
    """True if the run's last logged event is `session.end` (finished, nothing to
    follow). A resume appends events after a session.end, so only the LAST line
    counts."""
    try:
        with events_path.open("rb") as fh:
            last = b""
            for last in fh:  # noqa: B007 - keep the final line
                pass
    except OSError:
        return False
    return bool(last) and _line_is_session_end(last)


def _cmd_watch_plain(target: Path, *, since: int) -> int:  # noqa: PLR0911, PLR0912, PLR0915
    """Tail `logs.jsonl` line-by-line with no extra deps.

    Polls the file with 0.25s sleeps; rotates when the inode changes.
    Pretty-prints each event with the type and key fields. Returns 0 on
    EOF (run dir gone) or KeyboardInterrupt.
    """
    events_path = target / LOGS_NAME
    if not events_path.is_file():
        print(f"ERROR: no logs.jsonl in {target}", file=sys.stderr)
        return 2

    # Read the first event for the elapsed-time anchor. Binary readline: a
    # torn-mid-UTF-8 first line must not crash the watch before it starts.
    session_start_ts: float | None = None
    try:
        with events_path.open("rb") as fh:
            first = fh.readline()
        if first:
            obj0 = json.loads(first.decode("utf-8"))
            if isinstance(obj0, dict):
                session_start_ts = event_epoch(obj0.get("ts"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        session_start_ts = None

    print(
        f"[agent6] tailing {events_path}. Ctrl-C to exit.",
        file=sys.stderr,
    )

    # Binary reads throughout: the writer flushes long lines in several
    # syscalls, so a read can hit EOF mid multibyte UTF-8 sequence and a
    # text-mode readline would raise UnicodeDecodeError. Complete lines are
    # decoded (errors="replace"); a partial tail stays buffered until its
    # newline arrives.
    try:
        fh = events_path.open("rb")
    except OSError as exc:
        print(f"ERROR: cannot open {events_path}: {exc}", file=sys.stderr)
        return 2

    try:
        if since > 0:
            # Replay the last `since` lines before following.
            try:
                lines = fh.readlines()
            except OSError as exc:
                print(f"ERROR: read failed: {exc}", file=sys.stderr)
                return 2
            for raw in lines[-since:]:
                line = raw.decode("utf-8", errors="replace")
                # flush: piped/redirected output must not lose the replay to the
                # block buffer when the run is idle/finished (nothing else flushes).
                print(format_plain_event(line, session_start_ts=session_start_ts), flush=True)
            if lines and _line_is_session_end(lines[-1]):
                return 0  # already finished: replayed, nothing to follow
        else:
            # A finished run has no new events to follow; seeking to end would hang.
            if _session_has_ended(events_path):
                print("[agent6] run already finished.", file=sys.stderr)
                return 0
            # Seek to end; only show new events going forward.
            fh.seek(0, 2)
        try:
            current_ino = events_path.stat().st_ino
        except OSError:
            current_ino = -1
        pending = b""
        while True:
            chunk = fh.readline()
            if chunk:
                pending += chunk
                if not pending.endswith(b"\n"):
                    continue  # partial line at EOF; the rest arrives next read
                line = pending.decode("utf-8", errors="replace")
                pending = b""
                print(format_plain_event(line, session_start_ts=session_start_ts), flush=True)
                if _line_is_session_end(line):
                    return 0  # run ended: stop, like the default follower
                continue
            # No new data: check for rotation and sleep briefly.
            try:
                new_ino = events_path.stat().st_ino
            except OSError:
                time.sleep(0.5)
                continue
            if new_ino != current_ino:
                with contextlib.suppress(OSError):
                    fh.close()
                try:
                    fh = events_path.open("rb")
                except OSError:
                    time.sleep(0.5)
                    continue
                current_ino = new_ino
                pending = b""
                continue
            time.sleep(0.25)
    except KeyboardInterrupt:
        print("\n[agent6] watch: stopped.", file=sys.stderr)
        return 0
    finally:
        with contextlib.suppress(OSError):
            fh.close()
