# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Live run watching: the optional dashboard TUI co-process, the
worker stream modes, and the loop's console logger."""

from __future__ import annotations

import contextlib
import os
import signal
import subprocess
import sys
from collections.abc import Callable, Generator
from pathlib import Path

from agent6.ui.cli._console_view import ConsoleView


def loop_logger(mode: str, console_view: ConsoleView | None) -> Callable[[str], None]:
    """The workflow's text logger.

    When the live ConsoleView is rendering the product stream (foreground run),
    notices go THROUGH it (`console_view.notice`) so each clears the spinner
    line first and writes to the same stream under the same lock -- otherwise a
    notice printed to stdout while the stderr spinner is up garbles the line. The
    loop's internal state narration (`LOOP: LOAD_CONTEXT`, `compaction: …`,
    `compaction thresholds: …`) is pure noise on the glyph stream (`config
    show` prints the resolved thresholds), and these lines repeat what the
    stream already shows: `tool_error:` (the red `└`), `auto-commit:` /
    `final checkpoint:` (the sha on the ✎ item), the `STEER:` / `injecting
    steering instruction` pair (the operator item), `ask answered` (the done
    item). All are suppressed unless `AGENT6_DEBUG=1`; genuine notices (review
    decisions, a verify adoption, a steer's abort/detach/undo) pass.
    Headless/`ask` keep the full trace on their own stream (the log, not a
    live stream)."""
    if console_view is None:
        # No live console: a headless run's stdout (or ask's stderr) is
        # block-buffered when redirected to a file/pipe, so without an explicit
        # flush the whole LOOP trace only appears when the process EXITS -- a
        # `nohup agent6 run > log` reads as a dead run for its entire duration.
        # Flush each line so the log is followable as it happens.
        return _eprint if mode == "ask" else _print_flush
    debug = os.environ.get("AGENT6_DEBUG") == "1"

    def _filtered(msg: str) -> None:
        stripped = msg.removeprefix("[agent6] ")
        narration = (
            "compaction",
            "  tool_error:",
            "  auto-commit:",
            "  final checkpoint:",
            "STEER:",
            "  injecting steering instruction",
            "  ask answered",
        )
        if not debug and ("LOOP:" in msg or stripped.startswith(narration)):
            return
        console_view.notice(msg)

    return _filtered


def _print_flush(msg: str) -> None:
    """Headless loop logger: print to stdout and flush, so a redirected log
    (block-buffered) shows the LOOP trace live instead of only at exit."""
    print(msg, flush=True)


def _eprint(msg: str) -> None:
    """Loop logger that writes to stderr (used for `ask`, whose stdout is the
    answer and must stay clean for piping). Flushes so a redirected stderr is
    followable live, not buffered until exit."""
    print(msg, file=sys.stderr, flush=True)


def _tui_available() -> bool:
    import importlib.util  # noqa: PLC0415

    return importlib.util.find_spec("textual") is not None


def should_spawn_tui(*, tui: bool, interactive: bool, mode: str) -> bool:
    """Whether `agent6 run`/`plan`/`resume` opens the dashboard TUI.

    Headless by default (a scrolling CLI event stream); `--tui` opts into the
    full-screen dashboard. It needs a real TTY, is for `run` and `plan` (an ask
    stays text: its answer is the deliverable), and is mutually exclusive with
    `-i` (the stdin REPL). When `--tui` is asked for but cannot run, warn and
    stay headless rather than fail the run."""
    if not tui:
        return False
    if interactive or mode not in ("run", "plan"):
        print("[agent6] --tui is not available here; continuing in CLI mode.", file=sys.stderr)
        return False
    if not sys.stdout.isatty():
        print("[agent6] --tui needs a TTY; continuing in CLI mode.", file=sys.stderr)
        return False
    if not _tui_available():
        print(
            "[agent6] --tui needs 'textual' (part of the base install; this environment"
            " is missing it); continuing in CLI mode.",
            file=sys.stderr,
        )
        return False
    return True


def stream_modes(*, tui_enabled: bool) -> tuple[bool, bool]:
    """Return `(stream_text, console_stream)` for the worker provider.

    `stream_text` makes the provider stream and emit `role.text_delta` /
    `role.thinking_delta` events, which every live view renders as the model's
    reasoning + answer. `console_stream` additionally subscribes a
    `ConsoleView` to the EventSink, rendering the live conversation -- reasoning,
    text, and every tool call with its result -- to stderr.

    Streaming is on for an interactive stderr TTY (so a plain `agent6 ask`/`plan`
    shows live output) or when forced:
    - `AGENT6_FORCE_STREAM=1`: bench/CI -- emit AND echo (the Kimi/OpenRouter
      gateway corrupts the non-streaming body with SSE heartbeats).
    - `AGENT6_STREAM_TO_LOG=1`: set by the `agent6 tui` hub when it spawns a run
      detached and then watches it on the dashboard. Emit the delta EVENTS only,
      with NO console echo -- otherwise a long headless run pours its whole
      reasoning into the hub's discarded stderr temp file.
    """
    stream_to_log = os.environ.get("AGENT6_STREAM_TO_LOG") == "1"
    stream_text = (
        sys.stderr.isatty() or os.environ.get("AGENT6_FORCE_STREAM") == "1" or stream_to_log
    )
    # Echo to stderr only when there is a console to read it: not while the TUI
    # owns the terminal, and not for a hub-watched headless run (dashboard-only).
    console_stream = stream_text and not tui_enabled and not stream_to_log
    return stream_text, console_stream


@contextlib.contextmanager
def tui_session(session_dir: Path, *, enabled: bool) -> Generator[None]:
    """Run the dashboard TUI as a co-process that owns the terminal.

    While it is up, this process's own console chatter is redirected to
    `<session_dir>/tui_console.log` so it doesn't fight the TUI for the terminal;
    progress still flows through `logs.jsonl`, which the TUI tails, and approvals
    go through the file bridge. On `session.end` the TUI holds the finished
    dashboard until the user leaves (Ctrl+Q); we wait for that, so the terminal
    stays theirs until they are done looking. A spawn failure degrades
    to a normal (TUI-less) run rather than aborting."""
    if not enabled:
        yield
        return
    try:
        # Opened before the spawn: a failure after it would escape past the
        # wait below, orphaning a TUI that has taken the terminal.
        log_fh = (session_dir / "tui_console.log").open("w", encoding="utf-8")
    except OSError as exc:
        print(f"[agent6] could not start TUI ({exc}); continuing without it.", file=sys.stderr)
        yield
        return
    try:
        # -P keeps cwd (the workspace) off sys.path: a top-level `agent6/` in
        # the repo must not shadow the installed package in this co-process.
        proc = subprocess.Popen(
            [
                sys.executable,
                "-P",
                "-m",
                "agent6.ui.tui",
                "--watch",
                str(session_dir),
                "--exit-on-end",
            ]
        )
    except OSError as exc:
        log_fh.close()
        print(f"[agent6] could not start TUI ({exc}); continuing without it.", file=sys.stderr)
        yield
        return
    orig_out, orig_err = sys.stdout, sys.stderr
    sys.stdout = log_fh
    sys.stderr = log_fh
    try:
        yield
    finally:
        # The TUI holds the finished dashboard until the user leaves (Ctrl+Q),
        # so wait for THEM, not a deadline: killing the view at +8s is exactly
        # the payoff-vanishes defect the hold exists to fix. A wedged TUI is
        # visibly wedged under the hold hint; Ctrl-C (SIGINT to the group)
        # still tears everything down, textual restoring the terminal. Keep
        # our own output redirected until it's gone so nothing scribbles its
        # screen.
        try:
            proc.wait()
        except KeyboardInterrupt:
            proc.send_signal(signal.SIGINT)
            try:
                proc.wait(timeout=4)
            except subprocess.TimeoutExpired:
                proc.terminate()
                with contextlib.suppress(subprocess.TimeoutExpired):
                    proc.wait(timeout=3)
        sys.stdout, sys.stderr = orig_out, orig_err
        with contextlib.suppress(Exception):
            log_fh.close()
