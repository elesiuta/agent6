# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The agent6 run dashboard (`agent6 run` / `agent6 attach` / `agent6 tui`).

`textual` ships in the base install; importing this module fails clearly if it
has been stripped out. The CLI imports it lazily.

Architecture:
- `Agent6TUI` (the App) is the data plane: a background thread tails
  logs.jsonl -> apply_event -> call_from_thread, and the app owns the folded
  SessionState, the approval/question/steer prompt dispatch, run control (steer /
  stop / resume / fork), and the exit codes.
- `DashboardScreen` is the presentation: the panes, their key bindings and
  menus, and the coalesced repaint of the app's SessionState.

The dashboard is READ-ONLY on the log stream and only writes the answer files
the workflow polls: `<session_dir>/approvals/<id>.answer` (approve), `.../questions/
<id>.answer` (ask_user), `<session_dir>/steer.answer` (steer), and the
`<session_dir>/compact.request` marker (Compact now). Any other front-end can
mirror this contract.
"""

from __future__ import annotations

import contextlib
import os
import threading
import time
from collections.abc import Iterable
from pathlib import Path
from typing import ClassVar

try:
    from textual import work
    from textual.app import App, ScreenStackError, SystemCommand
    from textual.binding import Binding
    from textual.css.query import NoMatches
    from textual.screen import ModalScreen, Screen
except ImportError as e:  # pragma: no cover - clear runtime message
    raise ImportError(
        "agent6 TUI requires the 'textual' package (part of the base install)."
        " Reinstall agent6, or `pip install textual`."
    ) from e

from agent6.app.fork import create_fork
from agent6.app.reporter import Reporter
from agent6.app.undo import undo_fork
from agent6.config.layer import available_preset_names
from agent6.directive import parse_btw, parse_compact, parse_now
from agent6.paths import mkdir_for_real_user
from agent6.sessions.ipc import (
    register_frontend,
    request_compact,
    request_stop,
    submit_steer,
    unregister_frontend,
)
from agent6.sessions.layout import LOGS_NAME, bucket_dir, layout_of
from agent6.sessions.manifest import ManifestError, read_manifest
from agent6.tools.background import shells_text
from agent6.ui.btw import open_btw
from agent6.ui.spawn import (
    DETACHED_RUN_ENV,
    agent6_argv,
    run_cli_capture,
    spawn_and_locate,
    spawn_detached_resume,
)
from agent6.ui.tui.composer import SteerInput
from agent6.ui.tui.conversation import ConversationScreen
from agent6.ui.tui.dashboard import DashboardScreen
from agent6.ui.tui.modals import (
    ConfirmModal,
    TextModal,
)
from agent6.ui.tui.prompts import PromptDispatcher
from agent6.ui.tui.theme import (
    PALETTE_CSS,
    MuxPointerShapes,
    PlainNotify,
    setup_theme,
)
from agent6.viewmodel import restate
from agent6.viewmodel.events import SESSION_START_EVENTS
from agent6.viewmodel.listing import (
    LIVE_STATUS_WORDS,
    finished_needs_new_work,
    status_for_session_dir,
    task_snippet,
)
from agent6.viewmodel.state import (
    STREAM_DELTA_EVENTS,
    SessionState,
    apply_event,
    context_fill,
    fold_session,
    initial_state,
    status_facts,
)
from agent6.viewmodel.tail import tail_events

# Answer submitted after the worker died mid-modal: the file bridge has no
# reader, and the next resume re-asks the prompt itself.
_ANSWER_LOST = "the session is not live; the answer reached nothing (a resume re-asks the prompt)"

# Events after which dir_status is recomputed synchronously rather than on the
# ~1s heartbeat: session boundaries plus the operator-blocking prompt/answer
# pairs. The chip, worker_lost, and both composer bars all route off
# dir_status, so serving the previous state for even one heartbeat lies.
_STATUS_NOW_EVENTS = SESSION_START_EVENTS | {
    "session.end",
    "approval.prompt",
    "approval.answer",
    "question.prompt",
    "question.answer",
}

# Dashboard exit code meaning "quit the whole hub" (vs 0 == back to the hub).
QUIT_HUB_CODE = 99


class Agent6TUI(PlainNotify, MuxPointerShapes, App[int]):
    TITLE = "agent6"
    CSS = (
        PALETTE_CSS
        + """
    Screen { layers: base dropdown; background: $surface; }
    /* The flat Screen rule above also matches ModalScreens, which would make
       their backdrops opaque; restore textual's translucent dim (same
       specificity, later rule wins) so the screen shows through behind dialogs. */
    ModalScreen { background: $background 60%; }
    * { scrollbar-size-vertical: 1; scrollbar-size-horizontal: 1; }  /* half the 2-wide default */
    /* A footer that does not fit clips (textual's default); the 1-row widget has no
       room for the scrollbar the universal rule gives it, which replaced every hint. */
    Footer { scrollbar-size-vertical: 0; scrollbar-size-horizontal: 0; }
    /* I-beam over anything you can type into (kitty OSC 22; inert elsewhere). */
    Input, TextArea { pointer: text; }
    """
    )

    BINDINGS: ClassVar = [
        # App-level so it works from any screen (viewers included); the hub-aware
        # exit code needs our handler, not textual's default quit.
        Binding("ctrl+q", "quit_hub", "Quit", show=False),
        # Ctrl-Z means "step away": the run keeps going and the shell comes
        # back (raw mode keeps the key, and a real SIGTSTP would freeze a live
        # provider stream mid-response). priority so the composer TextArea's
        # built-in ctrl+z undo never shadows it; Ctrl-_ is undo there, and the
        # detach hint says so.
        Binding("ctrl+z", "detach_exit", "Detach", show=True, priority=True),
    ]

    def __init__(
        self,
        session_dir: Path,
        *,
        exit_on_end: bool = False,
        from_hub: bool = False,
        config_path: Path | None = None,
    ) -> None:
        super().__init__()
        self.session_dir = session_dir
        # The invocation's `--config F`; a detach-resume it spawns re-applies F.
        self.config_path = config_path
        # When launched from the hub loop, Esc returns to it and q quits the hub
        # (signalled by the exit code); standalone, both just close the dashboard.
        self.from_hub = from_hub
        self.logs_path = session_dir / LOGS_NAME
        self.state: SessionState = initial_state()
        self._prompts = PromptDispatcher(
            self,
            answerable=self.session_controllable,
            lost=_ANSWER_LOST,
            inline_approvals=lambda: isinstance(self._screen_or_none(), ConversationScreen),
        )
        self._seen_steer = 0
        self._dirty = False  # a structural event arrived; _tick coalesces the repaint
        self._light_dirty = False  # only stream deltas / heartbeat: light repaint
        self._stop = threading.Event()
        # When True (the auto-spawned co-process of `agent6 run`), the view
        # ends WITH the run: once it finishes, the dashboard holds on the
        # payoff until the user leaves (Ctrl+Q), and only then does the parent
        # command return. `agent6 attach --tui` leaves this False and keeps
        # following.
        self.exit_on_end = exit_on_end
        # The run ended under exit_on_end and the dashboard is holding.
        self._end_hold = False
        # The fork this view created (/undo on a finished run, Run > Fork; the
        # fold has no event for either); continue_as routes the follow-up there.
        self._continue_child = ""
        # Set by action_detach_exit; run_tui reads it to print the reattach hint.
        self.detached = False
        # THE (word, reason) for this run -- status_for_session_dir, the same
        # decision the hub row shows -- refreshed on the ~1/s heartbeat.
        # Derived, never latched: a crash->resume flips it back to running
        # (a one-way latch would keep "worker exited" painted over the live
        # resumed leg and drop operator steers).
        self.dir_status: tuple[str, str] = status_for_session_dir(
            session_dir, status_facts(self.state)
        )
        # Lines the log held at open; worker_lost waits for the fold to reach it.
        self._seed_log_count = 0
        # Header task line for a run with no session.start yet (parked/created):
        # the fold has no user_task, but the manifest knows the work.
        self.fallback_task = ""
        # The session's mode (run / plan / ask): the dashboard's title word, as
        # the web panel heading states it; "run" for a manifest-less dir.
        self.mode = "run"
        with contextlib.suppress(ManifestError):
            manifest = read_manifest(session_dir)
            self.fallback_task = manifest.user_task
            self.mode = manifest.mode or "run"
        # Live heartbeat: a run can be silent for a whole reasoning turn (or the
        # resume context-rebuild gap). Track when the last event landed and
        # repaint ~1/s while active so an elapsed timer + spinner visibly tick --
        # the difference between "thinking" and "hung" the user could not see.
        self.last_event_at = time.monotonic()
        self._heartbeat_at = 0.0
        self.spin = 0
        # The preset a resume from a composer continues under ("" = as
        # recorded); both run views' pickers read and write it.
        self.resume_preset = ""
        presets = available_preset_names(Path.cwd(), config_path)
        self._dash = DashboardScreen(presets=presets)
        self._conv = ConversationScreen(
            self.logs_path,
            title=self.screen_title,
            presets=presets,
            prompts=self._prompts,
        )

    def _task_lead(self) -> str:
        """What names this run in a title: the TASK (clipped), the pet name
        only when no task is known yet -- the web hub's rows lead the same
        way, and the id stays in the header line and every resume hint."""
        task = self.state.user_task or self.fallback_task
        return task_snippet(task, max_chars=57) or self.session_dir.name

    def screen_title(self, context: str) -> str:
        """Menu-bar subtitle for a run screen: the view's context word, the live
        task name, and -- once the run ended -- the status plus how to leave.
        Screens stamp via a provider calling this at stamp time, never a string
        frozen at construction: the task name lands after the first fold, and
        the end hold must survive whichever stamp runs last."""
        if self._end_hold:
            return f"{context} · {self._task_lead()} · {self.dir_status[0]} · Ctrl+Q to leave"
        return f"{context} · {self._task_lead()}"

    def run_title(self) -> str:
        return self.screen_title(self.mode)

    def on_mount(self) -> None:
        setup_theme(self)  # apply the saved theme before the first paint
        # Per-process claim file: nothing to defend or re-assert, concurrent
        # web/TUI/attach viewers each hold their own.
        register_frontend(self.session_dir, os.getpid())
        self.sub_title = self.run_title()  # menu-bar title context
        self._seed_from_disk()
        # Pushed (not the app's default screen): only the push path loads a
        # screen's CSS, and the hub pushes its HomeScreen the same way. The
        # conversation opens on top -- the primary view -- with the dashboard
        # beneath it; Ctrl+D toggles between them.
        # Installed, so popping the conversation hides rather than destroys it.
        self.push_screen(self._dash)
        self.install_screen(self._conv, "conversation")
        self.push_screen(self._conv)
        # Auto-spawn close: the exit condition (run over, prompts answered) is
        # polled from a timer in the app's OWN loop and exits there. Exit()
        # scheduled from inside a call_from_thread callback does not take effect,
        # but exiting from a timer callback does. The same timer also drives the
        # approval / question modals and the steer composer focus.
        self.set_interval(0.2, self._tick)
        self._thread = threading.Thread(target=self._reader_loop, daemon=True)
        self._thread.start()

    def _seed_from_disk(self) -> None:
        """Fold the log already on disk, before the reader thread starts.

        It seeds the status this viewer opens on, the line the fold must reach
        for an absent `session.end` to mean anything (`worker_lost`), and the
        steer baseline, so a request already in the log does not prompt.
        """
        with contextlib.suppress(OSError):
            seeded = fold_session(tail_events(self.logs_path, follow=False))
            self._seen_steer = seeded.steer_requests
            self._seed_log_count = seeded.log_count
            self.dir_status = status_for_session_dir(self.session_dir, status_facts(seeded))

    def on_unmount(self) -> None:
        self._stop.set()
        # Drop only our own front-end claim; concurrent viewers keep theirs.
        unregister_frontend(self.session_dir, os.getpid())

    # --- reader thread -----------------------------------------------

    def _reader_loop(self) -> None:
        for event in tail_events(
            self.logs_path,
            follow=True,
            stop_when_finished=self.exit_on_end,
            # Without this, closing the dashboard on a run that never ends
            # (finished + exit_on_end=False, or crashed) leaks this thread in
            # the idle poll forever -- one per run the hub session opens.
            should_stop=self._stop.is_set,
        ):
            if self._stop.is_set():
                return
            self.call_from_thread(self._handle_event, event)

    def seconds_since_event(self) -> int:
        """Idle seconds for the "working… Ns" heartbeat, anchored to the last
        EVENT's ts (the fold carries it), not to when this viewer folded it:
        an arrival anchor would read "working… 3s" on attach to a run wedged
        40 minutes. Falls back to the fold time for
        a log whose events carry no ts."""
        if self.state.last_event_ep is None:
            return int(time.monotonic() - self.last_event_at)
        return max(0, int(time.time() - self.state.last_event_ep))

    def _handle_event(self, event: dict[str, object]) -> None:
        self.state = apply_event(self.state, event)
        self.last_event_at = time.monotonic()  # the ts-less fallback anchor
        if event.get("type") == "session.undone" and self.state.undone_to:
            # /undo forked the run at the state before the operator's last
            # message: that fork is the continuation, and the message taken
            # back is the operator's to edit and resend (the web does the same).
            self._fill_composers(self.state.undone_text)
            self.notify(f"undone: continue as {self.state.undone_to}; your message is back to edit")
        if event.get("type") in SESSION_START_EVENTS:
            # A session boundary (fresh run OR a resumed leg -- a resume emits
            # only loop.resume.start, never a second session.start) restarts the
            # prompt id counters at approval-1/question-1; a stale seen-set
            # would swallow the new session's first prompts and the run would
            # block forever on a modal that never opens.
            self._prompts.reset()
            # The task is known once the boundary folds: retitle the menu bar
            # (the conversation screen stamps its own title while on top).
            if self._screen_or_none() is self._dash:
                self.sub_title = self.run_title()
        if event.get("type") in _STATUS_NOW_EVENTS:
            # A terminal / leg-boundary / operator-blocking event changes the
            # status NOW: refresh synchronously so the chip, the label, and the
            # composer routing never serve the previous state for up to a
            # heartbeat.
            self._refresh_dir_status()
        # Coalesce: mark dirty and let the 0.2s _tick repaint once. Replaying a
        # finished run floods hundreds of events on open; rendering each one would
        # rebuild the whole dashboard per event (UI thrash). Streaming deltas
        # only move the live stream pane, so they take the LIGHT repaint (a
        # reasoning burst would otherwise force full rebuilds 5x/s).
        if event.get("type") in STREAM_DELTA_EVENTS:
            self._light_dirty = True
        else:
            self._dirty = True

    @property
    def worker_lost(self) -> bool:
        """The recorded worker is gone without a session.end (kill -9 / OOM),
        the hub's "stale". Derived from dir_status, so a resume that brings a
        live worker back clears it. False while the fold trails the log: an end
        it has not reached is not an absent one."""
        return self.dir_status[0] == "stale" and self.state.log_count >= self._seed_log_count

    def _refresh_dir_status(self) -> None:
        """Recompute dir_status (a pid probe + a manifest read pre-start; the
        same cost class as the spinner tick, so it rides the ~1/s heartbeat).
        A change repaints and relabels BOTH composer bars -- the covered
        screen's too, which otherwise kept a stale label until its next
        event-driven paint (the two bars visibly disagreed live)."""
        status = status_for_session_dir(self.session_dir, status_facts(self.state))
        if status != self.dir_status:
            self.dir_status = status
            self._dirty = True
            self._conv.refresh_liveness()

    def _screen_or_none(self) -> Screen[object] | None:
        """The active screen, or None while the stack is empty (startup,
        teardown): `self.screen` raises there, and a tick or a replayed
        event in that window has nothing to render on."""
        try:
            return self.screen
        except ScreenStackError:
            return None

    def _tick(self) -> None:
        # Prompt modals only while the run can consume an answer: the fold
        # keeps an unanswered prompt across session.end and a worker death (it
        # clears only on the answer event or a leg boundary), so an open would
        # otherwise pop live-looking Allow/Deny over a dead run and write the
        # answer into a file nobody polls. Skipped ids are NOT marked seen, so
        # a prompt that outlives a stale probe still pops on the next tick.
        # No screen yet (the first screens land async at startup) or none
        # left (teardown): nothing can render a prompt, and unclaimed ids
        # pop on the next tick.
        if self.session_controllable() and self._screen_or_none() is not None:
            self._prompts.dispatch(self.session_dir, self.state)
        # Route an external steer request to the composer bar, once per Ctrl-C
        # (steer_requests is monotonic).
        if self.state.steer_requests > self._seen_steer:
            self._seen_steer = self.state.steer_requests
            self._steer_request_to_bar()
        # Heartbeat: refresh the dir status ~1/s (always -- it is how a death,
        # a parked resume, or a revival is noticed with no event to trigger a
        # paint), and while the run is live advance the spinner so the
        # "working… Ns" timer visibly ticks (thinking, not hung).
        now = time.monotonic()
        if now - self._heartbeat_at >= 1.0:
            self._heartbeat_at = now
            self._refresh_dir_status()
            if self.session_controllable():
                self.spin += 1
                self._light_dirty = True
        # Coalesced repaint: once per tick, and only when the dashboard is the
        # active, mounted screen. A pushed viewer, a modal, or shutdown leaves the
        # dashboard covered or torn down, so querying its widgets raises; defer
        # the paint (dirty stays set) until it is back on top. Structural events
        # rebuild the panes; deltas/heartbeat repaint only the light parts.
        # Read screen_stack, not App.screen: the interval outlives the stack
        # during shutdown, and a tick landing after the last screen pops must be
        # a no-op, not a ScreenStackError crash.
        stack = self.screen_stack
        if not self._stop.is_set() and stack and stack[-1] is self._dash:
            if self._dirty:
                self._dirty = self._light_dirty = False
                with contextlib.suppress(NoMatches):
                    self._dash.render_state()
            elif self._light_dirty:
                self._light_dirty = False
                with contextlib.suppress(NoMatches):
                    self._dash.render_heartbeat()
        # Once the run ended (a clean session.end, or the worker died without
        # one) and no modal is open -- never yank an in-flight answer, but a
        # ghost prompt on a dead run (its answer read by nobody) must not pin
        # the hold off forever -- the dashboard HOLDS on the payoff (verify,
        # diff, cost) instead of tearing down under the user. Ctrl+Q leaves;
        # the composer still routes a typed follow-up to resume.
        if (
            self.exit_on_end
            and not self._end_hold
            and (self.state.finished or self.worker_lost)
            and not (stack and isinstance(stack[-1], ModalScreen))
        ):
            self._end_hold = True
            self.sub_title = self.run_title()
            self.notify(
                f"{self.dir_status[0]} · Ctrl+Q to leave, or type below to continue the session",
                timeout=8.0,
            )
            self._dirty = True

    # --- run control (dispatched from the composer bars, keys, and menus) --

    def submit_instruction(self, text: str) -> None:
        """A composer-bar line. Live: inject it at the run's next safe boundary
        (after the current step, never mid tool-call) -- the run keeps going.
        Finished: resume THIS run with the instruction as the follow-up."""
        if text.strip() == "/undo":
            self._undo_session()
            return
        if text.strip() == "/restate":
            # Local and free: rendered from the journal, nothing reaches the model.
            rendered = restate(list(tail_events(self.logs_path, follow=False)))
            self.push_screen(TextModal("since your last message", rendered))
            return
        if text.strip() == "/shells":
            self.push_screen(TextModal("background commands", shells_text(self.session_dir)))
            return
        if self.session_controllable():
            question = parse_btw(text)
            if question is not None:
                opened, line = open_btw(self.session_dir, question)
                self.notify(line, severity="information" if opened else "warning")
                return
            focus = parse_compact(text)
            if focus is not None:
                # `/compact [focus]` is an out-of-band request, not steer text;
                # /pin and /parallel stay steers the loop parses itself.
                if request_compact(self.session_dir, focus=focus):
                    self.notify("compaction requested; applies before the next model call")
                else:
                    self.notify("could not write the compaction request", severity="warning")
                return
            urgent = parse_now(text)
            if urgent == "":
                self.notify("/now needs the instruction: /now <text>", severity="warning")
                return
            submit_steer(self.session_dir, urgent or text, now=urgent is not None)
            self.notify("steering this session now…" if urgent else "steering this session…")
        else:
            self.resume_with_instruction(text)

    def _undo_session(self) -> None:
        """`/undo`: fork this run at the state before its last operator message,
        unstarted; the hub lists the fork and the undone text is reported. A
        live run is refused: stop it first, then undo."""
        if self.session_controllable():
            # The loop forks at its next boundary and emits session.undone;
            # the fold's undone_to hands the follow-up to this app.
            submit_steer(self.session_dir, "/undo")
            self.notify("undo requested; applies at the next step")
            return
        said: list[str] = []
        result = undo_fork(
            None,
            self.session_dir.name,
            cwd=Path.cwd(),
            reporter=Reporter(out=said.append, err=said.append),
        )
        if result is None:
            self.notify(said[-1].strip() if said else "undo failed", severity="warning")
            return
        child, text = result
        self._continue_child = child
        self._fill_composers(text)
        self.notify(f"undone: continue as {child}; your message is back to edit")

    def plan_md(self) -> str:
        """A planning run's written deliverable (plan.md), "" for a run or a
        plan that has not written one."""
        if self.mode != "plan":
            return ""
        with contextlib.suppress(OSError):
            return (self.session_dir / "plan.md").read_text(encoding="utf-8")
        return ""

    @property
    def continue_as(self) -> str:
        """The session a typed follow-up resumes: the fork an undone run named
        (its continuation, from the fold), the fork this view created (/undo
        on a finished run, Run > Fork), else "" for this run itself."""
        return self.state.undone_to or self._continue_child

    def _fill_composers(self, text: str) -> None:
        """Put *text* in both composer bars (the covered view's too), ready to
        edit and resend."""
        for screen, bar_id in ((self._conv, "#conv-input"), (self._dash, "#dash-input")):
            with contextlib.suppress(NoMatches):
                screen.query_one(bar_id, SteerInput).load_text(text)

    def resume_with_instruction(self, text: str) -> None:
        """Resume this run (or, after /undo, the fork it named) with *text* as
        its first steering instruction (rides `agent6 resume --steer`, which
        seeds the steer files AFTER its stale-state clear; a pre-seed here
        would be wiped by that clear). The new session's steer poll injects
        the text at its first boundary."""
        target = self.continue_as or self.session_dir.name
        under = f" under preset {self.resume_preset}" if self.resume_preset else ""
        self._spawn_resume(
            target,
            steer=text,
            preset=self.resume_preset,
            started=f"resuming {target}{under} with your instruction…",
        )

    @work(thread=True)
    def _spawn_resume(
        self, target: str, *, started: str, steer: str = "", preset: str = ""
    ) -> None:
        """The detached resume, off the UI thread: the spawn waits until the
        child owns the run or refuses (its preflight takes a second or more),
        and the notice lands from here."""
        err = spawn_detached_resume(
            Path.cwd(), target, steer=steer, preset=preset, config_path=self.config_path
        )
        self.call_from_thread(
            self.notify, err or started, severity="error" if err else "information"
        )

    def _focus_composer(self) -> None:
        """Focus the visible composer bar; with a viewer or modal on top, the
        caller's notice alone points the operator at the bar."""
        stack = self.screen_stack
        if not stack:  # shutdown race: the tick fired after the last screen popped
            return
        if stack[-1] is self._conv:
            self._conv.focus_bar()
        elif stack[-1] is self._dash:
            with contextlib.suppress(NoMatches):
                self._dash.query_one("#dash-input", SteerInput).focus()

    def _steer_request_to_bar(self) -> None:
        """An external steer request (a CLI Ctrl-C on an attached run, `agent6
        steer`): route it to the visible composer bar instead of a popup --
        focus it and say why."""
        self._focus_composer()
        self.notify("steering requested: type an instruction and press Enter")

    def action_compact(self) -> None:
        """Run > Compact context now: the composer's `/compact`, behind the
        liveness check (the composer resumes a finished session with its
        text; a compact has nothing to resume)."""
        if not self.session_controllable():
            self.notify("nothing to compact: the session is not live", severity="warning")
            return
        self.submit_instruction("/compact")

    def action_stop_now(self) -> None:
        """Stop the run immediately: confirm, then write the abort answer over
        the file bridge -- the stream watchdog interrupts the in-flight turn and
        the run ends (resumable)."""
        if not self.session_controllable():
            self.notify("nothing to stop: the session is not live", severity="warning")
            return

        def _confirmed(yes: bool | None) -> None:
            if yes:
                submit_steer(self.session_dir, "abort")

        self.push_screen(
            ConfirmModal(
                "Stop this session now?",
                "Interrupts the current step; the run ends at once and can be resumed "
                "later with `agent6 resume`.",
                confirm_label="Stop now",
            ),
            _confirmed,
        )

    def action_stop_step(self) -> None:
        """Stop AFTER the current step completes: drop the stop.request marker
        the loop honors at its next completed-iteration boundary, so the step's
        tool results and auto-commit land before the run ends (resumable)."""
        if not self.session_controllable():
            self.notify("nothing to stop: the session is not live", severity="warning")
            return

        def _confirmed(yes: bool | None) -> None:
            if not yes:
                return
            if request_stop(self.session_dir):
                self.notify("stopping after this step…")
            else:
                self.notify("could not write the stop request", severity="warning")

        self.push_screen(
            ConfirmModal(
                "Stop after this step?",
                "The current step finishes (its tool results and auto-commit land), "
                "then the run stops. Resume later with `agent6 resume`.",
                confirm_label="Stop",
            ),
            _confirmed,
        )

    def action_delete_session(self) -> None:
        """Delete this run's history and return to the hub. History only: the run
        branch and its commits are git's (`sessions prune` is the branch verb)."""
        if self.session_controllable():
            self.notify("stop the session first: it is still live", severity="warning")
            return

        def _confirmed(yes: bool | None) -> None:
            if yes:
                ok, msg = run_cli_capture(
                    [*agent6_argv(self.config_path), "sessions", "rm", "--", self.session_dir.name],
                    Path.cwd(),
                )
                self.notify(
                    msg or ("removed" if ok else "could not remove"),
                    severity="information" if ok else "error",
                )
                if ok:
                    self.action_to_hub()

        self.push_screen(
            ConfirmModal(
                "Delete this session's history?",
                "Removes its transcripts, events and manifest from the state dir. "
                "The run branch and its commits are kept.",
                confirm_label="Delete",
            ),
            _confirmed,
        )

    def action_resume(self) -> None:
        """Resume a finished/stopped run: it continues in the background (appending
        to the same log) and this dashboard follows straight through. A run the
        agent ENDED has nothing to continue: the refusal `agent6 resume` would
        print lands here instead of on a detached child's stderr (the composer
        below gives it new work)."""
        if self.session_controllable():
            self.notify("nothing to resume: the session is still going", severity="warning")
            return
        if finished_needs_new_work(self.session_dir):
            self.notify(
                "this run finished; type what to do next below (Enter resumes it with the"
                " instruction)",
                severity="warning",
            )
            self._focus_composer()
            return
        self._spawn_resume(
            self.session_dir.name, started=f"resuming {self.session_dir.name} in the background…"
        )

    def action_run_plan(self) -> None:
        """Execute a finished plan: spawn `agent6 run --from <id>` detached
        (the hub lists and opens it). The plan session is untouched, so the
        composer keeps revising it."""
        if self.mode != "plan":
            self.notify("this session is not a plan", severity="warning")
            return
        if not (self.session_dir / "plan.md").is_file():
            self.notify("no plan.md yet (still planning, or never finished)", severity="warning")
            return
        runs = bucket_dir(layout_of(self.session_dir).state_dir, "runs")
        mkdir_for_real_user(runs)
        self._spawn_run_plan(runs)

    @work(thread=True)
    def _spawn_run_plan(self, runs: Path) -> None:
        """The detached `run --from`, off the UI thread: the locate waits for
        the run's first event, and the notice lands from here."""
        new_dir, err = spawn_and_locate(
            [*agent6_argv(self.config_path), "run", "--from", self.session_dir.name],
            Path.cwd(),
            before={p for p in runs.iterdir() if p.is_dir()},
            list_dirs=lambda: [p for p in runs.iterdir() if p.is_dir()],
            env={**os.environ, **DETACHED_RUN_ENV},
        )
        if new_dir is None:
            self.call_from_thread(self.notify, err or "could not start the run", severity="error")
            return
        self.call_from_thread(self.notify, f"run started: {new_dir.name} (open it from the hub)")

    def action_fork(self) -> None:
        """Fork this run at its latest checkpoint into a NEW run, unstarted. On
        a finished run the composer is handed to the fork: the next typed line
        is its instruction (Enter resumes the fork with it), the way /undo
        hands over its fork. On a live run the composer keeps steering THIS
        run, so the notice says how the fork starts. A fork that simply
        continued had no direction: of a finished run it re-read a done
        conversation and ended as a silent finish."""
        said: list[str] = []
        child, rc = create_fork(
            self.config_path,
            self.session_dir.name,
            cwd=Path.cwd(),
            reporter=Reporter(out=said.append, err=said.append),
        )
        if rc != 0:
            self.notify(said[-1].strip() if said else "fork failed", severity="error")
            return
        if self.session_controllable():
            self.notify(f"forked to {child} (unstarted); start it: agent6 resume {child} --steer …")
            return
        self._continue_child = child
        self.notify(f"forked to {child}; type what it should do below (Enter resumes it)")
        self._conv.refresh_liveness()
        with contextlib.suppress(NoMatches):
            self._dash.render_heartbeat()
        self._focus_composer()

    def context_pct(self) -> int | None:
        """Context-window fill (percent) at the last completed model call (the
        viewmodel's rule, shared with the web header and the pause menu)."""
        return context_fill(self.state)

    def session_controllable(self) -> bool:
        """True while the run can receive operator input over the file bridge:
        the dir status is a live word. Parked/created (never started), stale
        (worker gone) and every end word route the composer to resume -- the
        one action that will actually be read."""
        return self.dir_status[0] in LIVE_STATUS_WORDS

    def action_toggle_dashboard(self) -> None:
        """Flip between the conversation (the primary view) and the dashboard
        (Ctrl+D anywhere, t from the dashboard). The conversation is installed,
        so popping it hides it -- both views keep their state. No-op while a
        modal or a pushed viewer is on top."""
        if self.screen is self._conv:
            self.pop_screen()
        elif self.screen is self._dash:
            self.push_screen(self._conv)

    def action_to_hub(self) -> None:
        self.exit(0)  # back to the hub loop (or just close, standalone)

    def action_quit_hub(self) -> None:
        # In the hub loop, signal "quit the hub" via the exit code; standalone,
        # there's nothing to return to, so a plain close (0) is the same thing.
        self.exit(QUIT_HUB_CODE if self.from_hub else 0)

    def action_detach_exit(self) -> None:
        # A viewer (`attach --tui`, the hub) leaves a run that is detached
        # already. The view `agent6 run --tui` spawned (exit_on_end) fronts a
        # run in the terminal's own process, so it steers that run to detach:
        # at its next step boundary the lifecycle hands it to a background
        # resume and exits, as the CLI pause menu's /detach does, and the
        # shell comes back. run_tui prints the hint once the terminal is
        # restored.
        if self.exit_on_end:
            submit_steer(self.session_dir, "detach")
        self.detached = True
        self.exit(QUIT_HUB_CODE if self.from_hub else 0)

    def get_system_commands(self, screen: Screen[object]) -> Iterable[SystemCommand]:
        # Drop textual's "Keys" panel (our Help page replaces it), "Screenshot" (an
        # unused default whose SVG export is broken in our terminals), "Theme"
        # (replaced by our live-preview Theme… picker), and "Quit" (its plain exit()
        # returns the wrong code here -- our File menu's Back to hub / Quit do). All
        # of these are provided by MENUS via palette_commands, so nothing's added.
        for cmd in super().get_system_commands(screen):
            if cmd.title not in ("Keys", "Screenshot", "Theme", "Quit"):
                yield cmd


def run_tui(
    session_dir: Path,
    *,
    exit_on_end: bool = False,
    from_hub: bool = False,
    config_path: Path | None = None,
) -> int:
    app = Agent6TUI(
        session_dir, exit_on_end=exit_on_end, from_hub=from_hub, config_path=config_path
    )
    rc = app.run() or 0
    if app.detached:
        sid = session_dir.name
        if exit_on_end:
            # The lifecycle in the parent prints the reattach line once the run
            # has been handed to the background (after its current step).
            print(f"[agent6] leaving the view: {sid} detaches to the background after this step.")
        else:
            print(f"[agent6] detached: {sid} keeps running.")
            print(f"          reattach:  agent6 attach {sid}")
        print("          (Ctrl+_ undoes typing in the composer)")
    return rc
