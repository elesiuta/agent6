# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The `agent6 tui` Machines page: browse, view, run, and create state machines.

A separate page from the run hub (machines are not runs). Like the hub's run
actions, it never drives a machine in-process: Run and Create shell out to
`agent6 machine run|create` (detached). View is the one in-process step -- it
parses the .asm.toml via agent6.machine to show the machine's structure,
validation, and graph -- which is why ui depends on agent6.machine for this page
(see tach.toml).
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from pathlib import Path
from typing import ClassVar

try:
    from rich.text import Text
    from textual.app import App, ComposeResult
    from textual.binding import Binding
    from textual.containers import Container, Horizontal, VerticalScroll
    from textual.notifications import SeverityLevel
    from textual.screen import ModalScreen, Screen
    from textual.widgets import DataTable, Footer, Input, RichLog, Static
except ImportError as e:  # pragma: no cover - clear runtime message
    raise ImportError(
        "agent6 TUI requires the 'textual' package (part of the base install)."
        " Reinstall agent6, or `pip install textual`."
    ) from e

from agent6.app.machine.listing import MachineRow, machine_rows
from agent6.machine import (
    JournalError,
    MachineError,
    MachineJournal,
    MachineSpec,
    load_machine,
    render,
    validate_semantics,
    write_stop_request,
)
from agent6.sessions.ipc import (
    clear_steer_answer,
    read_worker_pid,
    register_frontend,
    request_steer,
    unregister_frontend,
    write_steer_answer,
)
from agent6.sessions.layout import LOGS_NAME, bucket_dir, machines_root
from agent6.ui.notify import desktop_notify
from agent6.ui.spawn import agent6_argv, spawn_and_confirm, spawn_and_locate
from agent6.ui.tui.menubar import Menu, MenuBar, MenuItem, menu_bindings
from agent6.ui.tui.modals import (
    ConfirmModal,
    SteerModal,
    TextInputModal,
)
from agent6.ui.tui.prompts import PromptDispatcher
from agent6.ui.tui.screen_chrome import MenuCommands, ScreenChrome
from agent6.ui.tui.theme import (
    PALETTE_CSS,
    MuxPointerShapes,
    PlainNotify,
    setup_theme,
    status_style,
)
from agent6.viewmodel import (
    MachineState,
    MachineWatchCursor,
    fold_machine,
    fold_session,
    machine_verb_refusal,
    machine_word_for_dir,
    newest_state_log,
    tail_events,
)
from agent6.viewmodel.events import tool_result_ok
from agent6.viewmodel.format import (
    format_transition,
    format_when,
    machine_state_mark,
    status_label,
)


def _list_drafts(agent6_dir: Path) -> list[Path]:
    """Machine-create draft dirs (newest first). Each holds the watchable
    logs.jsonl + prompt.txt + the authored candidate."""
    drafts = bucket_dir(agent6_dir, "machines")
    if not drafts.is_dir():
        return []
    out = [p for p in drafts.iterdir() if p.is_dir()]
    out.sort(key=lambda p: p.stat().st_mtime if p.exists() else 0.0, reverse=True)
    return out


def _discrete_log_line(evt: dict[str, object]) -> Text | None:
    """A compact line for a non-streaming agent-log event (tool calls, role start),
    or None to skip it. Thinking/text deltas are accumulated separately."""
    t = evt.get("type")
    if t == "role.call":
        return Text(f"  → {evt.get('role', '')}/{evt.get('model', '')} thinking…", style="cyan")
    if t == "tool.call":
        args = json.dumps(evt.get("args", {}), default=str)
        if len(args) > 80:
            args = args[:77] + "…"
        return Text(f"  ⚙ {evt.get('name', '')} {args}", style="yellow")
    if t == "tool.result":
        ok = tool_result_ok(evt.get("ok"))
        mark = "✓" if ok else "✗"
        return Text(f"  {mark} {evt.get('summary', '')}", style="green" if ok else "red")
    return None


# Answer submitted after the machine stopped mid-modal: the state's loop has
# exited, so the per-state answer file has no reader; a poke wakes a waiting one.
_ANSWER_LOST = (
    "machine is not running; the answer reached nothing (poke it to wake a waiting machine)"
)


class MachineWatchScreen(ScreenChrome, Screen[None]):
    """Live view of a running (or finished) machine: the state overview with the
    current state marked, each transition as it lands, and the active agent
    state's reasoning streamed from its per-state logs.jsonl -- the in-TUI
    equivalent of `agent6 attach`. Polls every 0.5s.

    Interactive: while open it registers as an answer front-end (a frontends/ claim on
    the instance dir), so the current agent state's `run_command` approvals and
    `ask_user` questions pop as modals here; `s` steers that state, `m` sends a
    message (a poke payload) to a waiting machine, and a `machine.notify` (or the
    machine's completion) fires a desktop + in-app notification."""

    HELP_TITLE: ClassVar = "agent6 machine — keys & actions"
    MENUS: ClassVar = (
        Menu("File", (MenuItem("Back", "close"),)),
        Menu(
            "Machine",
            (
                MenuItem("Steer the running state", "steer", "s"),
                MenuItem("Message a waiting machine", "poke", "m"),
                MenuItem("Stop at the next transition", "stop", "x"),
            ),
        ),
        Menu(
            "Help",
            (
                MenuItem("Keys & actions", "help", "question_mark"),
                MenuItem("Command palette", "command_palette", "ctrl+p"),
            ),
        ),
    )
    COMMANDS: ClassVar = Screen.COMMANDS | {MenuCommands}
    BINDINGS: ClassVar = [
        Binding("s", "steer", "Steer"),
        Binding("m", "poke", "Message"),
        Binding("x", "stop", "Stop"),
        Binding("question_mark", "help", "Help"),
        Binding("escape", "close", "Back", key_display="Esc/q"),
        Binding("q", "close", "Back", show=False),
        *menu_bindings(MENUS),
    ]
    CSS = (
        PALETTE_CSS
        + """
    MachineWatchScreen { layers: base; }
    #mw-head { height: 3; border: round $primary; padding: 0 1; }
    #mw-states { width: 32%; border: round $primary; }
    #mw-log { width: 1fr; border: round $primary; padding: 0 1; }
    """
    )

    def __init__(self, instance_dir: Path, spec: MachineSpec) -> None:
        super().__init__()
        self._root = instance_dir
        self._spec = spec
        self._journal = MachineJournal(instance_dir)
        self._cursor = MachineWatchCursor()
        self._pending = ""  # accumulated thinking/answer text, flushed in readable chunks
        self._ended = False
        self._was_steerable = True  # the footer's Steer key tracks _steerable() flips
        self._prompts = PromptDispatcher(self.app, answerable=self._answerable, lost=_ANSWER_LOST)
        self._end_notified = False
        self._steer_open = False

    def compose(self) -> ComposeResult:
        yield MenuBar(self.MENUS)
        yield Static(id="mw-head")
        with Horizontal():
            yield DataTable(id="mw-states", cursor_type="none")
            yield RichLog(id="mw-log", wrap=True, markup=False, highlight=False)
        yield Footer()

    def on_mount(self) -> None:
        table = self.query_one("#mw-states", DataTable)
        table.add_column(" ", key="mark")
        table.add_column("state", key="state")
        table.add_column("kind", key="kind")
        for name, state in self._spec.states.items():
            table.add_row("", name, state.kind, key=name)
        # Register as the answer front-end on the instance dir: a machine agent
        # state's approval/question/steer prompts bridge here while we watch. Seed
        # notification history so past notifies are not re-announced on open. The
        # dir may not exist yet (watching a just-spawned run), so create it first.
        self._root.mkdir(parents=True, exist_ok=True)
        # Per-process claim file: nothing to defend or re-assert, concurrent
        # web/TUI watchers each hold their own.
        register_frontend(self._root, os.getpid())
        try:
            events = self._journal.read()
        except JournalError:
            events = []  # the first poll surfaces the corruption in the header
        seeded = fold_machine(self._spec, events)
        self._cursor.seed_notifications(seeded)
        # An end that predates the open is history, not news (same as the web's
        # endedNotified seed); a machine ending WHILE watched still announces.
        self._end_notified = seeded.ended is not None
        self._poll()
        self.set_interval(0.5, self._poll)

    def on_unmount(self) -> None:
        # Drop only our own front-end claim; concurrent watchers keep theirs.
        unregister_frontend(self._root, os.getpid())

    def _current_state_dir(self) -> Path | None:
        """The current agent state's per-state dir (where its answer files live)."""
        log = newest_state_log(self._root)
        return log.parent if log is not None else None

    def action_close(self) -> None:
        # Standalone `agent6 attach <machine> --tui` mounts this directly on the
        # app's base screen, so there is nothing to pop back to; exit instead.
        if len(self.app.screen_stack) > 2:
            self.app.pop_screen()
        else:
            self.app.exit()

    def check_action(self, action: str, parameters: tuple[object, ...]) -> bool | None:
        # An ended machine takes no input: dim Steer/Message so the footer never
        # offers a control that would drop into a dead instance dir (matches the
        # web, which disables both buttons once the machine has ended). Stop
        # keeps its key: on a machine that cannot stop, the action prints the
        # refusal the CLI prints (Help lists the key either way).
        del parameters
        if action in ("steer", "poke"):
            return not machine_verb_refusal(self._root, self._root.name, action)
        return True

    def _steerable(self) -> bool:
        """A steer only reaches an agent state the worker is actively running
        (`machine_verb_refusal`): the footer's Steer key and the answer gate
        read this."""
        return not machine_verb_refusal(self._root, self._root.name, "steer")

    def _answerable(self) -> bool:
        return not machine_verb_refusal(self._root, self._root.name, "answer")

    def action_steer(self) -> None:
        """Steer the current agent state: drop a request marker + open the steer
        box; the state picks it up at its next safe boundary. No-op if none runs."""
        if refusal := machine_verb_refusal(self._root, self._root.name, "steer"):
            self.app.notify(refusal, timeout=6.0)
            return
        state_dir = self._current_state_dir()
        if state_dir is None or self._steer_open:
            self.app.notify("no agent state to steer", timeout=4.0)
            return
        self._steer_open = True
        clear_steer_answer(state_dir)
        request_steer(state_dir)
        self.app.push_screen(SteerModal(), self._on_steer(state_dir))

    def _on_steer(self, state_dir: Path) -> Callable[[str | None], None]:
        def cb(answer: str | None) -> None:
            self._steer_open = False
            write_steer_answer(state_dir, answer or "")

        return cb

    def action_stop(self) -> None:
        """Ask the running machine to park at its next transition boundary
        (the durable stop marker; the instance stays resumable)."""
        if refusal := machine_verb_refusal(self._root, self._root.name, "stop"):
            self.app.notify(refusal, timeout=6.0)
            return
        write_stop_request(self._root)
        self.app.notify("stop requested; the machine parks at its next boundary", timeout=4.0)

    def action_poke(self) -> None:
        """Send a message to a waiting machine (a poke payload the next tool reads)."""
        if self._ended:
            self.app.notify("machine ended; cannot send a message", timeout=4.0)
            return
        self.app.push_screen(
            TextInputModal("Send a message to the machine (poke):", "message…"), self._on_poke
        )

    def _on_poke(self, message: str | None) -> None:
        if message is None:
            return
        self._journal.poke(message or None)
        self.app.notify("poked", timeout=3.0)

    def _flush_pending(self) -> None:
        text = self._pending.strip()
        self._pending = ""
        if text:
            self.query_one("#mw-log", RichLog).write(Text(f"  {text}", style="dim"))

    def _poll(self) -> None:
        if self._ended:
            return
        # The footer's Steer key follows liveness, not just the _ended edge: a
        # --exit-on-wait park or a killed worker flips _steerable() with no
        # MachineEnd, and the lit key otherwise offered a steer nobody reads.
        steerable = self._steerable()
        if steerable != self._was_steerable:
            self._was_steerable = steerable
            self.refresh_bindings()
        try:
            ms = fold_machine(self._spec, self._journal.read())
        except JournalError as exc:
            # A corrupt journal line must not crash the screen every poll tick;
            # show it and keep polling (an append may heal or end the run).
            self.query_one("#mw-head", Static).update(Text(f"journal unreadable: {exc}"))
            return

        # Header + state-table markers. A parked (--exit-on-wait) instance reads
        # "waiting", not "running", so a paused machine never looks busy.
        if ms.ended is not None:
            status = f"ended: {ms.ended.status} ({ms.ended.reason})"
        else:
            status = f"{machine_word_for_dir(ms, self._root)} · {ms.current}"
        self.query_one("#mw-head", Static).update(
            Text(f"machine: {ms.machine}   {status}   transitions: {len(ms.transitions)}")
        )
        table = self.query_one("#mw-states", DataTable)
        for s in ms.states:
            mark = machine_state_mark(is_current=s.is_current, is_visited=s.is_visited)
            table.update_cell(s.name, "mark", mark)

        # Mark ended BEFORE rendering the log so a terminal instance's final
        # agent state doesn't render a live "thinking…" line (see C13).
        if ms.ended is not None and not self._ended:
            self._ended = True
            self.refresh_bindings()  # dim Steer/Message: a dead machine takes no input
        # Liveness for the render + prompt gate, recomputed AFTER the _ended
        # flip above (the line-286 `steerable` is for the footer edge and is
        # stale-True on the poll that first observes the end).
        live = self._steerable()

        log = self.query_one("#mw-log", RichLog)
        # New transitions.
        for t in self._cursor.new_transitions(ms):
            self._flush_pending()
            log.write(
                Text(format_transition(t.seq, t.state, t.label, t.goto, t.detail), style="bold")
            )

        # The current agent state's reasoning (switch logs as states change).
        newest, switched = self._cursor.advance_log(self._root)
        if switched:
            self._flush_pending()
            if newest is not None:
                log.write(Text(f"-- agent state: {newest.parent.name} --", style="cyan bold"))
        self._render_log_lines(log, live=live)
        self._flush_pending()  # show partial reasoning each tick

        self._dispatch_notifications(ms)
        self._dispatch_prompts(live=live)

    def _dispatch_notifications(self, ms: MachineState) -> None:
        """Pop an in-app + desktop notification for each new machine.notify, and
        once for the machine's completion. Dedup by identity (not a count) since
        ms.notifications is a sliding window."""
        for n in self._cursor.new_notifications(ms):
            sev: SeverityLevel = (
                "warning" if n.level == "warn" else "error" if n.level == "error" else "information"
            )
            self.app.notify(n.message, title=f"{ms.machine} · {n.state}", severity=sev, timeout=8.0)
            desktop_notify(f"agent6: {ms.machine}", n.message)
        ended = ms.ended
        if ended is not None and not self._end_notified:
            self._end_notified = True
            self.app.notify(
                ended.reason,
                title=f"{ms.machine} {ended.status}",
                severity="information" if ended.status == "ok" else "error",
                timeout=8.0,
            )
            desktop_notify(f"agent6: {ms.machine} {ended.status}", ended.reason)

    def _dispatch_prompts(self, *, live: bool) -> None:
        """Pop approval/question modals for the current agent state's pending
        prompts, writing answers back to that state's per-state dir.

        Only while the machine is RUNNING: a parked/stopped/ended instance's
        newest agent state is finished, so the fold still carries an unanswered
        prompt but nothing would poll the answer -- popping live-looking
        Allow/Deny (a destructive-command approval among them) over a dead
        machine is the machine twin of the run-modal gate."""
        if not live:
            return
        state_dir = self._current_state_dir()
        if state_dir is None:
            return
        self._prompts.dispatch(
            state_dir, fold_session(tail_events(state_dir / LOGS_NAME, follow=False))
        )

    def _render_log_lines(self, log: RichLog, *, live: bool) -> None:
        """Render new complete lines of the current state log: accumulate
        thinking/answer text in self._pending, write discrete events (tool
        calls) inline. The byte-offset cursor (partial trailing lines stay
        unconsumed) lives in MachineWatchCursor."""
        for raw in self._cursor.read_log_lines():
            try:
                evt = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if not isinstance(evt, dict):
                continue
            etype = evt.get("type")
            if etype in ("role.thinking_delta", "role.text_delta"):
                self._pending += str(evt.get("text", ""))
                continue
            if not live and etype == "role.call":
                # A stopped/parked/ended instance isn't "thinking…"; drop the
                # phantom (was gated on `_ended` only, so a killed worker or a
                # parked machine still ticked a live role.call line).
                continue
            discrete = _discrete_log_line(evt)
            if discrete is not None:
                self._flush_pending()
                log.write(discrete)


def machine_detail_text(path: Path) -> str:
    """A read-only text view of a parsed machine: name, initial, states, validation,
    and the mermaid graph. On a load error, the error itself (so the page never
    crashes on a half-written file)."""
    try:
        spec = load_machine(path)
    except (MachineError, OSError) as exc:
        return f"failed to load {path.name}:\n\n{exc}"
    lines = [
        f"machine: {spec.machine}",
        f"initial: {spec.initial}",
        "",
        f"states ({len(spec.states)}):",
    ]
    # `state.kind` (agent/tool/wait/branch/terminal): the user word the watch
    # screen and the web detail already show, not the internal class name.
    lines.extend(f"  {name}  ({state.kind})" for name, state in spec.states.items())
    problems = validate_semantics(spec)
    lines.append("")
    if problems:
        lines.append(f"semantics: {len(problems)} problem(s)")
        lines.extend(f"  - {p}" for p in problems)
    else:
        lines.append("semantics: OK  (`agent6 machine check` also lints the bundle's scripts)")
    lines += ["", "graph (mermaid):", render(spec, "mermaid")]
    return "\n".join(lines)


class MachineDetailScreen(Screen[None]):
    """Read-only view of one parsed machine (structure + validation + graph)."""

    CSS = """
    MachineDetailScreen { background: $surface; }
    #machine-detail-title {
        dock: top; height: 1; padding: 0 1; background: $panel; text-style: bold;
    }
    #machine-detail-body { height: 1fr; padding: 0 1; }
    #machine-detail-body Static { pointer: text; }  /* selectable text: I-beam */
    """

    BINDINGS: ClassVar = [
        Binding("escape", "close", "Back", key_display="Esc/q"),
        Binding("q", "close", "Back", show=False),
    ]

    def __init__(self, path: Path) -> None:
        super().__init__()
        self._path = path

    def compose(self) -> ComposeResult:
        yield Static(f"machine · {self._path.name}", id="machine-detail-title")
        with VerticalScroll(id="machine-detail-body"):
            yield Static(Text(machine_detail_text(self._path)))  # plain Text: no markup parsing
        yield Footer()

    def on_mount(self) -> None:
        self.query_one("#machine-detail-body", VerticalScroll).focus()

    def action_close(self) -> None:
        self.dismiss()


class CreateMachineModal(ModalScreen[str]):
    """Prompt for a natural-language task to author a machine. Result: the task text
    (or "" if cancelled)."""

    DEFAULT_CSS = """
    CreateMachineModal { align: center middle; }
    #create-box {
        width: 80%; max-width: 100; height: auto;
        border: round $accent; padding: 1 2; background: $surface;
    }
    #create-input { margin-top: 1; }
    """

    BINDINGS: ClassVar = [Binding("escape", "cancel", "Cancel", show=False)]

    def compose(self) -> ComposeResult:
        with Container(id="create-box"):
            text = Text()
            text.append("Create a machine\n\n", style="bold")
            # Split at the clause boundary so a narrow terminal (the box is 80%
            # wide) never wraps mid-phrase.
            text.append("Describe the loop to author;\nagent6 drafts a .asm.toml in this repo.")
            yield Static(text)
            yield Input(
                placeholder="e.g. nightly: pull, run tests, open an issue on failure",
                id="create-input",
            )

    def on_mount(self) -> None:
        self.query_one("#create-input", Input).focus()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        self.dismiss(event.value.strip())

    def action_cancel(self) -> None:
        self.dismiss("")


class MachinesScreen(ScreenChrome, Screen[None]):
    """List authored state machines; view (parsed), run, or create them. Run/Create
    shell out to the CLI (detached); View parses in-process."""

    CSS = (
        PALETTE_CSS
        + """
    MachinesScreen { layers: base dropdown; }
    #machines { height: 1fr; border: round $primary; background: $surface; }
    #machines:focus { border: round $accent; }
    #machines > .datatable--header { background: $panel; color: $foreground; text-style: bold; }
    #machines > .datatable--cursor { background: transparent; color: $foreground; }
    #machines:focus > .datatable--cursor {
        background: $primary 40%; color: $text; text-style: bold;
    }
    """
    )
    MENUS: ClassVar = (
        # File/<page>/View/Help, the same shape as the hub and the dashboard.
        Menu(
            "File",
            (
                MenuItem("Back", "close", "Esc/q"),
                MenuItem("Quit", "quit", "ctrl+q"),
            ),
        ),
        Menu(
            "Machines",
            (
                MenuItem("View", "view", "v"),
                MenuItem("Run", "run", "r"),
                MenuItem("Watch", "watch", "w"),
                MenuItem("Create…", "create", "c"),
                MenuItem("Refresh", "refresh", "f"),
            ),
        ),
        Menu(
            "View",
            (
                MenuItem("Theme…", "choose_theme"),
                MenuItem("Copy method…", "choose_copy_method"),
            ),
        ),
        Menu(
            "Help",
            (
                MenuItem("Keys & actions", "help", "question_mark"),
                MenuItem("Command palette", "command_palette", "ctrl+p"),
            ),
        ),
    )
    BINDINGS: ClassVar = [
        Binding("v", "view", "View"),
        Binding("r", "run", "Run"),
        Binding("w", "watch", "Watch"),
        Binding("c", "create", "Create"),
        Binding("f", "refresh", "Refresh"),
        Binding("question_mark", "help", "Help"),
        Binding("escape", "close", "Back", key_display="Esc/q"),
        Binding("q", "close", "Back", show=False),
        *menu_bindings(MENUS),
    ]
    COMMANDS: ClassVar = Screen.COMMANDS | {MenuCommands}
    HELP_TITLE: ClassVar = "agent6 machines — keys & actions"
    HELP_HINTS: ClassVar = ("Enter opens the selected machine",)

    def __init__(self, agent6_dir: Path, repo_cwd: Path, config_path: Path | None = None) -> None:
        super().__init__()
        self.agent6_dir = agent6_dir  # per-repo state dir; machine-create drafts live under it
        self.repo_cwd = repo_cwd
        self.config_path = config_path
        self._machines: list[MachineRow] = []

    def compose(self) -> ComposeResult:
        yield MenuBar(self.MENUS)
        yield DataTable(id="machines")
        yield Footer()

    def on_mount(self) -> None:
        table = self.query_one("#machines", DataTable)
        table.cursor_type = "row"
        table.add_columns("machine", "status", "state", "updated", "states", "spec", "file")
        self._reload()

    def _reload(self) -> None:
        """The rows `agent6 machine` lists (`app.machine.listing.machine_rows`):
        each instance's status and current state joined with its authored
        file's validity, then the files no instance ran."""
        table = self.query_one("#machines", DataTable)
        table.clear()
        self._machines = machine_rows(self.repo_cwd, self.agent6_dir)
        for row in self._machines:
            status = (
                Text(status_label(row.status, row.reason), style=status_style(row.status))
                if row.status
                else Text("")
            )
            table.add_row(
                Text(row.name),
                status,
                row.current or "-",
                format_when(row.mtime) if row.mtime else "-",
                row.states,
                row.spec,
                Text(row.file.name if row.file is not None else "-"),
            )
        table.show_cursor = table.row_count > 0
        # The count in the title (the hub does the same); an empty table alone
        # reads as still loading, so say none exist and what draws one.
        n = len(self._machines)
        tally = (
            "no machines yet (c creates one)" if not n else f"{n} machine{'' if n == 1 else 's'}"
        )
        self.app.sub_title = f"machines · {self.repo_cwd} · {tally}"

    def _selected(self) -> Path | None:
        """The selected row's authored file; None for a row without one (an
        instance whose file is gone: watchable, not viewable or runnable)."""
        table = self.query_one("#machines", DataTable)
        if self._machines and 0 <= table.cursor_row < len(self._machines):
            return self._machines[table.cursor_row].file
        return None

    def check_action(self, action: str, parameters: tuple[object, ...]) -> bool | None:
        """Dim the row actions on a page with no selectable row: View and Run
        need an authored file, Watch needs a row of any kind, and all three were
        silent no-ops the footer still offered."""
        del parameters
        if action in ("view", "run"):
            return self._selected() is not None
        if action == "watch":
            return bool(self._machines)
        return True

    def on_data_table_row_selected(self, _event: DataTable.RowSelected) -> None:
        # Enter on a row opens the parsed view (the DataTable consumes Enter, so the
        # screen's binding never fires -- handle the row event itself, like the hub).
        self.action_view()

    def action_view(self) -> None:
        path = self._selected()
        if path is not None:
            self.app.push_screen(MachineDetailScreen(path))

    def action_run(self) -> None:
        path = self._selected()
        if path is None:
            return
        self.app.push_screen(
            ConfirmModal(
                f"Run machine {path.name}?",
                "Runs `agent6 machine run` (detached) and opens the live watch view: "
                "the state overview, each transition, and the agent state's reasoning. "
                "The run keeps going if you close the view.",
                confirm_label="Run",
            ),
            self._on_run_confirm(path),
        )

    def _on_run_confirm(self, path: Path) -> Callable[[bool | None], None]:
        def cb(confirmed: bool | None) -> None:
            if not confirmed:
                return
            try:
                spec = load_machine(path)
            except MachineError as exc:
                self.app.notify(f"cannot load {path.name}: {exc}", severity="error", timeout=8.0)
                return
            # Started = the child wrote its own pid as the instance worker.pid
            # (it does so right after taking the machine lock). A refusal (lock
            # held, network refusal, bad bundle) exits nonzero before that and
            # its stderr surfaces here instead of a watch screen on nothing.
            instance = machines_root(self.agent6_dir) / spec.machine
            argv = agent6_argv(self.config_path)
            err = spawn_and_confirm(
                [*argv, "machine", "run", str(path)],
                self.repo_cwd,
                started=lambda pid: read_worker_pid(instance) == pid,
            )
            if err:
                self.app.notify(err, severity="error", timeout=8.0)
                return
            self._open_watch(path)  # follow it live (it runs detached regardless)

        return cb

    def action_watch(self) -> None:
        """Open the live watch view for the selected machine's instance (whether it
        is currently running or has finished); an instance whose authored file
        is gone is watched from the source it recorded."""
        path = self._selected()
        if path is not None:
            self._open_watch(path)
            return
        table = self.query_one("#machines", DataTable)
        if self._machines and 0 <= table.cursor_row < len(self._machines):
            instance = machines_root(self.agent6_dir) / self._machines[table.cursor_row].name
            self._open_watch(instance / "machine.asm.toml")

    def _open_watch(self, path: Path) -> None:
        try:
            spec = load_machine(path)
        except MachineError as exc:
            self.app.notify(f"cannot load {path.name}: {exc}", severity="error", timeout=8.0)
            return
        instance = machines_root(self.agent6_dir) / spec.machine
        self.app.push_screen(MachineWatchScreen(instance, spec))

    def action_create(self) -> None:
        self.app.push_screen(CreateMachineModal(), self._on_create)

    def _on_create(self, task: str | None) -> None:
        if not task:
            return
        # Spawn `agent6 machine create` detached, then open the dashboard on the
        # draft it produces so the authoring agent's reasoning + tool calls are
        # watchable live, exactly like a run. The create keeps running detached,
        # so quitting the dashboard is safe.
        draft_dir, error = spawn_and_locate(
            [*agent6_argv(self.config_path), "machine", "create", "--", task],
            self.repo_cwd,
            before=set(_list_drafts(self.agent6_dir)),
            list_dirs=lambda: _list_drafts(self.agent6_dir),
        )
        if draft_dir is not None:
            self.app.exit(draft_dir)  # the hub loop opens the dashboard on it
        else:
            self.app.notify(
                error or "Could not start machine create.", severity="error", timeout=8.0
            )

    def action_refresh(self) -> None:
        self._reload()

    def action_quit(self) -> None:
        self.app.exit()

    def action_close(self) -> None:
        self.dismiss()


class _MachineWatchApp(PlainNotify, MuxPointerShapes, App[None]):
    """One-screen host for `agent6 attach <machine> --tui`: the same live machine
    view the Machines page opens, runnable straight from the CLI."""

    CSS = (
        PALETTE_CSS
        + """
    * { scrollbar-size-vertical: 1; scrollbar-size-horizontal: 1; }  /* match the other apps */
    /* A footer that does not fit clips (textual's default); the 1-row widget has no
       room for the scrollbar the universal rule gives it, which replaced every hint. */
    Footer { scrollbar-size-vertical: 0; scrollbar-size-horizontal: 0; }
    Input, TextArea { pointer: text; }
    """
    )

    def __init__(self, instance_dir: Path, spec: MachineSpec) -> None:
        super().__init__()
        self._instance = instance_dir
        self._spec = spec

    def on_mount(self) -> None:
        setup_theme(self)  # apply the saved theme before the first paint
        self.push_screen(MachineWatchScreen(self._instance, self._spec))


def run_machine_watch_tui(instance_dir: Path, spec: MachineSpec) -> int:
    return _MachineWatchApp(instance_dir, spec).run() or 0
