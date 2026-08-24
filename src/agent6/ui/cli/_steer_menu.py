# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The interactive pause menu for a foreground CLI run (Ctrl-C, then decide).

Line input comes from `_menu_input` on Unix: editing, history recall (Up
recalls, Ctrl-R searches; seeded from the session journal, so it spans resumes
and steers typed on other surfaces), and a fish-style Tab preview of the slash
commands (Tab cycles the matches, descriptions shown). Windows has no termios,
so it keeps the plain one-line prompt (`_steer` gates on
:func:`agent6.ui.cli._menu_input.menu_capable`). Info commands answer from the
run's event log and re-prompt, so the operator can inspect the run before
steering it.

Parsing rule: a command fires only when it is the WHOLE line (one `/token`;
a unique prefix like `/sta` works, an ambiguous one re-asks). Any line with
a space -- or not starting with `/` -- is sent to the run verbatim as the
steering instruction, so no quoting is ever needed:

    /status   run status: tasks, tools, cost, ctx, preset
    /tasks    the task graph with statuses
    /compact  compact the context now (`/compact <focus>` steers the summary)
    /continue resume unchanged (same as Enter)
    /stop     stop the run now (resumable with `agent6 resume`)
    /detach   keep the run going in the background

`/parallel [spec] <task>` is a steer directive, not a menu command: it is sent
to the run verbatim and the loop fans out a sibling lane group for it. The spec
is an optional lane count or model list (omitted = one lane; a first token with
a comma or slash reads as the spec, a bare model name reads as task text);
repeat the exact `/parallel` token to queue more tasks in one message. See
`agent6.directive.parse_directive`.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from agent6.config.layer import load_effective
from agent6.directive import STEER_COMMANDS, parse_btw
from agent6.paths import data_dir
from agent6.sessions.ipc import request_compact
from agent6.sessions.layout import LOGS_NAME
from agent6.sessions.manifest import ManifestError, read_manifest
from agent6.skills import discover_skills, resolve_states, skill_search_dirs
from agent6.tools.background import roster_from_dir
from agent6.ui.cli._menu_input import menu_capable, menu_input
from agent6.viewmodel import (
    fold_session,
    operator_inputs,
    restate,
    status_for_session_dir,
    tail_events,
    task_snippet,
)
from agent6.viewmodel.format import TASK_STATUS_GLYPH, format_cost, status_label
from agent6.viewmodel.state import SessionState, context_fill, status_facts

PROMPT = "[agent6] paused: Enter=continue · type to steer · /help: "

# Command -> one-line help. The Tab preview menu and /help both read this table.
MENU_COMMANDS: dict[str, str] = {
    "/status": "run status: tasks, tools, cost, context, preset",
    "/tasks": "the task graph with statuses",
    "/pin": "list pinned instructions (pin one with `/pin <text>`)",
    "/compact": "compact the context now; `/compact <focus>` steers the summary",
    "/parallel": STEER_COMMANDS["/parallel"],
    "/btw": "ask a question beside the run: `/btw <question>` (answers inline, later)",
    "/shells": "background commands this run started, and how they ended",
    "/restate": "restate the conversation since your last message",
    "/undo": "fork back to before your last message (the text returns to edit and resend)",
    "/continue": "resume the run unchanged (same as Enter)",
    "/stop": "stop the run now (resume later with `agent6 resume`)",
    "/exit": "stop the run and leave (no follow-up prompt; resume later)",
    "/detach": "keep the run going in the background",
    "/help": "this list",
}


def _without_btw(config_path: Path | None = None) -> dict[str, str]:
    """The menu minus `/btw`, for a surface with nothing to spawn one from."""
    return {cmd: help_ for cmd, help_ in MENU_COMMANDS.items() if cmd != "/btw"}


def skill_menu_table(config_path: Path | None = None) -> dict[str, tuple[str, str]]:
    """`/name` -> (description, full SKILL.md text) for enabled skills.

    Built-in commands always win a name collision, so `/status` can never
    be shadowed by a skill. A broken config or store degrades to no skill
    commands, loudly, without breaking the pause prompt.
    """
    try:
        cfg = load_effective(Path.cwd(), config_path).config
        if not cfg.skills.enabled:
            return {}
        found, _warns = discover_skills(
            skill_search_dirs(cfg.skills.extra_dirs, data_dir() / "skills")
        )
        resolved = resolve_states(found, cfg.skills.state)
    except Exception as exc:  # the pause prompt must survive any config error
        print(f"[agent6] skill commands unavailable: {exc}")
        return {}
    return {
        f"/{s.name}": (s.description, s.text)
        for s in (*resolved.enabled, *resolved.always)
        if f"/{s.name}" not in MENU_COMMANDS
    }


def skill_steer_payload(name: str, text: str, args: str) -> str:
    """The steer message a `/skill-name [args]` menu line injects."""
    args_line = f"\nSkill arguments: {args}" if args else ""
    return (
        f"Apply the operator-installed skill {name!r} for the rest of this run."
        f"{args_line}\n\n"
        f'<skill name="{name}">\n{text.rstrip()}\n</skill>'
    )


def normalize_steer_choice(line: str | None) -> str | None:
    """Map a mid-run menu line to a canonical action: None/'' continue,
    'abort' stop, 'exit' stop-and-leave, 'detach' keep-running-in-background,
    else the instruction."""
    if line is None:
        return None
    choice = line.strip()
    low = choice.lower()
    if low in ("q", "quit", "stop", "abort"):
        return "abort"
    if low == "exit":
        return "exit"
    if low in ("d", "detach"):
        return "detach"
    return choice


@dataclass(slots=True)
class _Recall:
    """The pause prompt's history (Up recalls, Ctrl-R searches): seeded once
    per session from its journal, then grown with the lines accepted this
    process."""

    lines: list[str] = field(default_factory=list)
    seeded_from: str | None = None

    def seed(self, session_dir: Path) -> None:
        """The task, then every steer, newlines flattened for the one-line
        reader. A session seeds once; reseeding a later pause would drop the
        lines accepted since."""
        if self.seeded_from == str(session_dir):
            return
        self.seeded_from = str(session_dir)
        recorded = operator_inputs(tail_events(session_dir / LOGS_NAME, follow=False))
        self.lines[:] = [" ".join(text.split()) for text in recorded]


_RECALL = _Recall()


def _fold(session_dir: Path) -> SessionState:
    return fold_session(tail_events(session_dir / LOGS_NAME, follow=False))


def _read_preset(session_dir: Path) -> str:
    """The effective preset the run started with (manifest.json), or ""."""
    try:
        return read_manifest(session_dir).workflow.preset
    except ManifestError:
        return ""


def _print_status(session_dir: Path) -> None:
    s = _fold(session_dir)
    # THE dir decision, not the fold alone: an attached run's worker can be
    # gone ("stale"), and the fold-only label called that "running".
    label = status_label(*status_for_session_dir(session_dir, status_facts(s)))
    done = sum(1 for t in s.tasks if t.status in ("passed", "skipped"))
    tasks = f"{done}/{len(s.tasks)}" if s.tasks else "—"
    role = s.last_role
    model = f"{role.role}/{role.model}" if role else "—"
    cost = format_cost(s.budget.usd_total, partial=s.budget.usd_partial)
    ctx = ""
    if role is not None and role.ctx_tokens > 0:
        fill = context_fill(s)
        pct = f" ({fill}%)" if fill is not None else ""
        ctx = f" · ctx {role.ctx_tokens:,} tok{pct}"
    if s.compact_elided:
        ctx += f" · elided {s.compact_elided} ({s.compact_gists_live} gists)"
    if s.pins:
        ctx += f" · pins {len(s.pins)}"
    preset = _read_preset(session_dir)
    prof = f" · preset {preset}" if preset else ""
    print(f"[agent6] {label} · tasks {tasks} · {len(s.tool_calls)} tools · cost {cost}{ctx}{prof}")
    print(f"         model {model} · task: {task_snippet(s.user_task, max_chars=80)}")


def _print_pins(session_dir: Path) -> None:
    """Bare /pin: the recorded pins (fold truth). `/pin <text>` has a space, so
    the menu sends it verbatim as a steer and the loop's parser records it."""
    s = _fold(session_dir)
    if not s.pins:
        print("[agent6] no pinned instructions; pin one with `/pin <text>`")
        return
    print(f"[agent6] {len(s.pins)} pinned (survive context compaction; `/pin <text>` adds):")
    for i, pin in enumerate(s.pins, start=1):
        print(f"  {i}. {pin}")


def _print_tasks(session_dir: Path) -> None:
    s = _fold(session_dir)
    if not s.tasks:
        print("[agent6] (no tasks yet)")
        return
    for tv in s.tasks:
        icon = TASK_STATUS_GLYPH.get(tv.status, "·")
        marker = "▸ " if tv.is_cursor else ""
        print(f"  {'  ' * tv.depth}{marker}{icon} {tv.title}")


def _print_help(offered: dict[str, str]) -> None:
    width = max(len(c) for c in offered)
    for cmd, what in offered.items():
        print(f"  {cmd:<{width}}  {what}")
    print("  anything else is sent to the run as a steering instruction")
    print("  Up recalls this session's messages · Ctrl-R searches them · Tab previews commands")


# Starts a btw and delivers the finished answer to the console view. The menu
# owns the grammar; the CLI owns the spawn and the delivery. None (headless,
# tests) makes `/btw` say so rather than fail obscurely.
BtwRunner = Callable[[str, Path], str]


def _print_shells(session_dir: Path) -> None:
    """The run's background commands. Read off disk, not from the dispatcher:
    the menu answers from the same place every other surface reads."""
    lines = roster_from_dir(session_dir / "shells")
    if not lines:
        print("[agent6] no background commands this run")
        return
    for line in lines:
        print(f"  {line}")


def _start_btw(cmd: str, session_dir: Path, runner: BtwRunner | None) -> str:
    question = parse_btw(cmd)
    if not question:
        return "[agent6] ask something: `/btw <question>`"
    if runner is None:
        return "[agent6] /btw needs a live run with a terminal"
    return runner(question, session_dir)


# Commands that end the menu, mapped to the canonical steer action.
_ACTIONS: dict[str, str] = {
    "/continue": "",
    "/stop": "abort",
    "/exit": "exit",
    "/detach": "detach",
    # Verbatim: the loop parses the directive itself (fork + session.undone).
    "/undo": "/undo",
}


def _run_info_command(
    cmd: str,
    session_dir: Path,
    btw_runner: BtwRunner | None = None,
    config_path: Path | None = None,
) -> None:
    """Run a print-and-re-prompt command (everything not in `_ACTIONS`)."""
    if cmd == "/help":
        _print_help(MENU_COMMANDS if btw_runner is not None else _without_btw(config_path))
    elif cmd == "/status":
        _print_status(session_dir)
    elif cmd == "/tasks":
        _print_tasks(session_dir)
    elif cmd == "/pin":
        _print_pins(session_dir)
    elif cmd == "/compact":
        if request_compact(session_dir):
            print("[agent6] compaction requested; applies before the next model call")
        else:
            print("[agent6] could not write the compaction request; nothing was requested")
    elif cmd == "/parallel":
        print("[agent6] fan out needs a task: `/parallel [N|models] <task>`")
    elif cmd == "/shells":
        _print_shells(session_dir)
    elif cmd == "/restate":
        print(restate(list(tail_events(session_dir / LOGS_NAME, follow=False))))
    elif cmd.startswith("/btw"):
        print(_start_btw(cmd, session_dir, btw_runner))


def pause_menu(  # noqa: PLR0911, PLR0912
    session_dir: Path,
    *,
    input_fn: Callable[[str], str] | None = None,
    btw_runner: BtwRunner | None = None,
    config_path: Path | None = None,
) -> str | None:
    """The interactive pause menu. Returns the canonical steer action: None/''
    continue, 'abort' stop now, 'exit' stop-and-leave, 'detach' background,
    else the instruction sent
    verbatim. A command must be the whole line (unique prefixes fire, ambiguous
    ones re-ask); info commands print and re-prompt. EOF (Ctrl-D) continues."""
    skills = skill_menu_table(config_path)
    # A surface that cannot spawn a sibling session never offers `/btw`: an
    # offered command that answers "needs a live run" is not offered.
    offered = MENU_COMMANDS if btw_runner is not None else _without_btw(config_path)
    if input_fn is None:
        if menu_capable():
            _RECALL.seed(session_dir)
            display = {**offered, **{c: d[:70] for c, (d, _t) in skills.items()}}
            input_fn = lambda p: menu_input(p, display, _RECALL.lines)  # noqa: E731
        else:
            input_fn = input
    while True:
        try:
            line = input_fn(PROMPT)
        except EOFError:
            return None
        stripped = line.strip()
        if not stripped:
            return ""  # Enter: continue the run unchanged
        if not stripped.startswith("/"):
            return stripped  # a steering instruction, sent verbatim
        first, _, args = stripped.partition(" ")
        word = first.lower()
        if word in ("/h", "/?"):
            word = "/help"
        if args:
            # /compact, /btw and skill commands take arguments; any other line
            # with spaces stays a verbatim steer (the loop itself parses the
            # /pin and /parallel directives out of steer text).
            builtin = [c for c in MENU_COMMANDS if c.startswith(word)]
            smatches = [word] if word in skills else [c for c in skills if c.startswith(word)]
            if builtin == ["/compact"] and not smatches:
                if request_compact(session_dir, focus=args.strip()):
                    print(
                        "[agent6] compaction requested (focus noted);"
                        " applies before the next model call"
                    )
                else:
                    print("[agent6] could not write the compaction request; nothing was requested")
                continue
            if builtin == ["/btw"] and not smatches:
                # A btw is a question asked BESIDE the run; letting it fall
                # through would send it to the loop as steer text instead.
                print(_start_btw(stripped, session_dir, btw_runner))
                continue
            if len(smatches) == 1 and not builtin:
                return skill_steer_payload(smatches[0][1:], skills[smatches[0]][1], args.strip())
            return stripped
        if word in MENU_COMMANDS or word in skills:  # exact match (never both: the
            # table builder drops skills that collide with a built-in)
            matches = [word]
        else:
            builtin = [c for c in MENU_COMMANDS if c.startswith(word)]
            matches = builtin + [c for c in skills if c.startswith(word) and c not in builtin]
        if len(matches) > 1:
            print(f"[agent6] ambiguous: {'  '.join(matches)} (type more)")
        elif not matches:
            print(
                f"[agent6] unknown command {word!r}; /help lists them"
                " (a line with spaces is sent as a steer)"
            )
        elif matches[0] in _ACTIONS:
            return _ACTIONS[matches[0]]
        elif matches[0] in skills:
            return skill_steer_payload(matches[0][1:], skills[matches[0]][1], "")
        else:
            _run_info_command(matches[0], session_dir, btw_runner, config_path)
