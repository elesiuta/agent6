# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""`agent6 machine` lifecycle subcommands: argv adaptation + console rendering.

The read-only commands (list/status/replay/poke/stop, the watch follower) load
+ render directly; run/create adapt argv and hand the lifecycle to
`app.machine` behind the `MachineFrontend` seam. The interactive
network-refusal resolver stays here (it needs a TTY). The offline authoring
gate (check/test/graph) is `machine_check`."""

from __future__ import annotations

import contextlib
import difflib
import json
import sys
import time
from pathlib import Path
from typing import Any

from agent6.app._setup import detect_env
from agent6.app.machine import (
    MachineFrontend,
    machine_network_refusal,
    machine_spend,
)
from agent6.app.machine.create import create_machine
from agent6.app.machine.listing import machine_rows
from agent6.app.machine.run import run_machine
from agent6.app.reporter import STDIO_REPORTER
from agent6.config import (
    Config,
    ConfigError,
)
from agent6.config.io import upsert_toml_leaf
from agent6.config.layer import (
    load_effective_with_overlay,
    repo_config_path_for,
    resolved_state_dir,
)
from agent6.machine import (
    EngineError,
    JournalError,
    MachineError,
    MachineJournal,
    PendingWait,
    StepEvent,
    ToolState,
    drive,
    load_machine,
    write_stop_request,
)
from agent6.paths import chown_to_real_user
from agent6.sandbox.detect import IsolationUnavailableError, resolve_isolation
from agent6.sessions.ipc import read_worker_pid, worker_is_alive
from agent6.sessions.layout import machines_root
from agent6.types import IsolationLevel
from agent6.ui.cli._common import plural, styled_status
from agent6.ui.cli.machine_check import _cmd_machine_test
from agent6.ui.cli.plan_watch import format_plain_event
from agent6.ui.notify import desktop_notify
from agent6.viewmodel import (
    MachineState,
    MachineWatchCursor,
    event_epoch,
    fold_machine,
    machine_operator_blocked,
    machine_verb_refusal,
    machine_word_for_dir,
)
from agent6.viewmodel.format import (
    format_transition,
    format_usd,
    format_when,
    machine_state_mark,
)
from agent6.viewmodel.machine_state import wait_line


def _cmd_machine_list() -> int:
    """List this repo's machines (`app.machine.listing.machine_rows`, the rows
    the TUI machines page shows): every instance newest first joined with the
    authored `.asm.toml` that declares it, then the files no instance ran."""
    cwd = Path.cwd()
    machines = machine_rows(cwd, resolved_state_dir(cwd))
    if not machines:
        print('no machines yet. Draft one with `agent6 machine create "<task>"`.')
        return 0
    color = sys.stdout.isatty()
    rows: list[tuple[str, str, str, str, str, str, str]] = []
    for m in machines:
        styled, plain = styled_status(m.status, m.reason, color=color) if m.status else ("", "")
        rows.append(
            (
                format_when(m.mtime) if m.mtime else "-",
                styled,
                plain,
                m.current or "-",
                m.name,
                m.spec,
                str(m.file.relative_to(cwd)) if m.file is not None else "-",
            )
        )
    status_w = max(6, *(len(plain) for _, _, plain, *_ in rows))
    state_w = max(5, *(len(r[3]) for r in rows))
    name_w = max(7, *(len(r[4]) for r in rows))
    spec_w = max(4, *(len(r[5]) for r in rows))
    print(
        f"{'updated':<11}  {'status':<{status_w}}  {'state':<{state_w}}  {'machine':<{name_w}}"
        f"  {'spec':<{spec_w}}  file"
    )
    for when, styled, plain, state, name, spec, file in rows:
        pad = " " * (status_w - len(plain))
        print(
            f"{when:<11}  {styled}{pad}  {state:<{state_w}}  {name:<{name_w}}"
            f"  {spec:<{spec_w}}  {file}"
        )
    return 0


def _safe_input(prompt: str) -> str | None:
    """`input` that returns None on EOF / non-interactive stdin instead of raising."""
    try:
        return input(prompt)
    except (EOFError, KeyboardInterrupt):
        return None


def _suggested_network_fix(
    cfg: Config, isolation: IsolationLevel, tool_states: list[ToolState]
) -> dict[str, str] | None:
    """The minimal sandbox-config change that lets this machine's tool states run
    ON THIS PROFILE, or None if no config change can (a tool that REQUIRES network
    isolation needs `strict`, which config can't conjure).

    Two refusals this resolves: a tool that opted in (`network = "host"`)
    under a config that blocks egress, and -- on `hardened`, which can't give any
    tool its own netns -- a plain tool refused under `network = "session"`. The
    returned dict is applied in order
    so `config set`-style sequential writes never trip the combo validator."""
    if not tool_states:
        return None
    has_allow = any(s.network == "host" for s in tool_states)
    has_block = any(s.network == "none" for s in tool_states)
    if has_block:
        # A tool REQUIRES no network; only strict's per-tool netns isolates it.
        return None
    if isolation == "strict":
        # Plain no-network tools already run on strict; only a tool that opted
        # into the network needs the explicit-per-tool egress mode.
        return {"sandbox.network": "only_explicit_states"} if has_allow else None
    if isolation == "hardened":
        # hardened can't isolate one tool's netns, so EVERY tool (networked
        # or not) shares the host network; only an explicit "host" states that
        # honestly.
        return {"sandbox.network": "host"}
    return None


def _resolve_network_refusal(  # noqa: PLR0911
    path: Path,
    refusal: str,
    cfg: Config,
    isolation: IsolationLevel,
    tool_states: list[ToolState],
    cwd: Path,
    overlay: dict[str, Any],
) -> int | tuple[Config, IsolationLevel]:
    """A hard network refusal becomes a choice, not a dead end: explain it, then
    (interactively) offer to apply the minimal config fix and continue, simulate
    the machine offline, or stop. Headless prints the exact fix + simulate
    command and exits non-zero, it never relaxes a sandbox setting unattended.
    Returns the new `(cfg, isolation)` when the fix applied and re-validates
    clear, else an exit code."""
    print(f"REFUSING: {refusal}", file=sys.stderr)
    fix = _suggested_network_fix(cfg, isolation, tool_states)
    if fix is None:
        print(
            f"  No sandbox-config change fixes this on the '{isolation}' isolation"
            " (a tool needs isolation only 'strict' provides).",
            file=sys.stderr,
        )
        print(f"  Simulate it offline instead:  agent6 machine test {path}", file=sys.stderr)
        return 2
    if not sys.stdin.isatty():
        print("  To allow it, apply this to the per-repo config and re-run:", file=sys.stderr)
        for key, value in fix.items():
            print(f"    agent6 config set {key} {value} --repo", file=sys.stderr)
        print(f"  Or simulate it offline now:    agent6 machine test {path}", file=sys.stderr)
        return 2
    print("  agent6 can apply the minimal fix now (writes the per-repo config):", file=sys.stderr)
    for key, value in fix.items():
        print(f"    {key} = {value}", file=sys.stderr)
    choice = (_safe_input("  [a]pply & run, [s]imulate offline, or [Q]uit? ") or "").strip().lower()
    if choice == "s":
        return _cmd_machine_test(path, blackboard=None)
    if choice != "a":
        print("Stopped; nothing changed.", file=sys.stderr)
        return 2
    target = repo_config_path_for(cwd)
    target.parent.mkdir(parents=True, exist_ok=True)
    for key, value in fix.items():
        upsert_toml_leaf(target, key, value)
    chown_to_real_user(target.parent)
    chown_to_real_user(target)
    try:
        new_cfg = load_effective_with_overlay(cwd, overlay).config
        new_profile = resolve_isolation(new_cfg.sandbox.isolation, detect_env())
    except (ConfigError, IsolationUnavailableError) as exc:
        print(f"  Applied, but the config no longer validates: {exc}", file=sys.stderr)
        return 2
    if machine_network_refusal(new_cfg, new_profile, tool_states) is not None:
        print("  Applied, but a conflict remains; review the per-repo config.", file=sys.stderr)
        return 2
    print(f"  Applied to {target}. Continuing the run.", file=sys.stderr)
    return new_cfg, new_profile


def _no_instance_hint(machine_id: str, cwd: Path) -> str:
    """A ' Did you mean ...' suffix for a missing-instance error.

    `machine run` takes a FILE (`greet.asm.toml`); status/replay/poke/stop and
    `agent6 attach` take a machine ID (`greet-ok`). Passing the file where an
    id is expected otherwise dead-ends at "no machine instance at
    .../greet.asm.toml". When the argument is an `.asm.toml` file, read its
    `machine` name and suggest that instance id (a file that does not parse
    is still a file); else offer the closest existing instance name."""
    machines = machines_root(resolved_state_dir(cwd))
    existing = sorted(p.name for p in machines.iterdir() if p.is_dir()) if machines.is_dir() else []
    candidate = Path(machine_id)
    if machine_id.endswith(".asm.toml") or candidate.is_file():
        name = ""
        with contextlib.suppress(MachineError, OSError):
            name = load_machine(candidate).machine
        if name in existing:
            return (
                f" Did you mean the instance id {name!r}?"
                " (`machine run` takes the FILE; status/replay/poke/stop and"
                " `agent6 attach` take the ID.)"
            )
        if name:
            return f" That is a machine file; run it first with `agent6 machine run {machine_id}`."
        return (
            f" That is a machine file that does not load; see `agent6 machine check {machine_id}`."
        )
    close = difflib.get_close_matches(candidate.name, existing, n=1)
    return f" Did you mean {close[0]!r}?" if close else ""


def _cmd_machine_run(
    path: Path,
    *,
    config_path: Path | None = None,
    exit_on_wait: bool = False,
    disable_sandbox: bool = False,
    auto_approve: bool = False,
    no_commands: bool = False,
) -> int:
    return run_machine(
        path,
        _machine_frontend(),
        config_path=config_path,
        exit_on_wait=exit_on_wait,
        disable_sandbox=disable_sandbox,
        auto_approve=auto_approve,
        no_commands=no_commands,
    )


def _cmd_machine_replay(machine_id: str) -> int:
    cwd = Path.cwd()
    root = machines_root(resolved_state_dir(cwd)) / machine_id
    if not root.is_dir():
        print(
            f"ERROR: no machine instance at {root}.{_no_instance_hint(machine_id, cwd)}",
            file=sys.stderr,
        )
        return 2
    source_path = root / "machine.asm.toml"
    try:
        spec = load_machine(source_path)
    except MachineError as exc:
        print(f"FAIL: {source_path}", file=sys.stderr)
        for problem in exc.problems:
            print(f"  - {problem}", file=sys.stderr)
        return 1
    journal = MachineJournal(root)
    try:
        result = drive(spec, journal, None, live=False)
    except (JournalError, EngineError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(
        f"{result.status.upper()}: {spec.machine} replayed to {result.state!r}"
        f" after {plural(result.transitions, 'transition')} ({result.reason})"
    )
    return 0 if result.status in ("ok", "incomplete") else 1


def _read_pending_wait_tolerant(journal: MachineJournal) -> tuple[PendingWait | None, str]:
    """(pending wait, note): a corrupt wait.json yields `(None, reason)` so the
    caller keeps its readout going -- mirroring `machine_is_parked` tolerating
    it -- instead of the JournalError aborting the whole command."""
    try:
        return journal.read_pending_wait(), ""
    except JournalError as exc:
        return None, str(exc)


def _cmd_machine_status(machine_id: str) -> int:  # noqa: PLR0912
    cwd = Path.cwd()
    root = machines_root(resolved_state_dir(cwd)) / machine_id
    if not root.is_dir():
        print(
            f"ERROR: no machine instance at {root}.{_no_instance_hint(machine_id, cwd)}",
            file=sys.stderr,
        )
        return 2
    source_path = root / "machine.asm.toml"
    try:
        spec = load_machine(source_path)
    except MachineError as exc:
        print(f"FAIL: {source_path}", file=sys.stderr)
        for problem in exc.problems:
            print(f"  - {problem}", file=sys.stderr)
        return 1
    journal = MachineJournal(root)
    try:
        result = drive(spec, journal, None, live=False)
        events = journal.read()
        snapshot = journal.latest_snapshot()
    except (JournalError, EngineError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    # A corrupt wait.json must not hide the whole readout: the shared dir word
    # (machine_word_for_dir -> machine_is_parked) tolerates it as "parked, keep
    # streaming", so status mirrors that -- drop the wait DETAIL, note it, and
    # still print the state / transitions / spend / steps below.
    pending, pending_note = _read_pending_wait_tolerant(journal)

    alive = worker_is_alive(root)
    spend, inflight_state = machine_spend(events, root, alive=alive)
    # machine_word_for_dir is the ONE owner of running/waiting/stopped, shared
    # with the watch screen, the TUI header, and the web pill: it checks `parked`
    # BEFORE `alive`, so an alive-but-parked instance (a persisted wait written
    # while the worker is still live -- a teardown race) reads "waiting" here too,
    # not a bare "running". A terminal end shows its ok/failed, a crashed instance
    # "stopped" -- never the engine's raw "incomplete".
    word = machine_word_for_dir(fold_machine(spec, events), root)

    print(f"machine: {spec.machine} (v{spec.version})")
    if alive and word == "running":
        running_in = f", running {inflight_state!r}" if inflight_state else ""
        print(f"  status: running (worker pid {read_worker_pid(root)} alive){running_in}")
    else:
        # A live worker blocked on an operator prompt: the word is "waiting"
        # and the line names the state to answer in.
        print(
            f"  status: {word}"
            + (
                f" (an approval open in {blocked_in}: answer it in the TUI machine view"
                " or the web page)"
                if alive and (blocked_in := machine_operator_blocked(root))
                else ""
            )
        )
    print(f"  state: {result.state!r}")
    print(f"  transitions: {result.transitions}")
    print(
        f"  spend: {format_usd(spend.usd, partial=spend.partial)}"
        f" (in={spend.input_tokens} tok, out={spend.output_tokens} tok)"
    )
    state_spec = spec.states.get(result.state)
    # Every wait a poke wakes: a parked one (the persisted record, which a
    # foreground wait writes before it sleeps too) and a live worker in a
    # wait state. The signal is consumed at the next check, or the next run.
    in_wait = alive and state_spec is not None and state_spec.kind == "wait"
    waiting_in = pending.state if pending is not None else (result.state if in_wait else "")
    if waiting_in:
        # A timed wait wakes on its own; the poke is the way to wake it now.
        wake_at = pending.wake_at if pending is not None else ""
        print("  " + wait_line(machine_id, waiting_in, wake_at))
    if pending_note:
        print(f"  pending wait: unreadable ({pending_note})")
    poked, poke_payload = journal.read_pending_poke()
    if poked:
        print("  poke pending: " + ("bare" if poke_payload is None else repr(poke_payload)))
    if snapshot is not None and snapshot.blackboard:
        print("  blackboard:")
        for key, value in snapshot.blackboard.items():
            print(f"    {key} = {value!r}")
    step_events = [e for e in events if isinstance(e, StepEvent)]
    if step_events:
        print("  recent steps:")
        for event in step_events[-5:]:
            print(f"    {format_transition(event.seq, event.state, event.label, event.goto)}")
    return 0


def _cmd_machine_poke(
    machine_id: str, *, data: str | None = None, message: str | None = None
) -> int:
    cwd = Path.cwd()
    root = machines_root(resolved_state_dir(cwd)) / machine_id
    if not root.is_dir():
        print(
            f"ERROR: no machine instance at {root}.{_no_instance_hint(machine_id, cwd)}",
            file=sys.stderr,
        )
        return 2
    # An ended machine consumes no signals: a poke would sit unread, so the
    # "it will wake on its next signal check" reply would be a lie. Refuse.
    if refusal := machine_verb_refusal(root, machine_id, "poke"):
        print(f"ERROR: {refusal}", file=sys.stderr)
        return 1
    journal = MachineJournal(root)
    if message is not None:
        payload: Any = message
    elif data is not None:
        try:
            payload = json.loads(data)
        except json.JSONDecodeError as exc:
            print(f"ERROR: --data is not valid JSON: {exc}", file=sys.stderr)
            return 2
    else:
        payload = None
    try:
        journal.poke(payload)
    except JournalError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    carried = "" if payload is None else " (with payload)"
    print(f"poked {machine_id}: it will wake on its next signal check{carried}")
    return 0


def _cmd_machine_stop(machine_id: str) -> int:
    """Write the durable stop marker for a RUNNING machine.

    The engine parks at its next transition boundary (or wakes out of a sleep)
    without journaling an end, so the instance stays resumable. A machine that
    is not running gets a refusal, not a marker that would ambush the next
    `machine run`."""
    cwd = Path.cwd()
    root = machines_root(resolved_state_dir(cwd)) / machine_id
    if not root.is_dir():
        print(
            f"ERROR: no machine instance at {root}.{_no_instance_hint(machine_id, cwd)}",
            file=sys.stderr,
        )
        return 2
    if refusal := machine_verb_refusal(root, machine_id, "stop"):
        print(f"ERROR: {refusal}", file=sys.stderr)
        return 1
    write_stop_request(root)
    print(f"stop requested: {machine_id} parks at its next transition boundary")
    return 0


def _render_overview(ms: MachineState) -> str:
    """The state list with the current state and the visited ones marked
    (`machine_state_mark`): the at-a-glance overview, rendered from the shared fold."""
    lines = [f"machine: {ms.machine} (v{ms.version})  initial={ms.initial}", "states:"]
    for s in ms.states:
        mark = machine_state_mark(is_current=s.is_current, is_visited=s.is_visited)
        lines.append(f"  {mark} {s.name:<22} [{s.kind}]")
    return "\n".join(lines)


def _watch_liveness_exit(root: Path, machine_id: str, ms: MachineState) -> int | None:
    """Watch's exit code when nothing will ever append to the journal: parked
    (an armed --exit-on-wait wait, no worker) or crashed (stale worker.pid, no
    end, no wait). None while a live worker may still write, a worker blocked
    in a foreground wait included. Routed through machine_word_for_dir, the
    one owner of the running/waiting/stopped distinction, so watch agrees
    with status/TUI/web."""
    word = machine_word_for_dir(ms, root)
    current = next((st.name for st in ms.states if st.is_current), "?")
    if word == "waiting" and not worker_is_alive(root):
        print(
            f"\nWAITING in {current!r} (poke to resume):"
            f" agent6 machine poke {machine_id} [--message TEXT]"
        )
        return 0
    if word == "stopped" and read_worker_pid(root) is not None:
        # The pid guard keeps a valid instance that never started from
        # reading as crashed (same guard as the web machine loop).
        print(f"\nSTOPPED: worker exited in {current!r} without ending", file=sys.stderr)
        return 1
    return None


def _cmd_machine_watch(machine_id: str) -> int:  # noqa: PLR0911, PLR0912, PLR0915
    """Follow a running machine: the state overview, each transition as it lands,
    and the current agent state's live reasoning (its per-state logs.jsonl). Exits
    when the worker is dead (parked or crashed), when the instance ended, or on
    Ctrl-C. Read-only."""
    cwd = Path.cwd()
    root = machines_root(resolved_state_dir(cwd)) / machine_id
    if not root.is_dir():
        print(
            f"ERROR: no machine instance at {root}.{_no_instance_hint(machine_id, cwd)}",
            file=sys.stderr,
        )
        return 2
    source = root / "machine.asm.toml"
    try:
        spec = load_machine(source)
    except MachineError as exc:
        print(f"FAIL: {source}", file=sys.stderr)
        for problem in exc.problems:
            print(f"  - {problem}", file=sys.stderr)
        return 1
    journal = MachineJournal(root)
    try:
        ms = fold_machine(spec, journal.read())
    except JournalError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(_render_overview(ms), flush=True)
    if ms.ended is not None:
        print(f"\n{ms.ended.status.upper()}: ended in {ms.ended.state!r} ({ms.ended.reason})")
        return 0 if ms.ended.status == "ok" else 1
    code = _watch_liveness_exit(root, machine_id, ms)
    if code is not None:
        return code

    print("\n[agent6] watching (Ctrl-C to stop)...", file=sys.stderr)
    print(
        "[agent6] poke a waiting machine from another shell: "
        f"agent6 machine poke {machine_id} [--message TEXT]",
        file=sys.stderr,
    )
    cursor = MachineWatchCursor(seen_steps=len(ms.transitions))
    cursor.seed_notifications(ms)  # history already rendered by the overview
    anchor: float | None = None
    try:
        while True:
            try:
                ms = fold_machine(spec, journal.read())
            except JournalError as exc:
                # The same clean degradation `machine status` gives a corrupt
                # journal; attach must not turn it into a traceback.
                print(f"ERROR: {exc}", file=sys.stderr)
                return 1
            for t in cursor.new_transitions(ms):
                print(
                    f"  {format_transition(t.seq, t.state, t.label, t.goto, t.detail)}", flush=True
                )
            for n in cursor.new_notifications(ms):
                # Ring the bell + fire a desktop notification (if notify-send is
                # present) so an operator watching over ssh is alerted.
                print(f"\a  🔔 [{n.level}] {n.state}: {n.message}", flush=True)
                desktop_notify(f"agent6: {ms.machine}", n.message)
            newest, switched = cursor.advance_log(root)
            if switched:
                # Reset the elapsed-time anchor too: each state log re-derives its
                # own base from its first event, else states 2..N read inflated.
                anchor = None
                if newest is not None:
                    print(f"  -- agent state: {newest.parent.name} --", file=sys.stderr)
            for line in cursor.read_log_lines():
                if anchor is None:
                    with contextlib.suppress(json.JSONDecodeError):
                        anchor = event_epoch(json.loads(line).get("ts"))
                print("    " + format_plain_event(line, session_start_ts=anchor), flush=True)
            if ms.ended is not None:
                print(
                    f"\n{ms.ended.status.upper()}: ended in {ms.ended.state!r} after"
                    f" {ms.ended.transitions} transitions ({ms.ended.reason})"
                )
                return 0 if ms.ended.status == "ok" else 1
            code = _watch_liveness_exit(root, machine_id, ms)
            if code is not None:
                return code
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\n[agent6] watch: stopped.", file=sys.stderr)
        return 0


def _machine_frontend() -> MachineFrontend:
    """The presentation seam `app.machine` run/create drive: stdio output plus
    the interactive network-refusal resolver (needs a TTY, so it stays cli-side;
    `create_machine` uses only the reporter)."""
    return MachineFrontend(reporter=STDIO_REPORTER, resolve_network_fix=_resolve_network_refusal)


def _cmd_machine_create(
    task: str, *, output: Path | None, max_attempts: int, config_path: Path | None
) -> int:
    return create_machine(
        task, _machine_frontend(), output=output, max_attempts=max_attempts, config_path=config_path
    )
