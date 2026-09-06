# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The web front-end's server-sent-event streams: a run (an incremental fold
of logs.jsonl, streaming deltas coalesced, a dead worker closing the stream
truthfully) and a machine (a journal poll with an idle heartbeat).

HTTP-free: a handler binds its two socket writes into `SseChannel`, so the
streaming behaviour needs no server to exercise.
"""

from __future__ import annotations

import json
import queue
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agent6.machine import MachineError
from agent6.sessions.ipc import read_worker_pid, worker_is_alive
from agent6.sessions.layout import LOGS_NAME
from agent6.ui.web import model
from agent6.viewmodel import (
    apply_event,
    died_without_end,
    initial_state,
    machine_is_parked,
    machine_snapshot,
    manifest_header,
    session_state_as_dict,
    summarize_session_dir,
    tail_events,
)

# SSE tuning: coalesce high-frequency streaming deltas, heartbeat idle streams so
# a gone client is noticed, and poll a machine's journal at human cadence.
DELTA_COALESCE_S = 0.15
HEARTBEAT_S = 15.0
MACHINE_POLL_S = 0.5
STREAMING_DELTAS = frozenset({"role.text_delta", "role.thinking_delta"})


@dataclass(frozen=True, slots=True)
class SseChannel:
    """The two writes a stream makes, bound to one client's socket; each
    returns False when the client has gone away."""

    send: Callable[[Any], bool]
    ping: Callable[[], bool]


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


def stream_session(chan: SseChannel, session_dir: Path, *, repo: Path) -> None:
    """Stream one run to *chan* until it ends, the worker dies, or the client
    leaves: the tailer thread feeds a queue, the loop folds every queued event
    into one frame, coalesces delta bursts, and heartbeats idle spans."""
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
    header = manifest_header(session_dir, repo=repo)

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
                ev: dict[str, Any] | None = events.get(timeout=HEARTBEAT_S)
            except queue.Empty:
                if not chan.ping():
                    return
                # A run that reached terminal without its own session.end
                # (crash, went quiet, killed in preflight) would otherwise
                # pin this worker forever; ask the codebase's own
                # died_without_end rather than one word of it. `parked` is
                # deliberately excluded: a parked submission the operator
                # resumes starts logging into this same stream.
                word = summarize_session_dir(session_dir).status
                if word != "parked" and died_without_end(word):
                    chan.send(frame(dead=True))
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
                chan.send(frame())
                return
            now = time.monotonic()
            if last_type in STREAMING_DELTAS and (now - last_delta_emit) < DELTA_COALESCE_S:
                continue  # coalesce bursts of text/thinking deltas
            if not chan.send(frame()):
                return
            last_delta_emit = now
    finally:
        # cancel the tailer so it exits on disconnect / dead run, not just session.end
        stop.set()


def stream_machine(chan: SseChannel, machine_dir: Path) -> None:
    """Stream one machine to *chan*: poll the journal fold, push the combined
    snapshot when it changes, heartbeat when it does not, and close truthfully
    on a journaled end or a dead worker."""
    prev = ""
    idle = 0.0
    while True:
        try:
            payload = {
                "machine": machine_snapshot(machine_dir),
                "reasoning": model.machine_reasoning_snapshot(machine_dir),
            }
        except MachineError as exc:
            chan.send({"type": "error", "error": "; ".join(exc.problems)})
            return
        blob = json.dumps(payload, sort_keys=True)
        if blob != prev:
            # The age is derived at SEND time and deliberately outside the
            # comparison above: it changes every poll, so including it would
            # send a frame every poll. The epoch it comes from does not.
            if not chan.send(_with_idle_age(payload)):
                return
            prev = blob
            idle = 0.0
        else:
            idle += MACHINE_POLL_S
            if idle >= HEARTBEAT_S and not chan.ping():
                return
            if idle >= HEARTBEAT_S:
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
            chan.send(_with_idle_age(payload))
            return
        time.sleep(MACHINE_POLL_S)
