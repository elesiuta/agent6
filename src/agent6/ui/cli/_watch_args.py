# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Parser builders for observing and driving a run: `attach` (raw tail or
full-screen TUI on one run/machine), `steer` (queue an instruction for a
live run), `tui` (the run/plan/ask hub), and `web` (the browser UI)."""

from __future__ import annotations

import argparse

from agent6.ui.cli._common import _sub
from agent6.ui.cli.completers import (
    _complete_session_ids,
    _complete_session_ports,
    _complete_watch_targets,
)


def _add_attach_parser(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    watch_p = _sub(
        sub,
        "attach",
        help=(
            "Attach to a run or machine live and drive it: follow the conversation"
            " (the same render as `agent6 run`) and, on a terminal, answer its"
            " run_command approvals and ask_user questions. --raw is the no-deps"
            " event-line tail, --tui the full-screen TUI, --json a one-shot"
            " snapshot of the folded state. Omit the target for the most recent"
            " run."
        ),
    )
    watch_target = watch_p.add_argument(
        "target",
        nargs="?",
        default="",
        help="Session id (exact or prefix) or machine id. Omit for the most recent.",
    )
    watch_target.completer = _complete_watch_targets  # type: ignore[attr-defined]
    # One presentation at a time: JSON silently won over --raw/--tui when
    # combined, which read as the other flag being broken.
    watch_mode = watch_p.add_mutually_exclusive_group()
    watch_mode.add_argument(
        "--tui",
        action="store_true",
        help="Open the full-screen TUI instead of the default conversation follow.",
    )
    watch_mode.add_argument(
        "--json",
        action="store_true",
        help="Print a one-shot JSON snapshot of the folded state and exit (the web wire form).",
    )
    watch_mode.add_argument(
        "--raw",
        action="store_true",
        help="Follow the no-deps event-line tail (type + key fields) instead of the conversation.",
    )
    watch_p.add_argument(
        "--since",
        type=int,
        default=0,
        metavar="N",
        help="--raw only: replay the last N events before following (0 = from end).",
    )


def _add_tui_parser(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    _sub(
        sub,
        "tui",
        help="Open the TUI hub: browse runs and start a new run/plan/ask.",
    )


def _add_web_parser(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    web_p = _sub(
        sub,
        "web",
        help=(
            "Serve the browser UI (loopback by default): watch and drive runs and"
            " machines from a desktop or phone. Put `tailscale serve` in front for"
            " remote access."
        ),
    )
    web_target = web_p.add_argument(
        "target",
        nargs="?",
        default="",
        help="Session id (exact or prefix) or machine id to open on load. Omit for the hub.",
    )
    web_target.completer = _complete_watch_targets  # type: ignore[attr-defined]
    web_p.add_argument(
        "--host",
        default=None,
        metavar="ADDR",
        help=(
            "Bind address (default: [web].host, 127.0.0.1 unless configured)."
            " A non-loopback bind widens the network surface."
        ),
    )
    web_p.add_argument(
        "--port",
        type=int,
        default=None,
        metavar="N",
        help="Listen port (default: [web].port, 7658 unless configured).",
    )
    web_p.add_argument(
        "--allow-non-loopback",
        action="store_true",
        help="Opt in to bind a non-loopback --host (else a non-loopback bind is refused).",
    )


def _add_steer_parser(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    steer_p = _sub(
        sub,
        "steer",
        help=(
            "Queue a steering instruction for a live run: the same channel the"
            " TUI and web composers use, picked up at the run's next iteration"
            " boundary (pause-menu directives like abort and /undo ride the"
            " same way). Live runs only; for a session that is not running,"
            " `agent6 resume ID --steer TEXT` queues one for its next leg."
        ),
    )
    steer_target = steer_p.add_argument("target", help="Session id (exact or unique prefix).")
    steer_target.completer = _complete_session_ids  # type: ignore[attr-defined]
    steer_p.add_argument("text", help="The instruction; rides verbatim.")
    steer_p.add_argument(
        "--now",
        action="store_true",
        help=(
            "Interrupt the in-flight model call to take the steer immediately"
            " (the default waits for the next step boundary; an approval or"
            " question wait cannot be interrupted either way)."
        ),
    )


def _add_answer_parser(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    answer_p = _sub(
        sub,
        "answer",
        help=(
            "Answer a live run's ask_user question without a terminal: the same"
            " answer file the TUI, web and attach write. With no TEXT it prints"
            " the open question and its options; one TEXT per question, in order."
        ),
    )
    answer_target = answer_p.add_argument("target", help="Session id (exact or unique prefix).")
    answer_target.completer = _complete_session_ids  # type: ignore[attr-defined]
    answer_p.add_argument(
        "answers",
        nargs="*",
        metavar="TEXT",
        help="One answer per question, in the order asked; an option's text, or free text.",
    )


def _add_net_parsers(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """`exec` and `forward`: reach into a live run's session network.

    Top-level verbs, like `attach`, because they are things you do TO a running
    session -- and `exec` is the word every container tool already uses for it.
    """
    exec_p = _sub(
        sub,
        "exec",
        help=(
            "Run a command inside a live session's sandbox: the run's recorded"
            " isolation and network (mounts derive from your current config), so"
            " you see what the agent sees. The command is yours, not"
            " the model's, so it is never approved or recorded as a tool call."
            " `agent6 exec CMD...` runs in the newest session;"
            " `agent6 exec SESSION -- CMD...` names one. Only the first `--`"
            " separates; later ones belong to the command."
        ),
    )
    exec_rest = exec_p.add_argument(
        "rest",
        nargs=argparse.REMAINDER,
        help="[SESSION --] CMD... The command rides verbatim.",
    )
    exec_rest.completer = _complete_session_ids  # type: ignore[attr-defined]

    fwd_p = _sub(
        sub,
        "forward",
        help=(
            "Bridge a port inside a live session's network to one on this"
            " machine, so a browser can open the dev server the agent started."
            " Without a port, lists what the session is listening on."
        ),
    )
    fwd_target = fwd_p.add_argument(
        "target",
        nargs="?",
        default="",
        help=(
            "Session id (exact or prefix); omit for the newest. A bare number"
            " here is read as the PORT of the newest session (name a numeric"
            " session by giving both arguments)."
        ),
    )
    fwd_target.completer = _complete_session_ids  # type: ignore[attr-defined]
    fwd_port = fwd_p.add_argument(
        "port", nargs="?", type=int, help="The port inside the session. Omit to list them."
    )
    fwd_port.completer = _complete_session_ports  # type: ignore[attr-defined]
    fwd_p.add_argument(
        "--local-port",
        type=int,
        default=0,
        help="The port on this machine (default: the same number, as kubectl/docker/ssh mean it).",
    )
