# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Host-side preflight for `machine run`/`create`.

Before the engine composition drives a machine, these checks refuse a run that
can't be honored -- a tool-network need the isolation cannot enforce
(`machine_network_refusal`) -- and they resolve the machine's own read-only
protect paths (`machine_protect_paths`) and the operator notify hook
(`build_machine_notify_hook`). Pure computations; the hook itself runs through
`app/finalize.run_notify_hook`, the one runner both notify hooks share.
"""

from __future__ import annotations

import sys
from collections.abc import Callable, Mapping
from pathlib import Path

from agent6.app.confine import check_network_support
from agent6.app.finalize import hook_env, run_notify_hook
from agent6.config import Config
from agent6.machine import StateSpec, ToolState
from agent6.types import IsolationLevel


def machine_pass_env_refusal(cfg: Config, states: Mapping[str, StateSpec]) -> str | None:
    """A refusal message naming every tool state that asks for an environment
    variable the operator's `[machine].pass_env` does not allow, else None.
    The state declares, the operator permits: the allowlist lives in the
    global/repo config, never in the machine's own file."""
    allowed = set(cfg.machine.pass_env)
    asks = [
        f"[states.{name}] asks for {', '.join(n for n in state.pass_env if n not in allowed)}"
        for name, state in states.items()
        if isinstance(state, ToolState) and any(n not in allowed for n in state.pass_env)
    ]
    if not asks:
        return None
    return (
        f"{'; '.join(asks)}. [machine].pass_env does not allow these environment"
        " variables; list them there, in the global or repo config (never a machine"
        " overlay), to pass them."
    )


def machine_network_refusal(
    cfg: Config, isolation: IsolationLevel, tool_states: list[ToolState]
) -> str | None:
    """A refusal message if this machine's tool-network needs can't be honored.

    Layers machine-specific rules on top of `check_network_support` (which
    handles network=only_explicit_states / session on `hardened`). On
    `hardened` per-tool isolation is impossible, so we refuse, rather than
    silently mis-confine, whenever isolation is *required*: by the operator
    (`network = "session"`) or by a state (`network = "none"`). A
    networked state under `network` in {"session", "auto"} (both keep the
    tool off the host network) is a config conflict, refused on any isolation.
    Returns None when fine.
    """
    net_err = check_network_support(cfg, isolation)
    if net_err is not None:
        return net_err
    tn = cfg.sandbox.network
    no_tool_net = tn in ("session", "auto")  # both keep the tool off the host network
    has_allow = any(s.network == "host" for s in tool_states)
    has_block = any(s.network == "none" for s in tool_states)
    if has_allow and no_tool_net:
        # Name the ACTUAL value: a hardcoded 'block' misstates an 'auto'
        # config on a refusal surface.
        if isolation == "hardened":
            return (
                'a tool state sets network = "host" but sandbox.network ='
                f" {tn!r}. The hardened isolation cannot single out one tool's"
                " network namespace; let tools share the host network with"
                " sandbox.network = 'host', or run on strict for explicit"
                " per-tool egress."
            )
        return (
            'a tool state sets network = "host" but sandbox.network ='
            f" {tn!r}. Set sandbox.network = 'only_explicit_states' for"
            " explicit per-tool egress."
        )
    if tool_states and tn == "session" and isolation == "hardened":
        return (
            "isolating a machine's tool-state network requires the strict isolation"
            " (a per-tool network namespace); this host supports only 'hardened'."
            " Run on strict, or let tools share the host network with"
            " sandbox.network = 'host'."
        )
    if has_block and isolation == "hardened":
        return (
            'a tool state sets network = "none" (network must be denied),'
            " but the hardened isolation can't isolate one tool's network. Run on"
            ' strict, or use network = "auto" to tolerate the host network.'
        )
    return None


def machine_protect_paths(machine_path: Path, cwd: Path) -> tuple[Path, ...]:
    """The machine's own `.asm.toml` + `scripts/` bundle, to mark read-only
    in run jails. Only paths under the jail-mounted cwd are enforceable (a path
    outside cwd isn't in the child's view, so it can't edit it anyway)."""
    cwd_r = cwd.resolve()
    out: list[Path] = []
    for p in (machine_path, machine_path.parent / "scripts"):
        rp = p.resolve()
        if rp.exists() and rp.is_relative_to(cwd_r):
            out.append(rp)
    return tuple(out)


def _stderr_note(message: str) -> None:
    """A machine's own notices go to stderr: its stdout is the operator's
    run output."""
    print(f"[agent6] {message}", file=sys.stderr)


def build_machine_notify_hook(
    cfg: Config, machine_id: str, root: Path
) -> Callable[[str, str, str, str], None] | None:
    """The operator notify hook fired on `machine.notify`/`machine.end`, or None.

    The argv comes from `[machine.notify].on_event`, the mirror of
    `[notify].on_complete`; see `run_notify_hook` for how it runs.
    """
    notify = cfg.machine.notify
    if not notify.on_event:
        return None

    def fire(kind: str, state: str, message: str, level: str) -> None:
        env = hook_env(
            AGENT6_MACHINE_ID=machine_id,
            AGENT6_MACHINE_DIR=str(root),
            AGENT6_MACHINE_EVENT=kind,
            AGENT6_MACHINE_STATE=state,
            AGENT6_MACHINE_MESSAGE=message,
            AGENT6_MACHINE_LEVEL=level,
        )
        run_notify_hook(
            notify.on_event,
            env,
            timeout_s=notify.timeout_s,
            label="machine.notify hook",
            note=_stderr_note,
        )

    return fire
