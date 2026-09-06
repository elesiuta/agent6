# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The surface models: `[skills]`, `[machine]`, `[web]`, `[notify]`, and
`[parallel]`."""

from __future__ import annotations

from ipaddress import ip_address
from typing import Literal

from pydantic import BaseModel, Field, model_validator

from agent6.config._base import MODEL_CONFIG, Argv, StrTuple


class SkillsConfig(BaseModel):
    """`[skills]` section: operator-installed SKILL.md packs (agentskills.io).

    Skills live under `<data-dir>/skills/<name>/` (`agent6 skills install`)
    plus any `extra_dirs`. Installed means enabled: the run-mode system
    prompt lists each enabled skill's name + description and the worker loads
    content on demand; the `state` map holds only the exceptions. Skills are
    trusted like config (operator-chosen prompt content); nothing in a skill
    is ever executed by the loader.
    """

    model_config = MODEL_CONFIG

    # Master switch for the whole subsystem. Off = no index block, no
    # use_skill tool, slash commands don't register.
    enabled: bool = Field(
        default=True,
        description=(
            "Master switch for skills: `false` means no skill index in the prompt, no `use_skill` "
            "tool, and no slash commands."
        ),
    )
    # Additional skill directories scanned before the installed dir (a local
    # checkout during skill development wins over an installed copy). Each may
    # hold skill subdirectories or be a single skill dir itself.
    extra_dirs: StrTuple = Field(
        default=(),
        description=(
            "Additional directories scanned for skills, before the installed skills dir; a skill "
            "of the same name in an earlier dir wins."
        ),
    )
    # Per-skill exceptions, one value per skill so contradictory states are
    # unrepresentable: "disabled" drops it from the index; "always" injects
    # the full SKILL.md text into the system prompt instead of indexing it.
    # Absent = "enabled". Layered configs merge this map key-wise, so a repo
    # config can flip one skill without restating the rest.
    state: dict[str, Literal["enabled", "disabled", "always"]] = Field(
        default_factory=dict,
        description=(
            "Per-skill state by name: `enabled` (indexed, loaded on `use_skill`), `disabled` "
            "(dropped), or `always` (its full text sits in the system prompt). Layers merge key by "
            "key; `agent6 skills enable|disable [--repo]` writes it."
        ),
    )


class MachineNotifyConfig(BaseModel):
    """Optional out-of-band notify hook for a running machine.

    When `on_event` is set, `agent6 machine run` runs the argv tuple on each
    `machine.notify` (a state's `notify` message) and on the terminal
    `machine.end`, on the host OUTSIDE the jail (mirror of
    `[notify].on_complete`). The argv is operator-controlled and never
    includes LLM output. Env vars passed:

    - `AGENT6_MACHINE_ID`      , the machine id
    - `AGENT6_MACHINE_DIR`     , absolute path to the instance dir
    - `AGENT6_MACHINE_EVENT`   , `notify` or `end`
    - `AGENT6_MACHINE_STATE`   , the state that emitted it
    - `AGENT6_MACHINE_MESSAGE` , the notify message (or the end reason)
    - `AGENT6_MACHINE_LEVEL`   , `info`/`warn`/`error` for notify, or the
                                   `ok`/`failed` status for end

    Use it to fan out to a phone (ntfy/Pushover/Telegram/email); agent6 owns no
    push infra. A failed hook is logged and does not change the exit code.
    """

    model_config = MODEL_CONFIG

    on_event: Argv = Field(
        default=(),
        description=(
            "A command run on every machine notify event and at the machine's end, as argv (no "
            "shell), with the event in `AGENT6_MACHINE_*` variables. Empty: no hook."
        ),
    )
    timeout_s: float = Field(
        gt=0.0,
        default=30.0,
        description="Seconds the hook may run before it is killed.",
    )


class MachineConfig(BaseModel):
    """State-machine runtime knobs (`agent6 machine run`)."""

    model_config = MODEL_CONFIG

    # How many recent blackboard snapshots to keep per machine instance.
    # Recovery only reads the latest and `machine replay` rebuilds from the
    # journal, so old snapshots are an audit convenience, not state. 0 keeps
    # every snapshot (one file per transition; budget disk accordingly for
    # long-running machines).
    snapshot_keep: int = Field(
        ge=0,
        default=5,
        description=(
            "How many blackboard snapshots a machine instance keeps (recovery reads only the "
            "latest; `machine replay` rebuilds any state from the journal). `0` keeps all."
        ),
    )
    state_log_keep: int = Field(
        ge=0,
        default=50,
        description=(
            "How many per-state log dirs a machine instance keeps under `<instance>/states/` (the "
            "watchable logs of each state's leg; the journal keeps the full transition history "
            "regardless). `0` keeps all."
        ),
    )
    notify: MachineNotifyConfig = Field(default_factory=MachineNotifyConfig)


def is_loopback_host(host: str) -> bool:
    """True iff *host* is a loopback bind (the one source of truth for the web
    UI's secure-by-default gate; a wildcard like 0.0.0.0/:: is NOT loopback)."""
    normalized = host.strip()
    if normalized.startswith("[") and normalized.endswith("]"):
        normalized = normalized[1:-1]
    if normalized.lower() == "localhost":
        return True
    try:
        return ip_address(normalized).is_loopback
    except ValueError:
        return False


class WebConfig(BaseModel):
    """`agent6 web` server bind. Secure by default: loopback only.

    Remote access is expected behind `tailscale serve` (HTTPS + WireGuard) in
    front of the loopback bind; the tailnet identity is the access control, so
    there is no app-level auth. Binding a non-loopback address exposes the write
    surface (spawn runs, answer prompts) to anyone who can reach the port, so it
    is gated behind `allow_non_loopback = true` and carries no default.
    """

    model_config = MODEL_CONFIG

    host: str = Field(
        default="127.0.0.1",
        description=(
            "Address `agent6 web` binds; a non-loopback address also needs `allow_non_loopback = "
            "true`."
        ),
    )
    port: int = Field(
        ge=1,
        le=65535,
        default=7658,
        description="Port `agent6 web` listens on.",
    )
    # Opt-in required to bind a non-loopback host. Off by default so a typo or a
    # copied config can never silently expose the agent to the local network.
    allow_non_loopback: bool = Field(
        default=False,
        description=(
            "Allow `host` to be a non-loopback address, so a typo can never silently expose the "
            "write surface (approvals, steers, config writes) beyond this machine."
        ),
    )

    @model_validator(mode="after")
    def _guard_non_loopback(self) -> WebConfig:
        if not is_loopback_host(self.host) and not self.allow_non_loopback:
            raise ValueError(
                f"[web].host = {self.host!r} is not loopback. Binding a non-loopback"
                " address exposes the web UI's write surface; set [web]"
                " allow_non_loopback = true to opt in (and prefer `tailscale serve`"
                " in front of a 127.0.0.1 bind instead)."
            )
        return self


class NotifyConfig(BaseModel):
    """Optional post-run notification hook.

    When `on_complete` is set, agent6 runs the argv tuple after the
    workflow returns (`agent6 run` or `agent6 resume`). The argv is
    operator-controlled, it never includes LLM output, and runs OUTSIDE the
    jail under a curated env (PATH/HOME/locale + desktop vars, never provider
    keys; see `child_env.curated_env`) with these vars added:

    - `AGENT6_SESSION_ID`      , session id under the per-repo state dir
    - `AGENT6_SESSION_OK`      , `1` if the workflow finished cleanly, `0` otherwise
    - `AGENT6_SESSION_REASON`  , workflow termination reason (e.g. `finish_session`,
                                 `budget_exhausted`, `provider_error`)
    - `AGENT6_SESSION_VERIFIED`, `passed` / `failed` / `unverified` /
                                 `not_applicable` (the verify gate's verdict;
                                 a hook wanting "green" reads this, not `OK`)
    - `AGENT6_SESSION_DIR`     , absolute path to the session dir

    Use cases: desktop notification (`notify-send`), shell-bell, ssh
    push notification, mailx, etc. A failure of the notify command is
    logged but does not change the agent6 exit code.
    """

    model_config = MODEL_CONFIG

    on_complete: Argv = Field(
        default=(),
        description=(
            "A command run when a run or resume ends, as argv (no shell), with "
            "`AGENT6_SESSION_ID/DIR/OK/VERIFIED/REASON` in its environment. Empty: no hook."
        ),
    )
    timeout_s: float = Field(
        gt=0.0,
        default=30.0,
        description="Seconds the hook may run before it is killed.",
    )


class ParallelConfig(BaseModel):
    """`[parallel]` section: fan-out defaults for `agent6 run --parallel`.

    `--parallel N` (or a comma-separated model list) runs N isolated lanes,
    each a disposable clone of the repo, and auto-compares the results. These
    knobs bound and place that fan-out; nothing here mutates the origin repo.
    """

    model_config = MODEL_CONFIG

    # Hard cap on lanes per fan-out. `--parallel` over this refuses up front so a
    # typo (or a long model list) can't spawn an unbounded pile of clones+runs.
    # le: the cap itself must be bounded, or a huge max_lanes re-opens the
    # huge-count allocation parse_spec refuses against. Static, not CPU-derived:
    # lanes are I/O-bound detached runs, and the same repo config must load on
    # every box.
    max_lanes: int = Field(
        ge=1,
        le=1024,
        default=4,
        description=(
            "The most lanes one `--parallel` fan-out may run, `1` to `1024`; a spec asking for "
            "more is refused before anything is cloned."
        ),
    )
    # Base directory for subordinate working trees (a fan-out gets
    # `<workdir>/<repo-id>/<fanout-id>/lane-<i>`; a machine's run states use a
    # `machine-<id>` group the same way; a fork's worktree is
    # `<workdir>/<repo-id>/<fork-id>`). "" resolves to `<cache_dir>/parallel`,
    # a regenerable cache the orchestrator cleans up after importing each lane.
    # Point it at a fast disk for large repos.
    workdir: str = Field(
        default="",
        description=(
            "Base directory for the working trees lanes, machine run states, and forks work "
            "in, in a per-repo subdirectory. Empty: `<cache_dir>/parallel`. A lane's clone is "
            "removed after its work is imported; a fork's worktree by `sessions prune` once "
            "the fork is merged."
        ),
    )
