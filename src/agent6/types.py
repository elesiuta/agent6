# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Internal value types, frozen dataclasses, constructed by us only.

Compare with the pydantic models at the trust boundaries: `agent6.config.model`
(config), `agent6.tools.schema` (tool inputs), `agent6.machine.model` (machine
files). `agent6.providers.types` is the provider-neutral wire vocabulary,
plain dataclasses like these.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

TernaryMode = Literal["no", "ask", "yes"]
# `none` is the UNSANDBOXED isolation: child commands run as plain subprocesses
# with no kernel-enforced confinement. Reached when the host has no confinement
# mechanism at all (non-Linux, or a Linux kernel offering neither userns nor
# Landlock), or as a deliberate operator opt-out on any host via
# `sandbox.isolation = "none"`, `--dangerously-disable-sandbox`, or
# `AGENT6_DANGEROUSLY_DISABLE_SANDBOX=1` (self-authorizing, with a loud warning;
# see `detect.resolve_isolation`).
IsolationLevel = Literal["strict", "hardened", "none"]
NetworkMode = Literal["host", "session", "none"]
# The model roles a session can be driven by.
RoleName = Literal["worker", "reviewer", "planner"]
# The modes `agent6 resume` accepts. A narrower question than "is this a
# known mode", which is what `session_kind` answers.
ResumableMode = Literal["run", "plan", "ask"]
# What the after-auto-commit hook (`run -i`'s REPL) tells the loop to do next;
# `exit` is /exit: stop AND leave (no follow-up prompt).
AutoCommitDirective = Literal["continue", "stop", "undo", "exit"]


@dataclass(frozen=True, slots=True)
class SessionKind:
    """What a mode MEANS, in one record.

    One owner for "is this session allowed to X", so no surface re-derives it
    from a bare string and disagrees with another. The string stays the key
    and stays what is PERSISTED; this is derived from it at read time, never
    written. A future agent6 that
    changes what "plan" may do must reinterpret old sessions correctly, which
    storing the capabilities would prevent.
    """

    name: str
    role: RoleName
    # May mutate the workspace in-process (apply_edit / apply_patch), and owns
    # a background command's lifetime.
    edits: bool
    # May execute commands at all.
    runs_commands: bool
    # Forces approval even where config says "yes".
    clamps_commands: bool
    # `agent6 resume` can pick this up. A machine's states are driven by the
    # machine agent, not by the run lifecycle.
    resumable: bool


SESSION_KINDS: dict[str, SessionKind] = {
    kind.name: kind
    for kind in (
        SessionKind(
            name="run",
            role="worker",
            edits=True,
            runs_commands=True,
            clamps_commands=False,
            resumable=True,
        ),
        # Operator-present like ask, so it clamps run_commands yes->ask.
        SessionKind(
            name="plan",
            role="planner",
            edits=False,
            runs_commands=True,
            clamps_commands=True,
            resumable=True,
        ),
        # `agent6 ask`, kept out of the run history: investigates and answers
        # with no edit/DAG tools. run_command runs jailed and may write the
        # workspace, so its writes are approval-gated (clamps_commands).
        SessionKind(
            name="ask",
            role="worker",
            edits=False,
            runs_commands=True,
            clamps_commands=True,
            resumable=True,
        ),
        # Authoring a machine file, and one state of one running machine. The
        # deliverable is the finish_session payload; command tools only tempt a
        # weak model into spelunking.
        SessionKind(
            name="machine",
            role="worker",
            edits=False,
            runs_commands=False,
            clamps_commands=False,
            resumable=False,
        ),
        SessionKind(
            name="agent",
            role="worker",
            edits=False,
            runs_commands=False,
            clamps_commands=False,
            resumable=False,
        ),
    )
}


# The modes an operator starts from a hub or the CLI and resumes; machine and
# agent legs are driven by the machine agent.
OPERATOR_MODES: tuple[str, ...] = tuple(k.name for k in SESSION_KINDS.values() if k.resumable)


class UnknownSessionKind(ValueError):
    """A mode string this agent6 does not know."""


def session_kind(name: str) -> SessionKind:
    """The record for *name*, refusing anything this agent6 does not know.

    Refusing rather than defaulting: a damaged manifest must never silently
    escalate a read-only session to the privileged write tools.
    """
    kind = SESSION_KINDS.get(name)
    if kind is None:
        raise UnknownSessionKind(f"unknown session mode {name!r}")
    return kind


def session_bucket(name: str) -> str:
    """The bucket a session of mode *name* gets its own directory in.

    Derived, never stored, so a record cannot disagree with where its sessions
    actually go. The buckets sit under one `sessions/` root, which
    is what leaves the state dir's own `machines/` to live machine INSTANCES.
    An `agent` leg lives inside its machine instance's directory and has no
    bucket.
    """
    kind = session_kind(name)
    if kind.name == "agent":
        raise UnknownSessionKind("an agent leg lives under its machine instance, not sessions/")
    return f"{kind.name}s"


@dataclass(frozen=True, slots=True)
class CommandResult:
    """Result of running a command (in or out of the jail)."""

    argv: tuple[str, ...]
    returncode: int
    stdout: str
    stderr: str
    duration_s: float
    # True when the launcher could not execute the binary at all (bad path, not
    # on the jail PATH, missing interpreter, or a symlink that escapes the
    # sandbox roots). Distinct from "ran and exited non-zero": a
    # model can fix its own argv, but an operator verify/metric command that
    # cannot execute is a config/sandbox problem the run must surface loudly.
    exec_failed: bool = False

    @property
    def ok(self) -> bool:
        return self.returncode == 0


@dataclass(frozen=True, slots=True)
class BackgroundHandoff:
    """A command that outlived its check-in and is still running.

    Its own type rather than a CommandResult with a hole in it: a completed
    command and a running one answer different questions, and a returncode
    invented for the second would be a lie every caller has to remember to
    ignore. The tool result the model sees is still ONE shape (see
    `ExecResult`).
    """

    argv: tuple[str, ...]
    pid: int
    log: str
    # What the command printed before the hand-off, still split by stream.
    stdout: str
    stderr: str
    duration_s: float


@dataclass(frozen=True, slots=True)
class JailPolicy:
    """What the jail is allowed to do for a single child invocation."""

    cwd: Path
    argv: tuple[str, ...]
    isolation: IsolationLevel = "strict"
    env: tuple[tuple[str, str], ...] = ()
    # Which network this child joins: the machine's, the run's own (shared with
    # its siblings, no route off the box), or one of its own with nothing else
    # in it. "session" needs the run's SessionNetwork handed to the transport.
    network: NetworkMode = "none"
    extra_ro_paths: tuple[Path, ...] = ()
    extra_rw_paths: tuple[Path, ...] = ()
    # Paths inside `cwd` that the launcher must make read-only from the
    # child's view. Strict re-binds them RO on top of the workspace mount;
    # hardened switches its Landlock rules from "RW on cwd" to "R on cwd
    # + RW on each top-level entry except these". Used to keep an
    # LLM-driven `run_command` from rewriting `.git` even though it
    # lives inside the project root.
    extra_protect_paths: tuple[Path, ...] = ()
    # Real-location RO+exec bind mounts for operator-installed tools that live
    # outside the system dirs (uv in ~/.local/bin or the /opt target a
    # /usr/local/bin symlink resolves to), so a verify/run command finds them.
    # Distinct from `extra_ro_paths` (remapped under /ro, which breaks symlinks);
    # these keep their real paths. Read+execute only, never writable.
    tool_paths: tuple[Path, ...] = ()
    # Operator additions to the hidden set ([sandbox].hide_paths): masked from
    # the jail even under a broader grant. The launcher masks LAST, after every
    # bind, and agent6's own private dirs are always unioned in at
    # serialization -- no constructor can forget them.
    hide_paths: tuple[Path, ...] = ()
    timeout_s: float = 600.0
    # Per-process memory cap in MiB (RLIMIT_DATA, set by the launcher in the
    # child before exec and inherited by every descendant); 0 disables, which
    # is the default here and in `[sandbox].memory_limit_mb`: capping costs
    # real builds more than it buys, and the kernel already handles a memory
    # bomb.
    memory_limit_mb: int = 0


@dataclass(frozen=True, slots=True)
class CoChangePair:
    """Two files that changed together, and how many commits they co-changed in."""

    file_a: str
    file_b: str
    count: int


@dataclass(frozen=True, slots=True)
class HotSymbol:
    """A symbol whose rename/signature change would ripple across files."""

    name: str
    kind: str
    def_path: str
    def_line: int
    files_referenced: int


@dataclass(frozen=True, slots=True)
class RepoSummary:
    """Compact view of a repository handed to the planner."""

    root: Path
    branch: str
    head_sha: str
    file_count: int
    top_level: tuple[str, ...]
    agents_md: str
    recent_log: str
    # Top co-change pairs mined from `git log --name-only`. Tuple
    # of (file_a, file_b, count) sorted by count desc. Empty when the
    # repo has insufficient history (e.g. fresh --depth=1 clone in the
    # realworld bench) or when no pair co-changed at least 2 commits.
    co_change_pairs: tuple[CoChangePair, ...] = ()
    # Top "hot" symbols mined from the tree-sitter index. Tuple
    # of (name, kind, def_path, def_line, files_referenced) sorted by
    # cross-file reference count desc. Complements co_change_pairs:
    # works on fresh repos (no history needed). Empty when the index
    # is disabled or no symbol crosses the min_files_referenced
    # threshold.
    hot_symbols: tuple[HotSymbol, ...] = ()
    # Compact directory map built from `git ls-files`. Multi-line
    # string of `path/  (N files: a, b, ...)` rows, capped so it stays
    # within a few KB. Empty outside a git repo or when ls-files fails.
    repo_map: str = ""
    # per-file symbol outline mined from the tree-sitter index.
    # Multi-line string of `PATH:` headers followed by `  KIND NAME:LINE`
    # rows, ordered by source position. Capped so the block never exceeds
    # a few KB of system-prompt space; oversized files are truncated with
    # a `... (+N more)` row, and overflow at the file level is summarised
    # as `... (N more files)`. Empty when no parser is available or the
    # index is disabled.
    symbol_outline: str = ""
    # False when root is not a git repository (`agent6 ask` runs anywhere;
    # run/plan require git up front). branch/head_sha/recent_log/repo_map
    # are then empty and the prompt names the situation instead of
    # rendering a fake repo header.
    is_git: bool = True


@dataclass(frozen=True, slots=True)
class SandboxReport:
    """Result of one sandbox self-test."""

    name: str
    ok: bool
    detail: str
