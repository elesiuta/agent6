# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Python-side launcher for the `agent6-jail` Rust binary.

Serializes a JailPolicy to JSON on stdin and reads child stdout/stderr/return code
from the launcher's output. If the launcher is not available, falls back to a
plain (un-sandboxed) subprocess invocation only when the policy explicitly
opts in via `cwd-only-mode`, otherwise raises JailUnavailableError. This keeps
"silently weaker" failure modes out of the system.
"""

from __future__ import annotations

import contextlib
import ctypes
import errno
import functools
import json
import os
import select
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import IO, NoReturn

from agent6.paths import hidden_paths, private_dirs
from agent6.types import BackgroundHandoff, CommandResult, JailPolicy

# Loaded at import, never between fork and exec: dlopen allocates, and another
# thread mid-malloc at fork would deadlock a post-fork load.
_LIBC: ctypes.CDLL | None = ctypes.CDLL(None, use_errno=True) if sys.platform == "linux" else None


def die_with_parent(parent_pid: int, sig: int = signal.SIGTERM) -> Callable[[], None]:
    """A `preexec_fn` tying the child's life to *parent_pid* (Linux PDEATHSIG).

    The kernel delivers *sig* to the child when its parent dies -- any death,
    SIGKILL included. The re-check closes the fork window: a parent that died
    before the prctl landed leaves the child re-parented, so the signal would
    never come. Elsewhere (macOS best-effort) this is a no-op; the platform has
    no equivalent tie. Launcher spawns tie with SIGKILL (a dead agent cannot
    tear anything down gracefully); machine children tie with SIGTERM.
    """

    def _setup() -> None:
        if _LIBC is None:
            return
        _LIBC.prctl(1, sig)  # PR_SET_PDEATHSIG = 1
        if os.getppid() != parent_pid:
            os._exit(128 + sig)

    return _setup


# --- operator tool reachability ----------------------------------------------
# The jail's baseline PATH is /usr/bin:/bin and it bind-mounts only the system
# roots below. Operator tools (uv, node, ruff, ...) installed elsewhere are
# otherwise unreachable, so a jailed command dies 127. We add the standard bin
# dirs that exist to PATH, and for those outside the system roots (or whose
# symlinks resolve out to one, a pipx `uv` at /usr/local/bin -> /opt/pipx/...)
# pass the real dirs as tool_paths for a real-location RO+exec mount. Read+exec
# only; the jail still confines writes and network, so containment is
# unchanged. Owned here so run_command and verify (tools.dispatch), machine
# tool states (machine.engine), and the host-side probe (`machine check`)
# resolve tools identically.
_JAIL_BASE_PATH_DIRS = ("/usr/bin", "/bin")
_SYSTEM_ROOTS = (
    Path("/usr"),
    Path("/bin"),
    Path("/sbin"),
    Path("/lib"),
    Path("/lib64"),
    Path("/etc"),
    Path("/dev"),
)


def _under_system_root(p: Path) -> bool:
    return any(p.is_relative_to(r) for r in _SYSTEM_ROOTS)


def _never_mounted(p: Path) -> bool:
    """Dirs that never belong in a jail mount, however a tool symlink
    resolves.

    `operator_tool_paths` mounts `real.parent` for every symlink in a bin
    dir, so one resolving into the config dir mounted `secrets.toml` -- the
    provider API keys -- read-only into the jail, and one into the state dir
    mounted memory and transcripts. Containment cuts both ways: a
    mount CONTAINING a private dir grants the same reads from above, and a
    plain `~/.local/bin/x -> ~/x.sh` makes `real.parent` the whole home
    dir. So agent6's private dirs (:func:`agent6.paths.private_dirs`) are
    refused in either direction, and $HOME and its ancestors outright:
    mounting home or a dir above it would hand the jail `~/.ssh` and every
    credential the operator owns. A mount BELOW home (a tool target's own
    subdir) stays allowed; that is what keeps `~/.local/bin` tools working.
    Denied by identity rather than by inspecting contents.
    """
    if Path.home().is_relative_to(p):
        return True
    return any(p.is_relative_to(d) or d.is_relative_to(p) for d in private_dirs())


# How long the answer wait sleeps between checks. Short enough that an
# operator Stop reads as immediate, long enough to cost nothing while a
# command runs for minutes.
_ANSWER_POLL_S = 0.2

# The launcher's OWN environment. It becomes PID 1 of the jail's PID namespace
# and strict mounts a fresh /proc, so /proc/1/environ is readable by the jailed
# command -- inheriting the agent's env would put the operator's provider key
# there. The launcher reads nothing from the environment (its policy arrives on
# stdin and the child's env is passed explicitly in it), so it gets none.
_LAUNCHER_ENV: dict[str, str] = {}


def operator_tool_paths() -> tuple[str, tuple[Path, ...]]:
    """Return (PATH string, real-location mount dirs) so operator-installed tools
    resolve in the jail. Recomputed per call so a tool the operator (or model)
    just installed is picked up (dirs under a mounted system root only join PATH;
    dirs outside it, and the real dirs symlinks resolve out to, also need the
    RO+exec mount)."""
    path_dirs: list[str] = list(_JAIL_BASE_PATH_DIRS)
    candidates = _tool_bin_dirs()
    mounts: set[Path] = set()
    for d in candidates:
        if not d.is_dir():
            continue
        path_dirs.append(str(d))
        if not _under_system_root(d) and not _never_mounted(d):
            mounts.add(d)  # real binaries in a non-system dir need the dir itself
        try:
            entries = list(d.iterdir())
        except OSError:
            continue
        for entry in entries:
            if not entry.is_symlink():
                continue  # real files are covered by the dir / the /usr mount
            try:
                real = entry.resolve()
            except OSError:
                continue
            if real.is_file() and not _under_system_root(real) and not _never_mounted(real.parent):
                mounts.add(real.parent)  # e.g. /opt/pipx/venvs/uv/bin
    # Interpreter toolchains a repo venv's python may symlink to: uv-managed
    # CPython lives under XDG data, not any bin dir. Without this mount the
    # jail sees such a venv "linked to a non-existent interpreter" and an
    # in-jail `uv run` deletes and recreates the operator's .venv.
    # Mount-only, never a PATH entry.
    data_home = Path(os.environ.get("XDG_DATA_HOME") or Path.home() / ".local/share")
    uv_pythons = data_home / "uv" / "python"
    if uv_pythons.is_dir():
        mounts.add(uv_pythons)
    return ":".join(path_dirs), tuple(sorted(mounts))


def _tool_bin_dirs() -> tuple[Path, ...]:
    """The bin dirs scanned for operator-installed tools (PATH + mounts)."""
    home = Path.home()
    return (
        Path("/usr/local/bin"),
        Path("/usr/local/sbin"),
        home / ".local/bin",
        home / ".cargo/bin",
        Path("/opt/homebrew/bin"),
        Path("/snap/bin"),
    )


@dataclass(frozen=True, slots=True)
class ToolMountNotes:
    """What the operator should know about how their bin dirs resolve into the
    jail, for the once-per-run preflight. Both lists are `"<link> -> <target>"`
    strings; the mount decisions themselves are unchanged and silent."""

    # A symlink whose target's dir is never mounted, so the tool is absent
    # inside the jail (it would die 127 with nothing naming the reason).
    unreachable: tuple[str, ...] = ()
    # A symlink resolving OUT of its bin dir into another dir under $HOME,
    # which is therefore mounted read-only into the jail. Allowed on purpose
    # (it is what keeps ~/.local/bin tools working), but the operator placed
    # one symlink and got a whole directory exposed, so say which.
    exposes_home_dir: tuple[str, ...] = ()


def tool_mount_notes() -> ToolMountNotes:
    """Scan the bin dirs the jail puts on PATH and report both surprises: a
    tool the jail cannot reach, and a tool that drags a home directory into
    the jail with it."""
    home = Path.home()
    bin_dirs = _tool_bin_dirs()
    unreachable: list[str] = []
    exposes: list[str] = []
    for d in bin_dirs:
        try:
            entries = list(d.iterdir()) if d.is_dir() else []
        except OSError:
            continue
        for entry in entries:
            if not entry.is_symlink():
                continue
            try:
                real = entry.resolve()
            except OSError:
                continue
            if not real.is_file() or _under_system_root(real):
                continue
            parent = real.parent
            if _never_mounted(parent):
                unreachable.append(f"{entry} -> {real}")
            elif parent.is_relative_to(home) and parent not in bin_dirs:
                exposes.append(f"{entry} -> {real}")
    return ToolMountNotes(tuple(sorted(unreachable)), tuple(sorted(exposes)))


def jail_search_path() -> str:
    """The PATH a jailed command resolves against, for host-side reachability
    probes (`machine check`): the jail baseline plus the standard bin dirs that
    exist right now. Advisory only; the jail recomputes its own per call."""
    return operator_tool_paths()[0]


class JailUnavailableError(Exception):
    """`agent6-jail` could not be located, refused to set up the namespace, or
    could not guarantee the command left nothing running."""


class JailBinaryError(JailUnavailableError):
    """The launcher binary itself is unusable: missing, or one the kernel
    refuses to execute. No answer about the host's namespaces."""


def _lossy_text(v: object) -> str:
    """Decode child/launcher output for surfaces: one decode policy for this
    module. Command output is not guaranteed UTF-8 (grep over a binary, a
    latin-1 file), so bytes decode with errors="replace" -- a lossy result
    beats a crash or a dropped stream. str passes through; anything else
    (None from a drained pipe) is ""."""
    if isinstance(v, bytes):
        return v.decode(errors="replace")
    return v if isinstance(v, str) else ""


# Operator/test override (docs/config.md); checked before the bundled binary.
_ENV_VAR = "AGENT6_JAIL_BIN"


def locate_jail_binary() -> Path | None:
    """The launcher binary: an explicit override, else the one the build hook
    bundled into the installed package, else one on PATH.

    No source-tree fallback. The build hook compiles the crate into
    `sandbox/_bin/` on every install, editable ones included, so a checkout
    with cargo already has it there: rebuild and reinstall to pick a change up,
    or point `AGENT6_JAIL_BIN` at a `cargo build` output while iterating on
    the crate itself.
    """
    override = os.environ.get(_ENV_VAR)
    if override:
        p = Path(override)
        return p if p.is_file() else None
    # Bundled inside the installed package (the wheel ships the binary
    # under agent6/sandbox/_bin/agent6-jail; see hatch_build.py).
    bundled = Path(__file__).resolve().parent / "_bin" / "agent6-jail"
    if bundled.is_file():
        return bundled
    # Look in PATH
    found = shutil.which("agent6-jail")
    return Path(found) if found else None


# The holder unshares two namespaces and brings up loopback; anything slower
# than this is a launcher that never understood the flag.
_HOLDER_READY_TIMEOUT_S = 10.0


def _read_available(pipe: IO[bytes] | None, budget_s: float = 0.5) -> bytes:
    """Whatever is already in *pipe*, never waiting for EOF.

    Reading a killed launcher's stderr to EOF hangs whenever it left a child
    holding the write end -- which is exactly the wedged case this is for.
    """
    if pipe is None:
        return b""
    chunks: list[bytes] = []
    deadline = time.monotonic() + budget_s
    while time.monotonic() < deadline:
        readable, _, _ = select.select([pipe], [], [], 0.05)
        if not readable:
            break
        chunk = os.read(pipe.fileno(), 4096)
        if not chunk:
            break
        chunks.append(chunk)
    return b"".join(chunks)


@dataclass(frozen=True, slots=True)
class SessionNetwork:
    """The run's session network, held open by two file descriptors.

    One per run, created before anything that might join it. A holder process
    makes the namespaces and reports readiness; we open
    `/proc/<holder>/ns/{user,net}` and let it exit, because an open
    descriptor is what keeps a namespace alive, not a live process. Every
    jailed child whose policy says `network = "session"` is handed these and
    joins them, so the run's commands and its private MCP servers share one
    loopback with no route off the box.

    The user namespace travels with the network one because entering a netns
    needs CAP_SYS_ADMIN in the namespace that owns it (see the launcher's
    `join_network`).
    """

    userns_fd: int
    netns_fd: int
    # The holder stays alive for the run, because /proc/<pid>/ns/* is the only
    # way a SEPARATE process can name these namespaces: `agent6 exec` and
    # `agent6 forward` join through this pid. The descriptors keep the
    # namespaces alive; the pid keeps them nameable.
    holder_pid: int
    _holder: subprocess.Popen[bytes] | None = None

    @classmethod
    def open(cls) -> SessionNetwork:
        binary = _require_jail_binary()
        # Not `_spawn_launcher`: the holder is no command launcher. Nothing
        # sweeps it or takes its group down (`close` ends it through its
        # stdin), so it needs neither the helper's own session nor its
        # registration; it shares the exec-failure translation.
        try:
            proc = subprocess.Popen(
                [str(binary), "--hold-netns"],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=_LAUNCHER_ENV,
                preexec_fn=die_with_parent(os.getpid(), sig=signal.SIGKILL),  # noqa: PLW1509
            )
        except OSError as exc:
            _raise_for_exec_failure(binary, exc)
        fds: list[int] = []
        try:
            assert proc.stdout is not None
            # Bounded: a launcher that does not know --hold-netns reads a policy
            # from stdin instead and would leave the run blocked here forever,
            # before it has done anything a timeout could explain.
            ready, _, _ = select.select([proc.stdout], [], [], _HOLDER_READY_TIMEOUT_S)
            if not ready or proc.stdout.readline().strip() != b"ready":
                proc.kill()  # it is either wedged or not the launcher this code expects
                err = _read_available(proc.stderr)
                raise JailUnavailableError(
                    "the session network could not be created: "
                    + (
                        err.decode(errors="replace")[-400:]
                        if err.strip()
                        else f"the launcher said nothing in {_HOLDER_READY_TIMEOUT_S:.0f}s"
                        " (a stale AGENT6_JAIL_BIN cannot hold one)"
                    )
                )
            # BEFORE the holder exits: /proc/<pid> is gone the moment it does.
            for kind in ("user", "net"):
                fds.append(os.open(f"/proc/{proc.pid}/ns/{kind}", os.O_RDONLY))
        except OSError as exc:
            for fd in fds:  # the first may be open when the second fails
                with contextlib.suppress(OSError):
                    os.close(fd)
            proc.kill()
            raise JailUnavailableError(f"the session network could not be held: {exc}") from exc
        except BaseException:
            proc.kill()
            raise
        # Done with its output: the holder says "ready" once and then only waits
        # on stdin. Holding these would be two descriptors per run that nothing
        # closes until garbage collection -- which a long-lived web or hub
        # process accumulates.
        for pipe in (proc.stdout, proc.stderr):
            if pipe is not None:
                with contextlib.suppress(OSError):
                    pipe.close()
        return cls(userns_fd=fds[0], netns_fd=fds[1], holder_pid=proc.pid, _holder=proc)

    def args(self) -> list[str]:
        return ["--userns-fd", str(self.userns_fd), "--netns-fd", str(self.netns_fd)]

    def fds(self) -> tuple[int, int]:
        return (self.userns_fd, self.netns_fd)

    def close(self) -> None:
        """Drop the run's session network. Nothing can join it afterwards, and
        the kernel reclaims it once the last member exits.

        Closing the holder's stdin is what ends it, so a run that dies without
        reaching here still releases the namespace: the pipe breaks and the
        holder exits on its own."""
        if self._holder is not None:
            if self._holder.stdin is not None:
                with contextlib.suppress(OSError):
                    self._holder.stdin.close()
            try:  # reap it, or the run leaves a zombie holding a /proc entry
                self._holder.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._holder.kill()
                self._holder.wait(timeout=5)
        for fd in self.fds():
            with contextlib.suppress(OSError):
                os.close(fd)


def _join_args(
    policy: JailPolicy, session_net: SessionNetwork | None
) -> tuple[list[str], tuple[int, ...]]:
    """The launcher flags and inherited fds for this policy's network.

    A `session` policy with no network to join is a bug in the caller, not a
    reason to run isolated: the child would look confined and be alone, which
    is a different sandbox than the operator asked for.
    """
    if policy.network != "session":
        return [], ()
    if session_net is None:
        raise JailUnavailableError(
            "network = 'session' needs the run's session network; none was wired"
        )
    return session_net.args(), session_net.fds()


def _policy_to_json(policy: JailPolicy) -> str:
    return json.dumps(
        {
            "isolation": policy.isolation,
            "cwd": str(policy.cwd),
            "argv": list(policy.argv),
            "env": [list(pair) for pair in policy.env],
            "network": policy.network,
            "extra_ro_paths": [str(p) for p in policy.extra_ro_paths],
            "extra_rw_paths": [str(p) for p in policy.extra_rw_paths],
            "extra_device_paths": [str(p) for p in policy.extra_device_paths],
            "extra_protect_paths": [str(p) for p in policy.extra_protect_paths],
            "tool_paths": [str(p) for p in policy.tool_paths],
            # The builtin private set is unioned HERE, the one serialization
            # choke point, so no policy constructor can omit it: secrets and
            # state never enter the jail even under a $HOME-wide grant. A
            # policy grant BENEATH a hidden root is re-bound through the mask
            # by the launcher (the machine data contract).
            "hide_paths": sorted({str(p) for p in hidden_paths(policy.hide_paths)}),
            "timeout_s": policy.timeout_s,
            "memory_limit_mb": policy.memory_limit_mb,
        }
    )


def _run_unsandboxed(policy: JailPolicy) -> CommandResult:
    """Run `policy.argv` as a plain subprocess (no confinement).

    Used for the `none` isolation: the explicit opt-out, the
    dangerously-disable escape hatch, or `auto` on a host with no confinement
    mechanism. Inherits the parent
    environment (so `PATH` etc. resolve normally) overlaid with `policy.env`;
    runs in `policy.cwd`. The sandbox-only knobs (network, ro/rw/protect paths,
    memory_limit_mb) have no effect here, there is no kernel mechanism to
    enforce them.
    """
    env = {**os.environ, **{k: v for k, v in policy.env}}
    start = time.monotonic()
    # Unsandboxed escape hatch; see run_in_jail's docstring. Output is
    # captured as bytes and decoded lossily: a strict text=True decode would
    # raise UnicodeDecodeError out of communicate() on any non-UTF-8 byte,
    # breaking the return-a-result contract.
    try:
        proc = subprocess.run(
            list(policy.argv),
            cwd=str(policy.cwd),
            env=env,
            capture_output=True,
            check=False,
            preexec_fn=die_with_parent(os.getpid(), sig=signal.SIGKILL),
            # <= 0 is "no wall-clock kill" (agent6 exec, a model command whose
            # check-in replaces the kill); None is subprocess's way to say it.
            timeout=policy.timeout_s if policy.timeout_s > 0 else None,
        )
    except subprocess.TimeoutExpired as exc:
        # Match the jailed contract: a timeout is an rc=124 result,
        # not a raised exception the caller would have to special-case.
        return CommandResult(
            argv=tuple(policy.argv),
            returncode=124,
            stdout=_lossy_text(exc.stdout),
            stderr=_lossy_text(exc.stderr),
            duration_s=time.monotonic() - start,
        )
    duration = time.monotonic() - start
    return CommandResult(
        argv=tuple(policy.argv),
        returncode=int(proc.returncode),
        stdout=_lossy_text(proc.stdout),
        stderr=_lossy_text(proc.stderr),
        duration_s=duration,
    )


@functools.lru_cache(maxsize=1)
def strict_namespaces_work() -> bool:
    """Return True iff the jail binary can actually set up a `strict` namespace.

    The cheap `unshare -U -r true` probe in `detect.probe_userns_supported`
    under-reports on an AppArmor-restricted host (Ubuntu 24.04+ with
    `kernel.apparmor_restrict_unprivileged_userns=1`) where an AppArmor profile grants
    the *agent6-jail* binary userns but not `/usr/bin/unshare`. This runs the
    real jail binary with a trivial `strict` policy to get the authoritative
    answer. Cached for the process lifetime; the kernel/isolation state does not
    change mid-run. A binary the kernel cannot execute (JailBinaryError)
    propagates: it says nothing about namespaces, and the callers refuse with
    it rather than read it as "no strict".
    """
    if not Path("/usr/bin/true").exists():
        return False
    probe_cwd = Path(tempfile.gettempdir())
    try:
        res = run_in_jail(
            JailPolicy(
                cwd=probe_cwd,
                argv=("/usr/bin/true",),
                isolation="strict",
                network="none",
                timeout_s=10.0,
            )
        )
    except JailBinaryError:
        raise
    except JailUnavailableError:
        return False
    return res.returncode == 0


def _require_jail_binary() -> Path:
    binary = locate_jail_binary()
    if binary is None:
        raise JailBinaryError(
            "agent6-jail binary not found. Install agent6 from a built wheel"
            " (which bundles the binary), or build from source with"
            " `cargo build --release --locked --manifest-path src/agent6/jail/Cargo.toml`,"
            f" or set {_ENV_VAR}=/path/to/agent6-jail."
        )
    return binary


def _raise_for_exec_failure(binary: Path, exc: OSError) -> NoReturn:
    """Re-raise the launcher's spawn failure as the binary's refusal when the
    kernel refused to execute it (ENOEXEC: a build for another architecture;
    EACCES: no exec bit), naming the file and the remedy. Any other OSError
    (fork or descriptor pressure) says nothing about the binary and passes
    through unchanged."""
    if exc.errno not in (errno.ENOEXEC, errno.EACCES):
        raise exc
    raise JailBinaryError(
        f"agent6-jail at {binary} cannot be executed: {exc.strerror}."
        " Reinstall the bundled binary with `uv sync --reinstall-package agent6`,"
        f" or point {_ENV_VAR} at a build for this host."
    ) from exc


def _spawn_launcher(
    binary: Path,
    args: Sequence[str],
    *,
    stdin: int | IO[bytes] | None,
    stdout: int | IO[bytes] | None,
    stderr: int | IO[bytes] | None,
    pass_fds: Sequence[int] = (),
    die_with_agent: bool = True,
) -> subprocess.Popen[bytes]:
    """Start the launcher in its own session (a hang is one killpg of its
    group away, pidns-init and grandchildren included) and register it live
    for the escapee sweep, under the sweep lock so the sweep never sees it
    half-registered. It gets `_LAUNCHER_ENV` (see its note), and
    *die_with_agent* ties it to this process; a background job outlives the
    turn that started it and stays untied. A binary the kernel refuses to
    execute raises JailBinaryError naming the file and the remedy
    (`_raise_for_exec_failure`)."""
    try:
        with _sweep_lock:
            proc = subprocess.Popen(
                [str(binary), *args],
                stdin=stdin,
                stdout=stdout,
                stderr=stderr,
                pass_fds=pass_fds,
                start_new_session=True,
                preexec_fn=(  # noqa: PLW1509
                    die_with_parent(os.getpid(), sig=signal.SIGKILL) if die_with_agent else None
                ),
                env=_LAUNCHER_ENV,
            )
            _live_launchers.add(proc.pid)
    except OSError as exc:
        _raise_for_exec_failure(binary, exc)
    return proc


# --- escapee reaping ---------------------------------------------------------
# `strict` confines the child in a PID namespace, so nothing outlives it.
# `hardened` has none: a child that calls setsid() leaves the launcher's process
# group, survives the launcher's killpg, and reparents to init. The agent makes
# itself a subreaper so escapees land on it instead, and kills them once the
# command returns.
#
# The launcher cannot do this itself, and must not: its own Landlock ruleset
# denies /proc, and granting it there would hand every jailed child the agent's
# environ.
#
# A process is the command's only if it appeared during the call AND sits
# outside the agent's session. The launcher runs in its own session, so every
# jailed descendant is outside ours (setsid creates sessions, setpgid cannot
# cross one), while a deliberate same-session child -- git, notify-send -- can
# never be swept, whatever thread spawns it.
_PR_SET_CHILD_SUBREAPER = 36
_SWEEP_DEADLINE_S = 5.0
_sweep_lock = threading.Lock()
_live_launchers: set[int] = set()
# Children agent6 started ON PURPOSE in their own session: a `/btw` ask, a
# `/parallel` lane. They look exactly like an escapee -- a child of this process, different
# session -- so without this the first background command's teardown SIGKILLs
# them, destroying model work the operator has already paid for. Every detached
# spawn from this process must register here; `agent6.ui.spawn` is the one
# place that does it.
_own_detached: set[int] = set()


@functools.cache
def _become_subreaper() -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    libc.prctl.restype = ctypes.c_int
    if libc.prctl(_PR_SET_CHILD_SUBREAPER, 1, 0, 0, 0) != 0:
        err = ctypes.get_errno()
        raise JailUnavailableError(
            f"prctl(PR_SET_CHILD_SUBREAPER) failed: {os.strerror(err)}."
            " Without it a sandboxed command could leave a process running after it returns."
        )


def keep_out_of_the_sweep(pid: int) -> None:
    """Mark *pid* as a child agent6 started deliberately, not an escapee.

    Called right after a detached spawn. Never cleared: a pid this process
    started stays ours for its lifetime, and the set is bounded by how many
    sessions one run opens.
    """
    _own_detached.add(pid)


def _own_children() -> dict[int, int]:
    """`{pid: session id}` for this process's children, right now.

    Read as bytes: comm is whatever a process named itself, so the line need
    not be valid UTF-8 and one hostile name must not break the scan.

    Empty where there is no /proc: the sweep is a Linux mechanism, and macOS
    resolves to `isolation = "none"`, which agent6 supports. Letting the error
    out told the model a background command had failed to start AFTER it was
    already running, and left it untracked, so nothing could stop it.
    """
    me = str(os.getpid()).encode()
    found: dict[int, int] = {}
    try:
        entries = list(Path("/proc").iterdir())
    except OSError:
        return found
    for entry in entries:
        if not entry.name.isdigit():
            continue
        try:
            stat = (entry / "stat").read_bytes()
        except OSError:
            continue  # exited mid-scan
        # comm can hold spaces and parens, so fields are taken after its closing
        # one: state, ppid, pgrp, session.
        fields = stat[stat.rfind(b")") + 1 :].split()
        if len(fields) > 3 and fields[1] == me:
            found[int(entry.name)] = int(fields[3])
    return found


def _kill_group_of(pid: int) -> None:
    """SIGKILL *pid*'s process group, or *pid* alone when it does not lead one.

    A pgid is a leader's pid, and it is only reusable once that leader is
    reaped. We hold every one of these as an unreaped child, so a pgid equal to
    the pid we looked up cannot have been recycled underneath us. A pgid that
    is NOT the pid belongs to a leader we do not hold, and under sudo signalling
    a recycled one would kill an unrelated group as root.
    """
    with contextlib.suppress(OSError):
        if os.getpgid(pid) == pid:
            os.killpg(pid, signal.SIGKILL)
        else:
            os.kill(pid, signal.SIGKILL)


def _kill_escapees(exclude: frozenset[int]) -> frozenset[int]:
    """Kill what the command left behind. Returns whatever is still alive."""
    our_session = os.getsid(0)
    deadline = time.monotonic() + _SWEEP_DEADLINE_S
    with _sweep_lock:
        while True:
            children = _own_children()
            # Prune the live-launcher set against reality first. A launcher we
            # started is by definition our child, so a pid that is no longer
            # one cannot be a live launcher -- and leaving it behind means the
            # NEXT process to get that pid is skipped by the sweep. Derived
            # rather than discarded per call site: every caller remembering to
            # clean up is the bug, not the instance.
            _live_launchers.intersection_update(children)
            escapees = {
                pid
                for pid, session in children.items()
                if session != our_session
                and pid not in exclude
                and pid not in _live_launchers
                and pid not in _own_detached
            }
            if not escapees:
                return frozenset()
            for pid in escapees:
                _kill_group_of(pid)
                with contextlib.suppress(OSError):
                    # WNOHANG: a child wedged in uninterruptible sleep must not
                    # hang every later command behind the sweep lock.
                    os.waitpid(pid, os.WNOHANG)
            if time.monotonic() >= deadline:
                return frozenset(escapees)
            time.sleep(0.01)  # killing one layer orphans the next onto this process


class JailedProcess:
    """A jailed child agent6 talks to for a whole session -- an MCP server on a
    JSON-RPC pipe -- not one it collects (`run_in_jail`) or serves many through
    (`JailSession`).

    `close` bounds the child's whole lifetime. `hardened` has no PID namespace,
    so the server sits in the launcher's process group and a setsid child
    reparents onto the agent; signalling the launcher pid alone leaves both
    running. Close signals the group, reaps it, then runs the same escapee sweep
    the one-shot path does, excluding the own-children snapshot taken before the
    spawn.
    """

    def __init__(self, proc: subprocess.Popen[bytes], before: frozenset[int] | None = None) -> None:
        self.popen = proc
        # The child's three streams as plain attributes, like the Popen they
        # come from, so a caller's None-check narrows the later use.
        self.stdin = proc.stdin
        self.stdout = proc.stdout
        self.stderr = proc.stderr
        # `spawn_in_jail` passes the own-children set snapshotted before the
        # launcher; construction time is the fallback for a direct caller.
        self._before = frozenset(_own_children()) if before is None else before

    def close(self) -> None:
        """Best-effort and idempotent. Close stdin for a graceful exit, take the
        launcher's process group down, reap it, then sweep the escapees that
        reparented onto the agent."""
        proc = self.popen
        if proc.stdin is not None:
            with contextlib.suppress(OSError):
                proc.stdin.close()
        if proc.poll() is None:
            # start_new_session made it its own group leader, and Popen holds it
            # unreaped, so its pgid is its pid and cannot have been recycled.
            with contextlib.suppress(OSError):
                os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                with contextlib.suppress(OSError):
                    os.killpg(proc.pid, signal.SIGKILL)
                with contextlib.suppress(subprocess.TimeoutExpired):
                    proc.wait(timeout=1.0)
        with _sweep_lock:
            _live_launchers.discard(proc.pid)
        _kill_escapees(self._before | {proc.pid})


def spawn_in_jail(
    policy: JailPolicy,
    *,
    stdin: int | None = None,
    stdout: int | None = None,
    stderr: int | None = None,
    session_net: SessionNetwork | None = None,
) -> JailedProcess:
    """Start `policy.argv` inside the sandbox and hand back a `JailedProcess`.

    The third transport, beside `run_in_jail` (collect a command) and
    `JailSession` (serve many): for a child agent6 TALKS to for the whole
    session rather than collects -- an MCP server and its JSON-RPC pipe. The
    same policy, the same launcher, the same layers. The handle's `close` bounds
    the child's lifetime: it takes the launcher's whole process group down and
    sweeps the escapees a `hardened` server's setsid child leaves behind, which
    signalling the launcher pid alone would miss.

    The child's stdio is whatever the caller passes, straight through: fork,
    unshare, pivot_root, Landlock, seccomp and execve none of them touch the
    fd table, so a pipe handed in here survives every layer. The policy
    therefore cannot travel on stdin (that belongs to the child), and goes on
    fd 3 instead, which the launcher reads and closes before the child exists.

    `isolation = "none"` spawns the command directly, the same unsandboxed
    path `run_in_jail` documents.
    """
    argv = list(policy.argv)
    # Snapshot before the spawn: any later child of this process that is not in
    # it escaped the server, and `JailedProcess.close` sweeps exactly that set.
    before = frozenset(_own_children())
    if policy.isolation == "none":
        with _sweep_lock:
            proc = subprocess.Popen(
                argv,
                stdin=stdin,
                stdout=stdout,
                stderr=stderr,
                env=dict(policy.env),
                cwd=policy.cwd,
                start_new_session=True,
                preexec_fn=die_with_parent(os.getpid(), sig=signal.SIGKILL),  # noqa: PLW1509
            )
            # Registered like the jailed branch's launcher: the server sits in
            # its own session, so without this a SIBLING handle's close would
            # escapee-sweep it (it is in no later spawn's before-snapshot).
            # `JailedProcess.close` discards it on either branch.
            _live_launchers.add(proc.pid)
        return JailedProcess(proc, before)
    binary = _require_jail_binary()
    spec = json.loads(_policy_to_json(policy))
    spec["mode"] = "exec"
    _become_subreaper()
    # pass_fds keeps the descriptor's NUMBER in the child, so the launcher is
    # told the fd number os.pipe actually returned rather than a hardcoded 3.
    join_args, join_fds = _join_args(policy, session_net)
    policy_r, policy_w = os.pipe()
    try:
        proc = _spawn_launcher(
            binary,
            ["--policy-fd", str(policy_r), *join_args],
            stdin=stdin,
            stdout=stdout,
            stderr=stderr,
            pass_fds=(policy_r, *join_fds),
        )
    except BaseException:
        os.close(policy_w)
        raise
    finally:
        # Ours to close either way: the child has its own copy, and holding the
        # read end here would leave the launcher waiting on an EOF that the
        # write below cannot deliver.
        os.close(policy_r)
    # AFTER the spawn, so the reader exists: a policy larger than the pipe
    # buffer (a long path list) would otherwise block forever on the write.
    with os.fdopen(policy_w, "wb") as handle:
        handle.write((json.dumps(spec) + "\n").encode())
    return JailedProcess(proc, before)


def run_in_jail(policy: JailPolicy, *, session_net: SessionNetwork | None = None) -> CommandResult:
    """Run `policy.argv` inside the sandbox.

    Raises JailUnavailableError if the launcher binary is missing or setup fails.

    The `none` isolation is the unsandboxed path: the command runs as a plain
    subprocess with no kernel confinement. `auto` resolves to it wherever the
    host offers no mechanism at all (non-Linux, or a Linux kernel with neither
    namespaces nor Landlock); an explicit `isolation = "none"`,
    `--dangerously-disable-sandbox`, or `AGENT6_DANGEROUSLY_DISABLE_SANDBOX=1`
    selects it on any host. The CLI prints a prominent warning before any such
    run.

    Security review note: this is the single place where an
    LLM-influenced argv runs without the jail. It exists solely so agent6 is
    usable on platforms (macOS) where the Landlock/seccomp/namespace sandbox
    does not exist. Both real isolation levels still go through the Rust
    launcher; nothing here weakens the Linux boundary.
    """
    if policy.isolation == "none":
        return _run_unsandboxed(policy)
    binary = _require_jail_binary()
    join_args, join_fds = _join_args(policy, session_net)
    spec = _policy_to_json(policy)
    start = time.monotonic()
    _become_subreaper()
    # Snapshot first: anything that is a child of this process afterwards but was not before
    # escaped the command. A concurrent caller's launcher is excluded by pid.
    before = frozenset(_own_children())
    launcher = _spawn_launcher(
        binary,
        join_args,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        pass_fds=join_fds,
    )
    survivors: frozenset[int] = frozenset()
    try:
        result = _launcher_result(launcher, policy, spec, start, binary)
    finally:
        if launcher.poll() is None:
            # Abandoned mid-command (an interrupt raised through communicate()):
            # take the launcher's group down so the jailed tree goes with it.
            # start_new_session made it its own group leader, and Popen holds
            # it unreaped, so its pgid is its pid and cannot have been recycled.
            try:
                os.killpg(launcher.pid, signal.SIGKILL)
            except OSError:
                launcher.kill()
            with contextlib.suppress(subprocess.TimeoutExpired):
                launcher.communicate(timeout=5.0)
        with _sweep_lock:
            _live_launchers.discard(launcher.pid)
        survivors = _kill_escapees(before | {launcher.pid})
    if survivors:
        raise JailUnavailableError(
            f"could not kill everything the command left running (pids {sorted(survivors)});"
            " a process would have outlived this run."
        )
    return result


def _launcher_result(
    launcher: subprocess.Popen[bytes],
    policy: JailPolicy,
    spec: str,
    start: float,
    binary: Path,
) -> CommandResult:
    # The launcher enforces the command's own deadline; this one only bounds
    # the launcher's teardown after it. <= 0 disables both, so waiting five
    # seconds here would kill exactly the long command the caller allowed.
    wait_s = policy.timeout_s + 5.0 if policy.timeout_s > 0 else None
    try:
        raw_out, raw_err = launcher.communicate(input=spec.encode(), timeout=wait_s)
    except subprocess.TimeoutExpired as exc:
        # Kill the whole group, then drain whatever output was produced. Mirror
        # _run_unsandboxed: surface a timeout as the documented rc=124 result, not
        # a raised exception the caller would have to special-case.
        try:
            os.killpg(launcher.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            launcher.kill()
        try:
            raw_out, raw_err = launcher.communicate(timeout=5.0)
        except subprocess.TimeoutExpired:
            raw_out, raw_err = b"", b""
        return CommandResult(
            argv=tuple(policy.argv),
            returncode=124,
            stdout=_lossy_text(raw_out) or _lossy_text(exc.stdout),
            stderr=_lossy_text(raw_err) or _lossy_text(exc.stderr),
            duration_s=time.monotonic() - start,
        )
    proc = subprocess.CompletedProcess(
        args=[str(binary)],
        returncode=launcher.returncode,
        stdout=_lossy_text(raw_out),
        stderr=_lossy_text(raw_err),
    )
    duration = time.monotonic() - start
    # The launcher prints a single JSON line on stdout describing the child's result,
    # then exits 0 itself. Anything else means setup failed, with one exception:
    # a child that could not be EXECUTED at all (bad path, missing interpreter)
    # also surfaces as a launcher error, but the jail itself worked fine. Report
    # that as an ordinary failed command (shell-style 127) so the model fixes
    # its argv instead of concluding the sandbox is broken.
    if proc.returncode != 0:
        stderr = proc.stderr.strip()
        if "child execution failed" in stderr:
            return CommandResult(
                argv=tuple(policy.argv),
                returncode=127,
                stdout="",
                stderr=f"{policy.argv[0]}: command not found or not executable ({stderr})",
                duration_s=duration,
                exec_failed=True,
            )
        raise JailUnavailableError(f"agent6-jail launcher exited {proc.returncode}: {stderr}")
    try:
        result_json = json.loads(proc.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError) as exc:
        raise JailUnavailableError(
            f"agent6-jail produced unparseable output: {proc.stdout!r}"
        ) from exc
    return _with_launcher_warnings(
        _result_from_json(result_json, tuple(policy.argv), duration), proc.stderr
    )


def _with_launcher_warnings(result: CommandResult, launcher_stderr: str) -> CommandResult:
    """Carry the launcher's own diagnostics into the result.

    The child's stderr arrives in the result JSON, so anything on the
    launcher's stderr is agent6 reporting on the jail it just built: a mount it
    could not make, a grant or protect_path it had to skip. Read on SUCCESS
    too: a degraded-but-working jail is the case they exist for, and on
    failure an empty /proc surfaces as "cannot open shared object file" with
    nothing else naming the jail.
    """
    warnings = launcher_stderr.strip()
    if not warnings:
        return result
    return replace(result, stderr=f"{result.stderr}\n{warnings}".strip())


def _result_from_json(
    result_json: dict[str, object], argv: tuple[str, ...], duration: float
) -> CommandResult:
    """The launcher's result object as a CommandResult.

    `exec_failed` is the serving launcher saying the command could not be
    executed; it words that the same way the one-shot path does, so the model
    reads one message however its run is jailed.
    """
    failed_exec = bool(result_json.get("exec_failed", False))
    stderr = str(result_json.get("stderr", ""))
    return CommandResult(
        argv=argv,
        returncode=int(str(result_json["returncode"])),
        stdout=str(result_json.get("stdout", "")),
        stderr=(
            f"{argv[0]}: command not found or not executable ({stderr})" if failed_exec else stderr
        ),
        duration_s=duration,
        exec_failed=failed_exec,
    )


# --- detached commands -------------------------------------------------------
# A background command keeps running after the call that started it, so it is
# the one jailed child the escapee sweep must NOT kill: its launcher stays
# registered live until `stop`. Its own output is not captured here (the caller
# redirects it in argv); only the launcher's result JSON is, so the exit code
# survives the turn that started the command.
_RESULT_NAME = "result.json"
_LAUNCHER_ERR_NAME = "launcher.err"


@dataclass(frozen=True, slots=True)
class BackgroundStatus:
    """What a detached command is doing, right now.

    `running` is the live process, never an inference from output or age. A
    launcher that exited without a result reports `error`: a command whose
    fate is unknown is never reported as still running.
    """

    running: bool
    returncode: int | None
    error: str


def _write_outcome(outcome_dir: Path, returncode: int) -> None:
    """Record a command's exit code where a surface in ANOTHER process reads
    it: this run answers only its own."""
    with contextlib.suppress(OSError):
        (outcome_dir / _RESULT_NAME).write_text(
            json.dumps({"returncode": returncode}), encoding="utf-8"
        )


def _write_stopped(outcome_dir: Path) -> None:
    """Record that the command was STOPPED, for a launcher that was killed
    before it could report an exit code.

    No number is invented: nobody observed one, and a made-up 137 would be a
    surface stating the one thing an operator acts on, wrongly. Written only
    over a result READ as empty -- a command that exited moments before the kill
    keeps the code its launcher wrote, and a read that FAILED says nothing about
    what is on disk, so it is not grounds to overwrite it either.
    """
    result = outcome_dir / _RESULT_NAME
    try:
        existing = result.read_text(errors="replace").strip()
    except FileNotFoundError:
        existing = ""  # the launcher never opened it
    except OSError:
        return
    if existing:
        return
    with contextlib.suppress(OSError):
        result.write_text(json.dumps({"stopped": True}), encoding="utf-8")


def _stop_detached(proc: subprocess.Popen[bytes], descendants: frozenset[int]) -> bool:
    """Kill *proc*'s group and sweep what it left behind; True when it is gone.

    killpg only reaches the process's own group, so a child that called setsid()
    is missed exactly as it is for a foreground command -- and `run_in_jail`'s
    sweep can never catch it either, because by then it is not NEW.
    Unregistering first, then sweeping, makes this the moment its escapees stop
    being spared.
    """
    if proc.poll() is None:
        with contextlib.suppress(OSError):
            os.killpg(proc.pid, signal.SIGKILL)
        with contextlib.suppress(subprocess.TimeoutExpired):
            proc.wait(timeout=5.0)
    with _sweep_lock:
        _live_launchers.discard(proc.pid)
    _kill_escapees(descendants)
    return proc.poll() is not None


class LocalJob:
    """A detached command running with no confinement (`none` isolation).

    There is no launcher to write the exit code down, so this does it: the
    Popen IS the command, and its code is persisted on the first observed exit
    and on stop. Without that, `/shells` from another process read every such
    command as maybe-still-running for the run's life and after it.
    """

    def __init__(self, proc: subprocess.Popen[bytes], outcome_dir: Path) -> None:
        self._proc = proc
        self._outcome_dir = outcome_dir
        # Everything that was already ours when this command started: its own
        # escapees are whatever appears beyond this set.
        self._descendants = frozenset(_own_children())
        self._final: BackgroundStatus | None = None

    @property
    def pid(self) -> int:
        return self._proc.pid

    def status(self) -> BackgroundStatus:
        if self._final is not None:
            return self._final
        if self._proc.poll() is None:
            return BackgroundStatus(running=True, returncode=None, error="")
        return self._settle(int(self._proc.returncode))

    def stop(self) -> str:
        """Kill the command and everything it started. Idempotent.

        Answers "" when the command is gone, else why it might not be: a
        surface that prints "stopped" over a live process is stating the one
        thing an operator acts on, wrongly.
        """
        if not _stop_detached(self._proc, self._descendants):
            return f"the command {self._proc.pid} did not exit after SIGKILL"
        self._settle(int(self._proc.returncode))
        return ""

    def _settle(self, returncode: int) -> BackgroundStatus:
        self._final = BackgroundStatus(running=False, returncode=returncode, error="")
        _write_outcome(self._outcome_dir, returncode)
        return self._final


class BackgroundJob:
    """A jailed command detached from the call that started it: its launcher
    wrote the exit code to *outcome_dir* when the command ended."""

    def __init__(self, proc: subprocess.Popen[bytes], outcome_dir: Path) -> None:
        self._proc = proc
        self._outcome_dir = outcome_dir
        # Everything that was already ours when this command started: its own
        # escapees are whatever appears beyond this set.
        self._descendants = frozenset(_own_children())

    @property
    def pid(self) -> int:
        return self._proc.pid

    def status(self) -> BackgroundStatus:
        if self._proc.poll() is None:
            return BackgroundStatus(running=True, returncode=None, error="")
        self._unregister()
        raw = ""
        with contextlib.suppress(OSError):
            raw = (self._outcome_dir / _RESULT_NAME).read_text(errors="replace")
        with contextlib.suppress(ValueError, IndexError, KeyError):
            return BackgroundStatus(
                running=False,
                returncode=int(json.loads(raw.strip().splitlines()[-1])["returncode"]),
                error="",
            )
        err = ""
        with contextlib.suppress(OSError):
            err = (self._outcome_dir / _LAUNCHER_ERR_NAME).read_text(errors="replace").strip()
        return BackgroundStatus(
            running=False,
            returncode=None,
            error=err or f"the sandbox launcher exited {self._proc.returncode} without a result",
        )

    def stop(self) -> str:
        """Kill the command and everything it started. Idempotent; the same
        answer contract as :meth:`LocalJob.stop`.

        The kill takes the launcher down before it can write the exit code, so
        the ending is recorded here -- otherwise a surface in another process
        goes on reading a stopped command as maybe-still-running.
        """
        if not _stop_detached(self._proc, self._descendants):
            return f"the sandbox launcher {self._proc.pid} did not exit after SIGKILL"
        _write_stopped(self._outcome_dir)
        return ""

    def _unregister(self) -> None:
        with _sweep_lock:
            _live_launchers.discard(self._proc.pid)


class SessionJob:
    """A command left running inside a :class:`JailSession`.

    Its pid is namespace-local, so every question about it goes back through
    the session. The terminal answer is kept: once the launcher has reaped the
    command, asking again gets ECHILD, which is not "exit code unknown".
    """

    def __init__(self, session: JailSession, pid: int, outcome_dir: Path) -> None:
        self._session = session
        self._pid = pid
        self._outcome_dir = outcome_dir
        self._final: BackgroundStatus | None = None

    def status(self) -> BackgroundStatus:
        if self._final is not None:
            return self._final
        try:
            status = self._session.status_background(self._pid)
        except JailUnavailableError as exc:
            self._final = BackgroundStatus(running=False, returncode=None, error=str(exc))
            return self._final
        if not status.running:
            self._settle(status)
        return status

    def stop(self) -> str:
        """Kill the command and its group. Idempotent. Answers "" when the
        launcher confirmed it is gone, else what it said instead."""
        if self._final is not None:
            return ""
        try:
            code = self._session.stop_background(self._pid)
        except JailUnavailableError as exc:
            self._settle(BackgroundStatus(running=False, returncode=None, error=str(exc)))
            return str(exc)
        self._settle(BackgroundStatus(running=False, returncode=code, error=""))
        return ""

    def _settle(self, status: BackgroundStatus) -> None:
        self._final = status
        if status.returncode is not None:
            _write_outcome(self._outcome_dir, status.returncode)


def start_in_jail(policy: JailPolicy, *, outcome_dir: Path) -> BackgroundJob | LocalJob:
    """Spawn `policy.argv` in the sandbox and return WITHOUT waiting for it.

    The caller owns the command's own output: nothing is captured here, so
    `policy.argv` must redirect it somewhere both sides can read. Only the
    launcher's result JSON and stderr land in *outcome_dir*, which is what lets
    the exit code outlive the turn that started the command. The unsandboxed
    branch has no launcher to write that JSON, so its `LocalJob` writes it
    from the exit it observes.

    Security review note: same policy, same launcher, same confinement as
    `run_in_jail` -- the only difference is that this call does not wait. The
    escapee sweep spares it while it lives (it is a deliberate child, not
    something a command left behind) and `stop` takes its whole group down.
    """
    outcome_dir.mkdir(parents=True, exist_ok=True)
    if policy.isolation == "none":
        # Unsandboxed escape hatch; see run_in_jail's note.
        with _sweep_lock:
            proc = subprocess.Popen(
                list(policy.argv),
                cwd=str(policy.cwd),
                env={**os.environ, **dict(policy.env)},
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
            # Registered like the jailed launcher: the sweep spares what agent6
            # deliberately started. Unregistered, stopping ONE background
            # command swept every sibling as an escapee, and the sibling's next
            # status polled a reaped pid -- which reads as returncode 0, so
            # agent6 killed a command and then called it a clean exit.
            _live_launchers.add(proc.pid)
        return LocalJob(proc, outcome_dir)
    binary = _require_jail_binary()
    spec = _policy_to_json(policy)
    _become_subreaper()
    result = (outcome_dir / _RESULT_NAME).open("wb")
    errors = (outcome_dir / _LAUNCHER_ERR_NAME).open("wb")
    try:
        launcher = _spawn_launcher(
            binary, (), stdin=subprocess.PIPE, stdout=result, stderr=errors, die_with_agent=False
        )
    finally:
        result.close()
        errors.close()
    assert launcher.stdin is not None
    with contextlib.suppress(OSError):
        launcher.stdin.write(spec.encode())
    launcher.stdin.close()
    return BackgroundJob(launcher, outcome_dir)


def _abandon_launcher(proc: subprocess.Popen[bytes], interrupt_w: int) -> None:
    """A launcher that failed setup: out of the live set, every pipe closed,
    and the child reaped. Closing stdin here swallows the EPIPE a dead peer
    forces on the buffered spec write; left to garbage collection, that close
    re-raises as unraisable BrokenPipeError noise in the caller's log, and the
    unreaped child sits as a zombie for the rest of the process."""
    with _sweep_lock:
        _live_launchers.discard(proc.pid)
    with contextlib.suppress(OSError):
        os.close(interrupt_w)
    for pipe in (proc.stdin, proc.stdout, proc.stderr):
        if pipe is not None:
            with contextlib.suppress(OSError, ValueError):
                pipe.close()
    if proc.poll() is None:
        with contextlib.suppress(OSError):
            os.killpg(proc.pid, signal.SIGKILL)
    with contextlib.suppress(subprocess.TimeoutExpired, OSError):
        proc.wait(timeout=5.0)


@dataclass(slots=True)
class JailSession:
    """One long-lived launcher, serving every command of one run.

    The launcher establishes its namespaces, rootfs, Landlock and seccomp once
    and then reads one request per line, so the run's commands share a netns, a
    PID namespace and a /tmp: a server one command starts is reachable by the
    next, which per-command launchers cannot offer. Closing the session shuts
    stdin, and the PID namespace takes everything inside it down.

    Every isolation level serves, `none` included: the capture and background
    lifecycle are one implementation rather than one per level. Only `strict`
    has the PID namespace, so elsewhere the launcher sweeps what it backgrounded
    at EOF and this side sweeps each command's escapees (below).

    Not thread-safe: one loop drives it, one command at a time.
    """

    _proc: subprocess.Popen[bytes]
    _binary: Path
    # Whether the launcher's PID namespace bounds what a command leaves running.
    # Without one, a `setsid` escapee reparents to this process (a subreaper),
    # so each command's own sweep has to run here instead.
    _pid_namespaced: bool
    # Write end of the launcher's interrupt pipe: one byte asks it to hand the
    # RUNNING command back now instead of at the check-in. A second channel is
    # what the request pipe cannot be -- that one is in lockstep, and this side
    # is blocked reading the answer to the very request being interrupted.
    _interrupt_w: int
    # The run's cap, carried on every request: the launcher's own default
    # applies to what a request omits, which would ignore the operator's.
    _memory_limit_mb: int
    # Anything the launcher wrote to stderr during setup (a refused /proc mount,
    # a skipped grant): the jail came up degraded but still runs. The caller
    # surfaces it once; "" when setup was clean.
    startup_stderr: str = ""

    @classmethod
    def open(cls, policy: JailPolicy, *, session_net: SessionNetwork | None = None) -> JailSession:
        """Start a serving launcher confined by *policy* (its argv is ignored;
        each command arrives as a request)."""
        binary = _require_jail_binary()
        join_args, join_fds = _join_args(policy, session_net)
        _become_subreaper()
        interrupt_r, interrupt_w = os.pipe()
        try:
            proc = _spawn_launcher(
                binary,
                ["--interrupt-fd", str(interrupt_r), *join_args],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                pass_fds=(interrupt_r, *join_fds),
            )
        except BaseException:
            os.close(interrupt_w)
            raise
        finally:
            os.close(interrupt_r)
        spec = json.loads(_policy_to_json(policy))
        spec["mode"] = "serve"
        assert proc.stdin is not None and proc.stdout is not None
        try:
            proc.stdin.write((json.dumps(spec) + "\n").encode())
            proc.stdin.flush()
            # The launcher prints one ready line once setup is done; consuming
            # it before the first request keeps the request/answer lockstep AND
            # marks the point where any setup warning (a refused /proc mount, a
            # skipped grant) is on stderr. Read it there, once -- a degraded
            # jail that still runs otherwise says so instead of only surfacing
            # as a puzzling command failure later. A launcher that died in
            # setup gives EOF here.
            ready = proc.stdout.readline()
        except OSError as exc:
            # The launcher died before consuming the spec (EPIPE at the
            # write/flush): same failure as the EOF below, one error type.
            err = _lossy_text(_read_available(proc.stderr)).strip()
            _abandon_launcher(proc, interrupt_w)
            raise JailUnavailableError(f"jail session died during setup: {err or exc}") from exc
        if not ready:
            err = _lossy_text(_read_available(proc.stderr)).strip()
            _abandon_launcher(proc, interrupt_w)
            raise JailUnavailableError(f"jail session died during setup: {err or 'no output'}")
        startup_stderr = _lossy_text(_read_available(proc.stderr, budget_s=0.1)).strip()
        return cls(
            _proc=proc,
            _binary=binary,
            _pid_namespaced=policy.isolation == "strict",
            _interrupt_w=interrupt_w,
            _memory_limit_mb=policy.memory_limit_mb,
            startup_stderr=startup_stderr,
        )

    def run(
        self,
        argv: tuple[str, ...],
        *,
        env: tuple[tuple[str, str], ...] = (),
        timeout_s: float = 600.0,
        checkin_s: float = 0.0,
        log_dir: str = "",
        interrupted: Callable[[], bool] | None = None,
    ) -> CommandResult | BackgroundHandoff:
        """Run one command to completion in this session's namespaces.

        Without a PID namespace the command's escapees (a `setsid` daemon, a
        double-fork) survive the launcher's process-group kill and reparent to
        this process, so they are swept here: the sweep is what bounds a
        command's process lifetime.

        *interrupted* is polled while waiting for the answer. Once it says yes,
        the launcher is asked to hand the command back NOW rather than at
        `checkin_s`: the operator pressed Stop, and a 15-minute wait for a
        command that is already going to be abandoned reads as a hung agent.
        The command is not killed -- it becomes `bg<N>` exactly as the
        check-in would have made it, and teardown stops it.
        """
        start = time.monotonic()
        before = frozenset() if self._pid_namespaced else frozenset(_own_children())
        answer: dict[str, object] = {}
        try:
            answer = self._request(
                {
                    "kind": "run",
                    "argv": list(argv),
                    "env": [list(p) for p in env],
                    "timeout_s": timeout_s,
                    "checkin_s": checkin_s,
                    "log_dir": log_dir,
                    "memory_limit_mb": self._memory_limit_mb,
                },
                interrupted=interrupted,
            )
        finally:
            # Only for a command that ENDED: one handed back is still running,
            # and its own children are not escapees yet.
            if not self._pid_namespaced and not answer.get("backgrounded"):
                _kill_escapees(before | {self._proc.pid})
        elapsed = time.monotonic() - start
        if answer.get("backgrounded"):
            pid = answer.get("pid")
            if not isinstance(pid, int):
                raise JailUnavailableError(f"jail session handed back no pid: {answer}")
            return BackgroundHandoff(
                argv=argv,
                pid=pid,
                log=str(answer.get("log", "")),
                stdout=str(answer.get("stdout", "")),
                stderr=str(answer.get("stderr", "")),
                duration_s=elapsed,
            )
        return _result_from_json(answer, argv, elapsed)

    def start_background(
        self, argv: tuple[str, ...], *, env: tuple[tuple[str, str], ...] = ()
    ) -> int:
        """Start a command and leave it running for later commands to reach,
        answering with its pid. That pid is namespace-local: only this session
        can report on it or stop it."""
        answer = self._request(
            {
                "kind": "background",
                "argv": list(argv),
                "env": [list(p) for p in env],
                "memory_limit_mb": self._memory_limit_mb,
            }
        )
        pid = answer.get("pid")
        if not isinstance(pid, int):
            raise JailUnavailableError(f"jail session started no command: {answer}")
        return pid

    def status_background(self, pid: int) -> BackgroundStatus:
        answer = self._request({"kind": "status", "pid": pid})
        code = answer.get("returncode")
        return BackgroundStatus(
            running=bool(answer.get("running")),
            returncode=code if isinstance(code, int) else None,
            error=str(answer.get("error", "")),
        )

    def stop_background(self, pid: int) -> int | None:
        """Kill a backgrounded command and its group, answering the exit code
        when this call is the one that reaped it. Idempotent."""
        answer = self._request({"kind": "stop", "pid": pid})
        if not answer.get("stopped"):
            raise JailUnavailableError(f"jail session could not stop {pid}: {answer}")
        code = answer.get("returncode")
        return code if isinstance(code, int) else None

    def _request(
        self, request: dict[str, object], *, interrupted: Callable[[], bool] | None = None
    ) -> dict[str, object]:
        """One request, one answer line. The channel is in lockstep: every
        request gets exactly one answer, or the next one reads this one's.

        A dead launcher reaches the caller as JailUnavailableError, never as
        the raw pipe error: every handler in the run is written against this
        one, and an OSError escapes all of them -- including out of the
        dispatcher's close(), before teardown has stopped the shells.
        """
        assert self._proc.stdin is not None and self._proc.stdout is not None
        try:
            self._proc.stdin.write((json.dumps(request) + "\n").encode())
            self._proc.stdin.flush()
            line = self._await_answer(interrupted)
        except (OSError, ValueError) as exc:  # ValueError: the pipe is closed
            raise JailUnavailableError(f"jail session is gone: {exc}") from exc
        if not line:
            raise JailUnavailableError("jail session ended before answering")
        try:
            parsed = json.loads(line.decode(errors="replace"))
        except ValueError as exc:
            raise JailUnavailableError(
                f"jail session produced unparseable output: {line!r}"
            ) from exc
        if not isinstance(parsed, dict):
            raise JailUnavailableError(f"jail session answered with {type(parsed).__name__}")
        return parsed  # pyright: ignore[reportUnknownVariableType]

    def _await_answer(self, interrupted: Callable[[], bool] | None) -> bytes:
        """The launcher's answer line, waiting for it in a way an operator Stop
        can cut short.

        Selecting on the raw descriptor is safe because the channel is in
        lockstep: at most one unread answer can be in flight, so a complete
        answer can never be sitting in the reader's buffer while select says
        there is nothing to read. EOF selects ready and reads empty, which the
        caller already reports as a dead session.
        """
        assert self._proc.stdout is not None
        if interrupted is None:
            return self._proc.stdout.readline()
        asked = False
        while not select.select([self._proc.stdout], [], [], _ANSWER_POLL_S)[0]:
            if not asked and interrupted():
                # Once: the launcher drains the pipe, and a second byte would
                # convert whatever command runs next the moment it starts.
                with contextlib.suppress(OSError):
                    os.write(self._interrupt_w, b"\x01")
                asked = True
        return self._proc.stdout.readline()

    def close(self) -> None:
        """Shut the request channel; the launcher exits on the EOF, and under
        strict its PID namespace takes any survivors with it.

        `communicate()` closes stdin itself (signalling the serve loop's EOF)
        and drains stdout/stderr. It is NOT preceded by a manual
        `stdin.close()`: on Python 3.12/3.13 `communicate()` then flushes
        the already-closed pipe and raises `ValueError: flush of closed file`
        (3.14 tolerates it), which would be an unhandled crash in
        `ToolDispatcher.close()` teardown on the project's minimum Python."""
        with contextlib.suppress(OSError):
            os.close(self._interrupt_w)
        with contextlib.suppress(subprocess.TimeoutExpired, ValueError, OSError):
            self._proc.communicate(timeout=10.0)
        if self._proc.poll() is None:
            with contextlib.suppress(OSError):
                os.killpg(self._proc.pid, signal.SIGKILL)
        with _sweep_lock:
            _live_launchers.discard(self._proc.pid)
