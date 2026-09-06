# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Smoke tests for the Rust jail binary.

These tests are marked `needs_namespaces` and skipped unless unprivileged user
namespaces are available on the host. Building the jail is also opt-in via the
AGENT6_BUILD_JAIL env var so CI can choose when to pay the cost.
"""

from __future__ import annotations

import os
import shutil
import stat
import subprocess
import sys
import time
from pathlib import Path

import pytest

from agent6.sandbox.jail import JailUnavailableError, run_in_jail
from agent6.types import IsolationLevel, JailPolicy
from tests.jail_env import require_userns_jail


def _jail_binary() -> Path | None:
    env = os.environ.get("AGENT6_JAIL_BIN")
    if env and Path(env).is_file():
        return Path(env)
    p = Path(__file__).resolve().parents[2] / "src" / "agent6" / "jail" / "target"
    release = p / "release" / "agent6-jail"
    if release.is_file():
        return release
    debug = p / "debug" / "agent6-jail"
    if debug.is_file():
        return debug
    return None


def _require_fresh(bin_path: Path) -> None:
    """Fail loudly when the built binary predates the Rust sources.

    Every jail security test below runs against this binary, so a stale one
    means the whole Landlock/seccomp/protect-path suite greens against code
    that is no longer in the tree.
    """
    src = Path(__file__).resolve().parents[2] / "src" / "agent6" / "jail"
    newest = max(
        (p.stat().st_mtime for p in [*src.glob("src/*.rs"), src / "Cargo.toml"] if p.is_file()),
        default=0.0,
    )
    if newest > bin_path.stat().st_mtime:
        raise AssertionError(
            f"{bin_path} is older than the jail sources: rebuild with "
            "`cargo build --release --manifest-path src/agent6/jail/Cargo.toml` "
            "(these tests are meaningless against a stale binary)"
        )


pytestmark = pytest.mark.needs_namespaces


@pytest.fixture(scope="module")
def jail_bin() -> Path:
    require_userns_jail()
    bin_path = _jail_binary()
    if bin_path is None:
        if not os.environ.get("AGENT6_BUILD_JAIL"):
            pytest.skip("agent6-jail binary not built; set AGENT6_BUILD_JAIL=1 to build")
        cargo = shutil.which("cargo")
        if cargo is None:
            pytest.skip("cargo not available")
        repo_root = Path(__file__).resolve().parents[2]
        manifest = str(repo_root / "src" / "agent6" / "jail" / "Cargo.toml")
        subprocess.run(
            [cargo, "build", "--release", "--manifest-path", manifest],
            check=True,
        )
        bin_path = _jail_binary()
        assert bin_path is not None
    _require_fresh(bin_path)
    os.environ["AGENT6_JAIL_BIN"] = str(bin_path)
    return bin_path


def test_jail_runs_true(jail_bin: Path, tmp_path: Path) -> None:
    res = run_in_jail(JailPolicy(cwd=tmp_path, argv=("/usr/bin/true",), timeout_s=10.0))
    assert res.returncode == 0


def test_jail_blocks_network_when_disallowed(jail_bin: Path, tmp_path: Path) -> None:
    """A connect() to a REAL host listener is denied without network and
    succeeds with it. The positive control is the point: a DNS probe fails in
    the jail (no /etc/resolv.conf) and on any offline host either way, so it
    passed whether or not confinement existed."""
    import socket
    import threading

    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen(2)
    port = srv.getsockname()[1]
    threading.Thread(target=lambda: [srv.accept() for _ in range(2)], daemon=True).start()
    probe = (
        "import socket,sys;s=socket.socket();"
        f"sys.exit(0 if s.connect_ex(('127.0.0.1',{port}))==0 else 1)"
    )

    def _rc(*, allow: bool) -> int:
        return run_in_jail(
            JailPolicy(
                cwd=tmp_path,
                argv=("/usr/bin/python3", "-c", probe),
                network="host" if allow else "none",
                timeout_s=10.0,
            )
        ).returncode

    try:
        assert _rc(allow=False) != 0, "connect succeeded with network = none"
        assert _rc(allow=True) == 0, "the probe cannot connect even when allowed"
    finally:
        srv.close()


def test_jail_denies_write_outside_the_workspace(jail_bin: Path, tmp_path: Path) -> None:
    """Writes outside the workspace are DENIED (nonzero rc), not merely
    redirected. /tmp alone proves nothing: the in-jail /tmp is a fresh tmpfs,
    so the write there SUCCEEDS and only fails to reach the host -- an
    assertion about remapping, not confinement. /dev/shm became a second such
    tmpfs (a headless browser needs one), so the denial target here is $HOME,
    which is neither remapped nor granted."""
    for target in (str(Path.home() / "agent6-jail-escape"),):
        try:
            res = run_in_jail(
                JailPolicy(
                    cwd=tmp_path,
                    argv=("/bin/sh", "-c", f"echo escape > {target}"),
                    timeout_s=10.0,
                )
            )
        except JailUnavailableError:
            pytest.skip("jail unavailable")
        assert res.returncode != 0, f"jailed child wrote {target}"
        assert not Path(target).exists()


def test_jail_tmp_is_a_private_tmpfs(jail_bin: Path, tmp_path: Path) -> None:
    # /tmp is remapped, so a write there lands in the jail's own tmpfs and
    # never on the host (distinct from the denial the sibling test pins).
    marker = Path("/tmp/agent6-jail-host-escape-marker")
    if marker.exists():
        marker.unlink()
    try:
        res = run_in_jail(
            JailPolicy(
                cwd=tmp_path,
                argv=("/bin/sh", "-c", f"echo escape > {marker}"),
                timeout_s=10.0,
            )
        )
    except JailUnavailableError:
        pytest.skip("jail unavailable")
    assert res.returncode == 0  # it writes -- into the private tmpfs
    assert not marker.exists()  # ...and the host copy never appears


def test_jail_hardened_truncate_denied_outside_grants(jail_bin: Path, tmp_path: Path) -> None:
    """TRUNCATE is an ABI-v3 Landlock right. A ruleset that handles only through
    ABI v2 does not restrict truncate(2) at all, so a hardened jailed child could
    zero any file the operator can write -- the run state dir (transcripts,
    manifests, the memory store), ~/.ssh -- with no write grant. Truncate outside
    every grant must be refused; truncate inside the workspace must still work
    (every '>' redirect onto an existing file relies on it)."""
    shm = Path("/dev/shm")
    if not (shm.is_dir() and os.access(shm, os.W_OK)):
        pytest.skip("/dev/shm not usable as an out-of-grant target")
    victim = shm / "agent6-jail-truncate-victim"
    victim.write_text("SECRET-DO-NOT-ZERO\n", encoding="utf-8")
    ws = tmp_path / "ws"
    ws.mkdir()
    inside = ws / "mine.txt"
    inside.write_text("rewrite me\n", encoding="utf-8")
    # os.truncate is the path-based truncate(2) syscall: it does NOT open the
    # file, so it is gated only by LANDLOCK_ACCESS_FS_TRUNCATE (not WRITE_FILE).
    # coreutils `truncate` opens O_WRONLY first and would be stopped by the write
    # rule, masking whether truncate itself is handled.
    try:
        run_in_jail(
            JailPolicy(
                cwd=ws,
                argv=("/usr/bin/python3", "-c", f"import os; os.truncate({str(inside)!r}, 0)"),
                isolation="hardened",
                timeout_s=10.0,
            )
        )
        run_in_jail(
            JailPolicy(
                cwd=ws,
                argv=("/usr/bin/python3", "-c", f"import os; os.truncate({str(victim)!r}, 0)"),
                isolation="hardened",
                timeout_s=10.0,
            )
        )
    except JailUnavailableError:
        pytest.skip("jail unavailable")
    # Inside the workspace: truncate must still succeed.
    assert inside.read_text(encoding="utf-8") == "", (
        "hardened jail wrongly denied truncate inside cwd"
    )
    # Outside every grant: truncate must be refused, the file left intact.
    assert victim.read_text(encoding="utf-8") == "SECRET-DO-NOT-ZERO\n", (
        "hardened jail truncated a file outside its grants (TRUNCATE unhandled)"
    )
    victim.unlink(missing_ok=True)


def test_jail_dev_null_is_writable(jail_bin: Path, tmp_path: Path) -> None:
    """Writes to /dev/null and friends must succeed under both isolation levels.

    Regression test for the click-short-help bench task INTERNALERROR:
    pytest's logging plugin opens /dev/null O_WRONLY|O_APPEND when a
    `log_file` is configured (click's conftest does this), and the previous
    Landlock rules granted only read+execute on /dev — surfacing as
    PermissionError before any test could run.
    """
    for isolation in ("strict", "hardened"):
        res = run_in_jail(
            JailPolicy(
                cwd=tmp_path,
                argv=("/bin/sh", "-c", "echo x > /dev/null && echo OK"),
                isolation=isolation,
                timeout_s=10.0,
            )
        )
        assert res.returncode == 0, f"{isolation} stderr: {res.stderr!r}"
        assert "OK" in res.stdout, f"{isolation} stdout: {res.stdout!r}"


def test_jail_memory_limit_caps_child_allocation(jail_bin: Path, tmp_path: Path) -> None:
    """memory_limit_mb turns a runaway allocation into a plain failed command.

    A child allocating 200 MiB under a 64 MiB cap must die with MemoryError
    (RLIMIT_DATA, applied in run_child and shared by both isolation levels) while the
    host never approaches the OOM killer; the same allocation with the 0
    opt-out succeeds.
    """
    alloc = (
        "import sys\n"
        "try:\n"
        "    bytearray(200 * 1024 * 1024)\n"
        "except MemoryError:\n"
        "    sys.exit(9)\n"
        "print('ALLOC-OK')\n"
    )
    capped = run_in_jail(
        JailPolicy(
            cwd=tmp_path,
            argv=("/usr/bin/python3", "-c", alloc),
            memory_limit_mb=64,
            timeout_s=30.0,
        )
    )
    assert capped.returncode == 9, f"stderr: {capped.stderr!r}"
    uncapped = run_in_jail(
        JailPolicy(
            cwd=tmp_path,
            argv=("/usr/bin/python3", "-c", alloc),
            memory_limit_mb=0,
            timeout_s=30.0,
        )
    )
    assert uncapped.returncode == 0, f"stderr: {uncapped.stderr!r}"
    assert "ALLOC-OK" in uncapped.stdout


def test_jail_protect_paths_block_writes_to_subdir(jail_bin: Path, tmp_path: Path) -> None:
    """extra_protect_paths must make a sub-directory of cwd read-only."""
    git_dir = tmp_path / ".git"
    git_dir.mkdir()
    (git_dir / "HEAD").write_text("ref: refs/heads/main\n", encoding="utf-8")
    # First confirm without protection the write succeeds inside the jail.
    res_unprotected = run_in_jail(
        JailPolicy(
            cwd=tmp_path,
            argv=("/bin/sh", "-c", "echo pwned > .git/HEAD && cat .git/HEAD"),
            timeout_s=10.0,
        )
    )
    assert res_unprotected.returncode == 0
    assert "pwned" in res_unprotected.stdout
    # Reset and protect.
    (git_dir / "HEAD").write_text("ref: refs/heads/main\n", encoding="utf-8")
    res_protected = run_in_jail(
        JailPolicy(
            cwd=tmp_path,
            argv=("/bin/sh", "-c", "echo pwned > .git/HEAD; cat .git/HEAD"),
            extra_protect_paths=(git_dir,),
            timeout_s=10.0,
        )
    )
    # The shell write fails (EROFS) but `cat` still runs; HEAD is unchanged.
    assert "pwned" not in res_protected.stdout
    assert (git_dir / "HEAD").read_text(encoding="utf-8") == "ref: refs/heads/main\n"


def test_jail_protect_paths_block_writes_to_file(jail_bin: Path, tmp_path: Path) -> None:
    """extra_protect_paths must also protect individual files (not just dirs)."""
    cfg = tmp_path / "protected.txt"
    cfg.write_text("original\n", encoding="utf-8")
    res = run_in_jail(
        JailPolicy(
            cwd=tmp_path,
            argv=("/bin/sh", "-c", "echo pwned > protected.txt; cat protected.txt"),
            extra_protect_paths=(cfg,),
            timeout_s=10.0,
        )
    )
    assert "pwned" not in res.stdout
    assert cfg.read_text(encoding="utf-8") == "original\n"


def test_jail_hardened_symlink_escaping_cwd_gets_no_rw(jail_bin: Path, tmp_path: Path) -> None:
    """A top-level symlink whose target escapes cwd must not receive RW.

    Under hardened the per-top-level-entry RW carve-out used PathFd::new (which
    follows symlinks), so a symlink like ``./escape -> /outside`` got a
    recursive RW Landlock rule on the *outside* inode, letting the child write
    beyond the workspace. The target is placed under ``/dev/shm`` -- outside cwd
    and NOT under ``/tmp`` (which the jail grants RW), so /dev (read+exec only)
    is the governing rule unless the symlink wrongly widens it.
    """
    import shutil as _shutil
    import uuid as _uuid

    shm = Path("/dev/shm")
    if not shm.is_dir() or not os.access(shm, os.W_OK):
        pytest.skip("/dev/shm not usable for the out-of-cwd target")
    outside = shm / f"agent6-jail-escape-{_uuid.uuid4().hex}"
    outside.mkdir()
    try:
        (tmp_path / ".git").mkdir()  # a protect path so the carve-out loop runs
        (tmp_path / "src").mkdir()
        (tmp_path / "escape").symlink_to(outside, target_is_directory=True)
        res = run_in_jail(
            JailPolicy(
                cwd=tmp_path,
                argv=("/bin/sh", "-c", "echo ok > src/x.txt; echo pwned > escape/sentinel; true"),
                isolation="hardened",
                extra_protect_paths=(tmp_path / ".git",),
                timeout_s=10.0,
            )
        )
        # In-cwd sibling write still works; the escaping write is denied.
        assert (tmp_path / "src" / "x.txt").read_text(encoding="utf-8").strip() == "ok"
        escaped = (outside / "sentinel").exists()
        assert not escaped, f"escaped write succeeded; stderr={res.stderr!r}"
    finally:
        _shutil.rmtree(outside, ignore_errors=True)


def test_jail_hardened_symlinked_rw_path_cannot_shadow_a_protect_path(
    jail_bin: Path, tmp_path: Path
) -> None:
    """A symlinked extra_rw_path resolving to a protect-path ancestor must get no
    RW. The rw-shadow guard compares the CANONICAL rw_path against the (canonical)
    protect set, so a blanket grant can't slip past and shadow the carve-out."""
    secret = tmp_path / "secret"
    secret.mkdir()
    protected = secret / "key.txt"
    protected.write_text("SECRET\n", encoding="utf-8")
    (tmp_path / "rwlink").symlink_to(secret, target_is_directory=True)
    res = run_in_jail(
        JailPolicy(
            cwd=tmp_path,
            argv=("/bin/sh", "-c", "echo pwned > secret/key.txt; true"),
            isolation="hardened",
            extra_protect_paths=(protected,),
            extra_rw_paths=(tmp_path / "rwlink",),
            timeout_s=10.0,
        )
    )
    assert protected.read_text(encoding="utf-8").strip() == "SECRET", (
        f"a symlinked rw_path shadowed the protect path; stderr={res.stderr!r}"
    )


def test_jail_hardened_protect_paths_block_writes(jail_bin: Path, tmp_path: Path) -> None:
    """Hardened isolation blocks writes to protect_paths via Landlock carve-out.

    Hardened has no mount namespace so it cannot bind-remount RO; instead the
    launcher switches its Landlock rules from `RW on cwd` to `R on cwd + RW
    on every top-level entry except the protect set`. End result for paths
    that exist at jail-launch time is the same: writes are denied.
    """
    git_dir = tmp_path / ".git"
    git_dir.mkdir()
    (git_dir / "HEAD").write_text("ref: refs/heads/main\n", encoding="utf-8")
    # Make a sibling that the worker IS allowed to write to, to prove we
    # didn't accidentally lock down the whole cwd.
    (tmp_path / "src").mkdir()
    res = run_in_jail(
        JailPolicy(
            cwd=tmp_path,
            argv=(
                "/bin/sh",
                "-c",
                "echo ok > src/x.txt && echo pwned > .git/HEAD; cat src/x.txt; cat .git/HEAD",
            ),
            isolation="hardened",
            extra_protect_paths=(git_dir,),
            timeout_s=10.0,
        )
    )
    assert "ok" in res.stdout  # sibling write succeeded
    assert "pwned" not in res.stdout  # protected write rejected
    assert (git_dir / "HEAD").read_text(encoding="utf-8") == "ref: refs/heads/main\n"


def test_jail_tool_paths_make_a_nonworkspace_binary_reachable(
    jail_bin: Path, tmp_path: Path
) -> None:
    # A tool dir OUTSIDE the workspace (like ~/.local/bin or a pipx /opt target) is
    # unreachable by default; passing it as tool_paths bind-mounts it RO+exec at its
    # real path so PATH resolves it. This is what makes an operator's uv reachable.
    work = tmp_path / "work"
    work.mkdir()
    tools = tmp_path / "tools"  # sibling of cwd -> not under the workspace mount
    tools.mkdir()
    script = tools / "mytool"
    script.write_text("#!/bin/sh\necho tool-ran\n")
    script.chmod(0o755)
    env = (("PATH", f"/usr/bin:/bin:{tools}"),)

    with_mount = run_in_jail(
        JailPolicy(
            cwd=work,
            argv=("/bin/sh", "-c", "mytool"),
            env=env,
            tool_paths=(tools,),
            timeout_s=10.0,
        )
    )
    assert with_mount.returncode == 0, with_mount.stderr
    assert "tool-ran" in with_mount.stdout

    # Same PATH but no tool_paths: the dir is not mounted, so exec fails (guards
    # against the mount silently becoming a no-op).
    without_mount = run_in_jail(
        JailPolicy(cwd=work, argv=("/bin/sh", "-c", "mytool"), env=env, timeout_s=10.0)
    )
    assert without_mount.returncode != 0


def test_jail_extra_ro_paths_mount_at_their_real_location(jail_bin: Path, tmp_path: Path) -> None:
    # The documented contract: a granted toolchain (a conda env, a shared data
    # dir) is usable via its own absolute paths and shebangs. The grant used to
    # remap under an undocumented /ro<src>, where nothing could find it.
    work = tmp_path / "work"
    work.mkdir()
    toolchain = tmp_path / "toolchain"  # outside the workspace mount
    toolchain.mkdir()
    script = toolchain / "hello.sh"
    script.write_text("#!/bin/sh\necho reached-real-path\n")
    script.chmod(0o755)

    granted = run_in_jail(
        JailPolicy(cwd=work, argv=(str(script),), extra_ro_paths=(toolchain,), timeout_s=10.0)
    )
    assert granted.returncode == 0, granted.stderr
    assert "reached-real-path" in granted.stdout

    # Read-only: a write inside the grant is refused.
    ro = run_in_jail(
        JailPolicy(
            cwd=work,
            argv=("/bin/sh", "-c", f"echo x > {toolchain}/marker"),
            extra_ro_paths=(toolchain,),
            timeout_s=10.0,
        )
    )
    assert ro.returncode != 0
    assert not (toolchain / "marker").exists()

    # Without the grant the path does not exist inside the jail at all.
    ungranted = run_in_jail(JailPolicy(cwd=work, argv=(str(script),), timeout_s=10.0))
    assert ungranted.returncode != 0


def test_jail_preserves_non_utf8_output(jail_bin: Path, tmp_path: Path) -> None:
    """A command emitting non-UTF-8 bytes must return a lossy-decoded result,
    not a silently empty stdout. read_to_string dropped the whole stream to ""
    on the first invalid byte (grep over a binary, cat of a latin-1 file)."""
    for isolation in ("strict", "hardened"):
        res = run_in_jail(
            JailPolicy(
                cwd=tmp_path,
                argv=("/bin/sh", "-c", "printf 'caf'; printf '\\351'; printf 'x'"),
                isolation=isolation,
                timeout_s=10.0,
            )
        )
        assert res.returncode == 0, f"{isolation} stderr: {res.stderr!r}"
        # 0xe9 decodes to the replacement char; the surrounding bytes survive.
        assert res.stdout.startswith("caf"), f"{isolation} stdout: {res.stdout!r}"
        assert res.stdout.endswith("x"), f"{isolation} stdout: {res.stdout!r}"
        assert "�" in res.stdout, f"{isolation} stdout: {res.stdout!r}"


def test_jail_backgrounded_pipe_holder_does_not_hang(jail_bin: Path, tmp_path: Path) -> None:
    """A command that backgrounds a process inheriting stdout, then exits 0,
    must return promptly with rc=0 -- not block on the reader join until the
    (30s-sleeping) grandchild dies and then report a false rc=124 timeout.
    The process-group teardown runs on the normal-exit path, not only on
    timeout. Hardened has no PID namespace, so it is the exposed isolation."""
    import time

    start = time.monotonic()
    res = run_in_jail(
        JailPolicy(
            cwd=tmp_path,
            argv=("/bin/sh", "-c", "sleep 30 & echo done; exit 0"),
            isolation="hardened",
            timeout_s=10.0,
        )
    )
    elapsed = time.monotonic() - start
    assert res.returncode == 0, f"stderr: {res.stderr!r}"
    assert "done" in res.stdout
    assert elapsed < 8.0, f"launcher blocked on the backgrounded fd-holder ({elapsed:.1f}s)"


def test_jail_strict_seccomp_blocks_modern_mount_api(jail_bin: Path, tmp_path: Path) -> None:
    """A strict jailed child is userns-root over its own mount ns; without the
    modern mount API in the seccomp deny-list it could mount_setattr(2) away the
    RO flag on the .git protect bind and defeat protect_git. The syscall must
    return EPERM. Uses ctypes so no extra tooling is needed."""
    git_dir = tmp_path / ".git"
    git_dir.mkdir()
    (git_dir / "HEAD").write_text("ref: refs/heads/main\n", encoding="utf-8")
    prog = (
        "import ctypes, ctypes.util, os\n"
        "libc = ctypes.CDLL(ctypes.util.find_library('c'), use_errno=True)\n"
        # mount_setattr(dirfd=AT_FDCWD, path, flags=0, attr=NULL, size=0):
        # NULL attr makes it a pure permission probe -- EPERM (seccomp) vs
        # EFAULT/EINVAL (syscall reached the kernel) is what we assert on.
        f"r = libc.syscall(442, -100, {str(git_dir).encode()!r}, 0, 0, 0)\n"
        "e = ctypes.get_errno()\n"
        "print('EPERM' if (r == -1 and e == 1) else f'REACHED:{e}')\n"
    )
    res = run_in_jail(
        JailPolicy(
            cwd=tmp_path,
            argv=("/usr/bin/python3", "-c", prog),
            isolation="strict",
            extra_protect_paths=(git_dir,),
            timeout_s=10.0,
        )
    )
    assert res.returncode == 0, f"stderr: {res.stderr!r}"
    assert "EPERM" in res.stdout, f"mount_setattr not blocked: {res.stdout!r}"


def test_jail_extra_rw_paths_mount_at_their_real_location(jail_bin: Path, tmp_path: Path) -> None:
    # extra_rw (the machine data dir) is writable AT the host abspath, so
    # $AGENT6_MACHINE_DATA_DIR is the same string in every isolation.
    work = tmp_path / "work"
    work.mkdir()
    data = tmp_path / "data"  # outside the workspace mount
    data.mkdir()
    res = run_in_jail(
        JailPolicy(
            cwd=work,
            argv=("/bin/sh", "-c", f"echo persisted > {data}/out"),
            extra_rw_paths=(data,),
            timeout_s=10.0,
        )
    )
    assert res.returncode == 0, res.stderr
    assert (data / "out").read_text().strip() == "persisted"


def test_jail_hardened_protect_paths_nested_below_a_top_level_entry(
    jail_bin: Path, tmp_path: Path
) -> None:
    """A protect path does not have to sit at the root of cwd: `machine run
    ops/deploy.asm.toml` protects ops/deploy.asm.toml and ops/scripts, both
    NESTED under the top-level entry ops/. Comparing entries to the protect set
    by equality let ops/ take a recursive RW grant that covered them, so the
    jailed child could rewrite the machine's own spec and scripts. Landlock
    rules combine permissively, so an ancestor grant always wins."""
    ws = tmp_path / "ws"
    (ws / "ops" / "scripts").mkdir(parents=True)
    asm = ws / "ops" / "deploy.asm.toml"
    asm.write_text("machine = 'deploy'\n", encoding="utf-8")
    step = ws / "ops" / "scripts" / "step.py"
    step.write_text("print('original')\n", encoding="utf-8")
    sibling = ws / "ops" / "notes.md"  # a NON-protected sibling must stay writable
    sibling.write_text("notes\n", encoding="utf-8")

    targets = f"(({str(asm)!r}, 'ASM'), ({str(step)!r}, 'STEP'), ({str(sibling)!r}, 'SIBLING'))"
    script = (
        "import pathlib\n"
        f"for p, tag in {targets}:\n"
        "    try:\n"
        "        pathlib.Path(p).write_text('PWNED-' + tag)\n"
        "        print('WROTE', tag)\n"
        "    except OSError:\n"
        "        print('DENIED', tag)\n"
    )

    try:
        res = run_in_jail(
            JailPolicy(
                cwd=ws,
                argv=("/usr/bin/python3", "-c", script),
                isolation="hardened",
                extra_protect_paths=(asm, ws / "ops" / "scripts"),
                timeout_s=20.0,
            )
        )
    except JailUnavailableError:
        pytest.skip("jail unavailable")

    assert "DENIED ASM" in res.stdout, f"the machine spec was writable: {res.stdout!r}"
    assert "DENIED STEP" in res.stdout, f"the machine scripts were writable: {res.stdout!r}"
    # The carve-out must stay precise: a non-protected sibling under the same
    # directory keeps its write access (strict re-binds only the protect paths).
    assert "WROTE SIBLING" in res.stdout, f"the carve-out over-denied: {res.stdout!r}"
    assert asm.read_text(encoding="utf-8").startswith("machine =")
    assert step.read_text(encoding="utf-8").startswith("print('original')")


def test_jail_hardened_protect_path_symlink_cannot_be_written_through(
    jail_bin: Path, tmp_path: Path
) -> None:
    """A symlink whose target resolves AT OR BELOW a protect path must not open
    a write channel to the protected inode. The carve-out compared each entry to
    the protect set by EQUALITY, so `ops/link -> scripts/step.py` (canon
    ops/scripts/step.py, a strict descendant of the protected ops/scripts) was
    not skipped; PathFd::new followed the symlink and granted RW on step.py's own
    inode, so the child could rewrite it by its direct path."""
    ws = tmp_path / "ws"
    (ws / "ops" / "scripts").mkdir(parents=True)
    step = ws / "ops" / "scripts" / "step.py"
    step.write_text("print('original')\n", encoding="utf-8")
    sibling = ws / "ops" / "notes.md"  # a non-protected sibling stays writable
    sibling.write_text("notes\n", encoding="utf-8")
    (ws / "ops" / "link").symlink_to("scripts/step.py")  # canon -> under a protect path

    targets = (
        f"(({str(step)!r}, 'STEP'), ({str(ws / 'ops' / 'link')!r}, 'LINK'),"
        f" ({str(sibling)!r}, 'SIBLING'))"
    )
    script = (
        "import pathlib\n"
        f"for p, tag in {targets}:\n"
        "    try:\n"
        "        pathlib.Path(p).write_text('PWNED-' + tag)\n"
        "        print('WROTE', tag)\n"
        "    except OSError:\n"
        "        print('DENIED', tag)\n"
    )
    try:
        res = run_in_jail(
            JailPolicy(
                cwd=ws,
                argv=("/usr/bin/python3", "-c", script),
                isolation="hardened",
                extra_protect_paths=(ws / "ops" / "scripts",),
                timeout_s=20.0,
            )
        )
    except JailUnavailableError:
        pytest.skip("jail unavailable")

    assert "DENIED STEP" in res.stdout, f"protected file writable directly: {res.stdout!r}"
    assert "DENIED LINK" in res.stdout, f"protected file writable via symlink: {res.stdout!r}"
    assert "WROTE SIBLING" in res.stdout, f"the carve-out over-denied: {res.stdout!r}"
    assert step.read_text(encoding="utf-8").startswith("print('original')")


def test_hardened_protects_git_from_the_filter_escape(jail_bin: Path, tmp_path: Path) -> None:
    """`.git` must be unwritable under HARDENED too, not just strict.

    It used to be writable there ("recoverable, and nothing sensitive is
    exposed"), but a jailed command could plant a `filter.<n>.clean` in
    .git/config plus a .gitattributes, and agent6's own auto-commit then ran
    that command on the HOST -- outside the jail, in the agent's Landlock
    domain, where it read $HOME and reached the network. Hardened is the common
    downgrade (userns-blocked Ubuntu, default-seccomp Docker), so this is the
    default posture for most Linux hosts."""
    git_dir = tmp_path / ".git"
    git_dir.mkdir()
    (git_dir / "config").write_text("[core]\n\trepositoryformatversion = 0\n", encoding="utf-8")
    res = run_in_jail(
        JailPolicy(
            cwd=tmp_path,
            argv=(
                "/bin/sh",
                "-c",
                "printf '[filter]\\n\\tclean = sh evil.sh\\n' >> .git/config; cat .git/config",
            ),
            isolation="hardened",
            extra_protect_paths=(git_dir,),
            timeout_s=10.0,
        )
    )
    assert "filter" not in res.stdout
    assert "filter" not in (git_dir / "config").read_text(encoding="utf-8")


@pytest.mark.parametrize("isolation", ["strict", "hardened"])
def test_no_command_leaves_a_process_running(
    jail_bin: Path, tmp_path: Path, isolation: IsolationLevel
) -> None:
    """A command must not outlive its own call ("no persistence after the run").

    strict confines the child in a PID namespace, so its whole tree dies with
    it. hardened has no namespace: `setsid` leaves the launcher's process group,
    so the launcher's killpg misses it and it reparents away. It has to reparent
    to the agent (PR_SET_CHILD_SUBREAPER) and be killed there instead.

    The command waits for the daemon's marker before returning, so a jail where
    `setsid` never ran times out rather than passing on a daemon that was never
    started.
    """
    beat = tmp_path / "beat"
    up = tmp_path / "up"
    # Workspace-relative: strict's private /tmp tmpfs has none of the host's dirs.
    daemon = "echo up > up; for i in 1 2 3 4 5 6; do echo x >> beat; sleep 1; done"
    res = run_in_jail(
        JailPolicy(
            cwd=tmp_path,
            argv=(
                "/bin/sh",
                "-c",
                f"setsid /bin/sh -c {daemon!r} </dev/null >/dev/null 2>&1 &\n"
                "while [ ! -f up ]; do sleep 0.05; done",
            ),
            isolation=isolation,
            timeout_s=10.0,
        )
    )
    assert res.returncode == 0
    assert up.exists(), f"{isolation}: the daemon never started; the test would prove nothing"
    at_return = beat.read_text(encoding="utf-8") if beat.exists() else ""
    time.sleep(2.5)
    later = beat.read_text(encoding="utf-8") if beat.exists() else ""
    assert later == at_return, f"{isolation}: a process survived the command ({later!r})"


def test_a_hostile_process_name_cannot_break_the_sweep(jail_bin: Path, tmp_path: Path) -> None:
    """`/proc/<pid>/stat` carries comm verbatim, so a process can name itself
    something that is not valid UTF-8. Decoding the sweep's scan made ONE such
    process anywhere on the host -- the scan reads every pid, not just ours --
    raise out of every later jailed command, evading the sweep and killing
    run_command with it."""
    code = "import ctypes,time; ctypes.CDLL(None).prctl(15, b'x\\xffy'); time.sleep(30)"
    proc = subprocess.Popen([sys.executable, "-c", code])
    try:
        deadline = time.monotonic() + 5.0
        while b"\xff" not in Path(f"/proc/{proc.pid}/comm").read_bytes():
            if time.monotonic() > deadline:
                pytest.skip("child never renamed itself")
            time.sleep(0.05)
        res = run_in_jail(
            JailPolicy(cwd=tmp_path, argv=("/bin/true",), isolation="hardened", timeout_s=10.0)
        )
        assert res.returncode == 0
    finally:
        proc.kill()
        proc.wait()


@pytest.mark.parametrize("isolation", ["strict", "hardened"])
def test_a_jailed_command_cannot_set_the_setuid_bit(
    jail_bin: Path, tmp_path: Path, isolation: IsolationLevel
) -> None:
    """The bit lands on the HOST inode and outlives the jail. Under
    `sudo agent6 --allow-root` the uid_map makes the jailed child real root, so
    `cp /bin/sh x && chmod 4755 x` would leave a setuid-root shell in the
    operator's workspace -- local root for anyone who runs it. Mount nosuid
    does not help: it stops the JAIL honouring the bit, not the host.
    """
    target = tmp_path / "x"
    run_in_jail(
        JailPolicy(
            cwd=tmp_path,
            argv=("/bin/sh", "-c", "cp /bin/sh x && chmod 4755 x"),
            isolation=isolation,
            timeout_s=20.0,
        )
    )
    assert target.exists(), "the copy itself must still work"
    assert not target.stat().st_mode & stat.S_ISUID
    assert not target.stat().st_mode & stat.S_ISGID


def test_the_strict_jail_names_its_own_uts_namespace(jail_bin: Path, tmp_path: Path) -> None:
    """Unsharing CLONE_NEWUTS inherits the host's name, so `uname -n` read the
    operator's machine (on a cloud box, its project too) out to every jailed
    command and into the transcript."""
    res = run_in_jail(
        JailPolicy(
            cwd=tmp_path,
            argv=("python3", "-c", "import os; print(os.uname().nodename)"),
            isolation="strict",
            timeout_s=20.0,
        )
    )
    assert res.stdout.strip() == "agent6", res.stderr


@pytest.mark.parametrize("isolation", ["strict", "hardened"])
def test_the_setuid_block_covers_the_create_family(
    jail_bin: Path, tmp_path: Path, isolation: IsolationLevel
) -> None:
    """chmod is not the only way to write the bit: creat(2) and mknod(2) take a
    mode outright and open/openat take one with O_CREAT or O_TMPFILE, so the
    filter that stops `chmod 4755` left three ways to the same host inode.
    Ordinary creates through the same syscalls stay allowed.
    """
    probe = (
        "import ctypes, os, stat\n"
        "libc = ctypes.CDLL(None, use_errno=True)\n"
        "def t(fn):\n"
        "    try:\n"
        "        fn(); return 'ok'\n"
        "    except OSError:\n"
        "        return 'refused'\n"
        "print('open', t(lambda: os.close(os.open('a', os.O_CREAT | os.O_WRONLY, 0o4755))))\n"
        "print('setgid', t(lambda: os.close(os.open('g', os.O_CREAT | os.O_WRONLY, 0o2755))))\n"
        "print('tmpfile', t(lambda: os.close(os.open('.', os.O_TMPFILE | os.O_RDWR, 0o4755))))\n"
        "print('mknod', t(lambda: os.mknod('c', stat.S_IFREG | 0o4755, 0)))\n"
        "fd = libc.creat(b'b', ctypes.c_uint(0o4755))\n"
        "print('creat', 'ok' if fd >= 0 else 'refused')\n"
        "print('plain', t(lambda: os.close(os.open('p', os.O_CREAT | os.O_WRONLY, 0o755))))\n"
        "print('plain_mknod', t(lambda: os.mknod('m', stat.S_IFREG | 0o644, 0)))\n"
        "print('fifo', t(lambda: os.mkfifo('f')))\n"
    )
    res = run_in_jail(
        JailPolicy(cwd=tmp_path, argv=("python3", "-c", probe), isolation=isolation, timeout_s=20.0)
    )
    got = dict(line.split() for line in res.stdout.split("\n") if line.strip())
    assert got == {
        "open": "refused",
        "setgid": "refused",
        "tmpfile": "refused",
        "mknod": "refused",
        "creat": "refused",
        "plain": "ok",
        "plain_mknod": "ok",
        "fifo": "ok",
    }, res.stdout
    for name in ("a", "g", "c", "b"):
        assert not (tmp_path / name).exists() or not (tmp_path / name).stat().st_mode & (
            stat.S_ISUID | stat.S_ISGID
        )


@pytest.mark.parametrize("isolation", ["strict", "hardened"])
def test_the_setuid_block_covers_fchmodat2(
    jail_bin: Path, tmp_path: Path, isolation: IsolationLevel
) -> None:
    """The same threat via the syscall that SUPERSEDED fchmodat.

    fchmodat2 (Linux 6.6+) takes the mode in the same argument and was absent
    from the filter, so on a new kernel the exact write the block exists to stop
    went through: probed against a build without it, the bit landed and the file
    came back mode 4755. `chmod` cannot reach it -- coreutils still calls
    fchmodat -- so it takes a direct syscall to pin.
    """
    probe = (
        "import ctypes, os\n"
        "libc = ctypes.CDLL(None, use_errno=True)\n"
        "open('x', 'w').close()\n"
        "libc.syscall(ctypes.c_long(452), ctypes.c_int(-100), b'x',"
        " ctypes.c_uint(0o4755), ctypes.c_int(0))\n"
    )
    res = run_in_jail(
        JailPolicy(
            cwd=tmp_path,
            argv=("python3", "-c", probe),
            isolation=isolation,
            timeout_s=20.0,
        )
    )
    target = tmp_path / "x"
    if not target.exists():
        pytest.skip(f"probe did not run: {res.stderr[:200]}")
    assert not target.stat().st_mode & stat.S_ISUID
    assert not target.stat().st_mode & stat.S_ISGID


@pytest.mark.parametrize("isolation", ["strict", "hardened"])
def test_ordinary_chmod_still_works(
    jail_bin: Path, tmp_path: Path, isolation: IsolationLevel
) -> None:
    """Only the setid bits are refused; chmod is ordinary work and denying it
    outright would break builds, scripts and installers."""
    res = run_in_jail(
        JailPolicy(
            cwd=tmp_path,
            argv=("/bin/sh", "-c", "touch a && chmod 640 a && stat -c %a a"),
            isolation=isolation,
            timeout_s=20.0,
        )
    )
    assert res.returncode == 0
    assert res.stdout.strip() == "640"


def test_jail_hidden_paths_mask_secrets_under_a_broad_grant(
    jail_bin: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The agent6-private dirs never enter the jail, even through an explicit
    extra_read_paths grant of the home dir that CONTAINS them -- the launcher
    masks them last, after every bind. A policy grant BENEATH a hidden root
    (the machine data contract) is re-bound through the mask and stays
    writable. Everything else under the grant stays readable."""
    home = tmp_path / "fakehome"
    cfg_dir = home / ".config" / "agent6"
    cfg_dir.mkdir(parents=True)
    (cfg_dir / "secrets.toml").write_text("key = 'sk-SECRET'\n", encoding="utf-8")
    (home / "notes.txt").write_text("plain\n", encoding="utf-8")
    state = home / ".local" / "state" / "agent6"
    data = state / "repo" / "machines" / "m1" / "data"
    data.mkdir(parents=True)
    (data / "journal.txt").write_text("j1\n", encoding="utf-8")
    monkeypatch.setenv("AGENT6_CONFIG_HOME", str(cfg_dir))
    monkeypatch.setenv("AGENT6_STATE_HOME", str(state))

    ws = tmp_path / "ws"
    ws.mkdir()
    script = (
        f"cat {home}/notes.txt; "
        f"cat {cfg_dir}/secrets.toml 2>&1; "
        f"cat {data}/journal.txt && echo j2 > {data}/journal.txt && echo WROTE"
    )
    res = run_in_jail(
        JailPolicy(
            cwd=ws,
            argv=("/bin/sh", "-c", script),
            extra_ro_paths=(home,),
            extra_rw_paths=(data,),
            timeout_s=10.0,
        )
    )
    assert "plain" in res.stdout  # the grant itself works
    assert "SECRET" not in res.stdout  # the hidden dir masked out of it
    assert "j1" in res.stdout and "WROTE" in res.stdout  # the hole re-bound RW
    assert (data / "journal.txt").read_text(encoding="utf-8") == "j2\n"


def test_jail_hidden_paths_cover_the_workspace_alias(
    jail_bin: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """cwd = $HOME puts the private dirs inside the workspace bind; the mask
    covers the /workspace alias too, so there is no second door."""
    cfg_dir = tmp_path / ".config" / "agent6"
    cfg_dir.mkdir(parents=True)
    (cfg_dir / "secrets.toml").write_text("key = 'sk-SECRET'\n", encoding="utf-8")
    (tmp_path / "readme.txt").write_text("hello\n", encoding="utf-8")
    monkeypatch.setenv("AGENT6_CONFIG_HOME", str(cfg_dir))

    res = run_in_jail(
        JailPolicy(
            cwd=tmp_path,
            argv=("/bin/sh", "-c", "cat readme.txt; cat .config/agent6/secrets.toml 2>&1"),
            timeout_s=10.0,
        )
    )
    assert "hello" in res.stdout
    assert "SECRET" not in res.stdout


def test_jail_operator_hide_paths_mask_a_file(jail_bin: Path, tmp_path: Path) -> None:
    """A [sandbox].hide_paths FILE entry inside the workspace reads empty in
    the jail while its siblings stay readable, and the host copy is intact."""
    private = tmp_path / "cred.txt"
    private.write_text("token\n", encoding="utf-8")
    (tmp_path / "ok.txt").write_text("fine\n", encoding="utf-8")
    res = run_in_jail(
        JailPolicy(
            cwd=tmp_path,
            argv=("/bin/sh", "-c", "cat ok.txt; cat cred.txt; echo rc=$?"),
            hide_paths=(private,),
            timeout_s=10.0,
        )
    )
    assert "fine" in res.stdout
    assert "token" not in res.stdout
    assert private.read_text(encoding="utf-8") == "token\n"


def test_jail_home_exists_in_the_private_tmpfs(jail_bin: Path, tmp_path: Path) -> None:
    # strict's default HOME is /tmp/agent6-home; the launcher creates it in the
    # fresh tmpfs, so `cd ~` and a toolchain's first write under it work.
    try:
        res = run_in_jail(
            JailPolicy(
                cwd=tmp_path,
                argv=("/bin/sh", "-c", 'test -d "$HOME" && cd ~ && touch .probe'),
                env=(("HOME", "/tmp/agent6-home"),),
                timeout_s=10.0,
            )
        )
    except JailUnavailableError:
        pytest.skip("jail unavailable")
    assert res.returncode == 0, (res.stdout, res.stderr)


def test_jail_fork_worktree_reads_the_repository_git(jail_bin: Path, tmp_path: Path) -> None:
    """A fork's leg runs in a linked worktree whose `.git` is a pointer into
    the repository's; the policy grants the git dir agent6 recorded for the
    worktree read-only, so `git` works there under strict and cannot write
    it. A policy without the grant cannot even find the repository."""
    from agent6.config import Config
    from agent6.git_ops import add_worktree
    from agent6.tools.policy import jail_policy

    repo = tmp_path / "repo"
    subprocess.run(["git", "init", "-q", "-b", "main", str(repo)], check=True)
    (repo / "a.txt").write_text("a\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(repo), "add", "-A"], check=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-q", "-m", "init"], check=True)
    wt = tmp_path / "wt"
    add_worktree(repo, wt, "HEAD")

    def jailed(*argv: str, granted: bool) -> tuple[int, str, str]:
        # A git command needs no network; the run's own policy would attach
        # the session network, which this test has no run to take it from.
        # `granted=False` is a raw policy the builder never shaped.
        policy = (
            jail_policy(
                wt,
                Config(),
                "strict",
                argv,
                timeout_s=30.0,
                network="none",
                worktree_git_dir=(repo / ".git").resolve(),
            )
            if granted
            else JailPolicy(cwd=wt, argv=argv, timeout_s=30.0)
        )
        try:
            res = run_in_jail(policy)
        except JailUnavailableError:
            pytest.skip("jail unavailable")
        return res.returncode, res.stdout.strip(), res.stderr.strip()

    rc, out, _err = jailed("git", "rev-parse", "--show-toplevel", granted=True)
    assert rc == 0 and out == str(wt.resolve())
    rc, out, _err = jailed("git", "log", "--oneline", "-1", granted=True)
    assert rc == 0 and out.endswith("init")
    rc, _out, err = jailed("git", "commit", "-q", "--allow-empty", "-m", "x", granted=True)
    assert rc != 0 and "Read-only file system" in err
    rc, _out, err = jailed("git", "rev-parse", "--show-toplevel", granted=False)
    assert rc != 0 and "not a git repository" in err


def test_strict_runs_the_command_as_the_operators_own_uid(jail_bin: Path, tmp_path: Path) -> None:
    """The user namespace maps the operator's uid to itself, so `id -u` reads
    the same number in and out of the jail. Mapped to 0, a command read as
    root: `tar` restored archive owners, the single-uid map refused the chown,
    and extracting an archive the operator had just made exited 2."""
    (tmp_path / "f.txt").write_text("hi", encoding="utf-8")
    subprocess.run(["tar", "-cf", "a.tar", "f.txt"], cwd=tmp_path, check=True)
    (tmp_path / "out").mkdir()
    res = run_in_jail(
        JailPolicy(
            cwd=tmp_path,
            argv=("/bin/sh", "-c", "id -u && tar -xf a.tar -C out"),
            isolation="strict",
            timeout_s=20.0,
        )
    )
    assert res.returncode == 0, res.stderr
    assert res.stdout.split()[0] == str(os.getuid())
    assert (tmp_path / "out" / "f.txt").read_text(encoding="utf-8") == "hi"
