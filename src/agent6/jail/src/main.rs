// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Eric Lesiuta
//! agent6-jail: minimal sandbox launcher.
//!
//! Reads a single JSON policy on stdin, sets up:
//!   - new mount/pid/net/ipc/uts namespaces (user namespace too, so we can mount unprivileged)
//!   - minimal bind-mounted rootfs (cwd RW, /usr /bin /lib /lib64 RO, tmpfs /tmp)
//!   - Landlock FS rules
//!   - seccomp-bpf deny-list (default-allow, EPERM on the dangerous syscalls)
//!   - PR_SET_NO_NEW_PRIVS
//!   - RLIMIT_DATA memory cap on the child (policy `memory_limit_mb`)
//!
//! Then forks + execs the child, captures stdout/stderr, prints one JSON result line.
//!
//! Exits 0 if it successfully ran the child (regardless of child exit code).
//! Exits non-zero only when sandbox SETUP failed — in that case stderr explains why.

use std::collections::VecDeque;
use std::ffi::{CString, OsStr};
use std::fs::{self, File};
use std::io::{self, BufRead, Read, Write};
use std::os::fd::{FromRawFd, RawFd};
use std::os::unix::fs::{FileTypeExt, OpenOptionsExt};
use std::os::unix::process::{CommandExt, ExitStatusExt};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use landlock::{
    Access, AccessFs, BitFlags, PathBeneath, PathFd, Ruleset, RulesetAttr, RulesetCreated,
    RulesetCreatedAttr, RulesetStatus, ABI,
};
use nix::mount::{mount, umount2, MntFlags, MsFlags};
use nix::sched::{setns, unshare, CloneFlags};
use nix::sys::statvfs::{statvfs, FsFlags};
use nix::sys::wait::{waitid, waitpid, Id, WaitPidFlag, WaitStatus};
use nix::unistd::{chdir, fork, getgid, getuid, pivot_root, ForkResult, Pid};
use seccompiler::{BpfProgram, SeccompAction, SeccompFilter, TargetArch};
use serde::Deserialize;

// deny_unknown_fields: a policy field this binary does not know (version skew
// between the Python side and a stale or pinned AGENT6_JAIL_BIN) could be a
// restriction it would silently drop, so refuse instead of running weaker.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Policy {
    #[serde(default = "default_isolation")]
    isolation: String,
    cwd: PathBuf,
    argv: Vec<String>,
    #[serde(default)]
    env: Vec<(String, String)>,
    /// Which network this child joins: "host" (the machine's), "session" (the
    /// run's own, shared with its siblings, no route off the box) or "none" (its
    /// own, alone). "session" needs --userns-fd/--netns-fd naming the run's
    /// holder; joining is why those two arrive together (see join_network).
    #[serde(default = "default_network")]
    network: String,
    /// Operator-granted extra paths, bind-mounted at their REAL locations in
    /// strict (like tool_paths and hardened), so a granted
    /// toolchain works via its own absolute paths and shebangs. ro is
    /// read+execute, rw is read+write. A ro grant under a system mount is
    /// redundant (already visible read+exec) and skipped; an rw grant under
    /// one cannot be honored and fails the run loudly.
    #[serde(default)]
    extra_ro_paths: Vec<PathBuf>,
    #[serde(default)]
    extra_rw_paths: Vec<PathBuf>,
    /// Operator-granted device nodes under /dev ([sandbox].extra_device_paths),
    /// bound into the jail's /dev like the builtin five (plain bind, no nodev
    /// floor -- the floor would make the grant dead). Each must be a character
    /// or block device on the host; anything else refuses loudly. Landlock
    /// grants read+write on the node; ioctl on the opened fd is outside
    /// Landlock's file scope.
    #[serde(default)]
    extra_device_paths: Vec<PathBuf>,
    /// Real-location RO+exec bind mounts for operator-installed tools (uv, node,
    /// ...) that live outside the system dirs -- ~/.local/bin, ~/.cargo/bin, or the
    /// /opt target a /usr/local/bin symlink resolves to. Real paths mean PATH
    /// lookups and symlinks resolve. Read+execute only, never writable.
    #[serde(default)]
    tool_paths: Vec<PathBuf>,
    /// Paths masked from the child even when a broader grant covers them
    /// (agent6's own config + state dirs, plus operator additions):
    /// a dir masks as an empty tmpfs, a file as a bind of /dev/null. Masked
    /// LAST, after every bind, so no grant exposes them from above; a policy
    /// grant BENEATH a hidden root is then re-bound through the mask (the
    /// machine data contract: explicit holes in a default-deny cover).
    #[serde(default)]
    hide_paths: Vec<PathBuf>,
    /// Paths inside `cwd` to make READ-ONLY from the child's view. In
    /// strict, these are re-bound RO on top of the workspace mount. In
    /// hardened (no mount namespace), the Landlock ruleset switches from
    /// "RW on cwd" to "R on cwd + RW on each top-level entry except
    /// these" — same end result for files that exist at jail-launch time,
    /// at the cost of denying writes to new top-level entries created
    /// after the jail starts. Used to keep an LLM-driven `run_command`
    /// from rewriting `.git`. Each entry
    /// must be absolute; entries that don't exist on disk are skipped.
    #[serde(default)]
    extra_protect_paths: Vec<PathBuf>,
    #[serde(default = "default_timeout")]
    timeout_s: f64,
    /// Per-process memory cap in MiB, applied via RLIMIT_DATA in the child
    /// before exec and inherited by every descendant. 0 disables, and is the
    /// default on both sides: the kernel already handles a memory bomb, and a
    /// cap costs real builds more than it buys.
    #[serde(default = "default_memory_limit_mb")]
    memory_limit_mb: u64,
    /// "once" (default): run `argv` and exit. "serve": after the same setup,
    /// read newline-delimited JSON requests from stdin and run each in THESE
    /// namespaces, so a run's commands share one netns, one PID namespace and
    /// one /tmp -- and a backgrounded server outlives the command that started
    /// it. EOF on stdin tears the namespace (and everything in it) down.
    /// "exec": same setup, then run ONE long-lived child on our own stdio and
    /// exit with its status -- for a process agent6 talks to (an MCP server)
    /// rather than collects. Requires --policy-fd, since stdin is the child's.
    #[serde(default = "default_mode")]
    mode: String,
}

/// One command to run against an already-established jail.
// deny_unknown_fields, like Policy above: a request naming a field this launcher
// does not know is a version skew between the Python side and this binary, and
// silently dropping it could mean dropping a confinement the caller meant to set.
// Refuse it rather than run a request we only half-understand.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ChildRequest {
    argv: Vec<String>,
    #[serde(default)]
    env: Vec<(String, String)>,
    #[serde(default = "default_timeout")]
    timeout_s: f64,
    #[serde(default = "default_memory_limit_mb")]
    memory_limit_mb: u64,
    #[serde(default)]
    checkin_s: f64,
    #[serde(default)]
    log_dir: String,
}

impl ChildRequest {
    fn child_spec(&self) -> ChildSpec<'_> {
        ChildSpec {
            argv: &self.argv,
            env: &self.env,
            memory_limit_mb: self.memory_limit_mb,
            timeout_s: self.timeout_s,
            checkin_s: self.checkin_s,
            log_dir: &self.log_dir,
        }
    }
}

/// What a serving launcher is asked to do. A backgrounded command's pid is
/// namespace-local, so the launcher is the only process that can wait on it or
/// signal it: `status` and `stop` name that pid and it does the work.
#[derive(Deserialize)]
#[serde(tag = "kind", rename_all = "lowercase")]
enum Request {
    /// Run to completion; answer with the result.
    Run(ChildRequest),
    /// Start the command and answer at once, leaving it running in this
    /// session's namespaces. STRICT only: the PID namespace is what bounds it,
    /// and hardened has none, so there a backgrounded process would outlive
    /// the run.
    Background(ChildRequest),
    /// Report whether a backgrounded command is still running, reaping it and
    /// answering with its exit code once it is not.
    Status { pid: i32 },
    /// Kill a backgrounded command's process group, then reap it.
    Stop { pid: i32 },
}

fn default_mode() -> String {
    "once".to_string()
}

fn default_isolation() -> String {
    "strict".to_string()
}

fn default_network() -> String {
    "none".to_string()
}

fn default_timeout() -> f64 {
    600.0
}

fn default_memory_limit_mb() -> u64 {
    0
}

fn die(msg: impl AsRef<str>) -> ! {
    eprintln!("agent6-jail: {}", msg.as_ref());
    std::process::exit(2);
}

/// `--hold-netns`: create the run's session network and hold it open until
/// stdin closes. It runs no child and confines nothing -- the run opens
/// /proc/<pid>/ns/{user,net} once "ready" appears, and those descriptors are
/// what keep the namespaces alive, so this process may exit immediately after.
fn hold_netns() -> ! {
    let (uid, gid) = (getuid(), getgid());
    if let Err(e) = unshare(CloneFlags::CLONE_NEWUSER | CloneFlags::CLONE_NEWNET) {
        die(format!("session network: unshare failed: {e}"));
    }
    fs::write("/proc/self/setgroups", "deny").ok();
    if let Err(e) = fs::write("/proc/self/uid_map", format!("{uid} {uid} 1\n"))
        .and_then(|()| fs::write("/proc/self/gid_map", format!("{gid} {gid} 1\n")))
    {
        die(format!("session network: id map failed: {e}"));
    }
    if let Err(e) = bring_loopback_up() {
        die(format!("session network: loopback failed: {e}"));
    }
    println!("ready");
    if io::stdout().flush().is_err() {
        die("session network: could not report readiness");
    }
    let mut sink = String::new();
    let _ = io::stdin().lock().read_line(&mut sink);
    std::process::exit(0);
}

fn fd_arg(args: &[String], name: &str) -> Option<RawFd> {
    match args.iter().position(|a| a == name) {
        Some(i) => match args.get(i + 1).and_then(|n| n.parse::<RawFd>().ok()) {
            Some(fd) if fd > 2 => Some(fd),
            _ => die(format!("{name} needs a file descriptor number above 2")),
        },
        None => None,
    }
}

fn main() {
    // `--policy-fd N` reads the policy from an inherited fd instead of stdin,
    // for exec mode where stdin belongs to the child (its JSON-RPC pipe).
    // Without it, the policy is the FIRST LINE of stdin -- not the whole
    // stream: in serve mode the same pipe then carries one request per line.
    let mut input = String::new();
    let args: Vec<String> = std::env::args().collect();
    if args.iter().any(|a| a == "--hold-netns") {
        hold_netns();
    }
    let userns_fd = fd_arg(&args, "--userns-fd");
    let netns_fd = fd_arg(&args, "--netns-fd");
    let policy_fd = fd_arg(&args, "--policy-fd");
    // The client's way to say "hand the running command back NOW" without
    // waiting out the check-in: one byte on this pipe. A separate channel, not
    // the request pipe, because that one is in lockstep -- the client is
    // blocked reading the answer to the very request this interrupts.
    let interrupt_fd = fd_arg(&args, "--interrupt-fd");
    if let Some(fd) = interrupt_fd {
        prepare_interrupt_fd(fd);
    }
    let read_ok = match policy_fd {
        // from_raw_fd owns the fd and closes it at end of scope, so the child
        // never inherits the policy channel.
        Some(fd) => {
            let file = unsafe { std::fs::File::from_raw_fd(fd) };
            io::BufReader::new(file).read_line(&mut input).is_ok()
        }
        None => io::stdin().lock().read_line(&mut input).is_ok(),
    };
    if !read_ok || input.trim().is_empty() {
        die("failed to read policy");
    }
    let policy: Policy = match serde_json::from_str(&input) {
        Ok(p) => p,
        Err(e) => die(format!("invalid policy JSON: {e}")),
    };

    if !matches!(policy.network.as_str(), "host" | "session" | "none") {
        die(format!("unknown network: {}", policy.network));
    }
    // "session" is the run's shared network, and the only way in is the pair of
    // descriptors naming its holder. Refuse rather than silently running in a
    // namespace of our own, which would look identical and be isolated.
    let join = match (policy.network.as_str(), userns_fd, netns_fd) {
        ("session", Some(u), Some(n)) => Some((u, n)),
        ("session", _, _) => die("network = session needs --userns-fd and --netns-fd"),
        _ => None,
    };
    match policy.isolation.as_str() {
        "strict" => run_strict(&policy, join, interrupt_fd),
        "hardened" => run_hardened(&policy, interrupt_fd),
        "none" => run_unconfined(&policy, interrupt_fd),
        other => die(format!("unknown sandbox isolation: {other}")),
    }
}

/// Make the interrupt pipe non-blocking (the poll below must never wait) and
/// close-on-exec (a jailed command must not inherit the channel that steers
/// its own hand-back).
fn prepare_interrupt_fd(fd: RawFd) {
    unsafe {
        let flags = libc::fcntl(fd, libc::F_GETFL);
        if flags < 0 || libc::fcntl(fd, libc::F_SETFL, flags | libc::O_NONBLOCK) < 0 {
            die("could not set O_NONBLOCK on --interrupt-fd");
        }
        if libc::fcntl(fd, libc::F_SETFD, libc::FD_CLOEXEC) < 0 {
            die("could not set FD_CLOEXEC on --interrupt-fd");
        }
    }
}

/// Whether the client has asked for an immediate hand-back. Drains what is
/// there: one request is one hand-back, and a byte left behind would convert
/// the next command the moment it started.
fn interrupt_requested(fd: Option<RawFd>) -> bool {
    let Some(fd) = fd else { return false };
    let mut buf = [0u8; 64];
    let mut asked = false;
    loop {
        let n = unsafe { libc::read(fd, buf.as_mut_ptr().cast(), buf.len()) };
        if n <= 0 {
            return asked; // EAGAIN (nothing pending), EOF, or an error
        }
        asked = true;
    }
}

fn run_strict(policy: &Policy, join: Option<(RawFd, RawFd)>, interrupt_fd: Option<RawFd>) -> ! {
    let real_uid = getuid().as_raw();
    if let Err(e) = setup_namespaces(&policy.network, join) {
        die(format!("namespace setup failed: {e}"));
    }
    // After unshare(CLONE_NEWPID), the parent process itself remains in the OLD
    // pid namespace — only its children, forked after the unshare, enter the new
    // pid namespace. Anything that requires being inside the new pid ns (most
    // notably mounting a fresh /proc) must therefore happen in a forked child.
    let launcher_pid = std::process::id() as libc::pid_t;
    match unsafe { fork() } {
        Ok(ForkResult::Parent { child }) => match waitpid(child, None) {
            Ok(WaitStatus::Exited(_, code)) => std::process::exit(code),
            Ok(WaitStatus::Signaled(_, sig, _)) => std::process::exit(128 + sig as i32),
            Ok(other) => die(format!("unexpected wait status: {other:?}")),
            Err(e) => die(format!("waitpid failed: {e}")),
        },
        Ok(ForkResult::Child) => {
            // This process is PID 1 of the new pid namespace: when it dies the
            // kernel kills everything in the namespace. Tying it to the
            // launcher completes the chain (agent6 -> launcher -> here ->
            // every jailed process), so a SIGKILLed agent6 cannot strand a
            // running command. The fork race (launcher dead before the prctl
            // lands = no signal ever) cannot be closed with getppid: inside
            // the fresh pid namespace it reads 0 whether the launcher lives
            // or died. The inherited host /proc is still mounted here
            // (setup_rootfs runs below), so consult it; if /proc itself is
            // absent there is nothing to consult and the tie stands alone.
            unsafe {
                if libc::prctl(libc::PR_SET_PDEATHSIG, libc::SIGKILL, 0, 0, 0) != 0 {
                    die(format!(
                        "PR_SET_PDEATHSIG failed: {}",
                        io::Error::last_os_error()
                    ));
                }
            }
            let launcher_proc = format!("/proc/{launcher_pid}");
            if Path::new("/proc/self").exists() && !Path::new(&launcher_proc).exists() {
                die("launcher died before the death-signal tie landed");
            }
            if let Err(e) = setup_rootfs(policy, real_uid) {
                die(format!("rootfs setup failed: {e}"));
            }
            if let Err(e) = apply_landlock_strict(policy) {
                die(format!("landlock failed: {e}"));
            }
            if let Err(e) = apply_seccomp() {
                die(format!("seccomp failed: {e}"));
            }
            if let Err(e) = hide_from_children() {
                die(format!("PR_SET_DUMPABLE failed: {e}"));
            }
            if policy.mode == "serve" {
                serve(&policy.cwd, true, interrupt_fd);
            }
            // A one-shot launcher never hands a command back (its spec sets
            // no check-in), so there is no pid to carry out of here.
            let outcome = if policy.mode == "exec" {
                run_child_exec(&policy.child_spec(), &policy.cwd)
            } else {
                run_child(&policy.child_spec(), &policy.cwd, None).map(|_| ())
            };
            if let Err(e) = outcome {
                die(format!("child execution failed: {e}"));
            }
            std::process::exit(0);
        }
        Err(e) => die(format!("fork failed: {e}")),
    }
}

fn run_unconfined(policy: &Policy, interrupt_fd: Option<RawFd>) -> ! {
    // `isolation = "none"`: the operator's explicit opt-out, or a host with no
    // confinement mechanism at all. NOTHING here confines the child -- no
    // namespaces, no Landlock, no seccomp -- so this branch must stay reachable
    // ONLY for the literal string "none" (the match above is the whole gate).
    //
    // The launcher runs anyway so that output capture, the check-in and the
    // background lifecycle have ONE implementation at every isolation level
    // rather than one per level plus a Python copy. It costs the invariant that
    // "the launcher ran" implied "confinement was applied", so the warning
    // below is loud and the Python side surfaces it as `jail.degraded`.
    // The memory rlimit is not confinement and is not disabled here: a
    // configured memory_limit_mb still applies per request on every transport.
    eprintln!(
        "[agent6-jail] WARNING: running UNCONFINED (isolation = \"none\"): no namespaces, \
         no Landlock, no seccomp. Commands run with this process's own access."
    );
    let _ = io::stderr().flush();
    // Still worth doing without a sandbox: in serve mode the launcher answers
    // every request on its stdout, and a command that opened /proc/<pid>/fd/1
    // could write a result line the agent reads as its own.
    if let Err(e) = hide_from_children() {
        die(format!("PR_SET_DUMPABLE failed: {e}"));
    }
    if policy.mode == "serve" {
        serve(&policy.cwd, false, interrupt_fd);
    }
    // A one-shot launcher never hands a command back (its spec sets no
    // check-in), so there is no pid to carry out of here.
    let outcome = if policy.mode == "exec" {
        run_child_exec(&policy.child_spec(), &policy.cwd)
    } else {
        run_child(&policy.child_spec(), &policy.cwd, None).map(|_| ())
    };
    if let Err(e) = outcome {
        die(format!("child execution failed: {e}"));
    }
    std::process::exit(0);
}

fn run_hardened(policy: &Policy, interrupt_fd: Option<RawFd>) -> ! {
    // No namespaces, no pivot_root. Landlock confines the FS; seccomp +
    // NO_NEW_PRIVS bound the syscall surface; we still operate on the real cwd
    // and inherit the original /proc, /tmp, network namespace from the parent.
    // This is the isolation level that runs under default-seccomp Docker where
    // CLONE_NEWUSER is blocked.
    if let Err(e) = apply_landlock_hardened(policy) {
        die(format!("landlock failed: {e}"));
    }
    if let Err(e) = apply_seccomp() {
        die(format!("seccomp failed: {e}"));
    }
    if let Err(e) = hide_from_children() {
        die(format!("PR_SET_DUMPABLE failed: {e}"));
    }
    if policy.mode == "serve" {
        serve(&policy.cwd, false, interrupt_fd);
    }
    // A one-shot launcher never hands a command back (its spec sets no
    // check-in), so there is no pid to carry out of here.
    let outcome = if policy.mode == "exec" {
        run_child_exec(&policy.child_spec(), &policy.cwd)
    } else {
        run_child(&policy.child_spec(), &policy.cwd, None).map(|_| ())
    };
    if let Err(e) = outcome {
        die(format!("child execution failed: {e}"));
    }
    std::process::exit(0);
}

/// Run one command per stdin line inside the namespaces already established,
/// answering each with the same single-line JSON result the one-shot mode
/// prints. Exits when stdin closes, which takes the PID namespace (and any
/// backgrounded server in it) down with it.
fn serve(cwd: &Path, pid_namespaced: bool, interrupt_fd: Option<RawFd>) -> ! {
    // Setup (mounts, /proc, Landlock, seccomp) is complete by the time we are
    // called, so any warning it emitted is already on our stderr. Signal that
    // with one line the client consumes before its first request: it gives
    // JailSession.open() a known point to read those warnings off stderr,
    // instead of guessing with a timeout whether setup has flushed yet.
    println!("{{\"ready\":true}}");
    if io::stdout().flush().is_err() {
        die("serve: the request channel is gone");
    }
    let stdin = io::stdin();
    let mut line = String::new();
    // Backgrounded pids, for the EOF sweep when no PID namespace bounds them.
    let mut backgrounded: Vec<i32> = Vec::new();
    loop {
        line.clear();
        match stdin.lock().read_line(&mut line) {
            Ok(0) => {
                if !pid_namespaced {
                    sweep_backgrounded(&backgrounded);
                }
                std::process::exit(0)
            }
            Ok(_) => {}
            Err(e) => die(format!("serve: reading request failed: {e}")),
        }
        if line.trim().is_empty() {
            continue;
        }
        let request: Request = match serde_json::from_str(&line) {
            Ok(r) => r,
            Err(e) => die(format!("serve: invalid request JSON: {e}")),
        };
        let outcome: io::Result<()> = match request {
            // A run that was handed back keeps running, so it is swept and
            // polled exactly like one that started in the background.
            Request::Run(child) => {
                run_child(&child.child_spec(), cwd, interrupt_fd).map(|handed_back| {
                    if let Some(pid) = handed_back {
                        backgrounded.push(pid);
                    }
                })
            }
            Request::Background(child) => {
                spawn_detached(&child.child_spec(), cwd).map(|pid| backgrounded.push(pid))
            }
            Request::Status { pid } => answer_status(pid),
            Request::Stop { pid } => answer_stop(pid),
        };
        // A command that could not be EXECUTED (bad path, missing interpreter)
        // is the caller's argv being wrong, not this jail being broken: answer
        // it like the one-shot launcher does and keep serving. Dying here cost
        // the whole run its namespaces, and every backgrounded server in them,
        // over one typo.
        if let Err(e) = outcome {
            let result = serde_json::json!({
                "returncode": 127,
                "stdout": "",
                "stderr": format!("child execution failed: {e}"),
                "exec_failed": true,
            });
            println!("{result}");
            if io::stdout().flush().is_err() {
                die("serve: the request channel is gone");
            }
        }
    }
}

/// Set IFF_UP on `lo` in the CURRENT network namespace.
fn bring_loopback_up() -> io::Result<()> {
    // SIOCSIFFLAGS on a datagram socket: no netlink dependency, and this runs
    // before seccomp is installed.
    let sock = unsafe { libc::socket(libc::AF_INET, libc::SOCK_DGRAM, 0) };
    if sock < 0 {
        return Err(io::Error::last_os_error());
    }
    let mut req: libc::ifreq = unsafe { std::mem::zeroed() };
    for (i, b) in b"lo".iter().enumerate() {
        req.ifr_name[i] = *b as libc::c_char;
    }
    req.ifr_ifru.ifru_flags = (libc::IFF_UP | libc::IFF_RUNNING) as libc::c_short;
    // The ioctl request type differs by arch (u64 on x86_64, i32 on aarch64),
    // and the release wheels build both.
    let rc = unsafe { libc::ioctl(sock, libc::SIOCSIFFLAGS as _, &req) };
    unsafe { libc::close(sock) };
    if rc < 0 {
        return Err(io::Error::last_os_error());
    }
    Ok(())
}

/// Enter the run's session network: its user namespace first, because
/// setns(CLONE_NEWNET) needs CAP_SYS_ADMIN in the namespace that OWNS the
/// target netns, and an unprivileged joiner only has that inside it. The
/// caller therefore does NOT create a user namespace of its own; it holds every
/// capability in the one it joined, which is what the mounts below need.
fn join_network(userns_fd: RawFd, netns_fd: RawFd) -> io::Result<()> {
    for (fd, kind) in [
        (userns_fd, CloneFlags::CLONE_NEWUSER),
        (netns_fd, CloneFlags::CLONE_NEWNET),
    ] {
        // from_raw_fd owns it: the child never inherits a handle to a namespace
        // it could re-enter after we drop privileges.
        let file = unsafe { std::fs::File::from_raw_fd(fd) };
        setns(&file, kind).map_err(io_err)?;
    }
    Ok(())
}

/// Name the jail's own UTS namespace. Unsharing one inherits the host's
/// hostname, so `uname -n` read the operator's machine (on a cloud box, its
/// project too) out to every jailed command and into the transcript. Not a
/// boundary: a jail that cannot rename itself still runs, and says so.
fn name_the_uts_namespace() {
    if let Err(e) = nix::unistd::sethostname("agent6") {
        eprintln!(
            "[agent6-jail] warning: sethostname failed ({e}); jailed commands read the host's \
             own hostname"
        );
    }
}

fn setup_namespaces(network: &str, join: Option<(RawFd, RawFd)>) -> io::Result<()> {
    let uid = getuid();
    let gid = getgid();

    if let Some((userns_fd, netns_fd)) = join {
        join_network(userns_fd, netns_fd)?;
        // Everything else is still this child's alone; `lo` is already up in
        // the network we joined, and the user namespace is already mapped.
        unshare(
            CloneFlags::CLONE_NEWNS
                | CloneFlags::CLONE_NEWPID
                | CloneFlags::CLONE_NEWIPC
                | CloneFlags::CLONE_NEWUTS,
        )
        .map_err(io_err)?;
        name_the_uts_namespace();
        return make_mounts_private();
    }

    let mut flags = CloneFlags::CLONE_NEWUSER
        | CloneFlags::CLONE_NEWNS
        | CloneFlags::CLONE_NEWPID
        | CloneFlags::CLONE_NEWIPC
        | CloneFlags::CLONE_NEWUTS;
    if network != "host" {
        flags |= CloneFlags::CLONE_NEWNET;
    }
    unshare(flags).map_err(io_err)?;

    // Map the current uid/gid to themselves in the new user namespace. The
    // namespace grants its creator every capability over what it owns (the
    // mounts, the loopback ioctl) whatever the map says; what the map decides
    // is the uid a command reads. Kept real rather than 0: a command reading
    // as root restores archive owners (`tar`), refuses to run (`npm`) or warns
    // (`pip`), and with one uid mapped every chown to another fails EINVAL.
    // Written BEFORE touching the new netns: until then this process is the
    // overflow uid there, which inside a container costs the loopback ioctl
    // its permission (EACCES under `docker --security-opt seccomp=unconfined`).
    fs::write("/proc/self/setgroups", "deny").ok();
    fs::write("/proc/self/uid_map", format!("{uid} {uid} 1\n"))
        .map_err(|e| io::Error::other(format!("uid_map: {e}")))?;
    fs::write("/proc/self/gid_map", format!("{gid} {gid} 1\n"))
        .map_err(|e| io::Error::other(format!("gid_map: {e}")))?;
    name_the_uts_namespace();

    if network != "host" {
        // An empty netns has `lo` DOWN, so nothing inside can reach even
        // itself. Loopback in a namespace with no other interface and no route
        // out reaches nothing beyond that namespace, so this changes what the
        // jail's own commands can talk to, never what they can leave to.
        bring_loopback_up()?;
    }

    make_mounts_private()
}

/// Stop our mount changes propagating back to the host's mount tree.
fn make_mounts_private() -> io::Result<()> {
    mount(
        Some(""),
        "/",
        Some(""),
        MsFlags::MS_REC | MsFlags::MS_PRIVATE,
        Some(""),
    )
    .map_err(io_err)?;
    Ok(())
}

/// The host dirs strict bind-mounts read-only into the rootfs. Extra-path
/// grants check against this: a ro grant under one is already visible, and an
/// rw grant under one cannot be honored (the covering bind is read-only).
const SYSTEM_BINDS: [&str; 6] = [
    "/usr",
    "/bin",
    "/sbin",
    "/lib",
    "/lib64",
    "/etc/alternatives",
];

fn under_system_bind(p: &Path) -> bool {
    SYSTEM_BINDS.iter().any(|s| p.starts_with(s))
}

/// Mount flags a bind remount must repeat: the kernel refuses (EPERM) a
/// remount that would CLEAR nosuid/nodev/noexec/atime flags the bind
/// inherited from its source filesystem (e.g. a host /tmp mounted
/// nosuid,nodev), so read them off the mounted dst and carry them over.
fn carried_mount_flags(dst: &Path) -> MsFlags {
    let mut flags = MsFlags::empty();
    if let Ok(st) = statvfs(dst) {
        let f = st.flags();
        if f.contains(FsFlags::ST_NOSUID) {
            flags |= MsFlags::MS_NOSUID;
        }
        if f.contains(FsFlags::ST_NODEV) {
            flags |= MsFlags::MS_NODEV;
        }
        if f.contains(FsFlags::ST_NOEXEC) {
            flags |= MsFlags::MS_NOEXEC;
        }
        if f.contains(FsFlags::ST_NOATIME) {
            flags |= MsFlags::MS_NOATIME;
        }
        if f.contains(FsFlags::ST_NODIRATIME) {
            flags |= MsFlags::MS_NODIRATIME;
        }
        // ST_RELATIME by raw bit (0x1000, kernel ABI): the kernel sets it in
        // f_flag on every libc, but neither nix's FsFlags nor the libc crate
        // defines the constant on musl, and the release wheel builds musl.
        const ST_RELATIME_BIT: libc::c_ulong = 0x1000;
        if f.bits() & ST_RELATIME_BIT != 0 {
            flags |= MsFlags::MS_RELATIME;
        }
    }
    flags
}

/// One mountinfo path field, with the kernel's octal escapes (\040 space, \011
/// tab, \012 newline, \134 backslash) resolved. Bytes throughout: a path is not
/// required to be UTF-8, and a lossy decode would name a different file.
fn mountinfo_path(field: &[u8]) -> PathBuf {
    use std::ffi::OsString;
    use std::os::unix::ffi::OsStringExt;
    let mut out: Vec<u8> = Vec::with_capacity(field.len());
    let mut i = 0;
    while i < field.len() {
        let esc = field
            .get(i + 1..i + 4)
            .filter(|d| d.iter().all(|c| (b'0'..=b'7').contains(c)));
        match esc {
            Some(digits) if field[i] == b'\\' => {
                let val = digits
                    .iter()
                    .fold(0u32, |acc, d| acc * 8 + u32::from(d - b'0'));
                out.push(val as u8);
                i += 4;
            }
            _ => {
                out.push(field[i]);
                i += 1;
            }
        }
    }
    PathBuf::from(OsString::from_vec(out))
}

/// The mount points nested UNDER `dir` in this mount namespace that a path
/// still resolves to. `dir` is canonicalized first: mountinfo prints resolved
/// paths, so a symlinked component (a symlinked /tmp) would miss every prefix
/// match and silently skip a floor.
fn submounts_under(dir: &Path) -> io::Result<Vec<PathBuf>> {
    let dir = dir.canonicalize().unwrap_or_else(|_| dir.to_path_buf());
    let raw = fs::read("/proc/self/mountinfo")?;
    Ok(live_submounts(&raw, &dir))
}

/// The mountinfo entries under `dir` that are still reachable by path.
///
/// Lines appear in mount order, and a mount KEEPS its line after a later
/// mount covers its mount point or an ancestor of it. A remount addressed
/// through such a shadowed path lands on whatever the path resolves to now
/// (EINVAL where that is not a mount point). A recursive covering bind lists
/// its carried copies on their own later lines, so dropping shadowed entries
/// loses nothing.
fn live_submounts(raw: &[u8], dir: &Path) -> Vec<PathBuf> {
    let mounts: Vec<PathBuf> = raw
        .split(|b| *b == b'\n')
        .filter_map(|line| line.split(|b| *b == b' ').nth(4))
        .map(mountinfo_path)
        .collect();
    let mut out = Vec::new();
    for (i, mp) in mounts.iter().enumerate() {
        if mp == dir || !mp.starts_with(dir) {
            continue;
        }
        if mounts[i + 1..].iter().any(|later| mp.starts_with(later)) {
            continue; // shadowed: a later mount covers it or an ancestor
        }
        out.push(mp.clone());
    }
    out
}

/// Repeat a bind's own `flags` on every mount NESTED under `dst`.
///
/// MS_REC is silently IGNORED on MS_REMOUNT -- recursive attribute changes need
/// mount_setattr(AT_RECURSIVE) -- so a recursive bind carries its whole subtree
/// in and then makes only its TOP mount read-only. Probed: a tmpfs nested inside
/// a read-only grant arrived `rw,relatime` and a jailed command wrote a file
/// that was still on the host afterwards.
///
/// Hand-written on purpose, not a missing mount_setattr(AT_RECURSIVE) call:
/// mount_setattr is Linux 5.12+, and `strict` admits Landlock-less kernels
/// older than that, where these binds are the ONLY filesystem boundary. The
/// loop must work exactly on the kernels the syscall is absent from.
fn floor_submounts(dst: &Path, flags: MsFlags) -> io::Result<()> {
    for mp in submounts_under(dst)? {
        mount(
            Some(""),
            &mp,
            Some(""),
            MsFlags::MS_BIND | MsFlags::MS_REMOUNT | flags | carried_mount_flags(&mp),
            Some(""),
        )
        .map_err(io_err)?;
    }
    Ok(())
}

/// Read-only plus the nosuid/nodev floor: what every bind the child may not
/// write carries, top mount and submounts alike.
const RO_FLOOR: MsFlags = MsFlags::MS_RDONLY
    .union(MsFlags::MS_NOSUID)
    .union(MsFlags::MS_NODEV);
/// The floor alone, for the binds that are writable by design.
const RW_FLOOR: MsFlags = MsFlags::MS_NOSUID.union(MsFlags::MS_NODEV);

fn setup_rootfs(policy: &Policy, real_uid: u32) -> io::Result<()> {
    // Per-uid: a shared path in a sticky /tmp is a cross-user denial of
    // service, since whoever creates it first owns it and every other user's
    // jail then fails closed forever. *real_uid* is the caller's, captured
    // before entering the user namespace (strict maps it to itself).
    //
    // Shared by every launcher this user runs: each mounts its own tmpfs over
    // it in its own mount namespace, so they see different roots through the
    // same mount point, and create_dir_all is idempotent. Nothing needs
    // clearing: everything a launcher writes goes inside the tmpfs, so the
    // directory underneath is empty by construction, and a remove_dir_all here
    // would delete the tree a concurrent launcher is still building.
    let new_root = PathBuf::from(format!("/tmp/agent6-jail-root-{real_uid}"));
    fs::create_dir_all(&new_root).map_err(|e| {
        io::Error::other(format!(
            "jail root {} is unusable: {e} (remove it, or point TMPDIR elsewhere)",
            new_root.display()
        ))
    })?;
    // Make new_root a mount point (pivot_root requirement).
    // The floor here too. The /dev nodes bound underneath keep their own flags
    // (a bind's options are its own), so they stay usable.
    mount(
        Some(new_root.as_path()),
        &new_root,
        Some("tmpfs"),
        MsFlags::MS_NOSUID | MsFlags::MS_NODEV,
        Some("size=64m"),
    )
    .map_err(io_err)?;

    for dir in ["proc", "tmp", "dev", "etc", "home", "root"] {
        fs::create_dir_all(new_root.join(dir))?;
    }
    // Read-only bind mounts for system dirs.
    for src in SYSTEM_BINDS {
        if Path::new(src).exists() {
            let dst = new_root.join(src.trim_start_matches('/'));
            fs::create_dir_all(&dst)?;
            mount(
                Some(Path::new(src)),
                &dst,
                Some(""),
                MsFlags::MS_BIND | MsFlags::MS_REC,
                Some(""),
            )
            .map_err(io_err)?;
            // Remount read-only, with the same nosuid/nodev floor every other
            // bind here carries: these are the mounts most likely to HOLD a
            // setuid binary or a device node, so leaving them out made the
            // "unconditional floor" the comments claim conditional.
            mount(
                Some(""),
                &dst,
                Some(""),
                MsFlags::MS_BIND | MsFlags::MS_REMOUNT | RO_FLOOR | carried_mount_flags(&dst),
                Some(""),
            )
            .map_err(io_err)?;
            floor_submounts(&dst, RO_FLOOR)?;
        }
    }
    // /tmp -> tmpfs. strict's default HOME lives here (/tmp/agent6-home, so
    // toolchain caches have a writable root), and go's build cache alone needs
    // several hundred MB for stdlib artifacts -- at 64m `go test` died ENOSPC
    // and models burned budgets fighting the sandbox. 1g is a hard ceiling on
    // RAM-backed pages, not an allocation; a run that needs none uses none.
    // nosuid/nodev, the floor the binds carry. NOT noexec: HOME lives here, so
    // toolchains legitimately place and run helpers under it, and a child that
    // can already execute from the workspace gains nothing from being stopped
    // here -- that would be theatre, at the cost of real builds.
    mount(
        Some(""),
        &new_root.join("tmp"),
        Some("tmpfs"),
        MsFlags::MS_NOSUID | MsFlags::MS_NODEV,
        Some("size=1g"),
    )
    .map_err(io_err)?;
    // The default HOME itself: a missing home breaks `cd ~` and `git config
    // --global` before any cache dir gets created under it. A persistent HOME
    // ([sandbox].home = "cache") arrives as an extra_rw_paths bind instead.
    fs::create_dir_all(new_root.join("tmp/agent6-home")).map_err(io_err)?;
    // Operator tool dirs (uv etc.) at their REAL locations, RO. After /tmp so a dir
    // that happens to live under it is not shadowed by the fresh tmpfs. Best-effort:
    // a dir that fails to mount just leaves that tool unreachable rather than aborting
    // the run; dispatch only passes dirs OUTSIDE the system mounts above.
    for src in &policy.tool_paths {
        if !src.exists() {
            continue;
        }
        let dst = new_root.join(src.strip_prefix("/").unwrap_or(src));
        if fs::create_dir_all(&dst).is_err() {
            continue;
        }
        if mount(
            Some(src.as_path()),
            &dst,
            Some(""),
            MsFlags::MS_BIND | MsFlags::MS_REC,
            Some(""),
        )
        .is_err()
        {
            continue;
        }
        // Best-effort means UNREACHABLE, never writable: if the read-only
        // remount fails the bind is already up, so drop it rather than leave the
        // operator's tool dir writable from inside the jail (a jailed command
        // could then rewrite a binary that later runs on the HOST). A submount
        // that cannot be secured leaves the same hole, so it drops the bind too.
        if mount(
            Some(""),
            &dst,
            Some(""),
            MsFlags::MS_BIND | MsFlags::MS_REMOUNT | RO_FLOOR | carried_mount_flags(&dst),
            Some(""),
        )
        .is_err()
            || floor_submounts(&dst, RO_FLOOR).is_err()
        {
            // If it can be neither made read-only nor dropped, the invariant
            // cannot hold: refuse the run rather than proceed with a writable
            // host tool dir inside the jail.
            umount2(&dst, MntFlags::MNT_DETACH).map_err(io_err)?;
        }
    }
    // /proc — bind from host /proc (it's still our PID namespace's view from outside,
    // but inside the new pid ns we'll mount a fresh one below).
    fs::create_dir_all(new_root.join("proc"))?;
    // /dev minimal — just /dev/null /dev/zero /dev/urandom etc.
    // /dev/tty is intentionally OMITTED: child commands inherit pipes (not a
    // tty) and giving them access to the controlling terminal would let a
    // misbehaving (or LLM-orchestrated) child write escape sequences that
    // affect the agent's host terminal.
    for dev in ["null", "zero", "urandom", "random", "full"] {
        let src = PathBuf::from(format!("/dev/{dev}"));
        if !src.exists() {
            continue;
        }
        let dst = new_root.join(format!("dev/{dev}"));
        fs::File::create(&dst)?;
        mount(
            Some(src.as_path()),
            &dst,
            Some(""),
            MsFlags::MS_BIND,
            Some(""),
        )
        .map_err(io_err)?;
    }
    // Operator-granted device nodes ([sandbox].extra_device_paths): bound
    // exactly like the builtin five above (plain bind, no nodev floor -- the
    // floor exists to stop a node smuggled through a PATH grant; these ARE
    // the operator's explicit node grant). The node must be a character or
    // block device on the host: anything else refuses loudly rather than
    // widening by surprise.
    for src in &policy.extra_device_paths {
        let meta = fs::symlink_metadata(src).map_err(|e| {
            io::Error::other(format!(
                "extra_device path {} is not present on the host: {e}",
                src.display()
            ))
        })?;
        let ft = meta.file_type();
        if !(ft.is_char_device() || ft.is_block_device()) {
            return Err(io::Error::other(format!(
                "extra_device path {} is not a character or block device",
                src.display()
            )));
        }
        let dst = new_root.join(src.strip_prefix("/").unwrap_or(src));
        fs::create_dir_all(dst.parent().unwrap_or(Path::new("/")))?;
        fs::File::create(&dst)?;
        mount(
            Some(src.as_path()),
            &dst,
            Some(""),
            MsFlags::MS_BIND,
            Some(""),
        )
        .map_err(io_err)?;
    }
    // /dev/shm: a private tmpfs, like /tmp. POSIX shared memory is ordinary
    // for real toolchains -- a headless chromium aborts outright without it --
    // and it exposes nothing, being this mount namespace's own.
    let shm = new_root.join("dev/shm");
    fs::create_dir_all(&shm)?;
    mount(
        Some("tmpfs"),
        &shm,
        Some("tmpfs"),
        MsFlags::MS_NOSUID | MsFlags::MS_NODEV,
        Some("mode=1777"),
    )
    .map_err(io_err)?;
    // Bind the cwd RW, at its REAL path. Every mount in this root is at the
    // path it has outside -- tool dirs, operator grants, and the workspace
    // alike -- so a path means the same thing on both sides of the boundary.
    // A remapped /workspace made an absolute host path (the one the model
    // sees, the one an MCP server is configured with, the one in a server's
    // own argv) fail with ENOENT inside, and made every hidden path need
    // masking twice: once at its real location and once through the alias.
    let cwd_in = new_root.join(policy.cwd.strip_prefix("/").unwrap_or(&policy.cwd));
    fs::create_dir_all(&cwd_in)?;
    mount(
        Some(policy.cwd.as_path()),
        &cwd_in,
        Some(""),
        MsFlags::MS_BIND | MsFlags::MS_REC,
        Some(""),
    )
    .map_err(io_err)?;
    // nosuid/nodev on the workspace, the same unconditional floor the protect
    // and RO binds carry. A bind mount cannot set these in one step, so this is
    // a second remount. Under `sudo agent6 --allow-root` the uid_map makes the
    // jailed child real root and chmod is not seccomp-denied, so without nosuid
    // it could leave a setuid-root binary on the HOST inode -- surviving the run
    // and handing local root to anyone who runs it.
    mount(
        None::<&Path>,
        &cwd_in,
        Some(""),
        MsFlags::MS_BIND | MsFlags::MS_REMOUNT | RW_FLOOR | carried_mount_flags(&cwd_in),
        Some(""),
    )
    .map_err(io_err)?;
    // A workspace that CONTAINS a mount (a vendored tree on its own filesystem,
    // a bind) carried it in with the recursive bind above; the floor is the
    // whole point there too.
    floor_submounts(&cwd_in, RW_FLOOR)?;
    // Re-bind each protect path RO on top of the workspace mount. Subdirs
    // and individual files are both supported; non-existent entries are
    // skipped silently so a project without (e.g.) a `.git` dir is not
    // a fatal config error.
    for src in &policy.extra_protect_paths {
        // Canonicalize so a symlink at .git/ can't trick us into bind-mounting
        // a target outside cwd into the jail. If the path doesn't exist yet,
        // canonicalize fails — skip it (the protect-path is a no-op anyway).
        let canon_src = match src.canonicalize() {
            Ok(p) => p,
            Err(_) => continue,
        };
        let canon_cwd = policy
            .cwd
            .canonicalize()
            .unwrap_or_else(|_| policy.cwd.clone());
        // Reject paths outside cwd defensively (Python side filters them too,
        // but the launcher is its own trust boundary).
        let rel = match canon_src.strip_prefix(&canon_cwd) {
            Ok(r) => r,
            Err(_) => {
                eprintln!(
                    "agent6-jail: skipping protect_path {} (canonical {} not under cwd {})",
                    src.display(),
                    canon_src.display(),
                    canon_cwd.display(),
                );
                continue;
            }
        };
        let target = cwd_in.join(rel);
        // Ensure the mount point exists inside our new rootfs (it should,
        // via the cwd bind, but be defensive for first-run-on-fresh-repo).
        if canon_src.is_dir() {
            fs::create_dir_all(&target)?;
        } else if let Some(parent) = target.parent() {
            fs::create_dir_all(parent)?;
            if !target.exists() {
                fs::File::create(&target)?;
            }
        }
        // Bind the canonical host path onto the target inside the new rootfs,
        // then remount read-only. Binding from the host path (rather than
        // self-binding inside the new mount) avoids EPERM on kernels that
        // refuse re-binding paths already covered by a recursive parent
        // bind in a user namespace. RECURSIVE: a plain bind of a subtree
        // holding a locked child mount (`.git/objects` on its own bind) is
        // EINVAL in a user namespace, and the nested mount's content stays
        // visible rather than covered by an empty directory.
        mount(
            Some(canon_src.as_path()),
            &target,
            Some(""),
            MsFlags::MS_BIND | MsFlags::MS_REC,
            Some(""),
        )
        .map_err(|e| {
            io::Error::other(format!(
                "protect bind {} -> {}: {e}",
                canon_src.display(),
                target.display()
            ))
        })?;
        mount(
            Some(""),
            &target,
            Some(""),
            // MS_NOSUID | MS_NODEV are an unconditional floor (secure-by-default
            // even if the statvfs in carried_mount_flags fails). carried_mount_flags
            // adds the source's OTHER locked flags (noexec, the atime family): in
            // a user namespace the copied source mount's flags are locked, and a
            // remount that tries to CLEAR one is refused EPERM — so a workspace on
            // a noexec filesystem (a common CIS hardening for /home) failed every
            // jailed command under the default strict + protect_git config. Its
            // two sibling remounts (tool_paths, extra_ro_paths) already carry it;
            // this one was missed.
            MsFlags::MS_BIND | MsFlags::MS_REMOUNT | RO_FLOOR | carried_mount_flags(&target),
            Some(""),
        )
        .map_err(|e| io::Error::other(format!("protect remount-ro {}: {e}", target.display())))?;
        // The workspace's recursive bind may have carried a mount in BELOW this
        // protect path; this bind is not recursive, so that one is still
        // writable at its own path unless it is floored here.
        floor_submounts(&target, RO_FLOOR)?;
    }
    // Extra RO paths, at their REAL locations (matching tool_paths and the
    // hardened, and the documented contract: a granted toolchain works
    // via its own absolute paths and shebangs). A grant under a system bind is
    // redundant (already visible read+exec) and skipped. Failures are LOUD:
    // the operator listed the path, so a broken grant must not pass silently.
    for src in &policy.extra_ro_paths {
        if !src.exists() || under_system_bind(src) {
            continue;
        }
        let dst = new_root.join(src.strip_prefix("/").unwrap_or(src));
        fs::create_dir_all(dst.parent().unwrap_or(Path::new("/")))?;
        if src.is_dir() {
            fs::create_dir_all(&dst)?;
        } else {
            fs::File::create(&dst)?;
        }
        mount(
            Some(src.as_path()),
            &dst,
            Some(""),
            MsFlags::MS_BIND | MsFlags::MS_REC,
            Some(""),
        )
        .map_err(io_err)?;
        mount(
            Some(""),
            &dst,
            Some(""),
            MsFlags::MS_BIND | MsFlags::MS_REMOUNT | RO_FLOOR | carried_mount_flags(&dst),
            Some(""),
        )
        .map_err(io_err)?;
        floor_submounts(&dst, RO_FLOOR)?;
    }
    // Extra RW paths, at their REAL locations. Under a system bind the write
    // grant cannot be honored (the covering mount is read-only): refuse loudly
    // rather than mount a dead path.
    for src in &policy.extra_rw_paths {
        if !src.exists() {
            continue;
        }
        if under_system_bind(src) {
            return Err(io::Error::other(format!(
                "extra_rw path {} sits under a read-only system mount",
                src.display()
            )));
        }
        let dst = new_root.join(src.strip_prefix("/").unwrap_or(src));
        fs::create_dir_all(dst.parent().unwrap_or(Path::new("/")))?;
        if src.is_dir() {
            fs::create_dir_all(&dst)?;
        } else {
            fs::File::create(&dst)?;
        }
        mount(
            Some(src.as_path()),
            &dst,
            Some(""),
            MsFlags::MS_BIND | MsFlags::MS_REC,
            Some(""),
        )
        .map_err(io_err)?;
        // The nosuid/nodev floor every other bind here carries. This one is
        // writable by design, which is the case that most needs it: without the
        // remount a setuid binary or device node placed in a granted dir keeps
        // those bits inside the jail.
        mount(
            Some(""),
            &dst,
            Some(""),
            MsFlags::MS_BIND | MsFlags::MS_REMOUNT | RW_FLOOR | carried_mount_flags(&dst),
            Some(""),
        )
        .map_err(io_err)?;
        floor_submounts(&dst, RW_FLOOR)?;
    }

    mask_hidden_paths(policy, &new_root)?;

    // pivot_root into new_root.
    let put_old = new_root.join(".old_root");
    fs::create_dir_all(&put_old)?;
    pivot_root(&new_root, &put_old).map_err(io_err)?;
    chdir("/").map_err(io_err)?;

    // We are in the forked child (called from main after fork), which IS in the
    // new PID namespace. Mount a fresh /proc so the child sees only its own
    // PID namespace. ORDER MATTERS: this must happen while /.old_root is still
    // attached -- the kernel permits a userns proc mount only when the mount
    // namespace already contains a fully-visible proc instance, and the host's
    // /.old_root/proc is that instance. Mounting after the detach fails EPERM
    // ("mount too revealing" rule) and left /proc EMPTY, which breaks any tool
    // that reads /proc/self (observed: go cannot resolve GOROOT via
    // /proc/self/exe). If the kernel still refuses, log and continue with an
    // empty /proc; we deliberately do NOT bind-mount the host /proc as a
    // fallback because that would expose every host PID and /proc/sys tunable.
    let proc_target = Path::new("/proc");
    if let Err(e) = mount(
        Some("proc"),
        proc_target,
        Some("proc"),
        MsFlags::MS_NOSUID | MsFlags::MS_NODEV | MsFlags::MS_NOEXEC,
        Some(""),
    ) {
        // Naming the cost, not just the cause: an empty /proc is safe (never the
        // outer one, which would leak processes across the PID namespace) but
        // the dynamic loader resolves $ORIGIN through /proc/self/exe, so a
        // relocatable toolchain -- a downloaded python, node or conda -- fails
        // to start with "cannot open shared object file" and nothing to link it
        // to this. Measured inside rootless podman, which refuses the mount
        // because its own /proc is partly masked.
        eprintln!(
            "[agent6-jail] warning: fresh /proc mount failed ({e}); /proc will be empty inside \
             the jail, so binaries that find their libraries relative to themselves ($ORIGIN) \
             will not start"
        );
    }

    umount2(Path::new("/.old_root"), MntFlags::MNT_DETACH).map_err(io_err)?;
    fs::remove_dir("/.old_root").ok();

    chdir(&policy.cwd).map_err(io_err)?;
    Ok(())
}

/// Mask the policy's hidden paths out of the assembled root: an empty RO
/// tmpfs over a dir, a /dev/null bind over a file. Runs after EVERY bind so
/// no grant (workspace, extras, tools) exposes a hidden path from above --
/// mount order means the mask always wins. Policy grants BENEATH a hidden
/// root are then re-bound through the mask at their REAL path (the contract
/// every grant already has): default-deny with the policy's own explicit
/// holes, e.g. a machine's data dir under the state dir. Masks stay
/// writable until the re-binds land, then remount read-only (non-recursive,
/// so a re-bound RW hole keeps its writability).
fn mask_hidden_paths(policy: &Policy, new_root: &Path) -> io::Result<()> {
    let mut masked_dirs: Vec<PathBuf> = Vec::new();
    // One target per hidden path: every mount is at its real location, so
    // there is no second door to close.
    for hp in &policy.hide_paths {
        let dst = new_root.join(hp.strip_prefix("/").unwrap_or(hp));
        // Not in the assembled view: nothing mounted it, already invisible.
        let meta = match fs::symlink_metadata(&dst) {
            Ok(m) => m,
            Err(_) => continue,
        };
        if meta.is_dir() {
            mount(
                Some("tmpfs"),
                &dst,
                Some("tmpfs"),
                MsFlags::MS_NOSUID | MsFlags::MS_NODEV | MsFlags::MS_NOEXEC,
                Some("mode=0755"),
            )
            .map_err(|e| io::Error::other(format!("mask {}: {e}", dst.display())))?;
            masked_dirs.push(dst);
        } else {
            mount(
                Some(Path::new("/dev/null")),
                &dst,
                Some(""),
                MsFlags::MS_BIND,
                Some(""),
            )
            .map_err(|e| io::Error::other(format!("mask {}: {e}", dst.display())))?;
        }
    }
    // Re-open the policy's own grants beneath hidden roots. Failures are LOUD:
    // the policy promised these paths, and a mask must not silently eat one.
    for (src, ro) in policy
        .extra_rw_paths
        .iter()
        .map(|p| (p, false))
        .chain(policy.extra_ro_paths.iter().map(|p| (p, true)))
    {
        if !policy
            .hide_paths
            .iter()
            .any(|hp| src.starts_with(hp) && src != hp)
        {
            continue;
        }
        if !src.exists() {
            continue;
        }
        let dst = new_root.join(src.strip_prefix("/").unwrap_or(src));
        fs::create_dir_all(dst.parent().unwrap_or(Path::new("/")))?;
        if src.is_dir() {
            fs::create_dir_all(&dst)?;
        } else {
            fs::File::create(&dst)?;
        }
        mount(
            Some(src.as_path()),
            &dst,
            Some(""),
            MsFlags::MS_BIND | MsFlags::MS_REC,
            Some(""),
        )
        .map_err(|e| io::Error::other(format!("re-bind through mask {}: {e}", dst.display())))?;
        let floor = if ro { RO_FLOOR } else { RW_FLOOR };
        mount(
            Some(""),
            &dst,
            Some(""),
            MsFlags::MS_BIND | MsFlags::MS_REMOUNT | floor | carried_mount_flags(&dst),
            Some(""),
        )
        .map_err(io_err)?;
        floor_submounts(&dst, floor)?;
    }
    for dst in &masked_dirs {
        // Plain (non-bind, non-recursive) remount: the tmpfs itself goes RO
        // while a re-bound grant mounted inside keeps its own flags.
        mount(
            Some(""),
            dst.as_path(),
            Some(""),
            MsFlags::MS_REMOUNT
                | MsFlags::MS_RDONLY
                | MsFlags::MS_NOSUID
                | MsFlags::MS_NODEV
                | MsFlags::MS_NOEXEC,
            Some(""),
        )
        .map_err(|e| io::Error::other(format!("mask remount-ro {}: {e}", dst.display())))?;
    }
    Ok(())
}

/// The Landlock access sets both isolation levels build their rules from.
struct LandlockSets {
    /// Every right the ruleset restricts: the full ABI::V3 set. A right left
    /// out here is not restricted at all, so it is never narrower than `all`.
    handled: BitFlags<AccessFs>,
    /// The writable grant (cwd, /tmp, extra_rw_paths): everything but the
    /// creation of device nodes, which no rule grants.
    all: BitFlags<AccessFs>,
    /// Read plus execute (`from_read` carries Execute): the system paths and
    /// operator tool dirs a spawned binary runs from, and the read half of
    /// the device and protected-path grants.
    read: BitFlags<AccessFs>,
    /// Read without execute, for /proc.
    read_noexec: BitFlags<AccessFs>,
}

fn landlock_sets() -> LandlockSets {
    let handled = AccessFs::from_all(ABI::V3);
    let read = AccessFs::from_read(ABI::V3);
    LandlockSets {
        handled,
        all: handled & !AccessFs::MakeChar & !AccessFs::MakeBlock,
        read,
        read_noexec: read & !AccessFs::Execute,
    }
}

fn apply_landlock_strict(policy: &Policy) -> io::Result<()> {
    // Strict runs inside the pivoted rootfs; the cwd bind (at its real path)
    // and /tmp (a fresh private tmpfs, see setup_rootfs) are writable, and
    // /usr /bin /lib /lib64 /etc /dev are read-only bind mounts.
    // ABI::V3 (not V1/V2): V2 added LANDLOCK_ACCESS_FS_REFER, without which
    // EVERY cross-directory rename/hardlink fails EXDEV -- that breaks `cargo`
    // (hardlinks build artifacts between target/ subdirs), `mv` across dirs, and
    // similar tools even inside fully-writable paths. V3 added
    // LANDLOCK_ACCESS_FS_TRUNCATE, which MUST be handled: Landlock leaves any
    // right the ruleset does not handle unrestricted, so a V2 handled set never
    // checks truncate(2)/ftruncate(2) and a jailed child could zero any file it
    // can name outside its write grants (the run state dir, ~/.ssh) with no
    // write access at all. Granting these on rw paths only lets the child act
    // within hierarchies it can already write; the crate's best-effort mode
    // drops TRUNCATE (and REFER) on kernels too old to know them, matching
    // sandbox/landlock.py, which masks the same bit below its ABI.
    // MakeChar/MakeBlock are HANDLED and granted by no rule: a right the ruleset
    // handles and never grants is denied, while an unhandled one is left
    // unrestricted by the kernel. Device-node creation is denied here by
    // Landlock, by the user namespace (the child holds no CAP_MKNOD in the
    // initial one) and by the seccomp mode rule below.
    let sets = landlock_sets();
    let ruleset = Ruleset::default()
        .handle_access(sets.handled)
        .map_err(|e| io::Error::other(format!("handle_access: {e}")))?
        .create()
        .map_err(|e| io::Error::other(format!("create ruleset: {e}")))?;
    let mut ruleset = ruleset;
    if let Ok(fd) = PathFd::new(policy.cwd.as_path()) {
        ruleset = ruleset
            .add_rule(PathBeneath::new(fd, sets.all))
            .map_err(|e| io::Error::other(format!("rule cwd: {e}")))?;
    }
    // /tmp is a fresh private tmpfs in this jail's own mount namespace (mounted
    // in setup_rootfs), discarded when the jail exits. Grant it RW so toolchain
    // caches that key off $HOME or TMPDIR work (go-build, cargo, pip/uv); the
    // tmpfs is isolated, so RW here cannot reach the host. Mirrors the hardened
    // isolation level, which already grants /tmp RW.
    // /dev/shm is the same deal: this jail's own tmpfs, and POSIX shared memory
    // is ordinary for real toolchains (a headless chromium aborts without it).
    // Mounting it without granting it would have been a mount that does nothing.
    for writable in ["/tmp", "/dev/shm"] {
        if let Ok(fd) = PathFd::new(writable) {
            ruleset = ruleset
                .add_rule(PathBeneath::new(fd, sets.all))
                .map_err(|e| io::Error::other(format!("rule {writable}: {e}")))?;
        }
    }
    // /proc is the jail's OWN freshly-mounted procfs (private PID namespace,
    // see setup_rootfs), so reading it reveals only jail-local processes and
    // read-only kernel views -- the same exposure every container runtime
    // grants. Read WITHOUT execute. Without this rule every /proc read dies
    // EACCES and toolchains fail in confusing ways: go resolves GOROOT via
    // /proc/self/exe (observed: go 1.26 "cannot find GOROOT", the model then
    // rewrites verify.sh to fight the sandbox), python reads /proc/cpuinfo,
    // ps needs the listing.
    if let Ok(fd) = PathFd::new("/proc") {
        ruleset = ruleset
            .add_rule(PathBeneath::new(fd, sets.read_noexec))
            .map_err(|e| io::Error::other(format!("rule /proc: {e}")))?;
    }
    for ro in ["/usr", "/bin", "/sbin", "/lib", "/lib64", "/etc", "/dev"] {
        if let Ok(fd) = PathFd::new(ro) {
            ruleset = ruleset
                .add_rule(PathBeneath::new(fd, sets.read))
                .map_err(|e| io::Error::other(format!("rule {ro}: {e}")))?;
        }
    }
    // Operator tool dirs (mounted at real locations in setup_rootfs): read+exec.
    for tp in &policy.tool_paths {
        if let Ok(fd) = PathFd::new(tp) {
            ruleset = ruleset
                .add_rule(PathBeneath::new(fd, sets.read))
                .map_err(|e| io::Error::other(format!("rule tool: {e}")))?;
        }
    }
    // Grant WriteFile on the harmless sink devices. /dev/null and /dev/full
    // are bind-mounted from the host (see setup_rootfs) and pytest's logging
    // plugin opens /dev/null O_WRONLY|O_APPEND when log_file is configured,
    // which would otherwise EACCES under the /dev read-only rule above and
    // surface as INTERNALERROR. WriteFile on these specific inodes does not
    // grant create/symlink/unlink and cannot be used to escape the jail.
    for dev in ["null", "zero", "full"] {
        let p = format!("/dev/{dev}");
        if let Ok(fd) = PathFd::new(&p) {
            ruleset = ruleset
                .add_rule(PathBeneath::new(fd, AccessFs::WriteFile))
                .map_err(|e| io::Error::other(format!("rule {p}: {e}")))?;
        }
    }
    // Operator-granted device nodes: read+write on the node itself, so an
    // open(O_RDWR) succeeds; WriteFile on a device inode grants no
    // create/unlink and ioctl is outside Landlock's file scope.
    for dev in &policy.extra_device_paths {
        if let Ok(fd) = PathFd::new(dev) {
            ruleset = ruleset
                .add_rule(PathBeneath::new(fd, sets.read | AccessFs::WriteFile))
                .map_err(|e| io::Error::other(format!("rule dev {}: {e}", dev.display())))?;
        }
    }
    // Extra paths are bind-mounted by setup_rootfs at their REAL locations.
    // Without a matching Landlock rule the child would get EACCES on them
    // despite the mount, so grant the access here too. Paths that didn't
    // exist on the host were skipped at mount time, so PathFd::new simply
    // fails and is ignored; a ro grant under a system bind got no mount of
    // its own, but the rule on the real path applies to the covering bind's
    // content just the same.
    for ro in &policy.extra_ro_paths {
        if let Ok(fd) = PathFd::new(ro) {
            ruleset = ruleset
                .add_rule(PathBeneath::new(fd, sets.read))
                .map_err(|e| io::Error::other(format!("rule ro {}: {e}", ro.display())))?;
        }
    }
    for rw in &policy.extra_rw_paths {
        if let Ok(fd) = PathFd::new(rw) {
            ruleset = ruleset
                .add_rule(PathBeneath::new(fd, sets.all))
                .map_err(|e| io::Error::other(format!("rule rw {}: {e}", rw.display())))?;
        }
    }
    // Deliberately no NotEnforced check (contrast apply_landlock_hardened):
    // strict's boundary is namespaces + the pivoted rootfs with MS_RDONLY
    // binds + seccomp; Landlock is defense-in-depth, and the isolation contract
    // (docs/security.md) admits strict on Landlock-less kernels. The gap is
    // not silent: `warn_sandbox_gaps` says so once at run entry. Warning here
    // instead would land on every spawn's stderr, inside model-visible tool
    // output.
    ruleset
        .restrict_self()
        .map_err(|e| io::Error::other(format!("restrict_self: {e}")))?;
    Ok(())
}

/// Grant recursive RW under `dir` on everything EXCEPT the protect paths,
/// descending only into the directories that actually contain one.
///
/// Comparing each entry to the protect set by equality let a protect path
/// NESTED below an entry (a machine bundle at `ops/deploy.asm.toml`, whose
/// top-level entry is `ops/`) be covered by that ancestor's recursive grant, so
/// the jailed child could rewrite the machine's own spec and scripts. Landlock
/// rules combine permissively, so an ancestor grant always wins -- the sibling
/// `rw_paths` loop refuses such an ancestor outright for this reason. Here the
/// carve-out is kept precise instead: descending grants the siblings normally
/// and leaves only the protected leaf ungranted, which matches `strict`, where
/// each protect path is re-bound read-only at its real nested location.
fn grant_rw_carved(
    mut ruleset: RulesetCreated,
    dir: &Path,
    protect_set: &std::collections::HashSet<PathBuf>,
    canon_cwd: &Path,
    access_all: BitFlags<AccessFs>,
) -> io::Result<RulesetCreated> {
    let entries = match fs::read_dir(dir) {
        Ok(it) => it,
        Err(e) => {
            // Fail CLOSED, don't fail the launch: a directory we cannot
            // enumerate simply gets no RW grants for its children (they stay
            // read-only), matching the skip-and-warn of the outside-cwd path
            // below. Propagating the error made one unreadable directory
            // anywhere below cwd abort every jailed command.
            eprintln!(
                "agent6-jail: hardened: no rw grants under {} (read_dir: {e})",
                dir.display()
            );
            return Ok(ruleset);
        }
    };
    for entry in entries.flatten() {
        let p = entry.path();
        let canon = p.canonicalize().unwrap_or_else(|_| p.clone());
        // Skip an entry that IS a protect path, or (via a symlink) resolves AT
        // OR BELOW one. `PathFd::new` follows symlinks and Landlock attaches to
        // the resolved inode, so granting RW on such an entry would make the
        // protected file writable by its own direct path -- the exact bypass a
        // `ops/link -> scripts/step.py` symlink gave. `starts_with` covers the
        // protect path itself (equality) and the descendant case; the ANCESTOR
        // direction -- this entry CONTAINS a protect path -- is the descent
        // below. `contains(&p)` keeps the pre-canonicalize form for an entry
        // that could not be canonicalized.
        if protect_set.contains(&p) || protect_set.iter().any(|prot| canon.starts_with(prot)) {
            continue;
        }
        // A top-level symlink whose real target escapes cwd would otherwise
        // get a recursive RW rule on that outside inode (PathFd::new follows
        // symlinks; Landlock attaches to the resolved inode), letting the
        // child write outside the workspace and defeating cwd confinement
        // under hardened. Skip any entry that does not resolve
        // inside cwd. Mirrors the strip_prefix(cwd) check in setup_rootfs.
        if !canon.starts_with(canon_cwd) {
            eprintln!(
                "agent6-jail: hardened: skipping rw grant on {} (resolves outside cwd to {})",
                p.display(),
                canon.display()
            );
            continue;
        }
        // The skip above dropped every entry at or below a protect path, so a
        // hit here means this directory CONTAINS a protect path (is an ancestor
        // of one) -- descend and grant its non-protected children.
        if canon.is_dir() && protect_set.iter().any(|prot| prot.starts_with(&canon)) {
            ruleset = grant_rw_carved(ruleset, &p, protect_set, canon_cwd, access_all)?;
            continue;
        }
        if let Ok(fd) = PathFd::new(&p) {
            ruleset = ruleset
                .add_rule(PathBeneath::new(fd, access_all))
                .map_err(|e| io::Error::other(format!("rule rw {}: {e}", p.display())))?;
        }
    }
    Ok(ruleset)
}

fn apply_landlock_hardened(policy: &Policy) -> io::Result<()> {
    // Hardened runs in the real filesystem. We protect the host by
    // listing exactly the paths the child may read or write — its own cwd
    // (read+write), the extra_rw_paths, /tmp (write), and the system dirs
    // (read+execute only).
    // V3 handled set (TRUNCATE included): see apply_landlock_strict for why
    // handling truncate matters. It matters more here -- hardened has no
    // mount-namespace RO binds to fall back on, so Landlock is the only thing
    // standing between a jailed child and truncating files outside its grants.
    // Device nodes: MakeChar/MakeBlock handled and granted nowhere, so Landlock
    // denies their creation (see apply_landlock_strict). It matters most HERE:
    // hardened has no user namespace, so under sudo the child is real root
    // with CAP_MKNOD and no MS_NODEV bind, and this and the seccomp mode rule
    // below are the two locks left.
    let sets = landlock_sets();
    let ruleset = Ruleset::default()
        .handle_access(sets.handled)
        .map_err(|e| io::Error::other(format!("handle_access: {e}")))?
        .create()
        .map_err(|e| io::Error::other(format!("create ruleset: {e}")))?;
    let mut ruleset = ruleset;

    // protect_paths: in hardened we cannot do a bind-remount-RO (no mount
    // namespace). Instead, we DON'T grant RW on cwd as a whole. We grant R
    // on cwd recursively (so .git etc. stay readable), then enumerate
    // cwd's top-level entries and grant RW only to the ones that are not
    // in the protect set. Landlock rules are purely additive within a
    // single ruleset, so if no rule grants W on a path, writes to it are
    // denied — that's what gives us the read-only carve-out.
    //
    // Limitation (deliberate, secure side of the tradeoff): a directory the
    // walk had to DESCEND into because it CONTAINS a protect path gets RW rules
    // on its non-protected children but NOT on the directory itself, so
    // creating/unlinking a NEW entry directly in it is denied (overwriting an
    // existing child still works). Granting the dir its own create/remove
    // rights would, under Landlock's recursive rules, also grant them over the
    // protected subtree -- reopening a delete-then-recreate bypass of the
    // protected file -- so we keep it denied. A machine writing an output
    // beside its bundle therefore fails on hardened but works on strict (which
    // re-binds each protect path RO instead of carving). Same for new
    // top-level entries at the root of cwd.
    let has_protect = !policy.extra_protect_paths.is_empty();
    let protect_set: std::collections::HashSet<PathBuf> = policy
        .extra_protect_paths
        .iter()
        .filter_map(|p| p.canonicalize().ok().or_else(|| Some(p.clone())))
        .collect();

    if has_protect {
        // R on cwd recursively, so protected paths remain readable.
        if let Ok(fd) = PathFd::new(&policy.cwd) {
            ruleset = ruleset
                .add_rule(PathBeneath::new(fd, sets.read))
                .map_err(|e| {
                    io::Error::other(format!("rule r cwd {}: {e}", policy.cwd.display()))
                })?;
        }
        // RW only on non-protected top-level entries.
        let canon_cwd = policy
            .cwd
            .canonicalize()
            .unwrap_or_else(|_| policy.cwd.clone());
        ruleset = grant_rw_carved(ruleset, &policy.cwd, &protect_set, &canon_cwd, sets.all)?;
    } else {
        // No protect set: original behavior, RW on cwd as a whole.
        if let Ok(fd) = PathFd::new(&policy.cwd) {
            ruleset = ruleset
                .add_rule(PathBeneath::new(fd, sets.all))
                .map_err(|e| {
                    io::Error::other(format!("rule rw cwd {}: {e}", policy.cwd.display()))
                })?;
        }
    }

    // Read+write: /tmp and any explicitly granted rw paths. NOT /dev/shm, which
    // strict grants: there it is the jail's own tmpfs, here it would be the
    // HOST's, shared with every other process of this user. A hardened jail
    // cannot run a browser regardless (no /proc grant, and granting that would
    // hand a command the agent's own environ), so the compatibility it would
    // buy is not there to buy.
    let mut rw_paths: Vec<PathBuf> = vec![PathBuf::from("/tmp")];
    for p in &policy.extra_rw_paths {
        rw_paths.push(p.clone());
    }
    // Operator-granted device nodes: hardened has no mount namespace, so the
    // grant IS this Landlock rule -- read+write on the host node (open O_RDWR;
    // ioctl on the opened fd is outside Landlock's file scope). The same
    // char/block check strict's rootfs applies: anything else refuses loudly.
    for dev in &policy.extra_device_paths {
        let meta = fs::symlink_metadata(dev).map_err(|e| {
            io::Error::other(format!(
                "extra_device path {} is not present on the host: {e}",
                dev.display()
            ))
        })?;
        let ft = meta.file_type();
        if !(ft.is_char_device() || ft.is_block_device()) {
            return Err(io::Error::other(format!(
                "extra_device path {} is not a character or block device",
                dev.display()
            )));
        }
        if let Ok(fd) = PathFd::new(dev) {
            ruleset = ruleset
                .add_rule(PathBeneath::new(fd, sets.read | AccessFs::WriteFile))
                .map_err(|e| io::Error::other(format!("rule dev {}: {e}", dev.display())))?;
        }
    }
    for p in &rw_paths {
        // Skip any rw_path that would shadow a protect_path: Landlock combines
        // permissively, so a blanket RW grant on an ancestor of a protected path
        // defeats the carve-out. Compare the CANONICAL rw_path (protect_set is
        // canonical too) so a symlinked rw_path resolving to a protect ancestor
        // can't slip past -- PathFd::new below follows the symlink.
        let canon_p = p.canonicalize().unwrap_or_else(|_| p.clone());
        if has_protect && protect_set.iter().any(|prot| prot.starts_with(&canon_p)) {
            eprintln!(
                "agent6-jail: hardened: skipping rw grant on {} (would shadow a protect_path)",
                p.display()
            );
            continue;
        }
        if let Ok(fd) = PathFd::new(p) {
            ruleset = ruleset
                .add_rule(PathBeneath::new(fd, sets.all))
                .map_err(|e| io::Error::other(format!("rule rw {}: {e}", p.display())))?;
        }
    }

    // Read+execute: system dirs the child needs to load libraries / spawn binaries.
    let mut ro_paths: Vec<PathBuf> = vec![
        PathBuf::from("/usr"),
        PathBuf::from("/bin"),
        PathBuf::from("/sbin"),
        PathBuf::from("/lib"),
        PathBuf::from("/lib64"),
        PathBuf::from("/etc"),
        PathBuf::from("/dev"),
    ];
    for p in &policy.extra_ro_paths {
        ro_paths.push(p.clone());
    }
    // Operator tool dirs (uv etc.): read+exec at their real host paths (hardened
    // runs in the real filesystem, so no bind mount is needed, only the grant).
    for p in &policy.tool_paths {
        ro_paths.push(p.clone());
    }
    for p in &ro_paths {
        if let Ok(fd) = PathFd::new(p) {
            ruleset = ruleset
                .add_rule(PathBeneath::new(fd, sets.read))
                .map_err(|e| io::Error::other(format!("rule ro {}: {e}", p.display())))?;
        }
    }
    // Same sink-device carve-out as strict; see comment there.
    for dev in ["null", "zero", "full"] {
        let p = format!("/dev/{dev}");
        if let Ok(fd) = PathFd::new(&p) {
            ruleset = ruleset
                .add_rule(PathBeneath::new(fd, AccessFs::WriteFile))
                .map_err(|e| io::Error::other(format!("rule {p}: {e}")))?;
        }
    }

    // Fail CLOSED. Hardened has no mount namespace, so Landlock is the ONLY
    // filesystem boundary. The landlock crate defaults to BestEffort, so on a
    // kernel without Landlock restrict_self() returns Ok(NotEnforced) rather
    // than erroring — enforcing nothing. Refuse instead of running a child
    // with zero confinement behind the "hardened" label. (PartiallyEnforced on
    // an older ABI is real, if reduced, confinement and is accepted.) The
    // Python isolation resolution already refuses hardened without Landlock; this
    // is the launcher's own boundary check, not a substitute for it.
    let status = ruleset
        .restrict_self()
        .map_err(|e| io::Error::other(format!("restrict_self: {e}")))?;
    if status.ruleset == RulesetStatus::NotEnforced {
        return Err(io::Error::other(
            "hardened isolation requires Landlock, but the kernel enforced no \
             ruleset (Landlock unavailable); refusing to run unconfined",
        ));
    }
    Ok(())
}

fn apply_seccomp() -> io::Result<()> {
    // PR_SET_NO_NEW_PRIVS
    let rc = unsafe { libc::prctl(libc::PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) };
    if rc != 0 {
        return Err(io::Error::last_os_error());
    }
    let arch = if cfg!(target_arch = "x86_64") {
        TargetArch::x86_64
    } else if cfg!(target_arch = "aarch64") {
        TargetArch::aarch64
    } else {
        // Fail closed: strict and hardened promise a seccomp filter, and a
        // silent no-op would report isolation that was never installed. The
        // supported set is mirrored by sandbox/detect.py (auto degrades,
        // explicit refuses, both before this point on agent6's own path).
        return Err(io::Error::other(
            "no seccomp filter for this architecture (filters exist for \
             x86_64 and aarch64); set isolation = \"none\" to run without one",
        ));
    };
    // Default-allow with explicit deny of the worst offenders. We are inside
    // user-ns + landlock already; seccomp here is a third layer to block obvious
    // foot-guns: ptrace, mount, setns, unshare, kexec, bpf, perf, keyctl, etc.
    // libc (0.2.x) doesn't define SYS_kexec_file_load for the musl aarch64 target
    // even though the syscall exists there (arm64 #294), so name it explicitly —
    // otherwise the sandbox would silently stop blocking a new-kernel load on arm64.
    #[cfg(target_arch = "aarch64")]
    const SYS_KEXEC_FILE_LOAD: i64 = 294;
    #[cfg(not(target_arch = "aarch64"))]
    const SYS_KEXEC_FILE_LOAD: i64 = libc::SYS_kexec_file_load;
    let denied: &[i64] = &[
        libc::SYS_ptrace,
        libc::SYS_process_vm_readv,
        libc::SYS_process_vm_writev,
        libc::SYS_kcmp,
        // pidfd_getfd steals an already-open fd out of another process's table
        // -- the pidfd-era way to do what ptrace's fd access did, without
        // calling ptrace. It is gated only by ptrace_may_access, the SAME check
        // that gates process_vm_readv/writev and kcmp above; those are denied
        // regardless, so this belongs beside them. Left off, a jailed command's
        // one barrier to lifting a live fd (a provider socket, a secrets or
        // transcript fd) out of the agent under `hardened` -- no user namespace
        // there -- is that check plus the host's yama tunable, which is 0 on
        // many distros and in many containers. seccomp is the layer that must
        // not depend on either. pidfd_open (the handle) is harmless alone and
        // stays allowed; this is the reach.
        libc::SYS_pidfd_getfd,
        // io_uring submits file/network ops through a shared ring that kernel
        // worker threads execute -- so those ops never appear as syscalls and
        // this filter never sees them. Landlock still hooks the ring's FS ops
        // (worker threads inherit the domain on a current kernel), so it is not
        // a live FS bypass here; but that is a kernel-version property, and the
        // seccomp layer must not lean on it. Denying io_uring_setup is a
        // COMPLETE block, not a leaky one: enter/register need a ring this call
        // creates, so none can exist. Matches podman/docker, whose default
        // allow-list profiles omit io_uring entirely. Nothing a build/test does
        // needs it; a program that genuinely must have it is the signal to run
        // unsandboxed (isolation = "none"), the same coarse escape hatch a
        // container user reaches for with seccomp=unconfined.
        libc::SYS_io_uring_setup,
        // userfaultfd hands a process a fd that stalls page faults on demand --
        // the race-window primitive most kernel UAF/heap exploits use to win.
        // Not an escape by itself. Blocked here rather than left to the
        // vm.unprivileged_userfaultfd=0 sysctl, which is a host tunable this
        // layer must not depend on (same reasoning as pidfd_getfd + io_uring).
        libc::SYS_userfaultfd,
        libc::SYS_mount,
        // umount2 is the unmount call on every arch here. There is no second
        // spelling to add: the legacy `umount` is number 22 of the I386 table,
        // and 22 in the 64-bit one is `pipe`. A foreign-arch caller cannot reach
        // the i386 table either -- seccompiler's prologue kills on arch mismatch.
        libc::SYS_umount2,
        libc::SYS_pivot_root,
        // Modern mount API (new_mount_api, Linux 5.2+). A strict jailed child
        // is userns-root with CAP_SYS_ADMIN over its own mount namespace and
        // never drops caps, so without these it could mount_setattr(2) away the
        // MOUNT_ATTR_RDONLY on the .git protect bind (or open_tree+move_mount to
        // relocate it) and defeat protect_git. Classic mount(2) is already
        // denied above; these complete the coverage.
        libc::SYS_mount_setattr,
        libc::SYS_open_tree,
        libc::SYS_move_mount,
        libc::SYS_fsopen,
        libc::SYS_fsconfig,
        libc::SYS_fsmount,
        libc::SYS_fspick,
        libc::SYS_setns,
        libc::SYS_unshare,
        libc::SYS_kexec_load,
        SYS_KEXEC_FILE_LOAD,
        libc::SYS_bpf,
        libc::SYS_perf_event_open,
        libc::SYS_keyctl,
        libc::SYS_add_key,
        libc::SYS_request_key,
        libc::SYS_init_module,
        libc::SYS_finit_module,
        libc::SYS_delete_module,
        libc::SYS_reboot,
        libc::SYS_swapon,
        libc::SYS_swapoff,
        // The whole clock-setting family, not three of its four names:
        // clock_adjtime sets the offset just as adjtimex does.
        libc::SYS_settimeofday,
        libc::SYS_adjtimex,
        libc::SYS_clock_settime,
        libc::SYS_clock_adjtime,
    ];
    // Deliberately NOT here: clone/clone3 with CLONE_NEWUSER|CLONE_NEWNS. A
    // nested namespace grants no new access -- the whole mount family above is
    // denied, the Landlock domain and this filter are both inherited and
    // irrevocable, and mknod checks caps against the init userns. seccomp can't
    // read clone3's flags (behind a struct pointer) to filter them, and denying
    // clone3 outright would break glibc/Go spawning (they fall back to clone
    // only on ENOSYS). This deny-list is defense-in-depth over namespaces +
    // Landlock, not a boundary; see docs/security.md.
    let mut rules: std::collections::BTreeMap<i64, Vec<seccompiler::SeccompRule>> =
        denied.iter().map(|s| (*s, vec![])).collect();
    // Setuid/setgid bits, denied by ARGUMENT rather than by syscall: an ordinary
    // chmod is normal work, but setting S_ISUID/S_ISGID writes a bit that lands
    // on the HOST inode and outlives the jail. Under `sudo agent6 --allow-root`
    // the uid_map makes the child real root, so that bit would be a setuid-root
    // binary sitting in the operator's workspace -- local root for anyone who
    // runs it. mount nosuid does not help: it stops the JAIL honouring the bit,
    // not the host. The mode argument is a scalar, so seccomp can test it.
    // One rule per bit: rules for a syscall are OR-ed, conditions within a rule
    // are AND-ed, so a single MaskedEq(0o6000) would only catch a mode that set
    // BOTH -- `chmod 4755` sets just S_ISUID.
    // fchmodat2 (Linux 6.6+) supersedes fchmodat and takes the mode in the same
    // argument, so without it the newest kernels hand back the exact write this
    // filter exists to stop. libc 0.2.x does not define SYS_fchmodat2 for every
    // target, so name it -- 452 on both x86_64 and aarch64.
    const SYS_FCHMODAT2: i64 = 452;
    // arm64 has no bare chmod(2), only fchmod and the *at forms -- the same
    // shape as the mknod pair below, whose guard this one was missing.
    #[cfg(target_arch = "aarch64")]
    let chmod_syscalls: [(i64, u8); 3] = [
        (libc::SYS_fchmod, 1),
        (libc::SYS_fchmodat, 2),
        (SYS_FCHMODAT2, 2),
    ];
    #[cfg(not(target_arch = "aarch64"))]
    let chmod_syscalls: [(i64, u8); 4] = [
        (libc::SYS_chmod, 1),
        (libc::SYS_fchmod, 1),
        (libc::SYS_fchmodat, 2),
        (SYS_FCHMODAT2, 2),
    ];
    // The CREATE family carries the same bits to the same host inode: creat(2)
    // and mknod(2) take a mode outright, open/openat take one whenever the flags
    // ask for a new file. Their mode is a vararg the kernel ignores without
    // O_CREAT/O_TMPFILE (uninitialized stack on an ordinary open), so those
    // entries name the flags argument too -- conditions within a rule are AND-ed.
    // openat2 keeps its mode behind a struct pointer, out of seccomp's reach,
    // the same limit clone3 has above.
    // The syscall, the argument carrying the mode, and for open/openat the
    // flags argument with the flag that makes that mode real.
    type SetidWrite = (i64, u8, Option<(u8, i32)>);
    let mut setid_syscalls: Vec<SetidWrite> = vec![
        (libc::SYS_mknodat, 2, None),
        (libc::SYS_openat, 3, Some((2, libc::O_CREAT))),
        (libc::SYS_openat, 3, Some((2, libc::O_TMPFILE))),
    ];
    #[cfg(not(target_arch = "aarch64"))]
    setid_syscalls.extend([
        (libc::SYS_creat, 1, None),
        (libc::SYS_mknod, 1, None),
        (libc::SYS_open, 2, Some((1, libc::O_CREAT))),
        (libc::SYS_open, 2, Some((1, libc::O_TMPFILE))),
    ]);
    setid_syscalls.extend(chmod_syscalls.map(|(syscall, mode_arg)| (syscall, mode_arg, None)));
    for (syscall, mode_arg, flags) in setid_syscalls {
        for bit in [0o4000_u64, 0o2000] {
            let mut conds = vec![seccompiler::SeccompCondition::new(
                mode_arg,
                seccompiler::SeccompCmpArgLen::Dword,
                seccompiler::SeccompCmpOp::MaskedEq(bit),
                bit,
            )
            .map_err(|e| io::Error::other(format!("seccomp cond: {e}")))?];
            if let Some((flags_arg, wanted)) = flags {
                conds.push(
                    seccompiler::SeccompCondition::new(
                        flags_arg,
                        seccompiler::SeccompCmpArgLen::Dword,
                        seccompiler::SeccompCmpOp::MaskedEq(wanted as u64),
                        wanted as u64,
                    )
                    .map_err(|e| io::Error::other(format!("seccomp cond: {e}")))?,
                );
            }
            rules.entry(syscall).or_default().push(
                seccompiler::SeccompRule::new(conds)
                    .map_err(|e| io::Error::other(format!("seccomp rule: {e}")))?,
            );
        }
    }
    // Device nodes. Blocked by MODE, not outright: `mkfifo` and socket nodes go
    // through the same syscalls and builds legitimately use them. Under
    // `sudo agent6` on a profile with no user namespace the child is real root
    // with CAP_MKNOD and no MS_NODEV bind, so a block device for the host disk
    // in its own workspace reads and writes raw sectors past every path rule.
    // Landlock denies it too (MakeChar/MakeBlock handled and granted nowhere),
    // so hardened has two locks and strict three (the user namespace as well).
    // One rule per type: rules are OR-ed, conditions
    // AND-ed, so testing both types in one rule would match neither.
    // arm64 has no bare mknod(2); its mknodat carries the mode one arg later.
    #[cfg(target_arch = "aarch64")]
    let mknod_syscalls: [(i64, u8); 1] = [(libc::SYS_mknodat, 2)];
    #[cfg(not(target_arch = "aarch64"))]
    let mknod_syscalls: [(i64, u8); 2] = [(libc::SYS_mknod, 1), (libc::SYS_mknodat, 2)];
    for (syscall, mode_arg) in mknod_syscalls {
        let mut per_type = Vec::new();
        for kind in [libc::S_IFBLK as u64, libc::S_IFCHR as u64] {
            per_type.push(
                seccompiler::SeccompRule::new(vec![seccompiler::SeccompCondition::new(
                    mode_arg,
                    seccompiler::SeccompCmpArgLen::Dword,
                    seccompiler::SeccompCmpOp::MaskedEq(libc::S_IFMT as u64),
                    kind,
                )
                .map_err(|e| io::Error::other(format!("seccomp cond: {e}")))?])
                .map_err(|e| io::Error::other(format!("seccomp rule: {e}")))?,
            );
        }
        rules.entry(syscall).or_default().append(&mut per_type);
    }
    let filter = SeccompFilter::new(
        rules,
        SeccompAction::Allow,                     // default
        SeccompAction::Errno(libc::EPERM as u32), // matched (denied)
        arch,
    )
    .map_err(|e| io::Error::other(format!("seccomp build: {e}")))?;
    let program: BpfProgram = filter
        .try_into()
        .map_err(|e| io::Error::other(format!("seccomp compile: {e}")))?;
    seccompiler::apply_filter(&program)
        .map_err(|e| io::Error::other(format!("seccomp apply: {e}")))?;
    Ok(())
}

/// What executing ONE command needs, apart from the rootfs/confinement setup.
///
/// Separate from `Policy` because the setup happens once per launcher while a
/// command is a single request against it.
struct ChildSpec<'a> {
    argv: &'a [String],
    env: &'a [(String, String)],
    memory_limit_mb: u64,
    /// Wall-clock kill. <= 0 disables it: a command is bounded by the model or
    /// the operator stopping it, not by a number that cannot know whether a
    /// 20-minute build is stuck or working.
    timeout_s: f64,
    /// Hand the command back after this long instead of waiting on it. <= 0, or
    /// an empty `log_dir`, means wait (a one-shot launcher has nobody to hand
    /// it back TO).
    checkin_s: f64,
    /// Where a handed-back command's output continues, as a log.
    log_dir: &'a str,
}

impl Policy {
    fn child_spec(&self) -> ChildSpec<'_> {
        ChildSpec {
            argv: &self.argv,
            env: &self.env,
            memory_limit_mb: self.memory_limit_mb,
            timeout_s: self.timeout_s,
            // A one-shot launcher exits with its command, so there would be
            // nobody left to poll a handed-back one.
            checkin_s: 0.0,
            log_dir: "",
        }
    }
}

/// The child's argv, env, cwd and process group -- everything except how its
/// stdio is wired. Shared so a served/one-shot command and an exec-mode
/// long-lived server are launched identically; a second spelling here is how
/// one of them would quietly stop getting a hardening.
///
/// `tie_to_parent` sets PR_SET_PDEATHSIG(SIGKILL) in the child before exec,
/// so a jailed command cannot outlive a dead launcher.
/// The one caller that passes false is `spawn_detached`: a backgrounded
/// command is DESIGNED to outlive its request (the roster owns its lifetime,
/// and the escapee sweep covers the crash case).
fn build_command(spec: &ChildSpec<'_>, cwd: &Path, tie_to_parent: bool) -> Command {
    let mut cmd = Command::new(&spec.argv[0]);
    cmd.args(&spec.argv[1..]);
    cmd.env_clear();
    for (k, v) in spec.env {
        cmd.env(k, v);
    }
    // Minimal PATH so basic tools work inside the jail; the policy may extend it so
    // operator-installed tools outside /usr/bin (e.g. /usr/local/bin, ~/.local/bin)
    // resolve. Policy env is operator-side input.
    if !spec.env.iter().any(|(k, _)| k == "PATH") {
        cmd.env("PATH", "/usr/bin:/bin");
    }
    // HOME default; the policy may override it (e.g. a writable /tmp path so
    // toolchain caches like go-build work). Policy env is operator-side input.
    if !spec.env.iter().any(|(k, _)| k == "HOME") {
        cmd.env("HOME", "/home");
    }
    cmd.current_dir(cwd);
    // Put the child in its own process group (pgid == its pid) so we can kill
    // the whole tree on timeout/exit. Otherwise a backgrounded grandchild that
    // inherited our stdout/stderr write-end keeps the pipe open and the reader
    // threads' read_to_string() never sees EOF — hanging the launcher.
    cmd.process_group(0);
    if tie_to_parent {
        // Between our fork and the prctl the parent can die, leaving the child
        // re-parented and the signal never delivered: re-check the parent and
        // refuse to exec a command nobody owns. getppid/prctl are async-signal
        // -safe (pre_exec contract), and prctl is not in the seccomp denylist
        // (PR_SET_DUMPABLE already runs behind the same filter).
        let parent = std::process::id() as libc::pid_t;
        unsafe {
            cmd.pre_exec(move || {
                if libc::prctl(libc::PR_SET_PDEATHSIG, libc::SIGKILL, 0, 0, 0) != 0 {
                    return Err(io::Error::last_os_error());
                }
                if libc::getppid() != parent {
                    return Err(io::Error::other(
                        "parent died before the death-signal tie landed",
                    ));
                }
                Ok(())
            });
        }
    }
    cmd
}

/// Capability drop + the optional RLIMIT_DATA cap, in the child before exec.
fn apply_child_limits(cmd: &mut Command, memory_limit_mb: u64) {
    let mem_bytes: libc::rlim_t = if memory_limit_mb > 0 {
        (memory_limit_mb as libc::rlim_t).saturating_mul(1024 * 1024)
    } else {
        0
    };
    unsafe {
        cmd.pre_exec(move || {
            drop_capabilities()?;
            if mem_bytes == 0 {
                return Ok(());
            }
            // Clamp to the inherited hard limit: lowering is always
            // permitted, and if the operator's shell already set a
            // stricter hard cap the stricter value wins (never EPERM).
            let mut cur = libc::rlimit {
                rlim_cur: 0,
                rlim_max: 0,
            };
            if libc::getrlimit(libc::RLIMIT_DATA, &mut cur) != 0 {
                return Err(io::Error::last_os_error());
            }
            let cap = mem_bytes.min(cur.rlim_max);
            let lim = libc::rlimit {
                rlim_cur: cap,
                rlim_max: cap,
            };
            if libc::setrlimit(libc::RLIMIT_DATA, &lim) != 0 {
                return Err(io::Error::last_os_error());
            }
            Ok(())
        });
    }
}

/// `mode = "exec"`: run a LONG-LIVED child on our own stdio and exit with its
/// status. For a process agent6 talks to rather than collects -- an MCP server
/// needs its JSON-RPC pipe for the whole session, which the capture path
/// (pipes drained into a JSON result) cannot provide.
///
/// No timeout: the pipe closing is the lifetime. We stay alive as the child's
/// parent (PID 1 of its namespace under strict) so it is reaped and so the
/// namespace outlives no one.
fn run_child_exec(spec: &ChildSpec<'_>, cwd: &Path) -> io::Result<()> {
    if spec.argv.is_empty() {
        return Err(io::Error::new(io::ErrorKind::InvalidInput, "empty argv"));
    }
    let mut cmd = build_command(spec, cwd, true);
    apply_child_limits(&mut cmd, spec.memory_limit_mb);
    // Inherited, not piped: these three fds are the caller's pipe to the
    // server. fork/exec does not touch the fd table, so the JSON-RPC stream
    // survives the namespaces, the pivoted root, Landlock and seccomp.
    cmd.stdin(Stdio::inherit());
    cmd.stdout(Stdio::inherit());
    cmd.stderr(Stdio::inherit());
    let mut child = cmd.spawn()?;
    let status = child.wait()?;
    std::process::exit(status.code().unwrap_or(128 + libc::SIGKILL));
}

fn run_child(
    spec: &ChildSpec<'_>,
    cwd: &Path,
    interrupt_fd: Option<RawFd>,
) -> io::Result<Option<i32>> {
    if spec.argv.is_empty() {
        return Err(io::Error::new(io::ErrorKind::InvalidInput, "empty argv"));
    }
    let _ = CString::new(spec.argv[0].as_bytes());
    let _ = OsStr::new(""); // silence unused import on some targets

    let mut cmd = build_command(spec, cwd, true);
    apply_child_limits(&mut cmd, spec.memory_limit_mb);
    cmd.stdin(Stdio::null());
    cmd.stdout(Stdio::piped());
    cmd.stderr(Stdio::piped());

    let mut child = cmd.spawn()?;
    let child_pid = child.id() as i32; // == pgid, since process_group(0)
    let timeout = Duration::from_secs_f64(spec.timeout_s.max(0.0));
    let checkin = Duration::from_secs_f64(spec.checkin_s.max(0.0));
    let converts = spec.checkin_s > 0.0 && !spec.log_dir.is_empty();
    let start = std::time::Instant::now();

    // Drain stdout/stderr on background threads so a child that writes more
    // than the pipe buffer (~64KB) before exiting cannot deadlock us. The
    // earlier implementation only read pipes AFTER try_wait() returned Some,
    // which would deadlock the child on its write() while we were stuck
    // polling try_wait() forever.
    let stdout_pipe = child.stdout.take().expect("stdout piped");
    let stderr_pipe = child.stderr.take().expect("stderr piped");
    // ONE capture for both pipes: the arrival order is what a merged log needs,
    // and it cannot be recovered from two separate buffers later.
    let captured = Arc::new(Mutex::new(Capture::default()));
    let drained = Arc::new(AtomicUsize::new(0));
    for (pipe, tag) in [
        (Box::new(stdout_pipe) as Box<dyn Read + Send>, Stream::Out),
        (Box::new(stderr_pipe), Stream::Err),
    ] {
        let done = Arc::clone(&drained);
        let sink = Arc::clone(&captured);
        std::thread::spawn(move || {
            read_capped_into(pipe, tag, &sink);
            done.fetch_add(1, Ordering::SeqCst);
        });
    }

    // Poll the direct child WITHOUT reaping it (WNOWAIT leaves the zombie), so
    // child_pid (== pgid) cannot be recycled by the kernel before we tear the
    // group down below. try_wait() would reap on exit, forcing the old
    // two-path dance (killpg-before-wait on timeout, skip-killpg on normal
    // exit) — and skipping the killpg on normal exit leaked backgrounded
    // grandchildren that held the stdout/stderr pipe open, hanging the reader
    // joins and turning a successful command into a false rc=124 timeout.
    // waitid (not waitpid): WNOWAIT is a waitid(2) flag — glibc's waitpid
    // rejects it with EINVAL. WEXITED selects exited children; WNOHANG polls.
    let child_wait = Pid::from_raw(child_pid);
    let wait_flags = WaitPidFlag::WEXITED | WaitPidFlag::WNOHANG | WaitPidFlag::WNOWAIT;
    let mut timed_out = false;
    // Anything but StillAlive leaves the loop: Exited/Signaled (peeked, not
    // reaped) or an unexpected wait error both proceed to the unified teardown
    // + real reap below.
    while let Ok(WaitStatus::StillAlive) = waitid(Id::Pid(child_wait), wait_flags) {
        if spec.timeout_s > 0.0 && start.elapsed() > timeout {
            timed_out = true;
            break;
        }
        if converts && (start.elapsed() > checkin || interrupt_requested(interrupt_fd)) {
            // Hand the command back rather than kill it: whether a long command
            // is stuck or working is a judgement, so it goes to whoever can
            // make one. The client can ask for that judgement EARLY (a byte on
            // the interrupt pipe) when the operator has pressed Stop -- the
            // same hand-back, just not at the end of a 15-minute check-in.
            // Render BEFORE spilling -- the answer carries the output so far,
            // and spilling clears the retained capture.
            let (out, err) = {
                let buf = captured.lock().unwrap_or_else(|e| e.into_inner());
                (buf.render(Some(Stream::Out)), buf.render(Some(Stream::Err)))
            };
            let log = converted_log_path(spec.log_dir, child_pid);
            let file = File::options()
                .write(true)
                .create_new(true)
                .custom_flags(libc::O_NOFOLLOW)
                .open(&log)?;
            captured
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .spill_to(file)?;
            let answer = serde_json::json!({
                "backgrounded": true,
                "pid": child_pid,
                "log": log,
                "stdout": out,
                "stderr": err,
            });
            let mut stream = io::stdout().lock();
            writeln!(stream, "{answer}")?;
            stream.flush()?;
            // No killpg, no reap: the drain threads keep appending to the log
            // and the caller polls the pid through `status`/`stop`.
            std::mem::forget(child);
            return Ok(Some(child_pid));
        }
        std::thread::sleep(Duration::from_millis(50));
    }
    // Kill the whole process group BEFORE reaping — one path for both normal
    // exit and timeout. The direct child is still an unreaped zombie (normal
    // exit) or alive (timeout), so child_pid == pgid is unambiguously ours,
    // with no pid-reuse hazard. This tears down any backgrounded grandchild
    // that inherited the stdout/stderr write-end, so read_to_end() gets EOF
    // and the reader joins finish; it also means a command's process group
    // does not outlive the command (a backgrounded daemon is torn down, not
    // leaked — strict's PID namespace already enforces this; hardened now
    // matches).
    unsafe {
        libc::killpg(child_pid, libc::SIGKILL);
    }
    // Reap the direct child for its real exit code (the group SIGKILL above
    // already terminated it on the timeout path).
    let status = child.wait()?;
    let returncode: i32 = if timed_out {
        124
    } else {
        status
            .code()
            .unwrap_or_else(|| status.signal().map(|s| 128 + s).unwrap_or(-1))
    };
    // Take what the readers have, never wait on them: a grandchild that left
    // the process group (`setsid`, and every double-fork daemonize) survives
    // the killpg above and holds the write end, so the pipe never reaches EOF.
    // Joining here hung the launcher, and with it every later command in the
    // run -- one tool call, no error, no diagnostic. The threads keep draining
    // into the buffers; the process exit closes their fds.
    let drain_deadline = std::time::Instant::now() + Duration::from_millis(CAPTURE_DRAIN_MS);
    while drained.load(Ordering::SeqCst) < 2 && std::time::Instant::now() < drain_deadline {
        std::thread::sleep(Duration::from_millis(20));
    }
    let held_open = drained.load(Ordering::SeqCst) < 2;
    let (stdout, mut stderr) = {
        let buf = captured.lock().unwrap_or_else(|e| e.into_inner());
        (buf.render(Some(Stream::Out)), buf.render(Some(Stream::Err)))
    };
    if held_open {
        stderr.push_str(concat!(
            "\n[agent6-jail] output capture ended early: a process outside the",
            " command's group still holds its pipe",
        ));
    }
    if timed_out {
        stderr.push_str("\n[agent6-jail] timeout");
    }

    let result = serde_json::json!({
        "returncode": returncode,
        "stdout": stdout,
        "stderr": stderr,
    });
    let mut out = io::stdout().lock();
    writeln!(out, "{result}")?;
    Ok(None)
}

/// Where a handed-back command's output continues.
///
/// Named for the pid, which no command can predict, and created with O_EXCL |
/// O_NOFOLLOW below: the log root is granted read-write to every command in the
/// run, so a predictable name could be pre-planted as a symlink.
fn converted_log_path(log_dir: &str, pid: i32) -> String {
    format!("{log_dir}/converted-{pid}.log")
}

/// Start a command and answer at once with its pid, leaving it running in this
/// session's namespaces: no wait, and no process-group kill, so a server
/// survives the request that started it. Under a PID namespace that namespace
/// is the bound -- closing the session's stdin takes everything in it down;
/// without one the pid is swept by `sweep_backgrounded` at EOF.
/// Version 3 of the capset ABI: two 32-bit words per set.
const LINUX_CAPABILITY_VERSION_3: u32 = 0x2008_0522;

#[repr(C)]
struct CapHeader {
    version: u32,
    pid: i32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct CapData {
    effective: u32,
    permitted: u32,
    inheritable: u32,
}

/// Make this process's /proc entry unreadable to the commands it runs.
///
/// The launcher answers every request on its stdout and, under strict, is PID 1
/// of the jail's own PID namespace. A command that opens `/proc/1/fd/1` writes
/// the line the agent reads as a command result: it can hand itself an exit
/// code, and the real answer then serves the NEXT request, so a verify gate
/// reads the result of a command the model chose. Landlock does not cover it
/// (a pipe reopened through /proc/<pid>/fd is exempt) and neither does the
/// seccomp ptrace deny (reaching another /proc entry is a permission check,
/// not that syscall). With dumpable 0 the check is ptrace_may_access, which
/// demands CAP_SYS_PTRACE in this user namespace -- which every child gives up
/// in `drop_capabilities`. Both halves are needed: the command runs as the
/// launcher's own uid, which passes the same-uid check on its own.
fn hide_from_children() -> io::Result<()> {
    if unsafe { libc::prctl(libc::PR_SET_DUMPABLE, 0, 0, 0, 0) } != 0 {
        return Err(io::Error::last_os_error());
    }
    Ok(())
}

/// Drop every capability, in the child, between fork and exec.
///
/// The launcher created the jail's user namespace, so it holds a full
/// capability set over everything that namespace owns -- CAP_SYS_PTRACE over
/// itself included -- and a forked child inherits it. Nothing agent6 runs
/// needs a capability: the files it touches are owned by the command's own
/// uid. With uid 0 inside (`--allow-root`) exec would keep the bounding set as
/// the new permitted set, so the child drops everything itself.
/// Async-signal-safe: raw syscalls only.
///
/// # Safety
/// Called inside `pre_exec`, so it must not allocate or take locks.
unsafe fn drop_capabilities() -> io::Result<()> {
    // The bounding set first: what leaves it cannot be regained, file
    // capabilities included. EINVAL means we walked past this kernel's last.
    // EPERM means no CAP_SETPCAP; the bounding set stays, but the capset
    // below still clears effective/permitted/inheritable -- dropping one's
    // own sets needs no capability, so a profile that RETAINS caps without
    // CAP_SETPCAP (an elevated uid, a caps-granting container) still hands
    // the command an empty set rather than keeping CAP_SYS_PTRACE live.
    for cap in 0..=63 {
        if unsafe { libc::prctl(libc::PR_CAPBSET_DROP, cap as libc::c_ulong, 0, 0, 0) } != 0 {
            match io::Error::last_os_error().raw_os_error() {
                Some(libc::EINVAL) | Some(libc::EPERM) => break,
                _ => return Err(io::Error::last_os_error()),
            }
        }
    }
    let header = CapHeader {
        version: LINUX_CAPABILITY_VERSION_3,
        pid: 0,
    };
    let empty = [CapData {
        effective: 0,
        permitted: 0,
        inheritable: 0,
    }; 2];
    let rc = unsafe {
        libc::syscall(
            libc::SYS_capset,
            &header as *const CapHeader,
            empty.as_ptr(),
        )
    };
    if rc != 0 {
        return Err(io::Error::last_os_error());
    }
    Ok(())
}

/// The exit code a wait status carries, in the shape run_child reports it.
/// None means the child is not gone (stopped, traced), never "exited 0".
fn wait_code(status: WaitStatus) -> Option<i32> {
    match status {
        WaitStatus::Exited(_, code) => Some(code),
        WaitStatus::Signaled(_, sig, _) => Some(128 + sig as i32),
        _ => None,
    }
}

/// Whether a backgrounded command is still running, reaping it once it is not
/// so its pid cannot be recycled under a later request.
fn answer_status(pid: i32) -> io::Result<()> {
    let running = serde_json::json!({"running": true, "returncode": null});
    let result = match waitpid(Pid::from_raw(pid), Some(WaitPidFlag::WNOHANG)) {
        Ok(WaitStatus::StillAlive) => running,
        Ok(status) => match wait_code(status) {
            Some(code) => serde_json::json!({"running": false, "returncode": code}),
            None => running,
        },
        Err(e) => serde_json::json!({"running": false, "returncode": null, "error": e.to_string()}),
    };
    println!("{result}");
    io::stdout().flush()
}

/// Kill a backgrounded command's group and reap it. Idempotent. Bounded: an
/// unkillable process must not hold the run's only jail process.
fn answer_stop(pid: i32) -> io::Result<()> {
    // The group, not the pid: spawn_detached gives each one its own, so a
    // server's children go down with it.
    unsafe { libc::killpg(pid, libc::SIGKILL) };
    let deadline = std::time::Instant::now() + Duration::from_secs(5);
    let result = loop {
        match waitpid(Pid::from_raw(pid), Some(WaitPidFlag::WNOHANG)) {
            Ok(WaitStatus::StillAlive) => {}
            Ok(status) => {
                break serde_json::json!({"stopped": true, "returncode": wait_code(status)})
            }
            // ECHILD: a status request already reaped it, which is the state
            // stop is asking for. Its code went out with that answer.
            Err(nix::errno::Errno::ECHILD) => {
                break serde_json::json!({"stopped": true, "returncode": null})
            }
            Err(e) => break serde_json::json!({"stopped": false, "error": e.to_string()}),
        }
        if std::time::Instant::now() >= deadline {
            break serde_json::json!({
                "stopped": false,
                "error": format!("pid {pid} did not exit within 5s of SIGKILL"),
            });
        }
        std::thread::sleep(Duration::from_millis(20));
    };
    println!("{result}");
    io::stdout().flush()
}

fn spawn_detached(spec: &ChildSpec<'_>, cwd: &Path) -> io::Result<i32> {
    if spec.argv.is_empty() {
        return Err(io::Error::new(io::ErrorKind::InvalidInput, "empty argv"));
    }
    // The same child setup as the capture and exec transports (env, cwd,
    // process group, capability drop, memory limit): a backgrounded command
    // outlives the request, so it is the one with time to go looking for the
    // launcher's fds, and its rlimits must not depend on the transport.
    let mut cmd = build_command(spec, cwd, false);
    apply_child_limits(&mut cmd, spec.memory_limit_mb);
    // The caller redirects its own output (as the background tool does), so
    // nothing here holds a pipe open across requests.
    cmd.stdin(Stdio::null());
    cmd.stdout(Stdio::null());
    cmd.stderr(Stdio::null());
    let child = cmd.spawn()?;
    let pid = child.id() as i32;
    let result = serde_json::json!({
        "returncode": 0,
        "stdout": "",
        "stderr": "",
        "pid": pid,
    });
    println!("{result}");
    io::stdout().flush()?;
    Ok(pid)
}

/// Kill the groups of commands still running when the request channel closed.
///
/// A PID namespace does this by construction: the launcher is its init, and
/// tearing the namespace down takes everything in it. Without one -- hardened,
/// or unconfined -- a backgrounded command would outlive the run, so the pids
/// are tracked and swept here.
///
/// `waitid` with WNOWAIT first: a pid that is no longer our child was already
/// reaped, and the number may have been recycled onto an unrelated process by
/// now. Only a live child of ours is signalled.
fn sweep_backgrounded(pids: &[i32]) {
    let flags = WaitPidFlag::WEXITED | WaitPidFlag::WNOHANG | WaitPidFlag::WNOWAIT;
    for pid in pids {
        if let Ok(WaitStatus::StillAlive) = waitid(Id::Pid(Pid::from_raw(*pid)), flags) {
            unsafe {
                libc::killpg(*pid, libc::SIGKILL);
            }
            let _ = waitpid(Pid::from_raw(*pid), Some(WaitPidFlag::WNOHANG));
        }
    }
}

fn io_err<E: std::fmt::Display>(e: E) -> io::Error {
    io::Error::other(format!("{e}"))
}

// Retained-output cap per stream: head + tail, middle dropped with a marker.
// The child's memory_limit_mb binds the CHILD; nothing bound the launcher's
// buffers, so an endless writer grew them until the HOST ran out of memory
// before the timeout. 4 MiB + 4 MiB keeps real build/test logs (whose tail
// carries the verdict) intact while bounding the launcher and the JSON result
// line the Python side buffers in turn.
/// How long a command's teardown waits for its output readers before
/// answering anyway. Only an escapee holding the pipe reaches it.
const CAPTURE_DRAIN_MS: u64 = 2000;
const STREAM_RETAIN_HEAD: usize = 4 * 1024 * 1024;
const STREAM_RETAIN_TAIL: usize = 4 * 1024 * 1024;

/// Drain `stream` to EOF, retaining at most head+tail bytes, decoded lossily.
/// Bytes (not read_to_string): a strict decode returned Err and dropped the
/// whole stream to "" on the first non-UTF-8 byte (a grep over a binary, a
/// latin-1 file), misleading every consumer while the real rc was reported.
/// Which stream a captured chunk arrived on.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Stream {
    Out,
    Err,
}

/// A command's output as it arrived: chunks tagged by stream, in arrival order,
/// under one head+tail cap.
///
/// Three views come off the one capture. A synchronous result reports stdout
/// and stderr SEPARATELY, so each renders alone; a log wants chronology, so it
/// renders merged. Recording the order once is what lets a command that starts
/// synchronous and is later handed back as a background job keep both -- the
/// order cannot be reconstructed from two separate buffers afterwards.
///
/// Ordering is as accurate as any capture without a pty: a process block-buffers
/// stdout into a pipe while stderr is unbuffered, so the writer's own flushing
/// dominates, exactly as it does for a shared `2>&1` fd.
#[derive(Default)]
struct Capture {
    head: Vec<(Stream, Vec<u8>)>,
    head_bytes: usize,
    tail: VecDeque<(Stream, Vec<u8>)>,
    tail_bytes: usize,
    // Per stream, so a flood on stdout does not stamp the cap marker onto a
    // stderr result that lost nothing.
    dropped_out: u64,
    dropped_err: u64,
    /// Once the command is handed back, its output stops being a result and
    /// becomes a log: chunks go straight to the file, merged.
    spill: Option<File>,
}

impl Capture {
    fn push(&mut self, stream: Stream, bytes: &[u8]) {
        if let Some(file) = self.spill.as_mut() {
            let _ = file.write_all(bytes);
            let _ = file.flush();
            return;
        }
        if self.head_bytes < STREAM_RETAIN_HEAD {
            self.head_bytes += bytes.len();
            self.head.push((stream, bytes.to_vec()));
            return;
        }
        self.tail_bytes += bytes.len();
        self.tail.push_back((stream, bytes.to_vec()));
        // Whole chunks: they are one read each (<=64 KiB), so the cap is coarse
        // by at most one chunk and the tagging stays intact.
        while self.tail_bytes > STREAM_RETAIN_TAIL {
            match self.tail.pop_front() {
                Some((stream, dropped)) => {
                    self.tail_bytes -= dropped.len();
                    match stream {
                        Stream::Out => self.dropped_out += dropped.len() as u64,
                        Stream::Err => self.dropped_err += dropped.len() as u64,
                    }
                }
                None => break,
            }
        }
    }

    /// One stream's bytes, or the merged stream when *only* is None.
    fn render(&self, only: Option<Stream>) -> String {
        let keep = |s: &Stream| only.is_none_or(|want| want == *s);
        let mut out: Vec<u8> = Vec::new();
        for (stream, bytes) in &self.head {
            if keep(stream) {
                out.extend_from_slice(bytes);
            }
        }
        let dropped = match only {
            Some(Stream::Out) => self.dropped_out,
            Some(Stream::Err) => self.dropped_err,
            None => self.dropped_out + self.dropped_err,
        };
        if dropped > 0 {
            out.extend_from_slice(
                format!(
                    "\n[agent6-jail] output over the retained cap; {} bytes omitted here\n",
                    dropped
                )
                .as_bytes(),
            );
        }
        for (stream, bytes) in &self.tail {
            if keep(stream) {
                out.extend_from_slice(bytes);
            }
        }
        String::from_utf8_lossy(&out).into_owned()
    }

    /// Hand the capture over to a log: write what arrived so far, merged and in
    /// order, then append there instead of retaining. No seam to stitch -- the
    /// merged view already exists, because the order was recorded as it arrived.
    fn spill_to(&mut self, mut file: File) -> io::Result<()> {
        file.write_all(self.render(None).as_bytes())?;
        file.flush()?;
        self.head.clear();
        self.tail.clear();
        self.head_bytes = 0;
        self.tail_bytes = 0;
        self.dropped_out = 0;
        self.dropped_err = 0;
        self.spill = Some(file);
        Ok(())
    }
}

/// Drain *stream* into *sink* until EOF. Runs on its own thread, which may
/// outlive the command: a grandchild outside its process group keeps the write
/// end open, and nothing can make that read return.
fn read_capped_into(mut stream: impl Read, tag: Stream, sink: &Mutex<Capture>) {
    let mut chunk = [0u8; 64 * 1024];
    loop {
        match stream.read(&mut chunk) {
            Ok(0) | Err(_) => break,
            Ok(n) => sink
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .push(tag, &chunk[..n]),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Drain to EOF and render, the shape these retention tests pin. The
    /// launcher itself never waits for EOF (see the teardown in run_child).
    fn read_capped(stream: impl Read) -> String {
        let sink = Mutex::new(Capture::default());
        read_capped_into(stream, Stream::Out, &sink);
        let rendered = sink
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .render(Some(Stream::Out));
        rendered
    }

    #[test]
    fn landlock_sets_handle_everything_and_the_writable_set_cannot_make_devices() {
        let sets = landlock_sets();
        assert_eq!(sets.handled, AccessFs::from_all(ABI::V3));
        assert!(sets
            .handled
            .contains(AccessFs::MakeChar | AccessFs::MakeBlock));
        assert!(!sets.all.contains(AccessFs::MakeChar));
        assert!(!sets.all.contains(AccessFs::MakeBlock));
        assert_eq!(
            sets.all | AccessFs::MakeChar | AccessFs::MakeBlock,
            sets.handled
        );
    }

    #[test]
    fn landlock_read_carries_execute_and_read_noexec_does_not() {
        let sets = landlock_sets();
        assert!(sets.read.contains(AccessFs::Execute));
        assert!(!sets.read_noexec.contains(AccessFs::Execute));
        assert_eq!(sets.read_noexec | AccessFs::Execute, sets.read);
    }

    #[test]
    fn read_capped_passes_small_output_through_verbatim() {
        let data = b"hello \xe9 world\n"; // non-UTF-8 byte survives lossily
        let out = read_capped(&data[..]);
        assert_eq!(out, String::from_utf8_lossy(data));
        assert!(!out.contains("omitted"));
    }

    #[test]
    fn read_capped_keeps_head_and_tail_and_marks_the_drop() {
        // 3x the total cap: the head keeps the start, the tail keeps the end,
        // and the marker names how many bytes fell in between.
        let total = 3 * (STREAM_RETAIN_HEAD + STREAM_RETAIN_TAIL);
        let mut data = vec![b'x'; total];
        data[..5].copy_from_slice(b"START");
        let end = data.len();
        data[end - 3..].copy_from_slice(b"END");
        let out = read_capped(&data[..]);
        assert!(out.starts_with("START"));
        assert!(out.ends_with("END"));
        let dropped = total - STREAM_RETAIN_HEAD - STREAM_RETAIN_TAIL;
        assert!(out.contains(&format!("{dropped} bytes omitted")));
        // Bounded: retained payload + marker, nowhere near `total`.
        assert!(out.len() < STREAM_RETAIN_HEAD + STREAM_RETAIN_TAIL + 200);
    }

    #[test]
    fn live_submounts_lists_carried_submounts_and_decodes_escapes() {
        let raw: &[u8] = b"22 1 0:1 / / rw - ext4 /dev/x rw\n\
23 22 0:2 / /ws rw - ext4 /dev/y rw\n\
24 23 0:3 / /ws/a\\040b rw - tmpfs tmpfs rw\n";
        let got = live_submounts(raw, Path::new("/ws"));
        assert_eq!(got, vec![PathBuf::from("/ws/a b")]);
    }

    #[test]
    fn a_later_bind_over_an_ancestor_shadows_the_subtree() {
        // The workspace bind carried /ws/.git/objects in; the protect bind
        // then covered /ws/.git, so the old objects line no longer resolves
        // to its mount. The protect bind's own recursive copy does.
        let raw: &[u8] = b"23 22 0:2 / /ws rw - ext4 /dev/y rw\n\
24 23 0:3 / /ws/.git/objects rw - tmpfs tmpfs rw\n\
25 23 0:4 / /ws/.git ro - ext4 /dev/y ro\n\
26 25 0:3 / /ws/.git/objects ro - tmpfs tmpfs ro\n";
        let got = live_submounts(raw, Path::new("/ws"));
        assert_eq!(
            got,
            vec![PathBuf::from("/ws/.git"), PathBuf::from("/ws/.git/objects")]
        );
    }

    #[test]
    fn an_overmount_at_the_same_path_shadows_the_earlier_one() {
        let raw: &[u8] = b"23 22 0:2 / /ws rw - ext4 /dev/y rw\n\
24 23 0:3 / /ws/v rw - tmpfs a rw\n\
25 23 0:4 / /ws/v rw - tmpfs b rw\n";
        let got = live_submounts(raw, Path::new("/ws"));
        assert_eq!(got, vec![PathBuf::from("/ws/v")]);
    }

    #[test]
    fn a_descendant_mount_shadows_nothing_and_prefixes_are_component_wise() {
        let raw: &[u8] = b"23 22 0:2 / /ws rw - ext4 /dev/y rw\n\
24 23 0:3 / /ws/v rw - tmpfs a rw\n\
25 24 0:4 / /ws/v/deep rw - tmpfs b rw\n\
26 22 0:5 / /wsx rw - tmpfs c rw\n";
        let got = live_submounts(raw, Path::new("/ws"));
        assert_eq!(
            got,
            vec![PathBuf::from("/ws/v"), PathBuf::from("/ws/v/deep")]
        );
    }
}
