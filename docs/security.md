# Security

agent6 treats the LLM as untrusted.
Each section states what is enforced and what is not.

## Reporting

Report privately through GitHub's [Security tab](https://github.com/agent6-dev/agent6/security/advisories/new).
Include: agent6 version (`agent6 --version`), kernel + distro (`uname -a`, `/etc/os-release`), `agent6 check sandbox` output, and a minimal repro (ideally a failing test under `tests/security/`).

## Threat model

Adversary: a fully malicious worker model, or an honest model prompt-injected by a file in the workspace.

The adversary controls the text of every LLM response, the choice of tool calls and their arguments within the published JSON schema, and the content of any file the agent reads.
Outside its control: the kernel, the agent6 binary, the provider endpoints.

**Holds**

- No writes outside the workspace
    - `sandbox.extra_write_paths` and the per-repo memory dir widen it, visibly in `config show`; so does the jail's persistent `HOME` (below), visible in `agent6 check boundaries`
- No reads outside the workspace and a read-only system set for installed toolchains
    - `strict`: `/usr`, `/bin`, `/sbin`, `/lib`, `/lib64` and `/etc/alternatives` bound read-only into an otherwise empty rootfs, whose `/etc` holds nothing else, plus a curated `/dev`
    - `hardened`: Landlock grants `/usr`, `/bin`, `/sbin`, `/lib`, `/lib64`, `/etc`, `/dev`
    - `sandbox.extra_read_paths` adds more; `agent6 check boundaries` prints the resolved set
- `/tmp` is writable at every level
    - `strict`: a private tmpfs discarded with the run
    - `hardened`: the host's `/tmp`
- `HOME` is a directory of agent6's own, never the operator's
    - `strict`: `/tmp/agent6-home`, created inside the private tmpfs and gone with the run; `[sandbox].home = "cache"` swaps in the persistent dir below, bind-mounted read-write at its real path
    - `hardened` and `none`: `$XDG_CACHE_HOME/agent6/home` (default `~/.cache/agent6/home`), created `0700` by agent6 and, under `hardened`, Landlock-granted read-write like the workspace; the rest of the operator's home stays ungranted
    - a symlink, another user's directory, a mode open to group or others, or a path inside agent6's private dirs at that location refuses the run; the mode is checked on every run (a jailed command can `chmod` its own HOME) and the refusal names `chmod 700`, nothing restores it silently
    - persistence is a cross-run channel inside the jail's world: a poisoned cache or a `~/.gitconfig` alias reaches the next jailed run, never the operator's own tools
- agent6's own git never pushes, force-pushes, rewrites history, or `reset --hard` ([Git](#7-git))
    - a `git` the model runs through `run_command` is bounded by the sandbox instead: `protect_git` keeps `.git` unwritable under `strict`, and push needs egress
- No persistence after the run: no daemon, cron, `.bashrc` write, or setuid binary
    - the exception is the jail's persistent `HOME` (`hardened`, `none`, or `[sandbox].home = "cache"`): a file or binary written there reaches the next jailed run, and under `hardened` it is executable; the operator's own shell, tools, and dotfiles never read it
    - chmod-family syscalls (`fchmodat2` included) deny modes carrying `S_ISUID` / `S_ISGID`; ordinary chmod passes
    - every mount carries `nosuid` and `nodev`, except the bound `/dev` nodes (the builtin five: `null`, `zero`, `urandom`, `random`, `full`, plus any `sandbox.extra_device_paths` grant), which `nodev` would make unusable
    - `/tmp` allows exec (toolchain helpers)
    - children write inside the jail's mount namespace (`strict`) or the Landlock write grants (`hardened`)
    - nothing a command starts outlives it: `strict`'s PID namespace takes the tree down; `hardened` holds `PR_SET_CHILD_SUBREAPER` and kills every process that appeared during the command (a `setsid` daemon included)
    - a survivor the sweep cannot kill fails the command

**Does not hold**

- The agent process's own egress is unbounded
    - agent6 reaches the configured providers (each `[providers.*].base_url` host, plus the fixed ChatGPT OAuth authority for token grants); nothing stops the process reaching elsewhere
    - a jailed command's egress is bounded ([Network](#5-network))
- On `hardened`, a command can hand work to a user daemon already running (tmux, `systemd --user`): unix sockets have no Landlock hook and stay nameable without a mount namespace
    - `strict` does not expose them

## Defense layers

`agent6 check boundaries` prints the resolved picture per actor (in-process tools, jailed commands, each MCP server): reachable paths, network, approval mode, secrets posture, and the cause when `auto` selected less than `strict`.

### 1. Trust boundary

The agent's Python process is trusted and runs unconfined at every isolation level; the model is untrusted.

- the process holds the provider keys, writes the per-repo state dir, spawns the jail
- isolation levels differ only in which jail features the launcher enables

The model reaches the machine through two surfaces:

- The in-process tools, whose paths resolve against the file boundary ([File access](#3-file-access)).
- Anything that executes, which runs in the jail ([Sandbox](#2-sandbox)).

The model sees the fixed tool set in `src/agent6/tools/schema.py`.

- structured edits, read-only navigation, fixed-argv verify and metric commands, `finish_session`, `ask_user`, a curator task notepad, approval-gated `fetch` ([Network](#5-network)), capability-gated `run_command`
- no `shell`, no `write_file`, no `eval`
- adding a tool needs a security review note ([AGENTS.md](https://github.com/agent6-dev/agent6/blob/master/AGENTS.md))

Under `api_format = "claude_code"` the model runs inside the operator's installed Claude Code binary, unjailed, as the operator, with every built-in tool off (`--tools ""`).
Its only reach into the machine is agent6's tools served over the sdk MCP tunnel, so the dispatcher, the jail, and the approval gates are unchanged; the binary holds the operator's Claude login exactly as an interactive `claude` does, and agent6 never reads it.

### 2. Sandbox

Every tool call that allows the model to run arbitrary commands (`run_command`, `run_verify_command`, backgrounded commands, and MCP servers) runs in a jail at the effective isolation level.

`agent6 check mcp` and `agent6 mcp connect` start a spawned MCP server under that jail with the repository bound read-only.
`check mcp` applies the run's refusals first and leaves a server it cannot hold that way unstarted: `unconfined = true`, a write grant (`write_paths`, `sandbox.extra_write_paths`, `sandbox.extra_device_paths`), or no jail at all.
`mcp connect` probes under the same jail but only skips the probe when there is no jail at all, saying so; it is the operator's own invocation naming the server they are adding.
A diagnostic hands a server the repository to read and writes nothing but the config it was asked to write.

**Modes** (`[sandbox].isolation`)

| Mode | Applies |
|---|---|
| `strict` | user + mount + PID + IPC + UTS + network namespaces, `pivot_root` rootfs, private `/proc` and `/tmp`, curated `/dev`, Landlock, seccomp, `NO_NEW_PRIVS`, capability drop, timeout |
| `hardened` | Landlock (ABI >= 3), seccomp, `NO_NEW_PRIVS`, capability drop, timeout. No namespaces, no rootfs |
| `none` | Nothing. Timeout only |
| `auto` *(default)* | The strongest of the three this host supports |

Landlock is best-effort under `strict` (a kernel without it skips that layer, warned once per run) and is `hardened`'s only filesystem boundary.

**Resolution**

Probed per host: `unshare` for namespaces, the Landlock ABI syscall, the seccomp filter's architecture (x86_64 and aarch64).

| Setting | Host | Effective |
|---|---|---|
| `auto` | namespaces | `strict` |
| `auto` | no namespaces, Landlock ABI >= 3 | `hardened` |
| `auto` | no namespaces, Landlock ABI 1-2 | `hardened`, warned: truncation outside write grants is unconfined |
| `auto` | no namespaces, no Landlock | `none`, warned |
| `auto` | non-Linux, or no seccomp filter for the arch | `none`, warned |
| `strict` | no namespaces | refused |
| `hardened` | Landlock ABI < 3 | refused |
| `strict`, `hardened` | non-Linux, or no seccomp filter for the arch | refused |
| `none` | any | `none`, warned |

`--dangerously-disable-sandbox` and `AGENT6_DANGEROUSLY_DISABLE_SANDBOX=1` force `none` for one invocation.
Config, flag, and env var are operator-only; the model reaches neither argv nor env.

**Inside the jail**

- Mounts (`strict`): cwd and a private `/tmp` writable, system paths read-only, `extra_read_paths` / `extra_write_paths`, a persistent `HOME` under `[sandbox].home = "cache"`, and operator tool dirs at their real paths
    - every mount keeps the path it has outside
    - a fork's leg (cwd is its linked worktree) also grants the repository's `.git`: a read-only bind under `strict`, a Landlock read+exec rule under `hardened`, in both cases whatever `protect_git` says; the main checkout's `.git` is read-only only under `strict` with `protect_git`, so a fork's is the more confined of the two, and the model's prompt says so
    - the granted dir is the one agent6 recorded when it added the worktree (the manifest's `worktree_git_dir`, taken from the repository it ran in), never the worktree's own `.git` pointer, which a jailed command can rewrite under hardened; the policy builder refuses when the pointer no longer resolves to the record, and a linked worktree agent6 did not record gets no grant
    - the record reaches every policy consumer, the hardened exposure scan included: a `hide_paths` entry inside it refuses like any other unmaskable exposure
- Masked last, after every bind (`strict`): the config dir, the state base, `[sandbox].hide_paths`
    - a grant at or inside a private dir is refused at config load
    - `hardened` cannot mask: a grant containing a private dir warns, an unmaskable `hide_paths` entry refuses
- `/dev` (`strict`): `null`, `zero`, `urandom`, `random`, `full`, a private `shm`; no `/dev/tty`
    - `sandbox.extra_device_paths` binds named `/dev` nodes read-write (GPU compute); each must be a char/block device on the host or the launch refuses, and on `hardened` the same grant is a Landlock read+write rule on the node
- `/proc` (`strict`): fresh and private, empty if that fails
    - the launcher runs with an empty environment; it is PID 1 there, so the command can read `/proc/1/environ`
- seccomp: a 36-syscall deny-list returning `EPERM`, covering process inspection (`ptrace`, `pidfd_getfd`, `process_vm_readv`/`writev`, `kcmp`), `io_uring_setup`, `userfaultfd`, the whole mount family (`mount`, `umount2`, `pivot_root`, `mount_setattr`, `open_tree`, `move_mount`, `fsopen`, `fsconfig`, `fsmount`, `fspick`), `setns`, `unshare`, `kexec`, `bpf`, `perf_event_open`, the keyring calls (`keyctl`, `add_key`, `request_key`), module loading, `reboot`, swap, and the clock-setting family
    - anything not on the list is allowed; the list itself is the source (`jail/src/main.rs`), and it grows by syscall, never by class
- Capabilities: cleared between fork and exec.
- Timeout: `timeout_s` (verify and metric gates use `[workflow].verify_timeout_s`, default 600), then SIGKILL of the process group, rc=124
    - a model's `run_command` is not wall-clock killed: at `[workflow].command_checkin_s` it is handed back as a background job ([Commands and environment](#4-commands-and-environment))
- One launcher per run at every isolation level; its commands share that netns, PID namespace, and `/tmp`
    - closing the run's channel takes the PID namespace down
    - a launcher that cannot start leaves each command its own
- The policy arrives as JSON on the launcher's stdin, validated against a strict schema; unknown fields are refused.
- `[sandbox].memory_limit_mb` (default 0, off): per-process `RLIMIT_DATA`.

**Verify**

`agent6 check sandbox` prints the isolation this host resolves to, the reason when it is not `strict`, and every tool bin symlink resolving out of its bin dir (whose target directory is mounted read-only into the jail).
It then runs live probes at that isolation.
Exit 0 when all pass, 1 otherwise.

| Probe | Passes when |
|---|---|
| `landlock_abi` | the kernel reports Landlock ABI >= 1 |
| `jail_true` | `/usr/bin/true` runs in the jail, rc=0 |
| `jail_blocks_network` | a jailed `getent hosts example.com` fails. n/a under `hardened`, which has no per-command network namespace |
| `jail_blocks_etc_write` | a jailed write to `/etc/agent6-escape` leaves no file on the host |

### 3. File access

`read_file`, `list_dir`, `outline`, `find_definition`, `find_references`, `apply_edit`, and `apply_patch` run in the agent process and ask no approval.
Every path they take resolves through `Workspace`:

    workspace root
      + [sandbox].extra_read_paths (read) / extra_write_paths (write)
      - [sandbox].hide_paths
      - agent6's own config dir and state base

- The boundary comes from config values
    - a degradation (`auto` falling back to `hardened` or `none`, a host with no jail) leaves it unchanged; under `none` isolation it is the only boundary
- A denied path is refused with the reason
    - `list_dir` drops the entry and reports `hidden: N`
    - the jail masks the path instead (empty dir, empty file; [Sandbox](#2-sandbox))
- The symbol index skips a hidden file (an indexed one leaks symbol names and line numbers through `find_definition`)
- Under `sandbox.protect_git` (the default, and off only where `git.control = "model"` requires it) the edit tools refuse a write into the project's own `.git`, raw or symlink-resolved, at every isolation level
    - the name matches case-folded on every platform (macOS and Windows open `.GIT/config` as `.git/config`; macOS runs unsandboxed)
    - the same refusal covers a `pyvenv.cfg` dir, a `site-packages` ancestor, and an operator protect path
- Rewriting an editable-install `.pth` corrupts a venv invisibly (venvs are gitignored), so those writes refuse
    - reads stay allowed; an editable install still imports itself inside the jail (the repo is bound at its real path)
- Inside the state dir the edit tools may write the memory dir (`<state-dir>/<repo-id>/memory/`) and nothing else (the exempt list in `tools/_path_safety.py`).
- Uncovered: with `isolation = "none"` and `run_commands = "yes"`, a command reads a denied path directly (`run_command` argv is not screened)

**Repo memory** is a prompt-injection persistence channel: `MEMORY.md` joins the system prompt of later runs on the same repo.

- Entries are inert data.
- The injected index is size-capped and framed as untrusted.
- The store is auditable (`agent6 memory list` / `show`).
- The jail never mounts it.
- Files are freely editable, so a hostile write can erase earlier memories.
- Sandbox, egress, and git policy come from config, so memory content cannot move them.

### 4. Commands and environment

Every command tool (`run_command`, `run_verify_command`, `run_metric_command`, `stop_background`) answers to `[sandbox].run_commands`; the first three run jailed, and `stop_background` signals a jailed child agent6 already started.
`run_commands = "no"` withholds the verify gate too, and such a run starts gateless.
Under `ask`, a denied gate is withheld the same way for the rest of the run (no retry loop can discharge a refusal), and the run ends unverified.

The agent works within the environment it is given and cannot expand it:

- `sudo` cannot escalate, even passwordless: `NO_NEW_PRIVS` voids setuid, so jailed `sudo` fails under any `NOPASSWD` rule.
- Package installs fail: `apt` / `dnf` / `apk` need root (blocked), a route to a mirror (the default `network` has none off the box), and `/usr` or `/var` writes (denied)
- Host-installed toolchains compile and run; a networked build step needs `network` loosened.
- Toolchains, venvs, and deps are installed outside agent6. Access widens through config (`extra_read_paths`, `network`, `[providers.*].base_url`), all visible in `config show`.
- Running agent6 as root needs `--allow-root` / `AGENT6_ALLOW_ROOT=1` (plus a banner) and weakens the boundary
    - `strict` maps inside-root to real root: jailed children run as real root under Landlock, seccomp, and `NO_NEW_PRIVS` only
    - writes outside the workspace and routes off the box stay closed
    - readable files now include root-only ones (`/etc/shadow` under `hardened`; `strict`'s rootfs hides it)
- Under `sudo`, agent6 reads the real user's config and secrets (`SUDO_UID` / `SUDO_USER`) and chowns state-dir writes back
    - it does not drop privileges in-process; confinement comes from the jail

### 5. Network

The agent process's own egress is unbounded.
Everything below bounds a child of it.

A `claude_code` provider's child dials `api.anthropic.com` and `claude.ai` (login refresh) from the agent process; telemetry, crash reporting, and update hosts are disabled by its environment.
While it runs it listens on a same-uid session socket under `/tmp/cc-socks/`, reachable from a jailed command only under `hardened`, whose `/tmp` is the host's.

**`fetch`** is the model's only direct egress.

- One https URL, GET, no redirects followed, no credential, text only, 1 MiB.
- Hosts on `sandbox.fetch_hosts` are read without asking (empty by default); any other host prompts.
- Nothing resolves before that gate: a DNS query delivers the hostname to whoever runs its authoritative server, so an unapproved URL never reaches a resolver.
- Hidden entirely when `network = "host"`, where a jailed command has its own route off the box.

**Which network a jailed child joins.** One of:

- `host`: the machine's network.
- `session`: the run's own.
  Its members reach each other and nothing off the box.
- `none`: its own, alone.
- `auto` *(default)*: the safest of these the host supports.

The run owns one session network.
A holder process creates it, the run keeps it alive with an open descriptor on `/proc/<holder>/ns/{user,net}`, and every child that asks joins those.
Entering a network namespace needs `CAP_SYS_ADMIN` in the user namespace that owns it, so a joiner enters that user namespace too; it still gets its own mount, PID, IPC, and UTS namespaces, so two members cannot see or signal each other.

Under `strict`, the only level with namespaces:

| jailed command | `auto` *(def)* | `session` | `only_explicit_states` | `host` |
|---|---|---|---|---|
| `run_command` | the run's session network | same | same | host network |
| `tool`, `network` `auto`(def)/`none` | own, alone | own, alone | own, alone | own, alone |
| `tool`, `network = host` | ⛔ refuse | ⛔ refuse | host network | host network |

On `hardened` there is no network namespace: `auto` degrades to the agent process's network with a once-per-run warning, while `session` and `none` refuse.
Under `none` isolation nothing is enforced or refused.

**Refusals** (fail-closed)

| Configuration | When |
|---|---|
| a `tool` sets `network = host` under `network` `auto`/`session` | machine start |
| `network = only_explicit_states`, or explicit `network = session` | run start, `hardened` |
| a machine under `network = session`, or any `tool` with `network = none` | machine start, `hardened` |

**MCP servers** take the same values per server, default `auto`:

- `auto`: a network of its own where the host can give one, degrading to the host's with a warning.
- `none`: refuses instead of degrading.
- `session`: joins the run's network, so a browser server reaches the dev server a background command started.
- A server is spawned on stdio or dialled at an operator-set `url`, outbound either way.

**Ingress.** The loop opens no accept-side socket; the task graph is an in-process curator.
`agent6 web` opens one, and only when started.

- It binds loopback (`127.0.0.1`) by default with no app auth (run it behind `tailscale serve`, where the tailnet identity is the access control; see [the web UI](web.md)).
  A non-loopback bind needs `[web].allow_non_loopback = true` for `[web].host`, or `--allow-non-loopback` for `--host`.
- The server renders folded state and drives typed contracts.
  New-work spawns fixed argv with the task behind `--`; machine-run is allow-listed to authored files; answers write only the addressed run's answer files (session id, answer id, and machine target state dir each validated to one path component); merge, prune, and config-set are fixed agent6 subcommands.
- State-changing POSTs carry a CSRF guard: the body must be `Content-Type: application/json` (a cross-site `fetch` with it triggers a preflight the server never answers) and any `Origin` must match `Host`.
  It holds on loopback and behind `tailscale serve`, and does not cover DNS rebinding (that needs a Host allow-list incompatible with the tailnet name).
- Request framing is bounded: 1 MiB body cap (413), chunked refused (411), and any unread-body refusal closes the connection.
- The machine write surface (`POST /api/machine/<name>/{poke,stop,steer,approve,answer}`) uses the same guards.
  `poke` writes the instance's signal file (inert JSON the next `tool` reads) and `stop` its stop marker; `steer`, `approve` and `answer` write only the current agent state's per-state dir.
- PWA assets are static and the service worker is a no-op passthrough (no Web Push, no VAPID).
  No telemetry, no auto-update, no remote control plane.

### 6. Approvals

- `[sandbox].run_commands` (`ask` default, `yes`, `no`) gates every command tool; an `ask` prompt shows the argv
- Each `mcp__<server>__<tool>` call prompts with its arguments (`[mcp.servers.<name>].approve`, default `ask`).
- An "allow all" answer covers that server for the run, never the command tools or another server
- A tool name matching no configured server is refused, not prompted.
- The `fetch` off-list host prompt and the sandbox-off gate take no standing answer; both say so, and no front-end shows the button
- `isolation = "none"` with auto-approved `run_command` adds a one-time gate: `Continue?
  [y/N]` interactively, a warning in CI and `machine run`.
- A prompt with no operator to answer it: a headless run refuses to start under `ask` unless `AGENT6_DETACHED_AWAY` is `deny` (auto-deny), `wait` (park it for a front-end) or `approve` (grant every scope); an unattended machine auto-denies.
- `agent6 mcp serve` has no operator at all, so the tools that would prompt are not published: under `run_commands = "ask"` or `"no"` its command tools are absent from `tools/list`, and a client that names one is told which setting withheld it ([the tools](acp.md#as-an-mcp-server)).

### 7. Git

**agent6's own git**

- agent6's own git writes go through `git_ops.py` alone
    - it wraps the safe ops (status, add, commit, diff, branch, checkout)
    - it spells no destructive verb at all: `push`, `reset --hard`, `commit --amend`, `rebase`, `filter-branch` / `filter-repo`, `branch -D` / `--force` and any `--force` / `-f` appear nowhere in it, so there is nothing to enable (pinned by `test_git_ops_never_spells_a_destructive_verb`)
    - the collectors on the [subprocess allowlist](#12-host-side-subprocess-allowlist) carry the same hardening flags: `sessions diff` and `ask` read only, and `review` stages untracked files with `add -N` so they appear in its diff, undoing it with `reset` in a `finally`; `skills install` clones with fixed argv
- One operator-only exception: `sessions prune --delete-squashed` force-deletes a run branch the manifest confirms was squash-merged (the commit survives in the reflog).
- `git_ops.py` runs git with the configured `api_key_env` names removed from its environment: a credential helper or content driver never inherits one
    - PATH, SSH, proxy, and credential-helper vars stay
    - the read-only collectors inherit the environment untouched (no remote contact; the hardening flags leave no repo-controlled code to receive it)

**A `git` the model runs through `run_command`** is bounded by the sandbox, and its argv is not screened.

- `protect_git` (default on) keeps `.git` unwritable under `strict`: re-bound read-only, recursively (a mount nested under it stays visible and read-only)
    - a rewrite fails; `push` has no egress
- `protect_git` is strict-only: on `hardened` the default degrades with a warning, an explicit `true` refuses to run
    - a jailed command there can plant a `filter.<n>.clean` plus a `.gitattributes`, which agent6's own auto-commit (a temp-index `git add -A` on the host) then runs, reaching `$HOME` and the network
    - Landlock cannot express the exclusion: a directory grant is recursive and stacked rulesets intersect, so denying `.git` means not granting the workspace root either
- The protected scope is the project's own `.git` at every isolation level (the in-process edit tools refuse writes under it; [File access](#3-file-access))
    - a nested `.git` (a vendored repo's, a submodule's) is workspace content, writable like any other file

**Repo-controlled host code in a poisoned `.git/config`**

- `core.fsmonitor` and `diff.external` are always off.
- `.git/hooks/*` run only under `git.run_repo_hooks = true` (default false); `core.hooksPath` points away, so a hook cannot fire on agent6's auto-commit.
- Content drivers (`filter.<n>.clean` / `smudge` / `process`, `merge.<n>.driver`) are off by default (`git.run_repo_filters`), neutralized per name
    - the clean filter runs on the auto-commit's `git add`, the merge driver on the chain merge's `merge-tree`: a cloned poisoned repo fires one without any model action
    - `true` honors them (the Git-LFS opt-in)

### 8. Secrets and `connect`

- Provider keys live in `$XDG_CONFIG_HOME/agent6/secrets.toml`, `0600` and owner-only (refused if group- or other-readable, or foreign-owned), or come from `[providers.<name>].api_key_env` (env wins).
- They are absent from transcripts, redacted in `config show`, and masked from the jail: the config dir stays masked even under an explicit grant, and a grant naming it directly is refused at config load.
- No child agent6 spawns carries one: a jailed command's environment is built from its policy, and the unconfined paths (`isolation = "none"`, git subprocesses, MCP servers) drop every configured `api_key_env` name.
- `agent6 connect` prompts locally (`getpass`) and writes config and secrets
    - one read-only `GET` to the provider's key endpoint confirms auth (status only; `--no-verify` skips it)
    - it executes nothing a remote returns
- `agent6 connect chatgpt` is a PKCE OAuth sign-in
    - a browser hits the authorize page of OpenAI's fixed OAuth authority (a constant, not config); the code returns on `localhost:1455` (or is pasted), state-checked either way
    - token exchange and refreshes `POST` only to the authority's `/oauth/token`
    - tokens live in `secrets.toml` under the same `0600` and executes-nothing rules
    - agent6 never sends a rating or feedback on a response and has no rating surface: the backend may use rated turns for training, and that choice is never made on the operator's behalf
- `agent6 connect claude` stores nothing: Claude Code's own login (`~/.claude/.credentials.json`, `~/.claude.json`) is never read, copied, or mounted by agent6
    - the child's environment carries no `ANTHROPIC_*` or `CLAUDE*` variable from the operator shell, so a shell API key cannot override the subscription login; `CLAUDE_CONFIG_DIR` is the one passthrough
    - `claude auth status --json` is parsed for `loggedIn` only; its body (email, org) is never printed or journaled
    - the binary's initialize handshake carries the account block; agent6 keeps the email in memory only, to replace it with `<operator-email>` in the model's returned text, and records none of it

### 9. State and locks

- An in-process `GraphCurator` owns the task graph
    - every mutation validates against a pydantic schema before writing, under a per-mutation flock on the session dir
    - a write-path fault after the in-memory update reloads from disk before surfacing: a later read never observes a node that was never persisted
- Per-repo state lives at `$XDG_STATE_HOME/agent6/<repo-id>/` (override with `[agent6].state_dir`), outside the working directory jailed commands run in.
- The config write lock serializes read-modify-write cycles and enforces nothing
    - publishes are atomic: a torn config is impossible with or without it
    - it fails open (a planted symlink refuses `O_NOFOLLOW`; a stale root-owned lock is ignored); a write proceeding without it is kept, reported "kept as written" (docs/config.md)

### 10. Parallel lanes

`agent6 run --parallel` and a live run's `/parallel` steer directive (see [architecture.md, Parallel runs](architecture.md#parallel-runs)) each spawn subordinate work.

- Every lane is an ordinary run: a detached `agent6 run` on its own clone, its own jail per `sandbox.isolation`, its own `run_commands` policy
    - no sandbox socket is shared across lanes or with the parent
- Every spawned lane carries `AGENT6_SUBRUN=1`; both the `--parallel` flag and the coordinator's `lane_spawner` wiring refuse when it is set, so a lane cannot fan out or dispatch again.
- A lane's config carries key references, never secret values
    - the orchestrator writes each lane a `--config` via `materialize()` (the resolved `Config`: provider `base_url`, `api_key_env` names), never a raw key
    - the lane reads the same `secrets.toml` or provider env var as any other run
- Lane git plumbing (clone, fetch, merge) goes through `git_ops.py` and lane spawning through `ui/spawn.py`, both already on the [subprocess allowlist](#12-host-side-subprocess-allowlist).
- A lane starts from committed state only
    - the fan-out clones HEAD; a coordinator dispatch cuts lanes at the run's chain tip after chain-committing changes
    - `--parallel` refuses an origin with uncommitted tracked changes under `git.require_clean_worktree`

### 11. State machines

`machine run` is a supervisor that makes no network calls.
Each `tool` state is jailed, so a per-tool `network` sets its netns independently ([Network](#5-network)): a machine can keep agents on the provider API while one reviewed, fixed-argv `tool` reaches the network.

**Operator-gated policy**

- `network` is read only from the operator's config
    - a machine's `[config]` overlay is rejected at load if it declares `[providers.*]`, `[sandbox.*]`, `[presets.*]`, `[mcp.*]`, `machine.notify`, `notify.on_complete`, `git.run_repo_hooks`, or `git.run_repo_filters`
- A `tool` only declares `network`; honoring `allow` is the operator's call, and every conflict is refused at startup naming the state.

**Bundle confinement**

- Scripts live in a reviewed `scripts/` beside the `.asm.toml`
    - `machine check` verifies every entry and static reference resolves inside the bundle (escaping symlinks rejected)
- Scripts are operator-authored and committed, never fetched or generated at run time
    - the `.asm.toml` and `scripts/` are read-only in every jail during a run
- Front-ends render `machine.notify` as an overlay, and `attach` / TUI call `notify-send` with fixed argv, so a model message is inert data.
- The out-of-band hook `[machine.notify].on_event` runs an operator argv on the host with the minimal `hook_env` env plus `AGENT6_MACHINE_*`.

### 12. Host-side subprocess allowlist

Everything the model can influence runs through `run_in_jail` ([Sandbox](#2-sandbox)).
A fixed set of modules also shells out directly with `subprocess.run` / `Popen`, each with fixed argv depending only on operator input.
`tests/security/test_subprocess_allowlist.py` pins the file list; audit with `rg 'subprocess\.|os\.(system|exec|posix_spawn)' src/agent6/`.

- `git_ops.py`: agent6's own git operations ([Git](#7-git)).
- `sandbox/detect.py`: probes the host's sandboxing capabilities.
- `sandbox/jail.py`: the jail launcher.
- `tools/mcp_client.py`: operator-configured `[mcp.servers.*]` commands.
  A server with a sandbox policy spawns through the same launcher and `JailPolicy` a jailed command gets (`spawn_in_jail`); a server the operator opted out is a plain subprocess.
- `providers/token_command.py`: the `[providers.*].token_command` that mints a provider bearer.
- `providers/claude_code.py`: the `api_format = "claude_code"` provider runs the operator-installed Claude Code binary with fixed argv (binary, model, effort, literal flags, the path of a 0600 system-prompt file in a private empty directory); prompts, tool results, and notices travel on stdin, so model- or repo-derived text never becomes an argv element.
  Curated environment, `--tools ""`, `--allowedTools mcp__agent6`, `--setting-sources ""`, `--strict-mcp-config`, `--disable-slash-commands`, `--no-session-persistence`; CLAUDE.md, auto-memory, and auto-compaction off by environment; `system/init` is audited so a tool outside `mcp__agent6__*` or an API-key source refuses the run.
  Unjailed: it needs the operator's login under `$HOME` and its own egress, at the agent process's trust tier.
  Tool results stay under Claude Code's 50,000-byte persistence threshold (the loop caps them at 45,000 characters for this provider; a wider one is refused), so no tool output is written under `~/.claude`.
  `claude auth status --json` runs the same way for the sign-in preflight.
- `sessions/ipc.py`: `ps -p <pid> -o lstart=` on hosts without `/proc` (macOS), for the `worker.pid` start-time identity, over a pid agent6 recorded.
- `ui/btw.py`: spawns `agent6 ask` detached for `/btw` (every composer), so the side question keeps provider egress while the run is confined.
  Argv is the agent6 exe plus the question the operator typed, with `--` before it.
- `ui/spawn.py`: the shared front-end spawn helper.
  Spawns the agent6 CLI detached for run and machine launches, and captures `sessions merge` / `prune` / `config set`.
- `ui/notify.py`: `notify-send` with fixed argv (exe, `--`, two positional data args, no shell) for the device-present machine notification.
- `ui/cli/` helpers: `$EDITOR` for plan and steer editing; `git diff` / `log` for the review subcommand and the `sessions` / `ask` diff views, with argv from the run manifest the CLI wrote outside the jail; `rg` for history search; the fixed-argv `python -m agent6.ui.tui` co-process behind `run --tui`; `cp` / `rm` / `apparmor_parser` via sudo with fixed argv for `agent6 system apparmor`.
- `app/finalize.py`: both operator notify hooks (`run_notify_hook`): `[notify].on_complete` at run end and `[machine.notify].on_event` from a machine.
  Argv from config, env from `hook_env` (a minimal base plus `AGENT6_SESSION_*` or `AGENT6_MACHINE_*`, never the provider keys in the operator environment).
- `app/machine/_scriptcheck.py`: ruff and ty with fixed argv, reading generated scripts statically.
  Those scripts execute only via `run_in_jail`.
- `app/machine_agent.py`: spawns each agent state as a fixed-argv `python -m agent6.ui.cli.machine_agent` subprocess whose request travels in a temp file.
  Its `[machine.notify].on_event` hook runs on the host through `app/finalize.run_notify_hook`.
- `ui/cli/skills_cmds.py`: `git clone --depth 1 -- <url>` with fixed argv for `agent6 skills install`.
  The URL is operator-supplied and nothing fetched is executed.
- `ui/tui/clipboard.py`: `tmux set-buffer -w` with the copied transcript text as one data argument.
- `ui/tui/conversation.py`: the operator's `$PAGER`, argv from the environment, transcript text on stdin.

## Skills

- A skill is operator-installed config: install only from trusted sources
    - `skills install <url>` is an operator-initiated CLI fetch (the `connect` trust class); what it installs enters the system prompt and tool results verbatim
- Nothing in a skill runs at install or load
    - its scripts run only through the jailed command path, subject to `run_commands`
- `use_skill` is read-only and path-contained: the skill's own dir through a component-walked descriptor (any symlink hop or `..` refused)
    - skill dirs are not mounted into the jail; content reaches the model engine-side
- Repo-local `.claude/skills/` are not discovered; only the installed dir and `[skills].extra_dirs` are scanned

## Prompt-injection tests

[`tests/security/test_prompt_injection.py`](https://github.com/agent6-dev/agent6/blob/master/tests/security/test_prompt_injection.py) drives the dispatcher with the calls a compromised model would make: path traversal, a swapped symlink between check and open, an unknown tool name, extra fields, a write outside the workspace.
It asserts the tool surface refuses them whatever the model says; it does not test the model's judgement, and calls no model.
It catches prompt regressions; the structural defenses above confine a model that follows an injection.

## Known limitations

- User namespaces must be enabled; agent6 refuses `strict` on distros that disable them.
- AppArmor userns (Ubuntu 24.04+) blocks unprivileged userns without a profile
    - agent6 ships one scoped to the launcher (`agent6 system apparmor install`): with it `strict`, without it `hardened`
- seccomp is required; kernels that block it from unprivileged callers make the jail fail closed.
- Devcontainers get `hardened`: the container bounds filesystem damage; jailed commands share the container's network ([Network](#5-network))
    - the XDG state base is ephemeral (lost on rebuild): mount a volume at the state dir or set `[agent6].state_dir`
- agent6 installed inside the project it works on (pip into the project's own venv) puts the running agent's code in the jail's writable workspace
    - a jailed command can rewrite it; the next tool call runs the rewrite as you, outside the jail
    - install agent6 outside the tree (pipx, `uv tool`); agent6 warns at run entry on this shape
- Claude Code (`api_format = "claude_code"`) appends the account email to every system prompt it sends and has no switch for it; the model can echo it, and agent6 scrubs it only from returned text, never from tool inputs (an edit carrying it lands as written)
    - managed settings (`/etc/claude-code/managed-settings.json`) still apply inside the child; the `system/init` audit refuses the run when the tool list differs from what agent6 offered or `apiKeySource` is not `none`, and checks nothing else: a managed hook, or a managed MCP server exposing no tool, passes it
- Side channels: no claim about timing, cache, or speculative side channels.
- Supply chain: pin your install
    - runtime deps `pydantic`, `httpx2`, `argcomplete`, the `tree-sitter` pair, `textual`, `ruff`, `ty`; build dep `hatchling`; jail crates `nix`, `libc`, `landlock`, `seccompiler`, `serde`, `serde_json`
