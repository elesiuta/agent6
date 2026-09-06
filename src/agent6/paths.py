# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Filesystem path + identity resolution for agent6.

Single source of truth for:

- the global (user-level) config + secrets directory under XDG
  (`$XDG_CONFIG_HOME/agent6` or `~/.config/agent6`),
- the per-repo config path (`<state_dir>/config.toml`, out of the repo),
- the run-state directory (`$XDG_STATE_HOME/agent6/<repo-id>`), and
- the *real* operator when agent6 is invoked through `sudo`, so we read
  the user's config/secrets (not root's) and never leave root-owned files
  scattered in their repository.

Security model (see docs/security.md):

- Running an LLM-driven agent as root is dangerous. agent6 refuses to run
  as root unless the operator explicitly opts in via `--allow-root` or
  `AGENT6_ALLOW_ROOT=1`, and prints a loud banner either way.
- When `euid == 0` and the process was launched through `sudo` we
  resolve the invoking user from `SUDO_UID` / `SUDO_GID` / `SUDO_USER`
  and `chown` anything we create back to them. We do NOT drop privileges
  in-process: the whole point of `sudo agent6` is that verify/run
  commands need root, and those run inside the jail as root regardless, so
  juggling euid in the bookkeeping code would be theatre. The jail remains
  the real boundary.
"""

from __future__ import annotations

import contextlib
import hashlib
import os
import pwd
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

_ALLOW_ROOT_ENV = "AGENT6_ALLOW_ROOT"  # documented in docs/config.md


@dataclass(frozen=True, slots=True)
class RealUser:
    """The human operator agent6 is acting on behalf of.

    Differs from the process euid only when agent6 runs under `sudo`:
    there `uid`/`gid`/`home` describe the user who typed `sudo`,
    not root.
    """

    uid: int
    gid: int
    name: str
    home: Path
    via_sudo: bool


def _passwd_entry(uid: int) -> pwd.struct_passwd | None:
    try:
        return pwd.getpwuid(uid)
    except KeyError:
        return None


def _passwd_home(uid: int) -> Path | None:
    entry = _passwd_entry(uid)
    return Path(entry.pw_dir) if entry else None


def effective_user() -> RealUser:
    """Resolve the operator agent6 should act as.

    Under `sudo` (euid 0 + `SUDO_UID` set) this is the invoking user;
    otherwise it is the current process user.
    """
    euid = os.geteuid()
    sudo_uid = os.environ.get("SUDO_UID")
    if euid == 0 and sudo_uid and sudo_uid.isdigit():
        uid = int(sudo_uid)
        gid_raw = os.environ.get("SUDO_GID", "")
        gid = int(gid_raw) if gid_raw.isdigit() else uid
        # One passwd lookup, not three; and use the entry's existence (not "does
        # its home dir resolve") to decide whether we have a real name/home.
        entry = _passwd_entry(uid)
        name = os.environ.get("SUDO_USER", "") or (entry.pw_name if entry else str(uid))
        home = Path(entry.pw_dir) if entry else Path(os.environ.get("HOME", "/")).resolve()
        return RealUser(uid=uid, gid=gid, name=name, home=home, via_sudo=True)
    uid = os.getuid()
    gid = os.getgid()
    home_env = os.environ.get("HOME")
    home = Path(home_env) if home_env else (_passwd_home(uid) or Path("/"))
    try:
        name = pwd.getpwuid(uid).pw_name
    except KeyError:
        name = str(uid)
    return RealUser(uid=uid, gid=gid, name=name, home=home, via_sudo=False)


def _user_dir(user: RealUser | None, xdg_env: str, *home_parts: str) -> Path:
    """One rule for every agent6 user dir: `$<xdg_env>/agent6` (not through
    sudo, where root's XDG would be wrong) > `<real-user-home>/<home_parts>/agent6`."""
    user = user or effective_user()
    if not user.via_sudo:
        xdg = os.environ.get(xdg_env)
        if xdg:
            return Path(xdg) / "agent6"
    return user.home.joinpath(*home_parts) / "agent6"


def global_config_dir(user: RealUser | None = None) -> Path:
    """The agent6 global config directory: `$XDG_CONFIG_HOME/agent6` >
    `~/.config/agent6`."""
    return _user_dir(user, "XDG_CONFIG_HOME", ".config")


def global_config_path(user: RealUser | None = None) -> Path:
    return global_config_dir(user) / "config.toml"


def secrets_path(user: RealUser | None = None) -> Path:
    return global_config_dir(user) / "secrets.toml"


def ui_settings_path(user: RealUser | None = None) -> Path:
    """UI-only preferences (theme, etc.), a sibling of `config.toml`.

    Kept separate from the agent config on purpose: a theme is a machine-wide
    viewer preference, not agent behavior, so it never goes through the config
    schema or into the (shareable, per-repo) config layers.
    """
    return global_config_dir(user) / "ui.toml"


def cache_dir(user: RealUser | None = None) -> Path:
    """The agent6 user cache directory: `$XDG_CACHE_HOME/agent6` >
    `~/.cache/agent6`. Holds throwaway, regenerable data such as the provider
    model-list snapshots used for shell completion; safe to delete."""
    return _user_dir(user, "XDG_CACHE_HOME", ".cache")


def jail_cache_home(user: RealUser | None = None) -> Path:
    """The persistent HOME a jailed command gets (`<cache>/home`): under
    `hardened` and `none`, which have no private /tmp to put one in, and under
    `strict` with `[sandbox].home = "cache"`. Model-writable across runs, 0700,
    and never the operator's own home; `app.confine.check_jail_home` creates
    it and refuses a symlink or another user's directory at the path."""
    return cache_dir(user) / "home"


def data_dir(user: RealUser | None = None) -> Path:
    """The agent6 user data directory: `$XDG_DATA_HOME/agent6` >
    `~/.local/share/agent6`. Holds installed skills (`<data>/skills/<name>/`);
    unlike the cache it is authoritative and not regenerable."""
    return _user_dir(user, "XDG_DATA_HOME", ".local", "share")


# Per-repo agent6 state lives OUT of the workspace, under an XDG state base,
# namespaced by a per-repo id. Nothing the agent runs (a jailed command on its
# own cwd) can reach it, and a checkout never carries an `.agent6/` dir.
def state_base(user: RealUser | None = None) -> Path:
    """The agent6 state BASE directory (per-repo config + run state):
    `$XDG_STATE_HOME/agent6` > `~/.local/state/agent6`. Each repo gets
    `<base>/<repo-id>/`; the jail masks this base, and it is the one every
    run writes to."""
    return _user_dir(user, "XDG_STATE_HOME", ".local", "state")


def private_dirs() -> tuple[Path, ...]:
    """agent6 directories a jailed command must never see: the config dir
    (provider keys) and the state base (transcripts, memory, run history).
    ONE owner, because the jail masks them, the tool-mount scan refuses
    them, and the config validator rejects grants inside them.

    Not the data dir or the cache: data holds operator-INSTALLED skills, which
    the model is meant to use (a skill's bundled script has to be runnable),
    and the cache holds regenerable provider model lists. Neither is private,
    and hiding them only cost the skills case a way to work.

    Read per call: the XDG vars are per-process.
    """
    return (global_config_dir(), state_base())


def hidden_paths(extra: Iterable[Path]) -> tuple[Path, ...]:
    """Every tree hidden from a run: the operator's `[sandbox].hide_paths`
    plus :func:`private_dirs`.

    ONE owner, because two enforcers read it -- the jail masks these from a
    jailed command, and the in-process `Workspace` refuses them to the tools
    -- and a boundary they disagree about is a hole.
    """
    return (*extra, *private_dirs())


# A state dir names its workspace so `ls` sorts by location and a stale one is
# recognisable. The filesystem limit is 255 BYTES per component, and a path of
# CJK or emoji runs 3-4 bytes per character -- capping characters produced
# 271-byte names that failed to create with ENAMETOOLONG.
_ID_BYTES_MAX = 100
# Only the elided form needs a hash, and there it is the only thing separating
# two paths that elide alike; 12 hex chars = 48 bits, past casual brute force.
_ID_HASH_LEN = 12


def repo_id(repo_root: Path) -> str:
    """A directory name that identifies *repo_root*, and only it.

    `/` becomes `-`, and a trailing tag records which dashes were slashes:
    one bit per dash, most significant first, in hex. `/a/b/c` -> `a-b-c-3`,
    `/a/b-c` -> `a-b-c-2`, `/a-b-c` -> `a-b-c-0`. The mapping is
    reversible, so two different paths cannot produce the same id -- there is no
    hash in the common case and nothing to collide.

    Leading zeros need no sentinel: the name fixes how many dashes there are,
    so the tag's bit LENGTH is known and `01` cannot be read as `1`.

    Keyed on the RESOLVED path, so two checkouts never share state. Moving or
    renaming a checkout changes its id: its prior runs are simply not found
    from the new path.

    A path too long for one filesystem component is the exception: its middle
    is elided, the name stops being reversible, and a hash of the full path is
    what separates two that elide alike.
    """
    real = str(repo_root.resolve()).strip("/")
    flat = real.replace("/", "-")
    if len(flat.encode()) > _ID_BYTES_MAX:
        digest = hashlib.sha256(real.encode("utf-8")).hexdigest()[:_ID_HASH_LEN]
        head, tail = _ID_BYTES_MAX // 3, _ID_BYTES_MAX - _ID_BYTES_MAX // 3 - 2
        return f"{_head_bytes(flat, head)}--{_tail_bytes(flat, tail)}-{digest}"
    marks = "".join("1" if ch == "/" else "0" for ch in real if ch in "/-")
    tag = f"{int(marks or '0', 2):x}"
    # `/` flattens to nothing, and any sentinel word for it would be a legal
    # directory name too (`root-0` was both `/` and `/root`). The bare tag
    # cannot collide instead: every other id carries the joining dash.
    return f"{flat}-{tag}" if flat else tag


def repo_root_of_id(state_dir_name: str) -> Path | None:
    """The repo root a per-repo state-dir name encodes: `repo_id`'s inverse.

    None for a name `repo_id` cannot have produced (a stray directory in the
    state base, the elided-hash exception): the decoded candidate must
    re-encode to exactly *state_dir_name*, so a wrong read is impossible."""
    flat, _, tag = state_dir_name.rpartition("-")
    try:
        bits_val = int(tag, 16)
    except ValueError:
        return None
    dashes = flat.count("-")
    if bits_val >= (1 << dashes):
        return None
    marks = format(bits_val, f"0{dashes}b") if dashes else ""
    out: list[str] = []
    it = iter(marks)
    for ch in flat:
        out.append(("/" if next(it) == "1" else "-") if ch == "-" else ch)
    candidate = Path("/" + "".join(out))
    return candidate if repo_id(candidate) == state_dir_name else None


def _head_bytes(s: str, limit: int) -> str:
    """The longest prefix of *s* fitting in *limit* bytes, never splitting a
    character (a half-encoded one would not round-trip)."""
    return s.encode()[:limit].decode(errors="ignore")


def _tail_bytes(s: str, limit: int) -> str:
    """The longest suffix of *s* fitting in *limit* bytes, never splitting a
    character."""
    raw = s.encode()[-limit:]
    while raw:
        try:
            return raw.decode()
        except UnicodeDecodeError:
            raw = raw[1:]
    return ""


def project_root(start: Path) -> Path:
    """The repo *start* is inside, or *start* itself outside one.

    Walks for `.git` rather than asking git: this is on the path of every
    command, read-only ones included, and a subprocess per invocation is not.
    A linked worktree's `.git` is a file naming the repository's
    `.git/worktrees/<name>`, and the worktree is that repository's project;
    any other `.git` file (a submodule's, a pointer at a directory that is
    gone, a repository whose git dir sits outside its checkout) makes its
    directory a project of its own.

    No stop at `$HOME`: with `git init $HOME` every directory under it
    really IS one repo, and one repo has to be one project. Breaking the walk
    there gave each subdirectory its own state dir -- and its own
    `repo.lock`, while `git -C` still resolved every one of them to the
    same working tree, so two runs committed into it at once. That is exactly
    the interleaving the lock exists to prevent. Sharing state across a
    dotfiles repo is the operator's own choice; losing the lock is not.
    """
    root = checkout_root(start)
    if (root / ".git").is_file():
        git_dir = linked_worktree_git_dir(root)
        if git_dir is not None and git_dir.name == ".git":
            return git_dir.parent
    return root


def checkout_root(start: Path) -> Path:
    """The working tree *start* is inside (the nearest directory holding a
    `.git`, file or directory), or *start* itself outside one. A linked
    worktree is its own checkout; `project_root` maps it to its repository."""
    start = start.resolve()
    for candidate in (start, *start.parents):
        if (candidate / ".git").exists():
            return candidate
    return start


def linked_worktree_git_dir(root: Path) -> Path | None:
    """The repository git dir the linked worktree at *root* points into, or
    None for an ordinary checkout: its `.git` file (`gitdir: <repo>/.git/
    worktrees/<name>`) and that entry's `commondir`, resolved as git does.
    Both files sit in the workspace, writable by a jailed command under
    hardened: the pointer keys state and verifies a recorded grant, and never
    makes one."""
    pointer = root / ".git"
    if not pointer.is_file():
        return None
    try:
        text = pointer.read_text(encoding="utf-8", errors="replace").strip()
        if not text.startswith("gitdir:"):
            return None
        admin = Path(text[len("gitdir:") :].strip())
        if not admin.is_absolute():
            admin = root / admin
        common = Path((admin / "commondir").read_text(encoding="utf-8").strip())
        if not common.is_absolute():
            common = admin / common
        return common.resolve()
    except OSError:
        return None


def state_dir(repo_root: Path) -> Path:
    """The per-repo agent6 state directory (`<base>/<repo-id>`).

    Keyed on the PROJECT (`project_root`), not on where the operator is
    standing: keyed on the cwd, every cross-session feature (`sessions`,
    `resume`, `read_session`, memory) would silently find an empty project
    from any subdirectory.
    """
    return state_base() / repo_id(project_root(repo_root))


def repo_config_path(repo_root: Path) -> Path:
    """The per-repo config file (`<state_dir>/config.toml`), out of the repo."""
    return state_dir(repo_root) / "config.toml"


def is_root() -> bool:
    return os.geteuid() == 0


def root_optin_enabled(cli_flag: bool) -> bool:
    """True when the operator has explicitly allowed running as root:
    `--allow-root`, or `AGENT6_ALLOW_ROOT=1` (that value alone, as
    `AGENT6_DANGEROUSLY_DISABLE_SANDBOX` takes)."""
    return cli_flag or os.environ.get(_ALLOW_ROOT_ENV) == "1"


def mkdir_for_real_user(path: Path, user: RealUser | None = None) -> None:
    """Create *path* (with any missing ancestors), chowning what was created
    back to the real user.

    Under `sudo` a bare `mkdir(parents=True)` creates the missing ancestry
    as root, and a root-owned base blocks every later non-root sibling (the
    second repo's state dir, the next skill's install dir). The handover covers
    the topmost directory this call created, recursively -- pre-existing
    directories are never rechowned -- and falls back to *path* itself when
    nothing was missing.
    """
    missing: list[Path] = []
    cur = path
    while not cur.exists():
        missing.append(cur)
        if cur.parent == cur:
            break
        cur = cur.parent
    # The missing ancestry is created 0700: these are agent6's own single-user
    # dirs (state, config, data), and a non-traversable base shields the files
    # inside whatever the umask. A pre-existing dir is never re-chmodded.
    for d in reversed(missing):
        d.mkdir(mode=0o700, exist_ok=True)
    path.mkdir(parents=True, exist_ok=True)
    chown_to_real_user(missing[-1] if missing else path, user)


def chown_to_real_user(path: Path, user: RealUser | None = None) -> None:
    """Recursively `chown` *path* back to the real operator.

    No-op unless the process is root and was launched through sudo.

    The walk names every target relative to an open directory fd and never
    follows a link, so no walked component can be swapped between the walk
    and the call: a jailed command holds RW on some of these trees, and root
    resolving a swapped parent would chown anything on the host. The tree
    root itself is one `lchown` by path (links never followed).
    Best-effort: permission errors are swallowed (the file is still usable by
    root), and we never weaken perms to compensate.
    """
    if os.geteuid() != 0:
        return
    user = user or effective_user()
    if not user.via_sudo:
        return
    with contextlib.suppress(OSError):
        os.lchown(path, user.uid, user.gid)
    if path.is_symlink() or not path.is_dir():
        return
    for _dirpath, dirnames, filenames, dir_fd in os.fwalk(path, follow_symlinks=False):
        for name in (*dirnames, *filenames):
            with contextlib.suppress(OSError):
                os.chown(name, user.uid, user.gid, dir_fd=dir_fd, follow_symlinks=False)
