# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Git operations with hard safety invariants.

The destructive operations (push, force, history rewrite) are not exposed as a
code path here AT ALL -- there is nothing to refuse at runtime because nothing
spells them (pinned by test_git_ops_never_spells_a_destructive_verb). The one
sanctioned exception is force_delete_squash_merged_branch. The config can
*loosen* benign options (auto-stash, branch-per-run) and never these.
"""

from __future__ import annotations

import contextlib
import os
import re
import shutil
import subprocess
import tempfile
import textwrap
from collections import Counter
from collections.abc import Collection, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from agent6.types import CommandResult


class GitError(Exception):
    """Generic git failure."""


# Upper bound on any single git invocation. Local ops finish in well under a
# second; this only fires on a pathological hang (stuck filesystem, held lock).
_GIT_TIMEOUT_S = 120.0

# How long a timed-out git gets to exit on SIGTERM before SIGKILL. Its TERM
# handler only has to unlink its lockfiles, so this is generous.
_GIT_TERM_GRACE_S = 5.0


@dataclass(frozen=True, slots=True)
class GitStatus:
    """The worktree against HEAD. `is_clean` means no tracked file is modified
    and no untracked file exists outside the caller's `exclude` set (a run's
    `untracked_at_start`); `modified_count` alone is the operator's uncommitted
    work, which is what a start gate reads."""

    branch: str
    head_sha: str
    is_clean: bool
    untracked_count: int
    modified_count: int


@dataclass(frozen=True, slots=True)
class CommitIdentity:
    """Resolved name/email + provenance trailer used for commits this run.

    `name` and `email` are populated from `[git.commit]` overrides when set,
    otherwise left as None to mean "let git's own config resolution decide".
    `verify_git_identity` ensures that when both are None the project's git
    config has a usable identity before any commit is attempted. `trailer` is
    a rendered git trailer line (see :func:`render_commit_trailer`), appended
    once per commit.
    """

    name: str | None = None
    email: str | None = None
    trailer: str | None = None

    @property
    def has_override(self) -> bool:
        return bool(self.name or self.email or self.trailer)


def render_commit_trailer(fmt: str, *, models: Sequence[str]) -> str | None:
    """The `[git.commit].trailer` format string as a concrete trailer line, or
    None when unset. {model} names the model(s) that wrote the code, first-seen
    order (the primary worker first), ", "-joined and deduplicated; the model
    that wrote a commit MESSAGE never appears. The config validator pins the
    placeholder set and the "Key: value" shape."""
    if not fmt:
        return None
    return fmt.format(model=", ".join(dict.fromkeys(m for m in models if m)))


def verify_git_identity(path: Path, identity: CommitIdentity) -> tuple[str, str]:
    """Resolve the effective author identity, or raise GitError.

    Returns `(name, email)` that future commits will use. Order of
    precedence per field:

      1. The `[git.commit]` override (`identity.name` / `identity.email`).
      2. `git config user.name` / `git config user.email` in this repo.

    If after both steps either field is empty, we refuse to start. This is
    deliberately strict: silently committing as a missing/auto-generated
    identity is the kind of thing a user only notices weeks later when they
    `git log --author`.
    """
    name = identity.name or _run(path, "config", "user.name", check=False).stdout.strip()
    email = identity.email or _run(path, "config", "user.email", check=False).stdout.strip()
    missing: list[str] = []
    if not name:
        missing.append("user.name")
    if not email:
        missing.append("user.email")
    if missing:
        joined = " and ".join(missing)
        raise GitError(
            f"Git identity not configured: {joined} is empty. Either run\n"
            f"    git -C {path} config user.name 'Your Name'\n"
            f"    git -C {path} config user.email 'you@example.com'\n"
            f"or set [git.commit].name / [git.commit].email in your agent6 config."
        )
    return name, email


_SLUG_RE = re.compile(r"[^a-z0-9]+")


def _git() -> str:
    git = shutil.which("git")
    if git is None:
        raise GitError("git executable not found on PATH")
    return git


# Always-on hardening: neutralize repo-config keys that would otherwise run a
# repo-controlled command on the HOST (outside the jail) during agent6's own git
# operations. `-c` has the highest precedence, overriding `.git/config`.
# `core.fsmonitor` fires a command on every index refresh (status/add/commit);
# `diff.external` fires one on `git diff` (review/diff); `commit.gpgsign` fires
# the configured `gpg.program` (arbitrary host command) on every commit. All are
# pure overrides with no correctness cost here: fsmonitor is a perf cache, an
# empty diff.external uses git's builtin diff, and agent6 has no signing feature
# (its per-step auto-commits are unsigned by design; the operator signs at the
# end). The edit tools already refuse writes into `.git` under protect_git, but a
# repo cloned with a pre-poisoned `.git/config` would otherwise execute its
# payload the first time agent6 ran git here.
# Content-semantic drivers a commit/merge legitimately runs -- clean/smudge
# `filter.*` and `merge.*.driver` -- are the same RCE class but have no blanket
# `-c` off switch, so `_repo_driver_overrides` neutralizes each by NAME, gated
# by `run_repo_filters` (the Git-LFS opt-in, since LFS uses exactly these).
_GIT_HARDENING: tuple[str, ...] = (
    "-c",
    "core.fsmonitor=false",
    "-c",
    "diff.external=",
    "-c",
    "commit.gpgsign=false",
)

# Whether the repo's own `.git/hooks/*` run during agent6's git ops (notably the
# per-step auto-commit). Default false -- a repo hook is repo-controlled HOST
# code, so honoring it on agent6's commit is a host-RCE vector for an adversarial
# repo. Set once from `git.run_repo_hooks` at run/review startup. A module-level
# dict (mutated, not rebound) keeps the process-wide policy without a `global`
# statement.
_hook_policy: dict[str, bool] = {"honor_repo_hooks": False}


def set_repo_hook_policy(honor: bool) -> None:
    """Configure whether agent6's own git ops fire the repo's `.git/hooks/*`."""
    _hook_policy["honor_repo_hooks"] = honor


# Provider-key env var names (the configured `api_key_env`s) to drop from the
# environment agent6's git subprocesses inherit. git never needs a provider
# key, and a git subprocess -- a credential helper, a content driver we could
# not neutralize -- should not be handed one. Set once at startup from the
# config; empty when keys live only in secrets.toml (never in the environment).
# A module-level set, mutated not rebound, like _hook_policy.
_provider_key_env: set[str] = set()


def set_provider_key_env(names: Iterable[str]) -> None:
    """The provider-key env var names `_run` strips from git's environment."""
    _provider_key_env.clear()
    _provider_key_env.update(n for n in names if n)


# Whether the repo's own content drivers -- `filter.<n>.clean/smudge/process`
# and `merge.<n>.driver` -- run during agent6's git ops (the auto-commit's
# `git add`, the chain merge's `merge-tree`). Default false: a driver defined
# in `.git/config` is a host command, an RCE vector for a cloned poisoned repo.
# Git-LFS is why they exist, so honoring them is the LFS opt-in. There is no
# blanket `-c` off switch, so `_run` neutralizes each repo-defined driver by
# NAME.
_filter_policy: dict[str, bool] = {"honor_repo_filters": False}


def set_repo_filter_policy(honor: bool) -> None:
    """Configure whether agent6's git ops honor the repo's own content drivers
    (`filter.*`, `merge.*.driver`); false neutralizes each by name."""
    _filter_policy["honor_repo_filters"] = honor


# The config keys that name a driver command. Scoped to the repo's own config
# and the files IT includes (see `--local --includes` below): a Git-LFS filter
# the operator installed in ~/.gitconfig is theirs and trusted; the untrusted
# surface is the repo's `.git/config` (which a jailed command can write under
# hardened, and which a cloned repo brings pre-poisoned) and anything it pulls
# in via `[include]`.
_DRIVER_KEY_RE = r"^(filter\..*\.(clean|smudge|process)|merge\..*\.driver)$"


def _repo_driver_overrides(cwd: Path) -> tuple[str, ...]:
    """`-c` flags that blank every repo-defined content driver, or () when the
    policy honors them or the repo defines none.

    Read the driver NAMES from the repo's own config (reading names runs
    nothing) and emit an empty override per name: an empty `filter.<n>.clean`
    is a pass-through, and an empty `merge.<n>.driver` makes the merge report a
    conflict rather than run the command -- both stop the host command without
    a blanket switch git does not provide. Re-read per call so a driver written
    mid-run (hardened, where a jailed command can write `.git/config`) is caught
    too."""
    if _filter_policy["honor_repo_filters"]:
        return ()
    try:
        proc = subprocess.run(
            [
                _git(),
                "-C",
                str(cwd),
                "config",
                "--local",
                # Follow the repo's OWN includes: `--local` alone stops at
                # `.git/config`, but a git OP follows an `[include]` there to a
                # repo-controlled file, so a driver hidden behind one would run
                # while this enumeration missed it. `--includes` matches what
                # the op sees; `--local` still keeps the operator's trusted
                # global `~/.gitconfig` (and its includes) out of scope.
                "--includes",
                "--name-only",
                "--get-regexp",
                _DRIVER_KEY_RE,
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return ()
    # Dedup by name first: a filter with both clean and smudge yields two keys
    # for one driver, and the last `-c` wins anyway.
    filters: set[str] = set()
    merges: set[str] = set()
    for key in proc.stdout.split():
        if key.startswith("filter."):
            filters.add(key[len("filter.") : key.rindex(".")])
        elif key.startswith("merge."):
            merges.add(key[len("merge.") : key.rindex(".")])
    overrides: list[str] = []
    for name in sorted(filters):
        overrides += [
            "-c",
            f"filter.{name}.clean=",
            "-c",
            f"filter.{name}.smudge=",
            "-c",
            f"filter.{name}.process=",
        ]
    for name in sorted(merges):
        overrides += ["-c", f"merge.{name}.driver="]
    return tuple(overrides)


# Flags that force git's builtin diff/show renderer so a poisoned `.git/config`
# cannot run a host command: `--no-ext-diff` disables the `diff.external` driver,
# `--no-textconv` the per-file `diff.<d>.textconv` driver (neither is covered by
# the `-c` overrides above). Single source of truth so no diff/show call site
# drifts. Place AFTER the subcommand, alongside `git_hardening_flags()` before it.
DIFF_SHOW_SAFETY_FLAGS: tuple[str, ...] = ("--no-ext-diff", "--no-textconv")


def git_hardening_flags(cwd: Path) -> tuple[str, ...]:
    """The `-c` overrides every agent6 git invocation must carry: the fixed set
    (_GIT_HARDENING), the hooks path unless the policy honors repo hooks, and a
    blank override per content driver *cwd* defines (`_repo_driver_overrides`).

    Public so the callers that shell out to git outside this module (`agent6
    review` / `sessions diff` / `ask` collectors) carry the same hardening;
    place them BEFORE the subcommand. Diff/show callers also add
    DIFF_SHOW_SAFETY_FLAGS after the subcommand.
    """
    # /dev/null is not a directory, so git finds (and runs) no hooks there.
    hooks = () if _hook_policy["honor_repo_hooks"] else ("-c", "core.hooksPath=/dev/null")
    return (*_GIT_HARDENING, *hooks, *_repo_driver_overrides(cwd))


def _run(
    cwd: Path,
    *args: str,
    check: bool = True,
    env_extra: dict[str, str] | None = None,
    stdin_text: str | None = None,
) -> CommandResult:
    # GIT_TERMINAL_PROMPT=0: a git op that would otherwise block on a
    # username/password prompt (a network remote without cached creds) fails
    # fast instead of hanging with no output. Local ops are unaffected.
    # LC_ALL=C: git translates its human-readable output, and the bystander
    # rescue reads one of those sentences to learn which stash it dropped.
    env = {**os.environ, "GIT_TERMINAL_PROMPT": "0", "LC_ALL": "C", **(env_extra or {})}
    for name in _provider_key_env:
        env.pop(name, None)
    hardening = git_hardening_flags(cwd)
    # A poisoned `.git/config` reaches a host command two ways on a diff/show:
    # `diff.external` and per-file `diff.<d>.textconv`. The `-c` overrides above
    # cover neither cleanly (and git 2.53 dies rc=128 on the empty `diff.external`
    # override), so force the builtin renderer with DIFF_SHOW_SAFETY_FLAGS.
    argv = list(args)
    if argv and argv[0] in ("diff", "show"):
        argv[1:1] = DIFF_SHOW_SAFETY_FLAGS
    # Blank the repo's own content drivers on EVERY op, not a guessed list of
    # driver-running subcommands: enumerating which git verbs run a clean/smudge
    # /merge driver is enumerating badness, and missing one reopens the RCE.
    full_argv = (_git(), *hardening, *argv)
    index_lock = cwd / ".git" / "index.lock"
    lock_preexisted = index_lock.exists()
    proc = subprocess.Popen(
        full_argv,
        cwd=cwd,
        stdin=subprocess.PIPE if stdin_text is not None else subprocess.DEVNULL,
        # Bytes, decoded lossily below: git diff/show emit raw file bytes,
        # so a changed non-UTF-8 text file (latin-1 has no NULs, so git does
        # not classify it binary) would make a strict text=True decode raise
        # UnicodeDecodeError mid-run. Filenames are core.quotePath-escaped to
        # ASCII, and shas/porcelain status are ASCII, so replacement only
        # ever touches diff/show content.
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
    )
    try:
        stdin_bytes = stdin_text.encode() if stdin_text is not None else None
        out, err = proc.communicate(input=stdin_bytes, timeout=_GIT_TIMEOUT_S)
    except subprocess.TimeoutExpired as exc:
        # SIGTERM first: git's TERM handler unlinks its own lockfiles
        # (index.lock included), so a terminated child cleans up after itself
        # and a lock still on disk afterwards is a concurrent git's.
        proc.terminate()
        try:
            proc.communicate(timeout=_GIT_TERM_GRACE_S)
        except subprocess.TimeoutExpired:
            # TERM ignored (wedged uninterruptible): SIGKILL skips git's
            # cleanup, so clear the lock -- ONLY when it appeared under this
            # child. One that predates the spawn belongs to a concurrent git
            # process (operator shell, another lane), and deleting it would
            # break git's index mutual exclusion.
            proc.kill()
            proc.communicate()
            if not lock_preexisted:
                with contextlib.suppress(OSError):
                    index_lock.unlink(missing_ok=True)
        raise GitError(
            f"git {' '.join(args)} timed out after {_GIT_TIMEOUT_S:.0f}s"
            " (a stuck filesystem or a held .git/index.lock?)"
        ) from exc
    result = CommandResult(
        argv=full_argv,
        returncode=proc.returncode,
        stdout=out.decode(errors="replace"),
        stderr=err.decode(errors="replace"),
        duration_s=0.0,
    )
    if check and not result.ok:
        # Surface stdout too. `git commit` writes its informational
        # output (including "nothing to commit, working tree clean", pre-
        # commit hook output, and most user-facing messages) to STDOUT,
        # not stderr; stderr alone is empty for most commit failures.
        stderr_msg = result.stderr.strip()
        stdout_msg = result.stdout.strip()
        if stderr_msg and stdout_msg:
            detail = f"{stderr_msg} | stdout: {stdout_msg}"
        else:
            detail = stderr_msg or stdout_msg or f"exit {result.returncode}"
        raise GitError(f"git {' '.join(args)} failed: {detail}")
    return result


def is_git_repo(path: Path) -> bool:
    res = _run(path, "rev-parse", "--is-inside-work-tree", check=False)
    return res.ok and res.stdout.strip() == "true"


def toplevel(path: Path) -> Path | None:
    """The enclosing work tree's root directory, or None outside a git repo
    (or inside `.git` itself, where git reports no toplevel)."""
    res = _run(path, "rev-parse", "--show-toplevel", check=False)
    if not res.ok:
        return None
    text = res.stdout.strip()
    return Path(text) if text else None


def paths_dirty(path: Path, rel_paths: tuple[str, ...]) -> bool:
    """True iff any of `rel_paths` has uncommitted changes (untracked,
    modified, or staged) versus HEAD, i.e. a path-limited commit of just those
    paths would record something. Unlike whole-tree `status().is_clean`, this
    ignores unrelated dirt elsewhere in the worktree."""
    if not rel_paths:
        return False
    res = _run(path, "status", "--porcelain", "--", *rel_paths, check=False)
    return bool(res.stdout.strip())


def _porcelain_entries(path: Path) -> list[tuple[str, str]]:
    """`(code, repo-root-relative path)` per changed or untracked file, from
    `status --porcelain=v1 -z` (NUL-separated, so any filename round-trips;
    a rename's second field, the source path, is skipped)."""
    res = _run(path, "status", "--porcelain=v1", "-z", "--untracked-files=all", check=False)
    fields = res.stdout.split("\0")
    out: list[tuple[str, str]] = []
    i = 0
    while i < len(fields):
        entry = fields[i]
        i += 1
        if len(entry) < 4:
            continue
        code, rel = entry[:2], entry[3:]
        out.append((code, rel))
        if code[0] in "RC":
            i += 1
    return out


def modified_paths(path: Path) -> list[str]:
    """Tracked files with uncommitted changes (modified, staged, or deleted),
    repo-root-relative. Untracked files are never listed: they are the
    operator's, outside a run's commits."""
    return [rel for code, rel in _porcelain_entries(path) if code != "??"]


def untracked_paths(path: Path) -> frozenset[str]:
    """Every untracked, non-ignored file, repo-root-relative. Taken once at run
    start as the run's `untracked_at_start`: those files are the operator's,
    so chain commits and dirty checks leave them out (`exclude`)."""
    return frozenset(rel for code, rel in _porcelain_entries(path) if code == "??")


def status(path: Path, *, exclude: Collection[str] = ()) -> GitStatus:
    """The worktree against HEAD; untracked files in *exclude* (a run's
    `untracked_at_start`) are not counted."""
    if not is_git_repo(path):
        raise GitError(f"Not a git repository: {path}")
    branch_res = _run(path, "rev-parse", "--abbrev-ref", "HEAD", check=False)
    if branch_res.ok:
        branch = branch_res.stdout.strip()
    else:
        # Unborn HEAD (freshly `git init`, no commits yet): `rev-parse HEAD`
        # fails, but `branch --show-current` still reports the checked-out branch
        # name. Without this, every agent6 entry point that loads the repo
        # summary crashes in a brand-new repo.
        branch = _run(path, "branch", "--show-current", check=False).stdout.strip()
    head_res = _run(path, "rev-parse", "HEAD", check=False)
    head_sha = head_res.stdout.strip() if head_res.ok else ""
    untracked = 0
    modified = 0
    for code, rel in _porcelain_entries(path):
        if code == "??":
            if rel not in exclude:
                untracked += 1
        else:
            modified += 1
    return GitStatus(
        branch=branch,
        head_sha=head_sha,
        is_clean=(untracked == 0 and modified == 0),
        untracked_count=untracked,
        modified_count=modified,
    )


def stash_tracked_changes(path: Path, message: str) -> None:
    """Stash the tracked files' uncommitted changes under *message*. Untracked
    files stay where they are: a run leaves them out of its commits, so
    nothing needs moving them aside."""
    _run(path, "stash", "push", "--message", message)


def auto_stash_message(session_id: str) -> str:
    """The auto-stash identity: the run pushes with this message and the
    finalizer finds the stash BY it -- never by position, since stash@{0} may
    be a stash someone else pushed while the run was running."""
    return f"agent6 auto-stash before run {session_id}"


@dataclass(frozen=True, slots=True)
class StashEntry:
    """One `git stash list` entry: its position at lookup time (`ref`,
    e.g. `stash@{1}`, for operator-facing hints only) and its immutable
    commit (`sha`). Anything that mutates restores by sha: the position
    shifts the moment anyone pushes or drops a stash."""

    ref: str
    sha: str


def find_stash(path: Path, message: str) -> StashEntry | None:
    """The newest stash pushed with exactly *message*, or None.
    `git stash push -m MSG` records `On <branch>: MSG` and ':' cannot
    appear in a ref name, so anchoring `": MSG"` at the end matches the
    whole message -- lane run ids are ordinal (`…-l1`, `…-l10`), so one
    message can be a prefix of another."""
    res = _run(path, "stash", "list", "--format=%gd%x09%H%x09%gs", check=False)
    for line in res.stdout.splitlines():
        ref, _, rest = line.partition("\t")
        sha, _, subject = rest.partition("\t")
        if subject.endswith(f": {message}"):
            return StashEntry(ref=ref, sha=sha)
    return None


# `git stash drop` names the commit it removed: "Dropped stash@{0} (<sha>)".
_DROPPED_SHA_RE = re.compile(r"^Dropped .*\(([0-9a-f]{7,64})\)", re.MULTILINE)


def restore_stash(path: Path, stash: StashEntry) -> bool:
    """Apply *stash* back onto the working tree BY SHA -- a stash@{N} recorded
    earlier applies whatever sits at that position NOW, which is the wrong
    stash the moment another one was pushed. On a clean apply, drop the entry.
    On conflict (or any non-zero apply), leave everything in place so the
    user's work is never lost, and return False. We never `reset --hard` to
    undo a conflicted apply (refused), so a conflict leaves the markers for
    the user to resolve with their stash still intact. Raises GitError when a
    raced drop took a concurrent stash and putting it back failed; the apply
    itself has landed by then, and the message carries the recovery command."""
    if not _run(path, "stash", "apply", stash.sha, check=False).ok:
        return False
    _drop_by_sha(path, stash.sha)
    return True


def _drop_by_sha(path: Path, sha: str) -> None:
    """Drop the stash entry whose commit is *sha*, putting back a bystander we
    take by mistake.

    `git stash drop` addresses an entry by POSITION and refuses a sha outright
    ("is not a stash reference"), so the position has to be re-resolved from the
    list -- and a stash pushed in between shifts every position, aiming the drop
    at someone else's entry. git names the commit it dropped, so check it: one
    that is not ours is stored straight back under its own subject (position is
    not identity, so it returns at the top of the stack). Ours then stays
    listed; re-resolving to drop it again would race the same way, and leaking
    our own stash beats taking a second bystander.
    """
    listed = _run(path, "stash", "list", "--format=%gd%x09%H", check=False)
    ref = ""
    for line in listed.stdout.splitlines():
        entry_ref, _, entry_sha = line.partition("\t")
        if entry_sha == sha:
            ref = entry_ref
            break
    if not ref:
        return
    dropped = _DROPPED_SHA_RE.search(_run(path, "stash", "drop", ref, check=False).stdout)
    if dropped is None:
        return  # no drop happened, or git stopped naming what it dropped
    taken = dropped.group(1)
    # Prefix either way: git prints the full oid today, but an abbreviated one
    # must not read as a stranger's stash and get stored back on top of agent6's own.
    if sha.startswith(taken) or taken.startswith(sha):
        return
    # A stash commit's own subject is the "On <branch>: <message>" the reflog
    # showed, so the restored entry reads exactly as it did before.
    subject = _run(path, "log", "-1", "--format=%s", taken, check=False).stdout.strip()
    subject = subject or "restored by agent6"
    store = _run(path, "stash", "store", "-m", subject, taken, check=False)
    if not store.ok:
        # The bystander's entry is already gone from the list; its commit
        # still exists under `taken`, so the message carries the recovery.
        detail = store.stderr.strip() or store.stdout.strip() or f"exit {store.returncode}"
        raise GitError(
            f"a stash pushed concurrently ({subject!r}) was taken by a raced drop and "
            f"putting it back failed ({detail}); restore it with:\n"
            f"    git stash store -m {subject!r} {taken}"
        )


def branch_exists(path: Path, name: str) -> bool:
    """True if a local branch *name* exists."""
    return _run(path, "rev-parse", "--verify", "--quiet", f"refs/heads/{name}", check=False).ok


def valid_branch_name(name: str) -> bool:
    """True if *name* is usable as a git branch name (`git check-ref-format
    --branch`), the strictest of the ref roles a run id fills.

    A run's id becomes a branch (`agent6/<id>`) and a chain ref
    (`refs/agent6/<id>/head`); a value git's ref grammar rejects (a space, any of
    `~^:?*[\\`, `..`, `@{`, a leading `-`/`.`, a trailing `.`/`.lock`, ...) makes
    every auto/final commit's `update-ref` fail. A valid branch name is safe in
    both roles, so this is the one check `validate_explicit_session_id` needs. A
    pure string check with no repo side effects; run from "/" (always exists),
    like `clone_repo`."""
    return _run(Path("/"), "check-ref-format", "--branch", name, check=False).ok


def list_run_branches(path: Path) -> tuple[str, ...]:
    """Local branches under the `agent6/` namespace (run branches), sorted."""
    res = _run(path, "for-each-ref", "--format=%(refname:short)", "refs/heads/agent6/", check=False)
    return tuple(b for b in res.stdout.splitlines() if b.strip())


def run_branch_tips(path: Path) -> dict[str, str]:
    """{run branch: tip sha} for every `agent6/` branch, one git call: the
    listings' merged/unmerged mark needs every tip, and a per-row rev-parse
    would put ~50 subprocesses on the hub's poll."""
    res = _run(
        path,
        "for-each-ref",
        "--format=%(refname:short) %(objectname)",
        "refs/heads/agent6/",
        check=False,
    )
    out: dict[str, str] = {}
    for line in res.stdout.splitlines():
        branch, _, sha = line.partition(" ")
        if branch and sha:
            out[branch] = sha
    return out


def is_ancestor(path: Path, maybe_ancestor: str, ref: str) -> bool:
    """True if *maybe_ancestor* is reachable from *ref* (`git merge-base
    --is-ancestor`). Used to tell a reachable-merged run branch (an ancestor of its
    base) from a squash-merged one (content in the base, but not reachable)."""
    return _run(path, "merge-base", "--is-ancestor", maybe_ancestor, ref, check=False).ok


def delete_branch_if_merged(path: Path, branch: str) -> bool:
    """Delete *branch* with `git branch -d`, the SAFE delete: git refuses unless the
    branch is reachable-merged into the current HEAD (or its upstream). Returns True
    if deleted, False if git refused -- a squash-merged or genuinely unmerged branch,
    since neither is reachable. Never `branch -D` here; see
    `force_delete_squash_merged_branch` for the one operator-gated exception."""
    return _run(path, "branch", "-d", branch, check=False).ok


def branch_tip_sha(path: Path, branch: str) -> str | None:
    """The commit sha a branch points at, or None if it does not resolve."""
    res = _run(path, "rev-parse", "--verify", "--quiet", f"refs/heads/{branch}", check=False)
    sha = res.stdout.strip()
    return sha or None


def merge_stamp_holds(path: Path, run_branch: str, merged_tip: str) -> bool:
    """Does a run's merged stamp still describe its branch? A resumed run keeps
    committing on its branch under a PRIOR leg's stamp: "merged" holds only
    while the branch still points at the merged tip (the comparison
    `sessions prune` trusts). A gone branch (auto_prune), unreadable git, or a
    pre-`tip` stamp keeps the claim."""
    if not merged_tip or not run_branch:
        return True
    tip = None
    with contextlib.suppress(GitError):
        tip = branch_tip_sha(path, run_branch)
    return tip is None or tip == merged_tip


def force_delete_squash_merged_branch(path: Path, branch: str) -> bool:
    """`git branch -D` a run branch, the ONE sanctioned force-delete in agent6.

    `git branch -d` refuses a squash-merged branch because its commits are not
    reachable from the base (the squash collapsed them into one commit ON the
    base), even though the branch's content is safely in that base commit. So a
    default `sessions prune` can never remove a squash-merged branch, and the whole
    run -> merge -> prune lifecycle leaves the branch behind.

    This is the operator's explicit `sessions prune --delete-squashed` opt-in, only
    for a branch the manifest CONFIRMS was squash-merged into an existing base
    (its content is in that base commit; the individual per-step commits were
    collapsed anyway and survive in the reflog until GC). Operator-initiated,
    fixed argv, content-preserving. Returns True if deleted."""
    return _run(path, "branch", "-D", branch, check=False).ok


def create_branch(path: Path, name: str, *, start_point: str | None = None) -> None:
    """Create *name* and check it out, or just check it out if it already exists.

    *start_point* (a branch/sha) is where a NEW branch is cut from; None means
    the current HEAD. Idempotent: an existing branch is only checked out, never
    moved (that would be a force/rewrite, which is refused), so re-running or
    resuming a run reuses the run's branch."""
    existing = _run(path, "branch", "--list", name, check=False)
    if existing.ok and existing.stdout.strip():
        _run(path, "checkout", name)
    elif start_point:
        _run(path, "checkout", "-b", name, start_point)
    else:
        _run(path, "checkout", "-b", name)


_CHAIN_NS = "refs/agent6"
_CHAIN_KIND = "head"


def machine_chain_ref_for(machine_id: str) -> str:
    """The chain ref a machine's states continue from (`chain_ref_for`, under
    the machine namespace)."""
    return chain_ref_for(f"machine-{machine_id}")


# Every visible agent6 branch sits under this prefix: a run's `agent6/<id>`,
# a machine's `agent6/machine-<id>`.
BRANCH_PREFIX = "agent6/"


def run_branch_for(session_id: str) -> str:
    """The visible branch a run's chain advances (`[git].branch_per_run`),
    named for the session that cut it: `agent6/<id>`."""
    return f"{BRANCH_PREFIX}{session_id}"


def machine_branch_for(machine_id: str) -> str:
    """The visible branch a machine's `mode="run"` states land on."""
    return f"{BRANCH_PREFIX}machine-{machine_id}"


def chain_ref_for(session_id: str) -> str:
    """The ref holding a session's commit chain: `refs/agent6/<id>/head`.

    The session id is a NAMESPACE, not the ref itself, which is what keeps the
    short name `agent6/<id>` the visible branch's alone: git resolves
    `refs/<name>` before `refs/heads/<name>`, so a ref sitting AT
    `refs/agent6/<id>` shadows the branch the operator means, and every
    `git log|diff|checkout agent6/<id>` reports the name as ambiguous. The kind
    under the id follows `refs/pull/<n>/head`; it also leaves room for a
    second per-session ref, which a ref at the id itself could not (git refuses
    a ref inside a ref).
    """
    return f"{_CHAIN_NS}/{session_id}/{_CHAIN_KIND}"


def set_ref(path: Path, ref: str, sha: str) -> None:
    """Point *ref* at *sha* (plain `update-ref`, no checkout). For agent6's own
    refs (:func:`chain_ref_for`); branches go through create_branch_at."""
    _run(path, "update-ref", ref, sha)


def delete_ref(path: Path, ref: str) -> None:
    """Delete *ref* if it exists (`update-ref -d`); missing is a no-op."""
    _run(path, "update-ref", "-d", ref, check=False)


def list_chain_refs(path: Path) -> tuple[tuple[str, str], ...]:
    """(session_id, sha) for every chain ref, sorted by id. Globbed on the
    kind, so a future per-session ref beside it is not mistaken for a chain."""
    pattern = f"{_CHAIN_NS}/*/{_CHAIN_KIND}"
    out = _run(path, "for-each-ref", "--format=%(refname)%00%(objectname)", pattern).stdout
    rows: list[tuple[str, str]] = []
    for line in out.splitlines():
        ref, _, sha = line.partition("\x00")
        if sha and ref.startswith(f"{_CHAIN_NS}/") and ref.endswith(f"/{_CHAIN_KIND}"):
            rows.append((ref[len(_CHAIN_NS) + 1 : -(len(_CHAIN_KIND) + 1)], sha))
    return tuple(sorted(rows))


def checkout_detached(path: Path, rev: str) -> None:
    """Detached checkout of *rev*: for agent6-OWNED clones (a lane workspace
    cut at the coordinator's chain tip), never the operator's checkout."""
    _run(path, "checkout", "-q", "--detach", rev)


def create_branch_at(path: Path, name: str, sha: str) -> None:
    """Create branch *name* pointing at *sha* WITHOUT checking it out.

    Additive only (`git branch <name> <sha>`): it never touches HEAD or the
    working tree, so `agent6 fork` can cut the new run's branch at a historical
    sha while the operator's checkout stays put. No-op if *name* already points
    at *sha*; raises `GitError` if it exists pointing elsewhere (we never move
    a branch -- that would be a force/rewrite, which is refused)."""
    existing = _run(path, "rev-parse", "--verify", "--quiet", f"refs/heads/{name}", check=False)
    if existing.ok and existing.stdout.strip():
        if existing.stdout.strip() == sha:
            return
        raise GitError(
            f"branch {name!r} already exists at {existing.stdout.strip()[:12]}, not {sha[:12]}; "
            "refusing to move it"
        )
    _run(path, "branch", name, sha)


def init_repo(path: Path) -> None:
    """`git init` a new repository at *path*. Creating a repo is not a push /
    force / history-rewrite, so it is outside the refusal set."""
    _run(path, "init")


def clone_repo(origin: Path, dest: Path) -> None:
    """`git clone` *origin* into *dest*. Both are plain filesystem paths, so
    git's local-clone optimization applies automatically: hardlinks on the same
    filesystem, falling back to a copy across devices -- no flag needed.

    cwd is "/" (always exists), not *origin*: both argv paths are absolutized
    here so the anchor is irrelevant, and a missing *origin* must fail as a
    GitError from git itself, not a raw FileNotFoundError from subprocess's
    chdir before git ever runs."""
    _run(Path("/"), "clone", str(origin.absolute()), str(dest.absolute()))


def unignored(path: Path, candidates: tuple[str, ...]) -> tuple[str, ...]:
    """Return the subset of repo-relative *candidates* that git does NOT ignore.

    Used so the `init` git-setup offer commits only the trackable scaffold
    (AGENTS.md, .gitignore) and not files the just-written .gitignore covers
    (e.g. the per-repo config under the ignored agent6 dir)."""
    if not candidates:
        return ()
    # check-ignore prints the ignored inputs (one per line) and exits 1 when
    # none match, both are fine, only stdout is read. The "--" stops a path
    # that begins with "-" from being parsed as a git flag.
    res = _run(path, "check-ignore", "--", *candidates, check=False)
    ignored = {line.strip() for line in res.stdout.splitlines() if line.strip()}
    return tuple(c for c in candidates if c not in ignored)


def commit_all(
    path: Path,
    message: str,
    *,
    trailers: dict[str, str] | None = None,
    identity: CommitIdentity | None = None,
) -> str:
    """Stage everything and commit. Returns the new HEAD sha.

    `identity` lets the caller override the author/committer name+email and
    append a `Co-authored-by:` trailer. When `identity` is None the commit
    uses the project's existing git config identity, callers should have
    already validated that via `verify_git_identity` at startup.
    """
    _run(path, "add", "-A")
    return _commit(path, message, trailers=trailers, identity=identity)


def commit_paths(
    path: Path,
    message: str,
    paths: tuple[str, ...],
    *,
    trailers: dict[str, str] | None = None,
    identity: CommitIdentity | None = None,
) -> str:
    """Stage only `paths` (repo-relative) and commit JUST those paths. Returns
    the new HEAD sha.

    The commit is path-limited (`git commit -- <paths>`), so unrelated
    changes the user already STAGED stay staged and uncommitted, and unrelated
    WIP in the worktree is never swept in. Used by `agent6 init`'s scaffold
    commit, which must not fold the user's in-progress work into it.
    """
    if not paths:
        raise GitError("commit_paths requires at least one path")
    _run(path, "add", "--", *paths)
    return _commit(path, message, trailers=trailers, identity=identity, only_paths=paths)


def _identity_env(identity: CommitIdentity | None) -> dict[str, str] | None:
    """Author + committer env for a commit-creating git invocation, or None to fall
    back to the repo's configured identity."""
    if identity is None:
        return None
    env: dict[str, str] = {}
    if identity.name:
        env["GIT_AUTHOR_NAME"] = env["GIT_COMMITTER_NAME"] = identity.name
    if identity.email:
        env["GIT_AUTHOR_EMAIL"] = env["GIT_COMMITTER_EMAIL"] = identity.email
    return env or None


def _full_message(
    message: str, trailers: dict[str, str] | None, identity: CommitIdentity | None
) -> str:
    """The commit message with the identity trailer and any extra trailers
    appended once."""
    merged = dict(trailers or {})
    if identity is not None and identity.trailer and identity.trailer not in message:
        key, _, value = identity.trailer.partition(": ")
        merged[key] = value
    if not merged:
        return message
    trailer_lines = "\n".join(f"{k}: {v}" for k, v in merged.items())
    return f"{message}\n\n{trailer_lines}"


def chain_tip(path: Path, ref: str) -> str | None:
    """Current sha of *ref* (any ref name), or None when it does not exist."""
    res = _run(path, "rev-parse", "--verify", "--quiet", f"{ref}^{{commit}}", check=False)
    sha = res.stdout.strip()
    return sha if res.returncode == 0 and sha else None


def worktree_tree(path: Path, seed: str | None, exclude: Collection[str]) -> str:
    """Tree sha of the worktree's CURRENT content, staged into a temp index;
    the shared index is never read or written.

    *seed* (the commit the tree will be diffed or parented against; None in an
    unborn repo) pre-populates the index with that commit's tree before
    `add -A`: ignore rules apply only to UNTRACKED files, so an empty index
    made `add -A` skip tracked-but-ignored files and every chain commit
    silently dropped them, which a later merge turned into deletions. New
    ignored files stay out, and a file deleted from the worktree still leaves
    the tree, exactly as `add -A` behaves on the real index.

    *exclude* (repo-root-relative paths, the run's `untracked_at_start`) never
    enters the tree: `:(top,exclude,literal)` pathspecs, read from a file so
    the set's size and any filename are fine."""
    tmp = Path(tempfile.mkdtemp(prefix="agent6-chain-"))
    env = {"GIT_INDEX_FILE": str(tmp / "index")}
    try:
        if seed is not None:
            _run(path, "read-tree", seed, env_extra=env)
        if exclude:
            spec = tmp / "pathspec"
            spec.write_bytes(
                b"\0".join(
                    [b":/", *(f":(top,exclude,literal){rel}".encode() for rel in sorted(exclude))]
                )
            )
            _run(
                path,
                "add",
                "-A",
                f"--pathspec-from-file={spec}",
                "--pathspec-file-nul",
                env_extra=env,
            )
        else:
            _run(path, "add", "-A", env_extra=env)
        return _run(path, "write-tree", env_extra=env).stdout.strip()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# git's well-known empty tree: the diff base when a chain has no commits yet.
_EMPTY_TREE = "4b825dc642cb6eb9a060e54bf8d69288fbee4904"


def worktree_matches(path: Path, ref: str, paths: Collection[str]) -> bool:
    """True when the worktree's content of *paths* equals *ref*'s
    (`git diff --quiet <ref> -- <paths>`); the shared index is not consulted."""
    res = _run(path, "diff", "--quiet", ref, "--", *sorted(paths), check=False)
    return res.returncode == 0


def chain_dirty(
    path: Path, ref: str, fallback_parent: str | None, *, exclude: Collection[str] = ()
) -> bool:
    """True when the worktree's content (minus *exclude*) differs from the
    chain tip's tree (*ref*, else *fallback_parent*, else the empty tree in an
    unborn repo). Raises GitError outside a repo -- callers treat that as
    clean."""
    base = chain_tip(path, ref) or fallback_parent
    base_tree = _run(path, "rev-parse", f"{base}^{{tree}}").stdout.strip() if base else _EMPTY_TREE
    return base_tree != worktree_tree(path, base, exclude)


def chain_dirty_paths(
    path: Path,
    ref: str,
    fallback_parent: str | None,
    limit: int,
    *,
    exclude: Collection[str] = (),
) -> list[str]:
    """Paths whose worktree content (minus *exclude*) differs from the chain
    tip's tree, capped at *limit* (an unborn chain diffs against the empty
    tree)."""
    base = chain_tip(path, ref) or fallback_parent
    base_tree = _run(path, "rev-parse", f"{base}^{{tree}}").stdout.strip() if base else _EMPTY_TREE
    return tree_diff_paths(path, base_tree, worktree_tree(path, base, exclude))[:limit]


def tree_diff_paths(path: Path, old_tree: str, new_tree: str) -> list[str]:
    """Paths whose content differs between two tree shas."""
    out = _run(path, "diff-tree", "-r", "--name-only", old_tree, new_tree).stdout
    return [line for line in out.splitlines() if line]


def chain_commit(
    path: Path,
    message: str,
    *,
    ref: str,
    fallback_parent: str | None,
    trailers: dict[str, str] | None = None,
    identity: CommitIdentity | None = None,
    also_branch: str | None = None,
    exclude: Collection[str] = (),
) -> str | None:
    """Record the worktree's current content (minus *exclude*, the run's
    `untracked_at_start`) on the agent's own commit chain, touching neither
    HEAD, the operator's index, nor any checkout.

    Stages everything into a TEMP index, writes the tree, and `commit-tree`s
    it parented on *ref*'s current value -- the ref itself is the chain state,
    so resume and concurrent runs compose without bookkeeping. When the ref
    does not exist yet the parent is *fallback_parent* (HEAD at run start;
    None = a root commit in an unborn repo). Advances *ref*
    (:func:`chain_ref_for`, the gc anchor) and, when *also_branch* is set,
    `refs/heads/<also_branch>` -- a plain ref move, never a checkout. Returns
    the new sha, or None when the tree is identical to the parent's (nothing
    to record).
    """
    parent = chain_tip(path, ref) or fallback_parent
    tree = worktree_tree(path, parent, exclude)
    parent_args: list[str] = []
    if parent is not None:
        if _run(path, "rev-parse", f"{parent}^{{tree}}").stdout.strip() == tree:
            return None
        parent_args = ["-p", parent]
    sha = _run(
        path,
        "commit-tree",
        tree,
        *parent_args,
        "-m",
        _full_message(message, trailers, identity),
        env_extra=_identity_env(identity),
    ).stdout.strip()
    _run(path, "update-ref", ref, sha)
    if also_branch:
        _run(path, "update-ref", f"refs/heads/{also_branch}", sha)
    return sha


def chain_merge(
    path: Path,
    merge_rev: str,
    message: str,
    *,
    ref: str,
    fallback_parent: str | None = None,
    identity: CommitIdentity | None = None,
    also_branch: str | None = None,
) -> str | None:
    """Merge *merge_rev* into the chain at *ref* without touching HEAD, the
    shared index, or any checkout (`git merge-tree --write-tree` + a two-parent
    `commit-tree`). Advances *ref* (and *also_branch*) to the result and syncs
    the worktree from the old tip's tree to the merged tree, so the running
    agent sees the lane's files. An unborn ref merges onto *fallback_parent*
    (HEAD at run start); a *merge_rev* that descends from the tip
    fast-forwards instead of stacking an empty merge commit. Returns the new
    (or already-containing old) tip; None on textual conflicts (the chain and
    worktree are left untouched).
    """
    ours = chain_tip(path, ref) or fallback_parent
    if ours is None:
        raise GitError(f"chain ref {ref} does not exist and no fallback parent was given")
    theirs = _run(path, "rev-parse", "--verify", f"{merge_rev}^{{commit}}").stdout.strip()
    if _run(path, "merge-base", "--is-ancestor", theirs, ours, check=False).returncode == 0:
        return ours
    if _run(path, "merge-base", "--is-ancestor", ours, theirs, check=False).returncode == 0:
        sha = theirs
    else:
        res = _run(path, "merge-tree", "--write-tree", ours, theirs, check=False)
        if res.returncode != 0:
            return None
        tree = res.stdout.strip().splitlines()[0]
        sha = _run(
            path,
            "commit-tree",
            tree,
            "-p",
            ours,
            "-p",
            theirs,
            "-m",
            _full_message(message, None, identity),
            env_extra=_identity_env(identity),
        ).stdout.strip()
    _run(path, "update-ref", ref, sha)
    if also_branch:
        _run(path, "update-ref", f"refs/heads/{also_branch}", sha)
    sync_worktree(path, ours, sha)
    return sha


def sync_worktree(path: Path, from_rev: str, to_rev: str) -> None:
    """Update worktree files from *from_rev*'s tree to *to_rev*'s via a temp
    index (two-tree `read-tree -m -u`); HEAD and the shared index stay
    untouched. The worktree must currently match *from_rev*'s tree -- the
    chain invariant after a chain commit."""
    tmp = Path(tempfile.mkdtemp(prefix="agent6-chain-"))
    env = {"GIT_INDEX_FILE": str(tmp / "index")}
    try:
        _run(path, "read-tree", from_rev, env_extra=env)
        _run(path, "read-tree", "-m", "-u", from_rev, to_rev, env_extra=env)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _commit(
    path: Path,
    message: str,
    *,
    trailers: dict[str, str] | None,
    identity: CommitIdentity | None,
    only_paths: tuple[str, ...] | None = None,
) -> str:
    env_extra = _identity_env(identity)
    full_message = _full_message(message, trailers, identity)
    argv = ["commit", "-m", full_message]
    if only_paths is not None:
        # Path-limited commit: record only these paths (from the worktree),
        # disregarding anything else already staged; with no only_paths the
        # whole index is committed (commit_all).
        argv.extend(["--", *only_paths])
    _run(path, *argv, env_extra=env_extra)
    return _run(path, "rev-parse", "HEAD").stdout.strip()


@dataclass(frozen=True, slots=True)
class MergeResult:
    """Outcome of merging a run branch into a target. On conflict the merge is
    undone so the working tree is left clean (merged_sha empty, conflicts listed)."""

    merged_sha: str
    conflicted: bool
    conflicts: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class CommitRow:
    """One commit on a run branch (oldest-first), for listing + squash-condensing."""

    sha: str
    subject: str
    message: str  # full %B


def fetch_branch(path: Path, remote_path: Path, refspec: str) -> None:
    """`git fetch <remote_path> <refspec>` into *path*, e.g. `"b:b"` to land a
    same-named local branch from another repo without adding a remote."""
    _run(path, "fetch", str(remote_path), refspec)


def plumb_merge(
    path: Path,
    target: str,
    merge_rev: str,
    *,
    strategy: str,
    message: str | None = None,
    identity: CommitIdentity | None = None,
) -> MergeResult:
    """Land *merge_rev* on branch *target* with plumbing only: `merge-tree` +
    `commit-tree` + a compare-and-swap ref update. No checkout and no
    clean-tree requirement -- the operator's worktree is never the medium, so
    a worktree that (as after every run) carries the run's own work does not
    block the landing. When *target* is the checked-out branch, index entries
    the merge changed and the operator did not are brought forward so `git
    status` stays truthful.

    *strategy*: "merge" (two-parent commit), "squash" (one single-parent
    commit), "ff" (the ref moves to *merge_rev*; raises when not
    fast-forwardable). A *merge_rev* the target already contains is a clean
    no-op returning the unchanged tip. On conflict nothing moves and the
    conflicted paths are reported."""
    ref = f"refs/heads/{target}"
    ours = _run(path, "rev-parse", "--verify", f"{ref}^{{commit}}").stdout.strip()
    theirs = _run(path, "rev-parse", "--verify", f"{merge_rev}^{{commit}}").stdout.strip()
    if _run(path, "merge-base", "--is-ancestor", theirs, ours, check=False).returncode == 0:
        return MergeResult(ours, False, ())
    ff_able = _run(path, "merge-base", "--is-ancestor", ours, theirs, check=False).returncode == 0
    if strategy == "ff":
        if not ff_able:
            raise GitError(f"{target!r} has moved; a fast-forward to {merge_rev!r} is impossible")
        sha = theirs
    else:
        if ff_able:
            tree = _run(path, "rev-parse", f"{theirs}^{{tree}}").stdout.strip()
        else:
            res = _run(path, "merge-tree", "--write-tree", "--name-only", ours, theirs, check=False)
            lines = res.stdout.splitlines()
            if res.returncode == 1:
                # The tree oid, the conflicted paths, a blank line, then git's
                # informational messages ("Auto-merging f", "CONFLICT (content) ...").
                paths: list[str] = []
                for line in lines[1:]:
                    if not line.strip():
                        break
                    paths.append(line)
                return MergeResult("", True, tuple(paths))
            if res.returncode != 0 or not lines:
                raise GitError(f"merge-tree failed: {res.stderr.strip() or 'exit'}")
            tree = lines[0].strip()
        if tree == _run(path, "rev-parse", f"{ours}^{{tree}}").stdout.strip():
            return MergeResult(ours, False, ())  # content-identical: nothing to land
        text = message or f"Merge {merge_rev}"
        trailer = identity.trailer if identity else None
        if trailer and trailer not in text:
            text = f"{text}\n\n{trailer}"
        parent_args = ["-p", ours] if strategy == "squash" else ["-p", ours, "-p", theirs]
        sha = _run(
            path,
            "commit-tree",
            tree,
            *parent_args,
            "-m",
            text,
            env_extra=_identity_env(identity),
        ).stdout.strip()
    # Compare-and-swap: refuses (GitError) if the target moved concurrently.
    _run(path, "update-ref", ref, sha, ours)
    _bring_index_forward(path, target, ours, sha)
    return MergeResult(sha, False, ())


def _bring_index_forward(path: Path, target: str, old_tip: str, new_tip: str) -> None:
    """After moving the CHECKED-OUT branch's ref from *old_tip* to *new_tip*
    without a checkout, bring the shared index and the worktree forward for
    the paths the move changed -- each only where it still matches *old_tip*,
    so anything the operator staged or edited themselves is left exactly as
    they had it. Without the index half, `git status` shows a phantom staged
    reversal of the landed work; without the worktree half, a merge landed
    onto a reverted checkout would leave the files behind. A worktree that
    already holds the new content (as after every run) needs and gets no
    writes."""
    head = _run(path, "rev-parse", "--abbrev-ref", "HEAD", check=False).stdout.strip()
    if head != target:
        return
    changed = _run(path, "diff-tree", "-r", "--no-renames", "-z", old_tip, new_tip).stdout
    staged = {
        entry.split("\t", 1)[1]: entry.split("\t", 1)[0].split()
        for entry in _run(path, "ls-files", "--stage", "-z").stdout.split("\x00")
        if "\t" in entry
    }  # path -> [mode, sha, stage]
    updates: list[str] = []
    records = changed.split("\x00")
    i = 0
    while i + 1 < len(records):
        meta, rel = records[i], records[i + 1]
        i += 2
        if not meta.startswith(":"):
            continue
        old_mode, new_mode, old_sha, new_sha, _status = meta[1:].split(" ")[:5]
        entry = staged.get(rel)
        if (entry is None and old_mode == "000000") or (
            entry is not None and entry[0] == old_mode and entry[1] == old_sha
        ):
            # mode 000000 removes the entry (a merge-side deletion).
            updates.append(f"{new_mode} {new_sha}\t{rel}")
        _bring_worktree_file_forward(path, rel, old_mode, old_sha, new_mode, new_sha)
    if updates:
        _run(path, "update-index", "-z", "--index-info", stdin_text="\x00".join(updates) + "\x00")


def _bring_worktree_file_forward(
    path: Path, rel: str, old_mode: str, old_sha: str, new_mode: str, new_sha: str
) -> None:
    """Move ONE worktree file from the old tip's content to the new tip's,
    only when it still matches the old tip (absent counts as matching a
    deletion or a not-yet-added path); regular files only -- symlinks and
    submodule pointers are left to the operator."""
    file = path / rel
    if new_mode in ("120000", "160000") or old_mode in ("120000", "160000"):
        return
    try:
        if old_mode == "000000":
            current_matches = not file.exists()
        elif not file.is_file() or file.is_symlink():
            current_matches = False
        else:
            current_matches = (
                _run(path, "hash-object", "--", rel, check=False).stdout.strip() == old_sha
            )
        if not current_matches:
            return
        if new_mode == "000000":
            file.unlink(missing_ok=True)
            return
        file.parent.mkdir(parents=True, exist_ok=True)
        # Bytes straight from git to the file: _run decodes lossily (str), which
        # would corrupt a binary blob.
        with file.open("wb") as out:
            subprocess.run(
                [_git(), *git_hardening_flags(path), "cat-file", "blob", new_sha],
                cwd=path,
                stdout=out,
                stderr=subprocess.DEVNULL,
                check=True,
                timeout=_GIT_TIMEOUT_S,
            )
        if new_mode == "100755":
            file.chmod(file.stat().st_mode | 0o111)
    except (GitError, OSError, subprocess.SubprocessError):
        return  # best-effort: an unwritable path leaves truthful dirt, never a crash


def list_run_commits(path: Path, base_sha: str, run_branch: str) -> tuple[CommitRow, ...]:
    """Commits on *run_branch* since *base_sha*, oldest first."""
    # NUL-separate commits (-z): a commit body can contain any byte except NUL, so
    # \x1f/\x1e separators in a body would corrupt records/fields. Within a record,
    # split the \x1f fields at most twice so the body (last field) keeps any \x1f.
    fmt = "%H%x1f%s%x1f%B"
    res = _run(
        path, "log", "-z", "--reverse", f"--format={fmt}", f"{base_sha}..{run_branch}", check=False
    )
    if not res.ok:
        return ()
    rows: list[CommitRow] = []
    for rec in res.stdout.split("\x00"):
        if not rec.strip():
            continue
        fields = rec.split("\x1f", 2)
        if len(fields) >= 3:
            rows.append(CommitRow(sha=fields[0].strip(), subject=fields[1], message=fields[2]))
    return tuple(rows)


_ITER_SUBJECT_RE = re.compile(r"^agent6 iter \d+:\s*", re.IGNORECASE)


def condense_commit_message(rows: tuple[CommitRow, ...], *, subject: str) -> str:
    """Fold per-step commits into one readable message, so a squash reads as a
    single authored commit, not a squashed series.

    *subject* is the run's task (the headline). The body lists the distinct,
    de-noised per-step subjects (the `agent6 iter N:` prefix and checkpoint
    noise stripped). The provenance trailer is the commit emitter's job
    (identity.trailer), not this message's."""
    bullets: list[str] = []
    seen: set[str] = set()
    for row in rows:
        s = _ITER_SUBJECT_RE.sub("", row.subject).strip()
        if not s or s.lower().startswith("checkpoint") or s.lower() in seen:
            continue
        seen.add(s.lower())
        bullets.append(s)
    task = _ITER_SUBJECT_RE.sub("", subject).strip()
    headline = _headline_subject(task) or (bullets[0] if bullets else "agent6 run")
    parts = [headline]
    # If the subject truncated the task, wrap the full task into the body so
    # nothing is lost (git never wraps the subject line itself).
    full = " ".join(task.split())
    if full and full != headline:
        parts.append("")
        parts.extend(textwrap.wrap(full, width=72))
    if bullets:
        parts.append("")
        parts.extend(f"- {b}" for b in bullets)
    return "\n".join(parts)


_SUBJECT_LIMIT = 72  # git's soft subject cap; conventional tooling truncates past it


def _is_testish(p: str) -> bool:
    parts = PurePosixPath(p).parts
    name = parts[-1] if parts else ""
    return parts[:1] == ("tests",) or name.startswith("test_") or name == "conftest.py"


def _is_docish(p: str) -> bool:
    pp = PurePosixPath(p)
    return pp.suffix.lower() in (".md", ".rst") or pp.parts[:1] == ("docs",)


def _conventional_scope(paths: Sequence[str]) -> str:
    """The one common area the change touches, or "" when there is none: the
    package dir under `src/<pkg>/` (the module stem for a file directly under
    the package), else a second-level dir every path shares."""
    parts = [PurePosixPath(p).parts for p in paths if p]
    if not parts:
        return ""
    src_pkgs = [pp for pp in parts if len(pp) >= 3 and pp[0] == "src"]
    if src_pkgs:
        names = {pp[2] if len(pp) > 3 else str(PurePosixPath(pp[2]).stem) for pp in src_pkgs}
        return names.pop() if len(names) == 1 else ""
    tops = {pp[0] for pp in parts}
    if len(tops) != 1:
        return ""
    seconds = {pp[1] for pp in parts if len(pp) >= 3}
    return seconds.pop() if len(seconds) == 1 else ""


def conventional_commit_subject(changes: Sequence[tuple[str, str]], *, summary: str) -> str:
    """A Conventional Commits subject from `(status, path)` pairs, without a
    model call: all-tests -> `test`, all-docs -> `docs`, any added file ->
    `feat`, else `fix` (`chore` when nothing changed). Scope is the one
    common area (:func:`_conventional_scope`); the subject is *summary* with
    its head lowercased and any trailing period stripped, capped at 72."""
    paths = [p for _, p in changes]
    if not paths:
        ctype = "chore"
    elif all(_is_testish(p) for p in paths):
        ctype = "test"
    elif all(_is_docish(p) for p in paths):
        ctype = "docs"
    elif any(status.startswith("A") for status, _ in changes):
        ctype = "feat"
    else:
        ctype = "fix"
    scope = _conventional_scope(paths)
    head = f"{ctype}({scope}): " if scope else f"{ctype}: "
    subject = " ".join(summary.split()).rstrip(".")
    subject = (subject[:1].lower() + subject[1:]) if subject else "update"
    return (head + subject)[:_SUBJECT_LIMIT]


def worktree_name_status(path: Path) -> tuple[tuple[str, str], ...]:
    """`(status, path)` pairs for every pending change (``status
    --porcelain`), untracked reported as `A``: the conventional-subject
    deriver's input at checkpoint time."""
    res = _run(path, "status", "--porcelain", check=False)
    pairs: list[tuple[str, str]] = []
    for line in res.stdout.splitlines():
        if len(line) < 4:
            continue
        code = line[:2].strip() or "M"
        pairs.append(("A" if code in ("??", "A", "AM") else code[:1], line[3:].strip()))
    return tuple(pairs)


def range_name_status(path: Path, base: str, head: str) -> tuple[tuple[str, str], ...]:
    """`(status, path)` pairs for `base..head`: the conventional-subject
    deriver's input at squash time."""
    res = _run(path, "diff", "--name-status", f"{base}..{head}", check=False)
    pairs: list[tuple[str, str]] = []
    for line in res.stdout.splitlines():
        cols = line.split("\t")
        if len(cols) >= 2:
            pairs.append((cols[0][:1], cols[-1]))
    return tuple(pairs)


def _headline_subject(task: str, *, limit: int = _SUBJECT_LIMIT) -> str:
    """A short commit subject derived from the task's first clause: its first
    line, up to the first sentence end, whitespace-collapsed, capped at *limit*
    (an ellipsis marks a truncation). A run's whole task text as the subject
    reads as one unwrapped 180-char line that every git tool clips; the full
    task is wrapped into the body by the caller when this truncates it."""
    first_line = next((ln for ln in task.splitlines() if ln.strip()), "")
    match = re.search(r"[.!?](?:\s|$)", first_line)
    clause = first_line[: match.start()] if match else first_line
    clause = " ".join(clause.split())
    if len(clause) <= limit:
        return clause
    return clause[: limit - 1].rstrip() + "…"


def recent_log(path: Path, n: int = 20) -> str:
    res = _run(path, "log", f"-n{n}", "--oneline", check=False)
    return res.stdout if res.ok else ""


def tracked_files(path: Path) -> tuple[str, ...]:
    """Return the list of repo-tracked files via `git ls-files`.

    POSIX-style separators, sorted by `git`'s own order. Empty tuple
    outside a git repo or when ls-files fails - callers must treat this
    as "no map available" rather than "empty repo".
    """
    res = _run(path, "ls-files", "-z", check=False)
    if not res.ok:
        return ()
    return tuple(p for p in res.stdout.split("\x00") if p)


def co_change_pairs(
    path: Path,
    *,
    n_commits: int = 200,
    min_pair_count: int = 2,
    max_pairs: int = 30,
) -> list[tuple[str, str, int]]:
    """Mine git history for co-change file pairs.

    Walks the last *n_commits* commits, groups changed files per commit,
    and returns the top *max_pairs* most-frequent unordered (fileA, fileB)
    pairs that co-changed in at least *min_pair_count* commits. Each
    tuple is (file_a, file_b, count). Sorted by count descending, ties
    broken alphabetically.

    A cheap prior for the planner: files that repeatedly change together
    hint that an edit to one implicates the other. Returns an empty list if git history is too
    shallow to find any qualifying pairs (e.g. the fresh-clone bench
    case with --depth=1).

    Skips merge commits (--no-merges) so multi-parent diffs don't
    artificially inflate co-change frequencies.
    """
    res = _run(
        path,
        "log",
        f"-n{n_commits}",
        "--no-merges",
        "--name-only",
        "--pretty=format:%x00",
        check=False,
    )
    if not res.ok:
        return []
    # Output is groups of (NUL-separator, blank line, file paths...) per
    # commit. Split on NUL to get per-commit file lists.
    pair_counter: Counter[tuple[str, str]] = Counter()
    for chunk in res.stdout.split("\x00"):
        # --name-only emits only paths and blank lines; blanks are gone above.
        files = sorted({line.strip() for line in chunk.strip().splitlines() if line.strip()})
        if len(files) < 2:
            continue
        for i in range(len(files)):
            for j in range(i + 1, len(files)):
                pair_counter[(files[i], files[j])] += 1
    qualifying = [(a, b, c) for (a, b), c in pair_counter.items() if c >= min_pair_count]
    qualifying.sort(key=lambda t: (-t[2], t[0], t[1]))
    return qualifying[:max_pairs]


def diff_since(path: Path, base_sha: str, *, exclude: Collection[str] = ()) -> str:
    # `git diff <base>` only considers tracked content. Newly created files
    # from a worker edit are untracked at this point (commit_all stages and
    # commits later, after the reviewer is consulted), so a plain diff would
    # be empty and the reviewer would falsely conclude "the worker did
    # nothing". Register untracked files with `git add -N` (intent-to-add)
    # so they show up as additions in the diff. -N doesn't add content to the
    # index; commit_all's later `git add -A` overwrites the intent entries.
    #
    # *exclude* (the run's `untracked_at_start`) stays OUT of both the
    # intent-add and the diff: the chain already excludes those files, and a
    # review diff that showed them as the run's own additions had a panel
    # order their removal -- the model deleted an operator's untracked file.
    #
    # The intent-add runs against a TEMP COPY of the index (the chain's own
    # temp-index pattern): `-N` entries left in the real index survive the
    # run (chain commits never consume them) and turned a later ref-plumbing
    # merge into a staged-deletion artifact (`DA` in status) that read as
    # dirt and blocked the next run.
    specs = [f":(top,exclude,literal){rel}" for rel in sorted(exclude)]
    tmp = Path(tempfile.mkdtemp(prefix="agent6-review-diff-"))
    try:
        index_copy = tmp / "index"
        real_index = path / ".git" / "index"
        if real_index.is_file():
            shutil.copyfile(real_index, index_copy)
        env = {"GIT_INDEX_FILE": str(index_copy)}
        _run(path, "add", "-N", "--", ".", *specs, check=False, env_extra=env)
        res = _run(path, "diff", base_sha, "--", ".", *specs, check=False, env_extra=env)
        return res.stdout if res.ok else ""
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def diff_range(path: Path, base_sha: str, ref: str) -> str:
    """The committed diff `base_sha..ref` introduces, captured. "" when the
    range is unresolvable (a pruned branch, a bad sha) -- read-only, never blocks
    a caller comparing several candidates. Goes through `_run` so it carries the
    same host-RCE hardening + builtin-renderer flags every git diff here does."""
    res = _run(path, "diff", f"{base_sha}..{ref}", check=False)
    return res.stdout if res.ok else ""


def commit_diff(path: Path, sha: str, *, max_bytes: int = 16384) -> str:
    """The patch a single commit introduced (`git show <sha>`), or "" on error.

    Read-only, used to surface "what the worker just changed" to a live viewer.
    `--format=` keeps it to just the diff (no commit message). Truncated to
    `max_bytes` here so callers don't materialize an unbounded diff in memory."""
    res = _run(path, "show", "--format=", "--no-color", sha, "--", ".", check=False)
    if not res.ok:
        return ""
    return res.stdout[:max_bytes]


def show_commit(path: Path, sha: str, *, max_bytes: int = 16_384) -> str:
    """Return `git show --stat --patch <sha>` truncated to *max_bytes* for telemetry.

    Best-effort: returns empty string on error rather than raising.
    """
    res = _run(path, "show", "--stat", "--patch", sha, check=False)
    if not res.ok:
        return ""
    out = res.stdout
    if len(out) > max_bytes:
        return out[:max_bytes] + f"\n... [truncated, full size {len(out)} bytes]"
    return out
