# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""`agent6 init` command (scaffold a workspace + offer git setup)."""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

from agent6.config import Config, ConfigError
from agent6.config.layer import load_effective, repo_config_path_for
from agent6.errors import OperatorError
from agent6.git_ops import (
    GitError,
    commit_paths,
    init_repo,
    is_git_repo,
    paths_dirty,
    unignored,
)
from agent6.init import _ask, init_workspace
from agent6.paths import chown_to_real_user
from agent6.ui.cli._common import error

_SCAFFOLD_COMMIT_MESSAGE = "chore: scaffold agent6 config"


def _digest(path: Path) -> str | None:
    """The file's content hash, or None when it does not exist."""
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def _scaffold_rel_paths(root: Path, created: tuple[Path, ...]) -> tuple[str, ...]:
    """The repo-relative scaffold files git would record. The per-repo config
    lives out of the workspace under the state dir, so it is never a candidate
    here; filter to paths under root defensively, then unignored() drops
    anything the just-written .gitignore covers so we never `git add -f`.

    A path with nothing pending is dropped too, so the commit line names what
    the commit holds: init leaves an existing AGENTS.md alone, and listing it
    told the operator it had been committed."""
    candidates = unignored(
        root,
        tuple(
            str(p.relative_to(root)) for p in created if p.exists() and root in p.resolve().parents
        ),
    )
    return tuple(rel for rel in candidates if paths_dirty(root, (rel,)))


def _offer_git_setup(root: Path, created: tuple[Path, ...], *, interactive: bool) -> None:
    """Leave the repo ready for the advertised `agent6 run`: in a non-repo,
    offer to `git init` + commit the scaffold (non-interactively just print a
    note); in an existing repo, offer to commit the uncommitted scaffold, which
    would otherwise make `agent6 run` refuse on a dirty tree."""
    if is_git_repo(root):
        _offer_scaffold_commit(root, created, interactive=interactive)
        return
    print()
    if not interactive:
        print(f"Note: {root} is not a git repository; `agent6 run`/`plan` need one.")
        print('  Run: git init && git add -A && git commit -m "initial commit"')
        return
    if not _ask("This directory is not a git repository. Initialise one now?", default=True):
        print("  Skipped. `agent6 run` needs a repo; run `git init` here first.")
        return
    try:
        init_repo(root)
    except GitError as exc:
        print(f"  git init failed: {exc}")
        return
    print("  created: .git/  (git init)")
    rel = _scaffold_rel_paths(root, created)
    if not rel:
        print("  (nothing to commit; the created files are all gitignored)")
        return
    if not _ask("Commit the files agent6 just created?", default=True):
        print(f"  Not committed. When ready: git add {' '.join(rel)} && git commit")
        return
    try:
        commit_paths(root, _SCAFFOLD_COMMIT_MESSAGE, rel)
        print(f"  committed the agent6 scaffold ({', '.join(rel)})")
    except GitError as exc:
        # Most likely a missing git identity, actionable, not fatal.
        print(f"  commit skipped: {exc}")
        print("  Set git user.name / user.email, then: git add -A && git commit")


def _offer_scaffold_commit(root: Path, created: tuple[Path, ...], *, interactive: bool) -> None:
    """*root* is already a git repo, so the scaffold init wrote sits uncommitted
    and `agent6 run` refuses a dirty tree. Offer to commit it (auto-yes when
    non-interactive, i.e. --yes); when declined or the commit fails, print the
    exact command so the advertised next step works."""
    rel = _scaffold_rel_paths(root, created)
    if not rel:
        return
    try:
        if not paths_dirty(root, rel):
            # Scaffold already committed; nothing to commit for these paths.
            # (Whole-tree is_clean would false-trigger on unrelated WIP and then
            # fail the path-limited commit with "nothing to commit".)
            return
    except GitError:
        return
    manual = f"git add {' '.join(rel)} && git commit -m '{_SCAFFOLD_COMMIT_MESSAGE}'"
    print()
    if interactive and not _ask(
        "Commit the agent6 scaffold now (`agent6 run` needs a clean tree)?", default=True
    ):
        print(f"  Not committed. Before `agent6 run`: {manual}")
        return
    try:
        commit_paths(root, _SCAFFOLD_COMMIT_MESSAGE, rel)
    except GitError as exc:
        print(f"  commit failed: {exc}")
        print(f"  Commit it yourself before `agent6 run`: {manual}")
        return
    print(f"  committed the agent6 scaffold ({', '.join(rel)})")


def _print_next_steps(cwd: Path, config_path: Path | None) -> None:
    """The commands still between this repo and a first run: connect and model
    only while the effective config lacks a provider or a worker model."""
    try:
        cfg: Config | None = load_effective(cwd, config_path).config
    except ConfigError:
        cfg = None
    print()
    print("Next:")
    if cfg is None or not cfg.providers:
        print("  agent6 connect                 # add a provider + API key (global)")
    if cfg is None or cfg.models.resolve("worker") is None:
        print("  agent6 model worker <provider> <model>   # pick your worker model")
    print("  agent6 config show             # audit the effective config")
    gated = cfg is not None and bool(cfg.workflow.verify_command)
    print('  agent6 run "<task>"' + ("" if gated else "            # verify is inferred per run"))


def _cmd_init(*, ecosystem: str, assume_yes: bool = False, config_path: Path | None = None) -> int:
    cwd = Path.cwd()
    target = repo_config_path_for(cwd)
    if not assume_yes and not sys.stdin.isatty():
        # Refuse rather than silently take every default and write files:
        # consent comes from a TTY or --yes.
        error("no input. stdin is not a TTY; re-run with --yes to accept every default.")
        return 2
    interactive = not assume_yes
    # A scaffold path init leaves untouched is the operator's file, whether or
    # not this is a repo yet: committing it by path would put THEIR work in
    # agent6's scaffold commit, so it is excluded and reported instead.
    scaffold_all = (cwd / "AGENTS.md", cwd / ".gitignore")
    before = {p: _digest(p) for p in scaffold_all}
    try:
        rc = init_workspace(
            cwd,
            ecosystem=ecosystem,
            repo_config_target=target,
            interactive=interactive,
            config_path=config_path,
        )
    except ConfigError as exc:
        # init loads the effective config to infer a verify command; it is also
        # the command a user runs to repair their setup, so the refusal carries
        # the way out.
        raise OperatorError(
            f"{exc}\nFix or delete the per-repo config at {target}, then re-run `agent6 init`."
        ) from exc
    if rc == 0:
        theirs = tuple(p for p in scaffold_all if before[p] is not None and _digest(p) == before[p])
        # Only the repo-tracked scaffold; the per-repo config is out of the
        # workspace (under the state dir) and never committed.
        _offer_git_setup(
            cwd, tuple(p for p in scaffold_all if p not in theirs), interactive=interactive
        )
        # Only where a scaffold commit was on the table: outside a repo (and
        # after a declined `git init`) nothing was committed to leave out of.
        if theirs and is_git_repo(cwd):
            names = ", ".join(sorted(p.name for p in theirs))
            print(f"  left uncommitted (already edited): {names}")
        _print_next_steps(cwd, config_path)
    # Don't leave root-owned scaffolding in the user's repo (sudo case).
    chown_to_real_user(target.parent)
    return rc
