# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The `[git]` model: worktree policy, the run's detached chain, merge and
message styles."""

from __future__ import annotations

import re
import string
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from agent6.config._base import MODEL_CONFIG


class GitCommitCheckpointConfig(BaseModel):
    """Message style for the per-step commits a run makes on its branch."""

    model_config = MODEL_CONFIG

    # agent6: the `agent6 iter N:` subject. conventional: a `type(scope): subject`
    # derived from the diff without a model call. model: the model writes the
    # message from git facts, degrading to agent6 with a warning on any failure.
    message: Literal["agent6", "conventional", "model"] = Field(
        default="agent6",
        description=(
            "The message of each per-step commit: `agent6` (`agent6 iter N: <summary>`), "
            "`conventional` (a `type(scope): subject` derived from the diff, no model call), or "
            "`model` (the model writes it from the git facts, falling back to `agent6` with a "
            "warning on any failure)."
        ),
    )


class GitCommitSquashConfig(BaseModel):
    """Message style for the one commit a squash merge produces."""

    model_config = MODEL_CONFIG

    # As checkpoint's styles, plus combine: git's own squash message (the
    # concatenated per-step log).
    message: Literal["agent6", "conventional", "combine", "model"] = Field(
        default="agent6",
        description=(
            "The message of the one commit a squash merge produces: `agent6` (`agent6 iter N: "
            "<summary>` style), `conventional` (a `type(scope): subject` derived from the diff, no "
            "model call), `combine` (git's own squash message: the per-step log concatenated), or "
            "`model` (model-written, falling back to `agent6` with a warning on any failure)."
        ),
    )


class GitCommitConfig(BaseModel):
    """Overrides for the author/committer identity on agent6 commits, the
    provenance trailer, and the per-kind message styles.

    `name`/`email` default to None = the project's own `git config` identity;
    `agent6 run` refuses at startup when neither an override nor a resolvable
    identity exists, rather than committing as `(no author) <(none)>`.
    """

    model_config = MODEL_CONFIG

    name: str | None = Field(
        default=None,
        description=(
            "Author and committer name on the commits agent6 makes; unset uses the repo's own `git "
            "config`. A run with no resolvable identity refuses to start."
        ),
    )
    email: str | None = Field(
        default=None,
        description=(
            "Author and committer email on the commits agent6 makes; unset uses the repo's own "
            "`git config`. A run with no resolvable identity refuses to start."
        ),
    )
    # Appended to every commit agent6 makes when non-empty, e.g.
    # "Assisted-by: agent6:{model}". {model} = the model(s) that wrote the
    # code, first worker first, ", "-joined when several contributed.
    trailer: str = Field(
        default="",
        description=(
            "A git trailer line (`Key: value`) appended to every commit agent6 makes, e.g. "
            '`"Assisted-by: agent6:{model}"` or `"Co-authored-by: agent6:{model} '
            '<noreply@agent6.dev>"`. `{model}` is the model that wrote the code (several are '
            "joined with `, `). Empty: no trailer."
        ),
    )
    checkpoint: GitCommitCheckpointConfig = GitCommitCheckpointConfig()
    squash: GitCommitSquashConfig = GitCommitSquashConfig()

    @field_validator("trailer")
    @classmethod
    def _trailer_is_a_trailer_line(cls, v: str) -> str:
        if not v:
            return v
        fields = {f for _, f, _, _ in string.Formatter().parse(v) if f is not None}
        unknown = fields - {"model"}
        if unknown:
            raise ValueError(
                f"unknown placeholder {sorted(unknown)} in git.commit.trailer (known: {{model}})"
            )
        rendered = v.format(model="m")
        if not re.fullmatch(r"[A-Za-z][A-Za-z-]*: .+", rendered, re.DOTALL):
            raise ValueError(
                'git.commit.trailer must be a git trailer line, "Key: value"'
                ' (e.g. "Assisted-by: agent6:{model}")'
            )
        return v


class GitConfig(BaseModel):
    model_config = MODEL_CONFIG

    # Untracked files are never in question: a run records the ones present
    # at its start (`untracked-at-start`) and leaves them out of every commit
    # and dirty check.
    require_clean_worktree: bool = Field(
        default=True,
        description=(
            "When tracked files have uncommitted changes at start, ask how the run treats them "
            "(`stash` them for the run, `include` them in its commits, or `cancel`, which parks "
            "the run for a later resume); a run nobody can answer refuses to start. `false`: start "
            "without asking, the run's first commit records them. Untracked files never count and "
            "are never committed."
        ),
    )
    auto_stash: bool = Field(
        default=False,
        description=(
            "Stash the tracked files' uncommitted changes at start without asking; at the end the "
            "stash is applied back per `auto_stash_pop`, else its `git stash apply <sha>` line is "
            "printed."
        ),
    )
    # When auto_stash stashed pre-run changes, restore them at run end. Default
    # off (safe): the run-end reporter always prints how to pop the stash; with
    # this on, agent6 also pops it for you when it can do so cleanly (a clean
    # tree: a run that edited leaves its unmerged work in the tree, so this
    # fires after auto_merge or a no-edit run), and otherwise leaves the stash
    # with a message rather than risk a conflicted auto-apply.
    auto_stash_pop: bool = Field(
        default=False,
        description=(
            "Apply the pre-run stash back when the run ends and the tree is clean (a clean apply, "
            "no conflicts). On any doubt the stash stays and the apply line is printed. Never "
            "`reset --hard`. Requires `auto_stash`."
        ),
    )
    # Per-step commits land on the run's own detached chain
    # (refs/agent6/<session>/head), parented on HEAD at run start; HEAD, the
    # operator's index, and the checkout are never touched. branch_per_run
    # additionally advances a visible agent6/<slug> branch ref to the chain
    # tip (off = the hidden ref only). Forced on for --parallel lanes (work
    # is imported by branch).
    control: Literal["agent6", "model"] = Field(
        default="agent6",
        description=(
            "Who manages git during a run: `agent6` records every step on the run's own commit "
            "chain and branch, never touching HEAD; `model` hands git to the model: no per-step "
            "chain, no run branch, the model's own commits and branches are the record, and "
            "`sessions diff`/`merge`, `/undo`, and `fork` refuse for such runs. Requires "
            "`sandbox.protect_git = false`."
        ),
    )
    branch_per_run: bool = Field(
        default=True,
        description=(
            "Also advance a visible `agent6/<run-id>` branch to the run's chain tip; `false` keeps "
            "only the hidden `refs/agent6/<run-id>/head` ref. Forced on for `--parallel` lanes "
            "(their work is imported by branch)."
        ),
    )
    # Off = no per-step commits at all: sessions diff/commits/merge, fork
    # rollback, and the compare judge honestly degrade to "no step history";
    # resume still works from snapshots.
    commit_per_step: bool = Field(
        default=True,
        description=(
            "Commit each editing step onto the run's detached chain (a temp index; HEAD, your "
            "index, and your checkout are never touched). `false`: agent6 never commits; the work "
            "stays only in the worktree, and resume-from-git, `sessions diff`/`merge`, and "
            "`/parallel` dispatch from a changed tree degrade."
        ),
    )
    # Default strategy for `agent6 sessions merge`: how the run branch lands on
    # your branch. `squash` (one combined commit), `merge` (a
    # --no-ff merge keeping the per-step history), or `ff` (fast-forward only).
    # The per-step commits always happen on the run branch during the run; this
    # only governs how they are consolidated when you merge.
    merge_strategy: Literal["squash", "merge", "ff"] = Field(
        default="squash",
        description=(
            "How `agent6 sessions merge` lands a run on its base: `squash` (one commit), `merge` "
            "(a `--no-ff` merge that keeps the per-step history), or `ff` (fast-forward). "
            "Consolidation only; per-step commits always land on the run's chain."
        ),
    )
    # After a successful run, automatically run `merge_strategy` to land the
    # run's work on its base (what `agent6 sessions merge` does, run for you).
    # Default off: the run's refs are kept until you choose to merge. Works
    # with branch_per_run off too (the hidden chain ref is merged). With
    # auto_stash_pop the merge lands first, then your stashed pre-run changes
    # go back on top.
    auto_merge: bool = Field(
        default=False,
        description=(
            "After a run that finished with nothing red, merge its work into its base branch "
            "automatically (never over a red or stale verify). With `branch_per_run` off it merges "
            "the hidden chain ref. On a conflict nothing moves and the instructions are printed."
        ),
    )
    # After auto_merge, delete the run branch when it is safely deletable
    # (`git branch -d`: reachable-merged, so merge/ff strategies). A squash-merged
    # branch is unreachable and is reported with the `git branch -D` to remove it by
    # hand, never force-deleted. Requires auto_merge; no-op when branch_per_run
    # is off (there is no branch, and the hidden chain ref stays as the run's
    # record until `sessions rm`). With both on, run branches stop
    # accumulating, so agent6 looks like a direct-to-branch agent while keeping
    # the per-step commits during the run. Default off.
    auto_prune: bool = Field(
        default=False,
        description=(
            "After an `auto_merge`, delete the run branch when `git branch -d` can (a `merge` or "
            "`ff` merge). A squash-merged branch is reported with its `-D` line, never "
            "force-deleted. Requires `auto_merge`; nothing to do without a run branch."
        ),
    )
    # Whether the repo's own git hooks (`.git/hooks/*`) run during agent6's
    # OWN git operations (notably the per-step auto-commit). Default false:
    # secure-by-default (a hook is repo-controlled code that would execute on
    # the HOST, outside the jail, when agent6 commits -- a host-RCE vector for
    # an adversarial repo) and also avoids re-running a slow pre-commit hook on
    # every micro-commit. The verify_command is agent6's real success gate, not
    # git hooks. Set true to honor the repo's hooks (trust the repo). Either
    # way `core.fsmonitor`/`diff.external` stay neutralized (those fire on
    # status/diff and have no legitimate use here).
    run_repo_hooks: bool = Field(
        default=False,
        description=(
            "Run the repo's own `.git/hooks/*` during agent6's git operations. `false` skips "
            "them: a repo hook is repo-controlled code that would run on the host. "
            "`core.fsmonitor` and `diff.external` are always neutralized."
        ),
    )
    # Whether the repo's own content drivers -- `filter.<n>.clean/smudge/process`
    # and `merge.<n>.driver` -- run during agent6's OWN git operations. Default
    # false: like a hook, a driver defined in `.git/config` is repo-controlled
    # code that executes on the HOST, outside the jail, when agent6 stages or
    # merges (a host-RCE vector for a repo cloned with a poisoned `.git/config`).
    # agent6 neutralizes each repo-defined driver by name. Set true to honor
    # them -- the setting a Git-LFS repo needs, since LFS's clean/smudge filters
    # are exactly these drivers.
    run_repo_filters: bool = Field(
        default=False,
        description=(
            "Honor the repo's content drivers (`filter.<name>.clean/smudge/process`, "
            "`merge.<name>.driver`) during agent6's git operations. `false` neutralizes each by "
            "name: a driver defined in `.git/config` is repo-controlled code that would run on "
            "the host at every commit. `true` is what Git LFS needs (its clean/smudge filters "
            "are these drivers)."
        ),
    )
    commit: GitCommitConfig = Field(default_factory=GitCommitConfig)

    @model_validator(mode="after")
    def _check_auto_merge(self) -> GitConfig:
        if self.auto_stash_pop and not self.auto_stash:
            raise ValueError(
                "git.auto_stash_pop requires git.auto_stash: with nothing stashed "
                "pre-run there is nothing to restore at run end."
            )
        if self.auto_prune and not self.auto_merge:
            raise ValueError(
                "git.auto_prune requires git.auto_merge: pruning a run branch only makes "
                "sense once it has been merged."
            )
        return self
