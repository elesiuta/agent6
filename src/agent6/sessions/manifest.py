# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Read a session's manifest.json into the typed :class:`SessionManifest`: the one
reader and the on-disk shape, with `app.manifest` as the writer.

A leaf beside `layout.py`: pydantic + path arithmetic, no agent6 imports, so
app, the viewmodel, and the CLI parse a run's manifest through one owner and one
shape instead of each re-deriving the read + error-catch + stringly `.get`.

The model defaults every field and ignores unknown keys, so a partial or
foreign-keyed manifest renders what it does carry, which is lenience for damage
rather than a compatibility promise (the shape is liquid until 1.0; superseded keys are
dropped, never folded). Reading is lenient: `read_manifest` degrades a corrupt
file through `ManifestError`, which the render consumers already catch. The one
strict contract is `session_mode`, the fork/resume privilege gate, which refuses
an unknown mode rather than falling open to the write ("run") tools.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

from pydantic import BaseModel, ConfigDict, ValidationError

from agent6.git_ops import BRANCH_PREFIX
from agent6.sessions.layout import session_layout
from agent6.types import ResumableMode, UnknownSessionKind, session_kind

_MODEL_CONFIG = ConfigDict(frozen=True, extra="ignore")


class ManifestError(Exception):
    """A session's manifest.json is missing, unreadable, corrupt, not a JSON object,
    does not validate, or (via `session_mode`) records an unknown privilege
    mode. Carries the underlying cause as its message, so a caller that wants to
    surface a detail can render it."""


class ModelBrief(BaseModel):
    """`{provider, model}` for a resolved role."""

    model_config = _MODEL_CONFIG

    provider: str = ""
    model: str = ""


class ModelsBrief(BaseModel):
    """The models the run resolved: the one that DROVE it (the worker, or the
    planner for a plan run) and the reviewer. Null when the role is unset."""

    model_config = _MODEL_CONFIG

    driver: ModelBrief | None = None
    reviewer: ModelBrief | None = None


class PolicyStamp(BaseModel):
    """How the run was launched: the policy facts an operator wants to see
    without opening config. Recorded so every surface reads them from one
    place (the TUI and web are other processes with only the run dir), and so
    `agent6 exec` reproduces the run's isolation and network even
    after the config moved (mounts stay config-derived; exec's help says
    so)."""

    model_config = _MODEL_CONFIG

    run_commands: str = ""
    isolation: str = ""
    network: str = ""


class WorkflowStamp(BaseModel):
    """The in-loop strategy the run started with, so `resume` re-applies it."""

    model_config = _MODEL_CONFIG

    review_trigger: str = ""
    revise_prompt: str = ""
    preset: str = ""
    # The verify gate this run is pinned to, and where it came from:
    # "configured" (a config file, which the model cannot write), "inferred"
    # (repo signals / AGENTS.md, which it can), "adopted" (gained mid-run by a
    # run that started gateless), or "" for no gate at all. Pinned so a mid-run
    # edit to the source cannot move the gate under the run -- including on a
    # resumed leg, where only operator config outranks what is recorded here.
    verify_command: tuple[str, ...] = ()
    verify_origin: str = ""
    # Whether `preset` was chosen by --preset rather than by a config file.
    # The name alone is half the fact: replaying a config-selected one as a flag
    # splices it ABOVE the repo config it originally lost to (see replay_preset).
    preset_from_flag: bool = False

    @property
    def replay_preset(self) -> str:
        """The `--preset` override a resumed or forked leg must re-apply.

        Only a FLAG-selected preset: a config-selected one re-resolves
        identically from the same config files, whereas handing its name back as
        an override makes `_select_preset` call it a flag, which outranks every
        config layer. A run whose repo config beat a global preset therefore
        came back from resume with the preset winning instead -- gaining, for
        example, a blocking review veto the original never had.
        """
        return self.preset if self.preset_from_flag else ""


# The `sha` of a merge that added no commit: the target already held the
# branch's content. Its own tip would name a commit that is not the run's.
NO_MERGE_COMMIT = "0" * 40


class MergeStamp(BaseModel):
    """Recorded once a run branch is merged, so later tooling tells a merged run
    branch from an unmerged one."""

    model_config = _MODEL_CONFIG

    into: str = ""
    sha: str = ""  # the merge commit in `into`, or NO_MERGE_COMMIT
    ts: str = ""
    # The RUN BRANCH tip that was merged (`sha` is the commit in the base).
    # `sessions prune --delete-squashed` force-deletes only when the branch still
    # points here: a resumed run keeps committing on the same branch under this
    # stamp, and those commits exist in no other ref.
    tip: str = ""


class CompareStamp(BaseModel):
    """A fan-out lane's auto-compare placement. The fan-out id itself lives in
    the top-level `parallel_id`, not here."""

    model_config = _MODEL_CONFIG

    rank: int = 0
    of: int = 0
    winner: bool = False
    ranked_by: str = ""
    rationale: str = ""
    # The judge call's cost for the WHOLE group, recorded on every lane like
    # the rationale; summing it across lanes would double-count. 0.0 only when
    # no judge call was made (a failed judge that fell back mechanically still
    # spent); partial marks a lower bound (unpriced reviewer, no reported cost).
    judge_cost_usd: float = 0.0
    judge_cost_partial: bool = False


# The shape this binary writes. Stamp-rewrites re-stamp it (see write_manifest)
# so a manifest's version claim always matches the shape actually on disk.
MANIFEST_VERSION = 3


class SessionManifest(BaseModel):
    """The typed manifest.json a session starts with (and later stamps).

    Every field defaults and `extra="ignore"` drops keys this version does not
    know, so a manifest missing fields or carrying foreign keys still renders;
    the writer always emits the full shape. Known limitation: a stamp-rewrite
    by this version drops keys only a NEWER version knows (load -> model_copy
    -> dump cannot carry them), so the write path re-stamps `version` to keep
    the on-disk claim truthful.
    """

    model_config = _MODEL_CONFIG

    version: int = MANIFEST_VERSION
    agent6_version: str = ""
    session_id: str = ""
    # No default mode: the field is the privilege gate's only input, and a
    # manifest that lost the key (truncated, hand-edited, foreign writer) must
    # not read as the more-privileged "run". Display consumers show "?" for it.
    mode: str = ""
    start_ts: str = ""
    user_task: str = ""
    base_sha: str = ""
    base_branch: str = ""
    run_branch: str | None = None
    # Who managed git for this run ([git].control at start). "model" runs have
    # no chain/branch to diff or merge; the git surfaces refuse via
    # `model_git_refusal`. An old manifest folds the default.
    git_control: str = "agent6"
    models: ModelsBrief = ModelsBrief()
    workflow: WorkflowStamp = WorkflowStamp()
    policy: PolicyStamp = PolicyStamp()
    # A parked run: submitted, never started. Holds the VERBATIM task
    # (user_task above is the truncated display twin); non-empty means
    # `agent6 resume <id>` starts it fresh, whose manifest rewrite clears it.
    # `parked_reason` says why, for the operator: the checkout was busy, or the
    # working tree had uncommitted changes and the operator chose to wait.
    parked_task: str = ""
    parked_reason: str = ""
    # fork lineage (a non-forked run leaves these null)
    parent_session_id: str | None = None
    forked_from_turn: int | None = None
    forked_from_sha: str | None = None
    # A fork's own checkout: the linked git worktree `agent6 fork` added,
    # absolute. None for a session working in the operator's checkout; an
    # `/undo` fork names its source's.
    worktree: Path | None = None
    # The repository git dir that worktree points into, recorded when agent6
    # added it: the one path a fork leg's jail grants beyond the workspace.
    # Never read back from the worktree's own `.git` pointer, which a jailed
    # command can rewrite under hardened.
    worktree_git_dir: Path | None = None
    # merge stamp (null until the run branch is merged)
    merged: MergeStamp | None = None
    # parallel lineage + compare stamp (null outside a fan-out)
    parallel_id: str | None = None
    lane: int | None = None
    compare: CompareStamp | None = None

    def session_mode(self) -> ResumableMode:
        """The session's mode, refusing anything this agent6 does not know.

        Fork and resume act on this rather than the raw `mode` string, so a
        damaged manifest never silently escalates a read-only session to the
        privileged write ("run") tools. Pure-render consumers read `mode`
        directly: showing an unknown value is fine, acting on one is not.

        The vocabulary is `types.SESSION_KINDS`. Keeping a second list here let
        the two disagree -- this one refused "machine" and "agent" while the
        tool surface happily built one.
        """
        try:
            kind = session_kind(self.mode)
        except UnknownSessionKind as exc:
            raise ManifestError(str(exc)) from exc
        if not kind.resumable:
            raise ManifestError(f"a {kind.name!r} session is not resumable")
        # Guarded by `resumable` above, which the type system cannot follow.
        return cast(ResumableMode, kind.name)


def model_git_refusal(manifest: SessionManifest, verb: str) -> str | None:
    """The one refusal for git surfaces on a model-controlled run, or None.

    A `git.control = "model"` run has no agent6 chain or run branch: the
    model's own commits are the record, so there is nothing for *verb* to
    act on."""
    if manifest.git_control != "model":
        return None
    return (
        f"{verb}: run {manifest.session_id or '?'} managed git itself"
        ' ([git].control = "model"); its record is the model\'s own commits,'
        " not an agent6 chain. Inspect it with plain git."
    )


def read_manifest(session_dir: Path) -> SessionManifest:
    """Parse `<session_dir>/manifest.json` into a :class:`SessionManifest`, or raise
    `ManifestError`.

    Lenient by design: every field defaults, so any parseable historical manifest
    validates and renders. A file that cannot be read (`OSError`), is not JSON
    (any `ValueError`: a truncated JSON is a `JSONDecodeError` and a
    torn-UTF-8 tail a `UnicodeDecodeError`, both subclasses), is not a JSON
    object, or fails validation degrades through the one typed error the render
    consumers already catch; the fork/resume gate turns it into a loud refusal.
    """
    path = session_dir / "manifest.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ManifestError(str(exc)) from exc
    if not isinstance(data, dict):
        raise ManifestError("manifest is not a JSON object")
    try:
        return SessionManifest.model_validate(data)
    except ValidationError as exc:
        raise ManifestError(str(exc)) from exc


def manifest_for_branch(state_dir: Path, branch: str) -> SessionManifest | None:
    """The manifest of the session that cut *branch*, or None.

    Across buckets: any session that forks cuts `agent6/<id>`, and a forked plan
    lives in plans/. Reading out of runs/ alone broke the base-branch chain walk
    and left `sessions prune` unable to confirm such a branch merged.
    """
    layout = session_layout(state_dir, branch.removeprefix(BRANCH_PREFIX))
    if layout is None:
        return None
    try:
        return read_manifest(layout.session_dir)
    except ManifestError:
        return None
