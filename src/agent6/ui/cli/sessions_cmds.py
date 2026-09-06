# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""`agent6 sessions list/diff/commits/compare/stop/dir/rm` (the run-branch
read side; `merge`/`prune` are `sessions_merge`)."""

from __future__ import annotations

import contextlib
import json
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from agent6.app.fork_worktrees import (
    remove_fork_worktree,
    uncommitted_in_worktree,
    worktree_owners,
)
from agent6.app.resume import covering_stamp
from agent6.git_ops import (
    DIFF_SHOW_SAFETY_FLAGS,
    GitError,
    branch_exists,
    chain_ref_for,
    chain_tip,
    delete_ref,
    git_hardening_flags,
    list_run_commits,
    run_branch_for,
    run_branch_tips,
)
from agent6.paths import state_dir
from agent6.sessions.id import SessionIdError
from agent6.sessions.ipc import request_stop
from agent6.sessions.layout import (
    SESSION_BUCKETS,
    SessionLayout,
    bucket_dir,
    layout_of,
)
from agent6.sessions.manifest import (
    ManifestError,
    SessionManifest,
    model_git_refusal,
    read_manifest,
)
from agent6.types import SESSION_KINDS
from agent6.ui.cli._common import (
    _runs_dir,
    error,
    nothing_yet,
    print_nothing_yet,
    refuse,
    resolve_or_newest_layout,
    resolve_session_layout,
    styled_status,
)
from agent6.viewmodel import (
    is_winner,
    newest_session_dir,
    session_dirs,
    session_is_live,
    summarize_session_dir,
    task_snippet,
)
from agent6.viewmodel.format import (
    format_cost_cell,
    format_when,
    lane_count,
    lane_id_cell,
    listing_status_label,
    winner_id,
)
from agent6.viewmodel.listing import ListingRow, nested_rows, row_json
from agent6.viewmodel.snapshot import commits_ref


def _cmd_list(*, as_json: bool = False, lanes: bool = False) -> int:
    """List this repo's sessions, newest first: updated (last-activity time),
    status (the mode folded in when the word does not imply it, the failure
    reason, the unmerged mark), cost, id, task. A fan-out's lanes nest under
    its row: folded into a count, listed indented with *lanes*; the JSON row
    nests them always.

    EVERY bucket, unlike the TUI/web hubs: they give `machine create` drafts
    their own card, and the CLI has none -- so leaving drafts out here made a
    session `attach` opens happily appear in no listing at all.
    """

    cwd = Path.cwd()
    dirs = session_dirs(state_dir(cwd), SESSION_BUCKETS)
    if not dirs:
        print("[]" if as_json else nothing_yet())  # the empty listing is output, not an error
        return 0
    winners = {d.name for d in dirs if is_winner(d)}  # fan-out compare winners
    tips = run_branch_tips(cwd)
    listing = nested_rows(summarize_session_dir(d, branch_tips=tips) for d in dirs)
    if as_json:
        print(json.dumps([row_json(r, winners=winners) for r in listing], indent=2))
        return 0
    color = sys.stdout.isatty()

    def cells(row: ListingRow, id_cell: str) -> tuple[str, str, str, str, str, str]:
        s = row.summary
        styled, plain = styled_status(
            s.status,
            s.reason,
            color=color,
            label=listing_status_label(s.mode, s.status, s.reason, unmerged=s.unmerged),
        )
        cost = format_cost_cell(s.cost_usd, partial=s.usd_partial)
        return format_when(row.mtime), styled, plain, cost, id_cell, s.task

    rows: list[tuple[str, str, str, str, str, str]] = []

    def emit(row: ListingRow, depth: int) -> None:
        s = row.summary
        id_cell = winner_id(s.session_id, winner=s.session_id in winners)
        if depth:
            id_cell = lane_id_cell(id_cell, depth)
        elif row.lanes and not lanes:
            id_cell += f" ({lane_count(len(row.lanes))})"
        rows.append(cells(row, id_cell))
        if lanes:
            for lane in row.lanes:
                emit(lane, depth + 1)

    for row in listing:
        emit(row, 0)
    status_w = max(6, *(len(plain) for _, _, plain, *_ in rows))
    id_w = max(2, *(len(r[4]) for r in rows))
    # The task column takes what a tty has left (floor 24); piped output keeps
    # the fixed 60 the scripts around it read today.
    fixed = 11 + 2 + status_w + 2 + 8 + 2 + id_w + 2
    task_w = max(24, shutil.get_terminal_size().columns - fixed) if color else 60
    print(f"{'updated':<11}  {'status':<{status_w}}  {'cost':<8}  {'id':<{id_w}}  task")
    for when, styled, plain, cost, session_id, task in rows:
        pad = " " * (status_w - len(plain))
        snip = task_snippet(task, max_chars=task_w)
        print(f"{when:<11}  {styled}{pad}  {cost:<8}  {session_id:<{id_w}}  {snip}")
    return 0


def _cmd_diff(*, session_id: str, stat: bool, paths: tuple[str, ...], paginate: bool = True) -> int:
    """Print the git diff a run produced (manifest.base_sha -> branch HEAD).
    *paginate* False keeps git's pager out (the `run -i` REPL, whose prompt
    loop the pager would otherwise take over).

    Resolves the run id (or unique prefix; empty string means most-recent),
    reads `manifest.json` for `base_sha` and `run_branch`, then shells
    out to `git diff` with operator-controlled argv (no LLM input). The call
    streams to the terminal, so it cannot go through git_ops._run; it carries
    the same host-RCE hardening (`git_hardening_flags`: a poisoned
    `.git/config` `diff.external` / `diff.*.textconv` / `core.fsmonitor`
    / repo hook must not execute on the host) plus `DIFF_SHOW_SAFETY_FLAGS`,
    which force the builtin diff renderer (git >= 2.53 executes even an EMPTY
    `diff.external` override, so the `-c` flags alone would kill the printed
    patch) and disable the per-file textconv driver the `-c` flags do not reach.
    """
    cwd = Path.cwd()
    res = _resolve_session_manifest(
        cwd,
        session_id,
        recent_note="diffing most recent run",
        missing_hint=" (predates manifest support, or was killed before setup)",
    )
    if isinstance(res, int):
        return res
    _layout, manifest = res

    ref = _commits_ref(cwd, manifest)
    if not ref.head_ref:
        print(f"[agent6] {ref.reason}.")
        return 0
    if manifest.run_branch:
        pruned = _pruned_branch_note(cwd, manifest, manifest.run_branch)
        if pruned is not None:  # branch gone (pruned): say where the work went
            print(pruned)
            return 0
    base_sha = manifest.base_sha
    if not base_sha:
        error("manifest has no base_sha; nothing to diff against")
        return 2

    head_ref = ref.head_ref
    # The logical command; printed without the -c hardening overrides (the
    # same convention as git_ops error messages), executed with them.
    args: list[str] = ["diff", *DIFF_SHOW_SAFETY_FLAGS]
    if stat:
        args.append("--stat")
    args.extend([f"{base_sha}..{head_ref}"])
    if paths:
        args.append("--")
        args.extend(paths)
    print(
        f"[agent6] git {' '.join(args)}  (base_branch={manifest.base_branch!r})",
        file=sys.stderr,
    )
    # A zero-commit run would print the headers and then nothing; probe first
    # (`--quiet` = exit 0 when identical) and say so. Probe errors (rc > 1,
    # e.g. a missing sha) fall through so the real diff surfaces git's message.
    probe_args = ["diff", *DIFF_SHOW_SAFETY_FLAGS, "--quiet", f"{base_sha}..{head_ref}"]
    if paths:
        probe_args.extend(["--", *paths])
    probe = subprocess.run(
        ["git", *git_hardening_flags(cwd), *probe_args], cwd=cwd, check=False, capture_output=True
    )
    if probe.returncode == 0:
        # No COMMITTED changes yet. A run commits only after a verify pass, so a
        # live run mid-work has its edits uncommitted on the worktree and this
        # reads as "the agent did nothing". If the run branch is the current
        # checkout and its worktree is dirty, say so instead of a bare silence.
        dirty = _dirty_worktree_note(cwd, manifest.run_branch)
        print(dirty if dirty else "(no changes)")
        return 0
    pager = () if paginate else ("--no-pager",)
    proc = subprocess.run(["git", *pager, *git_hardening_flags(cwd), *args], cwd=cwd, check=False)
    return proc.returncode


def _dirty_worktree_note(cwd: Path, run_branch: object) -> str:
    """A note when the diffed run's branch is the current checkout and its
    worktree has uncommitted work (a run commits at each editing step),
    else "". Only speaks when the dirty files are unambiguously THIS run's:
    the current branch must equal run_branch. Best-effort; git errors -> "" ."""
    if not run_branch:
        return ""
    # Same host-RCE hardening as the diff/probe above: `git status` refreshes the
    # index and would fire a poisoned `.git/config` core.fsmonitor on the host.
    try:
        current = subprocess.run(
            ["git", *git_hardening_flags(cwd), "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=cwd,
            check=False,
            capture_output=True,
            text=True,
        )
        if current.returncode != 0 or current.stdout.strip() != str(run_branch):
            return ""
        status = subprocess.run(
            ["git", *git_hardening_flags(cwd), "status", "--porcelain"],
            cwd=cwd,
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return ""
    n = len([ln for ln in status.stdout.splitlines() if ln.strip()])
    if n == 0:
        return ""
    files = "file" if n == 1 else "files"
    return (
        f"(no committed changes yet; {n} {files} modified in the working tree: "
        "a run commits after each verify pass)"
    )


@dataclass(frozen=True, slots=True)
class _CommitsRef:
    """Where a session's commits end (`base_sha..head_ref`): the run branch,
    the hidden chain ref for a run with branch_per_run off, or "" when the
    session made none. `reason` says why there is no ref, and is "" exactly
    when `head_ref` is one -- so the branch verbs (commits/merge) refuse on
    `reason` while diff reads `head_ref`."""

    head_ref: str
    reason: str


def _commits_ref(cwd: Path, manifest: SessionManifest) -> _CommitsRef:
    """`commits_ref` (an existing run branch while it covers the chain, else
    the chain ref), else the manifest's branch NAME while no chain exists (the
    verbs read its absence themselves: pruned, never cut, or a lane's branch
    still in its clone), else the reason the run has no commits."""
    if ref := commits_ref(manifest, cwd):
        return _CommitsRef(head_ref=ref, reason="")
    if manifest.run_branch and chain_tip(cwd, chain_ref_for(manifest.session_id)) is None:
        return _CommitsRef(head_ref=manifest.run_branch, reason="")
    if manifest.parked_task:
        # A parked run never started, so `base..HEAD` is whatever the run that
        # HELD the checkout committed -- the one it was parked behind.
        return _CommitsRef(
            head_ref="", reason="this run was parked before it started, so it made no commits"
        )
    kind = SESSION_KINDS.get(manifest.mode)
    if kind is not None and not kind.edits:
        article = "an" if manifest.mode[:1] in "aeiou" else "a"
        return _CommitsRef(
            head_ref="",
            reason=f"{article} {manifest.mode} does not write to the repo, so it made no commits",
        )
    return _CommitsRef(head_ref="", reason="this run recorded no commits")


def _resolve_session_manifest(
    cwd: Path,
    session_id: str,
    *,
    recent_note: str = "using most recent run",
    missing_hint: str = "",
) -> tuple[SessionLayout, SessionManifest] | int:
    """Resolve a run id (or '' for most-recent) to its (layout, manifest), or an exit
    code on error. Shared by `sessions diff`/`merge`/`commits`; the two note strings vary
    per caller."""
    runs_dir = _runs_dir(cwd)
    if not session_id:
        # No id: the most recent RUN. These verbs are about a run's branch, and
        # a plan or an ask has none, so widening the default would answer a
        # question the operator did not ask.
        latest = newest_session_dir([runs_dir]) if runs_dir.is_dir() else None
        if latest is None:
            # Over a plan or an ask alone (sessions without a run branch) the
            # verb says so; a fresh state dir keeps the first-contact copy.
            print_nothing_yet("runs" if session_dirs(state_dir(cwd)) else "sessions")
            return 2
        layout = layout_of(latest)
        print(f"[agent6] {recent_note}: {layout.session_id}", file=sys.stderr)
    else:
        # An EXPLICIT id resolves across every bucket. A plan the operator named
        # exists; "no session matches" would deny that, when the real answer is that
        # it has no branch to show.
        try:
            layout = resolve_session_layout(cwd, session_id)
        except SessionIdError as exc:
            error(f"{exc}")
            return 2
    target_id = layout.session_id
    if not layout.manifest_path.is_file():
        error(f"session {target_id} has no manifest.json{missing_hint}")
        return 2
    try:
        manifest = read_manifest(layout.session_dir)
    except ManifestError as exc:
        error(f"could not read manifest: {exc}")
        return 2
    # A fan-out commits nothing by design: its lanes hold the work, and its
    # record is the newest run once it ends.
    refusal = (
        f"{target_id} is a fan-out; its lanes hold the commits"
        f" (`agent6 sessions show {target_id}` lists them)"
        if manifest.fanout is not None
        else model_git_refusal(manifest, "sessions")
    )
    if refusal is not None:
        refuse(f"{refusal}")
        return 2
    return layout, manifest


def _cmd_stop(*, session_id: str) -> int:
    """Ask a running detached run to stop cleanly after its current step.

    Drops the same 'stop after this step' marker the TUI/web Stop button uses:
    the run finishes the in-flight step (its tool results and auto-commit land),
    then ends and is resumable. For a running run only; a finished one is a no-op
    with a note."""
    cwd = Path.cwd()
    try:
        layout = resolve_or_newest_layout(cwd, session_id)
    except SessionIdError as exc:
        error(f"{exc}")
        return 2
    if layout is None:
        print_nothing_yet()
        return 2
    session_dir = layout.session_dir
    rid = session_dir.name
    if not session_is_live(session_dir):
        # The liveness owner, not the pid: a finished run's worker.pid lingers
        # through teardown, and "it ends after the current step" would promise
        # a stop the exited loop will never read.
        print(f"[agent6] {rid} is not running; nothing to stop.", file=sys.stderr)
        return 0
    if not request_stop(session_dir):
        print(f"[agent6] could not write the stop request for {rid}", file=sys.stderr)
        return 1
    fanout = None
    with contextlib.suppress(ManifestError):
        fanout = read_manifest(session_dir).fanout
    if fanout is not None:
        print(
            f"[agent6] requested stop for {rid}; its lanes are asked to stop and what"
            " landed is imported and ranked."
        )
        return 0
    print(f"[agent6] requested stop for {rid}; it ends after the current step.")
    print(f"  resume with:  agent6 resume {rid}")
    return 0


def _committed_nothing(cwd: Path, session_id: str) -> bool:
    """True when a run left no commit anywhere: the chain ref it commits to was
    never created, so its branch was never cut either."""
    return chain_tip(cwd, chain_ref_for(session_id)) is None


def _pruned_branch_note(cwd: Path, manifest: SessionManifest, run_branch: str) -> str | None:
    """A friendly message when a run's branch is absent, or None when it is
    there. Says where the work went instead of leaking a raw git fatal, and
    separates the ways to get here: a merged-then-pruned branch (the stamp
    covering every commit the run made), a branch deleted past its stamp or
    with no merge recorded (the chain ref keeps the commits), and a run that
    committed nothing, whose branch was never cut."""
    if branch_exists(cwd, run_branch):
        return None
    stamp = covering_stamp(cwd, manifest)
    if stamp is not None:
        note = f"[agent6] run branch {run_branch} was pruned; {stamp.landed()}"
        if stamp.commit:
            note += f"\n  see: git show {stamp.commit}"
        return note
    if _committed_nothing(cwd, manifest.session_id):
        return f"[agent6] this run committed nothing, so {run_branch} was never cut."
    chain = chain_ref_for(manifest.session_id)
    if manifest.merged is not None:
        return (
            f"[agent6] run branch {run_branch} is gone; its commits survive at {chain},"
            f" past the merge into {manifest.merged.into}."
        )
    return (
        f"[agent6] run branch {run_branch} is gone with no merge recorded; its commits"
        f" survive at {chain}."
    )


def _cmd_commits(*, session_id: str) -> int:
    """List the per-step commits on a run's branch (manifest.base_sha -> run branch)."""
    cwd = Path.cwd()
    res = _resolve_session_manifest(cwd, session_id)
    if isinstance(res, int):
        return res
    _layout, manifest = res
    ref = _commits_ref(cwd, manifest)
    if not ref.head_ref:
        error(f"this session has no branch to list commits from ({ref.reason}).")
        return 2
    base_sha = manifest.base_sha
    if not base_sha:
        error("manifest has no base_sha; nothing to list commits from")
        return 2
    run_branch = ref.head_ref
    # Only a RECORDED branch can be pruned; the HEAD fallback is not a ref
    # whose absence means anything (same guard as diff's).
    pruned = _pruned_branch_note(cwd, manifest, run_branch) if manifest.run_branch else None
    if pruned is not None:
        print(pruned)
        return 0
    rows = list_run_commits(cwd, base_sha, run_branch)
    if not rows:
        print("[agent6] no commits on the run branch.")
        return 0
    for row in rows:
        print(f"{row.sha[:12]}  {row.subject}")
    print(f"\n[agent6] {len(rows)} commit(s) on {run_branch}", file=sys.stderr)
    return 0


def _cmd_sessions_dir(session_id: str = "") -> int:
    """Print the per-repo state dir (where this repo's run history lives), or the
    named session's own directory.

    One bare line so it composes (`ls "$(agent6 sessions dir)"`, or delete a bucket
    outright). Sessions live under sessions/<bucket>/, one bucket per mode."""
    cwd = Path.cwd()
    if not session_id:
        print(state_dir(cwd))
        return 0
    try:
        layout = resolve_session_layout(cwd, session_id)
    except SessionIdError as exc:
        error(f"{exc}")
        return 2
    print(layout.session_dir)
    return 0


def _rm_asks(cwd: Path, session_id: str) -> int:
    """Clear this directory's asks bucket; a deletion failure is an error,
    never a success line over a surviving directory."""
    if session_id:
        error("--asks clears this directory's asks; drop the run id.")
        return 2
    bucket = bucket_dir(state_dir(cwd), "asks")
    gone = sum(1 for _ in bucket.iterdir()) if bucket.is_dir() else 0
    try:
        shutil.rmtree(bucket)
    except FileNotFoundError:
        pass
    except OSError as exc:
        error(f"could not remove {bucket}: {exc}")
        return 1
    print(f"removed {gone} ask{'' if gone == 1 else 's'} from {cwd}")
    return 0


def _rm_refusal(layout: SessionLayout, worktree: Path | None, tips: tuple[str, ...]) -> str:
    """Why this record cannot be deleted, or "". *worktree* is the fork's own
    (None when another session shares it, and keeps it).

    The record is the only thing that names a fork's worktree, so a delete that
    left one holding work no commit has left it with nothing to find it by,
    let alone remove it."""
    if session_is_live(layout.session_dir):
        return (
            f"{layout.session_id} is still live; stop it first"
            f" (agent6 sessions stop {layout.session_id})."
        )
    if worktree is None or not (dirt := uncommitted_in_worktree(worktree, tips)):
        return ""
    return (
        f"{layout.session_id}'s worktree {dirt} ({worktree}); deleting the record"
        f" would leave it with nothing naming it. Keep that work, then:"
        f" git -C {worktree} status"
    )


def _cmd_sessions_rm(*, session_id: str, asks: bool) -> int:
    """Delete run history from the state dir, plus the run's hidden chain ref
    (`refs/agent6/<id>/head`, the gc anchor -- meaningless once the record is gone,
    and left behind it would pin the run's objects forever) and, for a fork,
    the worktree its manifest records, unless another session (an `/undo`
    fork of it) still names that worktree.

    The run's visible branch and its commits are git's, and are left alone
    (`sessions prune` is the branch verb). `--asks` clears the asks made in
    THIS directory -- an ask is keyed by the directory it ran in, so asks made
    elsewhere are untouched."""
    cwd = Path.cwd()
    if asks:
        return _rm_asks(cwd, session_id)
    try:
        # rm is the surface that deletes a husk, so it resolves one.
        layout = resolve_or_newest_layout(cwd, session_id, allow_husk=True)
    except SessionIdError as exc:
        error(f"{exc}")
        return 2
    if layout is None:
        print_nothing_yet()
        return 2
    worktree: Path | None = None
    with contextlib.suppress(ManifestError):
        worktree = read_manifest(layout.session_dir).worktree
    sharing = (
        [
            d.name
            for d, _m in worktree_owners(state_dir(cwd)).get(worktree, [])
            if d != layout.session_dir
        ]
        if worktree is not None
        else []
    )
    landed = chain_tip(cwd, chain_ref_for(layout.session_id)) or ""
    tips = (landed,) if landed else ()
    if reason := _rm_refusal(layout, worktree if not sharing else None, tips):
        refuse(f"{reason}")
        return 2
    try:
        shutil.rmtree(layout.session_dir)
    except OSError as exc:
        # A partial delete leaves a real session remnant; success here would be
        # a lie and the chain-ref cleanup below would strand its commits.
        error(f"could not remove {layout.session_dir}: {exc}")
        return 1
    went: list[str] = []  # what went with the record
    stays = ""
    if worktree is not None and sharing:
        verb = "names" if len(sharing) == 1 else "name"
        stays = f"; its worktree stays: {', '.join(sharing)} still {verb} it"
    elif worktree is not None:
        gone, why = remove_fork_worktree(cwd, worktree, tips)
        if gone:
            went.append("its worktree")
        elif why:
            stays = f"; its worktree stays: it {why} ({worktree})"
    chain = chain_ref_for(layout.session_id)
    try:
        if (chain_head := chain_tip(cwd, chain)) is not None:
            branch = run_branch_for(layout.session_id)
            branch_kept = branch_exists(cwd, branch)
            delete_ref(cwd, chain)
            # With no visible branch the ref was the commits' only anchor, and
            # a chain ref has no reflog: the sha is the only way back to them,
            # so it goes on the line that deletes it (one gc from gone).
            went.append(
                "its chain ref"
                + (
                    f" (branch {branch} kept; `sessions prune` reports it)"
                    if branch_kept
                    else f" (its commits are now loose; until git gc:"
                    f" git branch <name> {chain_head[:12]})"
                )
            )
    except GitError:
        pass  # not a repo here, or git unreadable: state-dir removal stands
    what = [layout.session_id, *went]
    removed = f"{', '.join(what[:-1])} and {what[-1]}" if went else what[0]
    print(f"removed {removed}{stays}")
    return 0
