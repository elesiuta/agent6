# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""End of a run: the composed end block, exit code, auto-merge / auto-stash
finalizers, and the operator notify hook."""

from __future__ import annotations

import contextlib
import json
import shlex
import subprocess
from collections.abc import Callable, Collection, Sequence
from pathlib import Path

from agent6.app.merge import execute_merge, left_behind_line, noop_merge_line
from agent6.app.reporter import Reporter
from agent6.budget import BudgetTracker
from agent6.child_env import curated_env
from agent6.config import Config, NotifyConfig
from agent6.events import EventSink
from agent6.git_ops import (
    CommitIdentity,
    GitError,
    auto_stash_message,
    branch_exists,
    chain_ref_for,
    chain_tip,
    create_branch,
    delete_branch_if_merged,
    find_stash,
    merge_stamp_holds,
    render_commit_trailer,
    restore_stash,
    verify_git_identity,
)
from agent6.git_ops import (
    status as git_status,
)
from agent6.sessions.layout import LOGS_NAME, SessionLayout, read_untracked_at_start
from agent6.sessions.manifest import ManifestError, SessionManifest, read_manifest
from agent6.verify_infer import line_to_argv
from agent6.viewmodel import scan_session_log, summarize_session_dir, tail_events, worker_models
from agent6.viewmodel.format import format_cost
from agent6.workflows.loop import SessionResult

# Distinct exit code for a budget-exhausted run so automation can tell "raise
# the cap and `agent6 resume`" apart from a genuine failure. Documented in
# docs/config.md ([budget]); a budget-stopped run is resumable from its snapshot.
_EXIT_BUDGET_EXHAUSTED = 3
# The agent finished deliberately but the verify gate was red or stale. Its own
# code so a script can tell "the work is not green" from "the run broke" (1)
# without parsing the event log; `[workflow].verify_retries` bounds how often
# the same condition returns a finish to the model first. Public: the parallel
# fan-out exits with it when gates ran and no lane passed.
EXIT_VERIFY_FAILED = 4
# The agent finished and the gate (if any) was green, but the promised run
# branch never came into existence and the edits sit uncommitted
# (`stranded_edits`): the deliverable a script would collect on 0 is not
# there. A run that changed nothing stays 0.
EXIT_NO_COMMIT_LANDED = 5


def session_exit_code(result: SessionResult, *, stranded: bool = False) -> int:
    """Map a finished run to its process exit code.

    0 finished (nothing to gate on, or the gate was green) / 3 budget /
    4 finished over a not-green verify / 5 finished with its edits stranded
    uncommitted (`stranded_edits`) / 1 else.

    4 covers red AND unverified: the tree is not green, and that is what 4
    means -- exiting 0 on "no verify ran" would let a worker pass by never
    running the gate. 5 is the same principle for the deliverable: the
    promised branch never materialized and the edits sit uncommitted, so 0
    would tell a script the work landed. A red gate outranks 5 (the gate is
    the primary signal; the footer still says both). WHOSE failure it is
    shows in the word and the reason, not here; a script reading 0 would
    take it as passing."""
    if result.completed:
        if result.verified in ("failed", "unverified"):
            return EXIT_VERIFY_FAILED
        return EXIT_NO_COMMIT_LANDED if stranded else 0
    if result.reason == "budget_exhausted":
        return _EXIT_BUDGET_EXHAUSTED
    return 1


def stranded_edits(result: SessionResult, layout: SessionLayout, cwd: Path) -> bool:
    """A completed run whose promised branch never came into existence while
    edits sit uncommitted in the working tree. Exit code 5 and the end
    banner's WARNING read this one predicate, so the machine surface and the
    human surface cannot disagree."""
    if not result.completed:
        return False
    run_branch = ""
    merged = False
    with contextlib.suppress(ManifestError):
        manifest = read_manifest(layout.session_dir)
        run_branch = manifest.run_branch or ""
        merged = manifest.merged is not None and merge_stamp_holds(
            cwd, manifest.session_id, run_branch, manifest.merged.tip
        )
    if not run_branch or merged or branch_exists(cwd, run_branch):
        return False
    dirty = False
    with contextlib.suppress(GitError):
        exclude = read_untracked_at_start(layout.session_dir)
        dirty = not git_status(cwd, exclude=exclude).is_clean
    return dirty


def auto_merge_eligible(result: SessionResult) -> bool:
    """auto_merge lands only work the gate vouched for (or that had no gate):
    a red or unverified finish stays on its branch for the operator. One
    predicate for both lifecycles, so run and resume cannot drift."""
    return result.completed and result.verified in ("passed", "not_applicable")


def _sandbox_unreachable_tools(layout: SessionLayout) -> list[str]:
    """Binaries the run flagged as host-present but jail-broken
    (loop.sandbox_tool_unreachable events), for the operator diagnostic."""
    out: list[str] = []
    try:
        for line in layout.logs_path.read_text(encoding="utf-8").splitlines():
            if '"loop.sandbox_tool_unreachable"' not in line:
                continue
            try:
                binary = json.loads(line).get("binary")
            except ValueError:
                continue
            if isinstance(binary, str) and binary and binary not in out:
                out.append(binary)
    except OSError:
        pass
    return out


def _print_next_session(layout: SessionLayout, *, reporter: Reporter) -> None:
    """After a session that produced something to act on, name the next step.

    Seeding already exists; what was missing was the affordance -- an operator
    had to know the flag was there. An ask ends holding work someone else does.
    A plan ends holding OPEN QUESTIONS, and nothing named the loop that answers
    them: edit plan.md, then resume the planner over it (which re-reads the
    file). That loop is why there is no `plan revise` verb.
    """
    with contextlib.suppress(ManifestError):
        mode = read_manifest(layout.session_dir).mode
        session_id = layout.session_id
        if mode == "plan":
            # The plan is the deliverable, printed like an ask prints its
            # answer; the path alone sent the operator to `plan show`.
            with contextlib.suppress(OSError):
                plan = (layout.session_dir / "plan.md").read_text(encoding="utf-8").rstrip()
                if plan:
                    reporter.out(f"\n{plan}")
            reporter.out(f"\nedit:     agent6 plan edit {session_id}")
            reporter.out(f'revise:   agent6 resume {session_id} --steer "<what to change>"')
            reporter.out(f"execute:  agent6 run --from-plan {session_id}")
        elif mode == "ask":
            reporter.out(f'\nnext:  agent6 run --from {session_id} "<what to do with it>"')


def _print_unknown_baseline(
    result: SessionResult, *, layout: SessionLayout, reporter: Reporter
) -> None:
    """On a red gate nothing observed at the base, say so and name the check.

    A run whose FIRST verify ran against an unmodified tree already answered
    this for free, and ends `gate_red_at_base`. This is the other case: the
    model edited before it ever verified, so nobody knows. Saying "I do not
    know" beats what used to happen here -- a second full gate run in the
    teardown, holding the checkout for up to verify_timeout_s after the run
    visibly ended, whose own failures repeatedly answered the question wrong.
    """
    if result.verified != "failed" or result.reason == "gate_red_at_base":
        return
    gate = ()
    base = ""
    with contextlib.suppress(ManifestError):
        m = read_manifest(layout.session_dir)
        gate, base = m.workflow.verify_command, (m.forked_from_sha or m.base_sha)
    if not (gate and base):
        return
    reporter.out(
        f"\nthe gate is red, and nothing checked it before this run started ({base[:12]})."
    )
    # A worktree at the base sha, NOT `git stash`: the run's work is COMMITTED
    # on its branch, so a stash saves nothing, exits 0, and runs the gate
    # against the very commits it was meant to exclude -- reading back as "red
    # without my changes too".
    reporter.out("  to see whether this run caused it, check out the base commit somewhere else:")
    reporter.out(f"    git worktree add /tmp/agent6-base {base[:12]} \\")
    reporter.out(f"      && (cd /tmp/agent6-base && {shlex.join(gate)})")


def _print_unverified(result: SessionResult, *, layout: SessionLayout, reporter: Reporter) -> None:
    """A gated finish nothing observed: say what is missing, not "red"."""
    if result.verified != "unverified":
        return
    reporter.out(
        "\nnothing verified the final tree: no verify ran this leg, or edits landed"
        " after the last green."
    )
    reporter.out(f'  resume and run the gate:  agent6 resume {layout.session_id} --steer "verify"')


def _print_stale_gate(result: SessionResult, *, reporter: Reporter) -> None:
    """Surface a proposed gate replacement, and say plainly that nothing moved.

    The worker may declare the configured gate stale instead of reverting
    correct work to satisfy it. Applying the proposal is the operator's call,
    so this prints the exact command rather than doing anything.

    Never over a GREEN gate: a proposal alongside a gate that just passed asks
    the operator to replace something nothing found fault with. Red and
    unverified both surface it -- "cannot run at all" is a stale claim from a
    gate that never produced an observation.
    """
    if not result.stale_gate or result.verified not in ("failed", "unverified"):
        return
    reporter.out("\nthe worker says this run's verify gate no longer matches the task:")
    reporter.out(f"  it proposes: {result.stale_gate}")
    reporter.out("  nothing changed. To adopt it:")
    # `verify_command` is argv, so `config set` takes a JSON array: the shell
    # string the worker proposes is rejected as "not a valid tuple". Tokenised
    # by the one owner the inference uses, so a proposal with a pipeline or an
    # `&&` becomes `sh -c "..."` -- splitting it word by word printed a command
    # that installs a gate handing `&& ruff check` to pytest as arguments.
    argv = json.dumps(list(line_to_argv(result.stale_gate) or ()))
    reporter.out(f"    agent6 config set workflow.verify_command {shlex.quote(argv)}")


def print_session_end(
    result: SessionResult,
    *,
    layout: SessionLayout,
    cwd: Path,
    budget: BudgetTracker,
    console_stream: bool,
    reporter: Reporter,
) -> None:
    """One composed end-of-run block: outcome, summary, cost, and the next step.

    When the live ConsoleView already rendered the `● done <summary>` terminator
    (console_stream), this omits the summary and just adds what the stream
    lacks: cost and the branch / next-step footer."""
    # Read the outcome from the SAME fold `agent6 sessions` uses, not from
    # result.completed: completed means "the agent finished deliberately", which
    # is true for a finish_session even when verify never went green. status_word off
    # result.completed then prints "passed" while runs list reads the session.end
    # event's real all_passed and prints "finished" -- the exact disagreement
    # status_word exists to prevent. summarize_session_dir folds that event, so the
    # console headline and the listing can never diverge.
    summary = summarize_session_dir(layout.session_dir)
    word, reason = summary.status, summary.reason
    if not console_stream:
        # Headless: no ConsoleView ran, so this block is the only end output.
        headline = word if not reason else f"{word} · {reason.replace('_', ' ')}"
        reporter.out(f"\n{headline}")
        if result.summary:
            reporter.out(f"  {result.summary}")
    elif result.summary and result.reason not in ("finish_session", "finish_planning"):
        # The stream's done line carries the finish summary only for a clean
        # finish (pairing an earlier finish's text with a failure would read as
        # success), and session.end carries no message, so a failure's reason --
        # the URL, the errno, the budget line -- reaches the operator only here.
        reporter.out(f"  {result.summary}")
    reporter.out("")
    if unreachable := _sandbox_unreachable_tools(layout):
        # One remedy for the set: printed per binary, three unreachable tools
        # repeated the same four bullets three times.
        names = ", ".join(f"`{b}`" for b in unreachable)
        verb = "is" if len(unreachable) == 1 else "are"
        reporter.out(
            f"WARNING: {names} {verb} installed on this machine but did not work"
            " inside agent6's sandbox."
        )
        reporter.out(
            "  Likely a per-user or version-manager install (rustup, pyenv, nvm)"
            " whose toolchain the sandbox does not expose. Fix options:"
        )
        reporter.out("    - run them from a clean shell (a system-wide install)")
        reporter.out("    - install them into a standard bin dir (~/.local/bin, /usr/local/bin)")
        reporter.out("    - grant their real directories via [sandbox].extra_read_paths")
        reporter.out("    - run with --dangerously-disable-sandbox")
    _print_next_session(layout, reporter=reporter)
    _print_unknown_baseline(result, layout=layout, reporter=reporter)
    _print_unverified(result, layout=layout, reporter=reporter)
    _print_stale_gate(result, reporter=reporter)
    reporter.cost(budget.format_summary())
    _print_run_total_across_legs(layout, reporter=reporter)
    _print_run_branch_footer(result, layout=layout, cwd=cwd, reporter=reporter)


def _print_run_branch_footer(
    result: SessionResult, *, layout: SessionLayout, cwd: Path, reporter: Reporter
) -> None:
    """The where-are-my-changes footer: merged, on the run branch, uncommitted,
    or a resume hint. Every claim is checked against git reality -- merge/diff
    are only offered for a branch that actually exists."""
    run_branch = ""
    base_branch = ""
    merged_into = ""
    manifest: SessionManifest | None = None
    with contextlib.suppress(ManifestError):
        manifest = read_manifest(layout.session_dir)
        run_branch = manifest.run_branch or ""
        base_branch = manifest.base_branch
        if manifest.merged is not None and merge_stamp_holds(
            cwd, manifest.session_id, run_branch, manifest.merged.tip
        ):
            merged_into = manifest.merged.into or base_branch
    if result.completed and manifest is not None and manifest.git_control == "model":
        # The model managed git: report where IT left the checkout; there is
        # no agent6 branch to merge or diff.
        current = ""
        head = ""
        with contextlib.suppress(GitError):
            st = git_status(cwd)
            current = st.branch
            head = st.head_sha[:12]
        where = current or head or "the current checkout"
        reporter.out(
            f'\ngit was model-controlled ([git].control = "model"); its work is on {where}'
        )
        reporter.out("  inspect it with plain git (log/diff); sessions merge does not apply")
    elif result.completed and run_branch and merged_into:
        # auto_merge already merged this branch into the base (and auto_prune may
        # have deleted it); don't tell the operator to merge it again.
        reporter.out(f"\nchanges merged into {merged_into}")
        reporter.out(f"  inspect:     agent6 sessions diff {layout.session_id}")
    elif result.completed and run_branch and branch_exists(cwd, run_branch):
        reporter.out(f"\nchanges are on {run_branch}")
        reporter.out(f"  merge with:  agent6 sessions merge {layout.session_id}")
        reporter.out(f"  inspect:     agent6 sessions diff {layout.session_id}")
        # The chain never switches branches, but an operator who checked the
        # run branch out themselves should know how to leave it, or the next
        # run stacks on it and merge/prune defaults quietly shift.
        current = ""
        with contextlib.suppress(GitError):
            current = git_status(cwd).branch
        if current == run_branch and base_branch and base_branch != run_branch:
            reporter.out(f"  you are on {run_branch}; return with: git switch {base_branch}")
    elif result.completed and run_branch:
        # branch_per_run promised agent6/<id> but no commit ever reached it (an
        # update-ref failure the loop's best-effort commit absorbed, or nothing
        # to commit). Stranded edits are a real failure (and exit code 5, via
        # the same predicate); a clean tree means the run recorded nothing. A
        # tree git cannot READ gets the honest unknown, never a claim.
        try:
            exclude = read_untracked_at_start(layout.session_dir)
            tree_clean: bool | None = git_status(cwd, exclude=exclude).is_clean
        except GitError as exc:
            tree_clean = None
            reporter.out(
                f"\ncould not check the working tree (git failed: {exc}); inspect it manually."
            )
        if tree_clean is not None and stranded_edits(result, layout, cwd):
            reporter.out(
                f"\nWARNING: the run finished with no commit on {run_branch},"
                " so the branch was never created."
            )
            reporter.out(
                "  Edits are left uncommitted in the working tree (the commit failed;"
                " see the run log)."
            )
            reporter.out(f"  retry after fixing the cause:  agent6 resume {layout.session_id}")
        elif tree_clean is True:
            reporter.out("\nno changes were committed")
    elif not result.completed:
        reporter.out(f"\nresume with:  agent6 resume {layout.session_id}")


def _print_run_total_across_legs(layout: SessionLayout, *, reporter: Reporter) -> None:
    """After the leg's token+cost banner: the run's true cumulative spend when
    resume legs precede this one. The tracker is per-leg (each resume starts a
    fresh budget), so its "TOTAL" line undersells a resumed run without this."""
    scan = scan_session_log(layout.session_dir / LOGS_NAME)
    if scan.legs > 1 and scan.cost_usd is not None:
        cost = format_cost(scan.cost_usd, partial=scan.usd_partial)
        reporter.out(f"  RUN TOTAL (all {scan.legs} legs): {cost}")


def print_interrupt_end(
    *, layout: SessionLayout, cwd: Path, budget: BudgetTracker, reporter: Reporter
) -> None:
    """After a Ctrl-C interrupt: the cost so far, the resume hint, and the
    branch-return hint. The interrupt cuts the run before `print_session_end`, so
    without this the user saw only "run interrupted" -- no spend, no way to pick
    the (auto-committed, resumable) work back up, and no note they were left on
    the run branch. Mirrors the not-completed footer of `print_session_end`."""
    reporter.out("")
    reporter.cost(budget.format_summary())
    _print_run_total_across_legs(layout, reporter=reporter)
    reporter.out(f"\nresume with:  agent6 resume {layout.session_id}")
    run_branch = ""
    base_branch = ""
    with contextlib.suppress(ManifestError):
        manifest = read_manifest(layout.session_dir)
        run_branch = manifest.run_branch or ""
        base_branch = manifest.base_branch
    if run_branch:
        current = ""
        with contextlib.suppress(GitError):
            current = git_status(cwd).branch
        if current == run_branch and base_branch and base_branch != run_branch:
            reporter.out(f"  you are on {run_branch}; return with: git switch {base_branch}")


def finalize_auto_merge(
    cwd: Path,
    *,
    layout: SessionLayout,
    cfg: Config,
    reporter: Reporter,
    budget: BudgetTracker | None = None,
    events: EventSink | None = None,
) -> None:
    """After a successful run, land the run branch on its base using
    git.merge_strategy (git.auto_merge). Reads the run context from the manifest, so
    run + resume share it. Ref plumbing only: the checkout is never switched and
    the worktree (which carries the run's work) is no obstacle. With
    branch_per_run off the hidden chain ref is merged instead. Non-fatal and
    best-effort: on conflict or error the run's refs are left intact and the
    message says how to merge by hand."""
    try:
        manifest = read_manifest(layout.session_dir)
    except ManifestError:
        return
    base_branch = manifest.base_branch
    # The visible branch when there is one, else the run's chain ref; a run
    # that recorded no commits (unborn ref) has nothing to land.
    run_branch = manifest.run_branch or chain_ref_for(manifest.session_id)
    if not base_branch or chain_tip(cwd, run_branch) is None:
        return
    identity = CommitIdentity(
        name=cfg.git.commit.name,
        email=cfg.git.commit.email,
        trailer=render_commit_trailer(
            cfg.git.commit.trailer,
            models=worker_models(tail_events(layout.session_dir / LOGS_NAME, follow=False))
            or ((manifest.models.driver.model,) if manifest.models.driver else ()),
        ),
    )
    try:
        verify_git_identity(cwd, identity)
    except GitError as exc:
        reporter.note(
            f"auto_merge skipped: {exc}",
        )
        return
    outcome = execute_merge(
        cwd,
        layout=layout,
        manifest=manifest,
        run_branch=run_branch,
        target=base_branch,
        base_sha=manifest.base_sha,
        strategy=cfg.git.merge_strategy,
        message=None,
        cfg=cfg,
        identity=identity,
        budget=budget,
        events=events,
        warn=reporter.note,
    )
    if outcome.status == "merged":
        reporter.note(
            f"auto_merged {run_branch} into {base_branch} "
            f"({cfg.git.merge_strategy}) -> {outcome.merged_sha[:12]}"
        )
        if kept := left_behind_line(base_branch, outcome):
            reporter.note(kept)
    elif outcome.status == "noop":
        reporter.note(f"{noop_merge_line(run_branch, base_branch, outcome)}.")
    elif outcome.status == "conflict":
        reporter.note(
            f"auto_merge into {base_branch} hit conflicts "
            f"({', '.join(outcome.conflicts)}); nothing was moved and {run_branch} is "
            f"intact. Merge by hand:\n    git merge {run_branch}"
        )
    else:
        reporter.note(
            f"auto_merge failed: {outcome.error}",
        )
    if outcome.stamp_error:
        reporter.note(
            f"merge record could not be written: {outcome.stamp_error};"
            " `sessions prune` will call this branch unmerged"
        )
    # A recorded merge takes one post-merge path, whatever it added.
    # auto_prune is a branch verb: with branch_per_run off there is no
    # branch, and the chain ref stays as the run's record (sessions rm).
    landed = outcome.status == "merged" or outcome.recorded
    if landed and cfg.git.auto_prune and manifest.run_branch:
        if delete_branch_if_merged(cwd, run_branch):
            reporter.note(f"auto_pruned {run_branch}")
        else:
            reporter.note(
                f"auto_prune kept {run_branch} (squash-merged, unreachable; "
                f"remove with: git branch -D {run_branch})"
            )


def _stash_apply_cmd(cwd: Path, sha: str, base_branch: str) -> str:
    """The manual-recovery command for a stash, worded once for every caller.

    Always apply-by-SHA: a positional `pop 'stash@{N}'` printed now but run
    later restores whatever sits at that position by then, which is how a
    bystander's stash got applied and the pre-run work stayed hidden. The
    chain never moves the checkout, so a `git checkout <base>` prefix appears
    only when the operator is on some other branch right now."""
    apply = f"git stash apply {sha}"
    current = ""
    with contextlib.suppress(GitError):
        current = git_status(cwd).branch
    return f"git checkout {base_branch} && {apply}" if current != base_branch else apply


def stash_recovery_hint(cwd: Path, *, session_id: str, base_branch: str) -> str | None:
    """How to restore this run's pre-run auto-stash by hand, or None when the
    run pushed no stash. For callers that must tell the operator where their
    work went without restoring it (a detached continuation: the run is still
    going, so the stash has to wait)."""
    entry = find_stash(cwd, auto_stash_message(session_id))
    if entry is None:
        return None
    return _stash_apply_cmd(cwd, entry.sha, base_branch)


def finalize_auto_stash(
    cwd: Path,
    *,
    base_branch: str,
    run_branch: str | None,
    auto_pop: bool,
    session_id: str,
    exclude: Collection[str] = (),
    reporter: Reporter,
) -> None:
    """Restore or report the pre-run auto-stash so the user's work is never left in a
    hidden stash. With auto_pop off, print how to pop it. With auto_pop on, pop it
    onto the base branch when that is safe (clean worktree, conflict-free apply);
    otherwise leave the stash with a message. Never reset --hard (refused).
    *exclude* is the run's `untracked_at_start`: the operator's own untracked
    files do not make the tree unclean for the pop.

    The stash is found by the run-id message the run pushed it with, and
    restored by its immutable sha, never by position: a stash pushed DURING
    the run sat at stash@{0}, so a positional pop restored the wrong work and
    left the pre-run work hidden. The printed manual-recovery hint applies by
    sha too (`git stash apply <sha>`), which stays correct however the
    stash stack shifts later -- a positional `pop 'stash@{N}'` printed now
    but run after another stash push would restore the wrong one."""
    message = auto_stash_message(session_id)
    entry = find_stash(cwd, message)
    if entry is None:
        reporter.note("pre-run auto-stash not found (already restored?); nothing to pop")
        return
    # apply-by-sha is identity-stable; drop it yourself once you've confirmed.
    apply = f"git stash apply {entry.sha}"
    recover = _stash_apply_cmd(cwd, entry.sha, base_branch)
    if not auto_pop:
        reporter.note(f"pre-run changes are stashed; restore them with: {recover}")
        return
    try:
        st = git_status(cwd, exclude=exclude)
    except GitError:
        st = None
    if st is None or not st.is_clean:
        reporter.note(f"pre-run changes left stashed (worktree not clean); restore with: {recover}")
        return
    if run_branch and st.branch == run_branch:
        if not branch_exists(cwd, base_branch):
            reporter.note(
                f"base branch {base_branch} no longer exists; pre-run changes left "
                f"stashed (recover with: {apply})"
            )
            return
        try:
            create_branch(cwd, base_branch)  # checks out the existing base branch
        except GitError as exc:
            reporter.note(
                f"could not switch to {base_branch} to restore the stash ({exc}); "
                f"restore with: {recover}"
            )
            return
    try:
        restored = restore_stash(cwd, entry)
    except GitError as exc:
        # The apply itself landed; what failed is putting back a concurrent
        # stash the raced drop displaced. Say both -- finalization continues.
        reporter.note(f"restored your pre-run changes onto {base_branch}, but {exc}")
        return
    if restored:
        reporter.note(
            f"restored your pre-run changes onto {base_branch}",
        )
    else:
        reporter.note(
            "restoring your pre-run changes hit a conflict; resolve the markers"
            f" (your stash is preserved; re-apply with: git stash apply {entry.sha})"
        )


def hook_env(**agent6_vars: str) -> dict[str, str]:
    """The environment for an operator notify hook: the shared curated base
    plus the given `AGENT6_*` facts. The one owner for both hooks
    (`[notify].on_complete` here, `[machine.notify].on_event` in
    `app/machine/_preflight.py`), so their env-scope claims cannot drift."""
    return curated_env(extra=agent6_vars)


def run_notify_hook(
    argv: Sequence[str],
    env: dict[str, str],
    *,
    timeout_s: float,
    label: str,
    note: Callable[[str], None],
) -> None:
    """Run one operator notify hook. The one runner behind both, because the
    two drifted in both directions: one leaked the hook's stdout into the
    parent's (under `agent6 acp` that is the JSON-RPC stream, and one printed
    line desynchronises it), the other swallowed a non-zero exit, and a hook
    that fails silently stops notifying without anyone noticing.

    The argv is operator-controlled, never LLM output, so it runs on the host
    outside the jail. A failure is reported and never changes the exit code."""
    try:
        res = subprocess.run(
            list(argv),
            stdout=subprocess.DEVNULL,
            env=env,
            timeout=timeout_s,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        note(f"{label} failed: {exc}")
        return
    if res.returncode != 0:
        note(f"{label} exited {res.returncode}")


def fire_notify_hook(
    notify: NotifyConfig,
    *,
    session_id: str,
    session_dir: Path,
    ok: bool,
    reason: str,
    verified: str,
    reporter: Reporter,
) -> None:
    """Run the operator-configured post-completion hook.

    The argv comes from `[notify].on_complete` in your config; see
    `run_notify_hook` for how it runs.
    """
    if not notify.on_complete:
        return
    env = hook_env(
        AGENT6_SESSION_ID=session_id,
        # OK = the agent stopped deliberately; VERIFIED = what the gate said
        # (passed / failed / not_applicable). A hook that wants "green" reads
        # the second: OK alone is true for a finish over a red verify.
        AGENT6_SESSION_OK="1" if ok else "0",
        AGENT6_SESSION_VERIFIED=verified,
        AGENT6_SESSION_REASON=reason,
        AGENT6_SESSION_DIR=str(session_dir),
    )
    run_notify_hook(
        notify.on_complete,
        env,
        timeout_s=notify.timeout_s,
        label="notify.on_complete",
        note=reporter.note,
    )
