# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Pre-loop guards shared by `agent6 run`/`resume`: refusals, startup
warnings, branch-base resolution, and per-run verify-command resolution.
The interactive confirm prompts stay in `ui/cli/_preflight` (they own the
terminal) and are injected by the front-end."""

from __future__ import annotations

import contextlib
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from agent6.app._setup import apply_git_ops_policy
from agent6.app.providers import (
    InstrumentedProvider,
    build_role_provider,
)
from agent6.app.reporter import STDIO_REPORTER, Reporter
from agent6.budget import BudgetTracker
from agent6.config import Config, plan_metered
from agent6.events import EventSink
from agent6.git_ops import (
    CommitIdentity,
    GitError,
    chain_ref_for,
    chain_tip,
    is_git_repo,
    run_branch_for,
    verify_git_identity,
    worktree_matches,
)
from agent6.git_ops import (
    status as git_status,
)
from agent6.models.pricing import lookup_price
from agent6.providers import TranscriptSink
from agent6.sessions.ipc import AWAY_MODES, effective_run_commands
from agent6.sessions.manifest import ManifestError, read_manifest
from agent6.tools.schema import UserQuestion
from agent6.verify_infer import VERIFY_INFER_SYSTEM_PROMPT, infer_verify_command, read_agents_md
from agent6.viewmodel.listing import session_dirs
from agent6.workflows.review import parse_seat_spec


class SessionRefused(Exception):
    """A preflight refusal already reported through the Reporter; the caller
    returns `rc` as the process exit code."""

    def __init__(self, rc: int) -> None:
        super().__init__(f"session refused (exit {rc})")
        self.rc = rc


def budget_preflight(
    cfg: Config,
    extra_routes: Iterable[tuple[str, str]] = (),
    *,
    reporter: Reporter = STDIO_REPORTER,
) -> str | None:
    """Budget refusals + notices over every statically reachable model,
    before any spend: the resolved role models, any model a `[review].seats`
    spec pins, and *extra_routes* (a machine's per-state `(provider, model)`
    pins; an unknown provider rides as "" and is judged by price data alone).

    `max_tokens_fallback = 0` refuses when a reachable model cannot be
    metered (zero unmetered tokens allowed); `max_usd = 0` refuses when one
    CAN be (a run-nothing-metered rig). Otherwise an unpriced model gets a
    one-line notice naming the fallback bound that covers it. Models chosen
    later (a `/parallel` lane spec) are caught by the tracker's runtime
    backstop instead."""
    routes = {(rm.provider, rm.model) for rm in cfg.models.configured().values()}
    for spec in cfg.review.seats:
        _persona, seat_provider, seat_model = parse_seat_spec(spec)
        if seat_model:
            routes.add((seat_provider, seat_model))
    routes.update((prov, m) for prov, m in extra_routes if m)

    # Plan-metered routes (a ChatGPT or Claude subscription) live in the
    # percent ledger: they are never "unpriced fallback" spend, and their own
    # zero-refusal mirrors the siblings below.
    plan_models = sorted({m for prov, m in routes if plan_metered(cfg.providers.get(prov))})
    models = {m for prov, m in routes if not plan_metered(cfg.providers.get(prov))}
    if cfg.budget.max_percent == 0.0 and plan_models:
        return (
            "[budget].max_percent is 0 (plan-metered calls refused), but "
            f"{', '.join(repr(m) for m in plan_models)} route"
            f"{'s' if len(plan_models) == 1 else ''} through a subscription plan."
            " Raise max_percent or reroute those roles."
        )
    unpriced = sorted(m for m in models if lookup_price(m) is None)
    priced = sorted(m for m in models if lookup_price(m) is not None)
    if cfg.budget.max_tokens_fallback == 0 and unpriced:
        return (
            "[budget].max_tokens_fallback is 0 (unmetered calls refused), but "
            f"{', '.join(repr(m) for m in unpriced)} carr{'ies' if len(unpriced) == 1 else 'y'}"
            " no price data. Raise max_tokens_fallback or use priced models."
        )
    if cfg.budget.max_usd == 0.0 and priced:
        return (
            "[budget].max_usd is 0 (metered calls refused), but "
            f"{', '.join(repr(m) for m in priced)} {'is' if len(priced) == 1 else 'are'} priced."
            " Raise max_usd or use unpriced/local models."
        )
    if unpriced and cfg.budget.max_usd != 0.0:
        fb = cfg.budget.max_tokens_fallback
        bound = "unlimited tokens" if fb == -1 else f"{fb:,} fallback tokens"
        reporter.note(
            f"{', '.join(repr(m) for m in unpriced)} ha"
            f"{'s' if len(unpriced) == 1 else 've'} no price data: that spend is not"
            f" metered by max_usd and is bounded by {bound} instead."
        )
    if plan_models:
        pct = cfg.budget.max_percent
        bound = "the plan itself" if pct == -1 else f"max_percent {pct:g} points per run"
        reporter.note(
            f"{', '.join(repr(m) for m in plan_models)} draw"
            f"{'s' if len(plan_models) == 1 else ''} on a subscription plan"
            f" (no dollars; bounded by {bound})."
        )
    return None


def warn_if_prompt_override_incomplete(cfg: Config, *, reporter: Reporter = STDIO_REPORTER) -> None:
    """Warn when a custom `prompt.system_prompt_file` omits the core tool
    contracts the worker needs: `finish_session` is the only clean exit, and an
    edit primitive (`apply_edit`/`apply_patch`) is needed to do work. The
    override is advanced + operator-owned, so we don't block -- just flag the
    likely-broken case loudly and point at `agent6 prompt show`."""
    path = cfg.prompt.system_prompt_file
    if not path:
        return
    try:
        text = Path(path).expanduser().read_text(encoding="utf-8")
    except OSError:
        return  # config validation already enforces existence; nothing to add
    missing = [t for t in ("finish_session",) if t not in text]
    if "apply_edit" not in text and "apply_patch" not in text:
        missing.append("apply_edit/apply_patch")
    if missing:
        # Name every capability that is actually absent, not just one of them, so
        # a prompt missing both finish_session AND an edit primitive reads correctly.
        actions = []
        if "finish_session" in missing:
            actions.append("terminate")
        if "apply_edit/apply_patch" in missing:
            actions.append("make edits")
        reporter.warn(
            f"custom system_prompt_file ({path}) does not mention "
            f"{', '.join(missing)}; the worker may not know how to "
            f"{' or '.join(actions)}. The override "
            "replaces the built-in run-mode base, so the tool contracts are yours "
            "to preserve. Inspect the assembled prompt with `agent6 prompt show`."
        )


@dataclass(frozen=True, slots=True)
class GitPreflight:
    """Where the run starts: HEAD + branch at submission (empty for ask, which
    is read-only and may run outside a repo)."""

    base_sha: str
    base_branch: str


def git_preflight(
    cwd: Path,
    cfg: Config,
    mode: str,
    *,
    confirm_run_on_run_branch: Callable[[str], bool],
    reporter: Reporter,
) -> GitPreflight:
    """The git checks a session needs before it creates anything, raising
    :class:`SessionRefused` on each already-reported refusal.

    The auto-commit-on-verify-pass behaviour requires a clean working tree, so
    the same git assumptions apply; skipping these left first-time runs
    crashing on dirty-tree or missing-identity errors deep into a paid run.
    The egress policy is applied by the lifecycle's own config, not whichever
    front-end got here: `ui/cli` set it and `agent6 acp` did not, so a repo
    that opted into its own hooks silently kept them off under an editor -- a
    knob `config show` reports and one surface ignored.
    """
    apply_git_ops_policy(cfg)
    identity = CommitIdentity(name=cfg.git.commit.name, email=cfg.git.commit.email)
    # ask is read-only and may run outside a git repo (e.g. agent6 self-help),
    # so it skips the commit-oriented git pre-flight entirely.
    if mode == "ask":
        return GitPreflight(base_sha="", base_branch="")
    try:
        verify_git_identity(cwd, identity)
        # Captured BEFORE a run branch exists, so `agent6 sessions diff <id>`
        # knows where the run started.
        pre_status = git_status(cwd)
    except GitError as exc:
        reporter.error(str(exc))
        raise SessionRefused(2) from exc
    # Starting a run while checked out on ANOTHER run's branch (agent6/<id>) is
    # usually a slip -- the operator forgot to merge or switch back -- so the new
    # run would pile on top of an unmerged one. Confirm; they may instead intend
    # to continue that line with a fresh session, in which case proceed.
    if (
        mode == "run"
        and pre_status.branch.startswith("agent6/")
        and not confirm_run_on_run_branch(pre_status.branch)
    ):
        reporter.note(
            "aborted. Merge (agent6 sessions merge) or switch branches first, then re-run."
        )
        raise SessionRefused(2)
    return GitPreflight(base_sha=pre_status.head_sha, base_branch=pre_status.branch)


# What a run does with the operator's uncommitted changes to tracked files.
# Untracked files are never in question: the run leaves them out of its
# commits and dirty checks (`untracked_at_start`).
DirtyTreeChoice = Literal["stash", "include", "cancel"]
DIRTY_TREE_OPTIONS: tuple[str, ...] = ("stash", "include", "cancel")


def unmerged_run_holding_the_tree(
    cwd: Path, state_dir: Path, *, except_id: str, modified: Sequence[str]
) -> str:
    """The id of the newest earlier unmerged run whose chain tip holds exactly
    the working tree's content of the *modified* files, else "". A run's
    edits sit uncommitted on the checkout until its branch is merged, so the
    next run's dirty-tree question would otherwise call agent6's own last
    work "uncommitted changes" as if the operator had left them; naming the
    run points at the merge instead. Compared per file, so a commit that
    landed on the base since (any other file) does not hide the match."""
    if not modified:
        return ""
    for d in session_dirs(state_dir, buckets=("runs",))[:10]:
        if d.name == except_id:
            continue
        with contextlib.suppress(ManifestError):
            if read_manifest(d).merged is not None:
                continue
        tip = chain_tip(cwd, chain_ref_for(d.name))
        if tip is None:
            continue
        try:
            if worktree_matches(cwd, tip, modified):
                return d.name
        except GitError:
            return ""
    return ""


def _dirty_tree_listing(paths: Sequence[str], *, cap: int = 10, unmerged_run: str = "") -> str:
    # No leading indentation: a modal's text pane drops it, so a listing that
    # depends on it reads differently per surface.
    n = len(paths)
    head = f"{n} tracked {'file has' if n == 1 else 'files have'} uncommitted changes"
    head += (
        f" (the unmerged work of run {unmerged_run}, on {run_branch_for(unmerged_run)}):"
        if unmerged_run
        else ":"
    )
    lines = [f"- {p}" for p in paths[:cap]]
    if n > cap:
        lines.append(f"- ... {n - cap} more")
    return "\n".join([head, *lines])


def dirty_tree_question(paths: Sequence[str], *, unmerged_run: str = "") -> UserQuestion:
    """The start question a run with uncommitted tracked changes asks the
    operator (over the same channel as `ask_user`); the answer's first word is
    the choice, anything else cancels. *unmerged_run* names the earlier run
    whose branch holds exactly these changes, when one does."""
    merge_hint = (
        f"cancel: park the run; `agent6 sessions merge {unmerged_run}` lands them, then resume it"
        if unmerged_run
        else "cancel: park the run; resume it once they are committed or stashed"
    )
    return UserQuestion(
        question=(
            f"{_dirty_tree_listing(paths, unmerged_run=unmerged_run)}\n"
            "How should this run treat them?\n"
            "stash: set them aside for the run (applied back at the end when the tree is"
            " clean, else the `git stash apply` line is printed)\n"
            "include: the run's first commit records them with its own work\n"
            f"{merge_hint}"
        ),
        options=DIRTY_TREE_OPTIONS,
    )


def dirty_tree_choice(answer: str) -> DirtyTreeChoice:
    word = answer.strip().split(maxsplit=1)[0].rstrip(":").lower() if answer.strip() else ""
    if word == "stash":
        return "stash"
    if word == "include":
        return "include"
    return "cancel"


def dirty_tree_refusal(paths: Sequence[str], *, unmerged_run: str = "") -> str:
    """The refusal when nobody can answer :func:`dirty_tree_question`."""
    settle = (
        f" `agent6 sessions merge {unmerged_run}` lands them; or"
        if unmerged_run
        else " Commit or stash them first, or"
    )
    return (
        f"{_dirty_tree_listing(paths, unmerged_run=unmerged_run)}\n"
        "This run has no terminal and no front-end to ask how to treat them."
        f"{settle} decide in config: [git].auto_stash = true"
        " stashes them for the run; [git].require_clean_worktree = false lets the run's"
        " first commit record them."
    )


def git_repo_refusal(cwd: Path) -> str | None:
    """Refuse a workspace that is not a git repository, naming the fix.

    A clean early exit instead of the misleading "Git identity not configured"
    error (when there's no global identity) or an ugly failure deeper in the
    run. agent6 needs git to branch, commit per step, and let the user
    review/revert what the agent did.

    This is also the WALL on what becomes the model's workspace: whatever
    directory a run starts in is what the jail mounts writable. Every front-end
    that chooses one has to pass it through here -- `agent6 acp` takes that
    directory from the editor over the wire, and without this a client could
    point a run at any absolute path.

    Returns the message, or None when *cwd* is usable.
    """
    if not cwd.is_dir():
        # Asked git first, `subprocess` could not chdir into a missing
        # directory and the FileNotFoundError surfaced as an opaque
        # internal error. A stale workspace path is the ordinary editor
        # mistake, and it deserves the same named refusal as a wrong one.
        return f"{cwd} is not a directory."
    if is_git_repo(cwd):
        return None
    return (
        f"{cwd} is not a git repository.\n"
        "agent6 needs git here to create a run branch, commit each step, and let"
        " you review or revert what the agent did.\n"
        "  Fix: run `agent6 init` (it offers to set up git for you), or\n"
        '       `git init && git add -A && git commit -m "initial commit"`.'
    )


def require_git_repo(cwd: Path, *, reporter: Reporter = STDIO_REPORTER) -> bool:
    """:func:`git_repo_refusal` for a front-end that prints and branches."""
    refusal = git_repo_refusal(cwd)
    if refusal is None:
        return True
    reporter.refuse(refusal)
    return False


def headless_approval_refusal(
    cfg: Config, *, tui_enabled: bool, away: str, can_ask: bool, clamped: bool = False
) -> str | None:
    """Refuse a run that would block forever waiting to be approved.

    `run_commands = "ask"` needs someone to answer. With no TUI, no way for the
    front-end to ask, and no away-mode telling us what an absent operator meant,
    the first command PAUSES indefinitely -- and the verify gate is a command
    too, so nearly every run hits this, every `/parallel` lane included.
    Refuse with the fix rather than hang: a run that cannot ask should not start.

    *can_ask* is the front-end's own declaration. Testing the tty here instead
    made this the CLI's question rather than the surface's, so `agent6 acp` --
    whose stdin is the protocol pipe and which asks over
    `session/request_permission` -- had every run refused before it started.

    An *away* value outside `AWAY_MODES` is a typo, and a typo names no intent:
    it refuses on every surface rather than reading as one.

    *clamped* says this session kind clamps a standing `run_commands = "yes"`
    to `ask` (plan and ask do), so the remedy names the flag, not the config
    value that is already set.

    Returns the message, or None when approval is answerable.
    """
    if cfg.sandbox.run_commands != "ask":
        return None
    if away and away not in AWAY_MODES:
        return (
            f"AGENT6_DETACHED_AWAY={away!r} is not an away-mode, so an absent operator's"
            " intent is unknown and an approval would wait forever.\n"
            f"  - set AGENT6_DETACHED_AWAY={'|'.join(AWAY_MODES)}"
        )
    if tui_enabled or can_ask or away:
        return None
    unattended = (
        "--auto-approve (this session kind clamps a standing sandbox.run_commands = 'yes' to 'ask')"
        if clamped
        else "sandbox.run_commands = 'yes' (or --auto-approve)"
    )
    gate = "" if clamped else ", the verify gate included,"
    return (
        "sandbox.run_commands = 'ask' needs someone to answer, and this run has no"
        f" TUI and no away-mode. Every command{gate} would wait forever.\n"
        f"  - unattended: {unattended}, or 'no' to withhold commands entirely\n"
        "  - attended: start it from a terminal, or set an away-mode"
        f" (AGENT6_DETACHED_AWAY={'|'.join(AWAY_MODES)}) so an absent operator's intent is known"
    )


def drop_gate_if_unrunnable(cfg: Config, *, session_dir: Path, reporter: Reporter) -> Config:
    """Empty the verify command when this LEG cannot run one.

    Every command tool is withheld when the effective policy is `no` -- the
    operator's configured value, a session deny, or an away-mode of deny -- and
    the gate is a command. Keeping it made the leg unwinnable: nothing could go
    green, so nothing committed, and it finished red over work that may be fine.

    Decided ONCE per leg, by whichever lifecycle starts it, because the system
    prompt is frozen from the same config. Runs LAST at leg start -- after
    snapshot reuse and inference -- so nothing hands the gate back. A deny that
    lands MID-leg withdraws the tools (the dispatcher's own filter) but must
    not retroactively make a gate that already ran red look like a run that
    never had one.
    """
    if effective_run_commands(cfg.sandbox.run_commands, session_dir) != "no":
        return cfg
    if cfg.workflow.verify_command:
        reporter.note(
            "commands are withheld, and the verify gate is a command:"
            " running gateless (per-step commits, no green gate)."
        )
    return cfg.with_verify_command(())


def infer_verify_if_unset(
    cfg: Config,
    cwd: Path,
    *,
    mode: str,
    events: EventSink,
    transcript_sink: TranscriptSink,
    budget: BudgetTracker,
    reporter: Reporter = STDIO_REPORTER,
) -> Config:
    """When `workflow.verify_command` is unset for a run/plan, infer one and
    inject it IN-MEMORY (never persisted -- runs do not mutate config).

    Layered cheapest-first (AGENTS.md -> repo signals -> a reviewer-role LLM
    call over the manifests, skipped when there are none to read); see
    `agent6.verify_infer`. Emits `loop.verify_inferred` and
    prints what was picked + that it is per-run. If nothing can be inferred the
    run proceeds GATELESS (no verify gate; the loop commits each editing step).

    `drop_gate_if_unrunnable` runs AFTER this and has the last word: a leg
    that cannot run commands ends gateless, whatever was inferred.
    """
    if mode not in ("run", "plan") or cfg.workflow.verify_command:
        return cfg
    if not cfg.workflow.verify_infer:
        # Pinned gateless: the operator said no gate, so no tier runs and the
        # mid-run adoption stays off too (the loop reads the same knob).
        events.emit("loop.verify_inferred", command=[], source="disabled")
        if mode == "run":
            reporter.note(
                "verify_infer = false: running gateless"
                " (per-step commits, no green gate; nothing is inferred or adopted)."
            )
        return cfg
    agents_md = read_agents_md(cwd)

    def _llm_call(context: str) -> str:
        inner = build_role_provider(cfg, "reviewer", transcript_sink=transcript_sink, budget=budget)
        rm = cfg.models.resolve("reviewer")
        provider = InstrumentedProvider(
            inner=inner,
            role="verify_inferer",
            model=rm.model if rm else "",
            provider_name=rm.provider if rm else "",
            events=events,
            budget=budget,
        )
        resp = provider.call(
            system=VERIFY_INFER_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": context}],
            tools=[],
            max_tokens=512,
            temperature=0.0,
        )
        return resp.text or ""

    inferred = infer_verify_command(cwd, agents_md, llm_call=_llm_call)
    if inferred is None:
        events.emit("loop.verify_inferred", command=[], source="none")
        if mode == "run":
            reporter.note(
                "no verify_command set and none could be inferred; running"
                " gateless\n         (per-step commits, no green gate). If the run"
                " creates a recognizable project, a verify\n         command is"
                " adopted mid-run; pin one with workflow.verify_command."
            )
        return cfg
    events.emit("loop.verify_inferred", command=list(inferred.argv), source=inferred.source)
    reporter.note(
        f"verify_command not set; inferred from {inferred.source}:"
        f" {' '.join(inferred.argv)}\n         (this run only; pin it with"
        " workflow.verify_command in your per-repo config)"
    )
    return cfg.with_verify_command(inferred.argv)
