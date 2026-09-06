# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The two sides of a machine `agent` state: the host-side launcher that
spawns the subprocess, and the runner inside it.

A machine run's engine is a thin supervisor that stays in the host network
namespace and makes no network calls itself. Each `agent` state runs in its own
fresh process (`ui/cli/machine_agent` is the `python -m` entry), independent of
the engine and of sibling `tool` states; like every agent process it runs
unconfined, and the jail bounds the commands it dispatches.

`build_machine_agent_runner` (host side) builds the callable an `agent` state
fires: it spawns the subprocess with a fixed argv, hands it the request via a
temp file, and enforces the timeout by killing the process group. `run_one`
(subprocess side) reads that request, validates its isolation and hide-path
needs while still single-threaded, runs the agent loop to completion, and
writes the result.
`MachineAgentRequest` (here) owns the `request.json` file shape and
`AgentExecResult` (machine/engine.py) owns `result.json`: both sides
serialize/validate through the models, per the IPC rule
(`tests/unit/test_machine_agent_ipc.py` pins the bytes). The live conversation
view is the one presentation piece: `ui/cli` injects `attach_console` so this
module never imports `agent6.ui`.
"""

from __future__ import annotations

import contextlib
import os
import shutil
import signal
import subprocess
import sys
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, ValidationError

from agent6.app._session import tool_result_cap_chars
from agent6.app._setup import apply_git_ops_policy
from agent6.app.confine import check_hide_paths_support, check_network_support
from agent6.app.providers import (
    InstrumentedProvider,
    build_role_provider,
    build_summariser_provider,
    resolve_compaction_thresholds,
    resolve_decompose,
)
from agent6.app.reporter import STDIO_REPORTER, Reporter
from agent6.budget import BudgetTracker
from agent6.config import Config, ConfigError
from agent6.config.layer import load_effective_with_overlay, resolved_state_dir
from agent6.events import EventSink
from agent6.git_ops import (
    CommitIdentity,
    GitError,
    chain_tip,
    checkout_detached,
    fetch_branch,
    machine_branch_for,
    machine_chain_ref_for,
    render_commit_trailer,
)
from agent6.git_ops import status as git_status
from agent6.machine import AgentExecResult, AgentRequest, validate_record_payload
from agent6.providers import Provider, TranscriptSink
from agent6.sandbox.jail import die_with_parent
from agent6.sessions.ipc import (
    await_frontend_reply,
    away_mode,
    clear_pending_answers,
    clear_steer_answer,
    clear_steer_request,
    frontend_is_live,
    read_answer,
    read_question_answers,
    read_steer_answer,
    record_answer,
    steer_request_pending,
)
from agent6.tools.dispatch import ToolDispatcher
from agent6.tools.operator_prompts import (
    ApprovalAnswer,
    ApprovalRequest,
    OperatorPrompts,
    QuestionAnswer,
    QuestionRequest,
)
from agent6.types import IsolationLevel
from agent6.viewmodel.machine_state import Spend, read_budget_totals
from agent6.workflows.loop import Workflow
from agent6.workflows.subrun import SubrunError, clone_workspace


def _no_console(_events: EventSink) -> None:
    """The headless default when no front-end injects a live view."""


class MachineAgentRequest(BaseModel):
    """The `request.json` envelope of the machine-agent subprocess IPC.

    The host runner (`build_machine_agent_runner`) serializes it into the temp
    file the fixed argv (``python -m agent6.ui.cli.machine_agent <request.json>
    <result.json>``) names; the subprocess validates it back and hands it to
    `run_one`. One owner of the file shape per the IPC rule; `result.json` is
    owned the same way by `AgentExecResult`. The files are transient
    per-invocation (both sides are always the same install), and the bytes are
    pinned by `tests/unit/test_machine_agent_ipc.py`.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    cwd: Path
    root: Path
    # The machine's `[config]` overlay, applied over the effective config.
    overlay: dict[str, Any]
    isolation: IsolationLevel
    transcript_dir: Path
    # When set, the subprocess writes a watchable logs.jsonl here (role.*_delta
    # + tool.* events), so `machine create` and live `agent` states are
    # followable in the TUI/web dashboard exactly like a run.
    events_log: Path | None = None
    protect_paths: tuple[Path, ...] = ()
    # Resolved on the host (pre-Landlock, so it sees global git config); the
    # confined subprocess can't read ~/.gitconfig, so its mode="run" commits
    # would otherwise fail with "Author identity unknown". None for read-only
    # (mode="agent"/"machine") states.
    commit_identity: CommitIdentity | None = None
    request: AgentRequest


def _machine_head_sha(root: Path) -> str | None:
    """HEAD at state start, the chain's first parent; None when unreadable
    (an unborn repo roots the chain)."""
    try:
        return git_status(root).head_sha or None
    except (GitError, OSError):
        return None


def _finish_validator(r: AgentRequest) -> Callable[[dict[str, Any] | None], list[str]] | None:
    """The state's finish contract as a loop-injectable check: the same
    validator the engine judges the recorded fact with, over the schemas the
    request carried."""
    name = r.output_schema
    if name is None:
        return None

    def check(payload: dict[str, Any] | None) -> list[str]:
        return validate_record_payload(r.schemas, name, payload, where="finish_session payload")

    return check


def _task_with_contract(r: AgentRequest) -> str:
    """The task the leg runs: the state prompt, plus the finish contract when
    the state declares one, rendered as field: type lines (nested records
    included) so the model needs no guess about the accepted shape."""
    if r.output_schema is None:
        return r.prompt
    lines = [
        f"{r.prompt}",
        "",
        "finish_session must include `result` as a JSON object matching schema"
        f" {r.output_schema!r}:",
    ]
    seen: set[str] = set()
    queue = [r.output_schema]
    while queue:
        name = queue.pop(0)
        if name in seen or name not in r.schemas:
            continue
        seen.add(name)
        parts = []
        for fname, f in r.schemas[name].items():
            t = f.type + (" (optional)" if f.optional else "")
            if f.enum is not None:
                t += " one of [" + ", ".join(f.enum) + "]"
            parts.append(f"{fname}: {t}")
            if f.type in r.schemas:
                queue.append(f.type)
        lines.append(f"  {name} = {{{'; '.join(parts)}}}")
    return "\n".join(lines)


def _result(
    reason: str, payload: dict[str, Any] | None, budget: BudgetTracker | None
) -> AgentExecResult:
    usd = 0.0
    partial = False
    inp = out = 0
    if budget is not None:
        usd, partial = budget.estimate_usd()
        snap = budget.snapshot()
        inp, out = snap.input_total, snap.output_total
    return AgentExecResult(
        reason=reason,
        payload=payload,
        usd=usd,
        usd_partial=partial,
        input_tokens=inp,
        output_tokens=out,
    )


def _apply_operator_env_grants(cfg: Config) -> Config:
    """The supervisor's `machine run --auto-approve` / `--no-commands` choices,
    carried by env like the sandbox setter (operator-only, structurally
    LLM-unreachable; a machine [config] overlay must not and cannot set
    sandbox.*). Upgrades run_commands ask -> yes, never a withheld no, and
    withholds every command tool when the operator asked for that
    (`with_sandbox_overrides`)."""
    return cfg.with_sandbox_overrides(
        auto_approve=os.environ.get("AGENT6_AUTO_APPROVE") == "1",
        no_commands=os.environ.get("AGENT6_NO_COMMANDS") == "1",
    )


@dataclass(frozen=True, slots=True)
class _MachineBridges:
    """The interactivity bridges for one machine `agent` state.

    Answers are read from the per-state dir, but a front-end registers
    a `frontends/` claim on the instance dir, so the liveness gate probes the instance
    dir (`live_dir`). Prompt/answer events go to the per-state log the front-end
    already tails, so its SessionState fold surfaces them like a run's.
    """

    prompts: OperatorPrompts
    steer_requested: Callable[[], bool]
    steer_clear: Callable[[], None]
    steer_prompt: Callable[[], str | None]


def _build_machine_bridges(
    instance_dir: Path, state_dir: Path, events: EventSink
) -> _MachineBridges:
    """Wire run-level approval/question/steer bridges to a machine agent state.

    A live front-end (a `frontends/` claim on the instance dir) is asked in its
    own UI. Otherwise the instance's away-mode governs, exactly as for a
    detached run: a hub-spawned machine carries "wait" (park for the front-end,
    so the claim's timing never decides), and a pure headless machine keeps the
    safe default: deny an approval, answer a question with "", no steer.
    """
    # Crash recovery re-executes the same `<seq>-<state>` dir and its prompt-id
    # counters restart at 1, so an answer file left by the aborted attempt would
    # satisfy this execution's first prompt unseen. Drop the stale bridge state
    # first (front-end claims live on the instance dir, so this touches none).
    clear_pending_answers(state_dir)

    def approve(request: ApprovalRequest, /) -> ApprovalAnswer:
        if frontend_is_live(instance_dir):
            answer = read_answer(state_dir, request.id, live_dir=instance_dir)
            if answer is not None:
                return ApprovalAnswer(record_answer(state_dir, answer, request.scope), "frontend")
        if away_mode(instance_dir) == "wait":
            reply = await_frontend_reply(
                instance_dir,
                lambda: read_answer(
                    state_dir, request.id, timeout_s=20.0, dead_grace_s=8.0, live_dir=instance_dir
                ),
            )
            approved = reply is not None and record_answer(state_dir, reply, request.scope)
            return ApprovalAnswer(approved, "await-frontend")
        return ApprovalAnswer(False, "headless")  # no operator to ask: deny safely

    def ask(request: QuestionRequest, /) -> QuestionAnswer:
        empty = tuple("" for _ in request.questions)
        if frontend_is_live(instance_dir):
            answers = read_question_answers(state_dir, request.id, live_dir=instance_dir)
            if answers is not None:
                return QuestionAnswer(answers, "frontend")
        if away_mode(instance_dir) == "wait":
            # Hub-spawned: park for the front-end rather than inventing "".
            reply = await_frontend_reply(
                instance_dir,
                lambda: read_question_answers(
                    state_dir, request.id, timeout_s=20.0, dead_grace_s=8.0, live_dir=instance_dir
                ),
            )
            return QuestionAnswer(reply if isinstance(reply, tuple) else empty, "await-frontend")
        return QuestionAnswer(empty, "headless")

    prompts = OperatorPrompts(
        approver=approve, questioner=ask, journal=events.emit, session_dir=state_dir
    )

    def steer_requested() -> bool:
        return steer_request_pending(state_dir)

    def steer_clear() -> None:
        clear_steer_answer(state_dir)
        clear_steer_request(state_dir)

    def steer_prompt() -> str | None:
        if not frontend_is_live(instance_dir):
            clear_steer_request(state_dir)
            return None
        answer = read_steer_answer(state_dir, live_dir=instance_dir)
        if answer is None:
            clear_steer_request(state_dir)
        return answer

    return _MachineBridges(prompts, steer_requested, steer_clear, steer_prompt)


def _build_agent_providers(
    cfg: Config,
    req: MachineAgentRequest,
    *,
    budget: BudgetTracker,
    attach_console: Callable[[EventSink], None],
) -> tuple[InstrumentedProvider, Provider, EventSink | None]:
    """The agent state's worker provider (instrumented), its reviewer-role
    summariser, and the optional event sink.

    One TranscriptSink for both (its seq counter is per-instance), each seat
    stamped. An EventSink only when the caller passes events_log; the console
    attach is injected so this module never imports `agent6.ui`.

    ALWAYS streams: machine agents run headless and generate long, and
    OpenRouter-style gateways' SSE heartbeats corrupt a non-streaming body
    mid-read. Streaming also feeds the role.*_delta events."""
    transcript_sink = TranscriptSink(req.transcript_dir)
    inner_provider = build_role_provider(
        cfg, "worker", transcript_sink=transcript_sink, budget=budget
    )
    events_sink = EventSink(req.events_log) if req.events_log is not None else None
    rm = cfg.models.resolve("worker")
    if events_sink is not None:
        attach_console(events_sink)
    provider = InstrumentedProvider(
        inner=inner_provider,
        role="agent",
        model=rm.model if rm is not None else "",
        provider_name=rm.provider if rm is not None else "",
        events=events_sink,
        budget=budget,
        stream_text=True,
    )
    summariser_provider = build_summariser_provider(
        cfg, transcript_sink=transcript_sink, budget=budget, events=events_sink
    )
    return provider, summariser_provider, events_sink


def run_one(
    req: MachineAgentRequest,
    *,
    attach_console: Callable[[EventSink], None] = _no_console,
    reporter: Reporter = STDIO_REPORTER,
) -> AgentExecResult:
    isolation = req.isolation
    r = req.request
    # Config load + per-state overrides run FIRST, and can raise (a bad overlay,
    # or an override naming a provider that isn't configured). Salvage that into
    # a clean AgentExecResult the subprocess writes to result.json -- otherwise
    # the exception escapes to a pydantic traceback + a non-zero exit, and the
    # host runner only recovers it via its missing-result fallback.
    try:
        cfg = load_effective_with_overlay(req.cwd, req.overlay).config.with_machine_agent_overrides(
            provider=r.provider,
            model=r.model,
            effort=r.effort,
            temperature=r.temperature,
            max_usd=r.max_usd,
            max_tokens_fallback=r.max_tokens_fallback,
        )
        cfg = _apply_operator_env_grants(cfg)
    except (ConfigError, ValidationError) as exc:
        reporter.refuse(f"machine agent config error: {exc}")
        return _result("error", None, None)
    apply_git_ops_policy(cfg)
    # A mode="run" state commits its work, but this confined process can't read
    # ~/.gitconfig (not a Landlock read root): export the host-resolved identity
    # so git uses it regardless of where the config lives.
    if req.commit_identity is not None:
        if name := req.commit_identity.name:
            os.environ["GIT_AUTHOR_NAME"] = os.environ["GIT_COMMITTER_NAME"] = name
        if email := req.commit_identity.email:
            os.environ["GIT_AUTHOR_EMAIL"] = os.environ["GIT_COMMITTER_EMAIL"] = email
    # The engine already validated the isolation against the config; re-check
    # defensively and fail closed.
    net_err = check_network_support(cfg, isolation)
    if net_err is not None:
        reporter.refuse(net_err)
        return _result("error", None, None)
    hide_err = check_hide_paths_support(cfg, isolation, req.cwd)
    if hide_err is not None:
        reporter.refuse(hide_err)
        return _result("error", None, None)
    budget = BudgetTracker(
        max_usd=cfg.budget.max_usd,
        max_tokens_fallback=cfg.budget.max_tokens_fallback,
        max_percent=cfg.budget.max_percent,
        allow_paid_credits=cfg.budget.allow_paid_credits,
    )
    provider, summariser_provider, events_sink = _build_agent_providers(
        cfg, req, budget=budget, attach_console=attach_console
    )
    # Re-confirm the cwd-containment invariant at the subprocess boundary
    # (defense in depth, the engine already filtered these).
    root_r = req.root.resolve()
    protect = tuple(rp for p in req.protect_paths if (rp := p.resolve()).is_relative_to(root_r))
    # "machine" (the `machine create` authoring agent) and "agent" (a
    # running machine's `agent` state, unless it opted into mode="run") are
    # read-only structured-output loops: the dispatcher refuses edits AND
    # run_command/run_verify (defense in depth alongside the read-only tool
    # list) and the loop uses a finish_session-focused prompt.
    mode = r.mode
    read_only = mode in ("machine", "agent")
    # Bridge run-level interactivity (approve/ask_user/steer) to a front-end
    # watching this machine: answers land in the per-state dir, the liveness
    # gate probes the instance dir where the front-end registers its claim.
    # Needs a per-state log (events_sink) for the front-end to see the prompt.
    bridges: _MachineBridges | None = None
    if events_sink is not None and req.events_log is not None:
        state_dir = req.events_log.parent
        instance_dir = req.transcript_dir.parent
        bridges = _build_machine_bridges(instance_dir, state_dir, events_sink)
    dispatcher = ToolDispatcher(
        root=req.root,
        config=cfg,
        isolation=isolation,
        prompts=bridges.prompts if bridges is not None else None,
        events=events_sink,
        curator=None,
        run_root_node_id=None,
        mcp_manager=None,
        extra_protect_paths=protect,
        mode="machine" if read_only else "run",
        # The REPO's state dir (not this state's per-state dir above): a
        # mode="run" agent state participates in cross-run memory like any
        # other run; for read-only states the dispatcher mode guard and
        # the machine/agent prompt assembly keep it inert.
        state_dir=resolved_state_dir(req.root),
    )
    rm = cfg.models.resolve("worker")
    compact_drop, compact_summarise, keep_recent = resolve_compaction_thresholds(
        cfg, rm, log=reporter.err
    )
    cfg = resolve_decompose(cfg, rm, log=reporter.err)
    wf = Workflow(
        root=req.root,
        config=cfg,
        commit_trailer=render_commit_trailer(
            cfg.git.commit.trailer, models=(rm.model if rm is not None else "",)
        ),
        max_iterations=cfg.workflow.max_iterations,
        tool_result_cap_chars=tool_result_cap_chars(cfg),
        provider=provider,
        summariser_provider=summariser_provider,
        dispatcher=dispatcher,
        logger=reporter.err,
        mode=mode if mode in ("machine", "agent") else "run",
        # A mode="run" state commits on its own chain like any run; the
        # instance dir name is its session-unique id. Read-only states never
        # commit (mode gate), so the refs stay None there.
        chain_ref=(
            machine_chain_ref_for(req.transcript_dir.parent.name) if not read_only else None
        ),
        chain_fallback_parent=_machine_head_sha(req.root) if not read_only else None,
        commit_per_step=cfg.git.commit_per_step,
        state_dir=resolved_state_dir(req.root),
        compact_drop_at_chars=compact_drop,
        compact_summarise_at_chars=compact_summarise,
        context_summary_max_tokens=cfg.context.summary_max_tokens,
        keep_recent_chars=keep_recent,
        keep_thinking_turns=cfg.context.keep_thinking_turns,
        compact_elision_gists=cfg.context.elision_gists,
        steer_requested=bridges.steer_requested if bridges is not None else (lambda: False),
        steer_clear=bridges.steer_clear if bridges is not None else (lambda: None),
        steer_prompt=bridges.steer_prompt if bridges is not None else (lambda: None),
        finish_validator=_finish_validator(r),
    )
    result = wf.run(_task_with_contract(r))
    payload = result.finish_payload if result.reason == "finish_session" else None
    return _result(result.reason, payload, budget)


def build_machine_agent_runner(
    overlay: dict[str, Any],
    cwd: Path,
    isolation: IsolationLevel,
    transcript_dir: Path,
    protect_paths: tuple[Path, ...] = (),
    commit_identity: CommitIdentity | None = None,
    machine_id: str | None = None,
    clone_root: Path | None = None,
) -> Callable[[AgentRequest, Path | None], AgentExecResult]:
    """Build the host-side runner an `agent` state uses to drive a confined loop.

    The machine engine is a host-netns supervisor; each `agent` state runs in
    its OWN subprocess (`agent6.ui.cli.machine_agent`) driving the loop
    (`run_one` above), independently of the engine and of sibling `tool`
    states. Like every agent process it runs unconfined; the jail bounds the
    commands it dispatches. The subprocess is
    spawned with a fixed argv (no LLM-derived content) and handed the request via
    a temp file; the operator-authored prompt travels in that file, never on the
    command line. `timeout_secs` is enforced by killing the subprocess's whole
    process group (true mid-call cancellation, and the per-agent session-network
    holder dies with it).

    `events_log` is per CALL: the live World passes each agent-state execution
    its own `<instance>/states/<seq>-<state>/logs.jsonl` and `machine create`
    passes the draft log, so the subprocess writes a watchable event stream there.

    `machine_id` + `clone_root` (set together by `machine run` for a machine
    with `mode="run"` states): EVERY agent state then executes in a fresh
    clone under `<clone_root>/state-<seq>`, checked out at the machine
    chain's tip (the origin's HEAD before the first landing), so a read-only
    judge sees the machine's work too. A run-mode state's commits land back
    per state: the chain ref for the next state's continuation, and the
    visible `agent6/machine-<id>` branch at the same tip for the operator --
    the run story ("changes are on a branch; merge them") with the lane
    clone mechanism; a read-only state commits nothing, so its landing is a
    no-op cleanup. The operator's checkout is never touched. Without them
    (`machine create`, a machine with no run states) requests run in *cwd*.
    """

    def run_agent(request: AgentRequest, events_log: Path | None = None) -> AgentExecResult:
        # The salvage below must see only THIS call's events: machine create
        # shares one draft log across attempts, and an attempt that died before
        # its first budget.update would otherwise salvage the PRIOR attempt's
        # cumulative totals and double-book them. (Per-state logs are fresh per
        # execution, so the offset is 0 there.)
        start_offset = 0
        if events_log is not None:
            with contextlib.suppress(OSError):
                start_offset = events_log.stat().st_size

        def salvaged(reason: str) -> AgentExecResult:
            # No result.json (killed/timed-out/crashed): recover the loop's
            # running budget.update totals from the state's own event log, else a
            # timed-out state books $0 and the budget guard never trips.
            spend = (
                read_budget_totals(events_log, from_offset=start_offset)
                if events_log is not None
                else Spend()
            )
            return AgentExecResult(
                reason=reason,
                payload=None,
                usd=spend.usd,
                usd_partial=spend.partial,
                input_tokens=spend.input_tokens,
                output_tokens=spend.output_tokens,
            )

        clone: Path | None = None
        chain = machine_chain_ref_for(machine_id) if machine_id is not None else None
        if chain is not None and clone_root is not None:
            clone = clone_root / f"state-{request.step_seq:04d}"
            try:
                clone_at_machine_chain(cwd, clone, chain)
            except (SubrunError, GitError) as exc:
                return salvaged(f"error: clone for machine {machine_id!r} failed: {exc}")
        workdir = clone or cwd
        payload = MachineAgentRequest(
            cwd=workdir,
            root=workdir,
            overlay=overlay,
            isolation=isolation,
            transcript_dir=transcript_dir,
            events_log=events_log,
            # Protect the clone's own copy of the bundle: the origin copy is
            # what executes (and what M2's recorded-bundle compare holds), but
            # the letter of "a state cannot rewrite its machine logic" covers
            # the copy it can reach too.
            protect_paths=tuple(
                workdir / p.relative_to(cwd) if clone is not None and p.is_relative_to(cwd) else p
                for p in protect_paths
            ),
            commit_identity=commit_identity,
            request=request,
        )
        with tempfile.TemporaryDirectory(prefix="agent6-machine-agent-") as td:
            req_file = Path(td) / "request.json"
            out_file = Path(td) / "result.json"
            req_file.write_text(payload.model_dump_json(), encoding="utf-8")
            argv = [
                sys.executable,
                # -P keeps cwd (the workspace) off sys.path: a model-planted
                # `agent6/` must not shadow the installed package on the host.
                "-P",
                "-m",
                "agent6.ui.cli.machine_agent",
                str(req_file),
                str(out_file),
            ]
            # Own session/process group so the timeout kill takes the agent
            # subprocess AND its jail children with it; PDEATHSIG so the
            # whole tree dies with the supervisor instead of running on,
            # spending and committing, after a SIGTERM/SIGKILL nobody waits
            # out (the own-session child had no tie to its parent's life).
            # PLW1509 (fork-with-threads hazard): the hook is written for it,
            # async-signal-minimal -- libc preloaded at import, then only
            # prctl/getppid/_exit, no allocation or locks.
            # AGENT6_SUBRUN: a machine state is subordinate work and must
            # not itself fan out (the same depth-1 flag every lane carries).
            proc = subprocess.Popen(
                argv,
                start_new_session=True,
                env={**os.environ, "AGENT6_SUBRUN": "1"},
                preexec_fn=die_with_parent(os.getpid()),  # noqa: PLW1509
            )
            try:
                proc.wait(timeout=request.timeout_s)
            except subprocess.TimeoutExpired:
                with contextlib.suppress(ProcessLookupError):
                    # By pid: start_new_session made it the group leader, and an
                    # unreaped child's pgid cannot have been recycled. Looking
                    # it up first left a window where, under sudo, an unrelated
                    # group could be killed as root.
                    os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
                result = salvaged("timeout")
            else:
                if proc.returncode != 0 or not out_file.is_file():
                    result = salvaged("error")
                else:
                    try:
                        result = AgentExecResult.model_validate_json(
                            out_file.read_text(encoding="utf-8")
                        )
                    except (OSError, ValidationError):
                        # A malformed result.json is treated like a missing
                        # one: the spend salvage keeps the budget honest.
                        result = salvaged("error")
        if clone is not None and chain is not None and machine_id is not None:
            result = _land_machine_clone(cwd, clone, chain, machine_branch_for(machine_id), result)
        return result

    return run_agent


def clone_at_machine_chain(origin: Path, dest: Path, chain_ref: str) -> None:
    """Fresh clone checked out at the machine chain's tip.

    A clone copies branches, not `refs/agent6/*`, so the chain ref is fetched
    in and the worktree detached onto its tip -- state N+1 starts from state
    N's full tree. No chain yet (first run state, or the operator archived
    it): the clone's own HEAD is the continuation-from-merged-state start."""
    if dest.exists():
        shutil.rmtree(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    clone_workspace(origin, dest)
    tip = chain_tip(origin, chain_ref)
    if tip is not None:
        fetch_branch(dest, origin, f"{chain_ref}:{chain_ref}")
        checkout_detached(dest, tip)


def _land_machine_clone(
    origin: Path, clone: Path, chain_ref: str, branch: str, result: AgentExecResult
) -> AgentExecResult:
    """Land the state's work back in the origin and drop the clone.

    Two refs, one tip: the chain ref carries the next state's continuation,
    and the visible branch is the operator's handle on the same commits. Runs
    on EVERY outcome -- a timed-out or failed state's real commits still land
    (the outcome label routes the machine; work is never stranded). Serial
    states (the instance lock) make both updates fast-forwards. An import
    failure keeps the clone (the only copy; the prune sweep proves that and
    keeps it) and routes the state as failed with no captured payload."""
    advanced = chain_tip(clone, chain_ref)
    if advanced is None or advanced == chain_tip(origin, chain_ref):
        shutil.rmtree(clone, ignore_errors=True)
        return result
    try:
        fetch_branch(origin, clone, f"{chain_ref}:{chain_ref}")
        fetch_branch(origin, clone, f"{chain_ref}:refs/heads/{branch}")
    except GitError as exc:
        return result.model_copy(
            update={"reason": f"import of machine work failed: {exc}", "payload": None}
        )
    shutil.rmtree(clone, ignore_errors=True)
    return result
