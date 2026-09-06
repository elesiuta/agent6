# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Tool dispatch: validates incoming LLM tool calls and executes them.

File reads/writes resolve through the workspace boundary: *root* (the repo
cwd) plus the operator's extra grants and the per-repo memory dir. Commands
run jailed -- the run's `JailSession`, or a per-command `run_in_jail` when no
session exists. Capability gating (`run_commands = "no" | "ask" | "yes"`) is
enforced here.
"""

from __future__ import annotations

import itertools
import json
import os
import shlex
import shutil
import threading
import time
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any, Literal

from pydantic import ValidationError

from agent6.config import Config
from agent6.events import EventSink
from agent6.graph.curator import GraphCurator
from agent6.memory import memory_dir
from agent6.paths import data_dir
from agent6.sandbox._tool_paths import jail_search_path
from agent6.sandbox.jail import JailSession, JailUnavailableError, SessionNetwork, run_in_jail
from agent6.sessions.ipc import (
    COMMAND_SCOPE,
    MCP_SCOPE_PREFIX,
    effective_run_commands,
    session_deny_set,
    steer_answer_is_abort,
    stop_request_pending,
)
from agent6.sessions.layout import session_layout
from agent6.skills import (
    ResolvedSkills,
    operator_skills,
)
from agent6.tools._control_tools import finish_planning, finish_session
from agent6.tools._dag_tools import add_task, list_tasks, update_task
from agent6.tools._fs_tools import agent6_docs, apply_edit, apply_patch, list_dir, read_file
from agent6.tools._nav_tools import (
    find_definition,
    find_references,
    outline,
)
from agent6.tools._result_format import (
    parse_metric_score,
    truncate_args,
)
from agent6.tools._skill_tools import use_skill
from agent6.tools.background import SHELLS_DIR, BackgroundError, BackgroundShells
from agent6.tools.errors import OperatorCommandUnexecutable, ToolDenied, ToolError
from agent6.tools.fetch import FetchRefused, check_url, fetch, host_allowed
from agent6.tools.index import Symbol, SymbolIndex
from agent6.tools.mcp_client import (
    MCP_TOOL_PREFIX,
    MCPError,
    MCPManager,
    MCPToolDescriptor,
    split_tool_name,
)
from agent6.tools.operator_prompts import OperatorPrompts
from agent6.tools.policy import jail_policy, resolve_network, workspace_for
from agent6.tools.results import (
    AnswersResult,
    BackgroundResult,
    ExecResult,
    FetchResult,
    MetricResult,
    RawResult,
    ReadFileResult,
    SessionsResult,
    ToolResult,
)
from agent6.tools.schema import (
    ALL_TOOLS,
    Agent6DocsInput,
    ApplyEditInput,
    ApplyPatchInput,
    AskUserInput,
    DagAddTaskInput,
    DagListTasksInput,
    DagUpdateTaskInput,
    FetchInput,
    FindDefinitionInput,
    FindReferencesInput,
    FinishPlanningInput,
    FinishSessionInput,
    ListDirInput,
    OutlineInput,
    ReadBackgroundInput,
    ReadFileInput,
    ReadSessionInput,
    RunCommandInput,
    RunMetricInput,
    RunVerifyInput,
    StopBackgroundInput,
    UseSkillInput,
    mode_tools,
)
from agent6.tools.sessions import conversation, roster
from agent6.types import (
    BackgroundHandoff,
    CommandResult,
    IsolationLevel,
    JailPolicy,
    session_kind,
)


def _coerce_stringified_args(
    raw_input: dict[str, Any], exc: ValidationError
) -> dict[str, Any] | None:
    """Recover a tool call whose structured argument arrived as a JSON string.

    Weak models occasionally serialize an array/object argument to a string
    (e.g. apply_edit `edits` arriving as `'[{...}]'`), wasting a
    round-trip on a validation error the model must repair. For each top-level field named in the
    validation error whose provided value is a str, parse the string's head
    as JSON (`raw_decode` tolerates trailing junk like a leaked closing
    tag) and substitute the parsed value when it is a container. Fields the
    schema really declares as strings are unaffected: a wrong substitution
    fails re-validation and the caller re-raises the original error. Returns
    the coerced copy of `raw_input`, or None when nothing was coercible.
    """
    decoder = json.JSONDecoder()
    coerced: dict[str, Any] | None = None
    for err in exc.errors():
        loc = err.get("loc") or ()
        key = loc[0] if loc else None
        if not isinstance(key, str):
            continue
        val = raw_input.get(key)
        if not isinstance(val, str):
            continue
        try:
            parsed, _ = decoder.raw_decode(val.strip())
        except ValueError:
            continue
        if not isinstance(parsed, dict | list):
            continue
        if coerced is None:
            coerced = dict(raw_input)
        coerced[key] = parsed
    return coerced


def invalid_arguments(exc: ValidationError) -> str:
    """One line per real argument problem, for the model and the log: the
    dotted field, then the message without pydantic's "Value error, " lead
    and docs URL. A container error caused by an invalid item ("edits: Tuple
    should have at least 1 item after validation, not 0") is dropped: the
    item's own line already says what to fix."""
    errors = exc.errors(include_url=False)
    item_locs = [tuple(e["loc"]) for e in errors]
    parts: list[str] = []
    for err in errors:
        loc = tuple(err["loc"])
        if err["type"] == "too_short" and any(
            other != loc and other[: len(loc)] == loc for other in item_locs
        ):
            continue
        field = ".".join(str(p) for p in loc) or "arguments"
        msg = str(err["msg"]).removeprefix("Value error, ")
        parts.append(f"{field}: {msg}")
    return "invalid arguments: " + "; ".join(parts)


# Execution tools whose stdout/stderr IS the diagnostic signal. Their tool.result
# event carries a capped output tail (like verify.end) so logs.jsonl shows
# the command's output for quick observability -- not just a one-line summary --
# without opening the transcripts (where the full, uncapped output always lives).
_EXEC_OUTPUT_TOOLS = frozenset({RunCommandInput.TOOL_NAME, RunMetricInput.TOOL_NAME})
_TOOL_OUTPUT_TAIL = 2000  # chars, matching verify.end's stdout_tail/stderr_tail


# An MCP approval prompt carries the FULL arguments up to this bound; past it
# the complete payload goes to a session-dir file the prompt points at.
_PAYLOAD_IDS = itertools.count(1)
_APPROVAL_PROMPT_MAX_CHARS = 4096

_READ_HEAD_LINES = 6
_READ_HEAD_CHARS = 300


def _output_tails(name: str, result: ToolResult) -> dict[str, Any]:
    """Capped output excerpts an execution/read tool's result carries into its
    tool.result event, else {}. Commands get stdout/stderr tails; read_file
    gets a head preview + the true line count, so logs.jsonl shows what was
    read without opening the transcripts."""
    if isinstance(result, ExecResult | MetricResult) and name in _EXEC_OUTPUT_TOOLS:
        return {
            "stdout_tail": result.stdout[-_TOOL_OUTPUT_TAIL:],
            "stderr_tail": result.stderr[-_TOOL_OUTPUT_TAIL:],
        }
    if isinstance(result, ReadFileResult):
        head = "\n".join(result.content.splitlines()[:_READ_HEAD_LINES])
        return {
            "head_tail": head[:_READ_HEAD_CHARS],
            "lines_total": result.lines_total,
        }
    return {}


def _clip_tail(text: str, limit: int = 20_000) -> str:
    """The last `limit` chars, prefixed by a marker naming what was dropped
    (the read_background rendering does the same per line); an unmarked clip
    reads as the complete output."""
    if len(text) <= limit:
        return text
    return f"... {len(text) - limit} earlier chars clipped ...\n" + text[-limit:]


# Every tool that runs a command in the jail. They all execute model-influenced
# argv with the same reach, so one knob governs them: `run_commands = "no"`
# hides them, "ask" prompts (the session-allow marker keeps that to one prompt
# per run), "yes" runs. run_verify_command is here too -- its argv is the
# operator's when configured, but INFERRED from a file the model can edit when
# it is not, and either way it is a command in the same sandbox. So is
# run_metric_command: the operator's argv, run in that same jail, as often as
# the model asks.
_COMMAND_TOOLS = frozenset(
    {
        RunCommandInput.TOOL_NAME,
        RunVerifyInput.TOOL_NAME,
        RunMetricInput.TOOL_NAME,
        StopBackgroundInput.TOOL_NAME,
    }
)

# Bench / A-B arm for the symbol-tool surface, keyed by AGENT6_SYMBOL_TOOLS:
# "none" hides the three symbol tools (the rg-via-run_command floor). Unset
# or unknown hides nothing (the full surface); the bench harness validates
# the arm name, so a stray value cannot silently select an arm.
_SYMBOL_TOOL_ARMS: dict[str, frozenset[str]] = {
    "none": frozenset(
        {
            OutlineInput.TOOL_NAME,
            FindDefinitionInput.TOOL_NAME,
            FindReferencesInput.TOOL_NAME,
        }
    ),
}


def symbol_tools_hidden() -> frozenset[str]:
    """The symbol tools the AGENT6_SYMBOL_TOOLS bench switch hides right now."""
    return _SYMBOL_TOOL_ARMS.get(os.environ.get("AGENT6_SYMBOL_TOOLS", ""), frozenset())


def _roster(shells: BackgroundShells) -> tuple[str, ...]:
    return tuple(v.line() for v in shells.roster())


class ToolDispatcher:
    """Runtime tool dispatcher. Constructed once per workflow run."""

    def __init__(
        self,
        *,
        root: Path,
        config: Config,
        isolation: IsolationLevel = "strict",
        prompts: OperatorPrompts | None = None,
        events: EventSink | None = None,
        curator: GraphCurator | None = None,
        run_root_node_id: str | None = None,
        mcp_manager: MCPManager | None = None,
        extra_protect_paths: tuple[Path, ...] = (),
        worktree_git_dir: Path | None = None,
        mode: Literal["run", "plan", "ask", "machine"] = "run",
        state_dir: Path | None = None,
        session_dir: Path | None = None,
        use_jail_session: bool = False,
        session_net: SessionNetwork | None = None,
    ) -> None:
        self._root = root.resolve()
        self._config = config
        # The in-process file boundary. Every path-taking read/write tool
        # resolves through it, so the hidden set holds at every isolation level.
        # The per-repo memory dir rides along when state is wired: memory files
        # are read and edited with the ordinary tools (in-process only; the
        # jail never mounts it).
        mem = memory_dir(state_dir) if state_dir is not None else None
        if mem is not None:
            # The grant's target must exist: the model cannot mkdir outside
            # the jail, so a fresh repo's FIRST organic memory write would
            # fail with ENOENT.
            mem.mkdir(parents=True, exist_ok=True)
        self._ws = workspace_for(config, self._root, memory_dir=mem)
        # Public: the prompt builder reads it so the system prompt describes
        # THIS dispatcher's command behaviour (hardened-only caveats).
        self.isolation: IsolationLevel = isolation
        # In plan mode the LLM's tool list already omits apply_edit/apply_patch;
        # this is the defense-in-depth backstop so the dispatcher itself refuses
        # a source mutation even if something dispatched one directly.
        self._mode: Literal["run", "plan", "ask", "machine"] = mode
        # next() is atomic under the GIL; seats on the shared dispatcher get
        # distinct ids without a lock.
        self._call_seq = itertools.count(1)
        # Extra read-only paths layered into every run_command jail on top of
        # the strict-isolation protect_git bind (e.g. a running machine's own
        # .asm.toml + scripts bundle, so an agent state can't rewrite them
        # mid-run).
        # Also read by the prompt assembly: under hardened, Landlock carves the
        # workspace around these and denies new top-level entries.
        self.extra_protect_paths = extra_protect_paths
        # The repository git dir agent6 recorded for a fork's linked worktree
        # (jail_policy grants it once the worktree's pointer still resolves
        # to it); None for every other checkout.
        self._worktree_git_dir = worktree_git_dir
        self._events = events
        # The gate every approval and ask_user goes through: it journals the
        # prompt/answer pair and names the call it gates. A bare dispatcher
        # (a one-off tool, tests) gets one over its own journal and the stdin
        # fallbacks.
        self._prompts = prompts or OperatorPrompts(journal=self._emit, session_dir=session_dir)
        # The call being dispatched on this thread, stamped on the prompts its
        # handler raises. Per thread: concurrent review seats share one
        # dispatcher, and a shared attribute would stamp seat A's prompt with
        # seat B's call.
        self._gating = threading.local()
        # Seconds spent blocked on the operator (approvals, ask_user). The loop
        # subtracts them from its wall clock: waiting for a human is not the
        # model stalling.
        self.operator_wait_s = 0.0
        # Optional in-process GraphCurator + root-task id for the DAG-as-tool
        # surface. When wired, the dispatcher exposes add_task /
        # update_task / list_tasks.
        self._curator = curator
        # Read by the tool list: the three DAG tools answer "no curator" for
        # a run built without one (a machine agent state).
        self.dag_available = curator is not None
        self._run_root_node_id = run_root_node_id
        # Optional MCP (Model Context Protocol) manager. When
        # set, `dispatch` routes any tool name starting with the MCP
        # prefix to the manager. Discovered tool names are also added
        # to `available_tool_names()` so the workflow exposes them.
        self._mcp_manager = mcp_manager
        # Per-repo state dir: sessions roster reads plus the memory grant
        # above. None (tests, review/one-off dispatchers) leaves both off.
        self._state_dir = state_dir
        # Background commands live under the run dir so they die with the run
        # and `sessions rm` clears them. None (tests, review dispatchers) leaves
        # them unwired: the tools raise ToolError, like the DAG tools.
        self._shells = (
            BackgroundShells(session_dir / SHELLS_DIR) if session_dir is not None else None
        )
        # One jail process for the whole run, opened on the first jailed
        # command and closed at teardown, so a run's commands share a netns, a
        # PID namespace and a /tmp and pay the setup once. RUN-SCOPED: a bare
        # dispatcher (a one-off tool, an embedder) has no run to scope it to
        # and keeps the per-command launcher.
        self._use_session = use_jail_session
        self._session_net = session_net
        self._own_session_net: SessionNetwork | None = None
        self._session: JailSession | None = None
        self._session_failed = False
        # Two threads racing the lazy open would each start a launcher and one
        # would be dropped on the floor -- a leaked jail process, its
        # namespaces, and (under `session`) its network holder, with nothing
        # left holding a handle to close them.
        self._session_lock = threading.Lock()
        # The run's dir, for the effective command policy: the operator's
        # session choice and away-mode live there and can change mid-run.
        self._session_dir = session_dir
        self._handlers: dict[str, Callable[[dict[str, Any]], ToolResult]] = {
            Agent6DocsInput.TOOL_NAME: self._agent6_docs,
            ReadFileInput.TOOL_NAME: self._read_file,
            ListDirInput.TOOL_NAME: self._list_dir,
            OutlineInput.TOOL_NAME: self._outline,
            FindDefinitionInput.TOOL_NAME: self._find_definition,
            FindReferencesInput.TOOL_NAME: self._find_references,
            ApplyEditInput.TOOL_NAME: self._apply_edit,
            ApplyPatchInput.TOOL_NAME: self._apply_patch,
            RunVerifyInput.TOOL_NAME: lambda _raw: self.run_verify(),
            RunCommandInput.TOOL_NAME: self._run_command,
            ReadSessionInput.TOOL_NAME: self._read_session,
            FetchInput.TOOL_NAME: self._fetch,
            ReadBackgroundInput.TOOL_NAME: self._read_background,
            StopBackgroundInput.TOOL_NAME: self._stop_background,
            # run_metric: LLM-exposed via LOOP_EXTRA_TOOLS so the
            # loop can call it after a successful verify when
            # [workflow.metric] is configured.
            RunMetricInput.TOOL_NAME: self._run_metric,
            # finish_session signals the loop should exit. Handler
            # just echoes the summary; the workflow checks for this tool name
            # in resp.tool_uses and terminates after dispatching it.
            FinishSessionInput.TOOL_NAME: self._finish_session,
            FinishPlanningInput.TOOL_NAME: self._finish_planning,
            AskUserInput.TOOL_NAME: self._ask_user,
            # DAG-as-tool. Handlers raise ToolError if no curator was
            # wired (so standalone tests can omit it).
            DagAddTaskInput.TOOL_NAME: self._dag_add_task,
            DagUpdateTaskInput.TOOL_NAME: self._dag_update_task,
            DagListTasksInput.TOOL_NAME: self._dag_list_tasks,
            # Operator-installed skills; resolved lazily from config + the
            # data dir on first use (see _resolved_skills).
            UseSkillInput.TOOL_NAME: self._use_skill,
        }
        self._available = {cls.TOOL_NAME for cls in ALL_TOOLS}
        self._index: SymbolIndex | None = None
        # Guards the lazy build of self._index so concurrent explore-review
        # seats (sharing one dispatcher across ThreadPoolExecutor threads)
        # can't double-build it.
        self._index_lock = threading.Lock()
        # Operator-installed skills, resolved once on first use (a disk scan
        # of the configured skill dirs). None = not yet resolved.
        self._skills_cache: ResolvedSkills | None = None

    @property
    def root(self) -> Path:
        return self._root

    def set_run_root_node_id(self, node_id: str | None) -> None:
        """Workflow sets this after seeding the run's root task.
        `add_task` with parent_id=None falls back to this as the parent."""
        self._run_root_node_id = node_id

    def command_policy(self) -> str:
        """ "no" | "ask" | "yes" for this run, right now.

        Re-read rather than cached: an operator who denies for the session
        mid-run withdraws the tools from the next turn, and one who allows for
        the session stops being prompted from the next call.
        """
        configured = self._config.sandbox.run_commands
        if self._session_dir is None:
            return configured
        return effective_run_commands(configured, self._session_dir)

    def metric_configured(self) -> bool:
        """Whether `[workflow.metric]` gives `run_metric_command` anything to
        run. The loop exposes that tool as an extra, outside
        `available_tool_names`, so it asks this instead."""
        return self._config.workflow.metric is not None

    def _commands_have_the_network(self) -> bool:
        """Whether a jailed command reaches the network on its own, which is
        what `fetch` exists to stand in for."""
        return resolve_network(self._config, self.isolation) == "host"

    def tool_is_withheld(self, name: str) -> bool:
        """Whether the model is denied *name*, extras included. The tool list is
        built from the mode's surface, which carries tools that are not in
        ALL_TOOLS (`run_metric_command`), so `available_tool_names` cannot
        answer for them and one was offered under `run_commands = "no"` with
        nothing but a refusal behind it."""
        return name in _COMMAND_TOOLS and self.command_policy() == "no"

    def available_tool_names(self) -> tuple[str, ...]:
        names = [n for n in self._available if not self.tool_is_withheld(n)]
        # No verify_command (and none inferred) -> a gateless run: hide
        # run_verify_command rather than offer a tool that would error.
        if not self._config.workflow.verify_command:
            names = [n for n in names if n != RunVerifyInput.TOOL_NAME]
        # `fetch` exists because a jailed command has no network. Where one
        # DOES, the worker can already run curl, and two ways to do one thing
        # is the thing we do not do. The RESOLVED network answers that: only
        # strict has namespaces, so hardened and none put every command on the
        # host network whatever the config says.
        if self._commands_have_the_network():
            names = [n for n in names if n != FetchInput.TOOL_NAME]
        # Bench / A-B harness: constrain the symbol-tool surface without a
        # rebuild (see _SYMBOL_TOOL_ARMS).
        hidden_symbols = symbol_tools_hidden()
        if hidden_symbols:
            names = [n for n in names if n not in hidden_symbols]
        # Bench probe for the "tool-surface fit"
        # hypothesis. Hide `apply_edit` so the only edit primitive is
        # `apply_patch` (unified-diff). Lets us measure whether models
        # that look weak on agent6's diff-style search-and-replace
        # surface improve when handed a patch tool instead. No-op when
        # unset (default keeps both tools available).
        if os.environ.get("AGENT6_DISABLE_APPLY_EDIT") == "1":
            names = [n for n in names if n != ApplyEditInput.TOOL_NAME]
        names.extend(d.qualified_name for d in self.mcp_descriptors())
        return tuple(sorted(names))

    def mcp_denied(self, server: str) -> bool:
        """Whether the operator answered "deny all" for *server* this session.

        Read at both gates, like `command_policy`: the tool list drops the
        server so the model stops spending turns on a door that will not open,
        and the call gate refuses it because withdrawal is not refusal -- the
        model still has the previous turn's list in context.
        """
        if self._session_dir is None:
            return False
        return session_deny_set(self._session_dir, f"{MCP_SCOPE_PREFIX}{server}")

    def mcp_descriptors(self) -> tuple[MCPToolDescriptor, ...]:
        """The MCP tools on offer right now, minus any server the operator denied
        for the session."""
        if self._mcp_manager is None:
            return ()
        return tuple(
            d for d in self._mcp_manager.descriptors() if not self.mcp_denied(d.server_name)
        )

    def dispatch(self, name: str, raw_input: dict[str, Any]) -> ToolResult:
        # Returns the typed result; the caller serializes it with to_wire() at
        # the single wire boundary (the loop / review seat / mcp server).
        # Emit `tool.call` UP FRONT, before any guard, so EVERY dispatched tool
        # -- including ones a guard rejects (unknown name, disabled, wrong mode)
        # -- produces a matching `tool.result(ok=...)` pair. Otherwise a reader
        # sees a `loop.tool.call` with no result and has to guess what happened.
        # The emit + the ok flag live here in the dispatcher (not gated on the
        # model), so a prompt injection cannot suppress the event or fake
        # success; rejection reasons come from these hardcoded guards, not from
        # model-supplied content.
        # The finish tools' `summary` is the human end-of-run statement (shown on
        # the done line + in `watch`); keep it whole. Generic args stay clipped.
        max_chars = 2000 if name in ("finish_session", "finish_planning") else 200
        preview = truncate_args(raw_input, max_value_chars=max_chars)
        # Correlation id shared by this dispatch's call/result pair: concurrent
        # review seats interleave events through the one shared sink, and
        # name-based pairing cross-stamps same-name calls.
        cid = next(self._call_seq)
        self._emit("tool.call", name=name, args=preview, call_id=cid)
        outer = self._gating_call_id()
        self._gating.call_id = cid
        try:
            result = self._dispatch_inner(name, raw_input)
        except ToolError as exc:
            self._emit("tool.result", name=name, ok=False, summary=str(exc), call_id=cid)
            raise
        except OperatorCommandUnexecutable as exc:
            # Not a model-fixable tool error: an operator verify/metric command
            # that cannot execute in the jail. Record the failed result for the
            # audit trail, then propagate (NOT wrapped as ToolError) so the loop
            # aborts the run loudly instead of surfacing it as a normal failure.
            self._emit("tool.result", name=name, ok=False, summary=str(exc), call_id=cid)
            raise
        except ValidationError as exc:
            message = invalid_arguments(exc)
            self._emit("tool.result", name=name, ok=False, summary=message, call_id=cid)
            raise ToolError(message) from exc
        except Exception as exc:
            self._emit("tool.result", name=name, ok=False, summary=str(exc), call_id=cid)
            raise ToolError(f"failed: {exc}") from exc
        finally:
            self._gating.call_id = outer
        self._emit(
            "tool.result",
            name=name,
            ok=True,
            summary=result.summary(),
            call_id=cid,
            **_output_tails(name, result),
        )
        return result

    def _dispatch_inner(self, name: str, raw_input: dict[str, Any]) -> ToolResult:
        """Resolve + execute a tool. Raises ToolError on a rejected/failed call;
        the caller (`dispatch`) owns the tool.call/tool.result events."""
        # MCP routing happens BEFORE the built-in handler check so mcp__* names
        # don't collide with the built-in "Unknown tool" error path.
        if name.startswith(MCP_TOOL_PREFIX):
            if not session_kind(self._mode).edits:
                # MCP tools are arbitrary external capabilities agent6 cannot
                # classify as read-only, so every non-run mode refuses them --
                # the same dispatcher backstop the built-in mutating tools get,
                # covering the withheld edit/DAG tools of plan/ask and
                # the machine-authoring "do not edit or run anything" contract.
                raise ToolError(f"not available in {self._mode} mode (run mode only)")
            if self._mcp_manager is None:
                raise ToolError("MCP is not configured")
            self._approve_mcp_call(name, raw_input)
            try:
                return RawResult(self._mcp_manager.call(name, raw_input))
            except MCPError as exc:
                raise ToolError(str(exc)) from exc
        if name not in self._handlers:
            raise ToolError(f"Unknown tool: {name}")
        if name in _COMMAND_TOOLS and self.command_policy() == "no":
            raise ToolError("not available (run_commands = 'no')")
        if name == FetchInput.TOOL_NAME and self._commands_have_the_network():
            raise ToolError("not available (a jailed command has the network)")
        if name in symbol_tools_hidden():
            raise ToolError("not available (AGENT6_SYMBOL_TOOLS)")
        if os.environ.get("AGENT6_DISABLE_APPLY_EDIT") == "1" and name == ApplyEditInput.TOOL_NAME:
            raise ToolError(
                f"{name} is disabled (AGENT6_DISABLE_APPLY_EDIT=1); use apply_patch instead"
            )
        if name not in mode_tools(self._mode).permitted:
            # Backstop the mode's tool surface at the dispatcher, not just by
            # omitting tools from the LLM's list: a tool-list regression or a
            # hallucinated name must not mutate the repo or run commands
            # (including the approval-gate-free metric command) from a
            # read-only mode, or pause a non-run loop (ask_user). Enforcing
            # membership in the same surface `tool_definitions` exposes means
            # the two cannot drift.
            raise ToolError(f"not available in {self._mode} mode")
        return self._run_handler(name, raw_input)

    def _run_handler(self, name: str, raw_input: dict[str, Any]) -> ToolResult:
        """Execute the handler, retrying once with stringified-JSON args coerced."""
        # The provider couldn't parse the tool-call arguments as JSON and left the
        # `_raw_arguments` sentinel (after a lenient re-parse already failed). A
        # schema error about "_raw_arguments extra fields" would misdirect the
        # model; tell it plainly the JSON was malformed so it resends in one shot.
        if set(raw_input) == {"_raw_arguments"}:
            raw = raw_input.get("_raw_arguments")
            raw_len = len(raw) if isinstance(raw, str) else 0
            if raw_len > 20_000:
                # Not a formatting slip: the arguments ran away (observed: a
                # model emitting a 117KB pattern of one alternation repeated
                # until the output-token ceiling cut the JSON string mid-way).
                # "Resend" feedback makes such a model regenerate the same
                # runaway; name the actual problem instead.
                raise ToolError(
                    "the arguments were cut off mid-generation"
                    f" ({raw_len // 1000} KB, truncated before the JSON closed)."
                    " Do NOT resend the same call. Emit a much smaller call:"
                    " short literal values only (keep any pattern or argument"
                    " under a couple hundred characters), and split broad work"
                    " into several small calls."
                )
            raise ToolError(
                "the arguments were not valid JSON. Resend the call with a"
                " single valid JSON object of arguments."
            )
        try:
            return self._handlers[name](raw_input)
        except ValidationError as exc:
            coerced = _coerce_stringified_args(raw_input, exc)
            if coerced is None:
                raise
            try:
                return self._handlers[name](coerced)
            except ValidationError:
                # The coercion guessed wrong; the original shape error is the
                # honest one to surface.
                raise exc from None

    def _emit(self, event_type: str, /, **fields: Any) -> None:
        if self._events is not None:
            self._events.emit(event_type, **fields)

    def _gating_call_id(self) -> int | None:
        """The call this thread is dispatching; None outside a dispatch (a
        verify the harness runs itself)."""
        return getattr(self._gating, "call_id", None)

    # ----- handlers -----

    def _agent6_docs(self, raw: dict[str, Any]) -> ToolResult:
        return agent6_docs(raw)

    def _read_file(self, raw: dict[str, Any]) -> ToolResult:
        return read_file(self._ws, raw)

    def _list_dir(self, raw: dict[str, Any]) -> ToolResult:
        return list_dir(self._ws, raw)

    def _apply_edit(self, raw: dict[str, Any]) -> ToolResult:
        return apply_edit(self._ws, self._config, self.extra_protect_paths, self._index, raw)

    def _apply_patch(self, raw: dict[str, Any]) -> ToolResult:
        return apply_patch(self._ws, self._config, self.extra_protect_paths, self._index, raw)

    # ----- tree-sitter index handlers -----

    def _ensure_index(self) -> SymbolIndex:
        if self._index is None:
            with self._index_lock:
                if self._index is None:
                    self._index = SymbolIndex(self._ws)
        return self._index

    def hot_symbols(
        self,
        *,
        max_symbols: int = 20,
        min_files_referenced: int = 2,
    ) -> list[tuple[str, str, str, int, int]]:
        """Public passthrough to `SymbolIndex.hot_symbols`, sharing the
        dispatcher's index so an already-paid scan is not repeated."""
        idx = self._ensure_index()
        return idx.hot_symbols(
            max_symbols=max_symbols,
            min_files_referenced=min_files_referenced,
        )

    def file_outlines(self) -> dict[Path, list[Symbol]]:
        """Public passthrough to `SymbolIndex.file_outlines`.

        Used by `Workflow._load_repo_summary` to build the
        per-file symbol outline injected into the system prompt.
        """
        idx = self._ensure_index()
        return idx.file_outlines()

    def _outline(self, raw: dict[str, Any]) -> ToolResult:
        return outline(self._ws, self._ensure_index, raw)

    def _find_definition(self, raw: dict[str, Any]) -> ToolResult:
        return find_definition(self._ws, self._ensure_index, raw)

    def _find_references(self, raw: dict[str, Any]) -> ToolResult:
        return find_references(self._ws, self._ensure_index, raw)

    def settle_background(self) -> None:
        """Write down the ending of any background command that has finished.

        For the loop's turn boundary: only an observed exit reaches disk, and
        the model need never ask again after starting one.
        """
        if self._shells is not None:
            self._shells.settle()

    def close(self) -> None:
        """Release subprocess resources.

        Idempotent. Safe to call from CLI teardown alongside
        `mcp_manager.close()`.
        """
        if self._shells is not None:
            self._shells.stop_all()
        self.close_jail_session()
        if self._own_session_net is not None:  # never the run's; that is its own to close
            self._own_session_net.close()
            self._own_session_net = None

    def adopt_verify_command(self, argv: tuple[str, ...]) -> bool:
        """Adopt a verify command mid-run: the loop's gateless adoption after
        the tree materializes (see Workflow._maybe_adopt_verify). Same trust
        as preflight's in-memory injection: derived from the repo's own
        AGENTS.md fence or project signals, operator-origin, never persisted.

        False (nothing adopted) when a bare argv[0] does not resolve on the
        jail PATH: adopting a gate the sandbox cannot execute would turn a
        would-be honest settle into an unexecutable-verify abort. Path-form
        commands are accepted as-is (they resolve against the mounted cwd)."""
        if self.command_policy() == "no":
            # Every command tool is withheld, the gate included. Adopting one
            # would gate the run on something it can never run.
            return False
        exe = argv[0]
        if "/" not in exe and shutil.which(exe, path=jail_search_path()) is None:
            return False
        self._config = self._config.with_verify_command(argv)
        return True

    def drop_verify_command(self) -> None:
        """Un-adopt: the gate proved unrunnable, the run is gateless again."""
        self._config = self._config.with_verify_command(())

    def _approve_mcp_call(self, name: str, raw_input: dict[str, Any]) -> None:
        """Gate one MCP tool call on its server's `approve`, or raise ToolDenied.

        A server's tools are arbitrary external capabilities, so they are asked
        about like a command -- but on their OWN scope: "allow all" for one
        server grants that server, never the command tools and never a sibling
        server. `approve = "yes"` (or `--auto-approve`) is the standing consent.
        The ARGUMENTS are in the prompt because they are the whole risk: the
        server's actions are fixed, what the model chose to send is not. They go
        in WHOLE, never the log preview: a clipped arg is consent to an
        operation the operator never saw.
        """
        server, _tool = split_tool_name(name)
        entry = self._config.mcp.servers.get(server)
        if entry is None:
            # The scope becomes a filename, and the LLM chooses tool names: a
            # call to `mcp__../../tmp/x__t` would otherwise prompt about, and
            # then record a grant for, a "server" that is a path. Only a
            # configured name is a server, and the manager would refuse this
            # call anyway.
            raise ToolError(f"unknown MCP server in {name!r}")
        if self.mcp_denied(server):
            raise ToolError(f"not available ({server!r} was denied for this session)")
        if entry.approve == "yes":
            return
        args = json.dumps(raw_input, ensure_ascii=False, sort_keys=True)
        if len(args) > _APPROVAL_PROMPT_MAX_CHARS and self._session_dir is not None:
            # Full args are the consent rule -- but a wall of text is as
            # unread as a clipped one. Past the bound, the COMPLETE payload
            # goes to a file (the session dir a jailed command cannot reach)
            # and the prompt carries the head plus where to read the rest.
            # With no session dir the full text stays in the prompt: hiding
            # any of it with nowhere to point is the worse trade.
            # One file per prompt: concurrent review seats share a dispatcher,
            # and one shared name had the operator reading call B's payload
            # while approving call A.
            full = self._session_dir / f"approval_payload-{next(_PAYLOAD_IDS)}.json"
            full.write_text(args, encoding="utf-8")
            args = (
                f"{args[:_APPROVAL_PROMPT_MAX_CHARS]}"
                f" ...[{len(args)} chars total; full payload: {full}]"
            )
        if not self._approve(f"Allow {name}: {args}", scope=f"{MCP_SCOPE_PREFIX}{server}"):
            raise ToolDenied(
                f"{name} not approved (set [mcp.servers.{server}].approve = 'yes' to stop asking)"
            )

    def _approve(self, prompt: str, *, scope: str | None = None) -> bool:
        started = time.monotonic()
        try:
            return self._prompts.approve(prompt, scope=scope, call_id=self._gating_call_id())
        finally:
            self.operator_wait_s += time.monotonic() - started

    def _not_approved(self, name: str) -> ToolDenied:
        """The gate cannot tell a human "no" from an unattended run's auto-deny,
        so the message blames neither and names the knob; a stop requested
        while the approval waited is the one cause it can name."""
        if self._session_dir is not None and stop_request_pending(self._session_dir):
            return ToolDenied(f"{name} not run: the run was asked to stop while awaiting approval")
        return ToolDenied(f"{name} not approved (sandbox.run_commands='ask')")

    def run_verify(self, extra_argv: tuple[str, ...] = ()) -> ExecResult:
        """Run the gate: the model's `run_verify_command` and the harness's
        `verify_when` certification share this one path, approvals included.
        *extra_argv* appends to the configured command (the harness's scoped
        fallback passes the selected test paths); the result's `command`
        carries the argv that actually ran."""
        argv = tuple(self._config.workflow.verify_command) + extra_argv
        if self.command_policy() == "ask" and not self._approve(
            f"Allow run_verify_command: {shlex.join(argv)}", scope=COMMAND_SCOPE
        ):
            raise self._not_approved("run_verify_command")
        # per-call timeout from config. Defaults to the jail's
        # general 600s but bench configs crank it down so infinite-loop
        # edits fail fast instead of burning ~10 min of wall per attempt.
        timeout_s = self._config.workflow.verify_timeout_s
        self._emit("verify.start", cmd=list(argv), timeout_s=timeout_s)
        res = self._run_argv_in_jail(argv, label="verify_command", timeout_s=timeout_s)
        # Name the gate in the result: it is the operator's command, or one
        # inferred from the repo, so the worker cannot otherwise tell WHICH
        # thing judged it -- or that it is judging the wrong thing (stale_gate).
        res = replace(res, command=argv)
        self._emit(
            "verify.end",
            cmd=list(argv),
            exit_code=res.returncode,
            duration_s=res.duration_s,
            timeout_s=timeout_s,
            stdout_tail=res.stdout[-2000:],
            stderr_tail=res.stderr[-2000:],
        )
        if res.exec_failed:
            raise OperatorCommandUnexecutable(
                f"verify_command {list(argv)} could not be executed in the sandbox: "
                f"{res.stderr}. The jail PATH is /usr/bin:/bin plus the standard bin "
                "dirs that exist (/usr/local/bin, /usr/local/sbin, ~/.local/bin, "
                "~/.cargo/bin, /opt/homebrew/bin, /snap/bin), each mounted read-only; "
                "the command is on "
                "none of them. Install the tool into one of those on the host, use a "
                "path inside the workspace (e.g. .venv/bin/pytest), or grant its real "
                "directory via sandbox.extra_read_paths."
            )
        return res

    def _run_command(self, raw: dict[str, Any]) -> ExecResult:
        args = RunCommandInput.model_validate(raw)
        if self.command_policy() == "ask":
            # A shell-style command line, not a Python tuple repr: the operator
            # is approving a command, so show it the way they would type it.
            ok = self._approve(f"Allow run_command: {shlex.join(args.argv)}", scope=COMMAND_SCOPE)
            if not ok:
                raise self._not_approved("run_command")
        if args.background:
            return self._start_detached(args.argv)
        return self._run_model_command(args.argv)

    def _start_detached(self, argv: tuple[str, ...]) -> ExecResult:
        """`background: true` -- the same hand-back, at a check-in of zero.

        Only a session that EDITS owns a background command's lifetime: every
        other mode is a short read-only pass, and a command killed at its end
        would have been started for nothing. Derived from the same tool set that
        withholds read_background there, so the two cannot disagree.
        """
        if ReadBackgroundInput.TOOL_NAME not in mode_tools(self._mode).permitted:
            raise ToolError(
                f"background commands are not available in {self._mode} mode:"
                " nothing there could read or stop one before the run ends"
            )
        shells = self._background()
        try:
            view = shells.start(
                argv,
                lambda a, rw: self._jail_policy(a, extra_rw_paths=rw),
                session=self._run_session(),
            )
        except BackgroundError as exc:
            raise ToolError(str(exc)) from exc
        return ExecResult(
            returncode=None,
            stdout="",
            stderr="",
            duration_s=0.0,
            exec_failed=False,
            background_id=view.id,
        )

    def _run_model_command(self, argv: tuple[str, ...]) -> ExecResult:
        """A command the MODEL chose: where the mode can read a hand-back, no
        wall-clock kill and a hand-back instead of a guess about whether a
        long one is stuck; elsewhere the bounded run with its timeout.

        The check-in needs a session (something must stay alive to own the
        running command), a background roster to hand it to, and a mode whose
        tools can read and stop the hand-back (`_start_detached`'s rule: plan
        and ask withhold both); without any of these this is an ordinary
        bounded run.
        """
        session = self._run_session()
        shells = self._shells
        checkin = self._config.workflow.command_checkin_s
        if (
            session is None
            or shells is None
            or checkin <= 0
            or ReadBackgroundInput.TOOL_NAME not in mode_tools(self._mode).permitted
        ):
            return self._run_argv_in_jail(argv, label="run_command")
        try:
            policy = self._jail_policy(argv)
            outcome = session.run(
                argv,
                env=policy.env,
                timeout_s=0.0,  # the check-in replaces the kill
                checkin_s=checkin,
                log_dir=str(shells.log_root),
                # A Stop mid-command asks for the hand-back NOW. The operator's
                # gates (verify, metric) go through `_run_argv_in_jail`, which
                # has no check-in to jump to and its own timeout_s.
                interrupted=self._operator_wants_out,
            )
        except JailUnavailableError as exc:
            raise ToolError(f"jail unavailable: {exc}") from exc
        if isinstance(outcome, CommandResult):
            return ExecResult(
                returncode=outcome.returncode,
                stdout=_clip_tail(outcome.stdout),
                stderr=_clip_tail(outcome.stderr),
                duration_s=outcome.duration_s,
                exec_failed=outcome.exec_failed,
            )
        view = shells.adopt(argv, outcome.pid, Path(outcome.log), session=session)
        self._emit("command.backgrounded", id=view.id, pid=outcome.pid, seconds=outcome.duration_s)
        return ExecResult(
            returncode=None,
            stdout=_clip_tail(outcome.stdout),
            stderr=_clip_tail(outcome.stderr),
            duration_s=outcome.duration_s,
            exec_failed=False,
            background_id=view.id,
        )

    def _operator_wants_out(self) -> bool:
        """Whether the operator has asked this run to stop or abort.

        Only consulted to cut a WAIT short. The run still ends at its own
        boundary; this just stops a tool call from sitting on the request for
        up to the whole check-in interval.
        """
        if self._session_dir is None:
            return False
        return stop_request_pending(self._session_dir) or steer_answer_is_abort(self._session_dir)

    def _background(self) -> BackgroundShells:
        if self._shells is None:
            raise ToolError("background commands need a run directory; none was wired")
        return self._shells

    def _fetch(self, raw: dict[str, Any]) -> FetchResult:
        args = FetchInput.model_validate(raw)
        try:
            checked = check_url(args.url)
        except FetchRefused as exc:
            raise ToolError(str(exc)) from exc
        # On the list: read it. Off the list: ask. The list IS the standing
        # approval, and a prompt per doc read only trains a reflexive yes --
        # but a GET can carry data out in its path, so a host the operator
        # never named is their call, and an absent one is a no (the away-mode
        # approver refuses without waiting). Nothing has resolved yet: the DNS
        # query itself carries the hostname out, so `fetch` runs it behind
        # this gate.
        if not host_allowed(checked.host, self._config.sandbox.fetch_hosts) and not self._approve(
            f"Allow fetch: {checked.prompt()}"
        ):
            raise ToolDenied(
                f"fetch not approved for {checked.host} (add it to sandbox.fetch_hosts to allow it)"
            )
        try:
            got = fetch(checked)
        except FetchRefused as exc:
            raise ToolError(str(exc)) from exc
        return FetchResult(
            url=got.url,
            status=got.status,
            content_type=got.content_type,
            body=got.body,
            location=got.location,
        )

    def _read_session(self, raw: dict[str, Any]) -> SessionsResult:
        args = ReadSessionInput.model_validate(raw)
        if self._state_dir is None:
            raise ToolError("read_session needs the project state dir; none was wired")
        lines = roster(self._state_dir, args.query).lines()
        if not args.id:
            return SessionsResult(sessions=lines)
        layout = session_layout(self._state_dir, args.id)
        if layout is None:
            raise ToolError(f"no session {args.id!r} in this project")
        return SessionsResult(
            sessions=lines, conversation=conversation(layout, max_chars=args.max_chars)
        )

    def _read_background(self, raw: dict[str, Any]) -> BackgroundResult:
        args = ReadBackgroundInput.model_validate(raw)
        shells = self._background()
        if not args.id:
            return BackgroundResult(shells=_roster(shells))
        wait_s = self._config.workflow.command_checkin_s if args.wait_s is None else args.wait_s
        try:
            _view, output = shells.read(
                args.id,
                tail_lines=args.tail_lines,
                wait_s=wait_s,
                interrupted=self._operator_wants_out,
            )
        except BackgroundError as exc:
            raise ToolError(str(exc)) from exc
        return BackgroundResult(shells=_roster(shells), output=output)

    def _stop_background(self, raw: dict[str, Any]) -> BackgroundResult:
        args = StopBackgroundInput.model_validate(raw)
        shells = self._background()
        try:
            shells.stop(args.id)
        except BackgroundError as exc:
            raise ToolError(str(exc)) from exc
        return BackgroundResult(shells=_roster(shells))

    def _ask_user(self, raw: dict[str, Any]) -> ToolResult:
        args = AskUserInput.model_validate(raw)
        started = time.monotonic()
        try:
            answers = self._prompts.ask(args.questions, call_id=self._gating_call_id())
        finally:
            self.operator_wait_s += time.monotonic() - started
        return AnswersResult(answers=answers)

    def _finish_session(self, raw: dict[str, Any]) -> ToolResult:
        return finish_session(raw)

    def _finish_planning(self, raw: dict[str, Any]) -> ToolResult:
        return finish_planning(raw)

    # DAG-as-tool handlers.

    def _dag_add_task(self, raw: dict[str, Any]) -> ToolResult:
        return add_task(self._curator, self._run_root_node_id, raw)

    def _dag_update_task(self, raw: dict[str, Any]) -> ToolResult:
        return update_task(self._curator, raw)

    def _dag_list_tasks(self, raw: dict[str, Any]) -> ToolResult:
        return list_tasks(self._curator, raw)

    def resolved_skills(self) -> ResolvedSkills:
        """Discover + state-resolve operator skills, once per dispatcher.

        Same source of truth as the loop's system-prompt index:
        `[skills].extra_dirs` first, then the installed dir under the user
        data dir. An off switch resolves to nothing.
        """
        if self._skills_cache is None:
            self._skills_cache = operator_skills(
                self._config.skills.enabled,
                self._config.skills.extra_dirs,
                self._config.skills.state,
                data_dir() / "skills",
            )
        return self._skills_cache

    def skills_available(self) -> bool:
        """True when at least one enabled/always skill exists; gates whether
        `use_skill` is exposed in the loop's tool list."""
        resolved = self.resolved_skills()
        return bool(resolved.enabled or resolved.always)

    def _use_skill(self, raw: dict[str, Any]) -> ToolResult:
        return use_skill(self.resolved_skills, raw)

    def _run_metric(self, _raw: dict[str, Any]) -> MetricResult:
        """Run `cfg.workflow.metric.command` in the jail.

        Return shape mirrors `_run_argv_in_jail` (returncode / stdout /
        stderr / duration_s) plus `score`: the `pattern` regex's first
        capture group as a float, or null when it does not match or parse (the
        agent can then grep stdout itself). Raises ToolError when no metric is
        configured.
        """
        metric_cfg = self._config.workflow.metric
        if metric_cfg is None:
            raise ToolError("no [workflow.metric] configured")
        argv = tuple(metric_cfg.command)
        self._emit("metric.start", cmd=list(argv))
        res = self._run_argv_in_jail(
            argv, label="metric_command", timeout_s=self._config.workflow.verify_timeout_s
        )
        if res.exec_failed:
            raise OperatorCommandUnexecutable(
                f"metric_command {list(argv)} could not be executed in the sandbox: "
                f"{res.stderr}. See run_verify_command's note: PATH is /usr/bin:/bin "
                "plus the standard bin dirs; install the tool into one of those on the "
                "host, use a path inside the workspace, or grant its real directory "
                "via sandbox.extra_read_paths."
            )
        score = parse_metric_score(res.stdout, res.stderr, pattern=metric_cfg.pattern)
        self._emit(
            "metric.end",
            cmd=list(argv),
            exit_code=res.returncode,
            duration_s=res.duration_s,
            stdout_tail=res.stdout[-2000:],
            stderr_tail=res.stderr[-2000:],
            score=score,
        )
        return MetricResult.from_exec(res, score)

    def _jail_policy(
        self,
        argv: tuple[str, ...],
        *,
        timeout_s: float | None = None,
        extra_rw_paths: tuple[Path, ...] = (),
    ) -> JailPolicy:
        return jail_policy(
            self._root,
            self._config,
            self.isolation,
            argv,
            timeout_s=timeout_s,
            extra_rw_paths=extra_rw_paths,
            extra_protect_paths=self.extra_protect_paths,
            worktree_git_dir=self._worktree_git_dir,
        )

    def _net(self) -> SessionNetwork | None:
        """The session network this dispatcher's commands join, or None.

        The RUN owns one when there is a run, so its commands and its MCP
        servers share it. A dispatcher built without one (a machine agent
        state, `agent6 review`, agent6-as-an-MCP-server, a test) makes its own
        rather than refusing: its commands still reach each other, which is
        what `session` promises, and nothing else can reach in.
        """
        if self._session_net is not None:
            return self._session_net
        if self._own_session_net is None:
            self._own_session_net = SessionNetwork.open()
        return self._own_session_net

    def _run_session(self) -> JailSession | None:
        """The run's jail process, or None to give each command its own.

        Every isolation level, `none` included: the launcher owns output
        capture and the background lifecycle, so serving them from one process
        keeps that one implementation instead of a per-level copy, and hardened
        stops paying Landlock + seccomp setup on every command. A session that
        cannot start (an older bundled launcher, a platform with no launcher at
        all) answers None once and is not retried, so the per-command path --
        and, under `none`, the plain subprocess -- remains the fallback rather
        than the run failing.

        Its confinement is fixed when it opens, so the policy is the run's, not
        the first command's: every command in the run gets the same one, and
        the background log root is granted before any command asks for it.
        """
        if not self._use_session:
            return None
        with self._session_lock:
            if self._session is None and not self._session_failed:
                rw = () if self._shells is None else (self._shells.log_root,)
                try:
                    policy = self._jail_policy(("true",), extra_rw_paths=rw)
                    net = self._net() if policy.network == "session" else None
                    self._session = JailSession.open(policy, session_net=net)
                    if self._session.startup_stderr:
                        # The run's jail came up degraded but runs (e.g. rootless
                        # podman refusing the /proc mount, so $ORIGIN toolchains
                        # will not start). Say so ONCE, here at the run's single
                        # session open -- not per command, where it would repeat.
                        self._emit("jail.degraded", detail=self._session.startup_stderr)
                except (JailUnavailableError, OSError):
                    self._session_failed = True
            return self._session

    def close_jail_session(self) -> None:
        """End the run's jail process (under strict, its PID namespace takes
        any survivors with it)."""
        with self._session_lock:
            if self._session is not None:
                self._session.close()
                self._session = None

    def _run_argv_in_jail(
        self,
        argv: tuple[str, ...],
        *,
        label: str,
        timeout_s: float | None = None,
    ) -> ExecResult:
        try:
            policy = self._jail_policy(argv, timeout_s=timeout_s)
            session = self._run_session()
            # No check-in: this is the operator's gate (verify, metric, the
            # baseline re-run) and the loop needs a verdict, not a handle.
            outcome = (
                session.run(argv, env=policy.env, timeout_s=policy.timeout_s)
                if session is not None
                else run_in_jail(
                    policy, session_net=self._net() if policy.network == "session" else None
                )
            )
        except JailUnavailableError as exc:
            raise ToolError(f"{label}: jail unavailable: {exc}") from exc
        if isinstance(outcome, BackgroundHandoff):  # pragma: no cover - no check-in was asked for
            raise ToolError(f"{label}: the jail handed back a command that was never detachable")
        res: CommandResult = outcome
        return ExecResult(
            returncode=res.returncode,
            stdout=_clip_tail(res.stdout),
            stderr=_clip_tail(res.stderr),
            duration_s=res.duration_s,
            exec_failed=res.exec_failed,
            timeout_s=policy.timeout_s,
        )
