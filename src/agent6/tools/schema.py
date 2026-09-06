# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Tool input schemas, pydantic models converted to JSON Schema for Anthropic."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache
from typing import Annotated, Any, ClassVar, get_args

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from agent6.graph.models import NodeStatus
from agent6.types import session_kind

# Derived from the NodeStatus Literal so the task-status vocabulary has ONE
# owner (a new status can't silently drift the tool schema). Same order, so
# the LLM-facing pattern bytes are unchanged; pinned in
# tests/unit/test_tool_schema_wire.py.
_STATUS_PATTERN = f"^({'|'.join(get_args(NodeStatus))})$"

# A task id as the DAG tools accept it: ULIDs are exactly 26 chars.
Ulid = Annotated[str, StringConstraints(min_length=26, max_length=26)]


class _ToolInput(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    TOOL_NAME: ClassVar[str] = ""
    TOOL_DESCRIPTION: ClassVar[str] = ""


class ReadFileInput(_ToolInput):
    TOOL_NAME: ClassVar[str] = "read_file"
    TOOL_DESCRIPTION: ClassVar[str] = (
        "Read a UTF-8 text file. `path` is repo-root-relative (absolute only"
        " inside granted directories). start_line (1-based) and limit select a"
        " range; very large files truncate (truncated: true). The `outline`"
        " tool shows a file's structure without its content."
    )

    path: str = Field(min_length=1)
    start_line: int = Field(default=1, ge=1)
    limit: int | None = Field(default=None, gt=0)


class Agent6DocsInput(_ToolInput):
    TOOL_NAME: ClassVar[str] = "agent6_docs"
    TOOL_DESCRIPTION: ClassVar[str] = (
        "Read agent6's OWN documentation to answer questions about how to USE "
        "agent6 (configuring providers/models, sandbox isolation, machines, the "
        "CLI, budgets, etc.). Call with an empty `name` to list the available "
        "docs, or set `name` to one of them (e.g. README, USAGE, "
        "CONFIG, SECURITY, STATE-MACHINES, ARCHITECTURE) to read its markdown."
    )

    name: str = Field(default="")


class ListDirInput(_ToolInput):
    TOOL_NAME: ClassVar[str] = "list_dir"
    TOOL_DESCRIPTION: ClassVar[str] = (
        "List immediate entries in a directory (non-recursive). `path` is "
        "repo-root-relative; defaults to '.'. Dot-prefixed entries are listed;"
        " `hidden` counts entries the workspace boundary withholds. Returns"
        " names with a trailing '/' for directories, at most 1,000 (`truncated`"
        " says so). For a recursive view, use `run_command` (e.g. `rg --files`,"
        " `find`)."
    )

    path: str = Field(default=".")


class ApplyEditInput(_ToolInput):
    TOOL_NAME: ClassVar[str] = "apply_edit"
    TOOL_DESCRIPTION: ClassVar[str] = (
        "Edit one file. `edits` is an array of {old_string, new_string, kind?}."
        " Each old_string occurs exactly once in the file, byte for byte."
        ' kind="create" makes a new file and kind="overwrite" replaces an'
        " existing file whole: for both, empty old_string, the full content in"
        " new_string, the only edit in the array. An omitted kind follows the"
        ' pair: an empty old_string means "create", any other means "replace".'
        " A miss that matches exactly one region up to a uniform indent shift"
        " is healed and reported as `replace~indent`."
        " preview=true returns the would-be diff without touching disk."
    )

    path: str = Field(min_length=1)
    edits: tuple[EditPair, ...] = Field(min_length=1)
    preview: bool = False

    @model_validator(mode="after")
    def _check_whole_file_edit_is_sole(self) -> ApplyEditInput:
        # `create` and `overwrite` write the whole file from `new_string`, so
        # combining either with other edits is nonsensical: the dispatcher's
        # create branch only guards "file already exists" for the FIRST edit,
        # so a `create` placed after a `replace` would skip that guard and
        # silently overwrite the file (discarding the prior edits). Require a
        # whole-file edit to be the sole edit and fail loud at the trust
        # boundary instead.
        whole = [e.kind for e in self.edits if e.kind in WHOLE_FILE_KINDS]
        if len(self.edits) > 1 and whole:
            raise ValueError(
                f"kind={whole[0]!r} must be the only edit (it writes the entire file);"
                " do not combine it with other edits. An edit with an empty old_string"
                " resolves to 'create'; to add at the end of a file instead, use its"
                " last line as old_string"
            )
        return self


class ApplyPatchInput(_ToolInput):
    TOOL_NAME: ClassVar[str] = "apply_patch"
    TOOL_DESCRIPTION: ClassVar[str] = (
        "Patch files. Accepts a standard unified diff (`--- a/PATH`,"
        " `+++ b/PATH`, @@ hunks; `--- /dev/null` creates; `+++ /dev/null`"
        " deletes, the hunks must remove the whole file) or OpenAI's"
        " *** Begin/Add/Update/Delete File/End Patch format. Multi-file"
        " patches are staged all-or-nothing (unified needs `diff --git`"
        " separators between files); a write that fails part way names the"
        " files already changed. Context lines match exactly or heal through a strict"
        " ladder (trailing whitespace / uniform indent / unique moved block;"
        " the result names each heal). `path`"
        " optional (taken from headers; single-file only)."
        " preview=true echoes the diffs without writing."
    )

    path: str = ""
    patch: str = Field(min_length=1)
    preview: bool = False


# The edit kinds that write the whole file from `new_string`: `create` refuses
# an existing file (a model that thinks the file is new must not clobber it),
# `overwrite` states the intent to replace one whole (a rewrite from a stub,
# where a replace would need the byte-exact old text).
WHOLE_FILE_KINDS = frozenset({"create", "overwrite"})


class EditPair(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    # Omitting the discriminator is the common small-model shape, so it is
    # resolved from the pair itself rather than rejected: an empty old_string
    # can only mean "write this whole file", and a non-empty one can only mean
    # a replace. An omitted kind resolves to `create`, never `overwrite`, so a
    # model that thinks a file is new still cannot clobber one that exists.
    # Left as "replace", the natural write-a-new-file call was refused forever:
    # a live `machine create` spent three attempts ping-ponging between the
    # two refusals and stopped on the tool-error streak.
    kind: str = Field(default="", pattern="^(|replace|create|overwrite)$")
    old_string: str = ""
    new_string: str

    @model_validator(mode="before")
    @classmethod
    def _resolve_kind(cls, data: Any) -> Any:
        """Fill an omitted `kind` from the pair's shape (see the field)."""
        if isinstance(data, dict) and not data.get("kind"):
            data = {**data, "kind": "replace" if data.get("old_string") else "create"}
        return data

    @model_validator(mode="after")
    def _check_shape(self) -> EditPair:
        # kind="replace" with an empty old_string would match anywhere (or
        # nowhere depending on str.count semantics); reject it loud so the
        # model gets a clear error instead of a silent corruption.
        if self.kind == "replace" and self.old_string == "":
            raise ValueError(
                "old_string must be non-empty for kind='replace'. For a file that does not"
                " exist yet use kind='create'; to rewrite one whole, kind='overwrite'; to"
                " add at the end of one, use its last line as old_string"
            )
        # A whole-file kind ignores old_string; reject a non-empty value to
        # catch the common LLM mistake of pasting context into the wrong field.
        if self.kind in WHOLE_FILE_KINDS and self.old_string != "":
            raise ValueError(
                f"old_string must be empty for kind={self.kind!r}, which writes the whole"
                " file from new_string (kind='create' for a new file, 'overwrite' for an"
                " existing one)"
            )
        return self


class RunVerifyInput(_ToolInput):
    TOOL_NAME: ClassVar[str] = "run_verify_command"
    TOOL_DESCRIPTION: ClassVar[str] = (
        "Run this run's verify command in the sandbox: the operator's, or one"
        " inferred from the repo. No arguments; the result names the argv that ran."
    )


class RunCommandInput(_ToolInput):
    TOOL_NAME: ClassVar[str] = "run_command"
    TOOL_DESCRIPTION: ClassVar[str] = (
        "Run a command in the sandbox. argv is an array of strings, no shell."
        " Requires the run_commands capability; 'ask' prompts the operator."
        " Under jailed isolation PATH is minimal; absolute paths like"
        " /usr/bin/python3 resolve regardless. A command still running at the"
        " check-in is handed back with returncode null, still_running true,"
        " and a background_id,"
        " and keeps running: poll with read_background, stop with"
        " stop_background, or continue working; output printed so far comes"
        " with the hand-back. background=true returns the handle at once."
        " Commands in one run share a network, so a server started here answers"
        " later commands; under strict isolation that network is the run's own,"
        " elsewhere it is the host's. All background commands die when the run"
        " ends."
    )

    argv: tuple[str, ...] = Field(min_length=1)
    background: bool = False


class FetchInput(_ToolInput):
    TOOL_NAME: ClassVar[str] = "fetch"
    TOOL_DESCRIPTION: ClassVar[str] = (
        "Fetch an https URL (GET; other schemes are refused). Returns status,"
        " content type, and body text; a response over 1 MiB is refused, not"
        " truncated, and a redirect is returned, not followed."
    )

    url: str = Field(min_length=1)


class ReadSessionInput(_ToolInput):
    TOOL_NAME: ClassVar[str] = "read_session"
    TOOL_DESCRIPTION: ClassVar[str] = (
        "Read another session's transcript summary by id; with no id, list the"
        " sessions. `query` keeps the sessions whose task or journal contains"
        " it (40 newest); `max_chars` bounds the transcript, which keeps its"
        " tail. Read-only."
    )

    id: str = ""
    query: str = ""
    max_chars: int = Field(default=20_000, ge=500, le=200_000)


class ReadBackgroundInput(_ToolInput):
    TOOL_NAME: ClassVar[str] = "read_background"
    TOOL_DESCRIPTION: ClassVar[str] = (
        "Read a background command's output. `id` from run_command's"
        " hand-back; `tail_lines` bounds the tail; `wait_s` waits for exit up"
        " to that many seconds (0 = look now; omitted = the run's check-in"
        " interval, 900 s by default). Returns the output tail, running state,"
        " and returncode when finished. With no `id`, returns this run's"
        " background roster."
    )

    id: str = ""
    tail_lines: int = Field(default=200, ge=1, le=2000)
    # None = the operator's configured check-in; 0 = look without waiting. The
    # interval has ONE owner ([workflow].command_checkin_s), so the default is
    # resolved by the dispatcher rather than duplicated here.
    wait_s: float | None = Field(default=None, ge=0.0)


class StopBackgroundInput(_ToolInput):
    TOOL_NAME: ClassVar[str] = "stop_background"
    TOOL_DESCRIPTION: ClassVar[str] = (
        "Stop a background command by background_id. Returns the background roster with"
        " it marked stopped and its exit code; read its output first with read_background."
    )

    id: str


class RunMetricInput(_ToolInput):
    TOOL_NAME: ClassVar[str] = "run_metric_command"
    TOOL_DESCRIPTION: ClassVar[str] = (
        "Run the configured metric command. Returns the parsed score. The"
        " harness also runs it automatically after each verify-passing edit."
    )


class FinishSessionInput(_ToolInput):
    TOOL_NAME: ClassVar[str] = "finish_session"
    TOOL_DESCRIPTION: ClassVar[str] = (
        "End the run cleanly. summary: for the operator, what was done and"
        " left undone, a paragraph at most (a machine's step answers in one"
        " line). Tool calls after it are not executed."
    )

    summary: str = Field(min_length=1)
    result: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Optional JSON object. When the task names a result schema,"
            " return the matching object here; validated at the trust"
            " boundary."
        ),
    )
    stale_gate: str = Field(
        default="",
        description=(
            "Set only when the verify command no longer matches the task: it"
            " pins behaviour this run deliberately changed, or cannot run."
            " Give the command you believe is right; it records a proposal"
            " and never changes the gate or passes the run. A merely failing"
            " gate means fix the work."
        ),
    )


class FinishPlanningInput(_ToolInput):
    TOOL_NAME: ClassVar[str] = "finish_planning"
    TOOL_DESCRIPTION: ClassVar[str] = (
        "Signal that the planning pass is complete and the workflow should "
        "exit. Available ONLY in plan mode (`agent6 plan`); in execution "
        "mode use `finish_session` instead. `plan_markdown` is the full plan "
        "document (markdown) that gets saved to the run directory as "
        "`plan.md`. Include: a one-line `# Plan: <title>`, the original "
        "task, context discovered, an ordered task list with acceptance "
        "criteria, any open questions for the user as `**Q:** ...` blocks "
        "with blank `**A:**` lines, and the verification approach. The "
        "operator can edit this file (`agent6 plan edit <run-id>`) to "
        "fill in answers, then hand it to `agent6 run --from-plan "
        "<run-id>` to start execution. `summary` is a one-paragraph "
        "description surfaced to the operator at exit. Do not call any "
        "other tools after finish_planning."
    )

    # Per-field descriptions so the disambiguation lives IN the JSON schema the
    # model fills, not only in the prose above. finish_planning is the one finish
    # tool whose fields were self-undocumented, and models put the whole plan
    # into `summary` (listed first, and a natural sink for "primary output"),
    # leaving a degenerate plan.md that still passed min_length=1. finish_session's
    # `result` already carries a field description; this matches it.
    summary: str = Field(
        min_length=1,
        description=(
            "A one-paragraph description of the plan, surfaced to the operator at "
            "exit. This is NOT the plan itself -- the full plan goes in plan_markdown."
        ),
    )
    plan_markdown: str = Field(
        min_length=1,
        description=(
            "The FULL plan document in markdown, saved verbatim to plan.md and fed "
            "to `agent6 run --from-plan`. This is the deliverable: put the entire "
            "plan here (title, task, context, ordered task list with acceptance "
            "criteria, open questions, verification) -- not a short blurb, and not "
            "in summary."
        ),
    )


# DAG-as-tool surface. Lets the agent maintain its own task
# breakdown in the persistent curator-backed graph. Survives crashes via
# <run-dir>/graph.jsonl; operator can inspect via `agent6 attach`.


class DagAddTaskInput(_ToolInput):
    TOOL_NAME: ClassVar[str] = "add_task"
    TOOL_DESCRIPTION: ClassVar[str] = (
        "Add a subtask to the persistent task graph."
        " parent_id attaches under an existing task (default the root). title"
        " is a short imperative; acceptance the verifiable condition. after"
        " inserts directly after that sibling. depends_on lists task ULIDs"
        " that must pass first. standing=true marks the run's never-finishing"
        " fallback goal: worked when nothing else is ready, never passes,"
        " retired with skipped/obsolete. Returns the new task's ULID."
    )

    title: str = Field(min_length=1)
    # ULID is exactly 26 chars, like update_task; None still means
    # "under the run root". "" silently attached to root before the constraint.
    parent_id: str | None = Field(default=None, min_length=26, max_length=26)
    # A sibling under the same parent; the task lands right after it.
    after: str | None = Field(default=None, min_length=26, max_length=26)
    rationale: str = ""
    acceptance: str = ""
    relevant_paths: tuple[str, ...] = ()
    depends_on: tuple[Ulid, ...] = ()
    standing: bool = False


class DagUpdateTaskInput(_ToolInput):
    TOOL_NAME: ClassVar[str] = "update_task"
    TOOL_DESCRIPTION: ClassVar[str] = (
        "Update a task: status (in_progress marks the task being worked, which"
        " the harness sets on the focus task; passed once the verify confirms"
        " it, or once you have checked it yourself in a gateless run), or"
        " depends_on (task ULIDs that must be settled first); a note rides"
        " along with a status change. Fields omitted stay unchanged. An end is"
        " final: a passed task takes only obsolete, and a skipped or obsolete"
        " one stays retired -- add_task if the work is needed after all."
    )

    id: str = Field(min_length=26, max_length=26)
    status: str | None = Field(default=None, pattern=_STATUS_PATTERN)
    note: str = ""
    depends_on: tuple[Ulid, ...] = ()


class DagListTasksInput(_ToolInput):
    TOOL_NAME: ClassVar[str] = "list_tasks"
    TOOL_DESCRIPTION: ClassVar[str] = (
        "List the task graph: ids, titles, statuses, parents, acceptance and"
        " dependencies. `status` keeps only the tasks in that status."
    )

    # The same status enum update_task uses, so a typo is a schema rejection
    # rather than an empty result.
    status: str | None = Field(default=None, pattern=_STATUS_PATTERN)


class UseSkillInput(_ToolInput):
    TOOL_NAME: ClassVar[str] = "use_skill"
    TOOL_DESCRIPTION: ClassVar[str] = (
        "Load an installed skill's full instructions by name (from the <skills>"
        " index). `file` reads one of that skill's own supplementary files"
        " instead, by its path inside the skill's directory."
    )

    name: str = Field(min_length=1, max_length=100)
    file: str | None = Field(default=None, min_length=1, max_length=300)


class OutlineInput(_ToolInput):
    TOOL_NAME: ClassVar[str] = "outline"
    TOOL_DESCRIPTION: ClassVar[str] = (
        "Structural outline of a source file: top-level and nested defs,"
        " classes, and their line numbers. Cheaper than reading the file when"
        " you need shape, not content."
    )

    path: str = Field(min_length=1)


class FindDefinitionInput(_ToolInput):
    TOOL_NAME: ClassVar[str] = "find_definition"
    TOOL_DESCRIPTION: ClassVar[str] = (
        "Find where a symbol is defined (tree-sitter; excludes strings and"
        " comments). Returns name, kind, and file:line rows. Cheaper than"
        " grep for symbols."
    )

    symbol: str = Field(min_length=1)


class FindReferencesInput(_ToolInput):
    TOOL_NAME: ClassVar[str] = "find_references"
    TOOL_DESCRIPTION: ClassVar[str] = (
        "List references to a symbol across the repo (tree-sitter; excludes"
        " strings and comments). Returns file:line rows."
    )

    symbol: str = Field(min_length=1)


class UserQuestion(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    # Operator-facing model text, capped like every other model-written string
    # in this file: it reaches the journal, an ACP permission title, the TUI
    # modal and the web composer verbatim, and a model that runs away writes
    # every one of them. A question is a sentence and an option is a label.
    question: str = Field(min_length=1, max_length=2_000)
    options: tuple[Annotated[str, StringConstraints(max_length=200)], ...] = Field(
        default=(), max_length=10
    )


class AskUserInput(_ToolInput):
    TOOL_NAME: ClassVar[str] = "ask_user"
    TOOL_DESCRIPTION: ClassVar[str] = (
        "Ask the operator and wait. Use for decisions the repo and task cannot"
        " settle, or when the task says to check with the operator; a question"
        " written as plain text is never seen. `questions` is an array of"
        " {question, options?}; give 2-4 options for a choice (the CLI, TUI and"
        " web also take typed text; an editor over ACP answers only with an"
        " option, and skips a question that offers none); batch related"
        " questions into one call. Returns"
        " {answers: [...]} aligned to questions. Headless runs with nobody"
        " watching return empty answers."
    )

    questions: tuple[UserQuestion, ...] = Field(min_length=1, max_length=8)

    @model_validator(mode="before")
    @classmethod
    def _accept_flat_single_question(cls, data: Any) -> Any:
        # A model that sends a lone question flat (question=..., options=...) rather
        # than wrapping it in `questions` still works -- fold it into the list.
        if isinstance(data, dict) and "questions" not in data and "question" in data:
            q: dict[str, Any] = {"question": data.get("question")}
            if "options" in data:
                q["options"] = data.get("options")
            data = {k: v for k, v in data.items() if k not in ("question", "options")}
            data["questions"] = [q]
        return data


ApplyEditInput.model_rebuild()

ALL_TOOLS: tuple[type[_ToolInput], ...] = (
    ReadFileInput,
    ListDirInput,
    OutlineInput,
    FindDefinitionInput,
    FindReferencesInput,
    ApplyEditInput,
    ApplyPatchInput,
    RunVerifyInput,
    RunCommandInput,
    ReadSessionInput,
    FetchInput,
    ReadBackgroundInput,
    StopBackgroundInput,
)

# Extra tools exposed only to the single-loop workflow. Kept separate from
# ALL_TOOLS so the read-only ToolDispatcher surface used by tests and external
# callers does not advertise loop-only control tools.
LOOP_EXTRA_TOOLS: tuple[type[_ToolInput], ...] = (
    RunMetricInput,
    FinishSessionInput,
    AskUserInput,
    DagAddTaskInput,
    DagUpdateTaskInput,
    DagListTasksInput,
    # Operator-installed skills (hidden by the dispatcher when none are
    # installed or [skills].enabled is off).
    UseSkillInput,
)

# Tool list for plan mode (`agent6 plan`). Excludes the
# execution-mode terminal tool (`finish_session`) and the metric tool
# (planning never iterates a metric); adds `finish_planning` instead.
# Plan-mode also filters `apply_edit` / `apply_patch` out of `ALL_TOOLS`
# (mode_tools below) so a planner cannot accidentally mutate source.
PLAN_EXTRA_TOOLS: tuple[type[_ToolInput], ...] = (
    DagAddTaskInput,
    DagUpdateTaskInput,
    DagListTasksInput,
    FinishPlanningInput,
)

# Tool list for ask mode (`agent6 ask`). Edit-free Q&A: like plan it filters
# `apply_edit`/`apply_patch` out of `ALL_TOOLS` (mode_tools below), and it
# exposes NO control tools (no DAG, no finish_planning, no finish_session -- the
# agent answers by emitting its final message as prose, a "silent finish"). It
# DOES add `agent6_docs` so it can answer "how do I use agent6" questions.
ASK_EXTRA_TOOLS: tuple[type[_ToolInput], ...] = (Agent6DocsInput,)

# Tool list for a read-only machine `agent` state (the dispatcher's "machine"
# mode): navigation plus `finish_session`, whose `result` carries the state's
# structured output; no edit/patch/verify/run_command/DAG/metric tools, which
# only tempt a weak model into writing files or spelunking the repo.
MACHINE_EXTRA_TOOLS: tuple[type[_ToolInput], ...] = (FinishSessionInput,)


@dataclass(frozen=True, slots=True)
class ModeTools:
    """One mode's LLM tool surface: `base` (ALL_TOOLS minus the mode's blocked
    mutators) plus `extras` (its control tools). `tool_definitions` exposes
    exactly `base + extras`; the dispatcher refuses names outside `permitted`
    as its backstop, so exposure and enforcement cannot drift apart.
    `permitted` is `names` plus agent6_docs, which is exposed only in ask
    (elsewhere it is tool-list noise) but safe to execute anywhere."""

    base: tuple[type[_ToolInput], ...]
    extras: tuple[type[_ToolInput], ...]
    names: frozenset[str]
    permitted: frozenset[str]


# The mode-specific additions. Everything else about a mode is read off its
# `SessionKind`; these are the one thing a record cannot carry, being tool
# classes this module defines.
_EXTRA_TOOLS: dict[str, tuple[type[_ToolInput], ...]] = {
    "plan": PLAN_EXTRA_TOOLS,
    "ask": ASK_EXTRA_TOOLS,
    "machine": MACHINE_EXTRA_TOOLS,
    "agent": MACHINE_EXTRA_TOOLS,
}


@cache
def mode_tools(mode: str) -> ModeTools:
    kind = session_kind(mode)
    extras = _EXTRA_TOOLS.get(mode, LOOP_EXTRA_TOOLS)
    blocked: set[str] = set()
    if not kind.edits:
        # Read-only modes: no in-process file mutation.
        blocked = {ApplyEditInput.TOOL_NAME, ApplyPatchInput.TOOL_NAME}
    if not kind.runs_commands:
        # Machine authoring / agent states additionally never run commands:
        # the deliverable is the finish_session payload, and command tools only
        # tempt a weak model into spelunking.
        # `ask` keeps run_command for read-only, approval-gated investigation.
        # read_session and fetch go with them: a machine state answers about
        # ITS input, so this project's run history is not its business, and
        # neither is the network -- a deliverable assembled from a page the
        # state fetched is not the deliverable the operator asked for.
        blocked |= {
            RunVerifyInput.TOOL_NAME,
            RunCommandInput.TOOL_NAME,
            ReadSessionInput.TOOL_NAME,
            FetchInput.TOOL_NAME,
        }
    if not kind.edits:
        # Only a session that edits owns a background command's lifetime: every
        # other mode is a short read-only pass, and a command killed at its end
        # would be started for nothing.
        blocked |= {
            ReadBackgroundInput.TOOL_NAME,
            StopBackgroundInput.TOOL_NAME,
        }
    base = tuple(cls for cls in ALL_TOOLS if cls.TOOL_NAME not in blocked)
    names = frozenset(cls.TOOL_NAME for cls in (*base, *extras))
    return ModeTools(
        base=base,
        extras=extras,
        names=names,
        permitted=names | {Agent6DocsInput.TOOL_NAME},
    )


def _strip_titles(node: Any, *, keys_are_names: bool = False) -> Any:
    """Drop pydantic's auto "title" keys: they duplicate the field name in
    Title Case and carry no signal on the wire (~1.6k chars across the
    surface). Only schema-level titles go: inside a `properties` (or `$defs`)
    map the keys are field names, and a field NAMED title (add_task's) stays."""
    if isinstance(node, dict):
        out: dict[str, Any] = {}
        for k, v in node.items():
            if k == "title" and not keys_are_names and isinstance(v, str):
                continue
            out[k] = _strip_titles(v, keys_are_names=k in ("properties", "$defs"))
        return out
    if isinstance(node, list):
        return [_strip_titles(v) for v in node]
    return node


def wire_schema(cls: type[_ToolInput]) -> dict[str, Any]:
    """The JSON schema of *cls* as the model receives it: pydantic's, minus the
    title noise, with "type" present (the API wants the schema bare, not
    wrapped). The one builder behind the loop's tool list and the descriptor
    dump below, so what tests pin is what the model gets."""
    schema = _strip_titles(cls.model_json_schema())
    schema.setdefault("type", "object")
    return schema


def schemas_as_provider_tools() -> list[dict[str, Any]]:
    """Emit Anthropic-API-shape tool descriptors. (kept dict-typed to avoid circular import)"""
    return [
        {
            "name": cls.TOOL_NAME,
            "description": cls.TOOL_DESCRIPTION,
            "input_schema": wire_schema(cls),
        }
        for cls in ALL_TOOLS
    ]
