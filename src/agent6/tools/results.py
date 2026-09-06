# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Typed tool-handler results: every handler returns one of these frozen values
instead of a bare dict.
Each owns two representations, the model-facing `to_wire()` dict and the
one-line human `summary()`.

- `to_wire()`: the dict the loop JSON-dumps into the model's
  tool_result. This is frozen LLM I/O: keys, key ORDER (dicts preserve
  insertion order), and value formats are the model-facing contract. Pinned by
  `tests/unit/test_tool_result_wire.py`.
- `summary()`: the one-line human string for the log tail / TUI; each
  result states its own (never inferred from the dict's keys).

Internal values, so frozen dataclasses (not pydantic): the wire dict is
produced at the boundary, never validated back in.
"""

from __future__ import annotations

import abc
import shlex
from dataclasses import dataclass
from typing import Any


class ToolResult(abc.ABC):
    """One tool handler's typed result: it owns the model-facing `to_wire()`
    dict and its one-line `summary()`."""

    __slots__ = ()

    @abc.abstractmethod
    def to_wire(self) -> dict[str, Any]:
        """The model-facing dict, JSON-serialized verbatim by the loop."""

    def summary(self) -> str:
        """One-line log/TUI summary. Defaults to "ok"."""
        return "ok"


def _trunc(truncated: bool) -> str:
    return " (truncated)" if truncated else ""


# --- content access ----------------------------------------------------------


@dataclass(frozen=True, slots=True)
class DocsIndexResult(ToolResult):
    """agent6_docs with no name: the list of available docs."""

    available: tuple[str, ...]

    def to_wire(self) -> dict[str, Any]:
        return {"available": list(self.available)}


@dataclass(frozen=True, slots=True)
class DocsContentResult(ToolResult):
    """agent6_docs for a named doc."""

    name: str
    content: str
    truncated: bool

    def to_wire(self) -> dict[str, Any]:
        return {"name": self.name, "content": self.content, "truncated": self.truncated}


@dataclass(frozen=True, slots=True)
class ReadFileResult(ToolResult):
    content: str
    size: int
    lines_total: int
    # Present together only for a partial read (start_line/limit given); a
    # full read omits both. None is the "full read" sentinel.
    start_line: int | None = None
    lines_returned: int | None = None
    # The file was larger than the read cap; content and the line counts are of
    # the capped prefix only. A reader must not treat lines_total as the file's
    # true length when this is set.
    truncated: bool = False

    def to_wire(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "content": self.content,
            "size": self.size,
            "lines_total": self.lines_total,
        }
        if self.start_line is not None:
            out["start_line"] = self.start_line
            out["lines_returned"] = self.lines_returned
        if self.truncated:
            out["truncated"] = True
        return out

    def summary(self) -> str:
        return f"{self.size} bytes{' (truncated)' if self.truncated else ''}"


@dataclass(frozen=True, slots=True)
class ListDirResult(ToolResult):
    entries: tuple[str, ...]
    # Entries the workspace boundary hides. Counted rather than named: the
    # listing stays true without disclosing what is hidden.
    hidden: int = 0
    # The listing stops at the cap; the rest is there, unnamed.
    truncated: bool = False

    def to_wire(self) -> dict[str, Any]:
        out: dict[str, Any] = {"entries": list(self.entries)}
        if self.hidden:
            out["hidden"] = self.hidden
        if self.truncated:
            out["truncated"] = True
        return out

    def summary(self) -> str:
        extra = f", {self.hidden} hidden" if self.hidden else ""
        cut = " (truncated)" if self.truncated else ""
        return f"{len(self.entries)} entries{extra}{cut}"


# --- search / navigation -----------------------------------------------------


@dataclass(frozen=True, slots=True)
class OutlineResult(ToolResult):
    # Each symbol row: {name: str, kind: str, line: int, col: int}.
    symbols: tuple[dict[str, Any], ...]
    truncated: bool

    def to_wire(self) -> dict[str, Any]:
        return {"symbols": list(self.symbols), "truncated": self.truncated}

    def summary(self) -> str:
        return f"{len(self.symbols)} symbols{_trunc(self.truncated)}"


@dataclass(frozen=True, slots=True)
class DefinitionsResult(ToolResult):
    """find_definition's result envelope."""

    # Rows: {name: str, kind: str, path: str, line: int, col: int}.
    definitions: tuple[dict[str, Any], ...]
    truncated: bool

    def to_wire(self) -> dict[str, Any]:
        return {"definitions": list(self.definitions), "truncated": self.truncated}

    def summary(self) -> str:
        return f"{len(self.definitions)} definitions{_trunc(self.truncated)}"


@dataclass(frozen=True, slots=True)
class ReferencesResult(ToolResult):
    """find_references's result envelope."""

    # Rows: {name: str, path: str, line: int, col: int}.
    references: tuple[dict[str, Any], ...]
    truncated: bool

    def to_wire(self) -> dict[str, Any]:
        return {"references": list(self.references), "truncated": self.truncated}

    def summary(self) -> str:
        return f"{len(self.references)} references{_trunc(self.truncated)}"


# --- filesystem writes -------------------------------------------------------


@dataclass(frozen=True, slots=True)
class EditResult(ToolResult):
    """apply_edit that wrote (not preview)."""

    applied: tuple[str, ...]
    path: str

    def to_wire(self) -> dict[str, Any]:
        return {"applied": list(self.applied), "path": self.path}

    def summary(self) -> str:
        return f"applied={list(self.applied)} path={self.path}"


@dataclass(frozen=True, slots=True)
class PatchResult(ToolResult):
    """apply_patch that wrote (not preview). A multi-file patch carries one
    (path, bytes_written) row per file in `files`; `path`/`bytes_written`
    then hold the first file and the total, and the single-file wire is
    unchanged (`files` empty)."""

    path: str
    bytes_written: int
    files: tuple[tuple[str, int], ...] = ()
    # Paths this patch DELETED (unified `+++ /dev/null` or V4A
    # `*** Delete File:`); disjoint from the written `files` rows.
    deleted: tuple[str, ...] = ()
    # Hunks the matcher HEALED rather than matched exactly (`~rstrip`,
    # `~indent`, `~moved`): the patch applied, but not verbatim, and the
    # model should know its context was off.
    healed: tuple[str, ...] = ()

    def to_wire(self) -> dict[str, Any]:
        wire: dict[str, Any] = {"path": self.path, "bytes_written": self.bytes_written}
        if self.files:
            wire["files"] = [{"path": p, "bytes_written": b} for p, b in self.files]
        if self.deleted:
            wire["deleted"] = list(self.deleted)
        if self.healed:
            wire["healed"] = list(self.healed)
        return wire

    def summary(self) -> str:
        if not self.deleted:
            if self.files:
                return f"patched {len(self.files)} files bytes={self.bytes_written}"
            return f"patched path={self.path} bytes={self.bytes_written}"
        if not self.files:
            if len(self.deleted) == 1:
                return f"deleted path={self.deleted[0]}"
            return f"deleted {len(self.deleted)} files"
        return (
            f"patched {len(self.files)} files, deleted {len(self.deleted)}"
            f" bytes={self.bytes_written}"
        )


@dataclass(frozen=True, slots=True)
class PreviewResult(ToolResult):
    """apply_edit/apply_patch with preview=true: the dry-run diff. apply_edit
    carries would_apply (the per-edit kinds); apply_patch does not."""

    path: str
    diff: str
    hunks: int
    bytes_before: int
    bytes_after: int
    truncated: bool
    would_apply: tuple[str, ...] | None = None
    # Multi-file apply_patch preview: every previewed path, in patch order
    # (`path` holds the first). Empty for a single-file preview.
    files: tuple[str, ...] = ()

    def to_wire(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "preview": True,
            "path": self.path,
            "diff": self.diff,
            "hunks": self.hunks,
            "bytes_before": self.bytes_before,
            "bytes_after": self.bytes_after,
            "truncated": self.truncated,
        }
        if self.would_apply is not None:
            out["would_apply"] = list(self.would_apply)
        if self.files:
            out["files"] = list(self.files)
        return out


# --- execution (jail-backed) -------------------------------------------------


@dataclass(frozen=True, slots=True)
class FetchResult(ToolResult):
    """`fetch`: one URL's text. A 30x carries its Location for the model to
    decide on, since redirects are never followed."""

    url: str
    status: int
    content_type: str
    body: str
    location: str = ""

    def to_wire(self) -> dict[str, Any]:
        wire: dict[str, Any] = {
            "url": self.url,
            "status": self.status,
            "content_type": self.content_type,
            "body": self.body,
        }
        if self.location and 300 <= self.status < 400:
            wire["location"] = self.location
            wire["note"] = "redirects are not followed; fetch this URL if you still want it"
        return wire

    def summary(self) -> str:
        return f"{self.status} · {len(self.body)} bytes"


@dataclass(frozen=True, slots=True)
class ExecResult(ToolResult):
    """run_command and run_verify_command: the jailed command's outcome.

    ONE shape whether the command finished or is still running: a `returncode`
    of None with a `background_id` set means it outlived its check-in and was
    handed back, and the model polls it with read_background. Never "a result
    OR a handle", which would be two shapes for one tool.
    """

    returncode: int | None
    stdout: str
    stderr: str
    duration_s: float
    exec_failed: bool
    # What actually ran, for a command the model did not choose. run_command
    # already knows its own argv; a verify gate is the operator's (or inferred),
    # and a worker that cannot see it cannot tell a failure from a stale gate.
    command: tuple[str, ...] = ()
    # Set when the command outlived its check-in and is still running as this
    # background job. `returncode` is None until it ends.
    background_id: str = ""
    # The wall-clock cap the runner enforced, 0 when none. rc=124 is the
    # jail's documented timeout result; pairing it with the cap on the wire
    # is what lets the model tell "killed at 240s" from "tests failed".
    timeout_s: float = 0.0

    def to_wire(self) -> dict[str, Any]:
        wire: dict[str, Any] = {
            "returncode": self.returncode,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "duration_s": self.duration_s,
            "exec_failed": self.exec_failed,
        }
        if self.command:
            wire["command"] = shlex.join(self.command)
        if self.returncode == 124 and self.timeout_s > 0:
            wire["timed_out"] = True
            wire["timeout_s"] = self.timeout_s
        if self.background_id:
            wire["still_running"] = True
            wire["background_id"] = self.background_id
        return wire

    def summary(self) -> str:
        if self.background_id:
            return f"still running as {self.background_id} after {self.duration_s:.1f}s"
        if self.returncode == 124 and self.timeout_s > 0:
            return f"exit=124 (timed out at {self.timeout_s:.0f}s) in {self.duration_s:.1f}s"
        return f"exit={self.returncode} in {self.duration_s:.1f}s"


@dataclass(frozen=True, slots=True)
class MetricResult(ToolResult):
    """run_metric_command: the jail outcome plus the parsed score, appended
    after the exec fields."""

    returncode: int
    stdout: str
    stderr: str
    duration_s: float
    exec_failed: bool
    score: float | None
    timeout_s: float = 0.0

    @classmethod
    def from_exec(cls, res: ExecResult, score: float | None) -> MetricResult:
        if res.returncode is None:  # pragma: no cover - the gate sets no check-in
            raise ValueError("a metric command cannot be handed back: a score needs a verdict")
        return cls(
            returncode=res.returncode,
            stdout=res.stdout,
            stderr=res.stderr,
            duration_s=res.duration_s,
            exec_failed=res.exec_failed,
            score=score,
            timeout_s=res.timeout_s,
        )

    def to_wire(self) -> dict[str, Any]:
        wire: dict[str, Any] = {
            "returncode": self.returncode,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "duration_s": self.duration_s,
            "exec_failed": self.exec_failed,
            "score": self.score,
        }
        if self.returncode == 124 and self.timeout_s > 0:
            wire["timed_out"] = True
            wire["timeout_s"] = self.timeout_s
        return wire

    def summary(self) -> str:
        if self.returncode == 124 and self.timeout_s > 0:
            return f"exit=124 (timed out at {self.timeout_s:.0f}s) in {self.duration_s:.1f}s"
        return f"exit={self.returncode} in {self.duration_s:.1f}s"


# --- run control -------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class FinishSessionResult(ToolResult):
    summary_text: str
    result: dict[str, Any] | None
    stale_gate: str = ""

    def to_wire(self) -> dict[str, Any]:
        wire: dict[str, Any] = {
            "acknowledged": True,
            "summary": self.summary_text,
            "result": self.result,
        }
        if self.stale_gate:
            # Say plainly that nothing changed, so the model does not finish
            # believing it swapped the gate.
            wire["stale_gate"] = (
                f"recorded for the operator: {self.stale_gate}."
                " This run's gate is unchanged and this run does not pass."
            )
        return wire


@dataclass(frozen=True, slots=True)
class FinishPlanningResult(ToolResult):
    summary_text: str
    plan_bytes: int

    def to_wire(self) -> dict[str, Any]:
        return {"acknowledged": True, "summary": self.summary_text, "plan_bytes": self.plan_bytes}


@dataclass(frozen=True, slots=True)
class AnswersResult(ToolResult):
    answers: tuple[str, ...]

    def to_wire(self) -> dict[str, Any]:
        return {"answers": list(self.answers)}

    def summary(self) -> str:
        answered = sum(1 for a in self.answers if str(a).strip())
        return f"{answered}/{len(self.answers)} answered"


# --- DAG (task graph) --------------------------------------------------------


@dataclass(frozen=True, slots=True)
class AddTaskResult(ToolResult):
    id: str
    parent_id: str | None
    title: str
    status: str

    def to_wire(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "parent_id": self.parent_id,
            "title": self.title,
            "status": self.status,
        }

    def summary(self) -> str:
        return f"{self.status}: {str(self.title)[:60]}"


@dataclass(frozen=True, slots=True)
class UpdateTaskResult(ToolResult):
    id: str
    status: str
    title: str
    depends_on: tuple[str, ...] = ()

    def to_wire(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "status": self.status,
            "title": self.title,
            "depends_on": list(self.depends_on),
        }

    def summary(self) -> str:
        return f"{self.status}: {str(self.title)[:60]}"


@dataclass(frozen=True, slots=True)
class ListTasksResult(ToolResult):
    # Each task row: {id: str, parent_id: str | None, title: str, status: str,
    # acceptance: str, relevant_paths: list[str], depends_on: list[str]}.
    tasks: tuple[dict[str, Any], ...]
    count: int

    def to_wire(self) -> dict[str, Any]:
        return {"tasks": list(self.tasks), "count": self.count}

    def summary(self) -> str:
        return f"{self.count} tasks"


# --- operator knowledge ------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SkillResult(ToolResult):
    skill: str
    file: str
    content: str

    def to_wire(self) -> dict[str, Any]:
        return {"skill": self.skill, "file": self.file, "content": self.content}

    def summary(self) -> str:
        return f"skill {self.skill}/{self.file} ({len(self.content)} chars)"


# --- MCP passthrough ---------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RawResult(ToolResult):
    """An operator-configured MCP server's result: an opaque dict forwarded to
    the model unchanged. agent6 does not know its shape, so the summary is the
    generic 'ok'."""

    payload: dict[str, Any]

    def to_wire(self) -> dict[str, Any]:
        return self.payload


# --- background commands -----------------------------------------------------


@dataclass(frozen=True, slots=True)
class BackgroundResult(ToolResult):
    """A background command tool's result. The roster rides on every one of
    them: whatever the model asked, it also learns that a command it started
    has died."""

    shells: tuple[str, ...]
    output: str | None = None

    def to_wire(self) -> dict[str, Any]:
        wire: dict[str, Any] = {"shells": list(self.shells)}
        if self.output is not None:
            wire["output"] = self.output
        return wire

    def summary(self) -> str:
        return self.shells[0] if len(self.shells) == 1 else f"{len(self.shells)} background"


# --- other sessions ----------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SessionsResult(ToolResult):
    """The project's sessions, and one session's conversation when asked for."""

    sessions: tuple[str, ...]
    conversation: str | None = None

    def to_wire(self) -> dict[str, Any]:
        wire: dict[str, Any] = {"sessions": list(self.sessions)}
        if self.conversation is not None:
            wire["conversation"] = self.conversation
        return wire

    def summary(self) -> str:
        return f"{len(self.sessions)} session{'' if len(self.sessions) == 1 else 's'}"
