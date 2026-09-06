# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The shared view-model: the JSONL event stream folded into render-ready state.

This is the data contract every front-end consumes. The CLI, the textual TUI,
and the web UI all read the same `<run-dir>/logs.jsonl`, fold it through
the same pure functions here, and only differ in how they paint the result.
Keeping the fold in one place is what stops the front-ends from drifting.

Layout:
    state.py             pure event-fold: list[event] -> SessionState (a run / agent state).
    machine_state.py     pure fold: machine journal -> MachineState (+ the watch cursor).
    tail.py              stdlib JSONL file tailer (the event source).
    transcript.py        event-fold: logs.jsonl -> live conversation TranscriptItems.
    transcript_render.py fold + Markdown render of the per-call provider transcripts.
    listing.py           run-dir scan -> SessionSummary rows (sessions list / pickers).
    format.py            shared glyphs + cost/status formatting.
    config_view.py       effective-config tree -> the `config show` view.

No I/O in the folds, no textual, no async: just frozen dataclasses and pure
functions, so a viewer in any language mirrors `SessionState` / `MachineState`
field-for-field.
"""

from __future__ import annotations

from agent6.viewmodel.events import event_epoch
from agent6.viewmodel.listing import (
    LIVE_STATUS_WORDS,
    OPERATOR_PROMPT_EVENTS,
    LogScan,
    SessionSummary,
    StatusFacts,
    died_without_end,
    first_task_line,
    is_session_husk,
    is_winner,
    newest_session_dir,
    produced_result,
    scan_session_log,
    session_compare,
    session_dirs,
    session_is_live,
    session_mtime,
    status_for_session_dir,
    status_word,
    summarize_session_dir,
    summary_row,
    task_snippet,
)
from agent6.viewmodel.log_line import format_log_line
from agent6.viewmodel.machine_state import (
    MachineState,
    MachineStateView,
    MachineSummary,
    MachineWatchCursor,
    NotificationView,
    TransitionView,
    fold_machine,
    machine_files,
    machine_instance_dirs,
    machine_is_parked,
    machine_operator_blocked,
    machine_spend,
    machine_state_as_dict,
    machine_status_word,
    machine_verb_refusal,
    machine_word_for_dir,
    newest_state_log,
    notification_key,
    read_complete_lines,
    summarize_machine_dir,
)
from agent6.viewmodel.policy import SessionPolicy, session_policy
from agent6.viewmodel.snapshot import (
    UnknownStepError,
    existing_run_branch,
    machine_snapshot,
    manifest_branches,
    manifest_header,
    session_snapshot,
)
from agent6.viewmodel.state import (
    MAX_LOG_TAIL,
    ApprovalPrompt,
    BudgetView,
    CommitStep,
    QuestionPrompt,
    RoleCall,
    SessionState,
    TaskNodeView,
    ToolCallView,
    VerifyView,
    apply_event,
    approval_parts,
    fold_session,
    initial_state,
    open_question,
    session_state_as_dict,
    status_facts,
    task_tree_views,
)
from agent6.viewmodel.tail import LogTail, tail_events
from agent6.viewmodel.transcript import (
    TranscriptFold,
    TranscriptItem,
    fold_transcript,
    operator_inputs,
    restate,
    salient_arg,
    worker_models,
)

__all__ = [
    "LIVE_STATUS_WORDS",
    "MAX_LOG_TAIL",
    "OPERATOR_PROMPT_EVENTS",
    "ApprovalPrompt",
    "BudgetView",
    "CommitStep",
    "LogScan",
    "LogTail",
    "MachineState",
    "MachineStateView",
    "MachineSummary",
    "MachineWatchCursor",
    "NotificationView",
    "QuestionPrompt",
    "RoleCall",
    "SessionPolicy",
    "SessionState",
    "SessionSummary",
    "StatusFacts",
    "TaskNodeView",
    "ToolCallView",
    "TranscriptFold",
    "TranscriptItem",
    "TransitionView",
    "UnknownStepError",
    "VerifyView",
    "apply_event",
    "approval_parts",
    "died_without_end",
    "event_epoch",
    "existing_run_branch",
    "first_task_line",
    "fold_machine",
    "fold_session",
    "fold_transcript",
    "format_log_line",
    "initial_state",
    "is_session_husk",
    "is_winner",
    "machine_files",
    "machine_instance_dirs",
    "machine_is_parked",
    "machine_operator_blocked",
    "machine_snapshot",
    "machine_spend",
    "machine_state_as_dict",
    "machine_status_word",
    "machine_verb_refusal",
    "machine_word_for_dir",
    "manifest_branches",
    "manifest_header",
    "newest_session_dir",
    "newest_state_log",
    "notification_key",
    "open_question",
    "operator_inputs",
    "produced_result",
    "read_complete_lines",
    "restate",
    "salient_arg",
    "scan_session_log",
    "session_compare",
    "session_dirs",
    "session_is_live",
    "session_mtime",
    "session_policy",
    "session_snapshot",
    "session_state_as_dict",
    "status_facts",
    "status_for_session_dir",
    "status_word",
    "summarize_machine_dir",
    "summarize_session_dir",
    "summary_row",
    "tail_events",
    "task_snippet",
    "task_tree_views",
    "worker_models",
]
