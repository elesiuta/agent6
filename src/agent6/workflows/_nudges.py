# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Mid-run harness interjections: when the loop speaks and what it says.

Each nudge/gate is a threshold (when it fires) plus a directive (the text
injected as a user-role harness message). The loop owns detection and
injection; this module owns the tuning values and the words.
"""

from __future__ import annotations

import hashlib
import re

# No-progress spiral guard (run mode): N consecutive verify failures sharing
# ONE normalized signature. A green verify or a DIFFERENT failure (progress
# through the error list) resets the streak, so a healthy run never pays for
# it. Signatures ignore line numbers, addresses, and durations, else cosmetic
# drift between identical failures defeats the detector.
# Thresholds + evidence: bench/coreagent/FINDINGS.md.
NO_PROGRESS_NUDGE_AFTER = 4
NO_PROGRESS_ESCALATE_AFTER = 7
# Third stage: both nudges delivered and unheeded, so stop honestly rather
# than spend the rest of the budget on a proven non-strategy.
NO_PROGRESS_STOP_AFTER = 10

# Tool-error spiral guard (run mode). Distinct from the verify streak: this
# counts consecutive tool calls that raise the SAME error (name + error text
# with digits stripped, so a runaway that varies its args but trips the same
# "arguments not valid JSON" / "pattern too long" error still accumulates).
# Any successful tool call, or a different error, resets it.
TOOL_ERROR_NUDGE_AFTER = 3
TOOL_ERROR_ESCALATE_AFTER = 5
TOOL_ERROR_STOP_AFTER = 8

TOOL_ERROR_NUDGE = (
    "[harness tool-error] The same call has failed three times with the same"
    " error; the error comes from the call's shape, not from the code."
)
TOOL_ERROR_ESCALATION = (
    "[harness tool-error] The identical error persists; the run ends at the eighth."
)

# A streak of ToolDenied refusals (approval policy, the git guard): the call
# was REFUSED, not malformed, so the generic "fix the call shape" text would
# be false and invite pointless reshuffling of the same command.
TOOL_DENIED_NUDGE = (
    "[harness tool-error] Refused by policy, not a failure: the same call gets"
    " the same refusal, and the refusal names what applies. Tools that need no"
    " approval, and finish_session, are available."
)


# A verify command that exited nonzero almost instantly with one of these
# signatures did not RUN the tests -- the runner itself is absent/broken.
# Treating that as a normal red misleads the model into "fixing" passing code
# or finishing on an unchecked patch.
_VERIFY_DEAD_SIGNATURES = (
    "no module named pytest",
    "no module named _pytest",
    "no module named nose",
    "command not found",
    "no such file or directory",
    "can't open file",
    "is not recognized as an internal or external command",
    "modulenotfounderror",
    "importerror while loading conftest",
)


def unrunnable_signature(argv: tuple[str, ...], rc: int | None, stdout: str, stderr: str) -> str:
    """Why an ADOPTED gate cannot run here, or "" for an ordinary red: exit
    127 (no such executable), or "No module named <mod>" naming exactly the
    module the adopted `-m <mod>` runs. A failing suite exits 1 or 2 with
    test output and matches neither."""
    if rc == 127:
        return "exit 127, the command is not found"
    if "-m" in argv:
        mod = argv[argv.index("-m") + 1] if argv.index("-m") + 1 < len(argv) else ""
        blob = f"{stdout}\n{stderr}".lower()
        if mod and f"no module named {mod.lower()}" in blob:
            return f"no module named {mod}"
    return ""


VERIFY_UNADOPTED_NOTICE = (
    "[harness] The adopted verify command `{cmd}` cannot run here ({why});"
    " the run is gateless again: run_verify_command is not available and"
    " the harness commits each editing step."
)

BASELINE_RED_NOTICE = (
    "[harness] That verify ran on an unmodified tree: the gate was already"
    " failing before your changes; those failures predate the task."
)

VERIFY_BROKEN_NUDGE = (
    "[harness verify-broken] Verify exited at once without running tests:"
    " the runner is missing or misconfigured, not a test failure. The"
    " project's own test command (setup.cfg, tox.ini, pyproject, bin/test)"
    " runs via run_command."
)


def verify_did_not_run(stdout_tail: str, stderr_tail: str, duration_s: float) -> bool:
    """True when a FAILED verify almost certainly did not execute any tests
    (the runner is absent), so the loop can flag it instead of passing the
    blind failure to the model. Requires a fast exit to avoid flagging a real
    suite that happens to import-error deep in a long run."""
    if duration_s > 3.0:
        return False
    blob = f"{stdout_tail}\n{stderr_tail}".lower()
    return any(sig in blob for sig in _VERIFY_DEAD_SIGNATURES)


def tool_error_signature(name: str, error_text: str) -> str:
    """Stable signature of a tool error, insensitive to varying numbers so a
    runaway that changes its args but trips the same error still matches."""
    return f"{name}:{re.sub(r'[0-9]+', '#', error_text)[:200]}"


NO_PROGRESS_NUDGE = (
    "[harness no-progress] Verify has failed four times with the same error;"
    " the edits so far have not changed the outcome."
)

NO_PROGRESS_ESCALATION = (
    "[harness no-progress] The identical failure persists; the run ends at"
    " the tenth. Earlier file content is readable (`git show HEAD:<path>`)"
    " and restorable with apply_edit."
)

_SIG_NOISE = re.compile(r"line \d+|0x[0-9a-fA-F]+|\d+\.\d+s\b|:\d+:|/tmp/\S+|\bin \d+(\.\d+)?s\b")


def verify_failure_signature(stdout_tail: str, stderr_tail: str) -> str:
    """Stable hash of a verify failure, insensitive to cosmetic drift."""
    tail = f"{stdout_tail}\n{stderr_tail}".strip()[-800:]
    digest = hashlib.md5(
        _SIG_NOISE.sub("#", tail).encode("utf-8", "replace"), usedforsecurity=False
    )
    return digest.hexdigest()


# Plan-mode wrap-up: nudge once the budget fraction drops below the threshold,
# or after this many iterations without having finished (or even started) a
# plan at all. A plan rarely needs more than a handful of reads.
PLAN_BUDGET_NUDGE_BELOW = 0.35
PLAN_NUDGE_AFTER_ITERS = 12

# Task finish-gate: when the worker has broken the run into subtasks, don't let
# it finish (or silently stop) while subtasks are still open -- re-prompt with
# the open list instead. A weak model on a long task tends to quit early with
# work pending.
# Capped so a worker that genuinely can't close a task (and won't mark it
# obsolete/skipped) can't bounce the loop forever; after the cap the finish is
# honoured. Only SUBTASKS gate -- the always-pending auto-root would deadlock.
TASK_FINISH_PATIENCE = 3

# Opt-in hard finish gate (`require_verify_to_finish`): how many times to bounce a
# finish_session over a red/stale verify before honouring it anyway (as an honest
# all_passed=False "finished"). Bounded so a task that genuinely can't pass can't
# pin the loop to the iteration cap.
VERIFY_FINISH_PATIENCE = 3
VERIFY_FINISH_GATE = (
    "[harness] finish_session refused: verify is not green"
    " (require_verify_to_finish). A green verify lifts the refusal; the third"
    " refusal is the last, after which a finish is honored and reported as"
    " finished, not passed."
)

# verify-settled completion (run mode). A non-metric run has no positive "done"
# signal, clean exit depends on the worker volunteering finish_session, and a weak
# worker keeps re-running read-only commands after success. Once verify has
# passed, count iterations that
# make no progress (no new commit + no edit): nudge to finish at the first
# threshold, hard-stop at the second. NOT "green verify = instant stop", verify
# fires per-edit and is often lenient, so green-but-still-editing must continue.
# Thresholds are deliberately generous: the failure mode is only a little wasted
# budget on an already-done run, whereas a too-tight window could cut off a
# worker still reading toward its next edit in a big multi-file change.
VERIFY_SETTLED_NUDGE_AFTER = 3
VERIFY_SETTLED_STOP_AFTER = 6

VERIFY_SETTLED_NUDGE = (
    "[harness settled] Changes are committed and the last three turns changed"
    " nothing; finish_session ends the run, and at six unchanged turns the"
    " run ends on its own."
)

# A non-metric `run` injects a one-shot wrap-up directive when the budget gets
# low: a worker that solves the task but never re-runs verify leaves the
# settled detector unable to engage (it needs a green verify) and burns the
# remainder on read-only commands.
RUN_BUDGET_NUDGE_BELOW = 0.25

RUN_BUDGET_NUDGE = (
    "[harness budget] Under a quarter of the budget remains; the loop halts"
    " when a cap is crossed. run_verify_command certifies the work and"
    " finish_session ends the run."
)

# Gateless variant (no verify command this run): there is nothing to verify, so
# steer straight to finish_session.
RUN_BUDGET_NUDGE_GATELESS = (
    "[harness budget] Under a quarter of the budget remains; the loop halts"
    " when a cap is crossed. finish_session ends the run."
)

# plan.md on disk is the plan; the planner's conversation only ever holds a
# copy. The operator answers open questions with `agent6 plan edit`, so the
# loop re-reads the file each turn and prepends this header when it differs
# from what the planner was last shown.
PLAN_ON_DISK_HEADER = (
    "[harness plan] plan.md on disk now reads as follows; it supersedes every"
    " earlier version in this conversation (operator edits: answers under"
    " `**A:**`, new constraints, deletions). The plan_markdown passed to"
    " finish_planning overwrites the file."
)

PLAN_BUDGET_NUDGE = (
    "[harness budget] finish_planning has not been called, and the pass is"
    " past its turn allowance or low on budget; the loop halts when a cap is"
    " crossed, and the plan exists only once finish_planning writes it."
)


# Silent finish before any work (run mode). Observed on SWE-bench with
# kimi-k2.7: the model answered the problem statement in PROSE at iteration
# 2 (a chat-tuned habit), no edit or verify had happened, and the loop
# accepted it as an implicit finish -- the whole run ended patchless with
# the budget unspent. An EARLY prose turn (first iterations) on an untouched
# tree is a stall, not a finish; steer back to the tools a bounded number of
# times. Later prose finishes stay honored: a run that read its fill and
# answers in prose is the legitimate implicit-finish path.
SILENT_NO_WORK_PATIENCE = 2
SILENT_NO_WORK_NUDGE = (
    "[harness] Prose with no tool call on an untouched tree is not a finish"
    " here; the tools do the work, and finish_session ends the run (its"
    " summary carries a blocker)."
)


QUESTION_NUDGE = (
    "[harness] A question in prose reaches nobody; ask_user reaches the"
    " operator, and finish_session ends the run."
)


# Cross-run memory write nudges. Measured (bench/longhorizon FINDINGS #2):
# 46 legs across 2 models produced ZERO unprompted memory writes, so the
# <memory> block alone never causes writes. Prompt at the two moments a
# durable discovery is actually in hand: the first red-to-green verify flip
# (advisory, free) and the first finish_session after such a recovery
# (deferred once, the backstop). Each fires at most once per run, only in run
# mode with a memory store wired, and only while the worker has recorded
# nothing; a run whose verify never failed is never nudged.
# "State the rule, not the instance": measured on orchard leg 3 (FINDINGS #2
# day 3) — a store that spelled the house convention in words transferred to
# a new computation; a store carrying only the formula it was first seen in
# did not.
MEMORY_FLIP_NUDGE = (
    "[harness memory] Verify flipped green and nothing is recorded in the"
    " memory dir this run; it takes a durable non-obvious repo fact as a"
    " general rule (a new <name>.md plus its MEMORY.md line)."
)

MEMORY_FINISH_NUDGE = (
    "[harness memory] finish_session deferred once: verify recovered earlier"
    " and nothing is recorded. The memory dir takes a durable non-obvious"
    " repo fact as a general rule; the next finish_session call is honored"
    " either way."
)


def ends_with_question(text: str) -> bool:
    """Best-effort: the model's prose ends by asking the operator something. The
    last non-empty line ending in '?' catches the common 'Should I proceed?' /
    'Which option do you want?' close that a model writes instead of calling
    ask_user."""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return bool(lines) and lines[-1].endswith("?")


def standing_fruitless_nudge(reason: str, task_id: str, title: str, streak: int) -> str:
    """The re-entry notice once a round landed nothing: same continuation,
    harder push -- repeat-what-you-did is the one wrong answer."""
    return (
        f"[harness] The run would have ended here ({reason}), and nothing has"
        f" landed since the last re-entry (fruitless round {streak}). The"
        f" standing task ({task_id}: {title}) continues: dig deeper or try a"
        " different approach -- a different angle, tool, or part of the repo;"
        " do not repeat the previous round. The run ends on its budget or an"
        " operator stop."
    )


def standing_resume_nudge(reason: str, task_id: str, title: str) -> str:
    """The soft-end conversion for a run with a standing task: instead of
    ending, the loop re-enters the standing goal with this notice."""
    return (
        f"[harness] The run would have ended here ({reason}), but the standing"
        f" task ({task_id}: {title}) continues. Re-enter it now: pick the next"
        " piece of that goal, insert any new work you discover with add_task"
        " (ordinary tasks always run first), and write decisions down rather"
        " than asking questions. The run ends on its budget or an operator"
        " stop."
    )
