# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The `agent6 ask` Q&A flow: listing past asks, building a run
digest for context, the interactive ask REPL, and saving ask transcripts.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from agent6.budget import BudgetTracker
from agent6.errors import read_operator_file
from agent6.git_ops import DIFF_SHOW_SAFETY_FLAGS, branch_tip_sha, git_hardening_flags
from agent6.paths import state_dir
from agent6.sessions.id import SessionIdError, resolve_session
from agent6.sessions.layout import SessionLayout, bucket_dir
from agent6.sessions.manifest import ManifestError, SessionManifest, read_manifest
from agent6.ui.cli._common import error, warn
from agent6.ui.cli._steer import repl_prompt_sigint
from agent6.viewmodel import newest_session_dir
from agent6.workflows.loop import (
    SessionResult,
    Workflow,
)


def summarize_session_log(logs_path: Path) -> str:
    """Compact prose summary of a run's logs.jsonl: outcome + event counts +
    recent notable events. Used to seed `agent6 ask --from`."""
    if not logs_path.is_file():
        return "(no logs.jsonl for this run)"
    events: list[dict[str, Any]] = []
    for line in logs_path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            events.append(obj)
    if not events:
        return "(empty log)"
    counts: dict[str, int] = {}
    for e in events:
        counts[str(e.get("type", ""))] = counts.get(str(e.get("type", "")), 0) + 1
    out: list[str] = []
    end = next((e for e in reversed(events) if e.get("type") == "session.end"), None)
    if end is not None:
        out.append(f"Ended: reason={end.get('reason')!r} iterations={end.get('iterations')}")
    top = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:8]
    out.append("Event counts: " + ", ".join(f"{t}={n}" for t, n in top))
    notable_types = {
        "tool.call",
        "verify.end",
        "session.end",
        "loop.auto_commit",
        "loop.metric.sample",
    }
    notable = [e for e in events if e.get("type") in notable_types][-15:]
    if notable:
        out.append("Recent notable events:")
        out.extend(f"  - {fmt_run_event(e)}" for e in notable)
    return "\n".join(out)


def fmt_run_event(e: dict[str, Any]) -> str:
    """One-line summary of a logs.jsonl event for the ask `--from` digest."""
    t = str(e.get("type", ""))
    if t == "tool.call":
        return f"tool.call {e.get('name', '')} {str(e.get('args', ''))[:80]}".rstrip()
    if t == "verify.end":
        return f"verify.end exit={e.get('exit_code')}"
    if t == "session.end":
        return f"session.end reason={e.get('reason')}"
    if t == "loop.metric.sample":
        return f"loop.metric.sample score={e.get('score')}"
    return t


def _git_diff_text(cwd: Path, range_spec: str) -> tuple[int, str, str]:
    """Hardened `git diff <range>`, bytes-captured and lossy-decoded: the old
    `text=True` strict decode raised UnicodeDecodeError out of communicate()
    on a valid non-UTF-8 diff (a latin-1 file), crashing `ask --from`."""
    # operator-controlled argv, no LLM input (same as `agent6 sessions diff`).
    # Hardening flags: a poisoned .git/config diff.external or diff.*.textconv
    # would otherwise run on the host when the operator asks about a prior run.
    proc = subprocess.run(
        ["git", *git_hardening_flags(cwd), "diff", *DIFF_SHOW_SAFETY_FLAGS, range_spec],
        cwd=cwd,
        capture_output=True,
        check=False,
    )
    return (
        proc.returncode,
        proc.stdout.decode(errors="replace"),
        proc.stderr.decode(errors="replace"),
    )


def _diff_via_merge_stamp(
    cwd: Path, manifest: SessionManifest, base_sha: str, run_branch: str | None
) -> tuple[str, int, str, str] | None:
    """(label, rc, diff, err) via the manifest's merge stamp, for a primary
    range that is unreachable -- usually a run branch pruned after its merge,
    but a gc'd base_sha does it with the branch still there. None without a
    stamp. The label names which it was, since the model may go read the
    branch."""
    merged = manifest.merged
    merged_sha = merged.sha if merged else ""
    if not (run_branch and merged_sha):
        return None
    gone = branch_tip_sha(cwd, run_branch) is None
    why = "run branch pruned" if gone else "base unreachable"
    is_ff = merged is not None and merged_sha == merged.tip
    if is_ff:
        # Fast-forwarded: the stamped commit IS the run's tip, so its ^.. diff
        # is the last commit only. The full run is base..merged, both in the
        # base branch's history.
        label = f"{base_sha[:12]}..{merged_sha[:12]} ({why}; fast-forward merge)"
        rc, diff, err = _git_diff_text(cwd, f"{base_sha}..{merged_sha}")
        if rc == 0:
            return label, rc, diff, err
    partial = "; last run commit only" if is_ff else ""
    label = f"{merged_sha[:12]}^..{merged_sha[:12]} ({why}; merge commit{partial})"
    return label, *_git_diff_text(cwd, f"{merged_sha}^..{merged_sha}")


def build_ask_session_digest(cwd: Path, session_id: str, *, latest: bool) -> str | None:
    """Markdown digest of a prior SESSION to seed a new one, or None (after
    printing an error) when it can't be resolved.

    Any session kind seeds any other: a run, a plan and an ask all record the
    same shape, and the useful direction is whichever way the operator is
    working -- an ask that worked something out, then a run to do it.
    """
    state = state_dir(cwd)
    if latest:
        # runs/ and asks/ only: a machine draft is an authoring log, not a
        # session with a task and an outcome, and picking the newest one made
        # `--from-latest` fail on a project that had just written a machine.
        newest = newest_session_dir([bucket_dir(state, "runs"), bucket_dir(state, "asks")])
        if newest is None:
            error(f"--from-latest: no run or ask under {state}")
            return None
        session_id = newest.name
    try:
        layout = resolve_session(state, session_id)
    except SessionIdError as exc:
        error(f"{exc}")
        return None
    target = layout.session_id
    if not layout.manifest_path.is_file():
        error(f"run {target} has no manifest.json")
        return None
    try:
        manifest = read_manifest(layout.session_dir)
    except ManifestError as exc:
        error(f"could not read manifest for {target}: {exc}")
        return None
    base_sha = manifest.base_sha
    run_branch = manifest.run_branch
    diff_label = f"{base_sha}..{run_branch}"
    diff_body = "(no diff: the run recorded no base_sha)"
    if not run_branch:
        # A plan and an ask cut no branch and commit nothing. Diffing HEAD
        # instead handed the model whatever the operator happened to have
        # uncommitted, labelled as the session's work.
        diff_label = "(none)"
        diff_body = "(no diff: this session wrote no code)"
    elif base_sha:
        rc, diff, err = _git_diff_text(cwd, f"{base_sha}..{run_branch}")
        if rc != 0:
            fallback = _diff_via_merge_stamp(cwd, manifest, base_sha, run_branch)
            if fallback is not None:
                diff_label, rc, diff, err = fallback
        if rc != 0:
            # Loud, never an empty diff block the model reads as "no changes".
            diff_body = f"(diff unavailable: git diff exited {rc}: {err.strip()[:300]})"
        else:
            cap = 8000
            tail = "\n... (diff truncated; read more with git)" if len(diff) > cap else ""
            diff_body = f"```diff\n{diff[:cap]}{tail}\n```"
    plan_path = layout.session_dir / "plan.md"
    plan_section = f"\n## Plan\n{read_operator_file(plan_path)}\n" if plan_path.is_file() else ""
    return (
        f'<prior-run id="{target}">\n'
        "This question is about a PRIOR agent6 run. Its run state lives outside the"
        " workspace and is not reachable with read_file, so everything you have"
        " about it is in this digest.\n\n"
        f"## Run task\n{manifest.user_task}\n\n"
        f"## Outcome / key events\n{summarize_session_log(layout.logs_path)}\n\n"
        f"## Diff {diff_label}\n{diff_body}\n"
        f"{plan_section}"
        f"</prior-run>"
    )


def seed_files(cwd: Path, files: list[str]) -> str:
    """Wrap explicit --file seeds for an `ask` (a non-fatal, capped read)."""
    parts: list[str] = []
    for f in files:
        try:
            content = (cwd / f).read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            warn(f"--file {f}: {exc}")
            continue
        cap = 64 * 1024
        if len(content) > cap:
            content = content[:cap] + "\n... (truncated)"
        parts.append(f'<file path="{f}">\n{content}\n</file>')
    return "\n".join(parts)


def save_ask_transcript(layout: SessionLayout, *, question: str, answer: str) -> None:
    """Write the human-readable `ask` transcript (question + markdown answer).

    A resumed ask appends its own Q&A: the file exists only because an earlier
    leg wrote it, and overwriting would drop the answer the operator already
    has. Both halves are appended -- a bare second answer under the FIRST
    question read as a continuation of an answer to something else.
    """
    out = layout.session_dir / "transcript.md"
    if out.is_file():
        with out.open("a", encoding="utf-8") as fh:
            fh.write(f"\n## Question (continued)\n\n{question}\n\n## Answer\n\n{answer}\n")
        return
    out.write_text(
        f"# agent6 ask\n\n## Question\n\n{question}\n\n## Answer\n\n{answer}\n",
        encoding="utf-8",
    )


def save_ask_repl_transcript(layout: SessionLayout, conversation: list[tuple[str, str]]) -> None:
    """Write the cumulative transcript for an interactive ask session."""
    parts = ["# agent6 ask (interactive)\n"]
    for i, (q, a) in enumerate(conversation, 1):
        parts.append(f"## Q{i}\n\n{q}\n\n## A{i}\n\n{a}\n")
    (layout.session_dir / "transcript.md").write_text("\n".join(parts), encoding="utf-8")


def run_ask_repl(
    wf: Workflow, budget: BudgetTracker, layout: SessionLayout, *, first_question: str
) -> SessionResult:
    """Interactive multi-turn ask. Each follow-up re-enters the loop with the
    prior Q&A carried as context, reusing the one provider/jail/budget setup.
    The agent re-reads what it needs per turn (prompt-cached); the conversation
    text is what gives continuity."""
    print(
        "[agent6] ask REPL: type a follow-up, or /cost /reset /quit (Ctrl-D exits).",
        file=sys.stderr,
    )
    conversation: list[tuple[str, str]] = []
    pending = first_question.strip()
    result: SessionResult | None = None
    while True:
        if pending:
            question = pending
            pending = ""
        else:
            try:
                with repl_prompt_sigint():
                    question = input("\nask> ").strip()
            except (EOFError, KeyboardInterrupt):
                print(file=sys.stderr)
                break
        if not question:
            continue
        if question in ("/quit", "/q", "/exit"):
            break
        if question == "/cost":
            print(budget.format_summary(), file=sys.stderr)
            continue
        if question == "/reset":
            conversation = []
            print("[agent6] conversation reset.", file=sys.stderr)
            continue
        if conversation:
            ctx = "\n\n".join(f"Q: {q}\nA: {a}" for q, a in conversation)
            augmented = (
                f"<conversation-so-far>\n{ctx}\n</conversation-so-far>\n\nFollow-up: {question}"
            )
        else:
            augmented = question
        result = wf.run(augmented)
        print(result.summary, flush=True)
        conversation.append((question, result.summary))
        save_ask_repl_transcript(layout, conversation)
        if budget.is_exhausted():
            print("[agent6] budget exhausted; ending the REPL.", file=sys.stderr)
            break
    if result is None:
        return SessionResult(
            completed=True, reason="ask_repl_empty", summary="", iterations=0, tool_calls=0
        )
    return result
