# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""`agent6 machine create`: draft a `.asm.toml` + its scripts from a natural-
language task, validate each attempt (structural + bundle + lint + offline
tests + dry-run), and write the first fully-valid draft.

Authoring runs the same confined agent subprocess a running machine's `agent`
state uses (`build_machine_agent_runner`), in `mode="machine"` with a
finish_session-focused prompt. Output goes through the injected `MachineFrontend`
reporter; the watchable per-draft event log is a separate `EventSink`.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from agent6.app._session import select_isolation
from agent6.app._setup import check_provider_keys
from agent6.app.machine._bundle import validate_bundle
from agent6.app.machine._frontend import MachineFrontend
from agent6.app.machine._scriptcheck import lint_and_typecheck, run_offline_tests
from agent6.app.machine_agent import build_machine_agent_runner
from agent6.app.preflight import SessionRefused
from agent6.config import ConfigError
from agent6.config.layer import load_effective, resolved_state_dir
from agent6.events import EventSink
from agent6.machine import (
    SCRIPTS_PAYLOAD_KEY,
    TOML_PAYLOAD_KEY,
    AgentRequest,
    FieldSpec,
    MachineError,
    MachineSpec,
    build_authoring_prompt,
    dry_run,
    extract_scripts,
    extract_toml,
    load_machine,
)
from agent6.portable import atomic_write
from agent6.sessions.id import unused_session_id
from agent6.sessions.ipc import emit_session_start
from agent6.sessions.layout import LOGS_NAME, bucket_dir
from agent6.types import session_bucket

_CREATE_TIMEOUT_S = 900.0


_CREATE_STOP_REASONS = frozenset(
    {"budget_exhausted", "timeout", "provider_error", "prompt_revision_failed"}
)


def _write_scripts(base_dir: Path, scripts: dict[str, str]) -> None:
    """Write the bundle's helper scripts (keys are bundle-relative, already
    validated by extract_scripts to live under scripts/ with no `..`).

    Defense-in-depth: unlink a pre-existing symlink at the target before writing
    so a planted `scripts/<name>` -> elsewhere link can't redirect the write out
    of the bundle. `validate_bundle` (run by check/run before any execution) is
    the backstop for symlinks anywhere in the tree."""
    for rel, content in scripts.items():
        p = base_dir / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        if p.is_symlink():
            p.unlink()
        p.write_text(content if content.endswith("\n") else content + "\n", encoding="utf-8")


def _reread_scripts(base_dir: Path, scripts: dict[str, str]) -> dict[str, str]:
    """The same bundle keys, re-read from disk after ruff's safe fixes."""
    return {rel: (base_dir / rel).read_text(encoding="utf-8") for rel in scripts}


def _check_machine_text(
    text: str, scripts: dict[str, str], scratch: Path
) -> tuple[MachineSpec | None, list[str]]:
    """Validate a candidate `.asm.toml` + its scripts via `load_machine`.

    The scripts are written into the scratch bundle first so the missing-script
    check resolves against this attempt's files only (stale scripts from a prior
    attempt are cleared). Returns the parsed spec + empty problems on success, or
    `(None, problems)` when the source or its script bundle is invalid.
    """
    candidate_path = scratch / "candidate.asm.toml"
    candidate_path.write_text(text, encoding="utf-8")
    shutil.rmtree(scratch / "scripts", ignore_errors=True)
    _write_scripts(scratch, scripts)
    try:
        spec = load_machine(candidate_path)
    except MachineError as exc:
        return None, list(exc.problems)
    bundle_problems = validate_bundle(spec, candidate_path)
    if bundle_problems:
        return None, bundle_problems
    return spec, []


def _attempt_reason(problems: list[str]) -> str:
    """A one-line summary of why an attempt failed: the first problem's first
    line plus, when that line only introduces a block (ends with ':'), the
    block's last line (a test dump or traceback ends with the actual error;
    'offline test x failed (exit 1):' alone explains nothing). A count of
    the rest follows. Keeps the per-attempt log to a line while the full
    diagnostics still feed back into the next prompt."""
    first = problems[0] if problems and problems[0].strip() else ""
    lines = [ln.strip() for ln in first.splitlines() if ln.strip()]
    head = lines[0] if lines else "unknown"
    if head.endswith(":") and len(lines) > 1:
        # Clip the intro first: a long one truncates away the appended
        # error, the one part this line exists to surface.
        if len(head) > 100:
            head = head[:97] + "..."
        head = f"{head} {lines[-1]}"
    if len(head) > 160:
        head = head[:157] + "..."
    extra = f" (+{len(problems) - 1} more)" if len(problems) > 1 else ""
    return f"{head}{extra}"


def create_machine(  # noqa: PLR0911, PLR0912, PLR0915
    task: str,
    frontend: MachineFrontend,
    *,
    output: Path | None,
    max_attempts: int,
    config_path: Path | None = None,
) -> int:
    reporter = frontend.reporter
    if max_attempts < 1:
        reporter.error("--max-attempts must be >= 1.")
        return 2
    cwd = Path.cwd()
    try:
        eff = load_effective(cwd, config_path)
        cfg = eff.config
        cfg.require_runnable("worker")
    except ConfigError as exc:
        reporter.error(str(exc))
        return 2
    missing = check_provider_keys(cfg)
    if missing is not None:
        reporter.err(missing)
        return 2
    try:
        # The run lifecycle's own preflight; a create session withholds the
        # command tools, so the unconfined-autorun confirmation has nothing to
        # confirm.
        isolation = select_isolation(
            cfg,
            confirm_unconfined=lambda _isolation, _cfg: True,
            reporter=reporter,
            explicit_leaves=eff.explicit_leaves,
        )
    except SessionRefused as refusal:
        return refusal.rc

    state_dir = resolved_state_dir(cwd)
    bucket = session_bucket("machine")
    # Through the owner: `mkdir(exist_ok=True)` on a collision reused a live
    # draft's directory, overwriting its prompt and appending to its journal.
    scratch = bucket_dir(state_dir, bucket) / unused_session_id(state_dir, bucket)
    scratch.mkdir(parents=True)
    # Persist the natural-language task that drove this draft, so the draft dir is
    # self-describing (the agent_transcripts/ embed it inside the authoring prompt,
    # but a plain prompt.txt is what a human looks for).
    (scratch / "prompt.txt").write_text(task, encoding="utf-8")
    # A watchable event log for the draft: the TUI opens the dashboard on this dir
    # and follows the authoring agent live. The parent owns the session.start header
    # (the NL task) + the per-attempt markers + the final session.end; each attempt's
    # subprocess appends its own role.*_delta / tool.* events to the same file.
    events_log = scratch / LOGS_NAME
    events = EventSink(events_log)
    # Liveness marker, mirroring machine run: the draft dir is watchable (the
    # hub lists it, the SSE endpoints stream it). A terminal draft has its own
    # session.end, which the status decision reads first, so the marker needs
    # no clearing.
    emit_session_start(events, scratch, "session.start", user_task=task, mode="machine")
    # Authoring can take minutes with nothing on this terminal; say where the
    # live reasoning streams so the operator can follow instead of wondering.
    reporter.err(
        f"machine create: drafting as {scratch.name} (follow live: agent6 attach {scratch.name})"
    )
    # Authoring drafts a machine; it has no machine [config] overlay of its own.
    runner = build_machine_agent_runner({}, cwd, isolation, scratch / "agent_transcripts")

    prior_toml: str | None = None
    prior_scripts: dict[str, str] = {}
    diagnostics: list[str] | None = None
    spec: MachineSpec | None = None
    valid_toml: str | None = None
    valid_scripts: dict[str, str] = {}
    total_usd = 0.0
    total_in = 0
    total_out = 0
    # One ledger across attempts against the operator's budget: each attempt's
    # subprocess is otherwise a fresh tracker, so N retries could bill N full
    # budgets. -1 = unlimited (the config's own convention).
    create_cap = None if cfg.budget.max_usd == -1 else cfg.budget.max_usd
    # The authoring finish contract, leg-enforced like any machine state's:
    # a finish_session without result.toml bounces in-run with the problem
    # named, instead of ending the attempt for the outer loop to diagnose
    # (kimi returned summary-only finishes three times in a row against the
    # prose instruction alone).
    draft_schemas = {
        "draft": {
            TOML_PAYLOAD_KEY: FieldSpec(type="str"),
            SCRIPTS_PAYLOAD_KEY: FieldSpec(type="json", optional=True),
        }
    }
    attempt = 0  # bound for the session.end below (the loop always runs: max_attempts >= 1)
    for attempt in range(1, max_attempts + 1):
        if create_cap is not None and total_usd >= create_cap:
            reporter.err(
                f"machine create: budget max_usd (${create_cap}) exhausted after"
                f" {attempt - 1} attempt(s) (spent ~${total_usd:.4f}); stopping."
            )
            break
        prompt = build_authoring_prompt(
            task,
            attempt=attempt,
            prior_toml=prior_toml,
            diagnostics=diagnostics,
            prior_scripts=prior_scripts,
        )
        reporter.err(f"machine create: attempt {attempt}/{max_attempts}...")
        events.emit("loop.note", text=f"attempt {attempt}/{max_attempts}")
        # model omitted (=None): inherit the operator's effective worker model.
        # mode="machine": authoring system prompt + read-only tools (see loop.py).
        # effort="off": authoring is transcription of a described design, not
        # deep derivation. "low" is already the provider default and did not
        # help: kimi-k2.6 spiralled into 30-minute length-capped thinks and
        # timed out on every attempt (0/3 drafts across two spec sizes). With
        # the reasoning channel off it drafted in ~2.5 minutes for $0.02.
        remaining = None if create_cap is None else max(create_cap - total_usd, 0.0)
        result = runner(
            AgentRequest(
                prompt=prompt,
                timeout_s=_CREATE_TIMEOUT_S,
                mode="machine",
                effort="off",
                max_usd=remaining,
                output_schema="draft",
                schemas=draft_schemas,
            ),
            events_log,
        )
        total_usd += result.usd
        total_in += result.input_tokens
        total_out += result.output_tokens
        candidate = extract_toml(result.payload)
        if candidate is None:
            diagnostics = [
                f"You did not return a draft: call finish_session with result.{TOML_PAYLOAD_KEY}"
                " set to the complete .asm.toml source as a single string."
                f" (agent loop reason: {result.reason})"
            ]
            prior_toml = None
            prior_scripts = {}
            reporter.err(
                f"machine create: attempt {attempt} failed:"
                f" returned no draft (agent stop reason: {result.reason})"
            )
            if result.reason in _CREATE_STOP_REASONS:
                break
            continue
        candidate_scripts = extract_scripts(result.payload)
        candidate_spec, problems = _check_machine_text(candidate, candidate_scripts, scratch)
        if candidate_spec is None:
            # Structural / bundle failure. A missing-script problem (only produced
            # here, never by the lint/test pass below) gets an extra hint pointing
            # the agent at result.scripts.
            if any("not found in bundle" in p for p in problems):
                hint = (
                    f"Return each missing scripts/... file in finish_session"
                    f" result.{SCRIPTS_PAYLOAD_KEY} (a map of the path to its complete source)."
                )
                problems = [*problems, hint]
            if candidate_scripts:
                # The scripts exist whether or not the TOML parsed: lint them
                # now, so one attempt reveals every problem class instead of
                # schema-then-lint costing an attempt each.
                problems = [
                    *problems,
                    *lint_and_typecheck(
                        scratch / "scripts",
                        fix=True,
                        ruff_config_from=output.parent if output is not None else cwd,
                    ),
                ]
        else:
            # Structurally valid. Now make it production-ready: lint + type-check
            # the scripts, run their offline `*_test.py` mocks in a jail, and
            # dry-run the routing (synthesized facts through the real reducer;
            # catches e.g. a branch reading a field the schema doesn't declare).
            # Any failure becomes a retry diagnostic so the agent fixes it itself.
            reporter.err("machine create: linting + offline-testing scripts...")
            events.emit("loop.note", text="linting + offline-testing the draft")
            problems = lint_and_typecheck(
                scratch / "scripts",
                fix=True,
                # The scratch lives under the state dir, where ruff's discovery
                # finds no config; resolve from where the bundle publishes so
                # this gate and the operator's later `machine check` agree.
                ruff_config_from=output.parent if output is not None else cwd,
            )
            # ruff --fix rewrote the scratch copies, so those are the bytes that
            # passed. Publishing the model's originals writes a bundle that
            # fails the `machine check` this command sends the operator to.
            candidate_scripts = _reread_scripts(scratch, candidate_scripts)
            offline = run_offline_tests(scratch, isolation)
            problems.extend(offline.problems)
            if offline.skipped:
                reporter.err(
                    f"machine create: {offline.skipped} offline script test(s) NOT run"
                    f" ({offline.skip_reason}); static checks still applied"
                )
            report = dry_run(candidate_spec, None)
            problems.extend(
                f"dry-run state {c.name!r}: {c.detail}"
                for c in (*report.states, *report.branches)
                if not c.ok
            )
            if not problems:
                spec = candidate_spec
                valid_toml = candidate
                valid_scripts = candidate_scripts
                break
        # Reached only on a failed attempt (the success path broke above), so
        # surface a one-line reason instead of leaving "attempt N/M" unexplained.
        reporter.err(f"machine create: attempt {attempt} failed: {_attempt_reason(problems)}")
        prior_toml = candidate
        prior_scripts = candidate_scripts
        diagnostics = problems
        if result.reason in _CREATE_STOP_REASONS:
            break

    reporter.err(f"machine create: spent ~${total_usd:.4f}")
    # Each attempt's subprocess logs its OWN budget.update (resetting to that
    # attempt's spend), so the fold's last one shows only the last attempt. Emit
    # the true cumulative total across attempts so the watchable draft's cost is
    # the real spend, not the last retry's slice.
    events.emit(
        "budget.update",
        usd_total=total_usd,
        input_total=total_in,
        output_total=total_out,
    )
    # session.end reasons below are tokens, like every other emitter's: the listing
    # prints the reason as the detail beside "failed", where a prose sentence
    # contradicted the word. `iterations` = authoring attempts made.
    if spec is None or valid_toml is None:
        events.emit("session.end", reason="no_valid_machine", iterations=attempt, all_passed=False)
        reporter.err(f"FAILED: no valid machine after {max_attempts} attempt(s).")
        if diagnostics:
            reporter.err("Last diagnostics:")
            for problem in diagnostics:
                reporter.err(f"  - {problem}")
        if prior_toml is not None:
            reporter.err("The last (invalid) draft is on stdout for reference.")
            # reporter.out re-adds one trailing newline, so strip one to match the
            # original `print(..., end="")` byte-for-byte.
            draft = prior_toml if prior_toml.endswith("\n") else prior_toml + "\n"
            reporter.out(draft.removesuffix("\n"))
        return 1

    payload = valid_toml if valid_toml.endswith("\n") else valid_toml + "\n"
    target = output if output is not None else cwd / f"{spec.machine}.asm.toml"
    if output is None:
        # The default path is documented as clobbering nothing; that covers the
        # WHOLE bundle, not just the machine file -- an LLM-chosen script name
        # colliding with an operator's existing scripts/<name> would otherwise
        # be silently replaced (unrecoverable if uncommitted). `-o` keeps its
        # documented overwrite-freely contract.
        clashes = [
            p for p in (target, *(target.parent / rel for rel in valid_scripts)) if p.exists()
        ]
        if clashes:
            # The refusal fails the command with nothing written; session.end must
            # say so, not machine_created.
            events.emit(
                "session.end", reason="output_collision", iterations=attempt, all_passed=False
            )
            reporter.err("REFUSING to overwrite existing file(s):")
            for clash in clashes:
                reporter.err(f"  {clash}")
            reporter.err("The validated draft is on stdout; redirect it or re-run with -o <file>.")
            reporter.out(payload.removesuffix("\n"))
            return 2
    # The writes decide the outcome, so session.end waits for them.
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        # Scripts first, the machine file last and atomically: the .asm is the
        # bundle's commit point, so a death mid-publish leaves inert scripts,
        # never a machine whose scripts are missing.
        _write_scripts(target.parent, valid_scripts)
        atomic_write(target, payload)
    except OSError as exc:
        events.emit("session.end", reason="write_failed", iterations=attempt, all_passed=False)
        reporter.err(f"FAILED: could not write the bundle to {target.parent}: {exc}")
        reporter.err("The validated draft is on stdout; redirect it or re-run with -o <file>.")
        reporter.out(payload.removesuffix("\n"))
        return 1
    # The destination can differ from the validated scratch copy (e.g. a
    # pre-existing symlink under scripts/), so the structural check on what was
    # PUBLISHED decides the outcome: a success banner over a bundle that won't
    # run was a lie. Lint/types are not re-run: these bytes ARE the scratch copy
    # that passed (_reread_scripts picks up ruff's fixes before publishing).
    out_problems = validate_bundle(spec, target)
    if out_problems:
        events.emit("session.end", reason="bundle_invalid", iterations=attempt, all_passed=False)
        reporter.err(f"FAILED: the bundle written to {target.parent} does not validate:")
        for problem in out_problems:
            reporter.err(f"  - {problem}")
        return 1
    # End the watchable session; all_passed marks a valid machine authored,
    # written, and validated in place.
    events.emit("session.end", reason="machine_created", iterations=attempt, all_passed=True)
    scripts_note = f" + {len(valid_scripts)} script(s)" if valid_scripts else ""
    reporter.err(
        f"OK: wrote draft to {target} ({spec.machine}, {len(spec.states)} states){scripts_note}."
    )
    reporter.err("Review and commit it; `machine run` only accepts committed machines.")
    return 0
