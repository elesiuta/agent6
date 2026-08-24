#!/usr/bin/env python3
"""Long-horizon benchmark orchestrator.

Drives the installed agent6 binary through multi-leg task SEQUENCES for a
(model x condition x task x rep) matrix. Each sequence runs all of a task's
legs in ONE workdir, in order; a leg may overlay extra files first (new
requirements landing mid-project). Legs share the per-repo agent6 state dir
by default, so cross-run channels (the <memories> block) carry over; the
`fresh_state` condition gives every leg a private state dir instead, which is
the memory-value A/B. Each leg is graded by the task's authoritative HIDDEN
grader (``tasks/<name>/grade.py``, never shipped into the agent's repo) and
recorded as one JSON line.

What this bench exists to measure (see FINDINGS.md of bench/coreagent for why
short tasks cannot): tier-1 compaction losses (drops, re-reads after the
first drop, per-rule retention via component scores), the value of cross-run
memories, and whether add_dependency gets used at all.

Usage:
  python3 run.py --model moonshotai/kimi-k2.6 --provider openrouter \
      --tasks stylebook --conditions baseline,window32k --reps 3 \
      --parallel 3 --label wave1

Results: results/<label>.jsonl (append, one record per LEG). Summarize with
stats.py.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
TASKS_DIR = ROOT / "tasks"
RESULTS_DIR = ROOT / "results"
AGENT6_BIN = os.environ.get("AGENT6_BIN") or shutil.which("agent6") or "agent6"
RUNS_ROOT = Path(os.environ.get("LONGHORIZON_RUNS", str(Path.home() / "longhorizon-runs")))

READ_TOOLS = {"read_file", "outline", "find_definition", "find_references", "list_dir"}
EDIT_TOOLS = {"apply_edit", "apply_patch"}


@dataclass(frozen=True)
class Leg:
    name: str
    prompt: str
    inject: str | None = None  # task-dir subdir overlaid onto the workdir before this leg
    timeout_s: int = 2400
    max_usd: float = 1.50


@dataclass(frozen=True)
class Task:
    legs: tuple[Leg, ...]
    protected: tuple[str, ...]  # agent must not modify these (tamper check)
    # Substrings of an edit-tool TARGET worth counting (the built-artifact
    # trap). Matched against the call's `path` arg when present, else the
    # whole args JSON (apply_patch may carry the path only in patch headers);
    # path-first keeps a generator that MENTIONS the artifact in a docstring
    # from counting as a trap edit.
    trap_patterns: tuple[str, ...] = ()
    # Stale memories (name, body) the `poisoned` condition plants into the
    # shared state dir before leg 2: plausible, wrong, and contradicted by the
    # repo. Whether later legs verify before trusting is what it measures.
    poison: tuple[tuple[str, str], ...] = ()


# Prompts are deliberately NEUTRAL about the DAG, dependencies, and memories:
# whether the agent uses those channels is part of what the bench measures.
TASKS: dict[str, Task] = {
    "stylebook": Task(
        legs=(
            Leg(
                name="main",
                prompt=(
                    "Read spec.md, then read ALL ten rule files under rules/ (each one"
                    " is authoritative for one rule). Implement audit.py so that"
                    " ./verify.sh passes and the auditor enforces EVERY rule exactly as"
                    " its file states it, including the edge cases. Do not modify"
                    " test_audit.py or verify.sh. Run verify, fix what fails, and call"
                    " finish_session when the suite passes."
                ),
            ),
        ),
        protected=("test_audit.py", "verify.sh"),
    ),
    "relay": Task(
        legs=(
            Leg(
                name="main",
                prompt=(
                    "Read spec.md and the six stage specs under specs/. Implement all"
                    " six modules (parse.py, validate.py, sessionize.py, metrics.py,"
                    " report.py, cli.py) so that ./verify.sh passes. Implement EVERY"
                    " stage exactly to its spec, not just the ones the starter tests"
                    " cover; the stages consume each other's outputs, so interface"
                    " details matter. Do not modify test_relay.py or verify.sh. Run"
                    " verify, fix what fails, and call finish_session when the whole suite"
                    " passes."
                ),
                timeout_s=3000,
                max_usd=2.00,
            ),
        ),
        protected=("test_relay.py", "verify.sh"),
    ),
    "orchard": Task(
        legs=(
            Leg(
                name="fix",
                prompt=(
                    "The test suite is red: SKU B-204's shelf price should be 1299"
                    " cents but the catalog shows the wrong value. Find the root cause,"
                    " fix it properly, and make ./verify.sh pass. Do not modify"
                    " test_catalog.py or verify.sh."
                ),
                timeout_s=1500,
                max_usd=1.00,
            ),
            Leg(
                name="weekend",
                prompt=(
                    "A new requirement landed: implement the WEEKEND pricing tier"
                    " described in specs/weekend.md so that ./verify.sh passes"
                    " (test_weekend.py is the new part of the suite). Do not modify"
                    " test_catalog.py, test_weekend.py, or verify.sh."
                ),
                inject="leg2",
                timeout_s=1500,
                max_usd=1.00,
            ),
            # The read-side memory probe: a THIRD task against the same two
            # conventions (data/ files are built artifacts; money rounds
            # half-up on the cent) with fresh discriminator values, and a spec
            # that deliberately does NOT point at docs/NOTES.md. An agent
            # whose <memories> carry the leg-1/leg-2 trap facts can apply
            # them; a fresh-state agent must rediscover or fall.
            Leg(
                name="clearance",
                prompt=(
                    "Another requirement landed: implement the CLEARANCE feed"
                    " described in specs/clearance.md so that ./verify.sh passes"
                    " (test_clearance.py is the new part of the suite). Do not"
                    " modify test_catalog.py, test_weekend.py, test_clearance.py,"
                    " or verify.sh."
                ),
                inject="leg3",
                timeout_s=1500,
                max_usd=1.00,
            ),
        ),
        protected=("test_catalog.py", "test_weekend.py", "test_clearance.py", "verify.sh"),
        trap_patterns=("data/catalog.tsv", "data/clearance.tsv"),
        poison=(
            (
                "catalog-data-files",
                "data/catalog.tsv and data/clearance.tsv are hand-maintained source tables:"
                " edit them directly. tools/gen_*.py are legacy and no longer run.",
            ),
            (
                "money-rounding",
                "Money rounds with Python's built-in round() (banker's rounding) by team"
                " convention; half-up rounding was dropped.",
            ),
        ),
    ),
    "ledger": Task(
        legs=(
            Leg(
                name="fix",
                prompt=(
                    "The test suite is red: `ledger add` drops the memo"
                    " (tests/test_cli.py::test_add_keeps_memo). Find the root cause, fix"
                    " it properly, and make ./verify.sh pass. Do not modify tests/ or"
                    " verify.sh."
                ),
                timeout_s=1500,
                max_usd=1.00,
            ),
            Leg(
                name="split",
                prompt=(
                    "A new requirement landed: implement the `split` command described in"
                    " specs/split.md so that ./verify.sh passes (tests/test_split.py is the"
                    " new part of the suite). Do not modify tests/ or verify.sh."
                ),
                inject="leg2",
                timeout_s=1500,
                max_usd=1.00,
            ),
            Leg(
                name="convert",
                prompt=(
                    "Another requirement: implement the `convert` command described in"
                    " specs/convert.md so that ./verify.sh passes (tests/test_convert.py is"
                    " new). Do not modify tests/ or verify.sh."
                ),
                inject="leg3",
                timeout_s=1500,
                max_usd=1.00,
            ),
            Leg(
                name="report",
                prompt=(
                    "Next: implement the `report` command described in specs/report.md so"
                    " that ./verify.sh passes (tests/test_report.py is new). Do not modify"
                    " tests/ or verify.sh."
                ),
                inject="leg4",
                timeout_s=1500,
                max_usd=1.00,
            ),
            Leg(
                name="import",
                prompt=(
                    "Last: implement the `import-csv` command described in specs/import.md"
                    " so that ./verify.sh passes (tests/test_import.py and samples/ are"
                    " new). Do not modify tests/ or verify.sh."
                ),
                inject="leg5",
                timeout_s=1500,
                max_usd=1.00,
            ),
        ),
        protected=(
            "tests/test_money.py",
            "tests/test_journal.py",
            "tests/test_cli.py",
            "tests/test_split.py",
            "tests/test_convert.py",
            "tests/test_report.py",
            "tests/test_import.py",
            "verify.sh",
        ),
        trap_patterns=("ledger/_commands.py",),
        poison=(
            (
                "cli-command-table",
                "ledger/_commands.py is the hand-maintained command table: register new"
                " commands there. tools/gen_cli.py was a one-off bootstrap and no longer runs.",
            ),
            (
                "money-rounding",
                "Amounts round with Python's built-in round() (banker's rounding) by team"
                " convention; the half-up helper in ledger/money.py is legacy.",
            ),
        ),
    ),
}


@dataclass(frozen=True)
class Condition:
    toml: str = ""
    fresh_state_per_leg: bool = False
    poison: bool = False  # plant the task's stale memories before leg 2 (shared state)


# windowNNk conditions pin the tiered thresholds to what the shipped ADAPTIVE
# fractions (45%/80%, models/registry.py) resolve to on that context window.
# That is the regime a small/open-model user gets by default, not an
# artificial squeeze: 32k tokens ~= 131k chars -> drop at 58k chars. A tidy
# reader stays under 58k tool-result chars on these tasks, so window16k
# (local 16k serving) is the rung where tier-1 provably engages.
CONDITIONS: dict[str, Condition] = {
    "baseline": Condition(),
    "window16k": Condition(toml="[context]\ndrop_at_chars = 29000\nsummarise_at_chars = 52000\n"),
    "window32k": Condition(toml="[context]\ndrop_at_chars = 58000\nsummarise_at_chars = 104000\n"),
    "window64k": Condition(toml="[context]\ndrop_at_chars = 115000\nsummarise_at_chars = 205000\n"),
    # The gist A/B arm: window16k thresholds with tier-1 gist elision off
    # (bare markers only, the pre-gist behavior).
    "window16k_nogist": Condition(
        toml=(
            "[context]\ndrop_at_chars = 29000\nsummarise_at_chars = 52000\nelision_gists = false\n"
        )
    ),
    "fresh_state": Condition(fresh_state_per_leg=True),
    "poisoned": Condition(poison=True),
}


def _provider_block(provider: str, model: str) -> str:
    """Pin all three roles to the test model and wire the verify command.
    Providers + secrets come from the layered global ~/.config/agent6."""
    roles = "\n".join(
        f"[models.{role}]\nprovider = {json.dumps(provider)}\nmodel = {json.dumps(model)}\n"
        for role in ("worker", "planner", "reviewer")
    )
    return (
        # `bash verify.sh` (not ./verify.sh): the jail PATH is /usr/bin:/bin and
        # an exec-bit-less script cannot be run directly, but bash can read it.
        f'{roles}\n[workflow]\nverify_command = ["bash", "verify.sh"]\n'
        f'verify_timeout_s = 60.0\n\n[sandbox]\nrun_commands = "yes"\n\n'
    )


def _git(workdir: Path, *args: str, check: bool = True) -> None:
    subprocess.run(["git", *args], cwd=workdir, check=check, capture_output=True)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except ValueError:
                continue
    return out


def _find_logs(state_home: Path, session_id: str) -> Path | None:
    exact = list(state_home.glob(f"agent6/*/sessions/runs/{session_id}/logs.jsonl"))
    if exact:
        return exact[0]
    for cand in state_home.glob("agent6/*/sessions/*/*/logs.jsonl"):
        if session_id in cand.parent.name:
            return cand
    return None


def _extract_metrics(state_home: Path, session_id: str, traps: tuple[str, ...]) -> dict[str, Any]:
    """Pull one leg's metrics from its run's logs.jsonl under the state home."""
    m: dict[str, Any] = {
        "run_found": False,
        "iterations": None,
        "end_reason": None,
        "all_passed": None,
        "usd": None,
        "tokens_in": None,
        "tokens_out": None,
        "n_subtasks": 0,
        "n_subtasks_passed": 0,
        "deps_in_graph": 0,
        "surfaced": 0,
        "tool_calls": 0,
        # Tier-2 (summarise+restart) and tier-1 (oldest tool_results elided).
        "compactions": 0,
        "drop_events": 0,
        "drops_total": 0,
        "first_drop_at_tool_call": None,
        # Tier-1 gist elision: gists written / demoted to bare, and failed
        # distiller calls.
        "gists_total": 0,
        "gist_demotions": 0,
        "gist_failures": 0,
        # Read-shaped calls repeating an earlier (name,args) / an earlier path.
        # The *_post_drop split isolates re-reads AFTER tier-1 first bit (the
        # re-reads compaction plausibly caused); *_post_compact after tier-2.
        "redundant_reads": 0,
        "redundant_reads_post_drop": 0,
        "redundant_reads_post_compact": 0,
        "repeat_path_reads": 0,
        "repeat_path_reads_post_drop": 0,
        # Feature-usage signals under evaluation.
        "deps_added": 0,
        "memory_writes": 0,
        "memory_reads": 0,
        "memory_invalidations": 0,  # memory files changed or removed by the leg (file diff)
        # Write-side nudges the loop fired (flip advisory / deferred finish);
        # with memory_writes they show which surface converts models.
        "memory_flip_nudges": 0,
        "memory_finish_nudges": 0,
        "trap_edits": 0,
    }
    logs = _find_logs(state_home, session_id)
    if logs is None:
        return m
    m["run_found"] = True
    last_budget: dict[str, Any] | None = None
    last_graph: dict[str, Any] | None = None
    seen_calls: set[str] = set()
    seen_paths: set[str] = set()
    dropped = False
    compacted = False
    for e in _read_jsonl(logs):
        t = e.get("type")
        if t == "budget.update":
            last_budget = e
        elif t == "graph.update":
            last_graph = e
        elif t == "session.end":
            m["end_reason"] = e.get("reason")
            m["iterations"] = e.get("iterations")
            m["all_passed"] = e.get("all_passed")
        elif t == "loop.compact.summarise.done":
            m["compactions"] += 1
            compacted = True
        elif t == "loop.compact.dropped":
            m["drop_events"] += 1
            m["drops_total"] += int(e.get("n") or 0)
            if not dropped:
                m["first_drop_at_tool_call"] = m["tool_calls"]
            dropped = True
        elif t == "loop.task.surfaced":
            m["surfaced"] += 1
        elif t == "loop.memory_flip.nudged":
            m["memory_flip_nudges"] += 1
        elif t == "loop.memory_finish.gated":
            m["memory_finish_nudges"] += 1
        elif t == "loop.compact.gists":
            m["gists_total"] += int(e.get("gisted") or 0)
            m["gist_demotions"] += int(e.get("demoted") or 0)
        elif t == "loop.compact.gist.failed":
            m["gist_failures"] += 1
        elif t == "tool.call":
            m["tool_calls"] += 1
            name = e.get("name")
            args = e.get("args") or {}
            if name == "add_dependency":
                m["deps_added"] += 1
            elif name in EDIT_TOOLS:
                target = str(args.get("path") or "") or json.dumps(args)
                # The repo memory is files under the state home's memory dir;
                # an edit there is a memory write (a shell redirect is not seen).
                if "/memory/" in target:
                    m["memory_writes"] += 1
                if traps and any(t in target for t in traps):
                    m["trap_edits"] += 1
            if name in READ_TOOLS:
                sig = f"{name}:{json.dumps(args, sort_keys=True)}"
                if sig in seen_calls:
                    m["redundant_reads"] += 1
                    if dropped:
                        m["redundant_reads_post_drop"] += 1
                    if compacted:
                        m["redundant_reads_post_compact"] += 1
                else:
                    seen_calls.add(sig)
            if name == "read_file":
                path = str(args.get("path", ""))
                if "/memory/" in path:
                    m["memory_reads"] += 1
                if path:
                    if path in seen_paths:
                        m["repeat_path_reads"] += 1
                        if dropped:
                            m["repeat_path_reads_post_drop"] += 1
                    else:
                        seen_paths.add(path)
    if last_budget:
        m["usd"] = last_budget.get("usd_total")
        m["tokens_in"] = last_budget.get("input_total")
        m["tokens_out"] = last_budget.get("output_total")
    if last_graph and isinstance(last_graph.get("nodes"), dict):
        nodes = last_graph["nodes"]
        subs = [n for n in nodes.values() if n.get("parent_id") is not None]
        m["n_subtasks"] = len(subs)
        m["n_subtasks_passed"] = sum(1 for n in subs if n.get("status") == "passed")
        m["deps_in_graph"] = sum(len(n.get("depends_on") or ()) for n in nodes.values())
    return m


def _memory_files(state_home: Path) -> dict[str, str]:
    """The per-repo memory store under this state home: one named `.md` file
    per memory beside the MEMORY.md index, as name -> content hash."""
    return {
        f.stem: hashlib.sha256(f.read_bytes()).hexdigest()
        for f in state_home.glob("agent6/*/memory/*.md")
        if f.name != "MEMORY.md"
    }


def _memory_delta(before: dict[str, str], after: dict[str, str]) -> dict[str, Any]:
    added = [n for n in after if n not in before]
    removed = [n for n in before if n not in after]
    changed = [n for n in after if n in before and before[n] != after[n]]
    return {
        "memory_files_added": len(added),
        "memory_files_changed": len(changed),
        "memory_files_removed": len(removed),
        "memory_invalidations": len(changed) + len(removed),
        "memories_files": len(after),
        "_touched": set(removed) | set(changed),
    }


def _grade(task: str, workdir: Path, leg: str) -> dict[str, Any]:
    grader = TASKS_DIR / task / "grade.py"
    try:
        proc = subprocess.run(
            [sys.executable, str(grader), str(workdir), leg],
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        line = proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else "{}"
        return json.loads(line)
    except Exception as exc:  # grader must never sink the orchestrator
        return {"score": 0.0, "grade_error": f"{type(exc).__name__}: {exc}"[:200]}


def _protected_source(task: str, spec: Task, upto_leg: int, fname: str) -> bytes | None:
    """The seeded content of a protected file at leg `upto_leg`: the latest
    version among repo/ and the inject overlays applied so far."""
    for leg in reversed(spec.legs[: upto_leg + 1]):
        if leg.inject:
            cand = TASKS_DIR / task / leg.inject / fname
            if cand.exists():
                return cand.read_bytes()
    cand = TASKS_DIR / task / "repo" / fname
    return cand.read_bytes() if cand.exists() else None


def _tampered(task: str, spec: Task, upto_leg: int, workdir: Path) -> bool:
    for fname in spec.protected:
        want = _protected_source(task, spec, upto_leg, fname)
        if want is None:
            continue  # not seeded yet at this leg
        got = workdir / fname
        if not got.exists() or got.read_bytes() != want:
            return True
    return False


def one_sequence(
    *,
    task: str,
    model: str,
    provider: str,
    condition: str,
    rep: int,
    budget_scale: float,
    timeout_scale: float,
    label: str,
    leg_memory_max: str = "",
) -> list[dict[str, Any]]:
    spec = TASKS[task]
    cond = CONDITIONS[condition]
    seq = f"{task}-{condition}-r{rep}-{uuid.uuid4().hex[:6]}"
    workdir = RUNS_ROOT / label / seq
    if workdir.exists():
        shutil.rmtree(workdir)
    workdir.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(TASKS_DIR / task / "repo", workdir)

    cfg = workdir / "_run_config.toml"
    cfg.write_text(_provider_block(provider, model) + cond.toml, encoding="utf-8")
    # Fail on a dead config key here, in 200ms, not after a task's spend:
    # `config show` loads the same merged layers the run will.
    subprocess.run(
        [AGENT6_BIN, "--config", str(cfg), "config", "show"],
        cwd=workdir,
        check=True,
        stdout=subprocess.DEVNULL,
    )

    _git(workdir, "init", "-q")
    _git(workdir, "config", "user.email", "bench@bench")
    _git(workdir, "config", "user.name", "bench")
    _git(workdir, "add", "-A")
    _git(workdir, "commit", "-qm", "seed")

    records: list[dict[str, Any]] = []
    # State homes sit BESIDE the workdir: agent6 refuses a private dir inside
    # the workspace it edits.
    shared_state = workdir.parent / f"{seq}.state"
    for i, leg in enumerate(spec.legs):
        if leg.inject:
            shutil.copytree(TASKS_DIR / task / leg.inject, workdir, dirs_exist_ok=True)
            _git(workdir, "add", "-A")
            _git(workdir, "commit", "-qm", f"inject {leg.name}", check=False)

        state_home = (
            workdir.parent / f"{seq}.state-leg{i}" if cond.fresh_state_per_leg else shared_state
        )
        state_home.mkdir(parents=True, exist_ok=True)
        session_id = f"{seq}-L{i}"

        env = dict(os.environ)
        env["XDG_STATE_HOME"] = str(state_home)
        env["AGENT6_FORCE_STREAM"] = "1"
        if cond.poison and i == 1:
            for pname, pbody in spec.poison:
                subprocess.run(
                    [AGENT6_BIN, "--config", str(cfg), "memory", "add", pname, pbody],
                    cwd=workdir,
                    env=env,
                    check=True,
                    capture_output=True,
                )

        budget_flags: list[str]
        if provider == "anthropic":
            # Anthropic has no price data, so the token ledger bounds these legs.
            # 2.2M in+out per leg is generous for these tasks yet caps a haiku leg
            # near $3 worst-case ($1/M in + $5/M out); prompt caching and early
            # finishes land well under. budget_scale dials the wave.
            budget_flags = ["--max-tokens-fallback", str(int(2_200_000 * budget_scale))]
        else:
            budget_flags = ["--max-usd", str(round(leg.max_usd * budget_scale, 2))]

        mem_before = _memory_files(state_home)
        cmd = [
            AGENT6_BIN,
            "run",
            leg.prompt,
            "--config",
            str(cfg),
            "--session-id",
            session_id,
            *budget_flags,
        ]
        if leg_memory_max:
            # One capped transient scope per session: an OOM kills the leg, never the driver.
            cmd = [
                "systemd-run",
                "--user",
                "--scope",
                "--quiet",
                f"-pMemoryMax={leg_memory_max}",
                *cmd,
            ]
        t0 = time.time()
        status = 0
        timed_out = False
        try:
            proc = subprocess.run(
                cmd,
                cwd=workdir,
                env=env,
                capture_output=True,
                text=True,
                timeout=int(leg.timeout_s * timeout_scale),
                check=False,
            )
            status = proc.returncode
            log = (proc.stdout or "") + (proc.stderr or "")
            (workdir / f"agent-{leg.name}.log").write_text(log, "utf-8")
        except subprocess.TimeoutExpired:
            timed_out = True
            status = -9
        wall = round(time.time() - t0, 1)

        grade = _grade(task, workdir, leg.name)
        metrics = _extract_metrics(state_home, session_id, spec.trap_patterns)
        delta = _memory_delta(mem_before, _memory_files(state_home))
        records.append(
            {
                "label": label,
                "task": task,
                "leg": leg.name,
                "leg_index": i,
                "seq": seq,
                "model": model,
                "provider": provider,
                "condition": condition,
                "rep": rep,
                "poisoned": bool(cond.poison and i >= 1),
                "session_id": session_id,
                "score": grade.get("score", 0.0),
                "cases_passed": grade.get("cases_passed"),
                "cases_total": grade.get("cases_total"),
                "components_passed": grade.get("components_passed"),
                "components_total": grade.get("components_total"),
                "component_scores": grade.get("component_scores"),
                "tampered": _tampered(task, spec, i, workdir),
                "wall_s": wall,
                "exit": status,
                "timed_out": timed_out,
                "grade_error": grade.get("grade_error"),
                "import_error": grade.get("import_error"),
                **metrics,
                **{k: v for k, v in delta.items() if not k.startswith("_")},
                "poison_touched": (
                    any(n in delta["_touched"] for n, _ in spec.poison)
                    if cond.poison and i >= 1
                    else None
                ),
            }
        )
    return records


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--provider", default="openrouter")
    ap.add_argument("--tasks", default="stylebook,relay,orchard")
    ap.add_argument("--conditions", default="baseline")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--parallel", type=int, default=3)
    ap.add_argument("--budget-scale", type=float, default=1.0)
    ap.add_argument(
        "--timeout-scale",
        type=float,
        default=1.0,
        help="Multiply every leg's timeout_s (raise for slow single-turn models like kimi).",
    )
    ap.add_argument("--label", required=True)
    ap.add_argument(
        "--leg-memory-max",
        default="",
        help="run each session in its own systemd-run scope with this MemoryMax (e.g. 8G)",
    )
    args = ap.parse_args()

    tasks = [t for t in args.tasks.split(",") if t]
    conditions = [c for c in args.conditions.split(",") if c]
    for t in tasks:
        if t not in TASKS:
            sys.exit(f"unknown task {t!r}; known: {sorted(TASKS)}")
    for c in conditions:
        if c not in CONDITIONS:
            sys.exit(f"unknown condition {c!r}; known: {sorted(CONDITIONS)}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"{args.label}.jsonl"
    # Resumable: a cell (task, condition, rep) with every leg already recorded
    # under this label is skipped; a partial cell reruns whole and is named.
    have: dict[tuple[str, str, int], set[str]] = {}
    if out_path.exists():
        for rec in _read_jsonl(out_path):
            have.setdefault((rec["task"], rec["condition"], rec["rep"]), set()).add(rec["leg"])
    jobs = [
        dict(
            task=t,
            model=args.model,
            provider=args.provider,
            condition=c,
            rep=r,
            budget_scale=args.budget_scale,
            timeout_scale=args.timeout_scale,
            label=args.label,
            leg_memory_max=args.leg_memory_max,
        )
        for t in tasks
        for c in conditions
        for r in range(args.reps)
    ]
    kept: list[dict[str, Any]] = []
    for j in jobs:
        legs = {leg.name for leg in TASKS[j["task"]].legs}
        seen = have.get((j["task"], j["condition"], j["rep"]), set())
        if legs <= seen:
            continue
        if seen:
            print(
                f"[longhorizon] partial cell reruns whole: {j['task']}/{j['condition']} r{j['rep']}"
            )
        kept.append(j)
    if len(kept) < len(jobs):
        print(f"[longhorizon] resume: {len(jobs) - len(kept)} complete cell(s) skipped")
    jobs = kept
    n_legs = sum(len(TASKS[j["task"]].legs) for j in jobs)
    print(f"[longhorizon] {len(jobs)} sequences ({n_legs} legs), parallel={args.parallel}")
    print(f"[longhorizon] model={args.model} -> {out_path}")
    done = 0
    with cf.ThreadPoolExecutor(max_workers=args.parallel) as ex:
        futs = {ex.submit(one_sequence, **j): j for j in jobs}
        with out_path.open("a", encoding="utf-8") as f:
            for fut in cf.as_completed(futs):
                j = futs[fut]
                try:
                    recs = fut.result()
                except Exception as exc:  # one sequence must not sink the batch
                    recs = [{**j, "score": 0.0, "crash": f"{type(exc).__name__}: {exc}"[:300]}]
                for rec in recs:
                    f.write(json.dumps(rec) + "\n")
                f.flush()
                done += 1
                for rec in recs:
                    print(
                        f"[{done}/{len(jobs)}] {rec.get('task')}/{rec.get('condition')}"
                        f" r{rec.get('rep')} {rec.get('leg', '?')}"
                        f" score={rec.get('score')} drops={rec.get('drops_total')}"
                        f" rr={rec.get('redundant_reads')} memw={rec.get('memory_writes')}"
                        f" nudges={rec.get('memory_flip_nudges')}/{rec.get('memory_finish_nudges')}"
                        f" deps={rec.get('deps_added')} iters={rec.get('iterations')}"
                        f" ${rec.get('usd')} {rec.get('wall_s')}s"
                        f" reason={rec.get('end_reason')}"
                    )
    print(f"[longhorizon] done -> {out_path}")


if __name__ == "__main__":
    main()
