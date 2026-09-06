#!/usr/bin/env python3
"""Core-agent benchmark orchestrator.

Runs the installed agent6 binary on the multi-component tasks under
``tasks/<name>/`` for a (model x condition x task x rep) matrix, grades each run
with the task's authoritative HIDDEN grader (``tasks/<name>/grade.py``, never
shipped into the agent's repo), and records one JSON line per run.

Each run is fully isolated: a throwaway git repo seeded from ``tasks/<name>/repo``
plus a private ``XDG_STATE_HOME`` so the run's ``logs.jsonl`` is found
deterministically. Runs execute in a thread pool (separate agent6 processes), so
``--parallel N`` scales sample count within a fixed wall-clock budget.

Conditions are config-override fragments (see ``CONDITIONS``): ``baseline`` is the
shipped default; the others toggle one knob so an A/B isolates that knob. The
prompt handed to the agent is deliberately NEUTRAL about decomposition -- any
difference in DAG use comes from the condition's config, not the task prompt.

Usage:
  python3 run.py --model qwen/qwen3.6-35b-a3b --provider openrouter \
      --tasks textkit,rpn,ledger --conditions baseline,decompose \
      --reps 4 --parallel 4 --budget 0.60 --label screen1

Results: results/<label>.jsonl (append). Summarize with stats.py.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
TASKS_DIR = ROOT / "tasks"
RESULTS_DIR = ROOT / "results"
AGENT6_BIN = os.environ.get("AGENT6_BIN") or shutil.which("agent6") or "agent6"
RUNS_ROOT = Path(os.environ.get("COREAGENT_RUNS", str(Path.home() / "coreagent-runs")))

# Per-task module name + the neutral prompt. The prompt never mentions the DAG or
# decomposition; that behaviour is the variable under test (set by a condition).
TASKS: dict[str, str] = {
    "textkit": "textkit",
    "rpn": "rpn",
    "ledger": "ledger",
    "bugs": "shapes",  # DEBUG task: fix bugs in shapes.py
    "eventflow": "eventflow",  # DEBUG task: 5-module pipeline, spec-vs-suite gap
    "needle": "report",  # read 5 ref files (forces compaction), retain all 5 rules
}


def task_prompt(task: str, module: str) -> str:
    if task == "needle":
        return (
            "Read spec.md, then read ALL of ref1.md, ref2.md, ref3.md, ref4.md,"
            f" and ref5.md (each holds one precise rule). Implement {module}.py to"
            " satisfy EVERY rule so ./verify.sh passes (stdlib unittest). Do not"
            " modify the test file or verify.sh. Run verify, fix what fails, and"
            " call finish_session when the whole suite passes."
        )
    if task == "eventflow":
        return (
            "Read spec.md. The pipeline modules (config.py, parser.py, sessions.py,"
            " stats.py, report.py) contain several bugs. Find and fix EVERY bug so"
            " the code conforms to spec.md. ./verify.sh runs the committed unittest"
            " suite, but it covers only a SUBSET of the spec: a green suite does"
            " not mean the code is correct. Check every module against the spec."
            " Do not modify test_eventflow.py or verify.sh. Call finish_session when"
            " you are confident the code matches the spec."
        )
    if task == "bugs":
        return (
            f"Read spec.md. {module}.py implements 7 functions but several have"
            " bugs. Find and fix EVERY bug so that ./verify.sh passes (it runs the"
            " stdlib unittest suite). Check every function against the spec, not"
            " just the first failures. Do not modify the test file or verify.sh."
            " Run verify, fix what fails, and call finish_session when the whole suite"
            " passes."
        )
    return (
        f"Read spec.md and fully implement {module}.py so that ./verify.sh passes"
        " (it runs the stdlib unittest suite). Implement EVERY component described"
        " in the spec, not just the first. Do not modify the test file or"
        " verify.sh. Run verify, fix what fails, and call finish_session when the"
        " whole suite passes."
    )


# Condition -> extra config TOML appended to the per-run config. Empty == shipped
# default. New knobs plug in here once they exist.
CONDITIONS: dict[str, str] = {
    "baseline": "",
    # Thrust 2: front-loaded decomposition.
    "decompose": "[prompt]\ndecompose = true\n",
    # Aggressive thresholds that force compaction on the read-heavy `needle` task
    # (for future tier-1 compaction work). NB the keep-last-K tier-2 hybrid that
    # this once A/B'd was measured inert and scrapped -- see FINDINGS.md; tier-2
    # almost never fires because tier-1 keeps context bounded below its trigger.
    "compact_tight": "[context]\ndrop_at_chars = 18000\nsummarise_at_chars = 36000\n",
    # Skills/prompt-style thrust: conditions swap the static base prompt for a
    # rendered file under prompts/ ({ROOT} is filled with this directory at run
    # time). base_rendered.md is byte-equivalent to the shipped run-mode base;
    # each variant appends one block so the A/B isolates that block.
    # moose: positive control -- a trivially detectable marker instruction that
    # proves the prompt channel delivers (canary gate, not a hypothesis).
    # Measured 2026-07-10: appended at the base's END, BOTH qwen3-coder-30b and
    # mistral-small-3.2 ignored it completely (0/12, 0/27 prose turns).
    # brief: the run-mode base rewritten as a complete operating brief (a
    # workflow, the tool map, the constraints; ~2.2k chars vs the shipped 1.0k
    # mechanics-only base). Same sentinels as the base, so isolation-specific
    # rules render identically in both arms.
    "brief": '[prompt]\nsystem_prompt_file = "{ROOT}/prompts/brief.md"\n',
    "moose": '[prompt]\nsystem_prompt_file = "{ROOT}/prompts/moose.md"\n',
    "moose_top": '[prompt]\nsystem_prompt_file = "{ROOT}/prompts/moose_top.md"\n',
    # moose_user: same marker instruction as a TASK-PROMPT suffix (see
    # USER_SUFFIX) -- the channel `--skill`/steer-injected content rides.
    "moose_user": "",
    # H1 terse style, 3 arms: baseline / one-line concise control / distilled
    # caveman ruleset. The honest delta is ruleset vs terse_line (caveman's own
    # eval design); compliance is measured from transcripts, never assumed.
    "terse_line": '[prompt]\nsystem_prompt_file = "{ROOT}/prompts/terse_line.md"\n',
    "terse_ruleset": '[prompt]\nsystem_prompt_file = "{ROOT}/prompts/terse_ruleset.md"\n',
    # H3 negative control: a realistic 14-skill index, all irrelevant --
    # measures pure presence/distraction cost of a skills index block.
    "index_irrelevant": '[prompt]\nsystem_prompt_file = "{ROOT}/prompts/index_irrelevant.md"\n',
    # H4: superpowers systematic-debugging SKILL.md injected whole (the
    # `always`-skill delivery shape); pair with the `bugs` task only.
    "skill_debug": '[prompt]\nsystem_prompt_file = "{ROOT}/prompts/skill_debug.md"\n',
    # H5: existing knob, never measured per-dollar on small models.
    "priors_off": "[prompt]\nstructural_priors = false\n",
    # H2 delivery mechanism (needs an agent6 with the skills feature): the
    # same systematic-debugging content as skill_debug, delivered as an
    # always-skill vs an indexed on-demand skill (does the model call
    # use_skill unprompted?). skillpacks/ is the fixture skill dir
    # (superpowers, MIT).
    "skill_always": (
        '[skills]\nextra_dirs = ["{ROOT}/skillpacks"]\n'
        '[skills.state]\nsystematic-debugging = "always"\n'
    ),
    "skill_ondemand": '[skills]\nextra_dirs = ["{ROOT}/skillpacks"]\n',
    # H3b: length-matched NEUTRAL padding (~same bytes as the irrelevant
    # index). If this reproduces the index harm on mistral, the effect is
    # plain prompt-length sensitivity, not the skills index specifically.
    "padding_neutral": '[prompt]\nsystem_prompt_file = "{ROOT}/prompts/padding_neutral.md"\n',
    # Native spec-recheck finish gate: the skill mechanism's one-turn bounce
    # as a loop feature ([workflow].spec_recheck_on_finish). A/B against
    # baseline on eventflow (spec-vs-suite gap) and textkit (suite IS the
    # spec: measures the pure cost of a wasted bounce).
    # Header-less on purpose: extends the base block's trailing [workflow].
    "spec_recheck": "spec_recheck_on_finish = true\n",
    # Bootstrap question: with the using-superpowers meta-skill injected
    # always-on (verbatim, no harness tool-mapping added), does the model
    # start invoking use_skill for a matching indexed skill? Compare the
    # organic use_skill call rate against skill_ondemand.
    "skill_bootstrap": (
        '[skills]\nextra_dirs = ["{ROOT}/skillpacks"]\n'
        '[skills.state]\nusing-superpowers = "always"\n'
    ),
}

# Condition -> extra text appended to the task prompt (the USER channel, vs the
# system-prompt channel above). Both channels need positive-control coverage:
# skills content can be delivered over either.
USER_SUFFIX: dict[str, str] = {
    "moose_user": (
        "\n\nMANDATORY STYLE MARKER: End every reply that contains prose "
        "(any non-code text) with the single word MOOSE on its own line. "
        "Never omit it."
    ),
}


def _provider_block(provider: str, model: str, verify: list[str], effort: str = "") -> str:
    """Per-run base config: pin all three roles to the test model, wire verify,
    allow run_command. Providers + secrets come from the layered global config."""
    verify_toml = "[" + ", ".join(json.dumps(a) for a in verify) + "]"
    effort_line = f"effort = {json.dumps(effort)}\n" if effort else ""
    roles = "\n".join(
        f"[models.{role}]\nprovider = {json.dumps(provider)}\nmodel = {json.dumps(model)}\n"
        + effort_line
        for role in ("worker", "planner", "reviewer")
    )
    # [workflow] is deliberately the LAST section: a condition fragment
    # WITHOUT a section header (see CONDITIONS) legally extends it, whereas a
    # second [workflow] header would be invalid TOML (cannot declare twice).
    return (
        f'{roles}\n[sandbox]\nrun_commands = "yes"\n\n'
        f"[workflow]\nverify_command = {verify_toml}\n"
        f"verify_timeout_s = 60.0\n"
    )


def _git(workdir: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=workdir, check=True, capture_output=True)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except ValueError:
            continue
    return out


def _extract_metrics(state_home: Path) -> dict[str, Any]:
    """Pull run metrics from the single run's logs.jsonl under the isolated state."""
    logs = list(state_home.glob("agent6/*/sessions/*/*/logs.jsonl"))
    m: dict[str, Any] = {
        "run_found": bool(logs),
        "iterations": None,
        "end_reason": None,
        "all_passed": None,
        "usd": None,
        "tokens_in": None,
        "tokens_out": None,
        "n_subtasks": 0,
        "n_subtasks_passed": 0,
        "compactions": 0,
        "drops": 0,
        "surfaced": 0,
        "tool_calls": 0,
        # Redundant re-reads: a read-shaped tool.call whose (name,args) repeats an
        # earlier one. The primary compaction-quality signal -- the hard restart
        # makes the worker re-read elided files; a keep-last-K tail should cut it.
        # ``redundant_reads_post_compact`` isolates the re-reads after the first
        # tier-2 restart (the ones compaction actually causes).
        "redundant_reads": 0,
        "redundant_reads_post_compact": 0,
        # Repeat-PATH reads: a read_file of a path already read this run (ANY
        # offset/limit). Looser than redundant_reads -- catches re-opening an
        # elided file even with different ranges, which the hard restart should
        # cause more of than the keep-last-K tail.
        "repeat_path_reads": 0,
        "repeat_path_reads_post_compact": 0,
    }
    if not logs:
        return m
    events = _read_jsonl(logs[0])
    last_budget: dict[str, Any] | None = None
    last_graph: dict[str, Any] | None = None
    seen_calls: set[str] = set()
    seen_paths: set[str] = set()
    compacted = False
    read_tools = {"read_file", "outline", "find_definition", "find_references", "list_dir"}
    for e in events:
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
            m["drops"] += 1
        elif t == "loop.task.surfaced":
            m["surfaced"] += 1
        elif t == "loop.tool.call":
            m["tool_calls"] += 1
        elif t == "tool.call" and e.get("name") in read_tools:
            sig = f"{e.get('name')}:{json.dumps(e.get('args'), sort_keys=True)}"
            if sig in seen_calls:
                m["redundant_reads"] += 1
                if compacted:
                    m["redundant_reads_post_compact"] += 1
            else:
                seen_calls.add(sig)
        if t == "tool.call" and e.get("name") == "read_file":
            path = str((e.get("args") or {}).get("path", ""))
            if path:
                if path in seen_paths:
                    m["repeat_path_reads"] += 1
                    if compacted:
                        m["repeat_path_reads_post_compact"] += 1
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
    return m


def _grade(task: str, workdir: Path) -> dict[str, Any]:
    grader = TASKS_DIR / task / "grade.py"
    try:
        proc = subprocess.run(
            [sys.executable, str(grader), str(workdir)],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        line = proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else "{}"
        return json.loads(line)
    except Exception as exc:  # grader must never crash the orchestrator
        return {"score": 0.0, "grade_error": f"{type(exc).__name__}: {exc}"[:200]}


def _tampered(task: str, workdir: Path) -> bool:
    repo = TASKS_DIR / task / "repo"
    module = TASKS[task]
    for fname in (f"test_{module}.py", "verify.sh"):
        a, b = repo / fname, workdir / fname
        if not b.exists() or a.read_bytes() != b.read_bytes():
            return True
    return False


def one_run(
    *,
    task: str,
    model: str,
    provider: str,
    condition: str,
    rep: int,
    budget: float,
    label: str,
    effort: str = "",
) -> dict[str, Any]:
    module = TASKS[task]
    session_id = f"{task}-{condition}-r{rep}-{uuid.uuid4().hex[:6]}"
    workdir = RUNS_ROOT / label / session_id
    # Beside the workdir, never inside it: agent6 refuses a state base inside
    # the workspace (jailed commands could read transcripts and keys).
    state_home = RUNS_ROOT / label / ".state" / session_id
    if workdir.exists():
        shutil.rmtree(workdir)
    if state_home.exists():
        shutil.rmtree(state_home)
    workdir.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(TASKS_DIR / task / "repo", workdir)
    state_home.mkdir(parents=True, exist_ok=True)

    # Per-run config: model pins + verify + condition fragment.
    cfg = workdir / "_run_config.toml"
    cfg.write_text(
        # `bash verify.sh` (not ./verify.sh): the jail PATH is /usr/bin:/bin and
        # an exec-bit-less script can't be run directly, but bash can read it.
        _provider_block(provider, model, ["bash", "verify.sh"], effort)
        + CONDITIONS[condition].format(ROOT=ROOT),
        encoding="utf-8",
    )
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

    env = dict(os.environ)
    env["XDG_STATE_HOME"] = str(state_home)
    # Isolate the data dir too: skills installed on the host must never leak
    # into a bench run's <skills> index; skill arms opt in via extra_dirs.
    env["XDG_DATA_HOME"] = str(state_home / "data")
    env["AGENT6_FORCE_STREAM"] = "1"
    env["HOME"] = os.environ["HOME"]  # keep ~/.config/agent6 providers+secrets

    budget_flags: list[str]
    if provider == "anthropic":
        # Anthropic is unpriced, so max_usd cannot bound it: cap the token
        # ledger that prices-unknown calls count against instead.
        budget_flags = ["--max-tokens-fallback", "4400000"]
    else:
        budget_flags = ["--max-usd", str(budget)]

    cmd = [
        AGENT6_BIN,
        "run",
        task_prompt(task, module) + USER_SUFFIX.get(condition, ""),
        "--config",
        str(cfg),
        "--session-id",
        session_id,
        *budget_flags,
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
            timeout=900,
            check=False,
        )
        status = proc.returncode
        (workdir / "agent.log").write_text((proc.stdout or "") + (proc.stderr or ""), "utf-8")
    except subprocess.TimeoutExpired:
        timed_out = True
        status = -9
    wall = round(time.time() - t0, 1)

    grade = _grade(task, workdir)
    metrics = _extract_metrics(state_home)
    rec = {
        "label": label,
        "task": task,
        "model": model,
        "provider": provider,
        "condition": condition,
        "rep": rep,
        "session_id": session_id,
        "score": grade.get("score", 0.0),
        "cases_passed": grade.get("cases_passed"),
        "cases_total": grade.get("cases_total"),
        "components_passed": grade.get("components_passed"),
        "components_total": grade.get("components_total"),
        "tampered": _tampered(task, workdir),
        "wall_s": wall,
        "exit": status,
        "timed_out": timed_out,
        "grade_error": grade.get("grade_error"),
        "import_error": grade.get("import_error"),
        **metrics,
    }
    return rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--provider", default="openrouter")
    ap.add_argument("--tasks", default="textkit,rpn,ledger")
    ap.add_argument("--conditions", default="baseline")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--parallel", type=int, default=4)
    ap.add_argument("--budget", type=float, default=0.60)
    ap.add_argument("--effort", default="", help="pin [models.*].effort for every role")
    ap.add_argument("--label", required=True)
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
    jobs = [
        dict(
            task=t,
            model=args.model,
            provider=args.provider,
            condition=c,
            rep=r,
            budget=args.budget,
            label=args.label,
            effort=args.effort,
        )
        for t in tasks
        for c in conditions
        for r in range(args.reps)
    ]
    print(f"[coreagent] {len(jobs)} runs, parallel={args.parallel}, model={args.model}")
    print(f"[coreagent] -> {out_path}")
    done = 0
    with cf.ThreadPoolExecutor(max_workers=args.parallel) as ex:
        futs = {ex.submit(one_run, **j): j for j in jobs}
        with out_path.open("a", encoding="utf-8") as f:
            for fut in cf.as_completed(futs):
                j = futs[fut]
                try:
                    rec = fut.result()
                except Exception as exc:  # one run must not sink the batch
                    rec = {**j, "score": 0.0, "crash": f"{type(exc).__name__}: {exc}"[:300]}
                f.write(json.dumps(rec) + "\n")
                f.flush()
                done += 1
                print(
                    f"[{done}/{len(jobs)}] {rec.get('task')}/{rec.get('condition')}"
                    f" r{rec.get('rep')} score={rec.get('score')}"
                    f" subtasks={rec.get('n_subtasks')} comp={rec.get('compactions')}"
                    f" iters={rec.get('iterations')} ${rec.get('usd')} {rec.get('wall_s')}s"
                    f" reason={rec.get('end_reason')}"
                )
    print(f"[coreagent] done -> {out_path}")


if __name__ == "__main__":
    main()
