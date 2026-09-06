#!/usr/bin/env bash
# Anthropic perf-takehome head-to-head harness for agent6.
#
# Optimization task with a budget cap (default ~$5). Both agent6 and
# claude-code get the same task description, the same upstream commit of
# https://github.com/anthropics/original_performance_takehome, and the
# same budget. Scoring is the cycle count printed by
# `python tests/submission_tests.py` after the run, measured by the
# scoring harness (NOT by the agent — so cheating by mutating the tests
# can be detected post hoc).
#
# Cheat detection: after the run we check `git diff origin/main -- tests/`
# is empty. If not, the result is marked CHEATED in the summary and the
# cycle count is still reported but flagged.
#
# Usage:
#   ANTHROPIC_API_KEY=... bash bench/perf/run_perf.sh
#
# Outputs: $BENCH_ROOT/perf_agent6/        (the worktree the agent edited)
#          $BENCH_ROOT/_logs/agent6/       (stdout/stderr/result.json)
#          $BENCH_ROOT/result_agent6.json  (post-hoc score)

set -euo pipefail

REPO="${AGENT6_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
BENCH_ROOT="${BENCH_ROOT:-/tmp/agent6-perf}"
PERF_REPO_URL="${PERF_REPO_URL:-https://github.com/anthropics/original_performance_takehome.git}"
# Pinned default for reproducibility - bench numbers across iters need a
# fixed starting point. Override via env var to track upstream. SHA-pinned
# repo means a midnight upstream push can't silently change comparisons.
PERF_REPO_COMMIT="${PERF_REPO_COMMIT:-5452f74bd977807ac2e74f3d29432b9df6f25197}"
MODEL="${AGENT6_PERF_MODEL:-claude-sonnet-4-5}"
SUMMARY_MODEL="${AGENT6_PERF_SUMMARY_MODEL:-claude-haiku-4-5}"
# Budget: the same $5 the claude-code side gets, metered; the fallback bounds
# any call the meter cannot price. Either cap ends the run via the exit-3 path.
MAX_USD="${AGENT6_PERF_MAX_USD:-5.0}"
MAX_TOKENS_FALLBACK="${AGENT6_PERF_MAX_TOKENS_FALLBACK:-1500000}"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python3)}"

cd "$REPO"
export AGENT6_JAIL_BIN="${AGENT6_JAIL_BIN:-$REPO/src/agent6/jail/target/release/agent6-jail}"
AGENT6_BIN="$REPO/.venv/bin/agent6"
[ -x "$AGENT6_BIN" ] || { echo "agent6 not found at $AGENT6_BIN" >&2; exit 1; }
[ -x "$AGENT6_JAIL_BIN" ] || { echo "jail launcher missing at $AGENT6_JAIL_BIN" >&2; exit 1; }

mkdir -p "$BENCH_ROOT/_logs/agent6"
WORKDIR="$BENCH_ROOT/perf_agent6"
LOGDIR="$BENCH_ROOT/_logs/agent6"

# Fresh clone every run. init + fetch-by-SHA works for both SHA and
# branch values (clone --branch requires a branch name, breaking pinning).
rm -rf "$WORKDIR"
git init --quiet "$WORKDIR"
git -C "$WORKDIR" remote add origin "$PERF_REPO_URL"
git -C "$WORKDIR" fetch --quiet --depth 1 origin "$PERF_REPO_COMMIT"
git -C "$WORKDIR" checkout --quiet -B main FETCH_HEAD
# origin/main remote-tracking ref is what score.sh diffs against.
git -C "$WORKDIR" update-ref refs/remotes/origin/main FETCH_HEAD
git -C "$WORKDIR" config user.email "bench@agent6"
git -C "$WORKDIR" config user.name "bench"

# --- task description handed to the agent -----------------------------------
cat > "$WORKDIR/TASK.md" <<'MD'
# Optimize KernelBuilder.build_kernel for the simulated machine

This repo is Anthropic's original performance take-home. The simulated
machine in `problem.py` runs the kernel emitted by
`KernelBuilder.build_kernel` (in `perf_takehome.py`) and reports a cycle
count. The goal is to minimize that cycle count.

## How to measure cycles

Run:

```
/usr/bin/python3 tests/submission_tests.py
```

(Use the absolute `/usr/bin/python3` — a bare `python` is not on PATH
in this sandbox.)

It prints lines like `CYCLES:  18532` and tells you which speed-tier
tests pass. Lower is better. The baseline starting point is 147734
cycles; Claude Opus 4.5's best at launch was 1487 cycles.

**Important about `verify_command`**: this run's configured
`verify_command` is `pytest tests/submission_tests.py::CorrectnessTests`
— it only checks correctness, not the speed tiers. That is
deliberate: speed-tier `assert cycles < N` lines would fail every
step until the final tier is reached, which would terminate the run
after a single failing step. With correctness-only verify, every
commit that preserves behaviour is accepted and the planner can
chain many cycle-reducing edits.

**This run also has `[workflow.metric]` configured**: after every
passing step, the harness automatically runs
`/usr/bin/python3 tests/submission_tests.py`, parses out the
`CYCLES:` number, and feeds the trajectory (baseline, current, best,
last few commits) back to you as a `[harness metric]` block in the
next step's prompt. Keep verified edits only when that block shows a
new best; if it is flat or worse, revert or pivot to a different
bottleneck. You do **not** need to run the cycle command manually for
this feedback to happen — but feel free to run it inside an edit if
you want to double-check the impact of a specific change before
committing.

## HARD RULES — read carefully, violating these voids the result

1. **Do not modify anything in `tests/`.** Not `tests/submission_tests.py`,
   not `tests/frozen_problem.py`, nothing under that directory. The
   scoring harness will run `git diff origin/main -- tests/` after you
   finish; any non-empty diff is treated as cheating and the score is
   discarded. The `tests/frozen_problem.py` file is intentionally a
   frozen copy of the original `problem.py` — changing it to make the
   simulator "faster" is the exact failure mode the upstream README
   warns about.
2. **Do not change `N_CORES` in `problem.py`.** Multicore is disabled
   intentionally. Setting `N_CORES > 1` is documented as the canonical
   cheat and will be detected (any change to that constant is flagged).
3. **Correctness first.** `tests/submission_tests.py`'s
   `CorrectnessTests` must pass. If it does not, the cycle count is
   doubled (the upstream test does this automatically).
4. **All optimization happens in `perf_takehome.py`.** You may also
   edit `problem.py` for changes that are pure performance improvements
   to the simulator's *implementation* without changing its semantics
   (e.g. replacing `list` with `array.array`, hoisting a dict lookup),
   but do NOT change anything that affects the cycle count of a given
   kernel: the cycle accounting in `Machine.run`, `SLOT_LIMITS`,
   `VLEN`, `N_CORES`, `SCRATCH_SIZE`, `HASH_STAGES`, or the engine
   model. When in doubt, only edit `perf_takehome.py`.

## How to work

- Read `perf_takehome.py` top-to-bottom first — the docstring explains
  the engine model (alu / load / store / flow), the VLIW packing, and
  what `build_kernel` is allowed to emit.
- Read `problem.py` for the instruction semantics and cycle accounting.
- After each change, run `/usr/bin/python3 tests/submission_tests.py`
  and look at the `CYCLES:` line. Keep changes that reduce cycles AND
  keep correctness; revert anything else.
- Commit working improvements as you go so we have a history. If you
  break correctness, revert.

You have a fixed compute budget. Iterate as many times as the budget
allows. Stopping early is wasted budget — keep optimizing right up
until the cap. The scoring harness rescues the best (lowest-CYCLES)
commit on this branch after the run ends, so committing every
improvement matters.
MD

# --- agent6.toml -------------------------------------------------------------
# verify_command runs the upstream submission_tests.py. Returncode is 0 only
# when ALL speed tiers pass (which is effectively unreachable), so agent6
# will always retry. That's intended — we want it to spend the full budget.
cat > "$WORKDIR/agent6.toml" <<EOF
[agent6]
config_version = 1

[providers.anthropic]
api_format = "anthropic"
api_key_env = "ANTHROPIC_API_KEY"
prompt_caching = true

[models.worker]
provider = "anthropic"
model = "$MODEL"

[models.reviewer]
provider = "anthropic"
model = "$MODEL"

[sandbox]
isolation = "auto"
network = "session"
run_commands = "yes"
protect_git = true

[git]
dirty_tree = "ask"
branch_per_run = true

[workflow]
# Verify only checks correctness — speed-tier tests would always fail
# until the final improvement and would terminate the run after step 1.
# The agent still measures cycles itself by running
# `/usr/bin/python3 tests/submission_tests.py` directly.
# Absolute paths because the agent6 sandbox only bind-mounts /usr —
# a bare "python"/"pytest" is not on PATH inside the jail.
verify_command = [
  "/usr/bin/pytest",
  "-q",
  "tests/submission_tests.py::CorrectnessTests",
]
# Cap verify at 30s (baseline runs in ~2s). Infinite-loop /
# quadratic edits previously hung for the jail's 600s default, wasting
# ~10 min per failed step and 30+ min per step that exhausted retries.
verify_timeout_s = 30.0

[prompt]
revise_prompt = "${AGENT6_PERF_REVISE_PROMPT:-off}"

# Continuous-score metric: after every passing step, agent6 runs this
# command in the jail, parses the captured group out of stdout+stderr
# with \`\`pattern\`\`, and feeds the trajectory back to the worker on the
# NEXT step. When the initial plan finishes, the outer loop re-plans
# while the metric is still improving. This is the dial that lets us
# spend the full budget on continuous optimization rather than stopping
# after the planner's first 5-10 steps.
[workflow.metric]
command = ["/usr/bin/python3", "tests/submission_tests.py"]
pattern = 'CYCLES:\s*(\d+)'
goal = "minimize"

[budget]
max_usd = $MAX_USD
max_tokens_fallback = $MAX_TOKENS_FALLBACK
EOF

# Fail on a dead config key here, in 200ms, not after the run's spend:
# `config show` loads the same merged layers the run will.
( cd "$WORKDIR" && "$AGENT6_BIN" --config agent6.toml config show >/dev/null ) || exit 1

# Ignore the bench-only files so agent6's dirty-tree question does not fire.
{
  echo "agent6.toml"
  echo "TASK.md"
  echo ".agent6/"
} >> "$WORKDIR/.gitignore"
git -C "$WORKDIR" add .gitignore
git -C "$WORKDIR" commit -q -m "bench: gitignore harness files"

# Capture the starting cycle count for reference.
( cd "$WORKDIR" && "$PYTHON_BIN" tests/submission_tests.py 2>&1 ) > "$LOGDIR/cycles_before.txt" || true
start_cycles=$(grep -oE 'CYCLES:[[:space:]]+[0-9]+' "$LOGDIR/cycles_before.txt" | head -1 | awk '{print $2}')
echo "Starting cycles: ${start_cycles:-unknown}"

task_text=$(cat "$WORKDIR/TASK.md")

start_ns=$(date +%s%N)
set +e
( cd "$WORKDIR" && "$AGENT6_BIN" --config agent6.toml run "$task_text" ) \
  > "$LOGDIR/agent6.stdout" 2> "$LOGDIR/agent6.stderr"
ag_exit=$?
set -e
end_ns=$(date +%s%N)
wall_s=$(awk -v s="$start_ns" -v e="$end_ns" 'BEGIN{printf "%.1f", (e-s)/1e9}')

# Pull cost/tokens from the budget summary.
combined="$LOGDIR/agent6.stdout $LOGDIR/agent6.stderr"
cost=$(cat $combined 2>/dev/null | grep -oE 'cost~\$[0-9.]+' | tail -1 | tr -d '$' | sed 's/cost~//')
[ -z "$cost" ] && cost=0
in_tok=$(cat $combined 2>/dev/null | grep -oE 'TOTAL: in=[0-9]+' | tail -1 | sed 's/TOTAL: in=//')
[ -z "$in_tok" ] && in_tok=0
out_tok=$(cat $combined 2>/dev/null | grep -oE 'out=[0-9]+/' | tail -1 | tr -d '/' | sed 's/out=//')
[ -z "$out_tok" ] && out_tok=0

# --- post-hoc scoring (this is what we trust) ------------------------------
# branch_per_run puts the work on the agent6 run branch; HEAD stays at
# the baseline. Score the branch the summary names, or HEAD if none.
run_ref=$(cat $combined 2>/dev/null | grep -oE 'changes are on agent6/[A-Za-z0-9_-]+' | tail -1 | sed 's/^changes are on //')

bash "$REPO/bench/perf/score.sh" "$WORKDIR" "agent6" \
  "$wall_s" "$cost" "$in_tok" "$out_tok" "$ag_exit" "$start_cycles" "${run_ref:-HEAD}" \
  > "$BENCH_ROOT/result_agent6.json"

cat "$BENCH_ROOT/result_agent6.json"
