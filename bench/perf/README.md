# bench/perf: Anthropic perf-takehome head-to-head

A $5-budget head-to-head between agent6 and claude-code on Anthropic's
open-source [original_performance_takehome](https://github.com/anthropics/original_performance_takehome).
Both agents are given the same task description (`TASK.md` is generated
inline by each runner and is byte-for-byte identical), the same upstream
commit, the same model (`claude-sonnet-4-5`), and the same total budget
cap. Whoever achieves the lowest `CYCLES:` count from
`python tests/submission_tests.py` wins.

## The task

Optimize `KernelBuilder.build_kernel` in `perf_takehome.py` so the
simulated machine in `problem.py` runs the kernel in fewer clock
cycles. Baseline is 147734 cycles; the best public score at Claude
Opus 4.5's launch was 1487 cycles.

## Anti-cheat

Upstream warned that early models gamed the benchmark in two ways:

1. **Modifying `tests/`** to relax thresholds or weaken the frozen
   simulator (`tests/frozen_problem.py`).
2. **Changing `N_CORES`** in `problem.py` from 1 to N to fake a speedup
   on top of an intentionally-disabled multicore path.

`bench/perf/score.sh` enforces both:

- `git diff origin/main -- tests/` must be empty.
- `N_CORES = 1` must remain in `problem.py`.
- Before extracting the cycle count, the harness `git checkout
  origin/main -- tests/` to run the upstream tests against whatever the
  agent produced. If the agent quietly weakened the tests to "pass",
  the rescored cycles reflect reality.

If either anti-cheat check fails the `valid` field in the result JSON
is `false` and the score is flagged.

## Running

```bash
# agent6 side (~$5, single run, fresh clone)
ANTHROPIC_API_KEY=... bash bench/perf/run_perf.sh

# claude-code side (~$5, single run, fresh clone)
ANTHROPIC_API_KEY=... bash bench/perf/run_perf_claude.sh

# any OpenRouter model (Kimi K2.6 default, ~$1.50/run)
OPENROUTER_API_KEY=... bash bench/perf/run_perf_openrouter.sh

# a specific OpenRouter model (e.g. Qwen3-Max) with its own BENCH_ROOT
AGENT6_OR_MODEL=qwen/qwen3-max OPENROUTER_API_KEY=... \
  BENCH_ROOT=/tmp/agent6-qwen-perf bash bench/perf/run_perf_openrouter.sh

# Results land in $BENCH_ROOT (default /tmp/agent6-perf):
#   result_agent6.json
#   result_claude.json
#   _logs/{agent6,claude}/...   raw stdout/stderr + claude.json
#   perf_{agent6,claude}/       worktrees the agents edited
```

The runners are independent: you can run them serially or in
parallel (each gets its own `$BENCH_ROOT/perf_{agent6,claude}/` worktree).

`run_perf_openrouter.sh` is model-agnostic: pick the model with
`AGENT6_OR_MODEL` (default `moonshotai/kimi-k2.6`) and the result label
with `AGENT6_OR_LABEL` (default derived from the model slug). Use a
distinct `BENCH_ROOT` per model so their worktrees and result JSONs do
not collide.

## Budget mapping

claude-code accepts `--max-budget-usd` directly, and agent6's
`[budget].max_usd` is the same currency, so both sides get the identical
dollar cap (default $5, `AGENT6_PERF_MAX_USD`). `max_tokens_fallback`
bounds any call the meter cannot price. Whichever cap fires first ends
the run via the `BudgetExceeded` exit-3 path.

## Why both agents will spend the full budget

- claude-code is open-ended by design: it iterates until the budget cap.
- agent6's `verify_command` is `python tests/submission_tests.py`. That
  exits 0 only when ALL speed tiers (including the < 1363 tier) pass,
  which is effectively unreachable, so agent6's per-step retry loop
  keeps spending tokens on optimization attempts until the budget cap
  fires. This is intentional: the comparison is "given $5, how good a
  cycle count can you get".

## Recent results (guidance only: small N, high variance)

Baseline is 147734 cycles; lower is better. "Speedup" is
baseline ÷ final cycles. All runs are valid (anti-cheat checks passed).

**agent6 + claude-sonnet-4-5 vs claude-code** (~$5 budget):

| runner / model            | runs | best  | worst | cost/run    |
|---------------------------|------|-------|-------|-------------|
| agent6 · sonnet-4.5       | 3    | 5664 (26.1×) | 20016 (7.4×) | $4.9–5.1 |
| claude-code · sonnet-4.5  | 1    | 5829 (25.3×) | —     | $2.42       |

The three agent6 runs are on byte-identical code: 5664 / 8256 / 20016
cycles (7.4×–26.1×). The spread is the worker's stochastic search
path, not a code change, so the result is a range, not a single number.
agent6's best run (26.1×) edged out claude-code's single run (25.3×).

**agent6 + open-weights models via OpenRouter** (single run, equal
$5 budget each via `AGENT6_PERF_MAX_USD=5`):

| model (OpenRouter slug)              | final  | speedup | spent | notes                              |
|--------------------------------------|--------|---------|-------|------------------------------------|
| moonshotai/kimi-k2.6                 | 14389  | 10.3×   | $2.32 | best open-weights; hit input cap   |
| deepseek/deepseek-v3.2               | 57376  | 2.6×    | $0.86 | stopped early (declared done)      |
| qwen/qwen3-coder (480B A35B)         | 120864 | 1.2×    | $0.62 | stopped early                      |
| moonshotai/kimi-k2-thinking          | 147734 | 1.0×    | $1.04 | no usable optimization             |
| qwen/qwen3-coder-30b-a3b-instruct ²  | 147734 | 1.0×    | $0.21 | consumer-grade; no improvement     |

² Runnable on consumer hardware (30B, 3B active MoE; fits a 24GB GPU
at 4-bit).

**agent6 + gpt-5.6-sol (effort max) on the ChatGPT plan, one run each with a
standing goal** (`run_perf_chatgpt.sh`, official `score.sh`):

| date | runner / model | final | speedup | speed tiers | wall | tokens in / out | valid |
|---|---|---|---|---|---|---|---|
| 2026-08-20 | agent6 · gpt-5.6-sol max, standing goal | 1270 | 116.33× | 8 | 87 min | 215k / 116k | yes |
| 2026-09-05 | agent6 · gpt-5.6-sol max, standing goal | 1215 | 121.59× | 8 | 58 min | 303k / 117k | yes |
| 2026-09-05 | the same run resumed for a second leg | 1132 | 130.51× | 8 | +99 min | 3.76M / 94k | yes |

Plan-metered, $0 (the second run: 10 points of the plan for its first leg
and 7 for the resumed one, each leg ending at the 200-iteration cap); the
best of every prior run here is 26.1×, and the upstream tests' eleven-hour
Opus 4.5 mark is 1487.

> **Update (per-call output cap, 32768 → 65536).** Three fresh kimi-k2.6
> runs all landed at 1.0×, but with two distinct failure modes, both
> diagnosed turn-by-turn:
> - At the old 32768 metric cap, ~30% of turns ended `stop_reason="length"`:
>   kimi's reasoning ate the whole per-call budget and the turn closed
>   *before* it could emit a tool call. The run made 1 edit total.
> - Raising `metric_task_max_tokens` to 65536 fixed that (truncation
>   30% → 5%, 15 edits), so kimi now actually attempts the optimization,
>   but its edits *broke correctness* (0 passing verifies → nothing kept).
>
> So the cap bump is a real fix for the truncation waste (worth keeping for
> any reason-heavy worker), but kimi's *capability* on this kernel is the
> next wall, and it's high-variance: the 14389 above remains its best
> sample, not a number any single run reproduces.

> **Reasoning suppression on starvation-recovery turns (N=8, kimi-k2.6).**
> Forcing `reasoning` off for the turn after a starved one scored WORSE than
> leaving it on: 25% vs ~38% win-rate, best 1.50× vs 7.76×. K2.6's large
> speedups come *from* reasoning, so suppression trades the occasional big win
> for reliable-but-mediocre output. agent6 therefore does no automatic
> loop-level suppression; `thinking = "off"` stays an explicit operator knob.

Takeaways, with the small-N / high-variance caveat firmly attached:

- **The frontier gap is large.** Sonnet-4.5 reaches 7–26× and
  claude-code 25×; the best open-weights model (Kimi K2.6) managed
  ~10× on its strong run, and the rest landed at 1–2.6×.
- **Most open-weights runs quit early.** Four of five declared the task
  "done" or ran out of useful ideas well under the $5 cap (spending
  $0.21–$1.04), rather than iterating to the budget like sonnet does.
  Lower spend here reflects *giving up sooner*, not efficiency.
- **A dollar budget is not a dollar floor.** agent6 stops when the
  worker calls `finish_session`; the cap only bounds the *maximum* spend.
- **Reasoning-heavy models need an output-weighted budget.** Kimi is
  output-heavy, so the default 5:1 input:output split (see
  `INPUT_TO_OUTPUT_RATIO_FOR_USD_BUDGET` in `src/agent6/budget.py`)
  starved it: it hit the output ceiling at ~$1 with little progress.
  The 10.3× figure above is a re-run with output-weighted token caps
  (`AGENT6_PERF_MAX_IN=1500000 AGENT6_PERF_MAX_OUT=1150000`,
  `AGENT6_PERF_MAX_USD=0`) so it could actually use the compute.

Do not cite any single number here as a headline. Re-run the harness
to measure for your own model/budget combination.

