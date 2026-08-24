# Long-horizon experiments: findings

Numbers from the harness this directory documents (see README); day 1 built
it, day 2 measured the write-side nudges and gist elision, day 3 measured
memory READ-side value on the new orchard leg 3, day 4 took the harness to the
direct Anthropic API (haiku-4.5, sonnet-5) and replicated the gist result on a
second open model (kimi). Runs are real models over OpenRouter or the direct
Anthropic API, hidden-grader partial credit, records in `results/`. Adoption rule
unchanged from bench/coreagent: strictly better → default; helps-some →
knob; helps-nowhere → scrap.

## 1. Tier-1 compaction losses are real, regime-gated, and gist elision recovers most of them

qwen3-coder-30b, stylebook (10-rule retention probe) and relay (6-stage
pipeline), n=3 per cell:

| task | baseline | window32k | window16k |
|------|----------|-----------|-----------|
| stylebook score | 0.921 ± 0.022 | 0.825 ± 0.033 (−0.10) | **0.425 ± 0.202 (−0.50)** |
| stylebook drops / rereads / iters | 0 / 0.3 / 25 | 0.3 / 4.3 / 29 | 65 / 71.7 (all post-drop) / 98 |
| relay score | 0.975 ± 0.006 | 0.956 ± 0.025 (flat) | 0.994 ± 0.006 (flat) |

- **window32k is nearly inert for a tidy reader**: qwen keeps total
  tool-result volume ~50k chars, under the 58k drop threshold; drops fired
  in 1 of 6 wave-1 runs. The shipped adaptive thresholds on big-window
  models never engage on tasks this size (as bench/coreagent predicted).
  Heavy readers do engage there: kimi-k2.6 crossed it on the same task.
- **window16k (a real local-serving default) is a death spiral on
  retention work**: first drop lands mid-reading (tool call ~9), then
  elision → re-read → more pressure → elision, averaging 65 dropped
  results, ~5 tier-2 restarts, 4x the iterations, and half the score. The
  per-rule retention curve shows broad destruction (every rule ≤ 0.67 at
  n=3; in the mildest rep the early-read rules died first: r04 0.14, r01
  0.33, vs late/structural r06/r08/r09 at 1.00). Every redundant read is
  post-drop.
- **relay is score-immune in the same regime** (0.994, +11% iterations;
  one rep paid +58% with a tier-2 restart and still landed 1.0): code on
  disk is cheaply re-readable, spec nuance is not. Compaction taxes
  retention tasks in CORRECTNESS and implementation tasks in EFFICIENCY.
- **Facts-ledger gate (day 1): OPEN, scoped.** A tier-1 mitigation (facts
  ledger or salience-keep) is worth building FOR SMALL-WINDOW DEPLOYMENTS
  and should be A/B'd here under window16k on stylebook (score) + relay
  (iterations). Nothing so far justifies changing behavior for 128k+
  windows. Day 2 built and measured that mitigation; see the gist block
  below.
- Task fairness check: kimi-k2.6 produced a PERFECT 1.0 stylebook (all 12
  components) under window32k with a drop; the ceiling is reachable under
  compaction pressure; it just timed out at 2400s before finish_session
  (103k output tokens of reasoning; see methodology).

**Day 2 (gist elision shipped, gist1-qwen, n=3 per cell):** tier-1 elision
now decays a large read result content → distilled gist placeholder → bare
marker (one batched summariser call per drop event; `context.elision_gists`,
default on). A/B under the same window16k thresholds:

| stylebook (every leg drop-engaged) | gists on | gists off |
|------------------------------------|----------|-----------|
| score | **0.825 ± 0.040** | 0.533 ± 0.264 |
| iterations / usd / wall | 53.7 / $0.069 / 418s | 106.0 / $0.128 / 589s |
| drops / tier-2 restarts | 19.3 / 2.7 | 36.7 / 5.0 |

- Gists recover most of the day-1 damage: window16k stylebook went 0.425
  (day 1, bare markers) → 0.825, against the 0.921 uncompacted baseline.
  The whole retention curve lifts: with gists 6 of 12 components ≥ 0.90;
  without, all 12 ≤ 0.67.
- The failure mode gists remove is the re-read spiral: without them a leg
  re-reads what it lost, regrows context, and drops again (36.7 drops, 5.0
  tier-2 restarts, 106 iterations mean; one rep collapsed to 0.025 via
  silent_finish). With gists all three reps finished in 52–55 iterations
  and the spread collapsed (se 0.264 → 0.040).
- Cost halves ($0.128 → $0.069 per leg) even with the added summariser
  calls: 58 gists written across the wave, 10 demoted to bare markers under
  continued pressure, 0 distiller failures.
- Relay never engages the gist path on this rung (0 gists across all 6
  legs; 5 of 6 had zero drops) and stays flat: 0.987 ± 0.006 vs 0.981
  (excluding one loop_guard kill at iteration 12 with zero drops, which
  carries no A/B information). No-harm check passed.
- **Verdict: `elision_gists` stays default-on.** Strictly better in the
  engaged regime, inert outside it (gists only exist after a drop). This
  closes the facts-ledger gate; a further salience/ledger pass would chase
  the remaining 0.10 to baseline and is not opened.

**Day 4 (kimi replication, kimi-gist, n=3 per cell):** the gist A/B repeats on
a second open model. kimi-k2.6 stylebook under the window16k thresholds:

| stylebook | baseline | window16k (gists on) | window16k_nogist |
|-----------|----------|----------------------|------------------|
| score | 1.000 | 0.983 ± 0.017 | 0.917 ± 0.083 |
| drops / rereads / iters | 0 / 0.7 / 11 | 13.7 / 6.0 / 10 | 438 / 449.7 / 103 |
| usd | $0.29 | $0.37 | $0.82 |

- Same mechanism, sharper efficiency signal. Without gists kimi survives on
  score (0.917; it is more robust than qwen) but pays 10x the iterations (103
  vs 10), 75x the redundant reads (449.7 vs 6.0), and double the cost; the
  bare-marker legs thrash hardest (per-leg drops 186 / 505 / 623, up to 136
  iterations).
- With gists the window16k leg sits within noise of the uncompacted baseline on
  every axis (0.983 vs 1.0, 10 iters, 6 rereads). Second-model confirmation
  that `elision_gists` default-on is strictly better in the engaged regime.

## 2. Nobody writes memories unprompted; FIXED by the write-side nudges

Day 1: across all 46 legs / 2 models (qwen3-coder-30b, kimi-k2.6) spanning
orchard, relay, stylebook: `add_memory` calls = **0**. The `<memories>` block's
read side is proven (fake-provider e2e, 2026-07-06), but the write side was
inert as shipped: models with a discovered non-obvious fact in hand (the
orchard generated-file trap) never record it.

**Day 2 (nudges shipped, mem2-qwen, n=4 per cell):** two loop surfaces now
nudge `add_memory`: an advisory at the first red-to-green verify flip, and a
once-deferred finish_session after such a recovery when nothing was recorded.
Result: `memory_writes` went 0.0 → 0.5–0.8 per leg. In per-run traces the
flip advisory alone converted about half the writers; the finish backstop
caught the rest; decliners finished cleanly on the second call (no bounce
loops). Quality across the 9 stores written: ~8/9 are exactly the durable
trap facts ("data/catalog.tsv is generated from tools/catalog_source.tsv by
gen_catalog.py; shelf = base + (base*margin+50)//100"), 1 is a
confidently-wrong trap-faller rationalization, 0 are task-progress junk.

**Day 2: read-side value was still unmeasured; the 2-leg orchard cannot
measure it by construction.** Leg-1 writers are exactly the reps that
recovered (and so left an easy leg 2); trap-fallers, the population a memory
would help, never went green in leg 1 so never got nudged and wrote nothing.
The baseline-vs-fresh_state leg-1 delta (must be zero in expectation) came
out +0.03 score / 12 iters, the noise floor; the leg-2 deltas
sit inside it. Measuring read-side value needs a THIRD leg that touches the
generator again after leg 2's recovery memory exists.

**Day 3 (leg-3 "clearance" shipped, mem3-qwen, n=6 per cell): read-side
value is REAL, and it lands exactly on the two seeded conventions.** The new
leg re-probes the generated-file rule and the half-up rule with fresh
discriminators, in a spec that deliberately does not point at docs/NOTES.md;
the shipped acceptance test carries no half-cent sku, so a banker's/truncation
implementation goes verify-green and only the hidden grader sees it.

- clearance leg: baseline **0.922 ± 0.038** vs fresh_state **0.750 ± 0.097**
  (dscore −0.172). Same-wave leg-1 calibration delta (identical conditions by
  construction): −0.019, so the leg-3 delta is ~9x this wave's floor.
- Components: rounding **0.78 vs 0.33**, api 1.00 vs 0.83, regen 0.91 vs
  0.78. `trap_edits` 0.0 vs 0.7 per leg: no baseline rep touched a
  generated feed; two fresh reps hand-edited (the worst, with a dead API on
  top, landed 0.375, the no-generator mutant signature). Iterations are
  flat (30.8 vs 31.2): on this task memory buys correctness, not speed.
- Leg 2 same wave: -0.045 with rounding 1.00 vs 0.89; direction consistent,
  still near the floor, as day 2 predicted.
- Mechanism, per rep: 3/6 baseline reps entered leg 3 holding nudged
  memories; all three avoided the trap. What the memory SAYS predicts what
  transfers: the rep whose leg-2 store spelled out "rounding half-up" in
  words scored 1.0; the rep whose store only encoded the catalog formula
  (half-up implicit in `(x+50)//100`) reapplied the generated-file fact but
  truncated the NEW discount computation (rounding 0.0). Record the RULE,
  not the instance, a candidate wording tweak for the write nudges.
- The no-memory baseline reps went 1.0/1.0/0.906, so with n=6 the condition
  delta is carried by the memory readers plus rep luck; the component
  pattern (rounding and trap, nothing else) matches the memory mechanism,
  not generic drift.
- Two believed-done dents in baseline came from a fresh shape: a feed
  written with WRONG values (one rep stored discounts instead of prices;
  and wrote a confidently-wrong memory saying that is the format) while a
  compute-side API kept every probe green. The registers-read-the-feed
  grader catches it; the shipped suite structurally cannot. Second
  confidently-wrong memory observed across waves; `invalidate_memory`
  remains unused.

## 3. add_dependency: first real usage (mistral-small, decompose on)

Day 1: `deps_added = 0` across every qwen/kimi leg, including relay (a strict
6-stage chain, the friendliest shape).

**Day 2 (dep1-mistral, n=3 per cell):** mistral-small-3.2 (prompt.decompose
auto-on from the capability registry) calls it unprompted: `deps_added` 1.7
per leg on orchard-weekend, 0.3 on relay. The edges are sensible
gate-everything-on-investigation fan-outs ("Add weekend_cents column" /
"Implement weekend_price" / "Modify cart_total" each depend on "Investigate
the weekend pricing requirements"). **Verdict: keep the tool**; it is a
weak-model affordance, unused by capable models and free when unused.
Whether the edges improve sequencing is unmeasurable here (mistral's low
scores are capability-bound: 3/3 leg-1 trap falls at exactly 0.889, weekend
0.455 on the rounding discriminator).

## 4. The orchard trap catches real behavior

qwen leg-1 (smoke2): hand-edited the generated `data/catalog.tsv` 3 times,
tested around verify, called finish_session believing it was done; hidden
score 0.889 (exactly the unfixed-seed signature). Leg 2 then went through
verify, which regenerated the file and resurfaced the failure; the agent
root-caused `tools/catalog_source.tsv` properly and scored 1.0 including
the half-up-vs-banker's discriminator (F-310 → 909). kimi never falls in
(8/8 legs 1.0, trap_edits 0); mistral-small falls in 3/3 at exactly the
0.889 signature and never recovers. The task separates exactly the
population the memory experiment needs.

Day 3: the trap re-fires on the SAME model inside one sequence when the
cross-run store is wiped; fresh_state qwen hand-edited the new clearance
feed in 2/6 leg-3 reps (one landed 0.375 with a hand-built feed the build
cannot reproduce), while baseline touched a generated feed in 0/6. Note the
counting rule changed on day 3 (`trap_edits` now matches the edit call's
`path` when present, not the whole args JSON), so day-1/2 raw counts are
not comparable; day-3 leg-1 counts run 4–6 per faller under the new rule
because repeated hand-edit attempts each count.

## 5. Direct Anthropic API: the tier the agent is positioned for

Day 4 pointed the harness at the direct Anthropic API (`--provider anthropic`,
token-capped since Anthropic USD is unpriced here), n=3 per cell.

**Both tiers solve the suite; the memory A/B is null by ceiling.** haiku-4.5
and sonnet-5 both score 1.0 on all three orchard legs in BOTH conditions, so
the leg-3 read-side memory delta that is +0.17 for qwen (finding 2) is +0.00
here. Neither needs the cross-run store to dodge the generated-file / half-up
trap; both rediscover the conventions every leg (fresh_state even runs a hair
fewer iterations). Same story as findings 2 and 3: cross-run memory and
add_dependency are weak-model affordances with real value on qwen and mistral
and no headroom on capable models. stylebook baseline is 1.0 for both (haiku 68
iters $0.53; sonnet 10 iters $0.39).

**Found and fixed a real product bug: the SSE watchdog killed Sonnet on hard
tasks.** sonnet-5 stylebook first ran 0/3, every rep provider_error at 3-4
iterations ("Anthropic SSE stream idle for >45s mid-stream ... upstream
wedged"). Root cause: sonnet-5 does adaptive thinking on by default with
display:omitted, so a hard-task turn thinks >45s emitting only ping heartbeats;
the mid-stream idle watchdog counted a thinking-block start as output and held
it to the tight 45s budget, aborting every long think. Orchard (short thinks)
never tripped it; Opus and Fable would fail the same way. The fix gives a
thinking block a patient idle budget (genuine wedges stay bounded); the
identical cell then runs 3/3 at 1.0 (80/80 cases, 10 iters, $0.39, ~205s).
Before and after are in `results/sb-sonnet.jsonl` and
`results/sb-sonnet-fixed.jsonl`.

Separate latent issue (not fixed this pass): the provider still shapes extended
thinking as `budget_tokens`, which the API removed on sonnet-5 / opus-4.7+ /
fable-5 (a 400 if a thinking level is configured; the bench runs thinking off,
so it is not hit). Migrating the Anthropic thinking config to adaptive +
display:summarized is the proper follow-up and would make the watchdog's job
trivial (real deltas arrive during thinking).

## Methodology notes

- qwen3-coder-30b is the workhorse: $0.03–0.07 and 1–5 min per run.
  kimi-k2.6 spends 15–25 min inside a single reasoning turn when asked to
  write a precise module in one shot (30k+ output tokens before its first
  edit); use it as the slow second model, not the screen.
- Scores move on component curves, not just means: read `stats.py
  --components` before trusting a delta.
- `run_waves.sh` holds the fuller matrix (kimi compaction cells, more
  reps) for when more coverage is wanted (~$10–25).
- The gist1-qwen wave was interrupted by a host OOM (a leg's sandboxed
  process allocated 5.3GB; the box has 8GB and no swap) and resumed under a
  memory-capped systemd unit. Rep ids in that file restart (the nogist
  stylebook cell has two r0), but cells are balanced at n=3 and reps carry
  no seed. Bench waves should cap memory: `MemoryMax` on the unit plus
  `ulimit -v` under it.
- Day 4 added `--provider anthropic` (token-capped, since Anthropic is unpriced
  here) and `--timeout-scale` (kimi stylebook legs run 15-70 min; scale 2.0
  lifts the per-leg timeout so they finish rather than dying at 2400s). The
  anthropic path first surfaced the thinking-watchdog bug in finding 5.

## Cross-session memory campaign: the ledger task and the poisoned probe (2026-08-23)

Harness additions: the 5-leg `ledger` task (a generated command table,
integer cents with half-up rounding, a balanced-posting invariant; legs 2-5
each need a convention leg 1 exposes and their specs do not name), the
`poisoned` condition (the task's stale memories planted with `agent6 memory
add` before leg 2), memory metrics that read the real channel (edits and
reads of files under the repo memory dir, a per-leg file diff with
`poison_touched`), resumable labels, a per-session MemoryMax scope, and
state dirs beside the workdir (agent6 refuses one inside the workspace;
every earlier label would have failed the same way on a current build).

Grader validation (hidden, partial credit): the reference scores 1.0 on
every leg; the untouched seed 0.0 on legs 2-5; a hand-edited
`ledger/_commands.py` loses exactly `regen`; a banker's-rounding mutant
loses exactly `rounding` (split keeps half of it: one case rounds the same
either way).

Smoke on qwen3-coder-30b (one rep each, harness only, not a measurement):
orchard/poisoned ran all three legs with both poison files in place and
untouched (no memory reads or writes; leg 3 rounding 0.0, trap edits 3);
ledger/baseline: fix 1.0, split 0.75 (rounding 0.0: the first convention
miss), then the model corrupted `ledger/ops.py` with a broken docstring
quote and finished over a red gate, so convert/report/import scored 0.0
on an unimportable module. The measurement on the plan model is the next
step: ledger x {baseline, fresh_state, poisoned}, paired reps.
