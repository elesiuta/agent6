# Core-agent experiments: findings

Two thrusts on the agent6 core loop, front-loaded task decomposition and
keep-last-K-verbatim compaction, plus one opportunistic provider fix found
along the way. All numbers from the `bench/coreagent` harness (see README):
multi-component, hidden-grader-scored tasks; `score` = fraction of grader cases
passing (partial credit, low variance); `solved` = score == 1.0.

Adoption rule: strictly-better-everywhere → default; helps-some/hurts-others →
off-by-default config knob; helps-nowhere → scrap (no cruft "just in case").

## Thrust 2: task decomposition (`[prompt].decompose`) → SHIPPED as off-by-default knob

**Hypothesis:** small models finish multi-part tasks better when forced to break
the task into DAG subtasks first and work one at a time (the existing
surface-current-task + finish-gate machinery then walks the frontier).

**Result: helps a small model with headroom on every task; pure overhead on
capable models. → off-by-default config knob.**

Δscore (decompose − baseline), by model × task (mistral at n=10 where deepened):

| model | textkit | rpn | ledger | bugs | verdict |
|---|---|---|---|---|---|
| **mistral-small-3.2-24b** | **+0.53** (n=10) | **+0.13** (n=10) | +0.18 (n=4) | -0.00 (n=10) | win on implement-many |
| qwen3-coder-30b | 0.00 | 0.00 | 0.00 | 0.00 | ceiling, overhead |
| qwen3.6-35b | 0.00 | 0.00 | n/a | n/a | ceiling, overhead |
| claude-haiku-4-5 | 0.00 | 0.00 | 0.00 | 0.00 | ceiling, overhead |

- **mistral-small-3.2-24b** is the one model that prematurely finishes / drops
  components. Decompose + the finish-gate force it to address every component →
  large robust win on **textkit** (5 independent functions; baseline 0.18 → 0.71,
  ±0.12, n=10) and a solid win on **rpn** (0.85 → 0.98). On rpn it also cut
  iterations (~31 → 22, n=10): the finish-gate keeps it on-task instead of
  thrashing.
- The win is mechanism-specific, not universal across task TYPE: on **bugs**
  (a debug task) decompose shows NO reliable win; n=10 mean is flat (-0.00), and
  that average hides two disagreeing batches (deep n=6 −0.14, screen n=4 +0.21),
  i.e. high variance, not a clean effect either way. Fixing 7 seeded bugs isn't a
  "forgot a component" failure, so up-front decomposition has no clear handle.
  Decompose helps where the failure mode is dropping or under-finishing
  INDEPENDENT components, which is exactly its design intent.
- **Capable models** (qwen3-coder-30b, qwen3.6-35b, haiku) sit at the score
  ceiling both ways; decompose only adds overhead. haiku is the cleanest demo:
  score 1.00 to 1.00 but iterations 6 to 29 (rpn), 8 to 29 (bugs): a ~4x tax for
  zero gain, because it dutifully creates 5-8 subtasks for work it would have done
  directly. This 2-4x overhead is exactly why decompose must default off.
- `solved` (full credit) moves less than `score`: decompose reliably gets MORE
  components right, but a weak model still misses some edge cases.

**Decision:** off-by-default `[prompt].decompose`. Not a default (overhead on
capable models rules out strictly-better); not scrapped (mistral's consistent
wins rule out helps-nowhere). Documented in docs/config.md, visible in
`config show`. Turn it on when driving a small/open model on a multi-component
implementation task. (Refinement worth noting: the original `<dag-rules>` block
already nudged toward decomposition; making it the explicit first step plus the
finish-gate is what converts mistral's premature finishes into full coverage.)

## Thrust 1: keep-last-K-verbatim compaction (`[context].keep_recent_chars`) → SCRAPPED

**Hypothesis (COMPACTION_RESEARCH.md top idea):** agent6's tier-2 restart to
`[task, summary]` is the field's most aggressive choice; every other agent
(Anthropic, Codex, Cursor, LangChain) keeps a recent verbatim tail. Keeping the
last K balanced turns should cut post-compaction re-reads and cold-restart stalls.

**Result: structurally inert in agent6's compaction dynamics → scrapped.**

I implemented it cleanly (a `select_verbatim_tail` helper + a hybrid
`_summarise_and_restart`, `keep_recent_chars=0` = the historical hard restart,
6 unit tests, default-off). Then I measured it, and it does nothing:

- agent6 compaction is **two tiers**: tier-1 elides oldest tool_results (fires
  constantly, keeps context bounded); tier-2 summarises + restarts (what
  `keep_recent_chars` changes). `keep_recent_chars` touches ONLY tier-2.
- **Tier-2 almost never fires.** Because tier-1 keeps the context bounded *below*
  the tier-2 trigger, tier-2 only fires on pathologically long runs (a 100+-turn
  weak-model spiral), and there it is confounded with the model's own variance.
  In a forced-compaction A/B, kimi on the `needle` task (read five ~6k-char ref
  files, retain all five buried rules), aggressive 18k/36k thresholds, n=6 per
  condition, tier-2 fired 0 times in all 18 runs. keep_recent_chars was
  completely inactive; the hard-vs-hybrid score gap (0.91 vs 0.96) is noise (the
  hybrid even had MORE re-reads, 22 vs 15).
- **The real compaction cost is tier-1, which keep-last-K doesn't address.** Tight
  tier-1 elision dropped the ref files and the model re-read them 12-43x per run,
  knocking kimi from 1.00 (no compaction) to ~0.91. That is a tier-1 phenomenon,
  and only at artificially tight thresholds; at the shipped adaptive thresholds
  (~45%/80% of a 128k-256k window) neither tier fires on tasks this size.
- agent6's durable **task DAG + restart notice + surface-current-task** already
  carry task state across a restart, which is the continuity keep-last-K provides
  for memoryless chat agents. agent6 isn't memoryless, so the premise doesn't hold.

**Decision:** scrap (reverted from the tree; implementation preserved in
`scratchpad/thrust1_keeplastk_SCRAPPED.diff` and described here). No measurable
benefit in any regime I could produce → no cruft.

**Where the real compaction lever is (future work):** the cost lives in **tier-1**
(elision causing re-reads on long runs) and in summary fidelity, not in the tier-2
tail. The high-value, A/B-able candidates from the research are tier-1 salience-keep
(don't elide a file the worker is still editing) and a deterministic facts ledger
(record `path:line` + verify outcomes, no LLM call); see COMPACTION_RESEARCH.md
#1/#6. Both need a genuinely long-horizon benchmark to validate (these short
test-graded tasks don't trigger realistic compaction); building that is the right
next step before touching compaction again.

## Opportunistic fix: Gemini/Gemma `tool_code` tool calls

gemma-3-27b scored ~0 on every task: it emits tool calls as a fenced
```` ```tool_code\n[read_file(path='spec.md')]\n``` ```` block (Gemini family) in
`content` with empty native `tool_calls`, which agent6 didn't parse →
`silent_finish` after 1 iteration. Fix: parse that block with `ast` (never
executed) in `providers/openai.py` `_coerce_text_tool_calls`, the same recovery
path as the Qwen `<function=>` XML form. **Live-validated:** post-fix, gemma parses
a 5-call ```tool_code add_task plan and drives the DAG (0 → real engagement). It
still fails the tasks, but for unrelated reasons: heavy 429 rate-limiting on
OpenRouter's free DeepInfra backend, and an occasional UNFENCED `[fn(...)]` list
(a known remaining limit; the fenced form is the documented, unambiguous one).
A strict improvement for the whole Gemini/Gemma family. Shipped with 3 regression
tests.

## Reproduce

```bash
cd bench/coreagent
python3 run.py --model <m> --provider <p> --tasks textkit,rpn,ledger,bugs \
    --conditions baseline,decompose --reps 4 --parallel 6 --label myrun
python3 stats.py results/myrun.jsonl
```
Result JSONL for every experiment is committed under `results/`.

## Thrust 3: skills & prompt-style engineering (2026-07-10 campaign)

The skills-subsystem campaign: does prompt-level style/skill content change
small-model behavior in the agent loop, and which delivery mechanism works?
Every wave carried a positive control (a MOOSE style-marker instruction sent
through the channel under test, delivery byte-verified in the provider
transcripts), so treatment delivery is established independently of any
behavioral outcome. Models: qwen3-coder-30b-a3b, mistral-small-3.2-24b (plus
kimi-k2.7-code / glm-5.2 spot checks). Results JSONL committed alongside.

### Positive controls: style riders are inert in the agent loop

A trivially detectable instruction ("end every prose reply with MOOSE") got
ZERO compliance from both small models in every channel: appended to the
system base, prepended to it, and (qwen) in the user prompt (0/12, 0/13,
0/13, 0/27, 0/30 prose turns; occurrences in transcripts equal injected
copies exactly). mistral obeyed only via the USER prompt and then
overcomplied: 128 emissions and score 0.0 vs 0.93 baseline (n=1). kimi-k2.7
also ignored it; glm-5.2 emitted it once. Style riders in agentic loops are
ineffective across tiers, and counterproductive in the one case where they
did bind.

### H1 caveman-style terseness → measured null, nothing shipped

3 arms (baseline / one-line "be concise" / ~180-word distilled caveman
ruleset with auto-clarity carve-out), 2 tasks x 2 models x n=3, style
compliance measured from transcripts, never assumed:

- qwen: narration-opener rate 0.94-0.96 in ALL arms incl. baseline;
  chars/prose-turn 106-126 (overlapping); no cost delta. The ruleset moved
  nothing the one-liner didn't, and neither moved anything.
- mistral: baseline narration already ~0; terse arms show no reduction.
- kimi-k2.7: baseline emits ZERO prose between tool calls; nothing to
  compress.
- The output surface this class of skill targets (verbose prose) barely
  exists in agent6's loop.
  caveman's own 65% number comes from single-turn chat, a different regime.
  The dollars here go to iteration SPIRALS (40-70+ iters on failed mistral
  runs), which loop discipline (nudges/plateau/decompose) already targets.

### H3 index distraction → mistral-specific fragility, mostly length

Irrelevant 14-skill index (~1.3KB) vs length-matched neutral padding vs
baseline, mistral textkit n=6 each: baseline 0.644 / $0.056 / 39.8 iters;
padding 0.444 / $0.095 / 52.3; index 0.350 / $0.092 / 54.5. Padding
reproduces most of the harm → mistral-small is fragile to ANY system-prompt
addition on this task (bimodal spiral failures). The index-specific residue
is subtask SPAM (16.2 vs 6.2-7.2 without the index), and the spam titles
never mention the skills. qwen: flat everywhere (index is free). Guidance
shipped in docs/config.md: keep the index small on weak models.

### H2 delivery mechanism → all channels deliver; passive index never fires

Same systematic-debugging skill via baked-file vs [skills.state] "always" vs
index+use_skill, bugs task, both models, n=3, all arms on one agent6 build:
scores flat (task at ceiling), delivery byte-verified for baked and always,
and the on-demand arm's index present in every transcript, but NEITHER
model ever called use_skill organically (0 calls in 6 runs, despite the
skill description matching the task). Consistent with superpowers
requiring an unconditional session-start bootstrap injection in Claude Code
rather than relying on index-triggered activation. Consequence (documented): on small
models the reliable skill-delivery paths are `always`, `/name`, and
`--skill`; the passive index is a capable-model affordance.

### H4 process-skill value → VOID (no headroom)

`bugs` baselines: qwen 0.988, mistral 0.976-1.0. A debugging-methodology
skill cannot show value on a task the model already solves; needs a harder
debug task before this is testable.

### H5 structural_priors off → environment cannot discriminate

Bench repos are 1-2 files; the priors block is tiny there. No verdict;
default stands. A real-repo task set is the prerequisite.

### Methodology notes

- Wave gate: tool-result round-trip verified per run in the LAST provider
  transcript (every tool_call id answered), grader non-degenerate, usd>0.
  Runs without verified treatment delivery are structurally excluded.
- mistral serving was flaky (two 700-850s provider stalls inflating cost
  variance); treat mistral cost means accordingly.
- the data dir (`XDG_DATA_HOME`) is isolated per run (host-installed skills can never
  leak into an arm); {ROOT}-interpolated conditions + USER_SUFFIX support
  the prompt-file and user-channel arms.

### Addendum: bootstrap activation and capable-model terseness

**skill_bootstrap** (the ecosystem session-bootstrap pattern via
`[skills.state] using-superpowers = "always"`, verbatim, no tool mapping;
bugs, n=3/model, delivery verified in every transcript): qwen made zero
use_skill calls, with scores and cost identical to baseline; the block is
inert bytes there. mistral invoked use_skill in 3/3 runs (first organic
invocations observed), correctly selecting systematic-debugging, after
first re-fetching using-superpowers itself, whose text was already in its
prompt. Outcomes did not improve: mean score 0.64 vs 0.98 baseline, cost
x1.9, one run collapsed, matching the prompt-addition fragility the padding
control isolated. The bootstrap pattern can activate the index on some
small models but did not pay on either; direct delivery (`--skill`, a
targeted `always`) remains the small-model recommendation.

**Capable-model terse check** (kimi-k2.7-code, glm-5.2; textkit; n=4 fresh
+ the earlier n=2/1): scores 1.0 in all arms. kimi tokens_out 3901+/-2851
baseline vs 2566+/-1997 terse (direction consistent with the n=2
observation; intervals overlap; the reduction, if real, is in reasoning
length, as kimi emits almost no prose in either arm). glm complied with the
style rules (its occasional narration turn disappears) but per-run cost is
already ~$0.008, leaving nothing material to save. No adoption action from
either result.

## Thrust 4: competitive campaign (2026-07-11)

Goal: measure agent6 against aider, opencode, and Claude Code on shared
models, resolve SWE-bench Verified at a fixed budget, and fix what the
measurements surfaced. Ten harness/agent defects were found and fixed in the
process; each is its own conventional commit with the measurement that
motivated it.

### SWE-bench Verified (random-12 subset, $1/instance, official evaluator)

| model | resolved | notes |
|---|--:|---|
| claude-sonnet-5 | 7/12 (58.3%) | 2 instances rescued by the token-cap fix |
| claude-haiku-4-5 | 6/12 (50.0%) | ~1/7th sonnet's cost per instance |
| kimi-k2.7-code | 5/12 (41.7%) | was 1/12 before the fix stack below |
| claude-opus-4-8 | 1/2 | smoke only (cost) |

kimi's initial 1/12 decomposed into three failure modes, all diagnosable
only after the containers exported agent state: (a) the model answered the
problem statement in prose at iteration 2 and the loop accepted it as an
implicit finish (fixed: early prose finishes on an untouched tree are
bounced back to the tools, at most twice, gated to the first iterations);
(b) runaway tool arguments (a 117KB grep pattern, one alternation repeated
until the output-token ceiling truncated the JSON) looped on generic
"resend" feedback until timeout (fixed: the error now names the truncation
and directs a much smaller call); (c) the model is simply slow (12-18 tool
calls per 20 minutes), so the sweep timeout was raised to 2400s with the $1
budget cap unchanged as the real limiter.

### Head-to-head (bench/agents, shared models, model-keyed run dirs)

Go tasks: agent6 is competitive on wall and cost (kvstore-debug with kimi:
30s/$0.015 vs Claude Code with haiku 26s/$0.055). rust-ratelimit is the
one measured gap: agent6 spends 202-350s and $0.42-0.88 where Claude Code
spends 23-44s and $0.04-0.19; the verify-after-every-edit discipline
multiplies cargo build time. Documented as future work: compile-cost-aware
verify cadence. agent6 cost cells use the agent's own accounting (the
key-usage delta method requires an exclusive key; overlapping waves
invalidated those cells).

### No-progress guard (three stages, each measured)

Detection: fired on exactly the doomed runs (77-iteration score-0 spirals)
across 14 valid runs and never on a healthy one. Nudges alone did not
rescue any doomed run. The added stop stage ends such runs as resumable
`no_progress` at 40/59 iterations instead of the cap, saving roughly a
third of a doomed run's cost with no effect on healthy runs.

### Native spec-recheck gate: measured null-to-negative, kept off

The debugging-skill wins on eventflow (haiku +0.053 with variance
collapsing to zero at n=6, mistral skill_always +0.16 at n=8, qwen +0.025)
motivated distilling the mechanism into a one-turn finish bounce. The A/B
(three models, n=6/arm, textkit cost control) rejected it: no score gain
beyond noise, a drop on mistral-small, +38-88% cost. The methodology
content of the skill, not the reminder to re-check, carries the effect.
`[workflow].spec_recheck_on_finish` stays off and is a removal candidate.

### Infrastructure defects fixed along the way

git-less agents-comparison runs; agent config passed via --config; run
dirs keyed by model; clean-worktree gate disabled inside the throwaway
bench repo; SWE-bench containers pinned to the orchestrator's wheel; token
caps turned into loose backstops once Anthropic became priceable; curator
connect made condition-based (liveness probe) after fixed 5s deadlines
killed 7/8 runs under host load; strict-jail /proc mounted before the
old-root detach (was silently empty), Landlock granted read on the
jail-private procfs, jail /tmp raised 64m to 1g (go's build cache lives
under HOME there); state exported from SWE-bench containers; root-owned
run dirs cleaned with privilege; header-less bench conditions extend
[workflow].

## Thrust 5: ultracode round: failure mining + general robustness (2026-07-11)

A multi-agent mining pass over every unresolved SWE-bench/rust run, followed
by general (no-tool-specific) loop-robustness fixes on what it surfaced.

### Mining verdict: the ceiling is mostly model capability

34 agents diagnosed 24 unresolved runs against gold patches, adversarially
verified. Of 19 genuine failures (5 were mislabeled resolved runs from a
stale-report join): 11 (58%) are pure hidden-test near-misses -- the model
patched the right file with a plausible fix discriminated only by a hidden
test -- which agent6 cannot fix; 4 were kimi degeneracy spirals (agent6-
actionable for cost); 3 were a dead-verify cluster; 1 a cutoff. For Claude-
tier models the binding constraint is capability, not harness friction. This
sets the expectation: harness work bounds waste and catastrophic outcomes,
it does not lift a capable model's resolve rate much.

### General fixes shipped (all TDD, no tool enumeration)

- no-progress guard defers to metric runs (a regression: it would have
  truncated a budgeted optimization search and discarded a banked result).
- compaction placeholders stop instructing an identical re-call (contradicted
  the guard).
- dedupe of back-to-back identical tool results (a re-read spiral grew context
  to 125K tokens).
- tool-error spiral guard: nudge/escalate/stop on repeated identical tool
  errors (the verify guard only covered verify failures).
- verify-broken detection: a verify that exited instantly without running its
  tests (runner absent) is flagged, not passed to the model as a real red.
- sandbox-reachability diagnostic: when a run_command spirals on a binary that
  exists on the host, name it as a sandbox-reachability problem and how to fix
  (install / --dangerously-disable-sandbox / extra_read_paths). The sole
  signal is host-existence (shutil.which) of the exact failing binary -- no
  tool list -- so it covers rustup/pyenv/nvm/any proxy tool generally.

### Spiral-guard thresholds (the evidence behind the constants)

`workflows/_nudges.py` carries the thresholds; the observations that set them:

- **No-progress (verify) guard**, motivated by mistral-small (2026-07-11):
  nine consecutive verify failures with the IDENTICAL normalized error while
  the worker kept editing one file, burning a third of the run's budget on one
  failure. Third stage measured on the guard2 waves (n=14): the detector fired
  on exactly the doomed runs and never on a healthy one, but nudged runs still
  burned to the iteration cap at score 0 -- so ten consecutive identical
  failures (both nudges delivered and unheeded) is past any observed recovery,
  and the run stops honestly. Thresholds: nudge 4, escalate 7, stop 10.
- **Tool-error spiral guard**, observed on SWE-bench: kimi re-issuing
  malformed grep calls until the run timed out. Thresholds: 3 / 5 / 8.
- **Verify-broken detection**, observed on SWE-bench sympy testbeds across
  three models: `python -m pytest` with pytest absent, exit 1 in 0.0s, read by
  the model as a real red.
- **Verify-settled completion**: Kimi K2.6 observed running to 128 iterations
  on a task done at ~45, re-running read-only commands after success.
- **Low-budget wrap-up**: Kimi K2.6 observed solving the task, never
  re-running verify, never calling finish_session, and burning the remainder on
  read-only commands (the settled detector cannot engage without a green
  verify).
- **Task finish-gate**: a weak model on a long task observed quitting at
  silent_finish iter 7 with 7 subtasks still open.

### kimi re-measured (before/after, both clean full sweeps)

| metric | original | fresh (all fixes) |
|---|---|---|
| resolved | 5/12 | 7/12 |
| empty-patch spiral-outs | 4 | 2 |
| mean USD/run | ~0.57 | ~0.57 |

The two gained instances (matplotlib-20488, sklearn-11578) are exactly the
ones that were empty-due-to-spiral originally. Empty count 4->2 is the
clean signal (that is the failure the spiral fixes target); the 5->7 resolve
move is within stochastic noise for a ~50% model at n=1/instance, so it is
NOT claimed as caused by the fixes. Aggregate cost is flat -- the fixes cap
the worst-case tail (proven by unit tests on the exact mined spiral
patterns), not the mean, since spirals are a minority of runs. kimi at 7/12
now matches sonnet on this sample.

### Perf take-home revisited: model ceiling, both models

kimi-k2.7: 1.0x (10 edits, none improved cycles). glm-5.2: reasoning-
starvation (99k thinking deltas, 8 tool calls, 0 edits; the starvation nudge
fired but did not break it). Neither can do the take-home; the loop fixes do
not help because the constraint is model-side. Shipped a perf wall-timeout
(glm evaded the budget cap via cheap unbounded reasoning).

### Rust competitive gap re-attributed to the environment

The rust-ratelimit slowness (agent6 burned 32 commands hunting for cargo) was
NOT a verify-cadence problem and NOT an agent6 defect. Root cause: this bench
box's rust is a jail-hostile rustup proxy install (/usr/bin/cargo -> a per-
user rustup needing ~/.rustup, out of jail reach; no system rustc). The jail
is already general-correct: system python3/git and a normal system rust work
in it unmodified. A cargo/rustup special-case was started and reverted; the
general fix is the sandbox-reachability diagnostic above, which turns "agent6
seems broken" into an actionable operator message. No regression on long
tasks (longhorizon stylebook 1.0 on the fixed bin, identical to baseline).

## Thrust 6: a compact operating brief as the system prompt (2026-08-18) → NULL, not shipped

**Hypothesis:** the run-mode `<agent6>` block (task, tool notes, gate rules,
finish) says nothing about HOW to work; a compact brief in the style of the
larger coding agents (understand → change little → check with the gate →
finish cleanly, with the read/search tools named and the edit tools'
contracts spelled out; `prompts/brief.md`, 2.2k chars, replacing the block)
should lift small models. Condition `brief` (`--conditions baseline,brief`),
everything else identical.

**Result: task-dependent, zero net; not shipped.** Two waves (4 reps, then 8
more on the discriminating tasks), pooled:

| model | bugs | ledger | rpn | textkit |
|---|---|---|---|---|
| qwen3.6-35b-a3b (n=12) | −0.02 | **−0.50** (6/12 → 0/12 solved) | +0.17 (0/12 → 2/12) | 0.00 (n=4, floor) |
| mistral-small-3.2-24b (n=12) | −0.01 | **−0.39** (0.81 → 0.42) | **+0.21** (9/12 → 12/12) | −0.04 (n=4) |
| kimi-k2.6 (n=4) | 0 | 0 | 0 | 0 (all 1.0: ceiling) |

Cost per run unchanged (the block is a small share of the context).

- The rpn win replicates on both small models; the ledger loss replicates on
  both. Both are real at n=12 (qwen ledger 6/12 vs 0/12 solved) and they
  cancel; per the adoption rule (helps-some/hurts-others → a knob at most,
  and no knob is worth its config surface for a prompt paragraph) nothing
  ships. Consistent with Thrust 3: prompt prose is not the lever for small
  models.
- **What the ledger loss is.** Every qwen ledger `brief` run went quiet at
  ~10s: the model decides to rewrite the stub file whole, calls `apply_edit`
  with `kind="create"` on the existing file (a validation error), then
  narrates the fix ("let me use apply_patch instead") without calling
  anything, twice, and the went-quiet guard ends the run. The baseline runs
  hit the same wall half the time. mistral instead thrashes small edits
  (an indent-fuzzy miss, then a botched partial edit that leaves a syntax
  error the gate keeps reporting). The mechanism behind both: there is no
  whole-file overwrite. `create` refuses an existing file (a guard against a
  model that thinks the file is new) and `replace` needs the byte-exact old
  text, so "rewrite this stub" has no clean move. That is a tool-shape
  finding, not a prompt one; the next experiment is an explicit
  `apply_edit kind="overwrite"` (the model states the intent, so the create
  guard keeps its job), measured on the same three tasks against these
  baseline arms.
- Methodology: `--reps 8` was needed; the wave-1 n=4 cells disagreed with
  the pooled cells on bugs for qwen (+0.17 → −0.02).

### gpt-5.6-sol effort tiers on the dev slice (2026-08-20)

Three arms of the same 20 instances (bugs and ledger tasks, the ChatGPT
plan, $0), differing only in `[models.worker].effort`:

| arm | n | score | wall per run | iterations | output tokens per run |
|---|---|---|---|---|---|
| low | 20 | 1.000 | 25 s | 4.0 | 886 |
| medium | 20 | 1.000 | 29 s | 4.0 | 1041 |
| max | 20 | 1.000 | 46 s | 4.3 | 1919 |

Every arm solves every instance; max costs +84% wall and +117% output
tokens over low. The slice has no headroom for this model, so effort is
a cost axis here, not a score axis.

