# agent6 x SWE-bench Verified: findings and decisions

Durable record of the SWE-bench work since `dev-0.0.12`: measurements, the
decisions they drove, and the reasoning behind each.

## TL;DR

agent6 resolves SWE-bench Verified instances at a SWE-agent-comparable
budget. Building the benchmark surfaced two agent6 issues (both fixed) and
one efficiency finding (fix shipped, effect not measured).

## The harness (`bench/swebench/`)

- `run_sweep.py`: orchestrator. For each (model, instance): pull the official
  SWE-bench instance image, run agent6 inside it, take `git diff` as the
  prediction. **Source-only**: test-file diffs are stripped so the agent can
  never touch the gold grading tests.
- `in_container.sh`: installs agent6 from a locally-built wheel (`uv tool
  install`, Python 3.14), writes the run config, runs the agent on the problem.
- `score.py`: wraps the unmodified `swebench.harness.run_evaluation`
  (FAIL_TO_PASS + PASS_TO_PASS), so numbers are leaderboard-comparable.
- Sample: random-50 across all 500 (seed 20260623), `sample_50.json`.

**Decision: drive the agent inside SWE-bench's own Docker images, not agent6's
jail.** This benchmarks the agent's capability, not the sandbox (the jail has
its own tests). Non-privileged `sudo docker`; the container is the isolation.

## Results (random-6 pilot, ~$1/instance budget)

| model | resolved | budget notes |
|---|--:|---|
| GLM-5.2 | 3/6 | hard-capped ~$1.2 (priced, USD cap enforced) |
| kimi-k2.6 | 2/6 | hard-capped ~$1.2 |
| sonnet-4-6 | 3/6 | ran UNCAPPED (see issue 1): ~$2 avg, up to $6 |
| opus-4-8 | 0/6 | Anthropic credit exhausted mid-run, not a capability result |

Open models and sonnet resolve at the same rate; the open models cost less.
django instances are the hard ones (none resolved). n=6 is directional only.

**Decision: calibrate budget to SWE-agent's ~$1/instance.** If a capable model
can't resolve on a comparable budget, that's an agent6 issue to fix, not a
reason to raise the budget. Current Anthropic list prices (opus $5/$25, sonnet
$3/$15) verified against the API docs.

## Issue 1: USD-budget enforcement is a no-op for unpriced providers

> Historical record (dev-0.0.12). Superseded: `[budget]` is now `max_usd`
> (metered spend) + `max_tokens_fallback` (unmetered calls only), and the
> USD→token conversion described below no longer exists.

`best_effort_usd_limit` converts to token ceilings via the worker's price
(`config.py:_apply_usd_budget_override`). Anthropic publishes no pricing, so
the conversion returns `None` and the limit silently does nothing; only the
(large) token ceilings bound spend. A SWE-bench run set a $1 limit on sonnet and
it ran to ~$6 on one instance, draining ~$11.80 of credit.

**Decision: warn, don't guess or kill.** Run startup warns when the USD cap
cannot be enforced (unpriced worker), naming the model and pointing at the
token ceilings. No price is guessed and no auto-conversion happens: a wrong
guess could terminate a run mid-task. The `--max-usd` *flag* was already
guarded by `_explicit_usd_flag_error`; this closes the TOML-config path.

**For the benchmark specifically** (an accurate $1 is wanted): the harness
derives token caps from list price directly (`in_container.sh`), which is
accurate when the price is known out-of-band.

## Issue 2: turn (not token) inefficiency

agent6 took 29–77 turns to converge vs SWE-agent's ~15. Diagnosed from a real
transcript (glm/django-14155):

- `read_file` called 13x across 3 distinct files (same test file 8x, same
  source 4x); 21 explore turns to 3 edit turns.
- 84% prompt-cache hit: re-reads are cheap on tokens; the waste is
  round-trip turns. The $6 sonnet blowup was many turns times a large
  accumulated context, not cache failure.

**Root cause (confirmed in source):** run mode's system prompt had no
anti-re-read guidance; that discipline existed only in PLAN mode's
`<be-decisive>`, which the worker never sees.

**Decision: four text-only prompt nudges**, each grounded and
adversarially verified against the source by a review workflow:
1. run-mode `<be-decisive>`: cached context is authoritative; don't re-read
   content already above; post-edit/post-command re-reads still allowed.
2. `<budget-awareness>`: reframe a re-fetch as a full round-trip turn, not
   just token cost.
3. `read_file` description: outline-first + `offset`/`limit` for large files.
4. elision placeholder: stop actively inviting a same-args re-call after
   compaction.

**Rejected** (medium-risk code levers, by the adversarial pass): read-dedup
short-circuits and stale-mtime read caching (can feed stale content into
edits), loop-guard broadening (false positives on legitimate paging), per-turn
budget heartbeats / lowered nudge thresholds (can rush premature `finish_session`).
All four chosen levers are text-only: they cannot corrupt an edit or break a
call path, and current behavior is their floor.

**Validation: measured null, reverted.** An 18-instance/arm A/B (glm-5.2,
old-prompt wheel vs new-prompt wheel, bounded $1) showed no effect:

| | resolved | mean turns (12 fresh instances) |
|---|--:|--:|
| old prompt | 7/18 | 49.0 |
| new prompt (nudges) | 7/18 | 49.6 |

Identical resolve rate, identical mean turns; per-instance turn swings were large
in both directions (django-12125 113 to 32 but django-14053 63 to 118):
noise, no systematic shortening. The 6-instance signal (astropy-13579 50
to 9) was one draw from that noise. The nudges show zero resolve regression
and no measured efficiency gain, so they were reverted (`git revert`, not
history-rewrite; revert-the-revert stays available if a larger measurement
later shows an effect).

**Finding: turn-efficiency is model-bound, not harness-bound.** opus
converges in 12-20 turns and resolves 4/6; glm runs to ~49 mean turns
(several instances 96-118 or `budget_exhausted`) and resolves ~39%,
regardless of the prompt. No prompt nudge closes glm's gap to opus. A
harness-side efficiency win, if one exists, lives in loop/compaction/
tooling, not in run-mode prose.

## Fair bounded-$1 comparison (done)

Re-ran sonnet + opus at the bounded $1 token budget (new wheel) for a clean
comparison; the token-budget enforcement held (sonnet/django-15629 hit
`budget_exhausted` at 29 turns instead of the prior ~$6 runaway) and the
unpriced startup warning fired live. Ordering: opus 4/6, sonnet 3/6, glm
3/6, kimi 2/6 (n=6). opus was the only model to resolve a django instance
(django-14155).

## Verify was broken in every run above

Every resolve rate above was achieved with `run_verify_command`
non-functional: the agent never ran a test. The benchmark wheel was built locally (`uv build`)
without `AGENT6_JAIL_TARGET=musl`, so the jail binary linked against this VM's
glibc 2.39 and could not exec in the glibc-2.35 containers (`GLIBC_2.39 not
found`). The numbers above are lower bounds, achieved without test feedback;
verify-enabled rates measured later are higher.

Fixing it was a cascade, each layer masked by the prior one:
1. **glibc**: rebuild with `AGENT6_JAIL_TARGET=musl` (static binary; CI
   already does this, local builds must too).
2. **verify inference hardcoded `.venv/bin/python`**: an agent6 bug, broke
   verify in any container/system-python env. Fixed: fall back to `python3`
   on PATH when no `.venv`.
3. **jail PATH `/usr/bin:/bin`** excludes the conda interpreter; the harness
   uses its absolute path.
4. **jail couldn't exec the interpreter in-container**: strict bind-mounts
   extra_read_paths at `/ro<src>` (real path denied); hardened denied child
   exec entirely in the SWE-bench image. See the sandbox matrix below.

## Sandbox usability matrix

The benchmark forced agent6's sandbox to work in real container setups.
Validated empirically (`agent6 check sandbox` + a direct jail-exec probe):

| environment | effective auto | `check sandbox` | resolution |
|---|---|---|---|
| **unprivileged docker** | hardened | FAIL (etc-write escaped) | unsandboxed, new opt-in |
| **privileged docker** | strict | all probes pass | strict works |
| **podman rootless** | strict | all probes pass | strict works |

**New: careful unsandboxed opt-in.** `profile = "none"` runs the agent
unsandboxed with a loud startup warning. It is self-authorizing (an
operator-only, LLM-unreachable config value); the per-invocation forms are
`--dangerously-disable-sandbox` / `AGENT6_DANGEROUSLY_DISABLE_SANDBOX=1`.
`auto` never resolves to none on Linux: opting out is an explicit, typed
choice. The SWE-bench harness sets `profile = "none"`, and verify works
end-to-end (validated: 0 jail errors, real patches, the agent running tests).

**Minor sandbox findings (maintainer follow-ups):**
- hardened in unprivileged docker FAILED the `/etc`-write boundary probe
  (isolation gap, possibly root-related).
- podman rootless is NOT detected as a container (`in_container=False`); the
  unsandboxed opt-in would need the env override there; strict (the desired
  podman profile) is unaffected.
- strict exposes extra_read_paths at `/ro<src>`, so a granted conda
  interpreter's real path is absent; the `/ro`-prefixed path works.
- non-fatal `/proc` remount EPERM warning under podman strict.

## Capability: it's model + verify, not scaffolding (3 nulls)

Three independent A/Bs came back null or negative on resolve rate:

| lever tested | result |
|---|---|
| run-mode prompt nudges (anti-re-read) | null (7/18 = 7/18), reverted |
| Fugu distinct-model review panel (quorum gate) | null (3/6 = 3/6, 0 gate events) |
| structural priors (hot symbols / outline / co-change) | null/negative (ON 3/6 @ $1.22 vs OFF 4/6 @ $1.68) |

What moved resolve rate: working verify (+2/18, recovered django wins) and
model choice (opus 4/6 at 12-20 turns vs glm ~39% at ~49 turns). The
structural-prior A/B supplies more repo context than aider (ranked hot
symbols, tree-sitter outline, git co-change) and does not help SWE-bench;
upfront context is not the bottleneck. `prompt.structural_priors=false`
gives a leaner prompt with no measured resolve cost.

**Conclusion:** spend on the model and on verify quality, not on
prompt/review/context scaffolding. Escalation (cheap worker, auto-bump to a
strong model on stall) is now built and its mechanism verified, but the sample
below couldn't measure its resolve value (see next section).

## Escalation: built, evaluated twice, REVERTED (no measured value)

`[models.escalation]` (opt-in role) swapped the worker provider to a stronger model
ONE-WAY once a single run STALLED (degenerate loop, or post-edit no-progress, gated on
`ever_edited`). Reverted after two evals showed no value and a feasibility
check ruled out the only variant with a plausible payoff.

**Eval 1** (qwen3.6-27b -> glm-5.2, 6 instances, verify on, OpenRouter): all three
arms (cheap-alone / escalate / strong-alone) resolve the IDENTICAL set 3/6
{astropy-13579, pytest-7205, sympy-19346}. Escalation fired on 4/6 (glm calls: pytest
14, astropy-14365 5, sympy 5, astropy-13579 4) yet the escalate arm is byte-identical
to cheap-alone. No discriminating instance (none where qwen fails but glm succeeds), so
the lever had no room.

**Eval 2** (bigger gap: qwen3-coder-30b-a3b $0.07/M -> opus-4.7 $5/M, same 6): still no
valid discriminator. The lone candidate django-14155 (the only instance an earlier
opus-4.8 cracked) does NOT reproduce -- opus-4.7-ALONE also fails it, and so does the
escalate arm. weak-alone and escalate both score 2/6 on DIFFERENT instances
(weak {astropy-13579, pytest}, escalate {pytest, sympy}); escalate regressed
astropy-13579 to an empty patch. That swap is the cheap worker's run-to-run
nondeterminism, not a strong-model conversion. Net fail-to-resolve
conversions across both evals: zero.

**Why reverted, not kept-off-by-default:**
- **No discriminator, and the gate blocks the one case that would matter.** The
  `ever_edited` gate suppresses escalation on exactly the never-edited / empty-patch
  failures (django-14155, django-15629) where a strong model would most plausibly help.
- **The variant with a plausible payoff (long plans that de-escalate per
  subtask) is not cleanly feasible in a single run.** No loop-visible subtask boundary exists: the task
  DAG (`add_task`/`set_cursor`) is optional, self-managed worker bookkeeping the loop
  never reads for control flow; `set_cursor` is unreliable (workers skip/late/never-
  clear it), so resetting on a cursor move would mis-reset mid-task and re-trigger the
  escalate->stall->escalate cycle the one-way latch avoids. Redesign = HIGH complexity
  across 5+ files incl. resume-state persistence.
- **That win already exists in machine mode.** Each state runs as its own subprocess
  with its own per-state `model`/`provider` (`machine_cmds.py` spawns `machine_agent`
  per state; `engine.py` selects per-state model). A long job modeled as a DAG/state
  machine gets per-subtask model tiering for FREE -- assign a cheap model to easy states
  and a strong one to the hard state. That IS "escalate the hard subtask, de-escalate
  the next," with a true process boundary and no mid-task mis-reset risk.

Reverted, not rewritten, so the attempt stays visible. Revive only if a
discriminating-sample eval (cheap reliably fails, strong reliably resolves)
shows a cost-per-resolve win, redesigned around the `ever_edited` gap.

## Open / next

- **Re-run the benchmark WITH verify** (`profile = "none"`): done; the
  campaign section below supersedes the blind numbers above.
- **Per-subtask model tiering: use machine mode, not escalation.** Each
  agent state takes its own `model`/`provider`, which subsumes a strong
  model for the hard subtask with a real process boundary.
- Deeper turn-efficiency (if pursued): loop/compaction/tooling, replicated
  resolve-rate; prompt nudges were a measured null.

## Campaign: the v0.0.22 -> v0.1.0 program (kimi-k3)

Full program on `sample_50.json` (held-out eval; NEVER tuned on) with all
tuning on `dev_slice_25.json` (fixed seeded 25 of the 450 non-eval
instances). Official evaluator throughout; predictions source-only.

### Result (eval-50, official scorer)

| build | resolved | empty | wrong | $/solve |
|---|---|---|---|---|
| baseline (released wheel, old base) | 31/50 = 62.0% | 8 | 11 | $0.59 |
| final (minimal base, guard, fast verify, veto seat) | 32/50 = 64.0% | 1 | 17 | $0.58 |

The +2pp is within run variance (+/-2 resolves observed on identical
configs). The structural change is real: empty patches 8 -> 1 (attempt
rate 84% -> 98%). Recovered attempts on eval-hard instances mostly became
wrong patches, not resolves; wrong-patch is now the whole frontier.

### Same-slice head-to-head (dev_slice_25, same model, same materials)

| agent | resolved |
|---|---|
| mini-swe-agent 2.4.6 | 23/25 = 92% |
| pi (@earendil-works/pi-coding-agent) | 21/25 = 84% |
| agent6 final build + veto seat | 21/25 = 84% |
| agent6 pre-fix build | 19/25 = 76% |

Dev slice is easier than the full set (all agents score far above their
full-set numbers); cross-slice comparison is invalid. Published k3
figures elsewhere: mini-swe-agent 67.3, KimiCode 67.5, Claude Code 73.7
(full-500, vendor tables).

### What moved the needle and what did not

- Moved: a stagnation notice (wall clock with zero edits and zero
  verifies; recall spirals make 3-10 calls and die by timeout, invisible
  to call-count guards) and a fast verify (`-x`, parallel django, 240s
  cap; one 600s full-suite verify had eaten half a run's budget).
  Together: every prior empty-patch class eliminated on both slices.
- Did not: prompt opinion text. A 4.3k-char base and a 1.0k-char
  mechanics-only base resolve the same (the one measured-positive
  behavioural line moved to harness scaffolding); a same-model veto
  seat approves its own contract misses (17 wrong patches passed).
- Meter validated exact against provider billing (0.0% error, five runs).

### Harness disclosure (for reproduction)

- Verify: auto-detected per repo; pytest `-q -x`, django
  `runtests.py --parallel 2`; `verify_timeout_s = 240`.
- Instance scaffolding: an AGENTS.md committed pre-base with four lines
  steering derivation over upstream-fix recall (k3's recall attempts
  spiral; successful recall would raise scores, so this trades ceiling
  for reliability). Same file planted for every agent compared.
- Review: one `correctness` seat, same model, `decision = "veto"`,
  before finish, spend inside the same $1 cap.
- Caps: $1/instance, 1200s wall, conc 1-2 on 4 cores.
- Anti-cheat: no benchmark detection in product code; test-file diffs
  stripped from predictions; unmodified official scorer; dev/eval split
  registered in-repo before any tuning.

### Open directions

Wrong-patch (contract misses against hidden gold tests) is the frontier:
distinct-model review panels, contract search (find how the existing
suite pins the surface), and stronger models (harness is model-agnostic;
vendor-scaffold co-training is the competition's edge).

### Coverage total across all draws (2026-08-21)

310 distinct SWE-bench Verified instances scored (official evaluator):
238/310 = 76.8% resolved [71.8, 81.1]. By condition: verify-on 82/110
= 74.5%; gateless (v2 79/100 + v3 77/100) pooled 156/200 = 78.0%
[71.8, 83.2]. The v3 tranche completed at 77/100 [67.8, 84.2] with 0
empty preds and 0 errors across both waves -- the pull-failure guard
and the inline image prune held the counts clean end to end. Reference
full-500: mini-swe-agent 67.3, KimiCode 67.5, Claude Code 73.7.

### Condition-v2 tranche (gateless, random-100, official scorer, 2026-08-21)

The A/B's canonical condition (0.0.27 wheel with reasoning replay,
AGENT6_SB_VERIFY=none) on 100 fresh instances (seed 20260823, disjoint
from the earlier draws): 79/100 = 79.0% resolved, Wilson95
[70.0, 85.8]; empty=1 (a real DNF), err=0.

Cross-condition (disjoint instance sets, so a summary not a paired
test): verify-on 82/110 = 74.5% [65.7, 81.8]; gateless-v2 79/100 =
79.0% [70.0, 85.8]. Directions agree with the paired A/B (gateless
does not cost resolve rate and runs 3-4x faster). Total coverage
across all draws: 161/210 = 76.7% [70.5, 81.9].

Data-integrity note: 6 of this tranche's initial empties were Docker
Hub 429 PULL-FAILURES, not model DNFs (a full /mnt/bench forced
re-pulls). Purged and rerun before scoring; the harness now writes no
pred on a failed pull (status pull_failed) so this cannot recur.

### Paired A/B: verify shape, heal ladder, reasoning replay (2026-08-21)

Paired 30-instance sample from the scored 110 (15 baseline-resolved +
15 baseline-unresolved, seed registered), official scorer, baseline on
those ids = 15/30 by construction:

| arm | resolved | gained | lost |
|---|---|---|---|
| armN: 0.0.27 wheel (heal ladder + reasoning replay), verify on | 17/30 | 2 | 0 |
| armV: 0.0.26 wheel, GATELESS (AGENT6_SB_VERIFY=none) | 19/30 | 4 | 0 |

- Reasoning replay engaged in 29/29 armN runs (encrypted items present
  in every snapshot); the heal ladder fired 0 times in 29 (base miss
  rate ~7% of runs makes n=30 underpowered for it; its guard is the
  zero-regression row plus unit pins).
- armV's gains include the twice-DNF sympy-13031; its wall clock ran
  ~3-4x faster than verify-on arms (no 240s timeout stacks), matching
  the transcript mining (79/103 verify calls were timeouts).
- Neither arm lost a baseline-resolved instance. Sign-test one-sided
  p: armN 0.25, armV 0.0625; directions positive, n small; the
  wall/cost effect for gateless is large and unambiguous.
- Harness conclusion: coverage sweeps default to AGENT6_SB_VERIFY=none
  (speed + no resolve cost measured); verify-on remains a flag for
  arms that study the gate itself.

### gpt-5.6-sol sweep (random-110, official scorer, 2026-08-21)

Coverage grew to 110 random instances (three seeded draws, same config:
sol at effort medium, $1/instance token cap, auto-detected verify at
240s, wheel 0.0.26): 82/110 = 74.5% resolved, Wilson95 [65.6, 81.8];
empty=1, err=0. Reference full-500 numbers: mini-swe-agent 67.3,
KimiCode 67.5, Claude Code 73.7 (vendor tables; cross-table caveats,
and this is a sample). A spark side-arm stopped at n=11 (2 resolved,
7 empty) when its per-model window bound; smoke-quality only.

### gpt-5.6-sol sweep (random-80, official scorer, 2026-08-20, superseded)

ChatGPT-subscription provider, sol at effort medium, $1/instance token
cap, verify auto-detected with `verify_timeout_s = 240`, conc 2-3.
Two random draws from the 500 (seeds registered in the sample files):

| draw | resolved |
|---|---|
| sample_50 (seed 20260623) | 32/50 = 64.0% |
| slice2, 30 more (seed 20260820) | 27/30 = 90.0% |
| pooled | 59/80 = 73.8% (Wilson 95% [63.2%, 82.1%]) |

empty=2, err=0. The per-draw gap is wide at these n; the pooled
interval is the number to quote. Published full-500 comparisons
(vendor tables): mini-swe-agent 67.3, KimiCode 67.5, Claude Code 73.7;
cross-table comparability caveats apply, and this is a sample, not the
full set. Harness findings from the sweep's transcripts (verify
timeouts dominate wasted wall; containers lack rg; multi-file V4A
patches) are recorded with the fixes in the repo history.

### Cross-model health (dev-slice subsets, same harness)

| model | resolved | empties | note |
|---|---|---|---|
| kimi-k3 (+veto seat) | 21/25 = 84% | 0 | full slice |
| claude-sonnet-5 | 16/20 = 80% | 0 | first 20 |
| claude-opus-5 | 4/4 | 0 | health check, n=4 |

The harness is model-agnostic in behaviour: clean finishes, caching
visible, extraction identical. Found live and fixed: direct-Anthropic
ids were unpriced on a cold cache (the preflight now does a keyless
TTL-gated OpenRouter catalog refresh), and opus consistently proposes a
scoped verify over a 240s-capped full suite (stale_gate each run,
mechanism working as designed).

### Review-seat experiments (dev slice, veto before finish, same $1 cap)

Same-model k3 seat: 21/25, all finishes approved. Cross-model sonnet-5
seat: 22/25, zero vetoes in 22 reviews (+1 is variance; the gate never
fired). A diff-tier reviewer of either family approves the worker's
confident contract misses; catching them likely needs a reviewer that
runs tests or sees more than the diff.

### Failure census over the unresolved instances (2026-08-21)

Union across all scored sol draws: 211/267 distinct instances resolved;
56 unresolved, none with an empty pred (every miss is a WRONG patch, not
a DNF). Newest transcript per unresolved instance, official event log:

- Ends: 51 gate_stale, 3 finish_session, 2 unreadable. django is 26 of
  the 56; matplotlib 8.
- Verify-on arm misses: 12 ended with EVERY verify a 240s timeout (the
  django full suite cannot finish under the cap; the gate never gave a
  verdict). 22 more had at least one completed (red) verify.
- Tool friction: 40/56 runs hit at least one tool error (23 apply_patch
  errors, 17 red run_verify_command calls, 10 run_command). The heal
  ladder fired zero times in these runs (its base rate is ~7% of runs).
- Self-verification: 47/56 ran targeted tests via run_command (median 2
  per run); 9 ran none. Median 12 iterations / 20 tool calls; no
  compactions (context was never the binding constraint).
- Reading: the frontier is contract misses, confirmed at census scale:
  the model edits plausibly, runs a couple of targeted tests, and still
  misses the hidden FAIL_TO_PASS contract. Prompt prose is not implicated
  anywhere in the 56.

### Correction: the "gateless" condition adopted a broken gate

The AGENT6_SB_VERIFY=none arms (v2, v3, armV) started gateless, but
mid-run verify ADOPTION then armed the inferred `python3 -m pytest -q`
inside the container, where /usr/bin/python3 has no pytest: every
verify exited 1 in 0.0s with "No module named pytest". Effects:

- Wall clock and resolve rate match true gateless (the red gate cost
  0.0s per call), so the A/B's speed and resolve conclusions stand.
- Predictions are tree diffs, unaffected by commit gating; scores stand.
- Ends read gate_stale (exit 4) instead of a clean settle, which is
  where the census's gate_stale majority comes from.
- Product defects exposed: an operator could not PIN gatelessness
  (empty verify_command is indistinguishable from unset, so inference
  and adoption always re-armed) -- fixed: `[workflow].verify_infer =
  false` pins a gateless run (no inference, no adoption). The none arm
  switches to it with the first wheel that carries the knob (0.0.27
  rejects unknown keys, and in_container.sh is bind-mounted live into
  each new container, so the harness edit waits for the running tail).
  Adoption also probes only argv[0] on PATH,
  not that the command can run (`python -m mod` with the module absent
  for that interpreter adopted the no-op gate) -- queued with its
  candidate fix shapes.

### apply_patch friction does not discriminate (2026-08-21)

Paired mining, the 56 unresolved vs a seeded 60-run resolved control:
error rates 19% vs 22% of apply_patch calls (23 vs 26 errors), zero
abandonments in either group (every error was followed by a later
successful write). sol speaks V4A almost exclusively (239/243 calls).
Classes: ambiguous context 33/49 (the error names the fix; the model
adds context and recovers, about one turn each), multi-file V4A 11/49
(all from the 0.0.26 wheel; 0.0.27 accepts multi-file and v2/v3 show
zero), misc 5. Conclusion: patch friction is background noise, not a
differentiator; no tool-description A/B is warranted here, and the
uniqueness discipline stays.

### SWE-rebench 2026_03: the fresh-task check (2026-08-22)

agent6 + gpt-5.6-sol (effort medium, $1 token cap, 1200s wall, pinned
gateless, 0.0.28 wheel) on the full 110-instance 2026_03 leaderboard
split (official SWE-rebench fork, prebuilt swerebench images):
**48/110 = 43.6%** resolved, 7 empty (all wall timeouts, one rerun
each), 0 errors. Reference rows under the leaderboard's own scaffold:
gpt-5.6-sol[medium] 62.3, Junie 61.8, Claude Code 60.4, Codex 58.0
(cross-scaffold caveats: their wall/token limits are not published).

The gap is NOT cap pressure: 53 of the 55 wrong-patch runs finished
under their own power (2 wall timeouts), so a limits-raised arm would
not close it. Failure profile of the 55: 22 near-miss (some
FAIL_TO_PASS passing), 30 zero-F2P (contract missed entirely), and 14
broke PASS_TO_PASS -- regressions nothing caught before finish, the
price of gateless on UNFAMILIAR repos that Verified's 12 famous ones
never charged. Spot-checked evals are structurally sound (tests
collect and run; near-misses are genuine).

Reading: Verified overstates us the same way it overstates everyone
(familiar repos reward recall and tuned verify heuristics); fresh
diverse tasks expose contract discovery and regression safety as the
real frontier. This re-motivates the softened-rule/verify-shape A/B
(the 14 P2P-breakers are exactly what a certifying gate catches) and
puts contract search ahead of further Verified coverage.

### Paired A/B: the softened verify rule SHIPPED (2026-08-22)

Hard-30 sample (seed 20260825: 15 baseline-resolved + 15 unresolved
from the scored pool), verify-on, two 0.0.28 wheels differing in ONE
verify-block bullet ("tests only through the gate, never run_command"
vs "targeted run_command tests are fine; the gate certifies"),
official scorer:

| arm | resolved | targeted tests via run_command | gate runs | arm wall |
|---|---|---|---|---|
| never-rule | 17/30 | 103 | 65 | 2:23 |
| softened   | 19/30 | 206 | 40 | 2:01 |

Gained django-11885/13344/14493, lost pytest-7205; sign p=0.31 (n=4),
zero empties both arms. The base arm's 103 targeted tests show the
model already broke the "never" rule; softening legitimizes the
behaviour it was measured doing. Shipped on non-inferior resolve plus
the large behaviour shift and the 16% wall drop (fewer 240s gate
stacks) -- the gateless-default precedent's shape. The prompt change
is one bullet, reverted in one commit if the larger-n picture ever
disagrees.

### The verify rule: advice adds nothing over the fact (2026-08-22)

Third paired hard-30 arm, one bullet apart from the targeted-fine
wheel: the bare world-state bullet ("run_verify_command runs the
operator's gate; a passing run auto-commits the step").

| arm | resolved | self-tests | gate runs | wall |
|---|---|---|---|---|
| never-rule | 17/30 | 103 | 65 | 2:23 |
| targeted-fine | 19/30 | 206 | 40 | 2:01 |
| world-state fact | 19/30 | 177 | 39 | 1:58 |

Identical resolve (aware-vs-fine churn +2/-2, noise), identical gate
load and wall. The permission sentence is inert once the "never" rule
is gone: the model self-tests by default when nothing forbids it.
SHIPPED the bare fact (less-is-a-win on a measured null). First
confirmation of the world-state-everything direction; the full recast
of all model-facing text is the queued P5 arm.
