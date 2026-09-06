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
not close it. Failure profile of the 55 (classes overlap: a run can be
near-miss and broke-P2P, so the counts sum past 55): 22 near-miss (some
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

### Contract census on the rebench misses (2026-08-22)

Every 2026_03 run with a readable event log, by outcome: resolved,
near-miss (some FAIL_TO_PASS passing), zero-F2P (none passing), and
broke-P2P (a PASS_TO_PASS failing; takes precedence over the other two
miss classes). "F2P file" is the test file the hidden tests live in.

| class | n | tool calls | read the F2P file | ran the F2P file | read a test before the first edit |
|---|---|---|---|---|---|
| resolved | 48 | 22.3 | 40/48 | 39/48 | 44/48 |
| near-miss | 15 | 24.4 | 13/15 | 15/15 | 14/15 |
| broke-P2P | 14 | 25.4 | 12/14 | 12/14 | 14/14 |
| zero-F2P | 26 | 25.9 | 18/26 | 15/26 | 25/26 |

How much of the hidden contract exists in the repo at all (an F2P test
is "new" when the hidden test patch adds its function; class
membership here by report alone, overlapping):

| class | F2P tests | newly added | instances with every F2P test new |
|---|---|---|---|
| resolved (48) | 139 | 115 (83%) | 38/48 |
| near-miss (25) | 169 | 120 (71%) | 18/25 |
| broke-P2P (14) | 48 | 39 (81%) | 11/14 |
| zero-F2P (30) | 71 | 68 (96%) | 27/30 |

Reading: the model behaves the same way in every class and already
finds the file the hidden contract lands in; a zero-F2P miss is a
contract that exists only in tests not yet written (96%), so a search
over the existing suite cannot surface it. What is left is deriving
the exact expected behaviour from the issue text, and the 14
broke-P2P regressions a gate catches (this split ran gateless).

### The world-state recast of all run-mode text (2026-08-22)

Fourth paired hard-30 arm: every run-mode model-facing string recast to
facts (base prompt, DAG rules, verify block, git-protect rule, first
message, tool descriptions; nudges separately), compared against the
old best on the same sample per the no-regression rule (no fresh
baseline arm):

| arm | resolved | self-tests | gate runs |
|---|---|---|---|
| never-rule | 17/30 | 103 | 65 |
| targeted-fine | 19/30 | 206 | 40 |
| world-state fact | 19/30 | 177 | 39 |
| world-state recast | 19/30 | 207 | 35 |

Against the fact arm: gained django-13344, lost sympy-13974; zero
empties. Wall is not comparable: the arm was interrupted by a machine
change at 15/30 and finished at --conc 4 on 16 cores. SHIPPED with the
follow-up recast of plan/ask/agent/machine text and the nudges, which
SWE-bench does not exercise; those went out on paired live smokes.

### Tranche 4 and the coverage total (2026-08-22)

100 fresh Verified instances (seed 20260826, drawn from the 177 never
drawn nor scored), gateless, 0.0.28 wheel carrying the world-state
run-mode recast, --conc 6 then 4 on 16 cores, official scorer:
80/100 = 80.0% resolved, Wilson95 [71.1, 86.7]; empty=0, err=0. By
repo: django 40/45, sympy 12/19, sphinx 6/10, scikit-learn 3/5,
matplotlib 5/5, pytest 3/3, pydata 4/4, astropy 2/3, pylint 1/2, the
singletons 4/4.

Coverage across all draws: 318/410 = 77.6% [73.3, 81.3] (verify-on
82/110; gateless v2 79 + v3 77 + t4 80 = 236/300 = 78.7%).

Harness: at this concurrency the pipeline spends two Docker Hub pulls
per instance (the sweep prunes each image, the scorer re-pulls it) and
hit the anonymous window's 429 at ~100 pulls inside an hour; the
pull-failure guard wrote no false empties (75 pull_failed rows, zero
preds). The rest ran as batches of 12 with images kept until scored,
then pruned: ~10 minutes per batch. A scorer killed mid-run leaves its
named `sweb.eval.*` container, and the next run of that run id fails
with a 409 on the same instance; remove the container before
re-scoring.

### Tranche 5, the coverage remainder, and a paired verify-on arm (2026-08-23)

The 77 Verified instances never drawn nor scored (every id without a
prediction or a sample draw), gateless, 0.0.28 wheel, --conc 6 on 16
cores, official scorer: 58/77 = 75.3% resolved, Wilson95 [64.6, 83.6];
empty=0, err=0.

Coverage across all draws: 376/487 = 77.2% [73.3, 80.7] (verify-on
82/110; gateless v2 79 + v3 77 + t4 80 + t5 58 = 294/377 = 78.0%).

The same 77 under verify-on (the finish-gated mode, the rebench
certification cell's config), paired, same wheel and seeds:
58/77 resolved (56 shared; 2 P2P-broken -> resolved, 1 -> unresolved;
1 resolved -> P2P-broken, 1 -> unresolved, 1 empty patch);
PASS_TO_PASS regressions 4 vs 6; 1357 vs 1063 model calls (+28%);
wall per instance 7.0 vs 2.8 min mean (6.5 vs 2.8 median). With the
rebench cell (17/30 vs 16/30, regressions 5 vs 14, +50% calls, 2x
wall): the gate is resolve-rate neutral, cuts collateral damage (10 of
20 regressions across both samples, 1 introduced), and costs 1.3-1.5x
the calls and 2-2.5x the wall time.

Harness: the first pass lost 32 of the 77 to `pull_failed` because
`/mnt/bench` (Docker's root) was full: the layer extract fails and the
guard's message blames a Docker Hub 429 it did not see. `docker image
prune -af` between chains; the redo (run_sweep skips ids with a pred)
ran beside the verify-on arm, 6+6 containers per batch of 12, images
pruned after both scorers.

### The contract-examples step: a measured null (2026-08-23)

One extra worker call before the first edit derives input -> output
examples of the expected behaviour from the task text and shows them to
the model (`[prompt].contract_examples`, off by default). Paired against
the gateless SWE-rebench 2026_03 baseline on the same 110 ids, same
wheel, seeds and caps, official scorer:

| without -> with the step | SWE-rebench 2026_03, 110 ids |
|---|---|
| resolved | 48/110 -> 46/110 (7 gained, 9 lost) |
| model calls, total | 1457 -> 1821 (+25%) |

Gained: azure-search-openai-demo-3025, montepy-933, marshmallow-2925,
pypsa-1653, rapid-mlx-227, ultraplot-696, mtplx-21. Lost: beever-atlas-102,
loguru-1451, pygeoapi-2338, fusesoc-776, click-3239, pandas-64816,
pgmpy-3137, build-1027, sqlglot-7187. The knob and its bench plumbing
were removed.

### Preset smoke: ultra and paranoid on the ChatGPT provider (2026-08-23)

The two review-panel presets (`ultra`: a three-seat panel vetoing the
finish; `paranoid`: five explore-tier seats) on gpt-5.6-sol, the first 5
ids of the hard-30 sample, gateless, official scorer; a smoke that the
panels run on this provider, not an A/B (n=5).

| arm | resolved | ends reviewed by the panel |
|---|---|---|
| ultra, 0.0.28 wheel | 2/5 | 3 of 5 (2 ended "settled", no panel) |
| paranoid, 0.0.28 wheel | 3/5 | 4 of 5 (1 ended "settled", no panel) |
| ultra, fixed tree | 1/5 | 5 of 5 |
| paranoid, fixed tree | 3/5 | 5 of 5 |

A run that commits and goes idle ends "settled" without calling
`finish_session`; on the 0.0.28 wheel that end bypassed the before-finish
panel. The fix routes the settled stop (and a silent finish) through the
same gates a `finish_session` passes: every end in the rerun carries a
panel verdict (6 at `finish_session`, 4 at the settled stop).

### The leaderboard scaffold, compared (2026-08-23)

From swe-rebench.com/about and the SWE-rebench GitHub org: the board
evaluates every model on one fixed minimal ReAct scaffold
(mini-swe-agent), 128K context, default generation hyperparameters,
five runs per problem reporting the mean resolved rate (with SEM and a
separate pass@5), a `submit` command ends the session, and the prompt
forbids modifying or adding tests. No cost or time cap is published.

Against our 48/110 run: the statistic is comparable (their 62.3 is a
mean, not best-of); context is not the difference (ours is larger); the
no-test-edits rule is not the difference (0 of our 110 patches touch a
test file); their runs are uncapped where ours carry $1 and 1200s (7 of
our 110 hit the wall), and a ReAct loop retries freely within its step
budget where our runs finish when the model believes it is done. The
levers this leaves, in evidence order: the finish certification with
returns on the same 110 (the cert30 subset cut regressions 14 -> 5); a
raised-caps arm on a hard sample to price the budget shape; executed
test-first and conventions-mining arms for the zero-F2P class (the
imagined-examples step measured null-to-negative).

### The compute-shape parity arm: a null (2026-08-23)

Whether the board's uncapped scaffold explains the rebench gap: the
gateless baseline config with the caps raised from $1/1200s to $3/2400s
(the $ cap never binds on the subscription, so the variable is wall),
same 0.0.28 wheel, seed 20260828, 30 ids in two registered batches,
official scorer.

| baseline ($1/1200s) -> raised caps ($3/2400s) | resolved |
|---|---|
| 15 baseline-unresolved ids (the 9 wall-timeouts + 6 seeded misses) | 0/15 -> 2/15 |
| 15 baseline-resolved controls (seeded) | 15/15 -> 13/15 |
| net, all 30 | 15 -> 15 |

Every empty is the agent's own wall timeout (11 of 30: 9 hard, 2
controls; run.log markers). The old timeout ids mostly consume the
doubled wall reading without converging, and two controls that resolved
at 1200s wandered to a DNF at 2400s: run-to-run variance in both
directions, zero net movement. The DNF profile is model-side, not a
tool loop: the scikit-learn control made 6 model calls in 2400s (~7 min
of reasoning per call, 9 tool calls total, no re-read spiral) and timed
out mid-analysis. Doubling wall and tripling the cap does not close the
gap; the remaining suspects are the reasoning-latency variance and the
zero-F2P contract class, not budget and not the loop.

### The shipped finish gate on SWE-rebench 2026_03: gateless -> gate-on +5 (2026-08-24)

One run per side, same 110 fresh instances, same wheel/config apart from
the gate: baseline `verify_when = never` (the previous behaviour), arm
`verify_when = finish` with `verify_retries = 2` (the shipped default).
$1 / 1200s per instance, medium effort, conc 4, official scorer.

| metric (gateless -> gate-on) | gateless | gate-on | better |
|---|---|---|---|
| resolved | 48/110 | 53/110 | gate-on (+5) |
| empty patches | 7 | 12 | gateless (-5) |
| harness/pull errors | 0 | 0 | tie |

Instance-level: gate-on newly resolves 10, loses 5 (3 wrong-patch, 2
timed-out-to-empty). Of the 12 gate-on empties, 11 are container-wall
timeouts at 1200s mid-work (the model-side reasoning-latency DNF profile
the compute-shape arm autopsied; 6 of them were already empty gateless)
and 1 is a clean 124s finish that reported the gate pre-red and declined
to patch. The gate's cost is wall time on marginal instances; its gain is
catching wrong patches before finish.

Read: the shipped default is worth +5 resolved on this split (43.6% ->
48.2%). Against the leaderboard's 58-62 (mean of 5, 128K context) the
single-run gap narrows from 14-19 to 10-14 points. Caveats: n=1 per side;
the board reports 5-run means.

### Test-first on SWE-rebench 2026_03: a null on resolves, fewer wall empties (2026-08-24)

One run per side, same 110, same gate-on default (`verify_when = finish`);
the arm adds only the AGENTS.md probe-test instruction
(`AGENT6_SB_TESTFIRST=1`): write a failing test at /tmp/probe_test.py
reproducing the issue before the first edit. $1 / 1200s, medium effort,
conc 4, official scorer. Three image-pull 429s were re-run and re-scored
before this read (infra, the driver named them; no agent involvement).

| metric (gate-on -> test-first) | gate-on | test-first | better |
|---|---|---|---|
| resolved | 53/110 | 52/110 | within single-run noise |
| empty patches | 12 | 8 | test-first (-4) |
| harness/pull errors after re-run | 0 | 0 | tie |

Instance-level churn: 6 won, 7 lost. Three of gate-on's wall-timeout
empties resolve under test-first (meltano-9950, fromager-1124,
sqlglot-7457): the probe anchored the expected behaviour and the fix
followed. Every test-first empty is the agent's own 1200s wall timeout
(sampled and checked); the probe work costs wall and turns, which is
where the 7 losses went.

Read: test-first does NOT ship as a default on this evidence (a null on
resolves, n=1 per side). The empty-conversion signal (12 -> 8, with 3
direct DNF-to-resolve flips) says the wall, not the anchor, is the
binding constraint: the deadline-steer arm tests that lever directly.

### Deadline-steer on the rebench 110, and what the wall empties actually are (2026-08-24)

One run, same gate-on config, `AGENT6_SB_DEADLINE_STEER=120` (an in-container
`agent6 steer` at T-120s: "land your best fix now"). Scores are real
(110/110, no pull failures, empties the leg's own).

| metric (gate-on -> deadline-steer) | gate-on | deadline | better |
|---|---|---|---|
| resolved | 53/110 | 54/110 | within single-run noise |
| empty patches | 12 | 11 | within noise |

The arm under-exposed its own treatment: the steer FIRED in only 3 of 110
legs. Digging into why reattributed the empties themselves: 19 legs across
the three fleets (gate-on, test-first, deadline) show "parked: an approval
awaits a front-end" and then the wall - the recurring empty ids of every
arm (sqlglot 7457/7479, pygmt-4463, pandas 64796/64797, rapid-mlx 227/228,
scikit-learn-33565, azure-3025, ...). The prompt is the off-list `fetch`
gate (the model hunting the upstream issue: `Allow fetch: api.github.com
/search/issues?...`, standing=false); in a TTY-less container with no
away-mode the approver chooses wait-forever, the leg blocks with no
iteration boundary (so no steer pickup either), and dies at the wall.

Reads:
- the "model-side reasoning latency" DNF autopsy was partly wrong: a large
  fraction of the recurring wall empties are approval hangs, deterministic
  per instance, identical across arms;
- both arms' resolve nulls stand (52..54 across three gate-on-class runs
  is the single-run noise band);
- the real empty-killer candidate is the approver fix (headless no-away
  parks -> deny loudly, the questioner's shape), not either arm.

### The parked-instance rerun under away=deny: 6/12 resolve, zero empties (2026-08-24)

The 12 unique instances that parked on the off-list fetch approval (19 legs
across the three fleets), re-run once under the bench-side
`AGENT6_DETACHED_AWAY=deny` (same 0.0.29 wheel, gate-on config, no arms):
6/12 resolved, 0 empty, 0 errors, 0 parked again - every leg completed with
a real patch. Newly solving: astropy-19438, pandas-64797, pypa-build-1027,
rapid-mlx-228, scikit-learn-33565, sqlglot-7479 (several had produced
nothing but empties in all three fleets).

Totals with the rerun substituted for those 12 ids (before -> after; the
gate-on substitution is same-config; the two arm rows mix configs on the
12 and are so labelled):

| fleet | resolved | empties |
|---|---|---|
| gate-on (clean) | 53/110 -> 56/110 | 12 -> 5 |
| test-first (mixed) | 52/110 -> 55/110 | 8 -> 1 |
| deadline (mixed) | 54/110 -> 58/110 | 11 -> 1 |

Read: the approval-hang was worth ~+3 resolved and -7 empties on the
honest same-config comparison (53 -> 56, 50.9%); against the board's
58-62 (mean of 5, 128K context) the gap is now ~7-11 points. The deny fix
is bench config; the agent6 wait default stands per the A1 ruling.

### Autopsy sprint, pass 1: the 54 gate-on misses classified (2026-08-25)

From on-disk data only (eval reports, preds, the dataset's gold patch +
F2P/P2P lists, classified out of repo; parked-rerun results substituted):

| class | n | definition |
|---|---|---|
| zero-F2P | 23 | patch applied, P2P clean, no graded F2P passes |
| near-miss | 18 | some F2P pass, some fail (several at 1 failure: ytmusicapi 24/25, hats 14/15, wtforms 9/10) |
| broke-P2P | 8 | P2P regressions (mostly 1-2 tests) |
| slow-empty | 5 | no patch produced |

Checked and refuted: the gold patches often touch changelog/news
fragments (~13 misses), but the failing F2P are real behaviour tests in
every sampled case, never changelog-enforcement - a changelog-writing arm
would buy nothing.

What the data does say: in the zero-F2P class the failing test file
frequently names the module the fix belonged in (pygmt/tests/
test_grdmask.py vs the gold's pygmt/src/grdmask.py, which our patch never
touched; moto tests/test_kms vs moto/kms/responses.py). The graded tests
are usually ADDED by the dataset's test_patch, so they are not readable
in-checkout - but their pre-existing sibling files carry the conventions
and the issue text carries the literal strings they assert. Pass 2 (next):
per-class deep dives on samples to pin which in-checkout signal would have
redirected each miss; then the subset-piloted arms per G3.

### Autopsy pass 2, first dives: three near-misses, three mechanisms (2026-08-25)

Read from the dataset's test_patch (the graded tests' own source) vs our
patches, no containers:

- pallets-eco__wtforms-892 (9/10 F2P): the graded test asserts `default`
  populates the rendered `value` ATTRIBUTE while `form.b.data is None` -
  an interpretation subtlety pinned as an exact HTML literal. The
  conventional reading (default -> data) passes everything else.
- astronomy-commons__hats-648 (14/15): the graded test demands
  `ValueError` with match="does not have skymap information" - an exact
  error-string contract on one edge path.
- sigma67__ytmusicapi-909 (24/25): the one failing F2P is PRE-EXISTING -
  a red test already sitting runnable in the checkout; the gold fixes it,
  we never ran it.

Emerging per-class levers, each now evidence-backed at least once:
- exact-literal contracts (error strings, rendered output) that graded
  tests assert verbatim: an arm that makes the run honor the issue's
  quoted literals exactly;
- pre-existing red tests in the issue's neighbourhood: the touched-scope
  test runner (G3 step 2) would have found the ytmusicapi one directly.

### Autopsy pass 2, quantified: zero-F2P is "one layer short" (2026-08-25)

Across all 23 zero-F2P misses (signals tabulated out of repo): 22 graded
tests are added by the dataset's test_patch (unreadable in-checkout);
in-test literal contracts are rare here (2/23) - the literal lever belongs to the
near-miss class. The dominant signal: 16/23 patches never touched a gold
code file, and spot-checks show these are SUBSET-of-gold patches, not
wrong-place patches: moto fixed kms/models.py but not kms/responses.py
(the layer the tests drive); pygmt registered grdmask in every __init__
and doc but never wrote src/grdmask.py; pyinfra fixed operations/server.py
but not connectors/util.py. The fix stops one layer short of the interface
the graded tests exercise.

Converging lever set (all classes now point at the same two):
1. touched-scope/interface-layer test running (G3 step 2): catches
   one-layer-short (any interface-driving test fails at once),
   pre-existing red tests (ytmusicapi), and broke-P2P (8 misses, 1-2
   tests each).
2. an exactness behaviour for near-misses: honor the issue's quoted
   literals (error strings, rendered output) verbatim.
Pilot design follows: arm 1 on the zero-F2P + broke-P2P + near-miss set
once the feature exists; arm 2 as a prompt-block pilot on the near-miss
set.

### Combined pilot launched: scoped-verify wheel + completing-the-fix prompt (2026-08-25)

Prototype pass (arms combined by design; per-class conversion is the
attribution, no strict A/B): the 54 gate-on misses re-run once with
(a) the scoped-verify build: a harness pytest gate that overruns
verify_timeout_s re-runs scoped to the tests nearest the run's diff
instead of certifying nothing (targets broke-P2P 8 and the
finish-unverified tail), and (b) a run-mode base-prompt addendum via the
--prompt-file mount: drive the changed behaviour through the interface
layer the issue describes (targets the one-layer-short zero-F2P 16/23),
reproduce issue-quoted literals verbatim and run the issue-named test
(targets near-miss 18). Shape otherwise identical to the gate-on arm
(same container script, effort, timeouts, away=deny). One-leg smoke
verified the pairing before launch: prompt block present in the leg's
assembled state, patch produced, score path green. Runner rpl1..5 over
pilot-b1..5; compare per class against the gate-on results before any
conclusion. Cost ceiling ~5 plan points measured at ~1 point per 12 legs.

### broke-P2P mechanism: the gate catches it, the model edits the test (2026-08-25)

Reading all 8 broke-P2P legs: every one saw a red verify in-run, and 7 of 8
then edited an existing test file until the gate went green (click: 18 lines
of tests/test_termui.py rewritten to the new behaviour). Submitted patches
strip test files, grading restores the originals, and the buried break
resurfaces as the P2P failure. The class lever is a test-edit discipline,
not test selection: only 2 of the 8 failing test files would even
name-match a nearest-tests pick, and the gate already ran in every leg.
Prompt addendum extended mid-pilot with the no-test-weakening rule
(prototype license): batch 1 ran the v1 addendum (v1 kept beside it),
batches 2-5 carry v2; 7 of the 8 broke-P2P ids sit in batches 2-4.

### Combined pilot result: 10/54 misses converted; substituted 60.0% (2026-08-25)

54/54 legs, 1 empty, ~8 plan points total. Per class (converted/class):
broke-P2P 2/8, near-miss 4/18, slow-empty 2/5, zero-F2P 2/23.
Substituted full-110 (gate-on 50 + parked-rerun 6 + these 10) = 66/110 =
60.0%, board range 58-62. That substitution is a ratchet: it keeps every
prior success and harvests retry variance from the misses, so it is an
upper bound, not a score.

Waves (the wheel bug made an accidental control): batches 1-3 ran with the
scoped gate unreachable (prompt-only) and converted 5/33; batches 4-5 on
the fixed wheel converted 5/21. Suggestive for the wheel, small n.

Mechanisms observed:
- The no-test-weakening prompt line is inert: every v2 broke-P2P leg still
  edited tests (3, 2, 7 times) with the line verbatim in its prompt; the
  two class conversions edited tests too and won on the code half.
  Mechanisms beat prose, again. Next candidates are mechanisms: bench-side
  immutable test files, or a truthful harness notice when a red gate went
  green with only test-file edits in between.
- Both zero-F2P conversions touched exactly the gold file the gate-on leg
  missed (pipecat strategy module, sunpy frames.py); the other 21 did not
  convert, so the interface-layer prose is at best marginal.
- The scoped gate never fired in the field during the pilot: batches 1-3
  had the wiring bug (harness-site only; the model's own timed-out call is
  never re-judged), and after the fix the only 124 leg (pandas) hit the
  heuristic gap (package-level tests dirs unscanned). Both fixed and
  pinned on the branch; a 4-leg re-run of the 124 ids on the final wheel
  is the field test (rpl124).
- Empties stay solved: 1/54 vs the gate-on era's recurring walls.

### Scoped-verify field smoke: 2/2 firings, exact selections (2026-08-25)

The 4 pilot ids whose gates timed out, re-run on the final wheel (rpl124).
Both legs that hit a 124 fired the full mechanism, event-verified
(verify.end 124 -> loop.verify_scoped -> scoped verify.start/end):
graphistry scoped to graphistry/tests/compute/gfql/cypher/test_lowering.py
(the exact P2P test of the original miss), green in 179s; pandas scoped to
3 files including pandas/tests/indexes/datetimes/test_indexing.py (the
exact F2P test of the original miss), green in 7s against the 240s wall.
The two koxudaxi legs saw no timeout this attempt. 0/4 resolved in the
rerun: the mechanism is proven, conversion is a separate question. Note
for graders of run.log: harness notices do not render there; the
logs.jsonl events are the truth (two earlier "never fired" reads were
grep-of-the-wrong-surface errors).

### Near-miss failures are stable contracts, not variance (2026-08-25)

The 14 near-misses the pilot did not convert, gate-on vs pilot F2P
failure sets: 11 identical, 3 overlapping (one grew, two shrank by one),
0 disjoint. Two independent legs (different wheel, different prompt)
converge on the same partial solution and miss the same assertions, so
the residual is a property of the task, not retry luck; retries alone
will not clear this block (~14 instances, the largest left). The
class's lever must surface the missing contract itself: the issue text
quotes most of these literals, so a mechanism that extracts quoted
literals/expected outputs from the task and checks the change ever
produces them (a truthful notice when it does not) is the candidate;
prose asking for verbatim literals already nulled.

### Control prices retry variance: the pilot's arms show no subset effect (2026-08-26)

The same 54 miss ids on the plain master wheel, no prompt mount: 8/54
converted (pilot: 10/54). Overlap: 6 ids converted in BOTH runs (variance,
not arms - both zero-F2P "completions" among them: pipecat and sunpy
re-land the correct gold file on a plain retry too); pilot-only 4
(3 near-miss, 1 slow-empty); control-only 2. A 2-conversion spread at
n=54 is noise.

Two consequences. First, the substituted metric is a ratchet that any
rerun inflates: the CONTROL alone moves 56 -> 64/110 (58.2%); the number
carries no information about the arms and is retired from this program.
Second, retry variance on true misses is ~15% per attempt, so
single-attempt totals and mean-of-5 boards are not directly comparable;
the honest comparable is a fresh full-110 under one config, which is the
sequencer's final stage (scoped wheel + prompt v2). The stable-fail
near-miss block (11/14 identical failure sets across independent runs)
stands: those ids convert in neither run.

### P3 mechanism probe: chattr infeasible, the notice untriggered (2026-08-26)

The 8 broke-P2P ids on the test-edit-notice wheel with chattr +i over
existing test files: 1/8 resolved (anyio, the variance-flagged id).
Neither mechanism was actually exercised: chattr +i is unsupported on
the images' overlayfs, so the degrade warning fired and test files
stayed writable in every leg (the bench-side immutability option is
dead for this harness); and the notice's trigger (a red gate flipped
green after edits touching only test files) occurred in none of the 8
legs this attempt (isort, for one, died on six consecutive reds), so
the notice fired zero times. Untriggered is not refuted: the trigger
demonstrably occurs in other attempts (click's original leg). The
class's remaining live candidates are the product notice (needs
exposure to judge) and the literal-contract mechanism from the
stable-fail analysis (unbuilt).

### Fresh full-110 confirm: 58/99 attempted (58.6%), 0 empty (2026-08-26)

One fresh run, scoped-verify wheel + prompt v2, single attempt per
instance: 58 of 99 attempted resolved; the plan guard stopped the fleet
at its spend ceiling with batch 10 (11 ids) unattempted (6 of
its preds exist unscored; a spec KeyError blocks scoring them apart, so
they rescore with the tail). Floor if all 11 unattempted count as
failures: 58/110 = 52.7%, already past the prior best full run (53/110
= 48.2%). At the run's own rate the tail projects ~64/110 (~58%),
which sits at the 58-62 board range's lower edge.

Attribution stays honest: the subset control showed the arms add
nothing above ~15% retry variance, so the gain over the 48.2% prior
best is primarily (a) the away=deny fix now baked in (the prior run
lost ~19 legs to approval parks; 0 empties here vs 12 then) and (b)
sample variance (batches ran 5/11 to 10/11). The arms ride along
unproven. Completing the last 11 legs costs ~1 plan point and closes
the number.

Completion note: the sequencer's own batch-10 scoring landed after the
kill: 59/110 as-run (53.6%), 8 of batch 10's 11 legs empty as
guard-kill artifacts (killed mid-run), not model declines; 1 of its 6
extracted patches resolved. On legs that ran to completion the rate is
59/105 (56.2%). Prior best full run: 53/110 (48.2%). A clean 11-leg
rerun of batch 10 (~1 point) yields the artifact-free 110 number.

### Batch-10 clean rerun, the same-window board, and a staging defect (2026-08-30)

The guard-killed batch 10 of the fresh full-110 (scoped-verify wheel +
prompt v2) re-ran clean: 3/11 resolved (azure-search-openai-demo-3025,
rapid-mlx-341_interface, mtplx-21), 7 unresolved, 1 empty
(schemathesis-4087: a 9-call finish with 0 patch lines), 0 errors, 1.0
plan point (0.10/leg). Batches 1-9 on disk: 58/99. Clean full-110 =
61/110 (55.5%). The 2026-08-26 completion note's "59/105 on completed
legs" is 59/102 on disk (99 legs plus 3 completed in batch 10); its
~64/110 projection is superseded by the rerun.

The leaderboard on our window (swe-rebench.com, per-window stats from the
page's embedded JSON, problems 2026-03-01..05-14 = the 110 ids of the
2026_03 split, mean of 5 runs, SEM):

| agent | resolved | pass@5 |
|---|---|---|
| Junie | 61.6 +/- 0.64 | 72.7 |
| Codex | 60.4 +/- 1.37 | 71.8 |
| Claude Code | 59.6 +/- 1.98 | 72.7 |
| Cursor | 53.0 +/- 0.53 | 64.5 |
| GLM-5.2 [high] | 51.1 +/- 1.13 | 71.8 |
| Kimi K2.6 | 46.5 +/- 1.27 | 64.5 |
| agent6, gpt-5.6-sol medium (one run) | 55.5 | |

The "58-62" rows quoted earlier (GPT-5.6 Sol[medium] 62.3, Claude Code
60.4, Codex 58.0) are the board's default window, problems
2026-05-15..07-01; GPT-5.6 Sol, Fable 5, Opus 5 and Sonnet 5 have no
rows on the 2026_03 window. Single-run SD on 110 ids is ~3 points
(per-id resolve frequencies over the gate-on-class runs: 43 ids never
resolve, 41 always, 26 flip).

Staging defect in the harness: `in_container.sh` staged the patch with
`git add -u`, so a file the run CREATED never entered the prediction.
No prediction in any run family carries a `new file mode` hunk. Four of
the 110 ids need a new source module for the graded tests to import
(toqito-1484_interface, scim2-models-139_interface,
pygraphistry-1107_interface, pygmt-4463); in the judged legs the model
created that file (apply_edit create / apply_patch add-file), the
submitted patch lacked it, and the eval died on ModuleNotFoundError
(pygmt: `No module named 'pygmt.src.grdmask'`); sunpy-8548's autopsy
leg lost its own helper module the same way. Those legs were
unresolvable under the harness, not model misses. Fixed: the patch is
`git add -A` minus the files untracked at the base commit and agent6's
own files (the board's scaffold submits `git add -A`); the new-file
count prints beside the patch line count. Smoke on the two new-module ids
(pygmt-4463, toqito-1484_interface) under the fix: 2/2 resolved, patches
carrying 1 and 2 new files, 0.2 plan points. A fresh full-110 on the
fixed harness launched 06:59Z (same wheel and prompt as the 61/110 run;
out dirs rebench-f110x-b1..10, run-ids rfx1..rfx10).

Literal-contract dry run (scripts and per-id dumps out of repo): three
extractors (quoted spans; error/expected-output strings; the oracle of
string literals asserted in the FAIL_TO_PASS tests) over the 54 misses and 20 resolved controls.
Truthful issue-derived fires on the failing contract: 0 for every
buildable extractor; the quoted-span variant fires on 55% of resolved
controls. The oracle ceiling after removing the staging artifacts is
4/54, all four literals absent from the issue text (opensandbox-426
"docker network create", hats-648 "does not have skymap information",
koxudaxi-3071 "Invalid JSON:", docsight-437 key names). The arm is not
built.
