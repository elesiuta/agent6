# longhorizon bench: compaction, memory, and dependency use on long tasks

The evaluation vehicle bench/coreagent's FINDINGS called for: its short
test-graded tasks never trigger realistic compaction, so tier-1 elision
costs, summary fidelity, and every cross-run channel stayed unmeasurable.
This harness runs agent6 on genuinely long, multi-leg task sequences and
measures exactly those:

- **Tier-1 compaction losses**: whether elisions cause re-reads and
  forgotten spec details. Gates the deferred facts-ledger: build it only
  if these losses are real.
- **Cross-run memory value**: whether the `<memories>` block makes a second
  run on the same repo cheaper or better (add_memory / invalidate_memory
  shipped 2026-07-06).
- **add_dependency use**: whether any model creates DAG edges unprompted,
  and whether that correlates with better sequencing (standing decision:
  rip the tool out if useless).

## How it runs

Each (model x condition x task x rep) cell is a SEQUENCE: all of a task's
legs run in one throwaway git workdir, in order; a leg may first overlay
extra files (`tasks/<t>/<inject>/`, new requirements landing mid-project).
Legs share the per-repo agent6 state dir, so cross-run channels carry over;
the `fresh_state` condition gives each leg a private state dir instead;
that pair is the memory A/B. Every leg is graded by the task's
authoritative HIDDEN grader (`tasks/<t>/grade.py`, never shipped into the
agent's repo) with partial credit and per-component scores, and recorded as
one JSON line in `results/<label>.jsonl`.

Prompts stay NEUTRAL about the DAG, dependencies, and memories: whether the
agent uses those channels is part of what the bench measures.

## Tasks (`tasks/<name>/`)

| task | shape | legs | measures |
|------|-------|------|----------|
| `stylebook` | implement a 10-rule text auditor; each rule is one ~4k-char authoritative doc, with deliberate cross-file couplings (R01<->R09/R10, R08<->R09) | 1 | retention: per-rule component scores = which early-read rules survive elision |
| `relay` | 6-module event pipeline with a strict interface chain (incl. a ping count threaded through stages 3->6) | 1 | organic length, interface retention, add_dependency shape |
| `orchard` | leg 1: fix a price whose root cause is a SOURCE table behind a generated file (verify regenerates, hand-edits get clobbered); leg 2: WEEKEND tier touching the same generator + the cents/half-up convention; leg 3: CLEARANCE feed re-probing BOTH conventions with fresh discriminators, spec deliberately silent on them (no NOTES.md pointer) | 3 | memory value: leg-2/leg-3 iterations/score/trap-edits, baseline vs fresh_state. Leg 3 is the READ-side probe: leg-1/2 recoverers hold nudged memories by leg 3, the 2-leg design could not separate that from task ease |
| `ledger` | a double-entry CLI whose command table is GENERATED from `commands.toml` (verify regenerates), money in integer cents with half-up rounding (`docs/CONVENTIONS.md`), every posting through the balanced-transaction invariant; leg 1 fixes a handler mapping, legs 2-5 add `split`, `convert`, `report`, `import-csv`, each spec silent about the conventions it needs | 5 | cross-session memory value over a 5-session campaign: later-leg score, tool calls and tokens, re-reads, memory writes; the `poisoned` probe |

Graders were validated against reference solutions (reference scores 1.0,
stub 0.0, targeted mutants dent exactly their component; references live
outside the repo, in the authoring session's scratchpad). `orchard`'s
rounding component includes an engineered half-up vs banker's divergence
(F-310: shelf 790 -> weekend 909; float `round()` gives 908). Leg 3 carries
fresh ones the shipped acceptance test deliberately omits (clearance A-140
448 / E-905 1858 catch `round()`, B-204 649 catches truncation), and its
`regen` component deletes `data/*.tsv` before running the agent's build
tools, so a hand-written feed no generator reproduces scores zero.

## Conditions (`CONDITIONS` in run.py)

- `baseline`: shipped defaults (adaptive thresholds; memories on).
- `window16k` / `window32k` / `window64k`: `[context]` thresholds pinned to
  what the shipped adaptive fractions (45%/80%) resolve to on that window.
  That is the regime a small/open-model user gets BY DEFAULT, not an
  artificial squeeze; on a 256k model baseline compaction never fires, so
  these rungs are the compaction A/B. A tidy reader stays under the 32k
  rung's 58k-char drop threshold on these tasks; 16k (local serving) is
  where tier-1 provably engages.
- `window16k_nogist`: the same thresholds with `context.elision_gists`
  off: bare markers only, the pre-gist behavior. Pairs with `window16k`
  as the gist-elision A/B.
- `fresh_state`: private state dir per leg: the no-cross-run-memory
  control for sequences.

- `poisoned`: the shared state dir, plus the task's stale memories (plausible, wrong, contradicted by the repo) planted with `agent6 memory add` before leg 2; whether later legs verify before trusting shows in their trap components and `memory_invalidations`
## Metrics per leg (from the run's logs.jsonl + the grader)

`score`/`component_scores` (partial credit; per-rule retention curve on
stylebook), `drops_total`/`drop_events`/`first_drop_at_tool_call` (tier-1),
`gists_total`/`gist_demotions`/`gist_failures` (tier-1 gist elision),
`compactions` (tier-2), `redundant_reads`/`repeat_path_reads` and their
`_post_drop`/`_post_compact` splits, `memory_writes`/`memory_invalidations`
plus `memory_flip_nudges`/`memory_finish_nudges` (the loop's write-side
nudges) and a post-leg store snapshot (`memories_ids`/`memories_bytes`),
`deps_added`/`deps_in_graph`, `trap_edits` (edit calls whose TARGET is one
of orchard's generated feeds; matched on the `path` arg when present so a
generator merely mentioning the feed in a docstring does not count),
`iterations`, `usd`, `tokens`, `wall_s`, `end_reason`, `tampered`.

Memory metrics read the real channel: the repo memory is files under the
state home's `memory/` dir (plus the injected `MEMORY.md` index), so
`memory_writes` counts edit-tool calls targeting that dir, `memory_reads`
counts `read_file` calls there, and a before/after file diff per leg gives
`memory_files_added/changed/removed` (`memory_invalidations` = changed +
removed) and `poison_touched`: whether a planted stale memory was rewritten
or removed.

## State and resume

- A sequence's state dirs sit BESIDE its workdir (`<seq>.state`, `<seq>.state-leg<i>`): agent6 refuses a private dir inside the workspace it edits.
- A label is resumable: a cell (task, condition, rep) with every leg recorded under `results/<label>.jsonl` is skipped; a partial cell reruns whole and is named.
- `--leg-memory-max 8G` runs each session in its own `systemd-run --scope` with that MemoryMax, so an OOM kills the leg, never the driver.

## Running

```bash
# compaction A/B on the retention task
python3 run.py --model moonshotai/kimi-k2.6 --provider openrouter \
    --tasks stylebook,relay --conditions baseline,window32k \
    --reps 3 --parallel 3 --label wave1

# memory A/B on the sequence task
python3 run.py --model qwen/qwen3-coder-30b-a3b-instruct \
    --tasks orchard --conditions baseline,fresh_state \
    --reps 4 --parallel 3 --label mem1

# direct Anthropic API (token-capped); --timeout-scale lifts slow models'
# per-leg timeouts (kimi stylebook wants ~2x)
python3 run.py --model claude-sonnet-5 --provider anthropic \
    --tasks orchard --conditions baseline,fresh_state \
    --reps 3 --parallel 2 --label mem4-sonnet

python3 stats.py results/wave1.jsonl              # cells + deltas
python3 stats.py results/wave1.jsonl --components # per-rule retention curve
```

Providers + secrets come from the layered global `~/.config/agent6` config;
the harness pins the three model roles and wires `["bash", "verify.sh"]` as
the verify command. Anthropic models (unpriced) get token caps instead of
`--max-usd`. Results are append-only JSONL; each record carries every knob,
so any cell is reproducible from (model, condition, task, leg).
