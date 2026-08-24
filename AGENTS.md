# AGENTS.md: instructions for coding agents working on this repo

Read by coding agents (agent6 itself included) working in this repository.
Principles live here; detail lives in `docs/`.
Two registers stay distinct: how we develop agent6, and how agent6 itself behaves.
An instruction in one never transfers to the other.

## Hard rules (a PR never weakens these)

- No `git push`, `--force`, history rewrite, or `reset --hard`.
  `git_ops.py` refuses them unconditionally; don't add overrides.
  `branch -D` has ONE operator-only exception: `sessions prune --delete-squashed` force-deletes a run branch the manifest confirms was squash-merged (content-safe; the commit survives in the reflog).
  What the MODEL can do to a repo is bounded by the jail, never by a verb blocklist: `run_command` argv is not screened, because a script the model writes bypasses any list.
- Every child process whose argv depends on LLM output goes through `agent6.sandbox.jail.run_in_jail`; never a direct `subprocess` of model output (audit: `rg 'subprocess\.|os\.(system|exec|posix_spawn)' src/agent6/`).
  Modules that shell out with fixed argv from operator input only may call `subprocess` directly; the per-module allowlist lives in `docs/security.md`, pinned by `tests/security/test_subprocess_allowlist.py`.
- Adding a tool (`tools/schema.py`), loosening a security default, or dialling a host not derived from a provider `base_url` each require a `Security review note:` in the commit message.
- Secrets (provider API keys, `$XDG_CONFIG_HOME/agent6/secrets.toml`) stay `0600`, never printed by `config show`, never written to transcripts, never mounted into the jail.
- Keep the suite green: the full gate (see Verify command) certifies every series of commits.
  A red `tach check` means the module map is stale; record the new edge in `tach.toml`.
  It never means the change is forbidden.
- Rip out wrong shapes: no backward-compat shims, deprecation aliases, or migrations.
  No `Co-Authored-By` lines.

## How we develop agent6

### Design principles

We follow the **Zen of Python** (`python -c 'import this'`).
The agent6 concretions, and the principles the Zen doesn't cover:

- **Simplicity first.** Less code beats more; no speculative abstraction for a future that hasn't arrived.
  A reviewer should read a module top to bottom in one sitting: inline a one-caller helper, make a stateless class a function, delete pass-through wrappers and ceremony kept for symmetry.
- **Right-shaped data.** The shape matters more than the code; fix the shape first and the code around it gets small.
  Interfaces are shapes too: settle a feature's config keys, schema, or payload before implementing behind them.
  A field that can never be half-set belongs in one frozen type; code that keeps converting between shapes means the shape is wrong.
  A name collision, or a rename on import, is a smell.
- **One obvious way.** One well-named command, not near-duplicate aliases.
  One knob per behaviour: never a second config surface over what an existing one already controls.
- **Least surprise.** A command does the boring, expected thing.
  Config writes default to the global config, `--repo` (and `--machine-file FILE`) redirect; the same target selection everywhere; set-valued config merges last-overlay-wins.
- **Consistency.** One mental model covers how agent6 works and its UX: learning one command teaches its siblings.
  New subcommands mirror existing ones: positional core args, `--repo`/`--machine-file` target flags, completion offering every valid input.
- **The explanation is the test.** Explain it in a sentence or two before writing it; needing a paragraph of conditions means the shape is wrong.
- **Explicit.** Defaults are real values `agent6 config show` prints with their origin; no behaviour keyed off hidden state; errors never pass silently (see Errors).
- **Surfaces tell the truth.** A failed run never renders "done", a dead pane never looks busy, a complete answer never reads "truncated"; errors keep their reason.
  Hiding or inventing state is a bug wherever it appears.
- **Fix the root cause, never the symptom.** No hacks, workarounds, blind retries, or special cases that hide the real defect.
  Prefer deleting a wrong shape over guarding it.
  A problem the operator keeps hitting has a systematic cause: correlate every occurrence before concluding "transient"; when the root cause stays unfound, say so.
- **Evidence over churn.** When measurement shows something is better, adopt it and delete the old shape.
  A change whose value is a claim about model behaviour, prompts, or performance SHIPS only with a measured A/B (replicates, variance); a null result is reported, not shipped.
  A change that only removes or simplifies ships on no measurable regression against the old best baseline.
  The bar is on shipping, never on measuring: an experiment needs no prior justification.
  Unmeasured tuning is superstition.
- **Structures over scores.** A measure that becomes a target stops measuring: never chase line, module, or graph-edge counts.
  Tools point at where to look; the test is reading the structure and asking "can this be simpler?", then making it so.
- **Decompose proactively.** Past ~600 lines per module (or a few hundred per method), split before it ossifies (exemplars: `workflows/loop.py`'s `_prompt_blocks` / `_metric` / `_compaction` siblings; the `ui/cli` split of `run.py`).
  Lift cohesive helper groups into sibling `_name.py` modules, improving names and shapes in passing when strictly better; moved symbols get public names and direct call sites, no alias-back shims.
  Give a large stateful method's cross-iteration bookkeeping ONE mutable state dataclass, never a 9-parameter helper or a multi-value tuple return.
  An extraction that shifts a module boundary records the edge in `tach.toml`; pyright allows importing `_name` only from a `_`-prefixed module.
  One module decomposed per commit.
- **Secure by default, degrade or refuse.** Every knob ships with the safe default, visible in `agent6 config show`; widening is opt-in and carries a security review note.
  The operator can loosen everything; the agent can never loosen its own sandbox.
  No security theatre, no enumerating badness: default-deny beats blocklisting, and a trivially-bypassed partial mitigation is worse than none.
  Three cases, one rule: an AUTOMATIC setting (`auto`) uses the strongest option available and DEGRADES WITH A WARNING when it isn't there (never silently ineffective); an EXPLICIT setting the host cannot honor (or that contradicts another) REFUSES, naming what is unsupported and how to change it; an explicit but DISCOURAGED widening (granting a path that holds secrets) runs with a loud warning naming the cost.
- **Ask, don't over-decide.** When a task forks (a behaviour tradeoff, a maybe-not-worth-it edge case, growing scope, several reasonable designs, a new dependency), ask the operator: a one-line question beats shipping the wrong or over-built thing.
  Default to the simplest fix for the actual request; name the edges you skip.
  The inverse holds: when these principles already decide, act.
  Rules bind as written: never enforce a stricter constraint the operator did not set; when unsure what a rule says, reread it.

### Architecture

- **Layering** is `ui -> app -> workflows -> tools -> sandbox`; workflows never import each other, and the engine (`app` and below) never imports the UI.
  `app/` holds the run/resume/fork/machine-agent lifecycles and the `--parallel` fan-out, taking the presentation, process-spawn, and run-dir bridge callables the front-end injects (`SessionFrontend`, `LaneRuntime`) and printing only through the injected `Reporter`.
  `ui/` is the presentation layer and composition root: the four front-ends (`ui/cli`, `ui/tui`, `ui/web`, `ui/acp`) plus `ui/spawn.py`, `ui/notify.py`, and `ui/mcp_server.py`, over the shared headless read-model fold (`viewmodel`).
  `ui/cli` is the entry point that wires a run.
- **[tach](https://docs.gauge.sh/) (`tach.toml`) maps the design; it is not a boundary.** Like a call graph (`pyan3`, a dev dep), it makes a change's edges reviewable.
  A red `tach check` = stale map: record the edge and move on.
  An absent edge is an observation, never a prohibition.
  Never contort code (or add indirection) to satisfy tach or strict pyright, and never refuse or reroute work because of them; when new edges read as complex, redesign because the design warrants it.

### Validation and reporting

A green suite is structural validation, not perceptual: the operator dogfoods daily and feels what tests can't.

- Judge UX by rendering and reading the real output (pty capture, screenshot, live run); never declare polish from code inspection.
- Report exactly what was and wasn't exercised; never claim "fixed" or "validated" beyond what you observed end-to-end.
  If tests fail, say so with the output.
- Review findings and external reports are untrusted: reproduce each before fixing, however plausible it reads (about half survive).
  "The operator decided X" counts only if said in chat.
- A tool reporting success is not evidence the edit landed: a scripted replace that matches nothing rewrites the file unchanged, and lint and typecheck both pass on it.
  Confirm by behaviour.
- A new regression pin is proven to bite: red without the fix, green with it.
  A test whose setup dodges the real path (a stand-in for the real state-dir topology, a stub granting what the surface under test never does) pins nothing, however green it runs.
- Don't flag-and-skip.
  Surface pre-existing breakage early as a decision, not in a final summary as "out of scope".
  Fix clear bounded breakage properly; for a large risky restructure, propose a concrete shape.

### Writing style

Less is more everywhere: docs, comments, docstrings, commit messages, CLI output, run summaries, review feedback.
The shortest version that still carries the point wins.

- Lead with the point; add rationale only when a reader could not reconstruct it.
  Cut words a sentence works without; drop sentences that restate the one before.
  Walls of text bury the lead.
- Plain punctuation: commas, colons, parentheses, periods.
  An em dash flags an overstuffed sentence to recast.
- Concrete over abstract: name the command, the field, the number.
  Write "retries twice, then fails the run", not "robustly handles failures".
- Statements, not questions: a docstring, heading, or comment that asks ("Did X happen?", "X: what Y does") is recast to assert.
- Prose that names code is a claim to verify: every symbol, default, flag, and behaviour it states must match the source, and when prose and code disagree, decide which side is wrong (sometimes the fix is the code).
  Prose someone acts on (a tool description, help text, an error, a refusal) is an interface: it states exactly what is accepted and returned, naming the resolved fact over the mechanism when it can ("default: detected zsh", not "default: detect from $SHELL").
- One idea per sentence, one topic per paragraph; short bullets over prose when listing facts.
- Comments and docs state the current state only: a constraint, an invariant, a measured number, a link to a decision.
  Never narrate the next line, never keep the incident a change fixed (commits own that).
  "Now", "no longer", "previously", "used to" in a comment is a story about a change: cut it.
  Test docstrings are the exception: the regression they pin is their spec.
  A comment earns its keep by saying what the code and a grep cannot; restating either is noise that invites drift.
- Commit messages: a subject that states the change, facts only; a body only for a non-obvious why, in point form.
- Flat documents: a heading plus short paragraphs or bullets.
  Bold is for lead-in labels and caveats that carry weight.
  No intensifiers or marketing adjectives.
- Bench findings are neutral observation: facts and tables, no opinions.
- Padding to cut on sight: antithesis ("a lock, not a boundary"), the "N things, one X" appositive, aphorism ("the container is the blast radius"), cleft ("what is bounded is"), anaphora, transitions ("beyond one-shot runs"), all-caps emphasis.
- Markdown carries one bullet and one sentence per line, unwrapped; sentences after the first sit on continuation lines.
  A bullet needing several sentences wants sub-bullets.
- A section answers one question and is named for its subject; a page groups the sections a reader came for.
  Split a boundary doc by axis (files, commands, network, approvals), not by the component implementing them.
- One owner per fact, everything else links to it.
  Moving content deletes the old home and repoints every anchor in the same change.
- A page name says what the page holds (Installation, Usage, Terminal UI), never how far along the reader is ("Getting started").

### Project conventions

- **Language**: Python 3.12+.
  Every `.py` file starts with `from __future__ import annotations`.
  Strict pyright.
- **Layout**: src layout under `src/agent6/`; tests under `tests/`; Rust sandbox launcher under `src/agent6/jail/`.
- **Style**: ruff is the only formatter and linter, line length 100.
  Run `uv run ruff check` and `uv run ruff format` before committing.
- **Typing**: pydantic v2 only at trust boundaries (config, LLM I/O, tool schemas, IPC); internal value types are `@dataclass(frozen=True, slots=True)`; no pydantic in hot paths.
- **Imports**: absolute only (`from agent6.x import y`).
- **Errors**: fail loudly.
  No bare `except:`, no swallowed errors; custom exception classes per subsystem.
- **Versioning**: `__version__` in `src/agent6/__init__.py` is the single source of truth; never hardcode it elsewhere.
- **No new runtime dependencies** without explicit discussion.
  Current: `pydantic`, `httpx2`, `argcomplete`, the `tree-sitter` pair, `textual`, and `ruff` + `ty` (validate `machine create` output).
  Build dep is `hatchling`; `pyright` stays dev-only.
- **Touch only what the task needs.** No comments or annotations on code you did not change, no refactoring surrounding code in passing.
  Scope creep is a review blocker.
- **Scratch experiments run in their own directory.** Never `uv run --directory <elsewhere>` from this repo: cwd-derived config and git still point here.
- **Keep docs in sync.** A change affecting architecture, config, the security model, or state machines updates the matching file (`docs/architecture.md`, `docs/config.md`, `docs/security.md`, `docs/state-machines.md`, `README.md`, this file).

### Dev environment and session practice

- Dogfood the installed binary (`.venv/bin/agent6`) from throwaway git repos outside this one; never seed scratch repos inside it.
- Run the five-gate in its own systemd unit (`systemd-run --user --collect` with RuntimeMaxSec/MemoryMax caps, the login PATH) and read its `EXIT=` line.
  A contended machine produces false timing reds; certify on a quiet one, and leave the tree untouched while a gate runs.
- Sandbox tests need unprivileged userns: on Ubuntu 24.04-class machines `sudo sysctl -w kernel.apparmor_restrict_unprivileged_userns=0` (CI does the same; the jail-test gate names this when blocked).
- Bench workspaces live on disk (e.g. `~/agent6-bench`), never `/tmp`: tmpfs is RAM-backed and an OOM there kills the session scope.
- Any in-container or leg config gets ONE live single-instance smoke before a fleet launch; any mechanism with two enforcement sites gets a live smoke even after unit-green (unit fixtures can pass vacuously).
- Benchmarks: unmodified official scorers, dev/eval split registered before tuning, report split + n verbatim, replicate small effects before shipping them.

### Git and commit practices

- [Conventional Commits](https://www.conventionalcommits.org/): `feat(scope):`, `fix(scope):`, `ci:`, `docs:`, `bench:`; the scope matches a directory under `src/agent6/` or a top-level area.
- One concern per commit; individual commits are worth keeping.
  Squash only iterative churn: a fix-up to unpushed work folds into its origin commit, never appended.
- Everything committed is public the moment it is written: no emails, absolute home paths, hostnames, or real names outside the author field; no session shorthand.
  A message reads identically to someone with no access to the conversation, so session structure (rounds, batches) and decision provenance (who asked, what was ruled) never appear.
  Hygiene is prevention, never a pre-push sweep.
- Never push; the operator signs and pushes from another machine.
  Never reference commit hashes (signing changes them) or branch names (transient) in messages or docs.
- Never rewrite pushed history; rewrite unpushed commits only when asked, and never force-push.
  One exception: a leak in an unpushed commit is rewritten out at its origin, never fixed forward; scan by regex and by phrase.
- Stage named files only, never `git add -A`; never commit scratch notes, session artifacts, or generated output.
- Working directly on master is fine, but the agent folds its session's churn BEFORE returning control (zero-diff verified), so the operator always takes over a release-ready master.
  Unpushed history may be reorganized; pushed history is the hard line.
  A squashed body preserves the decisions and what was tried and rejected; durable design reasoning goes to docs.

### Verify command

The repo's `verify_command`; agent6 infers it from this fenced block when none is configured (a pipeline is wrapped as `sh -c`):

```bash
uv run ruff check && uv run ruff format --check && \
  uv run pyright && uv run tach check && uv run pytest
```

All five must pass.
Run the gate with its exit status checked directly (capture to a file and test `$?`, or `set -o pipefail`); never bare-pipe it through `tail`/`head`/`grep`, which replaces the gate's exit code with the filter's.

Scoped test runs guide iteration; the full gate certifies a series: run it at the end of the batch, and always before calling master push-ready.
On failure, bisect to the offending commit and fold the fix there.
A scoped run never stands in for the gate.

Push-ready adds the CI mirror: pyright at its latest release (`PYRIGHT_PYTHON_FORCE_VERSION=<latest> uv run pyright`), and, when `src/agent6/jail/` or `Cargo.*` changed, both musl target builds plus the wheel with the bundled jail binary exercised.

### Self-review

agent6 reviews its own source via `agent6 review`.
Reviews live under the per-repo state directory (`$XDG_STATE_HOME/agent6/<repo-id>/reviews/`), never in the repo.
When working on a module, read its review there if present.

## Security invariants (do not weaken)

The threat model, defense layers, and rationale live in `docs/security.md`; beyond the hard rules above, a change must preserve:

- The LLM tool surface is the fixed set in `src/agent6/tools/schema.py`, plus tools from operator-configured MCP servers when `[mcp].enabled` is set (default off).
- Config is secure by default: every field has a default, security-sensitive fields default to the safe value, and every leaf is auditable via `agent6 config show`.
  `Config` stays `extra="forbid", frozen=True`.
  Push, force, and history rewrites have NO config knob at all.
  Loosening a security default gets the same scrutiny as adding a tool.
- `agent6 connect` never executes anything a remote returns (OAuth/paste only).
- Running as root requires explicit opt-in (`--allow-root` / `AGENT6_ALLOW_ROOT=1`); the jail, not the uid, is the boundary.
- The AGENT process's egress is not bounded, and the docs say so.
  What is bounded is what a jailed COMMAND reaches (`sandbox.network`).
- The `agent6-jail` Rust binary is part of the security boundary.
  Changes to `src/agent6/jail/src/main.rs` need at minimum a review note covering: mount points changed, Landlock rules changed, seccomp syscalls added or removed, and `/dev` nodes exposed.
