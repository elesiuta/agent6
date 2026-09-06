# AGENTS.md: instructions for coding agents working on this repo

Read by coding agents (agent6 itself included) working in this repository.
Principles live here; detail lives in `docs/`.
Two registers stay distinct, each binding only itself: how we develop agent6, and how agent6 behaves.

## Hard rules (every PR preserves these)

- Push, `--force`, history rewrites and `reset --hard` belong to the operator: `git_ops.py` refuses them unconditionally, with no override.
  `branch -D` has ONE operator-only exception: `sessions prune --delete-squashed` on a branch the manifest confirms was squash-merged (the commit survives in the reflog).
  The jail bounds what the MODEL can do to a repo; `run_command` argv stays unscreened, because a script the model writes bypasses any blocklist.
- Every child process whose argv depends on LLM output goes through `agent6.sandbox.jail.run_in_jail` (audit: `rg 'subprocess\.|os\.(system|exec|posix_spawn)' src/agent6/`).
  A module shelling out with fixed argv from operator input may call `subprocess` directly; that allowlist lives in `docs/security.md`, pinned by `tests/security/test_subprocess_allowlist.py`.
- Adding a tool (`tools/schema.py`), loosening a security default, or dialling a host not derived from a provider `base_url` each require a `Security review note:` in the commit message.
- Secrets (provider API keys, `$XDG_CONFIG_HOME/agent6/secrets.toml`) stay `0600` and stay put: out of `config show`, out of transcripts, out of the jail.
- Keep the suite green: the full gate (see Verify command) certifies every series of commits.
  A red `tach check` means the module map is stale: record the new edge in `tach.toml` and move on.
- Rip out wrong shapes: a rename lands everywhere at once, without shims, aliases, or migrations.
- Commit messages carry no `Co-Authored-By` line.

## How we develop agent6

### Design principles

We follow the **Zen of Python** (`python -c 'import this'`).
The agent6 concretions, and the principles the Zen doesn't cover:

- **Simplicity first.** Less code beats more; build an abstraction when it is needed, never because it might be.
  Simple and stupid beats clever, in code, shapes and interfaces: what a beginner can follow is the target, because cleverness hides bugs and raises the cost of every later read.
  A reviewer reads a module top to bottom in one sitting: inline a one-caller helper, make a stateless class a function, delete pass-through wrappers and symmetry-for-its-own-sake.
  Refactoring is continuous: every series leaves the shapes it touched simpler.
- **Right-shaped data.** Fix the shape first and the code around it gets small.
  Interfaces are shapes too: settle a feature's config keys, schema and payload before implementing behind them.
  Fields that are set together belong in one frozen type; repeated conversion between shapes means the shape is wrong.
  One name per thing, so a collision or a rename on import is a smell.
- **One obvious way.** One well-named command; one knob per behaviour; one mechanism per job.
  A second implementation of the same decision (a shadow check, a twin fold) drifts from the first: harden the one that exists.
- **Least surprise.** A command does the boring, expected thing.
  Config writes land in the global config unless `--repo` or `--machine-file FILE` redirects them, the same way everywhere; set-valued config merges last-overlay-wins.
- **Consistency.** Learning one command teaches its siblings: positional core args, `--repo`/`--machine-file` target flags, completion over every valid input.
- **The explanation is the test.** Explain it in a sentence or two before writing it; needing a paragraph of conditions means the shape is wrong.
- **Explicit.** Defaults are real values `agent6 config show` prints with their origin; behaviour follows visible state, and errors are loud (see Errors).
- **Surfaces tell the truth.** A failed run reads failed, a dead pane reads dead, a truncated answer reads truncated, and an error keeps its reason.
  Hidden or invented state is a bug wherever it appears.
- **Fix the root cause.** A hack, a blind retry or a special case hides the defect; delete the wrong shape instead of guarding it.
  A recurring problem has a systematic cause: correlate every occurrence before calling one "transient", and say so plainly when the cause stays unfound.
- **Evidence over churn.** When measurement shows something is better, adopt it and delete the old shape.
  A change claiming better model behaviour, prompts or performance ships on a measured A/B (replicates, variance); a null result is reported.
  A change that only removes or simplifies ships on no measurable regression against the old best baseline.
  The bar is on shipping, so an experiment needs no prior justification; unmeasured tuning is superstition.
- **Structures over scores.** A measure that becomes a target stops measuring, so counts (lines, modules, graph edges) only point at where to look.
  The test is reading the structure, asking "can this be simpler?", and making it so.
- **Decompose proactively.** Past ~600 lines per module (or a few hundred per method), split before it ossifies (exemplar: `workflows/loop.py`'s `_prompt_blocks` / `_metric` / `_compaction` siblings).
  Lift cohesive helper groups into sibling `_name.py` modules, improving names and shapes in passing; moved symbols get public names and direct call sites.
  A large stateful method's cross-iteration bookkeeping goes in ONE mutable state dataclass, rather than a 9-parameter helper or a tuple return.
  An extraction that shifts a module boundary records the edge in `tach.toml`; pyright allows importing `_name` only from a `_`-prefixed module.
  One module decomposed per commit.
- **Secure by default, degrade or refuse.** Every knob ships with the safe default, visible in `agent6 config show`; widening is opt-in and carries a security review note.
  The operator loosens; the agent's own sandbox stays where the operator put it.
  Default-deny beats blocklisting, and a mitigation that is trivially bypassed is worse than none.
  Three cases, one rule: an AUTOMATIC setting (`auto`) takes the strongest option available and DEGRADES WITH A WARNING when it is absent; an EXPLICIT setting the host cannot honor (or that contradicts another) REFUSES, naming what is unsupported and how to change it; an explicit but DISCOURAGED widening (a path holding secrets) runs with a loud warning naming the cost.
- **Ask when the task forks.** A behaviour tradeoff, a maybe-not-worth-it edge case, growing scope, several reasonable designs, a new dependency: a one-line question beats shipping the wrong or over-built thing.
  Take the simplest fix for the actual request and name the edges you skip.
  When these principles already decide, act.
  Rules bind exactly as written: enforce what the operator set, and reread the rule when unsure.

### Architecture

- **Layering** is `ui -> app -> workflows -> tools -> sandbox`; workflows never import each other, and the engine (`app` and below) never imports the UI.
  `app/` holds the run/resume/fork/machine-agent lifecycles and the `--parallel` fan-out, taking the presentation, process-spawn, and run-dir bridge callables the front-end injects (`SessionFrontend`, `LaneRuntime`) and printing only through the injected `Reporter`.
  `ui/` is the presentation layer and composition root: the four front-ends (`ui/cli`, `ui/tui`, `ui/web`, `ui/acp`) plus `ui/spawn.py`, `ui/notify.py`, and `ui/mcp_server.py`, over the shared headless read-model fold (`viewmodel`).
  `ui/cli` is the entry point that wires a run.
- **[tach](https://docs.gauge.sh/) (`tach.toml`) maps the design; it is not a boundary.** Like a call graph (`pyan3`, a dev dep), it makes a change's edges reviewable.
  A red `tach check` = stale map: record the edge and move on; an absent edge is an observation.
  Write the code the design wants and let tach and strict pyright follow it; neither one decides whether a change lands.
  When new edges read as complex, redesign on the design's merits.

### Validation and reporting

A green suite is structural validation, not perceptual: the operator dogfoods daily and feels what tests can't.

- Judge UX by rendering and reading the real output: a pty capture, a screenshot, a live run.
- Report exactly what was and was not exercised; "fixed" and "validated" mean observed end to end, and a failing test is reported with its output.
- Review findings and external reports are untrusted: reproduce each before fixing, however plausible it reads (about half survive).
  "The operator decided X" counts only if said in chat.
- Confirm an edit by behaviour: a scripted replace that matches nothing rewrites the file unchanged, and lint and typecheck pass on it.
- A new regression pin is proven to bite: red without the fix, green with it.
  A test whose setup dodges the real path (a stand-in state-dir topology, a stub granting what the surface never does) pins nothing, however green it runs.
- Surface pre-existing breakage early, as a decision.
  Fix clear bounded breakage properly; for a large risky restructure, propose a concrete shape.

### Writing style

Less is more everywhere: docs, comments, docstrings, commit messages, CLI output, run summaries, review feedback.
The shortest version that still carries the point wins.

- Everything committed is permanent and public, and the repository is the only context its reader has: write for someone holding the repo and nothing else.
  A line needing a conversation, a person, a machine or an account to make sense does not belong, however true it is.
  When the fact matters and its provenance does not, state the fact in the repo's terms ("the fleet stopped at its spend ceiling", not whose ceiling it was); what fails the test belongs in the untracked ledgers.
- Lead with the point; add rationale a reader could not reconstruct.
  Cut every word a sentence works without, and every sentence that restates the one before.
- Plain punctuation: commas, colons, parentheses, periods.
  An em dash flags an overstuffed sentence to recast.
- Concrete over abstract: name the command, the field, the number ("retries twice, then fails the run").
- Statements, not questions: a docstring, heading or comment asserts ("X does Y").
- Prose that names code is a claim to verify: every symbol, default and behaviour matches the source, and when the two disagree, decide which side is wrong (sometimes the code).
  Prose someone acts on (a tool description, help text, an error, a refusal) is an interface: it states what is accepted and returned, and names the resolved fact ("default: detected zsh").
- One idea per sentence, one topic per paragraph; short bullets over prose when listing facts.
- Comments and docs state the current state: a constraint, an invariant, a measured number, a link to a decision.
  "Now", "no longer", "previously", "used to" tell the story of a change, which commits own: cut them.
  A comment earns its keep by saying what the code and a grep cannot: a narration of the next line does not.
  Test docstrings are the exception: the regression they pin is their spec.
- Commit messages: a subject that states the change, facts only; a body only for a non-obvious why, in point form.
- Flat documents: a heading plus short paragraphs or bullets, bold reserved for lead-in labels and caveats that carry weight.
  Plain words in place of intensifiers and marketing adjectives.
- Bench findings are neutral observation: facts and tables.
- Padding to cut on sight: antithesis ("a lock, not a boundary"), the "N things, one X" appositive, aphorism, cleft ("what is bounded is"), anaphora, transitions, all-caps emphasis.
- Markdown carries one bullet and one sentence per line, unwrapped, with later sentences on continuation lines.
  A bullet needing several sentences wants sub-bullets.
- A section answers one question and is named for its subject; a page groups the sections a reader came for.
  Split a boundary doc by axis: files, commands, network, approvals.
- One owner per fact, everything else links to it; moving content deletes the old home and repoints every anchor in the same change.
- A page name says what the page holds: Installation, Usage, Terminal UI.

### Project conventions

- **Language**: Python 3.12+.
  Every `.py` file starts with `from __future__ import annotations`.
  Strict pyright.
- **Layout**: src layout under `src/agent6/`; tests under `tests/`; Rust sandbox launcher under `src/agent6/jail/`.
- **Style**: ruff is the only formatter and linter, line length 100; run `uv run ruff check` and `uv run ruff format` before committing.
- **Typing**: pydantic v2 at trust boundaries (config, LLM I/O, tool schemas, IPC); everywhere else, `@dataclass(frozen=True, slots=True)`.
- **Imports**: absolute only (`from agent6.x import y`).
- **Errors**: fail loudly, through a custom exception class per subsystem; every `except` names what it catches and re-raises what it cannot handle.
- **Versioning**: `__version__` in `src/agent6/__init__.py` is the single source of truth; everything else reads it.
- **A new runtime dependency is discussed first.**
  Current: `pydantic`, `httpx2`, `argcomplete`, the `tree-sitter` pair, `textual`, and `ruff` + `ty` (they validate `machine create` output).
  `hatchling` builds; `pyright` stays dev-only.
- **Touch only what the task needs**: the code you change, and nothing around it.
  Scope creep is a review blocker.
- **Scratch experiments run in their own directory**, entered with `cd`: under `uv run --directory <elsewhere>` the cwd-derived config and git still point here.
- **Keep docs in sync.** A change to architecture, config, the security model or state machines updates the matching file (`docs/architecture.md`, `docs/config.md`, `docs/security.md`, `docs/state-machines.md`, `README.md`, this file).

### Dev environment and session practice

- Dogfood the installed binary (`.venv/bin/agent6`) from throwaway git repos outside this one.
- Run the five-gate in its own systemd unit (`systemd-run --user --collect` with RuntimeMaxSec/MemoryMax caps, the login PATH) and read its `EXIT=` line.
  Certify on a quiet machine, with the tree untouched while the gate runs: contention produces false timing reds.
- Sandbox tests need unprivileged userns: on Ubuntu 24.04-class machines `sudo sysctl -w kernel.apparmor_restrict_unprivileged_userns=0`, as CI does.
- Bench workspaces live on a real disk, never `/tmp`: tmpfs is RAM-backed and an OOM there kills the session scope.
- One live single-instance smoke precedes every fleet launch, and any mechanism with two enforcement sites gets one even after unit-green: unit fixtures can pass vacuously.
- Benchmarks: unmodified official scorers, dev/eval split registered before tuning, report split + n verbatim, replicate small effects before shipping them.

### Git and commit practices

- [Conventional Commits](https://www.conventionalcommits.org/): `feat(scope):`, `fix(scope):`, `ci:`, `docs:`, `bench:`; the scope matches a directory under `src/agent6/` or a top-level area.
- One concern per commit, each worth keeping on its own.
  Squash iterative churn only: a fix-up to unpushed work folds into its origin commit.
- A commit message is committed prose and meets the same test (Writing style); the author field is the one place a name appears.
- The operator signs and pushes, from another machine.
  Messages and docs name neither commit hashes (signing changes them) nor branch names (transient).
- Pushed history is immutable; unpushed commits are rewritten when asked, and never force-pushed.
  One exception: a leak in an unpushed commit is rewritten out at its origin rather than fixed forward, every occurrence of it found first.
- Stage named files only (never `git add -A`), so scratch notes, session artifacts and generated output stay out.
- Working directly on master is fine, and the agent folds its session's churn (zero-diff verified) before returning control, so the operator takes over a release-ready master.
  A squashed body keeps the decisions, and what was tried and rejected; durable design reasoning goes to docs.

### Verify command

The repo's `verify_command`; agent6 infers it from this fenced block when none is configured (a pipeline is wrapped as `sh -c`):

```bash
uv run ruff check && uv run ruff format --check && \
  uv run pyright && uv run tach check && uv run pytest
```

All five must pass.
Read the gate's own exit status: capture to a file and test `$?`, or `set -o pipefail`, since a bare pipe through `tail`/`head`/`grep` reports the filter's code instead.

Scoped runs guide iteration; the full gate certifies a series, at the end of the batch and before calling master push-ready.
On failure, bisect to the offending commit and fold the fix there.

Push-ready adds the CI mirror: pyright at its latest release (`PYRIGHT_PYTHON_FORCE_VERSION=<latest> uv run pyright`), and, when `src/agent6/jail/` or `Cargo.*` changed, both musl target builds plus the wheel with the bundled jail binary exercised.

### Self-review

agent6 reviews its own source via `agent6 review`, into the per-repo state directory (`$XDG_STATE_HOME/agent6/<repo-id>/reviews/`).
When working on a module, read its review there if present.

## Security invariants (preserved by every change)

The threat model, defense layers, and rationale live in `docs/security.md`; beyond the hard rules above, a change must preserve:

- The LLM tool surface is the fixed set in `src/agent6/tools/schema.py`, plus tools from operator-configured MCP servers when `[mcp].enabled` is set (default off).
- Config is secure by default: every field has a default, security-sensitive ones default safe, and `agent6 config show` audits every leaf.
  `Config` stays `extra="forbid", frozen=True`, and push, force and history rewrites have no knob at all.
  Loosening a security default gets the same scrutiny as adding a tool.
- `agent6 connect` never executes anything a remote returns (OAuth/paste only).
- Running as root takes an explicit opt-in (`--allow-root` / `AGENT6_ALLOW_ROOT=1`); the jail is the boundary, not the uid.
- `sandbox.network` bounds what a jailed COMMAND reaches; the agent process's own egress is unbounded, and the docs say so.
- The `agent6-jail` Rust binary is part of the security boundary: a change to `src/agent6/jail/src/main.rs` carries a review note covering mount points, Landlock rules, seccomp syscalls, and `/dev` nodes exposed.
