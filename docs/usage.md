# Usage

Assumes agent6 is installed (see [installation](installation.md)).

## Connect a provider

```sh
agent6 connect                       # pick a provider, paste an API key
```

The key lands in `~/.config/agent6/secrets.toml` (mode `0600`), shared across every repository.

- `agent6 connect` prompts locally; it executes nothing a remote returns
- already connected: skip this step
- `agent6 check`: every configured provider's key resolves (never calls the provider)
- `agent6 model`: the role assignments

agent6 routes three model roles independently:

| Role       | Set with            | Used by                                      |
| ---------- | ------------------- | -------------------------------------------- |
| `worker`   | `[models.worker]`   | `agent6 run` and `agent6 resume`             |
| `reviewer` | `[models.reviewer]` | `agent6 review` and the in-loop review panel |
| `planner`  | `[models.planner]`  | `agent6 plan`                                |

`reviewer` and `planner` fall back to `worker` when unset.

```sh
agent6 model worker anthropic claude-sonnet-5
agent6 model all openrouter moonshotai/kimi-k2.6   # set every role at once
```

## Your first run

```sh
cd your-repo
agent6 run "add a --json output mode to the CLI"
```

agent6 edits your working tree, commits each step to a per-run chain, and certifies the finished tree with the verify command.

- your branch, HEAD, and index are never touched (the chain gets an `agent6/<id>` branch by default)
- commands prompt for approval under the default `sandbox.run_commands = "ask"`: allow one call or the whole session; a headless run auto-denies, a hub-spawned one parks the prompt for a front-end
- the run ends when the model declares it finished or a budget ceiling stops it
- at a terminal it then asks for the next input: type to continue the session, `/exit` to finish (still resumable)
- without a terminal (CI, detached) the resume line prints instead

The verify command is the success gate.

- unset `workflow.verify_command`: inferred per run and printed (AGENTS.md, a root `verify.sh`, manifest files, loose `test_*.py`, then a model call)
- nothing inferable: the run proceeds gateless, committing each editing step
- pin one (per-repo config or `agent6 init`) to make it deterministic
- the harness runs it when the model finishes over an uncertified tree; a red returns to the model with the output (`workflow.verify_retries`, default 2), then the run ends red
- `workflow.verify_when` moves the harness run to every editing step (`step`) or leaves every run to the model (`never`); the model can always run it itself

`agent6 run` streams in your terminal, no full-screen UI.

- `--tui`: the full-screen conversation view, dashboard on Ctrl+D (`agent6 plan --tui` for a planning run)
- `-i`: drive the run from a stdin REPL

## Inspect a run

`agent6 attach [<target>]` follows live.

- a run renders its conversation (the `agent6 run` view); a machine streams its state overview and reasoning
- `--raw` tails the event stream; `--tui` opens the full-screen TUI
- session ids are positional, exact or an unambiguous prefix; omit for the most recent run

```sh
agent6 attach                 # follow the conversation live; --raw, --tui, --json
agent6 steer ID "wrap up now" # queue a steering instruction for a live run (cron-friendly)
agent6 sessions show          # status, iteration, elapsed, cost, where the changes are; --json to script
agent6 sessions diff          # the git diff the run produced
agent6 sessions commits       # the run's per-step commits
agent6 sessions merge         # land the run's work on your branch
agent6 sessions prune         # delete merged agent6/* branches; report the rest
agent6 sessions dir           # where this repo's run history lives (scriptable)
agent6 sessions rm            # delete one run's history; --asks clears saved asks
agent6 sessions compare <ids> # ranked comparison: >=2 runs, or one fan-out id (its lanes)
agent6 sessions transcript    # the full conversation, every tool call with I/O
agent6 sessions graph         # the persisted task graph
```

`agent6 history search <query>` greps across the transcripts of every run.
`agent6 ps` lists the live sessions of every repository on the machine, with the directory to cd to and the id to attach.

## When a run goes wrong

```sh
agent6 resume <session-id>           # continue from the last snapshot
agent6 fork <session-id> --at-turn 7 # new run from turn 7 (--steer seeds it)
```

- state is snapshotted before each model call and checkpointed per turn
- `fork` rolls a copy back to a turn and continues it as a new run; the original is unchanged

Exit codes for `agent6 run` and `resume`, for scripts to branch on:

| Code | Meaning |
|---|---|
| `0` | finished with a green gate, or nothing to gate on |
| `1` | the run broke (crash, provider error) |
| `2` | operator error (bad flag or config) |
| `3` | budget exhausted |
| `4` | finished over a red or never-run verify gate |
| `5` | finished, but no commit landed and the edits sit uncommitted (a run that changed nothing stays `0`) |
| `130` | interrupted |

## Plan, review, and ask

```sh
agent6 plan "refactor the config loader"      # edit-free plan; run --from-plan
agent6 plan edit <session-id>                 # answer the plan's open questions
agent6 resume <session-id> --steer "answered" # the planner re-reads and revises
agent6 review --base origin/main --head HEAD  # read-only diff review
agent6 ask "how does the task-graph curator work?"
```

- `agent6 review --reviewers 3 --personas security,correctness,tests`: a panel whose findings are checked against the diff, so only real problems gate
- `ask` runs in any directory
- `run` and `plan` need a git repository (branches, diffs, merges)

## Run options

- `--preset <name>`: a strategy preset (`standard`, `quick`, `ultra`, `paranoid`, or your own; the [presets table](config.md#presets) says what each sets)
  - `agent6 config presets` lists them; `agent6 config set preset <name>` persists one
  - a preset cannot change mid-run; `agent6 resume <id> --preset <name>` continues a stopped run under another
- `--parallel 3` (or `model-a,model-b`): isolated fan-out lanes, auto-compared into a ranked report
  - also from the TUI and web composers, or mid-run via the `/parallel [spec] <task>` steer directive ([configuration](config.md#parallel))
- `--standing "hunt and fix bugs"`: a never-finishing fallback task the run re-enters when the queue drains
  - new work outranks it; it never passes (retire as skipped or obsolete); budget, stop, and the iteration cap still end the run
- `agent6 prompt show [--mode run|plan|ask] [--json]`: everything the model receives on the first call (system prompt, tool definitions, the first user message)

## Configuration

Config is layered, lowest precedence first: built-in defaults, the global `~/.config/agent6/config.toml`, the per-repo config, then `--config FILE`.

- every field has a default; security-sensitive fields default safe (a repo can be zero-config)
- `agent6 config show`: every effective value with the layer that set it
- the [configuration reference](config.md) documents each field
