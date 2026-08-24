# Usage

Assumes agent6 is installed (see [installation](installation.md)).

## Connect a provider

```sh
agent6 connect                       # pick a provider, paste an API key
```

The key is written to `~/.config/agent6/secrets.toml` (mode `0600`) and is shared across every repository.
`agent6 connect` prompts locally and stores the key you paste; it executes nothing a remote returns.
If you are already connected, skip this step: `agent6 check` reports whether every configured provider has a key it can resolve (it never calls the provider), and `agent6 model` shows the role assignments.

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

agent6 edits files in your working tree, runs the verify command, and commits each passing step to a per-run commit chain (plus an `agent6/<id>` branch by default).
Your branch, HEAD, and index are never touched.
The run stops when the model calls `finish_session` or a budget ceiling is hit.

At a terminal the session then asks for the next input rather than ending: type the next instruction to continue in the same session, or `/exit` to finish.
Finishing leaves the session resumable like any other.
Without a terminal (CI, a detached run) the resume line is printed instead.

The verify command is the success gate.
When the repo has not set `workflow.verify_command`, agent6 infers one per run and prints what it picked, reading AGENTS.md, then a root `verify.sh`, the repo's manifest files, and loose `test_*.py` files, then a model call over those manifests.
A run that can infer nothing still proceeds, committing every editing step as an ungated checkpoint.
Pin one in the per-repo config, or with `agent6 init`, to make it deterministic.

`agent6 run` streams the run in your terminal, with no full-screen UI.
`--tui` opens the full-screen TUI instead (the run's conversation, with the dashboard on Ctrl+D; `agent6 plan --tui` does the same for a planning run), and `-i` drives the run from a stdin REPL.

## Inspect a run

`agent6 attach [<target>]` follows live: a run renders its conversation (the same view as `agent6 run`), a machine streams its state overview and reasoning.
`--raw` tails the plain event stream and `--tui` opens the full-screen TUI.
Every session id is a positional argument (an exact id or an unambiguous prefix); omit it for the most recent run.

```sh
agent6 attach                 # follow the conversation live; --raw, --tui, --json
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

State is snapshotted before each model call and checkpointed per turn.
`fork` rolls a copy back to a turn and continues it as a new run, leaving the original unchanged.

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

- `agent6 review --reviewers 3 --personas security,correctness,tests` runs a panel whose findings are checked against the diff, so only real problems gate.
- `ask` runs in any directory; `run` and `plan` need a git repository for branches, diffs, and merges.

## Run options

- `--preset <name>` selects a strategy preset (`standard`, `quick`, `ultra`, `paranoid`, or your own; the [presets table](config.md#presets) says what each sets).
  `agent6 config presets` lists them and `agent6 config set preset <name>` persists one.
  A preset cannot change mid-run; `agent6 resume <id> --preset <name>` continues a stopped run under another one and records it for later resumes.
- `agent6 run "task" --parallel 3` (or `model-a,model-b`) fans out isolated lanes and prints a ranked comparison.
  The same fan-out spawns from the TUI and web composers, or mid-run with the `/parallel [spec] <task>` steer directive ([configuration](config.md#parallel)).
- `agent6 run "task" --standing "hunt and fix bugs"` adds a standing goal: a never-finishing fallback task the run re-enters whenever the ordinary queue drains or the worker tries to stop.
  New work always outranks it, a standing task never passes (retire it as skipped or obsolete), and the run still ends on its budget, an operator stop, or its iteration cap.
- `agent6 prompt show [--mode run|plan|ask] [--json]` prints what the model receives on a run's first call here: the system prompt, the tool definitions this config exposes (name, description, input schema), and the first user message around the task.

## Configuration

Config is layered, lowest precedence first: built-in defaults, the global `~/.config/agent6/config.toml`, the per-repo config, then `--config FILE`.
Every field has a default and the security-sensitive ones default to the safe value, so a repo can be zero-config when the global config supplies a provider and model.
`agent6 config show` prints every effective value with the layer that set it, and the [configuration reference](config.md) documents each field.
