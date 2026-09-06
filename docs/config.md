<!-- Generated from docs/config_template.md by docs/gen_config.py; edit those, then regenerate. -->
# Configuration

Every field has a default and the security-sensitive ones default to the safe value, so you set only what you want to change.
This is the field reference; the [security model](security.md) covers what `[sandbox]` and `[git]` enforce.

## Where config lives

The layers, lowest precedence first:

| Layer | Path | Set with |
|---|---|---|
| built-in defaults | (none) | (secure defaults, always present) |
| global *(default location)* | `$XDG_CONFIG_HOME/agent6/config.toml` | `agent6 connect`, `agent6 model` |
| per-repo *(override)* | `<state-dir>/<repo-id>/config.toml` | `agent6 init`, `agent6 config set --repo` |
| explicit | `--config FILE` | `agent6 run --config FILE` |
| machine overlay | a machine file's `[config]` table | `agent6 config set --machine-file FILE` |

The per-repo config lives in the state dir, out of the workspace: per-machine, never committed.
It can be empty or absent when the global config supplies a provider and model; `workflow.verify_command` is inferred per run when unset.

## Creating and inspecting

- `agent6 connect`: add a provider + API key (stored `0600`), global.
- `agent6 model <role> <provider> <model> [--effort off|low|medium|high|xhigh|max]`.
- `agent6 init`: optional setup wizard (per-repo config, inferred `verify_command`, `.gitignore`, `AGENTS.md`); every step asks first.
- `agent6 config show`: every effective value and which layer set it.
  `--descriptions` adds each value's meaning under its row; `config show <key>...` prints the named keys (or sections) untruncated, meaning included.
- `agent6 config set|unset|add|remove <dotted.key> [value]` (`--repo`, or `--machine-file FILE` for a machine `[config]` overlay); `agent6 config get <dotted.key>` prints the effective value and its layer.
  Every edit is re-validated and rolled back if invalid.
  A sibling pair that must move together is set as one inline table: `agent6 config set context '{ drop_at_chars = 200000, summarise_at_chars = 400000 }'`.
- Writes are atomic; a blocked edit lock never blocks the write (worst case one lost update, reported as "kept as written").
  A symlinked config file is followed only when you own the target.
- `agent6 config fill`: materialize defaults + global config into the global file.
  The repo layer and any selected preset are left as-is.
- `agent6 config fix`: drop invalid entries (unknown keys, stale values), naming each; `--machine-file FILE` repairs an overlay instead.
- `agent6 check`: validate config + sandbox + provider keys without running.
  `config show` prints what an `auto` knob resolved to on this host, tagged `(adaptive)`; `check` adds why a level fell short, live jail probes, and each MCP server's network and `approve`.

---

## `[agent6]`

| Field | Default | Meaning |
|---|---|---|
| `config_version` | `1` | Config schema version; only `1` is accepted. |

## `[providers.<name>]`

One backend per block; `<name>` is referenced from `[models.<role>]`.
Three orthogonal choices describe any backend: **`api_format`** (the wire dialect, the only field that selects code), **`deployment`** (URL/placement quirks of where it is hosted), and **auth** (`auth_style` + `api_key_env` or `token_command`).
A minimal block is just `api_format` (plus `base_url` for a non-default host).

| Field | Default | Meaning |
|---|---|---|
| `api_format` | *(required)* | The wire format: `anthropic` (the Messages API), `openai` (Chat Completions: OpenAI, OpenRouter, Ollama, vLLM, LM Studio, llama.cpp, Gemini's OpenAI endpoint), `chatgpt` (the ChatGPT-subscription Codex backend, Responses API), or `claude_code` (the installed, signed-in Claude Code binary on a Claude subscription; no HTTP endpoint, no key). |
| `deployment` | `"direct"` | `direct`, `vertex` (Google Vertex AI), or `azure` (Azure OpenAI; `openai` format only): the URL shape and where the model name and API version go. |
| `base_url` | per (format, deployment) | The endpoint's host and path prefix (`https://api.anthropic.com/v1`); required for `vertex` and `azure`. Its host is the only network destination the agent dials for this provider. |
| `auth_style` | per (format, deployment) | How the key is sent: `x_api_key` (Anthropic), `bearer` (`Authorization: Bearer`, the OpenAI style), `api_key_header` (Azure), or `none` (an unauthenticated local endpoint). `agent6 connect` sets it. |
| `api_key_env` | none | The environment variable holding the API key; it wins over `secrets.toml`. Unset for a key `agent6 connect` stored, or an unauthenticated local endpoint. |
| `token_command` | `[]` | A command (argv) that prints a short-lived bearer token to stdout, re-run when `token_command_ttl_s` expires and once after a `401` or `403`. Wins over `api_key_env`. |
| `token_command_ttl_s` | `300.0` | Seconds a `token_command` token is reused before the command runs again. |
| `extra_headers` | `{}` | Extra HTTP headers on every request to this provider. Never a secret: the config file is not `0600`. |
| `extra_body` | `{}` | Provider-specific JSON merged last into every request body, so tuning keys (`max_tokens`, `temperature`) win; the structural keys agent6 owns (messages, model, stream, tools, tool choice, response shape) are filtered out. Values must be JSON-shaped (a TOML date or time is refused). OpenRouter's routing options go here. |
| `extra_query` | `{}` | Extra URL query parameters on every request (Azure's `api-version`). |
| `http_timeout_s` | `600.0` | Seconds one HTTP call may take to read or write; the connect phase is bounded at 20 s regardless. |
| `prompt_caching` | `true` | Anthropic prompt caching: the system prompt, the tools, and the growing conversation are re-read at 0.1x the input price. `anthropic` format only. |
| `binary` | `"claude"` | The Claude Code executable: a name on PATH or an absolute path. `claude_code` format only. |

### Deployments

```toml
# Anthropic direct (default): equivalent to a bare api_format = "anthropic"
[providers.anthropic]
api_format = "anthropic"

# Gemini on Vertex (OpenAI-compatible endpoint)
[providers.vertex-gemini]
api_format = "openai"
deployment = "vertex"
base_url = "https://LOCATION-aiplatform.googleapis.com/v1/projects/PROJ/locations/LOCATION/endpoints/openapi"
token_command = ["gcloud", "auth", "print-access-token"]

# Claude on Vertex (model in the URL, anthropic_version in the body: handled
# by deployment = "vertex")
[providers.vertex-claude]
api_format = "anthropic"
deployment = "vertex"
base_url = "https://LOCATION-aiplatform.googleapis.com/v1/projects/PROJ/locations/LOCATION/publishers/anthropic/models"
token_command = ["gcloud", "auth", "print-access-token"]

# Azure OpenAI (the model id is the deployment name; api-version required)
[providers.azure]
api_format = "openai"
deployment = "azure"
base_url = "https://RESOURCE.openai.azure.com"
api_key_env = "AZURE_OPENAI_API_KEY"
extra_query = { "api-version" = "2024-06-01" }
```

### ChatGPT subscription (`api_format = "chatgpt"`)

Uses a ChatGPT plan (Plus/Pro/Business) instead of an API key, over the Codex Responses backend.

```bash
agent6 connect chatgpt      # browser sign-in (paste fallback when headless)
agent6 model worker chatgpt gpt-5.6-sol
```

- `agent6 connect chatgpt` runs a PKCE OAuth sign-in against OpenAI's fixed OAuth authority (`https://auth.openai.com`, a constant, not config) and stores the tokens in `secrets.toml` (0600); they refresh automatically.
- Usage draws on the plan's own limits; cost meters show an authoritative $0 and the tokens meter nothing, so `[budget].max_percent` is the ledger that bounds these calls.
- Past the included window, calls draw on PURCHASED credits (real money; auto top-up can buy more).
  `[budget].allow_paid_credits = false` (the default) is a circuit breaker: a usage preflight before the first call and every response's headers report both windows and the credit state, and once a window is exhausted with credits present the run stops at its next boundary; a call already in flight completes, so a boundary-crossing call can spend before the stop.
- Whether these conversations train OpenAI's models follows the ChatGPT account's own data controls (Settings > Data controls > "Improve the model for everyone"); agent6 cannot change that setting.
  agent6 never calls the feedback/rating endpoints, which would opt the rated turns into training regardless of it; there is no rating surface.
- Model names complete from the backend's own listing for the signed-in plan (fetched like other providers' catalogs, never a static list), and its context windows size compaction.
- `agent6 connect chatgpt --logout` signs out: the grant is revoked at the OAuth authority (best effort) and the tokens leave `secrets.toml`.
- Spend is plan-metered, not dollar-metered: every response carries the account's rate-limit window, surfaces show `plan usage: N% of the 7-day window`, and `[budget].max_percent` caps the points one run may consume (`--max-percent` per run). Dollar figures stay an authoritative $0.

### Claude Code subscription (`api_format = "claude_code"`)

Runs the worker inside the installed, signed-in Claude Code binary: a Claude subscription (Pro/Max) instead of an API key.

```bash
claude auth login                        # once, in Claude Code itself
agent6 connect claude                    # checks the sign-in, writes [providers.claude]
agent6 model worker claude claude-sonnet-4-5
```

- agent6's own loop, tools, jail, verify gate, and approvals drive the run; the binary supplies the model.
  Inside it every Claude Code capability is off: built-in tools, hooks, settings, CLAUDE.md, MCP servers, skills, slash commands, session files, auto-memory, auto-compaction.
- `binary` names the executable (default `claude`, resolved on PATH).
  agent6 never reads the login under `~/.claude`; a `CLAUDE_CONFIG_DIR` in the environment selects a relocated one.
- Spend is plan-metered, not dollar-metered: every round reports the account's 5-hour and 7-day windows, surfaces show the fuller one as `plan usage: N% of the 7-day window (seven_day)`, and `[budget].max_percent` caps the points one run may consume (`--max-percent` per run).
  Dollar figures stay an authoritative $0; the binary's own list-price estimate is not recorded.
- Ignored: `[models.<role>].temperature` and the loop's per-call output-token cap (the binary owns sampling).
  Refused: `effort = "off"` (`--effort` has no off value; use `low`).
- Side roles keep their own providers; route one here explicitly (`agent6 model reviewer claude claude-haiku-4-5`).
  Each side call is one short-lived `claude` process.
- One `claude` process serves a worker leg.
  It restarts, replaying the conversation as one text message, on resume, fork, `/undo`, a steer or stop mid-turn, a tier-2 context restart, and when the live context nears the window.
  Tier-1 compaction shrinks the model's context at that next restart, not before.
- Claude Code appends the account email to every system prompt it sends; agent6 replaces it with `<operator-email>` in the model's returned text.
- Tool results are capped at 34,000 bytes of UTF-8 for this provider (60,000 elsewhere): Claude Code writes a result over 50,000 bytes under `~/.claude/projects` and hands the model a 2 KB preview of it, and the turn's notices (a verify tail, a review critique, the nudges) ride in the same payload, with 16,000 bytes kept for them.
- Use a full model id (`claude-sonnet-4-5`, not `sonnet`) so the context window is known for compaction sizing.
- `agent6 connect claude --logout` is refused: agent6 stores no Claude Code credentials; `claude auth logout` signs out.

### OpenRouter routing and caching (`extra_body`)

OpenRouter's default routing is not deterministic, so prompt caching may or may not engage call-to-call.
Pin it with `extra_body.provider` ([routing docs](https://openrouter.ai/docs/features/provider-routing)):

```toml
[providers.openrouter]
api_format = "openai"
base_url = "https://openrouter.ai/api/v1"
extra_body = { provider = { sort = "throughput" } }  # prefer fast backends
# Alternatives: { order = ["DeepInfra"], allow_fallbacks = true }
#               { max_price = { prompt = 1, completion = 2 } }
```

```bash
agent6 config set providers.openrouter.extra_body \
  '{ provider = { sort = "throughput" } }'
```

Caching matters more than payload size: the large per-call input is the same prefix every turn.
Watch `cache_r` in the cost summary to confirm it engages.

### Short-lived bearer tokens (`token_command`)

For endpoints authenticated by a refreshed bearer rather than a static key (Vertex OAuth, OIDC/STS gateways): point `token_command` at anything that prints a current token, e.g. `["gcloud", "auth", "print-access-token"]` or `["az", "account", "get-access-token", "--query", "accessToken", "-o", "tsv"]`.
Cached `token_command_ttl_s` seconds, re-run once on `401`/`403`.
It runs in agent6's own process with your environment (operator-only, same trust as an MCP `command`); non-zero exit, timeout, or empty output surfaces as a provider error.

## `[models.<role>]`

Role routing. **`worker`** drives `run`/`resume` (its pricing also drives the USD→token budget conversion); **`planner`** drives `plan`; **`reviewer`** drives `review`, the in-loop review panel, the context summariser and gister, and the prompt reviser.
`planner`/`reviewer` fall back to `worker`.
Cross-vendor mixes are fine.

<!-- the three roles are the same shape, so the table is rendered once -->
| Field | Default | Meaning |
|---|---|---|
| `provider` | *(required)* | A `[providers.<name>]` entry, by name. |
| `model` | *(required)* | Model id as that provider names it (`agent6 model` lists them). |
| `temperature` | `0.0` | Sampling temperature pinned on every call, `0.0` to `2.0`. `0.0` keeps tool use stable; unset leaves the provider's default. |
| `effort` | none | Reasoning effort: `off`, `low`, `medium`, `high`, `xhigh`, or `max` (the top tiers where the model offers them; Anthropic collapses them to its highest). Unset: what the wire applies, which `agent6 config show` prints resolved (`low` on openai-compatible reasoning models, no thinking on Anthropic). |

## `[sandbox]`

The field summary; the model is in security.md: [Sandbox](security.md#2-sandbox) and [Network](security.md#5-network).

`extra_read_paths`, `extra_write_paths` and `hide_paths` take absolute paths with no `..` segment, and `extra_device_paths` only paths under `/dev`; anything else refuses at config load.

| Field | Default | Meaning |
|---|---|---|
| `isolation` | `"auto"` | How jailed commands are confined: `strict` (user + mount namespaces, Landlock, seccomp), `hardened` (Landlock + seccomp, no namespaces), or `none` (unconfined). `auto` picks the strongest the host supports and says so when that is `none`. An explicit `strict` or `hardened` refuses to start where the host cannot honor it. `none` also via `--dangerously-disable-sandbox` or `AGENT6_DANGEROUSLY_DISABLE_SANDBOX=1`. |
| `network` | `"auto"` | Which network jailed commands join. `session`: the run's private network (commands reach each other, nothing off the box, nothing outside reaches in), refused where it cannot be enforced. `host`: the machine's network. `only_explicit_states`: strict only, machine `tool` states opt in. `auto`: `session` under `strict`, degraded to the host's network with a warning under `hardened` or `none`. A run's commands share one launcher, so there is no per-command `none`. |
| `run_commands` | `"ask"` | Whether the model may run commands (`run_command`, `run_verify_command`, `run_metric_command`, `stop_background`, one decision for all four): `yes` runs them, `no` withholds the tools (and the verify gate with them), `ask` prompts per call with allow-for-this-session answers. `ask` and `plan` clamp `yes` to `ask`. Per invocation: `--auto-approve` (never over a configured `no`), `--no-commands`. A run set to `ask` with nobody to answer refuses to start. |
| `fetch_hosts` | `[]` | Hosts the `fetch` tool reads without asking; any other host prompts, and an absent operator is a no. Empty: every fetch prompts. `["*"]`: any host. A leading dot allows subdomains (`.readthedocs.io`). Each entry is a host, never a URL prefix; the rest of fetch is fixed (https only, 1 MiB cap, redirects returned, not followed). Hidden when a jailed command already has the host network (`network = "host"`, or any isolation but `strict`); withheld from machine and agent states. |
| `protect_git` | `true` | Keep `.git/` unwritable by jailed commands, so a command cannot plant a git filter that agent6's host-side commits would execute. Needs a mount namespace: `strict` only. Under `hardened` the default `true` degrades with a warning; an explicit `true` refuses to start. The in-process edit tools refuse `.git` writes at every level regardless. |
| `home` | `"tmp"` | The HOME jailed commands get under `strict`: `tmp` is `/tmp/agent6-home` inside the run's private tmpfs, gone with the run; `cache` is the persistent `$XDG_CACHE_HOME/agent6/home` (created `0700`, refused once loosened), bind-mounted read-write at its real path. `hardened` and `none` have no private tmpfs and always use the cache dir; an explicit `tmp` refuses to start there. Persistence is a cross-run channel inside the jail's world: a poisoned cache or a `~/.gitconfig` alias written by one run reaches the next jailed run, never your own tools. |
| `memory_limit_mb` | `0` (off) | `RLIMIT_DATA` cap in MiB on each jailed process (inherited by its children). `0`: no cap. Set one to bound a specific task; a process over it fails as an ordinary command error. |
| `extra_read_paths` | `[]` | Absolute paths outside the repo the run may read and execute, at their real locations: a toolchain, an interpreter (conda, Go, Rust, Node), a shared data dir. Mounted for jailed commands and readable by the in-process tools. Widens the sandbox; list only what the build needs. |
| `extra_write_paths` | `[]` | Absolute paths outside the repo the run may read and write, at their real locations: a build cache, an output dir, a sibling checkout the task edits. Write implies read. Widens the sandbox; list only what the task writes. |
| `extra_device_paths` | `[]` | Device nodes under /dev the jail exposes read-write (GPU compute: /dev/nvidiactl, /dev/nvidia0, /dev/nvidia-uvm). Empty (the default) keeps the device wall: strict's /dev holds only null/zero/urandom/random/full. Each path must be an existing character or block device at run start, or the run refuses. Widens the sandbox: a device node is direct hardware access. |
| `hide_paths` | `[]` | Absolute paths the run may never read or write, even under a broader grant. agent6's config dir and state base are always hidden, so an `extra_read_paths` grant of `$HOME` never exposes `secrets.toml` or run history (the data dir and cache stay readable: installed skills work). Enforced twice: the in-process tools refuse them at every isolation level, and jailed commands see them masked (a dir reads empty, a file reads empty). Masking needs the mount namespace: under `hardened` an entry it cannot mask refuses the run, and a grant exposing the always-hidden dirs warns loudly instead. |

## `[git]`

| Field | Default | Meaning |
|---|---|---|
| `dirty_tree` | `"ask"` | What a run does with tracked files' uncommitted changes at start. `ask`: ask over the ask_user channel (`stash` them for the run, `include` them in its commits, or `cancel`, which parks the run for a later resume); a run nobody can answer refuses to start. `stash`: stash them without asking; at the end the stash is applied back per `auto_stash_pop`, else its `git stash apply <sha>` line is printed. `include`: start without asking, the run's first commit records them. Untracked files never count and are never committed. `--parallel` fans out under `stash` or `include` and refuses under `ask`. |
| `auto_stash_pop` | `false` | Apply the pre-run stash back when the run ends and the tree is clean (a clean apply, no conflicts). On any doubt the stash stays and the apply line is printed. Never `reset --hard`. Requires `dirty_tree = "stash"`. |
| `control` | `"agent6"` | Who manages git during a run: `agent6` records every step on the run's own commit chain and branch, never touching HEAD; `model` hands git to the model: no per-step chain, no run branch, the model's own commits and branches are the record, and `sessions diff`/`merge`, `/undo`, and `fork` refuse for such runs. Requires `sandbox.protect_git = false`. |
| `branch_per_run` | `true` | Also advance a visible `agent6/<run-id>` branch to the run's chain tip; `false` keeps only the hidden `refs/agent6/<run-id>/head` ref. Forced on for `--parallel` lanes (their work is imported by branch). |
| `commit_per_step` | `true` | Commit each editing step onto the run's detached chain (a temp index; HEAD, your index, and your checkout are never touched). `false`: agent6 never commits; the work stays only in the worktree, and resume-from-git, `sessions diff`/`merge`, and `/parallel` dispatch from a changed tree degrade. |
| `merge_strategy` | `"squash"` | How `agent6 sessions merge` lands a run on its base: `squash` (one commit), `merge` (a `--no-ff` merge that keeps the per-step history), or `ff` (fast-forward). Consolidation only; per-step commits always land on the run's chain. |
| `auto_merge` | `false` | After a run that finished with nothing red, merge its work into its base branch automatically (never over a red or stale verify). With `branch_per_run` off it merges the hidden chain ref. On a conflict nothing moves and the instructions are printed. |
| `auto_prune` | `false` | After an `auto_merge`, delete the run branch when `git branch -d` can (a `merge` or `ff` merge). A squash-merged branch is reported with its `-D` line, never force-deleted. Requires `auto_merge`; nothing to do without a run branch. |
| `run_repo_hooks` | `false` | Run the repo's own `.git/hooks/*` during agent6's git operations. `false` skips them: a repo hook is repo-controlled code that would run on the host. `core.fsmonitor` and `diff.external` are always neutralized. |
| `run_repo_filters` | `false` | Honor the repo's content drivers (`filter.<name>.clean/smudge/process`, `merge.<name>.driver`) during agent6's git operations. `false` neutralizes each by name: a driver defined in `.git/config` is repo-controlled code that would run on the host at every commit. `true` is what Git LFS needs (its clean/smudge filters are these drivers). |

### `[git.commit]`

| Field | Default | Meaning |
|---|---|---|
| `name` | none | Author and committer name on the commits agent6 makes; unset uses the repo's own `git config`. A run with no resolvable identity refuses to start. |
| `email` | none | Author and committer email on the commits agent6 makes; unset uses the repo's own `git config`. A run with no resolvable identity refuses to start. |
| `trailer` | `""` | A git trailer line (`Key: value`) appended to every commit agent6 makes, e.g. `"Assisted-by: agent6:{model}"` or `"Co-authored-by: agent6:{model} <noreply@agent6.dev>"`. `{model}` is the model that wrote the code (several are joined with `, `). Empty: no trailer. |

### `[git.commit.checkpoint]` and `[git.commit.squash]`

| Field | Default | Meaning |
|---|---|---|
| `checkpoint.message` | `"agent6"` | The message of each per-step commit: `agent6` (`agent6 iter N: <summary>`), `conventional` (a `type(scope): subject` derived from the diff, no model call), or `model` (the model writes it from the git facts, falling back to `agent6` with a warning on any failure). |
| `squash.message` | `"agent6"` | The message of the one commit a squash merge produces: `agent6` (`agent6 iter N: <summary>` style), `conventional` (a `type(scope): subject` derived from the diff, no model call), `combine` (git's own squash message: the per-step log concatenated), or `model` (model-written, falling back to `agent6` with a warning on any failure). |

## `preset` (top-level)

| Field | Default | Meaning |
|---|---|---|
| `preset` | `""` | The strategy preset in force: `standard` (plain defaults), `quick` (no review panel), `ultra` (a three-seat panel that advises and vetoes before finish), `paranoid` (five explore-tier seats), or a `[presets.<name>]` of your own. Fills many settings at once and overrides every section of the layer that selects it; `--preset` overrides per run, `resume --preset` per resumed leg. Empty: no preset. |

## `[workflow]`

| Field | Default | Meaning |
|---|---|---|
| `verify_command` | `[]` | The command that decides whether a step succeeded, as argv (no shell; wrap a pipeline as `["sh", "-c", "a && b"]`). Set it to pin the gate. Unset: each run infers one and prints it (an AGENTS.md `## Verify command` block first, then a root `verify.sh`, the repo's manifest files, and loose `test_*.py` files, then a model call over those manifests); a run that can infer none starts gateless and adopts the first gate a recognizable project created mid-run yields. |
| `verify_infer` | `true` | Infer a verify command when `verify_command` is unset (AGENTS.md fence, repo signals, a model call), and adopt one mid-run when a gateless run materializes a recognizable project; an adopted gate that cannot run (exit 127, or the module its `-m` names is missing) is dropped again, never re-adopted. false: such a run stays gateless, no inference and no adoption; a set `verify_command` is unaffected. |
| `verify_timeout_s` | `600.0` | Seconds one `verify_command` or `metric.command` call may take before it is killed and counted as failed. A pytest gate naming no paths that overruns this budget (the harness's run or the model's own `run_verify_command`) re-runs scoped to the test files nearest the run's diff, and harness gates run scoped until a full run of the gate passes; a scoped green ends the run `passed · scoped gate`. A model-chosen `run_command` is not bounded (see `command_checkin_s`). |
| `max_iterations` | `200` | Assistant turns one leg may take before the run stops with reason `max_iterations`; -1 is unlimited. A resumed leg gets a fresh allowance. |
| `command_checkin_s` | `900.0` | Seconds a model's `run_command` may run before it is handed back as a background job. Not a timeout: nothing is killed, the command keeps running, and the model is told (`returncode: null`, `still_running: true`, a `background_id`) so it can wait with `read_background`, stop it, or carry on. `0` disables the hand-back. |
| `standing_patience` | `-1` | Consecutive fruitless standing-goal re-entries (rounds with no executed tool call) the run absorbs before soft ends are honoured. `-1`: never on its own (the run ends on its budget, iteration cap, or an operator stop); `0`: the first fruitless round ends it; `N`: N fruitless re-entries get an escalating nudge, then ends are honoured. A round that lands work resets the streak. |
| `verify_when` | `"finish"` | When the harness runs `verify_command` itself: `finish` (when the model calls `finish_session` and the tree changed since the last green run), `step` (also after every turn that edits the tree), `never` (only the model's own `run_verify_command` calls run it). The tool stays available in every mode; a run with no verify command has no gate to run. |
| `verify_retries` | `2` | How many times a red finish certification returns to the model with the gate's output before the finish stands and the run reads `finished · gate red`, never passed. `0`: the first red ends the run. A gate that was red before the run touched anything is not returned. |

## `[review]`

| Field | Default | Meaning |
|---|---|---|
| `trigger` | `"off"` | When the in-loop review panel runs on the diff so far and its findings reach the model as a message: `off` (never), `on_verify_fail` (after each failed verify), `before_finish` (when the model calls `finish_session`; a gating `decision` can reject the finish), or `periodic` (every `period` iterations). With no `seats` the panel is one reviewer seat on `[models.reviewer]`, the model `agent6 review` uses. |
| `period` | `10` | Iterations between panels when `trigger = "periodic"`. |
| `decision` | `"advisory"` | What a panel's BLOCK verdicts do: `advisory` (the findings are injected as guidance, nothing is blocked), `veto` (one blocking seat rejects the finish), `quorum` (`quorum` distinct models must block), or `all` (every seat must block). A gate applies to `before_finish` only; the other triggers always advise. |
| `quorum` | `2` | How many seats must block for `decision = "quorum"`, counted per distinct model (two seats on one model count once, so a same-model panel cannot reach it). |
| `max_total_rejections` | `4` | How many finishes a gating panel may reject per run before it disarms to `advisory` for the rest of the run, so a panel can never stall a run forever. |
| `budget_fraction` | `0.25` | Skip the panel (the finish is accepted) once the run's remaining budget falls below this fraction of the whole. `0.25`: no panel in the last quarter. |
| `seats` | `[]` | The panel roster, one entry per seat: a persona name (`"security"`), routed via `[models.reviewer]`, or `"<persona>@<provider>/<model>"` to pin a model per seat (`"correctness@openrouter/moonshotai/kimi-k2"`). A persona is any short stance the reviewer's prompt adopts; the built-in set cycled when none is named is `security`, `correctness`, `tests`, `over-engineering`, `edge-cases`. Empty: one reviewer seat when `trigger` is on. `agent6 review --reviewers N --personas ...` builds the same roster for a one-off review. |
| `concurrency` | `1` | How many seats the in-loop panel runs at once (`1` = one after another; the panel's latency is its slowest seat). `agent6 review` always runs every seat in parallel. |
| `tier` | `"diff"` | How much a seat reads: `diff` (one call over the diff, the task, and the verify result) or `explore` (a read-only tool-using reviewer that also reads the repo around the diff to catch cross-file impact; several calls per seat, and it reads the checkout, so the reviewed head must be checked out with a clean tree). |

A `block` gates only when its `file:line` is in the diff and its category is one of `security`, `sandbox-bypass`, `off-topic-edit`, `data-loss`, `verify-uncovered-correctness`; every other finding is advisory and cannot stall the run.

## `[context]`

Tiered context compaction (approximate chars; tokens ≈ chars/4).

| Field | Default | Meaning |
|---|---|---|
| `drop_at_chars` | _adaptive_ | Tier-1 compaction threshold: once the accumulated tool results exceed this many characters (about 4 per token), the oldest results are replaced by short placeholders the model can re-fetch. Unset: sized from the model's context window (about 45% of it); set both thresholds to pin them. |
| `summarise_at_chars` | _adaptive_ | Tier-2 compaction threshold: once the whole context exceeds this many characters, the elided history is summarized and the conversation restarts on the summary (the task DAG survives). Unset: the model's window minus a 16k-token reserve. Must exceed `drop_at_chars`. |
| `keep_recent_chars` | `80000` | How many characters of the most recent history a tier-2 restart keeps verbatim after the summary. `0` keeps none. |
| `keep_thinking_turns` | `0` | At tier-1 moments, drop the model's thinking from assistant turns older than this many turns. `0` keeps all thinking. Wires that re-send thinking (Anthropic's signed blocks, ChatGPT's reasoning items) replay less; the OpenAI wire never re-sends it. |
| `summary_max_tokens` | `2048` | Cap on the tokens a tier-2 summary (and a gist distillation) may produce. A reasoning model's per-call floor (room for its reasoning tokens) overrides a smaller cap, and the chatgpt backend takes no cap. |
| `elision_gists` | `true` | At tier 1, replace a large `read_file` result with a model-written gist before the bare placeholder (the gist is dropped too under continued pressure, so the byte bound holds). `false`: straight to bare placeholders. |

## `[prompt]`

| Field | Default | Meaning |
|---|---|---|
| `system_prompt_file` | `""` | Path of a file that replaces run mode's built-in base system prompt (the dynamic blocks still append). The tool contracts become yours to state; a file missing the core tool names is warned about at startup. Empty: the built-in base. `agent6 prompt show` prints the assembled prompt, the tool definitions, and the first message. |
| `structural_priors` | `true` | Include the `<repo-priors>` block in the system prompt: the repo map, the symbol outline, co-change and hot-symbol hints, recent commits. `false` for a leaner, cheaper prompt. |
| `revise_prompt` | `"off"` | Rewrite the task prompt once with the reviewer model before the loop starts: `off`, `auto` (the revision is used as written), or `interactive` (you accept, keep the original, edit, or quit, which stops the run; needs a terminal to answer at, so a run under the TUI, an ACP client or a spawned lane skips it). |
| `decompose` | `"auto"` | Front-load task decomposition in run mode: the model lays the task out as ordered DAG subtasks before editing and works them one at a time. `on` helps small models that under-finish multi-part tasks (measured on mistral-small; capable models just pay 2-4x overhead), `off` never, `auto` decides per worker model from the capability registry (`config show` prints the resolved value). `--decompose` forces it for one run. |

## `[skills]`

Operator-installed SKILL.md packs (the agentskills.io format).
Installed under `$XDG_DATA_HOME/agent6/skills/<name>/`; `agent6 skills install <url>` takes a SKILL.md URL, a git repo (every `skills/*/SKILL.md`), or a local path.
Installed = enabled: an index in the system prompt, on-demand content via `use_skill`, a `/<name>` pause-menu command, and `run --skill <name>`.
The format is shared with Claude Code and most agentskills.io tooling: point `extra_dirs` at an existing collection (`~/.claude/skills`, …) or install to copy.
Repo-local skill dirs are not discovered (repo content is not config).
Trust model: [security.md](security.md).

| Field | Default | Meaning |
|---|---|---|
| `enabled` | `true` | Master switch for skills: `false` means no skill index in the prompt, no `use_skill` tool, and no slash commands. |
| `extra_dirs` | `[]` | Additional directories scanned for skills, before the installed skills dir; a skill of the same name in an earlier dir wins. |
| `state` | `{}` | Per-skill state by name: `enabled` (indexed, loaded on `use_skill`), `disabled` (dropped), or `always` (its full text sits in the system prompt). Layers merge key by key; `agent6 skills enable|disable [--repo]` writes it. |

Measured (2026-07): small and frontier open models alike almost never invoke a skill organically from the passive index, and no prompt lever made it reliable.
When a skill must apply, use `always`, `/name`, or `--skill`.

### Presets

A preset fills many settings at once.
`agent6 config presets` lists them; select with `--preset <name>`, `agent6 config set preset <name>` (`--repo`), or the preset picker of the TUI and web new-task composers.
A preset overrides config at the layer that selected it (most-specific source wins, presets never stack); a more-specific config layer, `--config FILE`, or an individual flag still beats it.
A `--config FILE` or machine overlay cannot select one.

| Preset | For | Sets |
|---|---|---|
| `standard` | the plain defaults, no review panel | nothing (the defaults) |
| `quick` | no review panel: fast and cheap | `review.trigger = "off"` |
| `ultra` | a three-seat review panel that vetoes the finish it does not pass | `review.trigger = "before_finish"`, `review.decision = "veto"`, `review.seats = ["security", "correctness", "tests"]`, `review.concurrency = 3` |
| `paranoid` | five explore-tier review seats vetoing the finish: maximum scrutiny | `review.trigger = "before_finish"`, `review.decision = "veto"`, `review.tier = "explore"`, `review.seats = ["security", "correctness", "tests", "edge-cases", "over-engineering"]`, `review.concurrency = 5` |

Define your own with a `[presets.<name>]` table (a partial config); a built-in's name replaces that built-in wholesale:

```toml
preset = "myteam"

[presets.myteam.review]
trigger = "before_finish"
decision = "veto"
seats = [
  "security@anthropic/claude-opus-4-8",
  "correctness@openrouter/moonshotai/kimi-k2",
]
```

### `[workflow.metric]` (optional)

A continuous score for measurable goals; `command` runs in the jail like `verify_command`.

| Field | Default | Meaning |
|---|---|---|
| `command` | *(required)* | The command that prints the score, as argv (no shell). Runs after every verify-passing edit, and on the model's `run_metric_command` call. |
| `pattern` | *(required)* | A regular expression over the command's output; its first capture group is the number, e.g. `"score: ([0-9.]+)"`. |
| `goal` | *(required)* | Which way is better: `minimize` (a smaller number wins) or `maximize`. The run reports the trajectory and can finish once a verified edit only ties the best. |

## `[budget]`

Hard stops; on hit the run ends (exit 3) and is resumable with a fresh budget.
Every call is bounded in exactly one currency: priceable calls (reported cost, else cached price × tokens) count against `max_usd`; a call carrying a plan-usage reading counts percentage points against `max_percent`; the rest count input+output tokens against `max_tokens_fallback`.
All three: `-1` unlimited, `0` refuse that ledger up front, `> 0` the cap.

| Field | Default | Meaning |
|---|---|---|
| `max_usd` | `10.0` | Cap on the metered spend of one run (provider-reported cost, else price times tokens at the model's fetched rates, cache-aware). Hitting it ends the run resumably (`budget_exhausted`); each resumed leg gets a fresh budget. `-1`: unlimited; `0`: refuse every metered call. `--max-usd` overrides per run. |
| `max_tokens_fallback` | `2000000` | Token cap (input plus output) for the calls the run cannot price: local models, a model with no price data. `-1`: unlimited; `0`: never run an unmeterable model. `--max-tokens-fallback` overrides per run. |
| `max_percent` | `-1.0` | Cap on the plan percentage points one run may consume on a subscription provider (the rise in the account's reported used-percent across the run, accumulated across window resets, so values above 100 are meaningful; with several windows, the one that moved most). The reading is account-global: a concurrent run's spend counts toward whichever run observes it next. `-1`: unlimited; `0`: refuse plan-metered calls. `--max-percent` overrides per run. |
| `allow_paid_credits` | `false` | Allow plan-metered calls (`chatgpt`, `claude_code`) to spend PURCHASED credits or extra usage once the included plan window is exhausted (auto top-up can buy more with the saved payment method). `false` is a circuit breaker, not a guarantee: the backend's usage readings (a chatgpt preflight and every response's headers, every claude_code round's rate-limit event) report the account's windows and credit state, and once a window is exhausted with credits present the run stops at its next boundary; a call already in flight completes. `true`: a chatgpt credit balance's drop across the run is read as dollars and meters against `max_usd`; a claude_code run reads no credit balance, so the extra usage it spends is not metered by `max_usd`. Included-plan usage is unaffected. |

`--max-usd` / `--max-tokens-fallback` override per run; a worker with no price data is not metered by `max_usd` at all and runs under `max_tokens_fallback`, with a note at startup.
Prices come from provider listings (OpenRouter's; cached under `$XDG_CACHE_HOME/agent6/models/`), and a direct-Anthropic id is priced via its OpenRouter listing.

## `[machine]`

| Field | Default | Meaning |
|---|---|---|
| `snapshot_keep` | `5` | How many blackboard snapshots a machine instance keeps (recovery reads only the latest; `machine replay` rebuilds any state from the journal). `0` keeps all. |
| `state_log_keep` | `50` | How many per-state log dirs a machine instance keeps under `<instance>/states/` (the watchable logs of each state's leg; the journal keeps the full transition history regardless). `0` keeps all. |
| `pass_env` | `[]` | Environment variable names a machine's `tool` state may receive from the operator's environment when its own `pass_env` names them; a state naming one not listed here refuses the run at startup. Global/repo config only (a machine `[config]` overlay setting it is rejected); a provider's `api_key_env` is never allowed. |

### `[machine.notify]` (optional)

Operator hook on every `machine.notify` and the terminal `machine.end`: an operator argv on the host with a minimal env (PATH/HOME/locale/desktop-bus + `AGENT6_MACHINE_ID/DIR/EVENT/STATE/MESSAGE/LEVEL`), never your full environment.
Global/repo config only (a machine `[config]` overlay setting it is rejected).
Fan out to ntfy/Pushover/email/Telegram yourself.

| Field | Default | Meaning |
|---|---|---|
| `on_event` | `[]` | A command run on every machine notify event and at the machine's end, as argv (no shell), with the event in `AGENT6_MACHINE_*` variables. Empty: no hook. |
| `timeout_s` | `30.0` | Seconds the hook may run before it is killed. |

## `[notify]` (optional)

Runs after `run`/`resume` with the same minimal env plus `AGENT6_SESSION_ID/DIR/OK/VERIFIED/REASON`.
`OK=1` means the agent stopped deliberately; `VERIFIED` is what the gate said; a hook that wants "green" reads the second.

| Field | Default | Meaning |
|---|---|---|
| `on_complete` | `[]` | A command run when a run or resume ends, as argv (no shell), with `AGENT6_SESSION_ID/DIR/OK/VERIFIED/REASON` in its environment. Empty: no hook. |
| `timeout_s` | `30.0` | Seconds the hook may run before it is killed. |

## `[web]`

Bind for `agent6 web` ([the web UI](web.md)).
Loopback only by default, no app auth: remote access is expected behind `tailscale serve`.

| Field | Default | Meaning |
|---|---|---|
| `host` | `"127.0.0.1"` | Address `agent6 web` binds; a non-loopback address also needs `allow_non_loopback = true`. |
| `port` | `7658` | Port `agent6 web` listens on. |
| `allow_non_loopback` | `false` | Allow `host` to be a non-loopback address, so a typo can never silently expose the write surface (approvals, steers, config writes) beyond this machine. |

## `[parallel]`

Fan-out defaults for `run --parallel N|model-a,model-b`.
Each lane is a disposable clone running its own `agent6/<id>` branch; lanes are imported and ranked, nothing merges for you.
`--max-usd` is per lane (the total is printed before spawning); `--auto-approve` forwards to every lane.
A live run dispatches lanes the same way via the `/parallel` steer directive (depth 1: a lane never fans out; headless surfaces without a dispatcher answer "not available").

| Field | Default | Meaning |
|---|---|---|
| `max_lanes` | `4` | The most lanes one `--parallel` fan-out may run, `1` to `1024`; a spec asking for more is refused before anything is cloned. |
| `workdir` | `""` | Base directory for the working trees lanes, machine run states, and forks work in, in a per-repo subdirectory. Empty: `<cache_dir>/parallel`. A lane's clone is removed after its work is imported; a fork's worktree by `sessions prune` once the fork is merged. |

## `[mcp]` and `[mcp.servers.<name>]` (optional)

agent6 as an MCP CLIENT; for the other direction (`agent6 mcp serve`, agent6 as a server) see [Editor integration](acp.md#as-an-mcp-server).

MCP servers, spawned (`command`) or connected (`url`); tools appear as `mcp__<name>__<tool>` in run mode only (plan, ask, machine and agent sessions never offer them: agent6 cannot classify an external tool as read-only).
A spawned server runs as a jailed child by default (its own `[mcp.servers.<name>.sandbox]` policy; `unconfined = true` opts out) with a curated env (never your provider keys; `pass_env` adds named vars); a `url` server is a process you run and confine yourself.
The model chooses the arguments, so each call is approved like a command (`approve`); audit each server like a `run_command` allow-list.
`agent6 mcp connect` handshakes first and only then writes the entry; a server that does not start is skipped with an `mcp.server_unavailable` journal event, never fatal.
`agent6 check mcp` and the connect handshake probe under the run's sandbox with the repository bound read-only and skip a server they cannot hold that way ([the rule](security.md#2-sandbox)).

```
agent6 mcp connect files -- npx -y @modelcontextprotocol/server-filesystem .
agent6 mcp connect browser --url http://127.0.0.1:8931/mcp --token-env PW_TOKEN
agent6 mcp list
agent6 mcp remove files
```

`agent6 mcp remove <name>` drops the entry from the global config (`--repo` for the per-repo one): `config unset` cannot, because the entry is a table rather than a leaf, and dropping its `command`/`url` is refused since an entry needs exactly one of them.

A spawned server's `[sandbox]` block names what it needs on top of the sandbox a `run_command` gets.

```toml
[mcp.servers.notes]
command = ["npx", "-y", "@modelcontextprotocol/server-filesystem", "~/notes"]

[mcp.servers.notes.sandbox]
read_paths  = ["~/notes"]   # additive; the system and tool dirs are already there
write_paths = ["~/notes"]
```

A confined spawn also loses the desktop-session addresses (`DBUS_SESSION_BUS_ADDRESS`, `XDG_RUNTIME_DIR`, `DISPLAY`, `WAYLAND_DISPLAY`): Landlock does not gate unix-socket `connect()`, and an unconfined session daemon would act on the server's behalf.
Anything else that reaches an unconfined process is still a way out, so name the narrowest paths that work.

| Field | Default | Meaning |
|---|---|---|
| `enabled` | `false` | Master switch for MCP servers: `false` means no `mcp__*` tools reach the model, whatever `[mcp.servers]` lists. |
| `servers.<name>.command` | `[]` | argv of a stdio MCP server agent6 spawns (jailed like a command, plus `sandbox`). Exactly one of `command` or `url`. |
| `servers.<name>.url` | `""` | An http(s) MCP endpoint you run yourself; agent6 only connects, owning none of its environment or confinement. Exactly one of `command` or `url`. |
| `servers.<name>.token_env` | `""` | For a `url` server: the environment variable holding its bearer token, named here and never inlined or logged. Over plaintext `http://` to a non-loopback host the token is readable on the wire: `mcp connect` asks first, and every run warns. |
| `servers.<name>.enabled` | `true` | `false` withholds this server's tools from the model without deleting the entry. |
| `servers.<name>.pass_env` | `[]` | Environment variables a spawned server needs, by name; everything else is agent6's curated base environment. |
| `servers.<name>.approve` | `"ask"` | `ask` prompts before each of this server's tool calls, showing the arguments the model chose; `yes` never asks. The session answers are per server: "allow all" covers this server for the run (not the command tools, not a sibling server), "deny all" withdraws its tools from the next turn. `--auto-approve` sets `yes` for the run. There is no `no`: `enabled = false` is how a server's tools are withheld. |
| `servers.<name>.startup_timeout_s` | `10.0` | Seconds the server gets to answer `initialize` and `tools/list` before it is given up on. |
| `servers.<name>.call_timeout_s` | `60.0` | Seconds one `tools/call` may take before it fails; a spawned server is restarted after a timeout. |
| `servers.<name>.httpx_trust_env` | `false` | For a `url` server: honor the ambient `HTTP(S)_PROXY`, `.netrc`, and `SSL_CERT_FILE` (httpx's `trust_env`). `false` so a local server's bearer token never routes to a proxy; set it for a server reachable only through the environment's proxy. |

### `[mcp.servers.<name>.sandbox]`

| Field | Default | Meaning |
|---|---|---|
| `read_paths` | `[]` | Read+execute paths for this server beyond the sandbox a jailed command gets (absolute or `~`). The workspace, system dirs, tool dirs and a writable `HOME` are already there, so a block names only the server's own data. |
| `write_paths` | `[]` | Paths it may write, likewise additive. |
| `network` | `"auto"` | Which network this server joins. `auto`: one of its own where the host can give a namespace, degrading to the host's with a warning. `none`: the same, refusing rather than running connected. `session`: the run's network, so a dev server a background command started answers this server too (a browser server driving the app under test), and still nothing off the box. `host`: the machine's network. |
| `unconfined` | `false` | No sandbox at all, for a server whose job is arbitrary host access. Contradicts every other field here, so setting both is refused rather than half-applied. |

---

## Reaching a run's network

A run's commands share one network with no route off the box, so a dev server the agent starts is reachable only through these operator-only commands:

```
agent6 sessions show <id>        # what it is serving, and the command to open it (the TUI dashboard and web run headers say the same)
agent6 forward <id>              # list the ports it is listening on
agent6 forward <id> 3000         # bridge that port to one on this machine
agent6 exec <id> -- curl ...     # run a command in the run's sandbox
```

`exec` runs in the whole sandbox (the run's recorded isolation and network, with mounts from your current config), so what you see is what the agent sees.
None of them is reachable by the model.

---

## Environment variables

| Variable | Effect |
|---|---|
| `XDG_CONFIG_HOME`, `XDG_STATE_HOME`, `XDG_CACHE_HOME`, `XDG_DATA_HOME` | The XDG base directories; agent6 lives in an `agent6/` dir under each (config and secrets; per-repo state; the cache; installed skills). |
| `AGENT6_DETACHED_AWAY` | `wait`, `deny` or `approve`: what a run with no operator at a terminal does at an approval or question. The hub and machine spawns set `wait`; a `--parallel` fan-out's lanes take the coordinator's own marker, else `wait` with a terminal to attach from, else `deny`. |
| `AGENT6_AUTO_APPROVE` | `1` grants every command approval to a machine's agent states, as `--auto-approve` does (a configured `no` stays no). |
| `AGENT6_NO_COMMANDS` | `1` withholds every command tool from a machine's agent states, as `--no-commands` does. |
| `AGENT6_JAIL_BIN` | Path to a specific `agent6-jail` binary (else bundled). |
| `AGENT6_DANGEROUSLY_DISABLE_SANDBOX` | `1` forces `sandbox.isolation = "none"` for one invocation (same as `--dangerously-disable-sandbox`). |
| `AGENT6_ALLOW_ROOT` | `1` permits running as root (same as `--allow-root`). |

A provider's `api_key_env` names the env var supplying its key; omit it to read `secrets.toml`.
The bench and development switches are listed in [architecture.md](architecture.md#bench-and-development-switches).
