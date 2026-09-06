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

<!-- config-table: agent6 -->

## `[providers.<name>]`

One backend per block; `<name>` is referenced from `[models.<role>]`.
Three orthogonal choices describe any backend: **`api_format`** (the wire dialect, the only field that selects code), **`deployment`** (URL/placement quirks of where it is hosted), and **auth** (`auth_style` + `api_key_env` or `token_command`).
A minimal block is just `api_format` (plus `base_url` for a non-default host).

<!-- config-table: providers.<name> -->

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
- Spend is plan-metered, not dollar-metered: every response carries the account's rate-limit window, surfaces show `plan usage (<entry>): N% of the 7-day window` (the provider entry named), and `[budget].max_percent` caps the points one run may consume (`--max-percent` per run). Dollar figures stay an authoritative $0.

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
- Spend is plan-metered, not dollar-metered: every round reports the account's 5-hour and 7-day windows, surfaces show the fuller one as `plan usage (<entry>): N% of the 7-day window (seven_day)`, and `[budget].max_percent` caps the points one run may consume (`--max-percent` per run).
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
<!-- config-table: models.worker -->

## `[sandbox]`

The field summary; the model is in security.md: [Sandbox](security.md#2-sandbox) and [Network](security.md#5-network).

`extra_read_paths`, `extra_write_paths` and `hide_paths` take absolute paths with no `..` segment, and `extra_device_paths` only paths under `/dev`; anything else refuses at config load.

<!-- config-table: sandbox -->

## `[git]`

<!-- config-table: git -->

### `[git.commit]`

<!-- config-table: git.commit -->

### `[git.commit.checkpoint]` and `[git.commit.squash]`

<!-- config-table: git.commit.checkpoint git.commit.squash -->

## `preset` (top-level)

<!-- config-table: preset -->

## `[workflow]`

<!-- config-table: workflow -->

## `[review]`

<!-- config-table: review -->

A `block` gates only when its `file:line` is in the diff and its category is one of `security`, `sandbox-bypass`, `off-topic-edit`, `data-loss`, `verify-uncovered-correctness`; every other finding is advisory and cannot stall the run.

## `[context]`

Tiered context compaction (approximate chars; tokens ≈ chars/4).

<!-- config-table: context -->

## `[prompt]`

<!-- config-table: prompt -->

## `[skills]`

Operator-installed SKILL.md packs (the agentskills.io format).
Installed under `$XDG_DATA_HOME/agent6/skills/<name>/`; `agent6 skills install <url>` takes a SKILL.md URL, a git repo (every `skills/*/SKILL.md`), or a local path.
Installed = enabled: an index in the system prompt, on-demand content via `use_skill`, a `/<name>` pause-menu command, and `run --skill <name>`.
The format is shared with Claude Code and most agentskills.io tooling: point `extra_dirs` at an existing collection (`~/.claude/skills`, …) or install to copy.
Repo-local skill dirs are not discovered (repo content is not config).
Trust model: [security.md](security.md).

<!-- config-table: skills -->

Measured (2026-07): small and frontier open models alike almost never invoke a skill organically from the passive index, and no prompt lever made it reliable.
When a skill must apply, use `always`, `/name`, or `--skill`.

### Presets

A preset fills many settings at once.
`agent6 config presets` lists them; select with `--preset <name>`, `agent6 config set preset <name>` (`--repo`), or the preset picker of the TUI and web new-task composers.
A preset overrides config at the layer that selected it (most-specific source wins, presets never stack); a more-specific config layer, `--config FILE`, or an individual flag still beats it.
A `--config FILE` or machine overlay cannot select one.

<!-- presets-table -->

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

<!-- config-table: workflow.metric -->

## `[budget]`

Hard stops; on hit the run ends (exit 3) and is resumable with a fresh budget.
Every call is bounded in exactly one currency: priceable calls (reported cost, else cached price × tokens) count against `max_usd`; a call carrying a plan-usage reading counts percentage points against `max_percent`; the rest count input+output tokens against `max_tokens_fallback`.
All three: `-1` unlimited, `0` refuse that ledger up front, `> 0` the cap.

<!-- config-table: budget -->

`--max-usd` / `--max-tokens-fallback` override per run; a worker with no price data is not metered by `max_usd` at all and runs under `max_tokens_fallback`, with a note at startup.
Prices come from provider listings (OpenRouter's; cached under `$XDG_CACHE_HOME/agent6/models/`), and a direct-Anthropic id is priced via its OpenRouter listing.

## `[machine]`

<!-- config-table: machine -->

### `[machine.notify]` (optional)

Operator hook on every `machine.notify` and the terminal `machine.end`: an operator argv on the host with a minimal env (PATH/HOME/locale/desktop-bus + `AGENT6_MACHINE_ID/DIR/EVENT/STATE/MESSAGE/LEVEL`), never your full environment.
Global/repo config only (a machine `[config]` overlay setting it is rejected).
Fan out to ntfy/Pushover/email/Telegram yourself.

<!-- config-table: machine.notify -->

## `[notify]` (optional)

Runs after `run`/`resume` with the same minimal env plus `AGENT6_SESSION_ID/DIR/OK/VERIFIED/REASON`.
`OK=1` means the agent stopped deliberately; `VERIFIED` is what the gate said; a hook that wants "green" reads the second.

<!-- config-table: notify -->

## `[web]`

Bind for `agent6 web` ([the web UI](web.md)).
Loopback only by default, no app auth: remote access is expected behind `tailscale serve`.

<!-- config-table: web -->

## `[parallel]`

Fan-out defaults for `run --parallel N|model-a,model-b`.
Each lane is a disposable clone running its own `agent6/<id>` branch; lanes are imported and ranked, nothing merges for you.
`--max-usd` is per lane (the total is printed before spawning); `--auto-approve` forwards to every lane.
A live run dispatches lanes the same way via the `/parallel` steer directive (depth 1: a lane never fans out; headless surfaces without a dispatcher answer "not available").

<!-- config-table: parallel -->

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

<!-- config-table: mcp mcp.servers.<name> -->

### `[mcp.servers.<name>.sandbox]`

<!-- config-table: mcp.servers.<name>.sandbox -->

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
