# agent6

A coding agent that jails model commands and uses editable state machines for long-running tasks.

The model can write code and ask to run commands, but those commands go through a jail with restricted filesystem and network access.
Long-running workflows can be written, reviewed, edited, resumed, and replayed as declarative state machines instead of being left to an open-ended agent loop.

**Full documentation: [agent6.dev](https://agent6.dev)**

<table>
  <tr>
    <td align="center" width="34%" valign="top">
      <a href="https://agent6.dev/screenshots/out/hero-tui.gif"><img src="https://agent6.dev/screenshots/out/hero-tui.gif" alt="the run TUI: conversation streaming, an approval modal, verify + auto-commit, the hub receipt"></a>
      <br><sub><b>the TUI</b><br>the full agent, as a live dashboard</sub>
    </td>
    <td align="center" width="33%" valign="top">
      <a href="https://agent6.dev/screenshots/out/hero-cli.gif"><img src="https://agent6.dev/screenshots/out/hero-cli.gif" alt="the CLI: a failing suite, one command, the run streams to a green verify and a diff"></a>
      <br><sub><b>the CLI</b><br>the full agent, in any terminal</sub>
    </td>
    <td align="center" width="33%" valign="top">
      <a href="https://agent6.dev/screenshots/out/hero-web.gif"><img src="https://agent6.dev/screenshots/out/hero-web.gif" alt="the web UI: the hub, a session view with expanded tool detail, the sandbox config"></a>
      <br><sub><b>the web UI</b><br>the full agent, desktop or phone</sub>
    </td>
  </tr>
</table>

## Features

- **Jailed commands**: Landlock + seccomp; `strict` (the `auto` pick where the host allows) adds user namespaces, `pivot_root`, read-only `.git`, no route off the box ([Security](https://agent6.dev/security/))
- **Providers**: Anthropic, any OpenAI-compatible endpoint (OpenAI, OpenRouter, Ollama, vLLM, llama.cpp, LM Studio), or a ChatGPT subscription ([Config](https://agent6.dev/config/))
  - model + reasoning effort per role
- **Clean checkout**: per-step commits on a detached ref; `sessions merge` lands them
  - snapshot resume; fork at any turn
- **Verify gate**: inferred when unset, pinned for the run, green/red on every surface
  - a worker proposes a replacement gate instead of reverting
- **Budget**: hard `max_usd` cap; token cap for unpriced calls
- **Sessions**: run, plan, ask (plan and ask never edit)
  - `--from <id>` seeds from another; `/btw` asks beside a live run
- **Four front-ends, one engine**: CLI, TUI, [browser](https://agent6.dev/web/) (stdlib server, no JS deps, phone), [editor over ACP](https://agent6.dev/acp/)
  - `attach`, `exec`, `forward`, `history`, `ps`
- **Background commands**: `background: true` hands back a handle; none outlive the run
- **Context control**: compaction visible on every surface; `/compact [focus]`, `/pin`; repo memory injected per run
- **State machines**: LLM-drafted, operator-reviewed, journaled, replayable ([State machines](https://agent6.dev/state-machines/))
  - pause for input, take events, steer from any front-end
- **Task graph**: the worker's plan as a persistent DAG, journaled with the run, live on every surface
  - survives crash and compaction restarts; `decompose = "auto"` front-loads it for models measured to need it
- **Code review**: `agent6 review` on any diff; an in-loop adversarial panel where only blocking findings gate
- **Parallel fan-out**: `--parallel N|model-a,model-b` clone-based lanes, auto-compared into a ranked report ([Architecture](https://agent6.dev/architecture/#parallel-runs))
- **Skills**: SKILL.md packs (the format most agents share) index into the prompt, fire as `/name` or `--skill`; repo instructions from `AGENTS.md`
- **Fixed tool surface**: extended only by operator-configured MCP servers, off by default, jailed by default
- **Eight runtime dependencies**, no telemetry, no auto-update

## Install

From [PyPI](https://pypi.org/project/agent6/) with [uv](https://docs.astral.sh/uv/getting-started/installation/) or [pipx](https://pipx.pypa.io/stable/how-to/install-pipx/):

```bash
uv tool install agent6        # or: pipx install agent6
```

If `agent6` is not found, you can add the uv or pipx bin dir (`~/.local/bin`) to your PATH with `uv tool update-shell` or `pipx ensurepath`.

Enable shell completion with `agent6 completions` (supports bash, zsh, fish, and xonsh).

agent6 requires **Python 3.12+** and the sandbox only supports **Linux** (x86_64/aarch64).
Other platforms run without the sandbox behind a warning.
See [installation](https://agent6.dev/installation/) for the full requirements and building from source.

## Usage

```bash
# Connect a provider (stored in ~/.config/agent6/, key in a 0600 secrets file).
# If already connected, skip both; `agent6 check` verifies it.
agent6 connect                # interactive: pick provider, paste API key
# (or `agent6 connect chatgpt` to sign in with a ChatGPT subscription)
agent6 model worker anthropic claude-sonnet-5

# Run the agent on a task, create a plan, or ask a question.
cd your-repo
agent6 run "add a --json output mode to the CLI"
agent6 plan "how to add a --json output mode to the CLI"
agent6 ask "how to add a --json output mode to the CLI"

# Watch and drive runs from a terminal, a TUI, a browser, or an editor.
agent6 attach <session-id>    # follow + answer a run live (--raw for events)
agent6 tui                    # full-screen dashboard hub
agent6 web                    # browser UI on http://127.0.0.1:7658
agent6 acp                    # speak ACP on stdio; an editor spawns this

# Audit the effective config, check the sandbox, resume or fork a run.
agent6 config show
agent6 check
agent6 resume <session-id>
agent6 fork <session-id> --at-turn 7

# See all commands with `agent6 --help` or `agent6 <command> --help`.
```

See [usage](https://agent6.dev/usage/) for the full command tour, [the web UI](https://agent6.dev/web/) for driving runs from a phone, [configuration](https://agent6.dev/config/) for every field, and the [security model](https://agent6.dev/security/) for what the sandbox enforces.

Config is layered, lowest precedence first: built-in defaults, the global `~/.config/agent6/config.toml`, the per-repo config (state dir, never committed), then `--config FILE`.

- `agent6 config show`: every effective value with the layer that set it
- every field has a default; security-sensitive fields default safe: `isolation = "auto"`, `network = "auto"`, `run_commands = "ask"`, `protect_git = true`
- `"auto"` picks the most secure option the host allows, warning when it falls short; an explicit value the host cannot enforce refuses to run
- `protect_git = true` (read-only `.git`) needs `strict`: on `hardened` the default warns, an explicit `true` refuses
- agent6 never pushes, rewrites history, or `reset --hard`; no config key can enable them
