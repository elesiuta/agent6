# Editor integration

Two ways in, both spawned by the editor over stdio: ACP drives whole runs, and `agent6 mcp serve` exposes a few tools to another agent (see [As an MCP server](#as-an-mcp-server)).

## Agent Client Protocol

`agent6 acp` runs agent6 as an [Agent Client Protocol](https://agentclientprotocol.com/) agent: an editor spawns it, sends prompts, and renders the run as it happens.
It uses the same engine, config, and jail as `agent6 run`.

```jsonc
// Zed: settings.json
{
  "agent_servers": {
    "agent6": { "command": "agent6", "args": ["acp"] }
  }
}
```

Any ACP client works the same way, and the command above is the whole configuration.

## What the editor sees

Every run writes one event journal, and the CLI, TUI, and web UI render it through the same fold.
ACP is a fourth projection of that fold, so an editor sees what `agent6 attach` shows: reasoning, each tool call and its outcome, auto-commits, and how the run ended.

A tool call arrives twice, as ACP models it.

- `tool_call` (`in_progress`) when the run dispatches it, `tool_call_update` (`completed` or `failed`, with the output) when its result lands
- a call waiting on an approval or an `ask_user` answer is updated to `pending` while its prompt is open, and back to `in_progress` once answered
- a long verify shows as in progress while it runs; a call the run never returned from settles as `failed` when the run's `session.end` is written (a worker killed without one leaves it in progress)
- `toolCallId` is `<run id>:<turn>:<call>`, unique for the life of the session: each turn is one leg of the run, and a leg's call numbers start at 1

Everything the lifecycle prints (the `agent6 run` footer: where the changes are, the auto-stash notice and how to restore it, a refusal's reason) arrives as an `[agent6]` agent message as it is printed, whatever state the journal is in.
The cost receipt goes to stderr only, where a client that shows the agent's log picks it up.

## Approvals

`session/request_permission` carries every approval the CLI would prompt for: `run_commands = "ask"`, an MCP tool call the server's `approve` does not cover, a `fetch` to a host outside the allow-list, an unsandboxed autorun.
The editor renders the buttons.
The request names the tool call it gates and carries the prompt as that call's title, which is the text the editor renders; it is sent once the run's journal tail has announced that call (a tail that stopped reading, or a cancelled turn, releases the request); a prompt that gates no call (a pre-run question) announces a tool call of its own and closes it with the answer.
The prompt and its answer are journaled as `approval.prompt` / `approval.answer` (`question.*` for an `ask_user`) by the same gate every front-end answers through, the answer with `source: "acp"` (`"headless"` when the client declared it cannot be asked), so `agent6 attach` and the web show the run as awaiting the answer.

Three rules hold whoever is driving:

- An unanswered request denies: after five minutes with no reply the approval is refused and the run continues without it.
- An off-list `fetch` host is offered as `allow_once` only, so an editor's "always allow" cannot cover a different host later.
- A standing "allow all" recorded on the run by an earlier front-end (a CLI leg's `a`) answers that scope's later prompts without asking the editor; the answer journals `source: "session"`.

## Sessions

One session is one directory, one conversation.

- `session/new` carries an absolute `cwd`; config is that directory's own layered config (global, repo, preset)
- the directory must be a git repository (the jail's writable mount; runs branch and commit each step)
- the first prompt starts an `agent6 run`; every later prompt resumes it with the text as its steering instruction (`resume --steer` semantics)
- a prompt whose prior turn died before the first snapshot starts fresh
- a busy session refuses a prompt rather than queueing it; the editor can offer it again
- one connection runs one prompt at a time across sessions (the commit cwd is process-global): a prompt on another session waits its turn, tells the editor which session it waits for, and a `session/cancel` while it waits answers `cancelled` at once
- `session/cancel` drops the `agent6 sessions stop` marker: the step in flight finishes and commits first

## Not implemented

- `session/load`: ACP v2 reorganises it, and resume carries agent6's own semantics (`agent6 resume`, `agent6 fork`), so `initialize` reports the capability as absent.
- Mid-run steering: ACP has no message for a prompt while a turn is running, so a session's follow-up is the next prompt, which resumes the run with that text as its first steering instruction.
- `fs/*` and `terminal/*`: ACP lets the client own the filesystem and the terminal, and agent6 keeps both behind the jail the operator configured.
- Embedded resources in a prompt: text and `resource_link` blocks are read (a link rides in as its uri; the workspace boundary still decides what it reaches)
    - images and embedded resources are dropped; `promptCapabilities.embeddedContext` says so

## Troubleshooting

- stdout is the protocol stream: nothing but JSON-RPC
- everything agent6 would print goes to stderr (the editor's agent logs)
- a wrapper echoing to stdout before exec'ing `agent6` breaks the connection irrecoverably; write to stderr

## As an MCP server

`agent6 mcp serve` speaks MCP over stdio, so another agent (an editor's own, or a second agent6 with `[mcp.servers]`) can use agent6's jail and run state.
It is the inverse of `[mcp]` in the config, which is agent6 as an MCP CLIENT.
The cwd's config decides everything: the sandbox the commands run in, and which tools exist at all.

Five tools, of which a default config publishes two:

| Tool | Withheld when |
| --- | --- |
| `query_dag` (a run's task graph) | never |
| `list_sessions` (this repository's sessions) | never |
| `run_verify` (the configured gate, jailed) | no `[workflow] verify_command`, or `[sandbox] run_commands` is not `yes` |
| `run_in_sandbox` (an argv, jailed) | `[sandbox] run_commands` is not `yes` |
| `apply_patch_in_sandbox` (a patch, then the gate) | either of the two above |

`run_commands = "ask"` withholds the command tools rather than offering ones that would refuse: the MCP boundary has no operator to prompt.
A client that calls a withheld tool by name is told which setting withheld it.

```jsonc
// Zed: settings.json -- agent6's jail as another agent's tool surface
{
  "context_servers": {
    "agent6": { "command": { "path": "agent6", "args": ["mcp", "serve"] } }
  }
}
```

The tools reach the repository the server was started in, under that repository's sandbox policy; `list_sessions` and `query_dag` read its state dir and nothing else.
