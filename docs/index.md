---
title: agent6
hide:
  - toc
---

<div class="a6-hero" markdown>

# agent6

<p class="a6-tagline">A coding agent that jails model commands and uses editable state machines for long-running tasks.</p>

<div class="a6-cta" markdown>
[:material-github: GitHub](https://github.com/agent6-dev/agent6){ .md-button }
[:simple-pypi: PyPI](https://pypi.org/project/agent6/){ .md-button }
</div>

</div>

<div class="a6-shot" markdown>
![The run dashboard: task graph, budget, tool calls, reasoning, log, and diff](screenshots/out/02-run-dashboard.png)
</div>

The model can write code and ask to run commands, but those commands go through a jail with restricted filesystem and network access.
Long-running workflows can be written, reviewed, edited, resumed, and replayed as declarative state machines instead of being left to an open-ended agent loop.

<div class="a6-grid" markdown>

<div class="a6-card" markdown>
### Command sandbox
Commands the model runs go through a jail to give you control over what the model can read and write, and to restrict network access.
</div>

<div class="a6-card" markdown>
### Verify gate
A run is measured against a verify command, inferred from the repo when unset and pinned for the run.
Every editing step commits; the gate certifies the tree the run ends on.
</div>

<div class="a6-card" markdown>
### Detached commit chain
Per-step commits land on a detached ref, leaving the branch, HEAD, and index untouched.
`agent6 sessions merge` lands the work.
</div>

<div class="a6-card" markdown>
### Resume and fork
State is snapshotted before every model call and checkpointed each turn.
A run resumes from its snapshot, or forks into a new run at any past turn.
</div>

<div class="a6-card" markdown>
### Agent state machines
Longer tasks run as declarative machines: model-drafted, operator-reviewed, journaled, replayable.
</div>

<div class="a6-card" markdown>
### Parallel fan-out
A task can run in isolated lanes on different models, ranked by a reviewer model, or by verify and cost.
Merging stays manual.
</div>

</div>

## The terminal UI

<video controls muted loop playsinline preload="metadata" class="no-lightbox"
       poster="/screenshots/out/02-run-dashboard.png">
  <source src="/screenshots/out/tour.webm" type="video/webm">
</video>

`agent6 run` streams the run's conversation in your terminal, with no full-screen UI.
`agent6 tui` opens the hub instead: every run for the repository with its mode, status, and cost, where you open a session to read its live conversation, toggle the dashboard (Ctrl+D), or scroll the event log.
`agent6 run --tui` starts on that conversation view, and `-i` drives the run from a stdin REPL.
The [terminal UI](terminal.md) page has a still of each screen.

## The web UI

<video controls muted loop playsinline preload="metadata" class="no-lightbox">
  <source src="/screenshots/out/web-desktop.webm" type="video/webm">
</video>

`agent6 web` serves the same views in a browser, from a desktop or a phone: start a run and watch it stream, steer it, approve prompts, answer questions, read the transcript, and browse and run state machines.
It binds `127.0.0.1`; put `tailscale serve` in front for encrypted remote access.
See [the web UI](web.md).

## Usage

```sh
uv tool install agent6                 # or: pipx install agent6
agent6 connect                         # pick a provider, paste an API key (once)
agent6 model worker anthropic claude-sonnet-5

cd your-repo
agent6 run "add a --json output mode to the CLI"
```

[Installation](installation.md) covers requirements, shell completion, and building from source.
[Usage](usage.md) covers the first run, inspecting it, and recovering one that went wrong.
