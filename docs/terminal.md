# Terminal UI

The full-screen TUI screen by screen, then the plain CLI.
Every image is from a recorded run; click to enlarge.

<video controls muted loop playsinline preload="metadata" class="no-lightbox"
       poster="/screenshots/out/01-hub.png">
  <source src="/screenshots/out/tour.webm" type="video/webm">
</video>

## Hub

`agent6 tui` lists every run for the repository with its mode, status, last activity, and cost.

- Enter opens a run
- `n`: an empty conversation to start a run, plan, or ask (mode and preset picked above the composer; a refusal shows there with the text kept)
- `c`: the config page; `?`: the keys

![The hub](screenshots/out/01-hub.png)

## Conversation

Opening a run lands on its conversation (also `agent6 sessions transcript`): the task, the model's reasoning, and every tool call with its complete input and output, following live.
A live run keeps a steer bar at the bottom.
An approval shows inline at the conversation's tail (the command, fixed-width) with a key row docked above the bar: `a` allow, `s` allow all this session, `d` deny, `x` deny all; answered, it collapses to one dim line.

![A run transcript](screenshots/out/05-transcript.png)

## Run dashboard

`Ctrl+D` toggles the dashboard: task graph beside live reasoning, tool calls with results, event log and latest commit diff side by side.

- the diff pane opens on the latest commit; its selector walks the run's per-step commits (newest first), `cumulative` shows the chain up to that step; a run whose model owns git has no chain and the pane says so
- the composer bar runs along the foot: type to steer, or to resume a finished run
- `/` completes the steer directives; Ctrl-R searches the session's past messages
- the View menu maximizes the focused pane

![The run dashboard](screenshots/out/02-run-dashboard.png)

## Event log

The View menu's Full log opens the JSONL event stream the dashboard is built from, scrollable over the whole run.

![The event log](screenshots/out/09-logs.png)

## Configuration

The config page shows every setting, its effective value, and where that value came from (a built-in default, the global config, or the per-repo config).
`/` filters by name.

![The config page](screenshots/out/03-config.png)

![Filtering the config by name](screenshots/out/04-config-search.png)

## Keys

![The keys and actions overlay](screenshots/out/08-help.png)

## Without the TUI

`agent6 run` executes in the foreground: steer it with Ctrl-C, no TUI required.

- the pause menu Tab-completes its commands; Up recalls, Ctrl-R searches past messages
- Ctrl-Z in a view, or `/detach` in the menu, hands the run to the background after its current step (`agent6 attach` reattaches)
- a viewer opened with `agent6 attach --tui` just closes
- TUI/web-hub runs start detached; `agent6 attach` covers both kinds: conversation by default, `--raw` line tail, `--tui` full screen, `--json` one-shot snapshot

<video controls muted loop playsinline preload="metadata" class="no-lightbox">
  <source src="/screenshots/out/temps-demo.webm" type="video/webm">
</video>

## Watching a state machine

An [agent state machine](state-machines.md) runs in the terminal like anything else: author the file, read its graph, watch it execute.
Here `code-fixer` runs a fix-loop: an agent state edits the repo to make a failing check pass, a tool state re-runs the check, and the machine routes on the result until it is green or the attempt budget is spent, with the agent's reasoning streamed live like a run.

<video controls muted loop playsinline preload="metadata" class="no-lightbox">
  <source src="/screenshots/out/machine-demo.webm" type="video/webm">
</video>

---

These are regenerated from recorded runs by the [pages workflow](https://github.com/agent6-dev/agent6/blob/master/.github/workflows/pages.yml), so they track the current UI.
