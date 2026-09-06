# Terminal UI

The full-screen TUI screen by screen, then the plain CLI.
Every image is from a recorded run; click to enlarge.

<video controls muted loop playsinline preload="metadata" class="no-lightbox"
       poster="/screenshots/out/01-hub.png">
  <source src="/screenshots/out/tour.webm" type="video/webm">
</video>

## Hub

`agent6 tui` lists every session for the repository in five columns: updated, status (the mode folded in), cost, id, and task.

- Enter opens a run
- `n`: an empty conversation to start a run, plan, or ask (mode and preset picked above the composer; a refusal shows there with the text kept)
- `l`: the selected run's event log; `m`: merge its branch (`sessions merge`); `d`: delete its history (`sessions rm`), both after a confirm
- Space: expand or fold a fan-out's lanes (one row with its lane count otherwise)
- `c`: the config page; `M`: the machines screen; `r`: refresh; `?`: the keys; `q`: quit

![The hub](screenshots/out/01-hub.png)

## Conversation

Opening a run lands on its conversation (also `agent6 sessions transcript`): the task, the model's reasoning, and every tool call with its input and output (clipped to the salient lines by default; Detail cycles none, collapsed, expanded), following live.
A live run keeps a steer bar at the bottom; above it, a live pane streams the turn in progress and lists the tool calls in flight (`→ run_command  sleep 60  · running`) until their results land in the transcript.
An approval shows inline at the conversation's tail (the command, fixed-width) with a key row docked above the bar: `a` allow, `s` allow all this session, `d` deny, `x` deny all; answered, it collapses to one dim line.
The composer keeps focus and owns the keys: they answer only while it is empty and focused (a typed message never answers); a click on a label answers from anywhere.
Away from the conversation (the dashboard, a machine screen) an approval opens a modal instead: `y` allow, `a` allow session, `n` deny, `x` deny all.

- `Ctrl+T` cycles the thinking and tool detail: hidden, collapsed, expanded
- `Ctrl+C` copies the selection, or the whole transcript when nothing is selected
- `Ctrl+Z` leaves the view: a run `agent6 run --tui` fronts detaches to the background after its current step (`agent6 attach` reattaches); one opened with `attach --tui` or from the hub keeps running as it was
- `Ctrl+_` undoes typing in the composer (`Ctrl+Z` is the detach key everywhere in the view)

![A run transcript](screenshots/out/05-transcript.png)

## Run dashboard

`Ctrl+D` toggles the dashboard: task graph beside live reasoning, tool calls with results, event log and latest commit diff side by side.

- the diff pane opens on the latest commit; its selector walks the run's per-step commits (newest first), `cumulative` shows the chain up to that step; the task tree and the cost line follow the selected step; a run whose model owns git has no chain and the pane says so
- the composer bar runs along the foot: type to steer, or to resume a finished run
- `/` completes the steer directives; Ctrl-R searches the session's past messages
- `/shells` lists the run's background commands and how they ended; `/restate` replays the conversation since your last message
- the View menu maximizes the focused pane

![The run dashboard](screenshots/out/02-run-dashboard.png)

## Event log

The View menu's Full log opens the structural events of the run's log, formatted as the dashboard's log pane formats them, scrollable over the whole run.

![The event log](screenshots/out/09-logs.png)

## Configuration

The config page shows every setting, its effective value, and the layer that set it: `default`, `preset`, `global`, `repo`, or `flag` (a `--config FILE`).
`/` filters by name.

![The config page](screenshots/out/03-config.png)

![Filtering the config by name](screenshots/out/04-config-search.png)

## Keys

![The keys and actions overlay](screenshots/out/08-help.png)

## Without the TUI

`agent6 run` executes in the foreground: steer it with Ctrl-C, no TUI required.

- the pause menu Tab-completes its commands; Up recalls, Ctrl-R searches past messages
    - `/status`, `/tasks`, `/pin`, `/compact`, `/parallel`, `/btw`, `/shells`, `/restate`, `/undo`, `/continue`, `/stop`, `/exit`, `/detach`, `/help`
- a steer sent from another surface (`agent6 steer`, the web or TUI composer) while the menu is open is taken as the answer
- `/detach` in the menu hands the run to the background after its current step (`agent6 attach` reattaches); Ctrl-Z prints the run's state and stands an armed pause down, it never suspends the run (a suspended agent would lose its live provider stream)
- `/exit` in the menu or the `run -i` REPL (bare `exit` at the fallback prompt) stops the run and leaves without the follow-up prompt (`agent6 resume` continues it)
- `run -i` prompts after every commit: `/continue` (bare Enter), `/cost`, `/diff`, `/watch`, `/mcp`, `/init`, `/undo`, `/help`, `/quit`, `/exit`
- a viewer opened with `agent6 attach --tui` just closes
- TUI/web-hub runs start detached; `agent6 attach` covers both kinds: conversation by default, `--raw` line tail, `--tui` full screen, `--json` one-shot snapshot

<video controls muted loop playsinline preload="metadata" class="no-lightbox">
  <source src="/screenshots/out/temps-demo.webm" type="video/webm">
</video>

## Watching a state machine

An [agent state machine](state-machines.md) runs in the terminal like anything else: author the file, read its graph, watch it execute.
Here `code-fixer` runs a fix-loop: an agent state edits the repo to make a failing check pass, a tool state re-runs the check, and the machine routes on the result until it is green or the attempt budget is spent, with the agent's reasoning streamed live like a run.

- the machines screen (`M` on the hub): `v` (or Enter) opens the parsed file, `r` runs it, `w` watches its instance, `c` creates a draft, `f` refreshes
- the watch screen (also `agent6 attach --tui <id>`): `s` steers the current agent state, `m` messages a waiting instance (`machine poke`), `x` stops it at the next transition

<video controls muted loop playsinline preload="metadata" class="no-lightbox">
  <source src="/screenshots/out/machine-demo.webm" type="video/webm">
</video>

---

These are regenerated from recorded runs by the [pages workflow](https://github.com/agent6-dev/agent6/blob/master/.github/workflows/pages.yml), so they track the current UI.
