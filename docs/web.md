# Web UI

`agent6 web` serves a browser front-end for driving agent6 from a desktop or a phone: watch a run stream, steer it, approve prompts, answer questions, read the conversation, and browse, create, run, and watch state machines.

<video controls muted loop playsinline preload="metadata" class="no-lightbox">
  <source src="/screenshots/out/web-desktop.webm" type="video/webm">
</video>

The same UI on a phone (single column, bottom nav):

<video controls muted loop playsinline preload="metadata" class="no-lightbox"
       style="max-width: 390px">
  <source src="/screenshots/out/web-phone.webm" type="video/webm">
</video>

## Start the server

```bash
agent6 web              # serve the hub on http://127.0.0.1:7658
agent6 web <session-id> # open a session on load
agent6 web <machine>    # open a machine instance on load
```

`--host` / `--port` override the [`[web]`](config.md#web) config for one invocation.
Stop it with Ctrl-C.

## Pages

Every page docks its text entry at the bottom, like a chat: type, Enter sends, Shift+Enter inserts a newline.

- **Sessions page**: every session (mode, status, last activity, cost)
    - the docked composer starts new work: run / plan / ask, under a chosen preset
    - prune merged run branches, squash-merged ones too when the box beside it is ticked (the CLI's `--delete-squashed`); clear saved asks
- **Machines page**: instances, `machine create` drafts, cards that run an authored machine file
    - the docked composer creates a new one
- **Session view** (live over SSE): the conversation is the page, the same folded transcript the CLI and TUI render, with the in-progress turn streaming underneath
    - a detail toggle cycles collapsed / expanded / hidden; any clipped item expands on click
    - the run's context (status, task graph, budget, tool calls, background shells, latest commit diff, event log) lives in a resizable details drawer
    - the docked composer steers a live run or resumes an ended one; `/` completes the steer directives, Ctrl-R (composer focused) searches the session's past messages
    - the Latest commit widget selects any per-step commit (cumulative toggle); the Budget and Task graph widgets then show that step's state; a model-controlled run has no chain and says so
    - stop now / stop after step, compact, merge, delete history, run a finished plan (`run --from-plan`, spawned detached), approve `run_command` and MCP-tool prompts, and answer `ask_user` questions inline
      ("Allow session" appears only where it would grant something beyond the one call it is clicked on)
- **Machine view**: the state overview, the path taken, the current agent state's conversation
    - approve and answer the current state's prompts inline (same controls as a run)
    - the docked entry submits as **Steer** (into the current agent state) or **Message** (a `poke` payload a waiting machine's next tool reads)
    - `machine.notify`/end: ephemeral banners and OS notifications
- **Config page**: every setting with value and source, filterable; click a row to set it
    - enum settings offer their choices; `models.*` autocompletes providers and model ids (the TUI/CLI completion)
    - secrets never shown

Start a machine on the Machines page and watch the current state stream, answering its approvals and questions in place:

<video controls muted loop playsinline preload="metadata" class="no-lightbox">
  <source src="/screenshots/out/web-machine.webm" type="video/webm">
</video>

## Layout

The layout reflows.

- desktop: the nav rail collapses to icons; the run view is a fixed pane, drawer and conversation scrolling internally
- phone: fixed top bar (theme toggle), bottom tab nav, composer docked above it, the page as the only scroller
- phone run view: one widget at a time (conversation by default); the top-bar menu switches to status, task graph, budget, tool calls, latest commit, or event log

## Notifications and installing (PWA)

The page installs as an app (phone home-screen icon or desktop window).

- **🔔 Notifications** on a machine view grants permission
- `machine.notify` and machine-end pop OS notifications: foreground anywhere, backgrounded on desktop, never on a backgrounded phone
- a notification never clears or blocks the inputs; mid-type text and focus survive
- a phone not open on the page: point [`[machine.notify].on_event`](config.md#machinenotify-optional) at a push service

## The HTTP API

The page reads the same wire form as `agent6 attach --json`:

```bash
curl -s localhost:7658/api/hub                       # hub state
curl -s localhost:7658/api/session/<id>              # a run's state, as JSON
curl -s localhost:7658/api/session/<id>/conversation # the folded conversation
curl -s localhost:7658/api/machine/<name>            # a machine's state, as JSON
curl -s localhost:7658/api/config                    # effective config
curl -sN localhost:7658/api/session/<id>/events      # SSE: a snapshot per change
```

- `curl /api/session/<id>`: exactly what `agent6 attach <id> --json` prints; `?step=<sha>` folds only up to that commit
- writes: small JSON `POST`s (`/api/new`, `/api/session/<id>/{steer,approve,answer,merge,undo,resume,run_plan,stop_step,compact,rm}`, `/api/machine/<name>/{poke,stop,steer,approve,answer}`, `/api/sessions/{prune,rm_asks}`, `/api/config`, `/api/machine/{create,run}`)
- every write drives the typed spawn / answer-file contracts, never arbitrary execution
- a machine's `approve`/`answer`/`steer` land in the current agent state's per-state dir; `poke` drops a signal (optional `message`/`data`) on the instance
- machine names and answer ids validate to a single path component: no traversal out of the instance dir

## Remote access (Tailscale)

The server binds `127.0.0.1` by default and has no app-level auth.
For remote access, put [Tailscale](https://tailscale.com) in front of the loopback bind:

```bash
agent6 web                # keep it on 127.0.0.1:7658
tailscale serve --bg 7658 # HTTPS + WireGuard, reachable on your tailnet
```

- the tailnet (WireGuard) identity is the access control; `tailscale serve` terminates HTTPS
- agent6 handles no tokens or passwords
- a non-loopback bind exposes the write surface (spawn runs, answer prompts) to anyone reaching the port
- it refuses without the opt-in: `[web].allow_non_loopback = true` for [`[web].host`](config.md#web), `--allow-non-loopback` for `--host`
- prefer `tailscale serve` over a raw non-loopback bind
