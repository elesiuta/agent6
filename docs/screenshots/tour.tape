# agent6 TUI tour: captures the release screenshots with vhs
# (https://github.com/charmbracelet/vhs).
#
#   bash docs/screenshots/generate.sh        # seeds fixtures, runs this tape
#
# Drives hub -> config -> search -> help -> log -> conversation -> dashboard
# and writes one PNG per stop. Requires `vhs`, `ttyd`, `ffmpeg`, and `agent6` on
# PATH. generate.sh exports XDG_STATE_HOME / XDG_CONFIG_HOME /
# AGENT6_DEMO_REPO (seeded, isolated) before invoking vhs; the tape inherits
# them. Hub stops use its direct key bindings (Enter/l/c/?/q and /); inside the
# run app the composer bar owns plain letters, so only its priority chords
# (Ctrl+D, PageUp, Esc) are used. The config/search/help/log screens are
# captured from the freshly mounted hub BEFORE entering the run, because
# hub<->run are separate apps and the app restart drops the first keystroke
# that follows it.
#
# This tape's job is the PNGs. vhs requires an Output, but this recording is a
# throwaway -- the tour.webm video is recorded separately by reel.tape (no
# Screenshots, so the toast timeline stays accurate). The wide
# 1920x1080 terminal keeps the dashboard's tool table and diff from clipping.
Output docs/screenshots/out/_capture.webm

Require agent6

Set Shell bash
Set FontFamily "JetBrains Mono"
Set FontSize 16
Set Width 1920
Set Height 1080
Set Padding 16
# agent6-dark, the TUI's own branded palette (src/agent6/ui/tui/theme.py):
# a near-black teal, the cyan brand's neutral, so the recordings look like
# the product rather than like a generic terminal theme.
Set Theme { "name": "agent6-dark", "background": "#0f1414", "foreground": "#C0CAF5", "cursor": "#06F5F3", "selection": "#1e2626", "black": "#161c1c", "red": "#F7768E", "green": "#9ECE6A", "yellow": "#E0AF68", "blue": "#7AA2F7", "magenta": "#BB9AF7", "cyan": "#06F5F3", "white": "#C0CAF5", "brightBlack": "#6E6E78", "brightRed": "#FF8FA3", "brightGreen": "#B4E07F", "brightYellow": "#FFA62B", "brightBlue": "#9CBCFF", "brightMagenta": "#D3B4FF", "brightCyan": "#66FFFD", "brightWhite": "#F2F4FF" }

Hide
Type "cd ${AGENT6_DEMO_REPO:-.}" Enter
Type "clear" Enter
Show

# --- the hub: run history for the repo -----------------------------------
# Wait for the hub to finish scanning runs + render before the shot.
Type "agent6 tui" Enter
Sleep 5s
Screenshot docs/screenshots/out/01-hub.png

# --- the config page (c): every value + where it came from ---------------
Type "c"
Sleep 2.5s
Screenshot docs/screenshots/out/03-config.png

# --- search the settings (/) ---------------------------------------------
Type "/"
Sleep 700ms
Type "sandbox"
Sleep 1.5s
Screenshot docs/screenshots/out/04-config-search.png
Escape
Sleep 500ms
Escape
Sleep 1s

# --- the keys & actions help (?) -----------------------------------------
Type "?"
Sleep 1.5s
Screenshot docs/screenshots/out/08-help.png
Escape
Sleep 1s

# --- the event log of the selected run (l), straight from the hub --------
Type "l"
Sleep 1.5s
Screenshot docs/screenshots/out/09-logs.png
Escape
Sleep 1s

# --- open the top run -> the conversation, the run app's primary view ----
# (assistant turns, tool calls, the composer bar). It opens following the
# end; two pages up centers an assistant turn with its reasoning + tool
# calls instead of the run's tail.
Enter
Sleep 3.5s
PageUp
Sleep 300ms
PageUp
Sleep 1s
Screenshot docs/screenshots/out/05-transcript.png

# --- the live dashboard (Ctrl+D toggles conversation <-> dashboard) ------
# (DAG, budget bar, tool table, reasoning stream, log tail, diff)
Ctrl+D
Sleep 3s
Screenshot docs/screenshots/out/02-run-dashboard.png

# --- back to the hub and quit --------------------------------------------
Escape
Sleep 2s
Type "q"
Sleep 1s
