# CLI demo for the docs: a real agent6 bug-fix run from the terminal, for people
# who live in the Claude-Code CLI. Not faked logs -- a real `agent6 run` whose
# LLM calls are served from a recorded cassette by llm_proxy.py (replay mode, no
# key), so the run is real (loop, tools, verify, commit) and deterministic.
# cli_demo.sh starts the proxy, seeds the buggy stats repo, and records this.
#
# Beats (one continuous scroll, like a real terminal -- no clears): a failing
# test -> hand it to agent6 (it finds + fixes the bug, reasoning streamed) ->
# `runs diff` (the one-line fix, the payoff) -> `attach --since` (the
# event-log detail: verify passed, auto-commit, finish) -> `runs show` (status).
#
# This tape is pure typing (no shortcut keys), so cli_demo.sh renders it straight
# with no keystroke-toast overlay pass.
#
# 1280x720 / Framerate 8, matching the other tapes.
Output docs/screenshots/out/cli-demo.webm

Require agent6

Set Shell bash
Set FontFamily "JetBrains Mono"
Set FontSize 13
Set Width 1280
Set Height 720
Set Padding 14
# agent6-dark, the TUI's own branded palette (src/agent6/ui/tui/theme.py):
# a near-black teal, the cyan brand's neutral, so the recordings look like
# the product rather than like a generic terminal theme.
Set Theme { "name": "agent6-dark", "background": "#0f1414", "foreground": "#C0CAF5", "cursor": "#06F5F3", "selection": "#1e2626", "black": "#161c1c", "red": "#F7768E", "green": "#9ECE6A", "yellow": "#E0AF68", "blue": "#7AA2F7", "magenta": "#BB9AF7", "cyan": "#06F5F3", "white": "#C0CAF5", "brightBlack": "#6E6E78", "brightRed": "#FF8FA3", "brightGreen": "#B4E07F", "brightYellow": "#FFA62B", "brightBlue": "#9CBCFF", "brightMagenta": "#D3B4FF", "brightCyan": "#66FFFD", "brightWhite": "#F2F4FF" }
Set Framerate 8

Hide
Type "cd ${AGENT6_DEMO_REPO:-.}" Enter
Type "export PS1='$ '" Enter
# Don't let the pre-show test write __pycache__ (it would dirty the worktree and
# the run would refuse to start under dirty_tree = "ask").
Type "export PYTHONDONTWRITEBYTECODE=1" Enter
Type "clear" Enter
Show

# 1. The failing suite, then hand the bug to agent6.
Sleep 1s
Type@55ms "python3 -m unittest discover -s tests -t . -q"
Enter
Sleep 2.5s
Type@55ms `agent6 run "Find and fix the failing test in tests/test_stats.py"`
Enter
# The replayed run streams to its summary, then waits at the follow-up prompt
# ("next (/exit to finish):"); typing the next command earlier would steer the
# run instead of the shell and garble the scroll.
Sleep 14s
Type@55ms "/exit"
Enter
Sleep 1.5s

# 2. The fix it made: base -> run branch, one minimal diff.
Type@55ms "agent6 sessions diff"
Enter
Sleep 5s

# 3. The event log behind it: verify passed, auto-commit, finish_session.
Type@55ms "agent6 attach --since 14"
Enter
Sleep 7s
Ctrl+C
Sleep 1.5s

# 4. Final liveness + cost.
Type@55ms "agent6 sessions show"
Enter
Sleep 4.5s
