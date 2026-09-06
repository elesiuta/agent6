#!/usr/bin/env bash
# CLI demo video: a real agent6 bug-fix run from the terminal, for people who
# live in the Claude-Code CLI. Not faked logs -- a real `agent6 run` (real loop,
# tools, verify, commit) whose LLM calls are served deterministically by the
# record/replay proxy (llm_proxy.py), so it reproduces exactly with no key.
#
#   bash docs/screenshots/temps_demo.sh            # replay (default): no key, renders the video
#   bash docs/screenshots/temps_demo.sh replay
#   bash docs/screenshots/temps_demo.sh record     # live: real key, recaptures the cassette
#
# record forwards each LLM call to OPENROUTER and saves the trajectory into
# seed/temps-cassette.jsonl (needs a real key in ~/.config/agent6/secrets.toml).
# replay serves that cassette and drives vhs to produce out/temps-demo.webm.
# The cassette and seed/temps-repo (the buggy stats repo) are committed together;
# the cassette's edits target that exact source.
#
# Needs vhs, ttyd, ffmpeg, agent6, python3 on PATH (replay); record needs only
# agent6 + python3 + a key.
set -euo pipefail

MODE="${1:-replay}"
ROOT="$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
cd "$ROOT"
SS="$ROOT/docs/screenshots"
OUT="$SS/out"
PORT=8903
CASSETTE="$SS/seed/temps-cassette.jsonl"

[ -x "$ROOT/.venv/bin/agent6" ] && export PATH="$ROOT/.venv/bin:$PATH"

# Always drive the streaming code path. A real terminal (vhs/ttyd) auto-enables
# streaming because stderr is a tty, so the cassette must hold SSE responses;
# forcing it on for both record and replay keeps the two halves in the same
# format (and a non-tty CI replay then matches too).
export AGENT6_FORCE_STREAM=1

# vhs renders a headless Chromium whose sandbox needs unprivileged userns; the
# jail needs it too. The default Ubuntu 24.04 AppArmor policy blocks it.
sudo sysctl -w kernel.apparmor_restrict_unprivileged_userns=0 >/dev/null 2>&1 || true

TMP="$(mktemp -d)"
DEMO_REPO="/tmp/acme-temps"
trap 'kill "${PROXY_PID:-0}" 2>/dev/null || true; rm -rf "$TMP" "$DEMO_REPO"' EXIT
export AGENT6_CONFIG_HOME="$TMP/config"
export AGENT6_STATE_HOME="$TMP/state"
export AGENT6_DEMO_REPO="$DEMO_REPO"
mkdir -p "$AGENT6_CONFIG_HOME"

# Provider points at the proxy; the demo never reaches a real model in replay.
# can reach the loopback proxy. the tool network stays private.
cat > "$AGENT6_CONFIG_HOME/config.toml" <<EOF
[sandbox]
network = "session"
run_commands = "yes"
protect_git = true
[git]
dirty_tree = "ask"
branch_per_run = true
[git.commit]
name = "agent6 demo"
email = "demo@agent6.dev"
[budget]
max_usd = 0.50
max_tokens_fallback = 2000000
[workflow]
verify_command = ["python3", "-m", "unittest", "discover", "-s", "tests", "-t", "."]
[providers.openrouter]
api_format = "openai"
base_url = "http://127.0.0.1:${PORT}/api/v1"
[models.worker]
provider = "openrouter"
model = "moonshotai/kimi-k3"
[models.planner]
provider = "openrouter"
model = "moonshotai/kimi-k3"
[models.reviewer]
provider = "openrouter"
model = "moonshotai/kimi-k3"
EOF

# Same starting repo as the recording: the buggy median() + failing test.
rm -rf "$DEMO_REPO"
mkdir -p "$DEMO_REPO"
cp -r "$SS/seed/temps-repo/." "$DEMO_REPO/"
git -C "$DEMO_REPO" init -q
git -C "$DEMO_REPO" -c user.email=demo@agent6.dev -c user.name="agent6 demo" add -A
git -C "$DEMO_REPO" -c user.email=demo@agent6.dev -c user.name="agent6 demo" commit -qm "temps: conversions"

TASK="A test in tests/test_temps.py is failing. Find and fix the bug in temps.py so the whole suite passes. Keep the change minimal."

if [ "$MODE" = record ]; then
  command -v agent6 >/dev/null || { echo "temps_demo.sh: missing agent6" >&2; exit 1; }
  [ -r "$HOME/.config/agent6/secrets.toml" ] || { echo "temps_demo.sh record: need ~/.config/agent6/secrets.toml" >&2; exit 1; }
  cp "$HOME/.config/agent6/secrets.toml" "$AGENT6_CONFIG_HOME/secrets.toml"
  chmod 600 "$AGENT6_CONFIG_HOME/secrets.toml"
  echo "temps demo: RECORD -> $CASSETTE"
  AGENT6_PROXY_MODE=record AGENT6_PROXY_UPSTREAM=https://openrouter.ai \
    AGENT6_PROXY_CASSETTE="$CASSETTE" AGENT6_PROXY_PORT="$PORT" \
    python3 "$SS/llm_proxy.py" & PROXY_PID=$!
  sleep 1
  ( cd "$DEMO_REPO" && agent6 run "$TASK" )
  echo "temps demo: captured $(grep -c . "$CASSETTE") exchanges"
  echo "temps demo: the fix the agent made:"
  git -C "$DEMO_REPO" --no-pager diff
  exit 0
fi

# replay: no key needed; prove it by writing a dummy one.
for bin in vhs ttyd ffmpeg agent6 python3; do
  command -v "$bin" >/dev/null 2>&1 || { echo "temps_demo.sh: missing required tool: $bin" >&2; exit 1; }
done
[ -s "$CASSETTE" ] || { echo "temps_demo.sh: missing cassette $CASSETTE (run 'temps_demo.sh record' first)" >&2; exit 1; }
printf '[providers.openrouter]\napi_key = "unused-in-replay"\n' > "$AGENT6_CONFIG_HOME/secrets.toml"
chmod 600 "$AGENT6_CONFIG_HOME/secrets.toml"

echo "temps demo: REPLAY proxy on :$PORT <- $CASSETTE"
AGENT6_PROXY_MODE=replay AGENT6_PROXY_CASSETTE="$CASSETTE" AGENT6_PROXY_PORT="$PORT" \
  AGENT6_PROXY_CHUNK_MS=4 python3 "$SS/llm_proxy.py" & PROXY_PID=$!
sleep 1

mkdir -p "$OUT"
rm -f "$OUT/temps-demo.webm"
# Pure typing, no shortcut keys: vhs renders straight to the final webm, no
# keystroke-toast overlay pass (that is for the TUI tapes with M/r/q shortcuts).
echo "temps demo: recording with vhs (temps_demo.tape)"
vhs "$SS/temps_demo.tape"
[ -s "$OUT/temps-demo.webm" ] || { echo "temps_demo.sh: failed to build temps-demo.webm" >&2; exit 1; }
echo "temps demo: done -> $OUT/temps-demo.webm"
