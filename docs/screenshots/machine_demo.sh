#!/usr/bin/env bash
# Machine demo video: a real agent6 state machine running in the TUI. Not faked
# logs -- a real `agent6 machine run` (the code-fixer fix-loop: an agent state
# edits the repo, a tool state verifies, a branch loops until green) whose agent
# LLM calls are served deterministically by the record/replay proxy
# (llm_proxy.py), so the watch view reproduces exactly with no key.
#
#   bash docs/screenshots/machine_demo.sh           # replay (default): no key, renders the video
#   bash docs/screenshots/machine_demo.sh replay
#   bash docs/screenshots/machine_demo.sh record    # live: real key, recaptures the cassette
#
# record runs the machine once against OPENROUTER and saves the agent's
# trajectory into seed/machine-cassette.jsonl (needs a real key in
# ~/.config/agent6/secrets.toml). replay serves that cassette, drives the TUI
# Machines page (run -> watch) with vhs, and produces out/machine-demo.webm. The
# machine bundle (seed/machine-repo/) and the cassette are committed together.
#
# AGENT6_FORCE_STREAM=1 throughout: the agent state's chat calls are SSE, so the
# watch view streams its reasoning live (the cassette is captured streaming, so
# replay re-streams it).
set -euo pipefail

MODE="${1:-replay}"
ROOT="$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
cd "$ROOT"
SS="$ROOT/docs/screenshots"
OUT="$SS/out"
PORT=8903
CASSETTE="$SS/seed/machine-cassette.jsonl"

[ -x "$ROOT/.venv/bin/agent6" ] && export PATH="$ROOT/.venv/bin:$PATH"
export AGENT6_FORCE_STREAM=1

sudo sysctl -w kernel.apparmor_restrict_unprivileged_userns=0 >/dev/null 2>&1 || true

TMP="$(mktemp -d)"
# A clean, project-looking path so the Machines header reads `.../acme-stats`.
DEMO_REPO="/tmp/acme-stats"
trap 'kill "${PROXY_PID:-0}" 2>/dev/null || true; rm -rf "$TMP" "$DEMO_REPO"' EXIT
export XDG_CONFIG_HOME="$TMP/config"
export XDG_STATE_HOME="$TMP/state"
export AGENT6_DEMO_REPO="$DEMO_REPO"
mkdir -p "$XDG_CONFIG_HOME/agent6"

# The agent state inherits the worker model, which points at the proxy. open
cat > "$XDG_CONFIG_HOME/agent6/config.toml" <<EOF
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
[providers.openrouter]
api_format = "openai"
base_url = "http://127.0.0.1:${PORT}/api/v1"
[models.worker]
provider = "openrouter"
model = "moonshotai/kimi-k2.6"
[models.planner]
provider = "openrouter"
model = "moonshotai/kimi-k2.6"
[models.reviewer]
provider = "openrouter"
model = "moonshotai/kimi-k2.6"
EOF

# The demo repo holds the machine bundle + the buggy source it fixes.
rm -rf "$DEMO_REPO"
mkdir -p "$DEMO_REPO/scripts"
cp "$SS/seed/machine-repo/code-fixer.asm.toml" "$DEMO_REPO/"
cp "$SS/seed/machine-repo/scripts/"*.py "$DEMO_REPO/scripts/"
cp "$SS/seed/machine-repo/stats.py" "$DEMO_REPO/"
git -C "$DEMO_REPO" init -q
git -C "$DEMO_REPO" -c user.email=demo@agent6.dev -c user.name="agent6 demo" add -A
git -C "$DEMO_REPO" -c user.email=demo@agent6.dev -c user.name="agent6 demo" commit -qm "code-fixer machine + buggy median"

if [ "$MODE" = record ]; then
  command -v agent6 >/dev/null || { echo "machine_demo.sh: missing agent6" >&2; exit 1; }
  [ -r "$HOME/.config/agent6/secrets.toml" ] || { echo "machine_demo.sh record: need ~/.config/agent6/secrets.toml" >&2; exit 1; }
  cp "$HOME/.config/agent6/secrets.toml" "$XDG_CONFIG_HOME/agent6/secrets.toml"
  chmod 600 "$XDG_CONFIG_HOME/agent6/secrets.toml"
  echo "machine demo: RECORD -> $CASSETTE"
  AGENT6_PROXY_MODE=record AGENT6_PROXY_UPSTREAM=https://openrouter.ai \
    AGENT6_PROXY_CASSETTE="$CASSETTE" AGENT6_PROXY_PORT="$PORT" \
    python3 "$SS/llm_proxy.py" & PROXY_PID=$!
  sleep 1
  ( cd "$DEMO_REPO" && agent6 machine run code-fixer.asm.toml )
  echo "machine demo: captured $(grep -c . "$CASSETTE") exchanges"
  echo "machine demo: the fix the agent made:"
  git -C "$DEMO_REPO" --no-pager diff HEAD~1 -- stats.py 2>/dev/null || git -C "$DEMO_REPO" --no-pager show --stat HEAD
  exit 0
fi

# replay: no key needed.
for bin in vhs ttyd ffmpeg agent6 python3; do
  command -v "$bin" >/dev/null 2>&1 || { echo "machine_demo.sh: missing required tool: $bin" >&2; exit 1; }
done
[ -s "$CASSETTE" ] || { echo "machine_demo.sh: missing cassette $CASSETTE (run 'machine_demo.sh record' first)" >&2; exit 1; }
printf '[providers.openrouter]\napi_key = "unused-in-replay"\n' > "$XDG_CONFIG_HOME/agent6/secrets.toml"
chmod 600 "$XDG_CONFIG_HOME/agent6/secrets.toml"

echo "machine demo: REPLAY proxy on :$PORT <- $CASSETTE"
AGENT6_PROXY_MODE=replay AGENT6_PROXY_CASSETTE="$CASSETTE" AGENT6_PROXY_PORT="$PORT" \
  AGENT6_PROXY_CHUNK_MS=12 python3 "$SS/llm_proxy.py" & PROXY_PID=$!
sleep 1

mkdir -p "$OUT"
rm -f "$OUT/_machine-raw.webm" "$OUT/machine-demo.webm"
echo "machine demo: recording with vhs (machine_demo.tape)"
vhs "$SS/machine_demo.tape"
python3 "$SS/keystroke_overlay.py" "$SS/machine_demo.tape" "$OUT/_machine-raw.webm" "$OUT/machine-demo.webm"
rm -f "$OUT/_machine-raw.webm"
[ -s "$OUT/machine-demo.webm" ] || { echo "machine_demo.sh: failed to build machine-demo.webm" >&2; exit 1; }
echo "machine demo: done -> $OUT/machine-demo.webm"
