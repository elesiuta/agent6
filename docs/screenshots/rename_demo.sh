#!/usr/bin/env bash
# CLI demo video: a real agent6 bug-fix run from the terminal, for people who
# live in the Claude-Code CLI. Not faked logs -- a real `agent6 run` (real loop,
# tools, verify, commit) whose LLM calls are served deterministically by the
# record/replay proxy (llm_proxy.py), so it reproduces exactly with no key.
#
#   bash docs/screenshots/rename_demo.sh            # replay (default): no key, renders the video
#   bash docs/screenshots/rename_demo.sh replay
#   bash docs/screenshots/rename_demo.sh record     # live: real key, recaptures the cassette
#
# record forwards each LLM call to OPENROUTER and saves the trajectory into
# seed/rename-cassette.jsonl (needs a real key in ~/.config/agent6/secrets.toml).
# replay serves that cassette and drives vhs to produce out/rename-demo.webm.
# The cassette and seed/rename-repo (the buggy stats repo) are committed together;
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
PORT=8904
CASSETTE="$SS/seed/rename-cassette.jsonl"

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
DEMO_REPO="/tmp/acme-shop"
trap 'kill "${PROXY_PID:-0}" 2>/dev/null || true; rm -rf "$TMP" "$DEMO_REPO"' EXIT
export XDG_CONFIG_HOME="$TMP/config"
export XDG_STATE_HOME="$TMP/state"
export AGENT6_DEMO_REPO="$DEMO_REPO"
mkdir -p "$XDG_CONFIG_HOME/agent6"

# Provider points at the proxy; the demo never reaches a real model in replay.
# can reach the loopback proxy. the tool network stays private.
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
[budget]
max_usd = 0.50
max_tokens_fallback = 2000000
[workflow]
verify_command = ["python3", "-m", "unittest", "-v"]
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
cp -r "$SS/seed/rename-repo/." "$DEMO_REPO/"
git -C "$DEMO_REPO" init -q
git -C "$DEMO_REPO" -c user.email=demo@agent6.dev -c user.name="agent6 demo" add -A
git -C "$DEMO_REPO" -c user.email=demo@agent6.dev -c user.name="agent6 demo" commit -qm "shop: cart package"

TASK="Complete the task described in TASK.md"

if [ "$MODE" = record ]; then
  command -v agent6 >/dev/null || { echo "rename_demo.sh: missing agent6" >&2; exit 1; }
  [ -r "$HOME/.config/agent6/secrets.toml" ] || { echo "rename_demo.sh record: need ~/.config/agent6/secrets.toml" >&2; exit 1; }
  cp "$HOME/.config/agent6/secrets.toml" "$XDG_CONFIG_HOME/agent6/secrets.toml"
  chmod 600 "$XDG_CONFIG_HOME/agent6/secrets.toml"
  echo "rename demo: RECORD -> $CASSETTE"
  AGENT6_PROXY_MODE=record AGENT6_PROXY_UPSTREAM=https://openrouter.ai \
    AGENT6_PROXY_CASSETTE="$CASSETTE" AGENT6_PROXY_PORT="$PORT" \
    python3 "$SS/llm_proxy.py" & PROXY_PID=$!
  sleep 1
  ( cd "$DEMO_REPO" && agent6 run "$TASK" )
  echo "rename demo: captured $(grep -c . "$CASSETTE") exchanges"
  echo "rename demo: the fix the agent made:"
  git -C "$DEMO_REPO" --no-pager diff
  exit 0
fi

# replay: no key needed; prove it by writing a dummy one.
for bin in vhs ttyd ffmpeg agent6 python3; do
  command -v "$bin" >/dev/null 2>&1 || { echo "rename_demo.sh: missing required tool: $bin" >&2; exit 1; }
done
[ -s "$CASSETTE" ] || { echo "rename_demo.sh: missing cassette $CASSETTE (run 'rename_demo.sh record' first)" >&2; exit 1; }
printf '[providers.openrouter]\napi_key = "unused-in-replay"\n' > "$XDG_CONFIG_HOME/agent6/secrets.toml"
chmod 600 "$XDG_CONFIG_HOME/agent6/secrets.toml"

echo "rename demo: REPLAY proxy on :$PORT <- $CASSETTE"
AGENT6_PROXY_MODE=replay AGENT6_PROXY_CASSETTE="$CASSETTE" AGENT6_PROXY_PORT="$PORT" \
  AGENT6_PROXY_CHUNK_MS=4 python3 "$SS/llm_proxy.py" & PROXY_PID=$!
sleep 1

mkdir -p "$OUT"
rm -f "$OUT/rename-demo.webm"
# Pure typing, no shortcut keys: vhs renders straight to the final webm, no
# keystroke-toast overlay pass (that is for the TUI tapes with M/r/q shortcuts).
echo "rename demo: recording with vhs (rename_demo.tape)"
vhs "$SS/rename_demo.tape"
[ -s "$OUT/rename-demo.webm" ] || { echo "rename_demo.sh: failed to build rename-demo.webm" >&2; exit 1; }
echo "rename demo: done -> $OUT/rename-demo.webm"
