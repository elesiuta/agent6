#!/usr/bin/env bash
# Record the TUI hero source: a real agent6 run in the run TUI, replayed from
# the committed CLI cassette (llm_proxy.py, no key), captured by vhs as
# out/hero-src-tui.webm. Same seed and cassette as cli_demo.sh; only the
# surface differs (the run TUI instead of the terminal).
#
#   bash docs/screenshots/hero_run.sh
#
# Needs vhs, ttyd, ffmpeg, agent6, python3 on PATH.
set -euo pipefail

ROOT="$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
cd "$ROOT"
SS="$ROOT/docs/screenshots"
OUT="$SS/out"
PORT=8903
CASSETTE="$SS/seed/cli-cassette.jsonl"

[ -x "$ROOT/.venv/bin/agent6" ] && export PATH="$ROOT/.venv/bin:$PATH"

# The cassette holds SSE responses; force the streaming path everywhere so tty
# and non-tty replays match it (see cli_demo.sh).
export AGENT6_FORCE_STREAM=1

# vhs renders a headless Chromium whose sandbox needs unprivileged userns; the
# jail needs it too. The default Ubuntu 24.04 AppArmor policy blocks it.
sudo sysctl -w kernel.apparmor_restrict_unprivileged_userns=0 >/dev/null 2>&1 || true

for bin in vhs ttyd ffmpeg agent6 python3; do
  command -v "$bin" >/dev/null 2>&1 || { echo "hero_run.sh: missing required tool: $bin" >&2; exit 1; }
done
[ -s "$CASSETTE" ] || { echo "hero_run.sh: missing cassette $CASSETTE (run 'cli_demo.sh record' first)" >&2; exit 1; }

TMP="$(mktemp -d)"
DEMO_REPO="/tmp/acme-stats-hero"
trap 'kill "${PROXY_PID:-0}" 2>/dev/null || true; rm -rf "$TMP" "$DEMO_REPO"' EXIT
export XDG_CONFIG_HOME="$TMP/config"
export XDG_STATE_HOME="$TMP/state"
export AGENT6_DEMO_REPO="$DEMO_REPO"
mkdir -p "$XDG_CONFIG_HOME/agent6"

# run_commands=ask so the verify command raises the TUI's approval modal: the
# permission story on film, and a deterministic pause the tape answers with y.
cat > "$XDG_CONFIG_HOME/agent6/config.toml" <<EOF
[sandbox]
network = "session"
run_commands = "ask"
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
model = "moonshotai/kimi-k2.6"
[models.planner]
provider = "openrouter"
model = "moonshotai/kimi-k2.6"
[models.reviewer]
provider = "openrouter"
model = "moonshotai/kimi-k2.6"
EOF
printf '[providers.openrouter]\napi_key = "unused-in-replay"\n' > "$XDG_CONFIG_HOME/agent6/secrets.toml"
chmod 600 "$XDG_CONFIG_HOME/agent6/secrets.toml"

# Same starting repo as the cassette recording: the buggy median() + failing test.
rm -rf "$DEMO_REPO"
mkdir -p "$DEMO_REPO"
cp -r "$SS/seed/cli-repo/." "$DEMO_REPO/"
git -C "$DEMO_REPO" init -q
git -C "$DEMO_REPO" -c user.email=demo@agent6.dev -c user.name="agent6 demo" add -A
git -C "$DEMO_REPO" -c user.email=demo@agent6.dev -c user.name="agent6 demo" commit -qm "stats: mean + median"

echo "hero_run: REPLAY proxy on :$PORT <- $CASSETTE"
# 60ms per SSE chunk paces the replay like a live model (~20s run), so the TUI
# has real streaming to show; cli_demo.sh uses 4ms because its terminal beats
# only need the result.
AGENT6_PROXY_MODE=replay AGENT6_PROXY_CASSETTE="$CASSETTE" AGENT6_PROXY_PORT="$PORT" \
  AGENT6_PROXY_CHUNK_MS=60 python3 "$SS/llm_proxy.py" & PROXY_PID=$!
sleep 1

mkdir -p "$OUT"
rm -f "$OUT/hero-src-tui.webm"
echo "hero_run: recording with vhs (hero_run.tape)"
vhs "$SS/hero_run.tape"
[ -s "$OUT/hero-src-tui.webm" ] || { echo "hero_run.sh: failed to build hero-src-tui.webm" >&2; exit 1; }
echo "hero_run: done -> $OUT/hero-src-tui.webm"
