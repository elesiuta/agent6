#!/usr/bin/env bash
# Record the CLI hero source: the same real run from the plain terminal,
# replayed from the committed CLI cassette (llm_proxy.py, no key), captured by
# vhs as out/hero-src-cli.webm. Same seed and cassette as cli_demo.sh; the
# pacing differs (model-speed streaming, so the reasoning reads on film).
#
#   bash docs/screenshots/hero_cli.sh
#
# Needs vhs, ttyd, ffmpeg, agent6, python3 on PATH.
set -euo pipefail

ROOT="$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
cd "$ROOT"
SS="$ROOT/docs/screenshots"
OUT="$SS/out"
PORT=8905
CASSETTE="$SS/seed/cli-cassette.jsonl"

[ -x "$ROOT/.venv/bin/agent6" ] && export PATH="$ROOT/.venv/bin:$PATH"

# The cassette holds SSE responses; force the streaming path everywhere so tty
# and non-tty replays match it (see cli_demo.sh).
export AGENT6_FORCE_STREAM=1

# vhs renders a headless Chromium whose sandbox needs unprivileged userns; the
# jail needs it too. The default Ubuntu 24.04 AppArmor policy blocks it.
sudo sysctl -w kernel.apparmor_restrict_unprivileged_userns=0 >/dev/null 2>&1 || true

for bin in vhs ttyd ffmpeg agent6 python3; do
  command -v "$bin" >/dev/null 2>&1 || { echo "hero_cli.sh: missing required tool: $bin" >&2; exit 1; }
done
[ -s "$CASSETTE" ] || { echo "hero_cli.sh: missing cassette $CASSETTE (run 'cli_demo.sh record' first)" >&2; exit 1; }

TMP="$(mktemp -d)"
DEMO_REPO="/tmp/acme-stats-hero-cli"
trap 'kill "${PROXY_PID:-0}" 2>/dev/null || true; rm -rf "$TMP" "$DEMO_REPO"' EXIT
export XDG_CONFIG_HOME="$TMP/config"
export XDG_STATE_HOME="$TMP/state"
export AGENT6_DEMO_REPO="$DEMO_REPO"
mkdir -p "$XDG_CONFIG_HOME/agent6"

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

echo "hero_cli: REPLAY proxy on :$PORT <- $CASSETTE"
# 60ms per SSE chunk paces the replay like a live model (~16s run), so the
# streamed reasoning reads on film; cli_demo.sh uses 4ms because its terminal
# beats only need the result.
AGENT6_PROXY_MODE=replay AGENT6_PROXY_CASSETTE="$CASSETTE" AGENT6_PROXY_PORT="$PORT" \
  AGENT6_PROXY_CHUNK_MS=60 python3 "$SS/llm_proxy.py" & PROXY_PID=$!
sleep 1

mkdir -p "$OUT"
rm -f "$OUT/hero-src-cli.webm"
echo "hero_cli: recording with vhs (hero_cli.tape)"
vhs "$SS/hero_cli.tape"
[ -s "$OUT/hero-src-cli.webm" ] || { echo "hero_cli.sh: failed to build hero-src-cli.webm" >&2; exit 1; }
echo "hero_cli: done -> $OUT/hero-src-cli.webm"
