#!/usr/bin/env bash
# Web state-machine tour video: the code-fixer machine started and watched from
# the browser. The same real `agent6 machine run` as machine_demo.sh (agent LLM
# calls served deterministically from the committed cassette by llm_proxy.py),
# but recorded from `agent6 web` with Playwright (web_demo.py --mode machine)
# instead of the TUI with vhs. Produces out/web-machine.webm; no API key.
#
#   WEB_DEMO_PY=/tmp/pw/bin/python bash docs/screenshots/web_machine_demo.sh
#
# Needs `agent6`, `python3` (agent6 importable), `ffmpeg`, and a
# Playwright-capable Python in $WEB_DEMO_PY (see web_demo.sh's header).
set -euo pipefail

ROOT="$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
cd "$ROOT"
SS="$ROOT/docs/screenshots"
OUT="$SS/out"
PROXY_PORT=8905
WEB_PORT="${WEB_MACHINE_DEMO_PORT:-8989}"
CASSETTE="$SS/seed/machine-cassette.jsonl"
PW_PY="${WEB_DEMO_PY:-python3}"

[ -x "$ROOT/.venv/bin/agent6" ] && export PATH="$ROOT/.venv/bin:$PATH"
export AGENT6_FORCE_STREAM=1

for bin in agent6 python3 ffmpeg; do
  command -v "$bin" >/dev/null 2>&1 || { echo "web_machine_demo.sh: missing tool: $bin" >&2; exit 1; }
done
"$PW_PY" -c "from playwright.sync_api import sync_playwright" 2>/dev/null || {
  echo "web_machine_demo.sh: \$WEB_DEMO_PY ($PW_PY) has no playwright; see web_demo.sh." >&2; exit 1; }
[ -s "$CASSETTE" ] || { echo "web_machine_demo.sh: missing cassette $CASSETTE" >&2; exit 1; }

# The machine's agent state runs jailed; its Chromium-style userns needs match
# machine_demo.sh (Ubuntu 24.04 AppArmor).
sudo sysctl -w kernel.apparmor_restrict_unprivileged_userns=0 >/dev/null 2>&1 || true

TMP="$(mktemp -d)"
# A clean, project-looking path so the machines header reads `.../acme-stats`.
DEMO_REPO="/tmp/acme-stats-web"
trap 'kill "${PROXY_PID:-0}" "${SERVER:-0}" 2>/dev/null || true; rm -rf "$TMP" "$DEMO_REPO"' EXIT
export AGENT6_CONFIG_HOME="$TMP/config"
export AGENT6_STATE_HOME="$TMP/state"
mkdir -p "$AGENT6_CONFIG_HOME"

# Same provider/sandbox shape as machine_demo.sh: the worker dials the replay
# proxy on loopback, so the jailed agent needs open agent egress; tools blocked.
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
[providers.openrouter]
api_format = "openai"
base_url = "http://127.0.0.1:${PROXY_PORT}/api/v1"
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
printf '[providers.openrouter]\napi_key = "unused-in-replay"\n' > "$AGENT6_CONFIG_HOME/secrets.toml"
chmod 600 "$AGENT6_CONFIG_HOME/secrets.toml"

# The demo repo holds the machine bundle + the buggy source it fixes.
rm -rf "$DEMO_REPO"
mkdir -p "$DEMO_REPO/scripts"
cp "$SS/seed/machine-repo/code-fixer.asm.toml" "$DEMO_REPO/"
cp "$SS/seed/machine-repo/scripts/"*.py "$DEMO_REPO/scripts/"
cp "$SS/seed/machine-repo/stats.py" "$DEMO_REPO/"
git -C "$DEMO_REPO" init -q
git -C "$DEMO_REPO" -c user.email=demo@agent6.dev -c user.name="agent6 demo" add -A
git -C "$DEMO_REPO" -c user.email=demo@agent6.dev -c user.name="agent6 demo" commit -qm "code-fixer machine + buggy median"

echo "web machine demo: REPLAY proxy on :$PROXY_PORT <- $CASSETTE"
# 40ms chunks (vs the TUI demo's 12): the browser tour has no fixed tape holds,
# so slower streaming is what makes the agent state visibly THINK on camera
# instead of the machine finishing before the viewer has opened it.
AGENT6_PROXY_MODE=replay AGENT6_PROXY_CASSETTE="$CASSETTE" AGENT6_PROXY_PORT="$PROXY_PORT" \
  AGENT6_PROXY_CHUNK_MS=40 python3 "$SS/llm_proxy.py" & PROXY_PID=$!
sleep 1

echo "web machine demo: starting agent6 web on :$WEB_PORT"
( cd "$DEMO_REPO" && exec agent6 web --host 127.0.0.1 --port "$WEB_PORT" ) >/dev/null 2>&1 &
SERVER=$!
for _ in $(seq 1 40); do
  curl -sf "http://127.0.0.1:$WEB_PORT/api/meta" >/dev/null 2>&1 && break
  sleep 0.25
done

mkdir -p "$OUT"
rm -f "$OUT/web-machine.webm"
"$PW_PY" "$SS/web_demo.py" --url "http://127.0.0.1:$WEB_PORT" --out "$OUT/web-machine.webm" --mode machine
rm -rf "$OUT"/_web_machine_raw
[ -s "$OUT/web-machine.webm" ] || { echo "web_machine_demo.sh: failed to produce web-machine.webm" >&2; exit 1; }
echo "web machine demo: done -> $OUT/web-machine.webm"
