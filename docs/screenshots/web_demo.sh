#!/usr/bin/env bash
# Record the web-UI tour videos (desktop + phone) for the docs site.
#
#   bash docs/screenshots/web_demo.sh
#
# Seeds the committed fixtures (docs/screenshots/seed/) into an isolated,
# temporary agent6 home, starts `agent6 web` against them, and drives it with
# Playwright (web_demo.py) to capture web-desktop.webm + web-phone.webm under
# docs/screenshots/out/ (gitignored). No live LLM calls, no network, no API key.
#
# Needs `agent6`, `python3` (agent6 importable), `ffmpeg` (trims the loading-screen
# head so each video opens on the hub), and a Playwright-capable Python with
# Chromium installed. Point $WEB_DEMO_PY at it, e.g.:
#   python3 -m venv /tmp/pw && /tmp/pw/bin/pip install playwright \
#     && /tmp/pw/bin/playwright install chromium
#   WEB_DEMO_PY=/tmp/pw/bin/python bash docs/screenshots/web_demo.sh
set -euo pipefail

ROOT="$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
cd "$ROOT"
OUT="$ROOT/docs/screenshots/out"
PORT="${WEB_DEMO_PORT:-8987}"
PW_PY="${WEB_DEMO_PY:-python3}"
mkdir -p "$OUT"

for bin in agent6 python3 ffmpeg; do
  command -v "$bin" >/dev/null 2>&1 || { echo "web_demo.sh: missing tool: $bin" >&2; exit 1; }
done
"$PW_PY" -c "from playwright.sync_api import sync_playwright" 2>/dev/null || {
  echo "web_demo.sh: \$WEB_DEMO_PY ($PW_PY) has no playwright; see the header." >&2; exit 1; }

TMP="$(mktemp -d)"
export XDG_CONFIG_HOME="$TMP/config"
export XDG_STATE_HOME="$TMP/state"
export AGENT6_DEMO_REPO="$TMP/demo-repo"
mkdir -p "$AGENT6_DEMO_REPO"
git -C "$AGENT6_DEMO_REPO" init -q

echo "web_demo: seeding fixtures into $TMP"
python3 docs/screenshots/seed.py >/dev/null

echo "web_demo: starting agent6 web on :$PORT"
( cd "$AGENT6_DEMO_REPO" && exec agent6 web --host 127.0.0.1 --port "$PORT" ) >/dev/null 2>&1 &
SERVER=$!
trap 'kill "$SERVER" 2>/dev/null || true; rm -rf "$TMP"' EXIT

for _ in $(seq 1 40); do
  curl -sf "http://127.0.0.1:$PORT/api/meta" >/dev/null 2>&1 && break
  sleep 0.25
done

# `hero` records the desktop tour WITHOUT the narration banner, for the README
# hero (hero.sh), which carries no overlay of any kind.
if [ "${1:-}" = "hero" ]; then
  rm -f "$OUT"/hero-src-web.webm
  "$PW_PY" docs/screenshots/web_demo.py --url "http://127.0.0.1:$PORT" --out "$OUT/hero-src-web.webm" \
    --mode desktop --no-narration
  rm -rf "$OUT"/_web_desktop_raw
  [ -s "$OUT/hero-src-web.webm" ] || { echo "web_demo.sh: failed to produce hero-src-web.webm" >&2; exit 1; }
  echo "web_demo: done -> $OUT/hero-src-web.webm"
  exit 0
fi

rm -f "$OUT"/web-desktop.webm "$OUT"/web-phone.webm
"$PW_PY" docs/screenshots/web_demo.py --url "http://127.0.0.1:$PORT" --out "$OUT/web-desktop.webm" --mode desktop
"$PW_PY" docs/screenshots/web_demo.py --url "http://127.0.0.1:$PORT" --out "$OUT/web-phone.webm" --mode phone
rm -rf "$OUT"/_web_desktop_raw "$OUT"/_web_phone_raw

for f in web-desktop web-phone; do
  [ -s "$OUT/$f.webm" ] || { echo "web_demo.sh: failed to produce $f.webm" >&2; exit 1; }
done
echo "web_demo: done -> $OUT/web-desktop.webm, $OUT/web-phone.webm"
