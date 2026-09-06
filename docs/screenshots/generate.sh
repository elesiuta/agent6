#!/usr/bin/env bash
# Generate the TUI screenshots + tour video for the docs site.
#
#   bash docs/screenshots/generate.sh
#
# Seeds committed fixtures (docs/screenshots/seed/) into an isolated, temporary
# agent6 home, then drives the TUI with vhs (docs/screenshots/tour.tape) to
# capture PNGs + tour.webm under docs/screenshots/out/ (gitignored).
# No live LLM calls, no network, no API key; everything renders from the seeded
# run logs. The pages workflow runs this before `mkdocs build`. Needs `vhs`,
# `ttyd`, `ffmpeg`, `agent6`, and `python3` (with agent6 importable) on PATH.
set -euo pipefail

ROOT="$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
cd "$ROOT"
OUT="$ROOT/docs/screenshots/out"
mkdir -p "$OUT"

# The screenshots tour.tape writes, kept in sync with the docs pages that embed
# them. generate.sh fails if it does not produce every one.
SHOTS=(01-hub 02-run-dashboard 03-config 04-config-search 05-transcript 08-help 09-logs)

for bin in vhs ttyd ffmpeg agent6 python3; do
  command -v "$bin" >/dev/null 2>&1 || { echo "generate.sh: missing required tool: $bin" >&2; exit 1; }
done

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
export XDG_CONFIG_HOME="$TMP/config"
export XDG_STATE_HOME="$TMP/state"
export AGENT6_DEMO_REPO="${AGENT6_DEMO_REPO:-$ROOT}"

echo "screenshots: seeding fixtures into $TMP"
python3 docs/screenshots/seed.py

# Fresh media each run; out/ is gitignored (committed: the .tape + seed only).
rm -f "$OUT"/*.png "$OUT"/tour.webm "$OUT"/_capture.webm

echo "screenshots: capturing PNGs with vhs (tour.tape)"
vhs docs/screenshots/tour.tape

missing=0
for s in "${SHOTS[@]}"; do
  [ -s "$OUT/$s.png" ] || { echo "generate.sh: tour.tape did not produce $s.png" >&2; missing=1; }
done
[ "$missing" = 0 ] || exit 1

rm -f "$OUT"/_capture.webm  # the throwaway video from the screenshot pass

# The tour.webm reel is one vhs recording of a single TUI session (reel.tape)
# with animated keystroke toasts overlaid (keystroke_overlay.py). The overlay
# scales the keypress timeline to the recording's ACTUAL duration, so the toasts
# stay aligned even though the live dashboard records a little fast (it redraws
# faster than vhs captures at this resolution).
echo "screenshots: recording reel.tape with vhs"
rm -f "$OUT/tour-raw.webm"
vhs docs/screenshots/reel.tape
python3 docs/screenshots/keystroke_overlay.py \
  docs/screenshots/reel.tape "$OUT/tour-raw.webm" "$OUT/tour.webm"
[ -s "$OUT/tour.webm" ] || { echo "generate.sh: failed to build tour.webm" >&2; exit 1; }
rm -f "$OUT/tour-raw.webm"  # overlay intermediate; hero sources come from hero_run/hero_cli

echo "screenshots: done -> $OUT"
