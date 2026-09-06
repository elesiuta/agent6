#!/usr/bin/env bash
# Exercise the three timing behaviors: `until` (deadline), the `--exit-on-wait`
# persisted-wake driver (pulse), and the v1 cron-raises limitation (cron-demo).
# Pure wait/terminal machines, so this needs no provider keys and runs confined.
#
# Usage:  bash bench/machines/wait-clock/run.sh [workdir]
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AGENT6="$(cd "$HERE/../../.." && pwd)/.venv/bin/agent6"
WORK="${1:-/tmp/agent6-machine-wait}"

rm -rf "$WORK"; mkdir -p "$WORK"
cp "$HERE"/*.asm.toml "$WORK/"
git -C "$WORK" init -q
git -C "$WORK" -c user.email=b@b -c user.name=b commit -q --allow-empty -m init
export XDG_STATE_HOME="$WORK/.agent6-state"
cd "$WORK"

echo "== deadline: until a past instant fires immediately =="
"$AGENT6" machine run deadline.asm.toml

echo
echo "== pulse: --exit-on-wait persists the wake and exits, no blocking =="
"$AGENT6" machine run pulse.asm.toml --exit-on-wait
echo "-- status (note next wake) --"
"$AGENT6" machine status pulse
echo "-- resume immediately: still waiting (wake not reached) --"
"$AGENT6" machine run pulse.asm.toml --exit-on-wait
echo "-- wait out the interval, then resume: advances to done --"
sleep 5
"$AGENT6" machine run pulse.asm.toml --exit-on-wait

echo
echo "== cron-demo: machine check REJECTS a cron wait at load time (fail-loud) =="
set +e
"$AGENT6" machine check cron-demo.asm.toml; echo "cron-demo check exit: $? (expected 1)"
set -e
