#!/usr/bin/env bash
# Run the council machine. Two jurors on different providers answer a yes/no
# question; if they disagree a stronger model breaks the tie. No tool states, so
# it runs fully confined (each agent's egress is pinned to its provider) on any
# profile, including the hardened fallback used when strict's broker is blocked.
#
# Usage:  bash bench/machines/multi-model-council/run.sh [workdir] [question]
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AGENT6="$(cd "$HERE/../../.." && pwd)/.venv/bin/agent6"
WORK="${1:-/tmp/agent6-machine-council}"

rm -rf "$WORK"; mkdir -p "$WORK"
cp "$HERE/council.asm.toml" "$WORK/"
git -C "$WORK" init -q
git -C "$WORK" -c user.email=b@b -c user.name=b commit -q --allow-empty -m init
export XDG_STATE_HOME="$WORK/.agent6-state"

echo "== running council =="
(cd "$WORK" && AGENT6_FORCE_STREAM=1 "$AGENT6" machine run council.asm.toml)
echo
echo "== status (verdicts + path) =="
(cd "$WORK" && "$AGENT6" machine status council)
