#!/usr/bin/env bash
# The smallest useful machine: one `agent` state that returns a structured
# greeting, then a terminal. No tools, fully confined, ~1 cent on anthropic.
#
# Usage:  bash bench/machines/hello/run.sh [workdir]
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AGENT6="$(cd "$HERE/../../.." && pwd)/.venv/bin/agent6"
WORK="${1:-/tmp/agent6-machine-hello}"
rm -rf "$WORK"; mkdir -p "$WORK"
cp "$HERE/hello.asm.toml" "$WORK/"
git -C "$WORK" init -q
git -C "$WORK" -c user.email=b@b -c user.name=b commit -q --allow-empty -m init
export XDG_STATE_HOME="$WORK/.agent6-state"
(cd "$WORK" && "$AGENT6" machine run hello.asm.toml)
(cd "$WORK" && "$AGENT6" machine status hello)
