#!/usr/bin/env bash
# Seed a small repo with a few commits, then run repo-digest over it. The digest
# lands in the machine's persistent data dir (printed at the end).
#
# Runs under the default sandbox config: the `tool` states get their own network
# namespace (strict). On a host that supports only the hardened profile, a tool
# state is refused under the default `sandbox.network = "session"` and agent6
# prints the one-line opt-in to apply.
#
# Usage:  bash bench/machines/repo-digest/run.sh [workdir]
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AGENT6="$(cd "$HERE/../../.." && pwd)/.venv/bin/agent6"
WORK="${1:-/tmp/agent6-machine-repo-digest}"

rm -rf "$WORK"; mkdir -p "$WORK/scripts"
cp "$HERE/repo-digest.asm.toml" "$WORK/"
cp "$HERE/scripts/"*.py "$WORK/scripts/"
git -C "$WORK" init -q
gc() { git -C "$WORK" -c user.email=dev@x -c user.name="$1" commit -q --allow-empty -m "$2"; }
gc "Ana"   "feat(api): add pagination to the list endpoint"
gc "Ben"   "fix(auth): reject expired refresh tokens"
gc "Cy"    "docs: document the rate-limit headers"
gc "Ana"   "perf(db): add an index on events(created_at)"
gc "Ben"   "test: cover the token-refresh edge cases"

export XDG_STATE_HOME="$WORK/.agent6-state"

echo "== running repo-digest =="
(cd "$WORK" && AGENT6_FORCE_STREAM=1 "$AGENT6" machine run repo-digest.asm.toml)
echo
echo "== status =="
(cd "$WORK" && "$AGENT6" machine status repo-digest)
echo
echo "== digest.md =="
find "$WORK/.agent6-state" -name digest.md -exec cat {} \;
