#!/usr/bin/env bash
# Seed an inbox with a few items, then run triage-inbox until it drains. Each
# cycle: scan -> classify (agent) -> route -> file/escalate -> wait. The wait
# paces the loop with a real (journaled) sleep between cycles.
#
# Runs under the default sandbox config (tool states get their own network
# namespace on strict). On a hardened-only host a tool state is refused under the
# default `sandbox.network = "session"`; agent6 prints the opt-in to apply.
#
# Usage:  bash bench/machines/triage-inbox/run.sh [workdir]
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AGENT6="$(cd "$HERE/../../.." && pwd)/.venv/bin/agent6"
WORK="${1:-/tmp/agent6-machine-triage}"

rm -rf "$WORK"; mkdir -p "$WORK/scripts" "$WORK/inbox"
cp "$HERE/triage-inbox.asm.toml" "$WORK/"
cp "$HERE/scripts/"*.py "$WORK/scripts/"
cat > "$WORK/inbox/01-outage.txt"  <<'EOF'
PagerDuty: production API returning 503 for 40% of requests, on-call paged.
EOF
cat > "$WORK/inbox/02-newsletter.txt" <<'EOF'
Weekly product newsletter: new themes, a roundup of blog posts, and a survey.
EOF
cat > "$WORK/inbox/03-invoice.txt" <<'EOF'
Reminder: your cloud invoice for last month is now available in the portal.
EOF
cat > "$WORK/inbox/04-spam.txt" <<'EOF'
CONGRATULATIONS!!! You have been selected to claim a $1000 gift card. Click now!
EOF
git -C "$WORK" init -q
git -C "$WORK" -c user.email=b@b -c user.name=b add -A
git -C "$WORK" -c user.email=b@b -c user.name=b commit -q -m "seed inbox"

export XDG_STATE_HOME="$WORK/.agent6-state"

echo "== running triage-inbox (drains $(ls "$WORK/inbox" | wc -l) items) =="
time (cd "$WORK" && "$AGENT6" machine run triage-inbox.asm.toml)
echo
echo "== where each item landed =="
find "$WORK/processed" -type f | sort
echo
echo "== status =="
(cd "$WORK" && "$AGENT6" machine status triage-inbox)
