#!/usr/bin/env bash
# Run the code-fixer machine against a fresh copy of the seeded buggy repo.
#
# The machine bundle (code-fixer.asm.toml + scripts/) and the buggy source
# (seed/stats.py) are copied into a throwaway git repo so each run starts from
# the same failing state and the agent's edits never touch this checkout.
#
# Runs under the default sandbox config: on strict, every jailed command (the
# tool states, and whatever the agent runs) gets the run's own network. A host
# that supports only hardened has no network namespace, so the default
# `sandbox.network = "auto"` degrades with a warning there; an explicit
# `session` refuses instead. Nothing here touches your global config.
#
# Usage:  bash bench/machines/code-fixer/run.sh [workdir]
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AGENT6="$(cd "$HERE/../../.." && pwd)/.venv/bin/agent6"
WORK="${1:-/tmp/agent6-machine-code-fixer}"

rm -rf "$WORK" "${WORK}-state" "${WORK}-check"; mkdir -p "$WORK/scripts"
cp "$HERE/code-fixer.asm.toml" "$HERE/ruff.toml" "$WORK/"
cp "$HERE/scripts/"*.py "$WORK/scripts/"
cp "$HERE/seed/stats.py" "$WORK/"
git -C "$WORK" init -q
git -C "$WORK" -c user.email=bench@bench -c user.name=bench add -A
git -C "$WORK" -c user.email=bench@bench -c user.name=bench commit -q -m "seed: buggy median"

# Keep all agent6 state (per-repo config + machine journal) beside the
# workspace, hermetic per run; a state dir inside it is refused (jailed
# commands could read transcripts, and commits would stage them).
export XDG_STATE_HOME="${WORK}-state"
# The mode="run" agent commits its fix. This host has no git identity at all, so
# give agent6 one to commit under (resolved on the host, exported into the
# confined agent which can't read ~/.gitconfig). A real repo with local or
# global git identity needs none of this.
(cd "$WORK" && "$AGENT6" config set git.commit.name "agent6 code-fixer" --repo >/dev/null)
(cd "$WORK" && "$AGENT6" config set git.commit.email "code-fixer@agent6.local" --repo >/dev/null)

echo "== before: verify reports failing =="
(cd "$WORK" && python3 scripts/verify.py)
echo
echo "== running code-fixer machine =="
# --auto-approve: the agent state runs the checker itself, and nothing is
# attached to answer approvals in a throwaway repo.
(cd "$WORK" && "$AGENT6" machine run code-fixer.asm.toml --auto-approve)
echo
echo "== after: verify the machine's branch (the checkout never moves) =="
rm -rf "${WORK}-check"
git clone -q -b agent6/machine-code-fixer "$WORK" "${WORK}-check"
(cd "${WORK}-check" && python3 scripts/verify.py)
echo
echo "== agent's fix (on agent6/machine-code-fixer) =="
git -C "$WORK" --no-pager diff HEAD..agent6/machine-code-fixer -- stats.py
git -C "$WORK" --no-pager log --oneline -5 agent6/machine-code-fixer
echo
echo "== machine status =="
(cd "$WORK" && "$AGENT6" machine status code-fixer)
