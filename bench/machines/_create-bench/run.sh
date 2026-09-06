#!/usr/bin/env bash
# Bench `agent6 machine create` across several providers/models on one fixed
# task: a poll -> classify -> act loop. Records attempts, spend, wall time, and
# whether the drafted bundle passes `machine check` + `machine test`.
#
# Each model runs in its own throwaway repo with an isolated XDG_STATE_HOME,
# its worker model pinned via per-repo config. Provider keys come from the
# global secrets store. Generated bundles are kept under <model-slug>/.
#
# Usage:  bash bench/machines/_create-bench/run.sh
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AGENT6="$(cd "$HERE/../../.." && pwd)/.venv/bin/agent6"
RESULTS="$HERE/results.jsonl"
: > "$RESULTS"

TASK="Poll a JSON status endpoint every 10 minutes with a helper script that \
prints the current status. When the status is \"degraded\", have an LLM classify \
the severity as low, medium, or high. On high severity, run a script that \
appends an alert line to a log file. Otherwise keep polling."

MODELS=(
  "openrouter|moonshotai/kimi-k2.6"
  "anthropic|claude-sonnet-4-6"
  "openrouter|z-ai/glm-5.2"
  "openrouter|openai/gpt-oss-120b"
  "openrouter|deepseek/deepseek-v3.2"
  "anthropic|claude-haiku-4-5"
)

for entry in "${MODELS[@]}"; do
  provider="${entry%%|*}"; model="${entry##*|}"
  slug="$(echo "$model" | tr '/.:' '___')"
  dir="$HERE/$slug"
  rm -rf "$dir"; mkdir -p "$dir"
  git -C "$dir" init -q
  git -C "$dir" -c user.email=b@b -c user.name=b commit -q --allow-empty -m init
  export XDG_STATE_HOME="$dir/.state"
  ( cd "$dir" && "$AGENT6" config set models.worker.provider "$provider" --repo >/dev/null \
      && "$AGENT6" config set models.worker.model "$model" --repo >/dev/null )

  echo "== creating with $provider / $model =="
  log="$dir/create.log"
  start=$(date +%s)
  ( cd "$dir" && timeout 900 "$AGENT6" machine create "$TASK" --max-attempts 4 \
      -o created.asm.toml ) >"$log" 2>&1
  code=$?
  secs=$(( $(date +%s) - start ))

  attempts=$(grep -cE "machine create: attempt" "$log" 2>/dev/null || true)
  attempts="${attempts:-0}"
  spent=$(grep -oE "spent ~\\\$[0-9.]+" "$log" 2>/dev/null | grep -oE "[0-9.]+" | tail -1)
  check=fail; testok=fail; scripts=0
  if [ -f "$dir/created.asm.toml" ]; then
    ( cd "$dir" && "$AGENT6" machine check created.asm.toml >/dev/null 2>&1 ) && check=ok
    ( cd "$dir" && "$AGENT6" machine test  created.asm.toml >/dev/null 2>&1 ) && testok=ok
    scripts=$(ls "$dir/scripts" 2>/dev/null | wc -l | tr -d ' ')
  fi
  printf '{"provider":"%s","model":"%s","exit":%d,"attempts":%s,"spent_usd":%s,"wall_secs":%d,"scripts":%s,"check":"%s","test":"%s"}\n' \
    "$provider" "$model" "$code" "${attempts:-0}" "${spent:-0}" "$secs" "${scripts:-0}" "$check" "$testok" \
    | tee -a "$RESULTS"
done

echo; echo "== summary =="
python3 - "$RESULTS" <<'PY'
import json, sys
rows = [json.loads(l) for l in open(sys.argv[1])]
for r in rows:
    print(f"  {r['model']:32} attempts={r['attempts']} spent=${r['spent_usd']:<6} "
          f"wall={r['wall_secs']}s scripts={r['scripts']} check={r['check']} test={r['test']}")
ok = [r for r in rows if r['check']=='ok' and r['test']=='ok']
print(f"  passed check+test: {len(ok)}/{len(rows)}")
PY
