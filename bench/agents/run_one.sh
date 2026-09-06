#!/usr/bin/env bash
# Run one agent on one task. Usage: run_one.sh <task> <agent>
# Tasks: go-logwindow | rust-ratelimit | go-kvstore-debug
# Agents: agent6 | aider | opencode | claude
set -u
TASK="$1"; AGENT="$2"
ROOT="${AGENTBENCH_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
TPL="$ROOT/tasks/$TASK"
RESULTS="${AGENTBENCH_RUNS:-$HOME/agentbench-runs}/results.jsonl"
PROMPT="Read README.md and implement what it specifies so that ./verify.sh passes (it runs the test suite). Do not modify the test files or verify.sh. Iterate until the tests pass."
TIMEOUT=1500
# Model selection (2026-07-11): parametrized so one harness sweeps models.
OR_MODEL="${AGENTBENCH_OR_MODEL:-moonshotai/kimi-k2.6}"
CLAUDE_MODEL="${AGENTBENCH_CLAUDE_MODEL:-claude-haiku-4-5}"
# agent6 rows: pin the worker to a provider/model instead of the global config.
A6_PROVIDER="${AGENTBENCH_AGENT6_PROVIDER:-}"
A6_MODEL="${AGENTBENCH_AGENT6_MODEL:-}"
# The run dir is keyed by (task, agent, MODEL): two chains sweeping different
# models in parallel must never share a working directory (observed: the
# anthropic and openrouter chains clobbering each other's agent6 runs).
case "$AGENT" in
  claude) EFF_MODEL="$CLAUDE_MODEL" ;;
  agent6) EFF_MODEL="${A6_MODEL:-global}" ;;
  *)      EFF_MODEL="$OR_MODEL" ;;
esac
MODEL_SLUG=$(printf '%s' "$EFF_MODEL" | tr '/.:' '---')
RUN="${AGENTBENCH_RUNS:-$HOME/agentbench-runs}/${TASK}__${AGENT}__${MODEL_SLUG}"

export PATH="$HOME/.local/bin:$HOME/.opencode/bin:/usr/bin:$PATH"
OPENROUTER_API_KEY=$(python3 -c 'import tomllib,pathlib;print(tomllib.loads((pathlib.Path.home()/".config/agent6/secrets.toml").read_text())["providers"]["openrouter"]["api_key"])')
ANTHROPIC_API_KEY=$(python3 -c 'import tomllib,pathlib;print(tomllib.loads((pathlib.Path.home()/".config/agent6/secrets.toml").read_text())["providers"]["anthropic"]["api_key"])')

or_usage() {
  curl -s https://openrouter.ai/api/v1/key -H "Authorization: Bearer $OPENROUTER_API_KEY" \
    | python3 -c 'import json,sys; print(json.load(sys.stdin)["data"]["usage"])' 2>/dev/null || echo 0
}

rm -rf "$RUN"; mkdir -p "$(dirname "$RUN")"
cp -r "$TPL" "$RUN"
cd "$RUN"
# every agent gets the same environment: a fresh git repo seeded with the task
git init -q
git config user.email bench@bench; git config user.name bench
git add -A && git commit -qm "task seed"

case "$TASK" in
  go-logwindow) IMPL=logwindow.go ;;
  rust-ratelimit) IMPL=src/lib.rs ;;
  go-kvstore-debug) IMPL=kvstore.go ;;
esac

BEFORE=$(or_usage)
START=$(date +%s)
STATUS=0
case "$AGENT" in
  agent6)
    # agent6 does not read in-repo config; pass an explicit --config file
    # (kept OUTSIDE the run repo so the clean-worktree gate stays green)
    A6_CFG="$(dirname "$RUN")/$(basename "$RUN")_config.toml"
    cat > "$A6_CFG" <<CFG
[workflow]
verify_command = ["bash", "verify.sh"]
[sandbox]
run_commands = "yes"
[git]
dirty_tree = "include"
CFG
    if [ -n "$A6_MODEL" ]; then
      for role in worker planner reviewer; do
        printf '[models.%s]\nprovider = "%s"\nmodel = "%s"\n' "$role" "$A6_PROVIDER" "$A6_MODEL" >> "$A6_CFG"
      done
    fi
    BUDGET_FLAGS="--max-usd 0.60"
    if [ "$A6_PROVIDER" = "anthropic" ]; then
      # anthropic is unpriced for agent6; cap the token ledger (as bench/coreagent)
      BUDGET_FLAGS="--max-tokens-fallback 4400000"
    fi
    AGENT6_FORCE_STREAM=1 timeout $TIMEOUT "${AGENT6_BIN:-agent6}" run "$PROMPT" --config "$A6_CFG" $BUDGET_FLAGS > agent.log 2>&1
    STATUS=$?
    ;;
  aider)
    OPENROUTER_API_KEY="$OPENROUTER_API_KEY" timeout $TIMEOUT aider \
      --model "openrouter/$OR_MODEL" --yes-always --no-check-update \
      --auto-test --test-cmd ./verify.sh --message "$PROMPT" "$IMPL" > agent.log 2>&1
    STATUS=$?
    ;;
  opencode)
    OPENROUTER_API_KEY="$OPENROUTER_API_KEY" timeout $TIMEOUT opencode run \
      -m "openrouter/$OR_MODEL" "$PROMPT" > agent.log 2>&1
    STATUS=$?
    ;;
  claude)
    export CLAUDE_CONFIG_DIR="$ROOT/claude-cfg"
    mkdir -p "$CLAUDE_CONFIG_DIR"
    ANTHROPIC_API_KEY="$ANTHROPIC_API_KEY" timeout $TIMEOUT claude -p "$PROMPT" \
      --model "$CLAUDE_MODEL" --dangerously-skip-permissions \
      --output-format json > agent.log 2> agent.err
    STATUS=$?
    ;;
esac
WALL=$(( $(date +%s) - START ))
sleep 12
AFTER=$(or_usage)

# independent verdict
./verify.sh > verdict.log 2>&1
PASS=$([ $? -eq 0 ] && echo true || echo false)
# test-file tamper check vs the template's committed copy
TAMPER=false
case "$TASK" in
  go-logwindow) TESTF=logwindow_test.go ;;
  rust-ratelimit) TESTF=tests/ratelimit_test.rs ;;
  go-kvstore-debug) TESTF=kvstore_test.go ;;
esac
if ! cmp -s "$TPL/$TESTF" "$RUN/$TESTF"; then TAMPER=true; fi
if ! cmp -s "$TPL/verify.sh" "$RUN/verify.sh"; then TAMPER=true; fi

COST=$(python3 -c "print(round($AFTER - $BEFORE, 4))" 2>/dev/null || echo "?")
CLAUDE_COST=""
if [ "$AGENT" = "claude" ]; then
  CLAUDE_COST=$(python3 -c 'import json;print(round(json.load(open("agent.log")).get("total_cost_usd",0),4))' 2>/dev/null || echo "?")
  COST=$CLAUDE_COST
fi
MODEL="$EFF_MODEL"
printf '{"task":"%s","agent":"%s","model":"%s","pass":%s,"tampered":%s,"wall_secs":%d,"cost_usd":%s,"exit":%d}\n' \
  "$TASK" "$AGENT" "$MODEL" "$PASS" "$TAMPER" "$WALL" "${COST:-0}" "$STATUS" | tee -a "$RESULTS"
