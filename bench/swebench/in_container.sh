#!/usr/bin/env bash
# Runs INSIDE a pulled SWE-bench instance container. Installs agent6 (uv-managed
# Python 3.14 + the mounted wheel, no Rust toolchain needed), points it at the
# worker model, runs it on the issue text in /testbed, and writes the resulting
# git diff to /out/patch.diff for SWE-bench's evaluator. The container is the
# isolation boundary; agent6 runs hardened inside it (no privileged, no userns),
# with the repo's conda env granted read+exec via sandbox.extra_read_paths.
set -uo pipefail
export HOME=/root
export PATH="/root/.local/bin:$PATH"
# Force a UTF-8 locale: the SWE-bench images default to ASCII (C/POSIX), so the
# conda python 3.6 launcher below would UnicodeEncodeError when the issue text
# contains a non-ASCII char (e.g. a zero-width space) passed as a subprocess
# argv -- crashing BEFORE agent6 starts and silently yielding an empty patch.
export LC_ALL=C.UTF-8 LANG=C.UTF-8
export PYTHONUTF8=1
export AGENT6_STATE_HOME=/root/a6state   # keep agent6's run state OUT of /testbed
export AGENT6_FORCE_STREAM=1             # OpenRouter SSE heartbeat-safe path
export AGENT6_ALLOW_ROOT=1               # SWE-bench images run as root; the container IS the boundary

MODEL="${AGENT6_SB_MODEL:?set AGENT6_SB_MODEL}"
MAX_USD="${AGENT6_SB_MAX_USD:-3.0}"
TIMEOUT_S="${AGENT6_SB_TIMEOUT:-1500}"
# exact wheel chosen by the orchestrator; fallback: newest by version, never
# lexicographic-first (a stale old wheel sorts first and its config schema
# rejects current keys)
WHL="/mnt/wheel/${AGENT6_SB_WHEEL:-$(basename "$(ls /mnt/wheel/*.whl | sort -V | tail -1)")}"

# agent6 prices Anthropic via its OpenRouter alias (models/pricing.py), so
# every model here is METERED: max_usd is the bound. max_tokens_fallback only
# binds calls the meter cannot price (a pricing regression), so it is a loose
# backstop, not a per-model computation.

uv python install 3.14 >/dev/null 2>&1
uv tool install --python 3.14 "$WHL" >/dev/null 2>&1

# The worker's provider is chosen from the model slug (claude-* -> Anthropic,
# else OpenRouter); BOTH provider blocks are always written so review seats
# may reference either (a cross-model seat needs the other provider too).
# An unused block is inert; keys resolve from the mounted secrets.
if [[ "$MODEL" == claude-* ]]; then
  PROVIDER=anthropic
elif [[ "$MODEL" == gpt-* ]]; then
  PROVIDER=chatgpt
else
  PROVIDER=openrouter
fi
PROVIDER_BLOCK='[providers.anthropic]
api_format = "anthropic"
api_key_env = "ANTHROPIC_API_KEY"
prompt_caching = true

[providers.chatgpt]
api_format = "chatgpt"

[providers.openrouter]
api_format = "openai"
api_key_env = "OPENROUTER_API_KEY"
base_url = "https://openrouter.ai/api/v1"
extra_headers = { "HTTP-Referer" = "https://github.com/agent6-dev/agent6", "X-Title" = "agent6-swebench" }'


# Optional review panel (Fugu dimension). AGENT6_SB_REVIEW_SEATS is a
# semicolon-separated list of "persona@provider/model" seats; when set the panel
# reviews before finish_session and gates per AGENT6_SB_REVIEW_DECISION (default
# quorum). Same-model vs distinct-model panels are just different seat lists.
REVIEW_LINES=""
if [ -n "${AGENT6_SB_REVIEW_SEATS:-}" ]; then
  ARR=""
  IFS=';' read -ra _SEATS <<< "$AGENT6_SB_REVIEW_SEATS"
  for s in "${_SEATS[@]}"; do ARR="${ARR}\"${s}\", "; done
  REVIEW_LINES="[review]
trigger = \"before_finish\"
decision = \"${AGENT6_SB_REVIEW_DECISION:-quorum}\"
quorum = ${AGENT6_SB_REVIEW_QUORUM:-2}
tier = \"diff\"
seats = [${ARR%, }]"
fi

# Verify command. The jail forces child PATH=/usr/bin:/bin, so a bare `python3`
# won't resolve the container's conda interpreter; use its ABSOLUTE path (exec is
# granted via sandbox.extra_read_paths). Auto-detect django's runner; default to
# pytest. Override with AGENT6_SB_VERIFY (space-separated argv) for odd repos.
# SWE-bench images root conda at /opt/miniconda3; SWE-rebench at /opt/conda.
CONDA_PY=$(ls /opt/miniconda3/envs/*/bin/python /opt/conda/envs/*/bin/python 2>/dev/null | head -1)
CONDA_PY="${CONDA_PY:-python3}"
# AGENT6_SB_VERIFY=none: a PINNED-GATELESS arm; the model self-verifies with
# targeted run_command tests. verify_infer = false is required: with only an
# unset verify_command, mid-run adoption armed the inferred `python3 -m
# pytest` whose interpreter lacks pytest, an always-red 0.0s no-op gate.
# Needs a wheel that carries the knob (0.0.27 rejects unknown keys).
if [ "${AGENT6_SB_VERIFY:-}" = "none" ]; then
  AGENT6_SB_VERIFY=""
  VERIFY_TOML="verify_infer = false"
elif [ -z "${AGENT6_SB_VERIFY:-}" ]; then
  if [ -f /testbed/tests/runtests.py ]; then
    AGENT6_SB_VERIFY="$CONDA_PY tests/runtests.py --verbosity 1 --parallel 2"
  elif "$CONDA_PY" -m pytest --version >/dev/null 2>&1; then
    # -x: first failure is the signal; a green pass still runs the suite.
    AGENT6_SB_VERIFY="$CONDA_PY -m pytest -q -x"
  elif [ -x /testbed/bin/test ]; then
    # A repo that ships its own top-level test runner (and whose env lacks
    # pytest); use it rather than a dead `pytest` that runs no tests and lets
    # a wrong patch pass unchecked.
    AGENT6_SB_VERIFY="$CONDA_PY bin/test"
  else
    echo "[in_container] WARNING: pytest absent and no ./bin/test; verify may not run tests" >&2
    AGENT6_SB_VERIFY="$CONDA_PY -m pytest -q"
  fi
fi
if [ -n "${AGENT6_SB_VERIFY:-}" ]; then
  VARR=""
  for w in $AGENT6_SB_VERIFY; do VARR="${VARR}\"${w}\", "; done
  VERIFY_TOML="verify_command = [${VARR%, }]"
fi

# A/B arm: the orchestrator mounts a replacement run-mode base prompt.
PROMPT_FILE_LINE=""
if [ -f /mnt/system_prompt.txt ]; then
  PROMPT_FILE_LINE='system_prompt_file = "/mnt/system_prompt.txt"'
fi

cat > /root/agent6.toml <<EOF
[agent6]
config_version = 1

$PROVIDER_BLOCK

[models.worker]
provider = "$PROVIDER"
model = "$MODEL"
${AGENT6_SB_EFFORT:+effort = \"$AGENT6_SB_EFFORT\"}

[models.reviewer]
provider = "$PROVIDER"
model = "$MODEL"

[sandbox]
# UNSANDBOXED: the container is the isolation. agent6's jail fights the container
# here (couldn't exec the conda interpreter under hardened/strict), so the config
# opts out of the kernel sandbox entirely: the standard SWE-bench setup, Docker as
# the boundary. profile="none" is self-authorizing (an operator-only config
# value); the per-invocation forms are --dangerously-disable-sandbox /
# AGENT6_DANGEROUSLY_DISABLE_SANDBOX=1.
isolation = "none"
network = "session"
run_commands = "yes"
protect_git = false
extra_read_paths = ["/opt/miniconda3", "/opt/conda"]

[git]
require_clean_worktree = true
auto_stash = false
branch_per_run = false

[workflow]
$VERIFY_TOML
${AGENT6_SB_VERIFY_WHEN:+verify_when = \"$AGENT6_SB_VERIFY_WHEN\"}
# Half a 1200s run died in ONE 600s full-suite verify; fast signal wins.
verify_timeout_s = 240

[prompt]
revise_prompt = "off"
structural_priors = ${AGENT6_SB_STRUCTURAL_PRIORS:-true}
$PROMPT_FILE_LINE

$REVIEW_LINES

[budget]
max_usd = $MAX_USD
max_tokens_fallback = 2000000
max_percent = ${AGENT6_SB_MAX_PERCENT:-30}
EOF

cd /testbed
git config user.email "swebench@agent6" 2>/dev/null
git config user.name "agent6" 2>/dev/null

# Bench scaffolding, not product prompt: AGENTS.md is the operator-owned
# channel for behavioural guidance and repo-priors ingests it. Famous-repo
# issues tempt the model into recalling the upstream fix; steer to
# derivation. Committed BEFORE the base capture: part of base state, so it
# never enters the prediction diff and the worktree stays clean (untracked
# scaffolding tripped require_clean_worktree and refused every run).
if [ ! -f AGENTS.md ]; then
  cat > AGENTS.md <<'AEOF'
If the task matches a known public issue, still derive the fix from
this checkout: never spend turns recalling or fetching the canonical
upstream commit. Anything remembered about the upstream fix is an
unverified hint, not a source.
AEOF
  git add AGENTS.md && git commit -q -m "bench scaffolding" 2>/dev/null
fi
BASE=$(git rev-parse HEAD)

# Pass the (long, special-char-laden) issue text as a single argv via Python so
# no shell quoting can corrupt it.
AGENT6_SB_TIMEOUT="$TIMEOUT_S" "$CONDA_PY" - <<'PYEOF'
import os, subprocess
problem = open("/mnt/problem.txt", encoding="utf-8").read()
try:
    subprocess.run(
        ["agent6", "--config", "/root/agent6.toml", "run", problem],
        cwd="/testbed", timeout=float(os.environ.get("AGENT6_SB_TIMEOUT", "1500")),
    )
except subprocess.TimeoutExpired:
    print("[in_container] agent6 run timed out")
PYEOF

# The model's patch = changes to files TRACKED at the base commit. `git add -u`
# (not -A) deliberately ignores untracked files, so a build/test the agent ran
# that generated artifacts (sklearn dumped 2000 .txt files) does not pollute
# the patch. SWE-bench gold patches edit existing source; a genuinely new
# source file is rare and not worth capturing thousands of build outputs for.
mkdir -p /out
git -C /testbed add -u
git -C /testbed diff --cached "$BASE" -- . ':(exclude).agent6' ':(exclude)agent6.toml' \
    > /out/patch.diff
echo "[in_container] patch lines: $(wc -l < /out/patch.diff)"

# Export the run's agent6 state (logs.jsonl + provider transcripts) so
# tool-call failures are diagnosable after the container is gone (observed:
# kimi-k2.7 malformed-JSON grep args, undiagnosable from run.log alone).
STATE_DIR="${AGENT6_STATE_HOME:-${XDG_STATE_HOME:-/root/.local/state}/agent6}"
if [ -d "$STATE_DIR" ]; then
  mkdir -p /out/state
  cp -r "$STATE_DIR"/. /out/state/ 2>/dev/null || true
fi

# True per-run cost: sum the provider's billed `cost` from every transcript.
# The run.log TOTAL line is absent when the run times out, which undercounted
# a 50-instance sweep by ~$3.7; this number survives regardless of how the
# run ended.
"$CONDA_PY" - <<'PYEOF' > /out/cost.json 2>/dev/null || true
import glob, json
total, calls = 0.0, 0
for t in glob.glob("/out/state/*/sessions/runs/*/transcripts/*.json"):
    try:
        d = json.load(open(t))
    except Exception:
        continue
    r = d.get("response")
    body = r.get("body") if isinstance(r, dict) else None
    u = body.get("usage") if isinstance(body, dict) else None
    if isinstance(u, dict):
        total += float(u.get("cost") or 0.0)
        calls += 1
print(json.dumps({"billed_usd": round(total, 4), "calls": calls}))
PYEOF
