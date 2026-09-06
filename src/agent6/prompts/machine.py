# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Machine-authoring grammar reference.

The complete `.asm.toml` grammar + worked example handed to the model in
`agent6 machine create`. Pure text; `machine.authoring` owns assembling the
per-attempt prompt around it.
"""

from __future__ import annotations

MACHINE_AUTHOR_GUIDE = """\
# agent6 state-machine (.asm.toml) authoring guide

A machine is a small, deterministic program whose building blocks are
sandboxed tool calls, agent6 agent runs, timed waits, and branches. It is
plain TOML. Author one complete machine per task.

## File skeleton

    machine = "kebab-or-snake-name"   # ^[a-z][a-z0-9_-]*$
    version = 1                       # always 1
    initial = "<state name>"          # the entry state

    [budget]
    max_usd = 1.0                     # optional cap on metered spend (see below)
    max_transitions = 100             # > 0; hard cap on state hops

  `max_usd` (optional, > 0) caps the machine's cumulative METERED spend
  (provider-reported cost, else price x tokens). A state whose model has
  no price data is bounded by the operator config's
  `[budget].max_tokens_fallback` instead, so the machine runs on any
  model. `max_transitions` is the always-binding runaway guard.

## The blackboard: three owner tables (write-authorization, one read namespace)

Variables live under exactly one owner table. The owner controls who may
WRITE; reads are a single flat namespace (refer to a variable by its BARE
name everywhere, never `vars.code.x`).

    [vars.operator]                   # constants; READ-ONLY at runtime
    threshold = { type = "int", value = 3 }

    [vars.code]                       # written only by `tool` states
    items = { type = "list[str]", default = [] }

    [vars.agent]                      # written only by `agent` states
    verdict = { type = "verdict", default = {} }

Rules:
  - Names are globally unique across the three tables. Identifiers match
    `^[a-z][a-z0-9_]*$`.
  - Reserved names (cannot be used): vars, operator, code, agent, result.
  - operator vars use `value = ...`; code/agent vars use `default = ...`.
  - A record-typed var's default is the empty table `{}`.

## Types and schemas

Field types: `str`, `int`, `float`, `bool`, `list[<scalar>]`, `json`
(opaque, not navigable), or the name of a `[schemas.*]` record (recursive).

    [schemas.verdict]
    approved = "bool"
    note = "str"
    # optional field:           reason = { type = "str", optional = true }
    # string enum:              level  = { type = "str", enum = ["low", "high"] }

To navigate `result.field` or `somevar.field` you MUST give it a record
type via a schema. Opaque `json` cannot be dotted.

## States

Each state is `[states.<name>]` with a `kind`. Names match the identifier
grammar. Terminal states end the machine.

### tool: run a sandboxed command
    [states.scan]
    kind = "tool"
    command = ["scan", "{{ threshold }}"]   # argv; see templating below
    output_schema = "scan_result"            # optional: types `result` so result.x works
    capture = { set = { items = "{{ result.items }}" } }   # writes [vars.code] only
    timeout_secs = 5
    on = { ok = "check", nonzero = "stop_fail", timeout = "stop_fail" }

  tool labels are exactly: ok, nonzero, timeout.

### agent: run a nested agent6 loop
    [states.review]
    kind = "agent"
    # model defaults to "inherit" (the operator's effective worker model).
    # OMIT it unless you must pin a specific one, a hardcoded model the
    # operator hasn't configured passes `machine check` but dies at run time.
    prompt = "Review the change and return a verdict."
    output_schema = "verdict"                # finish_session payload validated against this
    capture = { finish_json = "verdict" }    # whole payload -> a [vars.agent] var
    # or: capture = { set = { total = "{{ result.points }}" } }  # one field
    timeout_secs = 600
    on = { ok = "route", failed = "stop_fail", budget_exhausted = "halt", timeout = "expired" }

  agent labels are exactly: ok, failed, budget_exhausted, timeout.
  An agent state may write ONLY [vars.agent] vars.
  By default an agent state is a READ-ONLY structured-output judge (classify /
  score / decide -> a finish_session result; it cannot edit the repo). For a state
  that must do real coding work, add `mode = "run"` to give it the full edit /
  verify / commit tool set.

### branch: route on predicates (MUST be total)
    [states.check]
    kind = "branch"
    when = [
      { if = "len(items) == 0", goto = "stop_ok" },
      { else = true, goto = "record" },        # final `else` is REQUIRED
    ]

  Predicate allow-list: comparisons (== != < <= > >=), `and`/`or`/`not`,
  `in`, `len(x)`, `has(x)` (optional-field presence), record navigation
  `x.field`, and literals. NO arithmetic, no calls beyond those two, no
  method calls or comprehensions: compute in the state that writes the
  var, then branch on the stored value.

### wait: pause until an instant or a poke
    [states.poll]
    kind = "wait"
    every_secs = "{{ interval }}"   # OR  until = "2026-01-01T00:00:00Z"
    on = { tick = "scan", signal = "scan" }

  wait labels are exactly: tick, signal. `cron` is NOT supported by the v1
  runtime — use `every_secs` or `until`.

### terminal: stop
    [states.stop_ok]
    kind = "terminal"
    status = "ok"        # "ok" or "failed"
    reason = "done"

## Templating

`{{ ref }}` interpolates a variable; `{{ ref | len }}` / `{{ ref | json }}`
are the only two filters (both zero-arg). In an argv list, an element that
is EXACTLY `"{{ listvar }}"` splices the list into N arguments. In a
`capture.set`, a lone filter-less `{{ ref }}` captures the native VALUE
(its type must match the target var); any other template renders to a
string (target must be `str`).

## Capture ownership wall
  - `tool`  states may write only `[vars.code]`.
  - `agent` states may write only `[vars.agent]`.
  - `[vars.operator]` is read-only; `branch`/`wait`/`terminal` never write.

## Accumulating records across iterations (e.g. a paper-trade log)

Two durable stores, both available at every isolation level:

  - The JOURNAL (`machines/<id>/journal.jsonl` under the per-repo state dir) already records every
    transition with its fact (each `tool` stdout, each `agent` payload), so it
    IS a replayable audit log of everything that happened — for free.
  - `$AGENT6_MACHINE_DATA_DIR` is a persistent, WRITABLE directory every `tool`
    script may write to (the one writable spot under `hardened`, where new
    top-level files in the workspace are read-only). It's an absolute path valid
    inside the tool jail, so just open it — works at every isolation level:
    `open(os.environ["AGENT6_MACHINE_DATA_DIR"] + "/trades.jsonl", "a")`.

For values you branch on or template later, capture them into the blackboard:
keep counters / latest values (the blackboard has NO `list[record]` type —
`list[<scalar>]` elements must be scalars; use a `json` var for an opaque blob).
Don't try to grow an unbounded list of records IN the blackboard — write it to
the data dir (or rely on the journal) and keep just a count/latest in a var.

## Real inputs, secrets & the network

Author scripts that do the REAL task in production — not ones that read canned
data. So:
  - Read live inputs from their real source. For HTTP the standard library is
    enough: `urllib.request` makes real API calls — you do NOT need `requests`.
  - Pass NON-secret config (an endpoint, a user id) as an operator var spliced
    into `command`, e.g. `["python3", "scripts/fetch.py", "{{ feed_url }}"]`.
  - A jailed tool sees only `LANG`, `LC_ALL`, `TERM`, `CI`, `PATH`, `HOME`,
    `PYTHONDONTWRITEBYTECODE`, `UV_NO_SYNC` and `AGENT6_MACHINE_DATA_DIR`: the
    operator's exported secrets do NOT reach it.
    Never hard-code a secret in a script or the `.asm.toml`; a state that needs
    one says so in its comment, so the operator arranges it.
  - A tool that touches the network MUST set `network = "host"` on its
    state; without it the tool is network-isolated and the call fails. (The
    operator still has to permit egress via `sandbox.network`; if their
    config blocks it, `agent6 machine run` explains the one-line fix and offers
    to apply it.)
  - Persist outputs/state to `$AGENT6_MACHINE_DATA_DIR`: under `hardened` a
    tool jail cannot create new top-level entries in the workspace, and the
    machine's own bundle is read-only at every level.

## Test every non-trivial script (offline simulation)

Anything a script does that ISN'T a pure function of its argv/stdin is a SEAM
you must be able to control in a test: a network call, the clock
(`time`/`datetime.now`), randomness, reading an external file, a subprocess.
For each script that has a seam, ALSO emit `scripts/<name>_test.py` that:
  1. imports the script,
  2. replaces each seam with a fake — mock the network function, inject a fixed
     time, point file I/O at a `tempfile` dir,
  3. asserts the script's contract (the exact JSON it prints / the file it writes),
  4. exits 0 on success, non-zero on failure, using NO network.

Structure each script so its core logic is a small function the test calls
directly and `main()` only does argv/env/stdout — that makes the seam easy to
patch. `machine create` and `machine test` LINT (ruff), TYPE-CHECK (ty), and run
these tests in a no-network jail: a script that isn't typed, isn't lint-clean,
or whose test needs the real network fails the gate. Default to `python3` + the
standard library; you may name another available interpreter in `command`
(`["bash", "scripts/x.sh"]`), but Python is validated most thoroughly — reach
for a third-party package only when stdlib genuinely can't do it, and say in the
state's comment that the operator must install it in the jail's environment.

## Fault tolerance (machines run for days; transient failures are NORMAL)

A long-running machine WILL hit a down API, a 429, a timeout. Those surface as
labels, and your wiring decides whether one bad tick kills the machine:
  - tool failures: `nonzero` (script printed an error and exited non-zero) and
    `timeout`. Route both BACK to the wait state so the machine simply retries
    next tick: `on = { ok = ..., nonzero = "wait_tick", timeout = "wait_tick" }`.
  - agent failures: `failed` (provider error, malformed output) and `timeout`
    likewise route back to the wait state. Reserve a terminal state for
    `budget_exhausted` only — that one will not heal by retrying.
  - Never route a failure label to an `status = "ok"` terminal, and never wire
    a failure straight back to the SAME fetch state (that is a hot retry loop
    with no delay; going through the wait state is the backoff).
  - Scripts should exit non-zero on failure (stderr message) rather than
    printing fabricated JSON, so a bad tick is a visible `nonzero`, not silent
    garbage captured into the blackboard.

## Worked example: watch a live price feed, record a threshold crossing

A complete, valid, PRODUCTION machine: `fetch_price` makes a real HTTP call
(hence `network = "host"`); `record_buy` appends to the data dir. Each
script ships a `*_test.py` that mocks its seam so the whole machine simulates
offline.

    machine = "price-watch"
    version = 1
    initial = "wait_tick"

    [budget]
    max_usd = 1.0
    max_transitions = 1000

    [vars.operator]
    interval_secs = { type = "int", value = 60 }
    feed_url = { type = "str", value = "https://api.example.com/v1/price/BTC" }
    threshold = { type = "float", value = 100000.0 }

    [vars.code]
    price = { type = "float", default = 0.0 }
    recorded = { type = "int", default = 0 }

    [schemas.price_result]
    price = "float"

    [schemas.record_result]
    recorded = "int"

    [states.wait_tick]
    kind = "wait"
    every_secs = "{{ interval_secs }}"
    on = { tick = "fetch_price", signal = "fetch_price" }

    [states.fetch_price]
    kind = "tool"
    command = ["python3", "scripts/fetch_price.py", "{{ feed_url }}"]
    network = "host"                 # this tool reaches the real API
    output_schema = "price_result"          # types `result` so result.price works
    capture = { set = { price = "{{ result.price }}" } }
    timeout_secs = 15
    on = { ok = "decide", nonzero = "wait_tick", timeout = "wait_tick" }

    [states.decide]
    kind = "branch"
    when = [
      { if = "price >= threshold", goto = "record_buy" },
      { else = true, goto = "wait_tick" },
    ]

    [states.record_buy]
    kind = "tool"
    command = ["python3", "scripts/record_buy.py", "{{ price }}"]
    output_schema = "record_result"
    capture = { set = { recorded = "{{ result.recorded }}" } }
    timeout_secs = 10
    on = { ok = "wait_tick", nonzero = "wait_tick", timeout = "wait_tick" }

### …and the scripts it references (write ALL of these under `scripts/`)

Real, typed scripts plus an offline test per seam (the network seam in
`fetch_price`, the filesystem seam in `record_buy`).

`scripts/fetch_price.py`:
```python
# Fetch the current price from an HTTP JSON feed. Prints {"price": <float>}.
# The feed URL is argv[1] (an operator var); an optional bearer token comes from
# the PRICE_FEED_TOKEN env var -- secrets belong in the environment, never in the
# machine file. The tool state must set network = "host".
from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request


def fetch_price(url: str, token: str) -> float:
    # GET *url* and return the "price" field as a float (the network seam).
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=10) as resp:
        payload = json.loads(resp.read())
    return float(payload["price"])


def main() -> int:
    url = sys.argv[1] if len(sys.argv) > 1 else ""
    if not url:
        print("usage: fetch_price.py <feed-url>", file=sys.stderr)
        return 1
    try:
        price = fetch_price(url, os.environ.get("PRICE_FEED_TOKEN", ""))
    except (urllib.error.URLError, OSError, KeyError, ValueError) as exc:
        print(f"fetch failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({"price": price}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

`scripts/fetch_price_test.py` — mocks the network seam so it runs offline:
```python
# Offline test for fetch_price.py: mocks the network seam (urlopen) so the
# machine validates without a live feed. Run: python3 scripts/fetch_price_test.py
from __future__ import annotations

import json
import os
import sys
import unittest.mock as mock

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import fetch_price  # noqa: E402


class _FakeResp:
    # Stand-in for the urlopen() context manager (the seam we control).
    def __init__(self, body: bytes) -> None:
        self._body = body

    def __enter__(self) -> _FakeResp:
        return self

    def __exit__(self, *_: object) -> bool:
        return False

    def read(self) -> bytes:
        return self._body


def test_parses_price() -> None:
    body = json.dumps({"price": 123.5}).encode()
    with mock.patch.object(fetch_price.urllib.request, "urlopen", return_value=_FakeResp(body)):
        assert fetch_price.fetch_price("https://feed", "") == 123.5


if __name__ == "__main__":
    test_parses_price()
    print("ok")
```

`scripts/record_buy.py`:
```python
# Append a paper trade to $AGENT6_MACHINE_DATA_DIR/trades.jsonl; print the count.
from __future__ import annotations

import json
import os
import pathlib
import sys
import time


def record(data_dir: pathlib.Path, price: float) -> int:
    # Append one trade and return the running count (the filesystem seam).
    data_dir.mkdir(parents=True, exist_ok=True)
    log = data_dir / "trades.jsonl"
    with log.open("a") as fh:
        fh.write(json.dumps({"price": price, "ts": int(time.time())}) + "\\n")
    return sum(1 for _ in log.open())


def main() -> int:
    price = float(sys.argv[1]) if len(sys.argv) > 1 else 0.0
    data_dir = pathlib.Path(os.environ.get("AGENT6_MACHINE_DATA_DIR", "."))
    print(json.dumps({"recorded": record(data_dir, price)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

`scripts/record_buy_test.py` — a tempfile dir replaces the data-dir seam:
```python
# Offline test for record_buy.py: a temp dir replaces the data-dir seam.
from __future__ import annotations

import json
import os
import pathlib
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import record_buy  # noqa: E402


def test_appends_and_counts() -> None:
    with tempfile.TemporaryDirectory() as d:
        data = pathlib.Path(d)
        assert record_buy.record(data, 10.0) == 1
        assert record_buy.record(data, 20.0) == 2
        rows = (data / "trades.jsonl").read_text().splitlines()
        assert json.loads(rows[0])["price"] == 10.0


if __name__ == "__main__":
    test_appends_and_counts()
    print("ok")
```

## Common mistakes (each fails `machine check`/`create` or silently misbehaves)
  - Hardcoding `model = "..."` on an `agent` state — omit it (defaults to
    "inherit" = the worker model) unless pinning one on purpose.
  - A `tool` that calls the network but forgets `network = "host"` — it
    runs network-isolated and the call fails.
  - Hardcoding a secret/token in a script or the `.asm.toml` — a jailed tool
    sees no operator secret at all; say what the state needs in its comment.
  - A bare `{{ listvar }}` inside an `agent` prompt or any other string —
    a list only interpolates as `{{ listvar | json }}` (or spliced as a
    standalone argv element). Bare list references fail `machine check`.
  - A test that hits the REAL network — tests run with NO network and must
    mock the seam, or they fail the gate. A script with no `*_test.py` is not
    caught by the gate, so write one for every seam.
  - A script that isn't typed or isn't lint-clean — `machine create` runs ruff +
    ty and rejects it. Annotate functions; keep imports used.
  - `list[record]` / `list[{...}]` types — unsupported. Append records to a
    file from a tool script and keep a counter in the blackboard (see above).
  - A `tool` script that exits non-zero or times out — capture does not fire
    and the blackboard keeps its previous values. Exit 0 with empty or
    non-conforming stdout HALTS the machine: print one JSON object matching
    your `output_schema` on stdout and exit 0. stderr is free for diagnostics.
  - A `tool` script that writes outside `$AGENT6_MACHINE_DATA_DIR` — on
    `hardened` the rest of the workspace is read-only to tool jails (new
    top-level files are denied). Write to the data dir (or print to stdout).
  - A non-total `branch` — the last clause MUST be `{ else = true, goto = ... }`.

## Validity requirements (the file must pass `machine check`)
  - Every `on`/`goto`/`initial` target names an existing state.
  - Every state is reachable from `initial`.
  - Every `branch` is total (ends with `{ else = true, goto = ... }`).
  - Every reference resolves to a declared variable of a compatible type.
  - Every `capture` writes a var owned by the writing state kind.
"""
