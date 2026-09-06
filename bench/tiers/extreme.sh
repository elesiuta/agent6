#!/usr/bin/env bash
# Task fixtures for `bash bench/run_tier.sh extreme`; the runner supplies the
# machinery and sources this file. Not runnable on its own.

DEFAULT_BENCH_ROOT=/tmp/agent6-bench-extreme

tier_toml() {
  cat <<'EOF'
[agent6]
config_version = 1

[providers.anthropic]
api_format = "anthropic"
api_key_env = "ANTHROPIC_API_KEY"
prompt_caching = true

[models.worker]
provider = "anthropic"
model = "claude-sonnet-4-5"

[models.reviewer]
provider = "anthropic"
model = "claude-sonnet-4-5"

[sandbox]
isolation = "auto"
network = "session"
run_commands = "yes"
protect_git = true

[git]
dirty_tree = "ask"
branch_per_run = true

[workflow]
verify_command = ["python3", "-m", "unittest", "-v"]

[budget]
max_usd = 10.0
max_tokens_fallback = 1500000
EOF
}

tier_gitignore() {
  cat <<'GIT'
.agent6/
__pycache__/
result.json
final_pytest.txt
GIT
}

tier_agents_md() {
  cat <<'AG'
# Agent guide for this benchmark task

Synthetic benchmark repo. Treat TASK.md as the full spec.

- Python 3.12+. Built-in generics. Tests use stdlib `unittest`.
- Verify command: `python3 -m unittest -v`.
- Sandbox has ONLY the Python stdlib (no pytest, no third-party libs).
- Do not modify the test file unless TASK.md explicitly says so.
- Do not add new dependencies.
AG
}

TASKS=(
  "setup_task14 14-lru-eviction"
  "setup_task15 15-race-counter"
  "setup_task16 16-vending-fsm"
  "setup_task17 17-perf-longest-run"
)

setup_task14() {
  local dir="$BENCH_ROOT/14-lru-eviction"
  init_repo "$dir"
  cat > "$dir/lru.py" <<'EOF'
"""LRU cache. Convention: the OrderedDict orders entries from least-recently
used (front) to most-recently used (back). Eviction removes from the front.

Both `get` and `set` must mark the touched key as most-recently used.
"""
from __future__ import annotations
from collections import OrderedDict
from typing import Any

_MISSING = object()

class LRU:
    def __init__(self, capacity: int) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be > 0")
        self.capacity = capacity
        self._data: OrderedDict[str, Any] = OrderedDict()

    def get(self, key: str, default: Any = None) -> Any:
        if key not in self._data:
            return default
        # BUG: does not touch recency.
        return self._data[key]

    def set(self, key: str, value: Any) -> None:
        if key in self._data:
            self._data[key] = value
            return
        if len(self._data) >= self.capacity:
            # BUG: popitem() with no arg removes from the END (most-recent),
            # not the front. Should be popitem(last=False).
            self._data.popitem()
        self._data[key] = value

    def __len__(self) -> int:
        return len(self._data)

    def keys(self) -> list[str]:
        """Return keys in LRU→MRU order (front is least-recently used)."""
        return list(self._data.keys())
EOF
  cat > "$dir/test_lru.py" <<'EOF'
import unittest
from lru import LRU

class T(unittest.TestCase):
    def test_basic_set_get(self):
        c = LRU(2)
        c.set("a", 1); c.set("b", 2)
        self.assertEqual(c.get("a"), 1)
        self.assertEqual(c.get("b"), 2)
        self.assertEqual(c.get("missing", -1), -1)

    def test_eviction_evicts_least_recent(self):
        c = LRU(2)
        c.set("a", 1)
        c.set("b", 2)
        # touch 'a' so 'b' is now LRU
        self.assertEqual(c.get("a"), 1)
        c.set("c", 3)            # should evict 'b', not 'a'
        self.assertIsNone(c.get("b"))
        self.assertEqual(c.get("a"), 1)
        self.assertEqual(c.get("c"), 3)

    def test_set_overwrites_in_place(self):
        c = LRU(2)
        c.set("a", 1); c.set("b", 2)
        c.set("a", 99)           # overwrite, treated as recent
        c.set("c", 3)            # evict LRU which is now 'b'
        self.assertIsNone(c.get("b"))
        self.assertEqual(c.get("a"), 99)
        self.assertEqual(c.get("c"), 3)

    def test_get_promotes_to_mru(self):
        c = LRU(3)
        c.set("a", 1); c.set("b", 2); c.set("c", 3)
        c.get("a")               # 'a' becomes MRU, order: b, c, a
        self.assertEqual(c.keys(), ["b", "c", "a"])

    def test_capacity_one(self):
        c = LRU(1)
        c.set("a", 1)
        c.set("b", 2)
        self.assertIsNone(c.get("a"))
        self.assertEqual(c.get("b"), 2)

    def test_invalid_capacity(self):
        with self.assertRaises(ValueError):
            LRU(0)

if __name__ == "__main__":
    unittest.main()
EOF
  cat > "$dir/TASK.md" <<'EOF'
`lru.py` implements an LRU cache using `collections.OrderedDict`. The
intended convention is documented in the module docstring: entries are
ordered front (least-recently used) → back (most-recently used). Eviction
removes from the front when the cache is full.

There are TWO bugs:

1. `get` returns the value but does not move the touched entry to the
   most-recently-used position. Use `OrderedDict.move_to_end(key)` so that
   reads update recency.

2. `set` evicts the WRONG end when the cache is full. `popitem()` with no
   argument removes from the back (most-recently used). It must remove the
   least-recently used entry, i.e. `self._data.popitem(last=False)`.

Fix both bugs. Do not change the public API, the class name, or the
docstring conventions. Do not modify `test_lru.py`.

The verify command `python3 -m unittest -v` must report all 6 tests passing.
EOF
  ( cd "$dir" && git add -A && git commit -q -m "initial: LRU with eviction + recency bugs" )
  echo "$dir"
}

# --- task 15: concurrency race in shared counter -----------------------------
# `counter.py` has a plain `self.value += 1` with no lock; the test runs
# many threads that each increment many times and checks the total. The
# correct fix is to wrap the read-modify-write in self._lock.
setup_task15() {
  local dir="$BENCH_ROOT/15-race-counter"
  init_repo "$dir"
  cat > "$dir/counter.py" <<'EOF'
"""Thread-safe counter — except it isn't. `increment` is not actually
atomic: `self.value += 1` is read-modify-write and races under contention.
"""
from __future__ import annotations
import threading

class Counter:
    def __init__(self) -> None:
        self.value = 0
        self._lock = threading.Lock()

    def increment(self, by: int = 1) -> None:
        # BUG: not protected by the lock.
        self.value += by

    def snapshot(self) -> int:
        with self._lock:
            return self.value
EOF
  cat > "$dir/test_counter.py" <<'EOF'
import threading
import unittest
from counter import Counter

class T(unittest.TestCase):
    def test_single_threaded(self):
        c = Counter()
        for _ in range(1000):
            c.increment()
        self.assertEqual(c.snapshot(), 1000)

    def test_concurrent_increment_total(self):
        c = Counter()
        N_THREADS = 16
        PER_THREAD = 5000
        def worker():
            for _ in range(PER_THREAD):
                c.increment()
        threads = [threading.Thread(target=worker) for _ in range(N_THREADS)]
        for t in threads: t.start()
        for t in threads: t.join()
        self.assertEqual(c.snapshot(), N_THREADS * PER_THREAD)

    def test_concurrent_repeated(self):
        # Run the concurrent scenario several times; if any iteration loses
        # an update we'll see it.
        for _ in range(5):
            c = Counter()
            def worker():
                for _ in range(2000):
                    c.increment(2)
            ts = [threading.Thread(target=worker) for _ in range(8)]
            for t in ts: t.start()
            for t in ts: t.join()
            self.assertEqual(c.snapshot(), 8 * 2000 * 2)

if __name__ == "__main__":
    unittest.main()
EOF
  cat > "$dir/TASK.md" <<'EOF'
`counter.py` defines `Counter.increment`, which is supposed to be
thread-safe but isn't: `self.value += by` is a read-modify-write that
races under concurrent calls. There IS a `self._lock` already; it just
isn't being used in `increment`.

Fix `Counter.increment` so that concurrent calls cannot lose updates.
The fix is to wrap the read-modify-write in `with self._lock:`. Do not
change the public API or the `snapshot` method. Do not modify
`test_counter.py`.

The verify command `python3 -m unittest -v` must report all 3 tests passing.
Note that without the fix `test_concurrent_increment_total` will
intermittently undercount, and `test_concurrent_repeated` will almost
certainly fail at least once across its 5 trials.
EOF
  ( cd "$dir" && git add -A && git commit -q -m "initial: counter with unprotected increment race" )
  echo "$dir"
}

# --- task 16: vending machine state machine ----------------------------------
# Stub returns a constant; agent must implement a 4-state machine that
# accepts coins, dispenses, refunds, and handles invalid transitions.
setup_task16() {
  local dir="$BENCH_ROOT/16-vending-fsm"
  init_repo "$dir"
  cat > "$dir/vending.py" <<'EOF'
"""Vending machine state machine.

States: IDLE → ACCEPTING → READY → DISPENSED.

Public API:
    VendingMachine(price: int)
    insert(coin: int)       # adds to inserted_total
    select()                # only valid when inserted_total >= price
    refund()                # always allowed; resets to IDLE (see table)
    state                   # one of "IDLE", "ACCEPTING", "READY", "DISPENSED"
    inserted_total          # int, >= 0
    dispensed_count         # int, >= 0

Transition rules (return value of each call is the NEW state):

- IDLE:
    insert(c): coin must be > 0; new state ACCEPTING (or READY if c >= price).
    select(): raises InvalidTransition.
    refund(): no-op, stays IDLE, returns IDLE.
- ACCEPTING:
    insert(c): coin > 0; stay ACCEPTING, or move to READY if total >= price.
    select(): raises InvalidTransition (not enough money yet).
    refund(): resets inserted_total to 0, returns IDLE.
- READY:
    insert(c): allowed, stay READY (overpayment ok).
    select(): subtracts price from inserted_total, dispensed_count += 1,
              returns DISPENSED. (Any change remains as inserted_total.)
    refund(): allowed; returns remaining inserted_total to 0, state IDLE.
- DISPENSED:
    insert(c): raises InvalidTransition.
    select(): raises InvalidTransition.
    refund(): refunds any leftover, returns IDLE.

Coin amounts must be positive ints; otherwise raise ValueError.
"""
from __future__ import annotations

class InvalidTransition(Exception):
    pass

class VendingMachine:
    def __init__(self, price: int) -> None:
        if price <= 0:
            raise ValueError("price must be > 0")
        self.price = price
        self.state = "IDLE"
        self.inserted_total = 0
        self.dispensed_count = 0

    def insert(self, coin: int) -> str:
        # STUB
        return self.state

    def select(self) -> str:
        # STUB
        return self.state

    def refund(self) -> str:
        # STUB
        return self.state
EOF
  cat > "$dir/test_vending.py" <<'EOF'
import unittest
from vending import VendingMachine, InvalidTransition

class T(unittest.TestCase):
    def test_happy_path(self):
        v = VendingMachine(price=100)
        self.assertEqual(v.state, "IDLE")
        self.assertEqual(v.insert(25), "ACCEPTING")
        self.assertEqual(v.insert(50), "ACCEPTING")
        self.assertEqual(v.insert(25), "READY")
        self.assertEqual(v.select(), "DISPENSED")
        self.assertEqual(v.dispensed_count, 1)
        self.assertEqual(v.inserted_total, 0)

    def test_overpay(self):
        v = VendingMachine(price=100)
        v.insert(150)
        self.assertEqual(v.state, "READY")
        v.select()
        self.assertEqual(v.inserted_total, 50)
        self.assertEqual(v.state, "DISPENSED")

    def test_extra_insert_in_ready(self):
        v = VendingMachine(price=100)
        v.insert(100)
        self.assertEqual(v.state, "READY")
        v.insert(25)
        self.assertEqual(v.state, "READY")
        self.assertEqual(v.inserted_total, 125)

    def test_refund_from_accepting(self):
        v = VendingMachine(price=100)
        v.insert(40)
        self.assertEqual(v.refund(), "IDLE")
        self.assertEqual(v.inserted_total, 0)

    def test_refund_from_ready(self):
        v = VendingMachine(price=100)
        v.insert(150)
        self.assertEqual(v.refund(), "IDLE")
        self.assertEqual(v.inserted_total, 0)

    def test_refund_from_idle_is_noop(self):
        v = VendingMachine(price=100)
        self.assertEqual(v.refund(), "IDLE")

    def test_refund_after_dispense_clears_change(self):
        v = VendingMachine(price=100)
        v.insert(150)
        v.select()
        self.assertEqual(v.state, "DISPENSED")
        self.assertEqual(v.inserted_total, 50)
        self.assertEqual(v.refund(), "IDLE")
        self.assertEqual(v.inserted_total, 0)

    def test_select_in_idle_raises(self):
        v = VendingMachine(price=100)
        with self.assertRaises(InvalidTransition):
            v.select()

    def test_select_in_accepting_raises(self):
        v = VendingMachine(price=100)
        v.insert(50)
        with self.assertRaises(InvalidTransition):
            v.select()

    def test_insert_after_dispense_raises(self):
        v = VendingMachine(price=100)
        v.insert(100); v.select()
        with self.assertRaises(InvalidTransition):
            v.insert(25)

    def test_select_after_dispense_raises(self):
        v = VendingMachine(price=100)
        v.insert(100); v.select()
        with self.assertRaises(InvalidTransition):
            v.select()

    def test_invalid_coin(self):
        v = VendingMachine(price=100)
        with self.assertRaises(ValueError):
            v.insert(0)
        with self.assertRaises(ValueError):
            v.insert(-5)

    def test_invalid_price(self):
        with self.assertRaises(ValueError):
            VendingMachine(price=0)

if __name__ == "__main__":
    unittest.main()
EOF
  cat > "$dir/TASK.md" <<'EOF'
Implement the vending-machine state machine described in the docstring at
the top of `vending.py`. The three methods `insert`, `select`, and
`refund` are currently stubs that just return the current state.

Read the docstring transition table carefully; it specifies the exact
behaviour for every (state, action) pair, including which combinations
must raise `InvalidTransition`. The state names are exactly "IDLE",
"ACCEPTING", "READY", "DISPENSED" (strings).

Constraints:

- Do not modify `test_vending.py`.
- Do not modify the docstring or the public attribute names
  (`state`, `inserted_total`, `dispensed_count`, `price`).
- Coin and price validation already lives at the boundaries: coins must be
  positive ints (ValueError otherwise), and price must be > 0
  (ValueError otherwise) — keep these.
- All transitions must update `state` to one of the four documented values
  and return the new state.

The verify command `python3 -m unittest -v` must report all 13 tests passing.
EOF
  ( cd "$dir" && git add -A && git commit -q -m "initial: vending FSM stubs" )
  echo "$dir"
}

# --- task 17: performance constraint -----------------------------------------
# Find the length of the longest run of strictly increasing values. The stub
# is O(n^2). A test inserts a 200k-element input and asserts wall-time
# under a deadline that the O(n^2) cannot meet (it will take many seconds).
# The correct fix is the classic O(n) single-pass.
setup_task17() {
  local dir="$BENCH_ROOT/17-perf-longest-run"
  init_repo "$dir"
  cat > "$dir/longest_run.py" <<'EOF'
"""Find the length of the longest strictly-increasing contiguous run in
a list of integers.

Example:
    longest_increasing_run([1, 2, 1, 2, 3, 4, 0, 5]) == 4
                                  ^^^^^^^^^^^^
The current implementation is O(n^2) and is too slow for the large input
in test_perf.
"""
from __future__ import annotations

def longest_increasing_run(xs: list[int]) -> int:
    n = len(xs)
    best = 0 if n == 0 else 1
    # BUG: O(n^2). Restart from each index, extend while strictly increasing.
    for i in range(n):
        run = 1
        for j in range(i + 1, n):
            if xs[j] > xs[j - 1]:
                run += 1
            else:
                break
        if run > best:
            best = run
    return best
EOF
  cat > "$dir/test_longest_run.py" <<'EOF'
import time
import unittest
from longest_run import longest_increasing_run

class T(unittest.TestCase):
    def test_empty(self):
        self.assertEqual(longest_increasing_run([]), 0)

    def test_single(self):
        self.assertEqual(longest_increasing_run([7]), 1)

    def test_all_equal(self):
        self.assertEqual(longest_increasing_run([3, 3, 3, 3]), 1)

    def test_strictly_increasing(self):
        self.assertEqual(longest_increasing_run([1, 2, 3, 4, 5]), 5)

    def test_middle_run(self):
        self.assertEqual(longest_increasing_run([1, 2, 1, 2, 3, 4, 0, 5]), 4)

    def test_strict_only(self):
        # equal values break the run
        self.assertEqual(longest_increasing_run([1, 2, 2, 3, 4]), 3)

    def test_large_input_under_deadline(self):
        # Sawtooth of period 100 over 200k elements. Longest run = 100.
        # An O(n^2) restart-at-each-index will be roughly 100 * 200_000 = 2e7
        # inner steps but Python overhead pushes it into many seconds; the
        # deadline of 0.5s rules it out. O(n) finishes in a few hundredths.
        xs: list[int] = []
        for _ in range(2000):
            xs.extend(range(100))
        self.assertEqual(len(xs), 200_000)
        t0 = time.perf_counter()
        result = longest_increasing_run(xs)
        elapsed = time.perf_counter() - t0
        self.assertEqual(result, 100)
        self.assertLess(elapsed, 0.5,
                        f"too slow: {elapsed:.3f}s on 200k input")

if __name__ == "__main__":
    unittest.main()
EOF
  cat > "$dir/TASK.md" <<'EOF'
`longest_increasing_run` in `longest_run.py` returns the correct answer
but is O(n^2): for each index it restarts a forward scan. The test
`test_large_input_under_deadline` calls it on a 200,000-element sawtooth
list and requires the call to finish in under 0.5 seconds. The current
implementation cannot meet that deadline.

Rewrite `longest_increasing_run` to be O(n) — a single pass that keeps a
running `run` length, resetting to 1 whenever `xs[i] <= xs[i-1]`, and
tracking the maximum seen.

Constraints:

- Keep the function signature `(xs: list[int]) -> int`.
- Keep the empty-list contract: return 0 for `[]`.
- "Strictly increasing" means `xs[i] > xs[i-1]`; equal values break the run.
- Do not modify `test_longest_run.py`.

The verify command `python3 -m unittest -v` must report all 7 tests passing,
including the deadline-bounded `test_large_input_under_deadline`.
EOF
  ( cd "$dir" && git add -A && git commit -q -m "initial: O(n^2) longest_increasing_run" )
  echo "$dir"
}

# --- runner -------------------------------------------------------------------
