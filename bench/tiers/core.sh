#!/usr/bin/env bash
# Task fixtures for `bash bench/run_tier.sh core`; the runner supplies the
# machinery and sources this file. Not runnable on its own.

DEFAULT_BENCH_ROOT=/tmp/agent6-bench

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

This is a synthetic benchmark repository. Treat the task in `TASK.md` as the
full specification. Convention:

- Python 3.12+. Use built-in generics (`list[str]`, `dict[str, int]`).
- Tests live next to source as `test_*.py` and use stdlib `unittest`.
- The verify command is `python3 -m unittest -v`; it must pass for the task
  to be considered complete.
- Do not add new dependencies. The sandbox only has the Python stdlib;
  there is no pytest, no third-party libraries.
- Do not modify the test file unless the task text explicitly says so.
AG
}

TASKS=(
  "setup_task1 01-bugfix-factorial"
  "setup_task2 02-add-cli-flag"
  "setup_task3 03-refactor-dedupe"
  "setup_task4 04-type-annotations"
  "setup_task5 05-fix-deprecation"
  "setup_task6 06-add-subcommand"
  "setup_task7 07-add-logging"
  "setup_task8 08-extract-method"
)

# --- task 1: bug-fix ----------------------------------------------------------
setup_task1() {
  local dir="$BENCH_ROOT/01-bugfix-factorial"
  init_repo "$dir"
  cat > "$dir/calc.py" <<'EOF'
def factorial(n: int) -> int:
    if n < 0:
        raise ValueError("n must be >= 0")
    result = 1
    for i in range(1, n):  # bug: should be range(1, n+1)
        result *= i
    return result
EOF
  cat > "$dir/test_calc.py" <<'EOF'
import unittest
from calc import factorial

class T(unittest.TestCase):
    def test_zero(self): self.assertEqual(factorial(0), 1)
    def test_one(self): self.assertEqual(factorial(1), 1)
    def test_five(self): self.assertEqual(factorial(5), 120)
    def test_negative(self):
        with self.assertRaises(ValueError):
            factorial(-1)

if __name__ == "__main__":
    unittest.main()
EOF
  cat > "$dir/TASK.md" <<'EOF'
The four `unittest` tests in `test_calc.py` currently fail because `factorial`
in `calc.py` has a single off-by-one bug. Identify the off-by-one bug in the
`for` loop of `factorial` and fix it (and only it) so that `python3 -m unittest`
reports all four tests passing. Do not modify `test_calc.py`.
EOF
  ( cd "$dir" && git add -A && git commit -q -m "initial: failing factorial test suite" )
  echo "$dir"
}

# --- task 2: add a typed CLI flag --------------------------------------------
setup_task2() {
  local dir="$BENCH_ROOT/02-add-cli-flag"
  init_repo "$dir"
  cat > "$dir/printer.py" <<'EOF'
"""CLI that prints lines from a file, optionally numbered."""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="printer")
    p.add_argument("path", type=Path)
    p.add_argument("-n", "--numbered", action="store_true",
                   help="prefix each line with its 1-based line number")
    return p

def render(lines: list[str], numbered: bool) -> list[str]:
    if numbered:
        return [f"{i+1}: {line}" for i, line in enumerate(lines)]
    return list(lines)

def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    text = args.path.read_text().splitlines()
    for line in render(text, args.numbered):
        print(line)
    return 0

if __name__ == "__main__":
    sys.exit(main())
EOF
  cat > "$dir/test_printer.py" <<'EOF'
import unittest
from printer import render

class T(unittest.TestCase):
    def test_plain(self): self.assertEqual(render(["a", "b"], False), ["a", "b"])
    def test_numbered(self): self.assertEqual(render(["a", "b"], True), ["1: a", "2: b"])

if __name__ == "__main__":
    unittest.main()
EOF
  cat > "$dir/TASK.md" <<'EOF'
Extend `printer.py` with a new `--reverse` / `-r` boolean flag.

Requirements (all unambiguous — do not ask clarifying questions, implement
exactly as written):

1. Add `--reverse` / `-r` (action="store_true") to `build_parser()`.
2. Change `render(lines, numbered, reverse)` to take a third positional bool
   argument `reverse`. When `reverse` is True, each line's characters are
   reversed BEFORE numbering (if numbered is also True).
3. Update `main()` to pass `args.reverse` to `render`.
4. Add two `unittest.TestCase` methods to `test_printer.py`:
     - `test_reverse_plain` — `render(["ab", "cd"], False, True) == ["ba", "dc"]`
     - `test_reverse_numbered` — `render(["ab", "cd"], True, True) == ["1: ba", "2: dc"]`
5. Update the existing `test_plain` and `test_numbered` to pass `False` as
   the new third `reverse` argument so they keep passing.

The verify command `python3 -m unittest -v` must report 4 tests passing
when you are done.
EOF
  ( cd "$dir" && git add -A && git commit -q -m "initial: printer CLI without --reverse" )
  echo "$dir"
}

# --- task 3: deduplication refactor ------------------------------------------
setup_task3() {
  local dir="$BENCH_ROOT/03-refactor-dedupe"
  init_repo "$dir"
  cat > "$dir/users.py" <<'EOF'
"""Two endpoints that each validate a user record. Validation logic is duplicated."""
from __future__ import annotations

def create_user(name: str, email: str, age: int) -> dict:
    # validation block (copy 1)
    if not name or not name.strip():
        raise ValueError("name must be non-empty")
    if "@" not in email or email.startswith("@") or email.endswith("@"):
        raise ValueError("invalid email")
    if age < 0 or age > 150:
        raise ValueError("age out of range")
    return {"action": "create", "name": name.strip(), "email": email, "age": age}

def update_user(user_id: int, name: str, email: str, age: int) -> dict:
    # validation block (copy 2 — identical)
    if not name or not name.strip():
        raise ValueError("name must be non-empty")
    if "@" not in email or email.startswith("@") or email.endswith("@"):
        raise ValueError("invalid email")
    if age < 0 or age > 150:
        raise ValueError("age out of range")
    return {"action": "update", "id": user_id, "name": name.strip(), "email": email, "age": age}
EOF
  cat > "$dir/test_users.py" <<'EOF'
import unittest
from users import create_user, update_user

class T(unittest.TestCase):
    def test_create_ok(self): self.assertEqual(create_user("Ada", "a@b", 30)["name"], "Ada")
    def test_update_ok(self): self.assertEqual(update_user(7, "Ada", "a@b", 30)["id"], 7)
    def test_bad_name_create(self):
        with self.assertRaises(ValueError): create_user("", "a@b", 1)
    def test_bad_name_update(self):
        with self.assertRaises(ValueError): update_user(1, "", "a@b", 1)
    def test_bad_email_create(self):
        with self.assertRaises(ValueError): create_user("A", "no-at", 1)
    def test_bad_email_update(self):
        with self.assertRaises(ValueError): update_user(1, "A", "no-at", 1)

if __name__ == "__main__":
    unittest.main()
EOF
  cat > "$dir/TASK.md" <<'EOF'
Refactor `users.py` to remove the duplicated validation block. All existing
behaviour and tests must remain unchanged.

Do exactly this:

1. Add a module-level (NOT a method) private function
   `_validate_user(name: str, email: str, age: int) -> None` to `users.py`.
2. Move the three validation `if` statements from `create_user` and
   `update_user` into `_validate_user`. It must raise `ValueError` with the
   same messages ("name must be non-empty", "invalid email", "age out of
   range") in the same order.
3. Replace the duplicated blocks in both `create_user` and `update_user`
   with a single call to `_validate_user(name, email, age)`.
4. Append one new `unittest.TestCase` method to `test_users.py` named
   `test_helper_raises_value_error_on_empty_name` that does
   `with self.assertRaises(ValueError): _validate_user("", "a@b", 1)`.
   Add the import: `from users import _validate_user`.

The verify command `python3 -m unittest -v` must keep reporting all tests
passing.
EOF
  ( cd "$dir" && git add -A && git commit -q -m "initial: duplicated validation" )
  echo "$dir"
}

# --- task 4: add type annotations --------------------------------------------
setup_task4() {
  local dir="$BENCH_ROOT/04-type-annotations"
  init_repo "$dir"
  cat > "$dir/strutil.py" <<'EOF'
"""Small string utilities. Currently untyped."""
def split_csv(s):
    return [p.strip() for p in s.split(",") if p.strip()]

def join_with(parts, sep):
    return sep.join(parts)

def index_of(haystack, needle):
    try:
        return haystack.index(needle)
    except ValueError:
        return -1

def count_chars(text):
    counts = {}
    for ch in text:
        counts[ch] = counts.get(ch, 0) + 1
    return counts
EOF
  cat > "$dir/test_strutil.py" <<'EOF'
import unittest
from strutil import split_csv, join_with, index_of, count_chars

class T(unittest.TestCase):
    def test_split(self): self.assertEqual(split_csv("a, b , ,c"), ["a", "b", "c"])
    def test_join(self): self.assertEqual(join_with(["a", "b"], "-"), "a-b")
    def test_index_hit(self): self.assertEqual(index_of("hello", "l"), 2)
    def test_index_miss(self): self.assertEqual(index_of("hello", "z"), -1)
    def test_count(self): self.assertEqual(count_chars("aab"), {"a": 2, "b": 1})

if __name__ == "__main__":
    unittest.main()
EOF
  cat > "$dir/TASK.md" <<'EOF'
Add complete type annotations to every FUNCTION in `strutil.py` (function
parameters and return types only; do NOT annotate module-level variables or
add new classes/attributes). Functions that return `None` should be annotated
as `-> None` explicitly.

Requirements (all unambiguous):

1. Add `from __future__ import annotations` at the top of `strutil.py`.
2. Annotate every function with:
     - `split_csv(s: str) -> list[str]`
     - `join_with(parts: list[str], sep: str) -> str`
     - `index_of(haystack: str, needle: str) -> int`
     - `count_chars(text: str) -> dict[str, int]`
3. Do not change any function body.
4. Append a new `unittest.TestCase` method to `test_strutil.py` named
   `test_annotations_present` that does
   `self.assertEqual(split_csv.__annotations__["return"], "list[str]")`.

The verify command `python3 -m unittest -v` must report all tests passing.
EOF
  ( cd "$dir" && git add -A && git commit -q -m "initial: untyped utilities" )
  echo "$dir"
}

# --- task 5: replace deprecated datetime.utcnow ------------------------------
setup_task5() {
  local dir="$BENCH_ROOT/05-fix-deprecation"
  init_repo "$dir"
  cat > "$dir/clock.py" <<'EOF'
"""Tiny clock helper. Uses the deprecated datetime.utcnow()."""
from __future__ import annotations
from datetime import datetime

def utc_iso() -> str:
    return datetime.utcnow().isoformat()

def utc_epoch() -> float:
    return datetime.utcnow().timestamp()

def is_after(ts_iso: str) -> bool:
    return datetime.utcnow() > datetime.fromisoformat(ts_iso)
EOF
  cat > "$dir/test_clock.py" <<'EOF'
import unittest
from datetime import UTC, datetime, timedelta
from clock import utc_iso, utc_epoch, is_after

class T(unittest.TestCase):
    def test_iso_is_string(self): self.assertIsInstance(utc_iso(), str)
    def test_epoch_is_float(self): self.assertIsInstance(utc_epoch(), float)
    def test_is_after_true(self):
        past = (datetime.now(UTC) - timedelta(days=1)).isoformat()
        self.assertTrue(is_after(past))
    def test_is_after_false(self):
        future = (datetime.now(UTC) + timedelta(days=1)).isoformat()
        self.assertFalse(is_after(future))

if __name__ == "__main__":
    unittest.main()
EOF
  cat > "$dir/TASK.md" <<'EOF'
`clock.py` uses `datetime.utcnow()`, which is deprecated in Python 3.12+ and
emits `DeprecationWarning`. Replace every call to `datetime.utcnow()` in
`clock.py` with `datetime.now(UTC)` and add the corresponding import.

Requirements:

1. Add `UTC` to the existing `from datetime import datetime` line so it
   becomes `from datetime import UTC, datetime`.
2. Replace every `datetime.utcnow()` call with `datetime.now(UTC)`. There
   are three call sites; all must be updated.
3. Do not change the function names, signatures, or any other code.
4. Do not modify `test_clock.py`.

The verify command `python3 -m unittest -v` must keep reporting all four
tests passing, and `python3 -W error::DeprecationWarning -m unittest -v`
should also pass (you do not need to run this yourself — it is what a
production CI would do).
EOF
  ( cd "$dir" && git add -A && git commit -q -m "initial: uses deprecated utcnow" )
  echo "$dir"
}

# --- task 6: add CLI subcommand ---------------------------------------------
setup_task6() {
  local dir="$BENCH_ROOT/06-add-subcommand"
  init_repo "$dir"
  cat > "$dir/notes.py" <<'EOF'
"""Tiny notes CLI: currently only has 'list'."""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

STORE = Path("notes.json")

def load() -> list[str]:
    if not STORE.is_file():
        return []
    return json.loads(STORE.read_text())

def save(notes: list[str]) -> None:
    STORE.write_text(json.dumps(notes))

def cmd_list(_args: argparse.Namespace) -> int:
    for i, note in enumerate(load(), 1):
        print(f"{i}. {note}")
    return 0

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="notes")
    sub = p.add_subparsers(dest="cmd", required=True)
    p_list = sub.add_parser("list", help="list all notes")
    p_list.set_defaults(func=cmd_list)
    return p

def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)

if __name__ == "__main__":
    sys.exit(main())
EOF
  cat > "$dir/test_notes.py" <<'EOF'
import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
import notes

class T(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.cwd = os.getcwd()
        os.chdir(self.tmp.name)
        notes.STORE = Path("notes.json")
    def tearDown(self):
        os.chdir(self.cwd); self.tmp.cleanup()
    def test_list_empty(self):
        buf = io.StringIO()
        with redirect_stdout(buf):
            self.assertEqual(notes.main(["list"]), 0)
        self.assertEqual(buf.getvalue(), "")
    def test_list_one(self):
        Path("notes.json").write_text(json.dumps(["hello"]))
        buf = io.StringIO()
        with redirect_stdout(buf):
            self.assertEqual(notes.main(["list"]), 0)
        self.assertEqual(buf.getvalue(), "1. hello\n")

if __name__ == "__main__":
    unittest.main()
EOF
  cat > "$dir/TASK.md" <<'EOF'
Extend `notes.py` with two new subcommands and add tests for them.

Functional requirements:

1. Add an `add` subcommand: `notes add TEXT` appends `TEXT` (one positional
   string argument, required) to the notes list and saves. Implement as
   `cmd_add(args: argparse.Namespace) -> int` returning 0.
2. Add a `delete` subcommand: `notes delete N` removes the 1-based Nth note
   (one positional int argument, required) and saves. Returns 0 on success,
   or 2 if N is out of range (also print `error: index N out of range` to
   stderr in that case).
3. Wire both subcommands into `build_parser()` using `set_defaults(func=...)`
   so that `main()` continues to work as written (no changes to `main()`).

Test requirements (add to `test_notes.py`, do not modify existing tests):

4. `test_add_then_list`: call `notes.main(["add", "hello"])`, then
   `notes.main(["add", "world"])`, then check that `load()` returns
   `["hello", "world"]`.
5. `test_delete_middle`: write `["a", "b", "c"]` to the store, call
   `notes.main(["delete", "2"])`, then assert `load() == ["a", "c"]`.
6. `test_delete_out_of_range`: write `["only"]` to the store, call
   `notes.main(["delete", "5"])` and assert it returns 2.

The verify command `python3 -m unittest -v` must report 5 tests passing.
EOF
  ( cd "$dir" && git add -A && git commit -q -m "initial: notes CLI with only list" )
  echo "$dir"
}

# --- task 7: stdlib logging instrumentation ----------------------------------
setup_task7() {
  local dir="$BENCH_ROOT/07-add-logging"
  init_repo "$dir"
  cat > "$dir/cache.py" <<'EOF'
"""Tiny in-memory TTL cache. Currently logs nothing."""
from __future__ import annotations
import time

class TTLCache:
    def __init__(self, ttl_seconds: float) -> None:
        self.ttl = ttl_seconds
        self._data: dict[str, tuple[float, object]] = {}

    def set(self, key: str, value: object) -> None:
        self._data[key] = (time.monotonic() + self.ttl, value)

    def get(self, key: str) -> object | None:
        entry = self._data.get(key)
        if entry is None:
            return None
        expiry, value = entry
        if time.monotonic() >= expiry:
            del self._data[key]
            return None
        return value
EOF
  cat > "$dir/test_cache.py" <<'EOF'
import logging
import unittest
from cache import TTLCache

class T(unittest.TestCase):
    def test_set_get(self):
        c = TTLCache(60.0)
        c.set("k", 1)
        self.assertEqual(c.get("k"), 1)
    def test_get_missing(self):
        self.assertIsNone(TTLCache(60.0).get("nope"))

if __name__ == "__main__":
    unittest.main()
EOF
  cat > "$dir/TASK.md" <<'EOF'
Instrument `cache.py` with stdlib `logging`. Concrete requirements:

1. At the top of `cache.py` add `import logging` and `logger = logging.getLogger(__name__)`.
2. In `set()`, log `logger.debug("cache set key=%s ttl=%s", key, self.ttl)`.
3. In `get()`:
   - On a miss (entry is None), log `logger.debug("cache miss key=%s", key)`
     and return None as before.
   - On an expiry path (entry exists but has expired), log
     `logger.info("cache expired key=%s", key)` BEFORE deleting and
     returning None.
   - On a hit, log `logger.debug("cache hit key=%s", key)` and return the value.
4. Do not change any return values or raise any new exceptions.
5. Add a new test method `test_expiry_logs_info` to `test_cache.py` that:
   - Uses `self.assertLogs("cache", level="INFO")` as a context manager.
   - Inside it: create `TTLCache(0.0)`, `set("k", "v")`, then call
     `get("k")` (which will be on the expired path because ttl is 0).
   - Asserts the captured logs contain at least one record whose message
     matches `"cache expired key=k"` (use `any("cache expired key=k" in r.getMessage() for r in cm.records)`).

The verify command `python3 -m unittest -v` must report all 3 tests passing.
EOF
  ( cd "$dir" && git add -A && git commit -q -m "initial: TTLCache without logging" )
  echo "$dir"
}

# --- task 8: extract method refactor -----------------------------------------
setup_task8() {
  local dir="$BENCH_ROOT/08-extract-method"
  init_repo "$dir"
  cat > "$dir/report.py" <<'EOF'
"""Order report. price_summary mixes I/O-shaped accumulation with formatting."""
from __future__ import annotations

def price_summary(orders: list[dict]) -> str:
    # accumulate
    total = 0.0
    count = 0
    for o in orders:
        qty = o["qty"]
        price = o["price"]
        total += qty * price
        count += qty
    # format
    if count == 0:
        return "no items"
    avg = total / count
    return f"items={count} total={total:.2f} avg={avg:.2f}"
EOF
  cat > "$dir/test_report.py" <<'EOF'
import unittest
from report import price_summary

class T(unittest.TestCase):
    def test_empty(self): self.assertEqual(price_summary([]), "no items")
    def test_one(self):
        self.assertEqual(price_summary([{"qty": 2, "price": 3.0}]),
                         "items=2 total=6.00 avg=3.00")
    def test_many(self):
        self.assertEqual(price_summary([{"qty": 2, "price": 3.0},
                                        {"qty": 3, "price": 4.0}]),
                         "items=5 total=18.00 avg=3.60")

if __name__ == "__main__":
    unittest.main()
EOF
  cat > "$dir/TASK.md" <<'EOF'
Refactor `report.py` by extracting the accumulation step into a separate
module-level helper. Do not change any observable behaviour or test output.

Specific requirements:

1. Add a module-level function
   `_accumulate(orders: list[dict]) -> tuple[float, int]`
   that returns `(total, count)` — the sum of qty*price and the sum of qty.
2. Rewrite `price_summary` to delegate the accumulation step to
   `_accumulate`. The "format" half (the `if count == 0` and `return f"..."`)
   must stay in `price_summary` exactly as written.
3. Do not modify `test_report.py`.
4. Add a new test method `test_accumulate_helper` to `test_report.py` that
   imports `_accumulate` and asserts
   `_accumulate([{"qty": 2, "price": 3.0}, {"qty": 3, "price": 4.0}]) == (18.0, 5)`.
   Add the import: `from report import _accumulate`.

The verify command `python3 -m unittest -v` must report 4 tests passing.
EOF
  ( cd "$dir" && git add -A && git commit -q -m "initial: report with inline accumulation" )
  echo "$dir"
}

# --- runner -------------------------------------------------------------------
