#!/usr/bin/env bash
# Task fixtures for `bash bench/run_tier.sh megaextreme`; the runner supplies the
# machinery and sources this file. Not runnable on its own.

DEFAULT_BENCH_ROOT=/tmp/agent6-bench-mega

tier_toml() {
  cat <<'EOF'
[agent6]
config_version = 1

[providers.anthropic]
api_format = "anthropic"
api_key_env = "ANTHROPIC_API_KEY"
prompt_caching = true

# Thinker roles use opus: decomposition and review judgement benefit from
# the stronger reasoning model on harder tasks.
[models.reviewer]
provider = "anthropic"
model = "claude-opus-4-5"

# Worker stays on sonnet — it's the better pure code generator and we
# want fast iteration on the actual edits.
[models.worker]
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
max_usd = 20.0
max_tokens_fallback = 3000000
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
- Do not modify any file matching `test_*.py` or under `tests/`.
- Do not add new dependencies.
AG
}

TASKS=(
  "setup_task18 18-http-router"
  "setup_task19 19-sql-engine"
  "setup_task20 20-multibug"
  "setup_task21 21-sync-to-async"
)

setup_task18() {
  local dir="$BENCH_ROOT/18-http-router"
  init_repo "$dir"
  cat > "$dir/router.py" <<'EOF'
"""Tiny Express-style HTTP router.

Public API:

    r = Router()
    r.add("GET", "/users", handler)
    r.add("GET", "/users/:id", handler)
    r.use("/users", middleware)            # path-scoped middleware
    r.use(global_middleware)               # global middleware (matches all)
    response = r.dispatch(method, path)

A handler is `Callable[[Request], Response]`.
A middleware is `Callable[[Request, Next], Response]` where `Next` is a
zero-arg callable that invokes the next middleware (or the matched handler).

Request is a dataclass with:
    method: str
    path: str
    params: dict[str, str]   # filled in by the router for `:name` segments

Response is a dataclass with:
    status: int
    body: str

`dispatch(method, path)`:
- finds the most specific matching route for (method, path),
- builds the middleware chain in order: globals, then path-scoped ones whose
  prefix matches `path`, then the handler,
- returns the Response produced by the chain.

Matching rules:
- Path is split on `/`; empty leading segment is dropped.
- A segment of the form `:name` matches any non-empty segment and binds it.
- Static segments must match literally.
- When two routes both match, the one with the FEWER param segments wins
  (i.e. `/users/me` beats `/users/:id`).
- Trailing slashes are ignored: `/users/` and `/users` are the same path.

Errors:
- No route matched (path/method combination): return Response(404, "not found").
- Path matched but not the method: return Response(405, "method not allowed").
- A handler/middleware raising an exception: return Response(500, "<class>: <msg>").

Stubs below — implement them.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field


@dataclass
class Request:
    method: str
    path: str
    params: dict[str, str] = field(default_factory=dict)


@dataclass
class Response:
    status: int
    body: str


Handler = Callable[[Request], Response]
Next = Callable[[], Response]
Middleware = Callable[[Request, Next], Response]


class Router:
    def __init__(self) -> None:
        # STUB: design the storage you need.
        raise NotImplementedError

    def add(self, method: str, path: str, handler: Handler) -> None:
        """Register a handler for (method, path)."""
        raise NotImplementedError

    def use(self, *args: object) -> None:
        """Register a middleware.

        Two forms:
          r.use(mw)            -> global middleware
          r.use("/prefix", mw) -> middleware that runs only when the request
                                  path starts with the given prefix (segment-wise)
        """
        raise NotImplementedError

    def dispatch(self, method: str, path: str) -> Response:
        """Run the chain and return the resulting Response."""
        raise NotImplementedError
EOF
  cat > "$dir/test_router.py" <<'EOF'
import unittest
from router import Router, Request, Response


def H(status, body):
    def _h(req):
        return Response(status, body)
    return _h


class TestBasic(unittest.TestCase):
    def test_simple_static_route(self):
        r = Router()
        r.add("GET", "/hello", H(200, "hi"))
        resp = r.dispatch("GET", "/hello")
        self.assertEqual((resp.status, resp.body), (200, "hi"))

    def test_404_no_route(self):
        r = Router()
        r.add("GET", "/hello", H(200, "hi"))
        self.assertEqual(r.dispatch("GET", "/nope").status, 404)

    def test_trailing_slash_normalized(self):
        r = Router()
        r.add("GET", "/users", H(200, "list"))
        self.assertEqual(r.dispatch("GET", "/users/").body, "list")
        self.assertEqual(r.dispatch("GET", "/users").body, "list")


class TestParams(unittest.TestCase):
    def test_single_param(self):
        r = Router()
        r.add("GET", "/users/:id", lambda req: Response(200, req.params["id"]))
        self.assertEqual(r.dispatch("GET", "/users/42").body, "42")

    def test_multiple_params(self):
        r = Router()
        def h(req):
            return Response(200, f'{req.params["uid"]}-{req.params["pid"]}')
        r.add("GET", "/users/:uid/posts/:pid", h)
        self.assertEqual(r.dispatch("GET", "/users/7/posts/99").body, "7-99")

    def test_param_does_not_match_empty_segment(self):
        r = Router()
        r.add("GET", "/users/:id", H(200, "ok"))
        self.assertEqual(r.dispatch("GET", "/users/").status, 404)


class TestPrecedence(unittest.TestCase):
    def test_static_beats_param(self):
        r = Router()
        r.add("GET", "/users/:id", H(200, "param"))
        r.add("GET", "/users/me", H(200, "static"))
        self.assertEqual(r.dispatch("GET", "/users/me").body, "static")
        self.assertEqual(r.dispatch("GET", "/users/42").body, "param")

    def test_fewer_params_wins(self):
        r = Router()
        r.add("GET", "/a/:x/:y", H(200, "two"))
        r.add("GET", "/a/:x/c", H(200, "one"))
        self.assertEqual(r.dispatch("GET", "/a/b/c").body, "one")


class TestMethods(unittest.TestCase):
    def test_405_when_path_matches_but_method_doesnt(self):
        r = Router()
        r.add("GET", "/users", H(200, "list"))
        self.assertEqual(r.dispatch("POST", "/users").status, 405)

    def test_different_methods_same_path(self):
        r = Router()
        r.add("GET", "/users", H(200, "list"))
        r.add("POST", "/users", H(201, "created"))
        self.assertEqual(r.dispatch("GET", "/users").status, 200)
        self.assertEqual(r.dispatch("POST", "/users").status, 201)


class TestMiddleware(unittest.TestCase):
    def test_global_middleware_can_short_circuit(self):
        r = Router()
        r.add("GET", "/admin", H(200, "ok"))
        def auth(req, nxt):
            return Response(401, "no")
        r.use(auth)
        self.assertEqual(r.dispatch("GET", "/admin").status, 401)

    def test_global_middleware_can_pass_through(self):
        r = Router()
        r.add("GET", "/x", H(200, "x"))
        log = []
        def mw(req, nxt):
            log.append("before")
            resp = nxt()
            log.append(f"after:{resp.status}")
            return resp
        r.use(mw)
        resp = r.dispatch("GET", "/x")
        self.assertEqual(resp.body, "x")
        self.assertEqual(log, ["before", "after:200"])

    def test_middleware_ordering(self):
        r = Router()
        r.add("GET", "/x", H(200, "x"))
        order = []
        def mk(name):
            def mw(req, nxt):
                order.append(f"{name}-in")
                resp = nxt()
                order.append(f"{name}-out")
                return resp
            return mw
        r.use(mk("a"))
        r.use(mk("b"))
        r.dispatch("GET", "/x")
        self.assertEqual(order, ["a-in", "b-in", "b-out", "a-out"])

    def test_prefix_scoped_middleware(self):
        r = Router()
        r.add("GET", "/api/users", H(200, "u"))
        r.add("GET", "/public/home", H(200, "h"))
        seen = []
        def api_mw(req, nxt):
            seen.append(req.path)
            return nxt()
        r.use("/api", api_mw)
        r.dispatch("GET", "/api/users")
        r.dispatch("GET", "/public/home")
        self.assertEqual(seen, ["/api/users"])

    def test_prefix_scoped_is_segment_aware(self):
        # /api should NOT match /apifoo (prefix is segment-wise, not raw)
        r = Router()
        r.add("GET", "/apifoo", H(200, "f"))
        called = []
        def mw(req, nxt):
            called.append(req.path)
            return nxt()
        r.use("/api", mw)
        r.dispatch("GET", "/apifoo")
        self.assertEqual(called, [])

    def test_exception_in_handler_becomes_500(self):
        r = Router()
        def boom(req):
            raise ValueError("nope")
        r.add("GET", "/boom", boom)
        resp = r.dispatch("GET", "/boom")
        self.assertEqual(resp.status, 500)
        self.assertIn("ValueError", resp.body)
        self.assertIn("nope", resp.body)


if __name__ == "__main__":
    unittest.main()
EOF
  cat > "$dir/TASK.md" <<'EOF'
Implement the `Router` class in `router.py` plus its `Request` /
`Response` dataclasses. The module docstring fully specifies the
behaviour: static + parametric routes, method matching, middleware
chains (global and path-prefix-scoped), route precedence, error
handling.

`test_router.py` contains 15 unit tests covering every behavioural
clause. Do not modify it.

Constraints:

- Single file (`router.py`). No new files.
- Stdlib only — no third-party dependencies.
- Keep the public API exactly as documented (`Router`, `Request`,
  `Response`, `Handler`, `Middleware`, `Next` type aliases, the
  `add` / `use` / `dispatch` methods).
- The middleware chain must be built lazily so each `nxt()` call
  produces the next-in-line Response (this is what the ordering test
  verifies).
- Trailing slashes are equivalent to the non-trailing form.
- Exceptions from handlers or middlewares become a 500 with
  `f"{type(exc).__name__}: {exc}"` as the body.

The verify command `python3 -m unittest -v` must report all 15 tests passing.
EOF
  ( cd "$dir" && git add -A && git commit -q -m "initial: HTTP router stubs + 15 tests" )
  echo "$dir"
}

# --- task 19: SQL-like query engine ------------------------------------------
# Parse and execute a small SELECT/WHERE/ORDER BY/LIMIT subset against a
# list[dict]. Tests cover projection, predicates, ordering, limits,
# and the parser's error handling.
setup_task19() {
  local dir="$BENCH_ROOT/19-sql-engine"
  init_repo "$dir"
  cat > "$dir/qengine.py" <<'EOF'
"""Tiny SQL-like query engine over list-of-dict rows.

Public API:
    query(rows: list[dict], q: str) -> list[dict]

Grammar (whitespace-insensitive, identifiers and keywords case-insensitive,
string literals use single quotes, numbers are int or float):

    statement   := 'SELECT' projection 'FROM' identifier (where)? (order)? (limit)?
    projection  := '*' | identifier (',' identifier)*
    where       := 'WHERE' predicate ('AND' predicate)*
    predicate   := identifier op value
    op          := '=' | '!=' | '<' | '<=' | '>' | '>='
    value       := number | string | 'NULL'
    order       := 'ORDER' 'BY' identifier ('ASC' | 'DESC')?
    limit       := 'LIMIT' integer

`FROM <ident>` is purely a label — there's only one table, the `rows`
argument. Any identifier there is accepted; it is not validated against
`rows`.

Semantics:
- Projection: `*` returns the full row dict; otherwise build a NEW dict
  containing exactly the listed keys (in the listed order). Missing keys
  in a row → the projected dict simply omits that key (do NOT raise).
- WHERE: a predicate `k op v` is True only if the row has key `k`; rows
  missing `k` are filtered out (treated as not-matching) regardless of op.
  Predicates are combined with AND (left-to-right). String compares use
  Python ordering. NULL compares: only `= NULL` and `!= NULL` are valid;
  any other op with NULL raises `QueryError`. `= NULL` matches rows where
  `k` is absent or value is None; `!= NULL` matches rows where `k` is
  present and value is not None.
- ORDER BY: stable sort, ASC by default. Rows missing the key sort LAST
  in both directions (i.e. always at the end).
- LIMIT: clamp to the first N rows after ordering. N must be >= 0.

Errors → raise `QueryError` with a short message:
- syntax errors in the query string,
- unknown operator,
- LIMIT with a non-integer or negative value,
- `=` / `!=` is the only NULL-safe pair.

Implement parser + executor.
"""
from __future__ import annotations


class QueryError(Exception):
    pass


def query(rows: list[dict], q: str) -> list[dict]:
    raise NotImplementedError  # STUB
EOF
  cat > "$dir/test_qengine.py" <<'EOF'
import unittest
from qengine import query, QueryError


ROWS = [
    {"id": 1, "name": "alice", "age": 30, "city": "NYC"},
    {"id": 2, "name": "bob",   "age": 25, "city": "LA"},
    {"id": 3, "name": "carol", "age": 35, "city": "NYC"},
    {"id": 4, "name": "dave",  "age": 28},                 # no city
    {"id": 5, "name": "eve",   "age": None, "city": "LA"},
]


class TestSelect(unittest.TestCase):
    def test_select_star(self):
        out = query(ROWS, "SELECT * FROM users")
        self.assertEqual(len(out), 5)
        self.assertEqual(out[0]["name"], "alice")

    def test_select_subset(self):
        out = query(ROWS, "SELECT name, age FROM users")
        self.assertEqual(out[0], {"name": "alice", "age": 30})
        self.assertEqual(set(out[0].keys()), {"name", "age"})

    def test_select_missing_key_skipped(self):
        out = query(ROWS, "SELECT name, city FROM users")
        # dave has no 'city' → his projection just lacks it
        self.assertEqual(out[3], {"name": "dave"})


class TestWhere(unittest.TestCase):
    def test_eq_string(self):
        out = query(ROWS, "SELECT name FROM u WHERE city = 'NYC'")
        self.assertEqual([r["name"] for r in out], ["alice", "carol"])

    def test_gt_int(self):
        out = query(ROWS, "SELECT name FROM u WHERE age > 28")
        self.assertEqual([r["name"] for r in out], ["alice", "carol"])

    def test_neq(self):
        out = query(ROWS, "SELECT name FROM u WHERE city != 'NYC'")
        self.assertEqual([r["name"] for r in out], ["bob", "eve"])

    def test_and(self):
        out = query(ROWS, "SELECT name FROM u WHERE city = 'NYC' AND age > 30")
        self.assertEqual([r["name"] for r in out], ["carol"])

    def test_missing_key_filtered_out(self):
        out = query(ROWS, "SELECT name FROM u WHERE city = 'LA'")
        # dave has no city, eve does → only bob & eve
        self.assertEqual([r["name"] for r in out], ["bob", "eve"])

    def test_null_eq_matches_absent_and_none(self):
        # = NULL → city missing OR None
        rows = [
            {"id": 1, "city": "NYC"},
            {"id": 2, "city": None},
            {"id": 3},
        ]
        out = query(rows, "SELECT id FROM r WHERE city = NULL")
        self.assertEqual(sorted(r["id"] for r in out), [2, 3])

    def test_null_neq(self):
        rows = [{"id": 1, "x": 1}, {"id": 2, "x": None}, {"id": 3}]
        out = query(rows, "SELECT id FROM r WHERE x != NULL")
        self.assertEqual([r["id"] for r in out], [1])

    def test_null_with_other_op_raises(self):
        with self.assertRaises(QueryError):
            query(ROWS, "SELECT name FROM u WHERE age > NULL")


class TestOrder(unittest.TestCase):
    def test_order_asc_default(self):
        out = query(ROWS, "SELECT name FROM u ORDER BY age")
        # eve has None age → goes LAST
        names = [r["name"] for r in out]
        self.assertEqual(names[-1], "eve")
        # rest sorted by age asc: bob25 dave28 alice30 carol35
        self.assertEqual(names[:4], ["bob", "dave", "alice", "carol"])

    def test_order_desc(self):
        out = query(ROWS, "SELECT name FROM u ORDER BY age DESC")
        names = [r["name"] for r in out]
        self.assertEqual(names[-1], "eve")  # missing/None still last
        self.assertEqual(names[:4], ["carol", "alice", "dave", "bob"])

    def test_order_missing_key_last(self):
        out = query(ROWS, "SELECT name FROM u ORDER BY city")
        names = [r["name"] for r in out]
        # dave has no city → he's last; LA < NYC alphabetically
        self.assertEqual(names[-1], "dave")
        self.assertEqual(names[:2], ["bob", "eve"])

    def test_order_stable(self):
        rows = [{"id": i, "k": i % 2} for i in range(6)]
        out = query(rows, "SELECT id FROM r ORDER BY k")
        # ids with k=0 first in original order, then k=1
        self.assertEqual([r["id"] for r in out], [0, 2, 4, 1, 3, 5])


class TestLimit(unittest.TestCase):
    def test_limit_basic(self):
        out = query(ROWS, "SELECT name FROM u LIMIT 2")
        self.assertEqual(len(out), 2)

    def test_limit_with_order(self):
        out = query(ROWS, "SELECT name FROM u ORDER BY age LIMIT 2")
        self.assertEqual([r["name"] for r in out], ["bob", "dave"])

    def test_limit_zero(self):
        out = query(ROWS, "SELECT name FROM u LIMIT 0")
        self.assertEqual(out, [])

    def test_limit_negative_raises(self):
        with self.assertRaises(QueryError):
            query(ROWS, "SELECT name FROM u LIMIT -1")


class TestParse(unittest.TestCase):
    def test_case_insensitive_keywords(self):
        out = query(ROWS, "select name from u where city = 'NYC' order by age desc")
        self.assertEqual([r["name"] for r in out], ["carol", "alice"])

    def test_syntax_error(self):
        with self.assertRaises(QueryError):
            query(ROWS, "SELECT FROM u")        # no projection
        with self.assertRaises(QueryError):
            query(ROWS, "SELECT * users")       # missing FROM


if __name__ == "__main__":
    unittest.main()
EOF
  cat > "$dir/TASK.md" <<'EOF'
Implement `query(rows, q)` and the `QueryError` exception in `qengine.py`.
The module docstring fully specifies the grammar and the semantics of
projection, WHERE, ORDER BY, and LIMIT.

`test_qengine.py` has 19 tests covering the full surface — projection
edge cases, NULL handling, missing-key sorting, parser errors. Do not
modify it.

Constraints:

- Single file. Stdlib only.
- A hand-rolled tokenizer + recursive-descent parser is the expected
  approach — no need for a parsing library.
- The executor must NOT mutate the input `rows` list or its dict elements.
- ORDER BY: rows missing the sort key sort to the end of the result in
  both ASC and DESC.
- LIMIT N: N must be a non-negative integer or `QueryError` is raised.

The verify command `python3 -m unittest -v` must report all 19 tests passing.
EOF
  ( cd "$dir" && git add -A && git commit -q -m "initial: SQL query-engine stubs + 19 tests" )
  echo "$dir"
}

# --- task 20: cross-module multi-bug hunt ------------------------------------
# Three small modules with FOUR interacting bugs. Tests exercise each
# module in isolation AND together. The agent needs to localise all
# four; fixing only one or two leaves multiple tests failing.
setup_task20() {
  local dir="$BENCH_ROOT/20-multibug"
  init_repo "$dir"
  cat > "$dir/storage.py" <<'EOF'
"""Tiny in-memory KV store with a write-ahead log.

Bug: `commit` writes to the log but forgets to apply the staged writes
to the live dict. After commit, get() still returns the pre-commit value.
"""
from __future__ import annotations

class Storage:
    def __init__(self) -> None:
        self._data: dict[str, str] = {}
        self._staged: dict[str, str] = {}
        self.log: list[tuple[str, str]] = []

    def put(self, key: str, value: str) -> None:
        self._staged[key] = value

    def commit(self) -> None:
        for k, v in self._staged.items():
            self.log.append((k, v))
            # BUG: never copies _staged into _data
        self._staged.clear()

    def get(self, key: str) -> str | None:
        return self._data.get(key)
EOF
  cat > "$dir/cache.py" <<'EOF'
"""Bounded write-through cache that sits on top of a Storage.

Bug 1: capacity check uses `>` instead of `>=` so the cache can grow to
       capacity+1 before evicting.
Bug 2: eviction order is wrong — it evicts the MOST recently inserted
       key instead of the oldest one.
"""
from __future__ import annotations
from collections import OrderedDict

from storage import Storage


class Cache:
    def __init__(self, storage: Storage, capacity: int) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be > 0")
        self.capacity = capacity
        self.storage = storage
        self._mem: OrderedDict[str, str] = OrderedDict()

    def set(self, key: str, value: str) -> None:
        if key in self._mem:
            self._mem.move_to_end(key)
            self._mem[key] = value
        else:
            self._mem[key] = value
            # BUG 1: should be >= not >
            if len(self._mem) > self.capacity:
                # BUG 2: should pop the FIRST item (oldest), not the last
                self._mem.popitem(last=True)
        self.storage.put(key, value)
        self.storage.commit()

    def get(self, key: str) -> str | None:
        if key in self._mem:
            self._mem.move_to_end(key)
            return self._mem[key]
        return self.storage.get(key)
EOF
  cat > "$dir/scheduler.py" <<'EOF'
"""Round-robin scheduler that picks the next ready task.

Bug: when the ready queue is empty, `next_task` returns the LAST task
ever added (cached in `self._last`) instead of returning None.
"""
from __future__ import annotations
from collections import deque

class Scheduler:
    def __init__(self) -> None:
        self._ready: deque[str] = deque()
        self._last: str | None = None

    def add(self, name: str) -> None:
        self._ready.append(name)
        self._last = name

    def next_task(self) -> str | None:
        if not self._ready:
            # BUG: should return None
            return self._last
        t = self._ready.popleft()
        return t
EOF
  cat > "$dir/test_storage.py" <<'EOF'
import unittest
from storage import Storage

class T(unittest.TestCase):
    def test_put_then_commit_visible(self):
        s = Storage()
        s.put("a", "1")
        self.assertIsNone(s.get("a"))    # not yet committed
        s.commit()
        self.assertEqual(s.get("a"), "1")

    def test_commit_clears_staged(self):
        s = Storage()
        s.put("a", "1")
        s.commit()
        # second commit with no puts shouldn't change anything
        s.commit()
        self.assertEqual(s.get("a"), "1")

    def test_log_records_all_commits(self):
        s = Storage()
        s.put("a", "1"); s.commit()
        s.put("b", "2"); s.commit()
        self.assertEqual(s.log, [("a", "1"), ("b", "2")])

    def test_overwrite(self):
        s = Storage()
        s.put("a", "1"); s.commit()
        s.put("a", "2"); s.commit()
        self.assertEqual(s.get("a"), "2")

if __name__ == "__main__":
    unittest.main()
EOF
  cat > "$dir/test_cache.py" <<'EOF'
import unittest
from storage import Storage
from cache import Cache

class T(unittest.TestCase):
    def test_set_get_within_capacity(self):
        c = Cache(Storage(), 3)
        c.set("a", "1"); c.set("b", "2")
        self.assertEqual(c.get("a"), "1")
        self.assertEqual(c.get("b"), "2")

    def test_eviction_oldest_first(self):
        c = Cache(Storage(), 2)
        c.set("a", "1")
        c.set("b", "2")
        c.set("c", "3")            # should evict "a"
        # 'a' is gone from in-memory cache but persisted in storage
        self.assertEqual(c.get("b"), "2")
        self.assertEqual(c.get("c"), "3")
        # 'a' falls through to storage and is found there
        self.assertEqual(c.get("a"), "1")

    def test_size_never_exceeds_capacity(self):
        c = Cache(Storage(), 3)
        for i in range(10):
            c.set(f"k{i}", str(i))
        self.assertLessEqual(len(c._mem), 3)

    def test_get_promotes_to_mru(self):
        c = Cache(Storage(), 3)
        c.set("a", "1"); c.set("b", "2"); c.set("c", "3")
        c.get("a")                 # promote a
        c.set("d", "4")            # evicts oldest, which is now 'b'
        self.assertNotIn("b", c._mem)
        self.assertIn("a", c._mem)

if __name__ == "__main__":
    unittest.main()
EOF
  cat > "$dir/test_scheduler.py" <<'EOF'
import unittest
from scheduler import Scheduler

class T(unittest.TestCase):
    def test_fifo_order(self):
        s = Scheduler()
        s.add("a"); s.add("b"); s.add("c")
        self.assertEqual(s.next_task(), "a")
        self.assertEqual(s.next_task(), "b")
        self.assertEqual(s.next_task(), "c")

    def test_empty_returns_none(self):
        s = Scheduler()
        self.assertIsNone(s.next_task())

    def test_empty_after_drain_returns_none(self):
        s = Scheduler()
        s.add("a"); s.next_task()
        self.assertIsNone(s.next_task())

if __name__ == "__main__":
    unittest.main()
EOF
  cat > "$dir/test_integration.py" <<'EOF'
import unittest
from storage import Storage
from cache import Cache
from scheduler import Scheduler


class T(unittest.TestCase):
    def test_cache_persists_evicted_through_storage(self):
        st = Storage()
        c = Cache(st, 2)
        c.set("a", "1"); c.set("b", "2"); c.set("c", "3")
        # 'a' evicted from memory but persisted via storage commit
        self.assertEqual(st.get("a"), "1")
        self.assertEqual(c.get("a"), "1")

    def test_scheduler_drains_after_processing(self):
        sch = Scheduler()
        st = Storage()
        c = Cache(st, 5)
        for k in ["t1", "t2", "t3"]:
            sch.add(k)
        processed = []
        # cap iterations so a broken Scheduler.next_task() that never
        # returns None fails fast instead of hanging the test runner.
        for _ in range(20):
            t = sch.next_task()
            if t is None:
                break
            processed.append(t)
            c.set(t, "done")
        else:
            self.fail("scheduler did not drain (next_task never returned None)")
        self.assertEqual(processed, ["t1", "t2", "t3"])
        for k in ["t1", "t2", "t3"]:
            self.assertEqual(c.get(k), "done")


if __name__ == "__main__":
    unittest.main()
EOF
  cat > "$dir/TASK.md" <<'EOF'
Three modules — `storage.py`, `cache.py`, `scheduler.py` — each contain
one or more bugs. There are FOUR bugs in total across the three files.
The test files (`test_storage.py`, `test_cache.py`, `test_scheduler.py`,
`test_integration.py`) collectively cover all four.

Find and fix every bug. The bugs interact: e.g. the cache's eviction
bug only becomes visible when the storage commit bug is ALSO fixed,
because otherwise the integration test fails earlier in the call chain.

Constraints:

- Do not modify any test file.
- Do not change public APIs: class names, method names, attribute names,
  return types must stay as documented in the modules' docstrings.
- Each fix should be the smallest change that makes the relevant tests
  pass. No refactoring, no new helpers, no new modules.
- The bugs are localised — they should be one-line / few-line changes.

The verify command `python3 -m unittest -v` must report all 13 tests
across the 4 test files passing.

IMPORTANT — planning guidance: the four bugs are interdependent for the
purposes of the verify command. `python3 -m unittest -v` runs the entire
suite; any one unfixed bug leaves several tests failing, so verify will
NOT go green until all four bugs are fixed. Treat this as a SINGLE
atomic step in the plan ("fix all four bugs across storage.py, cache.py,
scheduler.py"). Do not split it across multiple steps — every
intermediate step would fail verify and be wasted work.
EOF
  ( cd "$dir" && git add -A && git commit -q -m "initial: 3 modules with 4 interacting bugs" )
  echo "$dir"
}

# --- task 21: long-form refactor — sync→async --------------------------------
# Convert a small stdlib-only "fake-network" client lib to a fully-async
# API. Three files. Tests use asyncio.run() and check semantics are
# preserved.
setup_task21() {
  local dir="$BENCH_ROOT/21-sync-to-async"
  init_repo "$dir"
  cat > "$dir/transport.py" <<'EOF'
"""Synchronous fake-network transport.

A `Transport` exposes one method, `fetch(url) -> str`, which sleeps for
a few milliseconds and then returns a deterministic body keyed off the
URL. Used as the bottom layer of the client stack.
"""
from __future__ import annotations
import time


class Transport:
    def __init__(self, latency_s: float = 0.001) -> None:
        self.latency_s = latency_s
        self.calls: list[str] = []

    def fetch(self, url: str) -> str:
        self.calls.append(url)
        time.sleep(self.latency_s)
        return f"body:{url}"
EOF
  cat > "$dir/client.py" <<'EOF'
"""Higher-level client. Wraps a Transport and exposes `get(path)`.

Right now everything is synchronous. The task is to make this async.
"""
from __future__ import annotations
from transport import Transport


class Client:
    def __init__(self, transport: Transport, base_url: str) -> None:
        self.transport = transport
        self.base_url = base_url.rstrip("/")

    def get(self, path: str) -> str:
        if not path.startswith("/"):
            raise ValueError("path must start with /")
        return self.transport.fetch(self.base_url + path)
EOF
  cat > "$dir/batch.py" <<'EOF'
"""Batch-fetch helper. Today it just loops sequentially over Client.get.

After the refactor, `batch_fetch` should issue all requests concurrently
using `asyncio.gather` and preserve input order.
"""
from __future__ import annotations
from client import Client


def batch_fetch(client: Client, paths: list[str]) -> list[str]:
    return [client.get(p) for p in paths]
EOF
  cat > "$dir/test_async.py" <<'EOF'
import asyncio
import inspect
import time
import unittest

import transport as transport_mod
import client as client_mod
import batch as batch_mod


class TestTransport(unittest.TestCase):
    def test_fetch_is_coroutine(self):
        self.assertTrue(
            asyncio.iscoroutinefunction(transport_mod.Transport.fetch),
            "Transport.fetch must be an async coroutine function",
        )

    def test_fetch_returns_body(self):
        t = transport_mod.Transport(latency_s=0.001)
        out = asyncio.run(t.fetch("http://x/y"))
        self.assertEqual(out, "body:http://x/y")
        self.assertEqual(t.calls, ["http://x/y"])

    def test_fetch_uses_asyncio_sleep_not_blocking(self):
        # Source-level check: the implementation must use asyncio.sleep,
        # not time.sleep, otherwise concurrent fetches won't overlap.
        src = inspect.getsource(transport_mod.Transport.fetch)
        self.assertNotIn("time.sleep", src)
        self.assertIn("asyncio.sleep", src)


class TestClient(unittest.TestCase):
    def test_get_is_coroutine(self):
        self.assertTrue(asyncio.iscoroutinefunction(client_mod.Client.get))

    def test_get_concatenates_base(self):
        t = transport_mod.Transport()
        c = client_mod.Client(t, "http://api/")
        body = asyncio.run(c.get("/users"))
        self.assertEqual(body, "body:http://api/users")

    def test_get_validates_path(self):
        c = client_mod.Client(transport_mod.Transport(), "http://api")
        async def _go():
            return await c.get("users")
        with self.assertRaises(ValueError):
            asyncio.run(_go())


class TestBatch(unittest.TestCase):
    def test_batch_is_coroutine(self):
        self.assertTrue(asyncio.iscoroutinefunction(batch_mod.batch_fetch))

    def test_batch_preserves_order(self):
        t = transport_mod.Transport()
        c = client_mod.Client(t, "http://api")
        paths = ["/a", "/b", "/c", "/d"]
        out = asyncio.run(batch_mod.batch_fetch(c, paths))
        self.assertEqual(out, [f"body:http://api{p}" for p in paths])

    def test_batch_runs_concurrently(self):
        # 10 fetches of 50ms each should take ~50ms total when concurrent,
        # not ~500ms when sequential. We give ourselves a 250ms ceiling
        # to be generous on slow CI.
        t = transport_mod.Transport(latency_s=0.05)
        c = client_mod.Client(t, "http://api")
        paths = [f"/p{i}" for i in range(10)]
        t0 = time.perf_counter()
        out = asyncio.run(batch_mod.batch_fetch(c, paths))
        elapsed = time.perf_counter() - t0
        self.assertEqual(len(out), 10)
        self.assertLess(elapsed, 0.25,
            f"batch_fetch took {elapsed:.2f}s — must run requests concurrently")


if __name__ == "__main__":
    unittest.main()
EOF
  cat > "$dir/TASK.md" <<'EOF'
Convert `transport.py`, `client.py`, and `batch.py` to use `asyncio`
end-to-end:

1. `Transport.fetch` becomes `async def fetch(...)` using `asyncio.sleep`
   in place of `time.sleep`. It still appends to `self.calls` and still
   returns `f"body:{url}"`.
2. `Client.get` becomes `async def get(...)` and awaits `transport.fetch`.
   Path validation behaviour is unchanged.
3. `batch_fetch` becomes `async def batch_fetch(...)`. It must issue all
   client.get calls CONCURRENTLY using `asyncio.gather`, preserving
   input order in the returned list.

Constraints:

- Do not modify `test_async.py`.
- Keep all class names, attribute names, constructor signatures, and
  module-level public API as they are; only the function signatures
  change to `async def` and the bodies adapt.
- `Transport.calls` is still a plain `list[str]`; appending from inside
  the coroutine is fine.
- No new files. Stdlib only.

The verify command `python3 -m unittest -v` must report all 9 tests in
`test_async.py` passing, including `test_batch_runs_concurrently` which
asserts that 10 × 50ms fetches complete in well under 250ms (proving
they actually overlap).
EOF
  ( cd "$dir" && git add -A && git commit -q -m "initial: sync HTTP client trio to refactor to async" )
  echo "$dir"
}

# --- runner -------------------------------------------------------------------
