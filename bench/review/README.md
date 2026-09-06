# Adversarial review panel: validation

agent6 can run N reviewer subagents over a diff and aggregate their verdicts. It
ships two ways: `agent6 review --reviewers N [--personas a,b,c]` (read-only,
post-hoc) and in-loop via the `[review]` config (the `before_finish` /
`on_verify_fail` / `periodic` triggers, with `advisory` / `veto` / `quorum`
decisions). The `ultra` profile enables a 3-seat `before_finish` veto panel. The
configuration reference documents the fields.

This file records why the panel does not reintroduce the false-blocking that got
the pre-0.0.4 `reviewer.py` deleted, and the runs behind enabling it.

## Why it does not false-block

The old `reviewer.py` blocked correct, green-verify work because its grounding was
prose ("do not block on speculation"), and models rationalize around prose. The
panel makes the grounding executable in the aggregator instead. A reviewer's
`block` gates only if a machine check passes: the cited `file_line` is actually
present in the diff the seat was shown, and the category is in the allowed-block
set (`security`, `sandbox-bypass`, `off-topic-edit`, `data-loss`,
`verify-uncovered-correctness`). A block that fails either check is downgraded to
`warn` before any quorum or veto counting. Taste, naming, "should add a test", and
uncited claims cannot gate.

## Validation

Primary gate: the false-block and stall rate on correct work must stay at the
panel-off baseline. Secondary: real-bug catch rate, added tokens and USD, wall
time.

### Post-hoc, planted diffs (`--reviewers 3`, kimi-k2.6, real API)

| Diff | Result |
|------|--------|
| Hardcoded API key + path traversal | 2 grounded blocks (deduped from 3 seats); both real, cited at real lines |
| `token ==` compare + stray `.err` files | 3 grounded blocks: timing side-channel + 2 off-topic-edit |
| Clean (constant-time compare, env secret) | PASS under veto; the lone finding was a non-gating `warn` |

The grounded aggregator caught real, cited, block-eligible defects and did not
false-block clean work with the veto on. About $0.03 to $0.06 per panel; 3
concurrent seats run in about 2 minutes.

### In-loop veto, full run

A `read_doc(name)` task with a path-traversal trap (`before_finish`, 3 seats,
veto). The worker's naive `open("files/" + name)` was blocked by all 3 seats. Its
first fix (an abspath check) still let symlink traversal through, and security
blocked again. It switched to `os.path.realpath` and the panel passed. Three
rounds, no stall, final code symlink-safe.

### A/B, off vs veto (3 seats, kimi)

| Task | off | veto |
|------|-----|------|
| traversal | vulnerable (4 it) | secure via basename (9 it) |
| token compare | vulnerable (4 it) | secure via env + compare_digest (12 it) |
| correct math | correct (4 it) | correct (4 it, no churn) |
| correct string | correct (4 it) | correct (4 it, no churn) |

Both real vulnerabilities the baseline shipped were caught and fixed under veto,
with no false-blocks on clean work (veto matched off's iteration count). A clean
run adds about one panel call; a buggy run costs the iterations to fix it.

Keep `XDG_STATE_HOME` out of the repo (the default). An earlier A/B put it
inside, so agent6's own logs were committed and the panel correctly blocked them
as off-topic edits.

### Explore tier, cross-file

`parse(s)` gained a required argument, breaking a caller in `main.py` that was not
in the diff. A diff-tier seat passed (it cannot see the caller). An explore-tier
seat (read-only `read_file` + `find_references`) found `main.py` and blocked with
`verify-uncovered-correctness`, cited at the diff line. The explore tier adds
value over the diff tier for cross-file impact; the diff tier is the default.
