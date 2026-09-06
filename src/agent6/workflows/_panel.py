# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Pure core of the adversarial review panel: the verdict types and the
grounded aggregator.

The aggregator is where the panel earns its keep: a "don't block on
speculation" rule held as *prose* gets rationalized around, and correct
green-verify work gets false-blocked. Here the rule is **executable**: a
reviewer's `block` only gates if a
machine check passes (the cited line is actually in the diff it was shown, and
the category is one we allow to block). Everything else is mechanically
downgraded to `warn` before any veto/quorum counting. `warn`/`nit` never
gate. This module is network-free and exhaustively unit-tested; `run_panel`
(the orchestration that actually calls models) lives separately.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from typing import Literal

Severity = Literal["block", "warn", "nit"]
Verdict = Literal["pass", "block"]
ReviewDecision = Literal["advisory", "veto", "quorum", "all"]

# Categories a finding may carry. Only the first set is allowed to GATE; the
# rest advise but never block (taste/test-gap/over-engineering findings are
# too noisy to gate).
ALLOWED_BLOCK_CATEGORIES: frozenset[str] = frozenset(
    {"security", "sandbox-bypass", "off-topic-edit", "data-loss", "verify-uncovered-correctness"}
)
ADVISORY_CATEGORIES: frozenset[str] = frozenset({"test-gap", "style", "over-eng", "other"})
ALL_CATEGORIES: frozenset[str] = ALLOWED_BLOCK_CATEGORIES | ADVISORY_CATEGORIES


@dataclass(frozen=True, slots=True)
class Finding:
    category: str
    severity: Severity
    file_line: str  # "path:line" or "path" (the citation the grounding check uses)
    title: str
    detail: str = ""


@dataclass(frozen=True, slots=True)
class ReviewVerdict:
    seat: str
    model: str
    verdict: Verdict
    findings: tuple[Finding, ...] = ()
    summary: str = ""
    error: str | None = None  # set => the seat failed and ABSTAINS (not a pass)


@dataclass(frozen=True, slots=True)
class ReviewContext:
    """What every seat is shown, and what the aggregator grounds findings against."""

    task: str = ""
    agents_md: str = ""
    diff: str = ""  # the working-tree delta since the last accepted finish
    verify_ok: bool | None = None  # None = no result: none configured, or `agent6 review`
    verify_output: str = ""
    persona: str = ""
    prior_findings: tuple[Finding, ...] = ()  # already-injected, for dedup (not re-count)


@dataclass(frozen=True, slots=True)
class PanelResult:
    panel_id: str
    decision: ReviewDecision
    blocked: bool
    merged_findings: tuple[Finding, ...]
    per_seat: tuple[ReviewVerdict, ...]
    n_block: int  # distinct-model blocking seats counted toward the gate
    n_abstain: int
    skipped_reason: str | None = None


def panel_is_inconclusive(result: PanelResult) -> bool:
    """Whether EVERY seat abstained, so nothing was actually reviewed. "0
    blocking findings" is not a clean bill then -- the single owner both the CLI
    verdict and the in-loop critique text ask, so neither can launder an
    all-abstain panel into a pass."""
    return bool(result.per_seat) and result.n_abstain == len(result.per_seat)


def inconclusive_note(result: PanelResult) -> str:
    """The human line for an all-abstain panel (see :func:`panel_is_inconclusive`)."""
    return f"review inconclusive: all {result.n_abstain} seats abstained; nothing was reviewed"


# ----------------------------------------------------------------------------
# Diff grounding: the (path, line) citations this diff supports.
# ----------------------------------------------------------------------------

# Capture BOTH the old-side (-A,B) and new-side (+C,D) line numbers so deletions
# ground against the pre-image path too.
_HUNK_RE = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")
_GIT_HDR_RE = re.compile(r"^diff --git a/(.*?) b/(.*)$")


def _unquote_git_path(p: str) -> str:
    """git quotes paths with special chars as a C-string ("b/a\\tb"); strip the
    quotes + the common backslash escapes so the header path matches a citation."""
    p = p.strip()
    if len(p) >= 2 and p.startswith('"') and p.endswith('"'):
        p = p[1:-1].replace('\\"', '"').replace("\\\\", "\\")
    return p


def _hdr_path(raw: str) -> str:
    """Path from a `--- `/`+++ ` header line ("" for /dev/null)."""
    target = _unquote_git_path(raw[4:].split("\t", 1)[0])
    return "" if target == "/dev/null" else re.sub(r"^[ab]/", "", target)


def diff_touched_ranges(diff: str) -> dict[str, list[tuple[int, int]]]:
    """Map each touched path to the line ranges its hunks changed, so a finding's
    `path:line` citation can be grounded: a block may only gate if its cited
    line is inside a range the diff changed. New-side ranges key the post-image
    path; old-side ranges key the pre-image path (for an in-place modification,
    the same key -- overlapping ranges are harmless), so a citation of deleted
    code at its OLD line number still grounds whether the whole file was deleted
    (post-image `/dev/null`) or lines were removed from a kept file. A `+++ `
    line only counts as a header when it follows a `--- ` (an added line whose
    content happens to start with `++ ` is not mistaken for one). A file the
    diff touches without hunks (binary, a pure rename, a mode change) is
    recorded with no ranges, so a path-only citation of it grounds and a line
    citation does not.
    """
    ranges: dict[str, list[tuple[int, int]]] = {}
    newpath = oldpath = ""
    prev_minus = False
    lines = diff.splitlines()
    for i, raw in enumerate(lines):
        if g := _GIT_HDR_RE.match(raw):
            ranges.setdefault(_unquote_git_path(g.group(1)), [])
            ranges.setdefault(_unquote_git_path(g.group(2)), [])
            oldpath = newpath = ""
            prev_minus = False
            continue
        # A real "--- " file header is always immediately followed by a "+++ "
        # header. Requiring that lookahead stops a DELETED line whose own text
        # begins with "-- " -- rendered "--- ..." in the diff -- from being
        # misparsed as a file header, which would clobber oldpath/newpath and
        # mis-attribute every later hunk's ranges (the symmetric "+++ " side is
        # already guarded by prev_minus).
        if raw.startswith("--- ") and i + 1 < len(lines) and lines[i + 1].startswith("+++ "):
            oldpath, newpath, prev_minus = _hdr_path(raw), "", True
            continue
        if raw.startswith("+++ ") and prev_minus:
            newpath, prev_minus = _hdr_path(raw), False
            continue
        prev_minus = False
        m = _HUNK_RE.match(raw)
        if m:
            o_start = int(m.group(1))
            o_count = int(m.group(2)) if m.group(2) is not None else 1
            n_start = int(m.group(3))
            n_count = int(m.group(4)) if m.group(4) is not None else 1
            if newpath:
                ranges.setdefault(newpath, []).append((n_start, n_start + max(n_count, 1) - 1))
            if oldpath:  # old-side range: grounds citations of pre-image (deleted) lines
                ranges.setdefault(oldpath, []).append((o_start, o_start + max(o_count, 1) - 1))
    return ranges


def _norm_path(p: str) -> str:
    return re.sub(r"^[ab]/", "", _unquote_git_path(p))


def _split_cite(file_line: str) -> tuple[str, tuple[int, int] | None]:
    """One citation parsed into `(normalized path, cited line span or None)`.

    The forms a reviewer writes: 'foo.py', 'a/foo.py:2', 'foo.py:2-4',
    'foo.py:2:' and 'foo.py:12:5' (the standard compiler/grep -n
    `path:line:col`). ONE owner: a per-consumer single-rpartition parse would read a line:col
    citation's COLUMN as the line and 'foo.py:12' as the path, so grounding
    would miss and a real block silently downgrade to a warning.
    """
    cite = file_line.strip().rstrip(":")
    if not cite:
        return "", None
    parts = cite.split(":")
    # Trailing numeric fields are position (line, then column); the path is
    # whatever precedes them. A Windows-style 'C:\\x.py' keeps its drive letter
    # because that field is not numeric-only.
    nums: list[str] = []
    while len(parts) > 1 and _is_span(parts[-1]):
        nums.insert(0, parts.pop())
    path = _norm_path(":".join(parts))
    if not nums:
        return path, None
    ends = nums[0].split("-", 1)  # "2-4" -> ["2", "4"]; "2" -> ["2"]
    lo, hi = int(ends[0]), int(ends[-1])
    return path, (min(lo, hi), max(lo, hi))  # a reversed range normalizes


def _is_span(field: str) -> bool:
    """True for a numeric position field: a line ("12") or a range ("2-4")."""
    ends = field.split("-", 1)
    return all(e.isdigit() for e in ends) and bool(ends[0])


def is_grounded(file_line: str, ranges: dict[str, list[tuple[int, int]]]) -> bool:
    """True iff the citation refers to something the diff actually changed:
    a touched path (path-only citation) or a line/range that OVERLAPS a touched
    range. A range citation "path:A-B" grounds if *any* line in [A, B] falls
    inside a touched range, not just the start line A -- otherwise a finding that
    cites a real changed range whose start happens to be unchanged-but-interior
    (e.g. the hunk modified the middle of the cited span) would be wrongly treated
    as ungrounded and a legit block silently downgraded to warn."""
    path, span = _split_cite(file_line)
    if not path:
        return False
    if span is None:  # path-only: grounded if the diff touched that file at all
        return path in ranges
    c_lo, c_hi = span
    return any(c_lo <= hi and lo <= c_hi for (lo, hi) in ranges.get(path, ()))


DedupKey = tuple[str, str, tuple[int, int] | None]


def _dedup_key(f: Finding, ranges: dict[str, list[tuple[int, int]]]) -> DedupKey:
    """A finding's identity across seats and iterations: its file, its category
    and the hunk its citation falls in (the first touched range overlapping the
    cited span), so a re-citation a few lines off collapses while a second
    finding in another hunk of the same file stays. A citation outside every
    hunk keys on its own span, a path-only one on the file."""
    path, span = _split_cite(f.file_line)
    if span is None:
        return (path, f.category, None)
    lo, hi = span
    for r_lo, r_hi in ranges.get(path, ()):
        if lo <= r_hi and r_lo <= hi:
            return (path, f.category, (r_lo, r_hi))
    return (path, f.category, span)


_SEV_ORDER = {"block": 0, "warn": 1, "nit": 2}


def _ground_severity(
    f: Finding, ctx: ReviewContext, ranges: dict[str, list[tuple[int, int]]]
) -> Severity:
    """A `block` survives only if grounded in the diff AND in a gating category
    (and `verify-uncovered-correctness` is coherent only when verify passed);
    otherwise it is downgraded to `warn`. `warn`/`nit` pass through."""
    if f.severity != "block":
        return f.severity
    coherent = f.category != "verify-uncovered-correctness" or ctx.verify_ok is True
    if f.category in ALLOWED_BLOCK_CATEGORIES and coherent and is_grounded(f.file_line, ranges):
        return "block"
    return "warn"


def _ground_seat(
    v: ReviewVerdict, ctx: ReviewContext, ranges: dict[str, list[tuple[int, int]]]
) -> ReviewVerdict:
    out: list[Finding] = []
    for f in v.findings:
        sev = _ground_severity(f, ctx, ranges)
        out.append(f if sev == f.severity else replace(f, severity=sev))
    return replace(v, findings=tuple(out))


def _has_new_block(
    v: ReviewVerdict, prior_keys: set[DedupKey], ranges: dict[str, list[tuple[int, int]]]
) -> bool:
    """True when the seat carries a surviving block that is NOT an already-
    injected prior finding. `prior_findings` is "for dedup (not re-count)":
    a block whose key dedups away is dropped from `merged_findings`, so
    letting it gate would reject the work while reporting no blocking
    findings."""
    return any(
        f.severity == "block" and _dedup_key(f, ranges) not in prior_keys for f in v.findings
    )


def _decide(
    decision: ReviewDecision,
    n_block: int,
    quorum: int,
    non_abstain: list[ReviewVerdict],
    n_total: int,
    prior_keys: set[DedupKey],
    ranges: dict[str, list[tuple[int, int]]],
) -> bool:
    if decision == "advisory" or not non_abstain:
        return False
    if decision == "veto":
        return n_block >= 1
    if decision == "quorum":
        return n_block >= max(1, quorum)
    if decision == "all":
        # "all" means UNANIMOUS agreement among the seats that actually reviewed.
        # Abstentions (provider error / unparseable / deadline) are not votes, so
        # they must not let a lone blocker gate while everyone else failed to
        # respond. Require both: (a) every non-abstaining seat blocked, AND (b) a
        # meaningful quorum actually responded -- a strict majority of all seats
        # must be non-abstaining. So a panel that mostly failed to respond does
        # NOT block on one vote, but a fully-responding (or majority-responding)
        # panel that unanimously blocks still gates.
        if not all(_has_new_block(v, prior_keys, ranges) for v in non_abstain):
            return False
        return len(non_abstain) * 2 > n_total
    return False  # pragma: no cover - exhaustive


def aggregate_verdicts(
    per_seat: list[ReviewVerdict],
    ctx: ReviewContext,
    *,
    decision: ReviewDecision,
    quorum: int,
    panel_id: str,
) -> PanelResult:
    """Fold per-seat verdicts into one panel result with EXECUTABLE grounding.

    1. Ground every `block` finding: it survives as a block only if its
       `file_line` is in the diff AND its category is allowed to block (and a
       `verify-uncovered-correctness` claim is only coherent when verify
       actually passed). Otherwise it is downgraded to `warn`.
    2. Dedup across seats and against `prior_findings` by (path, category,
       hunk): a re-citation of one defect a few lines off collapses, a second
       finding in another hunk of the same file stays (`_dedup_key`). An
       already-injected block neither re-surfaces nor counts toward the gate
       (otherwise a rejection could ship with zero merged findings).
    3. Decide: advisory never blocks; veto blocks on any surviving block; quorum
       needs >= `quorum` blocks counting **at most one per distinct model**
       (correlated same-model seats cannot fabricate a quorum); all needs every
       non-abstaining seat to block AND a strict majority of all seats to have
       actually responded (abstentions cannot let a lone blocker gate under "all").
    """
    ranges = diff_touched_ranges(ctx.diff)
    prior_keys = {_dedup_key(f, ranges) for f in ctx.prior_findings}

    grounded_seats: list[ReviewVerdict] = []
    blocking_models: set[str] = set()  # distinct models with >=1 surviving non-prior block
    n_abstain = 0
    for v in per_seat:
        if v.error is not None:
            n_abstain += 1
            grounded_seats.append(v)
            continue
        gv = _ground_seat(v, ctx, ranges)
        grounded_seats.append(gv)
        if _has_new_block(gv, prior_keys, ranges):
            blocking_models.add(v.model)

    # Merge + dedup all findings (post-grounding); drop ones already injected.
    merged: dict[DedupKey, Finding] = {}
    for v in grounded_seats:
        for f in v.findings:
            key = _dedup_key(f, ranges)
            if key in prior_keys:
                continue
            cur = merged.get(key)
            if cur is None or _SEV_ORDER[f.severity] < _SEV_ORDER[cur.severity]:
                merged[key] = f
    merged_findings = tuple(
        sorted(merged.values(), key=lambda f: (_SEV_ORDER[f.severity], f.file_line, f.category))
    )

    n_block = len(blocking_models)  # distinct-model blocking seats
    non_abstain = [v for v in grounded_seats if v.error is None]
    blocked = _decide(
        decision, n_block, quorum, non_abstain, len(grounded_seats), prior_keys, ranges
    )

    return PanelResult(
        panel_id=panel_id,
        decision=decision,
        blocked=blocked,
        merged_findings=merged_findings,
        per_seat=tuple(grounded_seats),
        n_block=n_block,
        n_abstain=n_abstain,
    )


def render_findings(findings: tuple[Finding, ...]) -> str:
    """Render merged findings as a compact `[review]` block for the worker /
    the post-hoc CLI. Empty -> ''."""
    if not findings:
        return ""
    lines = []
    for f in findings:
        loc = f" ({f.file_line})" if f.file_line.strip() else ""
        lines.append(f"- [{f.severity}:{f.category}]{loc} {f.title}")
        if f.detail.strip():
            lines.append(f"    {f.detail.strip()}")
    return "\n".join(lines)


__all__ = [
    "ALLOWED_BLOCK_CATEGORIES",
    "ALL_CATEGORIES",
    "Finding",
    "PanelResult",
    "ReviewContext",
    "ReviewDecision",
    "ReviewVerdict",
    "aggregate_verdicts",
    "diff_touched_ranges",
    "is_grounded",
    "render_findings",
]
