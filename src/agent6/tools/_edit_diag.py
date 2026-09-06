# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Diagnostics for apply_edit / apply_patch: dry-run preview, and on a failed
match the closest on-disk region so the model can retry without re-reading.
"""

from __future__ import annotations

import difflib

from agent6.tools.results import PreviewResult


def preview_result(
    path: str,
    old_text: str | None,
    new_text: str,
    *,
    bytes_before: int,
    bytes_after: int,
    applied: list[str] | None = None,
) -> PreviewResult:
    """Build the dry-run response for `apply_edit`/`apply_patch` with
    `preview=true`. Returns the unified diff (old vs new) and a hunk
    count, but does NOT write anything to disk. The byte counts are the
    caller's, measured on disk (`disk_bytes`), so they match an apply's.

    Lets the agent sanity-check a complex multi-edit call
    before committing to it. Diff is bounded so a preview of a 100k-line
    rewrite doesn't dump the whole file back into the conversation.
    """
    old_lines = (old_text or "").splitlines(keepends=True)
    new_lines = new_text.splitlines(keepends=True)
    label_a = "/dev/null" if old_text is None else f"a/{path}"
    label_b = f"b/{path}"
    diff_iter = difflib.unified_diff(old_lines, new_lines, fromfile=label_a, tofile=label_b, n=3)
    diff = "".join(diff_iter)
    hunks = sum(1 for line in diff.splitlines() if line.startswith("@@ "))
    truncated = False
    _MAX_DIFF_CHARS = 8000
    if len(diff) > _MAX_DIFF_CHARS:
        diff = diff[:_MAX_DIFF_CHARS] + f"\n... <truncated {len(diff) - _MAX_DIFF_CHARS} chars>\n"
        truncated = True
    return PreviewResult(
        path=path,
        diff=diff or "(no changes)",
        hunks=hunks,
        bytes_before=bytes_before,
        bytes_after=bytes_after,
        truncated=truncated,
        would_apply=None if applied is None else tuple(applied),
    )


# Cap the closest-match scan so a failed edit on a very large file does not turn
# into a quadratic diff. Above this the diagnostic falls back to file shape.
CLOSEST_MATCH_MAX_LINES = 6000


def _leading_ws(s: str) -> str:
    return s[: len(s) - len(s.lstrip())]


def _reindent(lines: list[str], old_base: str, new_base: str) -> list[str] | None:
    """Replace each non-blank line's leading `old_base` with `new_base`,
    preserving any indentation beyond the base. Blank lines pass through. Returns
    None if any non-blank line does not start with `old_base` (the shift does
    not apply cleanly, so it is not safe to guess)."""
    out: list[str] = []
    for ln in lines:
        if not ln.strip():
            out.append(ln)
        elif ln.startswith(old_base):
            out.append(new_base + ln[len(old_base) :])
        else:
            return None
    return out


def indent_tolerant_replacement(file_text: str, old_string: str, new_string: str) -> str | None:
    """Apply an edit whose `old_string` doesn't match verbatim but matches
    EXACTLY ONE on-disk region up to a uniform leading-indent shift -- the
    dominant weak-model mistake: correct lines, wrong indent depth. Returns the
    edited file text, or None whenever it is not provably safe (no match,
    multiple matches, or a non-uniform diff) so the caller keeps the exact-match
    error.

    Safety gate: the shift derived from the first content line is applied to
    `old_string` and must reproduce the matched region byte-for-byte before it
    is applied to `new_string`. So the region is only ever edited when the
    transform is proven correct for old -> disk; a wrong region cannot be hit.
    Trailing-whitespace or non-uniform mismatches fail the gate and fall back."""
    old_lines = old_string.split("\n")
    file_lines = file_text.split("\n")
    n = len(old_lines)
    if n == 0 or n > len(file_lines):
        return None
    old_stripped = [ln.strip() for ln in old_lines]
    if not any(old_stripped):  # all-blank old_string: nothing to anchor safely
        return None
    starts = [
        i
        for i in range(len(file_lines) - n + 1)
        if [ln.strip() for ln in file_lines[i : i + n]] == old_stripped
    ]
    if len(starts) != 1:  # no match, or ambiguous -> never guess
        return None
    start = starts[0]
    region = file_lines[start : start + n]
    old_base = next(_leading_ws(o) for o in old_lines if o.strip())
    new_base = next(_leading_ws(r) for r in region if r.strip())
    if _reindent(old_lines, old_base, new_base) != region:
        return None  # the shift is not uniform across the region -> unsafe
    new_region = _reindent(new_string.split("\n"), old_base, new_base)
    if new_region is None:
        return None  # new_string can't take the same shift cleanly -> fall back
    return "\n".join(file_lines[:start] + new_region + file_lines[start + n :])


def closest_on_disk_region(file_text: str, old_string: str) -> tuple[int, str, float] | None:
    """Find the file region most similar to a not-found `old_string`.

    Returns `(1-based start line, region text, similarity ratio)` for the best
    contiguous window with the same line count as `old_string`, or None when
    the scan is skipped (empty or oversized file). This lets a failed
    `apply_edit` hand the model the EXACT on-disk text to retry with, instead
    of telling it to re-read the whole file (the dominant small-model time sink).
    """
    file_lines = file_text.splitlines()
    if not file_lines or len(file_lines) > CLOSEST_MATCH_MAX_LINES:
        return None
    old_lines = old_string.splitlines() or [old_string]
    n = max(1, min(len(old_lines), len(file_lines)))
    matcher = difflib.SequenceMatcher(autojunk=False)
    matcher.set_seq2(old_string)
    best_ratio = -1.0
    best_idx = 0
    for i in range(0, len(file_lines) - n + 1):
        window = "\n".join(file_lines[i : i + n])
        matcher.set_seq1(window)
        # quick_ratio is a cheap upper bound; skip windows that cannot win.
        if matcher.quick_ratio() <= best_ratio:
            continue
        r = matcher.ratio()
        if r > best_ratio:
            best_ratio = r
            best_idx = i
    region = "\n".join(file_lines[best_idx : best_idx + n])
    return best_idx + 1, region, best_ratio


def edit_mismatch_error(path: str, edit_index: int, file_text: str, old_string: str) -> str:
    """Build the not-found error for `apply_edit`. Prefers a copy-paste-able
    closest on-disk region so the model retries directly; falls back to file
    shape only when no region is similar enough to be useful."""
    region_info = closest_on_disk_region(file_text, old_string)
    if region_info is not None and region_info[2] >= 0.5:
        start_line, region, ratio = region_info
        end_line = start_line + len(region.splitlines()) - 1
        diff = "\n".join(
            difflib.unified_diff(
                old_string.splitlines(),
                region.splitlines(),
                fromfile="your_old_string",
                tofile=f"on_disk_lines_{start_line}-{end_line}",
                lineterm="",
                n=1,
            )
        )
        whitespace_only = [ln.strip() for ln in old_string.splitlines()] == [
            ln.strip() for ln in region.splitlines()
        ]
        why = (
            "It matches your old_string except for whitespace/indentation."
            if whitespace_only
            else f"It is the closest region on disk ({ratio:.0%} similar)."
        )
        return (
            f"old_string not found in {path} (edit #{edit_index}). {why} Retry"
            f" apply_edit using the EXACT on-disk text below as old_string — do"
            f" NOT call read_file first; this IS the current content of lines"
            f" {start_line}-{end_line}.\n"
            f"<<<ON_DISK (copy this verbatim, without these <<< >>> markers)\n"
            f"{region}\n"
            f">>>ON_DISK\n"
            f"difference (- your old_string, + on disk):\n{diff}"
        )
    # No region similar enough: orient with file shape only (no body to copy,
    # so the model cannot plagiarise a wrong anchor).
    lines = file_text.splitlines()
    head = "\n".join(lines[:5])
    tail = "\n".join(lines[-5:]) if len(lines) > 10 else ""
    snippet = f"file size: {len(file_text)} bytes, {len(lines)} lines\nfirst 5 lines:\n{head}"
    if tail:
        snippet += f"\n...\nlast 5 lines:\n{tail}"
    return (
        f"old_string not found in {path} (edit #{edit_index}). Your old_string"
        f" does not match the file content byte-for-byte and no close region"
        f" exists, so it likely targets the wrong file or a stale expectation."
        f" Re-read with read_file, then retry with a shorter, uniquely-anchored"
        f" old_string. File shape (orientation only, do NOT use as old_string"
        f" verbatim):\n{snippet}"
    )
