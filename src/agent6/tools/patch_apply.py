# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Strict unified-diff parser and applier (single file per patch).

Accepts standard `diff -u` output:

    --- a/path/to/file
    +++ b/path/to/file
    @@ -OLD_START,OLD_COUNT +NEW_START,NEW_COUNT @@
     context
    -removed
    +added
     context

Design choices (pre-1.0):

- Multi-file patches are accepted at the tool layer (`split_patch_files`
  cuts them at `diff --git` / V4A file-directive boundaries; the caller
  applies per file, all-or-nothing). The parse/apply functions here stay
  single-file. A bare multi-file unified diff WITHOUT `diff --git`
  separators is still rejected: a `--- ` line is indistinguishable from a
  removal of `-- comment` content without hunk-structural context.
- Near-zero fuzz. A hunk matches exactly, or heals through a strict
  ladder (trailing whitespace; a single uniform indent shift, byte-
  verified, replacement re-indented to match; a unique exact match away
  from stale line numbers) with the heal named on the wire. Ambiguity
  and every looser miss stay hard errors; if any hunk fails, no change
  is written (all-or-nothing).
- `--- /dev/null` is allowed and means "create a new file"; the target
  file must not already exist.
- `+++ /dev/null` deletes the file; the hunk body must remove the entire
  on-disk content (the patch asserts what it deletes). V4A
  `*** Delete File:` deletes by name, per that format's grammar.
- The `\\ No newline at end of file` marker is honoured: when present
  on the `-` side, the original file must lack a trailing newline; on
  the `+` side, the result is written without one.
- Hunk-header line counts are validated; an inconsistent header is a
  hard error, not silently fixed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_HUNK_RE = re.compile(
    r"^@@ -(?P<old_start>\d+)(?:,(?P<old_count>\d+))? "
    r"\+(?P<new_start>\d+)(?:,(?P<new_count>\d+))? @@"
)


class PatchError(ValueError):
    """The patch could not be parsed or could not be applied cleanly."""


@dataclass(frozen=True, slots=True)
class _Hunk:
    old_start: int  # 1-based line in original file
    old_count: int
    new_start: int  # 1-based line in resulting file (informational)
    new_count: int
    # Each entry is (prefix, text). prefix is one of " ", "-", "+".
    # text has no trailing newline.
    body: tuple[tuple[str, str], ...]
    # Whether the last "-"-or-" " line lacks a trailing newline in the original.
    old_no_newline: bool
    # Whether the last "+"-or-" " line lacks a trailing newline in the result.
    new_no_newline: bool


@dataclass(frozen=True, slots=True)
class ParsedPatch:
    """A successfully-parsed single-file unified diff."""

    # Path from the `+++` header with the leading `b/` (if any) stripped.
    # For file creation (`--- /dev/null`), this is the new file's path; for
    # deletion (`+++ /dev/null`), the old file's path from the `---` header.
    target_path: str
    # True if the patch creates a new file (i.e. `--- /dev/null`).
    is_create: bool
    hunks: tuple[_Hunk, ...]
    # True if the patch deletes the file (i.e. `+++ /dev/null`); the applied
    # result must be empty, and the caller unlinks instead of writing.
    is_delete: bool = False


# ---------- parsing ----------


def _strip_ab_prefix(header_path: str) -> str:
    """Strip the conventional `a/` or `b/` prefix from a diff header path.

    `--- a/foo.py` and `+++ b/foo.py` are the format `git diff` emits.
    Some models also emit bare `--- foo.py`. Accept both.
    """
    if header_path.startswith(("a/", "b/")):
        return header_path[2:]
    return header_path


def parse_patch(text: str) -> ParsedPatch:  # noqa: PLR0912, PLR0915
    """Parse a single-file unified diff. Raises PatchError on malformed input."""
    if not text.strip():
        raise PatchError("Empty patch")

    lines = text.splitlines()
    # Locate the `---` and `+++` headers. Skip leading commentary lines (e.g.
    # `diff --git a/foo b/foo`, `index abc..def 100644`).
    i = 0
    while i < len(lines) and not lines[i].startswith("--- "):
        i += 1
    if i >= len(lines):
        raise PatchError("Missing `--- ` header line")
    minus_header = lines[i][4:].strip()
    i += 1
    if i >= len(lines) or not lines[i].startswith("+++ "):
        raise PatchError("Missing `+++ ` header line after `--- ` header")
    plus_header = lines[i][4:].strip()
    i += 1

    is_create = minus_header == "/dev/null"
    is_delete = plus_header == "/dev/null"
    if is_create and is_delete:
        raise PatchError("a patch cannot both create and delete (`/dev/null` on both sides)")

    target_path = _strip_ab_prefix(minus_header if is_delete else plus_header)
    if not target_path or target_path == "/dev/null":
        raise PatchError(f"Invalid target path in `+++` header: {plus_header!r}")

    # NOTE: multi-file patches are rejected structurally, where a hunk header
    # is expected (see the `_HUNK_RE` miss below). We must NOT pre-scan raw
    # lines for `--- ` here: a removal line whose *content* begins with `-- `
    # (a SQL/Lua/Haskell/Ada comment, say) is encoded as `-` + `-- foo` =
    # `--- foo` inside a hunk body, and a raw scan would wrongly reject the
    # legitimate single-file patch as multi-file.
    hunks: list[_Hunk] = []
    while i < len(lines):
        line = lines[i]
        if not line.strip():
            i += 1
            continue
        # A trailing `\ No newline at end of file` marker may live between the
        # last `+`/`-` line of the previous hunk and the next hunk header
        # (or end of input). Attribute it to the most recent hunk.
        if line.startswith("\\ "):
            if not hunks:
                raise PatchError("`\\ No newline` marker has no preceding hunk")
            prev = hunks[-1]
            last_prefix = prev.body[-1][0] if prev.body else " "
            new_old = prev.old_no_newline or last_prefix in ("-", " ")
            new_new = prev.new_no_newline or last_prefix in ("+", " ")
            hunks[-1] = _Hunk(
                old_start=prev.old_start,
                old_count=prev.old_count,
                new_start=prev.new_start,
                new_count=prev.new_count,
                body=prev.body,
                old_no_newline=new_old,
                new_no_newline=new_new,
            )
            i += 1
            continue
        m = _HUNK_RE.match(line)
        if not m:
            # A `--- ` line where a hunk header is expected is a real second
            # file's header (vs a `-`-removal of `-- ...` content, which is
            # consumed inside a hunk body above and never reaches here).
            if line.startswith("--- "):
                raise PatchError("Multi-file patches are not supported; submit one file at a time")
            raise PatchError(f"Expected hunk header `@@ -L,N +L,N @@`, got: {line!r}")
        old_start = int(m.group("old_start"))
        old_count = int(m.group("old_count")) if m.group("old_count") is not None else 1
        new_start = int(m.group("new_start"))
        new_count = int(m.group("new_count")) if m.group("new_count") is not None else 1
        i += 1
        body: list[tuple[str, str]] = []
        old_no_newline = False
        new_no_newline = False
        seen_old = 0
        seen_new = 0
        while i < len(lines) and (seen_old < old_count or seen_new < new_count):
            ln = lines[i]
            if ln.startswith("\\ "):
                # "\ No newline at end of file", applies to the immediately
                # preceding line. Determine which side based on its prefix.
                if not body:
                    raise PatchError("`\\ No newline` marker has no preceding line")
                prev_prefix, _ = body[-1]
                if prev_prefix == "-":
                    old_no_newline = True
                elif prev_prefix == "+":
                    new_no_newline = True
                else:  # " " context line — applies to both sides
                    old_no_newline = True
                    new_no_newline = True
                i += 1
                continue
            if not ln:
                # Empty line is a legitimate context line (encoded as " " + "").
                # Some patch producers strip the leading space on otherwise-empty
                # lines; accept both shapes.
                body.append((" ", ""))
                seen_old += 1
                seen_new += 1
                i += 1
                continue
            prefix = ln[0]
            text_part = ln[1:]
            if prefix == " ":
                body.append((" ", text_part))
                seen_old += 1
                seen_new += 1
            elif prefix == "-":
                body.append(("-", text_part))
                seen_old += 1
            elif prefix == "+":
                body.append(("+", text_part))
                seen_new += 1
            else:
                raise PatchError(
                    f"Unexpected line in hunk body (expected ` `, `-`, `+`, `\\ `): {ln!r}"
                )
            i += 1
        if seen_old != old_count or seen_new != new_count:
            raise PatchError(
                f"Hunk header @@ -{old_start},{old_count} +{new_start},{new_count} @@ "
                f"declares {old_count}/{new_count} lines but body has {seen_old}/{seen_new}"
            )
        hunks.append(
            _Hunk(
                old_start=old_start,
                old_count=old_count,
                new_start=new_start,
                new_count=new_count,
                body=tuple(body),
                old_no_newline=old_no_newline,
                new_no_newline=new_no_newline,
            )
        )

    if not hunks:
        raise PatchError("Patch contains no hunks")
    return ParsedPatch(
        target_path=target_path, is_create=is_create, hunks=tuple(hunks), is_delete=is_delete
    )


# ---------- application ----------


def _split_lines_keepends(text: str) -> tuple[list[str], bool]:
    """Split *text* into lines without trailing newlines; track final-newline state."""
    if text == "":
        return [], False
    has_trailing = text.endswith("\n")
    lines = text.split("\n")
    if has_trailing:
        # Final element after split is "", drop it.
        lines.pop()
    return lines, has_trailing


def apply_parsed_patch(  # noqa: PLR0912
    patch: ParsedPatch, original: str | None
) -> tuple[str, tuple[str, ...]]:
    """Apply *patch* to *original* file contents (None means file does not exist).

    Returns `(new_content, healed)`: `healed` names each hunk the matcher
    healed rather than matched exactly (`rstrip` trailing whitespace,
    `indent` a uniform leading-whitespace shift, `moved` a unique exact
    match away from the anchored line numbers), for the wire to report.
    Raises PatchError on any context mismatch or impossible-to-apply hunk.
    All-or-nothing: caller writes the returned string.
    """
    if patch.is_create:
        if original is not None:
            raise PatchError(
                f"Patch declares file creation (`--- /dev/null`) but "
                f"{patch.target_path!r} already exists"
            )
        base_lines: list[str] = []
        base_had_trailing = False
    else:
        if original is None:
            raise PatchError(
                f"Patch targets {patch.target_path!r} but the file does not exist; "
                f"use `--- /dev/null` to create it"
            )
        base_lines, base_had_trailing = _split_lines_keepends(original)

    # Work on a mutable copy. Apply hunks in original order; track the cumulative
    # offset between original-file line numbers and current-buffer line numbers.
    buf = list(base_lines)
    healed: list[str] = []
    offset = 0  # buf_index = original_index + offset
    # Track whether the final newline should be present after all hunks have been
    # applied. Starts at the file's current state; a hunk that touches the last
    # line can flip it.
    result_has_trailing = base_had_trailing

    for hunk in patch.hunks:
        # Map the hunk's 1-based original line to a 0-based buffer index.
        # Special case: a pure-insertion hunk has `old_count == 0` and its
        # `old_start` is the line number *after which* to insert (0 meaning
        # "at the very beginning"). For `old_count > 0`, `old_start` is the
        # 1-based first line of the replaced range.
        buf_start = hunk.old_start + offset if hunk.old_count == 0 else hunk.old_start - 1 + offset
        if buf_start < 0 or buf_start + hunk.old_count > len(buf):
            raise PatchError(
                f"Hunk @@ -{hunk.old_start},{hunk.old_count} @@ for "
                f"{patch.target_path!r} reaches outside the file "
                f"(file has {len(base_lines)} lines)"
            )

        expected_old: list[str] = []
        replacement_new: list[str] = []
        for prefix, txt in hunk.body:
            if prefix in (" ", "-"):
                expected_old.append(txt)
            if prefix in (" ", "+"):
                replacement_new.append(txt)

        actual_old = buf[buf_start : buf_start + hunk.old_count]
        moved_heal = False
        if actual_old != expected_old:
            heal = _heal_hunk(buf, buf_start, expected_old, replacement_new)
            if heal is None:
                raise PatchError(
                    f"Context mismatch in {patch.target_path!r} at "
                    f"hunk @@ -{hunk.old_start},{hunk.old_count} @@.\n"
                    f"Expected lines:\n{_render_lines(expected_old)}\n"
                    f"On-disk lines:\n{_render_lines(actual_old)}"
                )
            buf_start, replacement_new, kind = heal
            moved_heal = kind == "moved"
            # The label leads with the file like the V4A one: a multi-file
            # patch's "healed" list is unattributable without it.
            healed.append(f"{patch.target_path} @@ -{hunk.old_start},{hunk.old_count} ~{kind}")

        # Determine whether this hunk touches the file's tail from the ACTUAL
        # splice position: a `moved` heal relocates `buf_start` away from the
        # stale header numbers, which once kept a hunk healed onto the tail
        # from carrying its no-newline state (and vice versa).
        touches_tail = (
            buf_start == len(buf)
            if hunk.old_count == 0
            else (buf_start + hunk.old_count) == len(buf)
        )
        buf[buf_start : buf_start + hunk.old_count] = replacement_new
        offset += hunk.new_count - hunk.old_count
        if touches_tail:
            # The hunk's `new_no_newline` flag is authoritative for the result.
            # If the hunk didn't declare a no-newline marker on the new side,
            # the result has a trailing newline (standard diff convention).
            # A `moved` heal is the exception: it lands where the authored
            # coordinates never pointed, so the patch expresses no EOF intent
            # there; the file's tail state stands unless an explicit marker
            # travels with the block.
            if moved_heal:
                if hunk.new_no_newline:
                    result_has_trailing = False
            else:
                result_has_trailing = not hunk.new_no_newline

    if not buf:
        # Empty file, write empty string regardless of trailing-newline state.
        return "", tuple(healed)
    out = "\n".join(buf)
    if result_has_trailing:
        out += "\n"
    return out, tuple(healed)


def _common_shift(actual: list[str], expected: list[str]) -> tuple[str, str] | None:
    """The single uniform leading-whitespace transform turning *expected* into
    *actual*: (strip_prefix, add_prefix), byte-verified over every non-blank
    line. None when no one transform explains all of them."""
    if len(actual) != len(expected):
        return None
    transform: tuple[str, str] | None = None
    for act, exp in zip(actual, expected, strict=True):
        if act == exp:
            continue
        if not act.strip() and not exp.strip():
            continue
        exp_body = exp.lstrip()
        act_body = act.lstrip()
        if exp_body != act_body:
            return None
        pair = (exp[: len(exp) - len(exp_body)], act[: len(act) - len(act_body)])
        if transform is None:
            transform = pair
        elif transform != pair:
            return None
    return transform


def _reindent(lines: list[str], strip: str, add: str) -> list[str]:
    out: list[str] = []
    for ln in lines:
        if ln.startswith(strip) and ln.strip():
            out.append(add + ln[len(strip) :])
        else:
            out.append(ln)
    return out


def _heal_hunk(
    buf: list[str], buf_start: int, expected_old: list[str], replacement_new: list[str]
) -> tuple[int, list[str], str] | None:
    """The context-miss ladder for one anchored hunk, strictest first.

    (new_buf_start, new_replacement, kind) or None. The passes mirror the
    field (Codex heals trailing whitespace and location; apply_edit heals a
    uniform indent shift) with this repo's uniqueness discipline:

    - `rstrip`: on-disk lines equal modulo trailing whitespace, in place.
    - `indent`: ONE leading-whitespace transform explains every line, in
      place; the replacement is re-indented by the same transform.
    - `moved`: the exact expected block exists at EXACTLY ONE other position
      (stale line numbers); ambiguity stays a hard error.
    """
    count = len(expected_old)
    actual = buf[buf_start : buf_start + count]
    if len(actual) == count and [a.rstrip() for a in actual] == [e.rstrip() for e in expected_old]:
        return buf_start, replacement_new, "rstrip"
    shift = _common_shift(actual, expected_old) if len(actual) == count else None
    if shift is not None:
        strip, add = shift
        if _reindent(expected_old, strip, add) == actual:
            return buf_start, _reindent(replacement_new, strip, add), "indent"
    if count:
        hits = [i for i in range(len(buf) - count + 1) if buf[i : i + count] == expected_old]
        if len(hits) == 1:
            return hits[0], replacement_new, "moved"
    return None


def _render_lines(lines: list[str]) -> str:
    if not lines:
        return "  (empty)"
    return "\n".join(f"  {i + 1}| {ln}" for i, ln in enumerate(lines))


def apply_patch_text(
    patch_text: str, original: str | None
) -> tuple[str, str | None, tuple[str, ...]]:
    """Convenience: parse + apply. Returns (target_path, new_content, healed);
    new_content None means the patch deletes the file (its hunks removed
    the entire content, verified), and healed names each hunk the matcher
    healed rather than matched exactly."""
    patch = parse_patch(patch_text)
    new_content, healed = apply_parsed_patch(patch, original)
    if patch.is_delete:
        if new_content != "":
            raise PatchError(
                "a deletion patch (`+++ /dev/null`) must remove the entire file; "
                f"{len(new_content)} chars of content survive the hunks"
            )
        return patch.target_path, None, healed
    return patch.target_path, new_content, healed


# ---------- OpenAI "*** Begin Patch" (V4A) format ----------
#
# GPT / gpt-oss models emit patches in OpenAI's apply_patch format, NOT unified
# diff: `*** Begin Patch` / `*** End Patch` wrap one or more file directives
# (`*** Add File:` / `*** Update File:` / `*** Delete File:`); inside an Update,
# hunks use ` `/`-`/`+` line prefixes with optional `@@ <hint>` section markers
# and NO `@@ -L,N +L,N @@` line numbers (matching is by context, not position).
# Without this, every apply_patch from a GPT-family model fails ("got: '@@'")
# and the model death-spirals on re-reads. We map each context hunk onto the
# same safe unique-substring replacement apply_edit uses: zero fuzz, all-or-
# nothing, and a clear error when context is missing or ambiguous.


def is_v4a_patch(text: str) -> bool:
    """True if *text* looks like an OpenAI `*** Begin Patch` envelope."""
    return text.lstrip().startswith("*** Begin Patch")


def split_patch_files(text: str) -> list[str]:
    """Cut a possibly-multi-file patch into single-file patch texts.

    V4A: one section per `*** Add/Update/Delete File:` directive, each
    re-wrapped in its own envelope. Unified: one section per `diff --git `
    boundary (a hunk body line always carries a +/-/space prefix, so a
    column-0 `diff --git ` is only ever a file boundary). A single-file
    patch is returned as-is; the per-file parsers keep their own
    multi-file guards for anything a split cannot see.
    """
    if is_v4a_patch(text):
        raw = text.strip().splitlines()
        if not raw or raw[0].strip() != "*** Begin Patch" or raw[-1].strip() != "*** End Patch":
            return [text]  # malformed envelope: let apply_v4a_text name the error
        body = raw[1:-1]
        starts = [i for i, ln in enumerate(body) if _v4a_file_directive(ln) is not None]
        if len(starts) <= 1:
            return [text]
        sections = []
        for n, i in enumerate(starts):
            end = starts[n + 1] if n + 1 < len(starts) else len(body)
            sections.append("\n".join(["*** Begin Patch", *body[i:end], "*** End Patch"]))
        return sections
    lines = text.splitlines(keepends=True)
    starts = [i for i, ln in enumerate(lines) if ln.startswith("diff --git ")]
    if len(starts) <= 1:
        return [text]
    sections = []
    for n, i in enumerate(starts):
        end = starts[n + 1] if n + 1 < len(starts) else len(lines)
        sections.append("".join(lines[i:end]))
    return sections


def patch_target_path(text: str) -> str:
    """Extract the single target path from a patch (either format) without
    applying it. Raises `PatchError` if no path header is present."""
    if is_v4a_patch(text):
        for ln in text.splitlines():
            d = _v4a_file_directive(ln)
            if d is not None:
                return d[1]
        raise PatchError("V4A patch has no `*** Add/Update/Delete File:` directive")
    minus = ""
    for ln in text.splitlines():
        if ln.startswith("--- ") and not minus:
            minus = ln[4:].strip()
        if ln.startswith("+++ "):
            header = ln[4:].strip()
            if header == "/dev/null":  # deletion: the path lives in the `---` header
                if minus and minus != "/dev/null":
                    return _strip_ab_prefix(minus)
                raise PatchError("deletion patch has no `--- ` header to take a path from")
            return _strip_ab_prefix(header)
    raise PatchError("patch has no `+++ ` header to take a path from")


def _v4a_file_directive(line: str) -> tuple[str, str] | None:
    """Parse a `*** <Verb> File: <path>` directive into (verb, path), else None."""
    for verb in ("Add", "Update", "Delete"):
        prefix = f"*** {verb} File:"
        if line.startswith(prefix):
            return verb, line[len(prefix) :].strip()
    return None


def _v4a_delete(path: str, section: list[str], original: str | None) -> tuple[str, None]:
    """`*** Delete File:` is the bare directive: no content, file must exist."""
    if any(ln.strip() for ln in section):
        raise PatchError(
            f"V4A `*** Delete File: {path}` carries content; a deletion is the bare directive"
        )
    if original is None:
        raise PatchError(f"V4A `*** Delete File: {path}` but no such file exists")
    return path, None


def apply_v4a_text(
    patch_text: str, original: str | None
) -> tuple[str, str | None, tuple[str, ...]]:
    """Parse and apply a single-file OpenAI V4A patch.

    Returns `(target_path, new_content, healed)`; None content means
    `*** Delete File:` (that format deletes by name, no content assertion),
    and `healed` names each hunk the matcher healed rather than matched
    exactly. Raises `PatchError` on a malformed envelope, a multi-file
    patch, a missing/ambiguous context, or a file create/update mismatch.
    All-or-nothing: the caller writes (or unlinks) from the returned value.
    """
    raw = patch_text.strip().splitlines()
    if not raw or raw[0].strip() != "*** Begin Patch":
        raise PatchError("V4A patch must start with `*** Begin Patch`")
    if raw[-1].strip() != "*** End Patch":
        raise PatchError("V4A patch must end with `*** End Patch`")
    body = raw[1:-1]

    # Locate the single file directive. Multiple are rejected (one file per call,
    # same as the unified-diff applier).
    directives = [(i, _v4a_file_directive(ln)) for i, ln in enumerate(body)]
    file_starts = [(i, d) for i, d in directives if d is not None]
    if not file_starts:
        raise PatchError("V4A patch has no `*** Add/Update/Delete File:` directive")
    if len(file_starts) > 1:
        raise PatchError("Multi-file V4A patches are not supported; submit one file at a time")
    start_idx, (verb, path) = file_starts[0]
    if not path:
        raise PatchError("V4A file directive is missing a path")
    section = body[start_idx + 1 :]
    # Drop a `*** Move to:` line (rename; we only honour the content change at the
    # original path) and the optional `*** End of File` marker GPT emits for a
    # hunk that reaches EOF (our matching is whole-file, so it needs no anchor).
    section = [
        ln
        for ln in section
        if not ln.startswith("*** Move to:") and ln.strip() != "*** End of File"
    ]

    if verb == "Delete":
        p, none = _v4a_delete(path, section, original)
        return p, none, ()

    if verb == "Add":
        if original is not None:
            raise PatchError(f"V4A `*** Add File: {path}` but the file already exists")
        return path, _v4a_added_content(path, section), ()

    # Update File.
    if original is None:
        raise PatchError(f"V4A `*** Update File: {path}` but the file does not exist")
    return _v4a_apply_update(path, section, original)


def _v4a_added_content(path: str, section: list[str]) -> str:
    """An Add File body is all `+` lines (blank lines tolerated)."""
    added: list[str] = []
    for ln in section:
        if ln.startswith("+"):
            added.append(ln[1:])
        elif ln.strip() == "":
            added.append("")
        else:
            raise PatchError(f"V4A Add File body must be all `+` lines, got: {ln!r}")
    return "\n".join(added) + "\n" if added else ""


def _v4a_apply_update(
    path: str, section: list[str], original: str
) -> tuple[str, str, tuple[str, ...]]:
    """Apply an Update File body: locate each hunk (healing per the ladder),
    splice, and report `(path, content, healed)`."""
    hunks = _v4a_split_hunks(section)
    if not hunks:
        raise PatchError(f"V4A `*** Update File: {path}` has no hunks")
    content = original
    healed: list[str] = []
    for hints, old_block, new_block in hunks:
        if old_block == "":
            raise PatchError(
                f"V4A hunk for {path!r} has no context/removed lines to anchor on; "
                "include the surrounding lines so the change can be located"
            )
        matches = _line_anchored_indices(content, old_block)
        count = len(matches)
        if count == 0:
            heal = _v4a_heal(content, old_block, new_block)
            if heal is None:
                raise PatchError(
                    f"V4A hunk context not found in {path!r}. The ` `/`-` lines must match "
                    f"the file byte-for-byte. Closest-anchor failed; re-read and retry.\n"
                    f"Expected block:\n{_render_lines(old_block.split(chr(10)))}"
                )
            content, kind = heal
            healed.append(f"{path} ~{kind}")
            continue
        if count == 1:
            content = _v4a_splice(content, matches[0], old_block, new_block)
            continue
        # The block itself repeats; the `@@ <section>` hints disambiguate it. We
        # only apply when the hints pin a SINGLE occurrence -- otherwise the hunk
        # stays ambiguous and we refuse rather than edit the wrong copy.
        idx = _v4a_locate_with_hints(content, hints, old_block)
        if idx is None:
            raise PatchError(
                f"V4A hunk context is ambiguous in {path!r} ({count} matches); include "
                "more surrounding context lines, or a `@@ <section>` marker naming the "
                "enclosing def/class, so the location is unique"
            )
        content = _v4a_splice(content, idx, old_block, new_block)
    return path, content, tuple(healed)


def _v4a_heal(content: str, old_block: str, new_block: str) -> tuple[str, str] | None:
    """The V4A context-miss ladder, strictest first, uniqueness required:
    `rstrip` (on-disk lines equal modulo trailing whitespace) then `indent`
    (one leading-whitespace transform explains every line; the new block is
    re-indented the same way). Ambiguity or anything looser stays a miss."""
    expected = old_block.split("\n")
    lines = content.split("\n")
    count = len(expected)
    rstrip_hits: list[int] = []
    indent_hits: list[tuple[int, tuple[str, str]]] = []
    for i in range(len(lines) - count + 1):
        window = lines[i : i + count]
        if [w.rstrip() for w in window] == [e.rstrip() for e in expected]:
            rstrip_hits.append(i)
            continue
        shift = _common_shift(window, expected)
        if shift is not None and _reindent(expected, *shift) == window:
            indent_hits.append((i, shift))

    def _splice_lines(i: int, new_lines: list[str]) -> str:
        return "\n".join(lines[:i] + new_lines + lines[i + count :])

    if len(rstrip_hits) == 1:
        return _splice_lines(rstrip_hits[0], new_block.split("\n") if new_block else []), "rstrip"
    if not rstrip_hits and len(indent_hits) == 1:
        i, (strip, add) = indent_hits[0]
        new_lines = _reindent(new_block.split("\n"), strip, add) if new_block else []
        return _splice_lines(i, new_lines), "indent"
    return None


def _v4a_splice(content: str, idx: int, old_block: str, new_block: str) -> str:
    """Replace the block at *idx* with *new_block*.

    The blocks are line TEXT with no trailing newline, so a pure deletion (an
    empty new block) must take the newline that terminated the last removed
    line with it -- leaving it behind put a stray blank line where the deletion
    happened, and deleting every line left the file as a lone newline."""
    rest = content[idx + len(old_block) :]
    if not new_block and rest.startswith("\n"):
        rest = rest[1:]
    return content[:idx] + new_block + rest


def _v4a_split_hunks(section: list[str]) -> list[tuple[tuple[str, ...], str, str]]:
    """Split a V4A Update body into `(hints, old_block, new_block)` tuples.

    A `@@ <text>` line is a section LOCATOR HINT for the hunk that follows: its
    text (typically a `def`/`class` line) names the enclosing region, used to
    disambiguate when the hunk's own context lines repeat elsewhere in the file.
    One or more `@@` lines may precede a hunk; an empty `@@` is a bare hunk
    separator with no hint. Within a hunk, ` `/`-` lines build the old block and
    ` `/`+` lines build the new block.
    """
    hunks: list[tuple[list[str], list[str], list[str]]] = []
    cur_hints: list[str] = []
    cur_old: list[str] = []
    cur_new: list[str] = []

    def flush() -> None:
        nonlocal cur_hints, cur_old, cur_new
        if cur_old or cur_new:
            hunks.append((cur_hints, cur_old, cur_new))
        cur_hints, cur_old, cur_new = [], [], []

    for ln in section:
        if ln.startswith("@@"):
            # A `@@` after hunk content starts a NEW hunk; flush the current one
            # first (which clears its hints). Then record this `@@`'s text as a
            # locator hint for the hunk now beginning (empty `@@` = no hint).
            if cur_old or cur_new:
                flush()
            hint = ln[2:].strip()
            if hint:
                cur_hints.append(hint)
            continue
        if ln == "" or ln.startswith(" "):
            text = ln[1:] if ln.startswith(" ") else ""
            cur_old.append(text)
            cur_new.append(text)
        elif ln.startswith("-"):
            cur_old.append(ln[1:])
        elif ln.startswith("+"):
            cur_new.append(ln[1:])
        else:
            raise PatchError(f"Unexpected V4A hunk line (expected ` `, `-`, `+`, `@@`): {ln!r}")
    flush()
    return [(tuple(h), "\n".join(o), "\n".join(n)) for h, o, n in hunks]


def _v4a_locate_with_hints(content: str, hints: tuple[str, ...], old_block: str) -> int | None:
    """Index at which to apply *old_block* when it occurs more than once, using
    the `@@` *hints* to disambiguate. Returns None when the hints do not
    resolve it to a SINGLE location (the caller then reports the hunk as
    ambiguous -- we never guess which occurrence to edit). Each hint must appear
    in order; `old_block` must then occur exactly once at or after the last
    hint's position."""
    search_from = 0
    last_hint_pos = 0
    for hint in hints:
        pos = content.find(hint, search_from)
        if pos == -1:
            return None
        last_hint_pos = pos
        search_from = pos + len(hint)
    after = [i for i in _line_anchored_indices(content, old_block) if i >= last_hint_pos]
    if len(after) != 1:
        return None
    return after[0]


def _line_anchored_indices(content: str, block: str) -> list[int]:
    """Start indices where *block* occurs aligned to line boundaries: the match
    must begin at BOF or just after a newline and end at EOF or just before a
    newline. A V4A hunk block is always whole lines, so a substring match that
    straddles a line boundary (`-x = 1` inside `x = 10`) is a false positive
    that would splice mid-line and silently corrupt the file."""
    out: list[int] = []
    start = 0
    width = len(block)
    while True:
        i = content.find(block, start)
        if i == -1:
            break
        at_start = i == 0 or content[i - 1] == "\n"
        end = i + width
        at_end = end == len(content) or content[end] == "\n"
        if at_start and at_end:
            out.append(i)
        start = i + 1
    return out
