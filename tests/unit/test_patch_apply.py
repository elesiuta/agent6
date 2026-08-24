# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Tests for agent6.tools.patch_apply — parser + applier."""

from __future__ import annotations

import pytest

from agent6.tools.patch_apply import (
    PatchError,
    apply_patch_text,
    parse_patch,
)


def test_parse_simple_single_hunk() -> None:
    patch = "--- a/foo.py\n+++ b/foo.py\n@@ -1,3 +1,3 @@\n a\n-b\n+B\n c\n"
    p = parse_patch(patch)
    assert p.target_path == "foo.py"
    assert p.is_create is False
    assert len(p.hunks) == 1


def test_apply_replace_one_line() -> None:
    original = "a\nb\nc\n"
    patch = "--- a/foo.py\n+++ b/foo.py\n@@ -1,3 +1,3 @@\n a\n-b\n+B\n c\n"
    path, new, _healed = apply_patch_text(patch, original)
    assert path == "foo.py"
    assert new == "a\nB\nc\n"


def test_apply_multi_hunk_offset_tracking() -> None:
    original = "1\n2\n3\n4\n5\n6\n7\n8\n9\n"
    # First hunk inserts a line near the top; second hunk replaces near the
    # bottom. The second hunk's `old_start` is in original-file coordinates,
    # so the applier must track the cumulative offset (+1) from the first.
    patch = (
        "--- a/n.txt\n"
        "+++ b/n.txt\n"
        "@@ -1,3 +1,4 @@\n"
        " 1\n"
        "+1.5\n"
        " 2\n"
        " 3\n"
        "@@ -7,3 +8,3 @@\n"
        " 7\n"
        "-8\n"
        "+EIGHT\n"
        " 9\n"
    )
    _, new, _healed = apply_patch_text(patch, original)
    assert new == "1\n1.5\n2\n3\n4\n5\n6\n7\nEIGHT\n9\n"


def test_apply_pure_insertion_hunk() -> None:
    # An insertion anchored on a context line (@@ -3,1 +3,3 @@): old_count == 1,
    # so the `else` arm of the buf_start branch.
    original = "a\nb\nc\n"
    patch = "--- a/f.txt\n+++ b/f.txt\n@@ -3,1 +3,3 @@\n c\n+x\n+y\n"
    _, new, _healed = apply_patch_text(patch, original)
    assert new == "a\nb\nc\nx\ny\n"


def test_apply_pure_deletion_hunk() -> None:
    original = "a\nb\nc\n"
    patch = "--- a/f.txt\n+++ b/f.txt\n@@ -1,3 +1,2 @@\n a\n-b\n c\n"
    _, new, _healed = apply_patch_text(patch, original)
    assert new == "a\nc\n"


def test_create_via_dev_null() -> None:
    patch = "--- /dev/null\n+++ b/new.py\n@@ -0,0 +1,2 @@\n+x = 1\n+y = 2\n"
    path, new, _healed = apply_patch_text(patch, None)
    assert path == "new.py"
    assert new == "x = 1\ny = 2\n"


def test_create_when_file_exists_errors() -> None:
    patch = "--- /dev/null\n+++ b/exists.py\n@@ -0,0 +1,1 @@\n+x = 1\n"
    with pytest.raises(PatchError, match="already exists"):
        apply_patch_text(patch, "old contents\n")


def test_context_mismatch_errors_with_helpful_message() -> None:
    original = "a\nDIFFERENT\nc\n"
    patch = "--- a/f.py\n+++ b/f.py\n@@ -1,3 +1,3 @@\n a\n-b\n+B\n c\n"
    with pytest.raises(PatchError, match="Context mismatch"):
        apply_patch_text(patch, original)


def test_missing_file_errors() -> None:
    patch = "--- a/missing.py\n+++ b/missing.py\n@@ -1,1 +1,1 @@\n-a\n+b\n"
    with pytest.raises(PatchError, match="does not exist"):
        apply_patch_text(patch, None)


def test_delete_via_plus_dev_null() -> None:
    """`+++ /dev/null` deletes: new_content None, path from the `---` header.
    The hunks must remove the ENTIRE on-disk content (the patch asserts what
    it deletes); surviving content is a hard error, and file-vs-patch
    mismatch fails the ordinary context check."""
    assert apply_patch_text("--- a/f.py\n+++ /dev/null\n@@ -1,1 +0,0 @@\n-a\n", "a\n") == (
        "f.py",
        None,
        (),
    )
    with pytest.raises(PatchError, match="entire file"):
        apply_patch_text("--- a/f.py\n+++ /dev/null\n@@ -1,1 +0,0 @@\n-a\n", "a\nb\n")


def test_multi_file_patch_rejected() -> None:
    patch = (
        "--- a/one.py\n"
        "+++ b/one.py\n"
        "@@ -1,1 +1,1 @@\n"
        "-a\n"
        "+A\n"
        "--- a/two.py\n"
        "+++ b/two.py\n"
        "@@ -1,1 +1,1 @@\n"
        "-b\n"
        "+B\n"
    )
    with pytest.raises(PatchError, match="Multi-file"):
        apply_patch_text(patch, "a\n")


def test_single_file_patch_removing_dash_dash_comment_not_multifile() -> None:
    # Regression: a `-`-removal of a line whose content begins with `-- ` (a
    # SQL/Lua/Haskell comment) encodes as `--- ...`. The old raw `--- ` pre-scan
    # wrongly rejected this legitimate single-file patch as multi-file.
    original = "SELECT 1;\n-- a comment\n"
    patch = "--- a/x.sql\n+++ b/x.sql\n@@ -1,2 +1,1 @@\n SELECT 1;\n--- a comment\n"
    path, new, _healed = apply_patch_text(patch, original)
    assert path == "x.sql"
    assert new == "SELECT 1;\n"


def test_hunk_header_count_mismatch_rejected() -> None:
    # Header says 3 old lines but body only supplies 2.
    patch = "--- a/f.py\n+++ b/f.py\n@@ -1,3 +1,3 @@\n a\n-b\n+B\n"
    with pytest.raises(PatchError, match="declares 3"):
        apply_patch_text(patch, "a\nb\n")


def test_skips_git_diff_preamble() -> None:
    # `git diff` emits `diff --git ...` and `index ...` lines before `---`.
    patch = (
        "diff --git a/f.py b/f.py\n"
        "index abc..def 100644\n"
        "--- a/f.py\n"
        "+++ b/f.py\n"
        "@@ -1,1 +1,1 @@\n"
        "-a\n"
        "+A\n"
    )
    _, new, _healed = apply_patch_text(patch, "a\n")
    assert new == "A\n"


def test_no_newline_at_eof_on_old_side() -> None:
    # Original lacks a trailing newline; replacement adds one.
    original = "a\nb"
    patch = "--- a/f.py\n+++ b/f.py\n@@ -1,2 +1,2 @@\n a\n-b\n\\ No newline at end of file\n+B\n"
    _, new, _healed = apply_patch_text(patch, original)
    assert new == "a\nB\n"


def test_no_newline_at_eof_on_new_side() -> None:
    original = "a\nb\n"
    patch = "--- a/f.py\n+++ b/f.py\n@@ -1,2 +1,2 @@\n a\n-b\n+B\n\\ No newline at end of file\n"
    _, new, _healed = apply_patch_text(patch, original)
    assert new == "a\nB"


def test_omitted_count_means_one() -> None:
    # `@@ -1 +1 @@` is shorthand for `@@ -1,1 +1,1 @@`.
    patch = "--- a/f.py\n+++ b/f.py\n@@ -1 +1 @@\n-a\n+A\n"
    _, new, _healed = apply_patch_text(patch, "a\n")
    assert new == "A\n"


def test_empty_patch_rejected() -> None:
    with pytest.raises(PatchError, match="Empty"):
        apply_patch_text("", None)


def test_bare_path_header_no_a_b_prefix() -> None:
    # Accept patches without the conventional `a/`/`b/` prefix.
    patch = "--- f.py\n+++ f.py\n@@ -1 +1 @@\n-a\n+A\n"
    path, new, _healed = apply_patch_text(patch, "a\n")
    assert path == "f.py"
    assert new == "A\n"


# --- OpenAI V4A "*** Begin Patch" parser/applier ----------------------------

from agent6.tools.patch_apply import apply_v4a_text, is_v4a_patch, patch_target_path  # noqa: E402


def test_v4a_detect_and_target_path() -> None:
    patch = "*** Begin Patch\n*** Update File: pkg/m.py\n@@\n-a\n+b\n*** End Patch"
    assert is_v4a_patch(patch)
    assert patch_target_path(patch) == "pkg/m.py"


def test_v4a_update_context_hunk() -> None:
    orig = "def f():\n    x = 1\n    return x\n"
    patch = (
        "*** Begin Patch\n*** Update File: m.py\n@@ def f():\n"
        "     x = 1\n-    return x\n+    return x + 1\n*** End Patch"
    )
    path, new, _healed = apply_v4a_text(patch, orig)
    assert path == "m.py"
    assert new == "def f():\n    x = 1\n    return x + 1\n"


def test_v4a_multi_hunk() -> None:
    orig = "a = 1\nb = 2\nc = 3\nd = 4\ne = 5\n"
    patch = (
        "*** Begin Patch\n*** Update File: x.py\n"
        "@@\n a = 1\n-b = 2\n+b = 20\n@@\n d = 4\n-e = 5\n+e = 50\n*** End Patch"
    )
    _, new, _healed = apply_v4a_text(patch, orig)
    assert new == "a = 1\nb = 20\nc = 3\nd = 4\ne = 50\n"


def test_v4a_add_file() -> None:
    patch = "*** Begin Patch\n*** Add File: n.py\n+print(1)\n+print(2)\n*** End Patch"
    path, new, _healed = apply_v4a_text(patch, None)
    assert path == "n.py" and new == "print(1)\nprint(2)\n"


def test_v4a_ambiguous_context_rejected() -> None:
    patch = "*** Begin Patch\n*** Update File: a.py\n@@\n-x = 1\n+x = 2\n*** End Patch"
    with pytest.raises(PatchError, match="ambiguous"):
        apply_v4a_text(patch, "x = 1\nx = 1\n")


def test_v4a_section_hint_disambiguates_repeated_block() -> None:
    # `return 1` appears in both a() and b(); the `@@ def b():` section marker
    # pins the edit to b() instead of failing as ambiguous (the locator-hint bug).
    orig = "def a():\n    return 1\n\ndef b():\n    return 1\n"
    patch = (
        "*** Begin Patch\n*** Update File: m.py\n"
        "@@ def b():\n-    return 1\n+    return 2\n*** End Patch"
    )
    _, new, _healed = apply_v4a_text(patch, orig)
    assert new == "def a():\n    return 1\n\ndef b():\n    return 2\n"


def test_v4a_section_hint_that_does_not_resolve_stays_ambiguous() -> None:
    # The hint is present but the block still repeats AFTER it, so we must refuse
    # rather than guess which occurrence to edit (safety: never mis-apply).
    orig = "class C:\n    x = 1\n    x = 1\n"
    patch = (
        "*** Begin Patch\n*** Update File: m.py\n@@ class C:\n-    x = 1\n+    x = 2\n*** End Patch"
    )
    with pytest.raises(PatchError, match="ambiguous"):
        apply_v4a_text(patch, orig)


def test_v4a_context_not_found_rejected() -> None:
    patch = "*** Begin Patch\n*** Update File: a.py\n@@\n-missing\n+x\n*** End Patch"
    with pytest.raises(PatchError, match="not found"):
        apply_v4a_text(patch, "different\n")


def test_v4a_multi_file_rejected() -> None:
    patch = (
        "*** Begin Patch\n*** Update File: a.py\n@@\n-x\n+y\n"
        "*** Update File: b.py\n@@\n-p\n+q\n*** End Patch"
    )
    with pytest.raises(PatchError, match="one file at a time"):
        apply_v4a_text(patch, "x\n")


def test_v4a_delete() -> None:
    """`*** Delete File:` deletes by name (that format asserts no content):
    bare directive only, and the file must exist."""
    assert apply_v4a_text("*** Begin Patch\n*** Delete File: a.py\n*** End Patch", "x\n") == (
        "a.py",
        None,
        (),
    )
    with pytest.raises(PatchError, match="no such file"):
        apply_v4a_text("*** Begin Patch\n*** Delete File: a.py\n*** End Patch", None)
    with pytest.raises(PatchError, match="bare directive"):
        apply_v4a_text("*** Begin Patch\n*** Delete File: a.py\n@@\n-x\n*** End Patch", "x\n")


def test_v4a_partial_line_match_rejected_not_spliced() -> None:
    """A `-` line that is only a SUBSTRING of a longer on-disk line must not
    match: substring `.replace` spliced mid-line and silently corrupted the file
    (`-x = 1` against `x = 10` produced `0`). Matching is line-anchored."""
    patch = "*** Begin Patch\n*** Update File: a.py\n@@\n-x = 1\n*** End Patch"
    with pytest.raises(PatchError, match="context not found"):
        apply_v4a_text(patch, "x = 10\n")


def test_v4a_straddling_block_rejected() -> None:
    """A multi-line block whose first line straddles a longer on-disk line
    (`-value = 1` inside `myvalue = 1`) must not match."""
    patch = "*** Begin Patch\n*** Update File: a.py\n@@\n-value = 1\n-b\n+c\n*** End Patch"
    with pytest.raises(PatchError, match="context not found"):
        apply_v4a_text(patch, "myvalue = 1\nb\n")


def test_v4a_full_line_delete_still_applies() -> None:
    """Line-anchoring must not break a legitimate whole-line match."""
    patch = "*** Begin Patch\n*** Update File: a.py\n@@\n-x = 1\n+x = 2\n*** End Patch"
    assert apply_v4a_text(patch, "x = 1\n") == ("a.py", "x = 2\n", ())


def test_v4a_end_of_file_marker_accepted() -> None:
    """GPT emits `*** End of File` for a hunk reaching EOF; it is a marker, not a
    hunk line, so it must be dropped rather than raising 'Unexpected V4A line'."""
    patch = (
        "*** Begin Patch\n*** Update File: m.py\n@@\n last\n+added\n*** End of File\n*** End Patch"
    )
    assert apply_v4a_text(patch, "last\n") == ("m.py", "last\nadded\n", ())


def test_v4a_pure_deletion_removes_the_lines_whole() -> None:
    """A hunk with only `-` lines deleted the text but left the newline that
    terminated the last removed line, so the file kept a stray blank line where
    the deletion happened (deleting every line left the file as a lone "\\n")."""
    orig = "line1\nline2\nline3\nline4\n"
    patch = "*** Begin Patch\n*** Update File: m.py\n@@\n-line2\n-line3\n*** End Patch"
    _, new, _healed = apply_v4a_text(patch, orig)
    assert new == "line1\nline4\n"


def test_v4a_pure_deletion_of_the_whole_file_empties_it() -> None:
    orig = "only\n"
    patch = "*** Begin Patch\n*** Update File: m.py\n@@\n-only\n*** End Patch"
    _, new, _healed = apply_v4a_text(patch, orig)
    assert new == ""


def test_v4a_deletion_with_context_is_unaffected() -> None:
    """The context-anchored form already worked; it must keep working."""
    orig = "line1\nline2\nline3\nline4\n"
    patch = (
        "*** Begin Patch\n*** Update File: m.py\n@@\n line1\n-line2\n-line3\n line4\n*** End Patch"
    )
    _, new, _healed = apply_v4a_text(patch, orig)
    assert new == "line1\nline4\n"


def test_unified_hunk_heals_a_uniform_indent_shift() -> None:
    """A hunk whose lines are right but uniformly mis-indented applies with
    the replacement re-indented to the file's actual depth, and the heal is
    reported (`~indent`). One transform must explain every line; a mixed
    shift stays a context error."""
    original = "def f():\n    if x:\n        a = 1\n        b = 2\n"
    patch = (
        "--- a/f.py\n+++ b/f.py\n@@ -3,2 +3,2 @@\n"
        "-    a = 1\n-    b = 2\n+    a = 10\n+    b = 20\n"
    )
    _, new, healed = apply_patch_text(patch, original)
    assert new == "def f():\n    if x:\n        a = 10\n        b = 20\n"
    assert healed == ("f.py @@ -3,2 ~indent",)


def test_unified_hunk_heals_trailing_whitespace() -> None:
    original = "a  \nb\n"
    patch = "--- a/f.py\n+++ b/f.py\n@@ -1,1 +1,1 @@\n-a\n+A\n"
    _, new, healed = apply_patch_text(patch, original)
    assert new == "A\nb\n"
    assert healed == ("f.py @@ -1,1 ~rstrip",)


def test_unified_hunk_heals_stale_line_numbers_when_unique() -> None:
    """Stale anchors with exact content heal only at EXACTLY ONE match."""
    original = "x\ny\nz\ntarget\nw\n"
    patch = "--- a/f.py\n+++ b/f.py\n@@ -1,1 +1,1 @@\n-target\n+TARGET\n"
    _, new, healed = apply_patch_text(patch, original)
    assert new == "x\ny\nz\nTARGET\nw\n"
    assert healed == ("f.py @@ -1,1 ~moved",)


def test_unified_heal_refuses_ambiguity() -> None:
    original = "dup\nmid\ndup\n"
    patch = "--- a/f.py\n+++ b/f.py\n@@ -2,1 +2,1 @@\n-dup\n+DUP\n"
    with pytest.raises(PatchError, match="Context mismatch"):
        apply_patch_text(patch, original)


def test_v4a_hunk_heals_a_uniform_indent_shift() -> None:
    patch = "*** Begin Patch\n*** Update File: a.py\n@@\n-a = 1\n+a = 10\n*** End Patch"
    _, new, healed = apply_v4a_text(patch, "def f():\n    a = 1\n")
    assert new == "def f():\n    a = 10\n"
    assert healed == ("a.py ~indent",)


def test_v4a_heal_refuses_a_second_indent_candidate() -> None:
    patch = "*** Begin Patch\n*** Update File: a.py\n@@\n-a = 1\n+a = 10\n*** End Patch"
    with pytest.raises(PatchError, match="context not found"):
        apply_v4a_text(patch, "def f():\n    a = 1\ndef g():\n        a = 1\n")


def test_moved_heal_tail_state_follows_the_healed_position() -> None:
    """`touches_tail` was computed from the hunk's stale header numbers, not
    the healed position: a moved hunk relocated TO the tail kept the old
    trailing newline, and one relocated AWAY from the tail flipped it.
    The tail test reads the actual splice location."""
    # Stale header says lines 2-3; the exact block lives at EOF (lines 4-5).
    original = "x\ny\na\nb"  # no trailing newline
    patch = "--- a/f.py\n+++ b/f.py\n@@ -2,2 +2,2 @@\n a\n-b\n+B\n"
    _, new, healed = apply_patch_text(patch, original)
    assert healed == ("f.py @@ -2,2 ~moved",)
    # The replaced range IS the tail: the no-trailing-newline state of the
    # original tail must survive (stale coordinates said "not tail" and
    # re-added a newline).
    assert new == "x\ny\na\nB"

    # The inverse: stale header claims the tail, the block lives mid-file.
    original2 = "a\nb\nx\ny\n"
    patch2 = "--- a/f.py\n+++ b/f.py\n@@ -3,2 +3,2 @@\n a\n-b\n+B\n\\ No newline at end of file\n"
    _, new2, healed2 = apply_patch_text(patch2, original2)
    assert healed2 == ("f.py @@ -3,2 ~moved",)
    # The tail (y + trailing newline) was untouched by the healed mid-file
    # edit; the patch's no-newline marker must not strip the file's tail.
    assert new2 == "a\nB\nx\ny\n"
