# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""VerifyVerdict transitions: the streak/reset rules every consumer relies on."""

from __future__ import annotations

from agent6.workflows._verify_verdict import VerifyVerdict


def test_pass_resets_streak_and_covers_earlier_edits() -> None:
    v = VerifyVerdict()
    v.note_edit()
    v.note_fail("sig-a")
    v.note_pass()
    assert v.last_ok is True and v.ever_passed and v.ever_failed
    assert v.edited_since is False
    assert v.fail_streak == 0 and v.fail_signature == ""
    assert v.green_and_untouched


def test_same_signature_extends_the_streak_a_new_one_restarts_it() -> None:
    v = VerifyVerdict()
    v.note_fail("sig-a")
    v.note_fail("sig-a")
    assert v.fail_streak == 2
    v.note_fail("sig-b")  # a NEW stuck point
    assert v.fail_streak == 1 and v.fail_signature == "sig-b"


def test_a_red_verdict_covers_its_tree_like_a_green_one() -> None:
    """A green clears edited_since (the verdict covers the current tree); a red
    now does the same, so the harness does not re-run the gate on a red tree
    nothing has touched and does not count that red twice."""
    v = VerifyVerdict()
    v.note_edit()
    v.note_fail("sig")
    assert v.judged_and_untouched  # the red judged the tree as it stands
    assert not v.green_and_untouched  # but it is not green
    v.note_edit()
    assert not v.judged_and_untouched  # an edit moves the tree past the verdict


def test_an_edit_after_green_withdraws_the_green() -> None:
    v = VerifyVerdict()
    v.note_pass()
    assert v.green_and_untouched
    v.note_edit()
    assert not v.green_and_untouched
    assert v.last_ok is True  # the observation stands; only its coverage lapsed
