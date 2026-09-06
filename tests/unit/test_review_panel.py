# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Unit tests for the grounded review-panel aggregator (the false-block defense).

These lock in the property that got the pre-0.0.4 reviewer.py deleted once it was
fixed here: a reviewer can only GATE (block a finish) when its objection is
grounded in the actual diff AND in a category we allow to block. Taste, test
gaps, and uncited claims are mechanically downgraded and can never stall a run.
"""

from __future__ import annotations

from typing import Any

from agent6.workflows._panel import (
    Finding,
    PanelResult,
    ReviewContext,
    ReviewDecision,
    ReviewVerdict,
    _cite_path,  # pyright: ignore[reportPrivateUsage]
    aggregate_verdicts,
    diff_touched_ranges,
    is_grounded,
    render_findings,
)

# A real unified diff touching foo.py new-lines 10..14 and creating bar.py 1..2.
SAMPLE_DIFF = """\
--- a/foo.py
+++ b/foo.py
@@ -10,3 +10,5 @@ def f():
     x = 1
+    y = 2
+    z = 3
     return x
--- /dev/null
+++ b/bar.py
@@ -0,0 +1,2 @@
+import os
+VALUE = 1
"""


def _ctx(**kw: Any) -> ReviewContext:
    return ReviewContext(diff=SAMPLE_DIFF, **kw)


def _seat(
    model: str, *findings: Finding, seat: str = "s", error: str | None = None
) -> ReviewVerdict:
    verdict = "block" if any(f.severity == "block" for f in findings) else "pass"
    return ReviewVerdict(seat=seat, model=model, verdict=verdict, findings=findings, error=error)


def _block(category: str, file_line: str) -> Finding:
    return Finding(category=category, severity="block", file_line=file_line, title="x")


def _agg(
    seats: list[ReviewVerdict],
    *,
    decision: ReviewDecision = "veto",
    quorum: int = 2,
    ctx: ReviewContext | None = None,
) -> PanelResult:
    return aggregate_verdicts(seats, ctx or _ctx(), decision=decision, quorum=quorum, panel_id="p")


# --- diff grounding primitives ------------------------------------------------


def test_diff_touched_ranges_parses_paths_and_new_line_ranges() -> None:
    ranges = diff_touched_ranges(SAMPLE_DIFF)
    # New-side range plus the old-side range of the same hunk (pre-image line
    # numbers must ground too); a created file has no old side.
    assert ranges["foo.py"] == [(10, 14), (10, 12)]
    assert ranges["bar.py"] == [(1, 2)]


def test_is_grounded_line_in_range_path_only_and_misses() -> None:
    ranges = diff_touched_ranges(SAMPLE_DIFF)
    assert is_grounded("foo.py:11", ranges)  # inside 10..14
    assert is_grounded("foo.py", ranges)  # path-only, file touched
    assert not is_grounded("foo.py:99", ranges)  # outside the touched range
    assert not is_grounded("other.py:1", ranges)  # file not in the diff
    assert not is_grounded("", ranges)


def test_is_grounded_accepts_a_line_col_citation() -> None:
    """`path:line:col` is the standard compiler/grep -n location a reviewer
    copies. The single rpartition read the COLUMN as the line and the rest as
    the path, so the lookup missed and a real block was silently downgraded to
    a warning."""
    ranges = diff_touched_ranges(SAMPLE_DIFF)
    assert is_grounded("foo.py:11:5", ranges)  # line 11 is inside 10..14
    assert not is_grounded("foo.py:99:5", ranges)  # column must not rescue it
    assert _cite_path("foo.py:11:5") == "foo.py"  # dedup keys on the same path


# --- executable grounding in aggregation --------------------------------------


def test_grounded_security_block_gates_under_veto() -> None:
    res = _agg([_seat("m1", _block("security", "foo.py:11"))], decision="veto")
    assert res.blocked is True and res.n_block == 1
    assert res.merged_findings[0].severity == "block"


def test_ungrounded_block_is_downgraded_and_does_not_gate() -> None:
    # cites a line the diff never touched -> downgraded to warn -> no gate.
    res = _agg([_seat("m1", _block("security", "foo.py:99"))], decision="veto")
    assert res.blocked is False and res.n_block == 0
    assert res.merged_findings[0].severity == "warn"


def test_non_gating_category_block_is_downgraded() -> None:
    # a "test-gap" can never block even if grounded in the diff.
    res = _agg([_seat("m1", _block("test-gap", "foo.py:11"))], decision="veto")
    assert res.blocked is False
    assert res.merged_findings[0].severity == "warn"


def test_verify_uncovered_requires_verify_passed() -> None:
    f = _block("verify-uncovered-correctness", "foo.py:11")
    # verify failed (or unknown) -> the claim is incoherent -> downgraded.
    assert _agg([_seat("m1", f)], ctx=_ctx(verify_ok=False)).blocked is False
    assert _agg([_seat("m1", f)], ctx=_ctx(verify_ok=None)).blocked is False
    # verify passed -> a grounded uncovered-correctness block may gate.
    assert _agg([_seat("m1", f)], ctx=_ctx(verify_ok=True)).blocked is True


# --- decision policies --------------------------------------------------------


def test_advisory_never_blocks_even_with_grounded_security_block() -> None:
    res = _agg([_seat("m1", _block("security", "foo.py:11"))], decision="advisory")
    assert res.blocked is False
    assert res.merged_findings[0].severity == "block"  # still reported, just not gating


def test_quorum_counts_distinct_models_not_seats() -> None:
    g = lambda: _block("security", "foo.py:11")  # noqa: E731
    # two blocking seats but the SAME model -> counts as one -> quorum(2) unmet.
    same = [_seat("m1", g(), seat="a"), _seat("m1", g(), seat="b")]
    assert _agg(same, decision="quorum", quorum=2).blocked is False
    assert _agg(same, decision="quorum", quorum=2).n_block == 1
    # distinct models -> quorum met.
    diff_models = [_seat("m1", g(), seat="a"), _seat("m2", g(), seat="b")]
    assert _agg(diff_models, decision="quorum", quorum=2).blocked is True


def test_all_requires_every_non_abstaining_seat_to_block() -> None:
    blk = _seat("m1", _block("security", "foo.py:11"), seat="a")
    passing = _seat("m2", seat="b")  # no findings -> pass
    assert _agg([blk, passing], decision="all").blocked is False
    assert _agg(
        [blk, _seat("m2", _block("security", "bar.py:1"), seat="b")], decision="all"
    ).blocked


def test_all_lone_blocker_with_mostly_errored_panel_does_not_gate() -> None:
    # FINDING 1 regression: a 5-seat panel where 4 seats abstained (provider
    # error) and ONE seat blocks must NOT gate under "all" -- "all" means the
    # panel that actually reviewed unanimously agreed, and one vote is not a
    # quorum of five. Previously this blocked because abstentions were filtered
    # out before the all(...) check, so all(...) ran over the single survivor.
    blk = _seat("m1", _block("security", "foo.py:11"), seat="a")
    errs = [_seat(f"m{i}", seat=f"s{i}", error="provider timeout") for i in range(2, 6)]
    res = _agg([blk, *errs], decision="all")
    assert res.n_abstain == 4 and res.n_block == 1
    assert res.blocked is False  # no majority quorum responded
    # the same lone grounded block DOES gate under veto (one block is enough there).
    assert _agg([blk, *errs], decision="veto").blocked is True


def test_abstain_does_not_count_as_pass_or_block() -> None:
    blk = _seat("m1", _block("security", "foo.py:11"), seat="a")
    err = _seat("m2", seat="b", error="provider timeout")
    err2 = _seat("m3", seat="c", error="unparseable")
    # "all" must NOT gate when only a minority responded: a lone blocker with
    # everyone else abstaining is not unanimous agreement of the panel.
    res = _agg([blk, err], decision="all")  # 1 of 2 responded -> no majority quorum
    assert res.n_abstain == 1
    assert res.blocked is False
    # even more lopsided: one block, two abstentions -> still no gate.
    res3 = _agg([blk, err, err2], decision="all")
    assert res3.n_abstain == 2 and res3.blocked is False
    # but a strict majority responding and unanimously blocking still gates.
    blk2 = _seat("m2", _block("security", "bar.py:1"), seat="b")
    res4 = _agg([blk, blk2, err2], decision="all")  # 2 of 3 responded, both block
    assert res4.n_abstain == 1 and res4.blocked is True
    # under veto, a single grounded block still gates regardless of abstentions.
    assert _agg([blk, err], decision="veto").blocked is True
    # an all-abstain panel never blocks
    only_err = _agg([err], decision="veto")
    assert only_err.blocked is False and only_err.n_abstain == 1


# --- dedup / rendering --------------------------------------------------------


def test_dedup_across_seats_and_against_prior_findings() -> None:
    f = _block("security", "foo.py:11")
    res = _agg([_seat("m1", f, seat="a"), _seat("m2", f, seat="b")], decision="advisory")
    assert len(res.merged_findings) == 1  # same (path, category) merged
    prior = (Finding("security", "block", "foo.py:11", "already shown"),)
    res2 = aggregate_verdicts(
        [_seat("m1", f)], _ctx(prior_findings=prior), decision="advisory", quorum=2, panel_id="p"
    )
    assert res2.merged_findings == ()  # already injected -> not re-surfaced


def test_prior_deduped_block_does_not_count_toward_the_gate() -> None:
    # A seat whose only surviving block dedups away against prior_findings must
    # not gate: otherwise blocked=True ships with merged_findings=() and the
    # worker is rejected while being told "No blocking findings.".
    f = _block("security", "foo.py:11")
    prior = (Finding("security", "block", "foo.py:11", "already shown"),)
    for decision in ("veto", "quorum", "all"):
        res = aggregate_verdicts(
            [_seat("m1", f)], _ctx(prior_findings=prior), decision=decision, quorum=1, panel_id="p"
        )
        assert res.merged_findings == ()
        assert res.n_block == 0 and res.blocked is False, decision
    # A NEW grounded block alongside the deduped one still gates.
    new = _block("data-loss", "bar.py:1")
    res2 = aggregate_verdicts(
        [_seat("m1", f, new)], _ctx(prior_findings=prior), decision="veto", quorum=2, panel_id="p"
    )
    assert res2.blocked is True and res2.n_block == 1
    assert [x.category for x in res2.merged_findings] == ["data-loss"]


def test_render_findings_formats_and_empty() -> None:
    assert render_findings(()) == ""
    out = render_findings((Finding("security", "block", "foo.py:11", "leak", "fix it"),))
    assert "[block:security]" in out and "foo.py:11" in out and "leak" in out and "fix it" in out


# --- diff-parsing edge cases (regressions fixed in the pre-squash review) ------


def test_added_line_starting_like_a_header_is_not_a_file_header() -> None:
    # An added line whose CONTENT begins with "++ b/evil.py" renders as
    # "+++ b/evil.py"; it must not be mistaken for a +++ header (only a +++ that
    # follows a --- is one). A LATER hunk follows so the misparse would actually
    # re-attribute a range to "evil.py" if the prev_minus guard were dropped.
    diff = (
        "--- a/foo.py\n+++ b/foo.py\n"
        "@@ -1,2 +1,3 @@\n keep\n+++ b/evil.py\n+real = 1\n"
        "@@ -10,2 +10,3 @@\n ctx\n+added\n more\n"
    )
    ranges = diff_touched_ranges(diff)
    assert "evil.py" not in ranges
    assert ranges["foo.py"] == [(1, 3), (1, 2), (10, 12), (10, 11)]  # new + old side per hunk


def test_deleted_line_starting_like_a_header_is_not_a_file_header() -> None:
    # A DELETED line whose CONTENT begins with "-- " renders as "--- ..." and must
    # not be mistaken for a "--- " file header (the symmetric "+++ " side was
    # already guarded; the "--- " side was not). Otherwise it clobbers the path
    # and every LATER hunk's range is mis-attributed, so a grounded block citing a
    # real line in a later hunk is silently downgraded to a warning.
    diff = (
        "--- a/schema.sql\n+++ b/schema.sql\n"
        "@@ -10,3 +10,2 @@\n CREATE TABLE t (\n--- legacy column note\n   id INT\n"
        "@@ -50,2 +50,3 @@\n cols\n+  api_key TEXT\n more\n"
    )
    ranges = diff_touched_ranges(diff)
    assert "legacy column note" not in ranges  # the deletion was not read as a header
    assert is_grounded("schema.sql:51", ranges)  # the later hunk still grounds


def test_in_place_modification_grounds_old_side_lines() -> None:
    # A hunk that deletes lines from a kept (not renamed) file: a block citing
    # the deleted code at its OLD line number must ground. Previously the
    # old-side range was recorded only when oldpath != newpath, so such a
    # citation was ungrounded and the block silently downgraded to warn (the
    # gate failed open on reviews of deleted code).
    diff = "--- a/mod.py\n+++ b/mod.py\n@@ -100,5 +50,2 @@\n ctx\n-gone1\n-gone2\n-gone3\n ctx2\n"
    ranges = diff_touched_ranges(diff)
    assert (100, 104) in ranges["mod.py"]  # old side of the in-place hunk
    assert is_grounded("mod.py:103", ranges)
    res = _agg(
        [_seat("m1", _block("data-loss", "mod.py:103"))],
        decision="veto",
        ctx=ReviewContext(diff=diff),
    )
    assert res.blocked is True and res.n_block == 1
    assert res.merged_findings[0].severity == "block"


def test_pure_deletion_grounds_on_the_old_path() -> None:
    # A file deleted entirely (post-image /dev/null) must still ground a citation
    # of the deleted file so a data-loss/off-topic block on it can gate.
    diff = "--- a/gone.py\n+++ /dev/null\n@@ -1,3 +0,0 @@\n-a\n-b\n-c\n"
    ranges = diff_touched_ranges(diff)
    assert ranges["gone.py"] == [(1, 3)]
    assert is_grounded("gone.py:2", ranges)


def test_grounding_tolerates_trailing_colon_and_line_range() -> None:
    ranges = diff_touched_ranges(SAMPLE_DIFF)
    assert is_grounded("foo.py:11:", ranges)  # ripgrep-style trailing colon
    assert is_grounded("foo.py:11-13", ranges)  # a range fully inside 10..14


def test_grounding_range_overlap_not_just_start_line() -> None:
    # foo.py changed lines 10..14. A range whose START line is unchanged but whose
    # INTERIOR overlaps the touched range must still ground (FINDING 2 regression:
    # previously only the start line was checked, so this was wrongly ungrounded).
    ranges = diff_touched_ranges(SAMPLE_DIFF)
    assert is_grounded("foo.py:8-12", ranges)  # start 8 untouched, but 10..12 overlap
    assert is_grounded("foo.py:13-20", ranges)  # end 20 untouched, but 13..14 overlap
    assert is_grounded("foo.py:1-99", ranges)  # span fully contains the touched range
    assert is_grounded("foo.py:14-30", ranges)  # touches only the last changed line
    assert not is_grounded("foo.py:1-9", ranges)  # entirely before the touched range
    assert not is_grounded("foo.py:15-30", ranges)  # entirely after the touched range
    assert is_grounded("foo.py:12-10", ranges)  # reversed range normalized, still grounds


def test_range_block_with_unchanged_start_still_gates() -> None:
    # End-to-end: a grounded block citing a real range whose start line is
    # unchanged must keep blocking (not be silently downgraded to warn).
    res = _agg([_seat("m1", _block("security", "foo.py:8-12"))], decision="veto")
    assert res.blocked is True and res.n_block == 1
    assert res.merged_findings[0].severity == "block"


def test_all_abstain_panel_prints_inconclusive_not_pass(monkeypatch: Any, capsys: Any) -> None:
    """3 seats, 3 abstains, real dollars spent, ZERO review produced -- and the
    command printed "VERDICT: PASS". Nothing was reviewed; "0 blocking" is not
    a verdict. (The gate itself is fine: run_panel short-circuits on an
    all-abstain panel. The PRINTED verdict was the lie, so this pins the CLI.)"""
    from typing import cast

    from agent6.budget import BudgetTracker
    from agent6.config import Config
    from agent6.ui.cli import review_cmds

    class _Seat:
        persona = "security"
        tier = "diff"

    abstain = ReviewVerdict(
        seat="security",
        model="moonshotai/kimi-k3",
        verdict="pass",
        error="unparseable reviewer output",
    )
    res = PanelResult(
        panel_id="cli",
        decision="advisory",
        blocked=False,
        merged_findings=(),
        per_seat=(abstain, abstain, abstain),
        n_block=0,
        n_abstain=3,
    )

    def _fake_seats(*_a: object, **_k: object) -> list[_Seat]:
        return [_Seat(), _Seat(), _Seat()]

    def _fake_panel(*_a: object, **_k: object) -> PanelResult:
        return res

    monkeypatch.setattr(review_cmds, "build_review_seats", _fake_seats)
    monkeypatch.setattr(review_cmds, "run_panel", _fake_panel)
    rc = review_cmds._run_review_panel(  # pyright: ignore[reportPrivateUsage]
        Config(),
        base="",
        diff="d",
        agents_md="",
        reviewers=3,
        personas="security,correctness,tests",
        model_override="",
        transcript_sink=cast(Any, object()),  # only handed to the mocked seat builder
        budget=BudgetTracker(max_usd=-1, max_tokens_fallback=-1, max_percent=-1),
    )
    out, _err = capsys.readouterr()
    assert "VERDICT: PASS" not in out
    assert "INCONCLUSIVE" in out
    assert "abstained" in out  # the why is on the verdict line, not buried in stderr
    assert rc == 1  # a panel that reviewed nothing is not a success


def test_review_exit_code_is_consistent_across_verdicts(monkeypatch: Any, capsys: Any) -> None:
    """The exit code carried the verdict for INCONCLUSIVE (1) but left BLOCK at
    0 -- a CI gate passed a security block and failed on 'nothing reviewed'.
    PASS 0, INCONCLUSIVE 1, BLOCK 2, consistently."""
    from typing import cast

    from agent6.budget import BudgetTracker
    from agent6.config import Config
    from agent6.ui.cli import review_cmds

    class _Seat:
        persona = "security"
        tier = "diff"

    block = _seat("m1", _block("security", "foo.py:11"))  # a real gating verdict
    passing = _seat("m2", seat="s2")

    def run(
        per_seat: tuple[ReviewVerdict, ...], *, blocked: bool, findings: tuple[Any, ...]
    ) -> int:
        res = PanelResult(
            panel_id="cli",
            decision="veto",
            blocked=blocked,
            merged_findings=findings,
            per_seat=per_seat,
            n_block=1 if blocked else 0,
            n_abstain=sum(1 for v in per_seat if v.error),
        )

        def _seats(*_a: object, **_k: object) -> list[_Seat]:
            return [_Seat()]

        def _panel(*_a: object, **_k: object) -> PanelResult:
            return res

        monkeypatch.setattr(review_cmds, "build_review_seats", _seats)
        monkeypatch.setattr(review_cmds, "run_panel", _panel)
        rc = review_cmds._run_review_panel(  # pyright: ignore[reportPrivateUsage]
            Config(),
            base="",
            diff="d",
            agents_md="",
            reviewers=1,
            personas="security",
            model_override="",
            transcript_sink=cast(Any, object()),
            budget=BudgetTracker(max_usd=-1, max_tokens_fallback=-1, max_percent=-1),
        )
        capsys.readouterr()
        return rc

    assert run((passing,), blocked=False, findings=()) == 0  # PASS
    assert run((block,), blocked=True, findings=block.findings) == 2  # BLOCK
    abstain = _seat("m3", seat="s3", error="starved")
    assert run((abstain,), blocked=False, findings=()) == 1  # INCONCLUSIVE


def test_output_cap_truncated_case_folds_both_spellings() -> None:
    from agent6.providers import ProviderResponse, output_cap_truncated

    def r(stop: str) -> ProviderResponse:
        return ProviderResponse(
            text="",
            tool_uses=(),
            stop_reason=stop,
            input_tokens=1,
            output_tokens=1,
            cache_read_tokens=0,
            cache_creation_tokens=0,
        )

    assert output_cap_truncated(r("length"))
    assert output_cap_truncated(r("max_tokens"))  # anthropic spelling
    assert output_cap_truncated(r("MAX_TOKENS"))  # upper-casing gateway
    assert not output_cap_truncated(r("end_turn"))
    assert not output_cap_truncated(r("tool_use"))


def test_panel_is_inconclusive_owner() -> None:
    from agent6.workflows._panel import panel_is_inconclusive

    abstain = ReviewVerdict(seat="s", model="m", verdict="pass", error="starved")
    passing = ReviewVerdict(seat="s2", model="m2", verdict="pass")
    all_abstain = PanelResult(
        panel_id="p",
        decision="advisory",
        blocked=False,
        merged_findings=(),
        per_seat=(abstain, abstain),
        n_block=0,
        n_abstain=2,
    )
    assert panel_is_inconclusive(all_abstain) is True
    mixed = PanelResult(
        panel_id="p",
        decision="advisory",
        blocked=False,
        merged_findings=(),
        per_seat=(abstain, passing),
        n_block=0,
        n_abstain=1,
    )
    assert panel_is_inconclusive(mixed) is False
    empty = PanelResult(
        panel_id="p",
        decision="advisory",
        blocked=False,
        merged_findings=(),
        per_seat=(),
        n_block=0,
        n_abstain=0,
    )
    assert panel_is_inconclusive(empty) is False  # nothing to be inconclusive about


def test_review_degrades_on_an_unreadable_agents_md(
    monkeypatch: Any, tmp_path: Any, capsys: Any
) -> None:
    """AGENTS.md is optional review context (the run path reads it tolerantly);
    an unreadable one crashed `agent6 review` through the bug reporter instead
    of reviewing without it."""
    from types import SimpleNamespace

    from agent6.config import Config
    from agent6.ui.cli import review_cmds

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("AGENT6_STATE_HOME", str(tmp_path / "state"))
    agents = tmp_path / "AGENTS.md"
    agents.write_text("# rules\n", encoding="utf-8")
    agents.chmod(0o000)

    seen: dict[str, str] = {}

    def _fake_panel(_cfg: Config, **kwargs: Any) -> int:
        seen["agents_md"] = kwargs["agents_md"]
        return 0

    def _fake_effective(*_a: object, **_k: object) -> SimpleNamespace:
        return SimpleNamespace(config=Config())

    def _runnable(_self: Config, _role: str) -> None:
        return None

    def _no_key_error(_cfg: Config) -> None:
        return None

    def _fake_diff(*_a: object, **_k: object) -> SimpleNamespace:
        return SimpleNamespace(returncode=0, stdout="diff --git a/x\n+1\n", stderr="")

    monkeypatch.setattr(review_cmds, "load_effective", _fake_effective)
    monkeypatch.setattr(Config, "require_runnable", _runnable)
    monkeypatch.setattr(review_cmds, "check_provider_keys", _no_key_error)
    monkeypatch.setattr(review_cmds, "_collect_review_diff", _fake_diff)
    monkeypatch.setattr(review_cmds, "_run_review_panel", _fake_panel)
    try:
        rc = review_cmds._cmd_review(  # pyright: ignore[reportPrivateUsage]
            None, base="", head="", paths=(), reviewers=1
        )
    finally:
        agents.chmod(0o600)
    assert rc == 0
    assert seen["agents_md"] == ""  # reviewed without the unreadable context


def test_diff_touched_ranges_records_a_file_touched_without_hunks() -> None:
    """A binary change, a pure rename and a mode flip carry no hunks, so the
    file was absent from the map: a path-only citation of it was ungrounded
    and its block downgraded to a warning. The path is recorded with no
    ranges: grounded by path, not by line."""
    diff = (
        "diff --git a/img.png b/img.png\n"
        "index 1111111..2222222 100644\n"
        "Binary files a/img.png and b/img.png differ\n"
        "diff --git a/old.py b/new.py\n"
        "similarity index 100%\n"
        "rename from old.py\n"
        "rename to new.py\n"
        "diff --git a/run.sh b/run.sh\n"
        "old mode 100644\n"
        "new mode 100755\n"
    )
    ranges = diff_touched_ranges(diff)
    assert ranges == {"img.png": [], "old.py": [], "new.py": [], "run.sh": []}
    assert is_grounded("img.png", ranges) and is_grounded("new.py", ranges)
    assert not is_grounded("img.png:3", ranges)


def test_the_seat_prompt_says_verify_was_not_run_without_a_result() -> None:
    """`agent6 review` runs no verify command, and the loop has none to run
    when none is configured; the prompt told the seats "none configured" in
    both cases, wrong for a review of a repo that has one."""
    from agent6.workflows._review import _build_user_message  # pyright: ignore[reportPrivateUsage]

    prompt = _build_user_message(ReviewContext(task="t"))
    assert "VERIFY: not run." in prompt and "none configured" not in prompt
