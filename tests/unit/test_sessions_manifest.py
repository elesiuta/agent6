# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""runs.manifest: the typed SessionManifest reader. Every failure shape (missing,
unreadable, corrupt JSON, torn UTF-8, non-object) degrades through the typed
ManifestError; every historical run dir (old ``version: 1`` shapes, the pre-v2
flat merged_* keys, the legacy ``compare.group``) still parses for rendering;
and the fork/resume ``session_mode`` gate refuses an unknown mode rather than
falling open to write access."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent6.sessions.manifest import MANIFEST_VERSION, ManifestError, read_manifest

_DATA = Path(__file__).parent / "data"


def _write(session_dir: Path, payload: object) -> None:
    (session_dir / "manifest.json").write_text(json.dumps(payload), encoding="utf-8")


def test_reads_a_valid_manifest(tmp_path: Path) -> None:
    _write(tmp_path, {"session_id": "r-1", "mode": "plan", "base_sha": "abc"})
    m = read_manifest(tmp_path)
    assert m.session_id == "r-1"
    assert m.mode == "plan"
    assert m.base_sha == "abc"


def test_missing_fields_default_so_any_old_dir_renders(tmp_path: Path) -> None:
    # An almost-empty manifest still parses: every field defaults.
    _write(tmp_path, {})
    m = read_manifest(tmp_path)
    assert m.version == MANIFEST_VERSION
    # mode has NO default: it is the privilege gate's input (see the
    # fall-open test below), so an absent key stays absent.
    assert m.mode == ""
    assert m.run_branch is None
    assert m.models.driver is None
    assert m.merged is None and m.compare is None


def test_legacy_version_1_and_missing_profile(tmp_path: Path) -> None:
    # A real pre-reshape dir: version 1, workflow without `preset`.
    _write(
        tmp_path,
        {
            "version": 1,
            "mode": "run",
            "user_task": "do a thing",
            "workflow": {"critic": "off", "revise_prompt": "off"},
        },
    )
    m = read_manifest(tmp_path)
    assert m.version == 1
    assert m.user_task == "do a thing"
    assert m.workflow.preset == ""


def test_unknown_keys_are_dropped_never_folded(tmp_path: Path) -> None:
    """Superseded or foreign keys are ignored, not converted: a manifest
    carrying only flat merged_* keys reads as unmerged (`merged is None`), the
    safe direction -- prune's force-delete keys off the nested stamp."""
    _write(
        tmp_path,
        {"run_branch": "agent6/r", "merged_into": "main", "merged_sha": "abc123", "merged_ts": "t"},
    )
    m = read_manifest(tmp_path)
    assert m.merged is None
    assert m.run_branch == "agent6/r"


def test_legacy_compare_group_is_ignored(tmp_path: Path) -> None:
    # The pre-dedup stamp carried a `group` key (same fact as parallel_id); it is
    # dropped on read (extra="ignore"), the rest of the stamp survives.
    _write(
        tmp_path,
        {"compare": {"group": "fan", "rank": 1, "of": 2, "winner": True, "ranked_by": "judge"}},
    )
    m = read_manifest(tmp_path)
    assert m.compare is not None
    assert m.compare.rank == 1 and m.compare.winner is True
    assert not hasattr(m.compare, "group")


def test_session_mode_accepts_the_two_known_modes(tmp_path: Path) -> None:
    for mode in ("run", "plan"):
        _write(tmp_path, {"mode": mode})
        assert read_manifest(tmp_path).session_mode() == mode


def test_session_mode_refuses_an_unknown_mode(tmp_path: Path) -> None:
    # The security gate: a damaged mode must NOT silently fall open to write
    # ("run") access; session_mode refuses loudly. Rendering still reads it raw.
    _write(tmp_path, {"mode": "wat"})
    m = read_manifest(tmp_path)
    assert m.mode == "wat"  # lenient render read
    with pytest.raises(ManifestError, match="unknown session mode"):
        m.session_mode()


def test_missing_manifest_raises(tmp_path: Path) -> None:
    with pytest.raises(ManifestError):
        read_manifest(tmp_path)


def test_unreadable_manifest_raises(tmp_path: Path) -> None:
    # manifest.json as a directory: read_text raises IsADirectoryError (an
    # OSError) regardless of uid, unlike a chmod-000 probe that root ignores.
    (tmp_path / "manifest.json").mkdir()
    with pytest.raises(ManifestError):
        read_manifest(tmp_path)


def test_corrupt_json_raises(tmp_path: Path) -> None:
    (tmp_path / "manifest.json").write_text("{not json", encoding="utf-8")
    with pytest.raises(ManifestError):
        read_manifest(tmp_path)


def test_torn_utf8_raises(tmp_path: Path) -> None:
    # A torn multibyte write is a UnicodeDecodeError (a ValueError), which the
    # reader folds into the same typed error instead of leaking it.
    (tmp_path / "manifest.json").write_bytes(b'{"session_id": "\x80')
    with pytest.raises(ManifestError):
        read_manifest(tmp_path)


def test_non_object_manifest_raises(tmp_path: Path) -> None:
    for bad in ("[]", "null", '"x"', "3"):
        (tmp_path / "manifest.json").write_text(bad, encoding="utf-8")
        with pytest.raises(ManifestError, match="not a JSON object"):
            read_manifest(tmp_path)


def test_write_manifest_bytes_fresh(tmp_path: Path) -> None:
    # Byte pin of the writer's emitted JSON (the read side is pinned above; this
    # pins the EXACT bytes write_manifest lands on disk: key set, key order,
    # indent, null shape, trailing newline). A fresh run: no fork/merge/compare.
    from agent6.app.manifest import write_manifest
    from agent6.sessions.manifest import ModelBrief, ModelsBrief, SessionManifest, WorkflowStamp

    m = SessionManifest(
        agent6_version="0.1.0",
        session_id="r-fresh01",
        mode="run",
        start_ts="2026-07-16T00:00:00.000000+00:00",
        user_task="add a feature",
        base_sha="0" * 40,
        base_branch="master",
        run_branch="agent6/r-fresh01",
        models=ModelsBrief(
            driver=ModelBrief(provider="anthropic", model="claude-x"),
            reviewer=ModelBrief(provider="anthropic", model="claude-y"),
        ),
        workflow=WorkflowStamp(review_trigger="off", revise_prompt="on", preset="strict"),
    )
    path = tmp_path / "manifest.json"
    write_manifest(path, m)
    assert path.read_text(encoding="utf-8") == (_DATA / "golden_manifest_fresh.json").read_text(
        encoding="utf-8"
    )


def test_write_manifest_bytes_stamped_lane(tmp_path: Path) -> None:
    # Byte pin of a fully-stamped fan-out lane: fork lineage + merge stamp +
    # parallel_id/lane + compare, so every optional nested stamp's serialized
    # shape is frozen, not just the fresh subset.
    from agent6.app.manifest import write_manifest
    from agent6.sessions.manifest import (
        CompareStamp,
        MergeStamp,
        ModelBrief,
        ModelsBrief,
        SessionManifest,
        WorkflowStamp,
    )

    m = SessionManifest(
        agent6_version="0.1.0",
        session_id="r-lane02",
        mode="run",
        start_ts="2026-07-16T00:00:00.000000+00:00",
        user_task="fan-out lane",
        base_sha="1" * 40,
        base_branch="master",
        run_branch="agent6/r-lane02",
        models=ModelsBrief(driver=ModelBrief(provider="openai", model="gpt-z")),
        workflow=WorkflowStamp(review_trigger="on", revise_prompt="off", preset=""),
        parent_session_id="r-parent",
        forked_from_turn=7,
        forked_from_sha="2" * 40,
        merged=MergeStamp(
            into="master",
            sha="3" * 40,
            ts="2026-07-16T01:00:00.000000+00:00",
            tip="4" * 40,
        ),
        parallel_id="p-abc",
        lane=1,
        compare=CompareStamp(
            rank=1,
            of=3,
            winner=True,
            ranked_by="judge",
            rationale="cleanest diff",
            judge_cost_usd=0.0102,
            judge_cost_partial=True,
        ),
    )
    path = tmp_path / "manifest.json"
    write_manifest(path, m)
    golden = (_DATA / "golden_manifest_stamped.json").read_text(encoding="utf-8")
    assert path.read_text(encoding="utf-8") == golden
    # The pinned bytes round-trip back to an equal model (writer <-> reader).
    assert read_manifest(tmp_path) == m


def test_rewriting_a_newer_manifest_is_refused(tmp_path: Path) -> None:
    """Reads stay tolerant so every historical run keeps rendering, but a
    REWRITE of a manifest a newer agent6 wrote is refused: extra="ignore" drops
    the keys this binary doesn't know, and a merge/compare stamp would silently
    downgrade the record it was only supposed to annotate."""
    from agent6.app.manifest import write_manifest

    _write(tmp_path, {"version": 4, "session_id": "r-1", "future_key": {"x": 1}})
    m = read_manifest(tmp_path)  # reading it is fine
    assert m.session_id == "r-1"
    with pytest.raises(ManifestError, match="version 4"):
        write_manifest(tmp_path / "manifest.json", m)
    # Untouched on disk: the newer record keeps its version AND its keys.
    on_disk = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert on_disk["version"] == 4
    assert on_disk["future_key"] == {"x": 1}


def test_rewriting_an_older_manifest_upgrades_it(tmp_path: Path) -> None:
    """An OLDER manifest has no keys this binary can lose, so a stamp rewrite
    upgrades the version claim to the shape it actually wrote."""
    from agent6.app.manifest import write_manifest
    from agent6.sessions.manifest import MANIFEST_VERSION

    _write(tmp_path, {"version": 1, "session_id": "r-old"})
    write_manifest(tmp_path / "manifest.json", read_manifest(tmp_path))
    on_disk = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert on_disk["version"] == MANIFEST_VERSION
    assert on_disk["session_id"] == "r-old"


def test_merge_and_lane_stamps_survive_a_newer_manifest(tmp_path: Path) -> None:
    """Both rewrite paths degrade instead of crashing on a manifest they may not
    rewrite: the merge already happened and the lane import already stands, so
    each leaves the newer record untouched and (for the lane) reports it."""
    from agent6.app.merge import record_merge_in_manifest
    from agent6.app.parallel import _stamp  # pyright: ignore[reportPrivateUsage]
    from agent6.sessions.layout import SessionLayout

    session_dir = tmp_path / "sessions" / "runs" / "r-newer"
    session_dir.mkdir(parents=True)
    payload = {"version": 4, "session_id": "r-newer", "future_key": 1}
    _write(session_dir, payload)

    layout = SessionLayout(state_dir=tmp_path, session_id="r-newer")
    record_merge_in_manifest(layout, merged_into="main", merged_sha="abc123")
    assert json.loads((session_dir / "manifest.json").read_text(encoding="utf-8")) == payload

    err = _stamp(session_dir, lane=2)
    assert err is not None and "version 4" in err
    assert json.loads((session_dir / "manifest.json").read_text(encoding="utf-8")) == payload


def test_plan_run_stamps_the_planner_as_its_driver(tmp_path: Path) -> None:
    """`sessions show` reads one field for "the model that drove this run". It used
    to be the worker unconditionally, so a plan run -- driven by the planner --
    displayed a model that never ran, and disagreed with both the web (which
    reads the role events) and its own cost block."""
    from agent6.app.manifest import write_session_manifest
    from agent6.config import Config
    from agent6.sessions.layout import SessionLayout

    cfg = Config.model_validate(
        {
            "providers": {"anthropic": {"api_format": "anthropic"}},
            "models": {
                "worker": {"provider": "anthropic", "model": "worker-model"},
                "planner": {"provider": "anthropic", "model": "planner-model"},
            },
        }
    )
    for mode, expected in (("plan", "planner-model"), ("run", "worker-model")):
        layout = SessionLayout(state_dir=tmp_path / mode, session_id="r")
        layout.ensure()
        write_session_manifest(
            layout,
            session_id="r",
            user_task="t",
            base_sha="",
            base_branch="main",
            run_branch=None,
            cfg=cfg,
            mode=mode,
        )
        driver = read_manifest(layout.session_dir).models.driver
        assert driver is not None and driver.model == expected


def test_write_session_manifest_stores_the_operators_words(tmp_path: Path) -> None:
    """`user_task` is the display twin of the OPERATOR's words: `run --skill`
    and `--from` prepend a skill block and a prior-run digest to the task the
    engine gets, and the composed prompt reached every listing as the task."""
    from agent6.app.manifest import write_session_manifest
    from agent6.config import Config
    from agent6.sessions.layout import SessionLayout
    from agent6.task_text import SKILLS_PREAMBLE

    words = "fix the parser " * 20  # past the event's 200-char clip
    composed = (
        f'{SKILLS_PREAMBLE}\n<skill name="tidy">be tidy</skill>\n---\n'
        f'<prior-run id="r-earlier">what it found</prior-run>\n\n{words}'
    )
    layout = SessionLayout(state_dir=tmp_path, session_id="r-words")
    layout.ensure()
    write_session_manifest(
        layout,
        session_id="r-words",
        user_task=composed,
        base_sha="",
        base_branch="main",
        run_branch=None,
        cfg=Config(),
        mode="run",
    )
    assert read_manifest(layout.session_dir).user_task == words.strip()


def test_a_manifest_with_no_mode_key_does_not_fall_open_to_run(tmp_path: Path) -> None:
    """The privilege gate refused an unknown mode VALUE but not a missing KEY:
    the field defaulted to "run", so a manifest that lost its mode (truncated,
    hand-edited, written by something else) resumed or forked with the
    write-tool surface -- the exact escalation session_mode exists to stop."""
    _write(tmp_path, {"version": 3, "session_id": "r", "user_task": "t"})
    m = read_manifest(tmp_path)
    with pytest.raises(ManifestError, match="unknown session mode"):
        m.session_mode()


def test_a_plan_manifest_still_gates_as_plan(tmp_path: Path) -> None:
    for mode in ("run", "plan"):
        _write(tmp_path, {"version": 3, "mode": mode})
        assert read_manifest(tmp_path).session_mode() == mode


def test_the_gate_is_pinned_with_where_it_came_from(tmp_path: Path) -> None:
    """A run records the verify gate it is judged by AND its origin, so a later
    edit to the file an inferred gate came from cannot move it, and any surface
    can say whether an operator or the repo chose it."""
    from agent6.app.manifest import stamp_verify_gate

    (tmp_path / "manifest.json").write_text(
        json.dumps({"version": 3, "mode": "run", "session_id": "r"}), encoding="utf-8"
    )
    assert read_manifest(tmp_path).workflow.verify_origin == ""  # gateless until pinned
    stamp_verify_gate(tmp_path, ("uv", "run", "pytest"), "inferred")
    wf = read_manifest(tmp_path).workflow
    assert wf.verify_command == ("uv", "run", "pytest")
    assert wf.verify_origin == "inferred"
    # Re-stamping is what adoption does; it must not disturb the rest.
    assert read_manifest(tmp_path).mode == "run"


@pytest.mark.parametrize(
    ("configured", "has_gate", "pinned", "expected"),
    [
        (True, True, "inferred", "configured"),  # config outranks the pin
        (False, True, "adopted", "adopted"),  # an adopted gate stays adopted
        (False, True, "", "inferred"),  # the leg had to re-infer
        (False, False, "inferred", ""),  # gateless leg claims nothing
        (True, False, "inferred", ""),  # a dropped gate claims nothing, even over config
    ],
)
def test_a_resumed_leg_reports_whose_gate_it_used(
    configured: bool, has_gate: bool, pinned: str, expected: str
) -> None:
    """Precedence across legs: an operator's config outranks whatever the run
    pinned, the pin outranks re-inference, and the manifest names which one
    this leg actually ran under."""
    from agent6.app.resume import leg_gate_origin

    assert leg_gate_origin(configured=configured, has_gate=has_gate, pinned=pinned) == expected


def test_a_known_mode_is_never_reported_as_an_unknown_one(tmp_path: Path) -> None:
    """The bug this vocabulary exists to prevent.

    "What kind of session is this" used to be answered in a dozen places, each
    re-deriving it from a bare string -- and two of them disagreed: the
    manifest's own list refused `machine` and `agent` outright while
    `mode_tools` happily built a tool surface for both, so a real mode was
    reported as damaged data. One table now, and the two failures are
    distinguished: a mode this agent6 does not know, and a known mode that
    resume cannot pick up.
    """
    from agent6.types import SESSION_KINDS

    for name, kind in SESSION_KINDS.items():
        session_dir = tmp_path / name
        session_dir.mkdir()
        _write(session_dir, {"mode": name})
        manifest = read_manifest(session_dir)
        if kind.resumable:
            assert manifest.session_mode() == name
            continue
        with pytest.raises(ManifestError, match="not resumable"):
            manifest.session_mode()


def test_each_mode_gets_its_own_tool_surface() -> None:
    """Read off the record, not re-derived per call site."""
    from agent6.tools.schema import ASK_EXTRA_TOOLS, MACHINE_EXTRA_TOOLS, mode_tools
    from agent6.types import SESSION_KINDS, UnknownSessionKind

    assert mode_tools("machine").extras == MACHINE_EXTRA_TOOLS
    assert mode_tools("agent").extras == MACHINE_EXTRA_TOOLS
    assert mode_tools("ask").extras == ASK_EXTRA_TOOLS
    for name, kind in SESSION_KINDS.items():
        names = mode_tools(name).names
        assert ("apply_edit" in names) is kind.edits, name
        assert ("run_command" in names) is kind.runs_commands, name
    with pytest.raises(UnknownSessionKind):
        mode_tools("wat")
