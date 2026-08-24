# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""`machine test` dry-run: per-state synthesis + per-branch routing."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from agent6.machine import dry_run, load_machine
from agent6.machine._semantics import validate_record_payload
from agent6.machine.dryrun import synthesize_record

# tool -> branch -> (agent | tool) -> terminal, with a typed capture + an enum.
DEMO = """
machine = "demo"
version = 1
initial = "scan"

[budget]
max_usd = 1.0
max_transitions = 100

[vars.operator]
approved = { type = "bool", value = false }

[vars.code]
items = { type = "list[str]", default = [] }

[vars.agent]
verdict = { type = "review", default = { label = "low", score = 0 } }

[schemas.scan_result]
items = "list[str]"

[schemas.review]
label = { type = "str", enum = ["low", "high"] }
score = "int"

[states.scan]
kind = "tool"
command = ["scan"]
output_schema = "scan_result"
capture = { set = { items = "{{ result.items }}" } }
timeout_secs = 5
on = { ok = "check", nonzero = "stop_fail", timeout = "stop_fail" }

[states.check]
kind = "branch"
when = [
  { if = "approved", goto = "judge" },
  { else = true, goto = "stop_ok" },
]

[states.judge]
kind = "agent"
model = "claude-x"
prompt = "review {{ items | json }}"
output_schema = "review"
capture = { finish_json = "verdict" }
timeout_secs = 30
on = { ok = "stop_ok", failed = "stop_fail", budget_exhausted = "stop_fail", timeout = "stop_fail" }

[states.stop_ok]
kind = "terminal"
status = "ok"
reason = "done"

[states.stop_fail]
kind = "terminal"
status = "failed"
reason = "failed"
"""


def _write(tmp_path: Path, text: str = DEMO) -> Path:
    f = tmp_path / "m.asm.toml"
    f.write_text(text, encoding="utf-8")
    return f


# --- schema synthesis -------------------------------------------------------


def test_synthesize_record_is_schema_valid(tmp_path: Path) -> None:
    spec = load_machine(_write(tmp_path))
    payload = synthesize_record(spec, "review")
    # enum field -> first member; scalar -> zero value.
    assert payload == {"label": "low", "score": 0}
    # And it passes the same strict check the live agent path uses.
    assert (
        validate_record_payload(spec.schemas, "review", payload, where="finish_session payload")
        == []
    )


def test_synthesize_handles_lists(tmp_path: Path) -> None:
    spec = load_machine(_write(tmp_path))
    assert synthesize_record(spec, "scan_result") == {"items": []}


# --- per-state dry-run ------------------------------------------------------


def test_dry_run_states_route_and_capture(tmp_path: Path) -> None:
    spec = load_machine(_write(tmp_path))
    report = dry_run(spec)
    by_name = {s.name: s for s in report.states}
    # branch state is reported separately, not in the per-state pass.
    assert "check" not in by_name
    assert by_name["scan"].ok and by_name["scan"].goto == "check"
    assert "captures items" in by_name["scan"].detail
    assert by_name["judge"].ok and by_name["judge"].goto == "stop_ok"
    assert "captures verdict" in by_name["judge"].detail
    assert by_name["stop_ok"].kind == "terminal" and by_name["stop_ok"].ok
    assert report.ok


# --- per-branch routing -----------------------------------------------------


def test_branch_routes_to_else_by_default(tmp_path: Path) -> None:
    spec = load_machine(_write(tmp_path))
    report = dry_run(spec)  # approved defaults to false
    check = next(b for b in report.branches if b.name == "check")
    assert check.goto == "stop_ok"
    assert check.predicate == "else"
    assert check.ok


def test_branch_fixture_steers_routing(tmp_path: Path) -> None:
    spec = load_machine(_write(tmp_path))
    report = dry_run(spec, {"approved": True})
    check = next(b for b in report.branches if b.name == "check")
    assert check.clause_index == 0
    assert check.goto == "judge"
    assert check.predicate == "approved"


def test_branch_on_empty_record_default_synthesizes_fields(tmp_path: Path) -> None:
    # The realistic shape: an agent verdict var with the required `default = {}`
    # routed by a branch reading `verdict.field`. The dry-run must synthesize
    # the schema-zero record so the predicate evaluates instead of erroring on
    # a missing field (which made every such machine fail `machine test`).
    text = DEMO.replace(
        'verdict = { type = "review", default = { label = "low", score = 0 } }',
        'verdict = { type = "review", default = {} }',
    ).replace(
        '{ if = "approved", goto = "judge" }',
        '{ if = "verdict.score > 0", goto = "judge" }',
    )
    spec = load_machine(_write(tmp_path, text))
    report = dry_run(spec)
    check = next(b for b in report.branches if b.name == "check")
    assert check.ok, check.detail
    assert check.goto == "stop_ok"  # zero score -> else
    # A fixture still wins over the synthesized record.
    report2 = dry_run(spec, {"verdict": {"label": "high", "score": 5}})
    check2 = next(b for b in report2.branches if b.name == "check")
    assert check2.goto == "judge"


# --- CLI surface ------------------------------------------------------------


def test_cli_machine_test_passes(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    from agent6.ui.cli import main

    f = _write(tmp_path)
    assert main(["machine", "test", str(f)]) == 0
    out = capsys.readouterr().out
    assert "per-state dry-run" in out
    assert "per-branch routing" in out
    assert "dry-run passed" in out


def test_cli_machine_test_verdict_names_unrun_offline_tests(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """On a host that cannot jail the offline script tests, the OK verdict
    itself says how many were NOT run and why. The skip lived only in a stderr
    aside, so `machine test` read as "tests ran green" while a deliberately
    failing test never executed."""
    from types import SimpleNamespace

    from agent6.ui.cli import machine_check, main

    f = _write(tmp_path)
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    (scripts / "thing_test.py").write_text("raise SystemExit(1)\n", encoding="utf-8")
    monkeypatch.setattr(
        machine_check, "detect_env", lambda: SimpleNamespace(detected_isolation="hardened")
    )
    assert main(["machine", "test", str(f)]) == 0
    out = capsys.readouterr().out
    assert "1 offline script test NOT run" in out
    assert "hardened" in out


def test_cli_machine_test_with_blackboard(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from agent6.ui.cli import main

    f = _write(tmp_path)
    bb = tmp_path / "bb.toml"
    bb.write_text("approved = true\n", encoding="utf-8")
    assert main(["machine", "test", str(f), "--blackboard", str(bb)]) == 0
    out = capsys.readouterr().out
    # The branch ROW must show clause 0 routing to judge ("judge" alone also
    # matches the unconditional per-state row, proving nothing about routing).
    assert re.search(r"check\s+\[0\]\s+judge", out), out


def test_cli_machine_test_rejects_a_fixture_off_the_blackboard_schema(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The fixture merged into the blackboard unvalidated: a typo'd key was
    silently ignored and a string "false" replaced a bool and routed branches
    as truthy. Every key must name a declared var; every value must satisfy
    its type, exactly like the declared defaults."""
    from agent6.ui.cli import main

    f = _write(tmp_path)
    bad_type = tmp_path / "bad_type.toml"
    bad_type.write_text('approved = "false"\n', encoding="utf-8")
    assert main(["machine", "test", str(f), "--blackboard", str(bad_type)]) == 1
    err = capsys.readouterr().err
    assert "FAIL" in err and "approved" in err and "bool" in err

    typo = tmp_path / "typo.toml"
    typo.write_text("aproved = true\n", encoding="utf-8")
    assert main(["machine", "test", str(f), "--blackboard", str(typo)]) == 1
    assert "not a declared variable" in capsys.readouterr().err


def test_cli_machine_test_runs_check_first(tmp_path: Path) -> None:
    from agent6.ui.cli import main

    # An invalid machine (goto target missing) must fail like `machine check`.
    bad = DEMO.replace('goto = "judge"', 'goto = "nope"')
    f = _write(tmp_path, bad)
    assert main(["machine", "test", str(f)]) == 1


def test_cli_machine_test_missing_fixture(tmp_path: Path) -> None:
    from agent6.errors import OperatorError
    from agent6.ui.cli import main

    f = _write(tmp_path)
    with pytest.raises(OperatorError, match="could not read"):
        main(["machine", "test", str(f), "--blackboard", str(tmp_path / "nope.toml")])


def test_cli_machine_test_bad_fixture_toml(tmp_path: Path) -> None:
    from agent6.errors import OperatorError
    from agent6.ui.cli import main

    f = _write(tmp_path)
    bb = tmp_path / "bb.toml"
    bb.write_text("not = valid = toml", encoding="utf-8")
    with pytest.raises(OperatorError, match="not valid TOML"):
        main(["machine", "test", str(f), "--blackboard", str(bb)])


def test_cli_machine_test_unreadable_fixture_refuses(tmp_path: Path) -> None:
    """The fixture read caught a TOML parse error but not an OSError, so a
    root-owned blackboard crashed through the bug reporter instead of the
    operator-error refusal every other unreadable operator file gets."""
    from agent6.errors import OperatorError
    from agent6.ui.cli import main

    f = _write(tmp_path)
    bb = tmp_path / "bb.toml"
    bb.write_text("approved = true\n", encoding="utf-8")
    bb.chmod(0o000)
    try:
        with pytest.raises(OperatorError, match="could not read"):
            main(["machine", "test", str(f), "--blackboard", str(bb)])
    finally:
        bb.chmod(0o600)


def test_synthesized_records_omit_optional_fields(tmp_path: Path) -> None:
    """Dry-run models the weakest state the capture gate permits: an optional
    field stays absent, so a branch reading it unguarded fails `machine test`
    exactly as it halts live, instead of routing on invented data; the
    has()-guarded twin routes cleanly."""
    from agent6.ui.cli import main

    unguarded = tmp_path / "unguarded.asm.toml"
    unguarded.write_text(PRESENCE.format(pred="out.score > 0"), encoding="utf-8")
    assert main(["machine", "test", str(unguarded)]) == 1

    guarded = tmp_path / "guarded.asm.toml"
    guarded.write_text(PRESENCE.format(pred="has(out.score) and out.score > 0"), encoding="utf-8")
    assert main(["machine", "test", str(guarded)]) == 0


PRESENCE = """\
machine = "presence"
version = 1
initial = "route"

[budget]
max_transitions = 10

[schemas.report]
summary = {{ type = "str" }}
score = {{ type = "int", optional = true }}

[vars.code]
out = {{ type = "report", default = {{}} }}

[states.route]
kind = "branch"
when = [
  {{ if = "{pred}", goto = "good" }},
  {{ else = true, goto = "bad" }},
]

[states.good]
kind = "terminal"
status = "ok"
reason = "scored"

[states.bad]
kind = "terminal"
status = "failed"
reason = "unscored"
"""
