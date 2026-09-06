# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The web client shows lines the server renders; it keeps no copy of a
Python rendering decision.

client.js carried a `fmtUsd` twin of `format_usd`, a task-glyph map, a
`format_compare` mirror and a `format_transition` mirror, each with a "keep in
sync" comment and no pin. The cost twin had drifted: Python's `%.4f` rounds
half to even on the binary value and JS `toFixed` rounds half away, so
0.15625 printed `$0.1562` on the CLI and `$0.1563` on the web.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from agent6.config.layer import resolved_state_dir
from agent6.sessions.layout import bucket_dir, machines_root
from agent6.ui.cli import main
from agent6.ui.web.page import PAGE_HTML
from agent6.viewmodel.format import budget_usd_text
from agent6.viewmodel.listing import summarize_session_dir, summary_row
from agent6.viewmodel.snapshot import machine_snapshot, session_snapshot
from agent6.viewmodel.state import task_tree_views

ROUTER = """
machine = "router"
version = 1
initial = "route"

[budget]
max_transitions = 10

[states.route]
kind = "branch"
when = [{ else = true, goto = "done" }]

[states.done]
kind = "terminal"
status = "ok"
reason = "routed"
"""


def _run(tmp_path: Path, name: str, events: list[dict[str, object]]) -> Path:
    d = bucket_dir(resolved_state_dir(tmp_path), "runs") / name
    d.mkdir(parents=True)
    (d / "logs.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events), encoding="utf-8")
    return d


def test_the_page_carries_no_cost_formatter() -> None:
    assert "fmtUsd" not in PAGE_HTML and "toFixed(4)" not in PAGE_HTML


def test_the_budget_line_is_rendered_once() -> None:
    assert budget_usd_text(0.42, partial=False, usd_cap=0.0, usd_prior_legs=0.0) == "$0.42"
    assert budget_usd_text(0.42, partial=True, usd_cap=-1, usd_prior_legs=0.0) == (
        "~$0.42 (unlimited)"
    )
    assert budget_usd_text(0.42, partial=False, usd_cap=1.0, usd_prior_legs=0.0) == (
        "$0.42 / $1.00"
    )
    assert budget_usd_text(0.42, partial=False, usd_cap=1.0, usd_prior_legs=0.1) == (
        "$0.42 · leg $0.32 / $1.00"
    )


def test_the_hub_row_and_the_run_view_carry_rendered_cells(tmp_path: Path) -> None:
    spent = _run(
        tmp_path,
        "spent",
        [
            {"type": "session.start", "mode": "run", "user_task": "x"},
            {"type": "budget.update", "usd_total": 0.15625, "usd_cap": 1.0},
        ],
    )
    clean = _run(tmp_path, "clean", [{"type": "session.start", "mode": "run", "user_task": "y"}])
    assert summary_row(summarize_session_dir(spent))["cost"] == "$0.16"
    assert summary_row(summarize_session_dir(clean))["cost"] == ""
    assert session_snapshot(spent)["budget"]["usd_text"] == "$0.16 / $1.00"

    (spent / "manifest.json").write_text(
        json.dumps({"compare": {"rank": 1, "of": 2, "winner": True, "ranked_by": "judge"}}),
        encoding="utf-8",
    )
    assert session_snapshot(spent)["compare"]["line"] == "rank 1/2 · winner · judge"


def test_task_rows_carry_their_glyph() -> None:
    views = task_tree_views(
        {"a": {"title": "t", "status": "passed", "children": ["b"]}, "b": {"title": "u"}}, "b"
    )
    assert [(v.glyph, v.is_cursor) for v in views] == [("✓", False), ("·", True)]


def test_machine_transitions_and_spend_arrive_rendered(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("AGENT6_STATE_HOME", str(tmp_path / "state"))
    monkeypatch.setenv("AGENT6_CONFIG_HOME", str(tmp_path / "cfg"))
    f = tmp_path / "router.asm.toml"
    f.write_text(ROUTER, encoding="utf-8")
    assert main(["machine", "run", str(f)]) == 0
    capsys.readouterr()
    md = machines_root(resolved_state_dir(tmp_path)) / "router"
    snap = machine_snapshot(md)
    (first, *_rest) = snap["transitions"]
    assert (
        first["line"] == f"[{first['seq']}] {first['state']} --{first['label']}--> {first['goto']}"
    )
    assert first["state"] == "route" and first["goto"] == "done"
    assert snap["spend"]["text"] == "$0.0000"
    assert os.environ["AGENT6_STATE_HOME"]  # the run wrote under the isolated state home
