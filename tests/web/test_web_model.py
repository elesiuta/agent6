# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Unit tests for the pure web payload builders (no HTTP)."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from agent6.config.layer import load_effective, resolved_state_dir
from agent6.models.registry import resolved_adaptive_values
from agent6.sessions.layout import bucket_dir, machines_root
from agent6.ui.web import model
from agent6.viewmodel import machine_snapshot, session_snapshot
from agent6.viewmodel.config_view import render_show


def _bucket(cwd: Path, sub: str) -> Path:
    return bucket_dir(resolved_state_dir(cwd), sub)


def _run(cwd: Path, session_id: str, events: list[dict[str, object]]) -> Path:
    d = _bucket(cwd, "runs") / session_id
    d.mkdir(parents=True)
    (d / "logs.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events), encoding="utf-8")
    return d


def test_run_summary_captures_cost_and_status(tmp_path: Path) -> None:
    _run(
        tmp_path,
        "r1",
        [
            {"type": "session.start", "mode": "run", "user_task": "the task"},
            {"type": "budget.update", "usd_total": 0.0123},
            {"type": "session.end", "all_passed": True},
        ],
    )
    (s,) = model.hub_payload(tmp_path)["sessions"]
    assert s["mode"] == "run"
    assert s["task"] == "the task"
    assert s["status"] == "passed"
    assert s["cost_usd"] == 0.0123
    assert s["usd_partial"] is False
    assert s["label"] == "passed"  # the one shared human label, rendered verbatim


def test_run_summary_carries_the_partial_cost_marker(tmp_path: Path) -> None:
    # The hub row renders the same lower-bound marker as the run page.
    _run(
        tmp_path,
        "r1p",
        [
            {"type": "session.start", "mode": "run", "user_task": "t"},
            {"type": "budget.update", "usd_total": 0.0123, "usd_partial": True},
            {"type": "session.end", "all_passed": True},
        ],
    )
    (s,) = model.hub_payload(tmp_path)["sessions"]
    assert s["usd_partial"] is True


def test_run_summary_survives_torn_utf8_tail(tmp_path: Path) -> None:
    # A live writer can leave the log's last line torn mid multibyte UTF-8
    # sequence; the hub summary must fold the complete lines, not raise.
    d = _bucket(tmp_path, "runs") / "torn"
    d.mkdir(parents=True)
    full = json.dumps({"type": "role.text_delta", "text": "café"}, ensure_ascii=False).encode()
    cut = full.rindex(b"\xc3\xa9") + 1  # keep only the first byte of the é
    head = json.dumps({"type": "session.start", "mode": "run", "user_task": "torn tail"}).encode()
    (d / "logs.jsonl").write_bytes(head + b"\n" + full[:cut])
    (s,) = model.hub_payload(tmp_path)["sessions"]
    assert s["task"] == "torn tail"


def test_conversation_payload_folds_the_event_log(tmp_path: Path) -> None:
    # Items come from the shared TranscriptFold + item_lines renderer: a tool's
    # multi-line result is clipped to its first line + a "+N more lines" note,
    # with the full rendering carried separately for per-item expansion.
    dump = "3 validation errors for ApplyEditInput\npath\n  Field required"
    d = _run(
        tmp_path,
        "r2",
        [
            {"type": "session.start", "user_task": "x"},
            {"type": "tool.call", "name": "apply_edit", "args": {"path": "a.py"}},
            {"type": "tool.result", "name": "apply_edit", "ok": False, "summary": dump},
        ],
    )
    payload = model.conversation_payload(d)
    assert payload["session_id"] == "r2"
    (item,) = payload["items"]
    assert item["kind"] == "tool"
    flat = "".join(text for line in item["lines"] for text, _style in line)
    assert "(+2 more lines)" in flat
    assert "Field required" not in flat  # clipped in the collapsed rendering
    full = "".join(text for line in item["full"] for text, _style in line)
    assert "Field required" in full  # the expanded rendering carries it


def test_run_snapshot_embeds_the_compare_outcome(tmp_path: Path) -> None:
    # A fan-out lane's manifest carries the compare block; the run snapshot
    # embeds it so the page header can render rank/winner/rationale.
    d = _run(tmp_path, "lane1", [{"type": "session.start", "user_task": "x"}])
    (d / "manifest.json").write_text(
        json.dumps(
            {"compare": {"group": "fan", "rank": 1, "of": 2, "winner": True,
                         "ranked_by": "judge", "rationale": "cleanest diff"}}
        ),
        encoding="utf-8",
    )  # fmt: skip
    snap = session_snapshot(d)
    assert snap["compare"]["winner"] is True and snap["compare"]["rank"] == 1
    assert snap["compare"]["rationale"] == "cleanest diff"
    # A run with no compare block carries no `compare` key (non-lane runs).
    plain = _run(tmp_path, "plain", [{"type": "session.start", "user_task": "y"}])
    assert "compare" not in session_snapshot(plain)


def test_run_snapshot_resolves_the_task_from_the_manifest(tmp_path: Path) -> None:
    """The fold sets user_task only from session.start, so a parked/created/forked
    run folds it empty. The wire owner (session_state_as_dict) fills it from the
    manifest -- ONE task field; a second fallback_task the client had to
    coalesce is gone."""
    d = _bucket(tmp_path, "runs") / "parked1"
    d.mkdir(parents=True)
    (d / "manifest.json").write_text(
        json.dumps({"mode": "run", "user_task": "queued work", "parked_task": "queued work"}),
        encoding="utf-8",
    )
    snap = session_snapshot(d)
    assert snap["user_task"] == "queued work"
    assert "fallback_task" not in snap


def test_plan_snapshot_carries_the_plan_md(tmp_path: Path) -> None:
    """A planning run's deliverable rides the snapshot as plan_md (the web shows
    it in a Plan card; `agent6 plan show` prints the same file). A run, or a
    plan that has not written one, carries no such key."""
    d = _bucket(tmp_path, "plans") / "plan1"
    d.mkdir(parents=True)
    (d / "manifest.json").write_text(
        json.dumps({"mode": "plan", "user_task": "lay it out"}), encoding="utf-8"
    )
    assert "plan_md" not in session_snapshot(d)
    (d / "plan.md").write_text("# Plan: lay it out\n\n1. do\n", encoding="utf-8")
    assert session_snapshot(d)["plan_md"].startswith("# Plan: lay it out")
    run = _bucket(tmp_path, "runs") / "run1"
    run.mkdir(parents=True)
    (run / "manifest.json").write_text(
        json.dumps({"mode": "run", "user_task": "t"}), encoding="utf-8"
    )
    (run / "plan.md").write_text("stray\n", encoding="utf-8")
    assert "plan_md" not in session_snapshot(run)


def test_hub_marks_the_fan_out_winner(tmp_path: Path) -> None:
    d = _run(tmp_path, "lane-win", [{"type": "session.start", "mode": "run", "user_task": "t"}])
    (d / "manifest.json").write_text(
        json.dumps({"compare": {"rank": 1, "of": 2, "winner": True}}), encoding="utf-8"
    )
    (s,) = model.hub_payload(tmp_path)["sessions"]
    assert s["winner"] is True


def test_conversation_payload_carries_operator_inputs(tmp_path: Path) -> None:
    """The composer's Ctrl-R history search reads `operator_inputs`: the task,
    then every steer, raw text in journal order (the client flattens and
    reverses for display)."""
    d = _run(
        tmp_path,
        "r2h",
        [
            {"type": "session.start", "mode": "run", "user_task": "polish the web"},
            {"type": "loop.steer.injected", "chars": 14, "text": "focus on tests"},
            {"type": "loop.steer.injected", "chars": 8, "text": "ship\nit"},
        ],
    )
    payload = model.conversation_payload(d)
    assert payload["operator_inputs"] == ["polish the web", "focus on tests", "ship\nit"]


def test_conversation_payload_empty_without_log(tmp_path: Path) -> None:
    d = _bucket(tmp_path, "runs") / "r2b"
    d.mkdir(parents=True)
    assert model.conversation_payload(d) == {
        "session_id": "r2b",
        "items": [],
        "operator_inputs": [],
    }


def test_machine_conversation_payload_uses_newest_state_log(tmp_path: Path) -> None:
    md = machines_root(resolved_state_dir(tmp_path)) / "m2"
    (md / "states" / "0001-work").mkdir(parents=True)
    (md / "states" / "0001-work" / "logs.jsonl").write_text(
        json.dumps({"type": "loop.steer.injected", "text": "hello"}) + "\n", encoding="utf-8"
    )
    payload = model.machine_conversation_payload(md)
    assert payload["state_dir"] == "0001-work"
    (item,) = payload["items"]
    assert item["kind"] == "operator"
    assert model.machine_conversation_payload(
        machines_root(resolved_state_dir(tmp_path)) / "nope"
    ) == {
        "state_dir": "",
        "items": [],
    }


TINY_MACHINE = """
machine = "tiny"
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


def test_machine_snapshot_carries_the_dir_status_word(tmp_path: Path) -> None:
    """The machine wire payload stamps `status` (machine_word_for_dir), so a
    client can gate Steer and the prompt boxes on liveness -- with only
    `ended` it cannot tell a parked machine from a running one."""
    md = machines_root(resolved_state_dir(tmp_path)) / "m3"
    md.mkdir(parents=True)
    (md / "machine.asm.toml").write_text(TINY_MACHINE, encoding="utf-8")
    (md / "journal.jsonl").write_text("", encoding="utf-8")
    assert machine_snapshot(md)["status"] == "stopped"  # no wait, no worker
    (md / "wait.json").write_text('{"state":"route","wake_epoch":9999999999.0}\n', encoding="utf-8")
    assert machine_snapshot(md)["status"] == "waiting"  # parked


def test_hub_machine_pill_keeps_the_failure_reason(tmp_path: Path) -> None:
    """A failed machine's hub entry carries the reason label (failed · why), like
    run and draft rows, not a bare 'failed' word."""
    md = machines_root(resolved_state_dir(tmp_path)) / "m-fail"
    md.mkdir(parents=True)
    (md / "machine.asm.toml").write_text(TINY_MACHINE, encoding="utf-8")
    (md / "journal.jsonl").write_text(
        json.dumps({"type": "machine.begin", "ts": "t", "machine": "tiny", "version": 1})
        + "\n"
        + json.dumps(
            {
                "type": "machine.end",
                "ts": "t",
                "status": "failed",
                "reason": "boom",
                "state": "route",
                "transitions": 1,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (m,) = model.hub_payload(tmp_path)["machines"]
    assert m["status"] == "failed"
    assert m["label"] == "failed · boom"


def test_reasoning_snapshot_empty_without_state_log(tmp_path: Path) -> None:
    # A machine dir with no states/ subtree has no agent reasoning to fold.
    md = machines_root(resolved_state_dir(tmp_path)) / "m1"
    md.mkdir(parents=True)
    assert model.machine_reasoning_snapshot(md) == {}


def test_an_id_in_two_buckets_resolves_to_neither(tmp_path: Path) -> None:
    """State from before ids were one namespace can hold the same id in two
    buckets; showing whichever bucket iterates first silently served one of
    two sessions. Ambiguity resolves to None (a 404 the CLI resolver names)."""
    _run(tmp_path, "twin", [{"type": "session.start"}])
    assert model.session_dir_for(tmp_path, "twin") is not None
    d = _bucket(tmp_path, "plans") / "twin"
    d.mkdir(parents=True)
    (d / "logs.jsonl").write_text('{"type": "session.start"}\n', encoding="utf-8")
    assert model.session_dir_for(tmp_path, "twin") is None


def test_run_dir_for_rejects_traversal(tmp_path: Path) -> None:
    _run(tmp_path, "good-run", [{"type": "session.start"}])
    assert model.session_dir_for(tmp_path, "good-run") is not None
    for bad in ("..", ".", "", "../good-run", "a/b", "..\\x"):
        assert model.session_dir_for(tmp_path, bad) is None


def test_machine_dir_for_rejects_traversal(tmp_path: Path) -> None:
    (machines_root(resolved_state_dir(tmp_path)) / "m1").mkdir(parents=True)
    assert model.machine_dir_for(tmp_path, "m1") is not None
    for bad in ("..", "../m1", "a/b", ""):
        assert model.machine_dir_for(tmp_path, bad) is None


def test_hub_payload_shape(tmp_path: Path) -> None:
    _run(tmp_path, "r3", [{"type": "session.start", "mode": "plan"}])
    hub = model.hub_payload(tmp_path)
    assert [r["session_id"] for r in hub["sessions"]] == ["r3"]
    assert hub["machines"] == []


def test_hub_payload_lists_machine_drafts(tmp_path: Path) -> None:
    draft = resolved_state_dir(tmp_path) / "sessions" / "machines" / "breezy-fern-AB12CD"
    draft.mkdir(parents=True)
    (draft / "logs.jsonl").write_text(
        json.dumps({"type": "session.start", "mode": "run", "user_task": "author a triage machine"})
        + "\n",
        encoding="utf-8",
    )
    hub = model.hub_payload(tmp_path)
    (s,) = hub["drafts"]
    assert s["session_id"] == "breezy-fern-AB12CD"
    assert s["task"] == "author a triage machine"


def test_hub_and_lookup_skip_husk_run_dirs(tmp_path: Path) -> None:
    # A husk (neither manifest nor logs) is not listed, and must not shadow a
    # real ask of the same id when resolving #/session/<id>.
    (_bucket(tmp_path, "runs") / "echo-fern-AA11BB").mkdir(parents=True)
    ask = _bucket(tmp_path, "asks") / "echo-fern-AA11BB"
    ask.mkdir(parents=True)
    (ask / "logs.jsonl").write_text(
        json.dumps({"type": "session.start", "mode": "ask", "user_task": "q"}) + "\n",
        encoding="utf-8",
    )
    hub = model.hub_payload(tmp_path)
    assert [r["mode"] for r in hub["sessions"]] == ["ask"]
    assert model.session_dir_for(tmp_path, "echo-fern-AA11BB") == ask


def test_config_payload_resolves_adaptive_leaves_like_config_show(tmp_path: Path) -> None:
    """`prompt.decompose = auto` and the unset compaction thresholds resolve
    from the worker model at runtime; the page showed the raw placeholders
    (`auto`, `(unset)`) where `config show` printed the resolved values
    marked adaptive."""
    cfg = tmp_path / "c.toml"
    cfg.write_text(
        '[providers.o]\napi_format = "openai"\nbase_url = "https://x/v1"\n'
        '[models.worker]\nprovider = "o"\nmodel = "claude-haiku-4-5"\n',
        encoding="utf-8",
    )
    payload = model.config_payload(tmp_path, cfg)
    eff = load_effective(tmp_path, cfg)
    resolved = resolved_adaptive_values(eff.config)
    shown = json.loads(render_show(eff, as_json=True, resolved=resolved))
    for key in ("prompt.decompose", "context.drop_at_chars", "context.summarise_at_chars"):
        assert payload[key] == shown[key]
        assert payload[key]["adaptive"] is True
    assert payload["prompt.decompose"]["display"] == "off  (adaptive)"
    assert isinstance(payload["context.drop_at_chars"]["effective"], int)


def test_config_suggestions_providers_and_models(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # models.<role>.provider carries the configured provider names as CHOICES
    # in the config payload (a select, like an enum); models.<role>.model is
    # suggested from the role's provider's model ids via the one cache-first
    # listing the TUI config page and CLI completion use (`models.choices`).
    from agent6.models import choices

    cfg_home = Path(os.environ["AGENT6_CONFIG_HOME"])
    (cfg_home / "config.toml").write_text(
        '[providers.openrouter]\napi_format = "openai"\n'
        'base_url = "https://openrouter.ai/api/v1"\n'
        '[models.worker]\nprovider = "openrouter"\nmodel = "kimi"\n',
        encoding="utf-8",
    )
    seen: dict[str, object] = {}

    def _fake_list(provider: str, entry: object, api_key: object) -> list[str]:
        seen["provider"] = provider
        return ["kimi", "qwen3"]

    monkeypatch.setattr(choices, "list_models", _fake_list)
    payload = model.config_payload(tmp_path)
    assert payload["models.worker.provider"]["choices"] == ["openrouter"]
    assert payload["preset"]["choices"] == ["paranoid", "quick", "standard", "ultra"]
    assert model.config_suggestions(tmp_path, "models.worker.provider") == []
    assert model.config_suggestions(tmp_path, "models.worker.model") == ["kimi", "qwen3"]
    assert seen["provider"] == "openrouter"
    assert model.config_suggestions(tmp_path, "preset") == [
        "paranoid",
        "quick",
        "standard",
        "ultra",
    ]
    # unknown keys / roles suggest nothing
    assert model.config_suggestions(tmp_path, "web.port") == []
    assert model.config_suggestions(tmp_path, "models.nosuch.model") == []


def test_config_suggestions_parallel_models_pseudo_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The /parallel composer autocomplete: the worker's configured model plus the
    # worker provider's cached listing, cache-only so it never blocks.
    cfg_home = Path(os.environ["AGENT6_CONFIG_HOME"])
    (cfg_home / "config.toml").write_text(
        '[providers.openrouter]\napi_format = "openai"\n'
        'base_url = "https://openrouter.ai/api/v1"\n'
        '[models.worker]\nprovider = "openrouter"\nmodel = "role-only-model"\n',
        encoding="utf-8",
    )
    monkeypatch.setenv("AGENT6_CACHE_HOME", str(tmp_path / "cache"))
    cache = tmp_path / "cache" / "models"
    cache.mkdir(parents=True)
    (cache / "openrouter.json").write_text(
        json.dumps({"models": ["moonshotai/kimi-k2.6", "z-ai/glm-4.6"]}), encoding="utf-8"
    )
    out = model.config_suggestions(tmp_path, "parallel.models")
    assert out == ["moonshotai/kimi-k2.6", "role-only-model", "z-ai/glm-4.6"]


def test_parallel_models_suggestions_scoped_to_worker_provider(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Lanes inherit the WORKER provider (only the model is overridden per lane),
    # so the suggestions offer only models the lanes can actually run: a sibling
    # provider's cached catalog is excluded.
    cfg_home = Path(os.environ["AGENT6_CONFIG_HOME"])
    (cfg_home / "config.toml").write_text(
        '[providers.w]\napi_format = "openai"\nbase_url = "https://w.example/v1"\n'
        '[providers.s]\napi_format = "openai"\nbase_url = "https://s.example/v1"\n'
        '[models.worker]\nprovider = "w"\nmodel = "w/base-model"\n',
        encoding="utf-8",
    )
    monkeypatch.setenv("AGENT6_CACHE_HOME", str(tmp_path / "cache"))
    cache = tmp_path / "cache" / "models"
    cache.mkdir(parents=True)
    (cache / "w.json").write_text(json.dumps({"models": ["w/model-a"]}), encoding="utf-8")
    (cache / "s.json").write_text(json.dumps({"models": ["s/only-model"]}), encoding="utf-8")
    assert model.config_suggestions(tmp_path, "parallel.models") == ["w/base-model", "w/model-a"]


def test_run_snapshot_labels_a_parked_submission(tmp_path: Path) -> None:
    """A parked run (the busy-checkout refusal saved the task) has no events by
    construction, so the event fold alone reads it as "running" while the hub row
    says parked. The run page must not disagree with the hub about the same run."""
    d = _bucket(tmp_path, "runs") / "parked1"
    d.mkdir(parents=True)
    (d / "manifest.json").write_text(
        json.dumps(
            {
                "session_id": "parked1",
                "parked_task": "do the thing",
                "parked_reason": "checkout busy",
            }
        ),
        encoding="utf-8",
    )
    assert session_snapshot(d)["status_label"] == "parked · checkout busy"
    (hub_row,) = model.hub_payload(tmp_path)["sessions"]
    assert hub_row["status"] == "parked"  # the two surfaces lead with one word


def test_a_parked_runs_policy_names_the_configured_gates_origin(tmp_path: Path) -> None:
    """A fresh manifest carried the configured verify command with no origin
    (the leg's pin fills it in), so a run parked before its leg read
    `python3 -m pytest -q (unknown origin)` in every header."""
    from agent6.app.manifest import stamp_parked, write_session_manifest
    from agent6.config import Config
    from agent6.sessions.layout import SessionLayout

    layout = SessionLayout(state_dir=resolved_state_dir(tmp_path), session_id="parked-two-AAAAAA")
    layout.ensure()
    write_session_manifest(
        layout,
        session_id=layout.session_id,
        user_task="t",
        base_sha="0" * 40,
        base_branch="main",
        run_branch=None,
        cfg=Config.model_validate({"workflow": {"verify_command": ["python3", "-m", "pytest"]}}),
    )
    stamp_parked(layout.session_dir, task="t", reason="checkout busy")
    snap = session_snapshot(layout.session_dir)
    assert snap["status_label"] == "parked · checkout busy"
    assert snap["policy"].endswith("python3 -m pytest (configured)")
    # A gateless config stays gateless until the leg infers or adopts one.
    write_session_manifest(
        layout,
        session_id=layout.session_id,
        user_task="t",
        base_sha="0" * 40,
        base_branch="main",
        run_branch=None,
        cfg=Config(),
    )
    assert session_snapshot(layout.session_dir)["policy"].endswith("no verify gate")


def test_run_snapshot_labels_a_dead_worker_stale(tmp_path: Path) -> None:
    """A run whose recorded worker is gone and that never logged session.end folds to
    "running". The hub calls it stale off the same pid probe; the one-shot payload
    the page first paints from has to say so too, not only the SSE frame."""
    d = _run(tmp_path, "crashed1", [{"type": "session.start", "mode": "run", "user_task": "t"}])
    (d / "worker.pid").write_text("999999 12345678", encoding="utf-8")  # dead pid
    assert session_snapshot(d)["status_label"] == "stale"


def test_run_snapshot_labels_waiting_starting_created(tmp_path: Path) -> None:
    """The run page speaks EVERY listing word, not just parked/stale: blocked
    on an operator answer reads "waiting · needs answer" (it read "running"
    and sent the operator off to wait on the model while the run waited on
    THEM), a live pre-session.start worker "starting", a never-started dir
    "created"."""
    import os

    from agent6.sessions.ipc import write_worker_pid

    d = _run(
        tmp_path,
        "blocked1",
        [
            {"type": "session.start", "mode": "run", "user_task": "t"},
            {"type": "approval.prompt", "id": "approval-1", "prompt": "rm -rf?"},
        ],
    )
    write_worker_pid(d, os.getpid())
    assert session_snapshot(d)["status_label"] == "waiting · needs answer"

    e = _bucket(tmp_path, "runs") / "fresh1"
    e.mkdir(parents=True)
    (e / "manifest.json").write_text(json.dumps({"session_id": "fresh1"}), encoding="utf-8")
    write_worker_pid(e, os.getpid())
    assert session_snapshot(e)["status_label"] == "starting"
    (e / "worker.pid").unlink()
    assert session_snapshot(e)["status_label"] == "created"


def test_run_snapshot_leaves_a_finished_run_alone(tmp_path: Path) -> None:
    """The dir-derived relabels never touch a run that ended on its own terms."""
    d = _run(
        tmp_path,
        "done1",
        [
            {"type": "session.start", "mode": "run", "user_task": "t"},
            {
                "type": "session.end",
                "reason": "finish_session",
                "iterations": 1,
                "all_passed": True,
            },
        ],
    )
    (d / "worker.pid").write_text("999999 12345678", encoding="utf-8")
    assert session_snapshot(d)["status_label"] == "passed"


def test_run_snapshot_marks_a_parked_run_not_live(tmp_path: Path) -> None:
    """The page keys its composer and Stop/Compact buttons on liveness. The fold
    calls every unfinished run "running", so a parked run offered a steer
    composer and a Stop button that both dead-ended while resume -- the one
    action that works -- was unreachable. `live` is the dir-aware answer."""
    import os

    from agent6.sessions.ipc import write_worker_pid

    parked = _bucket(tmp_path, "runs") / "parked2"
    parked.mkdir(parents=True)
    (parked / "manifest.json").write_text(
        json.dumps({"session_id": "parked2", "parked_task": "do the thing"}), encoding="utf-8"
    )
    assert session_snapshot(parked)["live"] is False

    crashed = _run(
        tmp_path, "crashed2", [{"type": "session.start", "mode": "run", "user_task": "t"}]
    )
    (crashed / "worker.pid").write_text("999999999", encoding="utf-8")
    assert session_snapshot(crashed)["live"] is False

    alive = _run(tmp_path, "alive2", [{"type": "session.start", "mode": "run", "user_task": "t"}])
    write_worker_pid(alive, os.getpid())
    assert session_snapshot(alive)["live"] is True

    done = _run(
        tmp_path,
        "done2",
        [
            {"type": "session.start", "mode": "run", "user_task": "t"},
            {
                "type": "session.end",
                "reason": "finish_session",
                "iterations": 1,
                "all_passed": True,
            },
        ],
    )
    assert session_snapshot(done)["live"] is False


def test_the_states_that_offer_resume_are_not_live(tmp_path: Path) -> None:
    """The composer's resume-takeover poll waits for the run to come alive. It
    polled `finished === false`, which is ALREADY true for the parked and stale
    runs it routes into resume mode, so takeover was declared on the first poll
    and a resume that died on spawn was reported as success -- the spawn's
    stderr goes to DEVNULL, so nothing else could surface it. `live` is what
    separates the two, and both these states must read false."""
    parked = _bucket(tmp_path, "runs") / "parked-live"
    parked.mkdir(parents=True)
    (parked / "manifest.json").write_text(
        json.dumps({"session_id": "parked-live", "parked_task": "t"}), encoding="utf-8"
    )
    stale = _run(
        tmp_path, "stale-live", [{"type": "session.start", "mode": "run", "user_task": "t"}]
    )
    (stale / "worker.pid").write_text("999999 12345678", encoding="utf-8")  # dead pid

    for d in (parked, stale):
        snap = session_snapshot(d)
        assert snap["finished"] is False, d.name  # why the old poll fired at once
        assert snap["live"] is False, d.name  # ...and what actually distinguishes them


def test_conversation_payload_carries_an_in_flight_call(tmp_path: Path) -> None:
    """The page rebuilds its items from each payload, so a call the fold
    reports in flight shows as running until its result replaces it."""
    from agent6.sessions.ipc import write_worker_pid

    call = {"type": "tool.call", "name": "run_command", "args": {"argv": ["sleep", "60"]}}
    d = _run(tmp_path, "r3", [{"type": "session.start", "user_task": "x"}, {**call, "call_id": 1}])
    write_worker_pid(d, os.getpid())  # a live worker: the call is in flight

    def flat_items() -> list[str]:
        items = model.conversation_payload(d)["items"]
        return ["".join(t for line in it["lines"] for t, _s in line) for it in items]

    assert flat_items() == ["→ run_command  sleep 60  · running"]
    result = {"type": "tool.result", "name": "run_command", "ok": True, "summary": "exit 0"}
    with (d / "logs.jsonl").open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({**result, "call_id": 1}) + "\n")
    (settled,) = flat_items()
    assert "exit 0" in settled and "running" not in settled


def test_a_dead_workers_open_call_reads_dead_not_running(tmp_path: Path) -> None:
    """No session.end will ever settle a call the killed worker left open; the
    payload probes the worker and settles it, and /restate agrees."""
    call = {"type": "tool.call", "name": "run_command", "args": {"argv": ["sleep", "60"]}}
    d = _run(tmp_path, "r4", [{"type": "session.start", "user_task": "x"}, {**call, "call_id": 1}])
    (d / "worker.pid").write_text("4194304", encoding="utf-8")  # past pid_max: gone
    (item,) = model.conversation_payload(d)["items"]
    flat = "".join(t for line in item["lines"] for t, _s in line)
    assert "no result (the run died)" in flat and "running" not in flat
    assert "no result (the run died)" in model.restate_payload(d)["text"]


def test_the_hub_row_and_the_cli_json_row_are_one_shape(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """One name per fact: `/api/hub` said `id`/`usd`/`mtime` where `sessions
    list --json` said `session_id`/`cost_usd`/`updated`, so a script reading
    one could not read the other."""
    from agent6.ui.cli.sessions_cmds import _cmd_list  # pyright: ignore[reportPrivateUsage]

    monkeypatch.chdir(tmp_path)
    _run(
        tmp_path,
        "r1",
        [
            {"type": "session.start", "mode": "run", "user_task": "the task"},
            {"type": "session.end", "all_passed": True},
        ],
    )

    (hub_row,) = model.hub_payload(tmp_path)["sessions"]
    assert _cmd_list(as_json=True) == 0
    (cli_row,) = json.loads(capsys.readouterr().out)

    assert set(hub_row) == set(cli_row)
    assert hub_row["session_id"] == cli_row["session_id"] == "r1"


def test_a_waiting_machine_is_not_labelled_failed(tmp_path: Path) -> None:
    """`reason` is set for a live machine blocked on an operator prompt as well
    as for a failed end; hardcoding "failed · <reason>" told the operator it
    had died instead of sending them to answer the prompt."""
    from agent6.viewmodel.machine_state import MachineSummary

    row = model._machine_row(  # pyright: ignore[reportPrivateUsage]
        MachineSummary(
            name="inst",
            machine="m",
            status="waiting",
            reason="waiting on an approval in 0001-work",
            current="work",
            mtime=0.0,
        )
    )

    assert row["label"] == "waiting · waiting on an approval in 0001-work"
    assert row["level"] == "warn"


def test_the_web_machine_header_does_not_hide_a_zero_cost(tmp_path: Path) -> None:
    """`machine status` and the TUI watch print `spend: $0.0000`; the web
    header appended the cost only when `spend.usd` was truthy, so the figure
    an unattended machine is watched for was missing while it was zero."""
    from agent6.machine.journal import BranchFact, MachineJournal, StepEvent
    from agent6.ui.web.page import CLIENT_JS

    (tmp_path / "machine.asm.toml").write_text(
        """machine = "tiny"
version = 1
initial = "route"

[budget]
max_transitions = 10

[vars.code]
n = { type = "int", default = 0 }

[states.route]
kind = "branch"
when = [{ if = "n == 0", goto = "done" }, { else = true, goto = "done" }]

[states.done]
kind = "terminal"
status = "ok"
reason = "routed"
""",
        encoding="utf-8",
    )
    j = MachineJournal(tmp_path)
    j.ensure_dirs()
    j.begin(machine="tiny", version=1)
    j.append(
        StepEvent(
            ts="t",
            seq=0,
            state="route",
            label="n == 0",
            goto="done",
            fact=BranchFact(clause_index=0),
        )
    )
    snap = machine_snapshot(tmp_path)
    assert snap["spend"] == {"usd": 0.0, "usd_partial": False} or snap["spend"]["usd"] == 0.0

    start = CLIENT_JS.index("const sp = m.spend || {};")
    line = CLIENT_JS[start : CLIENT_JS.index("\n", CLIENT_JS.index("const cost", start))]
    assert "sp.usd ||" not in line, f"a clean $0 is falsy and drops the figure: {line!r}"
