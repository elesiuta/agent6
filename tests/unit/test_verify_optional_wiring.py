# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Gateless wiring: with no verify_command, the verify tool is hidden and the
system prompt swaps the verify block for the no-verify block."""

from __future__ import annotations

import dataclasses
import json
import subprocess as sp
from collections.abc import Callable
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from agent6.config import Config
from agent6.config.layer import EffectiveConfig, resolved_state_dir
from agent6.tools.dispatch import ToolDispatcher
from agent6.types import RepoSummary
from agent6.workflows._prompt_blocks import build_system_prompt
from agent6.workflows._verify_verdict import VerifyVerdict


def _cfg(*, verify: bool) -> Config:
    data = {"workflow": {"verify_command": ["true"]}} if verify else {}
    return Config.model_validate(data)


def _repo(root: Path) -> RepoSummary:
    return RepoSummary(
        root=root,
        branch="main",
        head_sha="0" * 40,
        file_count=0,
        top_level=(),
        agents_md="",
        recent_log="",
    )


def test_verify_tool_hidden_when_command_unset(tmp_path: Path) -> None:
    with_verify = ToolDispatcher(root=tmp_path, config=_cfg(verify=True))
    gateless = ToolDispatcher(root=tmp_path, config=_cfg(verify=False))
    assert "run_verify_command" in with_verify.available_tool_names()
    assert "run_verify_command" not in gateless.available_tool_names()


def test_adopt_verify_command_probes_the_jail_path(tmp_path: Path) -> None:
    """Mid-run adoption refuses a bare runner the jail PATH cannot resolve
    (adopting it would turn an honest settle into an unexecutable-verify
    abort) and accepts a resolvable one, which also unhides the verify tool.
    Path-form commands pass through: they resolve against the mounted cwd."""
    d = ToolDispatcher(root=tmp_path, config=_cfg(verify=False))
    assert d.adopt_verify_command(("no-such-binary-zq9", "test")) is False
    assert "run_verify_command" not in d.available_tool_names()
    assert d.adopt_verify_command(("sh", "-c", "true")) is True
    assert "run_verify_command" in d.available_tool_names()
    d2 = ToolDispatcher(root=tmp_path, config=_cfg(verify=False))
    assert d2.adopt_verify_command(("./scripts/check.sh",)) is True


def test_system_prompt_switches_verify_block(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    with_verify = build_system_prompt(config=_cfg(verify=True), repo=repo, mode="run", skills=None)
    gateless = build_system_prompt(config=_cfg(verify=False), repo=repo, mode="run", skills=None)
    assert "<verify-command>" in with_verify and "<no-verify-command>" not in with_verify
    assert "<no-verify-command>" in gateless and "<verify-command>" not in gateless
    # Every verify rule lives INSIDE the conditional block: the base leaked
    # gate prose ("run project tests only through...", the stale_gate rule,
    # "after each passing verify") into gateless prompts, which then needed
    # an "Ignore any other instruction" patch-line to disarm it.
    assert gateless.count("run_verify_command") == 1  # the block's own "not available"
    assert "stale_gate" not in gateless and "passing verify" not in gateless
    assert "commits each editing step" in gateless  # the gate-aware commit rule
    assert "run project tests only through" not in gateless.lower()
    assert "stale_gate" in with_verify and "commits each editing step" in with_verify
    # The per-step commit rule belongs to a gate that judges each step.
    never = Config.model_validate(
        {"workflow": {"verify_command": ["true"], "verify_when": "never"}}
    )
    assert "after each passing verify" in build_system_prompt(
        config=never, repo=repo, mode="run", skills=None
    )


def test_no_verify_block_wording_matches_the_mode(tmp_path: Path) -> None:
    """The gateless block states the gate's absence and nothing else, in every
    mode: the terminal tool is each base prompt's fact (run: `finish_session
    ends the run`; plan: `finish_planning` ends the pass); ask has none."""
    repo = _repo(tmp_path)
    cfg = _cfg(verify=False)
    run = build_system_prompt(config=cfg, repo=repo, mode="run", skills=None)
    plan = build_system_prompt(config=cfg, repo=repo, mode="plan", skills=None)
    ask = build_system_prompt(config=cfg, repo=repo, mode="ask", skills=None)

    def block(text: str) -> str:
        start = text.index("<no-verify-command>")
        return text[start : text.index("</no-verify-command>", start)]

    run_block, plan_block, ask_block = block(run), block(plan), block(ask)
    assert "finish_session" not in run_block and "finish_session ends the run" in run
    assert "finish_planning" not in plan_block and "`finish_planning` ends the pass" in plan
    assert "finish_session" not in plan_block and "commits" not in plan_block
    assert "finish_session" not in ask_block and "finish_planning" not in ask_block
    assert "commits" not in ask_block
    # The commit claim is the base sentinel's, one owner (run mode only);
    # the block no longer needs an "Ignore any other instruction" patch-line
    # because no verify prose leaks outside the verify block.
    assert "commits" not in run_block and "commits each editing step" in run
    for b in (run_block, plan_block, ask_block):
        assert "Ignore any" not in b


def test_a_leg_that_cannot_run_commands_is_gateless_wherever_it_starts(tmp_path: Path) -> None:
    """The rule lived only in preflight's fresh-run path, so a RESUMED leg was
    re-gated with every command tool withheld: nothing could go green, the leg
    committed nothing, and the manifest was re-pinned to claim a gate that
    never judged anything. Both lifecycles make the decision now, once, at leg
    start -- with the system prompt, which is frozen from the same config."""
    from agent6.app.preflight import drop_gate_if_unrunnable
    from agent6.app.reporter import Reporter
    from agent6.sessions.ipc import set_away_mode

    session_dir = tmp_path / "run"
    session_dir.mkdir()
    said: list[str] = []
    reporter = Reporter(out=said.append, err=said.append)
    gated = Config.model_validate({"workflow": {"verify_command": ["pytest", "-q"]}})

    assert drop_gate_if_unrunnable(
        gated, session_dir=session_dir, reporter=reporter
    ).workflow.verify_command == (
        "pytest",
        "-q",
    )
    withheld = Config.model_validate(
        {"workflow": {"verify_command": ["pytest", "-q"]}, "sandbox": {"run_commands": "no"}}
    )
    assert (
        drop_gate_if_unrunnable(
            withheld, session_dir=session_dir, reporter=reporter
        ).workflow.verify_command
        == ()
    )
    assert any("running gateless" in line for line in said)

    # An away-mode of deny reaches the same answer: the EFFECTIVE policy, not
    # just the configured knob.
    set_away_mode(session_dir, "deny")
    assert (
        drop_gate_if_unrunnable(
            gated, session_dir=session_dir, reporter=reporter
        ).workflow.verify_command
        == ()
    )


def test_a_deny_after_a_red_gate_does_not_turn_the_run_green(tmp_path: Path) -> None:
    """Reading the LIVE policy for the verdict made a mid-run "deny for the
    rest of the run" erase a gate that had already run and failed: verified
    flipped to not_applicable, the exit code to 0, and `git.auto_merge` merged
    the red branch. Gatedness is frozen at leg start; a later deny withdraws
    the tools, never the verdict."""
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    from agent6.workflows.loop import LoopState, Workflow

    wf = Workflow.__new__(Workflow)
    wf.config = SimpleNamespace(  # pyright: ignore[reportAttributeAccessIssue]
        workflow=SimpleNamespace(verify_command=("pytest", "-q"))
    )
    wf.dispatcher = MagicMock()
    wf.dispatcher.command_policy.return_value = "no"  # denied mid-run
    state = MagicMock(spec=LoopState)
    state.verify = VerifyVerdict(last_ok=False, edited_since=False)

    assert wf._tree_is_verify_green(state) is False  # pyright: ignore[reportPrivateUsage]


def test_a_deny_mid_run_takes_the_gate_with_it(tmp_path: Path) -> None:
    """`deny for the rest of the run` and an away-mode of deny both flip the
    EFFECTIVE policy to "no" while the config still names a gate. The leg kept
    the gate, lost the tool, and ended red."""
    from agent6.config import Config
    from agent6.sessions.ipc import set_away_mode
    from agent6.tools.dispatch import ToolDispatcher

    session_dir = tmp_path / "run"
    session_dir.mkdir()
    cfg = Config.model_validate({"workflow": {"verify_command": ["true"]}})
    d = ToolDispatcher(root=tmp_path, config=cfg, session_dir=session_dir)
    assert "run_verify_command" in d.available_tool_names()
    set_away_mode(session_dir, "deny")
    assert d.command_policy() == "no"
    assert "run_verify_command" not in d.available_tool_names()


def test_a_gate_is_never_adopted_when_the_worker_cannot_run_one(tmp_path: Path) -> None:
    """Adoption checked the jail PATH but not the policy, so a --no-commands
    run re-acquired a gate mid-run and undid the preflight drop."""
    from agent6.config import Config
    from agent6.tools.dispatch import ToolDispatcher

    session_dir = tmp_path / "run"
    session_dir.mkdir()
    cfg = Config.model_validate({"sandbox": {"run_commands": "no"}})
    d = ToolDispatcher(root=tmp_path, config=cfg, session_dir=session_dir)
    assert d.adopt_verify_command(("/bin/true",)) is False
    assert d._config.workflow.verify_command == ()  # pyright: ignore[reportPrivateUsage]


def test_the_worker_gets_the_tool_for_a_gate_adopted_mid_run(tmp_path: Path) -> None:
    """The tool list was built once per leg. A gateless run that adopted a gate
    was TOLD to run run_verify_command while that tool was absent from every
    remaining call: commits stopped, the finish was graded failed, exit 4."""
    from agent6.config import Config
    from agent6.tools.dispatch import ToolDispatcher
    from agent6.workflows._toolset import tool_definitions

    d = ToolDispatcher(root=tmp_path, config=Config())
    before = {t.name for t in tool_definitions(d, mode="run")}
    assert "run_verify_command" not in before

    assert d.adopt_verify_command(("/bin/true",)) is True
    after = {t.name for t in tool_definitions(d, mode="run")}
    assert "run_verify_command" in after, "the adopted gate has no tool to run it"


class _Stop(Exception):
    """Sentinel: the lifecycle reached pin_gate with this leg's final gate."""


def _git_repo(path: Path) -> None:
    sp.run(["git", "init", "-q", "-b", "main"], cwd=path, check=True)
    (path / "seed.txt").write_text("seed\n")
    sp.run(["git", "add", "-A"], cwd=path, check=True)
    sp.run(["git", "commit", "-q", "-m", "init"], cwd=path, check=True)


def _role_cfg(extra: dict[str, object]) -> Config:
    # Callers setenv("K", ...) so provider construction finds the key.
    return Config.model_validate(
        {
            "providers": {"anthropic": {"api_format": "anthropic", "api_key_env": "K"}},
            "models": {
                "worker": {"provider": "anthropic", "model": "m"},
                "reviewer": {"provider": "anthropic", "model": "m"},
            },
            **extra,
        }
    )


def _capture_pin(pinned: list[tuple[tuple[str, ...], str]]) -> Callable[..., None]:
    def _pin(_dir: Path, argv: object, origin: str, **_k: object) -> None:
        pinned.append((tuple(argv), origin))  # pyright: ignore[reportArgumentType]
        raise _Stop()

    return _pin


def test_a_withheld_resumed_leg_is_not_regated_by_the_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The drop must have the last word at leg start. With commands withheld,
    the snapshot-reuse block ran AFTER drop_gate_if_unrunnable and handed the
    dropped gate straight back: the leg resumed gated-but-unwinnable, printed
    two contradictory preamble lines, committed nothing all leg, and exited 4
    over a gate that never ran."""
    import agent6.app._session as session_mod
    import agent6.app._setup as setup_mod
    import agent6.app.resume as resume_mod
    from agent6.app.reporter import Reporter
    from agent6.ui.cli.run import session_frontend

    repo = tmp_path / "repo"
    repo.mkdir()
    _git_repo(repo)
    monkeypatch.chdir(repo)
    session_dir = resolved_state_dir(repo) / "sessions" / "runs" / "withheld-AAAA11"
    session_dir.mkdir(parents=True)
    (session_dir / "manifest.json").write_text(
        json.dumps(
            {"version": 3, "session_id": "withheld-AAAA11", "mode": "run", "user_task": "t"}
        ),
        encoding="utf-8",
    )
    (session_dir / "loop_state.json").write_text(
        json.dumps(
            {
                "version": 2,
                "system": "s",
                "messages": [],
                "tool_calls": 0,
                "next_iteration": 1,
                "root_task_id": None,
                "original_task": "t",
                "verify_command": ["pytest", "-q"],
            }
        ),
        encoding="utf-8",
    )
    cfg = _role_cfg({"sandbox": {"run_commands": "no"}})
    monkeypatch.setenv("K", "test-key")

    def _none(*_a: object, **_k: object) -> None:
        return None

    # The real type: preflight reads `explicit_leaves` to tell a default this
    # host cannot honour (degrade) from a value the operator set (refuse).
    def _load(*_a: object, **_k: object) -> EffectiveConfig:
        return EffectiveConfig(config=cfg, sources={}, layers=())

    def _strict(*_a: object, **_k: object) -> str:
        return "strict"

    def _provider(*_a: object, **_k: object) -> MagicMock:
        return MagicMock()

    def _yes(*_a: object) -> bool:
        return True

    pinned: list[tuple[tuple[str, ...], str]] = []
    monkeypatch.setattr(setup_mod, "load_effective", _load)
    monkeypatch.setattr(session_mod, "detect_env", object)
    monkeypatch.setattr(session_mod, "resolve_isolation", _strict)
    monkeypatch.setattr(session_mod, "warn_sandbox_gaps", _none)
    monkeypatch.setattr(session_mod, "check_network_support", _none)
    monkeypatch.setattr(session_mod, "budget_preflight", _none)
    monkeypatch.setattr(session_mod, "build_role_provider", _provider)
    monkeypatch.setattr(resume_mod, "check_provider_keys", _none)
    monkeypatch.setattr(resume_mod, "verify_git_identity", _none)
    monkeypatch.setattr(resume_mod, "pin_gate", _capture_pin(pinned))

    said: list[str] = []
    frontend = dataclasses.replace(session_frontend(), confirm_unconfined_autorun=_yes)
    with pytest.raises(_Stop):
        resume_mod.resume_task(
            None,
            "withheld-AAAA11",
            frontend=frontend,
            force=False,
            reporter=Reporter(out=said.append, err=said.append),
        )
    assert pinned == [((), "")], f"the withheld leg was re-gated: {pinned}"
    assert any("running gateless" in line for line in said)


def test_a_withheld_fresh_leg_is_not_regated_by_inference(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The same rule at the other lifecycle: with commands withheld, inference
    ran AFTER the drop, re-gated the leg from AGENTS.md, and the pin labelled
    the inferred command "configured"."""
    import agent6.app._session as session_mod
    import agent6.app.preflight as preflight_mod
    import agent6.app.run as run_mod
    from agent6.app.reporter import Reporter

    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "AGENTS.md").write_text("Verify: make check\n", encoding="utf-8")
    _git_repo(repo)
    monkeypatch.chdir(repo)
    monkeypatch.setenv("K", "test-key")
    cfg = _role_cfg(
        {
            "sandbox": {"run_commands": "no"},
            "workflow": {"verify_command": ["pytest", "-q"]},
            "git": {"branch_per_run": False},
        }
    )

    def _none(*_a: object, **_k: object) -> None:
        return None

    def _strict(*_a: object, **_k: object) -> str:
        return "strict"

    def _provider(*_a: object, **_k: object) -> MagicMock:
        return MagicMock()

    pinned: list[tuple[tuple[str, ...], str]] = []
    monkeypatch.setattr(session_mod, "detect_env", object)
    monkeypatch.setattr(session_mod, "resolve_isolation", _strict)
    monkeypatch.setattr(session_mod, "warn_sandbox_gaps", _none)
    monkeypatch.setattr(session_mod, "check_network_support", _none)
    monkeypatch.setattr(session_mod, "budget_preflight", _none)
    monkeypatch.setattr(session_mod, "build_role_provider", _provider)
    monkeypatch.setattr(preflight_mod, "verify_git_identity", _none)
    monkeypatch.setattr(run_mod, "pin_gate", _capture_pin(pinned))

    said: list[str] = []
    frontend = MagicMock()
    frontend.should_spawn_tui.return_value = False
    frontend.stream_modes.return_value = (False, False)
    with pytest.raises(_Stop):
        run_mod.run_task(
            cfg,
            "t",
            frontend=frontend,
            mode="run",
            reporter=Reporter(out=said.append, err=said.append),
        )
    assert pinned == [((), "")], f"the withheld leg was re-gated: {pinned}"
    assert any("running gateless" in line for line in said)


def test_hardened_fs_rule_renders_only_under_hardened(tmp_path: Path) -> None:
    """The hardened create-a-top-level-entry workaround is a real constraint
    only under hardened; a strict run reading it would route file creation
    through apply_edit for no reason (found reading a strict run's real
    prompt: the rule rendered unconditionally)."""
    repo = _repo(tmp_path)
    cfg = _cfg(verify=True)
    strict = build_system_prompt(
        config=cfg, repo=repo, mode="run", skills=None, isolation="strict", protected_paths=True
    )
    hardened = build_system_prompt(
        config=cfg, repo=repo, mode="run", skills=None, isolation="hardened", protected_paths=True
    )
    assert "Under hardened isolation" not in strict
    assert "__HARDENED_FS_RULE__" not in strict
    assert "Under hardened isolation" in hardened


def test_git_protect_rule_renders_only_when_the_bind_exists(tmp_path: Path) -> None:
    """The .git read-only bind exists only under strict with protect_git on;
    every unjailed run (isolation none, e.g. the SWE-bench containers) was
    told '.git/ is protected inside the jail' while nothing protected it."""
    repo = _repo(tmp_path)
    on = _cfg(verify=True)
    off = Config.model_validate(
        {"workflow": {"verify_command": ["true"]}, "sandbox": {"protect_git": False}}
    )
    marker = ".git/` is read-only inside the jail"
    for isolation, cfg, expect in (
        ("strict", on, True),
        ("strict", off, False),
        ("hardened", on, False),
        ("none", on, False),
    ):
        out = build_system_prompt(
            config=cfg,
            repo=repo,
            mode="run",
            skills=None,
            isolation=isolation,  # pyright: ignore[reportArgumentType]
        )
        assert (marker in out) is expect, (isolation, expect)
        assert "__GIT_PROTECT_RULE__" not in out


def test_agents_md_section_absent_when_repo_has_none(tmp_path: Path) -> None:
    """A repo without AGENTS.md got an 'AGENTS.md (project conventions):
    (empty)' header on every run -- noise standing where signal goes."""
    repo = _repo(tmp_path)
    out = build_system_prompt(config=_cfg(verify=True), repo=repo, mode="run", skills=None)
    assert "AGENTS.md (project conventions):" not in out
    assert "(empty)" not in out


def test_prompt_git_rules_match_git_control(tmp_path: Path) -> None:
    """The prompt states the world that exists: under [git].control = "model"
    the auto-commit chain does not run, so claiming "the harness commits
    automatically" (base block and gateless block both did) misdirects the
    model into never committing."""
    repo = _repo(tmp_path)
    agent6_cfg = Config.model_validate({"workflow": {"verify_command": ["true"]}})
    model_cfg = Config.model_validate(
        {
            "workflow": {"verify_command": ["true"]},
            "git": {"control": "model"},
            "sandbox": {"protect_git": False},
        }
    )
    agent6_prompt = build_system_prompt(config=agent6_cfg, repo=repo, mode="run", skills=None)
    model_prompt = build_system_prompt(config=model_cfg, repo=repo, mode="run", skills=None)
    assert "The harness commits" in agent6_prompt
    assert "You own git" not in agent6_prompt
    assert "The harness commits" not in model_prompt
    assert "You own git" in model_prompt

    gateless_cfg = Config.model_validate(
        {"git": {"control": "model"}, "sandbox": {"protect_git": False}}
    )
    gateless = build_system_prompt(
        config=gateless_cfg,
        repo=repo,
        mode="run",
        skills=None,
    )
    start = gateless.index("<no-verify-command>")
    block = gateless[start : gateless.index("</no-verify-command>", start)]
    assert "finish_session" not in block and "finish_session ends the run" in gateless
    assert "commits each editing step" not in block


def test_budget_block_names_the_plan_meter_for_subscription_runs(tmp_path: Path) -> None:
    """A subscription run meters in plan percent; a budget block naming only
    USD and fallback caps describes meters that never bind it. The plan line
    renders exactly when a configured role rides a chatgpt provider."""
    repo = _repo(tmp_path)
    sub = Config.model_validate(
        {
            "providers": {"gpt": {"api_format": "chatgpt"}},
            "models": {
                "worker": {"provider": "gpt", "model": "gpt-5.6-sol"},
                "reviewer": {"provider": "gpt", "model": "gpt-5.6-sol"},
            },
            "budget": {"max_percent": 3},
        }
    )
    prompt = build_system_prompt(config=sub, repo=repo, mode="run", skills=None)
    assert "meter in plan percent (max_percent 3 points per run)" in prompt
    plain = build_system_prompt(config=Config(), repo=repo, mode="run", skills=None)
    assert "plan percent" not in plain


def test_verify_infer_false_pins_gatelessness_at_preflight(tmp_path: Path) -> None:
    """An unset verify_command always ran the inference tiers; with a repo
    that infers (an AGENTS.md fence here) the run could never be made
    gateless on purpose. verify_infer = false skips every tier."""
    import json
    from unittest.mock import MagicMock

    from agent6.app.preflight import infer_verify_if_unset
    from agent6.budget import BudgetTracker
    from agent6.config import Config
    from agent6.events import EventSink

    (tmp_path / "AGENTS.md").write_text(
        "## Verify command\n\n```bash\ntrue\n```\n", encoding="utf-8"
    )
    budget = BudgetTracker(max_usd=-1.0, max_tokens_fallback=-1, max_percent=-1.0)

    cfg_on = infer_verify_if_unset(
        Config(),
        tmp_path,
        mode="run",
        events=EventSink(tmp_path / "on.jsonl"),
        transcript_sink=MagicMock(),
        budget=budget,
    )
    assert cfg_on.workflow.verify_command, "the fence must infer when the knob is on"

    off_log = tmp_path / "off.jsonl"
    cfg_off = infer_verify_if_unset(
        Config.model_validate({"workflow": {"verify_infer": False}}),
        tmp_path,
        mode="run",
        events=EventSink(off_log),
        transcript_sink=MagicMock(),
        budget=budget,
    )
    assert cfg_off.workflow.verify_command == ()
    rows = [json.loads(line) for line in off_log.read_text(encoding="utf-8").splitlines()]
    assert any(r["type"] == "loop.verify_inferred" and r["source"] == "disabled" for r in rows)


def test_verify_infer_false_pins_gatelessness_at_adoption(tmp_path: Path) -> None:
    """The mid-run adoption re-armed a gate on every gateless run whose tree
    is a recognizable project; inside a container whose python3 lacks pytest
    that gate was an always-red no-op. The same knob turns adoption off."""
    from unittest.mock import MagicMock

    from agent6.config import Config
    from agent6.workflows.loop import TurnState, Workflow

    (tmp_path / "AGENTS.md").write_text(
        "## Verify command\n\n```bash\ntrue\n```\n", encoding="utf-8"
    )
    dispatcher = MagicMock()
    wf = Workflow(
        root=tmp_path,
        config=Config.model_validate({"workflow": {"verify_infer": False}}),
        provider=MagicMock(),
        dispatcher=dispatcher,
        logger=lambda _line: None,
    )
    turn = TurnState(iteration=1, resp=MagicMock(), assistant=MagicMock())
    wf._maybe_adopt_verify(MagicMock(), turn)  # pyright: ignore[reportPrivateUsage]
    dispatcher.adopt_verify_command.assert_not_called()
    assert wf.config.workflow.verify_command == ()


def test_prompt_says_nothing_commits_under_commit_per_step_off(tmp_path: Path) -> None:
    """With `[git].commit_per_step = false` nothing commits, and the prompt
    still promised a commit after every passing verify; the model's work
    stayed uncommitted in the worktree while it was told otherwise."""
    repo = _repo(tmp_path)
    cfg = Config.model_validate(
        {"workflow": {"verify_command": ["true"]}, "git": {"commit_per_step": False}}
    )
    out = build_system_prompt(config=cfg, repo=repo, mode="run", skills=None)
    assert "Nothing commits automatically" in out
    assert "The harness commits" not in out and "You own git" not in out


def test_hardened_rule_renders_only_where_the_jail_carries_protect_paths(tmp_path: Path) -> None:
    """Landlock denies new top-level entries only when it carves around
    protect paths (a machine's bundle, a read-only session); an ordinary
    hardened run has none and was told to `apply_edit` placeholders it never
    needed."""
    repo = _repo(tmp_path)
    cfg = Config.model_validate({"workflow": {"verify_command": ["true"]}})
    plain = build_system_prompt(
        config=cfg, repo=repo, mode="run", skills=None, isolation="hardened"
    )
    assert "cannot CREATE new" not in plain
    carved = build_system_prompt(
        config=cfg, repo=repo, mode="run", skills=None, isolation="hardened", protected_paths=True
    )
    assert "cannot CREATE new" in carved
