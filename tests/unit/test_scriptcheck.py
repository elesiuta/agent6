# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Tests for `machine create`/`check`/`test` script validation (cli/scriptcheck)."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from agent6.app.machine import _scriptcheck as scriptcheck
from agent6.types import CommandResult, JailPolicy

_CLEAN = "import json\n\n\ndef f(x: int) -> str:\n    return json.dumps({'v': x})\n"


def _write(scripts_dir: Path, name: str, body: str) -> None:
    scripts_dir.mkdir(parents=True, exist_ok=True)
    (scripts_dir / name).write_text(body, encoding="utf-8")


def _need(tool: str) -> None:
    if tool not in scriptcheck.available_tools():
        pytest.skip(f"{tool} not installed in this environment")


# --- static: ruff + ty ------------------------------------------------------


def test_lint_typecheck_clean(tmp_path: Path) -> None:
    _need("ruff")
    _need("ty")
    _write(tmp_path / "scripts", "ok.py", _CLEAN)
    assert scriptcheck.lint_and_typecheck(tmp_path / "scripts") == []


def test_static_checks_disable_python_bytecode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write(tmp_path / "scripts", "ok.py", _CLEAN)
    monkeypatch.setenv("PYTHONDONTWRITEBYTECODE", "0")
    seen_env: dict[str, str] = {}

    def _resolve_tool(name: str) -> list[str] | None:
        return ["fake-tool"] if name == "ruff" else None

    def _run(
        _argv: list[str],
        *,
        capture_output: bool,
        text: bool,
        timeout: float,
        cwd: Path,
        check: bool,
        env: dict[str, str],
    ) -> object:
        del capture_output, text, timeout, cwd, check
        seen_env.update(env)
        return type("Result", (), {"returncode": 0, "stdout": "", "stderr": ""})()

    monkeypatch.setattr(scriptcheck, "_resolve_tool", _resolve_tool)
    monkeypatch.setattr(scriptcheck.subprocess, "run", _run)

    assert scriptcheck.lint_and_typecheck(tmp_path / "scripts") == []
    assert seen_env["PYTHONDONTWRITEBYTECODE"] == "1"


def test_lint_catches_undefined_name(tmp_path: Path) -> None:
    _need("ruff")
    _write(tmp_path / "scripts", "bad.py", "print(undefined_name)\n")
    problems = scriptcheck.lint_and_typecheck(tmp_path / "scripts")
    assert any("ruff" in p for p in problems)


def test_typecheck_catches_type_error(tmp_path: Path) -> None:
    _need("ty")
    _write(tmp_path / "scripts", "bad.py", "def f(x: str) -> int:\n    return x + 1\n")
    problems = scriptcheck.lint_and_typecheck(tmp_path / "scripts")
    assert any("ty" in p for p in problems)


def test_typecheck_skips_test_files(tmp_path: Path) -> None:
    """ty is NOT run on *_test.py (mock internals trip it); ruff still is."""
    _need("ty")
    # A type error that only ty would catch, in a *_test.py file -> not flagged.
    _write(tmp_path / "scripts", "x_test.py", "def f(a: str) -> int:\n    return a + 1\n")
    problems = scriptcheck.lint_and_typecheck(tmp_path / "scripts")
    assert not any("ty" in p for p in problems)


def test_no_python_scripts_is_clean(tmp_path: Path) -> None:
    _write(tmp_path / "scripts", "run.sh", "#!/bin/sh\necho hi\n")
    assert scriptcheck.lint_and_typecheck(tmp_path / "scripts") == []


def test_missing_scripts_dir_is_clean(tmp_path: Path) -> None:
    assert scriptcheck.lint_and_typecheck(tmp_path / "nope") == []


def test_lint_follows_the_bundles_ruff_config(tmp_path: Path) -> None:
    """`machine check` linted with `ruff --isolated`, so its verdict tracked the
    installed ruff's default rules rather than anything the operator wrote: the
    shipped code-fixer bundle failed on rules a newer ruff turned on. Ruff now
    runs on the real files, so its own discovery applies and the nearest config
    above the machine file (here the bundle's own ruff.toml) pins the rules."""
    _need("ruff")
    (tmp_path / "ruff.toml").write_text('[lint]\nignore = ["F401"]\n', encoding="utf-8")
    _write(tmp_path / "scripts", "imports.py", "import json\n")
    assert scriptcheck.lint_and_typecheck(tmp_path / "scripts") == []


def test_create_fix_mode_lints_under_the_destinations_config(tmp_path: Path) -> None:
    """`machine create` drafts in a scratch dir outside the repo, where ruff's
    discovery cannot see the config the published bundle will be checked under;
    `ruff_config_from` resolves it from the publish destination so the draft
    gate and the operator's later `machine check` agree."""
    _need("ruff")
    dest = tmp_path / "repo"
    dest.mkdir()
    (dest / "ruff.toml").write_text('[lint]\nignore = ["F401"]\n', encoding="utf-8")
    body = "import json\n"
    _write(tmp_path / "scratch" / "scripts", "imports.py", body)
    problems = scriptcheck.lint_and_typecheck(
        tmp_path / "scratch" / "scripts", fix=True, ruff_config_from=dest
    )
    assert not any("ruff" in p for p in problems)
    # F401 is off in the destination's config: nothing to fix, file untouched.
    assert (tmp_path / "scratch" / "scripts" / "imports.py").read_text(encoding="utf-8") == body


# --- dynamic: offline test execution (jail patched, no fork) ----------------


def _fake_jail(returncode: int, stderr: str = "") -> object:
    def run(policy: object) -> CommandResult:
        # Real jailed stderr names the TEMP COPY the runner executes in (the
        # real bundle is under the masked state dir); the fake mirrors that by
        # substituting the policy's cwd for a {cwd} placeholder.
        cwd = str(getattr(policy, "cwd", ""))
        return CommandResult(
            argv=("python3", "scripts/thing_test.py"),
            returncode=returncode,
            stdout="",
            stderr=stderr.replace("{cwd}", cwd),
            duration_s=0.0,
        )

    return run


def test_offline_tests_pass(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _write(tmp_path / "scripts", "thing_test.py", "print('ok')\n")
    monkeypatch.setattr(scriptcheck, "run_in_jail", _fake_jail(0))
    assert scriptcheck.run_offline_tests(tmp_path, "strict").problems == ()


def test_offline_tests_disable_python_bytecode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write(tmp_path / "scripts", "thing_test.py", "print('ok')\n")
    seen: list[JailPolicy] = []

    def _run(policy: JailPolicy) -> CommandResult:
        seen.append(policy)
        return CommandResult(
            argv=policy.argv,
            returncode=0,
            stdout="",
            stderr="",
            duration_s=0.0,
        )

    monkeypatch.setattr(scriptcheck, "run_in_jail", _run)

    assert scriptcheck.run_offline_tests(tmp_path, "strict").problems == ()
    assert ("PYTHONDONTWRITEBYTECODE", "1") in seen[0].env


def test_offline_tests_fail_surfaces_stderr(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write(tmp_path / "scripts", "thing_test.py", "raise SystemExit(1)\n")
    monkeypatch.setattr(scriptcheck, "run_in_jail", _fake_jail(1, "AssertionError: boom"))
    problems = scriptcheck.run_offline_tests(tmp_path, "strict").problems
    assert len(problems) == 1
    assert "thing_test.py" in problems[0]
    assert "boom" in problems[0]


def test_offline_tests_relativize_bundle_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Tracebacks from the jailed test name the absolute bundle dir; the
    # diagnostic is fed back into the authoring prompt, so host paths get
    # stripped down to bundle-relative ones.
    _write(tmp_path / "scripts", "thing_test.py", "raise SystemExit(1)\n")
    stderr = 'File "{cwd}/scripts/thing_test.py", line 1\nNameError: x'
    monkeypatch.setattr(scriptcheck, "run_in_jail", _fake_jail(1, stderr))
    problems = scriptcheck.run_offline_tests(tmp_path, "strict").problems
    assert "agent6-scripttest" not in problems[0]  # the temp copy's path is stripped
    assert 'File "scripts/thing_test.py"' in problems[0]


def test_offline_tests_skipped_on_none_profile(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write(tmp_path / "scripts", "thing_test.py", "print('ok')\n")
    called = False

    def _boom(_policy: object) -> CommandResult:  # pragma: no cover - must not run
        nonlocal called
        called = True
        return CommandResult(argv=(), returncode=0, stdout="", stderr="", duration_s=0.0)

    monkeypatch.setattr(scriptcheck, "run_in_jail", _boom)
    outcome = scriptcheck.run_offline_tests(tmp_path, "none")
    assert outcome.problems == ()
    assert outcome.skipped == 1 and "no sandbox" in outcome.skip_reason
    assert not called


def test_offline_tests_skipped_on_hardened_profile(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # hardened has a jail but no network namespace, so network="none" cannot
    # be honored: model-authored scripts would reach the host network. They must
    # NOT run, and the outcome carries the skip for the caller's verdict (a
    # stderr aside beside an OK verdict read as "tests ran green").
    _write(tmp_path / "scripts", "thing_test.py", "print('ok')\n")
    called = False

    def _boom(_policy: object) -> CommandResult:  # pragma: no cover - must not run
        nonlocal called
        called = True
        return CommandResult(argv=(), returncode=0, stdout="", stderr="", duration_s=0.0)

    monkeypatch.setattr(scriptcheck, "run_in_jail", _boom)
    outcome = scriptcheck.run_offline_tests(tmp_path, "hardened")
    assert outcome.problems == () and not called
    assert outcome.skipped == 1 and "hardened" in outcome.skip_reason
    assert capsys.readouterr().err == ""  # the caller owns the rendering


def test_offline_tests_no_test_files(tmp_path: Path) -> None:
    _write(tmp_path / "scripts", "real.py", "print('hi')\n")
    assert scriptcheck.run_offline_tests(tmp_path, "strict").problems == ()


def test_offline_tests_jail_unavailable_surfaces_diagnostic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from agent6.sandbox.jail import JailUnavailableError

    _write(tmp_path / "scripts", "thing_test.py", "print('ok')\n")

    def _raise(_policy: object) -> CommandResult:
        raise JailUnavailableError("no namespaces")

    monkeypatch.setattr(scriptcheck, "run_in_jail", _raise)
    problems = scriptcheck.run_offline_tests(tmp_path, "strict").problems
    assert len(problems) == 1
    assert "could not run offline tests" in problems[0]


def test_static_diagnostics_relativize_temp_paths(tmp_path: Path) -> None:
    # ruff diagnostics used to name the private temp copy; they now read as
    # bundle-relative paths, like the offline-test diagnostics.
    _need("ruff")
    _write(tmp_path / "scripts", "bad.py", "print(undefined_name)\n")
    problems = scriptcheck.lint_and_typecheck(tmp_path / "scripts")
    assert problems
    assert "agent6-scriptcheck-" not in problems[0]
    assert "scripts/bad.py" in problems[0]


def test_offline_tests_get_a_fresh_data_dir_per_test(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The docstring promise: state one test's script leaves in
    # $AGENT6_MACHINE_DATA_DIR must not leak into the next test.
    _write(tmp_path / "scripts", "a_test.py", "pass\n")
    _write(tmp_path / "scripts", "b_test.py", "pass\n")

    def _run(policy: JailPolicy) -> CommandResult:
        data = Path(policy.extra_rw_paths[0])
        marker = data / "marker"
        rc = 1 if marker.exists() else 0  # a leaked marker fails the later test
        marker.write_text("x", encoding="utf-8")
        return CommandResult(
            argv=policy.argv, returncode=rc, stdout="", stderr="leaked marker", duration_s=0.0
        )

    monkeypatch.setattr(scriptcheck, "run_in_jail", _run)
    assert scriptcheck.run_offline_tests(tmp_path, "strict").problems == ()
    assert not (tmp_path / ".scriptcheck_data").exists()  # still cleaned up after


def test_offline_tests_run_from_a_copy_outside_the_state_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The real bundle lives under the per-repo state dir, which the jail
    MASKS: tests run in place saw an empty tree ('python3: can't open file')
    or the launcher failed rootfs setup. The runner must hand the jail a
    private temp copy instead (caught by a live machine-create run; the old
    tmp_path fixtures covered the in-place path vacuously)."""
    monkeypatch.setenv("AGENT6_STATE_HOME", str(tmp_path / "statehome"))
    bundle = tmp_path / "statehome" / "repo-id" / "sessions" / "machines" / "draft"
    _write(bundle / "scripts", "thing_test.py", "print('ok')\n")
    seen_cwds: list[str] = []

    def run(policy: object) -> CommandResult:
        seen_cwds.append(str(getattr(policy, "cwd", "")))
        return CommandResult(
            argv=("python3", "scripts/thing_test.py"),
            returncode=0,
            stdout="",
            stderr="",
            duration_s=0.0,
        )

    monkeypatch.setattr(scriptcheck, "run_in_jail", run)
    assert scriptcheck.run_offline_tests(bundle, "strict").problems == ()
    assert seen_cwds and all("statehome" not in c for c in seen_cwds)


def test_fix_mode_applies_safe_fixes_and_writes_back(tmp_path: Path) -> None:
    """machine create validates its OWN generated bundle: a fixable-only
    problem (an unused import) must be fixed in place and not fail the
    attempt — a whole authoring round burned on it before. The default
    (operator-facing check/test) never mutates."""
    _need("ruff")
    body = "import json\nimport os\n\n\ndef f(x: int) -> str:\n    return json.dumps({'v': x})\n"
    _write(tmp_path / "scripts", "fixable.py", body)
    # Default: reported, file untouched.
    assert scriptcheck.lint_and_typecheck(tmp_path / "scripts")
    assert (tmp_path / "scripts" / "fixable.py").read_text() == body
    # Fix mode: repaired in place, nothing reported for the fixable part.
    problems = scriptcheck.lint_and_typecheck(tmp_path / "scripts", fix=True)
    fixed = (tmp_path / "scripts" / "fixable.py").read_text()
    assert "import os" not in fixed
    assert not any("ruff" in p for p in problems)


def test_the_create_lint_anchors_relative_patterns_where_machine_check_does(
    tmp_path: Path,
) -> None:
    """`--config` anchors a relative per-file-ignores glob to the cwd, so a
    `scripts/*` ignore in the repo's pyproject silenced the draft (linted from
    its workspace) and fired on the published bundle under `machine check`."""
    from agent6.app.machine._scriptcheck import (
        _resolve_tool,  # pyright: ignore[reportPrivateUsage]
        lint_and_typecheck,
    )

    if _resolve_tool("ruff") is None:
        pytest.skip("ruff is not installed")
    repo = tmp_path / "repo"
    (repo / "machines").mkdir(parents=True)
    (repo / "pyproject.toml").write_text(
        '[tool.ruff]\nline-length = 20\n[tool.ruff.lint]\nselect = ["E501"]\n'
        '[tool.ruff.lint.per-file-ignores]\n"scripts/*" = ["E501"]\n',
        encoding="utf-8",
    )
    workspace = tmp_path / "ws"
    (workspace / "scripts").mkdir(parents=True)
    (workspace / "scripts" / "helper.py").write_text(
        "VALUE = 'a line that is longer than twenty characters'\n", encoding="utf-8"
    )
    drafted = lint_and_typecheck(workspace / "scripts", ruff_config_from=repo / "machines")
    published = repo / "machines" / "m"
    shutil.copytree(workspace, published)
    checked = lint_and_typecheck(published / "scripts")
    assert any("E501" in p for p in checked), checked
    assert any("E501" in p for p in drafted), drafted
    assert not any(str(workspace) in p for p in drafted), drafted  # bundle-relative paths
