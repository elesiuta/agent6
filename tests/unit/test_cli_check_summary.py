# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""`agent6 check` summary keeps advisory statuses distinct from PASS."""

from __future__ import annotations

from pathlib import Path

import pytest

from agent6.config import Config
from agent6.ui.cli.check_cmds import _doctor_check_config  # pyright: ignore[reportPrivateUsage]


def test_no_providers_is_info_not_pass(capsys: pytest.CaptureFixture[str]) -> None:
    # A fresh setup (zero providers) is unusable until `agent6 connect`; the
    # check must not render that instruction as a PASS.
    checks = _doctor_check_config(Config())
    by_name = {c.name: c for c in checks}
    assert by_name["config.provider_keys"].status == "INFO"
    assert "agent6 connect" in by_name["config.provider_keys"].detail
    assert by_name["config.git_policy"].status == "PASS"
    # Sections state facts; the one summary states every verdict.
    assert "[INFO]" not in capsys.readouterr().out


def test_check_summary_carries_info_through(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # `check verify` on a default config: verify_command is unset, an advisory.
    # The summary line must say INFO (previously coerced to PASS) and exit 0.
    from agent6.ui.cli import main

    monkeypatch.chdir(tmp_path)
    rc = main(["check", "verify"])
    assert rc == 0
    out = capsys.readouterr().out
    summary = out.split("== summary ==", 1)[1]
    assert "[INFO] verify.argv" in summary
    assert "[PASS]" not in summary
    assert "—" not in out


def test_check_verify_says_what_this_repo_infers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """With verify_command unset, `check verify` names the command a run here
    would infer (the deterministic tiers) rather than "inferred per run", and
    says when there is nothing to infer from."""
    from agent6.ui.cli import main

    monkeypatch.chdir(tmp_path)
    assert main(["check", "verify"]) == 0
    assert "unset; nothing here to infer from" in capsys.readouterr().out
    (tmp_path / "verify.sh").write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    (tmp_path / "verify.sh").chmod(0o755)
    assert main(["check", "verify"]) == 0
    assert "unset; a run here infers ./verify.sh (from verify.sh)" in capsys.readouterr().out
