# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""A launcher binary the kernel will not execute is a refusal, not a crash."""

from __future__ import annotations

import errno
import os
import re
from pathlib import Path

import pytest

from agent6.sandbox import jail
from agent6.sandbox.jail import JailUnavailableError, run_in_jail
from agent6.types import JailPolicy


def test_an_unusable_launcher_binary_is_refused_with_the_remedy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The launcher's Popen raised its bare OSError (Exec format error for a
    build of another architecture, Permission denied for a missing exec bit),
    so `agent6 check`, the run preflight and every command tool crashed with a
    traceback instead of naming the binary and how to replace it."""
    fake = tmp_path / "agent6-jail"
    fake.write_text("#!/bin/sh\n", encoding="utf-8")
    fake.chmod(0o644)
    monkeypatch.setenv("AGENT6_JAIL_BIN", str(fake))
    policy = JailPolicy(cwd=tmp_path, argv=("/bin/true",), isolation="strict", timeout_s=5.0)
    with pytest.raises(JailUnavailableError, match=re.escape(str(fake))) as info:
        run_in_jail(policy)
    said = str(info.value)
    assert "Permission denied" in said
    assert "uv sync --reinstall-package agent6" in said and "AGENT6_JAIL_BIN" in said


def test_a_fork_or_descriptor_failure_is_not_blamed_on_the_binary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every OSError of the spawn read "cannot be executed ... reinstall":
    EAGAIN (fork), EMFILE (pipes) and ENOMEM got a remedy for a binary that
    was fine, and through the cached strict probe a transient fork failure at
    startup resolved `auto` to hardened for the whole run. Only ENOEXEC and
    EACCES speak about the binary; the rest pass through unchanged."""
    fake = tmp_path / "agent6-jail"
    fake.write_text("#!/bin/sh\n", encoding="utf-8")
    fake.chmod(0o755)
    monkeypatch.setenv("AGENT6_JAIL_BIN", str(fake))

    def _fork_fails(*_a: object, **_k: object) -> object:
        raise OSError(errno.EAGAIN, os.strerror(errno.EAGAIN))

    monkeypatch.setattr(jail.subprocess, "Popen", _fork_fails)
    policy = JailPolicy(cwd=tmp_path, argv=("/bin/true",), isolation="strict", timeout_s=5.0)
    with pytest.raises(OSError) as info:
        run_in_jail(policy)
    assert not isinstance(info.value, JailUnavailableError)
    assert info.value.errno == errno.EAGAIN
