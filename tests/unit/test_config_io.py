# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Config write surgery is crash-safe: writers publish through atomic_write
(tmp + rename), never truncating the live file in place."""

from __future__ import annotations

from pathlib import Path

import pytest

from agent6.config import io


def test_writers_go_through_atomic_write_and_never_truncate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = tmp_path / "config.toml"
    original = "[sandbox]\nprotect_git = true\n"
    cfg.write_text(original, encoding="utf-8")

    def boom(_path: Path, _text: str) -> None:
        raise RuntimeError("simulated crash during publish")

    # If a writer still called path.write_text, it would truncate cfg before any
    # rename and this patch would never fire; going through atomic_write means
    # the failure happens before the rename and the live file is untouched.
    monkeypatch.setattr(io, "atomic_write", boom)
    with pytest.raises(RuntimeError):
        io.upsert_toml_leaf(cfg, "sandbox.protect_git", False)
    assert cfg.read_text(encoding="utf-8") == original  # not truncated


def test_write_leaves_no_temp_siblings(tmp_path: Path) -> None:
    cfg = tmp_path / "config.toml"
    io.upsert_toml_leaf(cfg, "sandbox.network", "auto")
    io.upsert_toml_leaf(cfg, "sandbox.protect_git", False)
    assert 'network = "auto"' in cfg.read_text(encoding="utf-8")
    assert [p.name for p in tmp_path.iterdir()] == ["config.toml"]  # tmp files cleaned up


def test_a_quoted_leaf_key_is_the_same_leaf(tmp_path: Path) -> None:
    """`"protect_git" = true` is valid TOML naming the same leaf. Unmatched,
    the surgery appended a duplicate key, the write rolled back, and the value
    became unsettable from every surface with a message blaming the file."""
    path = tmp_path / "config.toml"
    path.write_text('[sandbox]\n"protect_git" = true\nhome = "tmp"\n', encoding="utf-8")

    io.upsert_toml_leaf(path, "sandbox.protect_git", False)

    assert io.read_toml_file(path) == {"sandbox": {"protect_git": False, "home": "tmp"}}
    assert path.read_text(encoding="utf-8").count("protect_git") == 1
