# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Shared pytest fixtures."""

from __future__ import annotations

import os

import pytest
import tree_sitter_language_pack
from tree_sitter_language_pack import PackConfig

from tests.jail_env import require_userns_jail


def pytest_runtest_setup(item: pytest.Item) -> None:
    """Make the `needs_namespaces` marker real: skip (with the host's actual
    reason) where the userns jail cannot run, instead of hard-failing with
    JailUnavailableError. The marker alone registered a policy nothing
    enforced."""
    if item.get_closest_marker("needs_namespaces") is not None:
        require_userns_jail()


@pytest.fixture(autouse=True)
def _hermetic_git(  # pyright: ignore[reportUnusedFunction]
    monkeypatch: pytest.MonkeyPatch, tmp_path_factory: pytest.TempPathFactory
) -> None:
    """Pin a suite-owned git identity; blank the system/global git config.

    Tests commit in throwaway repos and CLONES (a clone does not inherit the
    origin's repo-local user.name/email). The developer's ~/.gitconfig silently
    supplied the identity locally while a bare CI runner has none, so the suite
    was green here and red in CI. One suite-owned global config makes the two
    environments identical. A test that needs a MISSING identity overrides
    GIT_CONFIG_GLOBAL itself (see test_verify_git_identity_missing_raises).
    """
    cfg = tmp_path_factory.mktemp("git-identity") / "gitconfig"
    cfg.write_text("[user]\n\tname = t\n\temail = t@t\n", encoding="utf-8")
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(cfg))
    monkeypatch.setenv("GIT_CONFIG_SYSTEM", os.devnull)


# Resolved once, with the operator's own environment: the grammars a
# developer or a CI runner has already downloaded.
_GRAMMAR_CACHE = tree_sitter_language_pack.cache_dir()


@pytest.fixture(autouse=True)
def _isolate_state(  # pyright: ignore[reportUnusedFunction]
    monkeypatch: pytest.MonkeyPatch, tmp_path_factory: pytest.TempPathFactory
) -> None:
    """Point agent6's per-repo state base + global config at throwaway dirs.

    Run state + the per-repo config live out of the workspace under the state
    base (``$XDG_STATE_HOME/agent6``). Isolating that base keeps tests off the
    real ``~/.local/state``; the global config dir (``$XDG_CONFIG_HOME/agent6``)
    is pointed at an empty dir so no operator config or secret reaches a test.
    The cache and data homes are isolated for the same reason
    as the git config above: the developer's model-price cache made USD
    assertions pass locally while a bare CI runner has none, and the
    developer's installed skills would index into any run a test starts. A
    test that needs a price or a skill seeds its own dir. A test may still
    override any of these itself (its body runs after this fixture).
    """
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path_factory.mktemp("agent6-state")))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path_factory.mktemp("agent6-config")))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path_factory.mktemp("agent6-cache")))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path_factory.mktemp("agent6-data")))
    # The tree-sitter language pack keeps its downloaded grammars under the
    # same XDG cache; an empty one per test would re-download every grammar.
    tree_sitter_language_pack.configure(PackConfig(cache_dir=_GRAMMAR_CACHE))
