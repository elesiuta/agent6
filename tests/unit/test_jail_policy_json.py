# SPDX-License-Identifier: Apache-2.0
"""Contract test for the policy JSON the Python side sends the launcher.

The Rust launcher serde-defaults a MISSING ``memory_limit_mb`` (so an older
caller stays bounded), which means a Python-side regression that stops sending
the field would not fail loudly there. Pin the wire contract here: the field is
always present and carries the policy value, including the 0 opt-out.
"""

from __future__ import annotations

from pathlib import Path

from agent6.sandbox.jail import _policy_spec  # pyright: ignore[reportPrivateUsage]
from agent6.types import JailPolicy


def _fields(policy: JailPolicy) -> dict[str, object]:
    return _policy_spec(policy)


def test_policy_json_carries_the_uncapped_default_memory_limit(tmp_path: Path) -> None:
    """0 = off, matching [sandbox].memory_limit_mb: a cap is an operational
    guardrail the operator opts into, not a boundary (the kernel already
    handles a memory bomb), and the two sides must agree on the default so a
    policy that omits the field means the same thing in Rust."""
    fields = _fields(JailPolicy(cwd=tmp_path, argv=("/usr/bin/true",)))
    assert fields["memory_limit_mb"] == 0


def test_policy_json_carries_explicit_and_zero_memory_limit(tmp_path: Path) -> None:
    assert (
        _fields(JailPolicy(cwd=tmp_path, argv=("/usr/bin/true",), memory_limit_mb=512))[
            "memory_limit_mb"
        ]
        == 512
    )
    assert (
        _fields(JailPolicy(cwd=tmp_path, argv=("/usr/bin/true",), memory_limit_mb=0))[
            "memory_limit_mb"
        ]
        == 0
    )
