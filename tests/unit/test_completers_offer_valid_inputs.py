# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""A completer offers exactly what its argument accepts.

Offering less is a lie by omission: the operator tabs, sees no plan or ask, and
concludes the verb does not take one -- when it does. Offering more is worse,
since the suggestion is refused on Enter.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent6.config.layer import resolved_state_dir
from agent6.sessions.layout import bucket_dir
from agent6.ui.cli import completers


def _seed(tmp_path: Path) -> None:
    state = resolved_state_dir(tmp_path)
    for bucket, mode, sid in (
        ("runs", "run", "runny-one-AAAAAA"),
        ("plans", "plan", "planny-two-BBBBB"),
        ("asks", "ask", "asky-three-CCCCC"),
        ("machines", "machine", "drafty-four-DDDD"),
    ):
        session = bucket_dir(state, bucket) / sid
        session.mkdir(parents=True)
        (session / "logs.jsonl").write_text(
            json.dumps({"type": "session.start", "mode": mode}) + "\n", encoding="utf-8"
        )
    (state / "machines" / "live-machine").mkdir(parents=True)


def test_every_session_id_is_offered_where_any_is_accepted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`sessions show|diff|transcript|...` resolve across every bucket."""
    monkeypatch.chdir(tmp_path)
    _seed(tmp_path)
    offered = set(completers._complete_session_ids(""))  # pyright: ignore[reportPrivateUsage]
    assert offered == {
        "runny-one-AAAAAA",
        "planny-two-BBBBB",
        "asky-three-CCCCC",
        "drafty-four-DDDD",
    }


def test_resume_offers_only_what_it_can_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A machine draft is a session, but `resume` refuses it -- so suggesting it
    would be a suggestion the operator cannot act on."""
    monkeypatch.chdir(tmp_path)
    _seed(tmp_path)
    offered = set(completers._complete_resumable_ids(""))  # pyright: ignore[reportPrivateUsage]
    assert offered == {"runny-one-AAAAAA", "planny-two-BBBBB", "asky-three-CCCCC"}


def test_attach_offers_every_session_and_every_machine(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    _seed(tmp_path)
    offered = set(completers._complete_watch_targets(""))  # pyright: ignore[reportPrivateUsage]
    assert "live-machine" in offered
    assert {"runny-one-AAAAAA", "planny-two-BBBBB", "asky-three-CCCCC"} <= offered


def test_enum_value_completion_is_derived_from_the_schema(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A hand-kept enum table drifts: leaves whose type is a Literal got
    nothing on TAB because nobody added them. The choices come from the schema
    the config view already reads, so a new enum leaf completes for free."""
    import argparse

    from agent6.config import Config
    from agent6.config.layer import load_effective
    from agent6.viewmodel.config_view import build_config_view

    monkeypatch.chdir(tmp_path)
    schema_enums = {
        s.key
        for s in build_config_view(load_effective(tmp_path)).settings
        if s.choices and s.py_type == "choice"
    }
    assert len(schema_enums) > 10, "expected many enum leaves in the schema"
    for key in sorted(schema_enums):
        offered = completers._complete_config_values(  # pyright: ignore[reportPrivateUsage]
            "", argparse.Namespace(key=key)
        )
        assert offered, f"{key} offers no values on TAB"

    # A bool is as closed a set as any enum, and `config set` takes exactly
    # `true` or `false` there: the 17 bool leaves completed to nothing while
    # every enum completed, and `True` and `yes` are both refused.
    bools = {
        s.key for s in build_config_view(load_effective(tmp_path)).settings if s.py_type == "bool"
    }
    assert len(bools) > 10, "expected many bool leaves in the schema"
    for key in sorted(bools):
        offered = completers._complete_config_values(  # pyright: ignore[reportPrivateUsage]
            "", argparse.Namespace(key=key)
        )
        assert set(offered) == {"true", "false"}, f"{key} offers {offered}"

    # sandbox.isolation keeps its deliberate omission: TAB must not put
    # "disable the sandbox" one keystroke away.
    iso = completers._complete_config_values(  # pyright: ignore[reportPrivateUsage]
        "", argparse.Namespace(key="sandbox.isolation")
    )
    assert "none" not in iso and {"auto", "strict", "hardened"} <= set(iso)
    assert Config()  # the schema loaded
