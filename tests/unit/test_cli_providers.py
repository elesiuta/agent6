# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Tests for agent6.app.providers provider construction (config -> provider)."""

from __future__ import annotations

from unittest.mock import MagicMock

from agent6.app.providers import (
    build_role_provider,
)
from agent6.config import Config, ModelsConfig, OpenAIProviderEntry, RoleModel
from agent6.providers import OpenAIProvider


def test_build_role_provider_forwards_extra_body_and_headers() -> None:
    # The config -> provider pass-through is a one-liner; pin it so dropping
    # `extra_body=...` (or extra_headers) can't silently stop reaching the wire
    # — that would make `provider` routing / caching config a no-op with no
    # failing test.
    cfg = Config(
        providers={
            "openrouter": OpenAIProviderEntry(
                api_format="openai",
                base_url="https://openrouter.ai/api/v1",
                extra_headers={"X-Title": "agent6"},
                extra_body={"provider": {"sort": "throughput"}},
            )
        },
        models=ModelsConfig(worker=RoleModel(provider="openrouter", model="kimi")),
    )
    prov = build_role_provider(cfg, "worker", transcript_sink=MagicMock(), budget=MagicMock())
    assert isinstance(prov, OpenAIProvider)
    assert prov.extra_body == {"provider": {"sort": "throughput"}}
    assert ("X-Title", "agent6") in prov.extra_headers


def test_reviewer_family_builders_stamp_their_own_seats() -> None:
    """The prompt reviser, summariser, and a bare-persona review seat are
    distinct actors sharing the reviewer ROUTE; each stamps its own seat on
    transcripts. All of them stamping "reviewer" left persisted transcripts
    unable to tell which actor made a call."""
    from unittest.mock import call

    from agent6.app.providers import (
        build_prompt_reviser_provider,
        build_review_seats,
        reviewer_seat_provider,
    )
    from agent6.config import PromptConfig, ReviewConfig

    cfg = Config(
        providers={"o": OpenAIProviderEntry(api_format="openai", base_url="https://x/v1")},
        models=ModelsConfig(
            worker=RoleModel(provider="o", model="m"),
            reviewer=RoleModel(provider="o", model="m"),
        ),
        review=ReviewConfig(trigger="on_verify_fail", seats=("security",)),
        prompt=PromptConfig(revise_prompt="auto"),
    )

    sink = MagicMock()
    build_prompt_reviser_provider(cfg, transcript_sink=sink, budget=MagicMock(), events=MagicMock())
    assert sink.for_seat.call_args == call("prompt_reviser")

    sink = MagicMock()
    reviewer_seat_provider(
        cfg, "summariser", transcript_sink=sink, budget=MagicMock(), events=MagicMock()
    )
    assert sink.for_seat.call_args == call("summariser")

    sink = MagicMock()
    build_review_seats(cfg, transcript_sink=sink, budget=MagicMock(), n=1)
    assert sink.for_seat.call_args == call("review:security")

    # The role builders keep stamping the role itself.
    sink = MagicMock()
    build_role_provider(cfg, "worker", transcript_sink=sink, budget=MagicMock())
    assert sink.for_seat.call_args == call("worker")
