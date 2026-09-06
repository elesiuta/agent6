# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Public face of the review surfaces.

Re-exports the freeform review call (`code_review`, driving `agent6 review`)
and the adversarial review panel (`ReviewContext`, `render_findings`,
`run_panel`, `ReviewSeat`) from their private `_review`/`_panel`
siblings, so `ui/cli` imports both from one workflow-layer module instead of
reaching into privates.
"""

from __future__ import annotations

from agent6.workflows._panel import (
    ReviewContext,
    inconclusive_note,
    panel_is_inconclusive,
    render_findings,
)
from agent6.workflows._review import ReviewSeat, run_panel
from agent6.workflows.code_review import CodeReviewError, code_review

__all__ = [
    "CodeReviewError",
    "ReviewContext",
    "ReviewSeat",
    "code_review",
    "inconclusive_note",
    "panel_is_inconclusive",
    "render_findings",
    "run_panel",
]
