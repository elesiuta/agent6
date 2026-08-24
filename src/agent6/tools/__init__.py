# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Tool dispatch + schemas exposed to the LLM. Import the submodule you need;
the package itself loads nothing, so a read-model that wants one helper does
not pay for the dispatcher."""

from __future__ import annotations
