# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Task-graph package.

The graph is the persistent representation of a single `agent6 run`. It lives in
the run dir (under the per-repo state dir). The curator owns the task GRAPH files
(graph.jsonl, graph/*.md, cursor.json); the worker, the planner and the
operator's own steering all mutate through the single in-process
`GraphCurator`. The main agent process writes the resume snapshot
(loop_state.json), the event log (logs.jsonl), and transcripts in-process too.
The run dir stays safe from jailed commands because it lives OUT of the
workspace: jailed commands run on the repo cwd and the state dir is outside it,
so safety comes from the out-of-tree location, not from a single-writer
invariant. The run-level `worker.lock` flock nonetheless keeps this the sole
writer; the curator's own per-mutation flock guards against a concurrent
operator-CLI read/write of the same files.
"""

from __future__ import annotations
