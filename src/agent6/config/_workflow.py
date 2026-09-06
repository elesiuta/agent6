# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The loop-behaviour models: `[workflow]` (+ its metric), `[review]`,
`[context]`, `[prompt]`, and `[budget]`."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from agent6.config._base import MODEL_CONFIG, Argv, StrTuple


def parse_seat_spec(spec: str) -> tuple[str, str, str]:
    """A review seat, `persona[@provider/model]`, as `(persona, provider,
    model)`: `"security@openrouter/moonshotai/kimi-k2"` ->
    `("security", "openrouter", "moonshotai/kimi-k2")` (the model may itself
    contain `/`; only the first `/` after `@` splits provider from model),
    `"security"` -> `("security", "", "")` (routed via the reviewer role),
    `"@anthropic/claude-opus-4-8"` -> `("", "anthropic", "claude-opus-4-8")`.
    An `@` form must name BOTH a provider and a model, so a typo cannot
    degrade to the reviewer route in silence; it raises ValueError."""
    persona, sep, route = spec.partition("@")
    if not sep:
        return (spec.strip(), "", "")
    provider, slash, model = route.partition("/")
    if not (provider.strip() and slash and model.strip()):
        raise ValueError(
            f"{spec!r} must be 'persona@provider/model' (both provider and model required)"
        )
    return (persona.strip(), provider.strip(), model.strip())


# The review-seat depth (`[review].tier`); ReviewSeat.tier mirrors this, so the
# vocabulary has one owner.
ReviewTier = Literal["diff", "explore"]


class MetricConfig(BaseModel):
    """Optional continuous-score metric for tasks that have a measurable goal
    (cycles, wall time, kB, bench score) distinct from binary verify pass/fail.

    When configured, `run_metric_command` (the metric tool) runs `command`
    in the jail (same env as `verify_command`) and parses `pattern`'s
    first capture group as a number. `goal = "minimize"` for things like
    cycles/time; `"maximize"` for bench scores. `pattern` is a Python
    regex; the FIRST capture group must be a base-10 integer or float. If
    the pattern does not match in the command's combined stdout+stderr the
    metric is treated as missing.
    """

    model_config = MODEL_CONFIG

    command: Argv = Field(
        min_length=1,
        description=(
            "The command that prints the score, as argv (no shell). Runs after every "
            "verify-passing edit, and on the model's `run_metric_command` call."
        ),
    )
    pattern: str = Field(
        min_length=1,
        description=(
            "A regular expression over the command's output; its first capture group is the "
            'number, e.g. `"score: ([0-9.]+)"`.'
        ),
    )
    goal: Literal["minimize", "maximize"] = Field(
        description=(
            "Which way is better: `minimize` (a smaller number wins) or `maximize`. The run "
            "reports the trajectory and can finish once a verified edit only ties the best."
        ),
    )


class WorkflowConfig(BaseModel):
    model_config = MODEL_CONFIG

    # The command agent6 runs to decide whether a step "succeeded". This is
    # inherently repo-specific, so it has no useful global default and defaults
    # to empty. Optional: `agent6 run`/`plan` infer one per run when it is unset
    # (AGENTS.md -> repo signals -> a cheap LLM call; see agent6.verify_infer),
    # falling back to a gateless run. `agent6 init` can pin one.
    verify_command: Argv = Field(
        default=(),
        description=(
            "The command that decides whether a step succeeded, as argv (no shell; wrap a pipeline "
            'as `["sh", "-c", "a && b"]`). Set it to pin the gate. Unset: each run infers one and '
            "prints it (an AGENTS.md `## Verify command` block first, then a root `verify.sh`, "
            "the repo's manifest files, and loose `test_*.py` files, then a model call over "
            "those manifests); a run that can infer none starts gateless and adopts the first "
            "gate a recognizable project created mid-run yields."
        ),
    )
    # False pins gatelessness for a run with no verify_command: neither the
    # preflight inference nor the mid-run adoption arms a gate.
    verify_infer: bool = Field(
        default=True,
        description=(
            "Infer a verify command when `verify_command` is unset (AGENTS.md fence, repo "
            "signals, a model call), and adopt one mid-run when a gateless run materializes a "
            "recognizable project; an adopted gate that cannot run (exit 127, or the module "
            "its `-m` names is missing) is dropped again, never re-adopted. false: such a run "
            "stays gateless, no inference and no adoption; a set `verify_command` is unaffected."
        ),
    )
    # per-call timeout for verify_command (and metric_command) in
    # seconds. Defaults to the jail's general 600s but should be cranked
    # MUCH lower for benches where the verify is a fast correctness test
    # (perf-takehome's CorrectnessTests run in ~2s; a 30s cap detects
    # infinite-loop / quadratic edits 20x faster than the 600s default).
    # Setting too low for slow legitimate tests will cause false-positive
    # failures, so leave at 600 unless the verify is reliably fast.
    verify_timeout_s: float = Field(
        gt=0.0,
        default=600.0,
        description=(
            "Seconds one `verify_command` or `metric.command` call may take before it is killed "
            "and counted as failed. A pytest gate naming no paths that overruns this budget "
            "(the harness's run or the model's own `run_verify_command`) re-runs scoped to the "
            "test files nearest the run's diff, and harness gates run scoped until a full run "
            "of the gate passes; a scoped green ends the run `passed · scoped gate`. A "
            "model-chosen `run_command` is not bounded (see `command_checkin_s`)."
        ),
    )
    # Bounds one LEG: a resume gets a fresh allowance (numbering continues),
    # so a standing run is not capped by the sum of its legs.
    max_iterations: int = Field(
        default=200,
        description=(
            "Assistant turns one leg may take before the run stops with reason "
            "`max_iterations`; -1 is unlimited. A resumed leg gets a fresh allowance."
        ),
    )

    @field_validator("max_iterations")
    @classmethod
    def _iterations_unlimited_is_exactly_minus_one(cls, v: int) -> int:
        if v == 0 or v < -1:
            raise ValueError("max_iterations is >= 1, or exactly -1 for unlimited")
        return v

    # How long a run_command may run before the model is handed it back as a
    # background job. NOT a timeout: nothing is killed, the command keeps
    # running and the model decides whether to wait, poll or stop it -- a
    # judgement a number cannot make. 0 disables the hand-back (wait while it
    # lives), which is right when a human is watching and can interrupt.
    # 900 because the hand-back is non-destructive, so it can afford to be
    # patient: the cost of being early is a poll cycle of tokens, and the cost
    # of being late is nothing at all.
    command_checkin_s: float = Field(
        ge=0.0,
        default=900.0,
        description=(
            "Seconds a model's `run_command` may run before it is handed back as a background job. "
            "Not a timeout: nothing is killed, the command keeps running, and the model is told "
            "(`returncode: null`, `still_running: true`, a `background_id`) so it can wait with "
            "`read_background`, stop it, or carry on. `0` disables the hand-back."
        ),
    )
    standing_patience: int = Field(
        ge=-1,
        default=-1,
        description=(
            "Consecutive fruitless standing-goal re-entries (rounds with no executed tool call) "
            "the run absorbs before soft ends are honoured. `-1`: never on its own (the run ends "
            "on its budget, iteration cap, or an operator stop); `0`: the first fruitless round "
            "ends it; `N`: N fruitless re-entries get an escalating nudge, then ends are "
            "honoured. A round that lands work resets the streak."
        ),
    )
    # The harness-run gate. `finish` certifies the tree the run ends on;
    # `never` leaves every gate run to the model's own run_verify_command
    # calls (the measured model-driven shape); `step` is the expensive end.
    verify_when: Literal["finish", "step", "never"] = Field(
        default="finish",
        description=(
            "When the harness runs `verify_command` itself: `finish` (when the model calls "
            "`finish_session` and the tree changed since the last green run), `step` (also after "
            "every turn that edits the tree), `never` (only the model's own `run_verify_command` "
            "calls run it). The tool stays available in every mode; a run with no verify command "
            "has no gate to run."
        ),
    )
    verify_retries: int = Field(
        ge=0,
        default=2,
        description=(
            "How many times a red finish certification returns to the model with the gate's "
            "output before the finish stands and the run reads `finished · gate red`, never "
            "passed. `0`: the first red ends the run. A gate that was red before the run touched "
            "anything is not returned."
        ),
    )
    metric: MetricConfig | None = Field(
        default=None,
        description=(
            "An optional score to iterate on beside the pass/fail gate (a benchmark, a size, a "
            "count): the run calls it after every verify-passing edit and shows the model the "
            "trend. Unset: `run_metric_command` stays on the tool list and refuses, naming this "
            "key."
        ),
    )


class ContextConfig(BaseModel):
    """`[context]` section: tiered context-compaction thresholds."""

    model_config = MODEL_CONFIG

    # Tiered context-compaction thresholds (approximate chars; tokens ~=
    # chars/4). When cumulative *tool_result* content grows past
    # `drop_at_chars` the oldest tool_results are replaced by a
    # short placeholder (the worker can re-call the tool to refetch). When the
    # *whole* context (text + tool_use inputs + surviving tool_results) grows
    # past `summarise_at_chars` -- which must be > drop, so tier-2
    # escalates above tier-1 -- the conversation is summarized and restarted
    # (the durable task DAG survives; the restart notice points the worker at
    # `list_tasks` to recover task-level state).
    # `summary_max_tokens` caps the summarizer's output.
    #
    # Default `None` == ADAPTIVE: agent6 sizes both thresholds from the worker
    # model's context window (tier-1 at ~45% of it, tier-2 at the window
    # minus a 16k-token reserve), resolving
    # the window from a bundled table of tested models + the live model cache
    # (see `models.registry.compaction_thresholds`). Pin them by setting BOTH
    # explicitly (e.g. a self-hosted model agent6 can't size); leave BOTH unset
    # to stay adaptive. When the window is unknown, fixed 256k/768k
    # defaults apply.
    drop_at_chars: int | None = Field(
        default=None,
        gt=0,
        description=(
            "Tier-1 compaction threshold: once the accumulated tool results exceed this many "
            "characters (about 4 per token), the oldest results are replaced by short placeholders "
            "the model can re-fetch. Unset: sized from the model's context window (about 45% of "
            "it); set both thresholds to pin them."
        ),
    )
    summarise_at_chars: int | None = Field(
        default=None,
        gt=0,
        description=(
            "Tier-2 compaction threshold: once the whole context exceeds this many characters, the "
            "elided history is summarized and the conversation restarts on the summary (the task "
            "DAG survives). Unset: the model's window minus a 16k-token reserve. Must exceed "
            "`drop_at_chars`."
        ),
    )
    keep_recent_chars: int = Field(
        ge=0,
        default=80_000,
        description=(
            "How many characters of the most recent history a tier-2 restart keeps verbatim after "
            "the summary. `0` keeps none."
        ),
    )
    keep_thinking_turns: int = Field(
        ge=0,
        default=0,
        description=(
            "At tier-1 moments, drop the model's thinking from assistant turns older than this "
            "many turns. `0` keeps all thinking. Wires that re-send thinking (Anthropic's signed "
            "blocks, ChatGPT's reasoning items) replay less; the OpenAI wire never re-sends it."
        ),
    )
    summary_max_tokens: int = Field(
        gt=0,
        default=2048,
        description=(
            "Cap on the tokens a tier-2 summary (and a gist distillation) may produce. A "
            "reasoning model's per-call floor (room for its reasoning tokens) overrides a "
            "smaller cap, and the chatgpt backend takes no cap."
        ),
    )
    # Tier-1 gist elision: a large read_file result about to be elided decays
    # to a placeholder carrying a model-written gist of the file first (one
    # batched reviewer-model call per drop event), then to the bare marker
    # under continued pressure. Measured on the longhorizon bench: bare
    # elision of reference docs halves a retention task's score under a small
    # window. False = straight to bare markers (no distiller calls).
    elision_gists: bool = Field(
        default=True,
        description=(
            "At tier 1, replace a large `read_file` result with a model-written gist before the "
            "bare placeholder (the gist is dropped too under continued pressure, so the byte bound "
            "holds). `false`: straight to bare placeholders."
        ),
    )

    @model_validator(mode="after")
    def _check_compaction_thresholds(self) -> ContextConfig:
        drop, summarise = self.drop_at_chars, self.summarise_at_chars
        if (drop is None) != (summarise is None):
            raise ValueError(
                "set both context.drop_at_chars and"
                " summarise_at_chars, or NEITHER (neither == adaptive,"
                " sized from the worker model's context window). Both at once:"
                " agent6 config set context"
                " '{ drop_at_chars = 200000, summarise_at_chars = 400000 }'"
            )
        if drop is not None and summarise is not None and summarise <= drop:
            raise ValueError(
                "context.summarise_at_chars"
                f" ({summarise}) must be greater than"
                f" drop_at_chars ({drop}): tier-2"
                " summarise must escalate above tier-1 elision."
            )
        if summarise is not None and summarise <= self.keep_recent_chars:
            raise ValueError(
                f"context.summarise_at_chars ({summarise}) must be greater than"
                f" keep_recent_chars ({self.keep_recent_chars}): the verbatim tail"
                " alone would re-trigger tier 2 after every restart."
            )
        return self


class PromptConfig(BaseModel):
    """`[prompt]` section: system-prompt override, structural priors, and
    one-shot task-prompt revision."""

    model_config = MODEL_CONFIG

    # Advanced: replace run-mode's static base system prompt (role + edit/tool-use/
    # dag/scope rules) with the contents of this file. The dynamic blocks (verify,
    # metric, budget, repo-priors + AGENTS.md) still append, so repo context and
    # the budget cap are preserved. Empty = the built-in default. You own keeping
    # the tool contracts intact (apply_edit/apply_patch, run_verify_command,
    # finish_session); run startup warns if the override omits them. Inspect the
    # assembled result with `agent6 prompt show`.
    system_prompt_file: str = Field(
        default="",
        description=(
            "Path of a file that replaces run mode's built-in base system prompt (the dynamic "
            "blocks still append). The tool contracts become yours to state; a file missing the "
            "core tool names is warned about at startup. Empty: the built-in base. `agent6 prompt "
            "show` prints the assembled prompt, the tool definitions, and the first message."
        ),
    )
    # Include the structural-prior blocks in the run-mode <repo-priors>: hot
    # symbols (cross-file reference ranking), git co-change pairs, and the
    # tree-sitter symbol outline. Default on. Set false for a leaner/cheaper
    # prompt that relies purely on on-demand exploration (outline/find_definition)
    # -- the base repo map + AGENTS.md still ship.
    structural_priors: bool = Field(
        default=True,
        description=(
            "Include the `<repo-priors>` block in the system prompt: the repo map, the symbol "
            "outline, co-change and hot-symbol hints, recent commits. `false` for a leaner, "
            "cheaper prompt."
        ),
    )
    # one-shot task prompt revision before the worker loop starts.
    # Reuses the reviewer model, takes no tools, and is budget-tracked like
    # any other provider call. Default off: crisp prompts and frontier models
    # do not need revision.
    revise_prompt: Literal["off", "auto", "interactive"] = Field(
        default="off",
        description=(
            "Rewrite the task prompt once with the reviewer model before the loop starts: `off`, "
            "`auto` (the revision is used as written), or `interactive` (you accept, keep the "
            "original, or edit; needs the terminal, so a run under the TUI skips it)."
        ),
    )
    # Front-load task decomposition (run mode). When on the worker's system
    # prompt swaps the "DAG is optional" guidance for a "decompose first"
    # directive: lay the task out as ordered subtasks before editing, then work
    # one focused subtask at a time (the existing surface-current-task and
    # finish-gate machinery walks the frontier). Helps small/open models that
    # lose track of multi-part tasks; a capable model decomposes implicitly and
    # only pays the 2-4x turn overhead. "auto" (default) enables it ONLY for
    # worker models with a measured win in the capability registry
    # (models.registry.decompose_default); the CLI pins auto to on/off at run
    # start via `with_decompose`, and the engine treats any value other than
    # "on" as off. No effect on plan/ask/machine/agent modes. See
    # docs/config.md for the measured per-model effect.
    decompose: Literal["auto", "on", "off"] = Field(
        default="auto",
        description=(
            "Front-load task decomposition in run mode: the model lays the task out as ordered DAG "
            "subtasks before editing and works them one at a time. `on` helps small models that "
            "under-finish multi-part tasks (measured on mistral-small; capable models just pay "
            "2-4x overhead), `off` never, `auto` decides per worker model from the capability "
            "registry (`config show` prints the resolved value). `--decompose` forces it for one "
            "run."
        ),
    )

    @model_validator(mode="after")
    def _check_system_prompt_file(self) -> PromptConfig:
        # Fail loud at config time if the override path is set but missing, rather
        # than silently falling back to the default prompt at run start.
        if self.system_prompt_file:
            p = Path(self.system_prompt_file).expanduser()
            if not p.is_file():
                raise ValueError(f"prompt.system_prompt_file: not a readable file: {p}")
        return self


class ReviewConfig(BaseModel):
    """`[review]` section: the in-loop review panel and its trigger."""

    model_config = MODEL_CONFIG

    # When != "off", Workflow runs the review panel at the chosen trigger and
    # injects its findings as a user message the worker sees next turn. With no
    # `seats`, the panel is one seat on `[models.reviewer]` (same route
    # `agent6 review` uses).
    #   off              - never (default).
    #   on_verify_fail   - after every verify failure.
    #   before_finish    - intercept `finish_session`; a gating `decision`
    #                      rejects the finish while the panel is unsatisfied.
    #   periodic         - every `period` iterations.
    trigger: Literal["off", "on_verify_fail", "before_finish", "periodic"] = Field(
        default="off",
        description=(
            "When the in-loop review panel runs on the diff so far and its findings reach the "
            "model as a message: `off` (never), `on_verify_fail` (after each failed verify), "
            "`before_finish` (when the model calls `finish_session`; a gating `decision` can "
            "reject the finish), or `periodic` (every `period` iterations). With no `seats` the "
            "panel is one reviewer seat on `[models.reviewer]`, the model `agent6 review` uses."
        ),
    )
    period: int = Field(
        ge=1,
        default=10,
        description='Iterations between panels when `trigger = "periodic"`.',
    )
    # `seats` is THE roster: flat
    # "persona[@provider/model]" strings (e.g. "security" routes via
    # [models.reviewer]; "security@openrouter/moonshotai/kimi-k2" pins a
    # model). The `agent6 review --reviewers N`/`--personas` flags synthesize
    # an in-memory equivalent. `decision` is only a GATE in-loop; "advisory"
    # (default) just injects findings as guidance and never blocks.
    decision: Literal["advisory", "veto", "quorum", "all"] = Field(
        default="advisory",
        description=(
            "What a panel's BLOCK verdicts do: `advisory` (the findings are injected as guidance, "
            "nothing is blocked), `veto` (one blocking seat rejects the finish), `quorum` "
            "(`quorum` distinct models must block), or `all` (every seat must block). A gate "
            "applies to `before_finish` only; the other triggers always advise."
        ),
    )
    quorum: int = Field(
        ge=1,
        default=2,
        description=(
            'How many seats must block for `decision = "quorum"`, counted per distinct model (two '
            "seats on one model count once, so a same-model panel cannot reach it)."
        ),
    )
    # Per-run cap on total panel blocks before the gate auto-downgrades to
    # advisory for the rest of the run (so a gating panel can never stall forever).
    max_total_rejections: int = Field(
        ge=1,
        default=4,
        description=(
            "How many finishes a gating panel may reject per run before it disarms to `advisory` "
            "for the rest of the run, so a panel can never stall a run forever."
        ),
    )
    # Budget floor: the in-loop review panel is SKIPPED (approve-and-proceed) once
    # the run's remaining token budget falls below this fraction -- reviewing costs
    # most exactly when budget is scarcest. Default 0.25 = skip the panel in the
    # last quarter of the budget.
    budget_fraction: float = Field(
        gt=0.0,
        le=1.0,
        default=0.25,
        description=(
            "Skip the panel (the finish is accepted) once the run's remaining budget falls below "
            "this fraction of the whole. `0.25`: no panel in the last quarter."
        ),
    )
    seats: StrTuple = Field(
        default=(),
        description=(
            'The panel roster, one entry per seat: a persona name (`"security"`), routed via '
            '`[models.reviewer]`, or `"<persona>@<provider>/<model>"` to pin a model per seat '
            '(`"correctness@openrouter/moonshotai/kimi-k2"`). A persona is any short stance the '
            "reviewer's prompt adopts; the built-in set cycled when none is named is `security`, "
            "`correctness`, `tests`, `over-engineering`, `edge-cases`. Empty: one reviewer seat "
            "when `trigger` is on. `agent6 review --reviewers N --personas ...` builds the same "
            "roster for a one-off review."
        ),
    )
    # Seat concurrency for the in-loop panel (1 = sequential). The post-hoc
    # `agent6 review` runs all seats in parallel regardless (fast one-shot).
    concurrency: int = Field(
        ge=1,
        default=1,
        description=(
            "How many seats the in-loop panel runs at once (`1` = one after another; the panel's "
            "latency is its slowest seat). `agent6 review` always runs every seat in parallel."
        ),
    )
    # Reviewer tier: "diff" (one grounded call over the diff) or "explore" (a
    # read-only tool-using mini-loop that reads the broader repo first to catch
    # cross-file impact). explore is more thorough but costs several calls/seat.
    tier: ReviewTier = Field(
        default="diff",
        description=(
            "How much a seat reads: `diff` (one call over the diff, the task, and the verify "
            "result) or `explore` (a read-only tool-using reviewer that also reads the repo around "
            "the diff to catch cross-file impact; several calls per seat)."
        ),
    )

    @model_validator(mode="after")
    def _check_review_seats(self) -> ReviewConfig:
        # Each seats entry is "persona", "persona@provider/model", or
        # "@provider/model"; an "@" form must name BOTH a provider and a model so
        # a typo doesn't silently degrade to the reviewer route.
        for spec in self.seats:
            if not spec.strip():
                raise ValueError("review.seats entries must be non-empty")
            try:
                parse_seat_spec(spec)
            except ValueError as exc:
                raise ValueError(f"review.seats: {exc}") from exc
        return self

    @model_validator(mode="after")
    def _check_review_quorum(self) -> ReviewConfig:
        if self.decision == "quorum" and self.quorum > 1:
            models = {f"{p}/{m}" if p else "" for _, p, m in map(parse_seat_spec, self.seats)}
            if len(models) < self.quorum:
                raise ValueError(
                    f"review.decision='quorum' with quorum={self.quorum}"
                    f" needs >= {self.quorum} DISTINCT models (the gate counts one block per"
                    " distinct model). Provide them via seats"
                    " ('persona@provider/model'), or use decision='veto'."
                )
        return self


class BudgetConfig(BaseModel):
    """`[budget]`: every provider call is bounded in exactly ONE currency.

    A call the runtime can meter (provider-reported cost, else price x tokens
    at the model's fetched rates, cache-aware) counts against `max_usd`; a
    subscription call carrying a plan-usage reading counts consumed
    percentage points against `max_percent`; a call with neither counts its
    input+output tokens against `max_tokens_fallback`. The fields share one
    rule: `-1` = unlimited,
    `0` = refuse calls in that ledger up front (`max_tokens_fallback = 0`
    means never run an unmeterable model), `> 0` = the cap. Hitting a cap
    ends the run resumably (`budget_exhausted`); each resumed leg gets a
    fresh budget. The `--max-usd` / `--max-tokens-fallback` flags override
    per run."""

    model_config = MODEL_CONFIG

    max_usd: float = Field(
        default=10.0,
        description=(
            "Cap on the metered spend of one run (provider-reported cost, else price times tokens "
            "at the model's fetched rates, cache-aware). Hitting it ends the run resumably "
            "(`budget_exhausted`); each resumed leg gets a fresh budget. `-1`: unlimited; `0`: "
            "refuse every metered call. `--max-usd` overrides per run."
        ),
    )
    max_tokens_fallback: int = Field(
        ge=-1,
        default=2_000_000,
        description=(
            "Token cap (input plus output) for the calls the run cannot price: local models, a "
            "model with no price data. `-1`: unlimited; `0`: never run an unmeterable model. "
            "`--max-tokens-fallback` overrides per run."
        ),
    )

    max_percent: float = Field(
        default=-1.0,  # the float the loader validates it to, so `config fill` is idempotent
        description=(
            "Cap on the plan percentage points one run may consume on a subscription provider "
            "(the rise in the account's reported used-percent across the run, accumulated across "
            "window resets, so values above 100 are meaningful; with several windows, the one "
            "that moved most). The reading is account-global: "
            "a concurrent run's spend counts toward whichever run observes it next. `-1`: "
            "unlimited; `0`: refuse plan-metered calls. `--max-percent` overrides per run."
        ),
    )

    # Purchased Codex credits and Claude extra usage are real money after the
    # included window; a plan-metered call that would draw on them refuses
    # unless this is set.
    allow_paid_credits: bool = Field(
        default=False,
        description=(
            "Allow plan-metered calls (`chatgpt`, `claude_code`) to spend PURCHASED credits or "
            "extra usage once the included plan window is exhausted (auto top-up can buy more "
            "with the saved payment method). `false` is a circuit breaker, not a guarantee: "
            "the backend's usage readings (a chatgpt preflight and every response's headers, "
            "every claude_code round's rate-limit event) report the account's windows and "
            "credit state, and once a window is exhausted with credits present the run stops "
            "at its next boundary; a call already in flight completes. `true`: a chatgpt credit "
            "balance's drop across the run is read as dollars and meters against `max_usd`; a "
            "claude_code run reads no credit balance, so the extra usage it spends is not "
            "metered by `max_usd`. Included-plan usage is unaffected."
        ),
    )

    @field_validator("max_usd", "max_percent")
    @classmethod
    def _usd_unlimited_is_exactly_minus_one(cls, v: float) -> float:
        # Non-finite never binds (nan fails every comparison; inf exceeds any
        # spend), which would silently disable the hard budget.
        if not math.isfinite(v) or (v < 0 and v != -1):
            raise ValueError("a budget cap is finite and >= 0, or exactly -1 for unlimited")
        return v
