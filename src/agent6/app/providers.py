# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Construct role/reviser/summariser/review-seat providers for CLI commands."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from agent6.budget import BudgetTracker
from agent6.config import (
    AnthropicProviderEntry,
    ChatGPTProviderEntry,
    ClaudeCodeProviderEntry,
    Config,
    EffortLevel,
    RoleModel,
    RoleName,
    parse_seat_spec,
)
from agent6.events import EventSink
from agent6.models import registry as models_registry
from agent6.providers import (
    AnthropicProvider,
    ChatGPTCredential,
    ChatGPTProvider,
    ClaudeCodeProvider,
    CommandToken,
    OpenAIProvider,
    Provider,
    ProviderError,
    ProviderResponse,
    ToolDefinition,
    TranscriptRecorder,
    TranscriptSink,
)
from agent6.secrets import resolve_api_key
from agent6.workflows.review import ReviewSeat


def resolve_compaction_thresholds(
    cfg: Config, rm: RoleModel | None, *, log: Callable[[str], None] | None = None
) -> tuple[int, int, int]:
    """Effective `(drop_at_chars, summarise_at_chars, keep_recent_chars)` for the
    model *rm* drives the loop with: the explicit config values if set, else
    sized from the model's context window (bundled table + live model cache),
    else the historical fixed defaults. Logs the choice when adaptive so the
    operator can see what was picked. `rm is None` (model unresolved) falls
    through to explicit-or-fixed-default.

    An adaptive tier-2 threshold clamps the verbatim tail to half of itself:
    the config validator refuses an explicit pair whose tail is at or above the
    threshold, and a small window (a 32k model against the 80,000-char default)
    sizes exactly that; a restart under it would keep no verbatim turn at all."""
    drop_override = cfg.context.drop_at_chars
    summarise_override = cfg.context.summarise_at_chars
    provider = rm.provider if rm is not None else ""
    model = rm.model if rm is not None else ""
    drop, summarise = models_registry.compaction_thresholds(
        provider,
        model,
        drop_override=drop_override,
        summarise_override=summarise_override,
    )
    if log is not None and drop_override is None:
        ctx = models_registry.context_window(provider, model) if model else None
        src = (
            f"adaptive from {model} (context {ctx:,} tok)"
            if ctx
            else "fixed default (context window unknown)"
        )
        # These are the thresholds compaction WILL fire at, not a compaction
        # that happened; say "at" or a fresh run reads as a 1.3M-char event.
        log(f"compaction thresholds: drop at {drop:,} chars, summarise at {summarise:,} [{src}]")
    keep = cfg.context.keep_recent_chars
    if summarise_override is None and keep > summarise // 2:
        keep = summarise // 2
        if log is not None:
            log(
                f"context.keep_recent_chars {cfg.context.keep_recent_chars:,} ->"
                f" {keep:,} chars: half this model's tier-2 threshold, so a restart"
                " shrinks the context instead of re-triggering on its own tail"
            )
    return drop, summarise, keep


def resolve_decompose(
    cfg: Config, rm: RoleModel | None, *, log: Callable[[str], None] | None = None
) -> Config:
    """Pin `prompt.decompose = "auto"` to on/off for this run.

    On only when the worker model *rm* has a measured decompose win in the
    capability registry; explicit on/off (config or `--decompose`) passes
    through untouched. The engine treats any value other than "on" as off,
    so this resolution is what makes "auto" real."""
    if cfg.prompt.decompose != "auto":
        return cfg
    on = rm is not None and models_registry.decompose_default(rm.model)
    if on and log is not None and rm is not None:
        log(f"decompose: auto-enabled for {rm.model} (measured win, bench/coreagent)")
    return cfg.with_decompose("on" if on else "off")


def build_role_provider(
    cfg: Config,
    role: RoleName,
    *,
    transcript_sink: TranscriptSink,
    budget: BudgetTracker,
    model_override: str = "",
    seat: str = "",
) -> Provider:
    """Construct the configured provider for `role`. *seat* is the transcript
    seat stamp when the caller is a distinct actor sharing the role's route
    (a review seat, the summariser, ...); default = the role itself.

    Resolves the API key via `agent6.secrets.resolve_api_key` (env var named
    by `api_key_env` first, then `secrets.toml`). `model_override` (if
    truthy) replaces the model string; provider routing is unchanged. The
    role's `effort` level is wired to the provider's default reasoning
    effort. Callers should have validated routing via
    `cfg.require_runnable(role)` first.
    """
    rm = cfg.models.resolve(role)
    if rm is None:  # pragma: no cover - blocked by require_runnable
        raise ProviderError(f"no model configured for role {role!r}")
    model = model_override or rm.model
    entry = cfg.providers.get(rm.provider)
    if entry is None:  # pragma: no cover - blocked by config validation
        raise ProviderError(
            f"models.{role}.provider = {rm.provider!r} but [providers.{rm.provider}] missing"
        )
    return _provider_from_entry(
        rm.provider,
        entry,
        model,
        rm.effort,
        # Stamp the seat on this provider's transcripts: the conversation fold
        # keeps the worker's round-trips and skips compaction's side-calls,
        # whose one-message requests otherwise read as a restart.
        transcript_sink=transcript_sink.for_seat(seat or role),
        budget=budget,
    )


def _provider_from_entry(
    provider_name: str,
    entry: Any,
    model: str,
    effort: EffortLevel | None,
    *,
    transcript_sink: TranscriptRecorder,
    budget: BudgetTracker,
) -> Provider:
    """Build a Provider for an explicit `[providers.<provider_name>]` entry +
    model + effort. Shared by `build_role_provider` (role routing) and the
    review panel's explicit per-seat `provider/model` routing."""
    budget.note_route(model, provider_name)
    if isinstance(entry, ClaudeCodeProviderEntry):
        if effort == "off":
            raise ProviderError(
                f"effort = off has no Claude Code equivalent ([providers.{provider_name}]"
                " passes --effort low..max); set the role's effort to low or unset it",
                fatal=True,
            )
        return ClaudeCodeProvider(
            model=model,
            binary=entry.binary,
            effort=effort,
            transcript_sink=transcript_sink,
            budget=budget,
            context_tokens=models_registry.context_window(provider_name, model),
        )
    extra_headers = tuple(sorted(entry.extra_headers.items()))
    extra_body = dict(entry.extra_body)
    extra_query = dict(entry.extra_query)
    if isinstance(entry, ChatGPTProviderEntry):
        chatgpt_credential = ChatGPTCredential(provider_name)
        account = chatgpt_credential.account_id()  # raises the connect hint when not signed in
        if not account:
            raise ProviderError(
                f"The stored ChatGPT sign-in for {provider_name!r} carries no account id;"
                " run `agent6 connect chatgpt` to sign in again."
            )
        return ChatGPTProvider(
            model=model,
            credential=chatgpt_credential,
            account_id=account,
            base_url=entry.base_url,
            extra_headers=extra_headers,
            extra_body=extra_body,
            extra_query=extra_query,
            timeout_s=entry.http_timeout_s,
            transcript_sink=transcript_sink,
            budget=budget,
            reasoning_effort=effort,
        )
    key = resolve_api_key(provider_name, entry.api_key_env)
    credential = (
        CommandToken(entry.token_command, ttl_s=entry.token_command_ttl_s)
        if entry.token_command
        else None
    )
    if isinstance(entry, AnthropicProviderEntry):
        # Anthropic requires explicit auth (a missing key is a 401, not a local
        # endpoint); a token_command credential or `auth_style = "none"` satisfies it.
        if not key and credential is None and entry.auth_style != "none":
            raise ProviderError(
                f"No API key for provider {provider_name!r}. Run `agent6 connect`"
                f" to store one, or set the {entry.api_key_env or 'provider'} env var."
            )
        return AnthropicProvider(
            api_key=key or "",
            model=model,
            base_url=entry.base_url,
            deployment=entry.deployment,
            auth_style=entry.auth_style,
            prompt_caching=entry.prompt_caching,
            timeout_s=entry.http_timeout_s,
            transcript_sink=transcript_sink,
            budget=budget,
            effort=effort,
            extra_headers=extra_headers,
            extra_body=extra_body,
            extra_query=extra_query,
            credential=credential,
        )
    return OpenAIProvider(
        api_key=key or "",
        model=model,
        base_url=entry.base_url,
        deployment=entry.deployment,
        auth_style=entry.auth_style,
        extra_headers=extra_headers,
        extra_body=extra_body,
        extra_query=extra_query,
        timeout_s=entry.http_timeout_s,
        transcript_sink=transcript_sink,
        budget=budget,
        reasoning_effort=effort,
        credential=credential,
    )


def close_provider(provider: Provider) -> None:
    """Release what a provider holds: a `claude_code` session's child process.
    The HTTP providers hold nothing and have no `close`."""
    close = getattr(provider, "close", None)
    if callable(close):
        close()


def role_temperature(cfg: Config, role: RoleName) -> float | None:
    """The configured sampling temperature for *role* (worker fallback)."""
    rm = cfg.models.resolve(role)
    return rm.temperature if rm is not None else None


@dataclass(frozen=True, slots=True)
class InstrumentedProvider:
    """Wraps any Provider with role.call / role.result / budget.update emission.

    Pure decoration; the inner provider is unchanged. `events` None (a caller
    with no log to feed) simply emits nothing.
    """

    inner: Provider
    role: str
    model: str
    provider_name: str
    events: EventSink | None
    budget: BudgetTracker
    stream_text: bool = False

    def call(
        self,
        *,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[ToolDefinition] | None = None,
        max_tokens: int = 4096,
        temperature: float | None = None,
        reasoning_effort: str | None = None,
        text_delta_callback: Callable[[str], None] | None = None,
        thinking_delta_callback: Callable[[str], None] | None = None,
        should_abort: Callable[[], bool] | None = None,
        should_interrupt: Callable[[], bool] | None = None,
    ) -> ProviderResponse:
        if self.events is not None:
            self.events.emit(
                "role.call",
                role=self.role,
                model=self.model,
                provider=self.provider_name,
            )
        # When the inner provider streams, fan visible text + reasoning deltas
        # out as `role.text_delta` / `role.thinking_delta` events. Every live
        # view (TUI, `watch`, the CLI ConsoleView) subscribes to these; any
        # caller-passed callback is chained through unchanged.
        role_for_event = self.role
        events = self.events

        def _on_text(piece: str) -> None:
            if events is not None:
                events.emit("role.text_delta", role=role_for_event, text=piece)
            if text_delta_callback is not None:
                text_delta_callback(piece)

        def _on_thinking(piece: str) -> None:
            if events is not None:
                events.emit("role.thinking_delta", role=role_for_event, text=piece)
            if thinking_delta_callback is not None:
                thinking_delta_callback(piece)

        stream = (
            self.stream_text
            or text_delta_callback is not None
            or thinking_delta_callback is not None
        )
        effective_text_cb = _on_text if stream else None
        effective_thinking_cb = _on_thinking if stream else None
        try:
            resp = self.inner.call(
                system=system,
                messages=messages,
                tools=tools,
                max_tokens=max_tokens,
                temperature=temperature,
                reasoning_effort=reasoning_effort,
                text_delta_callback=effective_text_cb,
                thinking_delta_callback=effective_thinking_cb,
                should_abort=should_abort,
                should_interrupt=should_interrupt,
            )
        except Exception as exc:
            if isinstance(exc, ProviderError) and not exc.provider:
                exc.provider = self.provider_name  # the hint names the config key
            if self.events is not None:
                self.events.emit("role.result", role=self.role, ok=False, error=str(exc)[:200])
            self._emit_budget()
            raise
        if self.events is not None:
            self.events.emit(
                "role.result",
                role=self.role,
                ok=True,
                # The turn's settled prose. The deltas are the same text
                # arriving in pieces and are emitted only when streaming is on,
                # so a headless run (CI, a redirected stdout, every spawned
                # ask) wrote a journal with no assistant text at all -- and
                # every reader of it, `read_session` and `/btw` included, found
                # nothing.
                text=resp.text,
                tokens_in=resp.input_tokens,
                tokens_out=resp.output_tokens,
                cache_read=resp.cache_read_tokens,
                cache_creation=resp.cache_creation_tokens,
                stop_reason=resp.stop_reason,
            )
        self._emit_budget()
        return resp

    def close(self) -> None:
        close_provider(self.inner)

    def _emit_budget(self) -> None:
        """Cumulative spend after a call, whether it returned or RAISED.

        A stream cut after the provider reported usage is still billed, and the
        providers record it. These events are the only path that spend takes to
        a surface: the live cost meters fold them, and a machine's spend ledger
        (`app.machine._spend`) reconstructs a state's cost from the last one in
        its log. Emitting only on success left a failed call's real dollars out
        of every one of them, and out of the journal for good, since the
        end-of-run summary prints to the terminal and is never journalled.
        """
        if self.events is None:
            return
        snap = self.budget.snapshot()
        usd_total, usd_partial = self.budget.estimate_usd()
        plan = snap.plan_latest
        self.events.emit(
            "budget.update",
            input_total=snap.input_total,
            output_total=snap.output_total,
            usd_total=usd_total,
            usd_partial=usd_partial,
            usd_cap=snap.max_usd,
            tokens_unmetered=snap.unmetered_tokens,
            tokens_fallback_cap=snap.max_tokens_fallback,
            plan_used_percent=plan.used_percent if plan else 0.0,
            plan_consumed=snap.plan_consumed,
            plan_cap=snap.max_percent,
            plan_resets_at=plan.resets_at if plan else 0.0,
            # Every window the backend reported plus the purchased-credit
            # family, raw: the derived fields above cannot show a second
            # (e.g. 5-hour) window existing, and hiding tracked state from
            # the journal blinds the operator to it.
            plan_windows=[
                {
                    "name": w.name,
                    "used_percent": w.used_percent,
                    "window_minutes": w.window_minutes,
                    "resets_at": w.resets_at,
                }
                for w in (plan.windows if plan else ())
            ],
            credits_has=plan.has_credits if plan else False,
            credits_unlimited=plan.credits_unlimited if plan else False,
            credits_balance=plan.credits_balance if plan else "",
        )


def reviewer_seat_provider(
    cfg: Config,
    seat: str,
    *,
    transcript_sink: TranscriptSink,
    budget: BudgetTracker,
    events: EventSink | None,
) -> Provider:
    """The reviewer role routed under *seat*'s label, instrumented: a panel
    seat, the prompt reviser, or the tier-2 context summariser (always
    available, since compaction can fire on any run, and cheaper than the
    worker model)."""
    inner = build_role_provider(
        cfg, "reviewer", transcript_sink=transcript_sink, budget=budget, seat=seat
    )
    rm = cfg.models.resolve("reviewer")
    assert rm is not None  # "reviewer" resolves to the reviewer or worker model
    return InstrumentedProvider(
        inner=inner,
        role=seat,
        model=rm.model,
        provider_name=rm.provider,
        events=events,
        budget=budget,
    )


# The simple-form panel roster: adversarial lenses cycled when no explicit
# seats are configured.
_DEFAULT_PERSONAS = ("security", "correctness", "tests", "over-engineering", "edge-cases")


def build_review_seats(
    cfg: Config,
    *,
    transcript_sink: TranscriptSink,
    budget: BudgetTracker,
    n: int,
    personas: tuple[str, ...] = (),
    model_override: str = "",
    events: EventSink | None = None,
) -> list[ReviewSeat]:
    """Build the review-panel seats, one per roster entry. An entry is
    `persona[@provider/model]`: with a provider and model the seat is pinned
    to them (`--model X` overrides the model and keeps the provider), and a
    bare persona routes via `[models.reviewer]`. `cfg.review.seats` names
    the roster outright; otherwise `n` seats cycle *personas* (the
    `--personas` flag, else a built-in set), so both surfaces speak one
    grammar.

    With *events*, each seat is instrumented: only InstrumentedProvider emits
    `budget.update`, so bare seat providers spent real money no surface ever
    showed (the tracker enforced; the log never heard). `agent6 review` passes
    None -- it has no session log."""

    def _instrumented(provider: Provider, persona: str, model: str, provider_name: str) -> Provider:
        if events is None:
            return provider
        return InstrumentedProvider(
            inner=provider,
            role=f"review:{persona}",
            model=model,
            provider_name=provider_name,
            events=events,
            budget=budget,
        )

    if cfg.review.seats:
        specs = list(cfg.review.seats)
    else:
        pool = list(personas) if personas else list(_DEFAULT_PERSONAS)
        specs = [pool[i % len(pool)] for i in range(max(1, n))]
    rm = cfg.models.resolve("reviewer")
    seats: list[ReviewSeat] = []
    for spec in specs:
        try:
            persona, provider_name, model = parse_seat_spec(spec)
        except ValueError as exc:
            raise ProviderError(f"review seat: {exc}") from exc
        persona = persona or "general"
        if provider_name and model:
            entry = cfg.providers.get(provider_name)
            if entry is None:
                raise ProviderError(
                    f"review seat {spec!r} names provider {provider_name!r} but"
                    f" [providers.{provider_name}] is missing"
                )
            seat_model = model_override or model
            provider = _provider_from_entry(
                provider_name,
                entry,
                seat_model,
                None,
                # Stamp the seat like build_role_provider does: an unstamped
                # sink records seat="", which the conversation fold reads as
                # the driving seat and renders as worker turns.
                transcript_sink=transcript_sink.for_seat(f"review:{persona}"),
                budget=budget,
            )
            label = f"{provider_name}/{seat_model}"
            provider = _instrumented(provider, persona, seat_model, provider_name)
        else:
            provider = build_role_provider(
                cfg,
                "reviewer",
                transcript_sink=transcript_sink,
                budget=budget,
                model_override=model_override,
                seat=f"review:{persona}",
            )
            label = model_override or (rm.model if rm is not None else "reviewer")
            provider = _instrumented(
                provider, persona, label, rm.provider if rm is not None else ""
            )
        seats.append(
            ReviewSeat(persona=persona, model=label, provider=provider, tier=cfg.review.tier)
        )
    return seats


def build_prompt_reviser_provider(
    cfg: Config,
    *,
    transcript_sink: TranscriptSink,
    budget: BudgetTracker,
    events: EventSink,
) -> Provider | None:
    """Route the reviewer role as a one-shot prompt reviser."""
    if cfg.prompt.revise_prompt == "off":
        return None
    return reviewer_seat_provider(
        cfg, "prompt_reviser", transcript_sink=transcript_sink, budget=budget, events=events
    )
