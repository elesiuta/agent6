# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""argcomplete completers for the CLI parser."""

from __future__ import annotations

import argparse
import contextlib
import functools
from collections.abc import Callable
from pathlib import Path
from typing import Any

from agent6.config import (
    Config,
    ConfigError,
)
from agent6.config.layer import (
    EffectiveConfig,
    available_preset_names,
    leaf_keys,
    load_effective,
    preset_catalog,
    resolved_state_dir,
)
from agent6.ui.cli._common import (
    _plans_dir,
    session_bucket_dirs,
)


def _explicit_config(kw: dict[str, object]) -> Path | None:
    """The `--config FILE` already typed on the line being completed, so
    completions describe the config the command will actually run under."""
    parsed = kw.get("parsed_args")
    raw = getattr(parsed, "config", None)
    return raw if isinstance(raw, Path) else None


def _never_raises(fn: Callable[..., list[str]]) -> Callable[..., list[str]]:
    """Suggestions or nothing -- never an exception.

    argcomplete calls these on Tab, inside the operator's shell, where an
    exception is a traceback dumped over the command line. Every completer that
    touches the config or the filesystem wears this, instead of each growing
    its own try/except and several never growing one.
    """

    @functools.wraps(fn)
    def guarded(*args: Any, **kwargs: Any) -> list[str]:
        try:
            return fn(*args, **kwargs)
        except Exception:
            return []

    return guarded


@_never_raises
def _complete_providers(prefix: str, **_kw: object) -> list[str]:
    """argcomplete: connected provider names + known presets."""
    from agent6.config.write import PROVIDER_DEFAULTS  # noqa: PLC0415
    from agent6.ui.cli.model import _connected_providers  # noqa: PLC0415

    names = set(_connected_providers(None)) | set(PROVIDER_DEFAULTS)
    return sorted(n for n in names if n.startswith(prefix))


@_never_raises
def _complete_presets(prefix: str, **kw: object) -> list[str]:
    """argcomplete: built-in presets + configured [presets.*] names, under
    the `--config` already typed."""
    names = available_preset_names(Path.cwd(), _explicit_config(kw))
    return [n for n in names if n.startswith(prefix)]


@_never_raises
def _complete_skills(prefix: str, **_kw: object) -> list[str]:
    """argcomplete: installed + extra_dirs skill names."""
    from agent6.ui.cli.skills_cmds import resolved_skill_names_for_completion  # noqa: PLC0415

    return [n for n in resolved_skill_names_for_completion(Path.cwd()) if n.startswith(prefix)]


@_never_raises
def _complete_mcp_servers(prefix: str, **kw: object) -> list[str]:
    """argcomplete: the configured MCP server names."""
    cfg = load_effective(Path.cwd(), _explicit_config(kw)).config
    return sorted(n for n in cfg.mcp.servers if n.startswith(prefix))


@_never_raises
def _complete_models(
    prefix: str, parsed_args: argparse.Namespace | None = None, **_kw: object
) -> list[str]:
    """argcomplete: live + configured model ids for the already-typed provider."""
    provider = getattr(parsed_args, "provider", "") or ""
    if not provider:
        return []
    from agent6.ui.cli.model import _models_for  # noqa: PLC0415

    return [m for m in _models_for(None, provider) if m.startswith(prefix)]


def _all_parallel_model_names(config_path: Path | None = None) -> list[str]:
    """Model ids a `/parallel` lane can actually run: the WORKER provider's
    catalog (lanes inherit the worker provider; only the model is overridden per
    lane), from the same live + configured source `agent6 model` completes from."""
    try:
        eff = load_effective(Path.cwd(), config_path)
    except ConfigError:
        return []
    worker = eff.config.models.worker
    if worker is None:
        return []
    from agent6.ui.cli.model import _models_for  # noqa: PLC0415

    return sorted(set(_models_for(None, worker.provider)))


@_never_raises
def _complete_parallel_models(prefix: str, **kw: object) -> list[str]:
    """argcomplete for `run --parallel`: the worker provider's model ids,
    completing the token after the last comma so a `m1,m2,...` list completes
    member by member (an integer lane count is typed, not completed)."""
    head, sep, frag = prefix.rpartition(",")
    lead = head + sep  # "" for the first/only model, "m1," while extending a list
    names = _all_parallel_model_names(_explicit_config(kw))
    return sorted(lead + m for m in names if m.startswith(frag))


# Values TAB must not offer even though the schema allows them, keyed by leaf.
# `sandbox.isolation = "none"` is the unsandboxed opt-out: TAB should not put
# "disable the sandbox" one keystroke away. Type it explicitly to set it.
_WITHHELD_ENUM_VALUES: dict[str, frozenset[str]] = {"sandbox.isolation": frozenset({"none"})}


def _config_enum_choices(config_path: Path | None = None) -> dict[str, tuple[str, ...]]:
    """Every enum leaf's allowed values, read from the schema through the same
    view the config surfaces render. A hand-kept copy drifted: leaves added
    since offered nothing on TAB."""
    from agent6.viewmodel.config_view import build_config_view  # noqa: PLC0415

    try:
        view = build_config_view(load_effective(Path.cwd(), config_path))
    except ConfigError:
        # A config that does not load still gets completion: the schema is
        # what carries the choices, and a default config is all schema; the
        # preset names still come from the raw layers.
        view = build_config_view(
            EffectiveConfig(
                config=Config(),
                sources={},
                layers=(),
                presets=tuple(available_preset_names(Path.cwd(), config_path)),
            )
        )
    out: dict[str, tuple[str, ...]] = {}
    for setting in view.settings:
        if setting.py_type != "choice" or not setting.choices:
            continue
        withheld = _WITHHELD_ENUM_VALUES.get(setting.key, frozenset())
        out[setting.key] = tuple(c for c in setting.choices if c not in withheld)
    return out


def _user_preset_names() -> list[str]:
    """USER-defined [presets.*] names only, for key completion. Built-in names
    are deliberately absent: writing presets.ultra.* creates a user table that
    REPLACES the built-in wholesale, a footgun TAB should not put one keystroke
    away (the same rule keeps `none` out of sandbox.isolation completion)."""
    try:
        return [p.name for p in preset_catalog(Path.cwd()).presets if p.origin != "built-in"]
    except ConfigError:
        return []


@_never_raises
def _complete_config_keys(prefix: str, *, settable: bool = True, **kw: object) -> list[str]:
    """argcomplete: known dotted config leaf paths (effective + enum keys).
    From `preset` onward, also the user's presets.<name>.<leaf> paths (kept
    out of the bare-TAB listing, which is crowded enough already).

    `settable=False` for `config get`, which reads EFFECTIVE leaves only:
    both the enum keys (offered so `config set` can reach a leaf no layer has
    set yet) and `[presets.*]` paths (stripped before validation) are inputs
    `get` rejects, and a completer must offer what its command accepts.
    """
    explicit = _explicit_config(kw)
    try:
        keys = set(leaf_keys(load_effective(Path.cwd(), explicit)))
    except ConfigError:
        keys = set()
    if settable:
        keys |= set(_config_enum_choices(explicit))
    if settable and prefix.startswith("preset"):
        pool = {k for k in keys if k != "preset"}
        keys |= {f"presets.{name}.{k}" for name in _user_preset_names() for k in pool}
    return sorted(k for k in keys if k.startswith(prefix))


# Presets offered for any `providers.<name>.extra_body` value (the provider name
# varies, so this is matched by suffix, not a schema enum). The first
# is the recommended OpenRouter routing, a fast, prefix-caching backend.
_EXTRA_BODY_RECIPES: tuple[str, ...] = (
    '{ provider = { sort = "throughput" } }',
    '{ provider = { sort = "latency" } }',
    '{ provider = { sort = "price" } }',
)


@_never_raises
def _complete_config_values(
    prefix: str, parsed_args: argparse.Namespace | None = None, **_kw: object
) -> list[str]:
    """argcomplete: the values the config key already typed accepts: its enum
    or configured choices (the schema's literals, the preset names, the
    configured providers), a role's provider's model ids, the extra_body
    recipes."""
    key = getattr(parsed_args, "key", "") or ""
    raw = getattr(parsed_args, "config", None)
    config_path = raw if isinstance(raw, Path) else None
    choices = list(_config_enum_choices(config_path).get(key, ()))
    if key.endswith(".extra_body"):
        choices += list(_EXTRA_BODY_RECIPES)
    if not choices:
        with contextlib.suppress(ConfigError):
            from agent6.models.choices import config_value_choices  # noqa: PLC0415

            choices = config_value_choices(load_effective(Path.cwd(), config_path), key)
    return [v for v in choices if v.startswith(prefix)]


@_never_raises
def _complete_model_provider(
    prefix: str, parsed_args: argparse.Namespace | None = None, **_kw: object
) -> list[str]:
    """argcomplete for `agent6 model <role> <provider>`.

    Only offer provider names once a valid role has been typed. argcomplete
    bleeds every nargs='?' positional's completer into the first slot, so
    without this gate `agent6 model <TAB>` would mix provider names into the
    role choices (and `agent6 model openrouter` then fails the role validator).
    """
    role = getattr(parsed_args, "role", None)
    if role not in ("planner", "worker", "reviewer", "all"):
        return []
    return _complete_providers(prefix)


@_never_raises
def _complete_session_ids(prefix: str, **_kw: object) -> list[str]:
    """argcomplete: ids across every session bucket (runs, asks, machine
    drafts). Offers exactly what `--from` accepts, so the two cannot drift."""
    out: list[str] = []
    for bucket in session_bucket_dirs(Path.cwd()):
        if not bucket.is_dir():
            continue
        out += [d.name for d in bucket.iterdir() if d.is_dir() and d.name.startswith(prefix)]
    return sorted(out)


@_never_raises
def _complete_session_ports(prefix: str, parsed_args: object = None, **_kw: object) -> list[str]:
    """argcomplete: the ports that session is ACTUALLY listening on.

    Offering every valid input rather than nothing: the whole difficulty of
    reaching a run's dev server is not knowing its port, and only something
    inside the run's network can see it.
    """
    target = str(getattr(parsed_args, "target", "") or "")
    from agent6.sessions.ipc import listening_ports  # noqa: PLC0415
    from agent6.sessions.layout import session_layout  # noqa: PLC0415

    layout = session_layout(resolved_state_dir(Path.cwd()), target) if target else None
    if layout is None:
        return []
    return [str(p) for p in listening_ports(layout.session_dir) if str(p).startswith(prefix)]


@_never_raises
def _complete_resumable_ids(prefix: str, **_kw: object) -> list[str]:
    """argcomplete: ids `resume`/`fork` can actually pick up.

    Every bucket whose mode is resumable, so a plan and an ask are offered --
    but not a `machine create` draft, which resume refuses. Offering what a
    verb accepts, no less and no more.
    """
    out: list[str] = []
    from agent6.app.resume import resumable_bucket_dirs  # noqa: PLC0415

    for bucket in resumable_bucket_dirs(resolved_state_dir(Path.cwd())):
        if not bucket.is_dir():
            continue
        out += [d.name for d in bucket.iterdir() if d.is_dir() and d.name.startswith(prefix)]
    return sorted(out)


@_never_raises
def _complete_plan_session_ids(prefix: str, **_kw: object) -> list[str]:
    """argcomplete: plan ids (for --from-plan / plan show/edit)."""
    plans = _plans_dir(Path.cwd())
    if not plans.is_dir():
        return []
    return sorted(
        p.name
        for p in plans.iterdir()
        if p.is_dir() and p.name.startswith(prefix) and (p / "plan.md").is_file()
    )


@_never_raises
def _complete_machine_ids(prefix: str, **_kw: object) -> list[str]:
    """argcomplete: live machine instance ids (dirs under the per-repo state dir's machines/)."""
    from agent6.sessions.layout import machines_root  # noqa: PLC0415

    base = machines_root(resolved_state_dir(Path.cwd()))
    if not base.is_dir():
        return []
    return sorted(p.name for p in base.iterdir() if p.is_dir() and p.name.startswith(prefix))


@_never_raises
def _complete_watch_targets(prefix: str, **_kw: object) -> list[str]:
    """argcomplete: every session id plus every machine id -- what `attach`
    accepts. It resolves a session across all buckets, so offering only the
    runs bucket hid the plans and asks it opens happily."""
    return sorted(set(_complete_session_ids(prefix) + _complete_machine_ids(prefix)))


@_never_raises
def _complete_machine_files(prefix: str, **_kw: object) -> list[str]:
    """argcomplete: machine `*.asm.toml` files under cwd and the machines dir."""
    out: set[str] = set()
    from agent6.sessions.layout import machines_root  # noqa: PLC0415

    for base in (Path.cwd(), machines_root(resolved_state_dir(Path.cwd()))):
        if base.is_dir():
            out.update(str(p) for p in base.rglob("*.asm.toml"))
    return sorted(p for p in out if p.startswith(prefix))
