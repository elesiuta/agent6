# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Layered config resolution + auditing for agent6.

Config is assembled from layered sources, lowest precedence first:

1. `default`, the secure defaults baked into the pydantic model,
2. `global` , `$XDG_CONFIG_HOME/agent6/config.toml` (user-wide),
3. `repo`   , the per-repo config under the state dir (out of the workspace,
   `<state-base>/<repo-id>/config.toml`; see `agent6.paths.state_dir`),
4. `flag`   , an explicit `--config FILE` (power users / CI),
5. `machine`, the machine agent's per-state overlay,

plus a selected preset, injected as below. Raw TOML dicts are deep-merged in
that order and validated **once**, so a repo can override a single field
without restating the rest. Every leaf remembers which layer last set it,
which powers `agent6 config show`.

A selected preset is injected just ABOVE the config layer that
SELECTED it (`--preset` flag / repo / global top-level `preset`), so the
preset OVERRIDES that config while a more-specific config layer (or an explicit
`--config FILE` / machine overlay) still overrides the preset. Only the
most-specific source's preset is injected -- global and repo presets never
stack. See :func:`_apply_preset`.
"""

from __future__ import annotations

import contextlib
import tomllib
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from pydantic import ValidationError

from agent6.config.io import (
    format_toml_value,
    read_toml_file,
    read_toml_leaf,
    toml_key,
)
from agent6.config.model import (
    Config,
    ConfigError,
    validate_config,
)
from agent6.paths import (
    global_config_path,
    repo_config_path,
)

LayerName = Literal["default", "preset", "global", "repo", "flag", "machine"]

# Display order for `config show` / `config fill`, derived FROM the Config model's
# field declaration order so a new section can never be silently omitted. Scalar
# top-level fields (e.g. `preset`) carry no `[section]` table and are rendered
# inline by their parent, so the section ordering only needs the table names; we
# keep every field name here and the lookups below tolerate non-section entries.
SECTION_ORDER = tuple(Config.model_fields)


@dataclass(frozen=True, slots=True)
class Layer:
    name: LayerName
    path: Path | None
    data: dict[str, Any]


@dataclass(frozen=True, slots=True)
class EffectiveConfig:
    config: Config
    sources: dict[str, str]  # dotted leaf path -> layer name
    layers: tuple[Layer, ...]  # the layers that actually contributed (existing files)
    # The preset names this load knew: the built-ins plus the user's
    # `[presets.*]` tables (stripped from `layers` before validation), sorted.
    presets: tuple[str, ...] = ()

    @property
    def explicit_leaves(self) -> frozenset[str]:
        """The leaves a config layer set, without the ones sitting at their
        built-in default. A preflight that refuses what the host cannot honor
        asks this: the operator wrote the value down, so it is a demand rather
        than a default to degrade."""
        return frozenset(leaf for leaf, layer in self.sources.items() if layer != "default")


def _read_toml(path: Path) -> dict[str, Any]:
    try:
        return tomllib.loads(path.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"Config file is not valid TOML ({path}): {exc}") from exc
    except (OSError, UnicodeDecodeError) as exc:
        # An unreadable config is the operator's file, not an agent6 defect: a
        # root-owned one after a sudo run, a directory at the path, a stray
        # non-UTF-8 byte. It refuses, never crash-reports.
        raise ConfigError(f"Config file cannot be read ({path}): {exc}") from exc


def _forbid_layer_preset(layer_name: str, data: dict[str, Any]) -> None:
    """Reject a top-level `preset` key in a layer that cannot SELECT one.

    Only the global/repo configs and the --preset flag select one
    (_select_preset), so the key deep-merging in from a --config FILE or a
    machine [config] overlay would show as effective while never applying.
    Honoring it instead is not an option: a machine overlay selecting an
    operator [presets.*] could pick one that loosens the sandbox.
    """
    if "preset" in data:
        raise ConfigError(
            f"top-level `preset` selects a config preset only from the global/repo"
            f" config or the --preset flag, not the {layer_name} config; use"
            f" --preset <name> or set it in your repo/global config."
        )


def discover_layers(repo_root: Path, explicit_path: Path | None) -> list[Layer]:
    """The config layers that exist, in precedence order (low -> high).

    The repo config lives out of the workspace under the state dir.
    """
    layers: list[Layer] = []
    gpath = global_config_path()
    if gpath.is_file():
        layers.append(Layer("global", gpath, _read_toml(gpath)))
    rpath = repo_config_path(repo_root)
    if rpath.is_file():
        layers.append(Layer("repo", rpath, _read_toml(rpath)))
    if explicit_path is not None:
        if not explicit_path.is_file():
            raise ConfigError(f"--config file not found: {explicit_path}")
        data = _read_toml(explicit_path)
        _forbid_layer_preset("--config", data)
        layers.append(Layer("flag", explicit_path, data))
    return layers


# Built-in config presets: named presets that fill in many settings at once, so
# a task can pick a strategy with one knob (`--preset ultra`) instead of tuning
# the [review] / budget knobs by hand. Each value is a nested config dict spliced
# ABOVE the layer that SELECTED the preset (so the preset's settings OVERRIDE
# that layer's; see _apply_preset). Users add their own via [presets.<name>]
# tables in config.toml. BUILTIN_PRESET_NOTES says what each is for, one line,
# the docs table and the `--preset` help print it.
BUILTIN_PRESET_NOTES: dict[str, str] = {
    "standard": "the plain defaults, no review panel",
    "quick": "no review panel: fast and cheap",
    "ultra": "a three-seat review panel that vetoes the finish it does not pass",
    "paranoid": "five explore-tier review seats vetoing the finish: maximum scrutiny",
}
BUILTIN_PRESETS: dict[str, dict[str, Any]] = {
    # The pre-feature baseline: plain defaults, no review panel.
    "standard": {},
    # Fast/cheap: no review.
    "quick": {
        "review": {"trigger": "off"},
    },
    # The "ultracode" tier: a 3-seat grounded panel that advises + gates.
    "ultra": {
        "review": {
            "trigger": "before_finish",
            # veto, not quorum: the 3 seats share one model (the gate counts one
            # block per DISTINCT model, so quorum>1 would be unreachable here).
            "decision": "veto",
            "seats": ["security", "correctness", "tests"],
            "concurrency": 3,  # seats in parallel: panel latency = slowest seat
        },
    },
    # Maximum scrutiny: 5 explore-tier seats, before_finish veto.
    "paranoid": {
        "review": {
            "trigger": "before_finish",
            "decision": "veto",
            "tier": "explore",
            "seats": [
                "security",
                "correctness",
                "tests",
                "edge-cases",
                "over-engineering",
            ],
            "concurrency": 5,  # seats in parallel: panel latency = slowest seat
        },
    },
}


def resolve_preset(name: str, user_presets: dict[str, Any]) -> dict[str, Any]:
    """The config-override dict for preset *name* (user presets win over
    built-ins of the same name). "" -> {} (nothing selected). Raises
    ConfigError for an unknown name so a typo'd preset fails loudly.

    "standard" is not special-cased: it is a built-in like any other (an empty
    override), so a user table of that name replaces it exactly as the docs
    promise. A short-circuit here would drop those overrides silently while
    `config presets` reports them applied."""
    if not name:
        return {}
    if name in user_presets:
        prof = user_presets[name]
        if not isinstance(prof, dict):
            raise ConfigError(f"[presets.{name}] must be a table, got {type(prof).__name__}")
        return prof
    if name in BUILTIN_PRESETS:
        return BUILTIN_PRESETS[name]
    known = ", ".join(sorted({*BUILTIN_PRESETS, *user_presets}))
    raise ConfigError(f"unknown preset {name!r}. Known presets: {known}.")


def preset_names(layers: Iterable[Layer]) -> list[str]:
    """Preset names a chooser offers: the built-ins plus the user's
    `[presets.<name>]` tables in *layers* (the same source `--preset` resolves
    against), sorted and de-duplicated."""
    names: set[str] = set(BUILTIN_PRESETS)
    for layer in layers:
        prof = layer.data.get("presets")
        if isinstance(prof, dict):
            names.update(prof.keys())
    return sorted(names)


def available_preset_names(repo_root: Path, explicit_path: Path | None = None) -> list[str]:
    """`preset_names` over the discovered layers of *repo_root* (+ an explicit
    `--config` file). A config-read failure degrades to the built-ins alone,
    so a chooser never blocks on a bad config."""
    layers: list[Layer] = []
    with contextlib.suppress(Exception):
        layers = discover_layers(repo_root, explicit_path)
    return preset_names(layers)


@dataclass(frozen=True, slots=True)
class PresetInfo:
    """One preset as `config presets` shows it."""

    name: str
    overrides: dict[str, Any]  # the nested config dict it applies ({} = plain defaults)
    origin: str  # "built-in", "global", "repo", or "global+repo"
    replaces_builtin: bool  # a user preset with a built-in's name (wholesale replace)


@dataclass(frozen=True, slots=True)
class PresetCatalog:
    presets: tuple[PresetInfo, ...]  # built-ins in definition order, then user's sorted
    selected: str  # "" when no preset is selected anywhere
    source: str  # "repo" / "global" / "none"


def preset_catalog(repo_root: Path, explicit_path: Path | None = None) -> PresetCatalog:
    """Everything `config presets` lists: each known preset with the
    overrides it would apply (a user table REPLACES a same-named built-in, so
    only the effective body is reported), plus the selected name and its
    source. Unlike :func:`available_preset_names` this fails loudly on a
    broken config: it is an audit surface, not a chooser fallback."""
    layers = discover_layers(repo_root, explicit_path)
    user: dict[str, dict[str, Any]] = {}
    origins: dict[str, str] = {}
    for layer in layers:
        prof = layer.data.get("presets")
        if not isinstance(prof, dict):
            continue
        for name, body in prof.items():
            if not isinstance(body, dict):
                raise ConfigError(f"[presets.{name}] must be a table, got {type(body).__name__}")
            user[name] = _deep_merge(user.get(name, {}), body)
            origins[name] = f"{origins[name]}+{layer.name}" if name in origins else layer.name
    selected, source = _select_preset(layers, "")
    builtins = tuple(
        PresetInfo(name, user[name], origins[name], replaces_builtin=True)
        if name in user
        else PresetInfo(name, body, "built-in", replaces_builtin=False)
        for name, body in BUILTIN_PRESETS.items()
    )
    customs = tuple(
        PresetInfo(name, user[name], origins[name], replaces_builtin=False)
        for name in sorted(user)
        if name not in BUILTIN_PRESETS
    )
    return PresetCatalog((*builtins, *customs), selected, source)


def _format_changed(val: object, existing: object) -> bool:
    """The one wholesale-REPLACE rule the merge and the provenance walk share:
    a discriminated dict (e.g. a [providers.<name>] entry) whose `api_format`
    changes between layers must REPLACE, not deep-merge -- the lower layer's
    format-specific keys (an anthropic prompt_caching, say) are invalid under
    the new format and would otherwise survive the merge and surface as a
    confusing extra_forbidden error."""
    return (
        isinstance(val, dict)
        and isinstance(existing, dict)
        and "api_format" in val
        and "api_format" in existing
        and val.get("api_format") != existing.get("api_format")
    )


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, val in override.items():
        existing = out.get(key)
        if isinstance(val, dict) and isinstance(existing, dict):
            out[key] = val if _format_changed(val, existing) else _deep_merge(existing, val)
        else:
            out[key] = val
    return out


def _merge_layers(layers: list[Layer]) -> tuple[dict[str, Any], dict[str, str]]:
    """Deep-merge *layers* low->high and stamp per-leaf provenance IN the same
    walk, so the two can never diverge: on a wholesale replace (an
    api_format-changing provider entry) the stale sub-provenance dies with the
    subtree, then the winner's leaves are stamped. A separate provenance pass
    would keep source entries for leaves the merge discarded."""
    merged: dict[str, Any] = {}
    sources: dict[str, str] = {}

    def walk(
        base: dict[str, Any], override: dict[str, Any], layer_name: str, prefix: str
    ) -> dict[str, Any]:
        out = dict(base)
        for key, val in override.items():
            path = f"{prefix}{key}"
            existing = out.get(key)
            if (
                isinstance(val, dict)
                and isinstance(existing, dict)
                and not _format_changed(val, existing)
            ):
                out[key] = walk(existing, val, layer_name, f"{path}.")
            else:
                for stale in [k for k in sources if k == path or k.startswith(f"{path}.")]:
                    del sources[stale]
                out[key] = val
                if isinstance(val, dict) and val:
                    for leaf in flatten_leaves(val, prefix=f"{path}."):
                        sources[leaf] = layer_name
                else:
                    sources[path] = layer_name
        return out

    for layer in layers:
        merged = walk(merged, layer.data, layer.name, "")
    return merged, sources


def flatten_leaves(data: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    """Flatten nested dicts to dotted leaf paths.

    Lists (incl. arrays of tables) are treated as leaves, their provenance
    is the whole array, not individual elements.
    """
    out: dict[str, Any] = {}
    for key, val in data.items():
        path = f"{prefix}{key}"
        if isinstance(val, dict) and val:
            out.update(flatten_leaves(val, prefix=f"{path}."))
        else:
            out[path] = val
    return out


def _leaf_fix_hint(
    layers: list[Layer], source_of_leaf: dict[str, str]
) -> Callable[[str, str], str | None]:
    """Locator for validate_config: (dotted leaf, pydantic error type) -> "set in
    <layer> <path>; fix: <command>", or None when the value came from a built-in
    default (nothing to fix). Lets a stale value name the exact file and the
    command to correct it.

    A key `Config` has no field for (`extra_forbidden`) points at `agent6 config
    fix`, which drops it: `config set` refuses a key it cannot resolve.
    """
    by_name = {layer.name: layer for layer in layers}

    def locate(leaf: str, error_type: str) -> str | None:
        layer = by_name.get(source_of_leaf.get(leaf, ""))
        if layer is None or layer.path is None:
            return None
        if layer.name == "flag":
            fix = f"edit {layer.path}"
        elif error_type == "extra_forbidden":
            fix = "agent6 config fix"
        elif layer.name == "repo":
            fix = f"agent6 config set --repo {leaf} <value>"
        else:
            fix = f"agent6 config set {leaf} <value>"
        return f"    set in the {layer.name} config: {layer.path}\n    fix: {fix}"

    return locate


def _effective_from_layers(
    layers: list[Layer], *, source: str, presets: list[str]
) -> EffectiveConfig:
    """Merge *layers* low->high, validate, and build the per-leaf source map."""
    merged, source_of_leaf = _merge_layers(layers)
    config = validate_config(merged, source=source, locate=_leaf_fix_hint(layers, source_of_leaf))
    # Source map over the *effective* config: every leaf the model
    # produced, attributed to the layer that set it or "default".
    effective_leaves = flatten_leaves(config.model_dump(mode="python"))
    sources = {leaf: source_of_leaf.get(leaf, "default") for leaf in effective_leaves}
    return EffectiveConfig(
        config=config, sources=sources, layers=tuple(layers), presets=tuple(presets)
    )


def _own_preset(layer: Layer) -> str:
    """A layer's OWN raw top-level `preset` (not the merged value), or "".

    A non-string (a `[preset]` table from a typo'd ``config set
    preset.<name>``) fails here with its own message; str()-coercing it
    produced `unknown preset "{'porifle': 'ultra'}"`.
    """
    raw = layer.data.get("preset")
    if raw is None:
        return ""
    if not isinstance(raw, str):
        shape = "a [preset] table" if isinstance(raw, dict) else f"a {type(raw).__name__}"
        raise ConfigError(
            f"top-level `preset` in the {layer.name} config must be a preset name"
            f' string (e.g. preset = "ultra"), got {shape};'
            f" set it with `agent6 config set preset <name>`."
        )
    return raw


def _select_preset(cleaned: list[Layer], preset_override: str) -> tuple[str, str]:
    """Pick the (preset name, source) most-specific first from each layer's OWN
    raw top-level `preset` (never stacking global+repo): the `--preset`
    flag, else the `repo` layer's field, else the `global` layer's field,
    else ("", "none")."""
    if preset_override:
        return preset_override, "flag"
    by_name = {layer.name: layer for layer in cleaned}
    for source in ("repo", "global"):
        layer = by_name.get(source)
        if layer is not None and (name := _own_preset(layer)):
            return name, source
    return "", "none"


def _insert_preset(cleaned: list[Layer], preset: Layer, source: str) -> list[Layer]:
    """Splice *preset* into *cleaned* at the position for its *source*.

    `global`/`repo` -> right AFTER that config layer (so the preset
    overrides it but the more-specific config layer / flag still wins). `flag`
    (`--preset`) -> just BELOW an explicit `--config FILE` / machine overlay
    if present (those still win), else appended last (overrides all config).
    """
    out: list[Layer] = []
    inserted = False
    for layer in cleaned:
        if source == "flag" and not inserted and layer.name in ("flag", "machine"):
            out.append(preset)
            inserted = True
        out.append(layer)
        if not inserted and layer.name == source:  # source in {"global", "repo"}
            out.append(preset)
            inserted = True
    if not inserted:
        out.append(preset)
    return out


def _strip_presets(layers: list[Layer]) -> tuple[list[Layer], dict[str, Any]]:
    """The layers without their `[presets.*]` tables, plus those tables merged.

    Presets are meta-config the Config schema forbids, so every path that
    validates a layer strips them first, whether or not one is applied.
    """
    cleaned: list[Layer] = []
    user_presets: dict[str, Any] = {}
    for layer in layers:
        data = dict(layer.data)
        prof = data.pop("presets", None)
        if isinstance(prof, dict):
            user_presets = _deep_merge(user_presets, prof)
        cleaned.append(Layer(layer.name, layer.path, data))
    return cleaned, user_presets


def _apply_preset(layers: list[Layer], preset_override: str) -> list[Layer]:
    """Strip `[presets]` tables out of the user layers (they are meta-config,
    not part of the validated Config) and inject the selected preset
    just ABOVE the config layer that SELECTED it, so the preset OVERRIDES that
    config while a more-specific config layer (or an explicit `--config FILE` /
    machine overlay) still overrides the preset. Only the most-specific source's
    preset is injected -- global and repo presets never stack.

    Source is chosen by :func:`_select_preset` (`--preset` flag > repo's own
    top-level `preset` > global's own), and the preset is spliced in by
    :func:`_insert_preset`. Resulting precedence (low->high): default <
    global-config < [preset if global-selected] < repo-config <
    [preset if repo-selected] < [preset if --flag] < flag(`--config FILE`) <
    machine-overlay.
    """
    cleaned, user_presets = _strip_presets(layers)
    name, source = _select_preset(cleaned, preset_override)
    overrides = resolve_preset(name, user_presets)
    if not overrides:
        return cleaned
    return _insert_preset(cleaned, Layer("preset", None, overrides), source)


def load_effective(
    repo_root: Path, explicit_path: Path | None = None, *, preset: str = ""
) -> EffectiveConfig:
    """Merge + validate all layers and record per-leaf provenance; a named
    `preset` is injected per :func:`_apply_preset`."""
    layers = discover_layers(repo_root, explicit_path)
    presets = preset_names(layers)
    layers = _apply_preset(layers, preset)
    return _effective_from_layers(layers, source="(merged config layers)", presets=presets)


def load_global_only() -> EffectiveConfig:
    """Defaults plus the global config, with no repo layer and no preset
    applied: what `agent6 config fill` materializes.

    A fill writes the GLOBAL file, so baking the repo layer into it would
    follow the operator to every other repo, and baking a selected preset's
    effects would freeze them as explicit values while the selector -- which
    keeps applying at runtime -- is what the operator edits.
    """
    gpath = global_config_path()
    layers = [Layer("global", gpath, _read_toml(gpath))] if gpath.is_file() else []
    cleaned, _ = _strip_presets(layers)
    return _effective_from_layers(cleaned, source="(global config)", presets=preset_names(layers))


def load_effective_with_overlay(
    repo_root: Path, overlay: dict[str, Any], *, explicit_path: Path | None = None
) -> EffectiveConfig:
    """Like :func:`load_effective` but with *overlay* as the highest layer.

    Used by `agent6 machine run` to apply a machine file's `[config]`
    table on top of the repo/global/default layers. The overlay is merged
    and validated exactly like a config file; its leaves are labelled
    `machine` in the provenance map (`config show` style).

    `explicit_path` is the global `--config FILE` layer, which sits under
    the overlay like any other config file.
    """
    layers = discover_layers(repo_root, explicit_path)
    if overlay:
        _forbid_layer_preset("machine overlay", overlay)
        layers = [*layers, Layer("machine", None, overlay)]
    # Apply the selected preset (and strip [presets] tables) just like
    # load_effective, so a user's [presets.<name>] + the top-level `preset` work
    # under `machine run` / `config --machine` instead of failing validation.
    presets = preset_names(layers)
    layers = _apply_preset(layers, "")
    return _effective_from_layers(
        layers, source="(merged config layers + machine overlay)", presets=presets
    )


@dataclass(frozen=True, slots=True)
class InvalidEntry:
    """One invalid config leaf that `config fix` can drop, and where it lives."""

    leaf: str  # dotted config leaf, e.g. "prompt.decompose" (or a table name, e.g. "cli")
    value: Any  # the offending value, read back from the file
    layer: LayerName  # "global" | "repo" | "machine"
    path: Path  # the file to edit
    file_key: str  # dotted key WITHIN that file (leaf, or "config."+leaf for a machine overlay)
    is_table: bool = False  # True when the whole [leaf] table must be dropped, not one leaf


@dataclass(frozen=True, slots=True)
class ConfigDiagnosis:
    """The result of diagnosing the on-disk config for `config fix`.

    Empty `removable` + `None` `blocked` means the config is valid.
    """

    removable: tuple[InvalidEntry, ...]  # invalid leaves that map to a file (droppable)
    blocked: str | None  # invalid in a way fix can't auto-drop (a message), or None


def _fix_scope_layers(repo_root: Path, machine: Path | None) -> list[Layer]:
    """The layers `config fix` repairs: global + repo, or a machine's [config]
    overlay on top of them when *machine* is given. Presets are applied (and
    [presets] tables stripped) exactly as load_effective does, so validation and
    provenance match a real load."""
    layers = discover_layers(repo_root, None)
    if machine is not None:
        overlay = read_toml_file(machine).get("config", {})
        if isinstance(overlay, dict) and overlay:
            _forbid_layer_preset("machine overlay", overlay)
            layers = [*layers, Layer("machine", machine, overlay)]
    return _apply_preset(layers, "")


def _merge_with_origin(layers: list[Layer]) -> tuple[dict[str, Any], dict[str, Layer]]:
    """Deep-merge *layers* low->high and map each dotted leaf to the Layer that set
    it (the highest one), so an invalid leaf names its own file."""
    merged, sources = _merge_layers(layers)
    by_name = {layer.name: layer for layer in layers}
    return merged, {leaf: by_name[name] for leaf, name in sources.items() if name in by_name}


def _removable_for(loc: str, origin: dict[str, Layer]) -> tuple[str, Layer, bool] | None:
    """The `(file_key, layer, is_table)` to drop for a validation error at *loc*,
    or None when no config file is at fault (a built-in default). Handles three
    shapes: *loc* IS a file leaf; *loc* is UNDER a file leaf (walk down to the
    longest present prefix); and *loc* is an ANCESTOR table of file leaves -- an
    unknown/extra whole table reported at the table (e.g. a leftover `[cli]` is
    reported as `extra_forbidden` at `cli` while the file holds `cli.input`),
    which must be dropped whole."""
    parts = loc.split(".") if loc else []
    for i in range(len(parts), 0, -1):
        cand = ".".join(parts[:i])
        if cand in origin:
            return cand, origin[cand], False
    prefix = f"{loc}." if loc else ""
    child = next((k for k in origin if prefix and k.startswith(prefix)), None)
    if child is not None:
        return loc, origin[child], True  # an extra whole table -> drop the table
    return None


def _diagnose_errors(
    exc: ValidationError, origin: dict[str, Layer], *, only_layer: str | None
) -> ConfigDiagnosis:
    """Turn per-leaf validation errors into droppable entries + a blocked note for
    anything fix cannot drop (an error from a default/preset, or -- when scoped to
    a machine overlay -- an error that lives in the global/repo config instead)."""
    removable: list[InvalidEntry] = []
    blocked: list[str] = []
    seen: set[str] = set()
    for issue in exc.errors():
        loc = ".".join(str(part) for part in issue["loc"])
        note = f"  - {loc or '<root>'}: {issue['msg']}"
        match = _removable_for(loc, origin)
        if match is None:
            blocked.append(note)
            continue
        key, layer, is_table = match
        if key in seen:
            continue
        seen.add(key)
        if layer.path is None or (only_layer is not None and layer.name != only_layer):
            blocked.append(note)
            continue
        file_key = f"config.{key}" if layer.name == "machine" else key
        value = read_toml_leaf(read_toml_file(layer.path), file_key)
        removable.append(
            InvalidEntry(
                leaf=key,
                value=value,
                layer=layer.name,
                path=layer.path,
                file_key=file_key,
                is_table=is_table,
            )
        )
    return ConfigDiagnosis(tuple(removable), "\n".join(blocked) if blocked else None)


def find_invalid_entries(repo_root: Path, *, machine: Path | None = None) -> ConfigDiagnosis:
    """Diagnose the on-disk config for `agent6 config fix`.

    Returns the invalid leaves that can be dropped from a config FILE (each with its
    provenance), plus a `blocked` message when the config is invalid in a way fix
    cannot repair by dropping a leaf (an unknown preset name, a value only a
    built-in default/preset carries, unreadable TOML). Empty + None == valid.

    With *machine* set, only the machine file's `[config]` overlay entries are
    droppable; a global/repo problem surfaced by the merge is reported, not touched.
    """
    only = "machine" if machine is not None else None
    try:
        layers = _fix_scope_layers(repo_root, machine)
        merged, origin = _merge_with_origin(layers)
    except ConfigError as exc:
        return ConfigDiagnosis((), str(exc))
    try:
        Config.model_validate(merged)
    except ConfigError as exc:  # a model-level validator raised a standalone ConfigError
        return ConfigDiagnosis((), str(exc))
    except ValidationError as exc:
        return _diagnose_errors(exc, origin, only_layer=only)
    return ConfigDiagnosis((), None)


def leaf_keys(eff: EffectiveConfig) -> list[str]:
    """Every dotted leaf path in the effective config, sorted (for completion)."""
    return sorted(flatten_leaves(eff.config.model_dump(mode="python")))


def effective_leaf(eff: EffectiveConfig, dotted_key: str) -> tuple[Any, str] | None:
    """The `(value, source-layer)` for *dotted_key*, or None if it is not a leaf.

    Mirrors `config show`: the value comes from the merged+validated config and
    the source is the layer that set it (`default` when no layer did).
    """
    leaves = flatten_leaves(eff.config.model_dump(mode="python"))
    parts = dotted_key.split(".")
    if parts[0] == "presets" and len(parts) > 2 and ".".join(parts[2:]) in leaves:
        # A preset's leaf: the value the most specific layer's [presets.<name>]
        # table holds, or unset. `config get`, `set` and `unset` all address it,
        # so a preset write has an inverse.
        name, leaf = parts[1], ".".join(parts[2:])
        for layer in reversed(eff.layers):
            table = _file_presets(layer.path).get(name)
            if isinstance(table, dict) and leaf in flatten_leaves(table):
                return flatten_leaves(table)[leaf], f"preset {name} ({layer.name})"
        return None, "unset"
    if dotted_key not in leaves:
        return None
    return leaves[dotted_key], eff.sources.get(dotted_key, "default")


# ---------------------------------------------------------------------------
# Materialize: `config fill`
# ---------------------------------------------------------------------------


def _emit_table(path: str, data: dict[str, Any], lines: list[str]) -> None:
    """Emit one TOML table (and recurse into sub-tables / arrays of tables).

    `None` values are skipped, an unset optional field materializes as
    absent, i.e. "use the default".
    """
    scalars = {
        k: v
        for k, v in data.items()
        if v is not None and not isinstance(v, dict) and not _is_table_array(v)
    }
    subtables = {k: v for k, v in data.items() if isinstance(v, dict) and v}
    arraytables = {k: v for k, v in data.items() if _is_table_array(v)}
    # Emit a header for this path only when it carries scalar keys, or when
    # it is a genuine leaf table (no children at all). A pure parent table
    # like [providers] / [models] is left implicit so the emitter never prints an
    # empty header above its [providers.<name>] children.
    is_leaf = not subtables and not arraytables
    if scalars or is_leaf:
        lines.append(f"[{path}]")
        for key, value in scalars.items():
            lines.append(f"{toml_key(key)} = {format_toml_value(value)}")
        lines.append("")
    for key, sub in subtables.items():
        _emit_table(f"{path}.{toml_key(key)}" if path else toml_key(key), sub, lines)
    for key, arr in arraytables.items():
        for item in arr:
            lines.append(f"[[{path}.{toml_key(key)}]]" if path else f"[[{toml_key(key)}]]")
            for k2, v2 in item.items():
                if v2 is not None:
                    # Dicts render as inline tables via _toml_scalar; skipping
                    # them dropped an array item's nested objects.
                    lines.append(f"{toml_key(k2)} = {format_toml_value(v2)}")
            lines.append("")


def _is_table_array(value: Any) -> bool:
    return (
        isinstance(value, (list, tuple))
        and len(value) > 0
        and all(isinstance(v, dict) for v in value)
    )


def materialize(
    config: Config,
    *,
    keep_presets_from: Path | None = None,
    keep_preset_selector: bool = False,
) -> str:
    """Render the fully-resolved config as a complete TOML document.

    Used by `agent6 config fill` to snapshot every effective value into
    one explicit file (handy before tightening defaults or for an audit).

    `keep_presets_from` carries that file's own `[presets.*]` tables into the
    document. They are meta-config -- stripped before validation, so no `Config`
    holds them -- and a fill that rewrites the operator's config file would
    otherwise delete the definitions it cannot see.
    """
    data = config.model_dump(mode="python")
    # A `--config FILE` layer refuses a top-level `preset`, so a document
    # destined for one (a `--parallel` lane's snapshot) must not carry the
    # selector. `config fill` writes the GLOBAL config, which does accept it,
    # and there the selector is what the operator keeps editing -- dropping it
    # would silently deselect the preset and freeze today's values.
    data = data if keep_preset_selector else {k: v for k, v in data.items() if k != "preset"}
    lines: list[str] = [
        "# agent6 effective config, materialized by `agent6 config fill`.",
        "# Every value below is explicit; edit freely.",
        "",
    ]
    ordered = [s for s in SECTION_ORDER if s in data]
    ordered += [s for s in data if s not in SECTION_ORDER]
    # Top-level scalar fields (e.g. `preset`) carry no table header and must
    # precede every `[section]` in TOML, so emit them first as bare key=value.
    for section in ordered:
        value = data[section]
        if value is not None and not isinstance(value, dict) and not _is_table_array(value):
            lines.append(f"{section} = {format_toml_value(value)}")
    if lines[-1] != "":
        lines.append("")
    for section in ordered:
        value = data[section]
        if isinstance(value, dict):
            if not value:
                continue
            _emit_table(section, value, lines)
        elif _is_table_array(value):
            for item in value:
                lines.append(f"[[{section}]]")
                for k2, v2 in item.items():
                    if v2 is not None:
                        lines.append(f"{toml_key(k2)} = {format_toml_value(v2)}")
                lines.append("")
    if kept := _file_presets(keep_presets_from):
        _emit_table("presets", kept, lines)
    return "\n".join(lines).rstrip("\n") + "\n"


def _file_presets(path: Path | None) -> dict[str, Any]:
    """The `[presets.*]` tables *path* defines itself, if any."""
    if path is None or not path.is_file():
        return {}
    presets = _read_toml(path).get("presets")
    return presets if isinstance(presets, dict) else {}
