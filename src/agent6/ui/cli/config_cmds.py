# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""`agent6 config` subcommands (show/fill/fix/path/presets/get/set/unset/add/remove)."""

from __future__ import annotations

import math
import sys
import tempfile
from pathlib import Path

from agent6.config import (
    ConfigError,
)
from agent6.config.io import (
    parse_cli_value,
    read_toml_file,
    read_toml_leaf,
    remove_toml_leaf,
    remove_toml_table,
    upsert_toml_leaf,
)
from agent6.config.layer import (
    InvalidEntry,
    effective_leaf,
    find_invalid_entries,
    flatten_leaves,
    load_effective,
    load_effective_with_overlay,
    load_global_only,
    materialize,
    preset_catalog,
    repo_config_path_for,
    resolved_state_dir,
)
from agent6.config.write import (
    keep_or_rollback,
    merged_config_error,
    resolved_write_path,
    revalidate_write,
    set_config_value,
    unset_config_value,
    writing_config,
)
from agent6.errors import OperatorError, read_operator_file
from agent6.machine import (
    PROTECTED_OVERLAY_LEAVES,
    PROTECTED_OVERLAY_TABLES,
    MachineError,
    load_machine,
)
from agent6.models import registry as models_registry
from agent6.paths import (
    cache_dir,
    chown_to_real_user,
    data_dir,
    effective_user,
    global_config_path,
    mkdir_for_real_user,
    secrets_path,
    state_base,
)
from agent6.portable import atomic_write, locked_file
from agent6.viewmodel.config_view import format_value, render_key_detail, render_show


def _cmd_config_show(
    config_path: Path | None,
    *,
    as_json: bool,
    keys: list[str] | None = None,
    descriptions: bool = False,
) -> int:
    eff = load_effective(Path.cwd(), config_path)
    resolved = models_registry.resolved_adaptive_values(eff.config)
    if keys:
        # `config show <key>...`: leaves or whole section prefixes, untruncated
        # (JSON mode filters to the same match set). The detail view always
        # carries the meaning: a deliberately-asked-about key is the one place
        # a description can never bury the values.
        try:
            detail = render_key_detail(
                eff, keys, resolved=resolved, color=sys.stdout.isatty(), as_json=as_json
            )
        except KeyError as exc:
            print(
                f"ERROR: no config key matches {exc.args[0]!r} (see `agent6 config show`).",
                file=sys.stderr,
            )
            return 2
        print(detail, end="")
        return 0
    text = render_show(
        eff,
        as_json=as_json,
        resolved=resolved,
        color=sys.stdout.isatty(),
        descriptions=descriptions,
    )
    print(text, end="")
    return 0


def _cmd_config_path() -> int:
    """Every file and directory agent6 reads or writes, resolved.

    The four XDG bases each hold a different kind of thing and each has its
    own override, so "where did agent6 put that" was four lookups across the
    docs. One command answers it; `agent6 --help` carries the short form."""
    user = effective_user()
    rows: list[tuple[str, Path, bool]] = [
        ("global config", global_config_path(user), True),
        ("repo config", repo_config_path_for(Path.cwd()), True),
        ("secrets", secrets_path(user), True),
        ("config dir", global_config_path(user).parent, False),
        ("state (all repos)", state_base(user), False),
        ("state (this repo)", resolved_state_dir(Path.cwd()), False),
        ("skills", data_dir(user) / "skills", False),
        ("cache", cache_dir(user), False),
    ]
    width = max(len(label) for label, _p, _f in rows)
    for label, p, is_file in rows:
        present = p.is_file() if is_file else p.is_dir()
        print(f"{label:<{width}}: {p}{'' if present else '  (not present)'}")
    return 0


def _cmd_config_presets(config_path: Path | None = None) -> int:
    """List every known preset with the overrides it applies; mark the selection.

    Honours the global `--config` like every other config subcommand, so a
    `[presets.*]` table in an explicit file is listed by the command that
    exists to show which presets are available."""
    cat = preset_catalog(Path.cwd(), config_path)
    if cat.selected:
        print(f"preset = {cat.selected}  [{cat.source}]")
    else:
        print("no preset selected (plain defaults)")
    for info in cat.presets:
        tag = "built-in" if info.origin == "built-in" else f"user, {info.origin} config"
        if info.replaces_builtin:
            tag += ", replaces the built-in"
        sel = "  (selected)" if info.name == cat.selected else ""
        print(f"\n{info.name}  [{tag}]{sel}")
        leaves = flatten_leaves(info.overrides)
        if not leaves:
            print("  (plain defaults, no overrides)")
        for key, val in leaves.items():
            print(f"  {key} = {format_value(val)}")
    print(
        "\nSelect per run with --preset <name>;"
        " persist with `agent6 config set preset <name>` (--repo for this repo)."
    )
    return 0


def _open_target(target: Path) -> None:
    """Create the config dir and hand it straight back to the real operator.

    Under sudo the dir is created as root; handing it back only after a
    successful write left a root-owned dir behind on every refusal.
    """
    mkdir_for_real_user(target.parent)


def _cmd_config_fill(*, force: bool) -> int:
    """Materialize defaults plus global into the global config file, never
    the repo layer and never a preset's effects."""
    target = resolved_write_path(global_config_path())
    _open_target(target)
    # Load the effective config, existence-check, and publish all under the
    # target's lock and via atomic_write: reading the merged config BEFORE the
    # lock let a concurrent `config set` land in between, and the plain
    # write_text then overwrote it with the stale snapshot (lost update) and
    # could tear on a crash.
    with writing_config(target):
        eff = load_global_only()
        if target.is_file() and not force:
            print(
                f"ERROR: {target} already exists. Re-run with --force to overwrite.",
                file=sys.stderr,
            )
            return 2
        atomic_write(
            target,
            materialize(eff.config, keep_presets_from=target, keep_preset_selector=True),
        )
    print(f"Wrote fully-resolved config to {target}")
    return 0


def _config_write_target(*, repo: bool, machine: Path | None) -> tuple[Path, str]:
    """Resolve the file + dotted-key prefix a config write should target.

    Global by default; `--repo` writes the in-repo config; `--machine-file FILE`
    edits that machine's `[config]` overlay (so keys are prefixed `config.`
    and land in `[config.<section>]`). `--repo` and `--machine-file`
    together are ambiguous and rejected.
    """
    if machine is not None:
        if repo:
            raise OperatorError("use either --repo or --machine-file, not both")
        return machine, "config."
    if repo:
        return resolved_write_path(repo_config_path_for(Path.cwd())), ""
    return resolved_write_path(global_config_path()), ""


def _reject_machine_protected(key: str, machine: Path | None) -> str | None:
    """Error string if *key* is operator-only in a machine overlay, else None.

    Reads the same PROTECTED_OVERLAY_* sets the MachineSpec validator enforces,
    so this refuses exactly what the loader would: keeping a second, shorter
    copy here meant `config set --machine-file` wrote presets.*,
    machine.notify.*, and git.run_repo_hooks into overlays that can never load.
    """
    if machine is None:
        return None
    for table in PROTECTED_OVERLAY_TABLES:
        if key == table or key.startswith(f"{table}."):
            return (
                f"machine [config] overlays must not set {table}.*:"
                " connections/secrets, sandbox policy, and strategy presets are"
                " operator-only (global/repo config)"
            )
    for dotted, why in PROTECTED_OVERLAY_LEAVES.items():
        if key == dotted or key.startswith(f"{dotted}."):
            return f"machine [config] overlays must not set {dotted}: {why} (operator-only)"
    return None


def _machine_is_valid(text: str | None) -> bool:
    """True iff *text* parses as a complete, valid machine spec.

    Used to decide whether a `config set --machine-file` edit BROKE a working machine
    (block + roll back) versus merely touched an already-incomplete one (allow).
    """
    if text is None:
        return False
    with tempfile.NamedTemporaryFile("w", suffix=".asm.toml", delete=True, encoding="utf-8") as tf:
        tf.write(text)
        tf.flush()
        try:
            load_machine(Path(tf.name))
        except MachineError:
            return False
    return True


def _leaf_problems(text: str) -> str:
    """The `leaf: message` lines of a config validation error, one per line:
    the shape every other config writer prints (no header, no error type)."""
    leaves = [
        ln.strip()[2:].split(" (type=", 1)[0]
        for ln in text.splitlines()
        if ln.strip().startswith("- ")
    ]
    return "\n".join(leaves) if leaves else text


def _revalidate_machine(target: Path, prior_text: str | None, *, held: bool = True) -> str | None:
    """Re-validate a machine file after a `[config]`-overlay write; restore
    *prior_text* on failure (kept, saying so, when the lock failed open).

    Validates the overlay against the config stack, and the WHOLE machine spec
    when the file has `[states]` -- `config set --machine-file` must not BREAK
    a runnable machine. Blocks only when the edit made a previously-VALID
    machine invalid; one already invalid (or a brand-new stub) is left for the
    author to finish. The layered (global/repo) writers revalidate through
    `config.write` instead.
    """
    err: str | None = None
    try:
        data = read_toml_file(target)
        overlay = data.get("config", {})
        load_effective_with_overlay(Path.cwd(), overlay if isinstance(overlay, dict) else {})
        if "states" in data and _machine_is_valid(prior_text):
            load_machine(target)
    except ConfigError as exc:
        err = _leaf_problems(str(exc))
    except MachineError as exc:
        err = "; ".join(exc.problems)
    if err is None:
        return None
    return keep_or_rollback(target, prior_text, err, held=held)


def _warn_if_still_broken() -> None:
    """After a kept layered write: the config loads, or another layer still
    breaks it and the operator should hear which one, not a false success."""
    if (after := merged_config_error(Path.cwd())) is not None:
        print(
            "WARNING: the config is still invalid because of a value in another layer;"
            f" fix that one on its own:\n{after}",
            file=sys.stderr,
        )


def _cmd_config_get(config_path: Path | None, key: str, *, machine: Path | None) -> int:
    """Print a leaf's effective value + the layer that set it."""
    if machine is not None and not machine.is_file():
        # read_toml_file answers {} for a missing path, so a typo'd machine file
        # would read as "an empty overlay" and answer confidently from the
        # stack below it.
        raise OperatorError(f"no such machine file: {machine}")
    if machine is not None:
        overlay = read_toml_file(machine).get("config", {})
        eff = load_effective_with_overlay(
            Path.cwd(),
            overlay if isinstance(overlay, dict) else {},
            explicit_path=config_path,
        )
    else:
        eff = load_effective(Path.cwd(), config_path)
    found = effective_leaf(eff, key)
    if found is None:
        print(f"ERROR: {key!r} is not a config leaf (see `agent6 config show`).", file=sys.stderr)
        return 2
    value, source = found
    print(f"{key} = {format_value(value)}  [{source}]")
    return 0


def _flag_shadow_note(key: str, config_path: Path | None) -> str | None:
    """A note when an active `--config FILE` sets *key*: the write above landed
    in a lower layer, so invocations carrying the flag keep seeing the file's
    value and the edit reads as ineffective without this line."""
    if config_path is None:
        return None
    try:
        eff = load_effective(Path.cwd(), config_path)
    except ConfigError:
        return None
    if eff.sources.get(key) != "flag":
        return None
    return f"note: --config {config_path} overrides {key} while that flag is used."


def _cmd_config_set(
    key: str, value: str, *, repo: bool, machine: Path | None, config_path: Path | None = None
) -> int:
    """Set a scalar leaf in the target file (global / repo / machine overlay)."""
    if err := _reject_machine_protected(key, machine):
        print(f"ERROR: {err}", file=sys.stderr)
        return 2
    target, prefix = _config_write_target(repo=repo, machine=machine)
    parsed = parse_cli_value(value)
    if machine is None:
        err = set_config_value(Path.cwd(), key, value, to_repo=repo)
    else:
        _open_target(target)
        with writing_config(target) as held:
            prior = read_operator_file(target) if target.is_file() else None
            read_toml_file(target)  # refuse line surgery on a file that does not parse
            upsert_toml_leaf(target, prefix + key, parsed)
            err = _revalidate_machine(target, prior, held=held)
    if err:
        print(f"ERROR: {err}", file=sys.stderr)
        return 2
    if machine is None:
        _warn_if_still_broken()
    print(f"Set {key} = {format_value(parsed)} in {target}")
    if machine is None and (note := _flag_shadow_note(key, config_path)):
        print(note)
    return 0


def _cmd_config_unset(
    key: str, *, repo: bool, machine: Path | None, config_path: Path | None = None
) -> int:
    """Remove a leaf so it reverts to the next-lower layer / built-in default."""
    if err := _reject_machine_protected(key, machine):
        print(f"ERROR: {err}", file=sys.stderr)
        return 2
    if effective_leaf(load_effective(Path.cwd(), config_path), key) is None:
        print(f"ERROR: {key!r} is not a config leaf (see `agent6 config show`).", file=sys.stderr)
        return 2
    target, prefix = _config_write_target(repo=repo, machine=machine)
    if not target.is_file():
        print(f"ERROR: {target} does not exist; nothing to unset.", file=sys.stderr)
        return 2
    if machine is None:
        res = unset_config_value(Path.cwd(), key, to_repo=repo)
        removed, err = res.removed, res.error
    else:
        with writing_config(target) as held:
            prior = read_operator_file(target)
            read_toml_file(target)  # refuse line surgery on a file that does not parse
            removed = remove_toml_leaf(target, prefix + key)
            err = _revalidate_machine(target, prior, held=held) if removed else None
    if err:
        print(f"ERROR: unsetting {key} left an invalid config:\n{err}", file=sys.stderr)
        return 2
    if not removed:
        print(f"{key} is not set in {target}; nothing to unset.")
        return 0
    if machine is None:
        _warn_if_still_broken()
    print(f"Unset {key} in {target}")
    if machine is None and (note := _flag_shadow_note(key, config_path)):
        print(note)
    return 0


def _schema_says_not_a_list(key: str) -> bool:
    """True when the config schema knows *key* and its value is not a list.

    Guards `config add/remove` on keys the target file does not set yet: the
    effective (defaults-included) value reveals the leaf's shape, so a scalar
    like sandbox.network fails with "not a list field" instead of a
    contradictory revalidation error. Unknown keys and unloadable configs
    return False; revalidation still rejects those."""
    try:
        eff = load_effective(Path.cwd(), None)
    except ConfigError:
        return False
    leaf = effective_leaf(eff, key)
    # List leaves surface as list or tuple depending on the field's type. A
    # None effective value is an UNSET optional field (e.g. the list-valued
    # providers.*.token_command, default None); it doesn't prove the leaf is a
    # scalar, so fall through and let revalidation reject a genuine scalar.
    return leaf is not None and leaf[0] is not None and not isinstance(leaf[0], (list, tuple))


def _config_list_edit(key: str, value: str, *, repo: bool, machine: Path | None, add: bool) -> int:
    """Shared body for `config add` / `config remove` on a list field."""
    if err := _reject_machine_protected(key, machine):
        print(f"ERROR: {err}", file=sys.stderr)
        return 2
    target, prefix = _config_write_target(repo=repo, machine=machine)
    _open_target(target)
    # The lock spans from the current-items read: the list RMW starts there,
    # and two concurrent adds otherwise both read the same base list and the
    # later publish drops the earlier element.
    with writing_config(target) as held:
        current = read_toml_leaf(read_toml_file(target), prefix + key)
        if current is None:
            if _schema_says_not_a_list(key):
                print(f"ERROR: {key} is not a list field.", file=sys.stderr)
                return 2
            current = []
        if not isinstance(current, list):
            print(f"ERROR: {key} is not a list field in {target}.", file=sys.stderr)
            return 2
        parsed = parse_cli_value(value)
        items = list(current)
        if (parsed in items) == add:
            print(f"{format_value(parsed)} {'already' if add else 'not'} in {key}.")
            return 0
        items = [*items, parsed] if add else [x for x in items if x != parsed]
        prior = read_operator_file(target) if target.is_file() else None
        was_valid = machine is None and merged_config_error(Path.cwd()) is None
        upsert_toml_leaf(target, prefix + key, items)
        if machine is None:
            err = revalidate_write(
                Path.cwd(),
                target,
                prior,
                was_valid=was_valid,
                held=held,
                written=[(key, items)],
            )
        else:
            err = _revalidate_machine(target, prior, held=held)
        if err:
            print(f"ERROR: {value!r} is not valid for {key}:\n{err}", file=sys.stderr)
            return 2
    if machine is None:
        _warn_if_still_broken()
    verb, prep = ("Added", "to") if add else ("Removed", "from")
    print(f"{verb} {format_value(parsed)} {prep} {key} in {target}")
    return 0


def _cmd_config_add(key: str, value: str, *, repo: bool, machine: Path | None) -> int:
    return _config_list_edit(key, value, repo=repo, machine=machine, add=True)


def _cmd_config_remove(key: str, value: str, *, repo: bool, machine: Path | None) -> int:
    return _config_list_edit(key, value, repo=repo, machine=machine, add=False)


def _entry_is_stale(entry: InvalidEntry) -> bool:
    """Whether *entry*'s key no longer holds the value diagnosis read.

    `find_invalid_entries` reads unlocked and removal deletes by key NAME, so a
    concurrent `config set` that replaced this key with a valid value in between
    would have it deleted -- after that writer was told it had been saved.
    """
    try:
        data = read_toml_file(entry.path)
    except ConfigError:
        return True  # unreadable now: leave it to the loud paths
    # nan != nan by identity, so a still-present nan (scalar OR nested in a
    # table/list) otherwise reads "replaced by a concurrent writer" on every
    # pass and can never be removed -- `config fix` then loops to "changed under
    # the lock" and the entry is unfixable forever. Compare NaN-tolerantly at
    # every nesting depth.
    return not _equal_tolerating_nan(read_toml_leaf(data, entry.file_key), entry.value)


def _equal_tolerating_nan(a: object, b: object) -> bool:
    """Structural equality that treats NaN == NaN (float NaN is the only value
    unequal to itself), recursing through dict/list so a nested NaN matches."""
    if isinstance(a, float) and isinstance(b, float):
        return a == b or (math.isnan(a) and math.isnan(b))
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_equal_tolerating_nan(a[k], b[k]) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(
            _equal_tolerating_nan(x, y) for x, y in zip(a, b, strict=True)
        )
    return a == b


def _cmd_config_fix(*, machine: Path | None) -> int:
    """Drop every invalid entry from the config, printing what it was and where it
    lived (global / repo, or a machine's [config] overlay with --machine-file).

    Removing one entry can reveal another it shadowed, so it re-diagnoses until the
    config is clean or nothing droppable remains. An entry it cannot drop -- not a
    plain leaf (non-absolute state_dir, bad built-in default), or a TOML shape the
    line surgery cannot match (a dotted top-level key has no [table] header) -- is
    reported, never counted as removed.
    """
    repo_root = Path.cwd()
    removed: list[InvalidEntry] = []
    stuck: list[InvalidEntry] = []
    touched: set[Path] = set()
    diag = find_invalid_entries(repo_root, machine=machine)
    while diag.removable:
        progressed = False
        for entry in diag.removable:
            # The surgery publishes by rename, so it edits the file the layer
            # RESOLVES to; writing the link's own name would replace it.
            target = resolved_write_path(entry.path)
            # Re-check under the file's lock: diagnosis ran unlocked, so a
            # concurrent writer may have fixed this key since.
            with locked_file(target):
                if _entry_is_stale(entry):
                    continue
                try:
                    ok = (
                        remove_toml_table(target, entry.file_key)
                        if entry.is_table
                        else remove_toml_leaf(target, entry.file_key)
                    )
                except ConfigError:
                    # A leaf inside an inline table / dotted key: the surgery
                    # cannot carve it out, so it is stuck (fix has always
                    # reported this shape as stuck; the owner now refuses
                    # loudly instead of returning "not found").
                    ok = False
            if not ok:
                stuck.append(entry)
                continue
            progressed = True
            touched.add(target)
            removed.append(entry)
        if not progressed:
            # Nothing this pass could actually delete: halt honestly instead of
            # re-diagnosing the identical set forever (and lying "fixed").
            break
        stuck = []
        diag = find_invalid_entries(repo_root, machine=machine)
    for path in touched:
        chown_to_real_user(path)
    for entry in removed:
        what = (
            f"[{entry.leaf}] (whole table)"
            if entry.is_table
            else f"{entry.leaf} = {format_value(entry.value)}"
        )
        print(f"Removed {what}  [{entry.layer}: {entry.path}]")
    if diag.blocked:
        print(
            "ERROR: config still invalid (not an auto-removable entry); fix it by hand:\n"
            f"{diag.blocked}",
            file=sys.stderr,
        )
        return 2
    if stuck:
        names = "\n".join(
            f"  {e.leaf} = {format_value(e.value)}  [{e.layer}: {e.path}]" for e in stuck
        )
        print(
            "ERROR: config still invalid (flagged entries could not be auto-removed);"
            f" fix by hand:\n{names}",
            file=sys.stderr,
        )
        return 2
    # Measure before claiming: a no-progress break lands here with entries in
    # NO bucket (each read stale under the lock), and "valid"/"fixed" printed
    # unmeasured over a config every next command still refuses.
    final = find_invalid_entries(repo_root, machine=machine)
    if final.removable or final.blocked:
        print(
            "ERROR: config still invalid (flagged entries changed under the lock"
            " this pass); re-run `agent6 config fix` or fix by hand.",
            file=sys.stderr,
        )
        return 2
    if not removed:
        print("Config is valid; nothing to fix.")
        return 0
    n = len(removed)
    print(f"Fixed the config: dropped {n} invalid entr{'y' if n == 1 else 'ies'}.")
    return 0
