# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Machine script-bundle validation (security-critical).

A machine is a `.asm.toml` plus a sibling `scripts/` directory. Every entry
under `scripts/` must resolve to a path INSIDE the bundle (rejects symlinks
that escape via `..`/absolute), and every static tool-command element that
references a bundled script must exist and stay inside the bundle. ``machine
check`/`test` run this offline; `machine run`/`create`` run it again before
any execution, so a `scripts/` symlink escaping the bundle can never be read
by a tool on an isolation level that cannot RO-bind the bundle.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from agent6.machine import MachineError, MachineSpec, ToolState, load_machine, validate_semantics


def _bundle_script_ref(element: str) -> str | None:
    """Return the relative script path a static command element names, else None.

    A bundle script reference is a relative path whose first component is
    `scripts` (e.g. `scripts/fetch.sh` or `./scripts/fetch.sh`). Absolute
    paths (`/usr/bin/bash`) are interpreter/binary paths, not bundle refs.
    """
    cleaned = element[2:] if element.startswith("./") else element
    if not cleaned or cleaned.startswith("/"):
        return None
    parts = Path(cleaned).parts
    if parts and parts[0] == "scripts":
        return cleaned
    return None


def _check_scripts_dir(scripts_dir: Path, bundle: Path) -> list[str]:
    """Every entry under `scripts/` must resolve to a path inside the bundle."""
    if not scripts_dir.is_dir():
        return ["bundle 'scripts' exists but is not a directory"]
    problems: list[str] = []
    for entry in sorted(scripts_dir.rglob("*")):
        rel = entry.relative_to(scripts_dir)
        try:
            resolved = entry.resolve()
            # Python 3.14's resolve() stopped raising on a symlink loop (it
            # returns the path); stat(), which follows links, still raises
            # ELOOP, so a circular/broken symlink is reported instead of
            # silently accepted as an in-bundle path.
            entry.stat()
        except (OSError, RuntimeError) as exc:  # RuntimeError: circular symlink (<3.14)
            problems.append(f"scripts/{rel}: {exc}")
            continue
        if not resolved.is_relative_to(bundle):
            problems.append(f"scripts/{rel} resolves outside the bundle ({resolved}); refused")
    return problems


def _check_command_scripts(name: str, state: ToolState, bundle: Path) -> list[str]:
    """Static tool-command script references must exist and stay in the bundle."""
    problems: list[str] = []
    for element in state.command:
        if "{{" in element:
            continue  # templated; cannot resolve statically
        ref = _bundle_script_ref(element)
        if ref is None:
            continue
        target = bundle / ref
        try:
            resolved = target.resolve()
        except (OSError, RuntimeError) as exc:  # RuntimeError: circular symlink
            problems.append(f"state {name!r}: script {element!r}: {exc}")
            continue
        if not resolved.is_relative_to(bundle):
            problems.append(f"state {name!r}: script {element!r} escapes the bundle")
        elif not target.exists():
            problems.append(f"state {name!r}: script {element!r} not found in bundle")
    return problems


def validate_bundle(spec: MachineSpec, machine_path: Path) -> list[str]:
    """Validate a machine's script bundle (the `.asm.toml` + a sibling `scripts/`).

    Security-critical: every entry under `scripts/` must resolve to a path
    INSIDE the bundle (rejects symlinks that escape via `..`/absolute), and
    every static tool-command element that references a bundled script must
    exist and stay inside the bundle. Dynamic (templated) command elements are
    skipped, they cannot be resolved without a blackboard.
    """
    try:
        bundle = machine_path.parent.resolve()
    except OSError as exc:
        return [f"cannot resolve bundle directory for {machine_path}: {exc}"]
    problems: list[str] = []
    scripts_dir = bundle / "scripts"
    if scripts_dir.exists():
        problems.extend(_check_scripts_dir(scripts_dir, bundle))
    for name, state in spec.states.items():
        if isinstance(state, ToolState):
            problems.extend(_check_command_scripts(name, state, bundle))
    return problems


@dataclass(frozen=True, slots=True)
class MachineFileSummary:
    """One authored-file row for a machines listing."""

    name: str  # the declared `machine` name; "-" when the file does not parse
    states: str  # the state count; "-" when the file does not parse
    spec: str  # "valid", "N issue(s)", or "invalid" (does not parse)


def summarize_machine_file(path: Path) -> MachineFileSummary:
    """Whether the FILE checks out ("valid", never "ok": that word is a
    machine-run terminal status): parsed, then semantics + bundle, exactly what
    `machine check`/`run` refuse. A file that does not parse has no name to
    show (its declared `machine` is unknown), so the row keeps its path only."""
    try:
        spec = load_machine(path)
    except (MachineError, OSError):
        return MachineFileSummary("-", "-", "invalid")
    problems = validate_semantics(spec) + validate_bundle(spec, path)
    verdict = "valid" if not problems else f"{len(problems)} issue(s)"
    return MachineFileSummary(spec.machine, str(len(spec.states)), verdict)
