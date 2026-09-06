# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Skill discovery: the SKILL.md format (agentskills.io) read from operator dirs.

A skill is a directory holding a `SKILL.md` whose YAML frontmatter carries
`name` and `description`. This package is a pure leaf: it scans
operator-chosen directories, parses the two required fields, and applies the
operator's per-skill state map. Where the content goes (system-prompt index,
`use_skill` tool, `--skill` flag) is the consumers' business.

The frontmatter parser is deliberately minimal, not a YAML implementation: it
covers the scalar, quoted, folded (`>`) and literal (`|`) forms the two
required fields use in the wild. Unknown keys are surfaced and ignored so
ecosystem skills with extra fields load fine; anything unparseable is a
warning, never a crash.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

# letters/digits/hyphens, starting alphanumeric (agentskills.io name rule)
_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9-]*$")
_KEY_RE = re.compile(r"^([A-Za-z0-9_-]+):\s*(.*)$")


def is_valid_skill_name(name: str) -> bool:
    """True iff *name* is a valid skill name: alphanumeric-plus-hyphen, so it is
    also a single safe path component. Discovery and install both gate on this;
    an install path built from an unvalidated name would let `../` or an absolute
    path in untrusted SKILL.md frontmatter escape the skills dir."""
    return bool(_NAME_RE.match(name))


@dataclass(frozen=True, slots=True)
class Skill:
    """One discovered skill: identity plus the full SKILL.md text."""

    name: str
    description: str
    dir: Path
    text: str


@dataclass(frozen=True, slots=True)
class ResolvedSkills:
    """Discovery output after the operator's state map is applied.

    `enabled` feeds the system-prompt index (on-demand loading);
    `always` skills get their full text injected instead. A skill is in
    at most one of the two.
    """

    enabled: tuple[Skill, ...]
    always: tuple[Skill, ...]
    warnings: tuple[str, ...]


def parse_frontmatter(text: str) -> tuple[dict[str, str], list[str]]:
    """Parse a SKILL.md's leading `---` frontmatter block.

    Returns (fields, warnings). Missing or unclosed frontmatter yields no
    fields and a warning; the caller decides whether that disqualifies the
    file.
    """
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}, ["no frontmatter block (file must start with ---)"]
    try:
        end = next(i for i, ln in enumerate(lines[1:], start=1) if ln.strip() == "---")
    except StopIteration:
        return {}, ["unclosed frontmatter block (no closing ---)"]

    fields: dict[str, str] = {}
    warnings: list[str] = []
    i = 1
    while i < end:
        line = lines[i]
        if not line.strip() or line.lstrip().startswith("#"):
            i += 1
            continue
        m = _KEY_RE.match(line)
        if m is None:
            warnings.append(f"unparseable frontmatter line {i + 1}: {line.strip()!r}")
            i += 1
            continue
        key, value = m.group(1), m.group(2).strip()
        if value in (">", ">-", "|", "|-"):
            block: list[str] = []
            i += 1
            while i < end and (not lines[i].strip() or lines[i].startswith((" ", "\t"))):
                block.append(lines[i].strip())
                i += 1
            joiner = " " if value.startswith(">") else "\n"
            fields[key] = joiner.join(b for b in block if b).strip()
            continue
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        fields[key] = value
        i += 1
    return fields, warnings


def _load_skill(skill_dir: Path) -> tuple[Skill | None, list[str]]:
    path = skill_dir / "SKILL.md"
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        # Discovery runs at startup, so one unreadable or non-UTF-8 SKILL.md
        # took down EVERY run with a bare decode error naming no file. Degrade
        # to the warning every other malformed skill gets.
        return None, [f"{path}: unreadable ({exc})"]
    fields, warnings = parse_frontmatter(text)
    name = fields.get("name", "")
    description = fields.get("description", "")
    if not name or not description:
        return None, [f"{path}: missing required frontmatter name/description", *warnings]
    if not is_valid_skill_name(name):
        return None, [f"{path}: invalid skill name {name!r}", *warnings]
    prefixed = [f"{path}: {w}" for w in warnings]
    if name != skill_dir.name:
        prefixed.append(f"{path}: frontmatter name {name!r} != directory {skill_dir.name!r}")
    return Skill(name=name, description=description, dir=skill_dir, text=text), prefixed


def discover_skills(dirs: Sequence[Path]) -> tuple[tuple[Skill, ...], tuple[str, ...]]:
    """Scan directories for skills, in precedence order (first dir wins dupes).

    Each directory may hold skill subdirectories (`<dir>/<name>/SKILL.md`)
    or be a single skill itself (`<dir>/SKILL.md`). Dotted entries are
    ignored. Missing directories are fine (nothing installed yet).
    """
    found: dict[str, Skill] = {}
    warnings: list[str] = []
    for base in dirs:
        if not base.is_dir():
            continue
        if (base / "SKILL.md").is_file():
            candidates = [base]
        else:
            candidates = sorted(
                p
                for p in base.iterdir()
                if p.is_dir() and not p.name.startswith(".") and (p / "SKILL.md").is_file()
            )
        for skill_dir in candidates:
            skill, warns = _load_skill(skill_dir)
            warnings.extend(warns)
            if skill is None:
                continue
            if skill.name in found:
                warnings.append(
                    f"{skill_dir}: duplicate skill name {skill.name!r}"
                    f" (keeping {found[skill.name].dir})"
                )
                continue
            found[skill.name] = skill
    return tuple(found.values()), tuple(warnings)


def skill_search_dirs(extra_dirs: Sequence[str], installed_dir: Path) -> tuple[Path, ...]:
    """Search order: `extra_dirs` first, so a local checkout under active
    development wins over an installed copy of the same skill."""
    return (*(Path(d).expanduser() for d in extra_dirs), installed_dir)


def resolve_states(skills: Sequence[Skill], state: Mapping[str, str]) -> ResolvedSkills:
    """Apply the operator's `[skills.state]` map (absent name = enabled)."""
    warnings = [
        f"[skills.state] names an unknown skill: {name!r}"
        for name in state
        if name not in {s.name for s in skills}
    ]
    enabled = tuple(s for s in skills if state.get(s.name, "enabled") == "enabled")
    always = tuple(s for s in skills if state.get(s.name) == "always")
    return ResolvedSkills(enabled=enabled, always=always, warnings=tuple(warnings))


def operator_skills(
    enabled: bool, extra_dirs: Sequence[str], state: Mapping[str, str], installed_dir: Path
) -> ResolvedSkills:
    """The skills a run has, from `[skills]`: discovery over `extra_dirs` then
    the installed dir, with the operator's per-skill states applied.

    THE one owner of the master switch, asked by `--skill` and the pause menu
    alike: off means no skills anywhere, not "off for the model only"."""
    if not enabled:
        return ResolvedSkills(enabled=(), always=(), warnings=())
    found, warns = discover_skills(skill_search_dirs(extra_dirs, installed_dir))
    resolved = resolve_states(found, state)
    return ResolvedSkills(
        enabled=resolved.enabled,
        always=resolved.always,
        warnings=(*warns, *resolved.warnings),
    )


def skill_steer_payload(name: str, text: str, args: str) -> str:
    """The instruction a `/<skill> [args]` steer injects: the skill applied
    for the rest of the run, its full text inline, the arguments named."""
    args_line = f"\nSkill arguments: {args}" if args else ""
    return (
        f"Apply the operator-installed skill {name!r} for the rest of this run."
        f"{args_line}\n\n"
        f'<skill name="{name}">\n{text.rstrip()}\n</skill>'
    )


def skill_command(text: str, skills: ResolvedSkills | None) -> tuple[Skill, str] | None:
    """The skill a `/<name> [args]` steer names and its arguments, or None
    when *text* is not a skill command (no leading slash, or a name that is
    not an enabled or always-on skill)."""
    stripped = text.strip()
    if not stripped.startswith("/") or skills is None:
        return None
    word, _, args = stripped[1:].partition(" ")
    for skill in (*skills.enabled, *skills.always):
        if skill.name == word:
            return skill, args.strip()
    return None
