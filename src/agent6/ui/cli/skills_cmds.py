# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""`agent6 skills` command family: install/update/list/enable/disable/remove.

Install fetches operator-chosen skill content (a direct SKILL.md URL, a git
repository, or a local path) into the user data dir. Nothing fetched is ever
executed: skills are prompt text, trusted like config because the operator
chose them. Each installed skill carries a `.origin.toml` provenance file
(ignored by discovery) so `skills update` can re-fetch from the same source.
"""

from __future__ import annotations

import hashlib
import shutil
import subprocess
import sys
import tempfile
import tomllib
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import httpx2

from agent6.config.io import remove_toml_leaf, upsert_toml_leaf
from agent6.config.layer import load_effective, repo_config_path_for
from agent6.errors import OperatorError, read_operator_file
from agent6.paths import (
    chown_to_real_user,
    data_dir,
    global_config_path,
    mkdir_for_real_user,
)
from agent6.skills import (
    Skill,
    discover_skills,
    is_valid_skill_name,
    parse_frontmatter,
    resolve_states,
    skill_search_dirs,
)
from agent6.ui.cli._common import home_contracted, sgr
from agent6.ui.cli._steer_menu import MENU_COMMANDS

_ORIGIN_FILE = ".origin.toml"
_FETCH_TIMEOUT_S = 30.0
_FETCH_MAX_BYTES = 1_048_576  # a SKILL.md is prose; 1 MiB is already generous


def _term_width() -> int:
    return shutil.get_terminal_size((100, 24)).columns


def _one_line(text: str, width: int) -> str:
    """Collapse whitespace to a single line and truncate to *width* so a long
    description can never wrap into a wall of text."""
    text = " ".join(text.split())
    if width < 12 or len(text) <= width:
        return text
    return text[: width - 1].rstrip() + "…"


def _short_source(src: str) -> str:
    """A compact provenance label: drop the URL scheme and `.git`, contract
    $HOME to `~`."""
    src = src.removesuffix(".git")
    for scheme in ("https://", "http://", "ssh://", "git://"):
        if src.startswith(scheme):
            src = src[len(scheme) :]
            break
    return home_contracted(src)


def _installed_dir() -> Path:
    return data_dir() / "skills"


def _search_dirs(repo_root: Path, config_path: Path | None = None) -> tuple[Path, ...]:
    cfg = load_effective(repo_root, config_path).config
    return skill_search_dirs(cfg.skills.extra_dirs, _installed_dir())


def _toml_str(value: str) -> str:
    """A TOML basic string with backslashes and quotes escaped: a quote in a
    source path made the hand-built origin unparseable, so update lost its
    origin."""
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _write_origin(skill_dir: Path, *, url: str, kind: str, source_sha: str) -> None:
    digest = hashlib.sha256((skill_dir / "SKILL.md").read_bytes()).hexdigest()
    fetched = datetime.now(tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    body = (
        f"url = {_toml_str(url)}\nkind = {_toml_str(kind)}\n"
        f"source_sha = {_toml_str(source_sha)}\n"
        f"fetched_at = {_toml_str(fetched)}\nsha256 = {_toml_str(digest)}\n"
    )
    (skill_dir / _ORIGIN_FILE).write_text(body, encoding="utf-8")


def _read_origin(skill_dir: Path) -> dict[str, str] | None:
    p = skill_dir / _ORIGIN_FILE
    if not p.is_file():
        return None
    try:
        raw = tomllib.loads(p.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError):
        return None
    return {k: str(v) for k, v in raw.items()}


def _fetch_url(url: str) -> str:
    try:
        resp = httpx2.get(url, timeout=_FETCH_TIMEOUT_S, follow_redirects=True)
        resp.raise_for_status()
    except httpx2.HTTPError as exc:
        raise OperatorError(f"could not fetch {url}: {exc}") from exc
    if len(resp.content) > _FETCH_MAX_BYTES:
        raise OperatorError(f"{url}: remote file exceeds {_FETCH_MAX_BYTES} bytes")
    try:
        return resp.content.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise OperatorError(f"{url}: not UTF-8 text: {exc}") from exc


def _skill_name_from_text(text: str, source: str) -> str:
    fields, _warnings = parse_frontmatter(text)
    name, description = fields.get("name", ""), fields.get("description", "")
    if not name or not description:
        raise OperatorError(f"{source}: SKILL.md lacks required frontmatter name/description")
    # The name becomes a path component under the skills dir (and, under --force,
    # an rmtree target). Untrusted frontmatter must not contain `/`, `..`, or an
    # absolute path; gate on the same rule discovery enforces.
    if not is_valid_skill_name(name):
        raise OperatorError(
            f"{source}: invalid skill name {name!r} "
            "(letters, digits, and hyphens only, starting alphanumeric)"
        )
    return name


def _refuse_existing(name: str, *, force: bool) -> Path:
    """The target dir for *name*; refuses when it exists without --force.

    Never clears: the old install survives until the staged replacement is
    fully built (`_publish_staged`), so a copy or write fault cannot destroy
    a good skill."""
    target = _installed_dir() / name
    if target.exists() and not force:
        origin = _read_origin(target)
        src = f" (installed from {origin['url']})" if origin and origin.get("url") else ""
        raise OperatorError(f"skill {name!r} is already installed{src}; use --force to replace")
    return target


def _publish_staged(staging: Path, target: Path) -> None:
    """Swap the fully-built staging dir into place; the old install goes only
    now. The dot-prefixed staging name fails the skill-name gate, so a
    crash's leftover is invisible to discovery."""
    if target.exists():
        shutil.rmtree(target)
    staging.rename(target)


def _install_skill_dir(src: Path, *, url: str, kind: str, source_sha: str, force: bool) -> str:
    """Copy one skill directory (SKILL.md + supplementary files) into place.

    `symlinks=True`: the skill comes from an untrusted source, and copying a
    link's CONTENT turns `reference.md -> secrets.toml` into a real file
    `use_skill` will serve. Preserved, the link stays subject to `use_skill`'s
    containment check."""
    name = _skill_name_from_text(read_operator_file(src / "SKILL.md"), str(src))
    target = _refuse_existing(name, force=force)
    mkdir_for_real_user(target.parent)
    staging = target.parent / f".staging-{name}"
    shutil.rmtree(staging, ignore_errors=True)
    try:
        shutil.copytree(src, staging, symlinks=True)
        (staging / _ORIGIN_FILE).unlink(missing_ok=True)  # never inherit a copied origin
        _write_origin(staging, url=url, kind=kind, source_sha=source_sha)
        chown_to_real_user(staging)
        _publish_staged(staging, target)
    except OSError as exc:
        shutil.rmtree(staging, ignore_errors=True)
        raise OperatorError(f"could not install the skill from {src}: {exc}") from exc
    return name


def _install_skill_text(text: str, *, url: str, force: bool) -> str:
    """Install a single-file skill from raw SKILL.md text."""
    name = _skill_name_from_text(text, url)
    target = _refuse_existing(name, force=force)
    mkdir_for_real_user(target.parent)
    staging = target.parent / f".staging-{name}"
    shutil.rmtree(staging, ignore_errors=True)
    try:
        staging.mkdir()
        (staging / "SKILL.md").write_text(text, encoding="utf-8")
        _write_origin(staging, url=url, kind="skillmd", source_sha="")
        chown_to_real_user(staging)
        _publish_staged(staging, target)
    except OSError as exc:
        shutil.rmtree(staging, ignore_errors=True)
        raise OperatorError(f"could not install the skill from {url}: {exc}") from exc
    return name


def _git_clone(url: str, dest: Path) -> str:
    """Shallow-clone *url* (operator-chosen, CLI-side, nothing executed) and
    return the clone's HEAD sha."""
    try:
        subprocess.run(
            ["git", "clone", "--depth", "1", "--quiet", "--", url, str(dest)],
            check=True,
            capture_output=True,
            text=True,
        )
        head = subprocess.run(
            ["git", "-C", str(dest), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        raise OperatorError("git not found on PATH") from exc
    except subprocess.CalledProcessError as exc:
        raise OperatorError(f"git clone of {url} failed: {exc.stderr.strip()}") from exc
    return head.stdout.strip()


def _repo_skill_dirs(root: Path) -> list[Path]:
    """Skill directories inside a fetched repository: `skills/*/SKILL.md`
    (the superpowers/caveman layout) or the repository root itself."""
    out = [
        p
        for p in sorted((root / "skills").glob("*"))
        if p.is_dir() and not p.name.startswith(".") and (p / "SKILL.md").is_file()
    ]
    if not out and (root / "SKILL.md").is_file():
        out = [root]
    return out


def _refuse_any_existing(dirs: list[Path], *, force: bool) -> None:
    """Pre-check every skill name in a multi-skill install so a conflict
    refuses the WHOLE install up front (never a partial install)."""
    if force:
        return
    conflicts = [
        name
        for d in dirs
        if (name := _skill_name_from_text(read_operator_file(d / "SKILL.md"), str(d)))
        and (_installed_dir() / name).exists()
    ]
    if conflicts:
        raise OperatorError(
            f"already installed: {', '.join(conflicts)}; use --force to replace"
            " (nothing was installed)"
        )


def _install_from_local(local: Path, *, force: bool) -> list[str]:
    """A local SKILL.md file, one skill dir, or a repo checkout."""
    src_url = str(local.resolve())
    if local.is_file():
        return [_install_skill_text(read_operator_file(local), url=src_url, force=force)]
    if (local / "SKILL.md").is_file():
        return [_install_skill_dir(local, url=src_url, kind="dir", source_sha="", force=force)]
    dirs = _repo_skill_dirs(local)
    _refuse_any_existing(dirs, force=force)
    return [
        _install_skill_dir(d, url=src_url, kind="dir", source_sha="", force=force) for d in dirs
    ]


def _install_from_git(url: str, *, force: bool) -> list[str]:
    with tempfile.TemporaryDirectory(prefix="agent6-skill-") as tmp:
        clone = Path(tmp) / "repo"
        sha = _git_clone(url, clone)
        dirs = _repo_skill_dirs(clone)
        if not dirs:
            raise OperatorError(f"no skills found in {url} (expected skills/*/SKILL.md)")
        _refuse_any_existing(dirs, force=force)
        return [
            _install_skill_dir(d, url=url, kind="git", source_sha=sha, force=force) for d in dirs
        ]


def _cmd_skills_install(url: str, *, force: bool, config_path: Path | None = None) -> int:
    local = Path(url).expanduser()
    if local.exists():
        installed = _install_from_local(local, force=force)
    elif url.endswith(".md"):
        installed = [_install_skill_text(_fetch_url(url), url=url, force=force)]
    else:
        installed = _install_from_git(url, force=force)
    if not installed:
        raise OperatorError(f"no skills found in {url} (expected SKILL.md or skills/*/SKILL.md)")
    skills, _ = discover_skills([_installed_dir()])
    by_name = {s.name: s for s in skills}
    width = _term_width()
    if len(installed) == 1:
        name = installed[0]
        print(f"Installed {sgr(name, '1')}")
        if desc := (by_name[name].description if name in by_name else ""):
            print(f"  {sgr(_one_line(desc, width - 2), '2')}")
    else:
        print(sgr(f"Installed {len(installed)} skills from {_short_source(url)}:", "1"))
        name_w = min(32, max(len(n) for n in installed))
        for name in sorted(installed):
            desc = by_name[name].description if name in by_name else ""
            prefix = f"  {name:<{name_w}}  "
            print(f"{prefix}{sgr(_one_line(desc, max(20, width - len(prefix))), '2')}")
    for name in installed:
        if f"/{name}" in MENU_COMMANDS:
            print(
                f"note: /{name} is a built-in pause-menu command and keeps its meaning;"
                " the skill stays reachable via the <skills> index, use_skill, and --skill"
            )
    if _print_disabled_notes(installed, config_path):
        print(sgr("Installed; `agent6 skills list` shows the effective state.", "2"))
    else:
        print(sgr("Enabled and active now; `agent6 skills list` to review.", "2"))
    return 0


def _print_disabled_notes(installed: list[str], config_path: Path | None) -> bool:
    """Name every installed skill a surviving `skills.state = "disabled"` leaf
    covers; True when any did (the closing line must not read "active")."""
    disabled = sorted(n for n in installed if _state_map(config_path).get(n) == "disabled")
    for name in disabled:
        print(
            f'note: skills.state.{name} = "disabled" applies to this name;'
            f" `agent6 skills enable {name}` clears it"
        )
    return bool(disabled)


def _state_map(config_path: Path | None) -> dict[str, str]:
    """The effective `[skills.state]` map, {} when config is unreadable: the
    notes built on it then just do not print, and the command's own work is
    already done."""
    try:
        return dict(load_effective(Path.cwd(), config_path).config.skills.state)
    except Exception:
        return {}


def _refetch_skill(name: str, origin: dict[str, str]) -> tuple[str, str]:
    """Re-install one skill in place from its recorded origin. Return the
    installed name and "" on success, or *name* and a short skip note when the
    source no longer exists.

    Dispatch on the recorded `kind`, not on whether a path happens to exist: a
    local file or dir that was moved or deleted is a clean skip, not a failed
    HTTP fetch of the path.

    A skillmd origin whose frontmatter now declares a different name is a
    RENAME: the skill installs under the new name and the old directory goes,
    or the update would leave two live copies while reporting one.
    """
    url, kind = origin["url"], origin.get("kind", "skillmd")
    if kind == "git":
        with tempfile.TemporaryDirectory(prefix="agent6-skill-") as tmp:
            clone = Path(tmp) / "repo"
            sha = _git_clone(url, clone)
            src = next((d for d in _repo_skill_dirs(clone) if d.name == name), None)
            if src is None:
                return name, "(gone from origin)"
            _install_skill_dir(src, url=url, kind="git", source_sha=sha, force=True)
        return name, ""
    if kind == "dir":
        root = Path(url)
        candidates = _repo_skill_dirs(root)
        # A repo-style install records the repo root as the origin and finds the
        # skill in a subdir by name; a single-dir install records the dir itself.
        if candidates == [root]:
            src = root
        else:
            src = next((d for d in candidates if d.name == name), None)
        if src is None:
            return name, "(gone from origin)"
        _install_skill_dir(src, url=url, kind="dir", source_sha="", force=True)
        return name, ""
    # skillmd: a single SKILL.md, either a remote URL or a local file.
    if url.startswith(("http://", "https://")):
        text = _fetch_url(url)
    elif Path(url).is_file():
        text = read_operator_file(Path(url))
    else:
        return name, "(gone from origin)"
    installed = _install_skill_text(text, url=url, force=True)
    if installed != name:
        shutil.rmtree(_installed_dir() / name, ignore_errors=True)
    return installed, ""


def _cmd_skills_update(name: str) -> int:
    base = _installed_dir()
    if name and not (base / name).is_dir():
        raise OperatorError(f"{name!r} is not installed")
    targets = [base / name] if name else sorted(p for p in base.glob("*") if p.is_dir())
    if not targets:
        print("no skills installed. Install one with `agent6 skills install <url>`.")
        return 0
    name_w = min(32, max(len(p.name) for p in targets))

    def _row(skill: str, status: str, *, dim: bool, note: str = "") -> None:
        line = f"  {skill:<{name_w}}  {f'{status}  {note}'.rstrip()}"
        print(sgr(line, "2") if dim else line)

    counts = {"updated": 0, "unchanged": 0, "skipped": 0}
    for skill_dir in targets:
        origin = _read_origin(skill_dir)
        if origin is None or not origin.get("url"):
            _row(skill_dir.name, "skipped", dim=True, note="(no origin recorded)")
            counts["skipped"] += 1
            continue
        before = origin.get("sha256", "")
        try:
            installed, note = _refetch_skill(skill_dir.name, origin)
        except OperatorError as exc:
            # Re-raise with the skill's name: an all-skills sweep otherwise
            # names only the failing origin, not which install it belongs to.
            raise OperatorError(f"{skill_dir.name}: {exc}") from exc
        if note:
            _row(skill_dir.name, "skipped", dim=True, note=note)
            counts["skipped"] += 1
            continue
        if installed != skill_dir.name:
            _row(skill_dir.name, "updated", dim=False, note=f"(renamed to {installed})")
            counts["updated"] += 1
            continue
        after = _read_origin(base / skill_dir.name) or {}
        if after.get("sha256", "") != before:
            _row(skill_dir.name, "updated", dim=False)
            counts["updated"] += 1
        else:
            _row(skill_dir.name, "unchanged", dim=True)
            counts["unchanged"] += 1
    parts = [f"{counts[k]} {k}" for k in ("updated", "unchanged", "skipped") if counts[k]]
    print(sgr(", ".join(parts), "1"))
    return 0


def _cmd_skills_list(config_path: Path | None = None) -> int:
    repo_root = Path.cwd()
    try:
        cfg = load_effective(repo_root, config_path).config
    except Exception as exc:
        print(f"(config unreadable, showing installed dir only: {exc})", file=sys.stderr)
        cfg = None
    dirs = (
        skill_search_dirs(cfg.skills.extra_dirs, _installed_dir())
        if cfg is not None
        else (_installed_dir(),)
    )
    skills, warnings = discover_skills(dirs)
    state = dict(cfg.skills.state) if cfg is not None else {}
    if not skills:
        print("no skills installed. Install one with `agent6 skills install <url>`.")
        return 0

    states = [state.get(s.name, "enabled") for s in skills]
    counts = Counter(states)
    detail = [f"{counts[k]} {k}" for k in ("disabled", "always") if counts[k]]
    summary = f"{len(skills)} skill{'s' if len(skills) != 1 else ''}"
    if detail:
        summary += f"  ({', '.join(detail)})"
    print(sgr(summary, "1"))

    # Group by origin so a repo that ships 20 skills prints its URL once, not
    # once per line. The state tag column only appears when some skill is not
    # plain-enabled, so the common all-enabled listing stays tight.
    groups: dict[str, list[Skill]] = {}
    for s in skills:
        origin = _read_origin(s.dir)
        src = origin["url"] if origin and origin.get("url") else str(s.dir.parent)
        groups.setdefault(src, []).append(s)
    show_state = any(st != "enabled" for st in states)
    name_w = min(32, max(len(s.name) for s in skills))
    tag_w = len("[disabled]")
    for src, items in groups.items():
        print(f"\n{sgr(_short_source(src), '2')}")
        for s in sorted(items, key=lambda k: k.name):
            st = state.get(s.name, "enabled")
            name = s.name if len(s.name) <= name_w else s.name[: name_w - 1] + "…"
            prefix = f"  {name:<{name_w}}  "
            if show_state:
                prefix += f"{('' if st == 'enabled' else f'[{st}]'):<{tag_w}}  "
            print(f"{prefix}{_one_line(s.description, max(20, _term_width() - len(prefix)))}")
    for w in warnings:
        print(f"WARNING: {w}", file=sys.stderr)
    return 0


def _known_skill_names(repo_root: Path, config_path: Path | None = None) -> tuple[str, ...]:
    """Installed skill names. REFUSES when discovery itself fails.

    Swallowing that to an empty tuple made `skills enable/disable` answer
    "unknown skill 'x'; installed: (none)" -- sending the operator after a
    skill that is missing when one is unreadable.
    """
    try:
        skills, _ = discover_skills(_search_dirs(repo_root, config_path))
    except OSError as exc:
        raise OperatorError(f"could not read the installed skills: {exc}") from exc
    return tuple(s.name for s in skills)


def _state_target(repo: bool) -> Path:
    # Through the resolved owner: the raw path helper ignores the global
    # [agent6].state_dir override, so a custom-state setup wrote skill state
    # into the DEFAULT tree while the effective config read the override --
    # success printed, nothing changed.
    return repo_config_path_for(Path.cwd()) if repo else global_config_path()


def _require_known(name: str, repo_root: Path, config_path: Path | None = None) -> None:
    known = _known_skill_names(repo_root, config_path)
    if name not in known:
        raise OperatorError(f"unknown skill {name!r}; installed: {', '.join(known) or '(none)'}")


def _cmd_skills_enable(
    name: str, *, always: bool, repo: bool, config_path: Path | None = None
) -> int:
    # A state leaf can outlive its skill (remove deletes the install, never
    # the operator's config): clearing that leaf must not need the skill to
    # still exist, or the entry is unreachable by the CLI that wrote it.
    if not (not always and _state_map(config_path).get(name)):
        _require_known(name, Path.cwd(), config_path)
    target = _state_target(repo)
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        if always:
            upsert_toml_leaf(target, f"skills.state.{name}", "always")
            print(f'Set skills.state.{name} = "always" in {target}')
        # Absent means enabled; removing the key reverts to the default and
        # keeps the config free of no-op entries.
        elif remove_toml_leaf(target, f"skills.state.{name}") if target.is_file() else False:
            print(f"Unset skills.state.{name} in {target} (enabled is the default)")
        else:
            print(f"{name} is already enabled (no state entry in {target})")
    finally:
        chown_to_real_user(target)
    return 0


def _cmd_skills_disable(name: str, *, repo: bool, config_path: Path | None = None) -> int:
    _require_known(name, Path.cwd(), config_path)
    target = _state_target(repo)
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        upsert_toml_leaf(target, f"skills.state.{name}", "disabled")
    finally:
        chown_to_real_user(target)
    print(f'Set skills.state.{name} = "disabled" in {target}')
    return 0


def _cmd_skills_remove(name: str, config_path: Path | None = None) -> int:
    """Delete the installed skill *name* from the managed skills dir.

    The name becomes a path component under that dir and an rmtree target, so
    it must be one safe component (the same gate install applies).
    """
    if not is_valid_skill_name(name):
        raise OperatorError(
            f"invalid skill name {name!r} "
            "(letters, digits, and hyphens only, starting alphanumeric)"
        )
    target = _installed_dir() / name
    if not target.is_dir():
        # Distinguish "managed elsewhere" from "unknown" for a useful error.
        skills, _ = discover_skills(_search_dirs(Path.cwd(), config_path))
        match = next((s for s in skills if s.name == name), None)
        if match is not None:
            raise OperatorError(
                f"{name!r} lives in an extra_dirs location ({match.dir});"
                " remove it there or drop the dir from [skills].extra_dirs"
            )
        raise OperatorError(f"{name!r} is not installed")
    shutil.rmtree(target)
    print(f"removed {name}")
    if state := _state_map(config_path).get(name, ""):
        print(
            f'note: skills.state.{name} = "{state}" remains in config and applies if'
            f" {name!r} is installed again; `agent6 skills enable {name}` clears it"
        )
    return 0


def resolved_skill_names_for_completion(repo_root: Path) -> list[str]:
    """Names for argcomplete: cheap discovery, never raises.

    A shell completion has nowhere to show an error and must not raise into the
    shell, so a discovery failure is nothing at all -- unlike the enable/disable
    commands, which a human is reading.
    """
    try:
        return list(_known_skill_names(repo_root))
    except OperatorError:
        return []


__all__ = [
    "Skill",
    "resolve_states",
    "resolved_skill_names_for_completion",
]
