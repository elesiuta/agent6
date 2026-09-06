# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Content access & write handlers: agent6_docs, read_file, list_dir,
apply_edit, apply_patch.

All of these run in-process (never through `agent6.sandbox.jail`), so the
write handlers (apply_edit/apply_patch) carry their own protected-path guard:
`.git` (when `protect_git`), an in-repo virtualenv / installed-package
tree, and any operator/machine `extra_protect_paths` -- none of which the
jail's mount-based protections cover for an in-process write. See
`refuse_protected_writes`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from agent6.config import Config
from agent6.tools._agent6_docs import list_agent6_docs, read_agent6_doc
from agent6.tools._edit_diag import (
    edit_mismatch_error,
    indent_tolerant_replacement,
    preview_result,
)
from agent6.tools._path_safety import (
    SafePath,
    Workspace,
    fold_name,
    list_contained,
    path_within,
    read_contained,
    unlink_contained,
    write_contained,
)
from agent6.tools.errors import ToolError
from agent6.tools.index import SymbolIndex
from agent6.tools.patch_apply import (
    PatchError,
    apply_patch_text,
    apply_v4a_text,
    is_v4a_patch,
    patch_target_path,
    split_patch_files,
)
from agent6.tools.results import (
    DocsContentResult,
    DocsIndexResult,
    EditResult,
    ListDirResult,
    PatchResult,
    PreviewResult,
    ReadFileResult,
    ToolResult,
)
from agent6.tools.schema import (
    WHOLE_FILE_KINDS,
    Agent6DocsInput,
    ApplyEditInput,
    ApplyPatchInput,
    ListDirInput,
    ReadFileInput,
)

# Upper bound on what read_file pulls into memory. read_contained loads the
# whole file before slicing, so without this a large file (a checked-in blob, a
# log, a file a command produced) OOM-crashes the unsandboxed agent. Generous
# enough for any real source file; a bigger file returns its capped prefix with
# truncated=True. Revisit if a legitimate read hits it.
MAX_READ_CHARS = 5_000_000


def agent6_docs(raw: dict[str, Any]) -> ToolResult:
    args = Agent6DocsInput.model_validate(raw)
    available = list_agent6_docs()
    if not args.name:
        return DocsIndexResult(available=tuple(available))
    content = read_agent6_doc(args.name)
    if content is None:
        raise ToolError(
            f"unknown agent6 doc {args.name!r}; available: {', '.join(available) or '(none)'}"
        )
    cap = 60_000
    return DocsContentResult(
        name=args.name,
        content=content[:cap],
        truncated=len(content) > cap,
    )


def read_file(ws: Workspace, raw: dict[str, Any]) -> ReadFileResult:
    args = ReadFileInput.model_validate(raw)
    sp = ws.resolve_read(args.path)
    if not sp.abs_path.is_file():
        raise ToolError(f"Not a file: {args.path}")
    try:
        # Bounded read: the whole file was pulled into memory regardless of
        # start_line/limit, so a multi-GB file (a checked-in blob, a log, a
        # file a command produced) OOM-crashed the unsandboxed agent. Read one
        # char past the cap to detect the overflow, then trim; pagination and
        # the line counts operate on the capped prefix, and `truncated` says so.
        full = read_contained(sp, limit_chars=MAX_READ_CHARS + 1)
    except UnicodeDecodeError as exc:
        raise ToolError(f"File is not UTF-8 text: {args.path}") from exc
    read_truncated = len(full) > MAX_READ_CHARS
    if read_truncated:
        full = full[:MAX_READ_CHARS]
    # A NUL byte is what "binary" means in practice, and some binary payloads
    # decode as UTF-8 -- so the description's promise needs this, not just the
    # decode error. Without it such a file went verbatim into the transcript.
    if "\x00" in full:
        raise ToolError(f"File is binary (contains NUL bytes): {args.path}")
    # One split is the source of truth for every line count: lines_total is its
    # length in both branches (a full read and a later page of the same file
    # must agree), and lines_returned is the returned slice's real length (a
    # past-EOF start_line yields an empty slice, never negative arithmetic).
    lines = full.splitlines(keepends=True)
    if args.start_line == 1 and args.limit is None:
        return ReadFileResult(
            content=full, size=len(full), lines_total=len(lines), truncated=read_truncated
        )
    first = args.start_line - 1  # 1-based on the wire, 0-based slice
    end = len(lines) if args.limit is None else min(len(lines), first + args.limit)
    sliced = lines[first:end]
    slice_text = "".join(sliced)
    return ReadFileResult(
        content=slice_text,
        size=len(slice_text),
        lines_total=len(lines),
        start_line=args.start_line,
        lines_returned=len(sliced),
        truncated=read_truncated,
    )


def list_dir(ws: Workspace, raw: dict[str, Any]) -> ListDirResult:
    args = ListDirInput.model_validate(raw)
    sp = ws.resolve_read(args.path)
    if not sp.abs_path.is_dir():
        raise ToolError(f"Not a directory: {args.path}")
    listing = sorted(list_contained(sp), key=lambda e: e.name)
    # A hidden entry is dropped from the names but COUNTED: the listing stays
    # true ("something here is hidden") without disclosing what, and the model
    # stops probing a path it will only be refused.
    visible = [e for e in listing if not ws.is_denied(sp.abs_path / e.name)]
    return ListDirResult(
        entries=tuple(e.name + "/" if e.is_dir else e.name for e in visible),
        hidden=len(listing) - len(visible),
    )


def _under_project_dir(path: Path, dir_name: str) -> bool:
    """The one protected-directory test, shared by the raw and the resolved
    checks so they can never disagree: the workspace-relative *path* IS the
    top-level *dir_name*, or lies under it."""
    return path_within(path, Path(dir_name))


def _refuse_protected_write(
    candidate: str, dir_name: str, *, why: str, resolved: SafePath | None = None
) -> None:
    """Refuse an in-process `apply_edit` / `apply_patch` into the
    workspace's own top-level `dir_name`.

    `.git` (when `protect_git`): the edit tools write **in-process, outside
    the jail**, so without this an LLM could create or rewrite `.git/hooks/*`
    or `.git/config` (e.g. `core.fsmonitor`) and get code executed outside
    the sandbox on the next `git` invocation, or corrupt git history --
    defeating `protect_git` entirely (the strict jail's RO bind of `.git`
    never covers these in-process writes). The scope is the project's own
    repository, the one agent6 commits to each turn; a nested `.git`
    (vendored repo, submodule gitlink) is content, like any other file. Reads
    stay allowed. (Run state lives out of the workspace, so it is unreachable
    by edits and needs no guard.)

    Checks both the raw candidate string AND the post-symlink-resolution
    relative path, so a symlink `./decoy -> .git` can't launder a write past
    the raw check.
    """
    if _under_project_dir(Path(candidate), dir_name):
        raise ToolError(f"Refusing to write under {dir_name}/ ({why}): {candidate!r}")
    if resolved is not None and _under_project_dir(resolved.rel_path, dir_name):
        raise ToolError(
            f"Refusing to write under {dir_name}/ ({why}) via symlink: {candidate!r} "
            f"resolves to {resolved.rel_path!s}"
        )


def _refuse_env_write(candidate: str, resolved: SafePath) -> None:
    """Refuse an in-process edit into an in-repo virtualenv or installed-package
    tree. These are the operator's ENVIRONMENT, not source: a run editing them
    (e.g. rewriting an editable-install `.pth` to make an in-jail verify pass)
    silently corrupts the operator's venv, and since venvs are gitignored the
    damage never shows in `sessions diff` / merge.

    A directory holding `pyvenv.cfg` is a virtualenv root (the canonical
    marker, name-agnostic: `.venv` / `venv` / `env`); a `site-packages`
    ancestor is an installed tree. Reads stay allowed; only writes are refused.
    The check walks the post-symlink-resolution path so a decoy symlink can't
    launder the write."""
    ancestors = [resolved.abs_path, *resolved.abs_path.parents]
    for anc in ancestors:
        if fold_name(anc.name) == "site-packages":
            raise ToolError(
                f"Refusing to write into an installed-package tree (site-packages): "
                f"{candidate!r}. Installed packages are environment, not source; "
                f"editing them corrupts the operator's virtualenv."
            )
    # A venv root is an ancestor DIRECTORY containing pyvenv.cfg. Check ancestors
    # of the target (not the target itself, which is the file being written).
    for anc in resolved.abs_path.parents:
        try:
            if (anc / "pyvenv.cfg").is_file():
                raise ToolError(
                    f"Refusing to write inside a virtualenv ({anc.name}/): {candidate!r}. "
                    f"A venv is environment, not source; editing it corrupts the "
                    f"operator's setup and never shows in the run's diff."
                )
        except OSError:
            continue


def refuse_protected_writes(
    path: str,
    config: Config,
    extra_protect_paths: tuple[Path, ...],
    resolved: SafePath | None = None,
) -> None:
    """Refuse an in-process edit into a protected location (it bypasses the
    jail entirely). `.git` under `protect_git`, a virtualenv / installed
    package tree (see `_refuse_env_write`), plus any operator/machine
    protect paths (a machine bundle's `.asm.toml` + `scripts/`), which the
    jail marks read-only for `run_command` but the in-process edit tools
    would otherwise let a `mode="run"` state rewrite -- persisting a payload
    for the next run. Applies at both isolation levels."""
    if config.sandbox.protect_git:
        _refuse_protected_write(path, ".git", why="git history/metadata", resolved=resolved)
    if resolved is not None:
        _refuse_env_write(path, resolved)
    if resolved is not None and extra_protect_paths:
        target = resolved.abs_path
        for prot in extra_protect_paths:
            if path_within(target, prot):
                raise ToolError(
                    f"Refusing to write to a protected path (machine bundle): {path!r} "
                    f"resolves under {prot}"
                )


def _existing_text(sp: SafePath, rel_path: str) -> str | None:
    """The file's current text, or None when it does not exist yet (both edit
    tools create). A path that exists but is not a file gets the same clear
    error the read tools give -- letting read_text raise leaked
    "[Errno 21] Is a directory: /abs/host/path" into the model's transcript."""
    if not sp.abs_path.exists():
        return None
    if not sp.abs_path.is_file():
        raise ToolError(f"Not a file: {rel_path}")
    return read_contained(sp)


def apply_edit(
    ws: Workspace,
    config: Config,
    extra_protect_paths: tuple[Path, ...],
    index: SymbolIndex | None,
    raw: dict[str, Any],
) -> ToolResult:
    args = ApplyEditInput.model_validate(raw)
    refuse_protected_writes(args.path, config, extra_protect_paths)
    sp = ws.resolve_write(args.path)
    refuse_protected_writes(args.path, config, extra_protect_paths, sp)
    applied: list[str] = []
    existing = _existing_text(sp, args.path)
    new_content = existing
    for i, edit in enumerate(args.edits):
        if edit.kind in WHOLE_FILE_KINDS:
            if edit.kind == "create" and existing is not None:
                raise ToolError(
                    f"create requested but file already exists: {args.path}"
                    ' (kind="overwrite" replaces it whole)'
                )
            new_content = edit.new_string
            applied.append(edit.kind)
        else:
            if new_content is None:
                raise ToolError(f"replace requested but file does not exist: {args.path}")
            occurrences = new_content.count(edit.old_string)
            if occurrences == 0:
                # Weak models most often miss by indentation depth alone
                # (right lines, wrong leading whitespace). If old_string
                # matches EXACTLY ONE region up to a uniform indent shift,
                # apply it (the shift is verified to reproduce that region
                # byte-for-byte first, so a wrong region can't be edited) --
                # saving a full round-trip. Otherwise hand the model the exact
                # closest on-disk text so it retries directly; re-reading the
                # whole file is the dominant small-model time sink on a botched
                # edit.
                fuzzy = indent_tolerant_replacement(new_content, edit.old_string, edit.new_string)
                if fuzzy is not None:
                    new_content = fuzzy
                    applied.append("replace~indent")
                    continue
                raise ToolError(edit_mismatch_error(args.path, i, new_content, edit.old_string))
            if occurrences > 1:
                raise ToolError(
                    f"old_string is not unique in {args.path} "
                    f"(edit #{i}, {occurrences} matches); add more surrounding "
                    f"context to make it unique"
                )
            new_content = new_content.replace(edit.old_string, edit.new_string, 1)
            applied.append("replace")
    if new_content is None:
        raise ToolError("No content to write")
    if args.preview:
        return preview_result(args.path, existing, new_content, applied=applied)
    write_contained(sp, new_content)
    if index is not None:
        index.mark_changed(sp.abs_path)
    return EditResult(applied=tuple(applied), path=str(sp.rel_path))


def _first_repeated(paths: list[Path]) -> Path | None:
    """The first path two sections of one patch both target, or None."""
    seen: set[Path] = set()
    for path in paths:
        if path in seen:
            return path
        seen.add(path)
    return None


def _stage_patch_section(
    ws: Workspace,
    config: Config,
    extra_protect_paths: tuple[Path, ...],
    *,
    path_arg: str,
    section: str,
) -> tuple[SafePath, str, str | None, str | None, tuple[str, ...]]:
    """Resolve, security-check, and apply ONE single-file patch section in
    memory: (resolved, target, existing, new_content, healed); new_content
    None = the section deletes its file. Nothing touches disk here."""
    # The write location: the explicit `path` arg if given, else derived
    # from the patch headers (V4A always embeds it; GPT-family models omit
    # `path`). Either way it is resolved + protected-path-checked below, so
    # deriving it from the patch never widens where a write can land.
    try:
        derived_path = patch_target_path(section)
    except PatchError as exc:
        raise ToolError(f"apply_patch failed for {path_arg or '<unknown>'}: {exc}") from exc
    target = path_arg or derived_path
    # Security checks on the write location come first (absolute path, repo
    # escape, protected dirs), before the lower-priority model-confusion
    # check that an explicit `path` matches the patch header.
    refuse_protected_writes(target, config, extra_protect_paths)
    sp = ws.resolve_write(target)
    refuse_protected_writes(target, config, extra_protect_paths, sp)
    if path_arg and path_arg != derived_path:
        raise ToolError(
            f"apply_patch: `path` argument {path_arg!r} disagrees with the patch "
            f"header path {derived_path!r}; emit them consistently or omit `path`"
        )
    existing = _existing_text(sp, target)
    try:
        applier = apply_v4a_text if is_v4a_patch(section) else apply_patch_text
        _, new_content, healed = applier(section, existing)
    except PatchError as exc:
        raise ToolError(f"apply_patch failed for {target}: {exc}") from exc
    if new_content is None and existing is None:
        raise ToolError(f"cannot delete {target}: not a file")
    return sp, target, existing, new_content, healed


def apply_patch(
    ws: Workspace,
    config: Config,
    extra_protect_paths: tuple[Path, ...],
    index: SymbolIndex | None,
    raw: dict[str, Any],
) -> ToolResult:
    args = ApplyPatchInput.model_validate(raw)
    sections = split_patch_files(args.patch)
    if len(sections) > 1 and args.path:
        raise ToolError(
            f"apply_patch: `path` argument {args.path!r} is ambiguous for a "
            f"{len(sections)}-file patch; omit `path` (each file names itself)"
        )
    # Per file: resolve + security-check the target, apply in memory. Nothing
    # is written until EVERY section applied cleanly (all-or-nothing across
    # files, matching the per-hunk contract within one file). new_content
    # None = the section deletes its file.
    staged: list[tuple[SafePath, str, str | None]] = []
    seen_paths: list[Path] = []
    previews: list[PreviewResult] = []
    healed_all: list[str] = []
    for section in sections:
        sp, target, existing, new_content, healed = _stage_patch_section(
            ws, config, extra_protect_paths, path_arg=args.path, section=section
        )
        healed_all.extend(healed)
        seen_paths.append(sp.abs_path)
        if args.preview:
            # A deletion previews as the full-removal diff (bytes_after 0).
            previews.append(preview_result(target, existing, new_content or ""))
            continue
        staged.append((sp, target, new_content))
    if (dupe := _first_repeated(seen_paths)) is not None:
        # Each section reads the file from DISK during staging, so two sections
        # over one file both start from the original and the last write wins:
        # the earlier edit vanished while the result reported it applied. The
        # preview says the same thing, so it is refused there too.
        count = sum(1 for path in seen_paths if path == dupe)
        raise ToolError(
            f"apply_patch: {dupe} appears in {count} sections of one patch;"
            " a section reads the file as it is on disk, so only the last would"
            " land. Send one section per file, with every hunk for it inside."
        )
    if args.preview:
        if len(previews) == 1:
            return previews[0]
        return PreviewResult(
            path=previews[0].path,
            diff="".join(pv.diff for pv in previews),
            hunks=sum(pv.hunks for pv in previews),
            bytes_before=sum(pv.bytes_before for pv in previews),
            bytes_after=sum(pv.bytes_after for pv in previews),
            truncated=any(pv.truncated for pv in previews),
            files=tuple(pv.path for pv in previews),
        )
    # Staging was all-or-nothing; the writes are not, so a write that fails
    # part way names what already changed rather than reporting a failure
    # over a tree it altered.
    landed: list[str] = []
    for sp, _target, new_content in staged:
        try:
            if new_content is None:
                unlink_contained(sp)
                if index is not None:
                    index.mark_deleted(sp.abs_path)
            else:
                write_contained(sp, new_content)
                if index is not None:
                    index.mark_changed(sp.abs_path)
        except OSError as exc:
            changed = f"; already changed: {', '.join(landed)}" if landed else ""
            raise ToolError(f"apply_patch: {sp.rel_path}: {exc.strerror or exc}{changed}") from exc
        landed.append(str(sp.rel_path))
    rows = tuple((str(sp.rel_path), len(new)) for sp, _t, new in staged if new is not None)
    deleted = tuple(str(sp.rel_path) for sp, _t, new in staged if new is None)
    return PatchResult(
        path=(rows[0][0] if rows else deleted[0]),
        bytes_written=sum(b for _p, b in rows),
        files=rows if len(staged) > 1 else (),
        deleted=deleted,
        healed=tuple(healed_all),
    )
