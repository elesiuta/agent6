# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Comment-preserving TOML read/write surgery for config writers.

Low-level, UI-agnostic: used by the `config` CLI subcommands, by
`config.write`'s shared edit path, and (through it) by the TUI/web config
editors, so every writer preserves comments + siblings identically."""

from __future__ import annotations

import re
import tomllib
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from agent6.config.model import ConfigError
from agent6.errors import read_operator_file
from agent6.portable import atomic_write, locked_file, toml_basic_string


def _header_name(line: str) -> str | None:
    """The table name of a `[table]` header line, or None if it is not one.

    THE single owner of header matching; tolerates a trailing comment and
    interior whitespace (`[sandbox]  # the jail`, `[ sandbox ]`), both ordinary
    TOML. An array-of-tables (`[[x]]`) is deliberately not a match.
    """
    stripped = line.strip()
    if not stripped.startswith("["):
        return None
    end = stripped.find("]")
    if end == -1:
        return None
    trailing = stripped[end + 1 :].strip()
    if trailing and not trailing.startswith("#"):
        return None
    return stripped[1:end].strip()


def _section_name(line: str) -> str | None:
    """The dotted name of a `[table]` OR `[[array.of.tables]]` header line,
    or None if *line* is not one.

    For DROPPING a whole section: both forms are subtables that must go with
    their parent, so a `[[table.sub]]` under a dropped `[table]` is included.
    `_header_name` is the stricter single-table matcher for a lookup, which
    deliberately rejects `[[x]]`.
    """
    stripped = line.strip()
    if stripped.startswith("[["):
        close = stripped.find("]]")
        if close == -1:
            return None
        trailing = stripped[close + 2 :].strip()
        if trailing and not trailing.startswith("#"):
            return None
        return stripped[2:close].strip()
    return _header_name(line)


def _toml_value(value: str | bool) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return toml_basic_string(value)


def upsert_toml_table(path: Path, table: str, fields: dict[str, ConfigLeafValue]) -> None:
    """Insert or replace a single `[table]` block in *path*, preserving the
    rest of the file (other tables and their comments).

    Append-only-ish: we never round-trip the whole document through a TOML
    serializer (which would drop comments); we only rewrite the target
    table's span. `None` field values are omitted.

    The read-surgery-publish cycle runs under `locked_file` (as do the
    other writers below): two concurrent writers -- a CLI `config set` racing
    the web/TUI config editor -- otherwise both read the same base text and
    the second publish silently drops the first's update.
    """
    block_lines = [f"[{table}]"]
    for key, val in fields.items():
        if val is None:
            continue
        block_lines.append(f"{key} = {format_toml_value(val)}")
    block = "\n".join(block_lines)

    with locked_file(path):
        text = read_operator_file(path) if path.is_file() else ""
        lines = text.splitlines()
        start = _header_line(lines, table)
        if start is None:
            prefix = text if not text or text.endswith("\n") else text + "\n"
            sep = "\n" if prefix and not prefix.endswith("\n\n") else ""
            atomic_write(path, prefix + sep + block + "\n")
            return
        end = _region_end(lines, start + 1)
        new_lines = lines[:start] + block.splitlines() + [""] + lines[end:]
        atomic_write(path, "\n".join(new_lines).rstrip("\n") + "\n")


# What a config leaf can hold, matching what `format_toml_value` serializes:
# scalars, and a list for an array-valued leaf like an argv. `None` omits the
# leaf. Kept in sync with `format_toml_value`: a type this union omits makes a
# caller pre-serialize (an array passed as a string validates as a tuple of
# characters).
ConfigLeafValue = str | bool | int | float | Sequence[str] | None


def format_toml_value(value: object) -> str:  # noqa: PLR0911
    """Serialize a scalar, list, or (inline-table) dict to its TOML literal form."""
    if isinstance(value, bool):  # bool first: it is a subclass of int
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return repr(value)
    if isinstance(value, str):
        return _toml_value(value)
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(format_toml_value(v) for v in value) + "]"
    if isinstance(value, dict):
        # Inline table, e.g. an OpenRouter routing value:
        #   extra_body = { provider = { sort = "throughput" } }
        # Written on one line so the existing leaf-line surgery can replace it
        # wholesale (a nested `[table]` would collide with the inline parent).
        if not value:
            return "{}"
        items = ", ".join(f"{toml_key(k)} = {format_toml_value(v)}" for k, v in value.items())
        return "{ " + items + " }"
    # ConfigError (an OperatorError): a CLI value can land here as-parsed
    # (`config set key 2024-01-01` reads as a TOML date), so the refusal
    # carries to the boundary rather than the crash reporter.
    raise ConfigError(f"cannot serialize {value!r} to TOML")


def toml_key(key: object) -> str:
    """A TOML key: bare if it is a simple identifier, else a quoted string."""
    k = str(key)
    return k if re.fullmatch(r"[A-Za-z0-9_-]+", k) else _toml_value(k)


def parse_cli_value(value: str) -> object:
    """Interpret a CLI-supplied value the way TOML would.

    `true`/`false` become bools, numbers become int/float, quoted or
    bracketed text parses as a TOML string/array, and anything else (e.g. a
    bare enum like `provider_only` or a model id) is taken verbatim as a
    string. This keeps `config set sandbox.network auto` ergonomic
    while still allowing `config set sandbox.protect_git false`.
    """
    try:
        return tomllib.loads(f"_v = {value}")["_v"]
    except tomllib.TOMLDecodeError:
        return value


def _split_dotted_key(dotted_key: str) -> tuple[str, str]:
    """Split `sandbox.network` into `("sandbox", "network")`.

    A single-segment key (the top-level `profile`) splits to table `""`:
    the surgery below targets the file's bare top region, before any
    `[table]` header.
    """
    parts = dotted_key.split(".")
    if any(not p for p in parts):
        raise ConfigError(
            f"config key must be a dotted leaf path like 'sandbox.network', got {dotted_key!r}"
        )
    return ".".join(parts[:-1]), parts[-1]


def upsert_toml_leaf(path: Path, dotted_key: str, value: object) -> None:
    """Set a single `table.leaf` key in *path*, preserving the rest verbatim.

    Like :func:`upsert_toml_table` this is deliberate line surgery rather than
    a full serializer round-trip, so comments and sibling keys/tables survive.
    Creates the `[table]` block if it is absent.

    TOML forbids a bare top-level key and a same-named `[table]` coexisting
    (`profile` vs `[profile]`), so a write REPLACES the conflicting other
    shape. Revalidation still arbitrates whether the new value is semantically
    valid.
    """
    table, leaf = _split_dotted_key(dotted_key)
    new_line = f"{leaf} = {format_toml_value(value)}"
    with locked_file(path):
        text = read_operator_file(path) if path.is_file() else ""
        lines = text.splitlines()
        if table:
            # Refuse a leaf whose ancestor is a headerless table (inline table
            # or dotted key): the surgery only knows `[table]` headers, so it
            # would emit one that collides with the ancestor, and
            # _drop_top_region_key would then delete that ancestor with every
            # sibling inside it. Raised here, not in one command, so every
            # writer hits it (see `undeclared_table_ancestor`).
            if owner := undeclared_table_ancestor(path, dotted_key):
                raise ConfigError(
                    f"{dotted_key} lives inside {owner}, which is not a plain [table]"
                    " (a value, an inline table, a dotted key, or an array-of-tables),"
                    f" so it cannot be set on its own. Set {owner} as a whole, or edit"
                    f" {path} by hand."
                )
            lines = _drop_top_region_key(lines, table.split(".", 1)[0])
            # The other shape this key can already have: its own `[table.leaf]`
            # block. Left in place, the inline value written below declares the
            # same key twice and the file no longer parses.
            lines, _ = _drop_table_lines(lines, dotted_key)
            start = _header_line(lines, table)
            if start is None:
                text = "\n".join(lines) + "\n" if lines else ""
                sep = "\n" if text and not text.endswith("\n\n") else ""
                atomic_write(path, text + sep + f"[{table}]" + "\n" + new_line + "\n")
                return
            region = start + 1
        else:
            lines, _ = _drop_table_lines(lines, leaf)
            region = 0  # top-level key: the bare region before any [table] header
        end = _region_end(lines, region)
        j = _find_leaf_line(lines, region, end, leaf)
        if j is not None:
            # Replace the WHOLE value: a multi-line array or triple-quoted
            # string spans several lines, and rewriting only the opening one
            # orphans the rest into unparseable TOML. Keep a single-line value's
            # trailing comment.
            span = _value_line_span(lines, j)
            replacement = new_line
            if span == 1 and (comment := _line_comment(lines[j])):
                replacement = f"{new_line}  {comment}"
            lines[j : j + span] = [replacement]
            atomic_write(path, "\n".join(lines).rstrip("\n") + "\n")
            return
        insert_at = end
        while insert_at - 1 >= region and lines[insert_at - 1].strip() == "":
            insert_at -= 1
        # A fresh top-level key sitting flush against the first [table] header
        # reads as that table's member; keep a separating blank line.
        flush_against_header = insert_at < len(lines) and lines[insert_at].lstrip().startswith("[")
        gap = [""] if not table and flush_against_header else []
        lines[insert_at:insert_at] = [new_line, *gap]
        atomic_write(path, "\n".join(lines).rstrip("\n") + "\n")


def _drop_table_lines(lines: list[str], table: str) -> tuple[list[str], bool]:
    """*lines* without the `[table]` section (header, body, and `[table.sub]`
    subtables), plus whether anything was dropped."""
    kept: list[str] = []
    dropping = False
    removed = False
    j = 0
    while j < len(lines):
        line = lines[j]
        if line.strip().startswith("["):
            # _section_name, not _header_name: a `[[table.sub]]` is a subtable
            # that must be dropped with its parent (_header_name reports `[[x]]`
            # as not-a-table).
            name = _section_name(line)
            dropping = name is not None and (name == table or name.startswith(f"{table}."))
            removed = removed or dropping
            span = 1
        else:
            # Jump a multi-line value whole (see `_region_end`), else an
            # interior line starting with `[` flips `dropping` mid-value.
            span = _value_line_span(lines, j) if _ASSIGN_RE.match(line) else 1
        if not dropping:
            kept.extend(lines[j : j + span])
        j += span
    return kept, removed


# A line that OPENS a `leaf = value` assignment (not a comment, blank, or
# header). The value may then span more lines (a multi-line array / triple
# string), which _value_line_span measures.
_ASSIGN_RE = re.compile(r"^\s*[^#\s=\[][^=]*=")


def _region_end(lines: list[str], region: int) -> int:
    """Index of the first real `[header]` line at or after *region*, skipping
    the INTERIOR of every multi-line value on the way.

    THE single owner of "where does this table's body end", and the reason it
    cannot be a per-line `startswith("[")` scan: a triple-quoted value whose
    line begins with `[` (a regex character class) would end the region early
    and land the insert inside the operator's string.
    """
    j = region
    while j < len(lines):
        if lines[j].lstrip().startswith("["):
            return j
        # A value spanning several lines is jumped whole so its interior is
        # never mistaken for a header; _value_line_span is >= 1, so j advances.
        j += _value_line_span(lines, j) if _ASSIGN_RE.match(lines[j]) else 1
    return len(lines)


def _find_leaf_line(lines: list[str], region: int, end: int, leaf: str) -> int | None:
    """Index of the line assigning *leaf* within `[region, end)`, or None.

    The quoted spelling (`"protect_git" = true`) is valid TOML and names the
    same leaf, so it matches too; unmatched, the surgery would append a
    duplicate key and roll the write back.

    Skips multi-line value interiors (see `_region_end`)."""
    leaf_re = re.compile(rf"^\s*(\"|')?{re.escape(leaf)}(\"|')?\s*=")
    j = region
    while j < end:
        if leaf_re.match(lines[j]):
            return j
        # _value_line_span is >= 1, so this always advances.
        j += _value_line_span(lines, j) if _ASSIGN_RE.match(lines[j]) else 1
    return None


def _iter_headers(lines: list[str]) -> list[tuple[int, str]]:
    """`(index, name)` for each real `[table]` header, skipping multi-line
    value interiors so a `[header]`-looking line inside a string is never taken
    for one. THE owner every header lookup uses (see `_region_end`).
    """
    out: list[tuple[int, str]] = []
    j = 0
    while j < len(lines):
        name = _header_name(lines[j])
        if name is not None:
            out.append((j, name))
        j += _value_line_span(lines, j) if _ASSIGN_RE.match(lines[j]) else 1
    return out


def _header_line(lines: list[str], table: str) -> int | None:
    """Index of the `[table]` header line, value-span-aware (see _iter_headers)."""
    return next((i for i, name in _iter_headers(lines) if name == table), None)


def _drop_top_region_key(lines: list[str], key: str) -> list[str]:
    """*lines* without a bare top-level `key = ...` (multi-line value included).

    The top region ends at the first `[table]` header; a same-named key
    inside a table is someone else's and stays.
    """
    end = _region_end(lines, 0)
    key_re = re.compile(rf"^\s*{re.escape(key)}\s*=")
    j = 0
    while j < end:
        if key_re.match(lines[j]):
            return lines[:j] + lines[j + _value_line_span(lines, j) :]
        # Skip a multi-line value's interior so a `key = ...`-looking line inside
        # an EARLIER key's triple-quoted value is not matched and mis-dropped.
        j += _value_line_span(lines, j) if _ASSIGN_RE.match(lines[j]) else 1
    return lines


def _scan_toml_line(text: str, depth: int, triple: str | None) -> tuple[int, str | None]:
    """Advance the (bracket-depth, open-triple-quote) state across one line, so
    `_value_line_span` can tell where a multi-line value ends. Brackets and
    quotes inside a string, and everything after a `#` comment, do not count."""
    i, n = 0, len(text)
    while i < n:
        if triple is not None:
            triple, i = (None, i + 3) if text.startswith(triple, i) else (triple, i + 1)
            continue
        if text.startswith('"""', i) or text.startswith("'''", i):
            triple, i = text[i : i + 3], i + 3
            continue
        ch = text[i]
        if ch in ('"', "'"):
            i += 1
            while i < n and text[i] != ch:
                i += 2 if (ch == '"' and text[i] == "\\") else 1
            i += 1
            continue
        if ch == "#":
            break  # rest of the line is a comment
        depth += (ch in "[{") - (ch in "]}")
        i += 1
    return depth, triple


def _line_comment(line: str) -> str:
    """The trailing `# comment` (text only) on a single TOML line, or "" -- a
    `#` inside a string is not a comment."""
    i, n, triple = 0, len(line), None
    while i < n:
        if triple is not None:
            triple, i = (None, i + 3) if line.startswith(triple, i) else (triple, i + 1)
            continue
        if line.startswith('"""', i) or line.startswith("'''", i):
            triple, i = line[i : i + 3], i + 3
            continue
        ch = line[i]
        if ch in ('"', "'"):
            i += 1
            while i < n and line[i] != ch:
                i += 2 if (ch == '"' and line[i] == "\\") else 1
            i += 1
            continue
        if ch == "#":
            return line[i:].rstrip()
        i += 1
    return ""


def _value_line_span(lines: list[str], start: int) -> int:
    """How many lines the TOML value assigned on `lines[start]` spans (>=1).

    A multi-line array (`leaf = [`...`]`) or triple-quoted string occupies
    several lines; deleting only the opening line orphans the rest and leaves an
    unparseable file."""
    eq = lines[start].find("=")
    text = lines[start][eq + 1 :] if eq != -1 else lines[start]
    depth, triple = 0, None
    idx = start
    while True:
        depth, triple = _scan_toml_line(text, depth, triple)
        if triple is None and depth <= 0:
            return idx - start + 1
        idx += 1
        if idx >= len(lines):
            return idx - start  # unterminated value: consume to EOF
        text = lines[idx]


def remove_toml_leaf(path: Path, dotted_key: str) -> bool:
    """Delete a single `table.leaf` line from *path*. Returns True if removed.
    Removing the section's last leaf drops the now-empty `[table]` header too
    (a dangling header otherwise accretes across unsets); a section that still
    holds comments is kept, they are the operator's."""
    table, leaf = _split_dotted_key(dotted_key)
    with locked_file(path):
        if not path.is_file():
            return False
        # The removal twin of upsert_toml_leaf's refusal: without it a leaf
        # inside an inline table or dotted key reads "not found" here, and
        # callers translate False into "nothing to unset" while `config get`
        # shows
        # the leaf set.
        if table and (owner := undeclared_table_ancestor(path, dotted_key)):
            raise ConfigError(
                f"{dotted_key} lives inside {owner}, which is not a plain [table]"
                " (an inline table, a dotted key, or an array-of-tables), so it"
                f" cannot be unset on its own. Set {owner} as a whole, or edit {path}"
                " by hand."
            )
        lines = read_operator_file(path).splitlines()
        if table:
            start = _header_line(lines, table)
            if start is None:
                return False
            region = start + 1
        else:
            start = None  # top-level key: no header line to clean up after
            region = 0
        end = _region_end(lines, region)
        j = _find_leaf_line(lines, region, end, leaf)
        if j is not None:
            span = _value_line_span(lines, j)
            del lines[j : j + span]
            if start is not None:
                remaining_end = end - span  # next section header shifted up by span
                if all(not rest.strip() for rest in lines[start + 1 : remaining_end]):
                    del lines[start:remaining_end]
            out = "\n".join(lines).rstrip("\n") + "\n" if lines else ""
            atomic_write(path, out)
            return True
        return False


def remove_toml_table(path: Path, table: str) -> bool:
    """Delete a whole `[table]` section (its header, body, and any `[table.sub]`
    subtables) from *path*. Returns True if the table was present. Used by
    `config fix` to drop an unknown/extra top-level table (e.g. a leftover
    `[cli]` from a removed feature), where deleting a single leaf would leave an
    empty-but-still-invalid table behind."""
    with locked_file(path):
        if not path.is_file():
            return False
        lines = read_operator_file(path).splitlines()
        kept, removed = _drop_table_lines(lines, table)
        if not removed:
            return False
        out = "\n".join(kept).rstrip("\n") + "\n" if any(ln.strip() for ln in kept) else ""
        atomic_write(path, out)
        return True


def read_toml_file(path: Path) -> dict[str, Any]:
    """Parse *path* as TOML, or return an empty dict if it does not exist.

    Wrap a parse error in `ConfigError` (matching `config.layer._read_toml`)
    so the `config ... --machine-file FILE` commands surface a clean message
    instead of letting a raw `TOMLDecodeError` traceback escape -- and, for
    `set`/`add`, so the malformed file is reported before it is rewritten.
    """
    if not path.is_file():
        return {}
    try:
        return tomllib.loads(path.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"{path}: invalid TOML: {exc}") from exc
    except (OSError, UnicodeDecodeError) as exc:
        raise ConfigError(f"{path}: cannot be read: {exc}") from exc


def undeclared_table_ancestor(path: Path, dotted_key: str) -> str | None:
    """The outermost ancestor of *dotted_key* the leaf surgery can't write under
    -- a plain value, an inline table, a dotted key, or an array-of-tables
    (`[[x]]`) -- else None. The surgery only knows `[table]` headers, so writing
    under one emits a header that collides with it ("Cannot declare ... twice");
    the caller names the owning value instead of leaking the parser's complaint
    about a file it discarded.
    """
    if not path.is_file():
        return None
    text = read_operator_file(path)
    try:
        data = tomllib.loads(text)
    except tomllib.TOMLDecodeError:
        return None  # a file that does not parse is refused earlier, with its own message
    headers = [name for _, name in _iter_headers(text.splitlines())]
    parts = dotted_key.split(".")
    for i in range(1, len(parts)):  # proper ancestors, outermost first
        prefix = ".".join(parts[:i])
        val = read_toml_leaf(data, prefix)
        if val is None:
            continue  # absent: the surgery declares the [table] itself
        if isinstance(val, list):
            return prefix  # an array-of-tables: a leaf can't be set on it
        if not isinstance(val, dict):
            # A scalar where a table belongs. A bare top-level key is replaced
            # by the write itself (`_drop_top_region_key`); one inside a table
            # is not, and the header written under it declares it twice.
            if "." in prefix:
                return prefix
            continue
        if any(h == prefix or h.startswith(f"{prefix}.") for h in headers):
            continue  # a real [table] header declares it
        return prefix
    return None


def read_toml_leaf(data: dict[str, Any], dotted_key: str) -> object:
    """Walk *data* by the dotted key, returning the value or None if absent."""
    cur: object = data
    for part in dotted_key.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur
