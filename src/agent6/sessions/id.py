# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Friendly run identifiers and prefix resolution.

Run IDs have the shape `<adjective>-<noun>-<suffix>` where `suffix`
is six Crockford base32 characters: four derived from a fresh ULID's
timestamp tail followed by two random. Example: `sunny-otter-K4Q7B2`.

The leading 4 chars of the suffix encode the low 20 bits of the
current millisecond timestamp and are lexicographically sortable, so
directory listings under the per-repo run-state dir are mostly chronological
within the same `<adjective>-<noun>` pair (the timestamp rolls over
roughly every 17 minutes, which is fine for the typical dev session
listing). The trailing 2 chars supply 10 bits of entropy to keep IDs
unique even when several are minted in the same millisecond. The
format is otherwise purely cosmetic; nothing parses run IDs except the
prefix resolver here. Treat them as opaque strings everywhere else.
"""

from __future__ import annotations

import os
from pathlib import Path

from agent6._data.words import ADJECTIVES, NOUNS
from agent6.git_ops import valid_branch_name
from agent6.graph.ulid import new_ulid
from agent6.sessions.layout import (
    SESSION_BUCKETS,
    SessionLayout,
    bucket_dir,
    is_safe_session_id,
    session_matches,
)

_CROCKFORD = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"


class SessionIdError(Exception):
    """Raised when a user-supplied run id cannot be resolved. `ambiguous` is
    True when the query matched more than one run (vs no match), so a caller can
    surface the disambiguation instead of treating it as 'not found'."""

    def __init__(self, message: str, *, ambiguous: bool = False) -> None:
        super().__init__(message)
        self.ambiguous = ambiguous


def validate_explicit_session_id(session_id: str) -> str:
    """Return *session_id* if it can name both a run directory and a git ref,
    else raise. Checked BEFORE any run state is created, since the id fills both
    roles at once and a value that fails either produces a broken run.

    - DIRECTORY: a run id becomes a directory name under the state dir
      (``state_dir/<subdir>/<session_id>``). A separator, `.`/`..`, or an
      absolute path would place run state outside the state dir.
    - GIT REF: the run's commits land on a branch (`agent6/<id>`) and a chain
      ref (`refs/agent6/<id>/head`). A value git's ref grammar rejects (a space,
      any of `~^:?*[\\`, `..`, `@{`, a leading `-`/`.`, a trailing `.`/`.lock`)
      makes every commit's `update-ref` fail; caught here, the run never starts
      rather than reporting success while its work stays uncommitted.

    Generated ids are slug-safe by construction and skip this."""
    if not is_safe_session_id(session_id):
        raise SessionIdError(
            f"invalid --session-id {session_id!r}: must be a single name with no '/', '\\', or '..'"
        )
    if not valid_branch_name(session_id):
        raise SessionIdError(
            f"invalid --session-id {session_id!r}: must be usable as a git branch name "
            "(no spaces or any of ~^:?*[\\, no '..' or '@{', "
            "no leading '-'/'.', no trailing '.' or '.lock')"
        )
    return session_id


def friendly_token() -> str:
    """A fresh `<adj>-<noun>-<suffix>` token.

    A token, not a session id: naming a session DIRECTORY goes through
    :func:`unused_session_id`, which also checks the bucket. Callers that want
    a readable unique string for something else (an ACP connection, a fan-out
    group) use this.
    """

    rand = os.urandom(6)
    adj = ADJECTIVES[(rand[0] << 8 | rand[1]) % len(ADJECTIVES)]
    noun = NOUNS[(rand[2] << 8 | rand[3]) % len(NOUNS)]
    # 4 timestamp-derived chars (low 20 bits of ms timestamp = 1ms
    # resolution, wraps every ~17 min) followed by 2 random chars for
    # in-millisecond uniqueness.
    ts_part = new_ulid()[6:10]
    rnd_part = _CROCKFORD[rand[4] % 32] + _CROCKFORD[rand[5] % 32]
    return f"{adj}-{noun}-{ts_part}{rnd_part}"


def session_id_bucket(state_dir: Path, session_id: str) -> str | None:
    """The bucket whose directory already holds *session_id*, or None.

    Ids are ONE public namespace across every bucket: the CLI resolver, the
    web lookup, and the `agent6/<id>` branch all address a session by bare id,
    so an id living in two buckets is ambiguous on every surface. The mint and
    both explicit `--session-id` entry points check through here."""
    for bucket in SESSION_BUCKETS:
        if (bucket_dir(state_dir, bucket) / session_id).exists():
            return bucket
    return None


def unused_session_id(state_dir: Path, bucket: str) -> str:
    """A freshly minted id whose directory exists in NO session bucket
    (`session_id_bucket`), destined for *bucket*.

    An id carries 4 timestamp chars plus 2 random ones, so two minted in the
    same millisecond collide about once in 30 million. Every site that names a
    session directory mints through here; :func:`friendly_token` is the raw
    generator, for strings that are not directories.
    """
    for _ in range(8):
        candidate = friendly_token()
        if session_id_bucket(state_dir, candidate) is None:
            return candidate
    raise RuntimeError(f"could not mint an unused session id under {bucket_dir(state_dir, bucket)}")


def resolve_session(state_dir: Path, query: str) -> SessionLayout:
    """The one session *query* names (an id, or a prefix of exactly one id) in
    any bucket. Raises SessionIdError otherwise, `ambiguous` set when the
    prefix names several: every surface words a bad id the same way, and an
    ambiguous prefix never reads as "no such session"."""
    if not query:
        raise SessionIdError("empty run id")
    matches = session_matches(state_dir, query)
    if len(matches) > 1:
        preview = ", ".join(f"{m.subdir}/{m.session_id}" for m in matches[:5])
        raise SessionIdError(
            f"run id {query!r} is ambiguous ({len(matches)} matches): {preview}",
            ambiguous=True,
        )
    if not matches:
        raise SessionIdError(f"no session matches {query!r} (looked under {state_dir})")
    return matches[0]


def list_session_ids(runs_dir: Path) -> list[str]:
    """Return run-id directory names under `runs_dir` (unsorted)."""

    if not runs_dir.is_dir():
        return []
    return [p.name for p in runs_dir.iterdir() if p.is_dir()]


def resolve_session_id(runs_dir: Path, query: str) -> str:
    """Resolve `query` to an exact run-id under `runs_dir`.

    Accepts an exact match or an unambiguous prefix. Raises
    `SessionIdError` if no match or more than one match is found.
    """

    if not query:
        raise SessionIdError("empty run id")
    ids = list_session_ids(runs_dir)
    if query in ids:
        return query
    matches = [rid for rid in ids if rid.startswith(query)]
    if not matches:
        raise SessionIdError(f"no session matches {query!r} under {runs_dir}")
    if len(matches) > 1:
        preview = ", ".join(sorted(matches)[:5])
        raise SessionIdError(
            f"run id {query!r} is ambiguous ({len(matches)} matches): {preview}",
            ambiguous=True,
        )
    return matches[0]
