# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The environment a process agent6 spawns OUTSIDE the jail inherits.

A leaf, because its callers sit on opposite sides of the layering: the
operator's notify hooks (`app`), the MCP servers (`tools` and the app
setup). One owner, so their env-scope claims cannot drift apart.

Jailed commands do not come here -- `sandbox.jail` builds their env from the
policy, which is narrower still.
"""

from __future__ import annotations

import os
from collections.abc import Iterable

# Enough to execute a program. Never the whole environment: it carries the
# provider API keys resolved via `[providers.*].api_key_env`, and a child that
# logs or forwards its env -- a shell wrapper, a webhook poster, an MCP server
# -- would carry the key with it.
_KEEP = (
    "PATH",
    "HOME",
    "USER",
    "SHELL",
    "LANG",
    "LC_ALL",
    "TERM",
    "TMPDIR",
)

# How to reach the operator's desktop session. A notify hook needs these --
# `notify-send` talks to the session bus. A CONFINED child must not have them:
# the session bus reaches `systemd --user`, which is NOT confined and will
# gladly run a command on the caller's behalf. Landlock gates filesystem
# paths, not `connect()` to a unix socket, so a server denied `/etc/passwd`
# directly could still have systemd read it and write the result anywhere.
_DESKTOP = (
    "DISPLAY",
    "WAYLAND_DISPLAY",
    "DBUS_SESSION_BUS_ADDRESS",
    "XDG_RUNTIME_DIR",
)


# The `[providers.*].api_key_env` names this run resolves keys from, registered
# once at startup. agent6's own credentials: no child it spawns needs one, and
# a child that logs or forwards its environment would carry it. Mutated, not
# rebound, so importers hold the one set.
_provider_key_env: set[str] = set()


def set_provider_key_env(names: Iterable[str]) -> None:
    """Register the provider-key env var names agent6 keeps out of the children
    it spawns (git subprocesses, an unconfined command)."""
    _provider_key_env.clear()
    _provider_key_env.update(n for n in names if n)


def without_provider_keys(env: dict[str, str]) -> dict[str, str]:
    """*env* minus the registered provider-key names. For a child that inherits
    the operator's environment: at `isolation = "none"` a model-chosen command
    runs unconfined, and a key that lives only in the shell (never on disk) has
    no business in it."""
    return {k: v for k, v in env.items() if k not in _provider_key_env}


def curated_env(
    *,
    passthrough: tuple[str, ...] = (),
    extra: dict[str, str] | None = None,
    desktop: bool = True,
) -> dict[str, str]:
    """The base environment, plus *passthrough* names and *extra* values.

    `passthrough` is how an operator hands one child a variable it genuinely
    needs (an MCP server's API token). Named one at a time in config, because
    naming each one is the point: a provider key is never among them, since
    nobody would write it down.

    `desktop=False` also drops the session-bus and display addresses. Pass it
    for a child that is meant to be CONFINED: those addresses reach processes
    that are not, and delegating to one walks straight out of any sandbox.
    """
    keep = (*_KEEP, *_DESKTOP) if desktop else _KEEP
    env = {k: v for k in keep if (v := os.environ.get(k)) is not None}
    env.update({k: v for k in passthrough if (v := os.environ.get(k)) is not None})
    env.update(extra or {})
    return env
