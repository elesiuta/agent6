# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The seam between the CLI and the operator's terminal.

Model-influenced text reaches stdout and stderr from many places (a resume
note naming a file a jailed command created, a commit subject, a session's
summary, a task a plan wrote), and a terminal obeys a control sequence
wherever it appears in a line: OSC 52 writes the clipboard, a title change
or a forged line follows. `guarded_terminal` wraps the process's two streams
for the CLI's lifetime and `scrub_terminal_output` decides, once, what
passes; the `/dev/tty` writers in `_steer` scrub each write the same way.
Three writers sit under the wrapper: the ACP and MCP stdio protocols, whose
bytes go to a peer through `sys.stdout.buffer`; the spinners, whose erase
idiom goes to `raw_stream`; and the interactive composer, whose cursor
movement goes there too, with its rows scrubbed.
"""

from __future__ import annotations

import contextlib
import io
import sys
from collections.abc import Generator, Iterable
from typing import IO, Any

from agent6.viewmodel.transcript import scrub_terminal_output


class ScrubbedStream:
    """A text stream whose every write passes `scrub_terminal_output` and
    answers what the wrapped stream wrote; all else (flush, isatty, fileno,
    encoding, buffer) is the wrapped stream's."""

    def __init__(self, raw: IO[str]) -> None:
        self.raw = raw

    def write(self, text: str) -> int:
        return self.raw.write(scrub_terminal_output(text))

    def writelines(self, lines: Iterable[str]) -> None:
        for line in lines:
            self.write(line)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.raw, name)


def raw_stream(stream: IO[str] | ScrubbedStream) -> IO[str]:
    """The stream under *stream*'s scrubber, for a writer that erases a line
    or moves the cursor and scrubs its own text."""
    return stream.raw if isinstance(stream, ScrubbedStream) else stream


@contextlib.contextmanager
def guarded_terminal() -> Generator[None]:
    """stdout and stderr scrubbed for the block, the originals restored after.
    A redirected stdout is block-buffered, so a run's log file would stay empty
    until exit: it is line-buffered here, for the same lifetime."""
    out, err = sys.stdout, sys.stderr
    if isinstance(out, io.TextIOWrapper):
        out.reconfigure(line_buffering=True)
    sys.stdout, sys.stderr = ScrubbedStream(out), ScrubbedStream(err)  # type: ignore[assignment]
    try:
        yield
    finally:
        sys.stdout, sys.stderr = out, err
