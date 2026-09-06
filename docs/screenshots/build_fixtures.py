#!/usr/bin/env python3
"""Build small, sanitized seed fixtures from real agent6 runs (dev tool).

Run on a machine that has real sessions under $XDG_STATE_HOME/agent6/. It copies a
curated set of runs into docs/screenshots/seed/runs/<id>/, trimming the
token-delta bloat out of logs.jsonl (a 9 MB log is ~99% role.*_delta events)
and keeping only the structural events plus a tail of reasoning, so the TUI hub
and dashboard render the same but each fixture is tens of KB. Secrets are
already redacted by agent6; this also neutralizes absolute home/temp paths.

Not run in CI. The committed output is what CI replays (see seed.py).
"""

from __future__ import annotations

import json
import os
import re
import shutil
from pathlib import Path

# Curated source runs by run id (ids are unique; we glob for the dir). All
# generic RFC / library reimplementation tasks, safe to publish. We
# deliberately exclude the perf take-home runs.
SOURCES = [
    "willing-glen-9ZYWWB",  # url RFC 3986: rich, the dashboard + transcript star
    "friendly-crane-1X3ER0",  # csv RFC 4180: many tool calls
    "tidy-river-165YS6",  # html -> text
    "thoughtful-comet-1TQASW",  # restore click.format_filename
    "ready-rowan-A5P972",  # csv RFC 4180: small + clean
]

# The committed willing-glen fixture is hand-trimmed to event 132, a clean mid-run
# point right after a passing verify and the rich parse_url diff. The full run went
# on to event 661: around event 383 its recording sandbox lost the interpreter
# (run_command returns exit 127 "command not found", an environment artifact of the
# capture, not an agent failure) and it floundered to a failed end. Trimming drops
# that misleading tail so the featured run shows the dashboard mid-task. Re-apply
# the trim after rebuilding from source.

# Only one run needs transcripts on disk: the dashboard/transcript star. Hub and
# dashboard render from logs.jsonl alone; the conversation viewer reads
# transcripts/, so we keep just the opening calls of the featured run.
FEATURE_RUN = "willing-glen-9ZYWWB"
KEEP_TRANSCRIPTS_HEAD_FEATURE = 4

STATE = Path(os.environ.get("XDG_STATE_HOME", Path.home() / ".local/state")) / "agent6"
OUT = Path(__file__).resolve().parent / "seed" / "runs"

DELTA_EVENTS = {"role.thinking_delta", "role.text_delta"}
KEEP_DELTA_TAIL = 80  # keep the last N deltas so the stream pane shows reasoning

# Path scrubbing: anything machine-specific -> a neutral demo path.
SCRUBS = [
    (re.compile(re.escape(str(Path.home()))), "/home/agent6"),
    (re.compile(r"/tmp/[A-Za-z0-9._-]+/agent6[-_][a-z0-9]+"), "/home/agent6/demo"),  # noqa: S108
    (re.compile(r"/tmp/tmp[A-Za-z0-9]+"), "/home/agent6/demo"),  # noqa: S108
]


def scrub(text: str) -> str:
    for pat, repl in SCRUBS:
        text = pat.sub(repl, text)
    return text


def find_run(session_id: str) -> Path | None:
    hits = sorted(STATE.glob(f"*/sessions/*/{session_id}"))
    return hits[0] if hits else None


def trim_logs(src: Path, dst: Path) -> int:
    lines = src.read_text(encoding="utf-8", errors="replace").splitlines()
    structural: list[str] = []
    deltas: list[str] = []
    for line in lines:
        try:
            ev = json.loads(line).get("event") or json.loads(line).get("type", "")
        except json.JSONDecodeError:
            continue
        (deltas if ev in DELTA_EVENTS else structural).append(line)
    kept = structural + deltas[-KEEP_DELTA_TAIL:]

    # Re-sort by timestamp so the kept delta tail lands in chronological order.
    def ts(line: str) -> str:
        try:
            return json.loads(line).get("ts", "")
        except json.JSONDecodeError:
            return ""

    kept.sort(key=ts)
    dst.write_text(scrub("\n".join(kept)) + "\n", encoding="utf-8")
    return len(kept)


def copy_scrubbed(src: Path, dst: Path) -> None:
    dst.write_text(scrub(src.read_text(encoding="utf-8", errors="replace")), encoding="utf-8")


def main() -> None:
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    total = 0
    for session_id in SOURCES:
        src = find_run(session_id)
        if not src:
            print(f"  SKIP {session_id}: not found under {STATE}")
            continue
        dst = OUT / session_id
        (dst / "transcripts").mkdir(parents=True)
        for name in ("manifest.json", "graph.jsonl"):
            if (src / name).is_file():
                copy_scrubbed(src / name, dst / name)
        n = (
            trim_logs(src / "logs.jsonl", dst / "logs.jsonl")
            if (src / "logs.jsonl").is_file()
            else 0
        )
        tr = sorted((src / "transcripts").glob("*.json")) if (src / "transcripts").is_dir() else []
        keep = tr[:KEEP_TRANSCRIPTS_HEAD_FEATURE] if session_id == FEATURE_RUN else []
        seen: set[str] = set()
        for t in keep:
            if t.name in seen:
                continue
            seen.add(t.name)
            copy_scrubbed(t, dst / "transcripts" / t.name)
        size = sum(p.stat().st_size for p in dst.rglob("*") if p.is_file())
        total += size
        print(f"  {session_id}: {n} log lines, {len(seen)} transcripts, {size // 1024} kB")
    print(f"total fixtures: {total // 1024} kB -> {OUT}")


if __name__ == "__main__":
    main()
