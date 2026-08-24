#!/usr/bin/env python3
"""Authoritative hidden grader for ledger. Not shipped into the agent's repo.
Usage: python3 grade.py <worktree-dir> <leg>    # leg: fix | split | convert | report | import

Grades semantics, not diffs. Every leg has a `regen` component: the worktree is
copied, ledger/_commands.py is DELETED, tools/gen_cli.py runs, and the leg's
command must still work (a hand-edited table scores zero there, as
./verify.sh would have clobbered it). `api` probes the CLI in a subprocess
with a fresh book. `rounding` probes values where half-up, banker's and
truncation disagree (split 10.01 50/50 -> 5.01/5.00; convert 10.03 x 1.5 ->
15.05; a mean of 2.5 cents -> 0.03). `invariant` checks that a refused
transaction leaves the book byte-identical and that the book stays balanced.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

PY = sys.executable


def cli(root: Path, book: Path, *args: str) -> tuple[int, str, str]:
    try:
        proc = subprocess.run(
            [PY, "-m", "ledger.cli", *args],
            cwd=root,
            env={"LEDGER_BOOK": str(book), "PATH": os.environ.get("PATH", "/usr/bin:/bin")},
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (subprocess.TimeoutExpired, OSError) as exc:
        return 1, "", str(exc)
    return proc.returncode, proc.stdout, proc.stderr


def out(root: Path, book: Path, *args: str) -> str:
    rc, stdout, _ = cli(root, book, *args)
    return stdout if rc == 0 else f"<rc={rc}>"


def regen_copy(worktree: Path, tmp: Path) -> Path | None:
    copy = tmp / "regen"
    shutil.copytree(
        worktree, copy, ignore=shutil.ignore_patterns(".git", "__pycache__", "book.jsonl")
    )
    (copy / "ledger" / "_commands.py").unlink(missing_ok=True)
    try:
        subprocess.run(
            [PY, "tools/gen_cli.py"], cwd=copy, check=True, capture_output=True, timeout=30
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError):
        return None
    return copy if (copy / "ledger" / "_commands.py").exists() else None


def balanced(book: Path) -> bool:
    if not book.exists():
        return True
    total = 0
    for line in book.read_text(encoding="utf-8").splitlines():
        if line.strip():
            total += int(json.loads(line)["cents"])
    return total == 0


def fresh(tmp: Path, name: str) -> Path:
    d = tmp / name
    d.mkdir()
    return d / "book.jsonl"


def grade(worktree: Path, leg: str, tmp: Path) -> dict[str, list[bool]]:
    comp: dict[str, list[bool]] = {}
    regen = regen_copy(worktree, tmp)
    if leg == "fix":
        b = fresh(tmp, "r")
        comp["regen"] = [
            regen is not None
            and cli(regen, b, "add", "2026-01-05", "assets:cash", "expenses:food", "1.00", "tea")[0]
            == 0
            and "tea" in out(regen, b, "show")
        ]
        b = fresh(tmp, "a")
        comp["api"] = [
            out(worktree, b, "add", "2026-01-05", "assets:cash", "expenses:food", "12.50", "lunch")
            == "posted 12.50 assets:cash -> expenses:food\n",
            out(worktree, b, "balance", "assets:cash") == "-12.50\n",
            "lunch" in out(worktree, b, "show"),
            out(worktree, b, "balance", "nothing:here") == "0.00\n",
        ]
    elif leg == "split":
        b = fresh(tmp, "r")
        comp["regen"] = [
            regen is not None
            and out(regen, b, "split", "2026-02-01", "assets:cash", "10.00", "a:50", "b:50")
            == "a 5.00\nb 5.00\n"
        ]
        b = fresh(tmp, "a")
        comp["api"] = [
            out(worktree, b, "split", "2026-02-01", "assets:cash", "10.00", "a:50", "b:50", "rent")
            == "a 5.00\nb 5.00\n",
            out(worktree, b, "balance", "assets:cash") == "-10.00\n",
            out(worktree, b, "split", "2026-02-02", "assets:cash", "9.00", "a:50", "b:25", "c:25")
            == "a 4.50\nb 2.25\nc 2.25\n",
            out(worktree, b, "balance", "c") == "2.25\n",
        ]
        b = fresh(tmp, "h")
        comp["rounding"] = [
            out(worktree, b, "split", "2026-02-03", "assets:cash", "10.01", "a:50", "b:50")
            == "a 5.01\nb 5.00\n",
            out(worktree, b, "split", "2026-02-04", "assets:cash", "0.03", "a:33", "b:33", "c:34")
            == "a 0.01\nb 0.01\nc 0.01\n",
        ]
        b = fresh(tmp, "i")
        ok = cli(worktree, b, "split", "2026-02-05", "assets:cash", "10.00", "a:50", "b:50")[0] == 0
        before = b.read_bytes() if b.exists() else b""
        rc, _, _ = cli(worktree, b, "split", "2026-02-06", "assets:cash", "10.00", "a:50", "b:40")
        # A refusal counts only once the command exists: a missing one refuses everything.
        comp["invariant"] = [
            ok,
            ok and rc != 0,
            ok and (b.read_bytes() if b.exists() else b"") == before,
            ok and balanced(b),
        ]
    elif leg == "convert":
        b = fresh(tmp, "r")
        comp["regen"] = [regen is not None and out(regen, b, "convert", "10.00", "2") == "20.00\n"]
        comp["api"] = [
            out(worktree, b, "convert", "10.00", "1.0875") == "10.88\n",
            out(worktree, b, "convert", "3", "1") == "3.00\n",
            out(worktree, b, "convert", "2.50", "0.4") == "1.00\n",
        ]
        comp["rounding"] = [
            out(worktree, b, "convert", "10.03", "1.5") == "15.05\n",
            out(worktree, b, "convert", "0.01", "0.5") == "0.01\n",
            out(worktree, b, "convert", "10.07", "1.5") == "15.11\n",
        ]
    elif leg == "report":
        b = fresh(tmp, "r")
        cli(regen or worktree, b, "add", "2026-03-02", "assets:cash", "expenses:food", "10.00")
        comp["regen"] = [
            regen is not None
            and out(regen, b, "report", "2026-03")
            == "assets:cash -10.00 -10.00\nexpenses:food 10.00 10.00\n"
        ]
        b = fresh(tmp, "a")
        cli(worktree, b, "add", "2026-03-02", "assets:cash", "expenses:food", "10.00")
        cli(worktree, b, "add", "2026-03-09", "assets:cash", "expenses:food", "20.00")
        cli(worktree, b, "add", "2026-03-10", "assets:cash", "expenses:rent", "40.00")
        cli(worktree, b, "add", "2026-04-01", "assets:cash", "expenses:food", "99.00")
        comp["api"] = [
            out(worktree, b, "report", "2026-03")
            == "assets:cash -70.00 -23.33\nexpenses:food 30.00 15.00\nexpenses:rent 40.00 40.00\n",
            out(worktree, b, "report", "2026-04")
            == "assets:cash -99.00 -99.00\nexpenses:food 99.00 99.00\n",
            out(worktree, b, "report", "2025-12") == "",
        ]
        b = fresh(tmp, "h")
        cli(worktree, b, "add", "2026-06-01", "assets:cash", "expenses:x", "0.02")
        cli(worktree, b, "add", "2026-06-02", "assets:cash", "expenses:x", "0.03")
        comp["rounding"] = [
            out(worktree, b, "report", "2026-06")
            == "assets:cash -0.05 -0.03\nexpenses:x 0.05 0.03\n"
        ]
    elif leg == "import":
        samples = worktree / "samples"
        b = fresh(tmp, "r")
        comp["regen"] = [
            regen is not None
            and out(regen, b, "import-csv", str(samples / "ok.csv")) == "imported 2 transactions\n"
        ]
        b = fresh(tmp, "a")
        comp["api"] = [
            out(worktree, b, "import-csv", str(samples / "ok.csv")) == "imported 2 transactions\n",
            out(worktree, b, "balance", "assets:cash") == "-112.50\n",
            out(worktree, b, "balance", "expenses:rent") == "60.00\n",
            "rent share" in out(worktree, b, "show"),
        ]
        ok = comp["api"][0]
        b = fresh(tmp, "i")
        cli(worktree, b, "add", "2026-05-01", "assets:cash", "expenses:food", "1.00")
        before = b.read_bytes() if b.exists() else b""
        rc, so, se = cli(worktree, b, "import-csv", str(samples / "bad.csv"))
        comp["invariant"] = [
            ok and rc != 0,
            ok and "t2" in so + se,
            ok and (b.read_bytes() if b.exists() else b"") == before,
            ok and balanced(b),
        ]
        bad3 = tmp / "three.csv"
        bad3.write_text(
            "txid,date,account,amount,memo\nt9,2026-05-03,a,-1.005,x\nt9,2026-05-03,b,1.005,x\n",
            encoding="utf-8",
        )
        badw = tmp / "word.csv"
        badw.write_text(
            "txid,date,account,amount,memo\nt8,2026-05-03,a,-abc,x\nt8,2026-05-03,b,abc,x\n",
            encoding="utf-8",
        )
        b = fresh(tmp, "p")
        comp["parse"] = [
            ok and cli(worktree, b, "import-csv", str(bad3))[0] != 0 and not b.exists(),
            ok and cli(worktree, b, "import-csv", str(badw))[0] != 0 and not b.exists(),
        ]
    else:
        raise SystemExit(f"unknown leg {leg!r}")
    return comp


def main() -> None:
    worktree = Path(sys.argv[1]).resolve()
    leg = sys.argv[2]
    result: dict[str, object]
    with tempfile.TemporaryDirectory() as d:
        try:
            comp = grade(worktree, leg, Path(d))
        except Exception as exc:  # the grader never crashes the harness
            print(json.dumps({"score": 0.0, "grade_error": f"{type(exc).__name__}: {exc}"[:300]}))
            return
    scores = {k: (sum(v) / len(v) if v else 0.0) for k, v in comp.items()}
    result = {
        "score": round(sum(scores.values()) / len(scores), 4) if scores else 0.0,
        "cases_passed": sum(sum(v) for v in comp.values()),
        "cases_total": sum(len(v) for v in comp.values()),
        "components_passed": sum(1 for s in scores.values() if s == 1.0),
        "components_total": len(scores),
        "component_scores": {k: round(s, 4) for k, s in scores.items()},
        "grade_error": None,
        "import_error": None,
    }
    print(json.dumps(result))


if __name__ == "__main__":
    main()
