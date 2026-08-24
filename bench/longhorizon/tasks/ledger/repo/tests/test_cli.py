import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def cli(book: Path, *args: str) -> str:
    proc = subprocess.run(
        [sys.executable, "-m", "ledger.cli", *args],
        cwd=ROOT,
        env={"LEDGER_BOOK": str(book), "PATH": "/usr/bin:/bin"},
        capture_output=True,
        text=True,
        check=True,
    )
    return proc.stdout


class CliTest(unittest.TestCase):
    def test_add_and_balance(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            book = Path(d) / "book.jsonl"
            out = cli(book, "add", "2026-01-05", "assets:cash", "expenses:food", "12.50")
            self.assertEqual(out, "posted 12.50 assets:cash -> expenses:food\n")
            self.assertEqual(cli(book, "balance", "assets:cash"), "-12.50\n")

    def test_add_keeps_memo(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            book = Path(d) / "book.jsonl"
            cli(book, "add", "2026-01-05", "assets:cash", "expenses:food", "12.50", "lunch", "out")
            self.assertIn("lunch out", cli(book, "show"))
