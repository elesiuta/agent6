import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from tests.test_cli import ROOT, cli


class ImportTest(unittest.TestCase):
    def test_import_ok(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            book = Path(d) / "book.jsonl"
            out = cli(book, "import-csv", str(ROOT / "samples" / "ok.csv"))
            self.assertEqual(out, "imported 2 transactions\n")
            self.assertEqual(cli(book, "balance", "assets:cash"), "-112.50\n")

    def test_import_unbalanced_imports_nothing(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            book = Path(d) / "book.jsonl"
            proc = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "ledger.cli",
                    "import-csv",
                    str(ROOT / "samples" / "bad.csv"),
                ],
                cwd=ROOT,
                env={"LEDGER_BOOK": str(book), "PATH": "/usr/bin:/bin"},
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertNotEqual(proc.returncode, 0)
            self.assertIn("t2", proc.stdout + proc.stderr)
            self.assertFalse(book.exists())
