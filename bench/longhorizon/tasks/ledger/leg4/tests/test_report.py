import tempfile
import unittest
from pathlib import Path

from tests.test_cli import cli


class ReportTest(unittest.TestCase):
    def test_report_month(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            book = Path(d) / "book.jsonl"
            cli(book, "add", "2026-03-02", "assets:cash", "expenses:food", "10.00")
            cli(book, "add", "2026-03-09", "assets:cash", "expenses:food", "20.00")
            cli(book, "add", "2026-04-01", "assets:cash", "expenses:food", "99.00")
            out = cli(book, "report", "2026-03")
            self.assertEqual(out, "assets:cash -30.00 -15.00\nexpenses:food 30.00 15.00\n")
