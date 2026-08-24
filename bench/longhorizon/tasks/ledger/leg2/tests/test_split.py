import tempfile
import unittest
from pathlib import Path

from tests.test_cli import cli


class SplitTest(unittest.TestCase):
    def test_even_split(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            book = Path(d) / "book.jsonl"
            out = cli(book, "split", "2026-02-01", "assets:cash", "10.00", "a:50", "b:50", "rent")
            self.assertEqual(out, "a 5.00\nb 5.00\n")
            self.assertEqual(cli(book, "balance", "assets:cash"), "-10.00\n")
            self.assertEqual(cli(book, "balance", "b"), "5.00\n")
